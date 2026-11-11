#include "fluidity/host/surface_component_lineage_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace fluidity {
namespace {
struct IndexedMember {
  SurfaceComponentMember3D member;
  std::size_t component_index;
};

// Sort by particle ID once so the cross-partition join needs no spatial search.
template <typename Component>
absl::StatusOr<std::vector<IndexedMember>>
CollectMembers(std::span<const Component> components, std::size_t maximum_members) {
  if (components.size() > maximum_members) {
    return absl::ResourceExhaustedError("surface component count exceeds member budget");
  }
  std::size_t member_count = 0;
  for (const auto& component : components) {
    if (component.members.empty()) {
      return absl::InvalidArgumentError("surface material component has no members");
    }
    if (component.members.size() > maximum_members - member_count) {
      return absl::ResourceExhaustedError("surface component member budget exceeded");
    }
    member_count += component.members.size();
  }
  std::vector<IndexedMember> members;
  members.reserve(member_count);
  for (std::size_t component_index = 0; component_index < components.size(); ++component_index) {
    std::uint64_t prior_particle_id = 0;
    for (const auto& member : components[component_index].members) {
      if (member.particle_id <= prior_particle_id || !std::isfinite(member.reference_volume_m3)
          || member.reference_volume_m3 <= 0) {
        return absl::InvalidArgumentError("invalid or unordered surface component member");
      }
      members.push_back({.member = member, .component_index = component_index});
      prior_particle_id = member.particle_id;
    }
  }
  const auto particle_id = [](const IndexedMember& entry) {
    return entry.member.particle_id;
  };
  std::ranges::sort(members, {}, particle_id);
  if (std::ranges::adjacent_find(members, {}, particle_id) != members.end()) {
    return absl::InvalidArgumentError("particle belongs to multiple surface material components");
  }
  return members;
}

struct OverlapContribution {
  std::size_t parent_index;
  std::size_t child_index;
  std::uint64_t particle_id;
  double reference_volume_m3;
};
struct ComponentOverlap {
  std::size_t parent_index;
  std::size_t child_index;
  double shared_reference_volume_m3;
};

// Accumulate each pair in ascending particle ID order, independent of group
// traversal or filesystem ordering. Shared V0 gives a compression-invariant key.
absl::StatusOr<std::vector<ComponentOverlap>>
ComputeOverlaps(std::span<const IndexedMember> previous, std::span<const IndexedMember> current) {
  std::vector<OverlapContribution> contributions;
  contributions.reserve(std::min(previous.size(), current.size()));
  std::size_t previous_member_index = 0;
  std::size_t current_member_index = 0;
  while (previous_member_index < previous.size() && current_member_index < current.size()) {
    const auto& prior = previous[previous_member_index];
    const auto& next = current[current_member_index];
    if (prior.member.particle_id < next.member.particle_id) {
      ++previous_member_index;
    } else if (next.member.particle_id < prior.member.particle_id) {
      ++current_member_index;
    } else {
      if (prior.member.reference_volume_m3 != next.member.reference_volume_m3) {
        return absl::FailedPreconditionError("retained particle reference volume changed");
      }
      contributions.push_back({
          .parent_index = prior.component_index,
          .child_index = next.component_index,
          .particle_id = next.member.particle_id,
          .reference_volume_m3 = next.member.reference_volume_m3,
      });
      ++previous_member_index;
      ++current_member_index;
    }
  }
  std::ranges::sort(contributions, {}, [](const OverlapContribution& contribution) {
    return std::tuple(contribution.parent_index, contribution.child_index, contribution.particle_id);
  });
  std::vector<ComponentOverlap> overlaps;
  for (const auto& contribution : contributions) {
    const bool same_pair = !overlaps.empty() && overlaps.back().parent_index == contribution.parent_index
        && overlaps.back().child_index == contribution.child_index;
    if (same_pair) {
      auto& shared_reference_volume_m3 = overlaps.back().shared_reference_volume_m3;
      shared_reference_volume_m3 += contribution.reference_volume_m3;
      if (!std::isfinite(shared_reference_volume_m3)) {
        return absl::OutOfRangeError("surface component reference overlap overflowed");
      }
    } else {
      overlaps.push_back({
          .parent_index = contribution.parent_index,
          .child_index = contribution.child_index,
          .shared_reference_volume_m3 = contribution.reference_volume_m3,
      });
    }
  }
  return overlaps;
}
}  // namespace

absl::StatusOr<SurfaceComponentLineage3D> MatchSurfaceComponentLineage3D(
    std::span<const SurfaceComponentHistory3D> previous,
    std::span<const SurfaceComponentMaterial3D> current,
    std::uint64_t next_component_id,
    std::size_t maximum_members
) {
  if (next_component_id == 0) {
    return absl::InvalidArgumentError("surface component identity counter is zero");
  }
  maximum_members = std::min(maximum_members, std::size_t{1} << 20);
  auto previous_members = CollectMembers(previous, maximum_members);
  if (!previous_members.ok()) {
    return previous_members.status();
  }
  auto current_members = CollectMembers(current, maximum_members);
  if (!current_members.ok()) {
    return current_members.status();
  }
  std::uint64_t prior_component_id = 0;
  for (const auto& component : previous) {
    if (component.component_id <= prior_component_id || component.component_id >= next_component_id) {
      return absl::InvalidArgumentError("invalid or unordered prior surface component identity");
    }
    prior_component_id = component.component_id;
  }
  std::uint64_t prior_first_particle_id = 0;
  for (const auto& component : current) {
    const auto first_particle_id = component.members.front().particle_id;
    if (first_particle_id <= prior_first_particle_id || !std::isfinite(component.target_volume_m3)
        || component.target_volume_m3 <= 0) {
      return absl::InvalidArgumentError("invalid or unordered current surface material component");
    }
    prior_first_particle_id = first_particle_id;
  }
  auto overlaps = ComputeOverlaps(*previous_members, *current_members);
  if (!overlaps.ok()) {
    return overlaps.status();
  }
  SurfaceComponentLineage3D lineage{.next_component_id = next_component_id};
  lineage.components.resize(current.size());
  for (const auto& overlap : *overlaps) {
    lineage.components[overlap.child_index].parents.push_back({
        .parent_component_id = previous[overlap.parent_index].component_id,
        .shared_reference_volume_m3 = overlap.shared_reference_volume_m3,
    });
  }
  std::ranges::sort(*overlaps, {}, [](const ComponentOverlap& overlap) {
    return std::tuple(-overlap.shared_reference_volume_m3, overlap.parent_index, overlap.child_index);
  });
  std::vector<bool> retained_parent(previous.size());
  for (const auto& overlap : *overlaps) {
    auto& child = lineage.components[overlap.child_index];
    if (!retained_parent[overlap.parent_index] && child.component_id == 0) {
      child.component_id = previous[overlap.parent_index].component_id;
      retained_parent[overlap.parent_index] = true;
    }
  }
  for (std::size_t child_index = 0; child_index < current.size(); ++child_index) {
    auto& child = lineage.components[child_index];
    child.target_volume_m3 = current[child_index].target_volume_m3;
    if (child.component_id == 0) {
      if (lineage.next_component_id == std::numeric_limits<std::uint64_t>::max()) {
        return absl::ResourceExhaustedError("surface component identities exhausted");
      }
      child.component_id = lineage.next_component_id++;
    }
  }
  for (std::size_t parent_index = 0; parent_index < previous.size(); ++parent_index) {
    if (!retained_parent[parent_index]) {
      lineage.retired_component_ids.push_back(previous[parent_index].component_id);
    }
  }
  return lineage;
}
}  // namespace fluidity
