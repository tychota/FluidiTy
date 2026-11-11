#include "fluidity/host/adaptive_family_change_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace fluidity {
namespace {

auto CellOrder(CellIndex3D index) {
  return std::tuple{index.z, index.y, index.x};
}

std::optional<std::size_t> FindLeaf(std::span<const AdaptiveMacLeaf3D> cells, AdaptiveMacCellKey3D key) {
  const auto order = CellOrder(key.lower_corner);
  const auto cell = std::ranges::lower_bound(cells, order, {}, [](const AdaptiveMacLeaf3D& leaf) {
    return CellOrder(leaf.key.lower_corner);
  });
  if (cell == cells.end() || CellOrder(cell->key.lower_corner) != order
      || cell->key.width_exponent != key.width_exponent) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(cell - cells.begin());
}

absl::StatusOr<std::array<AdaptiveMacCellKey3D, 8>> ChildKeys(AdaptiveMacCellKey3D parent) {
  if (parent.width_exponent == 0 || parent.width_exponent > 30) {
    return absl::InvalidArgumentError("adaptive family requires parent exponent in [1,30]");
  }
  const std::int64_t parent_width = std::int64_t{1} << parent.width_exponent;
  const std::array lower{parent.lower_corner.x, parent.lower_corner.y, parent.lower_corner.z};
  for (const std::int32_t coordinate : lower) {
    if (coordinate % parent_width != 0) {
      return absl::InvalidArgumentError("adaptive family parent is not dyadically aligned");
    }
    if (coordinate + parent_width > std::numeric_limits<std::int32_t>::max()) {
      return absl::OutOfRangeError("adaptive family parent exceeds the signed lattice");
    }
  }
  const auto child_exponent = static_cast<std::uint8_t>(parent.width_exponent - 1);
  const auto child_width = static_cast<std::int32_t>(parent_width / 2);
  std::array<AdaptiveMacCellKey3D, 8> children{};
  std::size_t child = 0;
  for (const std::int32_t z : {0, child_width}) {
    for (const std::int32_t y : {0, child_width}) {
      for (const std::int32_t x : {0, child_width}) {
        children[child++] = {
            .lower_corner = {lower[0] + x, lower[1] + y, lower[2] + z},
            .width_exponent = child_exponent,
        };
      }
    }
  }
  return children;
}

// Pairwise midpoints keep the equal-volume mean within the pressure range
// without forming a sum that could overflow.
double AverageSiblingPressure(std::span<const double> pressures, const std::array<std::size_t, 8>& siblings) {
  std::array<double, 8> means{};
  for (std::size_t child = 0; child < siblings.size(); ++child) {
    means[child] = pressures[siblings[child]];
  }
  for (std::size_t pair_count = means.size() / 2; pair_count > 0; pair_count /= 2) {
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      means[pair] = std::midpoint(means[2 * pair], means[2 * pair + 1]);
    }
  }
  return means.front();
}

absl::StatusOr<AdaptiveFamilyChange3D> PrepareFamilyChange(
    const AdaptiveMacTopology3D& committed_topology,
    std::span<const double> committed_pressures,
    AdaptiveMacCellKey3D parent,
    bool refining
) {
  const auto committed_cells = committed_topology.cells();
  if (committed_cells.empty() || committed_pressures.size() != committed_cells.size()) {
    return absl::InvalidArgumentError("adaptive family requires one pressure per committed leaf");
  }
  if (!std::ranges::all_of(committed_pressures, [](double pressure) {
        return std::isfinite(pressure);
      })) {
    return absl::InvalidArgumentError("adaptive family warm starts must be finite");
  }
  const auto child_keys = ChildKeys(parent);
  if (!child_keys.ok()) {
    return child_keys.status();
  }
  std::size_t parent_ordinal = 0;
  std::array<std::size_t, 8> child_ordinals{};
  double family_pressure = 0.0;
  if (refining) {
    const auto parent_leaf = FindLeaf(committed_cells, parent);
    if (!parent_leaf) {
      return absl::FailedPreconditionError("adaptive refinement requires the committed parent leaf");
    }
    parent_ordinal = *parent_leaf;
    family_pressure = committed_pressures[parent_ordinal];
  } else {
    for (std::size_t child = 0; child < child_keys->size(); ++child) {
      const auto sibling = FindLeaf(committed_cells, (*child_keys)[child]);
      if (!sibling) {
        return absl::FailedPreconditionError("adaptive coarsening requires all eight committed siblings");
      }
      child_ordinals[child] = *sibling;
    }
    family_pressure = AverageSiblingPressure(committed_pressures, child_ordinals);
  }

  std::vector<AdaptiveMacCellKey3D> candidate_keys;
  const std::size_t candidate_count = refining ? committed_cells.size() + 7 : committed_cells.size() - 7;
  candidate_keys.reserve(candidate_count);
  for (std::size_t cell = 0; cell < committed_cells.size(); ++cell) {
    const bool replaced = refining ? cell == parent_ordinal : std::ranges::binary_search(child_ordinals, cell);
    if (!replaced) {
      candidate_keys.push_back(committed_cells[cell].key);
    }
  }
  if (refining) {
    candidate_keys.insert(candidate_keys.end(), child_keys->begin(), child_keys->end());
  } else {
    candidate_keys.push_back(parent);
  }
  auto candidate = AdaptiveMacTopology3D::Create(candidate_keys, committed_topology.finest_cell_width());
  if (!candidate.ok()) {
    return candidate.status();
  }

  const auto candidate_cells = candidate->cells();
  if (refining) {
    for (std::size_t child = 0; child < child_keys->size(); ++child) {
      const auto sibling = FindLeaf(candidate_cells, (*child_keys)[child]);
      if (!sibling) {
        return absl::InternalError("adaptive refinement candidate lost a child");
      }
      child_ordinals[child] = *sibling;
    }
  } else {
    const auto parent_leaf = FindLeaf(candidate_cells, parent);
    if (!parent_leaf) {
      return absl::InternalError("adaptive coarsening candidate lost its parent");
    }
    parent_ordinal = *parent_leaf;
  }
  std::vector<double> previous_pressures(candidate_count, family_pressure);
  std::vector<AdaptiveRetainedLeaf3D> retained_leaves;
  retained_leaves.reserve(refining ? committed_cells.size() - 1 : committed_cells.size() - 8);
  for (std::size_t cell = 0; cell < candidate_cells.size(); ++cell) {
    const auto old_cell = FindLeaf(committed_cells, candidate_cells[cell].key);
    if (old_cell) {
      previous_pressures[cell] = committed_pressures[*old_cell];
      retained_leaves.push_back({.old_cell_ordinal = *old_cell, .new_cell_ordinal = cell});
    }
  }
  return AdaptiveFamilyChange3D{
      .topology = std::move(*candidate),
      .previous_pressures = std::move(previous_pressures),
      .retained_leaves = std::move(retained_leaves),
      .parent_ordinal = parent_ordinal,
      .child_ordinals = child_ordinals,
  };
}

}  // namespace

absl::StatusOr<AdaptiveFamilyChange3D> PrepareAdaptiveRefinement3D(
    const AdaptiveMacTopology3D& committed_topology,
    std::span<const double> committed_pressures,
    AdaptiveMacCellKey3D parent
) {
  return PrepareFamilyChange(committed_topology, committed_pressures, parent, true);
}

absl::StatusOr<AdaptiveFamilyChange3D> PrepareAdaptiveCoarsening3D(
    const AdaptiveMacTopology3D& committed_topology,
    std::span<const double> committed_pressures,
    AdaptiveMacCellKey3D parent
) {
  return PrepareFamilyChange(committed_topology, committed_pressures, parent, false);
}

}  // namespace fluidity
