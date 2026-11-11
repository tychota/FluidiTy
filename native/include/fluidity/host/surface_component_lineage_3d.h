#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"

namespace fluidity {
struct SurfaceComponentMember3D {
  std::uint64_t particle_id = 0;
  double reference_volume_m3 = 0;  // Invariant V0, not compressed physical volume V0*J.
};
struct SurfaceComponentHistory3D {
  std::uint64_t component_id = 0;
  std::span<const SurfaceComponentMember3D> members{};
};
struct SurfaceComponentMaterial3D {
  std::span<const SurfaceComponentMember3D> members{};
  double target_volume_m3 = 0;  // Physical material inventory from accepted Simulate state.
};
struct SurfaceComponentParent3D {
  std::uint64_t parent_component_id = 0;
  double shared_reference_volume_m3 = 0;
};
struct SurfaceComponentAssignment3D {
  std::uint64_t component_id = 0;
  double target_volume_m3 = 0;
  std::vector<SurfaceComponentParent3D> parents{};  // Ascending parent ID, including nonretained parents.
};
struct SurfaceComponentLineage3D {
  std::uint64_t next_component_id = 1;
  std::vector<SurfaceComponentAssignment3D> components{};  // Same order as current components.
  std::vector<std::uint64_t> retired_component_ids{};  // Prior IDs not retained, including merged parents.
};

// Matches already segmented material components using overlap in immutable
// particle IDs weighted by their reference volumes. Previous groups are ordered
// by component ID, current groups by their first member ID. Each group's members
// are nonempty and strictly ordered by particle ID; partitions cannot repeat IDs.
// Retain IDs greedily by descending overlap, then parent ID and current first
// member ID. A split or merge keeps all ancestry edges, but only one child can
// retain each parent ID. Unmatched groups receive increasing fresh IDs. Current
// physical volume targets are copied exactly; overlap does not rescale them.
// This is a uniform-particle lineage policy; it does not segment geometry,
// infer vertex correspondence or account for adaptivity/resampling.
// Invalid controls/partitions return InvalidArgument, changed V0 on an existing
// member FailedPrecondition, overflow OutOfRange, exceeded member/ID budget
// ResourceExhausted. The bound (hard ceiling 2^20) applies to each partition;
// memory is linear in members and sorting costs O(n log n). Inputs are borrowed
// and never modified. No partial result on failure.
absl::StatusOr<SurfaceComponentLineage3D> MatchSurfaceComponentLineage3D(
    std::span<const SurfaceComponentHistory3D> previous,
    std::span<const SurfaceComponentMaterial3D> current,
    std::uint64_t next_component_id,
    std::size_t maximum_members = 1024
);
}  // namespace fluidity
