#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"

namespace fluidity {

struct AdaptiveRetainedLeaf3D {
  std::size_t old_cell_ordinal;
  std::size_t new_cell_ordinal;
};

// Owns an unpublished candidate. Parent ordinal is in the coarse topology
// and child ordinals are in the fine topology, in signed (z,y,x) order.
struct AdaptiveFamilyChange3D {
  AdaptiveMacTopology3D topology;
  std::vector<double> previous_pressures;  // Warm starts in Pa, in candidate cell order.
  std::vector<AdaptiveRetainedLeaf3D> retained_leaves;
  std::size_t parent_ordinal;
  std::array<std::size_t, 8> child_ordinals;
};

// Replace an existing parent by all eight children. Carry the parent's
// pressure to its children as an intensive warm start. Other leaves keep
// their pressures. Requires one finite pressure per committed cell, and a
// parent exponent of at least one. Missing parent returns FailedPrecondition.
absl::StatusOr<AdaptiveFamilyChange3D> PrepareAdaptiveRefinement3D(
    const AdaptiveMacTopology3D& committed_topology,
    std::span<const double> committed_pressures,
    AdaptiveMacCellKey3D parent
);

// Replace exactly eight existing sibling leaves by their parent. Their
// equal-volume pressure mean initializes the parent. Missing siblings or
// a grading violation returns FailedPrecondition. Both operations borrow
// inputs and preserve all committed data on failure. InvalidArgument,
// ResourceExhausted and OutOfRange follow the topology and numeric bounds.
absl::StatusOr<AdaptiveFamilyChange3D> PrepareAdaptiveCoarsening3D(
    const AdaptiveMacTopology3D& committed_topology,
    std::span<const double> committed_pressures,
    AdaptiveMacCellKey3D parent
);

}  // namespace fluidity
