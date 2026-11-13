#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"
#include "graded_liquid_boundary_geometry_3d.h"

namespace fluidity {

struct GradedLiquidParticlePath3D {
  std::uint64_t particle_id;
  Vec3 start_m;
  Vec3 end_m;
  std::size_t start_leaf_ordinal = 0;
  std::size_t end_leaf_ordinal = 0;
  // Required for cross-leaf paths. Global canonical interior subface slot.
  std::optional<std::size_t> crossing_face_slot = std::nullopt;
};

struct GradedLiquidWetTransitionRequest3D {
  LiquidClock3D expected_accepted_clock;
  double step_duration_s;
  TranslatingHalfspace3D start_solid;
  GradedLiquidPlanarSurface3D start_liquid;
  double end_liquid_plane_coordinate_m;  // Same normal as the start plane, linear motion in time.
};

struct GradedLiquidWetTransition3D {
  GradedLiquidWetTransitionRequest3D request;
  double end_time_s;  // Unsnapped endpoint in seconds; this result does not advance the clock.
  FrozenGradedPlanarLiquidGeometry3D before;
  FrozenGradedPlanarLiquidGeometry3D after;
  std::vector<LiquidAdaptiveCellState3D> leaves;
  std::vector<GradedLiquidParticlePath3D> paths;
};

// Prove a detached occupancy and inventory transition for a fixed graded roster.
// The request is bound to the entire accepted clock. The accepted lattice and
// request define both exact frozen endpoint poses. A path is required for every
// accepted particle, in storage order, with matching ID and start position.
// The caller supplies particle endpoints. Paths are straight in spacetime,
// strictly liquid/nonsolid at both ends, and remain in one half-open leaf or
// cross one interior subface orthogonally away from its edges. The admitted
// scope is implicit particle-owned liquid and certified empty air without
// tracked cavities. Inventories reproduce all represented particle lanes
// exactly in aggregate at both endpoints. Membership changes clear previous
// pressure and coarsening age. This helper publishes no state or carrier data.
// Invalid accepted data and malformed extents return InvalidArgument. Stale
// clocks, unsupported geometry/material/path, and inventory mismatches return
// FailedPrecondition. Unrepresentable arithmetic returns OutOfRange. Inputs
// are unchanged and there is no candidate on failure.
absl::StatusOr<GradedLiquidWetTransition3D> PrepareGradedLiquidWetTransition3D(
    const LiquidState3D& accepted,
    const GradedLiquidWetTransitionRequest3D& request,
    std::span<const GradedLiquidParticlePath3D> paths
);

}  // namespace fluidity
