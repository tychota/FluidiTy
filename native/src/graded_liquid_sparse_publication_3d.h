#pragma once

#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"
#include "graded_liquid_boundary_geometry_3d.h"

namespace fluidity {

struct GradedSparsePressureResultView3D {
  std::span<const double> cell_pressures;  // Pa, in prepared pressure row order.
  std::span<const double> corrected_face_velocities;  // m/s along the positive normal, in prepared link order.
};

// Publish a solver-accepted sparse result into detached trial records at the
// supplied clock. The frozen geometry and prepared graph must match the trial
// lattice, canonical wet membership and global face incidence. Stage checked
// predictor carriers in trial.adaptive_faces before calling. Open links keep
// the predictor's valid/low-mass state and receive the corrected velocity.
// Wet, pressure-sealed links retain the qualified zero predictor. Omitted
// and geometrically dry or blocked faces become missing with zero velocity. Air leaf
// warm-start pressure becomes zero. Side tags and structural flags are kept.
// The result spans must be disjoint from both trial record spans. A rejected
// solver status propagates. Malformed inputs return InvalidArgument; stale
// clock, unsupported occupancy and missing open carriers return FailedPrecondition.
// No trial record changes unless every preflight check succeeds.
absl::Status PublishGradedSparsePressure3D(
    LiquidState3D& trial,
    LiquidClock3D expected_trial_clock,
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    const GradedWetPressureSubset3D& subset,
    const absl::StatusOr<GradedSparsePressureResultView3D>& projection
);

}  // namespace fluidity
