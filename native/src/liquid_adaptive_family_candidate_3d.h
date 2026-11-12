#pragma once

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity::liquid_step_internal {

// Proposes one aligned family change for the stationary planar tank. Copies
// the committed clock, IDs and particle records, remaps pressure and signed
// volumetric flux in canonical order, and checks that particle-owned mass,
// volumes and momentum exactly match the remapped inventory in each leaf.
// FailedPrecondition denotes a change that requires certified resampling or
// incompatible face history. Capacity and checked topology errors propagate.
// The input remains unchanged on every result.
absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidFamilyCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D parent,
    bool refine
);

}  // namespace fluidity::liquid_step_internal
