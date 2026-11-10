#pragma once

#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "fluidity/host/wetted_piston_3d.h"

namespace fluidity {

// The checked CPU contract, with disk sections, area, edge certificates and
// patch flux evaluated by SM120 kernels. Uses existing arena copy/fence and
// fatal Internal poison semantics. An arena without opt-in wet-wall scratch
// returns FailedPrecondition. Inputs must also be disjoint from arena scratch.
// Retains the previous workspace publication on every failure.
absl::StatusOr<WettedPistonView3D> CudaComputeWettedPiston3D(
    CudaLiquidArena3D& arena,
    const WettedPistonInputs3D& inputs,
    WettedPistonWorkspace3D& workspace
);

}  // namespace fluidity
