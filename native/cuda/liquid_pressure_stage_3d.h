#pragma once

#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"

namespace fluidity::cuda_internal {
// Bounded projection using prepared host coefficients and the real CUDA mixed-
// precision recurrence/physical correction. Same SI units, ordering, warm start,
// nonempty counts and error categories as ProjectMacPressure3D. Nonconvergence
// or lost curvature returns Aborted; incompatible sealed flux is fatal
// FailedPrecondition. Capacity is ResourceExhausted; invalid inputs/aliases are
// InvalidArgument, numerical range is OutOfRange, identity failure is Internal.
// Inputs must not
// overlap arena scratch or unpublished workspace storage. No allocation/free.
// Runtime failures permanently poison the arena. Failed calls preserve all
// previously published projection values and addresses. Successful output borrows
// workspace until its next successful CPU/CUDA projection, move or destruction.
// Arena growth does not invalidate output; each call reacquires its named ranges.
// Preparation spans remain internal to this synchronous call. All returned data
// are copied after the final fence; no pinned arena scratch pointer escapes.
absl::StatusOr<MacPressureProjectionView3D> ProjectLiquidPressure3D(
    CudaLiquidArena3D& arena,
    SparseMacGridView3D grid,
    double step_duration,
    const LiquidPressureProblemView3D& problem,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace
);
}  // namespace fluidity::cuda_internal
