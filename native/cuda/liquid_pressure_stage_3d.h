#pragma once

#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"

namespace fluidity::cuda_internal {
/// Projects a frozen pressure problem with the bounded CUDA mixed-precision solver.
/// The inputs use the ordering, warm-start rules and SI units of ProjectMacPressure3D.
/// `step_duration` is positive and finite, in seconds. Cells are in (z,y,x) order and
/// the exact incident face union is in owned face-slot order, including sealed faces.
/// The caller keeps the grid keys, cells and faces unchanged and disjoint from arena
/// scratch and unpublished workspace storage. The call allocates and frees no storage.
/// The original-operator residual and physical defect include gauge rows.
/// @return After the final result copy and arena fence, a projection borrowing the
/// workspace until its next successful CPU or CUDA projection, move or destruction.
/// Arena growth preserves this borrow. Failure preserves published values and addresses.
/// Invalid inputs or aliases return InvalidArgument. Insufficient arena or workspace
/// capacity returns ResourceExhausted. Sealed incompatibility or a compiled
/// architecture other than 1200 returns FailedPrecondition. Lost curvature or
/// nonconvergence returns Aborted, numerical range failure returns OutOfRange, and
/// an inconsistent result returns Internal. Runtime failures poison the arena.
absl::StatusOr<MacPressureProjectionView3D> ProjectLiquidPressure3D(
    CudaLiquidArena3D& arena,
    SparseMacGridView3D grid,
    double step_duration,
    const LiquidPressureProblemView3D& problem,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace
);
}  // namespace fluidity::cuda_internal
