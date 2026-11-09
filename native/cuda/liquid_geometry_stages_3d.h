#pragma once
#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
namespace fluidity::cuda_internal {
// Bounded device sampling at frozen world positions (meters). Counts must fit the
// arena. Inputs/output must not alias arena scratch or each other. No allocation;
// outputs publish only after successful device arithmetic and fence. Invalid
// field/position/alias returns InvalidArgument; capacity returns ResourceExhausted;
// empty union or nonfinite device distance returns OutOfRange. Runtime errors are
// fatal Internal and permanently poison the arena; all external outputs survive.
// Ordered FP64 sphere union preserves strict-less ties. Empty positions still launch.
absl::Status SampleLiquidField3D(
    CudaLiquidArena3D& arena,
    const LiquidSceneField3D& field,
    std::span<const LiquidParticleState3D> particles,
    std::span<const Vec3> positions,
    std::span<double> distances
);
// Scalar adapter for the scheduler; same contract as the batched overload.
absl::StatusOr<double> SampleLiquidField3D(
    CudaLiquidArena3D& arena,
    const LiquidSceneField3D& field,
    std::span<const LiquidParticleState3D> particles,
    Vec3 position
);
// Full owned-slot outputs, each empty or grid.face_count(). Obstacle is already
// at the requested time; slab walls require side +/-1, independently of anchor side.
// Tags depend only on structural distance and anchor side, never mass/validity.
// Outputs must not alias one another, inputs, or arena scratch. Invalid extents,
// obstacle/side/alias return InvalidArgument, capacity ResourceExhausted, nonfinite
// device arithmetic OutOfRange, runtime errors fatal Internal. Outputs survive errors.
absl::Status SampleLiquidFaces3D(
    CudaLiquidArena3D& arena,
    SparseMacGridView3D grid,
    const LiquidObstacle3D& obstacle,
    std::int64_t side,
    std::span<LiquidBoundaryFaceTag3D> tags,
    std::span<LiquidPrescribedWallSample3D> walls
);
// Same units/statuses/borrow lifetimes as bounded CPU geometry; every arithmetic
// callback runs on device. Workspace's previous publication survives errors.
absl::StatusOr<FrozenLiquidBoundaryGeometryView3D> FreezeLiquidGeometry3D(
    CudaLiquidArena3D& arena,
    const LiquidBoundaryGeometryViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace
);
absl::StatusOr<LiquidPressureProblemView3D> BuildLiquidLinks3D(
    CudaLiquidArena3D& arena,
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const LiquidPressureFields3D& fields,
    LiquidBoundaryWorkspace3D& workspace
);
}  // namespace fluidity::cuda_internal
