#pragma once
#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
namespace fluidity::cuda_internal {
// Samples one signed distance in meters per frozen world position, in input
// order. Output size must equal the position count. Spans must be disjoint from
// arena scratch and the output must be disjoint from inputs. No allocation.
// Invalid data, extents or aliasing return InvalidArgument. Capacity returns
// ResourceExhausted. Sampling an empty union or producing a nonfinite distance
// returns OutOfRange. An unexpected compiled architecture returns
// FailedPrecondition. Runtime errors return Internal and poison the arena.
// Outputs change only after successful arithmetic and an arena-stream fence.
// Ordered FP64 sphere union preserves strict-less ties. Empty positions do not launch.
absl::Status SampleLiquidField3D(
    CudaLiquidArena3D& arena,
    const LiquidSceneField3D& field,
    std::span<const LiquidParticleState3D> particles,
    std::span<const Vec3> positions,
    std::span<double> distances
);
// World-position particle query with the same contract as the batched overload.
absl::StatusOr<double> SampleLiquidField3D(
    CudaLiquidArena3D& arena,
    const LiquidSceneField3D& field,
    std::span<const LiquidParticleState3D> particles,
    Vec3 position
);
// Checked device plane batch at original signed lattice y ordinates in cell
// widths. Shares the world sampler's arena and publication guarantees, with the
// substep batch capacity and compensated plane arithmetic.
absl::Status SampleLiquidLatticePlane3D(
    CudaLiquidArena3D& arena,
    const LiquidHorizontalPlaneSurface3D& plane,
    UniformMacGeometry3D geometry,
    std::span<const double> sample_lattice_ordinates_y,
    std::span<double> distances
);
// Samples in owned face-slot order. Each output span is empty or has
// grid.face_count() entries. The obstacle is already at the requested time.
// Slab wall samples require side +/-1 independently of anchor side, and open
// tanks require side zero. Tags depend on structural distance and anchor side.
// Outputs must be disjoint from one another, inputs and arena scratch. Invalid
// extents, obstacle, side or aliases return InvalidArgument. Capacity returns
// ResourceExhausted. Device arithmetic and runtime errors follow the field
// sampler's contract. The outputs change only after validation and a fence.
absl::Status SampleLiquidFaces3D(
    CudaLiquidArena3D& arena,
    SparseMacGridView3D grid,
    const LiquidObstacle3D& obstacle,
    std::int64_t side,
    std::span<LiquidBoundaryFaceTag3D> tags,
    std::span<LiquidPrescribedWallSample3D> walls
);
// Freezes geometry with the units, input order and workspace borrow rules of
// LiquidBoundaryWorkspace3D::Freeze. The host validates and consumes fenced
// device results in query order. Inputs must be disjoint from arena scratch.
// A device error preserves the previous geometry publication. The returned
// geometry also borrows grid keys from the input topology.
absl::StatusOr<FrozenLiquidBoundaryGeometryView3D> FreezeLiquidGeometry3D(
    CudaLiquidArena3D& arena,
    const LiquidBoundaryGeometryViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace
);
// Builds pressure links from live frozen geometry and fields in frozen cell
// and face order. Inputs must be disjoint from arena scratch. The host consumes
// fenced device results before publishing to the workspace's pressure-link
// region. Success borrows that region until its next successful build, workspace
// move or destruction. Failure preserves the preceding pressure-link result.
absl::StatusOr<LiquidPressureProblemView3D> BuildLiquidLinks3D(
    CudaLiquidArena3D& arena,
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const LiquidPressureFields3D& fields,
    LiquidBoundaryWorkspace3D& workspace
);
}  // namespace fluidity::cuda_internal
