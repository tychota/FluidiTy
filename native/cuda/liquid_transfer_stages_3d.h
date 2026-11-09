#pragma once
#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
namespace fluidity::cuda_internal {
// Bounded synchronous device stages for the typed scheduler callbacks. The arena
// and workspace are borrowed, single-threaded, and non-reentrant. Inputs may borrow
// published workspace regions, but no input/output may alias arena scratch; writable
// outputs must not overlap read-only inputs. Grid keys remain borrowed during the call.
// All operations allocate/free nothing, including errors and result destruction.
// Invalid input: InvalidArgument; insufficient configured capacity: ResourceExhausted;
// missing support/donors: FailedPrecondition; unrepresentable arithmetic: OutOfRange.
// Runtime submission/fence failure poisons the arena and returns Internal. No staging
// reuse follows poison. External outputs and previous workspace publications survive
// every failure. Successful borrowed results expire on the next successful operation
// in the SAME workspace region (CPU or CUDA), workspace move or destruction.
// Deposition checks complete original support before device side/eligibility masking,
// normalized quotient gradients, recentering and signed nearest-even quantization.
absl::StatusOr<LiquidBoundaryFaceDepositsView3D> DepositLiquid3D(
    CudaLiquidArena3D& arena,
    const LiquidBoundaryDepositViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace
);
// Decode signed sums to kg and m/s. Positive mass <= minimum_face_mass retains its
// physical kg value but publishes zero velocity and explicit kLowMass validity.
absl::StatusOr<NormalizedLiquidMacFacesView3D> NormalizeLiquid3D(
    CudaLiquidArena3D& arena,
    std::span<const MacFaceDepositQuanta3D> face_quanta,
    MacDepositQuantization3D quantization,
    double minimum_face_mass,
    LiquidBoundaryWorkspace3D& workspace
);
// Continue the old face in owned-slot donor order on the requested side, including
// rank/nearest/wall fallback, then add acceleration*dt once in its Cartesian axis.
// The finite positive duration is seconds; old history/validity remain untouched.
absl::StatusOr<double> PredictMissingLiquidFace3D(
    CudaLiquidArena3D& arena,
    const LiquidMissingFacePredictionInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace
);
// Full owned-slot old/predicted m/s arrays; world acceleration is m/s^2 and duration
// is finite positive seconds. Every predicted value is validated before publication.
absl::Status ApplyLiquidForces3D(
    CudaLiquidArena3D& arena,
    SparseMacGridView3D grid,
    std::span<const double> old_face_velocities,
    Vec3 world_acceleration,
    double step_duration,
    std::span<double> predicted_face_velocities
);
// Gather on the original complete quadratic basis with independently sampled old
// and new tags/walls. Returns true gradients separately from fitted APIC slopes.
// Ghost rank fallback and either history's low mass suppress retained velocity.
absl::StatusOr<LiquidBoundaryGatherView3D> GatherLiquid3D(
    CudaLiquidArena3D& arena,
    const LiquidBoundaryGatherViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace
);
// Update J by exp(dt*tr(true gradient)); separately constrain retained/stored and
// advecting velocities by continuous contact. Preserve ID/tag/side/mass/V0 exactly.
// Nonpositive/nonfinite J or displacement >0.5h returns Aborted (smaller-dt retry).
// All particle records remain unchanged on any failure, even after earlier particles
// have computed valid candidate results. No result pointer into arena scratch escapes.
absl::Status UpdateLiquidParticles3D(
    CudaLiquidArena3D& arena,
    const LiquidParticleUpdateInputs3D& inputs,
    std::span<LiquidParticleState3D> trial_particles
);
}  // namespace fluidity::cuda_internal
