#pragma once
#include <type_traits>

#include "fluidity/base/liquid_boundary_transfer_math_3d.h"
#include "fluidity/base/liquid_regular_cell_transport_3d.h"
namespace fluidity::cuda_internal {
enum class BoundaryTransferOperation3D {
  kDeposit,
  kNormalize,
  kGather,
  kFitGhost,
  kPredict,
  kForces,
  kUpdateParticles
};
enum class BoundaryHistoryStage3D {
  kOld,
  kNew
};
struct BoundaryTransferHistory3D {
  double face_velocity{};  // Positive normal m/s at this history time.
  LiquidFaceValidity3D validity{};
  LiquidBoundaryFaceTag3D tag{};
  LiquidPrescribedWallSample3D wall{};
};
struct BoundaryTransferFace3D {
  FaceNormalAxis axis{};
  Vec3 anchor_in_cell_widths{};  // Lattice coordinates avoid world-origin cancellation.
  LiquidBoundaryFaceTag3D deposit_tag{};
  BoundaryTransferHistory3D old_history{};
  BoundaryTransferHistory3D new_history{};
  MacFaceDepositQuanta3D face_quanta{};
};
struct BoundaryTransferParticle3D {
  LiquidBoundaryParticle3D particle{};
  // Particle update consumes only these numerical fields; host identity/mass/V0 stay untouched.
  LiquidBoundaryParticleGather3D gathered{};
  double liquid_distance{};  // Frozen start-time signed distance, meters.
  double volume_ratio{};  // Dimensionless start-time J.
  bool pic_initialized{};  // False suppresses retained particle velocity.
  LiquidRegularCellTransport3D regular_cell_transport{};  // Optional projected samples for the update.
  MacFaceStencilSample3D stencils[3][27]{};  // X/Y/Z, then original (z,y,x) node order.
};
struct BoundaryTransferFaceResult3D {
  MacFaceDepositQuanta3D face_quanta{};
  double decoded_face_mass{};  // Kilograms, before the low-mass validity decision.
  double normalized_face_velocity{};  // Momentum/mass in m/s, zero for missing/low mass.
  LiquidFaceValidity3D validity{};
};
struct BoundaryTransferParticleResult3D {
  CompatibleMacFaceSample3D stencils[3][27]{};
  LiquidBoundaryParticleGather3D gathered{};
  Vec3 updated_world_position{};  // Continuous-contact position, meters.
  Vec3 updated_stored_velocity{};  // Retained/contact-corrected m/s, not advection velocity.
  double updated_volume_ratio{};  // Start-time J times exp(dt times the transport divergence).
};
struct BoundaryTransferQuery3D {
  BoundaryTransferOperation3D operation = BoundaryTransferOperation3D::kDeposit;
  std::size_t particle_count{};
  std::size_t face_count{};
  double cell_width{};  // Positive meters.
  MacDepositQuantization3D quantization{};
  double scaled_contribution_magnitude_limit{};  // Inclusive per-particle signed-quanta magnitude bound.
  double minimum_face_mass{};  // Finite nonnegative kilograms.
  internal::LiquidGhostFitSample3D fit_samples[125]{};
  std::size_t fit_count{};
  std::size_t target_face_slot{};  // Checked owned slot for missing-face prediction.
  std::int64_t side_tag{};  // Requested continuation side.
  Vec3 world_acceleration{};  // m/s^2, applied once after old-history continuation.
  double step_duration{};  // Positive seconds.
  boundary_internal::Obstacle obstacle{};  // Start pose and constant translating velocity.
  double particle_energy_blend{};  // e in [0,1].
  double friction_coefficient{};  // Nonnegative dimensionless Coulomb coefficient.
  double penetration_tolerance{};  // Nonnegative meters.
};
struct BoundaryTransferReport3D {
  boundary_internal::Error error{};
  int compiled_architecture{};
  internal::LiquidGhostAffineFit3D fit{};
  double predicted_face_velocity{};  // Continued old velocity plus acceleration*dt, m/s.
  bool retry_step{};  // Physical displacement/J rejection, separate from arithmetic/runtime errors.
};
static_assert(std::is_trivially_copyable_v<BoundaryTransferQuery3D>);
static_assert(std::is_trivially_copyable_v<BoundaryTransferFace3D>);
static_assert(std::is_trivially_copyable_v<BoundaryTransferParticle3D>);
static_assert(std::is_trivially_copyable_v<BoundaryTransferParticleResult3D>);
// Bounded sequential numerical kernel. Resolved POD buffers remain owned until
// synchronization; error and architecture are written on every operation.
int LaunchLiquidBoundaryTransfer3D(
    const BoundaryTransferQuery3D*,
    const BoundaryTransferFace3D*,
    const BoundaryTransferParticle3D*,
    BoundaryTransferFaceResult3D*,
    BoundaryTransferParticleResult3D*,
    BoundaryTransferReport3D*,
    void* stream
);
}  // namespace fluidity::cuda_internal
