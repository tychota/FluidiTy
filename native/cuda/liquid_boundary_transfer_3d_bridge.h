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
  kForces
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
  MacFaceStencilSample3D stencils[3][27]{};  // X/Y/Z, then original (z,y,x) node order.
};
struct BoundaryTransferFaceResult3D {
  MacFaceDepositQuanta3D face_quanta{};
  double decoded_face_mass{};  // Kilograms, before the low-mass validity decision.
  double normalized_face_velocity{};  // Momentum/mass in m/s, zero for missing/low mass.
  LiquidFaceValidity3D validity{};
};
// The bounded normalization dispatch borrows the transfer face ranges with
// compact strides. Its inputs are MacFaceDepositQuanta3D in owned face order.
struct BoundaryNormalizedFace3D {
  double decoded_face_mass{};  // Kilograms, before the low-mass decision.
  double normalized_face_velocity{};  // m/s, zero for missing or low mass.
  LiquidFaceValidity3D validity{};
};
static_assert(sizeof(MacFaceDepositQuanta3D) <= sizeof(BoundaryTransferFace3D));
static_assert(alignof(MacFaceDepositQuanta3D) <= alignof(BoundaryTransferFace3D));
static_assert(sizeof(BoundaryNormalizedFace3D) <= sizeof(BoundaryTransferFaceResult3D));
static_assert(alignof(BoundaryNormalizedFace3D) <= alignof(BoundaryTransferFaceResult3D));

struct BoundaryTransferParticleResult3D {
  CompatibleMacFaceSample3D stencils[3][27]{};
  LiquidBoundaryParticleGather3D gathered{};
};
// A gather run reuses particle result scratch, with only the fields it copies back.
struct BoundaryGatherRunResult3D {
  LiquidBoundaryParticleGather3D gathered{};
  boundary_internal::Error error{};
  int compiled_architecture{};
};
static_assert(sizeof(BoundaryGatherRunResult3D) <= sizeof(BoundaryTransferParticleResult3D));
static_assert(alignof(BoundaryGatherRunResult3D) <= alignof(BoundaryTransferParticleResult3D));

// Update records reuse the checked transfer particle ranges. Identity, mass and
// reference volume stay with the host particle.
struct BoundaryUpdateParticle3D {
  Vec3 world_position{};
  LiquidBoundaryParticleGather3D gathered{};
  double liquid_distance{};  // Frozen start-time signed distance, meters.
  double volume_ratio{};  // Dimensionless start-time J.
  bool pic_initialized{};  // False suppresses retained particle velocity.
  LiquidRegularCellTransport3D regular_cell_transport{};  // Optional projected normal samples.
};
struct BoundaryUpdateResult3D {
  Vec3 world_position{};  // Continuous-contact position, meters.
  Vec3 stored_world_velocity{};  // Retained/contact-corrected m/s.
  double volume_ratio{};  // Start-time J times exp(dt times the transport divergence).
  boundary_internal::Error error{};
  bool retry_step{};
};
static_assert(sizeof(BoundaryUpdateParticle3D) <= sizeof(BoundaryTransferParticle3D));
static_assert(alignof(BoundaryUpdateParticle3D) <= alignof(BoundaryTransferParticle3D));
static_assert(sizeof(BoundaryUpdateResult3D) <= sizeof(BoundaryTransferParticleResult3D));
static_assert(alignof(BoundaryUpdateResult3D) <= alignof(BoundaryTransferParticleResult3D));

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
inline constexpr unsigned kBoundaryTransferErrorCodeCount = static_cast<unsigned>(boundary_internal::Error::kRange) + 1;
inline constexpr auto kNoParallelTransferError = ~0ULL;

struct BoundaryTransferReport3D {
  boundary_internal::Error error{};
  int compiled_architecture{};
  internal::LiquidGhostAffineFit3D fit{};
  double predicted_face_velocity{};  // Continued old velocity plus acceleration*dt, m/s.
  // Minimum ordered (ordinal * error code count + error code), or the sentinel.
  unsigned long long first_parallel_error = kNoParallelTransferError;
};
static_assert(std::is_trivially_copyable_v<BoundaryTransferQuery3D>);
static_assert(std::is_trivially_copyable_v<BoundaryTransferFace3D>);
static_assert(std::is_trivially_copyable_v<BoundaryTransferParticle3D>);
static_assert(std::is_trivially_copyable_v<BoundaryTransferParticleResult3D>);
// Update each independent particle. Checked input and result ranges remain live
// through the dispatch fence. The host consumes status in particle order.
int LaunchLiquidParticleUpdate3D(
    const BoundaryTransferQuery3D* device_query,
    const BoundaryUpdateParticle3D* device_particles,
    BoundaryUpdateResult3D* device_results,
    int* device_compiled_architecture,
    std::size_t particle_count,
    void* stream
);

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

// Launch one lane per particle/component, with cleared face quanta, or one
// lane per face for normalization or forces. Counts match the nonempty device query.
// The report starts at its default value and remains owned through the fence.
int LaunchLiquidBoundaryDeposit3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D* particle_results,
    BoundaryTransferReport3D* report,
    std::size_t particle_count,
    void* stream
);
// Normalize one decoded face per thread. The report carries the first error
// in owned face order, and callers fence before publishing any results.
int LaunchLiquidBoundaryNormalize3D(
    const BoundaryTransferQuery3D* query,
    const MacFaceDepositQuanta3D* face_quanta,
    BoundaryNormalizedFace3D* face_results,
    BoundaryTransferReport3D* report,
    std::size_t face_count,
    void* stream
);
int LaunchLiquidBoundaryFaceMap3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferReport3D* report,
    std::size_t face_count,
    void* stream
);

// Prepares the original stencil samples for one staged kGather particle.
// Uses the same bounded buffers and synchronization contract as the sequential launch.
int LaunchLiquidBoundarySingleParticleGather3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryTransferFaceResult3D* face_results,
    BoundaryTransferParticleResult3D* particle_results,
    BoundaryTransferReport3D* report,
    void* stream
);
// One block per particle, with one compact result and error in input order.
int LaunchLiquidBoundaryGatherRun3D(
    const BoundaryTransferQuery3D* query,
    const BoundaryTransferFace3D* faces,
    const BoundaryTransferParticle3D* particles,
    BoundaryGatherRunResult3D* results,
    unsigned particle_count,
    void* stream
);
}  // namespace fluidity::cuda_internal
