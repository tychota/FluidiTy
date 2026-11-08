#pragma once

#include <array>
#include <limits>
#include <memory>
#include <span>
#include <variant>

#include "fluidity/host/liquid_step_workspace_3d.h"

namespace fluidity {
// Finite analytic qualification field: |x-center|-radius in world meters.
struct LiquidAnalyticSphere3D {
  Vec3 world_center{};
  double radius = 1;
};
struct LiquidParticleSphereUnion3D {
  double radius = 1;  // Positive world meters; min in stable trial particle order.
};
using LiquidSceneField3D = std::variant<LiquidAnalyticSphere3D, LiquidParticleSphereUnion3D>;
struct LiquidEmissionSource3D {
  std::uint64_t source_ordinal = 0;
  double particles_per_second = 0;
  Vec3 world_position{};
  Vec3 world_position_jitter{};  // Nonnegative half extents, meters; independent Philox components.
  LiquidParticleState3D emitted_particle_template;  // Resolved payload; ID/position/PIC flag are assigned on emission.
};
struct LiquidSceneInputs3D {
  LiquidSceneField3D field = LiquidParticleSphereUnion3D{};
  LiquidObstacle3D obstacle;  // Pose at world time zero; constant world translation.
  Vec3 world_acceleration{0, -9.81, 0};  // m/s^2, applied once after the old snapshot.
  double liquid_density = 1000;  // kg/m^3.
  double surface_tension = 0;  // N/m.
  double ambient_pressure = 0;  // Pa, gauge.
  double bulk_modulus = std::numeric_limits<double>::infinity();  // Pa.
  double target_divergence = 0;  // 1/s, pressure constraint source.
  double friction_coefficient = 0;
  double penetration_tolerance = 0;  // m.
  std::span<const LiquidEmissionSource3D> sources;  // Copied at Create; exact committed identity order.
};
struct LiquidStepOptions3D {
  double particle_energy_blend = 0.6;  // e in [0,1]; stored alpha=0.99e, surface beta=e.
  double maximum_step_duration = 0.01;  // Positive seconds.
  double minimum_face_mass = 0;  // kg.
  double mass_per_quantum = 0x1p-40;  // Positive normal power of two, kg/quantum.
  double momentum_per_quantum = 0x1p-40;  // Positive normal power of two, kg*m/s per quantum.
  MacPressureSolverOptions3D pressure_solver_options;
  LiquidStateCapacities3D capacities;  // Initial reservations for each version and workspace.
  // All zero disables replay growth. Otherwise every family is nondecreasing and
  // at least one grows; preparation/copy/growth happen only between attempts.
  LiquidStateCapacities3D capacity_replay_reservations;
};
enum class LiquidPhysicalStage3D {
  kEmit,
  kActivate,
  kDeposit,
  kOldSnapshot,
  kForces,
  kWaves,
  kBodyContact,
  kInterface,
  kViscosity,
  kPressure,
  kNewSnapshot,
  kGather,
  kParticleUpdate,
  kWhitewater,
  kValidate,
  kAccept,
  kAdapt,
  kCount
};
inline constexpr std::size_t kLiquidPhysicalStageCount3D = static_cast<std::size_t>(LiquidPhysicalStage3D::kCount);
struct LiquidStepDiagnostics3D {
  std::array<LiquidPhysicalStage3D, kLiquidPhysicalStageCount3D> stages{};
  std::size_t stage_count = 0;
  std::size_t attempts = 0;
  std::size_t accepted_attempts = 0;
  std::size_t dt_retries = 0;
  std::size_t capacity_replays = 0;
  double attempted_duration = 0;  // Seconds for the latest physical attempt.
  MacPressureDiagnostics3D pressure_diagnostics;
  std::size_t old_pic_fallback_count = 0;
  std::size_t new_pic_fallback_count = 0;
  std::size_t lower_theta_regularization_count = 0;
  LiquidPhysicalStage3D failed_stage = LiquidPhysicalStage3D::kCount;
  absl::StatusCode failure_code = absl::StatusCode::kOk;
};
// Hook is diagnostics-only, invoked before each stage through acceptance.
// Disabled postaccept adaptation is recorded as an identity and cannot be faulted. Return code-only errors;
// Aborted requests dt retry, ResourceExhausted requests frame replay, all others fatal.
// The hook and context must remain alive and must not allocate inside an attempt.
struct LiquidStageFaultHook3D {
  void* context = nullptr;
  absl::Status (*before_stage)(void*, LiquidPhysicalStage3D, double) = nullptr;
};
struct LiquidSubstepBackend3D;
// Raw standard Philox4x32-10; counter/key word order is little-word first. No allocation.
std::array<std::uint32_t, 4>
LiquidPhilox4x32_10(std::array<std::uint32_t, 4> counter, std::array<std::uint32_t, 2> key);
// Bounded single-threaded owner; sources/settings are copied and frozen at creation.
// All attempts, including rejection/status destruction, allocate and free nothing.
// Create, frame capacity growth, move and destruction are outside physical attempts.
// Errors have empty messages. Physical state is published only by successful acceptance.
// Qualification limits: 1024 particles, 4096 candidate cells, 24576 owned faces.
// ID validation is bounded quadratic; per-particle ghost scans and sorted halo
// insertion remain reference algorithms, not a large-scene throughput claim.
// Particle side_tag is zero without a slab, and -1/+1 on the negative/positive
// slab sides. An analytic field is stationary in world coordinates; particle union
// centers freeze after emission, before sort/advection. Candidate bounding boxes
// and their complete 27-center/8-corner halos must fit the configured reservations.
class LiquidSubstepScheduler3D {
 public:
  LiquidSubstepScheduler3D(LiquidSubstepScheduler3D&&) noexcept;
  LiquidSubstepScheduler3D& operator=(LiquidSubstepScheduler3D&&) noexcept;
  ~LiquidSubstepScheduler3D();
  LiquidSubstepScheduler3D(const LiquidSubstepScheduler3D&) = delete;
  LiquidSubstepScheduler3D& operator=(const LiquidSubstepScheduler3D&) = delete;
  // Validates all resolved settings/identities and reserves transaction, numeric and
  // staging owners with checked nothrow allocation. No borrowed scene data survives.
  static absl::StatusOr<LiquidSubstepScheduler3D>
  Create(const LiquidState3D& initial, const LiquidSceneInputs3D& scene, LiquidStepOptions3D options);
  // Explicit backend: every typed operation must be supplied; no CPU fallback.
  // Backend context is borrowed through scheduler destruction and all frame attempts.
  static absl::StatusOr<LiquidSubstepScheduler3D> Create(
      const LiquidState3D& initial,
      const LiquidSceneInputs3D& scene,
      LiquidStepOptions3D options,
      const LiquidSubstepBackend3D& backend
  );
  // Read-only version; reacquire record spans after every attempt or capacity replay.
  const LiquidState3D& committed() const;
  // Latest attempt trace/pressure fields and frame-cumulative attempt/replay counts.
  // Accepted count includes work discarded by a capacity replay; it is not an epoch.
  const LiquidStepDiagnostics3D& diagnostics() const;
  // Complete dynamic payload of transaction versions, copied sources, numeric and
  // staging owners; excludes object/allocator overhead and backend-owned storage.
  std::size_t reserved_bytes() const;
  void SetStageFaultHookForTesting(LiquidStageFaultHook3D hook);
  // Begins a frame without allocation; target is finite and strictly later.
  absl::Status BeginFrame(double target_time);
  // Runs one explicit positive duration within the open frame; consumes its trial
  // even on failure. Returns disposition/retry duration; detailed code/stage are in
  // diagnostics. Fatal/exhausted retries close physical progress, never change committed.
  absl::StatusOr<LiquidAttemptOutcome3D> Attempt(double step_duration);
  // Finishes an explicitly stepped frame; no physical bytes/epoch are changed.
  absl::Status FinishFrame();
  // Chooses dt from velocity/source/wall/acceleration/capillary/remaining bounds;
  // accepts at most two dt halvings and one configured frame capacity replay.
  // A failure preserves the latest accepted version (or replay's frame start).
  absl::Status AdvanceFrame(double target_time);

 private:
  struct Impl;
  explicit LiquidSubstepScheduler3D(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace fluidity
