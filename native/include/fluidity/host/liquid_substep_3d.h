#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <variant>

#include "fluidity/host/liquid_step_workspace_3d.h"
#include "fluidity/host/ocean_boundary_3d.h"
#include "fluidity/host/rigid_body_3d.h"
#include "fluidity/host/whitewater_3d.h"

namespace fluidity {
// Finite analytic qualification field: |x-center|-radius in world meters.
struct LiquidAnalyticSphere3D {
  Vec3 world_center{};
  double radius = 1;
};
struct LiquidParticleSphereUnion3D {
  double radius = 1;  // Positive world meters; min in stable trial particle order.
};
// World signed distance is y minus this fixed height in meters. The scheduler
// requires an open tank with a strictly lower floor to bound pressure support.
struct LiquidHorizontalPlaneSurface3D {
  double world_surface_height_y = 0;
};
// Opt-in field for the authored filled, isolated drop on the uniform grid.
// Admits exactly 552 particles with side_tag == 0, positive surface tension,
// incompressibility, zero target divergence, and no obstacles or sources.
struct LiquidIsolatedCovarianceDrop3D {
  double physical_radius;  // Authored radius in meters, used to scale the grid level set.
  double seed_covariance_scale;  // Positive dimensionless scale fixed at authoring.
};
using LiquidSceneField3D = std::variant<
    LiquidAnalyticSphere3D,
    LiquidParticleSphereUnion3D,
    LiquidHorizontalPlaneSurface3D,
    LiquidIsolatedCovarianceDrop3D>;
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
struct WhitewaterCaptureSettings3D {
  Vec3 world_source_grid_origin{};
  WhitewaterSourceOptions3D source_options{};
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
  std::uint32_t motion_sample_intervals = 0;  // Zero disables effects capture; otherwise 1..64 accepted-time slots.
  std::optional<WhitewaterCaptureSettings3D> whitewater_capture{
  };  // Optional accepted source events; analytic control.
  std::optional<OceanBoundarySettings3D> ocean_boundary;  // Optional inlet and outlet velocity relaxation.
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
// Measurement boundaries shared by the host scheduler and CUDA dispatch.
enum class LiquidWorkPhase3D {
  kEmitActivationField,
  kDepositNormalization,
  kForcesGeometryPressure,
  kGatherContactAcceptance,
  kCount,
};
// Optional nonallocating observer. kCount ends a physical interval and returns
// to scheduler work. The callback may not retain borrowed state or reenter.
struct LiquidWorkPhaseObserver3D {
  void* context = nullptr;
  void (*enter)(void*, LiquidWorkPhase3D) = nullptr;
};
struct LiquidPistonStepPose3D {
  double step_start_low_x_m;  // Low-x wall used to freeze the pressure geometry.
  double step_end_low_x_m;  // Low-x wall used for end-of-step tags and contact.
};
// Observations of a single accepted attempt, copied before workspace reuse.
// Pressure uses the original FP64 residual and defect over the step-start
// pressure cells, including gauges. Empty pressure domains have zero metrics.
struct LiquidAcceptedStepMeasurement3D {
  LiquidClock3D accepted_clock;
  double accepted_dt_seconds = 0;  // Exact duration passed to the physical attempt.
  std::optional<double> step_start_capillary_bound_seconds;  // Absent when the frozen geometry imposes no bound.
  MacPressureDiagnostics3D pressure;
  double pressure_rhs_l2_m3_per_s = 0;  // L2 norm of the original integrated RHS, including gauge rows.
  double step_start_pressure_cell_volume_m3 = 0;  // Frozen liquid volume in center-selected pressure cells.
  // Compensated sum of explicit one-sided piston wall flux in owned face order.
  // Positive into selected cells and negative for a receding wall.
  double step_start_inward_piston_flux_m3_per_s = 0;
  std::optional<LiquidPistonStepPose3D> piston_pose;  // Present for an open tank, including a stationary piston.
};
// Actual work from one AdvanceFrame attempt, including unsuccessful work.
// Frame-local counters reset at BeginFrame. The capacity pass starts at zero
// and advances only on a successful restart. Start clock may repeat on replay.
struct LiquidAttemptMeasurement3D {
  LiquidClock3D start_clock;
  std::size_t attempt_in_frame;
  std::size_t capacity_pass;
  double attempted_dt_seconds;  // Simulated duration supplied to Attempt.
  double work_wall_seconds;  // Monotonic host elapsed time inside Attempt, including its fences.
  LiquidAttemptDisposition3D disposition;
  std::optional<double> retry_dt_seconds;  // Exact next duration requested by a successful dt retry decision.
  absl::StatusCode failure_code;
};
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
  // Present only after successful publication of the latest attempt. A
  // restarted frame invalidates prior accepted measurements from its old pass.
  std::optional<LiquidAcceptedStepMeasurement3D> accepted_measurement;
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
// Called outside physical attempts after acceptance or a capacity rollback. State
// and diagnostics borrow only through the call. rollback invalidates every prior
// intra-frame observation from this pass. A non-OK status stops scheduling with
// the latest coherent state retained; it does not publish a point frame. The
// callback must not reenter the scheduler or retain device source leases. Drain
// pending snapshot sources before AdvanceFrame; this callback is not a device
// reuse gate. Default-empty observers preserve the numerical schedule exactly.
// observe_attempt runs after every physical attempt and before the matching
// acceptance or rollback notification, even when scheduling will fail. It
// borrows a scalar receipt only for the call, cannot stop scheduling, and must
// not reenter the scheduler. Use observe to detect successful capacity rollback.
struct LiquidProgressObserver3D {
  void* context = nullptr;
  absl::Status (*observe)(void*, const LiquidState3D&, const LiquidStepDiagnostics3D&, bool rollback) = nullptr;
  void (*observe_attempt)(void*, const LiquidAttemptMeasurement3D&) = nullptr;
  LiquidWorkPhaseObserver3D work_phase{};
};
// Bounded single-threaded owner; sources/settings are copied and frozen at creation.
struct LiquidPlanarPistonAttempt3D {
  LiquidAttemptOutcome3D outcome;
  std::optional<RigidBody3D> loaded_piston;  // Present only when the liquid attempt is accepted.
};

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
// With a translating tank and selected pressure cells, particle positions must
// have certifiable half-open containing cells at step start or the attempt
// fails with OutOfRange. Inside selected cells, only usable incident carrier
// faces supply direct new-history values and continuation donors. Other
// particles use the full carrier. Old history and carrier publication retain
// their own eligibility rules.
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
  // Explicit backend: every operation required by the scene must be supplied.
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
  // The accepted-attempt count includes work discarded by capacity replay.
  // Use the accepted measurement's clock for the physical step ordinal.
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
  absl::StatusOr<LiquidAttemptOutcome3D> Attempt(double step_duration, LiquidWorkPhaseObserver3D work_phase = {});
  // Attempts the moving planar piston scene and returns the pressure load
  // only when the liquid state commits. The body is supplied by value and
  // the piston trajectory stays prescribed for this attempt.
  absl::StatusOr<LiquidPlanarPistonAttempt3D>
  AttemptWithPlanarPistonPressureLoad(double step_duration, RigidBody3D piston);
  // Finishes an explicitly stepped frame; no physical bytes/epoch are changed.
  absl::Status FinishFrame();
  // Chooses dt from velocity/source/wall/acceleration/capillary/remaining bounds;
  // accepts at most two dt halvings and one configured frame capacity replay.
  // A failure preserves the latest accepted version (or replay's frame start).
  absl::Status AdvanceFrame(double target_time, LiquidProgressObserver3D observer = {});

 private:
  absl::StatusOr<LiquidAttemptOutcome3D>
  AttemptImpl(double step_duration, LiquidWorkPhaseObserver3D work_phase, RigidBody3D* piston_load);
  struct Impl;
  explicit LiquidSubstepScheduler3D(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace fluidity
