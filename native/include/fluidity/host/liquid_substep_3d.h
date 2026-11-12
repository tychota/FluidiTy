#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "absl/functional/function_ref.h"
#include "fluidity/host/adaptive_width_policy_3d.h"
#include "fluidity/host/liquid_band_handoff_3d.h"
#include "fluidity/host/liquid_cavity.h"
#include "fluidity/host/liquid_eos.h"
#include "fluidity/host/liquid_step_workspace_3d.h"
#include "fluidity/host/ocean_boundary_3d.h"
#include "fluidity/host/plastic_solid_checkpoint_3d.h"
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
// uses an open tank with a strictly lower floor or an explicitly requested
// graded translating-halfspace trial to bound pressure support.
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
struct LiquidSceneEos3D {
  double reference_sound_speed;  // Positive finite m/s at the scene's reference density.
  LiquidEosLaw law = LiquidEosLaw::kHencky;
};
struct LiquidSceneDrift3D {
  double correction_time;  // Positive finite seconds.
  double maximum_rate;  // Finite nonnegative s⁻¹.
};
struct LiquidStepCavityBoundaries3D {
  std::span<const LiquidAirCellPressure3D> air_pressures;  // Sorted air centers and gas-only gauge Pa.
  double maximum_step_duration = std::numeric_limits<double>::infinity();  // Radial limit in seconds.
};

struct LiquidStepCavityProvider3D {
  void* context = nullptr;
  // Evaluates the step-start samples on each attempt, including retries.
  // The returned pressure span must remain valid for that attempt. Evaluation
  // can use caller scratch and must not publish accepted lineage state.
  absl::StatusOr<LiquidStepCavityBoundaries3D> (*prepare)(
      void* context,
      double step_start_time,
      const LiquidBoundaryGeometryViewInputs3D& samples
  ) = nullptr;
};

struct LiquidSceneCavity3D {
  double atmospheric_absolute_pressure;  // Positive absolute Pa.
  double birth_hydrostatic_gauge_pressure = 0.0;  // Nonnegative Pa for default spherical birth equilibrium.
  void* context = nullptr;
  // Optional certification of atmosphere outlets for the current samples.
  // The returned span is borrowed until the end of step preparation. The
  // initial snapshot uses a zero-duration query at the initial obstacle pose.
  absl::StatusOr<std::span<const FaceIndex3D>> (*atmosphere_outlets)(
      void* context,
      const LiquidBoundaryGeometryViewInputs3D& samples
  ) = nullptr;
  // Overrides spherical birth equilibrium from scene pressures and surface tension. Both callbacks
  // may be repeated after a rejected step or a frame replay.
  absl::StatusOr<LiquidCavityInventory3D> (*initialize)(void* context, const LiquidAirComponent3D& component) = nullptr;
};

struct LiquidSceneInputs3D {
  LiquidSceneField3D field = LiquidParticleSphereUnion3D{};
  LiquidObstacle3D obstacle;  // Pose at world time zero; constant world translation.
  Vec3 world_acceleration{0, -9.81, 0};  // m/s^2, applied once after the old snapshot.
  double liquid_density = 1000;  // kg/m^3.
  double dynamic_viscosity = 0;  // Pa s. Zero disables grid viscosity.
  double surface_tension = 0;  // N/m.
  double ambient_pressure = 0;  // Pa, gauge.
  double bulk_modulus = std::numeric_limits<double>::infinity();  // Legacy constant modulus in Pa.
  std::optional<LiquidSceneDrift3D> drift;  // Optional particle-band correction for side-zero liquid.
  std::optional<LiquidSceneEos3D> eos;  // Requires infinite legacy modulus. Absence keeps legacy behavior.
  double target_divergence = 0;  // 1/s, pressure constraint source.
  double friction_coefficient = 0;
  double penetration_tolerance = 0;  // m.
  std::span<const LiquidPhasePair3D> phase_pairs{};  // Copied at Create in (phase_a,phase_b) order.
  std::span<const LiquidConstituentMaterial3D> constituent_materials{};  // Copied at Create in substance ID order.
  std::span<const LiquidEmissionSource3D> sources;  // Copied at Create; exact committed identity order.
  std::optional<LiquidSceneCavity3D> cavity;  // Owns accepted gas state in the scheduler.
  LiquidStepCavityProvider3D cavity_provider{};  // Optional host evaluation, borrowed for scheduler lifetime.
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
  std::optional<double> step_start_cavity_bound_seconds;  // Present for a finite radial limit.
  std::optional<LiquidCavityStepPreparation3D> cavity;  // Accepted geometry and per-step inventory losses.
  MacPressureDiagnostics3D pressure;
  std::optional<double> viscous_max_momentum_residual_kg_m_per_s;  // Present after a coupled viscosity solve.
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
struct LiquidParticleUpdateInputs3D;
using LiquidPistonReactionUpdate3D =
    absl::FunctionRef<absl::StatusOr<double>(const LiquidParticleUpdateInputs3D&, std::span<LiquidParticleState3D>)>;

struct LiquidPlanarPistonAttempt3D {
  LiquidAttemptOutcome3D outcome;
  std::optional<RigidBody3D> loaded_piston;  // Present only when the liquid attempt is accepted.
};

struct LiquidSnowStepInputs3D {
  const SparseMacGrid3D& solid_grid;  // Uniform complete-support MAC layout.
  std::span<const PlasticSolidCheckpointParticle3D> accepted_particles;
  std::uint64_t body_id;
  std::uint64_t material_id;
  SnowParameters3D snow;
  DrySolidStep3D solid_step;  // Attempt duration replaces solid_step.duration.
  double support_radius;  // Common sphere-union support radius in metres.
};

struct LiquidSnowAttempt3D {
  LiquidAttemptOutcome3D outcome;
  // Present only on shared acceptance and ordered by the original particle IDs.
  std::optional<std::vector<PlasticSolidCheckpointParticle3D>> accepted_snow;
};

struct TankPressureLoad3D {
  Vec3 force{};  // World N exerted by the stationary tank on the liquid.
  Vec3 torque{};  // World N m about the world origin, exerted on the liquid.
};

struct OrientedFloatingBoxLoads3D {
  Vec3 external_force{};  // World N, supplied at step start.
  Vec3 external_torque{};  // World N m about the box center.
  Vec3 pressure_force{};  // World N on the frozen wet surface.
  Vec3 pressure_torque{};  // World N m about the frozen box center.
  Vec3 particle_impulse{};  // World N s on the box.
  Vec3 particle_angular_impulse{};  // World N m s about the center at impact.
  Vec3 liquid_gather_impulse{};  // World N s, committed-to-gathered particle momentum change before contact.
  Vec3 tank_particle_impulse{};  // World N s, stationary tank contact on particles.
  // World N s from the velocity correction on open MAC faces, using face dual
  // mass (aperture times gradient distance divided by inverse density).
  Vec3 pressure_grid_impulse{};
  std::optional<Vec3> particle_pressure_impulse;  // World N s of nonhydrostatic pressure transferred to particles.
  Vec3 routed_pressure_angular_impulse_change{};  // World N m s from moving unsupported impulses between MAC anchors.
  std::size_t routed_pressure_faces = 0;  // Nonzero source impulses moved to compatible wet support.
  std::optional<TankPressureLoad3D> tank_pressure;  // Present when the pressure cells cover every wet tank wall.
};
struct LiquidFloatingBoxAttempt3D {
  LiquidAttemptOutcome3D outcome;
  std::optional<RigidBody3D> body;  // Present only on liquid acceptance.
  std::optional<double> added_mass_kg;  // Measured for an accepted guided coupled attempt.
  std::optional<OrientedFloatingBoxLoads3D> oriented_loads;  // Present only on oriented acceptance.
};

// Borrowed synchronously for a single certified split. The refinement
// closure must name every required parent exactly once.
struct LiquidAdaptiveQuadraticSplitRequest3D {
  std::uint64_t accepted_epoch;
  std::span<const AdaptiveMacCellKey3D> exact_refinement_closure;
  AdaptiveMacCellKey3D split_parent;
  std::uint64_t source_particle_id;
};

// Borrowed synchronously for a single certified merge. Supply the explicit
// coarsening halo and all two to eight distinct source IDs for the selected
// family. Order of the closure and IDs does not determine reduction order.
struct LiquidAdaptiveQuadraticMergeRequest3D {
  std::uint64_t accepted_epoch;
  std::span<const AdaptiveMacCellKey3D> exact_coarsening_halo;
  AdaptiveMacCellKey3D merge_parent;
  std::span<const std::uint64_t> source_particle_ids;
};

struct LiquidAdaptivePhysicalExteriorPressure3D {
  double pressure_pa;  // Gauge pressure including any caller-supplied lift.
  double distance_m;  // Positive pressure gradient distance.
};

// Borrowed for one synchronous attempt. The resolved scene supplies the
// translating halfspace at time zero and the horizontal planar liquid surface.
struct LiquidAdaptivePhysicalGeometryRequest3D {
  LiquidClock3D accepted_clock;  // Must equal the current committed clock.
  std::span<const double> target_divergences_s_inv;  // One independent source per canonical leaf.
  // Empty seals all exteriors. Otherwise supply one optional entry per
  // canonical global subface slot. Interior slots must be absent.
  std::span<const std::optional<LiquidAdaptivePhysicalExteriorPressure3D>> exterior_pressures{};
};

// Width evidence for one accepted leaf and its candidate sibling family.
struct LiquidAdaptiveWidthRequest3D {
  std::uint64_t accepted_epoch;
  AdaptiveMacCellKey3D leaf;
  AdaptiveWidthDemand3D demand;
  // Complete measurements of the parent, particle support and movement halo.
  // The borrowed samples must describe accepted_epoch and live for this call.
  std::span<const AdaptiveCoarsenHaloSample3D> parent_and_halo;
  bool complete_halo;
  double hysteresis_duration_s;
};

// One band placement request for the current accepted graded epoch.
struct LiquidAdaptiveBandRequest3D {
  std::uint64_t accepted_epoch;
  AdaptiveMacCellKey3D leaf;
  LiquidBandMargins3D margins;
  std::optional<LiquidHandoffCarrier3D> destination_carrier;
  std::size_t reseed_particle_count = 1;
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

  // Request one graded family refinement (refine=true) or coarsening before
  // the physical attempt. Preparation errors leave the current frame open
  // and consume no attempt. Rejection preserves the accepted topology and inventories.
  absl::StatusOr<LiquidAttemptOutcome3D>
  AttemptWithAdaptiveFamily(double step_duration, AdaptiveMacCellKey3D parent, bool refine);
  // Refine the requested committed parents and the necessary 2:1 closure.
  // Coarser parents are prepared first. All changed inventories and signed
  // subfaces must pass exact rebinning and remap before one physical trial.
  // An empty request is invalid. Refusal or rejection leaves the committed
  // topology, histories and inventories unchanged.
  absl::StatusOr<LiquidAttemptOutcome3D>
  AttemptWithAdaptiveRefinementClosure(double step_duration, std::span<const AdaptiveMacCellKey3D> requested_parents);
  // Try the width policy for one leaf and one possible sibling family. The
  // scheduler checks the accepted epoch, derives sibling membership from the
  // committed keys and freezes any topology change before the physical trial.
  // Coarsening uses only previously accepted eligible seconds. A kept family
  // earns the proposed duration only on acceptance. An accepted trial resets
  // other eligibility, including when it receives no width evidence. Refusal
  // or rejection preserves the previous ages and topology.
  absl::StatusOr<LiquidAttemptOutcome3D>
  AttemptWithAdaptiveWidth(double step_duration, const LiquidAdaptiveWidthRequest3D& request);
  // Attempt a positive duration in seconds with one particle's quadratic
  // MAC transfer, selected by ID in the exact committed epoch. Requires a
  // graded, inviscid incompressible particle-only tank. Unsupported carrier/support
  // eligibility uses the ordinary radial transfer before the trial starts.
  // A stale epoch, absent ID, malformed input or later physical failure leaves
  // the committed state and backend publication unchanged.
  absl::StatusOr<LiquidAttemptOutcome3D>
  AttemptWithAdaptiveQuadraticTransfer(double step_duration, std::uint64_t committed_epoch, std::uint64_t particle_id);

  // Attempt a positive duration in seconds with a detached certified F3
  // split in an inviscid incompressible graded tank. The committed epoch
  // must match the request. All eight replacement transfers must qualify
  // before any physical stage. Rejection leaves the previous publication.
  absl::StatusOr<LiquidAttemptOutcome3D>
  AttemptWithAdaptiveQuadraticSplit(double step_duration, const LiquidAdaptiveQuadraticSplitRequest3D& request);

  // Attempt a positive duration in seconds with a detached certified F4
  // merge in an inviscid incompressible graded tank. The committed epoch
  // must match the request and the replacement transfer must qualify before
  // any physical stage. Rejection leaves the previous publication.
  absl::StatusOr<LiquidAttemptOutcome3D>
  AttemptWithAdaptiveQuadraticMerge(double step_duration, const LiquidAdaptiveQuadraticMergeRequest3D& request);

  // Evaluate the particle-band policy at the accepted stationary planar
  // interface using the named leaf's full-volume inventory and swept margins.
  // The caller certifies support, movement and any destination carrier. One
  // physical trial either keeps ownership, retires the leaf, or reseeds it.
  // Stale evidence or unavailable handoff inputs refuse without consuming a
  // trial. Rejection leaves both committed ownership versions unchanged.
  absl::StatusOr<LiquidAttemptOutcome3D>
  AttemptWithAdaptiveBandPlacement(double step_duration, const LiquidAdaptiveBandRequest3D& request);

  // Attempt an admitted graded halfspace scene on its unchanged leaf layout.
  // Every such attempt, including retries and restored continuations, requires
  // a clock-bound request with canonical source and exterior fields.
  // The wall pose is derived from the same resolved scene and accepted time.
  // The geometry requires positive wet volume on every leaf at both interval
  // endpoints. Unsupported activation, stale requests, and malformed fields
  // fail without publishing any trial state. Geometry and an independent
  // graded owner are allocated before the transaction enters its stages.
  absl::StatusOr<LiquidAttemptOutcome3D>
  AttemptWithAdaptivePhysicalGeometry(double step_duration, const LiquidAdaptivePhysicalGeometryRequest3D& request);

  // Request an ownership handoff before one physical trial. The caller
  // certifies full occupancy and a carrier at the previous accepted epoch.
  absl::StatusOr<LiquidAttemptOutcome3D> AttemptWithAdaptiveBandRetirement(
      double step_duration,
      AdaptiveMacCellKey3D leaf,
      LiquidHandoffOccupancy3D occupancy,
      const LiquidHandoffCarrier3D& carrier
  );
  absl::StatusOr<LiquidAttemptOutcome3D> AttemptWithAdaptiveCoreReseed(
      double step_duration,
      AdaptiveMacCellKey3D leaf,
      LiquidHandoffOccupancy3D occupancy,
      const LiquidHandoffCarrier3D& carrier,
      std::size_t particle_count
  );
  // Attempts the moving planar piston scene and returns the pressure load
  // only when the liquid state commits. The body is supplied by value and
  // the piston trajectory stays prescribed for this attempt.
  absl::StatusOr<LiquidPlanarPistonAttempt3D>
  AttemptWithPlanarPistonPressureLoad(double step_duration, RigidBody3D piston);
  // Advances an axially guided planar piston in a source-free
  // moving-piston scene. Body center.x is the wall coordinate in meters.
  // The old axial velocity drives the liquid boundary; accepted pressure and
  // external axial force and CPU particle reaction update the next velocity.
  // The guide
  // absorbs transverse loads and torque. Force and torque are cleared.
  // The returned body is present only on acceptance.
  absl::StatusOr<LiquidPlanarPistonAttempt3D> AttemptWithDynamicPlanarPiston(double step_duration, RigidBody3D piston);
  // Uses the supplied backend's reaction update for this attempt. The
  // callback must operate on the provided trial particles and return the
  // low-x wall impulse in N s. It must remain valid until this call returns.
  absl::StatusOr<LiquidPlanarPistonAttempt3D> AttemptWithDynamicPlanarPiston(
      double step_duration,
      RigidBody3D piston,
      LiquidPistonReactionUpdate3D update_with_reaction
  );
  // Couples the guided piston velocity to the incompressible pressure
  // domain using frozen wet areas and a pressure-response probe. This explicit
  // attempt interface supports the particle-free planar scene. The coupled
  // velocity sets both wall flux and drift. Unsupported topology rejects.
  absl::StatusOr<LiquidPlanarPistonAttempt3D> AttemptWithCoupledPlanarPiston(double step_duration, RigidBody3D piston);
  // Advances an upright box guided along world Y in a stationary open tank.
  // Requires CPU, an analytic horizontal surface, zero friction and no sources.
  // The start velocity drives frozen wall flux and drift. Solved pressure and
  // external force and particle reaction update the next velocity. The guide absorbs transverse
  // force and torque. Half extents are in meters and must stay clear of walls.
  absl::StatusOr<LiquidFloatingBoxAttempt3D>
  AttemptWithGuidedFloatingBox(double step_duration, RigidBody3D body, Vec3 half_extent);
  // Advances an axis-aligned box with all three translations free. Uses the
  // same stationary tank and material restrictions as the guided attempt.
  // A rotational constraint absorbs torque. Wall flux and drift use the
  // old velocity, and solved pressure and contact update the next velocity.
  absl::StatusOr<LiquidFloatingBoxAttempt3D>
  AttemptWithTranslatingFloatingBox(double step_duration, RigidBody3D body, Vec3 half_extent);
  // Couples vertical wall flux and body inertia over one frozen pressure solve.
  // Requires the incompressible, zero-surface-tension scene.
  // The coupled velocity drives final flux, drift and particle contact.
  // The accepted contact impulse updates the next velocity after pressure.
  absl::StatusOr<LiquidFloatingBoxAttempt3D>
  AttemptWithCoupledGuidedFloatingBox(double step_duration, RigidBody3D body, Vec3 half_extent);
  // Couples pressure traction to all six rigid degrees of freedom for a posed
  // box with no particles. The start pose and twist drive the pressure solve.
  // The solved load updates the twist before drift. Requires the CPU analytic
  // stationary tank and conservative sweep clearance. Rejection publishes
  // neither liquid state nor a candidate body.
  absl::StatusOr<LiquidFloatingBoxAttempt3D>
  AttemptWithOrientedFloatingBoxPressure(double step_duration, RigidBody3D body, Vec3 body_half_extent);
  // Advances the pressure-driven oriented box and its particle contact in
  // one trial. The pressure kick sets the swept drift. Reciprocal contact
  // impulses change the accepted velocities for the next step.
  absl::StatusOr<LiquidFloatingBoxAttempt3D>
  AttemptWithOrientedFloatingBox(double step_duration, RigidBody3D body, Vec3 body_half_extent);
  // Couples the six rigid velocity components to the frozen incompressible
  // liquid pressure response. Contact impulses update the accepted twist
  // after drift. Supports the oriented CPU box without surface tension.
  absl::StatusOr<LiquidFloatingBoxAttempt3D>
  AttemptWithCoupledOrientedFloatingBox(double step_duration, RigidBody3D body, Vec3 body_half_extent);
  // Attempts one uniform sampled snow and liquid substep. Uses a frozen
  // sphere-union wall and the host snow transfer with complete support.
  // Rejects intersecting particle sweeps and publishes the snow candidate
  // only after the liquid transaction accepts. The original snow records
  // remain immutable. A vertically translating overhead platen uses the same
  // start-of-step pose for liquid geometry, pressure loads, and snow contact.
  absl::StatusOr<LiquidSnowAttempt3D> AttemptWithSnow(double step_duration, const LiquidSnowStepInputs3D& snow);
  // Finishes an explicitly stepped frame; no physical bytes/epoch are changed.
  absl::Status FinishFrame();
  // Chooses dt from velocity/source/wall/acceleration/capillary/remaining bounds;
  // accepts at most two dt halvings and one configured frame capacity replay.
  // A failure preserves the latest accepted version (or replay's frame start).
  absl::Status AdvanceFrame(double target_time, LiquidProgressObserver3D observer = {});

  // Advances liquid and snow to target_time using both materials' step bounds.
  // Copies the initial records from inputs before advancing, so they may alias
  // accepted_snow. On failure after the frame begins, accepted_snow matches
  // the latest committed liquid step. Capacity replay restores both materials
  // to the frame start before notifying the observer.
  absl::Status AdvanceFrameWithSnow(
      double target_time,
      const LiquidSnowStepInputs3D& inputs,
      std::vector<PlasticSolidCheckpointParticle3D>& accepted_snow,
      LiquidProgressObserver3D observer = {}
  );

 private:
  enum class FloatingBoxMotion {
    kVerticalGuide,
    kTranslation,
    kCoupledVerticalGuide,
    kOrientedPressure,
    kCoupledOrientedPressure
  };
  struct FloatingBoxTrial {
    RigidBody3D body;
    AxisAlignedFloatingBoxGeometry3D geometry;
    std::optional<OrientedFloatingBoxGeometry3D> oriented_geometry;
    Vec3 half_extent;
    bool coupled;
    bool vertical_guide;
    std::optional<double> added_mass_kg;
    std::optional<OrientedFloatingBoxLoads3D> oriented_loads;
  };
  absl::StatusOr<LiquidFloatingBoxAttempt3D>
  AttemptFloatingBox(double step_duration, RigidBody3D body, Vec3 half_extent, FloatingBoxMotion motion);
  absl::Status ValidateSnowStep(const LiquidSnowStepInputs3D& inputs) const;
  absl::Status AdvanceFrameImpl(
      double target_time,
      LiquidProgressObserver3D observer,
      std::optional<LiquidSnowStepInputs3D> snow_inputs,
      std::vector<PlasticSolidCheckpointParticle3D>* accepted_snow
  );
  struct SnowTrial;
  struct AdaptiveStateTrial;
  struct AdaptivePhysicalTrial;
  struct AdaptiveWidthAgeTrial;
  absl::StatusOr<LiquidAttemptOutcome3D> AttemptPreparedAdaptiveState(
      double step_duration,
      absl::FunctionRef<absl::StatusOr<LiquidState3D>()> prepare,
      std::size_t certified_replacement_count = 0
  );
  absl::StatusOr<LiquidAttemptOutcome3D> AttemptImpl(
      double step_duration,
      LiquidWorkPhaseObserver3D work_phase,
      RigidBody3D* piston_load,
      bool dynamic_piston = false,
      bool coupled_piston = false,
      const LiquidPistonReactionUpdate3D* update_with_reaction = nullptr,
      FloatingBoxTrial* floating_box = nullptr,
      SnowTrial* snow = nullptr,
      AdaptiveStateTrial* adaptive_replacement = nullptr,
      const AdaptiveWidthAgeTrial* adaptive_age = nullptr,
      AdaptivePhysicalTrial* adaptive_physical = nullptr,
      std::optional<std::uint64_t> quadratic_particle_id = std::nullopt
  );
  absl::StatusOr<LiquidPlanarPistonAttempt3D> AttemptDynamicPistonImpl(
      double step_duration,
      RigidBody3D piston,
      bool coupled,
      const LiquidPistonReactionUpdate3D* update_with_reaction = nullptr
  );
  struct Impl;
  explicit LiquidSubstepScheduler3D(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace fluidity
