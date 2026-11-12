#pragma once

#include "fluidity/base/liquid_regular_cell_transport_3d.h"
#include "fluidity/host/liquid_substep_3d.h"
#include "fluidity/host/viscosity.h"
#include "fluidity/host/wetted_piston_3d.h"

namespace fluidity {
// Maximum queries in one substep field dispatch. Staging may use smaller
// chunks when the selected arena has fewer reserved sample slots.
inline constexpr std::size_t kLiquidFieldBatchCapacity3D = 256;
// The device arena reserves this many sample slots per unit of its largest
// particle, face, or cell capacity.
inline constexpr std::size_t kLiquidFieldSampleSlotsPerCapacity3D = 8;

struct LiquidParticleUpdateInputs3D {
  std::span<const LiquidBoundaryParticleGather3D> gathered;  // Frozen old/new gather, particle order.
  std::span<const double> liquid_distances;  // Frozen x^n field, meters, particle order.
  LiquidObstacle3D obstacle;  // Pose at substep start, constant translation during step_duration.
  double cell_width;  // Meters.
  double step_duration;  // Positive seconds.
  double particle_energy_blend;  // e in [0,1], alpha=0.99e and surface beta=e.
  double friction_coefficient;  // Nonnegative dimensionless Coulomb coefficient.
  double penetration_tolerance;  // Meters.
  // Empty or one sample per particle in the same order. Active samples select
  // paired advection and J. No input span may alias output particles or backend scratch.
  std::span<const LiquidRegularCellTransport3D> regular_cell_transport{};
};
struct LiquidMissingFacePredictionInputs3D {
  SparseMacGridView3D grid;
  LiquidBoundaryVelocityHistory3D old_history;  // Unmodified pre-force normalization/validity.
  std::size_t face_slot;
  std::int64_t side_tag;
  Vec3 world_acceleration;  // m/s^2, applied once to the continued old velocity.
  double step_duration;  // Seconds.
};
// Typed checked backend operations. Numeric outputs borrow their supplied
// workspace or write only trial spans. Attempt operations return code-only
// errors and allocate/free nothing, including on failure. Initialize and growth
// are setup operations outside attempts and may reserve/release resources.
// Context is borrowed and must outlive a single scheduler; the backend and
// scheduler are single-threaded and non-reentrant. Create requires every
// operation used by the scene and never substitutes CPU callbacks.
// Device sorting must publish through PrepareTrialTopologyFromOrder; host
// orchestration enumerates source/topology/sample coordinates without replacing
// the selected backend's distance, wall, transfer or projection numerics.
struct LiquidSubstepBackend3D {
  // Evaluate the frozen t^n field at one particle world position in meters.
  // Particle-union evaluation
  // visits trial_particles in input order. Return only a finite signed distance
  // (negative inside); invalid/range errors leave caller outputs untouched.
  using SampleField = absl::StatusOr<double>(
      void* context,
      const LiquidSceneField3D& field,
      std::span<const LiquidParticleState3D> trial_particles,
      Vec3 sample_world_position
  );
  // Evaluate a bounded batch of the frozen field, in world meters. Preserve
  // query and particle order. Output spans must match the query extent and
  // must not alias inputs or backend scratch. Empty batches perform no sampling.
  // Bad input returns InvalidArgument, insufficient storage ResourceExhausted,
  // nonfinite distances OutOfRange and device runtime failures Internal. Outputs
  // remain untouched on failure.
  using SampleFieldBatch = absl::Status(
      void* context,
      const LiquidSceneField3D& field,
      std::span<const LiquidParticleState3D> trial_particles,
      std::span<const Vec3> world_sample_positions,
      std::span<double> signed_distances
  );
  // Apply the same batch contract to a plane using original signed lattice y
  // in cell widths. Centers use half-integers and corners use integers. The
  // compensated distances are in world meters; an uncertifiable distance
  // returns OutOfRange.
  using SampleLatticePlane = absl::Status(
      void* context,
      const LiquidHorizontalPlaneSurface3D& plane,
      UniformMacGeometry3D geometry,
      std::span<const double> sample_lattice_ordinates_y,
      std::span<double> signed_distances
  );
  // At the supplied frozen obstacle pose, fill full owned-slot face tags and/or
  // prescribed walls. Each output is either empty (not requested) or covers the
  // entire grid. Thin-slab walls use requested_side_tag (+1/-1); tags determine
  // side from each face anchor. Tank faces use side zero, strict interior tags
  // and the active constraint's normal and velocity at the staggered lattice anchor.
  // Wall values depend only on the supplied grid, obstacle pose and side. The
  // caller owns both output spans, which must not overlap inputs or one another.
  // Gather and scalar field sampling leave wall samples intact until the caller
  // overwrites them. Failed outputs remain trial-only and are discarded by the scheduler.
  using SampleFaces = absl::Status(
      void* context,
      SparseMacGridView3D grid,
      const LiquidObstacle3D& obstacle_at_sample_time,
      std::int64_t requested_side_tag,
      std::span<LiquidBoundaryFaceTag3D> face_tags,
      std::span<LiquidPrescribedWallSample3D> wall_samples
  );
  // Prepare an independent initial device version. Failure must retain any live
  // prior backend state; all pending resources remain owned by the backend.
  using Initialize = absl::Status(void* context, const LiquidState3D& initial);
  // Stage every trial family with accepted_clock (trial.clock is still t^n).
  // Called only after physical validation; may fail, preserving the committed
  // device version. Host FinishAttempt can still reject. Commit must be an
  // infallible metadata flip, with no device calls or resource destruction.
  // Discard is also infallible and allocation-free; it discards candidate control
  // only and is safe even if this attempt never called PreparePublication.
  using PreparePublication = absl::Status(void* context, const LiquidState3D& trial, LiquidClock3D accepted_clock);
  using CommitPublication = void(void* context) noexcept;
  using DiscardPublication = void(void* context) noexcept;
  // Reserve replacement storage and stage frame_start before the host replaces
  // any live version. Enforce the device ceiling against peak old+new ownership.
  // Failure and host growth failure both call DiscardGrowth; success calls
  // CommitGrowth after the host replacement. These hooks execute outside an
  // attempt. Retire old owners only after all source leases are fenced; commit
  // must not fail. Neither path may invalidate a still-live source lease.
  using PrepareGrowth =
      absl::Status(void* context, LiquidStateCapacities3D capacities, const LiquidState3D& frame_start);
  using CommitGrowth = void(void* context) noexcept;
  using DiscardGrowth = void(void* context) noexcept;
  using PrepareTopology = absl::Status(
      void* context,
      std::span<const LiquidParticleState3D> particles,
      std::span<const CellIndex3D> pressure_candidates,
      const LiquidState3D& committed,
      LiquidStepWorkspace3D& workspace
  );
  using Deposit = absl::StatusOr<LiquidBoundaryFaceDepositsView3D>(
      void* context,
      const LiquidBoundaryDepositViewInputs3D& inputs,
      LiquidBoundaryWorkspace3D& workspace
  );
  using Normalize = absl::StatusOr<NormalizedLiquidMacFacesView3D>(
      void* context,
      std::span<const MacFaceDepositQuanta3D> face_quanta,
      MacDepositQuantization3D quantization,
      double minimum_face_mass,
      LiquidBoundaryWorkspace3D& workspace
  );
  using PredictMissingFace = absl::StatusOr<double>(
      void* context,
      const LiquidMissingFacePredictionInputs3D& inputs,
      LiquidBoundaryWorkspace3D& workspace
  );
  using ApplyForces = absl::Status(
      void* context,
      SparseMacGridView3D grid,
      std::span<const double> old_face_velocities,
      Vec3 world_acceleration,
      double step_duration,
      std::span<double> predicted_face_velocities
  );
  using Freeze = absl::StatusOr<FrozenLiquidBoundaryGeometryView3D>(
      void* context,
      const LiquidBoundaryGeometryViewInputs3D& inputs,
      LiquidBoundaryWorkspace3D& workspace
  );
  // Evaluate the actual frozen field at the moving low-x piston. The
  // supplied workspace owns the successful borrow through link assembly.
  using WettedPiston = absl::StatusOr<WettedPistonView3D>(
      void* context,
      const WettedPistonInputs3D& inputs,
      WettedPistonWorkspace3D& workspace
  );
  using BuildLinks = absl::StatusOr<LiquidPressureProblemView3D>(
      void* context,
      const FrozenLiquidBoundaryGeometryView3D& geometry,
      const LiquidPressureFields3D& fields,
      LiquidBoundaryWorkspace3D& workspace
  );
  using Project = absl::StatusOr<MacPressureProjectionView3D>(
      void* context,
      SparseMacGridView3D grid,
      double step_duration,
      const LiquidPressureProblemView3D& problem,
      MacPressureSolverOptions3D options,
      MacPressureWorkspace3D& workspace
  );
  // Optional coupled projection for supported strain geometry. On failure,
  // the scheduler rejects the attempt without publishing state.
  using ProjectViscous = absl::StatusOr<MacViscousPressureProjection3D>(
      void* context,
      const LiquidBoundaryGeometryViewInputs3D& geometry,
      const LiquidPressureProblemView3D& problem,
      double dynamic_viscosity,
      std::span<const double> cell_dynamic_viscosity,
      MacPressureSolverOptions3D options,
      MacPressureWorkspace3D& workspace
  );
  using Gather = absl::StatusOr<LiquidBoundaryGatherView3D>(
      void* context,
      const LiquidBoundaryGatherViewInputs3D& inputs,
      LiquidBoundaryWorkspace3D& workspace
  );
  // Apply swept contact and update position, velocity, J and APIC history.
  // With transport samples, malformed data returns InvalidArgument, nonfinite
  // transport arithmetic returns OutOfRange, and an unrepresentable J or
  // excessive displacement returns Aborted. Failed sampled updates publish no particles.
  using UpdateParticles = absl::Status(
      void* context,
      const LiquidParticleUpdateInputs3D& inputs,
      std::span<LiquidParticleState3D> trial_particles
  );
  void* context = nullptr;
  SampleField* sample_field = nullptr;
  SampleFaces* sample_faces = nullptr;
  Initialize* initialize = nullptr;
  PreparePublication* prepare_publication = nullptr;
  CommitPublication* commit_publication = nullptr;
  DiscardPublication* discard_publication = nullptr;
  PrepareGrowth* prepare_growth = nullptr;
  CommitGrowth* commit_growth = nullptr;
  DiscardGrowth* discard_growth = nullptr;
  PrepareTopology* topology = nullptr;
  Deposit* deposit = nullptr;
  Normalize* normalize = nullptr;
  PredictMissingFace* predict_missing_face = nullptr;
  ApplyForces* forces = nullptr;
  Freeze* freeze = nullptr;
  WettedPiston* wetted_piston = nullptr;  // Required for a translating low-x piston.
  BuildLinks* links = nullptr;
  Project* project = nullptr;
  ProjectViscous* project_viscous = nullptr;
  Gather* gather = nullptr;
  UpdateParticles* update_particles = nullptr;
  SampleFieldBatch* sample_field_batch = nullptr;
  SampleLatticePlane* sample_lattice_plane = nullptr;  // Required for horizontal-plane scenes.
  // Optional allocation-free diagnostic boundary, with kCount outside attempts.
  void (*set_work_phase)(void*, LiquidWorkPhase3D) = nullptr;
  // Optional speculative gather for two or more consecutive particles sharing
  // wall side and both velocity histories. Return one record per input in order
  // with a borrow valid across scalar and batched field samples. Failed runs leave
  // the bank intact. Internal is fatal; other failures allow ordered scalar replay.
  Gather* gather_run = nullptr;
  // Optional field batches for successfully grouped particles, in their
  // original order and bounded by kLiquidFieldBatchCapacity3D. Internal is
  // fatal; other failures allow scalar replay without publishing partial data.
  SampleFieldBatch* sample_gathered_fields = nullptr;
};
// CPU operations share the bounded operators and the swept-contact particle update.
LiquidSubstepBackend3D CpuLiquidSubstepBackend3D();
}  // namespace fluidity
