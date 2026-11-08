#pragma once

#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity {
struct LiquidParticleUpdateInputs3D {
  std::span<const LiquidBoundaryParticleGather3D> gathered;  // Frozen old/new gather, particle order.
  std::span<const double> liquid_distances;  // Frozen x^n field, meters, particle order.
  LiquidObstacle3D obstacle;  // Pose at substep start, constant translation during step_duration.
  double cell_width;  // Meters.
  double step_duration;  // Positive seconds.
  double particle_energy_blend;  // e in [0,1], alpha=0.99e and surface beta=e.
  double friction_coefficient;  // Nonnegative dimensionless Coulomb coefficient.
  double penetration_tolerance;  // Meters.
};
struct LiquidMissingFacePredictionInputs3D {
  SparseMacGridView3D grid;
  LiquidBoundaryVelocityHistory3D old_history;  // Unmodified pre-force normalization/validity.
  std::size_t face_slot;
  std::int64_t side_tag;
  Vec3 world_acceleration;  // m/s^2, applied once to the continued old velocity.
  double step_duration;  // Seconds.
};
// Typed checked backend operations. Each output borrows its supplied workspace or
// writes only the supplied trial spans. Inputs use the corresponding bounded CPU
// contract; every operation must return code-only errors and allocate/free nothing.
// Context may own device reservations but remains borrowed. A custom backend must
// fill EVERY slot; Create never substitutes CPU numerics. Topology is authoritative:
// a device stable-sort backend must publish its checked ordering through a checked
// workspace import seam; the current CPU operation uses PrepareTrialTopology.
// Task 6 must add that import seam rather than independently re-sort a device result.
// Shared orchestration currently enumerates/samples finite scene fields on the host.
// Device field evaluation needs a separate checked sampler seam in Task 6; this table
// does not claim that host sampling qualifies as CUDA numerical work.
struct LiquidSubstepBackend3D {
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
  using Gather = absl::StatusOr<LiquidBoundaryGatherView3D>(
      void* context,
      const LiquidBoundaryGatherViewInputs3D& inputs,
      LiquidBoundaryWorkspace3D& workspace
  );
  using UpdateParticles = absl::Status(
      void* context,
      const LiquidParticleUpdateInputs3D& inputs,
      std::span<LiquidParticleState3D> trial_particles
  );
  void* context = nullptr;
  PrepareTopology* topology = nullptr;
  Deposit* deposit = nullptr;
  Normalize* normalize = nullptr;
  PredictMissingFace* predict_missing_face = nullptr;
  ApplyForces* forces = nullptr;
  Freeze* freeze = nullptr;
  BuildLinks* links = nullptr;
  Project* project = nullptr;
  Gather* gather = nullptr;
  UpdateParticles* update_particles = nullptr;
};
// CPU operations share the accepted bounded operators and true-gradient J/contact.
LiquidSubstepBackend3D CpuLiquidSubstepBackend3D();
}  // namespace fluidity
