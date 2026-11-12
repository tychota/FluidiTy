#pragma once

#include "fluidity/host/liquid_boundary_transfer_3d.h"
#include "fluidity/host/solid_interaction_checkpoint_3d.h"

namespace fluidity {

struct ElasticBladeDeformation3D {
  LiquidBoundaryParticleGather3D gather;
  Mat3 deformation_gradient;  // Dimensionless candidate at the new time.
};

// Reconstructs two histories on the assigned blade side and applies
// F_new = (I + (new_time - old_time) L_new) F_old. Histories use owned-slot
// order and tags for the same blade at their respective sample times.
// Elastic deformation requires full-rank continuation and a resolved
// positive determinant. Malformed inputs return InvalidArgument, missing
// support, fallback or collapse returns FailedPrecondition, and range or
// capacity failures propagate from the gather. Inputs remain unchanged.
absl::StatusOr<ElasticBladeDeformation3D> GatherElasticBladeDeformation3D(
    const SparseMacGrid3D& mac_grid,
    const MacTransferParticle3D& particle,
    ElasticBladeSide3D blade_side,
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& new_history,
    Mat3 old_deformation_gradient
);

// Accepts an elastic particle batch from old and new owned MAC histories.
// The slab starts at the old time. Every elastic particle must have a
// matching committed side for this blade, in strictly increasing ID order.
// Retention is in [0, 1]; advection always uses PIC. Returns one reaction
// per particle in the same order. Failure preserves the entire checkpoint.
absl::StatusOr<std::vector<ElasticBladeContact3D>> AdvanceElasticBladeParticles3D(
    const SparseMacGrid3D& mac_grid,
    const TranslatingSlab3D& blade,
    std::uint64_t blade_id,
    double retained_velocity_fraction,
    double friction_coefficient,
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& new_history,
    SolidInteractionCheckpoint3D& checkpoint
);

struct ElasticBladeGridStep3D {
  TranslatingSlab3D blade;  // Pose at the start of this attempt.
  std::uint64_t blade_id;
  double duration;  // Seconds.
  double shear_modulus;  // Pascals.
  double first_lame_parameter;  // Pascals.
  MacDepositQuantization3D quantization;
  double minimum_face_mass = 0.0;  // Kilograms.
  double retained_velocity_fraction = 0.0;
  double friction_coefficient = 0.0;
};

struct ElasticBladeGridStepResult3D {
  std::vector<ElasticBladeContact3D> contacts;  // Particle order.
  double grid_stress_work = 0.0;  // Applied face force against new velocity, joules.
  double deformation_work = 0.0;  // dt times V0 stress:L, joules.
  double cut_work = 0.0;  // dt times V0 stress:(L - L'), joules.
  double omitted_stress_work = 0.0;  // Work of omitted force on completed samples, joules.
};

// Advances dry elastic particles on the fixed owned grid for a prescribed
// translating slab. Deposits compatible mass and momentum, adds the explicit
// stress force, and completes old/new same-side histories for particle
// acceptance. Faces changing side or entering the slab carry no new-time
// donor mass. The caller controls the explicit time step. Contact results
// are in particle order and failure preserves the entire checkpoint.
absl::StatusOr<ElasticBladeGridStepResult3D> StepElasticBladeGrid3D(
    const SparseMacGrid3D& mac_grid,
    const ElasticBladeGridStep3D& settings,
    SolidInteractionCheckpoint3D& checkpoint
);

}  // namespace fluidity
