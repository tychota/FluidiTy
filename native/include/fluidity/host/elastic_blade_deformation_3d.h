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

}  // namespace fluidity
