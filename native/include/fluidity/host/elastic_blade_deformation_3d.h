#pragma once

#include "fluidity/host/liquid_boundary_transfer_3d.h"
#include "fluidity/host/liquid_wall_contact_3d.h"

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

}  // namespace fluidity
