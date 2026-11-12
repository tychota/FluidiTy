#pragma once

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "fluidity/host/dry_solid_step_3d.h"
#include "fluidity/host/transfers_3d.h"

namespace fluidity::internal {

using DrySolidDeposit3D = absl::FunctionRef<
    absl::Status(const SparseMacGrid3D&, std::span<const MacTransferParticle3D>, std::span<MacFaceDeposit3D>)>;
using DrySolidGather3D = absl::FunctionRef<
    absl::
        Status(const SparseMacGrid3D&, std::span<const MacTransferParticle3D>, std::span<const double>, std::span<const double>, std::span<MacParticleGatherResult3D>)>;

struct AcceptedDrySolid3D {
  SolidDeformationState3D state;
  double wave_speed_squared;
};

// Checks accepted elastic and plastic state and computes stress and acoustic bound.
absl::StatusOr<AcceptedDrySolid3D>
EvaluateAcceptedDrySolid3D(const DrySolidParticle3D& particle, const DrySolidMaterial3D& material);

// Accumulates one particle's uncut reference-volume stress into private step scratch.
// The caller provides one initialized force slot per owned face. On failure discard scratch.
absl::Status AccumulateDrySolidStress3D(
    const SparseMacGrid3D& grid,
    Vec3 world_position,
    double reference_volume,
    Mat3 kirchhoff_stress,
    std::span<double> face_force
);

// Shares the dry step and its transactional publication across transfer backends.
absl::Status AdvanceDrySolidWithTransfers3D(
    const SparseMacGrid3D& grid,
    const DrySolidMaterial3D& material,
    const DrySolidStep3D& settings,
    std::span<DrySolidParticle3D> particles,
    DrySolidDeposit3D deposit,
    DrySolidGather3D gather,
    DrySolidStepBudget3D* budget
);

}  // namespace fluidity::internal
