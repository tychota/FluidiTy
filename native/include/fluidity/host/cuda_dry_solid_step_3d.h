#pragma once

#include "fluidity/base/mac_transfer_types_3d.h"
#include "fluidity/host/dry_solid_step_3d.h"

namespace fluidity {

// Advances the same bounded dry step as AdvanceDrySolid3D, using device 0 for quantized
// mass/momentum deposition and velocity gathering. Stress, constitutive state, wall sweeps
// and acceptance run on the host. Quantization must satisfy CudaDepositMacQuanta3D's
// scale and capacity rules. A backend or step failure preserves all particles and the optional budget. Budget uses the
// native units and disjoint-view rule.
absl::Status CudaAdvanceDrySolid3D(
    const SparseMacGrid3D& grid,
    const DrySolidMaterial3D& material,
    const DrySolidStep3D& settings,
    MacDepositQuantization3D quantization,
    std::span<DrySolidParticle3D> particles,
    DrySolidStepBudget3D* budget = nullptr
);

}  // namespace fluidity
