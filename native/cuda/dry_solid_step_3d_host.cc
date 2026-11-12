#include <utility>
#include <vector>

#include "../src/dry_solid_step_3d_internal.h"
#include "fluidity/host/cuda_dry_solid_step_3d.h"
#include "fluidity/host/cuda_liquid_boundary_3d.h"
#include "fluidity/host/cuda_transfers_3d.h"

namespace fluidity {

absl::Status CudaAdvanceDrySolid3D(
    const SparseMacGrid3D& grid,
    const DrySolidMaterial3D& material,
    const DrySolidStep3D& settings,
    MacDepositQuantization3D quantization,
    std::span<DrySolidParticle3D> particles,
    DrySolidStepBudget3D* budget
) {
  const auto deposit = [quantization](
                           const SparseMacGrid3D& layout,
                           std::span<const MacTransferParticle3D> points,
                           std::span<MacFaceDeposit3D> faces
                       ) {
    std::vector<MacFaceDepositQuanta3D> quanta(faces.size());
    CudaOperatorMeasurement measurement{};
    const auto status = CudaDepositMacQuanta3D(layout, points, quantization, quanta, measurement);
    if (!status.ok()) {
      return status;
    }
    for (std::size_t slot = 0; slot < faces.size(); ++slot) {
      faces[slot] = {
          .deposited_mass = static_cast<double>(quanta[slot].mass_quanta_sum) * quantization.mass_per_quantum,
          .deposited_pure_momentum =
              static_cast<double>(quanta[slot].pure_momentum_quanta_sum) * quantization.pure_momentum_per_quantum,
      };
    }
    return absl::OkStatus();
  };
  const auto gather = [](const SparseMacGrid3D& layout,
                         std::span<const MacTransferParticle3D> points,
                         std::span<const double> old_velocities,
                         std::span<const double> new_velocities,
                         std::span<MacParticleGatherResult3D> results) {
    CudaOperatorMeasurement measurement{};
    return CudaGatherMac3D(layout, points, old_velocities, new_velocities, results, measurement);
  };
  const auto floor_deposit = [](const LiquidBoundaryDepositInputs3D& inputs
                             ) -> absl::StatusOr<LiquidBoundaryFaceDeposits3D> {
    auto deposited = CudaDepositCompatibleMacQuanta3D(inputs);
    if (!deposited.ok()) {
      return deposited.status();
    }
    return std::move(deposited->deposits);
  };
  const auto floor_gather = [](const LiquidBoundaryGatherInputs3D& inputs
                            ) -> absl::StatusOr<LiquidBoundaryGatherResults3D> {
    auto gathered = CudaGatherLiquidBoundaryParticles3D(inputs);
    if (!gathered.ok()) {
      return gathered.status();
    }
    return std::move(gathered->gathered);
  };
  const internal::DrySolidFloorTransfers3D floor_transfers{quantization, floor_deposit, floor_gather};
  return internal::AdvanceDrySolidWithTransfers3D(
      grid,
      material,
      settings,
      particles,
      deposit,
      gather,
      budget,
      &floor_transfers
  );
}

}  // namespace fluidity
