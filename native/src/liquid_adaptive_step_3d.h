#pragma once

#include <vector>

#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/graded_mac_transfer_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "mac_pressure_3d_system.h"

namespace fluidity::liquid_step_internal {

// Frozen graded tank ownership and staging. The owner is built at scheduler
// admission and kept immutable across attempts and checkpoint restores.
class AdaptiveLiquidStep3D {
 public:
  static absl::StatusOr<AdaptiveLiquidStep3D>
  Create(const LiquidState3D& initial, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options);

  absl::Status Deposit(const LiquidState3D& trial, const LiquidStepOptions3D& options);
  absl::Status Project(
      LiquidState3D& trial,
      double duration,
      const LiquidSceneInputs3D& scene,
      const LiquidStepOptions3D& options,
      const LiquidSubstepBackend3D& backend,
      MacPressureWorkspace3D& workspace,
      LiquidAcceptedStepMeasurement3D& measurement
  );
  absl::Status Gather(const LiquidState3D& trial);
  absl::Status Update(
      LiquidState3D& trial,
      double duration,
      const LiquidSceneInputs3D& scene,
      const LiquidStepOptions3D& options,
      const LiquidSubstepBackend3D& backend
  );

 private:
  AdaptiveLiquidStep3D(AdaptiveMacTopology3D topology, GradedMacTransfer3D transfer);
  absl::Status AccumulateInventories(const LiquidState3D& state);

  AdaptiveMacTopology3D topology_;
  GradedMacTransfer3D transfer_;
  std::vector<std::size_t> interior_slots_;
  std::vector<pressure_internal::AdaptiveMacPressureSubface3D> subfaces_;
  std::vector<MacPressureCell3D> pressure_cells_;
  std::vector<LiquidAdaptiveFaceState3D> old_faces_;
  std::vector<LiquidBoundaryParticleGather3D> gathered_;
  std::vector<double> distances_;
  std::vector<LiquidRemapInventory3D> inventories_;
  double surface_height_ = 0;
};

}  // namespace fluidity::liquid_step_internal
