#pragma once

#include <vector>

#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/graded_mac_transfer_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "fluidity/host/viscosity.h"
#include "liquid_remap_3d_internal.h"
#include "mac_pressure_3d_system.h"

namespace fluidity::liquid_step_internal {

// Assemble nonzero symmetric-gradient quadratures per leaf from local staggered
// face anchors and owned apertures. Dynamic viscosity is in Pa s. Stationary sealed faces are lifted as zero velocity.
// All geometric arrays use canonical topology order. An optional scratch span of leaf count + 1 stores
// row offsets in leaf order; it must be disjoint from inputs and may change on failure.
// Rank-deficient anchor support fails without rows.
absl::StatusOr<std::vector<MacStrainRow3D>> AssembleGradedViscousStrainRows3D(
    const AdaptiveMacTopology3D& topology,
    Vec3 world_origin,
    double dynamic_viscosity,
    std::span<const GradedMacFace3D> faces,
    std::span<const pressure_internal::AdaptiveMacPressureSubface3D> subfaces,
    std::span<std::size_t> leaf_row_offsets = {}
);

// Frozen graded tank ownership and staging. The owner is built at scheduler
// admission and kept immutable across attempts and checkpoint restores.
class AdaptiveLiquidStep3D {
 public:
  static absl::StatusOr<AdaptiveLiquidStep3D>
  Create(const LiquidState3D& initial, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options);

  // Dynamic payload kept by the frozen step owner between trials.
  std::size_t reserved_bytes() const;
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
  absl::Status AdvectPressureHistory(const LiquidState3D& state, double duration);

  struct PressureHistory3D {
    Vec3 centered_velocity{};
    double donor_weight = 0.0;
    double donor_pressure = 0.0;
  };

  AdaptiveMacTopology3D topology_;
  GradedMacTransfer3D transfer_;
  std::vector<std::size_t> interior_slots_;
  std::vector<AdaptiveMacExteriorFace3D> exterior_faces_;
  // Nonempty only when an Eulerian core owns material in this frozen trial.
  std::vector<std::uint8_t> core_faces_;
  std::vector<double> core_fluxes_;
  std::vector<pressure_internal::AdaptiveMacPressureSubface3D> subfaces_;
  std::vector<MacPressureCell3D> pressure_cells_;
  std::vector<MacStrainRow3D> strain_rows_;
  // Geometry at unit viscosity, kept only for composition-dependent quadrature.
  std::vector<MacStrainRow3D> unit_strain_rows_;
  std::vector<std::size_t> leaf_strain_row_offsets_;
  std::vector<LiquidAdaptiveFaceState3D> old_faces_;
  std::vector<LiquidBoundaryParticleGather3D> gathered_;
  std::vector<double> distances_;
  std::vector<double> predictor_density_;
  std::vector<double> reference_density_;
  std::vector<liquid_remap_internal::ConstituentSums> constituent_sums_;
  std::vector<Vec3> normal_derivatives_;
  std::vector<PressureHistory3D> pressure_history_;
  bool update_volume_ratio_ = false;
  std::vector<LiquidRemapInventory3D> inventories_;
  double surface_height_ = 0;
};

}  // namespace fluidity::liquid_step_internal
