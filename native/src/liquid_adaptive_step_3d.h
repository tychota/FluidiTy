#pragma once

#include <array>
#include <optional>
#include <vector>

#include "fluidity/host/adaptive_liquid_core_transport_3d.h"
#include "fluidity/host/adaptive_liquid_owner_exchange_3d.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/graded_mac_transfer_3d.h"
#include "fluidity/host/graded_quadratic_mac_transfer_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "fluidity/host/viscosity.h"
#include "graded_liquid_boundary_geometry_3d.h"
#include "graded_liquid_wet_transition_3d.h"
#include "liquid_remap_3d_internal.h"
#include "mac_pressure_3d_system.h"

namespace fluidity::liquid_step_internal {

// Assemble nonzero symmetric-gradient quadratures per leaf from local staggered
// face anchors and owned apertures. Dynamic viscosity is in Pa s. On a fully
// blocked face the prescribed Cartesian wall flux divided by its area gives
// the normal anchor velocity on the conforming path. For frozen planar
// geometry, supply the matching canonical topology and signed pressure inputs
// from the same trial. Wet volumes weight the quadratures. Wet-open centroids
// locate the unknown velocities and wet-solid centroids locate the prescribed
// Cartesian components of wall_velocity in m/s. Dry leaves have no rows.
// A moving partial aperture without frozen geometry returns Unimplemented.
// All geometric arrays use canonical topology order. An optional scratch span of leaf count + 1 stores
// row offsets in leaf order; it must be disjoint from inputs and may change on failure.
// Rank-deficient anchor support fails without rows.
absl::StatusOr<std::vector<MacStrainRow3D>> AssembleGradedViscousStrainRows3D(
    const AdaptiveMacTopology3D& topology,
    Vec3 world_origin,
    double dynamic_viscosity,
    std::span<const GradedMacFace3D> faces,
    std::span<const pressure_internal::AdaptiveMacPressureSubface3D> subfaces,
    std::span<std::size_t> leaf_row_offsets = {},
    const FrozenGradedPlanarLiquidGeometry3D* geometry = nullptr,
    Vec3 wall_velocity = {}
);

// Canonical rows and identity of one selected step-start particle.
struct PreparedAdaptiveQuadraticParticle3D {
  std::size_t particle_ordinal;
  std::uint64_t particle_id;
  Vec3 world_position_m;
  GradedQuadraticMacStencil3D stencil;
};

// Borrow a complete staged roster in stable particle order and its epoch.
// The step owner retains both these rows and the normalized old carriers.
struct PreparedAdaptiveQuadraticTransfer3D {
  static constexpr std::size_t kMaximumParticles = 8;  // Explicit resampling request only.
  std::uint64_t epoch;
  std::span<const PreparedAdaptiveQuadraticParticle3D> particles;

  std::span<const PreparedAdaptiveQuadraticParticle3D> selections() const {
    return particles;
  }
};

// Frozen graded tank ownership and staging. The owner is built at scheduler
// admission and kept immutable across attempts and checkpoint restores.
class AdaptiveLiquidStep3D {
 public:
  static absl::StatusOr<AdaptiveLiquidStep3D>
  Create(const LiquidState3D& initial, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options);

  // Borrow the immutable topology for a candidate prepared from this epoch.
  const AdaptiveMacTopology3D& topology() const {
    return topology_;
  }

  // Query the represented world position in the frozen transfer layout.
  absl::StatusOr<double> ParticleSupportWidth(Vec3 world_position_m) const {
    return transfer_.ParticleSupportWidth(world_position_m);
  }
  // Dynamic payload kept by the frozen step owner between trials.
  std::size_t reserved_bytes() const;
  absl::Status PreparePhysicalTrial(
      const LiquidState3D& accepted,
      double duration,
      const LiquidSceneInputs3D& scene,
      const LiquidAdaptivePhysicalGeometryRequest3D& request
  );
  // Stage a complete mixed deposit and its normalized old carriers. Empty
  // means unsupported eligibility and selects the ordinary radial attempt.
  absl::StatusOr<std::optional<PreparedAdaptiveQuadraticTransfer3D>> PrepareQuadraticTransfer(
      const LiquidState3D& committed,
      const LiquidStepOptions3D& options,
      std::uint64_t particle_id
  );
  // Apply the stored basis roster in a normal trial. Unsupported complete
  // support returns an empty roster so the scheduler can reset before PIC.
  absl::StatusOr<std::optional<PreparedAdaptiveQuadraticTransfer3D>>
  PrepareStoredQuadraticTransfer(const LiquidState3D& committed, const LiquidStepOptions3D& options);
  // Stage one complete certified roster against the detached candidate.
  // Any missing, ineligible or malformed selection refuses the entire trial.
  absl::StatusOr<PreparedAdaptiveQuadraticTransfer3D> PrepareCertifiedQuadraticTransfer(
      const LiquidState3D& candidate,
      const LiquidStepOptions3D& options,
      std::span<const std::uint64_t> replacement_ids
  );
  absl::Status Deposit(
      const LiquidState3D& trial,
      const LiquidStepOptions3D& options,
      const PreparedAdaptiveQuadraticTransfer3D* quadratic = nullptr,
      bool blend_band_overlap = false
  );
  // Predict and project the frozen graph. Optional swell samples eligible
  // canonical subfaces at the trial end and leaves old carriers intact.
  absl::Status Project(
      LiquidState3D& trial,
      double duration,
      const LiquidSceneInputs3D& scene,
      const LiquidStepOptions3D& options,
      const LiquidSubstepBackend3D& backend,
      const ResolvedOceanBoundary3D* ocean_boundary,
      MacPressureWorkspace3D& workspace,
      LiquidAcceptedStepMeasurement3D& measurement
  );
  absl::Status CheckReseedSupport(
      const LiquidParticleState3D& seed,
      std::span<const LiquidAdaptiveFaceState3D> donors,
      const LiquidHandoffCarrier3D* accepted_affine = nullptr
  ) const {
    return transfer_.CheckCompleteRadialSupport(seed, donors, accepted_affine);
  }
  absl::Status CheckOldReseedSupport(const LiquidParticleState3D& seed) const {
    return CheckReseedSupport(seed, old_faces_);
  }
  absl::Status Gather(const LiquidState3D& trial, const PreparedAdaptiveQuadraticTransfer3D* quadratic = nullptr);
  absl::Status Update(
      LiquidState3D& trial,
      const LiquidState3D& accepted,
      double duration,
      const LiquidSceneInputs3D& scene,
      const LiquidStepOptions3D& options,
      const LiquidSubstepBackend3D& backend,
      bool allow_owner_exchange
  );

 private:
  AdaptiveLiquidStep3D(AdaptiveMacTopology3D topology, GradedMacTransfer3D transfer);
  absl::StatusOr<GradedLiquidWetTransition3D> CertifyWetTransition(
      const LiquidState3D& accepted,
      const GradedLiquidWetTransitionRequest3D& request,
      const LiquidSceneInputs3D& scene
  ) const;
  absl::StatusOr<std::optional<PreparedAdaptiveQuadraticTransfer3D>> StageQuadraticTransfer(
      const LiquidState3D& state,
      const LiquidStepOptions3D& options,
      std::span<const std::uint64_t> particle_ids,
      bool allow_radial_fallback,
      bool include_stored_basis
  );
  absl::Status CertifyExchangePaths(const LiquidState3D& trial, double duration, const LiquidSceneInputs3D& scene);
  absl::Status CheckExchangePaths(std::span<const LiquidParticleState3D> particles) const;
  absl::Status NormalizeDeposits(
      const LiquidState3D& state,
      const LiquidStepOptions3D& options,
      std::span<const MacFaceDeposit3D> deposits
  );
  absl::Status AccumulateInventories(const LiquidState3D& state);
  absl::Status AdvectPressureHistory(const LiquidState3D& state, double duration);

  struct PressureHistory3D {
    Vec3 centered_velocity{};
    double donor_weight = 0.0;
    double donor_pressure = 0.0;
  };

  struct PhysicalTrial3D {
    LiquidClock3D accepted_clock;
    LiquidObstacle3D start_wall;
    LiquidObstacle3D end_wall;
    std::vector<std::uint8_t> wet_open_faces;
    std::vector<double> end_fluid_volumes_m3;
    // Stored only when certified air leaves require a sparse pressure graph.
    std::optional<FrozenGradedPlanarLiquidGeometry3D> sparse_geometry;
    std::optional<GradedLiquidWetTransitionRequest3D> wet_transition;
    std::optional<FrozenGradedPlanarLiquidGeometry3D> sampled_end_geometry;
    std::span<const GradedLiquidCornerSample3D> sampled_end_corners;
  };

  AdaptiveMacTopology3D topology_;
  std::optional<PhysicalTrial3D> physical_trial_;
  GradedMacTransfer3D transfer_;
  std::vector<std::size_t> interior_slots_;
  std::vector<AdaptiveMacExteriorFace3D> exterior_faces_;
  // Nonempty only when an Eulerian core owns material in this frozen trial.
  std::vector<std::uint8_t> core_faces_;
  std::vector<double> core_fluxes_;
  std::optional<AdaptiveLiquidCoreTransportWorkspace3D> core_transport_;
  struct ExchangeParticlePath {
    Vec3 endpoint;
    std::uint64_t id;
    double mass;
    double reference_volume;
    double volume_ratio;
  };
  std::vector<AdaptiveLiquidOwnerExchangeFace3D> exchange_samples_;
  std::vector<ExchangeParticlePath> exchange_paths_;
  std::vector<pressure_internal::AdaptiveMacPressureSubface3D> subfaces_;
  std::vector<MacPressureCell3D> pressure_cells_;
  std::vector<MacStrainRow3D> strain_rows_;
  // Geometry at unit viscosity, kept only for composition-dependent quadrature.
  std::vector<MacStrainRow3D> unit_strain_rows_;
  std::vector<std::size_t> leaf_strain_row_offsets_;
  std::vector<LiquidAdaptiveFaceState3D> old_faces_;
  // Attempt-local PIC flags. The accepted carrier bank keeps the valid core carrier.
  std::vector<std::size_t> band_fallback_slots_;
  std::vector<LiquidAdaptiveFaceState3D> band_projected_history_;
  std::vector<PreparedAdaptiveQuadraticParticle3D> quadratic_particles_;
  std::vector<MacFaceDeposit3D> quadratic_deposits_;
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
