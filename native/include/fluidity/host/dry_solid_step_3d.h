#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <variant>

#include "absl/status/status.h"
#include "fluidity/base/solid_deformation_3d.h"
#include "fluidity/host/sparse_mac_grid_3d.h"

namespace fluidity {

using DrySolidMaterial3D = std::variant<GranularParameters3D, SnowParameters3D>;

struct DrySolidParticle3D {
  Vec3 world_position;  // Meters.
  Vec3 world_velocity{};  // Meters per second.
  Mat3 velocity_slopes{};  // APIC fit in inverse seconds.
  double mass;  // Constant mass in kg.
  double reference_volume;  // Constant reference volume in m³.
  Mat3 elastic_deformation;  // Previously accepted elastic map or an admissible initial map.
  double log_plastic_jacobian = 0;
};

struct DrySolidPlaten3D {
  double start_y;  // Height in meters above the fixed floor at y=0.
  double velocity_y;  // Constant world Y velocity in m/s.
};

struct DrySolidStep3D {
  double duration;  // Positive seconds.
  double gravity_y;  // Signed world acceleration in m/s².
  double floor_friction;  // Nonnegative Coulomb coefficient.
  double acoustic_cfl;  // Discrete acoustic safety factor in (0, 1].
  std::optional<DrySolidPlaten3D> platen;
  double particle_energy_blend = 0;  // [0, 1], applied to stored velocity only.
  bool compatible_floor = false;  // Granular, stationary floor, PIC storage only.
};

struct DrySolidStepBudget3D {
  double proposed_stress_impulse_l1 = 0;  // Sum of absolute face impulses before mass regularization, N s.
  double omitted_stress_impulse_l1 = 0;  // Sum of absolute face impulses removed by regularization, N s.
  double applied_stress_work = 0;  // Applied stress impulse paired with the accepted face field, J.
  double cut_stress_work = 0;  // Complete particle stress work plus unregularized compatible grid stress work, J.
  double omitted_stress_work = 0;  // Omitted stress impulse paired with that same field, J.
  double grid_wall_impulse_y = 0;  // Signed impulse from normal grid constraints on solid, N s.
  Vec3 wall_impulse_on_particles{};  // World impulse from the stored-velocity sweep on solid, N s.
  std::size_t particle_wall_contacts = 0;  // Count of swept particle contacts.
  std::size_t granular_projected_particles = 0;  // Granular trials changed by the return above relative roundoff.
  std::size_t granular_plastic_volume_changes = 0;  // Granular returns changing log plastic volume above roundoff.
  double particle_wall_kinetic_energy_change = 0;  // Signed translational kinetic energy change across the sweep, J.
};

// Advances a single dry body on a fixed uniform MAC layout with complete quadratic supports.
// The default uses APIC deposits, PIC advection, uncut stress quadrature, and a stationary floor with an optional
// platen. When enabled, the stored velocity retains 0.99 times the blend times the incoming particle residual against
// the pre-force grid. The supplied particle state seeds that history. Low-mass support uses PIC storage.
// The retained correction has zero net momentum over the eligible particles of this dry body.
// Snow grid normal constraints use face anchors. A continuous halfspace sweep constrains particle transport.
// Granular floor impacts without a platen split the deformation update at impact. The post-impact velocity
// gradient uses the active Coulomb response derivative.
// The force-acceleration denominator is floored at 1e-4 times the smallest particle mass for low-mass faces.
// The compatible floor uses normalized face weights for deposit and PIC,
// and a complete same-side ghost gather for deformation and affine feedback.
// Below-floor ghosts impose stationary normal velocity and continue tangential velocity.
// Compatible-floor transfer is bounded by the boundary transfer capacities.
// Requires the platen gap to exceed one cell throughout the step and enforces motion and acoustic bounds.
// Invalid inputs return InvalidArgument, inadmissible material or unsupported motion return FailedPrecondition,
// and unrepresentable arithmetic returns OutOfRange. If provided, budget must be disjoint from all inputs.
// A successful step replaces the budget. Failure preserves the entire particle span and budget.
absl::Status AdvanceDrySolid3D(
    const SparseMacGrid3D& grid,
    const DrySolidMaterial3D& material,
    const DrySolidStep3D& settings,
    std::span<DrySolidParticle3D> particles,
    DrySolidStepBudget3D* budget = nullptr
);

}  // namespace fluidity
