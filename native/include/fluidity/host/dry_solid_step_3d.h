#pragma once

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
};

// Advances a single dry body on a fixed uniform MAC layout with complete quadratic supports.
// Uses APIC/PIC, uncut stress quadrature, a stationary floor, and an optional translating flat platen.
// Snow grid normal constraints use face anchors. A continuous halfspace sweep constrains particle transport.
// The force-acceleration denominator is floored at 1e-4 times the smallest particle mass for low-mass faces.
// The bounded dry path uses no liquid pressure, cut support reconstruction, or body mixing.
// Requires the platen gap to exceed one cell throughout the step and enforces motion and acoustic bounds.
// Invalid inputs return InvalidArgument, inadmissible material or unsupported motion return FailedPrecondition,
// and unrepresentable arithmetic returns OutOfRange. On failure the entire particle span is preserved.
absl::Status AdvanceDrySolid3D(
    const SparseMacGrid3D& grid,
    const DrySolidMaterial3D& material,
    const DrySolidStep3D& settings,
    std::span<DrySolidParticle3D> particles
);

}  // namespace fluidity
