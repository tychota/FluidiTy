#pragma once

#include "absl/status/statusor.h"

namespace fluidity {

struct LiquidCavityInventory3D {
  double initial_volume;  // Positive m³.
  double initial_absolute_pressure;  // Positive Pa.
};

struct LiquidCavityPressureState3D {
  double absolute_pressure;  // Pa.
  double boundary_gauge_pressure;  // Pa, including the signed liquid-side Laplace jump.
  double equivalent_radius;  // m.
  double maximum_step_duration;  // s, from the radial cavity frequency.
  bool resolved;  // Equivalent radius is at least twice the cell width.
};

// Initializes the absolute inventory of a submerged spherical cavity in
// mechanical equilibrium. Volume is positive m³, atmospheric absolute
// pressure is positive Pa, hydrostatic gauge pressure is nonnegative Pa,
// and surface tension is nonnegative N/m. Nonfinite or invalid inputs return
// InvalidArgument. Unrepresentable inventory returns OutOfRange.
absl::StatusOr<LiquidCavityInventory3D> InitializeSubmergedLiquidCavity3D(
    double volume,
    double atmospheric_absolute_pressure,
    double hydrostatic_gauge_pressure,
    double surface_tension
);

// Evaluates a fixed inventory with polytropic exponent 1.4. Volume is positive
// m³, atmospheric pressure is positive absolute Pa, liquid density is positive
// kg/m³, surface tension is nonnegative N/m, curvature is signed outward from
// liquid in m⁻¹, and cell width is positive m. A spherical cavity has negative
// liquid curvature. The result is a single boundary pressure lift. Invalid
// inputs return InvalidArgument and unrepresentable results return OutOfRange.
absl::StatusOr<LiquidCavityPressureState3D> EvaluateLiquidCavity3D(
    LiquidCavityInventory3D inventory,
    double volume,
    double atmospheric_absolute_pressure,
    double liquid_density,
    double surface_tension,
    double liquid_curvature,
    double cell_width
);

}  // namespace fluidity
