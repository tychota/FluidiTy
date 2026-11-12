#include "fluidity/host/liquid_cavity.h"

#include <cmath>
#include <numbers>

namespace fluidity {
namespace {

constexpr double kPolytropicExponent = 1.4;

bool PositiveFinite(double value) {
  return std::isfinite(value) && value > 0.0;
}

bool NonnegativeFinite(double value) {
  return std::isfinite(value) && value >= 0.0;
}

double EquivalentRadius(double volume) {
  const double sphere_factor = std::cbrt(3.0 / (4.0 * std::numbers::pi));
  return sphere_factor * std::cbrt(volume);
}

// Keep radius times the acoustic density/pressure ratio representable when
// their intermediate product or quotient would leave the double range.
double RadialStepLimit(double radius, double density, double absolute_pressure) {
  int radius_exponent;
  int density_root_exponent;
  int pressure_root_exponent;
  const double radius_fraction = std::frexp(radius, &radius_exponent);
  const double density_root_fraction = std::frexp(std::sqrt(density), &density_root_exponent);
  const double pressure_root_fraction = std::frexp(std::sqrt(absolute_pressure), &pressure_root_exponent);
  const double step_fraction =
      (0.5 / std::sqrt(3.0 * kPolytropicExponent)) * radius_fraction * density_root_fraction / pressure_root_fraction;
  return std::scalbn(step_fraction, radius_exponent + density_root_exponent - pressure_root_exponent);
}

}  // namespace

absl::StatusOr<LiquidCavityInventory3D> InitializeSubmergedLiquidCavity3D(
    double volume,
    double atmospheric_absolute_pressure,
    double hydrostatic_gauge_pressure,
    double surface_tension
) {
  if (!PositiveFinite(volume) || !PositiveFinite(atmospheric_absolute_pressure)
      || !NonnegativeFinite(hydrostatic_gauge_pressure) || !NonnegativeFinite(surface_tension)) {
    return absl::InvalidArgumentError("invalid submerged liquid cavity data");
  }
  const double liquid_absolute_pressure = atmospheric_absolute_pressure + hydrostatic_gauge_pressure;
  const double curvature_magnitude = 2.0 / EquivalentRadius(volume);
  const double initial_absolute_pressure = std::fma(surface_tension, curvature_magnitude, liquid_absolute_pressure);
  if (!PositiveFinite(initial_absolute_pressure)) {
    return absl::OutOfRangeError("initial liquid cavity pressure is unrepresentable");
  }
  return LiquidCavityInventory3D{
      .initial_volume = volume,
      .initial_absolute_pressure = initial_absolute_pressure,
  };
}

absl::StatusOr<LiquidCavityPressureState3D> EvaluateLiquidCavity3D(
    LiquidCavityInventory3D inventory,
    double volume,
    double atmospheric_absolute_pressure,
    double liquid_density,
    double surface_tension,
    double liquid_curvature,
    double cell_width
) {
  if (!PositiveFinite(inventory.initial_volume) || !PositiveFinite(inventory.initial_absolute_pressure)
      || !PositiveFinite(volume) || !PositiveFinite(atmospheric_absolute_pressure) || !PositiveFinite(liquid_density)
      || !NonnegativeFinite(surface_tension) || !std::isfinite(liquid_curvature) || !PositiveFinite(cell_width)) {
    return absl::InvalidArgumentError("invalid liquid cavity state data");
  }
  double absolute_pressure;
  double gauge_pressure;
  const double relative_compression = (inventory.initial_volume - volume) / volume;
  if (relative_compression >= -0.5 && relative_compression <= 0.5) {
    // Preserve a small gauge increment even when absolute pressure rounds
    // back to the initial inventory pressure.
    const double pressure_increment =
        inventory.initial_absolute_pressure * std::expm1(kPolytropicExponent * std::log1p(relative_compression));
    absolute_pressure = inventory.initial_absolute_pressure + pressure_increment;
    gauge_pressure = (inventory.initial_absolute_pressure - atmospheric_absolute_pressure) + pressure_increment;
  } else {
    const double log_volume_ratio = std::log(inventory.initial_volume) - std::log(volume);
    absolute_pressure =
        std::exp(std::fma(kPolytropicExponent, log_volume_ratio, std::log(inventory.initial_absolute_pressure)));
    gauge_pressure = absolute_pressure - atmospheric_absolute_pressure;
  }
  const double boundary_gauge_pressure = std::fma(surface_tension, liquid_curvature, gauge_pressure);
  if (!PositiveFinite(absolute_pressure) || !std::isfinite(boundary_gauge_pressure)) {
    return absl::OutOfRangeError("liquid cavity pressure is unrepresentable");
  }
  const double equivalent_radius = EquivalentRadius(volume);
  const double maximum_step_duration = RadialStepLimit(equivalent_radius, liquid_density, absolute_pressure);
  if (!PositiveFinite(maximum_step_duration)) {
    return absl::OutOfRangeError("liquid cavity timestep is unrepresentable");
  }
  return LiquidCavityPressureState3D{
      .absolute_pressure = absolute_pressure,
      .boundary_gauge_pressure = boundary_gauge_pressure,
      .equivalent_radius = equivalent_radius,
      .maximum_step_duration = maximum_step_duration,
      .resolved = equivalent_radius / 2.0 >= cell_width,
  };
}

}  // namespace fluidity
