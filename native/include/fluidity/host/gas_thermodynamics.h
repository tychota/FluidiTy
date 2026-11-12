#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "absl/status/statusor.h"

namespace fluidity {

struct GasHeatingCell {
  double volume;  // Fixed gas volume in m³.
  double heat_rate;  // Volumetric external, reaction and net conductive heating in W/m³.
  std::size_t component;  // Index into the thermodynamic components in pressure connectivity order.
};

struct GasThermodynamicComponent {
  double absolute_pressure;  // Old-time pressure in Pa, or prescribed ambient pressure for an open component.
  bool open_to_ambient;
};

struct GasHeatingUpdate {
  std::vector<double> component_pressure;  // New-time absolute Pa in component input order.
  std::vector<double> target_divergence;  // s⁻¹ in cell input order.
  std::vector<double> enthalpy_pressure_work_rate;  // dp0/dt in W/m³ in component input order.
};

// Computes the fixed-domain low-Mach divergence and pressure work for gamma=1.4.
// Each cell must belong to one supplied component, and every component must
// contain at least one cell. Volumes, pressures and step duration (s) must be
// finite and positive. The divergence denominator uses old-time pressure.
// Invalid input returns InvalidArgument. Unrepresentable results or a
// nonpositive new pressure return OutOfRange. Errors carry no update.
absl::StatusOr<GasHeatingUpdate> AdvanceGasHeating(
    double step_duration,
    std::span<const GasHeatingCell> cells,
    std::span<const GasThermodynamicComponent> components
);

}  // namespace fluidity
