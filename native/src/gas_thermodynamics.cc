#include "fluidity/host/gas_thermodynamics.h"

#include <cmath>
#include <vector>

#include "absl/status/status.h"

namespace fluidity {
namespace {

constexpr double kGasGamma = 1.4;

struct ComponentHeating {
  double volume = 0.0;
  double integrated_heat_rate = 0.0;
  double mean_heat_rate = 0.0;
};

bool IsPositiveFinite(double value) {
  return value > 0.0 && std::isfinite(value);
}

}  // namespace

absl::StatusOr<GasHeatingUpdate> AdvanceGasHeating(
    double step_duration,
    std::span<const GasHeatingCell> cells,
    std::span<const GasThermodynamicComponent> components
) {
  if (!IsPositiveFinite(step_duration) || cells.empty() || components.empty()) {
    return absl::InvalidArgumentError("gas heating requires a positive step and nonempty cells and components");
  }
  std::vector<ComponentHeating> heating(components.size());
  for (const GasHeatingCell& cell : cells) {
    if (cell.component >= components.size()) {
      return absl::InvalidArgumentError("gas heating cell has an unknown component");
    }
    if (!IsPositiveFinite(cell.volume) || !std::isfinite(cell.heat_rate)) {
      return absl::InvalidArgumentError("gas heating cell has invalid volume or heat rate");
    }
    auto& component = heating[cell.component];
    component.volume += cell.volume;
    component.integrated_heat_rate += cell.volume * cell.heat_rate;
  }

  GasHeatingUpdate update;
  update.component_pressure.resize(components.size());
  update.target_divergence.reserve(cells.size());
  update.enthalpy_pressure_work_rate.resize(components.size());
  for (std::size_t ordinal = 0; ordinal < components.size(); ++ordinal) {
    const auto& component = components[ordinal];
    auto& component_heating = heating[ordinal];
    if (component_heating.volume == 0.0 || !IsPositiveFinite(component.absolute_pressure)) {
      return absl::InvalidArgumentError("gas component has no cells or invalid absolute pressure");
    }
    if (!std::isfinite(component_heating.volume) || !std::isfinite(component_heating.integrated_heat_rate)) {
      return absl::OutOfRangeError("integrated gas heat or component volume overflowed");
    }
    if (!component.open_to_ambient) {
      component_heating.mean_heat_rate = component_heating.integrated_heat_rate / component_heating.volume;
    }
    const double pressure_work_rate = (kGasGamma - 1.0) * component_heating.mean_heat_rate;
    const double new_pressure = component.absolute_pressure + step_duration * pressure_work_rate;
    if (!IsPositiveFinite(new_pressure)) {
      return absl::OutOfRangeError("updated gas pressure must remain finite and positive");
    }
    update.component_pressure[ordinal] = new_pressure;
    update.enthalpy_pressure_work_rate[ordinal] = pressure_work_rate;
  }

  // Subtract the mean before scaling so a uniformly heated sealed cell has
  // exactly zero expansion when the component mean is exactly representable.
  for (const GasHeatingCell& cell : cells) {
    const auto& component = components[cell.component];
    const double relative_heat_rate = cell.heat_rate - heating[cell.component].mean_heat_rate;
    const double divergence = ((kGasGamma - 1.0) / kGasGamma) * (relative_heat_rate / component.absolute_pressure);
    if (!std::isfinite(divergence)) {
      return absl::OutOfRangeError("gas heating divergence overflowed");
    }
    update.target_divergence.push_back(divergence);
  }
  return update;
}

}  // namespace fluidity
