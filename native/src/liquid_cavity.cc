#include "fluidity/host/liquid_cavity.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <tuple>

#include "fluidity/host/views.h"

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

absl::StatusOr<LiquidCavityInventory3D> CombineLiquidCavityInventoryShares3D(
    std::span<const LiquidCavityInventoryShare3D> shares
) {
  if (shares.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }

  double reference_pressure = 0.0;
  for (const auto& share : shares) {
    if (!PositiveFinite(share.parent.initial_volume) || !PositiveFinite(share.parent.initial_absolute_pressure)
        || !PositiveFinite(share.fraction) || share.fraction > 1.0) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    reference_pressure = std::max(reference_pressure, share.parent.initial_absolute_pressure);
  }

  // Scale the additive invariant by the largest pressure root so individual
  // volume contributions never grow. Form roots first to preserve pressure
  // ratios that would underflow before exponentiation.
  constexpr double kInversePolytropicExponent = 1.0 / kPolytropicExponent;
  const double reference_pressure_root = std::pow(reference_pressure, kInversePolytropicExponent);
  double reference_volume = 0.0;
  for (const auto& share : shares) {
    const double parent_pressure_root = std::pow(share.parent.initial_absolute_pressure, kInversePolytropicExponent);
    const double pressure_root_ratio = parent_pressure_root / reference_pressure_root;
    const double allocated_parent_volume = share.fraction * share.parent.initial_volume;
    reference_volume = std::fma(allocated_parent_volume, pressure_root_ratio, reference_volume);
  }
  if (!PositiveFinite(reference_volume)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return LiquidCavityInventory3D{
      .initial_volume = reference_volume,
      .initial_absolute_pressure = reference_pressure,
  };
}

absl::StatusOr<LiquidCavityInventory3D> InitializeSubmergedLiquidCavity3D(
    double volume,
    double atmospheric_absolute_pressure,
    double hydrostatic_gauge_pressure,
    double surface_tension
) {
  if (!PositiveFinite(volume) || !PositiveFinite(atmospheric_absolute_pressure)
      || !NonnegativeFinite(hydrostatic_gauge_pressure) || !NonnegativeFinite(surface_tension)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const double liquid_absolute_pressure = atmospheric_absolute_pressure + hydrostatic_gauge_pressure;
  const double curvature_magnitude = 2.0 / EquivalentRadius(volume);
  const double initial_absolute_pressure = std::fma(surface_tension, curvature_magnitude, liquid_absolute_pressure);
  if (!PositiveFinite(initial_absolute_pressure)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
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
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
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
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  const double equivalent_radius = EquivalentRadius(volume);
  const double maximum_step_duration = RadialStepLimit(equivalent_radius, liquid_density, absolute_pressure);
  if (!PositiveFinite(maximum_step_duration)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return LiquidCavityPressureState3D{
      .absolute_pressure = absolute_pressure,
      .gas_gauge_pressure = gauge_pressure,
      .boundary_gauge_pressure = boundary_gauge_pressure,
      .equivalent_radius = equivalent_radius,
      .maximum_step_duration = maximum_step_duration,
      .resolved = equivalent_radius / 2.0 >= cell_width,
  };
}

absl::StatusOr<LiquidCavityBoundaryPreparation3D> PrepareLiquidCavityBoundaries3D(
    const LiquidCavityBoundaryInputs3D& inputs,
    std::span<LiquidCavityBoundaryComponent3D> component_scratch,
    std::span<LiquidAirCellPressure3D> air_pressures
) {
  const auto invalid = [] {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  };
  if (inputs.gas_cells.size() != inputs.labels.size() || inputs.components.size() != inputs.inventories.size()
      || component_scratch.size() != inputs.components.size() || !PositiveFinite(inputs.atmospheric_absolute_pressure)
      || !PositiveFinite(inputs.liquid_density) || !PositiveFinite(inputs.cell_width)) {
    return invalid();
  }
  const auto component_bytes = std::as_bytes(component_scratch);
  const auto pressure_bytes = std::as_bytes(air_pressures);
  if (ByteViewsOverlap(component_bytes, pressure_bytes)) {
    return invalid();
  }
  for (const auto source : {
           std::as_bytes(inputs.gas_cells),
           std::as_bytes(inputs.labels),
           std::as_bytes(inputs.components),
           std::as_bytes(inputs.inventories),
           std::as_bytes(inputs.center_distances),
           std::as_bytes(std::span(&inputs, 1)),
       }) {
    if (ByteViewsOverlap(source, component_bytes) || ByteViewsOverlap(source, pressure_bytes)) {
      return invalid();
    }
  }
  const auto key = [](CellIndex3D cell) {
    return std::tuple{cell.z, cell.y, cell.x};
  };
  std::size_t next_component = 0;
  for (std::size_t index = 0; index < inputs.gas_cells.size(); ++index) {
    const auto& gas = inputs.gas_cells[index];
    const auto label = inputs.labels[index];
    if (label >= inputs.components.size() || !PositiveFinite(gas.gas_volume)
        || (index > 0 && key(inputs.gas_cells[index - 1].cell_index) >= key(gas.cell_index))) {
      return invalid();
    }
    if (label < next_component) {
      continue;
    }
    if (label != next_component || key(gas.cell_index) != key(inputs.components[label].first_cell)) {
      return invalid();
    }
    ++next_component;
  }
  if (next_component != inputs.components.size()) {
    return invalid();
  }
  for (std::size_t index = 0; index < inputs.center_distances.size(); ++index) {
    const auto& center = inputs.center_distances[index];
    if (!std::isfinite(center.signed_distance)
        || (index > 0 && key(inputs.center_distances[index - 1].cell_index) >= key(center.cell_index))) {
      return invalid();
    }
  }

  LiquidCavityBoundaryPreparation3D result;
  for (std::size_t index = 0; index < inputs.components.size(); ++index) {
    auto& output = component_scratch[index];
    output = {};
    const auto& component = inputs.components[index];
    if (component.connectivity > LiquidAirConnectivity3D::kVented || !PositiveFinite(component.gas_volume)) {
      return invalid();
    }
    if (component.connectivity != LiquidAirConnectivity3D::kEnclosed || !inputs.inventories[index]) {
      continue;
    }
    if (EquivalentRadius(component.gas_volume) / 2.0 < inputs.cell_width) {
      ++result.subresolution_inventory_count;
      continue;
    }
    const auto state = EvaluateLiquidCavity3D(
        *inputs.inventories[index],
        component.gas_volume,
        inputs.atmospheric_absolute_pressure,
        inputs.liquid_density,
        0.0,
        0.0,
        inputs.cell_width
    );
    if (!state.ok()) {
      return state.status();
    }
    output = {.active = true, .gas_gauge_pressure = state->gas_gauge_pressure};
    result.maximum_step_duration = std::min(result.maximum_step_duration, state->maximum_step_duration);
    ++result.active_cavity_count;
  }

  // Merge the ordered gas records and frozen pressure center samples.
  std::size_t center = 0;
  for (std::size_t index = 0; index < inputs.gas_cells.size(); ++index) {
    const auto& component = component_scratch[inputs.labels[index]];
    if (!component.active) {
      continue;
    }
    const auto cell = inputs.gas_cells[index].cell_index;
    const auto cell_key = key(cell);
    while (center < inputs.center_distances.size() && key(inputs.center_distances[center].cell_index) < cell_key) {
      ++center;
    }
    if (center == inputs.center_distances.size() || key(inputs.center_distances[center].cell_index) != cell_key
        || inputs.center_distances[center].signed_distance < 0.0) {
      continue;
    }
    if (result.air_pressure_count == air_pressures.size()) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    air_pressures[result.air_pressure_count++] = {cell, component.gas_gauge_pressure};
  }
  return result;
}

}  // namespace fluidity
