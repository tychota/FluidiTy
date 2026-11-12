#pragma once

#include <limits>
#include <optional>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_air_components.h"

namespace fluidity {

struct LiquidCavityInventory3D {
  double initial_volume;  // Positive m³.
  double initial_absolute_pressure;  // Positive Pa.
};

struct LiquidCavityPressureState3D {
  double absolute_pressure;  // Pa.
  double gas_gauge_pressure;  // Pa relative to atmosphere, before the capillary lift.
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

struct LiquidCavityBoundaryInputs3D {
  std::span<const LiquidAirVolumeCell3D> gas_cells;  // Sorted gas nodes from the sampled domain.
  std::span<const std::size_t> labels;  // Component index for each gas cell.
  std::span<const LiquidAirComponent3D> components;
  std::span<const std::optional<LiquidCavityInventory3D>> inventories;  // Matched by component.
  std::span<const LiquidCenterDistance3D> center_distances;  // Frozen pressure sample order (z,y,x).
  double atmospheric_absolute_pressure;  // Pa.
  double liquid_density;  // kg/m³.
  double cell_width;  // m.
};

struct LiquidCavityBoundaryComponent3D {
  bool active = false;
  double gas_gauge_pressure = 0.0;  // Pa before capillarity.
};

struct LiquidCavityBoundaryPreparation3D {
  std::size_t air_pressure_count = 0;
  std::size_t active_cavity_count = 0;
  std::size_t subresolution_inventory_count = 0;
  double maximum_step_duration = std::numeric_limits<double>::infinity();  // s.
};

// Prepares the gas-only pressure lift and minimum radial step bound for
// enclosed, resolved components with matched inventories. Unclassified
// and vented components supply no override. Emits only air centers present
// in the frozen pressure samples. Scratch spans are disjoint from all inputs
// and each other. InvalidArgument covers invalid inputs, ResourceExhausted
// covers insufficient pressure output, and cavity arithmetic errors propagate.
// Scratch may change on failure. No partially prepared result is returned.
absl::StatusOr<LiquidCavityBoundaryPreparation3D> PrepareLiquidCavityBoundaries3D(
    const LiquidCavityBoundaryInputs3D& inputs,
    std::span<LiquidCavityBoundaryComponent3D> component_scratch,
    std::span<LiquidAirCellPressure3D> air_pressures
);

}  // namespace fluidity
