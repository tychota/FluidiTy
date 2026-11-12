#include "fluidity/host/liquid_cavity.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <tuple>
#include <utility>

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

absl::StatusOr<LiquidCavityLineageResult3D> PrepareLiquidCavityLineage3D(
    const LiquidCavityLineageInputs3D& inputs,
    std::span<LiquidCavityLineageCandidate3D> candidates
) {
  if (candidates.size() != inputs.current.size() || inputs.initial_inventories.size() != inputs.current.size()
      || !PositiveFinite(inputs.cell_width) || inputs.next_id == 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto alias = CheckWritableViews(
      {std::as_bytes(candidates)},
      {std::as_bytes(inputs.previous),
       std::as_bytes(inputs.current),
       std::as_bytes(inputs.overlaps),
       std::as_bytes(inputs.initial_inventories),
       std::as_bytes(std::span(&inputs, 1))}
  );
  if (!alias.ok()) {
    return alias;
  }
  const auto valid_inventory = [](const LiquidCavityInventory3D& inventory) {
    return PositiveFinite(inventory.initial_volume) && PositiveFinite(inventory.initial_absolute_pressure);
  };
  for (const auto& parent : inputs.previous) {
    if (parent.inventory.has_value() != (parent.id != 0) || parent.id >= inputs.next_id
        || (parent.inventory && !valid_inventory(*parent.inventory))) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  for (std::size_t current = 0; current < inputs.current.size(); ++current) {
    const auto& component = inputs.current[current];
    const auto& initial = inputs.initial_inventories[current];
    if (!PositiveFinite(component.gas_volume) || component.connectivity > LiquidAirConnectivity3D::kVented
        || (initial && !valid_inventory(*initial))) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  const auto edge_key = [](const LiquidAirComponentOverlap3D& edge) {
    return std::pair{edge.previous_component, edge.current_component};
  };
  for (std::size_t index = 0; index < inputs.overlaps.size(); ++index) {
    const auto& edge = inputs.overlaps[index];
    if (edge.previous_component >= inputs.previous.size() || edge.current_component >= inputs.current.size()
        || !PositiveFinite(edge.shared_cell_volume)
        || (index > 0 && edge_key(inputs.overlaps[index - 1]) >= edge_key(edge))) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }

  const auto resolved = [&inputs](std::size_t current) {
    return EquivalentRadius(inputs.current[current].gas_volume) / 2.0 >= inputs.cell_width;
  };
  const auto accumulate_loss = [](double& loss, const LiquidCavityInventory3D& inventory, double fraction) {
    const double pressure_root = std::pow(inventory.initial_absolute_pressure, 1.0 / kPolytropicExponent);
    loss = std::fma(inventory.initial_volume * fraction, pressure_root, loss);
    return std::isfinite(loss);
  };
  std::ranges::fill(candidates, LiquidCavityLineageCandidate3D{});
  LiquidCavityLineageResult3D result{.next_id = inputs.next_id};
  auto remaining_edges = inputs.overlaps;
  for (std::size_t previous = 0; previous < inputs.previous.size(); ++previous) {
    std::size_t outgoing_count = 0;
    while (outgoing_count < remaining_edges.size() && remaining_edges[outgoing_count].previous_component == previous) {
      ++outgoing_count;
    }
    const auto outgoing = remaining_edges.first(outgoing_count);
    remaining_edges = remaining_edges.subspan(outgoing_count);
    const auto& parent = inputs.previous[previous];
    if (!parent.inventory) {
      continue;
    }
    if (outgoing.empty()) {
      if (!accumulate_loss(result.unmatched_inventory, *parent.inventory, 1.0)) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      continue;
    }

    double outgoing_volume = 0.0;
    const LiquidAirComponentOverlap3D* nomination = nullptr;
    for (const auto& edge : outgoing) {
      const auto connectivity = inputs.current[edge.current_component].connectivity;
      if (connectivity == LiquidAirConnectivity3D::kUnclassified) {
        return absl::Status(absl::StatusCode::kFailedPrecondition, {});
      }
      outgoing_volume += edge.shared_cell_volume;
      if (connectivity == LiquidAirConnectivity3D::kEnclosed && resolved(edge.current_component)
          && (nomination == nullptr || edge.shared_cell_volume > nomination->shared_cell_volume)) {
        nomination = &edge;
      }
    }
    if (!PositiveFinite(outgoing_volume)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    for (const auto& edge : outgoing) {
      const std::size_t current = edge.current_component;
      const double fraction = edge.shared_cell_volume / outgoing_volume;
      const auto connectivity = inputs.current[current].connectivity;
      if (connectivity == LiquidAirConnectivity3D::kVented) {
        if (!accumulate_loss(result.vented_inventory, *parent.inventory, fraction)) {
          return absl::Status(absl::StatusCode::kOutOfRange, {});
        }
        continue;
      }
      if (!resolved(current)) {
        if (!accumulate_loss(result.subresolution_inventory, *parent.inventory, fraction)) {
          return absl::Status(absl::StatusCode::kOutOfRange, {});
        }
        continue;
      }

      auto& candidate = candidates[current];
      const auto& existing = candidate.state.inventory;
      const LiquidCavityInventoryShare3D share{*parent.inventory, fraction};
      const std::array<LiquidCavityInventoryShare3D, 2> shares{{
          {existing.value_or(*parent.inventory), 1.0},
          share,
      }};
      const auto combined = CombineLiquidCavityInventoryShares3D(std::span(shares).last(existing ? 2 : 1));
      if (!combined.ok()) {
        return combined.status();
      }
      candidate.state.inventory = *combined;
      if (nomination == &edge
          && (edge.shared_cell_volume > candidate.inherited_id_overlap
              || (edge.shared_cell_volume == candidate.inherited_id_overlap && parent.id < candidate.state.id))) {
        candidate.state.id = parent.id;
        candidate.inherited_id_overlap = edge.shared_cell_volume;
      }
    }
  }

  for (std::size_t current = 0; current < candidates.size(); ++current) {
    if (inputs.current[current].connectivity != LiquidAirConnectivity3D::kEnclosed || !resolved(current)) {
      continue;
    }
    auto& state = candidates[current].state;
    if (!state.inventory) {
      const auto& initial = inputs.initial_inventories[current];
      if (!initial) {
        return absl::Status(absl::StatusCode::kFailedPrecondition, {});
      }
      state.inventory = *initial;
    }
    if (state.id == 0) {
      if (result.next_id == std::numeric_limits<std::uint64_t>::max()) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      state.id = result.next_id++;
    }
  }
  return result;
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

absl::StatusOr<LiquidCavityStepPreparation3D>
PrepareLiquidCavityStep3D(const LiquidCavityStepInputs3D& inputs, const LiquidCavityStepScratch3D& scratch) {
  if (inputs.previous_snapshot.component_count != inputs.previous_lineage.size()
      || !PositiveFinite(inputs.atmospheric_absolute_pressure) || !PositiveFinite(inputs.liquid_density)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto alias = CheckWritableViews(
      {std::as_bytes(scratch.gas_cells),
       std::as_bytes(scratch.labels),
       std::as_bytes(scratch.queue),
       std::as_bytes(scratch.components),
       std::as_bytes(scratch.overlaps),
       std::as_bytes(scratch.inventories),
       std::as_bytes(scratch.lineage),
       std::as_bytes(scratch.boundary_components),
       std::as_bytes(scratch.air_pressures)},
      {std::as_bytes(inputs.sampled_cells),
       std::as_bytes(inputs.corners),
       std::as_bytes(inputs.centers),
       std::as_bytes(inputs.atmosphere_outlets),
       std::as_bytes(inputs.previous_snapshot.cells),
       std::as_bytes(inputs.previous_snapshot.labels),
       std::as_bytes(inputs.previous_lineage),
       std::as_bytes(std::span(&inputs.obstacle, 1)),
       std::as_bytes(std::span(&inputs, 1)),
       std::as_bytes(std::span(&scratch, 1))}
  );
  if (!alias.ok()) {
    return alias;
  }
  const auto gas_count = BuildLiquidAirDomain3D(
      inputs.geometry,
      inputs.obstacle,
      inputs.sampled_cells,
      inputs.corners,
      inputs.atmosphere_outlets,
      scratch.gas_cells
  );
  if (!gas_count.ok()) {
    return gas_count.status();
  }
  if (*gas_count > scratch.labels.size() || *gas_count > scratch.queue.size()) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  const auto cells = scratch.gas_cells.first(*gas_count);
  const auto labels = scratch.labels.first(*gas_count);
  const auto component_count = LabelLiquidAirComponents3D(cells, labels, scratch.queue, scratch.components);
  if (!component_count.ok()) {
    return component_count.status();
  }
  if (*component_count > scratch.inventories.size() || *component_count > scratch.lineage.size()
      || *component_count > scratch.boundary_components.size()) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }

  const LiquidAirComponentSnapshot3D current{cells, labels, *component_count};
  const auto overlap_count = MeasureLiquidAirComponentOverlaps3D(inputs.previous_snapshot, current, scratch.overlaps);
  if (!overlap_count.ok()) {
    return overlap_count.status();
  }
  const auto overlaps = scratch.overlaps.first(*overlap_count);
  const auto inventories = scratch.inventories.first(*component_count);
  const auto components = scratch.components.first(*component_count);
  const auto lineage_candidates = scratch.lineage.first(*component_count);
  std::ranges::fill(inventories, std::nullopt);
  // The ID-overlap scratch also serves as an incoming-inventory marker until
  // the lineage stage clears it. Only births need equilibrium initialization.
  std::ranges::fill(lineage_candidates, LiquidCavityLineageCandidate3D{});
  for (const auto& overlap : overlaps) {
    if (inputs.previous_lineage[overlap.previous_component].inventory) {
      lineage_candidates[overlap.current_component].inherited_id_overlap = overlap.shared_cell_volume;
    }
  }
  for (std::size_t index = 0; index < components.size(); ++index) {
    const auto& component = components[index];
    if (component.connectivity != LiquidAirConnectivity3D::kEnclosed
        || EquivalentRadius(component.gas_volume) / 2.0 < inputs.geometry.cell_width
        || lineage_candidates[index].inherited_id_overlap > 0.0) {
      continue;
    }
    if (inputs.initialize == nullptr) {
      return absl::Status(absl::StatusCode::kFailedPrecondition, {});
    }
    const auto initial = inputs.initialize(inputs.initialization_context, component);
    if (!initial.ok()) {
      return initial.status();
    }
    inventories[index] = *initial;
  }
  const auto lineage = PrepareLiquidCavityLineage3D(
      {inputs.previous_lineage, components, overlaps, inventories, inputs.next_id, inputs.geometry.cell_width},
      lineage_candidates
  );
  if (!lineage.ok()) {
    return lineage.status();
  }
  for (std::size_t index = 0; index < inventories.size(); ++index) {
    inventories[index] = lineage_candidates[index].state.inventory;
  }
  const auto boundaries = PrepareLiquidCavityBoundaries3D(
      {cells,
       labels,
       components,
       inventories,
       inputs.centers,
       inputs.atmospheric_absolute_pressure,
       inputs.liquid_density,
       inputs.geometry.cell_width},
      scratch.boundary_components.first(*component_count),
      scratch.air_pressures
  );
  if (!boundaries.ok()) {
    return boundaries.status();
  }
  return LiquidCavityStepPreparation3D{
      .gas_cell_count = *gas_count,
      .component_count = *component_count,
      .lineage = *lineage,
      .boundaries = *boundaries,
  };
}

}  // namespace fluidity
