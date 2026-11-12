#include "fluidity/host/gas_substep_3d.h"

#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "fluidity/host/gas_thermodynamics.h"
#include "mac_pressure_3d_system.h"

namespace fluidity {
namespace {

bool PositiveFinite(double value) {
  return value > 0.0 && std::isfinite(value);
}

}  // namespace

absl::StatusOr<GasSubstep3D> AdvanceGasSubstep3D(
    const SparseMacGrid3D& grid,
    double step_duration,
    std::span<const GasCell3D> cells,
    std::span<const MacPressureFace3D> faces,
    std::span<const GasCellSource3D> sources,
    GasScalars ambient,
    double ambient_pressure,
    Vec3 gravity,
    absl::FunctionRef<absl::StatusOr<MacPressureProjection3D>(const MacPressureInputs3D&, MacPressureSolverOptions3D)>
        project
) {
  if (cells.size() != sources.size() || !PositiveFinite(ambient_pressure) || !PositiveFinite(ambient.carrier)
      || !PositiveFinite(ambient.enthalpy)) {
    return absl::InvalidArgumentError("gas substep requires matching sources and positive ambient state");
  }
  const std::array gravity_components{gravity.x, gravity.y, gravity.z};
  for (const double component : gravity_components) {
    if (!std::isfinite(component)) {
      return absl::InvalidArgumentError("gas gravity must be finite");
    }
  }
  std::vector<MacPressureCell3D> pressure_cells;
  pressure_cells.reserve(cells.size());
  for (const auto& cell : cells) {
    if (!PositiveFinite(cell.density.carrier) || !PositiveFinite(cell.density.enthalpy)
        || !PositiveFinite(cell.absolute_pressure)) {
      return absl::InvalidArgumentError("gas cell requires positive carrier, enthalpy and absolute pressure");
    }
    pressure_cells.push_back({
        .cell_index = cell.cell_index,
        .fluid_volume = cell.volume,
        .target_divergence = 0.0,
        .bulk_modulus = std::numeric_limits<double>::infinity(),
        .previous_pressure = cell.gauge_pressure,
    });
  }
  std::vector<MacPressureFace3D> pressure_faces(faces.begin(), faces.end());
  for (auto& face : pressure_faces) {
    const auto* anchor = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link);
    const auto* interior = std::get_if<InternalMacPressureLink3D>(&face.pressure_link);
    if (face.wall_normal_velocity != 0.0 || face.prescribed_wall_flux.value_or(0.0) != 0.0
        || (anchor && anchor->exterior_pressure != 0.0) || (interior && interior->pressure_jump != 0.0)
        || !std::isfinite(face.predicted_normal_velocity)) {
      return absl::InvalidArgumentError("fixed gas faces require stationary walls and a continuous zero exterior gauge"
      );
    }
    face.inverse_density = 1.0;
    face.predicted_normal_velocity = 0.0;
  }

  // Use the pressure assembler's checked face incidence and connectivity before
  // adding thermal forcing. Its provisional RHS is compatible even in sealed pockets.
  const auto topology =
      pressure_internal::PrepareMacPressureSystem3D({grid, step_duration, pressure_cells, pressure_faces});
  if (!topology.ok()) {
    return topology.status();
  }
  const auto roots = topology->component_ordinals();
  const auto links = topology->links();
  std::vector<GasThermodynamicComponent> components;
  std::vector<std::size_t> component_by_root(cells.size());
  std::vector<bool> open_root(cells.size(), false);
  for (const auto& link : links) {
    if (std::holds_alternative<DirichletMacPressureLink3D>(link.pressure_link)) {
      const auto cell = link.tail_cell_ordinal ? *link.tail_cell_ordinal : *link.head_cell_ordinal;
      open_root[roots[cell]] = true;
    }
  }
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    if (roots[cell] == cell) {
      const bool open = open_root[cell];
      component_by_root[cell] = components.size();
      components.push_back({
          .absolute_pressure = open ? ambient_pressure : cells[cell].absolute_pressure,
          .open_to_ambient = open,
      });
    }
  }

  std::vector<GasHeatingCell> heating;
  std::vector<GasTransportCell> transport;
  std::vector<double> fuel_burn_rates;
  heating.reserve(cells.size());
  transport.reserve(cells.size());
  fuel_burn_rates.reserve(cells.size());
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const std::size_t component = component_by_root[roots[cell]];
    if (!components[component].open_to_ambient
        && components[component].absolute_pressure != cells[cell].absolute_pressure) {
      return absl::InvalidArgumentError("sealed gas component has inconsistent absolute pressure");
    }
    const auto& source = sources[cell];
    if (!std::isfinite(source.burn_rate) || source.burn_rate < 0.0 || !std::isfinite(source.heat_per_fuel)
        || source.heat_per_fuel < 0.0 || !std::isfinite(source.soot_yield) || source.soot_yield < 0.0
        || source.soot_yield > 1.0 || !std::isfinite(cells[cell].density.fuel) || cells[cell].density.fuel < 0.0
        || !std::isfinite(cells[cell].density.soot) || cells[cell].density.soot < 0.0) {
      return absl::InvalidArgumentError("invalid gas reaction coefficients or scalar densities");
    }
    const double burned_fuel = -cells[cell].density.fuel * std::expm1(-step_duration * source.burn_rate);
    const double fuel_burn_rate = burned_fuel / step_duration;
    auto rates = source.rates;
    rates.fuel -= fuel_burn_rate;
    rates.soot += source.soot_yield * fuel_burn_rate;
    rates.enthalpy += source.heat_per_fuel * fuel_burn_rate;
    fuel_burn_rates.push_back(fuel_burn_rate);
    heating.push_back({cells[cell].volume, rates.enthalpy, component});
    transport.push_back({cells[cell].volume, cells[cell].density, rates});
  }

  const auto thermal = AdvanceGasHeating(step_duration, heating, components);
  if (!thermal.ok()) {
    return thermal.status();
  }
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    pressure_cells[cell].target_divergence = thermal->target_divergence[cell];
    transport[cell].source_rate.enthalpy += thermal->enthalpy_pressure_work_rate[heating[cell].component];
  }

  const double ambient_specific_enthalpy = ambient.enthalpy / ambient.carrier;
  const auto residual_gravity_factor = [ambient_specific_enthalpy](const GasScalars& density) {
    // With constant cp, the ratio of specific enthalpies is T/T_ambient.
    const double temperature_ratio = (density.enthalpy / density.carrier) / ambient_specific_enthalpy;
    return density.soot / density.carrier - (temperature_ratio - 1.0);
  };
  for (std::size_t index = 0; index < links.size(); ++index) {
    const auto& link = links[index];
    if (link.open_area == 0.0) {
      continue;
    }
    const auto& tail = link.tail_cell_ordinal ? cells[*link.tail_cell_ordinal].density : ambient;
    const auto& head = link.head_cell_ordinal ? cells[*link.head_cell_ordinal].density : ambient;
    pressure_faces[index].inverse_density = 1.0 / (0.5 * tail.carrier + 0.5 * head.carrier);
    const double factor = 0.5 * residual_gravity_factor(tail) + 0.5 * residual_gravity_factor(head);
    const auto axis = static_cast<std::size_t>(faces[index].face_index.axis);
    pressure_faces[index].predicted_normal_velocity =
        faces[index].predicted_normal_velocity + step_duration * (gravity_components[axis] * factor);
  }
  auto projection = project({grid, step_duration, pressure_cells, pressure_faces}, {});
  if (!projection.ok()) {
    return projection.status();
  }
  if (projection->corrected_face_velocities.size() != links.size()
      || projection->cell_pressures.size() != cells.size()) {
    return absl::InternalError("gas projection returned inconsistent extents");
  }
  std::vector<GasTransportFace> transport_faces;
  transport_faces.reserve(links.size());
  for (std::size_t index = 0; index < links.size(); ++index) {
    const auto& link = links[index];
    if (link.open_area > 0.0) {
      transport_faces.push_back({
          .tail = link.tail_cell_ordinal,
          .head = link.head_cell_ordinal,
          .volume_flux = link.open_area * projection->corrected_face_velocities[index],
      });
    }
  }
  const auto scalars = TransportGasScalars(step_duration, transport, transport_faces, ambient);
  if (!scalars.ok()) {
    return scalars.status();
  }
  GasSubstep3D result{
      .cells = {cells.begin(), cells.end()},
      .projection = std::move(*projection),
      .outward_rate = scalars->outward_rate,
      .fuel_burn_rate = std::move(fuel_burn_rates),
  };
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const auto& density = scalars->density[cell];
    if (!PositiveFinite(density.carrier) || !PositiveFinite(density.enthalpy)) {
      return absl::FailedPreconditionError("updated gas carrier and enthalpy must stay positive");
    }
    result.cells[cell].density = density;
    result.cells[cell].absolute_pressure = thermal->component_pressure[heating[cell].component];
    result.cells[cell].gauge_pressure = result.projection.cell_pressures[cell];
  }
  return result;
}

}  // namespace fluidity
