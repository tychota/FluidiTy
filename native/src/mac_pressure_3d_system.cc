#include "mac_pressure_3d_system.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"

namespace fluidity::pressure_internal {
namespace {

constexpr std::size_t kMaxPressureCells = 4096;
constexpr std::size_t kMaxPressureFaces = 24576;
constexpr double kCompatibilityRoundoff = 32.0 * std::numeric_limits<double>::epsilon();
constexpr double kIdentityRoundoff = 64.0 * std::numeric_limits<double>::epsilon();
constexpr std::array kFaceAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};

bool IsFinitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

// Normal products have the usual FP64 relative error bound. In the
// subnormal range an inexact product can erase an O(1) defect after division
// by a small cell volume. Test exactness on normalized mantissas so the FMA
// residual itself cannot underflow.
bool IsRepresentablePressureProduct(double coefficient, double quantity, double product) {
  if (!std::isfinite(product)) {
    return false;
  }
  if (coefficient == 0.0 || quantity == 0.0 || std::abs(product) >= std::numeric_limits<double>::min()) {
    return true;
  }
  if (product == 0.0) {
    return false;
  }
  int coefficient_exponent = 0;
  int quantity_exponent = 0;
  const double coefficient_mantissa = std::frexp(coefficient, &coefficient_exponent);
  const double quantity_mantissa = std::frexp(quantity, &quantity_exponent);
  const double normalized_product = std::scalbn(product, -(coefficient_exponent + quantity_exponent));
  return std::fma(coefficient_mantissa, quantity_mantissa, -normalized_product) == 0.0;
}

auto CellSortKey(CellIndex3D index) {
  return std::tuple{index.z, index.y, index.x};
}

// Deterministic Neumaier summation retains small physical terms next to large
// cancelling fluxes. Callers check both signed and absolute sums for finiteness.
struct CompensatedSum {
  double sum = 0.0;
  double correction = 0.0;

  void Add(double term) {
    const double next_sum = sum + term;
    if (std::abs(sum) >= std::abs(term)) {
      correction += (sum - next_sum) + term;
    } else {
      correction += (term - next_sum) + sum;
    }
    sum = next_sum;
  }

  double Total() const {
    return sum + correction;
  }
};

struct FluxRow {
  CompensatedSum flux;
  CompensatedSum absolute_terms;

  void Add(double signed_term) {
    flux.Add(signed_term);
    absolute_terms.Add(std::abs(signed_term));
  }
};

template <typename Accumulate>
void ForEachIncidentCell(const FrozenMacPressureLink3D& link, Accumulate accumulate) {
  if (link.tail_cell_ordinal.has_value()) {
    accumulate(*link.tail_cell_ordinal, 1.0);
  }
  if (link.head_cell_ordinal.has_value()) {
    accumulate(*link.head_cell_ordinal, -1.0);
  }
}

absl::StatusOr<PreparedMacPressureCell3D>
FreezeCell(const MacPressureCell3D& cell, double step_duration, double full_cell_volume) {
  if (!IsFinitePositive(cell.fluid_volume) || cell.fluid_volume > full_cell_volume) {
    return absl::InvalidArgumentError("pressure cell volume must be positive and at most the grid cell volume");
  }
  if (!std::isfinite(cell.target_divergence) || !std::isfinite(cell.previous_pressure)) {
    return absl::InvalidArgumentError("pressure cell target divergence and previous pressure must be finite");
  }
  double compliance = 0.0;
  if (cell.bulk_modulus != std::numeric_limits<double>::infinity()) {
    if (!IsFinitePositive(cell.bulk_modulus)) {
      return absl::InvalidArgumentError("pressure bulk modulus must be positive");
    }
    const double modulus_times_step_duration = cell.bulk_modulus * step_duration;
    if (!IsFinitePositive(modulus_times_step_duration)) {
      return absl::OutOfRangeError("pressure bulk modulus times step duration is unrepresentable");
    }
    compliance = cell.fluid_volume / modulus_times_step_duration;
    if (!IsFinitePositive(compliance)) {
      return absl::OutOfRangeError("pressure compliance is unrepresentable");
    }
  }

  const double integrated_target_flux = cell.fluid_volume * cell.target_divergence;
  if (!IsRepresentablePressureProduct(cell.fluid_volume, cell.target_divergence, integrated_target_flux)) {
    return absl::OutOfRangeError("pressure cell integrated target flux is unrepresentable");
  }
  return PreparedMacPressureCell3D{
      .cell_index = cell.cell_index,
      .fluid_volume = cell.fluid_volume,
      .integrated_target_flux = integrated_target_flux,
      .compliance = compliance,
      .previous_pressure = cell.previous_pressure,
  };
}

// The grid owns each low face at its head index. Check ownership for all six
// incident faces before matching the supplied table, even if a face is sealed.
absl::StatusOr<std::vector<std::size_t>> CollectIncidentFaceSlots(const MacPressureInputs3D& inputs) {
  std::vector<std::size_t> incident_face_slots;
  incident_face_slots.reserve(6 * inputs.pressure_cells.size());
  for (const auto& cell : inputs.pressure_cells) {
    const auto [x, y, z] = cell.cell_index;
    const std::array coordinates{x, y, z};
    for (std::size_t normal = 0; normal < kFaceAxes.size(); ++normal) {
      const std::int64_t high_coordinate = static_cast<std::int64_t>(coordinates[normal]) + 1;
      if (high_coordinate > std::numeric_limits<std::int32_t>::max()) {
        return absl::OutOfRangeError("pressure cell high face exceeds the signed lattice");
      }
      auto high_coordinates = coordinates;
      high_coordinates[normal] = static_cast<std::int32_t>(high_coordinate);
      const std::array face_indices{
          FaceIndex3D{kFaceAxes[normal], {x, y, z}},
          FaceIndex3D{kFaceAxes[normal], {high_coordinates[0], high_coordinates[1], high_coordinates[2]}},
      };
      for (const FaceIndex3D face_index : face_indices) {
        const auto face_slot = inputs.mac_grid.FaceSlot(face_index);
        if (!face_slot.ok()) {
          return face_slot.status();
        }
        incident_face_slots.push_back(*face_slot);
      }
    }
  }
  std::ranges::sort(incident_face_slots);
  const auto duplicates = std::ranges::unique(incident_face_slots);
  incident_face_slots.erase(duplicates.begin(), duplicates.end());
  return incident_face_slots;
}

std::optional<std::size_t> FindCellOrdinal(std::span<const MacPressureCell3D> cells, CellIndex3D cell_index) {
  const auto key = CellSortKey(cell_index);
  const auto cell = std::ranges::lower_bound(cells, key, {}, [](const MacPressureCell3D& row) {
    return CellSortKey(row.cell_index);
  });
  if (cell == cells.end() || CellSortKey(cell->cell_index) != key) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(cell - cells.begin());
}

std::optional<std::size_t> FindTailCellOrdinal(std::span<const MacPressureCell3D> cells, FaceIndex3D face_index) {
  const auto [x, y, z] = face_index.lattice_index;
  std::array coordinates{x, y, z};
  // FaceSlot has validated the normal. An unrepresentable exterior tail at
  // INT32_MIN is absent, while the low face itself is still on the lattice.
  const auto normal = static_cast<std::size_t>(face_index.axis);
  const std::int64_t tail_coordinate = static_cast<std::int64_t>(coordinates[normal]) - 1;
  if (tail_coordinate < std::numeric_limits<std::int32_t>::min()) {
    return std::nullopt;
  }
  coordinates[normal] = static_cast<std::int32_t>(tail_coordinate);
  return FindCellOrdinal(cells, {coordinates[0], coordinates[1], coordinates[2]});
}

absl::StatusOr<FrozenMacPressureLink3D> FreezeLink(
    const MacPressureFace3D& face,
    std::size_t face_slot,
    double full_face_area,
    const MacPressureInputs3D& inputs
) {
  if (!std::isfinite(face.open_area) || face.open_area < 0.0 || face.open_area > full_face_area) {
    return absl::InvalidArgumentError("pressure face open area must be within the full face area");
  }
  if (!std::isfinite(face.inverse_density) || !std::isfinite(face.pressure_gradient_distance)
      || !std::isfinite(face.predicted_normal_velocity) || !std::isfinite(face.wall_normal_velocity)) {
    return absl::InvalidArgumentError("pressure face mobility, distance and velocities must be finite");
  }

  const auto [x, y, z] = face.face_index.lattice_index;
  const auto tail_cell_ordinal = FindTailCellOrdinal(inputs.pressure_cells, face.face_index);
  const auto head_cell_ordinal = FindCellOrdinal(inputs.pressure_cells, {x, y, z});
  const bool has_both_endpoints = tail_cell_ordinal.has_value() && head_cell_ordinal.has_value();
  const bool has_one_endpoint = tail_cell_ordinal.has_value() != head_cell_ordinal.has_value();
  if (const auto* internal = std::get_if<InternalMacPressureLink3D>(&face.pressure_link)) {
    if (!has_both_endpoints || !std::isfinite(internal->pressure_jump)) {
      return absl::InvalidArgumentError("internal pressure link requires two endpoints and a finite jump");
    }
  } else if (const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link)) {
    if (!has_one_endpoint || face.open_area == 0.0 || !std::isfinite(boundary->exterior_pressure)) {
      return absl::InvalidArgumentError("Dirichlet pressure link requires one endpoint, open area and finite pressure");
    }
  } else {
    const bool is_sealed = std::holds_alternative<SealedMacPressureLink3D>(face.pressure_link);
    if (!is_sealed || face.open_area != 0.0) {
      return absl::InvalidArgumentError("sealed pressure link requires zero open area");
    }
  }

  double pressure_to_velocity_scale = 0.0;
  double conductance = 0.0;
  if (face.open_area > 0.0) {
    if (face.inverse_density <= 0.0 || face.pressure_gradient_distance <= 0.0) {
      return absl::InvalidArgumentError("open pressure link requires positive mobility and gradient distance");
    }
    const double step_scaled_mobility = inputs.step_duration * face.inverse_density;
    pressure_to_velocity_scale = step_scaled_mobility / face.pressure_gradient_distance;
    conductance = face.open_area * pressure_to_velocity_scale;
    if (!IsFinitePositive(pressure_to_velocity_scale) || !IsFinitePositive(conductance)) {
      return absl::OutOfRangeError("open pressure link coefficients are unrepresentable");
    }
  }
  return FrozenMacPressureLink3D{
      .face_slot = face_slot,
      .tail_cell_ordinal = tail_cell_ordinal,
      .head_cell_ordinal = head_cell_ordinal,
      .open_area = face.open_area,
      .blocked_area = full_face_area - face.open_area,
      .pressure_to_velocity_scale = pressure_to_velocity_scale,
      .conductance = conductance,
      .predicted_normal_velocity = face.predicted_normal_velocity,
      .wall_normal_velocity = face.wall_normal_velocity,
      .pressure_link = face.pressure_link,
  };
}

double IntegratedWallFlux(const FrozenMacPressureLink3D& link) {
  return link.blocked_area * link.wall_normal_velocity;
}

// The original RHS uses one signed flux per physical contribution. Track
// their magnitudes before cancellation for the final identity roundoff bound.
absl::StatusOr<std::vector<FluxRow>> AssembleIntegratedRhs(
    std::span<const PreparedMacPressureCell3D> cells,
    std::span<const FrozenMacPressureLink3D> links
) {
  std::vector<FluxRow> rows(cells.size());
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    rows[cell].Add(cells[cell].integrated_target_flux);
    const auto& coefficients = cells[cell];
    const double prior_compliance_flux = coefficients.compliance * coefficients.previous_pressure;
    if (!IsRepresentablePressureProduct(
            coefficients.compliance,
            coefficients.previous_pressure,
            prior_compliance_flux
        )) {
      return absl::OutOfRangeError("previous pressure compliance flux is unrepresentable");
    }
    rows[cell].Add(prior_compliance_flux);
  }
  for (const auto& link : links) {
    const double predicted_fluid_flux = link.open_area * link.predicted_normal_velocity;
    if (!IsRepresentablePressureProduct(link.open_area, link.predicted_normal_velocity, predicted_fluid_flux)) {
      return absl::OutOfRangeError("predicted pressure face fluid flux is unrepresentable");
    }
    const double wall_flux = IntegratedWallFlux(link);
    if (!IsRepresentablePressureProduct(link.blocked_area, link.wall_normal_velocity, wall_flux)) {
      return absl::OutOfRangeError("prescribed pressure face wall flux is unrepresentable");
    }
    ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
      rows[cell].Add(-outward_sign * predicted_fluid_flux);
      rows[cell].Add(-outward_sign * wall_flux);
    });
    if (const auto* internal = std::get_if<InternalMacPressureLink3D>(&link.pressure_link)) {
      const double jump_flux = link.conductance * internal->pressure_jump;
      if (!IsRepresentablePressureProduct(link.conductance, internal->pressure_jump, jump_flux)) {
        return absl::OutOfRangeError("pressure jump flux is unrepresentable");
      }
      ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
        rows[cell].Add(-outward_sign * jump_flux);
      });
    } else if (const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&link.pressure_link)) {
      // The Dirichlet lift has the same sign at either orientation because the
      // known pressure at the missing endpoint carries the opposite incidence.
      const double exterior_pressure_flux = link.conductance * boundary->exterior_pressure;
      if (!IsRepresentablePressureProduct(link.conductance, boundary->exterior_pressure, exterior_pressure_flux)) {
        return absl::OutOfRangeError("exterior pressure flux is unrepresentable");
      }
      ForEachIncidentCell(link, [&](std::size_t cell, double) {
        rows[cell].Add(exterior_pressure_flux);
      });
    }
  }
  return rows;
}

std::size_t FindComponentRoot(std::span<const std::size_t> parent, std::size_t cell) {
  while (parent[cell] != cell) {
    cell = parent[cell];
  }
  return cell;
}

// Union only positive internal edges, always toward the lower ordinal. Each
// root is then the first row in its component and the candidate gauge.
absl::StatusOr<std::vector<std::size_t>> SelectGauges(
    std::span<const PreparedMacPressureCell3D> cells,
    std::span<const FrozenMacPressureLink3D> links,
    std::span<std::size_t> component_ordinals
) {
  std::iota(component_ordinals.begin(), component_ordinals.end(), std::size_t{0});
  for (const auto& link : links) {
    if (link.conductance > 0.0 && link.tail_cell_ordinal && link.head_cell_ordinal) {
      const auto tail_root = FindComponentRoot(component_ordinals, *link.tail_cell_ordinal);
      const auto head_root = FindComponentRoot(component_ordinals, *link.head_cell_ordinal);
      component_ordinals[std::max(tail_root, head_root)] = std::min(tail_root, head_root);
    }
  }

  struct ComponentBalance {
    bool anchored = false;
    FluxRow physical_balance;
  };
  std::vector<ComponentBalance> balances(cells.size());
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const auto root = FindComponentRoot(component_ordinals, cell);
    component_ordinals[cell] = root;
    balances[root].anchored |= cells[cell].compliance > 0.0;
  }
  for (const auto& link : links) {
    if (std::holds_alternative<DirichletMacPressureLink3D>(link.pressure_link)) {
      ForEachIncidentCell(link, [&](std::size_t cell, double) {
        balances[component_ordinals[cell]].anchored = true;
      });
    }
  }

  // Internal predicted and jump fluxes cancel on a whole component. A blocked
  // face contributes prescribed wall flux only when it crosses the component
  // boundary. Test physical compatibility before any gauge is selected.
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    auto& balance = balances[component_ordinals[cell]];
    if (!balance.anchored) {
      balance.physical_balance.Add(cells[cell].integrated_target_flux);
    }
  }
  for (const auto& link : links) {
    const bool stays_in_component = link.tail_cell_ordinal && link.head_cell_ordinal
        && component_ordinals[*link.tail_cell_ordinal] == component_ordinals[*link.head_cell_ordinal];
    if (stays_in_component) {
      continue;
    }
    const double wall_flux = IntegratedWallFlux(link);
    ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
      auto& balance = balances[component_ordinals[cell]];
      if (!balance.anchored) {
        balance.physical_balance.Add(-outward_sign * wall_flux);
      }
    });
  }

  std::vector<std::size_t> gauges;
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const auto& balance = balances[cell];
    if (component_ordinals[cell] != cell || balance.anchored) {
      continue;
    }
    const double physical_balance = balance.physical_balance.flux.Total();
    const double absolute_physical_terms = balance.physical_balance.absolute_terms.Total();
    if (!std::isfinite(physical_balance) || !std::isfinite(absolute_physical_terms)) {
      return absl::OutOfRangeError("pressure component physical balance is unrepresentable");
    }
    if (std::abs(physical_balance) > kCompatibilityRoundoff * absolute_physical_terms) {
      return absl::FailedPreconditionError("incompatible unanchored pressure component");
    }
    gauges.push_back(cell);
  }
  return gauges;
}

// Apply the original integrated operator as R diag(k) R^T plus compliance.
// Missing endpoints have zero unknown pressure because exterior pressures are
// already lifted into the RHS. All rows contribute, including gauge rows.
absl::StatusOr<std::vector<FluxRow>> ApplyOriginalPressureOperator(
    const PreparedMacPressureSystem3D& system,
    std::span<const double> candidate_cell_pressures
) {
  std::vector<FluxRow> rows(system.cells().size());
  for (std::size_t cell = 0; cell < rows.size(); ++cell) {
    const double compliance = system.cells()[cell].compliance;
    const double pressure = candidate_cell_pressures[cell];
    const double operator_compliance_flux = compliance * pressure;
    if (!IsRepresentablePressureProduct(compliance, pressure, operator_compliance_flux)) {
      return absl::OutOfRangeError("candidate pressure compliance flux is unrepresentable");
    }
    rows[cell].Add(operator_compliance_flux);
  }
  for (const auto& link : system.links()) {
    if (link.conductance == 0.0) {
      continue;
    }
    const double tail_pressure = link.tail_cell_ordinal ? candidate_cell_pressures[*link.tail_cell_ordinal] : 0.0;
    const double head_pressure = link.head_cell_ordinal ? candidate_cell_pressures[*link.head_cell_ordinal] : 0.0;
    const double pressure_difference = tail_pressure - head_pressure;
    const double operator_flux = link.conductance * pressure_difference;
    if (!IsRepresentablePressureProduct(link.conductance, pressure_difference, operator_flux)) {
      return absl::OutOfRangeError("original operator pressure flux is unrepresentable");
    }
    ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
      rows[cell].Add(outward_sign * operator_flux);
    });
  }
  return rows;
}

// Accumulate actual corrected fluid flux, blocked wall flux and compliance
// without using the algebraic RHS or the original operator product.
absl::StatusOr<std::vector<FluxRow>> CorrectFacesAndAccumulatePhysicalDefect(
    const PreparedMacPressureSystem3D& system,
    std::span<const double> candidate_cell_pressures,
    std::span<double> corrected_face_velocities
) {
  std::vector<FluxRow> rows(system.cells().size());
  for (std::size_t cell = 0; cell < rows.size(); ++cell) {
    const auto& coefficients = system.cells()[cell];
    rows[cell].Add(-coefficients.integrated_target_flux);
    if (coefficients.compliance > 0.0) {
      const double pressure_change = candidate_cell_pressures[cell] - coefficients.previous_pressure;
      const double compliance_change_flux = coefficients.compliance * pressure_change;
      if (!IsRepresentablePressureProduct(coefficients.compliance, pressure_change, compliance_change_flux)) {
        return absl::OutOfRangeError("pressure change compliance flux is unrepresentable");
      }
      rows[cell].Add(compliance_change_flux);
    }
  }

  const auto links = system.links();
  for (std::size_t face = 0; face < links.size(); ++face) {
    const auto& link = links[face];
    double corrected_normal_velocity = link.predicted_normal_velocity;
    if (link.open_area > 0.0) {
      double exterior_pressure = 0.0;
      double pressure_jump = 0.0;
      if (const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&link.pressure_link)) {
        exterior_pressure = boundary->exterior_pressure;
      } else if (const auto* internal = std::get_if<InternalMacPressureLink3D>(&link.pressure_link)) {
        pressure_jump = internal->pressure_jump;
      }
      const double tail_pressure =
          link.tail_cell_ordinal ? candidate_cell_pressures[*link.tail_cell_ordinal] : exterior_pressure;
      const double head_pressure =
          link.head_cell_ordinal ? candidate_cell_pressures[*link.head_cell_ordinal] : exterior_pressure;
      const double lifted_pressure_difference = tail_pressure - head_pressure + pressure_jump;
      const double normal_velocity_correction = link.pressure_to_velocity_scale * lifted_pressure_difference;
      if (!IsRepresentablePressureProduct(
              link.pressure_to_velocity_scale,
              lifted_pressure_difference,
              normal_velocity_correction
          )) {
        return absl::OutOfRangeError("pressure face velocity correction is unrepresentable");
      }
      corrected_normal_velocity += normal_velocity_correction;
    }
    if (!std::isfinite(corrected_normal_velocity)) {
      return absl::OutOfRangeError("corrected pressure velocity is unrepresentable");
    }
    corrected_face_velocities[face] = corrected_normal_velocity;
    const double corrected_fluid_flux = link.open_area * corrected_normal_velocity;
    if (!IsRepresentablePressureProduct(link.open_area, corrected_normal_velocity, corrected_fluid_flux)) {
      return absl::OutOfRangeError("corrected pressure face fluid flux is unrepresentable");
    }
    const double wall_flux = IntegratedWallFlux(link);
    ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
      rows[cell].Add(outward_sign * corrected_fluid_flux);
      rows[cell].Add(outward_sign * wall_flux);
    });
  }
  return rows;
}

}  // namespace

std::span<const PreparedMacPressureCell3D> PreparedMacPressureSystem3D::cells() const {
  return cells_;
}

std::span<const FrozenMacPressureLink3D> PreparedMacPressureSystem3D::links() const {
  return links_;
}

std::span<const double> PreparedMacPressureSystem3D::integrated_rhs() const {
  return integrated_rhs_;
}

std::span<const double> PreparedMacPressureSystem3D::rhs_absolute_term_sums() const {
  return rhs_absolute_term_sums_;
}

std::span<const std::size_t> PreparedMacPressureSystem3D::component_ordinals() const {
  return component_ordinals_;
}

std::span<const std::size_t> PreparedMacPressureSystem3D::gauge_cell_ordinals() const {
  return gauge_cell_ordinals_;
}

absl::StatusOr<PreparedMacPressureSystem3D> PrepareMacPressureSystem3D(const MacPressureInputs3D& inputs) {
  if (inputs.pressure_cells.size() > kMaxPressureCells || inputs.pressure_faces.size() > kMaxPressureFaces) {
    return absl::ResourceExhaustedError("3D pressure system exceeds cell or face capacity");
  }
  if (inputs.pressure_cells.empty() || inputs.pressure_faces.empty()) {
    return absl::InvalidArgumentError("3D pressure system requires cells and their incident faces");
  }
  if (!IsFinitePositive(inputs.step_duration)) {
    return absl::InvalidArgumentError("pressure step duration must be positive and finite");
  }

  const double cell_width = inputs.mac_grid.geometry().cell_width;
  const double full_face_area = cell_width * cell_width;
  const double full_cell_volume = full_face_area * cell_width;
  if (!IsFinitePositive(full_face_area) || !IsFinitePositive(full_cell_volume)) {
    return absl::OutOfRangeError("pressure grid face area or cell volume is unrepresentable");
  }

  PreparedMacPressureSystem3D system;
  system.cells_.reserve(inputs.pressure_cells.size());
  for (const auto& cell : inputs.pressure_cells) {
    if (!system.cells_.empty() && CellSortKey(system.cells_.back().cell_index) >= CellSortKey(cell.cell_index)) {
      return absl::InvalidArgumentError("pressure cells must be unique and ordered by (z,y,x)");
    }
    const auto coefficients = FreezeCell(cell, inputs.step_duration, full_cell_volume);
    if (!coefficients.ok()) {
      return coefficients.status();
    }
    system.cells_.push_back(*coefficients);
  }

  const auto incident_face_slots = CollectIncidentFaceSlots(inputs);
  if (!incident_face_slots.ok()) {
    return incident_face_slots.status();
  }
  if (incident_face_slots->size() != inputs.pressure_faces.size()) {
    return absl::InvalidArgumentError("pressure faces must exactly cover all incident faces");
  }
  system.links_.reserve(inputs.pressure_faces.size());
  for (std::size_t face = 0; face < inputs.pressure_faces.size(); ++face) {
    const auto& pressure_face = inputs.pressure_faces[face];
    const auto face_slot = inputs.mac_grid.FaceSlot(pressure_face.face_index);
    if (!face_slot.ok()) {
      return face_slot.status();
    }
    if (*face_slot != (*incident_face_slots)[face]) {
      return absl::InvalidArgumentError("pressure faces must match the incident faces in unique owned slot order");
    }
    const auto link = FreezeLink(pressure_face, *face_slot, full_face_area, inputs);
    if (!link.ok()) {
      return link.status();
    }
    system.links_.push_back(*link);
  }

  const auto rhs_rows = AssembleIntegratedRhs(system.cells(), system.links());
  if (!rhs_rows.ok()) {
    return rhs_rows.status();
  }
  system.integrated_rhs_.reserve(rhs_rows->size());
  system.rhs_absolute_term_sums_.reserve(rhs_rows->size());
  for (const auto& row : *rhs_rows) {
    const double integrated_rhs = row.flux.Total();
    const double absolute_rhs_terms = row.absolute_terms.Total();
    if (!std::isfinite(integrated_rhs) || !std::isfinite(absolute_rhs_terms)) {
      return absl::OutOfRangeError("pressure integrated RHS or its terms are unrepresentable");
    }
    system.integrated_rhs_.push_back(integrated_rhs);
    system.rhs_absolute_term_sums_.push_back(absolute_rhs_terms);
  }
  system.component_ordinals_.resize(system.cells().size());
  auto gauges = SelectGauges(system.cells(), system.links(), system.component_ordinals_);
  if (!gauges.ok()) {
    return gauges.status();
  }
  system.gauge_cell_ordinals_ = std::move(*gauges);
  return system;
}

absl::StatusOr<MacPressureProjection3D>
EvaluateMacPressure3D(const PreparedMacPressureSystem3D& system, std::span<const double> candidate_cell_pressures) {
  const auto cells = system.cells();
  if (cells.empty() || candidate_cell_pressures.size() != cells.size()) {
    return absl::InvalidArgumentError("candidate pressure extent must match the prepared system");
  }
  for (const double pressure : candidate_cell_pressures) {
    if (!std::isfinite(pressure)) {
      return absl::InvalidArgumentError("candidate cell pressures must be finite");
    }
  }

  const auto operator_rows = ApplyOriginalPressureOperator(system, candidate_cell_pressures);
  if (!operator_rows.ok()) {
    return operator_rows.status();
  }
  MacPressureProjection3D projection;
  projection.corrected_face_velocities.resize(system.links().size());
  const auto physical_rows =
      CorrectFacesAndAccumulatePhysicalDefect(system, candidate_cell_pressures, projection.corrected_face_velocities);
  if (!physical_rows.ok()) {
    return physical_rows.status();
  }

  const auto integrated_rhs = system.integrated_rhs();
  const auto rhs_absolute_term_sums = system.rhs_absolute_term_sums();
  projection.true_residuals.reserve(cells.size());
  projection.cell_constraint_defects.reserve(cells.size());
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const double residual = integrated_rhs[cell] - (*operator_rows)[cell].flux.Total();
    const double fluid_volume = cells[cell].fluid_volume;
    const double volume_scaled_residual = std::abs(residual) / fluid_volume;
    const double defect = (*physical_rows)[cell].flux.Total() / fluid_volume;
    const double identity_error = fluid_volume * defect + residual;
    const double absolute_row_terms = rhs_absolute_term_sums[cell] + (*operator_rows)[cell].absolute_terms.Total()
        + (*physical_rows)[cell].absolute_terms.Total();
    if (!std::isfinite(volume_scaled_residual) || !std::isfinite(defect) || !std::isfinite(identity_error)
        || !std::isfinite(absolute_row_terms)) {
      return absl::OutOfRangeError("pressure residual, defect or identity arithmetic is unrepresentable");
    }
    if (std::abs(identity_error) > kIdentityRoundoff * absolute_row_terms) {
      return absl::InternalError("pressure residual disagrees with the independently corrected flux defect");
    }

    projection.true_residuals.push_back(residual);
    projection.cell_constraint_defects.push_back(defect);
    auto& diagnostics = projection.diagnostics;
    // hypot avoids squaring tiny or very large integrated residuals.
    diagnostics.true_residual_l2 = std::hypot(diagnostics.true_residual_l2, residual);
    diagnostics.max_volume_scaled_residual = std::max(diagnostics.max_volume_scaled_residual, volume_scaled_residual);
    diagnostics.max_cell_constraint_defect = std::max(diagnostics.max_cell_constraint_defect, std::abs(defect));
  }
  if (!std::isfinite(projection.diagnostics.true_residual_l2)) {
    return absl::OutOfRangeError("pressure residual norm is unrepresentable");
  }

  projection.cell_pressures.assign(candidate_cell_pressures.begin(), candidate_cell_pressures.end());
  projection.integrated_rhs.assign(integrated_rhs.begin(), integrated_rhs.end());
  const auto gauges = system.gauge_cell_ordinals();
  projection.gauge_cell_ordinals.assign(gauges.begin(), gauges.end());
  return projection;
}

}  // namespace fluidity::pressure_internal
