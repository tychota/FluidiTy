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
#include "mac_pressure_3d_arithmetic.h"

namespace fluidity::pressure_internal {
namespace {

using pressure_internal::arithmetic::ApplyOriginalPressureOperatorInto;
using pressure_internal::arithmetic::AssembleIntegratedRhsInto;
using pressure_internal::arithmetic::CellSortKey;
using pressure_internal::arithmetic::ComponentBalance;
using pressure_internal::arithmetic::CorrectFacesAndAccumulatePhysicalDefectInto;
using pressure_internal::arithmetic::FluxRow;
using pressure_internal::arithmetic::FreezeCell;
using pressure_internal::arithmetic::FreezeLink;
using pressure_internal::arithmetic::IsFinitePositive;
using pressure_internal::arithmetic::kFaceAxes;
using pressure_internal::arithmetic::kIdentityRoundoff;
using pressure_internal::arithmetic::kMaxPressureCells;
using pressure_internal::arithmetic::kMaxPressureFaces;
using pressure_internal::arithmetic::SelectGaugesInto;

// Check ownership for all six incident faces, even when sealed.
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

absl::StatusOr<std::vector<FluxRow>> AssembleIntegratedRhs(
    std::span<const PreparedMacPressureCell3D> cells,
    std::span<const FrozenMacPressureLink3D> links
) {
  std::vector<FluxRow> rows(cells.size());
  const auto status = AssembleIntegratedRhsInto(cells, links, rows);
  if (!status.ok()) {
    return status;
  }
  return rows;
}
absl::StatusOr<std::vector<std::size_t>> SelectGauges(
    std::span<const PreparedMacPressureCell3D> cells,
    std::span<const FrozenMacPressureLink3D> links,
    std::span<std::size_t> components
) {
  std::vector<ComponentBalance> balances(cells.size());
  std::vector<std::size_t> gauges(cells.size());
  const auto count = SelectGaugesInto(cells, links, components, balances, gauges);
  if (!count.ok()) {
    return count.status();
  }
  gauges.resize(*count);
  return gauges;
}
absl::StatusOr<std::vector<FluxRow>>
ApplyOriginalPressureOperator(const PreparedMacPressureSystem3D& system, std::span<const double> pressure) {
  std::vector<FluxRow> rows(system.cells().size());
  const auto status = ApplyOriginalPressureOperatorInto(system, pressure, rows);
  if (!status.ok()) {
    return status;
  }
  return rows;
}
absl::StatusOr<std::vector<FluxRow>> CorrectFacesAndAccumulatePhysicalDefect(
    const PreparedMacPressureSystem3D& system,
    std::span<const double> pressure,
    std::span<double> velocities
) {
  std::vector<FluxRow> rows(system.cells().size());
  const auto status = CorrectFacesAndAccumulatePhysicalDefectInto(system, pressure, velocities, rows);
  if (!status.ok()) {
    return status;
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

  const auto status = system.Finalize();
  if (!status.ok()) {
    return status;
  }
  return system;
}

absl::Status PreparedMacPressureSystem3D::Finalize() {
  const auto rhs_rows = AssembleIntegratedRhs(cells(), links());
  if (!rhs_rows.ok()) {
    return rhs_rows.status();
  }
  integrated_rhs_.reserve(rhs_rows->size());
  rhs_absolute_term_sums_.reserve(rhs_rows->size());
  for (const auto& row : *rhs_rows) {
    const double integrated_rhs = row.flux.Total();
    const double absolute_rhs_terms = row.absolute_terms.Total();
    if (!std::isfinite(integrated_rhs) || !std::isfinite(absolute_rhs_terms)) {
      return absl::OutOfRangeError("pressure integrated RHS or its terms are unrepresentable");
    }
    integrated_rhs_.push_back(integrated_rhs);
    rhs_absolute_term_sums_.push_back(absolute_rhs_terms);
  }
  component_ordinals_.resize(cells().size());
  auto gauges = SelectGauges(cells(), links(), component_ordinals_);
  if (!gauges.ok()) {
    return gauges.status();
  }
  gauge_cell_ordinals_ = std::move(*gauges);
  return absl::OkStatus();
}

absl::StatusOr<PreparedMacPressureSystem3D> PrepareExplicitMacPressureSystem3D(
    double step_duration,
    std::span<const MacPressureCell3D> pressure_cells,
    std::span<const ExplicitMacPressureFace3D> pressure_faces
) {
  if (pressure_cells.size() > kMaxPressureCells || pressure_faces.size() > kMaxPressureFaces) {
    return absl::ResourceExhaustedError("3D pressure graph exceeds cell or face capacity");
  }
  if (pressure_cells.empty()) {
    return absl::InvalidArgumentError("3D pressure graph requires cells");
  }
  if (!IsFinitePositive(step_duration)) {
    return absl::InvalidArgumentError("pressure step duration must be positive and finite");
  }

  PreparedMacPressureSystem3D system;
  system.cells_.reserve(pressure_cells.size());
  for (const auto& cell : pressure_cells) {
    if (!system.cells_.empty() && CellSortKey(system.cells_.back().cell_index) >= CellSortKey(cell.cell_index)) {
      return absl::InvalidArgumentError("pressure cells must be unique and ordered by (z,y,x)");
    }
    const auto coefficients = FreezeCell(cell, step_duration, std::numeric_limits<double>::infinity());
    if (!coefficients.ok()) {
      return coefficients.status();
    }
    system.cells_.push_back(*coefficients);
  }

  system.links_.reserve(pressure_faces.size());
  for (const auto& face : pressure_faces) {
    if (!system.links_.empty() && system.links_.back().face_slot >= face.face_slot) {
      return absl::InvalidArgumentError("pressure graph faces must be in unique owned slot order");
    }
    if (face.tail_cell_ordinal >= pressure_cells.size() || face.head_cell_ordinal >= pressure_cells.size()
        || face.tail_cell_ordinal == face.head_cell_ordinal) {
      return absl::InvalidArgumentError("pressure graph face requires two distinct in-range endpoints");
    }
    if (!IsFinitePositive(face.open_area) || !IsFinitePositive(face.inverse_density)
        || !IsFinitePositive(face.pressure_gradient_distance) || !std::isfinite(face.predicted_normal_velocity)) {
      return absl::InvalidArgumentError("pressure graph face coefficients are invalid");
    }

    const double step_scaled_mobility = step_duration * face.inverse_density;
    if (!arithmetic::IsRepresentablePressureProduct(step_duration, face.inverse_density, step_scaled_mobility)) {
      return absl::OutOfRangeError("pressure graph step-scaled mobility is unrepresentable");
    }
    const double pressure_to_velocity_scale = step_scaled_mobility / face.pressure_gradient_distance;
    const double conductance = face.open_area * pressure_to_velocity_scale;
    const double predicted_fluid_flux = face.open_area * face.predicted_normal_velocity;
    if (!std::isnormal(pressure_to_velocity_scale) || !IsFinitePositive(conductance)
        || !arithmetic::IsRepresentablePressureProduct(face.open_area, pressure_to_velocity_scale, conductance)
        || !arithmetic::IsRepresentablePressureProduct(
            face.open_area,
            face.predicted_normal_velocity,
            predicted_fluid_flux
        )) {
      return absl::OutOfRangeError("pressure graph face arithmetic is unrepresentable");
    }
    system.links_.push_back({
        .face_slot = face.face_slot,
        .tail_cell_ordinal = face.tail_cell_ordinal,
        .head_cell_ordinal = face.head_cell_ordinal,
        .open_area = face.open_area,
        .blocked_area = 0.0,
        .pressure_to_velocity_scale = pressure_to_velocity_scale,
        .conductance = conductance,
        .predicted_normal_velocity = face.predicted_normal_velocity,
        .prescribed_wall_flux = 0.0,
        .pressure_link = InternalMacPressureLink3D{.pressure_jump = 0.0},
    });
  }

  const auto status = system.Finalize();
  if (!status.ok()) {
    return status;
  }
  system.is_explicit_graph_ = true;
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

  const auto original_operator_rows = ApplyOriginalPressureOperator(system, candidate_cell_pressures);
  if (!original_operator_rows.ok()) {
    return original_operator_rows.status();
  }
  MacPressureProjection3D projection;
  projection.corrected_face_velocities.resize(system.links().size());
  const auto physical_defect_rows =
      CorrectFacesAndAccumulatePhysicalDefect(system, candidate_cell_pressures, projection.corrected_face_velocities);
  if (!physical_defect_rows.ok()) {
    return physical_defect_rows.status();
  }

  const auto integrated_rhs = system.integrated_rhs();
  const auto rhs_absolute_term_sums = system.rhs_absolute_term_sums();
  projection.true_residuals.reserve(cells.size());
  projection.cell_constraint_defects.reserve(cells.size());
  for (std::size_t pressure_cell_ordinal = 0; pressure_cell_ordinal < cells.size(); ++pressure_cell_ordinal) {
    const double integrated_residual =
        integrated_rhs[pressure_cell_ordinal] - (*original_operator_rows)[pressure_cell_ordinal].flux.Total();
    const double fluid_volume = cells[pressure_cell_ordinal].fluid_volume;
    const double volume_scaled_residual = std::abs(integrated_residual) / fluid_volume;
    const double volume_scaled_physical_defect =
        (*physical_defect_rows)[pressure_cell_ordinal].flux.Total() / fluid_volume;
    const double integrated_identity_error = fluid_volume * volume_scaled_physical_defect + integrated_residual;
    const double absolute_row_term_sum = rhs_absolute_term_sums[pressure_cell_ordinal]
        + (*original_operator_rows)[pressure_cell_ordinal].absolute_terms.Total()
        + (*physical_defect_rows)[pressure_cell_ordinal].absolute_terms.Total();
    if (!std::isfinite(volume_scaled_residual) || !std::isfinite(volume_scaled_physical_defect)
        || !std::isfinite(integrated_identity_error) || !std::isfinite(absolute_row_term_sum)) {
      return absl::OutOfRangeError("pressure residual, defect or identity arithmetic is unrepresentable");
    }
    if (std::abs(integrated_identity_error) > kIdentityRoundoff * absolute_row_term_sum) {
      return absl::InternalError("pressure residual disagrees with the independently corrected flux defect");
    }

    projection.true_residuals.push_back(integrated_residual);
    projection.cell_constraint_defects.push_back(volume_scaled_physical_defect);
    auto& diagnostics = projection.diagnostics;
    // hypot avoids squaring tiny or very large integrated residuals.
    diagnostics.true_residual_l2 = std::hypot(diagnostics.true_residual_l2, integrated_residual);
    diagnostics.max_volume_scaled_residual = std::max(diagnostics.max_volume_scaled_residual, volume_scaled_residual);
    diagnostics.max_cell_constraint_defect =
        std::max(diagnostics.max_cell_constraint_defect, std::abs(volume_scaled_physical_defect));
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
