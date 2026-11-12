#include "fluidity/host/mac_pressure_workspace_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

#include "fluidity/host/viscosity.h"
#include "mac_pressure_3d_arithmetic.h"
#include "mac_pressure_3d_workspace_internal.h"

namespace fluidity {
namespace {
using pressure_internal::FrozenMacPressureLink3D;
using pressure_internal::kMaximumPressureLevels3D;
using pressure_internal::kNoPressureUnknown3D;
using pressure_internal::MacPressureEdge3D;
using pressure_internal::MacPressureHierarchyView3D;
using pressure_internal::MacPressureLevelView3D;
using pressure_internal::MacPressureNeighbor3D;
using pressure_internal::MacPressureScratch3D;
using pressure_internal::MacPressureScratchMark3D;
using pressure_internal::PreparedMacPressureCell3D;
using pressure_internal::PreparedMacPressureView3D;
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

#define FLUIDITY_PRESSURE_TAKE(name, type, count) \
  auto name##_result = scratch.Take<type>(count); \
  if (!name##_result.ok()) { \
    return name##_result.status(); \
  } \
  auto name = *name##_result

absl::StatusOr<pressure_internal::PreparedMacPressureUploadView3D> PrepareUploadForSystem(
    const PreparedMacPressureView3D& system,
    std::span<const MacPressureFace3D> uniform_faces,
    MacPressureScratch3D& scratch,
    bool seed_uniform_dirichlet_rest_pressure
);

struct BorrowedInputs {
  SparseMacGridView3D mac_grid;
  double step_duration;
  std::span<const MacPressureCell3D> pressure_cells;
  std::span<const MacPressureFace3D> pressure_faces;
};

absl::StatusOr<PreparedMacPressureView3D> Prepare(const BorrowedInputs& inputs, MacPressureScratch3D& scratch) {
  const std::size_t cell_count = inputs.pressure_cells.size();
  const std::size_t face_count = inputs.pressure_faces.size();
  if (cell_count == 0 || face_count == 0 || !IsFinitePositive(inputs.step_duration)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const double cell_width = inputs.mac_grid.geometry().cell_width;
  const double full_face_area = cell_width * cell_width;
  const double full_cell_volume = full_face_area * cell_width;
  if (!IsFinitePositive(full_face_area) || !IsFinitePositive(full_cell_volume)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  FLUIDITY_PRESSURE_TAKE(cells, PreparedMacPressureCell3D, cell_count);
  FLUIDITY_PRESSURE_TAKE(links, FrozenMacPressureLink3D, face_count);
  FLUIDITY_PRESSURE_TAKE(integrated_rhs, double, cell_count);
  FLUIDITY_PRESSURE_TAKE(rhs_absolute_term_sums, double, cell_count);
  FLUIDITY_PRESSURE_TAKE(component_ordinals, std::size_t, cell_count);
  FLUIDITY_PRESSURE_TAKE(gauge_cell_ordinals, std::size_t, cell_count);
  PreparedMacPressureView3D system{
      .cell_records = cells,
      .link_records = links,
      .rhs = integrated_rhs,
      .rhs_absolute_terms = rhs_absolute_term_sums,
      .components = component_ordinals,
      .gauges = gauge_cell_ordinals,
  };
  MacPressureScratchMark3D temporary(scratch);
  FLUIDITY_PRESSURE_TAKE(incident_face_slots, std::size_t, 6 * cell_count);
  std::size_t incident_face_count = 0;
  for (std::size_t cell_ordinal = 0; cell_ordinal < cell_count; ++cell_ordinal) {
    const auto& cell = inputs.pressure_cells[cell_ordinal];
    if (cell_ordinal != 0
        && CellSortKey(inputs.pressure_cells[cell_ordinal - 1].cell_index) >= CellSortKey(cell.cell_index)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const auto frozen = FreezeCell(cell, inputs.step_duration, full_cell_volume);
    if (!frozen.ok()) {
      return frozen.status();
    }
    cells[cell_ordinal] = *frozen;
    const std::array cell_lattice_indices{cell.cell_index.x, cell.cell_index.y, cell.cell_index.z};
    for (std::size_t normal_axis = 0; normal_axis < 3; ++normal_axis) {
      if (cell_lattice_indices[normal_axis] == std::numeric_limits<std::int32_t>::max()) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      for (std::int32_t normal_side_offset = 0; normal_side_offset < 2; ++normal_side_offset) {
        auto face_lattice_indices = cell_lattice_indices;
        face_lattice_indices[normal_axis] += normal_side_offset;
        const auto face_slot = inputs.mac_grid.FaceSlot(
            {kFaceAxes[normal_axis], {face_lattice_indices[0], face_lattice_indices[1], face_lattice_indices[2]}}
        );
        if (!face_slot.ok()) {
          return face_slot.status();
        }
        incident_face_slots[incident_face_count++] = *face_slot;
      }
    }
  }
  std::ranges::sort(incident_face_slots);
  incident_face_count =
      static_cast<std::size_t>(std::ranges::unique(incident_face_slots).begin() - incident_face_slots.begin());
  if (incident_face_count != face_count) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  for (std::size_t face_ordinal = 0; face_ordinal < face_count; ++face_ordinal) {
    const auto& face = inputs.pressure_faces[face_ordinal];
    const auto face_slot = inputs.mac_grid.FaceSlot(face.face_index);
    if (!face_slot.ok()) {
      return face_slot.status();
    }
    if (*face_slot != incident_face_slots[face_ordinal]) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const auto link = FreezeLink(face, *face_slot, full_face_area, inputs);
    if (!link.ok()) {
      return link.status();
    }
    links[face_ordinal] = *link;
  }
  FLUIDITY_PRESSURE_TAKE(rhs_rows, FluxRow, cell_count);
  auto status = AssembleIntegratedRhsInto(cells, links, rhs_rows);
  if (!status.ok()) {
    return status;
  }
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    integrated_rhs[cell] = rhs_rows[cell].flux.Total();
    rhs_absolute_term_sums[cell] = rhs_rows[cell].absolute_terms.Total();
    if (!std::isfinite(integrated_rhs[cell]) || !std::isfinite(rhs_absolute_term_sums[cell])) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
  }
  FLUIDITY_PRESSURE_TAKE(component_balances, ComponentBalance, cell_count);
  const auto gauge_count = SelectGaugesInto(cells, links, component_ordinals, component_balances, gauge_cell_ordinals);
  if (!gauge_count.ok()) {
    return gauge_count.status();
  }
  system.gauges = gauge_cell_ordinals.first(*gauge_count);
  return system;
}

// Inspect only checked records. With zero compliance and no gauge, every
// connected component has a Dirichlet anchor.
std::optional<double> UniformDirichletRestPressure(const PreparedMacPressureView3D& system) {
  if (system.cells().empty() || !system.gauge_cell_ordinals().empty()) {
    return std::nullopt;
  }
  for (const auto& cell : system.cells()) {
    if (cell.compliance != 0 || cell.previous_pressure != 0 || cell.integrated_target_flux != 0) {
      return std::nullopt;
    }
  }

  std::optional<double> exterior_pressure;
  for (const auto& link : system.links()) {
    if (link.predicted_normal_velocity != 0 || link.prescribed_wall_flux != 0) {
      return std::nullopt;
    }
    const auto* internal = std::get_if<InternalMacPressureLink3D>(&link.pressure_link);
    if (internal != nullptr && internal->pressure_jump != 0) {
      return std::nullopt;
    }
    const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&link.pressure_link);
    if (boundary == nullptr) {
      continue;
    }
    if (!exterior_pressure.has_value()) {
      exterior_pressure = boundary->exterior_pressure;
    } else if (boundary->exterior_pressure != *exterior_pressure) {
      return std::nullopt;
    }
  }

  // At the constant iterate, internal differences and physical flux vanish.
  // The original operator has the same Dirichlet terms in the same order as
  // the checked RHS, so the residual/defect identity adds their magnitudes twice.
  for (const double rhs_absolute_terms : system.rhs_absolute_term_sums()) {
    if (!std::isfinite(rhs_absolute_terms + rhs_absolute_terms)) {
      return std::nullopt;
    }
  }
  return exterior_pressure;
}

// Cast only finite values; a nonzero FP64 residual or coarse correction must
// not disappear on entering the fine FP32 iteration.
absl::StatusOr<float> FineValue(double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  const float rounded = static_cast<float>(value);
  if (!std::isfinite(rounded) || (value != 0.0 && rounded == 0.0F)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return rounded;
}

// The fine call uses float for every multiply/add and difference. Coarse calls
// use double. Edge differences retain the graph's constant mode and avoid
// cancellation between independently rounded diagonal and off-diagonal terms.
template <std::floating_point Scalar>
absl::Status
ApplyLevel(const MacPressureLevelView3D& level, std::span<const Scalar> pressure, std::span<Scalar> integrated_flux) {
  for (std::size_t unknown = 0; unknown < pressure.size(); ++unknown) {
    Scalar row_flux = static_cast<Scalar>(level.anchor_diagonal[unknown]) * pressure[unknown];
    for (std::size_t offset = level.row_offsets[unknown]; offset < level.row_offsets[unknown + 1]; ++offset) {
      const auto neighbor = level.neighbors[offset];
      const Scalar pressure_difference = pressure[unknown] - pressure[neighbor.unknown_ordinal];
      const Scalar edge_flux =
          static_cast<Scalar>(level.edges[neighbor.edge_ordinal].conductance) * pressure_difference;
      row_flux += edge_flux;
    }
    if (!std::isfinite(row_flux)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    integrated_flux[unknown] = row_flux;
  }
  return absl::OkStatus();
}

// Every dot visits unknowns in ascending ordinal order and multiplies after
// promotion to FP64, independently of the FP32 iterative vector operations.
absl::StatusOr<double> Dot(std::span<const float> left, std::span<const float> right) {
  double product_sum = 0.0;
  for (std::size_t unknown = 0; unknown < left.size(); ++unknown) {
    const double product = static_cast<double>(left[unknown]) * static_cast<double>(right[unknown]);
    product_sum += product;
  }
  if (!std::isfinite(product_sum)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return product_sum;
}

absl::Status CopyTrueResidual(
    const MacPressureProjectionView3D& projection,
    const MacPressureHierarchyView3D& hierarchy,
    std::span<float> residual
) {
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const auto fine_residual = FineValue(projection.true_residuals[hierarchy.unknown_to_cell[unknown]]);
    if (!fine_residual.ok()) {
      return fine_residual.status();
    }
    residual[unknown] = *fine_residual;
  }
  return absl::OkStatus();
}

bool RecursiveResidualPasses(
    const PreparedMacPressureView3D& system,
    const MacPressureHierarchyView3D& hierarchy,
    std::span<const float> residual,
    const pressure_internal::MacPressureStoppingCriteria3D& criteria
) {
  double residual_norm = 0.0;
  double maximum_local_residual = 0.0;
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const double flux_residual = residual[unknown];
    residual_norm = std::hypot(residual_norm, flux_residual);
    const double local_residual =
        std::abs(flux_residual) / system.cells()[hierarchy.unknown_to_cell[unknown]].fluid_volume;
    maximum_local_residual = std::max(maximum_local_residual, local_residual);
  }
  return residual_norm <= criteria.integrated_residual_limit && maximum_local_residual <= criteria.volume_scaled_limit;
}

// A fixed linear symmetric V-cycle: one 2/3 weighted Jacobi sweep before and
// after P A_c^-1 Pᵀ correction. Both sweeps use the same mirrored operator and
// diagonal. Terminal blocks use the precomputed FP64 Cholesky factors.
template <std::floating_point Scalar>
absl::Status ApplyVcycle(
    const MacPressureHierarchyView3D& hierarchy,
    std::size_t level_ordinal,
    std::span<const Scalar> residual,
    std::span<Scalar> correction,
    MacPressureScratch3D& scratch
) {
  const auto& level = hierarchy.levels[level_ordinal];
  MacPressureScratchMark3D temporary(scratch);
  std::fill(correction.begin(), correction.end(), Scalar{0});
  if (level_ordinal + 1 == hierarchy.levels.size()) {
    std::size_t factor_offset = 0;
    std::array<double, 32> block_correction{};
    for (std::size_t block = 0; block + 1 < level.terminal_offsets.size(); ++block) {
      const std::size_t begin = level.terminal_offsets[block];
      const std::size_t count = level.terminal_offsets[block + 1] - begin;
      const auto unknowns = level.terminal_unknowns.subspan(begin, count);
      const auto factor = level.terminal_factors.subspan(factor_offset, count * count);
      factor_offset += count * count;
      for (std::size_t row = 0; row < count; ++row) {
        double forward_rhs = residual[unknowns[row]];
        for (std::size_t column = 0; column < row; ++column) {
          forward_rhs -= factor[row * count + column] * block_correction[column];
        }
        block_correction[row] = forward_rhs / factor[row * count + row];
      }
      for (std::size_t reverse_row = count; reverse_row > 0; --reverse_row) {
        const std::size_t row = reverse_row - 1;
        double backward_rhs = block_correction[row];
        for (std::size_t column = row + 1; column < count; ++column) {
          backward_rhs -= factor[column * count + row] * block_correction[column];
        }
        block_correction[row] = backward_rhs / factor[row * count + row];
        if (!std::isfinite(block_correction[row])) {
          return absl::Status(absl::StatusCode::kOutOfRange, {});
        }
        if constexpr (std::same_as<Scalar, float>) {
          const auto fine_correction = FineValue(block_correction[row]);
          if (!fine_correction.ok()) {
            return fine_correction.status();
          }
          correction[unknowns[row]] = *fine_correction;
        } else {
          correction[unknowns[row]] = block_correction[row];
        }
      }
    }
    return absl::OkStatus();
  }

  constexpr Scalar kJacobiWeight = Scalar{2} / Scalar{3};
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    correction[unknown] = kJacobiWeight * (residual[unknown] / static_cast<Scalar>(level.diagonal[unknown]));
  }
  FLUIDITY_PRESSURE_TAKE(correction_flux, Scalar, residual.size());
  auto status = ApplyLevel<Scalar>(level, correction, correction_flux);
  if (!status.ok()) {
    return status;
  }
  FLUIDITY_PRESSURE_TAKE(coarse_residual, double, hierarchy.levels[level_ordinal + 1].coordinates.size());
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const Scalar remaining_flux = residual[unknown] - correction_flux[unknown];
    coarse_residual[level.aggregate_ordinals[unknown]] += static_cast<double>(remaining_flux);
  }
  for (const double flux : coarse_residual) {
    if (!std::isfinite(flux)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
  }
  FLUIDITY_PRESSURE_TAKE(coarse_correction, double, coarse_residual.size());
  status = ApplyVcycle<double>(hierarchy, level_ordinal + 1, coarse_residual, coarse_correction, scratch);
  if (!status.ok()) {
    return status;
  }
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const double coarse_pressure = coarse_correction[level.aggregate_ordinals[unknown]];
    if constexpr (std::same_as<Scalar, float>) {
      const auto fine_pressure = FineValue(coarse_pressure);
      if (!fine_pressure.ok()) {
        return fine_pressure.status();
      }
      correction[unknown] += *fine_pressure;
    } else {
      correction[unknown] += coarse_pressure;
    }
  }
  status = ApplyLevel<Scalar>(level, correction, correction_flux);
  if (!status.ok()) {
    return status;
  }
  for (std::size_t unknown = 0; unknown < residual.size(); ++unknown) {
    const Scalar remaining_flux = residual[unknown] - correction_flux[unknown];
    correction[unknown] += kJacobiWeight * (remaining_flux / static_cast<Scalar>(level.diagonal[unknown]));
    if (!std::isfinite(correction[unknown])) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
  }
  return absl::OkStatus();
}

absl::Status Evaluate(
    const PreparedMacPressureView3D& system,
    std::span<const double> candidate_cell_pressures,
    std::span<double> velocities,
    std::span<double> residuals,
    std::span<double> defects,
    MacPressureDiagnostics3D& diagnostics,
    MacPressureScratch3D& scratch
) {
  const auto cells = system.cells();
  if (cells.empty() || candidate_cell_pressures.size() != cells.size()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  for (const double pressure : candidate_cell_pressures) {
    if (!std::isfinite(pressure)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }

  MacPressureScratchMark3D temporary(scratch);
  FLUIDITY_PRESSURE_TAKE(operator_rows, FluxRow, cells.size());
  auto status = ApplyOriginalPressureOperatorInto(system, candidate_cell_pressures, operator_rows);
  if (!status.ok()) {
    return status;
  }
  FLUIDITY_PRESSURE_TAKE(physical_rows, FluxRow, cells.size());
  status = CorrectFacesAndAccumulatePhysicalDefectInto(system, candidate_cell_pressures, velocities, physical_rows);
  if (!status.ok()) {
    return status;
  }
  diagnostics = {};

  const auto integrated_rhs = system.integrated_rhs();
  const auto rhs_absolute_term_sums = system.rhs_absolute_term_sums();
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const double residual = integrated_rhs[cell] - operator_rows[cell].flux.Total();
    const double fluid_volume = cells[cell].fluid_volume;
    const double volume_scaled_residual = std::abs(residual) / fluid_volume;
    const double defect = physical_rows[cell].flux.Total() / fluid_volume;
    const double identity_error = fluid_volume * defect + residual;
    const double absolute_row_terms = rhs_absolute_term_sums[cell] + operator_rows[cell].absolute_terms.Total()
        + physical_rows[cell].absolute_terms.Total();
    if (!std::isfinite(volume_scaled_residual) || !std::isfinite(defect) || !std::isfinite(identity_error)
        || !std::isfinite(absolute_row_terms)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    if (std::abs(identity_error) > kIdentityRoundoff * absolute_row_terms) {
      return absl::Status(absl::StatusCode::kInternal, {});
    }

    residuals[cell] = residual;
    defects[cell] = defect;
    // hypot avoids squaring tiny or very large integrated residuals.
    diagnostics.true_residual_l2 = std::hypot(diagnostics.true_residual_l2, residual);
    diagnostics.max_volume_scaled_residual = std::max(diagnostics.max_volume_scaled_residual, volume_scaled_residual);
    diagnostics.max_cell_constraint_defect = std::max(diagnostics.max_cell_constraint_defect, std::abs(defect));
  }
  if (!std::isfinite(diagnostics.true_residual_l2)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }

  return absl::OkStatus();
}
absl::StatusOr<MacPressureProjectionView3D> SolvePrepared(
    const pressure_internal::PreparedMacPressureUploadView3D& upload,
    MacPressureSolverOptions3D options,
    MacPressureScratch3D& scratch,
    std::span<double> pressures,
    std::span<double> velocities,
    std::span<double> rhs,
    std::span<double> residuals,
    std::span<double> defects,
    std::span<std::size_t> gauges
) {
  const auto* system = &upload.system;
  const auto* hierarchy = &upload.hierarchy;
  const auto* criteria = &upload.stopping;
  auto base_pressure = upload.initial_pressure;
  std::ranges::copy(system->rhs, rhs.begin());
  std::ranges::copy(system->gauges, gauges.begin());
  MacPressureProjectionView3D result{
      .cell_pressures = pressures,
      .corrected_face_velocities = velocities,
      .integrated_rhs = rhs,
      .true_residuals = residuals,
      .cell_constraint_defects = defects,
      .gauge_cell_ordinals = gauges.first(system->gauges.size()),
      .diagnostics = {},
      .frozen_measurement = upload.frozen_measurement,
  };
  const auto evaluate_correction = [&](std::span<const float> correction,
                                       std::size_t iterations) -> absl::StatusOr<MacPressureProjectionView3D> {
    std::ranges::copy(base_pressure, pressures.begin());
    for (std::size_t unknown = 0; unknown < correction.size(); ++unknown) {
      pressures[hierarchy->unknown_to_cell[unknown]] += static_cast<double>(correction[unknown]);
      if (!std::isfinite(pressures[hierarchy->unknown_to_cell[unknown]])) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
    }
    auto status = Evaluate(*system, pressures, velocities, residuals, defects, result.diagnostics, scratch);
    if (!status.ok()) {
      return status;
    }
    result.diagnostics.iteration_count = iterations;
    result.diagnostics.multigrid_level_count = hierarchy->levels.size();
    return result;
  };
  const std::size_t count = hierarchy->unknown_to_cell.size();
  FLUIDITY_PRESSURE_TAKE(correction, float, count);
  auto projection = evaluate_correction(correction, 0);
  if (!projection.ok()) {
    return projection.status();
  }
  if (pressure_internal::MeetsMacPressureStoppingCriteria3D(projection->diagnostics, *criteria)) {
    return projection;
  }
  if (count == 0) {
    return absl::Status(absl::StatusCode::kAborted, {});
  }

  FLUIDITY_PRESSURE_TAKE(residual, float, count);
  auto status = CopyTrueResidual(*projection, *hierarchy, residual);
  if (!status.ok()) {
    return status;
  }
  FLUIDITY_PRESSURE_TAKE(direction, float, count);
  FLUIDITY_PRESSURE_TAKE(direction_flux, float, count);
  FLUIDITY_PRESSURE_TAKE(preconditioned, float, count);
  double previous_preconditioned_product = 0.0;
  bool restart_direction = true;
  for (std::size_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
    status = ApplyVcycle<float>(*hierarchy, 0, residual, preconditioned, scratch);
    if (!status.ok()) {
      return status;
    }
    const auto preconditioned_product = Dot(residual, preconditioned);
    if (!preconditioned_product.ok()) {
      return preconditioned_product.status();
    }
    if (*preconditioned_product <= 0.0) {
      return absl::Status(absl::StatusCode::kAborted, {});
    }
    if (restart_direction) {
      std::ranges::copy(preconditioned, direction.begin());
    } else {
      const auto beta = FineValue(*preconditioned_product / previous_preconditioned_product);
      if (!beta.ok()) {
        return beta.status();
      }
      for (std::size_t unknown = 0; unknown < count; ++unknown) {
        const float previous_direction = *beta * direction[unknown];
        direction[unknown] = preconditioned[unknown] + previous_direction;
      }
    }
    status = ApplyLevel<float>(hierarchy->levels.front(), direction, direction_flux);
    if (!status.ok()) {
      return status;
    }
    const auto curvature = Dot(direction, direction_flux);
    if (!curvature.ok()) {
      return curvature.status();
    }
    if (*curvature <= 0.0) {
      return absl::Status(absl::StatusCode::kAborted, {});
    }
    const auto alpha = FineValue(*preconditioned_product / *curvature);
    if (!alpha.ok()) {
      return alpha.status();
    }
    for (std::size_t unknown = 0; unknown < count; ++unknown) {
      const float pressure_increment = *alpha * direction[unknown];
      const float flux_increment = *alpha * direction_flux[unknown];
      correction[unknown] += pressure_increment;
      residual[unknown] -= flux_increment;
      if (!std::isfinite(correction[unknown]) || !std::isfinite(residual[unknown])) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
    }
    restart_direction = false;
    previous_preconditioned_product = *preconditioned_product;
    const bool last_iteration = iteration == options.max_iterations;
    if (last_iteration || RecursiveResidualPasses(*system, *hierarchy, residual, *criteria)) {
      projection = evaluate_correction(correction, iteration);
      if (!projection.ok()) {
        return projection.status();
      }
      if (pressure_internal::MeetsMacPressureStoppingCriteria3D(projection->diagnostics, *criteria)) {
        return projection;
      }
      if (last_iteration) {
        break;
      }
      // The FP32 recursive residual can drift from the true FP64 residual.
      // Accumulate the rounded correction in the FP64 base and restart the
      // search from the recomputed residual within the original iteration budget.
      std::ranges::copy(projection->cell_pressures, base_pressure.begin());
      std::fill(correction.begin(), correction.end(), 0.0F);
      status = CopyTrueResidual(*projection, *hierarchy, residual);
      if (!status.ok()) {
        return status;
      }
      restart_direction = true;
    }
  }
  return absl::Status(absl::StatusCode::kAborted, {});
}

// Reserve every upper-bound segment separately, including its alignment. This
// overestimates simultaneous stage scratch but exactly describes the allocation.
// A level never gains nodes/edges. Every new coordinate is signed floor(x/2):
// after 31 steps all coordinates lie in {-1,0}^3, and one more spatial grouping
// leaves at most eight nodes per connected block. Thus 33 levels suffice.
// Terminal blocks have at most 32 nodes, so sum(block_size^2) <= 32*N.
class ScratchByteCounter {
 public:
  bool Add(std::size_t count, std::size_t stride, std::size_t repetitions = 1) {
    constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    constexpr std::size_t alignment = sizeof(std::max_align_t);
    if (count > (limit - alignment + 1) / stride) {
      return false;
    }
    const std::size_t raw = count * stride;
    const std::size_t padded = (raw + alignment - 1) / alignment * alignment;
    if (padded > (limit - total) / repetitions) {
      return false;
    }
    total += padded * repetitions;
    return true;
  }
  std::size_t total = 0;
};
bool CountOriginalPressureScratch(ScratchByteCounter& counter, std::size_t cell_capacity, std::size_t face_capacity) {
  return counter.Add(cell_capacity, sizeof(PreparedMacPressureCell3D))
      && counter.Add(face_capacity, sizeof(FrozenMacPressureLink3D)) && counter.Add(cell_capacity, sizeof(double), 2)
      && counter.Add(cell_capacity, sizeof(std::size_t), 2) && counter.Add(cell_capacity, 6 * sizeof(std::size_t))
      && counter.Add(cell_capacity, sizeof(FluxRow)) && counter.Add(cell_capacity, sizeof(ComponentBalance));
}

bool CountCoupledPressureScratch(
    ScratchByteCounter& counter,
    std::size_t cell_capacity,
    std::size_t face_capacity,
    std::size_t strain_row_capacity
) {
  using CompensatedSum = pressure_internal::arithmetic::CompensatedSum;
  return counter.Add(strain_row_capacity, sizeof(MacStrainRow3D), 2)
      && counter.Add(face_capacity, pressure_internal::kCoupledVelocityScratchFields3D * sizeof(double))
      && counter.Add(face_capacity, pressure_internal::kCoupledFaceScratchFields3D * sizeof(double))
      && counter.Add(cell_capacity, pressure_internal::kCoupledCellScratchFields3D * sizeof(double))
      && counter.Add(cell_capacity, sizeof(CompensatedSum)) && counter.Add(cell_capacity, sizeof(bool))
      && counter.Add(cell_capacity, 2 * pressure_internal::kCoupledPressureRestartLength3D * sizeof(double));
}

template <typename T>
absl::Status Reserve(LiquidRecords3D<T>& records, std::size_t capacity) {
  auto reserved = LiquidRecords3D<T>::Create(capacity);
  if (!reserved.ok()) {
    return reserved.status();
  }
  records = std::move(*reserved);
  return absl::OkStatus();
}
#undef FLUIDITY_PRESSURE_TAKE
}  // namespace

absl::StatusOr<std::size_t>
pressure_internal::QueryBoundedMacPressureSystemBytes3D(std::size_t cell_capacity, std::size_t face_capacity) {
  if (cell_capacity > kMaxPressureCells || face_capacity > kMaxPressureFaces) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  ScratchByteCounter counter;
  if (!CountOriginalPressureScratch(counter, cell_capacity, face_capacity)) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  return counter.total;
}

absl::StatusOr<std::size_t> pressure_internal::QueryCoupledMacPressureScratchBytes3D(
    std::size_t cell_capacity,
    std::size_t face_capacity,
    std::size_t strain_row_capacity,
    bool prepare_uniform_system
) {
  ScratchByteCounter counter;
  if (cell_capacity > kMaxPressureCells || face_capacity > kMaxPressureFaces
      || (prepare_uniform_system && !CountOriginalPressureScratch(counter, cell_capacity, face_capacity))
      || !CountCoupledPressureScratch(counter, cell_capacity, face_capacity, strain_row_capacity)) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  return counter.total;
}

absl::StatusOr<MacPressureWorkspaceBytes3D>
QueryMacPressureWorkspaceBytes3D(std::size_t cell_capacity, std::size_t face_capacity) {
  if (cell_capacity == 0 && face_capacity == 0) {
    return MacPressureWorkspaceBytes3D{};
  }
  ScratchByteCounter counter;
  const auto level_capacity = kMaximumPressureLevels3D;
  const bool fits = CountOriginalPressureScratch(counter, cell_capacity, face_capacity)
      && counter.Add(cell_capacity, sizeof(std::size_t), 2)
      && counter.Add(level_capacity, sizeof(MacPressureLevelView3D))
      && counter.Add(cell_capacity, sizeof(CellIndex3D), level_capacity)
      && counter.Add(cell_capacity, sizeof(double), 2 * level_capacity)
      && counter.Add(cell_capacity, sizeof(std::size_t), 3 * level_capacity)
      && counter.Add(1, sizeof(std::size_t), level_capacity)
      && counter.Add(face_capacity, sizeof(MacPressureEdge3D), level_capacity)
      && counter.Add(face_capacity, 2 * sizeof(MacPressureNeighbor3D), level_capacity)
      && counter.Add(cell_capacity, sizeof(std::size_t), 2) && counter.Add(1, sizeof(std::size_t))
      && counter.Add(cell_capacity, 32 * sizeof(double))
      // Coarsening order, CSR cursor and terminal inverse permutation.
      && counter.Add(cell_capacity, sizeof(std::size_t), 3) && counter.Add(1, sizeof(std::size_t))
      && counter.Add(face_capacity, sizeof(std::size_t))
      // PCG base, five FP32 vectors, prior gauge shifts and evaluator rows.
      && counter.Add(cell_capacity, sizeof(double), 2) && counter.Add(cell_capacity, sizeof(float), 5)
      && counter.Add(cell_capacity, sizeof(FluxRow), 2)
      // Recursive V-cycle flux, coarse RHS and coarse correction; FP64 bounds FP32.
      && counter.Add(cell_capacity, sizeof(double), 3 * level_capacity);
  if (!fits) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  ScratchByteCounter coupled_counter;
  if (!CountOriginalPressureScratch(coupled_counter, cell_capacity, face_capacity)
      || cell_capacity > std::numeric_limits<std::size_t>::max() / pressure_internal::kMaximumUniformStrainRowsPerCell3D
      || !CountCoupledPressureScratch(
          coupled_counter,
          cell_capacity,
          face_capacity,
          cell_capacity * pressure_internal::kMaximumUniformStrainRowsPerCell3D
      )) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  counter.total = std::max(counter.total, coupled_counter.total);
  MacPressureWorkspaceBytes3D result;
  result.scratch_bytes = counter.total;
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  constexpr auto projection_bytes_per_cell = 2 * (4 * sizeof(double) + sizeof(std::size_t));
  if (cell_capacity > (limit - counter.total) / projection_bytes_per_cell) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  result.projection_bytes = cell_capacity * projection_bytes_per_cell;
  if (face_capacity > (limit - counter.total - result.projection_bytes) / (2 * sizeof(double))) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  result.projection_bytes += face_capacity * 2 * sizeof(double);
  result.total_bytes = result.scratch_bytes + result.projection_bytes;
  result.maximum_levels = level_capacity;
  result.maximum_level_cells = cell_capacity;
  result.maximum_level_edges = face_capacity;
  result.maximum_terminal_factor_entries = 32 * cell_capacity;
  return result;
}

absl::StatusOr<MacPressureWorkspace3D>
MacPressureWorkspace3D::Create(std::size_t cell_capacity, std::size_t face_capacity) {
  const auto bytes = QueryMacPressureWorkspaceBytes3D(cell_capacity, face_capacity);
  if (!bytes.ok()) {
    return bytes.status();
  }
  MacPressureWorkspace3D workspace;
  workspace.bytes_ = *bytes;
  workspace.cell_capacity_ = cell_capacity;
  workspace.face_capacity_ = face_capacity;
  auto status = Reserve(workspace.scratch_, bytes->scratch_bytes / sizeof(std::max_align_t));
  if (!status.ok()) {
    return status;
  }
  status = workspace.scratch_.Resize(workspace.scratch_.capacity());
  if (!status.ok()) {
    return status;
  }
  for (auto& bank : workspace.projections_) {
    for (auto* records : {&bank.pressures, &bank.rhs, &bank.residuals, &bank.defects}) {
      status = Reserve(*records, cell_capacity);
      if (!status.ok()) {
        return status;
      }
    }
    status = Reserve(bank.velocities, face_capacity);
    if (!status.ok()) {
      return status;
    }
    status = Reserve(bank.gauges, cell_capacity);
    if (!status.ok()) {
      return status;
    }
  }
  return workspace;
}
std::size_t MacPressureWorkspace3D::used_bytes() const {
  std::size_t used = scratch_used_bytes_;
  for (const auto& bank : projections_) {
    used +=
        (bank.pressures.size() + bank.rhs.size() + bank.residuals.size() + bank.defects.size() + bank.velocities.size())
            * sizeof(double)
        + bank.gauges.size() * sizeof(std::size_t);
  }
  return used;
}
template <typename PrepareUpload>
absl::StatusOr<MacPressureProjectionView3D> MacPressureWorkspace3D::ProjectWithPreparation(
    std::size_t cell_count,
    std::size_t face_count,
    MacPressureSolverOptions3D options,
    PrepareUpload prepare_upload
) {
  if (options.max_iterations == 0 || options.max_iterations > 256) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (cell_count > cell_capacity_ || face_count > face_capacity_ || cell_count > kMaxPressureCells
      || face_count > kMaxPressureFaces) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  const std::size_t candidate_bank = 1 - published_bank_;
  auto& bank = projections_[candidate_bank];
  for (auto* records : {&bank.pressures, &bank.rhs, &bank.residuals, &bank.defects}) {
    const auto status = records->Resize(cell_count);
    if (!status.ok()) {
      return status;
    }
  }
  auto status = bank.velocities.Resize(face_count);
  if (!status.ok()) {
    return status;
  }
  status = bank.gauges.Resize(cell_count);
  if (!status.ok()) {
    return status;
  }
  MacPressureScratch3D scratch(scratch_.values());
  const auto upload = prepare_upload(scratch);
  const auto result = upload.ok() ? SolvePrepared(
                                        *upload,
                                        options,
                                        scratch,
                                        bank.pressures.values(),
                                        bank.velocities.values(),
                                        bank.rhs.values(),
                                        bank.residuals.values(),
                                        bank.defects.values(),
                                        bank.gauges.values()
                                    )
                                  : absl::StatusOr<MacPressureProjectionView3D>(upload.status());
  scratch_used_bytes_ = scratch.used();
  const std::size_t projection_used = used_bytes() - scratch_used_bytes_;
  high_water_bytes_ = std::max(high_water_bytes_, projection_used + scratch.high_water());
  if (!result.ok()) {
    return result.status();
  }
  status = bank.gauges.Resize(result->gauge_cell_ordinals.size());
  if (!status.ok()) {
    return status;
  }
  published_bank_ = candidate_bank;
  return result;
}
absl::StatusOr<MacPressureWorkspace3D::PhysicalComponents> MacPressureWorkspace3D::ClassifyPhysicalComponents(
    SparseMacGridView3D grid,
    double step_duration,
    const LiquidPressureProblemView3D& problem
) {
  if (problem.cells.size() > cell_capacity_ || problem.faces.size() > face_capacity_) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  MacPressureScratch3D scratch(scratch_.values());
  const auto prepared = Prepare({grid, step_duration, problem.cells, problem.faces}, scratch);
  scratch_used_bytes_ = scratch.used();
  const std::size_t projection_used = used_bytes() - scratch_used_bytes_;
  high_water_bytes_ = std::max(high_water_bytes_, projection_used + scratch.high_water());
  if (!prepared.ok()) {
    return prepared.status();
  }
  return PhysicalComponents{prepared->component_ordinals(), prepared->gauge_cell_ordinals()};
}

absl::StatusOr<MacPressureProjectionView3D> MacPressureWorkspace3D::ProjectPrepared(
    const pressure_internal::PreparedMacPressureSystem3D& system,
    MacPressureSolverOptions3D options
) {
  if (!system.is_explicit_graph() || system.cells().empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  return ProjectWithPreparation(
      system.cells().size(),
      system.links().size(),
      options,
      [&](MacPressureScratch3D& scratch) {
        const auto prepared = pressure_internal::BorrowMacPressureSystem3D(system);
        return PrepareUploadForSystem(prepared, {}, scratch, false);
      }
  );
}

absl::StatusOr<MacPressureProjectionView3D> ProjectMacPressure3D(
    SparseMacGridView3D grid,
    double step_duration,
    const LiquidPressureProblemView3D& problem,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace
) {
  return workspace
      .ProjectWithPreparation(problem.cells.size(), problem.faces.size(), options, [&](MacPressureScratch3D& scratch) {
        return pressure_internal::PrepareBoundedMacPressureUpload3D(
            grid,
            step_duration,
            problem.cells,
            problem.faces,
            scratch,
            problem.seed_uniform_dirichlet_rest_pressure
        );
      });
}

namespace {
absl::StatusOr<pressure_internal::PreparedMacPressureUploadView3D> PrepareUploadForSystem(
    const PreparedMacPressureView3D& prepared_system,
    std::span<const MacPressureFace3D> uniform_faces,
    MacPressureScratch3D& scratch,
    bool seed_uniform_dirichlet_rest_pressure
) {
  MacPressureScratchMark3D publication(scratch);
  const auto* system = &prepared_system;
  const auto hierarchy = pressure_internal::BuildBoundedMacPressureHierarchy3D(*system, scratch);
  if (!hierarchy.ok()) {
    return hierarchy.status();
  }
  double rhs_norm = 0;
  for (const double value : system->rhs) {
    rhs_norm = std::hypot(rhs_norm, value);
  }
  const auto stopping = pressure_internal::MacPressureStoppingCriteria(rhs_norm, system->links());
  if (!stopping.ok()) {
    return stopping.status();
  }
  const auto base_result = scratch.Take<double>(system->cells().size());
  if (!base_result.ok()) {
    return base_result.status();
  }
  auto base_pressure = *base_result;
  {
    MacPressureScratchMark3D temporary(scratch);
    const auto pin_result = scratch.Take<double>(base_pressure.size());
    if (!pin_result.ok()) {
      return pin_result.status();
    }
    auto pin_previous_pressures = *pin_result;
    for (const auto pin : system->gauges) {
      pin_previous_pressures[system->components[pin]] = system->cell_records[pin].previous_pressure;
    }
    for (std::size_t cell = 0; cell < base_pressure.size(); ++cell) {
      base_pressure[cell] =
          system->cell_records[cell].previous_pressure - pin_previous_pressures[system->components[cell]];
      if (!std::isfinite(base_pressure[cell])) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
    }
  }
  if (seed_uniform_dirichlet_rest_pressure) {
    const auto rest_pressure = UniformDirichletRestPressure(*system);
    if (rest_pressure) {
      std::ranges::fill(base_pressure, *rest_pressure);
    }
  }
  pressure_internal::arithmetic::CompensatedSum pressure_cell_volume;
  for (const auto& cell : system->cells()) {
    pressure_cell_volume.Add(cell.fluid_volume);
  }
  pressure_internal::arithmetic::CompensatedSum inward_prescribed_wall_flux;
  for (std::size_t face_ordinal = 0; face_ordinal < system->links().size(); ++face_ordinal) {
    const bool has_explicit_wall_flux =
        uniform_faces.empty() || uniform_faces[face_ordinal].prescribed_wall_flux.has_value();
    pressure_internal::arithmetic::AccumulateInwardExplicitWallFlux(
        system->links()[face_ordinal],
        has_explicit_wall_flux,
        inward_prescribed_wall_flux
    );
  }
  publication.KeepAllocations();
  return pressure_internal::PreparedMacPressureUploadView3D{
      .system = *system,
      .hierarchy = *hierarchy,
      .initial_pressure = base_pressure,
      .stopping = *stopping,
      .frozen_measurement =
          {
              .integrated_rhs_l2_m3_per_s = rhs_norm,
              .pressure_cell_volume_m3 = pressure_cell_volume.Total(),
              .inward_prescribed_wall_flux_m3_per_s = inward_prescribed_wall_flux.Total(),
          },
  };
}
}  // namespace

absl::StatusOr<pressure_internal::PreparedMacPressureView3D> pressure_internal::PrepareBoundedMacPressureSystem3D(
    SparseMacGridView3D grid,
    double step_duration,
    std::span<const MacPressureCell3D> cells,
    std::span<const MacPressureFace3D> faces,
    MacPressureScratch3D& scratch
) {
  if (cells.size() > kMaxPressureCells || faces.size() > kMaxPressureFaces) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  MacPressureScratchMark3D publication(scratch);
  const auto system = Prepare({grid, step_duration, cells, faces}, scratch);
  if (system.ok()) {
    publication.KeepAllocations();
  }
  return system;
}

absl::StatusOr<pressure_internal::PreparedMacPressureUploadView3D> pressure_internal::PrepareBoundedMacPressureUpload3D(
    SparseMacGridView3D grid,
    double step_duration,
    std::span<const MacPressureCell3D> cells,
    std::span<const MacPressureFace3D> faces,
    MacPressureScratch3D& scratch,
    bool seed_uniform_dirichlet_rest_pressure
) {
  MacPressureScratchMark3D publication(scratch);
  const auto system = PrepareBoundedMacPressureSystem3D(grid, step_duration, cells, faces, scratch);
  if (!system.ok()) {
    return system.status();
  }
  const auto upload = PrepareUploadForSystem(*system, faces, scratch, seed_uniform_dirichlet_rest_pressure);
  if (upload.ok()) {
    publication.KeepAllocations();
  }
  return upload;
}
}  // namespace fluidity
