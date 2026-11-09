#include "fluidity/host/mac_pressure_workspace_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

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

struct BorrowedInputs {
  SparseMacGridView3D mac_grid;
  double step_duration;
  std::span<const MacPressureCell3D> pressure_cells;
  std::span<const MacPressureFace3D> pressure_faces;
};

absl::StatusOr<PreparedMacPressureView3D> Prepare(const BorrowedInputs& inputs, MacPressureScratch3D& scratch) {
  const std::size_t count = inputs.pressure_cells.size();
  const std::size_t face_count = inputs.pressure_faces.size();
  if (count == 0 || face_count == 0 || !IsFinitePositive(inputs.step_duration)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const double width = inputs.mac_grid.geometry().cell_width;
  const double area = width * width;
  const double volume = area * width;
  if (!IsFinitePositive(area) || !IsFinitePositive(volume)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  FLUIDITY_PRESSURE_TAKE(cells, PreparedMacPressureCell3D, count);
  FLUIDITY_PRESSURE_TAKE(links, FrozenMacPressureLink3D, face_count);
  FLUIDITY_PRESSURE_TAKE(rhs, double, count);
  FLUIDITY_PRESSURE_TAKE(absolute_terms, double, count);
  FLUIDITY_PRESSURE_TAKE(components, std::size_t, count);
  FLUIDITY_PRESSURE_TAKE(gauges, std::size_t, count);
  PreparedMacPressureView3D system{cells, links, rhs, absolute_terms, components, gauges};
  MacPressureScratchMark3D temporary(scratch);
  FLUIDITY_PRESSURE_TAKE(incident_slots, std::size_t, 6 * count);
  std::size_t incident_count = 0;
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
    const auto& cell = inputs.pressure_cells[ordinal];
    if (ordinal != 0 && CellSortKey(inputs.pressure_cells[ordinal - 1].cell_index) >= CellSortKey(cell.cell_index)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const auto frozen = FreezeCell(cell, inputs.step_duration, volume);
    if (!frozen.ok()) {
      return frozen.status();
    }
    cells[ordinal] = *frozen;
    const std::array coordinates{cell.cell_index.x, cell.cell_index.y, cell.cell_index.z};
    for (std::size_t normal = 0; normal < 3; ++normal) {
      if (coordinates[normal] == std::numeric_limits<std::int32_t>::max()) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      for (std::int32_t side = 0; side < 2; ++side) {
        auto face = coordinates;
        face[normal] += side;
        const auto slot = inputs.mac_grid.FaceSlot({kFaceAxes[normal], {face[0], face[1], face[2]}});
        if (!slot.ok()) {
          return slot.status();
        }
        incident_slots[incident_count++] = *slot;
      }
    }
  }
  std::ranges::sort(incident_slots);
  incident_count = static_cast<std::size_t>(std::ranges::unique(incident_slots).begin() - incident_slots.begin());
  if (incident_count != face_count) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  for (std::size_t face = 0; face < face_count; ++face) {
    const auto slot = inputs.mac_grid.FaceSlot(inputs.pressure_faces[face].face_index);
    if (!slot.ok()) {
      return slot.status();
    }
    if (*slot != incident_slots[face]) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const auto link = FreezeLink(inputs.pressure_faces[face], *slot, area, inputs);
    if (!link.ok()) {
      return link.status();
    }
    links[face] = *link;
  }
  FLUIDITY_PRESSURE_TAKE(rows, FluxRow, count);
  auto status = AssembleIntegratedRhsInto(cells, links, rows);
  if (!status.ok()) {
    return status;
  }
  for (std::size_t cell = 0; cell < count; ++cell) {
    rhs[cell] = rows[cell].flux.Total();
    absolute_terms[cell] = rows[cell].absolute_terms.Total();
    if (!std::isfinite(rhs[cell]) || !std::isfinite(absolute_terms[cell])) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
  }
  FLUIDITY_PRESSURE_TAKE(balances, ComponentBalance, count);
  const auto gauge_count = SelectGaugesInto(cells, links, components, balances, gauges);
  if (!gauge_count.ok()) {
    return gauge_count.status();
  }
  system.gauges = gauges.first(*gauge_count);
  return system;
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
absl::StatusOr<MacPressureProjectionView3D> Solve(
    const BorrowedInputs& inputs,
    MacPressureSolverOptions3D options,
    MacPressureScratch3D& scratch,
    std::span<double> pressures,
    std::span<double> velocities,
    std::span<double> rhs,
    std::span<double> residuals,
    std::span<double> defects,
    std::span<std::size_t> gauges
) {
  const auto upload = pressure_internal::PrepareBoundedMacPressureUpload3D(
      inputs.mac_grid,
      inputs.step_duration,
      inputs.pressure_cells,
      inputs.pressure_faces,
      scratch
  );
  if (!upload.ok()) {
    return upload.status();
  }
  const auto* system = &upload->system;
  const auto* hierarchy = &upload->hierarchy;
  const auto* criteria = &upload->stopping;
  auto base_pressure = upload->initial_pressure;
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

absl::StatusOr<MacPressureWorkspaceBytes3D>
QueryMacPressureWorkspaceBytes3D(std::size_t cell_capacity, std::size_t face_capacity) {
  if (cell_capacity == 0 && face_capacity == 0) {
    return MacPressureWorkspaceBytes3D{};
  }
  ScratchByteCounter counter;
  const auto n = cell_capacity;
  const auto f = face_capacity;
  const auto levels = kMaximumPressureLevels3D;
  const bool fits = counter.Add(n, sizeof(PreparedMacPressureCell3D)) && counter.Add(f, sizeof(FrozenMacPressureLink3D))
      && counter.Add(n, sizeof(double), 2) && counter.Add(n, sizeof(std::size_t), 4)
      && counter.Add(levels, sizeof(MacPressureLevelView3D)) && counter.Add(n, sizeof(CellIndex3D), levels)
      && counter.Add(n, sizeof(double), 2 * levels) && counter.Add(n, sizeof(std::size_t), 3 * levels)
      && counter.Add(1, sizeof(std::size_t), levels) && counter.Add(f, sizeof(MacPressureEdge3D), levels)
      && counter.Add(f, 2 * sizeof(MacPressureNeighbor3D), levels) && counter.Add(n, sizeof(std::size_t), 2)
      && counter.Add(1, sizeof(std::size_t))
      && counter.Add(n, 32 * sizeof(double))
      // Assembly temporaries: incident duplicates, compensated RHS and balances.
      && counter.Add(n, 6 * sizeof(std::size_t)) && counter.Add(n, sizeof(FluxRow))
      && counter.Add(n, sizeof(ComponentBalance))
      // Coarsening order, CSR cursor and terminal inverse permutation.
      && counter.Add(n, sizeof(std::size_t), 3) && counter.Add(1, sizeof(std::size_t))
      && counter.Add(f, sizeof(std::size_t))
      // PCG base, five FP32 vectors, prior gauge shifts and evaluator rows.
      && counter.Add(n, sizeof(double), 2) && counter.Add(n, sizeof(float), 5)
      && counter.Add(n, sizeof(FluxRow), 2)
      // Recursive V-cycle flux, coarse RHS and coarse correction; FP64 bounds FP32.
      && counter.Add(n, sizeof(double), 3 * levels);
  if (!fits) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  MacPressureWorkspaceBytes3D result;
  result.scratch_bytes = counter.total;
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  constexpr auto cell_stride = 2 * (4 * sizeof(double) + sizeof(std::size_t));
  if (n > (limit - counter.total) / cell_stride) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  result.projection_bytes = n * cell_stride;
  if (f > (limit - counter.total - result.projection_bytes) / (2 * sizeof(double))) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  result.projection_bytes += f * 2 * sizeof(double);
  result.total_bytes = result.scratch_bytes + result.projection_bytes;
  result.maximum_levels = levels;
  result.maximum_level_cells = n;
  result.maximum_level_edges = f;
  result.maximum_terminal_factor_entries = 32 * n;
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
absl::StatusOr<MacPressureProjectionView3D> ProjectMacPressure3D(
    SparseMacGridView3D grid,
    double step_duration,
    const LiquidPressureProblemView3D& problem,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace
) {
  if (options.max_iterations == 0 || options.max_iterations > 256) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (problem.cells.size() > workspace.cell_capacity_ || problem.faces.size() > workspace.face_capacity_
      || problem.cells.size() > kMaxPressureCells || problem.faces.size() > kMaxPressureFaces) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  const std::size_t candidate_bank = 1 - workspace.published_bank_;
  auto& bank = workspace.projections_[candidate_bank];
  for (auto* records : {&bank.pressures, &bank.rhs, &bank.residuals, &bank.defects}) {
    const auto status = records->Resize(problem.cells.size());
    if (!status.ok()) {
      return status;
    }
  }
  auto status = bank.velocities.Resize(problem.faces.size());
  if (!status.ok()) {
    return status;
  }
  status = bank.gauges.Resize(problem.cells.size());
  if (!status.ok()) {
    return status;
  }
  MacPressureScratch3D scratch(workspace.scratch_.values());
  const auto result = Solve(
      {grid, step_duration, problem.cells, problem.faces},
      options,
      scratch,
      bank.pressures.values(),
      bank.velocities.values(),
      bank.rhs.values(),
      bank.residuals.values(),
      bank.defects.values(),
      bank.gauges.values()
  );
  workspace.scratch_used_bytes_ = scratch.used();
  const std::size_t projection_used = workspace.used_bytes() - workspace.scratch_used_bytes_;
  workspace.high_water_bytes_ = std::max(workspace.high_water_bytes_, projection_used + scratch.high_water());
  if (!result.ok()) {
    return result.status();
  }
  status = bank.gauges.Resize(result->gauge_cell_ordinals.size());
  if (!status.ok()) {
    return status;
  }
  workspace.published_bank_ = candidate_bank;
  return result;
}
absl::StatusOr<pressure_internal::PreparedMacPressureUploadView3D> pressure_internal::PrepareBoundedMacPressureUpload3D(
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
  const BorrowedInputs inputs{grid, step_duration, cells, faces};
  const auto system = Prepare(inputs, scratch);
  if (!system.ok()) {
    return system.status();
  }
  const auto hierarchy = pressure_internal::BuildBoundedMacPressureHierarchy3D(*system, scratch);
  if (!hierarchy.ok()) {
    return hierarchy.status();
  }
  double rhs_norm = 0;
  for (const double value : system->rhs) {
    rhs_norm = std::hypot(rhs_norm, value);
  }
  const pressure_internal::MacPressureStoppingCriteria3D stopping{.integrated_residual_limit = 2e-6 + 2e-6 * rhs_norm};
  if (!std::isfinite(rhs_norm) || !std::isfinite(stopping.integrated_residual_limit)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
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
  publication.KeepAllocations();
  return PreparedMacPressureUploadView3D{
      .system = *system,
      .hierarchy = *hierarchy,
      .initial_pressure = base_pressure,
      .stopping = stopping,
  };
}
}  // namespace fluidity
