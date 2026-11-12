#include "mac_pressure_3d_multigrid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "mac_pressure_3d_workspace_internal.h"

namespace fluidity::pressure_internal {
namespace {

bool PositiveFinite(double value) {
  return value > 0.0 && std::isfinite(value);
}

absl::StatusOr<double> RoundedFineCoefficient(double coefficient) {
  if (!PositiveFinite(coefficient) || coefficient > std::numeric_limits<float>::max()) {
    return absl::OutOfRangeError("pressure coefficient exceeds the positive finite FP32 range");
  }
  const float rounded = static_cast<float>(coefficient);
  if (!PositiveFinite(rounded)) {
    return absl::OutOfRangeError("positive pressure coefficient is not representable in FP32");
  }
  return static_cast<double>(rounded);
}

std::size_t FindRoot(std::vector<std::size_t>& roots, std::size_t unknown) {
  while (roots[unknown] != unknown) {
    roots[unknown] = roots[roots[unknown]];
    unknown = roots[unknown];
  }
  return unknown;
}

// Build both row references from each single shared edge. Neighbor order is
// ordinal order, making CPU reductions and later device uploads deterministic.
absl::Status CompleteLevel(MacPressureLevel3D& level) {
  const std::size_t count = level.coordinates.size();
  level.diagonal = level.anchor_diagonal;
  level.row_offsets.assign(count + 1, 0);
  level.component_ordinals.resize(count);
  std::iota(level.component_ordinals.begin(), level.component_ordinals.end(), 0);
  for (const auto& edge : level.edges) {
    level.diagonal[edge.lower_unknown_ordinal] += edge.conductance;
    level.diagonal[edge.higher_unknown_ordinal] += edge.conductance;
    ++level.row_offsets[edge.lower_unknown_ordinal + 1];
    ++level.row_offsets[edge.higher_unknown_ordinal + 1];
    const auto first_root = FindRoot(level.component_ordinals, edge.lower_unknown_ordinal);
    const auto second_root = FindRoot(level.component_ordinals, edge.higher_unknown_ordinal);
    level.component_ordinals[std::max(first_root, second_root)] = std::min(first_root, second_root);
  }
  for (std::size_t unknown = 0; unknown < count; ++unknown) {
    if (!PositiveFinite(level.diagonal[unknown]) || !std::isfinite(level.anchor_diagonal[unknown])) {
      return absl::OutOfRangeError("pressure graph diagonal is not positive and finite");
    }
    level.component_ordinals[unknown] = FindRoot(level.component_ordinals, unknown);
    level.row_offsets[unknown + 1] += level.row_offsets[unknown];
  }
  auto next_offset = level.row_offsets;
  level.neighbors.resize(level.row_offsets.back());
  for (std::size_t ordinal = 0; ordinal < level.edges.size(); ++ordinal) {
    const auto& edge = level.edges[ordinal];
    level.neighbors[next_offset[edge.lower_unknown_ordinal]++] = {edge.higher_unknown_ordinal, ordinal};
    level.neighbors[next_offset[edge.higher_unknown_ordinal]++] = {edge.lower_unknown_ordinal, ordinal};
  }
  for (std::size_t unknown = 0; unknown < count; ++unknown) {
    std::sort(
        level.neighbors.begin() + level.row_offsets[unknown],
        level.neighbors.begin() + level.row_offsets[unknown + 1],
        [](const auto& left, const auto& right) {
          return left.unknown_ordinal < right.unknown_ordinal;
        }
    );
  }
  return absl::OkStatus();
}

std::int32_t FloorHalf(std::int32_t coordinate) {
  return coordinate / 2 - (coordinate % 2 < 0 ? 1 : 0);
}

std::vector<std::vector<std::size_t>> ConnectedBlocks(const MacPressureLevel3D& level) {
  std::vector<std::vector<std::size_t>> blocks(level.coordinates.size());
  for (std::size_t unknown = 0; unknown < level.coordinates.size(); ++unknown) {
    blocks[level.component_ordinals[unknown]].push_back(unknown);
  }
  std::erase_if(blocks, [](const auto& block) {
    return block.empty();
  });
  return blocks;
}

// P is piecewise constant and R=Pᵀ. Internal aggregate edges cancel, crossing
// conductances sum once and anchor diagonals sum without subtracting large
// nearly equal diagonal/off-diagonal terms.
absl::StatusOr<MacPressureLevel3D> Coarsen(MacPressureLevel3D& fine) {
  const auto blocks = ConnectedBlocks(fine);
  fine.aggregate_ordinals.assign(fine.coordinates.size(), kNoPressureUnknown3D);
  MacPressureLevel3D coarse;
  for (const auto& block : blocks) {
    std::map<std::tuple<std::int32_t, std::int32_t, std::int32_t>, std::vector<std::size_t>> spatial_groups;
    for (const auto unknown : block) {
      const auto index = fine.coordinates[unknown];
      spatial_groups[{FloorHalf(index.z), FloorHalf(index.y), FloorHalf(index.x)}].push_back(unknown);
    }
    if (spatial_groups.size() < block.size() || block.size() == 1) {
      for (const auto& [coordinate, members] : spatial_groups) {
        const auto [z, y, x] = coordinate;
        const auto aggregate = coarse.coordinates.size();
        coarse.coordinates.push_back({x, y, z});
        for (const auto unknown : members) {
          fine.aggregate_ordinals[unknown] = aggregate;
        }
      }
    } else {
      // Signed coordinates can plateau across zero. Pair the lowest available
      // unknown with its lowest available neighbor to guarantee graph progress.
      for (const auto unknown : block) {
        if (fine.aggregate_ordinals[unknown] != kNoPressureUnknown3D) {
          continue;
        }
        const auto aggregate = coarse.coordinates.size();
        const auto index = fine.coordinates[unknown];
        coarse.coordinates.push_back({FloorHalf(index.x), FloorHalf(index.y), FloorHalf(index.z)});
        fine.aggregate_ordinals[unknown] = aggregate;
        for (std::size_t offset = fine.row_offsets[unknown]; offset < fine.row_offsets[unknown + 1]; ++offset) {
          const auto neighbor = fine.neighbors[offset].unknown_ordinal;
          if (fine.aggregate_ordinals[neighbor] == kNoPressureUnknown3D) {
            fine.aggregate_ordinals[neighbor] = aggregate;
            break;
          }
        }
      }
    }
  }
  coarse.anchor_diagonal.assign(coarse.coordinates.size(), 0.0);
  for (std::size_t unknown = 0; unknown < fine.coordinates.size(); ++unknown) {
    coarse.anchor_diagonal[fine.aggregate_ordinals[unknown]] += fine.anchor_diagonal[unknown];
  }
  std::map<std::pair<std::size_t, std::size_t>, double> crossing_conductances;
  for (const auto& edge : fine.edges) {
    const auto first = fine.aggregate_ordinals[edge.lower_unknown_ordinal];
    const auto second = fine.aggregate_ordinals[edge.higher_unknown_ordinal];
    if (first != second) {
      crossing_conductances[std::minmax(first, second)] += edge.conductance;
    }
  }
  for (const auto& [endpoints, conductance] : crossing_conductances) {
    if (!PositiveFinite(conductance)) {
      return absl::OutOfRangeError("Galerkin crossing conductance is unrepresentable");
    }
    coarse.edges.push_back({endpoints.first, endpoints.second, conductance});
  }
  const auto status = CompleteLevel(coarse);
  if (!status.ok()) {
    return status;
  }
  return coarse;
}

absl::Status FactorTerminalBlocks(MacPressureLevel3D& level) {
  const auto blocks = ConnectedBlocks(level);
  std::vector<std::size_t> local_ordinals(level.coordinates.size());
  for (const auto& block : blocks) {
    const std::size_t count = block.size();
    if (count > 32) {
      return absl::InternalError("pressure terminal block exceeds 32 unknowns");
    }
    MacPressureCoarseBlock3D terminal{.unknown_ordinals = block, .cholesky_lower = std::vector<double>(count * count)};
    for (std::size_t local = 0; local < count; ++local) {
      local_ordinals[block[local]] = local;
    }
    auto& factor = terminal.cholesky_lower;
    for (std::size_t row = 0; row < count; ++row) {
      const auto unknown = block[row];
      factor[row * count + row] = level.diagonal[unknown];
      for (std::size_t offset = level.row_offsets[unknown]; offset < level.row_offsets[unknown + 1]; ++offset) {
        const auto neighbor = level.neighbors[offset];
        factor[row * count + local_ordinals[neighbor.unknown_ordinal]] =
            -level.edges[neighbor.edge_ordinal].conductance;
      }
    }
    for (std::size_t row = 0; row < count; ++row) {
      for (std::size_t column = 0; column <= row; ++column) {
        double schur_coefficient = factor[row * count + column];
        for (std::size_t inner = 0; inner < column; ++inner) {
          schur_coefficient -= factor[row * count + inner] * factor[column * count + inner];
        }
        if (row == column) {
          if (!PositiveFinite(schur_coefficient)) {
            return absl::OutOfRangeError("pressure Cholesky pivot lost positive definiteness in FP64");
          }
          factor[row * count + column] = std::sqrt(schur_coefficient);
        } else {
          factor[row * count + column] = schur_coefficient / factor[column * count + column];
          if (!std::isfinite(factor[row * count + column])) {
            return absl::OutOfRangeError("pressure Cholesky factor is unrepresentable");
          }
        }
      }
      for (std::size_t column = row + 1; column < count; ++column) {
        factor[row * count + column] = 0.0;
      }
    }
    level.terminal_blocks.push_back(std::move(terminal));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<MacPressureHierarchy3D> BuildMacPressureHierarchy3D(const PreparedMacPressureSystem3D& system) {
  MacPressureHierarchy3D hierarchy;
  const auto cells = system.cells();
  hierarchy.cell_to_unknown.assign(cells.size(), kNoPressureUnknown3D);
  std::vector<bool> pinned(cells.size(), false);
  for (const auto cell : system.gauge_cell_ordinals()) {
    pinned[cell] = true;
  }
  MacPressureLevel3D fine;
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    if (pinned[cell]) {
      continue;
    }
    hierarchy.cell_to_unknown[cell] = hierarchy.unknown_to_cell.size();
    hierarchy.unknown_to_cell.push_back(cell);
    fine.coordinates.push_back(cells[cell].cell_index);
    fine.anchor_diagonal.push_back(cells[cell].compliance);
  }
  for (const auto& link : system.links()) {
    if (link.conductance == 0.0) {
      continue;
    }
    const auto tail =
        link.tail_cell_ordinal ? hierarchy.cell_to_unknown[*link.tail_cell_ordinal] : kNoPressureUnknown3D;
    const auto head =
        link.head_cell_ordinal ? hierarchy.cell_to_unknown[*link.head_cell_ordinal] : kNoPressureUnknown3D;
    if (tail != kNoPressureUnknown3D && head != kNoPressureUnknown3D) {
      const auto rounded_conductance = RoundedFineCoefficient(link.conductance);
      if (!rounded_conductance.ok()) {
        return rounded_conductance.status();
      }
      fine.edges.push_back({std::min(tail, head), std::max(tail, head), *rounded_conductance});
    } else if (tail != kNoPressureUnknown3D) {
      fine.anchor_diagonal[tail] += link.conductance;
    } else if (head != kNoPressureUnknown3D) {
      fine.anchor_diagonal[head] += link.conductance;
    }
  }
  // Round each anchor once after retaining all eliminated-pin conductances.
  for (auto& anchor : fine.anchor_diagonal) {
    if (anchor != 0.0) {
      const auto rounded_anchor = RoundedFineCoefficient(anchor);
      if (!rounded_anchor.ok()) {
        return rounded_anchor.status();
      }
      anchor = *rounded_anchor;
    }
  }
  std::ranges::sort(fine.edges, [](const auto& left, const auto& right) {
    return std::tie(left.lower_unknown_ordinal, left.higher_unknown_ordinal)
        < std::tie(right.lower_unknown_ordinal, right.higher_unknown_ordinal);
  });
  auto status = CompleteLevel(fine);
  if (!status.ok()) {
    return status;
  }
  for (const double diagonal : fine.diagonal) {
    const auto rounded_diagonal = RoundedFineCoefficient(diagonal);
    if (!rounded_diagonal.ok()) {
      return rounded_diagonal.status();
    }
  }
  hierarchy.levels.push_back(std::move(fine));
  for (;;) {
    const auto blocks = ConnectedBlocks(hierarchy.levels.back());
    const std::size_t maximum_block_size = hierarchy.levels.size() == 1 ? 1 : 32;
    const bool needs_coarsening = std::ranges::any_of(blocks, [&](const auto& block) {
      return block.size() > maximum_block_size;
    });
    if (!needs_coarsening) {
      break;
    }
    auto coarse = Coarsen(hierarchy.levels.back());
    if (!coarse.ok()) {
      return coarse.status();
    }
    hierarchy.levels.push_back(std::move(*coarse));
  }
  status = FactorTerminalBlocks(hierarchy.levels.back());
  if (!status.ok()) {
    return status;
  }
  return hierarchy;
}

absl::Status InitializeMacPressure3D(
    const PreparedMacPressureView3D& system,
    std::span<double> pin_previous_pressures,
    std::span<double> pressures
) {
  std::ranges::fill(pin_previous_pressures, 0.0);
  for (const auto pin : system.gauge_cell_ordinals()) {
    pin_previous_pressures[system.component_ordinals()[pin]] = system.cells()[pin].previous_pressure;
  }
  for (std::size_t cell = 0; cell < system.cells().size(); ++cell) {
    const double pressure =
        system.cells()[cell].previous_pressure - pin_previous_pressures[system.component_ordinals()[cell]];
    if (!std::isfinite(pressure)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    pressures[cell] = pressure;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<double>> InitialMacPressure3D(const PreparedMacPressureSystem3D& system) {
  std::vector<double> pins(system.cells().size());
  std::vector<double> pressures(system.cells().size());
  const auto status = InitializeMacPressure3D(BorrowMacPressureSystem3D(system), pins, pressures);
  if (!status.ok()) {
    return status;
  }
  return pressures;
}

absl::StatusOr<MacPressureStoppingCriteria3D> MacPressureStoppingCriteria(const PreparedMacPressureSystem3D& system) {
  double rhs_norm = 0.0;
  for (const double rhs : system.integrated_rhs()) {
    rhs_norm = std::hypot(rhs_norm, rhs);
  }
  return MacPressureStoppingCriteria(rhs_norm, system.links());
}

absl::StatusOr<MacPressureStoppingCriteria3D>
MacPressureStoppingCriteria(double rhs_norm, std::span<const FrozenMacPressureLink3D> links) {
  double limit = 2e-6 + 2e-6 * rhs_norm;
  if (!std::isfinite(rhs_norm) || rhs_norm < 0.0 || !std::isfinite(limit)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  // A fixed flux floor can accept a zero pressure iterate even when an
  // internal jump is the only forcing. Preserve the original local gates and
  // bound the global residual relative to the nonzero assembled signal.
  if (rhs_norm > 0.0 && std::ranges::any_of(links, [](const FrozenMacPressureLink3D& link) {
        const auto* internal = std::get_if<InternalMacPressureLink3D>(&link.pressure_link);
        return link.conductance > 0.0 && internal && internal->pressure_jump != 0.0;
      })) {
    limit = std::min(limit, std::max(std::numeric_limits<double>::denorm_min(), 1e-3 * rhs_norm));
  }
  return MacPressureStoppingCriteria3D{.integrated_residual_limit = limit};
}
bool MeetsMacPressureStoppingCriteria3D(
    const MacPressureDiagnostics3D& diagnostics,
    const MacPressureStoppingCriteria3D& criteria
) {
  return std::isfinite(diagnostics.true_residual_l2) && std::isfinite(diagnostics.max_volume_scaled_residual)
      && std::isfinite(diagnostics.max_cell_constraint_defect)
      && diagnostics.true_residual_l2 <= criteria.integrated_residual_limit
      && diagnostics.max_volume_scaled_residual <= criteria.volume_scaled_limit
      && diagnostics.max_cell_constraint_defect <= criteria.volume_scaled_limit;
}

}  // namespace fluidity::pressure_internal
