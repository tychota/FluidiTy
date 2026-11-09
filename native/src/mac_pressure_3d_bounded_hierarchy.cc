#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>

#include "mac_pressure_3d_workspace_internal.h"

namespace fluidity::pressure_internal {
namespace {
#define FLUIDITY_PRESSURE_TAKE(name, type, count) \
  auto name##_result = scratch.Take<type>(count); \
  if (!name##_result.ok()) { \
    return name##_result.status(); \
  } \
  auto name = *name##_result

bool PositiveFinite(double value) {
  return value > 0 && std::isfinite(value);
}
absl::StatusOr<double> RoundedFineCoefficient(double coefficient) {
  if (!PositiveFinite(coefficient) || coefficient > std::numeric_limits<float>::max()) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  const float rounded = static_cast<float>(coefficient);
  if (!PositiveFinite(rounded)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return static_cast<double>(rounded);
}
std::size_t FindRoot(std::span<std::size_t> roots, std::size_t unknown) {
  while (roots[unknown] != unknown) {
    roots[unknown] = roots[roots[unknown]];
    unknown = roots[unknown];
  }
  return unknown;
}
std::int32_t FloorHalf(std::int32_t coordinate) {
  return coordinate / 2 - (coordinate % 2 < 0 ? 1 : 0);
}
auto HalfKey(CellIndex3D index) {
  return std::tuple{FloorHalf(index.z), FloorHalf(index.y), FloorHalf(index.x)};
}

absl::Status CompleteLevel(MacPressureLevelView3D& level, MacPressureScratch3D& scratch) {
  const std::size_t count = level.coordinates.size();
  FLUIDITY_PRESSURE_TAKE(diagonal, double, count);
  FLUIDITY_PRESSURE_TAKE(offsets, std::size_t, count + 1);
  FLUIDITY_PRESSURE_TAKE(components, std::size_t, count);
  FLUIDITY_PRESSURE_TAKE(neighbors, MacPressureNeighbor3D, 2 * level.edges.size());
  level.diagonal = diagonal;
  level.row_offsets = offsets;
  level.component_ordinals = components;
  level.neighbors = neighbors;
  std::ranges::copy(level.anchor_diagonal, diagonal.begin());
  std::iota(components.begin(), components.end(), std::size_t{0});
  for (const auto& edge : level.edges) {
    diagonal[edge.lower_unknown_ordinal] += edge.conductance;
    diagonal[edge.higher_unknown_ordinal] += edge.conductance;
    ++offsets[edge.lower_unknown_ordinal + 1];
    ++offsets[edge.higher_unknown_ordinal + 1];
    const auto first = FindRoot(components, edge.lower_unknown_ordinal);
    const auto second = FindRoot(components, edge.higher_unknown_ordinal);
    components[std::max(first, second)] = std::min(first, second);
  }
  for (std::size_t unknown = 0; unknown < count; ++unknown) {
    if (!PositiveFinite(diagonal[unknown]) || !std::isfinite(level.anchor_diagonal[unknown])) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    components[unknown] = FindRoot(components, unknown);
    offsets[unknown + 1] += offsets[unknown];
  }
  MacPressureScratchMark3D temporary(scratch);
  FLUIDITY_PRESSURE_TAKE(next_offset, std::size_t, count + 1);
  std::ranges::copy(offsets, next_offset.begin());
  for (std::size_t ordinal = 0; ordinal < level.edges.size(); ++ordinal) {
    const auto& edge = level.edges[ordinal];
    neighbors[next_offset[edge.lower_unknown_ordinal]++] = {edge.higher_unknown_ordinal, ordinal};
    neighbors[next_offset[edge.higher_unknown_ordinal]++] = {edge.lower_unknown_ordinal, ordinal};
  }
  for (std::size_t unknown = 0; unknown < count; ++unknown) {
    std::sort(
        neighbors.begin() + offsets[unknown],
        neighbors.begin() + offsets[unknown + 1],
        [](const auto& left, const auto& right) {
          return left.unknown_ordinal < right.unknown_ordinal;
        }
    );
  }
  return absl::OkStatus();
}

// Group by connected block, signed half-coordinate, and incoming unknown.
// Crossing edges sort by endpoints then original ordinal, preserving each
// map sum's original fine-edge accumulation order exactly.
absl::StatusOr<MacPressureLevelView3D> Coarsen(MacPressureLevelView3D& fine, MacPressureScratch3D& scratch) {
  const std::size_t count = fine.coordinates.size();
  FLUIDITY_PRESSURE_TAKE(aggregates, std::size_t, count);
  fine.aggregate_ordinals = aggregates;
  std::fill(aggregates.begin(), aggregates.end(), kNoPressureUnknown3D);
  FLUIDITY_PRESSURE_TAKE(coordinates, CellIndex3D, count);
  FLUIDITY_PRESSURE_TAKE(anchors, double, count);
  FLUIDITY_PRESSURE_TAKE(edges, MacPressureEdge3D, fine.edges.size());
  MacPressureLevelView3D coarse;
  std::size_t coarse_count = 0;
  std::size_t edge_count = 0;
  {
    MacPressureScratchMark3D temporary(scratch);
    FLUIDITY_PRESSURE_TAKE(order, std::size_t, count);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::sort(order, [&](std::size_t left, std::size_t right) {
      return std::tuple{fine.component_ordinals[left], HalfKey(fine.coordinates[left]), left}
      < std::tuple{fine.component_ordinals[right], HalfKey(fine.coordinates[right]), right};
    });
    for (std::size_t begin = 0; begin < count;) {
      std::size_t end = begin + 1;
      while (end < count && fine.component_ordinals[order[end]] == fine.component_ordinals[order[begin]]) {
        ++end;
      }
      std::size_t groups = 1;
      for (std::size_t index = begin + 1; index < end; ++index) {
        groups += HalfKey(fine.coordinates[order[index]]) != HalfKey(fine.coordinates[order[index - 1]]);
      }
      if (groups < end - begin || end - begin == 1) {
        for (std::size_t index = begin; index < end; ++index) {
          if (index == begin
              || HalfKey(fine.coordinates[order[index]]) != HalfKey(fine.coordinates[order[index - 1]])) {
            const auto [z, y, x] = HalfKey(fine.coordinates[order[index]]);
            coordinates[coarse_count++] = {x, y, z};
          }
          aggregates[order[index]] = coarse_count - 1;
        }
      } else {
        std::sort(order.begin() + begin, order.begin() + end);
        for (std::size_t index = begin; index < end; ++index) {
          const auto unknown = order[index];
          if (aggregates[unknown] != kNoPressureUnknown3D) {
            continue;
          }
          const auto [z, y, x] = HalfKey(fine.coordinates[unknown]);
          coordinates[coarse_count] = {x, y, z};
          aggregates[unknown] = coarse_count;
          for (std::size_t offset = fine.row_offsets[unknown]; offset < fine.row_offsets[unknown + 1]; ++offset) {
            const auto neighbor = fine.neighbors[offset].unknown_ordinal;
            if (aggregates[neighbor] == kNoPressureUnknown3D) {
              aggregates[neighbor] = coarse_count;
              break;
            }
          }
          ++coarse_count;
        }
      }
      begin = end;
    }
    for (std::size_t unknown = 0; unknown < count; ++unknown) {
      anchors[aggregates[unknown]] += fine.anchor_diagonal[unknown];
    }
    FLUIDITY_PRESSURE_TAKE(edge_order, std::size_t, fine.edges.size());
    std::iota(edge_order.begin(), edge_order.end(), std::size_t{0});
    const auto endpoints = [&](std::size_t ordinal) {
      const auto& edge = fine.edges[ordinal];
      const auto pair = std::minmax(aggregates[edge.lower_unknown_ordinal], aggregates[edge.higher_unknown_ordinal]);
      return std::pair<std::size_t, std::size_t>{pair.first, pair.second};
    };
    std::ranges::sort(edge_order, [&](std::size_t left, std::size_t right) {
      return std::pair{endpoints(left), left} < std::pair{endpoints(right), right};
    });
    for (const auto ordinal : edge_order) {
      const auto [first, second] = endpoints(ordinal);
      if (first == second) {
        continue;
      }
      if (edge_count == 0 || edges[edge_count - 1].lower_unknown_ordinal != first
          || edges[edge_count - 1].higher_unknown_ordinal != second) {
        edges[edge_count++] = {first, second, 0};
      }
      edges[edge_count - 1].conductance += fine.edges[ordinal].conductance;
      if (!PositiveFinite(edges[edge_count - 1].conductance)) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
    }
  }
  coarse.coordinates = coordinates.first(coarse_count);
  coarse.anchor_diagonal = anchors.first(coarse_count);
  coarse.edges = edges.first(edge_count);
  const auto status = CompleteLevel(coarse, scratch);
  if (!status.ok()) {
    return status;
  }
  return coarse;
}

absl::Status FactorTerminal(MacPressureLevelView3D& level, MacPressureScratch3D& scratch) {
  const std::size_t count = level.coordinates.size();
  FLUIDITY_PRESSURE_TAKE(offsets, std::size_t, count + 1);
  FLUIDITY_PRESSURE_TAKE(unknowns, std::size_t, count);
  FLUIDITY_PRESSURE_TAKE(factors, double, 32 * count);
  level.terminal_unknowns = unknowns;
  std::iota(unknowns.begin(), unknowns.end(), std::size_t{0});
  std::ranges::sort(unknowns, [&](std::size_t left, std::size_t right) {
    return std::pair{level.component_ordinals[left], left} < std::pair{level.component_ordinals[right], right};
  });
  MacPressureScratchMark3D temporary(scratch);
  FLUIDITY_PRESSURE_TAKE(local_ordinals, std::size_t, count);
  std::size_t block_count = 0;
  std::size_t factor_count = 0;
  for (std::size_t begin = 0; begin < count;) {
    std::size_t end = begin + 1;
    while (end < count && level.component_ordinals[unknowns[end]] == level.component_ordinals[unknowns[begin]]) {
      ++end;
    }
    const std::size_t block_size = end - begin;
    if (block_size > 32) {
      return absl::Status(absl::StatusCode::kInternal, {});
    }
    offsets[block_count++] = begin;
    auto factor = factors.subspan(factor_count, block_size * block_size);
    for (std::size_t local = 0; local < block_size; ++local) {
      local_ordinals[unknowns[begin + local]] = local;
    }
    for (std::size_t row = 0; row < block_size; ++row) {
      const auto unknown = unknowns[begin + row];
      factor[row * block_size + row] = level.diagonal[unknown];
      for (std::size_t offset = level.row_offsets[unknown]; offset < level.row_offsets[unknown + 1]; ++offset) {
        const auto neighbor = level.neighbors[offset];
        factor[row * block_size + local_ordinals[neighbor.unknown_ordinal]] =
            -level.edges[neighbor.edge_ordinal].conductance;
      }
    }
    for (std::size_t row = 0; row < block_size; ++row) {
      for (std::size_t column = 0; column <= row; ++column) {
        double coefficient = factor[row * block_size + column];
        for (std::size_t inner = 0; inner < column; ++inner) {
          coefficient -= factor[row * block_size + inner] * factor[column * block_size + inner];
        }
        if (row == column) {
          if (!PositiveFinite(coefficient)) {
            return absl::Status(absl::StatusCode::kOutOfRange, {});
          }
          factor[row * block_size + column] = std::sqrt(coefficient);
        } else {
          factor[row * block_size + column] = coefficient / factor[column * block_size + column];
          if (!std::isfinite(factor[row * block_size + column])) {
            return absl::Status(absl::StatusCode::kOutOfRange, {});
          }
        }
      }
      for (std::size_t column = row + 1; column < block_size; ++column) {
        factor[row * block_size + column] = 0;
      }
    }
    factor_count += block_size * block_size;
    begin = end;
  }
  offsets[block_count] = count;
  level.terminal_offsets = offsets.first(block_count + 1);
  level.terminal_factors = factors.first(factor_count);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<MacPressureHierarchyView3D>
BuildBoundedMacPressureHierarchy3D(const PreparedMacPressureView3D& system, MacPressureScratch3D& scratch) {
  const std::size_t count = system.cells().size();
  FLUIDITY_PRESSURE_TAKE(unknown_to_cell, std::size_t, count);
  FLUIDITY_PRESSURE_TAKE(cell_to_unknown, std::size_t, count);
  FLUIDITY_PRESSURE_TAKE(levels, MacPressureLevelView3D, kMaximumPressureLevels3D);
  std::fill(cell_to_unknown.begin(), cell_to_unknown.end(), kNoPressureUnknown3D);
  FLUIDITY_PRESSURE_TAKE(coordinates, CellIndex3D, count);
  FLUIDITY_PRESSURE_TAKE(anchors, double, count);
  FLUIDITY_PRESSURE_TAKE(edges, MacPressureEdge3D, system.links().size());
  std::size_t unknown_count = 0;
  std::size_t edge_count = 0;
  for (std::size_t cell = 0; cell < count; ++cell) {
    if (std::ranges::binary_search(system.gauges, cell)) {
      continue;
    }
    cell_to_unknown[cell] = unknown_count;
    unknown_to_cell[unknown_count] = cell;
    coordinates[unknown_count] = system.cell_records[cell].cell_index;
    anchors[unknown_count++] = system.cell_records[cell].compliance;
  }
  for (const auto& link : system.links()) {
    if (link.conductance == 0) {
      continue;
    }
    const auto tail = link.tail_cell_ordinal ? cell_to_unknown[*link.tail_cell_ordinal] : kNoPressureUnknown3D;
    const auto head = link.head_cell_ordinal ? cell_to_unknown[*link.head_cell_ordinal] : kNoPressureUnknown3D;
    if (tail != kNoPressureUnknown3D && head != kNoPressureUnknown3D) {
      const auto coefficient = RoundedFineCoefficient(link.conductance);
      if (!coefficient.ok()) {
        return coefficient.status();
      }
      edges[edge_count++] = {std::min(tail, head), std::max(tail, head), *coefficient};
    } else if (tail != kNoPressureUnknown3D) {
      anchors[tail] += link.conductance;
    } else if (head != kNoPressureUnknown3D) {
      anchors[head] += link.conductance;
    }
  }
  for (auto& anchor : anchors.first(unknown_count)) {
    if (anchor != 0) {
      const auto rounded = RoundedFineCoefficient(anchor);
      if (!rounded.ok()) {
        return rounded.status();
      }
      anchor = *rounded;
    }
  }
  auto& fine = levels[0];
  fine.coordinates = coordinates.first(unknown_count);
  fine.anchor_diagonal = anchors.first(unknown_count);
  fine.edges = edges.first(edge_count);
  std::ranges::sort(fine.edges, [](const auto& left, const auto& right) {
    return std::tie(left.lower_unknown_ordinal, left.higher_unknown_ordinal)
        < std::tie(right.lower_unknown_ordinal, right.higher_unknown_ordinal);
  });
  auto status = CompleteLevel(fine, scratch);
  if (!status.ok()) {
    return status;
  }
  for (const double diagonal : fine.diagonal) {
    const auto rounded = RoundedFineCoefficient(diagonal);
    if (!rounded.ok()) {
      return rounded.status();
    }
  }
  std::size_t level_count = 1;
  for (;;) {
    bool needs_coarsening = false;
    {
      MacPressureScratchMark3D temporary(scratch);
      const auto& level = levels[level_count - 1];
      FLUIDITY_PRESSURE_TAKE(sizes, std::size_t, level.coordinates.size());
      for (const auto component : level.component_ordinals) {
        ++sizes[component];
      }
      const std::size_t maximum_block_size = level_count == 1 ? 1 : 32;
      needs_coarsening = std::ranges::any_of(sizes, [&](auto size) {
        return size > maximum_block_size;
      });
    }
    if (!needs_coarsening) {
      break;
    }
    if (level_count == levels.size()) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    auto coarse = Coarsen(levels[level_count - 1], scratch);
    if (!coarse.ok()) {
      return coarse.status();
    }
    levels[level_count++] = *coarse;
  }
  status = FactorTerminal(levels[level_count - 1], scratch);
  if (!status.ok()) {
    return status;
  }
  return MacPressureHierarchyView3D{unknown_to_cell.first(unknown_count), cell_to_unknown, levels.first(level_count)};
}

#undef FLUIDITY_PRESSURE_TAKE
}  // namespace fluidity::pressure_internal
