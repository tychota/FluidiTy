#include "fluidity/host/liquid_air_components.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include "fluidity/host/views.h"

namespace fluidity {
namespace {

auto CellKey(CellIndex3D cell) {
  return std::tuple{cell.z, cell.y, cell.x};
}

std::size_t NeighborOrdinal(std::span<const LiquidAirVolumeCell3D> cells, CellIndex3D center, std::size_t passage) {
  std::array<std::int32_t, 3> neighbor{center.x, center.y, center.z};
  auto& coordinate = neighbor[passage / 2];
  const bool high = passage % 2 != 0;
  if ((high && coordinate == std::numeric_limits<std::int32_t>::max())
      || (!high && coordinate == std::numeric_limits<std::int32_t>::min())) {
    return cells.size();
  }
  coordinate += high ? 1 : -1;
  const auto key = CellKey({neighbor[0], neighbor[1], neighbor[2]});
  const auto found = std::ranges::lower_bound(cells, key, {}, [](const LiquidAirVolumeCell3D& cell) {
    return CellKey(cell.cell_index);
  });
  return found != cells.end() && CellKey(found->cell_index) == key ? static_cast<std::size_t>(found - cells.begin())
                                                                   : cells.size();
}

}  // namespace

absl::StatusOr<std::size_t> LabelLiquidAirComponents3D(
    std::span<const LiquidAirVolumeCell3D> cells,
    std::span<std::size_t> labels,
    std::span<std::size_t> queue,
    std::span<LiquidAirComponent3D> components
) {
  if (labels.size() != cells.size() || queue.size() < cells.size()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const std::array<std::span<const std::byte>, 4> storage{
      std::as_bytes(cells),
      std::as_bytes(labels),
      std::as_bytes(queue),
      std::as_bytes(components),
  };
  for (std::size_t left = 0; left < storage.size(); ++left) {
    for (std::size_t right = left + 1; right < storage.size(); ++right) {
      if (ByteViewsOverlap(storage[left], storage[right])) {
        return absl::Status(absl::StatusCode::kInvalidArgument, {});
      }
    }
  }
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const auto& record = cells[cell];
    if (!std::isfinite(record.gas_volume) || record.gas_volume <= 0.0
        || (cell > 0 && CellKey(cells[cell - 1].cell_index) >= CellKey(record.cell_index))) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (std::ranges::any_of(record.passages, [](LiquidAirPassage3D passage) {
          return passage > LiquidAirPassage3D::kAtmosphere;
        })) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  const std::size_t unassigned = cells.size();
  std::ranges::fill(labels, unassigned);
  std::size_t component_count = 0;
  for (std::size_t seed = 0; seed < cells.size(); ++seed) {
    if (labels[seed] != unassigned) {
      continue;
    }
    if (component_count == components.size()) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    bool atmosphere = false;
    bool unknown_frontier = false;
    double gas_volume = 0.0;
    std::size_t queued = 1;
    queue[0] = seed;
    labels[seed] = component_count;
    for (std::size_t cursor = 0; cursor < queued; ++cursor) {
      const auto& cell = cells[queue[cursor]];
      gas_volume += cell.gas_volume;
      if (!std::isfinite(gas_volume)) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      for (std::size_t passage = 0; passage < cell.passages.size(); ++passage) {
        const auto state = cell.passages[passage];
        const std::size_t neighbor = NeighborOrdinal(cells, cell.cell_index, passage);
        if (neighbor == cells.size()) {
          atmosphere |= state == LiquidAirPassage3D::kAtmosphere;
          unknown_frontier |= state == LiquidAirPassage3D::kOpen;
          continue;
        }
        if (state == LiquidAirPassage3D::kAtmosphere || cells[neighbor].passages[passage ^ 1] != state) {
          return absl::Status(absl::StatusCode::kInvalidArgument, {});
        }
        if (state == LiquidAirPassage3D::kOpen && labels[neighbor] == unassigned) {
          labels[neighbor] = component_count;
          queue[queued++] = neighbor;
        }
      }
    }
    auto connectivity = LiquidAirConnectivity3D::kEnclosed;
    if (atmosphere) {
      connectivity = LiquidAirConnectivity3D::kVented;
    } else if (unknown_frontier) {
      connectivity = LiquidAirConnectivity3D::kUnclassified;
    }
    components[component_count++] = {
        .first_cell = cells[seed].cell_index,
        .gas_volume = gas_volume,
        .connectivity = connectivity,
    };
  }
  return component_count;
}

absl::StatusOr<std::size_t> MeasureLiquidAirComponentOverlaps3D(
    LiquidAirComponentSnapshot3D previous,
    LiquidAirComponentSnapshot3D current,
    std::span<LiquidAirComponentOverlap3D> overlaps
) {
  const auto output = std::as_bytes(overlaps);
  for (const auto snapshot : {previous, current}) {
    if (snapshot.labels.size() != snapshot.cells.size() || snapshot.component_count > snapshot.cells.size()
        || ByteViewsOverlap(output, std::as_bytes(snapshot.cells))
        || ByteViewsOverlap(output, std::as_bytes(snapshot.labels))) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    std::size_t next_component = 0;
    for (std::size_t index = 0; index < snapshot.cells.size(); ++index) {
      const auto& cell = snapshot.cells[index];
      const std::size_t label = snapshot.labels[index];
      if (!std::isfinite(cell.gas_volume) || cell.gas_volume <= 0.0 || label >= snapshot.component_count
          || label > next_component
          || (index > 0 && CellKey(snapshot.cells[index - 1].cell_index) >= CellKey(cell.cell_index))) {
        return absl::Status(absl::StatusCode::kInvalidArgument, {});
      }
      if (label == next_component) {
        ++next_component;
      }
    }
    if (next_component != snapshot.component_count) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }

  std::size_t previous_index = 0;
  std::size_t current_index = 0;
  std::size_t overlap_count = 0;
  while (previous_index < previous.cells.size() && current_index < current.cells.size()) {
    const auto& previous_cell = previous.cells[previous_index];
    const auto& current_cell = current.cells[current_index];
    const auto previous_key = CellKey(previous_cell.cell_index);
    const auto current_key = CellKey(current_cell.cell_index);
    if (previous_key < current_key) {
      ++previous_index;
      continue;
    }
    if (current_key < previous_key) {
      ++current_index;
      continue;
    }
    if (overlap_count == overlaps.size()) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    overlaps[overlap_count++] = {
        .previous_component = previous.labels[previous_index],
        .current_component = current.labels[current_index],
        .shared_cell_volume = std::min(previous_cell.gas_volume, current_cell.gas_volume),
    };
    ++previous_index;
    ++current_index;
  }

  auto entries = overlaps.first(overlap_count);
  // The volume tie-breaker fixes summation order without allocating sort scratch.
  std::ranges::sort(entries, {}, [](const LiquidAirComponentOverlap3D& edge) {
    return std::tuple{edge.previous_component, edge.current_component, edge.shared_cell_volume};
  });
  std::size_t edge_count = 0;
  for (const auto edge : entries) {
    if (edge_count > 0 && overlaps[edge_count - 1].previous_component == edge.previous_component
        && overlaps[edge_count - 1].current_component == edge.current_component) {
      auto& volume = overlaps[edge_count - 1].shared_cell_volume;
      volume += edge.shared_cell_volume;
      if (!std::isfinite(volume)) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
      continue;
    }
    overlaps[edge_count++] = edge;
  }
  return edge_count;
}

}  // namespace fluidity
