#include "fluidity/host/grid.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace fluidity {
namespace {

struct GridMetrics {
  std::size_t cell_count;
  std::size_t x_face_count;
  std::size_t face_count;
  double cell_volume;
  double boundary_center_distance;
};

// Validates geometry and counts once for the public grid queries.
absl::StatusOr<GridMetrics> CheckedGridMetrics(const Grid2D& grid) {
  if (grid.nx == 0 || grid.ny == 0) {
    return absl::InvalidArgumentError("grid dimensions must be positive");
  }
  const bool valid_spacing = grid.h > 0.0 && std::isfinite(grid.h);
  if (!valid_spacing) {
    return absl::InvalidArgumentError("grid spacing must be finite and positive");
  }
  if (!std::isfinite(grid.origin.x) || !std::isfinite(grid.origin.y)) {
    return absl::InvalidArgumentError("grid origin must be finite");
  }

  // Each product fits in uint64_t since both dimensions are uint32_t, including when
  // one factor is increased by one to account for the high boundary.
  const std::uint64_t cell_column_count = grid.nx;
  const std::uint64_t cell_row_count = grid.ny;
  const std::uint64_t cell_count = cell_column_count * cell_row_count;
  const std::uint64_t x_face_count = (cell_column_count + 1) * cell_row_count;
  const std::uint64_t y_face_count = cell_column_count * (cell_row_count + 1);
  constexpr auto max_extent = std::numeric_limits<std::size_t>::max();
  constexpr auto max_cell_index = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  const bool face_count_fits = x_face_count <= max_extent && y_face_count <= max_extent - x_face_count;
  const bool cell_indices_fit = cell_count <= max_cell_index;
  if (!face_count_fits || !cell_indices_fit) {
    return absl::OutOfRangeError("grid counts exceed the representable range");
  }

  // At unit depth, spacing is the area of a face and the center distance across an internal face.
  const double cell_volume = grid.h * grid.h;
  const double boundary_center_distance = grid.h * 0.5;
  if (cell_volume <= 0.0 || !std::isfinite(cell_volume)) {
    return absl::OutOfRangeError("cell volume must be finite and positive");
  }
  if (boundary_center_distance <= 0.0) {
    return absl::OutOfRangeError("boundary distance must be positive");
  }

  const double upper_x_coordinate = grid.origin.x + grid.h * static_cast<double>(cell_column_count);
  if (!std::isfinite(upper_x_coordinate)) {
    return absl::OutOfRangeError("upper grid x coordinate is outside the representable range");
  }
  const double upper_y_coordinate = grid.origin.y + grid.h * static_cast<double>(cell_row_count);
  if (!std::isfinite(upper_y_coordinate)) {
    return absl::OutOfRangeError("upper grid y coordinate is outside the representable range");
  }
  return GridMetrics{
      .cell_count = static_cast<std::size_t>(cell_count),
      .x_face_count = static_cast<std::size_t>(x_face_count),
      .face_count = static_cast<std::size_t>(x_face_count + y_face_count),
      .cell_volume = cell_volume,
      .boundary_center_distance = boundary_center_distance,
  };
}

}  // namespace

absl::Status ValidateGrid(const Grid2D& grid) {
  return CheckedGridMetrics(grid).status();
}

absl::StatusOr<std::size_t> CellCount(const Grid2D& grid) {
  const auto metrics = CheckedGridMetrics(grid);
  if (!metrics.ok()) {
    return metrics.status();
  }
  return metrics->cell_count;
}

absl::StatusOr<std::size_t> FaceCount(const Grid2D& grid) {
  const auto metrics = CheckedGridMetrics(grid);
  if (!metrics.ok()) {
    return metrics.status();
  }
  return metrics->face_count;
}

absl::StatusOr<double> CellVolume(const Grid2D& grid) {
  const auto metrics = CheckedGridMetrics(grid);
  if (!metrics.ok()) {
    return metrics.status();
  }
  return metrics->cell_volume;
}

absl::StatusOr<FaceInfo> FaceGeometry(const Grid2D& grid, std::size_t face_index) {
  const auto metrics = CheckedGridMetrics(grid);
  if (!metrics.ok()) {
    return metrics.status();
  }
  if (face_index >= metrics->face_count) {
    return absl::InvalidArgumentError("face index is outside the grid");
  }

  const bool is_x_normal = face_index < metrics->x_face_count;
  const std::size_t cell_column_count = grid.nx;
  const std::size_t face_row_stride = is_x_normal ? cell_column_count + 1 : cell_column_count;
  const std::size_t orientation_face_index = is_x_normal ? face_index : face_index - metrics->x_face_count;
  const std::size_t face_column = orientation_face_index % face_row_stride;
  const std::size_t face_row = orientation_face_index / face_row_stride;
  const Vec2 lattice_offset{
      .x = is_x_normal ? 0.0 : 0.5,
      .y = is_x_normal ? 0.5 : 0.0,
  };
  const Vec2 position{
      .x = grid.origin.x + grid.h * (static_cast<double>(face_column) + lattice_offset.x),
      .y = grid.origin.y + grid.h * (static_cast<double>(face_row) + lattice_offset.y),
  };

  const std::size_t normal_face_coordinate = is_x_normal ? face_column : face_row;
  const std::size_t normal_cell_count = is_x_normal ? cell_column_count : grid.ny;
  const std::size_t normal_cell_index_stride = is_x_normal ? 1 : cell_column_count;
  const std::size_t potential_head_cell_index = face_row * cell_column_count + face_column;
  const bool has_tail = normal_face_coordinate > 0;
  const bool has_head = normal_face_coordinate < normal_cell_count;
  const double face_distance = has_tail && has_head ? grid.h : metrics->boundary_center_distance;

  // The head index can lie beyond the cell array on a high boundary. Convert it only if the head exists.
  const std::int64_t tail_cell_index =
      has_tail ? static_cast<std::int64_t>(potential_head_cell_index - normal_cell_index_stride) : -1;
  const std::int64_t head_cell_index = has_head ? static_cast<std::int64_t>(potential_head_cell_index) : -1;
  return FaceInfo{
      .position = position,
      .area = grid.h,
      .distance = face_distance,
      .tail = tail_cell_index,
      .head = head_cell_index,
  };
}

}  // namespace fluidity
