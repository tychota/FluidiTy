#pragma once

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/base/types.h"

namespace fluidity {

// Uniform 2D MAC grid at unit depth, with nx and ny cells along the x and y axes.
// Spacing is in physical units. The origin is the corner at the low x and y boundaries.
struct Grid2D {
  std::uint32_t nx;
  std::uint32_t ny;
  double h;
  Vec2 origin{};
};

// Geometry for one oriented MAC face. Position, area at unit depth, and distance use physical units.
struct MacFaceGeometry {
  Vec2 world_position;  // World-space face center.
  double area;  // Face measure h at unit depth.
  double pressure_gradient_distance;  // h across an internal face, h/2 at a boundary.
  // The tail cell is on the low side of the face and the head cell is on the high side.
  // Their outward incidence is +1 and -1 respectively. A missing cell is -1.
  std::int64_t tail_cell_index;
  std::int64_t head_cell_index;
};

// Checks a grid on the host. Returns kInvalidArgument for zero dimensions, nonpositive spacing,
// or nonfinite inputs, and kOutOfRange for unrepresentable counts or derived geometry.
absl::Status ValidateGrid(const Grid2D& grid);

// Counts cells. The row-major cell index is row * grid.nx + column.
// Propagates ValidateGrid's errors.
absl::StatusOr<std::size_t> CellCount(const Grid2D& grid);

// Counts boundary and interior faces of both orientations: (nx + 1)*ny + nx*(ny + 1).
// Propagates ValidateGrid's errors.
absl::StatusOr<std::size_t> FaceCount(const Grid2D& grid);

// Returns the physical volume of one cell at unit depth. Propagates ValidateGrid's errors.
absl::StatusOr<double> CellVolume(const Grid2D& grid);

// Returns geometry for a zero-based face index. X-normal faces precede y-normal faces, with
// each lattice in row-major order. Distance is between neighboring cell centers or from the
// adjacent cell center to a boundary face. Returns kInvalidArgument for an invalid face index
// and propagates ValidateGrid's errors. Errors carry no face geometry.
absl::StatusOr<MacFaceGeometry> FaceGeometry(const Grid2D& grid, std::size_t face_index);

}  // namespace fluidity
