#include "fluidity/host/adaptive_mac_topology_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <tuple>

#include "absl/status/status.h"

namespace fluidity {
namespace {

constexpr std::uint8_t kMaxWidthExponent = 30;
constexpr std::size_t kMaxLeafCount = 4096;
constexpr std::size_t kMaxInteriorFaceCount = 6 * kMaxLeafCount;
using LatticeCoordinates = std::array<std::int64_t, 3>;

LatticeCoordinates Coordinates(CellIndex3D index) {
  return {index.x, index.y, index.z};
}

auto CellOrder(const LatticeCoordinates& coordinates) {
  return std::tuple{coordinates[2], coordinates[1], coordinates[0]};
}

auto FaceOrder(const AdaptiveMacInteriorFace3D& face) {
  const auto& index = face.face_index.lattice_index;
  return std::tuple{face.face_index.axis, index.z, index.y, index.x, face.width_exponent};
}

std::int64_t WidthInFinestCells(std::uint8_t width_exponent) {
  return std::int64_t{1} << width_exponent;
}

std::int64_t FloorToWidth(std::int64_t coordinate, std::int64_t width) {
  const std::int64_t remainder = coordinate % width;
  return coordinate - remainder - (remainder < 0 ? width : 0);
}

bool IsFinitePositive(double value) {
  return std::isfinite(value) && value > 0;
}

std::optional<std::size_t>
FindAtLowerCorner(std::span<const AdaptiveMacLeaf3D> cells, const LatticeCoordinates& lower) {
  const auto order = CellOrder(lower);
  const auto cell = std::ranges::lower_bound(cells, order, {}, [](const AdaptiveMacLeaf3D& candidate) {
    return CellOrder(Coordinates(candidate.key.lower_corner));
  });
  if (cell == cells.end() || CellOrder(Coordinates(cell->key.lower_corner)) != order) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(cell - cells.begin());
}

std::optional<std::size_t> FindContaining(
    std::span<const AdaptiveMacLeaf3D> cells,
    const LatticeCoordinates& point,
    std::uint8_t max_width_exponent
) {
  for (std::uint8_t exponent = 0; exponent <= max_width_exponent; ++exponent) {
    const std::int64_t width = WidthInFinestCells(exponent);
    const LatticeCoordinates lower{
        FloorToWidth(point[0], width),
        FloorToWidth(point[1], width),
        FloorToWidth(point[2], width),
    };
    const auto ordinal = FindAtLowerCorner(cells, lower);
    if (ordinal && cells[*ordinal].key.width_exponent == exponent) {
      return ordinal;
    }
  }
  return std::nullopt;
}

struct ContactSamples {
  std::array<std::int64_t, 2> coordinates;
  std::size_t count;

  std::span<const std::int64_t> values() const {
    return std::span(coordinates).first(count);
  }
};

// Sample the adjacent finest cell immediately outside each constrained
// coordinate. In a free coordinate, one point per possible child half
// resolves every 2:1 contact. Finer gaps missed on the coarse side are
// detected from the fine leaf, whose own probes hit the coarser leaf.
ContactSamples ContactAxis(std::int64_t lower, std::int64_t width, int direction) {
  if (direction < 0) {
    return {.coordinates = {lower - 1, 0}, .count = 1};
  }
  if (direction > 0) {
    return {.coordinates = {lower + width, 0}, .count = 1};
  }
  return {.coordinates = {lower + width / 4, lower + 3 * width / 4}, .count = width == 1 ? 1U : 2U};
}

AdaptiveMacInteriorFace3D Subface(
    std::span<const AdaptiveMacLeaf3D> cells,
    std::size_t axis,
    std::size_t tail_ordinal,
    std::size_t head_ordinal
) {
  const auto& tail = cells[tail_ordinal];
  const auto& head = cells[head_ordinal];
  const auto tail_lower = Coordinates(tail.key.lower_corner);
  const auto head_lower = Coordinates(head.key.lower_corner);
  LatticeCoordinates lower{
      std::max(tail_lower[0], head_lower[0]),
      std::max(tail_lower[1], head_lower[1]),
      std::max(tail_lower[2], head_lower[2]),
  };
  lower[axis] = head_lower[axis];
  const auto exponent = std::min(tail.key.width_exponent, head.key.width_exponent);
  const double face_width = std::min(tail.cell_width, head.cell_width);
  return {
      .face_index =
          {
              static_cast<FaceNormalAxis>(axis),
              {static_cast<std::int32_t>(lower[0]),
               static_cast<std::int32_t>(lower[1]),
               static_cast<std::int32_t>(lower[2])},
          },
      .width_exponent = exponent,
      .tail_cell_ordinal = tail_ordinal,
      .head_cell_ordinal = head_ordinal,
      .face_area = face_width * face_width,
      .cell_center_distance = tail.cell_width / 2 + head.cell_width / 2,
  };
}

absl::Status AppendInteriorFaces(
    std::span<const AdaptiveMacLeaf3D> cells,
    std::size_t cell_ordinal,
    std::uint8_t max_width_exponent,
    std::vector<AdaptiveMacInteriorFace3D>& faces
) {
  const auto& cell = cells[cell_ordinal];
  const auto lower = Coordinates(cell.key.lower_corner);
  const std::int64_t width = WidthInFinestCells(cell.key.width_exponent);
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int constrained_axes = std::abs(dx) + std::abs(dy) + std::abs(dz);
        if (constrained_axes == 0) {
          continue;
        }
        const std::array direction{dx, dy, dz};
        const auto x_samples = ContactAxis(lower[0], width, dx);
        const auto y_samples = ContactAxis(lower[1], width, dy);
        const auto z_samples = ContactAxis(lower[2], width, dz);
        std::array<std::size_t, 4> face_neighbors{};
        std::size_t neighbor_count = 0;
        const bool high_face = constrained_axes == 1 && dx + dy + dz == 1;
        for (const auto z : z_samples.values()) {
          for (const auto y : y_samples.values()) {
            for (const auto x : x_samples.values()) {
              const auto neighbor = FindContaining(cells, {x, y, z}, max_width_exponent);
              if (!neighbor) {
                continue;
              }
              const int level_difference = cells[*neighbor].key.width_exponent - cell.key.width_exponent;
              if (std::abs(level_difference) > 1) {
                return absl::FailedPreconditionError("touching adaptive leaves must be graded 2:1");
              }
              if (high_face) {
                const auto known = std::span(face_neighbors).first(neighbor_count);
                if (std::ranges::find(known, *neighbor) == known.end()) {
                  face_neighbors[neighbor_count++] = *neighbor;
                }
              }
            }
          }
        }
        if (neighbor_count == 0) {
          continue;
        }
        const std::size_t axis = static_cast<std::size_t>(std::ranges::find(direction, 1) - direction.begin());
        for (const std::size_t neighbor : std::span(face_neighbors).first(neighbor_count)) {
          if (faces.size() == kMaxInteriorFaceCount) {
            return absl::ResourceExhaustedError("adaptive interior face capacity exceeded");
          }
          const auto face = Subface(cells, axis, cell_ordinal, neighbor);
          if (!IsFinitePositive(face.face_area) || !IsFinitePositive(face.cell_center_distance)) {
            return absl::OutOfRangeError("adaptive face measures are unrepresentable");
          }
          faces.push_back(face);
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<AdaptiveMacTopology3D>
AdaptiveMacTopology3D::Create(std::span<const AdaptiveMacCellKey3D> leaf_keys, double finest_cell_width) {
  if (leaf_keys.size() > kMaxLeafCount) {
    return absl::ResourceExhaustedError("adaptive leaf capacity exceeded");
  }
  if (leaf_keys.empty() || !IsFinitePositive(finest_cell_width)) {
    return absl::InvalidArgumentError("adaptive topology requires leaves and a positive finite finest width");
  }
  AdaptiveMacTopology3D result;
  result.finest_cell_width_ = finest_cell_width;
  result.cells_.reserve(leaf_keys.size());
  std::uint8_t max_width_exponent = 0;
  for (const auto& key : leaf_keys) {
    if (key.width_exponent > kMaxWidthExponent) {
      return absl::InvalidArgumentError("adaptive width exponent exceeds the signed lattice limit");
    }
    const std::int64_t width = WidthInFinestCells(key.width_exponent);
    for (const std::int64_t coordinate : Coordinates(key.lower_corner)) {
      if (coordinate % width != 0) {
        return absl::InvalidArgumentError("adaptive lower corner must align to its dyadic width");
      }
      if (coordinate + width > std::numeric_limits<std::int32_t>::max()) {
        return absl::OutOfRangeError("adaptive high face exceeds the signed lattice");
      }
    }
    const double cell_width = std::ldexp(finest_cell_width, key.width_exponent);
    const double cell_volume = cell_width * cell_width * cell_width;
    if (!IsFinitePositive(cell_volume)) {
      return absl::OutOfRangeError("adaptive leaf volume is unrepresentable");
    }
    max_width_exponent = std::max(max_width_exponent, key.width_exponent);
    result.cells_.push_back({.key = key, .cell_width = cell_width, .cell_volume = cell_volume});
  }
  const auto order = [](const AdaptiveMacLeaf3D& cell) {
    return CellOrder(Coordinates(cell.key.lower_corner));
  };
  std::ranges::sort(result.cells_, {}, order);
  if (std::ranges::adjacent_find(result.cells_, {}, order) != result.cells_.end()) {
    return absl::InvalidArgumentError("adaptive leaves must have unique lower corners");
  }

  // Dyadically aligned Cartesian leaves can overlap only by ancestry.
  for (const auto& cell : result.cells_) {
    const auto lower = Coordinates(cell.key.lower_corner);
    for (std::uint8_t exponent = cell.key.width_exponent + 1; exponent <= max_width_exponent; ++exponent) {
      const std::int64_t width = WidthInFinestCells(exponent);
      const LatticeCoordinates ancestor_lower{
          FloorToWidth(lower[0], width),
          FloorToWidth(lower[1], width),
          FloorToWidth(lower[2], width),
      };
      const auto ancestor = FindAtLowerCorner(result.cells_, ancestor_lower);
      if (ancestor && result.cells_[*ancestor].key.width_exponent == exponent) {
        return absl::InvalidArgumentError("adaptive leaves overlap");
      }
    }
  }
  for (std::size_t cell = 0; cell < result.cells_.size(); ++cell) {
    const auto status = AppendInteriorFaces(result.cells_, cell, max_width_exponent, result.interior_faces_);
    if (!status.ok()) {
      return status;
    }
  }
  std::ranges::sort(result.interior_faces_, {}, FaceOrder);
  return result;
}

std::span<const AdaptiveMacLeaf3D> AdaptiveMacTopology3D::cells() const {
  return cells_;
}

std::span<const AdaptiveMacInteriorFace3D> AdaptiveMacTopology3D::interior_faces() const {
  return interior_faces_;
}

}  // namespace fluidity
