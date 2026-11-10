#include "fluidity/host/whitewater_volume_3d.h"

#include <algorithm>
#include <cmath>

namespace fluidity {
namespace {
bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
}  // namespace

absl::StatusOr<WhitewaterVolume3D> RasterizeWhitewaterVolume3D(
    std::span<const WhitewaterVolumePoint3D> points,
    const WhitewaterVolumeGrid3D& grid,
    std::span<const std::uint8_t> validity,
    std::size_t maximum_voxels
) {
  if (!Finite(grid.origin) || !std::isfinite(grid.spacing_m) || grid.spacing_m <= 0
      || *std::min_element(grid.extent.begin(), grid.extent.end()) < 2) {
    return absl::InvalidArgumentError("invalid whitewater volume geometry");
  }
  if (grid.extent[0] > maximum_voxels / grid.extent[1] / grid.extent[2]) {
    return absl::ResourceExhaustedError("whitewater voxel budget exceeded");
  }
  const auto count = grid.extent[0] * grid.extent[1] * grid.extent[2];
  if (validity.size() != count || std::any_of(validity.begin(), validity.end(), [](auto v) {
        return v > 1;
      })) {
    return absl::InvalidArgumentError("invalid whitewater volume validity mask");
  }
  const double voxel_volume = std::pow(grid.spacing_m, 3);
  if (!std::isfinite(voxel_volume) || voxel_volume == 0) {
    return absl::OutOfRangeError("whitewater voxel volume is unrepresentable");
  }
  WhitewaterVolume3D result{
      .grid = grid,
      .foam_count_per_m3 = std::vector<double>(count),
      .bubble_count_per_m3 = std::vector<double>(count),
      .gas_fraction = std::vector<double>(count),
      .validity = std::vector<std::uint8_t>(validity.begin(), validity.end()),
  };
  for (const auto& point : points) {
    if (point.kind > WhitewaterKind3D::kBubble || !Finite(point.position) || !std::isfinite(point.represented_count)
        || point.represented_count <= 0 || !std::isfinite(point.gas_volume_m3) || point.gas_volume_m3 < 0
        || (point.kind != WhitewaterKind3D::kBubble && point.gas_volume_m3 != 0)) {
      return absl::InvalidArgumentError("invalid whitewater volume point");
    }
    if (point.kind == WhitewaterKind3D::kSpray) {
      continue;
    }
    const std::array coordinates{
        (point.position.x - grid.origin.x) / grid.spacing_m,
        (point.position.y - grid.origin.y) / grid.spacing_m,
        (point.position.z - grid.origin.z) / grid.spacing_m,
    };
    std::array<std::size_t, 3> cell;
    std::array<double, 3> fraction;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(coordinates[axis]) || coordinates[axis] < 0 || coordinates[axis] > grid.extent[axis] - 1) {
        return absl::FailedPreconditionError("whitewater volume support lies outside the grid");
      }
      // A point at the last centre uses only that centre, not an absent neighbor.
      const double base = std::min(std::floor(coordinates[axis]), static_cast<double>(grid.extent[axis] - 2));
      cell[axis] = static_cast<std::size_t>(base);
      fraction[axis] = coordinates[axis] - base;
    }
    for (std::size_t corner = 0; corner < 8; ++corner) {
      auto node = cell;
      double weight = 1;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const bool upper = (corner & (std::size_t{1} << axis)) != 0;
        node[axis] += upper;
        weight *= upper ? fraction[axis] : 1 - fraction[axis];
      }
      if (weight == 0) {
        continue;
      }
      const auto index = (node[0] * grid.extent[1] + node[1]) * grid.extent[2] + node[2];
      if (validity[index] == 0) {
        return absl::FailedPreconditionError("whitewater volume donor is outside the known domain");
      }
      const double count_density = point.represented_count * weight / voxel_volume;
      if (point.kind == WhitewaterKind3D::kFoam) {
        result.foam_count_per_m3[index] += count_density;
      } else {
        result.bubble_count_per_m3[index] += count_density;
        result.gas_fraction[index] += point.gas_volume_m3 * weight / voxel_volume;
      }
      if (!std::isfinite(result.foam_count_per_m3[index]) || !std::isfinite(result.bubble_count_per_m3[index])
          || !std::isfinite(result.gas_fraction[index])) {
        return absl::OutOfRangeError("whitewater volume accumulation overflowed");
      }
    }
  }
  return result;
}
}  // namespace fluidity
