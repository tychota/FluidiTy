#include "fluidity/host/surface_field_3d.h"

#include <algorithm>
#include <cmath>

namespace fluidity {
namespace {
bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
bool ValidLattice(const SurfaceLattice3D& lattice) {
  const auto extent = lattice.extent;
  return Finite(lattice.origin) && std::isfinite(lattice.spacing) && lattice.spacing > 0
      && *std::min_element(extent.begin(), extent.end()) >= 2
      && extent[0] <= lattice.values.size() / extent[1] / extent[2]
      && extent[0] * extent[1] * extent[2] == lattice.values.size();
}
}  // namespace

absl::StatusOr<SurfaceLattice3D> BuildSurfaceDensity3D(
    std::span<const SurfaceMaterialPoint3D> particles,
    const SurfaceFieldOptions3D& options,
    std::size_t maximum_nodes
) {
  const auto extent = options.extent;
  if (!Finite(options.origin) || !std::isfinite(options.spacing_m) || options.spacing_m <= 0
      || !std::isfinite(options.sigma_m) || options.sigma_m <= 0
      || *std::min_element(extent.begin(), extent.end()) < 2) {
    return absl::InvalidArgumentError("invalid density field geometry");
  }
  for (const auto point : particles) {
    if (!Finite(point.position) || !std::isfinite(point.volume_m3) || point.volume_m3 <= 0) {
      return absl::InvalidArgumentError("invalid material point");
    }
  }
  if (extent[0] > maximum_nodes / extent[1] / extent[2]) {
    return absl::ResourceExhaustedError("density lattice node budget exceeded");
  }
  const double sigma_nodes = options.sigma_m / options.spacing_m;
  const double radius_value = std::floor(4 * sigma_nodes + 0.5);
  if (!std::isfinite(radius_value) || radius_value > 4096) {
    return absl::ResourceExhaustedError("density kernel budget exceeded");
  }
  if (sigma_nodes == 0 || radius_value < 1) {
    return absl::InvalidArgumentError("density kernel is unresolved");
  }
  const auto radius = static_cast<std::size_t>(radius_value);
  const std::size_t count = extent[0] * extent[1] * extent[2];
  SurfaceLattice3D lattice{
      .origin = options.origin,
      .spacing = options.spacing_m,
      .extent = extent,
      .values = std::vector<double>(count)
  };
  const auto index = [&](std::size_t x, std::size_t y, std::size_t z) {
    return (x * extent[1] + y) * extent[2] + z;
  };
  const double node_volume = std::pow(options.spacing_m, 3);
  if (!std::isfinite(node_volume) || node_volume == 0) {
    return absl::OutOfRangeError("density node volume is unrepresentable");
  }
  for (const auto point : particles) {
    const std::array coordinates{
        (point.position.x - options.origin.x) / options.spacing_m,
        (point.position.y - options.origin.y) / options.spacing_m,
        (point.position.z - options.origin.z) / options.spacing_m,
    };
    std::array<std::size_t, 3> cell;
    std::array<double, 3> fraction;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double base = std::floor(coordinates[axis]);
      // Leave a complete known-air shell beyond the finite filter support.
      if (!std::isfinite(base) || base < radius + 1.0 || base + radius + 2 >= extent[axis]) {
        return absl::FailedPreconditionError("density support clips the lattice boundary");
      }
      cell[axis] = static_cast<std::size_t>(base);
      fraction[axis] = coordinates[axis] - base;
    }
    for (std::size_t corner = 0; corner < 8; ++corner) {
      double weight = point.volume_m3 / node_volume;
      auto node = cell;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const bool upper = (corner & (std::size_t{1} << axis)) != 0;
        node[axis] += upper;
        weight *= upper ? fraction[axis] : 1 - fraction[axis];
      }
      lattice.values[index(node[0], node[1], node[2])] += weight;
    }
  }
  std::vector<double> kernel(2 * radius + 1);
  double normalization = 0;
  for (std::size_t i = 0; i < kernel.size(); ++i) {
    const double offset = (static_cast<double>(i) - radius) / sigma_nodes;
    kernel[i] = std::exp(-0.5 * offset * offset);
    normalization += kernel[i];
  }
  for (auto& weight : kernel) {
    weight /= normalization;
  }
  std::vector<double> filtered(count);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    for (std::size_t x = 0; x < extent[0]; ++x) {
      for (std::size_t y = 0; y < extent[1]; ++y) {
        for (std::size_t z = 0; z < extent[2]; ++z) {
          std::array node{x, y, z};
          double sum = 0;
          for (std::size_t k = 0; k < kernel.size(); ++k) {
            const auto coordinate = static_cast<std::int64_t>(node[axis]) + static_cast<std::int64_t>(k)
                - static_cast<std::int64_t>(radius);
            if (coordinate < 0 || static_cast<std::uint64_t>(coordinate) >= extent[axis]) {
              continue;
            }
            auto donor = node;
            donor[axis] = static_cast<std::size_t>(coordinate);
            sum += kernel[k] * lattice.values[index(donor[0], donor[1], donor[2])];
          }
          filtered[index(x, y, z)] = sum;
        }
      }
    }
    lattice.values.swap(filtered);
  }
  for (auto& value : lattice.values) {
    value = 0.5 - value;
    if (!std::isfinite(value)) {
      return absl::OutOfRangeError("density field overflowed");
    }
  }
  return lattice;
}

absl::StatusOr<SurfaceFieldSample3D> SampleSurfaceField3D(const SurfaceLattice3D& lattice, Vec3 position) {
  if (!ValidLattice(lattice) || !Finite(position)) {
    return absl::InvalidArgumentError("invalid surface field query");
  }
  const std::array coordinates{
      (position.x - lattice.origin.x) / lattice.spacing,
      (position.y - lattice.origin.y) / lattice.spacing,
      (position.z - lattice.origin.z) / lattice.spacing,
  };
  std::array<std::size_t, 3> cell;
  std::array<double, 3> fraction;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(coordinates[axis]) || coordinates[axis] < 0 || coordinates[axis] >= lattice.extent[axis] - 1) {
      return absl::OutOfRangeError("surface field query has no support");
    }
    const double base = std::floor(coordinates[axis]);
    cell[axis] = static_cast<std::size_t>(base);
    fraction[axis] = coordinates[axis] - base;
  }
  SurfaceFieldSample3D sample{.value = 0, .gradient = {}};
  std::array<double, 3> gradient{};
  for (std::size_t corner = 0; corner < 8; ++corner) {
    auto node = cell;
    std::array<double, 3> weights;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const bool upper = (corner & (std::size_t{1} << axis)) != 0;
      node[axis] += upper;
      weights[axis] = upper ? fraction[axis] : 1 - fraction[axis];
    }
    const double value = lattice.values[(node[0] * lattice.extent[1] + node[1]) * lattice.extent[2] + node[2]];
    if (!std::isfinite(value)) {
      return absl::OutOfRangeError("surface field donor is nonfinite");
    }
    sample.value += value * weights[0] * weights[1] * weights[2];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double derivative = (corner & (std::size_t{1} << axis)) != 0 ? 1 : -1;
      gradient[axis] += value * derivative * weights[(axis + 1) % 3] * weights[(axis + 2) % 3] / lattice.spacing;
    }
  }
  sample.gradient = {gradient[0], gradient[1], gradient[2]};
  if (!std::isfinite(sample.value) || !Finite(sample.gradient)) {
    return absl::OutOfRangeError("surface field query overflowed");
  }
  return sample;
}
}  // namespace fluidity
