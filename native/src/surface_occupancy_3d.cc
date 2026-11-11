#include "fluidity/host/surface_occupancy_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "absl/status/status.h"

namespace fluidity {
namespace {

constexpr double kCertifiedMagnitude = 0.5;

bool MatchesWorldLattice(const SurfaceLattice3D& particle_density, const SurfaceOccupancyView3D& occupancy) {
  const Vec3 origin = particle_density.origin;
  return origin.x == occupancy.world_origin_m.x && origin.y == occupancy.world_origin_m.y
      && origin.z == occupancy.world_origin_m.z && particle_density.spacing == occupancy.spacing_m
      && particle_density.extent == occupancy.extent;
}

bool HasDistinctWorldNodes(const SurfaceLattice3D& lattice) {
  const std::array origin{lattice.origin.x, lattice.origin.y, lattice.origin.z};
  for (std::size_t axis = 0; axis < origin.size(); ++axis) {
    double previous = origin[axis];
    for (std::size_t index = 1; index < lattice.extent[axis]; ++index) {
      const double coordinate = origin[axis] + index * lattice.spacing;
      if (!std::isfinite(coordinate) || coordinate <= previous) {
        return false;
      }
      previous = coordinate;
    }
  }
  return true;
}

// Follow the extractor's near-zero air tie. A cube with any crossing needs
// original samples throughout so certified constants cannot locate a surface.
absl::Status CheckSurfaceCoverage(
    const SurfaceLattice3D& field,
    std::span<const SurfaceOccupancyClass3D> classifications,
    double air_tie
) {
  const auto [nx, ny, nz] = field.extent;
  const auto node_index = [&](std::size_t x, std::size_t y, std::size_t z) {
    return (x * ny + y) * nz + z;
  };
  for (std::size_t x = 0; x < nx; ++x) {
    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t z = 0; z < nz; ++z) {
        const bool last_on_axis = x + 1 == nx || y + 1 == ny || z + 1 == nz;
        const bool boundary = x == 0 || y == 0 || z == 0 || last_on_axis;
        if (boundary && field.values[node_index(x, y, z)] <= 0) {
          return absl::FailedPreconditionError("qualified surface reaches the lattice boundary");
        }
        if (last_on_axis) {
          continue;
        }
        std::size_t liquid_corners = 0;
        std::size_t supported_corners = 0;
        for (std::size_t corner = 0; corner < 8; ++corner) {
          const std::size_t index = node_index(x + (corner & 1), y + ((corner >> 1) & 1), z + (corner >> 2));
          liquid_corners += field.values[index] < -air_tie;
          supported_corners += classifications[index] == SurfaceOccupancyClass3D::kSupportedInterface;
        }
        const bool crossing = liquid_corners > 0 && liquid_corners < 8;
        if (crossing && supported_corners != 8) {
          return absl::FailedPreconditionError("surface crossing requires eight supported particle samples");
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<SurfaceLattice3D> ApplySurfaceOccupancy3D(
    const SurfaceLattice3D& particle_density,
    const SurfaceOccupancyView3D& occupancy,
    std::size_t maximum_nodes
) {
  const auto [nx, ny, nz] = particle_density.extent;
  const double spacing = particle_density.spacing;
  const Vec3 origin = particle_density.origin;
  if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z) || !std::isfinite(spacing)
      || spacing <= 0 || std::min({nx, ny, nz}) < 2) {
    return absl::InvalidArgumentError("invalid particle surface lattice geometry");
  }
  if (!MatchesWorldLattice(particle_density, occupancy)) {
    return absl::InvalidArgumentError("occupancy must use the same world lattice as particle density");
  }
  const std::size_t node_count = particle_density.values.size();
  if (nx > node_count / ny / nz || nx * ny * nz != node_count) {
    return absl::InvalidArgumentError("particle surface lattice extents do not match the values");
  }
  if (node_count > maximum_nodes) {
    return absl::ResourceExhaustedError("surface occupancy node budget exceeded");
  }
  if (occupancy.classifications.size() != node_count) {
    return absl::FailedPreconditionError("occupancy must cover every surface lattice node");
  }
  const double air_tie = spacing * 1e-8;
  if (air_tie >= kCertifiedMagnitude || !HasDistinctWorldNodes(particle_density)) {
    return absl::OutOfRangeError("surface lattice cannot represent distinct world nodes or certified signs");
  }

  SurfaceLattice3D result = particle_density;
  for (std::size_t index = 0; index < node_count; ++index) {
    const double value = particle_density.values[index];
    if (!std::isfinite(value)) {
      return absl::InvalidArgumentError("particle surface lattice has a nonfinite sample");
    }
    switch (occupancy.classifications[index]) {
      case SurfaceOccupancyClass3D::kSupportedInterface:
        if (std::abs(value) > std::numeric_limits<double>::max() / 2) {
          return absl::OutOfRangeError("particle sample magnitude exceeds the interpolation range");
        }
        break;
      case SurfaceOccupancyClass3D::kCertifiedLiquid:
        result.values[index] = -kCertifiedMagnitude;
        break;
      case SurfaceOccupancyClass3D::kCertifiedAir:
        result.values[index] = kCertifiedMagnitude;
        break;
      case SurfaceOccupancyClass3D::kUnknown:
      case SurfaceOccupancyClass3D::kUnsupportedInterface:
        return absl::FailedPreconditionError("surface occupancy coverage is unknown or unsupported");
      default:
        return absl::InvalidArgumentError("invalid surface occupancy classification");
    }
  }
  const absl::Status coverage = CheckSurfaceCoverage(result, occupancy.classifications, air_tie);
  if (!coverage.ok()) {
    return coverage;
  }
  return result;
}

}  // namespace fluidity
