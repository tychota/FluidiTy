#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/statusor.h"
#include "fluidity/host/surface_mesh_3d.h"

namespace fluidity {

enum class SurfaceOccupancyClass3D : std::uint8_t {
  kUnknown = 0,
  kUnsupportedInterface,
  kSupportedInterface,
  kCertifiedLiquid,
  kCertifiedAir,
};

struct SurfaceOccupancyView3D {
  Vec3 world_origin_m{};
  double spacing_m = 0;
  std::array<std::size_t, 3> extent{};
  // One classification per world node, stored at (x * ny + y) * nz + z.
  std::span<const SurfaceOccupancyClass3D> classifications{};
};

// Preserves finite particle samples at supported interface nodes. Writes -0.5
// at certified deep-liquid nodes and +0.5 at certified air nodes, including
// enclosed air. Overrides use the density field's dimensionless units.
// The caller maps accepted occupancy and particle support to exactly the same
// world origin, metre spacing, extents and node order as the particle field.
// Each mixed-sign cube requires all eight nodes to have supported particle
// samples, preserving the interpolated zero set there. Requires a positive
// boundary shell and at most maximum_nodes samples. Inputs remain unchanged.
// Returns InvalidArgument for malformed data or mismatched lattices,
// OutOfRange for unrepresentable world geometry, ResourceExhausted for the
// node budget, or FailedPrecondition for unknown or refused coverage, a
// crossing without full interface coverage, or an open boundary.
absl::StatusOr<SurfaceLattice3D> ApplySurfaceOccupancy3D(
    const SurfaceLattice3D& particle_density,
    const SurfaceOccupancyView3D& occupancy,
    std::size_t maximum_nodes
);

}  // namespace fluidity
