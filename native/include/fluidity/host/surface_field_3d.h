#pragma once

#include <span>

#include "fluidity/host/surface_mesh_3d.h"

namespace fluidity {
struct SurfaceMaterialPoint3D {
  Vec3 position;  // World metres.
  double volume_m3;  // Positive current material volume, including J.
};
struct SurfaceFieldOptions3D {
  Vec3 origin;  // World position of node (0,0,0), metres.
  double spacing_m;
  std::array<std::size_t, 3> extent;
  double sigma_m;  // Gaussian standard deviation in metres.
};
struct SurfaceFieldSample3D {
  double value;  // Dimensionless 0.5 minus reconstructed volume fraction.
  Vec3 gradient;  // World-space derivative, inverse metres.
};

// Conservatively deposits material volume to eight nodes, then applies a
// normalized separable Gaussian truncated at round(4*sigma/spacing). This is
// the geometric density reference; it does not infer simulation occupancy.
// Requires the complete splat and filter support plus a positive shell inside
// the requested finite lattice. Invalid controls/particles return
// InvalidArgument, clipped support FailedPrecondition, excessive lattice/kernel
// size ResourceExhausted, and numerical overflow OutOfRange. Empty input yields
// a known-air field. No input is modified and failures return no partial field.
absl::StatusOr<SurfaceLattice3D> BuildSurfaceDensity3D(
    std::span<const SurfaceMaterialPoint3D> particles,
    const SurfaceFieldOptions3D& options,
    std::size_t maximum_nodes
);

// Trilinearly samples the field and its analytic trilinear gradient at a finite
// world position. Does not extrapolate or replace absent support with zero.
// Invalid lattice/position returns InvalidArgument, missing support or overflow
// OutOfRange. The lattice is borrowed for this call and is never changed.
absl::StatusOr<SurfaceFieldSample3D> SampleSurfaceField3D(const SurfaceLattice3D& lattice, Vec3 world_position);
}  // namespace fluidity
