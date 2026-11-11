#pragma once

#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/surface_component_lineage_3d.h"

namespace fluidity {
struct SurfaceMaterialPartition3D {
  std::vector<std::size_t> particle_indices{};  // Indices into input, in ascending particle ID order.
  std::vector<SurfaceComponentMember3D> members{};
  double target_volume_m3 = 0;  // Sum of V0*J in the same order, with each particle counted once.
};

// Partitions the interiors of Simulate's equal-radius particle sphere union.
// Positive overlap connects spheres transitively; point tangencies do not.
// Components and their members are ordered by minimum/member particle ID,
// independent of sparse-grid particle storage order. The sphere radius is the
// authoritative field radius in world metres, not the render Gaussian scale.
// Only the uniform side-zero field is supported. Does not classify obstacles,
// cavities or topology of the smoothed render field. Empty input yields no
// components. Invalid positions, IDs or controls return InvalidArgument;
// unsupported sides Unimplemented, unrepresentable diameter/volume OutOfRange,
// excessive particle count ResourceExhausted (hard 1024). Uses linear storage
// and a sorted x sweep with at most n*(n-1)/2 pair tests. Inputs are unchanged.
absl::StatusOr<std::vector<SurfaceMaterialPartition3D>> PartitionSurfaceSphereUnion3D(
    std::span<const LiquidParticleState3D> particles,
    double sphere_radius_m,
    std::size_t maximum_particles = 1024
);

enum class SurfaceComponentRepresentation3D : std::uint8_t {
  kMesh,
  kDroplet,
};
// Chooses exactly one representation using a caller-verified physical diameter
// in world metres. Mesh becomes droplet below 2h; droplet becomes mesh above
// 3h. At either threshold retain the prior representation. Does not determine
// diameter, transfer volume, allocate points or change component identity.
// Invalid representation, nonpositive/nonfinite diameter or cell width return
// InvalidArgument; unrepresentable diameter ratio OutOfRange.
absl::StatusOr<SurfaceComponentRepresentation3D>
ClassifySurfaceComponent3D(SurfaceComponentRepresentation3D previous, double diameter_m, double cell_width_m);
}  // namespace fluidity
