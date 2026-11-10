#pragma once
#include "fluidity/base/liquid_boundary_math.h"
#include "fluidity/base/liquid_boundary_transfer_types_3d.h"
namespace fluidity::cuda_internal {
enum class LiquidSampleFieldKind3D {
  kAnalyticSphere,
  kParticleUnion,
  kHorizontalPlane,
  kLatticeHorizontalPlane
};

// Device-only POD execution descriptor; no variant or host container crosses ABI.
struct LiquidSampleQuery3D {
  Vec3 sphere_center{};
  double radius = 0;
  double world_surface_height_y = 0;
  LiquidSampleFieldKind3D field_kind = LiquidSampleFieldKind3D::kAnalyticSphere;
  std::size_t particle_count = 0;
  std::size_t sample_count = 0;
  boundary_internal::Obstacle obstacle{};
  UniformMacGeometry3D geometry{};
  std::int64_t requested_side = 0;
};
struct LiquidSampleFaceResult3D {
  LiquidBoundaryFaceTag3D tag;
  LiquidPrescribedWallSample3D wall;
  boundary_internal::Error error = boundary_internal::Error::kNone;
};
// All pointers refer to checked persistent ranges and remain live through fence.
// Particle centers use Vec3 records packed by the host, never host-computed SDFs.
// Tank face coordinates are in cell widths so the device preserves lattice
// residuals. Lattice plane sampling uses the y coordinate in cell widths.
// Other faces and field sample positions are in world meters.
int LaunchLiquidFieldSamples3D(const LiquidSampleQuery3D*, const Vec3*, const Vec3*, double*, int*, void*);
int LaunchLiquidFaceSamples3D(const LiquidSampleQuery3D*, const Vec3*, LiquidSampleFaceResult3D*, int*, void*);
}  // namespace fluidity::cuda_internal
