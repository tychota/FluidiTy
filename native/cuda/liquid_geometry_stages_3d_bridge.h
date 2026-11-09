#pragma once
#include "fluidity/base/liquid_boundary_math.h"
#include "fluidity/base/liquid_boundary_transfer_types_3d.h"
namespace fluidity::cuda_internal {
// Device-only POD execution descriptor; no variant or host container crosses ABI.
struct LiquidSampleQuery3D {
  Vec3 sphere_center{};
  double radius = 0;
  bool particle_union = false;
  std::size_t particle_count = 0;
  std::size_t sample_count = 0;
  boundary_internal::Obstacle obstacle{};
  double cell_width = 0;
  std::int64_t requested_side = 0;
};
struct LiquidSampleFaceResult3D {
  LiquidBoundaryFaceTag3D tag;
  LiquidPrescribedWallSample3D wall;
  boundary_internal::Error error = boundary_internal::Error::kNone;
};
// All pointers refer to checked persistent ranges and remain live through fence.
// Particle centers use Vec3 records packed by the host, never host-computed SDFs.
int LaunchLiquidFieldSamples3D(const LiquidSampleQuery3D*, const Vec3*, const Vec3*, double*, int*, void*);
int LaunchLiquidFaceSamples3D(const LiquidSampleQuery3D*, const Vec3*, LiquidSampleFaceResult3D*, int*, void*);
}  // namespace fluidity::cuda_internal
