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

// POD execution descriptor packed by the host and copied to device scratch.
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
// Enqueues one signed distance in meters per query position in input order.
// The launch count matches device_query->sample_count and is positive. The
// input and output ranges cover that count. The union center range covers
// device_query->particle_count world centers. Lattice plane y is in cell
// widths. Other field positions are in world meters. Returns the CUDA launch
// error. Keep all ranges, the architecture slot and stream live through the
// caller's completion fence.
int LaunchLiquidFieldSamples3D(
    const LiquidSampleQuery3D* device_query,
    const Vec3* device_particle_centers,
    const Vec3* device_sample_positions,
    double* device_signed_distances,
    int* device_compiled_architecture,
    std::size_t sample_count,
    void* stream
);
// Enqueues one face result per anchor in input order. Both arrays cover
// sample_count, matching the device query. Tank anchors are in cell widths
// and other anchors are in world meters. A zero count still writes the
// architecture slot. Returns the CUDA launch error. Keep the ranges,
// architecture slot and stream live through the caller's completion fence.
int LaunchLiquidFaceSamples3D(
    const LiquidSampleQuery3D* device_query,
    const Vec3* device_face_positions,
    LiquidSampleFaceResult3D* device_face_samples,
    int* device_compiled_architecture,
    std::size_t sample_count,
    void* stream
);
}  // namespace fluidity::cuda_internal
