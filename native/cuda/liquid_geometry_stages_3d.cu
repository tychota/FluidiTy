#include <cuda_runtime.h>

#include "liquid_geometry_stages_3d_bridge.h"
namespace fluidity::cuda_internal {
namespace {
__global__ void LiquidFieldSamples3DKernel(
    const LiquidSampleQuery3D* query,
    const Vec3* centers,
    const Vec3* positions,
    double* distances,
    int* architecture
) {
  if (threadIdx.x != 0) {
    return;
  }
#if defined(__CUDA_ARCH__)
  *architecture = __CUDA_ARCH__;
#endif
  // Preserve input particle order and strict-less tie behavior of std::min.
  for (std::size_t sample = 0; sample < query->sample_count; ++sample) {
    const Vec3 position = positions[sample];
    if (query->field_kind == LiquidSampleFieldKind3D::kHorizontalPlane) {
      distances[sample] = position.y - query->world_surface_height_y;
      continue;
    }
    double distance = boundary_internal::numeric::numeric_limits<double>::infinity();
    const bool particle_union = query->field_kind == LiquidSampleFieldKind3D::kParticleUnion;
    const std::size_t count = particle_union ? query->particle_count : 1;
    for (std::size_t particle = 0; particle < count; ++particle) {
      const Vec3 center = particle_union ? centers[particle] : query->sphere_center;
      const double candidate =
          hypot(hypot(position.x - center.x, position.y - center.y), position.z - center.z) - query->radius;
      if (candidate < distance) {
        distance = candidate;
      }
    }
    distances[sample] = distance;
  }
}
__global__ void LiquidFaceSamples3DKernel(
    const LiquidSampleQuery3D* query,
    const Vec3* positions,
    LiquidSampleFaceResult3D* results,
    int* architecture
) {
#if defined(__CUDA_ARCH__)
  *architecture = __CUDA_ARCH__;
#endif
  const auto obstacle = query->obstacle;
  for (std::size_t sample = 0; sample < query->sample_count; ++sample) {
    LiquidSampleFaceResult3D result{};
    const Vec3 position = positions[sample];
    if (obstacle.kind == boundary_internal::ObstacleKind::kOpenTank) {
      const auto wall = boundary_internal::SampleTankFace(obstacle.tank, query->geometry, position);
      if (!wall.ok()) {
        result.error = wall.status();
      } else {
        result.tag = {0, wall->inward_distance > 0};
        if (fabs(wall->inward_distance) <= 2 * query->geometry.cell_width) {
          result.wall = {true, 0, wall->inward_normal, wall->wall_velocity};
        }
      }
    } else if (obstacle.kind != boundary_internal::ObstacleKind::kAbsent) {
      const double coordinate =
          obstacle.normal.x * position.x + obstacle.normal.y * position.y + obstacle.normal.z * position.z;
      const double offset = coordinate - obstacle.coordinate;
      const double distance =
          obstacle.kind == boundary_internal::ObstacleKind::kSlab ? fabs(offset) - obstacle.half_width : offset;
      if (!isfinite(coordinate) || !isfinite(distance)) {
        result.error = boundary_internal::Error::kRange;
        results[sample] = result;
        continue;
      }
      result.tag = {
          obstacle.kind == boundary_internal::ObstacleKind::kSlab ? (offset >= 0 ? 1 : -1) : 0,
          distance >= 0
      };
      if (fabs(distance) <= 2 * query->geometry.cell_width) {
        const double sign =
            obstacle.kind == boundary_internal::ObstacleKind::kSlab ? static_cast<double>(query->requested_side) : 1;
        result.wall = {
            true,
            query->requested_side,
            {sign * obstacle.normal.x, sign * obstacle.normal.y, sign * obstacle.normal.z},
            obstacle.velocity,
        };
      }
    }
    results[sample] = result;
  }
}
}  // namespace
int LaunchLiquidFieldSamples3D(
    const LiquidSampleQuery3D* query,
    const Vec3* centers,
    const Vec3* positions,
    double* distances,
    int* architecture,
    void* stream
) {
  LiquidFieldSamples3DKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      query,
      centers,
      positions,
      distances,
      architecture
  );
  return static_cast<int>(cudaGetLastError());
}
int LaunchLiquidFaceSamples3D(
    const LiquidSampleQuery3D* query,
    const Vec3* positions,
    LiquidSampleFaceResult3D* results,
    int* architecture,
    void* stream
) {
  LiquidFaceSamples3DKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(query, positions, results, architecture);
  return static_cast<int>(cudaGetLastError());
}
}  // namespace fluidity::cuda_internal
