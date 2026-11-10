#pragma once
#include <type_traits>

#include "fluidity/base/liquid_boundary_geometry_math_3d.h"
namespace fluidity::cuda_internal {
struct BoundaryReport3D {
  boundary_internal::GeometryOutput geometry{};
  boundary_internal::ContactOutput contact{};
  boundary_internal::Error error{};
  int compiled_architecture = 0;
};
static_assert(std::is_trivially_copyable_v<boundary_internal::GeometryInput>);
static_assert(std::is_trivially_copyable_v<boundary_internal::ContactInput>);
static_assert(std::is_trivially_copyable_v<BoundaryReport3D>);
static_assert(std::is_trivially_copyable_v<boundary_internal::GeometryBatchResult>);
// Launch exactly one device query on the caller stream; return the launch error.
// Inputs/output are checked allocations owned through stream synchronization.
int LaunchLiquidBoundaryGeometry3D(const boundary_internal::GeometryInput*, BoundaryReport3D*, void* stream);
// Launch one thread per query and write one result at the matching ordinal.
// The caller owns count queries and results through stream synchronization.
int LaunchLiquidBoundaryGeometryBatch3D(
    const boundary_internal::GeometryInput* queries,
    boundary_internal::GeometryBatchResult* results,
    std::size_t count,
    void* stream
);
int LaunchLiquidWallContact3D(const boundary_internal::ContactInput*, BoundaryReport3D*, void* stream);
}  // namespace fluidity::cuda_internal
