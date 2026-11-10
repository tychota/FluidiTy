#pragma once

#include "fluidity/host/surface_mesh_3d.h"

namespace fluidity {
struct SurfaceProjectionSample3D {
  double value;  // Zero on the geometric field; negative inside liquid.
  Vec3 gradient;  // World derivative of value, nonzero at a regular surface.
  std::uint64_t component;  // Stable material/sheet identity supplied by the field.
  double obstacle_distance_m;  // Nonnegative outside solids; finite.
};
using SurfaceProjectionSampler3D =
    absl::FunctionRef<absl::StatusOr<SurfaceProjectionSample3D>(Vec3 world_position, std::uint64_t expected_component)>;
struct SurfaceTrackingOptions3D {
  double maximum_projection_cells = 0.25;  // Maximum total displacement from the advected point, in h.
  double projection_tolerance_cells = 1e-5;
  std::size_t maximum_projection_iterations = 12;
  std::size_t maximum_advection_substeps = 4096;
};
struct SurfaceTrackingResult3D {
  SurfaceMesh3D mesh;
  SurfaceMetrics3D metrics;
  std::size_t retained_ids = 0;
  std::size_t advection_substeps = 0;  // Sum across vertices.
  double maximum_projection_m = 0;
};

// Advances a safe, closed mesh across the entire accepted MotionInterval using
// bounded RK2 and same-component Newton projection with decreasing residual.
// Preserves vertex IDs, lineage, topology and next_id. Updates age, geometric
// velocity and the independently sampled endpoint fluid velocity. Queries and
// candidate endpoints must remain outside obstacles. Missing support, changed
// component, failed projection or unsafe final geometry returns
// FailedPrecondition (local repair is required by the caller); query failures
// propagate. Invalid controls return InvalidArgument and numerical overflow
// OutOfRange. Input mesh is never modified; no partial result on failure.
absl::StatusOr<SurfaceTrackingResult3D> TrackSurfaceMesh3D(
    const SurfaceMesh3D& mesh,
    const MotionInterval3D& interval,
    double cell_width_m,
    const SurfaceTrackingOptions3D& options,
    MotionSpatialSampler3D carrier,
    SurfaceProjectionSampler3D surface
);
}  // namespace fluidity
