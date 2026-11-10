#include "fluidity/host/surface_tracking_3d.h"

#include <cmath>
#include <set>

namespace fluidity {
namespace {
bool Finite(Vec3 p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
double Distance(Vec3 a, Vec3 b) {
  return std::hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}
bool Safe(const SurfaceMetrics3D& metrics) {
  return metrics.boundary_edges == 0 && metrics.nonmanifold_edges == 0 && metrics.nonmanifold_vertices == 0
      && metrics.orientation_defects == 0 && metrics.degenerate_triangles == 0 && metrics.self_intersections == 0;
}
absl::Status ValidateProjection(const SurfaceProjectionSample3D& sample, std::uint64_t component) {
  if (!std::isfinite(sample.value) || !Finite(sample.gradient) || !std::isfinite(sample.obstacle_distance_m)) {
    return absl::OutOfRangeError("surface projection is nonfinite");
  }
  if (sample.component != component || sample.obstacle_distance_m < 0) {
    return absl::FailedPreconditionError("surface projection changes component or enters a solid");
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<SurfaceTrackingResult3D> TrackSurfaceMesh3D(
    const SurfaceMesh3D& mesh,
    const MotionInterval3D& interval,
    double width,
    const SurfaceTrackingOptions3D& options,
    MotionSpatialSampler3D carrier,
    SurfaceProjectionSampler3D surface
) {
  if (!std::isfinite(width) || width <= 0 || !std::isfinite(options.maximum_projection_cells)
      || options.maximum_projection_cells <= 0 || options.maximum_projection_cells > 0.5
      || !std::isfinite(options.projection_tolerance_cells) || options.projection_tolerance_cells <= 0
      || options.projection_tolerance_cells >= options.maximum_projection_cells
      || options.maximum_projection_iterations == 0 || options.maximum_advection_substeps == 0
      || interval.stamps().size() < 2) {
    return absl::InvalidArgumentError("invalid surface tracking controls or interval");
  }
  const auto original_metrics = MeasureSurfaceMesh3D(mesh);
  if (!original_metrics.ok()) {
    return original_metrics.status();
  }
  if (!Safe(*original_metrics)) {
    return absl::FailedPreconditionError("tracking requires a safe closed mesh");
  }
  std::set<std::uint64_t> ids;
  for (const auto& vertex : mesh.vertices) {
    if (vertex.id == 0 || vertex.id >= mesh.next_id || !ids.insert(vertex.id).second
        || !std::isfinite(vertex.age_seconds) || vertex.age_seconds < 0) {
      return absl::InvalidArgumentError("invalid surface vertex identity or age");
    }
  }
  const double first = interval.stamps().front().time_seconds;
  const double last = interval.stamps().back().time_seconds;
  const double duration = last - first;
  const double bound = options.maximum_projection_cells * width;
  const double tolerance = options.projection_tolerance_cells * width;
  SurfaceTrackingResult3D result{.mesh = mesh, .metrics = {}};
  for (auto& vertex : result.mesh.vertices) {
    const Vec3 original = vertex.position;
    auto advected =
        AdvectMotionPoint3D(interval, original, first, last, width, options.maximum_advection_substeps, carrier);
    if (!advected.ok()) {
      return advected.status();
    }
    result.advection_substeps += advected->accepted_substeps;
    Vec3 projected = advected->world_position;
    auto sample = surface(projected, vertex.component);
    if (!sample.ok()) {
      return sample.status();
    }
    bool converged = false;
    for (std::size_t iteration = 0; iteration <= options.maximum_projection_iterations; ++iteration) {
      const auto status = ValidateProjection(*sample, vertex.component);
      if (!status.ok()) {
        return status;
      }
      const double gradient_length = std::hypot(sample->gradient.x, sample->gradient.y, sample->gradient.z);
      if (!std::isfinite(gradient_length) || gradient_length == 0) {
        return absl::FailedPreconditionError("surface projection has no regular gradient");
      }
      const double residual = std::abs(sample->value) / gradient_length;
      if (residual <= tolerance) {
        vertex.normal = {
            sample->gradient.x / gradient_length,
            sample->gradient.y / gradient_length,
            sample->gradient.z / gradient_length
        };
        converged = true;
        break;
      }
      if (iteration == options.maximum_projection_iterations) {
        break;
      }
      bool accepted = false;
      double step = sample->value / gradient_length;
      for (int backtrack = 0; backtrack < 16; ++backtrack) {
        const Vec3 candidate{
            projected.x - step * (sample->gradient.x / gradient_length),
            projected.y - step * (sample->gradient.y / gradient_length),
            projected.z - step * (sample->gradient.z / gradient_length),
        };
        if (Finite(candidate) && Distance(candidate, advected->world_position) <= bound) {
          auto next = surface(candidate, vertex.component);
          if (!next.ok()) {
            return next.status();
          }
          const auto candidate_status = ValidateProjection(*next, vertex.component);
          if (!candidate_status.ok()) {
            return candidate_status;
          }
          if (std::abs(next->value) < std::abs(sample->value)) {
            projected = candidate;
            sample = *next;
            accepted = true;
            break;
          }
        }
        step *= 0.5;
      }
      if (!accepted) {
        break;
      }
    }
    if (!converged) {
      return absl::FailedPreconditionError("surface projection requires local repair");
    }
    auto fluid_velocity = SampleMotionCarrier3D(interval, projected, last, carrier);
    if (!fluid_velocity.ok()) {
      return fluid_velocity.status();
    }
    vertex.position = projected;
    vertex.geometric_velocity = {
        (projected.x - original.x) / duration,
        (projected.y - original.y) / duration,
        (projected.z - original.z) / duration
    };
    vertex.fluid_velocity = *fluid_velocity;
    vertex.age_seconds += duration;
    if (!Finite(vertex.geometric_velocity) || !std::isfinite(vertex.age_seconds)) {
      return absl::OutOfRangeError("surface tracking overflowed");
    }
    result.maximum_projection_m = std::max(result.maximum_projection_m, Distance(projected, advected->world_position));
    ++result.retained_ids;
  }
  auto measured = MeasureSurfaceMesh3D(result.mesh);
  if (!measured.ok()) {
    return measured.status();
  }
  if (!Safe(*measured)) {
    return absl::FailedPreconditionError("tracked geometry requires local repair");
  }
  result.metrics = *measured;
  return result;
}
}  // namespace fluidity
