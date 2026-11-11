#include "fluidity/host/surface_tracking_3d.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "absl/strings/str_cat.h"

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
  std::vector<double> tolerances(mesh.vertices.size(), tolerance);
  for (const auto triangle : mesh.triangles) {
    const auto a = mesh.vertices[triangle[0]].position;
    const auto b = mesh.vertices[triangle[1]].position;
    const auto c = mesh.vertices[triangle[2]].position;
    const Vec3 first_edge{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3 second_edge{c.x - a.x, c.y - a.y, c.z - a.z};
    const double twice_area = std::hypot(
        first_edge.y * second_edge.z - first_edge.z * second_edge.y,
        first_edge.z * second_edge.x - first_edge.x * second_edge.z,
        first_edge.x * second_edge.y - first_edge.y * second_edge.x
    );
    const double altitude = twice_area / std::max({Distance(a, b), Distance(b, c), Distance(c, a)});
    for (const auto index : triangle) {
      tolerances[index] = std::min(tolerances[index], 0.01 * altitude);
    }
  }
  for (std::size_t vertex_index = 0; vertex_index < result.mesh.vertices.size(); ++vertex_index) {
    auto& vertex = result.mesh.vertices[vertex_index];
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
      if (residual <= tolerances[vertex_index]) {
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
      const Vec3 direction{
          sample->gradient.x / gradient_length,
          sample->gradient.y / gradient_length,
          sample->gradient.z / gradient_length
      };
      for (int backtrack = 0; backtrack < 16; ++backtrack) {
        const Vec3 candidate{
            projected.x - step * direction.x,
            projected.y - step * direction.y,
            projected.z - step * direction.z,
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
    return absl::FailedPreconditionError(absl::StrCat(
        "tracked geometry requires local repair: intersections=",
        measured->self_intersections,
        " boundary_edges=",
        measured->boundary_edges,
        " nonmanifold_edges=",
        measured->nonmanifold_edges,
        " nonmanifold_vertices=",
        measured->nonmanifold_vertices,
        " winding=",
        measured->orientation_defects,
        " degenerate_triangles=",
        measured->degenerate_triangles
    ));
  }
  result.metrics = *measured;
  return result;
}
}  // namespace fluidity
