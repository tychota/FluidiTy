#include "fluidity/host/surface_tracking_3d.h"

#include <algorithm>
#include <cmath>
#include <map>
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
  const auto project_position =
      [&](Vec3 projected, Vec3 advected_position, std::uint64_t component, double residual_tolerance_m, Vec3& normal
      ) -> absl::StatusOr<Vec3> {
    auto sample = surface(projected, component);
    if (!sample.ok()) {
      return sample.status();
    }
    bool converged = false;
    for (std::size_t iteration = 0; iteration <= options.maximum_projection_iterations; ++iteration) {
      const auto status = ValidateProjection(*sample, component);
      if (!status.ok()) {
        return status;
      }
      const double gradient_length = std::hypot(sample->gradient.x, sample->gradient.y, sample->gradient.z);
      if (!std::isfinite(gradient_length) || gradient_length == 0) {
        return absl::FailedPreconditionError("surface projection has no regular gradient");
      }
      const double residual = std::abs(sample->value) / gradient_length;
      if (residual <= residual_tolerance_m) {
        normal = {
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
        if (Finite(candidate) && Distance(candidate, advected_position) <= bound) {
          auto next = surface(candidate, component);
          if (!next.ok()) {
            return next.status();
          }
          const auto candidate_status = ValidateProjection(*next, component);
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
    return projected;
  };
  std::vector<Vec3> advected_positions;
  advected_positions.reserve(mesh.vertices.size());
  for (std::size_t vertex_index = 0; vertex_index < result.mesh.vertices.size(); ++vertex_index) {
    auto& vertex = result.mesh.vertices[vertex_index];
    const Vec3 original = vertex.position;
    auto advected =
        AdvectMotionPoint3D(interval, original, first, last, width, options.maximum_advection_substeps, carrier);
    if (!advected.ok()) {
      return advected.status();
    }
    result.advection_substeps += advected->accepted_substeps;
    advected_positions.push_back(advected->world_position);
    const auto projected_position = project_position(
        advected->world_position,
        advected->world_position,
        vertex.component,
        tolerances[vertex_index],
        vertex.normal
    );
    if (!projected_position.ok()) {
      return projected_position.status();
    }
    const Vec3 projected = *projected_position;
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
  std::vector<std::array<std::size_t, 2>> crossings;
  auto measured = MeasureSurfaceMesh3D(result.mesh, &crossings);
  if (!measured.ok()) {
    return measured.status();
  }
  // A small tangential redistribution can unfold slivers when independent
  // nearest points are too close. Only touch vertices of intersecting faces.
  // Each proposal must still project onto the same component within the
  // original displacement/residual limits and strictly reduce intersections.
  if (!crossings.empty() && crossings.size() == measured->self_intersections) {
    std::vector<std::vector<std::uint32_t>> neighbors(mesh.vertices.size());
    for (const auto& triangle : mesh.triangles) {
      for (std::size_t corner = 0; corner < 3; ++corner) {
        const auto a = triangle[corner];
        const auto b = triangle[(corner + 1) % 3];
        neighbors[a].push_back(b);
        neighbors[b].push_back(a);
      }
    }
    for (auto& ring : neighbors) {
      std::ranges::sort(ring);
      const auto duplicates = std::ranges::unique(ring);
      ring.erase(duplicates.begin(), duplicates.end());
    }
    constexpr std::size_t kRepairSweeps = 8;
    constexpr std::size_t kRepairBacktracks = 5;
    for (std::size_t sweep = 0; sweep < kRepairSweeps && !crossings.empty(); ++sweep) {
      std::set<std::uint32_t> affected;
      for (const auto& pair : crossings) {
        for (const auto triangle : pair) {
          affected.insert(mesh.triangles[triangle].begin(), mesh.triangles[triangle].end());
        }
      }
      bool improved = false;
      double fraction = 0.5;
      for (std::size_t backtrack = 0; backtrack < kRepairBacktracks; ++backtrack, fraction *= 0.5) {
        auto candidate = result.mesh;
        bool valid = true;
        for (const auto index : affected) {
          const auto& vertex = result.mesh.vertices[index];
          if (vertex.unresolved_sheet) {
            continue;
          }
          const auto& ring = neighbors[index];
          Vec3 mean_offset{};
          for (const auto neighbor : ring) {
            const auto position = result.mesh.vertices[neighbor].position;
            mean_offset.x += (position.x - vertex.position.x) / ring.size();
            mean_offset.y += (position.y - vertex.position.y) / ring.size();
            mean_offset.z += (position.z - vertex.position.z) / ring.size();
          }
          const double normal_offset =
              mean_offset.x * vertex.normal.x + mean_offset.y * vertex.normal.y + mean_offset.z * vertex.normal.z;
          const Vec3 proposal{
              vertex.position.x + fraction * (mean_offset.x - normal_offset * vertex.normal.x),
              vertex.position.y + fraction * (mean_offset.y - normal_offset * vertex.normal.y),
              vertex.position.z + fraction * (mean_offset.z - normal_offset * vertex.normal.z),
          };
          if (Distance(proposal, advected_positions[index]) > bound) {
            valid = false;
            break;
          }
          auto& replacement = candidate.vertices[index];
          const auto projected = project_position(
              proposal,
              advected_positions[index],
              vertex.component,
              tolerances[index],
              replacement.normal
          );
          if (!projected.ok()) {
            valid = false;
            break;
          }
          if (vertex.contact) {
            // The correction contact band is wider than actual wall contact.
            // Permit motion in the band, while keeping points on the wall.
            const auto before = surface(vertex.position, vertex.component);
            const auto after = surface(*projected, vertex.component);
            const double wall_tolerance_m = 1e-8 * width;
            if (!before.ok() || !after.ok()
                || (before->obstacle_distance_m <= wall_tolerance_m && after->obstacle_distance_m > wall_tolerance_m)) {
              valid = false;
              break;
            }
          }
          replacement.position = *projected;
        }
        if (!valid) {
          continue;
        }
        std::vector<std::array<std::size_t, 2>> candidate_crossings;
        auto candidate_metrics = MeasureSurfaceMesh3D(candidate, &candidate_crossings);
        if (!candidate_metrics.ok()) {
          return candidate_metrics.status();
        }
        if (candidate_metrics->self_intersections >= measured->self_intersections || candidate_metrics->boundary_edges
            || candidate_metrics->nonmanifold_edges || candidate_metrics->nonmanifold_vertices
            || candidate_metrics->orientation_defects || candidate_metrics->degenerate_triangles) {
          continue;
        }
        result.mesh = std::move(candidate);
        measured = std::move(candidate_metrics);
        crossings = std::move(candidate_crossings);
        improved = true;
        break;
      }
      if (!improved) {
        break;
      }
    }
    // If redistributing vertices stalls, try changing the diagonal of an
    // affected two-triangle patch without moving vertices or its boundary.
    struct DirectedEdge {
      std::size_t triangle;
      std::uint32_t from;
      std::uint32_t to;
      std::uint32_t opposite;
    };
    constexpr std::size_t kMaximumFlipTrials = 64;
    std::size_t flip_trials = 0;
    while (!crossings.empty() && flip_trials < kMaximumFlipTrials) {
      std::set<std::size_t> affected_faces;
      for (const auto& pair : crossings) {
        affected_faces.insert(pair.begin(), pair.end());
      }
      std::map<std::array<std::uint32_t, 2>, DirectedEdge> first_owner;
      std::vector<std::array<DirectedEdge, 2>> candidates;
      for (std::size_t index = 0; index < result.mesh.triangles.size(); ++index) {
        const auto triangle = result.mesh.triangles[index];
        for (std::size_t corner = 0; corner < 3; ++corner) {
          const DirectedEdge edge{
              .triangle = index,
              .from = triangle[corner],
              .to = triangle[(corner + 1) % 3],
              .opposite = triangle[(corner + 2) % 3]
          };
          const std::array key{std::min(edge.from, edge.to), std::max(edge.from, edge.to)};
          const auto [prior, inserted] = first_owner.emplace(key, edge);
          if (!inserted && (affected_faces.contains(index) || affected_faces.contains(prior->second.triangle))) {
            candidates.push_back({prior->second, edge});
          }
        }
      }
      bool improved = false;
      for (const auto& pair : candidates) {
        if (flip_trials == kMaximumFlipTrials) {
          break;
        }
        const auto [a, b] = pair;
        if (a.from != b.to || a.to != b.from || a.opposite == b.opposite) {
          continue;
        }
        const std::array new_edge{std::min(a.opposite, b.opposite), std::max(a.opposite, b.opposite)};
        if (first_owner.contains(new_edge)) {
          continue;
        }
        const std::array quad{a.from, a.to, a.opposite, b.opposite};
        if (std::ranges::any_of(quad, [&](const auto vertex) {
              return result.mesh.vertices[vertex].unresolved_sheet;
            })) {
          continue;
        }
        ++flip_trials;
        auto candidate = result.mesh;
        candidate.triangles[a.triangle] = {a.opposite, a.from, b.opposite};
        candidate.triangles[b.triangle] = {a.opposite, b.opposite, a.to};
        std::vector<std::array<std::size_t, 2>> candidate_crossings;
        auto candidate_metrics = MeasureSurfaceMesh3D(candidate, &candidate_crossings);
        if (!candidate_metrics.ok()) {
          return candidate_metrics.status();
        }
        if (candidate_metrics->self_intersections >= measured->self_intersections || candidate_metrics->boundary_edges
            || candidate_metrics->nonmanifold_edges || candidate_metrics->nonmanifold_vertices
            || candidate_metrics->orientation_defects || candidate_metrics->degenerate_triangles) {
          continue;
        }
        result.mesh = std::move(candidate);
        measured = std::move(candidate_metrics);
        crossings = std::move(candidate_crossings);
        improved = true;
        break;
      }
      if (!improved) {
        break;
      }
    }
    if (Safe(*measured)) {
      result.maximum_projection_m = 0;
      for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
        auto& vertex = result.mesh.vertices[index];
        const auto original = mesh.vertices[index].position;
        const auto fluid_velocity = SampleMotionCarrier3D(interval, vertex.position, last, carrier);
        if (!fluid_velocity.ok()) {
          return fluid_velocity.status();
        }
        vertex.fluid_velocity = *fluid_velocity;
        vertex.geometric_velocity = {
            (vertex.position.x - original.x) / duration,
            (vertex.position.y - original.y) / duration,
            (vertex.position.z - original.z) / duration,
        };
        if (!Finite(vertex.geometric_velocity)) {
          return absl::OutOfRangeError("surface repair velocity overflowed");
        }
        result.maximum_projection_m =
            std::max(result.maximum_projection_m, Distance(vertex.position, advected_positions[index]));
      }
    }
  }
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
