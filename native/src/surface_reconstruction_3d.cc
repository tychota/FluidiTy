#include "fluidity/host/surface_reconstruction_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "fluidity/host/motion_carrier_3d.h"
#include "fluidity/host/surface_closest_point_3d.h"
#include "fluidity/host/surface_field_3d.h"
#include "fluidity/host/surface_material_components_3d.h"
#include "liquid_substep_3d_internal.h"

namespace fluidity {
namespace {
bool Safe(const SurfaceMetrics3D& m) {
  return m.boundary_edges == 0 && m.nonmanifold_edges == 0 && m.nonmanifold_vertices == 0 && m.orientation_defects == 0
      && m.degenerate_triangles == 0 && m.self_intersections == 0;
}
bool SameGeometry(UniformMacGeometry3D a, UniformMacGeometry3D b) {
  return a.cell_width == b.cell_width && a.world_origin.x == b.world_origin.x && a.world_origin.y == b.world_origin.y
      && a.world_origin.z == b.world_origin.z;
}

// Render smoothing can bridge a narrow gap, and spatial coincidence is not
// material ancestry. Check the authoritative union before constructing a
// density field. Until local repair is supported, accept only one continuing
// material component, with no replacement by an unrelated component.
absl::Status CheckSphereUnionContinuity(
    const MotionFrame3D& source,
    double sphere_radius_m,
    const SurfaceFrame3D* previous,
    std::size_t maximum_particles
) {
  const auto current_material =
      PartitionSurfaceSphereUnion3D(source.samples.back().particles.values(), sphere_radius_m, maximum_particles);
  if (!current_material.ok()) {
    return current_material.status();
  }
  if (current_material->size() != 1) {
    return absl::FailedPreconditionError("disconnected material components require separate Mesh geometry");
  }
  if (previous == nullptr) {
    return absl::OkStatus();
  }
  if (previous->volumes.size() != 1) {
    return absl::FailedPreconditionError("material component birth or change requires Mesh lineage");
  }
  const auto prior_material =
      PartitionSurfaceSphereUnion3D(source.samples.front().particles.values(), sphere_radius_m, maximum_particles);
  if (!prior_material.ok()) {
    return prior_material.status();
  }
  if (prior_material->size() != 1) {
    return absl::FailedPreconditionError("prior material components do not match the Mesh checkpoint");
  }
  const auto prior_component_id = previous->volumes.front().component;
  if (prior_component_id == std::numeric_limits<std::uint64_t>::max()) {
    return absl::ResourceExhaustedError("Mesh component identity counter exhausted");
  }
  const std::array history{SurfaceComponentHistory3D{
      .component_id = prior_component_id,
      .members = prior_material->front().members,
  }};
  const std::array material{SurfaceComponentMaterial3D{
      .members = current_material->front().members,
      .target_volume_m3 = current_material->front().target_volume_m3,
  }};
  const auto lineage = MatchSurfaceComponentLineage3D(history, material, prior_component_id + 1, maximum_particles);
  if (!lineage.ok()) {
    return lineage.status();
  }
  if (lineage->components.front().component_id != prior_component_id) {
    return absl::FailedPreconditionError("new material component requires Mesh lineage and fresh vertex IDs");
  }
  return absl::OkStatus();
}
// Interpolates centred nodal derivatives, giving a continuous offset direction
// across voxel boundaries rather than an unrelated one-sided cell derivative.
absl::StatusOr<Vec3> FieldNormal(const SurfaceLattice3D& field, Vec3 position) {
  const std::array coordinates{
      (position.x - field.origin.x) / field.spacing,
      (position.y - field.origin.y) / field.spacing,
      (position.z - field.origin.z) / field.spacing
  };
  std::array<std::size_t, 3> cell;
  std::array<double, 3> fraction;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(coordinates[axis]) || coordinates[axis] < 1 || coordinates[axis] >= field.extent[axis] - 2) {
      return absl::FailedPreconditionError("surface normal has no complete gradient halo");
    }
    const double base = std::floor(coordinates[axis]);
    cell[axis] = static_cast<std::size_t>(base);
    fraction[axis] = coordinates[axis] - base;
  }
  const auto value = [&](const auto& node) {
    return field.values[(node[0] * field.extent[1] + node[1]) * field.extent[2] + node[2]];
  };
  std::array<double, 3> gradient{};
  for (std::size_t corner = 0; corner < 8; ++corner) {
    auto node = cell;
    double weight = 1;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const bool upper = (corner & (std::size_t{1} << axis)) != 0;
      node[axis] += upper;
      weight *= upper ? fraction[axis] : 1 - fraction[axis];
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      auto lower = node;
      auto upper = node;
      --lower[axis];
      ++upper[axis];
      gradient[axis] += weight * (value(upper) - value(lower)) / (2 * field.spacing);
    }
  }
  const double length = std::hypot(gradient[0], gradient[1], gradient[2]);
  if (!std::isfinite(length) || length == 0) {
    return absl::FailedPreconditionError("surface field has no regular normal");
  }
  return Vec3{gradient[0] / length, gradient[1] / length, gradient[2] / length};
}
}  // namespace

absl::StatusOr<SurfaceReconstructionResult3D> ReconstructSurfaceFrame3D(
    const MotionFrame3D& source,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& step_options,
    const SurfaceFrame3D* previous,
    const SurfaceReconstructionOptions3D& options
) {
  if (source.samples.size() > 65) {
    return absl::ResourceExhaustedError("surface source sample budget exceeded");
  }
  if (source.samples.empty() || !std::isfinite(options.spacing_m) || options.spacing_m <= 0
      || !std::isfinite(options.sigma_m) || options.sigma_m <= 0 || options.maximum_correction_iterations == 0) {
    return absl::InvalidArgumentError("invalid surface source or reconstruction settings");
  }
  if (!std::holds_alternative<std::monostate>(scene.obstacle)) {
    return absl::UnimplementedError("surface contact adapter has not qualified");
  }
  const auto& endpoint = source.samples.back();
  std::vector<MotionStamp3D> stamps;
  for (const auto& sample : source.samples) {
    const auto limits = options.limits.cache.maximum_records;
    constexpr std::size_t hard_limit = std::size_t{1} << 20;
    if (sample.particles.size() > std::min(limits.particles, hard_limit)
        || sample.blocks.size() > std::min(limits.blocks, hard_limit)
        || sample.cells.size() > std::min(limits.cells, hard_limit)
        || sample.faces.size() > std::min(limits.faces, hard_limit)
        || sample.sources.size() > std::min(limits.sources, hard_limit)) {
      return absl::ResourceExhaustedError("surface source record budget exceeded");
    }
    const auto checked = ValidateLiquidState3D(sample);
    if (!checked.ok()) {
      return checked;
    }
    if (!SameGeometry(sample.geometry, endpoint.geometry) || sample.random_seed != endpoint.random_seed) {
      return absl::InvalidArgumentError("inconsistent surface source geometry or seed");
    }
    if (!stamps.empty()) {
      const auto previous_clock = source.samples[stamps.size() - 1].clock;
      const bool intermediate = stamps.size() + 1 < source.samples.size();
      if (sample.clock.accepted_step_ordinal <= previous_clock.accepted_step_ordinal
          || (intermediate
              && (sample.clock.frame_ordinal != source.first_clock.frame_ordinal || sample.clock.substep_ordinal == 0)
          )) {
        return absl::InvalidArgumentError("surface source intermediate clocks are inconsistent");
      }
    }
    stamps.push_back({sample.clock.time, sample.clock.epoch});
  }
  if (source.samples.front().clock != source.first_clock || endpoint.clock != source.last_clock) {
    return absl::InvalidArgumentError("surface source endpoint clocks differ");
  }
  std::optional<MotionInterval3D> interval;
  if (source.last_clock == LiquidClock3D{}) {
    if (source.first_clock != LiquidClock3D{} || source.samples.size() != 1 || previous != nullptr) {
      return absl::InvalidArgumentError("invalid surface seed or predecessor");
    }
  } else {
    if (previous == nullptr || previous->clock != source.first_clock
        || previous->cell_width_m != endpoint.geometry.cell_width) {
      return absl::FailedPreconditionError("surface interval lacks its exact predecessor");
    }
    const auto checked_previous = EncodeSurfaceFrame3D(*previous, options.limits);
    if (!checked_previous.ok()) {
      return checked_previous.status();
    }
    auto created = MotionInterval3D::Create(source.first_clock, source.last_clock, stamps);
    if (!created.ok()) {
      return created.status();
    }
    interval = std::move(*created);
  }
  SurfaceReconstructionResult3D result{
      .frame = {.clock = endpoint.clock, .cell_width_m = endpoint.geometry.cell_width},
      .raw_metrics = {},
      .corrected_metrics = {},
  };
  if (endpoint.particles.size() == 0) {
    if (previous != nullptr && !previous->mesh.vertices.empty()) {
      return absl::FailedPreconditionError("surface disappearance requires component lineage");
    }
    result.frame.mesh.next_id = previous == nullptr ? 1 : previous->mesh.next_id;
    return result;
  }
  if (const auto* sphere_union = std::get_if<LiquidParticleSphereUnion3D>(&scene.field)) {
    const auto continuity = CheckSphereUnionContinuity(
        source,
        sphere_union->radius,
        previous,
        options.limits.cache.maximum_records.particles
    );
    if (!continuity.ok()) {
      return continuity;
    }
  }
  std::optional<liquid_step_internal::PreparedIsolatedDrop3D> isolated;
  if (const auto* drop = std::get_if<LiquidIsolatedCovarianceDrop3D>(&scene.field)) {
    auto prepared = liquid_step_internal::PrepareIsolatedDrop3D(
        endpoint.particles.values(),
        endpoint.geometry.cell_width,
        drop->physical_radius,
        drop->seed_covariance_scale
    );
    if (!prepared.ok()) {
      return prepared.status();
    }
    isolated = std::move(*prepared);
  }
  const auto source_distance = [&](Vec3 position) -> absl::StatusOr<double> {
    if (isolated) {
      return isolated->ParticleSignedDistance(position);
    }
    const double distance = liquid_step_internal::FieldDistance(scene.field, endpoint.particles.values(), position);
    if (!std::isfinite(distance)) {
      return absl::FailedPreconditionError("surface source occupancy is unavailable");
    }
    return distance;
  };
  const auto first_position = endpoint.particles[0].world_position;
  std::array<double, 3> lower{first_position.x, first_position.y, first_position.z};
  auto upper = lower;
  std::vector<SurfaceMaterialPoint3D> particles;
  double material_volume = 0;
  for (const auto& particle : endpoint.particles.values()) {
    if (particle.side_tag != 0) {
      return absl::UnimplementedError("surface material-side adapter has not qualified");
    }
    const double volume = particle.reference_volume * particle.volume_ratio;
    if (!std::isfinite(volume) || volume <= 0) {
      return absl::OutOfRangeError("surface material contribution is unrepresentable");
    }
    material_volume += volume;
    particles.push_back({.position = particle.world_position, .volume_m3 = volume});
    const std::array position{particle.world_position.x, particle.world_position.y, particle.world_position.z};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      lower[axis] = std::min(lower[axis], position[axis]);
      upper[axis] = std::max(upper[axis], position[axis]);
    }
  }
  if (!std::isfinite(material_volume)) {
    return absl::OutOfRangeError("surface material volume overflowed");
  }
  const double padding = 4 * options.sigma_m + 3 * options.spacing_m;
  std::array<std::size_t, 3> extent;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double minimum_index = std::floor((lower[axis] - padding) / options.spacing_m);
    const double maximum_index = std::ceil((upper[axis] + padding) / options.spacing_m);
    const double nodes = maximum_index - minimum_index + 1;
    if (!std::isfinite(nodes) || nodes < 2 || nodes > options.maximum_nodes || std::abs(minimum_index) > 0x1p52
        || std::abs(maximum_index) > 0x1p52) {
      return absl::ResourceExhaustedError("surface lattice bounds exceed numerical or work limits");
    }
    lower[axis] = minimum_index * options.spacing_m;
    extent[axis] = static_cast<std::size_t>(nodes);
  }
  auto field = BuildSurfaceDensity3D(
      particles,
      {.origin = {lower[0], lower[1], lower[2]},
       .spacing_m = options.spacing_m,
       .extent = extent,
       .sigma_m = options.sigma_m},
      options.maximum_nodes
  );
  if (!field.ok()) {
    return field.status();
  }
  for (std::size_t x = 0; x < extent[0]; ++x) {
    for (std::size_t y = 0; y < extent[1]; ++y) {
      for (std::size_t z = 0; z < extent[2]; ++z) {
        const Vec3 position{
            lower[0] + x * options.spacing_m,
            lower[1] + y * options.spacing_m,
            lower[2] + z * options.spacing_m
        };
        const auto distance = source_distance(position);
        if (!distance.ok()) {
          return distance.status();
        }
        auto& value = field->values[(x * extent[1] + y) * extent[2] + z];
        if (*distance < -2 * endpoint.geometry.cell_width && value >= 0) {
          value = -0.5;
          ++result.deep_liquid_fills;
        } else if (*distance > 2 * endpoint.geometry.cell_width && value <= 0) {
          value = 0.5;
          ++result.deep_air_fills;
        }
      }
    }
  }
  auto raw_mesh = ExtractSurfaceMesh3D(*field, options.limits.maximum_vertices);
  if (!raw_mesh.ok()) {
    return raw_mesh.status();
  }
  auto raw_metrics = MeasureSurfaceMesh3D(*raw_mesh);
  if (!raw_metrics.ok()) {
    return raw_metrics.status();
  }
  if (!Safe(*raw_metrics) || raw_metrics->signed_volume <= 0 || raw_metrics->components != 1) {
    return absl::FailedPreconditionError("surface requires droplet, component or topology repair");
  }
  result.raw_metrics = *raw_metrics;
  std::uint64_t component = 1;
  if (previous != nullptr) {
    if (previous->volumes.size() != 1 || previous->mesh.vertices.empty()) {
      return absl::FailedPreconditionError("surface birth requires component lineage");
    }
    component = previous->volumes.front().component;
  }
  for (auto& vertex : raw_mesh->vertices) {
    vertex.component = component;
  }
  // Track the already volume-qualified scaffold. Reprojecting a corrected
  // surface onto the deficient raw zero set and expanding it every frame
  // introduces artificial drift even in a perfectly frozen source.
  const auto scaffold_metrics = CorrectSurfaceVolumeWithNormals3D(
      *raw_mesh,
      material_volume,
      endpoint.geometry.cell_width,
      0.005,
      options.maximum_correction_iterations
  );
  if (!scaffold_metrics.ok()) {
    return scaffold_metrics.status();
  }
  std::vector<FrozenMotionCarrier3D> carriers;
  for (const auto& sample : source.samples) {
    auto carrier = FrozenMotionCarrier3D::Create(sample, scene, step_options);
    if (!carrier.ok()) {
      return carrier.status();
    }
    carriers.push_back(std::move(*carrier));
  }
  const auto sample_carrier = [&](std::size_t stamp, Vec3 position) -> absl::StatusOr<Vec3> {
    const auto sample = carriers[stamp].Sample(position);
    if (!sample.ok()) {
      return sample.status();
    }
    return sample->velocity;
  };
  if (previous != nullptr) {
    const auto metrics = MeasureSurfaceMesh3D(previous->mesh);
    if (!metrics.ok()) {
      return metrics.status();
    }
    if (metrics->components != 1 || metrics->euler_characteristic != raw_metrics->euler_characteristic) {
      return absl::FailedPreconditionError("surface topology change requires local repair");
    }
    const auto scaffold = SurfaceClosestPoint3D::Create(*raw_mesh, options.limits.maximum_triangles);
    if (!scaffold.ok()) {
      return scaffold.status();
    }
    const auto project = [&](Vec3 position, std::uint64_t expected) -> absl::StatusOr<SurfaceProjectionSample3D> {
      const auto closest = scaffold->Sample(position, expected);
      if (!closest.ok()) {
        return closest.status();
      }
      const auto distance = source_distance(position);
      if (!distance.ok()) {
        return distance.status();
      }
      if (expected != component || *distance > 2 * endpoint.geometry.cell_width) {
        return absl::FailedPreconditionError("projection leaves its known liquid component neighborhood");
      }
      Vec3 gradient = closest->outward_normal;
      if (closest->distance_m > 0) {
        gradient = {
            (position.x - closest->position.x) / closest->distance_m,
            (position.y - closest->position.y) / closest->distance_m,
            (position.z - closest->position.z) / closest->distance_m
        };
      }
      return SurfaceProjectionSample3D{
          .value = closest->distance_m,
          .gradient = gradient,
          .component = component,
          .obstacle_distance_m = std::numeric_limits<double>::max(),
      };
    };
    auto tracked = TrackSurfaceMesh3D(
        previous->mesh,
        *interval,
        endpoint.geometry.cell_width,
        options.tracking,
        sample_carrier,
        project
    );
    if (!tracked.ok()) {
      return tracked.status();
    }
    result.frame.mesh = std::move(tracked->mesh);
    result.used_tracker = true;
    result.retained_ids = tracked->retained_ids;
    for (auto& vertex : result.frame.mesh.vertices) {
      const auto normal = FieldNormal(*field, vertex.position);
      if (!normal.ok()) {
        return normal.status();
      }
      vertex.normal = *normal;
    }
  } else {
    result.frame.mesh = std::move(*raw_mesh);
    for (auto& vertex : result.frame.mesh.vertices) {
      vertex.component = component;
    }
  }
  auto corrected = CorrectSurfaceVolumeWithNormals3D(
      result.frame.mesh,
      material_volume,
      endpoint.geometry.cell_width,
      0.005,
      options.maximum_correction_iterations
  );
  if (!corrected.ok()) {
    return corrected.status();
  }
  result.corrected_metrics = *corrected;
  result.frame.volumes.push_back({component, material_volume, raw_metrics->signed_volume, corrected->signed_volume});
  for (std::size_t index = 0; index < result.frame.mesh.vertices.size(); ++index) {
    auto& vertex = result.frame.mesh.vertices[index];
    const auto velocity = sample_carrier(carriers.size() - 1, vertex.position);
    if (!velocity.ok()) {
      return velocity.status();
    }
    vertex.fluid_velocity = *velocity;
    if (previous != nullptr) {
      const auto original = previous->mesh.vertices[index].position;
      const double duration = source.last_clock.time - source.first_clock.time;
      vertex.geometric_velocity = {
          (vertex.position.x - original.x) / duration,
          (vertex.position.y - original.y) / duration,
          (vertex.position.z - original.z) / duration
      };
    }
    const auto distance = source_distance(vertex.position);
    if (!distance.ok()) {
      return distance.status();
    }
    result.maximum_source_field_distance_m = std::max(result.maximum_source_field_distance_m, std::abs(*distance));
  }
  const auto encoded = EncodeSurfaceFrame3D(result.frame, options.limits);
  if (!encoded.ok()) {
    return encoded.status();
  }
  return result;
}
}  // namespace fluidity
