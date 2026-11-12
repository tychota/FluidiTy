#include "fluidity/host/surface_reconstruction_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "absl/strings/str_cat.h"
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
  const auto* beach = std::get_if<TranslatingHalfspace3D>(&scene.obstacle);
  if (!std::holds_alternative<std::monostate>(scene.obstacle)
      && (beach == nullptr || beach->wall_velocity.x != 0 || beach->wall_velocity.y != 0 || beach->wall_velocity.z != 0
      )) {
    return absl::UnimplementedError("surface contact requires a stationary halfspace");
  }
  if (beach
      && (!std::isfinite(beach->start_plane_coordinate)
          || std::abs(std::hypot(beach->unit_normal.x, beach->unit_normal.y, beach->unit_normal.z) - 1) > 1e-12
          || !liquid_step_internal::Finite(beach->unit_normal))) {
    return absl::InvalidArgumentError("surface contact plane is invalid");
  }
  const auto& endpoint = source.samples.back();
  const double cell_width_m = endpoint.geometry.cell_width;
  // Leave a roundoff margin above the collision plane. An affine signed
  // distance interpolates exactly within the extraction tetrahedra.
  const double contact_margin_m = 1e-9 * cell_width_m;
  const auto mark_contact = [&](SurfaceMesh3D& mesh) -> absl::Status {
    if (beach) {
      for (auto& vertex : mesh.vertices) {
        const double clearance_m = liquid_step_internal::SolidDistance(scene.obstacle, vertex.position);
        if (!std::isfinite(clearance_m) || clearance_m < 0) {
          return absl::FailedPreconditionError("surface vertex enters the stationary contact plane");
        }
        vertex.contact = clearance_m <= 0.25 * cell_width_m + contact_margin_m;
      }
    }
    return absl::OkStatus();
  };
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
    if (previous == nullptr) {
      const auto partition = PartitionSurfaceSphereUnion3D(
          endpoint.particles.values(),
          sphere_union->radius,
          options.limits.cache.maximum_records.particles
      );
      if (!partition.ok()) {
        return partition.status();
      }
      if (partition->size() > 1) {
        // Separate the material before density smoothing. The aggregate
        // encoder rejects intersecting components, including across dry gaps.
        constexpr std::size_t kMaximumSeedComponents = 16;
        if (partition->size() > kMaximumSeedComponents || partition->size() > options.maximum_nodes) {
          return absl::ResourceExhaustedError("material component seed exceeds the reconstruction budget");
        }
        auto component_options = options;
        component_options.maximum_nodes /= partition->size();
        using MeshBounds = std::array<Vec3, 2>;
        std::vector<MeshBounds> component_bounds;
        const auto contains_bounds = [](const MeshBounds& outer, const MeshBounds& inner) {
          return outer[0].x <= inner[0].x && outer[0].y <= inner[0].y && outer[0].z <= inner[0].z
              && outer[1].x >= inner[1].x && outer[1].y >= inner[1].y && outer[1].z >= inner[1].z;
        };
        for (const auto& material : *partition) {
          MotionFrame3D component_source{};
          auto& seed = component_source.samples.emplace_back();
          seed.geometry = endpoint.geometry;
          seed.random_seed = endpoint.random_seed;
          seed.next_particle_id = endpoint.next_particle_id;
          auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(material.particle_indices.size());
          if (!particles.ok()) {
            return particles.status();
          }
          const auto sized = particles->Resize(material.particle_indices.size());
          if (!sized.ok()) {
            return sized;
          }
          for (std::size_t index = 0; index < material.particle_indices.size(); ++index) {
            (*particles)[index] = endpoint.particles[material.particle_indices[index]];
          }
          seed.particles = std::move(*particles);
          auto reconstructed =
              ReconstructSurfaceFrame3D(component_source, scene, step_options, nullptr, component_options);
          if (!reconstructed.ok()) {
            return reconstructed.status();
          }
          auto& mesh = result.frame.mesh;
          auto& component_mesh = reconstructed->frame.mesh;
          if (component_mesh.vertices.size() > options.limits.maximum_vertices - mesh.vertices.size()
              || component_mesh.triangles.size() > options.limits.maximum_triangles - mesh.triangles.size()) {
            return absl::ResourceExhaustedError("combined Mesh seed exceeds the geometry budget");
          }
          const auto first_position = component_mesh.vertices.front().position;
          MeshBounds bounds{first_position, first_position};
          for (const auto& vertex : component_mesh.vertices) {
            const auto position = vertex.position;
            bounds[0] = {
                std::min(bounds[0].x, position.x),
                std::min(bounds[0].y, position.y),
                std::min(bounds[0].z, position.z)
            };
            bounds[1] = {
                std::max(bounds[1].x, position.x),
                std::max(bounds[1].y, position.y),
                std::max(bounds[1].z, position.z)
            };
          }
          // A triangle intersection test cannot detect nested closed meshes.
          // Fail closed if their boxes admit nesting.
          if (std::ranges::any_of(component_bounds, [&](const auto& prior) {
                return contains_bounds(prior, bounds) || contains_bounds(bounds, prior);
              })) {
            return absl::FailedPreconditionError("separate Mesh seed surfaces may enclose one another");
          }
          component_bounds.push_back(bounds);
          const auto component_id = endpoint.particles[material.particle_indices.front()].id;
          const auto vertex_offset = static_cast<std::uint32_t>(mesh.vertices.size());
          for (auto& vertex : component_mesh.vertices) {
            vertex.component = component_id;
            vertex.id = mesh.next_id++;
            mesh.vertices.push_back(vertex);
          }
          for (auto triangle : component_mesh.triangles) {
            for (auto& vertex_index : triangle) {
              vertex_index += vertex_offset;
            }
            mesh.triangles.push_back(triangle);
          }
          auto volume = reconstructed->frame.volumes.front();
          volume.component = component_id;
          result.frame.volumes.push_back(volume);
          result.raw_metrics.signed_volume += reconstructed->raw_metrics.signed_volume;
          result.raw_metrics.area += reconstructed->raw_metrics.area;
          result.raw_metrics.components += reconstructed->raw_metrics.components;
          result.raw_metrics.positive_volume_components += reconstructed->raw_metrics.positive_volume_components;
          result.raw_metrics.negative_volume_components += reconstructed->raw_metrics.negative_volume_components;
          result.raw_metrics.euler_characteristic += reconstructed->raw_metrics.euler_characteristic;
          result.deep_liquid_fills += reconstructed->deep_liquid_fills;
          result.deep_air_fills += reconstructed->deep_air_fills;
          result.maximum_source_field_distance_m =
              std::max(result.maximum_source_field_distance_m, reconstructed->maximum_source_field_distance_m);
        }
        auto measured = MeasureSurfaceMesh3D(result.frame.mesh);
        if (!measured.ok()) {
          return measured.status();
        }
        if (!Safe(*measured) || measured->components != partition->size()) {
          return absl::FailedPreconditionError("separate material component surfaces intersect or are unsafe");
        }
        result.corrected_metrics = *measured;
        const auto encoded = EncodeSurfaceFrame3D(result.frame, options.limits);
        if (!encoded.ok()) {
          return encoded.status();
        }
        return result;
      }
    }
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
  const bool reflect_density = beach && std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field);
  const auto add_density_sample = [&](Vec3 position, double volume_m3) {
    particles.push_back({.position = position, .volume_m3 = volume_m3});
    const std::array coordinates{position.x, position.y, position.z};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      lower[axis] = std::min(lower[axis], coordinates[axis]);
      upper[axis] = std::max(upper[axis], coordinates[axis]);
    }
  };
  for (const auto& particle : endpoint.particles.values()) {
    if (particle.side_tag != 0) {
      return absl::UnimplementedError("surface material-side adapter has not qualified");
    }
    const double volume = particle.reference_volume * particle.volume_ratio;
    if (!std::isfinite(volume) || volume <= 0) {
      return absl::OutOfRangeError("surface material contribution is unrepresentable");
    }
    material_volume += volume;
    add_density_sample(particle.world_position, volume);
    if (reflect_density) {
      const double clearance_m = liquid_step_internal::SolidDistance(scene.obstacle, particle.world_position);
      const Vec3 normal = beach->unit_normal;
      // The truncated separable filter and trilinear splat have box support.
      // Reflect only donors whose support can reach the stationary plane.
      const double reach_m = (4 * options.sigma_m + 1.5 * options.spacing_m)
          * (std::abs(normal.x) + std::abs(normal.y) + std::abs(normal.z));
      if (clearance_m >= 0 && clearance_m <= reach_m) {
        const Vec3 reflected =
            liquid_step_internal::Add(particle.world_position, liquid_step_internal::Scale(normal, -2 * clearance_m));
        add_density_sample(reflected, volume);
      }
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
        if (beach) {
          const double clearance_m = liquid_step_internal::SolidDistance(scene.obstacle, position);
          const double obstacle_field = (contact_margin_m - clearance_m) / options.spacing_m;
          if (!std::isfinite(obstacle_field)) {
            return absl::OutOfRangeError("surface contact field overflow");
          }
          value = std::max(value, obstacle_field);
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
  const auto scaffold_contact = mark_contact(*raw_mesh);
  if (!scaffold_contact.ok()) {
    return scaffold_contact;
  }
  const auto scaffold_metrics = CorrectSurfaceVolumeWithNormals3D(
      *raw_mesh,
      material_volume,
      endpoint.geometry.cell_width,
      0.005,
      options.maximum_correction_iterations
  );
  if (!scaffold_metrics.ok()) {
    if (absl::IsFailedPrecondition(scaffold_metrics.status())) {
      return absl::FailedPreconditionError(absl::StrCat(
          scaffold_metrics.status().message(),
          " (raw volume ",
          raw_metrics->signed_volume,
          " m^3, material volume ",
          material_volume,
          " m^3)"
      ));
    }
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
    auto velocity = sample->velocity;
    if (beach) {
      const double clearance_m = liquid_step_internal::SolidDistance(scene.obstacle, position);
      if (clearance_m <= 0.5 * cell_width_m + contact_margin_m) {
        // The free surface carrier may use continued faces next to the wall.
        // Remove its inward component within one accepted RK2 displacement.
        const double inward_speed_m_per_s = std::min(0.0, liquid_step_internal::Dot(velocity, beach->unit_normal));
        velocity =
            liquid_step_internal::Add(velocity, liquid_step_internal::Scale(beach->unit_normal, -inward_speed_m_per_s));
      }
    }
    return velocity;
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
          .obstacle_distance_m = beach ? liquid_step_internal::SolidDistance(scene.obstacle, position)
                                       : std::numeric_limits<double>::max(),
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
  const bool repaired_connectivity = previous && result.frame.mesh.triangles != previous->mesh.triangles;
  const auto final_contact = mark_contact(result.frame.mesh);
  if (!final_contact.ok()) {
    return final_contact;
  }
  // Rebuild the correction normals from the new triangle rings after flips.
  const auto correct_volume = repaired_connectivity ? CorrectSurfaceVolume3D : CorrectSurfaceVolumeWithNormals3D;
  auto corrected = correct_volume(
      result.frame.mesh,
      material_volume,
      endpoint.geometry.cell_width,
      0.005,
      options.maximum_correction_iterations
  );
  if (!corrected.ok()) {
    return corrected.status();
  }
  const auto corrected_contact = mark_contact(result.frame.mesh);
  if (!corrected_contact.ok()) {
    return corrected_contact;
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
