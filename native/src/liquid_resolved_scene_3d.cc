#include <cmath>
#include <tuple>

#include "fluidity/host/liquid_cache_3d.h"
#include "liquid_cache_3d_internal.h"
#include "resolved_scene_generated.h"

namespace fluidity {
namespace detail = liquid_cache_internal;
namespace wire = cache::v1;
namespace {
#define SCENE_TRY(expression) \
  do { \
    const auto status = (expression); \
    if (!status.ok()) { \
      return status; \
    } \
  } while (false)
absl::Status Invalid() {
  return detail::Error(absl::StatusCode::kInvalidArgument);
}
bool Zero(Vec3 vector) {
  return vector.x == 0 && vector.y == 0 && vector.z == 0;
}
bool PowerOfTwo(double value) {
  int exponent = 0;
  return std::isnormal(value) && value > 0 && std::frexp(value, &exponent) == 0.5;
}
bool UnitNormal(Vec3 value) {
  return detail::Finite(value) && std::abs(value.x * value.x + value.y * value.y + value.z * value.z - 1) <= 1e-12;
}
bool ValidSource(const LiquidEmissionSource3D& source) {
  const auto jitter = source.world_position_jitter;
  return detail::Nonnegative(source.particles_per_second) && detail::Finite(source.world_position)
      && detail::Finite(jitter) && jitter.x >= 0 && jitter.y >= 0 && jitter.z >= 0
      && detail::ValidParticle(source.emitted_particle_template);
}
absl::Status BoundScene(std::size_t sources, std::size_t upstream, LiquidCacheLimits3D limits) {
  if (sources > limits.maximum_records.sources || upstream > limits.maximum_upstream_records || sources > (1 << 20)
      || upstream > (1 << 20)) {
    return detail::Error(absl::StatusCode::kResourceExhausted);
  }
  return absl::OkStatus();
}
absl::Status ValidateScene(LiquidResolvedSceneView3D input, LiquidCacheLimits3D limits) {
  SCENE_TRY(BoundScene(input.scene.sources.size(), input.upstream.size(), limits));
  if (input.model_version != 1) {
    return detail::Error(absl::StatusCode::kFailedPrecondition);
  }
  if (input.backend > LiquidBackend3D::kCuda || !detail::Finite(input.geometry.world_origin)
      || !detail::Positive(input.geometry.cell_width)) {
    return Invalid();
  }
  const auto& scene = input.scene;
  const auto& options = input.options;
  const bool material_valid = std::isnormal(scene.liquid_density) && scene.liquid_density > 0
      && detail::Nonnegative(scene.surface_tension) && std::isfinite(scene.ambient_pressure) && scene.bulk_modulus > 0
      && std::isfinite(scene.target_divergence) && detail::Nonnegative(scene.friction_coefficient)
      && detail::Nonnegative(scene.penetration_tolerance);
  if (!material_valid || !detail::Finite(scene.world_acceleration)) {
    return Invalid();
  }
  const bool options_valid = detail::Nonnegative(options.particle_energy_blend) && options.particle_energy_blend <= 1
      && detail::Positive(options.maximum_step_duration) && detail::Nonnegative(options.minimum_face_mass)
      && PowerOfTwo(options.mass_per_quantum) && PowerOfTwo(options.momentum_per_quantum)
      && options.pressure_solver_options.max_iterations >= 1 && options.pressure_solver_options.max_iterations <= 256;
  if (!options_valid) {
    return Invalid();
  }
  if (const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field)) {
    if (!detail::Finite(sphere->world_center) || !detail::Positive(sphere->radius)) {
      return Invalid();
    }
  } else if (!detail::Positive(std::get<LiquidParticleSphereUnion3D>(scene.field).radius)) {
    return Invalid();
  }
  if (const auto* wall = std::get_if<TranslatingHalfspace3D>(&scene.obstacle)) {
    if (!UnitNormal(wall->unit_normal) || !std::isfinite(wall->start_plane_coordinate)
        || !detail::Finite(wall->wall_velocity)) {
      return Invalid();
    }
  }
  if (const auto* wall = std::get_if<TranslatingSlab3D>(&scene.obstacle)) {
    if (!UnitNormal(wall->unit_normal) || !std::isfinite(wall->start_center_coordinate)
        || !detail::Positive(wall->half_width) || !detail::Finite(wall->wall_velocity)) {
      return Invalid();
    }
  }
  for (std::size_t index = 0; index < scene.sources.size(); ++index) {
    if (!ValidSource(scene.sources[index])
        || (index > 0 && scene.sources[index - 1].source_ordinal >= scene.sources[index].source_ordinal)) {
      return Invalid();
    }
  }
  for (std::size_t index = 0; index < input.upstream.size(); ++index) {
    const auto& content = input.upstream[index];
    if (content.role > LiquidUpstreamRole3D::kObstacle) {
      return Invalid();
    }
    if (index > 0) {
      const auto& previous = input.upstream[index - 1];
      if (std::tuple(previous.role, previous.ordinal) >= std::tuple(content.role, content.ordinal)) {
        return Invalid();
      }
    }
  }
  return absl::OkStatus();
}
// Resolved v1 requires every scalar field, including defaults. Later schema
// defaults therefore cannot reinterpret an old artifact's omitted setting.
template <typename Table>
bool AllFieldsPresent(const Table* table, flatbuffers::voffset_t last) {
  for (flatbuffers::voffset_t field = 4; field <= last; field += 2) {
    if (!flatbuffers::IsFieldPresent(table, static_cast<typename Table::FlatBuffersVTableOffset>(field))) {
      return false;
    }
  }
  return true;
}
LiquidEmissionSource3D ReadSource(const wire::EmissionSource* source) {
  return {
      source->ordinal(),
      source->particles_per_second(),
      detail::Unpack(source->position()),
      detail::Unpack(source->jitter()),
      detail::Unpack(source->particle_template()),
  };
}
LiquidUpstreamContent3D ReadUpstream(const wire::UpstreamContent* upstream) {
  return {
      static_cast<LiquidUpstreamRole3D>(upstream->role()),
      upstream->ordinal(),
      upstream->byte_length(),
      detail::Unpack(upstream->digest()),
  };
}
absl::StatusOr<LiquidResolvedScene3D> ReadScalars(const wire::ResolvedScene* root) {
  if (root->model_version() != 1) {
    return detail::Error(absl::StatusCode::kFailedPrecondition);
  }
  if (root->backend() > wire::Backend::Cuda || root->field()->kind() > wire::FieldKind::AnalyticSphere
      || root->obstacle()->kind() > wire::ObstacleKind::Slab) {
    return detail::Corrupt();
  }
  if (!AllFieldsPresent(root, wire::ResolvedScene::VT_UPSTREAM)
      || !AllFieldsPresent(root->material(), wire::Material::VT_PENETRATION_TOLERANCE)
      || !AllFieldsPresent(root->options(), wire::StepOptions::VT_PRESSURE_ITERATIONS)
      || !AllFieldsPresent(root->field(), wire::Field::VT_RADIUS)
      || !AllFieldsPresent(root->obstacle(), wire::Obstacle::VT_VELOCITY)) {
    return detail::Corrupt();
  }
  LiquidResolvedScene3D result;
  result.geometry = {detail::Unpack(root->world_origin()), root->cell_width()};
  result.random_seed = root->random_seed();
  result.backend = static_cast<LiquidBackend3D>(root->backend());
  result.model_version = root->model_version();
  result.build_digest = detail::Unpack(root->build_digest());
  result.scene.world_acceleration = detail::Unpack(root->acceleration());
  const auto* material = root->material();
  result.scene.liquid_density = material->density();
  result.scene.surface_tension = material->surface_tension();
  result.scene.ambient_pressure = material->ambient_pressure();
  result.scene.bulk_modulus = material->bulk_modulus();
  result.scene.target_divergence = material->target_divergence();
  result.scene.friction_coefficient = material->friction();
  result.scene.penetration_tolerance = material->penetration_tolerance();
  const auto* options = root->options();
  result.options.particle_energy_blend = options->energy_blend();
  result.options.maximum_step_duration = options->maximum_step_duration();
  result.options.minimum_face_mass = options->minimum_face_mass();
  result.options.mass_per_quantum = options->mass_per_quantum();
  result.options.momentum_per_quantum = options->momentum_per_quantum();
  result.options.pressure_solver_options.max_iterations = options->pressure_iterations();
  const auto* field = root->field();
  if (field->kind() == wire::FieldKind::AnalyticSphere) {
    result.scene.field = LiquidAnalyticSphere3D{detail::Unpack(field->center()), field->radius()};
  } else {
    if (!Zero(detail::Unpack(field->center()))) {
      return detail::Corrupt();
    }
    result.scene.field = LiquidParticleSphereUnion3D{field->radius()};
  }
  const auto* obstacle = root->obstacle();
  const auto normal = detail::Unpack(obstacle->normal());
  const auto velocity = detail::Unpack(obstacle->velocity());
  if (obstacle->kind() == wire::ObstacleKind::Slab) {
    result.scene.obstacle = TranslatingSlab3D{normal, obstacle->start_coordinate(), obstacle->half_width(), velocity};
  } else if (obstacle->kind() == wire::ObstacleKind::Halfspace) {
    if (obstacle->half_width() != 0) {
      return detail::Corrupt();
    }
    result.scene.obstacle = TranslatingHalfspace3D{normal, obstacle->start_coordinate(), velocity};
  } else if (!Zero(normal) || !Zero(velocity) || obstacle->start_coordinate() != 0 || obstacle->half_width() != 0) {
    return detail::Corrupt();
  }
  return result;
}
}  // namespace

LiquidResolvedSceneView3D LiquidResolvedScene3D::View() const {
  LiquidResolvedSceneView3D result{
      .geometry = geometry,
      .random_seed = random_seed,
      .scene = scene,
      .options = options,
      .backend = backend,
      .model_version = model_version,
      .build_digest = build_digest,
      .upstream = upstream.values(),
  };
  result.scene.sources = sources.values();
  return result;
}
absl::StatusOr<LiquidCacheBytes3D>
EncodeLiquidResolvedScene3D(LiquidResolvedSceneView3D input, LiquidCacheLimits3D limits) {
  if (std::holds_alternative<AxisAlignedLiquidTank3D>(input.scene.obstacle)) {
    return absl::FailedPreconditionError("Open tank resolved encoding is not supported");
  }
  SCENE_TRY(ValidateScene(input, limits));
  // Exactly two tables per source (template and source), one per upstream record.
  const auto tables = 4 + 2 * input.scene.sources.size() + input.upstream.size();
  return detail::Build<wire::ResolvedScene>(
      tables,
      limits,
      wire::ResolvedSceneIdentifier(),
      [&](flatbuffers::FlatBufferBuilder& builder) -> absl::StatusOr<flatbuffers::Offset<wire::ResolvedScene>> {
        const auto& scene = input.scene;
        const auto& options = input.options;
        const auto material = wire::CreateMaterial(
            builder,
            scene.liquid_density,
            detail::Canonical(scene.surface_tension),
            detail::Canonical(scene.ambient_pressure),
            scene.bulk_modulus,
            detail::Canonical(scene.target_divergence),
            detail::Canonical(scene.friction_coefficient),
            detail::Canonical(scene.penetration_tolerance)
        );
        const auto settings = wire::CreateStepOptions(
            builder,
            detail::Canonical(options.particle_energy_blend),
            options.maximum_step_duration,
            detail::Canonical(options.minimum_face_mass),
            options.mass_per_quantum,
            options.momentum_per_quantum,
            static_cast<std::uint32_t>(options.pressure_solver_options.max_iterations)
        );
        wire::FieldKind field_kind = wire::FieldKind::ParticleUnion;
        Vec3 field_center{};
        double radius = 1;
        if (const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field)) {
          field_kind = wire::FieldKind::AnalyticSphere;
          field_center = sphere->world_center;
          radius = sphere->radius;
        } else {
          radius = std::get<LiquidParticleSphereUnion3D>(scene.field).radius;
        }
        const auto center = detail::PackCanonical(field_center);
        builder.Align(8);
        const auto field = wire::CreateField(builder, field_kind, &center, radius);
        wire::ObstacleKind obstacle_kind = wire::ObstacleKind::None;
        Vec3 normal{};
        Vec3 wall_velocity{};
        double coordinate = 0;
        double half_width = 0;
        if (const auto* wall = std::get_if<TranslatingHalfspace3D>(&scene.obstacle)) {
          obstacle_kind = wire::ObstacleKind::Halfspace;
          normal = wall->unit_normal;
          coordinate = wall->start_plane_coordinate;
          wall_velocity = wall->wall_velocity;
        }
        if (const auto* wall = std::get_if<TranslatingSlab3D>(&scene.obstacle)) {
          obstacle_kind = wire::ObstacleKind::Slab;
          normal = wall->unit_normal;
          coordinate = wall->start_center_coordinate;
          half_width = wall->half_width;
          wall_velocity = wall->wall_velocity;
        }
        const auto packed_normal = detail::PackCanonical(normal);
        const auto packed_velocity = detail::PackCanonical(wall_velocity);
        // flatc parses inline structs before starting their containing table. Keep
        // that alignment outside the vtable object extent for canonical v1 bytes.
        builder.Align(8);
        const auto obstacle = wire::CreateObstacle(
            builder,
            obstacle_kind,
            &packed_normal,
            detail::Canonical(coordinate),
            half_width,
            &packed_velocity
        );
        auto sources = detail::BuildVector<wire::EmissionSource>(builder, scene.sources.size(), [&](std::size_t index) {
          const auto& source = scene.sources[index];
          builder.Align(8);
          const auto particle = detail::PackParticle(builder, source.emitted_particle_template, true);
          const auto position = detail::PackCanonical(source.world_position);
          const auto jitter = detail::PackCanonical(source.world_position_jitter);
          return wire::CreateEmissionSource(
              builder,
              source.source_ordinal,
              detail::Canonical(source.particles_per_second),
              &position,
              &jitter,
              particle
          );
        });
        if (!sources.ok()) {
          return sources.status();
        }
        auto upstream =
            detail::BuildVector<wire::UpstreamContent>(builder, input.upstream.size(), [&](std::size_t index) {
              const auto& content = input.upstream[index];
              const auto digest = detail::Pack(content.digest);
              return wire::CreateUpstreamContent(
                  builder,
                  static_cast<wire::UpstreamRole>(content.role),
                  content.ordinal,
                  content.byte_length,
                  &digest
              );
            });
        if (!upstream.ok()) {
          return upstream.status();
        }
        const auto origin = detail::PackCanonical(input.geometry.world_origin);
        const auto acceleration = detail::PackCanonical(scene.world_acceleration);
        const auto build = detail::Pack(input.build_digest);
        return wire::CreateResolvedScene(
            builder,
            1,
            0,
            input.model_version,
            static_cast<wire::Backend>(input.backend),
            &build,
            &origin,
            input.geometry.cell_width,
            input.random_seed,
            &acceleration,
            material,
            settings,
            field,
            obstacle,
            *sources,
            *upstream
        );
      }
  );
}
absl::StatusOr<LiquidResolvedScene3D>
DecodeLiquidResolvedScene3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  auto verified = detail::Verify<wire::ResolvedScene>(bytes, wire::ResolvedSceneIdentifier(), limits);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* root = *verified;
  SCENE_TRY(BoundScene(root->sources()->size(), root->upstream()->size(), limits));
  auto result = ReadScalars(root);
  if (!result.ok()) {
    return result.status();
  }
  // Empty owning arrays let us validate all scalar settings without allocations.
  if (!ValidateScene(result->View(), limits).ok()) {
    return detail::Corrupt();
  }
  for (flatbuffers::uoffset_t index = 0; index < root->sources()->size(); ++index) {
    const auto* source = root->sources()->Get(index);
    if (!AllFieldsPresent(source, wire::EmissionSource::VT_PARTICLE_TEMPLATE)
        || !AllFieldsPresent(source->particle_template(), wire::Particle::VT_PIC_INITIALIZED)
        || !detail::ValidParticleBooleans(source->particle_template()) || !ValidSource(ReadSource(source))
        || (index > 0 && root->sources()->Get(index - 1)->ordinal() >= source->ordinal())) {
      return detail::Corrupt();
    }
  }
  for (flatbuffers::uoffset_t index = 0; index < root->upstream()->size(); ++index) {
    const auto* content = root->upstream()->Get(index);
    if (!AllFieldsPresent(content, wire::UpstreamContent::VT_DIGEST)
        || content->role() > wire::UpstreamRole::Obstacle) {
      return detail::Corrupt();
    }
    if (index > 0) {
      const auto* previous = root->upstream()->Get(index - 1);
      if (std::tuple(previous->role(), previous->ordinal()) >= std::tuple(content->role(), content->ordinal())) {
        return detail::Corrupt();
      }
    }
  }
  SCENE_TRY(detail::Materialize(result->sources, root->sources(), ReadSource));
  SCENE_TRY(detail::Materialize(result->upstream, root->upstream(), ReadUpstream));
  return std::move(*result);
}
absl::StatusOr<LiquidDigest3D> ResolvedSceneFingerprint(LiquidResolvedSceneView3D scene, LiquidCacheLimits3D limits) {
  auto bytes = EncodeLiquidResolvedScene3D(scene, limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  return detail::DomainDigest("fluidity/simulate/resolved/v1", bytes->values());
}
#undef SCENE_TRY
}  // namespace fluidity
