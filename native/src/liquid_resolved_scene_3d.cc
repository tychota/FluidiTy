#include <cmath>
#include <tuple>

#include "fluidity/base/liquid_boundary_math.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/whitewater_sphere_source_3d.h"
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
absl::Status CheckSceneRecordCounts(
    std::size_t source_count,
    std::size_t upstream_content_count,
    std::size_t material_count,
    std::size_t pair_count,
    LiquidCacheLimits3D limits
) {
  if (source_count > limits.maximum_records.sources || upstream_content_count > limits.maximum_upstream_records
      || source_count > (1 << 20) || upstream_content_count > (1 << 20) || material_count > (1 << 20)
      || material_count > limits.maximum_bytes / sizeof(LiquidConstituentMaterial3D) || pair_count > (1 << 20)
      || pair_count > limits.maximum_bytes / sizeof(LiquidPhasePair3D)) {
    return detail::Error(absl::StatusCode::kResourceExhausted);
  }
  return absl::OkStatus();
}
absl::Status CheckPlasticBodyCount(std::size_t body_count, LiquidCacheLimits3D limits) {
  if (body_count > (1 << 20) || body_count > limits.maximum_bytes / sizeof(PlasticSolidCheckpointBinding3D)) {
    return detail::Error(absl::StatusCode::kResourceExhausted);
  }
  return absl::OkStatus();
}

absl::Status ValidateResolvedScene(LiquidResolvedSceneView3D input, LiquidCacheLimits3D limits) {
  SCENE_TRY(CheckSceneRecordCounts(
      input.scene.sources.size(),
      input.upstream.size(),
      input.scene.constituent_materials.size(),
      input.scene.phase_pairs.size(),
      limits
  ));
  if (input.model_version != kLiquidOnlyResolvedModel3D && input.model_version != kPlasticBodyResolvedModel3D) {
    return detail::Error(absl::StatusCode::kFailedPrecondition);
  }
  SCENE_TRY(CheckPlasticBodyCount(input.plastic_bodies.size(), limits));
  const bool has_plastic_bodies = !input.plastic_bodies.empty();
  if (has_plastic_bodies != (input.model_version == kPlasticBodyResolvedModel3D)) {
    return Invalid();
  }
  SCENE_TRY(ValidatePlasticSolidCheckpointBindings3D(input.plastic_bodies));
  if (input.backend > LiquidBackend3D::kCuda || !detail::Finite(input.geometry.world_origin)
      || !detail::Positive(input.geometry.cell_width)) {
    return Invalid();
  }
  const auto& scene = input.scene;
  const auto& options = input.options;
  const bool material_valid = std::isnormal(scene.liquid_density) && scene.liquid_density > 0
      && detail::Nonnegative(scene.dynamic_viscosity) && detail::Nonnegative(scene.surface_tension)
      && std::isfinite(scene.ambient_pressure) && scene.bulk_modulus > 0 && std::isfinite(scene.target_divergence)
      && detail::Nonnegative(scene.friction_coefficient) && detail::Nonnegative(scene.penetration_tolerance);
  if (!material_valid || !detail::Finite(scene.world_acceleration)
      || !ValidLiquidConstituentMaterials3D(scene.constituent_materials)) {
    return Invalid();
  }
  SCENE_TRY(ValidateLiquidPhasePairs3D(scene.phase_pairs, scene.constituent_materials));
  if (scene.eos) {
    if (!std::isinf(scene.bulk_modulus) || !std::isfinite(scene.eos->reference_sound_speed)) {
      return Invalid();
    }
    const auto eos = EvaluateLiquidEosAtPressure3D(
        {scene.liquid_density, scene.eos->reference_sound_speed, scene.eos->law},
        scene.ambient_pressure
    );
    if (!eos.ok()) {
      return eos.status();
    }
  }
  if (scene.drift
      && ((!scene.eos && !std::isinf(scene.bulk_modulus)) || !detail::Positive(scene.drift->correction_time)
          || !detail::Nonnegative(scene.drift->maximum_rate))) {
    return Invalid();
  }
  if (scene.cavity_provider.prepare
      || (scene.cavity
          && (!detail::Positive(scene.cavity->atmospheric_absolute_pressure)
              || !detail::Nonnegative(scene.cavity->birth_hydrostatic_gauge_pressure) || scene.cavity->initialize
              || scene.cavity->atmosphere_outlets))) {
    return Invalid();
  }
  const bool options_valid = detail::Nonnegative(options.particle_energy_blend) && options.particle_energy_blend <= 1
      && detail::Positive(options.maximum_step_duration) && detail::Nonnegative(options.minimum_face_mass)
      && PowerOfTwo(options.mass_per_quantum) && PowerOfTwo(options.momentum_per_quantum)
      && options.pressure_solver_options.max_iterations >= 1 && options.pressure_solver_options.max_iterations <= 256
      && options.motion_sample_intervals <= 64;
  if (!options_valid) {
    return Invalid();
  }
  if (options.ocean_boundary) {
    if (options.ocean_boundary->inlet_particle_spacing_m > 0
        && !std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field)) {
      return Invalid();
    }
    const auto boundary = ResolvedOceanBoundary3D::Resolve(*options.ocean_boundary);
    if (!boundary.ok()) {
      return boundary.status();
    }
  }
  if (const auto& capture = options.whitewater_capture) {
    const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field);
    const bool supported_field = sphere || std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field);
    const bool supported_obstacle =
        std::holds_alternative<std::monostate>(scene.obstacle) || IsStationaryWhitewaterBeach3D(scene.obstacle);
    if (options.motion_sample_intervals == 0 || !supported_field || !supported_obstacle
        || (sphere && sphere->radius <= 2 * capture->source_options.spacing_m)) {
      return Invalid();
    }
    const auto valid_source = ComputeWhitewaterBirths3D(
        {},
        capture->world_source_grid_origin,
        {.start_time_seconds = 0, .end_time_seconds = 1, .quadrature_time_seconds = 1},
        capture->source_options
    );
    if (!valid_source.ok()) {
      return valid_source.status();
    }
  }
  if (const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field)) {
    if (!detail::Finite(sphere->world_center) || !detail::Positive(sphere->radius)) {
      return Invalid();
    }
  } else if (const auto* particles = std::get_if<LiquidParticleSphereUnion3D>(&scene.field)) {
    if (!detail::Positive(particles->radius)) {
      return Invalid();
    }
  } else if (const auto* drop = std::get_if<LiquidIsolatedCovarianceDrop3D>(&scene.field)) {
    if (!detail::Positive(drop->physical_radius) || !detail::Positive(drop->seed_covariance_scale)
        || !std::holds_alternative<std::monostate>(scene.obstacle) || !scene.sources.empty()
        || scene.surface_tension <= 0 || !std::isinf(scene.bulk_modulus) || scene.eos || scene.drift
        || scene.target_divergence != 0) {
      return Invalid();
    }
  } else {
    const auto& plane = std::get<LiquidHorizontalPlaneSurface3D>(scene.field);
    const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.obstacle);
    if (!tank || !std::isfinite(plane.world_surface_height_y) || plane.world_surface_height_y <= tank->floor_y) {
      return Invalid();
    }
  }
  if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.obstacle)) {
    if (scene.friction_coefficient != 0
        || boundary_internal::CheckTankStep(*tank, 0) != boundary_internal::Error::kNone) {
      return Invalid();
    }
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
        || !LiquidParticleUsesRegisteredConstituents3D(
            scene.sources[index].emitted_particle_template,
            scene.constituent_materials
        )
        || (scene.drift && scene.drift->maximum_rate > 0.0
            && scene.sources[index].emitted_particle_template.side_tag != 0)
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
bool AllFieldsPresent(const Table* table, flatbuffers::voffset_t last_vtable_field_offset) {
  for (flatbuffers::voffset_t field = 4; field <= last_vtable_field_offset; field += 2) {
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
absl::StatusOr<LiquidResolvedScene3D> DecodeResolvedSceneHeader(const wire::ResolvedScene* root) {
  if (root->model_version() != kLiquidOnlyResolvedModel3D && root->model_version() != kPlasticBodyResolvedModel3D) {
    return detail::Error(absl::StatusCode::kFailedPrecondition);
  }
  if (root->backend() > wire::Backend::Cuda || root->field()->kind() > wire::FieldKind::IsolatedCovarianceDrop
      || root->obstacle()->kind() > wire::ObstacleKind::OpenTank) {
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
  result.scene.dynamic_viscosity = material->dynamic_viscosity().value_or(0.0);
  result.scene.surface_tension = material->surface_tension();
  result.scene.ambient_pressure = material->ambient_pressure();
  result.scene.bulk_modulus = material->bulk_modulus();
  if (const auto speed = material->reference_sound_speed()) {
    result.scene.eos = LiquidSceneEos3D{
        .reference_sound_speed = *speed,
        .law = static_cast<LiquidEosLaw>(material->eos_law().value_or(wire::LiquidEosLaw::Hencky)),
    };
  } else if (material->eos_law().has_value()) {
    return Invalid();
  }
  const auto drift_time = material->drift_correction_time();
  const auto drift_rate = material->drift_maximum_rate();
  if (drift_time.has_value() != drift_rate.has_value()) {
    return Invalid();
  }
  if (drift_time) {
    result.scene.drift = LiquidSceneDrift3D{*drift_time, *drift_rate};
  }
  const auto cavity_atmosphere = material->cavity_atmospheric_absolute_pressure();
  const auto cavity_hydrostatic = material->cavity_birth_hydrostatic_gauge_pressure();
  if (cavity_atmosphere.has_value() != cavity_hydrostatic.has_value()) {
    return Invalid();
  }
  if (cavity_atmosphere) {
    result.scene.cavity = LiquidSceneCavity3D{
        .atmospheric_absolute_pressure = *cavity_atmosphere,
        .birth_hydrostatic_gauge_pressure = *cavity_hydrostatic,
    };
  }
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
  result.options.motion_sample_intervals = options->motion_sample_intervals();
  if (const auto* capture = options->whitewater_capture()) {
    result.options.whitewater_capture = WhitewaterCaptureSettings3D{
        .world_source_grid_origin = detail::Unpack(capture->source_grid_origin()),
        .source_options =
            {
                .spacing_m = capture->spacing_m(),
                .radius_m = capture->radius_m(),
                .kinetic = {capture->kinetic()->low(), capture->kinetic()->high()},
                .trapped_air = {capture->trapped_air()->low(), capture->trapped_air()->high()},
                .crest = {capture->crest()->low(), capture->crest()->high()},
                .trapped_air_weight = capture->trapped_air_weight(),
                .crest_weight = capture->crest_weight(),
            },
    };
  }
  if (const auto* boundary = options->ocean_boundary()) {
    if (!AllFieldsPresent(boundary, wire::OceanBoundary::VT_OUTLET_RATE_PER_S)) {
      return detail::Corrupt();
    }
    result.options.ocean_boundary = OceanBoundarySettings3D{
        .swell =
            {
                .still_water_height_m = boundary->still_water_height_m(),
                .water_depth_m = boundary->water_depth_m(),
                .wave_height_m = boundary->wave_height_m(),
                .wave_period_s = boundary->wave_period_s(),
                .gravity_m_per_s2 = boundary->gravity_m_per_s2(),
                .propagation_direction = detail::Unpack(boundary->propagation_direction()),
                .phase_at_world_origin_rad = boundary->phase_at_world_origin_rad(),
            },
        .inlet_wall_x_m = boundary->inlet_wall_x_m(),
        .inlet_interior_x_m = boundary->inlet_interior_x_m(),
        .outlet_interior_x_m = boundary->outlet_interior_x_m(),
        .outlet_wall_x_m = boundary->outlet_wall_x_m(),
        .inlet_rate_per_s = boundary->inlet_rate_per_s(),
        .outlet_rate_per_s = boundary->outlet_rate_per_s(),
        .inlet_particle_spacing_m = boundary->inlet_particle_spacing_m(),
        .inlet_low_z_m = boundary->inlet_low_z_m(),
        .inlet_high_z_m = boundary->inlet_high_z_m(),
    };
  }
  const auto* field = root->field();
  const auto* plane = field->horizontal_plane();
  const auto seed_covariance_scale = field->seed_covariance_scale();
  const bool has_horizontal_plane = plane != nullptr;
  const bool expects_horizontal_plane = field->kind() == wire::FieldKind::HorizontalPlane;
  const bool expects_covariance_scale = field->kind() == wire::FieldKind::IsolatedCovarianceDrop;
  if (has_horizontal_plane != expects_horizontal_plane
      || seed_covariance_scale.has_value() != expects_covariance_scale) {
    return detail::Corrupt();
  }
  if (plane) {
    if (!AllFieldsPresent(plane, wire::HorizontalPlane::VT_SURFACE_HEIGHT_Y) || field->radius() != 1
        || !Zero(detail::Unpack(field->center()))) {
      return detail::Corrupt();
    }
    result.scene.field = LiquidHorizontalPlaneSurface3D{plane->surface_height_y()};
  } else if (field->kind() == wire::FieldKind::AnalyticSphere) {
    result.scene.field = LiquidAnalyticSphere3D{detail::Unpack(field->center()), field->radius()};
  } else if (seed_covariance_scale.has_value()) {
    if (!Zero(detail::Unpack(field->center()))) {
      return detail::Corrupt();
    }
    result.scene.field = LiquidIsolatedCovarianceDrop3D{field->radius(), *seed_covariance_scale};
  } else {
    if (!Zero(detail::Unpack(field->center()))) {
      return detail::Corrupt();
    }
    result.scene.field = LiquidParticleSphereUnion3D{field->radius()};
  }
  const auto* obstacle = root->obstacle();
  const auto normal = detail::Unpack(obstacle->normal());
  const auto velocity = detail::Unpack(obstacle->velocity());
  const auto* tank = obstacle->open_tank();
  if ((tank != nullptr) != (obstacle->kind() == wire::ObstacleKind::OpenTank)) {
    return detail::Corrupt();
  }
  if (tank) {
    if (!AllFieldsPresent(tank, wire::OpenTank::VT_PISTON_VELOCITY_X) || !Zero(normal) || !Zero(velocity)
        || obstacle->start_coordinate() != 0 || obstacle->half_width() != 0) {
      return detail::Corrupt();
    }
    result.scene.obstacle = AxisAlignedLiquidTank3D{
        .start_low_x = tank->start_low_x(),
        .high_x = tank->high_x(),
        .floor_y = tank->floor_y(),
        .low_z = tank->low_z(),
        .high_z = tank->high_z(),
        .piston_velocity_x = tank->piston_velocity_x(),
    };
  } else if (obstacle->kind() == wire::ObstacleKind::Slab) {
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
      .plastic_bodies = plastic_bodies.values(),
  };
  result.scene.sources = sources.values();
  result.scene.constituent_materials = constituent_materials.values();
  result.scene.phase_pairs = phase_pairs.values();
  return result;
}
absl::StatusOr<LiquidCacheBytes3D>
EncodeLiquidResolvedScene3D(LiquidResolvedSceneView3D input, LiquidCacheLimits3D limits) {
  SCENE_TRY(ValidateResolvedScene(input, limits));
  const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&input.scene.field);
  const auto* drop = std::get_if<LiquidIsolatedCovarianceDrop3D>(&input.scene.field);
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&input.scene.obstacle);
  // Include the optional tables only for the active variants.
  const auto tables = 4 + 2 * input.scene.sources.size() + input.upstream.size() + (plane != nullptr)
      + (tank != nullptr) + input.options.whitewater_capture.has_value() + input.options.ocean_boundary.has_value();
  return detail::Build<wire::ResolvedScene>(
      tables + input.scene.constituent_materials.size() + input.scene.phase_pairs.size() + input.plastic_bodies.size(),
      limits,
      wire::ResolvedSceneIdentifier(),
      [&](flatbuffers::FlatBufferBuilder& builder) -> absl::StatusOr<flatbuffers::Offset<wire::ResolvedScene>> {
        const auto& scene = input.scene;
        const auto& options = input.options;
        flatbuffers::Optional<double> dynamic_viscosity;
        if (scene.dynamic_viscosity > 0.0) {
          dynamic_viscosity = scene.dynamic_viscosity;
        }
        flatbuffers::Optional<double> reference_sound_speed;
        flatbuffers::Optional<wire::LiquidEosLaw> eos_law;
        if (scene.eos) {
          reference_sound_speed = scene.eos->reference_sound_speed;
          if (scene.eos->law == LiquidEosLaw::kTait) {
            eos_law = wire::LiquidEosLaw::Tait;
          }
        }
        flatbuffers::Optional<double> drift_time;
        flatbuffers::Optional<double> drift_rate;
        if (scene.drift) {
          drift_time = scene.drift->correction_time;
          drift_rate = detail::Canonical(scene.drift->maximum_rate);
        }
        flatbuffers::Optional<double> cavity_atmosphere;
        flatbuffers::Optional<double> cavity_hydrostatic;
        if (scene.cavity) {
          cavity_atmosphere = scene.cavity->atmospheric_absolute_pressure;
          cavity_hydrostatic = detail::Canonical(scene.cavity->birth_hydrostatic_gauge_pressure);
        }
        const auto material = wire::CreateMaterial(
            builder,
            scene.liquid_density,
            detail::Canonical(scene.surface_tension),
            detail::Canonical(scene.ambient_pressure),
            scene.bulk_modulus,
            detail::Canonical(scene.target_divergence),
            detail::Canonical(scene.friction_coefficient),
            detail::Canonical(scene.penetration_tolerance),
            dynamic_viscosity,
            reference_sound_speed,
            eos_law,
            drift_time,
            drift_rate,
            cavity_atmosphere,
            cavity_hydrostatic
        );
        flatbuffers::Offset<wire::WhitewaterCapture> whitewater;
        if (const auto& capture = options.whitewater_capture) {
          const auto& source = capture->source_options;
          const auto origin = detail::PackCanonical(capture->world_source_grid_origin);
          const wire::WhitewaterRampBounds kinetic(detail::Canonical(source.kinetic.low), source.kinetic.high);
          const wire::WhitewaterRampBounds trapped_air(
              detail::Canonical(source.trapped_air.low),
              source.trapped_air.high
          );
          const wire::WhitewaterRampBounds crest(detail::Canonical(source.crest.low), source.crest.high);
          whitewater = wire::CreateWhitewaterCapture(
              builder,
              &origin,
              source.spacing_m,
              source.radius_m,
              &kinetic,
              &trapped_air,
              &crest,
              detail::Canonical(source.trapped_air_weight),
              detail::Canonical(source.crest_weight)
          );
        }
        flatbuffers::Offset<wire::OceanBoundary> ocean;
        if (const auto& boundary = options.ocean_boundary) {
          const auto& swell = boundary->swell;
          const auto direction = detail::PackCanonical(swell.propagation_direction);
          ocean = wire::CreateOceanBoundary(
              builder,
              detail::Canonical(swell.still_water_height_m),
              swell.water_depth_m,
              detail::Canonical(swell.wave_height_m),
              swell.wave_period_s,
              swell.gravity_m_per_s2,
              &direction,
              detail::Canonical(swell.phase_at_world_origin_rad),
              detail::Canonical(boundary->inlet_wall_x_m),
              detail::Canonical(boundary->inlet_interior_x_m),
              detail::Canonical(boundary->outlet_interior_x_m),
              detail::Canonical(boundary->outlet_wall_x_m),
              detail::Canonical(boundary->inlet_rate_per_s),
              detail::Canonical(boundary->outlet_rate_per_s),
              detail::Canonical(boundary->inlet_particle_spacing_m),
              detail::Canonical(boundary->inlet_low_z_m),
              detail::Canonical(boundary->inlet_high_z_m)
          );
        }
        wire::StepOptionsBuilder settings_builder(builder);
        if (options.ocean_boundary) {
          settings_builder.add_ocean_boundary(ocean);
        }
        // Preserve the original v1 bytes when capture is disabled.
        if (options.whitewater_capture) {
          settings_builder.add_whitewater_capture(whitewater);
        }
        if (options.motion_sample_intervals != 0) {
          settings_builder.add_motion_sample_intervals(options.motion_sample_intervals);
        }
        settings_builder.add_momentum_per_quantum(options.momentum_per_quantum);
        settings_builder.add_mass_per_quantum(options.mass_per_quantum);
        settings_builder.add_minimum_face_mass(detail::Canonical(options.minimum_face_mass));
        settings_builder.add_maximum_step_duration(options.maximum_step_duration);
        settings_builder.add_energy_blend(detail::Canonical(options.particle_energy_blend));
        settings_builder.add_pressure_iterations(
            static_cast<std::uint32_t>(options.pressure_solver_options.max_iterations)
        );
        const auto settings = settings_builder.Finish();
        wire::FieldKind field_kind = wire::FieldKind::ParticleUnion;
        Vec3 field_center{};
        double radius = 1;
        flatbuffers::Offset<wire::HorizontalPlane> plane_record;
        flatbuffers::Optional<double> seed_covariance_scale;
        if (const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field)) {
          field_kind = wire::FieldKind::AnalyticSphere;
          field_center = sphere->world_center;
          radius = sphere->radius;
        } else if (plane) {
          field_kind = wire::FieldKind::HorizontalPlane;
          plane_record = wire::CreateHorizontalPlane(builder, detail::Canonical(plane->world_surface_height_y));
        } else if (drop) {
          field_kind = wire::FieldKind::IsolatedCovarianceDrop;
          // The wire center stays zero because each substep fits a new centroid.
          radius = drop->physical_radius;
          seed_covariance_scale = drop->seed_covariance_scale;
        } else {
          radius = std::get<LiquidParticleSphereUnion3D>(scene.field).radius;
        }
        const auto center = detail::PackCanonical(field_center);
        builder.Align(8);
        const auto field = wire::CreateField(builder, field_kind, &center, radius, plane_record, seed_covariance_scale);
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
        flatbuffers::Offset<wire::OpenTank> tank_record;
        if (tank) {
          obstacle_kind = wire::ObstacleKind::OpenTank;
          tank_record = wire::CreateOpenTank(
              builder,
              detail::Canonical(tank->start_low_x),
              detail::Canonical(tank->high_x),
              detail::Canonical(tank->floor_y),
              detail::Canonical(tank->low_z),
              detail::Canonical(tank->high_z),
              detail::Canonical(tank->piston_velocity_x)
          );
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
            &packed_velocity,
            tank_record
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
        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<wire::ConstituentMaterial>>> constituent_materials;
        if (!scene.constituent_materials.empty()) {
          const auto records = detail::BuildVector<wire::ConstituentMaterial>(
              builder,
              scene.constituent_materials.size(),
              [&](std::size_t index) {
                const auto& record = scene.constituent_materials[index];
                return wire::CreateConstituentMaterial(
                    builder,
                    record.substance_id,
                    record.phase_id,
                    record.density_kg_per_m3,
                    record.dynamic_viscosity_pa_s
                );
              }
          );
          if (!records.ok()) {
            return records.status();
          }
          constituent_materials = *records;
        }
        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<wire::LiquidPhasePair>>> phase_pairs;
        if (!scene.phase_pairs.empty()) {
          const auto records =
              detail::BuildVector<wire::LiquidPhasePair>(builder, scene.phase_pairs.size(), [&](std::size_t index) {
                const auto& pair = scene.phase_pairs[index];
                return wire::CreateLiquidPhasePair(
                    builder,
                    pair.phase_a,
                    pair.phase_b,
                    detail::Canonical(pair.surface_tension)
                );
              });
          if (!records.ok()) {
            return records.status();
          }
          phase_pairs = *records;
        }
        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<wire::PlasticBodyBinding>>> plastic_bodies;
        if (!input.plastic_bodies.empty()) {
          const auto records = detail::BuildVector<wire::PlasticBodyBinding>(
              builder,
              input.plastic_bodies.size(),
              [&](std::size_t index) {
                const auto& binding = input.plastic_bodies[index];
                if (const auto* granular = std::get_if<GranularParameters3D>(&binding.material)) {
                  const wire::GranularPlasticMaterial packed(
                      detail::Canonical(granular->lame_lambda),
                      detail::Canonical(granular->shear_modulus),
                      detail::Canonical(granular->friction_angle),
                      detail::Canonical(granular->cohesion_log_strain)
                  );
                  return wire::CreatePlasticBodyBinding(builder, binding.body_id, binding.material_id, &packed);
                }
                const auto& snow = std::get<SnowParameters3D>(binding.material);
                const wire::SnowPlasticMaterial packed(
                    detail::Canonical(snow.lame_lambda),
                    detail::Canonical(snow.shear_modulus),
                    detail::Canonical(snow.compression_limit),
                    detail::Canonical(snow.stretch_limit),
                    detail::Canonical(snow.hardening)
                );
                return wire::CreatePlasticBodyBinding(builder, binding.body_id, binding.material_id, nullptr, &packed);
              }
          );
          if (!records.ok()) {
            return records.status();
          }
          plastic_bodies = *records;
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
            *upstream,
            constituent_materials,
            phase_pairs,
            plastic_bodies
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
  const auto* materials = root->constituent_materials();
  const std::size_t material_count = materials ? materials->size() : 0;
  const auto* pairs = root->liquid_phase_pairs();
  const std::size_t pair_count = pairs ? pairs->size() : 0;
  SCENE_TRY(
      CheckSceneRecordCounts(root->sources()->size(), root->upstream()->size(), material_count, pair_count, limits)
  );
  auto result = DecodeResolvedSceneHeader(root);
  if (!result.ok()) {
    return result.status();
  }
  if (materials) {
    for (flatbuffers::uoffset_t index = 0; index < materials->size(); ++index) {
      const auto* material = materials->Get(index);
      if (!detail::Positive(material->density_kg_per_m3())
          || (index > 0 && materials->Get(index - 1)->substance_id() >= material->substance_id())) {
        return detail::Corrupt();
      }
    }
    SCENE_TRY(detail::Materialize(result->constituent_materials, materials, [](const auto* material) {
      return LiquidConstituentMaterial3D{
          .substance_id = material->substance_id(),
          .phase_id = material->phase_id(),
          .density_kg_per_m3 = material->density_kg_per_m3(),
          .dynamic_viscosity_pa_s = material->dynamic_viscosity_pa_s(),
      };
    }));
  }
  if (pairs) {
    SCENE_TRY(detail::Materialize(result->phase_pairs, pairs, [](const auto* pair) {
      return LiquidPhasePair3D{pair->phase_a(), pair->phase_b(), pair->surface_tension()};
    }));
  }
  const auto* plastic_bodies = root->plastic_bodies();
  if (plastic_bodies != nullptr) {
    SCENE_TRY(CheckPlasticBodyCount(plastic_bodies->size(), limits));
    if (plastic_bodies->size() == 0 || root->model_version() != kPlasticBodyResolvedModel3D) {
      return detail::Corrupt();
    }
    for (const auto* binding : *plastic_bodies) {
      if ((binding->granular() == nullptr) == (binding->snow() == nullptr)) {
        return detail::Corrupt();
      }
    }
    SCENE_TRY(detail::Materialize(result->plastic_bodies, plastic_bodies, [](const auto* binding) {
      DrySolidMaterial3D material;
      if (const auto* granular = binding->granular()) {
        material = GranularParameters3D{
            .shear_modulus = granular->shear_modulus_pa(),
            .lame_lambda = granular->lame_lambda_pa(),
            .friction_angle = granular->friction_angle_rad(),
            .cohesion_log_strain = granular->cohesion_log_strain(),
        };
      } else {
        const auto* snow = binding->snow();
        material = SnowParameters3D{
            .shear_modulus = snow->shear_modulus_pa(),
            .lame_lambda = snow->lame_lambda_pa(),
            .compression_limit = snow->compression_limit(),
            .stretch_limit = snow->stretch_limit(),
            .hardening = snow->hardening(),
        };
      }
      return PlasticSolidCheckpointBinding3D{binding->body_id(), binding->material_id(), material};
    }));
  }
  // Validate the scalar settings and registries before materializing sources.
  if (!ValidateResolvedScene(result->View(), limits).ok()) {
    return detail::Corrupt();
  }
  for (flatbuffers::uoffset_t index = 0; index < root->sources()->size(); ++index) {
    const auto* source = root->sources()->Get(index);
    if (!AllFieldsPresent(source, wire::EmissionSource::VT_PARTICLE_TEMPLATE)
        || !AllFieldsPresent(source->particle_template(), wire::Particle::VT_PIC_INITIALIZED)
        || !detail::ValidParticleEncoding(source->particle_template()) || !ValidSource(ReadSource(source))
        || (index > 0 && root->sources()->Get(index - 1)->ordinal() >= source->ordinal())) {
      return detail::Corrupt();
    }
    if (!LiquidParticleUsesRegisteredConstituents3D(
            ReadSource(source).emitted_particle_template,
            result->constituent_materials.values()
        )) {
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
