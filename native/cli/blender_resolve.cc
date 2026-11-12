#include "blender_resolve.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <locale>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/solid_interaction_checkpoint_3d.h"

namespace fluidity::blender {
namespace {
namespace fs = std::filesystem;
namespace files = liquid_cache_files_internal;

constexpr std::size_t kMaximumSeedCount = 1024;

absl::Status WriteNew(const fs::path& path, std::span<const std::uint8_t> bytes) {
  files::NativeFile file;
  const auto opened = file.Open(path, true);
  if (!opened.ok()) {
    return opened;
  }
  const auto wrote = file.Write(bytes);
  return wrote.ok() ? file.Flush() : wrote;
}

bool ReadFiniteVector(std::istream& input, Vec3& vector) {
  return (input >> vector.x >> vector.y >> vector.z) && std::isfinite(vector.x) && std::isfinite(vector.y)
      && std::isfinite(vector.z);
}

// Version two exposes existing scene/material models without changing the solver.
// The caller still owns publication of both output artifacts as a private pair.
absl::Status ReadAuthoringParameters(std::istream& input, LiquidResolvedSceneView3D& scene, Vec3& initial_velocity) {
  auto& material = scene.scene;
  auto& options = scene.options;
  unsigned collision_kind = 0;
  if (!(input >> material.liquid_density >> material.surface_tension >> material.friction_coefficient
        >> options.particle_energy_blend >> options.maximum_step_duration
        >> options.pressure_solver_options.max_iterations)) {
    return absl::InvalidArgumentError("invalid material or solver parameters");
  }
  const bool valid_material = std::isfinite(material.liquid_density) && material.liquid_density > 0
      && std::isfinite(material.surface_tension) && material.surface_tension >= 0
      && std::isfinite(material.friction_coefficient) && material.friction_coefficient >= 0;
  if (!valid_material) {
    return absl::InvalidArgumentError("material parameters are outside supported bounds");
  }
  const bool valid_step = std::isfinite(options.particle_energy_blend) && options.particle_energy_blend >= 0
      && options.particle_energy_blend <= 1 && std::isfinite(options.maximum_step_duration)
      && options.maximum_step_duration > 0 && options.maximum_step_duration <= 1.0 / 24
      && options.pressure_solver_options.max_iterations > 0 && options.pressure_solver_options.max_iterations <= 256;
  if (!valid_step) {
    return absl::InvalidArgumentError("solver parameters are outside supported bounds");
  }
  if (!ReadFiniteVector(input, initial_velocity) || !(input >> collision_kind) || collision_kind > 3) {
    return absl::InvalidArgumentError("invalid initial velocity or collision model");
  }
  if (collision_kind == 0) {
    material.obstacle = std::monostate{};
    return absl::OkStatus();
  }
  if (collision_kind == 3) {
    AxisAlignedLiquidTank3D tank{};
    if (!(input >> tank.start_low_x >> tank.high_x >> tank.floor_y >> tank.low_z >> tank.high_z
          >> tank.piston_velocity_x)
        || !std::isfinite(tank.start_low_x) || !std::isfinite(tank.high_x) || !std::isfinite(tank.floor_y)
        || !std::isfinite(tank.low_z) || !std::isfinite(tank.high_z) || !std::isfinite(tank.piston_velocity_x)
        || tank.start_low_x >= tank.high_x || tank.low_z >= tank.high_z) {
      return absl::InvalidArgumentError("invalid open tank bounds");
    }
    material.obstacle = tank;
    return absl::OkStatus();
  }
  Vec3 normal{};
  Vec3 velocity{};
  double coordinate = 0;
  double half_width = 0;
  if (!ReadFiniteVector(input, normal) || !(input >> coordinate >> half_width) || !ReadFiniteVector(input, velocity)
      || !std::isfinite(coordinate) || !std::isfinite(half_width)
      || std::abs(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z - 1) > 1e-12
      || (collision_kind == 2 && half_width <= 0)) {
    return absl::InvalidArgumentError("invalid collider normal, extent or motion");
  }
  if (collision_kind == 1) {
    material.obstacle = TranslatingHalfspace3D{
        .unit_normal = normal,
        .start_plane_coordinate = coordinate,
        .wall_velocity = velocity,
    };
  } else {
    material.obstacle = TranslatingSlab3D{
        .unit_normal = normal,
        .start_center_coordinate = coordinate,
        .half_width = half_width,
        .wall_velocity = velocity,
    };
  }
  return absl::OkStatus();
}
struct ExtendedAuthoring3D {
  std::vector<LiquidConstituentMaterial3D> materials;
  std::vector<LiquidPhasePair3D> pairs;
  std::vector<LiquidEmissionSource3D> emitters;
};

const LiquidConstituentMaterial3D* FindMaterial(const ExtendedAuthoring3D& authoring, std::uint64_t substance_id) {
  const auto material =
      std::ranges::lower_bound(authoring.materials, substance_id, {}, &LiquidConstituentMaterial3D::substance_id);
  return material != authoring.materials.end() && material->substance_id == substance_id ? &*material : nullptr;
}

bool SetPureMaterial(LiquidParticleState3D& particle, const LiquidConstituentMaterial3D& material, double volume) {
  const double mass = material.density_kg_per_m3 * volume;
  if (!std::isfinite(volume) || volume <= 0 || !std::isfinite(mass) || mass <= 0) {
    return false;
  }
  particle.mass = mass;
  particle.reference_volume = volume;
  particle.phase_id = material.phase_id;
  particle.constituent_count = 1;
  particle.constituents[0] = {material.substance_id, 1.0};
  return true;
}

// Version three adds bounded canonical material, phase-pair and emission rows.
absl::Status ReadExtendedAuthoring(std::istream& input, ExtendedAuthoring3D& authoring, bool reads_viscosity) {
  constexpr std::size_t kMaximumMaterials = 64;
  constexpr std::size_t kMaximumPhasePairs = 64;
  constexpr std::size_t kMaximumEmitters = 64;
  std::size_t material_count = 0;
  std::size_t pair_count = 0;
  std::size_t emitter_count = 0;
  if (!(input >> material_count >> pair_count >> emitter_count) || material_count == 0
      || material_count > kMaximumMaterials || pair_count > kMaximumPhasePairs || emitter_count > kMaximumEmitters) {
    return absl::InvalidArgumentError("extended liquid authoring counts exceed bounds");
  }
  authoring.materials.resize(material_count);
  std::uint64_t previous_substance_id = 0;
  for (auto& material : authoring.materials) {
    if (!(input >> material.substance_id >> material.phase_id >> material.density_kg_per_m3)
        || material.substance_id <= previous_substance_id || material.phase_id == 0
        || !std::isfinite(material.density_kg_per_m3) || material.density_kg_per_m3 <= 0) {
      return absl::InvalidArgumentError("invalid authored liquid substance");
    }
    if (reads_viscosity) {
      double viscosity = 0.0;
      if (!(input >> viscosity) || !std::isfinite(viscosity) || (viscosity < 0.0 && viscosity != -1.0)) {
        return absl::InvalidArgumentError("invalid authored liquid viscosity");
      }
      if (viscosity >= 0.0) {
        material.dynamic_viscosity_pa_s = viscosity;
      }
    }
    previous_substance_id = material.substance_id;
  }
  authoring.pairs.resize(pair_count);
  for (auto& pair : authoring.pairs) {
    if (!(input >> pair.phase_a >> pair.phase_b >> pair.surface_tension)) {
      return absl::InvalidArgumentError("invalid authored liquid phase interaction");
    }
  }
  const auto pairs = ValidateLiquidPhasePairs3D(authoring.pairs, authoring.materials);
  if (!pairs.ok()) {
    return pairs;
  }
  authoring.emitters.resize(emitter_count);
  std::uint64_t previous_ordinal = 0;
  for (auto& emitter : authoring.emitters) {
    std::uint64_t substance_id = 0;
    double volume = 0;
    if (!(input >> emitter.source_ordinal >> substance_id >> emitter.particles_per_second >> volume)
        || !ReadFiniteVector(input, emitter.world_position) || !ReadFiniteVector(input, emitter.world_position_jitter)
        || !ReadFiniteVector(input, emitter.emitted_particle_template.world_velocity)) {
      return absl::InvalidArgumentError("invalid authored liquid emitter");
    }
    const auto* material = FindMaterial(authoring, substance_id);
    if (emitter.source_ordinal <= previous_ordinal || !material || !std::isfinite(emitter.particles_per_second)
        || emitter.particles_per_second <= 0 || emitter.world_position_jitter.x < 0
        || emitter.world_position_jitter.y < 0 || emitter.world_position_jitter.z < 0
        || !SetPureMaterial(emitter.emitted_particle_template, *material, volume)) {
      return absl::InvalidArgumentError("liquid emitter parameters are outside supported bounds");
    }
    previous_ordinal = emitter.source_ordinal;
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status ResolveBlenderScene(std::span<const fs::path> args) {
  if (args.size() != 6 || (args[2] != "cpu" && args[2] != "cuda")) {
    return absl::InvalidArgumentError("expected resolve SOLVER cpu|cuda PACKET SCENE STATE");
  }
  std::error_code error;
  if (fs::file_size(args[3], error) > 128 * 1024 || error) {
    return absl::InvalidArgumentError("scene input packet is missing or too large");
  }
  std::ifstream input(args[3], std::ios::binary);
  input.imbue(std::locale::classic());
  std::string version;
  unsigned preview = 0;
  std::size_t count = 0;
  double slab_center = 0;
  double slab_half_width = 0;
  double slab_velocity = 0;
  LiquidResolvedSceneView3D scene;
  double& cell_width = scene.geometry.cell_width;
  if (!(input >> version >> preview >> cell_width)
      || (version != "FLUIDITY-BLENDER-1" && version != "FLUIDITY-BLENDER-2" && version != "FLUIDITY-BLENDER-3"
          && version != "FLUIDITY-BLENDER-4" && version != "FLUIDITY-BLENDER-5")
      || preview > 1 || !std::isfinite(cell_width) || cell_width < 0.001 || cell_width > 10) {
    return absl::InvalidArgumentError("invalid packet version, Preview flag or cell width");
  }
  const bool modern_authoring = version != "FLUIDITY-BLENDER-1";
  const bool viscosity_authoring = version == "FLUIDITY-BLENDER-4";
  const bool snow_authoring = version == "FLUIDITY-BLENDER-5";
  const bool extended_authoring = version == "FLUIDITY-BLENDER-3" || viscosity_authoring || snow_authoring;
  Vec3 initial_velocity{};
  if (!ReadFiniteVector(input, scene.scene.world_acceleration)) {
    return absl::InvalidArgumentError("invalid gravity");
  }
  if (modern_authoring) {
    const auto parameters = ReadAuthoringParameters(input, scene, initial_velocity);
    if (!parameters.ok()) {
      return parameters;
    }
  } else {
    if (!(input >> slab_center >> slab_half_width >> slab_velocity) || !std::isfinite(slab_center)
        || !std::isfinite(slab_half_width) || slab_half_width <= 0 || !std::isfinite(slab_velocity)) {
      return absl::InvalidArgumentError("invalid paddle motion");
    }
    scene.scene.obstacle = TranslatingSlab3D{
        .unit_normal = {1, 0, 0},
        .start_center_coordinate = slab_center,
        .half_width = slab_half_width,
        .wall_velocity = {slab_velocity, 0, 0},
    };
  }
  std::optional<LiquidHorizontalPlaneSurface3D> planar_fill;
  if (viscosity_authoring) {
    unsigned planar = 0;
    if (!(input >> planar) || planar > 1) {
      return absl::InvalidArgumentError("invalid planar tank fill setting");
    }
    if (planar == 1) {
      const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.scene.obstacle);
      double height = 0.0;
      if (!(input >> height) || !tank || tank->piston_velocity_x != 0.0 || !std::isfinite(height)
          || height <= tank->floor_y) {
        return absl::InvalidArgumentError("planar fill requires a stationary tank and a surface above its floor");
      }
      planar_fill = LiquidHorizontalPlaneSurface3D{height};
    }
  }
  ExtendedAuthoring3D extended;
  if (extended_authoring) {
    const auto parameters = ReadExtendedAuthoring(input, extended, viscosity_authoring);
    if (!parameters.ok()) {
      return parameters;
    }
    scene.scene.constituent_materials = extended.materials;
    scene.scene.phase_pairs = extended.pairs;
    scene.scene.sources = extended.emitters;
    scene.options.motion_sample_intervals = snow_authoring ? 0 : 16;
  }
  if (!(input >> count) || count == 0 || count > kMaximumSeedCount) {
    return absl::InvalidArgumentError("invalid seed count");
  }
  const auto executable = files::ReadBytes(args[1], {.maximum_bytes = 1024 * 1024 * 1024});
  if (!executable.ok()) {
    return executable.status();
  }
  LiquidState3D state;
  state.geometry = scene.geometry;
  state.next_particle_id = count + 1;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(count);
  if (!particles.ok()) {
    return particles.status();
  }
  const auto resized = particles->Resize(count);
  if (!resized.ok()) {
    return resized;
  }
  const double volume = cell_width * cell_width * cell_width;
  for (std::size_t index = 0; index < count; ++index) {
    Vec3 position{};
    if (!ReadFiniteVector(input, position)) {
      return absl::InvalidArgumentError("invalid seed position");
    }
    if (planar_fill && position.y >= planar_fill->world_surface_height_y) {
      return absl::InvalidArgumentError("seed position is above the planar fill");
    }
    std::int64_t side_tag = 0;
    if (const auto* slab = std::get_if<TranslatingSlab3D>(&scene.scene.obstacle)) {
      const double signed_coordinate = slab->unit_normal.x * position.x + slab->unit_normal.y * position.y
          + slab->unit_normal.z * position.z - slab->start_center_coordinate;
      if (std::abs(signed_coordinate) <= slab->half_width) {
        return absl::InvalidArgumentError("seed position is inside the slab collider");
      }
      side_tag = signed_coordinate < 0 ? -1 : 1;
    }
    if (const auto* plane = std::get_if<TranslatingHalfspace3D>(&scene.scene.obstacle)) {
      const double signed_distance = plane->unit_normal.x * position.x + plane->unit_normal.y * position.y
          + plane->unit_normal.z * position.z - plane->start_plane_coordinate;
      if (signed_distance <= 0) {
        return absl::InvalidArgumentError("seed position is inside the plane collider");
      }
    }
    if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.scene.obstacle)) {
      if (position.x <= tank->start_low_x || position.x >= tank->high_x || position.y <= tank->floor_y
          || position.z <= tank->low_z || position.z >= tank->high_z) {
        return absl::InvalidArgumentError("seed position is outside the open tank");
      }
    }
    (*particles)[index] = {
        .id = index + 1,
        .world_position = position,
        .world_velocity = initial_velocity,
        .affine_enabled = true,
        .mass = scene.scene.liquid_density * volume,
        .reference_volume = volume,
        .volume_ratio = 1,
        .side_tag = side_tag,
    };
    if (extended_authoring) {
      std::uint64_t substance_id = 0;
      double particle_volume = 0;
      if (!(input >> substance_id >> particle_volume)) {
        return absl::InvalidArgumentError("missing extended seed substance or volume");
      }
      const auto* material = FindMaterial(extended, substance_id);
      if (!material || !SetPureMaterial((*particles)[index], *material, particle_volume)) {
        return absl::InvalidArgumentError("invalid extended seed substance or volume");
      }
    }
  }
  auto source_states = LiquidRecords3D<LiquidSourceState3D>::Create(extended.emitters.size());
  if (!source_states.ok()) {
    return source_states.status();
  }
  const auto sized_sources = source_states->Resize(extended.emitters.size());
  if (!sized_sources.ok()) {
    return sized_sources;
  }
  for (std::size_t index = 0; index < extended.emitters.size(); ++index) {
    (*source_states)[index].source_ordinal = extended.emitters[index].source_ordinal;
  }
  state.sources = std::move(*source_states);
  state.particles = std::move(*particles);
  PlasticSolidCheckpointBinding3D snow_binding{};
  SolidInteractionCheckpoint3D snow_state{};
  if (snow_authoring) {
    const auto* platen = std::get_if<TranslatingHalfspace3D>(&scene.scene.obstacle);
    if (platen == nullptr || platen->unit_normal.x != 0 || platen->unit_normal.y != -1 || platen->unit_normal.z != 0
        || platen->wall_velocity.x != 0 || platen->wall_velocity.y >= 0 || platen->wall_velocity.z != 0
        || extended.emitters.empty()) {
      return absl::InvalidArgumentError("snow authoring requires a descending vertical platen and a water emitter");
    }
    SnowParameters3D snow{};
    double snow_density = 0;
    double particle_volume = 0;
    std::size_t snow_count = 0;
    if (!(input >> snow_binding.body_id >> snow_binding.material_id >> snow.lame_lambda >> snow.shear_modulus
          >> snow.compression_limit >> snow.stretch_limit >> snow.hardening >> snow_density >> particle_volume
          >> snow_count)
        || snow_count == 0 || snow_count > kMaximumSeedCount || !std::isfinite(snow_density) || snow_density <= 0
        || !std::isfinite(particle_volume) || particle_volume <= 0) {
      return absl::InvalidArgumentError("invalid snow material, density, volume, or seed count");
    }
    snow_binding.material = snow;
    const auto bindings = std::span(&snow_binding, 1);
    const auto valid = ValidatePlasticSolidCheckpointBindings3D(bindings);
    if (!valid.ok()) {
      return valid;
    }
    for (std::size_t index = 0; index < snow_count; ++index) {
      Vec3 position{};
      if (!ReadFiniteVector(input, position) || position.y < 0 || position.y >= -platen->start_plane_coordinate) {
        return absl::InvalidArgumentError("place snow seeds between the floor and the platen");
      }
      const DrySolidParticle3D particle{
          .world_position = position,
          .mass = snow_density * particle_volume,
          .reference_volume = particle_volume,
          .elastic_deformation = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
      };
      const auto record = CapturePlasticSolidCheckpointParticle3D(
          index + 1,
          snow_binding.body_id,
          snow_binding.material_id,
          snow,
          particle
      );
      if (!record.ok()) {
        return record.status();
      }
      snow_state.plastic_particles.push_back(*record);
    }
    scene.model_version = kPlasticBodyResolvedModel3D;
    scene.plastic_bodies = bindings;
  }
  if (input >> version) {
    return absl::InvalidArgumentError("unexpected data after seed positions");
  }
  auto seed = [&]() -> absl::StatusOr<std::vector<std::uint8_t>> {
    if (snow_authoring) {
      return EncodeCoupledInteractionCheckpoint3D(state, snow_state, scene.plastic_bodies);
    }
    const auto liquid_seed = EncodeLiquidState3D(state);
    if (!liquid_seed.ok()) {
      return liquid_seed.status();
    }
    return std::vector(liquid_seed->values().begin(), liquid_seed->values().end());
  }();
  if (!seed.ok()) {
    return seed.status();
  }
  const LiquidUpstreamContent3D initial{
      .role = LiquidUpstreamRole3D::kInitialParticles,
      .byte_length = seed->size(),
      .digest = LiquidContentDigest3D(*seed),
  };
  if (planar_fill) {
    scene.scene.field = *planar_fill;
  } else {
    scene.scene.field = LiquidParticleSphereUnion3D{0.9 * cell_width};
  }
  if (preview != 0) {
    scene.options.maximum_step_duration /= 2;
  }
  scene.backend = args[2] == "cuda" ? LiquidBackend3D::kCuda : LiquidBackend3D::kCpu;
  scene.build_digest = LiquidContentDigest3D(executable->values());
  scene.upstream = std::span(&initial, 1);
  const auto resolved = EncodeLiquidResolvedScene3D(scene);
  if (!resolved.ok()) {
    return resolved.status();
  }
  const auto wrote_seed = WriteNew(args[5], *seed);
  if (!wrote_seed.ok()) {
    return wrote_seed;
  }
  return WriteNew(args[4], resolved->values());
}
}  // namespace fluidity::blender
