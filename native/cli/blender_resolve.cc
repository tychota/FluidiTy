#include "blender_resolve.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <locale>
#include <string>
#include <variant>

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/liquid_cache_3d.h"

namespace fluidity::blender {
namespace {
namespace fs = std::filesystem;
namespace files = liquid_cache_files_internal;

constexpr std::size_t kMaximumSeedCount = 1024;

absl::Status WriteNew(const fs::path& path, const LiquidCacheBytes3D& bytes) {
  files::NativeFile file;
  const auto opened = file.Open(path, true);
  if (!opened.ok()) {
    return opened;
  }
  const auto wrote = file.Write(bytes.values());
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
      || (version != "FLUIDITY-BLENDER-1" && version != "FLUIDITY-BLENDER-2") || preview > 1
      || !std::isfinite(cell_width) || cell_width < 0.001 || cell_width > 10) {
    return absl::InvalidArgumentError("invalid packet version, Preview flag or cell width");
  }
  const bool modern_authoring = version == "FLUIDITY-BLENDER-2";
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
  }
  if (input >> version) {
    return absl::InvalidArgumentError("unexpected data after seed positions");
  }
  state.particles = std::move(*particles);
  const auto seed = EncodeLiquidState3D(state);
  if (!seed.ok()) {
    return seed.status();
  }
  const LiquidUpstreamContent3D initial{
      .role = LiquidUpstreamRole3D::kInitialParticles,
      .byte_length = seed->size(),
      .digest = LiquidContentDigest3D(seed->values()),
  };
  scene.scene.field = LiquidParticleSphereUnion3D{0.9 * cell_width};
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
  return WriteNew(args[4], *resolved);
}
}  // namespace fluidity::blender
