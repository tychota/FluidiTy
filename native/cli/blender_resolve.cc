#include "blender_resolve.h"

#include <cmath>
#include <fstream>
#include <locale>
#include <string>

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
  if (!(input >> version >> preview >> cell_width) || version != "FLUIDITY-BLENDER-1" || preview > 1
      || !std::isfinite(cell_width) || cell_width < 0.001 || cell_width > 10) {
    return absl::InvalidArgumentError("invalid packet version, Preview flag or cell width");
  }
  if (!ReadFiniteVector(input, scene.scene.world_acceleration)
      || !(input >> slab_center >> slab_half_width >> slab_velocity >> count) || !std::isfinite(slab_center)
      || !std::isfinite(slab_half_width) || slab_half_width <= 0 || !std::isfinite(slab_velocity) || count == 0
      || count > kMaximumSeedCount) {
    return absl::InvalidArgumentError("invalid gravity, paddle motion or seed count");
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
    if (!ReadFiniteVector(input, position) || std::abs(position.x - slab_center) <= slab_half_width) {
      return absl::InvalidArgumentError("invalid seed position or seed inside the paddle");
    }
    (*particles)[index] = {
        .id = index + 1,
        .world_position = position,
        .affine_enabled = true,
        .mass = scene.scene.liquid_density * volume,
        .reference_volume = volume,
        .volume_ratio = 1,
        .side_tag = position.x < slab_center ? -1 : 1,
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
  scene.scene.obstacle = TranslatingSlab3D{
      .unit_normal = {1, 0, 0},
      .start_center_coordinate = slab_center,
      .half_width = slab_half_width,
      .wall_velocity = {slab_velocity, 0, 0},
  };
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
