#include <charconv>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/liquid_cache_3d.h"

namespace fluidity {
namespace {
namespace fs = std::filesystem;
namespace files = liquid_cache_files_internal;

constexpr Vec3 kFreeFallInitialPosition{2.5, 2.5, 2.5};
constexpr Vec3 kTankLayoutInitialPosition{0.625, 0.625, 0.625};
constexpr int kFixtureProtocolVersion = 1;

absl::StatusOr<double> EnergyBlend(const fs::path& case_name) {
  if (case_name == "free-fall-pic") {
    return 0.0;
  }
  if (case_name == "free-fall-default" || case_name == "tank-layout") {
    return 0.6;
  }
  if (case_name == "free-fall-retained") {
    return 1.0;
  }
  return absl::InvalidArgumentError("unknown acceptance scene");
}

absl::StatusOr<LiquidBackend3D> ParseBackend(const fs::path& backend) {
  if (backend == "cpu") {
    return LiquidBackend3D::kCpu;
  }
  if (backend == "cuda") {
    return LiquidBackend3D::kCuda;
  }
  return absl::InvalidArgumentError("backend must be cpu or cuda");
}

absl::StatusOr<LiquidDigest3D> ParseFingerprint(const fs::path& argument) {
  const auto& hex = argument.native();
#if defined(_WIN32)
  constexpr std::wstring_view kDigits = L"0123456789abcdef";
#else
  constexpr std::string_view kDigits = "0123456789abcdef";
#endif
  LiquidDigest3D fingerprint{};
  if (hex.size() != 2 * fingerprint.size()) {
    return absl::InvalidArgumentError("expected a 64-digit lowercase fingerprint");
  }
  for (std::size_t index = 0; index < fingerprint.size(); ++index) {
    const auto high = kDigits.find(hex[2 * index]);
    const auto low = kDigits.find(hex[2 * index + 1]);
    if (high == kDigits.npos || low == kDigits.npos) {
      return absl::InvalidArgumentError("expected a 64-digit lowercase fingerprint");
    }
    fingerprint[index] = static_cast<std::uint8_t>(16 * high + low);
  }
  return fingerprint;
}

absl::StatusOr<std::uint64_t> ParseFrameOrdinal(const fs::path& argument) {
  const auto& native = argument.native();
  std::string digits;
  for (const auto character : native) {
    if (character < '0' || character > '9') {
      return absl::InvalidArgumentError("frame ordinal must be an unsigned decimal integer");
    }
    digits += static_cast<char>(character);
  }
  std::uint64_t ordinal = 0;
  const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), ordinal);
  if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size()) {
    return absl::InvalidArgumentError("frame ordinal is out of range or empty");
  }
  return ordinal;
}

absl::Status RequireMissingPath(const fs::path& path) {
  std::error_code error;
  const auto status = fs::symlink_status(path, error);
  if (status.type() != fs::file_type::not_found) {
    return absl::AlreadyExistsError("scene and seed output paths must be fresh");
  }
  return absl::OkStatus();
}

absl::Status WriteNewFile(const fs::path& path, const LiquidCacheBytes3D& bytes) {
  files::NativeFile file;
  const auto opened = file.Open(path, /*write=*/true);
  if (!opened.ok()) {
    return opened;
  }
  const auto written = file.Write(bytes.values());
  if (!written.ok()) {
    return written;
  }
  return file.Flush();
}

void PrintClock(std::ostream& out, const LiquidClock3D& clock) {
  out << "\"frame\":" << clock.frame_ordinal << ",\"time_seconds\":" << clock.time
      << ",\"substep\":" << clock.substep_ordinal << ",\"accepted_step\":" << clock.accepted_step_ordinal
      << ",\"epoch\":" << clock.epoch;
}

void PrintIdentity(
    std::ostream& out,
    const LiquidDigest3D& fingerprint,
    const LiquidDigest3D& execution_id,
    LiquidBackend3D backend
) {
  out << "\"resolved_fingerprint\":\"" << LiquidDigestHex3D(fingerprint).data() << "\",\"execution_id\":\""
      << LiquidDigestHex3D(execution_id).data() << "\",\"backend\":\""
      << (backend == LiquidBackend3D::kCpu ? "cpu" : "cuda") << '"';
}

void PrintVector(std::ostream& out, Vec3 vector) {
  out << '[' << vector.x << ',' << vector.y << ',' << vector.z << ']';
}

absl::Status WriteScene(
    const fs::path& case_name,
    const fs::path& backend_argument,
    const fs::path& solver_executable_path,
    const fs::path& scene_path
) {
  const auto energy_blend = EnergyBlend(case_name);
  if (!energy_blend.ok()) {
    return energy_blend.status();
  }
  const auto backend = ParseBackend(backend_argument);
  if (!backend.ok()) {
    return backend.status();
  }
  auto seed_path = scene_path;
  seed_path += ".flst";
  for (const auto& path : {scene_path, seed_path}) {
    const auto fresh = RequireMissingPath(path);
    if (!fresh.ok()) {
      return fresh;
    }
  }

  const auto solver_executable_bytes = files::ReadBytes(solver_executable_path, {.maximum_bytes = 1024 * 1024 * 1024});
  if (!solver_executable_bytes.ok()) {
    return solver_executable_bytes.status();
  }

  const bool tank_layout = case_name == "tank-layout";
  const Vec3 initial_position = tank_layout ? kTankLayoutInitialPosition : kFreeFallInitialPosition;
  LiquidState3D state;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(1);
  if (!particles.ok()) {
    return particles.status();
  }
  const auto resized = particles->Resize(1);
  if (!resized.ok()) {
    return resized;
  }
  (*particles)[0] = {
      .id = 1,
      .world_position = initial_position,
      .world_velocity = {},
      .velocity_slopes = {},
      .affine_enabled = true,
      .mass = 1000,
      .reference_volume = 1,
      .volume_ratio = 1,
      .pic_initialized = false,
  };
  state.particles = std::move(*particles);
  state.next_particle_id = 2;
  const auto seed = EncodeLiquidState3D(state);
  if (!seed.ok()) {
    return seed.status();
  }
  const LiquidUpstreamContent3D initial_particles{
      .role = LiquidUpstreamRole3D::kInitialParticles,
      .ordinal = 0,
      .byte_length = seed->size(),
      .digest = LiquidContentDigest3D(seed->values()),
  };
  LiquidResolvedSceneView3D scene;
  scene.geometry = state.geometry;
  scene.random_seed = state.random_seed;
  if (tank_layout) {
    scene.scene.field = LiquidHorizontalPlaneSurface3D{0.75};
    scene.scene.obstacle = AxisAlignedLiquidTank3D{
        .start_low_x = -1,
        .high_x = 1,
        .floor_y = 0,
        .low_z = -1,
        .high_z = 1,
        .piston_velocity_x = 0.125,
    };
  } else {
    scene.scene.field = LiquidAnalyticSphere3D{initial_position, 1.1};
  }
  scene.scene.world_acceleration = {0, -9.81, 0};
  scene.scene.liquid_density = 1000;
  scene.scene.surface_tension = 0;
  scene.options.particle_energy_blend = *energy_blend;
  scene.options.maximum_step_duration = 0.01;
  scene.backend = *backend;
  scene.build_digest = LiquidContentDigest3D(solver_executable_bytes->values());
  scene.upstream = std::span(&initial_particles, 1);
  const auto resolved = EncodeLiquidResolvedScene3D(scene);
  if (!resolved.ok()) {
    return resolved.status();
  }
  const auto fingerprint = ResolvedSceneFingerprint(scene);
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  const auto wrote_seed = WriteNewFile(seed_path, *seed);
  if (!wrote_seed.ok()) {
    return wrote_seed;
  }
  const auto wrote_scene = WriteNewFile(scene_path, *resolved);
  if (!wrote_scene.ok()) {
    return wrote_scene;
  }
  std::cout << "{\"fixture_version\":" << kFixtureProtocolVersion << ",\"event\":\"scene\",";
  PrintIdentity(std::cout, *fingerprint, LiquidExecutionId3D(*fingerprint), *backend);
  std::cout << ",\"case\":\"" << case_name.string() << "\",\"energy_blend\":" << *energy_blend
            << ",\"solver_digest\":\"" << LiquidDigestHex3D(scene.build_digest).data() << "\",\"seed_digest\":\""
            << LiquidDigestHex3D(initial_particles.digest).data()
            << "\",\"seed_byte_length\":" << initial_particles.byte_length << "}\n";
  return absl::OkStatus();
}

absl::Status
ReadFrames(const LiquidCache3D& cache, LiquidBackend3D backend, std::uint64_t first_frame, std::uint64_t last_frame) {
  if (first_frame > last_frame) {
    return absl::InvalidArgumentError("first frame must precede or equal last frame");
  }
  for (auto frame = first_frame;; ++frame) {
    const auto committed = cache.ReadCommittedFrame(frame);
    if (!committed.ok()) {
      return absl::Status(committed.status().code(), "could not verify committed frame " + std::to_string(frame));
    }
    const auto& manifest = committed->manifest;
    std::cout << "{\"fixture_version\":" << kFixtureProtocolVersion << ",\"event\":\"frame\",";
    PrintIdentity(std::cout, manifest.resolved_fingerprint, manifest.execution_id, backend);
    std::cout << ',';
    PrintClock(std::cout, manifest.clock);
    std::cout << ",\"points_digest\":\"" << LiquidDigestHex3D(manifest.points.digest).data()
              << "\",\"points_byte_length\":" << manifest.points.byte_length << ",\"points\":[";
    const char* separator = "";
    for (const auto& point : committed->points.points.values()) {
      std::cout << separator << "{\"id\":" << point.id << ",\"tag\":" << point.tag << ",\"world_position\":";
      PrintVector(std::cout, point.world_position);
      std::cout << ",\"world_velocity\":";
      PrintVector(std::cout, point.world_velocity);
      std::cout << ",\"radius\":" << point.radius << ",\"volume_ratio\":" << point.volume_ratio
                << ",\"level\":" << static_cast<unsigned>(point.level) << '}';
      separator = ",";
    }
    std::cout << "]}\n";
    if (frame == last_frame) {
      return absl::OkStatus();
    }
  }
}

absl::Status ReadCheckpoint(const LiquidCache3D& cache, LiquidBackend3D backend, std::uint64_t maximum_frame) {
  const auto checkpoint = cache.ReadCheckpoint(maximum_frame);
  if (!checkpoint.ok()) {
    return checkpoint.status();
  }
  const auto& state = checkpoint->state;
  const auto& resume = checkpoint->resume;
  const auto& capacities = resume.capacities;
  const auto state_bytes = EncodeLiquidState3D(state);
  if (!state_bytes.ok()) {
    return state_bytes.status();
  }
  std::cout << "{\"fixture_version\":" << kFixtureProtocolVersion << ",\"event\":\"checkpoint\",";
  PrintIdentity(std::cout, resume.resolved_fingerprint, resume.execution_id, backend);
  std::cout << ',';
  PrintClock(std::cout, state.clock);
  std::cout << ",\"state_digest\":\"" << LiquidDigestHex3D(LiquidContentDigest3D(state_bytes->values())).data()
            << "\",\"random_seed\":" << state.random_seed << ",\"next_particle_id\":" << state.next_particle_id
            << ",\"capacities\":[" << capacities.particles << ',' << capacities.blocks << ',' << capacities.cells << ','
            << capacities.faces << ',' << capacities.sources << "],\"particles\":[";
  const char* separator = "";
  for (const auto& particle : state.particles.values()) {
    std::cout << separator << "{\"id\":" << particle.id << ",\"tag\":" << particle.tag << ",\"world_position\":";
    PrintVector(std::cout, particle.world_position);
    std::cout << ",\"world_velocity\":";
    PrintVector(std::cout, particle.world_velocity);
    std::cout << ",\"velocity_slopes\":[";
    PrintVector(std::cout, particle.velocity_slopes.x_row);
    std::cout << ',';
    PrintVector(std::cout, particle.velocity_slopes.y_row);
    std::cout << ',';
    PrintVector(std::cout, particle.velocity_slopes.z_row);
    std::cout << "],\"side_tag\":" << particle.side_tag << ",\"mass\":" << particle.mass
              << ",\"reference_volume\":" << particle.reference_volume << ",\"volume_ratio\":" << particle.volume_ratio
              << ",\"affine_enabled\":" << (particle.affine_enabled ? "true" : "false")
              << ",\"pic_initialized\":" << (particle.pic_initialized ? "true" : "false") << '}';
    separator = ",";
  }
  std::cout << "],\"pressure_cells\":[";
  separator = "";
  for (const auto& cell : state.cells.values()) {
    const auto index = cell.cell_index;
    std::cout << separator << "{\"cell_index\":[" << index.x << ',' << index.y << ',' << index.z
              << "],\"occupancy\":" << static_cast<unsigned>(cell.occupancy)
              << ",\"pressure_pa\":" << cell.previous_pressure << '}';
    separator = ",";
  }
  std::cout << "]}\n";
  return absl::OkStatus();
}

absl::Status Run(const std::vector<fs::path>& args) {
  if (args.size() == 6 && args[1] == "write-scene") {
    return WriteScene(args[2], args[3], args[4], args[5]);
  }
  const bool read_frames = args.size() == 6 && args[1] == "read-frames";
  const bool read_checkpoint = args.size() == 5 && args[1] == "read-checkpoint";
  if (!read_frames && !read_checkpoint) {
    return absl::InvalidArgumentError(
        "expected write-scene CASE CPU_OR_CUDA SOLVER_EXECUTABLE SCENE, read-frames CACHE FINGERPRINT FIRST LAST, "
        "or read-checkpoint CACHE FINGERPRINT MAXIMUM_FRAME"
    );
  }
  const auto fingerprint = ParseFingerprint(args[3]);
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  const auto first_or_maximum = ParseFrameOrdinal(args[4]);
  if (!first_or_maximum.ok()) {
    return first_or_maximum.status();
  }
  const auto cache = LiquidCache3D::OpenForReplay(args[2], *fingerprint);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto scene = cache->ReadResolvedScene();
  if (!scene.ok()) {
    return scene.status();
  }
  if (read_checkpoint) {
    return ReadCheckpoint(*cache, scene->backend, *first_or_maximum);
  }
  const auto last_frame = ParseFrameOrdinal(args[5]);
  if (!last_frame.ok()) {
    return last_frame.status();
  }
  return ReadFrames(*cache, scene->backend, *first_or_maximum, *last_frame);
}

int Entry(const std::vector<fs::path>& args) {
  std::cout.imbue(std::locale::classic());
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  const auto status = Run(args);
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return std::cout.flush() ? 0 : 1;
}
}  // namespace
}  // namespace fluidity

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
  return fluidity::Entry({argv, argv + argc});
}
#else
int main(int argc, char** argv) {
  return fluidity::Entry({argv, argv + argc});
}
#endif
