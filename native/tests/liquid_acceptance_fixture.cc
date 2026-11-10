#include <charconv>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <numbers>
#include <optional>
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
constexpr char kDropCase[] = "oscillating-drop-l2";
constexpr double kDropPhysicalRadius = 0.125;
constexpr int kDropCellsPerRadius = 5;
constexpr double kDropCellWidth = kDropPhysicalRadius / kDropCellsPerRadius;
constexpr std::size_t kDropParticleCount = 552;
constexpr double kDropCovarianceScale =
    kDropPhysicalRadius * kDropPhysicalRadius / (5.177536231884058 * kDropCellWidth * kDropCellWidth);

struct TankAcceptanceScene {
  AxisAlignedLiquidTank3D tank;
  double cell_size;
  double reference_surface_y;
  double maximum_dt;
  bool transported;
};

std::optional<TankAcceptanceScene> TankAcceptance(const fs::path& case_name) {
  if (case_name == "hydrostatic-column") {
    return TankAcceptanceScene{
        .tank =
            {
                .start_low_x = -0.25,
                .high_x = 0.25,
                .floor_y = 0,
                .low_z = -0.25,
                .high_z = 0.25,
                .piston_velocity_x = 0,
            },
        .cell_size = 0.125,
        .reference_surface_y = 1,
        .maximum_dt = 0.01,
        .transported = false,
    };
  }
  if (case_name == "open-tank-piston" || case_name == "open-tank-piston-refined") {
    return TankAcceptanceScene{
        .tank =
            {
                .start_low_x = -0.5,
                .high_x = 0.5,
                .floor_y = 0,
                .low_z = -0.25,
                .high_z = 0.25,
                .piston_velocity_x = 0.015625,
            },
        .cell_size = 0.0625,
        .reference_surface_y = 0.5,
        .maximum_dt = case_name == "open-tank-piston" ? 0.01 : 0.005,
        .transported = true,
    };
  }
  return std::nullopt;
}

absl::StatusOr<double> EnergyBlend(const fs::path& case_name) {
  if (case_name == "free-fall-pic") {
    return 0.0;
  }
  if (case_name == "free-fall-default" || case_name == "tank-layout" || case_name == kDropCase
      || TankAcceptance(case_name).has_value()) {
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
  const bool drop = case_name == kDropCase;
  const auto tank_acceptance = TankAcceptance(case_name);
  const Vec3 initial_position = tank_layout ? kTankLayoutInitialPosition : kFreeFallInitialPosition;
  LiquidState3D state;
  std::vector<Vec3> seed_positions;
  double reference_volume = 1;
  if (tank_acceptance) {
    const auto& tank = tank_acceptance->tank;
    const double cell_size = tank_acceptance->cell_size;
    state.geometry = {{0, 0, 0}, cell_size};
    reference_volume = cell_size * cell_size * cell_size;
    // The frozen bounds are exact integer multiples of the world lattice.
    const int first_x = static_cast<int>(tank.start_low_x / cell_size);
    const int end_x = static_cast<int>(tank.high_x / cell_size);
    const int first_z = static_cast<int>(tank.low_z / cell_size);
    const int end_z = static_cast<int>(tank.high_z / cell_size);
    const int end_y = static_cast<int>(tank_acceptance->reference_surface_y / cell_size);
    for (int z = first_z; z < end_z; ++z) {
      for (int y = 0; y < end_y; ++y) {
        for (int x = first_x; x < end_x; ++x) {
          seed_positions.push_back({(x + 0.5) * cell_size, (y + 0.5) * cell_size, (z + 0.5) * cell_size});
        }
      }
    }
  } else if (drop) {
    state.geometry = {{0, 0, 0}, kDropCellWidth};
    constexpr double stretch_y = 1.03;
    const double stretch_xz = 1 / std::sqrt(stretch_y);
    constexpr double kDropVolume =
        4 * std::numbers::pi * kDropPhysicalRadius * kDropPhysicalRadius * kDropPhysicalRadius / 3;
    reference_volume = kDropVolume / kDropParticleCount;
    seed_positions.reserve(kDropParticleCount);
    // Select the unperturbed lattice sites before applying the determinant-one stretch.
    for (int z = -kDropCellsPerRadius; z < kDropCellsPerRadius; ++z) {
      for (int y = -kDropCellsPerRadius; y < kDropCellsPerRadius; ++y) {
        for (int x = -kDropCellsPerRadius; x < kDropCellsPerRadius; ++x) {
          const double world_x = (x + 0.5) * kDropCellWidth;
          const double world_y = (y + 0.5) * kDropCellWidth;
          const double world_z = (z + 0.5) * kDropCellWidth;
          const double squared_distance = world_x * world_x + world_y * world_y + world_z * world_z;
          if (squared_distance < kDropPhysicalRadius * kDropPhysicalRadius) {
            seed_positions.push_back({world_x * stretch_xz, world_y * stretch_y, world_z * stretch_xz});
          }
        }
      }
    }
    if (seed_positions.size() != kDropParticleCount) {
      return absl::InternalError("unexpected oscillating drop seed count");
    }
  } else {
    seed_positions.push_back(initial_position);
  }
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(seed_positions.size());
  if (!particles.ok()) {
    return particles.status();
  }
  const auto resized = particles->Resize(seed_positions.size());
  if (!resized.ok()) {
    return resized;
  }
  for (std::size_t index = 0; index < seed_positions.size(); ++index) {
    (*particles)[index] = {
        .id = index + 1,
        .world_position = seed_positions[index],
        .world_velocity = {},
        .velocity_slopes = {},
        .affine_enabled = true,
        .mass = 1000 * reference_volume,
        .reference_volume = reference_volume,
        .volume_ratio = 1,
        .pic_initialized = false,
    };
  }
  state.particles = std::move(*particles);
  state.next_particle_id = seed_positions.size() + 1;
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
  } else if (tank_acceptance) {
    scene.scene.obstacle = tank_acceptance->tank;
    if (tank_acceptance->transported) {
      scene.scene.field = LiquidParticleSphereUnion3D{0.9 * tank_acceptance->cell_size};
    } else {
      scene.scene.field = LiquidHorizontalPlaneSurface3D{tank_acceptance->reference_surface_y};
    }
  } else if (drop) {
    scene.scene.field = LiquidIsolatedCovarianceDrop3D{kDropPhysicalRadius, kDropCovarianceScale};
  } else {
    scene.scene.field = LiquidAnalyticSphere3D{initial_position, 1.1};
  }
  scene.scene.world_acceleration = drop ? Vec3{} : Vec3{0, -9.81, 0};
  scene.scene.liquid_density = 1000;
  scene.scene.surface_tension = drop ? 1 : 0;
  scene.options.particle_energy_blend = *energy_blend;
  scene.options.maximum_step_duration = tank_acceptance ? tank_acceptance->maximum_dt : 0.01;
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

void PrintParticles(const LiquidState3D& state) {
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
}

absl::Status ReadSeed(const fs::path& path, const LiquidDigest3D& expected_digest) {
  const auto bytes = files::ReadBytes(path, {});
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto digest = LiquidContentDigest3D(bytes->values());
  if (digest != expected_digest) {
    return absl::DataLossError("initial state bytes differ from the resolved upstream identity");
  }
  const auto state = DecodeLiquidState3D(bytes->values());
  if (!state.ok()) {
    return state.status();
  }
  std::cout << "{\"fixture_version\":" << kFixtureProtocolVersion << ",\"event\":\"seed\",";
  PrintClock(std::cout, state->clock);
  std::cout << ",\"state_digest\":\"" << LiquidDigestHex3D(digest).data() << "\",\"random_seed\":" << state->random_seed
            << ",\"next_particle_id\":" << state->next_particle_id << ",\"particles\":[";
  PrintParticles(*state);
  std::cout << "]}\n";
  return absl::OkStatus();
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
  PrintParticles(state);
  std::cout << "],\"pressure_cells\":[";
  const char* separator = "";
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

absl::Status PrintTankScene(const LiquidResolvedScene3D& resolved, const LiquidDigest3D& fingerprint) {
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&resolved.scene.obstacle);
  const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&resolved.scene.field);
  const auto* spheres = std::get_if<LiquidParticleSphereUnion3D>(&resolved.scene.field);
  if (tank == nullptr || (plane == nullptr && spheres == nullptr)) {
    return absl::FailedPreconditionError("expected a bounded acceptance tank");
  }
  const auto upstream = resolved.upstream.values();
  if (upstream.size() != 1 || upstream[0].role != LiquidUpstreamRole3D::kInitialParticles || upstream[0].ordinal != 0) {
    return absl::FailedPreconditionError("expected one initial-particle input");
  }
  std::cout << "{\"fixture_version\":" << kFixtureProtocolVersion << ",\"event\":\"resolved_tank\",";
  PrintIdentity(std::cout, fingerprint, LiquidExecutionId3D(fingerprint), resolved.backend);
  std::cout << ",\"solver_digest\":\"" << LiquidDigestHex3D(resolved.build_digest).data() << "\",\"seed_digest\":\""
            << LiquidDigestHex3D(upstream[0].digest).data() << "\",\"seed_byte_length\":" << upstream[0].byte_length
            << ",\"random_seed\":" << resolved.random_seed << ",\"cell_size_m\":" << resolved.geometry.cell_width
            << ",\"world_origin_m\":";
  PrintVector(std::cout, resolved.geometry.world_origin);
  std::cout << ",\"gravity_m_per_s2\":";
  PrintVector(std::cout, resolved.scene.world_acceleration);
  std::cout << ",\"density_kg_per_m3\":" << resolved.scene.liquid_density
            << ",\"surface_tension_n_per_m\":" << resolved.scene.surface_tension
            << ",\"ambient_gauge_pa\":" << resolved.scene.ambient_pressure
            << ",\"incompressible\":" << (std::isinf(resolved.scene.bulk_modulus) ? "true" : "false")
            << ",\"target_divergence_per_s\":" << resolved.scene.target_divergence
            << ",\"friction_coefficient\":" << resolved.scene.friction_coefficient
            << ",\"penetration_tolerance_m\":" << resolved.scene.penetration_tolerance
            << ",\"minimum_face_mass_kg\":" << resolved.options.minimum_face_mass
            << ",\"mass_kg_per_quantum\":" << resolved.options.mass_per_quantum
            << ",\"momentum_kg_m_per_s_per_quantum\":" << resolved.options.momentum_per_quantum
            << ",\"pressure_iteration_limit\":" << resolved.options.pressure_solver_options.max_iterations
            << ",\"source_count\":" << resolved.sources.size()
            << ",\"energy_blend\":" << resolved.options.particle_energy_blend
            << ",\"maximum_dt_seconds\":" << resolved.options.maximum_step_duration
            << ",\"tank_low_x_m\":" << tank->start_low_x << ",\"tank_high_x_m\":" << tank->high_x
            << ",\"tank_floor_y_m\":" << tank->floor_y << ",\"tank_low_z_m\":" << tank->low_z
            << ",\"tank_high_z_m\":" << tank->high_z << ",\"piston_velocity_x_m_per_s\":" << tank->piston_velocity_x;
  if (plane != nullptr) {
    std::cout << ",\"surface_height_y_m\":" << plane->world_surface_height_y;
  } else {
    std::cout << ",\"sphere_union_radius_m\":" << spheres->radius;
  }
  std::cout << "}\n";
  return absl::OkStatus();
}

absl::Status Run(const std::vector<fs::path>& args) {
  if (args.size() == 6 && args[1] == "write-scene") {
    return WriteScene(args[2], args[3], args[4], args[5]);
  }
  if (args.size() == 4 && args[1] == "read-seed") {
    const auto digest = ParseFingerprint(args[3]);
    if (!digest.ok()) {
      return digest.status();
    }
    return ReadSeed(args[2], *digest);
  }
  const bool read_frames = args.size() == 6 && args[1] == "read-frames";
  const bool read_checkpoint = args.size() == 5 && args[1] == "read-checkpoint";
  const bool read_tank_scene = args.size() == 4 && args[1] == "read-tank-scene";
  if (!read_frames && !read_checkpoint && !read_tank_scene) {
    return absl::InvalidArgumentError(
        "expected write-scene CASE CPU_OR_CUDA SOLVER_EXECUTABLE SCENE, read-frames CACHE FINGERPRINT FIRST LAST, "
        "read-seed FLST CONTENT_DIGEST, read-tank-scene CACHE FINGERPRINT, "
        "or read-checkpoint CACHE FINGERPRINT MAXIMUM_FRAME"
    );
  }
  const auto fingerprint = ParseFingerprint(args[3]);
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  const auto cache = LiquidCache3D::OpenForReplay(args[2], *fingerprint);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto scene = cache->ReadResolvedScene();
  if (!scene.ok()) {
    return scene.status();
  }
  if (read_tank_scene) {
    return PrintTankScene(*scene, *fingerprint);
  }
  const auto first_or_maximum = ParseFrameOrdinal(args[4]);
  if (!first_or_maximum.ok()) {
    return first_or_maximum.status();
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
