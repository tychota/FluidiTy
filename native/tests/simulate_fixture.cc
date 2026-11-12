#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/effects_source_3d.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/whitewater_volume_cache_3d.h"
#include "graded_tank_fixture.h"
int Produce(int argc, const std::filesystem::path& path, bool digest) {
  if (argc != 3) {
    return 1;
  }
  if (digest) {
    fluidity::LiquidCacheLimits3D limits;
    limits.maximum_bytes = 1024 * 1024 * 1024;
    auto bytes = fluidity::liquid_cache_files_internal::ReadBytes(path, limits);
    if (!bytes.ok()) {
      return 2;
    }
    std::cout << fluidity::LiquidDigestHex3D(fluidity::LiquidContentDigest3D(bytes->values())).data() << '\n';
    return 0;
  }
  fluidity::LiquidResolvedSceneView3D scene;
  auto cache = fluidity::LiquidCache3D::Create(path, scene);
  if (!cache.ok()) {
    return 3;
  }
  auto fingerprint = fluidity::ResolvedSceneFingerprint(scene);
  if (!fingerprint.ok()) {
    return 4;
  }
  fluidity::LiquidResumeMetadata3D resume;
  resume.resolved_fingerprint = *fingerprint;
  resume.execution_id = fluidity::LiquidExecutionId3D(*fingerprint);
  fluidity::LiquidState3D state;
  fluidity::LiquidPointFrame3D points;
  return cache->PublishFrame(points, &state, &resume).ok() ? 0 : 5;
}

// Quarter-scale gravity-wave control with a single stationary planar beach.
// The reference shot and its resolution remain separate acceptance fixtures.
absl::Status SetBoundedWaveScene(
    fluidity::LiquidResolvedSceneView3D& scene,
    fluidity::LiquidState3D& state,
    bool fill_partial_columns
) {
  constexpr double kSpacingM = 0.25;
  constexpr double kBeachSlope = 0.15;
  constexpr double kStillDepthM = 0.75;
  const double normal_length = std::hypot(kBeachSlope, 1.0);
  scene.scene.world_acceleration = {0, -9.81, 0};
  scene.scene.field = fluidity::LiquidParticleSphereUnion3D{0.22};
  scene.scene.obstacle = fluidity::TranslatingHalfspace3D{
      .unit_normal = {-kBeachSlope / normal_length, 1 / normal_length, 0},
      .start_plane_coordinate = 0,
      .wall_velocity = {},
  };
  scene.options.maximum_step_duration = 0.01;
  scene.options.motion_sample_intervals = 4;
  scene.options.whitewater_capture = fluidity::WhitewaterCaptureSettings3D{
      .source_options =
          {
              .spacing_m = 0.25,
              .radius_m = 0.125,
              .kinetic = {0.01, 0.25},
              .trapped_air = {0, 1},
              .crest = {0, 1},
              .trapped_air_weight = 0,
              .crest_weight = 1,
          },
  };
  scene.options.ocean_boundary = fluidity::OceanBoundarySettings3D{
      .swell =
          {
              .still_water_height_m = kStillDepthM,
              .water_depth_m = kStillDepthM,
              .wave_height_m = 0.25,
              .wave_period_s = 2,
          },
      .inlet_wall_x_m = 0,
      .inlet_interior_x_m = 0.75,
      .outlet_interior_x_m = 3.5,
      .outlet_wall_x_m = 5,
      .inlet_rate_per_s = 5,
      .outlet_rate_per_s = 2,
      .inlet_particle_spacing_m = kSpacingM,
      .inlet_low_z_m = 0,
      .inlet_high_z_m = 1,
  };
  const auto boundary = fluidity::ResolvedOceanBoundary3D::Resolve(*scene.options.ocean_boundary);
  if (!boundary.ok()) {
    return boundary.status();
  }
  scene.geometry.cell_width = kSpacingM;
  state.geometry = scene.geometry;
  auto particles = fluidity::LiquidRecords3D<fluidity::LiquidParticleState3D>::Create(20 * 4 * 4);
  if (!particles.ok()) {
    return particles.status();
  }
  constexpr double kBedClearanceM = 0.03;
  struct Column {
    double lower_y_m;
    double layer_depth_m;
    int layer_count;
  };
  std::array<Column, 20> columns{};
  columns.fill({.lower_y_m = 0, .layer_depth_m = kSpacingM, .layer_count = 4});
  if (fill_partial_columns) {
    for (std::size_t x = 0; x < columns.size(); ++x) {
      const double world_x_m = (x + 0.5) * kSpacingM;
      const double bed_m = kBeachSlope * world_x_m + kBedClearanceM;
      const auto surface = boundary->SampleInlet({world_x_m, bed_m, 0}, 0);
      if (!surface.ok()) {
        return surface.status();
      }
      const double water_depth_m = surface->surface_height_m - bed_m;
      auto& column = columns[x];
      if (water_depth_m < kBedClearanceM) {
        column.layer_count = 0;
        continue;
      }
      column.lower_y_m = bed_m;
      column.layer_count = static_cast<int>(std::ceil(water_depth_m / kSpacingM));
      column.layer_depth_m = water_depth_m / column.layer_count;
    }
  }
  // Preserve the original z/y/x ID order for both layouts.
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < 4; ++y) {
      for (std::size_t x = 0; x < columns.size(); ++x) {
        const auto& column = columns[x];
        if (y >= column.layer_count) {
          continue;
        }
        const fluidity::Vec3 position{
            (x + 0.5) * kSpacingM,
            column.lower_y_m + (y + 0.5) * column.layer_depth_m,
            (z + 0.5) * kSpacingM,
        };
        const auto sample = boundary->SampleInlet(position, 0);
        if (!sample.ok()) {
          return sample.status();
        }
        if (position.y <= kBeachSlope * position.x + kBedClearanceM || position.y >= sample->surface_height_m) {
          continue;
        }
        const auto index = particles->size();
        const auto resized = particles->Resize(index + 1);
        if (!resized.ok()) {
          return resized;
        }
        const double particle_volume_m3 = kSpacingM * kSpacingM * column.layer_depth_m;
        (*particles)[index] = {
            .id = state.next_particle_id++,
            .world_position = position,
            .world_velocity = sample->world_velocity_m_per_s,
            .mass = scene.scene.liquid_density * particle_volume_m3,
            .reference_volume = particle_volume_m3,
        };
      }
    }
  }
  state.particles = std::move(*particles);
  return absl::OkStatus();
}

int SceneFixture(const std::vector<std::filesystem::path>& args) {
  if (args.size() != 6) {
    return 1;
  }
  auto executable = fluidity::liquid_cache_files_internal::ReadBytes(args[3], {1024 * 1024 * 1024});
  if (!executable.ok()) {
    return 2;
  }
  fluidity::LiquidResolvedSceneView3D scene;
  scene.build_digest = fluidity::LiquidContentDigest3D(executable->values());
  scene.backend = args[4] == "cuda" ? fluidity::LiquidBackend3D::kCuda : fluidity::LiquidBackend3D::kCpu;
  scene.random_seed = args[5] == "defaults" ? 0 : 123;
  if (args[5] != "defaults") {
    scene.scene.world_acceleration = {};
  }
  const bool slow_frame = args[5] == "slow";
  const bool whitewater_beach = args[5] == "whitewater-beach";
  const bool whitewater = args[5] == "whitewater-effects" || whitewater_beach;
  const bool ocean_level = args[5] == "ocean-level";
  const bool ocean = args[5] == "ocean" || ocean_level;
  const bool wave_column_seed = args[5] == "bounded-wave-column-seed";
  const bool bounded_wave = args[5] == "bounded-wave" || wave_column_seed;
  const bool graded_tank = args[5] == "graded-tank";
  const bool initial_particle =
      graded_tank || args[5] == "initial" || args[5] == "effects" || slow_frame || whitewater || ocean || bounded_wave;
  if (ocean) {
    scene.options.ocean_boundary = fluidity::OceanBoundarySettings3D{
        .swell =
            {
                .still_water_height_m = 3,
                .water_depth_m = 3,
                .wave_height_m = 1,
                .wave_period_s = 4,
            },
        .inlet_wall_x_m = 0,
        .inlet_interior_x_m = 4,
        .outlet_interior_x_m = 5,
        .outlet_wall_x_m = 8,
        .inlet_rate_per_s = 5,
        .outlet_rate_per_s = 4,
    };
  }
  scene.options.motion_sample_intervals = whitewater ? 1 : (args[5] == "effects" ? 16 : 0);
  if (whitewater) {
    scene.options.whitewater_capture = fluidity::WhitewaterCaptureSettings3D{
        .source_options =
            {
                .spacing_m = 0.5,
                .radius_m = 0.5,
                .kinetic = {0, 1},
                .trapped_air = {0, 1},
                .crest = {0, 1},
                .trapped_air_weight = 0,
                .crest_weight = 1,
            },
    };
  }
  if (ocean_level) {
    auto& settings = *scene.options.ocean_boundary;
    settings.swell.wave_height_m = 0;
    settings.inlet_wall_x_m = 2;
    settings.inlet_interior_x_m = 3;
    settings.inlet_particle_spacing_m = 1;
    settings.inlet_low_z_m = 2;
    settings.inlet_high_z_m = 3;
    scene.scene.liquid_density = 1;
    scene.scene.field = fluidity::LiquidParticleSphereUnion3D{1.1};
  } else if (whitewater_beach) {
    scene.scene.field = fluidity::LiquidParticleSphereUnion3D{1.1};
    scene.scene.obstacle = fluidity::TranslatingHalfspace3D{
        .unit_normal = {-0.6, 0.8, 0},
        .start_plane_coordinate = 0.125,
        .wall_velocity = {},
    };
  } else if (initial_particle) {
    scene.scene.field = fluidity::LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
  }
  if (args[5] != "defaults") {
    scene.options.maximum_step_duration = slow_frame ? 1e-6 : 0.03;
  }
  fluidity::LiquidState3D state;
  state.random_seed = scene.random_seed;
  fluidity::LiquidUpstreamContent3D upstream;
  fluidity::LiquidEmissionSource3D source;
  const bool exhaust_particle_capacity = args[5] == "capacity-failure";
  if (args[5] == "source" || args[5] == "boundary" || exhaust_particle_capacity) {
    source.source_ordinal = 7;
    // Avoid an integer emission boundary at the rounded binary 1/24 time.
    source.particles_per_second = args[5] == "boundary" ? 24 : 25;
    if (exhaust_particle_capacity) {
      source.particles_per_second = 100000;
    }
    source.world_position = {2.5, 2.5, 2.5};
    scene.scene.sources = std::span(&source, 1);
  }
  if (initial_particle) {
    if (graded_tank) {
      const auto authored = fluidity::test::PopulateGradedTankFixture3D(state, scene.scene);
      if (!authored.ok()) {
        std::cerr << authored << '\n';
        return 3;
      }
      scene.geometry = state.geometry;
    } else if (bounded_wave) {
      const auto authored = SetBoundedWaveScene(scene, state, wave_column_seed);
      if (!authored.ok()) {
        std::cerr << authored << '\n';
        return 3;
      }
    } else {
      state.particles = *fluidity::LiquidRecords3D<fluidity::LiquidParticleState3D>::Create(1);
      if (!state.particles.Resize(1).ok()) {
        return 3;
      }
      state.particles[0].id = 1;
      state.particles[0].tag = 19;
      state.particles[0].world_position = {2.5, 2.5, 2.5};
      if (whitewater) {
        state.particles[0].world_velocity.x = 2;
      }
      state.next_particle_id = 2;
    }
    auto initial = fluidity::EncodeLiquidState3D(state);
    if (!initial.ok()) {
      return 4;
    }
    auto initial_path = args[2];
    initial_path += ".flst";
    std::ofstream stream(initial_path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(initial->values().data()), initial->size());
    if (!stream) {
      return 5;
    }
    upstream.byte_length = initial->size();
    upstream.digest = fluidity::LiquidContentDigest3D(initial->values());
    scene.upstream = std::span(&upstream, 1);
  }
  if (args[1] == "wrong-clock" || args[1] == "growth-cache") {
    auto cache = fluidity::LiquidCache3D::Create(args[2], scene);
    if (!cache.ok()) {
      return 8;
    }
    auto fingerprint = fluidity::ResolvedSceneFingerprint(scene);
    if (!fingerprint.ok()) {
      return 9;
    }
    fluidity::LiquidResumeMetadata3D resume;
    resume.resolved_fingerprint = *fingerprint;
    resume.execution_id = fluidity::LiquidExecutionId3D(*fingerprint);
    fluidity::LiquidPointFrame3D points;
    if (args[1] == "growth-cache") {
      state.sources = *fluidity::LiquidRecords3D<fluidity::LiquidSourceState3D>::Create(1);
      if (!state.sources.Resize(1).ok()) {
        return 12;
      }
      state.sources[0].source_ordinal = source.source_ordinal;
      resume.capacities = {1, 128, 4096, 24576, 1};
      resume.capacity_replay_reservations = {16, 128, 4096, 24576, 1};
    }
    if (!cache->PublishFrame(points, &state, &resume).ok()) {
      return 10;
    }
    if (args[1] == "growth-cache") {
      return 0;
    }
    state.clock = {0.4, 10, 0, 10, 10};
    points.clock = state.clock;
    return cache->PublishFrame(points, &state, &resume).ok() ? 0 : 11;
  }
  auto bytes = fluidity::EncodeLiquidResolvedScene3D(scene);
  if (!bytes.ok()) {
    return 6;
  }
  std::ofstream stream(args[2], std::ios::binary);
  stream.write(reinterpret_cast<const char*>(bytes->values().data()), bytes->size());
  return stream ? 0 : 7;
}
int CheckpointFixture(const std::filesystem::path& path) {
  auto inspection = fluidity::LiquidCache3D::OpenForInspection(path);
  if (!inspection.ok()) {
    return 1;
  }
  auto scene = inspection->ReadResolvedScene();
  if (!scene.ok()) {
    return 2;
  }
  auto fingerprint = fluidity::ResolvedSceneFingerprint(scene->View());
  if (!fingerprint.ok()) {
    return 3;
  }
  auto cache = fluidity::LiquidCache3D::Open(path, *fingerprint);
  if (!cache.ok()) {
    return 4;
  }
  auto checkpoint = cache->ReadCheckpoint();
  if (!checkpoint.ok()) {
    return 5;
  }
  auto bytes = fluidity::EncodeLiquidState3D(checkpoint->state);
  if (!bytes.ok()) {
    return 6;
  }
  const auto& state = checkpoint->state;
  const auto& capacities = checkpoint->resume.capacities;
  const auto position = state.particles.size() ? state.particles[0].world_position : fluidity::Vec3{};
  const auto velocity = state.particles.size() ? state.particles[0].world_velocity : fluidity::Vec3{};
  std::cout.precision(17);
  std::cout << "{\"state_digest\":\""
            << fluidity::LiquidDigestHex3D(fluidity::LiquidContentDigest3D(bytes->values())).data()
            << "\",\"frame\":" << state.clock.frame_ordinal << ",\"seed\":" << state.random_seed
            << ",\"epoch\":" << state.clock.epoch << ",\"accepted_step\":" << state.clock.accepted_step_ordinal
            << ",\"particle_count\":" << state.particles.size() << ",\"next_particle_id\":" << state.next_particle_id
            << ",\"capacities\":[" << capacities.particles << ',' << capacities.blocks << ',' << capacities.cells << ','
            << capacities.faces << ',' << capacities.sources << ']'
            << ",\"first_tag\":" << (state.particles.size() ? state.particles[0].tag : 0) << ",\"first_position\":["
            << position.x << ',' << position.y << ',' << position.z << ']' << ",\"first_velocity\":[" << velocity.x
            << ',' << velocity.y << ',' << velocity.z << ']'
            << ",\"capacity_replay_particles\":" << checkpoint->resume.capacity_replay_reservations.particles
            << ",\"source_next_child\":" << (state.sources.size() ? state.sources[0].next_child_ordinal : 0) << "}\n";
  return 0;
}
int SourceBirthFixture(const std::filesystem::path& path) {
  const auto source = fluidity::EffectsSource3D::Open(path);
  if (!source.ok()) {
    return 2;
  }
  const auto frame = source->ReadFrame(1);
  if (!frame.ok()) {
    return 3;
  }
  const auto& births = frame->motion.whitewater_births;
  std::cout << "{\"birth_count\":" << births.size() << ",\"sample_count\":" << frame->motion.samples.size()
            << ",\"first_birth_step\":" << (births.empty() ? 0 : births.front().accepted_step_ordinal)
            << ",\"last_accepted_step\":" << frame->motion.last_clock.accepted_step_ordinal << "}\n";
  return 0;
}
int VolumeSettingsFixture(const std::filesystem::path& path, bool bounded_wave) {
  const fluidity::WhitewaterVolumeSettings3D settings{
      .grid = bounded_wave
          ? fluidity::WhitewaterVolumeGrid3D{.origin = {-4, -3, -3}, .spacing_m = 0.25, .extent = {52, 32, 36}}
          : fluidity::WhitewaterVolumeGrid3D{.origin = {-1, -1, -1}, .spacing_m = 0.25, .extent = {24, 24, 24}},
      .physical_bubble_radius_m = 0.05,
  };
  const auto bytes = fluidity::EncodeWhitewaterVolumeSettings3D(settings);
  if (!bytes.ok()) {
    return 2;
  }
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(bytes->values().data()), bytes->size());
  return output ? 0 : 3;
}
int Entry(const std::vector<std::filesystem::path>& args) {
  if (args.size() == 3 && args[1] == "volume-settings") {
    return VolumeSettingsFixture(args[2], false);
  }
  if (args.size() == 3 && args[1] == "wave-volume-settings") {
    return VolumeSettingsFixture(args[2], true);
  }
  if (args.size() == 3 && args[1] == "source-births") {
    return SourceBirthFixture(args[2]);
  }
  if (args.size() == 3 && args[1] == "checkpoint") {
    return CheckpointFixture(args[2]);
  }
  if (args.size() >= 2 && (args[1] == "scene" || args[1] == "wrong-clock" || args[1] == "growth-cache")) {
    return SceneFixture(args);
  }
  return args.size() == 3 ? Produce(static_cast<int>(args.size()), args[2], args[1] == "digest") : 1;
}
#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
  return Entry({argv, argv + argc});
}
#else
int main(int argc, char** argv) {
  return Entry({argv, argv + argc});
}
#endif
