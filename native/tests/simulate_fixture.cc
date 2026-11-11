#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "../src/liquid_cache_files_3d_internal.h"
#include "fluidity/host/effects_source_3d.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/whitewater_volume_cache_3d.h"
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
  const bool whitewater = args[5] == "whitewater-effects";
  const bool initial_particle = args[5] == "initial" || args[5] == "effects" || slow_frame || whitewater;
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
  if (initial_particle) {
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
  std::cout << "{\"state_digest\":\""
            << fluidity::LiquidDigestHex3D(fluidity::LiquidContentDigest3D(bytes->values())).data()
            << "\",\"frame\":" << state.clock.frame_ordinal << ",\"seed\":" << state.random_seed
            << ",\"epoch\":" << state.clock.epoch << ",\"accepted_step\":" << state.clock.accepted_step_ordinal
            << ",\"particle_count\":" << state.particles.size() << ",\"next_particle_id\":" << state.next_particle_id
            << ",\"capacities\":[" << capacities.particles << ',' << capacities.blocks << ',' << capacities.cells << ','
            << capacities.faces << ',' << capacities.sources << ']'
            << ",\"first_tag\":" << (state.particles.size() ? state.particles[0].tag : 0)
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
int VolumeSettingsFixture(const std::filesystem::path& path) {
  const fluidity::WhitewaterVolumeSettings3D settings{
      .grid = {.origin = {-1, -1, -1}, .spacing_m = 0.25, .extent = {24, 24, 24}},
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
    return VolumeSettingsFixture(args[2]);
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
