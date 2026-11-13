#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#include "fluidity/host/liquid_substep_backend_3d.h"
#include "fluidity/host/motion_carrier_3d.h"
#include "fluidity/host/surface_stage_3d.h"
#include "graded_tank_fixture.h"
#include "gtest/gtest.h"

#ifdef FLUIDITY_LIVE_GRADED_MESH_CUDA
#include "fluidity/host/cuda_liquid_substep_3d.h"
#endif

namespace fluidity {
namespace {

using CacheSnapshot = std::map<std::filesystem::path, std::pair<LiquidDigest3D, std::filesystem::file_time_type>>;

// Capture both content and write times for all source and derived cache files.
CacheSnapshot Snapshot(const std::filesystem::path& root) {
  CacheSnapshot snapshot;
  for (const auto& file : std::filesystem::recursive_directory_iterator(root)) {
    if (file.is_regular_file()) {
      std::ifstream stream(file.path(), std::ios::binary);
      EXPECT_TRUE(stream.good());
      const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(stream), {}};
      snapshot.emplace(
          file.path().lexically_relative(root),
          std::pair{LiquidContentDigest3D(bytes), file.last_write_time()}
      );
    }
  }
  return snapshot;
}

#ifdef FLUIDITY_LIVE_GRADED_MESH_CUDA
TEST(CudaLiveGradedMesh3D, RetainsCorrespondenceAcrossAcceptedClosureAndReplaysReadOnly) {
#else
TEST(LiveGradedMesh3D, RetainsCorrespondenceAcrossAcceptedClosureAndReplaysReadOnly) {
#endif
  // Keep content-addressed blob paths within the Windows path limit even when
  // CTest runs from a deeply nested build directory.
  std::error_code error;
  const auto temporary_directory = std::filesystem::temp_directory_path(error);
  ASSERT_FALSE(error) << error.message();
  const auto root = temporary_directory
      / ("live-graded-mesh-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto simulate_root = root / "simulate";
  const auto mesh_root = root / "mesh";
  LiquidResolvedSceneView3D resolved;
  resolved.build_digest[0] = 3;
  resolved.options.motion_sample_intervals = 2;
  resolved.options.capacities = {19, 0, 19, 160, 0};
  LiquidState3D initial;
  const auto authored = test::PopulateGradedRefinementClosureFixture3D(initial, resolved.scene, true);
  ASSERT_TRUE(authored.ok()) << authored;
  ASSERT_EQ(initial.adaptive_cells.size(), 5);
  ASSERT_EQ(initial.particles.size(), 19);
  resolved.geometry = initial.geometry;

  auto operations = CpuLiquidSubstepBackend3D();
#ifdef FLUIDITY_LIVE_GRADED_MESH_CUDA
  constexpr auto kBackend = "cuda";
  resolved.backend = LiquidBackend3D::kCuda;
  auto device = CudaLiquidSubstepBackend3D::Create(resolved.options.capacities, 0, false, false, true);
  ASSERT_TRUE(device.ok()) << device.status();
  operations = device->operations();
  ASSERT_NE(operations.project_adaptive_in_arena, nullptr);
#else
  constexpr auto kBackend = "cpu";
#endif

  const auto fingerprint = ResolvedSceneFingerprint(resolved);
  ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
  const LiquidResumeMetadata3D resume{
      .resolved_fingerprint = *fingerprint,
      .execution_id = LiquidExecutionId3D(*fingerprint),
      .capacities = resolved.options.capacities,
  };
  auto parent = LiquidCache3D::Create(simulate_root, resolved);
  ASSERT_TRUE(parent.ok()) << parent.status();
  auto source = EffectsCache3D::Create(
      simulate_root / "effects/source",
      {
          .stage = EffectsStage3D::kSource,
          .upstream_execution = resume.execution_id,
          .settings = *fingerprint,
          .build = resolved.build_digest,
      }
  );
  ASSERT_TRUE(source.ok()) << source.status();
  const auto points = BuildLiquidPointFrame3D(initial);
  const auto seed = EncodeMotionSeed3D(initial);
  ASSERT_TRUE(points.ok()) << points.status();
  ASSERT_TRUE(seed.ok()) << seed.status();
  ASSERT_TRUE(parent->PublishFrame(*points, &initial, &resume).ok());
  const auto seed_points = EncodeLiquidPointFrame3D(*points);
  ASSERT_TRUE(seed_points.ok()) << seed_points.status();
  const auto seed_published = source->Publish({}, LiquidContentDigest3D(seed_points->values()), {}, seed->values());
  ASSERT_TRUE(seed_published.ok()) << seed_published;

  constexpr double kStep = 1.0 / 128.0;
  constexpr double kFrameEnd = 2 * kStep;
  constexpr std::array requested{AdaptiveMacCellKey3D{.lower_corner = {4, 0, 0}, .width_exponent = 1}};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, resolved.scene, resolved.options, operations);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  auto capture = MotionCapture3D::Create(scheduler->committed(), kFrameEnd, resolved.options.motion_sample_intervals);
  ASSERT_TRUE(capture.ok()) << capture.status();
  ASSERT_TRUE(scheduler->BeginFrame(kFrameEnd).ok());
  const auto refined = scheduler->AttemptWithAdaptiveRefinementClosure(kStep, requested);
  ASSERT_TRUE(refined.ok()) << refined.status();
  ASSERT_EQ(refined->disposition, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_EQ(scheduler->committed().adaptive_cells.size(), 19);
  const auto sampled = capture->Observe(scheduler->committed(), false);
  ASSERT_TRUE(sampled.ok()) << sampled;
  const auto remainder = scheduler->Attempt(kStep);
  ASSERT_TRUE(remainder.ok()) << remainder.status();
  ASSERT_EQ(remainder->disposition, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  const auto& endpoint = scheduler->committed();
#ifdef FLUIDITY_LIVE_GRADED_MESH_CUDA
  EXPECT_EQ(device->usage().compiled_architecture, 1200);
  const auto committed_device = device->AcquireCommitted();
  ASSERT_TRUE(committed_device.ok()) << committed_device.status();
  auto readback = CloneLiquidState3D(endpoint, resolved.options.capacities);
  ASSERT_TRUE(readback.ok()) << readback.status();
  const auto copied = committed_device->Readback(*readback);
  ASSERT_TRUE(copied.ok()) << copied;
  EXPECT_EQ(*readback, endpoint);
#endif
  const auto end_sampled = capture->Observe(endpoint, false);
  ASSERT_TRUE(end_sampled.ok()) << end_sampled;
  ASSERT_EQ(endpoint.next_particle_id, initial.next_particle_id);
  ASSERT_TRUE(std::ranges::equal(
      endpoint.particles.values(),
      initial.particles.values(),
      {},
      &LiquidParticleState3D::id,
      &LiquidParticleState3D::id
  ));
  const auto interval = capture->Finish(endpoint);
  const auto end_points = BuildLiquidPointFrame3D(endpoint);
  ASSERT_TRUE(interval.ok()) << interval.status();
  ASSERT_TRUE(end_points.ok()) << end_points.status();
  const auto end_bytes = EncodeLiquidPointFrame3D(*end_points);
  ASSERT_TRUE(end_bytes.ok()) << end_bytes.status();
  ASSERT_TRUE(parent->PublishFrame(*end_points).ok());
  const auto interval_published = source->Publish(
      endpoint.clock,
      LiquidContentDigest3D(end_bytes->values()),
      LiquidContentDigest3D(seed->values()),
      interval->values()
  );
  ASSERT_TRUE(interval_published.ok()) << interval_published;

  const auto source_before = Snapshot(simulate_root);
  const auto accepted = EffectsSource3D::Open(simulate_root);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  const auto frame = accepted->ReadFrame(1);
  ASSERT_TRUE(frame.ok()) << frame.status();
  ASSERT_EQ(frame->motion.samples.size(), 3);
  EXPECT_EQ(frame->motion.samples[0].adaptive_cells.size(), 5);
  EXPECT_EQ(frame->motion.samples[1].adaptive_cells.size(), 19);
  EXPECT_EQ(frame->motion.samples[1].clock.time, kStep);
  EXPECT_EQ(frame->motion.samples[2], endpoint);

  LiquidDigest3D mesh_build{};
  mesh_build[0] = 9;
  const SurfaceReconstructionOptions3D options{.spacing_m = 0.25, .sigma_m = 0.5};
  auto mesh = SurfaceStage3D::OpenForPublication(simulate_root, mesh_root, mesh_build, options);
  ASSERT_TRUE(mesh.ok()) << mesh.status();
  const auto first = mesh->ProcessNextFrame();
  ASSERT_TRUE(first.ok()) << first.status();

  std::size_t missing_samples = 0;
  std::size_t continued_samples = 0;
  for (const auto& sample : frame->motion.samples) {
    const auto carrier = FrozenMotionCarrier3D::Create(sample, resolved.scene, resolved.options);
    ASSERT_TRUE(carrier.ok()) << carrier.status();
    for (const auto& vertex : first->reconstruction.frame.mesh.vertices) {
      const auto sampled_velocity = carrier->Sample(vertex.position);
      if (!sampled_velocity.ok()) {
        ++missing_samples;
      } else if (sampled_velocity->continued) {
        ++continued_samples;
      }
    }
  }
  std::cout << "graded_mesh_probe_missing=" << missing_samples << " graded_mesh_probe_continued=" << continued_samples
            << '\n';
  const auto tracked = mesh->ProcessNextFrame();
  ASSERT_TRUE(tracked.ok()) << tracked.status();
  EXPECT_TRUE(tracked->reconstruction.used_tracker);
  EXPECT_EQ(missing_samples, 0);
  EXPECT_EQ(tracked->reconstruction.retained_ids, first->reconstruction.frame.mesh.vertices.size());
  ASSERT_GT(tracked->reconstruction.retained_ids, 0);
  const auto ids = [](const SurfaceFrame3D& surface) {
    std::vector<std::uint64_t> result;
    for (const auto& vertex : surface.mesh.vertices) {
      result.push_back(vertex.id);
    }
    std::ranges::sort(result);
    return result;
  };
  EXPECT_EQ(ids(first->reconstruction.frame), ids(tracked->reconstruction.frame));

  for (const auto* result : {&*first, &*tracked}) {
    ASSERT_EQ(result->reconstruction.frame.volumes.size(), 1);
    const auto& volume = result->reconstruction.frame.volumes.front();
    EXPECT_DOUBLE_EQ(volume.material_m3, 12.0);
    EXPECT_NEAR(volume.corrected_m3, volume.material_m3, 0.005 * volume.material_m3);
    EXPECT_NEAR(result->reconstruction.corrected_metrics.signed_volume, volume.material_m3, 0.005 * volume.material_m3);
  }

  const auto mesh_before = Snapshot(mesh_root);
  auto replay = SurfaceStage3D::OpenForReplay(simulate_root, mesh_root, mesh_build);
  ASSERT_TRUE(replay.ok()) << replay.status();
  for (const auto* result : {&*first, &*tracked}) {
    const auto regenerated = replay->ProcessNextFrame();
    ASSERT_TRUE(regenerated.ok()) << regenerated.status();
    EXPECT_EQ(regenerated->payload_digest, result->payload_digest);
  }
  EXPECT_EQ(replay->ProcessNextFrame().status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(Snapshot(simulate_root), source_before);
  EXPECT_EQ(Snapshot(mesh_root), mesh_before);
  std::ostringstream receipt;
  receipt << std::setprecision(17) << "graded_mesh_backend=" << kBackend
          << " graded_mesh_retained_ids=" << tracked->reconstruction.retained_ids
          << " seed_corrected_m3=" << first->reconstruction.corrected_metrics.signed_volume
          << " tracked_corrected_m3=" << tracked->reconstruction.corrected_metrics.signed_volume
          << " seed_mesh_digest=" << LiquidDigestHex3D(first->payload_digest).data()
          << " tracked_mesh_digest=" << LiquidDigestHex3D(tracked->payload_digest).data();
  std::cout << receipt.str() << '\n';
}

}  // namespace
}  // namespace fluidity
