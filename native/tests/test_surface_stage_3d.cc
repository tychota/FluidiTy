#include <chrono>
#include <fstream>

#include "fluidity/host/surface_stage_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
// A real parent/source join with a stationary, unit-volume material particle.
struct SurfaceStageFixture {
  std::filesystem::path root;
  LiquidResolvedSceneView3D scene;
  LiquidState3D state;
  LiquidDigest3D build{};
  SurfaceReconstructionOptions3D options{.spacing_m = 0.125, .sigma_m = 0.25};
  SurfaceStageFixture() {
    root = std::filesystem::path(FLUIDITY_CACHE_SCRATCH_PATH)
        / ("surface-stage-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    build[0] = 9;
    scene.build_digest[0] = 3;
    scene.options.motion_sample_intervals = 16;
    scene.scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
    scene.scene.world_acceleration = {};
    scene.options.capacities = {16, 64, 128, 12288, 1};
    state.particles = *LiquidRecords3D<LiquidParticleState3D>::Create(1);
    EXPECT_TRUE(state.particles.Resize(1).ok());
    state.particles[0].id = 1;
    state.particles[0].world_position = {2.5, 2.5, 2.5};
    state.next_particle_id = 2;
  }
  void PublishSource() {
    auto parent = LiquidCache3D::Create(root / "simulate", scene);
    ASSERT_TRUE(parent.ok()) << parent.status();
    auto fingerprint = ResolvedSceneFingerprint(scene);
    ASSERT_TRUE(fingerprint.ok());
    auto source = EffectsCache3D::Create(
        root / "simulate/effects/source",
        {
            .stage = EffectsStage3D::kSource,
            .upstream_execution = LiquidExecutionId3D(*fingerprint),
            .settings = *fingerprint,
            .build = scene.build_digest,
        }
    );
    ASSERT_TRUE(source.ok());
    auto points = BuildLiquidPointFrame3D(state);
    ASSERT_TRUE(points.ok());
    const LiquidResumeMetadata3D resume{
        .resolved_fingerprint = *fingerprint,
        .execution_id = LiquidExecutionId3D(*fingerprint),
        .capacities = scene.options.capacities,
    };
    const auto published_seed = parent->PublishFrame(*points, &state, &resume);
    ASSERT_TRUE(published_seed.ok()) << published_seed;
    auto point_bytes = EncodeLiquidPointFrame3D(*points);
    auto seed = EncodeMotionSeed3D(state);
    ASSERT_TRUE(point_bytes.ok());
    ASSERT_TRUE(seed.ok());
    ASSERT_TRUE(source->Publish({}, LiquidContentDigest3D(point_bytes->values()), {}, seed->values()).ok());
    const auto previous_payload = LiquidContentDigest3D(seed->values());
    auto capture = MotionCapture3D::Create(state, 1.0 / 24, 16);
    auto scheduler = LiquidSubstepScheduler3D::Create(state, scene.scene, scene.options);
    ASSERT_TRUE(capture.ok());
    ASSERT_TRUE(scheduler.ok());
    const LiquidProgressObserver3D observer{
        .context = &*capture,
        .observe =
            [](void* context, const LiquidState3D& accepted, const LiquidStepDiagnostics3D&, bool rollback) {
              return static_cast<MotionCapture3D*>(context)->Observe(accepted, rollback);
            },
    };
    ASSERT_TRUE(scheduler->AdvanceFrame(1.0 / 24, observer).ok());
    const auto& endpoint = scheduler->committed();
    auto interval = capture->Finish(endpoint);
    auto end_points = BuildLiquidPointFrame3D(endpoint);
    ASSERT_TRUE(interval.ok());
    ASSERT_TRUE(end_points.ok());
    ASSERT_TRUE(parent->PublishFrame(*end_points).ok());
    auto end_bytes = EncodeLiquidPointFrame3D(*end_points);
    ASSERT_TRUE(end_bytes.ok());
    ASSERT_TRUE(
        source
            ->Publish(endpoint.clock, LiquidContentDigest3D(end_bytes->values()), previous_payload, interval->values())
            .ok()
    );
  }
};

TEST(SurfaceStage3D, PublishesBoundFramesAndReplaysWithoutChangingManifestTimes) {
  SurfaceStageFixture fixture;
  fixture.PublishSource();
  auto stage = SurfaceStage3D::OpenForPublication(
      fixture.root / "simulate",
      fixture.root / "mesh",
      fixture.build,
      fixture.options
  );
  ASSERT_TRUE(stage.ok()) << stage.status();
  auto seed = stage->ProcessNextFrame();
  ASSERT_TRUE(seed.ok()) << seed.status();
  EXPECT_EQ(seed->reconstruction.frame.clock.frame_ordinal, 0);
  auto next = stage->ProcessNextFrame();
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_TRUE(next->reconstruction.used_tracker);
  auto stored = EffectsCache3D::OpenForInspection(fixture.root / "mesh");
  ASSERT_TRUE(stored.ok());
  const auto committed = stored->ReadFrame(1);
  ASSERT_TRUE(committed.ok());
  EXPECT_EQ(committed->previous_payload, seed->payload_digest);
  EXPECT_EQ(committed->payload_digest, next->payload_digest);
  const auto manifest = fixture.root / "mesh/frames/00000000000000000001.flem";
  const auto timestamp = std::filesystem::last_write_time(manifest);
  auto replay = SurfaceStage3D::OpenForReplay(fixture.root / "simulate", fixture.root / "mesh", fixture.build);
  ASSERT_TRUE(replay.ok()) << replay.status();
  EXPECT_EQ(replay->ProcessNextFrame()->payload_digest, seed->payload_digest);
  EXPECT_EQ(replay->ProcessNextFrame()->payload_digest, next->payload_digest);
  EXPECT_EQ(std::filesystem::last_write_time(manifest), timestamp);
  EXPECT_EQ(replay->ProcessNextFrame().status().code(), absl::StatusCode::kNotFound);
}

TEST(SurfaceStage3D, RejectsForeignBuildSettingsAndCorruptionWithoutAdvancingCursor) {
  SurfaceStageFixture fixture;
  fixture.PublishSource();
  auto stage = SurfaceStage3D::OpenForPublication(
      fixture.root / "simulate",
      fixture.root / "mesh",
      fixture.build,
      fixture.options
  );
  ASSERT_TRUE(stage.ok());
  auto seed = stage->ProcessNextFrame();
  ASSERT_TRUE(seed.ok());
  ++fixture.build[0];
  EXPECT_EQ(
      SurfaceStage3D::OpenForReplay(fixture.root / "simulate", fixture.root / "mesh", fixture.build).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  --fixture.build[0];
  fixture.options.sigma_m *= 2;
  EXPECT_EQ(
      SurfaceStage3D::OpenForPublication(
          fixture.root / "simulate",
          fixture.root / "mesh",
          fixture.build,
          fixture.options
      )
          .status()
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
  auto replay = SurfaceStage3D::OpenForReplay(fixture.root / "simulate", fixture.root / "mesh", fixture.build);
  ASSERT_TRUE(replay.ok());
  const auto blob =
      fixture.root / "mesh/blobs" / (std::string(LiquidDigestHex3D(seed->payload_digest).data()) + ".flef");
  {
    std::ofstream corrupt(blob, std::ios::binary | std::ios::trunc);
    corrupt << "corrupt";
  }
  EXPECT_EQ(replay->ProcessNextFrame().status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(replay->next_frame(), 0);
}

TEST(SurfaceStage3D, RejectsOutputRootsThatOverlapSimulateOwnedData) {
  SurfaceStageFixture fixture;
  fixture.PublishSource();
  for (const auto& output : {fixture.root / "simulate", fixture.root, fixture.root / "simulate/frames/mesh"}) {
    SCOPED_TRACE(output.string());
    EXPECT_EQ(
        SurfaceStage3D::OpenForPublication(fixture.root / "simulate", output, fixture.build, fixture.options)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument
    );
  }
  EXPECT_FALSE(std::filesystem::exists(fixture.root / "simulate/identity.flei"));
}

TEST(SurfaceStage3D, RetriesCancellationBeforeAndAfterManifestWithoutLosingCheckpoint) {
  SurfaceStageFixture fixture;
  fixture.PublishSource();
  auto stage = SurfaceStage3D::OpenForPublication(
      fixture.root / "simulate",
      fixture.root / "mesh",
      fixture.build,
      fixture.options
  );
  ASSERT_TRUE(stage.ok());
  LiquidCachePublishStage3D stop_at = LiquidCachePublishStage3D::kBeforeManifest;
  const LiquidCachePublishControl3D control{
      .check =
          [](void* context, LiquidCachePublishStage3D current) {
            return current == *static_cast<LiquidCachePublishStage3D*>(context) ? absl::CancelledError("injected")
                                                                                : absl::OkStatus();
          },
      .context = &stop_at,
  };
  EXPECT_EQ(stage->ProcessNextFrame(control).status().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(stage->next_frame(), 0);
  auto inspection = EffectsCache3D::OpenForInspection(fixture.root / "mesh");
  ASSERT_TRUE(inspection.ok());
  EXPECT_EQ(inspection->ReadFrame(0).status().code(), absl::StatusCode::kNotFound);
  ASSERT_TRUE(stage->ProcessNextFrame().ok());
  stop_at = LiquidCachePublishStage3D::kManifestPublished;
  EXPECT_EQ(stage->ProcessNextFrame(control).status().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(stage->next_frame(), 1);
  auto committed = inspection->ReadFrame(1);
  ASSERT_TRUE(committed.ok());
  auto retried = stage->ProcessNextFrame();
  ASSERT_TRUE(retried.ok()) << retried.status();
  EXPECT_EQ(retried->payload_digest, committed->payload_digest);
  EXPECT_EQ(stage->next_frame(), 2);
}
}  // namespace
}  // namespace fluidity
