#include <chrono>

#include "fluidity/host/effects_source_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
struct SourceFixture {
  std::filesystem::path root;
  LiquidResolvedSceneView3D scene;
  LiquidState3D state;
  LiquidResumeMetadata3D resume;
  EffectsIdentity3D identity;
  SourceFixture() {
    root = std::filesystem::path(FLUIDITY_CACHE_SCRATCH_PATH)
        / ("effects-source-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    scene.random_seed = 7;
    scene.build_digest[0] = 3;
    scene.options.motion_sample_intervals = 16;
    state.random_seed = 7;
    state.particles = *LiquidRecords3D<LiquidParticleState3D>::Create(1);
    EXPECT_TRUE(state.particles.Resize(1).ok());
    state.particles[0].id = 1;
    state.particles[0].world_position = {2.5, 2.5, 2.5};
    state.next_particle_id = 2;
    const auto fingerprint = ResolvedSceneFingerprint(scene);
    EXPECT_TRUE(fingerprint.ok());
    resume.resolved_fingerprint = *fingerprint;
    resume.execution_id = LiquidExecutionId3D(*fingerprint);
    resume.capacities = {16, 64, 128, 12288, 1};
    identity = {
        .stage = EffectsStage3D::kSource,
        .upstream_execution = resume.execution_id,
        .settings = *fingerprint,
        .build = scene.build_digest
    };
  }
  absl::Status PublishParent(const LiquidCache3D& parent) const {
    const auto points = BuildLiquidPointFrame3D(state);
    if (!points.ok()) {
      return points.status();
    }
    return parent.PublishFrame(*points, &state, &resume);
  }
  absl::Status PublishSource(const EffectsCache3D& source, const LiquidState3D& captured) const {
    const auto points = BuildLiquidPointFrame3D(state);
    if (!points.ok()) {
      return points.status();
    }
    const auto encoded_points = EncodeLiquidPointFrame3D(*points);
    const auto encoded_source = EncodeMotionSeed3D(captured);
    if (!encoded_points.ok() || !encoded_source.ok()) {
      return absl::InternalError("could not encode source fixture");
    }
    return source.Publish({}, LiquidContentDigest3D(encoded_points->values()), {}, encoded_source->values());
  }
};

TEST(EffectsSource3D, RequiresCommittedParentAndBindsExactSceneAndSourceIdentity) {
  SourceFixture fixture;
  auto parent = LiquidCache3D::Create(fixture.root, fixture.scene);
  ASSERT_TRUE(parent.ok());
  auto source = EffectsCache3D::Create(fixture.root / "effects/source", fixture.identity);
  ASSERT_TRUE(source.ok());
  ASSERT_TRUE(fixture.PublishSource(*source, fixture.state).ok());
  auto reader = EffectsSource3D::Open(fixture.root);
  ASSERT_TRUE(reader.ok()) << reader.status();
  EXPECT_EQ(reader->ReadFrame(0).status().code(), absl::StatusCode::kNotFound);
  ASSERT_TRUE(fixture.PublishParent(*parent).ok());
  auto frame = reader->ReadFrame(0);
  ASSERT_TRUE(frame.ok()) << frame.status();
  EXPECT_EQ(frame->motion.samples.size(), 1);
  EXPECT_EQ(frame->motion.samples.front().particles[0].id, 1);
  EXPECT_EQ(reader->execution_id(), EffectsExecutionId3D(fixture.identity).value());
  EXPECT_EQ(reader->scene().options.motion_sample_intervals, 16);
}

TEST(EffectsSource3D, RejectsRehashedSourceEndpointThatDiffersFromParentPoints) {
  SourceFixture fixture;
  auto parent = LiquidCache3D::Create(fixture.root, fixture.scene);
  ASSERT_TRUE(parent.ok());
  ASSERT_TRUE(fixture.PublishParent(*parent).ok());
  auto source = EffectsCache3D::Create(fixture.root / "effects/source", fixture.identity);
  ASSERT_TRUE(source.ok());
  auto changed = CloneLiquidState3D(fixture.state, fixture.resume.capacities);
  ASSERT_TRUE(changed.ok());
  changed->particles[0].world_position.x += 0.1;
  ASSERT_TRUE(fixture.PublishSource(*source, *changed).ok());
  auto reader = EffectsSource3D::Open(fixture.root);
  ASSERT_TRUE(reader.ok());
  EXPECT_EQ(reader->ReadFrame(0).status().code(), absl::StatusCode::kDataLoss);
}

TEST(EffectsSource3D, RejectsForeignSourceSettings) {
  SourceFixture fixture;
  auto parent = LiquidCache3D::Create(fixture.root, fixture.scene);
  ASSERT_TRUE(parent.ok());
  ++fixture.identity.settings[0];
  auto source = EffectsCache3D::Create(fixture.root / "effects/source", fixture.identity);
  ASSERT_TRUE(source.ok());
  EXPECT_EQ(EffectsSource3D::Open(fixture.root).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(EffectsSource3D, RejectsIntervalDiscontinuityBeyondExportedPointFields) {
  SourceFixture fixture;
  auto parent = LiquidCache3D::Create(fixture.root, fixture.scene);
  ASSERT_TRUE(parent.ok());
  ASSERT_TRUE(fixture.PublishParent(*parent).ok());
  auto source = EffectsCache3D::Create(fixture.root / "effects/source", fixture.identity);
  ASSERT_TRUE(source.ok());
  ASSERT_TRUE(fixture.PublishSource(*source, fixture.state).ok());
  const auto seed = source->ReadFrame(0);
  ASSERT_TRUE(seed.ok());
  auto wrong_start = CloneLiquidState3D(fixture.state, fixture.resume.capacities);
  ASSERT_TRUE(wrong_start.ok());
  ASSERT_TRUE(wrong_start->blocks.Resize(1).ok());
  ASSERT_TRUE(wrong_start->cells.Resize(1).ok());
  wrong_start->cells[0].previous_pressure = 2;
  auto capture = MotionCapture3D::Create(*wrong_start, 1.0 / 24, 1);
  ASSERT_TRUE(capture.ok()) << capture.status();
  auto end = CloneLiquidState3D(fixture.state, fixture.resume.capacities);
  ASSERT_TRUE(end.ok());
  end->clock = {.time = 1.0 / 24, .frame_ordinal = 1, .accepted_step_ordinal = 1, .epoch = 1};
  ASSERT_TRUE(capture->Observe(*end, false).ok());
  auto interval = capture->Finish(*end);
  ASSERT_TRUE(interval.ok());
  auto points = BuildLiquidPointFrame3D(*end);
  ASSERT_TRUE(points.ok());
  ASSERT_TRUE(parent->PublishFrame(*points).ok());
  auto encoded = EncodeLiquidPointFrame3D(*points);
  ASSERT_TRUE(encoded.ok());
  ASSERT_TRUE(
      source->Publish(end->clock, LiquidContentDigest3D(encoded->values()), seed->payload_digest, interval->values())
          .ok()
  );
  auto reader = EffectsSource3D::Open(fixture.root);
  ASSERT_TRUE(reader.ok());
  EXPECT_EQ(reader->ReadFrame(1).status().code(), absl::StatusCode::kDataLoss);
}
}  // namespace
}  // namespace fluidity
