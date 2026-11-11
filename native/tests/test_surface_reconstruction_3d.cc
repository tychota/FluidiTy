#include <cmath>

#include "fluidity/host/surface_reconstruction_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
LiquidState3D Seed() {
  LiquidState3D state;
  state.particles = *LiquidRecords3D<LiquidParticleState3D>::Create(1);
  EXPECT_TRUE(state.particles.Resize(1).ok());
  state.particles[0].id = 1;
  state.particles[0].world_position = {2.5, 2.5, 2.5};
  state.next_particle_id = 2;
  return state;
}
TEST(SurfaceReconstruction3D, ReconstructsAndEncodesMeasuredSeedThenTracksAcceptedFrame) {
  const auto seed = Seed();
  const auto encoded_seed = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(encoded_seed.ok());
  const auto motion_seed = DecodeMotionFrame3D(encoded_seed->values());
  ASSERT_TRUE(motion_seed.ok());
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
  scene.world_acceleration = {};
  LiquidStepOptions3D step;
  step.capacities = {16, 64, 128, 12288, 1};
  const SurfaceReconstructionOptions3D options{.spacing_m = 0.125, .sigma_m = 0.25};
  auto first = ReconstructSurfaceFrame3D(*motion_seed, scene, step, nullptr, options);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_FALSE(first->used_tracker);
  EXPECT_LE(std::abs(first->corrected_metrics.signed_volume - 1), 0.005);
  EXPECT_EQ(first->frame.volumes.size(), 1);
  auto bytes = EncodeSurfaceFrame3D(first->frame);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  auto checkpoint = DecodeSurfaceFrame3D(bytes->values());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(seed, scene, step);
  ASSERT_TRUE(scheduler.ok());
  for (std::uint64_t frame = 1; frame <= 24; ++frame) {
    SCOPED_TRACE(frame);
    auto capture = MotionCapture3D::Create(scheduler->committed(), static_cast<double>(frame) / 24, 16);
    ASSERT_TRUE(capture.ok());
    const LiquidProgressObserver3D observer{
        .context = &*capture,
        .observe =
            [](void* context, const LiquidState3D& state, const LiquidStepDiagnostics3D&, bool rollback) {
              return static_cast<MotionCapture3D*>(context)->Observe(state, rollback);
            },
    };
    ASSERT_TRUE(scheduler->AdvanceFrame(static_cast<double>(frame) / 24, observer).ok());
    auto captured = capture->Finish(scheduler->committed());
    ASSERT_TRUE(captured.ok());
    auto interval = DecodeMotionFrame3D(captured->values());
    ASSERT_TRUE(interval.ok());
    if (frame == 1) {
      ASSERT_GT(interval->samples.size(), 2);
      const auto clock = interval->samples[1].clock;
      interval->samples[1].clock.frame_ordinal = 1;
      interval->samples[1].clock.substep_ordinal = 0;
      EXPECT_EQ(
          ReconstructSurfaceFrame3D(*interval, scene, step, &*checkpoint, options).status().code(),
          absl::StatusCode::kInvalidArgument
      );
      interval->samples[1].clock = clock;
    }
    auto next = ReconstructSurfaceFrame3D(*interval, scene, step, &*checkpoint, options);
    ASSERT_TRUE(next.ok()) << next.status();
    EXPECT_TRUE(next->used_tracker);
    EXPECT_EQ(next->retained_ids, checkpoint->mesh.vertices.size());
    EXPECT_EQ(next->frame.mesh.triangles, checkpoint->mesh.triangles);
    EXPECT_EQ(next->frame.clock.frame_ordinal, frame);
    EXPECT_LE(std::abs(next->corrected_metrics.signed_volume - 1), 0.005);
    EXPECT_TRUE(EncodeSurfaceFrame3D(next->frame).ok());
    for (std::size_t index = 0; index < next->frame.mesh.vertices.size(); ++index) {
      const auto& vertex = next->frame.mesh.vertices[index];
      const auto old = checkpoint->mesh.vertices[index].position;
      EXPECT_EQ(vertex.id, checkpoint->mesh.vertices[index].id);
      EXPECT_EQ(vertex.position.x, old.x);
      EXPECT_EQ(vertex.position.y, old.y);
      EXPECT_EQ(vertex.position.z, old.z);
      EXPECT_NEAR(vertex.age_seconds, static_cast<double>(frame) / 24, 1e-12);
      EXPECT_NEAR(vertex.geometric_velocity.x, (vertex.position.x - old.x) * 24, 1e-12);
      EXPECT_NEAR(vertex.fluid_velocity.x, 0, 1e-12);
    }
    auto next_bytes = EncodeSurfaceFrame3D(next->frame);
    ASSERT_TRUE(next_bytes.ok());
    checkpoint = DecodeSurfaceFrame3D(next_bytes->values());
    ASSERT_TRUE(checkpoint.ok());
  }
}
TEST(SurfaceReconstruction3D, RejectsMissingSeedAndUnsupportedContactBeforePublishingAnything) {
  MotionFrame3D empty;
  EXPECT_EQ(ReconstructSurfaceFrame3D(empty, {}, {}, nullptr, {}).status().code(), absl::StatusCode::kInvalidArgument);
  const auto seed = Seed();
  const auto encoded = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(encoded.ok());
  const auto motion = DecodeMotionFrame3D(encoded->values());
  ASSERT_TRUE(motion.ok());
  LiquidSceneInputs3D scene;
  scene.obstacle = TranslatingHalfspace3D{
      .unit_normal = {0, 1, 0},
      .start_plane_coordinate = 0,
      .wall_velocity = {},
  };
  EXPECT_EQ(
      ReconstructSurfaceFrame3D(*motion, scene, {}, nullptr, {}).status().code(),
      absl::StatusCode::kUnimplemented
  );
}

TEST(SurfaceReconstruction3D, EnforcesSampleBudgetBeforeValidatingOrAllocatingDerivedGeometry) {
  MotionFrame3D source;
  source.samples.resize(66);
  EXPECT_EQ(
      ReconstructSurfaceFrame3D(source, {}, {}, nullptr, {}).status().code(),
      absl::StatusCode::kResourceExhausted
  );
}

TEST(SurfaceReconstruction3D, EnforcesSourceRecordBudgetAndReportsMaterialUnderflow) {
  auto seed = Seed();
  auto bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(bytes.ok());
  auto source = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(source.ok());
  SurfaceReconstructionOptions3D options{.spacing_m = 0.125, .sigma_m = 0.25};
  options.limits.cache.maximum_records.particles = 0;
  EXPECT_EQ(
      ReconstructSurfaceFrame3D(*source, {}, {}, nullptr, options).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  seed.particles[0].reference_volume = 1e-200;
  seed.particles[0].volume_ratio = 1e-200;
  bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(bytes.ok());
  source = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(source.ok());
  EXPECT_EQ(ReconstructSurfaceFrame3D(*source, {}, {}, nullptr, {}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(SurfaceReconstruction3D, RejectsDisconnectedMaterialBeforeDensityCanBridgeTheGap) {
  auto seed = Seed();
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(2);
  ASSERT_TRUE(particles.ok());
  ASSERT_TRUE(particles->Resize(2).ok());
  (*particles)[0] = seed.particles[0];
  (*particles)[1] = seed.particles[0];
  (*particles)[1].id = 2;
  (*particles)[1].world_position.x += 0.21;
  seed.particles = std::move(*particles);
  seed.next_particle_id = 3;
  auto bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(bytes.ok());
  auto source = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(source.ok());
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{.radius = 0.1};
  // The gap is narrower than the Gaussian scale. Do not allocate the render
  // field or infer connected material from its potentially bridged geometry.
  const SurfaceReconstructionOptions3D options{.maximum_nodes = 1};
  const auto result = ReconstructSurfaceFrame3D(*source, scene, {}, nullptr, options);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(result.status().message().find("material component"), std::string_view::npos);
}

TEST(SurfaceReconstruction3D, RejectsCoincidentNewComponentInsteadOfReusingSurfaceIds) {
  auto seed = Seed();
  auto bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(bytes.ok());
  auto initial = DecodeMotionFrame3D(bytes->values());
  ASSERT_TRUE(initial.ok());
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{.radius = 1.1};
  SurfaceReconstructionOptions3D options{.spacing_m = 0.125, .sigma_m = 0.25};
  auto first = ReconstructSurfaceFrame3D(*initial, scene, {}, nullptr, options);
  ASSERT_TRUE(first.ok()) << first.status();
  auto capture = MotionCapture3D::Create(seed, 1.0 / 24, 1);
  ASSERT_TRUE(capture.ok());
  seed.particles[0].id = 2;
  seed.next_particle_id = 3;
  seed.clock = {.time = 1.0 / 24, .frame_ordinal = 1, .accepted_step_ordinal = 1, .epoch = 1};
  ASSERT_TRUE(capture->Observe(seed, false).ok());
  auto interval = capture->Finish(seed);
  ASSERT_TRUE(interval.ok());
  auto source = DecodeMotionFrame3D(interval->values());
  ASSERT_TRUE(source.ok());
  options.maximum_nodes = 1;
  const auto result = ReconstructSurfaceFrame3D(*source, scene, {}, &first->frame, options);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(result.status().message().find("lineage"), std::string_view::npos);
}
}  // namespace
}  // namespace fluidity
