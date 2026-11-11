#include "fluidity/host/motion_capture_3d.h"
#include "fluidity/host/whitewater_frame_3d.h"
#include "fluidity/host/whitewater_sphere_source_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(WhitewaterSphereSource3D, RealAcceptedSchedulerStepsProduceBirthsBetweenCarrierSamples) {
  LiquidState3D seed;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(1);
  ASSERT_TRUE(particles.ok());
  ASSERT_TRUE(particles->Resize(1).ok());
  (*particles)[0].id = 1;
  (*particles)[0].world_position = {2.5, 2.5, 2.5};
  (*particles)[0].world_velocity = {2, 0, 0};
  seed.particles = std::move(*particles);
  seed.next_particle_id = 2;
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
  scene.world_acceleration = {};
  LiquidStepOptions3D step;
  step.capacities = {16, 64, 128, 12288, 1};
  auto scheduler = LiquidSubstepScheduler3D::Create(seed, scene, step);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  auto capture = MotionCapture3D::Create(seed, 1.0 / 24, 1);
  ASSERT_TRUE(capture.ok());
  struct CaptureContext {
    MotionCapture3D& capture;
    const LiquidSceneInputs3D& scene;
    const LiquidStepOptions3D& step;
    double prior_time_seconds = 0;
  } context{*capture, scene, step};
  const LiquidProgressObserver3D observer{
      .context = &context,
      .observe =
          [](void* opaque, const LiquidState3D& accepted, const LiquidStepDiagnostics3D&, bool rollback) {
            auto& context = *static_cast<CaptureContext*>(opaque);
            if (rollback) {
              context.prior_time_seconds = accepted.clock.time;
              return context.capture.Observe(accepted, true);
            }
            const WhitewaterSourceOptions3D source{
                .spacing_m = 0.5,
                .radius_m = 0.5,
                .kinetic = {0, 1},
                .trapped_air = {0, 1},
                .crest = {0, 1},
                .trapped_air_weight = 0,
                .crest_weight = 1,
            };
            auto births = SampleAcceptedWhitewaterStep3D(
                accepted,
                context.scene,
                context.step,
                {},
                context.prior_time_seconds,
                source
            );
            if (!births.ok()) {
              return births.status();
            }
            const auto captured = context.capture.Observe(accepted, false, *births);
            if (captured.ok()) {
              context.prior_time_seconds = accepted.clock.time;
            }
            return captured;
          },
  };
  const auto advanced = scheduler->AdvanceFrame(1.0 / 24, observer);
  ASSERT_TRUE(advanced.ok()) << advanced;
  auto encoded = capture->Finish(scheduler->committed());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  auto motion = DecodeMotionFrame3D(encoded->values());
  ASSERT_TRUE(motion.ok());
  ASSERT_EQ(motion->samples.size(), 2);
  ASSERT_FALSE(motion->whitewater_births.empty());
  EXPECT_LT(motion->whitewater_births.front().accepted_step_ordinal, motion->last_clock.accepted_step_ordinal);
  EXPECT_GT(motion->whitewater_births.front().birth.event.velocity.x, 0.5);
  for (const auto& birth : motion->whitewater_births) {
    EXPECT_GT(birth.birth.event.volume_time_m3_seconds, 0);
    EXPECT_LE(birth.birth.event.time_seconds, motion->last_clock.time);
  }
  const auto seed_bytes = EncodeMotionSeed3D(seed);
  ASSERT_TRUE(seed_bytes.ok());
  const auto source_seed = DecodeMotionFrame3D(seed_bytes->values());
  ASSERT_TRUE(source_seed.ok());
  const WhitewaterFrameOptions3D options{.emission_rate_per_m3_second = 1000};
  const auto initial = ReconstructWhitewaterFrame3D(*source_seed, scene, step, nullptr, options);
  ASSERT_TRUE(initial.ok()) << initial.status();
  EXPECT_TRUE(initial->frame.points.empty());
  const auto next = ReconstructWhitewaterFrame3D(*motion, scene, step, &initial->frame, options);
  ASSERT_TRUE(next.ok()) << next.status();
  ASSERT_FALSE(next->frame.points.empty());
  EXPECT_GT(next->birth_count, 0);
  EXPECT_GT(next->transport_substeps, 0);
  EXPECT_EQ(next->frame.clock, motion->last_clock);
  EXPECT_EQ(next->frame.emission.last_accepted_step_ordinal, motion->last_clock.accepted_step_ordinal);
  EXPECT_TRUE(std::ranges::any_of(next->frame.points, [](const auto& record) {
    return record.point.kind == WhitewaterKind3D::kFoam;
  }));
  std::uint64_t previous_id = 0;
  for (const auto& record : next->frame.points) {
    EXPECT_GT(record.id, previous_id);
    EXPECT_LE(record.point.birth_time_seconds, next->frame.clock.time);
    previous_id = record.id;
  }
}
}  // namespace
}  // namespace fluidity
