#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_substep_3d.h"
#include "gtest/gtest.h"

namespace fluidity::test {

inline absl::StatusOr<SparseMacGrid3D> SnowSchedulerGrid(const UniformMacGeometry3D& geometry) {
  const std::array blocks{
      BlockIndex3D{0, 0, 0},
      BlockIndex3D{1, 0, 0},
      BlockIndex3D{0, 1, 0},
      BlockIndex3D{1, 1, 0},
      BlockIndex3D{0, 0, 1},
      BlockIndex3D{1, 0, 1},
      BlockIndex3D{0, 1, 1},
      BlockIndex3D{1, 1, 1},
  };
  return SparseMacGrid3D::Create(geometry, blocks, blocks.size());
}

inline void ExpectSnowJointAcceptance(LiquidSubstepScheduler3D& scheduler, const LiquidState3D& state) {
  const auto solid_grid = SnowSchedulerGrid(state.geometry);
  ASSERT_TRUE(solid_grid.ok()) << solid_grid.status();
  const SnowParameters3D snow{2, 3, 0.025, 0.01, 4};
  const DrySolidParticle3D particle{
      .world_position = {3.0, 3.0, 3.0},
      .world_velocity = {-0.1, 0.0, 0.0},
      .mass = 2,
      .reference_volume = 0.1,
      .elastic_deformation = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
  };
  const auto record = CapturePlasticSolidCheckpointParticle3D(7, 4, 1, snow, particle);
  ASSERT_TRUE(record.ok()) << record.status();
  const std::array accepted_snow{*record};
  const LiquidSnowStepInputs3D inputs{
      .solid_grid = *solid_grid,
      .accepted_particles = accepted_snow,
      .body_id = 4,
      .material_id = 1,
      .snow = snow,
      .solid_step = {.duration = 0.01, .acoustic_cfl = 0.25},
      .support_radius = 0.35,
  };
  ASSERT_TRUE(scheduler.BeginFrame(0.01).ok());
  bool rejected_once = false;
  scheduler.SetStageFaultHookForTesting({
      .context = &rejected_once,
      .before_stage =
          [](void* context, LiquidPhysicalStage3D stage, double) {
            auto& rejected = *static_cast<bool*>(context);
            if (stage == LiquidPhysicalStage3D::kAccept && !rejected) {
              rejected = true;
              return absl::AbortedError("test rejection after both trials");
            }
            return absl::OkStatus();
          },
  });

  const auto rejected = scheduler.AttemptWithSnow(0.01, inputs);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_TRUE(rejected_once);
  EXPECT_EQ(rejected->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_FALSE(rejected->accepted_snow.has_value());
  const auto before = EncodeLiquidState3D(state);
  const auto after = EncodeLiquidState3D(scheduler.committed());
  ASSERT_TRUE(before.ok());
  ASSERT_TRUE(after.ok());
  EXPECT_TRUE(std::ranges::equal(before->values(), after->values()));
  EXPECT_DOUBLE_EQ(accepted_snow[0].transfer.sorted_world_position.x, 3.0);

  const auto accepted = scheduler.AttemptWithSnow(0.005, inputs);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_EQ(accepted->outcome.disposition, LiquidAttemptDisposition3D::kAccepted)
      << static_cast<int>(scheduler.diagnostics().failure_code) << " stage "
      << static_cast<int>(scheduler.diagnostics().failed_stage);
  ASSERT_TRUE(accepted->accepted_snow.has_value());
  ASSERT_EQ(accepted->accepted_snow->size(), 1);
  EXPECT_EQ(accepted->accepted_snow->front().particle_id, 7);
  EXPECT_NE(accepted->accepted_snow->front().transfer.sorted_world_position.x, 3.0);
  EXPECT_GT(std::abs(accepted->accepted_snow->front().transfer.particle_velocity.x + 0.1), 1e-8);
  EXPECT_DOUBLE_EQ(scheduler.committed().clock.time, 0.005);
}

enum class SnowPlatenRun {
  kExplicitSteps,
  kRetry,
  kCapacityReplay
};

inline void ExpectSnowMovingPlaten(
    const LiquidState3D& state,
    LiquidStepOptions3D options,
    LiquidSubstepBackend3D backend,
    SnowPlatenRun run = SnowPlatenRun::kExplicitSteps
) {
  constexpr double start_y = 3.001;
  constexpr double wall_speed = -0.2;
  constexpr double duration = 0.005;
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
  scene.obstacle = TranslatingHalfspace3D{{0, -1, 0}, -start_y, {0, wall_speed, 0}};
  const bool capacity_replay = run == SnowPlatenRun::kCapacityReplay;
  if (capacity_replay) {
    options.capacity_replay_reservations = options.capacities;
    options.capacity_replay_reservations.particles *= 2;
  }
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options, backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto solid_grid = SnowSchedulerGrid(state.geometry);
  ASSERT_TRUE(solid_grid.ok()) << solid_grid.status();
  const SnowParameters3D snow{2, 3, 0.025, 0.01, 4};
  const DrySolidParticle3D particle{
      .world_position = {5.5, 3.0005, 2.5},
      .world_velocity = {0, 1, 0},
      .mass = 2,
      .reference_volume = 0.1,
      .elastic_deformation = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
  };
  const auto record = CapturePlasticSolidCheckpointParticle3D(7, 4, 1, snow, particle);
  ASSERT_TRUE(record.ok()) << record.status();
  std::array accepted_snow{*record};
  const LiquidSnowStepInputs3D inputs{
      .solid_grid = *solid_grid,
      .accepted_particles = accepted_snow,
      .body_id = 4,
      .material_id = 1,
      .snow = snow,
      .solid_step = {.duration = duration, .acoustic_cfl = 0.25},
      .support_radius = 0.35,
  };
  if (run != SnowPlatenRun::kExplicitSteps) {
    std::vector frame_snow{*record};
    auto frame_inputs = inputs;
    frame_inputs.accepted_particles = frame_snow;
    if (capacity_replay) {
      // The initial acoustic bound is between one half-step and a full step.
      frame_inputs.solid_step.acoustic_cfl = 0.0045;
    }
    struct Observation {
      const std::vector<PlasticSolidCheckpointParticle3D>& snow;
      bool capacity_replay;
      bool rejected = false;
      int accepted = 0;
      int rollbacks = 0;
    } observation{frame_snow, capacity_replay};
    scheduler->SetStageFaultHookForTesting({
        .context = &observation,
        .before_stage =
            [](void* context, LiquidPhysicalStage3D stage, double) {
              auto& observed = *static_cast<Observation*>(context);
              if (stage == LiquidPhysicalStage3D::kAccept && !observed.rejected
                  && (!observed.capacity_replay || observed.accepted == 1)) {
                observed.rejected = true;
                if (observed.capacity_replay) {
                  EXPECT_NEAR(observed.snow.front().transfer.sorted_world_position.y, 3.0, 2e-14);
                  return absl::ResourceExhaustedError("grow after one accepted joint step");
                }
                EXPECT_DOUBLE_EQ(observed.snow.front().transfer.sorted_world_position.y, 3.0005);
                return absl::AbortedError("reject the joint full step");
              }
              return absl::OkStatus();
            },
    });
    const auto status = scheduler->AdvanceFrameWithSnow(
        2 * duration,
        frame_inputs,
        frame_snow,
        {
            .context = &observation,
            .observe =
                [](void* context, const LiquidState3D& liquid, const LiquidStepDiagnostics3D&, bool replay) {
                  auto& observed = *static_cast<Observation*>(context);
                  if (replay) {
                    EXPECT_TRUE(observed.capacity_replay);
                    EXPECT_DOUBLE_EQ(liquid.clock.time, 0.0);
                    EXPECT_DOUBLE_EQ(observed.snow.front().transfer.sorted_world_position.y, 3.0005);
                    EXPECT_DOUBLE_EQ(observed.snow.front().transfer.particle_velocity.y, 1.0);
                    observed.accepted = 0;
                    ++observed.rollbacks;
                    return absl::OkStatus();
                  }
                  ++observed.accepted;
                  EXPECT_DOUBLE_EQ(liquid.clock.time, observed.accepted * duration);
                  const double platen_y = start_y + liquid.clock.time * wall_speed;
                  const double snow_y = observed.snow.front().transfer.sorted_world_position.y;
                  EXPECT_LE(snow_y, platen_y + 2e-14);
                  EXPECT_NEAR(snow_y, platen_y, 2e-8);
                  return absl::OkStatus();
                },
        }
    );
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_TRUE(observation.rejected);
    EXPECT_EQ(observation.accepted, 2);
    EXPECT_EQ(observation.rollbacks, capacity_replay ? 1 : 0);
    EXPECT_EQ(scheduler->diagnostics().dt_retries, capacity_replay ? 0 : 1);
    EXPECT_EQ(scheduler->committed().clock.frame_ordinal, 1);
    return;
  }

  // Snow lies outside liquid support, so the dry trajectory supplies an independent contact oracle.
  std::array dry_reference{particle};
  ASSERT_TRUE(scheduler->BeginFrame(2 * duration).ok());
  for (int step = 1; step <= 2; ++step) {
    const auto result = scheduler->AttemptWithSnow(duration, inputs);
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_EQ(result->outcome.disposition, LiquidAttemptDisposition3D::kAccepted)
        << static_cast<int>(scheduler->diagnostics().failure_code) << " stage "
        << static_cast<int>(scheduler->diagnostics().failed_stage);
    ASSERT_TRUE(result->accepted_snow.has_value());
    ASSERT_EQ(result->accepted_snow->size(), 1);
    accepted_snow[0] = result->accepted_snow->front();
    auto dry_settings = inputs.solid_step;
    dry_settings.platen = DrySolidPlaten3D{
        .start_y = start_y + (step - 1) * duration * wall_speed,
        .velocity_y = wall_speed,
    };
    const auto dry_status = AdvanceDrySolid3D(*solid_grid, snow, dry_settings, dry_reference);
    ASSERT_TRUE(dry_status.ok()) << dry_status;
    constexpr double position_tolerance_m = 2e-14;
    EXPECT_NEAR(
        accepted_snow[0].transfer.sorted_world_position.y,
        dry_reference[0].world_position.y,
        position_tolerance_m
    );
    EXPECT_LE(
        accepted_snow[0].transfer.sorted_world_position.y,
        start_y + step * duration * wall_speed + position_tolerance_m
    );
    EXPECT_NEAR(accepted_snow[0].transfer.particle_velocity.y, dry_reference[0].world_velocity.y, 2e-14);
    EXPECT_DOUBLE_EQ(scheduler->committed().clock.time, step * duration);
  }
}

}  // namespace fluidity::test
