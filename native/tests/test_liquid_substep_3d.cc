#include <array>
#include <chrono>
#include <cmath>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include "fluidity/host/liquid_substep_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "gtest/gtest.h"
namespace fluidity {
namespace {
LiquidState3D Initial(std::size_t count = 1) {
  LiquidState3D state;
  state.particles = *LiquidRecords3D<LiquidParticleState3D>::Create(count);
  EXPECT_TRUE(state.particles.Resize(count).ok());
  state.next_particle_id = count + 1;
  for (std::size_t i = 0; i < count; ++i) {
    state.particles[i].id = i + 1;
    state.particles[i].world_position = {2.5, 2.5, 2.5};
    state.particles[i].pic_initialized = true;
  }
  return state;
}
LiquidStepOptions3D Options() {
  LiquidStepOptions3D options;
  options.capacities = {16, 64, 128, 12288, 1};
  return options;
}
LiquidSceneInputs3D Scene() {
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
  return scene;
}
TEST(LiquidSubstep3D, PhiloxPublishedZeroKnownVector) {
  EXPECT_EQ(
      LiquidPhilox4x32_10({0, 0, 0, 0}, {0, 0}),
      (std::array<std::uint32_t, 4>{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8})
  );
}
TEST(LiquidSubstep3D, OrderedStagesPublishExactFrameEpoch) {
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
  EXPECT_EQ(scheduler->committed().clock.frame_ordinal, 1);
  EXPECT_EQ(scheduler->committed().clock.substep_ordinal, 0);
  EXPECT_DOUBLE_EQ(scheduler->committed().clock.time, 0.01);
  ASSERT_EQ(scheduler->diagnostics().stage_count, kLiquidPhysicalStageCount3D);
  for (std::size_t i = 0; i < kLiquidPhysicalStageCount3D; ++i) {
    EXPECT_EQ(static_cast<std::size_t>(scheduler->diagnostics().stages[i]), i);
  }
}
class LiquidGravity3D : public testing::TestWithParam<int> {};
TEST_P(LiquidGravity3D, AccelerationIndependentOfRetention) {
  auto options = Options();
  options.particle_energy_blend = std::array{0.0, 0.6, 1.0}[GetParam()];
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.y, -0.0981, 0.0000981);
}
INSTANTIATE_TEST_SUITE_P(LiquidSubstep3D, LiquidGravity3D, testing::Values(0, 1, 2));
TEST(LiquidSubstep3D, StoredResidualDoesNotAdvectInterior) {
  auto initial = Initial(2);
  initial.particles[0].world_velocity.x = 1;
  initial.particles[1].world_velocity.x = -1;
  auto scene = Scene();
  scene.world_acceleration = {};
  auto options = Options();
  options.particle_energy_blend = 1;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.x, 0.99, 1e-8);
  EXPECT_NEAR(scheduler->committed().particles[0].world_position.x, 2.5, 1e-8);
}
TEST(LiquidSubstep3D, FiniteExponentialVolumePreservesMassAndReferenceVolume) {
  auto initial = Initial();
  initial.particles[0].velocity_slopes = {{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}};
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.target_divergence = 0.3;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_GT(scheduler->committed().particles[0].volume_ratio, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].mass, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].reference_volume, 1);
}
TEST(LiquidSubstep3D, CapillaryAmbientAndMovingWallAreResolved) {
  auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.surface_tension = 0.1;
  scene.ambient_pressure = 3;
  scene.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {0.1, 0, 0}};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  ASSERT_GT(scheduler->committed().cells.size(), 0);
  EXPECT_GT(scheduler->committed().cells[0].previous_pressure, 3);
  EXPECT_LE(scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect, 2e-5);
}
class LiquidStageFailure3D : public testing::TestWithParam<int> {};
TEST_P(LiquidStageFailure3D, CommittedInventoryUnchanged) {
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  auto snapshot = CloneLiquidState3D(scheduler->committed(), Options().capacities);
  ASSERT_TRUE(snapshot.ok());
  auto stage = static_cast<LiquidPhysicalStage3D>(GetParam());
  scheduler->SetStageFaultHookForTesting({&stage, [](void* context, LiquidPhysicalStage3D current, double) {
                                            return current == *static_cast<LiquidPhysicalStage3D*>(context)
                                                ? absl::Status(absl::StatusCode::kInternal, {})
                                                : absl::OkStatus();
                                          }});
  ASSERT_TRUE(scheduler->BeginFrame(0.02).ok());
  auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_TRUE(scheduler->committed() == *snapshot);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, stage);
}
INSTANTIATE_TEST_SUITE_P(LiquidSubstep3D, LiquidStageFailure3D, testing::Range(0, 16));
TEST(LiquidSubstep3D, RetryHalvesTwiceButIncompatibilityIsFatal) {
  for (const auto code : {absl::StatusCode::kAborted, absl::StatusCode::kFailedPrecondition}) {
    auto initial = Initial();
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    auto error = code;
    scheduler->SetStageFaultHookForTesting({&error, [](void* context, LiquidPhysicalStage3D stage, double) {
                                              return stage == LiquidPhysicalStage3D::kPressure
                                                  ? absl::Status(*static_cast<absl::StatusCode*>(context), {})
                                                  : absl::OkStatus();
                                            }});
    EXPECT_FALSE(scheduler->AdvanceFrame(0.01).ok());
    EXPECT_EQ(scheduler->diagnostics().attempts, code == absl::StatusCode::kAborted ? 3 : 1);
    EXPECT_TRUE(scheduler->committed() == initial);
  }
}
TEST(LiquidSubstep3D, SourceRetryPreservesCounterAndSeededBirths) {
  auto initial = Initial(0);
  initial.sources = *LiquidRecords3D<LiquidSourceState3D>::Create(1);
  ASSERT_TRUE(initial.sources.Resize(1).ok());
  LiquidEmissionSource3D source;
  source.particles_per_second = 100;
  source.world_position = {2.5, 2.5, 2.5};
  source.world_position_jitter = {0.1, 0.1, 0.1};
  auto scene = Scene();
  scene.sources = std::span(&source, 1);
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(scheduler->committed().particles.size(), 1);
  EXPECT_EQ(scheduler->committed().sources[0].next_child_ordinal, 1);
  EXPECT_EQ(scheduler->committed().next_particle_id, 2);
}
TEST(LiquidSubstep3D, CapacityGrowthReplaysWholeFrameOnce) {
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            if (stage == LiquidPhysicalStage3D::kEmit
                                                && ++*static_cast<int*>(context) == 2) {
                                              return absl::Status(absl::StatusCode::kResourceExhausted, {});
                                            }
                                            return absl::OkStatus();
                                          }});
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(scheduler->diagnostics().capacity_replays, 1);
  EXPECT_EQ(scheduler->committed().clock.epoch, 2);
}
TEST(LiquidSubstep3D, RepeatedGravityAndUniformTranslationRemainAnalytic) {
  for (double energy : {0.0, 0.6, 1.0}) {
    auto initial = Initial();
    initial.particles[0].world_velocity = {0.5, 0.25, -0.1};
    auto options = Options();
    options.particle_energy_blend = energy;
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
    ASSERT_TRUE(scheduler.ok());
    for (int frame = 1; frame <= 5; ++frame) {
      ASSERT_TRUE(scheduler->AdvanceFrame(0.01 * frame).ok())
          << static_cast<int>(scheduler->diagnostics().failed_stage);
    }
    EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.x, 0.5, 0.0005);
    EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.y, 0.25 - 9.81 * 0.05, 0.0004905);
    EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.z, -0.1, 0.0001);
  }
}
TEST(LiquidSubstep3D, MovingWallKeepsDistinctOldAndNewEligibility) {
  auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 1.995, {1, 0, 0}};
  struct Observed {
    bool changed = false;
    bool times_match = true;
  } observed;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observed;
  backend.gather = [](void* context,
                      const LiquidBoundaryGatherViewInputs3D& inputs,
                      LiquidBoundaryWorkspace3D& workspace) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    auto& result = *static_cast<Observed*>(context);
    result.times_match = result.times_match && inputs.old_history.wall_sample_time == inputs.old_time
        && inputs.new_history.wall_sample_time == inputs.new_time;
    for (std::size_t i = 0; i < inputs.old_history.face_tags.size(); ++i) {
      result.changed = result.changed
          || (inputs.old_history.face_tags[i].structurally_eligible
              && !inputs.new_history.face_tags[i].structurally_eligible);
    }
    return GatherLiquidBoundaryParticles3D(inputs, workspace);
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok()) << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_TRUE(observed.changed);
  EXPECT_TRUE(observed.times_match);
}
TEST(LiquidSubstep3D, ThinSlabSharedStencilNodesUseBothIncidentWallNormals) {
  auto initial = Initial(2);
  initial.particles[0].world_position.x = 1.4;
  initial.particles[0].side_tag = -1;
  initial.particles[1].world_position.x = 2.6;
  initial.particles[1].side_tag = 1;
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{0.9};
  scene.world_acceleration = {};
  scene.obstacle = TranslatingSlab3D{{1, 0, 0}, 2, 0.05, {0.1, 0, 0}};
  std::array<bool, 2> seen{};
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &seen;
  backend.gather = [](void* context, const LiquidBoundaryGatherViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace
                   ) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    auto& observed = *static_cast<std::array<bool, 2>*>(context);
    const auto side = input.particles[0].side_tag;
    for (const auto& wall : input.old_history.wall_samples) {
      if (wall.available) {
        EXPECT_EQ(wall.side_tag, side);
        EXPECT_DOUBLE_EQ(wall.unit_normal.x, static_cast<double>(side));
        observed[side > 0 ? 1 : 0] = true;
      }
    }
    return GatherLiquidBoundaryParticles3D(input, workspace);
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok()) << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_TRUE(seen[0]);
  EXPECT_TRUE(seen[1]);
  EXPECT_LT(scheduler->committed().particles[0].world_position.x, 1.96);
  EXPECT_GT(scheduler->committed().particles[1].world_position.x, 2.06);
}
TEST(LiquidSubstep3D, EmptyUnionAndMalformedSignedExtentAreDistinguished) {
  auto empty = Initial(0);
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{0.5};
  auto scheduler = LiquidSubstepScheduler3D::Create(empty, scene, Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(scheduler->committed().cells.size(), 0);
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
  scene.field = LiquidAnalyticSphere3D{{static_cast<double>(INT32_MAX), 0, 0}, 1};
  auto invalid = LiquidSubstepScheduler3D::Create(empty, scene, Options());
  ASSERT_TRUE(invalid.ok());
  EXPECT_EQ(invalid->AdvanceFrame(0.01).code(), absl::StatusCode::kOutOfRange);
  EXPECT_TRUE(invalid->committed() == empty);
}
TEST(LiquidSubstep3D, SourceRetryAndWholeFrameReplayProduceIdenticalBirths) {
  auto initial = Initial(0);
  initial.random_seed = 123456789;
  initial.sources = *LiquidRecords3D<LiquidSourceState3D>::Create(1);
  ASSERT_TRUE(initial.sources.Resize(1).ok());
  LiquidEmissionSource3D source;
  source.particles_per_second = 400;
  source.world_position = {2.5, 2.5, 2.5};
  source.world_position_jitter = {0.2, 0.2, 0.2};
  auto scene = Scene();
  scene.sources = std::span(&source, 1);
  auto options = Options();
  options.maximum_step_duration = 0.005;
  auto reference = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(reference.ok());
  ASSERT_TRUE(reference->AdvanceFrame(0.01).ok());
  options.maximum_step_duration = 0.01;
  auto retry = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(retry.ok());
  int rejected = 0;
  retry->SetStageFaultHookForTesting({&rejected, [](void* c, LiquidPhysicalStage3D stage, double) {
                                        if (stage == LiquidPhysicalStage3D::kAccept && ++*static_cast<int*>(c) == 1) {
                                          return absl::Status(absl::StatusCode::kAborted, {});
                                        }
                                        return absl::OkStatus();
                                      }});
  ASSERT_TRUE(retry->AdvanceFrame(0.01).ok());
  EXPECT_TRUE(retry->committed() == reference->committed());
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  auto replay = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(replay.ok());
  rejected = 0;
  replay->SetStageFaultHookForTesting({&rejected, [](void* c, LiquidPhysicalStage3D stage, double) {
                                         if (stage == LiquidPhysicalStage3D::kAccept && ++*static_cast<int*>(c) == 2) {
                                           return absl::Status(absl::StatusCode::kResourceExhausted, {});
                                         }
                                         return absl::OkStatus();
                                       }});
  ASSERT_TRUE(replay->AdvanceFrame(0.01).ok());
  EXPECT_TRUE(replay->committed() == reference->committed());
}
TEST(LiquidSubstep3D, RetainedAgesAdvanceOnlyOnAcceptanceAndFinishDoesNotMutateEpoch) {
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  for (const auto& block : scheduler->committed().blocks.values()) {
    EXPECT_EQ(block.activation_age, 0);
  }
  ASSERT_TRUE(scheduler->BeginFrame(0.02).ok());
  auto accepted = scheduler->Attempt(0.01);
  ASSERT_TRUE(accepted.ok());
  ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
  for (const auto& block : scheduler->committed().blocks.values()) {
    EXPECT_EQ(block.activation_age, 1);
  }
  auto snapshot = CloneLiquidState3D(scheduler->committed(), Options().capacities);
  ASSERT_TRUE(snapshot.ok());
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  EXPECT_TRUE(scheduler->committed() == *snapshot);
}
TEST(LiquidSubstep3D, ExponentialUsesTrueGradientAndSeparationUsesSurfaceExpansion) {
  auto initial = Initial(2);
  for (auto& particle : initial.particles.values()) {
    particle.world_position.x = 3.3;
    particle.velocity_slopes = {{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}};
  }
  initial.particles[0].world_velocity.x = 1;
  initial.particles[1].world_velocity.x = -1;
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.target_divergence = 0.3;
  struct Observation {
    double divergence = 0;
    double pic = 0;
    double residual = 0;
  } observed;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observed;
  backend.update_particles =
      [](void* c, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
        const auto& value = inputs.gathered[0];
        *static_cast<Observation*>(c) = {
            value.velocity_gradient.x_row.x + value.velocity_gradient.y_row.y + value.velocity_gradient.z_row.z,
            value.new_pic_velocity.x,
            value.particle_minus_old_pic_velocity.x
        };
        return CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
      };
  auto options = Options();
  options.particle_energy_blend = 0.6;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  ASSERT_GT(observed.divergence, 0);
  ASSERT_GT(observed.residual, 0.5);
  const auto& particle = scheduler->committed().particles[0];
  EXPECT_NEAR(particle.volume_ratio, std::exp(0.01 * observed.divergence), 1e-13);
  EXPECT_NEAR(particle.world_velocity.x, observed.pic + 0.594 * observed.residual, 1e-12);
  EXPECT_NEAR(particle.world_position.x, 3.3 + 0.01 * (observed.pic + 0.6 * 0.594 * observed.residual), 1e-12);
}
TEST(LiquidSubstep3D, CompleteBackendIsRequiredAndRealPressureNonconvergenceRetries) {
  auto initial = Initial(8);
  for (std::size_t i = 0; i < 8; ++i) {
    auto& p = initial.particles[i];
    p.world_position = {1.5 + 2 * (i & 1), 1.5 + 2 * ((i >> 1) & 1), 1.5 + 2 * ((i >> 2) & 1)};
    p.world_velocity = {
        0.1 * static_cast<double>(i),
        0.03 * static_cast<double>(i * i),
        -0.07 * static_cast<double>(i)
    };
  }
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 2.1};
  auto options = Options();
  options.pressure_solver_options.max_iterations = 1;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.project = nullptr;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, scene, options, backend).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kAborted);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  EXPECT_EQ(scheduler->diagnostics().attempts, 3);
  EXPECT_TRUE(scheduler->committed() == initial);
}
#if defined(_WIN32) && defined(_DEBUG)
std::size_t scheduler_allocations = 0;
std::size_t scheduler_requested_bytes = 0;
std::size_t scheduler_last_request = 0;
std::size_t scheduler_frees = 0;
std::size_t scheduler_fail_allocation = 0;
_CRT_ALLOC_HOOK scheduler_previous_hook = nullptr;
int SchedulerAllocationHook(
    int operation,
    void* data,
    std::size_t size,
    int block_type,
    long request,
    const unsigned char* filename,
    int line
) {
  if (operation == _HOOK_ALLOC || operation == _HOOK_REALLOC) {
    ++scheduler_allocations;
    scheduler_requested_bytes += size;
    scheduler_last_request = size;
    if (scheduler_allocations == scheduler_fail_allocation) {
      return 0;
    }
  }
  if (operation == _HOOK_FREE) {
    ++scheduler_frees;
  }
  return scheduler_previous_hook ? scheduler_previous_hook(operation, data, size, block_type, request, filename, line)
                                 : 1;
}
TEST(LiquidSubstep3D, ActualSuccessfulAndRejectedAttemptsAllocateAndFreeNothing) {
  for (const auto failure :
       {LiquidPhysicalStage3D::kCount, LiquidPhysicalStage3D::kPressure, LiquidPhysicalStage3D::kAccept}) {
    auto initial = Initial();
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
    ASSERT_TRUE(scheduler.ok());
    ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
    ASSERT_TRUE(scheduler->BeginFrame(0.02).ok());
    auto stage = failure;
    scheduler->SetStageFaultHookForTesting({&stage, [](void* c, LiquidPhysicalStage3D current, double) {
                                              return current == *static_cast<LiquidPhysicalStage3D*>(c)
                                                  ? absl::Status(absl::StatusCode::kAborted, {})
                                                  : absl::OkStatus();
                                            }});
    scheduler_allocations = 0;
    scheduler_frees = 0;
    bool succeeded = false;
    LiquidAttemptDisposition3D disposition = LiquidAttemptDisposition3D::kFatal;
    scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
    {
      auto result = scheduler->Attempt(0.01);
      succeeded = result.ok();
      if (succeeded) {
        disposition = result->disposition;
      }
    }
    _CrtSetAllocHook(scheduler_previous_hook);
    EXPECT_TRUE(succeeded);
    EXPECT_EQ(
        disposition,
        failure == LiquidPhysicalStage3D::kCount ? LiquidAttemptDisposition3D::kAccepted
                                                 : LiquidAttemptDisposition3D::kRetryDt
    );
    EXPECT_EQ(scheduler_allocations, 0);
    EXPECT_EQ(scheduler_frees, 0);
  }
}
TEST(LiquidSubstep3D, PartialSchedulerReservationFailureIsCheckedAndCleansUp) {
  auto initial = Initial();
  auto scene = Scene();
  auto options = Options();
  scheduler_allocations = 0;
  scheduler_frees = 0;
  scheduler_fail_allocation = 4;
  absl::StatusCode code = absl::StatusCode::kOk;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
    code = scheduler.status().code();
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  scheduler_fail_allocation = 0;
  EXPECT_EQ(code, absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(scheduler_allocations, 4);
  EXPECT_EQ(scheduler_frees, 3);
}
#endif
TEST(LiquidSubstep3D, CapacityWithoutReplayReservationReportsResourceExhausted) {
  auto initial = Initial();
  auto options = Options();
  options.capacities.blocks = 0;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE(scheduler->committed() == initial);
  EXPECT_EQ(scheduler->diagnostics().capacity_replays, 0);
}
TEST(LiquidSubstep3D, ExplicitCapillaryOversizeRequestsSmallerDuration) {
  auto initial = Initial();
  auto scene = Scene();
  scene.surface_tension = 1e8;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ASSERT_TRUE(result->retry_duration.has_value());
  EXPECT_DOUBLE_EQ(*result->retry_duration, 0.005);
  EXPECT_TRUE(scheduler->committed() == initial);
}
TEST(LiquidSubstep3D, ActualSealedIncompatiblePressureIsFatalWithoutRetry) {
  auto initial = Initial();
  auto scene = Scene();
  scene.target_divergence = 1;
  // This backend qualification fixture closes every frozen face. The real bounded
  // pressure assembly must reject its physical source, rather than retry the solver.
  std::array<MacPressureFace3D, 128> sealed_faces{};
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &sealed_faces;
  backend.project = [](void* context,
                       SparseMacGridView3D grid,
                       double dt,
                       const LiquidPressureProblemView3D& problem,
                       MacPressureSolverOptions3D options,
                       MacPressureWorkspace3D& workspace) -> absl::StatusOr<MacPressureProjectionView3D> {
    auto& faces = *static_cast<std::array<MacPressureFace3D, 128>*>(context);
    if (problem.faces.size() > faces.size()) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    std::ranges::copy(problem.faces, faces.begin());
    for (auto& face : std::span(faces).first(problem.faces.size())) {
      face.open_area = 0;
      face.pressure_link = SealedMacPressureLink3D{};
    }
    return ProjectMacPressure3D(
        grid,
        dt,
        {problem.cells, std::span(faces).first(problem.faces.size())},
        options,
        workspace
    );
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler->diagnostics().attempts, 1);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  EXPECT_TRUE(scheduler->committed() == initial);
}
TEST(LiquidSubstep3D, CandidateValidationRejectsPoisonAfterActualNumericalUpdate) {
  auto initial = Initial();
  auto backend = CpuLiquidSubstepBackend3D();
  backend.update_particles =
      [](void*, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
        const auto status = CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
        if (status.ok()) {
          particles[0].volume_ratio = std::numeric_limits<double>::quiet_NaN();
        }
        return status;
      };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(scheduler->committed() == initial);
}
TEST(LiquidSubstep3D, StableSignedBinsAndZeroWeightOwnersSurviveIntegratedStep) {
  auto initial = Initial(3);
  initial.particles[0].world_position = {6, 6, 6};
  initial.particles[1].world_position = {-2, -2, -2};
  initial.particles[2].world_position = {-2, -2, -2};
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{0.9};
  scene.world_acceleration = {};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  const auto& state = scheduler->committed();
  EXPECT_EQ(state.particles[0].id, 2);
  EXPECT_EQ(state.particles[1].id, 3);
  EXPECT_EQ(state.particles[2].id, 1);
  auto workspace = LiquidStepWorkspace3D::Create(Options().capacities, state.geometry);
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(workspace->PrepareTrialTopology(state.particles.values(), {}, &state).ok());
  auto topology = workspace->prepared();
  ASSERT_TRUE(topology.ok());
  for (const auto& particle : state.particles.values()) {
    for (auto axis : {FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ}) {
      auto stencil = BuildOwnedMacFaceStencil3D(topology->grid, axis, particle.world_position, particle.world_position);
      ASSERT_TRUE(stencil.ok());
      EXPECT_EQ(stencil->size(), 27);
    }
  }
}
#if defined(_WIN32) && defined(_DEBUG)
TEST(LiquidSubstep3D, ActualDepositRejectionAllocatesAndFreesNothing) {
  auto initial = Initial();
  initial.particles[0].mass = 1e100;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  scheduler_allocations = 0;
  scheduler_frees = 0;
  bool fatal = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    auto result = scheduler->Attempt(0.01);
    fatal = result.ok() && result->disposition == LiquidAttemptDisposition3D::kFatal;
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_TRUE(fatal);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kDeposit);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kOutOfRange);
  EXPECT_EQ(scheduler_allocations, 0);
  EXPECT_EQ(scheduler_frees, 0);
  EXPECT_TRUE(scheduler->committed() == initial);
}
TEST(LiquidSubstep3D, ActualPressureNonconvergenceAllocatesAndFreesNothing) {
  auto initial = Initial(8);
  for (std::size_t i = 0; i < 8; ++i) {
    auto& p = initial.particles[i];
    p.world_position = {1.5 + 2 * (i & 1), 1.5 + 2 * ((i >> 1) & 1), 1.5 + 2 * ((i >> 2) & 1)};
    p.world_velocity = {
        0.1 * static_cast<double>(i),
        0.03 * static_cast<double>(i * i),
        -0.07 * static_cast<double>(i)
    };
  }
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 2.1};
  auto options = Options();
  options.pressure_solver_options.max_iterations = 1;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  scheduler_allocations = 0;
  scheduler_frees = 0;
  bool retry = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    auto result = scheduler->Attempt(0.01);
    retry = result.ok() && result->retry_duration.has_value();
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_TRUE(retry);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kAborted);
  EXPECT_EQ(scheduler_allocations, 0);
  EXPECT_EQ(scheduler_frees, 0);
  EXPECT_TRUE(scheduler->committed() == initial);
}
#endif
TEST(LiquidSubstep3D, BoundedParticleCapacityCompleteStepMeasurement) {
  auto initial = Initial(1024);
  auto options = Options();
  options.capacities.particles = 1024;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto begin = std::chrono::steady_clock::now();
  const auto result = scheduler->Attempt(0.01);
  const auto end = std::chrono::steady_clock::now();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kAccepted);
  EXPECT_EQ(scheduler->committed().particles.size(), 1024);
  RecordProperty(
      "complete_attempt_microseconds",
      static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())
  );
  RecordProperty("reserved_payload_bytes", std::to_string(scheduler->reserved_bytes()));
  RecordProperty("particle_count", 1024);
}
TEST(LiquidSubstep3D, ExplicitWallMotionAboveHalfCellRequestsRetry) {
  auto initial = Initial();
  auto scene = Scene();
  scene.obstacle = TranslatingHalfspace3D{{1, 0, 0}, -100, {100, 0, 0}};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_TRUE(scheduler->committed() == initial);
}
#if defined(_WIN32) && defined(_DEBUG)
TEST(LiquidSubstep3D, ReportedReservationMatchesActualRequestedPayload) {
  auto initial = Initial();
  auto scene = Scene();
  auto options = Options();
  scheduler_allocations = 0;
  scheduler_frees = 0;
  scheduler_requested_bytes = 0;
  scheduler_last_request = 0;
  std::size_t reported = 0;
  std::size_t owner_bytes = 0;
  bool created = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
    created = scheduler.ok();
    if (created) {
      reported = scheduler->reserved_bytes();
      owner_bytes = scheduler_last_request;
    }
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_TRUE(created);
  EXPECT_EQ(scheduler_requested_bytes, reported + owner_bytes);
  EXPECT_EQ(scheduler_allocations, scheduler_frees);
  RecordProperty("reserved_payload_bytes", std::to_string(reported));
  RecordProperty("owner_object_bytes", std::to_string(owner_bytes));
}
#endif

struct BackendLifecycleObservation {
  int initialized = 0;
  int prepared = 0;
  int committed = 0;
  int discarded = 0;
  int growth_prepared = 0;
  int growth_committed = 0;
  int growth_discarded = 0;
  int samples = 0;
  int face_samples = 0;
  bool fail_publication = false;
  bool fail_growth = false;
  LiquidClock3D staged_clock{};
  LiquidClock3D growth_clock{};
};
LiquidSubstepBackend3D ObservedBackend(BackendLifecycleObservation& observed) {
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observed;
  backend.initialize = [](void* context, const LiquidState3D&) {
    ++static_cast<BackendLifecycleObservation*>(context)->initialized;
    return absl::OkStatus();
  };
  backend.prepare_publication = [](void* context, const LiquidState3D&, LiquidClock3D clock) {
    auto& observation = *static_cast<BackendLifecycleObservation*>(context);
    ++observation.prepared;
    observation.staged_clock = clock;
    return absl::Status(observation.fail_publication ? absl::StatusCode::kInternal : absl::StatusCode::kOk, {});
  };
  backend.commit_publication = [](void* context) noexcept {
    ++static_cast<BackendLifecycleObservation*>(context)->committed;
  };
  backend.discard_publication = [](void* context) noexcept {
    ++static_cast<BackendLifecycleObservation*>(context)->discarded;
  };
  backend.prepare_growth = [](void* context, LiquidStateCapacities3D, const LiquidState3D& start) {
    auto& observation = *static_cast<BackendLifecycleObservation*>(context);
    ++observation.growth_prepared;
    observation.growth_clock = start.clock;
    return absl::Status(observation.fail_growth ? absl::StatusCode::kResourceExhausted : absl::StatusCode::kOk, {});
  };
  backend.commit_growth = [](void* context) noexcept {
    ++static_cast<BackendLifecycleObservation*>(context)->growth_committed;
  };
  backend.discard_growth = [](void* context) noexcept {
    ++static_cast<BackendLifecycleObservation*>(context)->growth_discarded;
  };
  backend.sample_field =
      [](void* context, const LiquidSceneField3D& field, std::span<const LiquidParticleState3D> particles, Vec3 position
      ) {
        ++static_cast<BackendLifecycleObservation*>(context)->samples;
        return CpuLiquidSubstepBackend3D().sample_field(nullptr, field, particles, position);
      };
  backend.sample_faces = [](void* context,
                            SparseMacGridView3D grid,
                            const LiquidObstacle3D& obstacle,
                            std::int64_t side,
                            std::span<LiquidBoundaryFaceTag3D> tags,
                            std::span<LiquidPrescribedWallSample3D> walls) {
    ++static_cast<BackendLifecycleObservation*>(context)->face_samples;
    return CpuLiquidSubstepBackend3D().sample_faces(nullptr, grid, obstacle, side, tags, walls);
  };
  return backend;
}
TEST(LiquidSubstep3D, BackendStagesExactAcceptedClockAndSamplesAllFields) {
  auto initial = Initial();
  auto options = Options();
  options.maximum_step_duration = 0.005;
  BackendLifecycleObservation observed;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options, ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(observed.initialized, 1);
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  for (int step = 0; step < 2; ++step) {
    const auto result = scheduler->Attempt(0.005);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kAccepted);
    EXPECT_EQ(observed.staged_clock, scheduler->committed().clock);
    EXPECT_EQ(observed.prepared, step + 1);
    EXPECT_EQ(observed.committed, step + 1);
  }
  EXPECT_EQ(observed.discarded, 0);
  EXPECT_GT(observed.samples, 1);
  EXPECT_GT(observed.face_samples, 1);
}
TEST(LiquidSubstep3D, BackendPublicationFailurePreservesCompleteCommittedState) {
  auto initial = Initial();
  BackendLifecycleObservation observed;
  observed.fail_publication = true;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_EQ(observed.prepared, 1);
  EXPECT_EQ(observed.committed, 0);
  EXPECT_EQ(observed.discarded, 1);
  EXPECT_TRUE(scheduler->committed() == initial);
}
TEST(LiquidSubstep3D, BackendGrowthPreparationFailureKeepsLastAcceptedVersion) {
  auto initial = Initial();
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  BackendLifecycleObservation observed;
  observed.fail_growth = true;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options, ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kEmit
                                                    && ++*static_cast<int*>(context) == 2
                                                ? absl::Status(absl::StatusCode::kResourceExhausted, {})
                                                : absl::OkStatus();
                                          }});
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
  EXPECT_EQ(scheduler->committed().clock.time, 0.005);
  EXPECT_EQ(observed.growth_prepared, 1);
  EXPECT_EQ(observed.growth_committed, 0);
  EXPECT_EQ(observed.growth_discarded, 1);
  EXPECT_EQ(observed.growth_clock, initial.clock);
}
TEST(LiquidSubstep3D, BackendGrowthRestoresFrameStartBeforeReplay) {
  auto initial = Initial();
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  BackendLifecycleObservation observed;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options, ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kEmit
                                                    && ++*static_cast<int*>(context) == 2
                                                ? absl::Status(absl::StatusCode::kResourceExhausted, {})
                                                : absl::OkStatus();
                                          }});
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(observed.growth_prepared, 1);
  EXPECT_EQ(observed.growth_committed, 1);
  EXPECT_EQ(observed.growth_discarded, 0);
  EXPECT_EQ(observed.growth_clock, initial.clock);
  EXPECT_EQ(observed.committed, 3);
  EXPECT_EQ(observed.staged_clock, scheduler->committed().clock);
}
TEST(LiquidSubstep3D, BackendSamplerFailureAndMissingLifecycleNeverFallBack) {
  auto initial = Initial();
  auto backend = CpuLiquidSubstepBackend3D();
  backend.sample_field = [](void*, const LiquidSceneField3D&, std::span<const LiquidParticleState3D>, Vec3
                         ) -> absl::StatusOr<double> {
    return absl::Status(absl::StatusCode::kInternal, {});
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(scheduler->committed() == initial);
  backend = CpuLiquidSubstepBackend3D();
  backend.commit_growth = nullptr;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), backend).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}
#if defined(_WIN32) && defined(_DEBUG)
TEST(LiquidSubstep3D, BackendGrowthDiscardRunsAfterActualHostReservationFailure) {
  auto initial = Initial();
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  BackendLifecycleObservation observed;
  auto backend = ObservedBackend(observed);
  backend.prepare_growth = [](void* context, LiquidStateCapacities3D, const LiquidState3D& start) {
    auto& observation = *static_cast<BackendLifecycleObservation*>(context);
    ++observation.growth_prepared;
    observation.growth_clock = start.clock;
    scheduler_allocations = 0;
    scheduler_fail_allocation = 1;
    scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
    return absl::OkStatus();
  };
  backend.discard_growth = [](void* context) noexcept {
    _CrtSetAllocHook(scheduler_previous_hook);
    scheduler_fail_allocation = 0;
    ++static_cast<BackendLifecycleObservation*>(context)->growth_discarded;
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options, backend);
  ASSERT_TRUE(scheduler.ok());
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kEmit
                                                    && ++*static_cast<int*>(context) == 2
                                                ? absl::Status(absl::StatusCode::kResourceExhausted, {})
                                                : absl::OkStatus();
                                          }});
  const auto result = scheduler->AdvanceFrame(0.01);
  _CrtSetAllocHook(scheduler_previous_hook);
  scheduler_fail_allocation = 0;
  EXPECT_EQ(result.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(scheduler_allocations, 1);
  EXPECT_EQ(observed.growth_prepared, 1);
  EXPECT_EQ(observed.growth_committed, 0);
  EXPECT_EQ(observed.growth_discarded, 1);
  EXPECT_EQ(observed.growth_clock, initial.clock);
  EXPECT_EQ(scheduler->committed().clock.time, 0.005);
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
}
TEST(LiquidSubstep3D, BackendPublicationFailureAllocatesAndFreesNothing) {
  auto initial = Initial();
  BackendLifecycleObservation observed;
  observed.fail_publication = true;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  scheduler_allocations = 0;
  scheduler_frees = 0;
  bool rejected = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    const auto result = scheduler->Attempt(0.01);
    rejected = result.ok() && result->disposition == LiquidAttemptDisposition3D::kFatal;
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_TRUE(rejected);
  EXPECT_EQ(scheduler_allocations, 0);
  EXPECT_EQ(scheduler_frees, 0);
  EXPECT_EQ(observed.discarded, 1);
  EXPECT_EQ(observed.committed, 0);
  EXPECT_TRUE(scheduler->committed() == initial);
}
#endif
}  // namespace
}  // namespace fluidity
