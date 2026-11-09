#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include "../src/liquid_substep_3d_internal.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_substep_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "gtest/gtest.h"
#include "liquid_exact_curvature_3d_cases.h"
#include "liquid_pressure_supported_gather_3d_cases.h"
#include "liquid_tank_staging_3d_cases.h"
#include "liquid_wall_sample_3d_cases.h"
#include "wetted_piston_pressure_3d_cases.h"
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
// Serialize through the production checkpoint codec so storage omitted by a
// hand-picked field comparison cannot pass the F9 bridge.
void ExpectIdenticalSerializedState(const LiquidState3D& before, const LiquidState3D& after) {
  const auto before_bytes = EncodeLiquidState3D(before);
  const auto after_bytes = EncodeLiquidState3D(after);
  ASSERT_TRUE(before_bytes.ok()) << before_bytes.status();
  ASSERT_TRUE(after_bytes.ok()) << after_bytes.status();
  const auto before_digest = LiquidContentDigest3D(before_bytes->values());
  const auto after_digest = LiquidContentDigest3D(after_bytes->values());
  EXPECT_TRUE(std::ranges::equal(before_bytes->values(), after_bytes->values()));
  EXPECT_EQ(before_digest, after_digest);
  std::cout << "f9_state_before=" << LiquidDigestHex3D(before_digest).data()
            << " f9_state_after=" << LiquidDigestHex3D(after_digest).data() << '\n';
}

TEST(LiquidSubstep3D, WettedPistonUsesClippedAndAttachedAreaInProjection) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wet = wetted_piston_pressure_test;
  for (const double height : {0.75, 1.125}) {
    for (const double speed : {wet::kPistonSpeed, -wet::kPistonSpeed, 0.0}) {
      SCOPED_TRACE(::testing::Message() << height << " " << speed);
      wet::ProjectionAudit audit;
      auto scheduler = LiquidSubstepScheduler3D::Create(
          {},
          wet::PlaneScene(height, speed),
          Options(),
          wet::Observe(audit, CpuLiquidSubstepBackend3D())
      );
      ASSERT_TRUE(scheduler.ok()) << scheduler.status();
      const auto status = scheduler->AdvanceFrame(wet::kDuration);
      ASSERT_TRUE(status.ok()) << status;
      wet::CheckPlaneProjection(*scheduler, audit, height, speed);
    }
  }
}

TEST(LiquidSubstep3D, WettedPistonRejectsWetEmptyPressureWithoutRetry) {
  namespace wet = wetted_piston_pressure_test;
  const LiquidState3D initial;
  auto scene = wet::PlaneScene(0.25, wet::kPistonSpeed);
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  EXPECT_EQ(scheduler->AdvanceFrame(wet::kDuration).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 0);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
  wet::ExpectCheckpointEqual(initial, scheduler->committed());
  scene.field = LiquidParticleSphereUnion3D{0.25};
  auto dry = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(dry.ok()) << dry.status();
  ASSERT_TRUE(dry->AdvanceFrame(wet::kDuration).ok());
  ASSERT_TRUE(dry->diagnostics().accepted_measurement.has_value());
  EXPECT_DOUBLE_EQ(dry->diagnostics().accepted_measurement->step_start_inward_piston_flux_m3_per_s, 0);
  EXPECT_EQ(dry->committed().clock.epoch, 1);
}

TEST(LiquidSubstep3D, WettedPistonRejectsArithmeticOverflowAfterFreeze) {
  namespace wet = wetted_piston_pressure_test;
  const LiquidState3D initial;
  // The legacy 1 m² face flux is finite. The 1.125 m² wet wall
  // overflows signed flux even though the piston advances less than half a cell.
  wet::ProjectionAudit audit;
  auto scheduler = LiquidSubstepScheduler3D::Create(
      initial,
      wet::PlaneScene(1.125, wet::kOverflowPistonSpeed),
      Options(),
      wet::Observe(audit, CpuLiquidSubstepBackend3D())
  );
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(wet::kOverflowStepDuration).ok());
  const auto result = scheduler->Attempt(wet::kOverflowStepDuration);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kInterface);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kOutOfRange);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 0);
  EXPECT_EQ(audit.wetted_stage_calls, 1);
  EXPECT_EQ(audit.projection_count, 0);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
  wet::ExpectCheckpointEqual(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, WettedPistonReplaysAfterGrowth) {
  namespace wet = wetted_piston_pressure_test;
  const auto scene = wet::PlaneScene(1.125, wet::kPistonSpeed);
  auto small = Options();
  small.capacities.faces = 64;
  small.capacity_replay_reservations = Options().capacities;
  auto replay = LiquidSubstepScheduler3D::Create({}, scene, small);
  ASSERT_TRUE(replay.ok()) << replay.status();
  const auto advanced = replay->AdvanceFrame(wet::kDuration);
  ASSERT_TRUE(advanced.ok()) << advanced;
  EXPECT_EQ(replay->diagnostics().capacity_replays, 1);
  auto direct = LiquidSubstepScheduler3D::Create({}, scene, Options());
  ASSERT_TRUE(direct.ok()) << direct.status();
  ASSERT_TRUE(direct->AdvanceFrame(wet::kDuration).ok());
  wet::ExpectCheckpointEqual(direct->committed(), replay->committed());
}

TEST(LiquidSubstep3D, WettedPistonRequiresSelectedOperationOnlyForMovingTank) {
  namespace wet = wetted_piston_pressure_test;
  auto operations = CpuLiquidSubstepBackend3D();
  operations.wetted_piston = nullptr;
  auto scene = wet::PlaneScene(0.75, wet::kPistonSpeed);
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create({}, scene, Options(), operations).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  scene.obstacle = wet::PlaneScene(0.75, 0).obstacle;
  auto stationary = LiquidSubstepScheduler3D::Create({}, scene, Options(), operations);
  ASSERT_TRUE(stationary.ok()) << stationary.status();
  EXPECT_TRUE(stationary->AdvanceFrame(wet::kDuration).ok());
  auto absent = LiquidSubstepScheduler3D::Create(Initial(), Scene(), Options(), operations);
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_TRUE(absent->AdvanceFrame(wet::kDuration).ok());
}

TEST(LiquidSubstep3D, PressureSupportedGatherExcludesForceOnlyNewDonors) {
  namespace fixture = pressure_supported_gather_test;
  auto options = Options();
  options.maximum_step_duration = fixture::kDuration;
  fixture::Audit audit(options.capacities.faces);
  fixture::active_audit = &audit;
  auto scheduler = LiquidSubstepScheduler3D::Create(
      fixture::Initial(),
      fixture::Scene(),
      options,
      fixture::Observe(audit, CpuLiquidSubstepBackend3D())
  );
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(fixture::kDuration);
  ASSERT_TRUE(status.ok()) << status;
  fixture::ExpectSymmetricPressureCell(*scheduler, audit);
}

TEST(LiquidSubstep3D, PressureSupportedGatherPreservesDisconnectedFreeFall) {
  namespace fixture = pressure_supported_gather_test;
  auto options = Options();
  options.maximum_step_duration = fixture::kDuration;
  fixture::ExpectMixedPressureDoesNotChangeUnsupportedGather(
      CpuLiquidSubstepBackend3D(),
      CpuLiquidSubstepBackend3D(),
      options
  );
}

TEST(LiquidSubstep3D, PressureSupportedGatherUsesHalfOpenContainingCell) {
  namespace fixture = pressure_supported_gather_test;
  auto options = Options();
  options.maximum_step_duration = fixture::kDuration;
  for (const double x : {std::nextafter(-1.0, -2.0), -1.0}) {
    fixture::ExpectSignedBoundaryMembership(CpuLiquidSubstepBackend3D(), options, x);
  }
  fixture::ExpectTranslatedMembership(CpuLiquidSubstepBackend3D(), options);
}

TEST(LiquidSubstep3D, CheckedContainingCellCertifiesFullSignedRange) {
  constexpr UniformMacGeometry3D geometry{{}, 1};
  constexpr double min_index = std::numeric_limits<std::int32_t>::min();
  constexpr double max_index = std::numeric_limits<std::int32_t>::max();
  const auto cell = liquid_step_internal::CheckedContainingCell(geometry, {min_index, max_index + 0.5, -1});
  ASSERT_TRUE(cell.ok()) << cell.status();
  EXPECT_EQ(cell->x, INT32_MIN);
  EXPECT_EQ(cell->y, INT32_MAX);
  EXPECT_EQ(cell->z, -1);
  for (const double x : {std::nextafter(min_index, min_index - 1), max_index + 1}) {
    EXPECT_EQ(
        liquid_step_internal::CheckedContainingCell(geometry, {x, 0, 0}).status().code(),
        absl::StatusCode::kOutOfRange
    );
  }
  // Compensated product arithmetic cannot certify the sign at this tiny
  // scale. The helper fails closed before converting a candidate index.
  const UniformMacGeometry3D tiny_grid{{}, 0x1p-970};
  EXPECT_EQ(
      liquid_step_internal::CheckedContainingCell(tiny_grid, {0x1p-969, 0, 0}).status().code(),
      absl::StatusCode::kOutOfRange
  );
}

TEST(LiquidSubstep3D, PressureSupportedGatherPreservesLowMassStatus) {
  namespace fixture = pressure_supported_gather_test;
  auto options = Options();
  options.maximum_step_duration = fixture::kDuration;
  fixture::ExpectPressureMaskKeepsLowMass(CpuLiquidSubstepBackend3D(), options);
}

TEST(LiquidSubstep3D, PressureSupportedGatherWithNoNewDonorRollsBack) {
  namespace fixture = pressure_supported_gather_test;
  const auto initial = fixture::NoDonorInitial();
  auto options = Options();
  options.maximum_step_duration = fixture::kDuration;
  options.minimum_face_mass = 1.125;
  fixture::Audit audit(options.capacities.faces);
  fixture::active_audit = &audit;
  auto scheduler = LiquidSubstepScheduler3D::Create(
      initial,
      fixture::Scene(),
      options,
      fixture::Observe(audit, CpuLiquidSubstepBackend3D())
  );
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(fixture::kDuration);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  fixture::ExpectNoDonorRollback(*scheduler, initial, audit);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, CompleteNonaffineGatherKeepsTrueGradientAndNativeNoDonorFailure) {
  pressure_supported_gather_test::ExpectCompleteNonaffineGradient(CpuLiquidSubstepBackend3D(), Options().capacities);
}

TEST(LiquidSubstep3D, DeduplicatesSignedFieldKeysBeforeSampling) {
  auto operations = CpuLiquidSubstepBackend3D();
  std::size_t evaluations = 0;
  operations.context = &evaluations;
  operations.sample_field_batch = [](void* context,
                                     const LiquidSceneField3D& field,
                                     std::span<const LiquidParticleState3D> particles,
                                     std::span<const Vec3> positions,
                                     std::span<double> distances) {
    *static_cast<std::size_t*>(context) += positions.size();
    return CpuLiquidSubstepBackend3D().sample_field_batch(nullptr, field, particles, positions, distances);
  };
  ASSERT_NO_FATAL_FAILURE(tank_staging_test::ExpectMixedSignedFieldSupport(operations, 0));
  EXPECT_EQ(evaluations, 262);
}

TEST(LiquidSubstep3D, CheckedFieldBatchesPreserveOutputs) {
  tank_staging_test::ExpectFieldBatchContract(CpuLiquidSubstepBackend3D());
}

TEST(LiquidSubstep3D, LateMalformedFieldBatchPreservesCommittedBytes) {
  const auto initial = Initial();
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 2.1};
  auto operations = CpuLiquidSubstepBackend3D();
  std::size_t batches = 0;
  operations.context = &batches;
  operations.sample_field_batch = [](void* context,
                                     const LiquidSceneField3D& field,
                                     std::span<const LiquidParticleState3D> particles,
                                     std::span<const Vec3> positions,
                                     std::span<double> distances) {
    if (++*static_cast<std::size_t*>(context) == 2) {
      distances = distances.first(distances.size() - 1);
    }
    return CpuLiquidSubstepBackend3D().sample_field_batch(nullptr, field, particles, positions, distances);
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), operations);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(batches, 2);
  ExpectIdenticalSerializedState(initial, scheduler->committed());

  operations.sample_field_batch = nullptr;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, scene, Options(), operations).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidSubstep3D, HorizontalPlaneActivatesSignedPressureAndFaceOwnersWithoutParticles) {
  tank_staging_test::ExpectPlaneSupport(CpuLiquidSubstepBackend3D(), Options());
}

TEST(LiquidSubstep3D, TranslatedHorizontalPlaneRetainsWetPressureCell) {
  tank_staging_test::ExpectTranslatedPlaneKeepsWetCell(CpuLiquidSubstepBackend3D(), Options());
}

TEST(LiquidSubstep3D, OpenTankSamplesPistonAndFixedWallsSeparately) {
  tank_staging_test::ExpectWallSamples(CpuLiquidSubstepBackend3D());
}

TEST(LiquidSubstep3D, OpenTankGhostTagPreservesTranslatedFaceResidual) {
  tank_staging_test::ExpectTranslatedFaceUsesLatticeResidual(CpuLiquidSubstepBackend3D());
}

TEST(LiquidSubstep3D, OpenTankConstrainedUpdateKeepsIndependentVelocityFields) {
  tank_staging_test::ExpectParticleUpdateUsesBothVelocityFields(CpuLiquidSubstepBackend3D());
}

TEST(LiquidSubstep3D, TankPoseClosureLeavesCommittedStateIntact) {
  auto scene = Scene();
  scene.field = LiquidHorizontalPlaneSurface3D{3};
  scene.obstacle = AxisAlignedLiquidTank3D{1.75, 3.5, 2, 1.5, 3.5, 2};
  const auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(1).ok());
  const auto result = scheduler->Attempt(1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 0);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, WettedPistonRefusesUncertifiedUnownedBroadSphere) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wet = wetted_piston_pressure_test;
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{1.1};
  scene.obstacle = wet::kBroadUncertifiedSphereTank;
  const auto initial = Initial();
  wet::ProjectionAudit audit;
  auto scheduler =
      LiquidSubstepScheduler3D::Create(initial, scene, Options(), wet::Observe(audit, CpuLiquidSubstepBackend3D()));
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kInterface);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 0);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
  wet::ExpectCheckpointEqual(initial, scheduler->committed());
  wet::ExpectBroadSphereRefusal(audit);
}

TEST(LiquidSubstep3D, AdvancesOpenTankWithTransportedField) {
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{wetted_piston_pressure_test::kSphereRadius};
  scene.obstacle = wetted_piston_pressure_test::kNarrowSphereTank;
  auto initial = Initial();
  initial.particles[0].world_position.z = 2.75;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(status.ok()) << status << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
  EXPECT_FALSE(scheduler->committed().cells.values().empty());
  EXPECT_EQ(scheduler->committed().particles[0].side_tag, 0);
  wetted_piston_pressure_test::ExpectSphereWallFlux(*scheduler);
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
  // The particle is near the continuous wet surface while its containing
  // cell center is dry. Its complete history can retain the old PIC residual.
  scene.field = LiquidAnalyticSphere3D{{2.3, 2.5, 2.5}, 1.1};
  scene.target_divergence = 0.3;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_GT(scheduler->committed().particles[0].volume_ratio, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].mass, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].reference_volume, 1);
}

TEST(LiquidSubstep3D, AnalyticSphereSuppliesExactExteriorPressureAtBothResolutions) {
  exact_curvature_test::ExpectAnalyticSphereExteriorPressure(CpuLiquidSubstepBackend3D());
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
  const auto& measurement = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement.has_value());
  EXPECT_EQ(measurement->accepted_clock, scheduler->committed().clock);
  EXPECT_DOUBLE_EQ(measurement->accepted_dt_seconds, 0.01);
  ASSERT_TRUE(measurement->step_start_capillary_bound_seconds.has_value());
  EXPECT_NEAR(*measurement->step_start_capillary_bound_seconds, 28.209479177387813, 1e-12);
  EXPECT_GT(measurement->pressure_rhs_l2_m3_per_s, 0);
  EXPECT_FALSE(measurement->piston_pose.has_value());
  EXPECT_DOUBLE_EQ(measurement->step_start_inward_piston_flux_m3_per_s, 0);
}
class LiquidStageFailure3D : public testing::TestWithParam<int> {};
TEST_P(LiquidStageFailure3D, CommittedInventoryUnchanged) {
  auto initial = Initial();
  initial.random_seed = 0xfedcba9876543210ULL;
  initial.sources = *LiquidRecords3D<LiquidSourceState3D>::Create(1);
  ASSERT_TRUE(initial.sources.Resize(1).ok());
  initial.sources[0] = {0, 8, 0.25};
  LiquidEmissionSource3D source;
  source.particles_per_second = 130;
  source.world_position = {2.5, 2.5, 2.5};
  source.world_position_jitter = {0.1, 0.1, 0.1};
  auto scene = Scene();
  scene.sources = std::span(&source, 1);
  scene.ambient_pressure = 3;
  scene.surface_tension = 0.1;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  auto snapshot = CloneLiquidState3D(scheduler->committed(), Options().capacities);
  ASSERT_TRUE(snapshot.ok());
  ASSERT_EQ(snapshot->random_seed, initial.random_seed);
  ASSERT_GT(snapshot->next_particle_id, initial.next_particle_id);
  ASSERT_GT(snapshot->sources[0].next_child_ordinal, initial.sources[0].next_child_ordinal);
  ASSERT_GT(snapshot->sources[0].fractional_remainder, 0);
  ASSERT_FALSE(snapshot->blocks.values().empty());
  ASSERT_TRUE(std::ranges::any_of(snapshot->cells.values(), [](const LiquidCellState3D& cell) {
    return cell.previous_pressure != 0;
  }));
  ASSERT_TRUE(std::ranges::any_of(snapshot->faces.values(), [](const LiquidFaceState3D& face) {
    return face.carrier_velocity != 0 && face.validity != LiquidFaceValidity3D::kMissing;
  }));
  struct RejectionProbe {
    LiquidPhysicalStage3D selected_stage;
    std::size_t injection_count = 0;
  } rejection_probe{.selected_stage = static_cast<LiquidPhysicalStage3D>(GetParam())};
  scheduler->SetStageFaultHookForTesting({
      &rejection_probe,
      [](void* context, LiquidPhysicalStage3D current_stage, double) {
        auto& probe = *static_cast<RejectionProbe*>(context);
        if (current_stage == probe.selected_stage) {
          ++probe.injection_count;
          return absl::InternalError("");
        }
        return absl::OkStatus();
      },
  });
  ASSERT_TRUE(scheduler->BeginFrame(0.02).ok());
  auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_TRUE(scheduler->committed() == *snapshot);
  ExpectIdenticalSerializedState(*snapshot, scheduler->committed());
  EXPECT_EQ(rejection_probe.injection_count, 1);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, rejection_probe.selected_stage);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kInternal);
}
INSTANTIATE_TEST_SUITE_P(
    LiquidSubstep3D,
    LiquidStageFailure3D,
    testing::Range(0, static_cast<int>(LiquidPhysicalStage3D::kAdapt))
);
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
  namespace wall_test = liquid_wall_samples_test;
  const auto initial = wall_test::SlabParticles(2);
  const auto scene = wall_test::SlabScene();
  auto reference = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(reference.ok());
  ASSERT_TRUE(reference->AdvanceFrame(0.01).ok());
  wall_test::Audit audit(options.capacities.faces);
  auto scheduler =
      LiquidSubstepScheduler3D::Create(initial, scene, options, wall_test::Observe(audit, CpuLiquidSubstepBackend3D()));
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
  wall_test::ExpectGrowthRefreshes(audit);
  ExpectIdenticalSerializedState(reference->committed(), scheduler->committed());
}
TEST(LiquidSubstep3D, ProgressObserverInvalidatesDiscardedPassAndPreservesSchedule) {
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  auto reference = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(reference.ok());
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            if (stage == LiquidPhysicalStage3D::kEmit
                                                && ++*static_cast<int*>(context) == 2) {
                                              return absl::ResourceExhaustedError("");
                                            }
                                            return absl::OkStatus();
                                          }});
  std::vector<std::pair<std::uint64_t, bool>> observations;
  ASSERT_TRUE(scheduler
                  ->AdvanceFrame(
                      0.01,
                      {&observations,
                       [](void* context, const LiquidState3D& state, const LiquidStepDiagnostics3D&, bool rollback) {
                         static_cast<decltype(observations)*>(context)->emplace_back(state.clock.epoch, rollback);
                         return absl::OkStatus();
                       }}
                  )
                  .ok());
  EXPECT_EQ(observations, (decltype(observations){{1, false}, {0, true}, {1, false}, {2, false}}));
  ASSERT_TRUE(reference->AdvanceFrame(0.01).ok());
  EXPECT_TRUE(scheduler->committed() == reference->committed());
}
TEST(LiquidSubstep3D, ProgressStopRetainsOneAcceptedSubstepWithoutFrameCompletion) {
  auto options = Options();
  options.maximum_step_duration = 0.005;
  auto scheduler = LiquidSubstepScheduler3D::Create(Initial(), Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(
      scheduler
          ->AdvanceFrame(
              0.01,
              {nullptr,
               [](void*, const LiquidState3D& state, const LiquidStepDiagnostics3D& diagnostics, bool rollback) {
                 EXPECT_FALSE(rollback);
                 EXPECT_EQ(state.clock.epoch, 1);
                 EXPECT_EQ(diagnostics.accepted_attempts, 1);
                 return absl::CancelledError("");
               }}
          )
          .code(),
      absl::StatusCode::kCancelled
  );
  EXPECT_EQ(scheduler->committed().clock.frame_ordinal, 0);
  EXPECT_EQ(scheduler->committed().clock.substep_ordinal, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().clock.time, 0.005);
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

TEST(LiquidSubstep3D, ReusesWallSamplesInCanonicalSlabSideRuns) {
  namespace wall_test = liquid_wall_samples_test;
  const auto options = Options();
  for (const std::size_t particle_count : {0, 2, 5}) {
    SCOPED_TRACE(particle_count);
    const auto initial = wall_test::SlabParticles(particle_count);
    wall_test::Audit audit(options.capacities.faces);
    auto scheduler = LiquidSubstepScheduler3D::Create(
        initial,
        wall_test::SlabScene(),
        options,
        wall_test::Observe(audit, CpuLiquidSubstepBackend3D())
    );
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
    const auto calls = wall_test::SlabCalls(2.501);
    const std::size_t expected_call_count = particle_count == 0 ? 0 : particle_count == 2 ? 6 : calls.size();
    wall_test::ExpectCalls(audit, std::span(calls).first(expected_call_count));
    EXPECT_EQ(audit.gather_count, particle_count);
    EXPECT_EQ(scheduler->diagnostics().old_pic_fallback_count, audit.old_fallbacks);
    EXPECT_EQ(scheduler->diagnostics().new_pic_fallback_count, audit.new_fallbacks);
    for (std::size_t index = 0; index < particle_count; ++index) {
      const auto& particle = scheduler->committed().particles[index];
      EXPECT_EQ(particle.id, index + 1);
      EXPECT_EQ(particle.side_tag, wall_test::kSides[index]);
      if (particle.side_tag < 0) {
        EXPECT_LT(particle.world_position.x, 2.45);
      } else {
        EXPECT_GT(particle.world_position.x, 2.551);
      }
    }
  }
}

TEST(LiquidSubstep3D, FailedOldOrNewWallRefreshRetainsOnlyEarlierDiagnostics) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wall_test = liquid_wall_samples_test;
  const auto options = Options();
  const auto initial = wall_test::SlabParticles();
  const auto scene = wall_test::SlabScene();
  for (const bool fail_new : {false, true}) {
    SCOPED_TRACE(fail_new);
    wall_test::Audit audit(options.capacities.faces);
    auto scheduler = LiquidSubstepScheduler3D::Create(
        initial,
        scene,
        options,
        wall_test::Observe(audit, CpuLiquidSubstepBackend3D())
    );
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    audit.fail_at_wall_coordinate = fail_new ? 2.501 : 2.5;
    const auto outcome = scheduler->Attempt(0.01);
    ASSERT_TRUE(outcome.ok()) << outcome.status();
    EXPECT_EQ(outcome->disposition, LiquidAttemptDisposition3D::kRetryDt);
    const auto failed_calls = wall_test::SlabCalls(2.501);
    wall_test::ExpectCalls(audit, std::span(failed_calls).first(fail_new ? 8 : 7));
    EXPECT_EQ(audit.gather_count, 2);
    EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kGather);
    EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kAborted);
    EXPECT_EQ(scheduler->diagnostics().old_pic_fallback_count, audit.old_fallbacks);
    EXPECT_EQ(scheduler->diagnostics().new_pic_fallback_count, audit.new_fallbacks);
    EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
    ExpectIdenticalSerializedState(initial, scheduler->committed());

    audit.Reset();
    audit.fail_at_wall_coordinate = 0;
    const auto retried = scheduler->Attempt(0.005);
    ASSERT_TRUE(retried.ok()) << retried.status();
    ASSERT_EQ(retried->disposition, LiquidAttemptDisposition3D::kAccepted);
    wall_test::ExpectCalls(audit, wall_test::SlabCalls(2.5005));
    auto reference = LiquidSubstepScheduler3D::Create(initial, scene, options);
    ASSERT_TRUE(reference.ok());
    ASSERT_TRUE(reference->BeginFrame(0.01).ok());
    const auto expected = reference->Attempt(0.005);
    ASSERT_TRUE(expected.ok());
    ASSERT_EQ(expected->disposition, LiquidAttemptDisposition3D::kAccepted);
    ExpectIdenticalSerializedState(reference->committed(), scheduler->committed());
  }
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
  std::cout << "CTEST_FULL_OUTPUT\n";
  auto initial = Initial();
  initial.random_seed = 123456789;
  initial.sources = *LiquidRecords3D<LiquidSourceState3D>::Create(1);
  ASSERT_TRUE(initial.sources.Resize(1).ok());
  initial.sources[0] = {0, 3, 0.25};
  LiquidEmissionSource3D source;
  source.particles_per_second = 400;
  source.world_position = {2.5, 2.5, 2.5};
  source.world_position_jitter = {0.2, 0.2, 0.2};
  auto scene = Scene();
  scene.sources = std::span(&source, 1);
  scene.ambient_pressure = 3;
  scene.surface_tension = 0.1;
  auto options = Options();
  // Dyadic times keep the retried and uninterrupted schedules bit-identical.
  options.maximum_step_duration = 0.0078125;
  auto bootstrap = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(bootstrap.ok());
  ASSERT_TRUE(bootstrap->AdvanceFrame(0.0078125).ok());
  const auto& warm_start = bootstrap->committed();
  ASSERT_TRUE(std::ranges::any_of(warm_start.cells.values(), [](const LiquidCellState3D& cell) {
    return cell.previous_pressure != 0;
  }));
  auto reference = LiquidSubstepScheduler3D::Create(warm_start, scene, options);
  ASSERT_TRUE(reference.ok());
  ASSERT_TRUE(reference->AdvanceFrame(0.0234375).ok());
  options.maximum_step_duration = 0.015625;
  auto retry = LiquidSubstepScheduler3D::Create(warm_start, scene, options);
  ASSERT_TRUE(retry.ok());
  int acceptance_stage_count = 0;
  retry->SetStageFaultHookForTesting({&acceptance_stage_count, [](void* context, LiquidPhysicalStage3D stage, double) {
                                        if (stage == LiquidPhysicalStage3D::kAccept
                                            && ++*static_cast<int*>(context) == 1) {
                                          return absl::Status(absl::StatusCode::kAborted, {});
                                        }
                                        return absl::OkStatus();
                                      }});
  ASSERT_TRUE(retry->AdvanceFrame(0.0234375).ok());
  ASSERT_EQ(retry->diagnostics().dt_retries, 1);
  EXPECT_TRUE(retry->committed() == reference->committed());
  ExpectIdenticalSerializedState(reference->committed(), retry->committed());
  // Dyadic times keep the retried and uninterrupted schedules bit-identical.
  options.maximum_step_duration = 0.0078125;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  auto replay = LiquidSubstepScheduler3D::Create(warm_start, scene, options);
  ASSERT_TRUE(replay.ok());
  acceptance_stage_count = 0;
  replay->SetStageFaultHookForTesting({&acceptance_stage_count, [](void* context, LiquidPhysicalStage3D stage, double) {
                                         if (stage == LiquidPhysicalStage3D::kAccept
                                             && ++*static_cast<int*>(context) == 2) {
                                           return absl::Status(absl::StatusCode::kResourceExhausted, {});
                                         }
                                         return absl::OkStatus();
                                       }});
  ASSERT_TRUE(replay->AdvanceFrame(0.0234375).ok());
  ASSERT_EQ(replay->diagnostics().capacity_replays, 1);
  EXPECT_TRUE(replay->committed() == reference->committed());
  ExpectIdenticalSerializedState(reference->committed(), replay->committed());
  std::cout << "f9_retry_attempts=" << retry->diagnostics().attempts
            << " f9_replay_attempts=" << replay->diagnostics().attempts
            << " f9_pressure_iterations=" << replay->diagnostics().pressure_diagnostics.iteration_count
            << " f9_reserved_bytes=" << replay->reserved_bytes() << '\n';
}
TEST(LiquidSubstep3D, RepeatedCapacityFailureRetainsSerializedFrameStart) {
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.005).ok());
  auto frame_start = CloneLiquidState3D(scheduler->committed(), options.capacities);
  ASSERT_TRUE(frame_start.ok());

  int acceptance_stage_count = 0;
  scheduler->SetStageFaultHookForTesting({
      &acceptance_stage_count,
      [](void* context, LiquidPhysicalStage3D stage, double) {
        if (stage == LiquidPhysicalStage3D::kAccept && ++*static_cast<int*>(context) >= 2) {
          return absl::ResourceExhaustedError("");
        }
        return absl::OkStatus();
      },
  });
  EXPECT_EQ(scheduler->AdvanceFrame(0.015).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(acceptance_stage_count, 3);
  EXPECT_EQ(scheduler->diagnostics().accepted_attempts, 1);
  EXPECT_EQ(scheduler->diagnostics().capacity_replays, 1);
  ExpectIdenticalSerializedState(*frame_start, scheduler->committed());
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
  // M1 adaptation is an identity stage recorded after publication. Closing
  // the schedule cannot install another committed version.
  ASSERT_GT(scheduler->diagnostics().stage_count, 0);
  EXPECT_EQ(scheduler->diagnostics().stages[scheduler->diagnostics().stage_count - 1], LiquidPhysicalStage3D::kAdapt);
  EXPECT_TRUE(scheduler->committed() == *snapshot);
  ExpectIdenticalSerializedState(*snapshot, scheduler->committed());
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
    bool new_pic_fallback = false;
  } observed;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observed;
  backend.update_particles =
      [](void* c, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
        const auto& value = inputs.gathered[0];
        *static_cast<Observation*>(c) = {
            value.velocity_gradient.x_row.x + value.velocity_gradient.y_row.y + value.velocity_gradient.z_row.z,
            value.new_pic_velocity.x,
            value.particle_minus_old_pic_velocity.x,
            value.new_pic_fallback,
        };
        return CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
      };
  auto options = Options();
  options.particle_energy_blend = 0.6;
  // Both points are inside the continuous wet surface. At center x=2.3 the
  // containing cell center is dry, so its complete history can retain the
  // old PIC residual. At x=2.5 projected donors have insufficient rank.
  for (const bool pressure_supported : {false, true}) {
    SCOPED_TRACE(pressure_supported);
    scene.field = LiquidAnalyticSphere3D{{pressure_supported ? 2.5 : 2.3, 2.5, 2.5}, 1.1};
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
    ASSERT_TRUE(scheduler.ok());
    ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
    ASSERT_GT(observed.divergence, 0);
    EXPECT_EQ(observed.new_pic_fallback, pressure_supported);
    if (pressure_supported) {
      EXPECT_DOUBLE_EQ(observed.residual, 0);
    } else {
      EXPECT_GT(observed.residual, 0.5);
    }
    const auto& particle = scheduler->committed().particles[0];
    EXPECT_NEAR(particle.volume_ratio, std::exp(0.01 * observed.divergence), 1e-13);
    EXPECT_NEAR(particle.world_velocity.x, observed.pic + 0.594 * observed.residual, 1e-12);
    EXPECT_NEAR(particle.world_position.x, 3.3 + 0.01 * (observed.pic + 0.6 * 0.594 * observed.residual), 1e-12);
  }
}
TEST(LiquidSubstep3D, CompleteBackendIsRequiredAndRealPressureNonconvergenceRetries) {
  auto initial = Initial(8);
  for (std::size_t i = 0; i < 8; ++i) {
    auto& p = initial.particles[i];
    p.world_position = {1.5 + 2 * (i & 1), 1.5 + 2 * ((i >> 1) & 1), 1.5 + 2 * ((i >> 2) & 1)};
    p.world_velocity = {
        0.1 * static_cast<double>(i),
        0.03 * static_cast<double>(i * i),
        -0.07 * static_cast<double>(i),
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
  std::vector<LiquidAttemptMeasurement3D> attempts;
  const LiquidProgressObserver3D observer{
      .context = &attempts,
      .observe_attempt =
          [](void* context, const LiquidAttemptMeasurement3D& attempt) {
            static_cast<decltype(attempts)*>(context)->push_back(attempt);
          },
  };
  EXPECT_EQ(scheduler->AdvanceFrame(0.01, observer).code(), absl::StatusCode::kAborted);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  EXPECT_EQ(scheduler->diagnostics().attempts, 3);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
  EXPECT_TRUE(scheduler->committed() == initial);
  ASSERT_EQ(attempts.size(), 3);
  constexpr std::array attempted_durations{0.01, 0.005, 0.0025};
  for (std::size_t index = 0; index < attempts.size(); ++index) {
    const auto& attempt = attempts[index];
    EXPECT_EQ(attempt.start_clock, initial.clock);
    EXPECT_EQ(attempt.attempt_in_frame, index + 1);
    EXPECT_EQ(attempt.capacity_pass, 0);
    EXPECT_DOUBLE_EQ(attempt.attempted_dt_seconds, attempted_durations[index]);
    EXPECT_TRUE(std::isfinite(attempt.work_wall_seconds));
    EXPECT_GE(attempt.work_wall_seconds, 0);
    EXPECT_EQ(attempt.failure_code, absl::StatusCode::kAborted);
    if (index + 1 == attempts.size()) {
      EXPECT_EQ(attempt.disposition, LiquidAttemptDisposition3D::kFatal);
      EXPECT_FALSE(attempt.retry_dt_seconds.has_value());
    } else {
      EXPECT_EQ(attempt.disposition, LiquidAttemptDisposition3D::kRetryDt);
      ASSERT_TRUE(attempt.retry_dt_seconds.has_value());
      EXPECT_DOUBLE_EQ(*attempt.retry_dt_seconds, attempted_durations[index + 1]);
    }
  }
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
TEST(LiquidSubstep3D, WettedPistonAttemptsAllocateAndFreeNothing) {
  namespace wet = wetted_piston_pressure_test;
  for (const bool overflow : {false, true}) {
    const double speed = overflow ? wet::kOverflowPistonSpeed : wet::kPistonSpeed;
    const double duration = overflow ? wet::kOverflowStepDuration : wet::kDuration;
    auto scheduler = LiquidSubstepScheduler3D::Create({}, wet::PlaneScene(1.125, speed), Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(duration).ok());
    scheduler_allocations = 0;
    scheduler_frees = 0;
    bool completed = false;
    auto disposition = LiquidAttemptDisposition3D::kFatal;
    scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
    {
      const auto result = scheduler->Attempt(duration);
      completed = result.ok();
      if (completed) {
        disposition = result->disposition;
      }
    }
    _CrtSetAllocHook(scheduler_previous_hook);
    ASSERT_TRUE(completed);
    EXPECT_EQ(disposition, overflow ? LiquidAttemptDisposition3D::kFatal : LiquidAttemptDisposition3D::kAccepted);
    EXPECT_EQ(scheduler_allocations, 0);
    EXPECT_EQ(scheduler_frees, 0);
  }
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
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}
TEST(LiquidSubstep3D, CandidateValidationRejectsPoisonAfterActualNumericalUpdate) {
  auto initial = Initial();
  std::size_t poison_injection_count = 0;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &poison_injection_count;
  backend.update_particles =
      [](void* context, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
        const auto status = CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
        if (status.ok()) {
          particles[0].volume_ratio = std::numeric_limits<double>::quiet_NaN();
          ++*static_cast<std::size_t*>(context);
        }
        return status;
      };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(poison_injection_count, 1);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kValidate);
  EXPECT_TRUE(scheduler->committed() == initial);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
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
        -0.07 * static_cast<double>(i),
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
