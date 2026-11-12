#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include "../src/liquid_adaptive_family_candidate_3d.h"
#include "../src/liquid_adaptive_step_3d.h"
#include "../src/liquid_substep_3d_internal.h"
#include "../src/viscosity_velocity_block_3d.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_substep_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "fluidity/host/solid_interaction_checkpoint_3d.h"
#include "graded_tank_fixture.h"
#include "gtest/gtest.h"
#include "liquid_cavity_scheduler_cases.h"
#include "liquid_curved_phase_cases.h"
#include "liquid_eos_hydrostatic_cases.h"
#include "liquid_regular_cell_transport_3d_cases.h"
#include "liquid_tank_staging_3d_cases.h"
#include "liquid_viscous_tank_cases.h"
#include "liquid_wall_sample_3d_cases.h"
#include "snow_liquid_scheduler_cases.h"
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

TEST(LiquidSubstep3D, SnowTrialPublishesOnlyWithLiquidAcceptance) {
  const auto state = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(state, Scene(), Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  test::ExpectSnowJointAcceptance(*scheduler, state);
}

TEST(LiquidSubstep3D, SnowAndLiquidUseSameMovingPlaten) {
  test::ExpectSnowMovingPlaten(Initial(), Options(), CpuLiquidSubstepBackend3D());
}

TEST(LiquidSubstep3D, SnowFrameRetriesAndPublishesBothMaterials) {
  test::ExpectSnowMovingPlaten(Initial(), Options(), CpuLiquidSubstepBackend3D(), test::SnowPlatenRun::kRetry);
}

TEST(LiquidSubstep3D, SnowFrameAcousticCapAndCapacityReplayRestoreBothMaterials) {
  test::ExpectSnowMovingPlaten(Initial(), Options(), CpuLiquidSubstepBackend3D(), test::SnowPlatenRun::kCapacityReplay);
}

TEST(LiquidSubstep3D, PublishesLiveAdaptiveFamilyOnlyAfterAcceptance) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedFamilyFixture3D(state, scene).ok());
  auto options = Options();
  options.capacities = {12, 0, 12, 64, 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  scheduler->SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject graded family")
                                                           : absl::OkStatus();
          },
  });
  const auto parent = state.adaptive_cells[0].key;
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto rejected = scheduler->AttemptWithAdaptiveFamily(0.01, parent, true);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ExpectIdenticalSerializedState(state, scheduler->committed());
  scheduler->SetStageFaultHookForTesting({});
  const auto refined = scheduler->AttemptWithAdaptiveFamily(0.005, parent, true);
  ASSERT_TRUE(refined.ok()) << refined.status();
  ASSERT_EQ(refined->disposition, LiquidAttemptDisposition3D::kAccepted);
  EXPECT_EQ(scheduler->committed().adaptive_cells.size(), 12);
  const auto remainder = scheduler->Attempt(0.005);
  ASSERT_TRUE(remainder.ok()) << remainder.status();
  ASSERT_EQ(remainder->disposition, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  const auto encoded = EncodeLiquidState3D(scheduler->committed());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidState3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  auto restarted = LiquidSubstepScheduler3D::Create(*decoded, scene, options);
  ASSERT_TRUE(restarted.ok()) << restarted.status();
  for (auto* run : {&*scheduler, &*restarted}) {
    ASSERT_TRUE(run->BeginFrame(0.02).ok());
    const auto coarsened = run->AttemptWithAdaptiveFamily(0.01, parent, false);
    ASSERT_TRUE(coarsened.ok()) << coarsened.status();
    ASSERT_EQ(coarsened->disposition, LiquidAttemptDisposition3D::kAccepted);
    ASSERT_TRUE(run->FinishFrame().ok());
    EXPECT_EQ(run->committed().adaptive_cells.size(), 5);
  }
  ExpectIdenticalSerializedState(scheduler->committed(), restarted->committed());
}

TEST(LiquidSubstep3D, PublishesProjectedAdaptiveOwnerExchangeOnlyAfterAcceptance) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedOwnerExchangeFixture3D(state, scene).ok());
  auto options = Options();
  options.capacities = {3, 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  options.maximum_step_duration = 0.125;
  auto ordinary = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(ordinary.ok()) << ordinary.status();
  ASSERT_TRUE(ordinary->BeginFrame(0.125).ok());
  const auto refused = ordinary->Attempt(0.125);
  ASSERT_TRUE(refused.ok()) << refused.status();
  EXPECT_EQ(refused->disposition, LiquidAttemptDisposition3D::kFatal);
  ExpectIdenticalSerializedState(state, ordinary->committed());

  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.125).ok());
  scheduler->SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject projected owner exchange")
                                                           : absl::OkStatus();
          },
  });
  const auto rejected = scheduler->AttemptWithAdaptiveOwnerExchange(0.125);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt) << scheduler->diagnostics().failure_code;
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kAccept);
  ExpectIdenticalSerializedState(state, scheduler->committed());
  scheduler->SetStageFaultHookForTesting({});
  const auto retry = scheduler->AttemptWithAdaptiveOwnerExchange(*rejected->retry_duration);
  ASSERT_TRUE(retry.ok()) << retry.status();
  ASSERT_EQ(retry->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].mass, 0.1875);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[1].mass, 0.1875);

  auto complete = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(complete.ok()) << complete.status();
  ASSERT_TRUE(complete->BeginFrame(0.125).ok());
  const auto accepted = complete->AttemptWithAdaptiveOwnerExchange(0.125);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted) << complete->diagnostics().failure_code;
  ASSERT_TRUE(complete->FinishFrame().ok());
  const auto& result = complete->committed();
  EXPECT_DOUBLE_EQ(result.particles[0].mass, 0.25);
  EXPECT_DOUBLE_EQ(result.particles[1].mass, 0.125);
  EXPECT_DOUBLE_EQ(result.particles[1].world_velocity.x, 1);
  EXPECT_DOUBLE_EQ(result.particles[1].world_velocity.z, -1);
  EXPECT_DOUBLE_EQ(result.adaptive_cells[0].inventory.current_volume_m3, 0.5);
  EXPECT_DOUBLE_EQ(result.adaptive_cells[0].inventory.momentum_kg_m_per_s.x, -0.125);
  EXPECT_DOUBLE_EQ(result.adaptive_cells[0].inventory.momentum_kg_m_per_s.z, 0.125);
  EXPECT_EQ(result.next_particle_id, state.next_particle_id);
  for (std::size_t particle = 0; particle < result.particles.size(); ++particle) {
    EXPECT_EQ(result.particles[particle].id, state.particles[particle].id);
  }
}

TEST(LiquidSubstep3D, PublishesRequiredGradedRefinementFamiliesAtomically) {
  constexpr std::array requested{AdaptiveMacCellKey3D{.lower_corner = {4, 0, 0}, .width_exponent = 1}};
  for (const bool exact_child_coverage : {false, true}) {
    SCOPED_TRACE(exact_child_coverage);
    LiquidState3D state;
    LiquidSceneInputs3D scene;
    const auto authored = test::PopulateGradedRefinementClosureFixture3D(state, scene, exact_child_coverage);
    ASSERT_TRUE(authored.ok()) << authored;
    auto options = Options();
    options.capacities = {19, 0, 19, 160, 0};
    auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    if (!exact_child_coverage) {
      const auto refused = scheduler->AttemptWithAdaptiveRefinementClosure(0.01, requested);
      EXPECT_EQ(refused.status().code(), absl::StatusCode::kFailedPrecondition);
      EXPECT_EQ(refused.status().message(), "graded family inventory requires certified particle resampling");
      EXPECT_EQ(scheduler->diagnostics().attempts, 0);
      ExpectIdenticalSerializedState(state, scheduler->committed());
      continue;
    }
    scheduler->SetStageFaultHookForTesting({
        .context = nullptr,
        .before_stage =
            [](void*, LiquidPhysicalStage3D stage, double) {
              return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject complete grading closure")
                                                             : absl::OkStatus();
            },
    });
    const auto rejected = scheduler->AttemptWithAdaptiveRefinementClosure(0.01, requested);
    ASSERT_TRUE(rejected.ok()) << rejected.status();
    ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
    ASSERT_EQ(rejected->retry_duration, 0.005);
    ExpectIdenticalSerializedState(state, scheduler->committed());
    scheduler->SetStageFaultHookForTesting({});
    const auto accepted = scheduler->AttemptWithAdaptiveRefinementClosure(*rejected->retry_duration, requested);
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
    EXPECT_EQ(scheduler->committed().adaptive_cells.size(), 19);
    EXPECT_EQ(
        std::ranges::count_if(
            scheduler->committed().adaptive_cells.values(),
            [](const auto& leaf) {
              return leaf.key.width_exponent == 0;
            }
        ),
        8
    );
    const auto particles = scheduler->committed().particles.values();
    ASSERT_EQ(particles.size(), state.particles.size());
    for (std::size_t ordinal = 0; ordinal < particles.size(); ++ordinal) {
      EXPECT_EQ(particles[ordinal].id, state.particles[ordinal].id);
      EXPECT_DOUBLE_EQ(particles[ordinal].mass, state.particles[ordinal].mass);
      EXPECT_DOUBLE_EQ(particles[ordinal].reference_volume, state.particles[ordinal].reference_volume);
      EXPECT_DOUBLE_EQ(particles[ordinal].volume_ratio, state.particles[ordinal].volume_ratio);
    }
  }
}

TEST(LiquidSubstep3D, GradedWidthHysteresisCreditsOnlyAcceptedCertifiedTime) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedFamilyFixture3D(state, scene).ok());
  auto options = Options();
  options.capacities = {12, 0, 12, 64, 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto parent = state.adaptive_cells[0].key;
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const std::array halo{AdaptiveCoarsenHaloSample3D{.demand = {.maximum_width = 2}}};
  LiquidAdaptiveWidthRequest3D request{
      .accepted_epoch = state.clock.epoch,
      .leaf = parent,
      .demand = {.maximum_width = 0.5},
      .parent_and_halo = {},
      .complete_halo = false,
      .hysteresis_duration_s = 0.01,
  };
  const auto refined = scheduler->AttemptWithAdaptiveWidth(0.01, request);
  ASSERT_TRUE(refined.ok()) << refined.status();
  ASSERT_EQ(refined->disposition, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  ASSERT_EQ(scheduler->committed().adaptive_cells.size(), 12);

  request.leaf.width_exponent = 0;
  request.demand.maximum_width = 2;
  request.parent_and_halo = halo;
  request.complete_halo = true;
  ASSERT_TRUE(scheduler->BeginFrame(0.02).ok());
  EXPECT_EQ(scheduler->AttemptWithAdaptiveWidth(0.005, request).status().code(), absl::StatusCode::kFailedPrecondition);
  request.accepted_epoch = scheduler->committed().clock.epoch;
  scheduler->SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject graded width trial")
                                                           : absl::OkStatus();
          },
  });
  const auto rejected = scheduler->AttemptWithAdaptiveWidth(0.01, request);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ASSERT_EQ(rejected->retry_duration, 0.005);
  for (const auto& leaf : scheduler->committed().adaptive_cells.values()) {
    EXPECT_DOUBLE_EQ(leaf.coarsening_eligible_duration_s, 0);
  }
  scheduler->SetStageFaultHookForTesting({});
  for (double expected_age : {0.005, 0.01}) {
    request.accepted_epoch = scheduler->committed().clock.epoch;
    const auto accepted = scheduler->AttemptWithAdaptiveWidth(0.005, request);
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    EXPECT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
    ASSERT_EQ(scheduler->committed().adaptive_cells.size(), 12);
    for (const auto& leaf : scheduler->committed().adaptive_cells.values()) {
      EXPECT_DOUBLE_EQ(leaf.coarsening_eligible_duration_s, leaf.key.lower_corner.x < 2 ? expected_age : 0.0);
    }
  }
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  const auto encoded = EncodeLiquidState3D(scheduler->committed());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidState3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  auto restarted = LiquidSubstepScheduler3D::Create(*decoded, scene, options);
  ASSERT_TRUE(restarted.ok()) << restarted.status();
  auto vetoed = LiquidSubstepScheduler3D::Create(*decoded, scene, options);
  ASSERT_TRUE(vetoed.ok()) << vetoed.status();
  ASSERT_TRUE(vetoed->BeginFrame(0.03).ok());
  request.accepted_epoch = vetoed->committed().clock.epoch;
  request.complete_halo = false;
  const auto kept = vetoed->AttemptWithAdaptiveWidth(0.03 - 0.02, request);
  ASSERT_TRUE(kept.ok()) << kept.status();
  ASSERT_EQ(kept->disposition, LiquidAttemptDisposition3D::kAccepted);
  EXPECT_EQ(vetoed->committed().adaptive_cells.size(), 12);
  for (const auto& leaf : vetoed->committed().adaptive_cells.values()) {
    EXPECT_DOUBLE_EQ(leaf.coarsening_eligible_duration_s, 0);
  }
  request.complete_halo = true;
  for (auto* run : {&*scheduler, &*restarted}) {
    ASSERT_TRUE(run->BeginFrame(0.03).ok());
    request.accepted_epoch = run->committed().clock.epoch;
    const auto coarsened = run->AttemptWithAdaptiveWidth(0.03 - 0.02, request);
    ASSERT_TRUE(coarsened.ok()) << coarsened.status();
    ASSERT_EQ(coarsened->disposition, LiquidAttemptDisposition3D::kAccepted);
    ASSERT_TRUE(run->FinishFrame().ok());
    EXPECT_EQ(run->committed().adaptive_cells.size(), 5);
    for (const auto& leaf : run->committed().adaptive_cells.values()) {
      EXPECT_DOUBLE_EQ(leaf.coarsening_eligible_duration_s, 0);
    }
  }
  ExpectIdenticalSerializedState(scheduler->committed(), restarted->committed());
}

TEST(LiquidSubstep3D, GradedExplicitQuadraticTransferChangesAcceptedParticleAndGrid) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto particle_id = test::PopulateGradedQuadraticTankFixture3D(state, scene);
  ASSERT_TRUE(particle_id.ok()) << particle_id.status();
  auto options = Options();
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto selected = LiquidSubstepScheduler3D::Create(state, scene, options);
  auto radial = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(selected.ok()) << selected.status();
  ASSERT_TRUE(radial.ok()) << radial.status();
  ASSERT_TRUE(selected->BeginFrame(0.001).ok());
  ASSERT_TRUE(radial->BeginFrame(0.001).ok());
  const auto accepted = selected->AttemptWithAdaptiveQuadraticTransfer(0.001, state.clock.epoch, *particle_id);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted)
      << selected->diagnostics().failure_code << " stage " << static_cast<int>(selected->diagnostics().failed_stage);
  const auto radial_accepted = radial->Attempt(0.001);
  ASSERT_TRUE(radial_accepted.ok()) << radial_accepted.status();
  ASSERT_EQ(radial_accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
  const auto& a = selected->committed();
  const auto& b = radial->committed();
  EXPECT_NE(a.particles[*particle_id - 1].world_velocity.x, b.particles[*particle_id - 1].world_velocity.x);
  EXPECT_FALSE(std::ranges::equal(a.adaptive_faces.values(), b.adaptive_faces.values()));
  std::vector<AdaptiveMacCellKey3D> keys;
  for (const auto& leaf : state.adaptive_cells.values()) {
    keys.push_back(leaf.key);
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto transfer = GradedQuadraticMacTransfer3D::Create(*topology, state.geometry.world_origin);
  ASSERT_TRUE(transfer.ok()) << transfer.status();
  // The interior support is fully open, so only its unit areas are read.
  const std::vector<double> apertures(state.adaptive_faces.size(), 1);
  const auto& start = state.particles[*particle_id - 1];
  const GradedQuadraticMacParticle3D quadratic_particle{
      .transfer =
          {
              .sorted_world_position = start.world_position,
              .trial_world_position = start.world_position,
              .particle_mass = start.mass,
              .particle_velocity = start.world_velocity,
              .particle_velocity_slopes = {},
          },
      .side_tag = start.side_tag,
  };
  const auto carriers = state.adaptive_faces.values();
  const auto quadratic_deposit = transfer->Deposit(std::span(&quadratic_particle, 1), apertures, carriers);
  ASSERT_TRUE(quadratic_deposit.ok()) << quadratic_deposit.status();
  const auto radial_transfer = GradedMacTransfer3D::Create(*topology, state.geometry.world_origin);
  ASSERT_TRUE(radial_transfer.ok()) << radial_transfer.status();
  // Reconstruct the old snapshot independently of the scheduler's dispatch,
  // preserving the single global particle reduction order.
  std::vector<MacFaceDeposit3D> totals(carriers.size());
  for (const auto& particle : state.particles.values()) {
    const auto contribution =
        particle.id == *particle_id ? quadratic_deposit : radial_transfer->Deposit(std::span(&particle, 1), carriers);
    ASSERT_TRUE(contribution.ok()) << contribution.status();
    for (std::size_t slot = 0; slot < totals.size(); ++slot) {
      totals[slot].deposited_mass += (*contribution)[slot].deposited_mass;
      totals[slot].deposited_pure_momentum += (*contribution)[slot].deposited_pure_momentum;
    }
  }
  std::vector<LiquidAdaptiveFaceState3D> old_carriers(carriers.begin(), carriers.end());
  for (std::size_t slot = 0; slot < totals.size(); ++slot) {
    const auto& total = totals[slot];
    if (total.deposited_mass > 0) {
      old_carriers[slot].carrier_velocity = total.deposited_pure_momentum / total.deposited_mass;
    }
  }
  const auto old = transfer->Gather(quadratic_particle, apertures, old_carriers, old_carriers);
  const auto projected = transfer->Gather(quadratic_particle, apertures, old_carriers, a.adaptive_faces.values());
  ASSERT_TRUE(old.ok()) << old.status();
  ASSERT_TRUE(projected.ok()) << projected.status();
  const auto velocity = a.particles[*particle_id - 1].world_velocity;
  const double alpha = 0.99 * options.particle_energy_blend;
  EXPECT_DOUBLE_EQ(
      velocity.x,
      projected->new_pic_velocity.x + alpha * (start.world_velocity.x - old->new_pic_velocity.x)
  );
  EXPECT_DOUBLE_EQ(
      velocity.y,
      projected->new_pic_velocity.y + alpha * (start.world_velocity.y - old->new_pic_velocity.y)
  );
  EXPECT_DOUBLE_EQ(
      velocity.z,
      projected->new_pic_velocity.z + alpha * (start.world_velocity.z - old->new_pic_velocity.z)
  );
  EXPECT_EQ(a.particles[*particle_id - 1].volume_ratio, start.volume_ratio);
  EXPECT_EQ(a.next_particle_id, state.next_particle_id);
  EXPECT_EQ(a.clock.epoch, state.clock.epoch + 1);
  const auto stale = selected->AttemptWithAdaptiveQuadraticTransfer(0.001, state.clock.epoch, *particle_id);
  EXPECT_TRUE(absl::IsFailedPrecondition(stale.status()));
  EXPECT_EQ(selected->diagnostics().attempts, 1);
}

TEST(LiquidSubstep3D, GradedExplicitQuadraticFallsBackWhenDepositedOldCarriersAreLowMass) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto particle_id = test::PopulateGradedQuadraticTankFixture3D(state, scene);
  ASSERT_TRUE(particle_id.ok()) << particle_id.status();
  auto options = Options();
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  options.minimum_face_mass = 4;
  auto selected = LiquidSubstepScheduler3D::Create(state, scene, options);
  auto radial = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(selected.ok()) << selected.status();
  ASSERT_TRUE(radial.ok()) << radial.status();
  ASSERT_TRUE(selected->BeginFrame(0.001).ok());
  ASSERT_TRUE(radial->BeginFrame(0.001).ok());
  const auto accepted = selected->AttemptWithAdaptiveQuadraticTransfer(0.001, state.clock.epoch, *particle_id);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted)
      << selected->diagnostics().failure_code << " stage " << static_cast<int>(selected->diagnostics().failed_stage);
  const auto radial_accepted = radial->Attempt(0.001);
  ASSERT_TRUE(radial_accepted.ok()) << radial_accepted.status();
  ASSERT_EQ(radial_accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
  ExpectIdenticalSerializedState(selected->committed(), radial->committed());
}

TEST(LiquidSubstep3D, GradedExplicitQuadraticRejectsChangedProjectedCarrier) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto particle_id = test::PopulateGradedQuadraticTankFixture3D(state, scene);
  ASSERT_TRUE(particle_id.ok()) << particle_id.status();
  auto options = Options();
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto owner = liquid_step_internal::AdaptiveLiquidStep3D::Create(state, scene, options);
  ASSERT_TRUE(owner.ok()) << owner.status();
  const auto prepared = owner->PrepareQuadraticTransfer(state, options, *particle_id);
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  ASSERT_TRUE(prepared->has_value());
  auto trial = CloneLiquidState3D(state, options.capacities);
  ASSERT_TRUE(trial.ok()) << trial.status();
  ASSERT_TRUE(owner->Deposit(*trial, options, &**prepared).ok());
  const auto slot = (**prepared).selections().front().stencil.rows[0][0].face_slot;
  trial->adaptive_faces[slot].validity = LiquidFaceValidity3D::kPredicted;
  EXPECT_TRUE(absl::IsFailedPrecondition(owner->Gather(*trial, &**prepared)));
}

TEST(LiquidSubstep3D, GradedCertifiedQuadraticSplitRefusesIneligibleOldCarriersBeforeTrial) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto fixture = test::PopulateQuadraticFamilyReplayFixture3D(state, scene, false);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  LiquidStepOptions3D options;
  options.capacities = test::QuadraticFamilyReplayFixture3D::kCapacities;
  options.minimum_face_mass = 4;
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const LiquidAdaptiveQuadraticSplitRequest3D request{
      .accepted_epoch = state.clock.epoch,
      .exact_refinement_closure = fixture->closure,
      .split_parent = test::QuadraticFamilyReplayFixture3D::kParent,
      .source_particle_id = fixture->source_ids.front(),
  };
  const auto refused = scheduler->AttemptWithAdaptiveQuadraticSplit(0.01, request);
  EXPECT_TRUE(absl::IsFailedPrecondition(refused.status())) << refused.status();
  EXPECT_EQ(scheduler->diagnostics().attempts, 0);
  ExpectIdenticalSerializedState(state, scheduler->committed());
}

TEST(LiquidSubstep3D, CertifiesDetachedQuadraticSplitAndMergeWithFullHalo) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto fixture = test::PopulateQuadraticFamilyReplayFixture3D(state, scene, false);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  constexpr auto kCenter = test::QuadraticFamilyReplayFixture3D::kParent;
  constexpr auto kCapacities = test::QuadraticFamilyReplayFixture3D::kCapacities;
  const auto& requested = fixture->closure;
  const auto parent_id = fixture->source_ids.front();
  const auto before = EncodeLiquidState3D(state);
  ASSERT_TRUE(before.ok()) << before.status();

  const auto unsupported = liquid_step_internal::PrepareAdaptiveLiquidQuadraticSplitCandidate3D(
      state,
      scene,
      kCapacities,
      std::span(&kCenter, 1),
      kCenter,
      parent_id
  );
  EXPECT_EQ(unsupported.status().code(), absl::StatusCode::kFailedPrecondition);
  auto staged = liquid_step_internal::PrepareAdaptiveLiquidQuadraticSplitCandidate3D(
      state,
      scene,
      kCapacities,
      requested,
      kCenter,
      parent_id
  );
  ASSERT_TRUE(staged.ok()) << staged.status();
  EXPECT_EQ(staged->adaptive_cells.size(), 314);
  EXPECT_EQ(staged->particles.size(), state.particles.size() + 7);
  EXPECT_EQ(staged->next_particle_id, state.next_particle_id + 8);
  EXPECT_EQ(staged->clock, state.clock);
  for (std::size_t ordinal = 0; ordinal < 8; ++ordinal) {
    const auto& child = staged->particles[parent_id - 1 + ordinal];
    EXPECT_EQ(child.id, state.next_particle_id + ordinal);
    EXPECT_EQ(child.mass, 0.25);
    EXPECT_EQ(child.reference_volume, 0.125);
  }
  const auto after = EncodeLiquidState3D(state);
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_TRUE(std::ranges::equal(before->values(), after->values()));

  // A strict F4 covariance witness uses dyadic child centers. The symmetric
  // F3 offsets alone do not certify a positive covariance gap on reversal.
  std::array<std::uint64_t, 8> child_ids{};
  for (std::size_t ordinal = 0; ordinal < child_ids.size(); ++ordinal) {
    auto& child = staged->particles[parent_id - 1 + ordinal];
    child_ids[ordinal] = child.id;
    child.world_position = {
        (ordinal & 1u) == 0 ? 0.25 : 0.75,
        (ordinal & 2u) == 0 ? 0.25 : 0.75,
        (ordinal & 4u) == 0 ? 0.25 : 0.75,
    };
  }
  const auto fine_before = EncodeLiquidState3D(*staged);
  ASSERT_TRUE(fine_before.ok()) << fine_before.status();
  const auto missing_merge_halo = liquid_step_internal::PrepareAdaptiveLiquidQuadraticMergeCandidate3D(
      *staged,
      scene,
      kCapacities,
      std::span(&kCenter, 1),
      kCenter,
      child_ids
  );
  EXPECT_EQ(missing_merge_halo.status().code(), absl::StatusCode::kFailedPrecondition);
  auto repeated_ids = child_ids;
  repeated_ids.back() = repeated_ids.front();
  const auto duplicate = liquid_step_internal::PrepareAdaptiveLiquidQuadraticMergeCandidate3D(
      *staged,
      scene,
      kCapacities,
      requested,
      kCenter,
      repeated_ids
  );
  EXPECT_EQ(duplicate.status().code(), absl::StatusCode::kInvalidArgument);
  std::ranges::reverse(child_ids);
  const auto merged = liquid_step_internal::PrepareAdaptiveLiquidQuadraticMergeCandidate3D(
      *staged,
      scene,
      kCapacities,
      requested,
      kCenter,
      child_ids
  );
  ASSERT_TRUE(merged.ok()) << merged.status();
  EXPECT_EQ(merged->adaptive_cells.size(), state.adaptive_cells.size());
  EXPECT_EQ(merged->particles.size(), state.particles.size());
  EXPECT_EQ(merged->next_particle_id, staged->next_particle_id + 1);
  auto expected_parent = state.particles[parent_id - 1];
  expected_parent.id = staged->next_particle_id;
  EXPECT_EQ(merged->particles[parent_id - 1], expected_parent);
  EXPECT_EQ(merged->adaptive_cells, state.adaptive_cells);
  EXPECT_EQ(merged->adaptive_faces, state.adaptive_faces);
  const auto fine_after = EncodeLiquidState3D(*staged);
  ASSERT_TRUE(fine_after.ok()) << fine_after.status();
  EXPECT_TRUE(std::ranges::equal(fine_before->values(), fine_after->values()));
}

TEST(LiquidSubstep3D, AppliesCertifiedBandPlacementFromAcceptedPlanarOccupancy) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto authored = test::PopulateDeepGradedBandFixture3D(state, scene);
  ASSERT_TRUE(authored.ok()) << authored;
  auto options = Options();
  options.capacities = {8, 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  LiquidAdaptiveBandRequest3D request{
      .accepted_epoch = state.clock.epoch,
      .leaf = state.adaptive_cells[0].key,
      .margins = {},
      .destination_carrier = std::nullopt,
      .reseed_particle_count = 1,
  };
  const auto unavailable = scheduler->AttemptWithAdaptiveBandPlacement(0.01, request);
  EXPECT_EQ(unavailable.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler->diagnostics().attempts, 0);
  request.destination_carrier = LiquidHandoffCarrier3D{.current_density = 2, .volume_ratio = 1};
  scheduler->SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject band placement")
                                                           : absl::OkStatus();
          },
  });
  const auto rejected = scheduler->AttemptWithAdaptiveBandPlacement(0.01, request);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ASSERT_TRUE(rejected->retry_duration.has_value());
  ExpectIdenticalSerializedState(state, scheduler->committed());
  scheduler->SetStageFaultHookForTesting({});
  const auto retired = scheduler->AttemptWithAdaptiveBandPlacement(*rejected->retry_duration, request);
  ASSERT_TRUE(retired.ok()) << retired.status();
  ASSERT_EQ(retired->disposition, LiquidAttemptDisposition3D::kAccepted);
  EXPECT_EQ(scheduler->committed().adaptive_cells[0].material_owner, AdaptiveLiquidMaterialOwner3D::kEulerian);
  EXPECT_EQ(scheduler->committed().particles.size(), 7);
  const auto& actual_inventory = scheduler->committed().adaptive_cells[0].inventory;
  const auto& expected_inventory = state.adaptive_cells[0].inventory;
  EXPECT_EQ(actual_inventory.mass_kg, expected_inventory.mass_kg);
  EXPECT_EQ(actual_inventory.reference_volume_m3, expected_inventory.reference_volume_m3);
  EXPECT_EQ(actual_inventory.current_volume_m3, expected_inventory.current_volume_m3);

  EXPECT_EQ(
      scheduler->AttemptWithAdaptiveBandPlacement(0.005, request).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  request.accepted_epoch = scheduler->committed().clock.epoch;
  request.margins.particle_support_radius_m = 4;
  const auto reseeded = scheduler->AttemptWithAdaptiveBandPlacement(0.005, request);
  ASSERT_TRUE(reseeded.ok()) << reseeded.status();
  ASSERT_EQ(reseeded->disposition, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  EXPECT_EQ(scheduler->committed().adaptive_cells[0].material_owner, AdaptiveLiquidMaterialOwner3D::kParticles);
  EXPECT_EQ(scheduler->committed().particles.size(), 8);
  EXPECT_EQ(scheduler->committed().particles[7].id, 9);
  const auto& reseeded_inventory = scheduler->committed().adaptive_cells[0].inventory;
  EXPECT_EQ(reseeded_inventory.mass_kg, expected_inventory.mass_kg);
  EXPECT_EQ(reseeded_inventory.reference_volume_m3, expected_inventory.reference_volume_m3);
  EXPECT_EQ(reseeded_inventory.current_volume_m3, expected_inventory.current_volume_m3);
}

TEST(LiquidSubstep3D, PublishesAdaptiveBandRetirementAndReseed) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedBandFixture3D(state, scene).ok());
  auto options = Options();
  options.capacities = {8, 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto key = state.adaptive_cells[1].key;
  const LiquidHandoffCarrier3D carrier{.current_density = 2, .volume_ratio = 1};
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  scheduler->SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject band handoff")
                                                           : absl::OkStatus();
          },
  });
  const auto rejected =
      scheduler->AttemptWithAdaptiveBandRetirement(0.01, key, LiquidHandoffOccupancy3D::kFullLiquid, carrier);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ExpectIdenticalSerializedState(scheduler->committed(), state);
  scheduler->SetStageFaultHookForTesting({});
  const auto retired =
      scheduler->AttemptWithAdaptiveBandRetirement(0.005, key, LiquidHandoffOccupancy3D::kFullLiquid, carrier);
  ASSERT_TRUE(retired.ok()) << retired.status();
  ASSERT_EQ(retired->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
  const auto core_step = scheduler->Attempt(0.005);
  ASSERT_TRUE(core_step.ok()) << core_step.status();
  ASSERT_EQ(core_step->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  ASSERT_EQ(scheduler->committed().particles.size(), 4);
  ASSERT_EQ(scheduler->committed().adaptive_cells[1].material_owner, AdaptiveLiquidMaterialOwner3D::kEulerian);
  EXPECT_EQ(scheduler->committed().adaptive_cells[1].inventory.mass_kg, state.adaptive_cells[1].inventory.mass_kg);
  const auto bytes = EncodeLiquidState3D(scheduler->committed());
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  const auto decoded = DecodeLiquidState3D(bytes->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  auto restarted = LiquidSubstepScheduler3D::Create(*decoded, scene, options);
  ASSERT_TRUE(restarted.ok()) << restarted.status();
  for (auto* run : {&*scheduler, &*restarted}) {
    ASSERT_TRUE(run->BeginFrame(0.02).ok());
    const auto reseeded =
        run->AttemptWithAdaptiveCoreReseed(0.01, key, LiquidHandoffOccupancy3D::kFullLiquid, carrier, 1);
    ASSERT_TRUE(reseeded.ok()) << reseeded.status();
    ASSERT_EQ(reseeded->disposition, LiquidAttemptDisposition3D::kAccepted) << run->diagnostics().failure_code;
    ASSERT_TRUE(run->FinishFrame().ok());
    ASSERT_EQ(run->committed().particles.size(), 5);
    EXPECT_EQ(run->committed().adaptive_cells[1].material_owner, AdaptiveLiquidMaterialOwner3D::kParticles);
    EXPECT_EQ(run->committed().next_particle_id, state.next_particle_id + 1);
  }
  ExpectIdenticalSerializedState(scheduler->committed(), restarted->committed());
}

TEST(LiquidSubstep3D, RejectsEulerianDrainThatWouldInvalidateRestart) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedBandFixture3D(state, scene).ok());
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(3);
  ASSERT_TRUE(particles.ok());
  ASSERT_TRUE(particles->Resize(3).ok());
  (*particles)[0] = state.particles[0];
  (*particles)[1] = state.particles[2];
  (*particles)[2] = state.particles[4];
  state.particles = std::move(*particles);
  for (const auto ordinal : {1U, 3U}) {
    auto& cell = state.adaptive_cells[ordinal];
    cell.material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian;
    cell.inventory.mass_kg /= 2;
    cell.inventory.reference_volume_m3 /= 2;
    cell.inventory.current_volume_m3 /= 2;
  }
  std::vector<AdaptiveMacCellKey3D> keys;
  for (const auto& cell : state.adaptive_cells.values()) {
    keys.push_back(cell.key);
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(ownership.ok()) << ownership.status();
  std::size_t donor_slot = state.adaptive_faces.size();
  for (std::size_t face = 0; face < topology->interior_faces().size(); ++face) {
    const auto& edge = topology->interior_faces()[face];
    if (edge.tail_cell_ordinal == 1 && edge.head_cell_ordinal == 3) {
      donor_slot = ownership->interior_face_slots[face];
    }
  }
  ASSERT_LT(donor_slot, state.adaptive_faces.size());
  std::array projection_shape{state.adaptive_cells.size(), state.adaptive_faces.size(), donor_slot};
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &projection_shape;
  backend.project_adaptive = [](void* context,
                                const pressure_internal::PreparedMacPressureSystem3D&,
                                MacPressureSolverOptions3D) -> absl::StatusOr<MacPressureProjection3D> {
    const auto [cells, faces, drained_slot] = *static_cast<const std::array<std::size_t, 3>*>(context);
    MacPressureProjection3D result;
    result.cell_pressures.resize(cells);
    result.corrected_face_velocities.resize(faces);
    // 2 m/s times 0.25 m² times 0.125 s drains precisely 0.0625 m³.
    result.corrected_face_velocities[drained_slot] = 2;
    return result;
  };
  backend.update_particles = [](void*, const LiquidParticleUpdateInputs3D&, std::span<LiquidParticleState3D>) {
    return absl::OkStatus();
  };
  auto options = Options();
  options.capacities = {3, 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options, backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.125).ok());
  const auto rejected = scheduler->Attempt(0.125);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ExpectIdenticalSerializedState(state, scheduler->committed());
}

void PopulateGradedPhysicalTrialFixture(
    LiquidState3D& state,
    LiquidSceneInputs3D& scene,
    double wall_speed_m_per_s = 0.03125
) {
  ASSERT_TRUE(
      test::PopulateGradedPhysicalHalfspaceFixture3D(
          state,
          scene,
          {.unit_normal = {1, 0, 0}, .start_plane_coordinate = 0.125, .wall_velocity = {wall_speed_m_per_s, 0, 0}}
      )
          .ok()
  );
}

TEST(LiquidSubstep3D, ReappliesGradedPhysicalGeometryAcrossRollbackAndFreshReplay) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  ASSERT_NO_FATAL_FAILURE(PopulateGradedPhysicalTrialFixture(state, scene));
  LiquidStepOptions3D options;
  options.capacities = {5, 0, 5, state.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double frame_duration_s = 0.125;
  EXPECT_EQ(scheduler->AdvanceFrame(frame_duration_s).code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(scheduler->BeginFrame(frame_duration_s).ok());
  EXPECT_EQ(scheduler->Attempt(frame_duration_s).status().code(), absl::StatusCode::kFailedPrecondition);
  bool fail_once = true;
  scheduler->SetStageFaultHookForTesting({
      .context = &fail_once,
      .before_stage =
          [](void* context, LiquidPhysicalStage3D stage, double) {
            if (stage == LiquidPhysicalStage3D::kParticleUpdate && *static_cast<bool*>(context)) {
              *static_cast<bool*>(context) = false;
              return absl::AbortedError("injected after physical pressure publication to trial");
            }
            return absl::OkStatus();
          },
  });
  const auto& wall = std::get<TranslatingHalfspace3D>(scene.obstacle);
  std::array<double, 5> source_divergences{};
  std::vector<std::optional<LiquidAdaptivePhysicalExteriorPressure3D>> exterior_pressures;
  auto attempt = [&](LiquidSubstepScheduler3D& run, double duration_s) {
    const auto clock = run.committed().clock;
    const double wall_x = wall.start_plane_coordinate + clock.time * wall.wall_velocity.x;
    source_divergences[0] = -wall.wall_velocity.x / (1.0 - wall_x);
    return run.AttemptWithAdaptivePhysicalGeometry(
        duration_s,
        {
            .accepted_clock = clock,
            .target_divergences_s_inv = source_divergences,
            .exterior_pressures = exterior_pressures,
        }
    );
  };

  const auto rejected = attempt(*scheduler, frame_duration_s);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ASSERT_EQ(rejected->retry_duration, frame_duration_s / 2);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kParticleUpdate);
  ExpectIdenticalSerializedState(state, scheduler->committed());
  const auto first = attempt(*scheduler, *rejected->retry_duration);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_EQ(first->disposition, LiquidAttemptDisposition3D::kAccepted)
      << static_cast<int>(scheduler->diagnostics().failure_code);
  ASSERT_TRUE(scheduler->diagnostics().accepted_measurement.has_value());
  EXPECT_NEAR(scheduler->diagnostics().accepted_measurement->step_start_pressure_cell_volume_m3, 1.375, 1e-14);
  EXPECT_NEAR(scheduler->diagnostics().accepted_measurement->pressure_rhs_l2_m3_per_s, 0, 1e-14);
  EXPECT_EQ(scheduler->Attempt(*rejected->retry_duration).status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(
      scheduler
          ->AttemptWithAdaptivePhysicalGeometry(
              *rejected->retry_duration,
              {.accepted_clock = state.clock, .target_divergences_s_inv = source_divergences}
          )
          .status()
          .code(),
      absl::StatusCode::kFailedPrecondition
  );
  const auto second = attempt(*scheduler, *rejected->retry_duration);
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_EQ(second->disposition, LiquidAttemptDisposition3D::kAccepted)
      << static_cast<int>(scheduler->diagnostics().failure_code);
  ASSERT_TRUE(scheduler->diagnostics().accepted_measurement.has_value());
  EXPECT_NEAR(
      scheduler->diagnostics().accepted_measurement->step_start_pressure_cell_volume_m3,
      1.375 - wall.wall_velocity.x * *rejected->retry_duration,
      1e-14
  );
  ASSERT_TRUE(scheduler->FinishFrame().ok());

  LiquidResumeMetadata3D resume{.capacities = options.capacities};
  resume.resolved_fingerprint[0] = 1;
  resume.execution_id = LiquidExecutionId3D(resume.resolved_fingerprint);
  const auto bytes = EncodeLiquidCheckpoint3D(scheduler->committed(), resume);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  const auto checkpoint = DecodeLiquidCheckpoint3D(bytes->values());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  auto restored = LiquidSubstepScheduler3D::Create(checkpoint->state, scene, options);
  ASSERT_TRUE(restored.ok()) << restored.status();

  std::vector<AdaptiveMacCellKey3D> keys;
  for (const auto& cell : checkpoint->state.adaptive_cells.values()) {
    keys.push_back(cell.key);
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, checkpoint->state.geometry.cell_width);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(ownership.ok()) << ownership.status();
  const auto outlet = std::ranges::find_if(ownership->exterior_faces, [](const AdaptiveMacExteriorFace3D& face) {
    return face.face_index.axis == FaceNormalAxis::kX && face.cell_is_tail;
  });
  ASSERT_NE(outlet, ownership->exterior_faces.end());
  exterior_pressures.resize(checkpoint->state.adaptive_faces.size());
  exterior_pressures[outlet->face_slot] =
      LiquidAdaptivePhysicalExteriorPressure3D{.pressure_pa = 0.5, .distance_m = 0.25};
  for (auto* run : {&*scheduler, &*restored}) {
    ASSERT_TRUE(run->BeginFrame(2 * frame_duration_s).ok());
    EXPECT_EQ(run->Attempt(frame_duration_s).status().code(), absl::StatusCode::kFailedPrecondition);
    const auto continued = attempt(*run, frame_duration_s);
    ASSERT_TRUE(continued.ok()) << continued.status();
    ASSERT_EQ(continued->disposition, LiquidAttemptDisposition3D::kAccepted)
        << static_cast<int>(run->diagnostics().failure_code);
    EXPECT_NEAR(run->committed().adaptive_cells[0].previous_pressure, 0.5, 2e-5);
    ASSERT_TRUE(run->FinishFrame().ok());
  }
  ExpectIdenticalSerializedState(scheduler->committed(), restored->committed());
}

TEST(LiquidSubstep3D, RefusesGradedPhysicalTopologyChangeAndIncompatibleSource) {
  for (const double wall_speed : {0.03125, 8.0}) {
    LiquidState3D state;
    LiquidSceneInputs3D scene;
    ASSERT_NO_FATAL_FAILURE(PopulateGradedPhysicalTrialFixture(state, scene, wall_speed));
    LiquidStepOptions3D options;
    options.capacities = {5, 0, 5, state.adaptive_faces.size(), 0};
    auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(0.125).ok());
    const std::array<double, 5> no_source{};
    const auto attempt = scheduler->AttemptWithAdaptivePhysicalGeometry(
        0.125,
        {.accepted_clock = state.clock, .target_divergences_s_inv = no_source}
    );
    if (wall_speed == 8.0) {
      EXPECT_EQ(attempt.status().code(), absl::StatusCode::kFailedPrecondition);
    } else {
      ASSERT_TRUE(attempt.ok()) << attempt.status();
      EXPECT_EQ(attempt->disposition, LiquidAttemptDisposition3D::kFatal);
      EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kFailedPrecondition);
      EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
    }
    ExpectIdenticalSerializedState(state, scheduler->committed());
  }
}

TEST(LiquidSubstep3D, AdvancesGradedTankThroughTheAcceptedScheduler) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto authored = test::PopulateGradedTankFixture3D(state, scene);
  ASSERT_TRUE(authored.ok()) << authored;
  auto options = Options();
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  bool reject = true;
  scheduler->SetStageFaultHookForTesting({
      .context = &reject,
      .before_stage =
          [](void* context, LiquidPhysicalStage3D stage, double) {
            auto& pending = *static_cast<bool*>(context);
            if (pending && stage == LiquidPhysicalStage3D::kAccept) {
              pending = false;
              return absl::Status(absl::StatusCode::kAborted, {});
            }
            return absl::OkStatus();
          },
  });
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1, 1, 1},
      .center = {0.75, 0.5, 0.5},
  };
  const auto unsupported = scheduler->AttemptWithTranslatingFloatingBox(0.01, body, {0.125, 0.125, 0.125});
  EXPECT_EQ(unsupported.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(scheduler->committed(), state);
  const auto rejected = scheduler->Attempt(0.01);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ExpectIdenticalSerializedState(state, scheduler->committed());
  for (int step = 0; step < 2; ++step) {
    const auto accepted = scheduler->Attempt(0.005);
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
  }
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  EXPECT_EQ(scheduler->committed().clock.frame_ordinal, 1);
  EXPECT_EQ(scheduler->committed().adaptive_faces.size(), state.adaptive_faces.size());
  EXPECT_EQ(scheduler->committed().particles.size(), state.particles.size());
  EXPECT_NE(scheduler->committed().particles[0].world_position.x, state.particles[0].world_position.x);
}

TEST(LiquidSubstep3D, GradedCompressionPreservesPressureAndVolumeAcrossRollback) {
  constexpr std::array<std::optional<LiquidEosLaw>, 3> kLaws{
      std::nullopt,
      LiquidEosLaw::kHencky,
      LiquidEosLaw::kTait,
  };
  for (std::size_t mode = 0; mode < kLaws.size(); ++mode) {
    SCOPED_TRACE(mode);
    LiquidState3D initial;
    LiquidSceneInputs3D scene;
    ASSERT_TRUE(test::PopulateGradedTankFixture3D(initial, scene).ok());
    constexpr double kBulkModulus = 8.0;
    scene.ambient_pressure = 1.0;
    if (kLaws[mode]) {
      scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0, .law = *kLaws[mode]};
    } else {
      scene.bulk_modulus = kBulkModulus;
    }
    for (auto& particle : initial.particles.values()) {
      particle.world_velocity = {};
    }
    for (auto& cell : initial.adaptive_cells.values()) {
      cell.inventory.momentum_kg_m_per_s = {};
    }
    auto options = Options();
    options.capacities = {initial.particles.size(), 0, initial.adaptive_cells.size(), initial.adaptive_faces.size(), 0};
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    bool reject = true;
    scheduler->SetStageFaultHookForTesting({
        .context = &reject,
        .before_stage =
            [](void* context, LiquidPhysicalStage3D stage, double) {
              if (stage == LiquidPhysicalStage3D::kAccept && std::exchange(*static_cast<bool*>(context), false)) {
                return absl::Status(absl::StatusCode::kAborted, {});
              }
              return absl::OkStatus();
            },
    });
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    const auto rejected = scheduler->Attempt(0.01);
    ASSERT_TRUE(rejected.ok()) << rejected.status();
    ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
    ExpectIdenticalSerializedState(initial, scheduler->committed());
    const auto accepted = scheduler->Attempt(0.005);
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
    const auto& state = scheduler->committed();
    for (std::size_t cell = 0; cell < state.adaptive_cells.size(); ++cell) {
      const auto& leaf = state.adaptive_cells[cell];
      EXPECT_GT(leaf.previous_pressure, 0.0);
      EXPECT_LT(leaf.previous_pressure, scene.ambient_pressure);
      EXPECT_LT(state.particles[cell].volume_ratio, 1.0);
      EXPECT_NEAR(state.particles[cell].volume_ratio, std::exp(-leaf.previous_pressure / kBulkModulus), 1e-6);
      EXPECT_NEAR(
          leaf.inventory.current_volume_m3,
          state.particles[cell].reference_volume * state.particles[cell].volume_ratio,
          1e-14
      );
    }
  }
}

TEST(LiquidSubstep3D, GradedViscousTensorReproducesAffineStrain) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedTankFixture3D(state, scene).ok());
  std::vector<AdaptiveMacCellKey3D> keys;
  for (const auto& cell : state.adaptive_cells.values()) {
    keys.push_back(cell.key);
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto transfer = GradedMacTransfer3D::Create(*topology, state.geometry.world_origin);
  ASSERT_TRUE(transfer.ok()) << transfer.status();
  std::vector<pressure_internal::AdaptiveMacPressureSubface3D> subfaces(transfer->faces().size());
  std::vector<double> velocities;
  for (std::size_t slot = 0; slot < transfer->faces().size(); ++slot) {
    const auto& face = transfer->faces()[slot];
    const auto position = face.world_anchor;
    const std::array components{
        1.0 + 2.0 * position.x + 3.0 * position.y + 5.0 * position.z,
        7.0 - position.x + 4.0 * position.y + 11.0 * position.z,
        13.0 + 17.0 * position.x + 19.0 * position.y + 6.0 * position.z,
    };
    velocities.push_back(components[static_cast<std::size_t>(face.axis)]);
    subfaces[slot].open_area_m2 = face.width_m * face.width_m;
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(ownership.ok()) << ownership.status();
  ASSERT_FALSE(ownership->exterior_faces.empty());
  std::size_t wall_slot = 0;
  for (const bool prescribed_wall : {false, true}) {
    SCOPED_TRACE(prescribed_wall);
    if (prescribed_wall) {
      auto& wall = subfaces[wall_slot];
      wall.prescribed_wall_flux_m3_per_s = wall.open_area_m2 * velocities[wall_slot];
      wall.open_area_m2 = 0.0;
      velocities[wall_slot] = 0.0;
    }
    const auto rows = liquid_step_internal::AssembleGradedViscousStrainRows3D(
        *topology,
        state.geometry.world_origin,
        2.0,
        transfer->faces(),
        subfaces
    );
    ASSERT_TRUE(rows.ok()) << rows.status();
    if (!prescribed_wall) {
      const auto wall = std::ranges::find_if(ownership->exterior_faces, [&](const auto& face) {
        return std::ranges::any_of(*rows, [&](const auto& row) {
          return row.first_axis != row.second_axis
              && std::ranges::any_of(std::span(row.terms).first(row.term_count), [&](const auto& term) {
                   return term.face_slot == face.face_slot;
                 });
        });
      });
      ASSERT_NE(wall, ownership->exterior_faces.end());
      wall_slot = wall->face_slot;
    }
    constexpr std::array expected_strains{2.0, 1.0, 11.0, 4.0, 15.0, 6.0};
    ASSERT_EQ(rows->size(), expected_strains.size() * topology->cells().size());
    for (std::size_t index = 0; index < rows->size(); ++index) {
      const auto& row = (*rows)[index];
      double strain = row.prescribed_strain;
      for (const auto& term : std::span(row.terms).first(row.term_count)) {
        strain += term.coefficient * velocities[term.face_slot];
      }
      EXPECT_NEAR(strain, expected_strains[index % expected_strains.size()], 1e-12);
      const double volume = topology->cells()[index / expected_strains.size()].cell_volume;
      EXPECT_DOUBLE_EQ(row.quadrature_weight, (row.first_axis == row.second_axis ? 4.0 : 8.0) * volume);
    }
  }
  subfaces[wall_slot].open_area_m2 = 0.5 * transfer->faces()[wall_slot].width_m * transfer->faces()[wall_slot].width_m;
  const auto moving_cut = liquid_step_internal::AssembleGradedViscousStrainRows3D(
      *topology,
      state.geometry.world_origin,
      2.0,
      transfer->faces(),
      subfaces
  );
  EXPECT_EQ(moving_cut.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(LiquidSubstep3D, GradedViscousTankDissipatesCarrierEnergyAcrossAcceptance) {
  std::array<double, 2> energy{};
  for (std::size_t mode = 0; mode < energy.size(); ++mode) {
    LiquidState3D initial;
    LiquidSceneInputs3D scene;
    ASSERT_TRUE(test::PopulateGradedTankFixture3D(initial, scene).ok());
    scene.dynamic_viscosity = mode == 0 ? 0.0 : 2.0;
    auto options = Options();
    options.capacities = {initial.particles.size(), 0, initial.adaptive_cells.size(), initial.adaptive_faces.size(), 0};
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    const auto accepted = scheduler->Attempt(0.01);
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
    ASSERT_TRUE(scheduler->FinishFrame().ok());
    const auto carrier_energy = test::GradedTankCarrierEnergy3D(scheduler->committed(), scene);
    ASSERT_TRUE(carrier_energy.ok()) << carrier_energy.status();
    energy[mode] = *carrier_energy;
    ASSERT_TRUE(scheduler->diagnostics().accepted_measurement);
    const auto& measurement = *scheduler->diagnostics().accepted_measurement;
    EXPECT_LE(measurement.pressure.max_cell_constraint_defect, 2e-5);
    if (mode != 0) {
      ASSERT_TRUE(measurement.viscous_max_momentum_residual_kg_m_per_s);
      EXPECT_LE(*measurement.viscous_max_momentum_residual_kg_m_per_s, 2e-5);
    }
  }
  EXPECT_GT(energy[1], 0.0);
  EXPECT_LT(energy[1], energy[0]);
}

TEST(LiquidSubstep3D, GradedMiscibleViscosityUsesAcceptedLeafComposition) {
  std::array<double, 4> energy{};
  for (std::size_t mode = 0; mode < energy.size(); ++mode) {
    LiquidState3D initial;
    LiquidSceneInputs3D scene;
    ASSERT_TRUE(test::PopulateGradedTankFixture3D(initial, scene).ok());
    test::SetGradedTankMiscible3D(initial, scene);
    std::array<LiquidConstituentMaterial3D, 2> materials{};
    std::ranges::copy(scene.constituent_materials, materials.begin());
    if (mode == 1) {
      materials[0].dynamic_viscosity_pa_s = 1.0;
      materials[1].dynamic_viscosity_pa_s = 10.0;
    }
    if (mode == 2) {
      materials[0].dynamic_viscosity_pa_s = 10.0;
      materials[1].dynamic_viscosity_pa_s = 1.0;
    }
    if (mode == 3) {
      scene.dynamic_viscosity = 2.0;
      materials[0].dynamic_viscosity_pa_s = 0.0;
      materials[1].dynamic_viscosity_pa_s = 0.0;
    }
    scene.constituent_materials = materials;
    auto options = Options();
    options.capacities = {initial.particles.size(), 0, initial.adaptive_cells.size(), initial.adaptive_faces.size(), 0};
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    bool reject = true;
    scheduler->SetStageFaultHookForTesting({
        .context = &reject,
        .before_stage =
            [](void* context, LiquidPhysicalStage3D stage, double) {
              if (stage == LiquidPhysicalStage3D::kAccept && std::exchange(*static_cast<bool*>(context), false)) {
                return absl::Status(absl::StatusCode::kAborted, {});
              }
              return absl::OkStatus();
            },
    });
    const auto rejected = scheduler->Attempt(0.01);
    ASSERT_TRUE(rejected.ok()) << rejected.status();
    ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
    ExpectIdenticalSerializedState(initial, scheduler->committed());
    const auto accepted = scheduler->Attempt(0.005);
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;

    const auto carrier_energy = test::GradedTankCarrierEnergy3D(scheduler->committed(), scene);
    ASSERT_TRUE(carrier_energy.ok()) << carrier_energy.status();
    energy[mode] = *carrier_energy;
    ASSERT_TRUE(scheduler->diagnostics().accepted_measurement);
    EXPECT_EQ(
        scheduler->diagnostics().accepted_measurement->viscous_max_momentum_residual_kg_m_per_s.has_value(),
        mode == 1 || mode == 2
    );
  }
  EXPECT_LT(energy[1], energy[0]);
  EXPECT_LT(energy[2], energy[0]);
  EXPECT_GT(std::abs(energy[1] - energy[2]), 1e-6);
  EXPECT_EQ(energy[3], energy[0]);
}

TEST(LiquidSubstep3D, GradedTaitDomainFailureKeepsTheAcceptedState) {
  LiquidState3D initial;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedTankFixture3D(initial, scene).ok());
  test::SetGradedTankEos3D(initial, scene);
  scene.eos->law = LiquidEosLaw::kTait;
  initial.adaptive_cells[0].previous_pressure = -2.0;
  auto options = Options();
  options.capacities = {initial.particles.size(), 0, initial.adaptive_cells.size(), initial.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kInvalidArgument);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, GradedEosBacktracesNonuniformAcceptedPressure) {
  std::array<double, 2> accepted_pressure{};
  for (std::size_t run = 0; run < accepted_pressure.size(); ++run) {
    LiquidState3D initial;
    LiquidSceneInputs3D scene;
    ASSERT_TRUE(test::PopulateGradedTankFixture3D(initial, scene).ok());
    test::SetGradedTankEos3D(initial, scene);
    initial.adaptive_cells[0].previous_pressure = 0.125;
    for (std::size_t cell = 0; cell < initial.adaptive_cells.size(); ++cell) {
      initial.particles[cell].volume_ratio = 0.875;
      initial.adaptive_cells[cell].inventory.current_volume_m3 *= 0.875;
      if (cell != 0) {
        initial.adaptive_cells[cell].previous_pressure = 0.5;
      }
    }
    if (run != 0) {
      std::vector<AdaptiveMacCellKey3D> keys;
      for (const auto& cell : initial.adaptive_cells.values()) {
        keys.push_back(cell.key);
      }
      const auto topology = AdaptiveMacTopology3D::Create(keys, initial.geometry.cell_width);
      ASSERT_TRUE(topology.ok()) << topology.status();
      const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
      ASSERT_TRUE(ownership.ok()) << ownership.status();
      for (std::size_t face = 0; face < topology->interior_faces().size(); ++face) {
        if (topology->interior_faces()[face].face_index.axis == FaceNormalAxis::kX) {
          initial.adaptive_faces[ownership->interior_face_slots[face]].carrier_velocity = 0.25;
        }
      }
    }
    auto options = Options();
    options.capacities = {initial.particles.size(), 0, initial.adaptive_cells.size(), initial.adaptive_faces.size(), 0};
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    const auto outcome = scheduler->Attempt(0.01);
    ASSERT_TRUE(outcome.ok()) << outcome.status();
    ASSERT_EQ(outcome->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
    accepted_pressure[run] = scheduler->committed().adaptive_cells[1].previous_pressure;
  }
  EXPECT_LT(accepted_pressure[1], accepted_pressure[0] - 1e-5);
}

TEST(LiquidSubstep3D, GradedMiscibleInventoriesSurviveCheckpointAndConserveVolume) {
  LiquidState3D initial;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedTankFixture3D(initial, scene).ok());
  test::SetGradedTankMiscible3D(initial, scene);
  const auto encoded = EncodeLiquidState3D(initial);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidState3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, initial);
  auto options = Options();
  options.capacities = {initial.particles.size(), 0, initial.adaptive_cells.size(), initial.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(*decoded, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto accepted = scheduler->Attempt(0.01);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  const auto& state = scheduler->committed();
  std::array<double, 2> volumes{};
  for (const auto& cell : state.adaptive_cells.values()) {
    ASSERT_EQ(cell.inventory.phase_id, 4);
    ASSERT_EQ(cell.inventory.constituent_count, 2);
    EXPECT_EQ(cell.inventory.constituents[0].substance_id, 8);
    EXPECT_EQ(cell.inventory.constituents[1].substance_id, 14);
    for (std::size_t component = 0; component < volumes.size(); ++component) {
      volumes[component] += cell.inventory.constituents[component].volume_m3;
    }
  }
  EXPECT_DOUBLE_EQ(volumes[0], 1.0);
  EXPECT_DOUBLE_EQ(volumes[1], 0.5);
  const auto checkpoint = EncodeLiquidState3D(state);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const auto restored = DecodeLiquidState3D(checkpoint->values());
  ASSERT_TRUE(restored.ok()) << restored.status();
  ExpectIdenticalSerializedState(state, *restored);
}

TEST(LiquidSubstep3D, GradedPlanarCapillarityUsesSmallestExposedLeafAndRetries) {
  LiquidState3D initial;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedTankFixture3D(initial, scene).ok());
  scene.surface_tension = 100.0;
  auto options = Options();
  options.capacities = {initial.particles.size(), 0, initial.adaptive_cells.size(), initial.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.02).ok());
  const auto rejected = scheduler->Attempt(0.02);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
  const auto accepted = scheduler->Attempt(0.01);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
  const auto& measurement = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement && measurement->step_start_capillary_bound_seconds);
  EXPECT_NEAR(*measurement->step_start_capillary_bound_seconds, std::sqrt(0.25 / (400.0 * std::numbers::pi)), 1e-14);
}

TEST(LiquidSubstep3D, RejectsDuplicateEulerianMaterialBeforeGradedSchedulerAdmission) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto authored = test::PopulateGradedTankFixture3D(state, scene);
  ASSERT_TRUE(authored.ok()) << authored;
  auto options = Options();
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  for (auto& face : state.adaptive_faces.values()) {
    face.validity = LiquidFaceValidity3D::kValid;
  }
  state.adaptive_cells[1].material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian;
  const auto before = CloneLiquidState3D(state, options.capacities);
  ASSERT_TRUE(before.ok()) << before.status();
  const auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  EXPECT_EQ(scheduler.status().code(), absl::StatusCode::kAborted);
  EXPECT_EQ(state, *before);
}

TEST(LiquidSubstep3D, RetainsGradedOwnershipAtSolidUpperCorner) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto authored = test::PopulateGradedTankFixture3D(state, scene);
  ASSERT_TRUE(authored.ok()) << authored;
  state.particles[state.particles.size() - 1].world_position = {1.5, 0.75, 1};
  auto options = Options();
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto attempt = scheduler->Attempt(0.01);
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  EXPECT_EQ(attempt->disposition, LiquidAttemptDisposition3D::kAccepted);
}

TEST(LiquidSubstep3D, PreservesHydrostaticRestAcrossVerticalGradedInterface) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto authored = test::PopulateGradedTankFixture3D(state, scene);
  ASSERT_TRUE(authored.ok()) << authored;
  scene.world_acceleration = {0, -9.81, 0};
  for (auto& particle : state.particles.values()) {
    particle.world_velocity = {};
  }
  for (auto& cell : state.adaptive_cells.values()) {
    cell.inventory.momentum_kg_m_per_s = {};
  }
  auto options = Options();
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto attempt = scheduler->Attempt(0.01);
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  ASSERT_EQ(attempt->disposition, LiquidAttemptDisposition3D::kAccepted);
  // The pressure solve stores the hydrostatic solution at production precision.
  for (const auto& face : scheduler->committed().adaptive_faces.values()) {
    EXPECT_NEAR(face.carrier_velocity, 0, 5e-8);
  }
}

TEST(LiquidSubstep3D, UniformSchedulerRefusesAdaptiveCheckpointState) {
  auto state = Initial(0);
  auto leaves = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(1);
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(6);
  ASSERT_TRUE(leaves.ok()) << leaves.status();
  ASSERT_TRUE(faces.ok()) << faces.status();
  state.adaptive_cells = std::move(*leaves);
  state.adaptive_faces = std::move(*faces);
  ASSERT_TRUE(state.adaptive_cells.Resize(1).ok());
  ASSERT_TRUE(state.adaptive_faces.Resize(6).ok());
  ASSERT_TRUE(ValidateLiquidState3D(state).ok());
  const auto result = LiquidSubstepScheduler3D::Create(state, Scene(), Options());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}
TEST(LiquidSubstep3D, EosHydrostaticCompressionMatchesColumn) {
  pressure_test::ExpectEosHydrostaticColumn(
      [](const LiquidState3D& initial, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options
      ) -> absl::StatusOr<pressure_test::HydrostaticRun> {
        auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
        if (!scheduler.ok()) {
          return scheduler.status();
        }
        return pressure_test::AdvanceHydrostaticColumn(*scheduler, options.capacities);
      }
  );
}

TEST(LiquidSubstep3D, SchedulerConsumesResolvedViscosityOnFullCells) {
  auto initial = Initial(2);
  initial.particles[0].world_position = {2.5, 2.5, 2.5};
  initial.particles[1].world_position = {3.5, 2.5, 2.5};
  initial.particles[0].world_velocity = {0.0, 0.0, 0.125};
  initial.particles[1].world_velocity = {0.0, 0.0, -0.125};
  LiquidResolvedSceneView3D request;
  request.scene.field = LiquidHorizontalPlaneSurface3D{3.0};
  request.scene.obstacle = AxisAlignedLiquidTank3D{2.0, 4.0, 2.0, 2.0, 4.0, 0.0};
  request.scene.world_acceleration = {};
  request.scene.dynamic_viscosity = 100.0;
  const auto encoded = EncodeLiquidResolvedScene3D(request);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto resolved = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  auto scene = resolved->View().scene;
  auto viscous = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(viscous.ok()) << viscous.status();
  const auto advanced = viscous->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  scene.dynamic_viscosity = 0.0;
  auto inviscid = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(inviscid.ok()) << inviscid.status();
  const auto inviscid_advanced = inviscid->AdvanceFrame(0.01);
  ASSERT_TRUE(inviscid_advanced.ok()) << inviscid_advanced;
  EXPECT_GT(
      std::abs(inviscid->committed().particles[0].world_velocity.z)
          - std::abs(viscous->committed().particles[0].world_velocity.z),
      1e-7
  );
}

TEST(LiquidSubstep3D, SamplesResolvedConstituentViscosityInCoupledProjection) {
  auto initial = Initial(2);
  for (std::size_t index = 0; index < initial.particles.size(); ++index) {
    auto& particle = initial.particles[index];
    particle.world_position = {2.5 + static_cast<double>(index), 2.5, 2.5};
    particle.world_velocity.z = index == 0 ? 0.125 : -0.125;
    particle.phase_id = 4;
    particle.constituent_count = 2;
    particle.constituents[0] = {8, 0.25};
    particle.constituents[1] = {14, 0.75};
  }
  LiquidResolvedSceneView3D request;
  request.scene.field = LiquidHorizontalPlaneSurface3D{3.0};
  request.scene.obstacle = AxisAlignedLiquidTank3D{2.0, 4.0, 2.0, 2.0, 4.0, 0.0};
  request.scene.world_acceleration = {};
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{
      {8, 4, 1000.0, 40.0},
      {14, 4, 1000.0, 120.0},
  }};
  request.scene.constituent_materials = kMaterials;
  const auto encoded = EncodeLiquidResolvedScene3D(request);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto resolved = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  auto scene = resolved->View().scene;
  ASSERT_EQ(scene.constituent_materials[0].dynamic_viscosity_pa_s, 40.0);
  ASSERT_EQ(scene.constituent_materials[1].dynamic_viscosity_pa_s, 120.0);
  std::vector<double> sampled;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &sampled;
  backend.project_viscous = [](void* context,
                               const LiquidBoundaryGeometryViewInputs3D& geometry,
                               const LiquidPressureProblemView3D& problem,
                               double viscosity,
                               std::span<const double> cell_viscosity,
                               MacPressureSolverOptions3D controls,
                               MacPressureWorkspace3D& workspace) {
    auto& observed = *static_cast<std::vector<double>*>(context);
    observed.assign(cell_viscosity.begin(), cell_viscosity.end());
    return CpuLiquidSubstepBackend3D()
        .project_viscous(nullptr, geometry, problem, viscosity, cell_viscosity, controls, workspace);
  };
  auto mixture = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(mixture.ok()) << mixture.status();
  const auto advanced = mixture->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  ASSERT_EQ(sampled.size(), 4);
  for (const double viscosity : sampled) {
    EXPECT_DOUBLE_EQ(viscosity, 100.0);
  }
  constexpr std::array<LiquidConstituentMaterial3D, 2> kUniformMaterials{{
      {8, 4, 1000.0},
      {14, 4, 1000.0},
  }};
  scene.constituent_materials = kUniformMaterials;
  scene.dynamic_viscosity = 100.0;
  auto uniform = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(uniform.ok()) << uniform.status();
  ASSERT_TRUE(uniform->AdvanceFrame(0.01).ok());
  const auto actual = EncodeLiquidState3D(mixture->committed());
  const auto expected = EncodeLiquidState3D(uniform->committed());
  ASSERT_TRUE(actual.ok() && expected.ok());
  EXPECT_TRUE(std::ranges::equal(actual->values(), expected->values()));
}
TEST(LiquidSubstep3D, CouplesImmiscibleViscosityAcrossPlanarTankCells) {
  auto initial = Initial(4);
  for (std::size_t index = 0; index < initial.particles.size(); ++index) {
    auto& particle = initial.particles[index];
    const bool oil = index % 2 != 0;
    particle.world_position = {oil ? 3.5 : 2.5, 2.5, index < 2 ? 2.5 : 3.5};
    particle.world_velocity.z = oil ? -0.125 : 0.125;
    particle.mass = oil ? 820.0 : 1000.0;
    particle.phase_id = oil ? 5 : 4;
    particle.constituent_count = 1;
    particle.constituents[0] = {oil ? 25u : 8u, 1.0};
  }
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{
      {8, 4, 1000.0, 40.0},
      {25, 5, 820.0, 120.0},
  }};
  constexpr std::array<LiquidPhasePair3D, 1> kPairs{{{4, 5, 0.031}}};
  LiquidSceneInputs3D scene;
  scene.field = LiquidHorizontalPlaneSurface3D{3.0};
  scene.obstacle = AxisAlignedLiquidTank3D{2.0, 4.0, 2.0, 2.0, 4.0, 0.0};
  scene.world_acceleration = {};
  scene.constituent_materials = kMaterials;
  scene.phase_pairs = kPairs;
  struct Audit {
    std::size_t crossings = 0;
    std::size_t viscous_calls = 0;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    static_cast<Audit*>(context)->crossings += fields.internal_interfaces.size();
    EXPECT_EQ(fields.internal_interfaces.size(), 2);
    for (const auto& crossing : fields.internal_interfaces) {
      EXPECT_DOUBLE_EQ(crossing.tail_fraction, 0.5);
      EXPECT_NEAR(crossing.curvature_tail_to_head, 0, 1e-13);
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  backend.project_viscous = [](void* context,
                               const LiquidBoundaryGeometryViewInputs3D& geometry,
                               const LiquidPressureProblemView3D& problem,
                               double viscosity,
                               std::span<const double> cell_viscosity,
                               MacPressureSolverOptions3D controls,
                               MacPressureWorkspace3D& workspace) {
    ++static_cast<Audit*>(context)->viscous_calls;
    EXPECT_EQ(cell_viscosity.size(), 4);
    for (std::size_t index = 0; index < cell_viscosity.size(); ++index) {
      EXPECT_DOUBLE_EQ(cell_viscosity[index], problem.cells[index].cell_index.x == 2 ? 40.0 : 120.0);
    }
    return CpuLiquidSubstepBackend3D()
        .project_viscous(nullptr, geometry, problem, viscosity, cell_viscosity, controls, workspace);
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_EQ(audit.crossings, 2);
  EXPECT_EQ(audit.viscous_calls, 1);
  const auto measurement = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement.has_value());
  EXPECT_LT(measurement->viscous_max_momentum_residual_kg_m_per_s.value_or(1), 1e-5);
  for (const auto& particle : scheduler->committed().particles.values()) {
    EXPECT_LT(std::abs(particle.world_velocity.z), 0.125);
  }
}
TEST(LiquidSubstep3D, PressureHistoryBacktracesAcrossSignedBlocks) {
  LiquidState3D committed;
  auto blocks = LiquidRecords3D<LiquidBlockState3D>::Create(2);
  auto cells = LiquidRecords3D<LiquidCellState3D>::Create(2);
  auto faces = LiquidRecords3D<LiquidFaceState3D>::Create(2);
  ASSERT_TRUE(blocks.ok() && cells.ok() && faces.ok());
  committed.blocks = std::move(*blocks);
  committed.cells = std::move(*cells);
  committed.faces = std::move(*faces);
  ASSERT_TRUE(committed.blocks.Resize(2).ok());
  ASSERT_TRUE(committed.cells.Resize(2).ok());
  ASSERT_TRUE(committed.faces.Resize(2).ok());
  committed.blocks[0].block_index = {-1, 0, 0};
  committed.blocks[1].block_index = {0, 0, 0};
  committed.cells[0] = {{-1, 0, 0}, LiquidCellOccupancy3D::kLiquid, 4.0};
  committed.cells[1] = {{0, 0, 0}, LiquidCellOccupancy3D::kLiquid, 8.0};
  committed.faces[0] = {{FaceNormalAxis::kX, {0, 0, 0}}, 192, 1.0, LiquidFaceValidity3D::kPredicted};
  committed.faces[1] = {{FaceNormalAxis::kX, {1, 0, 0}}, 193, 3.0, LiquidFaceValidity3D::kValid};
  ASSERT_TRUE(ValidateLiquidState3D(committed).ok());
  const auto transported = liquid_step_internal::AdvectLiquidPressureHistory3D(committed, {0, 0, 0}, 0.125, 1.0);
  ASSERT_TRUE(transported.ok()) << transported.status();
  EXPECT_EQ(*transported, 7.0);
  committed.cells[0].occupancy = LiquidCellOccupancy3D::kSolid;
  const auto clipped = liquid_step_internal::AdvectLiquidPressureHistory3D(committed, {0, 0, 0}, 0.125, 1.0);
  ASSERT_TRUE(clipped.ok()) << clipped.status();
  EXPECT_EQ(*clipped, 8.0);
  const auto new_cell = liquid_step_internal::AdvectLiquidPressureHistory3D(committed, {2, 0, 0}, 0.125, 1.0);
  ASSERT_TRUE(new_cell.ok()) << new_cell.status();
  EXPECT_EQ(*new_cell, 1.0);
}

TEST(LiquidSubstep3D, MeasuresLaplacePressureOnCurvedParticlePhase) {
  using curved_phase_test::kExpectedCurvature;
  using curved_phase_test::kTension;
  const auto fixture = curved_phase_test::MakeFixture();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  auto scene = fixture->scene;
  struct Audit {
    std::size_t crossings = 0;
    double square_error = 0.0;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    auto& audit = *static_cast<Audit*>(context);
    for (const auto& crossing : fields.internal_interfaces) {
      const auto& face = geometry.faces[crossing.face_ordinal];
      const auto anchor = geometry.mac_grid.FaceWorldAnchor(face.face_index);
      if (!anchor.ok()) {
        return absl::StatusOr<LiquidPressureProblemView3D>(anchor.status());
      }
      const std::array coordinate{anchor->x, anchor->y, anchor->z};
      const auto axis = static_cast<std::size_t>(face.face_index.axis);
      const double outward_curvature = crossing.curvature_tail_to_head * std::copysign(1.0, coordinate[axis]);
      const double error = outward_curvature / kExpectedCurvature - 1.0;
      audit.square_error += error * error;
      ++audit.crossings;
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  for (const double viscosity : {0.0, 0.001}) {
    SCOPED_TRACE(viscosity);
    scene.dynamic_viscosity = viscosity;
    audit = {};
    auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, scene, fixture->options, backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    const auto advanced = scheduler->AdvanceFrame(curved_phase_test::kStepDuration);
    ASSERT_TRUE(advanced.ok()) << advanced << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
    ASSERT_GT(audit.crossings, 0);
    const double relative_curvature_rms = std::sqrt(audit.square_error / static_cast<double>(audit.crossings));
    const auto [pressure_sum, core_cells] = curved_phase_test::CorePressureSum(scheduler->committed());
    ASSERT_EQ(core_cells, 8u);
    const double core_pressure = pressure_sum / static_cast<double>(core_cells);
    std::cout << "curved_phase_crossings=" << audit.crossings << " relative_curvature_rms=" << relative_curvature_rms
              << " core_pressure_pa=" << core_pressure << '\n';
    EXPECT_LT(relative_curvature_rms, 0.25);
    EXPECT_NEAR(core_pressure, kTension * kExpectedCurvature, 0.25 * kTension * kExpectedCurvature);
  }
}
TEST(LiquidSubstep3D, DiscoversParticlePhaseCrossingAndRetriesAtItsCapillaryLimit) {
  constexpr std::array<LiquidConstituentMaterial3D, 3> kMaterials{{
      {8, 4, 1000.0},
      {14, 4, 1020.0},
      {25, 5, 820.0},
  }};
  constexpr std::array<LiquidPhasePair3D, 1> kPairs{{{4, 5, 3.0e6}}};
  auto initial = Initial(2);
  initial.particles[0].phase_id = 4;
  initial.particles[0].constituent_count = 2;
  initial.particles[0].constituents[0] = {8, 0.25};
  initial.particles[0].constituents[1] = {14, 0.75};
  initial.particles[0].mass = 1015.0;
  initial.particles[1].world_position.x += 1.0;
  initial.particles[1].phase_id = 5;
  initial.particles[1].constituent_count = 1;
  initial.particles[1].constituents[0] = {25, 1.0};
  initial.particles[1].mass = 820.0;

  std::size_t crossings_seen = 0;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &crossings_seen;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    EXPECT_EQ(fields.cells.size(), 2);
    if (fields.cells.size() == 2) {
      EXPECT_DOUBLE_EQ(fields.cells[0].density.value_or(0.0), 1015.0);
      EXPECT_DOUBLE_EQ(fields.cells[1].density.value_or(0.0), 820.0);
    }
    EXPECT_EQ(fields.internal_interfaces.size(), 1);
    for (const auto& crossing : fields.internal_interfaces) {
      ++*static_cast<std::size_t*>(context);
      EXPECT_DOUBLE_EQ(crossing.tail_fraction, 0.5);
      EXPECT_DOUBLE_EQ(crossing.surface_tension, 3.0e6);
      EXPECT_NEAR(crossing.curvature_tail_to_head, 0.0, 1.0e-13);
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{.radius = 0.9};
  scene.world_acceleration = {};
  scene.constituent_materials = kMaterials;
  scene.phase_pairs = kPairs;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(status.ok()) << status << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 1);
  EXPECT_EQ(crossings_seen, 3);

  initial.particles[1].world_position = initial.particles[0].world_position;
  auto unresolved = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(unresolved.ok()) << unresolved.status();
  EXPECT_EQ(unresolved->AdvanceFrame(0.01).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(unresolved->committed() == initial);
}

TEST(LiquidSubstep3D, UsesCurrentVolumeMixtureDensityInPressureLinks) {
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1000.0}, {14, 4, 1020.0}}};
  struct Observation {
    std::size_t mixture_links = 0;
    bool finite_eos = false;
  } observation;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observation;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    constexpr double kExpectedDensity = 1007.5;
    auto& observed = *static_cast<Observation*>(context);
    EXPECT_EQ(fields.cells.size(), 1);
    if (!fields.cells.empty()) {
      EXPECT_DOUBLE_EQ(fields.cells.front().density.value_or(0.0), kExpectedDensity);
      if (observed.finite_eos) {
        EXPECT_DOUBLE_EQ(fields.cells.front().bulk_modulus, 4.0 * kExpectedDensity);
      }
    }
    const auto links = CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
    if (links.ok()) {
      for (const auto& face : links->faces) {
        if (face.open_area > 0.0) {
          EXPECT_DOUBLE_EQ(face.inverse_density, 1.0 / kExpectedDensity);
          ++observed.mixture_links;
        }
      }
    }
    return links;
  };
  auto initial = Initial(2);
  initial.particles[0].reference_volume = 0.5;
  initial.particles[0].volume_ratio = 2.0;
  initial.particles[0].mass = 1015.0;
  initial.particles[0].phase_id = 4;
  initial.particles[0].constituent_count = 2;
  initial.particles[0].constituents[0] = {8, 0.25};
  initial.particles[0].constituents[1] = {14, 0.75};
  initial.particles[1].reference_volume = 2.0;
  initial.particles[1].volume_ratio = 0.5;
  initial.particles[1].mass = 1000.0;
  initial.particles[1].phase_id = 4;
  initial.particles[1].constituent_count = 1;
  initial.particles[1].constituents[0] = {8, 1.0};
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 0.9};
  scene.world_acceleration = {};
  scene.constituent_materials = kMaterials;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(observation.mixture_links, 6);

  observation = {.finite_eos = true};
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0, .law = LiquidEosLaw::kTait};
  auto compliant = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(compliant.ok()) << compliant.status();
  const auto compliant_status = compliant->AdvanceFrame(0.01);
  ASSERT_TRUE(compliant_status.ok()) << compliant_status;
  EXPECT_EQ(observation.mixture_links, 6);
}

TEST(LiquidSubstep3D, TaitFreezesComplianceAtAdvectedPressureAndRestarts) {
  struct Observation {
    std::size_t cells = 0;
    bool saw_nonzero_pressure = false;
  } observation;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observation;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    auto& observed = *static_cast<Observation*>(context);
    for (const auto& cell : fields.cells) {
      ++observed.cells;
      observed.saw_nonzero_pressure |= std::abs(cell.previous_pressure) > 1e-8;
      EXPECT_NEAR(cell.bulk_modulus, 4000.0 + 7.0 * cell.previous_pressure, 1e-10);
      EXPECT_TRUE(cell.density.has_value());
      const double expected_density = 1000.0 * std::pow(1.0 + 7.0 * cell.previous_pressure / 4000.0, 1.0 / 7.0);
      EXPECT_NEAR(cell.density.value_or(0.0), expected_density, 1e-10);
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.ambient_pressure = 10.0;
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0, .law = LiquidEosLaw::kTait};
  auto scheduler = LiquidSubstepScheduler3D::Create(Initial(), scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto first = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(first.ok()) << first;
  const auto checkpoint = EncodeLiquidState3D(scheduler->committed());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const auto restored = DecodeLiquidState3D(checkpoint->values());
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto resumed = LiquidSubstepScheduler3D::Create(*restored, scene, Options());
  ASSERT_TRUE(resumed.ok()) << resumed.status();
  observation = {};
  const auto second = scheduler->AdvanceFrame(0.02);
  ASSERT_TRUE(second.ok()) << second;
  const auto resumed_step = resumed->AdvanceFrame(0.02);
  ASSERT_TRUE(resumed_step.ok()) << resumed_step;
  ExpectIdenticalSerializedState(scheduler->committed(), resumed->committed());
  EXPECT_GT(observation.cells, 0U);
  EXPECT_TRUE(observation.saw_nonzero_pressure);
}

TEST(LiquidSubstep3D, ExpandedEosDensityTightensCapillaryStep) {
  const auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 1.0};
  scene.ambient_pressure = -1000.0 * std::log(4.0);
  scene.surface_tension = 1000.0 / (4.0 * std::numbers::pi * 0.1 * 0.1);
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.08).ok());
  const auto attempted = scheduler->Attempt(0.08);
  ASSERT_TRUE(attempted.ok()) << attempted.status();
  EXPECT_EQ(attempted->disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kInterface);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, IncompressibleDriftBalancesSealedBandAndChecksPhysicalSources) {
  struct Observation {
    std::vector<MacPressureFace3D> sealed_faces;
    std::vector<MacPressureCell3D> corrected_cells;
  } observation;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observation;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    auto result = CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
    if (result.ok()) {
      auto& faces = static_cast<Observation*>(context)->sealed_faces;
      faces.assign(result->faces.begin(), result->faces.end());
      for (auto& face : faces) {
        if (std::holds_alternative<DirichletMacPressureLink3D>(face.pressure_link)) {
          face.open_area = 0.0;
          face.pressure_link = SealedMacPressureLink3D{};
        }
      }
      result->faces = faces;
    }
    return result;
  };
  backend.project = [](void* context,
                       SparseMacGridView3D grid,
                       double duration,
                       const LiquidPressureProblemView3D& problem,
                       MacPressureSolverOptions3D options,
                       MacPressureWorkspace3D& workspace) {
    auto& corrected = static_cast<Observation*>(context)->corrected_cells;
    corrected.assign(problem.cells.begin(), problem.cells.end());
    return CpuLiquidSubstepBackend3D().project(nullptr, grid, duration, problem, options, workspace);
  };
  auto initial = Initial(2);
  initial.particles[1].world_position.x = 3.5;
  initial.particles[0].volume_ratio = 2.0;
  LiquidSceneInputs3D scene;
  scene.field = LiquidHorizontalPlaneSurface3D{3.0};
  scene.obstacle = AxisAlignedLiquidTank3D{2.0, 4.0, 2.0, 2.0, 4.0, 0.0};
  scene.world_acceleration = {};
  scene.drift = LiquidSceneDrift3D{.correction_time = 1.0, .maximum_rate = 0.3};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  ASSERT_EQ(observation.corrected_cells.size(), 4U);
  EXPECT_NEAR(observation.corrected_cells[0].target_divergence, -0.3, 1e-14);
  EXPECT_NEAR(observation.corrected_cells[1].target_divergence, 0.3, 1e-14);
  EXPECT_EQ(observation.corrected_cells[2].target_divergence, 0.0);
  EXPECT_EQ(observation.corrected_cells[3].target_divergence, 0.0);

  observation.corrected_cells.clear();
  scene.target_divergence = 0.01;
  auto incompatible = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(incompatible.ok()) << incompatible.status();
  EXPECT_EQ(incompatible->AdvanceFrame(0.01).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(observation.corrected_cells.empty());
  ExpectIdenticalSerializedState(initial, incompatible->committed());
}

TEST(LiquidSubstep3D, FiniteEosDriftUsesOnlySensedCells) {
  struct Observation {
    std::size_t sensed = 0;
    std::size_t unsensed = 0;
  } observation;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observation;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    auto& observed = *static_cast<Observation*>(context);
    for (std::size_t index = 0; index < geometry.cells.size(); ++index) {
      const auto cell = geometry.cells[index].cell_index;
      if (cell.x == 2 && cell.y == 2 && cell.z == 2) {
        ++observed.sensed;
        EXPECT_NEAR(fields.cells[index].target_divergence, 0.1 - 0.3, 1e-15);
      } else {
        ++observed.unsensed;
        EXPECT_EQ(fields.cells[index].target_divergence, 0.1);
      }
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  auto initial = Initial();
  initial.particles[0].volume_ratio = 2.0;
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.target_divergence = 0.1;
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0};
  scene.drift = LiquidSceneDrift3D{.correction_time = 0.2, .maximum_rate = 0.3};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  EXPECT_EQ(observation.sensed, 1U);
  EXPECT_GT(observation.unsensed, 0U);
  initial.particles[0].side_tag = 1;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, scene, Options()).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidSubstep3D, TaitRejectsCandidateOutsideDomainBeforePublication) {
  std::vector<double> candidate_pressures;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &candidate_pressures;
  backend.project = [](void* context,
                       SparseMacGridView3D grid,
                       double step_duration,
                       const LiquidPressureProblemView3D& problem,
                       MacPressureSolverOptions3D options,
                       MacPressureWorkspace3D& workspace) {
    auto result = CpuLiquidSubstepBackend3D().project(nullptr, grid, step_duration, problem, options, workspace);
    if (result.ok()) {
      auto& pressures = *static_cast<std::vector<double>*>(context);
      pressures.assign(result->cell_pressures.size(), -1000.0);
      result->cell_pressures = pressures;
    }
    return result;
  };
  const auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0, .law = LiquidEosLaw::kTait};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0.01);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(candidate_pressures.empty());
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, UnsupportedViscousGeometryPreservesCommittedState) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  const auto initial = Initial();
  auto scene = Scene();
  scene.dynamic_viscosity = 1.0;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.project_viscous = nullptr;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend).status().code(),
      absl::StatusCode::kUnimplemented
  );

  const auto check_rejection = [&](absl::StatusCode expected) {
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    const auto attempt = scheduler->Attempt(0.01);
    ASSERT_TRUE(attempt.ok()) << attempt.status();
    EXPECT_EQ(attempt->disposition, LiquidAttemptDisposition3D::kFatal);
    EXPECT_EQ(scheduler->diagnostics().failure_code, expected);
    ExpectIdenticalSerializedState(initial, scheduler->committed());
  };
  check_rejection(absl::StatusCode::kFailedPrecondition);
  scene.obstacle = TranslatingHalfspace3D{{0.0, 1.0, 0.0}, 2.25, {}};
  check_rejection(absl::StatusCode::kUnimplemented);
}

TEST(LiquidSubstep3D, OceanInletRetiresDryParticleAndDiscardsRejectedBirths) {
  auto seed = Initial(2);
  seed.particles[0].world_position = {2.25, 1.25, 2.25};
  seed.particles[1].world_position = {2.25, 3.25, 2.25};
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{1.1};
  scene.world_acceleration = {};
  auto options = Options();
  options.ocean_boundary = OceanBoundarySettings3D{
      .swell =
          {
              .still_water_height_m = 3,
              .water_depth_m = 3,
              .wave_height_m = 0.5,
              .wave_period_s = 4,
              .phase_at_world_origin_rad = std::numbers::pi,
          },
      .inlet_wall_x_m = 2,
      .inlet_interior_x_m = 2.5,
      .outlet_interior_x_m = 5,
      .outlet_wall_x_m = 8,
      .inlet_particle_spacing_m = 0.5,
      .inlet_low_z_m = 2,
      .inlet_high_z_m = 2.5,
  };
  auto rejected = LiquidSubstepScheduler3D::Create(seed, scene, options);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  rejected->SetStageFaultHookForTesting({
      nullptr,
      [](void*, LiquidPhysicalStage3D stage, double) {
        return stage == LiquidPhysicalStage3D::kActivate ? absl::InternalError("reject after inlet edit")
                                                         : absl::OkStatus();
      },
  });
  ASSERT_TRUE(rejected->BeginFrame(0.01).ok());
  const auto attempt = rejected->Attempt(0.01);
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  EXPECT_EQ(attempt->disposition, LiquidAttemptDisposition3D::kFatal);
  ExpectIdenticalSerializedState(seed, rejected->committed());

  auto accepted = LiquidSubstepScheduler3D::Create(seed, scene, options);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  const auto advanced = accepted->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  const auto& state = accepted->committed();
  EXPECT_FALSE(std::ranges::any_of(state.particles.values(), [](const auto& particle) {
    return particle.id == 2;
  }));
  EXPECT_TRUE(std::ranges::any_of(state.particles.values(), [](const auto& particle) {
    return particle.id == 1;
  }));
  EXPECT_GT(state.next_particle_id, seed.next_particle_id);
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

TEST(LiquidSubstep3D, AcceptedPistonAttemptReturnsSolvedPressureLoad) {
  namespace wet = wetted_piston_pressure_test;
  auto scheduler = LiquidSubstepScheduler3D::Create({}, wet::PlaneScene(0.75, wet::kPistonSpeed), Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(wet::kDuration).ok());
  const RigidBody3D piston{.body_id = 17, .mass = 1.0, .principal_inertia = {1.0, 1.0, 1.0}};
  const auto attempt = scheduler->AttemptWithPlanarPistonPressureLoad(wet::kDuration, piston);
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  ASSERT_EQ(attempt->outcome.disposition, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_TRUE(attempt->loaded_piston.has_value());
  const auto cells = scheduler->committed().cells.values();
  ASSERT_EQ(cells.size(), 1);
  EXPECT_NEAR(attempt->loaded_piston->force.x, -0.75 * cells.front().previous_pressure, 1e-10);
  EXPECT_LT(attempt->loaded_piston->force.x, 0.0);
  EXPECT_DOUBLE_EQ(piston.force.x, 0.0);

  auto rejected = LiquidSubstepScheduler3D::Create({}, wet::PlaneScene(0.75, wet::kPistonSpeed), Options());
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_TRUE(rejected->BeginFrame(wet::kDuration).ok());
  rejected->SetStageFaultHookForTesting({
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("") : absl::OkStatus();
          },
  });
  const auto refused = rejected->AttemptWithPlanarPistonPressureLoad(wet::kDuration, piston);
  ASSERT_TRUE(refused.ok()) << refused.status();
  EXPECT_EQ(refused->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_FALSE(refused->loaded_piston.has_value());
  EXPECT_DOUBLE_EQ(rejected->committed().clock.time, 0.0);
}

TEST(LiquidSubstep3D, CpuPistonReactionCountsStoredMomentumAtMovingFaceOnly) {
  std::array particles{
      LiquidParticleState3D{.world_position = {0.0, 0.0, 0.5}, .mass = 2.0},
      LiquidParticleState3D{.world_position = {1.0, 0.5, 0.5}, .mass = 3.0},
  };
  const std::array gathers{
      LiquidBoundaryParticleGather3D{.new_pic_velocity = {-0.125, -0.125, 0.0}},
      LiquidBoundaryParticleGather3D{.new_pic_velocity = {0.125, 0.0, 0.0}},
  };
  constexpr std::array distances{-1.0, -1.0};
  const LiquidParticleUpdateInputs3D inputs{
      .gathered = gathers,
      .liquid_distances = distances,
      .obstacle = AxisAlignedLiquidTank3D{0.0, 1.0, 0.0, 0.0, 1.0, 0.25},
      .cell_width = 1.0,
      .step_duration = 0.125,
  };
  const auto impulse = UpdateCpuLiquidParticlesWithPistonReaction3D(inputs, particles);
  ASSERT_TRUE(impulse.ok()) << impulse.status();
  EXPECT_DOUBLE_EQ(*impulse, -0.75);
  EXPECT_DOUBLE_EQ(particles[0].world_velocity.x, 0.25);
  EXPECT_DOUBLE_EQ(particles[0].world_velocity.y, 0.0);
  EXPECT_DOUBLE_EQ(particles[1].world_velocity.x, 0.0);
}

TEST(LiquidSubstep3D, GuidedPistonCommitsPressureAndParticleReactionTogether) {
  namespace wet = wetted_piston_pressure_test;
  auto initial = Initial();
  initial.particles[0].world_position = {0x1p-13, 0.375, 0.5};
  initial.particles[0].mass = 2.0;
  initial.particles[0].reference_volume = 2.0;
  std::array gathered{
      LiquidBoundaryParticleGather3D{.new_pic_velocity = {-0.125, 0.0, 0.0}},
  };
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &gathered;
  backend.gather = [](void* context,
                      const LiquidBoundaryGatherViewInputs3D&,
                      LiquidBoundaryWorkspace3D&) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    return LiquidBoundaryGatherView3D{
        .particles = *static_cast<std::array<LiquidBoundaryParticleGather3D, 1>*>(context)
    };
  };
  auto scene = wet::PlaneScene(0.75, wet::kPistonSpeed);
  auto& tank = std::get<AxisAlignedLiquidTank3D>(scene.obstacle);
  tank.high_x = 4.0;
  tank.low_z = -1.0;
  tank.high_z = 2.0;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(wet::kDuration).ok());
  const RigidBody3D piston{
      .body_id = 17,
      .mass = 1000.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .linear_velocity = {wet::kPistonSpeed, 0.0, 0.0},
  };
  const auto attempt = scheduler->AttemptWithDynamicPlanarPiston(wet::kDuration, piston);
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  ASSERT_TRUE(attempt->loaded_piston.has_value())
      << scheduler->diagnostics().failure_code << " stage " << static_cast<int>(scheduler->diagnostics().failed_stage);
  auto prescribed = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(prescribed.ok()) << prescribed.status();
  ASSERT_TRUE(prescribed->BeginFrame(wet::kDuration).ok());
  const auto load = prescribed->AttemptWithPlanarPistonPressureLoad(wet::kDuration, piston);
  ASSERT_TRUE(load.ok()) << load.status();
  ASSERT_TRUE(load->loaded_piston.has_value());
  const double pressure_impulse = wet::kDuration * load->loaded_piston->force.x;
  constexpr double kPistonReaction = -2.0 * (wet::kPistonSpeed + 0.125);
  EXPECT_NEAR(
      attempt->loaded_piston->linear_velocity.x,
      wet::kPistonSpeed + (pressure_impulse + kPistonReaction) / piston.mass,
      1e-12
  );
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].world_velocity.x, wet::kPistonSpeed);

  ASSERT_TRUE(scheduler->FinishFrame().ok());
  const auto accepted_body = *attempt->loaded_piston;
  const SolidInteractionCheckpoint3D contact_solid{
      .accepted_step = scheduler->committed().clock.accepted_step_ordinal,
      .bodies = {{accepted_body, {piston.center, piston.orientation}}},
      .sides = {},
  };
  const auto checkpoint = EncodeCoupledInteractionCheckpoint3D(scheduler->committed(), contact_solid);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const auto restored = DecodeCoupledInteractionCheckpoint3D(*checkpoint);
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_EQ(restored->solid.bodies.size(), 1);
  auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, scene, Options(), backend);
  ASSERT_TRUE(replay.ok()) << replay.status();
  ASSERT_TRUE(scheduler->BeginFrame(2.0 * wet::kDuration).ok());
  ASSERT_TRUE(replay->BeginFrame(2.0 * wet::kDuration).ok());
  replay->SetStageFaultHookForTesting({
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("") : absl::OkStatus();
          },
  });
  const auto rejected = replay->AttemptWithDynamicPlanarPiston(wet::kDuration, restored->solid.bodies.front().body);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_FALSE(rejected->loaded_piston.has_value());
  const auto after_rejection = EncodeCoupledInteractionCheckpoint3D(replay->committed(), restored->solid);
  ASSERT_TRUE(after_rejection.ok()) << after_rejection.status();
  EXPECT_EQ(*after_rejection, *checkpoint);
  replay->SetStageFaultHookForTesting({});

  const auto uninterrupted = scheduler->AttemptWithDynamicPlanarPiston(wet::kDuration / 2.0, accepted_body);
  const auto replayed =
      replay->AttemptWithDynamicPlanarPiston(wet::kDuration / 2.0, restored->solid.bodies.front().body);
  ASSERT_TRUE(uninterrupted.ok()) << uninterrupted.status();
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  ASSERT_TRUE(uninterrupted->loaded_piston.has_value()) << scheduler->diagnostics().failure_code;
  ASSERT_TRUE(replayed->loaded_piston.has_value()) << replay->diagnostics().failure_code;
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].world_velocity.x, accepted_body.linear_velocity.x);
  const RigidBodyPose3D prior_pose{accepted_body.center, accepted_body.orientation};
  const auto expected_bytes = EncodeCoupledInteractionCheckpoint3D(
      scheduler->committed(),
      {.accepted_step = scheduler->committed().clock.accepted_step_ordinal,
       .bodies = {{*uninterrupted->loaded_piston, prior_pose}},
       .sides = {}}
  );
  const auto replayed_bytes = EncodeCoupledInteractionCheckpoint3D(
      replay->committed(),
      {.accepted_step = replay->committed().clock.accepted_step_ordinal,
       .bodies = {{*replayed->loaded_piston, prior_pose}},
       .sides = {}}
  );
  ASSERT_TRUE(expected_bytes.ok()) << expected_bytes.status();
  ASSERT_TRUE(replayed_bytes.ok()) << replayed_bytes.status();
  EXPECT_EQ(*expected_bytes, *replayed_bytes);
}

TEST(LiquidSubstep3D, TranslatingFloatingBoxReceivesLateralPressureFeedback) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 0.875, 0.0};
  auto scheduler = LiquidSubstepScheduler3D::Create({}, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double kDt = 0.125;
  constexpr Vec3 kHalfExtent{0.125, 0.3125, 0.125};
  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {1.0, 0.5625, 0.5},
      .linear_velocity = {0.125, 0.0, 0.0},
  };
  ASSERT_TRUE(scheduler->BeginFrame(kDt).ok());
  const auto attempt = scheduler->AttemptWithTranslatingFloatingBox(kDt, body, kHalfExtent);
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  ASSERT_TRUE(attempt->body.has_value()) << scheduler->diagnostics().failure_code << " stage "
                                         << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_DOUBLE_EQ(attempt->body->center.x, 1.015625);
  EXPECT_LT(attempt->body->linear_velocity.x, body.linear_velocity.x);
  ASSERT_EQ(scheduler->committed().cells.size(), 2);
  const auto cells = scheduler->committed().cells.values();
  EXPECT_LT(cells[0].previous_pressure, 0.0);
  EXPECT_GT(cells[1].previous_pressure, 0.0);
  const double wet_side_area = 0.125;
  const double lateral_force = wet_side_area * (cells[0].previous_pressure - cells[1].previous_pressure);
  EXPECT_NEAR(attempt->body->linear_velocity.x, body.linear_velocity.x + kDt * lateral_force / body.mass, 1e-12);
  EXPECT_LE(scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect, 2e-5);
}
TEST(LiquidSubstep3D, TranslatingFloatingBoxCommitsLateralParticleImpulse) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
  auto initial = Initial();
  initial.particles[0].world_position = {0.375, 0.375, 0.375};
  initial.particles[0].mass = 2.0;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.gather = [](void*,
                      const LiquidBoundaryGatherViewInputs3D&,
                      LiquidBoundaryWorkspace3D&) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    static constexpr std::array gathered{
        LiquidBoundaryParticleGather3D{.new_pic_velocity = {0.75, 0.0, 0.0}},
    };
    return LiquidBoundaryGatherView3D{.particles = gathered};
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr Vec3 kHalfExtent{0.125, 0.3125, 0.125};
  const RigidBody3D body{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.5625, 0.5625, 0.375},
  };
  ASSERT_TRUE(scheduler->BeginFrame(0.25).ok());
  scheduler->SetStageFaultHookForTesting({nullptr, [](void*, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("")
                                                                                           : absl::OkStatus();
                                          }});
  const auto rejected = scheduler->AttemptWithTranslatingFloatingBox(0.25, body, kHalfExtent);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kAccept);
  EXPECT_FALSE(rejected->body.has_value());
  EXPECT_EQ(scheduler->committed(), initial);
  scheduler->SetStageFaultHookForTesting({});

  const auto accepted = scheduler->AttemptWithTranslatingFloatingBox(0.125, body, kHalfExtent);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_TRUE(accepted->body.has_value());
  EXPECT_DOUBLE_EQ(accepted->body->linear_velocity.x, 0.75);
  EXPECT_DOUBLE_EQ(accepted->body->angular_velocity.z, 0.0);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].world_velocity.x, 0.0);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].world_position.x, 0.4375);
}
TEST(LiquidSubstep3D, GuidedFloatingBoxProductionGatherBalancesHydrostaticWeight) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
  scene.world_acceleration = {0.0, -4.0, 0.0};
  auto initial = Initial(4);
  initial.particles[0].world_position = {0.375, 0.1875, 0.375};
  initial.particles[1].world_position = {1.375, 0.375, 0.375};
  initial.particles[2].world_position = {0.375, 0.375, 1.375};
  initial.particles[3].world_position = {1.375, 0.375, 1.375};
  constexpr Vec3 kHalfExtent{0.125, 0.25, 0.125};
  constexpr double kInitialDraft = 0.25;
  constexpr double kBodyMass = 0.015625;
  constexpr double kWeight = 0.0625;
  for (const double duration : {0.03125, 0.015625}) {
    SCOPED_TRACE(duration);
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    RigidBody3D body{
        .body_id = 23,
        .mass = kBodyMass,
        .principal_inertia = {0.001, 0.001, 0.001},
        .center = {0.375, 0.75, 0.375},
    };
    double max_draft_error = 0.0;
    double max_net_force = 0.0;
    double max_defect = 0.0;
    for (int frame = 1; frame <= static_cast<int>(0.125 / duration); ++frame) {
      body.force.y = -kWeight;
      ASSERT_TRUE(scheduler->BeginFrame(frame * duration).ok());
      const auto attempt = scheduler->AttemptWithGuidedFloatingBox(duration, body, kHalfExtent);
      ASSERT_TRUE(attempt.ok()) << attempt.status();
      ASSERT_TRUE(attempt->body.has_value()) << scheduler->diagnostics().failure_code << " stage "
                                             << static_cast<int>(scheduler->diagnostics().failed_stage);
      const double draft = 0.75 - (attempt->body->center.y - kHalfExtent.y);
      max_draft_error = std::max(max_draft_error, std::abs(draft - kInitialDraft));
      max_net_force = std::max(
          max_net_force,
          std::abs(kBodyMass * (attempt->body->linear_velocity.y - body.linear_velocity.y) / duration)
      );
      max_defect = std::max(max_defect, scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect);
      for (const auto& cell : scheduler->committed().cells.values()) {
        EXPECT_NEAR(cell.previous_pressure, 1.0, 2e-5);
      }
      body = *attempt->body;
      ASSERT_TRUE(scheduler->FinishFrame().ok());
    }
    std::cout << "floating_box_equilibrium_dt_s=" << duration << " max_draft_error_m=" << max_draft_error
              << " max_net_force_n=" << max_net_force << " max_constraint_defect_per_s=" << max_defect << '\n';
    EXPECT_LE(max_draft_error, 1e-6);
    EXPECT_LE(max_net_force, 2e-6);
    EXPECT_LE(max_defect, 2e-5);
  }
}

void ExpectOrientedFloatingBoxHydrostaticWeight(bool explicit_composition) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
  scene.world_acceleration = {0.0, -4.0, 0.0};
  auto initial = Initial(4);
  initial.particles[0].world_position = {0.375, 0.1875, 0.375};
  initial.particles[1].world_position = {1.375, 0.375, 0.375};
  initial.particles[2].world_position = {0.375, 0.375, 1.375};
  initial.particles[3].world_position = {1.375, 0.375, 1.375};
  constexpr std::array<LiquidConstituentMaterial3D, 1> kMaterials{{{8, 4, 1.0}}};
  if (explicit_composition) {
    scene.constituent_materials = kMaterials;
    for (auto& particle : initial.particles.values()) {
      particle.phase_id = 4;
      particle.constituent_count = 1;
      particle.constituents[0] = {8, 1.0};
    }
  }
  constexpr Vec3 kHalfExtent{0.125, 0.125, 0.125};
  constexpr double kInitialDraft = 0.125;
  constexpr double kBodyMass = 0.0078125;
  constexpr double kWeight = 0.03125;
  for (const double duration : {0.03125, 0.015625}) {
    SCOPED_TRACE(duration);
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    RigidBody3D body{
        .body_id = 23,
        .mass = kBodyMass,
        .principal_inertia = {0.001, 0.001, 0.001},
        .center = {0.375, 0.75, 0.375},
    };
    double max_draft_error = 0.0;
    double max_net_force = 0.0;
    double max_defect = 0.0;
    for (int frame = 1; frame <= static_cast<int>(0.125 / duration); ++frame) {
      body.force.y = -kWeight;
      ASSERT_TRUE(scheduler->BeginFrame(frame * duration).ok());
      const auto attempt = scheduler->AttemptWithOrientedFloatingBox(duration, body, kHalfExtent);
      ASSERT_TRUE(attempt.ok()) << attempt.status();
      ASSERT_TRUE(attempt->body.has_value()) << scheduler->diagnostics().failure_code << " stage "
                                             << static_cast<int>(scheduler->diagnostics().failed_stage);
      ASSERT_TRUE(attempt->oriented_loads.has_value());
      EXPECT_DOUBLE_EQ(attempt->oriented_loads->external_force.y, -kWeight);
      EXPECT_NEAR(attempt->oriented_loads->pressure_force.y, kWeight, 2e-6);
      ASSERT_TRUE(attempt->oriented_loads->tank_pressure.has_value());
      const auto& tank_pressure = *attempt->oriented_loads->tank_pressure;
      EXPECT_NEAR(tank_pressure.force.x, 0.0, 2e-6);
      EXPECT_NEAR(tank_pressure.force.y, 7.65625, 2e-6);
      EXPECT_NEAR(tank_pressure.force.z, 0.0, 2e-6);
      EXPECT_NEAR(tank_pressure.torque.x, -7.65625, 2e-6);
      EXPECT_NEAR(tank_pressure.torque.y, 0.0, 2e-6);
      EXPECT_NEAR(tank_pressure.torque.z, 7.65625, 2e-6);
      const double draft = 0.75 - (attempt->body->center.y - kHalfExtent.y);
      max_draft_error = std::max(max_draft_error, std::abs(draft - kInitialDraft));
      max_net_force = std::max(
          max_net_force,
          std::abs(kBodyMass * (attempt->body->linear_velocity.y - body.linear_velocity.y) / duration)
      );
      max_defect = std::max(max_defect, scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect);
      for (const auto& cell : scheduler->committed().cells.values()) {
        EXPECT_NEAR(cell.previous_pressure, 1.0, 2e-5);
      }
      EXPECT_NEAR(attempt->body->angular_velocity.x, 0.0, 1e-8);
      EXPECT_NEAR(attempt->body->angular_velocity.z, 0.0, 1e-8);
      body = *attempt->body;
      ASSERT_TRUE(scheduler->FinishFrame().ok());
    }
    std::cout << "oriented_equilibrium_dt_s=" << duration << " max_draft_error_m=" << max_draft_error
              << " max_net_force_n=" << max_net_force << " max_constraint_defect_per_s=" << max_defect << '\n';
    EXPECT_LE(max_draft_error, 1e-6);
    EXPECT_LE(max_net_force, 2e-6);
    EXPECT_LE(max_defect, 2e-5);
  }
}

TEST(LiquidSubstep3D, OrientedFloatingBoxProductionGatherBalancesHydrostaticWeight) {
  ExpectOrientedFloatingBoxHydrostaticWeight(false);
}
TEST(LiquidSubstep3D, OrientedFloatingBoxBalancesExplicitSingleConstituentHydrostaticWeight) {
  ExpectOrientedFloatingBoxHydrostaticWeight(true);
}

TEST(LiquidSubstep3D, GuidedFloatingBoxMeasuresAddedMassAndLightBodyAmplification) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 0.875, 0.125, 0.125, 0.875, 0.0};
  constexpr Vec3 kHalfExtent{0.125, 0.3125, 0.125};
  constexpr double kDt = 0.125;
  constexpr double kAddedMass = 1.0 / 576.0;
  constexpr double kSpeed = 0.125;
  const auto advance = [&](double mass, double speed, bool coupled = false, bool reject = false
                       ) -> std::optional<RigidBody3D> {
    auto scheduler = LiquidSubstepScheduler3D::Create({}, scene, Options());
    EXPECT_TRUE(scheduler.ok()) << scheduler.status();
    if (!scheduler.ok() || !scheduler->BeginFrame(kDt).ok()) {
      return std::nullopt;
    }
    const RigidBody3D body{
        .body_id = 23,
        .mass = mass,
        .principal_inertia = {1.0, 1.0, 1.0},
        .center = {0.375, 0.5625, 0.375},
        .linear_velocity = {0.0, speed, 0.0},
    };
    if (reject) {
      scheduler->SetStageFaultHookForTesting({nullptr, [](void*, LiquidPhysicalStage3D stage, double) {
                                                return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("")
                                                                                               : absl::OkStatus();
                                              }});
    }
    const auto attempt = coupled ? scheduler->AttemptWithCoupledGuidedFloatingBox(kDt, body, kHalfExtent)
                                 : scheduler->AttemptWithGuidedFloatingBox(kDt, body, kHalfExtent);
    if (reject && attempt.ok()) {
      EXPECT_EQ(attempt->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
      EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kAccept);
      EXPECT_FALSE(attempt->added_mass_kg.has_value());
      EXPECT_DOUBLE_EQ(scheduler->committed().clock.time, 0.0);
    }
    if (coupled && attempt.ok() && attempt->body.has_value()) {
      EXPECT_TRUE(attempt->added_mass_kg.has_value());
      EXPECT_NEAR(attempt->added_mass_kg.value_or(0.0), kAddedMass, 1e-9);
      EXPECT_NEAR(scheduler->committed().cells.values().front().previous_pressure, -1.0 / 180.0, 1e-7);
    }
    EXPECT_TRUE(attempt.ok()) << attempt.status();
    return attempt.ok() ? attempt->body : std::nullopt;
  };

  const auto positive = advance(1.0, kSpeed);
  const auto negative = advance(1.0, -kSpeed);
  ASSERT_TRUE(positive && negative);
  const double added_mass = 1.0 - (positive->linear_velocity.y - negative->linear_velocity.y) / (2 * kSpeed);
  EXPECT_NEAR(added_mass, kAddedMass, 1e-9);
  const auto ordinary = advance(2 * kAddedMass, kSpeed);
  const auto light = advance(0.25 * kAddedMass, kSpeed);
  ASSERT_TRUE(ordinary && light);
  const double ordinary_gain = ordinary->linear_velocity.y / kSpeed;
  const double light_gain = light->linear_velocity.y / kSpeed;
  EXPECT_NEAR(ordinary_gain, 0.5, 1e-6);
  EXPECT_NEAR(light_gain, -3.0, 1e-6);
  const auto coupled_light = advance(0.25 * kAddedMass, kSpeed, true);
  ASSERT_TRUE(coupled_light);
  EXPECT_FALSE(advance(0.25 * kAddedMass, kSpeed, true, true));
  const double coupled_gain = coupled_light->linear_velocity.y / kSpeed;
  EXPECT_NEAR(coupled_gain, 0.2, 1e-6);
  EXPECT_NEAR(coupled_light->center.y, 0.5625 + kDt * 0.2 * kSpeed, 1e-8);
  std::cout << "floating_box_added_mass_kg=" << added_mass << " ordinary_step_gain=" << ordinary_gain
            << " light_step_gain=" << light_gain << " coupled_light_step_gain=" << coupled_gain << '\n';
}
TEST(LiquidSubstep3D, GuidedFloatingBoxProductionGatherReplaysTrajectory) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
  auto initial = Initial(4);
  initial.particles[0].world_position = {0.375, 0.1875, 0.375};
  initial.particles[1].world_position = {1.375, 0.375, 0.375};
  initial.particles[2].world_position = {0.375, 0.375, 1.375};
  initial.particles[3].world_position = {1.375, 0.375, 1.375};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double kDt = 0.015625;
  constexpr Vec3 kHalfExtent{0.125, 0.3125, 0.125};
  const RigidBody3D initial_body{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.375, 0.5625, 0.375},
      .linear_velocity = {0.0, 0.125, 0.0},
  };
  auto body = initial_body;
  std::vector<std::uint8_t> midpoint;
  std::vector<std::uint8_t> final;
  bool observed_pressure = false;
  bool observed_particle_motion = false;
  for (int frame = 1; frame <= 4; ++frame) {
    SCOPED_TRACE(frame);
    ASSERT_TRUE(scheduler->BeginFrame(frame * kDt).ok());
    const auto attempt = scheduler->AttemptWithGuidedFloatingBox(kDt, body, kHalfExtent);
    ASSERT_TRUE(attempt.ok()) << attempt.status();
    ASSERT_TRUE(attempt->body.has_value()) << scheduler->diagnostics().failure_code << " stage "
                                           << static_cast<int>(scheduler->diagnostics().failed_stage);
    EXPECT_NEAR(attempt->body->center.y, body.center.y + kDt * body.linear_velocity.y, 1e-14);
    EXPECT_LE(scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect, 2e-5);
    for (const auto& cell : scheduler->committed().cells.values()) {
      observed_pressure |= std::abs(cell.previous_pressure) > 1e-5;
    }
    const auto position = scheduler->committed().particles[0].world_position;
    observed_particle_motion |= std::abs(position.y - initial.particles[0].world_position.y) > 1e-9;
    const auto previous_pose = RigidBodyPose3D{body.center, body.orientation};
    body = *attempt->body;
    ASSERT_TRUE(scheduler->FinishFrame().ok());
    if (frame == 2 || frame == 4) {
      const auto checkpoint = EncodeCoupledInteractionCheckpoint3D(
          scheduler->committed(),
          {.accepted_step = static_cast<std::uint64_t>(frame), .bodies = {{body, previous_pose}}}
      );
      ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
      (frame == 2 ? midpoint : final) = *checkpoint;
    }
  }
  EXPECT_TRUE(observed_pressure);
  EXPECT_TRUE(observed_particle_motion);
  EXPECT_LT(body.linear_velocity.y, initial_body.linear_velocity.y);
  const auto restored = DecodeCoupledInteractionCheckpoint3D(midpoint);
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, scene, Options());
  ASSERT_TRUE(replay.ok()) << replay.status();
  auto history = restored->solid.bodies.front();
  for (int frame = 3; frame <= 4; ++frame) {
    ASSERT_TRUE(replay->BeginFrame(frame * kDt).ok());
    const auto attempt = replay->AttemptWithGuidedFloatingBox(kDt, history.body, kHalfExtent);
    ASSERT_TRUE(attempt.ok()) << attempt.status();
    ASSERT_TRUE(attempt->body.has_value());
    history = {*attempt->body, {history.body.center, history.body.orientation}};
    ASSERT_TRUE(replay->FinishFrame().ok());
  }
  const auto replayed =
      EncodeCoupledInteractionCheckpoint3D(replay->committed(), {.accepted_step = 4, .bodies = {history}});
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  EXPECT_EQ(*replayed, final);
}
Vec3 StoredLiquidMomentum(std::span<const LiquidParticleState3D> particles) {
  Vec3 momentum{};
  for (const auto& particle : particles) {
    momentum.x += particle.mass * particle.world_velocity.x;
    momentum.y += particle.mass * particle.world_velocity.y;
    momentum.z += particle.mass * particle.world_velocity.z;
  }
  return momentum;
}

void ExpectStoredLiquidMomentumBudget(
    Vec3 initial_momentum,
    std::span<const LiquidParticleState3D> particles,
    const OrientedFloatingBoxLoads3D& loads
) {
  const auto final_momentum = StoredLiquidMomentum(particles);
  for (const auto axis : {&Vec3::x, &Vec3::y, &Vec3::z}) {
    const double measured_change = final_momentum.*axis - initial_momentum.*axis;
    const double recorded_change =
        loads.liquid_gather_impulse.*axis + loads.tank_particle_impulse.*axis - loads.particle_impulse.*axis;
    EXPECT_NEAR(measured_change, recorded_change, 1e-12);
  }
}

Vec3 ExpectedPressureGridImpulse(
    const LiquidPressureProblemView3D& problem,
    std::span<const double> pressures,
    double duration
) {
  constexpr std::array kCoordinates{&CellIndex3D::x, &CellIndex3D::y, &CellIndex3D::z};
  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  Vec3 impulse{};
  for (const auto& face : problem.faces) {
    if (face.open_area == 0.0) {
      continue;
    }
    const auto axis = static_cast<std::size_t>(face.face_index.axis);
    const auto index = face.face_index.lattice_index;
    const CellIndex3D head{index.x, index.y, index.z};
    auto tail = head;
    --(tail.*kCoordinates[axis]);
    const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link);
    const auto pressure_at = [&](CellIndex3D coordinate) {
      for (std::size_t ordinal = 0; ordinal < problem.cells.size(); ++ordinal) {
        const auto cell = problem.cells[ordinal].cell_index;
        if (cell.x == coordinate.x && cell.y == coordinate.y && cell.z == coordinate.z) {
          return pressures[ordinal];
        }
      }
      EXPECT_NE(boundary, nullptr);
      return boundary ? boundary->exterior_pressure : 0.0;
    };
    const auto* internal = std::get_if<InternalMacPressureLink3D>(&face.pressure_link);
    const double jump = internal ? internal->pressure_jump : 0.0;
    impulse.*kComponents[axis] += duration * face.open_area * (pressure_at(tail) - pressure_at(head) + jump);
  }
  return impulse;
}

double PressureTractionImpulseResidual(const OrientedFloatingBoxLoads3D& loads, double duration) {
  const auto& tank = loads.tank_pressure->force;
  const auto& box = loads.pressure_force;
  const auto grid = loads.pressure_grid_impulse;
  return std::hypot(
      grid.x - duration * (tank.x - box.x),
      grid.y - duration * (tank.y - box.y),
      grid.z - duration * (tank.z - box.z)
  );
}

struct PressureFaceMeasurement {
  std::size_t slot;
  FaceNormalAxis axis;
  double predicted_velocity;
  double velocity_change;
  double pressure_dual_mass;
  double deposited_mass;
  Vec3 gathered_impulse{};
};

struct FloatingBoxPressureMeasurements {
  double energy_blend;
  double wall_flux{};
  double wet_volume{};
  Vec3 grid_impulse{};
  Vec3 particle_pressure_impulse{};
  Vec3 particle_prepressure_impulse{};
  std::vector<PressureFaceMeasurement> pressure_faces;
  std::vector<double> deposited_face_masses;
  std::vector<LiquidBoundaryParticle3D> deposit_particles;
  std::vector<LiquidCompatibleStencils3D> deposit_stencils;
  std::vector<double> normal_face_impulses;
  std::vector<double> face_mass_scratch;
  std::vector<Vec3> compatible_particle_impulses;
  std::vector<double> prepressure_velocities;
};

LiquidSubstepBackend3D MeasureFloatingBoxPressure(FloatingBoxPressureMeasurements& measurements) {
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &measurements;
  backend.deposit = [](void* context,
                       const LiquidBoundaryDepositViewInputs3D& inputs,
                       LiquidBoundaryWorkspace3D& workspace) -> absl::StatusOr<LiquidBoundaryFaceDepositsView3D> {
    auto& measured = *static_cast<FloatingBoxPressureMeasurements*>(context);
    auto result = CpuLiquidSubstepBackend3D().deposit(nullptr, inputs, workspace);
    if (!result.ok()) {
      return result.status();
    }
    measured.deposit_particles.assign(inputs.particles.begin(), inputs.particles.end());
    measured.deposit_stencils.assign(result->compatible_stencils.begin(), result->compatible_stencils.end());
    measured.deposited_face_masses.clear();
    for (const auto& quanta : result->face_quanta) {
      measured.deposited_face_masses.push_back(
          static_cast<double>(quanta.mass_quanta_sum) * inputs.quantization.mass_per_quantum
      );
    }
    return result;
  };
  backend.project = [](void* context,
                       SparseMacGridView3D grid,
                       double duration,
                       const LiquidPressureProblemView3D& problem,
                       MacPressureSolverOptions3D options,
                       MacPressureWorkspace3D& workspace) -> absl::StatusOr<MacPressureProjectionView3D> {
    auto& measured = *static_cast<FloatingBoxPressureMeasurements*>(context);
    auto result = CpuLiquidSubstepBackend3D().project(nullptr, grid, duration, problem, options, workspace);
    if (!result.ok()) {
      return result.status();
    }
    measured.wall_flux = 0.0;
    measured.wet_volume = 0.0;
    for (const auto& cell : problem.cells) {
      measured.wet_volume += cell.fluid_volume;
      measured.wall_flux += cell.integrated_inward_wall_flux;
    }
    measured.grid_impulse = ExpectedPressureGridImpulse(problem, result->cell_pressures, duration);
    measured.particle_pressure_impulse = {};
    measured.particle_prepressure_impulse = {};
    measured.pressure_faces.clear();
    measured.normal_face_impulses.assign(grid.face_count(), 0.0);
    for (std::size_t index = 0; index < problem.faces.size(); ++index) {
      const auto& face = problem.faces[index];
      const auto slot = grid.FaceSlot(face.face_index);
      if (!slot.ok()) {
        return slot.status();
      }
      const double pressure_dual_mass =
          face.open_area == 0.0 ? 0.0 : face.open_area * face.pressure_gradient_distance / face.inverse_density;
      measured.pressure_faces.push_back({
          .slot = *slot,
          .axis = face.face_index.axis,
          .predicted_velocity = face.predicted_normal_velocity,
          .velocity_change = result->corrected_face_velocities[index] - face.predicted_normal_velocity,
          .pressure_dual_mass = pressure_dual_mass,
          .deposited_mass = measured.deposited_face_masses.at(*slot),
      });
      measured.normal_face_impulses[*slot] = pressure_dual_mass * measured.pressure_faces.back().velocity_change;
    }
    measured.face_mass_scratch.resize(grid.face_count());
    measured.compatible_particle_impulses.resize(measured.deposit_particles.size());
    const auto distributed = DistributeCompatiblePressureImpulse3D(
        grid,
        measured.deposit_particles,
        measured.deposit_stencils,
        measured.normal_face_impulses,
        measured.face_mass_scratch,
        measured.compatible_particle_impulses
    );
    if (!distributed.ok()) {
      return distributed;
    }
    return result;
  };
  backend.gather = [](void* context,
                      const LiquidBoundaryGatherViewInputs3D& inputs,
                      LiquidBoundaryWorkspace3D& workspace) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    auto& measured = *static_cast<FloatingBoxPressureMeasurements*>(context);
    if (inputs.particles.size() != 1) {
      return absl::InternalError("floating-box diagnostic expects one particle per gather");
    }
    // Hold the final wall history and validity fixed, changing only the face
    // velocities supplied by projection. The production gather runs last.
    const auto velocities = inputs.new_history.face_velocities;
    const bool has_pressure_correction = std::ranges::any_of(measured.pressure_faces, [&](const auto& face) {
      return velocities[face.slot] != face.predicted_velocity;
    });
    if (!has_pressure_correction) {
      return CpuLiquidSubstepBackend3D().gather(nullptr, inputs, workspace);
    }
    measured.prepressure_velocities.assign(velocities.begin(), velocities.end());
    for (const auto& face : measured.pressure_faces) {
      measured.prepressure_velocities[face.slot] = face.predicted_velocity;
    }
    auto baseline_inputs = inputs;
    baseline_inputs.new_history.face_velocities = measured.prepressure_velocities;
    const auto baseline = CpuLiquidSubstepBackend3D().gather(nullptr, baseline_inputs, workspace);
    if (!baseline.ok()) {
      return baseline.status();
    }
    const auto baseline_particle = baseline->particles.front();
    // The trajectory seeds already have initialized PIC history and the
    // scheduler preserves it. Reproduce the pre-contact stored-velocity blend.
    const auto stored_velocity = [&](const LiquidBoundaryParticleGather3D& gathered, double Vec3::* component) {
      const bool fallback = gathered.old_pic_fallback || gathered.new_pic_fallback || gathered.low_mass_fallback;
      const double alpha = fallback ? 0.0 : 0.99 * measured.energy_blend;
      return gathered.new_pic_velocity.*component + alpha * gathered.particle_minus_old_pic_velocity.*component;
    };
    const auto& particle = inputs.particles.front().transfer;
    for (auto& face : measured.pressure_faces) {
      if (face.velocity_change == 0.0) {
        continue;
      }
      measured.prepressure_velocities[face.slot] = velocities[face.slot];
      const auto single_face = CpuLiquidSubstepBackend3D().gather(nullptr, baseline_inputs, workspace);
      measured.prepressure_velocities[face.slot] = face.predicted_velocity;
      if (!single_face.ok()) {
        return single_face.status();
      }
      for (const auto component : {&Vec3::x, &Vec3::y, &Vec3::z}) {
        const double before = stored_velocity(baseline_particle, component);
        const double after = stored_velocity(single_face->particles.front(), component);
        face.gathered_impulse.*component += particle.particle_mass * (after - before);
      }
    }
    auto result = CpuLiquidSubstepBackend3D().gather(nullptr, inputs, workspace);
    if (!result.ok()) {
      return result.status();
    }
    for (const auto component : {&Vec3::x, &Vec3::y, &Vec3::z}) {
      const double before = stored_velocity(baseline_particle, component);
      const double after = stored_velocity(result->particles.front(), component);
      measured.particle_pressure_impulse.*component += particle.particle_mass * (after - before);
      measured.particle_prepressure_impulse.*component +=
          particle.particle_mass * (before - particle.particle_velocity.*component);
    }
    return result;
  };
  return backend;
}

void ExpectFloatingBoxPressureMeasurement(
    const OrientedFloatingBoxLoads3D& loads,
    const FloatingBoxPressureMeasurements& measured
) {
  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  ASSERT_TRUE(loads.particle_pressure_impulse.has_value());
  Vec3 mass_weight_difference{};
  Vec3 gather_weight_difference{};
  Vec3 reconstructed_gather{};
  for (const auto& face : measured.pressure_faces) {
    const auto normal = kComponents[static_cast<std::size_t>(face.axis)];
    mass_weight_difference.*normal += (face.deposited_mass - face.pressure_dual_mass) * face.velocity_change;
    gather_weight_difference.*normal -= face.deposited_mass * face.velocity_change;
    for (const auto component : kComponents) {
      gather_weight_difference.*component += face.gathered_impulse.*component;
      reconstructed_gather.*component += face.gathered_impulse.*component;
    }
  }
  for (const auto component : kComponents) {
    EXPECT_NEAR(loads.pressure_grid_impulse.*component, measured.grid_impulse.*component, 1e-12);
    EXPECT_NEAR(reconstructed_gather.*component, measured.particle_pressure_impulse.*component, 1e-12);
    double distributed_impulse = 0.0;
    for (const auto& particle_impulse : measured.compatible_particle_impulses) {
      distributed_impulse += particle_impulse.*component;
    }
    EXPECT_NEAR(distributed_impulse, measured.grid_impulse.*component, 1e-12);
    EXPECT_NEAR((*loads.particle_pressure_impulse).*component, distributed_impulse, 1e-12);
    EXPECT_NEAR(
        mass_weight_difference.*component + gather_weight_difference.*component,
        measured.particle_pressure_impulse.*component - measured.grid_impulse.*component,
        1e-12
    );
    EXPECT_NEAR(
        loads.liquid_gather_impulse.*component,
        distributed_impulse + measured.particle_prepressure_impulse.*component,
        1e-12
    );
  }
  std::cout << "pressure_face_weights mass_difference_n_s="
            << std::hypot(mass_weight_difference.x, mass_weight_difference.y, mass_weight_difference.z)
            << " gather_difference_n_s="
            << std::hypot(gather_weight_difference.x, gather_weight_difference.y, gather_weight_difference.z) << '\n';
}

double MatchFloatingBoxSeedMassToWetVolume(LiquidState3D& state, double density, double submerged_box_volume) {
  // The four seeds represent the four horizontal cells of the qualification
  // tank. Their wet widths are 7/8 m and their wet depth is 5/8 m. Only the
  // cell containing particle 1 is cut by the initial box.
  constexpr double kWetCellVolume = 0.875 * 0.875 * 0.625;
  for (auto& particle : state.particles.values()) {
    particle.reference_volume = kWetCellVolume;
    if (particle.id == 1) {
      particle.reference_volume -= submerged_box_volume;
    }
    particle.mass = density * particle.reference_volume;
  }
  return 4 * kWetCellVolume - submerged_box_volume;
}

void CheckOrientedFloatingBoxTrajectory(bool match_seed_mass) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
  auto initial = Initial(4);
  initial.particles[0].world_position = {0.375, 0.1875, 0.375};
  initial.particles[1].world_position = {1.375, 0.375, 0.375};
  initial.particles[2].world_position = {0.375, 0.375, 1.375};
  initial.particles[3].world_position = {1.375, 0.375, 1.375};
  // Upright box has a 3/16 m draft and a 1/4 m square waterplane.
  const double initial_wet_volume =
      match_seed_mass ? MatchFloatingBoxSeedMassToWetVolume(initial, scene.liquid_density, 0.1875 * 0.25 * 0.25) : 0.0;
  for (const double kDt : {0.015625, 0.0078125}) {
    SCOPED_TRACE(kDt);
    SCOPED_TRACE(match_seed_mass);
    const int frames = static_cast<int>(0.0625 / kDt);
    const auto options = Options();
    FloatingBoxPressureMeasurements measured{.energy_blend = options.particle_energy_blend};
    const auto backend = MeasureFloatingBoxPressure(measured);
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();

    constexpr Vec3 kHalfExtent{0.125, 0.125, 0.125};
    const RigidBody3D initial_body{
        .body_id = 23,
        .mass = 2.0,
        .principal_inertia = {1.0, 1.0, 1.0},
        .center = {0.5, 0.6875, 0.5},
        .linear_velocity = {0.0, 0.125, 0.0},
        .angular_velocity = {0.0, 0.0, 0.0625},
    };
    auto body = initial_body;
    std::vector<std::uint8_t> midpoint;
    std::vector<std::uint8_t> final;
    double max_pressure_traction_residual = 0.0;
    double max_particle_pressure_impulse = 0.0;
    double max_grid_to_particle_pressure_difference = 0.0;
    bool observed_pressure = false;
    bool observed_particle_motion = false;
    for (int frame = 1; frame <= frames; ++frame) {
      SCOPED_TRACE(frame);
      const auto initial_momentum = StoredLiquidMomentum(scheduler->committed().particles.values());
      ASSERT_TRUE(scheduler->BeginFrame(frame * kDt).ok());
      const auto attempt = scheduler->AttemptWithOrientedFloatingBox(kDt, body, kHalfExtent);
      ASSERT_TRUE(attempt.ok()) << attempt.status();
      ASSERT_TRUE(attempt->body.has_value()) << scheduler->diagnostics().failure_code << " stage "
                                             << static_cast<int>(scheduler->diagnostics().failed_stage);
      ASSERT_TRUE(attempt->oriented_loads.has_value());
      ASSERT_TRUE(attempt->oriented_loads->tank_pressure.has_value());
      if (frame == 1 && match_seed_mass) {
        EXPECT_NEAR(measured.wet_volume, initial_wet_volume, 1e-12);
      }
      ExpectFloatingBoxPressureMeasurement(*attempt->oriented_loads, measured);
      const auto particle_pressure = measured.particle_pressure_impulse;
      const auto grid_pressure = attempt->oriented_loads->pressure_grid_impulse;
      max_particle_pressure_impulse = std::max(
          max_particle_pressure_impulse,
          std::hypot(particle_pressure.x, particle_pressure.y, particle_pressure.z)
      );
      max_grid_to_particle_pressure_difference = std::max(
          max_grid_to_particle_pressure_difference,
          std::hypot(
              particle_pressure.x - grid_pressure.x,
              particle_pressure.y - grid_pressure.y,
              particle_pressure.z - grid_pressure.z
          )
      );
      max_pressure_traction_residual =
          std::max(max_pressure_traction_residual, PressureTractionImpulseResidual(*attempt->oriented_loads, kDt));
      ExpectStoredLiquidMomentumBudget(
          initial_momentum,
          scheduler->committed().particles.values(),
          *attempt->oriented_loads
      );
      EXPECT_NEAR(attempt->body->center.y, body.center.y + kDt * attempt->body->linear_velocity.y, 1e-14);
      EXPECT_LE(scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect, 2e-5);
      for (const auto& cell : scheduler->committed().cells.values()) {
        observed_pressure |= std::abs(cell.previous_pressure) > 1e-5;
      }
      const auto position = scheduler->committed().particles[0].world_position;
      observed_particle_motion |= std::abs(position.y - initial.particles[0].world_position.y) > 1e-9;
      const auto previous_pose = RigidBodyPose3D{body.center, body.orientation};
      body = *attempt->body;
      ASSERT_TRUE(scheduler->FinishFrame().ok());
      if (frame == frames / 2 || frame == frames) {
        const auto checkpoint = EncodeCoupledInteractionCheckpoint3D(
            scheduler->committed(),
            {.accepted_step = static_cast<std::uint64_t>(frame), .bodies = {{body, previous_pose}}}
        );
        ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
        (frame == frames / 2 ? midpoint : final) = *checkpoint;
      }
    }
    EXPECT_GT(body.orientation.vector.z, 0.0);
    std::cout << "match_seed_mass=" << match_seed_mass << " pressure_traction_dt_s=" << kDt
              << " max_residual_n_s=" << max_pressure_traction_residual << '\n';
    std::cout << "match_seed_mass=" << match_seed_mass << " pressure_gather_dt_s=" << kDt
              << " max_particle_pressure_impulse_n_s=" << max_particle_pressure_impulse
              << " max_grid_to_particle_difference_n_s=" << max_grid_to_particle_pressure_difference << '\n';
    EXPECT_LE(max_pressure_traction_residual, 1e-12);
    EXPECT_GT(max_particle_pressure_impulse, 1e-8);
    EXPECT_TRUE(observed_pressure);
    EXPECT_TRUE(observed_particle_motion);
    EXPECT_LT(body.linear_velocity.y, initial_body.linear_velocity.y);
    const auto restored = DecodeCoupledInteractionCheckpoint3D(midpoint);
    ASSERT_TRUE(restored.ok()) << restored.status();
    auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, scene, Options());
    ASSERT_TRUE(replay.ok()) << replay.status();
    auto history = restored->solid.bodies.front();
    for (int frame = frames / 2 + 1; frame <= frames; ++frame) {
      ASSERT_TRUE(replay->BeginFrame(frame * kDt).ok());
      const auto attempt = replay->AttemptWithOrientedFloatingBox(kDt, history.body, kHalfExtent);
      ASSERT_TRUE(attempt.ok()) << attempt.status();
      ASSERT_TRUE(attempt->body.has_value());
      ASSERT_TRUE(attempt->oriented_loads.has_value());
      ASSERT_TRUE(attempt->oriented_loads->tank_pressure.has_value());
      history = {*attempt->body, {history.body.center, history.body.orientation}};
      ASSERT_TRUE(replay->FinishFrame().ok());
    }
    const auto replayed = EncodeCoupledInteractionCheckpoint3D(
        replay->committed(),
        {.accepted_step = static_cast<std::uint64_t>(frames), .bodies = {history}}
    );
    ASSERT_TRUE(replayed.ok()) << replayed.status();
    EXPECT_EQ(*replayed, final);
  }
}
void CheckCoupledOrientedFloatingBoxTrajectory(bool match_seed_mass) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
  auto initial = Initial(4);
  initial.particles[0].world_position = {0.375, 0.1875, 0.375};
  initial.particles[1].world_position = {1.375, 0.375, 0.375};
  initial.particles[2].world_position = {0.375, 0.375, 1.375};
  initial.particles[3].world_position = {1.375, 0.375, 1.375};
  // The tilted box projects vertical edge components of 0.6 and 0.8.
  // Its exposed triangular prism is 0.05 m high and 0.25 m thick.
  constexpr double kExposedVolume = 0.25 * 0.05 * 0.05 / (2.0 * 0.6 * 0.8);
  const double initial_wet_volume = match_seed_mass
      ? MatchFloatingBoxSeedMassToWetVolume(initial, scene.liquid_density, 0.25 * 0.25 * 0.25 - kExposedVolume)
      : 0.0;
  for (const double kDt : {0.015625, 0.0078125}) {
    SCOPED_TRACE(kDt);
    SCOPED_TRACE(match_seed_mass);
    const int frames = static_cast<int>(0.0625 / kDt);
    const auto options = Options();
    FloatingBoxPressureMeasurements measured{.energy_blend = options.particle_energy_blend};
    const auto backend = MeasureFloatingBoxPressure(measured);
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();

    constexpr Vec3 kHalfExtent{0.125, 0.125, 0.125};
    const RigidBody3D initial_body{
        .body_id = 23,
        .mass = 0.00004,
        .principal_inertia = {6.25e-7, 6.25e-7, 6.25e-7},
        .center = {0.5, 0.625, 0.5},
        .orientation = {.w = std::sqrt(0.9), .vector = {0.0, 0.0, std::sqrt(0.1)}},
        .linear_velocity = {0.0, 0.125, 0.0},
    };
    auto body = initial_body;
    std::vector<std::uint8_t> midpoint;
    std::vector<std::uint8_t> final;
    double max_pressure_traction_residual = 0.0;
    double max_particle_pressure_impulse = 0.0;
    double max_grid_to_particle_pressure_difference = 0.0;
    bool observed_pressure = false;
    bool observed_particle_motion = false;
    for (int frame = 1; frame <= frames; ++frame) {
      SCOPED_TRACE(frame);
      const auto initial_momentum = StoredLiquidMomentum(scheduler->committed().particles.values());
      ASSERT_TRUE(scheduler->BeginFrame(frame * kDt).ok());
      const auto attempt = scheduler->AttemptWithCoupledOrientedFloatingBox(kDt, body, kHalfExtent);
      ASSERT_TRUE(attempt.ok()) << attempt.status();
      ASSERT_TRUE(attempt->body.has_value()) << scheduler->diagnostics().failure_code << " stage "
                                             << static_cast<int>(scheduler->diagnostics().failed_stage);
      ASSERT_TRUE(attempt->oriented_loads.has_value());
      ASSERT_TRUE(attempt->oriented_loads->tank_pressure.has_value());
      const auto& loads = *attempt->oriented_loads;

      const double momentum_change = body.mass * (attempt->body->linear_velocity.y - body.linear_velocity.y);
      const double angular_momentum_change =
          body.principal_inertia.z * (attempt->body->angular_velocity.z - body.angular_velocity.z);
      EXPECT_NEAR(
          momentum_change,
          kDt * (loads.external_force.y + loads.pressure_force.y) + loads.particle_impulse.y,
          3e-12
      );
      EXPECT_NEAR(
          angular_momentum_change,
          kDt * (loads.external_torque.z + loads.pressure_torque.z) + loads.particle_angular_impulse.z,
          5e-14
      );
      if (frame == 1 && match_seed_mass) {
        EXPECT_NEAR(measured.wet_volume, initial_wet_volume, 1e-12);
      }
      ExpectFloatingBoxPressureMeasurement(*attempt->oriented_loads, measured);
      const auto particle_pressure = measured.particle_pressure_impulse;
      const auto grid_pressure = attempt->oriented_loads->pressure_grid_impulse;
      max_particle_pressure_impulse = std::max(
          max_particle_pressure_impulse,
          std::hypot(particle_pressure.x, particle_pressure.y, particle_pressure.z)
      );
      max_grid_to_particle_pressure_difference = std::max(
          max_grid_to_particle_pressure_difference,
          std::hypot(
              particle_pressure.x - grid_pressure.x,
              particle_pressure.y - grid_pressure.y,
              particle_pressure.z - grid_pressure.z
          )
      );
      max_pressure_traction_residual =
          std::max(max_pressure_traction_residual, PressureTractionImpulseResidual(*attempt->oriented_loads, kDt));
      ExpectStoredLiquidMomentumBudget(initial_momentum, scheduler->committed().particles.values(), loads);
      EXPECT_NEAR(attempt->body->center.y, body.center.y + kDt * attempt->body->linear_velocity.y, 1e-14);
      EXPECT_LE(scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect, 2e-5);
      for (const auto& cell : scheduler->committed().cells.values()) {
        observed_pressure |= std::abs(cell.previous_pressure) > 1e-5;
      }
      std::vector<CellIndex3D> cells;
      for (const auto& cell : scheduler->committed().cells.values()) {
        cells.push_back(cell.cell_index);
      }
      auto swept_body = body;
      swept_body.linear_velocity = attempt->body->linear_velocity;
      swept_body.angular_velocity = attempt->body->angular_velocity;
      const auto patches =
          BuildOrientedBoxPressurePatches3D(swept_body, kHalfExtent, 0.75, scheduler->committed().geometry, cells);
      ASSERT_TRUE(patches.ok()) << patches.status();
      const auto flux = ComputeRigidBodyInwardWallFlux3D(swept_body, cells.size(), *patches);
      ASSERT_TRUE(flux.ok()) << flux.status();
      EXPECT_NEAR(measured.wall_flux, std::accumulate(flux->begin(), flux->end(), 0.0), 1e-15);
      const auto position = scheduler->committed().particles[0].world_position;
      observed_particle_motion |= std::abs(position.y - initial.particles[0].world_position.y) > 1e-9;
      const auto previous_pose = RigidBodyPose3D{body.center, body.orientation};
      body = *attempt->body;
      ASSERT_TRUE(scheduler->FinishFrame().ok());
      if (frame == frames / 2 || frame == frames) {
        const auto checkpoint = EncodeCoupledInteractionCheckpoint3D(
            scheduler->committed(),
            {.accepted_step = static_cast<std::uint64_t>(frame), .bodies = {{body, previous_pose}}}
        );
        ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
        (frame == frames / 2 ? midpoint : final) = *checkpoint;
      }
    }
    EXPECT_LT(body.angular_velocity.z, 0.0);
    std::cout << "match_seed_mass=" << match_seed_mass << " pressure_traction_dt_s=" << kDt
              << " max_residual_n_s=" << max_pressure_traction_residual << '\n';
    std::cout << "match_seed_mass=" << match_seed_mass << " pressure_gather_dt_s=" << kDt
              << " max_particle_pressure_impulse_n_s=" << max_particle_pressure_impulse
              << " max_grid_to_particle_difference_n_s=" << max_grid_to_particle_pressure_difference << '\n';
    EXPECT_LE(max_pressure_traction_residual, 1e-12);
    EXPECT_GT(max_particle_pressure_impulse, 1e-8);
    EXPECT_TRUE(observed_pressure);
    EXPECT_TRUE(observed_particle_motion);
    EXPECT_LT(std::abs(body.linear_velocity.y), 0.04);
    const auto restored = DecodeCoupledInteractionCheckpoint3D(midpoint);
    ASSERT_TRUE(restored.ok()) << restored.status();
    auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, scene, Options());
    ASSERT_TRUE(replay.ok()) << replay.status();
    auto history = restored->solid.bodies.front();
    for (int frame = frames / 2 + 1; frame <= frames; ++frame) {
      ASSERT_TRUE(replay->BeginFrame(frame * kDt).ok());
      const auto attempt = replay->AttemptWithCoupledOrientedFloatingBox(kDt, history.body, kHalfExtent);
      ASSERT_TRUE(attempt.ok()) << attempt.status();
      ASSERT_TRUE(attempt->body.has_value());
      ASSERT_TRUE(attempt->oriented_loads.has_value());
      ASSERT_TRUE(attempt->oriented_loads->tank_pressure.has_value());
      history = {*attempt->body, {history.body.center, history.body.orientation}};
      ASSERT_TRUE(replay->FinishFrame().ok());
    }
    const auto replayed = EncodeCoupledInteractionCheckpoint3D(
        replay->committed(),
        {.accepted_step = static_cast<std::uint64_t>(frames), .bodies = {history}}
    );
    ASSERT_TRUE(replayed.ok()) << replayed.status();
    EXPECT_EQ(*replayed, final);
  }
}
TEST(LiquidSubstep3D, OrientedFloatingBoxProductionGatherReplaysTrajectory) {
  CheckOrientedFloatingBoxTrajectory(false);
}
TEST(LiquidSubstep3D, OrientedFloatingBoxProductionGatherReplaysMatchedMassTrajectory) {
  CheckOrientedFloatingBoxTrajectory(true);
}
TEST(LiquidSubstep3D, CoupledOrientedFloatingBoxProductionGatherReplaysLightBody) {
  CheckCoupledOrientedFloatingBoxTrajectory(false);
}
TEST(LiquidSubstep3D, CoupledOrientedFloatingBoxProductionGatherReplaysMatchedMassLightBody) {
  CheckCoupledOrientedFloatingBoxTrajectory(true);
}

TEST(LiquidSubstep3D, GuidedFloatingBoxCoupledLightBodyUsesProductionParticles) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
  auto initial = Initial(4);
  initial.particles[0].world_position = {0.375, 0.1875, 0.375};
  initial.particles[1].world_position = {1.375, 0.375, 0.375};
  initial.particles[2].world_position = {0.375, 0.375, 1.375};
  initial.particles[3].world_position = {1.375, 0.375, 1.375};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double kDt = 0.015625;
  constexpr double kInitialSpeed = 0.125;
  constexpr Vec3 kHalfExtent{0.125, 0.3125, 0.125};
  RigidBody3D body{
      .body_id = 23,
      .mass = 0.00025,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.375, 0.5625, 0.375},
      .linear_velocity = {0.0, kInitialSpeed, 0.0},
  };
  double max_speed = 0.0;
  for (int frame = 1; frame <= 4; ++frame) {
    SCOPED_TRACE(frame);
    ASSERT_TRUE(scheduler->BeginFrame(frame * kDt).ok());
    const auto attempt = scheduler->AttemptWithCoupledGuidedFloatingBox(kDt, body, kHalfExtent);
    ASSERT_TRUE(attempt.ok()) << attempt.status();
    ASSERT_TRUE(attempt->body.has_value()) << scheduler->diagnostics().failure_code << " stage "
                                           << static_cast<int>(scheduler->diagnostics().failed_stage);
    ASSERT_TRUE(attempt->added_mass_kg.has_value());
    EXPECT_GT(*attempt->added_mass_kg, body.mass);
    EXPECT_LE(scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect, 2e-5);
    if (frame == 1) {
      EXPECT_NEAR(
          attempt->body->linear_velocity.y,
          kInitialSpeed * body.mass / (body.mass + *attempt->added_mass_kg),
          1e-8
      );
    }
    max_speed = std::max(max_speed, std::abs(attempt->body->linear_velocity.y));
    body = *attempt->body;
    ASSERT_TRUE(scheduler->FinishFrame().ok());
  }
  EXPECT_LT(max_speed, kInitialSpeed);
  EXPECT_NE(scheduler->committed().particles[0].world_position.y, initial.particles[0].world_position.y);
  std::cout << "floating_box_particle_light_mass_kg=" << body.mass << " max_speed_m_s=" << max_speed << '\n';
}
TEST(LiquidSubstep3D, GuidedFloatingBoxCommitsParticleReactionWithLiquid) {
  for (const bool coupled : {false, true}) {
    SCOPED_TRACE(coupled);
    auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
    scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
    auto initial = Initial();
    initial.particles[0].world_position = {0.375, 0.1875, 0.375};
    initial.particles[0].mass = 2.0;
    auto backend = CpuLiquidSubstepBackend3D();
    backend.gather = [](void*,
                        const LiquidBoundaryGatherViewInputs3D&,
                        LiquidBoundaryWorkspace3D&) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
      static constexpr std::array gathered{
          LiquidBoundaryParticleGather3D{.new_pic_velocity = {0.0, 0.75, 0.0}},
      };
      return LiquidBoundaryGatherView3D{.particles = gathered};
    };
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    constexpr double kDt = 0.125;
    constexpr Vec3 kHalfExtent{0.125, 0.3125, 0.125};
    const RigidBody3D body{
        .body_id = 23,
        .mass = 2.0,
        .principal_inertia = {1.0, 1.0, 1.0},
        .center = {0.375, 0.5625, 0.375},
    };
    const auto encode = [&](const LiquidState3D& liquid, const RigidBody3D& box_body) {
      const SolidInteractionCheckpoint3D solid{
          .accepted_step = liquid.clock.accepted_step_ordinal,
          .bodies = {{box_body, {body.center, body.orientation}}},
      };
      return EncodeCoupledInteractionCheckpoint3D(liquid, solid);
    };
    const auto before_impact = encode(scheduler->committed(), body);
    ASSERT_TRUE(before_impact.ok()) << before_impact.status();
    const auto attempt_box = [&](LiquidSubstepScheduler3D& target, double duration, RigidBody3D current) {
      return coupled ? target.AttemptWithCoupledGuidedFloatingBox(duration, current, kHalfExtent)
                     : target.AttemptWithGuidedFloatingBox(duration, current, kHalfExtent);
    };
    ASSERT_TRUE(scheduler->BeginFrame(2.0 * kDt).ok());
    scheduler->SetStageFaultHookForTesting({nullptr, [](void*, LiquidPhysicalStage3D stage, double) {
                                              return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("")
                                                                                             : absl::OkStatus();
                                            }});
    const auto rejected = attempt_box(*scheduler, 2.0 * kDt, body);
    ASSERT_TRUE(rejected.ok()) << rejected.status();
    EXPECT_EQ(rejected->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
    EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kAccept);
    EXPECT_FALSE(rejected->body.has_value());
    EXPECT_FALSE(rejected->added_mass_kg.has_value());
    EXPECT_EQ(scheduler->committed(), initial);
    const auto rejected_bytes = encode(scheduler->committed(), body);
    ASSERT_TRUE(rejected_bytes.ok()) << rejected_bytes.status();
    EXPECT_EQ(*rejected_bytes, *before_impact);

    scheduler->SetStageFaultHookForTesting({});
    const auto accepted = attempt_box(*scheduler, kDt, body);
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    ASSERT_TRUE(accepted->body.has_value()) << static_cast<int>(scheduler->diagnostics().failed_stage);
    EXPECT_EQ(accepted->added_mass_kg.has_value(), coupled);
    EXPECT_DOUBLE_EQ(accepted->body->linear_velocity.y, 0.75);
    EXPECT_DOUBLE_EQ(accepted->body->center.y, body.center.y);
    EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].world_velocity.y, 0.0);
    EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].world_position.y, 0.25);

    const auto restored = DecodeCoupledInteractionCheckpoint3D(*before_impact);
    ASSERT_TRUE(restored.ok()) << restored.status();
    auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, scene, Options(), backend);
    ASSERT_TRUE(replay.ok()) << replay.status();
    ASSERT_TRUE(replay->BeginFrame(2.0 * kDt).ok());
    const auto replayed = attempt_box(*replay, kDt, restored->solid.bodies.front().body);
    ASSERT_TRUE(replayed.ok()) << replayed.status();
    ASSERT_TRUE(replayed->body.has_value());
    const auto replay_bytes = encode(replay->committed(), *replayed->body);
    const auto accepted_bytes = encode(scheduler->committed(), *accepted->body);
    ASSERT_TRUE(replay_bytes.ok() && accepted_bytes.ok());
    EXPECT_EQ(*replay_bytes, *accepted_bytes);
  }
}

TEST(LiquidSubstep3D, OrientedFloatingBoxCommitsParticleReactionWithLiquid) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
  auto initial = Initial(2);
  initial.particles[0].world_position = {0.4375, 0.3125, 0.5};
  initial.particles[0].mass = 2.0;
  initial.particles[1].world_position = {1.8125, 0.5, 1.5};
  initial.particles[1].mass = 2.0;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.gather = [](void*,
                      const LiquidBoundaryGatherViewInputs3D& inputs,
                      LiquidBoundaryWorkspace3D&) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    static constexpr std::array gathered{
        LiquidBoundaryParticleGather3D{.new_pic_velocity = {0.0, 0.75, 0.0}},
        LiquidBoundaryParticleGather3D{.new_pic_velocity = {1.0, 0.0, 0.0}},
    };
    const bool near_tank = inputs.particles.front().transfer.trial_world_position.x > 1.0;
    return LiquidBoundaryGatherView3D{.particles = std::span(gathered).subspan(near_tank ? 1 : 0, 1)};
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double kDt = 0.125;
  constexpr Vec3 kHalfExtent{0.125, 0.125, 0.125};
  const RigidBody3D body{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {1.0, 1.0, 2.0},
      .center = {0.5, 0.5, 0.5},
  };
  const auto encode = [&](const LiquidState3D& liquid, const RigidBody3D& box_body) {
    const SolidInteractionCheckpoint3D solid{
        .accepted_step = liquid.clock.accepted_step_ordinal,
        .bodies = {{box_body, {body.center, body.orientation}}},
    };
    return EncodeCoupledInteractionCheckpoint3D(liquid, solid);
  };
  const auto before_impact = encode(scheduler->committed(), body);
  ASSERT_TRUE(before_impact.ok()) << before_impact.status();
  const auto attempt_box = [&](LiquidSubstepScheduler3D& target, double duration, RigidBody3D current) {
    return target.AttemptWithOrientedFloatingBox(duration, current, kHalfExtent);
  };
  ASSERT_TRUE(scheduler->BeginFrame(2.0 * kDt).ok());
  scheduler->SetStageFaultHookForTesting({nullptr, [](void*, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("")
                                                                                           : absl::OkStatus();
                                          }});
  const auto rejected = attempt_box(*scheduler, 2.0 * kDt, body);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kAccept);
  EXPECT_FALSE(rejected->body.has_value());
  EXPECT_FALSE(rejected->oriented_loads.has_value());
  EXPECT_FALSE(rejected->added_mass_kg.has_value());
  EXPECT_EQ(scheduler->committed(), initial);
  const auto rejected_bytes = encode(scheduler->committed(), body);
  ASSERT_TRUE(rejected_bytes.ok()) << rejected_bytes.status();
  EXPECT_EQ(*rejected_bytes, *before_impact);

  scheduler->SetStageFaultHookForTesting({});
  const auto accepted = attempt_box(*scheduler, kDt, body);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_TRUE(accepted->body.has_value()) << static_cast<int>(scheduler->diagnostics().failed_stage);
  ASSERT_TRUE(accepted->oriented_loads.has_value());
  const auto& loads = *accepted->oriented_loads;
  EXPECT_DOUBLE_EQ(loads.external_force.y, 0.0);
  EXPECT_DOUBLE_EQ(loads.pressure_force.y, 0.0);
  EXPECT_DOUBLE_EQ(loads.pressure_torque.z, 0.0);
  EXPECT_DOUBLE_EQ(loads.particle_impulse.y, 1.5);
  EXPECT_DOUBLE_EQ(loads.particle_angular_impulse.z, -0.09375);
  EXPECT_DOUBLE_EQ(loads.liquid_gather_impulse.y, 1.5);
  EXPECT_DOUBLE_EQ(loads.tank_particle_impulse.y, 0.0);
  EXPECT_DOUBLE_EQ(loads.liquid_gather_impulse.x, 2.0);
  EXPECT_DOUBLE_EQ(loads.tank_particle_impulse.x, -2.0);
  EXPECT_DOUBLE_EQ(loads.particle_impulse.x, 0.0);
  ExpectStoredLiquidMomentumBudget(
      StoredLiquidMomentum(initial.particles.values()),
      scheduler->committed().particles.values(),
      loads
  );
  EXPECT_DOUBLE_EQ(accepted->body->angular_velocity.z, -0.046875);
  EXPECT_DOUBLE_EQ(accepted->body->linear_velocity.y, 0.75);
  EXPECT_DOUBLE_EQ(accepted->body->center.y, body.center.y);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].world_velocity.y, 0.0);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].world_position.y, 0.375);

  const auto restored = DecodeCoupledInteractionCheckpoint3D(*before_impact);
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, scene, Options(), backend);
  ASSERT_TRUE(replay.ok()) << replay.status();
  ASSERT_TRUE(replay->BeginFrame(2.0 * kDt).ok());
  const auto replayed = attempt_box(*replay, kDt, restored->solid.bodies.front().body);
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  ASSERT_TRUE(replayed->body.has_value());
  ASSERT_TRUE(replayed->oriented_loads.has_value());
  EXPECT_DOUBLE_EQ(replayed->oriented_loads->liquid_gather_impulse.y, loads.liquid_gather_impulse.y);
  EXPECT_DOUBLE_EQ(replayed->oriented_loads->tank_particle_impulse.x, loads.tank_particle_impulse.x);
  const auto replay_bytes = encode(replay->committed(), *replayed->body);
  const auto accepted_bytes = encode(scheduler->committed(), *accepted->body);
  ASSERT_TRUE(replay_bytes.ok() && accepted_bytes.ok());
  EXPECT_EQ(*replay_bytes, *accepted_bytes);
}

TEST(LiquidSubstep3D, GuidedFloatingBoxExcludesTankImpulseAndRetriesDoubleImpact) {
  const TranslatingContactBox3D box{{0.5, 0.25, 0.25}, {0.75, 0.875, 0.5}, {}};
  std::array gathered{LiquidBoundaryParticleGather3D{.new_pic_velocity = {1.0, 0.0, 0.0}}};
  const std::array distances{-0.25};
  const LiquidParticleUpdateInputs3D inputs{
      .gathered = gathered,
      .liquid_distances = distances,
      .obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0},
      .cell_width = 1.0,
      .step_duration = 1.0,
  };
  auto state = Initial();
  state.particles[0].world_position = {1.75, 0.5, 1.5};
  const auto tank_only = UpdateCpuLiquidParticlesWithBoxReaction3D(inputs, box, state.particles.values());
  ASSERT_TRUE(tank_only.ok()) << tank_only.status();
  EXPECT_DOUBLE_EQ(*tank_only, 0.0);
  EXPECT_DOUBLE_EQ(state.particles[0].world_position.x, 1.875);
  EXPECT_DOUBLE_EQ(state.particles[0].world_velocity.x, 0.0);

  state.particles[0].world_position = {0.25, 0.5, 0.375};
  gathered[0].new_pic_velocity.x = 2.0;
  const auto ambiguous = UpdateCpuLiquidParticlesWithBoxReaction3D(inputs, box, state.particles.values());
  EXPECT_EQ(ambiguous.status().code(), absl::StatusCode::kAborted);
}
TEST(LiquidSubstep3D, OrientedFloatingBoxMeasuresCoupledTranslationAndRotationAddedMass) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 1.875, 0.125, 0.125, 1.875, 0.0};
  auto initial = Initial(4);
  initial.particles[0].world_position = {0.375, 0.1875, 0.375};
  initial.particles[1].world_position = {1.375, 0.375, 0.375};
  initial.particles[2].world_position = {0.375, 0.375, 1.375};
  initial.particles[3].world_position = {1.375, 0.375, 1.375};
  constexpr double kDt = 0.03125;
  constexpr double kSpeed = 0.125;
  constexpr double kWaterplaneCentroidX = 1.0 / 96.0;
  constexpr double kInertiaPerMass = 0.015625;
  constexpr Vec3 kExtent{0.125, 0.125, 0.125};
  const auto advance = [&](double mass, double speed, bool coupled = false, bool reject = false
                       ) -> std::optional<RigidBody3D> {
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
    EXPECT_TRUE(scheduler.ok()) << scheduler.status();
    if (!scheduler.ok() || !scheduler->BeginFrame(kDt).ok()) {
      return std::nullopt;
    }
    const RigidBody3D body{
        .body_id = 23,
        .mass = mass,
        .principal_inertia = {mass * kInertiaPerMass, mass * kInertiaPerMass, mass * kInertiaPerMass},
        .center = {0.5, 0.625, 0.5},
        .orientation = {.w = std::sqrt(0.9), .vector = {0.0, 0.0, std::sqrt(0.1)}},
        .linear_velocity = {0.0, speed, 0.0},
    };
    if (reject) {
      scheduler->SetStageFaultHookForTesting({nullptr, [](void*, LiquidPhysicalStage3D stage, double) {
                                                return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("")
                                                                                               : absl::OkStatus();
                                              }});
    }
    const auto step = coupled ? scheduler->AttemptWithCoupledOrientedFloatingBox(kDt, body, kExtent)
                              : scheduler->AttemptWithOrientedFloatingBox(kDt, body, kExtent);
    EXPECT_TRUE(step.ok()) << step.status();
    if (reject && step.ok()) {
      EXPECT_EQ(step->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
      EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kAccept);
      EXPECT_EQ(scheduler->committed(), initial);
      return step->body;
    }
    if (step.ok()) {
      EXPECT_TRUE(step->body.has_value()) << scheduler->diagnostics().failure_code << " stage "
                                          << static_cast<int>(scheduler->diagnostics().failed_stage);
    }
    return step.ok() ? step->body : std::nullopt;
  };
  const auto positive = advance(1.0, kSpeed);
  const auto negative = advance(1.0, -kSpeed);
  ASSERT_TRUE(positive && negative);
  const double vertical_added_mass = 1.0 - (positive->linear_velocity.y - negative->linear_velocity.y) / (2 * kSpeed);
  ASSERT_GT(vertical_added_mass, 0.0);
  const double angular_impulse_per_speed =
      kInertiaPerMass * (positive->angular_velocity.z - negative->angular_velocity.z) / (2 * kSpeed);
  EXPECT_NEAR(angular_impulse_per_speed, -vertical_added_mass * kWaterplaneCentroidX, 1e-10);
  const double mode_mass = vertical_added_mass * (1.0 + kWaterplaneCentroidX * kWaterplaneCentroidX / kInertiaPerMass);
  const auto ordinary = advance(2.0 * mode_mass, kSpeed);
  const auto light = advance(0.25 * mode_mass, kSpeed);
  ASSERT_TRUE(ordinary && light);
  const auto mode_speed = [&](const RigidBody3D& body) {
    return body.linear_velocity.y + kWaterplaneCentroidX * body.angular_velocity.z;
  };
  const double ordinary_gain = mode_speed(*ordinary) / kSpeed;
  const double light_gain = mode_speed(*light) / kSpeed;
  EXPECT_NEAR(ordinary_gain, 0.5, 1e-5);
  EXPECT_NEAR(light_gain, -3.0, 1e-5);
  const auto coupled_light = advance(0.25 * mode_mass, kSpeed, true);
  ASSERT_TRUE(coupled_light);
  EXPECT_NEAR(mode_speed(*coupled_light) / kSpeed, 0.2, 1e-5);
  EXPECT_LT(coupled_light->angular_velocity.z, 0.0);
  EXPECT_FALSE(advance(0.25 * mode_mass, kSpeed, true, true));
  std::cout << "oriented_box_added_mass_kg=" << vertical_added_mass << " ordinary_mode_gain=" << ordinary_gain
            << " light_mode_gain=" << light_gain << '\n';
}
TEST(LiquidSubstep3D, SparseOrientedFloatingBoxRoutesUnsupportedPressureImpulse) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 2.875, 0.125, 0.125, 2.875, 0.0};
  auto initial = Initial();
  initial.particles[0].world_position = {0.375, 0.25, 0.375};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double kDuration = 0.015625;
  const RigidBody3D body{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.5, 0.6875, 0.5},
      .linear_velocity = {0.0, 0.125, 0.0},
  };
  constexpr Vec3 kHalfExtent{0.125, 0.125, 0.125};
  const auto encode = [&](const LiquidState3D& liquid, const RigidBody3D& box) {
    return EncodeCoupledInteractionCheckpoint3D(
        liquid,
        {
            .accepted_step = liquid.clock.accepted_step_ordinal,
            .bodies = {{box, {body.center, body.orientation}}},
        }
    );
  };
  const auto seed = encode(scheduler->committed(), body);
  ASSERT_TRUE(seed.ok()) << seed.status();
  ASSERT_TRUE(scheduler->BeginFrame(kDuration).ok());
  scheduler->SetStageFaultHookForTesting({nullptr, [](void*, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("")
                                                                                           : absl::OkStatus();
                                          }});
  const auto rejected = scheduler->AttemptWithOrientedFloatingBox(kDuration, body, kHalfExtent);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_TRUE(rejected->outcome.retry_duration.has_value());
  const double retry_duration = *rejected->outcome.retry_duration;
  EXPECT_FALSE(rejected->body.has_value());
  EXPECT_FALSE(rejected->oriented_loads.has_value());
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kAccept);
  const auto after_rejection = encode(scheduler->committed(), body);
  ASSERT_TRUE(after_rejection.ok()) << after_rejection.status();
  EXPECT_EQ(*after_rejection, *seed);
  scheduler->SetStageFaultHookForTesting({});
  const auto attempt = scheduler->AttemptWithOrientedFloatingBox(retry_duration, body, kHalfExtent);
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  ASSERT_TRUE(attempt->body.has_value()) << scheduler->diagnostics().failure_code;
  ASSERT_TRUE(attempt->oriented_loads.has_value());
  EXPECT_GT(std::abs(attempt->oriented_loads->pressure_grid_impulse.y), 1e-8);
  const auto& loads = *attempt->oriented_loads;
  ASSERT_TRUE(loads.particle_pressure_impulse.has_value());
  EXPECT_GT(loads.routed_pressure_faces, 0u);
  for (const auto component : {&Vec3::x, &Vec3::y, &Vec3::z}) {
    EXPECT_NEAR((*loads.particle_pressure_impulse).*component, loads.pressure_grid_impulse.*component, 1e-12);
  }
  EXPECT_GT(
      std::hypot(
          loads.routed_pressure_angular_impulse_change.x,
          loads.routed_pressure_angular_impulse_change.y,
          loads.routed_pressure_angular_impulse_change.z
      ),
      1e-8
  );
  EXPECT_EQ(scheduler->committed().clock.accepted_step_ordinal, 1u);
  ExpectStoredLiquidMomentumBudget(
      StoredLiquidMomentum(initial.particles.values()),
      scheduler->committed().particles.values(),
      loads
  );
  const auto accepted = encode(scheduler->committed(), *attempt->body);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  const auto restored = DecodeCoupledInteractionCheckpoint3D(*seed);
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, scene, Options());
  ASSERT_TRUE(replay.ok()) << replay.status();
  ASSERT_TRUE(replay->BeginFrame(kDuration).ok());
  const auto repeated =
      replay->AttemptWithOrientedFloatingBox(retry_duration, restored->solid.bodies.front().body, kHalfExtent);
  ASSERT_TRUE(repeated.ok() && repeated->body && repeated->oriented_loads);
  EXPECT_EQ(repeated->oriented_loads->routed_pressure_faces, loads.routed_pressure_faces);
  for (const auto component : {&Vec3::x, &Vec3::y, &Vec3::z}) {
    EXPECT_DOUBLE_EQ(
        repeated->oriented_loads->routed_pressure_angular_impulse_change.*component,
        loads.routed_pressure_angular_impulse_change.*component
    );
  }
  const auto repeated_bytes = encode(replay->committed(), *repeated->body);
  ASSERT_TRUE(repeated_bytes.ok()) << repeated_bytes.status();
  EXPECT_EQ(*repeated_bytes, *accepted);
}

TEST(LiquidSubstep3D, WideCoupledOrientedFloatingBoxUsesIterativePressure) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 4.875, 0.125, 0.125, 4.875, 0.0};
  auto initial = Initial();
  initial.particles[0].world_position = {0.375, 0.25, 0.375};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double kDuration = 0.015625;
  const RigidBody3D body{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.5, 0.6875, 0.5},
      .linear_velocity = {0.0, 0.125, 0.0},
  };
  ASSERT_TRUE(scheduler->BeginFrame(kDuration).ok());
  const auto accepted = scheduler->AttemptWithCoupledOrientedFloatingBox(kDuration, body, {0.125, 0.125, 0.125});
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_TRUE(accepted->body && accepted->oriented_loads)
      << scheduler->diagnostics().failure_code << " stage " << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_EQ(scheduler->committed().cells.size(), 25u);
  EXPECT_GT(scheduler->diagnostics().pressure_diagnostics.iteration_count, 0u);
  EXPECT_LE(scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect, 2e-5);
  EXPECT_LT(accepted->body->linear_velocity.y, body.linear_velocity.y);
  const auto& loads = *accepted->oriented_loads;
  ASSERT_TRUE(loads.particle_pressure_impulse.has_value());
  for (const auto component : {&Vec3::x, &Vec3::y, &Vec3::z}) {
    EXPECT_NEAR((*loads.particle_pressure_impulse).*component, loads.pressure_grid_impulse.*component, 1e-12);
  }
}

TEST(LiquidSubstep3D, OrientedFloatingBoxRejectsPressureWithNoReachableParticleSupport) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 2.875, 0.125, 0.125, 2.875, 0.0};
  auto initial = Initial();
  // The particle is above every wet pressure stencil, so no receiver exists.
  initial.particles[0].world_position = {0.375, 4.5, 0.375};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double kDuration = 0.015625;
  const RigidBody3D body{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.5, 0.6875, 0.5},
      .linear_velocity = {0.0, 0.125, 0.0},
  };
  const auto encode = [&] {
    const auto& liquid = scheduler->committed();
    return EncodeCoupledInteractionCheckpoint3D(
        liquid,
        {
            .accepted_step = liquid.clock.accepted_step_ordinal,
            .bodies = {{body, {body.center, body.orientation}}},
        }
    );
  };
  const auto seed = encode();
  ASSERT_TRUE(seed.ok()) << seed.status();
  ASSERT_TRUE(scheduler->BeginFrame(kDuration).ok());
  const auto attempt = scheduler->AttemptWithOrientedFloatingBox(kDuration, body, {0.125, 0.125, 0.125});
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  EXPECT_EQ(attempt->outcome.disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kFailedPrecondition);
  EXPECT_FALSE(attempt->body.has_value());
  EXPECT_FALSE(attempt->oriented_loads.has_value());
  const auto after = encode();
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(*after, *seed);
}

TEST(LiquidSubstep3D, OrientedFloatingBoxCommitsAngularPressureAndReplays) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 0.875, 0.125, 0.125, 0.875, 0.0};
  auto scheduler = LiquidSubstepScheduler3D::Create({}, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double kDt = 0.0625;
  constexpr Vec3 kHalfExtent{0.125, 0.125, 0.125};
  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.5, 0.625, 0.5},
      .orientation = {.w = std::sqrt(0.9), .vector = {0.0, 0.0, std::sqrt(0.1)}},
      .linear_velocity = {0.0, 0.25, 0.0},
      .angular_velocity = {0.0, 0.0, 0.125},
  };
  ASSERT_TRUE(scheduler->BeginFrame(kDt).ok());
  const auto first = scheduler->AttemptWithOrientedFloatingBoxPressure(kDt, body, kHalfExtent);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(first->body.has_value()) << static_cast<int>(scheduler->diagnostics().failed_stage);
  const auto& state = scheduler->committed();
  ASSERT_EQ(state.cells.size(), 1);
  const std::array pressure_cells{state.cells[0].cell_index};
  const std::array pressure{state.cells[0].previous_pressure};
  const auto patches = BuildOrientedBoxPressurePatches3D(body, kHalfExtent, 0.75, state.geometry, pressure_cells);
  ASSERT_TRUE(patches.ok()) << patches.status();
  const auto loaded = AccumulateRigidBodyPressure3D(body, pressure, scene.ambient_pressure, *patches);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_GT(std::abs(loaded->torque.z), 1e-8);
  // The tilted waterplane spans x = [-1/24, 1/16] relative to the center
  // and has width 1/4 in z. Its area and first moment give the net load.
  EXPECT_NEAR(loaded->force.y, pressure[0] * (5.0 / 192.0), 1e-12);
  EXPECT_NEAR(loaded->torque.z, pressure[0] * (5.0 / 18432.0), 1e-12);
  const auto expected = AdvanceRigidBody3D(*loaded, kDt);
  ASSERT_TRUE(expected.ok()) << expected.status();
  EXPECT_DOUBLE_EQ(first->body->angular_velocity.z, expected->end.angular_velocity.z);
  EXPECT_DOUBLE_EQ(first->body->orientation.vector.z, expected->end.orientation.vector.z);
  EXPECT_NE(first->body->orientation.vector.z, body.orientation.vector.z);
  ASSERT_TRUE(scheduler->FinishFrame().ok());

  const auto encode = [](const LiquidState3D& liquid, const RigidBody3D& current, const RigidBody3D& previous) {
    const SolidInteractionCheckpoint3D solid{
        .accepted_step = liquid.clock.accepted_step_ordinal,
        .bodies = {{current, {previous.center, previous.orientation}}},
    };
    return EncodeCoupledInteractionCheckpoint3D(liquid, solid);
  };
  const auto checkpoint = encode(scheduler->committed(), *first->body, body);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  auto restored = DecodeCoupledInteractionCheckpoint3D(*checkpoint);
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, scene, Options());
  ASSERT_TRUE(replay.ok()) << replay.status();

  ASSERT_TRUE(scheduler->BeginFrame(2.0 * kDt).ok());
  auto excessive_sweep = *first->body;
  excessive_sweep.torque.z = 2048.0;
  const auto sweep_retry = scheduler->AttemptWithOrientedFloatingBoxPressure(kDt, excessive_sweep, kHalfExtent);
  ASSERT_TRUE(sweep_retry.ok()) << sweep_retry.status();
  EXPECT_EQ(sweep_retry->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_FALSE(sweep_retry->body.has_value());
  scheduler->SetStageFaultHookForTesting({nullptr, [](void*, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("")
                                                                                           : absl::OkStatus();
                                          }});
  const auto rejected = scheduler->AttemptWithOrientedFloatingBoxPressure(kDt / 2.0, *first->body, kHalfExtent);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_FALSE(rejected->body.has_value());
  const auto after_rejection = encode(scheduler->committed(), *first->body, body);
  ASSERT_TRUE(after_rejection.ok()) << after_rejection.status();
  EXPECT_EQ(*after_rejection, *checkpoint);
  scheduler->SetStageFaultHookForTesting({});
  const auto retry = scheduler->AttemptWithOrientedFloatingBoxPressure(kDt / 4.0, *first->body, kHalfExtent);
  ASSERT_TRUE(retry.ok()) << retry.status();
  ASSERT_TRUE(retry->body.has_value());

  ASSERT_TRUE(replay->BeginFrame(2.0 * kDt).ok());
  const auto replayed =
      replay->AttemptWithOrientedFloatingBoxPressure(kDt / 4.0, restored->solid.bodies.front().body, kHalfExtent);
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  ASSERT_TRUE(replayed->body.has_value());
  const auto resumed_bytes = encode(replay->committed(), *replayed->body, *first->body);
  const auto direct_bytes = encode(scheduler->committed(), *retry->body, *first->body);
  ASSERT_TRUE(resumed_bytes.ok() && direct_bytes.ok());
  EXPECT_EQ(*resumed_bytes, *direct_bytes);
}
TEST(LiquidSubstep3D, GuidedFloatingBoxCommitsPressureAndDriftTogether) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 0.875, 0.125, 0.125, 0.875, 0.0};
  auto scheduler = LiquidSubstepScheduler3D::Create({}, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  constexpr double kDt = 0.125;
  constexpr Vec3 kHalfExtent{0.125, 0.3125, 0.125};
  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.375, 0.5625, 0.375},
      .linear_velocity = {0.0, 0.5, 0.0},
  };
  ASSERT_TRUE(scheduler->BeginFrame(kDt).ok());
  const auto first = scheduler->AttemptWithGuidedFloatingBox(kDt, body, kHalfExtent);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(first->body.has_value()) << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_DOUBLE_EQ(first->body->center.y, 0.625);
  EXPECT_NEAR(first->body->linear_velocity.y, 0.5 - kDt / 144.0, 1e-9);
  ASSERT_EQ(scheduler->committed().cells.size(), 1);
  EXPECT_NEAR(scheduler->committed().cells[0].previous_pressure, -1.0 / 9.0, 1e-7);
  ASSERT_TRUE(scheduler->FinishFrame().ok());

  const auto encode = [](const LiquidState3D& state, const RigidBody3D& current, const RigidBody3D& previous) {
    const SolidInteractionCheckpoint3D solid{
        .accepted_step = state.clock.accepted_step_ordinal,
        .bodies = {{current, {previous.center, previous.orientation}}},
    };
    return EncodeCoupledInteractionCheckpoint3D(state, solid);
  };
  const auto checkpoint = encode(scheduler->committed(), *first->body, body);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  auto restored = DecodeCoupledInteractionCheckpoint3D(*checkpoint);
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, scene, Options());
  ASSERT_TRUE(replay.ok()) << replay.status();

  ASSERT_TRUE(scheduler->BeginFrame(2.0 * kDt).ok());
  const auto old_clock = scheduler->committed().clock;
  scheduler->SetStageFaultHookForTesting({nullptr, [](void*, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("")
                                                                                           : absl::OkStatus();
                                          }});
  const auto rejected = scheduler->AttemptWithGuidedFloatingBox(kDt, *first->body, kHalfExtent);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_FALSE(rejected->body.has_value());
  EXPECT_EQ(scheduler->committed().clock, old_clock);
  const auto after_rejection = encode(scheduler->committed(), *first->body, body);
  ASSERT_TRUE(after_rejection.ok()) << after_rejection.status();
  EXPECT_EQ(*after_rejection, *checkpoint);
  scheduler->SetStageFaultHookForTesting({});
  const auto retry = scheduler->AttemptWithGuidedFloatingBox(kDt / 2.0, *first->body, kHalfExtent);
  ASSERT_TRUE(retry.ok()) << retry.status();
  ASSERT_TRUE(retry->body.has_value());
  EXPECT_NEAR(retry->body->center.y, 0.625 + first->body->linear_velocity.y * kDt / 2.0, 1e-12);
  EXPECT_NEAR(scheduler->diagnostics().accepted_measurement->step_start_pressure_cell_volume_m3, 0.32421875, 1e-12);

  ASSERT_TRUE(replay->BeginFrame(2.0 * kDt).ok());
  const auto replayed =
      replay->AttemptWithGuidedFloatingBox(kDt / 2.0, restored->solid.bodies.front().body, kHalfExtent);
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  ASSERT_TRUE(replayed->body.has_value());
  const auto resumed_bytes = encode(replay->committed(), *replayed->body, *first->body);
  const auto direct_bytes = encode(scheduler->committed(), *retry->body, *first->body);
  ASSERT_TRUE(resumed_bytes.ok() && direct_bytes.ok());
  EXPECT_EQ(*resumed_bytes, *direct_bytes);
}
TEST(LiquidSubstep3D, DynamicPlanarPistonFeedsAcceptedMotionIntoNextSolve) {
  namespace wet = wetted_piston_pressure_test;
  auto scheduler = LiquidSubstepScheduler3D::Create({}, wet::PlaneScene(0.75, wet::kPistonSpeed), Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(wet::kDuration).ok());
  const RigidBody3D piston{
      .body_id = 17,
      .mass = 1000.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .linear_velocity = {wet::kPistonSpeed, 0.0, 0.0},
  };
  const auto first = scheduler->AttemptWithDynamicPlanarPiston(wet::kDuration, piston);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(first->loaded_piston.has_value());
  const auto next = *first->loaded_piston;
  EXPECT_DOUBLE_EQ(next.center.x, wet::kDuration * wet::kPistonSpeed);
  EXPECT_LT(next.linear_velocity.x, wet::kPistonSpeed);
  EXPECT_GT(next.linear_velocity.x, 0.0);
  const auto cells = scheduler->committed().cells.values();
  ASSERT_EQ(cells.size(), 1);
  EXPECT_NEAR(
      next.linear_velocity.x,
      wet::kPistonSpeed - wet::kDuration * 0.75 * cells.front().previous_pressure / piston.mass,
      1e-12
  );

  ASSERT_TRUE(scheduler->FinishFrame().ok());
  SolidInteractionCheckpoint3D solid{
      .accepted_step = scheduler->committed().clock.accepted_step_ordinal,
      .bodies = {{next, {piston.center, piston.orientation}}},
      .sides = {},
  };
  const auto checkpoint = EncodeCoupledInteractionCheckpoint3D(scheduler->committed(), solid);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  ++solid.accepted_step;
  EXPECT_EQ(
      EncodeCoupledInteractionCheckpoint3D(scheduler->committed(), solid).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(DecodeSolidInteractionCheckpoint3D(*checkpoint).status().code(), absl::StatusCode::kInvalidArgument);
  const auto solid_only = EncodeSolidInteractionCheckpoint3D(solid);
  ASSERT_TRUE(solid_only.ok()) << solid_only.status();
  EXPECT_EQ(DecodeCoupledInteractionCheckpoint3D(*solid_only).status().code(), absl::StatusCode::kInvalidArgument);
  const auto restored = DecodeCoupledInteractionCheckpoint3D(*checkpoint);
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_EQ(restored->solid.bodies.size(), 1);
  auto replay = LiquidSubstepScheduler3D::Create(restored->liquid, wet::PlaneScene(0.75, wet::kPistonSpeed), Options());
  ASSERT_TRUE(replay.ok()) << replay.status();
  ASSERT_TRUE(scheduler->BeginFrame(2.0 * wet::kDuration).ok());
  ASSERT_TRUE(replay->BeginFrame(2.0 * wet::kDuration).ok());

  scheduler->SetStageFaultHookForTesting({
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("") : absl::OkStatus();
          },
  });
  const auto rejected = scheduler->AttemptWithDynamicPlanarPiston(wet::kDuration, next);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_FALSE(rejected->loaded_piston.has_value());
  EXPECT_DOUBLE_EQ(scheduler->committed().clock.time, wet::kDuration);
  scheduler->SetStageFaultHookForTesting({});
  const auto second = scheduler->AttemptWithDynamicPlanarPiston(wet::kDuration / 2.0, next);
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(second->loaded_piston.has_value());
  const auto& measured = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(measured && measured->piston_pose);
  EXPECT_DOUBLE_EQ(measured->piston_pose->step_start_low_x_m, next.center.x);
  EXPECT_NEAR(second->loaded_piston->center.x, next.center.x + next.linear_velocity.x * wet::kDuration / 2.0, 1e-15);
  const auto replayed =
      replay->AttemptWithDynamicPlanarPiston(wet::kDuration / 2.0, restored->solid.bodies.front().body);
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  ASSERT_TRUE(replayed->loaded_piston.has_value());
  solid.bodies.front() = {*second->loaded_piston, {next.center, next.orientation}};
  auto replay_solid = solid;
  replay_solid.bodies.front().body = *replayed->loaded_piston;
  const auto uninterrupted_bytes = EncodeCoupledInteractionCheckpoint3D(scheduler->committed(), solid);
  const auto replay_bytes = EncodeCoupledInteractionCheckpoint3D(replay->committed(), replay_solid);
  ASSERT_TRUE(uninterrupted_bytes.ok()) << uninterrupted_bytes.status();
  ASSERT_TRUE(replay_bytes.ok()) << replay_bytes.status();
  EXPECT_EQ(*uninterrupted_bytes, *replay_bytes);
}

TEST(LiquidSubstep3D, GuidedPistonMeasuresAddedMassAndLightBodyAmplification) {
  namespace wet = wetted_piston_pressure_test;
  constexpr double kHeight = 0.75;
  constexpr double kWettedArea = kHeight;
  constexpr double kPressureDistance = kHeight - 0.5;
  constexpr double kExpectedAddedMass = kWettedArea * kWettedArea * kPressureDistance;
  const auto body = [](double mass, double speed) {
    return RigidBody3D{
        .body_id = 17,
        .mass = mass,
        .principal_inertia = {1.0, 1.0, 1.0},
        .linear_velocity = {speed, 0.0, 0.0},
    };
  };

  std::array<double, 2> pressure_forces{};
  const std::array probe_speeds{-wet::kPistonSpeed, wet::kPistonSpeed};
  for (std::size_t probe = 0; probe < probe_speeds.size(); ++probe) {
    const double speed = probe_speeds[probe];
    auto scheduler = LiquidSubstepScheduler3D::Create({}, wet::PlaneScene(kHeight, speed), Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(wet::kDuration).ok());
    const auto attempt = scheduler->AttemptWithPlanarPistonPressureLoad(wet::kDuration, body(1.0, speed));
    ASSERT_TRUE(attempt.ok()) << attempt.status();
    ASSERT_TRUE(attempt->loaded_piston.has_value()) << scheduler->diagnostics().failure_code;
    pressure_forces[probe] = attempt->loaded_piston->force.x;
  }

  const double measured_added_mass =
      -wet::kDuration * (pressure_forces[1] - pressure_forces[0]) / (probe_speeds[1] - probe_speeds[0]);
  EXPECT_NEAR(measured_added_mass, kExpectedAddedMass, 1e-10);
  std::cout << "CTEST_FULL_OUTPUT\n"
            << "guided_piston_added_mass_kg=" << measured_added_mass << " analytic_added_mass_kg=" << kExpectedAddedMass
            << '\n';

  for (const double mass : {2.0 * kExpectedAddedMass, 0.25 * kExpectedAddedMass}) {
    SCOPED_TRACE(mass);
    auto scheduler = LiquidSubstepScheduler3D::Create({}, wet::PlaneScene(kHeight, wet::kPistonSpeed), Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(wet::kDuration).ok());
    const auto attempt = scheduler->AttemptWithDynamicPlanarPiston(wet::kDuration, body(mass, wet::kPistonSpeed));
    ASSERT_TRUE(attempt.ok()) << attempt.status();
    ASSERT_TRUE(attempt->loaded_piston.has_value()) << scheduler->diagnostics().failure_code;
    const double amplification = attempt->loaded_piston->linear_velocity.x / wet::kPistonSpeed;
    EXPECT_NEAR(amplification, 1.0 - measured_added_mass / mass, 1e-10);
    if (mass > measured_added_mass) {
      EXPECT_LT(std::abs(amplification), 1.0);
    } else {
      EXPECT_GT(std::abs(amplification), 1.0);
    }
    std::cout << "guided_piston_body_mass_kg=" << mass << " signed_step_gain=" << amplification << '\n';
  }
  const double light_mass = 0.25 * kExpectedAddedMass;
  auto coupled = LiquidSubstepScheduler3D::Create({}, wet::PlaneScene(kHeight, wet::kPistonSpeed), Options());
  ASSERT_TRUE(coupled.ok()) << coupled.status();
  ASSERT_TRUE(coupled->BeginFrame(wet::kDuration).ok());
  const auto attempt = coupled->AttemptWithCoupledPlanarPiston(wet::kDuration, body(light_mass, wet::kPistonSpeed));
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  ASSERT_TRUE(attempt->loaded_piston.has_value()) << coupled->diagnostics().failure_code;
  const double expected_velocity = wet::kPistonSpeed * light_mass / (light_mass + kExpectedAddedMass);
  EXPECT_NEAR(attempt->loaded_piston->linear_velocity.x, expected_velocity, 1e-12);
  EXPECT_NEAR(attempt->loaded_piston->center.x, wet::kDuration * expected_velocity, 1e-12);
  const auto& measurement = coupled->diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement.has_value());
  EXPECT_NEAR(measurement->step_start_inward_piston_flux_m3_per_s, kWettedArea * expected_velocity, 1e-12);

  auto rejected = LiquidSubstepScheduler3D::Create({}, wet::PlaneScene(kHeight, wet::kPistonSpeed), Options());
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_TRUE(rejected->BeginFrame(wet::kDuration).ok());
  rejected->SetStageFaultHookForTesting({
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("") : absl::OkStatus();
          },
  });
  const auto refused = rejected->AttemptWithCoupledPlanarPiston(wet::kDuration, body(light_mass, wet::kPistonSpeed));
  ASSERT_TRUE(refused.ok()) << refused.status();
  EXPECT_EQ(refused->outcome.disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_FALSE(refused->loaded_piston.has_value());
  EXPECT_DOUBLE_EQ(rejected->committed().clock.time, 0.0);

  auto wide_scene = wet::PlaneScene(kHeight, wet::kPistonSpeed);
  auto& wide_tank = std::get<AxisAlignedLiquidTank3D>(wide_scene.obstacle);
  wide_tank.low_z = -1.0;
  wide_tank.high_z = 2.0;
  auto wide = LiquidSubstepScheduler3D::Create({}, wide_scene, Options());
  ASSERT_TRUE(wide.ok()) << wide.status();
  ASSERT_TRUE(wide->BeginFrame(wet::kDuration).ok());
  const auto wide_step =
      wide->AttemptWithCoupledPlanarPiston(wet::kDuration, body(3.0 * light_mass, wet::kPistonSpeed));
  ASSERT_TRUE(wide_step.ok()) << wide_step.status();
  ASSERT_TRUE(wide_step->loaded_piston.has_value()) << wide->diagnostics().failure_code;
  EXPECT_EQ(wide->committed().cells.size(), 3);
  EXPECT_NEAR(wide_step->loaded_piston->linear_velocity.x, expected_velocity, 1e-8);
  const auto& wide_measurement = wide->diagnostics().accepted_measurement;
  ASSERT_TRUE(wide_measurement.has_value());
  EXPECT_NEAR(
      wide_measurement->step_start_inward_piston_flux_m3_per_s,
      3.0 * kWettedArea * wide_step->loaded_piston->linear_velocity.x,
      1e-12
  );
}

TEST(LiquidSubstep3D, UnresolvedAnalyticSphereWithNoAirLinksCanAdvance) {
  const LiquidState3D initial;
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{0.25, 0.25, 0.25}, std::numeric_limits<double>::denorm_min()};
  scene.surface_tension = 1;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(0x1p-7);
  ASSERT_TRUE(advanced.ok()) << advanced;
  EXPECT_TRUE(scheduler->committed().cells.values().empty());
  ASSERT_TRUE(scheduler->diagnostics().accepted_measurement.has_value());
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement->step_start_capillary_bound_seconds.has_value());
}

TEST(LiquidSubstep3D, MovingPistonGatherGroupsByPressureSupportAndPreservesCarrier) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wet = wetted_piston_pressure_test;
  const auto state = wet::PressureSupportGatherState();
  ASSERT_TRUE(state.ok()) << state.status();
  const auto& initial = *state;
  struct Audit {
    std::vector<bool> incident_faces;
    std::vector<std::size_t> run_sizes;
    std::vector<std::vector<LiquidFaceValidity3D>> new_validities;
    std::vector<std::vector<double>> new_velocities;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.freeze =
      [](void* context, const LiquidBoundaryGeometryViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
        const auto frozen = CpuLiquidSubstepBackend3D().freeze(nullptr, inputs, workspace);
        if (frozen.ok()) {
          auto& incident = static_cast<Audit*>(context)->incident_faces;
          incident.assign(frozen->mac_grid.face_count(), false);
          for (const auto& face : frozen->faces) {
            incident[face.face_slot] = true;
          }
        }
        return frozen;
      };
  backend.gather_run =
      [](void* context, const LiquidBoundaryGatherViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace) {
        auto& observed = *static_cast<Audit*>(context);
        observed.run_sizes.push_back(input.particles.size());
        observed.new_validities.emplace_back(input.new_history.validity.begin(), input.new_history.validity.end());
        observed.new_velocities.emplace_back(
            input.new_history.face_velocities.begin(),
            input.new_history.face_velocities.end()
        );
        return GatherLiquidBoundaryParticles3D(input, workspace);
      };
  auto options = Options();
  options.capacities.particles = initial.particles.size();
  const auto scene = wet::PressureSupportGatherScene();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0x1p-7);
  ASSERT_TRUE(status.ok()) << status << static_cast<int>(scheduler->diagnostics().failed_stage);
  ASSERT_EQ(audit.run_sizes, (std::vector<std::size_t>{2, 2}));
  const auto& supported_validities = audit.new_validities[0];
  const auto& unsupported_validities = audit.new_validities[1];
  ASSERT_EQ(supported_validities.size(), scheduler->committed().faces.size());
  ASSERT_EQ(unsupported_validities.size(), scheduler->committed().faces.size());
  ASSERT_EQ(audit.new_velocities[0].size(), scheduler->committed().faces.size());
  ASSERT_EQ(audit.new_velocities[1].size(), scheduler->committed().faces.size());
  std::size_t masked_donors = 0;
  for (const auto& face : scheduler->committed().faces.values()) {
    const auto slot = face.face_slot;
    const bool usable =
        face.validity == LiquidFaceValidity3D::kValid || face.validity == LiquidFaceValidity3D::kPredicted;
    const bool masked = !audit.incident_faces[slot] && face.structurally_eligible && usable;
    EXPECT_EQ(supported_validities[slot], masked ? LiquidFaceValidity3D::kMissing : face.validity);
    EXPECT_EQ(unsupported_validities[slot], face.validity);
    EXPECT_DOUBLE_EQ(audit.new_velocities[0][slot], face.carrier_velocity);
    EXPECT_DOUBLE_EQ(audit.new_velocities[1][slot], face.carrier_velocity);
    masked_donors += masked;
  }
  EXPECT_GT(masked_donors, 0);
  auto scalar = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scalar.ok()) << scalar.status();
  const auto scalar_status = scalar->AdvanceFrame(0x1p-7);
  ASSERT_TRUE(scalar_status.ok()) << scalar_status;
  ExpectIdenticalSerializedState(scalar->committed(), scheduler->committed());
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

absl::Status
UpdateCpuParticles(const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
  return CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
}

TEST(LiquidSubstep3D, RegularCellTransportPairsAdvectionWithVolume) {
  test_internal::CheckRegularCellTransport(UpdateCpuParticles);
}

TEST(LiquidSubstep3D, RegularCellTransportUsesSweptContact) {
  test_internal::CheckRegularCellTransportContact(UpdateCpuParticles);
}

TEST(LiquidSubstep3D, RejectsMalformedRegularCellTransportBeforeMutation) {
  test_internal::CheckMalformedRegularCellTransport(UpdateCpuParticles);
}

TEST(LiquidSubstep3D, RejectsRegularCellTransportAliasingTrialParticles) {
  test_internal::CheckAliasedRegularCellTransport(UpdateCpuParticles);
}
TEST(LiquidSubstep3D, RegularCellTransportPreservesParticleErrorOrder) {
  test_internal::CheckRegularCellTransportErrorOrder(UpdateCpuParticles);
}

TEST(LiquidSubstep3D, RegularCellTransportRejectsInvalidParticleInputsBeforeRetry) {
  test_internal::CheckRegularCellTransportRejectsInvalidParticleInputs(UpdateCpuParticles);
}

struct RegularTransportSchedulerAudit {
  std::vector<FrozenLiquidCell3D> cells;
  std::vector<FrozenLiquidFace3D> faces;
  std::vector<LiquidRegularCellTransport3D> transport;
  std::size_t update_count = 0;
};

LiquidSubstepBackend3D ObserveRegularTransport(RegularTransportSchedulerAudit& audit) {
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.freeze =
      [](void* context, const LiquidBoundaryGeometryViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
        auto& observed = *static_cast<RegularTransportSchedulerAudit*>(context);
        const auto frozen = CpuLiquidSubstepBackend3D().freeze(nullptr, inputs, workspace);
        if (frozen.ok()) {
          observed.cells.assign(frozen->cells.begin(), frozen->cells.end());
          observed.faces.assign(frozen->faces.begin(), frozen->faces.end());
        }
        return frozen;
      };
  backend.update_particles =
      [](void* context, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
        auto& observed = *static_cast<RegularTransportSchedulerAudit*>(context);
        ++observed.update_count;
        observed.transport.assign(inputs.regular_cell_transport.begin(), inputs.regular_cell_transport.end());
        return CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
      };
  return backend;
}

constexpr double kRegularTransportDuration = 0x1p-7;
constexpr Vec3 kRegularTransportStart{-1.375, 0.375, 2.625};

LiquidState3D RegularTransportInitial() {
  auto initial = Initial();
  initial.particles[0].world_position = kRegularTransportStart;
  initial.particles[0].world_velocity = {0.5, -0.25, 0.125};
  return initial;
}

LiquidSceneInputs3D RegularTransportScene() {
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{0.7};
  scene.obstacle = AxisAlignedLiquidTank3D{
      .start_low_x = -4,
      .high_x = 12,
      .floor_y = 0,
      .low_z = 0,
      .high_z = 5,
      .piston_velocity_x = 0,
  };
  return scene;
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportUsesProjectedFacesAndStationaryFloor) {
  const auto initial = RegularTransportInitial();
  const auto scene = RegularTransportScene();
  RegularTransportSchedulerAudit audit;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), ObserveRegularTransport(audit));
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(advanced.ok()) << advanced;
  ASSERT_EQ(audit.update_count, 1);
  ASSERT_EQ(audit.cells.size(), 1);
  const auto cell = audit.cells.front().cell_index;
  EXPECT_EQ(cell.x, -2);
  EXPECT_EQ(cell.y, 0);
  EXPECT_EQ(cell.z, 2);
  ASSERT_EQ(audit.faces.size(), 6);
  std::array<double, 3> low{};
  std::array<double, 3> high{};
  std::size_t sealed_faces = 0;
  for (const auto& face : audit.faces) {
    const auto axis = static_cast<std::size_t>(face.face_index.axis);
    const auto lattice = face.face_index.lattice_index;
    const std::array face_coordinates{lattice.x, lattice.y, lattice.z};
    const std::array cell_coordinates{cell.x, cell.y, cell.z};
    if (face.open_area == 0) {
      ++sealed_faces;
      EXPECT_EQ(axis, 1);
      EXPECT_EQ(lattice.y, 0);
      EXPECT_DOUBLE_EQ(face.blocked_area, 1);
      EXPECT_DOUBLE_EQ(face.wall_normal_velocity, 0);
      EXPECT_NE(scheduler->committed().faces[face.face_slot].carrier_velocity, 0);
      continue;
    }
    EXPECT_DOUBLE_EQ(face.open_area, 1);
    const auto& projected = scheduler->committed().faces[face.face_slot];
    EXPECT_TRUE(projected.structurally_eligible);
    EXPECT_EQ(projected.side_tag, initial.particles[0].side_tag);
    EXPECT_EQ(projected.validity, LiquidFaceValidity3D::kValid);
    if (face_coordinates[axis] == cell_coordinates[axis]) {
      low[axis] = projected.carrier_velocity;
    } else {
      EXPECT_EQ(face_coordinates[axis], cell_coordinates[axis] + 1);
      high[axis] = projected.carrier_velocity;
    }
  }
  ASSERT_EQ(sealed_faces, 1);
  const std::array fraction{0.625, 0.375, 0.625};
  EXPECT_DOUBLE_EQ(low[1], 0);
  EXPECT_NE(high[1], low[1]);
  const auto& particle = scheduler->committed().particles[0];
  const std::array start{kRegularTransportStart.x, kRegularTransportStart.y, kRegularTransportStart.z};
  const std::array end{particle.world_position.x, particle.world_position.y, particle.world_position.z};
  double divergence = 0;
  for (std::size_t axis = 0; axis < fraction.size(); ++axis) {
    const double derivative = high[axis] - low[axis];
    divergence += derivative;
    const double velocity = low[axis] + fraction[axis] * derivative;
    EXPECT_DOUBLE_EQ(end[axis], start[axis] + kRegularTransportDuration * velocity);
  }
  EXPECT_DOUBLE_EQ(particle.volume_ratio, std::exp(kRegularTransportDuration * divergence));
  ASSERT_EQ(audit.transport.size(), 1);
  const auto& sample = audit.transport.front();
  EXPECT_TRUE(sample.active);
  EXPECT_DOUBLE_EQ(sample.position_in_cell_widths.x, fraction[0]);
  EXPECT_DOUBLE_EQ(sample.position_in_cell_widths.y, fraction[1]);
  EXPECT_DOUBLE_EQ(sample.position_in_cell_widths.z, fraction[2]);
  EXPECT_DOUBLE_EQ(sample.low_face_normal_velocity.x, low[0]);
  EXPECT_DOUBLE_EQ(sample.low_face_normal_velocity.y, low[1]);
  EXPECT_DOUBLE_EQ(sample.low_face_normal_velocity.z, low[2]);
  EXPECT_DOUBLE_EQ(sample.high_face_normal_velocity.x, high[0]);
  EXPECT_DOUBLE_EQ(sample.high_face_normal_velocity.y, high[1]);
  EXPECT_DOUBLE_EQ(sample.high_face_normal_velocity.z, high[2]);
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportDeclinesFractionalFloorAndUnsupportedCell) {
  auto initial = RegularTransportInitial();
  auto scene = RegularTransportScene();
  auto& tank = std::get<AxisAlignedLiquidTank3D>(scene.obstacle);
  tank.floor_y = 0.125;
  RegularTransportSchedulerAudit partial_audit;
  auto partial = LiquidSubstepScheduler3D::Create(initial, scene, Options(), ObserveRegularTransport(partial_audit));
  ASSERT_TRUE(partial.ok()) << partial.status();
  const auto partial_status = partial->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(partial_status.ok()) << partial_status;
  EXPECT_EQ(partial_audit.update_count, 1);
  ASSERT_EQ(partial_audit.cells.size(), 1);
  EXPECT_TRUE(std::ranges::any_of(partial_audit.faces, [](const auto& face) {
    return face.open_area > 0 && face.open_area < 1;
  }));
  ASSERT_EQ(partial_audit.transport.size(), 1);
  EXPECT_FALSE(partial_audit.transport.front().active);

  initial = Initial(2);
  initial.particles[0] = RegularTransportInitial().particles[0];
  initial.particles[1].world_position = {8.0625, 2.0625, 2.0625};
  scene = RegularTransportScene();
  RegularTransportSchedulerAudit mixed_audit;
  auto mixed = LiquidSubstepScheduler3D::Create(initial, scene, Options(), ObserveRegularTransport(mixed_audit));
  ASSERT_TRUE(mixed.ok()) << mixed.status();
  const auto mixed_status = mixed->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(mixed_status.ok()) << mixed_status;
  EXPECT_EQ(mixed_audit.update_count, 1);
  ASSERT_EQ(mixed_audit.cells.size(), 1);
  EXPECT_EQ(mixed_audit.cells.front().cell_index.x, -2);
  ASSERT_EQ(mixed_audit.transport.size(), 2);
  for (std::size_t index = 0; index < mixed_audit.transport.size(); ++index) {
    EXPECT_EQ(mixed_audit.transport[index].active, mixed->committed().particles[index].id == 1);
  }
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportKeepsOtherSceneClassesUnstaged) {
  const auto initial = RegularTransportInitial();
  const auto candidate_scene = RegularTransportScene();
  const auto candidate = LiquidSubstepScheduler3D::Create(initial, candidate_scene, Options());
  ASSERT_TRUE(candidate.ok()) << candidate.status();
  const auto check = [&](const LiquidSceneInputs3D& scene) {
    RegularTransportSchedulerAudit audit;
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), ObserveRegularTransport(audit));
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    const auto advanced = scheduler->AdvanceFrame(kRegularTransportDuration);
    ASSERT_TRUE(advanced.ok()) << advanced;
    EXPECT_EQ(audit.update_count, 1);
    EXPECT_TRUE(audit.transport.empty());
    EXPECT_FALSE(audit.cells.empty());
    return;
  };
  auto analytic = candidate_scene;
  analytic.field = LiquidAnalyticSphere3D{kRegularTransportStart, 0.7};
  const auto noncandidate = LiquidSubstepScheduler3D::Create(initial, analytic, Options());
  ASSERT_TRUE(noncandidate.ok()) << noncandidate.status();
  EXPECT_EQ(
      candidate->reserved_bytes(),
      noncandidate->reserved_bytes() + Options().capacities.particles * sizeof(LiquidRegularCellTransport3D)
  );
  check(analytic);
  auto moving = candidate_scene;
  std::get<AxisAlignedLiquidTank3D>(moving.obstacle).piston_velocity_x = 0.125;
  check(moving);
  auto compliant = candidate_scene;
  compliant.bulk_modulus = 1e6;
  check(compliant);
  auto divergence_source = candidate_scene;
  divergence_source.target_divergence = 0.125;
  check(divergence_source);
  auto capillary = candidate_scene;
  capillary.surface_tension = 1;
  check(capillary);
  auto open_domain = candidate_scene;
  open_domain.obstacle = std::monostate{};
  check(open_domain);
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportRollsBackAndReplaysAfterGrowth) {
  const auto initial = RegularTransportInitial();
  const auto scene = RegularTransportScene();
  auto options = Options();
  options.capacities.faces = 64;
  options.capacity_replay_reservations = Options().capacities;
  RegularTransportSchedulerAudit audit;
  auto replay = LiquidSubstepScheduler3D::Create(initial, scene, options, ObserveRegularTransport(audit));
  ASSERT_TRUE(replay.ok()) << replay.status();
  bool reject_once = true;
  replay->SetStageFaultHookForTesting({
      .context = &reject_once,
      .before_stage =
          [](void* context, LiquidPhysicalStage3D stage, double) {
            if (stage == LiquidPhysicalStage3D::kAccept && *static_cast<bool*>(context)) {
              *static_cast<bool*>(context) = false;
              return absl::AbortedError("");
            }
            return absl::OkStatus();
          },
  });
  // Grow before the first complete update, then reject its selected trial.
  const auto advanced = replay->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(advanced.ok()) << advanced;
  EXPECT_EQ(replay->diagnostics().capacity_replays, 1);
  EXPECT_EQ(replay->diagnostics().dt_retries, 1);
  EXPECT_EQ(audit.update_count, 3);
  ASSERT_EQ(audit.transport.size(), 1);
  EXPECT_TRUE(audit.transport.front().active);
  auto direct_options = Options();
  direct_options.maximum_step_duration = kRegularTransportDuration / 2;
  auto direct = LiquidSubstepScheduler3D::Create(initial, scene, direct_options);
  ASSERT_TRUE(direct.ok()) << direct.status();
  const auto direct_status = direct->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(direct_status.ok()) << direct_status;
  EXPECT_EQ(replay->reserved_bytes(), direct->reserved_bytes());
  ExpectIdenticalSerializedState(direct->committed(), replay->committed());
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportPreservesDamMaterialThroughFrameSix) {
  constexpr double cell_width = 0.0625;
  constexpr double reference_volume = cell_width * cell_width * cell_width;
  constexpr std::size_t particle_count = 256;
  auto initial = Initial(particle_count);
  initial.geometry.cell_width = cell_width;
  std::size_t ordinal = 0;
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        auto& particle = initial.particles[ordinal++];
        particle.world_position = {(x + 0.5) * cell_width, (y + 0.5) * cell_width, (z + 0.5) * cell_width};
        particle.reference_volume = reference_volume;
        particle.mass = 1000 * reference_volume;
        particle.pic_initialized = false;
      }
    }
  }
  // Match the public dam fixture and its frozen default blend through the first drifting frame.
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{0.9 * cell_width};
  scene.obstacle = AxisAlignedLiquidTank3D{
      .start_low_x = 0,
      .high_x = 2,
      .floor_y = 0,
      .low_z = 0,
      .high_z = 0.25,
      .piston_velocity_x = 0,
  };
  LiquidStepOptions3D options;
  options.particle_energy_blend = 0.6;
  options.maximum_step_duration = 0.01;
  options.capacities = {1024, 128, 4096, 24576, 256};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  for (int frame = 1; frame <= 6; ++frame) {
    SCOPED_TRACE(frame);
    const auto advanced = scheduler->AdvanceFrame(frame / 24.0);
    ASSERT_TRUE(advanced.ok()) << advanced;
    ASSERT_EQ(scheduler->committed().particles.size(), particle_count);
    std::array<bool, particle_count> seen{};
    double volume = 0;
    for (const auto& particle : scheduler->committed().particles.values()) {
      ASSERT_GE(particle.id, 1);
      ASSERT_LE(particle.id, particle_count);
      EXPECT_FALSE(seen[particle.id - 1]);
      seen[particle.id - 1] = true;
      EXPECT_DOUBLE_EQ(particle.reference_volume, reference_volume);
      EXPECT_DOUBLE_EQ(particle.mass, 1000 * reference_volume);
      volume += particle.reference_volume * particle.volume_ratio;
    }
    EXPECT_LE(std::abs(volume / (particle_count * reference_volume) - 1), 0.01);
  }
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
  scene.target_divergence = 0.3;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_GT(scheduler->committed().particles[0].volume_ratio, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].mass, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].reference_volume, 1);
}
TEST(LiquidSubstep3D, AnalyticSpherePreservesExactRestPressureAtBothResolutions) {
  constexpr double kRadius = 0.125;
  constexpr double kDensity = 1000;
  struct AirPressureAudit {
    std::size_t air_links = 0;
    std::size_t incorrect_pressures = 0;
    std::size_t pressure_iterations = 0;
    double maximum_corrected_speed = 0;
  };
  for (const int cells_per_radius : {3, 5}) {
    SCOPED_TRACE(cells_per_radius);
    const double cell_width = kRadius / cells_per_radius;
    auto initial = Initial(cells_per_radius == 3 ? 136 : 552);
    initial.geometry.cell_width = cell_width;
    const double reference_volume = (4 * std::numbers::pi * kRadius * kRadius * kRadius / 3) / initial.particles.size();
    std::size_t ordinal = 0;
    for (int z = -cells_per_radius; z < cells_per_radius; ++z) {
      for (int y = -cells_per_radius; y < cells_per_radius; ++y) {
        for (int x = -cells_per_radius; x < cells_per_radius; ++x) {
          const Vec3 position{(x + 0.5) * cell_width, (y + 0.5) * cell_width, (z + 0.5) * cell_width};
          if (std::hypot(position.x, position.y, position.z) < kRadius) {
            ASSERT_LT(ordinal, initial.particles.size());
            auto& particle = initial.particles[ordinal++];
            particle.world_position = position;
            particle.reference_volume = reference_volume;
            particle.mass = kDensity * reference_volume;
            particle.pic_initialized = false;
          }
        }
      }
    }
    ASSERT_EQ(ordinal, initial.particles.size());
    LiquidSceneInputs3D scene;
    scene.field = LiquidAnalyticSphere3D{{}, kRadius};
    scene.world_acceleration = {};
    scene.liquid_density = kDensity;
    scene.surface_tension = 1;
    scene.ambient_pressure = 3;
    LiquidStepOptions3D options;
    options.capacities = {1024, 128, 4096, 24576, 256};
    AirPressureAudit audit;
    auto backend = CpuLiquidSubstepBackend3D();
    backend.context = &audit;
    backend.project = [](void* context,
                         SparseMacGridView3D grid,
                         double duration,
                         const LiquidPressureProblemView3D& problem,
                         MacPressureSolverOptions3D solver_options,
                         MacPressureWorkspace3D& workspace) {
      auto& observed = *static_cast<AirPressureAudit*>(context);
      for (const auto& face : problem.faces) {
        if (const auto* air = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link)) {
          ++observed.air_links;
          observed.incorrect_pressures += air->exterior_pressure != 3 + 2 / kRadius;
        }
      }
      const auto projection =
          CpuLiquidSubstepBackend3D().project(nullptr, grid, duration, problem, solver_options, workspace);
      if (projection.ok()) {
        observed.pressure_iterations += projection->diagnostics.iteration_count;
        for (const double velocity : projection->corrected_face_velocities) {
          observed.maximum_corrected_speed = std::max(observed.maximum_corrected_speed, std::abs(velocity));
        }
      }
      return projection;
    };
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    const auto status = scheduler->AdvanceFrame(1.0 / 120);
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ(audit.air_links, cells_per_radius == 3 ? 192 : 480);
    EXPECT_EQ(audit.incorrect_pressures, 0);
    EXPECT_EQ(audit.pressure_iterations, 0);
    EXPECT_DOUBLE_EQ(audit.maximum_corrected_speed, 0);
    scene.dynamic_viscosity = 1.0;
    auto viscous = LiquidSubstepScheduler3D::Create(initial, scene, options);
    ASSERT_TRUE(viscous.ok()) << viscous.status();
    const auto viscous_status = viscous->AdvanceFrame(1.0 / 120);
    ASSERT_TRUE(viscous_status.ok()) << viscous_status;
    for (const auto& cell : viscous->committed().cells.values()) {
      EXPECT_NEAR(cell.previous_pressure, 19.0, 1e-3);
    }
    for (const auto& particle : viscous->committed().particles.values()) {
      const auto velocity = particle.world_velocity;
      EXPECT_LT(std::hypot(velocity.x, velocity.y, velocity.z), 1e-5);
    }
    const auto& accepted = viscous->diagnostics().accepted_measurement;
    ASSERT_TRUE(accepted.has_value());
    ASSERT_TRUE(accepted->viscous_max_momentum_residual_kg_m_per_s.has_value());
    EXPECT_LT(*accepted->viscous_max_momentum_residual_kg_m_per_s, 1e-5);
  }
}

TEST(LiquidSubstep3D, ViscousFractionalTankFillPreservesHydrostaticRest) {
  auto fixture = viscous_tank_test::FractionalFill();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(viscous_tank_test::kStepDuration);
  ASSERT_TRUE(status.ok()) << status;
  viscous_tank_test::ExpectHydrostaticRest(*scheduler);

  auto& tank = std::get<AxisAlignedLiquidTank3D>(fixture->scene.obstacle);
  tank.start_low_x = 0.75;
  for (auto& particle : fixture->initial.particles.values()) {
    if (particle.world_position.x < 1.0) {
      particle.world_position.x = 0.875;
    }
  }
  auto solid_cut = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options);
  ASSERT_TRUE(solid_cut.ok()) << solid_cut.status();
  const auto rejected = solid_cut->AdvanceFrame(viscous_tank_test::kStepDuration);
  EXPECT_EQ(rejected.code(), absl::StatusCode::kUnimplemented);
  ExpectIdenticalSerializedState(fixture->initial, solid_cut->committed());
}

TEST(LiquidSubstep3D, ViscousCartesianCutTankPreservesHydrostaticRest) {
  const auto fixture = viscous_tank_test::CartesianCut();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(1.0 / 50.0);
  ASSERT_TRUE(status.ok()) << status;
  viscous_tank_test::ExpectHydrostaticRest(*scheduler);
}

TEST(LiquidSubstep3D, ViscousCartesianCutPistonUsesWallNormalStrainAndFlux) {
  auto fixture = viscous_tank_test::CartesianCut();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  auto& tank = std::get<AxisAlignedLiquidTank3D>(fixture->scene.obstacle);
  constexpr double kWallSpeed = 0.05;
  tank.piston_velocity_x = kWallSpeed;
  struct PistonAudit {
    std::size_t wall_rows = 0;
    double top_outward_flux = 0.0;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.project_viscous = [](void* context,
                               const LiquidBoundaryGeometryViewInputs3D& geometry,
                               const LiquidPressureProblemView3D& problem,
                               double viscosity,
                               std::span<const double> cell_viscosity,
                               MacPressureSolverOptions3D options,
                               MacPressureWorkspace3D& workspace) {
    return liquid_step_internal::ProjectUniformViscosity(
        geometry,
        problem,
        viscosity,
        cell_viscosity,
        options,
        workspace,
        [&](const pressure_internal::UniformViscousPressureInputs3D& inputs,
            std::span<const MacStrainRow3D> rows,
            MacPressureSolverOptions3D controls,
            pressure_internal::MacPressureScratch3D& pressure_scratch) {
          if (!rows.empty() && rows[0].low_cell.x == 0 && rows[0].low_cell.y == 0 && rows[0].low_cell.z == 0
              && rows[0].first_axis == FaceNormalAxis::kX && rows[0].second_axis == FaceNormalAxis::kX) {
            EXPECT_DOUBLE_EQ(rows[0].quadrature_weight, 15.0);
            EXPECT_DOUBLE_EQ(rows[0].terms[0].coefficient, -1.0 / 0.75);
            EXPECT_DOUBLE_EQ(rows[0].terms[1].coefficient, 1.0 / 0.75);
            ++static_cast<PistonAudit*>(context)->wall_rows;
          }
          const auto result =
              pressure_internal::ProjectScratchMacViscousPressure3D(inputs, rows, controls, pressure_scratch);
          if (result.ok()) {
            auto& observed = *static_cast<PistonAudit*>(context);
            observed.top_outward_flux = 0.0;
            for (std::size_t face = 0; face < inputs.pressure_faces.size(); ++face) {
              const auto& link = inputs.pressure_faces[face];
              if (link.face_index.axis == FaceNormalAxis::kY
                  && std::holds_alternative<DirichletMacPressureLink3D>(link.pressure_link)) {
                observed.top_outward_flux += link.open_area * result->face_velocities[face];
              }
            }
          }
          return result;
        }
    );
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options, backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(viscous_tank_test::kStepDuration);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_GT(audit.wall_rows, 0u);
  const auto& accepted = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(accepted.has_value());
  const double expected_flux = kWallSpeed * 3 * viscous_tank_test::kFillHeight;
  EXPECT_NEAR(accepted->step_start_inward_piston_flux_m3_per_s, expected_flux, 1e-12);
  EXPECT_NEAR(audit.top_outward_flux, expected_flux, 0.01 * expected_flux);
  EXPECT_LT(accepted->viscous_max_momentum_residual_kg_m_per_s.value_or(1.0), 1e-5);
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

TEST(LiquidSubstep3D, RejectedGatherRunReplaysOnlyThroughEarlierFieldFailure) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wall_test = liquid_wall_samples_test;
  struct Audit {
    int batch_calls = 0;
    int scalar_gather_calls = 0;
    int field_calls_after_batch = 0;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.gather_run = [](void* context,
                          const LiquidBoundaryGatherViewInputs3D& input,
                          LiquidBoundaryWorkspace3D&) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    auto& observed = *static_cast<Audit*>(context);
    ++observed.batch_calls;
    EXPECT_EQ(input.particles.size(), 2);
    EXPECT_EQ(input.particles.front().side_tag, input.particles.back().side_tag);
    return absl::OutOfRangeError("");
  };
  backend.gather = [](void* context, const LiquidBoundaryGatherViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace
                   ) {
    ++static_cast<Audit*>(context)->scalar_gather_calls;
    EXPECT_EQ(input.particles.size(), 1);
    return GatherLiquidBoundaryParticles3D(input, workspace);
  };
  backend.sample_field =
      [](void* context, const LiquidSceneField3D& field, std::span<const LiquidParticleState3D> particles, Vec3 position
      ) -> absl::StatusOr<double> {
    auto& observed = *static_cast<Audit*>(context);
    if (observed.batch_calls > 0) {
      ++observed.field_calls_after_batch;
      return absl::InvalidArgumentError("");
    }
    return CpuLiquidSubstepBackend3D().sample_field(nullptr, field, particles, position);
  };
  const auto initial = wall_test::SlabParticles();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, wall_test::SlabScene(), Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(audit.batch_calls, 1);
  EXPECT_EQ(audit.scalar_gather_calls, 1);
  EXPECT_EQ(audit.field_calls_after_batch, 1);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, GatheredFieldBatchMatchesScalarAndReplaysRejectionInOrder) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wall_test = liquid_wall_samples_test;
  const auto initial = wall_test::SlabParticles();
  const auto scene = wall_test::SlabScene();
  const auto options = Options();
  auto baseline = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  ASSERT_TRUE(baseline->AdvanceFrame(0.01).ok());
  for (const bool reject_batch : {false, true}) {
    SCOPED_TRACE(reject_batch);
    struct Audit {
      bool reject_batch;
      std::size_t batch_calls = 0;
      std::size_t gathered_field_samples = 0;
      std::size_t scalar_calls_after_batch = 0;
    } audit{reject_batch};
    auto backend = CpuLiquidSubstepBackend3D();
    backend.context = &audit;
    backend.gather_run = [](void*, const LiquidBoundaryGatherViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace
                         ) {
      return GatherLiquidBoundaryParticles3D(input, workspace);
    };
    backend.sample_gathered_fields = [](void* context,
                                        const LiquidSceneField3D& field,
                                        std::span<const LiquidParticleState3D> particles,
                                        std::span<const Vec3> positions,
                                        std::span<double> distances) {
      auto& observed = *static_cast<Audit*>(context);
      ++observed.batch_calls;
      EXPECT_EQ(positions.size(), 2);
      EXPECT_EQ(positions.size(), distances.size());
      for (const auto position : positions) {
        const auto expected = wall_test::ParticlePosition(observed.gathered_field_samples++);
        EXPECT_DOUBLE_EQ(position.x, expected.x);
        EXPECT_DOUBLE_EQ(position.y, expected.y);
        EXPECT_DOUBLE_EQ(position.z, expected.z);
      }
      if (observed.reject_batch) {
        return absl::InvalidArgumentError("");
      }
      return CpuLiquidSubstepBackend3D().sample_field_batch(nullptr, field, particles, positions, distances);
    };
    backend.sample_field = [](void* context,
                              const LiquidSceneField3D& field,
                              std::span<const LiquidParticleState3D> particles,
                              Vec3 position) -> absl::StatusOr<double> {
      auto& observed = *static_cast<Audit*>(context);
      if (observed.batch_calls > 0) {
        ++observed.scalar_calls_after_batch;
        if (observed.reject_batch) {
          EXPECT_DOUBLE_EQ(position.y, wall_test::ParticlePosition(0).y);
          return absl::OutOfRangeError("");
        }
      }
      return CpuLiquidSubstepBackend3D().sample_field(nullptr, field, particles, position);
    };
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    const auto status = scheduler->AdvanceFrame(0.01);
    if (reject_batch) {
      EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
      EXPECT_EQ(audit.batch_calls, 1);
      EXPECT_EQ(audit.gathered_field_samples, 2);
      ExpectIdenticalSerializedState(initial, scheduler->committed());
    } else {
      ASSERT_TRUE(status.ok()) << status;
      EXPECT_EQ(audit.batch_calls, 2);
      EXPECT_EQ(audit.gathered_field_samples, 4);
      ExpectIdenticalSerializedState(baseline->committed(), scheduler->committed());
      EXPECT_EQ(scheduler->diagnostics().old_pic_fallback_count, baseline->diagnostics().old_pic_fallback_count);
      EXPECT_EQ(scheduler->diagnostics().new_pic_fallback_count, baseline->diagnostics().new_pic_fallback_count);
    }
    EXPECT_EQ(audit.scalar_calls_after_batch, 1);
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
    EXPECT_EQ(scheduler->diagnostics().accepted_attempts, 0);
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
  source.emitted_particle_template.phase_id = 4;
  source.emitted_particle_template.constituent_count = 2;
  source.emitted_particle_template.constituents[0] = {8, 0.25};
  source.emitted_particle_template.constituents[1] = {14, 0.75};
  auto scene = Scene();
  scene.sources = std::span(&source, 1);
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1000.0}, {14, 4, 1020.0}}};
  scene.constituent_materials = kMaterials;
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
  for (const auto& particle : replay->committed().particles.values()) {
    if (particle.id != 1) {
      EXPECT_EQ(particle.phase_id, 4);
      ASSERT_EQ(particle.constituent_count, 2);
      EXPECT_EQ(particle.constituents[0], (LiquidConstituentFraction3D{8, 0.25}));
      EXPECT_EQ(particle.constituents[1], (LiquidConstituentFraction3D{14, 0.75}));
    }
  }
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
TEST(Viscosity, BoundedVelocityBlockSolveAllocatesNothingAndMatchesOwningPath) {
  const std::array<double, 3> inertia{2.0, 3.0, 0.0};
  std::array<double, 3> rhs{2.0, -1.0, 0.0};
  const std::array rows{MacStrainRow3D{
      .low_cell = {},
      .first_axis = FaceNormalAxis::kX,
      .second_axis = FaceNormalAxis::kY,
      .terms =
          {{{.face_slot = 0, .coefficient = -1.0},
            {.face_slot = 1, .coefficient = 1.0},
            {.face_slot = 2, .coefficient = 0.5}}},
      .term_count = 3,
      .quadrature_weight = 4.0,
  }};
  const MacViscousVelocityBlockInputs3D inputs{rows, inertia, 0.2};
  const auto owning = SolveMacViscousVelocityBlock3D(inputs, rhs);
  ASSERT_TRUE(owning.ok()) << owning.status();
  std::array<MacStrainRow3D, 1> weighted_rows;
  std::array<double, 3> diagonal{}, velocity{}, residual{}, preconditioned{}, direction{}, applied{};
  const viscosity_internal::VelocityBlockScratch3D scratch{
      .weighted_rows = weighted_rows,
      .diagonal = diagonal,
      .velocity = velocity,
      .residual = residual,
      .preconditioned = preconditioned,
      .direction = direction,
      .applied = applied,
  };
  scheduler_allocations = 0;
  scheduler_frees = 0;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  const auto bounded = viscosity_internal::SolveVelocityBlockWithScratch3D(inputs, rhs, {}, scratch);
  _CrtSetAllocHook(scheduler_previous_hook);
  ASSERT_TRUE(bounded.ok()) << bounded.status();
  EXPECT_TRUE(std::ranges::equal(bounded->velocity, owning->velocity));
  EXPECT_EQ(bounded->iterations, owning->iterations);
  EXPECT_DOUBLE_EQ(bounded->max_residual, owning->max_residual);
  EXPECT_EQ(scheduler_allocations, 0u);
  EXPECT_EQ(scheduler_frees, 0u);

  auto alias = scratch;
  alias.velocity = residual;
  scheduler_allocations = 0;
  scheduler_frees = 0;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  const auto rejected = viscosity_internal::SolveVelocityBlockWithScratch3D(inputs, rhs, {}, alias);
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(scheduler_allocations, 0u);
  EXPECT_EQ(scheduler_frees, 0u);
  const auto original_rhs = rhs;
  scheduler_allocations = 0;
  scheduler_frees = 0;
  auto failure_code = absl::StatusCode::kOk;
  bool empty_message = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    const auto exhausted =
        viscosity_internal::SolveVelocityBlockWithScratch3D(inputs, rhs, {.max_iterations = 1}, scratch);
    failure_code = exhausted.status().code();
    empty_message = exhausted.status().message().empty();
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_EQ(failure_code, absl::StatusCode::kAborted);
  EXPECT_TRUE(empty_message);
  EXPECT_EQ(rhs, original_rhs);
  EXPECT_EQ(scheduler_allocations, 0u);
  EXPECT_EQ(scheduler_frees, 0u);
}

TEST(LiquidSubstep3D, PositiveViscosityAttemptsAllocateAndFreeNothing) {
  const auto fixture = viscous_tank_test::FractionalFill();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  constexpr double duration = viscous_tank_test::kStepDuration;
  for (auto failure : {absl::StatusCode::kOk, absl::StatusCode::kAborted, absl::StatusCode::kFailedPrecondition}) {
    auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(duration).ok());
    scheduler->SetStageFaultHookForTesting({&failure, [](void* context, LiquidPhysicalStage3D stage, double) {
                                              if (stage != LiquidPhysicalStage3D::kAccept) {
                                                return absl::OkStatus();
                                              }
                                              return absl::Status(*static_cast<absl::StatusCode*>(context), {});
                                            }});
    scheduler_allocations = 0;
    scheduler_frees = 0;
    bool succeeded = false;
    auto disposition = LiquidAttemptDisposition3D::kFatal;
    scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
    {
      const auto result = scheduler->Attempt(duration);
      succeeded = result.ok();
      if (succeeded) {
        disposition = result->disposition;
      }
    }
    _CrtSetAllocHook(scheduler_previous_hook);
    ASSERT_TRUE(succeeded);
    EXPECT_EQ(scheduler_allocations, 0u);
    EXPECT_EQ(scheduler_frees, 0u);
    if (failure == absl::StatusCode::kOk) {
      EXPECT_EQ(disposition, LiquidAttemptDisposition3D::kAccepted);
      EXPECT_DOUBLE_EQ(scheduler->committed().clock.time, duration);
    } else {
      const auto expected = failure == absl::StatusCode::kAborted ? LiquidAttemptDisposition3D::kRetryDt
                                                                  : LiquidAttemptDisposition3D::kFatal;
      EXPECT_EQ(disposition, expected);
      ExpectIdenticalSerializedState(fixture->initial, scheduler->committed());
    }
  }
}

TEST(LiquidSubstep3D, ViscosityPhysicalFailuresAllocateAndFreeNothing) {
  constexpr double duration = viscous_tank_test::kStepDuration;
  for (const bool invalid_geometry : {false, true}) {
    const auto fixture = viscous_tank_test::ViscosityFailure(invalid_geometry);
    ASSERT_TRUE(fixture.ok()) << fixture.status();
    auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(duration).ok());
    scheduler_allocations = 0;
    scheduler_frees = 0;
    bool succeeded = false;
    auto disposition = LiquidAttemptDisposition3D::kFatal;
    scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
    {
      const auto result = scheduler->Attempt(duration);
      succeeded = result.ok();
      if (succeeded) {
        disposition = result->disposition;
      }
    }
    _CrtSetAllocHook(scheduler_previous_hook);
    ASSERT_TRUE(succeeded);
    EXPECT_EQ(scheduler_allocations, 0u);
    EXPECT_EQ(scheduler_frees, 0u);
    EXPECT_EQ(
        disposition,
        invalid_geometry ? LiquidAttemptDisposition3D::kFatal : LiquidAttemptDisposition3D::kRetryDt
    );
    EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
    EXPECT_EQ(
        scheduler->diagnostics().failure_code,
        invalid_geometry ? absl::StatusCode::kUnimplemented : absl::StatusCode::kAborted
    );
    ExpectIdenticalSerializedState(fixture->initial, scheduler->committed());
  }
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
TEST(LiquidSubstep3D, OwnsResolvedCavityAcrossAcceptedSteps) {
  const auto shell = cavity_test::ParticleShell();
  ASSERT_TRUE(shell.ok()) << shell.status();
  const auto& initial = *shell;
  int initializations = 0;
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{.radius = 0.9};
  scene.world_acceleration = {};
  scene.cavity = LiquidSceneCavity3D{
      .atmospheric_absolute_pressure = 100000.0,
      .context = &initializations,
      .initialize = [](void* context,
                       const LiquidAirComponent3D& component) -> absl::StatusOr<LiquidCavityInventory3D> {
        ++*static_cast<int*>(context);
        return InitializeSubmergedLiquidCavity3D(component.gas_volume, 100000.0, 0.0, 0.0);
      },
  };
  const auto options = cavity_test::ShellOptions(initial.particles.size());
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  int pressure_attempts = 0;
  scheduler->SetStageFaultHookForTesting({
      .context = &pressure_attempts,
      .before_stage =
          [](void* context, LiquidPhysicalStage3D stage, double) {
            if (stage == LiquidPhysicalStage3D::kPressure && ++*static_cast<int*>(context) == 2) {
              return absl::Status(absl::StatusCode::kResourceExhausted, {});
            }
            return absl::OkStatus();
          },
  });
  for (const double target : {0.002, 0.003}) {
    const auto status = scheduler->AdvanceFrame(target);
    ASSERT_TRUE(status.ok()) << status << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
    const auto& measurement = scheduler->diagnostics().accepted_measurement;
    EXPECT_EQ(scheduler->diagnostics().capacity_replays, target == 0.002 ? 1 : 0);
    ASSERT_TRUE(measurement && measurement->cavity);
    EXPECT_EQ(measurement->cavity->boundaries.active_cavity_count, 1);
    EXPECT_GT(measurement->cavity->boundaries.air_pressure_count, 0);
    EXPECT_EQ(measurement->cavity->lineage.next_id, 2);
    EXPECT_DOUBLE_EQ(measurement->cavity->lineage.unmatched_inventory, 0.0);
  }
  EXPECT_EQ(initializations, 1);

  LiquidResolvedSceneView3D resolved;
  resolved.scene = scene;
  resolved.scene.cavity = LiquidSceneCavity3D{
      .atmospheric_absolute_pressure = 100000.0,
      .birth_hydrostatic_gauge_pressure = 100.0,
  };
  resolved.options = options;
  const auto encoded = EncodeLiquidResolvedScene3D(resolved);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  auto loaded = LiquidSubstepScheduler3D::Create(initial, decoded->scene, options);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const auto loaded_step = loaded->AdvanceFrame(0.001);
  ASSERT_TRUE(loaded_step.ok()) << loaded_step;
  const auto& loaded_measurement = loaded->diagnostics().accepted_measurement;
  ASSERT_TRUE(loaded_measurement && loaded_measurement->cavity);
  EXPECT_EQ(loaded_measurement->cavity->boundaries.active_cavity_count, 1);
  EXPECT_GT(loaded_measurement->pressure_rhs_l2_m3_per_s, 0.0);

  ASSERT_EQ(loaded->committed().next_cavity_id, 2);
  ASSERT_GT(loaded->committed().cavity_gas.size(), 0);
  LiquidResumeMetadata3D resume;
  resume.resolved_fingerprint = *ResolvedSceneFingerprint(decoded->View());
  resume.execution_id = LiquidExecutionId3D(resume.resolved_fingerprint);
  resume.capacities = options.capacities;
  resume.capacity_replay_reservations = options.capacity_replay_reservations;
  const auto checkpoint = EncodeLiquidCheckpoint3D(loaded->committed(), resume);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const auto restored = DecodeLiquidCheckpoint3D(checkpoint->values());
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto restarted = LiquidSubstepScheduler3D::Create(restored->state, decoded->scene, options);
  ASSERT_TRUE(restarted.ok()) << restarted.status();
  ASSERT_TRUE(loaded->AdvanceFrame(0.002).ok());
  ASSERT_TRUE(restarted->AdvanceFrame(0.002).ok());
  EXPECT_EQ(restarted->committed().next_cavity_id, 2);
  ExpectIdenticalSerializedState(loaded->committed(), restarted->committed());
}

TEST(LiquidSubstep3D, PreparedCavityPressuresApplyOnlyAfterRadialRetry) {
  struct CavityFixture {
    std::array<LiquidAirCellPressure3D, 256> pressures;
    int calls = 0;
  } fixture;
  auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.cavity_provider = {
      .context = &fixture,
      .prepare = [](void* context, double step_start_time, const LiquidBoundaryGeometryViewInputs3D& samples
                 ) -> absl::StatusOr<LiquidStepCavityBoundaries3D> {
        auto& cavity = *static_cast<CavityFixture*>(context);
        ++cavity.calls;
        EXPECT_DOUBLE_EQ(step_start_time, 0.0);
        std::size_t count = 0;
        for (const auto& center : samples.center_distances) {
          if (center.signed_distance >= 0.0) {
            if (count == cavity.pressures.size()) {
              return absl::ResourceExhaustedError("cavity fixture capacity");
            }
            cavity.pressures[count++] = {center.cell_index, 17.0};
          }
        }
        return LiquidStepCavityBoundaries3D{
            .air_pressures = std::span(cavity.pressures).first(count),
            .maximum_step_duration = 0.006,
        };
      },
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto rejected = scheduler->Attempt(0.01);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ASSERT_TRUE(rejected->retry_duration);
  EXPECT_TRUE(scheduler->committed() == initial);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement);

  const auto accepted = scheduler->Attempt(*rejected->retry_duration);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  EXPECT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
  EXPECT_EQ(fixture.calls, 2);
  const auto& measurement = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement && measurement->step_start_cavity_bound_seconds);
  EXPECT_DOUBLE_EQ(*measurement->step_start_cavity_bound_seconds, 0.006);
  ASSERT_GT(scheduler->committed().cells.size(), 0);
  EXPECT_NEAR(scheduler->committed().cells[0].previous_pressure, 17.0, 1e-7);
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
TEST(LiquidSubstep3D, GuidedFloatingBoxAttemptsAllocateAndFreeNothing) {
  auto scene = wetted_piston_pressure_test::PlaneScene(0.75, 0.0);
  scene.obstacle = AxisAlignedLiquidTank3D{0.125, 0.875, 0.125, 0.125, 0.875, 0.0};
  auto scheduler = LiquidSubstepScheduler3D::Create({}, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.125).ok());
  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.375, 0.5625, 0.375},
      .linear_velocity = {0.0, 0.5, 0.0},
  };
  auto malformed = body;
  malformed.mass = 0.0;
  scheduler_allocations = 0;
  scheduler_frees = 0;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  bool rejected = false;
  bool accepted = false;
  {
    const auto invalid = scheduler->AttemptWithGuidedFloatingBox(0.125, malformed, {0.125, 0.3125, 0.125});
    rejected = absl::IsInvalidArgument(invalid.status());
    const auto valid = scheduler->AttemptWithGuidedFloatingBox(0.125, body, {0.125, 0.3125, 0.125});
    accepted = valid.ok() && valid->body.has_value();
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_TRUE(rejected);
  EXPECT_TRUE(accepted);
  EXPECT_EQ(scheduler_allocations, 0);
  EXPECT_EQ(scheduler_frees, 0);
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
TEST(LiquidSubstep3D, ViscousCapacityReplayReservesCoupledWorkspace) {
  const auto fixture = viscous_tank_test::FractionalFill();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  auto options = fixture->options;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.cells *= 2;
  auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  bool injected = false;
  scheduler->SetStageFaultHookForTesting({&injected, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            auto& hit = *static_cast<bool*>(context);
                                            if (stage != LiquidPhysicalStage3D::kAccept || hit) {
                                              return absl::OkStatus();
                                            }
                                            hit = true;
                                            return absl::Status(absl::StatusCode::kResourceExhausted, {});
                                          }});
  const auto status = scheduler->AdvanceFrame(viscous_tank_test::kStepDuration);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_TRUE(injected);
  EXPECT_EQ(scheduler->committed().clock.epoch, 1u);
  viscous_tank_test::ExpectHydrostaticRest(*scheduler);
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
