
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>

#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/solid_contact_move_particle_shorter_step_cases.h"

namespace fluidity::contact {
namespace {

// Fails the calling test unless the scene fails with kTooCloseToCall, that close call and that solid, and
// can_shorter_step_cure false, at its step T and at T / 2, T / 4 and T / 16.
void CheckSameCloseCallAtShorterSteps(
    const MoveParticleCase& scene,
    ContactCloseCall close_call,
    std::uint32_t solid_id
) {
  for (const int halving_count : {0, 1, 2, 4}) {
    SCOPED_TRACE("step divided by 2^" + std::to_string(halving_count));
    const MoveParticleCall call = ComputeMoveWithHalvings(scene, halving_count);
    EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
    EXPECT_EQ(call.failure.close_call, close_call);
    EXPECT_EQ(call.failure.solid_id, solid_id);
    EXPECT_FALSE(call.failure.can_shorter_step_cure);
  }
}

// --- Close calls that no shorter step cures: decided at the start of the step ---

TEST(MoveParticle, PaddleThroughBottomSideRestingOnIt_SameCloseCallAtEveryShorterStep) {
  // A paddle turning through the bottom side of the domain box, the particle resting on that side against the paddle:
  // hit at 0 on the paddle, the side within tol + tol / 16 at the carry's start, so the carry is not safe at any step
  // length.
  CheckSameCloseCallAtShorterSteps(
      BuildShorterStepCase("PaddleThroughBottomSideRestingOnIt"),
      ContactCloseCall::kTurningSolidNearOtherSolid,
      2
  );
}

TEST(MoveParticle, PaddleOnShaftInCrease_SameCloseCallAtEveryShorterStep) {
  // The crease of a paddle and the shaft it turns with, the particle at the rest gap of both: the paddle (id 3) is hit
  // at 0 and the shaft lies within the window.
  CheckSameCloseCallAtShorterSteps(
      BuildShorterStepCase("PaddleOnShaftInCrease"),
      ContactCloseCall::kTurningSolidNearOtherSolid,
      3
  );
}

TEST(MoveParticle, DrivenBoxesInCrease_SameCloseCallAtEveryShorterStep) {
  // Two driven boxes with reports_reaction, the particle in their crease: the start's one response pushes both, naming
  // the higher index, id 3.
  CheckSameCloseCallAtShorterSteps(
      BuildShorterStepCase("DrivenBoxesInCrease"),
      ContactCloseCall::kSecondReactingSolid,
      3
  );
}

TEST(MoveParticle, DescendingBoxAtBottomSide_SameSqueezeAtEveryShorterStep) {
  // A box 2 tol above the bottom side coming down on a particle at rest between them: the stick to the box still
  // approaches the side, a squeeze naming the domain box (id 1), the solid the failing velocity approaches fastest.
  CheckSameCloseCallAtShorterSteps(
      BuildShorterStepCase("DescendingBoxAtBottomSide"),
      ContactCloseCall::kSqueezedBetweenSolids,
      1
  );
}

TEST(MoveParticle, SeventeenFloorsAtStart_TooManyHitsAtEveryShorterStep) {
  // 17 active surfaces at the start, one more than one response reads (kMaxTrackedSurfaces), naming the record's solid.
  CheckSameCloseCallAtShorterSteps(BuildShorterStepCase("SeventeenFloorsAtStart"), ContactCloseCall::kTooManyHits, 1);
}

// The scene PaddleThroughBottomSideRestingOnIt with one more fixed solid near the particle, beyond the window of the
// active surfaces but within the closing bound of the carry, given the lowest id (first in the array) or the highest
// (last). The bottom side, within the window, fails at every step length whatever the order.
enum class ExtraSolidPlace : std::uint8_t {
  kFirst,  // Lowest id.
  kLast,  // Highest id.
};

MoveParticleCase BuildPaddleSceneWithExtraSolid(Solid extra, ExtraSolidPlace place) {
  MoveParticleCase scene = BuildShorterStepCase("PaddleThroughBottomSideRestingOnIt");
  Solid domain_box = scene.solids[0];
  Solid paddle = scene.solids[1];
  if (place == ExtraSolidPlace::kFirst) {
    extra.solid_id = 1;
    paddle.solid_id = 2;
    domain_box.solid_id = 3;
    scene.solids = {extra, paddle, domain_box};
  } else {
    domain_box.solid_id = 1;
    paddle.solid_id = 2;
    extra.solid_id = 3;
    scene.solids = {domain_box, paddle, extra};
  }
  return scene;
}

// A fixed flat ceiling 3 tol above the particle of PaddleThroughBottomSideRestingOnIt (2 tol beyond the rest gap).
Solid BuildCeilingAboveRestingParticle() {
  const MoveParticleCase base = BuildShorterStepCase("PaddleThroughBottomSideRestingOnIt");
  const double ceiling_height = base.inputs.start_position.z + (3 * base.inputs.contact_tolerance);  // m
  return ToSolid(
      FlatSolid{
          .point_on_surface = {.x = 0, .y = 0, .z = ceiling_height},
          .normal_out_of_solid = -move_particle_cases::kUnitZ,
      },
      SolidMotion(),
      1
  );
}

// A fixed ball of radius 4 mm whose surface is 0.6 mm beyond the rest gap of the same particle.
Solid BuildBallNearRestingParticle() {
  const MoveParticleCase base = BuildShorterStepCase("PaddleThroughBottomSideRestingOnIt");
  const Vec3 start = base.inputs.start_position;
  const double radius = 0.004;  // m
  const double centre_distance = radius + base.inputs.contact_tolerance + 6e-4;  // m
  const double offset_y = 0.0045;  // m
  const double offset_z = std::sqrt((centre_distance * centre_distance) - (offset_y * offset_y));  // m
  return ToSolid(
      Sphere{.center = {.x = start.x, .y = start.y + offset_y, .z = start.z + offset_z}, .radius = radius},
      SolidMotion(),
      1
  );
}

// Fails the calling test unless the scene fails with kTurningSolidNearOtherSolid naming the paddle (id 2) and
// can_shorter_step_cure false at T and at T / 2^k for each listed k.
void CheckNoCureAtHalvings(const MoveParticleCase& scene, std::initializer_list<int> halving_counts) {
  for (const int halving_count : halving_counts) {
    SCOPED_TRACE("step divided by 2^" + std::to_string(halving_count));
    const MoveParticleCall call = ComputeMoveWithHalvings(scene, halving_count);
    EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
    EXPECT_EQ(call.failure.close_call, ContactCloseCall::kTurningSolidNearOtherSolid);
    EXPECT_EQ(call.failure.solid_id, 2U);
    EXPECT_FALSE(call.failure.can_shorter_step_cure);
  }
}

TEST(MoveParticle, PaddleSceneWithCeilingFirst_NoCureWhateverTheOrder) {
  // The ceiling, beyond the window, fails the closing bound first in the array; the bottom side, within it, fails at
  // every step length, so the flag is false. Taken from the first failing solid it said true down to T / 2^25.
  CheckNoCureAtHalvings(
      BuildPaddleSceneWithExtraSolid(BuildCeilingAboveRestingParticle(), ExtraSolidPlace::kFirst),
      {0, 1, 2, 4, 8, 16, 25, 26}
  );
}

TEST(MoveParticle, PaddleSceneWithCeilingLast_NoCure) {
  CheckNoCureAtHalvings(
      BuildPaddleSceneWithExtraSolid(BuildCeilingAboveRestingParticle(), ExtraSolidPlace::kLast),
      {0, 1, 2, 4, 8, 16, 25, 26}
  );
}

TEST(MoveParticle, PaddleSceneWithBallFirst_NoCureWhateverTheOrder) {
  // The ball 0.6 mm away fails the closing bound at T only; taken from the first failing solid the flag said true at
  // T, false from T / 2.
  CheckNoCureAtHalvings(
      BuildPaddleSceneWithExtraSolid(BuildBallNearRestingParticle(), ExtraSolidPlace::kFirst),
      {0, 1, 2, 4}
  );
}

TEST(MoveParticle, PaddleSceneWithBallLast_NoCure) {
  CheckNoCureAtHalvings(
      BuildPaddleSceneWithExtraSolid(BuildBallNearRestingParticle(), ExtraSolidPlace::kLast),
      {0, 1, 2, 4}
  );
}

// --- Close calls that a shorter step cures: a hit after the start, or a test that reads the time left ---

TEST(MoveParticle, PaddleThroughBottomSideOneMillimetreAbove_CuredByOneHalving) {
  // The bottom side 1 mm below the particle is beyond the window: the closing bound of 1 ms, about 0.4 mm/ms times the
  // time left plus the fall, reaches it; that of 0.5 ms does not.
  EXPECT_EQ(
      ComputeCountHalvingsToSuccess(
          BuildShorterStepCase("PaddleThroughBottomSideOneMillimetreAbove"),
          ContactCloseCall::kTurningSolidNearOtherSolid,
          8
      ),
      1
  );
}

TEST(MoveParticle, DrivenBoxesNearCrease_CuredByThreeHalvings) {
  // The boxes reach the particle 2 ms into a step of 10 ms and one response pushes both; a step of 1.25 ms ends before.
  EXPECT_EQ(
      ComputeCountHalvingsToSuccess(
          BuildShorterStepCase("DrivenBoxesNearCrease"),
          ContactCloseCall::kSecondReactingSolid,
          8
      ),
      3
  );
}

TEST(MoveParticle, DescendingBoxClosingOnBottomSide_CuredByTwoHalvings) {
  // The box reaches the particle 0.4 ms into the step and squeezes it there; a step of 0.25 ms ends before. The squeeze
  // comes back at the start of a later step, once the box is at the particle (DescendingBoxAtBottomSide).
  EXPECT_EQ(
      ComputeCountHalvingsToSuccess(
          BuildShorterStepCase("DescendingBoxClosingOnBottomSide"),
          ContactCloseCall::kSqueezedBetweenSolids,
          8
      ),
      2
  );
}

TEST(MoveParticle, SeventeenFloorsBelowFalling_CuredByFourHalvings) {
  // The particle reaches the 17 floors 10 ms into a step of 100 ms; a step of 6.25 ms ends before.
  EXPECT_EQ(
      ComputeCountHalvingsToSuccess(
          BuildShorterStepCase("SeventeenFloorsBelowFalling"),
          ContactCloseCall::kTooManyHits,
          8
      ),
      4
  );
}

TEST(MoveParticle, OctagonRoomFifthHit_CuredByTwoHalvings) {
  EXPECT_EQ(ComputeCountHalvingsToSuccess(FindMoveParticleCase("OctagonRoom"), ContactCloseCall::kTooManyHits, 8), 2);
}

TEST(MoveParticle, EscapeTooFastForTimeLeft_CuredByOneHalving) {
  // The squeeze of the rounding over the time left, with two active surfaces.
  EXPECT_EQ(
      ComputeCountHalvingsToSuccess(
          FindMoveParticleCase("FloorRisingFastUnderSixtyDegreeWall"),
          ContactCloseCall::kSqueezedBetweenSolids,
          8
      ),
      1
  );
}

TEST(MoveParticle, HoverNearTurningBox_CuredByOneHalving) {
  EXPECT_EQ(
      ComputeCountHalvingsToSuccess(
          FindMoveParticleCase("TurningBoxHoverOnSpinAxis"),
          ContactCloseCall::kHoverNearTurningSolid,
          8
      ),
      1
  );
}

TEST(MoveParticle, FallingAboveSpeedBound_CuredByOneHalving) {
  // A particle 2 tol above a fixed floor falling at 1.01 times the speed whose response bound reaches tol / 64 over the
  // step, 2^40 tol / T: the bound doubles when the step halves.
  const double bound_speed = 0x1p40 * kTestContactTolerance / 0.1;  // m/s
  const MoveParticleCase scene = {
      .name = "FallingAboveSpeedBound",
      .solids = {move_particle_cases::BuildFloor(1)},
      .inputs = move_particle_cases::BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 2 * kTestContactTolerance},
          .velocity = {.x = 0, .y = 0, .z = -1.01 * bound_speed},
          .step_duration = 0.1,
          .friction = 0,
      }),
  };
  EXPECT_EQ(ComputeCountHalvingsToSuccess(scene, ContactCloseCall::kMovesTooFastForStep, 8), 1);
}

TEST(CheckEndOutsideSolids, EndTooClose_ShorterStepCures) {
  // The end check fails only when a solid closed on the particle faster than the hits and the settle kept it at tol;
  // a shorter step gives them the chance.
  const std::array solids = {move_particle_cases::BuildFloor(1)};
  SolidContactInputs inputs;
  inputs.solids = solids.data();
  inputs.solid_count = 1;
  inputs.step_duration = 0.1;
  inputs.contact_tolerance = kTestContactTolerance;
  ContactFailure failure;
  const Vec3 end = {.x = 0, .y = 0, .z = kTestContactTolerance / 4};
  EXPECT_EQ(CheckEndOutsideSolids(inputs, end, &failure), ContactError::kTooCloseToCall);
  EXPECT_EQ(failure.close_call, ContactCloseCall::kEndsTooCloseToSolid);
  EXPECT_TRUE(failure.can_shorter_step_cure);
}

// --- Every failure of the catalogue ---

}  // namespace
}  // namespace fluidity::contact
