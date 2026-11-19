#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <set>
#include <string>

#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_test_cases.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {
namespace {

// What one MoveParticle call returned.
struct MoveParticleCall {
  ContactError error = ContactError::kNone;
  SolidContactResult result{};
  ContactFailure failure{};
};

// Runs MoveParticle on a scene with its step divided by 2^halving_count, every other input kept, the contact tolerance
// included.
MoveParticleCall RunMoveWithHalvings(const MoveParticleCase& scene, int halving_count) {
  SolidContactInputs inputs = ToSolidContactInputs(scene);
  inputs.step_duration = std::ldexp(inputs.step_duration, -halving_count);
  MoveParticleCall call;
  call.error = MoveParticle(inputs, &call.result, &call.failure);
  return call;
}

// Fails the calling test unless the scene fails with kTooCloseToCall, that close call and that solid, and
// can_shorter_step_cure false, at its step T and at T / 2, T / 4 and T / 16.
void ExpectSameCloseCallAtShorterSteps(
    const MoveParticleCase& scene,
    ContactCloseCall close_call,
    std::uint32_t solid_id
) {
  for (const int halving_count : {0, 1, 2, 4}) {
    SCOPED_TRACE("step divided by 2^" + std::to_string(halving_count));
    const MoveParticleCall call = RunMoveWithHalvings(scene, halving_count);
    EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
    EXPECT_EQ(call.failure.close_call, close_call);
    EXPECT_EQ(call.failure.solid_id, solid_id);
    EXPECT_FALSE(call.failure.can_shorter_step_cure);
  }
}

// Halves the scene's step until MoveParticle succeeds, at most max_halving_count times, and returns how many halvings
// that took, or max_halving_count + 1 when none succeeded. Fails the calling test unless every failure on the way is
// kTooCloseToCall with that close call and can_shorter_step_cure true.
int CountHalvingsToSuccess(const MoveParticleCase& scene, ContactCloseCall close_call, int max_halving_count) {
  for (int halving_count = 0; halving_count <= max_halving_count; ++halving_count) {
    const MoveParticleCall call = RunMoveWithHalvings(scene, halving_count);
    if (call.error == ContactError::kNone) {
      return halving_count;
    }
    EXPECT_EQ(call.error, ContactError::kTooCloseToCall) << "after " << halving_count << " halvings";
    EXPECT_EQ(call.failure.close_call, close_call) << "after " << halving_count << " halvings";
    EXPECT_TRUE(call.failure.can_shorter_step_cure) << "after " << halving_count << " halvings";
  }
  return max_halving_count + 1;
}

// --- Close calls that no shorter step cures: decided at the start of the step ---

TEST(MoveParticle, PaddleThroughBottomSideRestingOnIt_SameCloseCallAtEveryShorterStep) {
  // A paddle turning through the bottom side of the domain box, the particle resting on that side against the paddle:
  // hit at 0 on the paddle, the side within tol + tol / 16 at the carry's start, so the carry is not safe at any step
  // length.
  ExpectSameCloseCallAtShorterSteps(
      BuildShorterStepCase("PaddleThroughBottomSideRestingOnIt"),
      ContactCloseCall::kTurningSolidNearOtherSolid,
      2
  );
}

TEST(MoveParticle, PaddleOnShaftInCrease_SameCloseCallAtEveryShorterStep) {
  // The crease of a paddle and the shaft it turns with, the particle at the rest gap of both: the paddle (id 3) is hit
  // at 0 and the shaft lies within the window.
  ExpectSameCloseCallAtShorterSteps(
      BuildShorterStepCase("PaddleOnShaftInCrease"),
      ContactCloseCall::kTurningSolidNearOtherSolid,
      3
  );
}

TEST(MoveParticle, DrivenBoxesInCrease_SameCloseCallAtEveryShorterStep) {
  // Two driven boxes with reports_reaction, the particle in their crease: the start's one response pushes both, naming
  // the higher index, id 3.
  ExpectSameCloseCallAtShorterSteps(
      BuildShorterStepCase("DrivenBoxesInCrease"),
      ContactCloseCall::kSecondReactingSolid,
      3
  );
}

TEST(MoveParticle, DescendingBoxAtBottomSide_SameSqueezeAtEveryShorterStep) {
  // A box 2 tol above the bottom side coming down on a particle at rest between them: the stick to the box still
  // approaches the side, a squeeze naming the domain box (id 1), the solid the failing velocity approaches fastest.
  ExpectSameCloseCallAtShorterSteps(
      BuildShorterStepCase("DescendingBoxAtBottomSide"),
      ContactCloseCall::kSqueezedBetweenSolids,
      1
  );
}

TEST(MoveParticle, SeventeenFloorsAtStart_TooManyHitsAtEveryShorterStep) {
  // 17 active surfaces at the start, one more than one response reads (kMaxTrackedSurfaces), naming the record's solid.
  ExpectSameCloseCallAtShorterSteps(BuildShorterStepCase("SeventeenFloorsAtStart"), ContactCloseCall::kTooManyHits, 1);
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
void ExpectNoCureAtHalvings(const MoveParticleCase& scene, std::initializer_list<int> halving_counts) {
  for (const int halving_count : halving_counts) {
    SCOPED_TRACE("step divided by 2^" + std::to_string(halving_count));
    const MoveParticleCall call = RunMoveWithHalvings(scene, halving_count);
    EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
    EXPECT_EQ(call.failure.close_call, ContactCloseCall::kTurningSolidNearOtherSolid);
    EXPECT_EQ(call.failure.solid_id, 2U);
    EXPECT_FALSE(call.failure.can_shorter_step_cure);
  }
}

TEST(MoveParticle, PaddleSceneWithCeilingFirst_NoCureWhateverTheOrder) {
  // The ceiling, beyond the window, fails the closing bound first in the array; the bottom side, within it, fails at
  // every step length, so the flag is false. Taken from the first failing solid it said true down to T / 2^25.
  ExpectNoCureAtHalvings(
      BuildPaddleSceneWithExtraSolid(BuildCeilingAboveRestingParticle(), ExtraSolidPlace::kFirst),
      {0, 1, 2, 4, 8, 16, 25, 26}
  );
}

TEST(MoveParticle, PaddleSceneWithCeilingLast_NoCure) {
  ExpectNoCureAtHalvings(
      BuildPaddleSceneWithExtraSolid(BuildCeilingAboveRestingParticle(), ExtraSolidPlace::kLast),
      {0, 1, 2, 4, 8, 16, 25, 26}
  );
}

TEST(MoveParticle, PaddleSceneWithBallFirst_NoCureWhateverTheOrder) {
  // The ball 0.6 mm away fails the closing bound at T only; taken from the first failing solid the flag said true at
  // T, false from T / 2.
  ExpectNoCureAtHalvings(
      BuildPaddleSceneWithExtraSolid(BuildBallNearRestingParticle(), ExtraSolidPlace::kFirst),
      {0, 1, 2, 4}
  );
}

TEST(MoveParticle, PaddleSceneWithBallLast_NoCure) {
  ExpectNoCureAtHalvings(
      BuildPaddleSceneWithExtraSolid(BuildBallNearRestingParticle(), ExtraSolidPlace::kLast),
      {0, 1, 2, 4}
  );
}

// --- Close calls that a shorter step cures: a hit after the start, or a test that reads the time left ---

TEST(MoveParticle, PaddleThroughBottomSideOneMillimetreAbove_CuredByOneHalving) {
  // The bottom side 1 mm below the particle is beyond the window: the closing bound of 1 ms, about 0.4 mm/ms times the
  // time left plus the fall, reaches it; that of 0.5 ms does not.
  EXPECT_EQ(
      CountHalvingsToSuccess(
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
      CountHalvingsToSuccess(BuildShorterStepCase("DrivenBoxesNearCrease"), ContactCloseCall::kSecondReactingSolid, 8),
      3
  );
}

TEST(MoveParticle, DescendingBoxClosingOnBottomSide_CuredByTwoHalvings) {
  // The box reaches the particle 0.4 ms into the step and squeezes it there; a step of 0.25 ms ends before. The squeeze
  // comes back at the start of a later step, once the box is at the particle (DescendingBoxAtBottomSide).
  EXPECT_EQ(
      CountHalvingsToSuccess(
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
      CountHalvingsToSuccess(BuildShorterStepCase("SeventeenFloorsBelowFalling"), ContactCloseCall::kTooManyHits, 8),
      4
  );
}

TEST(MoveParticle, OctagonRoomFifthHit_CuredByTwoHalvings) {
  EXPECT_EQ(CountHalvingsToSuccess(FindMoveParticleCase("OctagonRoom"), ContactCloseCall::kTooManyHits, 8), 2);
}

TEST(MoveParticle, EscapeTooFastForTimeLeft_CuredByOneHalving) {
  // The squeeze of the rounding over the time left, with two active surfaces.
  EXPECT_EQ(
      CountHalvingsToSuccess(
          FindMoveParticleCase("FloorRisingFastUnderSixtyDegreeWall"),
          ContactCloseCall::kSqueezedBetweenSolids,
          8
      ),
      1
  );
}

TEST(MoveParticle, HoverNearTurningBox_CuredByOneHalving) {
  EXPECT_EQ(
      CountHalvingsToSuccess(
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
  EXPECT_EQ(CountHalvingsToSuccess(scene, ContactCloseCall::kMovesTooFastForStep, 8), 1);
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

// Fails the calling test unless a failure with can_shorter_step_cure false comes back at T / 2, T / 4 and T / 16:
// the same error, close call and solid, still false.
void ExpectLastingFailure(const MoveParticleCase& scene, const ContactFailure& failure, ContactError error) {
  for (const int halving_count : {1, 2, 4}) {
    SCOPED_TRACE("step divided by 2^" + std::to_string(halving_count));
    const MoveParticleCall shorter = RunMoveWithHalvings(scene, halving_count);
    EXPECT_EQ(shorter.error, error);
    EXPECT_EQ(shorter.failure.close_call, failure.close_call);
    EXPECT_EQ(shorter.failure.solid_id, failure.solid_id);
    EXPECT_FALSE(shorter.failure.can_shorter_step_cure);
  }
}

// What the failures of the catalogue gave.
struct CatalogueFailureCounts {
  int lasting = 0;  // can_shorter_step_cure false.
  int cured = 0;  // can_shorter_step_cure true.
  int most_halvings = 0;  // The most halvings a cured failure needed.
  std::set<ContactError> errors;  // Every error met.
};

// The most halvings a failure of the catalogue with can_shorter_step_cure true needs: GrazingWedgeClosingFast's hit
// comes 2^-26 of 0.1 s into the step, and every longer step meets it.
constexpr int kMostCatalogueHalvings = 26;

// Checks one scene of the catalogue and counts it: a failure with can_shorter_step_cure false comes back at shorter
// steps (ExpectLastingFailure); one with it true is a close call cured within kMostCatalogueHalvings halvings, every
// failure on the way true.
void CheckCatalogueScene(const MoveParticleCase& scene, CatalogueFailureCounts& counts) {
  const MoveParticleCall call = RunMoveWithHalvings(scene, 0);
  if (call.error == ContactError::kNone) {
    return;
  }
  counts.errors.insert(call.error);
  if (!call.failure.can_shorter_step_cure) {
    ++counts.lasting;
    ExpectLastingFailure(scene, call.failure, call.error);
    return;
  }
  ++counts.cured;
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  const int halving_count = CountHalvingsToSuccess(scene, call.failure.close_call, kMostCatalogueHalvings);
  EXPECT_LE(halving_count, kMostCatalogueHalvings);
  counts.most_halvings = std::max(counts.most_halvings, halving_count);
}

TEST(MoveParticle, EveryCatalogueFailure_ShorterStepCuresExactlyWhenFlagSays) {
  // Every scene of BuildMoveParticleCases that fails (CheckCatalogueScene). Every error other than kTooCloseToCall is
  // false, and the four errors of a scene are each met.
  CatalogueFailureCounts counts;
  for (const MoveParticleCase& scene : BuildMoveParticleCases()) {
    SCOPED_TRACE(std::string(scene.name));
    CheckCatalogueScene(scene, counts);
  }
  std::cout << "catalogue failures: lasting " << counts.lasting << ", cured " << counts.cured << " (most halvings "
            << counts.most_halvings << ")\n";
  EXPECT_GT(counts.lasting, 0);
  EXPECT_GT(counts.cured, 0);
  EXPECT_EQ(counts.most_halvings, kMostCatalogueHalvings);
  const std::set<ContactError> scene_errors = {
      ContactError::kBadInput,
      ContactError::kStartsInsideSolid,
      ContactError::kOnWrongSideOfPlate,
      ContactError::kStartsInNarrowGap,
  };
  EXPECT_TRUE(std::ranges::includes(counts.errors, scene_errors));
}

}  // namespace
}  // namespace fluidity::contact
