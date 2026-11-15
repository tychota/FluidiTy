#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <map>
#include <numbers>
#include <string>
#include <string_view>

#include "fluidity/base/mat3.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_test_cases.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {
namespace {

constexpr double kTolerance = kTestContactTolerance;  // m, 2^-30.
constexpr double kPositionTolerance = 2 * kTolerance;  // m, contact design 7.

// What one MoveParticle call returned: the error, the result and the failure.
struct MoveParticleCall {
  ContactError error = ContactError::kNone;
  SolidContactResult result{};
  ContactFailure failure{};
};

// A result whose every field differs from what MoveParticle writes, to show that a failed call leaves it unchanged.
SolidContactResult BuildUntouchedResult() {
  SolidContactResult result;
  result.position = {.x = -7, .y = -7, .z = -7};
  result.velocity = {.for_next_step = {.x = -8, .y = -8, .z = -8}, .for_this_step = {.x = -9, .y = -9, .z = -9}};
  result.kind = ContactKind::kStartedInsideSolid;
  result.hit_fraction = -1;
  result.normal_out_of_solid = {.x = -2, .y = -2, .z = -2};
  result.approach_speed = {.for_next_step = -3, .for_this_step = -4};
  result.solid_index = 77;
  result.face = SolidFace::kLowRim;
  result.side_at_hit = SolidSide::kAgainstNormal;
  result.hit_count = 99;
  result.reaction = {
      .solid_index = 66,
      .hit_position = {.x = -5, .y = -5, .z = -5},
      .velocity_for_next_step_change = {.x = -6, .y = -6, .z = -6},
  };
  return result;
}

// Runs MoveParticle on one scene, starting from the untouched result and a failure that is not the default.
MoveParticleCall RunMove(const MoveParticleCase& test_case) {
  MoveParticleCall call;
  call.result = BuildUntouchedResult();
  call.failure = {
      .error = ContactError::kRunOnCallFailed,
      .close_call = ContactCloseCall::kSolidTurnsTooFar,
      .particle_id = 5,
      .solid_id = 5,
      .magnitude = 5,
  };
  call.error = MoveParticle(ToSolidContactInputs(test_case), &call.result, &call.failure);
  return call;
}

// Runs MoveParticle on the scene of that name.
MoveParticleCall RunMove(std::string_view case_name) {
  return RunMove(FindMoveParticleCase(case_name));
}

// Fails the calling test unless |actual - expected| <= 2^-44 * max(|actual|, |expected|, reference): the relative
// tolerance of the contact tests (contact design 7), with reference the largest input speed of the test (1 for
// fractions).
void ExpectNearRelative(double actual, double expected, double reference) {
  const double scale = std::max({std::abs(actual), std::abs(expected), reference});
  EXPECT_LE(std::abs(actual - expected), 0x1p-44 * scale) << actual << " vs " << expected;
}

// ExpectNearRelative for each axis.
void ExpectVec3NearRelative(Vec3 actual, Vec3 expected, double reference) {
  ExpectNearRelative(actual.x, expected.x, reference);
  ExpectNearRelative(actual.y, expected.y, reference);
  ExpectNearRelative(actual.z, expected.z, reference);
}

// Fails the calling test unless the two vectors are equal axis by axis, bit for bit up to the sign of zero.
void ExpectVec3Equal(Vec3 actual, Vec3 expected) {
  EXPECT_EQ(actual.x, expected.x) << "axis x";
  EXPECT_EQ(actual.y, expected.y) << "axis y";
  EXPECT_EQ(actual.z, expected.z) << "axis z";
}

// Fails the calling test unless the position, the velocities and the reaction are the untouched ones.
void ExpectUntouchedMotion(const SolidContactResult& result) {
  const SolidContactResult untouched = BuildUntouchedResult();
  ExpectVec3Equal(result.position, untouched.position);
  ExpectVec3Equal(result.velocity.for_next_step, untouched.velocity.for_next_step);
  ExpectVec3Equal(result.velocity.for_this_step, untouched.velocity.for_this_step);
  EXPECT_EQ(result.reaction.solid_index, untouched.reaction.solid_index);
  ExpectVec3Equal(result.reaction.hit_position, untouched.reaction.hit_position);
  ExpectVec3Equal(result.reaction.velocity_for_next_step_change, untouched.reaction.velocity_for_next_step_change);
}

// Fails the calling test unless the kind, the time, the normal and the approach speeds are the untouched ones.
void ExpectUntouchedHitMotion(const SolidContactResult& result) {
  const SolidContactResult untouched = BuildUntouchedResult();
  EXPECT_EQ(result.kind, untouched.kind);
  EXPECT_EQ(result.hit_fraction, untouched.hit_fraction);
  ExpectVec3Equal(result.normal_out_of_solid, untouched.normal_out_of_solid);
  EXPECT_EQ(result.approach_speed.for_next_step, untouched.approach_speed.for_next_step);
  EXPECT_EQ(result.approach_speed.for_this_step, untouched.approach_speed.for_this_step);
}

// Fails the calling test unless the solid, face, side and hit count are the untouched ones.
void ExpectUntouchedHitSolid(const SolidContactResult& result) {
  const SolidContactResult untouched = BuildUntouchedResult();
  EXPECT_EQ(result.solid_index, untouched.solid_index);
  EXPECT_EQ(result.face, untouched.face);
  EXPECT_EQ(result.side_at_hit, untouched.side_at_hit);
  EXPECT_EQ(result.hit_count, untouched.hit_count);
}

// Fails the calling test unless the result is the untouched one, field by field.
void ExpectUntouched(const SolidContactResult& result) {
  ExpectUntouchedMotion(result);
  ExpectUntouchedHitMotion(result);
  ExpectUntouchedHitSolid(result);
}

// Fails the calling test unless the failure is exactly the expected one.
void ExpectFailure(const ContactFailure& actual, ContactFailure expected) {
  EXPECT_EQ(actual.error, expected.error);
  EXPECT_EQ(actual.close_call, expected.close_call);
  EXPECT_EQ(actual.particle_id, expected.particle_id);
  EXPECT_EQ(actual.solid_id, expected.solid_id);
  EXPECT_EQ(actual.magnitude, expected.magnitude);
}

// Fails the calling test unless the call failed with kBadInput, said nothing else, and left the result unchanged.
void ExpectBadInput(std::string_view case_name) {
  const MoveParticleCall call = RunMove(case_name);
  EXPECT_EQ(call.error, ContactError::kBadInput);
  ExpectFailure(call.failure, {.error = ContactError::kBadInput});
  ExpectUntouched(call.result);
}

// Fails the calling test unless the reaction is empty: no reacting solid was hit.
void ExpectNoReaction(const SolidReaction& reaction) {
  EXPECT_EQ(reaction.solid_index, kNoSolid);
  ExpectVec3Equal(reaction.hit_position, {});
  ExpectVec3Equal(reaction.velocity_for_next_step_change, {});
}

// --- The examples of the method paper and the docstring ---

TEST(MoveParticle, NoSolid_EndIsStartPlusVelocityTimesStep) {
  const MoveParticleCase test_case = FindMoveParticleCase("NoSolid");
  const SolidContactInputs& inputs = test_case.inputs;
  const MoveParticleCall call = RunMove(test_case);
  ASSERT_EQ(call.error, ContactError::kNone);
  const Vec3 start = inputs.start_position;
  const Vec3 velocity = inputs.velocity.for_this_step;
  const double step_duration = inputs.step_duration;
  EXPECT_EQ(call.result.position.x, start.x + velocity.x * step_duration);
  EXPECT_EQ(call.result.position.y, start.y + velocity.y * step_duration);
  EXPECT_EQ(call.result.position.z, start.z + velocity.z * step_duration);
  ExpectVec3Equal(call.result.velocity.for_next_step, inputs.velocity.for_next_step);
  ExpectVec3Equal(call.result.velocity.for_this_step, inputs.velocity.for_this_step);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_fraction, 0);
  EXPECT_EQ(call.result.hit_count, 0);
  EXPECT_EQ(call.result.solid_index, kNoSolid);
  ExpectNoReaction(call.result.reaction);
}

TEST(MoveParticle, DocstringExample_StopsOneToleranceShort) {
  const MoveParticleCall call = RunMove("DocstringExample");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  ExpectNearRelative(result.hit_fraction, (0.1 - kTolerance) / 0.2, 1);
  ExpectVec3Near(result.position, {.x = 1 - kTolerance, .y = 0, .z = 0}, kPositionTolerance);
  ExpectVec3NearRelative(result.velocity.for_next_step, {}, 1);
  ExpectVec3NearRelative(result.velocity.for_this_step, {}, 1);
  ExpectNearRelative(result.approach_speed.for_next_step, -1, 1);
  ExpectNearRelative(result.approach_speed.for_this_step, -1, 1);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  ExpectVec3Equal(result.normal_out_of_solid, {.x = -1, .y = 0, .z = 0});
  EXPECT_EQ(result.solid_index, 0U);
  EXPECT_EQ(result.side_at_hit, SolidSide::kNoPlate);
  EXPECT_EQ(result.hit_count, 1);
  ExpectNoReaction(result.reaction);
}

TEST(MoveParticle, SecondSolidHitFirst_RecordNamesItsIndex) {
  // The record names the solid by its index in the array, 1 here, not by its id (2).
  const MoveParticleCall call = RunMove("SecondSolidHitFirst");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3Near(call.result.position, {.x = 1 - kTolerance, .y = 0, .z = 0.5}, kPositionTolerance);
}

TEST(MoveParticle, Success_WritesDefaultFailure) {
  const MoveParticleCall call = RunMove("DocstringExample");
  ASSERT_EQ(call.error, ContactError::kNone);
  ExpectFailure(call.failure, ContactFailure{});
}

TEST(MoveParticle, NullFailure_MovesAsWithFailure) {
  const MoveParticleCase moving = FindMoveParticleCase("DocstringExample");
  SolidContactResult result;
  EXPECT_EQ(MoveParticle(ToSolidContactInputs(moving), &result, nullptr), ContactError::kNone);
  ExpectVec3Near(result.position, {.x = 1 - kTolerance, .y = 0, .z = 0}, kPositionTolerance);
}

TEST(MoveParticle, NullFailure_FailsAsWithFailure) {
  const MoveParticleCase failing = FindMoveParticleCase("TwoReactingSolids");
  SolidContactResult result = BuildUntouchedResult();
  EXPECT_EQ(MoveParticle(ToSolidContactInputs(failing), &result, nullptr), ContactError::kTooCloseToCall);
  ExpectUntouched(result);
}

TEST(MoveParticle, HitAtEndOfStep_EndsAtTheHit) {
  // The step ends exactly at the hit time of the docstring example: the hit counts, and no time is left to sweep.
  MoveParticleCase test_case = FindMoveParticleCase("DocstringExample");
  test_case.inputs.step_duration = (1 - 0.9) - kTolerance;  // s, FindFirstHit's hit time, bit for bit.
  const MoveParticleCall call = RunMove(test_case);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_fraction, 1);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3Near(call.result.position, {.x = 1 - kTolerance, .y = 0, .z = 0}, kPositionTolerance);
  ExpectVec3NearRelative(call.result.velocity.for_this_step, {}, 1);
}

TEST(MoveParticle, PaperFigureRisingFloor_MatchesFigure) {
  // Method paper <fig-sweep-hit-slide>: hit at 0.6 - tol / 2 s; for_this_step slips to {2, 0, 0.5} m/s and
  // for_next_step to {0.75, 0, 0.5} m/s; the slide rides up with the floor to x = 2.3 m, tol above its end height of
  // 0.5 m. The reaction is the change of for_next_step, {-0.75, 0, 3} m/s, exact in binary.
  const MoveParticleCall call = RunMove("PaperFigureRisingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  constexpr double kLargestSpeed = 2.5;  // m/s
  ExpectNearRelative(result.hit_fraction, 0.6 - kTolerance / 2, 1);
  ExpectVec3NearRelative(result.velocity.for_this_step, {.x = 2, .y = 0, .z = 0.5}, kLargestSpeed);
  ExpectVec3NearRelative(result.velocity.for_next_step, {.x = 0.75, .y = 0, .z = 0.5}, kLargestSpeed);
  ExpectNearRelative(result.approach_speed.for_next_step, -3, kLargestSpeed);
  ExpectNearRelative(result.approach_speed.for_this_step, -2, kLargestSpeed);
  ExpectVec3Near(result.position, {.x = 2.3, .y = 0, .z = 0.5 + kTolerance}, kPositionTolerance);
  // The rising floor is touched: the slide moves with it, and rounding at the hit is not a second hit.
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.reaction.solid_index, 0U);
  ExpectVec3Equal(result.reaction.velocity_for_next_step_change, {.x = -0.75, .y = 0, .z = 3});
  ExpectVec3Near(result.reaction.hit_position, {.x = 1.5, .y = 0, .z = 0.3 + kTolerance}, kPositionTolerance);
}

TEST(MoveParticle, DifferentVelocities_EachGetsItsOwnApproach) {
  // for_next_step {1, 0, -3}: approach -3, sideways 1, and 0.5 >= 1 / 3 sticks at zero. for_this_step {0, 0, -1}: no
  // sideways speed, so it sticks too.
  const MoveParticleCall call = RunMove("DifferentVelocities");
  ASSERT_EQ(call.error, ContactError::kNone);
  ExpectVec3Equal(call.result.velocity.for_next_step, {});
  ExpectVec3Equal(call.result.velocity.for_this_step, {});
  EXPECT_EQ(call.result.approach_speed.for_next_step, -3);
  EXPECT_EQ(call.result.approach_speed.for_this_step, -1);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = kTolerance}, kPositionTolerance);
}

// --- Plates, moving and fixed ---

// Fails the calling test unless the scene of one plate row that hits ended as its closed form says. Plate row 0 meets
// the face toward -e, plate row 1 the face toward +e; the side hit is the one the particle is on, by the plate's
// normal.
void ExpectPlateRowHit(std::string_view case_name, int plate_row, Vec3 axis, bool is_normal_along_axis) {
  constexpr double kHalfThickness = 1.0 / 32;  // m
  constexpr double kLargestSpeed = 2;  // m/s
  const MoveParticleCall call = RunMove(case_name);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  const double side_sign = plate_row == 0 ? -1.0 : 1.0;
  const bool is_along_normal_side = (side_sign > 0) == is_normal_along_axis;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  ExpectNearRelative(result.hit_fraction, (1 - kHalfThickness - kTolerance) / 2, 1);
  ExpectVec3Near(result.normal_out_of_solid, side_sign * axis, 0);
  ExpectNearRelative(result.approach_speed.for_next_step, -2, kLargestSpeed);
  ExpectNearRelative(result.approach_speed.for_this_step, -2, kLargestSpeed);
  EXPECT_EQ(result.side_at_hit, is_along_normal_side ? SolidSide::kAlongNormal : SolidSide::kAgainstNormal);
  EXPECT_EQ(result.face, is_along_normal_side ? SolidFace::kHighZ : SolidFace::kLowZ);
  const double end_coordinate = plate_row == 0 ? -(kHalfThickness + kTolerance) : 1 + kHalfThickness + kTolerance;
  ExpectVec3Near(result.position, end_coordinate * axis, kPositionTolerance);
  const Vec3 end_velocity = plate_row == 0 ? Vec3{} : 2 * axis;
  ExpectVec3NearRelative(result.velocity.for_next_step, end_velocity, kLargestSpeed);
  ExpectVec3NearRelative(result.velocity.for_this_step, end_velocity, kLargestSpeed);
}

// Fails the calling test unless the scene of plate row 2 ended free, moved by e.
void ExpectPlateRowFree(std::string_view case_name, Vec3 axis) {
  const MoveParticleCall call = RunMove(case_name);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, axis, kPositionTolerance);
  ExpectVec3Equal(call.result.velocity.for_this_step, axis);
}

TEST(MoveParticle, PlateRowsOnEveryAxisAndNormal_MatchClosedForms) {
  // Plate half thickness 1/32 m, T = 1 s. Plate row 0: the particle at -e moving at 2e hits the fixed plate's face
  // toward -e at (1 - 1/32 - tol) / 2 s and stops at -(1/32 + tol) e. Plate row 1: the plate's centre, from -e at 2e,
  // reaches the particle at rest at 0 at the same time; the particle is carried at 2e to (1 + 1/32 + tol) e. Plate
  // row 2: the particle moves with the plate at e and is never hit.
  const std::array<Vec3, 3> axes = {
      Vec3{.x = 1, .y = 0, .z = 0},
      Vec3{.x = 0, .y = 1, .z = 0},
      Vec3{.x = 0, .y = 0, .z = 1},
  };
  const std::array<std::string, 3> axis_names = {"X", "Y", "Z"};
  for (int plate_row = 0; plate_row < 3; ++plate_row) {
    for (std::size_t axis_index = 0; axis_index < axes.size(); ++axis_index) {
      for (const bool is_normal_along_axis : {true, false}) {
        const std::string name = "PlateRow" + std::to_string(plate_row) + axis_names.at(axis_index)
            + (is_normal_along_axis ? "PlusNormal" : "MinusNormal");
        SCOPED_TRACE(name);
        if (plate_row == 2) {
          ExpectPlateRowFree(name, axes.at(axis_index));
        } else {
          ExpectPlateRowHit(name, plate_row, axes.at(axis_index), is_normal_along_axis);
        }
      }
    }
  }
}

// --- Several hits in one step ---

TEST(MoveParticle, FrictionOnSideBottom_LeavesSideways2Point5) {
  // Hit at t = 0.1 - tol; friction 0.5 takes 0.5 m/s of the 3 m/s sideways; the slide covers 2.5 (0.2 - t) m.
  const MoveParticleCall call = RunMove("FrictionOnSideBottom");
  ASSERT_EQ(call.error, ContactError::kNone);
  const double hit_elapsed = 0.1 - kTolerance;  // s
  ExpectVec3NearRelative(call.result.velocity.for_this_step, {.x = 2.5, .y = 0, .z = 0}, 3);
  ExpectVec3NearRelative(call.result.velocity.for_next_step, {.x = 2.5, .y = 0, .z = 0}, 3);
  ExpectVec3Near(
      call.result.position,
      {.x = 0.2 + 3 * hit_elapsed + 2.5 * (0.2 - hit_elapsed), .y = 0.5, .z = kTolerance},
      kPositionTolerance
  );
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  ExpectVec3Equal(call.result.normal_out_of_solid, {.x = 0, .y = 0, .z = 1});
  EXPECT_EQ(call.result.hit_count, 1);
}

TEST(MoveParticle, FrictionScaleTwo_DoublesFriction) {
  // Friction 0.5 times friction_scale 2 takes 1 m/s of the 3 m/s sideways.
  const MoveParticleCall call = RunMove("FrictionScaleTwo");
  ASSERT_EQ(call.error, ContactError::kNone);
  ExpectVec3NearRelative(call.result.velocity.for_this_step, {.x = 2, .y = 0, .z = 0}, 3);
  ExpectVec3NearRelative(call.result.velocity.for_next_step, {.x = 2, .y = 0, .z = 0}, 3);
  ExpectVec3Near(call.result.position, {.x = 2.5, .y = 0, .z = kTolerance}, kPositionTolerance);
}

TEST(MoveParticle, DomainCorner_ThreeSidesHitInOneStep) {
  // side_left at 0.1 - tol s, side_front 0.1 s later, side_bottom 0.1 s after that: each removes one axis.
  const MoveParticleCall call = RunMove("DomainCorner");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  ExpectVec3Near(result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kPositionTolerance);
  ExpectVec3NearRelative(result.velocity.for_next_step, {}, 1);
  ExpectVec3NearRelative(result.velocity.for_this_step, {}, 1);
  EXPECT_EQ(result.hit_count, 3);
  EXPECT_EQ(result.face, SolidFace::kLowX);
  ExpectNearRelative(result.hit_fraction, (0.1 - kTolerance) / 0.5, 1);
}

TEST(MoveParticle, OctagonRoom_FifthHitIsTooManyHits) {
  // The particle hits side 0 of the room, slides along sides 0, 1, 2 and 3, and meets side 4 (id 5) at 0.279 s.
  const MoveParticleCall call = RunMove("OctagonRoom");
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  ExpectFailure(
      call.failure,
      {.error = ContactError::kTooCloseToCall, .close_call = ContactCloseCall::kTooManyHits, .solid_id = 5}
  );
  ExpectUntouched(call.result);
}

TEST(MoveParticle, OctagonRoomShortStep_FifthHitBeforePoint3) {
  // The fifth hit comes before 0.3 s: a step of 0.3 s fails too, while a step of 0.27 s ends after four hits.
  MoveParticleCase test_case = FindMoveParticleCase("OctagonRoom");
  test_case.inputs.step_duration = 0.3;
  EXPECT_EQ(RunMove(test_case).error, ContactError::kTooCloseToCall);
  test_case.inputs.step_duration = 0.27;
  const MoveParticleCall four_hits = RunMove(test_case);
  ASSERT_EQ(four_hits.error, ContactError::kNone);
  EXPECT_EQ(four_hits.result.hit_count, 4);
  EXPECT_EQ(four_hits.result.solid_index, 0U);
}

TEST(MoveParticle, FloorThenRamp_SlidesUpTheRamp) {
  // The floor at 0.1 - tol s leaves {1, 0, 0}; the ramp with normal {-0.6, 0, 0.8} at 0.5 - tol / 3 s leaves
  // {0.64, 0, 0.48}, which leaves the floor. The settle must not pull the particle back to it.
  const MoveParticleCall call = RunMove("FloorThenRamp");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  EXPECT_EQ(call.result.solid_index, 0U);
  ExpectVec3NearRelative(call.result.velocity.for_this_step, {.x = 0.64, .y = 0, .z = 0.48}, 1);
  ExpectVec3Near(
      call.result.position,
      {.x = 1.32 - 0.12 * kTolerance, .y = 0, .z = 0.24 + 1.16 * kTolerance},
      kPositionTolerance
  );
}

TEST(MoveParticle, FloorThenMovingUpright_SecondSweepPosesItAtSweepStart) {
  // The second sweep starts at 0.1 - tol s, when the upright flat solid is at x = 0.95 + tol / 2: closing at 1.5 m/s
  // it hits at 2/3 - 2 tol / 3 s, and the particle then moves with it at -0.5 m/s to x = 0.5 - tol. Posed at time 0
  // it would be met late, at x = 0.7, inside it, and the settle would hide that in the end position: the reaction
  // keeps the hit's position.
  const MoveParticleCall call = RunMove("FloorThenMovingUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  ExpectVec3NearRelative(call.result.velocity.for_this_step, {.x = -0.5, .y = 0, .z = 0}, 1);
  ExpectVec3NearRelative(call.result.velocity.for_next_step, {.x = -0.5, .y = 0, .z = 0}, 1);
  ExpectVec3Near(call.result.position, {.x = 0.5 - kTolerance, .y = 0, .z = kTolerance}, kPositionTolerance);
  EXPECT_EQ(call.result.reaction.solid_index, 1U);
  ExpectVec3Equal(call.result.reaction.velocity_for_next_step_change, {.x = -1.5, .y = 0, .z = 0});
  ExpectVec3Near(
      call.result.reaction.hit_position,
      {.x = 2.0 / 3 - 2 * kTolerance / 3, .y = 0, .z = kTolerance},
      kPositionTolerance
  );
}

TEST(MoveParticle, TiltedUprightPushesBackToFloor_TouchedFloorIsHitAgain) {
  // The floor is hit at 0.1 - tol s, the tilted upright at about 0.5 s; its response leaves the particle approaching
  // the floor at 1.5 tol m/s, a real approach far above the floor's allowance of about 2^-46 m/s, so the touched floor
  // is hit again at once. That response leaves the upright approached by only about 2e-18 m/s, within its allowance,
  // so the third sweep is free and the particle rests at z = tol.
  const MoveParticleCall call = RunMove("TiltedUprightPushesBackToFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 3);
  EXPECT_NEAR(call.result.position.z, kTolerance, kTolerance / 16);
  EXPECT_NEAR(call.result.position.x, 1 - kTolerance, kPositionTolerance);
}

TEST(MoveParticle, TiltedUprightOverRisingFloor_MovingTouchedFloorIsHitAgain) {
  // As TiltedUprightPushesBackToFloor with the floor rising at 0.25 m/s: hit at (0.1 - tol) / 1.25 s, the upright at
  // about 0.5 s, then the moving touched floor again at once; the particle rides up with it to z = 0.25 + tol.
  const MoveParticleCall call = RunMove("TiltedUprightOverRisingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 3);
  EXPECT_NEAR(call.result.position.z, 0.25 + kTolerance, kTolerance / 16);
  EXPECT_NEAR(call.result.position.x, 1 - kTolerance, kPositionTolerance);
}

TEST(MoveParticle, ChasingSolidAtRightAngles_FirstSolidNotHitAgain) {
  // B's response leaves an approach to A of B's rounding; with only A's own allowance A was hit again at 0.
  const MoveParticleCall call = RunMove("ChasingSolidAtRightAngles");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
}

TEST(MoveParticle, NearStickSlipOnObliqueFlatSolid_HitsOnce) {
  // The slip leaves an approach of rounding of the incoming speed in a tiny returned velocity: within the allowance of
  // the incoming velocity, so not a second hit.
  const MoveParticleCall call = RunMove("NearStickSlipOnObliqueFlatSolid");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
}

TEST(MoveParticle, ObliqueFlatSolidRoundingAtHit_HitsOnce) {
  // The hit at 0.0625 - tol / 2.8 s leaves the particle 2.8e-18 m outside the rest gap, moving along the surface
  // at {-0.32, 0, 0.24} m/s with an approach of -2^-55 m/s: rounding, within the allowance, so not a second hit.
  const MoveParticleCall call = RunMove("ObliqueFlatSolidRoundingAtHit");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3NearRelative(call.result.velocity.for_this_step, {.x = -0.32, .y = 0, .z = 0.24}, 2);
}

TEST(MoveParticle, TurningBoxHitsRestingParticle_EndsInTooManyHits) {
  // A turning solid has no touching rule: after its hit at 0.8857 s the particle is hit again at 0 by every sweep,
  // until the fifth hit. The slide in the solid's axes is not part of MoveParticle yet.
  const MoveParticleCall call = RunMove("TurningBoxHitsRestingParticle");
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  ExpectFailure(
      call.failure,
      {.error = ContactError::kTooCloseToCall, .close_call = ContactCloseCall::kTooManyHits, .solid_id = 1}
  );
}

TEST(MoveParticle, SpinningSphereStick_TakesSurfaceVelocity) {
  // The gap 0.5 - tol closes at 1.5 m/s: the hit is at (0.5 - tol) / 1.5 s, at the top, 0.5 + tol above the centre.
  // The surface there moves at {0, 0, 0.5} + {2, 0, 0} x {0, 0, 0.5 + tol} = {0, -(1 + 2 tol), 0.5} m/s, which
  // friction 2 gives the particle (the arm is measured from the centre at the time of the hit). For the 2/3 + 2 tol / 3
  // s left it leaves along the tangent to {0, -2/3 - 2 tol, 1 + tol}.
  const MoveParticleCall call = RunMove("SpinningSphereStick");
  ASSERT_EQ(call.error, ContactError::kNone);
  const Vec3 surface_velocity = {.x = 0, .y = -(1 + 2 * kTolerance), .z = 0.5};
  ExpectVec3NearRelative(call.result.velocity.for_this_step, surface_velocity, 1);
  ExpectVec3NearRelative(call.result.velocity.for_next_step, surface_velocity, 1);
  ExpectNearRelative(call.result.hit_fraction, (0.5 - kTolerance) / 1.5, 1);
  EXPECT_EQ(call.result.face, SolidFace::kCurved);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3Near(
      call.result.position,
      {.x = 0, .y = -2.0 / 3 - 2 * kTolerance, .z = 1 + kTolerance},
      kPositionTolerance
  );
}

TEST(MoveParticle, StartOnSurfaceMovingAway_IsFree) {
  const MoveParticleCall call = RunMove("StartOnSurfaceMovingAway");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = 0.5}, kPositionTolerance);
}

TEST(MoveParticle, FollowingSinkingFloor_EndCheckReadsEndPose) {
  const MoveParticleCall call = RunMove("FollowingSinkingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  ExpectVec3Equal(call.result.position, {.x = 0, .y = 0, .z = -0.5});
}

// --- The reaction ---

TEST(MoveParticle, ReactingSolid_ReactionIsChangeOfNextStepVelocity) {
  const MoveParticleCase test_case = FindMoveParticleCase("ReactingSolid");
  const MoveParticleCall call = RunMove(test_case);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidReaction& reaction = call.result.reaction;
  EXPECT_EQ(reaction.solid_index, 0U);
  const Vec3 before = test_case.inputs.velocity.for_next_step;
  ExpectVec3Equal(reaction.velocity_for_next_step_change, call.result.velocity.for_next_step - before);
  ExpectVec3Near(reaction.hit_position, {.x = 1 - kTolerance, .y = 0, .z = 0}, kPositionTolerance);
}

TEST(MoveParticle, ReactingDomainCorner_ReactionSumsEveryChange) {
  // Three hits on one reacting solid: {1, 0, 0}, {0, 1, 0} and {0, 0, 1} m/s, each exact, at the first hit's position.
  const MoveParticleCall call = RunMove("ReactingDomainCorner");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidReaction& reaction = call.result.reaction;
  EXPECT_EQ(reaction.solid_index, 0U);
  ExpectVec3Equal(reaction.velocity_for_next_step_change, {.x = 1, .y = 1, .z = 1});
  ExpectVec3Near(
      reaction.hit_position,
      {.x = kTolerance, .y = 0.1 + kTolerance, .z = 0.2 + kTolerance},
      kPositionTolerance
  );
}

TEST(MoveParticle, ReactingUprightAfterFloor_ReactionCountsOnlyTheReactingSolid) {
  // The floor (index 0) does not react; the upright flat solid (index 1) does: only its change {-2, 0, 0}, at its hit.
  const MoveParticleCall call = RunMove("ReactingUprightAfterFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.solid_index, 0U);
  const SolidReaction& reaction = call.result.reaction;
  EXPECT_EQ(reaction.solid_index, 1U);
  ExpectVec3Equal(reaction.velocity_for_next_step_change, {.x = -2, .y = 0, .z = 0});
  ExpectVec3Near(reaction.hit_position, {.x = 1 - kTolerance, .y = 0, .z = kTolerance}, kPositionTolerance);
}

TEST(MoveParticle, TwoReactingSolids_IsSecondReactingSolid) {
  // The floor (id 1) is hit first, then the upright flat solid (id 2): the failure names the second reacting solid.
  const MoveParticleCall call = RunMove("TwoReactingSolids");
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  ExpectFailure(
      call.failure,
      {.error = ContactError::kTooCloseToCall, .close_call = ContactCloseCall::kSecondReactingSolid, .solid_id = 2}
  );
  ExpectUntouched(call.result);
}

TEST(MoveParticle, SecondReactingSolidWithZeroChange_DoesNotConflict) {
  // for_next_step {-1, 0, -2} approaches the reacting floor, which changes it by {0, 0, 2}, and then moves away from
  // the reacting upright, whose response changes it by exactly zero: no second reacting solid.
  const MoveParticleCall call = RunMove("SecondReactingSolidWithZeroChange");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  EXPECT_EQ(call.result.reaction.solid_index, 0U);
  ExpectVec3Equal(call.result.reaction.velocity_for_next_step_change, {.x = 0, .y = 0, .z = 2});
}

TEST(MoveParticle, ReactingSolidWithZeroChange_LeavesReactionEmpty) {
  const MoveParticleCall call = RunMove("ReactingSolidWithZeroChange");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectNoReaction(call.result.reaction);
}

// --- Failures ---

TEST(MoveParticle, NonFiniteStart_IsBadInput) {
  ExpectBadInput("NonFiniteStart");
}

TEST(MoveParticle, ZeroStepDuration_IsBadInput) {
  ExpectBadInput("ZeroStepDuration");
}

TEST(MoveParticle, NegativeFriction_IsBadInput) {
  ExpectBadInput("NegativeFriction");
}

TEST(MoveParticle, ZeroContactTolerance_IsBadInput) {
  ExpectBadInput("ZeroContactTolerance");
}

TEST(MoveParticle, NonFiniteVelocityForNextStep_IsBadInput) {
  ExpectBadInput("NonFiniteVelocityForNextStep");
}

TEST(MoveParticle, NegativeOverlapTolerance_IsBadInput) {
  ExpectBadInput("NegativeOverlapTolerance");
}

TEST(MoveParticle, SolidsWithEqualIds_IsBadInput) {
  ExpectBadInput("SolidsWithEqualIds");
}

TEST(MoveParticle, PlateSideWithoutPlateId_IsBadInput) {
  ExpectBadInput("PlateSideWithoutPlateId");
}

TEST(MoveParticle, PlateSideWithoutSide_IsBadInput) {
  ExpectBadInput("PlateSideWithoutSide");
}

TEST(MoveParticle, SolidsOutOfIdOrder_IsBadInput) {
  ExpectBadInput("SolidsOutOfIdOrder");
}

TEST(MoveParticle, PlateSideNamingNoSolid_IsBadInput) {
  ExpectBadInput("PlateSideNamingNoSolid");
}

TEST(MoveParticle, PlateSideNamingFlatSolid_IsBadInput) {
  ExpectBadInput("PlateSideNamingFlatSolid");
}

TEST(MoveParticle, StartsInsideSecondSolid_FailureNamesItsId) {
  // Inside the floor (index 1, id 9) by 0.01 m: the failure names the id, not the index, with the depth.
  const MoveParticleCall call = RunMove("StartsInsideSecondSolid");
  EXPECT_EQ(call.error, ContactError::kStartsInsideSolid);
  ExpectFailure(call.failure, {.error = ContactError::kStartsInsideSolid, .solid_id = 9, .magnitude = 0.01});
  ExpectUntouched(call.result);
}

TEST(MoveParticle, CeilingClosingFasterThanParticleLeaves_IsEndsTooCloseToSolid) {
  // The ceiling hits at 4/7 s and carries the particle down at 1.75 tol m/s; over the 3/7 s left the floor's
  // rate * time is -0.75 tol, above -tol, so the floor is not hit, and the particle ends at z = 0.25 tol.
  const MoveParticleCall call = RunMove("CeilingClosingFasterThanParticleLeaves");
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  EXPECT_EQ(call.failure.error, ContactError::kTooCloseToCall);
  EXPECT_EQ(call.failure.close_call, ContactCloseCall::kEndsTooCloseToSolid);
  EXPECT_EQ(call.failure.solid_id, 1U);
  EXPECT_NEAR(call.failure.magnitude, 0.25 * kTolerance, 0x1p-40 * kTolerance);
  ExpectUntouched(call.result);
}

TEST(MoveParticle, TurningBoxHoverOnSpinAxis_PassesOnCloseCall) {
  const MoveParticleCall call = RunMove("TurningBoxHoverOnSpinAxis");
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  ExpectFailure(
      call.failure,
      {
          .error = ContactError::kTooCloseToCall,
          .close_call = ContactCloseCall::kSlowApproachToTurningSolid,
          .solid_id = 1,
      }
  );
  ExpectUntouched(call.result);
}

// --- The start rules ---

// m. Tells a start moved to the rest gap from one left where it was, tol / 4 or more away; far above the rounding
// of the expected positions, which are below 1 m.
constexpr double kRestGapTolerance = kTolerance / 64;
// m. "At the rest gap" for a move off several surfaces: a distance within tol / 1024 of tol, the stop test of the move.
constexpr double kAtRestGapTolerance = kTolerance / 1024;

// Fails the calling test unless the call failed with the expected failure and left the result unchanged.
void ExpectStartFailure(std::string_view case_name, ContactFailure expected) {
  const MoveParticleCall call = RunMove(case_name);
  EXPECT_EQ(call.error, expected.error);
  ExpectFailure(call.failure, expected);
  ExpectUntouched(call.result);
}

TEST(MoveParticle, FlatSolidPushesRestingParticle_CarriesIt) {
  // At the rest gap, d = tol, the start rules leave the start alone; the floor rising at 2 m/s pushes into the
  // particle, so the touching rule makes it a hit at 0: both velocities {0, 0, 2}, carried to z = 0.2 + tol.
  const MoveParticleCall call = RunMove("FlatSolidPushesRestingParticle");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_fraction, 0);
  ExpectVec3Equal(result.velocity.for_this_step, {.x = 0, .y = 0, .z = 2});
  ExpectVec3Equal(result.velocity.for_next_step, {.x = 0, .y = 0, .z = 2});
  EXPECT_EQ(result.approach_speed.for_this_step, -2);
  EXPECT_EQ(result.approach_speed.for_next_step, -2);
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = 0.2 + kTolerance}, kPositionTolerance);
  EXPECT_EQ(result.hit_count, 1);
}

TEST(MoveParticle, FlatSolidPushesParticleWithinTolerance_HitAtStartFromRestGap) {
  // Closer than tol: moved to z = tol, then the approach of -2 m/s makes it a hit at 0, as at the rest gap.
  const MoveParticleCall call = RunMove("FlatSolidPushesParticleWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_fraction, 0);
  ExpectVec3Equal(result.velocity.for_this_step, {.x = 0, .y = 0, .z = 2});
  EXPECT_EQ(result.approach_speed.for_this_step, -2);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = 0.2 + kTolerance}, kRestGapTolerance);
  EXPECT_EQ(result.hit_count, 1);
}

TEST(MoveParticle, TouchingAndLeaving_IsFree) {
  const MoveParticleCall call = RunMove("TouchingAndLeaving");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  EXPECT_EQ(call.result.position.z, kTolerance + 0.1);
}

TEST(MoveParticle, ShallowOverlap_StartedInsideSolidAndMovedOut) {
  // 0.0005 m inside, within the overlap tolerance of 0.001 m: moved to z = tol with the response, at rest.
  const MoveParticleCall call = RunMove("ShallowOverlap");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.hit_fraction, 0);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.solid_index, 0U);
  ExpectVec3Equal(result.normal_out_of_solid, {.x = 0, .y = 0, .z = 1});
  ExpectVec3Equal(result.velocity.for_this_step, {});
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, ShallowOverlapRisingFloor_CarriedFromRestGap) {
  // The floor at its pose of time 0 is the one the start is inside: moved to z = tol, then carried at 1 m/s.
  const MoveParticleCall call = RunMove("ShallowOverlapRisingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.approach_speed.for_this_step, -1);
  ExpectVec3Equal(result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = 0.1 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DeepOverlap_IsStartsInsideSolid) {
  ExpectStartFailure("DeepOverlap", {.error = ContactError::kStartsInsideSolid, .solid_id = 1, .magnitude = 0.01});
}

TEST(MoveParticle, DeepOverlapSinkingFloor_IsStartsInsideSolidAtTimeZero) {
  ExpectStartFailure(
      "DeepOverlapSinkingFloor",
      {.error = ContactError::kStartsInsideSolid, .solid_id = 1, .magnitude = 0.01}
  );
}

TEST(MoveParticle, StartsWithinToleranceMovingAway_IsMovedOutToOneTolerance) {
  // Plan decision P12: a silent move to z = tol; nothing else changes.
  const MoveParticleCase test_case = FindMoveParticleCase("StartsWithinToleranceMovingAway");
  const MoveParticleCall call = RunMove(test_case);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kNone);
  EXPECT_EQ(result.hit_count, 0);
  EXPECT_EQ(result.hit_fraction, 0);
  EXPECT_EQ(result.solid_index, kNoSolid);
  ExpectVec3Equal(result.velocity.for_this_step, test_case.inputs.velocity.for_this_step);
  ExpectVec3Equal(result.velocity.for_next_step, test_case.inputs.velocity.for_next_step);
  ExpectNoReaction(result.reaction);
  EXPECT_EQ(result.position.x, 0.1);
  EXPECT_EQ(result.position.y, 0);
  EXPECT_NEAR(result.position.z, kTolerance + 0.2, kPositionTolerance);
  // Tighter than the brief's 2 tol: the start moved to the rest gap, not left at tol / 4.
  EXPECT_NEAR(result.position.z, kTolerance + 0.2, kRestGapTolerance);
}

TEST(MoveParticle, StartsWithinToleranceOfRisingFloorMovingAway_IsMovedOutSilently) {
  // 1 m/s away from the floor, which rises at 1 m/s: moved silently to tol above the floor at time 0.
  const MoveParticleCall call = RunMove("StartsWithinToleranceOfRisingFloorMovingAway");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Equal(call.result.velocity.for_this_step, {.x = 1, .y = 0, .z = 2});
  EXPECT_EQ(call.result.position.x, 0.1);
  EXPECT_NEAR(call.result.position.z, kTolerance + 0.2, kRestGapTolerance);
}

TEST(MoveParticle, ForNextStepApproachesWithinTolerance_IsHitAtStart) {
  // for_next_step {0, 0, -1} approaches: the hit stops it; for_this_step {1, 0, 1} leaves and comes back unchanged.
  const MoveParticleCall call = RunMove("ForNextStepApproachesWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.approach_speed.for_next_step, -1);
  EXPECT_EQ(result.approach_speed.for_this_step, 1);
  ExpectVec3Equal(result.velocity.for_next_step, {});
  ExpectVec3Equal(result.velocity.for_this_step, {.x = 1, .y = 0, .z = 1});
  ExpectVec3Near(result.position, {.x = 0.1, .y = 0, .z = kTolerance + 0.1}, kRestGapTolerance);
}

TEST(MoveParticle, SlowApproachWithinTolerance_IsHitAtStart) {
  // Any approach of either velocity is a hit at the start, however slow: the fall stops at z = tol.
  const MoveParticleCall call = RunMove("SlowApproachWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3Equal(call.result.velocity.for_this_step, {.x = 1, .y = 0, .z = 0});
  ExpectVec3Equal(call.result.velocity.for_next_step, {.x = 0, .y = 0, .z = 1});
  ExpectVec3Near(call.result.position, {.x = 0.1, .y = 0, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, StartsWithinToleranceOfSinkingFloorFollowing_ApproachMeasuredFromFloor) {
  // The approach is measured from the moving floor: +1 m/s, leaving, so the move is silent.
  const MoveParticleCall call = RunMove("StartsWithinToleranceOfSinkingFloorFollowing");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = kTolerance - 0.1}, kRestGapTolerance);
}

TEST(MoveParticle, FiveFloorsWithinStartOverlap_FifthStartHitIsTooManyHits) {
  // The start hits count toward kMaxHitsPerStep (plan decision P4): the fifth floor, id 5, is the fifth hit.
  ExpectStartFailure(
      "FiveFloorsWithinStartOverlap",
      {.error = ContactError::kTooCloseToCall, .close_call = ContactCloseCall::kTooManyHits, .solid_id = 5}
  );
}

TEST(MoveParticle, SphereApproachingWithinTolerance_IsHitAtStartOnCurvedFace) {
  // Moved along the normal {-1, 0, 0} to x = -(0.5 + tol), then the sphere's -1 m/s carries the particle.
  const MoveParticleCall call = RunMove("SphereApproachingWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.face, SolidFace::kCurved);
  EXPECT_EQ(result.approach_speed.for_this_step, -1);
  ExpectVec3Equal(result.velocity.for_this_step, {.x = -1, .y = 0, .z = 0});
  ExpectVec3Near(result.position, {.x = -(0.6 + kTolerance), .y = 0, .z = 0}, kRestGapTolerance);
  EXPECT_EQ(result.hit_count, 1);
}

TEST(MoveParticle, SpinningMovingSphereStartWithinTolerance_ApproachReadAtTimeZero) {
  // The surface velocity at the pose of time 0 has no part along the normal, so the particle leaves: a silent move.
  const MoveParticleCall call = RunMove("SpinningMovingSphereStartWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Equal(call.result.velocity.for_this_step, {.x = 1, .y = 0, .z = 0.1});
  ExpectVec3Near(call.result.position, {.x = 0.1, .y = 0, .z = 0.51 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, TurningBoxStartWithinTolerance_EndsInTooManyHits) {
  // The departure of MoveParticle's step 5 until the slide on a turning solid: no touching rule, so every sweep from
  // the rest gap hits the box at 0.
  ExpectStartFailure(
      "TurningBoxStartWithinTolerance",
      {.error = ContactError::kTooCloseToCall, .close_call = ContactCloseCall::kTooManyHits, .solid_id = 1}
  );
}

TEST(MoveParticle, CommittedSideInsideThickness_PushedToCommittedSide) {
  // Committed to kAlongNormal, the start at z = -0.001 is at d = -0.011 behind kHighZ: pushed through to 0.01 + tol.
  const MoveParticleCall call = RunMove("CommittedSideInsideThickness");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  EXPECT_EQ(result.side_at_hit, SolidSide::kAlongNormal);
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = 0.01 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, UncommittedInsideThickness_PushedToNearerFace) {
  // Not committed, the same start is nearer kLowZ, d = -0.009: pushed to -(0.01 + tol).
  const MoveParticleCall call = RunMove("UncommittedInsideThickness");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.face, SolidFace::kLowZ);
  EXPECT_EQ(result.side_at_hit, SolidSide::kAgainstNormal);
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = -(0.01 + kTolerance)}, kRestGapTolerance);
}

TEST(MoveParticle, CommittedAgainstNormalInsideThickness_PushedToCommittedSide) {
  // Committed to kAgainstNormal, the start at z = 0.001, nearer kHighZ, is pushed through to -(0.01 + tol).
  const MoveParticleCall call = RunMove("CommittedAgainstNormalInsideThickness");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = -(0.01 + kTolerance)}, kRestGapTolerance);
}

TEST(MoveParticle, CommittedSideInsideMovingPlate_PushedOutAndCarried) {
  // The plate rises at 1 m/s: pushed to its kHighZ face at time 0, then carried to 0.11 + tol.
  const MoveParticleCall call = RunMove("CommittedSideInsideMovingPlate");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.face, SolidFace::kHighZ);
  ExpectVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = 0.11 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, UncommittedInsideMovingPlate_PushedToLowFaceAndLeftBehind) {
  // Pushed to kLowZ at time 0; the plate rises away from the particle, which stays at rest.
  const MoveParticleCall call = RunMove("UncommittedInsideMovingPlate");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  EXPECT_EQ(call.result.approach_speed.for_this_step, 1);
  ExpectVec3Equal(call.result.velocity.for_this_step, {});
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = -(0.01 + kTolerance)}, kRestGapTolerance);
}

TEST(MoveParticle, CommittedBeyondOtherFace_IsOnWrongSideOfPlate) {
  // z = -0.02 is 0.01 m beyond kLowZ (plan decision P6: -(s * p.z + half_thickness)).
  ExpectStartFailure(
      "CommittedBeyondOtherFace",
      {.error = ContactError::kOnWrongSideOfPlate, .solid_id = 5, .magnitude = 0.01}
  );
}

TEST(MoveParticle, CommittedAgainstNormalBeyondOtherFace_IsOnWrongSideOfPlate) {
  ExpectStartFailure(
      "CommittedAgainstNormalBeyondOtherFace",
      {.error = ContactError::kOnWrongSideOfPlate, .solid_id = 5, .magnitude = 0.01}
  );
}

TEST(MoveParticle, CommittedBeyondOtherFaceOfMovingPlate_MeasuredAtTimeZero) {
  ExpectStartFailure(
      "CommittedBeyondOtherFaceOfMovingPlate",
      {.error = ContactError::kOnWrongSideOfPlate, .solid_id = 5, .magnitude = 0.01}
  );
}

TEST(MoveParticle, CommittedBeyondOtherFaceSmallOverlap_WrongSideComesBeforeDepth) {
  // Plan decision P3: the committed distance -0.03 m is also deeper than the overlap tolerance of 0.001 m.
  ExpectStartFailure(
      "CommittedBeyondOtherFaceSmallOverlap",
      {.error = ContactError::kOnWrongSideOfPlate, .solid_id = 5, .magnitude = 0.01}
  );
}

TEST(MoveParticle, CommittedJustBeyondToleranceOfOtherFace_IsOnWrongSideOfPlate) {
  const MoveParticleCall call = RunMove("CommittedJustBeyondToleranceOfOtherFace");
  EXPECT_EQ(call.error, ContactError::kOnWrongSideOfPlate);
  EXPECT_EQ(call.failure.solid_id, 5U);
  EXPECT_NEAR(call.failure.magnitude, 1.5 * kTolerance, kTolerance / 1024);
  ExpectUntouched(call.result);
}

TEST(MoveParticle, CommittedWithinToleranceOfOtherFace_PushedToCommittedSide) {
  // tol / 2 beyond kLowZ is not the wrong side: d = -(0.02 + tol / 2), pushed through to 0.01 + tol.
  const MoveParticleCall call = RunMove("CommittedWithinToleranceOfOtherFace");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.face, SolidFace::kHighZ);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = 0.01 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, CornerWithinTolerance_BothSolidsHitAtStart) {
  // The floor moves the start to z = tol and stops the fall: {1, 0, 0}. The upright, posed at time 0 at x = 1, moves
  // it to x = 1 - tol, and its approach of -1.5 m/s gives its velocity {-0.5, 0, 0}, which carries the particle.
  const MoveParticleCall call = RunMove("CornerWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 2);
  EXPECT_EQ(result.solid_index, 0U);
  EXPECT_EQ(result.approach_speed.for_this_step, -1);
  ExpectVec3Equal(result.velocity.for_this_step, {.x = -0.5, .y = 0, .z = 0});
  ExpectVec3Near(result.position, {.x = 0.95 - kTolerance, .y = 0, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, TwoReactingSolidsWithinTolerance_IsSecondReactingSolid) {
  ExpectStartFailure(
      "TwoReactingSolidsWithinTolerance",
      {.error = ContactError::kTooCloseToCall, .close_call = ContactCloseCall::kSecondReactingSolid, .solid_id = 2}
  );
}

TEST(MoveParticle, InsideUprightAfterApproachingFloor_KindIsStartedInsideSolid) {
  // The record is the floor's, the first hit; the start inside the upright, a later solid, sets the kind.
  const MoveParticleCall call = RunMove("InsideUprightAfterApproachingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.solid_index, 0U);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  EXPECT_EQ(result.approach_speed.for_this_step, -2);
  EXPECT_EQ(result.hit_count, 2);
  ExpectVec3Equal(result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
  ExpectVec3Near(result.position, {.x = 1 - kTolerance, .y = 0, .z = 0.1 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerWithinTolerance_MovedFromBothSidesSilently) {
  // Both near sides move the start to the rest gap: one move along the nearer side alone would leave z = 0.3 tol, and
  // the end check would fail.
  const MoveParticleCall call = RunMove("DomainCornerWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = 0.5, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerSecondSideBeyondTolerance_MovedFromNearSideOnly) {
  // side_bottom at 1.5 tol is no surface of the start rules, though the particle approaches it: no hit at the start,
  // and the sweep's gap of 0.5 tol closes in 0.125 s, after the step.
  const MoveParticleCall call = RunMove("DomainCornerSecondSideBeyondTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = 0.5, .z = 1.1 * kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainBoxOpenBack_LeavesThroughIt) {
  const MoveParticleCall call = RunMove("DomainBoxOpenBack");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  ExpectVec3Near(call.result.position, {.x = 0.5, .y = 1.1, .z = 0.5}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerAtOpenTop_MovedFromClosedSideOnly) {
  // The open top is no surface: the particle rising through it is not stopped.
  const MoveParticleCall call = RunMove("DomainCornerAtOpenTop");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = 0.5, .z = 1.1 - kTolerance / 4}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerSlightlyInside_MovedOutOfBothSides) {
  // The start is inside past side_left, the nearest, and side_bottom: both move it out; side_left responds at the
  // start, and the first sweep stops the fall on side_bottom at 0.
  const MoveParticleCall call = RunMove("DomainCornerSlightlyInside");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.face, SolidFace::kLowX);
  EXPECT_EQ(result.hit_count, 2);
  ExpectVec3Equal(result.velocity.for_this_step, {});
  ExpectVec3Near(result.position, {.x = kTolerance, .y = 0.5, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, WedgeStartPushedTowardFloor_MovesOffBothAtOnce) {
  // Shortfalls 0.5 tol (floor) and 0.1 tol (wedge), normals -0.8 apart: one move of {-5/6 tol, 0, 0.5 tol}. The fall
  // approaches the floor, which responds; the rising wedge is left. The slide along -x keeps z = tol.
  const MoveParticleCall call = RunMove("WedgeStartPushedTowardFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.solid_index, 0U);
  ExpectVec3Equal(result.velocity.for_this_step, {.x = -1, .y = 0, .z = 0});
  ExpectVec3Equal(result.velocity.for_next_step, {.x = -1, .y = 0, .z = 0});
  ExpectVec3Near(result.position, {.x = -0.1 - 5.0 / 6 * kTolerance, .y = 0, .z = kTolerance}, kAtRestGapTolerance);
}

TEST(MoveParticle, WedgeStartPushedIntoFloor_MovesOffBothAtOnce) {
  // 0.0005 m inside the wedge: one move off both faces at once. The floor is approached, the wedge was started inside:
  // two hits, and the kind is kStartedInsideSolid.
  const MoveParticleCall call = RunMove("WedgeStartPushedIntoFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.hit_count, 2);
  ExpectVec3Equal(result.velocity.for_this_step, {.x = -1, .y = 0, .z = 0});
  ExpectVec3Equal(result.velocity.for_next_step, {.x = -1, .y = 0, .z = 0});
  const double end_x = -0.1 - 0.0005 * 5.0 / 3 - 5.0 / 6 * kTolerance;  // m
  ExpectVec3Near(result.position, {.x = end_x, .y = 0, .z = kTolerance}, kAtRestGapTolerance);
}

TEST(MoveParticle, ParallelSolidAfterStartResponse_RoundingIsNotAnApproach) {
  // The approach of -2^-55 m/s that the first solid's response leaves is within its allowance, so the reacting second
  // solid, 0.75 tol away after the first move, moves the particle silently: one hit, no reaction. The silently moved
  // solid is a touched surface, so the sweep rests on it whichever side of the rest gap rounding leaves the start;
  // GeneratedParallelSolids_RoundingNeverHitsTheSecond draws both signs.
  const MoveParticleCall call = RunMove("ParallelSolidAfterStartResponse");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectNoReaction(call.result.reaction);
}

TEST(MoveParticle, FourSilentFloorsAndUpright_AllKeptAsTouchedSurfaces) {
  const MoveParticleCall call = RunMove("FourSilentFloorsAndUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 4U);
}

TEST(MoveParticle, FiveSilentFloorsAndUpright_FifthSilentMoveIsTooManyHits) {
  // The start alone decides this limit; a shorter step does not help.
  ExpectStartFailure(
      "FiveSilentFloorsAndUpright",
      {.error = ContactError::kTooCloseToCall, .close_call = ContactCloseCall::kTooManyHits, .solid_id = 5}
  );
}

TEST(MoveParticle, FiveSilentFloors_NoStartResponseNeverFailsForSilentMoves) {
  const MoveParticleCall call = RunMove("FiveSilentFloors");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
}

TEST(MoveParticle, SlidingFloorThenSlowUpright_ThresholdIsZeroBeforeAnyResponse) {
  // A silent entry has allowance 0: before any response the approach test compares with 0, so the upright's real
  // approach of 1e-13 m/s is a hit at the start.
  const MoveParticleCall call = RunMove("SlidingFloorThenSlowUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.velocity.for_next_step.x, 0);
}

TEST(MoveParticle, SilentFloorThenLeaningUpright_FloorNotTouchedWithoutStartHit) {
  // Plan decision P12: with no hit at the start the silent move leaves no touched surface, so the upright's push is a
  // drift on the floor's plain touching rule, not a hit on it: one hit, the upright's. The settle reads every solid
  // (plan decision P19, item 7), so it lifts the drift of 0.05 tol back to the rest gap.
  const MoveParticleCall call = RunMove("SilentFloorThenLeaningUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_NEAR(call.result.position.z, kTolerance, kAtRestGapTolerance);
}

TEST(MoveParticle, AtRestGapSlowApproach_TouchingRuleLetsItRest) {
  // d = tol exactly is outside the start rules' range, d < tol: the sweep's touching rule decides, and the approach of
  // tol m/s for 0.1 s, -0.1 tol, is not a hit. A start rule there would stop the particle at z = tol.
  const MoveParticleCall call = RunMove("AtRestGapSlowApproach");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  EXPECT_EQ(call.result.velocity.for_this_step.z, -kTolerance);
  // The sweep lets it sink to 0.9 tol; the settle, which reads every solid, lifts it back.
  EXPECT_NEAR(call.result.position.z, kTolerance, kAtRestGapTolerance);
}

TEST(MoveParticle, DomainCornerSlowApproachToSecondSide_IsHitOnThatSide) {
  // Each near side is a surface of its own: side_bottom, not the nearest, is approached and responds; side_left is
  // moved silently.
  const MoveParticleCall call = RunMove("DomainCornerSlowApproachToSecondSide");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  ExpectVec3Equal(call.result.velocity.for_this_step, {});
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = 0.5, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerNextStepIntoSecondSide_IsHitOnThatSide) {
  const MoveParticleCall call = RunMove("DomainCornerNextStepIntoSecondSide");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  EXPECT_EQ(call.result.approach_speed.for_next_step, -1);
  ExpectVec3Equal(call.result.velocity.for_next_step, {});
}

TEST(MoveParticle, DomainThreeSideCornerAtRest_MovedFromAllThreeSilently) {
  const MoveParticleCall call = RunMove("DomainThreeSideCornerAtRest");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainThreeSideCornerThisStepApproaches_HitOnEachSide) {
  // for_this_step {-1, -1, -1} approaches the three sides in turn: three hits, in face order, the record on kLowX.
  const MoveParticleCall call = RunMove("DomainThreeSideCornerThisStepApproaches");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 3);
  EXPECT_EQ(call.result.face, SolidFace::kLowX);
  ExpectVec3Equal(call.result.velocity.for_this_step, {});
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainThreeSideCornerNextStepApproaches_HitOnEachSide) {
  const MoveParticleCall call = RunMove("DomainThreeSideCornerNextStepApproaches");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 3);
  ExpectVec3Equal(call.result.velocity.for_next_step, {});
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainThreeSideCornerSlightlyInside_MovedOutOfAllThree) {
  const MoveParticleCall call = RunMove("DomainThreeSideCornerSlightlyInside");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.hit_count, 3);
  EXPECT_EQ(call.result.face, SolidFace::kLowX);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerNearestSideComesLater_RecordNamesTheSideHit) {
  // The fall approaches side_bottom only, so the record names kLowZ, the side read before the move, even though the
  // sides tie at tol after it and a tie keeps kLowX.
  const MoveParticleCall call = RunMove("DomainCornerNearestSideComesLater");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  ExpectVec3Equal(call.result.normal_out_of_solid, {.x = 0, .y = 0, .z = 1});
}

// --- A start near several surfaces at once (plan decision P19) ---

// Fails the calling test unless every solid of the inputs is at the rest gap of position at its end pose: a distance
// within tol / 1024 of tol, by ComputeDistanceToSolid itself.
void ExpectAtRestGapOfEverySolid(const SolidContactInputs& inputs, Vec3 position) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    const double distance =
        ComputeDistanceToSolid(solid, ComputeSolidPose(solid, inputs.step_duration), position).distance;
    EXPECT_NEAR(distance, kTolerance, kAtRestGapTolerance) << "solid index " << solid_index;
  }
}

// Runs MoveParticle on the scene of that name and fails the calling test unless it succeeded at the expected end, with
// every solid at the rest gap.
SolidContactResult ExpectEndAtRestGap(std::string_view case_name, Vec3 expected_end, double position_tolerance) {
  const MoveParticleCase scene = FindMoveParticleCase(case_name);
  const MoveParticleCall call = RunMove(scene);
  EXPECT_EQ(call.error, ContactError::kNone) << case_name;
  ExpectVec3Near(call.result.position, expected_end, position_tolerance);
  ExpectAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
  return call.result;
}

TEST(MoveParticle, SixtyDegreeWedgeRestingOnBoth_EndsAtRestGapOfBoth) {
  // One move of tol / sin(30 degrees) = 2 tol along the bisector. Moving off the floor, then off the other face, ended
  // 0.8125 tol from the floor.
  const SolidContactResult result = ExpectEndAtRestGap(
      "SixtyDegreeWedgeRestingOnBoth",
      {.x = std::numbers::sqrt3 * kTolerance, .y = 0, .z = kTolerance},
      kAtRestGapTolerance
  );
  EXPECT_EQ(result.kind, ContactKind::kNone);
  EXPECT_EQ(result.hit_count, 0);
}

TEST(MoveParticle, ThirtyDegreeWedgeRestingOnBoth_EndsAtRestGapOfBoth) {
  // A move of 3.8637 tol. Moving off one face at a time took the start inside the floor, and the sweeps hit it at 0
  // until kTooManyHits, at every step length.
  const SolidContactResult result = ExpectEndAtRestGap(
      "ThirtyDegreeWedgeRestingOnBoth",
      {.x = (2 + std::numbers::sqrt3) * kTolerance, .y = 0, .z = kTolerance},
      kAtRestGapTolerance
  );
  EXPECT_EQ(result.kind, ContactKind::kNone);
  EXPECT_EQ(result.hit_count, 0);
}

TEST(MoveParticle, TenDegreeWedgeFallingOnBoth_EndsAtRestGapOfBoth) {
  // The fall approaches the floor, which stops it, and leaves the other face: one hit, at the rest point
  // tol / tan(5 degrees) out along the floor.
  const double rest_x = kTolerance / std::tan(5 * std::numbers::pi / 180);  // m, 11.430052 tol.
  const SolidContactResult result =
      ExpectEndAtRestGap("TenDegreeWedgeFallingOnBoth", {.x = rest_x, .y = 0, .z = kTolerance}, kAtRestGapTolerance);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.solid_index, 0U);
  ExpectVec3Equal(result.velocity.for_this_step, {});
  ExpectVec3Equal(result.velocity.for_next_step, {});
}

TEST(MoveParticle, VerySharpFlatWedgeRestingOnBoth_EndsAtRestGapOfBoth) {
  // A wedge of 0.01 degrees: the rest point is tol / tan(0.005 degrees) = 11459.156 tol out. The first version of plan
  // decision P19 failed it, below its 0.4476 degree limit.
  const double rest_x = kTolerance / std::tan(0.005 * std::numbers::pi / 180);  // m
  const SolidContactResult result =
      ExpectEndAtRestGap("VerySharpFlatWedgeRestingOnBoth", {.x = rest_x, .y = 0, .z = kTolerance}, kTolerance / 256);
  EXPECT_EQ(result.kind, ContactKind::kNone);
}

TEST(MoveParticle, FlatWedgeNeedingMoveBeyondCap_IsStartsInNarrowGap) {
  // The move cap is 2^22 tol = 4.19e6 tol. A wedge of 6e-7 rad rests the particle 3333942 tol out, within 2e-4 of
  // tol / tan(3e-7); one of 3e-7 rad would need 6.67e6 tol, beyond the cap.
  const MoveParticleCase within_cap = FindMoveParticleCase("FlatWedgeNeedingMoveWithinCap");
  const MoveParticleCall within_call = RunMove(within_cap);
  ASSERT_EQ(within_call.error, ContactError::kNone);
  const double rest_x = kTolerance / std::tan(3e-7);  // m
  EXPECT_NEAR(within_call.result.position.x / rest_x, 1, 2e-4);
  ExpectAtRestGapOfEverySolid(ToSolidContactInputs(within_cap), within_call.result.position);
  ExpectStartFailure("FlatWedgeNeedingMoveBeyondCap", {.error = ContactError::kStartsInNarrowGap, .solid_id = 2});
}

TEST(MoveParticle, ThreeSolidAcuteCornerFallingOnAll_EndsAtRestGapOfAll) {
  // Three faces at 60 degrees to each other: one move of Sqrt(7) tol to the point at tol from all three. The fall
  // approaches the floor, which stops it.
  const double root_three = std::numbers::sqrt3;
  const SolidContactResult result = ExpectEndAtRestGap(
      "ThreeSolidAcuteCornerFallingOnAll",
      {.x = root_three * kTolerance, .y = root_three * kTolerance, .z = kTolerance},
      kAtRestGapTolerance
  );
  EXPECT_EQ(result.hit_count, 1);
  ExpectVec3Equal(result.velocity.for_this_step, {});
}

TEST(MoveParticle, ThirdSolidNearOnlyAfterFirstMove_JoinsNextIteration) {
  // The third face is 1.414 tol from the start, so it does not constrain the first iterate, which leaves it 0.19 tol
  // away; it constrains the next one: a move of 2.3052 tol in all.
  const SolidContactResult result = ExpectEndAtRestGap(
      "ThirdSolidNearOnlyAfterFirstMove",
      {.x = std::numbers::sqrt3 * kTolerance, .y = -1.146264 * kTolerance, .z = kTolerance},
      kAtRestGapTolerance
  );
  EXPECT_EQ(result.kind, ContactKind::kNone);
}

TEST(MoveParticle, DomainCornerCutBySolid_HoldsParticleOrIsStartsInNarrowGap) {
  // A cut 6 tol out leaves a pocket with an inscribed radius of 1.27 tol: the corner holds the particle at
  // {tol, tol, tol} and side_bottom stops its fall. A cut 4 tol out leaves 0.845 tol, too small for a point at tol from
  // all four faces; the cut, crossed by the move off the corner's sides, is named.
  const MoveParticleCall far_call = RunMove("DomainCornerCutBySolidFarEnough");
  ASSERT_EQ(far_call.error, ContactError::kNone);
  ExpectVec3Near(far_call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kAtRestGapTolerance);
  EXPECT_EQ(far_call.result.hit_count, 1);
  ExpectStartFailure("DomainCornerCutBySolidTooNear", {.error = ContactError::kStartsInNarrowGap, .solid_id = 2});
}

TEST(MoveParticle, ParallelFacesLessThanTwoTolerancesApart_IsStartsInNarrowGap) {
  // 1.5 tol apart, no point is at tol from both. 2.5 tol apart, the particle moves off the floor alone.
  ExpectStartFailure(
      "ParallelFacesLessThanTwoTolerancesApart",
      {.error = ContactError::kStartsInNarrowGap, .solid_id = 2}
  );
  const MoveParticleCall wide_call = RunMove("ParallelFacesMoreThanTwoTolerancesApart");
  ASSERT_EQ(wide_call.error, ContactError::kNone);
  EXPECT_EQ(wide_call.result.kind, ContactKind::kNone);
  ExpectVec3Equal(wide_call.result.position, {.x = 0, .y = 0, .z = kTolerance});
}

TEST(MoveParticle, BallRestingOnFloorStartInCusp_MovesToRestPoint) {
  // The rest point of a ball of radius R resting on a floor is 2 Sqrt(R tol) from the contact, both distances exactly
  // tol: {46340.95 tol, 0, tol} for R = 0.5. The first iterate overshoots out along the floor; the next come back.
  const MoveParticleCase scene = FindMoveParticleCase("BallRestingOnFloorStartInCusp");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_NEAR(call.result.position.x, 2 * std::sqrt(0.5 * kTolerance), kTolerance / 256);
  EXPECT_NEAR(call.result.position.y, 0, kTolerance / 256);
  EXPECT_NEAR(call.result.position.z, kTolerance, kAtRestGapTolerance);
  ExpectAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
}

TEST(MoveParticle, BallRestingOnFloorStartJustShortOfFloor_MovesToRestPoint) {
  // The first iterate moves 0.014 tol, less than tol / 64, but leaves the ball 0.0124 tol short: the iterates stop only
  // at a point at the rest gap of every solid, here the rest point.
  const SolidContactResult result = ExpectEndAtRestGap(
      "BallRestingOnFloorStartJustShortOfFloor",
      {.x = 2 * std::sqrt(0.5 * kTolerance), .y = 0, .z = kTolerance},
      kTolerance / 256
  );
  EXPECT_EQ(result.kind, ContactKind::kNone);
}

TEST(MoveParticle, BallRestingOnFloorDeepStart_MovesToRestPoint) {
  // R / tol = 2^40, the largest radius a scene allows: the rest point is x* = 0.001953125 m out. From 0.1 x* and from
  // 0.01 x*, 8 iterates end within 1e-4 of it. From 0.0005 x*, within 1.8e-8 R of the contact, the two normals are
  // opposite to the last bit (1 + Dot rounds to 0): no move is found.
  const double rest_x = 2 * std::sqrt(1024 * kTolerance);  // m
  for (const std::string_view case_name :
       {"BallRestingOnFloorDeepStartTenth", "BallRestingOnFloorDeepStartHundredth"}) {
    const MoveParticleCase scene = FindMoveParticleCase(case_name);
    const MoveParticleCall call = RunMove(scene);
    ASSERT_EQ(call.error, ContactError::kNone) << case_name;
    EXPECT_NEAR(call.result.position.x / rest_x, 1, 1e-4) << case_name;
    ExpectAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
  }
  ExpectStartFailure(
      "BallRestingOnFloorDeepStartNearContact",
      {.error = ContactError::kStartsInNarrowGap, .solid_id = 2}
  );
}

TEST(MoveParticle, TwoBallsTouchingStartInNeck_MovesToRestRing) {
  // Two balls of radius R touching: the ring where both are at tol is Sqrt(tol (2 R + tol)) = 32768.0 tol from the
  // contact for R = 0.5.
  const double ring_radius = std::sqrt(kTolerance * (1 + kTolerance));  // m
  const SolidContactResult result =
      ExpectEndAtRestGap("TwoBallsTouchingStartInNeck", {.x = 0.5, .y = ring_radius, .z = 0}, kTolerance / 256);
  EXPECT_EQ(result.kind, ContactKind::kNone);
}

TEST(MoveParticle, PlateAndSphereRing_MovesOffBothAlongBisector) {
  // The plate and the ball meet at 60 degrees on their ring: the move is 1.5 tol along the bisector, the plate is then
  // at the rest gap and the curved ball slightly beyond it. The fall approaches the plate, which stops it.
  const MoveParticleCase scene = FindMoveParticleCase("PlateAndSphereRing");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3Equal(call.result.velocity.for_this_step, {});
  const Vec3 ring_point = {.x = std::sqrt(0.1875), .y = 0, .z = 0.01};
  const Vec3 bisector = {.x = std::numbers::sqrt3 / 2, .y = 0, .z = 0.5};
  ExpectVec3Near(call.result.position, ring_point + 2 * kTolerance * bisector, kAtRestGapTolerance);
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  const double plate_distance =
      ComputeDistanceToSolid(inputs.solids[0], ComputeSolidPose(inputs.solids[0], 0), call.result.position).distance;
  const double sphere_distance =
      ComputeDistanceToSolid(inputs.solids[1], ComputeSolidPose(inputs.solids[1], 0), call.result.position).distance;
  EXPECT_NEAR(plate_distance, kTolerance, kAtRestGapTolerance);
  EXPECT_GE(sphere_distance, kTolerance - kAtRestGapTolerance);
  EXPECT_LE(sphere_distance, kTolerance + kTolerance / 64);
}

TEST(MoveParticle, ParallelSolidAfterStartResponse_NextStepRoundingIsNotAnApproach) {
  // The first solid's response to for_this_step at rest has an allowance of 0, but it leaves -2^-55 m/s in the moving
  // for_next_step. Compared with for_next_step's own allowance that is rounding, so the reacting second solid is moved
  // silently. One allowance for both velocities made it a second hit, with a reaction of 2.8e-17 m/s.
  const MoveParticleCall call = RunMove("ParallelSolidAfterStartResponseNextStepMoving");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.reaction.solid_index, kNoSolid);
}

TEST(MoveParticle, SlidingFloorThenSlowUprightNextStepOnly_NextStepThresholdIsZeroBeforeAnyResponse) {
  // A silent entry leaves no rounding of its own, so for_next_step's threshold stays 0 too: its approach of 1e-13 m/s
  // to the upright is a hit at the start.
  const MoveParticleCall call = RunMove("SlidingFloorThenSlowUprightNextStepOnly");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.velocity.for_next_step.x, 0);
}

TEST(MoveParticle, PlateAcrossSharpWedge_IsNotCrossedByTheMove) {
  // The pair of faces would move the start 3819.7 tol out, through the plate, whose near face is 100 tol out. The
  // crossing check makes the plate constrain the move, and no point between the apex and the plate is at tol from all
  // three. Without it the particle ends 2520 tol beyond the plate's far face.
  ExpectStartFailure("PlateAcrossSharpWedge", {.error = ContactError::kStartsInNarrowGap, .solid_id = 3});
}

TEST(MoveParticle, SecondFaceNearOnlyAfterFloorMove_MovesOffBoth) {
  // Only the floor is near, but its own move brings the 60 degree face to 0.95 tol: that move is not kept, and the move
  // off several surfaces ends at the rest gap of both.
  const SolidContactResult result = ExpectEndAtRestGap(
      "SecondFaceNearOnlyAfterFloorMove",
      {.x = std::numbers::sqrt3 * kTolerance, .y = 0, .z = kTolerance},
      kAtRestGapTolerance
  );
  EXPECT_EQ(result.kind, ContactKind::kNone);
}

TEST(MoveParticle, StartInsideFloorBelowPlate_IsNotCrossedByTheMove) {
  // The floor's own move to its rest gap would pass through the plate inside it, so it is not kept; with the plate
  // constraining the move no point is at the rest gap of both, and the plate is named.
  ExpectStartFailure("StartInsideFloorBelowPlate", {.error = ContactError::kStartsInNarrowGap, .solid_id = 2});
}

TEST(MoveParticle, DeepCuspBesideWall_MovesToRestPoint) {
  // The first iterate from 0.1 x* overshoots far out along the floor and crosses a wall 1.5 x* out. The rest point is
  // 0.5 x*, about 10^6 tol, from the wall: the move must still reach it.
  const MoveParticleCase scene = FindMoveParticleCase("BallRestingOnFloorDeepStartBesideWall");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_NEAR(call.result.position.x / (2 * std::sqrt(1024 * kTolerance)), 1, 1e-4);
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  for (std::uint32_t solid_index = 0; solid_index < 2; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    EXPECT_NEAR(
        ComputeDistanceToSolid(solid, ComputeSolidPose(solid, 0), call.result.position).distance,
        kTolerance,
        kAtRestGapTolerance
    );
  }
}

TEST(MoveParticle, SmallBallInDomainCornerCuspStart_MovesToRestPoint) {
  // A ball of radius 2^10 tol, the smallest feature, resting in a corner of the domain box, and a start in its cusp
  // with the floor: the first iterate crosses the domain's left side, which is R from the rest point.
  const MoveParticleCase scene = FindMoveParticleCase("SmallBallInDomainCornerCuspStart");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  ExpectAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
}

TEST(MoveParticle, DeepCuspWallWithinRoundingOfRestPoint_KnownLimitIsStartsInNarrowGap) {
  // A known limit, beside the start near the contact point of BallRestingOnFloorDeepStart: the rest point is 1.01 tol
  // from the wall, inside the window where the ball's rounded distance, ulp(R) / (2 Sqrt(tol / R)) = about 128 tol at R
  // = 1024 m, moves the linearised cusp's vertex past the wall, and no subset of the three targets has an answer. A
  // solver that tried targets relaxed by the slack would find it; this test then changes to a success.
  ExpectStartFailure(
      "BallRestingOnFloorDeepStartWallNearRestPoint",
      {.error = ContactError::kStartsInNarrowGap, .solid_id = 3}
  );
}

TEST(MoveParticle, CoincidentFloors_SixteenFitSeventeenIsStartsInNarrowGap) {
  // Surfaces with the same normal share one target, but one move tracks at most kMaxTrackedSurfaces = 16 surfaces.
  const MoveParticleCall sixteen_call = RunMove("SixteenCoincidentFloors");
  ASSERT_EQ(sixteen_call.error, ContactError::kNone);
  ExpectVec3Equal(sixteen_call.result.position, {.x = 0, .y = 0, .z = kTolerance});
  ExpectStartFailure("SeventeenCoincidentFloors", {.error = ContactError::kStartsInNarrowGap, .solid_id = 17});
}

TEST(MoveParticle, DomainCornerWithCoincidentGroundAndWall_EndsAtCorner) {
  // A ground lying on side_bottom and a wall lying on side_left make five near surfaces, but only three directions:
  // coincident surfaces share one target, and {tol, tol, tol} is at the rest gap of all five.
  const MoveParticleCall call = RunMove("DomainCornerWithCoincidentGroundAndWall");
  ASSERT_EQ(call.error, ContactError::kNone);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kAtRestGapTolerance);
}

TEST(MoveParticle, DomainCornerWithNearlyCoincidentGroundAndWall_EndsAtCorner) {
  // Normals that differ by rounding share one target too, so the five surfaces still leave {tol, tol, tol}.
  const MoveParticleCall call = RunMove("DomainCornerWithNearlyCoincidentGroundAndWall");
  ASSERT_EQ(call.error, ContactError::kNone);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kAtRestGapTolerance);
}

TEST(MoveParticle, CuspStartIntoBallOnly_ResponseUsesNormalBeforeMove) {
  // In the cusp, halfway to the rest point, moving at 1 m/s straight into the ball and away from the floor. The
  // response uses the ball's normal at the start, so it stops the particle; the normal after the move, turned by
  // about 2 Sqrt(tol / R) (1 - f), would leave about 4.3e-5 m/s.
  const MoveParticleCall call = RunMove("CuspStartIntoBallOnly");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_LT(math::Length(call.result.velocity.for_this_step), 1e-12);
}

// --- The move off near surfaces and its solver ---

TEST(MoveToRestGapOfNearSurfaces, BallRestingOnFloorCusp_FiveIterates) {
  // R / tol = 2^29: the first iterate overshoots out along the floor, and the iterates are still after the fifth.
  const MoveParticleCase scene = FindMoveParticleCase("BallRestingOnFloorStartInCusp");
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  const RestGapMove move = MoveToRestGapOfNearSurfaces(inputs, 0, inputs.start_position);
  ASSERT_TRUE(move.is_found);
  EXPECT_EQ(move.iteration_count, 5);
}

TEST(MoveToRestGapOfNearSurfaces, DeepCuspAtLargestRadius_EightIteratesWithinOneTenThousandth) {
  // R / tol = 2^40: from 0.1 x* the iterates use all of kMaxRestGapIterations and end within 1e-4 of the rest point.
  const MoveParticleCase scene = FindMoveParticleCase("BallRestingOnFloorDeepStartTenth");
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  const RestGapMove move = MoveToRestGapOfNearSurfaces(inputs, 0, inputs.start_position);
  ASSERT_TRUE(move.is_found);
  EXPECT_EQ(move.iteration_count, kMaxRestGapIterations);
  const double rest_x = 2 * std::sqrt(1024 * kTolerance);  // m
  EXPECT_NEAR(move.moved_point.x / rest_x, 1, 1e-4);
}

TEST(MoveToRestGapOfNearSurfaces, OneNearSolid_MovesAlongItsNormalBitForBit) {
  // One near solid moves the point as MoveToRestGap always did: the same bits, no iterate.
  const MoveParticleCase scene = FindMoveParticleCase("StartsWithinToleranceMovingAway");
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  const RestGapMove move = MoveToRestGapOfNearSurfaces(inputs, 0, inputs.start_position);
  ASSERT_TRUE(move.is_found);
  EXPECT_EQ(move.iteration_count, 0);
  ExpectVec3Equal(
      move.moved_point,
      MoveToRestGap(inputs, 0, ComputeSolidPose(inputs.solids[0], 0), inputs.start_position)
  );
}

// A CorrectionTargets of the given normals and targets.
CorrectionTargets BuildCorrectionTargets(std::initializer_list<Vec3> normals, std::initializer_list<double> targets) {
  CorrectionTargets built;
  int index = 0;
  for (const Vec3 normal : normals) {
    built.normals[index] = normal;
    ++index;
  }
  built.count = index;
  index = 0;
  for (const double target : targets) {
    built.targets[index] = target;
    ++index;
  }
  return built;
}

// Fails the calling test unless the solver found a change within 1e-12 of the expected one on each axis, through
// used_count surfaces.
void ExpectCorrection(const SmallestCorrection& correction, Vec3 expected_change, int used_count) {
  ASSERT_TRUE(correction.is_found);
  ExpectVec3Near(correction.change, expected_change, 1e-12);
  EXPECT_EQ(correction.used_count, used_count);
}

// Fails the calling test unless the two multipliers of a pair are the expected ones within 1e-12.
void ExpectPairMultipliers(const SmallestCorrection& correction, double first_multiplier, double second_multiplier) {
  EXPECT_NEAR(correction.multipliers[0], first_multiplier, 1e-12);
  EXPECT_NEAR(correction.multipliers[1], second_multiplier, 1e-12);
}

// Solves for a position with the slack of the solver's closed-form tests, 2^-40.
SmallestCorrection SolveTestCorrection(const CorrectionTargets& targets) {
  return SolveSmallestCorrection(targets, 0x1p-40, CorrectedQuantity::kPosition);
}

// The targets of a shallow pit: three faces tilted 3 degrees from {0, 0, 1}, 120 degrees apart around z, each with
// target 1.
CorrectionTargets BuildShallowPitTargets() {
  const double tilt = 3 * std::numbers::pi / 180;  // rad
  CorrectionTargets pit;
  for (int face_index = 0; face_index < 3; ++face_index) {
    const double around = face_index * 2 * std::numbers::pi / 3;  // rad
    pit.normals[face_index] = {
        .x = std::sin(tilt) * std::cos(around),
        .y = std::sin(tilt) * std::sin(around),
        .z = std::cos(tilt),
    };
    pit.targets[face_index] = 1;
  }
  pit.count = 3;
  return pit;
}

TEST(SolveSmallestCorrection, OneTwoAndThreeSurfaces_MatchClosedForms) {
  const double root_three = std::numbers::sqrt3;
  const Vec3 floor_normal = {.x = 0, .y = 0, .z = 1};
  const Vec3 sixty_degree_normal = {.x = root_three / 2, .y = 0, .z = -0.5};
  // The 60 degree wedge, both targets 1: the pair, multipliers {2, 2}.
  const SmallestCorrection on_both =
      SolveTestCorrection(BuildCorrectionTargets({floor_normal, sixty_degree_normal}, {1, 1}));
  ExpectCorrection(on_both, {.x = root_three, .y = 0, .z = 1}, 2);
  ExpectPairMultipliers(on_both, 2, 2);
  // The same wedge, targets {1, 0}: still the pair, since the floor alone leaves the other face short.
  const SmallestCorrection floor_short =
      SolveTestCorrection(BuildCorrectionTargets({floor_normal, sixty_degree_normal}, {1, 0}));
  ExpectCorrection(floor_short, {.x = 1 / root_three, .y = 0, .z = 1}, 2);
  ExpectPairMultipliers(floor_short, 4.0 / 3, 2.0 / 3);
  // A right angle with targets {1, -1}: the floor alone meets both.
  ExpectCorrection(
      SolveTestCorrection(BuildCorrectionTargets({floor_normal, {.x = 1, .y = 0, .z = 0}}, {1, -1})),
      floor_normal,
      1
  );
  // The three axes: the triple.
  ExpectCorrection(
      SolveTestCorrection(
          BuildCorrectionTargets({{.x = 1, .y = 0, .z = 0}, {.x = 0, .y = 1, .z = 0}, floor_normal}, {1, 1, 1})
      ),
      {.x = 1, .y = 1, .z = 1},
      3
  );
  // An obtuse wedge: the pair, multipliers {2/3, 2/3}.
  const SmallestCorrection obtuse =
      SolveTestCorrection(BuildCorrectionTargets({floor_normal, {.x = root_three / 2, .y = 0, .z = 0.5}}, {1, 1}));
  ExpectCorrection(obtuse, {.x = 1 / root_three, .y = 0, .z = 1}, 2);
  ExpectPairMultipliers(obtuse, 2.0 / 3, 2.0 / 3);
  // A shallow pit: straight up, 1 / cos(3 degrees) = 1.0013723.
  ExpectCorrection(
      SolveTestCorrection(BuildShallowPitTargets()),
      {.x = 0, .y = 0, .z = 1 / std::cos(3 * std::numbers::pi / 180)},
      3
  );
}

TEST(SolveSmallestCorrection, PairBelowVelocityLimit_IsUsedForPositionOnly) {
  // 1 + c = 2^-41, below kVelocityPairLimitShare: only the pair meets both targets. A position uses it; a velocity,
  // whose refinements could not bring the residual within its allowance, finds no change.
  const double one_plus_dot = 0x1p-41;
  const CorrectionTargets pair = BuildCorrectionTargets(
      {
          {.x = 0, .y = 0, .z = 1},
          {.x = std::sqrt(one_plus_dot * (2 - one_plus_dot)), .y = 0, .z = -(1 - one_plus_dot)},
      },
      {1, 1}
  );
  const SmallestCorrection position_change = SolveSmallestCorrection(pair, 0x1p-21, CorrectedQuantity::kPosition);
  ASSERT_TRUE(position_change.is_found);
  EXPECT_EQ(position_change.used_count, 2);
  EXPECT_FALSE(SolveSmallestCorrection(pair, 0x1p-21, CorrectedQuantity::kVelocity).is_found);
}

TEST(SolveSmallestCorrection, TripleBelowVelocityLimit_IsUsedForPositionOnly) {
  // The normals {1, 0, 0}, {0, 1, 0} and Normalized({-1, -1, 2^-22}) have a triple product of about 2^-22.5, below
  // kVelocityTripleLimit. With the targets {1, 1, 0} no single surface and no pair meets all three: only the triple,
  // with a change far along z. A position uses it; a velocity finds no change.
  const CorrectionTargets triple = BuildCorrectionTargets(
      {{.x = 1, .y = 0, .z = 0}, {.x = 0, .y = 1, .z = 0}, math::Normalized({.x = -1, .y = -1, .z = 0x1p-22})},
      {1, 1, 0}
  );
  const SmallestCorrection position_change = SolveSmallestCorrection(triple, 0x1p-21, CorrectedQuantity::kPosition);
  ASSERT_TRUE(position_change.is_found);
  EXPECT_EQ(position_change.used_count, 3);
  EXPECT_FALSE(SolveSmallestCorrection(triple, 0x1p-21, CorrectedQuantity::kVelocity).is_found);
}

// Builds one pair of the residual test: n_a = {0, 0, 1} and n_b = {s cos p, s sin p, -(1 - one_plus_dot)} with
// s = Sqrt(one_plus_dot (2 - one_plus_dot)), so 1 + Dot(n_a, n_b) is one_plus_dot exactly. Draws: p = 2 pi Next, then
// the two targets, each Next.
CorrectionTargets BuildResidualTestPair(CaseGenerator& generator, double one_plus_dot) {
  const double around = 2 * std::numbers::pi * generator.Next();  // rad
  const double across = std::sqrt(one_plus_dot * (2 - one_plus_dot));
  const Vec3 second_normal = {
      .x = across * std::cos(around),
      .y = across * std::sin(around),
      .z = -(1 - one_plus_dot),
  };
  const double first_target = generator.Next();
  const double second_target = generator.Next();
  return BuildCorrectionTargets({{.x = 0, .y = 0, .z = 1}, second_normal}, {first_target, second_target});
}

// Computes the larger of a pair's two residuals |Dot(n_i, change) - target_i|.
double ComputeLargestPairResidual(const CorrectionTargets& pair, Vec3 change) {
  const double first_residual = std::abs(math::Dot(pair.normals[0], change) - pair.targets[0]);
  const double second_residual = std::abs(math::Dot(pair.normals[1], change) - pair.targets[1]);
  return std::max(first_residual, second_residual);
}

// Computes the larger residual of one pair solved for a position, in rounding units over 1 + c, and fails the calling
// test unless the pair was solved through both surfaces.
double ComputePairResidualShare(const CorrectionTargets& pair, double one_plus_dot) {
  const SmallestCorrection first_solve = SolveSmallestCorrection(pair, 0x1p-40, CorrectedQuantity::kPosition);
  EXPECT_TRUE(first_solve.is_found && first_solve.used_count == 2);
  return ComputeLargestPairResidual(pair, first_solve.change) * one_plus_dot / 0x1p-52;
}

// Fails the calling test unless the pair, solved for a velocity with a slack of 2^-21, needed at most one refinement to
// come within a quarter of that slack, 2^-23.
void ExpectOneRefinementWithinQuarterSlack(const CorrectionTargets& pair) {
  const SmallestCorrection velocity_change = SolveSmallestCorrection(pair, 0x1p-21, CorrectedQuantity::kVelocity);
  EXPECT_TRUE(velocity_change.is_found);
  EXPECT_LE(velocity_change.refinement_count, 1);
  EXPECT_LE(ComputeLargestPairResidual(pair, velocity_change.change), 0x1p-23);
}

TEST(SolveSmallestCorrection, PairResidualWithoutRefinement_IsAboutEpsOverOnePlusC) {
  // 3000 pairs at each 1 + c = 2^-40, 2^-28, 2^-16, 2^-4 (BuildResidualTestPair). Each target is met to within
  // 4 * 2^-52 / (1 + c) by the pair alone, a position's solve; with a velocity's slack of 2^-21, at most one refinement
  // brings it within a quarter of that slack, 2^-23, at 2^-40 (ComputePairResidualShare).
  CaseGenerator generator;
  for (const double one_plus_dot : {0x1p-40, 0x1p-28, 0x1p-16, 0x1p-4}) {
    double largest_residual_share = 0;  // In 2^-52 / (1 + c).
    for (int pair_index = 0; pair_index < 3000; ++pair_index) {
      const CorrectionTargets pair = BuildResidualTestPair(generator, one_plus_dot);
      largest_residual_share = std::max(largest_residual_share, ComputePairResidualShare(pair, one_plus_dot));
      if (one_plus_dot == 0x1p-40) {
        ExpectOneRefinementWithinQuarterSlack(pair);
      }
    }
    std::cout << "pair residual at 1 + c = " << one_plus_dot << ": at most " << largest_residual_share
              << " rounding units over 1 + c\n";
    EXPECT_LE(largest_residual_share, 4);
  }
}

// --- Generated paths ---

// What the generated paths gave, counted for the report.
struct GeneratedCounts {
  int paths = 0;
  int free_paths = 0;
  int hit_paths = 0;
  int several_hit_paths = 0;
  int too_many_hits = 0;
  int second_reacting_solid = 0;
  int slow_approach = 0;
  std::map<std::string_view, int> too_many_hits_per_scene;
};

// Prints, for the report, the scenes with any kTooManyHits and how many.
void PrintTooManyHitsPerScene(const std::map<std::string_view, int>& too_many_hits_per_scene) {
  std::cout << "too many hits per scene:";
  for (const auto& [scene_name, count] : too_many_hits_per_scene) {
    std::cout << " " << scene_name << " " << count << ";";
  }
  std::cout << "\n";
}

// Fails the calling test unless every solid is at least tol / 2 from the end position at its end pose, by
// ComputeDistanceToSolid itself (the generated paths commit to no plate).
void ExpectEndOutsideEverySolid(const SolidContactInputs& inputs, Vec3 end_position) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    const double end_distance =
        ComputeDistanceToSolid(solid, ComputeSolidPose(solid, inputs.step_duration), end_position).distance;
    EXPECT_GE(end_distance, kTolerance / 2) << "solid index " << solid_index;
  }
}

// Fails the calling test unless the end is on the same side of every plate as the start, a property the end check
// does not test: a plate is unbounded, so a particle can reach its other side only by passing through it.
void ExpectEndOnStartSideOfEveryPlate(const SolidContactInputs& inputs, Vec3 end_position) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    if (solid.shape != SolidShape::kPlate) {
      continue;
    }
    const SolidSide start_side = FindSolidSide(solid, ComputeSolidPose(solid, 0), inputs.start_position);
    const SolidSide end_side = FindSolidSide(solid, ComputeSolidPose(solid, inputs.step_duration), end_position);
    EXPECT_EQ(end_side, start_side) << "plate index " << solid_index;
  }
}

// Computes whether a point is at least 2 tol from every solid at the start of the step.
bool ComputeIsWellOutsideEverySolid(const SolidContactInputs& inputs, Vec3 point) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    if (!(ComputeDistanceToSolid(solid, ComputeSolidPose(solid, 0), point).distance >= 2 * kTolerance)) {
      return false;
    }
  }
  return true;
}

// Builds the inputs of one generated path in a scene, or none when 50 starts in a row are closer than 2 tol to a
// solid. Draw order: three NextSigned for the start offset (m, up to 1 per axis from the scene's start, drawn again for
// each refused start), three for for_this_step and three for for_next_step (m/s, up to 2 per axis from the scene's
// velocities), one Next for the friction (0 to 1).
bool BuildGeneratedPath(const SolidContactInputs& scene_inputs, CaseGenerator& generator, SolidContactInputs* inputs) {
  *inputs = scene_inputs;
  bool is_outside = false;
  for (int draw_index = 0; draw_index < 50 && !is_outside; ++draw_index) {
    inputs->start_position = scene_inputs.start_position
        + Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    is_outside = ComputeIsWellOutsideEverySolid(*inputs, inputs->start_position);
  }
  if (!is_outside) {
    return false;
  }
  const Vec3 this_offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const Vec3 next_offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  inputs->velocity.for_this_step = scene_inputs.velocity.for_this_step + 2 * this_offset;
  inputs->velocity.for_next_step = scene_inputs.velocity.for_next_step + 2 * next_offset;
  inputs->friction = generator.Next();
  return true;
}

// Fails the calling test unless a successful move of a generated path ended outside every solid with a sound record.
void ExpectGeneratedSuccess(const SolidContactInputs& inputs, const SolidContactResult& result) {
  ExpectEndOutsideEverySolid(inputs, result.position);
  ExpectEndOnStartSideOfEveryPlate(inputs, result.position);
  EXPECT_LE(result.hit_count, kMaxHitsPerStep);
  EXPECT_EQ(result.kind == ContactKind::kNone, result.hit_count == 0);
  EXPECT_TRUE(result.hit_fraction >= 0 && result.hit_fraction <= 1);
}

// Runs MoveParticle on one generated path, checks what it returned and counts it. A successful move must end at least
// tol / 2 from every solid; a failure may only be a close call that the scenes can produce, never the end check, since
// no solid in these scenes closes on a particle faster than the hits can push it.
void CheckGeneratedPath(const SolidContactInputs& inputs, std::string_view scene_name, GeneratedCounts& counts) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  ++counts.paths;
  if (error == ContactError::kNone) {
    ExpectGeneratedSuccess(inputs, result);
    counts.free_paths += result.hit_count == 0 ? 1 : 0;
    counts.hit_paths += result.hit_count > 0 ? 1 : 0;
    counts.several_hit_paths += result.hit_count > 1 ? 1 : 0;
    return;
  }
  EXPECT_EQ(error, ContactError::kTooCloseToCall) << scene_name;
  EXPECT_NE(failure.close_call, ContactCloseCall::kEndsTooCloseToSolid) << scene_name;
  counts.too_many_hits += failure.close_call == ContactCloseCall::kTooManyHits ? 1 : 0;
  if (failure.close_call == ContactCloseCall::kTooManyHits) {
    ++counts.too_many_hits_per_scene[scene_name];
  }
  counts.second_reacting_solid += failure.close_call == ContactCloseCall::kSecondReactingSolid ? 1 : 0;
  counts.slow_approach += failure.close_call == ContactCloseCall::kSlowApproachToTurningSolid ? 1 : 0;
}

TEST(MoveParticle, GeneratedPathsInEveryScene_EndOutsideEverySolid) {
  // For every scene of BuildMoveParticleCases whose solids are in id order: 1500 paths (BuildGeneratedPath), no
  // committed plate, the test contact tolerance. A scene with a step or a start out of range uses 0.2 s and
  // {0.5, 0.5, 0.5}.
  CaseGenerator generator;
  GeneratedCounts counts;
  constexpr int kPathsPerScene = 1500;
  for (const MoveParticleCase& scene : BuildMoveParticleCases()) {
    SolidContactInputs scene_inputs = ToSolidContactInputs(scene);
    scene_inputs.plate_side = {};
    scene_inputs.contact_tolerance = kTolerance;
    const bool is_scene_in_range = scene_inputs.step_duration > 0 && math::IsFinite(scene_inputs.start_position);
    if (!is_scene_in_range) {
      scene_inputs.step_duration = 0.2;
      scene_inputs.start_position = {.x = 0.5, .y = 0.5, .z = 0.5};
    }
    if (CheckSolidContactInputs(scene_inputs) != ContactError::kNone) {
      continue;  // The scene about solids out of id order.
    }
    // The rising floor under the overhanging upright squeezes many random paths for real: they end in kTooManyHits
    // until the slide rules name the squeeze kSqueezedBetweenSolids, so the scene stays out of this set.
    if (scene.name == "TiltedUprightOverRisingFloor") {
      continue;
    }
    for (int path_index = 0; path_index < kPathsPerScene; ++path_index) {
      SolidContactInputs inputs;
      if (BuildGeneratedPath(scene_inputs, generator, &inputs)) {
        CheckGeneratedPath(inputs, scene.name, counts);
      }
    }
  }
  // The counts keep the set honest: most paths succeed, many hit, and several hits are common.
  std::cout << "paths " << counts.paths << ", free " << counts.free_paths << ", hit " << counts.hit_paths
            << ", several hits " << counts.several_hit_paths << ", too many hits " << counts.too_many_hits
            << ", second reacting solid " << counts.second_reacting_solid << ", slow approach " << counts.slow_approach
            << "\n";
  PrintTooManyHitsPerScene(counts.too_many_hits_per_scene);
  EXPECT_GT(counts.paths, 40000);
  EXPECT_GT(counts.hit_paths, counts.paths / 4);
  EXPECT_GT(counts.several_hit_paths, counts.paths / 60);
}

// Where a generated start near a solid lies, by its distance d to the chosen solid at time 0.
enum class NearStartKind : std::uint8_t {
  kWithinTolerance,  // 0 <= d < tol.
  kOnSurface,  // d = 0.
  kSlightlyInside,  // -tol <= d < -tol / 4.
};

// What the generated starts near a solid gave, counted for the report.
struct NearStartCounts {
  std::array<int, 3> starts_per_kind{};  // In NearStartKind order.
  int free_paths = 0;  // Success with kind kNone.
  int hit_paths = 0;  // Success with kind kHit.
  int started_inside_paths = 0;  // Success with kind kStartedInsideSolid.
  int too_many_hits = 0;
  int second_reacting_solid = 0;
  int slow_approach = 0;
  std::map<std::string_view, int> too_many_hits_per_scene;
};

// Computes whether a point is at least 4 tol from every solid but one at the start of the step.
bool ComputeIsFarFromOtherSolids(const SolidContactInputs& inputs, std::uint32_t near_solid_index, Vec3 point) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    const double distance = ComputeDistanceToSolid(solid, ComputeSolidPose(solid, 0), point).distance;
    if (solid_index != near_solid_index && !(distance >= 4 * kTolerance)) {
      return false;
    }
  }
  return true;
}

// Builds the inputs of one generated start near a solid of a scene, or none when the scene has no solid or 50 points
// in a row cannot be placed. Draw order: one Next for the solid (index floor(Next * solid_count)), one for the kind
// (floor(3 * Next) in NearStartKind order), one for the target distance (tol * Next within the tolerance, unused on
// the surface, -(0.25 + 0.75 * Next) tol inside); then, for each try, three NextSigned for a point up to 1 m per axis
// from the scene's start, moved along that solid's normal at time 0 to the target distance and kept when its distance
// to the solid is within tol / 8 of the target and every other solid is at least 4 tol away; then the velocities and
// the friction as BuildGeneratedPath draws them. The start overlap tolerance is 2 tol, so no start is too deep.
bool BuildGeneratedNearStart(
    const SolidContactInputs& scene_inputs,
    CaseGenerator& generator,
    SolidContactInputs* inputs,
    NearStartKind* start_kind
) {
  if (scene_inputs.solid_count == 0) {
    return false;
  }
  *inputs = scene_inputs;
  inputs->start_overlap_tolerance = 2 * kTolerance;
  const auto near_solid_index =
      static_cast<std::uint32_t>(std::floor(generator.Next() * static_cast<double>(scene_inputs.solid_count)));
  *start_kind = static_cast<NearStartKind>(static_cast<int>(std::floor(3 * generator.Next())));
  const double target_draw = generator.Next();
  double target_distance = 0;  // m, on the surface.
  if (*start_kind == NearStartKind::kWithinTolerance) {
    target_distance = kTolerance * target_draw;
  } else if (*start_kind == NearStartKind::kSlightlyInside) {
    target_distance = -(0.25 + 0.75 * target_draw) * kTolerance;
  }
  const Solid& near_solid = scene_inputs.solids[near_solid_index];
  const SolidPose start_pose = ComputeSolidPose(near_solid, 0);
  bool is_placed = false;
  for (int draw_index = 0; draw_index < 50 && !is_placed; ++draw_index) {
    const Vec3 point = scene_inputs.start_position
        + Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const DistanceToSolid at_point = ComputeDistanceToSolid(near_solid, start_pose, point);
    inputs->start_position = point + (target_distance - at_point.distance) * at_point.normal_out_of_solid;
    const double start_distance = ComputeDistanceToSolid(near_solid, start_pose, inputs->start_position).distance;
    is_placed = std::abs(start_distance - target_distance) <= kTolerance / 8
        && ComputeIsFarFromOtherSolids(*inputs, near_solid_index, inputs->start_position);
  }
  if (!is_placed) {
    return false;
  }
  const Vec3 this_offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const Vec3 next_offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  inputs->velocity.for_this_step = scene_inputs.velocity.for_this_step + 2 * this_offset;
  inputs->velocity.for_next_step = scene_inputs.velocity.for_next_step + 2 * next_offset;
  inputs->friction = generator.Next();
  return true;
}

// Fails the calling test unless the end is on the side of every plate the start rules leave the particle on: the
// start's side, or, from inside the thickness, the side of the nearer face.
void ExpectEndOnNearerSideOfEveryPlate(const SolidContactInputs& inputs, Vec3 end_position) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    if (solid.shape != SolidShape::kPlate) {
      continue;
    }
    const SolidPose start_pose = ComputeSolidPose(solid, 0);
    SolidSide start_side = FindSolidSide(solid, start_pose, inputs.start_position);
    if (start_side == SolidSide::kNoPlate) {
      const SolidFace nearer_face = ComputeDistanceToSolid(solid, start_pose, inputs.start_position).face;
      start_side = nearer_face == SolidFace::kHighZ ? SolidSide::kAlongNormal : SolidSide::kAgainstNormal;
    }
    const SolidSide end_side = FindSolidSide(solid, ComputeSolidPose(solid, inputs.step_duration), end_position);
    EXPECT_EQ(end_side, start_side) << "plate index " << solid_index;
  }
}

// Fails the calling test unless a successful move from a generated start near a solid ended outside every solid, on
// the side of every plate the start rules leave it on, with a sound record.
void ExpectNearStartSuccess(
    const SolidContactInputs& inputs,
    const SolidContactResult& result,
    NearStartKind start_kind,
    std::string_view scene_name
) {
  ExpectEndOutsideEverySolid(inputs, result.position);
  ExpectEndOnNearerSideOfEveryPlate(inputs, result.position);
  EXPECT_EQ(result.kind == ContactKind::kNone, result.hit_count == 0) << scene_name;
  // A start inside the solid is always a hit at 0 that sets the kind.
  if (start_kind == NearStartKind::kSlightlyInside) {
    EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid) << scene_name;
  }
}

// Runs MoveParticle on one generated start near a solid, checks what it returned and counts it. A success ends at
// least tol / 2 from every solid; a failure is a close call that the scenes can produce, never the end check, since
// the start rules put every start at the rest gap, and never a start too deep.
void CheckGeneratedNearStart(
    const SolidContactInputs& inputs,
    NearStartKind start_kind,
    std::string_view scene_name,
    NearStartCounts& counts
) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  ++counts.starts_per_kind.at(static_cast<std::size_t>(start_kind));
  if (error == ContactError::kNone) {
    ExpectNearStartSuccess(inputs, result, start_kind, scene_name);
    counts.free_paths += result.kind == ContactKind::kNone ? 1 : 0;
    counts.hit_paths += result.kind == ContactKind::kHit ? 1 : 0;
    counts.started_inside_paths += result.kind == ContactKind::kStartedInsideSolid ? 1 : 0;
    return;
  }
  EXPECT_EQ(error, ContactError::kTooCloseToCall) << scene_name;
  EXPECT_NE(failure.close_call, ContactCloseCall::kEndsTooCloseToSolid) << scene_name;
  counts.too_many_hits += failure.close_call == ContactCloseCall::kTooManyHits ? 1 : 0;
  if (failure.close_call == ContactCloseCall::kTooManyHits) {
    ++counts.too_many_hits_per_scene[scene_name];
  }
  counts.second_reacting_solid += failure.close_call == ContactCloseCall::kSecondReactingSolid ? 1 : 0;
  counts.slow_approach += failure.close_call == ContactCloseCall::kSlowApproachToTurningSolid ? 1 : 0;
}

// Prints the counts of the generated starts near a solid for the report, and fails the calling test unless each kind
// of start is common and the moves end free, with a hit and started inside, each often.
void ExpectNearStartCounts(const NearStartCounts& counts) {
  const std::array<int, 3>& starts = counts.starts_per_kind;
  std::cout << "near starts: within tolerance " << starts.at(0) << ", on surface " << starts.at(1)
            << ", slightly inside " << starts.at(2) << "; free " << counts.free_paths << ", hit " << counts.hit_paths
            << ", started inside " << counts.started_inside_paths << ", too many hits " << counts.too_many_hits
            << ", second reacting solid " << counts.second_reacting_solid << ", slow approach " << counts.slow_approach
            << "\n";
  PrintTooManyHitsPerScene(counts.too_many_hits_per_scene);
  EXPECT_GT(starts.at(0), 15000);
  EXPECT_GT(starts.at(1), 15000);
  EXPECT_GT(starts.at(2), 15000);
  EXPECT_GT(counts.free_paths, 5000);
  EXPECT_GT(counts.hit_paths, 10000);
  EXPECT_GT(counts.started_inside_paths, 10000);
}

// Builds the inputs a generated set draws around in one scene: no committed plate, the test contact tolerance, and 0.2
// s and {0.5, 0.5, 0.5} for a scene whose step or start is out of range. Says false for a scene the generated sets
// leave out: the one about solids out of id order, and the squeezing scene of
// GeneratedPathsInEveryScene_EndOutsideEverySolid.
bool BuildGeneratedSceneInputs(const MoveParticleCase& scene, SolidContactInputs* scene_inputs) {
  *scene_inputs = ToSolidContactInputs(scene);
  scene_inputs->plate_side = {};
  scene_inputs->contact_tolerance = kTolerance;
  const bool is_scene_in_range = scene_inputs->step_duration > 0 && math::IsFinite(scene_inputs->start_position);
  if (!is_scene_in_range) {
    scene_inputs->step_duration = 0.2;
    scene_inputs->start_position = {.x = 0.5, .y = 0.5, .z = 0.5};
  }
  return CheckSolidContactInputs(*scene_inputs) == ContactError::kNone && scene.name != "TiltedUprightOverRisingFloor";
}

TEST(MoveParticle, GeneratedStartsNearSolidsInEveryScene_EndOutsideEverySolid) {
  // For every scene of BuildMoveParticleCases with a solid, in id order: 1000 starts (BuildGeneratedNearStart) within
  // the tolerance of, on, or slightly inside one solid and far from the others, no committed plate, the test contact
  // tolerance. A scene with a step or a start out of range uses 0.2 s and {0.5, 0.5, 0.5}; the squeezing scene is out,
  // as in GeneratedPathsInEveryScene_EndOutsideEverySolid. Its kTooManyHits come from the three turning-box scenes
  // (every start: no touching rule until plan Task 13), the octagon room (100 m/s), and paths that run into the vertex
  // of the acute wedge scenes (plan Task 12); its kSecondReactingSolid from the three scenes with two reacting solids.
  CaseGenerator generator;
  NearStartCounts counts;
  constexpr int kStartsPerScene = 1000;
  for (const MoveParticleCase& scene : BuildMoveParticleCases()) {
    SolidContactInputs scene_inputs;
    const bool is_usable_scene = BuildGeneratedSceneInputs(scene, &scene_inputs);
    for (int start_index = 0; is_usable_scene && start_index < kStartsPerScene; ++start_index) {
      SolidContactInputs inputs;
      NearStartKind start_kind = NearStartKind::kOnSurface;
      if (BuildGeneratedNearStart(scene_inputs, generator, &inputs, &start_kind)) {
        CheckGeneratedNearStart(inputs, start_kind, scene.name, counts);
      }
    }
  }
  ExpectNearStartCounts(counts);
}

// One path of the parallel set: its two flat solids and its inputs, which point at them.
struct ParallelPath {
  std::array<Solid, 2> solids{};
  SolidContactInputs inputs{};
};

// Builds one path of the parallel set, or none when its velocity approaches the normal slower than 0.05 m/s: two
// parallel fixed flat solids, both reacting, the second delta = (0.05 + 0.9 * Next) tol beyond the first along a
// shared random normal n (three draws, NextSigned, NextSigned and 0.2 + 0.8 * Next, drawn again while shorter than
// 0.1); a base point 0 or (0.001, 0.3 or 1 m, by floor(3 * Next)) times (0.5 + Next) on each axis when Next < 0.5; a
// velocity of three NextSigned times 3 m/s turned to approach n; a start Next * (tol - delta) beyond the second, so
// within tol of both and inside neither; T of 0.01, 0.1, 0.5 or 1 s by floor(4 * Next); friction 0. Draws in that
// order. inputs.solids is left for the caller.
bool BuildParallelPath(CaseGenerator& generator, ParallelPath* path) {
  Vec3 direction{};
  while (math::Length(direction) < 0.1) {
    direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = 0.2 + 0.8 * generator.Next()};
  }
  const Vec3 normal = math::Normalized(direction);
  const double delta = (0.05 + 0.9 * generator.Next()) * kTolerance;  // m
  const bool has_offset = generator.Next() < 0.5;
  const std::array<double, 3> offsets = {0.001, 0.3, 1.0};  // m
  const double offset = offsets.at(static_cast<std::size_t>(std::floor(3 * generator.Next())));
  const double offset_scale = 0.5 + generator.Next();
  const double base_axis = has_offset ? offset * offset_scale : 0;  // m
  const Vec3 base = {.x = base_axis, .y = base_axis, .z = base_axis};
  Vec3 velocity = 3 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const double start_share = generator.Next();
  const std::array<double, 4> step_durations = {0.01, 0.1, 0.5, 1.0};  // s
  const double step_duration = step_durations.at(static_cast<std::size_t>(std::floor(4 * generator.Next())));
  if (math::Dot(velocity, normal) >= 0) {
    velocity = velocity - 2 * math::Dot(velocity, normal) * normal;
  }
  if (std::abs(math::Dot(velocity, normal)) < 0.05) {
    return false;
  }
  path->solids.at(0) = move_particle_cases::BuildFlatSolid(base, normal, move_particle_cases::kNoMotion, 1);
  path->solids.at(1) =
      move_particle_cases::BuildFlatSolid(base + delta * normal, normal, move_particle_cases::kNoMotion, 2);
  path->solids.at(0).reports_reaction = true;
  path->solids.at(1).reports_reaction = true;
  path->inputs.solid_count = 2;
  path->inputs.contact_tolerance = kTolerance;
  path->inputs.step_duration = step_duration;
  path->inputs.start_overlap_tolerance = 2 * kTolerance;
  path->inputs.start_position = base + (delta + start_share * (kTolerance - delta)) * normal;
  path->inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  return true;
}

TEST(MoveParticle, GeneratedParallelSolids_RoundingNeverHitsTheSecond) {
  // The review's set, 20,000 draws (BuildParallelPath). The start moves to the second solid's rest gap, a few rounding
  // units on either side of it; the first solid responds, and the second is moved silently. Any further hit, or a
  // second reacting solid, is rounding. Until plan decision P19 the start was drawn between the two faces, inside the
  // second; each solid is now classified at the start as given, so such a start is a real start inside the second,
  // with its own response.
  CaseGenerator generator;
  int paths = 0;
  int spurious_results = 0;
  int other_failures = 0;
  for (int draw_index = 0; draw_index < 20000; ++draw_index) {
    ParallelPath path;
    if (!BuildParallelPath(generator, &path)) {
      continue;
    }
    path.inputs.solids = path.solids.data();
    SolidContactResult result;
    ContactFailure failure;
    const ContactError error = MoveParticle(path.inputs, &result, &failure);
    ++paths;
    const bool is_repeat_hit = error == ContactError::kNone && result.hit_count != 1;
    const bool is_second_reacting = failure.close_call == ContactCloseCall::kSecondReactingSolid;
    spurious_results += is_repeat_hit || is_second_reacting ? 1 : 0;
    other_failures += error != ContactError::kNone && !is_second_reacting ? 1 : 0;
  }
  std::cout << "parallel solids " << paths << ", spurious results " << spurious_results << ", other failures "
            << other_failures << "\n";
  EXPECT_EQ(spurious_results, 0);
  EXPECT_EQ(other_failures, 0);
  EXPECT_GT(paths, 15000);
}

// What the generated two-surface starts gave, counted for the report.
struct TwoSurfaceCounts {
  int wedge_starts = 0;
  int acute_wedge_starts = 0;  // Normals more than a right angle apart: Dot(n_A, n_B) < 0.
  int corner_starts = 0;
  int free_paths = 0;  // Success with kind kNone.
  int hit_paths = 0;  // Success with kind kHit.
  int started_inside_paths = 0;  // Success with kind kStartedInsideSolid.
  int acute_too_many_hits = 0;  // kTooManyHits in an acute wedge, from a velocity that approaches a face.
  int acute_too_many_hits_approaching_both = 0;  // The part of those whose velocity approaches both faces.
  int failures = 0;  // Every other failure.
};

// Draws a velocity for a two-surface start: slow, up to tol / T per axis, when Next < 0.5, else up to 2 m/s per axis
// (four draws).
Vec3 DrawTwoSurfaceVelocity(CaseGenerator& generator, double step_duration) {
  const double speed = generator.Next() < 0.5 ? kTolerance / step_duration : 2.0;  // m/s
  return speed * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
}

// Builds a wedge: two fixed flat solids through a point q within 1 m of the origin per axis, normals nA (three
// NextSigned, drawn again while shorter than 0.1) and nB = cos(phi) nA + sin(phi) t, phi from 5 to 175 degrees (a
// wedge angle of 175 down to 5) and t a random direction made square to nA (three NextSigned, drawn again while that
// part is shorter than 0.1). The start is at distances dA and dB, each (2 * Next - 1) tol, and c = NextSigned m along
// nA x t. Draw order: nA, t, phi, q, dA, dB, c.
std::array<Solid, 2> BuildTwoSurfaceWedge(CaseGenerator& generator, Vec3* start_position) {
  Vec3 first_direction{};
  while (math::Length(first_direction) < 0.1) {
    first_direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  const Vec3 first_normal = math::Normalized(first_direction);
  Vec3 square_direction{};
  while (math::Length(square_direction) < 0.1) {
    const Vec3 draw = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    square_direction = draw - math::Dot(first_normal, draw) * first_normal;
  }
  const Vec3 square_unit = math::Normalized(square_direction);
  const double normal_angle = (5 + 170 * generator.Next()) * std::numbers::pi / 180;  // rad
  const Vec3 second_normal = std::cos(normal_angle) * first_normal + std::sin(normal_angle) * square_unit;
  const Vec3 point = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const double first_distance = generator.NextSigned() * kTolerance;  // m
  const double second_distance = generator.NextSigned() * kTolerance;  // m
  const double along_line = generator.NextSigned();  // m
  const double square_share =
      (second_distance - std::cos(normal_angle) * first_distance) / std::sin(normal_angle);  // m
  const Vec3 line_direction = {
      .x = first_normal.y * square_unit.z - first_normal.z * square_unit.y,
      .y = first_normal.z * square_unit.x - first_normal.x * square_unit.z,
      .z = first_normal.x * square_unit.y - first_normal.y * square_unit.x,
  };
  *start_position = point + first_distance * first_normal + square_share * square_unit + along_line * line_direction;
  const SolidMotion no_motion = move_particle_cases::kNoMotion;
  return {
      move_particle_cases::BuildFlatSolid(point, first_normal, no_motion, 1),
      move_particle_cases::BuildFlatSolid(point, math::Normalized(second_normal), no_motion, 2),
  };
}

// Builds a start in a corner of the unit domain box with an open top: two or three near sides (floor(2 * Next) + 2),
// the axes x, y, z in turn when three, else the two from floor(3 * Next) left out; per near axis the low or high side
// by Next < 0.5 (side_bottom for z, the top being open) and a distance (2 * Next - 1) tol; every other axis at
// 0.2 + 0.6 * Next m.
Vec3 BuildDomainCornerStart(CaseGenerator& generator) {
  const bool has_three_sides = generator.Next() < 0.5;
  const int axis_left_out = has_three_sides ? -1 : static_cast<int>(std::floor(3 * generator.Next()));
  std::array<double, 3> coordinates{};
  for (int axis_index = 0; axis_index < 3; ++axis_index) {
    const bool is_low_side = generator.Next() < 0.5 || axis_index == 2;
    const double distance = generator.NextSigned() * kTolerance;  // m, into the box.
    const double inner = 0.2 + 0.6 * generator.Next();  // m
    const double near_side = is_low_side ? distance : 1 - distance;  // m
    coordinates.at(static_cast<std::size_t>(axis_index)) = axis_index == axis_left_out ? inner : near_side;
  }
  return {.x = coordinates.at(0), .y = coordinates.at(1), .z = coordinates.at(2)};
}

// Counts the fixed flat solids of a wedge a velocity approaches: a negative dot with the normal.
int ComputeApproachedFaceCount(const SolidContactInputs& inputs, Vec3 velocity) {
  int approached_count = 0;
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Mat3 axes = inputs.solids[solid_index].pose_at_step_start.axes;
    const Vec3 normal = {.x = axes.xz, .y = axes.yz, .z = axes.zz};
    approached_count += math::Dot(normal, velocity) < 0 ? 1 : 0;
  }
  return approached_count;
}

// Runs MoveParticle on one two-surface start, checks it and counts it: a success ends at least tol / 2 from every
// solid; the only failure is kTooManyHits in an acute wedge from a velocity that approaches a face: the response on one
// face turns it into the other, and it hits one and then the other until the slide over several surfaces (plan Task
// 12) ends it.
void CheckTwoSurfaceStart(const SolidContactInputs& inputs, bool is_acute_wedge, TwoSurfaceCounts& counts) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  if (error != ContactError::kNone) {
    const bool is_too_many_hits =
        error == ContactError::kTooCloseToCall && failure.close_call == ContactCloseCall::kTooManyHits;
    const int approached_count = ComputeApproachedFaceCount(inputs, inputs.velocity.for_this_step);
    const bool is_expected = is_acute_wedge && is_too_many_hits && approached_count >= 1;
    counts.acute_too_many_hits_approaching_both += is_expected && approached_count == 2 ? 1 : 0;
    EXPECT_TRUE(is_expected) << "error " << static_cast<int>(error) << ", close call "
                             << static_cast<int>(failure.close_call) << ", acute " << is_acute_wedge;
    counts.acute_too_many_hits += is_expected ? 1 : 0;
    counts.failures += is_expected ? 0 : 1;
    return;
  }
  ExpectEndOutsideEverySolid(inputs, result.position);
  counts.free_paths += result.kind == ContactKind::kNone ? 1 : 0;
  counts.hit_paths += result.kind == ContactKind::kHit ? 1 : 0;
  counts.started_inside_paths += result.kind == ContactKind::kStartedInsideSolid ? 1 : 0;
}

TEST(MoveParticle, GeneratedTwoSurfaceStarts_SucceedDownToFiveDegrees) {
  // 30,000 draws, by Next < 0.5 a wedge (BuildTwoSurfaceWedge, wedge angles 5 to 175 degrees) or a domain corner
  // (BuildDomainCornerStart); then T = 0.01 + 0.09 * Next s, short enough that a fast particle stays near its corner,
  // away from the far sides, both velocities from one DrawTwoSurfaceVelocity, friction Next. Start overlap tolerance
  // 2 tol. Every start is within tol of, on, or slightly inside two or three surfaces. Right and obtuse wedges and
  // domain corners never fail; acute wedges never fail at the start (no kStartsInsideSolid, kStartsInNarrowGap or
  // kEndsTooCloseToSolid), and fail only with kTooManyHits from a velocity that approaches a face
  // (CheckTwoSurfaceStart): 2246 of 7497 acute draws, 30 % as the model of plan decision P19 gives.
  CaseGenerator generator;
  TwoSurfaceCounts counts;
  const std::array domain_solids = {move_particle_cases::BuildUnitDomainBox(1)};
  for (int draw_index = 0; draw_index < 30000; ++draw_index) {
    const bool is_wedge = generator.Next() < 0.5;
    SolidContactInputs inputs;
    inputs.contact_tolerance = kTolerance;
    inputs.start_overlap_tolerance = 2 * kTolerance;
    std::array<Solid, 2> wedge_solids{};
    bool is_acute_wedge = false;
    if (is_wedge) {
      wedge_solids = BuildTwoSurfaceWedge(generator, &inputs.start_position);
      inputs.solids = wedge_solids.data();
      inputs.solid_count = 2;
      ++counts.wedge_starts;
      const Mat3 first_axes = wedge_solids.at(0).pose_at_step_start.axes;
      const Mat3 second_axes = wedge_solids.at(1).pose_at_step_start.axes;
      is_acute_wedge =
          first_axes.xz * second_axes.xz + first_axes.yz * second_axes.yz + first_axes.zz * second_axes.zz < 0;
      counts.acute_wedge_starts += is_acute_wedge ? 1 : 0;
    } else {
      inputs.start_position = BuildDomainCornerStart(generator);
      inputs.solids = domain_solids.data();
      inputs.solid_count = 1;
      ++counts.corner_starts;
    }
    inputs.step_duration = 0.01 + 0.09 * generator.Next();
    const Vec3 velocity = DrawTwoSurfaceVelocity(generator, inputs.step_duration);
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    inputs.friction = generator.Next();
    CheckTwoSurfaceStart(inputs, is_acute_wedge, counts);
  }
  std::cout << "two-surface starts: wedges " << counts.wedge_starts << " (acute " << counts.acute_wedge_starts
            << "), domain corners " << counts.corner_starts << "; free " << counts.free_paths << ", hit "
            << counts.hit_paths << ", started inside " << counts.started_inside_paths << ", acute too many hits "
            << counts.acute_too_many_hits << " (approaching both faces " << counts.acute_too_many_hits_approaching_both
            << "), other failures " << counts.failures << "\n";
  EXPECT_EQ(counts.failures, 0);
  EXPECT_GT(counts.free_paths, 1000);
  EXPECT_GT(counts.hit_paths, 1000);
  EXPECT_GT(counts.started_inside_paths, 1000);
}

// --- Generated tangent contacts ---

// The scenes of the generated tangent contacts, in the order of the set.
enum class TangentScene : std::uint8_t {
  kBallOnFloor,  // A ball resting on the fixed floor through the contact point.
  kBallInDomainCorner,  // A ball touching the three low sides of the unit domain box.
  kTwoBalls,  // Two balls touching along x.
  kBallOnTiltedFlatSolid,  // A ball resting on a fixed flat solid tilted about y.
};

// One start of the tangent set: its solids, its inputs (which point at them) and what was drawn for the counts.
struct TangentStart {
  std::array<Solid, 2> solids{};
  SolidContactInputs inputs{};
  int velocity_class = 0;  // 0 at rest, 1 slow, 2 falling, 3 toward the contact, 4 random.
  Vec3 first_normal{};  // No unit, world. The floor's, the side's or the first ball's normal at the start.
  Vec3 second_normal{};  // No unit, world. The second ball's normal at the start.
};

// Builds one start of the tangent set in one scene (P11 generator). Draws, in this order and all of them for every
// scene: R = 0.05 + 0.45 Next and the second ball's radius the same; the contact's offset {NextSigned, NextSigned} m;
// the radial direction's angle 2 pi Next; x = x* (0.02 + 0.98 Next), with x* = 2 Sqrt(R_eff tol) the rest point's
// distance and R_eff = R, or R R2 / (R + R2) for two balls; the height in the gap, (1.5 Next - 0.5) of the gap x^2 /
// (2 R_eff), so some starts are up to half the gap inside the floor (inside the first ball for two balls); T from
// {1e-3, 0.01, 0.1} s by floor(3 Next); the velocity class floor(5 Next), three NextSigned and one Next (at rest; slow,
// the three times tol / T; falling, {0, 0, -9.81 T}; toward the contact along the radial direction, 2 Next m/s; random,
// the three times 2 m/s); the scene's own Next (the corner ball's side by floor(3 Next), bottom, left or front; the
// tilt 0.6 Next rad). Both velocities equal, friction 0, start overlap tolerance 2 tol.
TangentStart BuildTangentStart(CaseGenerator& generator, TangentScene scene) {
  const double radius = 0.05 + 0.45 * generator.Next();  // m
  const double second_radius = 0.05 + 0.45 * generator.Next();  // m
  const Vec3 offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = 0};
  const double radial_angle = 2 * std::numbers::pi * generator.Next();  // rad
  const double distance_share = 0.02 + 0.98 * generator.Next();
  const double height_share = 1.5 * generator.Next() - 0.5;
  const std::array<double, 3> step_durations = {1e-3, 0.01, 0.1};  // s
  const double step_duration = step_durations.at(static_cast<std::size_t>(std::floor(3 * generator.Next())));
  const int velocity_class = static_cast<int>(std::floor(5 * generator.Next()));
  const Vec3 velocity_draw = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const double speed_share = generator.Next();
  const double scene_draw = generator.Next();
  const double cos_radial = std::cos(radial_angle);
  const double sin_radial = std::sin(radial_angle);

  TangentStart start;
  Vec3 contact{};  // m, world.
  Vec3 floor_normal{};  // No unit, world: out of the floor, the side or the first ball at the contact.
  Vec3 radial{};  // No unit, world: away from the contact, square to floor_normal.
  double effective_radius = radius;  // m
  switch (scene) {
    case TangentScene::kBallOnFloor:
    case TangentScene::kBallOnTiltedFlatSolid: {
      const double tilt = scene == TangentScene::kBallOnTiltedFlatSolid ? 0.6 * scene_draw : 0;  // rad
      floor_normal = {.x = std::sin(tilt), .y = 0, .z = std::cos(tilt)};
      const Vec3 along_tilt = {.x = std::cos(tilt), .y = 0, .z = -std::sin(tilt)};
      radial = cos_radial * along_tilt + sin_radial * Vec3{.x = 0, .y = 1, .z = 0};
      contact = offset;
      start.solids.at(0) =
          move_particle_cases::BuildFlatSolid(contact, floor_normal, move_particle_cases::kNoMotion, 1);
      start.solids.at(1) = move_particle_cases::BuildFixedSphere(contact + radius * floor_normal, radius, 2);
      break;
    }
    case TangentScene::kBallInDomainCorner: {
      const int side_index = static_cast<int>(std::floor(3 * scene_draw));
      const std::array<Vec3, 3> axes = {
          Vec3{.x = 0, .y = 0, .z = 1},
          Vec3{.x = 1, .y = 0, .z = 0},
          Vec3{.x = 0, .y = 1, .z = 0},
      };
      floor_normal = axes.at(static_cast<std::size_t>(side_index));
      radial = cos_radial * axes.at(static_cast<std::size_t>((side_index + 1) % 3))
          + sin_radial * axes.at(static_cast<std::size_t>((side_index + 2) % 3));
      const Vec3 center = {.x = radius, .y = radius, .z = radius};
      contact = center - radius * floor_normal;
      start.solids.at(0) = move_particle_cases::BuildUnitDomainBox(1);
      start.solids.at(1) = move_particle_cases::BuildFixedSphere(center, radius, 2);
      break;
    }
    case TangentScene::kTwoBalls: {
      effective_radius = radius * second_radius / (radius + second_radius);
      floor_normal = {.x = 1, .y = 0, .z = 0};
      radial = {.x = 0, .y = cos_radial, .z = sin_radial};
      contact = offset + Vec3{.x = radius, .y = 0, .z = 0};
      start.solids.at(0) = move_particle_cases::BuildFixedSphere(offset, radius, 1);
      start.solids.at(1) = move_particle_cases::BuildFixedSphere(
          offset + Vec3{.x = radius + second_radius, .y = 0, .z = 0},
          second_radius,
          2
      );
      break;
    }
  }
  const double rest_distance = 2 * std::sqrt(effective_radius * kTolerance);  // m, x*.
  const double contact_distance = distance_share * rest_distance;  // m
  const double gap = contact_distance * contact_distance / (2 * effective_radius);  // m
  // For two balls the gap is measured from the first ball's surface, which lies x^2 / (2 R) behind the contact plane.
  const double first_surface_height =
      scene == TangentScene::kTwoBalls ? -contact_distance * contact_distance / (2 * radius) : 0;  // m
  const Vec3 start_position =
      contact + contact_distance * radial + (first_surface_height + height_share * gap) * floor_normal;

  Vec3 velocity{};  // m/s
  if (velocity_class == 1) {
    velocity = kTolerance / step_duration * velocity_draw;
  } else if (velocity_class == 2) {
    velocity = {.x = 0, .y = 0, .z = -9.81 * step_duration};
  } else if (velocity_class == 3) {
    velocity = -2 * speed_share * radial;
  } else if (velocity_class == 4) {
    velocity = 2 * velocity_draw;
  }
  start.velocity_class = velocity_class;
  start.inputs.start_position = start_position;
  start.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  start.inputs.solid_count = 2;
  start.inputs.step_duration = step_duration;
  start.inputs.contact_tolerance = kTolerance;
  start.inputs.start_overlap_tolerance = 2 * kTolerance;
  const Solid& first_solid = start.solids.at(0);
  const Solid& second_solid = start.solids.at(1);
  start.first_normal = scene == TangentScene::kTwoBalls
      ? ComputeDistanceToSolid(first_solid, first_solid.pose_at_step_start, start_position).normal_out_of_solid
      : floor_normal;
  start.second_normal =
      ComputeDistanceToSolid(second_solid, second_solid.pose_at_step_start, start_position).normal_out_of_solid;
  return start;
}

// What the generated tangent starts of one scene gave, counted for the report.
struct TangentCounts {
  int successes = 0;
  int too_many_hits = 0;  // From a velocity that approaches one of the two contact surfaces.
  int too_many_hits_approaching_both = 0;  // The part of those whose velocity approaches both.
  std::array<int, 5> too_many_hits_per_class{};  // In the velocity classes of TangentStart.
  int other_failures = 0;
};

// Runs MoveParticle on one tangent start, checks it and counts it: a success ends at least tol / 2 from every solid;
// the only failure is kTooManyHits from a velocity that approaches one of the two contact surfaces.
void CheckTangentStart(TangentScene scene, const TangentStart& start, TangentCounts& counts) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(start.inputs, &result, &failure);
  if (error == ContactError::kNone) {
    ExpectEndOutsideEverySolid(start.inputs, result.position);
    ++counts.successes;
    return;
  }
  const Vec3 velocity = start.inputs.velocity.for_this_step;
  const bool approaches_first = math::Dot(start.first_normal, velocity) < 0;
  const bool approaches_second = math::Dot(start.second_normal, velocity) < 0;
  const bool is_too_many_hits =
      error == ContactError::kTooCloseToCall && failure.close_call == ContactCloseCall::kTooManyHits;
  const bool is_expected = is_too_many_hits && (approaches_first || approaches_second);
  EXPECT_TRUE(is_expected) << "scene " << static_cast<int>(scene) << ", error " << static_cast<int>(error)
                           << ", close call " << static_cast<int>(failure.close_call);
  if (!is_expected) {
    ++counts.other_failures;
    return;
  }
  ++counts.too_many_hits;
  counts.too_many_hits_approaching_both += approaches_first && approaches_second ? 1 : 0;
  ++counts.too_many_hits_per_class.at(static_cast<std::size_t>(start.velocity_class));
}

TEST(MoveParticle, GeneratedTangentContactStarts_Succeed) {
  // 2000 starts per scene (BuildTangentStart), the scenes in TangentScene order, starts in the cusp of each tangent
  // contact from 0.02 to 1 times the rest point's distance. Every success ends at least tol / 2 from every solid. The
  // start never fails; the only failure is kTooManyHits from a velocity that approaches one of the two surfaces and
  // then bounces between them into the cusp, which the slide over several surfaces (plan Task 12) ends.
  CaseGenerator generator;
  const std::array scenes = {
      TangentScene::kBallOnFloor,
      TangentScene::kBallInDomainCorner,
      TangentScene::kTwoBalls,
      TangentScene::kBallOnTiltedFlatSolid,
  };
  for (const TangentScene scene : scenes) {
    TangentCounts counts;
    for (int draw_index = 0; draw_index < 2000; ++draw_index) {
      TangentStart start = BuildTangentStart(generator, scene);
      start.inputs.solids = start.solids.data();
      CheckTangentStart(scene, start, counts);
    }
    const std::array<int, 5>& per_class = counts.too_many_hits_per_class;
    std::cout << "tangent scene " << static_cast<int>(scene) << ": success " << counts.successes << ", too many hits "
              << counts.too_many_hits << " (approaching both " << counts.too_many_hits_approaching_both
              << "; per velocity class " << per_class.at(0) << " " << per_class.at(1) << " " << per_class.at(2) << " "
              << per_class.at(3) << " " << per_class.at(4) << "), other failures " << counts.other_failures << "\n";
    EXPECT_EQ(counts.other_failures, 0);
  }
}

// --- The touched surfaces in the sweep ---

// The sweep that follows the first hit of the scene ObliqueFlatSolidRoundingAtHit: the hit position and the response's
// velocity, as MoveParticle computes them, with the allowance of that response.
struct SweepAfterObliqueHit {
  MoveParticleCase scene;
  Sweep sweep;
  TouchedSurface touched_surface;
};

// Builds the sweep after the first hit of ObliqueFlatSolidRoundingAtHit.
SweepAfterObliqueHit BuildSweepAfterObliqueHit() {
  SweepAfterObliqueHit after_hit;
  after_hit.scene = FindMoveParticleCase("ObliqueFlatSolidRoundingAtHit");
  const SolidContactInputs inputs = ToSolidContactInputs(after_hit.scene);
  const Vec3 velocity = inputs.velocity.for_this_step;
  FirstHit first_hit;
  EXPECT_EQ(
      FindFirstHit(inputs, inputs.start_position, velocity, 0, inputs.step_duration, &first_hit),
      ContactError::kNone
  );
  const Vec3 normal = first_hit.at_hit.normal_out_of_solid;
  after_hit.sweep = {
      .start = inputs.start_position + velocity * first_hit.elapsed,
      .velocity_for_this_step = ComputeContactResponse(velocity, {}, normal, 0),
      .start_elapsed = first_hit.elapsed,
      .duration = inputs.step_duration - first_hit.elapsed,
  };
  after_hit.touched_surface = {
      .solid_index = 0,
      .normal_out_of_solid = normal,
      .approach_allowance = ComputeLargestApproachLeftAfterResponse(velocity, {}),
  };
  return after_hit;
}

TEST(FindFirstHitOfSweep, NoTouchedSurfaceAfterRoundedHit_HitsSameSolidAgain) {
  // The reason for the touched surfaces: without them the start, 2.8e-18 m outside the rest gap, is outside, and the
  // rounding-sized approach of the response makes the closed form hit the same solid again.
  const SweepAfterObliqueHit after_hit = BuildSweepAfterObliqueHit();
  FirstHit first_hit;
  const TouchedSurfaces no_touched_surfaces;
  const ContactError error = FindFirstHitOfSweep(
      ToSolidContactInputs(after_hit.scene),
      after_hit.sweep,
      no_touched_surfaces,
      &first_hit,
      nullptr
  );
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_EQ(first_hit.solid_index, 0U);
}

TEST(FindFirstHitOfSweep, TouchedSolidAfterRoundedHit_IsFree) {
  const SweepAfterObliqueHit after_hit = BuildSweepAfterObliqueHit();
  FirstHit first_hit;
  const ContactError error = FindFirstHitOfSweep(
      ToSolidContactInputs(after_hit.scene),
      after_hit.sweep,
      {.surfaces = &after_hit.touched_surface, .count = 1},
      &first_hit,
      nullptr
  );
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_EQ(first_hit.solid_index, kNoSolid);
}

TEST(FindFirstHitOfSweep, TouchedSolidReallyApproached_IsHitAtStart) {
  // The same start, now moving into the surface at 1e-9 m/s, far beyond the allowance: hit at 0.
  SweepAfterObliqueHit after_hit = BuildSweepAfterObliqueHit();
  after_hit.sweep.velocity_for_this_step =
      after_hit.sweep.velocity_for_this_step - 1e-9 * after_hit.touched_surface.normal_out_of_solid;
  FirstHit first_hit;
  const ContactError error = FindFirstHitOfSweep(
      ToSolidContactInputs(after_hit.scene),
      after_hit.sweep,
      {.surfaces = &after_hit.touched_surface, .count = 1},
      &first_hit,
      nullptr
  );
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
}

TEST(FindFirstHitOfSweep, TouchedSolidFartherThanSixteenthOfTolerance_TakesClosedForm) {
  // A start tol / 2 above the rest gap of the touched surface, approaching at 1 m/s: outside the touched window of
  // tol / 16, so the closed form finds the hit tol / 2 later, not at 0.
  SweepAfterObliqueHit after_hit = BuildSweepAfterObliqueHit();
  const Vec3 normal = after_hit.touched_surface.normal_out_of_solid;
  after_hit.sweep.start = after_hit.sweep.start + 0.5 * kTolerance * normal;
  after_hit.sweep.velocity_for_this_step = -1.0 * normal;
  FirstHit first_hit;
  const ContactError error = FindFirstHitOfSweep(
      ToSolidContactInputs(after_hit.scene),
      after_hit.sweep,
      {.surfaces = &after_hit.touched_surface, .count = 1},
      &first_hit,
      nullptr
  );
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 0.5 * kTolerance, kTolerance / 64);
}

// One path of the oblique set: its one flat solid and its inputs, which point at that solid.
struct ObliquePath {
  std::array<Solid, 1> solids{};
  SolidContactInputs inputs{};
};

// Builds one path of the oblique set: one flat solid of random normal through a random point within 2 m of the origin,
// fixed or moving at up to 3 m/s per axis; a start within 2 m per axis, at least 2 tol outside; both velocities up to
// 4 m/s per axis; T from 0.01 to 2 s; friction 0 to 1. Draw order: three NextSigned for the normal (drawn again while
// its length is below 0.1), three for the point, one Next for moving or fixed, three NextSigned for the linear
// velocity, three for the start (drawn again while closer than 2 tol), three for for_this_step, three for
// for_next_step, one Next for T, one for the friction. inputs.solids is left for the caller, since the path is returned
// by value.
ObliquePath BuildObliquePath(CaseGenerator& generator) {
  Vec3 normal{};
  while (math::Length(normal) < 0.1) {
    normal = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  const Vec3 point = 2 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const bool is_moving = generator.Next() < 0.5;
  const Vec3 linear_velocity =
      3 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const SolidMotion motion = {.linear_velocity = is_moving ? linear_velocity : Vec3{}, .angular_velocity = {}};
  ObliquePath path;
  path.solids.at(0) = move_particle_cases::BuildFlatSolid(point, math::Normalized(normal), motion, 1);
  path.inputs.solids = path.solids.data();
  path.inputs.solid_count = 1;
  path.inputs.contact_tolerance = kTolerance;
  do {
    path.inputs.start_position =
        2 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  } while (!ComputeIsWellOutsideEverySolid(path.inputs, path.inputs.start_position));
  path.inputs.velocity.for_this_step =
      4 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  path.inputs.velocity.for_next_step =
      4 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  path.inputs.step_duration = 0.01 + 1.99 * generator.Next();
  path.inputs.friction = generator.Next();
  return path;
}

TEST(MoveParticle, GeneratedObliqueFlatSolid_NeverHitsItAgain) {
  // The reviewer's set of 20,000 paths (BuildObliquePath). One solid that is hit once is never approached again, so
  // a second hit can only be rounding: none is allowed, and no kTooManyHits.
  CaseGenerator generator;
  int hit_paths = 0;
  int repeat_hits = 0;
  int too_many_hits = 0;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    ObliquePath path = BuildObliquePath(generator);
    path.inputs.solids = path.solids.data();
    SolidContactResult result;
    ContactFailure failure;
    const ContactError error = MoveParticle(path.inputs, &result, &failure);
    const bool is_too_many_hits = failure.close_call == ContactCloseCall::kTooManyHits;
    too_many_hits += is_too_many_hits ? 1 : 0;
    const bool is_success = error == ContactError::kNone;
    hit_paths += is_success && result.hit_count > 0 ? 1 : 0;
    repeat_hits += is_success && result.hit_count > 1 ? 1 : 0;
  }
  std::cout << "oblique paths 20000, hit " << hit_paths << ", repeat hits " << repeat_hits << ", too many hits "
            << too_many_hits << "\n";
  EXPECT_EQ(repeat_hits, 0);
  EXPECT_EQ(too_many_hits, 0);
  EXPECT_GT(hit_paths, 5000);
}

TEST(MoveParticle, GeneratedCylinderRimPaths_NeverHitItAgain) {
  // A fixed cylinder at the origin along z, radius 0.5 m and half height 1 m, and 5000 paths falling onto its high rim:
  // a start at radius 0.5 + 0.02 * NextSigned m, angle 2 pi * Next, height 1.5 + Next m, a velocity
  // {0.2 * NextSigned, 0.2 * NextSigned, -1 - Next} m/s, T 2 s, friction 0 (draws in that order). A closed-form rim hit
  // leaves the particle up to Sqrt(2) tol from the rim, in the rim corner, where the re-sweep's start rule tests the
  // approach; the cylinder is convex, so the tangent path cannot meet it again.
  const SolidMotion no_motion;
  const std::array solids = {
      ToSolid(Cylinder{.center = {}, .axis = {.x = 0, .y = 0, .z = 1}, .radius = 0.5, .half_height = 1}, no_motion, 1),
  };
  CaseGenerator generator;
  int hit_paths = 0;
  int repeat_hits = 0;
  for (int path_index = 0; path_index < 5000; ++path_index) {
    const double radius = 0.5 + 0.02 * generator.NextSigned();  // m
    const double angle = 2 * std::numbers::pi * generator.Next();  // rad
    const double height = 1.5 + generator.Next();  // m
    SolidContactInputs inputs;
    inputs.solids = solids.data();
    inputs.solid_count = 1;
    inputs.contact_tolerance = kTolerance;
    inputs.step_duration = 2;
    inputs.start_position = {.x = radius * std::cos(angle), .y = radius * std::sin(angle), .z = height};
    const Vec3 velocity = {
        .x = 0.2 * generator.NextSigned(),
        .y = 0.2 * generator.NextSigned(),
        .z = -1 - generator.Next(),
    };
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    SolidContactResult result;
    const bool is_success = MoveParticle(inputs, &result, nullptr) == ContactError::kNone;
    hit_paths += is_success && result.hit_count > 0 ? 1 : 0;
    repeat_hits += is_success && result.hit_count > 1 ? 1 : 0;
  }
  std::cout << "rim paths 5000, hit " << hit_paths << ", repeat hits " << repeat_hits << "\n";
  EXPECT_EQ(repeat_hits, 0);
  EXPECT_GT(hit_paths, 2000);
}

// One path of the chasing set: a fixed flat solid A and a flat solid B chasing the particle along a normal at right
// angles to A's.
struct ChasingPath {
  std::array<Solid, 2> solids{};
  SolidContactInputs inputs{};
};

// Builds one path of the chasing set. A goes through the origin with a random normal nA; B's normal is a random
// direction made square to nA, its surface is 5 to 15 m away (B at 100 m/s) or 50 to 150 m (at 1000 m/s) and it moves
// along -nB, so it reaches the particle after A has. The particle starts at 0.05 nA moving at -nA plus up to 0.5 m/s
// along each axis, both velocities equal, T 0.2 s, friction 0. Draw order: three NextSigned for nA (drawn again while
// shorter than 0.1), three for B's direction (drawn again while its part square to nA is shorter than 0.1), one Next
// for the speed, one for the distance, three NextSigned for the velocity's offset. inputs.solids is left for the
// caller.
ChasingPath BuildChasingPath(CaseGenerator& generator) {
  Vec3 first_direction{};
  while (math::Length(first_direction) < 0.1) {
    first_direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  const Vec3 first_normal = math::Normalized(first_direction);
  Vec3 chasing_direction{};
  while (math::Length(chasing_direction) < 0.1) {
    const Vec3 second = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    chasing_direction = second - math::Dot(first_normal, second) * first_normal;
  }
  const Vec3 chasing_normal = math::Normalized(chasing_direction);
  const bool is_fast = generator.Next() < 0.5;
  const double chasing_speed = is_fast ? 1000.0 : 100.0;  // m/s
  const double distance = (is_fast ? 50.0 : 5.0) + (is_fast ? 100.0 : 10.0) * generator.Next();  // m
  const Vec3 offset = 0.5 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  ChasingPath path;
  const SolidMotion no_motion;
  path.solids.at(0) = move_particle_cases::BuildFlatSolid(Vec3{}, first_normal, no_motion, 1);
  path.solids.at(1) = move_particle_cases::BuildFlatSolid(
      distance * chasing_normal,
      -chasing_normal,
      {.linear_velocity = -chasing_speed * chasing_normal, .angular_velocity = {}},
      2
  );
  path.inputs.solid_count = 2;
  path.inputs.contact_tolerance = kTolerance;
  path.inputs.step_duration = 0.2;
  path.inputs.start_position = 0.05 * first_normal;
  const Vec3 velocity = offset - first_normal;
  path.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  return path;
}

TEST(MoveParticle, GeneratedChasingSolid_FirstSolidRarelyHitAgain) {
  // The reviewer's set, 20,000 paths (BuildChasingPath): A is hit, then B, which carries the particle away along a
  // direction square to A's normal. Neither is approached again, so a third hit is a re-hit of A by rounding.
  CaseGenerator generator;
  int two_hit_paths = 0;
  int re_hits = 0;
  int failures = 0;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    ChasingPath path = BuildChasingPath(generator);
    path.inputs.solids = path.solids.data();
    SolidContactResult result;
    const bool is_success = MoveParticle(path.inputs, &result, nullptr) == ContactError::kNone;
    failures += is_success ? 0 : 1;
    two_hit_paths += is_success && result.hit_count == 2 ? 1 : 0;
    re_hits += is_success && result.hit_count > 2 ? 1 : 0;
  }
  std::cout << "chasing paths 20000, two hits " << two_hit_paths << ", re-hits " << re_hits << ", failures " << failures
            << "\n";
  EXPECT_EQ(re_hits, 0);
  EXPECT_EQ(failures, 0);
  EXPECT_GT(two_hit_paths, 15000);
}

TEST(MoveParticle, GeneratedSteepHits_NeverHitAgain) {
  // 20,000 near head-on hits on one fixed flat solid through the origin: a random normal n (three NextSigned, drawn
  // again while shorter than 0.1), a start at 0.5 n, both velocities -(1 + 4 * Next) n plus up to 1e-4 m/s per axis
  // (three NextSigned), T 1 s, friction 0. The response leaves a sideways speed of about 1e-4 m/s, far smaller than the
  // incoming speed, and the particle slides on from the rest gap: never a second hit. With friction 0 the response's
  // second removal leaves rounding of that small velocity, so this set cannot tell the incoming bound from the
  // outgoing one; GeneratedNearStickSlips_NeverHitAgain can.
  CaseGenerator generator;
  int hit_paths = 0;
  int repeat_hits = 0;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    Vec3 direction{};
    while (math::Length(direction) < 0.1) {
      direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    }
    const Vec3 normal = math::Normalized(direction);
    const double speed = 1 + 4 * generator.Next();  // m/s
    const Vec3 offset =
        1e-4 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const SolidMotion no_motion;
    const std::array solids = {move_particle_cases::BuildFlatSolid(Vec3{}, normal, no_motion, 1)};
    SolidContactInputs inputs;
    inputs.solids = solids.data();
    inputs.solid_count = 1;
    inputs.contact_tolerance = kTolerance;
    inputs.step_duration = 1;
    inputs.start_position = 0.5 * normal;
    const Vec3 velocity = offset - speed * normal;
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    SolidContactResult result;
    const bool is_success = MoveParticle(inputs, &result, nullptr) == ContactError::kNone;
    hit_paths += is_success && result.hit_count > 0 ? 1 : 0;
    repeat_hits += is_success && result.hit_count > 1 ? 1 : 0;
  }
  std::cout << "steep paths 20000, hit " << hit_paths << ", repeat hits " << repeat_hits << "\n";
  EXPECT_EQ(repeat_hits, 0);
  EXPECT_EQ(hit_paths, 20000);
}

TEST(MoveParticle, GeneratedNearStickSlips_NeverHitAgain) {
  // 20,000 slips just below the stick limit on one fixed flat solid through the origin: a random normal n (three
  // NextSigned, drawn again while shorter than 0.1), a start at 0.05 n, both velocities with three NextSigned times
  // 3 m/s per axis (drawn again while approaching slower than 0.2 m/s), T 1 s, and the friction (1 - delta) times the
  // stick limit, sideways speed over speed into the solid, with delta = 10^(-8 + 7 * Next). The friction blend rounds
  // by a share of the incoming speed while the returned velocity is tiny: an allowance from the returned velocity
  // would call that a new approach and hit the solid again.
  CaseGenerator generator;
  int hit_paths = 0;
  int repeat_hits = 0;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    Vec3 direction{};
    while (math::Length(direction) < 0.1) {
      direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    }
    const Vec3 normal = math::Normalized(direction);
    Vec3 velocity{};
    while (!(math::Dot(normal, velocity) < -0.2)) {
      velocity = 3 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    }
    const double approach_speed = math::Dot(normal, velocity);  // m/s, negative.
    const double sideways_speed = math::Length(velocity - approach_speed * normal);  // m/s
    const double stick_limit = sideways_speed / -approach_speed;  // No unit.
    const double below_limit_share = std::pow(10.0, -8 + 7 * generator.Next());  // No unit.
    const SolidMotion no_motion;
    const std::array solids = {move_particle_cases::BuildFlatSolid(Vec3{}, normal, no_motion, 1)};
    SolidContactInputs inputs;
    inputs.solids = solids.data();
    inputs.solid_count = 1;
    inputs.contact_tolerance = kTolerance;
    inputs.step_duration = 1;
    inputs.start_position = 0.05 * normal;
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    inputs.friction = (1 - below_limit_share) * stick_limit;
    SolidContactResult result;
    const bool is_success = MoveParticle(inputs, &result, nullptr) == ContactError::kNone;
    hit_paths += is_success && result.hit_count > 0 ? 1 : 0;
    repeat_hits += is_success && result.hit_count > 1 ? 1 : 0;
  }
  std::cout << "near-stick paths 20000, hit " << hit_paths << ", repeat hits " << repeat_hits << "\n";
  EXPECT_EQ(repeat_hits, 0);
  EXPECT_EQ(hit_paths, 20000);
}

// --- The settle ---

// A move that ended at the given position after one hit on solid 0, with the scene's inputs.
ParticleMove BuildMoveTouchingSolidZero(Vec3 end_position) {
  ParticleMove move;
  move.result.position = end_position;
  move.result.hit_count = 1;
  move.touched_surfaces[0] = {.solid_index = 0, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}};
  move.touched_surface_count = 1;
  return move;
}

TEST(ApplySettleAtEnd, TouchedFloorCloserThanTolerance_MovesToRestGap) {
  const std::array solids = {move_particle_cases::BuildFloor(1)};
  SolidContactInputs inputs;
  inputs.solids = solids.data();
  inputs.solid_count = 1;
  inputs.step_duration = 1;
  inputs.contact_tolerance = kTolerance;
  ParticleMove move = BuildMoveTouchingSolidZero({.x = 0.5, .y = 0, .z = 0.25 * kTolerance});
  ApplySettleAtEnd(inputs, &move);
  ExpectVec3Equal(move.result.position, {.x = 0.5, .y = 0, .z = kTolerance});
}

TEST(ApplySettleAtEnd, TouchedFloorFartherThanTolerance_StaysPut) {
  const std::array solids = {move_particle_cases::BuildFloor(1)};
  SolidContactInputs inputs;
  inputs.solids = solids.data();
  inputs.solid_count = 1;
  inputs.step_duration = 1;
  inputs.contact_tolerance = kTolerance;
  ParticleMove move = BuildMoveTouchingSolidZero({.x = 0.5, .y = 0, .z = 2 * kTolerance});
  ApplySettleAtEnd(inputs, &move);
  ExpectVec3Equal(move.result.position, {.x = 0.5, .y = 0, .z = 2 * kTolerance});
}

TEST(ApplySettleAtEnd, RisingFloor_SettlesAtEndPose) {
  // The floor rises at 0.25 m/s for 1 s: at the end it is at z = 0.25, and a particle 0.25 tol above it moves to
  // 0.25 + tol. At the start pose the particle would be far above the floor and stay put.
  const std::array solids = {
      move_particle_cases::BuildFlatSolid(
          {},
          {.x = 0, .y = 0, .z = 1},
          {.linear_velocity = {.x = 0, .y = 0, .z = 0.25}},
          1
      ),
  };
  SolidContactInputs inputs;
  inputs.solids = solids.data();
  inputs.solid_count = 1;
  inputs.step_duration = 1;
  inputs.contact_tolerance = kTolerance;
  ParticleMove move = BuildMoveTouchingSolidZero({.x = 0, .y = 0, .z = 0.25 + 0.25 * kTolerance});
  ApplySettleAtEnd(inputs, &move);
  EXPECT_NEAR(move.result.position.z, 0.25 + kTolerance, 0x1p-52);
}

TEST(ApplySettleAtEnd, SixtyDegreeWedgeBothNearAtEnd_EndsAtRestGapOfBoth) {
  // tol / 2 from the floor and from the 60 degree face, both touched: one move to {Sqrt(3) tol, 0, tol}. One move per
  // surface in turn left the floor at 0.625 tol.
  const MoveParticleCase scene = FindMoveParticleCase("SixtyDegreeWedgeRestingOnBoth");
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  ParticleMove move =
      BuildMoveTouchingSolidZero({.x = std::numbers::sqrt3 / 2 * kTolerance, .y = 0, .z = kTolerance / 2});
  move.touched_surfaces[1] = {
      .solid_index = 1,
      .normal_out_of_solid = {.x = std::numbers::sqrt3 / 2, .y = 0, .z = -0.5},
  };
  move.touched_surface_count = 2;
  ApplySettleAtEnd(inputs, &move);
  ExpectVec3Near(
      move.result.position,
      {.x = std::numbers::sqrt3 * kTolerance, .y = 0, .z = kTolerance},
      kAtRestGapTolerance
  );
  ExpectAtRestGapOfEverySolid(inputs, move.result.position);
}

TEST(ApplySettleAtEnd, UntouchedNearSolid_IsMovedOffToo) {
  // The same end with only the floor touched: the settle reads every solid, so the 60 degree face counts too. Settling
  // the touched floor alone, to z = tol, left the face at 0.25 tol.
  const MoveParticleCase scene = FindMoveParticleCase("SixtyDegreeWedgeRestingOnBoth");
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  ParticleMove move =
      BuildMoveTouchingSolidZero({.x = std::numbers::sqrt3 / 2 * kTolerance, .y = 0, .z = kTolerance / 2});
  ApplySettleAtEnd(inputs, &move);
  ExpectVec3Near(
      move.result.position,
      {.x = std::numbers::sqrt3 * kTolerance, .y = 0, .z = kTolerance},
      kAtRestGapTolerance
  );
  ExpectAtRestGapOfEverySolid(inputs, move.result.position);
}

}  // namespace
}  // namespace fluidity::contact
