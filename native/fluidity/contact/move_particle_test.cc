#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

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

TEST(MoveParticle, NullFailure_MovesAndFailsAsWithFailure) {
  const MoveParticleCase moving = FindMoveParticleCase("DocstringExample");
  SolidContactResult result;
  EXPECT_EQ(MoveParticle(ToSolidContactInputs(moving), &result, nullptr), ContactError::kNone);
  ExpectVec3Near(result.position, {.x = 1 - kTolerance, .y = 0, .z = 0}, kPositionTolerance);
  const MoveParticleCase failing = FindMoveParticleCase("TwoReactingSolids");
  result = BuildUntouchedResult();
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

TEST(MoveParticle, SettleOnRisingFloorAfterTiltedUpright_EndsAtRestGap) {
  // The tilted upright flat solid leaves the particle approaching the rising floor by 0.75 tol over the rest of the
  // step, too little for a hit. The floor is a touched surface, so the settle, at the floor's end pose, lifts the
  // particle from 0.25 + 0.25 tol back to 0.25 + tol; without it the end check would fail.
  const MoveParticleCall call = RunMove("SettleOnRisingFloorAfterTiltedUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  EXPECT_NEAR(call.result.position.z, 0.25 + kTolerance, kTolerance / 16);
  EXPECT_NEAR(call.result.position.x, 1 - kTolerance, kPositionTolerance);
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

TEST(MoveParticle, TurningBoxSlowApproach_PassesOnCloseCall) {
  const MoveParticleCall call = RunMove("TurningBoxSlowApproach");
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  ExpectFailure(
      call.failure,
      {.error = ContactError::kTooCloseToCall, .close_call = ContactCloseCall::kSlowApproachToTurningSolid}
  );
  ExpectUntouched(call.result);
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
};

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
  EXPECT_GT(counts.paths, 40000);
  EXPECT_GT(counts.hit_paths, counts.paths / 4);
  EXPECT_GT(counts.several_hit_paths, counts.paths / 40);
}

}  // namespace
}  // namespace fluidity::contact
