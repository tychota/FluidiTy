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
#include <utility>
#include <vector>

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
constexpr SolidMotion kNoMotion = move_particle_cases::kNoMotion;  // A solid that neither moves nor turns.
constexpr Vec3 kUnitX = move_particle_cases::kUnitX;
constexpr Vec3 kUnitY = move_particle_cases::kUnitY;
constexpr Vec3 kUnitZ = move_particle_cases::kUnitZ;

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

// Fails the calling test unless the call failed with kTooCloseToCall and the close call, naming solid_id, magnitude 0,
// and left the result unchanged.
void ExpectCloseCall(const MoveParticleCall& call, ContactCloseCall close_call, std::uint32_t solid_id) {
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  ExpectFailure(call.failure, {.error = ContactError::kTooCloseToCall, .close_call = close_call, .solid_id = solid_id});
  ExpectUntouched(call.result);
}

// Fails the calling test unless both velocities are within tolerance of expected on each axis.
void ExpectBothVelocitiesNear(const SolidContactResult& result, Vec3 expected, double tolerance) {
  ExpectVec3Near(result.velocity.for_next_step, expected, tolerance);
  ExpectVec3Near(result.velocity.for_this_step, expected, tolerance);
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

TEST(MoveParticle, TiltedUprightPushesBackToFloor_SlideRemovesFloorApproach) {
  // The floor is hit at 0.1 - tol s, the tilted upright at about 0.5 s; its response leaves the particle approaching
  // the floor at 1.5 tol m/s, a real approach far above the floor's allowance of about 2^-46 m/s. The floor is active
  // there, so the joint response over both removes that approach at once, and the particle rests at z = tol with two
  // hits. Before the change over several surfaces (plan Task 12) the touched floor was hit again: three hits.
  const MoveParticleCall call = RunMove("TiltedUprightPushesBackToFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  EXPECT_NEAR(call.result.position.z, kTolerance, kTolerance / 16);
  EXPECT_NEAR(call.result.position.x, 1 - kTolerance, kPositionTolerance);
}

TEST(MoveParticle, TiltedUprightOverRisingFloor_SlideCarriesParticleWithFloor) {
  // As TiltedUprightPushesBackToFloor with the floor rising at 0.25 m/s: hit at (0.1 - tol) / 1.25 s, the upright at
  // about 0.5 s, and the joint response over both gives the velocity that approaches neither, {-3.5e-10, 0, 0.25}: the
  // particle rides up with the floor to z = 0.25 + tol. Before it the moving touched floor was hit again: three hits.
  const MoveParticleCall call = RunMove("TiltedUprightOverRisingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
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

TEST(MoveParticle, CeilingClosingFasterThanParticleLeaves_IsSqueezedBetweenSolids) {
  // The ceiling hits at 4/7 s and carries the particle down at 1.75 tol m/s toward the floor, at its rest gap: the
  // floor is an active surface of that hit, and the two close on the particle. No change leaves both; the particle
  // sticks to the ceiling, the surface it approaches fastest, which still approaches the floor: a squeeze naming the
  // floor, the solid the stuck velocity approaches fastest (the record is the ceiling's). Before the floor was read at
  // the hit, its touching rule let the particle sink to z = 0.25 tol, and the end check failed (kEndsTooCloseToSolid).
  ExpectCloseCall(RunMove("CeilingClosingFasterThanParticleLeaves"), ContactCloseCall::kSqueezedBetweenSolids, 1);
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

TEST(MoveParticle, FiveFloorsWithinStartOverlap_FastestFloorCarries) {
  // Five parallel floors rising at 1, 1.1, 1.2, 1.3 and 1.4 m/s, the start inside all of them: one joint response over
  // the five, one hit, which no start can exceed any more (plan decision P20). The fastest floor, id 5, pushes: the
  // particle takes its 1.4 m/s, and the record is its own.
  const MoveParticleCall call = RunMove("FiveFloorsWithinStartOverlap");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 4U);
  ExpectBothVelocitiesNear(call.result, {.x = 0, .y = 0, .z = 1.4}, 1e-15);
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

TEST(MoveParticle, CornerWithinTolerance_OneHitOverBothSolids) {
  // The floor moves the start to z = tol and stops the fall; the upright, posed at time 0 at x = 1, moves it to
  // x = 1 - tol, and its approach of -1.5 m/s gives its velocity {-0.5, 0, 0}, which carries the particle. One joint
  // response over both: one hit, the record the floor's, the first in array order.
  const MoveParticleCall call = RunMove("CornerWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
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
  // The record is the floor's, the first in array order; the start inside the upright, a later solid, sets the kind.
  // One joint response over both: one hit.
  const MoveParticleCall call = RunMove("InsideUprightAfterApproachingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.solid_index, 0U);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  EXPECT_EQ(result.approach_speed.for_this_step, -2);
  EXPECT_EQ(result.hit_count, 1);
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
  // The start is inside past side_left, the nearest, and side_bottom: both move it out, and one joint response over
  // both sides stops it: one hit, the record on side_left, the first in face order.
  const MoveParticleCall call = RunMove("DomainCornerSlightlyInside");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.face, SolidFace::kLowX);
  EXPECT_EQ(result.hit_count, 1);
  ExpectVec3Equal(result.velocity.for_this_step, {});
  ExpectVec3Near(result.position, {.x = kTolerance, .y = 0.5, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, WedgeStartPushedTowardFloor_MovesOffBothAtOnce) {
  // Shortfalls 0.5 tol (floor) and 0.1 tol (wedge), normals -0.8 apart: one move of {-5/6 tol, 0, 0.5 tol}. The fall
  // approaches the floor, which responds; the rising wedge is left. The particle slides along -x at z = tol.
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
  // one joint response, one hit, and the kind is kStartedInsideSolid.
  const MoveParticleCall call = RunMove("WedgeStartPushedIntoFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.hit_count, 1);
  ExpectVec3Equal(result.velocity.for_this_step, {.x = -1, .y = 0, .z = 0});
  ExpectVec3Equal(result.velocity.for_next_step, {.x = -1, .y = 0, .z = 0});
  const double end_x = -0.1 - 0.0005 * 5.0 / 3 - 5.0 / 6 * kTolerance;  // m
  ExpectVec3Near(result.position, {.x = end_x, .y = 0, .z = kTolerance}, kAtRestGapTolerance);
}

TEST(MoveParticle, ParallelSolidAfterStartResponse_RoundingIsNotAnApproach) {
  // Two parallel solids tol / 4 apart: the move takes the start to the rest gap of the upper, reacting one, and the
  // lower one, 1.25 tol behind it, is not an active surface. So one hit, and the reaction is the real one, the upper
  // solid's push, not rounding: the response leaves an approach of -2^-55 m/s, within its allowance, which the sweep
  // reads as rest on every solid; GeneratedParallelSolids_RoundingNeverHitsTheSecond draws both signs.
  const MoveParticleCall call = RunMove("ParallelSolidAfterStartResponse");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.reaction.solid_index, 1U);
  ExpectVec3Near(call.result.reaction.velocity_for_next_step_change, {.x = 1.68, .y = 0, .z = 2.24}, 1e-15);
}

TEST(MoveParticle, FourCoincidentFloorsBesideUpright_OneHit) {
  const MoveParticleCall call = RunMove("FourSilentFloorsAndUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 4U);
}

TEST(MoveParticle, FourCoincidentFloorsAndUpright_SlideReadsThemAsOneDirection) {
  // The upright's hit reads the four floors and the upright: the floors are exact copies, so two active surfaces facing
  // two ways. Read one direction per surface, it would be kTooManyHits.
  const MoveParticleCall call = RunMove("FourCoincidentFloorsAndUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 4U);
  ExpectVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
}

TEST(MoveParticle, FiveCoincidentFloorsBesideUpright_OneHit) {
  // Nothing is listed per solid any more, so a fifth floor beside the upright's start response costs nothing: one hit,
  // the record the upright's, the velocity {0, 0, 1}.
  const MoveParticleCall call = RunMove("FiveSilentFloorsAndUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 5U);
  ExpectVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
}

TEST(MoveParticle, FiveStackedFloorsLeft_OneSilentMoveNoHit) {
  // Five floors moved off at once are one silent move: no response, no hit, nothing to fail.
  const MoveParticleCall call = RunMove("FiveSilentFloors");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
}

TEST(MoveParticle, SlidingFloorThenSlowUpright_ThresholdIsZeroBeforeAnyResponse) {
  // Before any response the approach test compares with 0, so the upright's real approach of 1e-13 m/s beside a floor
  // sliding at 100 m/s is a hit at the start, and the joint response removes it: the solver's no change reads the
  // slack passed, 0, not its own allowance of the 100 m/s (plan decision P20, item 3).
  const MoveParticleCall call = RunMove("SlidingFloorThenSlowUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.velocity.for_next_step.x, 0);
}

TEST(MoveParticle, SilentFloorThenLeaningUpright_FloorRespondsToThePush) {
  // Plan decision P12: with no hit at the start the move is silent. The upright's hit then pushes the particle into
  // the floor, at its rest gap: the floor is an active surface of that hit, so one joint response over both, one hit,
  // the record the upright's. Before the floor was read at the hit, the push was a drift of 0.05 tol on the floor's
  // plain touching rule, which the settle lifted back.
  const MoveParticleCall call = RunMove("SilentFloorThenLeaningUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_NEAR(call.result.position.z, kTolerance, kAtRestGapTolerance);
}

TEST(MoveParticle, AtRestGapSlowApproach_FaceAtRestGapApproachedResponds) {
  // d = tol exactly is outside the start rules' move, d < tol, but a face at the rest gap that a velocity approaches is
  // an active surface: the approach of tol m/s is a hit at 0 and the particle
  // stops at z = tol. Before, the touching rule let it sink by 0.1 tol in the step, which the settle lifted back; in a
  // wedge sharper than 2^-21 rad that sinking left a gap no move could leave.
  const MoveParticleCall call = RunMove("AtRestGapSlowApproach");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.velocity.for_this_step.z, 0);
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

TEST(MoveParticle, DomainThreeSideCornerThisStepApproaches_OneHitOverAllThree) {
  // for_this_step {-1, -1, -1} approaches the three sides: one joint response over them, one hit, the record on kLowX,
  // the first in face order.
  const MoveParticleCall call = RunMove("DomainThreeSideCornerThisStepApproaches");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowX);
  ExpectVec3Equal(call.result.velocity.for_this_step, {});
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainThreeSideCornerNextStepApproaches_OneHitOverAllThree) {
  const MoveParticleCall call = RunMove("DomainThreeSideCornerNextStepApproaches");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3Equal(call.result.velocity.for_next_step, {});
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainThreeSideCornerSlightlyInside_MovedOutOfAllThree) {
  const MoveParticleCall call = RunMove("DomainThreeSideCornerSlightlyInside");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.hit_count, 1);
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
  // The response to for_this_step at rest has an allowance of 0, but it leaves -2^-55 m/s in the moving
  // for_next_step. Compared with for_next_step's own allowance that is rounding: one hit, and the reaction is the upper
  // solid's real push, not rounding. One allowance for both velocities once made the rounding a second hit.
  const MoveParticleCall call = RunMove("ParallelSolidAfterStartResponseNextStepMoving");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.reaction.solid_index, 1U);
  ExpectVec3Near(call.result.reaction.velocity_for_next_step_change, {.x = 1.68, .y = 0, .z = 2.24}, 1e-15);
}

TEST(MoveParticle, SlidingFloorThenSlowUprightNextStepOnly_NextStepThresholdIsZeroBeforeAnyResponse) {
  // Before any response for_next_step's threshold is 0 too: its approach of 1e-13 m/s to the upright is a hit at the
  // start.
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

TEST(MoveParticle, CuspStartIntoBallOnly_ResponseUsesNormalAtMovedStart) {
  // In the cusp, halfway to the rest point, moving at 1 m/s straight into the ball and away from the floor. The
  // response reads the ball's normal at the moved start (plan decision P20, item 1), the normal the sweep that follows
  // reads, turned from the one at the start as given by about 2 Sqrt(tol / R) (1 - f): it leaves 4.3e-5 m/s along the
  // ball, and the particle slides on to 46804 tol from the contact.
  const MoveParticleCall call = RunMove("CuspStartIntoBallOnly");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {.x = 4.31583759e-05, .y = 0, .z = 3.72529e-09}, 1e-9);
  ExpectVec3Near(call.result.position, {.x = 46804.36 * kTolerance, .y = 0, .z = 1.04 * kTolerance}, 0.01 * kTolerance);
}

// --- The slide along several surfaces (plan decision P19, items 3 to 6) ---

// Runs MoveParticle on the scene of that name, and fails the calling test unless it succeeded with that many hits, at
// rest, at the expected end within position_tolerance, with every solid at the rest gap.
SolidContactResult
ExpectStopAtRestGap(std::string_view case_name, int hit_count, Vec3 expected_end, double position_tolerance) {
  const SolidContactResult result = ExpectEndAtRestGap(case_name, expected_end, position_tolerance);
  EXPECT_EQ(result.hit_count, hit_count) << case_name;
  ExpectBothVelocitiesNear(result, {}, 1e-15);
  return result;
}

TEST(MoveParticle, AcuteWedge_StopsWithoutError) {
  // The 45 degree wedge's face is hit at x = 1 - (1 + Sqrt(2)) tol, where the particle also rests on the floor: one
  // joint response over both stops it at the apex, one hit.
  const MoveParticleCase scene = FindMoveParticleCase("AcuteWedge");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {}, 1e-15);
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    EXPECT_GE(
        ComputeDistanceToSolid(solid, ComputeSolidPose(solid, inputs.step_duration), call.result.position).distance,
        kTolerance / 2
    );
  }
}

TEST(MoveParticle, RightAngleCorner_SlidesAlongSecondSurface) {
  // The floor at 0.05 - tol / 4 s leaves {2, 1, 0}; the upright at x = 1 - tol leaves {0, 1, 0}, along the corner.
  const MoveParticleCall call = RunMove("RightAngleCorner");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  ExpectBothVelocitiesNear(call.result, {.x = 0, .y = 1, .z = 0}, 1e-15);
  EXPECT_NEAR(call.result.position.x, 1 - kTolerance, kPositionTolerance);
  EXPECT_NEAR(call.result.position.z, kTolerance, kPositionTolerance);
}

TEST(MoveParticle, SlideRemovalOnReactingFloor_CountsInReaction) {
  // The floor at 0.1 - tol / 2 s, at {0.6 - tol / 2, 0, tol}: {1, 0, 0}, a change of {0, 0, 2}. The upright sliding
  // down along itself at 1 m/s is hit at 0.5 - tol s: approach -1, sideways speed 1 against it, and friction 0.5 leaves
  // {0, 0, -0.5}, into the floor. The change removes that approach, {0, 0, 0.5}, charged to the floor; the upright is
  // not approached again. Every value is a binary fraction: without the change the reaction is {0, 0, 2}.
  const MoveParticleCall call = RunMove("SlideRemovalOnReactingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  ExpectBothVelocitiesNear(call.result, {}, 0);
  ExpectVec3Near(call.result.position, {.x = 1 - kTolerance, .y = 0, .z = kTolerance}, kPositionTolerance);
  EXPECT_EQ(call.result.reaction.solid_index, 0U);
  ExpectVec3Equal(call.result.reaction.velocity_for_next_step_change, {.x = 0, .y = 0, .z = 2.5});
  ExpectVec3Near(
      call.result.reaction.hit_position,
      {.x = 0.6 - kTolerance / 2, .y = 0, .z = kTolerance},
      kPositionTolerance
  );
}

TEST(MoveParticle, AcuteWedgeOnReactingFloor_EachSurfaceChargedItsOwnPush) {
  // The wedge face is hit at x = 1 - (1 + Sqrt(2)) tol, where the particle rests on the reacting floor too: one joint
  // response over both, one hit. The frictionless change from {1, 0, -1} to {0, 0, 0} is the pair's,
  // {-1, 0, 1} = 2 * {0, 0, 1} + Sqrt(2) * Normalized({-1, 0, -1}): the floor is charged its own push {0, 0, 2}, the
  // face's {-1, 0, -1} is not counted. The whole change would give the floor {-1, 0, 1}. The solver's rounding leaves
  // the floor's push within 4e-15 of 2.
  const MoveParticleCall call = RunMove("AcuteWedgeOnReactingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {}, 1e-15);
  EXPECT_EQ(call.result.reaction.solid_index, 0U);
  ExpectVec3Near(call.result.reaction.velocity_for_next_step_change, {.x = 0, .y = 0, .z = 2}, 4e-15);
  const double face_hit_x = 1 - (1 + std::numbers::sqrt2) * kTolerance;  // m
  ExpectVec3Near(call.result.reaction.hit_position, {.x = face_hit_x, .y = 0, .z = kTolerance}, kPositionTolerance);
  // The first hit fills the record: the face, at (face_hit_x - 0.5) / 2 s of the 0.5 s step.
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_NEAR(call.result.hit_fraction, (face_hit_x - 0.5) / 2 / 0.5, 1e-12);
}

TEST(MoveParticle, ObliqueCornerResidue_SlidesOnWithoutSqueezeOrStop) {
  // The first solid's response to {-2, 0, -2} leaves {-0.32, 0, 0.24} with an approach of -2^-55 m/s
  // (contact_response_test.cc), rounding within its allowance of 5.7e-14 m/s; the particle slides along it to the
  // upright, reached at 0.625 s at z = 0.15, and the upright's response leaves {0, 0, 0.24}, which leaves the first
  // solid. It rises along the upright to z = 0.24. A stop at the first hit would end near z = 0.043.
  const MoveParticleCall call = RunMove("ObliqueCornerResidue");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  ExpectBothVelocitiesNear(call.result, {.x = 0, .y = 0, .z = 0.24}, 1e-15);
  EXPECT_NEAR(call.result.position.x, -0.2 + kTolerance, kPositionTolerance);
  EXPECT_NEAR(call.result.position.z, 0.24, 1e-3);
}

TEST(MoveParticle, GrooveSlideAlongEdge_KeepsSpeedAlongEdge) {
  // Both faces of the 60 degree wedge are active at the start: one joint response keeps the part along the common line.
  // The velocity of the solid hit last (design 5.5 as it was) gives {0, 0, 0}.
  const SolidContactResult result = ExpectEndAtRestGap(
      "GrooveSlideAlongEdge",
      {.x = std::numbers::sqrt3 * kTolerance, .y = 0.01, .z = kTolerance},
      kTolerance / 1024
  );
  EXPECT_EQ(result.hit_count, 1);
  ExpectBothVelocitiesNear(result, {.x = 0, .y = 1, .z = 0}, 1e-15);
}

// Where and how a particle should rest, step after step: from step rest_step on (counting from 1) each step has
// hit_count hits and ends at rest_point within tolerance in x and z.
struct RestExpectation {
  int rest_step = 0;
  std::uint8_t hit_count = 0;
  Vec3 rest_point{};  // m, world.
  double tolerance = 0;  // m
};

// Runs one step of a particle under gravity: adds {0, 0, -9.81 T} to for_this_step, makes both velocities that, moves
// the particle, and makes the end and for_this_step the next step's start and velocity. Returns false on failure.
bool RunGravityStep(SolidContactInputs* inputs, SolidContactResult* result) {
  const Vec3 gravity_change = {.x = 0, .y = 0, .z = -9.81 * inputs->step_duration};  // m/s
  inputs->velocity.for_this_step = inputs->velocity.for_this_step + gravity_change;
  inputs->velocity.for_next_step = inputs->velocity.for_this_step;
  if (MoveParticle(*inputs, result, nullptr) != ContactError::kNone) {
    return false;
  }
  inputs->start_position = result->position;
  inputs->velocity.for_this_step = result->velocity.for_this_step;
  return true;
}

// Computes whether one step's result is at rest as expected: the hit count, and the end within tolerance in x and z.
bool ComputeIsRestingAsExpected(const SolidContactResult& result, const RestExpectation& expected) {
  const bool is_near_x = std::abs(result.position.x - expected.rest_point.x) <= expected.tolerance;
  const bool is_near_z = std::abs(result.position.z - expected.rest_point.z) <= expected.tolerance;
  return result.hit_count == expected.hit_count && is_near_x && is_near_z;
}

// Runs 1000 gravity steps (RunGravityStep) from the inputs and fails the calling test unless every step succeeds and
// every step from expected.rest_step on rests as expected (ComputeIsRestingAsExpected). Returns the last result.
SolidContactResult ExpectRestsForThousandSteps(SolidContactInputs inputs, const RestExpectation& expected) {
  SolidContactResult result;
  int failed_steps = 0;
  int steps_not_at_rest = 0;
  for (int step = 1; step <= 1000; ++step) {
    failed_steps += RunGravityStep(&inputs, &result) ? 0 : 1;
    const bool is_resting = ComputeIsRestingAsExpected(result, expected);
    steps_not_at_rest += step >= expected.rest_step && !is_resting ? 1 : 0;
  }
  EXPECT_EQ(failed_steps, 0);
  EXPECT_EQ(steps_not_at_rest, 0);
  return result;
}

// Runs the V-groove scene of that name for 1000 steps of 0.01 s with gravity and the given friction, and fails the
// calling test unless, from step 12 on, each step has one hit, the joint response over both faces, and ends at the
// apex point within tol / 1024 in x and z (the model reaches it in step 10 or 11), at the rest gap of both faces; the
// velocities after the last step are 0 within 1e-15.
void ExpectGrooveRestsAtApex(std::string_view case_name, double friction, Vec3 apex_point) {
  const MoveParticleCase scene = FindMoveParticleCase(case_name);
  SolidContactInputs inputs = ToSolidContactInputs(scene);
  inputs.friction = friction;
  const RestExpectation at_apex = {
      .rest_step = 12,
      .hit_count = 1,
      .rest_point = apex_point,
      .tolerance = kTolerance / 1024,
  };
  const SolidContactResult result = ExpectRestsForThousandSteps(inputs, at_apex);
  ExpectAtRestGapOfEverySolid(inputs, result.position);
  ExpectBothVelocitiesNear(result, {}, 1e-15);
}

TEST(MoveParticle, VGrooveSlideIntoApex_ComesToRestThere) {
  // The 30 degree groove: the apex point is tol / sin(15 degrees) = 3.8637033 tol above the line where the faces meet.
  // The model reaches it in step 10 with friction 0 and in step 11 with 0.5.
  const Vec3 apex_point = {.x = 0, .y = 0, .z = kTolerance / std::sin(15 * std::numbers::pi / 180)};  // m
  ExpectGrooveRestsAtApex("VGroove", 0, apex_point);
  ExpectGrooveRestsAtApex("VGroove", 0.5, apex_point);
}

TEST(MoveParticle, TenDegreeVGrooveSlideIntoApex_ComesToRestThere) {
  // The 10 degree groove: tol / sin(5 degrees) = 11.473713 tol.
  const Vec3 apex_point = {.x = 0, .y = 0, .z = kTolerance / std::sin(5 * std::numbers::pi / 180)};  // m
  ExpectGrooveRestsAtApex("TenDegreeVGroove", 0, apex_point);
  ExpectGrooveRestsAtApex("TenDegreeVGroove", 0.5, apex_point);
}

TEST(MoveParticle, NarrowWedgeAtRestPoint_KeepsSpeedAlongEdge) {
  // A wedge of 2e-6 rad, 1 + c = 2e-12, above the velocity pair limit of 2^-40: one joint response over the side and
  // the floor keeps the part along the common line.
  const MoveParticleCase scene = FindMoveParticleCase("NarrowWedgeAtRestPoint");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {.x = 0, .y = 1, .z = 0}, 1e-9);
  ExpectAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
}

// Runs the sliver scene with both velocities set to velocity and the reacting solid chosen, and fails the calling test
// unless it succeeds with one hit, both velocities {0, 0, 0} exactly, at the rest point. Returns the result.
SolidContactResult ExpectSliverStop(Vec3 velocity, bool is_side_reacting) {
  const double angle = 1e-7;  // rad
  MoveParticleCase scene = FindMoveParticleCase("SliverStop");
  scene.solids[0].reports_reaction = !is_side_reacting;
  scene.solids[1].reports_reaction = is_side_reacting;
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  const MoveParticleCall call = RunMove(scene);
  EXPECT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {}, 0);
  const double rest_x = kTolerance / std::tan(angle / 2);  // m
  ExpectVec3Near(call.result.position, {.x = rest_x, .y = 0, .z = kTolerance}, kTolerance / 1024);
  return call.result;
}

TEST(MoveParticle, SliverStop_ChargedToSolidApproachedFastest) {
  // A wedge of 1e-7 rad, 1 + c = 5e-15, below the pair limit: a sliver, so no change is found and the particle sticks
  // to the surface it approaches fastest (plan decision P20, item 7). Moving at {-2, 1, 0} that is the side, which
  // does not react: no reaction, the record the side's. Moving at {-2, 1, -0.001} it is the reacting floor, charged
  // the whole stop {2, -1, 0.001}. With the side reacting instead, {-2, 1, 0} charges the side {2, -1, 0}.
  const SolidContactResult into_side = ExpectSliverStop({.x = -2, .y = 1, .z = 0}, false);
  EXPECT_EQ(into_side.solid_index, 1U);
  ExpectNoReaction(into_side.reaction);
  const SolidContactResult into_floor = ExpectSliverStop({.x = -2, .y = 1, .z = -0.001}, false);
  EXPECT_EQ(into_floor.solid_index, 0U);
  EXPECT_EQ(into_floor.reaction.solid_index, 0U);
  ExpectVec3Equal(into_floor.reaction.velocity_for_next_step_change, {.x = 2, .y = -1, .z = 0.001});
  const SolidContactResult reacting_side = ExpectSliverStop({.x = -2, .y = 1, .z = 0}, true);
  EXPECT_EQ(reacting_side.reaction.solid_index, 1U);
  ExpectVec3Equal(reacting_side.reaction.velocity_for_next_step_change, {.x = 2, .y = -1, .z = 0});
}

TEST(MoveParticle, ThreeSolidAcuteCornerInto_StopsInCorner) {
  // The three faces are hit at one moment: one joint response, one hit.
  ExpectStopAtRestGap(
      "ThreeSolidAcuteCornerInto",
      1,
      {.x = std::numbers::sqrt3 * kTolerance, .y = std::numbers::sqrt3 * kTolerance, .z = kTolerance},
      kTolerance / 1024
  );
}

TEST(MoveParticle, CeilingHitFarAboveTouchedFloor_SlidesDownToApex) {
  // The floor responds at the start (inside), the ceiling is hit at x = 0.10526 and leaves {2.72, 0, -2.04}, which
  // approaches the floor 0.021 m below; the floor is not active there, so the particle slides down the ceiling to the
  // apex and stops. With the floor in the change it stops at the ceiling hit, {0.10526, 0, 0.02105}.
  const SolidContactResult result = ExpectStopAtRestGap(
      "CeilingHitFarAboveTouchedFloor",
      3,
      {.x = (0.08 - 1.8 * kTolerance) / 0.6, .y = 0, .z = kTolerance},
      kTolerance / 1024
  );
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
}

TEST(MoveParticle, CornerPushOnSecondReactingSolid_IsSecondReactingSolid) {
  // The floor and the side are both active at the start: one joint response pushes both, two reacting solids, and the
  // charges, in ascending solid index, fail at the side, the higher index, id 2. Without the side reacting: one hit,
  // and the floor is charged its push {0, 0, 2 + 1 / Sqrt(3)}.
  ExpectCloseCall(RunMove("CornerPushOnSecondReactingSolid"), ContactCloseCall::kSecondReactingSolid, 2);
  const MoveParticleCall call = RunMove("CornerPushOnNonReactingSide");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {}, 1e-15);
  EXPECT_EQ(call.result.reaction.solid_index, 1U);
  ExpectVec3Near(
      call.result.reaction.velocity_for_next_step_change,
      {.x = 0, .y = 0, .z = 2 + 1 / std::numbers::sqrt3},
      1e-15
  );
}

// --- Round solids: the slide into a cusp, a neck and a ring ---

TEST(MoveParticle, BallRestingOnFloorSlidingIntoCusp_ComesToRestThere) {
  // The sphere is hit at the rest point itself, 2 Sqrt(0.5 tol) = 46340.95 tol from the contact, where the particle
  // rests on the floor too: one joint response over both stops it, one hit. The first version of P19 failed it.
  const double rest_x = 2 * std::sqrt(0.5 * kTolerance);  // m
  ExpectStopAtRestGap("BallRestingOnFloorSlidingIntoCusp", 1, {.x = rest_x, .y = 0, .z = kTolerance}, kTolerance / 256);
}

TEST(MoveParticle, BallRestingOnFloorRestsInCuspForThousandSteps) {
  // 1000 steps of 0.01 s with gravity, from 0.05 m out at -0.5 m/s: the rest point is reached in step 9 (the model),
  // and from then on each step has one hit, the floor's, and ends at the rest point.
  const MoveParticleCase scene = FindMoveParticleCase("BallRestingOnFloorRollingIntoCusp");
  const RestExpectation at_rest_point = {
      .rest_step = 12,
      .hit_count = 1,
      .rest_point = {.x = 2 * std::sqrt(0.5 * kTolerance), .y = 0, .z = kTolerance},
      .tolerance = kTolerance / 256,
  };
  ExpectRestsForThousandSteps(ToSolidContactInputs(scene), at_rest_point);
}

TEST(MoveParticle, TwoBallsTouchingSlidingIntoNeck_ComesToRest) {
  // The neck ring Sqrt(tol (2 R + tol)) = 32768.0 tol from the contact, where both balls are at tol: one hit there.
  const double ring_radius = std::sqrt(kTolerance * (1 + kTolerance));  // m
  ExpectStopAtRestGap("TwoBallsTouchingSlidingIntoNeck", 1, {.x = 0.5, .y = ring_radius, .z = 0}, kTolerance / 256);
}

TEST(MoveParticle, PlateAndSphereSlidingIn_ComesToRestAtRing) {
  // The sphere is hit at (0.5 - Sqrt(0.1875) - Sqrt(3) tol) / 0.1 of the step, where the particle is tol from it and
  // from the plate: one joint response over both, one hit. Both normals at the ring are 120 degrees apart, so the rest
  // point is 2 tol out along their bisector {Sqrt(3) / 2, 0, 1 / 2}.
  const Vec3 ring_point = {.x = std::sqrt(0.1875), .y = 0, .z = 0.01};
  const Vec3 bisector = {.x = std::numbers::sqrt3 / 2, .y = 0, .z = 0.5};
  const SolidContactResult result =
      ExpectStopAtRestGap("PlateAndSphereSlidingIn", 1, ring_point + 2 * kTolerance * bisector, kTolerance / 1024);
  EXPECT_EQ(result.solid_index, 1U);
  EXPECT_EQ(result.face, SolidFace::kCurved);
  EXPECT_NEAR(result.hit_fraction, (0.5 - std::sqrt(0.1875) - std::numbers::sqrt3 * kTolerance) / 0.1, 1e-9);
}

TEST(MoveParticle, CylinderLyingOnFloorSlidingIntoCusp_ComesToRestThere) {
  // The cross-section of BallRestingOnFloorSlidingIntoCusp: the wall is hit at the rest point 2 Sqrt(0.5 tol) from the
  // contact line, where the particle rests on the floor too: one joint response over both stops it, one hit.
  const double rest_y = 2 * std::sqrt(0.5 * kTolerance);  // m
  ExpectStopAtRestGap(
      "CylinderLyingOnFloorSlidingIntoCusp",
      1,
      {.x = 0, .y = rest_y, .z = kTolerance},
      kTolerance / 256
  );
}

TEST(MoveParticle, CylinderSunkIntoFloorSlidingIn_ComesToRest) {
  // The cross-section of PlateAndSphereSlidingIn: the wall meets the floor at 60 degrees at y = Sqrt(0.1875), and the
  // point at tol from both is Sqrt(3) tol out along the floor.
  const Vec3 rest_point = {.x = 0, .y = std::sqrt(0.1875) + std::numbers::sqrt3 * kTolerance, .z = kTolerance};
  ExpectStopAtRestGap("CylinderSunkIntoFloorSlidingIn", 1, rest_point, kTolerance / 1024);
}

// --- Spheres and cylinders alone ---

// The distance from a point to the first solid of a scene at its pose at the end of the step.
double ComputeEndDistance(const MoveParticleCase& scene, Vec3 point) {
  const Solid& solid = scene.solids.at(0);
  return ComputeDistanceToSolid(solid, ComputeSolidPose(solid, scene.inputs.step_duration), point).distance;
}

TEST(MoveParticle, SphereHeadOn_StopsOneToleranceShort) {
  // The grown sphere is reached at x = -(0.5 + tol), where the normal is {-1, 0, 0} and the whole speed is removed.
  const MoveParticleCall call = RunMove("SphereHeadOn");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.face, SolidFace::kCurved);
  EXPECT_EQ(result.hit_count, 1);
  ExpectVec3Near(result.position, {.x = -(0.5 + kTolerance), .y = 0, .z = 0}, kPositionTolerance);
  ExpectBothVelocitiesNear(result, {}, 1e-15);
  EXPECT_EQ(result.approach_speed.for_next_step, -1);
  EXPECT_EQ(result.approach_speed.for_this_step, -1);
}

TEST(MoveParticle, SphereOblique_SlidesOffAlongTheTangent) {
  // The hit is at {-0.4, 0.3, 0} times 1 + 2 tol, with the normal {-0.8, 0.6, 0}: the response leaves
  // {0.36, 0.48, 0}, which the 0.4 s left carry to {-0.256, 0.492, 0}. The sphere is convex, so the tangent path does
  // not meet it again: one hit.
  const MoveParticleCall call = RunMove("SphereOblique");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.hit_count, 1);
  ExpectBothVelocitiesNear(result, {.x = 0.36, .y = 0.48, .z = 0}, 1e-8);
  ExpectVec3Near(result.position, {.x = -0.256, .y = 0.492, .z = 0}, 1e-8);
  const double grown_radius = 0.5 + kTolerance;  // m
  ExpectNearRelative(result.hit_fraction, (2 - std::sqrt(grown_radius * grown_radius - 0.09)) / 2, 1);
}

TEST(MoveParticle, SphereTangentWithinRounding_NeverEndsInside) {
  // The path runs tol beyond the sphere's surface, along the grown sphere's equator up to rounding: a hit or not, the
  // particle keeps its speed along x and ends outside.
  const MoveParticleCase scene = FindMoveParticleCase("SphereTangentWithinRounding");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_GE(ComputeEndDistance(scene, call.result.position), kTolerance / 2);
  EXPECT_LE(std::abs(call.result.velocity.for_this_step.x - 1), 0x1p-40);
}

TEST(MoveParticle, SphereStartWithinToleranceMovingAway_MovedOutSilently) {
  // Plan decision P12: moved along the normal {-1, 0, 0} to the rest gap; the velocity along the surface does not
  // approach, so the move is silent.
  const MoveParticleCase scene = FindMoveParticleCase("SphereStartWithinToleranceMovingAway");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Equal(call.result.velocity.for_next_step, {.x = 0, .y = 1, .z = 0});
  ExpectVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 1, .z = 0});
  EXPECT_GE(ComputeEndDistance(scene, call.result.position), kTolerance);
}

TEST(MoveParticle, CylinderRim_HitAtItsTrueDistance) {
  // The cylinder grown with sharp edges is entered at {0.5 + tol / 2, 0, 1 + tol}, Sqrt(1.25) tol from the rim; the rim
  // search goes on down the path to tol from the rim, Sqrt(0.75) tol above the cap's plane, where the normal is
  // {1 / 2, 0, Sqrt(3) / 2} and the response leaves {0, 0, -1} + (Sqrt(3) / 2) {1 / 2, 0, Sqrt(3) / 2} =
  // {Sqrt(3) / 4, 0, -1 / 4} (as built, the corner's {0.4, 0, -0.2}). Nothing moves the hit: the reaction is recorded
  // on the path.
  const MoveParticleCase scene = FindMoveParticleCase("CylinderRim");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.face, SolidFace::kHighRim);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_NEAR(result.hit_fraction, (2 - std::sqrt(0.75) * kTolerance) / 4, kTolerance / 4096);
  ExpectBothVelocitiesNear(result, {.x = std::numbers::sqrt3 / 4, .y = 0, .z = -0.25}, 1e-6);
  EXPECT_EQ(result.reaction.solid_index, 0U);
  const Vec3 hit_position = result.reaction.hit_position;
  EXPECT_EQ(hit_position.x, 0.5 + kTolerance / 2);
  EXPECT_NEAR(hit_position.z, 1 + std::sqrt(0.75) * kTolerance, kTolerance / 512);
  const Solid& cylinder = scene.solids.at(0);
  const double hit_distance = ComputeDistanceToSolid(cylinder, cylinder.pose_at_step_start, hit_position).distance;
  EXPECT_GE(hit_distance, kTolerance - 0x1p-40);
  EXPECT_LE(hit_distance, kTolerance + kTolerance / 1024 + 0x1p-40);
}

TEST(MoveParticle, CylinderRimFromTheSide_HitAtItsTrueDistance) {
  // Across the axis tol / 2 above the cap's plane: the rim search hits the rim Sqrt(0.75) tol beyond the wall, where
  // the normal is {Sqrt(3) / 2, 0, 1 / 2}, and the response leaves {-1 / 4, 0, Sqrt(3) / 4}.
  const MoveParticleCall call = RunMove("CylinderRimFromTheSide");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kHighRim);
  ExpectBothVelocitiesNear(call.result, {.x = -0.25, .y = 0, .z = std::numbers::sqrt3 / 4}, 1e-6);
  EXPECT_NEAR(call.result.reaction.hit_position.x, 0.5 + std::sqrt(0.75) * kTolerance, kTolerance / 512);
}

TEST(MoveParticle, CylinderCap_SlidesOnThePlane) {
  // The cap's grown plane z = 1 + tol is reached at 2 - tol s, at x = 0.3; the response keeps {0.1, 0, 0}, which the
  // 1 + tol s left carry to x = 0.4.
  const MoveParticleCall call = RunMove("CylinderCap");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  ExpectNearRelative(result.hit_fraction, (2 - kTolerance) / 3, 1);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  EXPECT_EQ(result.hit_count, 1);
  ExpectBothVelocitiesNear(result, {.x = 0.1, .y = 0, .z = 0}, 1e-15);
  ExpectVec3Near(result.position, {.x = 0.4, .y = 0, .z = 1 + kTolerance}, kPositionTolerance);
}

// What a particle pressed on a spinning top ends with: both velocities and the end position.
struct PressedOnTop {
  Vec3 velocity{};  // m/s, world.
  Vec3 end{};  // m, world.
};

// Runs one scene of a particle pressed on a spinning top and fails the calling test unless it succeeded with one hit,
// both velocities expected.velocity and the end expected.end, within 2 tol.
void ExpectPressedOnSpinningTop(std::string_view case_name, PressedOnTop expected) {
  const MoveParticleCall call = RunMove(case_name);
  ASSERT_EQ(call.error, ContactError::kNone) << case_name;
  EXPECT_EQ(call.result.hit_count, 1) << case_name;
  ExpectBothVelocitiesNear(call.result, expected.velocity, 1e-15);
  ExpectVec3Near(call.result.position, expected.end, kPositionTolerance);
}

TEST(MoveParticle, SpinningCylinderPressedOnCap_MovesOnlyThroughFriction) {
  // Plan decision P17: a spin about the axis is not a turn, so the cap keeps its closed form, and the response feels
  // the surface's {0, 1, 0} m/s through friction alone: none, a slip of friction times the push of 1 m/s, a stick.
  const double top = 0.5 + kTolerance;  // m
  ExpectPressedOnSpinningTop(
      "SpinningCylinderPressedOnCapNoFriction",
      {.velocity = {}, .end = {.x = 0.5, .y = 0, .z = top}}
  );
  ExpectPressedOnSpinningTop(
      "SpinningCylinderPressedOnCapSlipping",
      {.velocity = {.x = 0, .y = 0.25, .z = 0}, .end = {.x = 0.5, .y = 0.025, .z = top}}
  );
  ExpectPressedOnSpinningTop(
      "SpinningCylinderPressedOnCapSticking",
      {.velocity = {.x = 0, .y = 1, .z = 0}, .end = {.x = 0.5, .y = 0.1, .z = top}}
  );
}

TEST(MoveParticle, SpinningPlatePressedOn_MovesOnlyThroughFriction) {
  // A plate spinning about its normal does not turn either (plan decision P17 as amended): the same answers.
  const double top = 0.5 + kTolerance;  // m
  ExpectPressedOnSpinningTop("SpinningPlatePressedOnNoFriction", {.velocity = {}, .end = {.x = 0.5, .y = 0, .z = top}});
  ExpectPressedOnSpinningTop(
      "SpinningPlatePressedOnSlipping",
      {.velocity = {.x = 0, .y = 0.25, .z = 0}, .end = {.x = 0.5, .y = 0.025, .z = top}}
  );
  ExpectPressedOnSpinningTop(
      "SpinningPlatePressedOnSticking",
      {.velocity = {.x = 0, .y = 1, .z = 0}, .end = {.x = 0.5, .y = 0.1, .z = top}}
  );
}

TEST(MoveParticle, SlowlyTiltingCylinderCapSinkingAway_HitOnce) {
  // The tilt is below plan decision P17's bound, so the cap keeps its closed form, but its surface sinks at 3e-10 m/s
  // under the particle. The response leaves the particle sinking with it, and the next sweep reads that approach from
  // the same solid velocity as the response, so the cap is not hit again. Measured from the cap's linear velocity
  // alone it saw an approach of 3e-10 m/s, far beyond the response's allowance, and hit the cap at 0 in every sweep
  // until kTooManyHits, at every step length.
  const MoveParticleCall call = RunMove("SlowlyTiltingCylinderCapSinkingAway");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kHighZ);
  EXPECT_NEAR(call.result.velocity.for_this_step.z, -3e-10, 1e-15);
}

// Where the steps of MoveParticle leave a particle before the settle, and the faces of the hits along the path.
struct MoveBeforeSettle {
  Vec3 position{};  // m, world.
  std::vector<SolidFace> hit_faces;  // The start's response, when it has one, then each hit's face, in order.
};

// Runs the steps of MoveParticle on one scene up to the end of the last sweep, without the settle and the end check,
// and returns the position there with the faces of the hits; a failure fails the calling test.
MoveBeforeSettle RunMoveBeforeSettle(const MoveParticleCase& scene) {
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  MoveBeforeSettle before_settle;
  ParticleMove move;
  move.result.position = inputs.start_position;
  move.result.velocity = inputs.velocity;
  ContactFailure failure;
  ContactError error = ApplyStartRules(inputs, &move, &failure);
  if (move.result.hit_count > 0) {
    before_settle.hit_faces.push_back(move.result.face);
  }
  for (int sweep_index = 0; sweep_index <= kMaxHitsPerStep && error == ContactError::kNone; ++sweep_index) {
    const double duration = inputs.step_duration - move.elapsed;  // s
    if (duration <= 0) {
      break;
    }
    const Sweep sweep = {
        .start = move.result.position,
        .velocity_for_this_step = move.result.velocity.for_this_step,
        .start_elapsed = move.elapsed,
        .duration = duration,
    };
    const TouchedSolid touched = {
        .is_touched = move.has_response,
        .approach_allowance = move.this_step_approach_allowance,
    };
    FirstHit first_hit;
    error = FindFirstHitOfSweep(inputs, sweep, touched, &first_hit, nullptr);
    if (error == ContactError::kNone && first_hit.solid_index == kNoSolid) {
      move.result.position = move.result.position + move.result.velocity.for_this_step * duration;
      break;
    }
    if (error == ContactError::kNone) {
      before_settle.hit_faces.push_back(first_hit.at_hit.face);
      error = ApplyHit(inputs, first_hit, &move, &failure);
    }
  }
  EXPECT_EQ(error, ContactError::kNone);
  before_settle.position = move.result.position;
  return before_settle;
}

// The position RunMoveBeforeSettle returns.
Vec3 ComputePositionBeforeSettle(const MoveParticleCase& scene) {
  return RunMoveBeforeSettle(scene).position;
}

TEST(MoveParticle, RimOnFloorSweepingParticle_RimPassesOutsideItsTrueDistance) {
  // The rim, sliding over the floor and rising at 1e-3 m/s, passes 1.069 tol from the particle resting on the floor, so
  // its true distance never reaches tol: one hit, the start's on the floor, which takes the fall. As built, the sharp
  // corner of the grown cylinder was hit, and moving that hit to the rest gap lifted the particle off the floor: two
  // hits.
  const MoveParticleCase scene = FindMoveParticleCase("RimOnFloorSweepingParticle");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {}, 1e-15);
  EXPECT_NEAR(ComputeEndDistance(scene, call.result.position), kTolerance, kTolerance / 1024);
}

TEST(MoveParticle, RimOnFloorSweepingParticleLower_RimAndFloorRespondTogether) {
  // Rising at 9e-4 m/s the rim comes within tol: hit on kLowRim 1.0002 tol away, with the floor at exactly tol, which
  // the joint response keeps, since the particle is not moved. The particle stays on the floor, carried along by the
  // rim's push: two hits, the start's and the rim's (as built, the hit moved to the rest gap lifted it to 1.106 tol of
  // the floor).
  const MoveParticleCase scene = FindMoveParticleCase("RimOnFloorSweepingParticleLower");
  const MoveBeforeSettle before_settle = RunMoveBeforeSettle(scene);
  ASSERT_EQ(before_settle.hit_faces.size(), 2U);
  EXPECT_EQ(before_settle.hit_faces.at(1), SolidFace::kLowRim);
  EXPECT_NEAR(ComputeEndDistance(scene, before_settle.position), kTolerance, kTolerance / 1024);
  const Solid& cylinder = scene.solids.at(1);
  const SolidPose end_pose = ComputeSolidPose(cylinder, scene.inputs.step_duration);
  EXPECT_GE(ComputeDistanceToSolid(cylinder, end_pose, before_settle.position).distance, kTolerance / 2);
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  ExpectVec3Near(call.result.velocity.for_this_step, {.x = 0.875932, .y = 5.139e-5, .z = 0}, 1e-5);
}

TEST(MoveParticle, SlowlyTiltingCylinderRimCornerStart_RisingRimIsNotHit) {
  // The start is 1.27 tol from the rim, in its rim corner. The rim search reads the cylinder at the pose of the sweep's
  // start, and the rate from linear_velocity, 0: free. The rim rises 0.019 tol toward the particle during the step,
  // within plan decision P17's slack, and the true distance stays above 1.2 tol (as built a hit at 0, from the rate of
  // the surface's velocity, moved to the rest gap).
  const MoveParticleCase scene = FindMoveParticleCase("SlowlyTiltingCylinderRimCornerStart");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Equal(call.result.velocity.for_this_step, {});
  ExpectVec3Equal(call.result.velocity.for_next_step, {});
  EXPECT_GT(ComputeEndDistance(scene, call.result.position), 1.2 * kTolerance);
}

TEST(MoveParticle, SlowlyTiltingCylinderBesideRimLevelWithCap_EndsAboveTheBoundBeforeTheSettle) {
  // The hard case of a resting sweep on a cylinder tilting within plan decision P17's bound: a wide cylinder, radius
  // 1 m and half height 0.1 m (Length(half_size) = Sqrt(2.01) = 1.418 m), tilting about x at 0.99 of the bound for a
  // step of 1 s; a start beside the rim on the -y side, 0.9 tol beyond the wall and tol / 8 inside the cap plane's rest
  // gap, crossing the cap to y = 0.95 m while sinking toward its plane by 0.99 tol / 64. The sweep rests on the cap's
  // plane, read at the pose of the sweep's start, while the cap rises under the far end by the tilt angle times the
  // arm, 0.99 tol / (8 * 1.418) * 0.95 = 0.0829 tol. Before the settle: 7 / 8 - 0.99 / 64 - 0.0829 = 0.7766 tol,
  // measured 0.77661; above 39 / 64 tol (FindFirstHitOfSweep, step 3) and the floor 47 / 64 tol that P17's tol / 8
  // allows. The settle brings it back to tol.
  const Vec3 half_size = {.x = 1, .y = 1, .z = 0.1};
  const double tilt_rate = 0.99 * (kTolerance / 8) / math::Length(half_size);  // rad/s
  MoveParticleCase scene;
  scene.name = "WideCylinderTiltingAtTheBound";
  scene.solids = {
      ToSolid(
          Cylinder{.center = {}, .axis = kUnitZ, .radius = 1, .half_height = 0.1},
          {.linear_velocity = {}, .angular_velocity = {.x = tilt_rate, .y = 0, .z = 0}},
          1
      ),
  };
  scene.inputs = FindMoveParticleCase("CylinderRim").inputs;
  scene.inputs.start_position = {.x = 0, .y = -(1 + 0.9 * kTolerance), .z = 0.1 + 7.0 / 8 * kTolerance};
  const Vec3 velocity = {.x = 0, .y = 1.95, .z = -0.99 * kTolerance / 64};  // m/s
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  scene.inputs.step_duration = 1;
  ASSERT_FALSE(ComputeIsTurningSolid(ToSolidContactInputs(scene), 0));
  const double before_settle = ComputeEndDistance(scene, ComputePositionBeforeSettle(scene));  // m
  EXPECT_NEAR(before_settle, 0.7766 * kTolerance, 0.001 * kTolerance);
  EXPECT_GT(before_settle, 47.0 / 64 * kTolerance);
  EXPECT_GT(before_settle, 39.0 / 64 * kTolerance);
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  EXPECT_NEAR(ComputeEndDistance(scene, call.result.position), kTolerance, kTolerance / 1024);
  std::cout << "wide cylinder tilting at the bound, beside its rim level with its cap: before the settle "
            << before_settle / kTolerance << " tol\n";
}

TEST(MoveParticle, RimCornerBesideRestingFace_RoundingIsNoApproach) {
  // After the start's response on the upright, the sweep starts 1.15 tol from the rim, in its rim corner, beyond the
  // touched window, and moves square to the rim's normal: the rate is rounding, of either sign, within the response's
  // allowance, so the rim is not hit. Compared with 0, one of the two directions hit it at 0.
  for (const std::string_view case_name : {"RimCornerBesideRestingFaceDown", "RimCornerBesideRestingFaceUp"}) {
    const MoveParticleCall call = RunMove(case_name);
    ASSERT_EQ(call.error, ContactError::kNone) << case_name;
    EXPECT_EQ(call.result.hit_count, 1) << case_name;
    EXPECT_EQ(call.result.solid_index, 1U) << case_name;
  }
}

// What a rim hit must give: both velocities after MoveParticle within velocity_tolerance of velocity, and the distance
// at the hit position, at the pose of the hit's time, within distance_tolerance of tol.
struct RimHitExpectation {
  Vec3 velocity{};  // m/s, world.
  double velocity_tolerance = 0;  // m/s.
  double distance_tolerance = 0;  // m.
};

// Fails the calling test unless the scene's first hit is on kHighRim, within distance_tolerance of tol from the
// cylinder, its first solid, at the pose of the hit's time, and ApplyHit leaves the particle on the path, at start +
// velocity * first_hit.elapsed bit for bit, where the cylinder's distance is first_hit.at_hit.distance bit for bit (no
// hit is moved, plan decision P21).
void ExpectRimHitOnThePath(const MoveParticleCase& scene, double distance_tolerance) {
  ParticleMove move;
  move.result.position = scene.inputs.start_position;
  move.result.velocity = scene.inputs.velocity;
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  const Vec3 velocity = inputs.velocity.for_this_step;
  FirstHit first_hit;
  ASSERT_EQ(
      FindFirstHit(inputs, inputs.start_position, velocity, 0, inputs.step_duration, &first_hit),
      ContactError::kNone
  );
  ASSERT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, distance_tolerance);
  ContactFailure failure;
  ASSERT_EQ(ApplyHit(inputs, first_hit, &move, &failure), ContactError::kNone);
  ExpectVec3Equal(move.result.position, inputs.start_position + velocity * first_hit.elapsed);
  const Solid& cylinder = scene.solids.at(0);
  const SolidPose hit_pose = ComputeSolidPose(cylinder, first_hit.elapsed);
  EXPECT_EQ(ComputeDistanceToSolid(cylinder, hit_pose, move.result.position).distance, first_hit.at_hit.distance);
}

// Fails the calling test unless the scene's rim hit is on the path (ExpectRimHitOnThePath) and MoveParticle succeeds
// with one hit and both velocities expected.velocity.
void ExpectRimHitAtTrueDistance(std::string_view case_name, RimHitExpectation expected) {
  const MoveParticleCase scene = FindMoveParticleCase(case_name);
  SCOPED_TRACE(case_name);
  ExpectRimHitOnThePath(scene, expected.distance_tolerance);
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, expected.velocity, expected.velocity_tolerance);
}

TEST(MoveParticle, CylinderRimNearRestGap_HitAtItsTrueDistance) {
  // The entry is Sqrt(17 / 16) tol = 1.0308 tol from the rim; the rim search stops 3e-5 tol outside tol, where the wall
  // distance is tol / 4 and the normal {1 / 4, 0, Sqrt(15) / 4}: the response leaves {Sqrt(15) / 16, 0, -1 / 16} (as
  // built, moved along the corner's normal, {4 / 17, 0, -1 / 17}).
  ExpectRimHitAtTrueDistance(
      "CylinderRimNearRestGap",
      {
          .velocity = {.x = std::sqrt(15.0) / 16, .y = 0, .z = -1.0 / 16},
          .velocity_tolerance = 2e-5,
          .distance_tolerance = kTolerance / 1024,
      }
  );
}

TEST(MoveParticle, RisingCylinderRim_HitReadAtThePoseOfTheHit) {
  // The rim hit is read at the pose of 2 - Sqrt(0.75) tol s, 2 m above the start pose. The particle at rest takes the
  // rim's 1 m/s along its normal {1 / 2, 0, Sqrt(3) / 2}: {Sqrt(3) / 4, 0, 3 / 4}.
  ExpectRimHitAtTrueDistance(
      "RisingCylinderRim",
      {
          .velocity = {.x = std::numbers::sqrt3 / 4, .y = 0, .z = 0.75},
          .velocity_tolerance = 1e-6,
          .distance_tolerance = kTolerance / 1024,
      }
  );
}

TEST(MoveParticle, SlowlyTiltingCylinderRim_HitAtItsTrueDistance) {
  // A tilt within plan decision P17's bound keeps the closed form and the rim search, which read the cylinder at the
  // pose of the sweep's start; at the pose of the hit the distance differs by the tilt's slack, 0.024 tol here.
  ExpectRimHitAtTrueDistance(
      "SlowlyTiltingCylinderRim",
      {
          .velocity = {.x = std::numbers::sqrt3 / 4, .y = 0, .z = -0.25},
          .velocity_tolerance = 1e-5,
          .distance_tolerance = kTolerance / 16,
      }
  );
}

// --- Flush seams and a rim's neighbourhood (plan decision P21) ---

// Runs a scene from another start, with both velocities as the scene has them.
MoveParticleCall RunMoveFrom(std::string_view case_name, Vec3 start_position) {
  MoveParticleCase scene = FindMoveParticleCase(case_name);
  scene.inputs.start_position = start_position;
  return RunMove(scene);
}

// Another start of a scene: the position, both velocities and the step.
struct SceneStart {
  Vec3 position{};  // m, world.
  Vec3 velocity{};  // m/s, world, both velocities.
  double step_duration = 0;  // s.
  double friction = 0;  // No unit.
};

// A scene from another start.
MoveParticleCase BuildSceneFrom(std::string_view case_name, SceneStart start) {
  MoveParticleCase scene = FindMoveParticleCase(case_name);
  scene.inputs.start_position = start.position;
  scene.inputs.velocity = {.for_next_step = start.velocity, .for_this_step = start.velocity};
  scene.inputs.step_duration = start.step_duration;
  scene.inputs.friction = start.friction;
  return scene;
}

// Runs a scene from another start.
MoveParticleCall RunMoveWith(std::string_view case_name, SceneStart start) {
  return RunMove(BuildSceneFrom(case_name, start));
}

TEST(MoveParticle, StackedDrumsOfOneRadius_SlideDownTheSeamKeepsItsVelocity) {
  // Down the upper drum's wall at its rest gap and past the seam: the start beside the lower drum's high rim rests on
  // that drum's wall, taken as unbounded along the axis, so the lower drum is not hit, and the velocity stays {0, 0,
  // -1} bit for bit. The same from one rounding unit and from tol / 32 inside the rest gap, which the start rules move
  // out. As built, the sharp grown rim edge was hit 1.41 tol from the rim and the particle left at {0.5, 0, -0.5}.
  const MoveParticleCase scene = FindMoveParticleCase("StackedDrums");
  const Solid& lower_drum = scene.solids.at(1);
  const double rest_x = 0.5 + kTolerance;  // m
  for (const double start_x : {rest_x, rest_x - 0x1p-53, rest_x - kTolerance / 32}) {
    const MoveParticleCall call = RunMoveFrom("StackedDrums", {.x = start_x, .y = 0, .z = 1.8});
    ASSERT_EQ(call.error, ContactError::kNone) << start_x;
    EXPECT_EQ(call.result.kind, ContactKind::kNone);
    EXPECT_EQ(call.result.hit_count, 0);
    ExpectVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 0, .z = -1});
    ExpectVec3Equal(call.result.velocity.for_next_step, {.x = 0, .y = 0, .z = -1});
    ExpectVec3Near(call.result.position, {.x = rest_x, .y = 0, .z = 0.8}, 0x1p-50);
    const double end_distance =
        ComputeDistanceToSolid(lower_drum, lower_drum.pose_at_step_start, call.result.position).distance;  // m
    EXPECT_NEAR(end_distance, kTolerance, 0x1p-50);
  }
}

TEST(MoveParticle, PuckFlushInFloor_SlideOntoItAndOffKeepsItsVelocity) {
  // Along the floor at its rest gap onto the puck lying flush in it, and off it: the start beside the puck's rim rests
  // on the plane of its top, so the puck is not hit, and the velocity is kept bit for bit. As built, the sharp grown
  // rim edge stopped the particle onto it, which left at {-0.5, 0, 0.5}.
  const MoveParticleCase scene = FindMoveParticleCase("PuckFlushInFloor");
  const Solid& floor = scene.solids.at(0);
  const Solid& puck = scene.solids.at(1);
  const MoveParticleCall onto = RunMove(scene);
  ASSERT_EQ(onto.error, ContactError::kNone);
  EXPECT_EQ(onto.result.kind, ContactKind::kNone);
  EXPECT_EQ(onto.result.hit_count, 0);
  ExpectVec3Equal(onto.result.velocity.for_this_step, -kUnitX);
  ExpectVec3Equal(onto.result.velocity.for_next_step, -kUnitX);
  ExpectVec3Near(onto.result.position, {.x = 0.4 + 3 * kTolerance, .y = 0, .z = kTolerance}, kTolerance / 1024);
  const Vec3 onto_end = onto.result.position;
  EXPECT_NEAR(
      ComputeDistanceToSolid(floor, floor.pose_at_step_start, onto_end).distance,
      kTolerance,
      kTolerance / 1024
  );
  EXPECT_NEAR(ComputeDistanceToSolid(puck, puck.pose_at_step_start, onto_end).distance, kTolerance, kTolerance / 1024);
  const MoveParticleCall off = RunMoveWith(
      "PuckFlushInFloor",
      {.position = {.x = 0.45, .y = 0, .z = kTolerance}, .velocity = kUnitX, .step_duration = 0.1}
  );
  ASSERT_EQ(off.error, ContactError::kNone);
  EXPECT_EQ(off.result.hit_count, 0);
  ExpectVec3Equal(off.result.velocity.for_this_step, kUnitX);
  ExpectVec3Equal(off.result.velocity.for_next_step, kUnitX);
  ExpectVec3Near(off.result.position, {.x = 0.55, .y = 0, .z = kTolerance}, kTolerance / 1024);
  const Vec3 off_end = off.result.position;
  EXPECT_NEAR(ComputeDistanceToSolid(floor, floor.pose_at_step_start, off_end).distance, kTolerance, kTolerance / 1024);
}

// Fails the calling test unless a move succeeded with no hit and both velocities velocity, bit for bit.
void ExpectFreeWithVelocity(const MoveParticleCall& call, Vec3 velocity) {
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Equal(call.result.velocity.for_this_step, velocity);
  ExpectVec3Equal(call.result.velocity.for_next_step, velocity);
}

// Fails the calling test unless a move succeeded with one hit and both velocities within 1e-15 of velocity.
void ExpectOneHitWithVelocity(const MoveParticleCall& call, Vec3 velocity) {
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, velocity, 1e-15);
}

TEST(MoveParticle, StackedDrums_StepStartingJustAboveTheSeamKeepsItsVelocity) {
  // A step that starts 3 / 16 or 5 / 16 tol above the seam, beside the lower drum's rim 1.0174 or 1.0477 tol away, at
  // the level of its wall: the lower drum is no active surface there (plan decision P21, item 16), so nothing answers
  // the rim, and the step is free with {0, 0, -1} bit for bit (as built the start's response on the rim turned it to
  // {0.1811321, 0, -0.9660377} and {0.2846975, 0, -0.9110320}). Pushed into the upper wall at 0.16 m/s, the start's
  // response is on that wall alone: {0, 0, -1}, or {0, 0, -0.952} with friction 0.3. From tol / 32 inside the wall's
  // rest gap and tol / 8 above the seam, inside the rim's own touching window, the sweep tries the wall first: free
  // ({0.1230769, 0, -0.9846154} as built).
  const double rest_x = 0.5 + kTolerance;  // m
  const Vec3 down = -kUnitZ;
  const Vec3 pressed = {.x = -0.16, .y = 0, .z = -1};
  for (const double height_share : {3.0 / 16, 5.0 / 16}) {
    const Vec3 start = {.x = rest_x, .y = 0, .z = 1 + height_share * kTolerance};
    ExpectFreeWithVelocity(
        RunMoveWith("StackedDrums", {.position = start, .velocity = down, .step_duration = 0.1}),
        down
    );
    const MoveParticleCall pushed =
        RunMoveWith("StackedDrums", {.position = start, .velocity = pressed, .step_duration = 0.1});
    ExpectOneHitWithVelocity(pushed, down);
    EXPECT_EQ(pushed.result.face, SolidFace::kCurved);
    EXPECT_EQ(pushed.result.solid_index, 0U);
    ExpectOneHitWithVelocity(
        RunMoveWith("StackedDrums", {.position = start, .velocity = pressed, .step_duration = 0.1, .friction = 0.3}),
        {.x = 0, .y = 0, .z = -0.952}
    );
  }
  const Vec3 inside = {.x = rest_x - kTolerance / 32, .y = 0, .z = 1 + kTolerance / 8};
  ExpectFreeWithVelocity(
      RunMoveWith("StackedDrums", {.position = inside, .velocity = down, .step_duration = 0.1}),
      down
  );
}

TEST(MoveParticle, StackedDrumsLowerOneWider_ALedgeDeeperThanAnEighthIsSeen) {
  // The stacked drums with the lower one wider by 3 / 16 tol, a ledge: from the upper wall's rest gap, 0.6 tol above
  // the seam, the lower wall's gap is -3 / 16 tol, deeper than the level window's tol / 8, so the lower drum is an
  // active surface, 1.010 tol away, and the falling start is answered on its rim at once: hit_fraction 0. A window
  // twice as deep would leave the drum out, and the sweep would meet the rim a little later instead.
  MoveParticleCase scene = FindMoveParticleCase("StackedDrums");
  const Cylinder wider_lower = {
      .center = {.x = 0, .y = 0, .z = 0.5},
      .axis = kUnitZ,
      .radius = 0.5 + 3.0 / 16 * kTolerance,
      .half_height = 0.5,
  };
  scene.solids.at(1) = ToSolid(wider_lower, kNoMotion, 2);
  scene.inputs.start_position = {.x = 0.5 + kTolerance, .y = 0, .z = 1 + 0.6 * kTolerance};
  scene.inputs.step_duration = 0.1;
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.face, SolidFace::kHighRim);
  EXPECT_EQ(call.result.hit_fraction, 0);
}

TEST(MoveParticle, StackedDrums_HitOnTheUpperWallJustAboveTheSeamLeavesTheLowerOut) {
  // Thrown at the upper drum's wall from 0.01 m out and above, the particle meets it 3 / 16 tol above the seam, level
  // with the lower drum's wall: at that hit the lower drum is no active surface either (plan decision P21, item 16 a,
  // at a hit as at the start), so the response removes the approach to the upper wall alone: {0, 0, -1}. Listed, the
  // lower rim would turn it to {0.147, 0, -0.785}.
  const MoveParticleCall call = RunMoveWith(
      "StackedDrums",
      {
          .position = {.x = 0.5 + kTolerance + 0.01, .y = 0, .z = 1 + 3.0 / 16 * kTolerance + 0.01},
          .velocity = {.x = -1, .y = 0, .z = -1},
          .step_duration = 0.1,
      }
  );
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 0U);
  ExpectBothVelocitiesNear(call.result, -kUnitZ, 1e-15);
}

TEST(MoveParticle, RisingPuckFlushInFloor_LevelReadAtThePoseOfTheHit) {
  // The puck and the floor rising together at 0.1 m/s, a particle landing on the floor 3 / 16 tol beyond the rim: at
  // the hit, read at the pose of the hit's time, the particle is level with the puck's top, so the floor alone
  // responds and the particle keeps {-1, 0, 0.1}. Read at the pose of the step's start, the puck would be 1 mm lower,
  // not level, and its rim would turn the particle.
  MoveParticleCase scene = FindMoveParticleCase("PuckFlushInFloor");
  const SolidMotion rising = {.linear_velocity = {.x = 0, .y = 0, .z = 0.1}, .angular_velocity = {}};
  scene.solids.at(0) = move_particle_cases::BuildFlatSolid({}, kUnitZ, rising, 1);
  const Cylinder puck = {.center = {.x = 0, .y = 0, .z = -0.1}, .axis = kUnitZ, .radius = 0.5, .half_height = 0.1};
  scene.solids.at(1) = ToSolid(puck, rising, 2);
  scene.inputs.start_position = {.x = 0.5 + 3.0 / 16 * kTolerance + 0.01, .y = 0, .z = kTolerance + 0.011};
  scene.inputs.velocity = {.for_next_step = {.x = -1, .y = 0, .z = -1}, .for_this_step = {.x = -1, .y = 0, .z = -1}};
  scene.inputs.step_duration = 0.1;
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 0U);
  ExpectBothVelocitiesNear(call.result, {.x = -1, .y = 0, .z = 0.1}, 1e-15);
}

// The rim scene of the known choice of an approach with no response: the fixed cylinder of CylinderRim, a start at the
// rim's rest gap 0.3 tol beyond the wall and level with the cap's plane (cap gap Sqrt(0.91) - 1 = -0.046 tol), and a
// velocity along the rim's normal that sinks approach.sink_share tol in a step of 0.1 s; approach.x_units moves the
// start's x by that many units in the last place.
struct RimSlowApproach {
  double sink_share = 0;  // No unit: the sink along the rim's normal over the step, in tolerances.
  int x_units = 0;  // Units in the last place added to the start's x.
};

// Builds the rim scene of one slow approach (RimSlowApproach).
MoveParticleCase BuildRimSlowApproachScene(RimSlowApproach approach) {
  const double sink_share = approach.sink_share;
  const int x_units = approach.x_units;
  MoveParticleCase scene = FindMoveParticleCase("CylinderRim");
  const Vec3 rim_normal = math::Normalized(Vec3{.x = 0.3, .y = 0, .z = std::sqrt(1 - 0.3 * 0.3)});
  Vec3 start = Vec3{.x = 0.5, .y = 0, .z = 1} + kTolerance * rim_normal;
  for (int unit = 0; unit < x_units; ++unit) {
    start.x = std::nextafter(start.x, 1.0);
  }
  scene.inputs.start_position = start;
  const Vec3 velocity = (-sink_share * kTolerance / 0.1) * rim_normal;  // m/s
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  scene.inputs.step_duration = 0.1;
  return scene;
}

// Fails the calling test unless the rim's slow approach of sink_share tol, from the start as built, gets no hit and no
// reaction, is (1 - sink_share) tol from the rim before the settle and at tol after it.
void ExpectRimSlowApproachUnanswered(double sink_share) {
  SCOPED_TRACE(sink_share);
  const MoveParticleCase scene = BuildRimSlowApproachScene({.sink_share = sink_share, .x_units = 0});
  const double before_settle = ComputeEndDistance(scene, ComputePositionBeforeSettle(scene));  // m
  EXPECT_NEAR(before_settle, (1 - sink_share) * kTolerance, kTolerance / 1024);
  EXPECT_GT(before_settle, 0);
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  EXPECT_EQ(call.result.reaction.solid_index, kNoSolid);
  ExpectVec3Equal(call.result.velocity.for_this_step, scene.inputs.velocity.for_this_step);
  ExpectVec3Equal(call.result.velocity.for_next_step, scene.inputs.velocity.for_next_step);
  EXPECT_NEAR(ComputeEndDistance(scene, call.result.position), kTolerance, kTolerance / 1024);
}

TEST(MoveParticle, RimLevelWithCapSlowApproach_NoResponse_KnownChoice) {
  // A known choice (plan decision P21, item 16): at the rim's rest gap, level with the cap's plane, a particle
  // approaching the rim along its normal by 0.9 tol in the step gets no response. The cylinder is no active surface at
  // the start; the sweep does not rest on the cap's plane (it sinks 0.86 tol toward it), so the plain touching rule on
  // the rim rules, free while the path sinks less than tol. No hit, no reaction, both velocities kept as given (-9
  // tol/s along the rim's normal, or -9.99 tol/s for the sink of 0.999 tol); before the settle the particle is 0.1 tol
  // from the rim, never inside, and the settle lifts it back to tol. From the rest gap a sink below tol keeps d >= 0
  // (the distance is convex along the path): a sink of 0.999 tol ends 0.001 tol away. This side holds because the
  // start, as built here, has a computed distance at or below tol (see the rounding test below).
  ExpectRimSlowApproachUnanswered(0.9);
  ExpectRimSlowApproachUnanswered(0.999);
}

TEST(MoveParticle, RimLevelWithCapSlowApproach_RoundingOfTheStartDecides_KnownChoice) {
  // Which side of the known choice a start at the rim's rest gap falls on is decided by the rounding of its distance.
  // The start of the test above: computed at or below tol, the plain touching rule frees the path, no hit, both
  // velocities kept (-9 tol/s along the rim's normal). The same start moved by one unit in the last place on x:
  // computed a hair above tol, it is outside, entry 0, and the rim search finds the gap within tol / 1024 and closing,
  // a hit at 0 whose response removes the whole approach along the rim's normal: both velocities {0, 0, 0}. Both keep
  // the path outside the cylinder and end at tol after the settle.
  const MoveParticleCase on_rest_gap = BuildRimSlowApproachScene({.sink_share = 0.9, .x_units = 0});
  const MoveParticleCall free_call = RunMove(on_rest_gap);
  ASSERT_EQ(free_call.error, ContactError::kNone);
  EXPECT_EQ(free_call.result.hit_count, 0);
  EXPECT_NEAR(ComputeEndDistance(on_rest_gap, free_call.result.position), kTolerance, kTolerance / 1024);
  const MoveParticleCase a_unit_above = BuildRimSlowApproachScene({.sink_share = 0.9, .x_units = 1});
  const MoveParticleCall hit_call = RunMove(a_unit_above);
  ASSERT_EQ(hit_call.error, ContactError::kNone);
  EXPECT_EQ(hit_call.result.hit_count, 1);
  EXPECT_EQ(hit_call.result.face, SolidFace::kHighRim);
  EXPECT_EQ(hit_call.result.hit_fraction, 0);
  ExpectBothVelocitiesNear(hit_call.result, {}, 1e-15);
  EXPECT_NEAR(ComputeEndDistance(a_unit_above, hit_call.result.position), kTolerance, kTolerance / 1024);
}

TEST(MoveParticle, PuckFlushInFloor_StepStartingJustBeforeTheSeamKeepsItsVelocity) {
  // The puck's counterpart: a step that starts 3 / 16 or 5 / 16 tol beyond its rim, level with its top's plane, is
  // free with {-1, 0, 0} bit for bit (as built {-0.9660377, 0, 0.1811321} and {-0.9110320, 0, 0.2846975}); pushed into
  // the floor, the start's response is the floor's alone.
  for (const double offset_share : {3.0 / 16, 5.0 / 16}) {
    const Vec3 start = {.x = 0.5 + offset_share * kTolerance, .y = 0, .z = kTolerance};
    ExpectFreeWithVelocity(
        RunMoveWith("PuckFlushInFloor", {.position = start, .velocity = -kUnitX, .step_duration = 0.1}),
        -kUnitX
    );
    const MoveParticleCall pushed = RunMoveWith(
        "PuckFlushInFloor",
        {.position = start, .velocity = {.x = -1, .y = 0, .z = -0.16}, .step_duration = 0.1}
    );
    ExpectOneHitWithVelocity(pushed, -kUnitX);
    EXPECT_EQ(pushed.result.face, SolidFace::kHighZ);
    EXPECT_EQ(pushed.result.solid_index, 0U);
  }
}

TEST(MoveParticle, LoneCylinderBesideRimLevelWithCap_FallingIsHitAtTheTrueDistance) {
  // A lone cylinder and a start 5 / 16 tol beyond its wall, level with its cap's plane, 1.048 tol from the rim: the
  // cylinder is no active surface at the start, so falling is answered by the rim search where the true distance is
  // tol, with the normal {0.3125, 0, 0.95}: 0.95 * {0.3125, 0, 0.95} - {0, 0, 1} = {0.2969, 0, -0.0976} (as built the
  // start's response 1.048 tol away gave {0.2847, 0, -0.0890}). Moving inward along the cap's plane is free.
  MoveParticleCase scene = FindMoveParticleCase("CylinderRim");
  scene.solids.at(0).reports_reaction = false;
  scene.inputs.start_position = {.x = 0.5 + 5.0 / 16 * kTolerance, .y = 0, .z = 1 + kTolerance};
  scene.inputs.step_duration = 0.1;
  scene.inputs.velocity = {.for_next_step = -kUnitZ, .for_this_step = -kUnitZ};
  const MoveParticleCall falling = RunMove(scene);
  ASSERT_EQ(falling.error, ContactError::kNone);
  EXPECT_EQ(falling.result.hit_count, 1);
  EXPECT_EQ(falling.result.face, SolidFace::kHighRim);
  EXPECT_GT(falling.result.hit_fraction, 0);
  ExpectBothVelocitiesNear(falling.result, Vec3{.x = 0.95 * 0.3125, .y = 0, .z = 0.95 * 0.95 - 1}, 2e-4);
  scene.inputs.velocity = {.for_next_step = -kUnitX, .for_this_step = -kUnitX};
  ExpectFreeWithVelocity(RunMove(scene), -kUnitX);
}

// A row of the runs across a seam in many steps: a scene, the point where the path crosses the seam, the direction
// back from there to the starts, the path's direction, the push's direction, the step as a share of tol, the push and
// the friction.
struct SeamRow {
  std::string_view case_name;
  Vec3 seam_point{};  // m, world.
  Vec3 back_from_seam{};  // No unit, unit vector.
  Vec3 along_path{};  // No unit, unit vector; the path moves along it at 1 m/s.
  Vec3 push_direction{};  // No unit, unit vector, into the surface slid on.
  double step_share = 0;  // No unit: the step is step_share tol long, in step_share tol s.
  double push = 0;  // m/s.
  double friction = 0;  // No unit.
};

// Runs one particle across the seam of row_scene, row's scene built once, in Floor(12 / step_share) + 2 steps from
// offset tol before it, the same input
// velocity every step from the last end, and says whether every step succeeded with no hit without a push and one
// with, kept the speed along the path that friction on the surface slid on leaves (1 - friction * push, within 1e-12),
// and stayed at tol - tol / 1024 or more from every solid before the settle.
bool ComputeIsSeamRunRight(const MoveParticleCase& row_scene, const SeamRow& row, double offset) {
  const double step_duration = row.step_share * kTolerance;  // s
  const Vec3 velocity = row.along_path + row.push * row.push_direction;  // m/s
  const double kept_speed = 1 - row.friction * row.push;  // m/s
  const int step_count = static_cast<int>(std::floor(12 / row.step_share)) + 2;
  const bool is_pushed = row.push > 0;
  Vec3 position = row.seam_point + offset * row.back_from_seam;
  for (int step_index = 0; step_index < step_count; ++step_index) {
    MoveParticleCase scene = row_scene;
    scene.inputs.start_position = position;
    scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    scene.inputs.step_duration = step_duration;
    scene.inputs.friction = row.friction;
    const MoveParticleCall call = RunMove(scene);
    if (call.error != ContactError::kNone) {
      return false;
    }
    const bool is_right_hit_count = call.result.hit_count == (is_pushed ? 1 : 0);
    const double speed = math::Dot(call.result.velocity.for_this_step, row.along_path);  // m/s
    const Vec3 before_settle = ComputePositionBeforeSettle(scene);
    bool stays_out = true;
    for (const Solid& solid : scene.solids) {
      const SolidPose end_pose = ComputeSolidPose(solid, step_duration);
      stays_out = stays_out
          && ComputeDistanceToSolid(solid, end_pose, before_settle).distance >= kTolerance - kTolerance / 1024;
    }
    if (!is_right_hit_count || std::abs(speed - kept_speed) > 1e-12 || !stays_out) {
      return false;
    }
    position = call.result.position;
  }
  return true;
}

// Runs the 18 rows of runs (three steps, three pushes and frictions) for one kind of seam, 400 offsets 4 + s (k + 0.5)
// / 400 tol before the seam each, and fails the calling test unless every run is right.
void ExpectSeamRowsKeepTheSpeed(SeamRow row) {
  // The scene is built once: FindMoveParticleCase builds every scene, about 40 times the cost of a move.
  const MoveParticleCase row_scene = FindMoveParticleCase(row.case_name);
  constexpr std::array<double, 3> kStepShares = {0.3, 0.7, 3};
  constexpr std::array<std::array<double, 2>, 3> kPushesAndFrictions = {{{0, 0}, {0.16, 0}, {0.16, 0.3}}};
  for (const double step_share : kStepShares) {
    for (const auto& push_and_friction : kPushesAndFrictions) {
      row.step_share = step_share;
      row.push = push_and_friction.at(0);
      row.friction = push_and_friction.at(1);
      int wrong_runs = 0;
      for (int offset_index = 0; offset_index < 400; ++offset_index) {
        const double offset = (4 + step_share * (offset_index + 0.5) / 400) * kTolerance;  // m
        wrong_runs += ComputeIsSeamRunRight(row_scene, row, offset) ? 0 : 1;
      }
      std::cout << "seam runs " << row.case_name << ", step " << step_share << " tol, push " << row.push
                << ", friction " << row.friction << ": " << wrong_runs << " of 400 wrong\n";
      EXPECT_EQ(wrong_runs, 0) << row.case_name << " " << step_share;
    }
  }
}

TEST(MoveParticle, SeamsOverManySteps_Cylinders) {
  // Down the stacked drums and along the floor onto the puck, from 4 to 4 + s tol before the seam, in steps of s = 0.3,
  // 0.7 and 3 tol, unpressed or pressed at 0.16 m/s into the surface slid on, with friction 0 or 0.3: every step keeps
  // the speed (plan decision P21, item 16). As built 400, 205 or 224 and 48 runs of 400 lost speed at s = 0.3, 0.7 and
  // 3: some step started within tol + tol / 16 of the next drum's rim and was answered on it.
  ExpectSeamRowsKeepTheSpeed({
      .case_name = "StackedDrums",
      .seam_point = {.x = 0.5 + kTolerance, .y = 0, .z = 1},
      .back_from_seam = kUnitZ,
      .along_path = -kUnitZ,
      .push_direction = -kUnitX,
  });
  ExpectSeamRowsKeepTheSpeed({
      .case_name = "PuckFlushInFloor",
      .seam_point = {.x = 0.5, .y = 0, .z = kTolerance},
      .back_from_seam = kUnitX,
      .along_path = -kUnitX,
      .push_direction = -kUnitZ,
  });
}

TEST(MoveParticle, DrumStandingOnFloor_GrooveStopsAndLetsSlideAround) {
  // A guard: the groove between a drum and the floor it stands on is a true corner. Along the floor into it, the wall
  // is hit at 0.5 - tol s; down the wall into it, the floor at 0.7 - tol s; each stops the particle at tol from both.
  // Along the groove, the particle rests on both and is free.
  const MoveParticleCall along_floor = RunMove("DrumStandingOnFloor");
  ASSERT_EQ(along_floor.error, ContactError::kNone);
  EXPECT_EQ(along_floor.result.hit_count, 1);
  EXPECT_EQ(along_floor.result.solid_index, 1U);
  EXPECT_EQ(along_floor.result.face, SolidFace::kCurved);
  ExpectNearRelative(along_floor.result.hit_fraction, 0.5 - kTolerance, 1);
  ExpectBothVelocitiesNear(along_floor.result, {}, 1e-15);
  const Vec3 groove = {.x = 0.5 + kTolerance, .y = 0, .z = kTolerance};
  ExpectVec3Near(along_floor.result.position, groove, kTolerance / 1024);
  const MoveParticleCall down_wall = RunMoveWith(
      "DrumStandingOnFloor",
      {.position = {.x = 0.5 + kTolerance, .y = 0, .z = 0.7}, .velocity = -kUnitZ, .step_duration = 1}
  );
  ASSERT_EQ(down_wall.error, ContactError::kNone);
  EXPECT_EQ(down_wall.result.hit_count, 1);
  EXPECT_EQ(down_wall.result.solid_index, 0U);
  EXPECT_EQ(down_wall.result.face, SolidFace::kHighZ);
  ExpectNearRelative(down_wall.result.hit_fraction, 0.7 - kTolerance, 1);
  ExpectBothVelocitiesNear(down_wall.result, {}, 1e-15);
  ExpectVec3Near(down_wall.result.position, groove, kTolerance / 1024);
  const MoveParticleCall around =
      RunMoveWith("DrumStandingOnFloor", {.position = groove, .velocity = kUnitY, .step_duration = 0.25});
  ASSERT_EQ(around.error, ContactError::kNone);
  EXPECT_EQ(around.result.hit_count, 0);
  EXPECT_NEAR(around.result.position.y, 0.25, kTolerance / 1024);
}

TEST(MoveParticle, RimCrestUnderFloor_IsNotHit) {
  // The crest of a slanted cylinder's high rim lies 0.41 tol under the floor's surface; the particle sliding at the
  // floor's rest gap passes it 1.41 tol away, beyond the true distance's reach, though through the sharp grown corner:
  // no hit, and the end at tol from the floor. As built, the corner was hit and moved to the rest gap, which pulled
  // the particle 0.41 tol toward the floor.
  const MoveParticleCase scene = FindMoveParticleCase("RimCrestUnderFloor");
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Equal(call.result.velocity.for_this_step, kUnitX);
  ExpectVec3Equal(call.result.velocity.for_next_step, kUnitX);
  EXPECT_NEAR(call.result.position.z, kTolerance, kTolerance / 1024);
}

TEST(MoveParticle, SlideOffCapPastRim_AndUpWallPastRim_OneHitThenFree) {
  // A guard. On the cap's rest gap, 8 tol inside the rim, pressed in and sliding out: the start's response keeps
  // {1, 0, 0}, and the slide past the rim is free, ending 8 tol beyond the wall and tol above the cap's plane, Sqrt(65)
  // tol from the rim. The same up the wall past the rim.
  const MoveParticleCase scene = FindMoveParticleCase("CylinderRim");
  const Solid& cylinder = scene.solids.at(0);
  const double slide_duration = 16 * kTolerance;  // s
  const MoveParticleCall off_cap = RunMoveWith(
      "CylinderRim",
      {
          .position = {.x = 0.5 - 8 * kTolerance, .y = 0, .z = 1 + kTolerance},
          .velocity = {.x = 1, .y = 0, .z = -0.5},
          .step_duration = slide_duration,
      }
  );
  ASSERT_EQ(off_cap.error, ContactError::kNone);
  EXPECT_EQ(off_cap.result.hit_count, 1);
  EXPECT_EQ(off_cap.result.hit_fraction, 0);
  EXPECT_EQ(off_cap.result.face, SolidFace::kHighZ);
  ExpectBothVelocitiesNear(off_cap.result, kUnitX, 1e-15);
  const double rim_distance = std::sqrt(65.0) * kTolerance;  // m
  EXPECT_NEAR(
      ComputeDistanceToSolid(cylinder, cylinder.pose_at_step_start, off_cap.result.position).distance,
      rim_distance,
      kTolerance / 1024
  );
  const MoveParticleCall up_wall = RunMoveWith(
      "CylinderRim",
      {
          .position = {.x = 0.5 + kTolerance, .y = 0, .z = 1 - 8 * kTolerance},
          .velocity = {.x = -0.5, .y = 0, .z = 1},
          .step_duration = slide_duration,
      }
  );
  ASSERT_EQ(up_wall.error, ContactError::kNone);
  EXPECT_EQ(up_wall.result.hit_count, 1);
  EXPECT_EQ(up_wall.result.face, SolidFace::kCurved);
  ExpectBothVelocitiesNear(up_wall.result, kUnitZ, 1e-15);
  EXPECT_NEAR(
      ComputeDistanceToSolid(cylinder, cylinder.pose_at_step_start, up_wall.result.position).distance,
      rim_distance,
      kTolerance / 1024
  );
}

// The faces a particle carried around a rim in steps of T 1 s ends on, each counted once per run of equal faces, in
// order; and the end of the last step. Each step reads the normal n at the position, t = {n.z, 0, -n.x}, and moves with
// both velocities tol / 2 * t * tangent_sign - tol / 4 * n: along the surface, pressed into it.
struct CarriedAroundRim {
  std::vector<std::pair<SolidFace, int>> face_runs;
  Vec3 end{};  // m, world.
};

// Runs one step of a particle carried around the high rim of the fixed cylinder of CylinderRim from position, which
// must succeed with one hit, and returns the distance at the end of the step.
DistanceToSolid RunCarriedAroundRimStep(const Solid& cylinder, double tangent_sign, Vec3* position) {
  const Vec3 normal = ComputeDistanceToSolid(cylinder, cylinder.pose_at_step_start, *position).normal_out_of_solid;
  const Vec3 tangent = tangent_sign * Vec3{.x = normal.z, .y = 0, .z = -normal.x};
  const Vec3 velocity = (kTolerance / 2) * tangent - (kTolerance / 4) * normal;  // m/s
  const MoveParticleCall call =
      RunMoveWith("CylinderRim", {.position = *position, .velocity = velocity, .step_duration = 1});
  EXPECT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  *position = call.result.position;
  return ComputeDistanceToSolid(cylinder, cylinder.pose_at_step_start, *position);
}

// Runs 64 steps of a particle carried around the high rim of the fixed cylinder of CylinderRim; each must succeed with
// one hit and end between tol - tol / 1024 and 1.13 tol from the cylinder.
CarriedAroundRim RunCarriedAroundRim(Vec3 start, double tangent_sign) {
  const MoveParticleCase scene = FindMoveParticleCase("CylinderRim");
  const Solid& cylinder = scene.solids.at(0);
  CarriedAroundRim carried;
  Vec3 position = start;
  for (int step_index = 0; step_index < 64; ++step_index) {
    const DistanceToSolid at_end = RunCarriedAroundRimStep(cylinder, tangent_sign, &position);
    EXPECT_GE(at_end.distance, kTolerance - kTolerance / 1024) << step_index;
    EXPECT_LE(at_end.distance, 1.13 * kTolerance) << step_index;
    if (carried.face_runs.empty() || carried.face_runs.back().first != at_end.face) {
      carried.face_runs.emplace_back(at_end.face, 0);
    }
    ++carried.face_runs.back().second;
  }
  carried.end = position;
  return carried;
}

TEST(MoveParticle, ParticleCarriedAroundRim_OneHitPerStep) {
  // Pressed into the surface and moved along it in 64 steps, from the cap 2 tol inside the rim to the wall: the start's
  // response is the one hit of each step, the rim does not stop the particle, and it ends within 1.13 tol of the
  // cylinder; ends on the cap 4 times, beside the rim 3 times, on the wall 57 times, tol beyond the wall and 28.39 tol
  // below the cap's plane. Then from the wall to the cap, mirrored.
  const std::vector<std::pair<SolidFace, int>> cap_to_wall = {
      {SolidFace::kHighZ, 4},
      {SolidFace::kHighRim, 3},
      {SolidFace::kCurved, 57},
  };
  const CarriedAroundRim outward = RunCarriedAroundRim({.x = 0.5 - 2 * kTolerance, .y = 0, .z = 1 + kTolerance}, 1);
  EXPECT_EQ(outward.face_runs, cap_to_wall);
  EXPECT_NEAR(outward.end.x, 0.5 + kTolerance, kTolerance / 512);
  EXPECT_NEAR(outward.end.z, 1 - 28.39 * kTolerance, 0.01 * kTolerance);
  const std::vector<std::pair<SolidFace, int>> wall_to_cap = {
      {SolidFace::kCurved, 4},
      {SolidFace::kHighRim, 3},
      {SolidFace::kHighZ, 57},
  };
  const CarriedAroundRim inward = RunCarriedAroundRim({.x = 0.5 + kTolerance, .y = 0, .z = 1 - 2 * kTolerance}, -1);
  EXPECT_EQ(inward.face_runs, wall_to_cap);
  EXPECT_NEAR(inward.end.x, 0.5 - 28.39 * kTolerance, 0.01 * kTolerance);
  EXPECT_NEAR(inward.end.z, 1 + kTolerance, kTolerance / 512);
}

TEST(MoveParticle, LongCylinderTiltingUnderSlide_EndsWithinAQuarterToleranceBeforeTheSettle) {
  // The touching rule reads the surface's normal speed where the sweep starts; the particle slides 3.9 m along a
  // surface tilting within plan decision P17's bound, which sinks at one end and rises at the other, so before the
  // settle it ends 0.241 tol closer than the rest gap: within 2 |omega_across| max|arm| T < tol / 4, and above
  // tol - tol / 16 - tol / 4 (FindFirstHitOfSweep, step 3), so the end check's tol / 2 is never at stake. The settle
  // brings it back to tol.
  const MoveParticleCase scene = FindMoveParticleCase("LongCylinderTiltingUnderSlide");
  ASSERT_FALSE(ComputeIsTurningSolid(ToSolidContactInputs(scene), 0));
  const double before_settle = ComputeEndDistance(scene, ComputePositionBeforeSettle(scene));  // m
  EXPECT_NEAR(before_settle, 0.759 * kTolerance, 0.005 * kTolerance);
  EXPECT_GT(before_settle, kTolerance - kTolerance / 16 - kTolerance / 4);
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_NEAR(ComputeEndDistance(scene, call.result.position), kTolerance, kTolerance / 1024);
}

// The names of two scenes that differ only in the order of their solids or in an exact copy.
struct SettleScenes {
  std::string_view first;
  std::string_view second;
};

// Fails the calling test unless two scenes that differ only in the order of their solids, or in an exact copy, both
// succeed with the same hits and velocities, bit for bit, and end within four units in the last place of each other,
// apart on at least one axis: the known limit of the move off near surfaces (MoveToRestGapOfNearSurfaces, Order).
void ExpectSettleDiffersInLastPlace(SettleScenes scenes) {
  const MoveParticleCall call = RunMove(scenes.first);
  const MoveParticleCall other = RunMove(scenes.second);
  ASSERT_EQ(call.error, ContactError::kNone);
  ASSERT_EQ(other.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, other.result.hit_count);
  ExpectVec3Equal(other.result.velocity.for_this_step, call.result.velocity.for_this_step);
  ExpectVec3Equal(other.result.velocity.for_next_step, call.result.velocity.for_next_step);
  const Vec3 end = call.result.position;
  const Vec3 difference = other.result.position - end;  // m
  const double largest_coordinate = std::max({std::abs(end.x), std::abs(end.y), std::abs(end.z)});  // m
  const double largest_difference =
      std::max({std::abs(difference.x), std::abs(difference.y), std::abs(difference.z)});  // m
  EXPECT_GT(largest_difference, 0) << "the limit is gone: update MoveToRestGapOfNearSurfaces' Order";
  EXPECT_LE(largest_difference, 4 * 0x1p-52 * largest_coordinate);
}

TEST(MoveParticle, SettleWithExactCopy_LastPlaceKnownLimit) {
  // With the copy two solids are near at the end, so the settle iterates where one solid moved the particle along its
  // normal: the end differs by one unit in the last place in y.
  ExpectSettleDiffersInLastPlace(
      {.first = "SettleNearSpinningCylinder", .second = "SettleNearSpinningCylinderWithCopy"}
  );
}

TEST(MoveParticle, SettleInReversedOrder_LastPlaceKnownLimit) {
  // Reversed, the near surfaces reach the solver in another order, and the end differs in the last place.
  ExpectSettleDiffersInLastPlace(
      {.first = "SettleBesideMovingCylinder", .second = "SettleBesideMovingCylinderReversed"}
  );
}

TEST(ApplyHit, TurningCylinderRimHit_StaysWhereTheSearchStopped) {
  // The stepping search stops near the rim, 1.11 tol from it, in the rim region (plan decision P18): its hit is used
  // where the search stopped, as every hit is (plan decision P21). MoveParticle itself ends in kTooManyHits here until
  // the slide on a turning solid is built (plan Task 13), so the hit is applied alone.
  const MoveParticleCase scene = FindMoveParticleCase("TurningCylinderRimHit");
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  const Vec3 velocity = inputs.velocity.for_this_step;
  FirstHit first_hit;
  ASSERT_EQ(
      FindFirstHit(inputs, inputs.start_position, velocity, 0, inputs.step_duration, &first_hit),
      ContactError::kNone
  );
  ASSERT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 0.49999999958, 1e-10);
  ASSERT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  ASSERT_GT(first_hit.at_hit.distance, kTolerance + kTolerance / 16);
  ParticleMove move;
  move.result.position = inputs.start_position;
  move.result.velocity = inputs.velocity;
  ContactFailure failure;
  ASSERT_EQ(ApplyHit(inputs, first_hit, &move, &failure), ContactError::kNone);
  EXPECT_EQ(move.result.face, SolidFace::kHighRim);
  const Vec3 hit_position = move.result.reaction.hit_position;
  ExpectVec3Near(hit_position, inputs.start_position + velocity * first_hit.elapsed, kTolerance / 1024);
  const Solid& cylinder = scene.solids.at(0);
  const double hit_distance =
      ComputeDistanceToSolid(cylinder, ComputeSolidPose(cylinder, first_hit.elapsed), hit_position).distance;  // m
  EXPECT_NEAR(hit_distance, first_hit.at_hit.distance, kTolerance / 1024);
}

// --- Closing solids: carried out along the apex, or squeezed ---

// Runs the closing wedge scene of that name and fails the calling test unless it succeeded with one hit, the joint
// response over both faces, and the expected velocities within relative_tolerance of their largest axis.
SolidContactResult ExpectCarriedOut(std::string_view case_name, Vec3 expected_velocity, double relative_tolerance) {
  const MoveParticleCall call = RunMove(case_name);
  EXPECT_EQ(call.error, ContactError::kNone) << case_name;
  EXPECT_EQ(call.result.hit_count, 1) << case_name;
  const double scale =
      std::max({std::abs(expected_velocity.x), std::abs(expected_velocity.y), std::abs(expected_velocity.z)});
  ExpectBothVelocitiesNear(call.result, expected_velocity, relative_tolerance * scale);
  return call.result;
}

TEST(MoveParticle, FloorRisingUnderSixtyDegreeWall_CarriesParticleAlongApex) {
  // The velocity that approaches neither: {V / tan 60, 0, V}, not a squeeze. After 0.1 s the floor is 0.1 m higher and
  // the particle is at the rest gap of both at the end pose.
  const MoveParticleCase scene = FindMoveParticleCase("FloorRisingUnderSixtyDegreeWall");
  const SolidContactResult result =
      ExpectCarriedOut("FloorRisingUnderSixtyDegreeWall", {.x = 1 / std::numbers::sqrt3, .y = 0, .z = 1}, 1e-15);
  const Vec3 expected_end = {
      .x = std::numbers::sqrt3 * kTolerance + 0.1 / std::numbers::sqrt3,
      .y = 0,
      .z = 0.1 + kTolerance,
  };
  ExpectVec3Near(result.position, expected_end, kTolerance / 1024);
  ExpectAtRestGapOfEverySolid(ToSolidContactInputs(scene), result.position);
}

TEST(MoveParticle, BothFacesClosingAtSixtyDegrees_CarriesParticleOut) {
  ExpectCarriedOut("BothFacesClosingAtSixtyDegrees", {.x = std::numbers::sqrt3 / 2, .y = 0, .z = 0.5}, 1e-15);
}

TEST(MoveParticle, FloorRisingFastUnderSixtyDegreeWall_EscapeTooFastForStepIsCuredByShorterStep) {
  // The escape {500 / tan 60, 0, 500} m/s is within the speed cap, but its allowance, 2^-46 times the 1290 m/s of
  // speeds it involves, times the 1 s left is above tol / 64: kSqueezedBetweenSolids naming the rising floor, the
  // solid the particle approaches fastest. In a step of 0.25 s it is carried out, one hit; the solver's rounding at
  // 577 m/s leaves the velocity within 1e-15 of that speed of the closed form.
  ExpectCloseCall(RunMove("FloorRisingFastUnderSixtyDegreeWall"), ContactCloseCall::kSqueezedBetweenSolids, 1);
  MoveParticleCase shorter = FindMoveParticleCase("FloorRisingFastUnderSixtyDegreeWall");
  shorter.inputs.step_duration = 0.25;
  const MoveParticleCall call = RunMove(shorter);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  const double speed = 1000 / std::numbers::sqrt3;  // m/s
  ExpectBothVelocitiesNear(call.result, {.x = 500 / std::numbers::sqrt3, .y = 0, .z = 500}, 1e-15 * speed);
}

TEST(MoveParticle, FloorRisingThroughFixedFloor_PushChargedToTheFloorThatPushes) {
  // The start moves to the apex rest point; one joint response over the fixed floor, the rising floor and the side
  // carries the particle along the apex at {0.25 / tan 60, 0, 0.25}, one hit. The push along {0, 0, 1} is the rising
  // floor's, whose target is the larger: charged to it, it counts in no reaction. Charged to the first floor of the
  // direction, the fixed reacting one, it would fill a reaction although that floor never pushed the particle.
  const MoveParticleCall call = RunMove("FloorRisingThroughFixedFloorUnderSixtyDegreeWall");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {.x = 0.25 / std::numbers::sqrt3, .y = 0, .z = 0.25}, 1e-15);
  ExpectNoReaction(call.result.reaction);
}

TEST(MoveParticle, OneFaceClosingAtOneDegree_CarriesParticleOut) {
  // {1 / tan(1 degree), 0, 1} = {57.28996, 0, 1}: a speed of 1 / sin(1 degree) = 57.2987 m/s, below 64 times the
  // largest speed involved, the floor's 1 m/s.
  const double angle = std::numbers::pi / 180;  // rad
  ExpectCarriedOut("OneFaceClosingAtOneDegree", {.x = 1 / std::tan(angle), .y = 0, .z = 1}, 1e-9);
}

TEST(MoveParticle, OneFaceClosingAtPointEightDegrees_IsSqueezedBetweenSolids) {
  // The change would give 1 / sin(0.8 degrees) = 71.622 m/s, above 64 m/s. It names the rising floor, the solid the
  // particle approaches fastest. A step ten times shorter does not cure it. The limit is the closed form
  // asin(1 / 64) = 0.8953 degrees: 0.89 is squeezed (64.38 m/s), 0.90 carried out (63.6646 m/s), since the cap reads
  // the speeds before the response.
  ExpectCloseCall(RunMove("OneFaceClosingAtPointEightDegrees"), ContactCloseCall::kSqueezedBetweenSolids, 1);
  MoveParticleCase shorter = FindMoveParticleCase("OneFaceClosingAtPointEightDegrees");
  shorter.inputs.step_duration = 0.001;
  ExpectCloseCall(RunMove(shorter), ContactCloseCall::kSqueezedBetweenSolids, 1);
  const double degree = std::numbers::pi / 180;  // rad
  const MoveParticleCase below_limit =
      move_particle_cases::BuildClosingWedgeCase("OneFaceClosingAtPointEightNineDegrees", 0.89 * degree, 1, 0);
  ExpectCloseCall(RunMove(below_limit), ContactCloseCall::kSqueezedBetweenSolids, 1);
  const MoveParticleCase above_limit =
      move_particle_cases::BuildClosingWedgeCase("OneFaceClosingAtPointNineDegrees", 0.9 * degree, 1, 0);
  const MoveParticleCall call = RunMove(above_limit);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_NEAR(math::Length(call.result.velocity.for_this_step), 1 / std::sin(0.9 * degree), 1e-9);
}

TEST(MoveParticle, TwoFacesClosingSymmetrically_CapAtOnePointSevenNineDegrees) {
  // Each face at 1 m/s along its own normal: the escape is 1 / sin(a / 2), within the cap down to the closed form
  // 2 asin(1 / 64) = 1.7906 degrees, since the cap reads the speeds before the response. At 2 degrees, 57.2987 m/s, one
  // hit; at 1.6 degrees it would be 71.62 m/s, squeezed, naming the floor; at 1.79 degrees 64.02 m/s, squeezed; at
  // 1.80, 63.6646 m/s, carried out.
  const MoveParticleCall call = RunMove("TwoFacesClosingAtTwoDegrees");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_NEAR(math::Length(call.result.velocity.for_this_step), 1 / std::sin(std::numbers::pi / 180), 1e-9);
  ExpectCloseCall(RunMove("TwoFacesClosingAtOnePointSixDegrees"), ContactCloseCall::kSqueezedBetweenSolids, 1);
  const double degree = std::numbers::pi / 180;  // rad
  const MoveParticleCase below_limit =
      move_particle_cases::BuildClosingWedgeCase("TwoFacesClosingAtOnePointSevenNineDegrees", 1.79 * degree, 1, 1);
  ExpectCloseCall(RunMove(below_limit), ContactCloseCall::kSqueezedBetweenSolids, 1);
  const MoveParticleCase above_limit =
      move_particle_cases::BuildClosingWedgeCase("TwoFacesClosingAtOnePointEightDegrees", 1.8 * degree, 1, 1);
  const MoveParticleCall above_call = RunMove(above_limit);
  ASSERT_EQ(above_call.error, ContactError::kNone);
  EXPECT_NEAR(math::Length(above_call.result.velocity.for_this_step), 1 / std::sin(0.9 * degree), 1e-9);
}

TEST(MoveParticle, GrazingWedgeClosingFast_IsSqueezedAtEveryStepLength) {
  // The escape would be 1 / tan(1e-5) = 1e5 m/s: before the cap the model launched the particle 1 km in a step of
  // 0.01 s. Squeezed at 0.1, 0.01 and 0.001 s, naming the side, the solid the particle approaches fastest.
  for (const double step_duration : {0.1, 0.01, 0.001}) {
    MoveParticleCase scene = FindMoveParticleCase("GrazingWedgeClosingFast");
    scene.inputs.step_duration = step_duration;
    ExpectCloseCall(RunMove(scene), ContactCloseCall::kSqueezedBetweenSolids, 2);
  }
}

TEST(MoveParticle, SpinningFloorMovedSilently_RoundingIsNotAHit) {
  // The spinning floor is moved silently at the start: the particle's approach to it, 7.7e-15 m/s, is within the
  // rounding of the spin's omega x r, 2^-46 * |omega x r|_1. The upright responds, and the sweep that follows measures
  // its rate from the floor's full surface velocity, as the start rules and the joint response do. One hit, the
  // upright's. As built before, the start rules compared the floor's approach with 0 and the sweep measured from
  // linear_velocity alone: the floor was hit at 0 by rounding, two hits; either fix alone leaves two.
  const MoveParticleCall call = RunMove("SpinningFloorMovedSilently");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
}

TEST(MoveParticle, ClosingCeiling_IsSqueezedBetweenSolids) {
  // The ceiling reaches the particle at 0.05 - tol s and carries it down onto the floor, hit at 0.1 - 2 tol s, the
  // record of that hit; there the two close at 1 m/s and no velocity approaches neither. The particle sticks to the
  // ceiling, the solid it approaches fastest, which still approaches the floor: squeezed, naming the ceiling, id 2.
  ExpectCloseCall(RunMove("ClosingCeiling"), ContactCloseCall::kSqueezedBetweenSolids, 2);
}

// --- The slide: the first review's scenes ---

// Runs MoveParticle twice from the inputs: the step, then the next step from its end with both velocities its
// for_next_step. Fails the calling test unless both succeed and end at the rest gap of every solid.
void ExpectTwoStepsAtRestGap(SolidContactInputs inputs) {
  SolidContactResult result;
  ContactFailure failure;
  for (int step = 1; step <= 2; ++step) {
    const ContactError error = MoveParticle(inputs, &result, &failure);
    ASSERT_EQ(error, ContactError::kNone)
        << "step " << step << ", T " << inputs.step_duration << ", close call " << static_cast<int>(failure.close_call);
    ExpectAtRestGapOfEverySolid(inputs, result.position);
    inputs.start_position = result.position;
    inputs.velocity = {.for_next_step = result.velocity.for_next_step, .for_this_step = result.velocity.for_next_step};
  }
}

TEST(MoveParticle, SliverStopAtShorterSteps_EndsAndStartsAgainAtRestPoint) {
  // At the sliver's rest point the floor is at exactly tol: never near, and once neither touched nor active. The
  // side's response turned the particle into the floor at 2 sin(1e-7) * 2 m/s, which its touching rule let pass when
  // that times the step was under tol: at 0.004 and 0.003 s the end check failed, and at 0.002 s the particle ended
  // 0.57 tol from the floor and the next step was kStartsInNarrowGap. Now both are active surfaces of one joint
  // response, and the pair, a sliver, sticks: every step length gives two steps that end at the rest point.
  for (const double step_duration : {0.01, 0.006, 0.004, 0.003, 0.002, 0.001}) {
    const MoveParticleCase scene = FindMoveParticleCase("SliverStop");
    SolidContactInputs inputs = ToSolidContactInputs(scene);
    inputs.step_duration = step_duration;
    ExpectTwoStepsAtRestGap(inputs);
  }
}

TEST(MoveParticle, SlowSlideIntoSliverWedge_FiftyStepsNeverInNarrowGap) {
  // A particle sliding along the floor at 0.1 m/s into a wedge of 1e-8 rad, 50 steps of 0.01 s, each from the end of
  // the one before with its for_next_step: it reaches the side and stops, at least tol / 2 from both solids, and never
  // starts in a narrow gap, at rest at {205368709 tol, 0, tol} (the model). Before the floor was an active surface of
  // the side's hit, step 6 was kStartsInNarrowGap.
  const MoveParticleCase scene = FindMoveParticleCase("SlowSlideIntoSliverWedge");
  SolidContactInputs inputs = ToSolidContactInputs(scene);
  SolidContactResult result;
  ContactFailure failure;
  int failed_steps = 0;
  for (int step = 1; step <= 50; ++step) {
    const ContactError error = MoveParticle(inputs, &result, &failure);
    failed_steps += error == ContactError::kNone ? 0 : 1;
    inputs.start_position = result.position;
    inputs.velocity = {.for_next_step = result.velocity.for_next_step, .for_this_step = result.velocity.for_next_step};
  }
  EXPECT_EQ(failed_steps, 0);
  ExpectBothVelocitiesNear(result, {}, 1e-15);
  ExpectVec3Near(result.position, {.x = 205368709 * kTolerance, .y = 0, .z = kTolerance}, kTolerance);
}

// Runs the edge scene of that name with a friction and a velocity, and returns its result; fails the calling test
// unless it succeeded.
SolidContactResult RunEdgeScene(std::string_view case_name, double friction, Vec3 velocity) {
  MoveParticleCase scene = FindMoveParticleCase(case_name);
  scene.inputs.friction = friction;
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  const MoveParticleCall call = RunMove(scene);
  EXPECT_EQ(call.error, ContactError::kNone) << case_name;
  return call.result;
}

TEST(MoveParticle, DomainBoxEdgeWithFriction_SameAsFlatSolids) {
  // The particle meets the box's x and z sides at the same time: one joint response over both, as over the two flat
  // solids in the same place, one hit. Both fixed, so the friction limits add: with friction 0.5 and {-1, 0.5, -1},
  // pushes 1 and 1, a limit of 1 >= 0.5, {0, 0, 0}; friction 2 and {-1, 2, -1}, a stick too; friction 0.5 and
  // {-1, 2, -1}, 2 - 0.5 (1 + 1) = 1, {0, 1, 0} (the responses one side after the other gave 1.1645898). Before the
  // first review, the box's second side was removed without friction: v_y 0.276 and 0.211.
  struct EdgeDraw {
    double friction = 0;
    Vec3 velocity{};  // m/s
  };
  const std::array<EdgeDraw, 3> draws = {
      EdgeDraw{.friction = 0.5, .velocity = {.x = -1, .y = 0.5, .z = -1}},
      EdgeDraw{.friction = 2, .velocity = {.x = -1, .y = 2, .z = -1}},
      EdgeDraw{.friction = 0.5, .velocity = {.x = -1, .y = 2, .z = -1}},
  };
  for (const EdgeDraw draw : draws) {
    const SolidContactResult box = RunEdgeScene("DomainBoxEdgeWithFriction", draw.friction, draw.velocity);
    const SolidContactResult flats = RunEdgeScene("FlatSolidsEdgeWithFriction", draw.friction, draw.velocity);
    EXPECT_EQ(box.hit_count, 1);
    EXPECT_EQ(flats.hit_count, 1);
    ExpectVec3Near(box.velocity.for_next_step, flats.velocity.for_next_step, 1e-15);
    ExpectVec3Near(box.position, flats.position, kTolerance / 1024);
  }
  ExpectVec3Near(
      RunEdgeScene("DomainBoxEdgeWithFriction", 0.5, {.x = -1, .y = 0.5, .z = -1}).velocity.for_next_step,
      {},
      1e-15
  );
  ExpectVec3Near(
      RunEdgeScene("DomainBoxEdgeWithFriction", 2, {.x = -1, .y = 2, .z = -1}).velocity.for_next_step,
      {},
      1e-15
  );
  ExpectVec3Near(
      RunEdgeScene("DomainBoxEdgeWithFriction", 0.5, {.x = -1, .y = 2, .z = -1}).velocity.for_next_step,
      {.x = 0, .y = 1, .z = 0},
      1e-15
  );
}

// Runs the chamfered domain corner at one step length and velocity, the start 0.2 T {1, 1, 1}, and fails the calling
// test unless it succeeds at the rest gap of all five surfaces or farther. Returns the result.
SolidContactResult ExpectChamferedCornerSuccess(double step_duration, Vec3 velocity) {
  MoveParticleCase scene = FindMoveParticleCase("DomainCornerWithTwoChamfers");
  scene.inputs.step_duration = step_duration;
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  scene.inputs.start_position = 0.2 * step_duration * Vec3{.x = 1, .y = 1, .z = 1};
  const MoveParticleCall call = RunMove(scene);
  EXPECT_EQ(call.error, ContactError::kNone)
      << "T " << step_duration << ", close call " << static_cast<int>(call.failure.close_call);
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    EXPECT_GE(
        ComputeDistanceToSolid(solid, ComputeSolidPose(solid, step_duration), call.result.position).distance,
        kTolerance - kTolerance / 128
    );
  }
  return call.result;
}

TEST(MoveParticle, DomainCornerWithTwoChamfers_FiveDirectionsAtEveryStepLength) {
  // The chamfers, then the box's sides: five directions at once, which one joint response reads (at most three of them
  // meet the smallest change as equalities). Before, more than four was kTooManyHits at every step length. The
  // particle stops in the corner at the rest gap of all five surfaces, at steps of 1, 0.01 and 1e-4 s (the start
  // 0.2 T out), for the review's three velocities. The chamfer through the corner line is inside the corner's own cone:
  // the responses one surface after the other gave the particle speed along y from it ({0, 0.5, 0} or
  // {0, 0.15, 0} after four hits), the joint response does not.
  for (const double step_duration : {1.0, 0.01, 1e-4}) {
    for (const Vec3 velocity :
         {Vec3{.x = -1, .y = -1, .z = -1}, Vec3{.x = -1, .y = -2, .z = -0.5}, Vec3{.x = -0.3, .y = -1, .z = -0.2}}) {
      const SolidContactResult result = ExpectChamferedCornerSuccess(step_duration, velocity);
      ExpectBothVelocitiesNear(result, {}, 1e-15);
      // Straight into the corner every surface is met at once; the other two meet a chamfer first.
      const bool is_straight_in = velocity.x == velocity.y && velocity.y == velocity.z;
      EXPECT_EQ(result.hit_count, is_straight_in ? 1 : 3) << "T " << step_duration;
    }
  }
}

TEST(MoveParticle, DomainCornerWithTwoChamfersSearchDraw_ThreeHitsAndStops) {
  // Draw 44 of a search over paths into the chamfered corner (T 0.01 s, friction 0). The responses one surface after
  // the other made four hits here and left a fifth surface approached; 868 of 20,000 such paths were kTooManyHits at
  // every step length. One joint response per hit reads every surface the particle is on: three hits, and it stops.
  MoveParticleCase scene = FindMoveParticleCase("DomainCornerWithTwoChamfers");
  // NOLINTBEGIN(modernize-use-std-numbers): measured values, not constants.
  const Vec3 velocity = {.x = -1.0473316932328587, .y = -0.98622076700109773, .z = -0.93008650554402639};
  scene.inputs.start_position = {.x = 0.0047600466537494742, .y = 0.0067707440232294162, .z = 0.0079880132798848865};
  // NOLINTEND(modernize-use-std-numbers)
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  scene.inputs.step_duration = 0.01;
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone) << static_cast<int>(call.failure.close_call);
  EXPECT_EQ(call.result.hit_count, 3);
  ExpectBothVelocitiesNear(call.result, {}, 1e-15);
}

TEST(MoveParticle, FallingOnSeventeenCoincidentFloors_SlidesOnOneDirection) {
  // Seventeen fixed floors through the origin, the particle falling on them at {1, 0, -1} m/s from z = 0.05 for 0.1 s:
  // the first is hit, and the joint response reads seventeen surfaces of one normal, one velocity and one friction:
  // exact copies, which take one place in the active surfaces (ComputeIsExactCopy). Listed one by one they were more
  // than sixteen: kTooManyHits.
  MoveParticleCase scene = FindMoveParticleCase("SeventeenCoincidentFloors");
  scene.inputs.start_position = {.x = 0, .y = 0, .z = 0.05};
  scene.inputs.velocity = {.for_next_step = {.x = 1, .y = 0, .z = -1}, .for_this_step = {.x = 1, .y = 0, .z = -1}};
  scene.inputs.step_duration = 0.1;
  const MoveParticleCall call = RunMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone) << static_cast<int>(call.failure.close_call);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {.x = 1, .y = 0, .z = 0}, 1e-15);
}

TEST(MoveParticle, MovingCornerWithCoincidentFaceFastEscape_AllowanceRaisedForThisStep) {
  // The review's draw 3053: the escape is much faster than the velocity the response received, so the rounding it may
  // leave is larger than that velocity's own allowance; the response's allowance counts the change too, which keeps
  // the sweeps from hitting a face again by rounding. Three hits, and the velocity within 8.6e-14 of the speed of the
  // sequence of responses as built before (plan decision P20 changes it by rounding only here).
  const MoveParticleCall call = RunMove("MovingCornerWithCoincidentFaceFastEscape");
  ASSERT_EQ(call.error, ContactError::kNone) << static_cast<int>(call.failure.close_call);
  EXPECT_EQ(call.result.hit_count, 3);
}

TEST(MoveParticle, MovingCornerWithCoincidentFaceNextStepEscape_Succeeds) {
  // The review's draw 115326, which separated for_next_step's allowance from for_this_step's: the coincident fourth
  // face took the rounding a change left in for_next_step for an approach. The start's faces are active at once: one
  // joint response, one hit.
  const MoveParticleCall call = RunMove("MovingCornerWithCoincidentFaceNextStepEscape");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
}

TEST(MoveParticle, ObtuseWedgeStartInBothOrders_Same) {
  // In a wedge of 120 degrees, friction 0, {-1, 1, -1}: the responses one solid after the other gave {-0.25, 1, 0.433}
  // with the floor first and {0.183, 1, 0} with the side first. One joint response gives the smallest frictionless
  // change over both, {0, 1, 0}, one hit, the same bits in both orders.
  const MoveParticleCall floor_first = RunMove("ObtuseWedgeFloorFirst");
  const MoveParticleCall side_first = RunMove("ObtuseWedgeSideFirst");
  ASSERT_EQ(floor_first.error, ContactError::kNone);
  ASSERT_EQ(side_first.error, ContactError::kNone);
  EXPECT_EQ(floor_first.result.hit_count, 1);
  EXPECT_EQ(side_first.result.hit_count, 1);
  ExpectBothVelocitiesNear(floor_first.result, {.x = 0, .y = 1, .z = 0}, 1e-15);
  ExpectVec3Equal(side_first.result.velocity.for_next_step, floor_first.result.velocity.for_next_step);
  ExpectVec3Equal(side_first.result.velocity.for_this_step, floor_first.result.velocity.for_this_step);
}

TEST(MoveParticle, CoincidentReactingFloorsNormalsApartByRounding_NoSecondReactingSolid) {
  // Two reacting floors whose normals are 2e-13 apart share one direction, so one response pushes once: no false
  // kSecondReactingSolid at any step length. The push is charged to the direction's pushing surface: equal friction,
  // both reacting, so the first in the sorted order, the floor whose normal's x is the smaller, solid_index 1.
  for (const double step_duration : {0.2, 0.001}) {
    MoveParticleCase scene = FindMoveParticleCase("CoincidentReactingFloorsNormalsApartByRounding");
    scene.inputs.step_duration = step_duration;
    const MoveParticleCall call = RunMove(scene);
    ASSERT_EQ(call.error, ContactError::kNone) << static_cast<int>(call.failure.close_call);
    EXPECT_EQ(call.result.hit_count, 1);
    ExpectBothVelocitiesNear(call.result, {.x = 1, .y = 0, .z = 0}, 1e-15);
    EXPECT_EQ(call.result.reaction.solid_index, 1U);
    ExpectVec3Near(call.result.reaction.velocity_for_next_step_change, {.x = 0, .y = 0, .z = 1}, 1e-15);
  }
}

// Runs the rest-point wedge of that angle name listed floor first and side first, and fails the calling test unless
// both succeed with one hit and the same velocities, bit for bit. Returns the floor-first result.
SolidContactResult ExpectSameInBothOrders(std::string_view angle_name) {
  const std::string floor_first = std::string(angle_name) + "DegreeRestPointFloorFirst";
  const std::string side_first = std::string(angle_name) + "DegreeRestPointSideFirst";
  const MoveParticleCall floor_call = RunMove(floor_first);
  const MoveParticleCall side_call = RunMove(side_first);
  EXPECT_EQ(floor_call.error, ContactError::kNone) << floor_first;
  EXPECT_EQ(side_call.error, ContactError::kNone) << side_first;
  EXPECT_EQ(floor_call.result.hit_count, 1) << floor_first;
  EXPECT_EQ(side_call.result.hit_count, 1) << side_first;
  ExpectVec3Equal(side_call.result.velocity.for_next_step, floor_call.result.velocity.for_next_step);
  ExpectVec3Equal(side_call.result.velocity.for_this_step, floor_call.result.velocity.for_this_step);
  return floor_call.result;
}

TEST(MoveParticle, RestPointWedgesWithFriction_SameInBothOrders) {
  // At the rest point both faces are active: one joint response, friction 0.5. Both surfaces are fixed, so their
  // friction limits add (plan decision P20, item 5): the velocity along the apex is 1 - 0.5 (push_floor + push_side),
  // or 0, a stick. 30 degrees, {-1, 1, -1}: pushes 2.732 and 2, a limit of 2.366 >= 1, {0, 0, 0} exactly. 45, 60 and
  // 75 degrees, {-0.2, 1, -1}: y 0.258578643763, 0.326794919243 and 0.369677462716. The same bits in both orders.
  const SolidContactResult thirty = ExpectSameInBothOrders("Thirty");
  ExpectBothVelocitiesNear(thirty, {}, 0);
  const std::array<std::pair<std::string_view, double>, 3> slips = {
      std::pair<std::string_view, double>{"FortyFive", 0.258578643763},
      std::pair<std::string_view, double>{"Sixty", 0.326794919243},
      std::pair<std::string_view, double>{"SeventyFive", 0.369677462716},
  };
  for (const auto& [angle_name, along_apex] : slips) {
    const SolidContactResult result = ExpectSameInBothOrders(angle_name);
    EXPECT_NEAR(result.velocity.for_next_step.y, along_apex, 1e-12) << angle_name;
    EXPECT_NEAR(result.velocity.for_next_step.x, 0, 1e-15) << angle_name;
    EXPECT_NEAR(result.velocity.for_next_step.z, 0, 1e-15) << angle_name;
  }
}

TEST(MoveParticle, MicroradianWedgeOnReactingFloor_FloorApproachedFastestIsCharged) {
  // A wedge of 1e-6 rad, a sliver (1 + c = 5e-13 < 2^-40), at its rest point: no change is found, and the particle
  // sticks to the surface it approaches fastest. Moving at {-2, 1, 0} that is the side: {0, 0, 0}, no reaction. Moving
  // at {-2, 1, -0.001} it is the reacting floor, charged the whole stop {2, -1, 0.001}.
  const MoveParticleCall into_side = RunMove("MicroradianWedgeOnReactingFloor");
  ASSERT_EQ(into_side.error, ContactError::kNone);
  EXPECT_EQ(into_side.result.hit_count, 1);
  ExpectBothVelocitiesNear(into_side.result, {}, 0);
  ExpectNoReaction(into_side.result.reaction);
  MoveParticleCase scene = FindMoveParticleCase("MicroradianWedgeOnReactingFloor");
  const Vec3 into_floor_velocity = {.x = -2, .y = 1, .z = -0.001};
  scene.inputs.velocity = {.for_next_step = into_floor_velocity, .for_this_step = into_floor_velocity};
  const MoveParticleCall into_floor = RunMove(scene);
  ASSERT_EQ(into_floor.error, ContactError::kNone);
  EXPECT_EQ(into_floor.result.hit_count, 1);
  ExpectBothVelocitiesNear(into_floor.result, {}, 0);
  EXPECT_EQ(into_floor.result.reaction.solid_index, 0U);
  ExpectVec3Equal(into_floor.result.reaction.velocity_for_next_step_change, {.x = 2, .y = -1, .z = 0.001});
}

TEST(MoveParticle, CopyFuzzScenes_ExactCopyChangesNothing) {
  // Two draws of the copy fuzz where an exact copy of a face comes after it in the array. The copy is dropped from the
  // active surfaces (ComputeIsExactCopy), so each scene with its copy gives the hit count and the velocities of the
  // scene without it, bit for bit. Three faces (draw 131): one hit, {0, 0, 0}, a stick in the corner. Two faces (draw
  // 51): as before.
  for (const std::string_view case_name : {"CopyFuzzThreeFaces", "CopyFuzzTwoFaces"}) {
    const MoveParticleCall without_copy = RunMove(case_name);
    const MoveParticleCall with_copy = RunMove(std::string(case_name) + "WithCopy");
    ASSERT_EQ(without_copy.error, ContactError::kNone) << case_name;
    ASSERT_EQ(with_copy.error, ContactError::kNone) << case_name;
    EXPECT_EQ(with_copy.result.hit_count, without_copy.result.hit_count) << case_name;
    ExpectVec3Equal(with_copy.result.velocity.for_next_step, without_copy.result.velocity.for_next_step);
    ExpectVec3Equal(with_copy.result.velocity.for_this_step, without_copy.result.velocity.for_this_step);
  }
  const MoveParticleCall three_faces = RunMove("CopyFuzzThreeFaces");
  EXPECT_EQ(three_faces.result.hit_count, 1);
  ExpectBothVelocitiesNear(three_faces.result, {}, 0);
}

TEST(MoveParticle, DomainBoxAndSideWithGround_SameAsWithoutGround) {
  // A ground lying on the domain box's bottom, fixed, with the bottom's normal bit for bit: an exact copy when its
  // friction is the same, dropped; with twice the friction it is the bottom's direction's pushing surface, and its
  // friction acts. At the rest point of the box's bottom and a side at 60 degrees, {-1, 1, -1}, friction 0.5: the
  // pushes give a friction limit above the speed along the apex, so all three scenes stick, {0, 0, 0}, one hit. The
  // result reads the active surfaces, not which face rounds below tol, so the libm of std::tan no longer changes it.
  for (const std::string_view case_name :
       {"DomainBoxAndSide", "DomainBoxAndSideWithGround", "DomainBoxAndSideWithRoughGround"}) {
    const MoveParticleCall call = RunMove(case_name);
    ASSERT_EQ(call.error, ContactError::kNone) << case_name;
    EXPECT_EQ(call.result.hit_count, 1) << case_name;
    ExpectBothVelocitiesNear(call.result, {}, 0);
  }
}

// --- The joint response (plan decision P20) ---

// Builds a scene from its solids, in that array order, with the solid_ids renumbered 1, 2, ... so they ascend.
MoveParticleCase BuildOrderedScene(std::vector<Solid> solids, const SolidContactInputs& inputs) {
  for (std::size_t solid_index = 0; solid_index < solids.size(); ++solid_index) {
    solids[solid_index].solid_id = static_cast<std::uint32_t>(solid_index + 1);
  }
  return {.name = "", .solids = std::move(solids), .inputs = inputs};
}

// The normal of BuildWedgeSide for a wedge of that angle with the floor, the same bits.
Vec3 ToWedgeSideNormal(double wedge_angle) {
  const double normal_angle = std::numbers::pi - wedge_angle;  // rad
  return {.x = std::sin(normal_angle), .y = 0, .z = std::cos(normal_angle)};
}

// The inputs of a particle at the rest point {tol / tan(a / 2), 0, tol} of a wedge of angle a with the floor, both
// velocities velocity, T 0.01 s, overlap 2 tol, with that friction.
SolidContactInputs BuildRestPointInputs(double wedge_angle, Vec3 velocity, double friction) {
  SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(
      {.x = kTolerance / std::tan(wedge_angle / 2), .y = 0, .z = kTolerance},
      velocity,
      0.01
  );
  inputs.friction = friction;
  return inputs;
}

// A flat solid through the origin with that normal, moving at that velocity, friction scale and id.
//
// clang-tidy: the normal and the velocity are both Vec3; the swappable-parameters check flags the two.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Solid BuildMovingFlatSolid(Vec3 normal_out_of_solid, Vec3 linear_velocity, double friction_scale, std::uint32_t id) {
  Solid solid = move_particle_cases::BuildFlatSolid({}, normal_out_of_solid, {.linear_velocity = linear_velocity}, id);
  solid.friction_scale = friction_scale;
  return solid;
}

// Fails the calling test unless both calls have the same error, kind, hit count and both velocities, bit for bit.
void ExpectSameBits(const MoveParticleCall& actual, const MoveParticleCall& expected, std::string_view label) {
  EXPECT_EQ(actual.error, expected.error) << label;
  EXPECT_EQ(actual.result.kind, expected.result.kind) << label;
  EXPECT_EQ(actual.result.hit_count, expected.result.hit_count) << label;
  ExpectVec3Equal(actual.result.velocity.for_next_step, expected.result.velocity.for_next_step);
  ExpectVec3Equal(actual.result.velocity.for_this_step, expected.result.velocity.for_this_step);
}

// The solids in the given order of their indexes.
std::vector<Solid> ToPermutedSolids(const std::vector<Solid>& solids, const std::vector<std::size_t>& order) {
  std::vector<Solid> permuted;
  permuted.reserve(order.size());
  for (const std::size_t index : order) {
    permuted.push_back(solids[index]);
  }
  return permuted;
}

// Runs the solids in every order of the array and fails the calling test unless every order gives the bits of the
// first. Returns the call of the order given.
MoveParticleCall ExpectSameBitsInEveryOrder(const std::vector<Solid>& solids, const SolidContactInputs& inputs) {
  const MoveParticleCall first = RunMove(BuildOrderedScene(solids, inputs));
  std::vector<std::size_t> order(solids.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  while (std::ranges::next_permutation(order).found) {
    ExpectSameBits(RunMove(BuildOrderedScene(ToPermutedSolids(solids, order), inputs)), first, "permuted");
  }
  return first;
}

// The scene of C1 (re-review 2, q1): floor_count coincident fixed floors and a wall at x = 0.5, the particle falling
// from {0.3, 0, 0.01} at {10, 0, -2} m/s for 0.1 s, friction 0.
MoveParticleCase BuildFloorsAndWallScene(int floor_count) {
  std::vector<Solid> solids(static_cast<std::size_t>(floor_count), move_particle_cases::BuildFloor(1));
  solids.push_back(move_particle_cases::BuildFlatSolid({.x = 0.5, .y = 0, .z = 0}, -kUnitX, kNoMotion, 1));
  return BuildOrderedScene(
      solids,
      move_particle_cases::BuildSeveralSurfaceInputs({.x = 0.3, .y = 0, .z = 0.01}, {.x = 10, .y = 0, .z = -2}, 0.1)
  );
}

TEST(MoveParticle, EightCoincidentFloorsAndWall_TwoHits) {
  // Re-review 2's C1: eight coincident floors, then a wall. The floors are exact copies, one active surface; nothing
  // is listed per solid, so nothing overflows. The fall is one hit at 0.05 - tol / 0.2 of the step, the wall a second:
  // {0, 0, 0} at {0.5 - tol, 0, tol}. The same with 17 and 40 floors (built once under AddressSanitizer for round 3).
  for (const int floor_count : {8, 17, 40}) {
    const MoveParticleCall call = RunMove(BuildFloorsAndWallScene(floor_count));
    ASSERT_EQ(call.error, ContactError::kNone) << floor_count;
    EXPECT_EQ(call.result.hit_count, 2) << floor_count;
    ExpectBothVelocitiesNear(call.result, {}, 0);
    ExpectVec3Near(call.result.position, {.x = 0.5 - kTolerance, .y = 0, .z = kTolerance}, kAtRestGapTolerance);
    EXPECT_NEAR(call.result.hit_fraction, 0.05 - kTolerance / 0.2, 1e-15) << floor_count;
  }
}

// The solids of I-1 (re-review 2, q3): a floor moving at {0, 0.3, 0} and a fixed side at that wedge angle, and a third
// solid when given.
std::vector<Solid> BuildMovingFloorWedge(double wedge_angle, const Solid* third) {
  std::vector<Solid> solids = {
      BuildMovingFlatSolid(kUnitZ, {.x = 0, .y = 0.3, .z = 0}, 1, 1),
      move_particle_cases::BuildWedgeSide(std::numbers::pi - wedge_angle, 2),
  };
  if (third != nullptr) {
    solids.push_back(*third);
  }
  return solids;
}

TEST(MoveParticle, CoincidentFloorOneRoundingUnitFaster_SameBitsAsWithoutIt) {
  // Re-review 2's I-1: beside a floor moving at {0, 0.3, 0}, a coincident floor at {0, 0.1 + 0.2, 0}, one rounding
  // unit faster, pushes as much; equal friction, so the first in the sorted order acts, the slower floor: the same bits
  // as without it, and as with an exact copy. Friction 0.5: y 0.15 at 30 and 60 degrees. Friction 0.1: y
  // 1 - 0.1 (2.7320508 + 2) = 0.526794919243 at 30 degrees and 0.726794919243 at 60.
  const double degree = std::numbers::pi / 180;  // rad
  const Solid faster_floor = BuildMovingFlatSolid(kUnitZ, {.x = 0, .y = 0.1 + 0.2, .z = 0}, 1, 3);
  const Solid copy_floor = BuildMovingFlatSolid(kUnitZ, {.x = 0, .y = 0.3, .z = 0}, 1, 3);
  struct Expected {
    double wedge_angle = 0;  // rad
    double friction = 0;
    double along_apex = 0;  // m/s
  };
  const std::array<Expected, 4> rows = {
      Expected{.wedge_angle = 30 * degree, .friction = 0.5, .along_apex = 0.15},
      Expected{.wedge_angle = 60 * degree, .friction = 0.5, .along_apex = 0.15},
      Expected{.wedge_angle = 30 * degree, .friction = 0.1, .along_apex = 0.526794919243},
      Expected{.wedge_angle = 60 * degree, .friction = 0.1, .along_apex = 0.726794919243},
  };
  for (const Expected& row : rows) {
    const SolidContactInputs inputs = BuildRestPointInputs(row.wedge_angle, {.x = -1, .y = 1, .z = -1}, row.friction);
    const MoveParticleCall alone = RunMove(BuildOrderedScene(BuildMovingFloorWedge(row.wedge_angle, nullptr), inputs));
    ASSERT_EQ(alone.error, ContactError::kNone);
    EXPECT_EQ(alone.result.hit_count, 1);
    EXPECT_NEAR(alone.result.velocity.for_next_step.y, row.along_apex, 1e-12);
    for (const Solid* third : {&faster_floor, &copy_floor}) {
      ExpectSameBits(RunMove(BuildOrderedScene(BuildMovingFloorWedge(row.wedge_angle, third), inputs)), alone, "third");
    }
  }
}

// Runs a floor moving at {0, 0.3, 0} and a 60 degree side moving at {0, side_speed, 0} from their rest point at
// {-1, 1, -1} with that friction, and fails the calling test unless one hit leaves {0, along_apex, 0} within
// tolerance on y and 1e-15 on x and z.
//
// clang-tidy: the friction, the speed and the expected speed are doubles; the swappable-parameters check flags them.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ExpectTwoMovingFacesAlongApex(double friction, double side_speed, double along_apex, double tolerance) {
  const double wedge_angle = std::numbers::pi / 3;  // rad
  const SolidContactInputs inputs = BuildRestPointInputs(wedge_angle, {.x = -1, .y = 1, .z = -1}, friction);
  const std::vector<Solid> solids = {
      BuildMovingFlatSolid(kUnitZ, {.x = 0, .y = 0.3, .z = 0}, 1, 1),
      BuildMovingFlatSolid(ToWedgeSideNormal(wedge_angle), {.x = 0, .y = side_speed, .z = 0}, 1, 2),
  };
  const MoveParticleCall call = RunMove(BuildOrderedScene(solids, inputs));
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_NEAR(call.result.velocity.for_next_step.y, along_apex, tolerance);
  EXPECT_NEAR(call.result.velocity.for_next_step.x, 0, 1e-15);
  EXPECT_NEAR(call.result.velocity.for_next_step.z, 0, 1e-15);
}

TEST(MoveParticle, TwoMovingFacesOneRoundingUnitApart_SameAsOneVelocity) {
  // A floor and a 60 degree side both moving at {0, 0.3, 0}: one solid velocity, the friction limits add. With the
  // side at {0, 0.1 + 0.2, 0} they move differently by one rounding unit, and the mean over the orders agrees to
  // rounding. Friction 0.5: {0, 0.3, 0}, a stick; friction 0.1: y 0.726794919243. Adding each surface's friction from
  // the same velocity instead gives y -0.4 in the second scene: friction reversing the sliding.
  for (const double side_speed : {0.3, 0.1 + 0.2}) {
    ExpectTwoMovingFacesAlongApex(0.5, side_speed, 0.3, 1e-15);
    ExpectTwoMovingFacesAlongApex(0.1, side_speed, 0.726794919243, 1e-12);
  }
}

// Runs the floors at z = 0 and z = tol / 2 from z = tol / 4 at that velocity, the lower one reacting or not and listed
// first or second, and fails the calling test unless the start inside the upper one is one hit, kind
// kStartedInsideSolid, both velocities {0, 0, 0}, and no reaction.
void ExpectUpperFloorResponds(Vec3 velocity, bool is_lower_reacting, bool is_lower_first) {
  Solid lower_floor = move_particle_cases::BuildFloor(1);
  lower_floor.reports_reaction = is_lower_reacting;
  const Solid upper_floor =
      move_particle_cases::BuildFlatSolid({.x = 0, .y = 0, .z = kTolerance / 2}, kUnitZ, kNoMotion, 1);
  const std::vector<Solid> solids =
      is_lower_first ? std::vector<Solid>{lower_floor, upper_floor} : std::vector<Solid>{upper_floor, lower_floor};
  const SolidContactInputs inputs =
      move_particle_cases::BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = kTolerance / 4}, velocity, 0.01);
  const MoveParticleCall call = RunMove(BuildOrderedScene(solids, inputs));
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectBothVelocitiesNear(call.result, {}, 0);
  ExpectNoReaction(call.result.reaction);
}

TEST(MoveParticle, ParallelFloorsStartInsideUpper_SameInBothOrders) {
  // Re-review 2's m-f: floors at z = 0 and z = tol / 2, the start at tol / 4, inside the upper one. The move takes it
  // to 1.5 tol, where the lower floor is not active: the upper one responds, in either order, one hit, kind
  // kStartedInsideSolid. With the lower floor reacting, listed first or second, no reaction: it never pushed.
  for (const bool is_lower_reacting : {false, true}) {
    for (const bool is_lower_first : {true, false}) {
      ExpectUpperFloorResponds({.x = 0, .y = 0, .z = -1}, is_lower_reacting, is_lower_first);
      ExpectUpperFloorResponds({}, is_lower_reacting, is_lower_first);
    }
  }
}

TEST(MoveParticle, MovingFourFaceCorner_SucceedsAtEveryStepLength) {
  // Re-review 2's m-g, draw 102164 of q4b: four flat solids through the origin, two of them moving, friction 2.92.
  // The responses one surface after the other gave kTooManyHits at every step length; one joint response per hit
  // gives two hits and the same velocity at 0.01, 1e-4 and 1e-6 s.
  // NOLINTBEGIN(modernize-use-std-numbers): measured values, not constants.
  const std::vector<Solid> solids = {
      BuildMovingFlatSolid(
          {.x = -0.67950032971940066, .y = 0.15463576612760666, .z = 0.71719389410769119},
          {.x = -0.035902373724512791, .y = 0.16310544388377485, .z = 0.26346007690904943},
          1,
          1
      ),
      BuildMovingFlatSolid({.x = -0.71548192049742465, .y = 0.068979324215572471, .z = 0.69521757333375855}, {}, 1, 2),
      BuildMovingFlatSolid(
          {.x = -0.0055145910467134171, .y = 0.58262279289502583, .z = -0.81272398173352034},
          {},
          1,
          3
      ),
      BuildMovingFlatSolid(
          {.x = 0.6295852542262288, .y = 0.63222080802298863, .z = 0.45157419939989274},
          {.x = 0.25357877268218021, .y = -0.17551066158987808, .z = -0.013789205914272273},
          1,
          4
      ),
  };
  SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(
      {.x = -5.9984278331639708e-10, .y = 2.0052907849860375e-09, .z = 1.2189571010398511e-10},
      {.x = 0.37392763599320999, .y = 0.13654011589841342, .z = -0.24647174345867018},
      0.01
  );
  inputs.friction = 2.923388931079602;
  // NOLINTEND(modernize-use-std-numbers)
  for (const double step_duration : {0.01, 1e-4, 1e-6}) {
    inputs.step_duration = step_duration;
    const MoveParticleCall call = RunMove(BuildOrderedScene(solids, inputs));
    ASSERT_EQ(call.error, ContactError::kNone) << step_duration;
    EXPECT_EQ(call.result.hit_count, 2) << step_duration;
    ExpectBothVelocitiesNear(call.result, {.x = -0.152699427, .y = 0.200466685, .z = 0.144745987}, 1e-8);
  }
}

// Runs the scene with a copy of solids[copied_index] inserted at every place of the array, and fails the calling test
// unless each gives the hit count and the velocities of the scene without it, bit for bit.
void ExpectCopyInEveryPlaceChangesNothing(const MoveParticleCase& scene, std::size_t copied_index) {
  const MoveParticleCall without_copy = RunMove(BuildOrderedScene(scene.solids, scene.inputs));
  ASSERT_EQ(without_copy.error, ContactError::kNone);
  for (std::size_t place = 0; place <= scene.solids.size(); ++place) {
    std::vector<Solid> solids = scene.solids;
    solids.insert(solids.begin() + static_cast<std::ptrdiff_t>(place), scene.solids[copied_index]);
    ExpectSameBits(RunMove(BuildOrderedScene(solids, scene.inputs)), without_copy, "copy");
  }
}

TEST(MoveParticle, CopiesInEveryPlaceOfTheArray_SameBits) {
  // Re-review 2's m-h: an exact copy listed before the surface it copies took a response of its own. Now a copy is
  // dropped wherever it stands: the two copy fuzz scenes, the moving floor of CoincidentFloorOneRoundingUnitFaster and
  // draw 15798 of the review's r2x, the copy at every index, give the bits of the scene without it.
  ExpectCopyInEveryPlaceChangesNothing(FindMoveParticleCase("CopyFuzzThreeFaces"), 0);
  ExpectCopyInEveryPlaceChangesNothing(FindMoveParticleCase("CopyFuzzTwoFaces"), 1);
  const double wedge_angle = std::numbers::pi / 3;  // rad
  const MoveParticleCase moving_floor = BuildOrderedScene(
      BuildMovingFloorWedge(wedge_angle, nullptr),
      BuildRestPointInputs(wedge_angle, {.x = -1, .y = 1, .z = -1}, 0.5)
  );
  ExpectCopyInEveryPlaceChangesNothing(moving_floor, 0);
  // Draw 15798 of r2x: three fixed faces at their rest point, friction 0; with a copy of the third listed first, the
  // responses one surface after the other gave a velocity 0.45 of the speed away from the scene without it.
  // NOLINTBEGIN(modernize-use-std-numbers): measured values, not constants.
  const MoveParticleCase three_faces = BuildOrderedScene(
      {
          BuildMovingFlatSolid(
              {.x = 0.60795077209177761, .y = -0.3057230405462848, .z = 0.73275458455894105},
              {},
              1,
              1
          ),
          BuildMovingFlatSolid(
              {.x = 0.68403749630042443, .y = -0.71878160902002475, .z = 0.12428073941536956},
              {},
              1,
              2
          ),
          BuildMovingFlatSolid(
              {.x = 0.010885836871239393, .y = -0.72153534111472639, .z = 0.69229202658854017},
              {},
              1,
              3
          ),
      },
      move_particle_cases::BuildSeveralSurfaceInputs(
          {.x = 7.4131727375170571e-10, .y = -5.33687953892031e-10, .z = 7.7738580710700797e-10},
          {.x = -0.18896864807755898, .y = 0.46035055733376096, .z = -0.6954726815656882},
          0.01
      )
  );
  // NOLINTEND(modernize-use-std-numbers)
  ExpectCopyInEveryPlaceChangesNothing(three_faces, 2);
}

TEST(MoveParticle, RoughGroundOnWedgeFloor_RougherOneActs) {
  // At the rest point of a 60 degree wedge, {-0.2, 1, -1}, friction 0.2: floor and side give y
  // 1 - 0.2 (1.1154700538 + 0.2309401077) = 0.730717967697. A floor of friction scale 2 in the floor's place:
  // 0.507623956930. The floor, the side and that rough floor, in each of the six orders, and with two more plain
  // floors: the rough one pushes for the floor's direction, 0.507623956930, the same bits. With the plain floor
  // reacting and no rough one, it is charged its push and its friction share: {0, -0.223094010768, 1.115470053838}.
  const double wedge_angle = std::numbers::pi / 3;  // rad
  const SolidContactInputs inputs = BuildRestPointInputs(wedge_angle, {.x = -0.2, .y = 1, .z = -1}, 0.2);
  const Solid floor = move_particle_cases::BuildFloor(1);
  const Solid side = move_particle_cases::BuildWedgeSide(std::numbers::pi - wedge_angle, 2);
  Solid rough_floor = floor;
  rough_floor.friction_scale = 2;
  const MoveParticleCall plain = RunMove(BuildOrderedScene({floor, side}, inputs));
  ASSERT_EQ(plain.error, ContactError::kNone);
  EXPECT_NEAR(plain.result.velocity.for_next_step.y, 0.730717967697, 1e-12);
  const MoveParticleCall rough = RunMove(BuildOrderedScene({rough_floor, side}, inputs));
  ASSERT_EQ(rough.error, ContactError::kNone);
  EXPECT_NEAR(rough.result.velocity.for_next_step.y, 0.507623956930, 1e-12);
  const MoveParticleCall all_three = ExpectSameBitsInEveryOrder({floor, side, rough_floor}, inputs);
  ExpectSameBits(all_three, rough, "three");
  ExpectSameBits(RunMove(BuildOrderedScene({floor, side, rough_floor, floor, floor}, inputs)), rough, "five");
  Solid reacting_floor = floor;
  reacting_floor.reports_reaction = true;
  const MoveParticleCall reacting = RunMove(BuildOrderedScene({reacting_floor, side}, inputs));
  ASSERT_EQ(reacting.error, ContactError::kNone);
  EXPECT_EQ(reacting.result.reaction.solid_index, 0U);
  ExpectVec3Near(
      reacting.result.reaction.velocity_for_next_step_change,
      {.x = 0, .y = -0.223094010768, .z = 1.115470053838},
      1e-12
  );
}

TEST(MoveParticle, ConveyorOnFloor_FirstInSortedOrderActs) {
  // A pinned choice (plan decision P20, item 4): a floor and a coincident conveyor moving at {1, 0, 0}, the particle
  // falling at 1 m/s onto both, friction 0.5. Both push as much; with equal friction the first in the sorted order
  // acts, the fixed floor, whose x velocity is the smaller: {0, 0, 0} in both orders. The conveyor with friction
  // scale 2 acts: {1, 0, 0}; the fixed floor with scale 2: {0, 0, 0}.
  SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(
      {.x = 0, .y = 0, .z = kTolerance},
      {.x = 0, .y = 0, .z = -1},
      0.01
  );
  inputs.friction = 0.5;
  const Solid floor = move_particle_cases::BuildFloor(1);
  const Solid conveyor = BuildMovingFlatSolid(kUnitZ, kUnitX, 1, 2);
  ExpectBothVelocitiesNear(ExpectSameBitsInEveryOrder({floor, conveyor}, inputs).result, {}, 0);
  const Solid rough_conveyor = BuildMovingFlatSolid(kUnitZ, kUnitX, 2, 2);
  ExpectBothVelocitiesNear(ExpectSameBitsInEveryOrder({floor, rough_conveyor}, inputs).result, kUnitX, 0);
  Solid rough_floor = floor;
  rough_floor.friction_scale = 2;
  ExpectBothVelocitiesNear(ExpectSameBitsInEveryOrder({rough_floor, conveyor}, inputs).result, {}, 0);
}

TEST(MoveParticle, ConveyorAndFixedSide_MeanOverTheOrders) {
  // A floor moving at {0, 1, 0} and a fixed side with normal {1, 0, 0}, {-1, 1, -1} at their corner: pushes 1 and 1,
  // v0 {0, 1, 0}. The floor's friction takes nothing (v0 moves with it), the side's takes min(friction, 1): the floor
  // first, {0, 1 - friction, 0}; the side first, the floor then brings the particle back toward its own velocity. The
  // mean over the two orders: friction 0 {0, 1, 0}, 0.2 {0, 0.9, 0}, 0.5 {0, 0.75, 0}, 2 {0, 0.5, 0}, both orders the
  // same bits.
  const Solid conveyor = BuildMovingFlatSolid(kUnitZ, kUnitY, 1, 1);
  const Solid side = move_particle_cases::BuildFlatSolid({}, kUnitX, kNoMotion, 2);
  const std::array<std::pair<double, double>, 4> rows = {
      std::pair<double, double>{0, 1},
      std::pair<double, double>{0.2, 0.9},
      std::pair<double, double>{0.5, 0.75},
      std::pair<double, double>{2, 0.5},
  };
  for (const auto& [friction, along_y] : rows) {
    SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(
        {.x = kTolerance, .y = 0, .z = kTolerance},
        {.x = -1, .y = 1, .z = -1},
        0.01
    );
    inputs.friction = friction;
    const MoveParticleCall call = ExpectSameBitsInEveryOrder({conveyor, side}, inputs);
    ASSERT_EQ(call.error, ContactError::kNone);
    EXPECT_EQ(call.result.hit_count, 1);
    ExpectBothVelocitiesNear(call.result, {.x = 0, .y = along_y, .z = 0}, 1e-15);
  }
}

TEST(MoveParticle, RisingWallBesideFloor_FrictionLiftsOnlyAboveThePush) {
  // A floor and a wall with normal {-1, 0, 0} rising at 2 m/s, the particle at their corner moving at {1, 0, -1}:
  // pushes 1 and 1. Friction lifts the particle along the wall only when it overcomes the push that pressed it on the
  // floor: the second solve starts from v + F. Friction 0 and 0.5: {0, 0, 0}; 1.5: {0, 0, 0.5}. From v0 + F, friction
  // 0.5 would lift it at 0.5 m/s.
  const Solid floor = move_particle_cases::BuildFloor(1);
  const Solid rising_wall = BuildMovingFlatSolid(-kUnitX, {.x = 0, .y = 0, .z = 2}, 1, 2);
  const std::array<std::pair<double, double>, 3> rows = {
      std::pair<double, double>{0, 0},
      std::pair<double, double>{0.5, 0},
      std::pair<double, double>{1.5, 0.5},
  };
  for (const auto& [friction, along_z] : rows) {
    SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(
        {.x = -kTolerance, .y = 0, .z = kTolerance},
        {.x = 1, .y = 0, .z = -1},
        0.01
    );
    inputs.friction = friction;
    const MoveParticleCall call = RunMove(BuildOrderedScene({floor, rising_wall}, inputs));
    ASSERT_EQ(call.error, ContactError::kNone) << friction;
    ExpectBothVelocitiesNear(call.result, {.x = 0, .y = 0, .z = along_z}, 1e-15);
  }
}

TEST(MoveParticle, ThreeFacesSharingOneLine_FrictionDoesNotJumpWithTheVelocity) {
  // A floor at z = (Sqrt(2) - 1) tol (friction scale 1) and two faces through the origin at 45 degrees, {s, 0, s}
  // (scale 0.5) and {-s, 0, s} (scale 2), s = Sqrt(0.5): their normals share the y axis, so the smallest change has
  // several sets of pushes. The solver keeps the largest sum of friction limits: the two faces, y
  // 1 - 0.3 (0.5 + 2) / Sqrt(2) = 0.469669914110 for the velocities {e, 1, -1}, e = -1e-13, 0 and 1e-13, in every
  // order. The first subset in order gave 0.7 for e >= 0: a jump of 0.23 for a change of 1e-13.
  const double half_root_two = std::sqrt(0.5);
  const Solid floor = move_particle_cases::BuildFlatSolid(
      {.x = 0, .y = 0, .z = (std::numbers::sqrt2 - 1) * kTolerance},
      kUnitZ,
      kNoMotion,
      1
  );
  const Solid smooth_face = BuildMovingFlatSolid({.x = half_root_two, .y = 0, .z = half_root_two}, {}, 0.5, 2);
  const Solid rough_face = BuildMovingFlatSolid({.x = -half_root_two, .y = 0, .z = half_root_two}, {}, 2, 3);
  for (const double sideways : {-1e-13, 0.0, 1e-13}) {
    SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(
        {.x = 0, .y = 0, .z = std::numbers::sqrt2 * kTolerance},
        {.x = sideways, .y = 1, .z = -1},
        0.01
    );
    inputs.friction = 0.3;
    const MoveParticleCall call = ExpectSameBitsInEveryOrder({floor, smooth_face, rough_face}, inputs);
    ASSERT_EQ(call.error, ContactError::kNone);
    EXPECT_EQ(call.result.hit_count, 1);
    EXPECT_NEAR(call.result.velocity.for_next_step.y, 0.469669914110, 1e-12) << sideways;
  }
}

TEST(MoveParticle, TwoReactingSolidsInCrease_IsSecondReactingSolid) {
  // Plan decision P5, stated: at the rest point of a 60 degree wedge, both faces reacting, a velocity {-1, 0, -2}
  // pushes into both. One response charges both: kSecondReactingSolid naming the higher index, solid_id 2, in both
  // orders. With only the floor reacting: one hit, its push {0, 0, 2.5773502691896262}.
  const double wedge_angle = std::numbers::pi / 3;  // rad
  const SolidContactInputs inputs = BuildRestPointInputs(wedge_angle, {.x = -1, .y = 0, .z = -2}, 0);
  Solid floor = move_particle_cases::BuildFloor(1);
  Solid side = move_particle_cases::BuildWedgeSide(std::numbers::pi - wedge_angle, 2);
  floor.reports_reaction = true;
  side.reports_reaction = true;
  ExpectCloseCall(RunMove(BuildOrderedScene({floor, side}, inputs)), ContactCloseCall::kSecondReactingSolid, 2);
  ExpectCloseCall(RunMove(BuildOrderedScene({side, floor}, inputs)), ContactCloseCall::kSecondReactingSolid, 2);
  side.reports_reaction = false;
  const MoveParticleCall call = RunMove(BuildOrderedScene({floor, side}, inputs));
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.reaction.solid_index, 0U);
  ExpectVec3Near(call.result.reaction.velocity_for_next_step_change, {.x = 0, .y = 0, .z = 2.5773502691896262}, 1e-15);
}

TEST(MoveParticle, StartInsideAndLeaving_IsOneHitWithoutChange) {
  // A start tol / 2 inside a floor, leaving it at {1, 0, 2}: the start was inside, so it is one hit, kind
  // kStartedInsideSolid, and the response to a velocity that leaves changes nothing, bit for bit. A start tol / 2
  // outside, leaving: the move is silent, no hit.
  const Vec3 velocity = {.x = 1, .y = 0, .z = 2};
  const std::vector<Solid> solids = {move_particle_cases::BuildFloor(1)};
  const MoveParticleCall inside = RunMove(BuildOrderedScene(
      solids,
      move_particle_cases::BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = -kTolerance / 2}, velocity, 0.01)
  ));
  ASSERT_EQ(inside.error, ContactError::kNone);
  EXPECT_EQ(inside.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(inside.result.hit_count, 1);
  ExpectBothVelocitiesNear(inside.result, velocity, 0);
  const MoveParticleCall outside = RunMove(BuildOrderedScene(
      solids,
      move_particle_cases::BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = kTolerance / 2}, velocity, 0.01)
  ));
  ASSERT_EQ(outside.error, ContactError::kNone);
  EXPECT_EQ(outside.result.kind, ContactKind::kNone);
  EXPECT_EQ(outside.result.hit_count, 0);
}

// Computes whether two calls have the same error, close call, kind, hit count and for_next_step, bit for bit.
bool ComputeIsSameBits(const MoveParticleCall& actual, const MoveParticleCall& expected) {
  const Vec3 actual_velocity = actual.result.velocity.for_next_step;
  const Vec3 expected_velocity = expected.result.velocity.for_next_step;
  const bool is_same_velocity = actual_velocity.x == expected_velocity.x && actual_velocity.y == expected_velocity.y
      && actual_velocity.z == expected_velocity.z;
  const bool is_same_outcome = actual.error == expected.error
      && actual.failure.close_call == expected.failure.close_call && actual.result.kind == expected.result.kind
      && actual.result.hit_count == expected.result.hit_count;
  return is_same_outcome && is_same_velocity;
}

// The active surfaces of the solids at a position, at the pose of time 0, as a joint response reads them; the inputs
// point at solids, which must outlive them.
struct JointScene {
  std::vector<Solid> solids;
  SolidContactInputs inputs{};
  ActiveSurfaces active{};
};

// Builds the joint scene of those solids, with the particle at position and both velocities velocity, friction.
JointScene BuildJointScene(std::vector<Solid> solids, Vec3 position, Vec3 velocity, double friction) {
  JointScene scene;
  scene.solids = std::move(solids);
  for (std::size_t solid_index = 0; solid_index < scene.solids.size(); ++solid_index) {
    scene.solids[solid_index].solid_id = static_cast<std::uint32_t>(solid_index + 1);
  }
  scene.inputs = move_particle_cases::BuildSeveralSurfaceInputs(position, velocity, 0.01);
  scene.inputs.friction = friction;
  scene.inputs.solids = scene.solids.data();
  scene.inputs.solid_count = static_cast<std::uint32_t>(scene.solids.size());
  ParticleMove move;
  move.result.position = position;
  move.result.velocity = scene.inputs.velocity;
  scene.active = FindActiveSurfaces(scene.inputs, move, nullptr);
  return scene;
}

// Finds the friction share of the floor (normal {0, 0, 1}) in the joint response of a velocity over two active
// surfaces that move differently, from the pieces ComputeJointChange uses: the targets, the frictionless solve and the
// mean over orders. Fails the calling test unless both directions are used and the mean is taken.
Vec3 FindFloorFrictionShare(const ActiveSurfaces& active, Vec3 velocity) {
  const JointTargets built = BuildJointTargets(active, velocity);
  const SmallestCorrection correction = SolveSmallestCorrection(built.targets, 0, CorrectedQuantity::kVelocity);
  EXPECT_EQ(correction.used_count, 2);
  const JointFriction joint_friction = ComputeJointFriction(active, built, correction, velocity);
  EXPECT_TRUE(joint_friction.moves_differently);
  Vec3 floor_share{};
  for (int used_index = 0; used_index < correction.used_count; ++used_index) {
    const int direction_index = correction.used_surfaces[used_index];
    const ActiveSurface& pushing = active.surfaces[built.pushing_surfaces[direction_index]];
    floor_share = pushing.normal_out_of_solid.z == 1 ? joint_friction.shares[used_index] : floor_share;
  }
  return floor_share;
}

TEST(MoveParticle, MeanOverOrdersBetweenMovingSolids_FrictionCanPassStick_KnownLimit) {
  // A known limit of the mean over orders (plan decision P20, item 5), pinned so that an exact rule for solids that
  // move differently shows up as a change here. The review's scene: a 26.73 degree wedge whose floor moves at
  // {-0.1206, -0.0055, 0} and whose side moves along itself, friction 0.716, at the rest point. Each order's floor step
  // stops at sticking to the floor, but the side's step then moves the velocity on: in the mean, the floor's own
  // friction share {-0.0086, -0.2203, 0} points along the particle's final slip relative to the floor (-0.444 m/s
  // along y): the floor's friction pushed the particle past sticking to it, by 0.2 of the summed speeds.
  // NOLINTBEGIN(modernize-use-std-numbers): measured values, not constants.
  const double wedge_angle = 26.729187711001572 * std::numbers::pi / 180;  // rad
  const Vec3 floor_velocity = {.x = -0.12057519706619924, .y = -0.0054700361225109573, .z = 0};
  const Vec3 side_velocity = {.x = -0.3412340199331419, .y = -0.89345098324273942, .z = -0.17184069222524945};
  const Vec3 velocity = {.x = -0.67764259646844816, .y = 0.76482039646153233, .z = 0.32567414729292232};
  const double friction = 0.71581604590110215;
  // NOLINTEND(modernize-use-std-numbers)
  const JointScene scene = BuildJointScene(
      {
          BuildMovingFlatSolid(kUnitZ, floor_velocity, 1, 1),
          BuildMovingFlatSolid(ToWedgeSideNormal(wedge_angle), side_velocity, 1, 2),
      },
      {.x = kTolerance / std::tan(wedge_angle / 2), .y = 0, .z = kTolerance},
      velocity,
      friction
  );
  ASSERT_EQ(scene.active.count, 2);
  JointChange change;
  std::uint32_t squeezing_solid_index = kNoSolid;
  ASSERT_EQ(ComputeJointChange(scene.active, velocity, 0, 1, &change, &squeezing_solid_index), ContactError::kNone);
  const double speeds = math::Length(velocity) + math::Length(floor_velocity) + math::Length(side_velocity);  // m/s
  const Vec3 floor_share = FindFloorFrictionShare(scene.active, velocity);
  const Vec3 slip_on_floor = change.velocity - floor_velocity;
  const Vec3 sideways_slip = {.x = slip_on_floor.x, .y = slip_on_floor.y, .z = 0};
  EXPECT_GT(math::Dot(floor_share, sideways_slip) / math::Length(floor_share), 0.15 * speeds);
  EXPECT_NEAR(slip_on_floor.y, -0.444, 0.001);
}

TEST(MoveParticle, CoincidentReactingFloorCopyInAnyOrder_StickChargesTheReactingOne) {
  // A stick on two coincident fixed floors that differ only in reports_reaction: equal in every value, the reacting
  // one sorts first, so the stick is charged to it in every order of the array (the sliver of SliverStop, moving at
  // {-2, 1, -0.001} into the floor). As built in round 3, the array order decided and the reaction came and went.
  const double angle = 1e-7;  // rad
  Solid reacting_floor = move_particle_cases::BuildFloor(1);
  reacting_floor.reports_reaction = true;
  const Solid floor = move_particle_cases::BuildFloor(1);
  const Solid side = move_particle_cases::BuildWedgeSide(std::numbers::pi - angle, 1);
  const SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(
      {.x = kTolerance / std::tan(angle / 2), .y = 0, .z = kTolerance},
      {.x = -2, .y = 1, .z = -0.001},
      0.01
  );
  const std::array<std::vector<Solid>, 3> orders = {
      std::vector<Solid>{floor, side, reacting_floor},
      std::vector<Solid>{reacting_floor, side, floor},
      std::vector<Solid>{side, floor, reacting_floor},
  };
  const std::array<std::uint32_t, 3> reacting_indexes = {2, 0, 2};
  for (std::size_t order_index = 0; order_index < orders.size(); ++order_index) {
    const MoveParticleCall call = RunMove(BuildOrderedScene(orders.at(order_index), inputs));
    ASSERT_EQ(call.error, ContactError::kNone) << order_index;
    ExpectBothVelocitiesNear(call.result, {}, 0);
    EXPECT_EQ(call.result.reaction.solid_index, reacting_indexes.at(order_index)) << order_index;
    ExpectVec3Equal(call.result.reaction.velocity_for_next_step_change, {.x = 2, .y = -1, .z = 0.001});
  }
}

TEST(MoveParticle, ReactingFloorTiltedByRounding_PushesForItsDirection) {
  // At the rest point of a 60 degree wedge, a plain floor and a reacting floor whose normal is tilted by 2^-45, which
  // sorts it after the plain one: one direction, both push, equal friction, so the one that reports a reaction pushes
  // (plan decision P20, item 4) and is charged: in both orders the reaction is the reacting floor's.
  const double wedge_angle = std::numbers::pi / 3;  // rad
  Solid reacting_floor = move_particle_cases::BuildFlatSolid({}, {.x = 0x1p-45, .y = 0, .z = 1}, kNoMotion, 1);
  reacting_floor.reports_reaction = true;
  const Solid floor = move_particle_cases::BuildFloor(1);
  const Solid side = move_particle_cases::BuildWedgeSide(std::numbers::pi - wedge_angle, 1);
  const SolidContactInputs inputs = BuildRestPointInputs(wedge_angle, {.x = -0.2, .y = 1, .z = -1}, 0);
  const MoveParticleCall plain_first = RunMove(BuildOrderedScene({floor, reacting_floor, side}, inputs));
  ASSERT_EQ(plain_first.error, ContactError::kNone);
  EXPECT_EQ(plain_first.result.reaction.solid_index, 1U);
  const MoveParticleCall reacting_first = RunMove(BuildOrderedScene({reacting_floor, floor, side}, inputs));
  ASSERT_EQ(reacting_first.error, ContactError::kNone);
  EXPECT_EQ(reacting_first.result.reaction.solid_index, 0U);
}

TEST(MoveParticle, RoughFloorTiltedByRounding_StillActsForItsDirection) {
  // RoughGroundOnWedgeFloor with the rough floor's normal tilted by -2^-45: its own target is 5.7e-15 m/s below the
  // plain floor's. It still pushes, within the rounding window 2^-40 (|v|_1 + |u|_1) of the direction's target, and
  // being the rougher it acts: y 0.507623956930, not the plain floor's 0.730717967697.
  const double wedge_angle = std::numbers::pi / 3;  // rad
  Solid rough_floor = move_particle_cases::BuildFlatSolid({}, {.x = -0x1p-45, .y = 0, .z = 1}, kNoMotion, 1);
  rough_floor.friction_scale = 2;
  const SolidContactInputs inputs = BuildRestPointInputs(wedge_angle, {.x = -0.2, .y = 1, .z = -1}, 0.2);
  const MoveParticleCall call = RunMove(BuildOrderedScene(
      {
          move_particle_cases::BuildFloor(1),
          rough_floor,
          move_particle_cases::BuildWedgeSide(std::numbers::pi - wedge_angle, 1),
      },
      inputs
  ));
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_NEAR(call.result.velocity.for_next_step.y, 0.507623956930, 1e-12);
}

TEST(ApplySecondSolve, SlowApproachBesideFastSolid_IsRemoved) {
  // Plan decision P20, item 3 (a), for the second solve: "no change" reads the allowance passed, not the solve's own
  // 2^-46 (speeds). Beside a floor sliding at 100 m/s, a velocity with friction {1e-13, 0, 0} approaches the fixed
  // upright at x = 1 by 1e-13 m/s, within 2^-46 * 100 m/s but above the allowance 0: the change removes it, x 0.
  const JointScene scene = BuildJointScene(
      {
          BuildMovingFlatSolid(kUnitZ, {.x = 100, .y = 0, .z = 0}, 1, 1),
          move_particle_cases::BuildFlatSolid(kUnitX, -kUnitX, kNoMotion, 1),
      },
      {.x = 1 - kTolerance, .y = 0, .z = kTolerance},
      {},
      0
  );
  ASSERT_EQ(scene.active.count, 2);
  JointChange change;
  std::uint32_t squeezing_solid_index = kNoSolid;
  const ContactError error =
      ApplySecondSolve(scene.active, {.x = 1e-13, .y = 0, .z = 0}, 0, 100, &change, &squeezing_solid_index);
  ASSERT_EQ(error, ContactError::kNone);
  EXPECT_EQ(change.velocity.x, 0);
}

TEST(ComputeJointChange, SecondSolve_AllowanceKeepsTheFirstSolves) {
  // RisingWallBesideFloor at friction 1.5: speeds 4 m/s; the first solve's change {-1, 0, 1} gives the allowance
  // 2^-46 (4 + 2) = 6 * 2^-46; the friction {0, 0, 1.5} leaves z = {1, 0, 0.5}, and the second solve's {-1, 0, 0}
  // gives 2^-46 (3.5 + 1) = 4.5 * 2^-46. The response keeps the larger, the first solve's: its rounding is in the
  // velocity too.
  const JointScene scene = BuildJointScene(
      {
          move_particle_cases::BuildFloor(1),
          BuildMovingFlatSolid(-kUnitX, {.x = 0, .y = 0, .z = 2}, 1, 1),
      },
      {.x = -kTolerance, .y = 0, .z = kTolerance},
      {.x = 1, .y = 0, .z = -1},
      1.5
  );
  ASSERT_EQ(scene.active.count, 2);
  JointChange change;
  std::uint32_t squeezing_solid_index = kNoSolid;
  const Vec3 velocity = {.x = 1, .y = 0, .z = -1};
  ASSERT_EQ(ComputeJointChange(scene.active, velocity, 0, 2, &change, &squeezing_solid_index), ContactError::kNone);
  ExpectVec3Equal(change.velocity, {.x = 0, .y = 0, .z = 0.5});
  EXPECT_EQ(change.approach_allowance, 6 * kApproachLeftAfterResponseShare);
}

TEST(ApplyJointResponse, OneFloor_RaisesBothAllowances) {
  // One response on a fixed floor: each velocity's allowance is ComputeLargestApproachLeftAfterResponse of the
  // velocity it passed, {3, 0, -1} for this step and {1, 0, -1} for the next: 4 and 2 times 2^-46.
  const JointScene scene =
      BuildJointScene({move_particle_cases::BuildFloor(1)}, {.x = 0, .y = 0, .z = kTolerance}, {}, 0);
  ASSERT_EQ(scene.active.count, 1);
  ParticleMove move;
  move.result.position = {.x = 0, .y = 0, .z = kTolerance};
  move.result.velocity = {.for_next_step = {.x = 1, .y = 0, .z = -1}, .for_this_step = {.x = 3, .y = 0, .z = -1}};
  ContactFailure failure;
  ASSERT_EQ(
      ApplyJointResponse(scene.inputs, scene.active, scene.active.surfaces[0], &move, &failure),
      ContactError::kNone
  );
  EXPECT_EQ(move.this_step_approach_allowance, 4 * kApproachLeftAfterResponseShare);
  EXPECT_EQ(move.next_step_approach_allowance, 2 * kApproachLeftAfterResponseShare);
  EXPECT_TRUE(move.has_response);
  EXPECT_EQ(move.result.hit_count, 1);
}

// One generated corner: two to four flat faces through the origin whose normals all lean toward a common free
// direction, the solids, and the particle's inputs.
struct GeneratedCorner {
  std::vector<Solid> solids;
  SolidContactInputs inputs{};
};

// Draws a unit vector from the generator: three signed coordinates, again until the length is in [0.1, 1].
Vec3 DrawUnitVector(CaseGenerator* generator) {
  Vec3 drawn{};
  do {
    drawn = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
  } while (math::Length(drawn) < 0.1 || math::Length(drawn) > 1);
  return math::Normalized(drawn);
}

// Builds one corner: in this order, the face count, the free direction d, for each face its sideways lean, whether
// it moves and its velocity and friction scale, then the friction, the start kind and the velocity. Each normal is
// Normalized(d + 0.9 lean) with lean a unit vector, so every normal is within 65 degrees of d. Half of the draws move
// their solids, up to 1 m/s per axis; friction scales are 1 or 2 Next(); friction 0 or 2 Next(). The start is the
// rest point found by the move off near surfaces (from the origin), a point 0.3 tol inside the rest gap along d, or
// 0.002 m out along d on a path into the corner.
GeneratedCorner BuildGeneratedCorner(CaseGenerator* generator) {
  GeneratedCorner corner;
  const int face_count = 2 + static_cast<int>(3 * generator->Next());
  const Vec3 free_direction = DrawUnitVector(generator);
  const bool is_moving = generator->Next() < 0.5;
  for (int face_index = 0; face_index < face_count; ++face_index) {
    const Vec3 normal = math::Normalized(free_direction + 0.9 * DrawUnitVector(generator));
    const Vec3 velocity = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
    const double friction_scale = generator->Next() < 0.5 ? 1 : 2 * generator->Next();
    corner.solids.push_back(BuildMovingFlatSolid(normal, is_moving ? velocity : Vec3{}, friction_scale, 1));
  }
  const double friction = generator->Next() < 0.2 ? 0 : 2 * generator->Next();
  const double start_kind = generator->Next();
  const Vec3 velocity = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
  const MoveParticleCase scene =
      BuildOrderedScene(corner.solids, move_particle_cases::BuildSeveralSurfaceInputs({}, velocity, 0.01));
  const SolidContactInputs at_origin = ToSolidContactInputs(scene);
  const RestGapMove rest = MoveToRestGapOfNearSurfaces(at_origin, 0, 0.5 * kTolerance * free_direction);
  Vec3 start = rest.is_found ? rest.moved_point : 4 * kTolerance * free_direction;
  if (start_kind < 0.3) {
    start = start - 0.3 * kTolerance * free_direction;
  } else if (start_kind < 0.6) {
    start = start + 0.002 * free_direction;
  }
  corner.inputs = move_particle_cases::BuildSeveralSurfaceInputs(start, velocity, 0.01);
  corner.inputs.friction = friction;
  return corner;
}

TEST(MoveParticle, GeneratedOneSurfaceHits_SameBitsAsContactResponse) {
  // Plan decision P20, item 2: with one active surface each velocity gets ComputeContactResponse with that surface and
  // nothing else, the single-surface response of Task 7 bit for bit. 2000 flat solids through the origin, in this
  // order of draws: the normal (DrawUnitVector), a solid velocity of three NextSigned / 2 m/s or none, the velocity
  // (three NextSigned m/s), friction 2 Next; the start 0.05 m out along the normal, T 0.1 s. Every call whose path hits
  // the solid has both velocities equal to the response of the velocity given, at the hit's normal.
  CaseGenerator generator;
  int hit_calls = 0;
  int differing_calls = 0;
  for (int draw = 0; draw < 2000; ++draw) {
    const Vec3 normal = DrawUnitVector(&generator);
    const bool is_moving = generator.Next() < 0.5;
    const Vec3 drawn_velocity = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const Vec3 solid_velocity = is_moving ? 0.5 * drawn_velocity : Vec3{};
    const Vec3 velocity = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(0.05 * normal, velocity, 0.1);
    inputs.friction = 2 * generator.Next();
    const MoveParticleCall call =
        RunMove(BuildOrderedScene({BuildMovingFlatSolid(normal, solid_velocity, 1, 1)}, inputs));
    if (call.error != ContactError::kNone || call.result.hit_count != 1) {
      continue;
    }
    ++hit_calls;
    const Vec3 expected =
        ComputeContactResponse(velocity, solid_velocity, call.result.normal_out_of_solid, inputs.friction);
    const Vec3 actual = call.result.velocity.for_next_step;
    const bool is_same = actual.x == expected.x && actual.y == expected.y && actual.z == expected.z;
    differing_calls += is_same ? 0 : 1;
  }
  std::cout << "one-surface hits " << hit_calls << ", differing from ComputeContactResponse " << differing_calls
            << "\n";
  EXPECT_GT(hit_calls, 400);  // 456 of the draws hit.
  EXPECT_EQ(differing_calls, 0);
}

TEST(MoveParticle, GeneratedCornersInEveryOrder_SameBits) {
  // Plan decision P20, item 1: the active surfaces are sorted by value, so every order of the solids array gives the
  // same outcome, kind, hit count and velocity bits; only the named solid and the end position (by rounding of the
  // move off near surfaces, at most 3.4e-9 tol in the model) may follow the array. 2000 corners of two to four faces.
  CaseGenerator generator;
  int differing_draws = 0;
  int failed_draws = 0;
  for (int draw = 0; draw < 2000; ++draw) {
    const GeneratedCorner corner = BuildGeneratedCorner(&generator);
    const MoveParticleCall first = RunMove(BuildOrderedScene(corner.solids, corner.inputs));
    failed_draws += first.error == ContactError::kNone ? 0 : 1;
    std::vector<std::size_t> order(corner.solids.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
      order[index] = index;
    }
    bool is_same = true;
    while (std::ranges::next_permutation(order).found) {
      const MoveParticleCall call = RunMove(BuildOrderedScene(ToPermutedSolids(corner.solids, order), corner.inputs));
      is_same = is_same && ComputeIsSameBits(call, first);
    }
    differing_draws += is_same ? 0 : 1;
  }
  std::cout << "generated corners 2000, failed " << failed_draws << ", differing between orders " << differing_draws
            << "\n";
  EXPECT_EQ(differing_draws, 0);
  EXPECT_LE(failed_draws, 100);  // 0: the comparison is not between failures.
}

// The coordinate moved by one rounding unit down (direction -1), not at all (0) or up (+1).
double ToNeighbourDouble(double coordinate, int direction) {
  if (direction == 0) {
    return coordinate;
  }
  return std::nextafter(coordinate, direction > 0 ? INFINITY : -INFINITY);
}

// Computes whether the wedge of that angle, with its rest-point inputs (BuildRestPointInputs), gives the same bits from
// the 27 starts one rounding unit around the rest point, in both array orders. A failing rest-point start counts as
// differing, so the comparison is never between failures.
bool ComputeIsSameForEveryRounding(double wedge_angle, const SolidContactInputs& center) {
  const Solid floor = move_particle_cases::BuildFloor(1);
  const Solid side = move_particle_cases::BuildWedgeSide(std::numbers::pi - wedge_angle, 2);
  const MoveParticleCall reference = RunMove(BuildOrderedScene({floor, side}, center));
  bool is_same = reference.error == ContactError::kNone;
  for (int start_index = 0; start_index < 27; ++start_index) {
    SolidContactInputs inputs = center;
    inputs.start_position = {
        .x = ToNeighbourDouble(center.start_position.x, start_index % 3 - 1),
        .y = ToNeighbourDouble(center.start_position.y, start_index / 3 % 3 - 1),
        .z = ToNeighbourDouble(center.start_position.z, start_index / 9 - 1),
    };
    is_same = is_same && ComputeIsSameBits(RunMove(BuildOrderedScene({floor, side}, inputs)), reference)
        && ComputeIsSameBits(RunMove(BuildOrderedScene({side, floor}, inputs)), reference);
  }
  return is_same;
}

TEST(MoveParticle, GeneratedRestPointStarts_SameBitsForEveryRounding) {
  // Re-review 2's d1: at the rest point of a wedge of 10 to 170 degrees, friction 0 or 0.1 to 2, a velocity in the
  // unit ball (at least 0.05) approaching a face. The responses one surface after the other jumped in 16 % of the
  // draws when the start moved by one rounding unit. One joint response reads the same active surfaces from every one
  // of the 27 starts: the 54 results, both array orders, are the same bits.
  CaseGenerator generator;
  int differing_draws = 0;
  for (int draw = 0; draw < 2000; ++draw) {
    const double wedge_angle = (10 + 160 * generator.Next()) * std::numbers::pi / 180;  // rad
    const double friction = generator.Next() < 0.5 ? 0 : 0.1 + 1.9 * generator.Next();
    Vec3 velocity{};
    do {
      velocity = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    } while (math::Length(velocity) > 1 || math::Length(velocity) < 0.05
             || (velocity.z >= 0 && math::Dot(ToWedgeSideNormal(wedge_angle), velocity) >= 0));
    const SolidContactInputs center = BuildRestPointInputs(wedge_angle, velocity, friction);
    differing_draws += ComputeIsSameForEveryRounding(wedge_angle, center) ? 0 : 1;
  }
  std::cout << "rest-point starts 2000 x 54, draws with differing results " << differing_draws << "\n";
  EXPECT_EQ(differing_draws, 0);
}

// What the fixed corners of GeneratedFixedCorners count.
struct FixedCornerCounts {
  int successes = 0;
  int energy_gains = 0;  // |v'|^2 > |v|^2 (1 + 1e-12).
  int reversed_edges = 0;  // Two faces: the part along their common line changed sign.
  int grown_edges = 0;  // Two faces: the part along their common line grew.
};

// Draws one fixed corner and adds what it shows to *counts. In this order: the face count (2 to 4), each unit normal
// (DrawUnitVector), the start kind, the velocity (each axis in [-1, 1)) and the friction (0 with probability 0.2, else
// up to 3). The start is the rest point found by the move off near surfaces from tol / 2 along the bisector of the
// first two normals, 0.002 m out along it (kind below 0.2) or 0.3 tol inside the rest gap (below 0.4). A draw whose
// rest point is not found, or whose move fails, counts no success.
void ApplyFixedCornerDraw(CaseGenerator* generator, FixedCornerCounts* counts) {
  const int face_count = 2 + static_cast<int>(3 * generator->Next());
  std::vector<Solid> solids;
  std::vector<Vec3> normals;
  for (int face_index = 0; face_index < face_count; ++face_index) {
    normals.push_back(DrawUnitVector(generator));
    solids.push_back(move_particle_cases::BuildFlatSolid({}, normals.back(), kNoMotion, 1));
  }
  const double start_kind = generator->Next();
  const Vec3 velocity = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
  const double friction = generator->Next() < 0.2 ? 0 : 3 * generator->Next();
  const Vec3 lean = math::Normalized(normals[0] + normals[1]);
  MoveParticleCase scene =
      BuildOrderedScene(solids, move_particle_cases::BuildSeveralSurfaceInputs({}, velocity, 0.01));
  const RestGapMove rest = MoveToRestGapOfNearSurfaces(ToSolidContactInputs(scene), 0, kTolerance / 2 * lean);
  if (!rest.is_found) {
    return;
  }
  double start_offset = 0;  // m
  if (start_kind < 0.2) {
    start_offset = 0.002;
  } else if (start_kind < 0.4) {
    start_offset = -0.3 * kTolerance;
  }
  scene.inputs.start_position = rest.moved_point + start_offset * lean;
  scene.inputs.friction = friction;
  const MoveParticleCall call = RunMove(scene);
  if (call.error != ContactError::kNone) {
    return;
  }
  ++counts->successes;
  const Vec3 result = call.result.velocity.for_next_step;
  counts->energy_gains += math::Dot(result, result) > math::Dot(velocity, velocity) * (1 + 1e-12) ? 1 : 0;
  if (face_count != 2) {
    return;
  }
  const Vec3 edge = math::Normalized(math::Cross(normals[0], normals[1]));
  const double along_before = math::Dot(velocity, edge);  // m/s
  const double along_after = math::Dot(result, edge);  // m/s
  counts->reversed_edges += along_before * along_after < -1e-12 ? 1 : 0;
  counts->grown_edges += std::abs(along_after) > std::abs(along_before) + 1e-12 ? 1 : 0;
}

TEST(MoveParticle, GeneratedFixedCorners_FrictionNeverAddsEnergy) {
  // Re-review 2's q4b, fixed draws: two to four fixed flat faces through the origin, the start at the rest point, 0.3
  // tol inside it or 0.002 m out, friction 0 or up to 3. Against fixed solids the joint response never lengthens the
  // velocity, |v'|^2 <= |v|^2 (1 + 1e-12); with two faces the part along their common line is neither reversed nor
  // grown (plan decision P20, item 13).
  CaseGenerator generator;
  FixedCornerCounts counts;
  for (int draw = 0; draw < 2000; ++draw) {
    ApplyFixedCornerDraw(&generator, &counts);
  }
  std::cout << "fixed corners 2000, succeeded " << counts.successes << ", energy gains " << counts.energy_gains
            << ", edge reversed " << counts.reversed_edges << ", grown " << counts.grown_edges << "\n";
  EXPECT_GE(counts.successes, 1700);  // 1763: the invariants are not vacuous.
  EXPECT_EQ(counts.energy_gains, 0);
  EXPECT_EQ(counts.reversed_edges, 0);
  EXPECT_EQ(counts.grown_edges, 0);
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

TEST(SolveSmallestCorrection, VelocityWithinSlack_IsNoChange) {
  // A velocity already within its slack of every target needs no change: used_count 0 and a zero change, so rounding
  // never changes it. A position has no empty subset: the same target moves it along the normal.
  const CorrectionTargets far_target = BuildCorrectionTargets({{.x = 0, .y = 0, .z = 1}}, {1});
  ExpectCorrection(
      SolveSmallestCorrection(far_target, 0x1p-46, CorrectedQuantity::kVelocity),
      {.x = 0, .y = 0, .z = 1},
      1
  );
  const CorrectionTargets rounding_target = BuildCorrectionTargets({{.x = 0, .y = 0, .z = 1}}, {0x1p-50});
  const SmallestCorrection velocity_change =
      SolveSmallestCorrection(rounding_target, 0x1p-46, CorrectedQuantity::kVelocity);
  ASSERT_TRUE(velocity_change.is_found);
  EXPECT_EQ(velocity_change.used_count, 0);
  ExpectVec3Equal(velocity_change.change, {});
  const SmallestCorrection position_change =
      SolveSmallestCorrection(rounding_target, 0x1p-46, CorrectedQuantity::kPosition);
  ExpectCorrection(position_change, {.x = 0, .y = 0, .z = 0x1p-50}, 1);
}

TEST(SolveSmallestCorrection, VelocityNoChangeReadsTheSlackPassed) {
  // Plan decision P20, item 3 (a): no change only when every target is within the slack passed. A target of 2^-50 m/s
  // with speeds of 1 m/s before the change is within the candidate's own allowance 2^-46 (speeds), but above a slack of
  // 0: a real approach beside a fast solid, which the change removes.
  CorrectionTargets slow_target = BuildCorrectionTargets({{.x = 0, .y = 0, .z = 1}}, {0x1p-50});
  slow_target.speeds_before_change = 1;
  ExpectCorrection(
      SolveSmallestCorrection(slow_target, 0, CorrectedQuantity::kVelocity),
      {.x = 0, .y = 0, .z = 0x1p-50},
      1
  );
  EXPECT_EQ(SolveSmallestCorrection(slow_target, 0x1p-46, CorrectedQuantity::kVelocity).used_count, 0);
}

TEST(SolveSmallestCorrection, VelocityTie_KeepsLargestFrictionSum) {
  // Plan decision P20, item 3 (b): a floor and two faces at 45 degrees whose normals share the y axis, targets 1,
  // Sqrt(0.5) and Sqrt(0.5): the floor alone and the pair of faces give the same change {0, 0, 1}. With frictions 0.3,
  // 0.15 and 0.6 the pair's sum, Sqrt(0.5) (0.15 + 0.6) = 0.53, beats the floor's 0.3: the pair is kept. With no
  // friction the sums tie at 0, and the first in subset order, the floor alone, is kept. A position keeps the first.
  const double half_root_two = std::sqrt(0.5);
  CorrectionTargets shared_line = BuildCorrectionTargets(
      {
          {.x = 0, .y = 0, .z = 1},
          {.x = half_root_two, .y = 0, .z = half_root_two},
          {.x = -half_root_two, .y = 0, .z = half_root_two},
      },
      {1, half_root_two, half_root_two}
  );
  const SmallestCorrection frictionless = SolveSmallestCorrection(shared_line, 0, CorrectedQuantity::kVelocity);
  ExpectCorrection(frictionless, {.x = 0, .y = 0, .z = 1}, 1);
  shared_line.frictions[0] = 0.3;
  shared_line.frictions[1] = 0.15;
  shared_line.frictions[2] = 0.6;
  const SmallestCorrection rough = SolveSmallestCorrection(shared_line, 0, CorrectedQuantity::kVelocity);
  ExpectCorrection(rough, {.x = 0, .y = 0, .z = 1}, 2);
  EXPECT_EQ(rough.used_surfaces[0], 1);
  EXPECT_EQ(rough.used_surfaces[1], 2);
  ExpectPairMultipliers(rough, half_root_two, half_root_two);
  EXPECT_EQ(SolveSmallestCorrection(shared_line, 0, CorrectedQuantity::kPosition).used_count, 1);
}

TEST(SolveSmallestCorrection, FiveDirectionsTheFifthBinding_IsSolvedOverAllOfThem) {
  // A velocity's change reads up to kMaxSlideDirections directions; the smallest change meets at most three as
  // equalities, so every subset of at most three is tried. Here the fifth alone binds: its target 1 along
  // Normalized({0, 1, 1}) meets the other four (dots 0, 0.707, 0.707, 0.5) at no cost.
  const CorrectionTargets five = BuildCorrectionTargets(
      {
          {.x = 1, .y = 0, .z = 0},
          {.x = 0, .y = 1, .z = 0},
          {.x = 0, .y = 0, .z = 1},
          math::Normalized({.x = 1, .y = 1, .z = 0}),
          math::Normalized({.x = 0, .y = 1, .z = 1}),
      },
      {0, 0, 0, 0, 1}
  );
  const SmallestCorrection change = SolveSmallestCorrection(five, 0x1p-46, CorrectedQuantity::kVelocity);
  ExpectCorrection(change, math::Normalized({.x = 0, .y = 1, .z = 1}), 1);
  EXPECT_EQ(change.used_surfaces[0], 4);
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

// Draws a unit vector: three NextSigned, drawn again while shorter than 0.1.
Vec3 DrawUnitVector(CaseGenerator& generator) {
  Vec3 direction{};
  while (math::Length(direction) < 0.1) {
    direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  return math::Normalized(direction);
}

// Draws a velocity of the sharp-surface set: 2^(60 Next - 30) m/s times three NextSigned.
Vec3 DrawWideRangeVelocity(CaseGenerator& generator) {
  const double scale = std::exp2(60 * generator.Next() - 30);  // m/s
  return scale * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
}

// Builds the normals of one case of the sharp-surface set: by Next < 0.5 a pair with 1 + c = 2^(-40 + 36 Next) (n_a a
// DrawUnitVector, a second one made square to it, and n_b = -(1 - s) n_a + Sqrt(s (2 - s)) t), else a triple of
// DrawUnitVector drawn again until |Dot(n_1, Cross(n_2, n_3))| >= 2^-20.
CorrectionTargets BuildSharpSurfaceNormals(CaseGenerator& generator) {
  CorrectionTargets built;
  if (generator.Next() < 0.5) {
    const double one_plus_dot = std::exp2(-40 + 36 * generator.Next());
    const Vec3 first_normal = DrawUnitVector(generator);
    Vec3 square = DrawUnitVector(generator);
    square = math::Normalized(square - math::Dot(first_normal, square) * first_normal);
    built.normals[0] = first_normal;
    built.normals[1] = -(1 - one_plus_dot) * first_normal + std::sqrt(one_plus_dot * (2 - one_plus_dot)) * square;
    built.count = 2;
    return built;
  }
  double triple_product = 0;
  while (std::abs(triple_product) < kVelocityTripleLimit) {
    for (int surface_index = 0; surface_index < 3; ++surface_index) {
      built.normals[surface_index] = DrawUnitVector(generator);
    }
    triple_product = math::Dot(built.normals[0], math::Cross(built.normals[1], built.normals[2]));
  }
  built.count = 3;
  return built;
}

// What one case of the sharp-surface set gave: its refinements and the largest approach it left, in rounding units of
// the summed speeds (0 when no change was needed).
struct SharpSurfaceResult {
  int refinement_count = 0;
  double largest_share_left = 0;
};

// Draws one case of the sharp-surface set (BuildSharpSurfaceNormals, then v and each u_i by DrawWideRangeVelocity),
// solves it for a velocity with slack 0, and fails the calling test unless a change is found that leaves at every
// surface an approach -Dot(n_i, v + w - u_i) within 2^-46 (|v|_1 + |w|_1 + sum |u_i|_1).
SharpSurfaceResult CheckSharpSurfaceCase(CaseGenerator& generator) {
  CorrectionTargets targets = BuildSharpSurfaceNormals(generator);
  const Vec3 velocity = DrawWideRangeVelocity(generator);
  std::array<Vec3, 3> solid_velocities{};
  double speeds = std::abs(velocity.x) + std::abs(velocity.y) + std::abs(velocity.z);  // m/s
  for (int surface_index = 0; surface_index < targets.count; ++surface_index) {
    const Vec3 solid_velocity = DrawWideRangeVelocity(generator);
    solid_velocities.at(static_cast<std::size_t>(surface_index)) = solid_velocity;
    speeds += std::abs(solid_velocity.x) + std::abs(solid_velocity.y) + std::abs(solid_velocity.z);
    targets.targets[surface_index] = -math::Dot(targets.normals[surface_index], velocity - solid_velocity);
  }
  targets.speeds_before_change = speeds;
  const SmallestCorrection correction = SolveSmallestCorrection(targets, 0, CorrectedQuantity::kVelocity);
  SharpSurfaceResult checked;
  if (!correction.is_found) {
    ADD_FAILURE() << "no change found";
    return checked;
  }
  checked.refinement_count = correction.refinement_count;
  const Vec3 changed = velocity + correction.change;
  const Vec3 change = correction.change;
  const double summed_speeds = speeds + std::abs(change.x) + std::abs(change.y) + std::abs(change.z);  // m/s
  for (int surface_index = 0; surface_index < targets.count; ++surface_index) {
    const Vec3 solid_velocity = solid_velocities.at(static_cast<std::size_t>(surface_index));
    const double approach_left = -math::Dot(targets.normals[surface_index], changed - solid_velocity);  // m/s
    EXPECT_LE(approach_left, 0x1p-46 * summed_speeds);
    // A velocity already within the slack keeps its approach, up to 64 rounding units: count only the changes.
    const double share_left = correction.used_count > 0 ? approach_left / (0x1p-52 * summed_speeds) : 0;
    checked.largest_share_left = std::max(checked.largest_share_left, share_left);
  }
  return checked;
}

TEST(SolveSmallestCorrection, SharpPairsAndCornersNearLimit_ApproachLeftWithinShare) {
  // 20,000 cases (BuildSharpSurfaceNormals, then v and each u_i by DrawWideRangeVelocity, in that order), the targets
  // -Dot(n_i, v - u_i), solved for a velocity with slack 0, so the slack is the change's own allowance. The approach
  // left at every surface, -Dot(n_i, v + w - u_i), is within 2^-46 (|v|_1 + |w|_1 + sum |u_i|_1), and at most 3
  // refinements are used (the model: at most 16 * 2^-52 of that sum with the refinements; 177 for pairs near 0.45
  // degrees and 2697 for triples without them).
  CaseGenerator generator;
  int largest_refinement_count = 0;
  double largest_share_left = 0;  // In 2^-52 of the summed speeds.
  for (int case_index = 0; case_index < 20000; ++case_index) {
    const SharpSurfaceResult checked = CheckSharpSurfaceCase(generator);
    largest_refinement_count = std::max(largest_refinement_count, checked.refinement_count);
    largest_share_left = std::max(largest_share_left, checked.largest_share_left);
  }
  std::cout << "sharp pairs and corners: at most " << largest_share_left << " rounding units of the speeds left by a "
            << "change, " << largest_refinement_count << " refinements\n";
  EXPECT_LE(largest_refinement_count, 3);
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
  int squeezed = 0;
  std::map<std::string_view, std::array<int, 8>> failures_per_scene;  // Per close call, in ContactCloseCall order.
};

// Prints, for the report, each scene with a failure: its counts of kTooManyHits, kSqueezedBetweenSolids and
// kSecondReactingSolid.
void PrintFailuresPerScene(const std::map<std::string_view, std::array<int, 8>>& failures_per_scene) {
  std::cout << "failures per scene (too many hits, squeezed, second reacting solid):";
  for (const auto& [scene_name, counts] : failures_per_scene) {
    std::cout << " " << scene_name << " " << counts.at(static_cast<std::size_t>(ContactCloseCall::kTooManyHits)) << " "
              << counts.at(static_cast<std::size_t>(ContactCloseCall::kSqueezedBetweenSolids)) << " "
              << counts.at(static_cast<std::size_t>(ContactCloseCall::kSecondReactingSolid)) << ";";
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
  ++counts.failures_per_scene[scene_name].at(static_cast<std::size_t>(failure.close_call));
  counts.second_reacting_solid += failure.close_call == ContactCloseCall::kSecondReactingSolid ? 1 : 0;
  counts.slow_approach += failure.close_call == ContactCloseCall::kSlowApproachToTurningSolid ? 1 : 0;
  counts.squeezed += failure.close_call == ContactCloseCall::kSqueezedBetweenSolids ? 1 : 0;
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
  // The counts keep the set honest: most paths succeed, many hit, and several hits are common. Measured: 292,046 paths,
  // 68,480 hits, 7,079 with several. The scenes of one sphere or one cylinder, last in the list, are mostly missed by
  // paths drawn within 1 m of their start, so a fifth of the paths hit, not a quarter.
  std::cout << "paths " << counts.paths << ", free " << counts.free_paths << ", hit " << counts.hit_paths
            << ", several hits " << counts.several_hit_paths << ", too many hits " << counts.too_many_hits
            << ", second reacting solid " << counts.second_reacting_solid << ", slow approach " << counts.slow_approach
            << ", squeezed " << counts.squeezed << "\n";
  PrintFailuresPerScene(counts.failures_per_scene);
  EXPECT_GT(counts.paths, 40000);
  EXPECT_GT(counts.hit_paths, counts.paths / 5);
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
  int squeezed = 0;
  std::map<std::string_view, std::array<int, 8>> failures_per_scene;  // Per close call, in ContactCloseCall order.
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
  ++counts.failures_per_scene[scene_name].at(static_cast<std::size_t>(failure.close_call));
  counts.second_reacting_solid += failure.close_call == ContactCloseCall::kSecondReactingSolid ? 1 : 0;
  counts.slow_approach += failure.close_call == ContactCloseCall::kSlowApproachToTurningSolid ? 1 : 0;
  counts.squeezed += failure.close_call == ContactCloseCall::kSqueezedBetweenSolids ? 1 : 0;
}

// Prints the counts of the generated starts near a solid for the report, and fails the calling test unless each kind
// of start is common and the moves end free, with a hit and started inside, each often.
void ExpectNearStartCounts(const NearStartCounts& counts) {
  const std::array<int, 3>& starts = counts.starts_per_kind;
  std::cout << "near starts: within tolerance " << starts.at(0) << ", on surface " << starts.at(1)
            << ", slightly inside " << starts.at(2) << "; free " << counts.free_paths << ", hit " << counts.hit_paths
            << ", started inside " << counts.started_inside_paths << ", too many hits " << counts.too_many_hits
            << ", second reacting solid " << counts.second_reacting_solid << ", slow approach " << counts.slow_approach
            << ", squeezed " << counts.squeezed << "\n";
  PrintFailuresPerScene(counts.failures_per_scene);
  EXPECT_GT(starts.at(0), 15000);
  EXPECT_GT(starts.at(1), 15000);
  EXPECT_GT(starts.at(2), 15000);
  EXPECT_GT(counts.free_paths, 5000);
  EXPECT_GT(counts.hit_paths, 10000);
  EXPECT_GT(counts.started_inside_paths, 10000);
}

// Builds the inputs a generated set draws around in one scene: no committed plate, the test contact tolerance, and 0.2
// s and {0.5, 0.5, 0.5} for a scene whose step or start is out of range. Says false for the scene the generated sets
// leave out: the one about solids out of id order.
bool BuildGeneratedSceneInputs(const MoveParticleCase& scene, SolidContactInputs* scene_inputs) {
  *scene_inputs = ToSolidContactInputs(scene);
  scene_inputs->plate_side = {};
  scene_inputs->contact_tolerance = kTolerance;
  const bool is_scene_in_range = scene_inputs->step_duration > 0 && math::IsFinite(scene_inputs->start_position);
  if (!is_scene_in_range) {
    scene_inputs->step_duration = 0.2;
    scene_inputs->start_position = {.x = 0.5, .y = 0.5, .z = 0.5};
  }
  return CheckSolidContactInputs(*scene_inputs) == ContactError::kNone;
}

TEST(MoveParticle, GeneratedStartsNearSolidsInEveryScene_EndOutsideEverySolid) {
  // For every scene of BuildMoveParticleCases with a solid, in id order: 1000 starts (BuildGeneratedNearStart) within
  // the tolerance of, on, or slightly inside one solid and far from the others, no committed plate, the test contact
  // tolerance. A scene with a step or a start out of range uses 0.2 s and {0.5, 0.5, 0.5}. Its kTooManyHits come from
  // the three turning-box scenes (every start: no touching rule until plan Task 13) and the octagon room (100 m/s);
  // its kSecondReactingSolid from the scenes with two reacting solids; its kSqueezedBetweenSolids from the scenes whose
  // solids close on the particle.
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
  int one_hit_paths = 0;  // Success with one hit: the start's joint response, or one hit along the path.
  int two_hit_paths = 0;  // Success with two hits.
  int failures = 0;
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

// Runs MoveParticle on one two-surface start, checks it and counts it: it succeeds, and ends at least tol / 2 from
// every solid. Before the slide over several surfaces (plan Task 12) an acute wedge failed with kTooManyHits in 30 % of
// the draws: the response on one face turned the velocity into the other, and the particle hit one and then the other.
// It counts the hits too: since plan decision P20 a start on two surfaces is one hit, and a second comes only along
// the path.
void CheckTwoSurfaceStart(const SolidContactInputs& inputs, TwoSurfaceCounts& counts) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  if (error != ContactError::kNone) {
    ADD_FAILURE() << "error " << static_cast<int>(error) << ", close call " << static_cast<int>(failure.close_call);
    ++counts.failures;
    return;
  }
  ExpectEndOutsideEverySolid(inputs, result.position);
  counts.free_paths += result.kind == ContactKind::kNone ? 1 : 0;
  counts.hit_paths += result.kind == ContactKind::kHit ? 1 : 0;
  counts.started_inside_paths += result.kind == ContactKind::kStartedInsideSolid ? 1 : 0;
  counts.one_hit_paths += result.hit_count == 1 ? 1 : 0;
  counts.two_hit_paths += result.hit_count == 2 ? 1 : 0;
}

// Runs the 30,000 draws of the two-surface set: by Next < 0.5 a wedge (BuildTwoSurfaceWedge, wedge angles 5 to 175
// degrees) or a domain corner (BuildDomainCornerStart); then T = 0.01 + 0.09 * Next s, short enough that a fast
// particle stays near its corner, away from the far sides, both velocities from one DrawTwoSurfaceVelocity, friction
// Next. With moving_wedges, each wedge solid then gets a linear_velocity of three NextSigned m/s, solid A then B; the
// domain box stays fixed. Start overlap tolerance 2 tol. Every start is within tol of, on, or slightly inside two or
// three surfaces.
TwoSurfaceCounts RunTwoSurfaceStarts(bool moving_wedges) {
  CaseGenerator generator;
  TwoSurfaceCounts counts;
  const std::array domain_solids = {move_particle_cases::BuildUnitDomainBox(1)};
  for (int draw_index = 0; draw_index < 30000; ++draw_index) {
    const bool is_wedge = generator.Next() < 0.5;
    SolidContactInputs inputs;
    inputs.contact_tolerance = kTolerance;
    inputs.start_overlap_tolerance = 2 * kTolerance;
    std::array<Solid, 2> wedge_solids{};
    if (is_wedge) {
      wedge_solids = BuildTwoSurfaceWedge(generator, &inputs.start_position);
      inputs.solids = wedge_solids.data();
      inputs.solid_count = 2;
      ++counts.wedge_starts;
      const Mat3 first_axes = wedge_solids.at(0).pose_at_step_start.axes;
      const Mat3 second_axes = wedge_solids.at(1).pose_at_step_start.axes;
      const bool is_acute_wedge =
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
    for (Solid& wedge_solid : wedge_solids) {
      if (moving_wedges && is_wedge) {
        wedge_solid.motion.linear_velocity = {
            .x = generator.NextSigned(),
            .y = generator.NextSigned(),
            .z = generator.NextSigned(),
        };
      }
    }
    CheckTwoSurfaceStart(inputs, counts);
  }
  std::cout << "two-surface starts" << (moving_wedges ? " with moving wedges" : "") << ": wedges "
            << counts.wedge_starts << " (acute " << counts.acute_wedge_starts << "), domain corners "
            << counts.corner_starts << "; free " << counts.free_paths << ", hit " << counts.hit_paths
            << ", started inside " << counts.started_inside_paths << "; one hit " << counts.one_hit_paths
            << ", two hits " << counts.two_hit_paths << ", failures " << counts.failures << "\n";
  return counts;
}

TEST(MoveParticle, GeneratedTwoSurfaceStarts_SucceedDownToFiveDegrees) {
  // Fixed wedges and corners (RunTwoSurfaceStarts): no failure at all (the model: 0 of 10,001 acute draws). Before the
  // slide over several surfaces, 2246 of 7497 acute draws ended in kTooManyHits.
  const TwoSurfaceCounts counts = RunTwoSurfaceStarts(false);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_GT(counts.free_paths, 1000);
  EXPECT_GT(counts.hit_paths, 1000);
  EXPECT_GT(counts.started_inside_paths, 1000);
}

TEST(MoveParticle, GeneratedMovingTwoSurfaceStarts_SucceedDownToFiveDegrees) {
  // The same draws with each wedge solid moving at up to 1 m/s per axis: no failure. Two surfaces that are not
  // parallel always leave a velocity that approaches neither, and from 5 degrees up the escape is at most
  // 1 / sin(2.5 degrees) = 23 times the solids' speeds, below the speed cap (the model: 0 of 35,000).
  const TwoSurfaceCounts counts = RunTwoSurfaceStarts(true);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_GT(counts.hit_paths, 1000);
}

// --- Generated tangent contacts ---

// The scenes of the generated tangent contacts, in the order of the set.
enum class TangentScene : std::uint8_t {
  kBallOnFloor,  // A ball resting on the fixed floor through the contact point.
  kBallInDomainCorner,  // A ball touching the three low sides of the unit domain box.
  kTwoBalls,  // Two balls touching along x.
  kBallOnTiltedFlatSolid,  // A ball resting on a fixed flat solid tilted about y.
  kCylinderLyingOnFloor,  // A cylinder lying on the fixed floor along a line through the contact point.
  kCylinderStandingInDomainCorner,  // A cylinder standing on the unit domain box's bottom, touching its left and front.
  kRimOnFloor,  // A tilted cylinder resting on the fixed floor at one point of its low rim.
};

// One start of the tangent set: its solids, its inputs (which point at them) and what was drawn for the counts.
struct TangentStart {
  std::array<Solid, 2> solids{};
  SolidContactInputs inputs{};
  int velocity_class = 0;  // 0 at rest, 1 slow, 2 falling, 3 toward the contact, 4 random.
  Vec3 first_normal{};  // No unit, world. The floor's, the side's or the first ball's normal at the start.
  Vec3 second_normal{};  // No unit, world. The second ball's normal at the start.
  Vec3 contact{};  // m, world: the contact point.
  Vec3 contact_normal{};  // No unit, world: out of the floor, the side or the first ball at the contact.
  Vec3 radial{};  // No unit, world: away from the contact, square to contact_normal.
  double rest_distance = 0;  // m: x*, the rest point's distance from the contact.
  double first_radius = 0;  // m: the first ball's radius for two balls, else 0.
};

// Computes the gap from a point of a contact plane to a solid, along the plane's normal: the height above the point at
// which the solid begins, at most 2 tol, found by 64 halvings of [0, 2 tol]. The solid is convex and the point outside
// it, so along the normal its distance is first positive, then not.
double ComputeGapToSolid(const Solid& solid, Vec3 point_on_plane, Vec3 normal) {
  const double largest_gap = 2 * kTolerance;  // m
  const auto distance_at_height = [&](double height) {
    return ComputeDistanceToSolid(solid, solid.pose_at_step_start, point_on_plane + height * normal).distance;
  };
  if (distance_at_height(largest_gap) > 0) {
    return largest_gap;
  }
  double outside_height = 0;  // m
  double inside_height = largest_gap;  // m
  for (int halving_index = 0; halving_index < 64; ++halving_index) {
    const double middle_height = (outside_height + inside_height) / 2;  // m
    if (distance_at_height(middle_height) > 0) {
      outside_height = middle_height;
    } else {
      inside_height = middle_height;
    }
  }
  return outside_height;
}

// Builds one start of the tangent set in one scene (P11 generator). Draws, in this order and all of them for every
// scene: R = 0.05 + 0.45 Next and the second ball's radius the same; the contact's offset {NextSigned, NextSigned} m;
// the radial direction's angle 2 pi Next; x = x* (0.02 + 0.98 Next), with x* = 2 Sqrt(R_eff tol) the rest point's
// distance and R_eff = R, or R R2 / (R + R2) for two balls; the height in the gap, (1.5 Next - 0.5) of the gap x^2 /
// (2 R_eff), so some starts are up to half the gap inside the floor (inside the first ball for two balls); T from
// {1e-3, 0.01, 0.1} s by floor(3 Next); the velocity class floor(5 Next), three NextSigned and one Next (at rest; slow,
// the three times tol / T; falling, {0, 0, -9.81 T}; toward the contact along the radial direction, 2 Next m/s; random,
// the three times 2 m/s); the scene's own Next (the corner ball's side by floor(3 Next), bottom, left or front; the
// tilt 0.6 Next rad). Both velocities equal, friction 0, start overlap tolerance 2 tol.
// The cylinder scenes read the same draws, with the cylinder's radius R and half height H = R2 (2 R2 lying), and the
// gap measured along the contact normal to the cylinder (ComputeGapToSolid), at most 2 tol, which is the ball's gap at
// x*:
//   lying on the floor  its axis along {cos, sin, 0} of the radial angle; the radial direction square to it in the
//                       floor, on the side the scene's Next gives (below 0.5, the left of the axis).
//   standing in corner  centre {R, R, H}, its low cap on the bottom; the contact is with the left side at {0, R, H}
//                       (radial +y) or the front side at {R, 0, H} (radial +x), by floor(2 Next) of the scene's Next.
//   rim on the floor    its axis tilted about y by 0.1 + 0.5 Next rad of the scene's Next, its lowest rim point on the
//                       floor at the offset; the radial direction is the radial angle's in the floor. x* is the
//                       nominal 2 Sqrt(R tol): the space under the cap is a wedge of the tilt, not a cusp.
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
    case TangentScene::kCylinderLyingOnFloor: {
      floor_normal = {.x = 0, .y = 0, .z = 1};
      const Vec3 along_axis = {.x = cos_radial, .y = sin_radial, .z = 0};
      const double side_sign = scene_draw < 0.5 ? 1.0 : -1.0;
      radial = side_sign * Vec3{.x = -sin_radial, .y = cos_radial, .z = 0};
      contact = offset;
      const Cylinder lying = {
          .center = contact + radius * floor_normal,
          .axis = along_axis,
          .radius = radius,
          .half_height = 2 * second_radius,
      };
      start.solids.at(0) =
          move_particle_cases::BuildFlatSolid(contact, floor_normal, move_particle_cases::kNoMotion, 1);
      start.solids.at(1) = ToSolid(lying, move_particle_cases::kNoMotion, 2);
      break;
    }
    case TangentScene::kCylinderStandingInDomainCorner: {
      const bool touches_left = std::floor(2 * scene_draw) == 0;
      floor_normal = touches_left ? Vec3{.x = 1, .y = 0, .z = 0} : Vec3{.x = 0, .y = 1, .z = 0};
      radial = touches_left ? Vec3{.x = 0, .y = 1, .z = 0} : Vec3{.x = 1, .y = 0, .z = 0};
      const Vec3 center = {.x = radius, .y = radius, .z = second_radius};
      contact = center - radius * floor_normal;
      const Cylinder standing =
          {.center = center, .axis = {.x = 0, .y = 0, .z = 1}, .radius = radius, .half_height = second_radius};
      start.solids.at(0) = move_particle_cases::BuildUnitDomainBox(1);
      start.solids.at(1) = ToSolid(standing, move_particle_cases::kNoMotion, 2);
      break;
    }
    case TangentScene::kRimOnFloor: {
      const double tilt = 0.1 + 0.5 * scene_draw;  // rad
      floor_normal = {.x = 0, .y = 0, .z = 1};
      radial = {.x = cos_radial, .y = sin_radial, .z = 0};
      contact = offset;
      const Vec3 axis = {.x = std::sin(tilt), .y = 0, .z = std::cos(tilt)};
      // The direction in the cap's plane that goes down most steeply: the lowest rim point is R along it.
      const Vec3 down_in_cap = {.x = std::cos(tilt), .y = 0, .z = -std::sin(tilt)};
      const Cylinder tilted = {
          .center = contact + second_radius * axis - radius * down_in_cap,
          .axis = axis,
          .radius = radius,
          .half_height = second_radius,
      };
      start.solids.at(0) =
          move_particle_cases::BuildFlatSolid(contact, floor_normal, move_particle_cases::kNoMotion, 1);
      start.solids.at(1) = ToSolid(tilted, move_particle_cases::kNoMotion, 2);
      break;
    }
  }
  const double rest_distance = 2 * std::sqrt(effective_radius * kTolerance);  // m, x*.
  const double contact_distance = distance_share * rest_distance;  // m
  const bool is_cylinder_scene = scene == TangentScene::kCylinderLyingOnFloor
      || scene == TangentScene::kCylinderStandingInDomainCorner || scene == TangentScene::kRimOnFloor;
  const double gap = is_cylinder_scene
      ? ComputeGapToSolid(start.solids.at(1), contact + contact_distance * radial, floor_normal)
      : contact_distance * contact_distance / (2 * effective_radius);  // m
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
  start.contact = contact;
  start.contact_normal = floor_normal;
  start.radial = radial;
  start.rest_distance = rest_distance;
  start.first_radius = scene == TangentScene::kTwoBalls ? radius : 0;
  return start;
}

// Runs MoveParticle on one tangent start and fails the calling test unless it succeeded at least tol / 2 from every
// solid. Before the slide over several surfaces (plan Task 12) 37 to 49 % per scene ended in kTooManyHits, bouncing
// between the two surfaces into the cusp.
bool ExpectTangentStartSucceeds(TangentScene scene, const SolidContactInputs& inputs) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  if (error != ContactError::kNone) {
    ADD_FAILURE() << "scene " << static_cast<int>(scene) << ", error " << static_cast<int>(error) << ", close call "
                  << static_cast<int>(failure.close_call);
    return false;
  }
  ExpectEndOutsideEverySolid(inputs, result.position);
  return true;
}

TEST(MoveParticle, GeneratedTangentContactStarts_Succeed) {
  // 2000 starts per scene (BuildTangentStart), the scenes in TangentScene order, starts in the cusp of each tangent
  // contact from 0.02 to 1 times the rest point's distance. Every start succeeds and ends at least tol / 2 from every
  // solid.
  CaseGenerator generator;
  const std::array scenes = {
      TangentScene::kBallOnFloor,
      TangentScene::kBallInDomainCorner,
      TangentScene::kTwoBalls,
      TangentScene::kBallOnTiltedFlatSolid,
  };
  for (const TangentScene scene : scenes) {
    int successes = 0;
    for (int draw_index = 0; draw_index < 2000; ++draw_index) {
      TangentStart start = BuildTangentStart(generator, scene);
      start.inputs.solids = start.solids.data();
      successes += ExpectTangentStartSucceeds(scene, start.inputs) ? 1 : 0;
    }
    std::cout << "tangent scene " << static_cast<int>(scene) << ": success " << successes << " of 2000\n";
  }
}

// Turns one tangent start into a slide by Next < 1/3: the particle starts tol from the floor (for two balls, from the
// first ball), at x* + (0.05 - x*) Next from the contact along the radial direction, moving toward the contact at
// 2 Next m/s with gravity, {0, 0, -9.81 T} m/s. Otherwise it keeps the start; the two Next are drawn either way.
void ApplyTangentSlide(CaseGenerator& generator, TangentStart* start) {
  const bool is_slide = generator.Next() < 1.0 / 3;
  const double distance_share = generator.Next();
  const double speed_share = generator.Next();
  if (!is_slide) {
    return;
  }
  const double contact_distance = start->rest_distance + (0.05 - start->rest_distance) * distance_share;  // m
  const double first_surface_height =
      start->first_radius > 0 ? -contact_distance * contact_distance / (2 * start->first_radius) : 0;  // m
  start->inputs.start_position =
      start->contact + contact_distance * start->radial + (first_surface_height + kTolerance) * start->contact_normal;
  const Vec3 velocity =
      -2 * speed_share * start->radial + Vec3{.x = 0, .y = 0, .z = -9.81 * start->inputs.step_duration};
  start->inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
}

// Sets the motion of one tangent start's solids, each sliding along the other: the first solid (the floor, the tilted
// side or the first ball) at a velocity square to the contact normal, and the ball or the cylinder at that velocity
// plus another square to it and a part along the contact normal of 0, 0, -1e-3 or 1e-3 m/s (sinking or rising) by
// floor(4 Next). The tangent parts are NextSigned m/s along two directions square to the contact normal (radial and its
// cross with the normal), four draws, then the one for the normal part. The ball and the cylinder in the domain corner
// keep still: moving, they would leave or enter the other sides. The draws are made in every scene.
void ApplyTangentMotion(CaseGenerator& generator, TangentScene scene, TangentStart* start) {
  const Vec3 normal = start->contact_normal;
  const Vec3 along = start->radial;
  const Vec3 across = math::Cross(normal, along);
  const Vec3 first_velocity = generator.NextSigned() * along + generator.NextSigned() * across;  // m/s
  const Vec3 slide_velocity = generator.NextSigned() * along + generator.NextSigned() * across;  // m/s
  const std::array<double, 4> normal_speeds = {0, 0, -1e-3, 1e-3};  // m/s
  const double normal_speed = normal_speeds.at(static_cast<std::size_t>(std::floor(4 * generator.Next())));
  if (scene == TangentScene::kBallInDomainCorner || scene == TangentScene::kCylinderStandingInDomainCorner) {
    return;
  }
  start->solids.at(0).motion.linear_velocity = first_velocity;
  start->solids.at(1).motion.linear_velocity = first_velocity + slide_velocity + normal_speed * normal;
}

// What the tangent slides of one scene and motion gave, counted for the report.
struct TangentSlideCounts {
  int successes = 0;
  int too_many_hits = 0;
  // kTooManyHits by the number of halvings of the step after which it first succeeded, 1 to 8; the last, none.
  std::array<int, 9> cured_after_halvings{};
  int squeezed = 0;  // kSqueezedBetweenSolids, only where the ball closes on the other solid.
  int other_failures = 0;
};

// Runs MoveParticle on one tangent slide and counts it: a success ends at least tol / 2 from every solid. A failure may
// only be kTooManyHits, run again with the step halved up to eight times (1/256) until one succeeds, or
// kSqueezedBetweenSolids where the ball closes on the other solid (a ball sinking into a floor): the speed cap's
// squeeze, which the geometry decides and a shorter step does not cure (plan decision P19, item 6).
void CheckTangentSlide(TangentScene scene, const TangentStart& start, TangentSlideCounts& counts) {
  SolidContactInputs inputs = start.inputs;
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  if (error == ContactError::kNone) {
    ExpectEndOutsideEverySolid(inputs, result.position);
    ++counts.successes;
    return;
  }
  const bool is_close_call = error == ContactError::kTooCloseToCall;
  const Vec3 closing_velocity = start.solids.at(1).motion.linear_velocity - start.solids.at(0).motion.linear_velocity;
  const bool is_ball_closing = math::Dot(closing_velocity, start.contact_normal) < 0;
  if (is_close_call && failure.close_call == ContactCloseCall::kSqueezedBetweenSolids && is_ball_closing) {
    ++counts.squeezed;
    return;
  }
  if (!is_close_call || failure.close_call != ContactCloseCall::kTooManyHits) {
    ADD_FAILURE() << "scene " << static_cast<int>(scene) << ", error " << static_cast<int>(error) << ", close call "
                  << static_cast<int>(failure.close_call);
    ++counts.other_failures;
    return;
  }
  ++counts.too_many_hits;
  int halving_count = 1;
  for (; halving_count <= 8; ++halving_count) {
    inputs.step_duration = inputs.step_duration / 2;
    if (MoveParticle(inputs, &result, &failure) == ContactError::kNone) {
      break;
    }
  }
  ++counts.cured_after_halvings.at(static_cast<std::size_t>(halving_count - 1));
}

// Prints, for the report, what the tangent slides of one scene and motion gave.
void PrintTangentSlideCounts(TangentScene scene, bool is_moving, const TangentSlideCounts& counts) {
  const std::array<int, 9>& cured = counts.cured_after_halvings;
  std::cout << "tangent slides scene " << static_cast<int>(scene) << (is_moving ? " moving" : " fixed") << ": success "
            << counts.successes << ", too many hits " << counts.too_many_hits << " (cured after 1 to 8 halvings:";
  for (std::size_t halving_index = 0; halving_index < 8; ++halving_index) {
    std::cout << " " << cured.at(halving_index);
  }
  std::cout << "; not cured " << cured.at(8) << "), squeezed " << counts.squeezed << ", other failures "
            << counts.other_failures << "\n";
}

// Runs 2000 draws of the tangent slides of one scene: BuildTangentStart, then ApplyTangentSlide, then
// ApplyTangentMotion, applied only when is_moving, and CheckTangentSlide.
TangentSlideCounts RunTangentSlides(CaseGenerator& generator, TangentScene scene, bool is_moving) {
  TangentSlideCounts counts;
  for (int draw_index = 0; draw_index < 2000; ++draw_index) {
    TangentStart start = BuildTangentStart(generator, scene);
    ApplyTangentSlide(generator, &start);
    TangentStart moving_start = start;
    ApplyTangentMotion(generator, scene, &moving_start);
    TangentStart& used = is_moving ? moving_start : start;
    used.inputs.solids = used.solids.data();
    CheckTangentSlide(scene, used, counts);
  }
  return counts;
}

// Fails the calling test unless the slides of one scene had no other failure, every kTooManyHits was cured within
// eight halvings, and at least 1950 of the 2000 draws succeeded (1958 in the worst scene), so the checks are not
// vacuous.
void ExpectTangentSlideCounts(const TangentSlideCounts& counts) {
  EXPECT_EQ(counts.other_failures, 0);
  EXPECT_EQ(counts.cured_after_halvings.at(8), 0);
  EXPECT_GE(counts.successes, 1950);
}

TEST(MoveParticle, GeneratedTangentContactSlides_FailOnlyWhereAShorterStepCures) {
  // 2000 draws per scene, in TangentScene order, fixed and then moving: BuildTangentStart, then ApplyTangentSlide, then
  // ApplyTangentMotion (drawn for the fixed ones too, not applied). A third of the draws slide into the cusp from up to
  // 5 cm at up to 2 m/s with gravity; the others start in it. Every success ends at least tol / 2 from every solid; the
  // only failures are kTooManyHits, each cured by a step at most 256 times shorter, and the speed cap's squeeze of a
  // ball sinking into a floor (CheckTangentSlide). Measured: two balls are the only kTooManyHits, a fast particle
  // zig-zagging into their neck, 42 of 2000 fixed draws (steps of 0.1 s moving it up to 28 cm), 38 cured by 1/16 and
  // the others by 1/32 to 1/256; the model's set, with other draws, cured all 31 of 40,000 by 1/16. The 256 holds for
  // these draws only: a rule that responds hit by hit, four hits a step, reaches the neck's pair only once both balls
  // are within tol at once, so the halvings needed grow with the speed over Sqrt(R tol), and faster drops or other
  // draws can need more.
  CaseGenerator generator;
  const std::array scenes = {
      TangentScene::kBallOnFloor,
      TangentScene::kBallInDomainCorner,
      TangentScene::kTwoBalls,
      TangentScene::kBallOnTiltedFlatSolid,
  };
  for (const bool is_moving : {false, true}) {
    for (const TangentScene scene : scenes) {
      const TangentSlideCounts counts = RunTangentSlides(generator, scene, is_moving);
      PrintTangentSlideCounts(scene, is_moving, counts);
      ExpectTangentSlideCounts(counts);
    }
  }
}

TEST(MoveParticle, GeneratedCylinderTangentContactStarts_Succeed) {
  // The starts of GeneratedTangentContactStarts_Succeed in the cylinder scenes, 2000 per scene in TangentScene order,
  // from a generator of their own, so the ball scenes keep their draws: a cylinder lying on a floor, standing in a
  // domain corner, and resting on its rim on a floor. Every start succeeds and ends at least tol / 2 from every solid.
  CaseGenerator generator;
  const std::array scenes = {
      TangentScene::kCylinderLyingOnFloor,
      TangentScene::kCylinderStandingInDomainCorner,
      TangentScene::kRimOnFloor,
  };
  for (const TangentScene scene : scenes) {
    int successes = 0;
    for (int draw_index = 0; draw_index < 2000; ++draw_index) {
      TangentStart start = BuildTangentStart(generator, scene);
      start.inputs.solids = start.solids.data();
      successes += ExpectTangentStartSucceeds(scene, start.inputs) ? 1 : 0;
    }
    std::cout << "cylinder tangent scene " << static_cast<int>(scene) << ": success " << successes << " of 2000\n";
    EXPECT_EQ(successes, 2000);
  }
}

TEST(MoveParticle, GeneratedCylinderTangentContactSlides_FailOnlyWhereAShorterStepCures) {
  // The slides of GeneratedTangentContactSlides_FailOnlyWhereAShorterStepCures in the cylinder scenes, 2000 draws per
  // scene, fixed and then moving, from a generator of their own. The cylinder in the domain corner keeps still.
  CaseGenerator generator;
  const std::array scenes = {
      TangentScene::kCylinderLyingOnFloor,
      TangentScene::kCylinderStandingInDomainCorner,
      TangentScene::kRimOnFloor,
  };
  for (const bool is_moving : {false, true}) {
    for (const TangentScene scene : scenes) {
      const TangentSlideCounts counts = RunTangentSlides(generator, scene, is_moving);
      PrintTangentSlideCounts(scene, is_moving, counts);
      ExpectTangentSlideCounts(counts);
    }
  }
}

// --- The touched rule in the sweep ---

// The sweep that follows the first hit of the scene ObliqueFlatSolidRoundingAtHit: the hit position and the response's
// velocity, as MoveParticle computes them, with the allowance of that response and the hit's normal.
struct SweepAfterObliqueHit {
  MoveParticleCase scene;
  Sweep sweep;
  TouchedSolid touched;
  Vec3 normal_out_of_solid{};
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
  after_hit.touched = {
      .is_touched = true,
      .approach_allowance = ComputeLargestApproachLeftAfterResponse(velocity, {}),
  };
  after_hit.normal_out_of_solid = normal;
  return after_hit;
}

TEST(FindFirstHitOfSweep, NotTouchedAfterRoundedHit_HitsSameSolidAgain) {
  // The reason for the touched rule: without it the start, 2.8e-18 m outside the rest gap, is outside, and the
  // rounding-sized approach of the response makes the closed form hit the same solid again.
  const SweepAfterObliqueHit after_hit = BuildSweepAfterObliqueHit();
  FirstHit first_hit;
  const TouchedSolid not_touched;
  const ContactError error =
      FindFirstHitOfSweep(ToSolidContactInputs(after_hit.scene), after_hit.sweep, not_touched, &first_hit, nullptr);
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_EQ(first_hit.solid_index, 0U);
}

TEST(FindFirstHitOfSweep, TouchedSolidAfterRoundedHit_IsFree) {
  const SweepAfterObliqueHit after_hit = BuildSweepAfterObliqueHit();
  FirstHit first_hit;
  const ContactError error = FindFirstHitOfSweep(
      ToSolidContactInputs(after_hit.scene),
      after_hit.sweep,
      after_hit.touched,
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
      after_hit.sweep.velocity_for_this_step - 1e-9 * after_hit.normal_out_of_solid;
  FirstHit first_hit;
  const ContactError error = FindFirstHitOfSweep(
      ToSolidContactInputs(after_hit.scene),
      after_hit.sweep,
      after_hit.touched,
      &first_hit,
      nullptr
  );
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
}

TEST(FindFirstHitOfSweep, TouchedSolidFartherThanSixteenthOfTolerance_TakesClosedForm) {
  // A start tol / 2 above the rest gap of the solid hit, approaching at 1 m/s: outside the touched window of tol / 16,
  // so the closed form finds the hit tol / 2 later, not at 0.
  SweepAfterObliqueHit after_hit = BuildSweepAfterObliqueHit();
  const Vec3 normal = after_hit.normal_out_of_solid;
  after_hit.sweep.start = after_hit.sweep.start + 0.5 * kTolerance * normal;
  after_hit.sweep.velocity_for_this_step = -1.0 * normal;
  FirstHit first_hit;
  const ContactError error = FindFirstHitOfSweep(
      ToSolidContactInputs(after_hit.scene),
      after_hit.sweep,
      after_hit.touched,
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
  // {0.2 * NextSigned, 0.2 * NextSigned, -1 - Next} m/s, T 2 s, friction 0 (draws in that order). A rim hit leaves
  // the particle within tol / 1024 of the rest gap, inside the touched window, where the re-sweep's touching rule reads
  // the approach the response left; the cylinder is convex, so the tangent path cannot meet it again.
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

// One path of the round set: its one sphere or cylinder and its inputs. inputs.solids is left for the caller.
struct RoundSolidPath {
  std::array<Solid, 1> solids{};
  SolidContactInputs inputs{};
};

// Builds one path of the round set: a sphere or a cylinder (Next < 0.5 a sphere) at the origin, radius 0.1 + 0.4 Next
// m, half height 0.1 + 0.9 Next m, its axis three NextSigned (drawn again while shorter than 0.1); T = 0.01 + 0.99 Next
// s; moving when Next < 0.5 at three NextSigned m/s; spinning about its axis at 20 NextSigned rad/s, plus a turn across
// the axis of Next times 0.99 of plan decision P17's bound, tol / (8 T Length(half_size)) rad/s, about the cross of the
// axis with three NextSigned (drawn again while shorter than 0.1); a start on the direction of three NextSigned (drawn
// again while shorter than 0.1), Length(half_size) + 0.5 m from the centre; three NextSigned for the target and one
// Next for its kind; a velocity relative to the solid that reaches the target after (0.3 + 0.6 Next) T, both velocities
// equal; friction Next. Draws in that order, all of them for both shapes. The target is 0.8 Length(half_size) times the
// three NextSigned from the centre, or, for a cylinder when the kind's Next is below 0.3, a point within tol of the
// grown cylinder's rim edge, in its own axes {(R + tol + tol a) cos(pi b), (R + tol + tol a) sin(pi b), sign(c) (H +
// tol + tol c)} with a, b, c the three NextSigned, so the path may enter through the rim corner. The cylinder never
// counts as turning; for a sphere the whole angular velocity is a spin.
RoundSolidPath BuildRoundSolidPath(CaseGenerator& generator) {
  const bool is_sphere = generator.Next() < 0.5;
  const double radius = 0.1 + 0.4 * generator.Next();  // m
  const double half_height = 0.1 + 0.9 * generator.Next();  // m
  const auto draw_direction = [&generator] {
    Vec3 direction{};
    while (math::Length(direction) < 0.1) {
      direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    }
    return math::Normalized(direction);
  };
  const Vec3 axis = draw_direction();
  const double step_duration = 0.01 + 0.99 * generator.Next();  // s
  const bool is_moving = generator.Next() < 0.5;
  const Vec3 linear_draw = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const double spin_rate = 20 * generator.NextSigned();  // rad/s
  const double tilt_share = generator.Next();
  Vec3 across_axis{};
  while (math::Length(across_axis) < 0.1) {
    across_axis = math::Cross(axis, draw_direction());
  }
  const Vec3 start_direction = draw_direction();
  const Vec3 target_draw = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const bool is_rim_aim = !is_sphere && generator.Next() < 0.3;
  const double reach_share = 0.3 + 0.6 * generator.Next();
  const double friction = generator.Next();

  const Vec3 half_size =
      is_sphere ? Vec3{.x = radius, .y = radius, .z = radius} : Vec3{.x = radius, .y = radius, .z = half_height};
  const double reach = math::Length(half_size);  // m
  const double largest_tilt_rate = kTolerance / (8 * step_duration * reach);  // rad/s
  const Vec3 linear_velocity = is_moving ? linear_draw : Vec3{};  // m/s
  const SolidMotion motion = {
      .linear_velocity = linear_velocity,
      .angular_velocity = spin_rate * axis + 0.99 * tilt_share * largest_tilt_rate * math::Normalized(across_axis),
  };
  RoundSolidPath path;
  path.solids.at(0) = is_sphere
      ? ToSolid(Sphere{.center = {}, .radius = radius}, motion, 1)
      : ToSolid(Cylinder{.center = {}, .axis = axis, .radius = radius, .half_height = half_height}, motion, 1);
  path.inputs.solid_count = 1;
  path.inputs.contact_tolerance = kTolerance;
  path.inputs.step_duration = step_duration;
  path.inputs.start_position = (reach + 0.5) * start_direction;
  // m, from the solid's centre at the time it is reached.
  Vec3 target = 0.8 * reach * target_draw;
  if (is_rim_aim) {
    const double rim_radius = radius + kTolerance + kTolerance * target_draw.x;  // m
    const double angle = std::numbers::pi * target_draw.y;  // rad
    const double cap_sign = target_draw.z < 0 ? -1.0 : 1.0;
    const Vec3 own_target = {
        .x = rim_radius * std::cos(angle),
        .y = rim_radius * std::sin(angle),
        .z = cap_sign * (half_height + kTolerance + kTolerance * target_draw.z),
    };
    target = path.solids.at(0).pose_at_step_start.axes * own_target;
  }
  const Vec3 velocity =
      linear_velocity + (1 / (reach_share * step_duration)) * (target - path.inputs.start_position);  // m/s
  path.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  path.inputs.friction = friction;
  return path;
}

// What the round set gave, counted for the checks and the report.
struct RoundSolidCounts {
  int hit_paths = 0;
  int rim_hit_paths = 0;  // The first hit on a rim.
  int repeat_hits = 0;  // More than one hit.
  int failures = 0;
};

// Runs MoveParticle on one path of the round set and counts it; a failure fails the calling test.
void CheckRoundSolidPath(const RoundSolidPath& path, RoundSolidCounts& counts) {
  EXPECT_FALSE(ComputeIsTurningSolid(path.inputs, 0));
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(path.inputs, &result, &failure);
  if (error != ContactError::kNone) {
    ADD_FAILURE() << "error " << static_cast<int>(error) << ", close call " << static_cast<int>(failure.close_call);
    ++counts.failures;
    return;
  }
  ExpectEndOutsideEverySolid(path.inputs, result.position);
  const bool is_hit = result.hit_count > 0;
  const bool is_rim = result.face == SolidFace::kLowRim || result.face == SolidFace::kHighRim;
  counts.hit_paths += is_hit ? 1 : 0;
  counts.rim_hit_paths += is_hit && is_rim ? 1 : 0;
  counts.repeat_hits += result.hit_count > 1 ? 1 : 0;
}

TEST(MoveParticle, GeneratedRoundSolidPaths_NeverHitAgain) {
  // 20,000 paths (BuildRoundSolidPath) onto one sphere or cylinder, spinning, moving, and for a cylinder tilting within
  // plan decision P17's bound, with friction. After the response at the hit, the path is tangent to the solid or
  // leaves it, measured from the surface's velocity there, and the solid is convex, so a second hit can only come from
  // reading that approach differently: none is allowed. The touching rule now reads the distance, normal and solid
  // velocity a joint response reads, by the same expressions, and a rim hit is where the rim search found the true
  // distance within tol / 1024 of tol. Before, with the rate measured from the linear velocity alone, 2757 paths onto
  // cylinders ended in kTooManyHits (2755 of them only with the tilt), and 144 rim hits, left in the rim corner a hair
  // outside the grown edge, were hit again. With rim hits moved to the rest gap (plan Task 12a) 10,490 paths hit, 548
  // on a rim; the rim search lets 207 of those pass, which the true distance never brings within tol (plan decision
  // P21).
  CaseGenerator generator;
  RoundSolidCounts counts;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    RoundSolidPath path = BuildRoundSolidPath(generator);
    path.inputs.solids = path.solids.data();
    CheckRoundSolidPath(path, counts);
  }
  std::cout << "round paths 20000, hit " << counts.hit_paths << ", rim hits " << counts.rim_hit_paths
            << ", repeat hits " << counts.repeat_hits << ", failures " << counts.failures << "\n";
  EXPECT_EQ(counts.repeat_hits, 0);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_GT(counts.hit_paths, 9000);  // Measured: 10,283.
  EXPECT_GT(counts.rim_hit_paths, 250);  // Measured: 341.
}

// One path across a flush seam: two solids, the particle's inputs, and the direction along the path.
struct FlushSeamPath {
  std::array<Solid, 2> solids{};
  SolidContactInputs inputs{};
  Vec3 along_path{};  // No unit, unit vector, world: the direction of the velocity relative to the solids.
  Vec3 solid_velocity{};  // m/s, world: both solids move with it.
};

// The kinds of flush seam the generated set crosses.
enum class FlushSeamKind : std::uint8_t {
  kStackedDrums,  // Two drums of one radius on one axis, down the seam along the wall.
  kPuckInFloor,  // A puck flush in a floor, onto it or off it along the floor.
};

// A turn whose three columns have no zero entry: {0.36, 0.48, -0.8}, {-0.8, 0.6, 0}, {0.48, 0.64, 0.6}.
constexpr Mat3 kSeamTiltedAxes = {
    .xx = 0.36,
    .xy = -0.8,
    .xz = 0.48,
    .yx = 0.48,
    .yy = 0.6,
    .yz = 0.64,
    .zx = -0.8,
    .zy = 0,
    .zz = 0.6,
};

// Draws a unit vector from three NextSigned, drawn again while shorter than 0.1.
Vec3 DrawSeamDirection(CaseGenerator& generator) {
  Vec3 direction{};
  while (math::Length(direction) < 0.1) {
    direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  return math::Normalized(direction);
}

// Builds one path across a flush seam. Draws, in this order, for every kind: the axes, kSeamTiltedAxes when Next < 0.5,
// else the world axes; a centre 2 Next times a direction (DrawSeamDirection); the solids' velocity, a direction when
// Next < 1 / 3, else zero; a first size 0.1 + 0.9 Next m; a second 0.1 + 0.9 Next m; a third 0.05 + 0.45 Next m; an
// azimuth 2 pi Next; the speed 0.5 + 1.5 Next m/s; the push into the surface, 0, 0, 0.16 or 1 m/s by floor(4 Next); the
// distance before the seam 0.01 + 0.14 Next m; a sideways share -0.3 + 0.6 Next; onto or off the puck, Next < 0.5; the
// travel 0.16 + 0.14 Next m, which sets T. Stacked drums: radius the first size, half heights the second and the
// third, the upper drum's low cap on the lower drum's high cap; the start at the upper drum's wall rest gap at the
// azimuth, the distance before the seam; the path down the axis, pushed into the wall. Puck: radius the first size,
// half height the third, its top flush with the floor; the start at the floor's rest gap, the distance before the rim,
// outside it going on or inside it going off, with the sideways share across the radius; pushed into the floor.
FlushSeamPath BuildFlushSeamPath(CaseGenerator& generator, FlushSeamKind kind) {
  const bool is_tilted = generator.Next() < 0.5;
  const Vec3 center = 2 * generator.Next() * DrawSeamDirection(generator);
  const bool is_moving = generator.Next() < 1.0 / 3;
  const Vec3 moving_direction = DrawSeamDirection(generator);
  const double first_size = 0.1 + 0.9 * generator.Next();  // m
  const double second_size = 0.1 + 0.9 * generator.Next();  // m
  const double third_size = 0.05 + 0.45 * generator.Next();  // m
  const double azimuth = 2 * std::numbers::pi * generator.Next();  // rad
  const double speed = 0.5 + 1.5 * generator.Next();  // m/s
  constexpr std::array<double, 4> kPushes = {0, 0, 0.16, 1};  // m/s
  const double push = kPushes.at(static_cast<std::size_t>(std::floor(4 * generator.Next())));  // m/s
  const double before_seam = 0.01 + 0.14 * generator.Next();  // m
  const double sideways_share = -0.3 + 0.6 * generator.Next();
  const bool is_onto_puck = generator.Next() < 0.5;
  const double travel = 0.16 + 0.14 * generator.Next();  // m

  const Mat3 axes = is_tilted ? kSeamTiltedAxes : kIdentity;
  const Vec3 axis = {.x = axes.xz, .y = axes.yz, .z = axes.zz};
  const Vec3 solid_velocity = is_moving ? moving_direction : Vec3{};  // m/s
  const SolidMotion motion = {.linear_velocity = solid_velocity, .angular_velocity = {}};
  FlushSeamPath path;
  path.solid_velocity = solid_velocity;
  Vec3 relative_velocity{};  // m/s, along the surface, before the push.
  Vec3 push_direction{};  // No unit: into the surface the particle rests on.
  if (kind == FlushSeamKind::kStackedDrums) {
    const double radius = first_size;  // m
    const Cylinder upper = {.center = center, .axis = axis, .radius = radius, .half_height = second_size};
    const Vec3 lower_center = center - (second_size + third_size) * axis;
    const Cylinder lower = {.center = lower_center, .axis = axis, .radius = radius, .half_height = third_size};
    path.solids = {ToSolid(upper, motion, 1), ToSolid(lower, motion, 2)};
    const Mat3 own_axes = path.solids.at(0).pose_at_step_start.axes;
    const Vec3 radial = std::cos(azimuth) * Vec3{.x = own_axes.xx, .y = own_axes.yx, .z = own_axes.zx}
        + std::sin(azimuth) * Vec3{.x = own_axes.xy, .y = own_axes.yy, .z = own_axes.zy};
    path.inputs.start_position = center + (radius + kTolerance) * radial + (-second_size + before_seam) * axis;
    relative_velocity = -speed * axis;
    push_direction = -radial;
  } else {
    const double radius = first_size;  // m
    const Cylinder puck =
        {.center = center - third_size * axis, .axis = axis, .radius = radius, .half_height = third_size};
    const FlatSolid floor = {.point_on_surface = center, .normal_out_of_solid = axis};
    path.solids = {ToSolid(floor, motion, 1), ToSolid(puck, motion, 2)};
    const Mat3 own_axes = path.solids.at(1).pose_at_step_start.axes;
    const Vec3 radial = std::cos(azimuth) * Vec3{.x = own_axes.xx, .y = own_axes.yx, .z = own_axes.zx}
        + std::sin(azimuth) * Vec3{.x = own_axes.xy, .y = own_axes.yy, .z = own_axes.zy};
    const Vec3 across = math::Cross(axis, radial);
    const double start_radius = is_onto_puck ? radius + before_seam : radius - before_seam;  // m
    const double radial_sign = is_onto_puck ? -1.0 : 1.0;
    path.inputs.start_position = center + start_radius * radial + kTolerance * axis;
    relative_velocity = speed * (radial_sign * radial + sideways_share * across);
    push_direction = -axis;
  }
  const Vec3 velocity = solid_velocity + relative_velocity + push * push_direction;  // m/s
  path.along_path = math::Normalized(relative_velocity);
  path.inputs.solid_count = 2;
  path.inputs.contact_tolerance = kTolerance;
  path.inputs.step_duration = travel / speed;
  path.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  return path;
}

// What one kind of flush seam gave over its paths.
struct FlushSeamCounts {
  int paths = 0;
  int failures = 0;
  int several_hits = 0;
  int lost_speed = 0;  // Paths whose speed along the path, measured from the solids, fell below 1 - 1e-6 of its start.
};

// Runs MoveParticle on one path across a flush seam and counts it.
void CheckFlushSeamPath(const FlushSeamPath& path, FlushSeamCounts& counts) {
  ++counts.paths;
  SolidContactResult result;
  ContactFailure failure;
  if (MoveParticle(path.inputs, &result, &failure) != ContactError::kNone) {
    ++counts.failures;
    return;
  }
  counts.several_hits += result.hit_count > 1 ? 1 : 0;
  const Vec3 start_relative = path.inputs.velocity.for_this_step - path.solid_velocity;  // m/s
  const double start_speed = math::Dot(start_relative, path.along_path);  // m/s
  const double this_step_speed = math::Dot(result.velocity.for_this_step - path.solid_velocity, path.along_path);
  const double next_step_speed = math::Dot(result.velocity.for_next_step - path.solid_velocity, path.along_path);
  const double least_kept_speed = (1 - 1e-6) * start_speed;  // m/s
  const bool keeps_speed = this_step_speed >= least_kept_speed && next_step_speed >= least_kept_speed;
  counts.lost_speed += keeps_speed ? 0 : 1;
}

// Prints one kind's counts for the report and fails the calling test unless all 2,000 paths succeeded with at most one
// hit and kept their speed.
void ExpectFlushSeamCounts(FlushSeamKind kind, const FlushSeamCounts& counts) {
  std::cout << "flush seams kind " << static_cast<int>(kind) << ": paths " << counts.paths << ", failures "
            << counts.failures << ", several hits " << counts.several_hits << ", lost speed " << counts.lost_speed
            << "\n";
  EXPECT_EQ(counts.paths, 2000);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_EQ(counts.several_hits, 0);
  EXPECT_EQ(counts.lost_speed, 0);
}

TEST(MoveParticle, GeneratedFlushSeams_KeepTheSpeedAlongTheSeam) {
  // 2,000 paths across each kind of flush seam (BuildFlushSeamPath), friction 0: a seam is no step, so the particle
  // crosses it without a hit, or with the start's one when it is pushed into the surface, and keeps its speed along the
  // path. Measured with the sharp grown rim edge of plan Task 12a: 1,044 of the stacked drums and 629 of the pucks lost
  // speed, 578 and 360 with a second hit; plan decision P21's model, on its own draws, loses none.
  for (const FlushSeamKind kind : {FlushSeamKind::kStackedDrums, FlushSeamKind::kPuckInFloor}) {
    CaseGenerator generator;
    FlushSeamCounts counts;
    for (int path_index = 0; path_index < 2000; ++path_index) {
      FlushSeamPath path = BuildFlushSeamPath(generator, kind);
      path.inputs.solids = path.solids.data();
      CheckFlushSeamPath(path, counts);
    }
    ExpectFlushSeamCounts(kind, counts);
  }
}

// Moves a flush-seam path's start to offset before its seam, along the line BuildFlushSeamPath measured its distance
// before the seam on: the drums' axis, the puck's radius.
void ApplySeamOffset(FlushSeamKind kind, double offset, FlushSeamPath* path) {
  const Vec3 start = path->inputs.start_position;
  if (kind == FlushSeamKind::kStackedDrums) {
    const Solid& upper = path->solids.at(0);
    const Vec3 axis = {
        .x = upper.pose_at_step_start.axes.xz,
        .y = upper.pose_at_step_start.axes.yz,
        .z = upper.pose_at_step_start.axes.zz,
    };
    const double height = math::Dot(start - upper.pose_at_step_start.origin, axis);  // m
    path->inputs.start_position = start + ((-upper.half_size.z + offset) - height) * axis;
    return;
  }
  const Solid& puck = path->solids.at(1);
  const Vec3 axis = {
      .x = puck.pose_at_step_start.axes.xz,
      .y = puck.pose_at_step_start.axes.yz,
      .z = puck.pose_at_step_start.axes.zz,
  };
  const Vec3 from_center = start - puck.pose_at_step_start.origin;  // m
  const Vec3 across = from_center - math::Dot(from_center, axis) * axis;  // m
  const double axis_distance = math::Length(across);  // m
  const double radius = puck.half_size.x;  // m
  const double new_distance = axis_distance > radius ? radius + offset : radius - offset;  // m
  path->inputs.start_position = start + ((new_distance - axis_distance) / axis_distance) * across;
}

// What the runs across a seam in many steps gave for one kind.
struct SeamRunCounts {
  int runs = 0;
  int failures = 0;  // Runs with a step that did not succeed.
  int lost_speed = 0;  // Runs with a step that lost speed along the path.
};

// Runs one particle across a flush seam in many steps (GeneratedFlushSeamsOverManySteps_KeepTheSpeed) and counts it.
// Draws, after BuildFlushSeamPath's: the start 1.5 + 4.5 Next tol before the seam; the step's length 2^(-2 + 12 Next)
// tol along the path; friction 0 when Next < 2 / 3, else 0.3.
void CheckSeamRun(CaseGenerator& generator, FlushSeamKind kind, SeamRunCounts& counts) {
  FlushSeamPath path = BuildFlushSeamPath(generator, kind);
  const double offset = (1.5 + 4.5 * generator.Next()) * kTolerance;  // m
  const double step_length = std::exp2(-2 + 12 * generator.Next()) * kTolerance;  // m
  const double friction = generator.Next() < 2.0 / 3 ? 0.0 : 0.3;
  ApplySeamOffset(kind, offset, &path);
  const Vec3 relative_velocity = path.inputs.velocity.for_this_step - path.solid_velocity;  // m/s
  const double start_speed = math::Dot(relative_velocity, path.along_path);  // m/s
  const double push = math::Length(relative_velocity - start_speed * path.along_path);  // m/s
  const double least_kept_speed = (1 - 1e-6) * (start_speed - friction * push);  // m/s
  const double step_duration = step_length / start_speed;  // s
  const int step_count = std::min(60, static_cast<int>(std::floor(14 * kTolerance / step_length)) + 2);
  path.inputs.step_duration = step_duration;
  path.inputs.friction = friction;
  ++counts.runs;
  bool is_failure = false;
  bool is_slower = false;
  for (int step_index = 0; step_index < step_count && !is_failure; ++step_index) {
    path.inputs.solids = path.solids.data();
    SolidContactResult result;
    ContactFailure failure;
    is_failure = MoveParticle(path.inputs, &result, &failure) != ContactError::kNone;
    const double speed = math::Dot(result.velocity.for_this_step - path.solid_velocity, path.along_path);  // m/s
    is_slower = is_slower || (!is_failure && speed < least_kept_speed);
    path.inputs.start_position = result.position;
    for (Solid& solid : path.solids) {
      solid.pose_at_step_start.origin = solid.pose_at_step_start.origin + step_duration * path.solid_velocity;
    }
  }
  counts.failures += is_failure ? 1 : 0;
  counts.lost_speed += is_slower ? 1 : 0;
}

TEST(MoveParticle, GeneratedFlushSeamsOverManySteps_KeepTheSpeed) {
  // 1,000 runs per kind of flush seam (CheckSeamRun): the solids of BuildFlushSeamPath, fixed or moving together, world
  // or tilted axes, a start 1.5 to 6 tol before the seam, steps of tol / 4 to 1,024 tol, Min(60, Floor(14 tol / step) +
  // 2) of them, the same input velocity every step: every step succeeds and keeps the speed along the path that
  // friction on the surface slid on leaves (plan decision P21, item 16). Plan decision P21's model, on its own draws:
  // 0 of 1,000 lose speed; as built, 165 (drums) and 69 (pucks).
  for (const FlushSeamKind kind : {FlushSeamKind::kStackedDrums, FlushSeamKind::kPuckInFloor}) {
    CaseGenerator generator;
    SeamRunCounts counts;
    for (int run_index = 0; run_index < 1000; ++run_index) {
      CheckSeamRun(generator, kind, counts);
    }
    std::cout << "flush seams over many steps, kind " << static_cast<int>(kind) << ": runs " << counts.runs
              << ", failures " << counts.failures << ", lost speed " << counts.lost_speed << "\n";
    EXPECT_EQ(counts.runs, 1000);
    EXPECT_EQ(counts.failures, 0);
    EXPECT_EQ(counts.lost_speed, 0);
  }
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

// A move that ended at the given position after one hit, with the scene's inputs.
ParticleMove BuildMoveAfterOneHit(Vec3 end_position) {
  ParticleMove move;
  move.result.position = end_position;
  move.result.hit_count = 1;
  move.has_response = true;
  return move;
}

TEST(ApplySettleAtEnd, TouchedFloorCloserThanTolerance_MovesToRestGap) {
  const std::array solids = {move_particle_cases::BuildFloor(1)};
  SolidContactInputs inputs;
  inputs.solids = solids.data();
  inputs.solid_count = 1;
  inputs.step_duration = 1;
  inputs.contact_tolerance = kTolerance;
  ParticleMove move = BuildMoveAfterOneHit({.x = 0.5, .y = 0, .z = 0.25 * kTolerance});
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
  ParticleMove move = BuildMoveAfterOneHit({.x = 0.5, .y = 0, .z = 2 * kTolerance});
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
  ParticleMove move = BuildMoveAfterOneHit({.x = 0, .y = 0, .z = 0.25 + 0.25 * kTolerance});
  ApplySettleAtEnd(inputs, &move);
  EXPECT_NEAR(move.result.position.z, 0.25 + kTolerance, 0x1p-52);
}

TEST(ApplySettleAtEnd, SixtyDegreeWedgeBothNearAtEnd_EndsAtRestGapOfBoth) {
  // tol / 2 from the floor and from the 60 degree face, both touched: one move to {Sqrt(3) tol, 0, tol}. One move per
  // surface in turn left the floor at 0.625 tol.
  const MoveParticleCase scene = FindMoveParticleCase("SixtyDegreeWedgeRestingOnBoth");
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  ParticleMove move = BuildMoveAfterOneHit({.x = std::numbers::sqrt3 / 2 * kTolerance, .y = 0, .z = kTolerance / 2});
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
  ParticleMove move = BuildMoveAfterOneHit({.x = std::numbers::sqrt3 / 2 * kTolerance, .y = 0, .z = kTolerance / 2});
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
