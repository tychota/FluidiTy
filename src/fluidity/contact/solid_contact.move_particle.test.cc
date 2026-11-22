#include "fluidity/contact/solid_contact.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/solid_contact_move_particle_cases.h"

namespace fluidity::contact {
namespace {
// m. Two contact tolerances: an expected rest point is written to within the rest gap, and rounding adds far less.
constexpr double kPositionTolerance = 2 * kTolerance;

constexpr Vec3 kUnitY = move_particle_cases::kUnitY;

// Runs a scene from another start, with both velocities as the scene has them.
MoveParticleCall ComputeMoveFrom(std::string_view case_name, Vec3 start_position) {
  MoveParticleCase scene = FindMoveParticleCase(case_name);
  scene.inputs.start_position = start_position;
  return ComputeMove(scene);
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
MoveParticleCall ComputeMoveWith(std::string_view case_name, SceneStart start) {
  return ComputeMove(BuildSceneFrom(case_name, start));
}

// Fails the calling test unless |actual - expected| <= 2^-44 * max(|actual|, |expected|, reference): the relative
// tolerance of the contact tests, a few hundred rounding units of the larger value: room for the rounding of a closed
// form, far below what a wrong formula gives. reference is the largest input speed of the test (1 for
// fractions).
void CheckNearRelative(double actual, double expected, double reference) {
  const double scale = std::max({std::abs(actual), std::abs(expected), reference});
  EXPECT_LE(std::abs(actual - expected), 0x1p-44 * scale) << actual << " vs " << expected;
}

// CheckNearRelative for each axis.
void CheckVec3NearRelative(Vec3 actual, Vec3 expected, double reference) {
  CheckNearRelative(actual.x, expected.x, reference);
  CheckNearRelative(actual.y, expected.y, reference);
  CheckNearRelative(actual.z, expected.z, reference);
}

// Fails the calling test unless the two vectors are equal axis by axis, bit for bit up to the sign of zero.
void CheckVec3Equal(Vec3 actual, Vec3 expected) {
  EXPECT_EQ(actual.x, expected.x) << "axis x";
  EXPECT_EQ(actual.y, expected.y) << "axis y";
  EXPECT_EQ(actual.z, expected.z) << "axis z";
}

// Fails the calling test unless the position, the velocities and the reaction are the untouched ones.
void CheckUntouchedMotion(const SolidContactResult& result) {
  const SolidContactResult untouched = BuildUntouchedResult();
  CheckVec3Equal(result.position, untouched.position);
  CheckVec3Equal(result.velocity.for_next_step, untouched.velocity.for_next_step);
  CheckVec3Equal(result.velocity.for_this_step, untouched.velocity.for_this_step);
  EXPECT_EQ(result.reaction.solid_index, untouched.reaction.solid_index);
  CheckVec3Equal(result.reaction.hit_position, untouched.reaction.hit_position);
  CheckVec3Equal(result.reaction.velocity_for_next_step_change, untouched.reaction.velocity_for_next_step_change);
}

// Fails the calling test unless the kind, the time, the normal and the approach speeds are the untouched ones.
void CheckUntouchedHitMotion(const SolidContactResult& result) {
  const SolidContactResult untouched = BuildUntouchedResult();
  EXPECT_EQ(result.kind, untouched.kind);
  EXPECT_EQ(result.hit_fraction, untouched.hit_fraction);
  CheckVec3Equal(result.normal_out_of_solid, untouched.normal_out_of_solid);
  EXPECT_EQ(result.approach_speed.for_next_step, untouched.approach_speed.for_next_step);
  EXPECT_EQ(result.approach_speed.for_this_step, untouched.approach_speed.for_this_step);
}

// Fails the calling test unless the solid, face, side and hit count are the untouched ones.
void CheckUntouchedHitSolid(const SolidContactResult& result) {
  const SolidContactResult untouched = BuildUntouchedResult();
  EXPECT_EQ(result.solid_index, untouched.solid_index);
  EXPECT_EQ(result.face, untouched.face);
  EXPECT_EQ(result.side_at_hit, untouched.side_at_hit);
  EXPECT_EQ(result.hit_count, untouched.hit_count);
}

// Fails the calling test unless the result is the untouched one, field by field.
void CheckUntouched(const SolidContactResult& result) {
  CheckUntouchedMotion(result);
  CheckUntouchedHitMotion(result);
  CheckUntouchedHitSolid(result);
}

// Fails the calling test unless the failure is exactly the expected one.
void CheckFailure(const ContactFailure& actual, ContactFailure expected) {
  EXPECT_EQ(actual.error, expected.error);
  EXPECT_EQ(actual.close_call, expected.close_call);
  EXPECT_EQ(actual.can_shorter_step_cure, expected.can_shorter_step_cure);
  EXPECT_EQ(actual.particle_id, expected.particle_id);
  EXPECT_EQ(actual.solid_id, expected.solid_id);
  EXPECT_EQ(actual.magnitude, expected.magnitude);
}

// Fails the calling test unless the call failed with kBadInput, said nothing else, and left the result unchanged.
void CheckBadInput(std::string_view case_name) {
  const MoveParticleCall call = ComputeMove(case_name);
  EXPECT_EQ(call.error, ContactError::kBadInput);
  CheckFailure(call.failure, {.error = ContactError::kBadInput});
  CheckUntouched(call.result);
}

// Fails the calling test unless the reaction is empty: no reacting solid was hit.
void CheckNoReaction(const SolidReaction& reaction) {
  EXPECT_EQ(reaction.solid_index, kNoSolid);
  CheckVec3Equal(reaction.hit_position, {});
  CheckVec3Equal(reaction.velocity_for_next_step_change, {});
}

// Whether a shorter step can cure a close call (ContactFailure::can_shorter_step_cure), as a test expects it.
enum class ShorterStep : std::uint8_t {
  kDoesNotCure,  // Decided at the start of the step: the same close call at every step length.
  kCures,  // A hit after the start, or a test that reads the time left.
};

// Fails the calling test unless the call failed with kTooCloseToCall and the close call, naming solid_id, magnitude 0,
// with can_shorter_step_cure as shorter_step says, and left the result unchanged.
void CheckCloseCall(
    const MoveParticleCall& call,
    ContactCloseCall close_call,
    std::uint32_t solid_id,
    ShorterStep shorter_step
) {
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  CheckFailure(
      call.failure,
      {
          .error = ContactError::kTooCloseToCall,
          .close_call = close_call,
          .can_shorter_step_cure = shorter_step == ShorterStep::kCures,
          .solid_id = solid_id,
      }
  );
  CheckUntouched(call.result);
}

// --- The examples of the method paper and the docstring ---

TEST(MoveParticle, NoSolid_EndIsStartPlusVelocityTimesStep) {
  const MoveParticleCase test_case = FindMoveParticleCase("NoSolid");
  const SolidContactInputs& inputs = test_case.inputs;
  const MoveParticleCall call = ComputeMove(test_case);
  ASSERT_EQ(call.error, ContactError::kNone);
  const Vec3 start = inputs.start_position;
  const Vec3 velocity = inputs.velocity.for_this_step;
  const double step_duration = inputs.step_duration;
  EXPECT_EQ(call.result.position.x, start.x + velocity.x * step_duration);
  EXPECT_EQ(call.result.position.y, start.y + velocity.y * step_duration);
  EXPECT_EQ(call.result.position.z, start.z + velocity.z * step_duration);
  CheckVec3Equal(call.result.velocity.for_next_step, inputs.velocity.for_next_step);
  CheckVec3Equal(call.result.velocity.for_this_step, inputs.velocity.for_this_step);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_fraction, 0);
  EXPECT_EQ(call.result.hit_count, 0);
  EXPECT_EQ(call.result.solid_index, kNoSolid);
  CheckNoReaction(call.result.reaction);
}

TEST(MoveParticle, DocstringExample_StopsOneToleranceShort) {
  const MoveParticleCall call = ComputeMove("DocstringExample");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  CheckNearRelative(result.hit_fraction, (0.1 - kTolerance) / 0.2, 1);
  ExpectVec3Near(result.position, {.x = 1 - kTolerance, .y = 0, .z = 0}, kPositionTolerance);
  CheckVec3NearRelative(result.velocity.for_next_step, {}, 1);
  CheckVec3NearRelative(result.velocity.for_this_step, {}, 1);
  CheckNearRelative(result.approach_speed.for_next_step, -1, 1);
  CheckNearRelative(result.approach_speed.for_this_step, -1, 1);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  CheckVec3Equal(result.normal_out_of_solid, {.x = -1, .y = 0, .z = 0});
  EXPECT_EQ(result.solid_index, 0U);
  EXPECT_EQ(result.side_at_hit, SolidSide::kNoPlate);
  EXPECT_EQ(result.hit_count, 1);
  CheckNoReaction(result.reaction);
}

TEST(MoveParticle, SecondSolidHitFirst_RecordNamesItsIndex) {
  // The record names the solid by its index in the array, 1 here, not by its id (2).
  const MoveParticleCall call = ComputeMove("SecondSolidHitFirst");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3Near(call.result.position, {.x = 1 - kTolerance, .y = 0, .z = 0.5}, kPositionTolerance);
}

TEST(MoveParticle, Success_WritesDefaultFailure) {
  const MoveParticleCall call = ComputeMove("DocstringExample");
  ASSERT_EQ(call.error, ContactError::kNone);
  CheckFailure(call.failure, ContactFailure{});
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
  CheckUntouched(result);
}

TEST(MoveParticle, HitAtEndOfStep_EndsAtTheHit) {
  // The step ends exactly at the hit time of the docstring example: the hit counts, and no time is left to sweep.
  MoveParticleCase test_case = FindMoveParticleCase("DocstringExample");
  test_case.inputs.step_duration = (1 - 0.9) - kTolerance;  // s, FindFirstHit's hit time, bit for bit.
  const MoveParticleCall call = ComputeMove(test_case);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_fraction, 1);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3Near(call.result.position, {.x = 1 - kTolerance, .y = 0, .z = 0}, kPositionTolerance);
  CheckVec3NearRelative(call.result.velocity.for_this_step, {}, 1);
}

TEST(MoveParticle, PaperFigureRisingFloor_MatchesFigure) {
  // Method paper <fig-sweep-hit-slide>: hit at 0.6 - tol / 2 s; for_this_step slips to {2, 0, 0.5} m/s and
  // for_next_step to {0.75, 0, 0.5} m/s; the slide rides up with the floor to x = 2.3 m, tol above its end height of
  // 0.5 m. The reaction is the change of for_next_step, {-0.75, 0, 3} m/s, exact in binary.
  const MoveParticleCall call = ComputeMove("PaperFigureRisingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  constexpr double kLargestSpeed = 2.5;  // m/s
  CheckNearRelative(result.hit_fraction, 0.6 - kTolerance / 2, 1);
  CheckVec3NearRelative(result.velocity.for_this_step, {.x = 2, .y = 0, .z = 0.5}, kLargestSpeed);
  CheckVec3NearRelative(result.velocity.for_next_step, {.x = 0.75, .y = 0, .z = 0.5}, kLargestSpeed);
  CheckNearRelative(result.approach_speed.for_next_step, -3, kLargestSpeed);
  CheckNearRelative(result.approach_speed.for_this_step, -2, kLargestSpeed);
  ExpectVec3Near(result.position, {.x = 2.3, .y = 0, .z = 0.5 + kTolerance}, kPositionTolerance);
  // The rising floor is touched: the slide moves with it, and rounding at the hit is not a second hit.
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.reaction.solid_index, 0U);
  CheckVec3Equal(result.reaction.velocity_for_next_step_change, {.x = -0.75, .y = 0, .z = 3});
  ExpectVec3Near(result.reaction.hit_position, {.x = 1.5, .y = 0, .z = 0.3 + kTolerance}, kPositionTolerance);
}

TEST(MoveParticle, DifferentVelocities_EachGetsItsOwnApproach) {
  // for_next_step {1, 0, -3}: approach -3, sideways 1, and 0.5 >= 1 / 3 sticks at zero. for_this_step {0, 0, -1}: no
  // sideways speed, so it sticks too.
  const MoveParticleCall call = ComputeMove("DifferentVelocities");
  ASSERT_EQ(call.error, ContactError::kNone);
  CheckVec3Equal(call.result.velocity.for_next_step, {});
  CheckVec3Equal(call.result.velocity.for_this_step, {});
  EXPECT_EQ(call.result.approach_speed.for_next_step, -3);
  EXPECT_EQ(call.result.approach_speed.for_this_step, -1);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = kTolerance}, kPositionTolerance);
}

// --- Plates, moving and fixed ---

// Fails the calling test unless the scene of one plate row that hits ended as its closed form says. Plate row 0 meets
// the face toward -e, plate row 1 the face toward +e; the side hit is the one the particle is on, by the plate's
// normal.
void CheckPlateRowHit(std::string_view case_name, int plate_row, Vec3 axis, bool is_normal_along_axis) {
  constexpr double kHalfThickness = 1.0 / 32;  // m
  constexpr double kLargestSpeed = 2;  // m/s
  const MoveParticleCall call = ComputeMove(case_name);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  const double side_sign = plate_row == 0 ? -1.0 : 1.0;
  const bool is_along_normal_side = (side_sign > 0) == is_normal_along_axis;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  CheckNearRelative(result.hit_fraction, (1 - kHalfThickness - kTolerance) / 2, 1);
  ExpectVec3Near(result.normal_out_of_solid, side_sign * axis, 0);
  CheckNearRelative(result.approach_speed.for_next_step, -2, kLargestSpeed);
  CheckNearRelative(result.approach_speed.for_this_step, -2, kLargestSpeed);
  EXPECT_EQ(result.side_at_hit, is_along_normal_side ? SolidSide::kAlongNormal : SolidSide::kAgainstNormal);
  EXPECT_EQ(result.face, is_along_normal_side ? SolidFace::kHighZ : SolidFace::kLowZ);
  const double end_coordinate = plate_row == 0 ? -(kHalfThickness + kTolerance) : 1 + kHalfThickness + kTolerance;
  ExpectVec3Near(result.position, end_coordinate * axis, kPositionTolerance);
  const Vec3 end_velocity = plate_row == 0 ? Vec3{} : 2 * axis;
  CheckVec3NearRelative(result.velocity.for_next_step, end_velocity, kLargestSpeed);
  CheckVec3NearRelative(result.velocity.for_this_step, end_velocity, kLargestSpeed);
}

// Fails the calling test unless the scene of plate row 2 ended free, moved by e.
void CheckPlateRowFree(std::string_view case_name, Vec3 axis) {
  const MoveParticleCall call = ComputeMove(case_name);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, axis, kPositionTolerance);
  CheckVec3Equal(call.result.velocity.for_this_step, axis);
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
          CheckPlateRowFree(name, axes.at(axis_index));
        } else {
          CheckPlateRowHit(name, plate_row, axes.at(axis_index), is_normal_along_axis);
        }
      }
    }
  }
}

// --- Several hits in one step ---

TEST(MoveParticle, FrictionOnSideBottom_LeavesSideways2Point5) {
  // Hit at t = 0.1 - tol; friction 0.5 takes 0.5 m/s of the 3 m/s sideways; the slide covers 2.5 (0.2 - t) m.
  const MoveParticleCall call = ComputeMove("FrictionOnSideBottom");
  ASSERT_EQ(call.error, ContactError::kNone);
  const double hit_elapsed = 0.1 - kTolerance;  // s
  CheckVec3NearRelative(call.result.velocity.for_this_step, {.x = 2.5, .y = 0, .z = 0}, 3);
  CheckVec3NearRelative(call.result.velocity.for_next_step, {.x = 2.5, .y = 0, .z = 0}, 3);
  ExpectVec3Near(
      call.result.position,
      {.x = 0.2 + 3 * hit_elapsed + 2.5 * (0.2 - hit_elapsed), .y = 0.5, .z = kTolerance},
      kPositionTolerance
  );
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  CheckVec3Equal(call.result.normal_out_of_solid, {.x = 0, .y = 0, .z = 1});
  EXPECT_EQ(call.result.hit_count, 1);
}

TEST(MoveParticle, FrictionScaleTwo_DoublesFriction) {
  // Friction 0.5 times friction_scale 2 takes 1 m/s of the 3 m/s sideways.
  const MoveParticleCall call = ComputeMove("FrictionScaleTwo");
  ASSERT_EQ(call.error, ContactError::kNone);
  CheckVec3NearRelative(call.result.velocity.for_this_step, {.x = 2, .y = 0, .z = 0}, 3);
  CheckVec3NearRelative(call.result.velocity.for_next_step, {.x = 2, .y = 0, .z = 0}, 3);
  ExpectVec3Near(call.result.position, {.x = 2.5, .y = 0, .z = kTolerance}, kPositionTolerance);
}

TEST(MoveParticle, DomainCorner_ThreeSidesHitInOneStep) {
  // side_left at 0.1 - tol s, side_front 0.1 s later, side_bottom 0.1 s after that: each removes one axis.
  const MoveParticleCall call = ComputeMove("DomainCorner");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  ExpectVec3Near(result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kPositionTolerance);
  CheckVec3NearRelative(result.velocity.for_next_step, {}, 1);
  CheckVec3NearRelative(result.velocity.for_this_step, {}, 1);
  EXPECT_EQ(result.hit_count, 3);
  EXPECT_EQ(result.face, SolidFace::kLowX);
  CheckNearRelative(result.hit_fraction, (0.1 - kTolerance) / 0.5, 1);
}

// Moves a particle from {start_x, 0, 0} at 1 m/s for 0.5 s toward the fixed flat solid x >= 1 and returns its hit
// fraction, or -1 when it does not hit.
double ComputeHitFractionTowardFlatSolid(double start_x) {
  const std::array solids = {move_particle_cases::BuildFlatSolid({.x = 1, .y = 0, .z = 0}, -kUnitX, kNoMotion, 1)};
  SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
      .start_position = {.x = start_x, .y = 0, .z = 0},
      .velocity = kUnitX,
      .step_duration = 0.5,
      .friction = 0,
  });
  inputs.solids = solids.data();
  inputs.solid_count = 1;
  SolidContactResult result;
  EXPECT_EQ(MoveParticle(inputs, &result, nullptr), ContactError::kNone);
  return result.kind == ContactKind::kHit ? result.hit_fraction : -1;
}

TEST(MoveParticle, HitAtEndOfStep_HitFractionIsAtMostOne) {
  // 16 starts one rounding unit apart around 0.5 - tol, where the hit lands at the very end of the step
  // (ComputeHitFractionTowardFlatSolid): no hit fraction exceeds 1, and some are exactly 1, the boundary the field
  // promises (the first hit's time is within the first sweep, which starts at 0).
  double start_x = 0.5 - kTolerance;  // m
  for (int step_back = 0; step_back < 8; ++step_back) {
    start_x = std::nextafter(start_x, 0.0);
  }
  int hit_count = 0;
  int at_one_count = 0;
  for (int start_index = 0; start_index < 16; ++start_index) {
    const double hit_fraction = ComputeHitFractionTowardFlatSolid(start_x);
    hit_count += hit_fraction >= 0 ? 1 : 0;
    at_one_count += hit_fraction == 1.0 ? 1 : 0;
    EXPECT_LE(hit_fraction, 1.0) << start_x;
    start_x = std::nextafter(start_x, 1.0);
  }
  EXPECT_GT(hit_count, 0);
  EXPECT_GT(at_one_count, 0);
}

TEST(MoveParticle, OctagonRoom_FifthHitIsTooManyHits) {
  // The particle hits side 0 of the room, slides along sides 0, 1, 2 and 3, and meets side 4 (id 5) at 0.279 s.
  const MoveParticleCall call = ComputeMove("OctagonRoom");
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  CheckFailure(
      call.failure,
      {
          .error = ContactError::kTooCloseToCall,
          .close_call = ContactCloseCall::kTooManyHits,
          .can_shorter_step_cure = true,
          .solid_id = 5,
      }
  );
  CheckUntouched(call.result);
}

TEST(MoveParticle, OctagonRoomShortStep_FifthHitBeforePoint3) {
  // The fifth hit comes before 0.3 s: a step of 0.3 s fails too, while a step of 0.27 s ends after four hits.
  MoveParticleCase test_case = FindMoveParticleCase("OctagonRoom");
  test_case.inputs.step_duration = 0.3;
  EXPECT_EQ(ComputeMove(test_case).error, ContactError::kTooCloseToCall);
  test_case.inputs.step_duration = 0.27;
  const MoveParticleCall four_hits = ComputeMove(test_case);
  ASSERT_EQ(four_hits.error, ContactError::kNone);
  EXPECT_EQ(four_hits.result.hit_count, 4);
  EXPECT_EQ(four_hits.result.solid_index, 0U);
}

TEST(MoveParticle, FloorThenRamp_SlidesUpTheRamp) {
  // The floor at 0.1 - tol s leaves {1, 0, 0}; the ramp with normal {-0.6, 0, 0.8} at 0.5 - tol / 3 s leaves
  // {0.64, 0, 0.48}, which leaves the floor. The settle must not pull the particle back to it.
  const MoveParticleCall call = ComputeMove("FloorThenRamp");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  EXPECT_EQ(call.result.solid_index, 0U);
  CheckVec3NearRelative(call.result.velocity.for_this_step, {.x = 0.64, .y = 0, .z = 0.48}, 1);
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
  const MoveParticleCall call = ComputeMove("FloorThenMovingUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  CheckVec3NearRelative(call.result.velocity.for_this_step, {.x = -0.5, .y = 0, .z = 0}, 1);
  CheckVec3NearRelative(call.result.velocity.for_next_step, {.x = -0.5, .y = 0, .z = 0}, 1);
  ExpectVec3Near(call.result.position, {.x = 0.5 - kTolerance, .y = 0, .z = kTolerance}, kPositionTolerance);
  EXPECT_EQ(call.result.reaction.solid_index, 1U);
  CheckVec3Equal(call.result.reaction.velocity_for_next_step_change, {.x = -1.5, .y = 0, .z = 0});
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
  // hits. With one response per surface in turn, the touched floor would be hit again: three hits.
  const MoveParticleCall call = ComputeMove("TiltedUprightPushesBackToFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  EXPECT_NEAR(call.result.position.z, kTolerance, kTolerance / 16);
  EXPECT_NEAR(call.result.position.x, 1 - kTolerance, kPositionTolerance);
}

TEST(MoveParticle, TiltedUprightOverRisingFloor_SlideCarriesParticleWithFloor) {
  // As TiltedUprightPushesBackToFloor with the floor rising at 0.25 m/s: hit at (0.1 - tol) / 1.25 s, the upright at
  // about 0.5 s, and the joint response over both gives the velocity that approaches neither, {-3.5e-10, 0, 0.25}: the
  // particle rides up with the floor to z = 0.25 + tol. Before it the moving touched floor was hit again: three hits.
  const MoveParticleCall call = ComputeMove("TiltedUprightOverRisingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  EXPECT_NEAR(call.result.position.z, 0.25 + kTolerance, kTolerance / 16);
  EXPECT_NEAR(call.result.position.x, 1 - kTolerance, kPositionTolerance);
}

TEST(MoveParticle, ChasingSolidAtRightAngles_FirstSolidNotHitAgain) {
  // B's response leaves an approach to A of B's rounding; with only A's own allowance A was hit again at 0.
  const MoveParticleCall call = ComputeMove("ChasingSolidAtRightAngles");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
}

TEST(MoveParticle, NearStickSlipOnObliqueFlatSolid_HitsOnce) {
  // The slip leaves an approach of rounding of the incoming speed in a tiny returned velocity: within the allowance of
  // the incoming velocity, so not a second hit.
  const MoveParticleCall call = ComputeMove("NearStickSlipOnObliqueFlatSolid");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
}

TEST(MoveParticle, ObliqueFlatSolidRoundingAtHit_HitsOnce) {
  // The hit at 0.0625 - tol / 2.8 s leaves the particle 2.8e-18 m outside the rest gap, moving along the surface
  // at {-0.32, 0, 0.24} m/s with an approach of -2^-55 m/s: rounding, within the allowance, so not a second hit.
  const MoveParticleCall call = ComputeMove("ObliqueFlatSolidRoundingAtHit");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckVec3NearRelative(call.result.velocity.for_this_step, {.x = -0.32, .y = 0, .z = 0.24}, 2);
}

// --- A slide on a turning solid: the rest of the step carried in its axes ---

// m. The distance from a point to the first solid of a scene at the solid's pose at the end of the step.
double ComputeDistanceAtEndPose(const MoveParticleCase& scene, Vec3 point) {
  const Solid& solid = scene.solids.at(0);
  return ComputeDistanceToSolid(solid, ComputeSolidPose(solid, scene.inputs.step_duration), point).distance;
}

TEST(MoveParticle, TurningBoxHitsRestingParticle_EndsOnFaceAtEndPose) {
  // The face kHighX reaches the particle at rest when 0.6 cos(t - 0.3) = 0.5 + tol, about 0.8857 s; the response leaves
  // it moving with the face, and the carry along the face for the 0.1143 s left keeps it near the rest gap of the box
  // at its end pose. One hit: before the carry, every later sweep hit the box again at 0 until kTooManyHits.
  const MoveParticleCase scene = FindMoveParticleCase("TurningBoxHitsRestingParticle");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.face, SolidFace::kHighX);
  EXPECT_NEAR(result.hit_fraction, 0.3 + std::acos((0.5 + kTolerance) / 0.6), 0x1p-30);
  const double end_distance = ComputeDistanceAtEndPose(scene, result.position);  // m
  EXPECT_GE(end_distance, kTolerance / 2);
  EXPECT_LE(end_distance, 2 * kTolerance);
}

TEST(MoveParticle, TurningBoxNearFloor_IsTurningSolidNearOtherSolid) {
  // The same hit with a fixed floor 0.05 m below the particle: in the 0.1143 s left the floor is within the carry's
  // closing bound (method paper <sec-move-particle>), so the step is too close to call, naming the turning box.
  CheckCloseCall(
      ComputeMove("TurningBoxNearFloor"),
      ContactCloseCall::kTurningSolidNearOtherSolid,
      1,
      ShorterStep::kCures
  );
}

TEST(MoveParticle, TurningBoxSlide_KeepsSidewaysPartInSolidAxes) {
  // At rest on the face kHighX at d == tol, the particle is hit at 0 (no touching rule on a turning solid, method paper
  // <sec-turning-solid>) and carried. Its velocity relative to the face, {0, -(0.5 + tol), 1} m/s, turned into the
  // box's axes of the middle of the step, 0.05 rad on, is {-(0.5 + tol) sin 0.05, -(0.5 + tol) cos 0.05, 1}: the face
  // turns toward the line the particle follows, so the part into it is removed, and it ends at own {0.5 + tol, -0.1
  // (0.5 + tol) cos 0.05, 0.1} of the box's pose at 0.1 s, its velocities unchanged. (The carry with the axes of the
  // hit would end at {0.5 + tol, -0.05, 0.1}, 6.2e-5 m away.)
  const MoveParticleCase scene = FindMoveParticleCase("TurningBoxSlide");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.hit_fraction, 0);
  EXPECT_EQ(result.face, SolidFace::kHighX);
  CheckVec3Equal(result.velocity.for_this_step, kUnitZ);
  CheckVec3Equal(result.velocity.for_next_step, kUnitZ);
  const SolidPose end_pose = ComputeSolidPose(scene.solids.at(0), 0.1);
  const Vec3 own_end = {.x = 0.5 + kTolerance, .y = -0.1 * (0.5 + kTolerance) * std::cos(0.05), .z = 0.1};
  ExpectVec3Near(result.position, end_pose.origin + end_pose.axes * own_end, kPositionTolerance);
}

// The state after a run of steps on solids that keep moving: the position, the velocities, and how the steps went.
struct TurningStepsRun {
  Vec3 position{};  // m, world, at the end of the last step.
  PerParticleVelocity<Vec3> velocity{};  // m/s, world.
  int failed_steps = 0;
  int steps_with_one_hit = 0;
  double least_end_share = std::numeric_limits<double>::infinity();  // Of tol, to the first solid at each end pose.
  double largest_end_share = 0;  // Of tol.
};

// How a run of steps goes: the step's duration, the friction and the number of steps.
struct TurningSteps {
  double step_duration = 0;  // s.
  double friction = 0;  // No unit.
  int step_count = 50;
};

// Runs steps.step_count steps from start at rest: each step adds gravity times the step to the velocity the last one
// kept, poses the solids at the step's start, and moves the particle with that friction.
TurningStepsRun ComputeTurningSteps(const std::vector<Solid>& solids, Vec3 start, TurningSteps steps) {
  const double step_duration = steps.step_duration;  // s
  const double friction = steps.friction;
  const int step_count = steps.step_count;
  const Vec3 gravity = {.x = 0, .y = 0, .z = -9.81};  // m/s^2
  TurningStepsRun run;
  run.position = start;
  for (int step = 0; step < step_count; ++step) {
    std::vector<Solid> posed = solids;
    for (Solid& solid : posed) {
      solid.pose_at_step_start = ComputeSolidPose(solid, step * step_duration);
    }
    SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
        .start_position = run.position,
        .velocity = run.velocity.for_next_step + step_duration * gravity,
        .step_duration = step_duration,
        .friction = friction,
    });
    inputs.solids = posed.data();
    inputs.solid_count = static_cast<std::uint32_t>(posed.size());
    SolidContactResult result;
    ContactFailure failure;
    if (MoveParticle(inputs, &result, &failure) != ContactError::kNone) {
      ++run.failed_steps;
      continue;
    }
    run.steps_with_one_hit += result.hit_count == 1 ? 1 : 0;
    const Solid& first = posed.at(0);
    const double end_share =
        ComputeDistanceToSolid(first, ComputeSolidPose(first, step_duration), result.position).distance / kTolerance;
    run.least_end_share = std::min(run.least_end_share, end_share);
    run.largest_end_share = std::max(run.largest_end_share, end_share);
    run.position = result.position;
    run.velocity = result.velocity;
  }
  return run;
}

// A box of half size 0.5 m with its top at z = 0, spinning at 0.5 rad/s about z: a turntable.
Solid BuildTurntable() {
  return ToSolid(
      Box{.center = {.x = 0, .y = 0, .z = -0.5}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
      {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 0.5}},
      1
  );
}

// Fails the calling test unless 50 frictionless steps of step_duration on the turntable, from rest at radius 0.3 m,
// each take one hit, end at the rest gap with velocity 0, and leave the particle where the midpoint rule's residue puts
// it: per step of turn a = omega T, radius r (1 + a^2 - 2 a sin(a / 2))^(1/2) = r (1 + a^4 / 48 + ...), and r a^3 / 24
// behind its start along the circle (ComputeSolidAxesCarry), each within 5 %.
void CheckTurntableFrictionlessResidue(double step_duration) {
  const Vec3 start = {.x = 0.3, .y = 0, .z = kTolerance};
  const TurningStepsRun run = ComputeTurningSteps({BuildTurntable()}, start, {.step_duration = step_duration});
  ASSERT_EQ(run.failed_steps, 0);
  EXPECT_EQ(run.steps_with_one_hit, 50);
  EXPECT_GE(run.least_end_share, 0.99);
  EXPECT_LE(run.largest_end_share, 1.01);
  ExpectVec3Near(run.velocity.for_next_step, {.x = 0, .y = 0, .z = 0}, 1e-15);
  const double turn = 0.5 * step_duration;  // rad, a
  const double radius = std::hypot(run.position.x, run.position.y);  // m
  const double expected_growth = 50 * std::pow(turn, 4) / 48;  // share of r
  EXPECT_NEAR(radius / 0.3 - 1, expected_growth, 0.05 * expected_growth + 1e-14);
  const double moved = std::hypot(run.position.x - start.x, run.position.y - start.y);  // m
  const double expected_moved = 50 * 0.3 * std::pow(turn, 3) / 24;  // m
  EXPECT_NEAR(moved, expected_moved, 0.05 * expected_moved + 4 * kTolerance);
}

TEST(MoveParticle, TurntableFrictionless_StaysPutWithinMidpointResidue) {
  // A frictionless particle resting on the turntable keeps velocity 0 along the top, so it should not move. Over 50
  // steps: T = 0.1 s, a = 0.05, radius +6.51e-6 of r, moved by 7.8e-5 m; T = 1 / 120 s, radius +3.14e-10 of r, moved
  // by 4.5e-8 m. With the axes of the hit the radius grew by r a^2 / 2 per step (+6.4 % over 50 steps of 0.1 s).
  CheckTurntableFrictionlessResidue(0.1);
  CheckTurntableFrictionlessResidue(1.0 / 120);
}

// Fails the calling test unless 50 steps of step_duration on the turntable with friction 2, from rest at radius 0.3 m,
// each take one hit and carry the particle round at radius 0.3 m and 0.15 m/s through the angle 50 omega T, each to
// rounding.
void CheckTurntableStickingCarriedRound(double step_duration) {
  const Vec3 start = {.x = 0.3, .y = 0, .z = kTolerance};
  const TurningStepsRun run =
      ComputeTurningSteps({BuildTurntable()}, start, {.step_duration = step_duration, .friction = 2});
  ASSERT_EQ(run.failed_steps, 0);
  EXPECT_EQ(run.steps_with_one_hit, 50);
  EXPECT_NEAR(std::hypot(run.position.x, run.position.y), 0.3, 1e-13);
  const double speed = math::Length(run.velocity.for_next_step);  // m/s
  EXPECT_NEAR(0.5 * speed * speed, 0.5 * 0.15 * 0.15, 1e-13);
  EXPECT_NEAR(std::atan2(run.position.y, run.position.x), 50 * 0.5 * step_duration, 1e-12);
}

TEST(MoveParticle, TurntableSticking_CarriedRoundAtConstantRadiusAndEnergy) {
  // Friction 2: each step sticks the particle to the top, from the first (friction times gravity's approach, 2 * 9.81
  // T, is above the sideways speed omega r = 0.15 m/s even at T = 1 / 120 s; with friction 1 that first step slips),
  // so its relative velocity is 0 and the carry keeps its own point: radius, energy and angle to rounding, a few units
  // per step over 50 steps.
  CheckTurntableStickingCarriedRound(0.1);
  CheckTurntableStickingCarriedRound(1.0 / 120);
}

TEST(MoveParticle, FaceTurningAwayFromSlide_CarryLeavesFaceKnownChoice) {
  // Known choice of the midpoint carry (ComputeSolidAxesCarry). A box whose top face is at z = 0 turns at 0.5 rad/s
  // about x through its centre, 0.5 m below, so the top tilts away from a frictionless particle sliding along +y at
  // 0.5 m/s from the rest gap at y = 0.3 m with a gravity-like 0.981 m/s down. Hit at 0; the surface there moves at
  // {0, 0.25, -0.15} m/s, so the response leaves the relative velocity {0, 0.25, 0} m/s along the top. Turned by half
  // the step's turn, a / 2 = 0.025 rad, it points out of the face by 0.25 sin 0.025 m/s, which the carry keeps: the
  // particle ends 0.1 * 0.25 sin 0.025 = 6.249e-4 m, about 6.7e5 tol, off the face, while the velocity it returns has
  // no part along the normal. The bound is sin(a / 2) |v - u| t_left; the lift does not build up over steps (the next
  // sweep and gravity bring the particle back) and the path still converges at first order in the step.
  MoveParticleCase scene;
  scene.name = "FaceTurningAwayFromSlide";
  scene.solids = {
      ToSolid(
          Box{.center = {.x = 0, .y = 0, .z = -0.5}, .axes = kIdentity, .half_size = {.x = 1, .y = 1, .z = 0.5}},
          {.linear_velocity = {}, .angular_velocity = {.x = -0.5, .y = 0, .z = 0}},
          1
      ),
  };
  scene.inputs = move_particle_cases::BuildCaseInputs({
      .start_position = {.x = 0, .y = 0.3, .z = kTolerance},
      .velocity = {.x = 0, .y = 0.5, .z = -0.981},
      .step_duration = 0.1,
      .friction = 0,
  });
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  const Solid& box = scene.solids.at(0);
  const Vec3 relative_velocity = call.result.velocity.for_next_step
      - ComputeSolidVelocity(box, ComputeSolidPose(box, 0), scene.inputs.start_position);
  EXPECT_NEAR(relative_velocity.z, 0, 1e-15);
  const double half_turn = 0.5 * 0.1 / 2;  // rad
  const double bound = std::sin(half_turn) * math::Length(relative_velocity) * 0.1 + kTolerance;  // m
  const double end_distance = ComputeDistanceAtEndPose(scene, call.result.position);  // m
  EXPECT_NEAR(end_distance, 0.1 * 0.25 * std::sin(0.025) + kTolerance, 1e-9);
  EXPECT_LE(end_distance, bound + 2 * kTolerance);
}

TEST(MoveParticle, TiltingPaddle_SlidesFrictionlessRidesSticking) {
  // A paddle of half size {0.5, 0.25, 0.01} m tilting at 0.05 rad/s about x, a particle on its top under gravity, 50
  // steps of 0.01 s. Every step is one hit at 0 and a carry. Friction 0: it slides down the tilting face, faster each
  // step (0.005 m/s, then 0.060 m/s at the end), and a carry may leave the face by up to sin(a / 2) |v - u| T, a =
  // 5e-4 rad (FaceTurningAwayFromSlide_CarryLeavesFaceKnownChoice): up to 160.6 tol measured, under the bound of 176
  // tol at 0.065 m/s. Friction 1: it rides along at omega r = 0.05 * 0.1 m/s, its own point kept, each carry ending
  // at the rest gap.
  const Solid paddle = ToSolid(
      Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.25, .z = 0.01}},
      {.linear_velocity = {}, .angular_velocity = {.x = 0.05, .y = 0, .z = 0}},
      1
  );
  const Vec3 start = {.x = 0.1, .y = 0.1, .z = 0.01 + kTolerance};
  const TurningStepsRun sliding = ComputeTurningSteps({paddle}, start, {.step_duration = 0.01});
  ASSERT_EQ(sliding.failed_steps, 0);
  EXPECT_EQ(sliding.steps_with_one_hit, 50);
  EXPECT_GE(sliding.least_end_share, 0.99);
  EXPECT_LE(sliding.largest_end_share, 1.01 + std::sin(0.05 * 0.01 / 2) * 0.065 * 0.01 / kTolerance);
  EXPECT_GT(sliding.largest_end_share, 100);
  EXPECT_NEAR(math::Length(sliding.velocity.for_next_step), 0.060, 0.005);
  const TurningStepsRun riding = ComputeTurningSteps({paddle}, start, {.step_duration = 0.01, .friction = 1});
  ASSERT_EQ(riding.failed_steps, 0);
  EXPECT_EQ(riding.steps_with_one_hit, 50);
  EXPECT_GE(riding.least_end_share, 0.99);
  EXPECT_LE(riding.largest_end_share, 1.01);
  const SolidPose end_pose = ComputeSolidPose(paddle, 0.5);
  const Vec3 own_end = math::Transpose(end_pose.axes) * (riding.position - end_pose.origin);
  ExpectVec3Near(own_end, start, 2 * kTolerance);
}

// A box that moves at {0.2, -0.1, 0.05} m/s and turns at {0.3, 0.4, 0.5} rad/s, about no axis of its own, with a
// particle on its top face at the rest gap moving across it.
MoveParticleCase BuildMovingTurningBoxSlide() {
  MoveParticleCase scene;
  scene.name = "MovingTurningBoxSlide";
  scene.solids = {
      ToSolid(
          Box{.center = {.x = 0.1, .y = 0.2, .z = -0.5}, .axes = kIdentity, .half_size = {.x = 1, .y = 1, .z = 0.5}},
          {.linear_velocity = {.x = 0.2, .y = -0.1, .z = 0.05}, .angular_velocity = {.x = 0.3, .y = 0.4, .z = 0.5}},
          1
      ),
  };
  scene.inputs = move_particle_cases::BuildCaseInputs({
      .start_position = {.x = 0.3, .y = 0.1, .z = kTolerance},
      .velocity = {},
      .step_duration = 0.2,
      .friction = 0,
  });
  return scene;
}

TEST(MoveParticle, MovingTurningBoxSlide_CarriedWithMidStepAxes) {
  // Hit at 0 on the top face; the particle's velocity relative to the surface there, v - u(x), is turned into the
  // box's own axes with its axes at 0.1 s, the middle of the step, its part into the top removed, and the particle
  // moves on that straight line in own axes for 0.2 s, placed with the pose at 0.2 s. The test computes that end
  // itself. Carried with the axes of the hit or of the end instead, it would be off by about (a / 2) |v - u| T,
  // a = 0.141 rad the turn and |v - u| = 0.655 m/s: 9.3e-3 m here.
  MoveParticleCase scene = BuildMovingTurningBoxSlide();
  const Vec3 velocity = {.x = 1, .y = -0.5, .z = 0};
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  const Solid& box = scene.solids.at(0);
  const Vec3 start = scene.inputs.start_position;
  const SolidPose start_pose = ComputeSolidPose(box, 0);
  const Mat3 to_middle_axes = math::Transpose(ComputeSolidPose(box, 0.1).axes);
  Vec3 own_velocity = to_middle_axes * (velocity - ComputeSolidVelocity(box, start_pose, start));  // m/s
  own_velocity.z = std::max(own_velocity.z, 0.0);
  const Vec3 own_start = math::Transpose(start_pose.axes) * (start - start_pose.origin);
  const SolidPose end_pose = ComputeSolidPose(box, 0.2);
  const Vec3 expected_end = end_pose.origin + end_pose.axes * (own_start + 0.2 * own_velocity);
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kHighZ);
  ExpectVec3Near(call.result.position, expected_end, kPositionTolerance);
}

TEST(MoveParticle, MovingTurningBoxSlide_VelocityAgreesWithPathToFirstOrder) {
  // The velocities stay as the response made them; the carried path's mean velocity, (x_end - x) / T, differs from
  // for_next_step by at most (a / 2) (2 |v - v_lin| + |omega| r) + a^2 (|omega| r + |v - v_lin|), a = |omega| T
  // (ApplyCarryInSolidAxes): first order in the turn. On the turntable, where nothing is removed, a frictionless rest
  // moves by r a^3 / 24 a step (TurntableFrictionless_StaysPutWithinMidpointResidue).
  for (const Vec3 velocity : {Vec3{.x = 1, .y = -0.5, .z = 0}, Vec3{.x = 0.2, .y = -0.1, .z = 0}}) {
    MoveParticleCase scene = BuildMovingTurningBoxSlide();
    scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    const Solid& box = scene.solids.at(0);
    const MoveParticleCall call = ComputeMove(scene);
    ASSERT_EQ(call.error, ContactError::kNone);
    const double step_duration = scene.inputs.step_duration;  // s
    const Vec3 mean_velocity = (1 / step_duration) * (call.result.position - scene.inputs.start_position);  // m/s
    const double turn = math::Length(box.motion.angular_velocity) * step_duration;  // rad
    const double relative_speed = math::Length(velocity - box.motion.linear_velocity);  // m/s
    const double arm_speed = math::Length(box.motion.angular_velocity)
        * math::Length(scene.inputs.start_position - box.pose_at_step_start.origin);  // m/s
    const double bound = turn / 2 * (2 * relative_speed + arm_speed) + turn * turn * (arm_speed + relative_speed)
        + 4 * kTolerance / step_duration;  // m/s
    EXPECT_LE(math::Length(mean_velocity - call.result.velocity.for_next_step), bound);
  }
}

TEST(MoveParticle, TurningBoxLeavingParticle_CarriedAwayFromFace) {
  // At the rest gap of the face kHighX and moving away from it at 1 m/s: hit at 0 all the same, with no change, and
  // carried with the part along the normal kept: its relative velocity {1, -(0.5 + tol), 0} m/s turned into the axes
  // of the middle of the step has the part cos 0.05 - (0.5 + tol) sin 0.05 along the normal, so it ends that times
  // 0.1 s, plus tol, from the box at its end pose (0.0974 m; a particle on that world line is 0.0970 m away). Only a
  // part into the face is removed.
  const MoveParticleCase scene = BuildSceneFrom(
      "TurningBoxSlide",
      {.position = {.x = 0.5 + kTolerance, .y = 0, .z = 0}, .velocity = kUnitX, .step_duration = 0.1}
  );
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckVec3Equal(call.result.velocity.for_next_step, kUnitX);
  const double expected_distance = 0.1 * (std::cos(0.05) - (0.5 + kTolerance) * std::sin(0.05)) + kTolerance;  // m
  EXPECT_NEAR(ComputeDistanceAtEndPose(scene, call.result.position), expected_distance, kPositionTolerance);
}

TEST(MoveParticle, RestingOnTurningBoxNearFloor_StepLengthDecidesCloseCall) {
  // Hit at 0 with the whole step left; the carry moves at {0, -(0.5 + tol) cos(T / 2), 0} m/s in the box's axes (its
  // relative velocity turned by half the turn, the part into the face removed), 0.6726812 m from its origin, so its
  // closing bound against the floor 0.05 m below is T ((0.5 + tol) cos(T / 2) + 0.6726812) (method paper
  // <sec-move-particle>), which passes 0.05 - tol from T = 0.0426415 s (with the axes of the hit the carry would give
  // 0.0426373 s). T 0.1: 0.1172 m, too close to call. T 0.04: 0.0469 m, a carry that ends at the rest gap of the box
  // at its end pose, at the same height above the floor.
  CheckCloseCall(
      ComputeMove("RestingOnTurningBoxNearFloor"),
      ContactCloseCall::kTurningSolidNearOtherSolid,
      1,
      ShorterStep::kCures
  );
  const MoveParticleCase scene = BuildSceneFrom(
      "RestingOnTurningBoxNearFloor",
      {.position = {.x = 0.5 + kTolerance, .y = 0, .z = -0.45}, .velocity = {}, .step_duration = 0.04}
  );
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.hit_fraction, 0);
  EXPECT_EQ(call.result.face, SolidFace::kHighX);
  const double end_distance = ComputeDistanceAtEndPose(scene, call.result.position);  // m
  EXPECT_GE(end_distance, kTolerance / 2);
  EXPECT_LE(end_distance, 2 * kTolerance);
  EXPECT_NEAR(call.result.position.z, -0.45, kPositionTolerance);
}

TEST(MoveParticle, RestingOnTurningBoxNearMovingFloor_FloorMotionAddsToClosingBound) {
  // The step of 0.04 s that passes with a fixed floor, with the floor moving along itself at 1 m/s, or spinning about
  // its normal at 1 rad/s about its origin 0.5025 m from the particle: neither brings the floor nearer, but the closing
  // bound bounds every motion of the other solid by its speed, |v_lin_j| + |omega_j| r_j (method paper
  // <sec-move-particle>), 0.0869 m and 0.0670 m against 0.05 m, so both are too close to call.
  const MoveParticleCase scene = BuildSceneFrom(
      "RestingOnTurningBoxNearFloor",
      {.position = {.x = 0.5 + kTolerance, .y = 0, .z = -0.45}, .velocity = {}, .step_duration = 0.04}
  );
  const std::array<SolidMotion, 2> floor_motions = {
      SolidMotion{.linear_velocity = kUnitX, .angular_velocity = {}},
      SolidMotion{.linear_velocity = {}, .angular_velocity = kUnitZ},
  };
  for (const SolidMotion& floor_motion : floor_motions) {
    MoveParticleCase moving_floor_scene = scene;
    moving_floor_scene.solids.at(1).motion = floor_motion;
    CheckCloseCall(
        ComputeMove(moving_floor_scene),
        ContactCloseCall::kTurningSolidNearOtherSolid,
        1,
        ShorterStep::kCures
    );
  }
}

TEST(MoveParticle, CylinderTurningAcrossAxisHitsRestingParticle_EndsOnWallAtEndPose) {
  // A cylinder turning across its axis is a turning solid (method paper <sec-slide-and-spin>): its wall reaches the
  // particle at 1.2 - asin((0.5 + tol) / 0.6) s, and the carry keeps the particle near its rest gap at the end pose.
  const MoveParticleCase scene = FindMoveParticleCase("CylinderTurningAcrossAxisHitsRestingParticle");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kCurved);
  EXPECT_NEAR(call.result.hit_fraction, 0.21488921385421111, 0x1p-30);
  const double end_distance = ComputeDistanceAtEndPose(scene, call.result.position);  // m
  EXPECT_GE(end_distance, kTolerance / 2);
  EXPECT_LE(end_distance, 2 * kTolerance);
}

TEST(MoveParticle, TurningBoxStartInsideApproaching_OneHitAtTheStart) {
  // The start is inside the box and moves into it: the start's response removes the approach (velocities {0, 0, 0}),
  // then the first sweep finds the box at 0, at the rest gap where that response read it. That is the same moment, not
  // a second hit: hit_count 1, and the carry.
  const MoveParticleCall call = ComputeMove("TurningBoxStartInsideApproaching");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kHighX);
  CheckBothVelocitiesNear(call.result, {.x = 0, .y = 0, .z = 0}, 1e-15);
}

TEST(MoveParticle, TurningBoxWithCopy_SameVelocitiesHitsAndRecord) {
  // An exact copy of the turning box, with another id, fills the same space at every time: the carry keeps the particle
  // out of the box, so out of the copy, and the test of the other solids leaves it out. The move is the same, bit for
  // bit, as with the box alone.
  const MoveParticleCase scene = FindMoveParticleCase("TurningBoxHitsRestingParticle");
  MoveParticleCase with_copy = scene;
  Solid copy = scene.solids.at(0);
  copy.solid_id = 2;
  with_copy.solids.push_back(copy);
  const MoveParticleCall call = ComputeMove(scene);
  const MoveParticleCall other = ComputeMove(with_copy);
  ASSERT_EQ(call.error, ContactError::kNone);
  ASSERT_EQ(other.error, ContactError::kNone);
  EXPECT_EQ(other.result.hit_count, call.result.hit_count);
  EXPECT_EQ(other.result.hit_fraction, call.result.hit_fraction);
  EXPECT_EQ(other.result.solid_index, call.result.solid_index);
  EXPECT_EQ(other.result.face, call.result.face);
  CheckVec3Equal(other.result.velocity.for_this_step, call.result.velocity.for_this_step);
  CheckVec3Equal(other.result.velocity.for_next_step, call.result.velocity.for_next_step);
  CheckVec3Equal(other.result.position, call.result.position);
}

TEST(MoveParticle, TurningBoxWithFixedBoxInItsPlace_IsTurningSolidNearOtherSolid) {
  // A box of the same shape and pose that does not turn does not fill the same space through the step: 0.073 m from
  // the particle at the start, it is within the carry's closing bound after the hit, so the step is too close to call.
  MoveParticleCase scene = FindMoveParticleCase("TurningBoxHitsRestingParticle");
  Solid fixed_box = scene.solids.at(0);
  fixed_box.solid_id = 2;
  fixed_box.motion = kNoMotion;
  scene.solids.push_back(fixed_box);
  CheckCloseCall(ComputeMove(scene), ContactCloseCall::kTurningSolidNearOtherSolid, 1, ShorterStep::kCures);
}

TEST(MoveParticle, StartInsideFloorThenTurningBox_SecondHitHasItsResponse) {
  // The start is tol / 4 inside a fixed floor and leaves it at {1, 0, 1} m/s: a response at time 0 (kind
  // kStartedInsideSolid). The path then meets the face kLowX of a box turning slowly about z, 1 m away, at about 1 s:
  // a hit after a response, but not at 0, so it has its own response, which removes the approach along the face's
  // normal. The floor is 1 m below, beyond the carry's closing bound for the 0.5 s left.
  MoveParticleCase scene;
  scene.name = "StartInsideFloorThenTurningBox";
  scene.solids = {
      move_particle_cases::BuildFloor(1),
      ToSolid(
          Box{.center = {.x = 1.25, .y = 0, .z = 1}, .axes = kIdentity, .half_size = {.x = 0.25, .y = 0.5, .z = 0.5}},
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 0.01}},
          2
      ),
  };
  scene.inputs = move_particle_cases::BuildCaseInputs({
      .start_position = {.x = 0, .y = 0, .z = -kTolerance / 4},
      .velocity = {.x = 1, .y = 0, .z = 1},
      .step_duration = 1.5,
      .friction = 0,
  });
  scene.inputs.start_overlap_tolerance = kTolerance;
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  ASSERT_TRUE(ComputeIsTurningSolid(inputs, 1));
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.hit_count, 2);
  // At the hit the face's normal is near {-1, 0, 0}: the particle keeps at most the face's own speed along it.
  EXPECT_LT(std::abs(call.result.velocity.for_next_step.x), 0.02);
  EXPECT_NEAR(call.result.velocity.for_next_step.z, 1, 1e-12);
}

TEST(MoveParticle, FloorStartThenTurningBoxBeyondWindow_SecondResponseAtTheSameTime) {
  // tol 0.01 m. The start is 1.05 tol above a fixed floor and approaches it and a box spinning slowly about z whose
  // face kLowX is 1.2 tol away: the start's response reads the floor alone (the box is beyond tol + tol / 16). The
  // first sweep then finds the box at 0 (gap 0.2 tol <= tol / 4): the same time, but a surface that response did not
  // read, so it has its own response, which removes the approach to the box. The floor's gap, 0.05 tol, is beyond
  // the closing bound of 1e-3 s. Without the window test the moment rule took it for the same moment, and the
  // particle kept {0.1, 0, 0} m/s into the box.
  constexpr double kCoarseTolerance = 0.01;  // m
  const double particle_x = 0.3;  // m
  MoveParticleCase scene;
  scene.name = "FloorStartThenTurningBoxBeyondWindow";
  scene.solids = {
      move_particle_cases::BuildFloor(1),
      ToSolid(
          Box{
              .center = {.x = particle_x + 1.2 * kCoarseTolerance + 0.1, .y = 0, .z = 0.25},
              .axes = kIdentity,
              .half_size = {.x = 0.1, .y = 0.1, .z = 0.25},
          },
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 0.01}},
          2
      ),
  };
  scene.inputs = move_particle_cases::BuildCaseInputs({
      .start_position = {.x = particle_x, .y = 0, .z = 1.05 * kCoarseTolerance},
      .velocity = {.x = 0.1, .y = 0, .z = -0.05},
      .step_duration = 1e-3,
      .friction = 0,
  });
  scene.inputs.contact_tolerance = kCoarseTolerance;
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  EXPECT_EQ(call.result.face, SolidFace::kHighZ);
  ExpectVec3Near(call.result.velocity.for_next_step, {.x = 0, .y = 0, .z = 0}, 1e-3);
}

TEST(MoveParticle, TurningBoxBesideFarFloorInEitherOrder_SameMove) {
  // A fixed floor 0.5 m below the particle, beyond the carry's closing bound, before or after the turning box in the
  // array: the same velocities, hit count and position, bit for bit; the record names the box's own index.
  MoveParticleCase box_first = FindMoveParticleCase("TurningBoxHitsRestingParticle");
  box_first.solids.push_back(move_particle_cases::BuildFlatSolid({.x = 0, .y = 0, .z = -0.5}, kUnitZ, kNoMotion, 2));
  MoveParticleCase floor_first = box_first;
  floor_first.solids = {box_first.solids.at(1), box_first.solids.at(0)};
  floor_first.solids.at(0).solid_id = 1;
  floor_first.solids.at(1).solid_id = 2;
  const MoveParticleCall call = ComputeMove(box_first);
  const MoveParticleCall other = ComputeMove(floor_first);
  ASSERT_EQ(call.error, ContactError::kNone);
  ASSERT_EQ(other.error, ContactError::kNone);
  EXPECT_EQ(call.result.solid_index, 0U);
  EXPECT_EQ(other.result.solid_index, 1U);
  EXPECT_EQ(other.result.hit_count, call.result.hit_count);
  CheckVec3Equal(other.result.velocity.for_this_step, call.result.velocity.for_this_step);
  CheckVec3Equal(other.result.velocity.for_next_step, call.result.velocity.for_next_step);
  CheckVec3Equal(other.result.position, call.result.position);
}

// A particle at the rest gap of the +y face of a box 0.8 x 0.02 x 1 m centred at {0, 0, 1} and turning at 2 rad/s
// about z, at x = 0.2 m, leaving the face at 3 m/s along +y for a step of 1 ms, in the domain box from {-1, -1, 0} to
// {1, 1, 2} with the contact tolerance of cells of 0.01 m (about 1.18e-11 m).
MoveParticleCase BuildLeavingTurningFace() {
  constexpr double kStepDuration = 1e-3;  // s
  std::vector<Solid> solids = {
      ToSolid(
          DomainBox{.low_corner = {.x = -1, .y = -1, .z = 0}, .high_corner = {.x = 1, .y = 1, .z = 2}, .closed = {}},
          1
      ),
      ToSolid(
          Box{.center = {.x = 0, .y = 0, .z = 1}, .axes = kIdentity, .half_size = {.x = 0.4, .y = 0.01, .z = 0.5}},
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 2}},
          2
      ),
  };
  const double tolerance = ComputeContactTolerance(ComputeContactScale(solids.data(), 2, kStepDuration, 0.01));  // m
  SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
      .start_position = {.x = 0.2, .y = 0.01 + tolerance, .z = 1},
      .velocity = {.x = 0, .y = 3, .z = 0},
      .step_duration = kStepDuration,
      .friction = 0,
  });
  inputs.contact_tolerance = tolerance;
  return {.name = "LeavingTurningFace", .solids = std::move(solids), .inputs = inputs};
}

TEST(MoveParticle, LeavingTurningFaceKnownChoice_RecordsHitWithPositiveApproachSpeed) {
  // A turning solid has no touching rule: the stepping search hits the particle at 0 whatever its velocity. The face
  // moves at {-0.02, 0.4, 0} m/s under it, so the record says kHit at fraction 0 with approach speed 3 - 0.4 = +2.6
  // m/s, and the response leaves both velocities as they were.
  const MoveParticleCall call = ComputeMove(BuildLeavingTurningFace());
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.hit_fraction, 0.0);
  EXPECT_NEAR(call.result.approach_speed.for_this_step, 2.6, 1e-15);
  EXPECT_NEAR(call.result.approach_speed.for_next_step, 2.6, 1e-15);
  CheckBothVelocitiesNear(call.result, {.x = 0, .y = 3, .z = 0}, 0);
}

TEST(MoveParticle, LeavingTurningFaceKnownChoice_CarryBendsFreeFlight) {
  // The rest of the step is carried in the box's axes: a straight line there, bent by the turn of 2e-3 rad in world
  // axes. The free flight would end at x = 0.2; the carry ends 3e-6 m short of it (measured 0.1999970000039).
  const MoveParticleCase scene = BuildLeavingTurningFace();
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  const Vec3 free_flight_end = scene.inputs.start_position + scene.inputs.velocity.for_this_step * 1e-3;
  EXPECT_NEAR(free_flight_end.x - call.result.position.x, 3e-6, 1e-9);
  EXPECT_NEAR(call.result.position.y, 0.0129999984451, 1e-12);
  EXPECT_EQ(call.result.position.z, 1.0);
}

TEST(MoveParticle, TurningStackedBoxes_SeamIsNearOtherSolid) {
  // Known limit: a turning solid has no bounding surface (method paper <sec-move-particle>), and the carry sweeps
  // nothing, so a particle sliding down two flush boxes that turn together cannot cross their seam. Hit at 0 on the
  // upper box, its carry for 1 s could reach the lower one: too close to call. A step of 0.1 s keeps the lower box
  // beyond the closing bound, and the carry stays on the upper box. At the seam, within tol of both, every step length
  // is too close to call.
  CheckCloseCall(
      ComputeMove("TurningStackedBoxes"),
      ContactCloseCall::kTurningSolidNearOtherSolid,
      1,
      ShorterStep::kCures
  );
  const MoveParticleCall short_step = ComputeMoveWith(
      "TurningStackedBoxes",
      {.position = {.x = 0.5 + kTolerance, .y = 0, .z = 1.8}, .velocity = -kUnitZ, .step_duration = 0.1}
  );
  ASSERT_EQ(short_step.error, ContactError::kNone);
  EXPECT_EQ(short_step.result.hit_count, 1);
  EXPECT_NEAR(short_step.result.position.z, 1.7, kPositionTolerance);
  for (const double step_duration : {0.1, 1e-6}) {
    CheckCloseCall(
        ComputeMoveWith(
            "TurningStackedBoxes",
            {.position = {.x = 0.5 + kTolerance, .y = 0, .z = 1}, .velocity = -kUnitZ, .step_duration = step_duration}
        ),
        ContactCloseCall::kTurningSolidNearOtherSolid,
        1,
        ShorterStep::kDoesNotCure
    );
  }
}

TEST(MoveParticle, SpinningSphereStick_TakesSurfaceVelocity) {
  // The gap 0.5 - tol closes at 1.5 m/s: the hit is at (0.5 - tol) / 1.5 s, at the top, 0.5 + tol above the centre.
  // The surface there moves at {0, 0, 0.5} + {2, 0, 0} x {0, 0, 0.5 + tol} = {0, -(1 + 2 tol), 0.5} m/s, which
  // friction 2 gives the particle (the arm is measured from the centre at the time of the hit). For the 2/3 + 2 tol / 3
  // s left it leaves along the tangent to {0, -2/3 - 2 tol, 1 + tol}.
  const MoveParticleCall call = ComputeMove("SpinningSphereStick");
  ASSERT_EQ(call.error, ContactError::kNone);
  const Vec3 surface_velocity = {.x = 0, .y = -(1 + 2 * kTolerance), .z = 0.5};
  CheckVec3NearRelative(call.result.velocity.for_this_step, surface_velocity, 1);
  CheckVec3NearRelative(call.result.velocity.for_next_step, surface_velocity, 1);
  CheckNearRelative(call.result.hit_fraction, (0.5 - kTolerance) / 1.5, 1);
  EXPECT_EQ(call.result.face, SolidFace::kCurved);
  EXPECT_EQ(call.result.hit_count, 1);
  ExpectVec3Near(
      call.result.position,
      {.x = 0, .y = -2.0 / 3 - 2 * kTolerance, .z = 1 + kTolerance},
      kPositionTolerance
  );
}

TEST(MoveParticle, StartOnSurfaceMovingAway_IsFree) {
  const MoveParticleCall call = ComputeMove("StartOnSurfaceMovingAway");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = 0.5}, kPositionTolerance);
}

TEST(MoveParticle, FollowingSinkingFloor_EndCheckReadsEndPose) {
  const MoveParticleCall call = ComputeMove("FollowingSinkingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  CheckVec3Equal(call.result.position, {.x = 0, .y = 0, .z = -0.5});
}

// --- The reaction ---

TEST(MoveParticle, ReactingSolid_ReactionIsChangeOfNextStepVelocity) {
  const MoveParticleCase test_case = FindMoveParticleCase("ReactingSolid");
  const MoveParticleCall call = ComputeMove(test_case);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidReaction& reaction = call.result.reaction;
  EXPECT_EQ(reaction.solid_index, 0U);
  const Vec3 before = test_case.inputs.velocity.for_next_step;
  CheckVec3Equal(reaction.velocity_for_next_step_change, call.result.velocity.for_next_step - before);
  ExpectVec3Near(reaction.hit_position, {.x = 1 - kTolerance, .y = 0, .z = 0}, kPositionTolerance);
}

TEST(MoveParticle, ReactingDomainCorner_ReactionSumsEveryChange) {
  // Three hits on one reacting solid: {1, 0, 0}, {0, 1, 0} and {0, 0, 1} m/s, each exact, at the first hit's position.
  const MoveParticleCall call = ComputeMove("ReactingDomainCorner");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidReaction& reaction = call.result.reaction;
  EXPECT_EQ(reaction.solid_index, 0U);
  CheckVec3Equal(reaction.velocity_for_next_step_change, {.x = 1, .y = 1, .z = 1});
  ExpectVec3Near(
      reaction.hit_position,
      {.x = kTolerance, .y = 0.1 + kTolerance, .z = 0.2 + kTolerance},
      kPositionTolerance
  );
}

TEST(MoveParticle, ReactingUprightAfterFloor_ReactionCountsOnlyTheReactingSolid) {
  // The floor (index 0) does not react; the upright flat solid (index 1) does: only its change {-2, 0, 0}, at its hit.
  const MoveParticleCall call = ComputeMove("ReactingUprightAfterFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.solid_index, 0U);
  const SolidReaction& reaction = call.result.reaction;
  EXPECT_EQ(reaction.solid_index, 1U);
  CheckVec3Equal(reaction.velocity_for_next_step_change, {.x = -2, .y = 0, .z = 0});
  ExpectVec3Near(reaction.hit_position, {.x = 1 - kTolerance, .y = 0, .z = kTolerance}, kPositionTolerance);
}

TEST(MoveParticle, TwoReactingSolids_IsSecondReactingSolid) {
  // The floor (id 1) is hit first, then the upright flat solid (id 2): the failure names the second reacting solid.
  const MoveParticleCall call = ComputeMove("TwoReactingSolids");
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  CheckFailure(
      call.failure,
      {
          .error = ContactError::kTooCloseToCall,
          .close_call = ContactCloseCall::kSecondReactingSolid,
          .can_shorter_step_cure = true,
          .solid_id = 2,
      }
  );
  CheckUntouched(call.result);
}

TEST(MoveParticle, SecondReactingSolidWithZeroChange_DoesNotConflict) {
  // for_next_step {-1, 0, -2} approaches the reacting floor, which changes it by {0, 0, 2}, and then moves away from
  // the reacting upright, whose response changes it by exactly zero: no second reacting solid.
  const MoveParticleCall call = ComputeMove("SecondReactingSolidWithZeroChange");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  EXPECT_EQ(call.result.reaction.solid_index, 0U);
  CheckVec3Equal(call.result.reaction.velocity_for_next_step_change, {.x = 0, .y = 0, .z = 2});
}

TEST(MoveParticle, ReactingSolidWithZeroChange_LeavesReactionEmpty) {
  const MoveParticleCall call = ComputeMove("ReactingSolidWithZeroChange");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckNoReaction(call.result.reaction);
}

// --- Failures ---

TEST(MoveParticle, NonFiniteStart_IsBadInput) {
  CheckBadInput("NonFiniteStart");
}

TEST(MoveParticle, ZeroStepDuration_IsBadInput) {
  CheckBadInput("ZeroStepDuration");
}

TEST(MoveParticle, NegativeFriction_IsBadInput) {
  CheckBadInput("NegativeFriction");
}

TEST(MoveParticle, ZeroContactTolerance_IsBadInput) {
  CheckBadInput("ZeroContactTolerance");
}

TEST(MoveParticle, NonFiniteVelocityForNextStep_IsBadInput) {
  CheckBadInput("NonFiniteVelocityForNextStep");
}

TEST(MoveParticle, NegativeOverlapTolerance_IsBadInput) {
  CheckBadInput("NegativeOverlapTolerance");
}

TEST(MoveParticle, SolidsWithEqualIds_IsBadInput) {
  CheckBadInput("SolidsWithEqualIds");
}

TEST(MoveParticle, PlateSideWithoutPlateId_IsBadInput) {
  CheckBadInput("PlateSideWithoutPlateId");
}

TEST(MoveParticle, PlateSideWithoutSide_IsBadInput) {
  CheckBadInput("PlateSideWithoutSide");
}

TEST(MoveParticle, SolidsOutOfIdOrder_IsBadInput) {
  CheckBadInput("SolidsOutOfIdOrder");
}

TEST(MoveParticle, PlateSideNamingNoSolid_IsBadInput) {
  CheckBadInput("PlateSideNamingNoSolid");
}

TEST(MoveParticle, PlateSideNamingFlatSolid_IsBadInput) {
  CheckBadInput("PlateSideNamingFlatSolid");
}

TEST(MoveParticle, StartsInsideSecondSolid_FailureNamesItsId) {
  // Inside the floor (index 1, id 9) by 0.01 m: the failure names the id, not the index, with the depth.
  const MoveParticleCall call = ComputeMove("StartsInsideSecondSolid");
  EXPECT_EQ(call.error, ContactError::kStartsInsideSolid);
  CheckFailure(call.failure, {.error = ContactError::kStartsInsideSolid, .solid_id = 9, .magnitude = 0.01});
  CheckUntouched(call.result);
}

TEST(MoveParticle, CeilingClosingFasterThanParticleLeaves_IsSqueezedBetweenSolids) {
  // The ceiling hits at 4/7 s and carries the particle down at 1.75 tol m/s toward the floor, at its rest gap: the
  // floor is an active surface of that hit, and the two close on the particle. No change leaves both; the particle
  // sticks to the ceiling, the surface it approaches fastest, which still approaches the floor: a squeeze naming the
  // floor, the solid the stuck velocity approaches fastest (the record is the ceiling's). Before the floor was read at
  // the hit, its touching rule let the particle sink to z = 0.25 tol, and the end check failed (kEndsTooCloseToSolid).
  CheckCloseCall(
      ComputeMove("CeilingClosingFasterThanParticleLeaves"),
      ContactCloseCall::kSqueezedBetweenSolids,
      1,
      ShorterStep::kCures
  );
}

// A particle 2 tol above the fixed floor (id 1), falling at `speed_share` times the speed whose response bound reaches
// tol / 64 over the step: 2^40 tol / T, with tol 2^-30 m and T 0.1 s, 10240 m/s. The hit comes after tol / speed, so
// the time left is T up to about 1e-13 s.
MoveParticleCase BuildFallingAtShareOfSpeedBound(double speed_share) {
  const double bound_speed = 0x1p40 * kTolerance / 0.1;  // m/s
  return {
      .name = "FallingAtShareOfSpeedBound",
      .solids = {move_particle_cases::BuildFloor(1)},
      .inputs = move_particle_cases::BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 2 * kTolerance},
          .velocity = {.x = 0, .y = 0, .z = -speed_share * bound_speed},
          .step_duration = 0.1,
          .friction = 0,
      }),
  };
}

TEST(MoveParticle, FallingAboveSpeedBound_IsMovesTooFastForStepNamingFloor) {
  // One active surface: the response's allowance times the time left above tol / 64 is a close call, checked in every
  // build, naming that surface's solid. A shorter step cures it.
  CheckCloseCall(
      ComputeMove(BuildFallingAtShareOfSpeedBound(1.01)),
      ContactCloseCall::kMovesTooFastForStep,
      1,
      ShorterStep::kCures
  );
}

TEST(MoveParticle, FallingBelowSpeedBound_HitsAndStops) {
  const MoveParticleCall call = ComputeMove(BuildFallingAtShareOfSpeedBound(0.99));
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.velocity.for_this_step.z, 0.0);
}

TEST(MoveParticle, TurningBoxHoverOnSpinAxis_PassesOnCloseCall) {
  const MoveParticleCall call = ComputeMove("TurningBoxHoverOnSpinAxis");
  EXPECT_EQ(call.error, ContactError::kTooCloseToCall);
  CheckFailure(
      call.failure,
      {
          .error = ContactError::kTooCloseToCall,
          .close_call = ContactCloseCall::kHoverNearTurningSolid,
          .can_shorter_step_cure = true,
          .solid_id = 1,
      }
  );
  CheckUntouched(call.result);
}

// --- The start rules ---

// m. Tells a start moved to the rest gap from one left where it was, tol / 4 or more away; far above the rounding
// of the expected positions, which are below 1 m.
constexpr double kRestGapTolerance = kTolerance / 64;

// m. "At the rest gap" for a move off several surfaces: a distance within tol / 1024 of tol, the stop test of the move.
constexpr double kAtRestGapTolerance = kTolerance / 1024;

// Fails the calling test unless the call failed with the expected failure and left the result unchanged.
void CheckStartFailure(std::string_view case_name, ContactFailure expected) {
  const MoveParticleCall call = ComputeMove(case_name);
  EXPECT_EQ(call.error, expected.error);
  CheckFailure(call.failure, expected);
  CheckUntouched(call.result);
}

TEST(MoveParticle, FlatSolidPushesRestingParticle_CarriesIt) {
  // At the rest gap, d = tol, the start rules leave the start alone; the floor rising at 2 m/s pushes into the
  // particle, so the touching rule makes it a hit at 0: both velocities {0, 0, 2}, carried to z = 0.2 + tol.
  const MoveParticleCall call = ComputeMove("FlatSolidPushesRestingParticle");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_fraction, 0);
  CheckVec3Equal(result.velocity.for_this_step, {.x = 0, .y = 0, .z = 2});
  CheckVec3Equal(result.velocity.for_next_step, {.x = 0, .y = 0, .z = 2});
  EXPECT_EQ(result.approach_speed.for_this_step, -2);
  EXPECT_EQ(result.approach_speed.for_next_step, -2);
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = 0.2 + kTolerance}, kPositionTolerance);
  EXPECT_EQ(result.hit_count, 1);
}

TEST(MoveParticle, FlatSolidPushesParticleWithinTolerance_HitAtStartFromRestGap) {
  // Closer than tol: moved to z = tol, then the approach of -2 m/s makes it a hit at 0, as at the rest gap.
  const MoveParticleCall call = ComputeMove("FlatSolidPushesParticleWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_fraction, 0);
  CheckVec3Equal(result.velocity.for_this_step, {.x = 0, .y = 0, .z = 2});
  EXPECT_EQ(result.approach_speed.for_this_step, -2);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = 0.2 + kTolerance}, kRestGapTolerance);
  EXPECT_EQ(result.hit_count, 1);
}

TEST(MoveParticle, TouchingAndLeaving_IsFree) {
  const MoveParticleCall call = ComputeMove("TouchingAndLeaving");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  EXPECT_EQ(call.result.position.z, kTolerance + 0.1);
}

TEST(MoveParticle, ShallowOverlap_StartedInsideSolidAndMovedOut) {
  // 0.0005 m inside, within the overlap tolerance of 0.001 m: moved to z = tol with the response, at rest.
  const MoveParticleCall call = ComputeMove("ShallowOverlap");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.hit_fraction, 0);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.solid_index, 0U);
  CheckVec3Equal(result.normal_out_of_solid, {.x = 0, .y = 0, .z = 1});
  CheckVec3Equal(result.velocity.for_this_step, {});
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, ShallowOverlapRisingFloor_CarriedFromRestGap) {
  // The floor at its pose of time 0 is the one the start is inside: moved to z = tol, then carried at 1 m/s.
  const MoveParticleCall call = ComputeMove("ShallowOverlapRisingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.approach_speed.for_this_step, -1);
  CheckVec3Equal(result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = 0.1 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DeepOverlap_IsStartsInsideSolid) {
  CheckStartFailure("DeepOverlap", {.error = ContactError::kStartsInsideSolid, .solid_id = 1, .magnitude = 0.01});
}

TEST(MoveParticle, DeepOverlapSinkingFloor_IsStartsInsideSolidAtTimeZero) {
  CheckStartFailure(
      "DeepOverlapSinkingFloor",
      {.error = ContactError::kStartsInsideSolid, .solid_id = 1, .magnitude = 0.01}
  );
}

TEST(MoveParticle, StartsWithinToleranceMovingAway_IsMovedOutToOneTolerance) {
  // Method paper <sec-move-particle>: a silent move to z = tol; nothing else changes.
  const MoveParticleCase test_case = FindMoveParticleCase("StartsWithinToleranceMovingAway");
  const MoveParticleCall call = ComputeMove(test_case);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kNone);
  EXPECT_EQ(result.hit_count, 0);
  EXPECT_EQ(result.hit_fraction, 0);
  EXPECT_EQ(result.solid_index, kNoSolid);
  CheckVec3Equal(result.velocity.for_this_step, test_case.inputs.velocity.for_this_step);
  CheckVec3Equal(result.velocity.for_next_step, test_case.inputs.velocity.for_next_step);
  CheckNoReaction(result.reaction);
  EXPECT_EQ(result.position.x, 0.1);
  EXPECT_EQ(result.position.y, 0);
  EXPECT_NEAR(result.position.z, kTolerance + 0.2, kPositionTolerance);
  // Tighter than kPositionTolerance: the start moved to the rest gap, not left at tol / 4.
  EXPECT_NEAR(result.position.z, kTolerance + 0.2, kRestGapTolerance);
}

TEST(MoveParticle, StartsWithinToleranceOfRisingFloorMovingAway_IsMovedOutSilently) {
  // 1 m/s away from the floor, which rises at 1 m/s: moved silently to tol above the floor at time 0.
  const MoveParticleCall call = ComputeMove("StartsWithinToleranceOfRisingFloorMovingAway");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  CheckVec3Equal(call.result.velocity.for_this_step, {.x = 1, .y = 0, .z = 2});
  EXPECT_EQ(call.result.position.x, 0.1);
  EXPECT_NEAR(call.result.position.z, kTolerance + 0.2, kRestGapTolerance);
}

TEST(MoveParticle, ForNextStepApproachesWithinTolerance_IsHitAtStart) {
  // for_next_step {0, 0, -1} approaches: the hit stops it; for_this_step {1, 0, 1} leaves and comes back unchanged.
  const MoveParticleCall call = ComputeMove("ForNextStepApproachesWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.approach_speed.for_next_step, -1);
  EXPECT_EQ(result.approach_speed.for_this_step, 1);
  CheckVec3Equal(result.velocity.for_next_step, {});
  CheckVec3Equal(result.velocity.for_this_step, {.x = 1, .y = 0, .z = 1});
  ExpectVec3Near(result.position, {.x = 0.1, .y = 0, .z = kTolerance + 0.1}, kRestGapTolerance);
}

TEST(MoveParticle, SlowApproachWithinTolerance_IsHitAtStart) {
  // Any approach of either velocity is a hit at the start, however slow: the fall stops at z = tol.
  const MoveParticleCall call = ComputeMove("SlowApproachWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckVec3Equal(call.result.velocity.for_this_step, {.x = 1, .y = 0, .z = 0});
  CheckVec3Equal(call.result.velocity.for_next_step, {.x = 0, .y = 0, .z = 1});
  ExpectVec3Near(call.result.position, {.x = 0.1, .y = 0, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, StartsWithinToleranceOfSinkingFloorFollowing_ApproachMeasuredFromFloor) {
  // The approach is measured from the moving floor: +1 m/s, leaving, so the move is silent.
  const MoveParticleCall call = ComputeMove("StartsWithinToleranceOfSinkingFloorFollowing");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = kTolerance - 0.1}, kRestGapTolerance);
}

TEST(MoveParticle, FiveFloorsWithinStartOverlap_FastestFloorCarries) {
  // Five parallel floors rising at 1, 1.1, 1.2, 1.3 and 1.4 m/s, the start inside all of them: one joint response over
  // the five, one hit, which no start can exceed any more (method paper <sec-move-particle>). The fastest floor, id 5,
  // pushes: the particle takes its 1.4 m/s, and the record is its own.
  const MoveParticleCall call = ComputeMove("FiveFloorsWithinStartOverlap");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 4U);
  CheckBothVelocitiesNear(call.result, {.x = 0, .y = 0, .z = 1.4}, 1e-15);
}

TEST(MoveParticle, SphereApproachingWithinTolerance_IsHitAtStartOnCurvedFace) {
  // Moved along the normal {-1, 0, 0} to x = -(0.5 + tol), then the sphere's -1 m/s carries the particle.
  const MoveParticleCall call = ComputeMove("SphereApproachingWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.face, SolidFace::kCurved);
  EXPECT_EQ(result.approach_speed.for_this_step, -1);
  CheckVec3Equal(result.velocity.for_this_step, {.x = -1, .y = 0, .z = 0});
  ExpectVec3Near(result.position, {.x = -(0.6 + kTolerance), .y = 0, .z = 0}, kRestGapTolerance);
  EXPECT_EQ(result.hit_count, 1);
}

TEST(MoveParticle, SpinningMovingSphereStartWithinTolerance_ApproachReadAtTimeZero) {
  // The surface velocity at the pose of time 0 has no part along the normal, so the particle leaves: a silent move.
  const MoveParticleCall call = ComputeMove("SpinningMovingSphereStartWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  CheckVec3Equal(call.result.velocity.for_this_step, {.x = 1, .y = 0, .z = 0.1});
  ExpectVec3Near(call.result.position, {.x = 0.1, .y = 0, .z = 0.51 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, TurningBoxStartWithinTolerance_CarriedOnTheTurnAxis) {
  // Moved silently to the rest gap above the top face, on the turn axis, where the face does not move: hit at 0, since
  // a turning solid has no touching rule, then carried in the box's axes, where it stays put. One hit; before the
  // carry, every later sweep hit the box at 0 again, until kTooManyHits.
  const MoveParticleCall call = ComputeMove("TurningBoxStartWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.hit_fraction, 0);
  EXPECT_EQ(call.result.face, SolidFace::kHighZ);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = 0.5 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, CommittedSideInsideThickness_PushedToCommittedSide) {
  // Committed to kAlongNormal, the start at z = -0.001 is at d = -0.011 behind kHighZ: pushed through to 0.01 + tol.
  const MoveParticleCall call = ComputeMove("CommittedSideInsideThickness");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  EXPECT_EQ(result.side_at_hit, SolidSide::kAlongNormal);
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = 0.01 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, UncommittedInsideThickness_PushedToNearerFace) {
  // Not committed, the same start is nearer kLowZ, d = -0.009: pushed to -(0.01 + tol).
  const MoveParticleCall call = ComputeMove("UncommittedInsideThickness");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.face, SolidFace::kLowZ);
  EXPECT_EQ(result.side_at_hit, SolidSide::kAgainstNormal);
  ExpectVec3Near(result.position, {.x = 0, .y = 0, .z = -(0.01 + kTolerance)}, kRestGapTolerance);
}

TEST(MoveParticle, CommittedAgainstNormalInsideThickness_PushedToCommittedSide) {
  // Committed to kAgainstNormal, the start at z = 0.001, nearer kHighZ, is pushed through to -(0.01 + tol).
  const MoveParticleCall call = ComputeMove("CommittedAgainstNormalInsideThickness");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = -(0.01 + kTolerance)}, kRestGapTolerance);
}

TEST(MoveParticle, CommittedSideInsideMovingPlate_PushedOutAndCarried) {
  // The plate rises at 1 m/s: pushed to its kHighZ face at time 0, then carried to 0.11 + tol.
  const MoveParticleCall call = ComputeMove("CommittedSideInsideMovingPlate");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.face, SolidFace::kHighZ);
  CheckVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = 0.11 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, UncommittedInsideMovingPlate_PushedToLowFaceAndLeftBehind) {
  // Pushed to kLowZ at time 0; the plate rises away from the particle, which stays at rest.
  const MoveParticleCall call = ComputeMove("UncommittedInsideMovingPlate");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  EXPECT_EQ(call.result.approach_speed.for_this_step, 1);
  CheckVec3Equal(call.result.velocity.for_this_step, {});
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = -(0.01 + kTolerance)}, kRestGapTolerance);
}

TEST(MoveParticle, CommittedBeyondOtherFace_IsOnWrongSideOfPlate) {
  // z = -0.02 is 0.01 m beyond kLowZ (method paper <sec-move-particle>: -(s * p.z + half_thickness)).
  CheckStartFailure(
      "CommittedBeyondOtherFace",
      {.error = ContactError::kOnWrongSideOfPlate, .solid_id = 5, .magnitude = 0.01}
  );
}

TEST(MoveParticle, CommittedAgainstNormalBeyondOtherFace_IsOnWrongSideOfPlate) {
  CheckStartFailure(
      "CommittedAgainstNormalBeyondOtherFace",
      {.error = ContactError::kOnWrongSideOfPlate, .solid_id = 5, .magnitude = 0.01}
  );
}

TEST(MoveParticle, CommittedBeyondOtherFaceOfMovingPlate_MeasuredAtTimeZero) {
  CheckStartFailure(
      "CommittedBeyondOtherFaceOfMovingPlate",
      {.error = ContactError::kOnWrongSideOfPlate, .solid_id = 5, .magnitude = 0.01}
  );
}

TEST(MoveParticle, CommittedBeyondOtherFaceSmallOverlap_WrongSideComesBeforeDepth) {
  // Method paper <sec-plate-sides>: the committed distance -0.03 m is also deeper than the overlap tolerance of 0.001
  // m.
  CheckStartFailure(
      "CommittedBeyondOtherFaceSmallOverlap",
      {.error = ContactError::kOnWrongSideOfPlate, .solid_id = 5, .magnitude = 0.01}
  );
}

TEST(MoveParticle, CommittedJustBeyondToleranceOfOtherFace_IsOnWrongSideOfPlate) {
  const MoveParticleCall call = ComputeMove("CommittedJustBeyondToleranceOfOtherFace");
  EXPECT_EQ(call.error, ContactError::kOnWrongSideOfPlate);
  EXPECT_EQ(call.failure.solid_id, 5U);
  EXPECT_NEAR(call.failure.magnitude, 1.5 * kTolerance, kTolerance / 1024);
  CheckUntouched(call.result);
}

TEST(MoveParticle, CommittedWithinToleranceOfOtherFace_PushedToCommittedSide) {
  // tol / 2 beyond kLowZ is not the wrong side: d = -(0.02 + tol / 2), pushed through to 0.01 + tol.
  const MoveParticleCall call = ComputeMove("CommittedWithinToleranceOfOtherFace");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.face, SolidFace::kHighZ);
  ExpectVec3Near(call.result.position, {.x = 0, .y = 0, .z = 0.01 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, CornerWithinTolerance_OneHitOverBothSolids) {
  // The floor moves the start to z = tol and stops the fall; the upright, posed at time 0 at x = 1, moves it to
  // x = 1 - tol, and its approach of -1.5 m/s gives its velocity {-0.5, 0, 0}, which carries the particle. One joint
  // response over both: one hit, the record the floor's, the first in array order.
  const MoveParticleCall call = ComputeMove("CornerWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.solid_index, 0U);
  EXPECT_EQ(result.approach_speed.for_this_step, -1);
  CheckVec3Equal(result.velocity.for_this_step, {.x = -0.5, .y = 0, .z = 0});
  ExpectVec3Near(result.position, {.x = 0.95 - kTolerance, .y = 0, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, TwoReactingSolidsWithinTolerance_IsSecondReactingSolid) {
  CheckStartFailure(
      "TwoReactingSolidsWithinTolerance",
      {
          .error = ContactError::kTooCloseToCall,
          .close_call = ContactCloseCall::kSecondReactingSolid,
          .can_shorter_step_cure = false,
          .solid_id = 2,
      }
  );
}

TEST(MoveParticle, InsideUprightAfterApproachingFloor_KindIsStartedInsideSolid) {
  // The record is the floor's, the first in array order; the start inside the upright, a later solid, sets the kind.
  // One joint response over both: one hit.
  const MoveParticleCall call = ComputeMove("InsideUprightAfterApproachingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.solid_index, 0U);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  EXPECT_EQ(result.approach_speed.for_this_step, -2);
  EXPECT_EQ(result.hit_count, 1);
  CheckVec3Equal(result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
  ExpectVec3Near(result.position, {.x = 1 - kTolerance, .y = 0, .z = 0.1 + kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerWithinTolerance_MovedFromBothSidesSilently) {
  // Both near sides move the start to the rest gap: one move along the nearer side alone would leave z = 0.3 tol, and
  // the end check would fail.
  const MoveParticleCall call = ComputeMove("DomainCornerWithinTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = 0.5, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerSecondSideBeyondTolerance_MovedFromNearSideOnly) {
  // side_bottom at 1.5 tol is no surface of the start rules, though the particle approaches it: no hit at the start,
  // and the sweep's gap of 0.5 tol closes in 0.125 s, after the step.
  const MoveParticleCall call = ComputeMove("DomainCornerSecondSideBeyondTolerance");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = 0.5, .z = 1.1 * kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainBoxOpenBack_LeavesThroughIt) {
  const MoveParticleCall call = ComputeMove("DomainBoxOpenBack");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  ExpectVec3Near(call.result.position, {.x = 0.5, .y = 1.1, .z = 0.5}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerAtOpenTop_MovedFromClosedSideOnly) {
  // The open top is no surface: the particle rising through it is not stopped.
  const MoveParticleCall call = ComputeMove("DomainCornerAtOpenTop");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = 0.5, .z = 1.1 - kTolerance / 4}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerSlightlyInside_MovedOutOfBothSides) {
  // The start is inside past side_left, the nearest, and side_bottom: both move it out, and one joint response over
  // both sides stops it: one hit, the record on side_left, the first in face order.
  const MoveParticleCall call = ComputeMove("DomainCornerSlightlyInside");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.face, SolidFace::kLowX);
  EXPECT_EQ(result.hit_count, 1);
  CheckVec3Equal(result.velocity.for_this_step, {});
  ExpectVec3Near(result.position, {.x = kTolerance, .y = 0.5, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, WedgeStartPushedTowardFloor_MovesOffBothAtOnce) {
  // Shortfalls 0.5 tol (floor) and 0.1 tol (wedge), normals -0.8 apart: one move of {-5/6 tol, 0, 0.5 tol}. The fall
  // approaches the floor, which responds; the rising wedge is left. The particle slides along -x at z = tol.
  const MoveParticleCall call = ComputeMove("WedgeStartPushedTowardFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.solid_index, 0U);
  CheckVec3Equal(result.velocity.for_this_step, {.x = -1, .y = 0, .z = 0});
  CheckVec3Equal(result.velocity.for_next_step, {.x = -1, .y = 0, .z = 0});
  ExpectVec3Near(result.position, {.x = -0.1 - 5.0 / 6 * kTolerance, .y = 0, .z = kTolerance}, kAtRestGapTolerance);
}

TEST(MoveParticle, WedgeStartPushedIntoFloor_MovesOffBothAtOnce) {
  // 0.0005 m inside the wedge: one move off both faces at once. The floor is approached, the wedge was started inside:
  // one joint response, one hit, and the kind is kStartedInsideSolid.
  const MoveParticleCall call = ComputeMove("WedgeStartPushedIntoFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(result.hit_count, 1);
  CheckVec3Equal(result.velocity.for_this_step, {.x = -1, .y = 0, .z = 0});
  CheckVec3Equal(result.velocity.for_next_step, {.x = -1, .y = 0, .z = 0});
  const double end_x = -0.1 - 0.0005 * 5.0 / 3 - 5.0 / 6 * kTolerance;  // m
  ExpectVec3Near(result.position, {.x = end_x, .y = 0, .z = kTolerance}, kAtRestGapTolerance);
}

TEST(MoveParticle, ParallelSolidAfterStartResponse_RoundingIsNotAnApproach) {
  // Two parallel solids tol / 4 apart: the move takes the start to the rest gap of the upper, reacting one, and the
  // lower one, 1.25 tol behind it, is not an active surface. So one hit, and the reaction is the real one, the upper
  // solid's push, not rounding: the response leaves an approach of -2^-55 m/s, within its allowance, which the sweep
  // reads as rest on every solid; GeneratedParallelSolids_RoundingNeverHitsTheSecond draws both signs.
  const MoveParticleCall call = ComputeMove("ParallelSolidAfterStartResponse");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.reaction.solid_index, 1U);
  ExpectVec3Near(call.result.reaction.velocity_for_next_step_change, {.x = 1.68, .y = 0, .z = 2.24}, 1e-15);
}

TEST(MoveParticle, FourCoincidentFloorsBesideUpright_OneHit) {
  const MoveParticleCall call = ComputeMove("FourSilentFloorsAndUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 4U);
}

TEST(MoveParticle, FourCoincidentFloorsAndUpright_SlideReadsThemAsOneDirection) {
  // The upright's hit reads the four floors and the upright: the floors are exact copies, so two active surfaces facing
  // two ways. Read one direction per surface, it would be kTooManyHits.
  const MoveParticleCall call = ComputeMove("FourCoincidentFloorsAndUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 4U);
  CheckVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
}

TEST(MoveParticle, FiveCoincidentFloorsBesideUpright_OneHit) {
  // Nothing is listed per solid any more, so a fifth floor beside the upright's start response costs nothing: one hit,
  // the record the upright's, the velocity {0, 0, 1}.
  const MoveParticleCall call = ComputeMove("FiveSilentFloorsAndUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 5U);
  CheckVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 0, .z = 1});
}

TEST(MoveParticle, FiveStackedFloorsLeft_OneSilentMoveNoHit) {
  // Five floors moved off at once are one silent move: no response, no hit, nothing to fail.
  const MoveParticleCall call = ComputeMove("FiveSilentFloors");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
}

TEST(MoveParticle, SlidingFloorThenSlowUpright_ThresholdIsZeroBeforeAnyResponse) {
  // Before any response the approach test compares with 0, so the upright's real approach of 1e-13 m/s beside a floor
  // sliding at 100 m/s is a hit at the start, and the joint response removes it: the solver's no change reads the
  // slack passed, 0, not its own allowance of the 100 m/s (method paper <sec-move-particle>).
  const MoveParticleCall call = ComputeMove("SlidingFloorThenSlowUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.velocity.for_next_step.x, 0);
}

TEST(MoveParticle, SilentFloorThenLeaningUpright_FloorRespondsToThePush) {
  // Method paper <sec-move-particle>: with no hit at the start the move is silent. The upright's hit then pushes the
  // particle into the floor, at its rest gap: the floor is an active surface of that hit, so one joint response over
  // both, one hit, the record the upright's. Before the floor was read at the hit, the push was a drift of 0.05 tol on
  // the floor's plain touching rule, which the settle lifted back.
  const MoveParticleCall call = ComputeMove("SilentFloorThenLeaningUpright");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_NEAR(call.result.position.z, kTolerance, kAtRestGapTolerance);
}

TEST(MoveParticle, AtRestGapSlowApproach_FaceAtRestGapApproachedResponds) {
  // d = tol exactly is outside the start rules' move, d < tol, but a face at the rest gap that a velocity approaches is
  // an active surface: the approach of tol m/s is a hit at 0 and the particle stops at z = tol. Before, the touching
  // rule let it sink by 0.1 tol in the step, which the settle lifted back; in a wedge sharper than 2^-21 rad that
  // sinking left a gap no move could leave.
  const MoveParticleCall call = ComputeMove("AtRestGapSlowApproach");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.velocity.for_this_step.z, 0);
  EXPECT_NEAR(call.result.position.z, kTolerance, kAtRestGapTolerance);
}

TEST(MoveParticle, DomainCornerSlowApproachToSecondSide_IsHitOnThatSide) {
  // Each near side is a surface of its own: side_bottom, not the nearest, is approached and responds; side_left is
  // moved silently.
  const MoveParticleCall call = ComputeMove("DomainCornerSlowApproachToSecondSide");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  CheckVec3Equal(call.result.velocity.for_this_step, {});
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = 0.5, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerNextStepIntoSecondSide_IsHitOnThatSide) {
  const MoveParticleCall call = ComputeMove("DomainCornerNextStepIntoSecondSide");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  EXPECT_EQ(call.result.approach_speed.for_next_step, -1);
  CheckVec3Equal(call.result.velocity.for_next_step, {});
}

TEST(MoveParticle, DomainThreeSideCornerAtRest_MovedFromAllThreeSilently) {
  const MoveParticleCall call = ComputeMove("DomainThreeSideCornerAtRest");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainThreeSideCornerThisStepApproaches_OneHitOverAllThree) {
  // for_this_step {-1, -1, -1} approaches the three sides: one joint response over them, one hit, the record on kLowX,
  // the first in face order.
  const MoveParticleCall call = ComputeMove("DomainThreeSideCornerThisStepApproaches");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowX);
  CheckVec3Equal(call.result.velocity.for_this_step, {});
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainThreeSideCornerNextStepApproaches_OneHitOverAllThree) {
  const MoveParticleCall call = ComputeMove("DomainThreeSideCornerNextStepApproaches");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckVec3Equal(call.result.velocity.for_next_step, {});
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainThreeSideCornerSlightlyInside_MovedOutOfAllThree) {
  const MoveParticleCall call = ComputeMove("DomainThreeSideCornerSlightlyInside");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowX);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kRestGapTolerance);
}

TEST(MoveParticle, DomainCornerNearestSideComesLater_RecordNamesTheSideHit) {
  // The fall approaches side_bottom only, so the record names kLowZ, the side read before the move, even though the
  // sides tie at tol after it and a tie keeps kLowX.
  const MoveParticleCall call = ComputeMove("DomainCornerNearestSideComesLater");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowZ);
  CheckVec3Equal(call.result.normal_out_of_solid, {.x = 0, .y = 0, .z = 1});
}

// --- A start near several surfaces at once (method paper <sec-move-particle>) ---

// Fails the calling test unless every solid of the inputs is at the rest gap of position at its end pose: a distance
// within tol / 1024 of tol, by ComputeDistanceToSolid itself.
void CheckAtRestGapOfEverySolid(const SolidContactInputs& inputs, Vec3 position) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    const double distance =
        ComputeDistanceToSolid(solid, ComputeSolidPose(solid, inputs.step_duration), position).distance;
    EXPECT_NEAR(distance, kTolerance, kAtRestGapTolerance) << "solid index " << solid_index;
  }
}

// Runs MoveParticle on the scene of that name and fails the calling test unless it succeeded at the expected end, with
// every solid at the rest gap.
SolidContactResult CheckEndAtRestGap(std::string_view case_name, Vec3 expected_end, double position_tolerance) {
  const MoveParticleCase scene = FindMoveParticleCase(case_name);
  const MoveParticleCall call = ComputeMove(scene);
  EXPECT_EQ(call.error, ContactError::kNone) << case_name;
  ExpectVec3Near(call.result.position, expected_end, position_tolerance);
  CheckAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
  return call.result;
}

TEST(MoveParticle, SixtyDegreeWedgeRestingOnBoth_EndsAtRestGapOfBoth) {
  // One move of tol / sin(30 degrees) = 2 tol along the bisector. Moving off the floor, then off the other face, ended
  // 0.8125 tol from the floor.
  const SolidContactResult result = CheckEndAtRestGap(
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
  const SolidContactResult result = CheckEndAtRestGap(
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
      CheckEndAtRestGap("TenDegreeWedgeFallingOnBoth", {.x = rest_x, .y = 0, .z = kTolerance}, kAtRestGapTolerance);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.solid_index, 0U);
  CheckVec3Equal(result.velocity.for_this_step, {});
  CheckVec3Equal(result.velocity.for_next_step, {});
}

TEST(MoveParticle, VerySharpFlatWedgeRestingOnBoth_EndsAtRestGapOfBoth) {
  // A wedge of 0.01 degrees: the rest point is tol / tan(0.005 degrees) = 11459.156 tol out. A limit of 0.4476 degrees
  // on the wedge angle would fail it.
  const double rest_x = kTolerance / std::tan(0.005 * std::numbers::pi / 180);  // m
  const SolidContactResult result =
      CheckEndAtRestGap("VerySharpFlatWedgeRestingOnBoth", {.x = rest_x, .y = 0, .z = kTolerance}, kTolerance / 256);
  EXPECT_EQ(result.kind, ContactKind::kNone);
}

TEST(MoveParticle, FlatWedgeNeedingMoveBeyondCap_IsStartsInNarrowGap) {
  // The move cap is 2^22 tol = 4.19e6 tol. A wedge of 6e-7 rad rests the particle 3333942 tol out, within 2e-4 of
  // tol / tan(3e-7); one of 3e-7 rad would need 6.67e6 tol, beyond the cap.
  const MoveParticleCase within_cap = FindMoveParticleCase("FlatWedgeNeedingMoveWithinCap");
  const MoveParticleCall within_call = ComputeMove(within_cap);
  ASSERT_EQ(within_call.error, ContactError::kNone);
  const double rest_x = kTolerance / std::tan(3e-7);  // m
  EXPECT_NEAR(within_call.result.position.x / rest_x, 1, 2e-4);
  CheckAtRestGapOfEverySolid(ToSolidContactInputs(within_cap), within_call.result.position);
  CheckStartFailure("FlatWedgeNeedingMoveBeyondCap", {.error = ContactError::kStartsInNarrowGap, .solid_id = 2});
}

TEST(MoveParticle, ThreeSolidAcuteCornerFallingOnAll_EndsAtRestGapOfAll) {
  // Three faces at 60 degrees to each other: one move of Sqrt(7) tol to the point at tol from all three. The fall
  // approaches the floor, which stops it.
  const double root_three = std::numbers::sqrt3;
  const SolidContactResult result = CheckEndAtRestGap(
      "ThreeSolidAcuteCornerFallingOnAll",
      {.x = root_three * kTolerance, .y = root_three * kTolerance, .z = kTolerance},
      kAtRestGapTolerance
  );
  EXPECT_EQ(result.hit_count, 1);
  CheckVec3Equal(result.velocity.for_this_step, {});
}

TEST(MoveParticle, ThirdSolidNearOnlyAfterFirstMove_JoinsNextIteration) {
  // The third face is 1.414 tol from the start, so it does not constrain the first iterate, which leaves it 0.19 tol
  // away; it constrains the next one: a move of 2.3052 tol in all.
  const SolidContactResult result = CheckEndAtRestGap(
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
  const MoveParticleCall far_call = ComputeMove("DomainCornerCutBySolidFarEnough");
  ASSERT_EQ(far_call.error, ContactError::kNone);
  ExpectVec3Near(far_call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kAtRestGapTolerance);
  EXPECT_EQ(far_call.result.hit_count, 1);
  CheckStartFailure("DomainCornerCutBySolidTooNear", {.error = ContactError::kStartsInNarrowGap, .solid_id = 2});
}

TEST(MoveParticle, ParallelFacesLessThanTwoTolerancesApart_IsStartsInNarrowGap) {
  // 1.5 tol apart, no point is at tol from both. 2.5 tol apart, the particle moves off the floor alone.
  CheckStartFailure(
      "ParallelFacesLessThanTwoTolerancesApart",
      {.error = ContactError::kStartsInNarrowGap, .solid_id = 2}
  );
  const MoveParticleCall wide_call = ComputeMove("ParallelFacesMoreThanTwoTolerancesApart");
  ASSERT_EQ(wide_call.error, ContactError::kNone);
  EXPECT_EQ(wide_call.result.kind, ContactKind::kNone);
  CheckVec3Equal(wide_call.result.position, {.x = 0, .y = 0, .z = kTolerance});
}

TEST(MoveParticle, BallRestingOnFloorStartInCusp_MovesToRestPoint) {
  // The rest point of a ball of radius R resting on a floor is 2 Sqrt(R tol) from the contact, both distances exactly
  // tol: {46340.95 tol, 0, tol} for R = 0.5. The first iterate overshoots out along the floor; the next come back.
  const MoveParticleCase scene = FindMoveParticleCase("BallRestingOnFloorStartInCusp");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_NEAR(call.result.position.x, 2 * std::sqrt(0.5 * kTolerance), kTolerance / 256);
  EXPECT_NEAR(call.result.position.y, 0, kTolerance / 256);
  EXPECT_NEAR(call.result.position.z, kTolerance, kAtRestGapTolerance);
  CheckAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
}

TEST(MoveParticle, BallRestingOnFloorStartJustShortOfFloor_MovesToRestPoint) {
  // The first iterate moves 0.014 tol, less than tol / 64, but leaves the ball 0.0124 tol short: the iterates stop only
  // at a point at the rest gap of every solid, here the rest point.
  const SolidContactResult result = CheckEndAtRestGap(
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
    const MoveParticleCall call = ComputeMove(scene);
    ASSERT_EQ(call.error, ContactError::kNone) << case_name;
    EXPECT_NEAR(call.result.position.x / rest_x, 1, 1e-4) << case_name;
    CheckAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
  }
  CheckStartFailure(
      "BallRestingOnFloorDeepStartNearContact",
      {.error = ContactError::kStartsInNarrowGap, .solid_id = 2}
  );
}

TEST(MoveParticle, TwoBallsTouchingStartInNeck_MovesToRestRing) {
  // Two balls of radius R touching: the ring where both are at tol is Sqrt(tol (2 R + tol)) = 32768.0 tol from the
  // contact for R = 0.5.
  const double ring_radius = std::sqrt(kTolerance * (1 + kTolerance));  // m
  const SolidContactResult result =
      CheckEndAtRestGap("TwoBallsTouchingStartInNeck", {.x = 0.5, .y = ring_radius, .z = 0}, kTolerance / 256);
  EXPECT_EQ(result.kind, ContactKind::kNone);
}

TEST(MoveParticle, PlateAndSphereRing_MovesOffBothAlongBisector) {
  // The plate and the ball meet at 60 degrees on their ring: the move is 1.5 tol along the bisector, the plate is then
  // at the rest gap and the curved ball slightly beyond it. The fall approaches the plate, which stops it.
  const MoveParticleCase scene = FindMoveParticleCase("PlateAndSphereRing");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckVec3Equal(call.result.velocity.for_this_step, {});
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
  const MoveParticleCall call = ComputeMove("ParallelSolidAfterStartResponseNextStepMoving");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.reaction.solid_index, 1U);
  ExpectVec3Near(call.result.reaction.velocity_for_next_step_change, {.x = 1.68, .y = 0, .z = 2.24}, 1e-15);
}

TEST(MoveParticle, SlidingFloorThenSlowUprightNextStepOnly_NextStepThresholdIsZeroBeforeAnyResponse) {
  // Before any response for_next_step's threshold is 0 too: its approach of 1e-13 m/s to the upright is a hit at the
  // start.
  const MoveParticleCall call = ComputeMove("SlidingFloorThenSlowUprightNextStepOnly");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.velocity.for_next_step.x, 0);
}

TEST(MoveParticle, PlateAcrossSharpWedge_IsNotCrossedByTheMove) {
  // The pair of faces would move the start 3819.7 tol out, through the plate, whose near face is 100 tol out. The
  // crossing check makes the plate constrain the move, and no point between the apex and the plate is at tol from all
  // three. Without it the particle ends 2520 tol beyond the plate's far face.
  CheckStartFailure("PlateAcrossSharpWedge", {.error = ContactError::kStartsInNarrowGap, .solid_id = 3});
}

TEST(MoveParticle, SecondFaceNearOnlyAfterFloorMove_MovesOffBoth) {
  // Only the floor is near, but its own move brings the 60 degree face to 0.95 tol: that move is not kept, and the move
  // off several surfaces ends at the rest gap of both.
  const SolidContactResult result = CheckEndAtRestGap(
      "SecondFaceNearOnlyAfterFloorMove",
      {.x = std::numbers::sqrt3 * kTolerance, .y = 0, .z = kTolerance},
      kAtRestGapTolerance
  );
  EXPECT_EQ(result.kind, ContactKind::kNone);
}

TEST(MoveParticle, StartInsideFloorBelowPlate_IsNotCrossedByTheMove) {
  // The floor's own move to its rest gap would pass through the plate inside it, so it is not kept; with the plate
  // constraining the move no point is at the rest gap of both, and the plate is named.
  CheckStartFailure("StartInsideFloorBelowPlate", {.error = ContactError::kStartsInNarrowGap, .solid_id = 2});
}

TEST(MoveParticle, DeepCuspBesideWall_MovesToRestPoint) {
  // The first iterate from 0.1 x* overshoots far out along the floor and crosses an upright 1.5 x* out. The rest point
  // is 0.5 x*, about 10^6 tol, from the wall: the move must still reach it.
  const MoveParticleCase scene = FindMoveParticleCase("BallRestingOnFloorDeepStartBesideWall");
  const MoveParticleCall call = ComputeMove(scene);
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
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  CheckAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
}

TEST(MoveParticle, DeepCuspWallWithinRoundingOfRestPoint_KnownLimitIsStartsInNarrowGap) {
  // A known limit, beside the start near the contact point of BallRestingOnFloorDeepStart: the rest point is 1.01 tol
  // from the wall, inside the window where the ball's rounded distance, ulp(R) / (2 Sqrt(tol / R)) = about 128 tol at R
  // = 1024 m, moves the linearised cusp's vertex past the wall, and no subset of the three targets has an answer. A
  // solver that tried targets relaxed by the slack would find it; this test then changes to a success.
  CheckStartFailure(
      "BallRestingOnFloorDeepStartWallNearRestPoint",
      {.error = ContactError::kStartsInNarrowGap, .solid_id = 3}
  );
}

TEST(MoveParticle, CoincidentFloors_SixteenFitSeventeenIsStartsInNarrowGap) {
  // Surfaces with the same normal share one target, but one move tracks at most kMaxTrackedSurfaces = 16 surfaces.
  const MoveParticleCall sixteen_call = ComputeMove("SixteenCoincidentFloors");
  ASSERT_EQ(sixteen_call.error, ContactError::kNone);
  CheckVec3Equal(sixteen_call.result.position, {.x = 0, .y = 0, .z = kTolerance});
  CheckStartFailure("SeventeenCoincidentFloors", {.error = ContactError::kStartsInNarrowGap, .solid_id = 17});
}

TEST(MoveParticle, DomainCornerWithCoincidentGroundAndWall_EndsAtCorner) {
  // A ground lying on side_bottom and an upright lying on side_left make five near surfaces, but only three directions:
  // coincident surfaces share one target, and {tol, tol, tol} is at the rest gap of all five.
  const MoveParticleCall call = ComputeMove("DomainCornerWithCoincidentGroundAndWall");
  ASSERT_EQ(call.error, ContactError::kNone);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kAtRestGapTolerance);
}

TEST(MoveParticle, DomainCornerWithNearlyCoincidentGroundAndWall_EndsAtCorner) {
  // Normals that differ by rounding share one target too, so the five surfaces still leave {tol, tol, tol}.
  const MoveParticleCall call = ComputeMove("DomainCornerWithNearlyCoincidentGroundAndWall");
  ASSERT_EQ(call.error, ContactError::kNone);
  ExpectVec3Near(call.result.position, {.x = kTolerance, .y = kTolerance, .z = kTolerance}, kAtRestGapTolerance);
}

TEST(MoveParticle, CuspStartIntoBallOnly_ResponseUsesNormalAtMovedStart) {
  // In the cusp, halfway to the rest point, moving at 1 m/s straight into the ball and away from the floor. The
  // response reads the ball's normal at the moved start (method paper <sec-move-particle>), the normal the sweep that
  // follows reads, turned from the one at the start as given by about 2 Sqrt(tol / R) (1 - f): it leaves 4.3e-5 m/s
  // along the ball, and the particle slides on to 46804 tol from the contact.
  const MoveParticleCall call = ComputeMove("CuspStartIntoBallOnly");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {.x = 4.31583759e-05, .y = 0, .z = 3.72529e-09}, 1e-9);
  ExpectVec3Near(call.result.position, {.x = 46804.36 * kTolerance, .y = 0, .z = 1.04 * kTolerance}, 0.01 * kTolerance);
}

// --- The slide along several surfaces (method paper <sec-move-particle>) ---

// Runs MoveParticle on the scene of that name, and fails the calling test unless it succeeded with that many hits, at
// rest, at the expected end within position_tolerance, with every solid at the rest gap.
SolidContactResult
CheckStopAtRestGap(std::string_view case_name, int hit_count, Vec3 expected_end, double position_tolerance) {
  const SolidContactResult result = CheckEndAtRestGap(case_name, expected_end, position_tolerance);
  EXPECT_EQ(result.hit_count, hit_count) << case_name;
  CheckBothVelocitiesNear(result, {}, 1e-15);
  return result;
}

TEST(MoveParticle, AcuteWedge_StopsWithoutError) {
  // The 45 degree wedge's face is hit at x = 1 - (1 + Sqrt(2)) tol, where the particle also rests on the floor: one
  // joint response over both stops it at the apex, one hit.
  const MoveParticleCase scene = FindMoveParticleCase("AcuteWedge");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {}, 1e-15);
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
  const MoveParticleCall call = ComputeMove("RightAngleCorner");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  CheckBothVelocitiesNear(call.result, {.x = 0, .y = 1, .z = 0}, 1e-15);
  EXPECT_NEAR(call.result.position.x, 1 - kTolerance, kPositionTolerance);
  EXPECT_NEAR(call.result.position.z, kTolerance, kPositionTolerance);
}

TEST(MoveParticle, SlideRemovalOnReactingFloor_CountsInReaction) {
  // The floor at 0.1 - tol / 2 s, at {0.6 - tol / 2, 0, tol}: {1, 0, 0}, a change of {0, 0, 2}. The upright sliding
  // down along itself at 1 m/s is hit at 0.5 - tol s: approach -1, sideways speed 1 against it, and friction 0.5 leaves
  // {0, 0, -0.5}, into the floor. The change removes that approach, {0, 0, 0.5}, charged to the floor; the upright is
  // not approached again. Every value is a binary fraction: without the change the reaction is {0, 0, 2}.
  const MoveParticleCall call = ComputeMove("SlideRemovalOnReactingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  CheckBothVelocitiesNear(call.result, {}, 0);
  ExpectVec3Near(call.result.position, {.x = 1 - kTolerance, .y = 0, .z = kTolerance}, kPositionTolerance);
  EXPECT_EQ(call.result.reaction.solid_index, 0U);
  CheckVec3Equal(call.result.reaction.velocity_for_next_step_change, {.x = 0, .y = 0, .z = 2.5});
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
  const MoveParticleCall call = ComputeMove("AcuteWedgeOnReactingFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {}, 1e-15);
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
  // (solid_contact_types.response.test.cc), rounding within its allowance of 5.7e-14 m/s; the particle slides along it
  // to the upright, reached at 0.625 s at z = 0.15, and the upright's response leaves {0, 0, 0.24}, which leaves the
  // first solid. It rises along the upright to z = 0.24. A stop at the first hit would end near z = 0.043.
  const MoveParticleCall call = ComputeMove("ObliqueCornerResidue");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  CheckBothVelocitiesNear(call.result, {.x = 0, .y = 0, .z = 0.24}, 1e-15);
  EXPECT_NEAR(call.result.position.x, -0.2 + kTolerance, kPositionTolerance);
  EXPECT_NEAR(call.result.position.z, 0.24, 1e-3);
}

TEST(MoveParticle, GrooveSlideAlongEdge_KeepsSpeedAlongEdge) {
  // Both faces of the 60 degree wedge are active at the start: one joint response keeps the part along the common line.
  // A rule that took the velocity of the solid hit last, one surface after the other, would give {0, 0, 0}.
  const SolidContactResult result = CheckEndAtRestGap(
      "GrooveSlideAlongEdge",
      {.x = std::numbers::sqrt3 * kTolerance, .y = 0.01, .z = kTolerance},
      kTolerance / 1024
  );
  EXPECT_EQ(result.hit_count, 1);
  CheckBothVelocitiesNear(result, {.x = 0, .y = 1, .z = 0}, 1e-15);
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
bool ComputeGravityStep(SolidContactInputs* inputs, SolidContactResult* result) {
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

// Runs 1000 gravity steps (ComputeGravityStep) from the inputs and fails the calling test unless every step succeeds
// and every step from expected.rest_step on rests as expected (ComputeIsRestingAsExpected). Returns the last result.
SolidContactResult CheckRestsForThousandSteps(SolidContactInputs inputs, const RestExpectation& expected) {
  SolidContactResult result;
  int failed_steps = 0;
  int steps_not_at_rest = 0;
  for (int step = 1; step <= 1000; ++step) {
    failed_steps += ComputeGravityStep(&inputs, &result) ? 0 : 1;
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
void CheckGrooveRestsAtApex(std::string_view case_name, double friction, Vec3 apex_point) {
  const MoveParticleCase scene = FindMoveParticleCase(case_name);
  SolidContactInputs inputs = ToSolidContactInputs(scene);
  inputs.friction = friction;
  const RestExpectation at_apex = {
      .rest_step = 12,
      .hit_count = 1,
      .rest_point = apex_point,
      .tolerance = kTolerance / 1024,
  };
  const SolidContactResult result = CheckRestsForThousandSteps(inputs, at_apex);
  CheckAtRestGapOfEverySolid(inputs, result.position);
  CheckBothVelocitiesNear(result, {}, 1e-15);
}

TEST(MoveParticle, VGrooveSlideIntoApex_ComesToRestThere) {
  // The 30 degree groove: the apex point is tol / sin(15 degrees) = 3.8637033 tol above the line where the faces meet.
  // The model reaches it in step 10 with friction 0 and in step 11 with 0.5.
  const Vec3 apex_point = {.x = 0, .y = 0, .z = kTolerance / std::sin(15 * std::numbers::pi / 180)};  // m
  CheckGrooveRestsAtApex("VGroove", 0, apex_point);
  CheckGrooveRestsAtApex("VGroove", 0.5, apex_point);
}

TEST(MoveParticle, TenDegreeVGrooveSlideIntoApex_ComesToRestThere) {
  // The 10 degree groove: tol / sin(5 degrees) = 11.473713 tol.
  const Vec3 apex_point = {.x = 0, .y = 0, .z = kTolerance / std::sin(5 * std::numbers::pi / 180)};  // m
  CheckGrooveRestsAtApex("TenDegreeVGroove", 0, apex_point);
  CheckGrooveRestsAtApex("TenDegreeVGroove", 0.5, apex_point);
}

TEST(MoveParticle, NarrowWedgeAtRestPoint_KeepsSpeedAlongEdge) {
  // A wedge of 2e-6 rad, 1 + c = 2e-12, above the velocity pair limit of 2^-40: one joint response over the side and
  // the floor keeps the part along the common line.
  const MoveParticleCase scene = FindMoveParticleCase("NarrowWedgeAtRestPoint");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {.x = 0, .y = 1, .z = 0}, 1e-9);
  CheckAtRestGapOfEverySolid(ToSolidContactInputs(scene), call.result.position);
}

// Runs the sliver scene with both velocities set to velocity and the reacting solid chosen, and fails the calling test
// unless it succeeds with one hit, both velocities {0, 0, 0} exactly, at the rest point. Returns the result.
SolidContactResult CheckSliverStop(Vec3 velocity, bool is_side_reacting) {
  const double angle = 1e-7;  // rad
  MoveParticleCase scene = FindMoveParticleCase("SliverStop");
  scene.solids[0].reports_reaction = !is_side_reacting;
  scene.solids[1].reports_reaction = is_side_reacting;
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  const MoveParticleCall call = ComputeMove(scene);
  EXPECT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {}, 0);
  const double rest_x = kTolerance / std::tan(angle / 2);  // m
  ExpectVec3Near(call.result.position, {.x = rest_x, .y = 0, .z = kTolerance}, kTolerance / 1024);
  return call.result;
}

TEST(MoveParticle, SliverStop_ChargedToSolidApproachedFastest) {
  // A wedge of 1e-7 rad, 1 + c = 5e-15, below the pair limit: a sliver, so no change is found and the particle sticks
  // to the surface it approaches fastest (method paper <sec-move-particle>). Moving at {-2, 1, 0} that is the side,
  // which does not react: no reaction, the record the side's. Moving at {-2, 1, -0.001} it is the reacting floor,
  // charged the whole stop {2, -1, 0.001}. With the side reacting instead, {-2, 1, 0} charges the side {2, -1, 0}.
  const SolidContactResult into_side = CheckSliverStop({.x = -2, .y = 1, .z = 0}, false);
  EXPECT_EQ(into_side.solid_index, 1U);
  CheckNoReaction(into_side.reaction);
  const SolidContactResult into_floor = CheckSliverStop({.x = -2, .y = 1, .z = -0.001}, false);
  EXPECT_EQ(into_floor.solid_index, 0U);
  EXPECT_EQ(into_floor.reaction.solid_index, 0U);
  CheckVec3Equal(into_floor.reaction.velocity_for_next_step_change, {.x = 2, .y = -1, .z = 0.001});
  const SolidContactResult reacting_side = CheckSliverStop({.x = -2, .y = 1, .z = 0}, true);
  EXPECT_EQ(reacting_side.reaction.solid_index, 1U);
  CheckVec3Equal(reacting_side.reaction.velocity_for_next_step_change, {.x = 2, .y = -1, .z = 0});
}

TEST(MoveParticle, ThreeSolidAcuteCornerInto_StopsInCorner) {
  // The three faces are hit at one moment: one joint response, one hit.
  CheckStopAtRestGap(
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
  const SolidContactResult result = CheckStopAtRestGap(
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
  CheckCloseCall(
      ComputeMove("CornerPushOnSecondReactingSolid"),
      ContactCloseCall::kSecondReactingSolid,
      2,
      ShorterStep::kDoesNotCure
  );
  const MoveParticleCall call = ComputeMove("CornerPushOnNonReactingSide");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {}, 1e-15);
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
  // rests on the floor too: one joint response over both stops it, one hit. A limit of 0.4476 degrees on the wedge
  // angle would fail it.
  const double rest_x = 2 * std::sqrt(0.5 * kTolerance);  // m
  CheckStopAtRestGap("BallRestingOnFloorSlidingIntoCusp", 1, {.x = rest_x, .y = 0, .z = kTolerance}, kTolerance / 256);
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
  CheckRestsForThousandSteps(ToSolidContactInputs(scene), at_rest_point);
}

TEST(MoveParticle, TwoBallsTouchingSlidingIntoNeck_ComesToRest) {
  // The neck ring Sqrt(tol (2 R + tol)) = 32768.0 tol from the contact, where both balls are at tol: one hit there.
  const double ring_radius = std::sqrt(kTolerance * (1 + kTolerance));  // m
  CheckStopAtRestGap("TwoBallsTouchingSlidingIntoNeck", 1, {.x = 0.5, .y = ring_radius, .z = 0}, kTolerance / 256);
}

TEST(MoveParticle, PlateAndSphereSlidingIn_ComesToRestAtRing) {
  // The sphere is hit at (0.5 - Sqrt(0.1875) - Sqrt(3) tol) / 0.1 of the step, where the particle is tol from it and
  // from the plate: one joint response over both, one hit. Both normals at the ring are 120 degrees apart, so the rest
  // point is 2 tol out along their bisector {Sqrt(3) / 2, 0, 1 / 2}.
  const Vec3 ring_point = {.x = std::sqrt(0.1875), .y = 0, .z = 0.01};
  const Vec3 bisector = {.x = std::numbers::sqrt3 / 2, .y = 0, .z = 0.5};
  const SolidContactResult result =
      CheckStopAtRestGap("PlateAndSphereSlidingIn", 1, ring_point + 2 * kTolerance * bisector, kTolerance / 1024);
  EXPECT_EQ(result.solid_index, 1U);
  EXPECT_EQ(result.face, SolidFace::kCurved);
  EXPECT_NEAR(result.hit_fraction, (0.5 - std::sqrt(0.1875) - std::numbers::sqrt3 * kTolerance) / 0.1, 1e-9);
}

TEST(MoveParticle, CylinderLyingOnFloorSlidingIntoCusp_ComesToRestThere) {
  // The cross-section of BallRestingOnFloorSlidingIntoCusp: the wall is hit at the rest point 2 Sqrt(0.5 tol) from the
  // contact line, where the particle rests on the floor too: one joint response over both stops it, one hit.
  const double rest_y = 2 * std::sqrt(0.5 * kTolerance);  // m
  CheckStopAtRestGap(
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
  CheckStopAtRestGap("CylinderSunkIntoFloorSlidingIn", 1, rest_point, kTolerance / 1024);
}

// --- Spheres and cylinders alone ---

// The distance from a point to the first solid of a scene at its pose at the end of the step.
double ComputeEndDistance(const MoveParticleCase& scene, Vec3 point) {
  const Solid& solid = scene.solids.at(0);
  return ComputeDistanceToSolid(solid, ComputeSolidPose(solid, scene.inputs.step_duration), point).distance;
}

TEST(MoveParticle, SphereHeadOn_StopsOneToleranceShort) {
  // The grown sphere is reached at x = -(0.5 + tol), where the normal is {-1, 0, 0} and the whole speed is removed.
  const MoveParticleCall call = ComputeMove("SphereHeadOn");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.face, SolidFace::kCurved);
  EXPECT_EQ(result.hit_count, 1);
  ExpectVec3Near(result.position, {.x = -(0.5 + kTolerance), .y = 0, .z = 0}, kPositionTolerance);
  CheckBothVelocitiesNear(result, {}, 1e-15);
  EXPECT_EQ(result.approach_speed.for_next_step, -1);
  EXPECT_EQ(result.approach_speed.for_this_step, -1);
}

TEST(MoveParticle, SphereOblique_SlidesOffAlongTheTangent) {
  // The hit is at {-0.4, 0.3, 0} times 1 + 2 tol, with the normal {-0.8, 0.6, 0}: the response leaves
  // {0.36, 0.48, 0}, which the 0.4 s left carry to {-0.256, 0.492, 0}. The sphere is convex, so the tangent path does
  // not meet it again: one hit.
  const MoveParticleCall call = ComputeMove("SphereOblique");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.hit_count, 1);
  CheckBothVelocitiesNear(result, {.x = 0.36, .y = 0.48, .z = 0}, 1e-8);
  ExpectVec3Near(result.position, {.x = -0.256, .y = 0.492, .z = 0}, 1e-8);
  const double grown_radius = 0.5 + kTolerance;  // m
  CheckNearRelative(result.hit_fraction, (2 - std::sqrt(grown_radius * grown_radius - 0.09)) / 2, 1);
}

TEST(MoveParticle, SphereTangentWithinRounding_NeverEndsInside) {
  // The path runs tol beyond the sphere's surface, along the grown sphere's equator up to rounding: a hit or not, the
  // particle keeps its speed along x and ends outside.
  const MoveParticleCase scene = FindMoveParticleCase("SphereTangentWithinRounding");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_GE(ComputeEndDistance(scene, call.result.position), kTolerance / 2);
  EXPECT_LE(std::abs(call.result.velocity.for_this_step.x - 1), 0x1p-40);
}

TEST(MoveParticle, SphereStartWithinToleranceMovingAway_MovedOutSilently) {
  // Method paper <sec-move-particle>: moved along the normal {-1, 0, 0} to the rest gap; the velocity along the surface
  // does not approach, so the move is silent.
  const MoveParticleCase scene = FindMoveParticleCase("SphereStartWithinToleranceMovingAway");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  CheckVec3Equal(call.result.velocity.for_next_step, {.x = 0, .y = 1, .z = 0});
  CheckVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 1, .z = 0});
  EXPECT_GE(ComputeEndDistance(scene, call.result.position), kTolerance);
}

TEST(MoveParticle, CylinderRim_HitAtItsTrueDistance) {
  // The cylinder grown with sharp edges is entered at {0.5 + tol / 2, 0, 1 + tol}, Sqrt(1.25) tol from the rim; the rim
  // search goes on down the path to tol from the rim, Sqrt(0.75) tol above the cap's plane, where the normal is
  // {1 / 2, 0, Sqrt(3) / 2} and the response leaves {0, 0, -1} + (Sqrt(3) / 2) {1 / 2, 0, Sqrt(3) / 2} =
  // {Sqrt(3) / 4, 0, -1 / 4} (the sharp corner's hit would give {0.4, 0, -0.2}). Nothing moves the hit: the reaction is
  // recorded on the path.
  const MoveParticleCase scene = FindMoveParticleCase("CylinderRim");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.face, SolidFace::kHighRim);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_NEAR(result.hit_fraction, (2 - std::sqrt(0.75) * kTolerance) / 4, kTolerance / 4096);
  CheckBothVelocitiesNear(result, {.x = std::numbers::sqrt3 / 4, .y = 0, .z = -0.25}, 1e-6);
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
  const MoveParticleCall call = ComputeMove("CylinderRimFromTheSide");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kHighRim);
  CheckBothVelocitiesNear(call.result, {.x = -0.25, .y = 0, .z = std::numbers::sqrt3 / 4}, 1e-6);
  EXPECT_NEAR(call.result.reaction.hit_position.x, 0.5 + std::sqrt(0.75) * kTolerance, kTolerance / 512);
}

TEST(MoveParticle, CylinderCap_SlidesOnThePlane) {
  // The cap's grown plane z = 1 + tol is reached at 2 - tol s, at x = 0.3; the response keeps {0.1, 0, 0}, which the
  // 1 + tol s left carry to x = 0.4.
  const MoveParticleCall call = ComputeMove("CylinderCap");
  ASSERT_EQ(call.error, ContactError::kNone);
  const SolidContactResult& result = call.result;
  CheckNearRelative(result.hit_fraction, (2 - kTolerance) / 3, 1);
  EXPECT_EQ(result.face, SolidFace::kHighZ);
  EXPECT_EQ(result.hit_count, 1);
  CheckBothVelocitiesNear(result, {.x = 0.1, .y = 0, .z = 0}, 1e-15);
  ExpectVec3Near(result.position, {.x = 0.4, .y = 0, .z = 1 + kTolerance}, kPositionTolerance);
}

// What a particle pressed on a spinning top ends with: both velocities and the end position.
struct PressedOnTop {
  Vec3 velocity{};  // m/s, world.
  Vec3 end{};  // m, world.
};

// Runs one scene of a particle pressed on a spinning top and fails the calling test unless it succeeded with one hit,
// both velocities expected.velocity and the end expected.end, within 2 tol.
void CheckPressedOnSpinningTop(std::string_view case_name, PressedOnTop expected) {
  const MoveParticleCall call = ComputeMove(case_name);
  ASSERT_EQ(call.error, ContactError::kNone) << case_name;
  EXPECT_EQ(call.result.hit_count, 1) << case_name;
  CheckBothVelocitiesNear(call.result, expected.velocity, 1e-15);
  ExpectVec3Near(call.result.position, expected.end, kPositionTolerance);
}

TEST(MoveParticle, SpinningCylinderPressedOnCap_MovesOnlyThroughFriction) {
  // Method paper <sec-slide-and-spin>: a spin about the axis is not a turn, so the cap keeps its closed form, and the
  // response feels the surface's {0, 1, 0} m/s through friction alone: none, a slip of friction times the push of 1
  // m/s, a stick.
  const double top = 0.5 + kTolerance;  // m
  CheckPressedOnSpinningTop(
      "SpinningCylinderPressedOnCapNoFriction",
      {.velocity = {}, .end = {.x = 0.5, .y = 0, .z = top}}
  );
  CheckPressedOnSpinningTop(
      "SpinningCylinderPressedOnCapSlipping",
      {.velocity = {.x = 0, .y = 0.25, .z = 0}, .end = {.x = 0.5, .y = 0.025, .z = top}}
  );
  CheckPressedOnSpinningTop(
      "SpinningCylinderPressedOnCapSticking",
      {.velocity = {.x = 0, .y = 1, .z = 0}, .end = {.x = 0.5, .y = 0.1, .z = top}}
  );
}

TEST(MoveParticle, SpinningPlatePressedOn_MovesOnlyThroughFriction) {
  // A plate spinning about its normal does not turn either (method paper <sec-not-turning>): the same answers.
  const double top = 0.5 + kTolerance;  // m
  CheckPressedOnSpinningTop("SpinningPlatePressedOnNoFriction", {.velocity = {}, .end = {.x = 0.5, .y = 0, .z = top}});
  CheckPressedOnSpinningTop(
      "SpinningPlatePressedOnSlipping",
      {.velocity = {.x = 0, .y = 0.25, .z = 0}, .end = {.x = 0.5, .y = 0.025, .z = top}}
  );
  CheckPressedOnSpinningTop(
      "SpinningPlatePressedOnSticking",
      {.velocity = {.x = 0, .y = 1, .z = 0}, .end = {.x = 0.5, .y = 0.1, .z = top}}
  );
}

TEST(MoveParticle, SlowlyTiltingCylinderCapSinkingAway_HitOnce) {
  // The tilt is below the bound of method paper <sec-slide-and-spin>, so the cap keeps its closed form, but its surface
  // sinks at 3e-10 m/s under the particle. The response leaves the particle sinking with it, and the next sweep reads
  // that approach from the same solid velocity as the response, so the cap is not hit again. Measured from the cap's
  // linear velocity alone it saw an approach of 3e-10 m/s, far beyond the response's allowance, and hit the cap at 0 in
  // every sweep until kTooManyHits, at every step length.
  const MoveParticleCall call = ComputeMove("SlowlyTiltingCylinderCapSinkingAway");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kHighZ);
  EXPECT_NEAR(call.result.velocity.for_this_step.z, -3e-10, 1e-15);
}

TEST(MoveParticle, RimOnFloorSweepingParticle_RimPassesOutsideItsTrueDistance) {
  // The rim, sliding over the floor and rising at 1e-3 m/s, passes 1.069 tol from the particle resting on the floor, so
  // its true distance never reaches tol: one hit, the start's on the floor, which takes the fall. A rule that hit the
  // sharp corner of the grown cylinder and moved that hit to the rest gap would lift the particle off the floor: two
  // hits.
  const MoveParticleCase scene = FindMoveParticleCase("RimOnFloorSweepingParticle");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {}, 1e-15);
  EXPECT_NEAR(ComputeEndDistance(scene, call.result.position), kTolerance, kTolerance / 1024);
}

TEST(MoveParticle, RimOnFloorSweepingParticleLower_RimAndFloorRespondTogether) {
  // Rising at 9e-4 m/s the rim comes within tol: hit on kLowRim 1.0002 tol away, with the floor at exactly tol, which
  // the joint response keeps, since the particle is not moved. The particle stays on the floor, carried along by the
  // rim's push: two hits, the start's and the rim's (a hit moved to the rest gap would lift it to 1.106 tol of the
  // floor).
  const MoveParticleCase scene = FindMoveParticleCase("RimOnFloorSweepingParticleLower");
  const MoveBeforeSettle before_settle = ComputeMoveBeforeSettle(scene);
  ASSERT_EQ(before_settle.hit_faces.size(), 2U);
  EXPECT_EQ(before_settle.hit_faces.at(1), SolidFace::kLowRim);
  EXPECT_NEAR(ComputeEndDistance(scene, before_settle.position), kTolerance, kTolerance / 1024);
  const Solid& cylinder = scene.solids.at(1);
  const SolidPose end_pose = ComputeSolidPose(cylinder, scene.inputs.step_duration);
  EXPECT_GE(ComputeDistanceToSolid(cylinder, end_pose, before_settle.position).distance, kTolerance / 2);
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 2);
  ExpectVec3Near(call.result.velocity.for_this_step, {.x = 0.875932, .y = 5.139e-5, .z = 0}, 1e-5);
}

TEST(MoveParticle, SlowlyTiltingCylinderRimCornerStart_RisingRimIsNotHit) {
  // The start is 1.27 tol from the rim, in its rim corner. The rim search reads the cylinder at the pose of the sweep's
  // start, and the rate from linear_velocity, 0: free. The rim rises 0.019 tol toward the particle during the step,
  // within the slack of method paper <sec-slide-and-spin>, and the true distance stays above 1.2 tol (a rule that read
  // the rate of the surface's velocity would hit it at 0 and move the hit to the rest gap).
  const MoveParticleCase scene = FindMoveParticleCase("SlowlyTiltingCylinderRimCornerStart");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  CheckVec3Equal(call.result.velocity.for_this_step, {});
  CheckVec3Equal(call.result.velocity.for_next_step, {});
  EXPECT_GT(ComputeEndDistance(scene, call.result.position), 1.2 * kTolerance);
}

TEST(MoveParticle, SlowlyTiltingCylinderBesideRimLevelWithCap_EndsAboveTheBoundBeforeTheSettle) {
  // The hard case of a resting sweep on a cylinder tilting within the bound of method paper <sec-slide-and-spin>: a
  // wide cylinder, radius
  // 1 m and half height 0.1 m (Length(half_size) = Sqrt(2.01) = 1.418 m), tilting about x at 0.99 of the bound for a
  // step of 1 s; a start beside the rim on the -y side, 0.9 tol beyond the wall and tol / 8 inside the cap plane's rest
  // gap, crossing the cap to y = 0.95 m while sinking toward its plane by 0.99 tol / 64. The sweep rests on the cap's
  // plane, read at the pose of the sweep's start, while the cap rises under the far end by the tilt angle times the
  // arm, 0.99 tol / (8 * 1.418) * 0.95 = 0.0829 tol. Before the settle: 7 / 8 - 0.99 / 64 - 0.0829 = 0.7766 tol,
  // measured 0.77661; above 39 / 64 tol (FindFirstHitOfSweep, step 3) and the floor 47 / 64 tol that the tilt's tol / 8
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
  const MoveParticleCall call = ComputeMove(scene);
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
    const MoveParticleCall call = ComputeMove(case_name);
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
// hit is moved, method paper <sec-move-particle>).
void CheckRimHitOnThePath(const MoveParticleCase& scene, double distance_tolerance) {
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
  CheckVec3Equal(move.result.position, inputs.start_position + velocity * first_hit.elapsed);
  const Solid& cylinder = scene.solids.at(0);
  const SolidPose hit_pose = ComputeSolidPose(cylinder, first_hit.elapsed);
  EXPECT_EQ(ComputeDistanceToSolid(cylinder, hit_pose, move.result.position).distance, first_hit.at_hit.distance);
}

// Fails the calling test unless the scene's rim hit is on the path (CheckRimHitOnThePath) and MoveParticle succeeds
// with one hit and both velocities expected.velocity.
void CheckRimHitAtTrueDistance(std::string_view case_name, RimHitExpectation expected) {
  const MoveParticleCase scene = FindMoveParticleCase(case_name);
  SCOPED_TRACE(case_name);
  CheckRimHitOnThePath(scene, expected.distance_tolerance);
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, expected.velocity, expected.velocity_tolerance);
}

TEST(MoveParticle, CylinderRimNearRestGap_HitAtItsTrueDistance) {
  // The entry is Sqrt(17 / 16) tol = 1.0308 tol from the rim; the rim search stops 3e-5 tol outside tol, where the wall
  // distance is tol / 4 and the normal {1 / 4, 0, Sqrt(15) / 4}: the response leaves {Sqrt(15) / 16, 0, -1 / 16} (as
  // built, moved along the corner's normal, {4 / 17, 0, -1 / 17}).
  CheckRimHitAtTrueDistance(
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
  CheckRimHitAtTrueDistance(
      "RisingCylinderRim",
      {
          .velocity = {.x = std::numbers::sqrt3 / 4, .y = 0, .z = 0.75},
          .velocity_tolerance = 1e-6,
          .distance_tolerance = kTolerance / 1024,
      }
  );
}

TEST(MoveParticle, SlowlyTiltingCylinderRim_HitAtItsTrueDistance) {
  // A tilt within the bound of method paper <sec-slide-and-spin> keeps the closed form and the rim search, which read
  // the cylinder at the pose of the sweep's start; at the pose of the hit the distance differs by the tilt's slack,
  // 0.024 tol here.
  CheckRimHitAtTrueDistance(
      "SlowlyTiltingCylinderRim",
      {
          .velocity = {.x = std::numbers::sqrt3 / 4, .y = 0, .z = -0.25},
          .velocity_tolerance = 1e-5,
          .distance_tolerance = kTolerance / 16,
      }
  );
}

// --- Flush seams and a rim's neighbourhood (method paper <sec-touching-rule>) ---

TEST(MoveParticle, StackedDrumsOfOneRadius_SlideDownTheSeamKeepsItsVelocity) {
  // Down the upper drum's wall at its rest gap and past the seam: the start beside the lower drum's high rim rests on
  // that drum's wall, taken as unbounded along the axis, so the lower drum is not hit, and the velocity stays
  // {0, 0, -1} bit for bit. The same from one rounding unit and from tol / 32 inside the rest gap, which the start
  // rules move out. The sharp grown rim edge alone would be hit 1.41 tol from the rim and leave {0.5, 0, -0.5}.
  const MoveParticleCase scene = FindMoveParticleCase("StackedDrums");
  const Solid& lower_drum = scene.solids.at(1);
  const double rest_x = 0.5 + kTolerance;  // m
  for (const double start_x : {rest_x, rest_x - 0x1p-53, rest_x - kTolerance / 32}) {
    const MoveParticleCall call = ComputeMoveFrom("StackedDrums", {.x = start_x, .y = 0, .z = 1.8});
    ASSERT_EQ(call.error, ContactError::kNone) << start_x;
    EXPECT_EQ(call.result.kind, ContactKind::kNone);
    EXPECT_EQ(call.result.hit_count, 0);
    CheckVec3Equal(call.result.velocity.for_this_step, {.x = 0, .y = 0, .z = -1});
    CheckVec3Equal(call.result.velocity.for_next_step, {.x = 0, .y = 0, .z = -1});
    ExpectVec3Near(call.result.position, {.x = rest_x, .y = 0, .z = 0.8}, 0x1p-50);
    const double end_distance =
        ComputeDistanceToSolid(lower_drum, lower_drum.pose_at_step_start, call.result.position).distance;  // m
    EXPECT_NEAR(end_distance, kTolerance, 0x1p-50);
  }
}

TEST(MoveParticle, PuckFlushInFloor_SlideOntoItAndOffKeepsItsVelocity) {
  // Along the floor at its rest gap onto the puck lying flush in it, and off it: the start beside the puck's rim rests
  // on the plane of its top, so the puck is not hit, and the velocity is kept bit for bit. The sharp grown
  // rim edge alone would stop the particle onto it, which would leave at {-0.5, 0, 0.5}.
  const MoveParticleCase scene = FindMoveParticleCase("PuckFlushInFloor");
  const Solid& floor = scene.solids.at(0);
  const Solid& puck = scene.solids.at(1);
  const MoveParticleCall onto = ComputeMove(scene);
  ASSERT_EQ(onto.error, ContactError::kNone);
  EXPECT_EQ(onto.result.kind, ContactKind::kNone);
  EXPECT_EQ(onto.result.hit_count, 0);
  CheckVec3Equal(onto.result.velocity.for_this_step, -kUnitX);
  CheckVec3Equal(onto.result.velocity.for_next_step, -kUnitX);
  ExpectVec3Near(onto.result.position, {.x = 0.4 + 3 * kTolerance, .y = 0, .z = kTolerance}, kTolerance / 1024);
  const Vec3 onto_end = onto.result.position;
  EXPECT_NEAR(
      ComputeDistanceToSolid(floor, floor.pose_at_step_start, onto_end).distance,
      kTolerance,
      kTolerance / 1024
  );
  EXPECT_NEAR(ComputeDistanceToSolid(puck, puck.pose_at_step_start, onto_end).distance, kTolerance, kTolerance / 1024);
  const MoveParticleCall off = ComputeMoveWith(
      "PuckFlushInFloor",
      {.position = {.x = 0.45, .y = 0, .z = kTolerance}, .velocity = kUnitX, .step_duration = 0.1}
  );
  ASSERT_EQ(off.error, ContactError::kNone);
  EXPECT_EQ(off.result.hit_count, 0);
  CheckVec3Equal(off.result.velocity.for_this_step, kUnitX);
  CheckVec3Equal(off.result.velocity.for_next_step, kUnitX);
  ExpectVec3Near(off.result.position, {.x = 0.55, .y = 0, .z = kTolerance}, kTolerance / 1024);
  const Vec3 off_end = off.result.position;
  EXPECT_NEAR(ComputeDistanceToSolid(floor, floor.pose_at_step_start, off_end).distance, kTolerance, kTolerance / 1024);
}

// Fails the calling test unless a move succeeded with no hit and both velocities velocity, bit for bit.
void CheckFreeWithVelocity(const MoveParticleCall& call, Vec3 velocity) {
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  CheckVec3Equal(call.result.velocity.for_this_step, velocity);
  CheckVec3Equal(call.result.velocity.for_next_step, velocity);
}

// Fails the calling test unless a move succeeded with one hit and both velocities within 1e-15 of velocity.
void CheckOneHitWithVelocity(const MoveParticleCall& call, Vec3 velocity) {
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, velocity, 1e-15);
}

TEST(MoveParticle, StackedDrums_StepStartingJustAboveTheSeamKeepsItsVelocity) {
  // A step that starts 3 / 16 or 5 / 16 tol above the seam, beside the lower drum's rim 1.0174 or 1.0477 tol away, at
  // the level of its wall: the lower drum is no active surface there (method paper <sec-touching-rule>), so nothing
  // answers the rim, and the step is free with {0, 0, -1} bit for bit (with the rim as an active surface, the start's
  // response would turn it to {0.1811321, 0, -0.9660377} and {0.2846975, 0, -0.9110320}). Pushed into the upper wall at
  // 0.16 m/s, the start's response is on that wall alone: {0, 0, -1}, or {0, 0, -0.952} with friction 0.3. From
  // tol / 32 inside the wall's rest gap and tol / 8 above the seam, inside the rim's own touching window, the sweep
  // tries the wall first: free (the touching rule on the rim first would give {0.1230769, 0, -0.9846154}).
  const double rest_x = 0.5 + kTolerance;  // m
  const Vec3 down = -kUnitZ;
  const Vec3 pressed = {.x = -0.16, .y = 0, .z = -1};
  for (const double height_share : {3.0 / 16, 5.0 / 16}) {
    const Vec3 start = {.x = rest_x, .y = 0, .z = 1 + height_share * kTolerance};
    CheckFreeWithVelocity(
        ComputeMoveWith("StackedDrums", {.position = start, .velocity = down, .step_duration = 0.1}),
        down
    );
    const MoveParticleCall pushed =
        ComputeMoveWith("StackedDrums", {.position = start, .velocity = pressed, .step_duration = 0.1});
    CheckOneHitWithVelocity(pushed, down);
    EXPECT_EQ(pushed.result.face, SolidFace::kCurved);
    EXPECT_EQ(pushed.result.solid_index, 0U);
    CheckOneHitWithVelocity(
        ComputeMoveWith(
            "StackedDrums",
            {.position = start, .velocity = pressed, .step_duration = 0.1, .friction = 0.3}
        ),
        {.x = 0, .y = 0, .z = -0.952}
    );
  }
  const Vec3 inside = {.x = rest_x - kTolerance / 32, .y = 0, .z = 1 + kTolerance / 8};
  CheckFreeWithVelocity(
      ComputeMoveWith("StackedDrums", {.position = inside, .velocity = down, .step_duration = 0.1}),
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
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.face, SolidFace::kHighRim);
  EXPECT_EQ(call.result.hit_fraction, 0);
}

TEST(MoveParticle, StackedDrums_HitOnTheUpperWallJustAboveTheSeamLeavesTheLowerOut) {
  // Thrown at the upper drum's wall from 0.01 m out and above, the particle meets it 3 / 16 tol above the seam, level
  // with the lower drum's wall: at that hit the lower drum is no active surface either (method paper
  // <sec-touching-rule>, at a hit as at the start), so the response removes the approach to the upper wall alone: {0,
  // 0, -1}. Listed, the lower rim would turn it to {0.147, 0, -0.785}.
  const MoveParticleCall call = ComputeMoveWith(
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
  CheckBothVelocitiesNear(call.result, -kUnitZ, 1e-15);
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
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 0U);
  CheckBothVelocitiesNear(call.result, {.x = -1, .y = 0, .z = 0.1}, 1e-15);
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

// Fails the calling test unless the rim's slow approach of sink_share tol, from the start built here, gets no hit and
// no reaction, is (1 - sink_share) tol from the rim before the settle and at tol after it.
void CheckRimSlowApproachUnanswered(double sink_share) {
  SCOPED_TRACE(sink_share);
  const MoveParticleCase scene = BuildRimSlowApproachScene({.sink_share = sink_share, .x_units = 0});
  const double before_settle = ComputeEndDistance(scene, ComputePositionBeforeSettle(scene));  // m
  EXPECT_NEAR(before_settle, (1 - sink_share) * kTolerance, kTolerance / 1024);
  EXPECT_GT(before_settle, 0);
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  EXPECT_EQ(call.result.reaction.solid_index, kNoSolid);
  CheckVec3Equal(call.result.velocity.for_this_step, scene.inputs.velocity.for_this_step);
  CheckVec3Equal(call.result.velocity.for_next_step, scene.inputs.velocity.for_next_step);
  EXPECT_NEAR(ComputeEndDistance(scene, call.result.position), kTolerance, kTolerance / 1024);
}

TEST(MoveParticle, RimLevelWithCapSlowApproach_NoResponse_KnownChoice) {
  // A known choice (method paper <sec-touching-rule>): at the rim's rest gap, level with the cap's plane, a particle
  // approaching the rim along its normal by 0.9 tol in the step gets no response. The cylinder is no active surface at
  // the start; the sweep does not rest on the cap's plane (it sinks 0.86 tol toward it), so the plain touching rule on
  // the rim rules, free while the path sinks less than tol. No hit, no reaction, both velocities kept as given (-9
  // tol/s along the rim's normal, or -9.99 tol/s for the sink of 0.999 tol); before the settle the particle is 0.1 tol
  // from the rim, never inside, and the settle lifts it back to tol. From the rest gap a sink below tol keeps d >= 0
  // (the distance is convex along the path): a sink of 0.999 tol ends 0.001 tol away. This side holds because the
  // start, as this test builds it, has a computed distance at or below tol (see the rounding test below).
  CheckRimSlowApproachUnanswered(0.9);
  CheckRimSlowApproachUnanswered(0.999);
}

TEST(MoveParticle, RimLevelWithCapSlowApproach_RoundingOfTheStartDecides_KnownChoice) {
  // Which side of the known choice a start at the rim's rest gap falls on is decided by the rounding of its distance.
  // The start of the test above: computed at or below tol, the plain touching rule frees the path, no hit, both
  // velocities kept (-9 tol/s along the rim's normal). The same start moved by one unit in the last place on x:
  // computed a hair above tol, it is outside, entry 0, and the rim search finds the gap within tol / 1024 and closing,
  // a hit at 0 whose response removes the whole approach along the rim's normal: both velocities {0, 0, 0}. Both keep
  // the path outside the cylinder and end at tol after the settle.
  const MoveParticleCase on_rest_gap = BuildRimSlowApproachScene({.sink_share = 0.9, .x_units = 0});
  const MoveParticleCall free_call = ComputeMove(on_rest_gap);
  ASSERT_EQ(free_call.error, ContactError::kNone);
  EXPECT_EQ(free_call.result.hit_count, 0);
  EXPECT_NEAR(ComputeEndDistance(on_rest_gap, free_call.result.position), kTolerance, kTolerance / 1024);
  const MoveParticleCase a_unit_above = BuildRimSlowApproachScene({.sink_share = 0.9, .x_units = 1});
  const MoveParticleCall hit_call = ComputeMove(a_unit_above);
  ASSERT_EQ(hit_call.error, ContactError::kNone);
  EXPECT_EQ(hit_call.result.hit_count, 1);
  EXPECT_EQ(hit_call.result.face, SolidFace::kHighRim);
  EXPECT_EQ(hit_call.result.hit_fraction, 0);
  CheckBothVelocitiesNear(hit_call.result, {}, 1e-15);
  EXPECT_NEAR(ComputeEndDistance(a_unit_above, hit_call.result.position), kTolerance, kTolerance / 1024);
}

TEST(MoveParticle, PuckFlushInFloor_StepStartingJustBeforeTheSeamKeepsItsVelocity) {
  // The puck's counterpart: a step that starts 3 / 16 or 5 / 16 tol beyond its rim, level with its top's plane, is free
  // with {-1, 0, 0} bit for bit (a response on the rim would give {-0.9660377, 0, 0.1811321} and
  // {-0.9110320, 0, 0.2846975}); pushed into the floor, the start's response is the floor's alone.
  for (const double offset_share : {3.0 / 16, 5.0 / 16}) {
    const Vec3 start = {.x = 0.5 + offset_share * kTolerance, .y = 0, .z = kTolerance};
    CheckFreeWithVelocity(
        ComputeMoveWith("PuckFlushInFloor", {.position = start, .velocity = -kUnitX, .step_duration = 0.1}),
        -kUnitX
    );
    const MoveParticleCall pushed = ComputeMoveWith(
        "PuckFlushInFloor",
        {.position = start, .velocity = {.x = -1, .y = 0, .z = -0.16}, .step_duration = 0.1}
    );
    CheckOneHitWithVelocity(pushed, -kUnitX);
    EXPECT_EQ(pushed.result.face, SolidFace::kHighZ);
    EXPECT_EQ(pushed.result.solid_index, 0U);
  }
}

TEST(MoveParticle, LoneCylinderBesideRimLevelWithCap_FallingIsHitAtTheTrueDistance) {
  // A lone cylinder and a start 5 / 16 tol beyond its wall, level with its cap's plane, 1.048 tol from the rim: the
  // cylinder is no active surface at the start, so falling is answered by the rim search where the true distance is
  // tol, with the normal {0.3125, 0, 0.95}: 0.95 * {0.3125, 0, 0.95} - {0, 0, 1} = {0.2969, 0, -0.0976} (a start's
  // response 1.048 tol away would give {0.2847, 0, -0.0890}). Moving inward along the cap's plane is free.
  MoveParticleCase scene = FindMoveParticleCase("CylinderRim");
  scene.solids.at(0).reports_reaction = false;
  scene.inputs.start_position = {.x = 0.5 + 5.0 / 16 * kTolerance, .y = 0, .z = 1 + kTolerance};
  scene.inputs.step_duration = 0.1;
  scene.inputs.velocity = {.for_next_step = -kUnitZ, .for_this_step = -kUnitZ};
  const MoveParticleCall falling = ComputeMove(scene);
  ASSERT_EQ(falling.error, ContactError::kNone);
  EXPECT_EQ(falling.result.hit_count, 1);
  EXPECT_EQ(falling.result.face, SolidFace::kHighRim);
  EXPECT_GT(falling.result.hit_fraction, 0);
  CheckBothVelocitiesNear(falling.result, Vec3{.x = 0.95 * 0.3125, .y = 0, .z = 0.95 * 0.95 - 1}, 2e-4);
  scene.inputs.velocity = {.for_next_step = -kUnitX, .for_this_step = -kUnitX};
  CheckFreeWithVelocity(ComputeMove(scene), -kUnitX);
}

TEST(MoveParticle, DrumStandingOnFloor_GrooveStopsAndLetsSlideAround) {
  // A guard: the groove between a drum and the floor it stands on is a true corner. Along the floor into it, the wall
  // is hit at 0.5 - tol s; down the wall into it, the floor at 0.7 - tol s; each stops the particle at tol from both.
  // Along the groove, the particle rests on both and is free.
  const MoveParticleCall along_floor = ComputeMove("DrumStandingOnFloor");
  ASSERT_EQ(along_floor.error, ContactError::kNone);
  EXPECT_EQ(along_floor.result.hit_count, 1);
  EXPECT_EQ(along_floor.result.solid_index, 1U);
  EXPECT_EQ(along_floor.result.face, SolidFace::kCurved);
  CheckNearRelative(along_floor.result.hit_fraction, 0.5 - kTolerance, 1);
  CheckBothVelocitiesNear(along_floor.result, {}, 1e-15);
  const Vec3 groove = {.x = 0.5 + kTolerance, .y = 0, .z = kTolerance};
  ExpectVec3Near(along_floor.result.position, groove, kTolerance / 1024);
  const MoveParticleCall down_wall = ComputeMoveWith(
      "DrumStandingOnFloor",
      {.position = {.x = 0.5 + kTolerance, .y = 0, .z = 0.7}, .velocity = -kUnitZ, .step_duration = 1}
  );
  ASSERT_EQ(down_wall.error, ContactError::kNone);
  EXPECT_EQ(down_wall.result.hit_count, 1);
  EXPECT_EQ(down_wall.result.solid_index, 0U);
  EXPECT_EQ(down_wall.result.face, SolidFace::kHighZ);
  CheckNearRelative(down_wall.result.hit_fraction, 0.7 - kTolerance, 1);
  CheckBothVelocitiesNear(down_wall.result, {}, 1e-15);
  ExpectVec3Near(down_wall.result.position, groove, kTolerance / 1024);
  const MoveParticleCall around =
      ComputeMoveWith("DrumStandingOnFloor", {.position = groove, .velocity = kUnitY, .step_duration = 0.25});
  ASSERT_EQ(around.error, ContactError::kNone);
  EXPECT_EQ(around.result.hit_count, 0);
  EXPECT_NEAR(around.result.position.y, 0.25, kTolerance / 1024);
}

TEST(MoveParticle, RimCrestUnderFloor_IsNotHit) {
  // The crest of a slanted cylinder's high rim lies 0.41 tol under the floor's surface; the particle sliding at the
  // floor's rest gap passes it 1.41 tol away, beyond the true distance's reach, though through the sharp grown corner:
  // no hit, and the end at tol from the floor. A rule that hit the corner and moved the hit to the rest gap would pull
  // the particle 0.41 tol toward the floor.
  const MoveParticleCase scene = FindMoveParticleCase("RimCrestUnderFloor");
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  CheckVec3Equal(call.result.velocity.for_this_step, kUnitX);
  CheckVec3Equal(call.result.velocity.for_next_step, kUnitX);
  EXPECT_NEAR(call.result.position.z, kTolerance, kTolerance / 1024);
}

TEST(MoveParticle, SlideOffCapPastRim_AndUpWallPastRim_OneHitThenFree) {
  // A guard. On the cap's rest gap, 8 tol inside the rim, pressed in and sliding out: the start's response keeps
  // {1, 0, 0}, and the slide past the rim is free, ending 8 tol beyond the wall and tol above the cap's plane, Sqrt(65)
  // tol from the rim. The same up the wall past the rim.
  const MoveParticleCase scene = FindMoveParticleCase("CylinderRim");
  const Solid& cylinder = scene.solids.at(0);
  const double slide_duration = 16 * kTolerance;  // s
  const MoveParticleCall off_cap = ComputeMoveWith(
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
  CheckBothVelocitiesNear(off_cap.result, kUnitX, 1e-15);
  const double rim_distance = std::sqrt(65.0) * kTolerance;  // m
  EXPECT_NEAR(
      ComputeDistanceToSolid(cylinder, cylinder.pose_at_step_start, off_cap.result.position).distance,
      rim_distance,
      kTolerance / 1024
  );
  const MoveParticleCall up_wall = ComputeMoveWith(
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
  CheckBothVelocitiesNear(up_wall.result, kUnitZ, 1e-15);
  EXPECT_NEAR(
      ComputeDistanceToSolid(cylinder, cylinder.pose_at_step_start, up_wall.result.position).distance,
      rim_distance,
      kTolerance / 1024
  );
}

TEST(MoveParticle, StackedBoxes_SlideDownTheSeamKeepsItsVelocity) {
  // Down the upper box's side at its rest gap and past the seam: beside the lower box's top edge, its kHighZ gives the
  // distance and the start rests on the plane of its kHighX, so the lower box is not hit and the velocity stays
  // {0, 0, -1} bit for bit (method paper <sec-touching-rule>). Without that rule the path enters through the plane of
  // the lower box's top and stops dead.
  const MoveParticleCall call = ComputeMove("StackedBoxes");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  CheckVec3Equal(call.result.velocity.for_this_step, -kUnitZ);
  CheckVec3Equal(call.result.velocity.for_next_step, -kUnitZ);
  ExpectVec3Near(call.result.position, {.x = 0.5 + kTolerance, .y = 0, .z = 0.8}, kTolerance / 1024);
}

TEST(MoveParticle, BoxesSideBySide_SlideAcrossTheSeamKeepsItsVelocity) {
  // Along two boxes side by side in one floor at the rest gap of their tops: beside the second box its kLowX gives the
  // distance and the start rests on the plane of its top, so it is not hit and the velocity stays {1, 0, 0} bit for
  // bit. Pressed into the boxes at 0.16 m/s, the start's response on the first box is the one hit and leaves {1, 0, 0}.
  // Without that rule the particle would stop on the second box's kLowX at 0.3 - tol s.
  const MoveParticleCall sliding = ComputeMove("BoxesSideBySide");
  ASSERT_EQ(sliding.error, ContactError::kNone);
  EXPECT_EQ(sliding.result.hit_count, 0);
  CheckVec3Equal(sliding.result.velocity.for_this_step, kUnitX);
  CheckVec3Equal(sliding.result.velocity.for_next_step, kUnitX);
  ExpectVec3Near(sliding.result.position, {.x = 0.8, .y = 0, .z = kTolerance}, kTolerance / 1024);
  const MoveParticleCall pressed = ComputeMoveWith(
      "BoxesSideBySide",
      {.position = {.x = 0.2, .y = 0, .z = kTolerance}, .velocity = {.x = 1, .y = 0, .z = -0.16}, .step_duration = 0.6}
  );
  ASSERT_EQ(pressed.error, ContactError::kNone);
  EXPECT_EQ(pressed.result.hit_count, 1);
  EXPECT_EQ(pressed.result.face, SolidFace::kHighZ);
  EXPECT_EQ(pressed.result.solid_index, 0U);
  CheckBothVelocitiesNear(pressed.result, kUnitX, 1e-15);
  ExpectVec3Near(pressed.result.position, {.x = 0.8, .y = 0, .z = kTolerance}, kTolerance / 1024);
}

TEST(MoveParticle, BoxFlushInFloor_SlideOntoItKeepsItsVelocity) {
  // Along the floor at its rest gap onto a box set flush in it: the start rests on the plane of the box's top, so the
  // box is not hit and the velocity is kept bit for bit. Without that rule kHighX would stop the particle.
  const MoveParticleCall call = ComputeMove("BoxFlushInFloor");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  CheckVec3Equal(call.result.velocity.for_this_step, -kUnitX);
  CheckVec3Equal(call.result.velocity.for_next_step, -kUnitX);
}

TEST(MoveParticle, BoxSeams_StepStartingJustBeforeTheSeamKeepsItsVelocity) {
  // A step that starts 33 / 32 tol beyond the seam, level with the next box's top, where its side face's plane,
  // continued above the top, is 1.03 tol away and its edge 1.43 tol: the box is no active surface (method paper
  // <sec-touching-rule>), so the step keeps its velocity bit for bit; pressed into the surface slid on, the start's
  // response is on that surface alone, with friction 0.3 {0.952, 0, 0}. With the box as an active surface every one of
  // these would stop dead.
  const double offset = 33.0 / 32 * kTolerance;  // m
  const Vec3 floor_start = {.x = 0.5 + offset, .y = 0, .z = kTolerance};
  CheckFreeWithVelocity(
      ComputeMoveWith("BoxFlushInFloor", {.position = floor_start, .velocity = -kUnitX, .step_duration = 0.1}),
      -kUnitX
  );
  CheckOneHitWithVelocity(
      ComputeMoveWith(
          "BoxFlushInFloor",
          {.position = floor_start, .velocity = {.x = -1, .y = 0, .z = -0.16}, .step_duration = 0.1}
      ),
      -kUnitX
  );
  // tol / 32 under the floor's rest gap and tol / 64 inside the box's side plane's window: still level, still free.
  const Vec3 under_rest_gap = {.x = 0.5 + kTolerance - kTolerance / 64, .y = 0, .z = kTolerance - kTolerance / 32};
  CheckFreeWithVelocity(
      ComputeMoveWith("BoxFlushInFloor", {.position = under_rest_gap, .velocity = -kUnitX, .step_duration = 0.1}),
      -kUnitX
  );
  const Vec3 side_by_side_start = {.x = 0.5 - offset, .y = 0, .z = kTolerance};
  CheckFreeWithVelocity(
      ComputeMoveWith("BoxesSideBySide", {.position = side_by_side_start, .velocity = kUnitX, .step_duration = 0.1}),
      kUnitX
  );
  const Vec3 pressed = {.x = 1, .y = 0, .z = -0.16};
  const MoveParticleCall pushed =
      ComputeMoveWith("BoxesSideBySide", {.position = side_by_side_start, .velocity = pressed, .step_duration = 0.1});
  CheckOneHitWithVelocity(pushed, kUnitX);
  EXPECT_EQ(pushed.result.face, SolidFace::kHighZ);
  EXPECT_EQ(pushed.result.solid_index, 0U);
  CheckOneHitWithVelocity(
      ComputeMoveWith(
          "BoxesSideBySide",
          {.position = side_by_side_start, .velocity = pressed, .step_duration = 0.1, .friction = 0.3}
      ),
      {.x = 0.952, .y = 0, .z = 0}
  );
  const Vec3 stacked_start = {.x = 0.5 + kTolerance, .y = 0, .z = 1 + offset};
  CheckFreeWithVelocity(
      ComputeMoveWith("StackedBoxes", {.position = stacked_start, .velocity = -kUnitZ, .step_duration = 0.1}),
      -kUnitZ
  );
  CheckOneHitWithVelocity(
      ComputeMoveWith(
          "StackedBoxes",
          {.position = stacked_start, .velocity = {.x = -0.16, .y = 0, .z = -1}, .step_duration = 0.1}
      ),
      -kUnitZ
  );
}

// The scene BoxFlushInFloor without its floor: a lone box of half size 0.5 m, its top at z = 0.
MoveParticleCase BuildLoneBoxScene() {
  MoveParticleCase scene = FindMoveParticleCase("BoxFlushInFloor");
  scene.solids.erase(scene.solids.begin());
  scene.solids.at(0).solid_id = 1;
  return scene;
}

TEST(MoveParticle, LoneBoxBesideItLevelWithItsTop_NoSurfaceThatIsNotThere) {
  // A lone box, its top at z = 0, and a start 33 / 32 tol beyond kHighX at the top's level: the box is no active
  // surface there. Falling, nothing holds the particle up (a bounding surface listed in the box's place would). Moving
  // inward it keeps {-1, 0, 0} (with the box as an active surface, {0, 0, 0}). Moving inward and down, the sharp grown
  // edge is entered after tol / 32 and hit on kHighX (method paper <sec-touching-rule>), which leaves {0, 0, -0.5}.
  MoveParticleCase scene = BuildLoneBoxScene();
  scene.inputs.start_position = {.x = 0.5 + 33.0 / 32 * kTolerance, .y = 0, .z = kTolerance};
  const auto run_with = [&scene](Vec3 velocity) {
    scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    return ComputeMove(scene);
  };
  CheckFreeWithVelocity(run_with(-kUnitZ), -kUnitZ);
  CheckFreeWithVelocity(run_with(-kUnitX), -kUnitX);
  const MoveParticleCall into_edge = run_with({.x = -1, .y = 0, .z = -0.5});
  ASSERT_EQ(into_edge.error, ContactError::kNone);
  EXPECT_EQ(into_edge.result.hit_count, 1);
  EXPECT_EQ(into_edge.result.face, SolidFace::kHighX);
  EXPECT_NEAR(into_edge.result.hit_fraction, (kTolerance / 32) / 0.1, 1e-15);
  CheckBothVelocitiesNear(into_edge.result, {.x = 0, .y = 0, .z = -0.5}, 1e-15);
}

TEST(MoveParticle, MovingBoxBesideFixedBox_NeighbourActsOnlyFromTheSeam) {
  // Two boxes side by side, the second sliding along x at 0.3 m/s, friction 0.5, and a particle pressed into the first
  // at 0.16 m/s, 33 / 32 tol before the seam: the second box is no active surface, so the first's friction alone acts,
  // {1 - 0.5 * 0.16, 0, 0} = {0.92, 0, 0} (with the second box as an active surface, {0.3, 0, 0}: stuck to the moving
  // box's side plane). With the second box rising at 0.1 m/s instead it is a real step: hit on its kLowX, which stops
  // the particle.
  MoveParticleCase conveyor = FindMoveParticleCase("BoxesSideBySide");
  const Box second_box = {
      .center = {.x = 1, .y = 0, .z = -0.5},
      .axes = kIdentity,
      .half_size = {.x = 0.5, .y = 0.5, .z = 0.5},
  };
  conveyor.solids.at(1) =
      ToSolid(second_box, {.linear_velocity = {.x = 0.3, .y = 0, .z = 0}, .angular_velocity = {}}, 2);
  const Vec3 start = {.x = 0.5 - 33.0 / 32 * kTolerance, .y = 0, .z = kTolerance};
  conveyor.inputs.start_position = start;
  conveyor.inputs.velocity = {
      .for_next_step = {.x = 1, .y = 0, .z = -0.16},
      .for_this_step = {.x = 1, .y = 0, .z = -0.16},
  };
  conveyor.inputs.step_duration = 0.1;
  conveyor.inputs.friction = 0.5;
  CheckOneHitWithVelocity(ComputeMove(conveyor), {.x = 0.92, .y = 0, .z = 0});
  MoveParticleCase rising = FindMoveParticleCase("BoxesSideBySide");
  rising.solids.at(1) = ToSolid(second_box, {.linear_velocity = {.x = 0, .y = 0, .z = 0.1}, .angular_velocity = {}}, 2);
  rising.inputs.start_position = start;
  rising.inputs.step_duration = 0.1;
  const MoveParticleCall call = ComputeMove(rising);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kLowX);
  EXPECT_EQ(call.result.solid_index, 1U);
  CheckBothVelocitiesNear(call.result, {}, 1e-15);
}

TEST(MoveParticle, BoxFlushInFloor_LandingBesideTheSeamLeavesTheBoxOut) {
  // Thrown at the floor from 0.01 m above, landing 33 / 32 tol beyond the box set flush in it: at that hit the
  // particle is level with the box's top, so the box is no active surface there either (method paper
  // <sec-touching-rule>, at a hit as at the start), and the floor alone responds: {-1, 0, 0}. Listed, the box's side
  // plane, continued above the top, would stop the particle dead.
  const MoveParticleCall call = ComputeMoveWith(
      "BoxFlushInFloor",
      {
          .position = {.x = 0.5 + 33.0 / 32 * kTolerance + 0.01, .y = 0, .z = kTolerance + 0.01},
          .velocity = {.x = -1, .y = 0, .z = -1},
          .step_duration = 0.1,
      }
  );
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 0U);
  CheckBothVelocitiesNear(call.result, -kUnitX, 1e-15);
}

TEST(MoveParticle, RisingBoxFlushInFloor_LevelReadAtThePoseOfTheHit) {
  // The same landing with the floor and the box rising together at 0.1 m/s: at the hit, read at the pose of the hit's
  // time, the particle is level with the box's top, and keeps {-1, 0, 0.1}. Read at the pose of the step's start, the
  // box would be 1 mm lower, not level, and its side plane would stop the particle.
  MoveParticleCase scene = FindMoveParticleCase("BoxFlushInFloor");
  const SolidMotion rising = {.linear_velocity = {.x = 0, .y = 0, .z = 0.1}, .angular_velocity = {}};
  scene.solids.at(0) = move_particle_cases::BuildFlatSolid({}, kUnitZ, rising, 1);
  const Box box = {
      .center = {.x = 0, .y = 0, .z = -0.5},
      .axes = kIdentity,
      .half_size = {.x = 0.5, .y = 0.5, .z = 0.5},
  };
  scene.solids.at(1) = ToSolid(box, rising, 2);
  scene.inputs.start_position = {.x = 0.5 + 33.0 / 32 * kTolerance + 0.01, .y = 0, .z = kTolerance + 0.011};
  scene.inputs.velocity = {.for_next_step = {.x = -1, .y = 0, .z = -1}, .for_this_step = {.x = -1, .y = 0, .z = -1}};
  scene.inputs.step_duration = 0.1;
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 0U);
  CheckBothVelocitiesNear(call.result, {.x = -1, .y = 0, .z = 0.1}, 1e-15);
}

TEST(MoveParticle, BoxesUpsideDown_LevelWithALowFacePlane) {
  // The low face planes are bounding surfaces too. A ceiling of two boxes side by side, bottoms at z = 0, and a
  // particle 33 / 32 tol before the seam at the ceiling's rest gap, moving along it: free with {1, 0, 0}. A lone box
  // above, a particle level with its bottom 33 / 32 tol beyond its side, moving inward: free with {-1, 0, 0}. Testing
  // only the high planes, kLowX of the second box and kHighX of the lone box would stop the particle.
  const Vec3 half_size = {.x = 0.5, .y = 0.5, .z = 0.5};
  MoveParticleCase ceiling = FindMoveParticleCase("BoxesSideBySide");
  ceiling.solids.at(0) =
      ToSolid(Box{.center = {.x = 0, .y = 0, .z = 0.5}, .axes = kIdentity, .half_size = half_size}, kNoMotion, 1);
  ceiling.solids.at(1) =
      ToSolid(Box{.center = {.x = 1, .y = 0, .z = 0.5}, .axes = kIdentity, .half_size = half_size}, kNoMotion, 2);
  ceiling.inputs.start_position = {.x = 0.5 - 33.0 / 32 * kTolerance, .y = 0, .z = -kTolerance};
  ceiling.inputs.step_duration = 0.1;
  CheckFreeWithVelocity(ComputeMove(ceiling), kUnitX);
  MoveParticleCase lone_above = ceiling;
  lone_above.solids.pop_back();
  lone_above.inputs.start_position = {.x = 0.5 + 33.0 / 32 * kTolerance, .y = 0, .z = -kTolerance};
  lone_above.inputs.velocity = {.for_next_step = -kUnitX, .for_this_step = -kUnitX};
  CheckFreeWithVelocity(ComputeMove(lone_above), -kUnitX);
}

TEST(MoveParticle, LoneBoxLastEighthOfGrownTop_FallsOff_KnownChoice) {
  // A known choice (method paper <sec-touching-rule>): a particle at the top's rest gap, 15 / 16 tol beyond the real
  // edge, on the last tol / 8 of the grown top, is level with the plane of kHighX, so the sweep rests on that plane and
  // the particle falls past the box with both velocities {0, 0, -1} bit for bit, no hit. The top held it before the
  // bounding surfaces came first: a start response on the top left {0, 0, 0}.
  MoveParticleCase scene = BuildLoneBoxScene();
  scene.inputs.start_position = {.x = 0.5 + 15.0 / 16 * kTolerance, .y = 0, .z = kTolerance};
  scene.inputs.velocity = {.for_next_step = -kUnitZ, .for_this_step = -kUnitZ};
  CheckFreeWithVelocity(ComputeMove(scene), -kUnitZ);
}

TEST(MoveParticle, LoneBoxEdgeStripSlowApproach_NoResponse_KnownChoice) {
  // A known choice (method paper <sec-touching-rule>): on the edge strip, 0.95 tol beyond the side at the top's rest
  // gap, level with the plane of kHighX, a particle moving inward and sinking into the top by 0.9 tol in the step gets
  // no response: the box is no active surface at the start, the sweep does not rest on the side plane while the top,
  // which gives the distance, has the plain touching rule, free while the path sinks less than tol. No hit, no
  // reaction, both velocities kept as given, {-1, 0, -9 tol/s}; before the settle the particle is 0.1 tol from the box,
  // never inside, and the settle lifts it back to tol. The start's distance is exactly tol here; one unit in the last
  // place of the box's own coordinate higher it is outside by rounding, and the entry through the top's plane hits it
  // at once instead, which leaves both velocities {-1, 0, 0}, also keeps the path outside and ends at tol.
  MoveParticleCase scene = BuildLoneBoxScene();
  scene.solids.at(0).reports_reaction = true;
  scene.inputs.start_position = {.x = 0.5 + 0.95 * kTolerance, .y = 0, .z = kTolerance};
  const Vec3 velocity = {.x = -1, .y = 0, .z = -0.9 * kTolerance / 0.1};  // m/s
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  const double before_settle = ComputeEndDistance(scene, ComputePositionBeforeSettle(scene));  // m
  EXPECT_NEAR(before_settle, 0.1 * kTolerance, kTolerance / 1024);
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 0);
  EXPECT_EQ(call.result.reaction.solid_index, kNoSolid);
  CheckVec3Equal(call.result.velocity.for_this_step, velocity);
  CheckVec3Equal(call.result.velocity.for_next_step, velocity);
  EXPECT_NEAR(ComputeEndDistance(scene, call.result.position), kTolerance, kTolerance / 1024);
  // One unit in the last place of the box's own coordinate 0.5 + tol, so the box reads the start that much higher.
  scene.inputs.start_position.z = kTolerance + 0x1p-53;
  const MoveParticleCall a_unit_above = ComputeMove(scene);
  ASSERT_EQ(a_unit_above.error, ContactError::kNone);
  EXPECT_EQ(a_unit_above.result.hit_count, 1);
  EXPECT_EQ(a_unit_above.result.face, SolidFace::kHighZ);
  CheckBothVelocitiesNear(a_unit_above.result, -kUnitX, 1e-15);
  EXPECT_NEAR(ComputeEndDistance(scene, a_unit_above.result.position), kTolerance, kTolerance / 1024);
}

// The scene BoxesSideBySide with its second box raised by step_height.
MoveParticleCase BuildRaisedSecondBox(double step_height) {
  MoveParticleCase scene = FindMoveParticleCase("BoxesSideBySide");
  const Box raised = {
      .center = {.x = 1, .y = 0, .z = -0.5 + step_height},
      .axes = kIdentity,
      .half_size = {.x = 0.5, .y = 0.5, .z = 0.5},
  };
  scene.solids.at(1) = ToSolid(raised, kNoMotion, 2);
  return scene;
}

TEST(MoveParticle, BoxesSideBySideStepUp_AThirtySecondIsNotSeenAQuarterStops) {
  // The limit of method paper <sec-touching-rule>: a ledge lower than tol / 8 is not seen. The second box raised by
  // tol / 32: the start rests on the plane of its top, 31 / 32 tol away, crosses onto it without a hit, and the settle
  // lifts it to tol. Raised by tol / 4: the plane of its top is a quarter inside the rest gap, so kLowX stops the
  // particle at 0.3 - tol s.
  const MoveParticleCase low_step = BuildRaisedSecondBox(kTolerance / 32);
  const Solid& raised_box = low_step.solids.at(1);
  const SolidPose low_end_pose = ComputeSolidPose(raised_box, low_step.inputs.step_duration);
  const Vec3 before_settle = ComputePositionBeforeSettle(low_step);
  EXPECT_NEAR(
      ComputeDistanceToSolid(raised_box, low_end_pose, before_settle).distance,
      31.0 / 32 * kTolerance,
      kTolerance / 1024
  );
  const MoveParticleCall over_low_step = ComputeMove(low_step);
  ASSERT_EQ(over_low_step.error, ContactError::kNone);
  EXPECT_EQ(over_low_step.result.hit_count, 0);
  EXPECT_NEAR(
      ComputeDistanceToSolid(raised_box, low_end_pose, over_low_step.result.position).distance,
      kTolerance,
      kTolerance / 1024
  );
  const MoveParticleCall at_quarter_step = ComputeMove(BuildRaisedSecondBox(kTolerance / 4));
  ASSERT_EQ(at_quarter_step.error, ContactError::kNone);
  EXPECT_EQ(at_quarter_step.result.hit_count, 1);
  EXPECT_EQ(at_quarter_step.result.face, SolidFace::kLowX);
  EXPECT_EQ(at_quarter_step.result.solid_index, 1U);
  CheckBothVelocitiesNear(at_quarter_step.result, {}, 1e-15);
  EXPECT_NEAR(at_quarter_step.result.position.x, 0.5 - kTolerance, kTolerance / 1024);
}

TEST(MoveParticle, BoxesSideBySideStepUp_ALedgeDeeperThanAnEighthIsSeen) {
  // The second box raised by 3 / 16 tol, a step: 33 / 32 tol before it, its top's gap is -3 / 16 tol, deeper than the
  // level window's tol / 8, so the box is an active surface by its side face, 1.03 tol away, and the start's response
  // stops the particle at once: hit_fraction 0. A window twice as deep would leave the box out, and the sweep would
  // stop it tol / 32 later instead.
  MoveParticleCase scene = BuildRaisedSecondBox(3.0 / 16 * kTolerance);
  scene.inputs.start_position = {.x = 0.5 - 33.0 / 32 * kTolerance, .y = 0, .z = kTolerance};
  scene.inputs.step_duration = 0.1;
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
  EXPECT_EQ(call.result.face, SolidFace::kLowX);
  EXPECT_EQ(call.result.hit_fraction, 0);
  CheckBothVelocitiesNear(call.result, {}, 1e-15);
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
DistanceToSolid ComputeCarriedAroundRimStep(const Solid& cylinder, double tangent_sign, Vec3* position) {
  const Vec3 normal = ComputeDistanceToSolid(cylinder, cylinder.pose_at_step_start, *position).normal_out_of_solid;
  const Vec3 tangent = tangent_sign * Vec3{.x = normal.z, .y = 0, .z = -normal.x};
  const Vec3 velocity = (kTolerance / 2) * tangent - (kTolerance / 4) * normal;  // m/s
  const MoveParticleCall call =
      ComputeMoveWith("CylinderRim", {.position = *position, .velocity = velocity, .step_duration = 1});
  EXPECT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  *position = call.result.position;
  return ComputeDistanceToSolid(cylinder, cylinder.pose_at_step_start, *position);
}

// Runs 64 steps of a particle carried around the high rim of the fixed cylinder of CylinderRim; each must succeed with
// one hit and end between tol - tol / 1024 and 1.13 tol from the cylinder.
CarriedAroundRim ComputeCarriedAroundRim(Vec3 start, double tangent_sign) {
  const MoveParticleCase scene = FindMoveParticleCase("CylinderRim");
  const Solid& cylinder = scene.solids.at(0);
  CarriedAroundRim carried;
  Vec3 position = start;
  for (int step_index = 0; step_index < 64; ++step_index) {
    const DistanceToSolid at_end = ComputeCarriedAroundRimStep(cylinder, tangent_sign, &position);
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
  const CarriedAroundRim outward = ComputeCarriedAroundRim({.x = 0.5 - 2 * kTolerance, .y = 0, .z = 1 + kTolerance}, 1);
  EXPECT_EQ(outward.face_runs, cap_to_wall);
  EXPECT_NEAR(outward.end.x, 0.5 + kTolerance, kTolerance / 512);
  EXPECT_NEAR(outward.end.z, 1 - 28.39 * kTolerance, 0.01 * kTolerance);
  const std::vector<std::pair<SolidFace, int>> wall_to_cap = {
      {SolidFace::kCurved, 4},
      {SolidFace::kHighRim, 3},
      {SolidFace::kHighZ, 57},
  };
  const CarriedAroundRim inward = ComputeCarriedAroundRim({.x = 0.5 + kTolerance, .y = 0, .z = 1 - 2 * kTolerance}, -1);
  EXPECT_EQ(inward.face_runs, wall_to_cap);
  EXPECT_NEAR(inward.end.x, 0.5 - 28.39 * kTolerance, 0.01 * kTolerance);
  EXPECT_NEAR(inward.end.z, 1 + kTolerance, kTolerance / 512);
}

TEST(MoveParticle, LongCylinderTiltingUnderSlide_EndsWithinAQuarterToleranceBeforeTheSettle) {
  // The touching rule reads the surface's normal speed where the sweep starts; the particle slides 3.9 m along a
  // surface tilting within the bound of method paper <sec-slide-and-spin>, which sinks at one end and rises at the
  // other, so before the settle it ends 0.241 tol closer than the rest gap: within 2 |omega_across| max|arm| T <
  // tol / 4, and above tol - tol / 16 - tol / 4 (FindFirstHitOfSweep, step 3), so the end check's tol / 2 is never at
  // stake. The settle brings it back to tol.
  const MoveParticleCase scene = FindMoveParticleCase("LongCylinderTiltingUnderSlide");
  ASSERT_FALSE(ComputeIsTurningSolid(ToSolidContactInputs(scene), 0));
  const double before_settle = ComputeEndDistance(scene, ComputePositionBeforeSettle(scene));  // m
  EXPECT_NEAR(before_settle, 0.759 * kTolerance, 0.005 * kTolerance);
  EXPECT_GT(before_settle, kTolerance - kTolerance / 16 - kTolerance / 4);
  const MoveParticleCall call = ComputeMove(scene);
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
void CheckSettleDiffersInLastPlace(SettleScenes scenes) {
  const MoveParticleCall call = ComputeMove(scenes.first);
  const MoveParticleCall other = ComputeMove(scenes.second);
  ASSERT_EQ(call.error, ContactError::kNone);
  ASSERT_EQ(other.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, other.result.hit_count);
  CheckVec3Equal(other.result.velocity.for_this_step, call.result.velocity.for_this_step);
  CheckVec3Equal(other.result.velocity.for_next_step, call.result.velocity.for_next_step);
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
  CheckSettleDiffersInLastPlace(
      {.first = "SettleNearSpinningCylinder", .second = "SettleNearSpinningCylinderWithCopy"}
  );
}

TEST(MoveParticle, SettleInReversedOrder_LastPlaceKnownLimit) {
  // Reversed, the near surfaces reach the solver in another order, and the end differs in the last place.
  CheckSettleDiffersInLastPlace(
      {.first = "SettleBesideMovingCylinder", .second = "SettleBesideMovingCylinderReversed"}
  );
}

TEST(ApplyHit, TurningCylinderRimHit_StaysWhereTheSearchStopped) {
  // The stepping search stops near the rim, 1.11 tol from it, in the rim region (method paper <sec-turning-solid>): its
  // hit is used where the search stopped, as every hit is (method paper <sec-move-particle>). The hit applied alone;
  // MoveParticle's own check is TurningCylinderRimHit_UsedWhereTheSearchStopped.
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

TEST(MoveParticle, TurningCylinderRimHit_UsedWhereTheSearchStopped) {
  // The whole move: no error, one hit on the rim, recorded where the stepping search stopped, at the distance it
  // reported (about 1.11 tol), then the carry in the cylinder's axes.
  const MoveParticleCase scene = FindMoveParticleCase("TurningCylinderRimHit");
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  const Vec3 velocity = inputs.velocity.for_this_step;
  FirstHit first_hit;
  ASSERT_EQ(
      FindFirstHit(inputs, inputs.start_position, velocity, 0, inputs.step_duration, &first_hit),
      ContactError::kNone
  );
  ASSERT_EQ(first_hit.solid_index, 0U);
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.face, SolidFace::kHighRim);
  const Vec3 hit_position = call.result.reaction.hit_position;
  ExpectVec3Near(hit_position, inputs.start_position + velocity * first_hit.elapsed, kTolerance / 1024);
  const Solid& cylinder = scene.solids.at(0);
  const double hit_distance =
      ComputeDistanceToSolid(cylinder, ComputeSolidPose(cylinder, first_hit.elapsed), hit_position).distance;  // m
  EXPECT_NEAR(hit_distance, first_hit.at_hit.distance, kTolerance / 1024);
  EXPECT_NEAR(first_hit.at_hit.distance, 1.11 * kTolerance, 0.01 * kTolerance);
  EXPECT_GE(ComputeDistanceAtEndPose(scene, call.result.position), kTolerance / 2);
}

TEST(FindActiveSurfaces, TurningBoxLevelWithOtherFacePlane_IsActive) {
  // Beside the face kHighX, 33 / 32 tol away and level with the plane of kHighZ: a box that does not turn is passed
  // at a bounding surface's level and gives no active surface (method paper <sec-touching-rule>); a turning box has no
  // bounding surface, so it is active.
  const Vec3 position = {.x = 0.5 + 33.0 / 32 * kTolerance, .y = 0, .z = 0.5 + kTolerance};
  for (const bool is_turning : {false, true}) {
    const SolidMotion motion = is_turning ? SolidMotion{.linear_velocity = {}, .angular_velocity = kUnitZ} : kNoMotion;
    const std::array solids = {
        ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, motion, 1),
    };
    SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({.step_duration = 0.1});
    inputs.solids = solids.data();
    inputs.solid_count = 1;
    ASSERT_EQ(ComputeIsTurningSolid(inputs, 0), is_turning);
    ParticleMove move;
    move.result.position = position;
    EXPECT_EQ(FindActiveSurfaces(inputs, move, nullptr).count, is_turning ? 1 : 0) << "turning " << is_turning;
  }
}

// --- Closing solids: carried out along the apex, or squeezed ---

// Runs the closing wedge scene of that name and fails the calling test unless it succeeded with one hit, the joint
// response over both faces, and the expected velocities within relative_tolerance of their largest axis.
SolidContactResult CheckCarriedOut(std::string_view case_name, Vec3 expected_velocity, double relative_tolerance) {
  const MoveParticleCall call = ComputeMove(case_name);
  EXPECT_EQ(call.error, ContactError::kNone) << case_name;
  EXPECT_EQ(call.result.hit_count, 1) << case_name;
  const double scale =
      std::max({std::abs(expected_velocity.x), std::abs(expected_velocity.y), std::abs(expected_velocity.z)});
  CheckBothVelocitiesNear(call.result, expected_velocity, relative_tolerance * scale);
  return call.result;
}

TEST(MoveParticle, FloorRisingUnderSixtyDegreeWall_CarriesParticleAlongApex) {
  // The velocity that approaches neither: {V / tan 60, 0, V}, not a squeeze. After 0.1 s the floor is 0.1 m higher and
  // the particle is at the rest gap of both at the end pose.
  const MoveParticleCase scene = FindMoveParticleCase("FloorRisingUnderSixtyDegreeWall");
  const SolidContactResult result =
      CheckCarriedOut("FloorRisingUnderSixtyDegreeWall", {.x = 1 / std::numbers::sqrt3, .y = 0, .z = 1}, 1e-15);
  const Vec3 expected_end = {
      .x = std::numbers::sqrt3 * kTolerance + 0.1 / std::numbers::sqrt3,
      .y = 0,
      .z = 0.1 + kTolerance,
  };
  ExpectVec3Near(result.position, expected_end, kTolerance / 1024);
  CheckAtRestGapOfEverySolid(ToSolidContactInputs(scene), result.position);
}

TEST(MoveParticle, BothFacesClosingAtSixtyDegrees_CarriesParticleOut) {
  CheckCarriedOut("BothFacesClosingAtSixtyDegrees", {.x = std::numbers::sqrt3 / 2, .y = 0, .z = 0.5}, 1e-15);
}

TEST(MoveParticle, FloorRisingFastUnderSixtyDegreeWall_EscapeTooFastForStepIsCuredByShorterStep) {
  // The escape {500 / tan 60, 0, 500} m/s is within the speed cap, but its allowance, 2^-46 times the 1290 m/s of
  // speeds it involves, times the 1 s left is above tol / 64: kSqueezedBetweenSolids naming the rising floor, the
  // solid the particle approaches fastest. In a step of 0.25 s it is carried out, one hit; the solver's rounding at
  // 577 m/s leaves the velocity within 1e-15 of that speed of the closed form.
  CheckCloseCall(
      ComputeMove("FloorRisingFastUnderSixtyDegreeWall"),
      ContactCloseCall::kSqueezedBetweenSolids,
      1,
      ShorterStep::kCures
  );
  MoveParticleCase shorter = FindMoveParticleCase("FloorRisingFastUnderSixtyDegreeWall");
  shorter.inputs.step_duration = 0.25;
  const MoveParticleCall call = ComputeMove(shorter);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  const double speed = 1000 / std::numbers::sqrt3;  // m/s
  CheckBothVelocitiesNear(call.result, {.x = 500 / std::numbers::sqrt3, .y = 0, .z = 500}, 1e-15 * speed);
}

TEST(MoveParticle, FloorRisingThroughFixedFloor_PushChargedToTheFloorThatPushes) {
  // The start moves to the apex rest point; one joint response over the fixed floor, the rising floor and the side
  // carries the particle along the apex at {0.25 / tan 60, 0, 0.25}, one hit. The push along {0, 0, 1} is the rising
  // floor's, whose target is the larger: charged to it, it counts in no reaction. Charged to the first floor of the
  // direction, the fixed reacting one, it would fill a reaction although that floor never pushed the particle.
  const MoveParticleCall call = ComputeMove("FloorRisingThroughFixedFloorUnderSixtyDegreeWall");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {.x = 0.25 / std::numbers::sqrt3, .y = 0, .z = 0.25}, 1e-15);
  CheckNoReaction(call.result.reaction);
}

TEST(MoveParticle, OneFaceClosingAtOneDegree_CarriesParticleOut) {
  // {1 / tan(1 degree), 0, 1} = {57.28996, 0, 1}: a speed of 1 / sin(1 degree) = 57.2987 m/s, below 64 times the
  // largest speed involved, the floor's 1 m/s.
  const double angle = std::numbers::pi / 180;  // rad
  CheckCarriedOut("OneFaceClosingAtOneDegree", {.x = 1 / std::tan(angle), .y = 0, .z = 1}, 1e-9);
}

TEST(MoveParticle, OneFaceClosingAtPointEightDegrees_IsSqueezedBetweenSolids) {
  // The change would give 1 / sin(0.8 degrees) = 71.622 m/s, above 64 m/s. It names the rising floor, the solid the
  // particle approaches fastest. A step ten times shorter does not cure it. The limit is the closed form
  // asin(1 / 64) = 0.8953 degrees: 0.89 is squeezed (64.38 m/s), 0.90 carried out (63.6646 m/s), since the cap reads
  // the speeds before the response.
  CheckCloseCall(
      ComputeMove("OneFaceClosingAtPointEightDegrees"),
      ContactCloseCall::kSqueezedBetweenSolids,
      1,
      ShorterStep::kDoesNotCure
  );
  MoveParticleCase shorter = FindMoveParticleCase("OneFaceClosingAtPointEightDegrees");
  shorter.inputs.step_duration = 0.001;
  CheckCloseCall(ComputeMove(shorter), ContactCloseCall::kSqueezedBetweenSolids, 1, ShorterStep::kDoesNotCure);
  const double degree = std::numbers::pi / 180;  // rad
  const MoveParticleCase below_limit =
      move_particle_cases::BuildClosingWedgeCase("OneFaceClosingAtPointEightNineDegrees", 0.89 * degree, 1, 0);
  CheckCloseCall(ComputeMove(below_limit), ContactCloseCall::kSqueezedBetweenSolids, 1, ShorterStep::kDoesNotCure);
  const MoveParticleCase above_limit =
      move_particle_cases::BuildClosingWedgeCase("OneFaceClosingAtPointNineDegrees", 0.9 * degree, 1, 0);
  const MoveParticleCall call = ComputeMove(above_limit);
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_NEAR(math::Length(call.result.velocity.for_this_step), 1 / std::sin(0.9 * degree), 1e-9);
}

TEST(MoveParticle, TwoFacesClosingSymmetrically_CapAtOnePointSevenNineDegrees) {
  // Each face at 1 m/s along its own normal: the escape is 1 / sin(a / 2), within the cap down to the closed form
  // 2 asin(1 / 64) = 1.7906 degrees, since the cap reads the speeds before the response. At 2 degrees, 57.2987 m/s, one
  // hit; at 1.6 degrees it would be 71.62 m/s, squeezed, naming the floor; at 1.79 degrees 64.02 m/s, squeezed; at
  // 1.80, 63.6646 m/s, carried out.
  const MoveParticleCall call = ComputeMove("TwoFacesClosingAtTwoDegrees");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_NEAR(math::Length(call.result.velocity.for_this_step), 1 / std::sin(std::numbers::pi / 180), 1e-9);
  CheckCloseCall(
      ComputeMove("TwoFacesClosingAtOnePointSixDegrees"),
      ContactCloseCall::kSqueezedBetweenSolids,
      1,
      ShorterStep::kDoesNotCure
  );
  const double degree = std::numbers::pi / 180;  // rad
  const MoveParticleCase below_limit =
      move_particle_cases::BuildClosingWedgeCase("TwoFacesClosingAtOnePointSevenNineDegrees", 1.79 * degree, 1, 1);
  CheckCloseCall(ComputeMove(below_limit), ContactCloseCall::kSqueezedBetweenSolids, 1, ShorterStep::kDoesNotCure);
  const MoveParticleCase above_limit =
      move_particle_cases::BuildClosingWedgeCase("TwoFacesClosingAtOnePointEightDegrees", 1.8 * degree, 1, 1);
  const MoveParticleCall above_call = ComputeMove(above_limit);
  ASSERT_EQ(above_call.error, ContactError::kNone);
  EXPECT_NEAR(math::Length(above_call.result.velocity.for_this_step), 1 / std::sin(0.9 * degree), 1e-9);
}

TEST(MoveParticle, GrazingWedgeClosingFast_IsSqueezedAtStepsDownToOneMillisecond) {
  // The escape would be 1 / tan(1e-5) = 1e5 m/s: before the cap the model launched the particle 1 km in a step of
  // 0.01 s. Squeezed at 0.1, 0.01 and 0.001 s, naming the side, the solid the particle approaches fastest. The squeeze
  // comes at a hit after the start of the step, so a step that ends before it cures it: 2^-26 of 0.1 s does
  // (EveryCatalogueFailure_ShorterStepCuresExactlyWhenFlagSays).
  for (const double step_duration : {0.1, 0.01, 0.001}) {
    MoveParticleCase scene = FindMoveParticleCase("GrazingWedgeClosingFast");
    scene.inputs.step_duration = step_duration;
    CheckCloseCall(ComputeMove(scene), ContactCloseCall::kSqueezedBetweenSolids, 2, ShorterStep::kCures);
  }
}

TEST(MoveParticle, SpinningFloorMovedSilently_RoundingIsNotAHit) {
  // The spinning floor is moved silently at the start: the particle's approach to it, 7.7e-15 m/s, is within the
  // rounding of the spin's omega x r, 2^-46 * |omega x r|_1. The upright responds, and the sweep that follows measures
  // its rate from the floor's full surface velocity, as the start rules and the joint response do. One hit, the
  // upright's. With the floor's approach compared with 0 at the start and the sweep's rate measured from
  // linear_velocity alone, the floor would be hit at 0 by rounding, two hits; either change alone leaves two.
  const MoveParticleCall call = ComputeMove("SpinningFloorMovedSilently");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
  EXPECT_EQ(call.result.solid_index, 1U);
}

TEST(MoveParticle, ClosingCeiling_IsSqueezedBetweenSolids) {
  // The ceiling reaches the particle at 0.05 - tol s and carries it down onto the floor, hit at 0.1 - 2 tol s, the
  // record of that hit; there the two close at 1 m/s and no velocity approaches neither. The particle sticks to the
  // ceiling, the solid it approaches fastest, which still approaches the floor: squeezed, naming the ceiling, id 2.
  CheckCloseCall(ComputeMove("ClosingCeiling"), ContactCloseCall::kSqueezedBetweenSolids, 2, ShorterStep::kCures);
}

// --- The slide: scenes of a first outside check ---

// Runs MoveParticle twice from the inputs: the step, then the next step from its end with both velocities its
// for_next_step. Fails the calling test unless both succeed and end at the rest gap of every solid.
void CheckTwoStepsAtRestGap(SolidContactInputs inputs) {
  SolidContactResult result;
  ContactFailure failure;
  for (int step = 1; step <= 2; ++step) {
    const ContactError error = MoveParticle(inputs, &result, &failure);
    ASSERT_EQ(error, ContactError::kNone)
        << "step " << step << ", T " << inputs.step_duration << ", close call " << static_cast<int>(failure.close_call);
    CheckAtRestGapOfEverySolid(inputs, result.position);
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
    CheckTwoStepsAtRestGap(inputs);
  }
}

// Runs the edge scene of that name with a friction and a velocity, and returns its result; fails the calling test
// unless it succeeded.
SolidContactResult ComputeEdgeScene(std::string_view case_name, double friction, Vec3 velocity) {
  MoveParticleCase scene = FindMoveParticleCase(case_name);
  scene.inputs.friction = friction;
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  const MoveParticleCall call = ComputeMove(scene);
  EXPECT_EQ(call.error, ContactError::kNone) << case_name;
  return call.result;
}

TEST(MoveParticle, DomainBoxEdgeWithFriction_SameAsFlatSolids) {
  // The particle meets the box's x and z sides at the same time: one joint response over both, as over the two flat
  // solids in the same place, one hit. Both fixed, so the friction limits add: with friction 0.5 and {-1, 0.5, -1},
  // pushes 1 and 1, a limit of 1 >= 0.5, {0, 0, 0}; friction 2 and {-1, 2, -1}, a stick too; friction 0.5 and
  // {-1, 2, -1}, 2 - 0.5 (1 + 1) = 1, {0, 1, 0} (the responses one side after the other gave 1.1645898). Before the
  // first outside check, the box's second side was removed without friction: v_y 0.276 and 0.211.
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
    const SolidContactResult box = ComputeEdgeScene("DomainBoxEdgeWithFriction", draw.friction, draw.velocity);
    const SolidContactResult flats = ComputeEdgeScene("FlatSolidsEdgeWithFriction", draw.friction, draw.velocity);
    EXPECT_EQ(box.hit_count, 1);
    EXPECT_EQ(flats.hit_count, 1);
    ExpectVec3Near(box.velocity.for_next_step, flats.velocity.for_next_step, 1e-15);
    ExpectVec3Near(box.position, flats.position, kTolerance / 1024);
  }
  ExpectVec3Near(
      ComputeEdgeScene("DomainBoxEdgeWithFriction", 0.5, {.x = -1, .y = 0.5, .z = -1}).velocity.for_next_step,
      {},
      1e-15
  );
  ExpectVec3Near(
      ComputeEdgeScene("DomainBoxEdgeWithFriction", 2, {.x = -1, .y = 2, .z = -1}).velocity.for_next_step,
      {},
      1e-15
  );
  ExpectVec3Near(
      ComputeEdgeScene("DomainBoxEdgeWithFriction", 0.5, {.x = -1, .y = 2, .z = -1}).velocity.for_next_step,
      {.x = 0, .y = 1, .z = 0},
      1e-15
  );
}

// Runs the chamfered domain corner at one step length and velocity, the start 0.2 T {1, 1, 1}, and fails the calling
// test unless it succeeds at the rest gap of all five surfaces or farther. Returns the result.
SolidContactResult CheckChamferedCornerSuccess(double step_duration, Vec3 velocity) {
  MoveParticleCase scene = FindMoveParticleCase("DomainCornerWithTwoChamfers");
  scene.inputs.step_duration = step_duration;
  scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  scene.inputs.start_position = 0.2 * step_duration * Vec3{.x = 1, .y = 1, .z = 1};
  const MoveParticleCall call = ComputeMove(scene);
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
  // meet the smallest change as equalities). Before, more than four was kTooManyHits at every step length. The particle
  // stops in the corner at the rest gap of all five surfaces, at steps of 1, 0.01 and 1e-4 s (the start 0.2 T out), for
  // the outside check's three velocities. The chamfer through the corner line is inside the corner's own cone: the
  // responses one surface after the other gave the particle speed along y from it ({0, 0.5, 0} or {0, 0.15, 0} after
  // four hits), the joint response does not.
  for (const double step_duration : {1.0, 0.01, 1e-4}) {
    for (const Vec3 velocity :
         {Vec3{.x = -1, .y = -1, .z = -1}, Vec3{.x = -1, .y = -2, .z = -0.5}, Vec3{.x = -0.3, .y = -1, .z = -0.2}}) {
      const SolidContactResult result = CheckChamferedCornerSuccess(step_duration, velocity);
      CheckBothVelocitiesNear(result, {}, 1e-15);
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
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone) << static_cast<int>(call.failure.close_call);
  EXPECT_EQ(call.result.hit_count, 3);
  CheckBothVelocitiesNear(call.result, {}, 1e-15);
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
  const MoveParticleCall call = ComputeMove(scene);
  ASSERT_EQ(call.error, ContactError::kNone) << static_cast<int>(call.failure.close_call);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {.x = 1, .y = 0, .z = 0}, 1e-15);
}

TEST(MoveParticle, MovingCornerWithCoincidentFaceFastEscape_AllowanceRaisedForThisStep) {
  // Draw 3053 of a generated search: the escape is much faster than the velocity the response received, so the rounding
  // it may leave is larger than that velocity's own allowance; the response's allowance counts the change too, which
  // keeps the sweeps from hitting a face again by rounding. Three hits, and the velocity within 8.6e-14 of the speed of
  // a sequence of one response per surface (the joint response changes it by rounding only here).
  const MoveParticleCall call = ComputeMove("MovingCornerWithCoincidentFaceFastEscape");
  ASSERT_EQ(call.error, ContactError::kNone) << static_cast<int>(call.failure.close_call);
  EXPECT_EQ(call.result.hit_count, 3);
}

TEST(MoveParticle, MovingCornerWithCoincidentFaceNextStepEscape_Succeeds) {
  // Draw 115326 of the same search, which separated for_next_step's allowance from for_this_step's: the coincident
  // fourth face took the rounding a change left in for_next_step for an approach. The start's faces are active at once:
  // one joint response, one hit.
  const MoveParticleCall call = ComputeMove("MovingCornerWithCoincidentFaceNextStepEscape");
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.hit_count, 1);
}

TEST(MoveParticle, ObtuseWedgeStartInBothOrders_Same) {
  // In a wedge of 120 degrees, friction 0, {-1, 1, -1}: the responses one solid after the other gave {-0.25, 1, 0.433}
  // with the floor first and {0.183, 1, 0} with the side first. One joint response gives the smallest frictionless
  // change over both, {0, 1, 0}, one hit, the same bits in both orders.
  const MoveParticleCall floor_first = ComputeMove("ObtuseWedgeFloorFirst");
  const MoveParticleCall side_first = ComputeMove("ObtuseWedgeSideFirst");
  ASSERT_EQ(floor_first.error, ContactError::kNone);
  ASSERT_EQ(side_first.error, ContactError::kNone);
  EXPECT_EQ(floor_first.result.hit_count, 1);
  EXPECT_EQ(side_first.result.hit_count, 1);
  CheckBothVelocitiesNear(floor_first.result, {.x = 0, .y = 1, .z = 0}, 1e-15);
  CheckVec3Equal(side_first.result.velocity.for_next_step, floor_first.result.velocity.for_next_step);
  CheckVec3Equal(side_first.result.velocity.for_this_step, floor_first.result.velocity.for_this_step);
}

TEST(MoveParticle, CoincidentReactingFloorsNormalsApartByRounding_NoSecondReactingSolid) {
  // Two reacting floors whose normals are 2e-13 apart share one direction, so one response pushes once: no false
  // kSecondReactingSolid at any step length. The push is charged to the direction's pushing surface: equal friction,
  // both reacting, so the first in the sorted order, the floor whose normal's x is the smaller, solid_index 1.
  for (const double step_duration : {0.2, 0.001}) {
    MoveParticleCase scene = FindMoveParticleCase("CoincidentReactingFloorsNormalsApartByRounding");
    scene.inputs.step_duration = step_duration;
    const MoveParticleCall call = ComputeMove(scene);
    ASSERT_EQ(call.error, ContactError::kNone) << static_cast<int>(call.failure.close_call);
    EXPECT_EQ(call.result.hit_count, 1);
    CheckBothVelocitiesNear(call.result, {.x = 1, .y = 0, .z = 0}, 1e-15);
    EXPECT_EQ(call.result.reaction.solid_index, 1U);
    ExpectVec3Near(call.result.reaction.velocity_for_next_step_change, {.x = 0, .y = 0, .z = 1}, 1e-15);
  }
}

// Runs the rest-point wedge of that angle name listed floor first and side first, and fails the calling test unless
// both succeed with one hit and the same velocities, bit for bit. Returns the floor-first result.
SolidContactResult CheckSameInBothOrders(std::string_view angle_name) {
  const std::string floor_first = std::string(angle_name) + "DegreeRestPointFloorFirst";
  const std::string side_first = std::string(angle_name) + "DegreeRestPointSideFirst";
  const MoveParticleCall floor_call = ComputeMove(floor_first);
  const MoveParticleCall side_call = ComputeMove(side_first);
  EXPECT_EQ(floor_call.error, ContactError::kNone) << floor_first;
  EXPECT_EQ(side_call.error, ContactError::kNone) << side_first;
  EXPECT_EQ(floor_call.result.hit_count, 1) << floor_first;
  EXPECT_EQ(side_call.result.hit_count, 1) << side_first;
  CheckVec3Equal(side_call.result.velocity.for_next_step, floor_call.result.velocity.for_next_step);
  CheckVec3Equal(side_call.result.velocity.for_this_step, floor_call.result.velocity.for_this_step);
  return floor_call.result;
}

TEST(MoveParticle, RestPointWedgesWithFriction_SameInBothOrders) {
  // At the rest point both faces are active: one joint response, friction 0.5. Both surfaces are fixed, so their
  // friction limits add (method paper <sec-move-particle>): the velocity along the apex is 1 - 0.5 (push_floor +
  // push_side), or 0, a stick. 30 degrees, {-1, 1, -1}: pushes 2.732 and 2, a limit of 2.366 >= 1, {0, 0, 0} exactly.
  // 45, 60 and
  // 75 degrees, {-0.2, 1, -1}: y 0.258578643763, 0.326794919243 and 0.369677462716. The same bits in both orders.
  const SolidContactResult thirty = CheckSameInBothOrders("Thirty");
  CheckBothVelocitiesNear(thirty, {}, 0);
  const std::array<std::pair<std::string_view, double>, 3> slips = {
      std::pair<std::string_view, double>{"FortyFive", 0.258578643763},
      std::pair<std::string_view, double>{"Sixty", 0.326794919243},
      std::pair<std::string_view, double>{"SeventyFive", 0.369677462716},
  };
  for (const auto& [angle_name, along_apex] : slips) {
    const SolidContactResult result = CheckSameInBothOrders(angle_name);
    EXPECT_NEAR(result.velocity.for_next_step.y, along_apex, 1e-12) << angle_name;
    EXPECT_NEAR(result.velocity.for_next_step.x, 0, 1e-15) << angle_name;
    EXPECT_NEAR(result.velocity.for_next_step.z, 0, 1e-15) << angle_name;
  }
}

TEST(MoveParticle, MicroradianWedgeOnReactingFloor_FloorApproachedFastestIsCharged) {
  // A wedge of 1e-6 rad, a sliver (1 + c = 5e-13 < 2^-40), at its rest point: no change is found, and the particle
  // sticks to the surface it approaches fastest. Moving at {-2, 1, 0} that is the side: {0, 0, 0}, no reaction. Moving
  // at {-2, 1, -0.001} it is the reacting floor, charged the whole stop {2, -1, 0.001}.
  const MoveParticleCall into_side = ComputeMove("MicroradianWedgeOnReactingFloor");
  ASSERT_EQ(into_side.error, ContactError::kNone);
  EXPECT_EQ(into_side.result.hit_count, 1);
  CheckBothVelocitiesNear(into_side.result, {}, 0);
  CheckNoReaction(into_side.result.reaction);
  MoveParticleCase scene = FindMoveParticleCase("MicroradianWedgeOnReactingFloor");
  const Vec3 into_floor_velocity = {.x = -2, .y = 1, .z = -0.001};
  scene.inputs.velocity = {.for_next_step = into_floor_velocity, .for_this_step = into_floor_velocity};
  const MoveParticleCall into_floor = ComputeMove(scene);
  ASSERT_EQ(into_floor.error, ContactError::kNone);
  EXPECT_EQ(into_floor.result.hit_count, 1);
  CheckBothVelocitiesNear(into_floor.result, {}, 0);
  EXPECT_EQ(into_floor.result.reaction.solid_index, 0U);
  CheckVec3Equal(into_floor.result.reaction.velocity_for_next_step_change, {.x = 2, .y = -1, .z = 0.001});
}

TEST(MoveParticle, CopyFuzzScenes_ExactCopyChangesNothing) {
  // Two draws of the copy fuzz where an exact copy of a face comes after it in the array. The copy is dropped from the
  // active surfaces (ComputeIsExactCopy), so each scene with its copy gives the hit count and the velocities of the
  // scene without it, bit for bit. Three faces (draw 131): one hit, {0, 0, 0}, a stick in the corner. Two faces (draw
  // 51): as before.
  for (const std::string_view case_name : {"CopyFuzzThreeFaces", "CopyFuzzTwoFaces"}) {
    const MoveParticleCall without_copy = ComputeMove(case_name);
    const MoveParticleCall with_copy = ComputeMove(std::string(case_name) + "WithCopy");
    ASSERT_EQ(without_copy.error, ContactError::kNone) << case_name;
    ASSERT_EQ(with_copy.error, ContactError::kNone) << case_name;
    EXPECT_EQ(with_copy.result.hit_count, without_copy.result.hit_count) << case_name;
    CheckVec3Equal(with_copy.result.velocity.for_next_step, without_copy.result.velocity.for_next_step);
    CheckVec3Equal(with_copy.result.velocity.for_this_step, without_copy.result.velocity.for_this_step);
  }
  const MoveParticleCall three_faces = ComputeMove("CopyFuzzThreeFaces");
  EXPECT_EQ(three_faces.result.hit_count, 1);
  CheckBothVelocitiesNear(three_faces.result, {}, 0);
}

TEST(MoveParticle, DomainBoxAndSideWithGround_SameAsWithoutGround) {
  // A ground lying on the domain box's bottom, fixed, with the bottom's normal bit for bit: an exact copy when its
  // friction is the same, dropped; with twice the friction it is the bottom's direction's pushing surface, and its
  // friction acts. At the rest point of the box's bottom and a side at 60 degrees, {-1, 1, -1}, friction 0.5: the
  // pushes give a friction limit above the speed along the apex, so all three scenes stick, {0, 0, 0}, one hit. The
  // result reads the active surfaces, not which face rounds below tol, so the libm of std::tan no longer changes it.
  for (const std::string_view case_name :
       {"DomainBoxAndSide", "DomainBoxAndSideWithGround", "DomainBoxAndSideWithRoughGround"}) {
    const MoveParticleCall call = ComputeMove(case_name);
    ASSERT_EQ(call.error, ContactError::kNone) << case_name;
    EXPECT_EQ(call.result.hit_count, 1) << case_name;
    CheckBothVelocitiesNear(call.result, {}, 0);
  }
}

// Fails the calling test unless both calls have the same error, kind, hit count and both velocities, bit for bit.
void CheckSameBits(const MoveParticleCall& actual, const MoveParticleCall& expected, std::string_view label) {
  EXPECT_EQ(actual.error, expected.error) << label;
  EXPECT_EQ(actual.result.kind, expected.result.kind) << label;
  EXPECT_EQ(actual.result.hit_count, expected.result.hit_count) << label;
  CheckVec3Equal(actual.result.velocity.for_next_step, expected.result.velocity.for_next_step);
  CheckVec3Equal(actual.result.velocity.for_this_step, expected.result.velocity.for_this_step);
}

// Runs the solids in every order of the array and fails the calling test unless every order gives the bits of the
// first. Returns the call of the order given.
MoveParticleCall CheckSameBitsInEveryOrder(const std::vector<Solid>& solids, const SolidContactInputs& inputs) {
  const MoveParticleCall first = ComputeMove(BuildOrderedScene(solids, inputs));
  std::vector<std::size_t> order(solids.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  while (std::ranges::next_permutation(order).found) {
    CheckSameBits(ComputeMove(BuildOrderedScene(ToPermutedSolids(solids, order), inputs)), first, "permuted");
  }
  return first;
}

// Coincident floors and an upright: floor_count coincident fixed floors and an upright at x = 0.5, the particle falling
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
  // Eight coincident floors, then an upright. The floors are exact copies, one active surface; nothing is listed per
  // solid, so nothing overflows. The fall is one hit at 0.05 - tol / 0.2 of the step, the upright a second: {0, 0, 0}
  // at {0.5 - tol, 0, tol}. The same with 17 and 40 floors (built once under AddressSanitizer).
  for (const int floor_count : {8, 17, 40}) {
    const MoveParticleCall call = ComputeMove(BuildFloorsAndWallScene(floor_count));
    ASSERT_EQ(call.error, ContactError::kNone) << floor_count;
    EXPECT_EQ(call.result.hit_count, 2) << floor_count;
    CheckBothVelocitiesNear(call.result, {}, 0);
    ExpectVec3Near(call.result.position, {.x = 0.5 - kTolerance, .y = 0, .z = kTolerance}, kAtRestGapTolerance);
    EXPECT_NEAR(call.result.hit_fraction, 0.05 - kTolerance / 0.2, 1e-15) << floor_count;
  }
}

// A coincident floor one rounding unit faster: a floor moving at {0, 0.3, 0} and a fixed side at that wedge angle, and
// a third solid when given.
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
  // Beside a floor moving at {0, 0.3, 0}, a coincident floor at {0, 0.1 + 0.2, 0}, one rounding unit faster, pushes as
  // much; equal friction, so the first in the sorted order acts, the slower floor: the same bits as without it, and as
  // with an exact copy. Friction 0.5: y 0.15 at 30 and 60 degrees. Friction 0.1: y 1 - 0.1 (2.7320508 + 2) =
  // 0.526794919243 at 30 degrees and 0.726794919243 at 60.
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
    const MoveParticleCall alone =
        ComputeMove(BuildOrderedScene(BuildMovingFloorWedge(row.wedge_angle, nullptr), inputs));
    ASSERT_EQ(alone.error, ContactError::kNone);
    EXPECT_EQ(alone.result.hit_count, 1);
    EXPECT_NEAR(alone.result.velocity.for_next_step.y, row.along_apex, 1e-12);
    for (const Solid* third : {&faster_floor, &copy_floor}) {
      CheckSameBits(
          ComputeMove(BuildOrderedScene(BuildMovingFloorWedge(row.wedge_angle, third), inputs)),
          alone,
          "third"
      );
    }
  }
}

// Runs a floor moving at {0, 0.3, 0} and a 60 degree side moving at {0, side_speed, 0} from their rest point at
// {-1, 1, -1} with that friction, and fails the calling test unless one hit leaves {0, along_apex, 0} within
// tolerance on y and 1e-15 on x and z.
//
// clang-tidy: the friction, the speed and the expected speed are doubles; the swappable-parameters check flags them.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void CheckTwoMovingFacesAlongApex(double friction, double side_speed, double along_apex, double tolerance) {
  const double wedge_angle = std::numbers::pi / 3;  // rad
  const SolidContactInputs inputs = BuildRestPointInputs(wedge_angle, {.x = -1, .y = 1, .z = -1}, friction);
  const std::vector<Solid> solids = {
      BuildMovingFlatSolid(kUnitZ, {.x = 0, .y = 0.3, .z = 0}, 1, 1),
      BuildMovingFlatSolid(ToWedgeSideNormal(wedge_angle), {.x = 0, .y = side_speed, .z = 0}, 1, 2),
  };
  const MoveParticleCall call = ComputeMove(BuildOrderedScene(solids, inputs));
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
    CheckTwoMovingFacesAlongApex(0.5, side_speed, 0.3, 1e-15);
    CheckTwoMovingFacesAlongApex(0.1, side_speed, 0.726794919243, 1e-12);
  }
}

// Runs the floors at z = 0 and z = tol / 2 from z = tol / 4 at that velocity, the lower one reacting or not and listed
// first or second, and fails the calling test unless the start inside the upper one is one hit, kind
// kStartedInsideSolid, both velocities {0, 0, 0}, and no reaction.
void CheckUpperFloorResponds(Vec3 velocity, bool is_lower_reacting, bool is_lower_first) {
  Solid lower_floor = move_particle_cases::BuildFloor(1);
  lower_floor.reports_reaction = is_lower_reacting;
  const Solid upper_floor =
      move_particle_cases::BuildFlatSolid({.x = 0, .y = 0, .z = kTolerance / 2}, kUnitZ, kNoMotion, 1);
  const std::vector<Solid> solids =
      is_lower_first ? std::vector<Solid>{lower_floor, upper_floor} : std::vector<Solid>{upper_floor, lower_floor};
  const SolidContactInputs inputs =
      move_particle_cases::BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = kTolerance / 4}, velocity, 0.01);
  const MoveParticleCall call = ComputeMove(BuildOrderedScene(solids, inputs));
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(call.result.hit_count, 1);
  CheckBothVelocitiesNear(call.result, {}, 0);
  CheckNoReaction(call.result.reaction);
}

TEST(MoveParticle, ParallelFloorsStartInsideUpper_SameInBothOrders) {
  // Floors at z = 0 and z = tol / 2, the start at tol / 4, inside the upper one. The move takes it to 1.5 tol, where
  // the lower floor is not active: the upper one responds, in either order, one hit, kind kStartedInsideSolid. With the
  // lower floor reacting, listed first or second, no reaction: it never pushed.
  for (const bool is_lower_reacting : {false, true}) {
    for (const bool is_lower_first : {true, false}) {
      CheckUpperFloorResponds({.x = 0, .y = 0, .z = -1}, is_lower_reacting, is_lower_first);
      CheckUpperFloorResponds({}, is_lower_reacting, is_lower_first);
    }
  }
}

TEST(MoveParticle, MovingFourFaceCorner_SucceedsAtEveryStepLength) {
  // Draw 102164 of the generated fixed corners: four flat solids through the origin, two of them moving, friction 2.92.
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
    const MoveParticleCall call = ComputeMove(BuildOrderedScene(solids, inputs));
    ASSERT_EQ(call.error, ContactError::kNone) << step_duration;
    EXPECT_EQ(call.result.hit_count, 2) << step_duration;
    CheckBothVelocitiesNear(call.result, {.x = -0.152699427, .y = 0.200466685, .z = 0.144745987}, 1e-8);
  }
}

// Runs the scene with a copy of solids[copied_index] inserted at every place of the array, and fails the calling test
// unless each gives the hit count and the velocities of the scene without it, bit for bit.
void CheckCopyInEveryPlaceChangesNothing(const MoveParticleCase& scene, std::size_t copied_index) {
  const MoveParticleCall without_copy = ComputeMove(BuildOrderedScene(scene.solids, scene.inputs));
  ASSERT_EQ(without_copy.error, ContactError::kNone);
  for (std::size_t place = 0; place <= scene.solids.size(); ++place) {
    std::vector<Solid> solids = scene.solids;
    solids.insert(solids.begin() + static_cast<std::ptrdiff_t>(place), scene.solids[copied_index]);
    CheckSameBits(ComputeMove(BuildOrderedScene(solids, scene.inputs)), without_copy, "copy");
  }
}

TEST(MoveParticle, CopiesInEveryPlaceOfTheArray_SameBits) {
  // An exact copy listed before the surface it copies would take a response of its own. Now a copy is dropped wherever
  // it stands: the two copy fuzz scenes, the moving floor of CoincidentFloorOneRoundingUnitFaster and draw 15798 of a
  // generated copy search, the copy at every index, give the bits of the scene without it.
  CheckCopyInEveryPlaceChangesNothing(FindMoveParticleCase("CopyFuzzThreeFaces"), 0);
  CheckCopyInEveryPlaceChangesNothing(FindMoveParticleCase("CopyFuzzTwoFaces"), 1);
  const double wedge_angle = std::numbers::pi / 3;  // rad
  const MoveParticleCase moving_floor = BuildOrderedScene(
      BuildMovingFloorWedge(wedge_angle, nullptr),
      BuildRestPointInputs(wedge_angle, {.x = -1, .y = 1, .z = -1}, 0.5)
  );
  CheckCopyInEveryPlaceChangesNothing(moving_floor, 0);
  // Draw 15798 of that search: three fixed faces at their rest point, friction 0; with a copy of the third listed
  // first, the responses one surface after the other gave a velocity 0.45 of the speed away from the scene without it.
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
  CheckCopyInEveryPlaceChangesNothing(three_faces, 2);
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
  const MoveParticleCall plain = ComputeMove(BuildOrderedScene({floor, side}, inputs));
  ASSERT_EQ(plain.error, ContactError::kNone);
  EXPECT_NEAR(plain.result.velocity.for_next_step.y, 0.730717967697, 1e-12);
  const MoveParticleCall rough = ComputeMove(BuildOrderedScene({rough_floor, side}, inputs));
  ASSERT_EQ(rough.error, ContactError::kNone);
  EXPECT_NEAR(rough.result.velocity.for_next_step.y, 0.507623956930, 1e-12);
  const MoveParticleCall all_three = CheckSameBitsInEveryOrder({floor, side, rough_floor}, inputs);
  CheckSameBits(all_three, rough, "three");
  CheckSameBits(ComputeMove(BuildOrderedScene({floor, side, rough_floor, floor, floor}, inputs)), rough, "five");
  Solid reacting_floor = floor;
  reacting_floor.reports_reaction = true;
  const MoveParticleCall reacting = ComputeMove(BuildOrderedScene({reacting_floor, side}, inputs));
  ASSERT_EQ(reacting.error, ContactError::kNone);
  EXPECT_EQ(reacting.result.reaction.solid_index, 0U);
  ExpectVec3Near(
      reacting.result.reaction.velocity_for_next_step_change,
      {.x = 0, .y = -0.223094010768, .z = 1.115470053838},
      1e-12
  );
}

TEST(MoveParticle, ConveyorOnFloor_FirstInSortedOrderActs) {
  // A pinned choice (method paper <sec-move-particle>): a floor and a coincident conveyor moving at {1, 0, 0}, the
  // particle falling at 1 m/s onto both, friction 0.5. Both push as much; with equal friction the first in the sorted
  // order acts, the fixed floor, whose x velocity is the smaller: {0, 0, 0} in both orders. The conveyor with friction
  // scale 2 acts: {1, 0, 0}; the fixed floor with scale 2: {0, 0, 0}.
  SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(
      {.x = 0, .y = 0, .z = kTolerance},
      {.x = 0, .y = 0, .z = -1},
      0.01
  );
  inputs.friction = 0.5;
  const Solid floor = move_particle_cases::BuildFloor(1);
  const Solid conveyor = BuildMovingFlatSolid(kUnitZ, kUnitX, 1, 2);
  CheckBothVelocitiesNear(CheckSameBitsInEveryOrder({floor, conveyor}, inputs).result, {}, 0);
  const Solid rough_conveyor = BuildMovingFlatSolid(kUnitZ, kUnitX, 2, 2);
  CheckBothVelocitiesNear(CheckSameBitsInEveryOrder({floor, rough_conveyor}, inputs).result, kUnitX, 0);
  Solid rough_floor = floor;
  rough_floor.friction_scale = 2;
  CheckBothVelocitiesNear(CheckSameBitsInEveryOrder({rough_floor, conveyor}, inputs).result, {}, 0);
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
    const MoveParticleCall call = CheckSameBitsInEveryOrder({conveyor, side}, inputs);
    ASSERT_EQ(call.error, ContactError::kNone);
    EXPECT_EQ(call.result.hit_count, 1);
    CheckBothVelocitiesNear(call.result, {.x = 0, .y = along_y, .z = 0}, 1e-15);
  }
}

TEST(MoveParticle, RisingWallBesideFloor_FrictionLiftsOnlyAboveThePush) {
  // A floor and an upright with normal {-1, 0, 0} rising at 2 m/s, the particle at their corner moving at {1, 0, -1}:
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
    const MoveParticleCall call = ComputeMove(BuildOrderedScene({floor, rising_wall}, inputs));
    ASSERT_EQ(call.error, ContactError::kNone) << friction;
    CheckBothVelocitiesNear(call.result, {.x = 0, .y = 0, .z = along_z}, 1e-15);
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
    const MoveParticleCall call = CheckSameBitsInEveryOrder({floor, smooth_face, rough_face}, inputs);
    ASSERT_EQ(call.error, ContactError::kNone);
    EXPECT_EQ(call.result.hit_count, 1);
    EXPECT_NEAR(call.result.velocity.for_next_step.y, 0.469669914110, 1e-12) << sideways;
  }
}

TEST(MoveParticle, TwoReactingSolidsInCrease_IsSecondReactingSolid) {
  // Method paper <sec-move-particle>: at the rest point of a 60 degree wedge, both faces reacting, a velocity {-1, 0,
  // -2} pushes into both. One response charges both: kSecondReactingSolid naming the higher index, solid_id 2, in both
  // orders. With only the floor reacting: one hit, its push {0, 0, 2.5773502691896262}.
  const double wedge_angle = std::numbers::pi / 3;  // rad
  const SolidContactInputs inputs = BuildRestPointInputs(wedge_angle, {.x = -1, .y = 0, .z = -2}, 0);
  Solid floor = move_particle_cases::BuildFloor(1);
  Solid side = move_particle_cases::BuildWedgeSide(std::numbers::pi - wedge_angle, 2);
  floor.reports_reaction = true;
  side.reports_reaction = true;
  CheckCloseCall(
      ComputeMove(BuildOrderedScene({floor, side}, inputs)),
      ContactCloseCall::kSecondReactingSolid,
      2,
      ShorterStep::kDoesNotCure
  );
  CheckCloseCall(
      ComputeMove(BuildOrderedScene({side, floor}, inputs)),
      ContactCloseCall::kSecondReactingSolid,
      2,
      ShorterStep::kDoesNotCure
  );
  side.reports_reaction = false;
  const MoveParticleCall call = ComputeMove(BuildOrderedScene({floor, side}, inputs));
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
  const MoveParticleCall inside = ComputeMove(BuildOrderedScene(
      solids,
      move_particle_cases::BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = -kTolerance / 2}, velocity, 0.01)
  ));
  ASSERT_EQ(inside.error, ContactError::kNone);
  EXPECT_EQ(inside.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(inside.result.hit_count, 1);
  CheckBothVelocitiesNear(inside.result, velocity, 0);
  const MoveParticleCall outside = ComputeMove(BuildOrderedScene(
      solids,
      move_particle_cases::BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = kTolerance / 2}, velocity, 0.01)
  ));
  ASSERT_EQ(outside.error, ContactError::kNone);
  EXPECT_EQ(outside.result.kind, ContactKind::kNone);
  EXPECT_EQ(outside.result.hit_count, 0);
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
  // A known limit of the mean over orders (method paper <sec-move-particle>), pinned so that an exact rule for solids
  // that move differently shows up as a change here. A scene of an outside check: a 26.73 degree wedge whose floor
  // moves at {-0.1206, -0.0055, 0} and whose side moves along itself, friction 0.716, at the rest point. Each order's
  // floor step stops at sticking to the floor, but the side's step then moves the velocity on: in the mean, the floor's
  // own friction share {-0.0086, -0.2203, 0} points along the particle's final slip relative to the floor (-0.444 m/s
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
  // {-2, 1, -0.001} into the floor). Without that key the array order would decide, and the reaction would come and go.
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
    const MoveParticleCall call = ComputeMove(BuildOrderedScene(orders.at(order_index), inputs));
    ASSERT_EQ(call.error, ContactError::kNone) << order_index;
    CheckBothVelocitiesNear(call.result, {}, 0);
    EXPECT_EQ(call.result.reaction.solid_index, reacting_indexes.at(order_index)) << order_index;
    CheckVec3Equal(call.result.reaction.velocity_for_next_step_change, {.x = 2, .y = -1, .z = 0.001});
  }
}

TEST(MoveParticle, ReactingFloorTiltedByRounding_PushesForItsDirection) {
  // At the rest point of a 60 degree wedge, a plain floor and a reacting floor whose normal is tilted by 2^-45, which
  // sorts it after the plain one: one direction, both push, equal friction, so the one that reports a reaction pushes
  // (method paper <sec-move-particle>) and is charged: in both orders the reaction is the reacting floor's.
  const double wedge_angle = std::numbers::pi / 3;  // rad
  Solid reacting_floor = move_particle_cases::BuildFlatSolid({}, {.x = 0x1p-45, .y = 0, .z = 1}, kNoMotion, 1);
  reacting_floor.reports_reaction = true;
  const Solid floor = move_particle_cases::BuildFloor(1);
  const Solid side = move_particle_cases::BuildWedgeSide(std::numbers::pi - wedge_angle, 1);
  const SolidContactInputs inputs = BuildRestPointInputs(wedge_angle, {.x = -0.2, .y = 1, .z = -1}, 0);
  const MoveParticleCall plain_first = ComputeMove(BuildOrderedScene({floor, reacting_floor, side}, inputs));
  ASSERT_EQ(plain_first.error, ContactError::kNone);
  EXPECT_EQ(plain_first.result.reaction.solid_index, 1U);
  const MoveParticleCall reacting_first = ComputeMove(BuildOrderedScene({reacting_floor, floor, side}, inputs));
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
  const MoveParticleCall call = ComputeMove(BuildOrderedScene(
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
  // Method paper <sec-move-particle>, for the second solve: "no change" reads the allowance passed, not the solve's own
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
  CheckVec3Equal(change.velocity, {.x = 0, .y = 0, .z = 0.5});
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
  CheckVec3Equal(
      move.moved_point,
      MoveToRestGap(inputs, 0, ComputeSolidPose(inputs.solids[0], 0), inputs.start_position)
  );
}

// Fails the calling test unless the solver found a change within 1e-12 of the expected one on each axis, through
// used_count surfaces.
void CheckCorrection(const SmallestCorrection& correction, Vec3 expected_change, int used_count) {
  ASSERT_TRUE(correction.is_found);
  ExpectVec3Near(correction.change, expected_change, 1e-12);
  EXPECT_EQ(correction.used_count, used_count);
}

// Fails the calling test unless the two multipliers of a pair are the expected ones within 1e-12.
void CheckPairMultipliers(const SmallestCorrection& correction, double first_multiplier, double second_multiplier) {
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

// Three equalities whose normals {1, 0, 0}, {0, 1, 0} and Normalized({1, 0, 2^-560}) give a triple product of 2^-560
// (about 2.6e-169), not 0, while the first and third dot rounds to 1, so the determinant of the dots is exactly 0:
// normals in one plane up to a rounding unit. Right sides 1.
SubsetEquations BuildDotsDeterminantZeroEquations() {
  SubsetEquations equations;
  equations.count = 3;
  equations.normals[0] = kUnitX;
  equations.normals[1] = kUnitY;
  equations.normals[2] = math::Normalized({.x = 1, .y = 0, .z = 0x1p-560});
  for (double& right_side : equations.right_sides) {
    right_side = 1;
  }
  return equations;
}

TEST(SolveSubsetMultipliers, DotsDeterminantZeroWithTripleProductNot_IsNotSolved) {
  // No division by a determinant that rounds to 0: not solved, as for three normals in one plane.
  const SubsetEquations equations = BuildDotsDeterminantZeroEquations();
  ASSERT_NE(math::Dot(equations.normals[0], math::Cross(equations.normals[1], equations.normals[2])), 0.0);
  EXPECT_FALSE(SolveSubsetMultipliers(equations, CorrectedQuantity::kPosition).is_solved);
}

TEST(SolveCorrectionOfSubset, DotsDeterminantZero_HasNoCandidate) {
  // The same outcome as before the guard, when the quotients were NaN and rejected as not finite.
  const SubsetEquations equations = BuildDotsDeterminantZeroEquations();
  CorrectionTargets targets;
  targets.count = 3;
  for (int surface_index = 0; surface_index < 3; ++surface_index) {
    targets.normals[surface_index] = equations.normals[surface_index];
    targets.targets[surface_index] = 1;
  }
  const SmallestCorrection candidate =
      SolveCorrectionOfSubset(targets, {.count = 3, .surfaces = {0, 1, 2}}, 0, CorrectedQuantity::kPosition);
  EXPECT_FALSE(candidate.is_found);
}

TEST(SolveSmallestCorrection, OneTwoAndThreeSurfaces_MatchClosedForms) {
  const double root_three = std::numbers::sqrt3;
  const Vec3 floor_normal = {.x = 0, .y = 0, .z = 1};
  const Vec3 sixty_degree_normal = {.x = root_three / 2, .y = 0, .z = -0.5};
  // The 60 degree wedge, both targets 1: the pair, multipliers {2, 2}.
  const SmallestCorrection on_both =
      SolveTestCorrection(BuildCorrectionTargets({floor_normal, sixty_degree_normal}, {1, 1}));
  CheckCorrection(on_both, {.x = root_three, .y = 0, .z = 1}, 2);
  CheckPairMultipliers(on_both, 2, 2);
  // The same wedge, targets {1, 0}: still the pair, since the floor alone leaves the other face short.
  const SmallestCorrection floor_short =
      SolveTestCorrection(BuildCorrectionTargets({floor_normal, sixty_degree_normal}, {1, 0}));
  CheckCorrection(floor_short, {.x = 1 / root_three, .y = 0, .z = 1}, 2);
  CheckPairMultipliers(floor_short, 4.0 / 3, 2.0 / 3);
  // A right angle with targets {1, -1}: the floor alone meets both.
  CheckCorrection(
      SolveTestCorrection(BuildCorrectionTargets({floor_normal, {.x = 1, .y = 0, .z = 0}}, {1, -1})),
      floor_normal,
      1
  );
  // The three axes: the triple.
  CheckCorrection(
      SolveTestCorrection(
          BuildCorrectionTargets({{.x = 1, .y = 0, .z = 0}, {.x = 0, .y = 1, .z = 0}, floor_normal}, {1, 1, 1})
      ),
      {.x = 1, .y = 1, .z = 1},
      3
  );
  // An obtuse wedge: the pair, multipliers {2/3, 2/3}.
  const SmallestCorrection obtuse =
      SolveTestCorrection(BuildCorrectionTargets({floor_normal, {.x = root_three / 2, .y = 0, .z = 0.5}}, {1, 1}));
  CheckCorrection(obtuse, {.x = 1 / root_three, .y = 0, .z = 1}, 2);
  CheckPairMultipliers(obtuse, 2.0 / 3, 2.0 / 3);
  // A shallow pit: straight up, 1 / cos(3 degrees) = 1.0013723.
  CheckCorrection(
      SolveTestCorrection(BuildShallowPitTargets()),
      {.x = 0, .y = 0, .z = 1 / std::cos(3 * std::numbers::pi / 180)},
      3
  );
}

TEST(SolveSmallestCorrection, VelocityWithinSlack_IsNoChange) {
  // A velocity already within its slack of every target needs no change: used_count 0 and a zero change, so rounding
  // never changes it. A position has no empty subset: the same target moves it along the normal.
  const CorrectionTargets far_target = BuildCorrectionTargets({{.x = 0, .y = 0, .z = 1}}, {1});
  CheckCorrection(
      SolveSmallestCorrection(far_target, 0x1p-46, CorrectedQuantity::kVelocity),
      {.x = 0, .y = 0, .z = 1},
      1
  );
  const CorrectionTargets rounding_target = BuildCorrectionTargets({{.x = 0, .y = 0, .z = 1}}, {0x1p-50});
  const SmallestCorrection velocity_change =
      SolveSmallestCorrection(rounding_target, 0x1p-46, CorrectedQuantity::kVelocity);
  ASSERT_TRUE(velocity_change.is_found);
  EXPECT_EQ(velocity_change.used_count, 0);
  CheckVec3Equal(velocity_change.change, {});
  const SmallestCorrection position_change =
      SolveSmallestCorrection(rounding_target, 0x1p-46, CorrectedQuantity::kPosition);
  CheckCorrection(position_change, {.x = 0, .y = 0, .z = 0x1p-50}, 1);
}

TEST(SolveSmallestCorrection, VelocityNoChangeReadsTheSlackPassed) {
  // Method paper <sec-move-particle>: no change only when every target is within the slack passed. A target of 2^-50
  // m/s with speeds of 1 m/s before the change is within the candidate's own allowance 2^-46 (speeds), but above a
  // slack of 0: a real approach beside a fast solid, which the change removes.
  CorrectionTargets slow_target = BuildCorrectionTargets({{.x = 0, .y = 0, .z = 1}}, {0x1p-50});
  slow_target.speeds_before_change = 1;
  CheckCorrection(
      SolveSmallestCorrection(slow_target, 0, CorrectedQuantity::kVelocity),
      {.x = 0, .y = 0, .z = 0x1p-50},
      1
  );
  EXPECT_EQ(SolveSmallestCorrection(slow_target, 0x1p-46, CorrectedQuantity::kVelocity).used_count, 0);
}

TEST(SolveSmallestCorrection, VelocityTie_KeepsLargestFrictionSum) {
  // Method paper <sec-move-particle>: a floor and two faces at 45 degrees whose normals share the y axis, targets 1,
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
  CheckCorrection(frictionless, {.x = 0, .y = 0, .z = 1}, 1);
  shared_line.frictions[0] = 0.3;
  shared_line.frictions[1] = 0.15;
  shared_line.frictions[2] = 0.6;
  const SmallestCorrection rough = SolveSmallestCorrection(shared_line, 0, CorrectedQuantity::kVelocity);
  CheckCorrection(rough, {.x = 0, .y = 0, .z = 1}, 2);
  EXPECT_EQ(rough.used_surfaces[0], 1);
  EXPECT_EQ(rough.used_surfaces[1], 2);
  CheckPairMultipliers(rough, half_root_two, half_root_two);
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
  CheckCorrection(change, math::Normalized({.x = 0, .y = 1, .z = 1}), 1);
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

// --- The touched rule in the sweep ---

// The sweep that follows the first hit of the scene ObliqueFlatSolidRoundingAtHit: the hit position and the response's
// velocity, as MoveParticle computes them, with the allowance of that response and the hit's normal.
struct SweepAfterObliqueHit {
  MoveParticleCase scene;
  Sweep sweep;
  TouchedRule touched;
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
  const TouchedRule not_touched;
  const ContactError error =
      FindFirstHitOfSweep(ToSolidContactInputs(after_hit.scene), after_hit.sweep, not_touched, &first_hit, nullptr);
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_EQ(first_hit.solid_index, 0U);
}

TEST(FindFirstHitOfSweep, TouchedRuleAfterRoundedHit_IsFree) {
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

TEST(FindFirstHitOfSweep, TouchedRuleReallyApproached_IsHitAtStart) {
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

TEST(FindFirstHitOfSweep, TouchedRuleFartherThanSixteenthOfTolerance_TakesClosedForm) {
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
  CheckVec3Equal(move.result.position, {.x = 0.5, .y = 0, .z = kTolerance});
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
  CheckVec3Equal(move.result.position, {.x = 0.5, .y = 0, .z = 2 * kTolerance});
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
  CheckAtRestGapOfEverySolid(inputs, move.result.position);
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
  CheckAtRestGapOfEverySolid(inputs, move.result.position);
}

// A particle above the fixed floor z <= 0.5 at a height of `height` m, moving at {0.2, 0, -1} m/s with friction 0.4
// for 1 ms, tol 2^-30 m: the boundary between a start clear of the floor and one on it.
MoveParticleCase BuildStartNearActiveWindow(double height) {
  return {
      .name = "StartNearActiveWindow",
      .solids = {move_particle_cases::BuildFlatSolid({.x = 0, .y = 0, .z = 0.5}, kUnitZ, kNoMotion, 1)},
      .inputs = move_particle_cases::BuildCaseInputs({
          .start_position = {.x = 0x1.3333333333333p-2, .y = 0x1.6666666666666p-1, .z = height},
          .velocity = {.x = 0.2, .y = 0, .z = -1},
          .step_duration = 1e-3,
          .friction = 0.4,
      }),
  };
}

// Checks that a start at `height` responds at once and stays there: kind kHit at hit fraction 0, the position unmoved.
void CheckRespondsAtStart(double height) {
  const MoveParticleCall call = ComputeMove(BuildStartNearActiveWindow(height));
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_EQ(call.result.hit_fraction, 0.0);
  EXPECT_EQ(call.result.position.x, 0x1.3333333333333p-2);
  EXPECT_EQ(call.result.position.z, height);
}

TEST(MoveParticle, StartExactlyAtActiveWindow_RespondsAtStart) {
  // 0x1.000000088p-1 is exactly 0.5 + 17/16 tol: the floor is within the active window tol + tol / 16 (the window
  // includes its edge), so the start is not clear of it. The start's response sticks the particle (friction 0.4 takes
  // the 0.2 m/s sideways speed of a 1 m/s approach) and it stays at the start.
  CheckRespondsAtStart(0x1.000000088p-1);
}

TEST(MoveParticle, StartOneUnitBelowActiveWindow_RespondsAtStart) {
  CheckRespondsAtStart(std::nextafter(0x1.000000088p-1, 0.0));
}

TEST(MoveParticle, StartOneUnitAboveActiveWindow_IsClearAndHitLater) {
  // One unit above the window the start is clear of the floor: no response at the start; the sweep meets the floor
  // after about tol / 16 s and the particle slides on, so it ends past the start in x, at the rest gap.
  const double height = std::nextafter(0x1.000000088p-1, 1.0);
  const MoveParticleCall call = ComputeMove(BuildStartNearActiveWindow(height));
  ASSERT_EQ(call.error, ContactError::kNone);
  EXPECT_EQ(call.result.kind, ContactKind::kHit);
  EXPECT_GT(call.result.hit_fraction, 0.0);
  EXPECT_GT(call.result.position.x, 0x1.3333333333333p-2);
}

}  // namespace
}  // namespace fluidity::contact
