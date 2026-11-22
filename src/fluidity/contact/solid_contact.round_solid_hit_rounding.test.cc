#include "fluidity/contact/solid_contact.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <string_view>

#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/solid_contact_round_solid_hit_rounding_cases.h"

// The tests of how far from the grown surface a sphere's or a cylinder's hit can be, at the edge of MoveParticle's
// premise: every coordinate within 2^40 contact tolerances of the origin (the tolerance of ComputeContactTolerance with
// the particles inside the largest coordinate) and every solid at least 1024 tolerances thick (CheckContactTolerance).
// A start far from a small round solid is where the closed forms lose the most digits: the discriminant of the hit's
// quadratic is a small difference of terms of size |start - center|^2 |velocity|^2.
//
// The reference is double-double arithmetic: a number held as the unevaluated sum high + low of two doubles, about
// 106 bits. Its sums and products of doubles are exact (Knuth's two-sum, and the product's error from one fused
// multiply-add, std::fma, which IEEE 754 rounds once); its other operations round by at most a few units of 2^-106
// (Joldes, Muller and Popescu, "Tight and rigorous error bounds for basic building blocks of double-word arithmetic",
// 2017). Every quantity here is below 2^42 tol, so a squared length of up to 2^84 tol^2 is held to 2^-20 tol^2, and a
// distance near a solid of radius 512 tol or more, (|p|^2 - radius^2) / (|p| + radius), to 2^-30 tol: 2^25 times
// finer than the bound the tests check. long double would not do: on Apple arm64 and with MSVC it is the double itself.
// The reference reads the inputs exactly as the solid and the path give them (the axes as stored, the time as
// returned), so it measures the code's own rounding and nothing else.

namespace fluidity::contact {
namespace {
constexpr SolidMotion kNoMotion = SolidMotion();  // A solid that neither moves nor turns.

// Returns x / y for y != 0, within a few units of 2^-106: a quotient of the highs, then one correction.
DoubleDouble Divide(DoubleDouble x, DoubleDouble y) {
  const double first_quotient = x.high / y.high;
  const DoubleDouble remainder = Subtract(x, Multiply(y, ToDoubleDouble(first_quotient)));
  const double correction = remainder.high / y.high;
  return Renormalize(first_quotient, correction);
}

// Returns the square root of x >= 0, within a few units of 2^-106: the double root, corrected once by the remainder.
DoubleDouble SquareRoot(DoubleDouble x) {
  if (x.high <= 0) {
    return {};
  }
  const double first_root = std::sqrt(x.high);
  const DoubleDouble remainder = Subtract(x, MultiplyExactly(first_root, first_root));
  return Renormalize(first_root, remainder.high / (2 * first_root));
}

// s. The first time the path reaches the grown sphere, or the grown wall of a cylinder taken as unbounded, from the
// exact inputs: the smaller root of |q + w t|^2 = R^2 (for the wall, the x and y parts only), R = radius + tol held
// exactly as a double-double. In double-double the discriminant's cancellation costs 2^-104 |q|^2 |w|^2, which moves
// the root by at most 2^-30 tol along the distance. Requires a start outside and a path that reaches it.
double ComputeReferenceEntryTime(const RoundScene& scene) {
  const Solid& solid = scene.solid;
  DoubleDoubleVec3 own_start = ComputeReferenceOwnPoint(scene, 0);
  const DoubleDoubleVec3 at_one_second = ComputeReferenceOwnPoint(scene, 1);
  DoubleDoubleVec3 own_velocity = {
      .x = Subtract(at_one_second.x, own_start.x),
      .y = Subtract(at_one_second.y, own_start.y),
      .z = Subtract(at_one_second.z, own_start.z),
  };
  if (solid.shape == SolidShape::kCylinder) {
    own_start.z = {};
    own_velocity.z = {};
  }
  const DoubleDouble grown_radius = AddExactly(solid.half_size.x, kTolerance);
  const DoubleDouble speed_squared = Dot(own_velocity, own_velocity);
  const DoubleDouble start_dot_velocity = Dot(own_start, own_velocity);
  const DoubleDouble excess = Subtract(Dot(own_start, own_start), Multiply(grown_radius, grown_radius));
  const DoubleDouble discriminant =
      Subtract(Multiply(start_dot_velocity, start_dot_velocity), Multiply(speed_squared, excess));
  return Divide(excess, Add(Negate(start_dot_velocity), SquareRoot(discriminant))).high;
}

// m. The center of the small round solids of the written scenes, near a corner of the scale.
constexpr Vec3 kCornerCenter = {.x = -0.75, .y = -0.625, .z = -0.875};

// A fixed sphere of radius radius_in_tolerances tol at kCornerCenter.
Solid BuildCornerSphere(double radius_in_tolerances) {
  return ToSolid(Sphere{.center = kCornerCenter, .radius = radius_in_tolerances * kTolerance}, kNoMotion, 1);
}

// A fixed cylinder of radius 1000 tol and half height 2000 tol along z at kCornerCenter.
Solid BuildCornerCylinder() {
  return ToSolid(
      Cylinder{.center = kCornerCenter, .radius = 1000 * kTolerance, .half_height = 2000 * kTolerance},
      kNoMotion,
      1
  );
}

// The scene of a path in one second from start through the point kCornerCenter + across_in_tolerances tol along a
// direction across both that path and the z axis, and on for 3000 tol beyond it.
RoundScene BuildCornerScene(const Solid& solid, Vec3 start, double across_in_tolerances) {
  const Vec3 across = math::Normalized(math::Cross(kCornerCenter - start, {.x = 0, .y = 0, .z = 1}));
  const Vec3 to_aim = kCornerCenter + across * (across_in_tolerances * kTolerance) - start;
  const double aim_distance = math::Length(to_aim);
  return {
      .solid = solid,
      .start = start,
      .velocity = to_aim * ((aim_distance + 3000 * kTolerance) / aim_distance),
  };
}

// Fails the calling test unless the scene's path is hit where the reference distance is tol, within
// kHitDistanceBound, and at the reference entry time within the time the distance takes to close by that bound.
void ExpectHitAtGrownSurface(const RoundScene& scene, std::string_view name) {
  ExpectWithinScale(scene);
  const double expected_elapsed = ComputeReferenceEntryTime(scene);
  // m/s. How fast the distance closes at the entry: over 16 tol of the path before it.
  const double time_step = 16 * kTolerance / math::Length(scene.velocity);  // s
  const double closing_speed = (ComputeReferenceDistance(scene, expected_elapsed - time_step)
                                - ComputeReferenceDistance(scene, expected_elapsed))
      / time_step;
  const FirstHit first_hit = FindSceneHit(scene);
  ASSERT_EQ(first_hit.solid_index, 0U) << name;
  const double distance_at_hit = ComputeReferenceDistance(scene, first_hit.elapsed);
  std::cout << name << ": hit at " << first_hit.elapsed << " s, reference " << expected_elapsed
            << " s, distance at hit - tol " << (distance_at_hit - kTolerance) / kTolerance << " tol\n";
  EXPECT_NEAR(distance_at_hit, kTolerance, kHitDistanceBound) << name;
  EXPECT_NEAR(first_hit.elapsed, expected_elapsed, kHitDistanceBound / closing_speed) << name;
}

TEST(FindFirstHit, FarSmallSphereHeadOn_IsHitAtGrownSurface) {
  // A sphere of radius 1000 tol 2^41.5 tol from the start, the path through its center. Before the fix the
  // discriminant B^2 - A C, a difference of two terms of about 2^83 tol^2 |w|^2 that each round by about
  // 2^30 tol^2 |w|^2, came out too large, and the hit stopped the particle 32,150 tol short of the sphere.
  const RoundScene scene = BuildCornerScene(BuildCornerSphere(1000), kOppositeCorner, 0);
  ASSERT_LT(ComputeSmallestReferenceDistance(scene, 1), 0);
  ExpectHitAtGrownSurface(scene, "far sphere head on");
}

TEST(FindFirstHit, FarSmallSphereCrossedOffCenter_IsHitAtGrownSurface) {
  // A sphere of radius 1500 tol, the path 187.5 tol from its center, 1312.5 tol deep. Before the fix the discriminant
  // rounded below zero and the path went through the sphere with no hit.
  const RoundScene scene = BuildCornerScene(BuildCornerSphere(1500), kOppositeCorner, 187.5);
  ASSERT_LT(ComputeSmallestReferenceDistance(scene, 1), -1300 * kTolerance);
  ExpectHitAtGrownSurface(scene, "far sphere crossed off center");
}

TEST(FindFirstHit, FarSmallSpherePassedBeside_IsFree) {
  // A sphere of radius 1000 tol, the path 1250 tol from its center, 249 tol beyond the grown sphere. Before the fix
  // the discriminant rounded above zero and the path was hit 32,174 tol from the sphere.
  const RoundScene scene = BuildCornerScene(BuildCornerSphere(1000), kOppositeCorner, 1250);
  ExpectWithinScale(scene);
  ASSERT_GT(ComputeSmallestReferenceDistance(scene, 1), 249 * kTolerance);
  EXPECT_EQ(FindSceneHit(scene).solid_index, kNoSolid);
}

TEST(FindFirstHit, FarSmallCylinderWallCrossed_IsHitAtGrownSurface) {
  // A cylinder of radius 1000 tol along z, the path 300 tol from its axis at mid height, 700 tol deep. Before the fix
  // the wall's discriminant rounded below zero, the wall interval was empty and the path went through with no hit.
  const RoundScene scene = BuildCornerScene(BuildCornerCylinder(), kOppositeCorner, 300);
  ASSERT_LT(ComputeSmallestReferenceDistance(scene, 1), -690 * kTolerance);
  ExpectHitAtGrownSurface(scene, "far cylinder wall crossed");
}

TEST(FindFirstHit, FarSmallCylinderWallPassedBeside_IsFree) {
  // The same cylinder from {0.875, 0.6875, 0.9375}, the path 2000 tol from its axis, 1000 tol from the wall. Before
  // the fix the wall's discriminant rounded to about zero, and the path was hit where it passes nearest, 1000 tol from
  // the cylinder.
  const RoundScene scene = BuildCornerScene(BuildCornerCylinder(), {.x = 0.875, .y = 0.6875, .z = 0.9375}, 2000);
  ExpectWithinScale(scene);
  ASSERT_GT(ComputeSmallestReferenceDistance(scene, 1), 999 * kTolerance);
  EXPECT_EQ(FindSceneHit(scene).solid_index, kNoSolid);
}

// Moves a particle along the scene's path with MoveParticle at kTolerance and fails the calling test unless it hits
// the solid once and ends at the rest gap, within kHitDistanceBound by the reference distance.
void ExpectMoveStopsAtRestGap(const RoundScene& scene, std::string_view name) {
  ExpectWithinScale(scene);
  SolidContactInputs inputs;
  inputs.start_position = scene.start;
  inputs.velocity.for_this_step = scene.velocity;
  inputs.velocity.for_next_step = scene.velocity;
  inputs.solids = &scene.solid;
  inputs.solid_count = 1;
  inputs.step_duration = scene.duration;
  inputs.contact_tolerance = kTolerance;
  SolidContactResult result;
  ContactFailure failure;
  ASSERT_EQ(MoveParticle(inputs, &result, &failure), ContactError::kNone) << name;
  EXPECT_EQ(result.hit_count, 1) << name;
  const RoundScene at_end = {.solid = scene.solid, .start = result.position, .velocity = {}, .duration = 0};
  const double end_distance = ComputeReferenceDistance(at_end, 0);
  std::cout << name << ": end distance - tol " << (end_distance - kTolerance) / kTolerance << " tol\n";
  EXPECT_NEAR(end_distance, kTolerance, kHitDistanceBound) << name;
}

// m. A start 0.3 m from kCornerCenter along each axis, 2^39.1 tol away, so the path's sum of absolute axes, 0.9 m
// and 3000 tol, is within the 2^40 tol MoveParticle allows at a hit.
constexpr Vec3 kNearerStart = {.x = -0.45, .y = -0.325, .z = -0.575};

TEST(MoveParticle, FarSmallSphereHeadOn_StopsAtRestGap) {
  // Before the fix the particle went through the sphere of radius 1000 tol and ended 2000 tol beyond it, with no hit
  // and no failure.
  ExpectMoveStopsAtRestGap(BuildCornerScene(BuildCornerSphere(1000), kNearerStart, 0), "move into far sphere");
}

TEST(MoveParticle, FarSmallCylinderWallHeadOn_StopsAtRestGap) {
  // Before the fix the move failed with kTooCloseToCall.
  ExpectMoveStopsAtRestGap(BuildCornerScene(BuildCornerCylinder(), kNearerStart, 0), "move into far cylinder");
}

// Where a scene of ExpectNoJumpAcrossTheSwitch starts: the solid and when its sweep of 1 s starts in the step.
struct SwitchSceneSolid {
  Solid solid;
  double start_elapsed = 0;  // s
};

// The scene of a start at Sqrt(2) R (1 + step 2^-40) from the centre or the axis of the solid, on one own direction,
// toward a point 0.3 R off the centre or the axis.
RoundScene BuildSwitchScene(const SwitchSceneSolid& switch_solid, int step) {
  const Solid& solid = switch_solid.solid;
  const double start_elapsed = switch_solid.start_elapsed;  // s
  const SolidPose pose = ComputeSolidPose(solid, start_elapsed);
  const Vec3 own_direction =
      math::Normalized(Vec3{.x = 0.6, .y = -0.48, .z = solid.shape == SolidShape::kSphere ? 0.64 : 0});
  const Vec3 own_across = math::Normalized(math::Cross(own_direction, {.x = 0, .y = 0, .z = 1}));
  const double grown_radius = solid.half_size.x + kTolerance;  // m
  const double distance = std::numbers::sqrt2 * grown_radius * (1 + step * 0x1p-40);  // m
  const Vec3 own_start = own_direction * distance;
  const Vec3 own_aim = own_across * (0.3 * grown_radius);
  return {
      .solid = solid,
      .start = pose.origin + pose.axes * own_start,
      .velocity = solid.motion.linear_velocity + pose.axes * (own_aim - own_start),
      .duration = 1,
      .start_elapsed = start_elapsed,
  };
}

// m. The reference distance minus tol at the scene's first hit, or NaN without one, so every check fails.
double ComputeHitDistanceError(const RoundScene& scene) {
  const FirstHit first_hit = FindSceneHit(scene);
  if (first_hit.solid_index != 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return ComputeReferenceDistance(scene, first_hit.elapsed) - kTolerance;
}

// Fails the calling test unless the starts of BuildSwitchScene, step from -16 to 16, take both forms of the
// discriminant and are each hit within kHitDistanceBound of the grown surface, with no jump between neighbouring
// starts above kHitDistanceBound.
void ExpectNoJumpAcrossTheSwitch(const Solid& solid, double start_elapsed) {
  const SwitchSceneSolid switch_solid = {.solid = solid, .start_elapsed = start_elapsed};
  int near_count = 0;
  double previous_error = ComputeHitDistanceError(BuildSwitchScene(switch_solid, -17));  // m
  for (int step = -16; step <= 16; ++step) {
    const RoundScene scene = BuildSwitchScene(switch_solid, step);
    ExpectWithinScale(scene);
    near_count += ComputeIsNearForm(scene) ? 1 : 0;
    const double error = ComputeHitDistanceError(scene);  // m
    EXPECT_LE(std::abs(error), kHitDistanceBound) << "step " << step;
    EXPECT_LE(std::abs(error - previous_error), kHitDistanceBound) << "step " << step;
    previous_error = error;
  }
  EXPECT_GT(near_count, 0);
  EXPECT_LT(near_count, 33);
}

TEST(FindFirstHit, StartsEitherSideOfTheDiscriminantSwitch_HitWithoutAJump) {
  // A sphere and a cylinder of radius 2^30 tol near a corner, so the switch lies 2^30.5 tol from the centre and the
  // largest coordinate is 2^40 tol; fixed, then moving with the sweep starting 0.75 s into the step.
  const Vec3 center = {.x = -0.5, .y = 0.25, .z = -0.375};  // m
  const double radius = 0x1p30 * kTolerance;  // m
  const SolidMotion moving = {.linear_velocity = {.x = 0.03125, .y = -0.0625, .z = 0.046875}, .angular_velocity = {}};
  const Vec3 axis = math::Normalized(Vec3{.x = 0.36, .y = 0.48, .z = 0.8});
  for (const SolidMotion& motion : {kNoMotion, moving}) {
    const double start_elapsed = motion.linear_velocity.x == 0 ? 0 : 0.75;  // s
    ExpectNoJumpAcrossTheSwitch(ToSolid(Sphere{.center = center, .radius = radius}, motion, 1), start_elapsed);
    ExpectNoJumpAcrossTheSwitch(
        ToSolid(Cylinder{.center = center, .axis = axis, .radius = radius, .half_height = radius}, motion, 1),
        start_elapsed
    );
  }
}

TEST(FindFirstHit, FarSmallMovingSphereLaterInTheStep_IsHitAtGrownSurface) {
  // The head-on scene of a sphere of radius 1000 tol, the sphere moving and the sweep starting 0.75 s into a step of
  // 1 s: the pose at 0.75 s, origin + linear_velocity * 0.75, rounds by about eps L and moves the solid against
  // the path (FindSphereHit, Rounding).
  const SolidMotion motion = {.linear_velocity = {.x = 0.0625, .y = 0.03125, .z = -0.046875}, .angular_velocity = {}};
  const Solid sphere = ToSolid(Sphere{.center = kCornerCenter, .radius = 1000 * kTolerance}, motion, 1);
  const double start_elapsed = 0.75;  // s
  const Vec3 center_at_start = ComputeSolidPose(sphere, start_elapsed).origin;  // m
  const Vec3 to_center = center_at_start - kOppositeCorner;  // m
  const double center_distance = math::Length(to_center);  // m
  const Vec3 relative_velocity = to_center * ((center_distance + 3000 * kTolerance) / center_distance / 0.25);  // m/s
  const RoundScene scene = {
      .solid = sphere,
      .start = kOppositeCorner,
      .velocity = relative_velocity + motion.linear_velocity,
      .duration = 0.25,
      .start_elapsed = start_elapsed,
  };
  ExpectHitAtGrownSurface(scene, "far moving sphere later in the step");
}

}  // namespace
}  // namespace fluidity::contact
