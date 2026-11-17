#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <string_view>

#include "fluidity/base/axis.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_test_cases.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

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

// m. The contact tolerance of a scene whose every coordinate is within 1 m of the origin, at its smallest:
// ComputeContactTolerance gives at least 2^-40 of the largest coordinate. 1 m is 2^40 tol.
constexpr double kTolerance = 0x1p-40;
// m. The bound on how far a sphere's or a cylinder's hit is from the grown surface, along the true distance: counted to
// first order, 14 machine epsilons of the largest own coordinate along the path and 3 of the largest coordinate
// (FindSphereHit, Rounding), about 0.013 tol at this scale.
constexpr double kHitDistanceBound = kTolerance / 64;
// m. How far a rim hit may lie beyond tol: where the rim search stops (kRimHitGapShare).
constexpr double kRimHitGap = kTolerance / 1024;
// m. The cell width the scale check reads: wide enough, so only the solids' thickness decides.
constexpr double kCellWidth = 0x1p-20;
// The largest coordinate a scene here may reach, m: 2^40 tol.
constexpr double kLargestCoordinate = 1;
constexpr SolidMotion kNoMotion = SolidMotion();  // A solid that neither moves nor turns.

// One sweep from the start of the step against one solid.
struct RoundScene {
  Solid solid;
  Vec3 start{};  // m, world.
  Vec3 velocity{};  // m/s, world.
  double duration = 1;  // s
  double start_elapsed = 0;  // s. When the sweep starts in the step, which poses the solid then.
};

// --- The reference: double-double arithmetic ---

// A number held as the unevaluated sum high + low, with |low| at most half a unit in the last place of high.
struct DoubleDouble {
  double high = 0;
  double low = 0;
};

// Returns a + b exactly (Knuth's two-sum: six operations, no condition on the sizes).
DoubleDouble AddExactly(double a, double b) {
  const double sum = a + b;
  const double b_share = sum - a;
  const double a_share = sum - b_share;
  const double rounding = (a - a_share) + (b - b_share);
  return {.high = sum, .low = rounding};
}

// Returns a * b exactly: the fused multiply-add rounds a * b - product once, and that difference is a double.
DoubleDouble MultiplyExactly(double a, double b) {
  const double product = a * b;
  return {.high = product, .low = std::fma(a, b, -product)};
}

// Returns high + low exactly as a normalized pair, when |high| >= |low| (Dekker's fast two-sum).
DoubleDouble Renormalize(double high, double low) {
  const double sum = high + low;
  return {.high = sum, .low = low - (sum - high)};
}

// Returns x + y, within 3 units of 2^-106 of the sum (the accurate double-word sum).
DoubleDouble Add(DoubleDouble x, DoubleDouble y) {
  const DoubleDouble high_sum = AddExactly(x.high, y.high);
  const DoubleDouble low_sum = AddExactly(x.low, y.low);
  const DoubleDouble first = Renormalize(high_sum.high, high_sum.low + low_sum.high);
  return Renormalize(first.high, first.low + low_sum.low);
}

DoubleDouble Negate(DoubleDouble x) {
  return {.high = -x.high, .low = -x.low};
}

DoubleDouble Subtract(DoubleDouble x, DoubleDouble y) {
  return Add(x, Negate(y));
}

DoubleDouble ToDoubleDouble(double value) {
  return {.high = value, .low = 0};
}

// Returns x * y, within 7 units of 2^-106 of the product.
DoubleDouble Multiply(DoubleDouble x, DoubleDouble y) {
  const DoubleDouble high_product = MultiplyExactly(x.high, y.high);
  const double cross_terms = std::fma(x.high, y.low, x.low * y.high);
  return Renormalize(high_product.high, high_product.low + cross_terms);
}

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

// A vector of double-doubles.
struct DoubleDoubleVec3 {
  DoubleDouble x;
  DoubleDouble y;
  DoubleDouble z;
};

DoubleDouble Dot(DoubleDoubleVec3 left, DoubleDoubleVec3 right) {
  return Add(Add(Multiply(left.x, right.x), Multiply(left.y, right.y)), Multiply(left.z, right.z));
}

// The point of a path at elapsed seconds into the sweep, from the solid's origin at that moment, in the solid's own
// axes, from the exact inputs: A^T ((start - origin) + (velocity - linear_velocity) * elapsed - linear_velocity *
// start_elapsed), with A the solid's axes as stored. The solid does not spin.
DoubleDoubleVec3 ComputeReferenceOwnPoint(const RoundScene& scene, double elapsed) {
  const Solid& solid = scene.solid;
  const Vec3 start = scene.start;
  const Vec3 velocity = scene.velocity;
  const Vec3 origin = solid.pose_at_step_start.origin;
  const Vec3 linear_velocity = solid.motion.linear_velocity;
  // One world axis, each difference and the pose's product exact.
  const auto compute_world_axis = [&](Axis axis) {
    const DoubleDouble offset = AddExactly(ComputeCoordinate(start, axis), -ComputeCoordinate(origin, axis));
    const DoubleDouble relative_speed =
        AddExactly(ComputeCoordinate(velocity, axis), -ComputeCoordinate(linear_velocity, axis));
    const DoubleDouble solid_travel_before =
        MultiplyExactly(ComputeCoordinate(linear_velocity, axis), scene.start_elapsed);
    return Subtract(Add(offset, Multiply(relative_speed, ToDoubleDouble(elapsed))), solid_travel_before);
  };
  const DoubleDoubleVec3 world = {
      .x = compute_world_axis(Axis::kX),
      .y = compute_world_axis(Axis::kY),
      .z = compute_world_axis(Axis::kZ),
  };
  const Mat3 axes = solid.pose_at_step_start.axes;
  // Own axis k is column k of the axes: p_k = Dot(column k, world).
  const auto compute_own_axis = [&world](double column_x, double column_y, double column_z) {
    return Dot({.x = ToDoubleDouble(column_x), .y = ToDoubleDouble(column_y), .z = ToDoubleDouble(column_z)}, world);
  };
  return {
      .x = compute_own_axis(axes.xx, axes.yx, axes.zx),
      .y = compute_own_axis(axes.xy, axes.yy, axes.zy),
      .z = compute_own_axis(axes.xz, axes.yz, axes.zz),
  };
}

// Returns sqrt(squared_length) - length as a double, for squared_length >= 0 and length > 0, without cancellation:
// (squared_length - length^2) / (sqrt(squared_length) + length).
double ComputeRootMinusLength(DoubleDouble squared_length, double length) {
  const DoubleDouble excess = Subtract(squared_length, MultiplyExactly(length, length));
  return excess.high / (std::sqrt(squared_length.high) + length);
}

// m. The distance from the path's point at elapsed seconds to the solid, a sphere or a cylinder, at that moment, from
// the exact inputs, within 2^-30 tol near the solid: the cylinder's formula is the one of ComputeDistanceToCylinder.
double ComputeReferenceDistance(const RoundScene& scene, double elapsed) {
  const Solid& solid = scene.solid;
  const DoubleDoubleVec3 own_point = ComputeReferenceOwnPoint(scene, elapsed);
  if (solid.shape == SolidShape::kSphere) {
    return ComputeRootMinusLength(Dot(own_point, own_point), solid.half_size.x);
  }
  const DoubleDouble axis_distance_squared =
      Add(Multiply(own_point.x, own_point.x), Multiply(own_point.y, own_point.y));
  const double wall_distance = ComputeRootMinusLength(axis_distance_squared, solid.half_size.x);
  const DoubleDouble height = own_point.z.high < 0 ? Negate(own_point.z) : own_point.z;
  const double cap_distance = Subtract(height, ToDoubleDouble(solid.half_size.z)).high;
  if (wall_distance > 0 && cap_distance > 0) {
    return std::hypot(wall_distance, cap_distance);
  }
  if (wall_distance > 0) {
    return wall_distance;
  }
  if (cap_distance > 0) {
    return cap_distance;
  }
  return std::max(wall_distance, cap_distance);
}

// Returns whether the path's point at elapsed seconds is beside a cylinder's rim by the reference: beyond both the wall
// and a cap's plane, where the rim search, not the sharp entry, decides the hit.
bool IsBesideRimByReference(const RoundScene& scene, double elapsed) {
  const Solid& solid = scene.solid;
  if (solid.shape != SolidShape::kCylinder) {
    return false;
  }
  const DoubleDoubleVec3 own_point = ComputeReferenceOwnPoint(scene, elapsed);
  const DoubleDouble axis_distance_squared =
      Add(Multiply(own_point.x, own_point.x), Multiply(own_point.y, own_point.y));
  const double wall_distance = ComputeRootMinusLength(axis_distance_squared, solid.half_size.x);
  const DoubleDouble height = own_point.z.high < 0 ? Negate(own_point.z) : own_point.z;
  const double cap_distance = Subtract(height, ToDoubleDouble(solid.half_size.z)).high;
  return wall_distance > 0 && cap_distance > 0;
}

// m. The smallest reference distance from the path to the solid between 0 and end_elapsed seconds. The true distance
// to a convex solid is convex along a straight line, so a search that drops a third of the interval each time finds
// the one minimum; 160 steps leave (2/3)^160, below 2^-93 of the interval, under 2^-51 tol of a path of 2^42 tol.
double ComputeSmallestReferenceDistance(const RoundScene& scene, double end_elapsed) {
  double low = 0;
  double high = end_elapsed;
  for (int iteration = 0; iteration < 160; ++iteration) {
    const double lower_third = low + (high - low) / 3;
    const double upper_third = high - (high - low) / 3;
    const double at_lower_third = ComputeReferenceDistance(scene, lower_third);
    const double at_upper_third = ComputeReferenceDistance(scene, upper_third);
    if (at_lower_third < at_upper_third) {
      high = upper_third;
    } else {
      low = lower_third;
    }
  }
  return std::min({
      ComputeReferenceDistance(scene, 0),
      ComputeReferenceDistance(scene, low),
      ComputeReferenceDistance(scene, end_elapsed),
  });
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

// --- The scenes ---

// Fails the calling test unless the scene is within MoveParticle's premise at kTolerance: the solid's largest
// coordinate over the step, its reach included, and the path's two ends within 2^40 tol of the origin; and unless
// CheckContactTolerance accepts it, the solid at least 1024 tol thick.
void ExpectWithinScale(const RoundScene& scene) {
  const double step_duration = scene.start_elapsed + scene.duration;  // s
  const ContactScale scale = ComputeContactScale(&scene.solid, 1, step_duration, kCellWidth);
  EXPECT_LE(scale.largest_coordinate, kLargestCoordinate);
  EXPECT_EQ(CheckContactTolerance(scale, kTolerance), ContactError::kNone);
  const Vec3 end = scene.start + scene.velocity * scene.duration;
  for (const double coordinate : {scene.start.x, scene.start.y, scene.start.z, end.x, end.y, end.z}) {
    EXPECT_LE(std::abs(coordinate), kLargestCoordinate);
  }
}

// Runs FindFirstHit on the scene's sweep at kTolerance and fails the calling test unless it succeeds.
FirstHit FindSceneHit(const RoundScene& scene) {
  SolidContactInputs inputs;
  inputs.solids = &scene.solid;
  inputs.solid_count = 1;
  inputs.step_duration = scene.start_elapsed + scene.duration;
  inputs.contact_tolerance = kTolerance;
  FirstHit first_hit;
  const ContactError error =
      FindFirstHit(inputs, scene.start, scene.velocity, scene.start_elapsed, scene.duration, &first_hit);
  EXPECT_EQ(error, ContactError::kNone);
  return first_hit;
}

// m. The center of the small round solids of the written scenes, near a corner of the scale.
constexpr Vec3 kCornerCenter = {.x = -0.75, .y = -0.625, .z = -0.875};
// m. A start near the opposite corner: 2.80 m from kCornerCenter, 2^41.5 tol.
constexpr Vec3 kOppositeCorner = {.x = 0.875, .y = 0.75, .z = 0.9375};

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

// Returns whether FindSphereHit or ComputeTimeInsideGrownWall takes its near form for this start: C <= R^2, with C
// computed as they compute it, from the own start at the pose of start_elapsed. Only the solid's axes as stored are
// used, so the solid does not spin.
bool ComputeIsNearForm(const RoundScene& scene) {
  const Solid& solid = scene.solid;
  const SolidPose pose = ComputeSolidPose(solid, scene.start_elapsed);
  const Vec3 own_start = math::Transpose(pose.axes) * (scene.start - pose.origin);  // m
  const double radius = solid.half_size.x;  // m
  const bool is_sphere = solid.shape == SolidShape::kSphere;
  const double center_distance = is_sphere ? math::Sqrt(math::Dot(own_start, own_start))
                                           : math::Sqrt(own_start.x * own_start.x + own_start.y * own_start.y);  // m
  const double start_gap = center_distance - radius - kTolerance;  // m
  const double grown_radius = radius + kTolerance;  // m
  return start_gap * (center_distance + grown_radius) <= grown_radius * grown_radius;
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

// --- Generated sets over the whole scale ---

// What a generated set found.
struct GeneratedCounts {
  int built = 0;
  int correct = 0;  // Every check of the case held.
  int hits = 0;
  int must_hit = 0;  // The path's reference distance falls below tol - kHitDistanceBound before its end.
  int rim_hits = 0;
  int near_form = 0;  // Starts at which the closed form takes B^2 - A C, C <= R^2 (ComputeIsNearForm).
  double largest_distance_error = 0;  // m. The largest |distance at hit - tol| over the hits beside no rim.
};

// Checks one case against the reference and counts it: the scene is within the scale; a hit has its reference
// distance within kHitDistanceBound of tol (beside a rim, up to kRimHitGap more, where the rim search stops); and the
// path before it, or the whole path when it is free, never comes nearer than tol - kHitDistanceBound. Returns whether
// every check held.
bool CheckAgainstReference(const RoundScene& scene, GeneratedCounts* counts) {
  ExpectWithinScale(scene);
  const FirstHit first_hit = FindSceneHit(scene);
  const bool is_hit = first_hit.solid_index != kNoSolid;
  const double end_elapsed = is_hit ? first_hit.elapsed : scene.duration;  // s
  ++counts->built;
  if (ComputeIsNearForm(scene)) {
    ++counts->near_form;
  }
  const double smallest_over_path = ComputeSmallestReferenceDistance(scene, scene.duration);
  if (smallest_over_path < kTolerance - kHitDistanceBound) {
    ++counts->must_hit;
  }
  const bool ends_within_sweep = end_elapsed >= 0 && end_elapsed <= scene.duration;
  const double smallest_before_end = ComputeSmallestReferenceDistance(scene, end_elapsed);
  const bool stays_outside_before_end = smallest_before_end >= kTolerance - kHitDistanceBound;
  bool is_hit_at_grown_surface = true;
  if (is_hit) {
    ++counts->hits;
    const double distance_at_hit = ComputeReferenceDistance(scene, first_hit.elapsed);
    // Beside a rim by the reference, not by the face the code reports.
    const bool is_rim = IsBesideRimByReference(scene, first_hit.elapsed);
    const double largest_distance = kTolerance + kHitDistanceBound + (is_rim ? kRimHitGap : 0);  // m
    is_hit_at_grown_surface = distance_at_hit <= largest_distance;
    if (is_rim) {
      ++counts->rim_hits;
    } else {
      counts->largest_distance_error = std::max(counts->largest_distance_error, std::abs(distance_at_hit - kTolerance));
    }
  }
  const bool is_correct = ends_within_sweep && stays_outside_before_end && is_hit_at_grown_surface;
  if (is_correct) {
    ++counts->correct;
  }
  return is_correct;
}

// How a generated path is aimed, in the solid's own axes.
enum class AimKind : std::uint8_t {
  kNearCenter,  // Through a point within half the radius of the center or the axis.
  kGrazing,  // Past the center or the axis at the grown radius, within 8 tol.
  kAnywhere,  // Past the center or the axis at up to 1.5 times the grown radius.
};

// m. How far along the straight line own_line.start + own_line.velocity * s (a unit velocity, so s is a length) it
// enters the sphere, or the wall of a cylinder taken as unbounded, of radius grown_radius, in doubles: only to place a
// generated path's end near it.
double ComputeApproximateEntryDistance(SolidShape shape, OwnPath own_line, double grown_radius) {
  Vec3 across_start = own_line.start;
  Vec3 across_direction = own_line.velocity;
  if (shape == SolidShape::kCylinder) {
    across_start.z = 0;
    across_direction.z = 0;
  }
  const double across_share = math::Length(across_direction);  // No unit.
  const double along = -math::Dot(across_start, across_direction) / across_share;  // m
  const double closest_squared = math::Dot(across_start, across_start) - along * along;  // m^2
  const double half_chord = std::sqrt(std::max(grown_radius * grown_radius - closest_squared, 0.0));  // m
  return (along - half_chord) / across_share;
}

// m. How far from the centre or the axis a generated path of this aim kind passes, for offset_draw in [-1, 1].
double ComputeAcrossOffset(AimKind aim_kind, double radius, double offset_draw) {
  const double grown_radius = radius + kTolerance;  // m
  switch (aim_kind) {
    case AimKind::kNearCenter:
      return 0.5 * radius * offset_draw;
    case AimKind::kGrazing:
      return grown_radius + 8 * kTolerance * offset_draw;
    case AimKind::kAnywhere:
      break;
  }
  return 1.5 * grown_radius * offset_draw;
}

// Where a generated path starts.
enum class StartKind : std::uint8_t {
  kAnywhere,  // Anywhere within 0.999 m of the origin.
  kOppositeCorner,  // Near the corner opposite the solid: up to 2 Sqrt(3) 2^40 tol from it.
  kNearSurface,  // From 2 tol to 2 tol + 0.4 R beyond the grown sphere or wall.
  kAtSwitch,  // Sqrt(2) R from the centre or the axis, to within 2^-1..2^-51 of it either way: where FindSphereHit and
              // ComputeTimeInsideGrownWall change the discriminant's form.
};

// The start kind of a draw in [0, 1): anywhere 30 %, the opposite corner 30 %, near the surface 20 %, at the switch 20
// %.
StartKind ToStartKind(double draw) {
  if (draw < 0.3) {
    return StartKind::kAnywhere;
  }
  if (draw < 0.6) {
    return StartKind::kOppositeCorner;
  }
  if (draw < 0.8) {
    return StartKind::kNearSurface;
  }
  return StartKind::kAtSwitch;
}

// Builds one generated case: a fixed or moving sphere or cylinder of radius 2^9 tol (a diameter of 1024 tol, the
// thinnest solid accepted) up to the largest that fits within 1 m with its reach and motion, about 2^39 tol; a sweep of
// 2^-20 to 2^8 s that starts at 0 or later in the step; a start anywhere, at the opposite corner, just outside the
// grown surface, or at the near/far switch; a path through the solid or to within 64 tol of its grown surface. Draw
// order: the radius; the half height share (drawn for a sphere too); the duration; whether the sweep starts later and
// when; whether the solid moves, then its velocity (three NextSigned()); the start kind; the origin (three
// NextSigned()); for a cylinder the axis (three NextSigned()); the start (three NextSigned(), then one more for its
// distance, drawn again while within 2 tol of the solid); the aim kind and its offset; the aim's height, whether it
// aims at a rim, and two offsets near the rim; whether the path ends near the entry, and the end's offset.
RoundScene BuildGeneratedCaseOnce(SolidShape shape, CaseGenerator* generator) {
  const bool is_sphere = shape == SolidShape::kSphere;
  const double drawn_radius = std::ldexp(kTolerance, 9) * std::exp2(30 * generator->Next());  // m
  const double half_height_share = 1 + 2 * generator->Next();  // The height is no thinner than the diameter.
  // m. The reach ComputeContactScale counts, Length(half_size), is reach_share times the radius; it stays within
  // 0.92 m, so the solid and its motion of at most 0.0625 m per axis stay within 1 m.
  const double reach_share = std::sqrt(2 + (is_sphere ? 1 : half_height_share * half_height_share));
  const double radius = std::min(drawn_radius, 0.92 / reach_share);  // m
  const double half_height = radius * half_height_share;  // m
  const double reach = reach_share * radius;  // m
  const double drawn_duration = std::exp2(-20 + 28 * generator->Next());  // s
  const bool starts_later = generator->Next() < 0.5;
  const double start_elapsed = starts_later ? 2 * drawn_duration * generator->Next() : 0;  // s
  // s. The time left in a step of start_elapsed + drawn_duration, as a move computes it, so the sweep stays in the
  // step.
  const double duration = (start_elapsed + drawn_duration) - start_elapsed;
  const bool is_moving = generator->Next() < 0.5;
  const Vec3 drawn_solid_velocity = {
      .x = generator->NextSigned(),
      .y = generator->NextSigned(),
      .z = generator->NextSigned(),
  };
  // m/s. At most 0.0625 m per axis over the whole step.
  const Vec3 solid_velocity = is_moving ? (0.0625 / (start_elapsed + duration)) * drawn_solid_velocity : Vec3{};
  const StartKind start_kind = ToStartKind(generator->Next());
  const double origin_room = kLargestCoordinate - reach - 0.07;  // m
  const Vec3 drawn_origin = {
      .x = generator->NextSigned(),
      .y = generator->NextSigned(),
      .z = generator->NextSigned(),
  };
  const Vec3 corner_sign = {
      .x = std::copysign(1.0, drawn_origin.x),
      .y = std::copysign(1.0, drawn_origin.y),
      .z = std::copysign(1.0, drawn_origin.z),
  };
  const bool is_at_corner = start_kind == StartKind::kOppositeCorner;
  const Vec3 origin = is_at_corner ? -origin_room * corner_sign : origin_room * drawn_origin;  // m, at time 0.
  Vec3 axis = {.x = 0, .y = 0, .z = 1};
  if (!is_sphere) {
    const Vec3 drawn_axis = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
    axis = math::Normalized(drawn_axis + Vec3{.x = 0, .y = 0, .z = 0.25});
  }
  const SolidMotion motion = {.linear_velocity = solid_velocity, .angular_velocity = {}};
  const Solid solid = is_sphere
      ? ToSolid(Sphere{.center = origin, .radius = radius}, motion, 1)
      : ToSolid(Cylinder{.center = origin, .axis = axis, .radius = radius, .half_height = half_height}, motion, 1);
  const Mat3 axes = solid.pose_at_step_start.axes;
  const Vec3 origin_at_start = ComputeSolidPose(solid, start_elapsed).origin;  // m
  const double grown_radius = radius + kTolerance;  // m

  Vec3 start{};  // m
  RoundScene probe = {.solid = solid, .start = {}, .velocity = {}, .duration = 0, .start_elapsed = start_elapsed};
  do {
    const Vec3 drawn_start = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
    const double distance_draw = generator->Next();
    switch (start_kind) {
      case StartKind::kAnywhere:
        start = 0.999 * drawn_start;
        break;
      case StartKind::kOppositeCorner:
        start = {
            .x = 0.999 * corner_sign.x,
            .y = 0.999 * corner_sign.y,
            .z = (0.9 + 0.099 * drawn_start.z) * corner_sign.z,
        };
        break;
      case StartKind::kNearSurface:
      case StartKind::kAtSwitch: {
        // In own axes: a direction from the centre, or across the axis at a height within the caps' reach.
        Vec3 own_direction = drawn_start;
        double own_height = 0;  // m
        if (!is_sphere) {
          own_direction.z = 0;
          own_height = half_height * drawn_start.z;
        }
        own_direction = math::Normalized(own_direction);
        const double near_distance =
            grown_radius + 2 * kTolerance + 0.4 * grown_radius * std::exp2(-45 * distance_draw);
        const double switch_share = std::copysign(std::exp2(-1 - 50 * distance_draw), drawn_start.x);
        const double switch_distance = std::numbers::sqrt2 * grown_radius * (1 + switch_share);
        const double distance = start_kind == StartKind::kNearSurface ? near_distance : switch_distance;  // m
        const Vec3 own_start = own_direction * distance + Vec3{.x = 0, .y = 0, .z = own_height};
        start = origin_at_start + axes * own_start;
        break;
      }
    }
    probe.start = start;
  } while (ComputeReferenceDistance(probe, 0) <= 2 * kTolerance);

  // In own axes at the sweep's start: the start from the origin, and a unit direction across the line from it to the
  // center or the axis.
  const Vec3 own_start = math::Transpose(axes) * (start - origin_at_start);  // m
  Vec3 own_toward_center = -own_start;  // m
  if (!is_sphere) {
    own_toward_center.z = 0;
  }
  const Vec3 side_direction = is_sphere ? Vec3{.x = 0.6, .y = 0, .z = 0.8} : Vec3{.x = 0, .y = 0, .z = 1};
  const Vec3 own_across = math::Normalized(math::Cross(own_toward_center, side_direction));
  const double aim_draw = generator->Next();
  AimKind aim_kind = AimKind::kAnywhere;
  if (aim_draw < 0.25) {
    aim_kind = AimKind::kNearCenter;
  } else if (aim_draw < 0.75) {
    aim_kind = AimKind::kGrazing;
  }
  const double offset_draw = generator->NextSigned();
  double across_offset = ComputeAcrossOffset(aim_kind, radius, offset_draw);  // m
  // For a cylinder, the aim's height: within the caps mostly; for one case in five, within 2 tol of a rim's grown
  // edge, beside the grown wall by up to 2 tol either way.
  const double height_draw = generator->NextSigned();
  const bool aims_at_rim = !is_sphere && generator->Next() < 0.2;
  const double rim_height_draw = generator->NextSigned();
  const double rim_across_draw = generator->NextSigned();
  double aim_height = is_sphere ? 0 : 0.8 * half_height * height_draw;  // m
  if (aims_at_rim) {
    aim_height = std::copysign(half_height + kTolerance + 2 * kTolerance * rim_height_draw, height_draw);
    across_offset = std::copysign(grown_radius + 2 * kTolerance * rim_across_draw, offset_draw);
  }
  const Vec3 own_aim = own_across * across_offset + Vec3{.x = 0, .y = 0, .z = aim_height};  // m

  // The path: toward the aim and on past it by the radius and 100 tol, or to within 64 tol of the entry.
  const Vec3 own_to_aim = own_aim - own_start;  // m
  const double aim_distance = math::Length(own_to_aim);  // m
  const bool ends_near_entry = generator->Next() < 0.5;
  const double end_draw = generator->NextSigned();
  double path_length = aim_distance + radius + 100 * kTolerance;  // m
  if (ends_near_entry) {
    const OwnPath own_line = {.start = own_start, .velocity = own_to_aim * (1 / aim_distance)};
    path_length = ComputeApproximateEntryDistance(shape, own_line, grown_radius) + 64 * kTolerance * end_draw;
  }
  const Vec3 own_velocity = own_to_aim * (std::max(path_length, kTolerance) / aim_distance / duration);  // m/s
  return {
      .solid = solid,
      .start = start,
      .velocity = axes * own_velocity + solid_velocity,
      .duration = duration,
      .start_elapsed = start_elapsed,
  };
}

// Builds one generated case (BuildGeneratedCaseOnce), drawn again while the path's start or end has a coordinate
// beyond 0.999 m: a large solid near the edge leaves some paths no room.
RoundScene BuildGeneratedCase(SolidShape shape, CaseGenerator* generator) {
  while (true) {
    const RoundScene scene = BuildGeneratedCaseOnce(shape, generator);
    const Vec3 end = scene.start + scene.velocity * scene.duration;
    const double largest_path_coordinate = std::max({
        std::abs(scene.start.x),
        std::abs(scene.start.y),
        std::abs(scene.start.z),
        std::abs(end.x),
        std::abs(end.y),
        std::abs(end.z),
    });
    if (largest_path_coordinate <= 0.999) {
      return scene;
    }
  }
}

// Runs case_count generated cases of one shape, fails the calling test on each mistake with its case, and returns
// the counts.
GeneratedCounts CheckGeneratedCases(SolidShape shape, int case_count) {
  CaseGenerator generator;
  GeneratedCounts counts;
  for (int case_index = 0; case_index < case_count; ++case_index) {
    const RoundScene scene = BuildGeneratedCase(shape, &generator);
    const bool is_correct = CheckAgainstReference(scene, &counts);
    EXPECT_TRUE(is_correct) << "case " << case_index;
  }
  return counts;
}

void PrintCounts(const char* name, const GeneratedCounts& counts) {
  std::cout << name << ": built " << counts.built << ", correct " << counts.correct << ", hit " << counts.hits
            << ", must hit " << counts.must_hit << ", rim hits " << counts.rim_hits << ", near form "
            << counts.near_form << ", largest |distance at hit - tol| " << counts.largest_distance_error / kTolerance
            << " tol\n";
}

TEST(FindFirstHit, GeneratedFarSphereSweepsAcrossTheScale_HitWithinBoundOfGrownSurface) {
  const GeneratedCounts counts = CheckGeneratedCases(SolidShape::kSphere, 10000);
  PrintCounts("generated sphere sweeps across the scale", counts);
  EXPECT_EQ(counts.correct, counts.built);
  // Enough paths must hit for the check to mean something, and enough are free.
  EXPECT_GT(counts.must_hit, 5000);
  EXPECT_LT(counts.hits, 7000);
  // Both forms of the discriminant, each on many starts.
  EXPECT_GT(counts.near_form, 1000);
  EXPECT_LT(counts.near_form, 9000);
}

TEST(FindFirstHit, GeneratedFarCylinderSweepsAcrossTheScale_HitWithinBoundOfGrownSurface) {
  const GeneratedCounts counts = CheckGeneratedCases(SolidShape::kCylinder, 10000);
  PrintCounts("generated cylinder sweeps across the scale", counts);
  EXPECT_EQ(counts.correct, counts.built);
  EXPECT_GT(counts.must_hit, 5000);
  EXPECT_LT(counts.hits, 7000);
  EXPECT_GT(counts.near_form, 1000);
  EXPECT_LT(counts.near_form, 9000);
  EXPECT_GT(counts.rim_hits, 60);
}

}  // namespace
}  // namespace fluidity::contact
