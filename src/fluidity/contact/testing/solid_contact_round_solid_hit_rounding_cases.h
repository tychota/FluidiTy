// Test-only: the helpers that the unit tests and the functional tests of solid_contact_round_solid_hit_rounding share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    kTolerance, kHitDistanceBound, kCellWidth, kLargestCoordinate, RoundScene, DoubleDouble, AddExactly,
//          MultiplyExactly, Renormalize, Add, Negate, Subtract, ToDoubleDouble, Multiply, DoubleDoubleVec3, Dot,
//          ComputeReferenceOwnPoint, ComputeRootMinusLength, ComputeReferenceDistance,
//          ComputeSmallestReferenceDistance, ExpectWithinScale, FindSceneHit, kOppositeCorner, ComputeIsNearForm
// Uses:    the headers it includes
// Used by: src/fluidity/contact/solid_contact.round_solid_hit_rounding.test.cc and
//          tests/contact/round_solid_hit_on_generated_sweeps.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

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
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"

namespace fluidity::contact {

// m. The contact tolerance of a scene whose every coordinate is within 1 m of the origin, at its smallest:
// ComputeContactTolerance gives at least 2^-40 of the largest coordinate. 1 m is 2^40 tol.
constexpr double kTolerance = 0x1p-40;

// m. The bound on how far a sphere's or a cylinder's hit is from the grown surface, along the true distance: counted to
// first order, 14 machine epsilons of the largest own coordinate along the path and 3 of the largest coordinate
// (FindSphereHit, Rounding), about 0.013 tol at this scale.
constexpr double kHitDistanceBound = kTolerance / 64;

// m. The cell width the scale check reads: wide enough, so only the solids' thickness decides.
constexpr double kCellWidth = 0x1p-20;

// The largest coordinate a scene here may reach, m: 2^40 tol.
constexpr double kLargestCoordinate = 1;

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
inline DoubleDouble AddExactly(double a, double b) {
  const double sum = a + b;
  const double b_share = sum - a;
  const double a_share = sum - b_share;
  const double rounding = (a - a_share) + (b - b_share);
  return {.high = sum, .low = rounding};
}

// Returns a * b exactly: the fused multiply-add rounds a * b - product once, and that difference is a double.
inline DoubleDouble MultiplyExactly(double a, double b) {
  const double product = a * b;
  return {.high = product, .low = std::fma(a, b, -product)};
}

// Returns high + low exactly as a normalized pair, when |high| >= |low| (Dekker's fast two-sum).
inline DoubleDouble Renormalize(double high, double low) {
  const double sum = high + low;
  return {.high = sum, .low = low - (sum - high)};
}

// Returns x + y, within 3 units of 2^-106 of the sum (the accurate double-word sum).
inline DoubleDouble Add(DoubleDouble x, DoubleDouble y) {
  const DoubleDouble high_sum = AddExactly(x.high, y.high);
  const DoubleDouble low_sum = AddExactly(x.low, y.low);
  const DoubleDouble first = Renormalize(high_sum.high, high_sum.low + low_sum.high);
  return Renormalize(first.high, first.low + low_sum.low);
}

inline DoubleDouble Negate(DoubleDouble x) {
  return {.high = -x.high, .low = -x.low};
}

inline DoubleDouble Subtract(DoubleDouble x, DoubleDouble y) {
  return Add(x, Negate(y));
}

inline DoubleDouble ToDoubleDouble(double value) {
  return {.high = value, .low = 0};
}

// Returns x * y, within 7 units of 2^-106 of the product.
inline DoubleDouble Multiply(DoubleDouble x, DoubleDouble y) {
  const DoubleDouble high_product = MultiplyExactly(x.high, y.high);
  const double cross_terms = std::fma(x.high, y.low, x.low * y.high);
  return Renormalize(high_product.high, high_product.low + cross_terms);
}

// A vector of double-doubles.
struct DoubleDoubleVec3 {
  DoubleDouble x;
  DoubleDouble y;
  DoubleDouble z;
};

inline DoubleDouble Dot(DoubleDoubleVec3 left, DoubleDoubleVec3 right) {
  return Add(Add(Multiply(left.x, right.x), Multiply(left.y, right.y)), Multiply(left.z, right.z));
}

// The point of a path at elapsed seconds into the sweep, from the solid's origin at that moment, in the solid's own
// axes, from the exact inputs: A^T ((start - origin) + (velocity - linear_velocity) * elapsed - linear_velocity *
// start_elapsed), with A the solid's axes as stored. The solid does not spin.
inline DoubleDoubleVec3 ComputeReferenceOwnPoint(const RoundScene& scene, double elapsed) {
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
inline double ComputeRootMinusLength(DoubleDouble squared_length, double length) {
  const DoubleDouble excess = Subtract(squared_length, MultiplyExactly(length, length));
  return excess.high / (std::sqrt(squared_length.high) + length);
}

// m. The distance from the path's point at elapsed seconds to the solid, a sphere or a cylinder, at that moment, from
// the exact inputs, within 2^-30 tol near the solid: the cylinder's formula is the one of ComputeDistanceToCylinder.
inline double ComputeReferenceDistance(const RoundScene& scene, double elapsed) {
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

// m. The smallest reference distance from the path to the solid between 0 and end_elapsed seconds. The true distance
// to a convex solid is convex along a straight line, so a search that drops a third of the interval each time finds
// the one minimum; 160 steps leave (2/3)^160, below 2^-93 of the interval, under 2^-51 tol of a path of 2^42 tol.
inline double ComputeSmallestReferenceDistance(const RoundScene& scene, double end_elapsed) {
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

// --- The scenes ---

// Fails the calling test unless the scene is within MoveParticle's premise at kTolerance: the solid's largest
// coordinate over the step, its reach included, and the path's two ends within 2^40 tol of the origin; and unless
// CheckContactTolerance accepts it, the solid at least 1024 tol thick.
inline void ExpectWithinScale(const RoundScene& scene) {
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
inline FirstHit FindSceneHit(const RoundScene& scene) {
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

// m. A start near the opposite corner: 2.80 m from kCornerCenter, 2^41.5 tol.
constexpr Vec3 kOppositeCorner = {.x = 0.875, .y = 0.75, .z = 0.9375};

// Returns whether FindSphereHit or ComputeTimeInsideGrownWall takes its near form for this start: C <= R^2, with C
// computed as they compute it, from the own start at the pose of start_elapsed. Only the solid's axes as stored are
// used, so the solid does not spin.
inline bool ComputeIsNearForm(const RoundScene& scene) {
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

}  // namespace fluidity::contact
