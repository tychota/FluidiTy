// Test-only: the helpers that the unit tests and the functional tests of solid_contact_steps_find_first_hit share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    kTolerance, kNoMotion, kTiltedAxes, BuildInputs, Path, FindHit, ToColumn, BuildSphere, BuildCylinder,
//          kAxisZ, RimSearch, SearchRimFromEntry, TurningSearch, SearchTurningSolid, kReferenceAllowance,
//          BuildQuarterTurnLongBox, OutwardPath, BuildOutwardPath
// Uses:    the headers it includes
// Used by: src/fluidity/contact/solid_contact_steps.find_first_hit.test.cc and
//          tests/contact/find_first_hit_on_generated_sweeps.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"

namespace fluidity::contact {

constexpr double kTolerance = kTestContactTolerance;  // m, 2^-30.

constexpr SolidMotion kNoMotion = SolidMotion();  // A solid that neither moves nor turns.

// A turn whose three columns have no zero entry: {0.36, 0.48, -0.8}, {-0.8, 0.6, 0}, {0.48, 0.64, 0.6}. Each column
// has squared length 1 and each pair a dot of 0 in exact arithmetic; in doubles both hold within a few rounding units.
constexpr Mat3 kTiltedAxes = {
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

// The inputs of one sweep against the solids, with the test contact tolerance and no committed plate side.
inline SolidContactInputs BuildInputs(std::span<const Solid> solids, double step_duration) {
  SolidContactInputs inputs;
  inputs.solids = solids.data();
  inputs.solid_count = static_cast<std::uint32_t>(solids.size());
  inputs.step_duration = step_duration;
  inputs.contact_tolerance = kTolerance;
  return inputs;
}

// One straight path: start + velocity * t for 0 <= t <= duration, with the solids posed start_elapsed + t into the
// step.
struct Path {
  Vec3 start{};  // m, world.
  Vec3 velocity{};  // m/s, world.
  double start_elapsed = 0;  // s.
  double duration = 0;  // s.
};

// Runs FindFirstHit on one path and fails the calling test unless it succeeds.
inline FirstHit FindHit(const SolidContactInputs& inputs, Path path) {
  FirstHit first_hit;
  const ContactError error =
      FindFirstHit(inputs, path.start, path.velocity, path.start_elapsed, path.duration, &first_hit);
  EXPECT_EQ(error, ContactError::kNone);
  return first_hit;
}

inline Vec3 ToColumn(Mat3 axes, int column) {
  if (column == 0) {
    return {.x = axes.xx, .y = axes.yx, .z = axes.zx};
  }
  if (column == 1) {
    return {.x = axes.xy, .y = axes.yy, .z = axes.zy};
  }
  return {.x = axes.xz, .y = axes.yz, .z = axes.zz};
}

// --- A sphere ---

// A sphere of radius 0.5 m at center, with id 1.
inline Solid BuildSphere(Vec3 center, SolidMotion motion) {
  return ToSolid(Sphere{.center = center, .radius = 0.5}, motion, 1);
}

// A cylinder of radius 0.5 m and half height 1 m at center along axis, with id 1.
inline Solid BuildCylinder(Vec3 center, Vec3 axis, SolidMotion motion) {
  return ToSolid(Cylinder{.center = center, .axis = axis, .radius = 0.5, .half_height = 1}, motion, 1);
}

constexpr Vec3 kAxisZ = {.x = 0, .y = 0, .z = 1};  // No unit, the world z axis.

// What the rim search returned for one own path, and how many advances it made.
struct RimSearch {
  HitTime hit_time{};
  int advance_count = -1;
};

// Runs the rim search on a path in a cylinder's own axes from the entry the two intervals give, as FindCylinderHit
// does; a path that does not enter the grown cylinder within the duration is free with no advance.
inline RimSearch SearchRimFromEntry(OwnPath own_path, Vec3 half_size, double duration) {
  const TimeInterval inside_wall = ComputeTimeInsideGrownWall(own_path, half_size, kTolerance);
  const TimeInterval between_caps = ComputeTimeBetweenGrownCaps(own_path, half_size, kTolerance);
  RimSearch search;
  search.advance_count = 0;
  if (inside_wall.is_empty || between_caps.is_empty) {
    return search;
  }
  const double entry_elapsed = std::max(inside_wall.entry_elapsed, between_caps.entry_elapsed);  // s
  const bool leaves_wall_first = inside_wall.has_exit && inside_wall.exit_elapsed < entry_elapsed;
  const bool leaves_caps_first = between_caps.has_exit && between_caps.exit_elapsed < entry_elapsed;
  if (leaves_wall_first || leaves_caps_first || !(entry_elapsed <= duration)) {
    return search;
  }
  search.hit_time = FindCylinderRimHit(own_path, half_size, kTolerance, entry_elapsed, duration, &search.advance_count);
  return search;
}

// The answer of the stepping search toward solid 0, called directly, with the advances it made.
struct TurningSearch {
  ContactError error = ContactError::kNone;
  FirstHit first_hit{};
  int advance_count = 0;
};

// Runs FindHitOnTurningSolid toward solid 0 along one path.
inline TurningSearch SearchTurningSolid(const SolidContactInputs& inputs, Path path) {
  const Sweep sweep = {
      .start = path.start,
      .velocity_for_this_step = path.velocity,
      .start_elapsed = path.start_elapsed,
      .duration = path.duration,
  };
  TurningSearch search;
  search.error = FindHitOnTurningSolid(inputs, 0, sweep, &search.first_hit, &search.advance_count);
  return search;
}

// m. Far above the rounding of a distance at unit scale, far below the contact tolerance.
constexpr double kReferenceAllowance = 1e-13;

// A long box of half sizes {0.5, 4, 0.5} with world axes, centred at the origin at the start of the step, moving at
// linear_velocity and turning a quarter turn per second about z: kMaxTurnPerStep in a step of 1 s.
inline Solid BuildQuarterTurnLongBox(Vec3 linear_velocity) {
  return ToSolid(
      Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 4, .z = 0.5}},
      {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0, .y = 0, .z = std::numbers::pi / 2}},
      1
  );
}

// A path that starts r0 m from the origin of a BuildQuarterTurnLongBox moving radially at box_speed m/s, along the
// own direction {sin phi0, -cos phi0, 0} at the pose of start_elapsed, and moves outward along that line at
// outward_speed m/s relative to the box, to the end of the step. The box's arm sweeps onto it while it moves out,
// so the gap closes at nearly the full bound, and the particle is farther from the origin at the end than at the
// start: the far end of r_max is the one that bounds the late advances.
struct OutwardPath {
  double start_elapsed = 0;  // s.
  double start_radius = 0;  // m, r0.
  double outward_speed = 0;  // m/s, relative to the box.
  double start_angle = 0;  // rad, phi0.
  double box_speed = 0;  // m/s, along the same direction; negative is toward the origin's other side.
};

// Returns the box and the path an OutwardPath describes.
inline std::pair<Solid, Path> BuildOutwardPath(OutwardPath outward) {
  const Vec3 own_direction = {.x = std::sin(outward.start_angle), .y = -std::cos(outward.start_angle), .z = 0};
  const Mat3 axes_at_start =
      math::Rotation({.x = 0, .y = 0, .z = std::numbers::pi / 2 * outward.start_elapsed}) * kIdentity;
  const Vec3 direction = axes_at_start * own_direction;
  const Solid solid = BuildQuarterTurnLongBox(outward.box_speed * direction);
  const Vec3 origin_at_start = ComputeSolidPose(solid, outward.start_elapsed).origin;
  const Path path = {
      .start = origin_at_start + outward.start_radius * direction,
      .velocity = solid.motion.linear_velocity + outward.outward_speed * direction,
      .start_elapsed = outward.start_elapsed,
      .duration = 1 - outward.start_elapsed,
  };
  return {solid, path};
}

}  // namespace fluidity::contact
