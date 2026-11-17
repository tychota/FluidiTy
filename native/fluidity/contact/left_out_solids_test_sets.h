// The scene sets on which leaving out a solid a particle cannot come near must change no result bit, and the bits of
// MoveParticle's answer on each scene.
//
// Purpose: give the two programs of the identity test, one built with the rule and one without
//   (FLUIDITY_CONTACT_READ_EVERY_SOLID), the same scenes in the same order and one way to turn an answer into bits.
//
// Idea: each set is a list of scenes from a fixed generator, so both builds see the same inputs. ComputeResultBits
//   runs MoveParticle on every scene of a set and appends, per scene, the error, the failure and every field of the
//   result as integers, the doubles as their 64 bits: two lists are equal exactly when every answer has the same bits.
//   Example: the set kCatalogue holds BuildMoveParticleCases(), BuildObliqueCases(10000) and BuildRoundCases(10000).
//
// Owns:    LeftOutSet, kLeftOutSetCount, ComputeExpectedOutOfRangeDistance, FindStartAtBoundary,
//          BuildLeftOutBoundaryCases, BuildLeftOutSet, ComputeResultBits
// Uses:    the scene catalogue and generator (contact/contact_test_cases.h), MoveParticle (contact/solid_contact.h)
// Used by: contact/move_particle_left_out_test.cc and contact/move_particle_left_out_reference_test.cc
// Read:    docs/method-paper/fluidity-method.typ <proc-move-particle>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_test_cases.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// The sets of the identity test.
enum class LeftOutSet : std::uint8_t {
  kCatalogue,  // BuildMoveParticleCases(), BuildObliqueCases(10000), BuildRoundCases(10000).
  kCrowded,  // 20,000 particles among 16 solids of every kind and motion, near and far.
  kFar,  // 20,000 particles far from 32 solids, a few near them.
  kMixed,  // 20,000 particles with some solids in range and some not, moving, spinning and turning ones, starts near
           // the bound of the rule.
  kBoundary,  // The boundary and behaviour scenes of BuildLeftOutBoundaryCases.
};
inline constexpr int kLeftOutSetCount = 5;

namespace left_out_sets {

inline constexpr SolidMotion kStill = SolidMotion();  // No motion.
inline constexpr Vec3 kOrigin = {.x = 0, .y = 0, .z = 0};  // m.
inline constexpr PerParticleVelocity<Vec3> kAtRest = {.for_next_step = kOrigin, .for_this_step = kOrigin};  // m/s.

// Draws a vector of three NextSigned values times `scale`.
inline Vec3 BuildSignedVector(CaseGenerator& generator, double scale) {
  return {
      .x = scale * generator.NextSigned(),
      .y = scale * generator.NextSigned(),
      .z = scale * generator.NextSigned(),
  };
}

// The 16 solids of the crowded set: a domain box, a floor, a plate spinning about its normal, 12 boxes and moving
// spheres, and a cylinder turning across its axis.
inline std::vector<Solid> BuildCrowdedSolids() {
  std::vector<Solid> solids;
  solids.push_back(ToSolid(
      DomainBox{.low_corner = {.x = -1, .y = -1, .z = -1}, .high_corner = {.x = 2, .y = 2, .z = 2}, .closed = {}},
      1
  ));
  solids.push_back(
      ToSolid(FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}}, kStill, 2)
  );
  solids.push_back(ToSolid(
      Plate{.center = {.x = 0, .y = 0, .z = 1.5}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.01},
      {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 2}},
      3
  ));
  for (int k = 0; k < 12; ++k) {
    const Vec3 center = {.x = 0.1 + (0.07 * k), .y = 0.5 + (0.03 * (k % 3)), .z = 0.2 + (0.05 * (k % 4))};
    const auto solid_id = static_cast<std::uint32_t>(4 + k);
    if (k % 2 == 1) {
      const SolidMotion motion = {.linear_velocity = {.x = 0.1, .y = 0, .z = 0}, .angular_velocity = {}};
      solids.push_back(ToSolid(Sphere{.center = center, .radius = 0.02}, motion, solid_id));
    } else {
      const Box box = {.center = center, .axes = kIdentity, .half_size = {.x = 0.02, .y = 0.03, .z = 0.01}};
      solids.push_back(ToSolid(box, kStill, solid_id));
    }
  }
  solids.push_back(ToSolid(
      Cylinder{
          .center = {.x = 1, .y = 1, .z = 0.5},
          .axis = {.x = 0, .y = 0, .z = 1},
          .radius = 0.05,
          .half_height = 0.1,
      },
      {.linear_velocity = {}, .angular_velocity = {.x = 1, .y = 0, .z = 0}},
      20
  ));
  return solids;
}

// The 32 solids of the far set: a floor at z = -1 and 31 small boxes, spheres and cylinders below z = -0.5, some
// moving, spinning or turning; the particles live above z = 0.
inline std::vector<Solid> BuildFarSolids() {
  std::vector<Solid> solids;
  solids.push_back(ToSolid(
      FlatSolid{.point_on_surface = {.x = 0, .y = 0, .z = -1}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
      kStill,
      1
  ));
  for (int k = 0; k < 31; ++k) {
    const double row = std::floor(static_cast<double>(k) / 8);  // The row of eight solids k is in.
    const Vec3 center = {.x = 0.1 * (k % 8), .y = 0.1 * row, .z = -0.6 - (0.05 * (k % 3))};
    const auto solid_id = static_cast<std::uint32_t>(2 + k);
    const SolidMotion motion = {
        .linear_velocity = {.x = 0, .y = 0.2 * (k % 2), .z = 0},
        .angular_velocity = {.x = 0, .y = 0, .z = 3.0 * (k % 3)},
    };
    if (k % 3 == 0) {
      solids.push_back(ToSolid(Sphere{.center = center, .radius = 0.03}, motion, solid_id));
    } else if (k % 3 == 1) {
      const Box box = {.center = center, .axes = kIdentity, .half_size = {.x = 0.02, .y = 0.03, .z = 0.04}};
      solids.push_back(ToSolid(box, motion, solid_id));
    } else {
      const Cylinder cylinder =
          {.center = center, .axis = {.x = 1, .y = 0, .z = 0}, .radius = 0.03, .half_height = 0.04};
      solids.push_back(ToSolid(cylinder, motion, solid_id));
    }
  }
  return solids;
}

// The solids of the mixed set: a floor moving up, a wall, a box turning about z (a paddle), a sphere moving toward
// the particles, a spinning sphere, a cylinder spinning about its axis, a plate moving sideways and a domain box.
inline std::vector<Solid> BuildMixedSolids() {
  std::vector<Solid> solids;
  solids.push_back(ToSolid(
      DomainBox{.low_corner = {.x = -2, .y = -2, .z = -2}, .high_corner = {.x = 3, .y = 3, .z = 3}, .closed = {}},
      1
  ));
  solids.push_back(ToSolid(
      FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
      {.linear_velocity = {.x = 0, .y = 0, .z = 0.5}, .angular_velocity = {}},
      2
  ));
  solids.push_back(ToSolid(
      FlatSolid{.point_on_surface = {.x = 1.2, .y = 0, .z = 0}, .normal_out_of_solid = {.x = -1, .y = 0, .z = 0}},
      kStill,
      3
  ));
  solids.push_back(ToSolid(
      Box{.center = {.x = 0.5, .y = 0.5, .z = 0.3}, .axes = kIdentity, .half_size = {.x = 0.3, .y = 0.02, .z = 0.05}},
      {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 6}},
      4
  ));
  solids.push_back(ToSolid(
      Sphere{.center = {.x = 0.2, .y = 0.8, .z = 0.4}, .radius = 0.05},
      {.linear_velocity = {.x = 0, .y = -4, .z = 0}, .angular_velocity = {}},
      5
  ));
  solids.push_back(ToSolid(
      Sphere{.center = {.x = 0.8, .y = 0.2, .z = 0.5}, .radius = 0.05},
      {.linear_velocity = {}, .angular_velocity = {.x = 3, .y = 1, .z = 0}},
      6
  ));
  solids.push_back(ToSolid(
      Cylinder{
          .center = {.x = 0.3, .y = 0.3, .z = 0.6},
          .axis = {.x = 0, .y = 1, .z = 0},
          .radius = 0.04,
          .half_height = 0.1,
      },
      {.linear_velocity = {.x = 0.3, .y = 0, .z = 0}, .angular_velocity = {.x = 0, .y = 5, .z = 0}},
      7
  ));
  solids.push_back(ToSolid(
      Plate{.center = {.x = 0, .y = 0, .z = 0.9}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.01},
      {.linear_velocity = {.x = 1, .y = 0, .z = -0.5}, .angular_velocity = {}},
      8
  ));
  return solids;
}

// Inputs for one particle of a generated set: tol 2^-30 m, no start overlap, no committed plate.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): the step and the friction, as BuildCaseInputs orders them.
inline SolidContactInputs BuildSetInputs(Vec3 start, PerParticleVelocity<Vec3> velocity, double step, double friction) {
  SolidContactInputs inputs;
  inputs.start_position = start;
  inputs.velocity = velocity;
  inputs.step_duration = step;
  inputs.friction = friction;
  inputs.contact_tolerance = kTestContactTolerance;
  return inputs;
}

}  // namespace left_out_sets

// Computes, as the test's own statement of the rule, the distance from the start beyond which a solid is out of range:
// the particle's reach R (travel at for_this_step plus two moves off one near solid of 6 tol + 2
// start_overlap_tolerance), plus the solid's travel (|v_lin| T plus min(|w| T, 2) r, r = 0 for a sphere,
// Length(half_size) for a box or a cylinder, and for a flat solid or a plate r_far, the start's distance to the origin
// plus R plus |v_lin| T), plus a window of 4 tol, times 2 for a turning box; for a turning solid, at least the travel
// plus (tol + tol / 64 + (R + |v_lin| T + |w| T r_far) / 127), times 2 for a box. Written in the order
// ComputeOutOfRangeDistance computes it, so the two agree to the bit when both state the same rule: these tests pin
// the formula, not its safety, which the identity test checks.
inline double ComputeExpectedOutOfRangeDistance(const SolidContactInputs& inputs, const Solid& solid) {
  const double step = inputs.step_duration;
  const double tolerance = inputs.contact_tolerance;
  const double rest_gap_move = (6 * tolerance) + (2 * inputs.start_overlap_tolerance);
  const double reach = (math::Length(inputs.velocity.for_this_step) * step) + (2 * rest_gap_move);
  const double linear_speed = math::Length(solid.motion.linear_velocity);
  const double turn_rate = math::Length(solid.motion.angular_velocity);
  const bool has_turn = turn_rate > 0;
  // The boundary scenes' turning solid, a box, turns whenever its shape's turn is not zero (ComputeIsTurningSolid).
  const bool is_turning = has_turn && math::Length(ComputeShapeTurningAngularVelocity(solid)) > 0;
  const double far_distance = has_turn
      ? math::Length(inputs.start_position - solid.pose_at_step_start.origin) + reach + (linear_speed * step)
      : 0;
  const double chord_share = math::Min(turn_rate * step, 2);
  double turn_arm = 0;
  if (has_turn) {
    const bool has_no_end = solid.shape == SolidShape::kFlatSolid || solid.shape == SolidShape::kPlate;
    if (has_no_end) {
      turn_arm = far_distance;
    } else if (solid.shape != SolidShape::kSphere) {
      turn_arm = math::Length(solid.half_size);
    }
  }
  const double travel = reach + (linear_speed * step) + (chord_share * turn_arm);
  const double share = is_turning && solid.shape == SolidShape::kBox ? 2 : 1;
  double out_distance = travel + (share * 4 * tolerance);
  if (is_turning) {
    const double search_travel = reach + (linear_speed * step) + (turn_rate * step * far_distance);
    const double search_gap = tolerance + (tolerance / 64) + (search_travel / 127);
    out_distance = math::Max(out_distance, travel + (share * search_gap));
  }
  return out_distance;
}

// Finds, along the ray origin + s direction, the two starts that straddle the expected out-of-range distance of one
// solid: the last start whose distance at the pose of time 0 is at most the expected distance computed for that very
// start, and the next double of s after it. Bisects on s; the distance grows with s faster than the bound does.
inline std::array<Vec3, 2>
FindStartAtBoundary(const Solid& solid, SolidContactInputs inputs, Vec3 origin, Vec3 direction) {
  const auto excess_at = [&](double s) {
    inputs.start_position = origin + (s * direction);
    const double distance = ComputeDistanceToSolid(solid, ComputeSolidPose(solid, 0), inputs.start_position).distance;
    return distance - ComputeExpectedOutOfRangeDistance(inputs, solid);
  };
  double low = 0;  // excess_at(low) <= 0
  double high = 1;  // excess_at(high) > 0
  while (excess_at(high) <= 0) {
    high *= 2;
  }
  while (std::nextafter(low, high) < high) {
    const double middle = low + ((high - low) / 2);
    if (middle <= low || middle >= high) {
      break;
    }
    (excess_at(middle) <= 0 ? low : high) = middle;
  }
  // low and high are adjacent doubles: the starts are the two points along the ray.
  return {origin + (low * direction), origin + (high * direction)};
}

// A boundary or behaviour scene, named so a test can find it and count its reads.
inline std::vector<MoveParticleCase> BuildLeftOutBoundaryCases() {
  using left_out_sets::BuildSetInputs;
  using left_out_sets::kAtRest;
  using left_out_sets::kOrigin;
  using left_out_sets::kStill;
  std::vector<MoveParticleCase> cases;
  // For each shape, a solid moving sideways at 0.5 m/s and a particle moving along it at 1 m/s for 0.01 s, the start
  // straight above the solid at the expected out-of-range distance (read) and one double farther (left out). One
  // cylinder also spins about its axis, so the turn's term counts.
  const SolidMotion sideways = {.linear_velocity = {.x = 0, .y = 0.5, .z = 0}, .angular_velocity = {}};
  const SolidMotion spinning = {
      .linear_velocity = {.x = 0, .y = 0.5, .z = 0},
      .angular_velocity = {.x = 0, .y = 0, .z = 40},
  };
  const SolidMotion turning = {
      .linear_velocity = {.x = 0, .y = 0.5, .z = 0},
      .angular_velocity = {.x = 0, .y = 0, .z = 20},
  };
  const std::array<Solid, 7> shapes = {
      ToSolid(FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}}, sideways, 1),
      ToSolid(Plate{.center = {}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.01}, sideways, 1),
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.1, .y = 0.2, .z = 0.05}}, sideways, 1),
      ToSolid(Sphere{.center = {}, .radius = 0.1}, sideways, 1),
      ToSolid(
          Cylinder{.center = {}, .axis = {.x = 0, .y = 0, .z = 1}, .radius = 0.1, .half_height = 0.05},
          spinning,
          1
      ),
      ToSolid(
          Cylinder{.center = {}, .axis = {.x = 0, .y = 0, .z = 1}, .radius = 0.1, .half_height = 0.05},
          sideways,
          1
      ),
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.1, .y = 0.2, .z = 0.05}}, turning, 1),
  };
  const std::array<std::string_view, 7> names =
      {"Flat", "Plate", "Box", "Sphere", "SpinningCylinder", "Cylinder", "TurningBox"};
  for (std::size_t shape = 0; shape < shapes.size(); ++shape) {
    SolidContactInputs inputs = BuildSetInputs(
        kOrigin,
        {.for_next_step = {.x = 1, .y = 0, .z = 0}, .for_this_step = {.x = 1, .y = 0, .z = 0}},
        0.01,
        0.3
    );
    const std::array<Vec3, 2> starts = FindStartAtBoundary(shapes.at(shape), inputs, kOrigin, {.x = 0, .y = 0, .z = 1});
    for (int side = 0; side < 2; ++side) {
      inputs.start_position = starts.at(side);
      cases.push_back({.name = names.at(shape), .solids = {shapes.at(shape)}, .inputs = inputs});
    }
  }
  // A sphere arriving: radius 0.05 m, 0.05 m above a particle at rest and coming down at 10 m/s for 0.01 s. Only its
  // own travel brings it to the particle.
  cases.push_back({
      .name = "ArrivingSphere",
      .solids =
          {
              ToSolid(
                  Sphere{.center = {.x = 0, .y = 0, .z = 0.1}, .radius = 0.05},
                  {.linear_velocity = {.x = 0, .y = 0, .z = -10}, .angular_velocity = {}},
                  1
              ),
          },
      .inputs = BuildSetInputs(kOrigin, kAtRest, 0.01, 0.3),
  });
  // A turning paddle: a box 0.6 m long turning about z at 20 rad/s for 0.02 s (0.4 rad), a particle at rest 0.25 rad
  // ahead of it near its tip. Only the turn brings the tip to the particle.
  cases.push_back({
      .name = "TurningPaddle",
      .solids =
          {
              ToSolid(
                  Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.3, .y = 0.01, .z = 0.05}},
                  {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 20}},
                  1
              ),
          },
      .inputs =
          BuildSetInputs({.x = 0.28 * 0.9689124217106447, .y = 0.28 * 0.24740395925452294, .z = 0}, kAtRest, 0.02, 0.3),
  });
  // A redirect: a particle at rest 0.5 tol above the rest gap of a wall at x = 0 that moves at 20 m/s along +x for
  // 0.01 s, and a sphere of radius 0.05 m 0.1 m away along +x: out of range for the particle at rest, reached once
  // the wall pushes it.
  cases.push_back({
      .name = "PushedIntoFarSphere",
      .solids =
          {
              ToSolid(
                  FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 1, .y = 0, .z = 0}},
                  {.linear_velocity = {.x = 20, .y = 0, .z = 0}, .angular_velocity = {}},
                  1
              ),
              ToSolid(Sphere{.center = {.x = 0.15, .y = 0, .z = 0}, .radius = 0.05}, kStill, 2),
          },
      .inputs = BuildSetInputs({.x = 1.5 * kTestContactTolerance, .y = 0, .z = 0}, kAtRest, 0.01, 0.3),
  });
  // A flat solid turning about an axis in its own plane, the y axis through the origin, at 1 rad/s for 0.01 s, and a
  // particle at rest 0.05 m above it 10 m from the axis: the plane rises 0.1 m under the particle there. Only the
  // particle's arm from the origin, in the plane's bound, says so.
  cases.push_back({
      .name = "TurningFloorFarFromAxis",
      .solids =
          {
              ToSolid(
                  FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
                  {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = -1, .z = 0}},
                  1
              ),
          },
      .inputs = BuildSetInputs({.x = 10, .y = 0, .z = 0.05}, kAtRest, 0.01, 0.3),
  });
  // A cylinder 3 m from a particle at rest, spinning at 1e5 rad/s about an axis tilted by 1e-5 rad from its own: it
  // turns (plan decision P17), and the stepping search's bound counts the whole spin, so the search runs out of
  // advances in 2 ms: kSlowApproachToTurningSolid, which only the rule's step 4 keeps.
  cases.push_back({
      .name = "FastSpinningTiltedCylinder",
      .solids =
          {
              ToSolid(
                  Cylinder{
                      .center = {.x = 3, .y = 0, .z = 0},
                      .axis = {.x = 0, .y = 0, .z = 1},
                      .radius = 0.01,
                      .half_height = 0.01,
                  },
                  {.linear_velocity = {}, .angular_velocity = {.x = 1, .y = 0, .z = 1e5}},
                  1
              ),
          },
      .inputs = BuildSetInputs(kOrigin, kAtRest, 0.002, 0.3),
  });
  // A long move at the start: a V of two flat solids through the origin, normals {sin a, +-cos a, 0} with sin a = 2^-6,
  // and the particle on the bisector at x = 32 tol, 0.5 tol from both faces. The move off both faces takes it 32 tol to
  // the rest point at x = 64 tol; a third flat solid, normal {-1, 0, 0}, has its face 1.03125 tol beyond that point,
  // 33 tol from the start, so the start leaves it out. The particle approaches it at 1e-3 tol per step.
  {
    constexpr double kSinOfHalfAngle = 0x1p-6;
    const double cos_of_half_angle = std::sqrt(1 - (kSinOfHalfAngle * kSinOfHalfAngle));
    const double rest_x = kTestContactTolerance / kSinOfHalfAngle;  // m, 64 tol
    constexpr double kStep = 0.01;  // s
    const Vec3 approach = {.x = 1e-3 * kTestContactTolerance / kStep, .y = 0, .z = 0};  // m/s
    const Vec3 upper_normal = {.x = kSinOfHalfAngle, .y = cos_of_half_angle, .z = 0};
    const Vec3 lower_normal = {.x = kSinOfHalfAngle, .y = -cos_of_half_angle, .z = 0};
    const Vec3 wall_point = {.x = rest_x + (1.03125 * kTestContactTolerance), .y = 0, .z = 0};
    cases.push_back({
        .name = "LongStartMoveIntoLeftOutWall",
        .solids =
            {
                ToSolid(FlatSolid{.point_on_surface = {}, .normal_out_of_solid = upper_normal}, kStill, 1),
                ToSolid(FlatSolid{.point_on_surface = {}, .normal_out_of_solid = lower_normal}, kStill, 2),
                ToSolid(
                    FlatSolid{.point_on_surface = wall_point, .normal_out_of_solid = {.x = -1, .y = 0, .z = 0}},
                    kStill,
                    3
                ),
            },
        .inputs = BuildSetInputs(
            {.x = rest_x / 2, .y = 0, .z = 0},
            {.for_next_step = approach, .for_this_step = approach},
            kStep,
            0.3
        ),
    });
  }
  // A domain box far from the particle: never left out.
  cases.push_back({
      .name = "FarDomainBox",
      .solids =
          {
              ToSolid(
                  DomainBox{
                      .low_corner = {.x = -10, .y = -10, .z = -10},
                      .high_corner = {.x = 10, .y = 10, .z = 10},
                      .closed = ClosedSides(),
                  },
                  1
              ),
          },
      .inputs = BuildSetInputs(kOrigin, kAtRest, 0.01, 0.3),
  });
  return cases;
}

// Builds the scenes of one set, the same on every build and platform.
//
// Method: kCatalogue as its Owns line says. The generated sets draw, per particle and in this order from one
//   CaseGenerator: the start, three Next times the set's box; the velocity, three NextSigned times a speed scale drawn
//   from {0, 0.3, 3, 30} m/s by Next; for_next_step = 0.9 for_this_step; friction Next; the step, 1 ms or 10 ms by
//   Next. In the mixed set one particle in four starts on a line toward a solid at a distance near the rule's bound.
// Example: kFar has 20,000 scenes over the same 32 solids.
inline std::vector<MoveParticleCase> BuildLeftOutSet(LeftOutSet set) {
  using left_out_sets::BuildSetInputs;
  using left_out_sets::BuildSignedVector;
  std::vector<MoveParticleCase> cases;
  if (set == LeftOutSet::kBoundary) {
    return BuildLeftOutBoundaryCases();
  }
  if (set == LeftOutSet::kCatalogue) {
    cases = BuildMoveParticleCases();
    for (MoveParticleCase& test_case : BuildObliqueCases(10000)) {
      cases.push_back(std::move(test_case));
    }
    for (MoveParticleCase& test_case : BuildRoundCases(10000)) {
      cases.push_back(std::move(test_case));
    }
    return cases;
  }
  std::vector<Solid> solids = left_out_sets::BuildMixedSolids();
  if (set == LeftOutSet::kCrowded) {
    solids = left_out_sets::BuildCrowdedSolids();
  } else if (set == LeftOutSet::kFar) {
    solids = left_out_sets::BuildFarSolids();
  }
  const Vec3 box = set == LeftOutSet::kFar ? Vec3{.x = 1, .y = 1, .z = 1} : Vec3{.x = 1.2, .y = 1.2, .z = 0.8};
  CaseGenerator generator;
  constexpr int kParticleCount = 20000;
  cases.reserve(kParticleCount);
  for (int particle = 0; particle < kParticleCount; ++particle) {
    Vec3 start = {.x = box.x * generator.Next(), .y = box.y * generator.Next(), .z = box.z * generator.Next()};
    const double speed_scale = std::array<double, 4>{0, 0.3, 3, 30}[static_cast<int>(4 * generator.Next())];
    const Vec3 velocity = BuildSignedVector(generator, speed_scale);
    const double friction = generator.Next();
    const double step = generator.Next() < 0.5 ? 1e-3 : 1e-2;
    if (set == LeftOutSet::kMixed && particle % 4 == 0) {
      // Near the bound of the rule for solid 5, the sphere moving toward the particles at 4 m/s: a start straight
      // above its top, at a distance its travel and the particle's could cover, or just not.
      const double share = 0.5 + generator.Next();
      start = {.x = 0.2, .y = 0.8, .z = 0.45 + (share * (4 + speed_scale) * step)};
    }
    cases.push_back({
        .name = "LeftOutSet",
        .solids = solids,
        .inputs = BuildSetInputs(start, {.for_next_step = 0.9 * velocity, .for_this_step = velocity}, step, friction),
    });
  }
  return cases;
}

// Computes the bits of MoveParticle's answer on every scene of a set: per scene the error, the failure's close call,
// solid_id and magnitude, and on success every field of the result, the doubles as their 64 bits.
inline std::vector<std::uint64_t> ComputeResultBits(LeftOutSet set) {
  std::vector<std::uint64_t> bits;
  const auto append_vector = [&bits](Vec3 vector) {
    bits.push_back(std::bit_cast<std::uint64_t>(vector.x));
    bits.push_back(std::bit_cast<std::uint64_t>(vector.y));
    bits.push_back(std::bit_cast<std::uint64_t>(vector.z));
  };
  for (const MoveParticleCase& test_case : BuildLeftOutSet(set)) {
    SolidContactResult result;
    ContactFailure failure;
    const ContactError error = MoveParticle(ToSolidContactInputs(test_case), &result, &failure);
    bits.push_back(static_cast<std::uint64_t>(error));
    bits.push_back(static_cast<std::uint64_t>(failure.close_call));
    bits.push_back(failure.solid_id);
    bits.push_back(std::bit_cast<std::uint64_t>(failure.magnitude));
    if (error != ContactError::kNone) {
      continue;
    }
    append_vector(result.position);
    append_vector(result.velocity.for_next_step);
    append_vector(result.velocity.for_this_step);
    bits.push_back(static_cast<std::uint64_t>(result.kind));
    bits.push_back(std::bit_cast<std::uint64_t>(result.hit_fraction));
    append_vector(result.normal_out_of_solid);
    bits.push_back(std::bit_cast<std::uint64_t>(result.approach_speed.for_next_step));
    bits.push_back(std::bit_cast<std::uint64_t>(result.approach_speed.for_this_step));
    bits.push_back(result.solid_index);
    bits.push_back(static_cast<std::uint64_t>(result.face));
    bits.push_back(static_cast<std::uint64_t>(result.side_at_hit));
    bits.push_back(result.hit_count);
    bits.push_back(result.reaction.solid_index);
    append_vector(result.reaction.hit_position);
    append_vector(result.reaction.velocity_for_next_step_change);
  }
  return bits;
}

}  // namespace fluidity::contact
