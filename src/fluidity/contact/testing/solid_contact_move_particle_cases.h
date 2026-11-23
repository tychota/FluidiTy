// Test-only: the helpers that the unit tests and the functional tests of solid_contact_move_particle share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    kTolerance, kNoMotion, kUnitX, kUnitZ, MoveParticleCall, BuildUntouchedResult, ComputeMove, ComputeMove,
//          MoveBeforeSettle, ApplyStepsBeforeSettle, ComputeMoveBeforeSettle, ComputePositionBeforeSettle,
//          BuildOrderedScene, ToWedgeSideNormal, BuildRestPointInputs, BuildMovingFlatSolid, ToPermutedSolids,
//          BuildUnitVector, BuildUnitVector, BuildCorrectionTargets, SharpSurfaceResult, CheckBothVelocitiesNear
// Uses:    the headers it includes
// Used by: src/fluidity/contact/solid_contact.move_particle.test.cc,
//          tests/contact/move_particle_on_generated_paths.test.cc,
//          tests/contact/move_particle_slides_over_many_steps.test.cc and
//          tests/contact/smallest_correction_on_generated_cases.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"

namespace fluidity::contact {

constexpr double kTolerance = kTestContactTolerance;  // m, 2^-30.

constexpr SolidMotion kNoMotion = move_particle_cases::kNoMotion;  // A solid that neither moves nor turns.

constexpr Vec3 kUnitX = move_particle_cases::kUnitX;

constexpr Vec3 kUnitZ = move_particle_cases::kUnitZ;

// What one MoveParticle call returned: the error, the result and the failure.
struct MoveParticleCall {
  ContactError error = ContactError::kNone;
  SolidContactResult result{};
  ContactFailure failure{};
};

// A result whose every field differs from what MoveParticle writes, to show that a failed call leaves it unchanged.
inline SolidContactResult BuildUntouchedResult() {
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
inline MoveParticleCall ComputeMove(const MoveParticleCase& test_case) {
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
inline MoveParticleCall ComputeMove(std::string_view case_name) {
  return ComputeMove(FindMoveParticleCase(case_name));
}

// Where the steps of MoveParticle leave a particle before the settle, and the faces of the hits along the path.
struct MoveBeforeSettle {
  Vec3 position{};  // m, world.
  std::vector<SolidFace> hit_faces;  // The start's response, when it has one, then each hit's face, in order.
};

// Applies the steps of MoveParticle to one move up to the end of the last sweep, without the settle and the end check,
// adds the face of each hit to *hit_faces, and returns the error that stopped it, kNone when it reached the settle.
inline ContactError
ApplyStepsBeforeSettle(const SolidContactInputs& inputs, ParticleMove* move, std::vector<SolidFace>* hit_faces) {
  move->result.position = inputs.start_position;
  move->result.velocity = inputs.velocity;
  ContactFailure failure;
  ContactError error = ApplyStartRules(inputs, move, &failure);
  if (move->result.hit_count > 0) {
    hit_faces->push_back(move->result.face);
  }
  for (int sweep_index = 0; sweep_index <= kMaxHitsPerStep && error == ContactError::kNone; ++sweep_index) {
    const double duration = inputs.step_duration - move->elapsed;  // s
    if (duration <= 0) {
      break;
    }
    const Sweep sweep = {
        .start = move->result.position,
        .velocity_for_this_step = move->result.velocity.for_this_step,
        .start_elapsed = move->elapsed,
        .duration = duration,
    };
    const TouchedRule touched = {
        .is_touched = move->has_response,
        .approach_allowance = move->this_step_approach_allowance,
    };
    FirstHit first_hit;
    error = FindFirstHitOfSweep(inputs, sweep, touched, &first_hit, nullptr);
    if (error != ContactError::kNone) {
      break;
    }
    if (first_hit.solid_index == kNoSolid) {
      move->result.position = move->result.position + move->result.velocity.for_this_step * duration;
      break;
    }
    hit_faces->push_back(first_hit.at_hit.face);
    // As MoveParticle: a hit on a turning solid ends the sweeps with the carry in its axes.
    if (ComputeIsTurningSolid(inputs, first_hit.solid_index)) {
      error = ApplyHitOnTurningSolid(inputs, first_hit, move, &failure);
      break;
    }
    error = ApplyHit(inputs, first_hit, move, &failure);
  }
  return error;
}

// Runs the steps of MoveParticle on one scene up to the end of the last sweep, without the settle and the end check,
// and returns the position there with the faces of the hits; a failure fails the calling test.
inline MoveBeforeSettle ComputeMoveBeforeSettle(const MoveParticleCase& scene) {
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  MoveBeforeSettle before_settle;
  ParticleMove move;
  EXPECT_EQ(ApplyStepsBeforeSettle(inputs, &move, &before_settle.hit_faces), ContactError::kNone);
  before_settle.position = move.result.position;
  return before_settle;
}

// The position ComputeMoveBeforeSettle returns.
inline Vec3 ComputePositionBeforeSettle(const MoveParticleCase& scene) {
  return ComputeMoveBeforeSettle(scene).position;
}

// --- The joint response (method paper <sec-move-particle>) ---

// Builds a scene from its solids, in that array order, with the solid_ids renumbered 1, 2, ... so they ascend.
inline MoveParticleCase BuildOrderedScene(std::vector<Solid> solids, const SolidContactInputs& inputs) {
  for (std::size_t solid_index = 0; solid_index < solids.size(); ++solid_index) {
    solids[solid_index].solid_id = static_cast<std::uint32_t>(solid_index + 1);
  }
  return {.name = "", .solids = std::move(solids), .inputs = inputs};
}

// The normal of BuildWedgeSide for a wedge of that angle with the floor, the same bits.
inline Vec3 ToWedgeSideNormal(double wedge_angle) {
  const double normal_angle = std::numbers::pi - wedge_angle;  // rad
  return {.x = std::sin(normal_angle), .y = 0, .z = std::cos(normal_angle)};
}

// The inputs of a particle at the rest point {tol / tan(a / 2), 0, tol} of a wedge of angle a with the floor, both
// velocities velocity, T 0.01 s, overlap 2 tol, with that friction.
inline SolidContactInputs BuildRestPointInputs(double wedge_angle, Vec3 velocity, double friction) {
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
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline Solid
BuildMovingFlatSolid(Vec3 normal_out_of_solid, Vec3 linear_velocity, double friction_scale, std::uint32_t id) {
  Solid solid = move_particle_cases::BuildFlatSolid({}, normal_out_of_solid, {.linear_velocity = linear_velocity}, id);
  solid.friction_scale = friction_scale;
  return solid;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// The solids in the given order of their indexes.
inline std::vector<Solid> ToPermutedSolids(const std::vector<Solid>& solids, const std::vector<std::size_t>& order) {
  std::vector<Solid> permuted;
  permuted.reserve(order.size());
  for (const std::size_t index : order) {
    permuted.push_back(solids[index]);
  }
  return permuted;
}

// Draws a unit vector from the generator: three signed coordinates, again until the length is in [0.1, 1].
inline Vec3 BuildUnitVector(CaseGenerator* generator) {
  Vec3 drawn{};
  do {
    drawn = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
  } while (math::Length(drawn) < 0.1 || math::Length(drawn) > 1);
  return math::Normalized(drawn);
}

// Draws a unit vector: three NextSigned, drawn again while shorter than 0.1.
inline Vec3 BuildUnitVector(CaseGenerator& generator) {
  Vec3 direction{};
  while (math::Length(direction) < 0.1) {
    direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  return math::Normalized(direction);
}

// A CorrectionTargets of the given normals and targets.
inline CorrectionTargets
BuildCorrectionTargets(std::initializer_list<Vec3> normals, std::initializer_list<double> targets) {
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

// What one case of the sharp-surface set gave: its refinements and the largest approach it left, in rounding units of
// the summed speeds (0 when no change was needed).
struct SharpSurfaceResult {
  int refinement_count = 0;
  double largest_share_left = 0;
};

// Fails the calling test unless both velocities are within tolerance of expected on each axis.
inline void CheckBothVelocitiesNear(const SolidContactResult& result, Vec3 expected, double tolerance) {
  CheckVec3IsNear(result.velocity.for_next_step, expected, tolerance);
  CheckVec3IsNear(result.velocity.for_this_step, expected, tolerance);
}

}  // namespace fluidity::contact
