// The flat-solid and plate cases of the old contact code, with what the old code returned for each.
//
// Purpose: give the comparison of MoveParticle with the code it replaces one list of cases, each with its inputs and
//   the old code's outcome written out as data, so the comparison needs no code from the old branch.
//
// Idea: the old code (branch main, ResolveLiquidWallContact3D in native/src/liquid_wall_contact_3d.cc, whose sweep is
//   ResolveContact in native/include/fluidity/base/liquid_wall_contact_math_3d.h) moved one particle against one
//   solid that moved without turning: the old code's half space, the solid n . x <= c, which this code calls a flat
//   solid, or a two-faced solid |n . x - c| <= h, which this code calls a plate. Its tests
//   (native/tests/liquid_wall_contact_3d_cases.h on main) call it with the cases below. Each case keeps the old inputs
//   as written there, in the old Y-up axes, and the old outcome: an error class, or the end position, both velocities,
//   whether the path met the solid and, when it did, the share of the step at which it met it and the normal there.
//   ToZUp turns a vector into this code's Z-up axes, (x, y, z) -> (x, -z, y): a quarter turn about x, so lengths, dot
//   products and the plane coordinate c stay the same. The old code stopped a particle on the surface itself; this code
//   stops it one contact tolerance outside, so an end position agrees within a few tolerances, and a hit's velocities
//   agree within rounding. Example: the case of line 237 is the flat solid x >= 2 (normal {-1, 0, 0}, c = -2) and a
//   particle at {0, 1, 0} moving at {4, 2, 0} m/s for 1 s: the old code ends it at {2, 3, 0} with
//   velocity.for_this_step {0, 2, 0}. How the old outcomes were found: by reading ResolveContact and
//   ApplyOneWayResponse, case by case, and checking each against the expectation of the old test, which holds every one
//   of them within 2e-14 (its kTolerance); a velocity the old test does not check is the input, which ResolveContact
//   returns unchanged when the path misses or the velocity does not approach. The old code was not built or run for
//   this list.
//
// Owns:    MainSolidKind, MainStatusCode, MainAgreement, MainComparisonCase, ToZUp, BuildMainComparisonCases
// Uses:    Vec3 (base/vec3.h), CPU-only guard (base/cpu_only.h)
// Used by: contact/move_particles.scene.test.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "fluidity/base/vec3.h"

namespace fluidity::contact {

// The solid of one old case.
enum class MainSolidKind : std::uint8_t {
  kNone,  // No solid: the particle moves freely.
  kFlatSolid,  // The flat solid normal . x <= plane_coordinate.
  kPlate,  // The two-faced solid |normal . x - plane_coordinate| <= half_thickness.
};

// The status code the old code returned for one case.
enum class MainStatusCode : std::uint8_t {
  kOk,  // A result: the end position and both velocities.
  kInvalidArgument,  // A value out of range; this code's kBadInput.
  kFailedPrecondition,  // A start too deep, or on neither side of a plate; this code's kStartsInsideSolid.
};

// Whether this code must give the old outcome for a case.
enum class MainAgreement : std::uint8_t {
  kAgrees,  // The same outcome: within the comparison's bounds, or the matching error.
  kDiffers,  // A different outcome on purpose; why_different says why, and a test of its own pins this code's.
};

// One call of the old code, in the old Y-up axes, and what it returned.
struct MainComparisonCase {
  int source_line = 0;  // Line of the call in native/tests/liquid_wall_contact_3d_cases.h on main.
  std::string_view old_test;  // The old test that makes the call.
  MainSolidKind solid_kind = MainSolidKind::kNone;
  Vec3 normal{};  // No unit, unit vector, Y-up. Out of a flat solid; along a plate's axis.
  double plane_coordinate = 0;  // m. Of the surface of a flat solid, or of the middle of a plate, along normal.
  double half_thickness = 0;  // m. A plate's half thickness; 0 for a flat solid.
  Vec3 solid_velocity{};  // m/s, Y-up. The solid moves without turning.
  Vec3 start_position{};  // m, Y-up.
  Vec3 velocity_for_next_step{};  // m/s, Y-up. The velocity the old code kept on the particle.
  Vec3 velocity_for_this_step{};  // m/s, Y-up. The velocity that moved the particle in the old code.
  double step_duration = 0;  // s.
  double friction = 0;  // No unit.
  double start_overlap_tolerance = 0;  // m. The deepest start the old code moved out.
  MainStatusCode old_status = MainStatusCode::kOk;
  bool old_has_hit = false;  // The old result had a hit time. Only for kOk.
  double old_hit_fraction = 0;  // No unit, 0 to 1. The old hit time over the step. Only with old_has_hit.
  Vec3 old_normal_out_of_solid{};  // No unit, Y-up. The old normal of the face hit. Only with old_has_hit.
  Vec3 old_position{};  // m, Y-up. The old end position. Only for kOk.
  Vec3 old_velocity_for_next_step{};  // m/s, Y-up. Only for kOk.
  Vec3 old_velocity_for_this_step{};  // m/s, Y-up. Only for kOk.
  MainAgreement agreement = MainAgreement::kAgrees;
  const char* why_different = nullptr;  // Null when the outcomes agree.
};

// Converts a vector of the old Y-up axes into this code's Z-up axes: (x, y, z) -> (x, -z, y).
//
// Example: the old up {0, 1, 0} becomes {0, 0, 1}; the old {0, 0, 1} becomes {0, -1, 0}.
constexpr Vec3 ToZUp(Vec3 y_up) {
  return {.x = y_up.x, .y = -y_up.z, .z = y_up.y};
}

// Builds the old cases of native/tests/liquid_wall_contact_3d_cases.h on main, lines 209 to 398 and 441 to 487: every
// call against a flat solid or a two-faced solid, or with no solid, outside the open box of the old code (lines 38 to
// 144), its table read from a file (145 to 208) and its tests of rounding enclosures (399 to 440). In the order of the
// file.
//
// Example: 24 cases, 12 with a result (11 agree, 1 differs) and 12 with an error (11 agree, 1 differs).
inline std::vector<MainComparisonCase> BuildMainComparisonCases() {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  constexpr double kNotANumber = std::numeric_limits<double>::quiet_NaN();
  constexpr Vec3 kUnitX = {.x = 1, .y = 0, .z = 0};
  // The valid call that each case of RejectsInvalidContactInputs (lines 368 to 397) changes in one value.
  const MainComparisonCase valid_call = {
      .source_line = 368,
      .old_test = "RejectsInvalidContactInputs",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = kUnitX,
      .start_position = kUnitX,
      .step_duration = 1,
      .old_status = MainStatusCode::kInvalidArgument,
  };
  std::vector<MainComparisonCase> cases;
  cases.push_back({
      .source_line = 211,
      .old_test = "ObliqueReversedSlabPreservesIncidentSide",
      .solid_kind = MainSolidKind::kPlate,
      .normal = {.x = -0.6, .y = -0.8, .z = 0},
      .half_thickness = 1,
      .start_position = {.x = 3, .y = 4, .z = 2},
      .velocity_for_next_step = {.x = -6, .y = -8, .z = 3},
      .velocity_for_this_step = {.x = -6, .y = -8, .z = 4},
      .step_duration = 1,
      .old_has_hit = true,
      .old_hit_fraction = 0.4,
      .old_normal_out_of_solid = {.x = 0.6, .y = 0.8, .z = 0},
      .old_position = {.x = 0.6, .y = 0.8, .z = 6},
      .old_velocity_for_next_step = {.x = 0, .y = 0, .z = 3},
      .old_velocity_for_this_step = {.x = 0, .y = 0, .z = 4},
  });
  cases.push_back({
      .source_line = 237,
      .old_test = "OneSidedPlaneOnlyBlocksMotionIntoSolid",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = {.x = -1, .y = 0, .z = 0},
      .plane_coordinate = -2,
      .start_position = {.x = 0, .y = 1, .z = 0},
      .velocity_for_next_step = {.x = -3, .y = 2, .z = 0},
      .velocity_for_this_step = {.x = 4, .y = 2, .z = 0},
      .step_duration = 1,
      .old_has_hit = true,
      .old_hit_fraction = 0.5,
      .old_normal_out_of_solid = {.x = -1, .y = 0, .z = 0},
      .old_position = {.x = 2, .y = 3, .z = 0},
      .old_velocity_for_next_step = {.x = -3, .y = 2, .z = 0},
      .old_velocity_for_this_step = {.x = 0, .y = 2, .z = 0},
  });
  cases.push_back({
      .source_line = 249,
      .old_test = "OneSidedPlaneOnlyBlocksMotionIntoSolid",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = {.x = -1, .y = 0, .z = 0},
      .plane_coordinate = -2,
      .start_position = {.x = 2, .y = 1, .z = 0},
      .velocity_for_next_step = {.x = -3, .y = 2, .z = 0},
      .velocity_for_this_step = {.x = -4, .y = 2, .z = 0},
      .step_duration = 1,
      .old_position = {.x = -2, .y = 3, .z = 0},
      .old_velocity_for_next_step = {.x = -3, .y = 2, .z = 0},
      .old_velocity_for_this_step = {.x = -4, .y = 2, .z = 0},
  });
  cases.push_back({
      .source_line = 256,
      .old_test = "TangentialMotionDoesNotTriggerFriction",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = kUnitX,
      .plane_coordinate = 2,
      .solid_velocity = {.x = 1, .y = 1, .z = 0},
      .start_position = {.x = 2, .y = 1, .z = 0},
      .velocity_for_next_step = {.x = 1, .y = 5, .z = 0},
      .velocity_for_this_step = {.x = 1, .y = 3, .z = 0},
      .step_duration = 2,
      .friction = 10,
      .old_position = {.x = 4, .y = 7, .z = 0},
      .old_velocity_for_next_step = {.x = 1, .y = 5, .z = 0},
      .old_velocity_for_this_step = {.x = 1, .y = 3, .z = 0},
  });
  cases.push_back({
      .source_line = 272,
      .old_test = "TouchingWallConstrainsStoredInwardVelocityDuringTangentAdvection",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = kUnitX,
      .velocity_for_next_step = {.x = -2, .y = 3, .z = 0},
      .velocity_for_this_step = {.x = 0, .y = 4, .z = 0},
      .step_duration = 1,
      .friction = 1,
      .old_has_hit = true,
      .old_hit_fraction = 0,
      .old_normal_out_of_solid = kUnitX,
      .old_position = {.x = 0, .y = 4, .z = 0},
      .old_velocity_for_next_step = {.x = 0, .y = 1, .z = 0},
      .old_velocity_for_this_step = {.x = 0, .y = 4, .z = 0},
  });
  cases.push_back({
      .source_line = 290,
      .old_test = "FrictionUsesEachIncomingRelativeNormalSpeed",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = kUnitX,
      .solid_velocity = {.x = 1, .y = 2, .z = 3},
      .start_position = {.x = 1, .y = 2, .z = 3},
      .velocity_for_next_step = {.x = -1, .y = 1, .z = 3},
      .velocity_for_this_step = {.x = -3, .y = 5, .z = 3},
      .step_duration = 1,
      .friction = 0.5,
      .old_has_hit = true,
      .old_hit_fraction = 0.25,
      .old_normal_out_of_solid = kUnitX,
      .old_position = {.x = 1, .y = 5.5, .z = 6},
      .old_velocity_for_next_step = {.x = 1, .y = 2, .z = 3},
      .old_velocity_for_this_step = {.x = 1, .y = 3, .z = 3},
  });
  cases.push_back({
      .source_line = 309,
      .old_test = "RecoversTolerablePenetrationOnNearestSide",
      .solid_kind = MainSolidKind::kPlate,
      .normal = kUnitX,
      .half_thickness = 1,
      .start_position = {.x = -0.875, .y = 2, .z = 0},
      .velocity_for_next_step = {.x = 1, .y = 0, .z = 0},
      .velocity_for_this_step = {.x = -1, .y = 3, .z = 0},
      .step_duration = 1,
      .start_overlap_tolerance = 0.125,
      .old_has_hit = true,
      .old_hit_fraction = 0,
      .old_normal_out_of_solid = {.x = -1, .y = 0, .z = 0},
      .old_position = {.x = -2, .y = 5, .z = 0},
      .old_velocity_for_next_step = {.x = 0, .y = 0, .z = 0},
      .old_velocity_for_this_step = {.x = -1, .y = 3, .z = 0},
  });
  cases.push_back({
      .source_line = 334,
      .old_test = "RejectsDeepAndAmbiguousPenetration",
      .solid_kind = MainSolidKind::kPlate,
      .normal = kUnitX,
      .half_thickness = 1,
      .start_position = {.x = 0.75, .y = 0, .z = 0},
      .step_duration = 1,
      .start_overlap_tolerance = 0.125,
      .old_status = MainStatusCode::kFailedPrecondition,
  });
  cases.push_back({
      .source_line = 337,
      .old_test = "RejectsDeepAndAmbiguousPenetration",
      .solid_kind = MainSolidKind::kPlate,
      .normal = kUnitX,
      .half_thickness = 1,
      .start_position = {.x = 0, .y = 0, .z = 0},
      .step_duration = 1,
      .start_overlap_tolerance = 2,
      .old_status = MainStatusCode::kFailedPrecondition,
      .agreement = MainAgreement::kDiffers,
      .why_different = "a start on the middle plane of a plate, 1 m deep within an overlap tolerance of 2 m: the old "
                       "code could not choose a side; this code counts the middle plane as the side its normal "
                       "points to and moves the start out through that face (kHighZ), as for any start inside a solid",
  });
  cases.push_back({
      .source_line = 339,
      .old_test = "RejectsDeepAndAmbiguousPenetration",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = kUnitX,
      .plane_coordinate = 4,
      .step_duration = 1,
      .start_overlap_tolerance = 2,
      .old_status = MainStatusCode::kFailedPrecondition,
  });
  cases.push_back({
      .source_line = 350,
      .old_test = "DetectsEndpointContactAndFreeAdvection",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = kUnitX,
      .start_position = {.x = 1, .y = 2, .z = 3},
      .velocity_for_next_step = {.x = 3, .y = 2, .z = 1},
      .velocity_for_this_step = {.x = -2, .y = 0, .z = 2},
      .step_duration = 0.5,
      .old_has_hit = true,
      .old_hit_fraction = 1,
      .old_normal_out_of_solid = kUnitX,
      .old_position = {.x = 0, .y = 2, .z = 4},
      .old_velocity_for_next_step = {.x = 3, .y = 2, .z = 1},
      .old_velocity_for_this_step = {.x = 0, .y = 0, .z = 2},
  });
  cases.push_back({
      .source_line = 358,
      .old_test = "DetectsEndpointContactAndFreeAdvection",
      .solid_kind = MainSolidKind::kNone,
      .start_position = {.x = 1, .y = 2, .z = 3},
      .velocity_for_next_step = {.x = 3, .y = 2, .z = 1},
      .velocity_for_this_step = {.x = -2, .y = 0, .z = 2},
      .step_duration = 0.5,
      .old_position = {.x = 0, .y = 2, .z = 4},
      .old_velocity_for_next_step = {.x = 3, .y = 2, .z = 1},
      .old_velocity_for_this_step = {.x = -2, .y = 0, .z = 2},
  });
  MainComparisonCase invalid_call = valid_call;
  invalid_call.source_line = 375;
  invalid_call.start_position.x = kInfinity;
  cases.push_back(invalid_call);
  invalid_call = valid_call;
  invalid_call.source_line = 378;
  invalid_call.velocity_for_next_step.y = kNotANumber;
  cases.push_back(invalid_call);
  invalid_call = valid_call;
  invalid_call.source_line = 381;
  invalid_call.step_duration = 0;
  cases.push_back(invalid_call);
  invalid_call = valid_call;
  invalid_call.source_line = 384;
  invalid_call.friction = -1;
  cases.push_back(invalid_call);
  invalid_call = valid_call;
  invalid_call.source_line = 387;
  invalid_call.start_overlap_tolerance = kInfinity;
  cases.push_back(invalid_call);
  invalid_call = valid_call;
  invalid_call.source_line = 390;
  invalid_call.normal = {.x = 2, .y = 0, .z = 0};
  cases.push_back(invalid_call);
  // Lines 391 to 396 keep the call of line 390 and replace its solid by a plate.
  invalid_call = valid_call;
  invalid_call.solid_kind = MainSolidKind::kPlate;
  invalid_call.source_line = 392;
  invalid_call.plane_coordinate = kInfinity;
  invalid_call.half_thickness = 1;
  cases.push_back(invalid_call);
  invalid_call.source_line = 394;
  invalid_call.plane_coordinate = 0;
  invalid_call.half_thickness = 0;
  cases.push_back(invalid_call);
  invalid_call.source_line = 396;
  invalid_call.half_thickness = 1;
  invalid_call.solid_velocity = {.x = 0, .y = kInfinity, .z = 0};
  cases.push_back(invalid_call);
  cases.push_back({
      .source_line = 442,
      .old_test = "ExactObliqueTangencyDoesNotBecomeAnImpact",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = {.x = 0.6, .y = 0.8, .z = 0},
      .velocity_for_next_step = {.x = 0.8, .y = -0.6, .z = 3},
      .velocity_for_this_step = {.x = 0.8, .y = -0.6, .z = 2},
      .step_duration = 1,
      .friction = 2,
      .old_position = {.x = 0.8, .y = -0.6, .z = 2},
      .old_velocity_for_next_step = {.x = 0.8, .y = -0.6, .z = 3},
      .old_velocity_for_this_step = {.x = 0.8, .y = -0.6, .z = 2},
  });
  cases.push_back({
      .source_line = 456,
      .old_test = "FrictionlessContactPreservesSmallWorldTangentAtFastWall",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = kUnitX,
      .solid_velocity = {.x = 0, .y = 1e20, .z = 0},
      .start_position = {.x = 1, .y = 2, .z = 0},
      .velocity_for_next_step = {.x = -3, .y = 5, .z = 0},
      .velocity_for_this_step = {.x = -2, .y = 1, .z = 0},
      .step_duration = 1,
      .old_has_hit = true,
      .old_hit_fraction = 0.5,
      .old_normal_out_of_solid = kUnitX,
      .old_position = {.x = 0, .y = 3, .z = 0},
      .old_velocity_for_next_step = {.x = 0, .y = 5, .z = 0},
      .old_velocity_for_this_step = {.x = 0, .y = 1, .z = 0},
      .agreement = MainAgreement::kDiffers,
      .why_different = "a solid sliding at 1e20 m/s along its surface: this code bounds the speeds involved at a hit "
                       "times the time left by 2^40 contact tolerances, so its rounding stays within the rest gap, "
                       "and fails with kMovesTooFastForStep; the old code had no such bound",
  });
  cases.push_back({
      .source_line = 472,
      .old_test = "ObliqueMovingPlaneCarriesTheConstrainedRemainder",
      .solid_kind = MainSolidKind::kFlatSolid,
      .normal = {.x = 0.6, .y = 0.8, .z = 0},
      .solid_velocity = {.x = 3, .y = 4, .z = 1},
      .start_position = {.x = 3, .y = 4, .z = 0},
      .velocity_for_next_step = {.x = 0, .y = 0, .z = -2},
      .velocity_for_this_step = {.x = 0, .y = 0, .z = 2},
      .step_duration = 1.5,
      .old_has_hit = true,
      .old_hit_fraction = 2.0 / 3.0,
      .old_normal_out_of_solid = {.x = 0.6, .y = 0.8, .z = 0},
      .old_position = {.x = 4.5, .y = 6, .z = 3},
      .old_velocity_for_next_step = {.x = 3, .y = 4, .z = -2},
      .old_velocity_for_this_step = {.x = 3, .y = 4, .z = 2},
  });
  return cases;
}

}  // namespace fluidity::contact
