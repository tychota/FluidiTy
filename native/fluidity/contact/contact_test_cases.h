// Values and checks that the contact tests share.
//
// Purpose: give every contact test one contact tolerance, one way to compare two vectors axis by axis, one generator
//   of repeatable test cases, and one list of the scenes MoveParticle is tested on.
//
// Idea: an analytic contact test passes a fixed contact tolerance, so its expected values can be written by hand,
//   including the rest gap: a particle in contact rests one tolerance outside the surface. ExpectVec3Near fails the
//   calling test when any axis differs by more than the tolerance it is given.
//   Example: ExpectVec3Near({1, 2, 3}, {1, 2, 3 + 1e-12}, 1e-9) passes.
//   A generated test set draws its numbers from one CaseGenerator in a fixed order, so every run and every platform
//   tests the same cases.
//   Every MoveParticle test reads its scene from BuildMoveParticleCases by name, so the tests of MoveParticles on
//   both processors can run the same scenes.
//   Example: FindMoveParticleCase("DocstringExample") is the flat solid at x = 1 and the particle at x = 0.9.
//
//   The sets far from the origin (BuildObliqueCases, BuildRoundCases) use the contact tolerance of a real scene
//   instead, the one ComputeContactTolerance gives 1100 m from the origin, and ComputeEndDistanceCounts checks any of
//   them: every end at least tol / 2 outside every solid, beside a naive control that hits at the surface and has no
//   settle.
//
// Owns:    kTestContactTolerance, ExpectVec3Near, CaseGenerator, MoveParticleCase, ToSolidContactInputs,
//          BuildMoveParticleCases, FindMoveParticleCase, kFarFromOriginContactTolerance, BuildObliqueCases,
//          BuildRoundCases, EndDistanceCounts, ComputeEndDistanceCounts, BuildShorterStepCase
// Uses:    Vec3 (base/vec3.h), Mat3, kIdentity (base/mat3.h), CPU-only guard (base/cpu_only.h), Solid, ToSolid and
//          the shape structs (contact/solid_shapes.h), SolidContactInputs (contact/solid_contact_types.h),
//          ComputeContactScale, ComputeContactTolerance (contact/contact_tolerance.h), MoveParticle and FindFirstHit
//          (contact/solid_contact.h), GoogleTest
// Used by: the tests of contact/
// Read:    docs/method-paper/fluidity-method.typ <sec-contact-tolerance>, <proc-move-particle>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// m. The contact tolerance of every analytic contact test, unless the test is about the tolerance itself. It is
// small enough to stay below any length a test checks and large enough to sit far above rounding at unit scale.
inline constexpr double kTestContactTolerance = 0x1p-30;

// Fails the calling test when an axis of actual differs from the same axis of expected by more than tolerance.
//
// Units: the unit of the two vectors; tolerance is in the same unit.
// Example: ExpectVec3Near({1, 0, 0}, {1, 0, 1e-3}, 1e-6) fails on axis z.
// Read: docs/method-paper/fluidity-method.typ <sec-contact-tolerance>
inline void ExpectVec3Near(Vec3 actual, Vec3 expected, double tolerance) {
  EXPECT_NEAR(actual.x, expected.x, tolerance) << "axis x";
  EXPECT_NEAR(actual.y, expected.y, tolerance) << "axis y";
  EXPECT_NEAR(actual.z, expected.z, tolerance) << "axis z";
}

// Repeatable numbers for generated test cases: integer steps only, so every compiler and platform draws the same
// sequence. Each test set starts a new generator and states the order of its draws.
//
// Method: Next() sets state = state * 6364136223846793005 + 1442695040888963407, wrapping at 2^64, and returns the top
//   53 bits of state as a double in [0, 1): (state >> 11) * 2^-53, exact. NextSigned() is 2 * Next() - 1, in [-1, 1).
// Example: two generators built the same way return the same numbers in the same order.
class CaseGenerator {
 public:
  double Next() {
    state_ = state_ * 6364136223846793005U + 1442695040888963407U;
    return static_cast<double>(state_ >> 11U) * 0x1p-53;
  }
  double NextSigned() {
    return 2 * Next() - 1;
  }

 private:
  std::uint64_t state_ = 0x2545F4914F6CDD1DU;
};

// One scene for MoveParticle: the solids and the particle's inputs. inputs.solids and inputs.solid_count are left
// empty, because the scene is copied; ToSolidContactInputs fills them from solids.
struct MoveParticleCase {
  std::string_view name;  // The name a test finds the scene by.
  std::vector<Solid> solids;  // In ascending solid_id order unless the scene is about that order.
  SolidContactInputs inputs;  // Everything but the solids.
};

// Converts one scene into the inputs of MoveParticle: the scene's inputs with its solids.
//
// Example: the scene "DocstringExample" gives inputs with solid_count 1 that point at its one flat solid.
// Requires: test_case outlives the result, which points into test_case.solids.
inline SolidContactInputs ToSolidContactInputs(const MoveParticleCase& test_case) {
  SolidContactInputs inputs = test_case.inputs;
  inputs.solids = test_case.solids.empty() ? nullptr : test_case.solids.data();
  inputs.solid_count = static_cast<std::uint32_t>(test_case.solids.size());
  return inputs;
}

namespace move_particle_cases {

constexpr double kTolerance = kTestContactTolerance;  // m, 2^-30.
constexpr SolidMotion kNoMotion = SolidMotion();  // A solid that neither moves nor turns.
constexpr Vec3 kUnitX = {.x = 1, .y = 0, .z = 0};
constexpr Vec3 kUnitY = {.x = 0, .y = 1, .z = 0};
constexpr Vec3 kUnitZ = {.x = 0, .y = 0, .z = 1};

// A flat solid with the given surface point, normal and motion.
inline Solid
BuildFlatSolid(Vec3 point_on_surface, Vec3 normal_out_of_solid, SolidMotion motion, std::uint32_t solid_id) {
  return ToSolid(
      FlatSolid{.point_on_surface = point_on_surface, .normal_out_of_solid = normal_out_of_solid},
      motion,
      solid_id
  );
}

// The fixed floor: the flat solid z <= 0.
inline Solid BuildFloor(std::uint32_t solid_id) {
  return BuildFlatSolid({}, kUnitZ, kNoMotion, solid_id);
}

// The fixed upright flat solid at x = 1, which fills x >= 1.
inline Solid BuildUprightAtXOne(std::uint32_t solid_id) {
  return BuildFlatSolid(kUnitX, -kUnitX, kNoMotion, solid_id);
}

// The domain box from {0, 0, 0} to {1, 1, 1} with an open top.
inline Solid BuildUnitDomainBox(std::uint32_t solid_id) {
  return ToSolid(DomainBox{.low_corner = {}, .high_corner = {.x = 1, .y = 1, .z = 1}, .closed = {}}, solid_id);
}

// The particle of one scene, with both velocities equal.
struct CaseParticle {
  Vec3 start_position{};  // m, world.
  Vec3 velocity{};  // m/s, world, both velocities.
  double step_duration = 0;  // s.
  double friction = 0;  // No unit.
};

// Inputs for one particle, with the test contact tolerance, no start overlap and no committed plate.
inline SolidContactInputs BuildCaseInputs(CaseParticle particle) {
  SolidContactInputs inputs;
  inputs.start_position = particle.start_position;
  inputs.velocity = {.for_next_step = particle.velocity, .for_this_step = particle.velocity};
  inputs.step_duration = particle.step_duration;
  inputs.friction = particle.friction;
  inputs.contact_tolerance = kTolerance;
  return inputs;
}

// The scenes that follow the method paper's examples and the docstring of MoveParticle.
inline std::vector<MoveParticleCase> BuildExampleCases() {
  std::vector<MoveParticleCase> cases;
  cases.push_back({
      .name = "NoSolid",
      .solids = {},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.1, .y = -0.2, .z = 0.3},
          .velocity = {.x = 1.5, .y = -2.25, .z = 0.75},
          .step_duration = 0.3,
          .friction = 0.5,
      }),
  });
  // The flat solid of surface x = 1 and normal {-1, 0, 0}, and the particle at x = 0.9 moving at 1 m/s for 0.2 s.
  cases.push_back({
      .name = "DocstringExample",
      .solids = {BuildUprightAtXOne(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.9, .y = 0, .z = 0},
          .velocity = kUnitX,
          .step_duration = 0.2,
          .friction = 0,
      }),
  });
  // The docstring example above a floor that the particle never reaches: the first hit is on the second solid.
  cases.push_back({
      .name = "SecondSolidHitFirst",
      .solids = {BuildFloor(1), BuildUprightAtXOne(2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.9, .y = 0, .z = 0.5},
          .velocity = kUnitX,
          .step_duration = 0.2,
          .friction = 0,
      }),
  });
  Solid reacting_upright = BuildUprightAtXOne(1);
  reacting_upright.reports_reaction = true;
  cases.push_back({
      .name = "ReactingSolid",
      .solids = {reacting_upright},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.9, .y = 0, .z = 0},
          .velocity = kUnitX,
          .step_duration = 0.2,
          .friction = 0,
      }),
  });
  // The method paper's figure of one step: a reacting floor rising at 0.5 m/s, friction 0.25, two velocities.
  Solid rising_floor = BuildFlatSolid({}, kUnitZ, {.linear_velocity = {.x = 0, .y = 0, .z = 0.5}}, 1);
  rising_floor.reports_reaction = true;
  SolidContactInputs figure_inputs = BuildCaseInputs({
      .start_position = {.x = 0, .y = 0, .z = 1.2},
      .velocity = {},
      .step_duration = 1,
      .friction = 0.25,
  });
  figure_inputs.velocity = {
      .for_next_step = {.x = 1.5, .y = 0, .z = -2.5},
      .for_this_step = {.x = 2.5, .y = 0, .z = -1.5},
  };
  cases.push_back({.name = "PaperFigureRisingFloor", .solids = {rising_floor}, .inputs = figure_inputs});
  // The fixed floor and the two velocities of the method paper's response section, friction 0.5.
  SolidContactInputs different_inputs = BuildCaseInputs({
      .start_position = {.x = 0, .y = 0, .z = 0.5},
      .velocity = {},
      .step_duration = 1,
      .friction = 0.5,
  });
  different_inputs.velocity = {
      .for_next_step = {.x = 1, .y = 0, .z = -3},
      .for_this_step = {.x = 0, .y = 0, .z = -1},
  };
  cases.push_back({.name = "DifferentVelocities", .solids = {BuildFloor(1)}, .inputs = different_inputs});
  return cases;
}

// The plate scenes of the proof of concept, as closed forms: plate half thickness 1/32, T = 1, for each axis e and
// normal sigma * e. Plate row 0: start -e at 2e, plate fixed at 0. Plate row 1: at rest at 0, plate centre from -e at
// 2e. Plate row 2: from 0 at e, plate centre from -e at e. Names: PlateRow<number><axis><Plus or Minus>Normal, in that
// order, the plus normal first.
inline std::vector<MoveParticleCase> BuildPlateRowCases() {
  // String literals live as long as the program, as the names of the scenes need.
  static constexpr std::array<std::string_view, 18> kNames = {
      "PlateRow0XPlusNormal",
      "PlateRow0XMinusNormal",
      "PlateRow0YPlusNormal",
      "PlateRow0YMinusNormal",
      "PlateRow0ZPlusNormal",
      "PlateRow0ZMinusNormal",
      "PlateRow1XPlusNormal",
      "PlateRow1XMinusNormal",
      "PlateRow1YPlusNormal",
      "PlateRow1YMinusNormal",
      "PlateRow1ZPlusNormal",
      "PlateRow1ZMinusNormal",
      "PlateRow2XPlusNormal",
      "PlateRow2XMinusNormal",
      "PlateRow2YPlusNormal",
      "PlateRow2YMinusNormal",
      "PlateRow2ZPlusNormal",
      "PlateRow2ZMinusNormal",
  };
  // Per plate row, along e: the plate's centre and speed, the particle's start and speed.
  constexpr std::array<double, 3> kPlateCenter = {0, -1, -1};  // m
  constexpr std::array<double, 3> kPlateSpeed = {0, 2, 1};  // m/s
  constexpr std::array<double, 3> kStart = {-1, 0, 0};  // m
  constexpr std::array<double, 3> kParticleSpeed = {2, 0, 1};  // m/s
  constexpr double kHalfThickness = 1.0 / 32;  // m
  const std::array<Vec3, 3> axes = {kUnitX, kUnitY, kUnitZ};
  std::vector<MoveParticleCase> cases;
  for (std::size_t name_index = 0; name_index < kNames.size(); ++name_index) {
    const std::size_t plate_row = name_index / 6;
    const Vec3 axis = axes.at((name_index / 2) % 3);
    const double normal_sign = name_index % 2 == 0 ? 1.0 : -1.0;
    const Solid plate = ToSolid(
        Plate{
            .center = kPlateCenter.at(plate_row) * axis,
            .normal = normal_sign * axis,
            .half_thickness = kHalfThickness,
        },
        {.linear_velocity = kPlateSpeed.at(plate_row) * axis, .angular_velocity = {}},
        1
    );
    cases.push_back({
        .name = kNames.at(name_index),
        .solids = {plate},
        .inputs = BuildCaseInputs({
            .start_position = kStart.at(plate_row) * axis,
            .velocity = kParticleSpeed.at(plate_row) * axis,
            .step_duration = 1,
            .friction = 0,
        }),
    });
  }
  return cases;
}

// The scenes with several hits in one step: in the domain box, in the octagon room and between two flat solids.
inline std::vector<MoveParticleCase> BuildSeveralHitCases() {
  std::vector<MoveParticleCase> cases;
  cases.push_back({
      .name = "FrictionOnSideBottom",
      .solids = {BuildUnitDomainBox(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.2, .y = 0.5, .z = 0.1},
          .velocity = {.x = 3, .y = 0, .z = -1},
          .step_duration = 0.2,
          .friction = 0.5,
      }),
  });
  cases.push_back({
      .name = "DomainCorner",
      .solids = {BuildUnitDomainBox(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.1, .y = 0.2, .z = 0.3},
          .velocity = {.x = -1, .y = -1, .z = -1},
          .step_duration = 0.5,
          .friction = 0,
      }),
  });
  Solid reacting_domain_box = BuildUnitDomainBox(1);
  reacting_domain_box.reports_reaction = true;
  cases.push_back({
      .name = "ReactingDomainCorner",
      .solids = {reacting_domain_box},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.1, .y = 0.2, .z = 0.3},
          .velocity = {.x = -1, .y = -1, .z = -1},
          .step_duration = 0.5,
          .friction = 0,
      }),
  });
  // Eight flat solids k = 0..7 at distance 1 from the origin, each looking at the origin, ids k + 1.
  std::vector<Solid> octagon_sides;
  for (int side_index = 0; side_index < 8; ++side_index) {
    const double angle = side_index * std::numbers::pi / 4;  // rad
    const Vec3 point_on_surface = {.x = std::cos(angle), .y = std::sin(angle), .z = 0};
    octagon_sides.push_back(
        BuildFlatSolid(point_on_surface, -point_on_surface, kNoMotion, static_cast<std::uint32_t>(side_index + 1))
    );
  }
  cases.push_back({
      .name = "OctagonRoom",
      .solids = octagon_sides,
      .inputs = BuildCaseInputs({
          .start_position = {},
          .velocity = {.x = 100, .y = 20, .z = 0},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  Solid reacting_floor = BuildFloor(1);
  reacting_floor.reports_reaction = true;
  Solid reacting_upright = BuildUprightAtXOne(2);
  reacting_upright.reports_reaction = true;
  cases.push_back({
      .name = "TwoReactingSolids",
      .solids = {reacting_floor, reacting_upright},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.8, .y = 0, .z = 0.1},
          .velocity = {.x = 2, .y = 0, .z = -2},
          .step_duration = 0.2,
          .friction = 0,
      }),
  });
  // The two reacting solids, with for_next_step {-1, 0, -2}: it approaches the floor but moves away from the upright,
  // so the upright's response changes for_next_step by exactly zero.
  SolidContactInputs zero_change_inputs = BuildCaseInputs({
      .start_position = {.x = 0.8, .y = 0, .z = 0.1},
      .velocity = {.x = 2, .y = 0, .z = -2},
      .step_duration = 0.2,
      .friction = 0,
  });
  zero_change_inputs.velocity.for_next_step = {.x = -1, .y = 0, .z = -2};
  cases.push_back({
      .name = "SecondReactingSolidWithZeroChange",
      .solids = {reacting_floor, reacting_upright},
      .inputs = zero_change_inputs,
  });
  // The reacting upright alone, with for_next_step {-1, 0, 0} moving away from it: a zero change.
  SolidContactInputs away_inputs = BuildCaseInputs({
      .start_position = {.x = 0.9, .y = 0, .z = 0},
      .velocity = kUnitX,
      .step_duration = 0.2,
      .friction = 0,
  });
  away_inputs.velocity.for_next_step = -kUnitX;
  cases.push_back({.name = "ReactingSolidWithZeroChange", .solids = {reacting_upright}, .inputs = away_inputs});
  cases.push_back({
      .name = "ReactingUprightAfterFloor",
      .solids = {BuildFloor(1), reacting_upright},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.8, .y = 0, .z = 0.1},
          .velocity = {.x = 2, .y = 0, .z = -2},
          .step_duration = 0.2,
          .friction = 0,
      }),
  });
  Solid rough_floor = BuildFloor(1);
  rough_floor.friction_scale = 2;
  cases.push_back({
      .name = "FrictionScaleTwo",
      .solids = {rough_floor},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 0.5},
          .velocity = {.x = 3, .y = 0, .z = -1},
          .step_duration = 1,
          .friction = 0.5,
      }),
  });
  // A ramp at x = 1 leaning back: normal {-0.6, 0, 0.8}, so a particle sliding along the floor goes up it.
  cases.push_back({
      .name = "FloorThenRamp",
      .solids = {BuildFloor(1), BuildFlatSolid(kUnitX, {.x = -0.6, .y = 0, .z = 0.8}, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5, .y = 0, .z = 0.1},
          .velocity = {.x = 1, .y = 0, .z = -1},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  Solid moving_upright = BuildFlatSolid(kUnitX, -kUnitX, {.linear_velocity = {.x = -0.5, .y = 0, .z = 0}}, 2);
  moving_upright.reports_reaction = true;
  // The upright flat solid at x = 1 moving toward the particle at 0.5 m/s, hit by the second sweep, which starts at
  // 0.1 - tol s. It reports its reaction, so the position of its hit is kept.
  cases.push_back({
      .name = "FloorThenMovingUpright",
      .solids = {BuildFloor(1), moving_upright},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 0.1},
          .velocity = {.x = 1, .y = 0, .z = -1},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // A fixed floor and an upright flat solid at x = 1 tilted so slightly that its response leaves the particle
  // approaching the floor at 1.5 tol m/s: a real approach, far above the floor's approach allowance, so the touched
  // floor is hit again at once.
  const double tilt = 1.5 * kTolerance;  // No unit.
  cases.push_back({
      .name = "TiltedUprightPushesBackToFloor",
      .solids = {BuildFloor(1), BuildFlatSolid(kUnitX, math::Normalized({.x = -1, .y = 0, .z = -tilt}), kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5, .y = 0, .z = 0.1},
          .velocity = {.x = 1, .y = 0, .z = -1},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // The same tilted upright over a floor rising at 0.25 m/s: the floor carries the particle into the overhang, a real
  // approach to the touched floor after the upright's response, so the moving floor is hit again.
  const Solid rising_floor = BuildFlatSolid({}, kUnitZ, {.linear_velocity = {.x = 0, .y = 0, .z = 0.25}}, 1);
  cases.push_back({
      .name = "TiltedUprightOverRisingFloor",
      .solids = {rising_floor, BuildFlatSolid(kUnitX, math::Normalized({.x = -1, .y = 0, .z = -tilt}), kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5, .y = 0, .z = 0.1},
          .velocity = {.x = 1, .y = 0, .z = -1},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // A slip just below the stick limit on one fixed flat solid through the origin: the response returns about
  // {1.5e-7, 5.4e-8, 2.4e-7} m/s with an approach of -3.8e-17 m/s, rounding of the incoming speed. The bound of the
  // incoming velocity, 7.3e-14 m/s, covers it; one from the returned velocity, 6.3e-21 m/s, would not.
  // The y axis of this random normal is close to 1 / Sqrt(pi) by chance, not by meaning.
  // NOLINTNEXTLINE(modernize-use-std-numbers): see the comment above.
  const Vec3 near_stick_normal = {.x = -0.7462749575584682, .y = 0.5645329322916014, .z = 0.3526701803092409};
  cases.push_back({
      .name = "NearStickSlipOnObliqueFlatSolid",
      .solids = {BuildFlatSolid({}, near_stick_normal, kNoMotion, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = -0.13431360048800245, .y = 0.033902899848675824, .z = -0.07045626995076695},
          .velocity = {.x = 2.7659064005760463, .y = -0.16737819979015056, .z = 2.2161120245085217},
          .step_duration = 1,
          .friction = 2.374646385678357,
      }),
  });
  // A chasing solid: a fixed flat solid A through the origin, and a flat solid B 10 m away along a normal
  // at right angles to A's up to rounding, chasing the particle at 100 m/s. A is hit at 0.05 s, B 0.37 ms later; B's
  // response leaves along A's normal rounding of B's speeds, larger than A's own allowance.
  const Vec3 first_normal = {.x = 0.5752141167476622, .y = 0.014005927519369528, .z = 0.8178829707779282};
  const Vec3 chasing_normal = {.x = -0.32394700710139007, .y = 0.9220068524474481, .z = 0.2120417426593205};
  cases.push_back({
      .name = "ChasingSolidAtRightAngles",
      .solids =
          {
              BuildFlatSolid({}, first_normal, kNoMotion, 1),
              BuildFlatSolid(10 * chasing_normal, -chasing_normal, {.linear_velocity = -100 * chasing_normal}, 2),
          },
      .inputs = BuildCaseInputs({
          .start_position = 0.05 * first_normal,
          .velocity = {.x = -0.45638497505415865, .y = -0.35221335191322467, .z = -0.8956634031667221},
          .step_duration = 0.2,
          .friction = 0,
      }),
  });
  // One fixed flat solid with normal {0.6, 0, 0.8} through the origin. Rounding leaves the hit position 2.8e-18 m
  // outside the rest gap, and the response leaves an approach of -2^-55 m/s.
  cases.push_back({
      .name = "ObliqueFlatSolidRoundingAtHit",
      .solids = {BuildFlatSolid({}, {.x = 0.6, .y = 0, .z = 0.8}, kNoMotion, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.125, .y = 0, .z = 0.125},
          .velocity = {.x = -2, .y = 0, .z = -2},
          .step_duration = 0.5,
          .friction = 0,
      }),
  });
  // A box turning at 1 rad/s about z and a particle at rest on the path of its face kHighX: the face reaches it at
  // 0.8857 s, and the rest of the step is carried in the box's axes.
  const Solid turning_box_beside_particle = ToSolid(
      Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
      {.linear_velocity = {}, .angular_velocity = kUnitZ},
      1
  );
  cases.push_back({
      .name = "TurningBoxHitsRestingParticle",
      .solids = {turning_box_beside_particle},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.6 * std::cos(0.3), .y = 0.6 * std::sin(0.3), .z = 0},
          .velocity = {},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // A sphere of radius 0.5 m at the origin rising at 0.5 m/s and spinning at {2, 0, 0} rad/s, and a particle falling
  // on its top: friction 2 sticks, so the particle takes the surface's velocity there, about {0, -1, 0.5} m/s.
  const Solid spinning_sphere = ToSolid(
      Sphere{.center = {}, .radius = 0.5},
      {.linear_velocity = {.x = 0, .y = 0, .z = 0.5}, .angular_velocity = {.x = 2, .y = 0, .z = 0}},
      1
  );
  cases.push_back({
      .name = "SpinningSphereStick",
      .solids = {spinning_sphere},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 1},
          .velocity = {.x = 0, .y = 0, .z = -1},
          .step_duration = 1,
          .friction = 2,
      }),
  });
  return cases;
}

// The scenes whose move fails, or whose start or end is decided by the rules at the start and the end of the step.
inline std::vector<MoveParticleCase> BuildStartAndEndCases() {
  std::vector<MoveParticleCase> cases;
  // A ceiling 3 tol above the fixed floor closes at 1.75 tol m/s on a particle at rest at z = tol: it hits at 4/7 s and
  // carries the particle down onto the floor, a squeeze.
  const Solid ceiling = BuildFlatSolid(
      {.x = 0, .y = 0, .z = 3 * kTolerance},
      -kUnitZ,
      {.linear_velocity = {.x = 0, .y = 0, .z = -1.75 * kTolerance}},
      2
  );
  cases.push_back({
      .name = "CeilingClosingFasterThanParticleLeaves",
      .solids = {BuildFloor(1), ceiling},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = kTolerance},
          .velocity = {},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // Two solids, ids 7 and 9; the start is 0.01 m inside the second, deeper than the overlap tolerance.
  SolidContactInputs inside_inputs = BuildCaseInputs({
      .start_position = {.x = 0, .y = 0, .z = -0.01},
      .velocity = {},
      .step_duration = 0.1,
      .friction = 0,
  });
  inside_inputs.start_overlap_tolerance = 0.001;
  cases.push_back({
      .name = "StartsInsideSecondSolid",
      .solids = {BuildUprightAtXOne(7), BuildFloor(9)},
      .inputs = inside_inputs,
  });
  // A start exactly on the fixed floor, moving away at 1 m/s: distance 0 is not inside.
  cases.push_back({
      .name = "StartOnSurfaceMovingAway",
      .solids = {BuildFloor(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 0},
          .velocity = {.x = 0, .y = 0, .z = 1},
          .step_duration = 0.5,
          .friction = 0,
      }),
  });
  // A floor sinking at 1 m/s and a particle falling with it, 0.5 m above: never hit. At the end the particle is below
  // where the floor started, so the end check must read the floor at its end pose.
  cases.push_back({
      .name = "FollowingSinkingFloor",
      .solids = {BuildFlatSolid({}, kUnitZ, {.linear_velocity = {.x = 0, .y = 0, .z = -1}}, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 0.5},
          .velocity = {.x = 0, .y = 0, .z = -1},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // A box turning at 1 rad/s about z and a particle at rest 2^-17 m above its top face, on the turn axis: the gap never
  // changes, but the stepping search would need 182 advances to reach the end of the step (FindFirstHit's tests,
  // TurningBoxHoverOnSpinAxisBeyondAdvanceLimit_IsCloseCall).
  const Solid turning_box = ToSolid(
      Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
      {.linear_velocity = {}, .angular_velocity = kUnitZ},
      1
  );
  cases.push_back({
      .name = "TurningBoxHoverOnSpinAxis",
      .solids = {turning_box},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 0.5 + 0x1p-17},
          .velocity = {},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  return cases;
}

// A flat solid with normal {0, 0, 1} whose surface passes through {0, 0, height}, moving at linear_velocity.
inline Solid BuildMovingFloor(double height, Vec3 linear_velocity, std::uint32_t solid_id) {
  return BuildFlatSolid({.x = 0, .y = 0, .z = height}, kUnitZ, {.linear_velocity = linear_velocity}, solid_id);
}

// The plate of the start rules' scenes: middle plane through the origin, normal {0, 0, 1}, half thickness 0.01 m, id 5,
// moving at linear_velocity.
inline Solid BuildStartPlate(Vec3 linear_velocity) {
  return ToSolid(
      Plate{.center = {}, .normal = kUnitZ, .half_thickness = 0.01},
      {.linear_velocity = linear_velocity, .angular_velocity = {}},
      5
  );
}

// Where a particle at rest starts on the z axis, and how deep inside a solid it may start.
struct RestingStart {
  double height = 0;  // m, world z.
  double start_overlap_tolerance = 0;  // m.
};

// Inputs of a particle at rest at {0, 0, start.height} for 0.1 s, friction 0, with the start's overlap tolerance.
inline SolidContactInputs BuildRestingStartInputs(RestingStart start) {
  SolidContactInputs inputs = BuildCaseInputs({
      .start_position = {.x = 0, .y = 0, .z = start.height},
      .velocity = {},
      .step_duration = 0.1,
      .friction = 0,
  });
  inputs.start_overlap_tolerance = start.start_overlap_tolerance;
  return inputs;
}

// Inputs of a particle on the plate of BuildStartPlate at {0, 0, height}, committed to plate_side, at rest for 0.1 s,
// with a start overlap tolerance.
inline SolidContactInputs BuildPlateStartInputs(double height, PlateSide plate_side, double start_overlap_tolerance) {
  SolidContactInputs inputs =
      BuildRestingStartInputs({.height = height, .start_overlap_tolerance = start_overlap_tolerance});
  inputs.plate_side = plate_side;
  return inputs;
}

// The scenes of the start rules on a floor: a start within the tolerance, on the surface or inside, on fixed and
// moving floors.
inline std::vector<MoveParticleCase> BuildFloorStartCases() {
  std::vector<MoveParticleCase> cases;
  const Solid rising_floor = BuildMovingFloor(0, {.x = 0, .y = 0, .z = 2}, 1);
  // The method paper's example: the floor rises at 2 m/s under a particle at rest at the rest gap.
  cases.push_back({
      .name = "FlatSolidPushesRestingParticle",
      .solids = {rising_floor},
      .inputs = BuildRestingStartInputs({.height = kTolerance, .start_overlap_tolerance = 0}),
  });
  cases.push_back({
      .name = "FlatSolidPushesParticleWithinTolerance",
      .solids = {rising_floor},
      .inputs = BuildRestingStartInputs({.height = kTolerance / 2, .start_overlap_tolerance = 0}),
  });
  cases.push_back({
      .name = "TouchingAndLeaving",
      .solids = {BuildFloor(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = kTolerance},
          .velocity = kUnitZ,
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  cases.push_back({
      .name = "ShallowOverlap",
      .solids = {BuildFloor(1)},
      .inputs = BuildRestingStartInputs({.height = -0.0005, .start_overlap_tolerance = 0.001}),
  });
  cases.push_back({
      .name = "ShallowOverlapRisingFloor",
      .solids = {BuildMovingFloor(0, kUnitZ, 1)},
      .inputs = BuildRestingStartInputs({.height = -0.0005, .start_overlap_tolerance = 0.001}),
  });
  cases.push_back({
      .name = "DeepOverlap",
      .solids = {BuildFloor(1)},
      .inputs = BuildRestingStartInputs({.height = -0.01, .start_overlap_tolerance = 0.001}),
  });
  // At the end of the step the sinking floor is 0.1 m lower, far below the start: only the pose of time 0 finds the
  // start inside.
  cases.push_back({
      .name = "DeepOverlapSinkingFloor",
      .solids = {BuildMovingFloor(0, -kUnitZ, 1)},
      .inputs = BuildRestingStartInputs({.height = -0.01, .start_overlap_tolerance = 0.001}),
  });
  // Plan decision P12: the paper's second example, moving away from the fixed floor.
  cases.push_back({
      .name = "StartsWithinToleranceMovingAway",
      .solids = {BuildFloor(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = kTolerance / 4},
          .velocity = {.x = 1, .y = 0, .z = 2},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // The same particle above a floor rising at 1 m/s: it still moves away, at 1 m/s relative to the floor.
  cases.push_back({
      .name = "StartsWithinToleranceOfRisingFloorMovingAway",
      .solids = {BuildMovingFloor(0, kUnitZ, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = kTolerance / 4},
          .velocity = {.x = 1, .y = 0, .z = 2},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // for_this_step leaves the fixed floor and for_next_step approaches it: either approaching makes the start a hit.
  SolidContactInputs next_step_inputs = BuildCaseInputs({
      .start_position = {.x = 0, .y = 0, .z = kTolerance / 2},
      .velocity = {.x = 1, .y = 0, .z = 1},
      .step_duration = 0.1,
      .friction = 0,
  });
  next_step_inputs.velocity.for_next_step = -kUnitZ;
  cases.push_back(
      {.name = "ForNextStepApproachesWithinTolerance", .solids = {BuildFloor(1)}, .inputs = next_step_inputs}
  );
  // for_this_step approaches the fixed floor at tol m/s, so slowly that the touching rule alone would let it rest
  // (rate * T = -0.1 tol), and for_next_step leaves: the start rule still makes it a hit and stops the approach.
  SolidContactInputs slow_inputs = BuildCaseInputs({
      .start_position = {.x = 0, .y = 0, .z = kTolerance / 2},
      .velocity = {.x = 1, .y = 0, .z = -kTolerance},
      .step_duration = 0.1,
      .friction = 0,
  });
  slow_inputs.velocity.for_next_step = kUnitZ;
  cases.push_back({.name = "SlowApproachWithinTolerance", .solids = {BuildFloor(1)}, .inputs = slow_inputs});
  // A floor sinking at 2 m/s and a particle tol / 2 above it falling at 1 m/s: it falls, but leaves the floor at 1 m/s,
  // so the move to the rest gap is silent.
  cases.push_back({
      .name = "StartsWithinToleranceOfSinkingFloorFollowing",
      .solids = {BuildMovingFloor(0, {.x = 0, .y = 0, .z = -2}, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = kTolerance / 2},
          .velocity = -kUnitZ,
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // Five floors rising at 1, 1.1, ..., 1.4 m/s, ids 1 to 5, at heights 0, 0.0001, ..., 0.0004 m, and a start 0.0005 m
  // below the first: the start is inside all five, which share one direction of the move and of the joint response;
  // the fastest pushes, one hit. (At one speed they would be exact copies, one active surface.)
  std::vector<Solid> stacked_floors;
  stacked_floors.reserve(5);
  for (int floor_index = 0; floor_index < 5; ++floor_index) {
    const Vec3 rising = {.x = 0, .y = 0, .z = 1 + 0.1 * floor_index};  // m/s
    stacked_floors.push_back(
        BuildMovingFloor(0.0001 * floor_index, rising, static_cast<std::uint32_t>(floor_index + 1))
    );
  }
  cases.push_back({
      .name = "FiveFloorsWithinStartOverlap",
      .solids = stacked_floors,
      .inputs = BuildRestingStartInputs({.height = -0.0005, .start_overlap_tolerance = 0.001}),
  });
  // A sphere of radius 0.5 m at the origin moving at 1 m/s toward a particle at rest tol / 2 from its surface.
  cases.push_back({
      .name = "SphereApproachingWithinTolerance",
      .solids = {ToSolid(Sphere{.center = {}, .radius = 0.5}, {.linear_velocity = -kUnitX, .angular_velocity = {}}, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = -(0.5 + kTolerance / 2), .y = 0, .z = 0},
          .velocity = {},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // A sphere of radius 0.5 m at the origin moving at {1, 0, 0} m/s and spinning at {0, 2, 0} rad/s, and a particle tol
  // / 2 above its top leaving it at {1, 0, 0.1} m/s: the surface there moves at {1 + 2 * (0.5 + tol / 2), 0, 0} m/s,
  // with no part along the normal, so the move to the rest gap is silent. Posed at the end of the step, the arm from
  // the centre would lean by -0.1 m along x and the spin would give the surface 0.2 m/s along the normal, an approach.
  const Solid spinning_moving_sphere =
      ToSolid(Sphere{.center = {}, .radius = 0.5}, {.linear_velocity = kUnitX, .angular_velocity = 2 * kUnitY}, 1);
  cases.push_back({
      .name = "SpinningMovingSphereStartWithinTolerance",
      .solids = {spinning_moving_sphere},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 0.5 + kTolerance / 2},
          .velocity = {.x = 1, .y = 0, .z = 0.1},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // The turning box of TurningBoxHoverOnSpinAxis and a particle at rest tol / 2 above its top face, on the turn axis:
  // moved silently to the rest gap, then hit at 0, since a turning solid has no touching rule, and carried in the box's
  // axes, where it stays put.
  const Solid turning_box = ToSolid(
      Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
      {.linear_velocity = {}, .angular_velocity = kUnitZ},
      1
  );
  cases.push_back({
      .name = "TurningBoxStartWithinTolerance",
      .solids = {turning_box},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = 0.5 + kTolerance / 2},
          .velocity = {},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  return cases;
}

// The scenes of the start rules on the plate of BuildStartPlate, committed to either side or to none, fixed and
// moving along its normal at 1 m/s.
inline std::vector<MoveParticleCase> BuildPlateStartCases() {
  std::vector<MoveParticleCase> cases;
  const Solid fixed_plate = BuildStartPlate({});
  const Solid moving_plate = BuildStartPlate(kUnitZ);
  constexpr PlateSide kAlongNormal = {.plate_id = 5, .side = SolidSide::kAlongNormal};
  constexpr PlateSide kAgainstNormal = {.plate_id = 5, .side = SolidSide::kAgainstNormal};
  constexpr PlateSide kNotCommitted = {};
  // Inside the thickness, nearer the low face.
  cases.push_back({
      .name = "CommittedSideInsideThickness",
      .solids = {fixed_plate},
      .inputs = BuildPlateStartInputs(-0.001, kAlongNormal, 0.05),
  });
  cases.push_back({
      .name = "UncommittedInsideThickness",
      .solids = {fixed_plate},
      .inputs = BuildPlateStartInputs(-0.001, kNotCommitted, 0.05),
  });
  cases.push_back({
      .name = "CommittedAgainstNormalInsideThickness",
      .solids = {fixed_plate},
      .inputs = BuildPlateStartInputs(0.001, kAgainstNormal, 0.05),
  });
  cases.push_back({
      .name = "CommittedSideInsideMovingPlate",
      .solids = {moving_plate},
      .inputs = BuildPlateStartInputs(-0.001, kAlongNormal, 0.05),
  });
  cases.push_back({
      .name = "UncommittedInsideMovingPlate",
      .solids = {moving_plate},
      .inputs = BuildPlateStartInputs(-0.001, kNotCommitted, 0.05),
  });
  // Beyond the other face by 0.01 m, more than tol.
  cases.push_back({
      .name = "CommittedBeyondOtherFace",
      .solids = {fixed_plate},
      .inputs = BuildPlateStartInputs(-0.02, kAlongNormal, 0.05),
  });
  cases.push_back({
      .name = "CommittedAgainstNormalBeyondOtherFace",
      .solids = {fixed_plate},
      .inputs = BuildPlateStartInputs(0.02, kAgainstNormal, 0.05),
  });
  cases.push_back({
      .name = "CommittedBeyondOtherFaceOfMovingPlate",
      .solids = {moving_plate},
      .inputs = BuildPlateStartInputs(-0.02, kAlongNormal, 0.05),
  });
  // The committed distance, -0.03 m, is also deeper than this overlap tolerance: the wrong side is said first.
  cases.push_back({
      .name = "CommittedBeyondOtherFaceSmallOverlap",
      .solids = {fixed_plate},
      .inputs = BuildPlateStartInputs(-0.02, kAlongNormal, 0.001),
  });
  // Beyond the other face by 1.5 tol: the wrong side, just past the tolerance.
  cases.push_back({
      .name = "CommittedJustBeyondToleranceOfOtherFace",
      .solids = {fixed_plate},
      .inputs = BuildPlateStartInputs(-(0.01 + 1.5 * kTolerance), kAlongNormal, 0.05),
  });
  // Beyond the other face by tol / 2 only: not the wrong side; pushed through to the committed side.
  cases.push_back({
      .name = "CommittedWithinToleranceOfOtherFace",
      .solids = {fixed_plate},
      .inputs = BuildPlateStartInputs(-(0.01 + kTolerance / 2), kAlongNormal, 0.05),
  });
  return cases;
}

// The scenes of the start rules against several solids: corners, the domain box, wedges and reacting solids.
inline std::vector<MoveParticleCase> BuildSeveralSolidStartCases() {
  std::vector<MoveParticleCase> cases;
  const CaseParticle into_corner = {
      .start_position = {.x = 1 - kTolerance / 2, .y = 0, .z = kTolerance / 2},
      .velocity = {.x = 1, .y = 0, .z = -1},
      .step_duration = 0.1,
      .friction = 0,
  };
  // The fixed floor and an upright flat solid at x = 1 moving toward the particle at 0.5 m/s, both within tol.
  cases.push_back({
      .name = "CornerWithinTolerance",
      .solids = {BuildFloor(1), BuildFlatSolid(kUnitX, -kUnitX, {.linear_velocity = {.x = -0.5, .y = 0, .z = 0}}, 2)},
      .inputs = BuildCaseInputs(into_corner),
  });
  Solid reacting_floor = BuildFloor(1);
  reacting_floor.reports_reaction = true;
  Solid reacting_upright = BuildFlatSolid(kUnitX, -kUnitX, {.linear_velocity = {.x = -0.5, .y = 0, .z = 0}}, 2);
  reacting_upright.reports_reaction = true;
  cases.push_back({
      .name = "TwoReactingSolidsWithinTolerance",
      .solids = {reacting_floor, reacting_upright},
      .inputs = BuildCaseInputs(into_corner),
  });
  // A floor rising at 1 m/s under a particle falling at 1 m/s tol / 2 above it, and the fixed upright at x = 1, which
  // the start is 0.0002 m inside: the floor fills the record with kind kHit, and the upright makes it
  // kStartedInsideSolid.
  SolidContactInputs inside_second_inputs = BuildCaseInputs({
      .start_position = {.x = 1.0002, .y = 0, .z = kTolerance / 2},
      .velocity = -kUnitZ,
      .step_duration = 0.1,
      .friction = 0,
  });
  inside_second_inputs.start_overlap_tolerance = 0.001;
  cases.push_back({
      .name = "InsideUprightAfterApproachingFloor",
      .solids = {BuildMovingFloor(0, kUnitZ, 1), BuildUprightAtXOne(2)},
      .inputs = inside_second_inputs,
  });
  // The unit domain box and a particle at rest tol / 4 from side_left and 0.3 tol above side_bottom.
  cases.push_back({
      .name = "DomainCornerWithinTolerance",
      .solids = {BuildUnitDomainBox(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = kTolerance / 4, .y = 0.5, .z = 0.3 * kTolerance},
          .velocity = {},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // The same box and a particle tol / 4 from side_left and 1.5 tol above side_bottom, sinking at 4 tol m/s: side_bottom
  // is farther than tol, so the start rules leave it to the sweep, which finds no hit within the step.
  cases.push_back({
      .name = "DomainCornerSecondSideBeyondTolerance",
      .solids = {BuildUnitDomainBox(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = kTolerance / 4, .y = 0.5, .z = 1.5 * kTolerance},
          .velocity = {.x = 0, .y = 0, .z = -4 * kTolerance},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // A unit domain box whose side_back is open as well as its top, and a particle leaving through side_back.
  const Solid domain_box_open_back = ToSolid(
      DomainBox{
          .low_corner = {},
          .high_corner = {.x = 1, .y = 1, .z = 1},
          .closed = {.left = true, .right = true, .front = true, .back = false, .bottom = true, .top = false},
      },
      1
  );
  cases.push_back({
      .name = "DomainBoxOpenBack",
      .solids = {domain_box_open_back},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5, .y = 0.9, .z = 0.5},
          .velocity = kUnitY,
          .step_duration = 0.2,
          .friction = 0,
      }),
  });
  // The same box and a particle tol / 4 from side_left and tol / 4 below the open top, rising at 1 m/s through it.
  cases.push_back({
      .name = "DomainCornerAtOpenTop",
      .solids = {BuildUnitDomainBox(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = kTolerance / 4, .y = 0.5, .z = 1 - kTolerance / 4},
          .velocity = kUnitZ,
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // The same box and a start 0.0002 m beyond side_left and 0.0001 m below side_bottom, moving into both.
  SolidContactInputs domain_inside_inputs = BuildCaseInputs({
      .start_position = {.x = -0.0002, .y = 0.5, .z = -0.0001},
      .velocity = {.x = -1, .y = 0, .z = -1},
      .step_duration = 0.1,
      .friction = 0,
  });
  domain_inside_inputs.start_overlap_tolerance = 0.001;
  cases.push_back({
      .name = "DomainCornerSlightlyInside",
      .solids = {BuildUnitDomainBox(1)},
      .inputs = domain_inside_inputs,
  });
  // A wedge sharper than a right angle (36.87 degrees, normals -0.8 apart): the fixed floor, and a flat solid rising at
  // 0.25 m/s with normal {-0.6, 0, -0.8} whose surface passes through {0, 0, 1.625 tol}. The particle falls on the
  // floor from tol / 2, short of its rest gap by 0.5 tol and of the wedge's by 0.1 tol: one move of {-5/6 tol, 0,
  // 0.5 tol} brings it to the rest gap of both. Moving off the floor alone, then off the wedge, took it 0.4 tol back
  // down toward the floor.
  const Solid wedge = BuildFlatSolid(
      {.x = 0, .y = 0, .z = 1.625 * kTolerance},
      math::Normalized({.x = -0.6, .y = 0, .z = -0.8}),
      {.linear_velocity = {.x = 0, .y = 0, .z = 0.25}},
      2
  );
  cases.push_back({
      .name = "WedgeStartPushedTowardFloor",
      .solids = {BuildFloor(1), wedge},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = kTolerance / 2},
          .velocity = {.x = -1, .y = 0, .z = -1},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // The same wedge 0.0005 m deeper into the particle's start, within the overlap tolerance: one move off both, where
  // moving off the wedge alone took the particle 0.0004 m into the floor.
  const Solid deep_wedge = BuildFlatSolid(
      {.x = 0, .y = 0, .z = 1.625 * kTolerance - 0.000625},
      math::Normalized({.x = -0.6, .y = 0, .z = -0.8}),
      {.linear_velocity = {.x = 0, .y = 0, .z = 0.25}},
      2
  );
  SolidContactInputs deep_wedge_inputs = BuildCaseInputs({
      .start_position = {.x = 0, .y = 0, .z = kTolerance / 2},
      .velocity = {.x = -1, .y = 0, .z = -1},
      .step_duration = 0.1,
      .friction = 0,
  });
  deep_wedge_inputs.start_overlap_tolerance = 0.001;
  cases.push_back(
      {.name = "WedgeStartPushedIntoFloor", .solids = {BuildFloor(1), deep_wedge}, .inputs = deep_wedge_inputs}
  );
  // Two parallel fixed flat solids with normal {0.6, 0, 0.8}, the second, reacting, tol / 4 above the first; the
  // particle starts tol / 2 above the first moving at {-2, 0, -2} m/s. The first's response leaves an approach of
  // -2^-55 m/s along the shared normal (ObliqueFlatSolidRoundingAtHit): rounding, not an approach to the second.
  const Vec3 oblique_normal = math::Normalized({.x = 0.6, .y = 0, .z = 0.8});
  Solid reacting_parallel = BuildFlatSolid(0.25 * kTolerance * oblique_normal, oblique_normal, kNoMotion, 2);
  reacting_parallel.reports_reaction = true;
  cases.push_back({
      .name = "ParallelSolidAfterStartResponse",
      .solids = {BuildFlatSolid({}, oblique_normal, kNoMotion, 1), reacting_parallel},
      .inputs = BuildCaseInputs({
          .start_position = 0.5 * kTolerance * oblique_normal,
          .velocity = {.x = -2, .y = 0, .z = -2},
          .step_duration = 0.5,
          .friction = 0,
      }),
  });
  return cases;
}

// A particle in the unit domain box (BuildUnitDomainBox) for 0.1 s with both velocities, friction 0.
inline SolidContactInputs BuildDomainStartInputs(Vec3 start_position, Vec3 velocity) {
  return BuildCaseInputs({
      .start_position = start_position,
      .velocity = velocity,
      .step_duration = 0.1,
      .friction = 0,
  });
}

// The scenes of the start rules in a corner of the domain box: each closed side closer than tol is a surface of its
// own, silent, approached by either velocity alone, or slightly inside.
inline std::vector<MoveParticleCase> BuildDomainCornerStartCases() {
  std::vector<MoveParticleCase> cases;
  const Solid domain_box = BuildUnitDomainBox(1);
  const Vec3 two_side_start = {.x = kTolerance / 4, .y = 0.5, .z = 0.3 * kTolerance};
  const Vec3 three_side_start = {.x = kTolerance / 4, .y = kTolerance / 2, .z = 0.3 * kTolerance};
  // side_bottom, not the nearest side, is approached at 0.9 tol per step, slower than the touching rule would see.
  cases.push_back({
      .name = "DomainCornerSlowApproachToSecondSide",
      .solids = {domain_box},
      .inputs = BuildDomainStartInputs(two_side_start, {.x = 0, .y = 0, .z = -0.9 * kTolerance / 0.1}),
  });
  // Only for_next_step approaches side_bottom.
  SolidContactInputs next_step_inputs = BuildDomainStartInputs(two_side_start, {});
  next_step_inputs.velocity.for_next_step = -kUnitZ;
  cases.push_back({.name = "DomainCornerNextStepIntoSecondSide", .solids = {domain_box}, .inputs = next_step_inputs});
  cases.push_back({
      .name = "DomainThreeSideCornerAtRest",
      .solids = {domain_box},
      .inputs = BuildDomainStartInputs(three_side_start, {}),
  });
  SolidContactInputs this_step_inputs = BuildDomainStartInputs(three_side_start, {.x = -1, .y = -1, .z = -1});
  this_step_inputs.velocity.for_next_step = {};
  cases.push_back({
      .name = "DomainThreeSideCornerThisStepApproaches",
      .solids = {domain_box},
      .inputs = this_step_inputs,
  });
  SolidContactInputs next_only_inputs = BuildDomainStartInputs(three_side_start, {});
  next_only_inputs.velocity.for_next_step = {.x = -1, .y = -1, .z = -1};
  cases.push_back({
      .name = "DomainThreeSideCornerNextStepApproaches",
      .solids = {domain_box},
      .inputs = next_only_inputs,
  });
  SolidContactInputs inside_inputs = BuildDomainStartInputs({.x = -0.0001, .y = -0.0002, .z = -0.0003}, {});
  inside_inputs.start_overlap_tolerance = 0.001;
  cases.push_back({.name = "DomainThreeSideCornerSlightlyInside", .solids = {domain_box}, .inputs = inside_inputs});
  // side_bottom is the nearest side before the move, side_left comes first in face order, and the two tie at tol after
  // the move.
  cases.push_back({
      .name = "DomainCornerNearestSideComesLater",
      .solids = {domain_box},
      .inputs = BuildDomainStartInputs({.x = 0.6 * kTolerance, .y = 0.5, .z = 0.1 * kTolerance}, -kUnitZ),
  });
  return cases;
}

// A fixed flat solid through the origin whose normal makes the given angle with {0, 0, 1}, toward +x: with the fixed
// floor, a wedge of 180 degrees minus that angle.
inline Solid BuildWedgeSide(double normal_angle, std::uint32_t solid_id) {
  return BuildFlatSolid({}, {.x = std::sin(normal_angle), .y = 0, .z = std::cos(normal_angle)}, kNoMotion, solid_id);
}

// The scenes of acute wedges with a particle at rest on both surfaces, and the scene that pins the start rule's range.
inline std::vector<MoveParticleCase> BuildWedgeStartCases() {
  std::vector<MoveParticleCase> cases;
  // A particle at rest on both surfaces of a wedge, at their common line, with an overlap tolerance of 2 tol.
  const SolidContactInputs on_both_inputs =
      BuildRestingStartInputs({.height = 0, .start_overlap_tolerance = 2 * kTolerance});
  cases.push_back({
      .name = "SixtyDegreeWedgeRestingOnBoth",
      .solids = {BuildFloor(1), BuildWedgeSide(2 * std::numbers::pi / 3, 2)},
      .inputs = on_both_inputs,
  });
  cases.push_back({
      .name = "ThirtyDegreeWedgeRestingOnBoth",
      .solids = {BuildFloor(1), BuildWedgeSide(5 * std::numbers::pi / 6, 2)},
      .inputs = on_both_inputs,
  });
  // Four, then five, fixed floors stacked tol / 16 apart (ids 1 to 4 or 5) and the fixed upright at x = 1 (the next
  // id): the particle leaves every floor and runs into the upright, tol / 2 away. The floors share one direction of the
  // move and are exact copies, one active surface of the upright's start response.
  for (const int floor_count : {4, 5}) {
    std::vector<Solid> floors_and_upright;
    floors_and_upright.reserve(static_cast<std::size_t>(floor_count) + 1);
    for (int floor_index = 0; floor_index < floor_count; ++floor_index) {
      floors_and_upright.push_back(
          BuildMovingFloor(kTolerance / 16 * floor_index, {}, static_cast<std::uint32_t>(floor_index + 1))
      );
    }
    floors_and_upright.push_back(BuildUprightAtXOne(static_cast<std::uint32_t>(floor_count + 1)));
    const double top_floor_height = kTolerance / 16 * (floor_count - 1);  // m
    cases.push_back({
        .name = floor_count == 4 ? "FourSilentFloorsAndUpright" : "FiveSilentFloorsAndUpright",
        .solids = floors_and_upright,
        .inputs = BuildCaseInputs({
            .start_position = {.x = 1 - kTolerance / 2, .y = 0, .z = top_floor_height + kTolerance / 2},
            .velocity = {.x = 1, .y = 0, .z = 1},
            .step_duration = 0.1,
            .friction = 0,
        }),
    });
  }
  // Four fixed floors through the origin, their surfaces coincident (ids 1 to 4), and the fixed upright at x = 1 (id
  // 5): the particle leaves the floors and runs into the upright, tol / 2 away. The upright's hit reads the four
  // floors as one active surface, an exact copy each, beside the upright.
  std::vector<Solid> coincident_floors_and_upright;
  coincident_floors_and_upright.reserve(5);
  for (int floor_index = 0; floor_index < 4; ++floor_index) {
    coincident_floors_and_upright.push_back(BuildFloor(static_cast<std::uint32_t>(floor_index + 1)));
  }
  coincident_floors_and_upright.push_back(BuildUprightAtXOne(5));
  cases.push_back({
      .name = "FourCoincidentFloorsAndUpright",
      .solids = coincident_floors_and_upright,
      .inputs = BuildCaseInputs({
          .start_position = {.x = 1 - kTolerance / 2, .y = 0, .z = kTolerance / 2},
          .velocity = {.x = 1, .y = 0, .z = 1},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // Five fixed floors stacked tol / 16 apart, which a particle tol / 2 above the top one leaves: one silent move off
  // all five, no start response, no hit.
  std::vector<Solid> five_floors;
  five_floors.reserve(5);
  for (int floor_index = 0; floor_index < 5; ++floor_index) {
    five_floors.push_back(
        BuildMovingFloor(kTolerance / 16 * floor_index, {}, static_cast<std::uint32_t>(floor_index + 1))
    );
  }
  cases.push_back({
      .name = "FiveSilentFloors",
      .solids = five_floors,
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = kTolerance / 4 + kTolerance / 2},
          .velocity = {.x = 1, .y = 0, .z = 1},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // A floor sliding along itself at 100 m/s, which the particle leaves, and the fixed upright at x = 1, which it
  // approaches at 1e-13 m/s. Before any response the threshold of the approach test is 0, so the upright is a hit at
  // the start; the floor's 100 m/s must not raise it.
  cases.push_back({
      .name = "SlidingFloorThenSlowUpright",
      .solids = {BuildMovingFloor(0, {.x = 100, .y = 0, .z = 0}, 1), BuildUprightAtXOne(2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 1 - kTolerance / 2, .y = 0, .z = kTolerance / 2},
          .velocity = {.x = 1e-13, .y = 0, .z = 1},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // The fixed floor, which a particle tol / 2 above it and moving at {1, 0, 0} m/s leaves alone, and an upright flat
  // solid at x = 1 + 2 tol leaning 0.5 tol over the floor: it is hit after tol s and sends the particle down at
  // 0.5 tol m/s. No solid responded at the start, so the move was silent; the floor, at its rest gap, is an active
  // surface of the upright's hit, and one joint response stops the push into it.
  cases.push_back({
      .name = "SilentFloorThenLeaningUpright",
      .solids =
          {
              BuildFloor(1),
              BuildFlatSolid(
                  {.x = 1 + 2 * kTolerance, .y = 0, .z = 0},
                  math::Normalized({.x = -1, .y = 0, .z = -0.5 * kTolerance}),
                  kNoMotion,
                  2
              ),
          },
      .inputs = BuildCaseInputs({
          .start_position = {.x = 1, .y = 0, .z = kTolerance / 2},
          .velocity = kUnitX,
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // Exactly at the rest gap, approaching at tol m/s for 0.1 s. The start's move does not apply at d = tol, but the
  // approached face at the rest gap responds after the start rules.
  cases.push_back({
      .name = "AtRestGapSlowApproach",
      .solids = {BuildFloor(1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = 0, .z = kTolerance},
          .velocity = {.x = 0, .y = 0, .z = -kTolerance},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // SlidingFloorThenSlowUpright with only for_next_step approaching the upright, at 1e-13 m/s: before any response
  // for_next_step's own threshold is 0 too.
  SolidContactInputs next_step_slow_inputs = BuildCaseInputs({
      .start_position = {.x = 1 - kTolerance / 2, .y = 0, .z = kTolerance / 2},
      .velocity = {.x = -1, .y = 0, .z = 1},
      .step_duration = 0.1,
      .friction = 0,
  });
  next_step_slow_inputs.velocity.for_next_step = {.x = 1e-13, .y = 0, .z = 1};
  cases.push_back({
      .name = "SlidingFloorThenSlowUprightNextStepOnly",
      .solids = {BuildMovingFloor(0, {.x = 100, .y = 0, .z = 0}, 1), BuildUprightAtXOne(2)},
      .inputs = next_step_slow_inputs,
  });
  return cases;
}

// A particle at rest at {0, 0, 0} for 0.1 s with an overlap tolerance of 2 tol, as on_both_inputs of
// BuildWedgeStartCases.
inline SolidContactInputs BuildApexAtRestInputs() {
  return BuildRestingStartInputs({.height = 0, .start_overlap_tolerance = 2 * kTolerance});
}

// The inputs of a particle at start_position with both velocities, the test contact tolerance, an overlap tolerance of
// 2 tol and friction 0.
inline SolidContactInputs BuildSeveralSurfaceInputs(Vec3 start_position, Vec3 velocity, double step_duration) {
  SolidContactInputs inputs = BuildCaseInputs({
      .start_position = start_position,
      .velocity = velocity,
      .step_duration = step_duration,
      .friction = 0,
  });
  inputs.start_overlap_tolerance = 2 * kTolerance;
  return inputs;
}

// A fixed sphere with the given centre, radius and id.
inline Solid BuildFixedSphere(Vec3 center, double radius, std::uint32_t solid_id) {
  return ToSolid(Sphere{.center = center, .radius = radius}, kNoMotion, solid_id);
}

// The scenes of a start near several surfaces at once: acute and flat wedges, corners and pockets, faces facing each
// other, and the tangent contacts of round solids (plan decision P19). Fixed solids through the origin and friction 0
// unless stated.
inline std::vector<MoveParticleCase> BuildSeveralSurfaceStartCases() {
  std::vector<MoveParticleCase> cases;
  // A wedge of 10 degrees falling on both faces at 0.981 m/s for 0.1 s.
  cases.push_back({
      .name = "TenDegreeWedgeFallingOnBoth",
      .solids = {BuildFloor(1), BuildWedgeSide(17 * std::numbers::pi / 18, 2)},
      .inputs = BuildSeveralSurfaceInputs({}, {.x = 0, .y = 0, .z = -0.981}, 0.1),
  });
  // Flat wedges at rest on both faces: 0.01 degrees, 6e-7 rad (rest point 3.33e6 tol out, within the move cap) and
  // 3e-7 rad (6.67e6 tol out, beyond the cap of 2^22 tol = 4.19e6 tol).
  cases.push_back({
      .name = "VerySharpFlatWedgeRestingOnBoth",
      .solids = {BuildFloor(1), BuildWedgeSide(std::numbers::pi - 0.01 * std::numbers::pi / 180, 2)},
      .inputs = BuildApexAtRestInputs(),
  });
  cases.push_back({
      .name = "FlatWedgeNeedingMoveWithinCap",
      .solids = {BuildFloor(1), BuildWedgeSide(std::numbers::pi - 6e-7, 2)},
      .inputs = BuildApexAtRestInputs(),
  });
  cases.push_back({
      .name = "FlatWedgeNeedingMoveBeyondCap",
      .solids = {BuildFloor(1), BuildWedgeSide(std::numbers::pi - 3e-7, 2)},
      .inputs = BuildApexAtRestInputs(),
  });
  // Three faces meeting at the origin, each pair at 60 degrees, falling on all three at 0.0981 m/s for 0.01 s.
  const double half_root_three = std::numbers::sqrt3 / 2;
  cases.push_back({
      .name = "ThreeSolidAcuteCornerFallingOnAll",
      .solids =
          {
              BuildFloor(1),
              BuildFlatSolid({}, {.x = half_root_three, .y = 0, .z = -0.5}, kNoMotion, 2),
              BuildFlatSolid({}, {.x = 0, .y = half_root_three, .z = -0.5}, kNoMotion, 3),
          },
      .inputs = BuildSeveralSurfaceInputs({}, {.x = 0, .y = 0, .z = -0.0981}, 0.01),
  });
  // The 60 degree wedge and a third face through {2 tol, 0, 0}, 1.414 tol from the start and 0.19 tol from the point
  // the wedge's pair of faces moves it to.
  cases.push_back({
      .name = "ThirdSolidNearOnlyAfterFirstMove",
      .solids =
          {
              BuildFloor(1),
              BuildWedgeSide(2 * std::numbers::pi / 3, 2),
              BuildFlatSolid(
                  {.x = 2 * kTolerance, .y = 0, .z = 0},
                  math::Normalized({.x = -1, .y = -1, .z = 0}),
                  kNoMotion,
                  3
              ),
          },
      .inputs = BuildSeveralSurfaceInputs({}, {}, 0.01),
  });
  // A corner of the unit domain box cut by a flat solid through {L tol, 0, 0} with normal -{1, 1, 1} / Sqrt(3), the
  // particle at the corner falling at 1 m/s for 0.01 s. The pocket's inscribed radius is L / (3 + Sqrt(3)) tol: 1.27
  // tol for L = 6, 0.845 tol for L = 4, too small for a point at tol from all four faces.
  for (const double cut_distance : {6.0, 4.0}) {
    cases.push_back({
        .name = cut_distance == 6 ? "DomainCornerCutBySolidFarEnough" : "DomainCornerCutBySolidTooNear",
        .solids =
            {
                BuildUnitDomainBox(1),
                BuildFlatSolid(
                    {.x = cut_distance * kTolerance, .y = 0, .z = 0},
                    math::Normalized({.x = -1, .y = -1, .z = -1}),
                    kNoMotion,
                    2
                ),
            },
        .inputs = BuildSeveralSurfaceInputs({}, {.x = 0, .y = 0, .z = -1}, 0.01),
    });
  }
  // The fixed floor and a fixed ceiling facing it, 1.5 tol above with the particle at rest midway, and 2.5 tol above
  // with the particle tol / 2 above the floor; no overlap allowed.
  SolidContactInputs narrow_inputs = BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = 0.75 * kTolerance}, {}, 0.01);
  narrow_inputs.start_overlap_tolerance = 0;
  cases.push_back({
      .name = "ParallelFacesLessThanTwoTolerancesApart",
      .solids = {BuildFloor(1), BuildFlatSolid({.x = 0, .y = 0, .z = 1.5 * kTolerance}, -kUnitZ, kNoMotion, 2)},
      .inputs = narrow_inputs,
  });
  SolidContactInputs wide_inputs = BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = 0.5 * kTolerance}, {}, 0.01);
  wide_inputs.start_overlap_tolerance = 0;
  cases.push_back({
      .name = "ParallelFacesMoreThanTwoTolerancesApart",
      .solids = {BuildFloor(1), BuildFlatSolid({.x = 0, .y = 0, .z = 2.5 * kTolerance}, -kUnitZ, kNoMotion, 2)},
      .inputs = wide_inputs,
  });
  // A ball of radius 0.5 resting on the fixed floor at the origin, the particle at rest in the cusp, 0.25 tol from
  // both, for 0.01 s. Its rest point is {2 Sqrt(0.5 tol), 0, tol}.
  cases.push_back({
      .name = "BallRestingOnFloorStartInCusp",
      .solids = {BuildFloor(1), BuildFixedSphere({.x = 0, .y = 0, .z = 0.5}, 0.5, 2)},
      .inputs = BuildSeveralSurfaceInputs({.x = std::sqrt(0.5 * kTolerance), .y = 0, .z = kTolerance / 4}, {}, 0.01),
  });
  // The same ball, the particle at rest 0.014 tol short of the floor and 0.0016 tol beyond the ball's rest gap: the
  // floor's move alone, 0.014 tol, is shorter than tol / 64 but leaves the ball 0.0124 tol short.
  const double short_height = 0.986 * kTolerance;  // m
  const double short_x = std::sqrt(
      (0.5 + 1.0016 * kTolerance) * (0.5 + 1.0016 * kTolerance) - (0.5 - short_height) * (0.5 - short_height)
  );  // m
  cases.push_back({
      .name = "BallRestingOnFloorStartJustShortOfFloor",
      .solids = {BuildFloor(1), BuildFixedSphere({.x = 0, .y = 0, .z = 0.5}, 0.5, 2)},
      .inputs = BuildSeveralSurfaceInputs({.x = short_x, .y = 0, .z = short_height}, {}, 0.01),
  });
  // A ball of radius 1024 m resting on the floor (R / tol = 2^40, the bound kMaxRestGapIterations is measured at; its
  // top lies at 2^41 tol, beyond MoveParticle's premise on coordinates, so MoveParticles names it as bad input while
  // MoveParticle runs it), and the particle at rest halfway up the gap, f times the rest point's distance x* = 2
  // Sqrt(1024 tol) = 0.001953125 m from the contact: f = 0.1, 0.01, and 0.0005, within 1.8e-8 R of the contact.
  const double deep_rest_distance = 2 * std::sqrt(1024 * kTolerance);  // m
  for (const double share : {0.1, 0.01, 0.0005}) {
    const double start_x = share * deep_rest_distance;  // m
    std::string_view name = "BallRestingOnFloorDeepStartNearContact";
    if (share == 0.1) {
      name = "BallRestingOnFloorDeepStartTenth";
    } else if (share == 0.01) {
      name = "BallRestingOnFloorDeepStartHundredth";
    }
    cases.push_back({
        .name = name,
        .solids = {BuildFloor(1), BuildFixedSphere({.x = 0, .y = 0, .z = 1024}, 1024, 2)},
        .inputs = BuildSeveralSurfaceInputs({.x = start_x, .y = 0, .z = start_x * start_x / 4096}, {}, 0.01),
    });
  }
  // The ball of radius 1024 m with the start at 0.1 x*, and an upright flat solid at x = 1.5 x* facing the cusp: the
  // first iterate, shortened to the move cap, crosses it.
  const double deep_start_x = 0.1 * deep_rest_distance;  // m
  const double deep_gap = 1024 - std::sqrt(1024 * 1024 - deep_start_x * deep_start_x);  // m
  cases.push_back({
      .name = "BallRestingOnFloorDeepStartBesideWall",
      .solids =
          {
              BuildFloor(1),
              BuildFixedSphere({.x = 0, .y = 0, .z = 1024}, 1024, 2),
              BuildFlatSolid({.x = 1.5 * deep_rest_distance, .y = 0, .z = 0}, -kUnitX, kNoMotion, 3),
          },
      .inputs = BuildSeveralSurfaceInputs({.x = deep_start_x, .y = 0, .z = deep_gap / 2}, {}, 0.01),
  });
  // A ball of radius 2^10 tol resting in the corner of the unit domain box, touching its bottom, left and front
  // sides, and the particle at rest halfway up the ball-floor cusp, 0.01 x* from the contact toward side_left.
  const double small_radius = 1024 * kTolerance;  // m
  const double small_start_x = 0.01 * 2 * std::sqrt(small_radius * kTolerance);  // m
  const double small_gap = small_radius - std::sqrt(small_radius * small_radius - small_start_x * small_start_x);  // m
  cases.push_back({
      .name = "SmallBallInDomainCornerCuspStart",
      .solids =
          {
              BuildUnitDomainBox(1),
              BuildFixedSphere({.x = small_radius, .y = small_radius, .z = small_radius}, small_radius, 2),
          },
      .inputs = BuildSeveralSurfaceInputs(
          {.x = small_radius - small_start_x, .y = small_radius, .z = small_gap / 2},
          {},
          0.01
      ),
  });
  // The ball of radius 1024 m with the start at 0.01 x*, and an upright flat solid 1.01 tol beyond the rest point,
  // facing the cusp: the rest point fits, but lies within the rounding window of the ball's distance (ulp(1024 m) over
  // the cusp's angle, about 128 tol here), where no subset of the three linearised surfaces meets every target.
  const double window_start_x = 0.01 * deep_rest_distance;  // m
  const double window_gap = 1024 - std::sqrt(1024 * 1024 - window_start_x * window_start_x);  // m
  cases.push_back({
      .name = "BallRestingOnFloorDeepStartWallNearRestPoint",
      .solids =
          {
              BuildFloor(1),
              BuildFixedSphere({.x = 0, .y = 0, .z = 1024}, 1024, 2),
              BuildFlatSolid({.x = deep_rest_distance + 1.01 * kTolerance, .y = 0, .z = 0}, -kUnitX, kNoMotion, 3),
          },
      .inputs = BuildSeveralSurfaceInputs({.x = window_start_x, .y = 0, .z = window_gap / 2}, {}, 0.01),
  });
  // Sixteen, then seventeen, fixed floors lying on each other through the origin (ids 1 to 16 or 17), the particle at
  // rest tol / 2 above them: one direction, but one move tracks at most kMaxTrackedSurfaces surfaces.
  for (const int floor_count : {16, 17}) {
    std::vector<Solid> coincident_floors;
    coincident_floors.reserve(static_cast<std::size_t>(floor_count));
    for (int floor_index = 0; floor_index < floor_count; ++floor_index) {
      coincident_floors.push_back(BuildFloor(static_cast<std::uint32_t>(floor_index + 1)));
    }
    cases.push_back({
        .name = floor_count == 16 ? "SixteenCoincidentFloors" : "SeventeenCoincidentFloors",
        .solids = coincident_floors,
        .inputs = BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = kTolerance / 2}, {}, 0.01),
    });
  }
  // The unit domain box, a ground lying exactly on side_bottom and an upright lying exactly on side_left, the particle
  // tol / 2 from all of them, falling at 0.1 m/s for 0.01 s.
  cases.push_back({
      .name = "DomainCornerWithCoincidentGroundAndWall",
      .solids = {BuildUnitDomainBox(1), BuildFloor(2), BuildFlatSolid({}, kUnitX, kNoMotion, 3)},
      .inputs = BuildSeveralSurfaceInputs(
          {.x = kTolerance / 2, .y = kTolerance / 2, .z = kTolerance / 2},
          {.x = 0, .y = 0, .z = -0.1},
          0.01
      ),
  });
  // The same corner with a ground and an upright whose normals are tilted by 2^-44 rad, rounding-sized: still three
  // directions, since normals within kSameNormalAxisDifference on every axis share one target.
  cases.push_back({
      .name = "DomainCornerWithNearlyCoincidentGroundAndWall",
      .solids =
          {
              BuildUnitDomainBox(1),
              BuildFlatSolid({}, math::Normalized({.x = 0x1p-44, .y = 0, .z = 1}), kNoMotion, 2),
              BuildFlatSolid({}, math::Normalized({.x = 1, .y = 0x1p-44, .z = 0}), kNoMotion, 3),
          },
      .inputs = BuildSeveralSurfaceInputs(
          {.x = kTolerance / 2, .y = kTolerance / 2, .z = kTolerance / 2},
          {.x = 0, .y = 0, .z = -0.1},
          0.01
      ),
  });
  // The ball of radius 0.5 m resting on the floor, the particle halfway to the rest point and halfway up the gap,
  // moving at 1 m/s straight into the ball along minus its normal at the start, away from the floor.
  const double half_rest_x = 0.5 * 2 * std::sqrt(0.5 * kTolerance);  // m
  const double half_rest_gap = 0.5 - std::sqrt(0.25 - half_rest_x * half_rest_x);  // m
  const Vec3 into_ball_start = {.x = half_rest_x, .y = 0, .z = half_rest_gap / 2};
  const Solid half_meter_ball = BuildFixedSphere({.x = 0, .y = 0, .z = 0.5}, 0.5, 2);
  const Vec3 ball_normal =
      ComputeDistanceToSolid(half_meter_ball, half_meter_ball.pose_at_step_start, into_ball_start).normal_out_of_solid;
  cases.push_back({
      .name = "CuspStartIntoBallOnly",
      .solids = {BuildFloor(1), half_meter_ball},
      .inputs = BuildSeveralSurfaceInputs(into_ball_start, -ball_normal, 0.01),
  });
  // Two balls of radius 0.5 touching at {0.5, 0, 0}, the particle at rest in their neck, half way to the ring
  // Sqrt(tol (1 + tol)) from the contact where both are at tol.
  const double neck_ring_radius = std::sqrt(kTolerance * (1 + kTolerance));  // m
  cases.push_back({
      .name = "TwoBallsTouchingStartInNeck",
      .solids = {BuildFixedSphere({}, 0.5, 1), BuildFixedSphere(kUnitX, 0.5, 2)},
      .inputs = BuildSeveralSurfaceInputs({.x = 0.5, .y = neck_ring_radius / 2, .z = 0}, {}, 0.01),
  });
  // A plate through the origin with normal {0, 0, 1} and half thickness 0.01, and a ball of radius 0.5 centred at
  // {0, 0, 0.26}: they cross on the ring of radius Sqrt(0.1875) at z = 0.01, their normals 120 degrees apart. The
  // particle is 0.5 tol out of the ring along the bisector, falling at 0.0981 m/s for 0.01 s.
  const Vec3 ring_point = {.x = std::sqrt(0.1875), .y = 0, .z = 0.01};
  const Vec3 ring_bisector = {.x = half_root_three, .y = 0, .z = 0.5};
  cases.push_back({
      .name = "PlateAndSphereRing",
      .solids =
          {
              ToSolid(Plate{.center = {}, .normal = kUnitZ, .half_thickness = 0.01}, kNoMotion, 1),
              BuildFixedSphere({.x = 0, .y = 0, .z = 0.26}, 0.5, 2),
          },
      .inputs = BuildSeveralSurfaceInputs(
          ring_point + 0.5 * kTolerance * ring_bisector,
          {.x = 0, .y = 0, .z = -0.0981},
          0.01
      ),
  });
  // ParallelSolidAfterStartResponse with for_this_step at rest and for_next_step {-2, 0, -2} m/s: the response to a
  // for_this_step at rest has an allowance of 0, while it leaves -2^-55 m/s in for_next_step.
  const Vec3 oblique_normal = math::Normalized({.x = 0.6, .y = 0, .z = 0.8});
  Solid reacting_parallel = BuildFlatSolid(0.25 * kTolerance * oblique_normal, oblique_normal, kNoMotion, 2);
  reacting_parallel.reports_reaction = true;
  SolidContactInputs next_step_inputs = BuildCaseInputs({
      .start_position = 0.5 * kTolerance * oblique_normal,
      .velocity = {},
      .step_duration = 0.5,
      .friction = 0,
  });
  next_step_inputs.velocity.for_next_step = {.x = -2, .y = 0, .z = -2};
  cases.push_back({
      .name = "ParallelSolidAfterStartResponseNextStepMoving",
      .solids = {BuildFlatSolid({}, oblique_normal, kNoMotion, 1), reacting_parallel},
      .inputs = next_step_inputs,
  });
  // The 60 degree wedge with the particle at rest tol / 2 from the floor and 1.2 tol from the other face: moving off
  // the floor alone would leave that face 0.95 tol away, so the move goes on to the rest gap of both.
  const double second_face_x = (1.2 + 0.25) * 2 / std::numbers::sqrt3 * kTolerance;  // m
  cases.push_back({
      .name = "SecondFaceNearOnlyAfterFloorMove",
      .solids = {BuildFloor(1), BuildWedgeSide(2 * std::numbers::pi / 3, 2)},
      .inputs = BuildSeveralSurfaceInputs({.x = second_face_x, .y = 0, .z = kTolerance / 2}, {}, 0.01),
  });
  // The fixed floor and a plate 1200 tol thick inside it, its middle plane at z = -0.00025, with the particle at rest
  // 0.0005 m inside the floor, below the plate: the floor's own move to its rest gap would pass through the plate.
  SolidContactInputs below_plate_inputs = BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = -0.0005}, {}, 0.01);
  below_plate_inputs.start_overlap_tolerance = 0.001;
  cases.push_back({
      .name = "StartInsideFloorBelowPlate",
      .solids =
          {
              BuildFloor(1),
              ToSolid(
                  Plate{
                      .center = {.x = 0, .y = 0, .z = -0.00025},
                      .normal = kUnitZ,
                      .half_thickness = 600 * kTolerance,
                  },
                  kNoMotion,
                  2
              ),
          },
      .inputs = below_plate_inputs,
  });
  // A wedge of 0.03 degrees, whose pair of faces would move a start at the apex 3819.7 tol out along the bisector b,
  // and a plate across it with normal b, half thickness 600 tol, centred 700 tol along b: its faces 100 and 1300 tol
  // out. The particle at rest at the apex for 0.1 s.
  const double sharp_angle = 0.03 * std::numbers::pi / 180;  // rad
  const Vec3 sharp_bisector = {.x = std::cos(sharp_angle / 2), .y = 0, .z = std::sin(sharp_angle / 2)};
  cases.push_back({
      .name = "PlateAcrossSharpWedge",
      .solids =
          {
              BuildFloor(1),
              BuildWedgeSide(std::numbers::pi - sharp_angle, 2),
              ToSolid(
                  Plate{
                      .center = 700 * kTolerance * sharp_bisector,
                      .normal = sharp_bisector,
                      .half_thickness = 600 * kTolerance,
                  },
                  kNoMotion,
                  3
              ),
          },
      .inputs = BuildApexAtRestInputs(),
  });
  return cases;
}

// The side of a wedge of the given angle with the fixed floor, as BuildWedgeSide(pi - wedge_angle), moving at
// normal_speed along its own normal (into the wedge when positive).
//
// clang-tidy: an angle in rad and a speed in m/s; the swappable-parameters check flags the two doubles.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): see the comment above.
inline Solid BuildMovingWedgeSide(double wedge_angle, double normal_speed, std::uint32_t solid_id) {
  const double normal_angle = std::numbers::pi - wedge_angle;  // rad
  const Vec3 normal = {.x = std::sin(normal_angle), .y = 0, .z = std::cos(normal_angle)};
  return BuildFlatSolid({}, normal, {.linear_velocity = normal_speed * normal}, solid_id);
}

// The scene of a closing wedge with a particle at rest at its rest point {tol / tan(a / 2), 0, tol}: the floor moving
// at floor_speed along {0, 0, 1} and the side at side_speed along its own normal, both into the wedge, for 0.01 s,
// friction 0, an overlap tolerance of 2 tol.
inline MoveParticleCase
BuildClosingWedgeCase(std::string_view name, double wedge_angle, double floor_speed, double side_speed) {
  const Vec3 rest_point = {.x = kTolerance / std::tan(wedge_angle / 2), .y = 0, .z = kTolerance};
  return {
      .name = name,
      .solids =
          {
              BuildMovingFloor(0, {.x = 0, .y = 0, .z = floor_speed}, 1),
              BuildMovingWedgeSide(wedge_angle, side_speed, 2),
          },
      .inputs = BuildSeveralSurfaceInputs(rest_point, {}, 0.01),
  };
}

// The scenes of a particle moving into an acute wedge or a corner, then sliding on along it or stopped at its apex.
inline std::vector<MoveParticleCase> BuildSlideIntoWedgeCases() {
  std::vector<MoveParticleCase> cases;
  // The floor and a face through {1, 0, 0} with normal Normalized({-1, 0, -1}): a 45 degree wedge whose apex is at
  // x = 1. The particle rests on the floor and moves into the apex at 2 m/s for 0.5 s.
  const Solid wedge_face = BuildFlatSolid(kUnitX, math::Normalized({.x = -1, .y = 0, .z = -1}), kNoMotion, 2);
  const SolidContactInputs into_apex_inputs = BuildCaseInputs({
      .start_position = {.x = 0.5, .y = 0, .z = kTolerance},
      .velocity = {.x = 2, .y = 0, .z = 0},
      .step_duration = 0.5,
      .friction = 0,
  });
  cases.push_back({.name = "AcuteWedge", .solids = {BuildFloor(1), wedge_face}, .inputs = into_apex_inputs});
  Solid reacting_floor = BuildFloor(1);
  reacting_floor.reports_reaction = true;
  cases.push_back({
      .name = "AcuteWedgeOnReactingFloor",
      .solids = {reacting_floor, wedge_face},
      .inputs = into_apex_inputs,
  });
  // The floor and the upright at x = 1: a right-angle corner, entered at {2, 1, -4} m/s from {0.5, 0, 0.2}.
  cases.push_back({
      .name = "RightAngleCorner",
      .solids = {BuildFloor(1), BuildUprightAtXOne(2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5, .y = 0, .z = 0.2},
          .velocity = {.x = 2, .y = 1, .z = -4},
          .step_duration = 0.5,
          .friction = 0,
      }),
  });
  // A reacting floor without friction (friction_scale 0) and an upright at x = 1 sliding down along itself at 1 m/s,
  // not reacting; friction 0.5. The upright's slip leaves the particle falling at 0.5 m/s toward the floor.
  Solid frictionless_reacting_floor = BuildFloor(1);
  frictionless_reacting_floor.reports_reaction = true;
  frictionless_reacting_floor.friction_scale = 0;
  cases.push_back({
      .name = "SlideRemovalOnReactingFloor",
      .solids =
          {
              frictionless_reacting_floor,
              BuildFlatSolid(kUnitX, -kUnitX, {.linear_velocity = {.x = 0, .y = 0, .z = -1}}, 2),
          },
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5, .y = 0, .z = 0.2},
          .velocity = {.x = 1, .y = 0, .z = -2},
          .step_duration = 1,
          .friction = 0.5,
      }),
  });
  // A flat solid through the origin with normal {0.6, 0, 0.8} and an upright x <= -0.2 facing +x: an obtuse corner of
  // 126.87 degrees. The particle hits the first, slides down along it at {-0.32, 0, 0.24} m/s, hits the upright and
  // rises along it.
  cases.push_back({
      .name = "ObliqueCornerResidue",
      .solids =
          {
              BuildFlatSolid({}, math::Normalized({.x = 0.6, .y = 0, .z = 0.8}), kNoMotion, 1),
              BuildFlatSolid({.x = -0.2, .y = 0, .z = 0}, kUnitX, kNoMotion, 2),
          },
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.3, .y = 0, .z = 0.4},
          .velocity = {.x = -2, .y = 0, .z = -2},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // The 60 degree wedge, the particle at its apex moving at {-1, 1, -1} m/s for 0.01 s: into both faces and along the
  // common line.
  cases.push_back({
      .name = "GrooveSlideAlongEdge",
      .solids = {BuildFloor(1), BuildWedgeSide(2 * std::numbers::pi / 3, 2)},
      .inputs = BuildSeveralSurfaceInputs({}, {.x = -1, .y = 1, .z = -1}, 0.01),
  });
  // V-grooves through the origin opening upward, each face at half_angle from the vertical: 30 and 10 degrees wide.
  // The particle starts at rest 0.05 m up face 1, at its rest gap, for steps of 0.01 s.
  const std::array<std::pair<std::string_view, double>, 2> grooves = {
      std::pair<std::string_view, double>{"VGroove", 15 * std::numbers::pi / 180},
      std::pair<std::string_view, double>{"TenDegreeVGroove", 5 * std::numbers::pi / 180},
  };
  for (const auto& [name, half_angle] : grooves) {
    const Vec3 first_normal = {.x = -std::cos(half_angle), .y = 0, .z = std::sin(half_angle)};
    const Vec3 second_normal = {.x = std::cos(half_angle), .y = 0, .z = std::sin(half_angle)};
    const Vec3 up_first_face = {.x = std::sin(half_angle), .y = 0, .z = std::cos(half_angle)};
    cases.push_back({
        .name = name,
        .solids = {BuildFlatSolid({}, first_normal, kNoMotion, 1), BuildFlatSolid({}, second_normal, kNoMotion, 2)},
        .inputs = BuildSeveralSurfaceInputs(0.05 * up_first_face + kTolerance * first_normal, {}, 0.01),
    });
  }
  // Wedges of 2e-6 rad (above the velocity pair limit) and 1e-7 rad (a sliver, below it), the particle at the rest
  // point moving at {-2, 1, 0} m/s for 0.01 s: into the apex and along the common line. The sliver's floor reacts.
  cases.push_back({
      .name = "NarrowWedgeAtRestPoint",
      .solids = {BuildFloor(1), BuildWedgeSide(std::numbers::pi - 2e-6, 2)},
      .inputs = BuildSeveralSurfaceInputs(
          {.x = kTolerance / std::tan(1e-6), .y = 0, .z = kTolerance},
          {.x = -2, .y = 1, .z = 0},
          0.01
      ),
  });
  cases.push_back({
      .name = "SliverStop",
      .solids = {reacting_floor, BuildWedgeSide(std::numbers::pi - 1e-7, 2)},
      .inputs = BuildSeveralSurfaceInputs(
          {.x = kTolerance / std::tan(5e-8), .y = 0, .z = kTolerance},
          {.x = -2, .y = 1, .z = 0},
          0.01
      ),
  });
  // Three faces meeting at the origin, each pair at 60 degrees, entered at {-1, -1, -1} m/s with friction 0.5.
  const double half_root_three = std::numbers::sqrt3 / 2;
  SolidContactInputs into_corner_inputs = BuildSeveralSurfaceInputs({}, {.x = -1, .y = -1, .z = -1}, 0.01);
  into_corner_inputs.friction = 0.5;
  cases.push_back({
      .name = "ThreeSolidAcuteCornerInto",
      .solids =
          {
              BuildFloor(1),
              BuildFlatSolid({}, {.x = half_root_three, .y = 0, .z = -0.5}, kNoMotion, 2),
              BuildFlatSolid({}, {.x = 0, .y = half_root_three, .z = -0.5}, kNoMotion, 3),
          },
      .inputs = into_corner_inputs,
  });
  // The floor and a ceiling through {0, 0, 0.1} with normal {-0.6, 0, -0.8}: a wedge of 36.87 degrees whose apex is
  // at x = 0.1333. The particle starts tol / 2 inside the floor moving at {5, 0, 1} m/s for 0.2 s: it leaves the floor,
  // hits the ceiling 0.021 m above it and slides down the ceiling into the apex.
  SolidContactInputs far_above_inputs =
      BuildSeveralSurfaceInputs({.x = 0, .y = 0, .z = -kTolerance / 2}, {.x = 5, .y = 0, .z = 1}, 0.2);
  far_above_inputs.start_overlap_tolerance = kTolerance;
  cases.push_back({
      .name = "CeilingHitFarAboveTouchedFloor",
      .solids =
          {
              BuildFloor(1),
              BuildFlatSolid({.x = 0, .y = 0, .z = 0.1}, {.x = -0.6, .y = 0, .z = -0.8}, kNoMotion, 2),
          },
      .inputs = far_above_inputs,
  });
  // The 60 degree side (id 1, index 0) and the floor (id 2, index 1), the floor reacting, a start 0.999 tol from both
  // moving at {-1, 0, -2} m/s for 0.01 s: it leaves the side and falls on the floor, whose response turns it into the
  // side. The side reacts too, or not.
  Solid reacting_side = BuildWedgeSide(2 * std::numbers::pi / 3, 1);
  reacting_side.reports_reaction = true;
  Solid reacting_second_floor = BuildFloor(2);
  reacting_second_floor.reports_reaction = true;
  const SolidContactInputs corner_push_inputs = BuildSeveralSurfaceInputs(
      0.999 * Vec3{.x = std::numbers::sqrt3 * kTolerance, .y = 0, .z = kTolerance},
      {.x = -1, .y = 0, .z = -2},
      0.01
  );
  cases.push_back({
      .name = "CornerPushOnSecondReactingSolid",
      .solids = {reacting_side, reacting_second_floor},
      .inputs = corner_push_inputs,
  });
  cases.push_back({
      .name = "CornerPushOnNonReactingSide",
      .solids = {BuildWedgeSide(2 * std::numbers::pi / 3, 1), reacting_second_floor},
      .inputs = corner_push_inputs,
  });
  // A flat solid through the origin spinning about its oblique normal n at 100 m/s at the particle (id 1), and a fixed
  // upright square to it (id 2), the particle tol / 2 from both, moving along the floor into the upright at 0.05 m/s
  // with an approach to the floor of 7.7e-15 m/s, below the rounding of the spin's omega x r: the floor is moved
  // silently, only the upright responds. The values are draw 20 of a search over such scenes, written to 17 digits.
  // NOLINTBEGIN(modernize-use-std-numbers): measured values, not constants.
  const Vec3 spin_normal = {.x = 0.54771969589076452, .y = 0.36362434797457288, .z = 0.75351208901609223};
  const Vec3 spin = {.x = 62.737753218534053, .y = 41.650820261955403, .z = 86.309942553721427};  // rad/s
  const Solid spinning_floor = ToSolid(
      FlatSolid{.point_on_surface = {}, .normal_out_of_solid = spin_normal},
      {.linear_velocity = {}, .angular_velocity = spin},
      1
  );
  const Solid square_upright = BuildFlatSolid(
      {.x = 0.54798547944392606, .y = -0.67578075230676038, .z = -0.072211322220003038},
      {.x = -0.51702170497473798, .y = 0.85517793520497221, .z = -0.036868085434085374},
      kNoMotion,
      2
  );
  cases.push_back({
      .name = "SpinningFloorMovedSilently",
      .solids = {spinning_floor, square_upright},
      .inputs = BuildSeveralSurfaceInputs(
          {.x = 0.54798547920316909, .y = -0.67578075190853715, .z = -0.072211322237171083},
          {.x = 0.038494232238438127, .y = -0.034491592187039413, .z = -0.011336336311113944},
          0.01
      ),
  });
  // NOLINTEND(modernize-use-std-numbers)
  return cases;
}

// The scenes of round solids touching others: a particle sliding into the cusp under a ball resting on a floor, into
// the neck of two touching balls, and into the ring where a plate and a ball cross. Friction 0.
inline std::vector<MoveParticleCase> BuildSlideIntoCuspCases() {
  std::vector<MoveParticleCase> cases;
  const Solid half_meter_ball = BuildFixedSphere({.x = 0, .y = 0, .z = 0.5}, 0.5, 2);
  cases.push_back({
      .name = "BallRestingOnFloorSlidingIntoCusp",
      .solids = {BuildFloor(1), half_meter_ball},
      .inputs = BuildSeveralSurfaceInputs({.x = 0.05, .y = 0, .z = kTolerance}, {.x = -1, .y = 0, .z = 0}, 0.1),
  });
  cases.push_back({
      .name = "BallRestingOnFloorRollingIntoCusp",
      .solids = {BuildFloor(1), half_meter_ball},
      .inputs = BuildSeveralSurfaceInputs({.x = 0.05, .y = 0, .z = kTolerance}, {.x = -0.5, .y = 0, .z = 0}, 0.01),
  });
  cases.push_back({
      .name = "TwoBallsTouchingSlidingIntoNeck",
      .solids = {BuildFixedSphere({}, 0.5, 1), BuildFixedSphere(kUnitX, 0.5, 2)},
      .inputs = BuildSeveralSurfaceInputs({.x = 0.5, .y = 0.05, .z = 0}, {.x = 0, .y = -1, .z = 0}, 0.1),
  });
  cases.push_back({
      .name = "PlateAndSphereSlidingIn",
      .solids =
          {
              ToSolid(Plate{.center = {}, .normal = kUnitZ, .half_thickness = 0.01}, kNoMotion, 1),
              BuildFixedSphere({.x = 0, .y = 0, .z = 0.26}, 0.5, 2),
          },
      .inputs = BuildSeveralSurfaceInputs({.x = 0.5, .y = 0, .z = 0.01 + kTolerance}, {.x = -1, .y = 0, .z = 0}, 0.1),
  });
  return cases;
}

// The scenes of moving solids closing a wedge on a particle at rest at its rest point: carried out along the apex, or
// squeezed when the escape would be faster than kMaxSqueezeOutSpeedFactor times the speeds involved; and a ceiling
// closing on a floor.
inline std::vector<MoveParticleCase> BuildClosingWedgeCases() {
  std::vector<MoveParticleCase> cases;
  const double degree = std::numbers::pi / 180;  // rad
  // The floor rising at 1 m/s under the fixed 60 degree side, and both faces of that wedge moving into it at 0.5 m/s
  // along their own normals, for 0.1 s.
  MoveParticleCase rising_floor = BuildClosingWedgeCase("FloorRisingUnderSixtyDegreeWall", 60 * degree, 1, 0);
  rising_floor.inputs.step_duration = 0.1;
  cases.push_back(rising_floor);
  MoveParticleCase both_closing = BuildClosingWedgeCase("BothFacesClosingAtSixtyDegrees", 60 * degree, 0.5, 0.5);
  both_closing.inputs.step_duration = 0.1;
  cases.push_back(both_closing);
  // The floor rising at 1 m/s under a fixed side at 1 and 0.8 degrees.
  cases.push_back(BuildClosingWedgeCase("OneFaceClosingAtOneDegree", 1 * degree, 1, 0));
  cases.push_back(BuildClosingWedgeCase("OneFaceClosingAtPointEightDegrees", 0.8 * degree, 1, 0));
  // Both faces closing at 1 m/s along their own normals at 2 and 1.6 degrees.
  cases.push_back(BuildClosingWedgeCase("TwoFacesClosingAtTwoDegrees", 2 * degree, 1, 1));
  cases.push_back(BuildClosingWedgeCase("TwoFacesClosingAtOnePointSixDegrees", 1.6 * degree, 1, 1));
  // A wedge of 1e-5 rad whose side closes at 1 m/s, the particle at rest 4 tol / tan(1e-5) out along the floor, where
  // the side is 3 tol away, for 0.1 s.
  const double grazing_angle = 1e-5;  // rad
  cases.push_back({
      .name = "GrazingWedgeClosingFast",
      .solids = {BuildFloor(1), BuildMovingWedgeSide(grazing_angle, 1, 2)},
      .inputs =
          BuildSeveralSurfaceInputs({.x = 4 * kTolerance / std::tan(grazing_angle), .y = 0, .z = kTolerance}, {}, 0.1),
  });
  // The floor rising at 500 m/s under the fixed 60 degree side, for 1 s: the escape {500 / tan 60, 0, 500} is within
  // the speed cap, but the change's allowance, 2^-46 times the 1290 m/s it involves, times the time left is above
  // tol / 64.
  MoveParticleCase fast_rising_floor =
      BuildClosingWedgeCase("FloorRisingFastUnderSixtyDegreeWall", 60 * degree, 500, 0);
  fast_rising_floor.inputs.step_duration = 1;
  cases.push_back(fast_rising_floor);
  // A fixed reacting floor (id 1) and a floor rising at 0.25 m/s through it (id 2), not reacting, their surfaces
  // coincident at the start, under the fixed 60 degree side (id 3); the particle at rest half way to the apex rest
  // point, tol / 2 from all three, for 0.01 s. The two floors share one direction of the joint response, whose push the
  // rising floor makes.
  Solid fixed_reacting_floor = BuildFloor(1);
  fixed_reacting_floor.reports_reaction = true;
  cases.push_back({
      .name = "FloorRisingThroughFixedFloorUnderSixtyDegreeWall",
      .solids =
          {
              fixed_reacting_floor,
              BuildMovingFloor(0, {.x = 0, .y = 0, .z = 0.25}, 2),
              BuildMovingWedgeSide(60 * degree, 0, 3),
          },
      .inputs =
          BuildSeveralSurfaceInputs({.x = std::numbers::sqrt3 / 2 * kTolerance, .y = 0, .z = kTolerance / 2}, {}, 0.01),
  });
  // The fixed floor and a ceiling 0.1 m above it coming down at 1 m/s on a particle at rest half way, for 0.1 s.
  cases.push_back({
      .name = "ClosingCeiling",
      .solids =
          {
              BuildFloor(1),
              BuildFlatSolid({.x = 0, .y = 0, .z = 0.1}, -kUnitZ, {.linear_velocity = {.x = 0, .y = 0, .z = -1}}, 2),
          },
      .inputs = BuildRestingStartInputs({.height = 0.05, .start_overlap_tolerance = 0}),
  });
  return cases;
}

// Slides in edges and corners: an edge of the domain box against the same edge of two flat solids,
// with friction; a domain corner with two chamfers across it (five directions); moving three-face corners with a
// coincident fourth face; an obtuse wedge listed in both orders; two coincident reacting solids whose normals differ
// by rounding; a slow slide into a sliver wedge.
inline std::vector<MoveParticleCase> BuildEdgeAndCornerSlideCases() {
  std::vector<MoveParticleCase> cases;
  // The unit domain box, and the flat solids x = 0, y = 0 and z = 0 in the same place: the particle at
  // {0.005, 0.5, 0.005} moving at {-1, 0.5, -1} m/s for 0.02 s meets the x and z sides at the same time. Friction 0.5.
  const SolidContactInputs edge_inputs = BuildCaseInputs({
      .start_position = {.x = 0.005, .y = 0.5, .z = 0.005},
      .velocity = {.x = -1, .y = 0.5, .z = -1},
      .step_duration = 0.02,
      .friction = 0.5,
  });
  cases.push_back({.name = "DomainBoxEdgeWithFriction", .solids = {BuildUnitDomainBox(1)}, .inputs = edge_inputs});
  cases.push_back({
      .name = "FlatSolidsEdgeWithFriction",
      .solids =
          {
              BuildFlatSolid({}, kUnitX, kNoMotion, 1),
              BuildFlatSolid({}, kUnitY, kNoMotion, 2),
              BuildFlatSolid({}, kUnitZ, kNoMotion, 3),
          },
      .inputs = edge_inputs,
  });
  // The unit domain box and two flat chamfers with normals Normalized({1, 1, 0}) and Normalized({0, 1, 1}), each
  // 0.999 tol from the corner's rest point {tol, tol, tol}; the particle at {0.2, 0.2, 0.2} moving at {-1, -1, -1} m/s
  // for 1 s: the two chamfers, then the box's sides, five directions at once.
  const Vec3 corner_rest_point = {.x = kTolerance, .y = kTolerance, .z = kTolerance};
  const Vec3 first_chamfer_normal = math::Normalized({.x = 1, .y = 1, .z = 0});
  const Vec3 second_chamfer_normal = math::Normalized({.x = 0, .y = 1, .z = 1});
  cases.push_back({
      .name = "DomainCornerWithTwoChamfers",
      .solids =
          {
              BuildUnitDomainBox(1),
              BuildFlatSolid(
                  corner_rest_point - 0.999 * kTolerance * first_chamfer_normal,
                  first_chamfer_normal,
                  kNoMotion,
                  2
              ),
              BuildFlatSolid(
                  corner_rest_point - 0.999 * kTolerance * second_chamfer_normal,
                  second_chamfer_normal,
                  kNoMotion,
                  3
              ),
          },
      .inputs = BuildSeveralSurfaceInputs({.x = 0.2, .y = 0.2, .z = 0.2}, {.x = -1, .y = -1, .z = -1}, 1),
  });
  // Moving three-face corners through the origin with a fourth solid coincident with the first, last in array order:
  // draws 3053 and 115326 of a generated search, written to 17 digits, each normal normalized again as the search
  // built it. In the first the escape is much faster than the velocity the response received, so its allowance must
  // count the change; in the second the same for for_next_step's allowance.
  // NOLINTBEGIN(modernize-use-std-numbers): measured values, not constants.
  const Vec3 fast_normal =
      math::Normalized({.x = -0.53355339409363756, .y = 0.78721619356863037, .z = 0.30921099630264787});
  const Vec3 fast_velocity = {.x = -0.77483383356200208, .y = 0.30335032059547618, .z = -0.54292335287386662};
  const Vec3 third_normal =
      math::Normalized({.x = 0.40280233806491872, .y = -0.13285702927717774, .z = 0.90559333379893991});
  const Vec3 third_velocity = {.x = -0.62439235349061195, .y = -0.60919352813720618, .z = 0.42189108101411299};
  const SolidContactInputs this_step_inputs = BuildSeveralSurfaceInputs(
      {.x = 0.01406050152026247, .y = 0.011234967740510875, .z = -0.0035773627666760765},
      {.x = -0.46348774667975889, .y = 0.12961149058146249, .z = 0.41462575661629564},
      0.00390625
  );
  cases.push_back({
      .name = "MovingCornerWithCoincidentFaceFastEscape",
      .solids =
          {
              BuildFlatSolid({}, fast_normal, {.linear_velocity = fast_velocity}, 1),
              BuildFlatSolid(
                  {},
                  {.x = 0.0076395093178087256, .y = -0.30456177716485122, .z = -0.95246194768482528},
                  kNoMotion,
                  2
              ),
              BuildFlatSolid({}, third_normal, {.linear_velocity = third_velocity}, 3),
              BuildFlatSolid({}, fast_normal, {.linear_velocity = fast_velocity}, 4),
          },
      .inputs = this_step_inputs,
  });
  const Vec3 sliding_normal =
      math::Normalized({.x = 0.65778296211934739, .y = -0.011871601561922683, .z = -0.75311396204150416});
  const Vec3 sliding_velocity = {.x = 0.46599492144065291, .y = -0.78654928644800204, .z = -0.012813640966278239};
  cases.push_back({
      .name = "MovingCornerWithCoincidentFaceNextStepEscape",
      .solids =
          {
              BuildFlatSolid({}, sliding_normal, {.linear_velocity = sliding_velocity}, 1),
              BuildFlatSolid(
                  {},
                  {.x = 0.26194660831403604, .y = 0.96177550773756182, .z = 0.079823850501765575},
                  kNoMotion,
                  2
              ),
              BuildFlatSolid(
                  {},
                  {.x = -0.60349893121568587, .y = -0.027131697348259116, .z = 0.79690207116089695},
                  kNoMotion,
                  3
              ),
              BuildFlatSolid({}, sliding_normal, {.linear_velocity = sliding_velocity}, 4),
          },
      .inputs = BuildSeveralSurfaceInputs(
          {.x = 1.2732426756317884e-08, .y = -3.4591972253611191e-09, .z = 1.0692083265214589e-08},
          {.x = 0.36260184477733048, .y = 0.043237413449602746, .z = -0.84629524815196766},
          0.00390625
      ),
  });
  // NOLINTEND(modernize-use-std-numbers)
  // A wedge of 120 degrees with the particle at its rest point moving at {-1, 1, -1} m/s for 0.01 s, friction 0, with
  // the floor listed first (ids 1, 2) or the side first.
  const double obtuse_angle = 2 * std::numbers::pi / 3;  // rad
  const SolidContactInputs obtuse_inputs = BuildSeveralSurfaceInputs(
      {.x = kTolerance / std::tan(obtuse_angle / 2), .y = 0, .z = kTolerance},
      {.x = -1, .y = 1, .z = -1},
      0.01
  );
  cases.push_back({
      .name = "ObtuseWedgeFloorFirst",
      .solids = {BuildFloor(1), BuildWedgeSide(std::numbers::pi - obtuse_angle, 2)},
      .inputs = obtuse_inputs,
  });
  cases.push_back({
      .name = "ObtuseWedgeSideFirst",
      .solids = {BuildWedgeSide(std::numbers::pi - obtuse_angle, 1), BuildFloor(2)},
      .inputs = obtuse_inputs,
  });
  // Two reacting flat solids through the origin, the first with normal Normalized({2e-13, 0, 1}), the second the
  // floor: their normals share one direction (kSameNormalAxisDifference). The particle 0.999 tol above both moving at
  // {1, 0, -1} m/s for 0.2 s.
  Solid tilted_reacting_floor = BuildFlatSolid({}, math::Normalized({.x = 2e-13, .y = 0, .z = 1}), kNoMotion, 1);
  tilted_reacting_floor.reports_reaction = true;
  Solid second_reacting_floor = BuildFloor(2);
  second_reacting_floor.reports_reaction = true;
  cases.push_back({
      .name = "CoincidentReactingFloorsNormalsApartByRounding",
      .solids = {tilted_reacting_floor, second_reacting_floor},
      .inputs = BuildSeveralSurfaceInputs({.x = 0.95, .y = 0, .z = 0.999 * kTolerance}, {.x = 1, .y = 0, .z = -1}, 0.2),
  });
  // A wedge of 1e-8 rad (a sliver), the particle on the floor 0.005 m outside the rest point sliding into it at
  // 0.1 m/s, in steps of 0.01 s.
  const double sliver_angle = 1e-8;  // rad
  cases.push_back({
      .name = "SlowSlideIntoSliverWedge",
      .solids = {BuildFloor(1), BuildWedgeSide(std::numbers::pi - sliver_angle, 2)},
      .inputs = BuildSeveralSurfaceInputs(
          {.x = kTolerance / std::tan(sliver_angle / 2) + 0.005, .y = 0, .z = kTolerance},
          {.x = -0.1, .y = 0, .z = 0},
          0.01
      ),
  });
  return cases;
}

// One wedge at its rest point, listed both ways: the two scene names, the wedge angle and the velocity.
struct RestPointWedge {
  std::string_view floor_first_name;
  std::string_view side_first_name;
  double angle = 0;  // rad
  Vec3 velocity{};  // m/s
};

// Slides in wedges and over coincident grounds: wedges at their rest point listed in both orders, with friction; a
// narrow wedge on a reacting floor near the velocity pair limit; a domain box and a side with and without a ground
// lying on the box's bottom, or a ground with twice the friction; the copy fuzz scenes.
inline std::vector<MoveParticleCase> BuildWedgeOrderAndCoincidentGroundCases() {
  std::vector<MoveParticleCase> cases;
  const double degree = std::numbers::pi / 180;  // rad
  // Wedges of 30, 45, 60 and 75 degrees with the floor, the particle at the rest point {tol / tan(a / 2), 0, tol} for
  // 0.01 s, friction 0.5, the floor listed first (ids 1, 2) or the side first (ids 1, 2).
  const std::array<RestPointWedge, 4> wedges = {
      RestPointWedge{
          .floor_first_name = "ThirtyDegreeRestPointFloorFirst",
          .side_first_name = "ThirtyDegreeRestPointSideFirst",
          .angle = 30 * degree,
          .velocity = {.x = -1, .y = 1, .z = -1},
      },
      RestPointWedge{
          .floor_first_name = "FortyFiveDegreeRestPointFloorFirst",
          .side_first_name = "FortyFiveDegreeRestPointSideFirst",
          .angle = 45 * degree,
          .velocity = {.x = -0.2, .y = 1, .z = -1},
      },
      RestPointWedge{
          .floor_first_name = "SixtyDegreeRestPointFloorFirst",
          .side_first_name = "SixtyDegreeRestPointSideFirst",
          .angle = 60 * degree,
          .velocity = {.x = -0.2, .y = 1, .z = -1},
      },
      RestPointWedge{
          .floor_first_name = "SeventyFiveDegreeRestPointFloorFirst",
          .side_first_name = "SeventyFiveDegreeRestPointSideFirst",
          .angle = 75 * degree,
          .velocity = {.x = -0.2, .y = 1, .z = -1},
      },
  };
  for (const RestPointWedge& wedge : wedges) {
    SolidContactInputs inputs = BuildSeveralSurfaceInputs(
        {.x = kTolerance / std::tan(wedge.angle / 2), .y = 0, .z = kTolerance},
        wedge.velocity,
        0.01
    );
    inputs.friction = 0.5;
    cases.push_back({
        .name = wedge.floor_first_name,
        .solids = {BuildFloor(1), BuildWedgeSide(std::numbers::pi - wedge.angle, 2)},
        .inputs = inputs,
    });
    cases.push_back({
        .name = wedge.side_first_name,
        .solids = {BuildWedgeSide(std::numbers::pi - wedge.angle, 1), BuildFloor(2)},
        .inputs = inputs,
    });
  }
  // A wedge of 1e-6 rad on a reacting floor, the particle at the rest point moving at {-2, 1, 0} m/s for 0.01 s.
  Solid reacting_floor = BuildFloor(1);
  reacting_floor.reports_reaction = true;
  cases.push_back({
      .name = "MicroradianWedgeOnReactingFloor",
      .solids = {reacting_floor, BuildWedgeSide(std::numbers::pi - 1e-6, 2)},
      .inputs = BuildSeveralSurfaceInputs(
          {.x = kTolerance / std::tan(5e-7), .y = 0, .z = kTolerance},
          {.x = -2, .y = 1, .z = 0},
          0.01
      ),
  });
  // The domain box [-1, 1]^2 x [0, 1] (id 1) and a side through {0, 0.5, 0} at 60 degrees to its bottom (id 2), the
  // particle at their rest point moving at {-1, 1, -1} m/s for 0.01 s, friction 0.5; then the same with a ground, a
  // flat solid at z = 0 (id 3), lying on the box's bottom, and with that ground at friction_scale 2.
  const Solid wide_box = ToSolid(
      DomainBox{.low_corner = {.x = -1, .y = -1, .z = 0}, .high_corner = {.x = 1, .y = 1, .z = 1}, .closed = {}},
      1
  );
  const double sixty = 60 * degree;  // rad
  const Vec3 side_normal = {.x = std::sin(std::numbers::pi - sixty), .y = 0, .z = std::cos(std::numbers::pi - sixty)};
  const Solid box_side = BuildFlatSolid({.x = 0, .y = 0.5, .z = 0}, side_normal, kNoMotion, 2);
  SolidContactInputs box_inputs = BuildSeveralSurfaceInputs(
      {.x = kTolerance / std::tan(sixty / 2), .y = 0.5, .z = kTolerance},
      {.x = -1, .y = 1, .z = -1},
      0.01
  );
  box_inputs.friction = 0.5;
  cases.push_back({.name = "DomainBoxAndSide", .solids = {wide_box, box_side}, .inputs = box_inputs});
  cases.push_back(
      {.name = "DomainBoxAndSideWithGround", .solids = {wide_box, box_side, BuildFloor(3)}, .inputs = box_inputs}
  );
  Solid rough_ground = BuildFloor(3);
  rough_ground.friction_scale = 2;
  cases.push_back(
      {.name = "DomainBoxAndSideWithRoughGround", .solids = {wide_box, box_side, rough_ground}, .inputs = box_inputs}
  );
  // Two draws of a generated copy search: fixed flat solids through the origin at a rest point, T 0.01 s, and the same
  // scene with an exact copy of one solid inserted later in the array, written to 17 digits, each normal normalized
  // again as the fuzz built it. Draw 131: three faces, friction 1.2666, a copy of the first after the second. Draw 51:
  // two faces, friction 0.6248, a copy of the second just before it.
  // NOLINTBEGIN(modernize-use-std-numbers): measured values, not constants.
  const Vec3 corner_first =
      math::Normalized({.x = 0.93951377197048747, .y = -0.11228124952476785, .z = 0.32358429084698065});
  const Vec3 corner_second =
      math::Normalized({.x = -0.87700018288461101, .y = 0.46952715008584572, .z = -0.10205358667201614});
  const Vec3 corner_third =
      math::Normalized({.x = -0.18075399089670258, .y = 0.51700952359533392, .z = 0.83667744518819231});
  SolidContactInputs corner_inputs = BuildSeveralSurfaceInputs(
      {.x = 5.8700994994953113e-10, .y = 3.6071506808514274e-09, .z = 2.4254392139062201e-09},
      {.x = -0.14281725048184113, .y = -0.43233682913915139, .z = -0.69458475428245836},
      0.01
  );
  corner_inputs.friction = 1.2666050571487417;
  cases.push_back({
      .name = "CopyFuzzThreeFaces",
      .solids =
          {
              BuildFlatSolid({}, corner_first, kNoMotion, 1),
              BuildFlatSolid({}, corner_second, kNoMotion, 2),
              BuildFlatSolid({}, corner_third, kNoMotion, 3),
          },
      .inputs = corner_inputs,
  });
  cases.push_back({
      .name = "CopyFuzzThreeFacesWithCopy",
      .solids =
          {
              BuildFlatSolid({}, corner_first, kNoMotion, 1),
              BuildFlatSolid({}, corner_second, kNoMotion, 2),
              BuildFlatSolid({}, corner_first, kNoMotion, 3),
              BuildFlatSolid({}, corner_third, kNoMotion, 4),
          },
      .inputs = corner_inputs,
  });
  const Vec3 wedge_first =
      math::Normalized({.x = -0.882055171725313, .y = 0.43661791686511747, .z = 0.17704086733009927});
  const Vec3 wedge_second =
      math::Normalized({.x = 0.75957377463132725, .y = 0.15963882663198797, .z = 0.63052607077255707});
  SolidContactInputs wedge_inputs = BuildSeveralSurfaceInputs(
      {.x = -2.2307814826944045e-10, .y = 1.0859759391088158e-09, .z = 1.4708399922751076e-09},
      {.x = -0.75257177526028574, .y = 0.61407627377774832, .z = -0.29496983861154258},
      0.01
  );
  wedge_inputs.friction = 0.62475835316586448;
  cases.push_back({
      .name = "CopyFuzzTwoFaces",
      .solids = {BuildFlatSolid({}, wedge_first, kNoMotion, 1), BuildFlatSolid({}, wedge_second, kNoMotion, 2)},
      .inputs = wedge_inputs,
  });
  cases.push_back({
      .name = "CopyFuzzTwoFacesWithCopy",
      .solids =
          {
              BuildFlatSolid({}, wedge_first, kNoMotion, 1),
              BuildFlatSolid({}, wedge_second, kNoMotion, 2),
              BuildFlatSolid({}, wedge_second, kNoMotion, 3),
          },
      .inputs = wedge_inputs,
  });
  // NOLINTEND(modernize-use-std-numbers)
  return cases;
}

// The scenes of a particle against a sphere or a cylinder: a hit head on and at a slant, a path that grazes the grown
// sphere, a start within tol leaving, a hit on a rim and on a cap, spinning solids felt through friction alone, a rim
// hit of the stepping search, a cylinder whose axis tilts too little to count as turning, and a cylinder lying on or
// sunk into a floor. They come last in BuildMoveParticleCases, so the generated sets keep the draws of the scenes
// before them.
inline std::vector<MoveParticleCase> BuildRoundSolidCases() {
  std::vector<MoveParticleCase> cases;
  const Solid half_meter_ball = BuildFixedSphere({}, 0.5, 1);
  // A fixed sphere of radius 0.5 m at the origin, hit head on, at a slant (the hit normal {-0.8, 0.6, 0}), and grazed
  // by a path tol beyond the grown sphere's equator; and a start tol / 4 inside the rest gap, leaving along the
  // surface.
  cases.push_back({
      .name = "SphereHeadOn",
      .solids = {half_meter_ball},
      .inputs = BuildCaseInputs({
          .start_position = {.x = -2, .y = 0, .z = 0},
          .velocity = kUnitX,
          .step_duration = 2,
          .friction = 0,
      }),
  });
  cases.push_back({
      .name = "SphereOblique",
      .solids = {half_meter_ball},
      .inputs = BuildCaseInputs({
          .start_position = {.x = -2, .y = 0.3, .z = 0},
          .velocity = kUnitX,
          .step_duration = 2,
          .friction = 0,
      }),
  });
  cases.push_back({
      .name = "SphereTangentWithinRounding",
      .solids = {half_meter_ball},
      .inputs = BuildCaseInputs({
          .start_position = {.x = -2, .y = 0.5 + kTolerance, .z = 0},
          .velocity = kUnitX,
          .step_duration = 4,
          .friction = 0,
      }),
  });
  cases.push_back({
      .name = "SphereStartWithinToleranceMovingAway",
      .solids = {half_meter_ball},
      .inputs = BuildCaseInputs({
          .start_position = {.x = -(0.5 + kTolerance / 4), .y = 0, .z = 0},
          .velocity = kUnitY,
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // A fixed cylinder along z at the origin, radius 0.5 m and half height 1 m. A particle falling tol / 2 outside the
  // wall enters the cylinder grown with sharp edges Sqrt(1.25) tol from the rim, and the rim search hits it at tol
  // from the rim; there the cylinder reports a reaction. A particle falling on the cap at a slant slides on along it.
  const Cylinder upright_cylinder = {.center = {}, .axis = kUnitZ, .radius = 0.5, .half_height = 1};
  Solid reacting_cylinder = ToSolid(upright_cylinder, kNoMotion, 1);
  reacting_cylinder.reports_reaction = true;
  cases.push_back({
      .name = "CylinderRim",
      .solids = {reacting_cylinder},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + kTolerance / 2, .y = 0, .z = 3},
          .velocity = -kUnitZ,
          .step_duration = 4,
          .friction = 0,
      }),
  });
  cases.push_back({
      .name = "CylinderCap",
      .solids = {ToSolid(upright_cylinder, kNoMotion, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.1, .y = 0, .z = 3},
          .velocity = {.x = 0.1, .y = 0, .z = -1},
          .step_duration = 3,
          .friction = 0,
      }),
  });
  // A cylinder along z, radius 1 m and half height 0.5 m, and a plate with normal z and half thickness 0.5 m, both
  // spinning at {0, 0, 2} rad/s about their own z axis, and a particle tol above the top pressed on it at 1 m/s, with
  // friction 0, 0.25 and 1. The surface under it moves at {0, 1, 0} m/s: a spin, not a turn, so only friction carries
  // the particle sideways.
  const SolidMotion spin_about_z = {.linear_velocity = {}, .angular_velocity = 2 * kUnitZ};
  const Solid spinning_cylinder =
      ToSolid(Cylinder{.center = {}, .axis = kUnitZ, .radius = 1, .half_height = 0.5}, spin_about_z, 1);
  const Solid spinning_plate = ToSolid(Plate{.center = {}, .normal = kUnitZ, .half_thickness = 0.5}, spin_about_z, 1);
  const std::array<std::string_view, 3> cylinder_names = {
      "SpinningCylinderPressedOnCapNoFriction",
      "SpinningCylinderPressedOnCapSlipping",
      "SpinningCylinderPressedOnCapSticking",
  };
  const std::array<std::string_view, 3> plate_names = {
      "SpinningPlatePressedOnNoFriction",
      "SpinningPlatePressedOnSlipping",
      "SpinningPlatePressedOnSticking",
  };
  const std::array<double, 3> frictions = {0, 0.25, 1};
  for (std::size_t friction_index = 0; friction_index < frictions.size(); ++friction_index) {
    const CaseParticle pressed = {
        .start_position = {.x = 0.5, .y = 0, .z = 0.5 + kTolerance},
        .velocity = -kUnitZ,
        .step_duration = 0.1,
        .friction = frictions.at(friction_index),
    };
    cases.push_back({
        .name = cylinder_names.at(friction_index),
        .solids = {spinning_cylinder},
        .inputs = BuildCaseInputs(pressed),
    });
    cases.push_back({
        .name = plate_names.at(friction_index),
        .solids = {spinning_plate},
        .inputs = BuildCaseInputs(pressed),
    });
  }
  // A cylinder along z at the origin, radius 0.5 m and half height 0.5 m, turning at {1e-6, 0, 0} rad/s across its
  // axis: 1e-6 rad/s * 1 s * Length(half_size) = 7.1e-7 m > tol / 8, so it is a turning solid, which the stepping
  // search finds, while the slow turn keeps its rim within 1e-13 m of where it started. The particle moves at 1.5 m/s
  // along {-cos 0.05, 1, -sin 0.05} and would pass {0.5 + tol / 2, 0, 0.5 + tol / 2}, just beyond the high rim, at
  // 0.5 s; the cylinder reports a reaction.
  Solid turning_cylinder = ToSolid(
      Cylinder{.center = {}, .axis = kUnitZ, .radius = 0.5, .half_height = 0.5},
      {.linear_velocity = {}, .angular_velocity = {.x = 1e-6, .y = 0, .z = 0}},
      1
  );
  turning_cylinder.reports_reaction = true;
  const Vec3 oblique_velocity =
      1.5 * math::Normalized(Vec3{.x = -std::cos(0.05), .y = 1, .z = -std::sin(0.05)});  // m/s
  const Vec3 beyond_rim = {.x = 0.5 + kTolerance / 2, .y = 0, .z = 0.5 + kTolerance / 2};  // m
  cases.push_back({
      .name = "TurningCylinderRimHit",
      .solids = {turning_cylinder},
      .inputs = BuildCaseInputs({
          .start_position = beyond_rim - 0.5 * oblique_velocity,
          .velocity = oblique_velocity,
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // The same cylinder with its axis tilting at {1e-9, 0, 0} rad/s: 1e-9 rad/s * 0.1 s * Length(half_size) = 7.1e-11 m
  // is below tol / 8, so it is not a turning solid and keeps its closed form. Its cap sinks at 3e-10 m/s at
  // {0, -0.3, 0.5}, where a particle falling at 1 m/s hits it.
  const Solid tilting_cylinder = ToSolid(
      Cylinder{.center = {}, .axis = kUnitZ, .radius = 0.5, .half_height = 0.5},
      {.linear_velocity = {}, .angular_velocity = {.x = 1e-9, .y = 0, .z = 0}},
      1
  );
  cases.push_back({
      .name = "SlowlyTiltingCylinderCapSinkingAway",
      .solids = {tilting_cylinder},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0, .y = -0.3, .z = 0.55},
          .velocity = -kUnitZ,
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // A cylinder along z at the origin, radius 0.5 m and half height 0.25 m, tilting at {1e-9, 0, 0} rad/s, below plan
  // decision P17's bound for 0.1 s; a particle at rest in the rim corner over its high rim on the +y side, 0.9 tol
  // beyond the wall and the cap, 1.27 tol from the rim, where the rim rises toward it at 1.77e-10 m/s.
  const Solid tilting_flat_cylinder = ToSolid(
      Cylinder{.center = {}, .axis = kUnitZ, .radius = 0.5, .half_height = 0.25},
      {.linear_velocity = {}, .angular_velocity = {.x = 1e-9, .y = 0, .z = 0}},
      1
  );
  const Vec3 rim_corner = {.x = 0, .y = 0.5 + 0.9 * kTolerance, .z = 0.25 + 0.9 * kTolerance};  // m
  cases.push_back({
      .name = "SlowlyTiltingCylinderRimCornerStart",
      .solids = {tilting_flat_cylinder},
      .inputs = BuildCaseInputs({.start_position = rim_corner, .velocity = {}, .step_duration = 0.1, .friction = 0}),
  });
  // A rim corner of the same cylinder, fixed, 41/64 tol beyond the wall and 61/64 tol beyond the cap (both exact in
  // binary), 1.15 tol from the rim, beside an upright flat solid at x = tol that the particle rests on and pushes into
  // at {1, 61/64, -41/64} or {1, -61/64, 41/64} m/s: the start's response leaves {0, 61/64, -41/64} or its opposite,
  // square to the rim's normal {0, 41, 61} / Sqrt(5402) up to the rounding of that normal, -1.1e-16 m/s along it for
  // the first, and the sweep after it starts in the rim corner.
  const Solid fixed_flat_cylinder =
      ToSolid(Cylinder{.center = {}, .axis = kUnitZ, .radius = 0.5, .half_height = 0.25}, kNoMotion, 1);
  const Solid upright_beside_rim = BuildFlatSolid({.x = kTolerance, .y = 0, .z = 0}, -kUnitX, kNoMotion, 2);
  const Vec3 exact_rim_corner = {.x = 0, .y = 0.5 + 41.0 / 64 * kTolerance, .z = 0.25 + 61.0 / 64 * kTolerance};  // m
  cases.push_back({
      .name = "RimCornerBesideRestingFaceDown",
      .solids = {fixed_flat_cylinder, upright_beside_rim},
      .inputs = BuildCaseInputs({
          .start_position = exact_rim_corner,
          .velocity = {.x = 1, .y = 61.0 / 64, .z = -41.0 / 64},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  cases.push_back({
      .name = "RimCornerBesideRestingFaceUp",
      .solids = {fixed_flat_cylinder, upright_beside_rim},
      .inputs = BuildCaseInputs({
          .start_position = exact_rim_corner,
          .velocity = {.x = 1, .y = -61.0 / 64, .z = 41.0 / 64},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // A draw of the moving rim-on-floor slides (GeneratedCylinderTangentContactSlides): a cylinder of radius 0.26 m
  // tilted by 0.368 rad rests on its low rim on a floor, both sliding, the cylinder 1.07 m/s faster and rising at
  // 1e-3 m/s; a particle falling at 0.981 m/s starts at the floor's rest gap 8.7e-6 m from the contact point, which the
  // rim reaches at 1.86e-6 s. The digits are those of the draw.
  // NOLINTBEGIN(modernize-use-std-numbers)
  const Solid sliding_floor = BuildFlatSolid(
      {.x = 0.85898938044651341, .y = 0.13829372338769108, .z = 0},
      kUnitZ,
      {.linear_velocity = {.x = 0.098132328722849951, .y = 0.052614906110469414, .z = 0}, .angular_velocity = {}},
      1
  );
  const Cylinder rim_on_floor = {
      .center = {.x = 0.670218562072598, .y = 0.13829372338769108, .z = 0.23249606641035031},
      .axis = {.x = 0.36011994161504451, .y = 0, .z = 0.93290601222801484},
      .radius = 0.2598319012556512,
      .half_height = 0.14891684208214101,
  };
  const SolidMotion rim_motion = {
      .linear_velocity = {.x = 0.87754499374485984, .y = -0.61674061158165361, .z = 0.001},
      .angular_velocity = {},
  };
  SolidContactInputs rim_sweep_inputs = BuildCaseInputs({
      .start_position = {.x = 0.85899101328368643, .y = 0.13830232324675262, .z = -8.5167620574356801e-10},
      .velocity = {.x = 0, .y = 0, .z = -0.98100000000000009},
      .step_duration = 0.1,
      .friction = 0,
  });
  // NOLINTEND(modernize-use-std-numbers)
  rim_sweep_inputs.start_overlap_tolerance = 2 * kTolerance;
  cases.push_back({
      .name = "RimOnFloorSweepingParticle",
      .solids = {sliding_floor, ToSolid(rim_on_floor, rim_motion, 2)},
      .inputs = rim_sweep_inputs,
  });
  // The cross-sections of the ball scenes of BuildSlideIntoCuspCases: a cylinder along x, radius 0.5 m and half height
  // 1 m, lying on the floor along a line, or sunk into it so that its wall meets the floor at 60 degrees, at y =
  // Sqrt(0.1875).
  const Cylinder lying_cylinder =
      {.center = {.x = 0, .y = 0, .z = 0.5}, .axis = kUnitX, .radius = 0.5, .half_height = 1};
  cases.push_back({
      .name = "CylinderLyingOnFloorSlidingIntoCusp",
      .solids = {BuildFloor(1), ToSolid(lying_cylinder, kNoMotion, 2)},
      .inputs = BuildSeveralSurfaceInputs({.x = 0, .y = 0.05, .z = kTolerance}, {.x = 0, .y = -1, .z = 0}, 0.1),
  });
  Cylinder sunk_cylinder = lying_cylinder;
  sunk_cylinder.center = {.x = 0, .y = 0, .z = 0.25};
  cases.push_back({
      .name = "CylinderSunkIntoFloorSlidingIn",
      .solids = {BuildFloor(1), ToSolid(sunk_cylinder, kNoMotion, 2)},
      .inputs = BuildSeveralSurfaceInputs({.x = 0, .y = 0.5, .z = kTolerance}, {.x = 0, .y = -1, .z = 0}, 0.1),
  });
  // The rim of the fixed cylinder of CylinderRim, reached tol / 4 outside the wall: the entry into the cylinder grown
  // with sharp edges is Sqrt(17 / 16) tol = 1.0308 tol from the rim, inside the touched window, and the rim search goes
  // on to tol.
  cases.push_back({
      .name = "CylinderRimNearRestGap",
      .solids = {ToSolid(upright_cylinder, kNoMotion, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + kTolerance / 4, .y = 0, .z = 3},
          .velocity = -kUnitZ,
          .step_duration = 4,
          .friction = 0,
      }),
  });
  // The cylinder of CylinderRim rising at 1 m/s under a particle at rest tol / 2 outside its wall: its rim reaches the
  // particle tol from it at 2 - Sqrt(0.75) tol s, 2 m above where it started, where the hit is read.
  cases.push_back({
      .name = "RisingCylinderRim",
      .solids = {ToSolid(upright_cylinder, {.linear_velocity = kUnitZ, .angular_velocity = {}}, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + kTolerance / 2, .y = 0, .z = 3},
          .velocity = {},
          .step_duration = 4,
          .friction = 0,
      }),
  });
  // The cylinder of CylinderRim tilting at {1e-11, 0, 0} rad/s, below plan decision P17's bound for 4 s
  // (1e-11 * 4 * Length(half_size) = 4.5e-11 m < tol / 8): its closed form and rim search find the rim hit at the pose
  // of the sweep's start.
  cases.push_back({
      .name = "SlowlyTiltingCylinderRim",
      .solids =
          {ToSolid(upright_cylinder, {.linear_velocity = {}, .angular_velocity = {.x = 1e-11, .y = 0, .z = 0}}, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + kTolerance / 2, .y = 0, .z = 3},
          .velocity = -kUnitZ,
          .step_duration = 4,
          .friction = 0,
      }),
  });
  // A long thin cylinder along x, radius 0.1 m and half height 2 m, tilting about y at 0.99 of plan decision P17's
  // bound for 1 s, measured with the largest arm, Sqrt(0.1^2 + 2^2): a particle tol above its top at x = 1.95 slides
  // along it to x = -1.95 while falling onto it, so the surface under it moves the other way from where the sweep
  // started.
  const double long_cylinder_tilt_rate = 0.99 * (kTolerance / 8) / std::sqrt(0.1 * 0.1 + 2.0 * 2.0);  // rad/s
  const SolidMotion long_cylinder_tilt = {
      .linear_velocity = {},
      .angular_velocity = {.x = 0, .y = long_cylinder_tilt_rate, .z = 0},
  };
  cases.push_back({
      .name = "LongCylinderTiltingUnderSlide",
      .solids =
          {ToSolid(Cylinder{.center = {}, .axis = kUnitX, .radius = 0.1, .half_height = 2}, long_cylinder_tilt, 1)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 1.95, .y = 0, .z = 0.1 + kTolerance},
          .velocity = {.x = -3.9, .y = 0, .z = -1},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // Two generated scenes whose settle depends on the array (MoveToRestGapOfNearSurfaces, Order); the
  // digits are the probe's. A small cylinder spinning about its axis and tilting within plan decision P17's bound,
  // alone and with an exact copy appended; a floor, a moving cylinder and a ball, in array order and reversed.
  // NOLINTBEGIN(modernize-use-std-numbers)
  const Vec3 spinning_axis = {.x = 0.54424759236949605, .y = 0.82685055739460867, .z = -0.14181930029524106};
  const Vec3 tilt_axis = {.x = 0.25166100768697314, .y = 0.0003495224909581074, .z = 0.96781538272751455};
  const SolidMotion spin_and_tilt = {
      .linear_velocity = {},
      .angular_velocity = 0.62777707137839833 * spinning_axis + 7.5918087161475426e-08 * tilt_axis,
  };
  const Solid small_spinning_cylinder = ToSolid(
      Cylinder{.center = {}, .axis = spinning_axis, .radius = 0x1p-10, .half_height = 0.0005472749110384536},
      spin_and_tilt,
      2
  );
  SolidContactInputs spinning_settle_inputs = BuildSeveralSurfaceInputs(
      {.x = -0.0014995795403805331, .y = -0.001060870078780135, .z = -0.00069999076679189522},
      {.x = 0.0010976265701581423, .y = 0.00153252610932527, .z = 0.00015802081368675625},
      0.97019310186369823
  );
  spinning_settle_inputs.friction = 0.18828305070590459;
  cases.push_back({
      .name = "SettleNearSpinningCylinder",
      .solids = {small_spinning_cylinder},
      .inputs = spinning_settle_inputs,
  });
  Solid spinning_copy = small_spinning_cylinder;
  spinning_copy.solid_id = 9;
  cases.push_back({
      .name = "SettleNearSpinningCylinderWithCopy",
      .solids = {small_spinning_cylinder, spinning_copy},
      .inputs = spinning_settle_inputs,
  });
  Solid settle_floor = BuildFlatSolid({.x = 0, .y = 0, .z = -1.2093127196222974e-07}, kUnitZ, kNoMotion, 1);
  Solid moving_cylinder = ToSolid(
      Cylinder{
          .center = {},
          .axis = {.x = 0.58232782619082502, .y = -0.71084212380994805, .z = -0.39445884178356433},
          .radius = 1.1920928955078125e-07,
          .half_height = 6.7554625712041718e-08,
      },
      {
          .linear_velocity = {.x = -1.0803464904894532e-07, .y = -3.2175558651067555e-08, .z = 9.9445779674895571e-09},
          .angular_velocity = {},
      },
      2
  );
  moving_cylinder.reports_reaction = true;
  Solid settle_ball = BuildFixedSphere(
      {.x = 1.9592410165257721e-07, .y = 7.4709606268181416e-08, .z = 4.0107509563519118e-08},
      1.1835357514480232e-07,
      3
  );
  const SolidContactInputs beside_cylinder_inputs = BuildSeveralSurfaceInputs(
      {.x = 2.1554142195350923e-07, .y = -1.9235249551791681e-07, .z = 3.5112707995058077e-08},
      {.x = -1.4195860903363816e-06, .y = 1.1753898597513622e-06, .z = -5.1846746680689854e-07},
      0.33351149169167332
  );
  cases.push_back({
      .name = "SettleBesideMovingCylinder",
      .solids = {settle_floor, moving_cylinder, settle_ball},
      .inputs = beside_cylinder_inputs,
  });
  settle_ball.solid_id = 1;
  settle_floor.solid_id = 3;
  cases.push_back({
      .name = "SettleBesideMovingCylinderReversed",
      .solids = {settle_ball, moving_cylinder, settle_floor},
      .inputs = beside_cylinder_inputs,
  });
  // NOLINTEND(modernize-use-std-numbers)
  return cases;
}

// The scenes of a cylinder's rim at its true distance and of flush seams between solids (plan decision P21): two drums
// of one radius stacked, a puck flush in a floor, a drum standing on a floor, a rim crest just under a floor, a rim met
// from the side, the rim sliding over a floor a little lower, two boxes stacked, two boxes side by side and a box flush
// in a floor. They come after every other scene in BuildMoveParticleCases, so the generated sets keep the draws of the
// scenes before them.
inline std::vector<MoveParticleCase> BuildRimAndSeamCases() {
  std::vector<MoveParticleCase> cases;
  // Two drums along z, radius 0.5 m and half height 0.5 m, the upper from z = 1 to 2 and the lower from 0 to 1, and a
  // particle sliding down the seam at the rest gap of both walls.
  const Cylinder upper_drum = {.center = {.x = 0, .y = 0, .z = 1.5}, .axis = kUnitZ, .radius = 0.5, .half_height = 0.5};
  const Cylinder lower_drum = {.center = {.x = 0, .y = 0, .z = 0.5}, .axis = kUnitZ, .radius = 0.5, .half_height = 0.5};
  cases.push_back({
      .name = "StackedDrums",
      .solids = {ToSolid(upper_drum, kNoMotion, 1), ToSolid(lower_drum, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + kTolerance, .y = 0, .z = 1.8},
          .velocity = -kUnitZ,
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // A puck, radius 0.5 m and half height 0.1 m, set flush into the fixed floor, its top at z = 0, and a particle
  // sliding along the floor at the rest gap onto it.
  const Cylinder puck = {.center = {.x = 0, .y = 0, .z = -0.1}, .axis = kUnitZ, .radius = 0.5, .half_height = 0.1};
  cases.push_back({
      .name = "PuckFlushInFloor",
      .solids = {BuildFloor(1), ToSolid(puck, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + 3 * kTolerance, .y = 0, .z = kTolerance},
          .velocity = -kUnitX,
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // A drum, radius 0.5 m and half height 0.5 m, standing on the fixed floor, and a particle sliding along the floor
  // into the groove between them.
  cases.push_back({
      .name = "DrumStandingOnFloor",
      .solids = {BuildFloor(1), ToSolid(lower_drum, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 1, .y = 0, .z = kTolerance},
          .velocity = -kUnitX,
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // A cylinder, radius 0.5 m and half height 0.25 m, its axis at 45 degrees in the x-z plane, sunk into the fixed floor
  // so that the crest of its high rim, at x = -0.25 Sqrt(0.5), lies 0.41 tol under the floor's surface; a particle
  // sliding along the floor at the rest gap passes over the crest 1.41 tol from it.
  // NOLINTBEGIN(modernize-use-std-numbers)
  const double half_root_two = std::sqrt(0.5);
  const Cylinder sunk_slanted = {
      .center = {.x = 0, .y = 0, .z = -(0.25 + 0.5) * half_root_two - 0.41 * kTolerance},
      .axis = {.x = half_root_two, .y = 0, .z = half_root_two},
      .radius = 0.5,
      .half_height = 0.25,
  };
  // NOLINTEND(modernize-use-std-numbers)
  cases.push_back({
      .name = "RimCrestUnderFloor",
      .solids = {BuildFloor(1), ToSolid(sunk_slanted, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = -0.25 * half_root_two - 0.25, .y = 0, .z = kTolerance},
          .velocity = kUnitX,
          .step_duration = 0.5,
          .friction = 0,
      }),
  });
  // The fixed cylinder of CylinderRim met from the side, tol / 2 above the cap's plane: the rim search hits it where
  // the true distance is tol, Sqrt(0.75) tol beyond the wall.
  Solid reacting_cylinder =
      ToSolid(Cylinder{.center = {}, .axis = kUnitZ, .radius = 0.5, .half_height = 1}, kNoMotion, 1);
  reacting_cylinder.reports_reaction = true;
  cases.push_back({
      .name = "CylinderRimFromTheSide",
      .solids = {reacting_cylinder},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 2, .y = 0, .z = 1 + kTolerance / 2},
          .velocity = -kUnitX,
          .step_duration = 4,
          .friction = 0,
      }),
  });
  // The scene RimOnFloorSweepingParticle with the cylinder rising at 9e-4 m/s instead of 1e-3: its low rim now passes
  // within tol of the particle resting on the floor. The digits are those of the draw.
  // NOLINTBEGIN(modernize-use-std-numbers)
  const Solid sliding_floor = BuildFlatSolid(
      {.x = 0.85898938044651341, .y = 0.13829372338769108, .z = 0},
      kUnitZ,
      {.linear_velocity = {.x = 0.098132328722849951, .y = 0.052614906110469414, .z = 0}, .angular_velocity = {}},
      1
  );
  const Cylinder rim_on_floor = {
      .center = {.x = 0.670218562072598, .y = 0.13829372338769108, .z = 0.23249606641035031},
      .axis = {.x = 0.36011994161504451, .y = 0, .z = 0.93290601222801484},
      .radius = 0.2598319012556512,
      .half_height = 0.14891684208214101,
  };
  const SolidMotion lower_rim_motion = {
      .linear_velocity = {.x = 0.87754499374485984, .y = -0.61674061158165361, .z = 0.0009},
      .angular_velocity = {},
  };
  SolidContactInputs rim_sweep_inputs = BuildCaseInputs({
      .start_position = {.x = 0.85899101328368643, .y = 0.13830232324675262, .z = -8.5167620574356801e-10},
      .velocity = {.x = 0, .y = 0, .z = -0.98100000000000009},
      .step_duration = 0.1,
      .friction = 0,
  });
  // NOLINTEND(modernize-use-std-numbers)
  rim_sweep_inputs.start_overlap_tolerance = 2 * kTolerance;
  cases.push_back({
      .name = "RimOnFloorSweepingParticleLower",
      .solids = {sliding_floor, ToSolid(rim_on_floor, lower_rim_motion, 2)},
      .inputs = rim_sweep_inputs,
  });
  // Two boxes of half size 0.5 m stacked into one side, the upper from z = 1 to 2 and the lower from 0 to 1, and a
  // particle sliding down the seam at the rest gap of their kHighX faces.
  const Vec3 unit_half_size = {.x = 0.5, .y = 0.5, .z = 0.5};
  const Box upper_box = {.center = {.x = 0, .y = 0, .z = 1.5}, .axes = kIdentity, .half_size = unit_half_size};
  const Box lower_box = {.center = {.x = 0, .y = 0, .z = 0.5}, .axes = kIdentity, .half_size = unit_half_size};
  cases.push_back({
      .name = "StackedBoxes",
      .solids = {ToSolid(upper_box, kNoMotion, 1), ToSolid(lower_box, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + kTolerance, .y = 0, .z = 1.8},
          .velocity = -kUnitZ,
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // Two boxes side by side in one floor, of half size 0.5 m with their tops at z = 0, side by side along x, and a
  // particle sliding along them at the rest gap across the seam at x = 0.5.
  const Box first_floor_box = {.center = {.x = 0, .y = 0, .z = -0.5}, .axes = kIdentity, .half_size = unit_half_size};
  const Box second_floor_box = {.center = {.x = 1, .y = 0, .z = -0.5}, .axes = kIdentity, .half_size = unit_half_size};
  cases.push_back({
      .name = "BoxesSideBySide",
      .solids = {ToSolid(first_floor_box, kNoMotion, 1), ToSolid(second_floor_box, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.2, .y = 0, .z = kTolerance},
          .velocity = kUnitX,
          .step_duration = 0.6,
          .friction = 0,
      }),
  });
  // A box of half size 0.5 m set flush into the fixed floor, its top at z = 0, and a particle sliding along the floor
  // at the rest gap onto it.
  cases.push_back({
      .name = "BoxFlushInFloor",
      .solids = {BuildFloor(1), ToSolid(first_floor_box, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + 3 * kTolerance, .y = 0, .z = kTolerance},
          .velocity = -kUnitX,
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  return cases;
}

// The scenes with one value out of range: each is the docstring example with one change.
inline std::vector<MoveParticleCase> BuildBadInputCases() {
  std::vector<MoveParticleCase> cases;
  const SolidContactInputs valid = BuildCaseInputs({
      .start_position = {.x = 0.9, .y = 0, .z = 0},
      .velocity = kUnitX,
      .step_duration = 0.2,
      .friction = 0,
  });
  SolidContactInputs non_finite_start = valid;
  non_finite_start.start_position.y = std::numeric_limits<double>::quiet_NaN();
  cases.push_back({.name = "NonFiniteStart", .solids = {BuildUprightAtXOne(1)}, .inputs = non_finite_start});
  SolidContactInputs zero_step = valid;
  zero_step.step_duration = 0;
  cases.push_back({.name = "ZeroStepDuration", .solids = {BuildUprightAtXOne(1)}, .inputs = zero_step});
  SolidContactInputs negative_friction = valid;
  negative_friction.friction = -1;
  cases.push_back({.name = "NegativeFriction", .solids = {BuildUprightAtXOne(1)}, .inputs = negative_friction});
  SolidContactInputs zero_tolerance = valid;
  zero_tolerance.contact_tolerance = 0;
  cases.push_back({.name = "ZeroContactTolerance", .solids = {BuildUprightAtXOne(1)}, .inputs = zero_tolerance});
  cases.push_back({.name = "SolidsOutOfIdOrder", .solids = {BuildUprightAtXOne(2), BuildFloor(1)}, .inputs = valid});
  SolidContactInputs non_finite_velocity = valid;
  non_finite_velocity.velocity.for_next_step.z = std::numeric_limits<double>::infinity();
  cases.push_back(
      {.name = "NonFiniteVelocityForNextStep", .solids = {BuildUprightAtXOne(1)}, .inputs = non_finite_velocity}
  );
  SolidContactInputs negative_overlap = valid;
  negative_overlap.start_overlap_tolerance = -0.001;
  cases.push_back({.name = "NegativeOverlapTolerance", .solids = {BuildUprightAtXOne(1)}, .inputs = negative_overlap});
  cases.push_back({.name = "SolidsWithEqualIds", .solids = {BuildUprightAtXOne(1), BuildFloor(1)}, .inputs = valid});
  SolidContactInputs unknown_plate = valid;
  unknown_plate.plate_side = {.plate_id = 99, .side = SolidSide::kAlongNormal};
  cases.push_back({.name = "PlateSideNamingNoSolid", .solids = {BuildUprightAtXOne(1)}, .inputs = unknown_plate});
  SolidContactInputs not_a_plate = valid;
  not_a_plate.plate_side = {.plate_id = 1, .side = SolidSide::kAlongNormal};
  cases.push_back({.name = "PlateSideNamingFlatSolid", .solids = {BuildUprightAtXOne(1)}, .inputs = not_a_plate});
  SolidContactInputs side_without_plate = valid;
  side_without_plate.plate_side = {.plate_id = 0, .side = SolidSide::kAlongNormal};
  cases.push_back({.name = "PlateSideWithoutPlateId", .solids = {BuildUprightAtXOne(1)}, .inputs = side_without_plate});
  SolidContactInputs plate_without_side = valid;
  plate_without_side.plate_side = {.plate_id = 1, .side = SolidSide::kNoPlate};
  const Solid plate_behind =
      ToSolid(Plate{.center = {.x = 2, .y = 0, .z = 0}, .normal = kUnitX, .half_thickness = 0.25}, kNoMotion, 1);
  cases.push_back({.name = "PlateSideWithoutSide", .solids = {plate_behind}, .inputs = plate_without_side});
  return cases;
}

// The scenes of a slide on a turning solid: the rest of the step carried in its axes (plan decision P7), and the test
// of the other solids that comes before it.
inline std::vector<MoveParticleCase> BuildTurningSlideCases() {
  std::vector<MoveParticleCase> cases;
  // A box of half size 0.5 m at the origin turning at 1 rad/s about z: the turning box of the scenes below.
  const Solid turning_box = ToSolid(
      Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
      {.linear_velocity = {}, .angular_velocity = kUnitZ},
      1
  );
  // The particle of TurningBoxHitsRestingParticle, with a fixed floor 0.05 m below it: the box reaches the particle at
  // 0.8857 s, and the floor is within the carry's closing bound for the 0.1143 s left.
  cases.push_back({
      .name = "TurningBoxNearFloor",
      .solids = {turning_box, BuildFlatSolid({.x = 0, .y = 0, .z = -0.05}, kUnitZ, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.6 * std::cos(0.3), .y = 0.6 * std::sin(0.3), .z = 0},
          .velocity = {},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // A particle resting on the face kHighX, at the rest gap, moving at 1 m/s along z: the face moves at
  // {0, 0.5 + tol, 0} m/s there, so in the box's axes the particle slides at {0, -(0.5 + tol), 1} m/s.
  cases.push_back({
      .name = "TurningBoxSlide",
      .solids = {turning_box},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + kTolerance, .y = 0, .z = 0},
          .velocity = kUnitZ,
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // The turning box on a fixed floor at z = -0.5, touching its bottom, and a particle at rest on the face kHighX,
  // 0.05 m above the floor: the carry's closing bound against the floor grows with the step.
  cases.push_back({
      .name = "RestingOnTurningBoxNearFloor",
      .solids = {turning_box, BuildFlatSolid({.x = 0, .y = 0, .z = -0.5}, kUnitZ, kNoMotion, 2)},
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + kTolerance, .y = 0, .z = -0.45},
          .velocity = {},
          .step_duration = 0.1,
          .friction = 0,
      }),
  });
  // A cylinder of radius 0.5 m and half height 1 m along x, turning at 1 rad/s about z, across its axis, and a particle
  // at rest 0.6 m from the origin at 1.2 rad from the axis: the wall reaches it when 0.6 sin(1.2 - t) = 0.5 + tol.
  cases.push_back({
      .name = "CylinderTurningAcrossAxisHitsRestingParticle",
      .solids =
          {
              ToSolid(
                  Cylinder{.center = {}, .axis = kUnitX, .radius = 0.5, .half_height = 1},
                  {.linear_velocity = {}, .angular_velocity = kUnitZ},
                  1
              ),
          },
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.6 * std::cos(1.2), .y = 0.6 * std::sin(1.2), .z = 0},
          .velocity = {},
          .step_duration = 1,
          .friction = 0,
      }),
  });
  // A particle 0.25 tol inside the face kHighX of the turning box, moving into it at 1 m/s, with a start overlap
  // tolerance of tol: moved to the rest gap, it starts inside and approaches, so the start has a response.
  SolidContactInputs inside_start_inputs = BuildCaseInputs({
      .start_position = {.x = 0.5 - kTolerance / 4, .y = 0, .z = 0},
      .velocity = -kUnitX,
      .step_duration = 0.1,
      .friction = 0,
  });
  inside_start_inputs.start_overlap_tolerance = kTolerance;
  cases.push_back({
      .name = "TurningBoxStartInsideApproaching",
      .solids = {turning_box},
      .inputs = inside_start_inputs,
  });
  // Two boxes of half size 0.5 m stacked along z, the upper from z = 1 to 2 and the lower from 0 to 1, both turning at
  // 1 rad/s about z, so their faces stay flush, and a particle sliding down their kHighX faces at the rest gap.
  const Vec3 unit_half_size = {.x = 0.5, .y = 0.5, .z = 0.5};
  const SolidMotion turn_about_z = {.linear_velocity = {}, .angular_velocity = kUnitZ};
  cases.push_back({
      .name = "TurningStackedBoxes",
      .solids =
          {
              ToSolid(
                  Box{.center = {.x = 0, .y = 0, .z = 1.5}, .axes = kIdentity, .half_size = unit_half_size},
                  turn_about_z,
                  1
              ),
              ToSolid(
                  Box{.center = {.x = 0, .y = 0, .z = 0.5}, .axes = kIdentity, .half_size = unit_half_size},
                  turn_about_z,
                  2
              ),
          },
      .inputs = BuildCaseInputs({
          .start_position = {.x = 0.5 + kTolerance, .y = 0, .z = 1.8},
          .velocity = -kUnitZ,
          .step_duration = 1,
          .friction = 0,
      }),
  });
  return cases;
}

}  // namespace move_particle_cases

// Builds every scene MoveParticle is tested on. Each MoveParticle test finds its scene here by name, so the tests of
// MoveParticles can run the same scenes on both processors.
//
// Example: the scene "OctagonRoom" holds eight flat solids around the origin and a particle at the origin moving at
//   {100, 20, 0} m/s for 1 s.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
inline std::vector<MoveParticleCase> BuildMoveParticleCases() {
  std::vector<MoveParticleCase> cases;
  for (MoveParticleCase& test_case : move_particle_cases::BuildExampleCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildPlateRowCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildSeveralHitCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildStartAndEndCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildFloorStartCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildPlateStartCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildSeveralSolidStartCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildDomainCornerStartCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildWedgeStartCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildSeveralSurfaceStartCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildSlideIntoWedgeCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildSlideIntoCuspCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildClosingWedgeCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildEdgeAndCornerSlideCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildWedgeOrderAndCoincidentGroundCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildBadInputCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildRoundSolidCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildRimAndSeamCases()) {
    cases.push_back(std::move(test_case));
  }
  for (MoveParticleCase& test_case : move_particle_cases::BuildTurningSlideCases()) {
    cases.push_back(std::move(test_case));
  }
  return cases;
}

// Finds one scene of BuildMoveParticleCases by name, and fails the calling test when there is none.
//
// Example: FindMoveParticleCase("NoSolid") has no solid.
inline MoveParticleCase FindMoveParticleCase(std::string_view name) {
  for (const MoveParticleCase& test_case : BuildMoveParticleCases()) {
    if (test_case.name == name) {
      return test_case;
    }
  }
  ADD_FAILURE() << "no MoveParticle case named " << name;
  return {};
}

// m. The contact tolerance of the generated sets far from the origin: ComputeContactTolerance for a scene that reaches
// 1100 m from the origin, with its smallest cell and its thinnest solid 0.01 m wide. About 1.0e-9 m, so 2^40 times
// smaller than the largest coordinate, where the test tolerance is 2^30 times smaller than a coordinate of 1 m.
inline constexpr double kFarFromOriginContactTolerance = ComputeContactTolerance({
    .largest_coordinate = 1100,
    .smallest_cell_width = 0.01,
    .thinnest_solid = 0.01,
});

namespace far_from_origin_cases {

constexpr double kStepDuration = 0x1p-6;  // s, the step of every case far from the origin.

// Draws three NextSigned, in x, y, z order.
inline Vec3 BuildSignedVector(CaseGenerator& generator) {
  return {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
}

// Draws a unit vector: three NextSigned, drawn again while shorter than 0.25, then normalized.
inline Vec3 BuildDirection(CaseGenerator& generator) {
  Vec3 draw{};
  while (math::Length(draw) < 0.25) {
    draw = BuildSignedVector(generator);
  }
  return math::Normalized(draw);
}

// Inputs far from the origin for one particle: both velocities, the step kStepDuration, kFarFromOriginContactTolerance,
// no start overlap and no committed plate.
inline SolidContactInputs BuildFarInputs(Vec3 start_position, PerParticleVelocity<Vec3> velocity, double friction) {
  SolidContactInputs inputs;
  inputs.start_position = start_position;
  inputs.velocity = velocity;
  inputs.step_duration = kStepDuration;
  inputs.friction = friction;
  inputs.contact_tolerance = kFarFromOriginContactTolerance;
  return inputs;
}

}  // namespace far_from_origin_cases

// Builds count paths onto one flat solid far from the origin, each a scene of its own named "ObliqueFarFromOrigin":
// the oblique set far from the origin. The solid moves, the particle meets it at a share of the step, slides on with
// friction, and must end the step outside it.
//
// Method: per case, in this order of draws (P11 generator, one for the whole set): the normal (three NextSigned,
//   drawn again while shorter than 0.25, then normalized); the point on the surface, 1000 times three NextSigned m;
//   the solid's linear velocity, 10 times three NextSigned m/s; gap = 0.01 Next + 2 tol m; the tangent, the normalized
//   cross product of the normal with three NextSigned, drawn again while that product is shorter than 0.25;
//   sideways = 5 NextSigned m/s; hit_share = 0.1 + 0.8 Next; for_this_step = linear velocity + sideways tangent -
//   gap / (T hit_share) normal; for_next_step = for_this_step + 0.5 times three NextSigned; start = point + gap normal
//   + 50 NextSigned tangent; friction Next. T = 2^-6 s, the solid's id 1, tol kFarFromOriginContactTolerance.
// Example: every case starts gap from the solid and comes nearer by gap / hit_share per step relative to it, so the
//   path reaches the solid's rest gap at a share of the step a little below hit_share.
inline std::vector<MoveParticleCase> BuildObliqueCases(int count) {
  using far_from_origin_cases::BuildDirection;
  using far_from_origin_cases::BuildSignedVector;
  constexpr double kStepDuration = far_from_origin_cases::kStepDuration;  // s
  CaseGenerator generator;
  std::vector<MoveParticleCase> cases;
  cases.reserve(static_cast<std::size_t>(count));
  for (int case_index = 0; case_index < count; ++case_index) {
    const Vec3 normal = BuildDirection(generator);
    const Vec3 point_on_surface = 1e3 * BuildSignedVector(generator);  // m
    const Vec3 linear_velocity = 10 * BuildSignedVector(generator);  // m/s
    const double gap = 0.01 * generator.Next() + 2 * kFarFromOriginContactTolerance;  // m
    Vec3 tangent_draw{};
    while (math::Length(tangent_draw) < 0.25) {
      tangent_draw = math::Cross(normal, BuildSignedVector(generator));
    }
    const Vec3 tangent = math::Normalized(tangent_draw);
    const double sideways_speed = 5 * generator.NextSigned();  // m/s
    const double hit_share = 0.1 + 0.8 * generator.Next();
    const double approach_speed = gap / (kStepDuration * hit_share);  // m/s, relative to the solid.
    const Vec3 velocity_for_this_step = linear_velocity + sideways_speed * tangent - approach_speed * normal;
    const Vec3 velocity_for_next_step = velocity_for_this_step + 0.5 * BuildSignedVector(generator);
    const Vec3 start_position = point_on_surface + gap * normal + 50 * generator.NextSigned() * tangent;
    const double friction = generator.Next();
    const SolidMotion motion = {.linear_velocity = linear_velocity, .angular_velocity = {}};
    cases.push_back({
        .name = "ObliqueFarFromOrigin",
        .solids = {move_particle_cases::BuildFlatSolid(point_on_surface, normal, motion, 1)},
        .inputs = far_from_origin_cases::BuildFarInputs(
            start_position,
            {.for_next_step = velocity_for_next_step, .for_this_step = velocity_for_this_step},
            friction
        ),
    });
  }
  return cases;
}

// Builds count paths onto one sphere or cylinder far from the origin, each a scene of its own named
// "RoundFarFromOrigin": the round set far from the origin. The solid moves without turning; the particle starts outside
// its bounding ball and aims, relative to it, at a point of the box around it, reached at a share of the step.
//
// Method: per case, in this order of draws (P11 generator, one for the whole set): is_sphere = Next < 0.5; the centre,
//   1000 times three NextSigned m; radius = 0.05 + 0.45 Next m; half_height = 0.05 + 0.45 Next m (drawn for both
//   shapes, used by the cylinder); the axis (three NextSigned, drawn again while shorter than 0.25, then normalized;
//   drawn for both shapes); the linear velocity, 10 times three NextSigned m/s; the direction, drawn as the axis was;
//   start = centre + (Length(half_size) + 0.01 + 0.05 Next) direction; target = centre + axes (0.9 half_size times
//   three NextSigned, axis by axis); hit_share = 0.1 + 0.8 Next; for_this_step = linear velocity + (target - start) /
//   (T hit_share); for_next_step = for_this_step + 0.5 times three NextSigned; friction Next. T = 2^-6 s, the solid's
//   id 1, tol kFarFromOriginContactTolerance.
// Example: a cylinder's target is within 0.9 of its half height of its middle, and within 0.9 radius along each own
//   axis across it, so up to 1.27 radius from its axis: most cylinder paths, and fewer sphere paths, hit.
inline std::vector<MoveParticleCase> BuildRoundCases(int count) {
  using far_from_origin_cases::BuildDirection;
  using far_from_origin_cases::BuildSignedVector;
  constexpr double kStepDuration = far_from_origin_cases::kStepDuration;  // s
  CaseGenerator generator;
  std::vector<MoveParticleCase> cases;
  cases.reserve(static_cast<std::size_t>(count));
  for (int case_index = 0; case_index < count; ++case_index) {
    const bool is_sphere = generator.Next() < 0.5;
    const Vec3 center = 1e3 * BuildSignedVector(generator);  // m
    const double radius = 0.05 + 0.45 * generator.Next();  // m
    const double half_height = 0.05 + 0.45 * generator.Next();  // m
    const Vec3 axis = BuildDirection(generator);
    const SolidMotion motion = {.linear_velocity = 10 * BuildSignedVector(generator), .angular_velocity = {}};
    const Vec3 direction = BuildDirection(generator);
    const Solid solid = is_sphere
        ? ToSolid(Sphere{.center = center, .radius = radius}, motion, 1)
        : ToSolid(Cylinder{.center = center, .axis = axis, .radius = radius, .half_height = half_height}, motion, 1);
    const Vec3 half_size = solid.half_size;  // m
    const Vec3 start_position = center + (math::Length(half_size) + 0.01 + 0.05 * generator.Next()) * direction;
    const Vec3 target_draw = BuildSignedVector(generator);
    const Vec3 own_target = {
        .x = 0.9 * half_size.x * target_draw.x,
        .y = 0.9 * half_size.y * target_draw.y,
        .z = 0.9 * half_size.z * target_draw.z,
    };  // m, along the solid's own axes from its centre.
    const Vec3 target = center + solid.pose_at_step_start.axes * own_target;  // m, world.
    const double hit_share = 0.1 + 0.8 * generator.Next();
    const Vec3 velocity_for_this_step =
        motion.linear_velocity + (1 / (kStepDuration * hit_share)) * (target - start_position);
    const Vec3 velocity_for_next_step = velocity_for_this_step + 0.5 * BuildSignedVector(generator);
    const double friction = generator.Next();
    cases.push_back({
        .name = "RoundFarFromOrigin",
        .solids = {solid},
        .inputs = far_from_origin_cases::BuildFarInputs(
            start_position,
            {.for_next_step = velocity_for_next_step, .for_this_step = velocity_for_this_step},
            friction
        ),
    });
  }
  return cases;
}

// What ComputeEndDistanceCounts found over a list of scenes.
struct EndDistanceCounts {
  int cases = 0;
  int failures = 0;  // MoveParticle returned an error.
  int hits = 0;  // Successes with at least one hit.
  int ends_closer_than_half_tolerance = 0;  // Successes that end closer than tol / 2 to a solid.
  double least_end_distance_share = std::numeric_limits<double>::infinity();  // Of tol, over every success and solid.
  int naive_ends_inside = 0;  // Naive controls that end inside a solid (distance below 0).
};

// Runs MoveParticle on every scene and measures how far each successful end is from every solid at its end pose,
// beside a naive control of the same path: what the rest gap and the settle are for.
//
// Method: a success ends closer than tol / 2 to a solid when ComputeDistanceToSolid at ComputeSolidPose(solid, T)
//   is below tol / 2 for one of them. The naive control runs FindFirstHit with contact tolerance 0, so it hits at the
//   surface itself; at a hit it applies ComputeContactResponse to for_this_step with the solid's velocity at the hit
//   point, the hit's normal and the friction times the solid's friction scale, and moves on for the time left
//   without another sweep; without a hit it moves for the whole step. It ends inside when its least distance to a
//   solid at the end pose is below 0. Every scene is world-aligned or not, any shape; turning solids are allowed, but
//   the control then reads the solid velocity at the hit's pose only.
// Example: a path onto a moving flat solid far from the origin ends tol from it; its naive control ends a few rounding
//   units on either side of the surface.
inline EndDistanceCounts ComputeEndDistanceCounts(const std::vector<MoveParticleCase>& cases) {
  EndDistanceCounts counts;
  for (const MoveParticleCase& test_case : cases) {
    const SolidContactInputs inputs = ToSolidContactInputs(test_case);
    const double step_duration = inputs.step_duration;  // s
    ++counts.cases;
    SolidContactResult result;
    ContactFailure failure;
    if (MoveParticle(inputs, &result, &failure) != ContactError::kNone) {
      ++counts.failures;
    } else {
      counts.hits += result.hit_count > 0 ? 1 : 0;
      bool is_closer_than_half_tolerance = false;
      for (const Solid& solid : test_case.solids) {
        const double end_distance =
            ComputeDistanceToSolid(solid, ComputeSolidPose(solid, step_duration), result.position).distance;  // m
        counts.least_end_distance_share =
            std::min(counts.least_end_distance_share, end_distance / inputs.contact_tolerance);
        is_closer_than_half_tolerance = is_closer_than_half_tolerance || end_distance < inputs.contact_tolerance / 2;
      }
      counts.ends_closer_than_half_tolerance += is_closer_than_half_tolerance ? 1 : 0;
    }

    SolidContactInputs naive_inputs = inputs;
    naive_inputs.contact_tolerance = 0;
    const Vec3 velocity = inputs.velocity.for_this_step;
    FirstHit naive_hit;
    if (FindFirstHit(naive_inputs, inputs.start_position, velocity, 0, step_duration, &naive_hit)
        != ContactError::kNone) {
      continue;
    }
    Vec3 naive_end = inputs.start_position + velocity * step_duration;  // m
    if (naive_hit.solid_index != kNoSolid) {
      const Solid& hit_solid = test_case.solids.at(naive_hit.solid_index);
      const Vec3 hit_position = inputs.start_position + velocity * naive_hit.elapsed;  // m
      const Vec3 solid_velocity =
          ComputeSolidVelocity(hit_solid, ComputeSolidPose(hit_solid, naive_hit.elapsed), hit_position);  // m/s
      const Vec3 response = ComputeContactResponse(
          velocity,
          solid_velocity,
          naive_hit.at_hit.normal_out_of_solid,
          inputs.friction * hit_solid.friction_scale
      );
      naive_end = hit_position + response * (step_duration - naive_hit.elapsed);
    }
    double naive_end_distance = std::numeric_limits<double>::infinity();  // m
    for (const Solid& solid : test_case.solids) {
      const double distance =
          ComputeDistanceToSolid(solid, ComputeSolidPose(solid, step_duration), naive_end).distance;  // m
      naive_end_distance = std::min(naive_end_distance, distance);
    }
    counts.naive_ends_inside += naive_end_distance < 0 ? 1 : 0;
  }
  return counts;
}

// The scenes of the close calls that a shorter step cures or does not cure (ContactFailure::can_shorter_step_cure),
// each in two forms where it has both: touching at the start, where no shorter step cures it, and a millimetre away,
// where a step that ends before the hit does. They are not in BuildMoveParticleCases, whose counts other tests print.
// A domain box 2 m wide, from {-1, -1, 0} to {1, 1, 2}, its top open, with cells of 0.01 m. Each scene's contact
// tolerance is ComputeContactTolerance for its solids over a step of 1 ms (about 1.18e-11 m), kept when a test
// shortens the step. Friction 0 unless named.
//   PaddleThroughBottomSideRestingOnIt         a box paddle 0.8 x 0.02 x 1 m turning at 2 rad/s about z, its
//                                              lower end 1 cm below the bottom side; the particle at rest on the
//                                              bottom side, at the rest gap of the paddle's face
//   PaddleThroughBottomSideOneMillimetreAbove  the same, the particle 1 mm above the bottom side
//   PaddleOnShaftInCrease         a shaft (cylinder of radius 5 cm) spinning at 2 rad/s and a paddle box turning with
//                                 it; the particle at rest in their crease, at the rest gap of both
//   DrivenBoxesInCrease           an L of two boxes moving together at {0.5, 0, 0.5} m/s, both with reports_reaction;
//                                 the particle at rest in their crease, at the rest gap of both
//   DrivenBoxesNearCrease         the same, the particle 1 mm from each box, over a step of 10 ms
//   DescendingBoxAtBottomSide         a box moving down at 0.5 m/s, its bottom 2 tol above the bottom side; the
//                                     particle at rest on the bottom side under it
//   DescendingBoxClosingOnBottomSide  the same box 0.2 mm above the particle
//   SeventeenFloorsAtStart        17 fixed flat solids through the origin with normal +z, friction scales 1 to 17 (so
//                                 no two are copies), friction 0.5, contact tolerance 2^-30 m; the particle tol / 32
//                                 beyond the rest gap, falling at 0.1 m/s for 10 ms: 17 active surfaces
//   SeventeenFloorsBelowFalling   the same floors, the particle 1 mm above them, falling at 0.1 m/s for 0.1 s
// Example: BuildShorterStepCase("DescendingBoxAtBottomSide") is the box of id 2 over the bottom side of the domain box,
//   id 1.
// Requires: name outlives the scene, which keeps it as its name.
inline MoveParticleCase BuildShorterStepCase(std::string_view name) {
  constexpr double kCellWidth = 0.01;  // m
  constexpr double kStepDuration = 1e-3;  // s
  const Solid domain_box = ToSolid(
      DomainBox{.low_corner = {.x = -1, .y = -1, .z = 0}, .high_corner = {.x = 1, .y = 1, .z = 2}, .closed = {}},
      1
  );
  const SolidMotion turning = {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 2}};
  // The tolerance of a scene depends on its solids only, so it is found before the start that uses it.
  const auto compute_tolerance = [&](const std::vector<Solid>& solids) -> double {
    return ComputeContactTolerance(
        ComputeContactScale(solids.data(), static_cast<std::uint32_t>(solids.size()), kStepDuration, kCellWidth)
    );
  };
  const auto build_case = [&](std::vector<Solid> solids, move_particle_cases::CaseParticle particle) {
    SolidContactInputs inputs = move_particle_cases::BuildCaseInputs(particle);
    inputs.start_overlap_tolerance = 1e-4;
    inputs.contact_tolerance = compute_tolerance(solids);
    return MoveParticleCase{.name = name, .solids = std::move(solids), .inputs = inputs};
  };
  if (name == "PaddleThroughBottomSideRestingOnIt" || name == "PaddleThroughBottomSideOneMillimetreAbove") {
    const std::vector<Solid> solids = {
        domain_box,
        ToSolid(
            Box{.center = {.x = 0, .y = 0, .z = 0.49}, .axes = kIdentity, .half_size = {.x = 0.4, .y = 0.01, .z = 0.5}},
            turning,
            2
        ),
    };
    const double tolerance = compute_tolerance(solids);  // m
    const double height = name == "PaddleThroughBottomSideRestingOnIt" ? tolerance : 1e-3;  // m
    const Vec3 gravity_step = {.x = 0, .y = 0, .z = -9.81 * kStepDuration};  // m/s
    return build_case(
        solids,
        {
            .start_position = {.x = 0.2, .y = 0.01 + tolerance, .z = height},
            .velocity = gravity_step,
            .step_duration = kStepDuration,
        }
    );
  }
  if (name == "PaddleOnShaftInCrease") {
    const std::vector<Solid> solids = {
        domain_box,
        ToSolid(
            Cylinder{
                .center = {.x = 0, .y = 0, .z = 1},
                .axis = {.x = 0, .y = 0, .z = 1},
                .radius = 0.05,
                .half_height = 0.8,
            },
            turning,
            2
        ),
        ToSolid(
            Box{.center = {.x = 0.2, .y = 0, .z = 1}, .axes = kIdentity, .half_size = {.x = 0.2, .y = 0.01, .z = 0.3}},
            {.linear_velocity = {.x = 0, .y = 0.4, .z = 0}, .angular_velocity = {.x = 0, .y = 0, .z = 2}},
            3
        ),
    };
    const double tolerance = compute_tolerance(solids);  // m
    const double crease_y = 0.01 + tolerance;  // m
    const double crease_x = std::sqrt(((0.05 + tolerance) * (0.05 + tolerance)) - (crease_y * crease_y));  // m
    return build_case(
        solids,
        {.start_position = {.x = crease_x, .y = crease_y, .z = 1}, .velocity = {}, .step_duration = kStepDuration}
    );
  }
  if (name == "DrivenBoxesInCrease" || name == "DrivenBoxesNearCrease") {
    const SolidMotion driven = {.linear_velocity = {.x = 0.5, .y = 0, .z = 0.5}, .angular_velocity = {}};
    std::vector<Solid> solids = {
        domain_box,
        ToSolid(
            Box{.center = {.x = 0, .y = 0, .z = 1}, .axes = kIdentity, .half_size = {.x = 0.3, .y = 0.3, .z = 0.05}},
            driven,
            2
        ),
        ToSolid(
            Box{
                .center = {.x = -0.35, .y = 0, .z = 1.2},
                .axes = kIdentity,
                .half_size = {.x = 0.05, .y = 0.3, .z = 0.25},
            },
            driven,
            3
        ),
    };
    solids[1].reports_reaction = true;
    solids[2].reports_reaction = true;
    const double tolerance = compute_tolerance(solids);  // m
    if (name == "DrivenBoxesInCrease") {
      return build_case(
          solids,
          {
              .start_position = {.x = -0.3 + tolerance, .y = 0, .z = 1.05 + tolerance},
              .velocity = {},
              .step_duration = kStepDuration,
          }
      );
    }
    return build_case(
        solids,
        {
            .start_position = {.x = -0.3 + 1e-3, .y = 0, .z = 1.05 + 1e-3},
            .velocity = {},
            .step_duration = 10 * kStepDuration,
        }
    );
  }
  if (name == "DescendingBoxAtBottomSide" || name == "DescendingBoxClosingOnBottomSide") {
    const SolidMotion down = {.linear_velocity = {.x = 0, .y = 0, .z = -0.5}, .angular_velocity = {}};
    const Box descending = {
        .center = {.x = 0, .y = 0, .z = 0.1002},
        .axes = kIdentity,
        .half_size = {.x = 0.3, .y = 0.3, .z = 0.1},
    };
    const double tolerance = compute_tolerance({domain_box, ToSolid(descending, down, 2)});  // m
    Box placed = descending;
    if (name == "DescendingBoxAtBottomSide") {
      placed.center.z = 0.1 + 2 * tolerance;
    }
    MoveParticleCase scene = build_case(
        {domain_box, ToSolid(placed, down, 2)},
        {.start_position = {.x = 0, .y = 0, .z = tolerance}, .velocity = {}, .step_duration = kStepDuration}
    );
    scene.inputs.contact_tolerance = tolerance;
    return scene;
  }
  if (name == "SeventeenFloorsAtStart" || name == "SeventeenFloorsBelowFalling") {
    std::vector<Solid> floors;
    for (std::uint32_t floor_id = 1; floor_id <= 17; ++floor_id) {
      Solid floor = ToSolid(
          FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
          SolidMotion(),
          floor_id
      );
      floor.friction_scale = floor_id;
      floors.push_back(floor);
    }
    const bool is_at_start = name == "SeventeenFloorsAtStart";
    const double height = is_at_start ? kTestContactTolerance + (kTestContactTolerance / 32) : 1e-3;  // m
    MoveParticleCase scene = build_case(
        floors,
        {
            .start_position = {.x = 0, .y = 0, .z = height},
            .velocity = {.x = 0, .y = 0, .z = -0.1},
            .step_duration = is_at_start ? 10 * kStepDuration : 100 * kStepDuration,
            .friction = 0.5,
        }
    );
    scene.inputs.contact_tolerance = kTestContactTolerance;
    return scene;
  }
  ADD_FAILURE() << "no shorter-step case named " << name;
  return {};
}

}  // namespace fluidity::contact
