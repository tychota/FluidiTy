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
// Owns:    kTestContactTolerance, ExpectVec3Near, CaseGenerator, MoveParticleCase, ToSolidContactInputs,
//          BuildMoveParticleCases, FindMoveParticleCase
// Uses:    Vec3 (base/vec3.h), Mat3, kIdentity (base/mat3.h), CPU-only guard (base/cpu_only.h), Solid, ToSolid and
//          the shape structs (contact/solid_shapes.h), SolidContactInputs (contact/solid_contact_types.h), GoogleTest
// Used by: the tests of contact/
// Read:    docs/method-paper/fluidity-method.typ <sec-contact-tolerance>, <proc-move-particle>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

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
#include "fluidity/base/vec3.h"
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
  // The reviewer's chasing solid: a fixed flat solid A through the origin, and a flat solid B 10 m away along a normal
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
  // The reviewer's input: one fixed flat solid with normal {0.6, 0, 0.8} through the origin. Rounding leaves the hit
  // position 2.8e-18 m outside the rest gap, and the response leaves an approach of -2^-55 m/s.
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
  // 0.8857 s; with no touching rule on a turning solid the particle is hit again at 0 by each later sweep.
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
  // A ceiling 3 tol above the fixed floor closes at 1.75 tol m/s on a particle at rest at z = tol: it hits at 4/7 s,
  // and the particle, carried down, ends at z = 0.25 tol, too close to the floor.
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
  for (MoveParticleCase& test_case : move_particle_cases::BuildBadInputCases()) {
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

}  // namespace fluidity::contact
