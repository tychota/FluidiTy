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
  // Five floors rising at 1 m/s, ids 1 to 5, at heights 0, 0.0001, ..., 0.0004 m, and a start 0.0005 m below the
  // first: the start is inside all five, which share one direction of the move; each is a hit at time 0, and the
  // fifth is the fifth hit.
  std::vector<Solid> stacked_floors;
  stacked_floors.reserve(5);
  for (int floor_index = 0; floor_index < 5; ++floor_index) {
    stacked_floors.push_back(
        BuildMovingFloor(0.0001 * floor_index, kUnitZ, static_cast<std::uint32_t>(floor_index + 1))
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
  // moved silently to the rest gap, then hit at 0 by every sweep, since a turning solid has no touching rule.
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
  // The review's scene: side_bottom, not the nearest side, is approached at 0.9 tol per step, slower than the
  // touching rule would see.
  cases.push_back({
      .name = "DomainCornerSlowApproachToSecondSide",
      .solids = {domain_box},
      .inputs = BuildDomainStartInputs(two_side_start, {.x = 0, .y = 0, .z = -0.9 * kTolerance / 0.1}),
  });
  // The review's scene: only for_next_step approaches side_bottom.
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
  // The review's scene for mutant S26: side_bottom is the nearest side before the move, side_left comes first in face
  // order, and the two tie at tol after the move.
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
  // move. Each floor is moved silently and, since the upright responds, becomes a touched surface without a hit; the
  // list keeps at most four of them.
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
  // Five fixed floors stacked tol / 16 apart, which a particle tol / 2 above the top one leaves: five silent moves and
  // no start response, so no entry is kept and nothing fails.
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
  // The review's scene: a floor sliding along itself at 100 m/s, which the particle leaves, and the fixed upright at
  // x = 1, which it approaches at 1e-13 m/s. The silent floor's entry must not raise the threshold of the upright's
  // approach test above 0, so the upright is a hit at the start.
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
  // 0.5 tol m/s. No solid responded at the start, so the floor is not a touched surface: its touching rule lets the
  // particle sink 0.05 tol without a hit, and the settle, which reads every solid, lifts it back.
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
  // The review's scene for mutant S11: exactly at the rest gap, approaching at tol m/s for 0.1 s. The start rules do
  // not apply at d = tol, and the touching rule lets the particle rest: it sinks 0.1 tol without a hit, and the settle
  // lifts it back to tol.
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
  // SlidingFloorThenSlowUpright with only for_next_step approaching the upright, at 1e-13 m/s: the silent floor's entry
  // must not raise for_next_step's own threshold above 0 either.
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
  // A ball of radius 1024 m resting on the floor (R / tol = 2^40), the largest a scene allows, and the particle at rest
  // halfway up the gap, f times the rest point's distance x* = 2 Sqrt(1024 tol) = 0.001953125 m from the contact:
  // f = 0.1, 0.01, and 0.0005, within 1.8e-8 R of the contact.
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
  // The unit domain box, a ground lying exactly on side_bottom and a wall lying exactly on side_left, the particle
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
  // The same corner with a ground and a wall whose normals are tilted by 2^-44 rad, rounding-sized: still three
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
