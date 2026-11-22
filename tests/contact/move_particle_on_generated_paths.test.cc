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
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/solid_contact_move_particle_cases.h"

namespace fluidity::contact {
namespace {

// A row of the runs across a seam in many steps: a scene, the point where the path crosses the seam, the direction
// back from there to the starts, the path's direction, the push's direction, the step as a share of tol, the push and
// the friction.
struct SeamRow {
  std::string_view case_name;
  Vec3 seam_point{};  // m, world.
  Vec3 back_from_seam{};  // No unit, unit vector.
  Vec3 along_path{};  // No unit, unit vector; the path moves along it at 1 m/s.
  Vec3 push_direction{};  // No unit, unit vector, into the surface slid on.
  double step_share = 0;  // No unit: the step is step_share tol long, in step_share tol s.
  double push = 0;  // m/s.
  double friction = 0;  // No unit.
};

// Runs one particle across the seam of row_scene, row's scene built once, in Floor(12 / step_share) + 2 steps from
// offset tol before it, the same input velocity every step from the last end, and says whether every step succeeded
// with no hit without a push and one with, kept the speed along the path that friction on the surface slid on leaves (1
// - friction * push, within 1e-12), and stayed at tol - tol / 1024 or more from every solid before the settle.
bool ComputeIsSeamRunRight(const MoveParticleCase& row_scene, const SeamRow& row, double offset) {
  const double step_duration = row.step_share * kTolerance;  // s
  const Vec3 velocity = row.along_path + row.push * row.push_direction;  // m/s
  const double kept_speed = 1 - row.friction * row.push;  // m/s
  const int step_count = static_cast<int>(std::floor(12 / row.step_share)) + 2;
  const bool is_pushed = row.push > 0;
  Vec3 position = row.seam_point + offset * row.back_from_seam;
  for (int step_index = 0; step_index < step_count; ++step_index) {
    MoveParticleCase scene = row_scene;
    scene.inputs.start_position = position;
    scene.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    scene.inputs.step_duration = step_duration;
    scene.inputs.friction = row.friction;
    const MoveParticleCall call = ComputeMove(scene);
    if (call.error != ContactError::kNone) {
      return false;
    }
    const bool is_right_hit_count = call.result.hit_count == (is_pushed ? 1 : 0);
    const double speed = math::Dot(call.result.velocity.for_this_step, row.along_path);  // m/s
    const Vec3 before_settle = ComputePositionBeforeSettle(scene);
    bool stays_out = true;
    for (const Solid& solid : scene.solids) {
      const SolidPose end_pose = ComputeSolidPose(solid, step_duration);
      stays_out = stays_out
          && ComputeDistanceToSolid(solid, end_pose, before_settle).distance >= kTolerance - kTolerance / 1024;
    }
    if (!is_right_hit_count || std::abs(speed - kept_speed) > 1e-12 || !stays_out) {
      return false;
    }
    position = call.result.position;
  }
  return true;
}

// Runs the 18 rows of runs (three steps, three pushes and frictions) for one kind of seam, 400 offsets 4 + s (k + 0.5)
// / 400 tol before the seam each, and fails the calling test unless every run is right.
void CheckSeamRowsKeepTheSpeed(SeamRow row) {
  // The scene is built once: FindMoveParticleCase builds every scene, about 40 times the cost of a move.
  const MoveParticleCase row_scene = FindMoveParticleCase(row.case_name);
  constexpr std::array<double, 3> kStepShares = {0.3, 0.7, 3};
  constexpr std::array<std::array<double, 2>, 3> kPushesAndFrictions = {{{0, 0}, {0.16, 0}, {0.16, 0.3}}};
  for (const double step_share : kStepShares) {
    for (const auto& push_and_friction : kPushesAndFrictions) {
      row.step_share = step_share;
      row.push = push_and_friction.at(0);
      row.friction = push_and_friction.at(1);
      int wrong_runs = 0;
      for (int offset_index = 0; offset_index < 400; ++offset_index) {
        const double offset = (4 + step_share * (offset_index + 0.5) / 400) * kTolerance;  // m
        wrong_runs += ComputeIsSeamRunRight(row_scene, row, offset) ? 0 : 1;
      }
      std::cout << "seam runs " << row.case_name << ", step " << step_share << " tol, push " << row.push
                << ", friction " << row.friction << ": " << wrong_runs << " of 400 wrong\n";
      EXPECT_EQ(wrong_runs, 0) << row.case_name << " " << step_share;
    }
  }
}

TEST(MoveParticle, SeamsOverManySteps_Cylinders) {
  // Down the stacked drums and along the floor onto the puck, from 4 to 4 + s tol before the seam, in steps of s = 0.3,
  // 0.7 and 3 tol, unpressed or pressed at 0.16 m/s into the surface slid on, with friction 0 or 0.3: every step keeps
  // the speed (method paper <sec-touching-rule>). Without that rule, 400, 205 or 224 and 48 runs of 400 lost speed at s
  // = 0.3, 0.7 and 3: some step started within tol + tol / 16 of the next drum's rim and was answered on it.
  CheckSeamRowsKeepTheSpeed({
      .case_name = "StackedDrums",
      .seam_point = {.x = 0.5 + kTolerance, .y = 0, .z = 1},
      .back_from_seam = kUnitZ,
      .along_path = -kUnitZ,
      .push_direction = -kUnitX,
  });
  CheckSeamRowsKeepTheSpeed({
      .case_name = "PuckFlushInFloor",
      .seam_point = {.x = 0.5, .y = 0, .z = kTolerance},
      .back_from_seam = kUnitX,
      .along_path = -kUnitX,
      .push_direction = -kUnitZ,
  });
}

TEST(MoveParticle, SeamsOverManySteps_Boxes) {
  // As SeamsOverManySteps_Cylinders, across two boxes side by side, onto a box flush in a floor and down two stacked
  // boxes: every run keeps the speed. With the next box as an active surface, 83, 35 and 8 runs of 400 lost speed at
  // steps of 0.3, 0.7 and 3 tol.
  CheckSeamRowsKeepTheSpeed({
      .case_name = "BoxesSideBySide",
      .seam_point = {.x = 0.5, .y = 0, .z = kTolerance},
      .back_from_seam = -kUnitX,
      .along_path = kUnitX,
      .push_direction = -kUnitZ,
  });
  CheckSeamRowsKeepTheSpeed({
      .case_name = "BoxFlushInFloor",
      .seam_point = {.x = 0.5, .y = 0, .z = kTolerance},
      .back_from_seam = kUnitX,
      .along_path = -kUnitX,
      .push_direction = -kUnitZ,
  });
  CheckSeamRowsKeepTheSpeed({
      .case_name = "StackedBoxes",
      .seam_point = {.x = 0.5 + kTolerance, .y = 0, .z = 1},
      .back_from_seam = kUnitZ,
      .along_path = -kUnitZ,
      .push_direction = -kUnitX,
  });
}

// Computes whether two calls have the same error, close call, kind, hit count and for_next_step, bit for bit.
bool ComputeIsSameBits(const MoveParticleCall& actual, const MoveParticleCall& expected) {
  const Vec3 actual_velocity = actual.result.velocity.for_next_step;
  const Vec3 expected_velocity = expected.result.velocity.for_next_step;
  const bool is_same_velocity = actual_velocity.x == expected_velocity.x && actual_velocity.y == expected_velocity.y
      && actual_velocity.z == expected_velocity.z;
  const bool is_same_outcome = actual.error == expected.error
      && actual.failure.close_call == expected.failure.close_call && actual.result.kind == expected.result.kind
      && actual.result.hit_count == expected.result.hit_count;
  return is_same_outcome && is_same_velocity;
}

// One generated corner: two to four flat faces through the origin whose normals all lean toward a common free
// direction, the solids, and the particle's inputs.
struct GeneratedCorner {
  std::vector<Solid> solids;
  SolidContactInputs inputs{};
};

// Builds one corner: in this order, the face count, the free direction d, for each face its sideways lean, whether
// it moves and its velocity and friction scale, then the friction, the start kind and the velocity. Each normal is
// Normalized(d + 0.9 lean) with lean a unit vector, so every normal is within 65 degrees of d. Half of the draws move
// their solids, up to 1 m/s per axis; friction scales are 1 or 2 Next(); friction 0 or 2 Next(). The start is the
// rest point found by the move off near surfaces (from the origin), a point 0.3 tol inside the rest gap along d, or
// 0.002 m out along d on a path into the corner.
GeneratedCorner BuildGeneratedCorner(CaseGenerator* generator) {
  GeneratedCorner corner;
  const int face_count = 2 + static_cast<int>(3 * generator->Next());
  const Vec3 free_direction = BuildUnitVector(generator);
  const bool is_moving = generator->Next() < 0.5;
  for (int face_index = 0; face_index < face_count; ++face_index) {
    const Vec3 normal = math::Normalized(free_direction + 0.9 * BuildUnitVector(generator));
    const Vec3 velocity = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
    const double friction_scale = generator->Next() < 0.5 ? 1 : 2 * generator->Next();
    corner.solids.push_back(BuildMovingFlatSolid(normal, is_moving ? velocity : Vec3{}, friction_scale, 1));
  }
  const double friction = generator->Next() < 0.2 ? 0 : 2 * generator->Next();
  const double start_kind = generator->Next();
  const Vec3 velocity = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
  const MoveParticleCase scene =
      BuildOrderedScene(corner.solids, move_particle_cases::BuildSeveralSurfaceInputs({}, velocity, 0.01));
  const SolidContactInputs at_origin = ToSolidContactInputs(scene);
  const RestGapMove rest = MoveToRestGapOfNearSurfaces(at_origin, 0, 0.5 * kTolerance * free_direction);
  Vec3 start = rest.is_found ? rest.moved_point : 4 * kTolerance * free_direction;
  if (start_kind < 0.3) {
    start = start - 0.3 * kTolerance * free_direction;
  } else if (start_kind < 0.6) {
    start = start + 0.002 * free_direction;
  }
  corner.inputs = move_particle_cases::BuildSeveralSurfaceInputs(start, velocity, 0.01);
  corner.inputs.friction = friction;
  return corner;
}

TEST(MoveParticle, GeneratedOneSurfaceHits_SameBitsAsContactResponse) {
  // Method paper <sec-move-particle>: with one active surface each velocity gets ComputeContactResponse with that
  // surface and nothing else, the single-surface response bit for bit. 2000 flat solids through the origin, in this
  // order of draws: the normal (BuildUnitVector), a solid velocity of three NextSigned / 2 m/s or none, the velocity
  // (three NextSigned m/s), friction 2 Next; the start 0.05 m out along the normal, T 0.1 s. Every call whose path hits
  // the solid has both velocities equal to the response of the velocity given, at the hit's normal.
  CaseGenerator generator;
  int hit_calls = 0;
  int differing_calls = 0;
  for (int draw = 0; draw < 2000; ++draw) {
    const Vec3 normal = BuildUnitVector(&generator);
    const bool is_moving = generator.Next() < 0.5;
    const Vec3 drawn_velocity = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const Vec3 solid_velocity = is_moving ? 0.5 * drawn_velocity : Vec3{};
    const Vec3 velocity = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    SolidContactInputs inputs = move_particle_cases::BuildSeveralSurfaceInputs(0.05 * normal, velocity, 0.1);
    inputs.friction = 2 * generator.Next();
    const MoveParticleCall call =
        ComputeMove(BuildOrderedScene({BuildMovingFlatSolid(normal, solid_velocity, 1, 1)}, inputs));
    if (call.error != ContactError::kNone || call.result.hit_count != 1) {
      continue;
    }
    ++hit_calls;
    const Vec3 expected =
        ComputeContactResponse(velocity, solid_velocity, call.result.normal_out_of_solid, inputs.friction);
    const Vec3 actual = call.result.velocity.for_next_step;
    const bool is_same = actual.x == expected.x && actual.y == expected.y && actual.z == expected.z;
    differing_calls += is_same ? 0 : 1;
  }
  std::cout << "one-surface hits " << hit_calls << ", differing from ComputeContactResponse " << differing_calls
            << "\n";
  EXPECT_GT(hit_calls, 400);  // 456 of the draws hit.
  EXPECT_EQ(differing_calls, 0);
}

TEST(MoveParticle, GeneratedCornersInEveryOrder_SameBits) {
  // Method paper <sec-move-particle>: the active surfaces are sorted by value, so every order of the solids array gives
  // the same outcome, kind, hit count and velocity bits; only the named solid and the end position (by rounding of the
  // move off near surfaces, at most 3.4e-9 tol in the model) may follow the array. 2000 corners of two to four faces.
  CaseGenerator generator;
  int differing_draws = 0;
  int failed_draws = 0;
  for (int draw = 0; draw < 2000; ++draw) {
    const GeneratedCorner corner = BuildGeneratedCorner(&generator);
    const MoveParticleCall first = ComputeMove(BuildOrderedScene(corner.solids, corner.inputs));
    failed_draws += first.error == ContactError::kNone ? 0 : 1;
    std::vector<std::size_t> order(corner.solids.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
      order[index] = index;
    }
    bool is_same = true;
    while (std::ranges::next_permutation(order).found) {
      const MoveParticleCall call =
          ComputeMove(BuildOrderedScene(ToPermutedSolids(corner.solids, order), corner.inputs));
      is_same = is_same && ComputeIsSameBits(call, first);
    }
    differing_draws += is_same ? 0 : 1;
  }
  std::cout << "generated corners 2000, failed " << failed_draws << ", differing between orders " << differing_draws
            << "\n";
  EXPECT_EQ(differing_draws, 0);
  EXPECT_LE(failed_draws, 100);  // 0: the comparison is not between failures.
}

// The coordinate moved by one rounding unit down (direction -1), not at all (0) or up (+1).
double ToNeighbourDouble(double coordinate, int direction) {
  if (direction == 0) {
    return coordinate;
  }
  return std::nextafter(coordinate, direction > 0 ? INFINITY : -INFINITY);
}

// Computes whether the wedge of that angle, with its rest-point inputs (BuildRestPointInputs), gives the same bits from
// the 27 starts one rounding unit around the rest point, in both array orders. A failing rest-point start counts as
// differing, so the comparison is never between failures.
bool ComputeIsSameForEveryRounding(double wedge_angle, const SolidContactInputs& center) {
  const Solid floor = move_particle_cases::BuildFloor(1);
  const Solid side = move_particle_cases::BuildWedgeSide(std::numbers::pi - wedge_angle, 2);
  const MoveParticleCall reference = ComputeMove(BuildOrderedScene({floor, side}, center));
  bool is_same = reference.error == ContactError::kNone;
  for (int start_index = 0; start_index < 27; ++start_index) {
    SolidContactInputs inputs = center;
    inputs.start_position = {
        .x = ToNeighbourDouble(center.start_position.x, start_index % 3 - 1),
        .y = ToNeighbourDouble(center.start_position.y, start_index / 3 % 3 - 1),
        .z = ToNeighbourDouble(center.start_position.z, start_index / 9 - 1),
    };
    is_same = is_same && ComputeIsSameBits(ComputeMove(BuildOrderedScene({floor, side}, inputs)), reference)
        && ComputeIsSameBits(ComputeMove(BuildOrderedScene({side, floor}, inputs)), reference);
  }
  return is_same;
}

TEST(MoveParticle, GeneratedRestPointStarts_SameBitsForEveryRounding) {
  // At the rest point of a wedge of 10 to 170 degrees, friction 0 or 0.1 to 2, a velocity in the unit ball (at least
  // 0.05) approaching a face. The responses one surface after the other jumped in 16 % of the draws when the start
  // moved by one rounding unit. One joint response reads the same active surfaces from every one of the 27 starts: the
  // 54 results, both array orders, are the same bits.
  CaseGenerator generator;
  int differing_draws = 0;
  for (int draw = 0; draw < 2000; ++draw) {
    const double wedge_angle = (10 + 160 * generator.Next()) * std::numbers::pi / 180;  // rad
    const double friction = generator.Next() < 0.5 ? 0 : 0.1 + 1.9 * generator.Next();
    Vec3 velocity{};
    do {
      velocity = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    } while (math::Length(velocity) > 1 || math::Length(velocity) < 0.05
             || (velocity.z >= 0 && math::Dot(ToWedgeSideNormal(wedge_angle), velocity) >= 0));
    const SolidContactInputs center = BuildRestPointInputs(wedge_angle, velocity, friction);
    differing_draws += ComputeIsSameForEveryRounding(wedge_angle, center) ? 0 : 1;
  }
  std::cout << "rest-point starts 2000 x 54, draws with differing results " << differing_draws << "\n";
  EXPECT_EQ(differing_draws, 0);
}

// What the fixed corners of GeneratedFixedCorners count.
struct FixedCornerCounts {
  int successes = 0;
  int energy_gains = 0;  // |v'|^2 > |v|^2 (1 + 1e-12).
  int reversed_edges = 0;  // Two faces: the part along their common line changed sign.
  int grown_edges = 0;  // Two faces: the part along their common line grew.
};

// Draws one fixed corner and adds what it shows to *counts. In this order: the face count (2 to 4), each unit normal
// (BuildUnitVector), the start kind, the velocity (each axis in [-1, 1)) and the friction (0 with probability 0.2, else
// up to 3). The start is the rest point found by the move off near surfaces from tol / 2 along the bisector of the
// first two normals, 0.002 m out along it (kind below 0.2) or 0.3 tol inside the rest gap (below 0.4). A draw whose
// rest point is not found, or whose move fails, counts no success.
void ApplyFixedCornerDraw(CaseGenerator* generator, FixedCornerCounts* counts) {
  const int face_count = 2 + static_cast<int>(3 * generator->Next());
  std::vector<Solid> solids;
  std::vector<Vec3> normals;
  for (int face_index = 0; face_index < face_count; ++face_index) {
    normals.push_back(BuildUnitVector(generator));
    solids.push_back(move_particle_cases::BuildFlatSolid({}, normals.back(), kNoMotion, 1));
  }
  const double start_kind = generator->Next();
  const Vec3 velocity = {.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
  const double friction = generator->Next() < 0.2 ? 0 : 3 * generator->Next();
  const Vec3 lean = math::Normalized(normals[0] + normals[1]);
  MoveParticleCase scene =
      BuildOrderedScene(solids, move_particle_cases::BuildSeveralSurfaceInputs({}, velocity, 0.01));
  const RestGapMove rest = MoveToRestGapOfNearSurfaces(ToSolidContactInputs(scene), 0, kTolerance / 2 * lean);
  if (!rest.is_found) {
    return;
  }
  double start_offset = 0;  // m
  if (start_kind < 0.2) {
    start_offset = 0.002;
  } else if (start_kind < 0.4) {
    start_offset = -0.3 * kTolerance;
  }
  scene.inputs.start_position = rest.moved_point + start_offset * lean;
  scene.inputs.friction = friction;
  const MoveParticleCall call = ComputeMove(scene);
  if (call.error != ContactError::kNone) {
    return;
  }
  ++counts->successes;
  const Vec3 result = call.result.velocity.for_next_step;
  counts->energy_gains += math::Dot(result, result) > math::Dot(velocity, velocity) * (1 + 1e-12) ? 1 : 0;
  if (face_count != 2) {
    return;
  }
  const Vec3 edge = math::Normalized(math::Cross(normals[0], normals[1]));
  const double along_before = math::Dot(velocity, edge);  // m/s
  const double along_after = math::Dot(result, edge);  // m/s
  counts->reversed_edges += along_before * along_after < -1e-12 ? 1 : 0;
  counts->grown_edges += std::abs(along_after) > std::abs(along_before) + 1e-12 ? 1 : 0;
}

TEST(MoveParticle, GeneratedFixedCorners_FrictionNeverAddsEnergy) {
  // The generated fixed corners, fixed draws: two to four fixed flat faces through the origin, the start at the rest
  // point, 0.3 tol inside it or 0.002 m out, friction 0 or up to 3. Against fixed solids the joint response never
  // lengthens the velocity, |v'|^2 <= |v|^2 (1 + 1e-12); with two faces the part along their common line is neither
  // reversed nor grown. Why: with every solid fixed the allowed velocities form a cone through 0, the smallest change
  // is the projection on it, which does not lengthen a velocity, and friction only shortens the result toward 0.
  CaseGenerator generator;
  FixedCornerCounts counts;
  for (int draw = 0; draw < 2000; ++draw) {
    ApplyFixedCornerDraw(&generator, &counts);
  }
  std::cout << "fixed corners 2000, succeeded " << counts.successes << ", energy gains " << counts.energy_gains
            << ", edge reversed " << counts.reversed_edges << ", grown " << counts.grown_edges << "\n";
  EXPECT_GE(counts.successes, 1700);  // 1763: the invariants are not vacuous.
  EXPECT_EQ(counts.energy_gains, 0);
  EXPECT_EQ(counts.reversed_edges, 0);
  EXPECT_EQ(counts.grown_edges, 0);
}

// --- Generated paths ---

// What the generated paths gave, counted for the report.
struct GeneratedCounts {
  int paths = 0;
  int free_paths = 0;
  int hit_paths = 0;
  int several_hit_paths = 0;
  int too_many_hits = 0;
  int second_reacting_solid = 0;
  int hover_near_turning_solid = 0;
  int near_other_solid = 0;  // kTurningSolidNearOtherSolid.
  int squeezed = 0;
  std::map<std::string_view, std::array<int, 8>> failures_per_scene;  // Per close call, in ContactCloseCall order.
};

// Prints, for the report, each scene with a failure: its counts of kTooManyHits, kSqueezedBetweenSolids,
// kSecondReactingSolid and kTurningSolidNearOtherSolid.
void WriteFailuresPerScene(const std::map<std::string_view, std::array<int, 8>>& failures_per_scene) {
  std::cout << "failures per scene (too many hits, squeezed, second reacting solid, turning solid near other solid):";
  for (const auto& [scene_name, counts] : failures_per_scene) {
    std::cout << " " << scene_name << " " << counts.at(static_cast<std::size_t>(ContactCloseCall::kTooManyHits)) << " "
              << counts.at(static_cast<std::size_t>(ContactCloseCall::kSqueezedBetweenSolids)) << " "
              << counts.at(static_cast<std::size_t>(ContactCloseCall::kSecondReactingSolid)) << " "
              << counts.at(static_cast<std::size_t>(ContactCloseCall::kTurningSolidNearOtherSolid)) << ";";
  }
  std::cout << "\n";
}

// Fails the calling test unless every solid is at least tol / 2 (of inputs.contact_tolerance) from the end position at
// its end pose, by ComputeDistanceToSolid itself (the generated paths commit to no plate).
void CheckEndOutsideEverySolid(const SolidContactInputs& inputs, Vec3 end_position) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    const double end_distance =
        ComputeDistanceToSolid(solid, ComputeSolidPose(solid, inputs.step_duration), end_position).distance;
    EXPECT_GE(end_distance, inputs.contact_tolerance / 2) << "solid index " << solid_index;
  }
}

// Fails the calling test unless the end is on the same side of every plate as the start, a property the end check
// does not test: a plate is unbounded, so a particle can reach its other side only by passing through it.
void CheckEndOnStartSideOfEveryPlate(const SolidContactInputs& inputs, Vec3 end_position) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    if (solid.shape != SolidShape::kPlate) {
      continue;
    }
    const SolidSide start_side = FindSolidSide(solid, ComputeSolidPose(solid, 0), inputs.start_position);
    const SolidSide end_side = FindSolidSide(solid, ComputeSolidPose(solid, inputs.step_duration), end_position);
    EXPECT_EQ(end_side, start_side) << "plate index " << solid_index;
  }
}

// Computes whether a point is at least 2 tol from every solid at the start of the step.
bool ComputeIsWellOutsideEverySolid(const SolidContactInputs& inputs, Vec3 point) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    if (!(ComputeDistanceToSolid(solid, ComputeSolidPose(solid, 0), point).distance >= 2 * kTolerance)) {
      return false;
    }
  }
  return true;
}

// Builds the inputs of one generated path in a scene, or none when 50 starts in a row are closer than 2 tol to a solid.
// Draw order: three NextSigned for the start offset (m, up to 1 per axis from the scene's start, drawn again for each
// start that is too close), three for for_this_step and three for for_next_step (m/s, up to 2 per axis from the scene's
// velocities), one Next for the friction (0 to 1).
bool BuildGeneratedPath(const SolidContactInputs& scene_inputs, CaseGenerator& generator, SolidContactInputs* inputs) {
  *inputs = scene_inputs;
  bool is_outside = false;
  for (int draw_index = 0; draw_index < 50 && !is_outside; ++draw_index) {
    inputs->start_position = scene_inputs.start_position
        + Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    is_outside = ComputeIsWellOutsideEverySolid(*inputs, inputs->start_position);
  }
  if (!is_outside) {
    return false;
  }
  const Vec3 this_offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const Vec3 next_offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  inputs->velocity.for_this_step = scene_inputs.velocity.for_this_step + 2 * this_offset;
  inputs->velocity.for_next_step = scene_inputs.velocity.for_next_step + 2 * next_offset;
  inputs->friction = generator.Next();
  return true;
}

// Fails the calling test unless a successful move of a generated path ended outside every solid with a sound record.
void CheckGeneratedSuccess(const SolidContactInputs& inputs, const SolidContactResult& result) {
  CheckEndOutsideEverySolid(inputs, result.position);
  CheckEndOnStartSideOfEveryPlate(inputs, result.position);
  EXPECT_LE(result.hit_count, kMaxHitsPerStep);
  EXPECT_EQ(result.kind == ContactKind::kNone, result.hit_count == 0);
  EXPECT_TRUE(result.hit_fraction >= 0 && result.hit_fraction <= 1);
}

// Runs MoveParticle on one generated path, checks what it returned and counts it. A successful move must end at least
// tol / 2 from every solid; a failure may only be a close call that the scenes can produce, never the end check, since
// no solid in these scenes closes on a particle faster than the hits can push it.
void CheckGeneratedPath(const SolidContactInputs& inputs, std::string_view scene_name, GeneratedCounts& counts) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  ++counts.paths;
  if (error == ContactError::kNone) {
    CheckGeneratedSuccess(inputs, result);
    counts.free_paths += result.hit_count == 0 ? 1 : 0;
    counts.hit_paths += result.hit_count > 0 ? 1 : 0;
    counts.several_hit_paths += result.hit_count > 1 ? 1 : 0;
    return;
  }
  EXPECT_EQ(error, ContactError::kTooCloseToCall) << scene_name;
  EXPECT_NE(failure.close_call, ContactCloseCall::kEndsTooCloseToSolid) << scene_name;
  counts.too_many_hits += failure.close_call == ContactCloseCall::kTooManyHits ? 1 : 0;
  ++counts.failures_per_scene[scene_name].at(static_cast<std::size_t>(failure.close_call));
  counts.second_reacting_solid += failure.close_call == ContactCloseCall::kSecondReactingSolid ? 1 : 0;
  counts.hover_near_turning_solid += failure.close_call == ContactCloseCall::kHoverNearTurningSolid ? 1 : 0;
  counts.near_other_solid += failure.close_call == ContactCloseCall::kTurningSolidNearOtherSolid ? 1 : 0;
  counts.squeezed += failure.close_call == ContactCloseCall::kSqueezedBetweenSolids ? 1 : 0;
}

TEST(MoveParticle, GeneratedPathsInEveryScene_EndOutsideEverySolid) {
  // For every scene of BuildMoveParticleCases whose solids are in id order: 1500 paths (BuildGeneratedPath), no
  // committed plate, the test contact tolerance. A scene with a step or a start out of range uses 0.2 s and
  // {0.5, 0.5, 0.5}.
  CaseGenerator generator;
  GeneratedCounts counts;
  constexpr int kPathsPerScene = 1500;
  for (const MoveParticleCase& scene : BuildMoveParticleCases()) {
    SolidContactInputs scene_inputs = ToSolidContactInputs(scene);
    scene_inputs.plate_side = {};
    scene_inputs.contact_tolerance = kTolerance;
    const bool is_scene_in_range = scene_inputs.step_duration > 0 && math::IsFinite(scene_inputs.start_position);
    if (!is_scene_in_range) {
      scene_inputs.step_duration = 0.2;
      scene_inputs.start_position = {.x = 0.5, .y = 0.5, .z = 0.5};
    }
    if (CheckSolidContactInputs(scene_inputs) != ContactError::kNone) {
      continue;  // The scene about solids out of id order.
    }
    for (int path_index = 0; path_index < kPathsPerScene; ++path_index) {
      SolidContactInputs inputs;
      if (BuildGeneratedPath(scene_inputs, generator, &inputs)) {
        CheckGeneratedPath(inputs, scene.name, counts);
      }
    }
  }
  // The counts keep the set honest: most paths succeed, many hit, and several hits are common. Measured: 314,546 paths,
  // 72,610 hits, 7,172 with several. kTooManyHits comes from the octagon room alone (476); the turning scenes, 818
  // before the carry, have none: their kTurningSolidNearOtherSolid (517) come from the scenes with a turning box near
  // another solid. The scenes of one sphere or one cylinder, last in the list, are mostly missed by
  // paths drawn within 1 m of their start, so a fifth of the paths hit, not a quarter.
  std::cout << "paths " << counts.paths << ", free " << counts.free_paths << ", hit " << counts.hit_paths
            << ", several hits " << counts.several_hit_paths << ", too many hits " << counts.too_many_hits
            << ", second reacting solid " << counts.second_reacting_solid << ", hover near turning solid "
            << counts.hover_near_turning_solid << ", turning solid near other solid " << counts.near_other_solid
            << ", squeezed " << counts.squeezed << "\n";
  WriteFailuresPerScene(counts.failures_per_scene);
  EXPECT_GT(counts.paths, 40000);
  EXPECT_GT(counts.hit_paths, counts.paths / 5);
  EXPECT_GT(counts.several_hit_paths, counts.paths / 60);
}

// Where a generated start near a solid lies, by its distance d to the chosen solid at time 0.
enum class NearStartKind : std::uint8_t {
  kWithinTolerance,  // 0 <= d < tol.
  kOnSurface,  // d = 0.
  kSlightlyInside,  // -tol <= d < -tol / 4.
};

// What the generated starts near a solid gave, counted for the report.
struct NearStartCounts {
  std::array<int, 3> starts_per_kind{};  // In NearStartKind order.
  int free_paths = 0;  // Success with kind kNone.
  int hit_paths = 0;  // Success with kind kHit.
  int started_inside_paths = 0;  // Success with kind kStartedInsideSolid.
  int too_many_hits = 0;
  int second_reacting_solid = 0;
  int hover_near_turning_solid = 0;
  int near_other_solid = 0;  // kTurningSolidNearOtherSolid.
  int squeezed = 0;
  std::map<std::string_view, std::array<int, 8>> failures_per_scene;  // Per close call, in ContactCloseCall order.
};

// Computes whether a point is at least 4 tol from every solid but one at the start of the step.
bool ComputeIsFarFromOtherSolids(const SolidContactInputs& inputs, std::uint32_t near_solid_index, Vec3 point) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    const double distance = ComputeDistanceToSolid(solid, ComputeSolidPose(solid, 0), point).distance;
    if (solid_index != near_solid_index && !(distance >= 4 * kTolerance)) {
      return false;
    }
  }
  return true;
}

// Builds the inputs of one generated start near a solid of a scene, or none when the scene has no solid or 50 points
// in a row cannot be placed. Draw order: one Next for the solid (index floor(Next * solid_count)), one for the kind
// (floor(3 * Next) in NearStartKind order), one for the target distance (tol * Next within the tolerance, unused on
// the surface, -(0.25 + 0.75 * Next) tol inside); then, for each try, three NextSigned for a point up to 1 m per axis
// from the scene's start, moved along that solid's normal at time 0 to the target distance and kept when its distance
// to the solid is within tol / 8 of the target and every other solid is at least 4 tol away; then the velocities and
// the friction as BuildGeneratedPath draws them. The start overlap tolerance is 2 tol, so no start is too deep.
bool BuildGeneratedNearStart(
    const SolidContactInputs& scene_inputs,
    CaseGenerator& generator,
    SolidContactInputs* inputs,
    NearStartKind* start_kind
) {
  if (scene_inputs.solid_count == 0) {
    return false;
  }
  *inputs = scene_inputs;
  inputs->start_overlap_tolerance = 2 * kTolerance;
  const auto near_solid_index =
      static_cast<std::uint32_t>(std::floor(generator.Next() * static_cast<double>(scene_inputs.solid_count)));
  *start_kind = static_cast<NearStartKind>(static_cast<int>(std::floor(3 * generator.Next())));
  const double target_draw = generator.Next();
  double target_distance = 0;  // m, on the surface.
  if (*start_kind == NearStartKind::kWithinTolerance) {
    target_distance = kTolerance * target_draw;
  } else if (*start_kind == NearStartKind::kSlightlyInside) {
    target_distance = -(0.25 + 0.75 * target_draw) * kTolerance;
  }
  const Solid& near_solid = scene_inputs.solids[near_solid_index];
  const SolidPose start_pose = ComputeSolidPose(near_solid, 0);
  bool is_placed = false;
  for (int draw_index = 0; draw_index < 50 && !is_placed; ++draw_index) {
    const Vec3 point = scene_inputs.start_position
        + Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const DistanceToSolid at_point = ComputeDistanceToSolid(near_solid, start_pose, point);
    inputs->start_position = point + (target_distance - at_point.distance) * at_point.normal_out_of_solid;
    const double start_distance = ComputeDistanceToSolid(near_solid, start_pose, inputs->start_position).distance;
    is_placed = std::abs(start_distance - target_distance) <= kTolerance / 8
        && ComputeIsFarFromOtherSolids(*inputs, near_solid_index, inputs->start_position);
  }
  if (!is_placed) {
    return false;
  }
  const Vec3 this_offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const Vec3 next_offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  inputs->velocity.for_this_step = scene_inputs.velocity.for_this_step + 2 * this_offset;
  inputs->velocity.for_next_step = scene_inputs.velocity.for_next_step + 2 * next_offset;
  inputs->friction = generator.Next();
  return true;
}

// Fails the calling test unless the end is on the side of every plate the start rules leave the particle on: the
// start's side, or, from inside the thickness, the side of the nearer face.
void CheckEndOnNearerSideOfEveryPlate(const SolidContactInputs& inputs, Vec3 end_position) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    if (solid.shape != SolidShape::kPlate) {
      continue;
    }
    const SolidPose start_pose = ComputeSolidPose(solid, 0);
    SolidSide start_side = FindSolidSide(solid, start_pose, inputs.start_position);
    if (start_side == SolidSide::kNoPlate) {
      const SolidFace nearer_face = ComputeDistanceToSolid(solid, start_pose, inputs.start_position).face;
      start_side = nearer_face == SolidFace::kHighZ ? SolidSide::kAlongNormal : SolidSide::kAgainstNormal;
    }
    const SolidSide end_side = FindSolidSide(solid, ComputeSolidPose(solid, inputs.step_duration), end_position);
    EXPECT_EQ(end_side, start_side) << "plate index " << solid_index;
  }
}

// Fails the calling test unless a successful move from a generated start near a solid ended outside every solid, on
// the side of every plate the start rules leave it on, with a sound record.
void CheckNearStartSuccess(
    const SolidContactInputs& inputs,
    const SolidContactResult& result,
    NearStartKind start_kind,
    std::string_view scene_name
) {
  CheckEndOutsideEverySolid(inputs, result.position);
  CheckEndOnNearerSideOfEveryPlate(inputs, result.position);
  EXPECT_EQ(result.kind == ContactKind::kNone, result.hit_count == 0) << scene_name;
  // A start inside the solid is always a hit at 0 that sets the kind.
  if (start_kind == NearStartKind::kSlightlyInside) {
    EXPECT_EQ(result.kind, ContactKind::kStartedInsideSolid) << scene_name;
  }
}

// Runs MoveParticle on one generated start near a solid, checks what it returned and counts it. A success ends at
// least tol / 2 from every solid; a failure is a close call that the scenes can produce, never the end check, since
// the start rules put every start at the rest gap, and never a start too deep.
void CheckGeneratedNearStart(
    const SolidContactInputs& inputs,
    NearStartKind start_kind,
    std::string_view scene_name,
    NearStartCounts& counts
) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  ++counts.starts_per_kind.at(static_cast<std::size_t>(start_kind));
  if (error == ContactError::kNone) {
    CheckNearStartSuccess(inputs, result, start_kind, scene_name);
    counts.free_paths += result.kind == ContactKind::kNone ? 1 : 0;
    counts.hit_paths += result.kind == ContactKind::kHit ? 1 : 0;
    counts.started_inside_paths += result.kind == ContactKind::kStartedInsideSolid ? 1 : 0;
    return;
  }
  EXPECT_EQ(error, ContactError::kTooCloseToCall) << scene_name;
  EXPECT_NE(failure.close_call, ContactCloseCall::kEndsTooCloseToSolid) << scene_name;
  counts.too_many_hits += failure.close_call == ContactCloseCall::kTooManyHits ? 1 : 0;
  ++counts.failures_per_scene[scene_name].at(static_cast<std::size_t>(failure.close_call));
  counts.second_reacting_solid += failure.close_call == ContactCloseCall::kSecondReactingSolid ? 1 : 0;
  counts.hover_near_turning_solid += failure.close_call == ContactCloseCall::kHoverNearTurningSolid ? 1 : 0;
  counts.near_other_solid += failure.close_call == ContactCloseCall::kTurningSolidNearOtherSolid ? 1 : 0;
  counts.squeezed += failure.close_call == ContactCloseCall::kSqueezedBetweenSolids ? 1 : 0;
}

// Prints the counts of the generated starts near a solid for the report, and fails the calling test unless each kind
// of start is common and the moves end free, with a hit and started inside, each often.
void CheckNearStartCounts(const NearStartCounts& counts) {
  const std::array<int, 3>& starts = counts.starts_per_kind;
  std::cout << "near starts: within tolerance " << starts.at(0) << ", on surface " << starts.at(1)
            << ", slightly inside " << starts.at(2) << "; free " << counts.free_paths << ", hit " << counts.hit_paths
            << ", started inside " << counts.started_inside_paths << ", too many hits " << counts.too_many_hits
            << ", second reacting solid " << counts.second_reacting_solid << ", hover near turning solid "
            << counts.hover_near_turning_solid << ", turning solid near other solid " << counts.near_other_solid
            << ", squeezed " << counts.squeezed << "\n";
  WriteFailuresPerScene(counts.failures_per_scene);
  EXPECT_GT(starts.at(0), 15000);
  EXPECT_GT(starts.at(1), 15000);
  EXPECT_GT(starts.at(2), 15000);
  EXPECT_GT(counts.free_paths, 5000);
  EXPECT_GT(counts.hit_paths, 10000);
  EXPECT_GT(counts.started_inside_paths, 10000);
}

// Builds the inputs a generated set draws around in one scene: no committed plate, the test contact tolerance, and 0.2
// s and {0.5, 0.5, 0.5} for a scene whose step or start is out of range. Says false for the scene the generated sets
// leave out: the one about solids out of id order.
bool BuildGeneratedSceneInputs(const MoveParticleCase& scene, SolidContactInputs* scene_inputs) {
  *scene_inputs = ToSolidContactInputs(scene);
  scene_inputs->plate_side = {};
  scene_inputs->contact_tolerance = kTolerance;
  const bool is_scene_in_range = scene_inputs->step_duration > 0 && math::IsFinite(scene_inputs->start_position);
  if (!is_scene_in_range) {
    scene_inputs->step_duration = 0.2;
    scene_inputs->start_position = {.x = 0.5, .y = 0.5, .z = 0.5};
  }
  return CheckSolidContactInputs(*scene_inputs) == ContactError::kNone;
}

TEST(MoveParticle, GeneratedStartsNearSolidsInEveryScene_EndOutsideEverySolid) {
  // For every scene of BuildMoveParticleCases with a solid, in id order: 1000 starts (BuildGeneratedNearStart) within
  // the tolerance of, on, or slightly inside one solid and far from the others, no committed plate, the test contact
  // tolerance. A scene with a step or a start out of range uses 0.2 s and {0.5, 0.5, 0.5}. Its kTooManyHits come from
  // the octagon room (100 m/s) and two settle scenes (one each); the four turning scenes ended every start in
  // kTooManyHits before the carry (4,000). Its kSecondReactingSolid come from the scenes with two reacting solids; its
  // kSqueezedBetweenSolids from the scenes whose solids close on the particle; its kTurningSolidNearOtherSolid (1,677)
  // from the scenes with a turning box near another solid.
  CaseGenerator generator;
  NearStartCounts counts;
  constexpr int kStartsPerScene = 1000;
  for (const MoveParticleCase& scene : BuildMoveParticleCases()) {
    SolidContactInputs scene_inputs;
    const bool is_usable_scene = BuildGeneratedSceneInputs(scene, &scene_inputs);
    for (int start_index = 0; is_usable_scene && start_index < kStartsPerScene; ++start_index) {
      SolidContactInputs inputs;
      NearStartKind start_kind = NearStartKind::kOnSurface;
      if (BuildGeneratedNearStart(scene_inputs, generator, &inputs, &start_kind)) {
        CheckGeneratedNearStart(inputs, start_kind, scene.name, counts);
      }
    }
  }
  CheckNearStartCounts(counts);
}

// One path of the parallel set: its two flat solids and its inputs, which point at them.
struct ParallelPath {
  std::array<Solid, 2> solids{};
  SolidContactInputs inputs{};
};

// Builds one path of the parallel set, or none when its velocity approaches the normal slower than 0.05 m/s: two
// parallel fixed flat solids, both reacting, the second delta = (0.05 + 0.9 * Next) tol beyond the first along a
// shared random normal n (three draws, NextSigned, NextSigned and 0.2 + 0.8 * Next, drawn again while shorter than
// 0.1); a base point 0 or (0.001, 0.3 or 1 m, by floor(3 * Next)) times (0.5 + Next) on each axis when Next < 0.5; a
// velocity of three NextSigned times 3 m/s turned to approach n; a start Next * (tol - delta) beyond the second, so
// within tol of both and inside neither; T of 0.01, 0.1, 0.5 or 1 s by floor(4 * Next); friction 0. Draws in that
// order. inputs.solids is left for the caller.
bool BuildParallelPath(CaseGenerator& generator, ParallelPath* path) {
  Vec3 direction{};
  while (math::Length(direction) < 0.1) {
    direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = 0.2 + 0.8 * generator.Next()};
  }
  const Vec3 normal = math::Normalized(direction);
  const double delta = (0.05 + 0.9 * generator.Next()) * kTolerance;  // m
  const bool has_offset = generator.Next() < 0.5;
  const std::array<double, 3> offsets = {0.001, 0.3, 1.0};  // m
  const double offset = offsets.at(static_cast<std::size_t>(std::floor(3 * generator.Next())));
  const double offset_scale = 0.5 + generator.Next();
  const double base_axis = has_offset ? offset * offset_scale : 0;  // m
  const Vec3 base = {.x = base_axis, .y = base_axis, .z = base_axis};
  Vec3 velocity = 3 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const double start_share = generator.Next();
  const std::array<double, 4> step_durations = {0.01, 0.1, 0.5, 1.0};  // s
  const double step_duration = step_durations.at(static_cast<std::size_t>(std::floor(4 * generator.Next())));
  if (math::Dot(velocity, normal) >= 0) {
    velocity = velocity - 2 * math::Dot(velocity, normal) * normal;
  }
  if (std::abs(math::Dot(velocity, normal)) < 0.05) {
    return false;
  }
  path->solids.at(0) = move_particle_cases::BuildFlatSolid(base, normal, move_particle_cases::kNoMotion, 1);
  path->solids.at(1) =
      move_particle_cases::BuildFlatSolid(base + delta * normal, normal, move_particle_cases::kNoMotion, 2);
  path->solids.at(0).reports_reaction = true;
  path->solids.at(1).reports_reaction = true;
  path->inputs.solid_count = 2;
  path->inputs.contact_tolerance = kTolerance;
  path->inputs.step_duration = step_duration;
  path->inputs.start_overlap_tolerance = 2 * kTolerance;
  path->inputs.start_position = base + (delta + start_share * (kTolerance - delta)) * normal;
  path->inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  return true;
}

TEST(MoveParticle, GeneratedParallelSolids_RoundingNeverHitsTheSecond) {
  // A set of 20,000 draws (BuildParallelPath). The start moves to the second solid's rest gap, a few rounding
  // units on either side of it; the first solid responds, and the second is moved silently. Any further hit, or a
  // second reacting solid, is rounding. The start is drawn beyond the second face, inside neither solid, not between
  // the two faces: each solid is classified at the start as given, so a start between the faces would be a real start
  // inside the second, with its own response.
  CaseGenerator generator;
  int paths = 0;
  int spurious_results = 0;
  int other_failures = 0;
  for (int draw_index = 0; draw_index < 20000; ++draw_index) {
    ParallelPath path;
    if (!BuildParallelPath(generator, &path)) {
      continue;
    }
    path.inputs.solids = path.solids.data();
    SolidContactResult result;
    ContactFailure failure;
    const ContactError error = MoveParticle(path.inputs, &result, &failure);
    ++paths;
    const bool is_repeat_hit = error == ContactError::kNone && result.hit_count != 1;
    const bool is_second_reacting = failure.close_call == ContactCloseCall::kSecondReactingSolid;
    spurious_results += is_repeat_hit || is_second_reacting ? 1 : 0;
    other_failures += error != ContactError::kNone && !is_second_reacting ? 1 : 0;
  }
  std::cout << "parallel solids " << paths << ", spurious results " << spurious_results << ", other failures "
            << other_failures << "\n";
  EXPECT_EQ(spurious_results, 0);
  EXPECT_EQ(other_failures, 0);
  EXPECT_GT(paths, 15000);
}

// Where a generated set of starts lies and which tolerance it uses: near the origin with the test tolerance, as the
// analytic tests, or far from the origin with the tolerance ComputeContactTolerance gives there.
struct GeneratedSetPlace {
  double contact_tolerance = kTolerance;  // m
  Vec3 origin{};  // m, world: added to every point the set builds, so {0, 0, 0} keeps the set's bits.
  double size_factor = 1;  // Multiplies every radius, half height and the domain box's width.
};

// Far from the origin: about 830 m from it, with the production tolerance of a scene 1100 m across, and the same sizes.
constexpr GeneratedSetPlace kFarFromOrigin = {
    .contact_tolerance = kFarFromOriginContactTolerance,
    .origin = {.x = -640, .y = 410, .z = 300},
    .size_factor = 1,
};

// Far from the origin with every radius 2000 times larger, from 100 to 1000 m, so R / tol reaches about 2^40. Its
// solids reach up to 3720 m from the origin (a lying cylinder 2000 m long), where ComputeContactTolerance would give
// up to 3.4e-9 m: the tolerance of a scene 1100 m across is 3.4 times smaller, so a coordinate's rounding unit is a
// larger share of it (tol / 2200 instead of tol / 9000), a harder test than the scene's own tolerance would make.
constexpr GeneratedSetPlace kFarFromOriginLargeSolids = {
    .contact_tolerance = kFarFromOriginContactTolerance,
    .origin = {.x = -640, .y = 410, .z = 300},
    .size_factor = 2000,
};

// The domain box of a generated set: from place.origin to place.origin + place.size_factor {1, 1, 1}, open top. Near
// the origin it is BuildUnitDomainBox.
Solid BuildPlacedDomainBox(const GeneratedSetPlace& place) {
  const Vec3 high_corner = place.origin + place.size_factor * Vec3{.x = 1, .y = 1, .z = 1};
  return ToSolid(DomainBox{.low_corner = place.origin, .high_corner = high_corner, .closed = {}}, 1);
}

// What the generated two-surface starts gave, counted for the report.
struct TwoSurfaceCounts {
  int wedge_starts = 0;
  int acute_wedge_starts = 0;  // Normals more than a right angle apart: Dot(n_A, n_B) < 0.
  int corner_starts = 0;
  int free_paths = 0;  // Success with kind kNone.
  int hit_paths = 0;  // Success with kind kHit.
  int started_inside_paths = 0;  // Success with kind kStartedInsideSolid.
  int one_hit_paths = 0;  // Success with one hit: the start's joint response, or one hit along the path.
  int two_hit_paths = 0;  // Success with two hits.
  int failures = 0;
};

// Draws a velocity for a two-surface start: slow, up to tol / T per axis, when Next < 0.5, else up to 2 m/s per axis
// (four draws).
Vec3 BuildTwoSurfaceVelocity(CaseGenerator& generator, double step_duration, double contact_tolerance) {
  const double speed = generator.Next() < 0.5 ? contact_tolerance / step_duration : 2.0;  // m/s
  return speed * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
}

// Builds a wedge: two fixed flat solids through a point q within 1 m of the origin per axis, normals nA (three
// NextSigned, drawn again while shorter than 0.1) and nB = cos(phi) nA + sin(phi) t, phi from 5 to 175 degrees (a
// wedge angle of 175 down to 5) and t a random direction made square to nA (three NextSigned, drawn again while that
// part is shorter than 0.1). The start is at distances dA and dB, each (2 * Next - 1) tol, and c = NextSigned m along
// nA x t. Draw order: nA, t, phi, q, dA, dB, c. The point q is moved by place.origin and tol is place's.
std::array<Solid, 2>
BuildTwoSurfaceWedge(CaseGenerator& generator, const GeneratedSetPlace& place, Vec3* start_position) {
  Vec3 first_direction{};
  while (math::Length(first_direction) < 0.1) {
    first_direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  const Vec3 first_normal = math::Normalized(first_direction);
  Vec3 square_direction{};
  while (math::Length(square_direction) < 0.1) {
    const Vec3 draw = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    square_direction = draw - math::Dot(first_normal, draw) * first_normal;
  }
  const Vec3 square_unit = math::Normalized(square_direction);
  const double normal_angle = (5 + 170 * generator.Next()) * std::numbers::pi / 180;  // rad
  const Vec3 second_normal = std::cos(normal_angle) * first_normal + std::sin(normal_angle) * square_unit;
  const Vec3 point =
      place.origin + Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const double first_distance = generator.NextSigned() * place.contact_tolerance;  // m
  const double second_distance = generator.NextSigned() * place.contact_tolerance;  // m
  const double along_line = generator.NextSigned();  // m
  const double square_share =
      (second_distance - std::cos(normal_angle) * first_distance) / std::sin(normal_angle);  // m
  const Vec3 line_direction = {
      .x = first_normal.y * square_unit.z - first_normal.z * square_unit.y,
      .y = first_normal.z * square_unit.x - first_normal.x * square_unit.z,
      .z = first_normal.x * square_unit.y - first_normal.y * square_unit.x,
  };
  *start_position = point + first_distance * first_normal + square_share * square_unit + along_line * line_direction;
  const SolidMotion no_motion = move_particle_cases::kNoMotion;
  return {
      move_particle_cases::BuildFlatSolid(point, first_normal, no_motion, 1),
      move_particle_cases::BuildFlatSolid(point, math::Normalized(second_normal), no_motion, 2),
  };
}

// Builds a start in a corner of the unit domain box with an open top: two or three near sides (floor(2 * Next) + 2),
// the axes x, y, z in turn when three, else the two from floor(3 * Next) left out; per near axis the low or high side
// by Next < 0.5 (side_bottom for z, the top being open) and a distance (2 * Next - 1) tol; every other axis at
// 0.2 + 0.6 * Next m. In place's domain box (BuildPlacedDomainBox), with every length but the distance scaled by its
// size factor and tol place's.
Vec3 BuildDomainCornerStart(CaseGenerator& generator, const GeneratedSetPlace& place) {
  const bool has_three_sides = generator.Next() < 0.5;
  const int axis_left_out = has_three_sides ? -1 : static_cast<int>(std::floor(3 * generator.Next()));
  std::array<double, 3> coordinates{};
  for (int axis_index = 0; axis_index < 3; ++axis_index) {
    const bool is_low_side = generator.Next() < 0.5 || axis_index == 2;
    const double distance = generator.NextSigned() * place.contact_tolerance;  // m, into the box.
    const double inner = place.size_factor * (0.2 + 0.6 * generator.Next());  // m
    const double near_side = is_low_side ? distance : place.size_factor - distance;  // m
    coordinates.at(static_cast<std::size_t>(axis_index)) = axis_index == axis_left_out ? inner : near_side;
  }
  return place.origin + Vec3{.x = coordinates.at(0), .y = coordinates.at(1), .z = coordinates.at(2)};
}

// Runs MoveParticle on one two-surface start, checks it and counts it: it succeeds, and ends at least tol / 2 from
// every solid. With one response per surface in turn, an acute wedge failed with kTooManyHits in 30 % of the draws: the
// response on one face turned the velocity into the other, and the particle hit one and then the other. It counts the
// hits too: a start on two surfaces is one hit (method paper <sec-move-particle>), and a second comes only along the
// path.
void CheckTwoSurfaceStart(const SolidContactInputs& inputs, TwoSurfaceCounts& counts) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  if (error != ContactError::kNone) {
    ADD_FAILURE() << "error " << static_cast<int>(error) << ", close call " << static_cast<int>(failure.close_call);
    ++counts.failures;
    return;
  }
  CheckEndOutsideEverySolid(inputs, result.position);
  counts.free_paths += result.kind == ContactKind::kNone ? 1 : 0;
  counts.hit_paths += result.kind == ContactKind::kHit ? 1 : 0;
  counts.started_inside_paths += result.kind == ContactKind::kStartedInsideSolid ? 1 : 0;
  counts.one_hit_paths += result.hit_count == 1 ? 1 : 0;
  counts.two_hit_paths += result.hit_count == 2 ? 1 : 0;
}

// Runs the 30,000 draws of the two-surface set: by Next < 0.5 a wedge (BuildTwoSurfaceWedge, wedge angles 5 to 175
// degrees) or a domain corner (BuildDomainCornerStart); then T = 0.01 + 0.09 * Next s, short enough that a fast
// particle stays near its corner, away from the far sides, both velocities from one BuildTwoSurfaceVelocity, friction
// Next. With moving_wedges, each wedge solid then gets a linear_velocity of three NextSigned m/s, solid A then B; the
// domain box stays fixed. Start overlap tolerance 2 tol. Every start is within tol of, on, or slightly inside two or
// three surfaces. The set lies at place (GeneratedSetPlace).
TwoSurfaceCounts ComputeTwoSurfaceStarts(bool moving_wedges, const GeneratedSetPlace& place) {
  CaseGenerator generator;
  TwoSurfaceCounts counts;
  const std::array domain_solids = {BuildPlacedDomainBox(place)};
  for (int draw_index = 0; draw_index < 30000; ++draw_index) {
    const bool is_wedge = generator.Next() < 0.5;
    SolidContactInputs inputs;
    inputs.contact_tolerance = place.contact_tolerance;
    inputs.start_overlap_tolerance = 2 * place.contact_tolerance;
    std::array<Solid, 2> wedge_solids{};
    if (is_wedge) {
      wedge_solids = BuildTwoSurfaceWedge(generator, place, &inputs.start_position);
      inputs.solids = wedge_solids.data();
      inputs.solid_count = 2;
      ++counts.wedge_starts;
      const Mat3 first_axes = wedge_solids.at(0).pose_at_step_start.axes;
      const Mat3 second_axes = wedge_solids.at(1).pose_at_step_start.axes;
      const bool is_acute_wedge =
          first_axes.xz * second_axes.xz + first_axes.yz * second_axes.yz + first_axes.zz * second_axes.zz < 0;
      counts.acute_wedge_starts += is_acute_wedge ? 1 : 0;
    } else {
      inputs.start_position = BuildDomainCornerStart(generator, place);
      inputs.solids = domain_solids.data();
      inputs.solid_count = 1;
      ++counts.corner_starts;
    }
    inputs.step_duration = 0.01 + 0.09 * generator.Next();
    const Vec3 velocity = BuildTwoSurfaceVelocity(generator, inputs.step_duration, place.contact_tolerance);
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    inputs.friction = generator.Next();
    for (Solid& wedge_solid : wedge_solids) {
      if (moving_wedges && is_wedge) {
        wedge_solid.motion.linear_velocity = {
            .x = generator.NextSigned(),
            .y = generator.NextSigned(),
            .z = generator.NextSigned(),
        };
      }
    }
    CheckTwoSurfaceStart(inputs, counts);
  }
  const bool is_far = place.origin.x != 0 || place.origin.y != 0 || place.origin.z != 0;
  std::cout << "two-surface starts" << (moving_wedges ? " with moving wedges" : "")
            << (is_far ? " far from origin" : "") << ": wedges " << counts.wedge_starts << " (acute "
            << counts.acute_wedge_starts << "), domain corners " << counts.corner_starts << "; free "
            << counts.free_paths << ", hit " << counts.hit_paths << ", started inside " << counts.started_inside_paths
            << "; one hit " << counts.one_hit_paths << ", two hits " << counts.two_hit_paths << ", failures "
            << counts.failures << "\n";
  return counts;
}

TEST(MoveParticle, GeneratedTwoSurfaceStarts_SucceedDownToFiveDegrees) {
  // Fixed wedges and corners (ComputeTwoSurfaceStarts): no failure at all (the model: 0 of 10,001 acute draws). Before
  // the slide over several surfaces, 2246 of 7497 acute draws ended in kTooManyHits.
  const TwoSurfaceCounts counts = ComputeTwoSurfaceStarts(false, {});
  EXPECT_EQ(counts.failures, 0);
  EXPECT_GT(counts.free_paths, 1000);
  EXPECT_GT(counts.hit_paths, 1000);
  EXPECT_GT(counts.started_inside_paths, 1000);
}

TEST(MoveParticle, GeneratedMovingTwoSurfaceStarts_SucceedDownToFiveDegrees) {
  // The same draws with each wedge solid moving at up to 1 m/s per axis: no failure. Two surfaces that are not
  // parallel always leave a velocity that approaches neither, and from 5 degrees up the escape is at most
  // 1 / sin(2.5 degrees) = 23 times the solids' speeds, below the speed cap (the model: 0 of 35,000).
  const TwoSurfaceCounts counts = ComputeTwoSurfaceStarts(true, {});
  EXPECT_EQ(counts.failures, 0);
  EXPECT_GT(counts.hit_paths, 1000);
}

// --- Generated tangent contacts ---

// The scenes of the generated tangent contacts, in the order of the set.
enum class TangentScene : std::uint8_t {
  kBallOnFloor,  // A ball resting on the fixed floor through the contact point.
  kBallInDomainCorner,  // A ball touching the three low sides of the unit domain box.
  kTwoBalls,  // Two balls touching along x.
  kBallOnTiltedFlatSolid,  // A ball resting on a fixed flat solid tilted about y.
  kCylinderLyingOnFloor,  // A cylinder lying on the fixed floor along a line through the contact point.
  kCylinderStandingInDomainCorner,  // A cylinder standing on the unit domain box's bottom, touching its left and front.
  kRimOnFloor,  // A tilted cylinder resting on the fixed floor at one point of its low rim.
};

// One start of the tangent set: its solids, its inputs (which point at them) and what was drawn for the counts.
struct TangentStart {
  std::array<Solid, 2> solids{};
  SolidContactInputs inputs{};
  int velocity_class = 0;  // 0 at rest, 1 slow, 2 falling, 3 toward the contact, 4 random.
  Vec3 first_normal{};  // No unit, world. The floor's, the side's or the first ball's normal at the start.
  Vec3 second_normal{};  // No unit, world. The second ball's normal at the start.
  Vec3 contact{};  // m, world: the contact point.
  Vec3 contact_normal{};  // No unit, world: out of the floor, the side or the first ball at the contact.
  Vec3 radial{};  // No unit, world: away from the contact, square to contact_normal.
  double rest_distance = 0;  // m: x*, the rest point's distance from the contact.
  double first_radius = 0;  // m: the first ball's radius for two balls, else 0.
};

// Computes the gap from a point of a contact plane to a solid, along the plane's normal: the height above the point at
// which the solid begins, at most 2 tol, found by 64 halvings of [0, 2 tol]. The solid is convex and the point outside
// it, so along the normal its distance is first positive, then not.
double ComputeGapToSolid(const Solid& solid, Vec3 point_on_plane, Vec3 normal, double contact_tolerance) {
  const double largest_gap = 2 * contact_tolerance;  // m
  const auto distance_at_height = [&](double height) {
    return ComputeDistanceToSolid(solid, solid.pose_at_step_start, point_on_plane + height * normal).distance;
  };
  if (distance_at_height(largest_gap) > 0) {
    return largest_gap;
  }
  double outside_height = 0;  // m
  double inside_height = largest_gap;  // m
  for (int halving_index = 0; halving_index < 64; ++halving_index) {
    const double middle_height = (outside_height + inside_height) / 2;  // m
    if (distance_at_height(middle_height) > 0) {
      outside_height = middle_height;
    } else {
      inside_height = middle_height;
    }
  }
  return outside_height;
}

// Builds one start of the tangent set in one scene (CaseGenerator). Draws, in this order and all of them for every
// scene: R = 0.05 + 0.45 Next and the second ball's radius the same; the contact's offset {NextSigned, NextSigned} m;
// the radial direction's angle 2 pi Next; x = x* (0.02 + 0.98 Next), with x* = 2 Sqrt(R_eff tol) the rest point's
// distance and R_eff = R, or R R2 / (R + R2) for two balls; the height in the gap, (1.5 Next - 0.5) of the gap x^2 /
// (2 R_eff), so some starts are up to half the gap inside the floor (inside the first ball for two balls); T from
// {1e-3, 0.01, 0.1} s by floor(3 Next); the velocity class floor(5 Next), three NextSigned and one Next (at rest; slow,
// the three times tol / T; falling, {0, 0, -9.81 T}; toward the contact along the radial direction, 2 Next m/s; random,
// the three times 2 m/s); the scene's own Next (the corner ball's side by floor(3 Next), bottom, left or front; the
// tilt 0.6 Next rad). Both velocities equal, friction 0, start overlap tolerance 2 tol.
// The cylinder scenes read the same draws, with the cylinder's radius R and half height H = R2 (2 R2 lying), and the
// gap measured along the contact normal to the cylinder (ComputeGapToSolid), at most 2 tol, which is the ball's gap at
// x*:
//   lying on the floor  its axis along {cos, sin, 0} of the radial angle; the radial direction square to it in the
//                       floor, on the side the scene's Next gives (below 0.5, the left of the axis).
//   standing in corner  centre {R, R, H}, its low cap on the bottom; the contact is with the left side at {0, R, H}
//                       (radial +y) or the front side at {R, 0, H} (radial +x), by floor(2 Next) of the scene's Next.
//   rim on the floor    its axis tilted about y by 0.1 + 0.5 Next rad of the scene's Next, its lowest rim point on the
//                       floor at the offset; the radial direction is the radial angle's in the floor. x* is the
//                       nominal 2 Sqrt(R tol): the space under the cap is a wedge of the tilt, not a cusp.
// The set lies at place (GeneratedSetPlace): every point moved by its origin, the radii, the half heights and the
// domain box scaled by its size factor (not the offset), and tol its own.
TangentStart BuildTangentStart(CaseGenerator& generator, TangentScene scene, const GeneratedSetPlace& place) {
  const double radius = place.size_factor * (0.05 + 0.45 * generator.Next());  // m
  const double second_radius = place.size_factor * (0.05 + 0.45 * generator.Next());  // m
  const Vec3 offset = place.origin + Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = 0};
  const double radial_angle = 2 * std::numbers::pi * generator.Next();  // rad
  const double distance_share = 0.02 + 0.98 * generator.Next();
  const double height_share = 1.5 * generator.Next() - 0.5;
  const std::array<double, 3> step_durations = {1e-3, 0.01, 0.1};  // s
  const double step_duration = step_durations.at(static_cast<std::size_t>(std::floor(3 * generator.Next())));
  const int velocity_class = static_cast<int>(std::floor(5 * generator.Next()));
  const Vec3 velocity_draw = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const double speed_share = generator.Next();
  const double scene_draw = generator.Next();
  const double cos_radial = std::cos(radial_angle);
  const double sin_radial = std::sin(radial_angle);

  TangentStart start;
  Vec3 contact{};  // m, world.
  Vec3 floor_normal{};  // No unit, world: out of the floor, the side or the first ball at the contact.
  Vec3 radial{};  // No unit, world: away from the contact, square to floor_normal.
  double effective_radius = radius;  // m
  switch (scene) {
    case TangentScene::kBallOnFloor:
    case TangentScene::kBallOnTiltedFlatSolid: {
      const double tilt = scene == TangentScene::kBallOnTiltedFlatSolid ? 0.6 * scene_draw : 0;  // rad
      floor_normal = {.x = std::sin(tilt), .y = 0, .z = std::cos(tilt)};
      const Vec3 along_tilt = {.x = std::cos(tilt), .y = 0, .z = -std::sin(tilt)};
      radial = cos_radial * along_tilt + sin_radial * Vec3{.x = 0, .y = 1, .z = 0};
      contact = offset;
      start.solids.at(0) =
          move_particle_cases::BuildFlatSolid(contact, floor_normal, move_particle_cases::kNoMotion, 1);
      start.solids.at(1) = move_particle_cases::BuildFixedSphere(contact + radius * floor_normal, radius, 2);
      break;
    }
    case TangentScene::kBallInDomainCorner: {
      const int side_index = static_cast<int>(std::floor(3 * scene_draw));
      const std::array<Vec3, 3> axes = {
          Vec3{.x = 0, .y = 0, .z = 1},
          Vec3{.x = 1, .y = 0, .z = 0},
          Vec3{.x = 0, .y = 1, .z = 0},
      };
      floor_normal = axes.at(static_cast<std::size_t>(side_index));
      radial = cos_radial * axes.at(static_cast<std::size_t>((side_index + 1) % 3))
          + sin_radial * axes.at(static_cast<std::size_t>((side_index + 2) % 3));
      const Vec3 center = place.origin + Vec3{.x = radius, .y = radius, .z = radius};
      contact = center - radius * floor_normal;
      start.solids.at(0) = BuildPlacedDomainBox(place);
      start.solids.at(1) = move_particle_cases::BuildFixedSphere(center, radius, 2);
      break;
    }
    case TangentScene::kTwoBalls: {
      effective_radius = radius * second_radius / (radius + second_radius);
      floor_normal = {.x = 1, .y = 0, .z = 0};
      radial = {.x = 0, .y = cos_radial, .z = sin_radial};
      contact = offset + Vec3{.x = radius, .y = 0, .z = 0};
      start.solids.at(0) = move_particle_cases::BuildFixedSphere(offset, radius, 1);
      start.solids.at(1) = move_particle_cases::BuildFixedSphere(
          offset + Vec3{.x = radius + second_radius, .y = 0, .z = 0},
          second_radius,
          2
      );
      break;
    }
    case TangentScene::kCylinderLyingOnFloor: {
      floor_normal = {.x = 0, .y = 0, .z = 1};
      const Vec3 along_axis = {.x = cos_radial, .y = sin_radial, .z = 0};
      const double side_sign = scene_draw < 0.5 ? 1.0 : -1.0;
      radial = side_sign * Vec3{.x = -sin_radial, .y = cos_radial, .z = 0};
      contact = offset;
      const Cylinder lying = {
          .center = contact + radius * floor_normal,
          .axis = along_axis,
          .radius = radius,
          .half_height = 2 * second_radius,
      };
      start.solids.at(0) =
          move_particle_cases::BuildFlatSolid(contact, floor_normal, move_particle_cases::kNoMotion, 1);
      start.solids.at(1) = ToSolid(lying, move_particle_cases::kNoMotion, 2);
      break;
    }
    case TangentScene::kCylinderStandingInDomainCorner: {
      const bool touches_left = std::floor(2 * scene_draw) == 0;
      floor_normal = touches_left ? Vec3{.x = 1, .y = 0, .z = 0} : Vec3{.x = 0, .y = 1, .z = 0};
      radial = touches_left ? Vec3{.x = 0, .y = 1, .z = 0} : Vec3{.x = 1, .y = 0, .z = 0};
      const Vec3 center = place.origin + Vec3{.x = radius, .y = radius, .z = second_radius};
      contact = center - radius * floor_normal;
      const Cylinder standing =
          {.center = center, .axis = {.x = 0, .y = 0, .z = 1}, .radius = radius, .half_height = second_radius};
      start.solids.at(0) = BuildPlacedDomainBox(place);
      start.solids.at(1) = ToSolid(standing, move_particle_cases::kNoMotion, 2);
      break;
    }
    case TangentScene::kRimOnFloor: {
      const double tilt = 0.1 + 0.5 * scene_draw;  // rad
      floor_normal = {.x = 0, .y = 0, .z = 1};
      radial = {.x = cos_radial, .y = sin_radial, .z = 0};
      contact = offset;
      const Vec3 axis = {.x = std::sin(tilt), .y = 0, .z = std::cos(tilt)};
      // The direction in the cap's plane that goes down most steeply: the lowest rim point is R along it.
      const Vec3 down_in_cap = {.x = std::cos(tilt), .y = 0, .z = -std::sin(tilt)};
      const Cylinder tilted = {
          .center = contact + second_radius * axis - radius * down_in_cap,
          .axis = axis,
          .radius = radius,
          .half_height = second_radius,
      };
      start.solids.at(0) =
          move_particle_cases::BuildFlatSolid(contact, floor_normal, move_particle_cases::kNoMotion, 1);
      start.solids.at(1) = ToSolid(tilted, move_particle_cases::kNoMotion, 2);
      break;
    }
  }
  const double contact_tolerance = place.contact_tolerance;  // m
  const double rest_distance = 2 * std::sqrt(effective_radius * contact_tolerance);  // m, x*.
  const double contact_distance = distance_share * rest_distance;  // m
  const bool is_cylinder_scene = scene == TangentScene::kCylinderLyingOnFloor
      || scene == TangentScene::kCylinderStandingInDomainCorner || scene == TangentScene::kRimOnFloor;
  const double gap = is_cylinder_scene
      ? ComputeGapToSolid(start.solids.at(1), contact + contact_distance * radial, floor_normal, contact_tolerance)
      : contact_distance * contact_distance / (2 * effective_radius);  // m
  // For two balls the gap is measured from the first ball's surface, which lies x^2 / (2 R) behind the contact plane.
  const double first_surface_height =
      scene == TangentScene::kTwoBalls ? -contact_distance * contact_distance / (2 * radius) : 0;  // m
  const Vec3 start_position =
      contact + contact_distance * radial + (first_surface_height + height_share * gap) * floor_normal;

  Vec3 velocity{};  // m/s
  if (velocity_class == 1) {
    velocity = contact_tolerance / step_duration * velocity_draw;
  } else if (velocity_class == 2) {
    velocity = {.x = 0, .y = 0, .z = -9.81 * step_duration};
  } else if (velocity_class == 3) {
    velocity = -2 * speed_share * radial;
  } else if (velocity_class == 4) {
    velocity = 2 * velocity_draw;
  }
  start.velocity_class = velocity_class;
  start.inputs.start_position = start_position;
  start.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  start.inputs.solid_count = 2;
  start.inputs.step_duration = step_duration;
  start.inputs.contact_tolerance = contact_tolerance;
  start.inputs.start_overlap_tolerance = 2 * contact_tolerance;
  const Solid& first_solid = start.solids.at(0);
  const Solid& second_solid = start.solids.at(1);
  start.first_normal = scene == TangentScene::kTwoBalls
      ? ComputeDistanceToSolid(first_solid, first_solid.pose_at_step_start, start_position).normal_out_of_solid
      : floor_normal;
  start.second_normal =
      ComputeDistanceToSolid(second_solid, second_solid.pose_at_step_start, start_position).normal_out_of_solid;
  start.contact = contact;
  start.contact_normal = floor_normal;
  start.radial = radial;
  start.rest_distance = rest_distance;
  start.first_radius = scene == TangentScene::kTwoBalls ? radius : 0;
  return start;
}

// Runs MoveParticle on one tangent start and fails the calling test unless it succeeded at least tol / 2 from every
// solid. With one response per surface in turn, 37 to 49 % per scene ended in kTooManyHits, bouncing between the two
// surfaces into the cusp.
bool CheckTangentStartSucceeds(TangentScene scene, const SolidContactInputs& inputs) {
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  if (error != ContactError::kNone) {
    ADD_FAILURE() << "scene " << static_cast<int>(scene) << ", error " << static_cast<int>(error) << ", close call "
                  << static_cast<int>(failure.close_call);
    return false;
  }
  CheckEndOutsideEverySolid(inputs, result.position);
  return true;
}

TEST(MoveParticle, GeneratedTangentContactStarts_Succeed) {
  // 2000 starts per scene (BuildTangentStart), the scenes in TangentScene order, starts in the cusp of each tangent
  // contact from 0.02 to 1 times the rest point's distance. Every start succeeds and ends at least tol / 2 from every
  // solid.
  CaseGenerator generator;
  const std::array scenes = {
      TangentScene::kBallOnFloor,
      TangentScene::kBallInDomainCorner,
      TangentScene::kTwoBalls,
      TangentScene::kBallOnTiltedFlatSolid,
  };
  for (const TangentScene scene : scenes) {
    int successes = 0;
    for (int draw_index = 0; draw_index < 2000; ++draw_index) {
      TangentStart start = BuildTangentStart(generator, scene, {});
      start.inputs.solids = start.solids.data();
      successes += CheckTangentStartSucceeds(scene, start.inputs) ? 1 : 0;
    }
    std::cout << "tangent scene " << static_cast<int>(scene) << ": success " << successes << " of 2000\n";
  }
}

// Turns one tangent start into a slide by Next < 1/3: the particle starts tol from the floor (for two balls, from the
// first ball), at x* + (0.05 - x*) Next from the contact along the radial direction, moving toward the contact at
// 2 Next m/s with gravity, {0, 0, -9.81 T} m/s. Otherwise it keeps the start; the two Next are drawn either way.
void ApplyTangentSlide(CaseGenerator& generator, TangentStart* start) {
  const bool is_slide = generator.Next() < 1.0 / 3;
  const double distance_share = generator.Next();
  const double speed_share = generator.Next();
  if (!is_slide) {
    return;
  }
  const double contact_distance = start->rest_distance + (0.05 - start->rest_distance) * distance_share;  // m
  const double first_surface_height =
      start->first_radius > 0 ? -contact_distance * contact_distance / (2 * start->first_radius) : 0;  // m
  start->inputs.start_position = start->contact + contact_distance * start->radial
      + (first_surface_height + start->inputs.contact_tolerance) * start->contact_normal;
  const Vec3 velocity =
      -2 * speed_share * start->radial + Vec3{.x = 0, .y = 0, .z = -9.81 * start->inputs.step_duration};
  start->inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
}

// Sets the motion of one tangent start's solids, each sliding along the other: the first solid (the floor, the tilted
// side or the first ball) at a velocity square to the contact normal, and the ball or the cylinder at that velocity
// plus another square to it and a part along the contact normal of 0, 0, -1e-3 or 1e-3 m/s (sinking or rising) by
// floor(4 Next). The tangent parts are NextSigned m/s along two directions square to the contact normal (radial and its
// cross with the normal), four draws, then the one for the normal part. The ball and the cylinder in the domain corner
// keep still: moving, they would leave or enter the other sides. The draws are made in every scene.
void ApplyTangentMotion(CaseGenerator& generator, TangentScene scene, TangentStart* start) {
  const Vec3 normal = start->contact_normal;
  const Vec3 along = start->radial;
  const Vec3 across = math::Cross(normal, along);
  const Vec3 first_velocity = generator.NextSigned() * along + generator.NextSigned() * across;  // m/s
  const Vec3 slide_velocity = generator.NextSigned() * along + generator.NextSigned() * across;  // m/s
  const std::array<double, 4> normal_speeds = {0, 0, -1e-3, 1e-3};  // m/s
  const double normal_speed = normal_speeds.at(static_cast<std::size_t>(std::floor(4 * generator.Next())));
  if (scene == TangentScene::kBallInDomainCorner || scene == TangentScene::kCylinderStandingInDomainCorner) {
    return;
  }
  start->solids.at(0).motion.linear_velocity = first_velocity;
  start->solids.at(1).motion.linear_velocity = first_velocity + slide_velocity + normal_speed * normal;
}

// What the tangent slides of one scene and motion gave, counted for the report.
struct TangentSlideCounts {
  int successes = 0;
  int too_many_hits = 0;
  // kTooManyHits by the number of halvings of the step after which it first succeeded, 1 to 8; the last, none.
  std::array<int, 9> cured_after_halvings{};
  int squeezed = 0;  // kSqueezedBetweenSolids, only where the ball closes on the other solid.
  int other_failures = 0;
};

// Fails the calling test unless a squeeze of the tangent slides keeps to its can_shorter_step_cure: one at the start
// comes back at a sixteenth of the step, still false; one at a later hit is cured by a step that ends before it, within
// 40 halvings.
void CheckTangentSqueezeKeepsToFlag(SolidContactInputs inputs, const ContactFailure& failure) {
  SolidContactResult result;
  ContactFailure shorter_failure;
  if (!failure.can_shorter_step_cure) {
    inputs.step_duration = inputs.step_duration / 16;
    EXPECT_EQ(MoveParticle(inputs, &result, &shorter_failure), ContactError::kTooCloseToCall);
    EXPECT_EQ(shorter_failure.close_call, ContactCloseCall::kSqueezedBetweenSolids);
    EXPECT_FALSE(shorter_failure.can_shorter_step_cure);
    return;
  }
  bool is_cured = false;
  for (int halving_count = 1; halving_count <= 40 && !is_cured; ++halving_count) {
    inputs.step_duration = inputs.step_duration / 2;
    is_cured = MoveParticle(inputs, &result, &shorter_failure) == ContactError::kNone;
  }
  EXPECT_TRUE(is_cured);
}

// Runs MoveParticle on one tangent slide and counts it: a success ends at least tol / 2 from every solid. A failure may
// only be kTooManyHits, which a shorter step cures (can_shorter_step_cure), run again with the step halved up to eight
// times (1/256) until one succeeds, or kSqueezedBetweenSolids where the ball closes on the other solid (a ball sinking
// into a floor): the speed cap's squeeze, which the geometry decides (method paper <sec-move-particle>): at every step
// length when the start's response meets it, and until a step ends before the hit when a later hit does
// (CheckTangentSqueezeKeepsToFlag).
void CheckTangentSlide(TangentScene scene, const TangentStart& start, TangentSlideCounts& counts) {
  SolidContactInputs inputs = start.inputs;
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(inputs, &result, &failure);
  if (error == ContactError::kNone) {
    CheckEndOutsideEverySolid(inputs, result.position);
    ++counts.successes;
    return;
  }
  const bool is_close_call = error == ContactError::kTooCloseToCall;
  const Vec3 closing_velocity = start.solids.at(1).motion.linear_velocity - start.solids.at(0).motion.linear_velocity;
  const bool is_ball_closing = math::Dot(closing_velocity, start.contact_normal) < 0;
  if (is_close_call && failure.close_call == ContactCloseCall::kSqueezedBetweenSolids && is_ball_closing) {
    CheckTangentSqueezeKeepsToFlag(inputs, failure);
    ++counts.squeezed;
    return;
  }
  if (!is_close_call || failure.close_call != ContactCloseCall::kTooManyHits) {
    ADD_FAILURE() << "scene " << static_cast<int>(scene) << ", error " << static_cast<int>(error) << ", close call "
                  << static_cast<int>(failure.close_call);
    ++counts.other_failures;
    return;
  }
  ++counts.too_many_hits;
  EXPECT_TRUE(failure.can_shorter_step_cure);
  int halving_count = 1;
  for (; halving_count <= 8; ++halving_count) {
    inputs.step_duration = inputs.step_duration / 2;
    if (MoveParticle(inputs, &result, &failure) == ContactError::kNone) {
      break;
    }
  }
  ++counts.cured_after_halvings.at(static_cast<std::size_t>(halving_count - 1));
}

// Prints, for the report, what the tangent slides of one scene and motion gave.
void WriteTangentSlideCounts(TangentScene scene, bool is_moving, const TangentSlideCounts& counts) {
  const std::array<int, 9>& cured = counts.cured_after_halvings;
  std::cout << "tangent slides scene " << static_cast<int>(scene) << (is_moving ? " moving" : " fixed") << ": success "
            << counts.successes << ", too many hits " << counts.too_many_hits << " (cured after 1 to 8 halvings:";
  for (std::size_t halving_index = 0; halving_index < 8; ++halving_index) {
    std::cout << " " << cured.at(halving_index);
  }
  std::cout << "; not cured " << cured.at(8) << "), squeezed " << counts.squeezed << ", other failures "
            << counts.other_failures << "\n";
}

// Runs 2000 draws of the tangent slides of one scene: BuildTangentStart at place, then ApplyTangentSlide, then
// ApplyTangentMotion, applied only when is_moving, and CheckTangentSlide.
TangentSlideCounts
ComputeTangentSlides(CaseGenerator& generator, TangentScene scene, bool is_moving, const GeneratedSetPlace& place) {
  TangentSlideCounts counts;
  for (int draw_index = 0; draw_index < 2000; ++draw_index) {
    TangentStart start = BuildTangentStart(generator, scene, place);
    ApplyTangentSlide(generator, &start);
    TangentStart moving_start = start;
    ApplyTangentMotion(generator, scene, &moving_start);
    TangentStart& used = is_moving ? moving_start : start;
    used.inputs.solids = used.solids.data();
    CheckTangentSlide(scene, used, counts);
  }
  return counts;
}

// Fails the calling test unless the slides of one scene had no other failure, every kTooManyHits was cured within
// eight halvings, and at least 1950 of the 2000 draws succeeded (1958 in the worst scene), so the checks are not
// vacuous.
void CheckTangentSlideCounts(const TangentSlideCounts& counts) {
  EXPECT_EQ(counts.other_failures, 0);
  EXPECT_EQ(counts.cured_after_halvings.at(8), 0);
  EXPECT_GE(counts.successes, 1950);
}

TEST(MoveParticle, GeneratedTangentContactSlides_FailOnlyWhereAShorterStepCures) {
  // 2000 draws per scene, in TangentScene order, fixed and then moving: BuildTangentStart, then ApplyTangentSlide, then
  // ApplyTangentMotion (drawn for the fixed ones too, not applied). A third of the draws slide into the cusp from up to
  // 5 cm at up to 2 m/s with gravity; the others start in it. Every success ends at least tol / 2 from every solid; the
  // only failures are kTooManyHits, each cured by a step at most 256 times shorter, and the speed cap's squeeze of a
  // ball sinking into a floor (CheckTangentSlide). Measured: two balls are the only kTooManyHits, a fast particle
  // zig-zagging into their neck, 42 of 2000 fixed draws (steps of 0.1 s moving it up to 28 cm), 38 cured by 1/16 and
  // the others by 1/32 to 1/256; the model's set, with other draws, cured all 31 of 40,000 by 1/16. The 256 holds for
  // these draws only: a rule that responds hit by hit, four hits a step, reaches the neck's pair only once both balls
  // are within tol at once, so the halvings needed grow with the speed over Sqrt(R tol), and faster drops or other
  // draws can need more.
  CaseGenerator generator;
  const std::array scenes = {
      TangentScene::kBallOnFloor,
      TangentScene::kBallInDomainCorner,
      TangentScene::kTwoBalls,
      TangentScene::kBallOnTiltedFlatSolid,
  };
  for (const bool is_moving : {false, true}) {
    for (const TangentScene scene : scenes) {
      const TangentSlideCounts counts = ComputeTangentSlides(generator, scene, is_moving, {});
      WriteTangentSlideCounts(scene, is_moving, counts);
      CheckTangentSlideCounts(counts);
    }
  }
}

TEST(MoveParticle, GeneratedCylinderTangentContactStarts_Succeed) {
  // The starts of GeneratedTangentContactStarts_Succeed in the cylinder scenes, 2000 per scene in TangentScene order,
  // from a generator of their own, so the ball scenes keep their draws: a cylinder lying on a floor, standing in a
  // domain corner, and resting on its rim on a floor. Every start succeeds and ends at least tol / 2 from every solid.
  CaseGenerator generator;
  const std::array scenes = {
      TangentScene::kCylinderLyingOnFloor,
      TangentScene::kCylinderStandingInDomainCorner,
      TangentScene::kRimOnFloor,
  };
  for (const TangentScene scene : scenes) {
    int successes = 0;
    for (int draw_index = 0; draw_index < 2000; ++draw_index) {
      TangentStart start = BuildTangentStart(generator, scene, {});
      start.inputs.solids = start.solids.data();
      successes += CheckTangentStartSucceeds(scene, start.inputs) ? 1 : 0;
    }
    std::cout << "cylinder tangent scene " << static_cast<int>(scene) << ": success " << successes << " of 2000\n";
    EXPECT_EQ(successes, 2000);
  }
}

TEST(MoveParticle, GeneratedCylinderTangentContactSlides_FailOnlyWhereAShorterStepCures) {
  // The slides of GeneratedTangentContactSlides_FailOnlyWhereAShorterStepCures in the cylinder scenes, 2000 draws per
  // scene, fixed and then moving, from a generator of their own. The cylinder in the domain corner keeps still.
  CaseGenerator generator;
  const std::array scenes = {
      TangentScene::kCylinderLyingOnFloor,
      TangentScene::kCylinderStandingInDomainCorner,
      TangentScene::kRimOnFloor,
  };
  for (const bool is_moving : {false, true}) {
    for (const TangentScene scene : scenes) {
      const TangentSlideCounts counts = ComputeTangentSlides(generator, scene, is_moving, {});
      WriteTangentSlideCounts(scene, is_moving, counts);
      CheckTangentSlideCounts(counts);
    }
  }
}

// --- The generated wedges, corners and tangent contacts far from the origin ---

TEST(MoveParticle, GeneratedTwoSurfaceStartsFarFromOrigin_SucceedDownToFiveDegrees) {
  // The draws of GeneratedTwoSurfaceStarts_SucceedDownToFiveDegrees and of its moving wedges, moved about 830 m from
  // the origin with the production tolerance there (kFarFromOrigin: tol about 1.0e-9 m, a coordinate's rounding unit
  // about tol / 9000), where the start distances, the slow speeds and the rest gaps are that tolerance's. No failure.
  const TwoSurfaceCounts fixed_counts = ComputeTwoSurfaceStarts(false, kFarFromOrigin);
  EXPECT_EQ(fixed_counts.failures, 0);
  EXPECT_GT(fixed_counts.free_paths, 1000);
  EXPECT_GT(fixed_counts.hit_paths, 1000);
  EXPECT_GT(fixed_counts.started_inside_paths, 1000);
  const TwoSurfaceCounts moving_counts = ComputeTwoSurfaceStarts(true, kFarFromOrigin);
  EXPECT_EQ(moving_counts.failures, 0);
  EXPECT_GT(moving_counts.hit_paths, 1000);
}

// The tangent scenes in TangentScene order: the balls, then the cylinders.
constexpr std::array kEveryTangentScene = {
    TangentScene::kBallOnFloor,
    TangentScene::kBallInDomainCorner,
    TangentScene::kTwoBalls,
    TangentScene::kBallOnTiltedFlatSolid,
    TangentScene::kCylinderLyingOnFloor,
    TangentScene::kCylinderStandingInDomainCorner,
    TangentScene::kRimOnFloor,
};

TEST(MoveParticle, GeneratedTangentContactStartsFarFromOrigin_Succeed) {
  // The starts of the tangent set in every scene, 2000 per scene from one generator, about 830 m from the origin with
  // the production tolerance there and every radius and half height 2000 times larger (kFarFromOriginLargeSolids):
  // radii from 100 to 1000 m, so R / tol reaches about 2^40 and the cusp's rest point is up to 2 Sqrt(R tol) = 2 mm
  // from the contact. Every start succeeds and ends at least tol / 2 from every solid.
  CaseGenerator generator;
  for (const TangentScene scene : kEveryTangentScene) {
    int successes = 0;
    for (int draw_index = 0; draw_index < 2000; ++draw_index) {
      TangentStart start = BuildTangentStart(generator, scene, kFarFromOriginLargeSolids);
      start.inputs.solids = start.solids.data();
      successes += CheckTangentStartSucceeds(scene, start.inputs) ? 1 : 0;
    }
    std::cout << "tangent scene " << static_cast<int>(scene) << " far from origin: success " << successes
              << " of 2000\n";
    EXPECT_EQ(successes, 2000);
  }
}

// Computes whether a moving tangent scene has a curved solid closing on the other one at a cusp, where the speed cap
// squeezes a particle far from the origin: a ball on a floor, two balls, a ball on a tilted floor, a cylinder lying on
// a floor. The domain-corner scenes keep still, and the rim on a floor meets the floor at its tilt, at least 0.1 rad,
// where sinking at 1e-3 m/s pushes the particle out at most 1e-3 / sin(0.1) = 0.01 m/s.
bool ComputeHasClosingCusp(TangentScene scene) {
  return scene == TangentScene::kBallOnFloor || scene == TangentScene::kTwoBalls
      || scene == TangentScene::kBallOnTiltedFlatSolid || scene == TangentScene::kCylinderLyingOnFloor;
}

// Fails the calling test unless the far slides of one scene had no other failure, every kTooManyHits was cured within
// eight halvings, and only a moving scene with a closing cusp (ComputeHasClosingCusp) was squeezed. At least 1950 of
// the 2000 draws succeed, as near the origin, or 1550 in a moving scene with a closing cusp: measured 1626 to 1683 here
// and at least 1622 over 12 generator seeds, about 1650 on average with a standard deviation of about 17
// draws (a squeeze share near 0.18 of 2000), so 1550 is 4 such deviations below the least seen.
void CheckFarTangentSlideCounts(TangentScene scene, const TangentSlideCounts& counts, bool is_moving) {
  const bool can_squeeze = is_moving && ComputeHasClosingCusp(scene);
  EXPECT_EQ(counts.other_failures, 0);
  EXPECT_EQ(counts.cured_after_halvings.at(8), 0);
  EXPECT_TRUE(can_squeeze || counts.squeezed == 0);
  EXPECT_GE(counts.successes, can_squeeze ? 1550 : 1950);
}

TEST(MoveParticle, GeneratedTangentContactSlidesFarFromOrigin_FailOnlyWhereAShorterStepCures) {
  // The slides of the tangent set in every scene, fixed and then moving, 2000 draws each from one generator, placed as
  // in GeneratedTangentContactStartsFarFromOrigin_Succeed. As near the origin: every success ends at least tol / 2 from
  // every solid, and the only failures are kTooManyHits cured by a shorter step and the speed cap's squeeze of a ball
  // or a lying cylinder sinking into the other solid (CheckTangentSlide). That squeeze is far more common here, and
  // the geometry decides it: at the rest point the two surfaces meet at a slope of 2 Sqrt(tol / R), 6e-6 for R = 100 m,
  // so sinking at 1e-3 m/s pushes the particle out at about 1e-3 / 6e-6 = 160 m/s, above 64 times the speeds of the
  // draw (method paper <sec-move-particle>); near the origin the slope is about 1e-4 and the escape below the cap.
  // Measured: 317 to 374 of the 2000 moving draws of the ball on a floor, the two balls, the ball on a tilted floor and
  // the lying cylinder, about three quarters of the quarter that sink (CheckFarTangentSlideCounts gives their floor).
  CaseGenerator generator;
  for (const bool is_moving : {false, true}) {
    for (const TangentScene scene : kEveryTangentScene) {
      const TangentSlideCounts counts = ComputeTangentSlides(generator, scene, is_moving, kFarFromOriginLargeSolids);
      std::cout << "far from origin, ";
      WriteTangentSlideCounts(scene, is_moving, counts);
      CheckFarTangentSlideCounts(scene, counts, is_moving);
    }
  }
}

// One path of the oblique set: its one flat solid and its inputs, which point at that solid.
struct ObliquePath {
  std::array<Solid, 1> solids{};
  SolidContactInputs inputs{};
};

// Builds one path of the oblique set: one flat solid of random normal through a random point within 2 m of the origin,
// fixed or moving at up to 3 m/s per axis; a start within 2 m per axis, at least 2 tol outside; both velocities up to
// 4 m/s per axis; T from 0.01 to 2 s; friction 0 to 1. Draw order: three NextSigned for the normal (drawn again while
// its length is below 0.1), three for the point, one Next for moving or fixed, three NextSigned for the linear
// velocity, three for the start (drawn again while closer than 2 tol), three for for_this_step, three for
// for_next_step, one Next for T, one for the friction. inputs.solids is left for the caller, since the path is returned
// by value.
ObliquePath BuildObliquePath(CaseGenerator& generator) {
  Vec3 normal{};
  while (math::Length(normal) < 0.1) {
    normal = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  const Vec3 point = 2 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const bool is_moving = generator.Next() < 0.5;
  const Vec3 linear_velocity =
      3 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const SolidMotion motion = {.linear_velocity = is_moving ? linear_velocity : Vec3{}, .angular_velocity = {}};
  ObliquePath path;
  path.solids.at(0) = move_particle_cases::BuildFlatSolid(point, math::Normalized(normal), motion, 1);
  path.inputs.solids = path.solids.data();
  path.inputs.solid_count = 1;
  path.inputs.contact_tolerance = kTolerance;
  do {
    path.inputs.start_position =
        2 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  } while (!ComputeIsWellOutsideEverySolid(path.inputs, path.inputs.start_position));
  path.inputs.velocity.for_this_step =
      4 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  path.inputs.velocity.for_next_step =
      4 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  path.inputs.step_duration = 0.01 + 1.99 * generator.Next();
  path.inputs.friction = generator.Next();
  return path;
}

TEST(MoveParticle, GeneratedObliqueFlatSolid_NeverHitsItAgain) {
  // A set of 20,000 paths (BuildObliquePath). One solid that is hit once is never approached again, so a second hit can
  // only be rounding: none is allowed, and no kTooManyHits.
  CaseGenerator generator;
  int hit_paths = 0;
  int repeat_hits = 0;
  int too_many_hits = 0;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    ObliquePath path = BuildObliquePath(generator);
    path.inputs.solids = path.solids.data();
    SolidContactResult result;
    ContactFailure failure;
    const ContactError error = MoveParticle(path.inputs, &result, &failure);
    const bool is_too_many_hits = failure.close_call == ContactCloseCall::kTooManyHits;
    too_many_hits += is_too_many_hits ? 1 : 0;
    const bool is_success = error == ContactError::kNone;
    hit_paths += is_success && result.hit_count > 0 ? 1 : 0;
    repeat_hits += is_success && result.hit_count > 1 ? 1 : 0;
  }
  std::cout << "oblique paths 20000, hit " << hit_paths << ", repeat hits " << repeat_hits << ", too many hits "
            << too_many_hits << "\n";
  EXPECT_EQ(repeat_hits, 0);
  EXPECT_EQ(too_many_hits, 0);
  EXPECT_GT(hit_paths, 5000);
}

TEST(MoveParticle, GeneratedCylinderRimPaths_NeverHitItAgain) {
  // A fixed cylinder at the origin along z, radius 0.5 m and half height 1 m, and 5000 paths falling onto its high rim:
  // a start at radius 0.5 + 0.02 * NextSigned m, angle 2 pi * Next, height 1.5 + Next m, a velocity
  // {0.2 * NextSigned, 0.2 * NextSigned, -1 - Next} m/s, T 2 s, friction 0 (draws in that order). A rim hit leaves
  // the particle within tol / 1024 of the rest gap, inside the touched window, where the re-sweep's touching rule reads
  // the approach the response left; the cylinder is convex, so the tangent path cannot meet it again.
  const SolidMotion no_motion;
  const std::array solids = {
      ToSolid(Cylinder{.center = {}, .axis = {.x = 0, .y = 0, .z = 1}, .radius = 0.5, .half_height = 1}, no_motion, 1),
  };
  CaseGenerator generator;
  int hit_paths = 0;
  int repeat_hits = 0;
  for (int path_index = 0; path_index < 5000; ++path_index) {
    const double radius = 0.5 + 0.02 * generator.NextSigned();  // m
    const double angle = 2 * std::numbers::pi * generator.Next();  // rad
    const double height = 1.5 + generator.Next();  // m
    SolidContactInputs inputs;
    inputs.solids = solids.data();
    inputs.solid_count = 1;
    inputs.contact_tolerance = kTolerance;
    inputs.step_duration = 2;
    inputs.start_position = {.x = radius * std::cos(angle), .y = radius * std::sin(angle), .z = height};
    const Vec3 velocity = {
        .x = 0.2 * generator.NextSigned(),
        .y = 0.2 * generator.NextSigned(),
        .z = -1 - generator.Next(),
    };
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    SolidContactResult result;
    const bool is_success = MoveParticle(inputs, &result, nullptr) == ContactError::kNone;
    hit_paths += is_success && result.hit_count > 0 ? 1 : 0;
    repeat_hits += is_success && result.hit_count > 1 ? 1 : 0;
  }
  std::cout << "rim paths 5000, hit " << hit_paths << ", repeat hits " << repeat_hits << "\n";
  EXPECT_EQ(repeat_hits, 0);
  EXPECT_GT(hit_paths, 2000);
}

// One path of the round set: its one sphere or cylinder and its inputs. inputs.solids is left for the caller.
struct RoundSolidPath {
  std::array<Solid, 1> solids{};
  SolidContactInputs inputs{};
};

// Builds one path of the round set: a sphere or a cylinder (Next < 0.5 a sphere) at the origin, radius 0.1 + 0.4 Next
// m, half height 0.1 + 0.9 Next m, its axis three NextSigned (drawn again while shorter than 0.1); T = 0.01 + 0.99 Next
// s; moving when Next < 0.5 at three NextSigned m/s; spinning about its axis at 20 NextSigned rad/s, plus a turn across
// the axis of Next times 0.99 of the bound of method paper <sec-slide-and-spin>, tol / (8 T Length(half_size)) rad/s,
// about the cross of the axis with three NextSigned (drawn again while shorter than 0.1); a start on the direction of
// three NextSigned (drawn again while shorter than 0.1), Length(half_size) + 0.5 m from the centre; three NextSigned
// for the target and one Next for its kind; a velocity relative to the solid that reaches the target after (0.3 + 0.6
// Next) T, both velocities equal; friction Next. Draws in that order, all of them for both shapes. The target is 0.8
// Length(half_size) times the three NextSigned from the centre, or, for a cylinder when the kind's Next is below 0.3, a
// point within tol of the grown cylinder's rim edge, in its own axes {(R + tol + tol a) cos(pi b), (R + tol + tol a)
// sin(pi b), sign(c) (H + tol + tol c)} with a, b, c the three NextSigned, so the path may enter through the rim
// corner. The cylinder never counts as turning; for a sphere the whole angular velocity is a spin.
RoundSolidPath BuildRoundSolidPath(CaseGenerator& generator) {
  const bool is_sphere = generator.Next() < 0.5;
  const double radius = 0.1 + 0.4 * generator.Next();  // m
  const double half_height = 0.1 + 0.9 * generator.Next();  // m
  const auto draw_direction = [&generator] {
    Vec3 direction{};
    while (math::Length(direction) < 0.1) {
      direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    }
    return math::Normalized(direction);
  };
  const Vec3 axis = draw_direction();
  const double step_duration = 0.01 + 0.99 * generator.Next();  // s
  const bool is_moving = generator.Next() < 0.5;
  const Vec3 linear_draw = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const double spin_rate = 20 * generator.NextSigned();  // rad/s
  const double tilt_share = generator.Next();
  Vec3 across_axis{};
  while (math::Length(across_axis) < 0.1) {
    across_axis = math::Cross(axis, draw_direction());
  }
  const Vec3 start_direction = draw_direction();
  const Vec3 target_draw = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  const bool is_rim_aim = !is_sphere && generator.Next() < 0.3;
  const double reach_share = 0.3 + 0.6 * generator.Next();
  const double friction = generator.Next();

  const Vec3 half_size =
      is_sphere ? Vec3{.x = radius, .y = radius, .z = radius} : Vec3{.x = radius, .y = radius, .z = half_height};
  const double reach = math::Length(half_size);  // m
  const double largest_tilt_rate = kTolerance / (8 * step_duration * reach);  // rad/s
  const Vec3 linear_velocity = is_moving ? linear_draw : Vec3{};  // m/s
  const SolidMotion motion = {
      .linear_velocity = linear_velocity,
      .angular_velocity = spin_rate * axis + 0.99 * tilt_share * largest_tilt_rate * math::Normalized(across_axis),
  };
  RoundSolidPath path;
  path.solids.at(0) = is_sphere
      ? ToSolid(Sphere{.center = {}, .radius = radius}, motion, 1)
      : ToSolid(Cylinder{.center = {}, .axis = axis, .radius = radius, .half_height = half_height}, motion, 1);
  path.inputs.solid_count = 1;
  path.inputs.contact_tolerance = kTolerance;
  path.inputs.step_duration = step_duration;
  path.inputs.start_position = (reach + 0.5) * start_direction;
  // m, from the solid's centre at the time it is reached.
  Vec3 target = 0.8 * reach * target_draw;
  if (is_rim_aim) {
    const double rim_radius = radius + kTolerance + kTolerance * target_draw.x;  // m
    const double angle = std::numbers::pi * target_draw.y;  // rad
    const double cap_sign = target_draw.z < 0 ? -1.0 : 1.0;
    const Vec3 own_target = {
        .x = rim_radius * std::cos(angle),
        .y = rim_radius * std::sin(angle),
        .z = cap_sign * (half_height + kTolerance + kTolerance * target_draw.z),
    };
    target = path.solids.at(0).pose_at_step_start.axes * own_target;
  }
  const Vec3 velocity =
      linear_velocity + (1 / (reach_share * step_duration)) * (target - path.inputs.start_position);  // m/s
  path.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  path.inputs.friction = friction;
  return path;
}

// What the round set gave, counted for the checks and the report.
struct RoundSolidCounts {
  int hit_paths = 0;
  int rim_hit_paths = 0;  // The first hit on a rim.
  int repeat_hits = 0;  // More than one hit.
  int failures = 0;
};

// Runs MoveParticle on one path of the round set and counts it; a failure fails the calling test.
void CheckRoundSolidPath(const RoundSolidPath& path, RoundSolidCounts& counts) {
  EXPECT_FALSE(ComputeIsTurningSolid(path.inputs, 0));
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(path.inputs, &result, &failure);
  if (error != ContactError::kNone) {
    ADD_FAILURE() << "error " << static_cast<int>(error) << ", close call " << static_cast<int>(failure.close_call);
    ++counts.failures;
    return;
  }
  CheckEndOutsideEverySolid(path.inputs, result.position);
  const bool is_hit = result.hit_count > 0;
  const bool is_rim = result.face == SolidFace::kLowRim || result.face == SolidFace::kHighRim;
  counts.hit_paths += is_hit ? 1 : 0;
  counts.rim_hit_paths += is_hit && is_rim ? 1 : 0;
  counts.repeat_hits += result.hit_count > 1 ? 1 : 0;
}

TEST(MoveParticle, GeneratedRoundSolidPaths_NeverHitAgain) {
  // 20,000 paths (BuildRoundSolidPath) onto one sphere or cylinder, spinning, moving, and for a cylinder tilting within
  // the bound of method paper <sec-slide-and-spin>, with friction. After the response at the hit, the path is tangent
  // to the solid or leaves it, measured from the surface's velocity there, and the solid is convex, so a second hit can
  // only come from reading that approach differently: none is allowed. The touching rule now reads the distance, normal
  // and solid velocity a joint response reads, by the same expressions, and a rim hit is where the rim search found the
  // true distance within tol / 1024 of tol. Before, with the rate measured from the linear velocity alone, 2757 paths
  // onto cylinders ended in kTooManyHits (2755 of them only with the tilt), and 144 rim hits, left in the rim corner a
  // hair outside the grown edge, were hit again. With rim hits moved to the rest gap 10,490 paths hit, 548 on a rim;
  // the rim search lets 207 of those pass, which the true distance never brings within tol (method paper
  // <sec-round-first-hit>).
  CaseGenerator generator;
  RoundSolidCounts counts;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    RoundSolidPath path = BuildRoundSolidPath(generator);
    path.inputs.solids = path.solids.data();
    CheckRoundSolidPath(path, counts);
  }
  std::cout << "round paths 20000, hit " << counts.hit_paths << ", rim hits " << counts.rim_hit_paths
            << ", repeat hits " << counts.repeat_hits << ", failures " << counts.failures << "\n";
  EXPECT_EQ(counts.repeat_hits, 0);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_GT(counts.hit_paths, 9000);  // Measured: 10,283.
  EXPECT_GT(counts.rim_hit_paths, 250);  // Measured: 341.
}

// One path across a flush seam: two solids, the particle's inputs, and the direction along the path.
struct FlushSeamPath {
  std::array<Solid, 2> solids{};
  SolidContactInputs inputs{};
  Vec3 along_path{};  // No unit, unit vector, world: the direction of the velocity relative to the solids.
  Vec3 solid_velocity{};  // m/s, world: both solids move with it.
};

// The kinds of flush seam the generated set crosses.
enum class FlushSeamKind : std::uint8_t {
  kStackedDrums,  // Two drums of one radius on one axis, down the seam along the wall.
  kPuckInFloor,  // A puck flush in a floor, onto it or off it along the floor.
  kBoxesSideBySide,  // Two boxes side by side in one floor, across the seam along their tops.
};

// A turn whose three columns have no zero entry: {0.36, 0.48, -0.8}, {-0.8, 0.6, 0}, {0.48, 0.64, 0.6}.
constexpr Mat3 kSeamTiltedAxes = {
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

// Draws a unit vector from three NextSigned, drawn again while shorter than 0.1.
Vec3 BuildSeamDirection(CaseGenerator& generator) {
  Vec3 direction{};
  while (math::Length(direction) < 0.1) {
    direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  return math::Normalized(direction);
}

// Builds one path across a flush seam. Draws, in this order, for every kind: the axes, kSeamTiltedAxes when Next < 0.5,
// else the world axes; a centre 2 Next times a direction (BuildSeamDirection); the solids' velocity, a direction when
// Next < 1 / 3, else zero; a first size 0.1 + 0.9 Next m; a second 0.1 + 0.9 Next m; a third 0.05 + 0.45 Next m; an
// azimuth 2 pi Next; the speed 0.5 + 1.5 Next m/s; the push into the surface, 0, 0, 0.16 or 1 m/s by floor(4 Next); the
// distance before the seam 0.01 + 0.14 Next m; a sideways share -0.3 + 0.6 Next; onto or off the puck, Next < 0.5; the
// travel 0.16 + 0.14 Next m, which sets T. Stacked drums: radius the first size, half heights the second and the
// third, the upper drum's low cap on the lower drum's high cap; the start at the upper drum's wall rest gap at the
// azimuth, the distance before the seam; the path down the axis, pushed into the wall. Puck: radius the first size,
// half height the third, its top flush with the floor; the start at the floor's rest gap, the distance before the rim,
// outside it going on or inside it going off, with the sideways share across the radius; pushed into the floor. Box
// boxes side by side: half sizes the first, second and third sizes along own x, y, z, the second box of half length the
// second size along own x beside the first, both tops at the centre's height; the start at the first top's rest gap,
// the distance before the seam, 0.1 m off the middle times the sideways share; the path along own x with the sideways
// share along own y, pushed into the tops.
FlushSeamPath BuildFlushSeamPath(CaseGenerator& generator, FlushSeamKind kind) {
  const bool is_tilted = generator.Next() < 0.5;
  const Vec3 center = 2 * generator.Next() * BuildSeamDirection(generator);
  const bool is_moving = generator.Next() < 1.0 / 3;
  const Vec3 moving_direction = BuildSeamDirection(generator);
  const double first_size = 0.1 + 0.9 * generator.Next();  // m
  const double second_size = 0.1 + 0.9 * generator.Next();  // m
  const double third_size = 0.05 + 0.45 * generator.Next();  // m
  const double azimuth = 2 * std::numbers::pi * generator.Next();  // rad
  const double speed = 0.5 + 1.5 * generator.Next();  // m/s
  constexpr std::array<double, 4> kPushes = {0, 0, 0.16, 1};  // m/s
  const double push = kPushes.at(static_cast<std::size_t>(std::floor(4 * generator.Next())));  // m/s
  const double before_seam = 0.01 + 0.14 * generator.Next();  // m
  const double sideways_share = -0.3 + 0.6 * generator.Next();
  const bool is_onto_puck = generator.Next() < 0.5;
  const double travel = 0.16 + 0.14 * generator.Next();  // m

  const Mat3 axes = is_tilted ? kSeamTiltedAxes : kIdentity;
  const Vec3 axis = {.x = axes.xz, .y = axes.yz, .z = axes.zz};
  const Vec3 solid_velocity = is_moving ? moving_direction : Vec3{};  // m/s
  const SolidMotion motion = {.linear_velocity = solid_velocity, .angular_velocity = {}};
  FlushSeamPath path;
  path.solid_velocity = solid_velocity;
  Vec3 relative_velocity{};  // m/s, along the surface, before the push.
  Vec3 push_direction{};  // No unit: into the surface the particle rests on.
  if (kind == FlushSeamKind::kStackedDrums) {
    const double radius = first_size;  // m
    const Cylinder upper = {.center = center, .axis = axis, .radius = radius, .half_height = second_size};
    const Vec3 lower_center = center - (second_size + third_size) * axis;
    const Cylinder lower = {.center = lower_center, .axis = axis, .radius = radius, .half_height = third_size};
    path.solids = {ToSolid(upper, motion, 1), ToSolid(lower, motion, 2)};
    const Mat3 own_axes = path.solids.at(0).pose_at_step_start.axes;
    const Vec3 radial = std::cos(azimuth) * Vec3{.x = own_axes.xx, .y = own_axes.yx, .z = own_axes.zx}
        + std::sin(azimuth) * Vec3{.x = own_axes.xy, .y = own_axes.yy, .z = own_axes.zy};
    path.inputs.start_position = center + (radius + kTolerance) * radial + (-second_size + before_seam) * axis;
    relative_velocity = -speed * axis;
    push_direction = -radial;
  } else if (kind == FlushSeamKind::kBoxesSideBySide) {
    const Vec3 first_half_size = {.x = first_size, .y = second_size, .z = third_size};  // m
    const Vec3 second_half_size = {.x = second_size, .y = second_size, .z = third_size};  // m
    const Vec3 own_x = {.x = axes.xx, .y = axes.yx, .z = axes.zx};
    const Vec3 own_y = {.x = axes.xy, .y = axes.yy, .z = axes.zy};
    const Vec3 first_center = center - third_size * axis;
    const Vec3 second_center = first_center + (first_size + second_size) * own_x;
    const Box first_floor_box = {.center = first_center, .axes = axes, .half_size = first_half_size};
    const Box second_floor_box = {.center = second_center, .axes = axes, .half_size = second_half_size};
    path.solids = {ToSolid(first_floor_box, motion, 1), ToSolid(second_floor_box, motion, 2)};
    path.inputs.start_position =
        center + (first_size - before_seam) * own_x + 0.1 * sideways_share * own_y + kTolerance * axis;
    relative_velocity = speed * (own_x + sideways_share * own_y);
    push_direction = -axis;
  } else {
    const double radius = first_size;  // m
    const Cylinder puck =
        {.center = center - third_size * axis, .axis = axis, .radius = radius, .half_height = third_size};
    const FlatSolid floor = {.point_on_surface = center, .normal_out_of_solid = axis};
    path.solids = {ToSolid(floor, motion, 1), ToSolid(puck, motion, 2)};
    const Mat3 own_axes = path.solids.at(1).pose_at_step_start.axes;
    const Vec3 radial = std::cos(azimuth) * Vec3{.x = own_axes.xx, .y = own_axes.yx, .z = own_axes.zx}
        + std::sin(azimuth) * Vec3{.x = own_axes.xy, .y = own_axes.yy, .z = own_axes.zy};
    const Vec3 across = math::Cross(axis, radial);
    const double start_radius = is_onto_puck ? radius + before_seam : radius - before_seam;  // m
    const double radial_sign = is_onto_puck ? -1.0 : 1.0;
    path.inputs.start_position = center + start_radius * radial + kTolerance * axis;
    relative_velocity = speed * (radial_sign * radial + sideways_share * across);
    push_direction = -axis;
  }
  const Vec3 velocity = solid_velocity + relative_velocity + push * push_direction;  // m/s
  path.along_path = math::Normalized(relative_velocity);
  path.inputs.solid_count = 2;
  path.inputs.contact_tolerance = kTolerance;
  path.inputs.step_duration = travel / speed;
  path.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  return path;
}

// What one kind of flush seam gave over its paths.
struct FlushSeamCounts {
  int paths = 0;
  int failures = 0;
  int several_hits = 0;
  int lost_speed = 0;  // Paths whose speed along the path, measured from the solids, fell below 1 - 1e-6 of its start.
};

// Runs MoveParticle on one path across a flush seam and counts it.
void CheckFlushSeamPath(const FlushSeamPath& path, FlushSeamCounts& counts) {
  ++counts.paths;
  SolidContactResult result;
  ContactFailure failure;
  if (MoveParticle(path.inputs, &result, &failure) != ContactError::kNone) {
    ++counts.failures;
    return;
  }
  counts.several_hits += result.hit_count > 1 ? 1 : 0;
  const Vec3 start_relative = path.inputs.velocity.for_this_step - path.solid_velocity;  // m/s
  const double start_speed = math::Dot(start_relative, path.along_path);  // m/s
  const double this_step_speed = math::Dot(result.velocity.for_this_step - path.solid_velocity, path.along_path);
  const double next_step_speed = math::Dot(result.velocity.for_next_step - path.solid_velocity, path.along_path);
  const double least_kept_speed = (1 - 1e-6) * start_speed;  // m/s
  const bool keeps_speed = this_step_speed >= least_kept_speed && next_step_speed >= least_kept_speed;
  counts.lost_speed += keeps_speed ? 0 : 1;
}

// Prints one kind's counts for the report and fails the calling test unless all 2,000 paths succeeded with at most one
// hit and kept their speed.
void CheckFlushSeamCounts(FlushSeamKind kind, const FlushSeamCounts& counts) {
  std::cout << "flush seams kind " << static_cast<int>(kind) << ": paths " << counts.paths << ", failures "
            << counts.failures << ", several hits " << counts.several_hits << ", lost speed " << counts.lost_speed
            << "\n";
  EXPECT_EQ(counts.paths, 2000);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_EQ(counts.several_hits, 0);
  EXPECT_EQ(counts.lost_speed, 0);
}

TEST(MoveParticle, GeneratedFlushSeams_KeepTheSpeedAlongTheSeam) {
  // 2,000 paths across each kind of flush seam (BuildFlushSeamPath), friction 0: a seam is no step, so the particle
  // crosses it without a hit, or with the start's one when it is pushed into the surface, and keeps its speed along the
  // path. Measured with the sharp grown rim edge as the hit: 1,044 of the stacked drums and 629 of the pucks lost
  // speed, 578 and 360 with a second hit; a model of the rule in double precision, on its own draws, loses none.
  for (const FlushSeamKind kind :
       {FlushSeamKind::kStackedDrums, FlushSeamKind::kPuckInFloor, FlushSeamKind::kBoxesSideBySide}) {
    CaseGenerator generator;
    FlushSeamCounts counts;
    for (int path_index = 0; path_index < 2000; ++path_index) {
      FlushSeamPath path = BuildFlushSeamPath(generator, kind);
      path.inputs.solids = path.solids.data();
      CheckFlushSeamPath(path, counts);
    }
    CheckFlushSeamCounts(kind, counts);
  }
}

// Moves a flush-seam path's start to offset before its seam, along the line BuildFlushSeamPath measured its distance
// before the seam on: the drums' axis, the puck's radius, the boxes' own x axis.
void ApplySeamOffset(FlushSeamKind kind, double offset, FlushSeamPath* path) {
  const Vec3 start = path->inputs.start_position;
  if (kind == FlushSeamKind::kStackedDrums) {
    const Solid& upper = path->solids.at(0);
    const Vec3 axis = {
        .x = upper.pose_at_step_start.axes.xz,
        .y = upper.pose_at_step_start.axes.yz,
        .z = upper.pose_at_step_start.axes.zz,
    };
    const double height = math::Dot(start - upper.pose_at_step_start.origin, axis);  // m
    path->inputs.start_position = start + ((-upper.half_size.z + offset) - height) * axis;
    return;
  }
  if (kind == FlushSeamKind::kBoxesSideBySide) {
    const Solid& first_box = path->solids.at(0);
    const Mat3 axes = first_box.pose_at_step_start.axes;
    const Vec3 own_x = {.x = axes.xx, .y = axes.yx, .z = axes.zx};
    const double along_x = math::Dot(start - first_box.pose_at_step_start.origin, own_x);  // m
    path->inputs.start_position = start + ((first_box.half_size.x - offset) - along_x) * own_x;
    return;
  }
  const Solid& puck = path->solids.at(1);
  const Vec3 axis = {
      .x = puck.pose_at_step_start.axes.xz,
      .y = puck.pose_at_step_start.axes.yz,
      .z = puck.pose_at_step_start.axes.zz,
  };
  const Vec3 from_center = start - puck.pose_at_step_start.origin;  // m
  const Vec3 across = from_center - math::Dot(from_center, axis) * axis;  // m
  const double axis_distance = math::Length(across);  // m
  const double radius = puck.half_size.x;  // m
  const double new_distance = axis_distance > radius ? radius + offset : radius - offset;  // m
  path->inputs.start_position = start + ((new_distance - axis_distance) / axis_distance) * across;
}

// What the runs across a seam in many steps gave for one kind.
struct SeamRunCounts {
  int runs = 0;
  int failures = 0;  // Runs with a step that did not succeed.
  int lost_speed = 0;  // Runs with a step that lost speed along the path.
};

// Runs one particle across a flush seam in many steps (GeneratedFlushSeamsOverManySteps_KeepTheSpeed) and counts it.
// Draws, after BuildFlushSeamPath's: the start 1.5 + 4.5 Next tol before the seam; the step's length 2^(-2 + 12 Next)
// tol along the path; friction 0 when Next < 2 / 3, else 0.3.
void CheckSeamRun(CaseGenerator& generator, FlushSeamKind kind, SeamRunCounts& counts) {
  FlushSeamPath path = BuildFlushSeamPath(generator, kind);
  const double offset = (1.5 + 4.5 * generator.Next()) * kTolerance;  // m
  const double step_length = std::exp2(-2 + 12 * generator.Next()) * kTolerance;  // m
  const double friction = generator.Next() < 2.0 / 3 ? 0.0 : 0.3;
  ApplySeamOffset(kind, offset, &path);
  const Vec3 relative_velocity = path.inputs.velocity.for_this_step - path.solid_velocity;  // m/s
  const double start_speed = math::Dot(relative_velocity, path.along_path);  // m/s
  const double push = math::Length(relative_velocity - start_speed * path.along_path);  // m/s
  const double least_kept_speed = (1 - 1e-6) * (start_speed - friction * push);  // m/s
  const double step_duration = step_length / start_speed;  // s
  const int step_count = std::min(60, static_cast<int>(std::floor(14 * kTolerance / step_length)) + 2);
  path.inputs.step_duration = step_duration;
  path.inputs.friction = friction;
  ++counts.runs;
  bool is_failure = false;
  bool is_slower = false;
  for (int step_index = 0; step_index < step_count && !is_failure; ++step_index) {
    path.inputs.solids = path.solids.data();
    SolidContactResult result;
    ContactFailure failure;
    is_failure = MoveParticle(path.inputs, &result, &failure) != ContactError::kNone;
    const double speed = math::Dot(result.velocity.for_this_step - path.solid_velocity, path.along_path);  // m/s
    is_slower = is_slower || (!is_failure && speed < least_kept_speed);
    path.inputs.start_position = result.position;
    for (Solid& solid : path.solids) {
      solid.pose_at_step_start.origin = solid.pose_at_step_start.origin + step_duration * path.solid_velocity;
    }
  }
  counts.failures += is_failure ? 1 : 0;
  counts.lost_speed += is_slower ? 1 : 0;
}

TEST(MoveParticle, GeneratedFlushSeamsOverManySteps_KeepTheSpeed) {
  // 1,000 runs per kind of flush seam (CheckSeamRun): the solids of BuildFlushSeamPath, fixed or moving together, world
  // or tilted axes, a start 1.5 to 6 tol before the seam, steps of tol / 4 to 1,024 tol, Min(60, Floor(14 tol / step) +
  // 2) of them, the same input velocity every step: every step succeeds and keeps the speed along the path that
  // friction on the surface slid on leaves (method paper <sec-touching-rule>). A model of the rule in double precision,
  // on its own draws: 0 of 1,000 lose speed; with the next solid as an active surface, 165 (drums), 69 (pucks) and 36
  // (boxes).
  for (const FlushSeamKind kind :
       {FlushSeamKind::kStackedDrums, FlushSeamKind::kPuckInFloor, FlushSeamKind::kBoxesSideBySide}) {
    CaseGenerator generator;
    SeamRunCounts counts;
    for (int run_index = 0; run_index < 1000; ++run_index) {
      CheckSeamRun(generator, kind, counts);
    }
    std::cout << "flush seams over many steps, kind " << static_cast<int>(kind) << ": runs " << counts.runs
              << ", failures " << counts.failures << ", lost speed " << counts.lost_speed << "\n";
    EXPECT_EQ(counts.runs, 1000);
    EXPECT_EQ(counts.failures, 0);
    EXPECT_EQ(counts.lost_speed, 0);
  }
}

// One path of the chasing set: a fixed flat solid A and a flat solid B chasing the particle along a normal at right
// angles to A's.
struct ChasingPath {
  std::array<Solid, 2> solids{};
  SolidContactInputs inputs{};
};

// Builds one path of the chasing set. A goes through the origin with a random normal nA; B's normal is a random
// direction made square to nA, its surface is 5 to 15 m away (B at 100 m/s) or 50 to 150 m (at 1000 m/s) and it moves
// along -nB, so it reaches the particle after A has. The particle starts at 0.05 nA moving at -nA plus up to 0.5 m/s
// along each axis, both velocities equal, T 0.2 s, friction 0. Draw order: three NextSigned for nA (drawn again while
// shorter than 0.1), three for B's direction (drawn again while its part square to nA is shorter than 0.1), one Next
// for the speed, one for the distance, three NextSigned for the velocity's offset. inputs.solids is left for the
// caller.
ChasingPath BuildChasingPath(CaseGenerator& generator) {
  Vec3 first_direction{};
  while (math::Length(first_direction) < 0.1) {
    first_direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  const Vec3 first_normal = math::Normalized(first_direction);
  Vec3 chasing_direction{};
  while (math::Length(chasing_direction) < 0.1) {
    const Vec3 second = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    chasing_direction = second - math::Dot(first_normal, second) * first_normal;
  }
  const Vec3 chasing_normal = math::Normalized(chasing_direction);
  const bool is_fast = generator.Next() < 0.5;
  const double chasing_speed = is_fast ? 1000.0 : 100.0;  // m/s
  const double distance = (is_fast ? 50.0 : 5.0) + (is_fast ? 100.0 : 10.0) * generator.Next();  // m
  const Vec3 offset = 0.5 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  ChasingPath path;
  const SolidMotion no_motion;
  path.solids.at(0) = move_particle_cases::BuildFlatSolid(Vec3{}, first_normal, no_motion, 1);
  path.solids.at(1) = move_particle_cases::BuildFlatSolid(
      distance * chasing_normal,
      -chasing_normal,
      {.linear_velocity = -chasing_speed * chasing_normal, .angular_velocity = {}},
      2
  );
  path.inputs.solid_count = 2;
  path.inputs.contact_tolerance = kTolerance;
  path.inputs.step_duration = 0.2;
  path.inputs.start_position = 0.05 * first_normal;
  const Vec3 velocity = offset - first_normal;
  path.inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  return path;
}

TEST(MoveParticle, GeneratedChasingSolid_FirstSolidRarelyHitAgain) {
  // A set of 20,000 paths (BuildChasingPath): A is hit, then B, which carries the particle away along a
  // direction square to A's normal. Neither is approached again, so a third hit is a re-hit of A by rounding.
  CaseGenerator generator;
  int two_hit_paths = 0;
  int re_hits = 0;
  int failures = 0;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    ChasingPath path = BuildChasingPath(generator);
    path.inputs.solids = path.solids.data();
    SolidContactResult result;
    const bool is_success = MoveParticle(path.inputs, &result, nullptr) == ContactError::kNone;
    failures += is_success ? 0 : 1;
    two_hit_paths += is_success && result.hit_count == 2 ? 1 : 0;
    re_hits += is_success && result.hit_count > 2 ? 1 : 0;
  }
  std::cout << "chasing paths 20000, two hits " << two_hit_paths << ", re-hits " << re_hits << ", failures " << failures
            << "\n";
  EXPECT_EQ(re_hits, 0);
  EXPECT_EQ(failures, 0);
  EXPECT_GT(two_hit_paths, 15000);
}

TEST(MoveParticle, GeneratedSteepHits_NeverHitAgain) {
  // 20,000 near head-on hits on one fixed flat solid through the origin: a random normal n (three NextSigned, drawn
  // again while shorter than 0.1), a start at 0.5 n, both velocities -(1 + 4 * Next) n plus up to 1e-4 m/s per axis
  // (three NextSigned), T 1 s, friction 0. The response leaves a sideways speed of about 1e-4 m/s, far smaller than the
  // incoming speed, and the particle slides on from the rest gap: never a second hit. With friction 0 the response's
  // second removal leaves rounding of that small velocity, so this set cannot tell the incoming bound from the
  // outgoing one; GeneratedNearStickSlips_NeverHitAgain can.
  CaseGenerator generator;
  int hit_paths = 0;
  int repeat_hits = 0;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    Vec3 direction{};
    while (math::Length(direction) < 0.1) {
      direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    }
    const Vec3 normal = math::Normalized(direction);
    const double speed = 1 + 4 * generator.Next();  // m/s
    const Vec3 offset =
        1e-4 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const SolidMotion no_motion;
    const std::array solids = {move_particle_cases::BuildFlatSolid(Vec3{}, normal, no_motion, 1)};
    SolidContactInputs inputs;
    inputs.solids = solids.data();
    inputs.solid_count = 1;
    inputs.contact_tolerance = kTolerance;
    inputs.step_duration = 1;
    inputs.start_position = 0.5 * normal;
    const Vec3 velocity = offset - speed * normal;
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    SolidContactResult result;
    const bool is_success = MoveParticle(inputs, &result, nullptr) == ContactError::kNone;
    hit_paths += is_success && result.hit_count > 0 ? 1 : 0;
    repeat_hits += is_success && result.hit_count > 1 ? 1 : 0;
  }
  std::cout << "steep paths 20000, hit " << hit_paths << ", repeat hits " << repeat_hits << "\n";
  EXPECT_EQ(repeat_hits, 0);
  EXPECT_EQ(hit_paths, 20000);
}

TEST(MoveParticle, GeneratedNearStickSlips_NeverHitAgain) {
  // 20,000 slips just below the stick limit on one fixed flat solid through the origin: a random normal n (three
  // NextSigned, drawn again while shorter than 0.1), a start at 0.05 n, both velocities with three NextSigned times
  // 3 m/s per axis (drawn again while approaching slower than 0.2 m/s), T 1 s, and the friction (1 - delta) times the
  // stick limit, sideways speed over speed into the solid, with delta = 10^(-8 + 7 * Next). The friction blend rounds
  // by a share of the incoming speed while the returned velocity is tiny: an allowance from the returned velocity
  // would call that a new approach and hit the solid again.
  CaseGenerator generator;
  int hit_paths = 0;
  int repeat_hits = 0;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    Vec3 direction{};
    while (math::Length(direction) < 0.1) {
      direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    }
    const Vec3 normal = math::Normalized(direction);
    Vec3 velocity{};
    while (!(math::Dot(normal, velocity) < -0.2)) {
      velocity = 3 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    }
    const double approach_speed = math::Dot(normal, velocity);  // m/s, negative.
    const double sideways_speed = math::Length(velocity - approach_speed * normal);  // m/s
    const double stick_limit = sideways_speed / -approach_speed;  // No unit.
    const double below_limit_share = std::pow(10.0, -8 + 7 * generator.Next());  // No unit.
    const SolidMotion no_motion;
    const std::array solids = {move_particle_cases::BuildFlatSolid(Vec3{}, normal, no_motion, 1)};
    SolidContactInputs inputs;
    inputs.solids = solids.data();
    inputs.solid_count = 1;
    inputs.contact_tolerance = kTolerance;
    inputs.step_duration = 1;
    inputs.start_position = 0.05 * normal;
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    inputs.friction = (1 - below_limit_share) * stick_limit;
    SolidContactResult result;
    const bool is_success = MoveParticle(inputs, &result, nullptr) == ContactError::kNone;
    hit_paths += is_success && result.hit_count > 0 ? 1 : 0;
    repeat_hits += is_success && result.hit_count > 1 ? 1 : 0;
  }
  std::cout << "near-stick paths 20000, hit " << hit_paths << ", repeat hits " << repeat_hits << "\n";
  EXPECT_EQ(repeat_hits, 0);
  EXPECT_EQ(hit_paths, 20000);
}

// --- Generated moves near turning solids ---

// One generated set of moves near a turning solid: its solids (the turning solid first), the step, the contact
// tolerance and the length every draw is scaled by.
struct TurningMoveSet {
  const char* name = "";
  std::vector<Solid> solids;
  double step_duration = 0;  // s.
  double contact_tolerance = 0;  // m.
  double length_scale = 0;  // m. The size of the turning solid: starts and speeds are drawn in proportion to it.
  double largest_coordinate = 0;  // m. A bound on the coordinates of every start and solid origin of the set.
};

// Where a generated start near a turning solid lies, by its distance d to it at time 0.
enum class TurningStartKind : std::uint8_t {
  kWithinWindow,  // 0 <= d <= 1.25 tol: the stepping search hits it at 0.
  kAtRestGap,  // d = tol, up to the placement's rounding.
  kSlightlyInside,  // -tol <= d < -tol / 4.
  kAway,  // d between 0.01 and 0.51 length scales, moving toward the solid.
};

// What a generated set of moves near a turning solid gave, counted for the report.
struct TurningMoveCounts {
  int moves = 0;
  std::array<int, kMaxHitsPerStep + 1> successes_per_hit_count{};  // Successes, by hit_count.
  std::array<int, 8> failures_per_close_call{};  // In ContactCloseCall order.
  int other_errors = 0;  // Any error but kTooCloseToCall: none expected.
  // Moves whose error, hit count or velocities differ, bit for bit, with an exact copy of the turning solid; for a
  // start closer than tol, whose move off the near surfaces reads the solids in array order, the error and hit count
  // only.
  int copy_differences = 0;
  // No unit. For a start closer than tol, the largest velocity difference with the copy over Max(speed, 1 m/s).
  double largest_moved_start_copy_difference = 0;
  int order_differences = 0;  // Moves whose error, hit count or velocities differ with the solids in reverse order.
  int carried_moves = 0;  // Successes whose last hit was on the turning solid, so ended in a carry.
  double least_carried_end_share = std::numeric_limits<double>::infinity();  // Of tol, before the settle.
  std::vector<int> advance_counts;  // The stepping search's advances on the path from the start as given.
};

// Builds the inputs of one generated move near the turning solid of a set, or none when 50 points in a row cannot be
// placed. Draw order: one Next for the kind (floor(4 * Next) in TurningStartKind order), one for the target distance;
// then, for each try, three NextSigned for a point up to one length scale per axis from the solid's origin, moved along
// its normal at time 0 to the target distance and kept when its distance is within tol / 8 of the target and every
// other solid is at least 4 tol away; then three NextSigned for a velocity of up to 2 length scales per second per axis
// relative to the surface there (an away start also gets (0.5 + Next) length scales per step into the solid), and one
// Next for the friction.
bool BuildTurningMove(const TurningMoveSet& set, CaseGenerator& generator, SolidContactInputs* inputs) {
  const double contact_tolerance = set.contact_tolerance;
  const double length_scale = set.length_scale;
  const auto start_kind = static_cast<TurningStartKind>(static_cast<int>(std::floor(4 * generator.Next())));
  const double target_draw = generator.Next();
  double target_distance = contact_tolerance;  // m, kAtRestGap.
  if (start_kind == TurningStartKind::kWithinWindow) {
    target_distance = 1.25 * contact_tolerance * target_draw;
  } else if (start_kind == TurningStartKind::kSlightlyInside) {
    target_distance = -(0.25 + 0.75 * target_draw) * contact_tolerance;
  } else if (start_kind == TurningStartKind::kAway) {
    target_distance = (0.01 + 0.5 * target_draw) * length_scale;
  }
  const Solid& turning_solid = set.solids.at(0);
  const SolidPose start_pose = ComputeSolidPose(turning_solid, 0);
  bool is_placed = false;
  for (int draw_index = 0; draw_index < 50 && !is_placed; ++draw_index) {
    const Vec3 offset = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const Vec3 point = start_pose.origin + length_scale * offset;
    const DistanceToSolid at_point = ComputeDistanceToSolid(turning_solid, start_pose, point);
    inputs->start_position = point + (target_distance - at_point.distance) * at_point.normal_out_of_solid;
    const double start_distance = ComputeDistanceToSolid(turning_solid, start_pose, inputs->start_position).distance;
    bool is_far_from_others = true;
    for (std::size_t solid_index = 1; solid_index < set.solids.size(); ++solid_index) {
      const Solid& other = set.solids.at(solid_index);
      const double other_distance =
          ComputeDistanceToSolid(other, ComputeSolidPose(other, 0), inputs->start_position).distance;
      is_far_from_others = is_far_from_others && other_distance >= 4 * contact_tolerance;
    }
    is_placed = std::abs(start_distance - target_distance) <= contact_tolerance / 8 && is_far_from_others;
  }
  if (!is_placed) {
    return false;
  }
  const DistanceToSolid at_start = ComputeDistanceToSolid(turning_solid, start_pose, inputs->start_position);
  const Vec3 relative_draw = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  Vec3 velocity = ComputeSolidVelocity(turning_solid, start_pose, inputs->start_position)
      + 2 * length_scale * relative_draw;  // m/s
  if (start_kind == TurningStartKind::kAway) {
    velocity = velocity - (0.5 + generator.Next()) * length_scale / set.step_duration * at_start.normal_out_of_solid;
  }
  inputs->velocity = {.for_next_step = velocity, .for_this_step = velocity};
  inputs->friction = generator.Next();
  return true;
}

// Fails the calling test unless the end position is at least tol / 2 from every solid at its end pose, by
// ComputeDistanceToSolid itself, with the inputs' own contact tolerance.
void CheckEndOutsideEverySolidByItsTolerance(const SolidContactInputs& inputs, Vec3 end_position) {
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    const double end_distance =
        ComputeDistanceToSolid(solid, ComputeSolidPose(solid, inputs.step_duration), end_position).distance;
    EXPECT_GE(end_distance, inputs.contact_tolerance / 2) << "solid index " << solid_index;
  }
}

// Runs MoveParticle on the inputs with the solids given in their place, ids renumbered from 1 in that order.
MoveParticleCall ComputeMoveWithSolids(const SolidContactInputs& inputs, std::vector<Solid> solids) {
  for (std::size_t solid_index = 0; solid_index < solids.size(); ++solid_index) {
    solids.at(solid_index).solid_id = static_cast<std::uint32_t>(solid_index + 1);
  }
  SolidContactInputs inputs_with_solids = inputs;
  inputs_with_solids.solids = solids.data();
  inputs_with_solids.solid_count = static_cast<std::uint32_t>(solids.size());
  MoveParticleCall call;
  call.error = MoveParticle(inputs_with_solids, &call.result, &call.failure);
  return call;
}

// Computes whether two moves are the same: the same error and, on success, the same hit count and velocities, bit for
// bit, and the same position too when is_position_compared.
bool ComputeIsSameMove(const MoveParticleCall& first, const MoveParticleCall& second, bool is_position_compared) {
  if (first.error != second.error) {
    return false;
  }
  if (first.error != ContactError::kNone) {
    return true;
  }
  const bool is_same_position =
      !is_position_compared || ComputeIsSameVector(first.result.position, second.result.position);
  return first.result.hit_count == second.result.hit_count && is_same_position
      && ComputeIsSameVector(first.result.velocity.for_this_step, second.result.velocity.for_this_step)
      && ComputeIsSameVector(first.result.velocity.for_next_step, second.result.velocity.for_next_step);
}

// Runs one generated move near a turning solid, checks it and counts it: a success ends at least tol / 2 from every
// solid on the side of every plate the start rules leave it on; an exact copy of the turning solid and the reverse
// order of the solids change neither the hit count nor the velocities (the settle may move the end by a unit in the
// last place, and so may the start's move, MoveToRestGapOfNearSurfaces' known limit); a failure is a close call.
void CheckTurningMove(const TurningMoveSet& set, const SolidContactInputs& inputs, TurningMoveCounts& counts) {
  ++counts.moves;
  const Sweep sweep = {
      .start = inputs.start_position,
      .velocity_for_this_step = inputs.velocity.for_this_step,
      .start_elapsed = 0,
      .duration = inputs.step_duration,
  };
  FirstHit search_hit;
  int advance_count = 0;
  static_cast<void>(FindHitOnTurningSolid(inputs, 0, sweep, &search_hit, &advance_count));
  counts.advance_counts.push_back(advance_count);
  const MoveParticleCall call = ComputeMoveWithSolids(inputs, set.solids);
  std::vector<Solid> solids_with_copy = set.solids;
  solids_with_copy.push_back(set.solids.at(0));
  const MoveParticleCall copy_call = ComputeMoveWithSolids(inputs, solids_with_copy);
  // A start closer than tol is moved off the solid and its copy at once (MoveToRestGapOfNearSurfaces), which reads the
  // solids in array order: the moved start may differ by a unit in the last place, and so may the surface velocity a
  // turning solid has there. That known limit is measured, not required to be zero.
  const Solid& turning_solid = set.solids.at(0);
  const double start_distance =
      ComputeDistanceToSolid(turning_solid, ComputeSolidPose(turning_solid, 0), inputs.start_position).distance;  // m
  if (start_distance < inputs.contact_tolerance) {
    const bool is_same_outcome = call.error == copy_call.error && call.result.hit_count == copy_call.result.hit_count;
    counts.copy_differences += is_same_outcome ? 0 : 1;
    if (call.error == ContactError::kNone && copy_call.error == ContactError::kNone) {
      const Vec3 velocity_difference = call.result.velocity.for_next_step - copy_call.result.velocity.for_next_step;
      const double speed = std::max(math::Length(call.result.velocity.for_next_step), 1.0);  // m/s
      counts.largest_moved_start_copy_difference =
          std::max(counts.largest_moved_start_copy_difference, math::Length(velocity_difference) / speed);
    }
  } else {
    counts.copy_differences += ComputeIsSameMove(call, copy_call, false) ? 0 : 1;
  }
  const std::vector<Solid> reversed_solids(set.solids.rbegin(), set.solids.rend());
  counts.order_differences += ComputeIsSameMove(call, ComputeMoveWithSolids(inputs, reversed_solids), false) ? 0 : 1;
  const ContactError error = call.error;
  const SolidContactResult& result = call.result;
  const ContactFailure& failure = call.failure;
  if (error == ContactError::kTooCloseToCall) {
    ++counts.failures_per_close_call.at(static_cast<std::size_t>(failure.close_call));
    return;
  }
  if (error != ContactError::kNone) {
    ++counts.other_errors;
    return;
  }
  CheckEndOutsideEverySolidByItsTolerance(inputs, result.position);
  CheckEndOnNearerSideOfEveryPlate(inputs, result.position);
  ++counts.successes_per_hit_count.at(result.hit_count);
  ParticleMove move;
  std::vector<SolidFace> hit_faces;
  const ContactError steps_error = ApplyStepsBeforeSettle(inputs, &move, &hit_faces);
  EXPECT_EQ(steps_error, ContactError::kNone);
  const bool is_carried = move.elapsed == inputs.step_duration && ComputeIsTurningSolid(inputs, 0)
      && result.hit_count > 0 && !hit_faces.empty();
  if (is_carried) {
    const double end_distance = ComputeDistanceToSolid(
                                    turning_solid,
                                    ComputeSolidPose(turning_solid, inputs.step_duration),
                                    move.result.position
    )
                                    .distance;  // m
    ++counts.carried_moves;
    counts.least_carried_end_share = std::min(counts.least_carried_end_share, end_distance / set.contact_tolerance);
  }
}

// Builds the generated sets of moves near turning solids: a tilted box, a flat solid, a plate and a cylinder that turn
// across their normal or axis while they move, a paddle of the generated paddle sweeps (FindFirstHit's tests), and
// the box with a fixed floor beside it.
std::vector<TurningMoveSet> BuildTurningMoveSets() {
  const Mat3 tilted_axes = math::Rotation({.x = 0.3, .y = -0.4, .z = 0.5});
  const Vec3 tilted_normal = ComputeColumn(tilted_axes, Axis::kZ);
  const Vec3 center = {.x = 0.25, .y = 0.5, .z = -0.125};
  const Vec3 linear_velocity = {.x = 0.25, .y = -0.5, .z = 0.125};
  // Each angular velocity has a part across the normal or the axis, and turns at most 1.05 rad in the step of 1 s.
  const Solid box = ToSolid(
      Box{.center = center, .axes = tilted_axes, .half_size = {.x = 0.5, .y = 0.25, .z = 1}},
      {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0.3, .y = -0.6, .z = 0.8}},
      1
  );
  const Solid flat_solid = ToSolid(
      FlatSolid{.point_on_surface = center, .normal_out_of_solid = tilted_normal},
      {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0.6, .y = 0.3, .z = -0.2}},
      1
  );
  const Solid plate = ToSolid(
      Plate{.center = center, .normal = tilted_normal, .half_thickness = 0.05},
      {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0.5, .y = -0.5, .z = 0.2}},
      1
  );
  const Solid cylinder = ToSolid(
      Cylinder{.center = center, .axis = tilted_normal, .radius = 0.5, .half_height = 1},
      {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0.7, .y = 0.2, .z = 0.1}},
      1
  );
  const Solid paddle = ToSolid(
      Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.25, .y = 0.01, .z = 0.1}},
      {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 6}},
      1
  );
  // The floor cuts the lowest part of the box, so starts near the box are often near the floor too.
  const Solid floor_below_box = move_particle_cases::BuildFlatSolid({.x = 0, .y = 0, .z = -0.9}, kUnitZ, kNoMotion, 2);
  return {
      {
          .name = "box",
          .solids = {box},
          .step_duration = 1,
          .contact_tolerance = kTolerance,
          .length_scale = 1,
          .largest_coordinate = 3,
      },
      {
          .name = "flat solid",
          .solids = {flat_solid},
          .step_duration = 1,
          .contact_tolerance = kTolerance,
          .length_scale = 1,
          .largest_coordinate = 3,
      },
      {
          .name = "plate",
          .solids = {plate},
          .step_duration = 1,
          .contact_tolerance = kTolerance,
          .length_scale = 1,
          .largest_coordinate = 3,
      },
      {
          .name = "cylinder",
          .solids = {cylinder},
          .step_duration = 1,
          .contact_tolerance = kTolerance,
          .length_scale = 1,
          .largest_coordinate = 3,
      },
      {
          .name = "paddle",
          .solids = {paddle},
          .step_duration = 1.0 / 120,
          .contact_tolerance = 0x1p-30 * 0.01,
          .length_scale = 0.25,
          .largest_coordinate = 0.5,
      },
      {
          .name = "box above floor",
          .solids = {box, floor_below_box},
          .step_duration = 0.1,
          .contact_tolerance = kTolerance,
          .length_scale = 1,
          .largest_coordinate = 3,
      },
  };
}

// Prints the counts of one generated set of moves near a turning solid, for the report.
void WriteTurningMoveCounts(const char* set_name, const TurningMoveCounts& counts) {
  std::vector<int> advance_counts = counts.advance_counts;
  std::ranges::sort(advance_counts);
  int advance_sum = 0;
  for (const int advance_count : advance_counts) {
    advance_sum += advance_count;
  }
  const std::array<int, kMaxHitsPerStep + 1>& hits = counts.successes_per_hit_count;
  const std::array<int, 8>& failures = counts.failures_per_close_call;
  std::cout << set_name << ": moves " << counts.moves << "; successes by hit count 0 " << hits.at(0) << ", 1 "
            << hits.at(1) << ", 2 " << hits.at(2) << ", 3 " << hits.at(3) << ", 4 " << hits.at(4) << "; carried "
            << counts.carried_moves << ", least end before the settle " << counts.least_carried_end_share
            << " tol; too many hits " << failures.at(static_cast<std::size_t>(ContactCloseCall::kTooManyHits))
            << ", squeezed " << failures.at(static_cast<std::size_t>(ContactCloseCall::kSqueezedBetweenSolids))
            << ", hover near turning solid "
            << failures.at(static_cast<std::size_t>(ContactCloseCall::kHoverNearTurningSolid)) << ", near other solid "
            << failures.at(static_cast<std::size_t>(ContactCloseCall::kTurningSolidNearOtherSolid))
            << ", second reacting solid "
            << failures.at(static_cast<std::size_t>(ContactCloseCall::kSecondReactingSolid)) << ", ends too close "
            << failures.at(static_cast<std::size_t>(ContactCloseCall::kEndsTooCloseToSolid)) << "; copy differences "
            << counts.copy_differences << " (moved starts: largest velocity difference "
            << counts.largest_moved_start_copy_difference << "), order differences " << counts.order_differences
            << "; advances mean " << static_cast<double>(advance_sum) / static_cast<double>(advance_counts.size())
            << ", 99th percentile " << advance_counts.at((advance_counts.size() * 99) / 100) << ", most "
            << advance_counts.back() << "\n";
}

// Fails the calling test unless a generated set of moves near a turning solid had no difference with an exact copy of
// the turning solid or with the solids in reverse order.
void CheckSameTurningMovesWithCopyAndOrder(const TurningMoveSet& set, const TurningMoveCounts& counts) {
  EXPECT_EQ(counts.copy_differences, 0) << set.name;
  EXPECT_EQ(counts.order_differences, 0) << set.name;
  // Beside a cylinder's rim the normal is read from a vector about tol long, so a last-place change of the moved
  // start, ulp(coordinate), turns it by about ulp(coordinate) / tol (MoveToRestGapOfNearSurfaces, Order): 2^-22 at
  // tol = 2^-30 m and coordinates near 1 m, up to about 2^-12 at the smallest tolerance. Four units of the set's
  // largest coordinate over its tolerance.
  const double largest_difference = 4 * 0x1p-52 * set.largest_coordinate / set.contact_tolerance;  // No unit
  EXPECT_LE(counts.largest_moved_start_copy_difference, largest_difference) << set.name;
}

// Fails the calling test unless a generated set of moves near a turning solid had no error but close calls, none of
// them kTooManyHits or kEndsTooCloseToSolid, more than half its moves carried, and every carry ending at 0.99 tol or
// more before the settle.
void CheckTurningMoveOutcomes(const char* set_name, const TurningMoveCounts& counts) {
  const std::array<int, 8>& failures = counts.failures_per_close_call;
  EXPECT_EQ(counts.other_errors, 0) << set_name;
  EXPECT_EQ(failures.at(static_cast<std::size_t>(ContactCloseCall::kTooManyHits)), 0) << set_name;
  EXPECT_EQ(failures.at(static_cast<std::size_t>(ContactCloseCall::kEndsTooCloseToSolid)), 0) << set_name;
  EXPECT_GT(counts.carried_moves, counts.moves / 2) << set_name;
  EXPECT_GE(counts.least_carried_end_share, 0.99) << set_name;
}

TEST(MoveParticle, GeneratedMovesNearTurningSolids_EndOutsideEverySolid) {
  // 2000 moves per set (BuildTurningMoveSets, BuildTurningMove), friction 0 to 1, a start overlap tolerance of 2 tol.
  // Every success ends at least tol / 2 from every solid (the end check's promise, checked here by the distance
  // itself); an exact copy of the turning solid, or the solids in reverse order, change no hit count and no velocity,
  // bit for bit; no move ends in kTooManyHits, which every hit on a turning solid did before the carry; a carry ends no
  // nearer than 0.99 tol before the settle.
  constexpr int kMovesPerSet = 2000;
  for (const TurningMoveSet& set : BuildTurningMoveSets()) {
    CaseGenerator generator;
    TurningMoveCounts counts;
    SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({.step_duration = set.step_duration});
    inputs.solids = set.solids.data();
    inputs.solid_count = static_cast<std::uint32_t>(set.solids.size());
    inputs.contact_tolerance = set.contact_tolerance;
    inputs.start_overlap_tolerance = 2 * set.contact_tolerance;
    ASSERT_EQ(CheckSolidContactInputs(inputs), ContactError::kNone) << set.name;
    ASSERT_TRUE(ComputeIsTurningSolid(inputs, 0)) << set.name;
    for (int move_index = 0; move_index < kMovesPerSet; ++move_index) {
      if (BuildTurningMove(set, generator, &inputs)) {
        CheckTurningMove(set, inputs, counts);
      }
    }
    WriteTurningMoveCounts(set.name, counts);
    EXPECT_GT(counts.moves, kMovesPerSet * 9 / 10) << set.name;
    CheckTurningMoveOutcomes(set.name, counts);
    CheckSameTurningMovesWithCopyAndOrder(set, counts);
  }
}

}  // namespace
}  // namespace fluidity::contact
