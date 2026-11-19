#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_test_cases.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

// The tests that replace the rounding enclosures: a particle that meets a solid stops at the rest gap, one contact
// tolerance outside it, and the tolerance is set from the scene's size, so rounding far from the origin never brings it
// inside. They run with the contact tolerance of a real scene, not the test tolerance of the analytic tests. They show
// what the rest gap does; the settle at the end of a step, for solids that close on a particle, has tests of its own
// (ApplySettleAtEnd in move_particle_test.cc), and a settle that does nothing passes these.

namespace fluidity::contact {
namespace {

// --- Far from the origin: every end outside ---

TEST(MoveParticle, ObliqueMovingSolidFarFromOrigin_NeverEndsInside) {
  // BuildObliqueCases(10000): one moving flat solid up to 1 km from the origin, hit at 0.1 to 0.9 of the step, with
  // the production tolerance there (about 1e-9 m, so a coordinate's rounding unit is about tol / 9000). Every move
  // succeeds and ends at least tol / 2 outside the solid at its end pose. The control: the same paths hit at the
  // surface itself and moved on without a settle end a few rounding units on either side of it, so about half of them
  // inside (measured 4,886); the bounds 4,000 and 6,000 leave about 20 standard deviations of a fair coin on each side.
  // The rest gap is what keeps the particles outside.
  const EndDistanceCounts counts = ComputeEndDistanceCounts(BuildObliqueCases(10000));
  std::cout << "oblique far from origin: cases " << counts.cases << ", failures " << counts.failures << ", hits "
            << counts.hits << ", ends closer than tol / 2 " << counts.ends_closer_than_half_tolerance
            << ", least end distance " << counts.least_end_distance_share << " tol; naive control ends inside "
            << counts.naive_ends_inside << "\n";
  EXPECT_EQ(counts.cases, 10000);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_EQ(counts.ends_closer_than_half_tolerance, 0);
  EXPECT_EQ(counts.hits, 10000);  // Every path is aimed at the solid.
  EXPECT_GE(counts.least_end_distance_share, 0.5);
  EXPECT_GE(counts.naive_ends_inside, 4000);
  EXPECT_LE(counts.naive_ends_inside, 6000);
}

TEST(MoveParticle, RoundSolidsFarFromOrigin_NeverEndInside) {
  // BuildRoundCases(10000): one moving sphere or cylinder up to 1 km from the origin, with the production tolerance
  // there. Every move succeeds and ends at least tol / 2 outside the solid at its end pose. The rest gap's proof does
  // not read the shape, so the control of the oblique set is not repeated as a check; it is printed.
  const EndDistanceCounts counts = ComputeEndDistanceCounts(BuildRoundCases(10000));
  std::cout << "round far from origin: cases " << counts.cases << ", failures " << counts.failures << ", hits "
            << counts.hits << ", ends closer than tol / 2 " << counts.ends_closer_than_half_tolerance
            << ", least end distance " << counts.least_end_distance_share << " tol; naive control ends inside "
            << counts.naive_ends_inside << "\n";
  EXPECT_EQ(counts.cases, 10000);
  EXPECT_EQ(counts.failures, 0);
  EXPECT_EQ(counts.ends_closer_than_half_tolerance, 0);
  EXPECT_GE(counts.hits, 4500);
  EXPECT_GE(counts.least_end_distance_share, 0.5);
}

// --- Many steps on one moving solid ---

TEST(MoveParticle, SlidesTenThousandSteps_NeverStartsInside) {
  // A particle pressed onto a moving oblique flat solid for 10,000 steps of 1/240 s, each step starting where the last
  // ended, with both velocities set again each step to the solid's velocity + {0.2, 0.1, -0.3} - 0.5 normal m/s, and
  // friction 0.3. The production contact_tolerance of a scene 100 m across. Every step succeeds with one hit, the
  // response at the start, and ends between tol / 2 and 2 tol from the solid: rounding never builds up over the steps.
  const Vec3 normal = math::Normalized({.x = 1, .y = 2, .z = 3});
  const SolidMotion motion = {.linear_velocity = {.x = 0.5, .y = -0.25, .z = 0.125}, .angular_velocity = {}};
  const Vec3 point_on_surface = {.x = 0.3, .y = -0.2, .z = 0.1};  // m
  std::array solids = {move_particle_cases::BuildFlatSolid(point_on_surface, normal, motion, 1)};
  const double contact_tolerance = ComputeContactTolerance({
      .largest_coordinate = 100,
      .smallest_cell_width = 0.01,
      .thinnest_solid = 0.01,
  });  // m
  constexpr double kStepDuration = 1.0 / 240;  // s
  const Vec3 velocity = motion.linear_velocity + Vec3{.x = 0.2, .y = 0.1, .z = -0.3} - 0.5 * normal;  // m/s
  SolidContactInputs inputs;
  inputs.start_position = point_on_surface + contact_tolerance * normal;
  inputs.solids = solids.data();
  inputs.solid_count = 1;
  inputs.step_duration = kStepDuration;
  inputs.friction = 0.3;
  inputs.contact_tolerance = contact_tolerance;
  int failed_steps = 0;
  int steps_not_one_hit = 0;
  int steps_out_of_range = 0;
  double least_share = 2;  // Of tol.
  double largest_share = 0;  // Of tol.
  for (int step = 0; step < 10000; ++step) {
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    SolidContactResult result;
    ContactFailure failure;
    if (MoveParticle(inputs, &result, &failure) != ContactError::kNone) {
      ++failed_steps;
      break;
    }
    Solid& solid = solids.at(0);
    solid.pose_at_step_start = ComputeSolidPose(solid, kStepDuration);
    const double end_share =
        ComputeDistanceToSolid(solid, solid.pose_at_step_start, result.position).distance / contact_tolerance;
    least_share = std::min(least_share, end_share);
    largest_share = std::max(largest_share, end_share);
    steps_out_of_range += end_share >= 0.5 && end_share <= 2 ? 0 : 1;
    steps_not_one_hit += result.hit_count == 1 ? 0 : 1;
    inputs.start_position = result.position;
  }
  std::cout << "ten thousand slide steps: least end distance " << least_share << " tol, largest " << largest_share
            << " tol\n";
  EXPECT_EQ(failed_steps, 0);
  EXPECT_EQ(steps_out_of_range, 0);
  EXPECT_EQ(steps_not_one_hit, 0);
}

// --- A thin plate at the production tolerance ---

TEST(MoveParticle, FastParticleThinPlate_IsStopped) {
  // A plate 0.1 mm thick at the origin and a particle falling onto it at 1000 m/s from 5 m, so a step of 0.01 s would
  // carry it 10 m, through the plate and far beyond. The contact_tolerance is the one ComputeContactScale and
  // ComputeContactTolerance give for cells of 0.01 m (2^-30 * 0.01 m, about 9.3e-12 m), which CheckContactTolerance
  // passes. The particle is hit on the plate's upper face: hit_fraction (5 - 5e-5 - tol) / 10, end on the face's rest
  // gap with no velocity left.
  const Plate plate = {.center = {}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 5e-5};
  const std::array solids = {
      ToSolid(plate, move_particle_cases::kNoMotion, 1),
  };
  constexpr double kStepDuration = 0.01;  // s
  const ContactScale scale = ComputeContactScale(solids.data(), 1, kStepDuration, 0.01);
  const double contact_tolerance = ComputeContactTolerance(scale);  // m
  ASSERT_EQ(CheckContactTolerance(scale, contact_tolerance), ContactError::kNone);
  SolidContactInputs inputs;
  inputs.start_position = {.x = 0, .y = 0, .z = 5};
  inputs.velocity = {.for_next_step = {.x = 0, .y = 0, .z = -1000}, .for_this_step = {.x = 0, .y = 0, .z = -1000}};
  inputs.solids = solids.data();
  inputs.solid_count = 1;
  inputs.step_duration = kStepDuration;
  inputs.contact_tolerance = contact_tolerance;
  SolidContactResult result;
  ContactFailure failure;
  ASSERT_EQ(MoveParticle(inputs, &result, &failure), ContactError::kNone);
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  EXPECT_EQ(result.side_at_hit, SolidSide::kAlongNormal);
  // The hit fraction is a share of the 10 m path: tol / 1024 of it is (tol / 1024) / 10 m, 9e-16, about 9 times the
  // hit time's rounding (5 m over 1000 m/s, a share of about 1e-16 of the step).
  const double path_length = 10;  // m: 1000 m/s for 0.01 s.
  EXPECT_NEAR(
      result.hit_fraction,
      (5 - 5e-5 - contact_tolerance) / path_length,
      contact_tolerance / 1024 / path_length
  );
  EXPECT_NEAR(result.position.z, 5e-5 + contact_tolerance, contact_tolerance / 1024);
  ExpectVec3Near(result.velocity.for_this_step, {}, 1e-12);
}

// --- The squeeze cap reads a solid's speed by its length ---

// The floor z <= 0 rising at 1 m/s and sliding along y at sliding_speed, a fixed side meeting it at 0.8 degrees along
// the y axis, and the particle at rest at the rest point of both, tol from each, for 0.01 s, friction 0.
MoveParticleCase BuildSlidingFloorUnderSide(double sliding_speed) {
  const double wedge_angle = 0.8 * std::numbers::pi / 180;  // rad
  const double contact_tolerance = move_particle_cases::kTolerance;  // m
  const Vec3 rest_point = {.x = contact_tolerance / std::tan(wedge_angle / 2), .y = 0, .z = contact_tolerance};
  return {
      .name = "SlidingFloorUnderSideAtPointEightDegrees",
      .solids =
          {
              move_particle_cases::BuildMovingFloor(0, {.x = 0, .y = sliding_speed, .z = 1}, 1),
              move_particle_cases::BuildMovingWedgeSide(wedge_angle, 0, 2),
          },
      .inputs = move_particle_cases::BuildSeveralSurfaceInputs(rest_point, {}, 0.01),
  };
}

TEST(MoveParticle, SlidingFloorUnderSide_SqueezeCapReadsTheFloorsSpeedByItsLength) {
  // The particle must leave the wedge at {1 / tan(0.8 degrees), 0, 1}, 1 / sin(0.8 degrees) = 71.622 m/s, whatever the
  // floor's sliding along the wedge's edge, which a frictionless response does not read. The cap is 64 times the
  // largest speed involved, the floor's Length (method paper <sec-move-particle>): sliding at 0.75 m/s its speed is
  // 1.25 m/s, a cap of 80 m/s, and the particle is carried out; at 0.4 m/s it is 1.0770 m/s, a cap of 68.93 m/s, and
  // the particle is squeezed. A cap that read the floor's largest axis, 1 m/s in both, would squeeze both: no turn of a
  // scene can show that, since a turn that permutes axes keeps every vector's largest axis.
  const double wedge_angle = 0.8 * std::numbers::pi / 180;  // rad
  const MoveParticleCase carried_scene = BuildSlidingFloorUnderSide(0.75);
  SolidContactResult result;
  ContactFailure failure;
  ASSERT_EQ(MoveParticle(ToSolidContactInputs(carried_scene), &result, &failure), ContactError::kNone);
  EXPECT_EQ(result.hit_count, 1);
  const Vec3 escape_velocity = {.x = 1 / std::tan(wedge_angle), .y = 0, .z = 1};  // m/s
  ExpectVec3Near(result.velocity.for_this_step, escape_velocity, 1e-9 * escape_velocity.x);
  ExpectVec3Near(result.velocity.for_next_step, escape_velocity, 1e-9 * escape_velocity.x);
  const MoveParticleCase squeezed_scene = BuildSlidingFloorUnderSide(0.4);
  EXPECT_EQ(MoveParticle(ToSolidContactInputs(squeezed_scene), &result, &failure), ContactError::kTooCloseToCall);
  EXPECT_EQ(failure.close_call, ContactCloseCall::kSqueezedBetweenSolids);
}

// --- A scene turned rigidly by an exact turn ---

// Turns a whole scene by turn about the origin: the start, both velocities, and every solid's origin, axes and motion.
MoveParticleCase TurnScene(const MoveParticleCase& scene, Mat3 turn) {
  MoveParticleCase turned = scene;
  SolidContactInputs& inputs = turned.inputs;
  inputs.start_position = turn * scene.inputs.start_position;
  inputs.velocity = {
      .for_next_step = turn * scene.inputs.velocity.for_next_step,
      .for_this_step = turn * scene.inputs.velocity.for_this_step,
  };
  for (Solid& solid : turned.solids) {
    solid.pose_at_step_start = {
        .origin = turn * solid.pose_at_step_start.origin,
        .axes = turn * solid.pose_at_step_start.axes,
    };
    solid.motion = {
        .linear_velocity = turn * solid.motion.linear_velocity,
        .angular_velocity = turn * solid.motion.angular_velocity,
    };
  }
  return turned;
}

// The columns of a matrix: a solid's own x, y and z axes in world.
std::array<Vec3, 3> ToColumns(Mat3 axes) {
  return {
      Vec3{.x = axes.xx, .y = axes.yx, .z = axes.zx},
      Vec3{.x = axes.xy, .y = axes.yy, .z = axes.zy},
      Vec3{.x = axes.xz, .y = axes.yz, .z = axes.zz},
  };
}

// Computes whether a vector has three nonzero components.
bool ComputeHasThreeNonzeroComponents(Vec3 vector) {
  return vector.x != 0 && vector.y != 0 && vector.z != 0;
}

// Computes whether some vector of a scene's inputs has three nonzero components: the start, both velocities, and every
// solid's origin, axes, half size and motion. When none has, every sum of products the move makes from them has at
// most two nonzero terms, and the same terms, in a turned scene whose entries are 0 and +-1, are added in the other
// order or with signs flipped, which is exact (a + b = b + a), so the turned move is the move turned, bit for bit.
bool ComputeHasVectorOutOfEveryAxisPlane(const MoveParticleCase& scene) {
  const SolidContactInputs& inputs = scene.inputs;
  bool has_such_vector = ComputeHasThreeNonzeroComponents(inputs.start_position)
      || ComputeHasThreeNonzeroComponents(inputs.velocity.for_this_step)
      || ComputeHasThreeNonzeroComponents(inputs.velocity.for_next_step);
  for (const Solid& solid : scene.solids) {
    const std::array<Vec3, 3> columns = ToColumns(solid.pose_at_step_start.axes);
    has_such_vector = has_such_vector || ComputeHasThreeNonzeroComponents(solid.pose_at_step_start.origin)
        || ComputeHasThreeNonzeroComponents(solid.half_size)
        || ComputeHasThreeNonzeroComponents(solid.motion.linear_velocity)
        || ComputeHasThreeNonzeroComponents(solid.motion.angular_velocity)
        || std::ranges::any_of(columns, ComputeHasThreeNonzeroComponents);
  }
  return has_such_vector;
}

// Computes whether two axes of two solids share one direction without being equal: every component within
// kSameNormalAxisDifference, not every component equal. The joint response (method paper <sec-move-particle>) then
// reads the shared direction's normal from the surface that comes first when the surfaces are sorted by their normal's
// x, y, z, and charges a tie to the first in that order; a turn that permutes the axes permutes that order.
bool ComputeIsSameDirectionNotEqual(Vec3 first, Vec3 second) {
  const Vec3 difference = first - second;
  const bool is_within = std::abs(difference.x) <= kSameNormalAxisDifference
      && std::abs(difference.y) <= kSameNormalAxisDifference && std::abs(difference.z) <= kSameNormalAxisDifference;
  const bool is_equal = difference.x == 0 && difference.y == 0 && difference.z == 0;
  return is_within && !is_equal;
}

// Computes whether two solids of a scene have surfaces that share one direction without equal normals
// (ComputeIsSameDirectionNotEqual): any axis of one against any axis of the other, either way round.
bool ComputeHasSurfacesSharingDirection(const MoveParticleCase& scene) {
  for (std::size_t first = 0; first < scene.solids.size(); ++first) {
    for (std::size_t second = first + 1; second < scene.solids.size(); ++second) {
      for (const Vec3 first_axis : ToColumns(scene.solids.at(first).pose_at_step_start.axes)) {
        for (const Vec3 second_axis : ToColumns(scene.solids.at(second).pose_at_step_start.axes)) {
          if (ComputeIsSameDirectionNotEqual(first_axis, second_axis)
              || ComputeIsSameDirectionNotEqual(first_axis, -second_axis)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Computes how much a scene's flat surfaces amplify a rounding unit of their normals in a joint response or a rest
// point: 1 over the smallest sine between two normals, or the smallest |triple product| of three, of its flat solids
// and plates, at least 1. Normals closer than 2^-40 count as one direction (method paper <sec-move-particle>), and the
// result is capped at 2^20: the solver uses no pair below kVelocityPairLimitShare = 2^-40 (a sine of about 1.4e-6) nor
// a triple below kVelocityTripleLimit = 2^-20, so the change never solves along a worse pair or triple.
double ComputeNormalsAmplification(const MoveParticleCase& scene) {
  std::vector<Vec3> normals;
  for (const Solid& solid : scene.solids) {
    const bool is_flat = solid.shape == SolidShape::kFlatSolid || solid.shape == SolidShape::kPlate;
    if (is_flat) {
      normals.push_back(ToColumns(solid.pose_at_step_start.axes).at(2));
    }
  }
  double least_measure = 1;
  const std::size_t count = normals.size();
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = first + 1; second < count; ++second) {
      const Vec3 cross = math::Cross(normals.at(first), normals.at(second));
      const double sine = math::Length(cross);
      least_measure = sine > kSameNormalAxisDifference ? std::min(least_measure, sine) : least_measure;
      for (std::size_t third = second + 1; third < count; ++third) {
        const double triple = std::abs(math::Dot(cross, normals.at(third)));
        least_measure = triple > kSameNormalAxisDifference ? std::min(least_measure, triple) : least_measure;
      }
    }
  }
  return std::min(1 / least_measure, 0x1p20);
}

// How a turned scene's floating values are compared, read from the scene alone.
struct TurnComparison {
  // No unit. 0: bit for bit. Otherwise the largest difference, as a share of the scene's coordinate scale (positions)
  // or speed scale (velocities), or of 1 (the hit fraction and the normal).
  double share = 0;
  bool compares_reacting_solid = true;  // false when surfaces share a direction: a tie may charge the other.
  double coordinate_scale = 0;  // m: the largest |coordinate| of the inputs plus the reach, or the speed times T.
  double speed_scale = 0;  // m/s: the largest speed of the particle or of a solid's surface near its path.
};

// Computes how a turned scene is compared under a turn whose entries are 0 and +-1, given the speeds of the two moves'
// velocities.
//
// Method: bit for bit when no input vector has three nonzero components (ComputeHasVectorOutOfEveryAxisPlane).
//   Otherwise sums of three products are added in another order, and the joint response sorts the surfaces of a joint
//   response by their normal's x, y, z, which the turn permutes, so the subsets of the solve are walked in another
//   order: each such sum may differ by about 2 rounding units of its largest term, a move makes a few in series (the
//   start, a hit, a response, the settle), and a joint response or a rest point amplifies them by its normals'
//   conditioning (ComputeNormalsAmplification). So the share is 2^-48, 16 rounding units, times that amplification
//   (measured at most 3 rounding units times it). The speed scale is the largest speed of the particle, of a solid's
//   surface near its path, and of either move's resulting velocities, since a response's rounding is relative to the
//   speeds it computes, as its allowance is (method paper <sec-move-particle>); the coordinate scale is the largest
//   |coordinate| of the inputs plus the solids' reach, or that speed times the step. When two surfaces share a
//   direction without equal normals (ComputeHasSurfacesSharingDirection), the turn may make the other one's normal the
//   direction's, which moves a change by up to kSameNormalAxisDifference of the speeds (method paper
//   <sec-move-particle>): the share is at least 2^-40 (measured 901 rounding units, 2^-42.2), and the reacting solid of
//   a tie is not compared.
TurnComparison ComputeTurnComparison(const MoveParticleCase& scene, double largest_result_speed) {
  const SolidContactInputs& inputs = scene.inputs;
  const auto largest_axis = [](Vec3 vector) {
    return std::max({std::abs(vector.x), std::abs(vector.y), std::abs(vector.z)});
  };
  double coordinate_scale = largest_axis(inputs.start_position);  // m
  double speed_scale = std::max(
      {math::Length(inputs.velocity.for_this_step), math::Length(inputs.velocity.for_next_step), largest_result_speed}
  );  // m/s
  for (const Solid& solid : scene.solids) {
    const double reach = solid.shape == SolidShape::kFlatSolid ? 0 : math::Length(solid.half_size);  // m
    const double lever = math::Length(inputs.start_position - solid.pose_at_step_start.origin) + reach;  // m
    coordinate_scale = std::max(coordinate_scale, largest_axis(solid.pose_at_step_start.origin) + reach);
    speed_scale = std::max(
        speed_scale,
        math::Length(solid.motion.linear_velocity) + math::Length(solid.motion.angular_velocity) * lever
    );
  }
  coordinate_scale = std::max(coordinate_scale, speed_scale * inputs.step_duration);
  const bool shares_direction = ComputeHasSurfacesSharingDirection(scene);
  double share = ComputeHasVectorOutOfEveryAxisPlane(scene) ? 0x1p-48 * ComputeNormalsAmplification(scene) : 0;
  share = shares_direction ? std::max(share, kSameNormalAxisDifference) : share;
  return {
      .share = share,
      .compares_reacting_solid = !shares_direction,
      .coordinate_scale = coordinate_scale,
      .speed_scale = speed_scale,
  };
}

// Computes whether two values agree within share of scale; with share 0, bit for bit.
//
// clang-tidy: the two values play the same part, so swapping them changes nothing; the swappable-parameters check
// flags them.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool ComputeIsWithinShare(double first, double second, double share, double scale) {
  return share == 0 ? first == second : std::abs(first - second) <= share * scale;
}

// Computes whether two vectors agree axis by axis (ComputeIsWithinShare).
//
// clang-tidy: as the form for two values.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool ComputeIsWithinShare(Vec3 first, Vec3 second, double share, double scale) {
  return ComputeIsWithinShare(first.x, second.x, share, scale) && ComputeIsWithinShare(first.y, second.y, share, scale)
      && ComputeIsWithinShare(first.z, second.z, share, scale);
}

// Fails the calling test unless the turned move's result is the first result turned: every discrete field equal (the
// reacting solid as TurnComparison says), and every floating value within the comparison's share.
void ExpectResultTurned(
    const SolidContactResult& result,
    const SolidContactResult& turned,
    Mat3 turn,
    const TurnComparison& comparison,
    const std::string& label
) {
  const double share = comparison.share;
  const double coordinate = comparison.coordinate_scale;  // m
  const double speed = comparison.speed_scale;  // m/s
  const bool is_same_hits = result.kind == turned.kind && result.hit_count == turned.hit_count
      && result.solid_index == turned.solid_index && result.face == turned.face
      && result.side_at_hit == turned.side_at_hit;
  const bool is_same_record = ComputeIsWithinShare(result.hit_fraction, turned.hit_fraction, share, 1)
      && ComputeIsWithinShare(turn * result.normal_out_of_solid, turned.normal_out_of_solid, share, 1)
      && ComputeIsWithinShare(result.approach_speed.for_this_step, turned.approach_speed.for_this_step, share, speed)
      && ComputeIsWithinShare(result.approach_speed.for_next_step, turned.approach_speed.for_next_step, share, speed);
  const bool is_same_motion = ComputeIsWithinShare(turn * result.position, turned.position, share, coordinate)
      && ComputeIsWithinShare(turn * result.velocity.for_this_step, turned.velocity.for_this_step, share, speed)
      && ComputeIsWithinShare(turn * result.velocity.for_next_step, turned.velocity.for_next_step, share, speed);
  const SolidReaction& reaction = result.reaction;
  const SolidReaction& turned_reaction = turned.reaction;
  const bool is_same_reaction = (reaction.solid_index == turned_reaction.solid_index
                                 || !comparison.compares_reacting_solid)
      && ComputeIsWithinShare(turn * reaction.hit_position, turned_reaction.hit_position, share, coordinate)
      && ComputeIsWithinShare(turn * reaction.velocity_for_next_step_change,
                              turned_reaction.velocity_for_next_step_change,
                              share,
                              speed);
  EXPECT_TRUE(is_same_hits) << label << ": kind, hit count, solid, face or side";
  EXPECT_TRUE(is_same_record) << label << ": hit fraction, normal or approach speed";
  EXPECT_TRUE(is_same_motion) << label << ": position or velocity";
  EXPECT_TRUE(is_same_reaction) << label << ": reaction";
}

// What the turned scenes gave, counted for the report.
struct TurnCounts {
  int scenes = 0;  // Scenes compared: those without a domain box.
  int successes = 0;
  int scenes_bit_for_bit = 0;
  int scenes_sharing_direction = 0;
};

// Moves a scene and the same scene turned, and fails the calling test unless the turned move is the first move turned:
// the same error, close call and solid named, and on success the result turned (ExpectResultTurned).
void ExpectTurnedMoveIsMoveTurned(
    const MoveParticleCase& scene,
    Mat3 turn,
    std::string_view turn_name,
    TurnCounts& counts
) {
  const MoveParticleCase turned_scene = TurnScene(scene, turn);
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = MoveParticle(ToSolidContactInputs(scene), &result, &failure);
  SolidContactResult turned_result;
  ContactFailure turned_failure;
  const ContactError turned_error = MoveParticle(ToSolidContactInputs(turned_scene), &turned_result, &turned_failure);
  const std::string label = std::string(turn_name) + " " + std::string(scene.name);
  EXPECT_EQ(turned_error, error) << label;
  EXPECT_EQ(turned_failure.close_call, failure.close_call) << label;
  EXPECT_EQ(turned_failure.solid_id, failure.solid_id) << label;
  const double largest_result_speed = std::max({
      math::Length(result.velocity.for_this_step),
      math::Length(result.velocity.for_next_step),
      math::Length(turned_result.velocity.for_this_step),
      math::Length(turned_result.velocity.for_next_step),
  });  // m/s, 0 for a move that failed: its result is not written.
  const TurnComparison comparison = ComputeTurnComparison(scene, largest_result_speed);
  ++counts.scenes;
  counts.scenes_bit_for_bit += comparison.share == 0 ? 1 : 0;
  counts.scenes_sharing_direction += comparison.compares_reacting_solid ? 0 : 1;
  const bool are_both_successes = error == ContactError::kNone && turned_error == ContactError::kNone;
  if (are_both_successes) {
    ++counts.successes;
    ExpectResultTurned(result, turned_result, turn, comparison, label);
  }
}

// Moves every scene of BuildMoveParticleCases without a domain box (its closed sides are named by world axis, so a turn
// would also have to rename them) and the same scene turned, and fails the calling test on any difference
// (ExpectTurnedMoveIsMoveTurned).
void ExpectEveryTurnedSceneMovesTurned(Mat3 turn, std::string_view turn_name) {
  TurnCounts counts;
  for (const MoveParticleCase& scene : BuildMoveParticleCases()) {
    const bool has_domain_box = std::ranges::any_of(scene.solids, [](const Solid& solid) {
      return solid.shape == SolidShape::kDomainBox;
    });
    if (!has_domain_box) {
      ExpectTurnedMoveIsMoveTurned(scene, turn, turn_name, counts);
    }
  }
  std::cout << turn_name << ": scenes " << counts.scenes << ", successes " << counts.successes << ", bit for bit "
            << counts.scenes_bit_for_bit << ", surfaces sharing a direction " << counts.scenes_sharing_direction
            << "\n";
  EXPECT_EQ(counts.scenes, 215);
  EXPECT_EQ(counts.successes, 174);
}

TEST(MoveParticle, SceneTurnedExactly_ResultTurnsWithIt) {
  // Contact has no favoured direction: every scene turned about the origin moves as the scene does, turned. The turns
  // are exact, entries 0 and +-1, built without sin or cos: a quarter turn about x (it takes Y-up to Z-up), about y
  // and about z, and the turn that takes x to y, y to z and z to x. Every discrete field is equal; floating values are
  // equal bit for bit, or within the share ComputeTurnComparison derives from the scene. Measured: of 215 scenes, 106
  // have no vector out of every axis plane and are bit for bit under every turn (one of them, two reacting floors whose
  // normals are 2e-13 apart, shares a direction and charges the other floor under the turn about y); the others differ
  // by at most 1.3 rounding units of their scale, and the two moving corners with a coincident face by 2602 units.
  const Mat3 quarter_turn_about_x = {.xx = 1, .xy = 0, .xz = 0, .yx = 0, .yy = 0, .yz = -1, .zx = 0, .zy = 1, .zz = 0};
  const Mat3 quarter_turn_about_y = {.xx = 0, .xy = 0, .xz = 1, .yx = 0, .yy = 1, .yz = 0, .zx = -1, .zy = 0, .zz = 0};
  const Mat3 quarter_turn_about_z = {.xx = 0, .xy = -1, .xz = 0, .yx = 1, .yy = 0, .yz = 0, .zx = 0, .zy = 0, .zz = 1};
  const Mat3 axes_one_step_round = {.xx = 0, .xy = 0, .xz = 1, .yx = 1, .yy = 0, .yz = 0, .zx = 0, .zy = 1, .zz = 0};
  ExpectEveryTurnedSceneMovesTurned(quarter_turn_about_x, "quarter turn about x");
  ExpectEveryTurnedSceneMovesTurned(quarter_turn_about_y, "quarter turn about y");
  ExpectEveryTurnedSceneMovesTurned(quarter_turn_about_z, "quarter turn about z");
  ExpectEveryTurnedSceneMovesTurned(axes_one_step_round, "x to y to z to x");
}

}  // namespace
}  // namespace fluidity::contact
