// Scene tests of contact: what a solid's reaction adds up to over many particles, a particle on a round solid over many
// steps, and the cases of the old contact code on branch main.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_error_message.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/contact_main_comparison_cases.h"
#include "fluidity/contact/testing/move_particles_runs.h"
#include "fluidity/contact/testing/move_particles_scene_cases.h"

namespace fluidity::contact {
namespace {

// The largest absolute axis of a vector.
double ComputeLargestAxis(Vec3 vector) {
  return std::max({std::abs(vector.x), std::abs(vector.y), std::abs(vector.z)});
}

// Classifies the one particle of a range against its one flat solid that moves without turning: the solid's distance
// changes at the constant rate Dot(normal, for_this_step - solid velocity), so the path's gap at the end of the step is
// start distance - tol + rate * T. Below -1e-9 m it hits, above 1e-9 m it misses.
ExpectedHit ComputeFlatSolidPathClass(const ParticleRange& range) {
  const Solid& solid = range.solids[0];
  const Vec3 normal = {
      .x = solid.pose_at_step_start.axes.xz,
      .y = solid.pose_at_step_start.axes.yz,
      .z = solid.pose_at_step_start.axes.zz,
  };
  const double start_distance = math::Dot(normal, range.start_position[0] - solid.pose_at_step_start.origin);  // m
  const double rate = math::Dot(normal, range.velocity_for_this_step[0] - solid.motion.linear_velocity);  // m/s
  const double end_gap = start_distance - range.contact_tolerance + (rate * range.step_duration);  // m
  if (end_gap < -1e-9) {
    return ExpectedHit::kHit;
  }
  return end_gap > 1e-9 ? ExpectedHit::kMiss : ExpectedHit::kTooCloseToTell;
}

// Whether one particle's reaction is its change of velocity.for_next_step: with a reaction, after minus before within
// 2^-44 of the larger of the two on every axis; without one, for_next_step unchanged bit for bit.
bool ComputeIsReactionTheChange(Vec3 before, Vec3 after, const SolidReaction& reaction) {
  if (reaction.solid_index == kNoSolid) {
    return ToBits(after.x) == ToBits(before.x) && ToBits(after.y) == ToBits(before.y)
        && ToBits(after.z) == ToBits(before.z);
  }
  const double bound = 0x1p-44 * std::max(ComputeLargestAxis(before), ComputeLargestAxis(after));  // m/s
  const Vec3 difference = reaction.velocity_for_next_step_change - (after - before);  // m/s
  return ComputeLargestAxis(difference) <= bound;
}

// Classifies whether the one particle of a range must be charged to its one flat solid, which moves without turning:
// the hit's response changes velocity.for_next_step only when that velocity approaches the solid, Dot(normal,
// for_next_step - solid velocity) < 0, and a flat solid is hit once, so a clear hit whose for_next_step approaches by
// more than 1e-9 m/s must react (kHit), one whose for_next_step leaves by more than 1e-9 m/s must not (kMiss: the
// response returns it bit for bit and charges nothing), and a path that is not a clear hit must not react either.
ExpectedHit ComputeFlatSolidReactionClass(const ParticleRange& range, ExpectedHit expected_hit) {
  if (expected_hit != ExpectedHit::kHit) {
    return expected_hit;
  }
  const Solid& solid = range.solids[0];
  const Vec3 normal = {
      .x = solid.pose_at_step_start.axes.xz,
      .y = solid.pose_at_step_start.axes.yz,
      .z = solid.pose_at_step_start.axes.zz,
  };
  const double next_step_rate = math::Dot(normal, range.velocity_for_next_step[0] - solid.motion.linear_velocity);
  if (next_step_rate < -1e-9) {
    return ExpectedHit::kHit;
  }
  return next_step_rate > 1e-9 ? ExpectedHit::kMiss : ExpectedHit::kTooCloseToTell;
}

ObliqueReactionRun ComputeObliqueReactionRun(const MoveParticleCase& test_case) {
  ParticleRange range = BuildOneParticleRange(test_case, 1);
  range.solids[0] = ToReactingSolid(range.solids[0]);
  ObliqueReactionRun run;
  run.expected = ComputeFlatSolidPathClass(range);
  run.expected_reaction = ComputeFlatSolidReactionClass(range, run.expected);
  const MovedRange moved = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  run.has_moved = moved.result.has_value();
  if (!run.has_moved) {
    return run;
  }
  const SolidReaction& reaction = moved.reaction[0];
  run.is_reacting = reaction.solid_index == 0;
  const ExpectedHit seen_hit = moved.record[0].kind != ContactKind::kNone ? ExpectedHit::kHit : ExpectedHit::kMiss;
  const ExpectedHit seen_reaction = run.is_reacting ? ExpectedHit::kHit : ExpectedHit::kMiss;
  run.is_classified_right = ComputeIsClassifiedRight(run.expected, seen_hit)
      && ComputeIsClassifiedRight(run.expected_reaction, seen_reaction);
  run.is_reaction_the_change =
      ComputeIsReactionTheChange(range.velocity_for_next_step[0], moved.velocity_for_next_step[0], reaction);
  return run;
}

// Fails the calling test unless this code gave, for an old case that ended in an error, the matching error: kBadInput
// for the old InvalidArgument, kStartsInsideSolid for the old FailedPrecondition.
void CheckSameErrorAsMain(const MainComparisonCase& old_case, const ZUpMove& move) {
  const ContactError expected_error = old_case.old_status == MainStatusCode::kInvalidArgument
      ? ContactError::kBadInput
      : ContactError::kStartsInsideSolid;
  EXPECT_EQ(move.error, expected_error);
}

// Fails the calling test unless this code's first hit matches the old hit of a case with one. The face's normal is the
// old normal turned to Z-up, within 2^-52 on every axis. The share of the step: this code stops one tolerance before
// the surface, so a hit along the path comes earlier by tol over the speed at which the gap closes, rate = Dot(normal,
// for_this_step - solid velocity), and its share is old - tol / (|rate| T) within 1e-12 (the rounding of a time near 1
// s is about 1e-16); a hit at the start is at share 0 in both. Example: line 211 closes at 10 m/s over 1 s, so the
// share is 0.4 - tol / 10 = 0.3999999999069.
void CheckSameHitAsMain(const MainComparisonCase& old_case, const ZUpMove& move) {
  const Vec3 normal = ToZUp(old_case.old_normal_out_of_solid);
  ExpectVec3Near(move.result.normal_out_of_solid, normal, 0x1p-52);
  if (old_case.old_hit_fraction == 0) {
    EXPECT_EQ(move.result.hit_fraction, 0);
    return;
  }
  const double rate = math::Dot(normal, ToZUp(old_case.velocity_for_this_step - old_case.solid_velocity));  // m/s
  const double expected_fraction = old_case.old_hit_fraction - (kTolerance / (std::abs(rate) * old_case.step_duration));
  EXPECT_NEAR(move.result.hit_fraction, expected_fraction, 1e-12);
}

// Fails the calling test unless this code moved the particle as the old code did: hit or no hit the same, and the hit
// as CheckSameHitAsMain checks it; the end within 1.25 tol of the old end on every axis (this code rests tol off the
// surface, the old code on it; measured: at most 1.0 tol on an axis); both velocities within 2^-40 of the largest speed
// of the case's inputs (measured: equal to the old tests' expected values, difference 0; the bound is far above the
// rounding of a response and far below any change a wrong rule makes).
void CheckSameMoveAsMain(const MainComparisonCase& old_case, const ZUpMove& move) {
  ASSERT_EQ(move.error, ContactError::kNone);
  EXPECT_EQ(move.result.kind != ContactKind::kNone, old_case.old_has_hit);
  if (old_case.old_has_hit) {
    CheckSameHitAsMain(old_case, move);
  }
  ExpectVec3Near(move.result.position, ToZUp(old_case.old_position), 1.25 * kTolerance);
  const double largest_speed = std::max({
      ComputeLargestAxis(old_case.velocity_for_next_step),
      ComputeLargestAxis(old_case.velocity_for_this_step),
      ComputeLargestAxis(old_case.solid_velocity),
  });  // m/s
  const double velocity_bound = 0x1p-40 * largest_speed;  // m/s
  ExpectVec3Near(move.result.velocity.for_next_step, ToZUp(old_case.old_velocity_for_next_step), velocity_bound);
  ExpectVec3Near(move.result.velocity.for_this_step, ToZUp(old_case.old_velocity_for_this_step), velocity_bound);
}

TEST(MoveParticle, SlidingOffFrictionlessSphere_SmallStepsLastHitNearAnalyticAngle) {
  // The same slide at 0.125 and 0.0625 ms (16,833 and 33,666 steps): the last hit settles about 0.004 rad before the
  // analytic angle (measured 0.83690 and 0.83722 rad, differences -0.0042 and -0.0038), within 0.005 rad of it and
  // before it. The removal of the speed into the sphere at each hit is what keeps it short of the angle.
  for (const double step_duration : {1.25e-4, 6.25e-5}) {
    SCOPED_TRACE(step_duration);
    const SlideOffSphere slide = ComputeSlideOffSphere(step_duration);
    std::cout << std::setprecision(10) << "sliding off at " << step_duration << " s: last hit at "
              << slide.last_hit_angle << " rad\n";
    ASSERT_FALSE(slide.has_failed);
    EXPECT_LT(slide.last_hit_angle, kAnalyticLeavingAngle);
    EXPECT_NEAR(slide.last_hit_angle, kAnalyticLeavingAngle, 0.005);
  }
}

// --- Many particles over many steps ---

// m^2/s^2 per unit mass: g z + |v|^2 / 2.
double ComputeEnergy(Vec3 position, Vec3 velocity) {
  return (kGravity * position.z) + (0.5 * math::Dot(velocity, velocity));
}

// What particles in a closed domain box did over many steps.
struct DomainBoxRun {
  int move_count = 0;  // Successful moves.
  int failure_count = 0;  // Moves that failed; the particle stopped there.
  int outside_count = 0;  // Ends closer than tol / 2 to a side of the box, by the box's own distance.
  double largest_energy_gain = -1;  // m^2/s^2 per unit mass: over every particle and step, the end minus the start.
  int resting_count = 0;  // Particles with friction that end the run at rest on the floor: velocity 0, z within tol.
};

// Moves 16 particles in the closed unit domain box (every side closed), each on its own, for 3000 steps of 1 ms; each
// step first subtracts 9.81 T from the z of both velocities. CaseGenerator, per particle: the start, 0.1 + 0.8 Next on
// each axis; the velocity, 2 NextSigned m/s on each axis; the friction alternates 0 and 0.5.
DomainBoxRun ComputeDomainBoxRun() {
  constexpr int kCount = 16;
  constexpr int kStepCount = 3000;
  constexpr double kStepDuration = 1e-3;  // s
  ClosedSides every_side_closed;
  every_side_closed.top = true;
  const Solid box =
      ToSolid(DomainBox{.low_corner = {}, .high_corner = {.x = 1, .y = 1, .z = 1}, .closed = every_side_closed}, 1);
  CaseGenerator generator;
  DomainBoxRun run;
  for (int particle = 0; particle < kCount; ++particle) {
    Vec3 position = {.x = 0, .y = 0, .z = 0};  // m
    position.x = 0.1 + (0.8 * generator.Next());
    position.y = 0.1 + (0.8 * generator.Next());
    position.z = 0.1 + (0.8 * generator.Next());
    Vec3 velocity = 2 * far_from_origin_cases::BuildSignedVector(generator);  // m/s
    const double friction = particle % 2 == 0 ? 0 : 0.5;
    const double start_energy = ComputeEnergy(position, velocity);
    for (int step = 0; step < kStepCount; ++step) {
      velocity.z -= kGravity * kStepDuration;
      SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
          .start_position = position,
          .velocity = velocity,
          .step_duration = kStepDuration,
          .friction = friction,
      });
      inputs.solids = &box;
      inputs.solid_count = 1;
      SolidContactResult result;
      if (MoveParticle(inputs, &result, nullptr) != ContactError::kNone) {
        ++run.failure_count;
        break;
      }
      ++run.move_count;
      position = result.position;
      velocity = result.velocity.for_next_step;
      const double distance = ComputeDistanceToSolid(box, box.pose_at_step_start, position).distance;  // m
      run.outside_count += distance < kTolerance / 2 ? 1 : 0;
      run.largest_energy_gain = std::max(run.largest_energy_gain, ComputeEnergy(position, velocity) - start_energy);
    }
    const bool is_at_rest = velocity.x == 0 && velocity.y == 0 && velocity.z == 0 && position.z <= 2 * kTolerance;
    run.resting_count += friction > 0 && is_at_rest ? 1 : 0;
  }
  return run;
}

TEST(MoveParticle, ParticlesInClosedDomainBox_NeverLeaveAndNeverGainEnergy) {
  // 16 particles in the closed unit box (ComputeDomainBoxRun), 3000 steps of 1 ms each with gravity added: 48,000
  // moves, none fails, no end is closer than tol / 2 to a side (read with the box's distance, not the record), and
  // g z + |v|^2 / 2 never rises above its start: a response only removes speed into a side and friction only slows the
  // sideways speed, and a corner's joint response is the smallest change, which never lengthens a velocity. The rest
  // gap moves a particle by at most tol, so a gain could be at most g tol = 9e-9 m^2/s^2; measured: every end lies
  // below its start's energy, the closest by 4.8e-5 m^2/s^2.
  // With friction 0.5 a particle on the floor slows by 0.5 g per second; started at most 2 m/s sideways, the 8
  // particles with friction are all at rest on the floor after 3 s, velocity exactly 0 (a stick returns the solid's
  // velocity).
  const DomainBoxRun run = ComputeDomainBoxRun();
  std::cout << "domain box: moves " << run.move_count << ", largest energy gain " << run.largest_energy_gain
            << " m^2/s^2, at rest " << run.resting_count << "\n";
  EXPECT_EQ(run.move_count, 48000);
  EXPECT_EQ(run.failure_count, 0);
  EXPECT_EQ(run.outside_count, 0);
  EXPECT_LE(run.largest_energy_gain, 0);
  EXPECT_EQ(run.resting_count, 8);
}

constexpr int kRestingOnFloorCount = 1000;

constexpr int kRestingOnFloorStepCount = 20;

// What many particles resting on a reacting floor gave over several steps of MoveParticles.
struct RestingOnFloorRun {
  int successful_call_count = 0;
  int exact_change_count =
      0;  // Particle steps whose reaction is exactly {0, 0, g T} and whose velocity ends exactly 0.
  std::vector<double> change_sums;  // m/s: per step, the sum over the particles of the reaction's z, in index order.
};

// Rests kRestingOnFloorCount particles at the rest gap of the reacting floor z <= 0 (at z = tol, spread over x and y),
// at rest, for kRestingOnFloorStepCount steps of 1 ms through MoveParticles: each step adds {0, 0, -9.81 T} to both
// velocities, and the ends and velocity.for_next_step feed the next step.
RestingOnFloorRun ComputeRestingOnFloorRun() {
  constexpr double kStepDuration = 1e-3;  // s
  const Vec3 expected_change = {.x = 0, .y = 0, .z = kGravity * kStepDuration};  // m/s
  ParticleRange range;
  range.solids = {ToReactingSolid(move_particle_cases::BuildFloor(1))};
  for (int index = 0; index < kRestingOnFloorCount; ++index) {
    const SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
        .start_position = {.x = 0.001 * index, .y = 0.5 - (0.0007 * index), .z = kTolerance},
        .velocity = {},
        .step_duration = kStepDuration,
        .friction = 0.3,
    });
    range.AddParticle(inputs, static_cast<std::uint64_t>(index) + 1);
  }
  RestingOnFloorRun run;
  for (int step = 0; step < kRestingOnFloorStepCount; ++step) {
    for (std::size_t index = 0; index < range.size(); ++index) {
      range.velocity_for_next_step[index].z -= kGravity * kStepDuration;
      range.velocity_for_this_step[index] = range.velocity_for_next_step[index];
    }
    const MovedRange moved = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
    if (!moved.result.has_value()) {
      return run;
    }
    ++run.successful_call_count;
    double change_sum = 0;  // m/s
    for (std::size_t index = 0; index < range.size(); ++index) {
      const Vec3 change = moved.reaction[index].velocity_for_next_step_change;  // m/s
      const Vec3 velocity = moved.velocity_for_next_step[index];  // m/s
      const bool is_exact = change.x == expected_change.x && change.y == expected_change.y
          && change.z == expected_change.z && velocity.x == 0 && velocity.y == 0 && velocity.z == 0;
      run.exact_change_count += is_exact ? 1 : 0;
      change_sum += change.z;
      range.start_position[index] = moved.position[index];
      range.velocity_for_next_step[index] = velocity;
    }
    run.change_sums.push_back(change_sum);
  }
  return run;
}

TEST(MoveParticles, ParticlesRestingOnReactingFloor_ReactionsSumToWeightImpulse) {
  // 1000 particles at rest on the reacting floor (ComputeRestingOnFloorRun), 20 steps. A reaction is the change of
  // velocity.for_next_step the floor made, in m/s; times the particle's mass it is the impulse on the particle, and the
  // floor takes its opposite. Each step gravity adds -g T = -9.81e-3 m/s, the floor's normal {0, 0, 1} is exact and the
  // response removes it whole: each reaction is exactly {0, 0, g T} and each velocity ends exactly 0, so per unit mass
  // the reactions sum to the weight impulse N g T = 9.81 m/s per step, within the rounding of 1000 sums, 2^-40 of it.
  // A second run gives the same sums bit for bit.
  const RestingOnFloorRun run = ComputeRestingOnFloorRun();
  const RestingOnFloorRun repeat = ComputeRestingOnFloorRun();
  EXPECT_EQ(run.successful_call_count, kRestingOnFloorStepCount);
  EXPECT_EQ(run.exact_change_count, kRestingOnFloorCount * kRestingOnFloorStepCount);
  const double weight_impulse = kRestingOnFloorCount * kGravity * 1e-3;  // m/s per unit mass, per step.
  const auto is_weight_impulse = [weight_impulse](double change_sum) {
    return std::abs(change_sum - weight_impulse) <= 0x1p-40 * weight_impulse;
  };
  EXPECT_EQ(std::ranges::count_if(run.change_sums, is_weight_impulse), kRestingOnFloorStepCount);
  EXPECT_EQ(ToBits(repeat.change_sums), ToBits(run.change_sums));
}

// The swarm's scene: the closed domain box from {0, 0, 0} to {2, 2, 2} m and, inside it, a solid of every shape, fixed,
// moving, spinning or turning, the turning one far from the others (more than 0.2 m between their reaches):
//   id 1  the domain box, every side closed
//   id 2  a fixed sphere of radius 0.25 m at {0.5, 0.5, 0.5}
//   id 3  a cylinder of radius 0.2 m and half height 0.25 m along z at {1.5, 0.5, 0.5}, spinning at 3 rad/s about it
//   id 4  a box of half size {0.2, 0.15, 0.1} m at {0.5, 1.5, 0.5} moving at {0.05, 0, 0} m/s
//   id 5  a sphere of radius 0.2 m at {1.5, 1.5, 0.5} rising at {0, 0, 0.02} m/s and spinning at {1, 2, 0} rad/s
//   id 6  a box of half size 0.1 m at {1, 1, 0.6} turning at {0, 0.3, 0.4} rad/s
//   id 7  a fixed plate of half thickness 0.02 m at z = 1.6, unbounded across the box
//   id 8  a fixed flat solid filling the corner x + y + z >= 5.4
std::vector<Solid> BuildSwarmSolids() {
  ClosedSides every_side_closed;
  every_side_closed.top = true;
  const Vec3 corner_normal = math::Normalized({.x = -1, .y = -1, .z = -1});
  return {
      ToSolid(DomainBox{.low_corner = {}, .high_corner = {.x = 2, .y = 2, .z = 2}, .closed = every_side_closed}, 1),
      ToSolid(Sphere{.center = {.x = 0.5, .y = 0.5, .z = 0.5}, .radius = 0.25}, kNoMotion, 2),
      ToSolid(
          Cylinder{
              .center = {.x = 1.5, .y = 0.5, .z = 0.5},
              .axis = {.x = 0, .y = 0, .z = 1},
              .radius = 0.2,
              .half_height = 0.25,
          },
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 3}},
          3
      ),
      ToSolid(
          Box{
              .center = {.x = 0.5, .y = 1.5, .z = 0.5},
              .axes = kIdentity,
              .half_size = {.x = 0.2, .y = 0.15, .z = 0.1},
          },
          {.linear_velocity = {.x = 0.05, .y = 0, .z = 0}, .angular_velocity = {}},
          4
      ),
      ToSolid(
          Sphere{.center = {.x = 1.5, .y = 1.5, .z = 0.5}, .radius = 0.2},
          {.linear_velocity = {.x = 0, .y = 0, .z = 0.02}, .angular_velocity = {.x = 1, .y = 2, .z = 0}},
          5
      ),
      ToSolid(
          Box{
              .center = {.x = 1, .y = 1, .z = 0.6},
              .axes = kIdentity,
              .half_size = {.x = 0.1, .y = 0.1, .z = 0.1},
          },
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0.3, .z = 0.4}},
          6
      ),
      ToSolid(
          Plate{.center = {.x = 1, .y = 1, .z = 1.6}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.02},
          kNoMotion,
          7
      ),
      ToSolid(
          FlatSolid{.point_on_surface = {.x = 1.8, .y = 1.8, .z = 1.8}, .normal_out_of_solid = corner_normal},
          kNoMotion,
          8
      ),
  };
}

// m. The least distance from a point to the solids, each at its pose in `poses`, by the solids' own distance function.
double ComputeLeastDistance(std::span<const Solid> solids, std::span<const SolidPose> poses, Vec3 point) {
  double least = std::numeric_limits<double>::infinity();  // m
  for (std::size_t index = 0; index < solids.size(); ++index) {
    least = std::min(least, ComputeDistanceToSolid(solids[index], poses[index], point).distance);
  }
  return least;
}

// The pose of each solid `elapsed` seconds into the step.
std::vector<SolidPose> ComputePoses(std::span<const Solid> solids, double elapsed) {
  std::vector<SolidPose> poses;
  for (const Solid& solid : solids) {
    poses.push_back(ComputeSolidPose(solid, elapsed));
  }
  return poses;
}

// What the swarm did.
struct SwarmRun {
  int start_count = 0;  // Particles placed.
  std::int64_t hit_count = 0;  // Particle steps with kind kHit or kStartedInsideSolid.
  int successful_call_count = 0;  // MoveParticles calls that succeeded.
  std::int64_t move_count = 0;  // Particle steps that ended.
  int ends_closer_than_half_tolerance = 0;  // Ends closer than tol / 2 to a solid at the end pose.
  int ends_inside = 0;  // Ends at a negative distance.
  std::map<std::string, int> failures;  // Failing particles by "error/close call", each removed when it failed.
  int largest_hit_count = 0;  // Over every particle step.
};

constexpr int kSwarmCount = 1000;

constexpr int kSwarmStepCount = 120;

constexpr double kSwarmStepDuration = 2e-3;  // s

// Builds the swarm's particles among `solids`, ids 1 + index. CaseGenerator, per particle: the start, 0.05 + 1.9 Next
// m on each axis, drawn again while within 0.02 m of a solid; the velocity, NextSigned m/s on each axis; the friction,
// Next.
ParticleRange BuildSwarmRange(const std::vector<Solid>& solids) {
  CaseGenerator generator;
  ParticleRange range;
  range.solids = solids;
  const std::vector<SolidPose> start_poses = ComputePoses(solids, 0);
  while (std::cmp_less(range.size(), kSwarmCount)) {
    const Vec3 start = {
        .x = 0.05 + (1.9 * generator.Next()),
        .y = 0.05 + (1.9 * generator.Next()),
        .z = 0.05 + (1.9 * generator.Next()),
    };
    const Vec3 velocity = far_from_origin_cases::BuildSignedVector(generator);  // m/s
    const double friction = generator.Next();
    if (ComputeLeastDistance(solids, start_poses, start) < 0.02) {
      continue;
    }
    const SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
        .start_position = start,
        .velocity = velocity,
        .step_duration = kSwarmStepDuration,
        .friction = friction,
    });
    range.AddParticle(inputs, range.size() + 1);
  }
  return range;
}

// Removes the particle at `index` from every array of the range and numbers the rest again, so an id stays 1 + index.
void ApplyParticleRemoval(ParticleRange& range, std::size_t index) {
  const auto erase_at = [index](auto& values) {
    values.erase(values.begin() + static_cast<std::ptrdiff_t>(index));
  };
  erase_at(range.particle_id);
  erase_at(range.start_position);
  erase_at(range.velocity_for_next_step);
  erase_at(range.velocity_for_this_step);
  erase_at(range.friction);
  erase_at(range.plate_side);
  for (std::size_t renumbered = 0; renumbered < range.size(); ++renumbered) {
    range.particle_id[renumbered] = renumbered + 1;
  }
}

// Runs one step's MoveParticles call over the range; a failing particle (MoveParticles reports the lowest id) is
// counted by its error and close call in run.failures and removed, and the call made again without it.
MovedRange ComputeSwarmStep(ParticleRange& range, SwarmRun& run) {
  MovedRange moved = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  while (!moved.result.has_value() && moved.result.error().has_particle_id) {
    const ContactFailure failure = moved.result.error();
    ++run.failures[std::string(ToName(failure.error)) + "/" + std::string(ToName(failure.close_call))];
    ApplyParticleRemoval(range, failure.particle_id - 1);
    moved = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  }
  return moved;
}

// Moves the swarm (BuildSwarmRange) among the swarm's solids for kSwarmStepCount steps of kSwarmStepDuration through
// MoveParticles (ComputeSwarmStep). Each step first subtracts 9.81 T from the z of both velocities; after it each
// solid's pose moves to its pose at the end of the step.
SwarmRun ComputeSwarmRun() {
  std::vector<Solid> solids = BuildSwarmSolids();
  ParticleRange range = BuildSwarmRange(solids);
  SwarmRun run;
  run.start_count = static_cast<int>(range.size());
  for (int step = 0; step < kSwarmStepCount; ++step) {
    for (std::size_t index = 0; index < range.size(); ++index) {
      range.velocity_for_next_step[index].z -= kGravity * kSwarmStepDuration;
      range.velocity_for_this_step[index] = range.velocity_for_next_step[index];
    }
    const MovedRange moved = ComputeSwarmStep(range, run);
    if (!moved.result.has_value()) {
      return run;
    }
    ++run.successful_call_count;
    const std::vector<SolidPose> end_poses = ComputePoses(solids, kSwarmStepDuration);
    for (std::size_t index = 0; index < range.size(); ++index) {
      const double distance = ComputeLeastDistance(solids, end_poses, moved.position[index]);  // m
      run.hit_count += moved.record[index].kind != ContactKind::kNone ? 1 : 0;
      run.ends_closer_than_half_tolerance += distance < kTolerance / 2 ? 1 : 0;
      run.ends_inside += distance < 0 ? 1 : 0;
      run.largest_hit_count = std::max<int>(run.largest_hit_count, moved.record[index].hit_count);
      range.start_position[index] = moved.position[index];
      range.velocity_for_next_step[index] = moved.velocity_for_next_step[index];
      ++run.move_count;
    }
    for (std::size_t index = 0; index < solids.size(); ++index) {
      solids[index].pose_at_step_start = end_poses[index];
    }
    range.solids = solids;
  }
  return run;
}

TEST(MoveParticles, SwarmAmongSolidsOfEveryShape_NeverEndsWithinHalfTolerance) {
  // 1000 particles among the swarm's solids (BuildSwarmSolids, ComputeSwarmRun) for 120 steps of 2 ms: 120,000 moves,
  // of which more than 10,000 meet a solid (measured 10,777, at most 2 hits in one move: the particles fall up to 0.3 m
  // onto the floor, the solids and the plate). Every end is read with ComputeDistanceToSolid at the end pose, apart
  // from the move's own record: none closer than tol / 2, none inside. The failures are counted by kind: none. Every
  // step's call succeeds. The scene is kept to 1000 particles so the test runs in well under a second in a debug build;
  // 2000 particles for 200 steps of 1 ms also gave no failure and no end within tol / 2 (2.2 s).
  const SwarmRun run = ComputeSwarmRun();
  std::cout << "swarm: " << run.start_count << " particles, " << run.move_count << " moves, " << run.hit_count
            << " with a hit, " << run.failures.size() << " kinds of failure, largest hit count "
            << run.largest_hit_count << "\n";
  EXPECT_EQ(run.start_count, kSwarmCount);
  EXPECT_EQ(run.successful_call_count, kSwarmStepCount);
  EXPECT_EQ(run.move_count, 120000);
  EXPECT_GE(run.hit_count, 10000);
  EXPECT_EQ(run.failures, (std::map<std::string, int>{}));
  EXPECT_EQ(run.ends_closer_than_half_tolerance, 0);
  EXPECT_EQ(run.ends_inside, 0);
  EXPECT_LE(run.largest_hit_count, kMaxHitsPerStep);
}

TEST(MoveParticles, ReactionSum_EqualsSumOfVelocityChanges) {
  // The first 100 oblique cases (BuildObliqueCases), each against its own solid made reacting, one MoveParticles call
  // per case: a moving flat solid far from the origin, paths that meet it at a share of the step, frictions from 0 to
  // 1, two different velocities. Per particle, the reaction is velocity.for_next_step after minus before, within 2^-44
  // of the larger of the two; a particle without a reaction keeps for_next_step bit for bit. So the reactions sum to
  // the change of the particles' velocities, which the solids module turns into a force. Each path is classified
  // without MoveParticle, its hit (ComputeFlatSolidPathClass) and whether it is charged
  // (ComputeFlatSolidReactionClass), and the hits and reactions seen must be those. The set is built so every path hits
  // (hit_share <= 0.9): 100 clear hits; for_next_step is for_this_step plus 0.5 times three NextSigned, which leaves
  // the solid in 10 of them: 90 must react.
  std::vector<ObliqueReactionRun> runs;
  for (const MoveParticleCase& test_case : BuildObliqueCases(100)) {
    runs.push_back(ComputeObliqueReactionRun(test_case));
  }
  std::vector<int> failing_cases;  // Indexes of the cases whose call failed or broke one of the rules.
  for (std::size_t case_index = 0; case_index < runs.size(); ++case_index) {
    const ObliqueReactionRun& run = runs[case_index];
    if (!(run.has_moved && run.is_classified_right && run.is_reaction_the_change)) {
      failing_cases.push_back(static_cast<int>(case_index));
    }
  }
  const auto clear_hit_count = std::ranges::count_if(runs, [](const ObliqueReactionRun& run) {
    return run.expected == ExpectedHit::kHit;
  });
  const auto must_react_count = std::ranges::count_if(runs, [](const ObliqueReactionRun& run) {
    return run.expected_reaction == ExpectedHit::kHit;
  });
  const auto reacting_count = std::ranges::count_if(runs, [](const ObliqueReactionRun& run) {
    return run.is_reacting;
  });
  std::cout << "oblique cases: clear hits " << clear_hit_count << ", must react " << must_react_count << ", reacting "
            << reacting_count << "\n";
  EXPECT_EQ(failing_cases, std::vector<int>{});
  EXPECT_EQ(clear_hit_count, 100);
  EXPECT_EQ(must_react_count, 90);
  EXPECT_EQ(reacting_count, 90);
}

TEST(MoveParticle, RestingOnTopOfSphere_Stays) {
  // A fixed sphere of radius 1 m at the origin and a particle at rest on its top, {0, 0, 1 + tol}; 1000 steps of
  // 1 ms, each adding {0, 0, -9.81 T} to both velocities. The normal at the top is {0, 0, 1} exactly, so each step's
  // response removes the whole velocity: x and y stay exactly 0, both velocities end every step exactly 0, and the
  // particle stays at the rest gap.
  const std::vector<Solid> solids = {ToSolid(Sphere{.center = {}, .radius = 1}, kNoMotion, 1)};
  FallingParticle particle = {.position = {.x = 0, .y = 0, .z = 1 + kTolerance}, .velocity = {}};
  for (int step = 0; step < 1000; ++step) {
    SCOPED_TRACE(step);
    particle.velocity.z -= kGravity * 1e-3;
    SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
        .start_position = particle.position,
        .velocity = particle.velocity,
        .step_duration = 1e-3,
        .friction = 0,
    });
    inputs.solids = solids.data();
    inputs.solid_count = 1;
    SolidContactResult result;
    ASSERT_EQ(MoveParticle(inputs, &result, nullptr), ContactError::kNone);
    EXPECT_EQ(result.position.x, 0);
    EXPECT_EQ(result.position.y, 0);
    EXPECT_NEAR(result.position.z, 1 + kTolerance, 2 * kTolerance);
    CheckEqualAxes(result.velocity.for_next_step, {});
    CheckEqualAxes(result.velocity.for_this_step, {});
    particle = {.position = result.position, .velocity = result.velocity.for_next_step};
  }
}

TEST(MoveParticle, SlidingOffFrictionlessSphere_NeverGainsEnergy) {
  // The same slide. Removing the speed into the solid takes energy out, and nothing puts any in, so g z + |v|^2 / 2
  // never rises above its start. Measured: 0.0141 m^2/s^2 lost of 9.81 at 1 ms, about half as much at 0.5 ms (the loss
  // shrinks with the step), and no step's end above the start.
  const SlideOffSphere slide = ComputeSlideOffSphere(1e-3);
  std::cout << std::setprecision(8) << "sliding off: energy " << slide.start_energy << " at the start, "
            << slide.end_energy << " at the end, largest gain " << slide.largest_energy_gain << " m^2/s^2\n";
  ASSERT_FALSE(slide.has_failed);
  EXPECT_LE(slide.largest_energy_gain, 0);
  EXPECT_LT(slide.end_energy, slide.start_energy);
}

TEST(MoveParticle, MainFlatSolidAndPlateCases_AgreeAfterTurningToZUp) {
  // Every old case marked kAgrees, turned to Z-up: a result as CheckSameMoveAsMain checks it, an error as
  // CheckSameErrorAsMain does. The two cases marked kDiffers have tests of their own below. The counts come from the
  // list alone.
  const std::vector<MainComparisonCase> cases = BuildMainComparisonCases();
  const auto is_differing = [](const MainComparisonCase& old_case) {
    return old_case.agreement == MainAgreement::kDiffers;
  };
  const auto has_reason = [](const MainComparisonCase& old_case) {
    return old_case.why_different != nullptr;
  };
  EXPECT_EQ(std::ranges::count_if(cases, is_differing), 2);
  EXPECT_EQ(std::ranges::count_if(cases, has_reason), 2);
  int moved_count = 0;
  int failed_count = 0;
  for (const MainComparisonCase& old_case : cases) {
    SCOPED_TRACE("main line " + std::to_string(old_case.source_line));
    if (is_differing(old_case)) {
      continue;
    }
    const ZUpMove move = ComputeZUpMove(old_case);
    if (old_case.old_status == MainStatusCode::kOk) {
      ++moved_count;
      CheckSameMoveAsMain(old_case, move);
    } else {
      ++failed_count;
      CheckSameErrorAsMain(old_case, move);
    }
  }
  EXPECT_EQ(moved_count, 11);
  EXPECT_EQ(failed_count, 11);
}

}  // namespace
}  // namespace fluidity::contact
