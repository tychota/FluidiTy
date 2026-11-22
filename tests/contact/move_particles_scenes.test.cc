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
#include "fluidity/contact/testing/move_particles_runs.h"
#include "fluidity/contact/testing/move_particles_scene_cases.h"

namespace fluidity::contact {
namespace {

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

}  // namespace
}  // namespace fluidity::contact
