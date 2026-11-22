#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/solid_contact_steps_find_first_hit_cases.h"

namespace fluidity::contact {
namespace {

TEST(ComputeIsTurningSolid, GeneratedSpinsAboutNormal_AreNeverTurning) {
  // 20000 flat solids and plates, alternately, with random normals (three NextSigned(), drawn again while their
  // length is below 0.1, then Normalized) spinning about that normal at 10^(-3 + 6 Next()) rad/s: none may turn,
  // although the part across the normal computes to more than zero on most of them (counted, so the set bites).
  CaseGenerator generator;
  int turning_count = 0;
  int rounded_across_count = 0;
  constexpr int kSolidCount = 20000;
  for (int solid_index = 0; solid_index < kSolidCount; ++solid_index) {
    Vec3 direction{};
    do {
      direction = Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    } while (math::Length(direction) < 0.1);
    const Vec3 normal = math::Normalized(direction);
    const double spin_rate = std::pow(10.0, -3 + 6 * generator.Next());  // rad/s
    const SolidMotion spin = {.linear_velocity = {}, .angular_velocity = spin_rate * normal};
    const Solid solid = solid_index % 2 == 0
        ? ToSolid(FlatSolid{.point_on_surface = {}, .normal_out_of_solid = normal}, spin, 1)
        : ToSolid(Plate{.center = {}, .normal = normal, .half_thickness = 0.01}, spin, 1);
    const std::array solids = {solid};
    turning_count += ComputeIsTurningSolid(BuildInputs(solids, 0.1), 0) ? 1 : 0;
    const Vec3 own_z_axis = ToColumn(solid.pose_at_step_start.axes, 2);
    const Vec3 rounded_across = spin.angular_velocity - math::Dot(spin.angular_velocity, own_z_axis) * own_z_axis;
    rounded_across_count += math::Length(rounded_across) > 0 ? 1 : 0;
  }
  EXPECT_EQ(turning_count, 0);
  EXPECT_GT(rounded_across_count, kSolidCount / 3);
  testing::Test::RecordProperty("rounded_across_count", rounded_across_count);
}

// --- Generated sweeps ---

// Fails the calling test unless the path stays farther than tol from the solid, checked at 64 times, until the hit
// (or over the whole duration when free), and a hit sits at distance tol up to rounding. Assumes start_elapsed 0.
void CheckFirstHitWhereDistanceReachesTolerance(const Solid& solid, Path path, FirstHit first_hit, int case_index) {
  constexpr double kRoundingAllowance = 1e-14;  // m. Far above the rounding of a distance at unit scale.
  const bool is_hit = first_hit.solid_index != kNoSolid;
  const double end_elapsed = is_hit ? first_hit.elapsed : path.duration;
  for (int time_index = 0; time_index < 64; ++time_index) {
    const double elapsed = end_elapsed * time_index / 64;
    const Vec3 position = path.start + path.velocity * elapsed;
    const double distance = ComputeDistanceToSolid(solid, ComputeSolidPose(solid, elapsed), position).distance;
    EXPECT_GT(distance, kTolerance - kRoundingAllowance) << "case " << case_index << " time " << time_index;
  }
  if (!is_hit) {
    return;
  }
  const Vec3 hit_position = path.start + path.velocity * first_hit.elapsed;
  const DistanceToSolid at_hit =
      ComputeDistanceToSolid(solid, ComputeSolidPose(solid, first_hit.elapsed), hit_position);
  EXPECT_NEAR(at_hit.distance, kTolerance, kRoundingAllowance) << "case " << case_index;
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, kRoundingAllowance) << "case " << case_index;
}

TEST(FindFirstHit, GeneratedBoxSweeps_HitWhereDistanceReachesTolerance) {
  // A box of half sizes {0.5, 0.25, 1} turned by kTiltedAxes, moving at {0.25, -0.5, 0.125} m/s. Draw order per case:
  // a start of 3 times three NextSigned() values, drawn again while it is within 2 tol of the box (so the start is
  // outside the grown box and the closed form decides); a target of 0.75 times three NextSigned() values; a speed
  // factor 1 + Next(). Measured from the box the path heads for the target and reaches it within 1 / factor <= 1 s,
  // so it hits when the target is in the grown box and often misses otherwise.
  // The box's distance is its largest face gap, so the grown box is where that distance is at most tol. A hit must
  // then sit at distance tol up to rounding, and the path before it, checked at 64 times, must stay farther; a free
  // path must stay farther over the whole duration.
  const SolidMotion motion = {.linear_velocity = {.x = 0.25, .y = -0.5, .z = 0.125}, .angular_velocity = {}};
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kTiltedAxes, .half_size = {.x = 0.5, .y = 0.25, .z = 1}}, motion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  CaseGenerator generator;
  int hit_count = 0;
  for (int case_index = 0; case_index < 1000; ++case_index) {
    Vec3 start{};
    do {
      start = 3 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    } while (ComputeDistanceToSolid(solids[0], solids[0].pose_at_step_start, start).distance <= 2 * kTolerance);
    const Vec3 target =
        0.75 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const double speed_factor = 1 + generator.Next();  // 1/s
    const Vec3 velocity = motion.linear_velocity + speed_factor * (target - start);
    const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = velocity, .duration = 1});
    CheckFirstHitWhereDistanceReachesTolerance(
        solids[0],
        {.start = start, .velocity = velocity, .duration = 1},
        first_hit,
        case_index
    );
    if (first_hit.solid_index != kNoSolid) {
      ++hit_count;
    }
  }
  // Enough cases hit for the check to mean something, and enough miss.
  EXPECT_GT(hit_count, 100);
  EXPECT_LT(hit_count, 900);
}

// The distance from the path's position elapsed seconds into the sweep to the solid at the pose of that moment.
double ComputeDistanceAlongPath(const Solid& solid, Path path, double elapsed) {
  const Vec3 position = path.start + path.velocity * elapsed;
  return ComputeDistanceToSolid(solid, ComputeSolidPose(solid, path.start_elapsed + elapsed), position).distance;
}

// The smallest distance from the path to a round solid between 0 and end_elapsed seconds into the sweep, by a search
// that shrinks the interval by a third each time. The solid only translates (a spin leaves its shape), so measured
// from it the path is a straight line, and the true distance to a convex solid is a convex function along a line: one
// minimum, which the search finds without assuming anything of FindFirstHit.
double ComputeSmallestDistanceAlongPath(const Solid& solid, Path path, double end_elapsed) {
  double low = 0;
  double high = end_elapsed;
  // 100 steps shrink the interval to (2/3)^100 of its length: below 3e-17 s for the longest sweep, 12 s.
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double lower_third = low + (high - low) / 3;
    const double upper_third = high - (high - low) / 3;
    if (ComputeDistanceAlongPath(solid, path, lower_third) < ComputeDistanceAlongPath(solid, path, upper_third)) {
      high = upper_third;
    } else {
      low = lower_third;
    }
  }
  return std::min({
      ComputeDistanceAlongPath(solid, path, 0),
      ComputeDistanceAlongPath(solid, path, low),
      ComputeDistanceAlongPath(solid, path, end_elapsed),
  });
}

// Fails the calling test unless the first hit agrees with the slow reference: before a hit, and over the whole
// duration of a free path, the true distance never comes below tol; at a hit it is between tol and
// largest_distance_at_hit (tol for a sphere, tol + tol / 1024 for a cylinder, whose rim search stops within
// kRimHitGapShare of tol). A hit is never behind the start nor after the duration.
void CheckFirstHitAgreesWithReference(
    const Solid& solid,
    double largest_distance_at_hit,
    Path path,
    FirstHit first_hit,
    int case_index
) {
  const bool is_hit = first_hit.solid_index != kNoSolid;
  const double end_elapsed = is_hit ? first_hit.elapsed : path.duration;
  EXPECT_GE(end_elapsed, 0) << "case " << case_index;
  EXPECT_LE(end_elapsed, path.duration) << "case " << case_index;
  EXPECT_GE(ComputeSmallestDistanceAlongPath(solid, path, end_elapsed), kTolerance - kReferenceAllowance)
      << "case " << case_index;
  if (!is_hit) {
    return;
  }
  const double distance_at_hit = ComputeDistanceAlongPath(solid, path, first_hit.elapsed);
  EXPECT_LE(distance_at_hit, largest_distance_at_hit + kReferenceAllowance) << "case " << case_index;
  EXPECT_NEAR(first_hit.at_hit.distance, distance_at_hit, kReferenceAllowance) << "case " << case_index;
}

// Runs 1000 generated sweeps against one moving round solid and checks each against the slow reference. Draw order
// per case: start_elapsed 0.5 * Next(); a start of 2.5 times three NextSigned() values from the solid's origin at
// that moment, drawn again while it is within 2 tol of the solid (so the closed form decides); a target of 0.9 times
// three NextSigned() values from that origin; a speed factor 1 + Next(). Measured from the solid the path heads for
// the target and reaches it within 1 / factor <= 1 s, so it hits when the target is in the grown solid and often
// misses otherwise. Returns the number of hits.
int CheckGeneratedRoundSweeps(const Solid& solid, double largest_distance_at_hit) {
  const std::array solids = {solid};
  const SolidContactInputs inputs = BuildInputs(solids, 1.5);
  CaseGenerator generator;
  int hit_count = 0;
  for (int case_index = 0; case_index < 1000; ++case_index) {
    const double start_elapsed = 0.5 * generator.Next();
    const SolidPose pose = ComputeSolidPose(solid, start_elapsed);
    Vec3 start{};
    do {
      start = pose.origin
          + 2.5 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    } while (ComputeDistanceToSolid(solid, pose, start).distance <= 2 * kTolerance);
    const Vec3 target =
        pose.origin + 0.9 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const double speed_factor = 1 + generator.Next();  // 1/s
    const Vec3 velocity = solid.motion.linear_velocity + speed_factor * (target - start);
    const Path path = {.start = start, .velocity = velocity, .start_elapsed = start_elapsed, .duration = 1};
    const FirstHit first_hit = FindHit(inputs, path);
    CheckFirstHitAgreesWithReference(solid, largest_distance_at_hit, path, first_hit, case_index);
    if (first_hit.solid_index != kNoSolid) {
      ++hit_count;
    }
  }
  return hit_count;
}

TEST(FindFirstHit, GeneratedSphereSweeps_AgreeWithSlowReference) {
  // A sphere of radius 0.5 m moving at {0.25, -0.5, 0.125} m/s and spinning, which changes nothing.
  const SolidMotion motion = {
      .linear_velocity = {.x = 0.25, .y = -0.5, .z = 0.125},
      .angular_velocity = {.x = 1, .y = -2, .z = 0.5},
  };
  const int hit_count = CheckGeneratedRoundSweeps(BuildSphere({.x = 0.25, .y = 0.5, .z = -0.125}, motion), kTolerance);
  // Enough cases hit for the check to mean something, and enough miss.
  EXPECT_GT(hit_count, 100);
  EXPECT_LT(hit_count, 900);
}

TEST(FindFirstHit, GeneratedCylinderSweeps_AgreeWithSlowReference) {
  // A cylinder of radius 0.5 m and half height 0.75 m along the tilted axis, moving at {0.25, -0.5, 0.125} m/s and
  // spinning about its own axis, which leaves the shape. A hit beside a rim is where the rim search stops, up to
  // tol / 1024 beyond tol (method paper <sec-round-first-hit>).
  const Vec3 axis = ToColumn(kTiltedAxes, 2);
  const std::array solids = {
      ToSolid(
          Cylinder{.center = {.x = 0.25, .y = 0.5, .z = -0.125}, .axis = axis, .radius = 0.5, .half_height = 0.75},
          {.linear_velocity = {.x = 0.25, .y = -0.5, .z = 0.125}, .angular_velocity = 3 * axis},
          1
      ),
  };
  ASSERT_EQ(CheckSolid(solids[0], 1.5), ContactError::kNone);
  const int hit_count = CheckGeneratedRoundSweeps(solids[0], kTolerance + kTolerance / 1024);
  std::cout << "generated cylinder sweeps 1000, hit " << hit_count << "\n";
  EXPECT_GT(hit_count, 100);
  EXPECT_LT(hit_count, 900);
}

// What the paths through the grown rim edge of one cylinder gave: the mistakes against the slow reference, the paths
// the reference says must be hit, and the hits per number of advances of the rim search (the last entry counts the hits
// that entered on the wall or a cap proper and made no search).
struct RimEdgeCounts {
  int mistakes = 0;
  int must_hit = 0;
  std::array<int, kMaxRimAdvanceCount + 2> hits_per_advance_count{};
};

// Counts the paths through the grown rim edge of one cylinder that FindFirstHit gets wrong, against the slow
// reference. Draw order per path: azimuth 2 pi Next(); angle in the plane of the axis 0.05 + (pi / 2 - 0.1) Next();
// distance before the edge 0.5 + 1.5 Next(); the cap, high when Next() < 0.5; an offset of the line from the edge of
// -8 to 8 units of 2^-53, as floor(17 Next()) - 8; speed 0.5 + 1.5 Next(). In the cylinder's own axes the path passes
// within that offset of the grown rim circle {R cos(azimuth), R sin(azimuth), +-H}, going toward the axis and the
// middle plane, for three times the time it takes to reach the edge. A hit is right when the true distance there is
// between tol and tol + tol / 1024 up to kReferenceAllowance, and the path before it stays at tol or more. A path the
// reference says must be hit has a true distance below tol - kReferenceAllowance.
RimEdgeCounts ComputeRimEdgeMistakeCounts(const Solid& solid, int path_count) {
  const std::array solids = {solid};
  // The longest sweep is 3 * 2 / 0.5 = 12 s.
  const SolidContactInputs inputs = BuildInputs(solids, 12);
  const Mat3 axes = solid.pose_at_step_start.axes;
  const double grown_radius = solid.half_size.x + kTolerance;
  const double grown_half_height = solid.half_size.z + kTolerance;
  CaseGenerator generator;
  RimEdgeCounts counts;
  for (int path_index = 0; path_index < path_count; ++path_index) {
    const double azimuth = 2 * std::numbers::pi * generator.Next();
    const double angle = 0.05 + (std::numbers::pi / 2 - 0.1) * generator.Next();
    const double distance_before_edge = 0.5 + 1.5 * generator.Next();  // m
    const double cap_side_sign = generator.Next() < 0.5 ? 1.0 : -1.0;
    const double offset = (std::floor(17 * generator.Next()) - 8) * 0x1p-53;  // m
    const double speed = 0.5 + 1.5 * generator.Next();  // m/s
    const Vec3 radial = {.x = std::cos(azimuth), .y = std::sin(azimuth), .z = 0};
    const double radial_part = std::cos(angle);
    const double axial_part = std::sin(angle);
    const Vec3 own_start = (grown_radius + distance_before_edge * radial_part + offset) * radial
        + Vec3{.x = 0, .y = 0, .z = cap_side_sign * (grown_half_height + distance_before_edge * axial_part)};
    const Vec3 own_velocity = -speed * (radial_part * radial + Vec3{.x = 0, .y = 0, .z = cap_side_sign * axial_part});
    const Path path = {
        .start = solid.pose_at_step_start.origin + axes * own_start,
        .velocity = axes * own_velocity,
        .duration = 3 * distance_before_edge / speed,
    };
    const FirstHit first_hit = FindHit(inputs, path);
    const double smallest_distance = ComputeSmallestDistanceAlongPath(solid, path, path.duration);
    const bool must_hit = smallest_distance < kTolerance - kReferenceAllowance;
    counts.must_hit += must_hit ? 1 : 0;
    if (first_hit.solid_index == kNoSolid) {
      counts.mistakes += must_hit ? 1 : 0;
      continue;
    }
    const double distance_at_hit = ComputeDistanceAlongPath(solid, path, first_hit.elapsed);
    const bool is_at_grown_surface = distance_at_hit >= kTolerance - kReferenceAllowance
        && distance_at_hit <= kTolerance + kTolerance / 1024 + kReferenceAllowance;
    const bool is_first = first_hit.elapsed == 0
        || ComputeSmallestDistanceAlongPath(solid, path, first_hit.elapsed) >= kTolerance - kReferenceAllowance;
    counts.mistakes += is_at_grown_surface && is_first ? 0 : 1;
    // The search again, alone, from the entry of the two intervals, for its count of advances.
    const Sweep sweep = {.start = path.start, .velocity_for_this_step = path.velocity, .duration = path.duration};
    const OwnPath own_path = ComputeOwnPath(inputs, 0, solid.pose_at_step_start, sweep);
    const RimSearch search = FindRimFromEntry(own_path, solid.half_size, path.duration);
    const bool is_rim_entry = search.hit_time.has_hit && first_hit.at_hit.face != SolidFace::kCurved
        && first_hit.at_hit.face != SolidFace::kLowZ && first_hit.at_hit.face != SolidFace::kHighZ;
    const int count_index = is_rim_entry ? search.advance_count : kMaxRimAdvanceCount + 1;
    ++counts.hits_per_advance_count.at(static_cast<std::size_t>(count_index));
  }
  return counts;
}

// Prints one cylinder's counts of the rim edge sweeps, for the report.
void WriteRimEdgeCounts(const Solid& cylinder, const RimEdgeCounts& counts) {
  std::cout << "rim edge sweeps, radius " << cylinder.half_size.x << ": mistakes " << counts.mistakes << ", must hit "
            << counts.must_hit << ", hits per advance count";
  for (const int hit_count : counts.hits_per_advance_count) {
    std::cout << " " << hit_count;
  }
  std::cout << "\n";
}

TEST(FindFirstHit, GeneratedRimEdgeSweeps_AgreeWithSlowReference) {
  // 50,000 paths through the grown rim edge of each of four cylinders, two along z and two tilted, within 8 units of
  // 2^-53 of the edge, where the wall and cap tests both sit on their boundaries. The reference, the smallest true
  // distance along the path, cannot miss a path that enters the solid: it is a convex function of time.
  const Vec3 tilted_axis = ToColumn(kTiltedAxes, 2);
  const std::array cylinders = {
      ToSolid(Cylinder{.center = {}, .axis = kAxisZ, .radius = 0.5, .half_height = 1}, kNoMotion, 1),
      ToSolid(Cylinder{.center = {}, .axis = kAxisZ, .radius = 1.75, .half_height = 0.375}, kNoMotion, 1),
      ToSolid(
          Cylinder{
              .center = {.x = 0.25, .y = -0.5, .z = 0.125},
              .axis = tilted_axis,
              .radius = 0.3,
              .half_height = 0.7,
          },
          kNoMotion,
          1
      ),
      ToSolid(
          Cylinder{.center = {.x = -1, .y = 2, .z = 3}, .axis = tilted_axis, .radius = 0.125, .half_height = 2.875},
          kNoMotion,
          1
      ),
  };
  // A model of the rim search in double precision, on the same draws, for the two cylinders along z: 1, 2 or 3 advances
  // in 7,189, 32,918 and 9,893 paths; the cylinder's own axes turn the start, which may move a few between neighbours.
  constexpr std::array<int, 3> kModelHitsPerAdvanceCount = {7189, 32918, 9893};
  for (std::size_t cylinder_index = 0; cylinder_index < cylinders.size(); ++cylinder_index) {
    const Solid& cylinder = cylinders.at(cylinder_index);
    const RimEdgeCounts counts = ComputeRimEdgeMistakeCounts(cylinder, 50000);
    EXPECT_EQ(counts.mistakes, 0) << "radius " << cylinder.half_size.x;
    // Nearly every path goes deep into the cylinder, so the check is about hits that must not be missed.
    EXPECT_GT(counts.must_hit, 45000) << "radius " << cylinder.half_size.x;
    WriteRimEdgeCounts(cylinder, counts);
    const bool is_along_z = cylinder_index < 2;
    for (std::size_t advance_count = 1; is_along_z && advance_count <= 3; ++advance_count) {
      const int model_count = kModelHitsPerAdvanceCount.at(advance_count - 1);
      EXPECT_NEAR(counts.hits_per_advance_count.at(advance_count), model_count, 5) << advance_count;
    }
  }
}

// The kinds of path past a cylinder's rim (method paper <sec-round-first-hit>).
enum class RimPathKind : std::uint8_t {
  kThroughRim,  // Through the rim's neighbourhood at every angle.
  kDownWall,  // Down the wall at its rest gap toward the rim, from beyond the cap.
  kAlongCap,  // Inward along the cap's plane at its rest gap, from beyond the wall.
  kGraze,  // Tangent to the round rim or the wall, within a few rounding units or tol / 8.
  kCornerStart,  // From the rim corner of the sharp grown shape, outside the true one.
  kAnywhere,  // From anywhere around the cylinder toward anywhere near it.
};

// One path past a cylinder's rim, in the cylinder's own axes, with the touched rule's state.
struct RimPath {
  RimPathKind kind = RimPathKind::kThroughRim;
  OwnPath own_path{};
  double duration = 0;  // s.
  TouchedRule touched{};
};

// A uniform draw between low and high.
double BuildBetween(CaseGenerator& generator, double low, double high) {
  return low + (high - low) * generator.Next();
}

// A unit vector from three NextSigned, drawn again while shorter than 0.1.
Vec3 BuildUnitDirection(CaseGenerator& generator) {
  Vec3 direction{};
  while (math::Length(direction) < 0.1) {
    direction = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
  }
  return math::Normalized(direction);
}

// Where a point lies beside a rim: its wall and cap distances, in tolerances.
struct RimPointShares {
  double wall_share = 0;
  double cap_share = 0;
};

// A point beside the rim, in own axes: wall distance shares.wall_share tol and cap distance shares.cap_share tol, at an
// azimuth of 2 pi Next and on the high cap when Next < 0.5 (two draws). *radial is the unit vector across the axis
// toward the point and *cap_sign the cap's side.
Vec3 BuildPointBesideRim(
    CaseGenerator& generator,
    Vec3 half_size,
    RimPointShares shares,
    Vec3* radial,
    double* cap_sign
) {
  const double azimuth = 2 * std::numbers::pi * generator.Next();  // rad
  *cap_sign = generator.Next() < 0.5 ? 1.0 : -1.0;
  *radial = {.x = std::cos(azimuth), .y = std::sin(azimuth), .z = 0};
  const double axis_distance = half_size.x + shares.wall_share * kTolerance;  // m
  const double height = half_size.z + shares.cap_share * kTolerance;  // m
  return axis_distance * *radial + Vec3{.x = 0, .y = 0, .z = *cap_sign * height};
}

// An offset from a rest gap, in tolerances: 0, a share up to 2^-2 to 2^-30, or up to 8 units of 2^-44, by floor(3
// Next). Shares out to a quarter put starts beyond the bounding surface's window on both sides.
double BuildRestGapOffset(CaseGenerator& generator) {
  const int choice = static_cast<int>(std::floor(3 * generator.Next()));
  const double share = generator.NextSigned();
  const double exponent = std::floor(BuildBetween(generator, 2, 31));
  const double units = std::floor(BuildBetween(generator, -8, 9));
  if (choice == 0) {
    return 0;
  }
  return choice == 1 ? share * std::exp2(-exponent) : units * 0x1p-44;
}

// A slant of a path off a plane: 0, 0, 1e-12, 1e-6 or 1e-3, by floor(5 Next), times NextSigned.
double BuildSlant(CaseGenerator& generator) {
  constexpr std::array<double, 5> kSlants = {0, 0, 1e-12, 1e-6, 1e-3};
  const double slant = kSlants.at(static_cast<std::size_t>(std::floor(5 * generator.Next())));
  return slant * generator.NextSigned();
}

// Builds one path past the rim of a cylinder of the given half size, in its own axes. Draws, in this order: the kind
// (0.35 through the rim, 0.15 down the wall, 0.15 along the cap, 0.15 a graze, 0.1 from the corner, 0.1 anywhere, by
// one Next); the kind's own draws (the speed 2^(6 Next - 3) m/s, path lengths tol 2^(1 + 11 Next)); and whether the
// touched rule applies, Next < 0.3, with an allowance of 2^-46 times the sum of the velocity's absolute coordinates.
RimPath BuildRimPath(CaseGenerator& generator, Vec3 half_size) {
  const double radius = half_size.x;  // m
  const double half_height = half_size.z;  // m
  const double kind_draw = generator.Next();
  RimPath path;
  Vec3 radial{};
  double cap_sign = 1;
  const auto draw_speed = [&generator] {
    return std::exp2(BuildBetween(generator, -3, 3));
  };  // m/s
  if (kind_draw < 0.35) {
    path.kind = RimPathKind::kThroughRim;
    const double wall_share = BuildBetween(generator, -2.5, 2.5);
    const double cap_share = BuildBetween(generator, -2.5, 2.5);
    const Vec3 target = BuildPointBesideRim(
        generator,
        half_size,
        {.wall_share = wall_share, .cap_share = cap_share},
        &radial,
        &cap_sign
    );
    const Vec3 direction = BuildUnitDirection(generator);
    const double length = kTolerance * std::exp2(BuildBetween(generator, 1, 12));  // m
    const double speed = draw_speed();
    path.own_path = {.start = target - length * direction, .velocity = speed * direction};
    path.duration = 2 * length / speed;
  } else if (kind_draw < 0.5) {
    path.kind = RimPathKind::kDownWall;
    const double offset = BuildRestGapOffset(generator);
    const Vec3 start = BuildPointBesideRim(
        generator,
        half_size,
        {.wall_share = 1 + offset, .cap_share = BuildBetween(generator, 2, 2000)},
        &radial,
        &cap_sign
    );
    const double speed = draw_speed();
    const double slant = BuildSlant(generator);
    path.own_path = {.start = start, .velocity = speed * (slant * radial - Vec3{.x = 0, .y = 0, .z = cap_sign})};
    const double length = std::abs(start.z) - half_height + std::min(half_height, 100 * kTolerance);  // m
    path.duration = 1.5 * length / speed;
  } else if (kind_draw < 0.65) {
    path.kind = RimPathKind::kAlongCap;
    const double offset = BuildRestGapOffset(generator);
    const Vec3 start = BuildPointBesideRim(
        generator,
        half_size,
        {.wall_share = BuildBetween(generator, 2, 2000), .cap_share = 1 + offset},
        &radial,
        &cap_sign
    );
    const double speed = draw_speed();
    const double slant = BuildSlant(generator);
    const double sideways = BuildBetween(generator, -0.5, 0.5);
    const Vec3 across = {.x = -radial.y, .y = radial.x, .z = 0};
    path.own_path = {
        .start = start,
        .velocity = speed * (sideways * across - radial + Vec3{.x = 0, .y = 0, .z = cap_sign * slant}),
    };
    const double length = (std::hypot(start.x, start.y) - radius) + std::min(radius, 100 * kTolerance);  // m
    path.duration = 1.5 * length / speed;
  } else if (kind_draw < 0.8) {
    path.kind = RimPathKind::kGraze;
    const double angle = BuildBetween(generator, 0, std::numbers::pi / 2);  // rad
    const bool is_rounding_offset = generator.Next() < 0.5;
    const double rounding_offset = std::floor(BuildBetween(generator, -64, 65)) * 0x1p-46;
    const double share_offset = BuildBetween(generator, -0.125, 0.125);
    const double offset = is_rounding_offset ? rounding_offset : share_offset;
    const RimPointShares shares = {
        .wall_share = (1 + offset) * std::cos(angle),
        .cap_share = (1 + offset) * std::sin(angle),
    };
    const Vec3 target = BuildPointBesideRim(generator, half_size, shares, &radial, &cap_sign);
    const Vec3 rim_normal = std::cos(angle) * radial + Vec3{.x = 0, .y = 0, .z = cap_sign * std::sin(angle)};
    Vec3 direction{};
    while (math::Length(direction) < 0.1) {
      const Vec3 drawn = BuildUnitDirection(generator);
      direction = drawn - math::Dot(drawn, rim_normal) * rim_normal;
    }
    direction = math::Normalized(direction);
    const double length = kTolerance * std::exp2(BuildBetween(generator, 1, 12));  // m
    const double speed = draw_speed();
    path.own_path = {.start = target - length * direction, .velocity = speed * direction};
    path.duration = 2 * length / speed;
  } else if (kind_draw < 0.9) {
    path.kind = RimPathKind::kCornerStart;
    double wall_share = 0;
    double cap_share = 0;
    while (wall_share * wall_share + cap_share * cap_share <= 1.0001) {
      wall_share = generator.Next();
      cap_share = generator.Next();
    }
    const Vec3 start = BuildPointBesideRim(
        generator,
        half_size,
        {.wall_share = wall_share, .cap_share = cap_share},
        &radial,
        &cap_sign
    );
    const Vec3 direction = BuildUnitDirection(generator);
    const double speed = draw_speed();
    path.own_path = {.start = start, .velocity = speed * direction};
    path.duration = kTolerance * std::exp2(BuildBetween(generator, 0, 8)) / speed;
  } else {
    path.kind = RimPathKind::kAnywhere;
    Vec3 start{};
    do {
      start = {
          .x = BuildBetween(generator, -3, 3) * radius,
          .y = BuildBetween(generator, -3, 3) * radius,
          .z = BuildBetween(generator, -3, 3) * half_height,
      };
    } while (ComputeDistanceToCylinder(start, half_size, kIdentity).distance <= 2 * kTolerance);
    const Vec3 target = {
        .x = BuildBetween(generator, -1.2, 1.2) * radius,
        .y = BuildBetween(generator, -1.2, 1.2) * radius,
        .z = BuildBetween(generator, -1.2, 1.2) * half_height,
    };
    const double speed_factor = 1 + generator.Next();  // 1/s
    path.own_path = {.start = start, .velocity = speed_factor * (target - start)};
    path.duration = 1;
  }
  const Vec3 velocity = path.own_path.velocity;
  const double allowance = 0x1p-46 * (std::abs(velocity.x) + std::abs(velocity.y) + std::abs(velocity.z));  // m/s
  path.touched =
      generator.Next() < 0.3 ? TouchedRule{.is_touched = true, .approach_allowance = allowance} : TouchedRule{};
  return path;
}

// What the paths past a cylinder's rim gave.
struct RimPathCounts {
  int paths = 0;
  int deep_starts = 0;  // Starts deeper than tol inside, not swept.
  int hits = 0;
  // Free by the touching rule on the distance (method paper <sec-touching-rule>, <sec-round-starts>).
  int free_by_touching_rule = 0;
  int free_resting = 0;  // Free because the start rests on a bounding surface (method paper <sec-touching-rule>).
  int free_other = 0;
  int near_limit =
      0;  // Starts within rounding of a limit of the bounding surfaces' rule, where either answer is right.
  int mistakes = 0;
  std::array<int, kMaxRimAdvanceCount + 1> searches_per_advance_count{};
  double least_resting_distance = 1;  // m, the smallest true distance along a resting path.
};

// Whether a start beside a cylinder's rim rests on its wall or its cap's plane by the numbers of method paper
// <sec-touching-rule>, written here (-tol / 8, tol / 16, tol / 64), on gaps and rates computed apart from the code: in
// world axes, with std::hypot. Near a limit, within 2^-20 tol, rounding may decide either way, and is_near_limit says
// so.
struct RestingStart {
  bool is_beside_rim = false;
  bool is_resting = false;
  bool is_near_limit = false;
};

// Classifies one start by the decision's numbers (RestingStart).
RestingStart ComputeRestingStartClass(const Solid& cylinder, Path path) {
  const Vec3 axis = ToColumn(cylinder.pose_at_step_start.axes, 2);
  const Vec3 from_center = path.start - cylinder.pose_at_step_start.origin;  // m
  const double height = math::Dot(from_center, axis);  // m
  const Vec3 across = from_center - height * axis;  // m
  const double axis_distance = std::hypot(across.x, across.y, across.z);  // m
  const Vec3 radial = (1 / axis_distance) * across;
  const double wall_gap = axis_distance - cylinder.half_size.x - kTolerance;  // m
  const double cap_gap = std::abs(height) - cylinder.half_size.z - kTolerance;  // m
  const double wall_sink = math::Dot(radial, path.velocity) * path.duration;  // m
  const double cap_sink = (height >= 0 ? 1.0 : -1.0) * math::Dot(axis, path.velocity) * path.duration;  // m
  const double near = 0x1p-20 * kTolerance;  // m
  const auto distance_to_limit = [](double value, double limit) {
    return std::abs(value - limit);
  };
  const auto is_resting_on = [&](double gap, double sink) {
    return gap >= -kTolerance / 8 && gap <= kTolerance / 16 && sink >= -kTolerance / 64;
  };
  const auto is_near = [&](double gap, double sink) {
    return distance_to_limit(gap, -kTolerance / 8) <= near || distance_to_limit(gap, kTolerance / 16) <= near
        || distance_to_limit(sink, -kTolerance / 64) <= near;
  };
  RestingStart start;
  start.is_beside_rim = wall_gap + kTolerance > near && cap_gap + kTolerance > near;
  start.is_resting = start.is_beside_rim && (is_resting_on(wall_gap, wall_sink) || is_resting_on(cap_gap, cap_sink));
  start.is_near_limit = start.is_beside_rim && (is_near(wall_gap, wall_sink) || is_near(cap_gap, cap_sink));
  return start;
}

// Checks a path that the closed form and the rim search decide: the search run alone from the entry gives the same
// answer bit for bit within 8 advances; the path stays at tol or more before a hit or over the sweep; a hit is between
// tol and tol + tol / 1024, up to kReferenceAllowance.
bool CheckSearchedRimPath(
    const Solid& cylinder,
    Path path,
    OwnPath own_path,
    FirstHit first_hit,
    RimPathCounts& counts
) {
  const bool is_hit = first_hit.solid_index != kNoSolid;
  const double end_elapsed = is_hit ? first_hit.elapsed : path.duration;  // s
  const RimSearch search = FindRimFromEntry(own_path, cylinder.half_size, path.duration);
  const bool is_same = search.hit_time.has_hit == is_hit && search.hit_time.elapsed == first_hit.elapsed;
  const SolidFace face = first_hit.at_hit.face;
  const bool is_rim_hit = is_hit && (face == SolidFace::kLowRim || face == SolidFace::kHighRim);
  if (search.advance_count > 0 || is_rim_hit) {
    ++counts.searches_per_advance_count.at(static_cast<std::size_t>(search.advance_count));
  }
  counts.hits += is_hit ? 1 : 0;
  counts.free_other += is_hit ? 0 : 1;
  const bool stays_outside =
      ComputeSmallestDistanceAlongPath(cylinder, path, end_elapsed) >= kTolerance - kReferenceAllowance;
  const double distance_at_hit = is_hit ? ComputeDistanceAlongPath(cylinder, path, first_hit.elapsed) : kTolerance;
  const bool is_at_rest_gap = distance_at_hit >= kTolerance - kReferenceAllowance
      && distance_at_hit <= kTolerance + kTolerance / 1024 + kReferenceAllowance;
  return is_same && search.advance_count <= 8 && stays_outside && is_at_rest_gap;
}

// Sweeps one path past a cylinder's rim and checks it against the slow reference, the start classified apart from the
// code (ComputeRestingStartClass), before the touching rule as the sweep tries it: a start that rests on a bounding
// surface is free and keeps the smallest true distance at 55 / 64 tol - kReferenceAllowance or more; near a limit of
// the decision's numbers, that or CheckSearchedRimPath's answer; a path freed by the touching rule keeps its own linear
// bound; any other path as CheckSearchedRimPath checks it.
void CheckRimPath(const Solid& cylinder, const RimPath& rim_path, RimPathCounts& counts) {
  const std::array solids = {cylinder};
  const SolidContactInputs inputs = BuildInputs(solids, rim_path.duration);
  const SolidPose pose = cylinder.pose_at_step_start;
  const Path path = {
      .start = pose.origin + pose.axes * rim_path.own_path.start,
      .velocity = pose.axes * rim_path.own_path.velocity,
      .duration = rim_path.duration,
  };
  ++counts.paths;
  const DistanceToSolid at_start = ComputeDistanceToSolid(cylinder, pose, path.start);
  const double start_gap = at_start.distance - kTolerance;  // m
  if (start_gap < -kTolerance) {
    ++counts.deep_starts;
    return;
  }
  const Sweep sweep = {.start = path.start, .velocity_for_this_step = path.velocity, .duration = path.duration};
  FirstHit first_hit;
  ASSERT_EQ(FindFirstHitOfSweep(inputs, sweep, rim_path.touched, &first_hit, nullptr), ContactError::kNone);
  const bool is_hit = first_hit.solid_index != kNoSolid;
  const double end_elapsed = is_hit ? first_hit.elapsed : path.duration;  // s
  const OwnPath own_path = ComputeOwnPath(inputs, 0, pose, sweep);
  const RestingStart resting = ComputeRestingStartClass(cylinder, path);
  const double resting_gap_limit = rim_path.touched.is_touched ? kTolerance / 16 : 0;  // m
  const bool is_touching_rule = start_gap <= resting_gap_limit;
  const double least_resting_distance = 55 * kTolerance / 64 - kReferenceAllowance;  // m
  bool is_right = true;
  if (resting.is_near_limit) {
    // Either rule may decide here; the safe bound of a resting sweep holds whichever does.
    const double smallest_distance = ComputeSmallestDistanceAlongPath(cylinder, path, end_elapsed);  // m
    const bool is_rested = !is_hit && smallest_distance >= least_resting_distance;
    is_right = is_rested || is_touching_rule || CheckSearchedRimPath(cylinder, path, own_path, first_hit, counts);
    ++counts.near_limit;
  } else if (resting.is_resting) {
    const double smallest_distance = ComputeSmallestDistanceAlongPath(cylinder, path, end_elapsed);  // m
    is_right = !is_hit && smallest_distance >= least_resting_distance;
    counts.least_resting_distance = std::min(counts.least_resting_distance, smallest_distance);
    ++counts.free_resting;
  } else if (is_touching_rule) {
    // The touching rule's own bound: the true distance, convex along the path, stays above its tangent at the start.
    const double start_rate = math::Dot(at_start.normal_out_of_solid, path.velocity);  // m/s
    const double linear_bound = at_start.distance + std::min(start_rate * end_elapsed, 0.0);  // m
    is_right = ComputeSmallestDistanceAlongPath(cylinder, path, end_elapsed) >= linear_bound - kReferenceAllowance;
    counts.free_by_touching_rule += is_hit ? 0 : 1;
    counts.hits += is_hit ? 1 : 0;
  } else {
    is_right = CheckSearchedRimPath(cylinder, path, own_path, first_hit, counts);
  }
  if (!is_right) {
    ++counts.mistakes;
    ADD_FAILURE() << "kind " << static_cast<int>(rim_path.kind) << ", hit " << is_hit;
  }
}

// Prints the counts of the paths past a cylinder's rim, for the report, and returns the searches and their advances.
std::pair<int, int> WriteRimPathCounts(const RimPathCounts& counts) {
  int searches = 0;
  int advances = 0;
  std::cout << "paths past a rim " << counts.paths << ", deep starts " << counts.deep_starts << ", hits " << counts.hits
            << ", free by the touching rule " << counts.free_by_touching_rule << ", resting " << counts.free_resting
            << ", other free " << counts.free_other << ", near a limit " << counts.near_limit << ", mistakes "
            << counts.mistakes << ", least resting distance " << counts.least_resting_distance / kTolerance
            << " tol, searches per advance count";
  for (std::size_t advance_count = 0; advance_count < counts.searches_per_advance_count.size(); ++advance_count) {
    const int search_count = counts.searches_per_advance_count.at(advance_count);
    std::cout << " " << search_count;
    searches += search_count;
    advances += search_count * static_cast<int>(advance_count);
  }
  std::cout << "\n";
  return {searches, advances};
}

TEST(FindFirstHit, GeneratedPathsPastACylinderRim_AgreeWithSlowReference) {
  // 20,000 paths (BuildRimPath) past the rims of the four cylinders of GeneratedRimEdgeSweeps, the cylinder by one
  // Next before each path, in six kinds: through the rim's neighbourhood, down the wall and along the cap at their rest
  // gaps, grazing the round rim, from the rim corner, and from anywhere, a third with the touched rule. The model on
  // its own draws: no missed hit, no hit farther than tol / 1024 plus rounding, at most 5 advances, 2.0 on average, and
  // the smallest true gap along a resting path -0.126 tol.
  const Vec3 tilted_axis = ToColumn(kTiltedAxes, 2);
  const std::array cylinders = {
      ToSolid(Cylinder{.center = {}, .axis = kAxisZ, .radius = 0.5, .half_height = 1}, kNoMotion, 1),
      ToSolid(Cylinder{.center = {}, .axis = kAxisZ, .radius = 1.75, .half_height = 0.375}, kNoMotion, 1),
      ToSolid(
          Cylinder{
              .center = {.x = 0.25, .y = -0.5, .z = 0.125},
              .axis = tilted_axis,
              .radius = 0.3,
              .half_height = 0.7,
          },
          kNoMotion,
          1
      ),
      ToSolid(
          Cylinder{.center = {.x = -1, .y = 2, .z = 3}, .axis = tilted_axis, .radius = 0.125, .half_height = 2.875},
          kNoMotion,
          1
      ),
  };
  CaseGenerator generator;
  RimPathCounts counts;
  for (int path_index = 0; path_index < 20000; ++path_index) {
    const Solid& cylinder = cylinders.at(static_cast<std::size_t>(std::floor(4 * generator.Next())));
    const RimPath rim_path = BuildRimPath(generator, cylinder.half_size);
    CheckRimPath(cylinder, rim_path, counts);
  }
  const auto [searches, advances] = WriteRimPathCounts(counts);
  EXPECT_EQ(counts.paths, 20000);
  EXPECT_EQ(counts.mistakes, 0);
  EXPECT_GT(searches, 3000);
  EXPECT_GT(counts.free_resting, 3000);
  EXPECT_GT(counts.hits, 5000);
  EXPECT_LE(advances, 3 * searches);
}

TEST(FindFirstHit, GeneratedGrazingRoundSweeps_NeverHitBehindStart) {
  // Paths that pass the grown sphere or the grown cylinder wall at a distance of R + k 2^-52 m from the centre or the
  // axis, k from -64 to 64: tangent up to a few rounding units, where the root's square root is the rounding of a
  // near-zero difference. Each must succeed, and agree with the slow reference.
  const Mat3 axes =
      ToSolid(Cylinder{.center = {}, .axis = ToColumn(kTiltedAxes, 2), .radius = 0.5, .half_height = 1}, kNoMotion, 1)
          .pose_at_step_start.axes;
  const Vec3 along_path = ToColumn(axes, 0);
  const Vec3 toward_offset = ToColumn(axes, 1);
  const std::array round_solids = {
      BuildSphere({}, kNoMotion),
      BuildCylinder({}, ToColumn(axes, 2), kNoMotion),
  };
  for (const Solid& solid : round_solids) {
    const std::array solids = {solid};
    const SolidContactInputs inputs = BuildInputs(solids, 4);
    for (int offset_units = -64; offset_units <= 64; ++offset_units) {
      const double offset = 0.5 + kTolerance + offset_units * 0x1p-52;  // m
      const Path path = {.start = -2 * along_path + offset * toward_offset, .velocity = along_path, .duration = 4};
      const FirstHit first_hit = FindHit(inputs, path);
      CheckFirstHitAgreesWithReference(solid, kTolerance, path, first_hit, offset_units);
    }
  }
}

// --- Generated sweeps toward turning solids ---

// m/s. How fast the distance from a path to a turning solid can change over the first end_elapsed seconds of the
// sweep, written apart from FindFirstHit's bound so that a mistake there does not hide here:
// |v - v_lin| + |omega| (|x(0) - o(start_elapsed)| + |v - v_lin| end_elapsed). By the triangle inequality the bracket
// is at least the particle's distance to the solid's origin at every moment of the sweep, so this is at least the
// particle's speed relative to the solid.
double ComputeReferenceRateBound(const Solid& solid, Path path, double end_elapsed) {
  const Vec3 relative_velocity = path.velocity - solid.motion.linear_velocity;
  const double relative_speed = math::Length(relative_velocity);
  const double distance_to_origin_at_start =
      math::Length(path.start - ComputeSolidPose(solid, path.start_elapsed).origin);
  const double largest_distance_to_origin = distance_to_origin_at_start + relative_speed * end_elapsed;
  return relative_speed + math::Length(solid.motion.angular_velocity) * largest_distance_to_origin;
}

// One time of a sweep and the distance from the path to the solid then.
struct DistanceSample {
  double elapsed = 0;  // s, since the start of the sweep.
  double distance = 0;  // m.
};

// Returns whether the distance from the path to the solid surely stays at least tol - kReferenceAllowance between 0
// and end_elapsed seconds into the sweep: the path never comes inside the grown solid by more than rounding. Between
// two samples a and b the distance changes no faster than the reference rate bound, so it is at least the larger of
// d(a) - bound (t - a) and d(b) - bound (b - t), whose smallest value over [a, b] is (d(a) + d(b) - bound (b - a)) / 2.
// An interval where that is too small is cut in half until it passes, a sample is too close, or it can no longer be
// cut: so the answer cannot miss a dip, whatever FindFirstHit did. *sample_count counts the distances computed.
bool ComputeIsOutsideGrownSolidThroughout(const Solid& solid, Path path, double end_elapsed, int* sample_count) {
  const double floor_distance = kTolerance - kReferenceAllowance;  // m
  const double rate_bound = ComputeReferenceRateBound(solid, path, end_elapsed);  // m/s
  const DistanceSample at_start = {.elapsed = 0, .distance = ComputeDistanceAlongPath(solid, path, 0)};
  const DistanceSample at_end = {
      .elapsed = end_elapsed,
      .distance = ComputeDistanceAlongPath(solid, path, end_elapsed),
  };
  std::vector<std::array<DistanceSample, 2>> pending_intervals;
  pending_intervals.push_back({at_start, at_end});
  *sample_count += 2;
  while (!pending_intervals.empty()) {
    const auto [low, high] = pending_intervals.back();
    pending_intervals.pop_back();
    if (low.distance < floor_distance || high.distance < floor_distance) {
      return false;
    }
    const double width = high.elapsed - low.elapsed;  // s
    const double lowest_possible_distance = (low.distance + high.distance - rate_bound * width) / 2;  // m
    if (lowest_possible_distance >= floor_distance) {
      continue;
    }
    const double middle_elapsed = low.elapsed + width / 2;  // s
    const bool can_cut = middle_elapsed > low.elapsed && middle_elapsed < high.elapsed;
    if (!can_cut) {
      return false;
    }
    const DistanceSample middle = {
        .elapsed = middle_elapsed,
        .distance = ComputeDistanceAlongPath(solid, path, middle_elapsed),
    };
    ++*sample_count;
    pending_intervals.push_back({low, middle});
    pending_intervals.push_back({middle, high});
  }
  return true;
}

// How a generated set checks one path: whether the path surely stays at distance tol - allowance or more from solid 0
// between 0 and end_elapsed seconds into the sweep, and that allowance.
struct SweepReference {
  bool (*is_outside_throughout)(const SolidContactInputs& inputs, Path path, double end_elapsed, int* sample_count);
  double allowance = 0;  // m
};

// The lower-bound reference (ComputeIsOutsideGrownSolidThroughout) as a SweepReference, for the test contact tolerance.
bool ComputeIsOutsideByLowerBound(const SolidContactInputs& inputs, Path path, double end_elapsed, int* sample_count) {
  return ComputeIsOutsideGrownSolidThroughout(inputs.solids[0], path, end_elapsed, sample_count);
}

constexpr SweepReference kLowerBoundReference = {
    .is_outside_throughout = ComputeIsOutsideByLowerBound,
    .allowance = kReferenceAllowance,
};

// The outcomes of generated sweeps toward one turning solid, against a reference.
struct TurningSweepCounts {
  int hit_count = 0;
  int free_count = 0;
  int close_call_count = 0;  // kTooCloseToCall: the documented answer to a path the search cannot finish.
  int mistake_count = 0;  // Disagreements with the reference.
  int reference_sample_count = 0;  // Distances the reference computed.
  std::vector<int> advance_counts;  // The advances of the stepping search, one per path, close calls included.
};

// Runs FindFirstHit on one path toward the turning solid 0 and adds its outcome to *counts, with the advances that
// FindHitOnTurningSolid, called directly, makes for it. A mistake is: a hit or a free path that is not surely outside
// the grown solid before the hit or over the whole duration (the reference); a hit whose distance is above
// (1 + kTurningSolidHitGapShare) tol + allowance, so not at the grown surface, or that reports another distance; a
// time outside the sweep; a direct search that answers otherwise; or any other error.
void ApplyTurningSweepCounts(
    const SolidContactInputs& inputs,
    Path path,
    SweepReference reference,
    TurningSweepCounts* counts
) {
  const Solid& solid = inputs.solids[0];
  const double contact_tolerance = inputs.contact_tolerance;
  FirstHit first_hit;
  const ContactError error =
      FindFirstHit(inputs, path.start, path.velocity, path.start_elapsed, path.duration, &first_hit);
  const TurningSearch search = FindTurningSolid(inputs, path);
  counts->advance_counts.push_back(search.advance_count);
  const bool search_agrees = search.error == error && search.first_hit.solid_index == first_hit.solid_index
      && search.first_hit.elapsed == first_hit.elapsed;
  if (!search_agrees) {
    ++counts->mistake_count;
    return;
  }
  if (error == ContactError::kTooCloseToCall) {
    ++counts->close_call_count;
    return;
  }
  const bool is_hit = first_hit.solid_index != kNoSolid;
  const double end_elapsed = is_hit ? first_hit.elapsed : path.duration;
  const bool is_within_sweep = end_elapsed >= 0 && end_elapsed <= path.duration;
  const bool stays_outside_grown_solid =
      is_within_sweep && reference.is_outside_throughout(inputs, path, end_elapsed, &counts->reference_sample_count);
  if (error != ContactError::kNone || !stays_outside_grown_solid) {
    ++counts->mistake_count;
    return;
  }
  if (!is_hit) {
    ++counts->free_count;
    return;
  }
  const double distance_at_hit = ComputeDistanceAlongPath(solid, path, first_hit.elapsed);
  const bool is_at_grown_surface =
      distance_at_hit <= (1 + kTurningSolidHitGapShare) * contact_tolerance + reference.allowance;
  const bool reports_that_distance = std::abs(first_hit.at_hit.distance - distance_at_hit) <= reference.allowance;
  if (is_at_grown_surface && reports_that_distance) {
    ++counts->hit_count;
  } else {
    ++counts->mistake_count;
  }
}

// Runs 1000 generated sweeps toward one moving, turning solid, in a step of 1.5 s. Draw order per case: start_elapsed
// 0.5 * Next(); a start of 2.5 times three NextSigned() values from the solid's origin at that moment, drawn again
// while it is within 2 tol of the solid (so the search does not stop at once); a target of 0.9 times three NextSigned()
// values from that origin; a speed factor 1 + Next(). The path heads for the target, which it reaches within
// 1 / factor <= 1 s, while the solid turns under it.
TurningSweepCounts ComputeGeneratedTurningSweepCounts(const Solid& solid) {
  const std::array solids = {solid};
  const SolidContactInputs inputs = BuildInputs(solids, 1.5);
  CaseGenerator generator;
  TurningSweepCounts counts;
  for (int case_index = 0; case_index < 1000; ++case_index) {
    const double start_elapsed = 0.5 * generator.Next();
    const SolidPose pose = ComputeSolidPose(solid, start_elapsed);
    Vec3 start{};
    do {
      start = pose.origin
          + 2.5 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    } while (ComputeDistanceToSolid(solid, pose, start).distance <= 2 * kTolerance);
    const Vec3 target =
        pose.origin + 0.9 * Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    const double speed_factor = 1 + generator.Next();  // 1/s
    const Vec3 velocity = solid.motion.linear_velocity + speed_factor * (target - start);
    ApplyTurningSweepCounts(
        inputs,
        {.start = start, .velocity = velocity, .start_elapsed = start_elapsed, .duration = 1},
        kLowerBoundReference,
        &counts
    );
  }
  return counts;
}

// What a generated set must meet.
struct SweepCriteria {
  int largest_close_call_count = 0;
  double largest_mean_advance_count = 10;
  int largest_99th_percentile_advance_count = 30;
  bool requires_hits = true;  // true: more than a tenth of the paths must be hits, so the hit answer is checked.
};

// The 99th percentile of the advance counts: the count that 99 % of the paths do not exceed.
int ComputeAdvanceCountPercentile99(std::vector<int> advance_counts) {
  std::ranges::sort(advance_counts);
  return advance_counts.at((advance_counts.size() * 99) / 100);
}

// Fails the calling test when a generated set has a mistake, more close calls than the criteria allow, a mean or a 99th
// percentile of advances above them, or, when the criteria require it, hits on no more than a tenth of its paths. A
// close call is safe but costs a shorter step; it remains for paths that hover near the solid with almost no motion
// along its normal, where each advance is only about Sqrt(2 g / K).
void CheckTurningSweepCounts(const TurningSweepCounts& counts, SweepCriteria criteria, const char* set_name) {
  const int path_count = counts.hit_count + counts.free_count + counts.close_call_count + counts.mistake_count;
  int advance_sum = 0;
  for (const int advance_count : counts.advance_counts) {
    advance_sum += advance_count;
  }
  const double mean_advance_count = static_cast<double>(advance_sum) / static_cast<double>(path_count);
  const int percentile_99_advance_count = ComputeAdvanceCountPercentile99(counts.advance_counts);
  EXPECT_EQ(counts.mistake_count, 0) << set_name;
  if (criteria.requires_hits) {
    EXPECT_GT(counts.hit_count, path_count / 10) << set_name;
  }
  EXPECT_LE(counts.close_call_count, criteria.largest_close_call_count) << set_name;
  EXPECT_LE(mean_advance_count, criteria.largest_mean_advance_count) << set_name;
  EXPECT_LE(percentile_99_advance_count, criteria.largest_99th_percentile_advance_count) << set_name;
  // Recorded for the report on the test, not a check.
  const std::string name = set_name;
  testing::Test::RecordProperty(name + "_hits", counts.hit_count);
  testing::Test::RecordProperty(name + "_free", counts.free_count);
  testing::Test::RecordProperty(name + "_close_calls", counts.close_call_count);
  testing::Test::RecordProperty(name + "_reference_samples", counts.reference_sample_count);
  testing::Test::RecordProperty(name + "_mean_advances", std::to_string(mean_advance_count));
  testing::Test::RecordProperty(name + "_advances_99th_percentile", percentile_99_advance_count);
  testing::Test::RecordProperty(name + "_most_advances", *std::ranges::max_element(counts.advance_counts));
}

TEST(FindFirstHit, GeneratedTurningSweeps_AgreeWithLowerBoundReference) {
  // Five solids that move at {0.25, -0.5, 0.125} m/s and turn about different axes, each at most 1.05 rad/s so that
  // the turn of a step of 1.5 s stays within a quarter turn: two boxes, a plate, a flat solid and a cylinder whose axis
  // tilts. 1000 paths each.
  const Vec3 center = {.x = 0.25, .y = 0.5, .z = -0.125};
  const Vec3 linear_velocity = {.x = 0.25, .y = -0.5, .z = 0.125};
  const Box box = {.center = center, .axes = kTiltedAxes, .half_size = {.x = 0.5, .y = 0.25, .z = 1}};
  const Vec3 tilted_axis = ToColumn(kTiltedAxes, 2);
  const auto solids = std::to_array<std::pair<const char*, Solid>>({
      {
          "box",
          ToSolid(box, {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0.48, .y = -0.64, .z = 0.3}}, 1),
      },
      {
          "box_about_z",
          ToSolid(box, {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0, .y = 0, .z = -1}}, 1),
      },
      {
          "plate",
          ToSolid(
              Plate{.center = center, .normal = tilted_axis, .half_thickness = 0.125},
              {.linear_velocity = linear_velocity, .angular_velocity = {.x = -0.5, .y = 0.2, .z = 0.6}},
              1
          ),
      },
      {
          "flat_solid",
          ToSolid(
              FlatSolid{.point_on_surface = center, .normal_out_of_solid = tilted_axis},
              {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0.3, .y = 0.9, .z = 0}},
              1
          ),
      },
      {
          "cylinder",
          ToSolid(
              Cylinder{.center = center, .axis = tilted_axis, .radius = 0.5, .half_height = 0.75},
              {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0.7, .y = 0.4, .z = -0.5}},
              1
          ),
      },
  });
  for (const auto& [solid_name, solid] : solids) {
    ASSERT_EQ(CheckSolid(solid, 1.5), ContactError::kNone) << solid_name;
    const TurningSweepCounts counts = ComputeGeneratedTurningSweepCounts(solid);
    // Measured: no close call (first order alone: 32 to 56 per solid).
    CheckTurningSweepCounts(counts, {.largest_close_call_count = 1}, solid_name);
    // Enough paths miss for the free answer to be checked too.
    EXPECT_GT(counts.free_count, 100) << solid_name;
  }
}

// A long box of half sizes {0.5, 4, 0.5} turned by kTiltedAxes: a point far along its face kHighX moves almost
// straight toward that face when the box turns about its own z axis.
constexpr Box kLongTiltedBox = {
    .center = {.x = 0.25, .y = 0.5, .z = -0.125},
    .axes = kTiltedAxes,
    .half_size = {.x = 0.5, .y = 4, .z = 0.5},
};

TEST(FindFirstHit, GeneratedNearlyTightBoundSweeps_AgreeWithLowerBoundReference) {
  // Paths on which the bound is nearly tight, so that a bound too small by a few percent would let an advance pass
  // the hit: a long box of half sizes {0.5, 4, 0.5} turned by kTiltedAxes, turning at 1 rad/s about its own z axis,
  // so the turn carries a point far along face kHighX almost straight toward it. Some paths also slide outward along
  // the face, so the particle is farther from the box's origin at the end of the sweep than at its start. 2000 paths,
  // each with its own box velocity, so the box's origin moves differently between the step's start, the sweep's start
  // and its end. Draw order per path: the box velocity, three NextSigned() values drawn again while their length is
  // below 0.1, scaled to 2 m/s; start_elapsed 0.5 Next(); an own start {0.55 + 0.3 Next(), -(0.5 + 2 Next()),
  // 0.4 NextSigned()} at the pose of start_elapsed; an own velocity relative to the box {-0.5 Next(), -1.5 Next(),
  // 0.1 NextSigned()} in the axes of that pose. Sweeps of 1 s in a step of 1.5 s.
  const Vec3 own_z_axis = ToColumn(kTiltedAxes, 2);
  CaseGenerator generator;
  TurningSweepCounts counts;
  constexpr int kPathCount = 2000;
  for (int path_index = 0; path_index < kPathCount; ++path_index) {
    Vec3 direction{};
    do {
      direction = Vec3{.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};
    } while (math::Length(direction) < 0.1);
    const Solid solid = ToSolid(
        kLongTiltedBox,
        {.linear_velocity = 2 * math::Normalized(direction), .angular_velocity = own_z_axis},
        1
    );
    const double start_elapsed = 0.5 * generator.Next();
    const Vec3 own_start = {
        .x = 0.55 + 0.3 * generator.Next(),
        .y = -(0.5 + 2 * generator.Next()),
        .z = 0.4 * generator.NextSigned(),
    };
    const Vec3 own_relative_velocity = {
        .x = -0.5 * generator.Next(),
        .y = -1.5 * generator.Next(),
        .z = 0.1 * generator.NextSigned(),
    };
    const SolidPose pose = ComputeSolidPose(solid, start_elapsed);
    const std::array solids = {solid};
    const Path path = {
        .start = pose.origin + pose.axes * own_start,
        .velocity = solid.motion.linear_velocity + pose.axes * own_relative_velocity,
        .start_elapsed = start_elapsed,
        .duration = 1,
    };
    ApplyTurningSweepCounts(BuildInputs(solids, 1.5), path, kLowerBoundReference, &counts);
  }
  // Measured: no close call, since the approach uses nearly the whole bound.
  CheckTurningSweepCounts(counts, {.largest_close_call_count = 2}, "nearly_tight_bound");
}

// How a generated path is aimed at a point of a turning solid: it crosses own_target, a point in the solid's own
// axes, aim_elapsed seconds into a sweep of 1 s that starts start_elapsed seconds into the step, approaching along
// own_inward (no unit, any length) at speed relative to the surface there.
struct PathAim {
  double start_elapsed = 0;  // s.
  double aim_elapsed = 0;  // s, since the start of the sweep.
  Vec3 own_target{};  // m, own axes, from the origin.
  Vec3 own_inward{};  // No unit, own axes.
  double speed = 0;  // m/s, relative to the surface at the target.
};

// Returns the path that an aim describes: the world velocity, and the start that goes with it.
Path BuildPathAimedAtOwnPoint(const Solid& solid, PathAim aim) {
  const SolidPose aim_pose = ComputeSolidPose(solid, aim.start_elapsed + aim.aim_elapsed);
  const Vec3 target = aim_pose.origin + aim_pose.axes * aim.own_target;
  const Vec3 surface_velocity = ComputeSolidVelocity(solid, aim_pose, target);
  const Vec3 velocity = surface_velocity + aim.speed * math::Normalized(aim_pose.axes * aim.own_inward);
  return {
      .start = target - velocity * aim.aim_elapsed,
      .velocity = velocity,
      .start_elapsed = aim.start_elapsed,
      .duration = 1,
  };
}

// Draws one aim at an edge or a corner of a box of the given half sizes, within 2 tol of its grown shape. Draw order:
// start_elapsed 0.5 Next(); aim_elapsed 0.25 + 0.5 Next(); a corner when Next() < 0.5, else an edge along the own axis
// floor(3 Next()) at NextSigned() times its half size; per own axis a sign, + when Next() < 0.5, and an offset of
// 2 tol NextSigned(); an approach along minus Next() times the feature's outward diagonal plus 0.8 times three
// NextSigned() values, so some paths pass the feature sideways; speed relative to the surface 0.5 + 1.5 Next().
PathAim BuildEdgeOrCornerAim(Vec3 half_size, CaseGenerator* generator) {
  const std::array<double, 3> half_sizes = {half_size.x, half_size.y, half_size.z};
  PathAim aim;
  aim.start_elapsed = 0.5 * generator->Next();
  aim.aim_elapsed = 0.25 + 0.5 * generator->Next();
  const bool is_corner = generator->Next() < 0.5;
  const int free_axis = is_corner ? -1 : static_cast<int>(3 * generator->Next());
  const double free_coordinate_share = is_corner ? 0 : generator->NextSigned();
  std::array<double, 3> own_target{};
  std::array<double, 3> own_outward{};
  for (int axis = 0; axis < 3; ++axis) {
    const double sign = generator->Next() < 0.5 ? 1.0 : -1.0;
    const double offset = 2 * kTolerance * generator->NextSigned();  // m
    const bool is_free_axis = axis == free_axis;
    own_target[axis] =
        is_free_axis ? free_coordinate_share * half_sizes[axis] : sign * (half_sizes[axis] + kTolerance) + offset;
    own_outward[axis] = is_free_axis ? 0 : sign;
  }
  aim.own_target = {.x = own_target[0], .y = own_target[1], .z = own_target[2]};
  const double inward_weight = generator->Next();
  aim.own_inward = -inward_weight * Vec3{.x = own_outward[0], .y = own_outward[1], .z = own_outward[2]}
      + 0.8 * Vec3{.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
  aim.speed = 0.5 + 1.5 * generator->Next();
  return aim;
}

// Draws one aim at the rim circle of a cylinder of radius 0.5 m and half height 0.75 m, within 2 tol of its grown
// shape. Draw order: start_elapsed 0.5 Next(); aim_elapsed 0.25 + 0.5 Next(); azimuth 2 pi Next(); the cap, high
// when Next() < 0.5; offsets 2 tol NextSigned() from the grown radius and from the grown half height; an approach
// angle in the plane of the axis 0.05 + (pi / 2 - 0.1) Next(), plus 0.5 times three NextSigned() values; speed
// relative to the surface 0.5 + 1.5 Next().
PathAim BuildRimAim(CaseGenerator* generator) {
  PathAim aim;
  aim.start_elapsed = 0.5 * generator->Next();
  aim.aim_elapsed = 0.25 + 0.5 * generator->Next();
  const double azimuth = 2 * std::numbers::pi * generator->Next();
  const double cap_side_sign = generator->Next() < 0.5 ? 1.0 : -1.0;
  const double radius_offset = 2 * kTolerance * generator->NextSigned();  // m
  const double height_offset = 2 * kTolerance * generator->NextSigned();  // m
  const double angle = 0.05 + (std::numbers::pi / 2 - 0.1) * generator->Next();
  const Vec3 radial = {.x = std::cos(azimuth), .y = std::sin(azimuth), .z = 0};
  aim.own_target = (0.5 + kTolerance + radius_offset) * radial
      + Vec3{.x = 0, .y = 0, .z = cap_side_sign * (0.75 + kTolerance + height_offset)};
  aim.own_inward = -(std::cos(angle) * radial + Vec3{.x = 0, .y = 0, .z = cap_side_sign * std::sin(angle)})
      + 0.5 * Vec3{.x = generator->NextSigned(), .y = generator->NextSigned(), .z = generator->NextSigned()};
  aim.speed = 0.5 + 1.5 * generator->Next();
  return aim;
}

// Returns whether a path starts farther than 2 tol from the solid, at the pose of its start_elapsed: a generated path
// that does not is drawn again, so the search does not stop at once.
bool ComputeIsStartAwayFromSolid(const Solid& solid, Path path) {
  return ComputeDistanceToSolid(solid, ComputeSolidPose(solid, path.start_elapsed), path.start).distance
      > 2 * kTolerance;
}

TEST(FindFirstHit, GeneratedOutwardSweeps_AgreeWithLowerBoundReference) {
  // 3000 OutwardPaths. Draw order per path: start_elapsed 0 when Next() < 0.5, else 0.25 Next(); start_radius
  // 0.5 + 0.5 Next(); outward_speed 1 + 5 Next(); start_angle 0.3 + 1.7 Next(); box_speed 10 NextSigned(). A path that
  // starts within 2 tol of the box is drawn again.
  CaseGenerator generator;
  TurningSweepCounts counts;
  constexpr int kPathCount = 3000;
  for (int path_index = 0; path_index < kPathCount; ++path_index) {
    std::pair<Solid, Path> solid_and_path;
    do {
      OutwardPath outward;
      outward.start_elapsed = generator.Next() < 0.5 ? 0 : 0.25 * generator.Next();
      outward.start_radius = 0.5 + 0.5 * generator.Next();
      outward.outward_speed = 1 + 5 * generator.Next();
      outward.start_angle = 0.3 + 1.7 * generator.Next();
      outward.box_speed = 10 * generator.NextSigned();
      solid_and_path = BuildOutwardPath(outward);
    } while (!ComputeIsStartAwayFromSolid(solid_and_path.first, solid_and_path.second));
    const std::array solids = {solid_and_path.first};
    ApplyTurningSweepCounts(BuildInputs(solids, 1), solid_and_path.second, kLowerBoundReference, &counts);
  }
  // Measured: no close call (first order alone: 118).
  CheckTurningSweepCounts(counts, {.largest_close_call_count = 3}, "outward");
}

TEST(FindFirstHit, GeneratedTurningBoxEdgeAndCornerSweeps_AgreeWithLowerBoundReference) {
  // 4000 paths aimed within 2 tol of an edge or a corner of the grown box of half sizes {0.5, 0.25, 1}, which moves at
  // {0.25, -0.5, 0.125} m/s and turns at {0.48, -0.64, 0.3} rad/s (BuildEdgeOrCornerAim, drawn again until
  // ComputeIsStartAwayFromSolid).
  const Vec3 half_size = {.x = 0.5, .y = 0.25, .z = 1};
  const Solid solid = ToSolid(
      Box{.center = {.x = 0.25, .y = 0.5, .z = -0.125}, .axes = kTiltedAxes, .half_size = half_size},
      {.linear_velocity = {.x = 0.25, .y = -0.5, .z = 0.125}, .angular_velocity = {.x = 0.48, .y = -0.64, .z = 0.3}},
      1
  );
  ASSERT_EQ(CheckSolid(solid, 1.5), ContactError::kNone);
  const std::array solids = {solid};
  const SolidContactInputs inputs = BuildInputs(solids, 1.5);
  CaseGenerator generator;
  TurningSweepCounts counts;
  constexpr int kPathCount = 4000;
  for (int path_index = 0; path_index < kPathCount; ++path_index) {
    Path path;
    do {
      path = BuildPathAimedAtOwnPoint(solid, BuildEdgeOrCornerAim(half_size, &generator));
    } while (!ComputeIsStartAwayFromSolid(solid, path));
    ApplyTurningSweepCounts(inputs, path, kLowerBoundReference, &counts);
  }
  // Measured: no close call (first order alone: 805, mostly paths that pass the edge or the corner sideways within a
  // few tolerances).
  CheckTurningSweepCounts(counts, {.largest_close_call_count = 4}, "box_edges_and_corners");
}

TEST(FindFirstHit, GeneratedTiltingCylinderRimSweeps_AgreeWithLowerBoundReference) {
  // 4000 paths aimed within 2 tol of the grown rim circle of a cylinder of radius 0.5 m and half height 0.75 m along
  // the tilted axis, moving at {0.25, -0.5, 0.125} m/s and turning at {0.7, 0.4, -0.5} rad/s, which tilts its axis
  // (BuildRimAim, drawn again until ComputeIsStartAwayFromSolid).
  const Solid solid = ToSolid(
      Cylinder{
          .center = {.x = 0.25, .y = 0.5, .z = -0.125},
          .axis = ToColumn(kTiltedAxes, 2),
          .radius = 0.5,
          .half_height = 0.75,
      },
      {.linear_velocity = {.x = 0.25, .y = -0.5, .z = 0.125}, .angular_velocity = {.x = 0.7, .y = 0.4, .z = -0.5}},
      1
  );
  ASSERT_EQ(CheckSolid(solid, 1.5), ContactError::kNone);
  const std::array solids = {solid};
  const SolidContactInputs inputs = BuildInputs(solids, 1.5);
  CaseGenerator generator;
  TurningSweepCounts counts;
  constexpr int kPathCount = 4000;
  for (int path_index = 0; path_index < kPathCount; ++path_index) {
    Path path;
    do {
      path = BuildPathAimedAtOwnPoint(solid, BuildRimAim(&generator));
    } while (!ComputeIsStartAwayFromSolid(solid, path));
    ApplyTurningSweepCounts(inputs, path, kLowerBoundReference, &counts);
  }
  // Measured: no close call (first order alone: 291).
  CheckTurningSweepCounts(counts, {.largest_close_call_count = 4}, "cylinder_rims");
}

TEST(FindFirstHit, GeneratedEndFaceSlideSweeps_AgreeWithLowerBoundReference) {
  // 3000 paths that slide past the end face kHighY of the quarter-turn arm (BuildQuarterTurnLongBox), whose normal
  // turns with the arm, so that the Coriolis term and the near end of r_max both count in K. Sweeps of 1 s from the
  // step's start. Draw order per path: a gap exp(log(1e-6) + Next() (log(0.5) - log(1e-6))) m; an own start
  // {0.6 NextSigned(), 4 + tol + gap, 0.4 NextSigned()}, which is also the world start; a relative velocity
  // {w, -share * scale, 0.3 NextSigned()}, with w = 8 Next(), negated when Next() < 1/3, then share = Next() and scale
  // 0, 0.01, 0.1 or 1 by floor(4 Next()). The velocity is the surface velocity at the start plus that.
  const Solid solid = BuildQuarterTurnLongBox({});
  const std::array solids = {solid};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  constexpr std::array<double, 4> kSlowdownScales = {0, 0.01, 0.1, 1};  // m/s
  CaseGenerator generator;
  TurningSweepCounts counts;
  constexpr int kPathCount = 3000;
  for (int path_index = 0; path_index < kPathCount; ++path_index) {
    const double gap = std::exp(std::log(1e-6) + generator.Next() * (std::log(0.5) - std::log(1e-6)));  // m
    const double start_x = 0.6 * generator.NextSigned();  // m
    const double start_z = 0.4 * generator.NextSigned();  // m
    const Vec3 start = {.x = start_x, .y = 4 + kTolerance + gap, .z = start_z};
    const double slide_speed = 8 * generator.Next();  // m/s
    const double slide_sign = generator.Next() < 1.0 / 3 ? -1.0 : 1.0;
    const double slowdown_share = generator.Next();
    const double slowdown_scale = kSlowdownScales.at(static_cast<std::size_t>(4 * generator.Next()));  // m/s
    const Vec3 relative_velocity = {
        .x = slide_sign * slide_speed,
        .y = -slowdown_share * slowdown_scale,
        .z = 0.3 * generator.NextSigned(),
    };
    const Vec3 velocity = ComputeSolidVelocity(solid, ComputeSolidPose(solid, 0), start) + relative_velocity;
    ApplyTurningSweepCounts(
        inputs,
        {.start = start, .velocity = velocity, .duration = 1},
        kLowerBoundReference,
        &counts
    );
  }
  // Measured: no close call; 5.9 advances on average, 18 at the 99th percentile, 55 at most.
  CheckTurningSweepCounts(counts, {.largest_close_call_count = 3}, "end_face_slide");
}

TEST(FindFirstHit, GeneratedInwardSweeps_AgreeWithLowerBoundReference) {
  // 3000 OutwardPaths that move inward, toward the origin of the quarter-turn arm, so that the particle is farthest
  // from the origin at the start: the start end of r_max bounds K. Draw order per path: start_elapsed 0 when
  // Next() < 0.5, else 0.25 Next(); start_radius 1.5 + 2.4 Next(); speed 1 + 5 Next() inward; start_angle
  // 0.2 + 1.8 Next(); box_speed 0. A path that starts within 2 tol of the box, or inside it, is drawn again.
  CaseGenerator generator;
  TurningSweepCounts counts;
  constexpr int kPathCount = 3000;
  for (int path_index = 0; path_index < kPathCount; ++path_index) {
    std::pair<Solid, Path> solid_and_path;
    do {
      OutwardPath inward;
      inward.start_elapsed = generator.Next() < 0.5 ? 0 : 0.25 * generator.Next();
      inward.start_radius = 1.5 + 2.4 * generator.Next();
      inward.outward_speed = -(1 + 5 * generator.Next());
      inward.start_angle = 0.2 + 1.8 * generator.Next();
      inward.box_speed = 0;
      solid_and_path = BuildOutwardPath(inward);
    } while (!ComputeIsStartAwayFromSolid(solid_and_path.first, solid_and_path.second));
    const std::array solids = {solid_and_path.first};
    ApplyTurningSweepCounts(BuildInputs(solids, 1), solid_and_path.second, kLowerBoundReference, &counts);
  }
  // Measured: no close call; 5.5 advances on average, 7 at most.
  CheckTurningSweepCounts(counts, {.largest_close_call_count = 3}, "inward");
}

// The six face-plane distances of a box at one time of a path, in face order: -p.x - half_size.x, p.x - half_size.x,
// and so on, with p the point in the box's own axes. The box's distance is the largest of them.
std::array<double, 6> ComputeFacePlaneDistances(const Solid& box, Path path, double elapsed) {
  const SolidPose pose = ComputeSolidPose(box, path.start_elapsed + elapsed);
  const Vec3 own_point = math::Transpose(pose.axes) * (path.start + path.velocity * elapsed - pose.origin);
  const Vec3 half_size = box.half_size;
  return {
      -own_point.x - half_size.x,
      own_point.x - half_size.x,
      -own_point.y - half_size.y,
      own_point.y - half_size.y,
      -own_point.z - half_size.z,
      own_point.z - half_size.z,
  };
}

// One time of a sweep and the six face-plane distances of the box then.
struct FacePlaneSample {
  double elapsed = 0;  // s, since the start of the sweep.
  std::array<double, 6> distances{};  // m, in face order.
};

// Returns whether the distance from the path to the box surely stays at least tol - tol / 64 between 0 and end_elapsed
// seconds into the sweep, written without the stepping search's bounds: each face-plane distance f(t) is a smooth
// function of time whose second derivative is the own point's acceleration along a fixed own axis, at most
// M = |omega|^2 (r0 + |v_rel| end_elapsed) + 2 |omega| |v_rel| with r0 the start's distance to the origin. On [a, b]
// such an f is at least the smaller of f(a) and f(b) minus M (b - a)^2 / 8, and the box's distance, the largest of
// the six, is at least the largest of those. An interval where that is too small is cut in half until it passes, a
// sample is too close, or it can no longer be cut. The lower-bound reference would need about gap / bound samples
// along a particle that hovers a few tolerances off a face, and would not finish here.
bool ComputeIsOutsideBoxThroughout(const SolidContactInputs& inputs, Path path, double end_elapsed, int* sample_count) {
  const Solid& box = inputs.solids[0];
  const double floor_distance = inputs.contact_tolerance - inputs.contact_tolerance / 64;  // m
  const double relative_speed = math::Length(path.velocity - box.motion.linear_velocity);  // m/s
  const double turn_rate = math::Length(box.motion.angular_velocity);  // rad/s
  const double radius =
      math::Length(path.start - ComputeSolidPose(box, path.start_elapsed).origin) + relative_speed * end_elapsed;  // m
  const double curvature_bound = turn_rate * turn_rate * radius + 2 * turn_rate * relative_speed;  // m/s^2
  const auto largest = [](const std::array<double, 6>& distances) {
    return *std::ranges::max_element(distances);
  };
  std::vector<std::array<FacePlaneSample, 2>> pending_intervals;
  pending_intervals.push_back({
      FacePlaneSample{.elapsed = 0, .distances = ComputeFacePlaneDistances(box, path, 0)},
      FacePlaneSample{.elapsed = end_elapsed, .distances = ComputeFacePlaneDistances(box, path, end_elapsed)},
  });
  *sample_count += 2;
  while (!pending_intervals.empty()) {
    const auto [low, high] = pending_intervals.back();
    pending_intervals.pop_back();
    if (largest(low.distances) < floor_distance || largest(high.distances) < floor_distance) {
      return false;
    }
    const double width = high.elapsed - low.elapsed;  // s
    double lowest_possible_distance = -1;  // m
    for (std::size_t face_index = 0; face_index < 6; ++face_index) {
      const double face_lowest =
          std::min(low.distances.at(face_index), high.distances.at(face_index)) - curvature_bound * width * width / 8;
      lowest_possible_distance = std::max(lowest_possible_distance, face_lowest);
    }
    if (lowest_possible_distance >= floor_distance) {
      continue;
    }
    const double middle_elapsed = low.elapsed + width / 2;  // s
    const bool can_cut = middle_elapsed > low.elapsed && middle_elapsed < high.elapsed;
    if (!can_cut) {
      return false;
    }
    const FacePlaneSample middle = {
        .elapsed = middle_elapsed,
        .distances = ComputeFacePlaneDistances(box, path, middle_elapsed),
    };
    ++*sample_count;
    pending_intervals.push_back({low, middle});
    pending_intervals.push_back({middle, high});
  }
  return true;
}

TEST(FindFirstHit, GeneratedPaddleSweeps_AgreeWithFacePlaneReference) {
  // A paddle: a box of half sizes {0.25, 0.01, 0.1} m at the origin turning at {0, 0, 6} rad/s, with the contact
  // tolerance of a 1 cm cell, 2^-30 * 0.01 m, and a step of 1/120 s swept whole from its start. 2000 paths of each
  // kind, approaching, separating and moving with the face, in turn (path_index % 3), near a large face. Draw order per
  // path: own x 0.24 NextSigned(); own z 0.09 NextSigned(); side +1 when Next() < 0.5, else -1; gap
  // exp(log(0.3 tol) + Next() (log(0.01) - log(0.3 tol))) m; normal speed 0.15 * 10^(-8 Next()) m/s; slide
  // 0.45 NextSigned() m/s along x. The start is {x, side (0.01 + tol + gap), z}; the velocity is the surface velocity
  // there, plus the slide, plus the normal speed along {0, side, 0}: negated when approaching, zero when moving with
  // the face. Reference: ComputeIsOutsideBoxThroughout, allowance tol / 64.
  const Solid paddle = ToSolid(
      Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.25, .y = 0.01, .z = 0.1}},
      {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 6}},
      1
  );
  constexpr double kStepDuration = 1.0 / 120;  // s
  ASSERT_EQ(CheckSolid(paddle, kStepDuration), ContactError::kNone);
  const std::array solids = {paddle};
  SolidContactInputs inputs = BuildInputs(solids, kStepDuration);
  inputs.contact_tolerance = 0x1p-30 * 0.01;
  const double contact_tolerance = inputs.contact_tolerance;
  const SweepReference reference = {
      .is_outside_throughout = ComputeIsOutsideBoxThroughout,
      .allowance = contact_tolerance / 64,
  };
  constexpr std::array<double, 3> kNormalSpeedSigns = {-1, 1, 0};  // approaching, separating, moving with the face
  constexpr std::array<const char*, 3> kKindNames = {
      "paddle_approaching",
      "paddle_separating",
      "paddle_moving_with_face",
  };
  std::array<TurningSweepCounts, 3> counts_per_kind;
  CaseGenerator generator;
  constexpr int kPathCount = 6000;
  for (int path_index = 0; path_index < kPathCount; ++path_index) {
    const auto kind = static_cast<std::size_t>(path_index % 3);
    const double own_x = 0.24 * generator.NextSigned();  // m
    const double own_z = 0.09 * generator.NextSigned();  // m
    const double side = generator.Next() < 0.5 ? 1.0 : -1.0;
    const double gap = std::exp(
        std::log(0.3 * contact_tolerance) + generator.Next() * (std::log(0.01) - std::log(0.3 * contact_tolerance))
    );  // m
    const double normal_speed = 0.15 * std::pow(10.0, -8 * generator.Next());  // m/s
    const double slide_speed = 0.45 * generator.NextSigned();  // m/s
    const Vec3 start = {.x = own_x, .y = side * (0.01 + contact_tolerance + gap), .z = own_z};
    const Vec3 velocity = ComputeSolidVelocity(paddle, ComputeSolidPose(paddle, 0), start)
        + Vec3{.x = slide_speed, .y = 0, .z = 0}
        + kNormalSpeedSigns.at(kind) * normal_speed * Vec3{.x = 0, .y = side, .z = 0};
    ApplyTurningSweepCounts(
        inputs,
        {.start = start, .velocity = velocity, .duration = kStepDuration},
        reference,
        &counts_per_kind.at(kind)
    );
  }
  // Measured: no close call of any kind; 5.5, 6.3 and 7.9 advances on average, 36, 32 and 41 at the 99th percentile,
  // 119 at most. The first-order advance alone gave close calls on most of these paths.
  for (std::size_t kind = 0; kind < 3; ++kind) {
    CheckTurningSweepCounts(
        counts_per_kind.at(kind),
        {
            .largest_close_call_count = 20,
            .largest_mean_advance_count = 12,
            .largest_99th_percentile_advance_count = 80,
            .requires_hits = false,
        },
        kKindNames.at(kind)
    );
  }
}

}  // namespace
}  // namespace fluidity::contact
