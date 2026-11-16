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
#include "fluidity/contact/contact_test_cases.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {
namespace {

constexpr double kTolerance = kTestContactTolerance;  // m, 2^-30.
constexpr SolidMotion kNoMotion = SolidMotion();  // A solid that neither moves nor turns.
constexpr Vec3 kAtRest = {.x = 0, .y = 0, .z = 0};  // m/s.
// A turn whose three columns have no zero entry: {0.36, 0.48, -0.8}, {-0.8, 0.6, 0}, {0.48, 0.64, 0.6}. Each column
// has squared length 1 and each pair a dot of 0 in exact arithmetic; in doubles both hold within a few rounding units.
constexpr Mat3 kTiltedAxes = {
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

// The ground: the flat solid z <= 0, which does not move.
Solid BuildGround(std::uint32_t solid_id) {
  return ToSolid(
      FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
      kNoMotion,
      solid_id
  );
}

// The inputs of one sweep against the solids, with the test contact tolerance and no committed plate side.
SolidContactInputs BuildInputs(std::span<const Solid> solids, double step_duration) {
  SolidContactInputs inputs;
  inputs.solids = solids.data();
  inputs.solid_count = static_cast<std::uint32_t>(solids.size());
  inputs.step_duration = step_duration;
  inputs.contact_tolerance = kTolerance;
  return inputs;
}

// One straight path: start + velocity * t for 0 <= t <= duration, with the solids posed start_elapsed + t into the
// step.
struct Path {
  Vec3 start{};  // m, world.
  Vec3 velocity{};  // m/s, world.
  double start_elapsed = 0;  // s.
  double duration = 0;  // s.
};

// Runs FindFirstHit on one path and fails the calling test unless it succeeds.
FirstHit FindHit(const SolidContactInputs& inputs, Path path) {
  FirstHit first_hit;
  const ContactError error =
      FindFirstHit(inputs, path.start, path.velocity, path.start_elapsed, path.duration, &first_hit);
  EXPECT_EQ(error, ContactError::kNone);
  return first_hit;
}

// Fails the calling test unless |actual - expected| <= 2^-44 * max(|actual|, |expected|, reference): the relative
// tolerance of the contact tests (contact design 7), with reference the largest input of the test (1 for times of
// about a second).
void ExpectNearRelative(double actual, double expected, double reference) {
  const double scale = std::max({std::abs(actual), std::abs(expected), reference});
  EXPECT_LE(std::abs(actual - expected), 0x1p-44 * scale) << actual << " vs " << expected;
}

// Fails the calling test unless the path is free: no solid, and the other fields at their defaults.
void ExpectFree(FirstHit first_hit) {
  const DistanceToSolid defaults{};
  EXPECT_EQ(first_hit.solid_index, kNoSolid);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.distance, defaults.distance);
  EXPECT_EQ(first_hit.at_hit.face, defaults.face);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, defaults.normal_out_of_solid, 0);
}

// Fails the calling test unless a distance reported at a rim hit is where the rim search stops: at least tol and at
// most tol + tol / 1024 (kRimHitGapShare), up to rounding.
void ExpectRimHitDistance(double distance) {
  EXPECT_GE(distance, kTolerance - 0x1p-40);
  EXPECT_LE(distance, kTolerance + kTolerance / 1024 + 0x1p-40);
}

// --- The examples of the method paper ---

TEST(FindFirstHit, NoSolids_IsFree) {
  const SolidContactInputs inputs = BuildInputs({}, 1);
  ExpectFree(FindHit(inputs, {.start = {}, .velocity = {.x = 1, .y = 2, .z = 3}, .start_elapsed = 0, .duration = 1}));
}

TEST(FindFirstHit, FlatSolidAhead_HitsAtGapOverApproach) {
  // Surface x = 1 looking to -x; start x = 0.9 at +1 m/s. The own z coordinate of the start is 1 - 0.9, which rounds
  // to 0.09999999999999998; the gap is that minus tol, and the approach speed 1 m/s divides it exactly.
  const std::array solids = {
      ToSolid(
          FlatSolid{.point_on_surface = {.x = 1, .y = 0, .z = 0}, .normal_out_of_solid = {.x = -1, .y = 0, .z = 0}},
          kNoMotion,
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 0.2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0.9, .y = 0, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 0.2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, (1 - 0.9) - kTolerance);
  ExpectNearRelative(first_hit.elapsed, 0.1 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 2 * kTolerance);
}

TEST(FindFirstHit, BoxFace_ReportsEnteringFace) {
  // kLowX is entered at (0.5 - tol) / 2 s and kHighX left at (1.5 + tol) / 2 s; the y and z faces stay behind the
  // start. All values are exact in doubles, so the hit lands exactly one tolerance from the face.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -1, .y = 0.1, .z = 0.2}, .velocity = {.x = 2, .y = 0, .z = 0}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, (0.5 - kTolerance) / 2);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowX);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, BoxCornerTieAtEndOfStep_ReportsLowX) {
  // The box fills [1, 2] x [1, 2] x [0, 1]. From {-tol, -tol, 0.5} at {1, 1, 0}, the gaps to kLowX and kLowY are both
  // (1 + tol) - tol = 1, so both planes are crossed at exactly 1 s, the end of the sweep: the end counts, and the tie
  // keeps kLowX, first in face order.
  const std::array solids = {
      ToSolid(
          Box{.center = {.x = 1.5, .y = 1.5, .z = 0.5}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
          kNoMotion,
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = -kTolerance, .y = -kTolerance, .z = 0.5}, .velocity = {.x = 1, .y = 1, .z = 0}, .duration = 1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowX);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
}

TEST(FindFirstHit, BoxEdgeGraze_StaysFree) {
  // Same box. From {0, 1 + 4 tol, 0.5} at {1, 1, 0}, the path leaves the region 1 - tol <= y <= 2 + tol between
  // the grown y faces at 1 - 3 tol, before it crosses the grown kLowX plane at 1 - tol: it passes beside the edge.
  const std::array solids = {
      ToSolid(
          Box{.center = {.x = 1.5, .y = 1.5, .z = 0.5}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
          kNoMotion,
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = 0, .y = 1 + 4 * kTolerance, .z = 0.5}, .velocity = {.x = 1, .y = 1, .z = 0}, .duration = 1}
  ));
}

TEST(FindFirstHit, TwoSolidsSameTime_LowerIndexWins) {
  // Two grounds at the same plane, ids 3 and 4: the same hit time, so index 0 is kept.
  const std::array solids = {BuildGround(3), BuildGround(4)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = 0.5}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0.5 - kTolerance);
}

TEST(FindFirstHit, CommittedPlate_UsesOnlyCommittedFace) {
  // Plate through the origin, normal +z, half thickness 0.01; the start is inside it at z = -0.005, moving at
  // +1 m/s. Seen by an uncommitted particle the plate has two faces and the nearer is kLowZ: distance -0.005, a start
  // deeper than tol inside, so a hit at 0 on kLowZ. Committed to kAlongNormal, the plate is the half space below its
  // kHighZ face (plan decision P3): distance -0.005 - 0.01 = -0.015, also a hit at 0, but on kHighZ with normal +z.
  // The brief asked for "free" in the committed case; P3's half space contains the whole thickness, so a committed
  // particle can never be freer than an uncommitted one, and only the face tells the two apart.
  const std::array solids = {
      ToSolid(Plate{.center = {}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.01}, kNoMotion, 5),
  };
  SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {.start = {.x = 0, .y = 0, .z = -0.005}, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 1};

  const FirstHit uncommitted = FindHit(inputs, path);
  EXPECT_EQ(uncommitted.solid_index, 0U);
  EXPECT_EQ(uncommitted.elapsed, 0);
  EXPECT_EQ(uncommitted.at_hit.face, SolidFace::kLowZ);
  ExpectVec3Near(uncommitted.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = -1}, 0);
  EXPECT_EQ(uncommitted.at_hit.distance, 0.005 - 0.01);

  inputs.plate_side = {.plate_id = 5, .side = SolidSide::kAlongNormal};
  const FirstHit committed = FindHit(inputs, path);
  EXPECT_EQ(committed.solid_index, 0U);
  EXPECT_EQ(committed.elapsed, 0);
  EXPECT_EQ(committed.at_hit.face, SolidFace::kHighZ);
  ExpectVec3Near(committed.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = 1}, 0);
  EXPECT_EQ(committed.at_hit.distance, -0.005 - 0.01);
}

TEST(FindFirstHit, DomainBoxOpenTop_ExitIsFree) {
  const std::array solids = {ToSolid(DomainBox{.low_corner = {}, .high_corner = {.x = 1, .y = 1, .z = 1}}, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(
      FindHit(inputs, {.start = {.x = 0.5, .y = 0.5, .z = 0.9}, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 1})
  );
}

TEST(FindFirstHit, MovingSolid_UsesRelativeVelocity) {
  // Plate of half thickness 1/32 centred at x = -1 with normal +x, moving at +2 m/s toward a particle at rest at the
  // origin: gap 1 - 1/32 - tol, closed at 2 m/s. The hit position is the start, since the particle does not move,
  // and the plate has come to exactly one tolerance from it.
  const std::array solids = {
      ToSolid(
          Plate{.center = {.x = -1, .y = 0, .z = 0}, .normal = {.x = 1, .y = 0, .z = 0}, .half_thickness = 0.03125},
          {.linear_velocity = {.x = 2, .y = 0, .z = 0}, .angular_velocity = {}},
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(inputs, {.start = {}, .velocity = kAtRest, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, (1 - 0.03125 - kTolerance) / 2);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = 1, .y = 0, .z = 0}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, StartElapsed_SamplesSolidLater) {
  // Surface at x = 2 looking to -x, moving at -1 m/s; a particle at rest at the origin. Half a second into the step
  // the surface is at x = 1.5, so the gap is 1.5 - tol; from the start of the step it would be 2 - tol.
  const std::array solids = {
      ToSolid(
          FlatSolid{.point_on_surface = {.x = 2, .y = 0, .z = 0}, .normal_out_of_solid = {.x = -1, .y = 0, .z = 0}},
          {.linear_velocity = {.x = -1, .y = 0, .z = 0}, .angular_velocity = {}},
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(inputs, {.start = {}, .velocity = kAtRest, .start_elapsed = 0.5, .duration = 1.5});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 1.5 - kTolerance);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, StartElapsed_TouchingRuleUsesLaterPose) {
  // Ground rising at 1 m/s; half a second into the step its surface is at z = 0.5, so a particle at rest at
  // z = 0.5 + tol rests on it (gap 0) and the ground pushes into it at 1 m/s: hit at 0. Posed at the start of the
  // step, the gap would be 0.5 - tol and the ground would move through the particle.
  const std::array solids = {
      ToSolid(
          FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
          {.linear_velocity = {.x = 0, .y = 0, .z = 1}, .angular_velocity = {}},
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path =
      {.start = {.x = 0, .y = 0, .z = 0.5 + kTolerance}, .velocity = kAtRest, .start_elapsed = 0.5, .duration = 0.5};
  const FirstHit first_hit = FindHit(inputs, path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, StartElapsed_TouchingRuleRisingWithGroundIsFree) {
  // The same start, now rising with the ground and sliding sideways: measured from the ground it only slides.
  const std::array solids = {
      ToSolid(
          FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
          {.linear_velocity = {.x = 0, .y = 0, .z = 1}, .angular_velocity = {}},
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(
      inputs,
      {
          .start = {.x = 0, .y = 0, .z = 0.5 + kTolerance},
          .velocity = {.x = 1, .y = 0, .z = 1},
          .start_elapsed = 0.5,
          .duration = 0.5,
      }
  ));
}

TEST(FindFirstHit, StartElapsed_PlateTouchingRuleUsesLaterPose) {
  // A plate of half thickness 0.25 rising at 1 m/s: at 0.5 s its kHighZ face is at z = 0.75, and a particle at rest at
  // z = 0.75 + tol rests on it: hit at 0, uncommitted and committed to kAlongNormal alike. Posed at the start of the
  // step it would be 0.5 above the face and free.
  const std::array solids = {
      ToSolid(
          Plate{.center = {}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.25},
          {.linear_velocity = {.x = 0, .y = 0, .z = 1}, .angular_velocity = {}},
          5
      ),
  };
  SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path =
      {.start = {.x = 0, .y = 0, .z = 0.75 + kTolerance}, .velocity = kAtRest, .start_elapsed = 0.5, .duration = 0.5};
  const FirstHit uncommitted = FindHit(inputs, path);
  EXPECT_EQ(uncommitted.solid_index, 0U);
  EXPECT_EQ(uncommitted.elapsed, 0);
  EXPECT_EQ(uncommitted.at_hit.face, SolidFace::kHighZ);
  inputs.plate_side = {.plate_id = 5, .side = SolidSide::kAlongNormal};
  const FirstHit committed = FindHit(inputs, path);
  EXPECT_EQ(committed.solid_index, 0U);
  EXPECT_EQ(committed.elapsed, 0);
  EXPECT_EQ(committed.at_hit.face, SolidFace::kHighZ);
}

TEST(FindFirstHit, StartElapsed_BoxTouchingRuleUsesLaterPose) {
  // A box of half size 0.5 rising at 1 m/s: at 0.5 s its top face is at z = 1, and a particle at rest at z = 1 + tol
  // rests on it: hit at 0 on kHighZ.
  const std::array solids = {
      ToSolid(
          Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
          {.linear_velocity = {.x = 0, .y = 0, .z = 1}, .angular_velocity = {}},
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0, .y = 0, .z = 1 + kTolerance}, .velocity = kAtRest, .start_elapsed = 0.5, .duration = 0.5}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, StartOnExitPlane_LeavesBeforeEntering) {
  // Box half 0.5 at the origin; from {-1, 0.5 + tol, 0} the kHighY gap is exactly 0, and the path at {1, 1, 0} leaves
  // through that plane at once, before it crosses the kLowX plane at 0.5 - tol: free.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = -1, .y = 0.5 + kTolerance, .z = 0}, .velocity = {.x = 1, .y = 1, .z = 0}, .duration = 1}
  ));
}

TEST(FindFirstHit, SlidingOnTouchedFace_IsNotAHit) {
  // Resting at z = tol (gap 0) and sinking at tol / 4 m/s for 1 s: the path goes at most tol / 4 below the start, so
  // it cannot cross the surface. The touching rule frees it.
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = 0, .y = 0, .z = kTolerance}, .velocity = {.x = 5, .y = 0, .z = -kTolerance / 4}, .duration = 1}
  ));
}

TEST(FindFirstHit, PushingIntoTouchedFace_HitsAtZero) {
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0, .y = 0, .z = kTolerance}, .velocity = {.x = 5, .y = 0, .z = -1}, .duration = 1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = 1}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

// --- Every face of every shape ---

// One face of a solid and its normal out of the solid, in world axes.
struct FaceCase {
  SolidFace face = SolidFace::kHighZ;
  Vec3 normal_out_of_solid{};  // No unit, unit vector, world: the normal out of the solid on that face.
  Vec3 sideways{};  // No unit, unit vector, world, along the face: offsets the start from the face's centre line.
  double half_size_along = 0;  // m. A box's half size along the face's normal; 0 where it does not apply.
};

Vec3 ToColumn(Mat3 axes, int column) {
  if (column == 0) {
    return {.x = axes.xx, .y = axes.yx, .z = axes.zx};
  }
  if (column == 1) {
    return {.x = axes.xy, .y = axes.yy, .z = axes.zy};
  }
  return {.x = axes.xz, .y = axes.yz, .z = axes.zz};
}

// The six faces of a box with these axes and half sizes, in face order, each with a sideways direction along the next
// own axis.
std::array<FaceCase, 6> BuildBoxFaceCases(Mat3 axes, Vec3 half_size) {
  const Vec3 column_x = ToColumn(axes, 0);
  const Vec3 column_y = ToColumn(axes, 1);
  const Vec3 column_z = ToColumn(axes, 2);
  return {
      {
          {
              .face = SolidFace::kLowX,
              .normal_out_of_solid = -column_x,
              .sideways = column_y,
              .half_size_along = half_size.x,
          },
          {
              .face = SolidFace::kHighX,
              .normal_out_of_solid = column_x,
              .sideways = column_y,
              .half_size_along = half_size.x,
          },
          {
              .face = SolidFace::kLowY,
              .normal_out_of_solid = -column_y,
              .sideways = column_z,
              .half_size_along = half_size.y,
          },
          {
              .face = SolidFace::kHighY,
              .normal_out_of_solid = column_y,
              .sideways = column_z,
              .half_size_along = half_size.y,
          },
          {
              .face = SolidFace::kLowZ,
              .normal_out_of_solid = -column_z,
              .sideways = column_x,
              .half_size_along = half_size.z,
          },
          {
              .face = SolidFace::kHighZ,
              .normal_out_of_solid = column_z,
              .sideways = column_x,
              .half_size_along = half_size.z,
          },
      },
  };
}

TEST(FindFirstHit, MovingBoxEachFace_HitsThatFace) {
  // A moving box with world axes, half sizes {0.5, 0.25, 0.125}. Each start is 1 m in front of one face, 1/16 m to
  // the side, and approaches it at 2 m/s measured from the box: hit at (1 - tol) / 2. Every value is a sum of powers
  // of two, so the times are exact.
  const Vec3 half_size = {.x = 0.5, .y = 0.25, .z = 0.125};
  const Vec3 center = {.x = 1, .y = 2, .z = 3};
  const SolidMotion motion = {.linear_velocity = {.x = 0.5, .y = -0.25, .z = 1}, .angular_velocity = {}};
  const std::array solids = {ToSolid(Box{.center = center, .axes = kIdentity, .half_size = half_size}, motion, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  for (const FaceCase& face_case : BuildBoxFaceCases(kIdentity, half_size)) {
    const Vec3 start =
        center + (face_case.half_size_along + 1) * face_case.normal_out_of_solid + 0.0625 * face_case.sideways;
    const Vec3 velocity = motion.linear_velocity - 2 * face_case.normal_out_of_solid;
    const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = velocity, .duration = 1});
    EXPECT_EQ(first_hit.solid_index, 0U) << static_cast<int>(face_case.face);
    EXPECT_EQ(first_hit.elapsed, (1 - kTolerance) / 2) << static_cast<int>(face_case.face);
    EXPECT_EQ(first_hit.at_hit.face, face_case.face);
    ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, face_case.normal_out_of_solid, 0);
    EXPECT_EQ(first_hit.at_hit.distance, kTolerance) << static_cast<int>(face_case.face);
  }
}

TEST(FindFirstHit, TiltedMovingBoxEachFace_HitsThatFace) {
  // The same box turned by kTiltedAxes, so no face is aligned with a world axis. The turn rounds the start's own
  // coordinates by a few units of 2^-52, so the times agree with (1 - tol) / 2 to the relative tolerance.
  const Vec3 half_size = {.x = 0.5, .y = 0.25, .z = 0.125};
  const Vec3 center = {.x = 1, .y = 2, .z = 3};
  const SolidMotion motion = {.linear_velocity = {.x = 0.5, .y = -0.25, .z = 1}, .angular_velocity = {}};
  const std::array solids = {ToSolid(Box{.center = center, .axes = kTiltedAxes, .half_size = half_size}, motion, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ASSERT_EQ(CheckSolid(solids[0], 1), ContactError::kNone);
  for (const FaceCase& face_case : BuildBoxFaceCases(kTiltedAxes, half_size)) {
    const Vec3 start =
        center + (face_case.half_size_along + 1) * face_case.normal_out_of_solid + 0.0625 * face_case.sideways;
    const Vec3 velocity = motion.linear_velocity - 2 * face_case.normal_out_of_solid;
    const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = velocity, .duration = 1});
    EXPECT_EQ(first_hit.solid_index, 0U) << static_cast<int>(face_case.face);
    ExpectNearRelative(first_hit.elapsed, (1 - kTolerance) / 2, 1);
    EXPECT_EQ(first_hit.at_hit.face, face_case.face);
    ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, face_case.normal_out_of_solid, 0);
    EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 2 * kTolerance);
  }
}

TEST(FindFirstHit, TiltedMovingPlateBothFaces_HitsThatFace) {
  // A plate whose normal is the tilted z column, moving. From 1 m in front of either face, approaching at 2 m/s:
  // hit at (1 - tol) / 2 on that face. A plate is unbounded sideways, so the start may be far to the side.
  const Vec3 normal = ToColumn(kTiltedAxes, 2);
  const Vec3 center = {.x = 0, .y = 0, .z = 1};
  const SolidMotion motion = {.linear_velocity = {.x = 0.25, .y = 0, .z = -0.5}, .angular_velocity = {}};
  const std::array solids = {ToSolid(Plate{.center = center, .normal = normal, .half_thickness = 0.25}, motion, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const std::array face_cases = {
      FaceCase{.face = SolidFace::kLowZ, .normal_out_of_solid = -normal, .sideways = ToColumn(kTiltedAxes, 1)},
      FaceCase{.face = SolidFace::kHighZ, .normal_out_of_solid = normal, .sideways = ToColumn(kTiltedAxes, 0)},
  };
  for (const FaceCase& face_case : face_cases) {
    const Vec3 start = center + 1.25 * face_case.normal_out_of_solid + 3 * face_case.sideways;
    const Vec3 velocity = motion.linear_velocity - 2 * face_case.normal_out_of_solid;
    const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = velocity, .duration = 1});
    EXPECT_EQ(first_hit.solid_index, 0U) << static_cast<int>(face_case.face);
    ExpectNearRelative(first_hit.elapsed, (1 - kTolerance) / 2, 1);
    EXPECT_EQ(first_hit.at_hit.face, face_case.face);
    ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, face_case.normal_out_of_solid, 1e-15);
    EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 2 * kTolerance);
  }
}

TEST(FindFirstHit, TiltedMovingFlatSolid_HitsItsFace) {
  const Vec3 normal = ToColumn(kTiltedAxes, 2);
  const Vec3 point = {.x = 1, .y = 1, .z = 1};
  const SolidMotion motion = {.linear_velocity = {.x = -0.5, .y = 0.25, .z = 0}, .angular_velocity = {}};
  const std::array solids = {
      ToSolid(FlatSolid{.point_on_surface = point, .normal_out_of_solid = normal}, motion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = point + normal + 2 * ToColumn(kTiltedAxes, 0);
  const FirstHit first_hit =
      FindHit(inputs, {.start = start, .velocity = motion.linear_velocity - 2 * normal, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, (1 - kTolerance) / 2, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, normal, 1e-15);
}

TEST(FindFirstHit, DomainBoxEachSide_HitsThatSide) {
  // Every side closed. From the centre of [0, 1]^3, 1/16 m to the side, toward each side at 2 m/s: gap 0.5 - tol,
  // hit at (0.5 - tol) / 2. The normal points into the box, against the motion.
  DomainBox domain_box = {.low_corner = {}, .high_corner = {.x = 1, .y = 1, .z = 1}};
  domain_box.closed.top = true;
  const std::array solids = {ToSolid(domain_box, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 center = {.x = 0.5, .y = 0.5, .z = 0.5};
  for (const FaceCase& face_case : BuildBoxFaceCases(kIdentity, {.x = 0.5, .y = 0.5, .z = 0.5})) {
    const Vec3 start = center + 0.0625 * face_case.sideways;
    const FirstHit first_hit =
        FindHit(inputs, {.start = start, .velocity = 2 * face_case.normal_out_of_solid, .duration = 1});
    EXPECT_EQ(first_hit.solid_index, 0U) << static_cast<int>(face_case.face);
    EXPECT_EQ(first_hit.elapsed, (0.5 - kTolerance) / 2) << static_cast<int>(face_case.face);
    EXPECT_EQ(first_hit.at_hit.face, face_case.face);
    ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, -face_case.normal_out_of_solid, 0);
    EXPECT_EQ(first_hit.at_hit.distance, kTolerance) << static_cast<int>(face_case.face);
  }
}

TEST(FindFirstHit, DomainBoxCorner_RestingOnTwoSides_IsFree) {
  // Resting one tolerance from side_left and side_front, sinking toward both at tol / 4 m/s and rising: each closed
  // side is its own flat face with its own touching rule, so neither is hit.
  const std::array solids = {ToSolid(DomainBox{.low_corner = {}, .high_corner = {.x = 1, .y = 1, .z = 1}}, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 velocity = {.x = -kTolerance / 4, .y = -kTolerance / 4, .z = 0.25};
  ExpectFree(
      FindHit(inputs, {.start = {.x = kTolerance, .y = kTolerance, .z = 0.5}, .velocity = velocity, .duration = 1})
  );
}

TEST(FindFirstHit, DomainBoxCorner_PushingIntoSecondSide_HitsIt) {
  // Resting on side_left and side_front; the path slides along side_left and pushes into side_front: hit at 0 on
  // side_front (kLowY), not on side_left, which comes first in face order but is not pushed into.
  const std::array solids = {ToSolid(DomainBox{.low_corner = {}, .high_corner = {.x = 1, .y = 1, .z = 1}}, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = kTolerance, .y = kTolerance, .z = 0.5}, .velocity = {.x = 0, .y = -1, .z = 0.25}, .duration = 1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowY);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 1, .z = 0}, 0);
}

TEST(FindFirstHit, DomainBoxCornerTie_ReportsLeftSide) {
  // From the centre of [0, 1]^3 toward the corner at {-1, -1, 0} m/s: side_left and side_front are both reached at
  // 0.5 - tol, and the tie keeps side_left (kLowX), first in face order.
  const std::array solids = {ToSolid(DomainBox{.low_corner = {}, .high_corner = {.x = 1, .y = 1, .z = 1}}, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0.5, .y = 0.5, .z = 0.5}, .velocity = {.x = -1, .y = -1, .z = 0}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0.5 - kTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowX);
}

// --- The start: at, inside and outside the tolerance ---

TEST(FindFirstHit, StartOneToleranceOutsideRest_HitsAfterGapOverApproach) {
  // At z = 2 tol the gap is tol > 0: the start is outside the grown ground, so the closed form gives tol / 1 s, not 0.
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0, .y = 0, .z = 2 * kTolerance}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, kTolerance);
}

TEST(FindFirstHit, RisingGroundUnderRestingParticle_HitsAtZero) {
  // A particle at rest one tolerance above a ground that rises at 1 m/s: measured from the ground it pushes in at
  // 1 m/s, so the touching rule gives a hit at 0. Measured in the world it would not move and be free.
  const std::array solids = {
      ToSolid(
          FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
          {.linear_velocity = {.x = 0, .y = 0, .z = 1}, .angular_velocity = {}},
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = kTolerance}, .velocity = kAtRest, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
}

TEST(FindFirstHit, StartJustInsideRest_SlidingIsFree) {
  // At z = tol / 2 the gap is -tol / 2, inside the touching window [-tol, 0].
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(
      inputs,
      {
          .start = {.x = 0, .y = 0, .z = kTolerance / 2},
          .velocity = {.x = 5, .y = 0, .z = -kTolerance / 4},
          .duration = 1,
      }
  ));
}

TEST(FindFirstHit, StartJustInsideRest_PushingInHitsAtZero) {
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0, .y = 0, .z = kTolerance / 2}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance / 2);
}

TEST(FindFirstHit, StartOnSurface_ParallelPathIsFree) {
  // At z = 0 the gap is exactly -tol, the lower end of the touching window: a path along the surface is free.
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(inputs, {.start = {}, .velocity = {.x = 3, .y = 4, .z = 0}, .duration = 1}));
}

TEST(FindFirstHit, StartJustBelowSurfaceBeyondWindow_HitsAtZero) {
  // At z = -2^-50 the gap is -tol - 2^-50, below the touching window: hit at 0 even on a path along the surface.
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = -0x1p-50}, .velocity = {.x = 3, .y = 4, .z = 0}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.distance, -0x1p-50);
}

TEST(FindFirstHit, TouchingPushExactlyOneTolerance_IsFree) {
  // rate * duration == -tol exactly: the path ends exactly on the surface, still not inside. The limit is free.
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = 0, .y = 0, .z = kTolerance}, .velocity = {.x = 0, .y = 0, .z = -kTolerance}, .duration = 1}
  ));
}

TEST(FindFirstHit, TouchingPushJustOverOneTolerance_HitsAtZero) {
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 velocity = {.x = 0, .y = 0, .z = -kTolerance * (1 + 0x1p-20)};
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = kTolerance}, .velocity = velocity, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
}

TEST(FindFirstHit, BoxStartAtCentre_HitsAtZeroOnLowX) {
  // Deep inside: hit at 0, with the distance of ComputeDistanceToSolid, -0.5 on kLowX (all three axes tie).
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(inputs, {.start = {}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowX);
  EXPECT_EQ(first_hit.at_hit.distance, -0.5);
}

TEST(FindFirstHit, BoxRestingOnTop_SlidingOffTheEdgeIsFree) {
  // Resting one tolerance above the top face, moving sideways past the edge: the box lies behind the top face's
  // plane, which the path never crosses.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = 0, .y = 0, .z = 0.5 + kTolerance}, .velocity = {.x = 2, .y = 0, .z = 0}, .duration = 1}
  ));
}

TEST(FindFirstHit, BoxRestingOnEdge_UsesTheFaceThatGivesTheDistance) {
  // One tolerance from both kHighX and kHighZ: their gaps tie at 0 and the tie keeps kHighX. Sinking along the x
  // face is free; pushing into it is a hit at 0 on kHighX.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = 0.5 + kTolerance, .y = 0, .z = 0.5 + kTolerance};
  ExpectFree(FindHit(inputs, {.start = start, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1}));
  const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = {.x = -1, .y = 0, .z = 0}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
}

// --- The end of the sweep, paths along a face, and edges ---

TEST(FindFirstHit, HitExactlyAtEnd_IsHit) {
  // From z = 1 + tol at -1 m/s the gap 1 closes at exactly 1 s, the duration.
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0, .y = 0, .z = 1 + kTolerance}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 1);
}

TEST(FindFirstHit, HitJustAfterEnd_IsFree) {
  // The same path with the duration one rounding unit short of 1 s.
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(
      inputs,
      {
          .start = {.x = 0, .y = 0, .z = 1 + kTolerance},
          .velocity = {.x = 0, .y = 0, .z = -1},
          .duration = 0x1.fffffffffffffp-1,
      }
  ));
}

TEST(FindFirstHit, PathParallelToFlatSolid_IsFree) {
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 10);
  ExpectFree(
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = 1}, .velocity = {.x = 3, .y = 4, .z = 0}, .duration = 10})
  );
}

TEST(FindFirstHit, PathParallelAboveBoxFace_IsFree) {
  // Above the top face (gap 0.25 - tol > 0) and moving along it: that face is never reached, although the path
  // crosses the x range of the box.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  ExpectFree(
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0.75}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 4})
  );
}

TEST(FindFirstHit, PathParallelBetweenBoxFaces_HitsTheFaceAhead) {
  // Between the z faces and moving along them: they neither enter nor leave, and kLowX is entered at 1.5 - tol.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0.25}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 4});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 1.5 - kTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowX);
}

TEST(FindFirstHit, BoxEdgeMetExactly_IsHit) {
  // Box half 0.5 at the origin; from {-1, 2 tol, 0} at {1, 1, 0}. kLowX is entered at 1 - 0.5 - tol = 0.5 - tol, and
  // kHighY is left at (0.5 + tol - 2 tol) / 1 = 0.5 - tol: entry equals exit, the path meets the grown edge
  // exactly, and that counts as a hit on kLowX.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = -1, .y = 2 * kTolerance, .z = 0}, .velocity = {.x = 1, .y = 1, .z = 0}, .duration = 1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0.5 - kTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowX);
}

TEST(FindFirstHit, BoxEdgeMissedByOneUnit_IsFree) {
  // The same path from y = 2 tol + 2^-52: kHighY is left one rounding unit before kLowX is entered.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = -1, .y = 2 * kTolerance + 0x1p-52, .z = 0};
  ExpectFree(FindHit(inputs, {.start = start, .velocity = {.x = 1, .y = 1, .z = 0}, .duration = 1}));
}

// --- Several solids, the committed plate, and the outputs ---

TEST(FindFirstHit, LaterSolidHitFirst_Wins) {
  // A flat solid at x = 2 (index 0) is reached at 1.5 - tol, the ground (index 1) at 0.5 - tol: the earlier hit wins.
  const std::array solids = {
      ToSolid(
          FlatSolid{.point_on_surface = {.x = 2, .y = 0, .z = 0}, .normal_out_of_solid = {.x = -1, .y = 0, .z = 0}},
          kNoMotion,
          1
      ),
      BuildGround(2),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0.5, .y = 0, .z = 0.5}, .velocity = {.x = 1, .y = 0, .z = -1}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 1U);
  EXPECT_EQ(first_hit.elapsed, 0.5 - kTolerance);
}

TEST(FindFirstHit, CommittedToOtherPlate_SeesBothFaces) {
  // Committed to plate 9 (index 1); plate 7 (index 0) is seen with both faces, so a start below it reaches kLowZ.
  const std::array solids = {
      ToSolid(Plate{.center = {}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.25}, kNoMotion, 7),
      ToSolid(
          Plate{.center = {.x = 0, .y = 0, .z = 10}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.25},
          kNoMotion,
          9
      ),
  };
  SolidContactInputs inputs = BuildInputs(solids, 1);
  inputs.plate_side = {.plate_id = 9, .side = SolidSide::kAgainstNormal};
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = -1.25}, .velocity = {.x = 0, .y = 0, .z = 2}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, (1 - kTolerance) / 2);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowZ);
}

TEST(FindFirstHit, CommittedAgainstNormal_HitsLowFaceFromBelow) {
  // Committed to kAgainstNormal, the plate is the half space above its kLowZ face: from z = -1.25 at 2 m/s the hit
  // is on kLowZ at (1 - tol) / 2, normal -z.
  const std::array solids = {
      ToSolid(Plate{.center = {}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.25}, kNoMotion, 7),
  };
  SolidContactInputs inputs = BuildInputs(solids, 1);
  inputs.plate_side = {.plate_id = 7, .side = SolidSide::kAgainstNormal};
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = -1.25}, .velocity = {.x = 0, .y = 0, .z = 2}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, (1 - kTolerance) / 2);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowZ);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = -1}, 0);
}

TEST(FindFirstHit, Success_WritesNoCloseCall) {
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  FirstHit first_hit;
  ContactCloseCall close_call = ContactCloseCall::kTooManyHits;
  const ContactError error =
      FindFirstHit(inputs, {.x = 0, .y = 0, .z = 0.5}, {.x = 0, .y = 0, .z = -1}, 0, 1, &first_hit, &close_call);
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_EQ(close_call, ContactCloseCall::kNone);
  EXPECT_EQ(first_hit.solid_index, 0U);
}

TEST(FindFirstHit, CloseCall_NamesTurningSolid) {
  // A particle at rest 2^-17 m above the top face of a box turning at 1 rad/s, on its turn axis: the gap never
  // changes, but each advance of the stepping search is only about Sqrt(2 g / K), so the search runs out of advances
  // before the end of the sweep and the step is too close to call. The first hit then names the box, index 1 after a
  // fixed flat solid far below, and holds nothing else.
  const std::array solids = {
      ToSolid(
          FlatSolid{.point_on_surface = {.x = 0, .y = 0, .z = -5}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
          kNoMotion,
          1
      ),
      ToSolid(
          Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}},
          2
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  FirstHit first_hit = {.elapsed = 7, .solid_index = 3, .at_hit = {.distance = 7, .normal_out_of_solid = {}}};
  ContactCloseCall close_call = ContactCloseCall::kNone;
  const ContactError error =
      FindFirstHit(inputs, {.x = 0, .y = 0, .z = 0.5 + 0x1p-17}, kAtRest, 0, 1, &first_hit, &close_call);
  EXPECT_EQ(error, ContactError::kTooCloseToCall);
  EXPECT_EQ(close_call, ContactCloseCall::kSlowApproachToTurningSolid);
  EXPECT_EQ(first_hit.solid_index, 1U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.distance, 0);
}

// --- A sphere ---

// A sphere of radius 0.5 m at center, with id 1.
Solid BuildSphere(Vec3 center, SolidMotion motion) {
  return ToSolid(Sphere{.center = center, .radius = 0.5}, motion, 1);
}

// A cylinder of radius 0.5 m and half height 1 m at center along axis, with id 1.
Solid BuildCylinder(Vec3 center, Vec3 axis, SolidMotion motion) {
  return ToSolid(Cylinder{.center = center, .axis = axis, .radius = 0.5, .half_height = 1}, motion, 1);
}

constexpr Vec3 kAxisZ = {.x = 0, .y = 0, .z = 1};  // No unit, the world z axis.

TEST(FindFirstHit, SphereHeadOn_HitsAtGapOverSpeed) {
  // From 2 m in front of the centre at 1 m/s, the grown sphere of radius 0.5 + tol is reached at 1.5 - tol.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

TEST(FindFirstHit, SphereGrazing_HitsAtQuadraticRoot) {
  // 0.3 m off the centre line: x meets -Sqrt((0.5 + tol)^2 - 0.09), near -0.4, so the normal is near {-0.8, 0.6, 0}.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -2, .y = 0.3, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 2 - std::sqrt((0.5 + kTolerance) * (0.5 + kTolerance) - 0.09), 1);
  ExpectNearRelative(first_hit.elapsed, 1.5999999988358469, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -0.8, .y = 0.6, .z = 0}, 1e-8);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

TEST(FindFirstHit, SphereTangentWithinRounding_FreeOrHitNearTwo) {
  // The path touches the grown sphere at one point, {0, 0.5 + tol, 0} at 2 s, up to rounding. Free and a hit there
  // are both safe; an error or a hit far from that point is not.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = -2, .y = 0.5 + kTolerance, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2}
  );
  if (first_hit.solid_index == kNoSolid) {
    ExpectFree(first_hit);
    return;
  }
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 2, 0x1p-20);
  EXPECT_NEAR(first_hit.at_hit.normal_out_of_solid.x, 0, 0x1p-20);
}

TEST(FindFirstHit, SphereMiss_IsFree) {
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  ExpectFree(
      FindHit(inputs, {.start = {.x = -2, .y = 0.6, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2})
  );
}

TEST(FindFirstHit, SphereMovingAway_IsFree) {
  // Dot(start, velocity) = 2 >= 0: the path does not approach the centre.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  ExpectFree(
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0}, .velocity = {.x = -1, .y = 0, .z = 0}, .duration = 2})
  );
}

TEST(FindFirstHit, SphereTouchingTangentPath_IsNotAHit) {
  // At distance tol (gap 0) and moving along the tangent plane: rate 0, so the touching rule frees it (plan P16).
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = -(0.5 + kTolerance), .y = 0, .z = 0}, .velocity = {.x = 0, .y = 1, .z = 0}, .duration = 2}
  ));
}

TEST(FindFirstHit, SphereTouchingPushingIn_HitsAtZero) {
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = -(0.5 + kTolerance), .y = 0, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, MovingSphere_UsesRelativeVelocity) {
  // The centre starts at {2, 0, 0} and moves at {-1, 0, 0} m/s toward a particle at rest at the origin. The hit
  // position is the start, and at the pose of the hit the sphere is one tolerance from it.
  const std::array solids = {
      BuildSphere({.x = 2, .y = 0, .z = 0}, {.linear_velocity = {.x = -1, .y = 0, .z = 0}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(inputs, {.start = {}, .velocity = kAtRest, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

TEST(FindFirstHit, SpinningSphere_IsClosedForm) {
  // A spin about the centre does not change the shape (plan P17): the head-on time of the sphere at rest, at the
  // start of the step and a quarter of a second into it, when the spin has turned the sphere's axes by 1.25 rad.
  const std::array solids = {
      BuildSphere({}, {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 5}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 2.25);
  for (const double start_elapsed : {0.0, 0.25}) {
    const Path path = {
        .start = {.x = -2, .y = 0, .z = 0},
        .velocity = {.x = 1, .y = 0, .z = 0},
        .start_elapsed = start_elapsed,
        .duration = 2,
    };
    const FirstHit first_hit = FindHit(inputs, path);
    EXPECT_EQ(first_hit.solid_index, 0U) << start_elapsed;
    ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
    EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
    ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
  }
}

TEST(FindFirstHit, MovingSphereEachDirection_HitsItsSurface) {
  // A moving sphere at {1, 2, 3}, a quarter of a second into the step. Each start is 2 m from the centre along one
  // direction and approaches at 1 m/s measured from the sphere: hit at 1.5 - tol, with that direction as normal.
  const SolidMotion motion = {.linear_velocity = {.x = 0.5, .y = -0.25, .z = 1}, .angular_velocity = {}};
  const std::array solids = {BuildSphere({.x = 1, .y = 2, .z = 3}, motion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2.25);
  const Vec3 center_at_start = Vec3{.x = 1, .y = 2, .z = 3} + 0.25 * motion.linear_velocity;
  const std::array directions = {
      Vec3{.x = -1, .y = 0, .z = 0},
      Vec3{.x = 1, .y = 0, .z = 0},
      Vec3{.x = 0, .y = -1, .z = 0},
      Vec3{.x = 0, .y = 1, .z = 0},
      Vec3{.x = 0, .y = 0, .z = -1},
      Vec3{.x = 0, .y = 0, .z = 1},
      Vec3{.x = 0.6, .y = 0, .z = 0.8},
      Vec3{.x = -0.48, .y = 0.6, .z = -0.64},
  };
  for (const Vec3 direction : directions) {
    const Path path = {
        .start = center_at_start + 2 * direction,
        .velocity = motion.linear_velocity - direction,
        .start_elapsed = 0.25,
        .duration = 2,
    };
    const FirstHit first_hit = FindHit(inputs, path);
    EXPECT_EQ(first_hit.solid_index, 0U);
    ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
    EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
    ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, direction, 1e-8);
    EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
  }
}

TEST(FindFirstHit, SphereStartOneToleranceOutsideRest_HitsAfterGapOverSpeed) {
  // At distance 2 tol the gap is tol > 0: the closed form decides, and the path at 0.7 m/s closes it in tol / 0.7 s.
  // The textbook root (-B - Sqrt(B^2 - A C)) / A would subtract 0.35 + 1.4 tol and about 0.35 + 0.7 tol and keep
  // only some 8 of the 16 digits; the form of the method paper keeps them all.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = -(0.5 + 2 * kTolerance), .y = 0, .z = 0}, .velocity = {.x = 0.7, .y = 0, .z = 0}, .duration = 1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, kTolerance / 0.7, kTolerance);
}

TEST(FindFirstHit, SphereStartJustInsideRest_PushingInHitsAtZero) {
  // At distance tol / 2 the gap is -tol / 2, inside the touching window.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = 0, .y = 0, .z = 0.5 + kTolerance / 2};
  ExpectFree(FindHit(inputs, {.start = start, .velocity = {.x = 1, .y = 0, .z = -kTolerance / 4}, .duration = 1}));
  const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance / 2);
}

TEST(FindFirstHit, SphereStartAtCentre_HitsAtZeroWithDegenerateNormal) {
  // Deeper than one tolerance inside: hit at 0 whatever the path, with the distance at the start, -0.5, and the
  // degenerate normal, minus the sphere's own x axis.
  const std::array solids = {BuildSphere({.x = 1, .y = 2, .z = 3}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 1, .y = 2, .z = 3}, .velocity = {.x = 0, .y = 0, .z = 0}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, -0.5);
}

TEST(FindFirstHit, SphereHitExactlyAtEnd_IsHit) {
  // The duration set to the hit time the same path finds with a longer duration: the end counts.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const Path long_path = {.start = {.x = -2, .y = 0.3, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2};
  const double hit_elapsed = FindHit(inputs, long_path).elapsed;
  Path path_ending_at_hit = long_path;
  path_ending_at_hit.duration = hit_elapsed;
  const FirstHit first_hit = FindHit(inputs, path_ending_at_hit);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, hit_elapsed);
  Path path_ending_before_hit = long_path;
  path_ending_before_hit.duration = hit_elapsed * (1 - 0x1p-40);
  ExpectFree(FindHit(inputs, path_ending_before_hit));
}

TEST(FindFirstHit, StartElapsed_SphereTouchingRuleUsesLaterPose) {
  // A sphere rising at 1 m/s: half a second into the step its centre is at z = 0.5, so a particle at rest at
  // z = 1 + tol rests on top of it (gap 0) and the sphere pushes into it: hit at 0. Posed at the start of the step,
  // the start would be half a metre outside and the hit half a second later.
  const std::array solids = {
      BuildSphere({}, {.linear_velocity = {.x = 0, .y = 0, .z = 1}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path =
      {.start = {.x = 0, .y = 0, .z = 1 + kTolerance}, .velocity = kAtRest, .start_elapsed = 0.5, .duration = 0.5};
  const FirstHit first_hit = FindHit(inputs, path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = 1}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, StartElapsed_SphereRisingWithParticleIsFree) {
  // The same start, now rising with the sphere and sliding sideways: measured from the sphere it moves along the
  // tangent plane, so the touching rule frees it.
  const std::array solids = {
      BuildSphere({}, {.linear_velocity = {.x = 0, .y = 0, .z = 1}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(
      inputs,
      {
          .start = {.x = 0, .y = 0, .z = 1 + kTolerance},
          .velocity = {.x = 1, .y = 0, .z = 1},
          .start_elapsed = 0.5,
          .duration = 0.5,
      }
  ));
}

TEST(FindFirstHit, StartElapsed_SphereDeepStartUsesLaterPose) {
  // A sphere moving at {1, 0, 0} m/s from {-1, 0, 0}: one second into the step its centre is at the particle, deeper
  // than one tolerance inside, so a hit at 0 with the degenerate normal. Posed at the start of the step, the path
  // would start outside and be hit about half a second later.
  const std::array solids = {
      BuildSphere({.x = -1, .y = 0, .z = 0}, {.linear_velocity = {.x = 1, .y = 0, .z = 0}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(inputs, {.start = {}, .velocity = kAtRest, .start_elapsed = 1, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, -0.5);
}

TEST(FindFirstHit, StartElapsed_SphereHitUsesPoseOfTheHit) {
  // A sphere moving at {-1, 0, 0} m/s from {3, 0, 0}, swept from one second into the step: its centre is then at
  // {2, 0, 0}, and a particle at rest at the origin is reached 1.5 - tol later, one tolerance from the surface at the
  // pose of that moment, 2.5 - tol into the step.
  const std::array solids = {
      BuildSphere({.x = 3, .y = 0, .z = 0}, {.linear_velocity = {.x = -1, .y = 0, .z = 0}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 3);
  const FirstHit first_hit = FindHit(inputs, {.start = {}, .velocity = kAtRest, .start_elapsed = 1, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

// --- A cylinder ---

TEST(FindFirstHit, CylinderWall_HitsAtRadialGap) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0.25}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

TEST(FindFirstHit, CylinderCap_HitsAtHeightGap) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0.1, .y = 0.2, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = 1}, 1e-8);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

TEST(FindFirstHit, CylinderObliqueWall_CapPlaneOutsideDiscRejected) {
  // The path comes between the grown caps' planes at 1 - 2 tol s, still 1 + 2 tol from the axis, outside the grown
  // disc; it enters the grown wall at 1.5 - tol s, at height 0.75 + tol / 2, the later entry.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 1.5}, .velocity = {.x = 1, .y = 0, .z = -0.5}, .duration = 4});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
}

TEST(FindFirstHit, CylinderOverRimIntoWall_HitsWall) {
  // The path comes between the grown caps' planes at 1 - tol s, 1 + tol from the axis, and enters the grown wall at
  // 1.5 - tol s, at height 0.5 + tol: the later entry.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 2}, .velocity = {.x = 1, .y = 0, .z = -1}, .duration = 4});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
}

TEST(FindFirstHit, CylinderThroughGrownRimEdge_IsHit) {
  // Aimed exactly at the grown rim edge {0.5 + tol, 0, 1 + tol}, reached at 0.45 s, then deep into the cylinder. Here
  // the wall time rounds two units early, so the height at that time lands just above the grown cap, and the cap
  // time puts the path just outside the grown disc: a form that checks each candidate against the other surface
  // rejects both and lets the path through. The intervals inside the grown wall and between the grown caps overlap
  // from 0.45 s on, whatever the rounding. That entry is beside the rim, Sqrt(2) tol from it, so the rim search goes on
  // to the true distance: the path {1 - 3 s, 0, 1 - s} tol from the corner meets the circle of radius tol about the rim
  // at s = (8 - Sqrt(24)) / 20, 2 advances.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {
      .start = {.x = 1.8500000009313227, .y = 0, .z = 1.4500000009313225},
      .velocity = {.x = -3, .y = 0, .z = -1},
      .duration = 1,
  };
  const FirstHit first_hit = FindHit(inputs, path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 0.45 + (8 - std::sqrt(24.0)) / 20 * kTolerance, kTolerance / 1024);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  ExpectRimHitDistance(first_hit.at_hit.distance);
}

TEST(FindFirstHit, CylinderParallelToAxisOutside_IsFree) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  ExpectFree(
      FindHit(inputs, {.start = {.x = 0.6, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4})
  );
}

TEST(FindFirstHit, CylinderParallelToAxisInGrownRadius_HitsRim) {
  // The path never meets the wall (no motion across the axis) and enters the grown disc at {0.5 + tol / 2, 0, 1 + tol}
  // at 2 - tol s, beside the high rim, Sqrt(1.25) tol from it: the sharp corner of the grown shape. The rim search goes
  // on to where the true distance is tol, Sqrt(0.75) tol above the cap's plane, in 2 advances (plan decision P21).
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0.5 + kTolerance / 2, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 2 - std::sqrt(0.75) * kTolerance, kTolerance / 1024);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = 0.5, .y = 0, .z = std::sqrt(0.75)}, 1e-5);
  ExpectRimHitDistance(first_hit.at_hit.distance);
}

TEST(FindFirstHit, CylinderRimHeadOnAtFortyFiveDegrees_HitsAtOneTolerance) {
  // Along the rim's own normal {1, 0, 1} / Sqrt(2): the entry is the corner {0.5 + tol, 0, 1 + tol}, Sqrt(2) tol from
  // the rim, and one advance of the gap over its rate lands at tol, at 1 - Sqrt(0.5) tol s (as built, the corner at
  // 1 - tol s).
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 1.5, .y = 0, .z = 2}, .velocity = {.x = -1, .y = 0, .z = -1}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 1 - std::sqrt(0.5) * kTolerance, kTolerance / 1024);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = std::sqrt(0.5), .y = 0, .z = std::sqrt(0.5)}, 1e-5);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

TEST(FindFirstHit, CylinderRimPassedBeyondOneTolerance_IsFree) {
  // Paths {0.75 + a, 0, 0.75 + a} + {-1, 0, 1} t pass the rim at their closest at t = 0.25, a Sqrt(2) from it, through
  // the sharp corner of the grown shape. At a = 25/32 tol (1.105 tol) and 23/32 tol (1.016 tol) the true distance stays
  // above tol: free, where the entry into the sharp shape was a hit. At 22/32 tol (0.972 tol) it goes below: a hit.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 0.5);
  const Vec3 velocity = {.x = -1, .y = 0, .z = 1};
  for (const double passing_share : {25.0 / 32, 23.0 / 32}) {
    const double offset = passing_share * kTolerance;  // m
    const Vec3 start = {.x = 0.75 + offset, .y = 0, .z = 0.75 + offset};
    ExpectFree(FindHit(inputs, {.start = start, .velocity = velocity, .duration = 0.5}));
  }
  const double offset = 22.0 / 32 * kTolerance;  // m
  const Vec3 start = {.x = 0.75 + offset, .y = 0, .z = 0.75 + offset};
  const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = velocity, .duration = 0.5});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  ExpectRimHitDistance(first_hit.at_hit.distance);
}

TEST(FindFirstHit, CylinderParallelToCapsAbove_IsFree) {
  // The path is inside the grown wall from 1.5 - tol s, but never between the grown caps' planes: free.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  ExpectFree(
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 1.5}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 4})
  );
}

TEST(FindFirstHit, CylinderTouchingWallTangentPath_IsNotAHit) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = -(0.5 + kTolerance), .y = 0, .z = 0}, .velocity = {.x = 0, .y = 1, .z = 0}, .duration = 2}
  ));
}

TEST(FindFirstHit, CylinderTouchingWallPushingIn_HitsAtZero) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = -(0.5 + kTolerance), .y = 0, .z = 0}, .velocity = {.x = 1, .y = 1, .z = 0}, .duration = 2}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, CylinderRimCornerStartMovingAway_IsFree) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  // Wall and cap distances tol / 2: true distance Sqrt(0.5) tol, gap (Sqrt(0.5) - 1) tol < 0, so the touching rule
  // decides, and the path leaves along the rim normal.
  const Vec3 touching_start = {.x = 0.5 + kTolerance / 2, .y = 0, .z = 1 + kTolerance / 2};
  ExpectFree(FindHit(inputs, {.start = touching_start, .velocity = {.x = 1, .y = 0, .z = 1}, .duration = 1}));
  // Wall and cap distances 0.9 tol: inside the grown sharp cylinder, but at the true distance 0.9 Sqrt(2) tol > tol,
  // in its rim corner. The entry is 0 and the rim search decides: along the rim's normal the true distance reaches tol
  // after (0.9 - Sqrt(0.5)) tol s, one advance. Along {-1, 0, 0} the start rests on the cap's plane, 0.1 tol inside
  // its rest gap, and does not sink: free (plan decision P21, item 1). Moving away is free.
  const Vec3 corner_start = {.x = 0.5 + 0.9 * kTolerance, .y = 0, .z = 1 + 0.9 * kTolerance};
  const FirstHit approaching =
      FindHit(inputs, {.start = corner_start, .velocity = {.x = -1, .y = 0, .z = -1}, .duration = 1});
  EXPECT_EQ(approaching.solid_index, 0U);
  EXPECT_NEAR(approaching.elapsed, (0.9 - std::sqrt(0.5)) * kTolerance, kTolerance / 1024);
  EXPECT_EQ(approaching.at_hit.face, SolidFace::kHighRim);
  ExpectVec3Near(approaching.at_hit.normal_out_of_solid, {.x = std::sqrt(0.5), .y = 0, .z = std::sqrt(0.5)}, 1e-5);
  ExpectRimHitDistance(approaching.at_hit.distance);
  ExpectFree(FindHit(inputs, {.start = corner_start, .velocity = {.x = -1, .y = 0, .z = 0}, .duration = 1}));
  ExpectFree(FindHit(inputs, {.start = corner_start, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 1}));
}

TEST(FindFirstHit, CylinderRimCornerStartAlongTangent_IsFree) {
  // Wall and cap distances both exactly 0.875 tol, so the rim normal has equal x and z parts and the path along
  // {1, 0, -1} has rate exactly 0: it neither approaches the rim nor leaves it, and the true distance, convex along the
  // path, only grows. Free; only an approach (rate < 0) is a hit.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 corner_start = {.x = 0.5 + 0.875 * kTolerance, .y = 0, .z = 1 + 0.875 * kTolerance};
  ExpectFree(FindHit(inputs, {.start = corner_start, .velocity = {.x = 1, .y = 0, .z = -1}, .duration = 1}));
}

TEST(FindFirstHit, SpinningCylinderAboutAxis_IsClosedForm) {
  // A spin about the cylinder's own axis does not change its shape (plan P17): the wall time of the cylinder at rest,
  // at the start of the step and a quarter of a second into it.
  const std::array solids = {
      BuildCylinder({}, kAxisZ, {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 5}}),
  };
  ASSERT_EQ(CheckSolid(solids[0], 2.25), ContactError::kNone);
  const SolidContactInputs inputs = BuildInputs(solids, 2.25);
  for (const double start_elapsed : {0.0, 0.25}) {
    const Path path = {
        .start = {.x = -2, .y = 0, .z = 0.25},
        .velocity = {.x = 1, .y = 0, .z = 0},
        .start_elapsed = start_elapsed,
        .duration = 2,
    };
    const FirstHit first_hit = FindHit(inputs, path);
    EXPECT_EQ(first_hit.solid_index, 0U) << start_elapsed;
    ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
    EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
    ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
  }
}

TEST(FindFirstHit, TiltedSpinningCylinder_IsClosedForm) {
  // A spin about a tilted axis leaves a part across the axis of a few rounding units, far below the bound of plan
  // P17, so the cylinder is still swept by the closed form: the cap is hit 2 - tol after a start 3 m along the axis.
  const Vec3 axis = ToColumn(kTiltedAxes, 2);
  const std::array solids = {BuildCylinder({}, axis, {.linear_velocity = {}, .angular_velocity = 5 * axis})};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit = FindHit(inputs, {.start = 3 * axis, .velocity = -1 * axis, .duration = 4});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, axis, 1e-8);
}

// A cylinder of radius 2 m and half height 1 m along z, whose reach Length(half_size) is exactly 3 m, turning at
// across_rate rad/s about x, across its axis.
Solid BuildCylinderTurningAcross(double across_rate) {
  return ToSolid(
      Cylinder{.center = {}, .axis = kAxisZ, .radius = 2, .half_height = 1},
      {.linear_velocity = {}, .angular_velocity = {.x = across_rate, .y = 0, .z = 0}},
      1
  );
}

TEST(FindFirstHit, CylinderTurningAcrossBelowBound_IsClosedForm) {
  // In a step of 2 s the turn moves the surface by at most (tol / 64) * 2 * 3 = 0.09375 tol <= tol / 8 (plan P17):
  // the wall is hit at 2 - tol by the closed form.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 64)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -4, .y = 0, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 2 - kTolerance, 1);
}

TEST(FindFirstHit, CylinderTurningAcrossAtBound_IsClosedForm) {
  // In a step of 1 s at tol / 24 rad/s the bound's product (tol / 24) * 1 * 3 rounds to exactly tol / 8: the bound
  // itself still counts as not turning (plan P17 turns only above it).
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 24)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -3, .y = 0, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 1 - kTolerance, 1);
}

TEST(FindFirstHit, CylinderTurningAcrossAboveBound_RestingParticleIsHitAtZero) {
  // (tol / 32) * 2 * 3 = 0.1875 tol > tol / 8: a turning solid, which has no touching rule. A particle at rest on the
  // wall, on the axis it turns about, stays at distance tol, so it is hit at 0 where the closed form would free it.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 32)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -(2 + kTolerance), .y = 0, .z = 0}, .velocity = kAtRest, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
}

TEST(FindFirstHit, CylinderTurningAcrossAboveBound_ShortSweepRestingParticleIsHitAtZero) {
  // The bound counts the whole step, not the sweep: a sweep of a quarter of the step from half a second in is still
  // a turning solid, although (tol / 32) * 0.5 * 3 = 0.047 tol would be below tol / 8.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 32)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = -(2 + kTolerance), .y = 0, .z = 0}, .velocity = kAtRest, .start_elapsed = 0.5, .duration = 0.5}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
}

TEST(FindFirstHit, CylinderTurningAcrossBelowBound_RestingParticleIsFree) {
  // (tol / 64) * 2 * 3 = 0.09375 tol <= tol / 8: the closed form, whose touching rule frees a particle at rest on
  // the wall.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 64)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  ExpectFree(FindHit(inputs, {.start = {.x = -(2 + kTolerance), .y = 0, .z = 0}, .velocity = kAtRest, .duration = 2}));
}

TEST(FindFirstHit, CylinderTurningAcrossAtBound_RestingParticleIsFree) {
  // (tol / 24) * 1 * 3 rounds to exactly tol / 8, which still counts as not turning: the touching rule frees it.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 24)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  ExpectFree(FindHit(inputs, {.start = {.x = -(2 + kTolerance), .y = 0, .z = 0}, .velocity = kAtRest, .duration = 1}));
}

TEST(FindFirstHit, CylinderTurningAcrossBelowBound_LaterShortSweepIsClosedForm) {
  // Below the bound, a quarter-step sweep from half a second in uses the closed form at the pose of that moment,
  // whose axes have tilted by only (tol / 64) * 0.5 rad: from x = -2.5 at 1 m/s the grown wall, at x = -(2 + tol), is
  // reached after 0.5 - tol s.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 64)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(
      inputs,
      {
          .start = {.x = -2.5, .y = 0, .z = 0},
          .velocity = {.x = 1, .y = 0, .z = 0},
          .start_elapsed = 0.5,
          .duration = 0.5,
      }
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 0.5 - kTolerance, 1);
}

TEST(FindFirstHit, CylinderPathAlongAxis_HitsEachCap) {
  // On the axis with no motion across it, the path is inside the grown wall all along: it enters between the grown
  // caps' planes, through the high cap from above and the low cap from below.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit from_above =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4});
  EXPECT_EQ(from_above.solid_index, 0U);
  ExpectNearRelative(from_above.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(from_above.at_hit.face, SolidFace::kHighZ);
  const FirstHit from_below =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = -3}, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 4});
  EXPECT_EQ(from_below.solid_index, 0U);
  ExpectNearRelative(from_below.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(from_below.at_hit.face, SolidFace::kLowZ);
  ExpectVec3Near(from_below.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = -1}, 1e-8);
}

TEST(FindFirstHit, CylinderMovingAwayFromCapOrWall_IsFree) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  ExpectFree(
      FindHit(inputs, {.start = {.x = 0.1, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 4})
  );
  ExpectFree(
      FindHit(inputs, {.start = {.x = 0.1, .y = 0, .z = -3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4})
  );
  ExpectFree(
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0}, .velocity = {.x = -1, .y = 0, .z = 0}, .duration = 4})
  );
}

TEST(FindFirstHit, CylinderDownItsGrownWallPastTheRim_IsFree) {
  // Down the axis direction at exactly the grown radius 0.5 + tol, beside the high rim: the start rests on the wall
  // taken as unbounded along the axis, which the whole cylinder lies behind, and does not sink toward it, so the sweep
  // is free (plan decision P21, item 1). As built, the sharp grown edge stopped it Sqrt(2) tol from the rim.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = 0.5 + kTolerance, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4}
  ));
}

TEST(FindFirstHit, CylinderAlongItsGrownCapPlanePastTheRim_IsFree) {
  // Across the axis at exactly the grown half height 1 + tol: the start rests on the plane of the high cap, beside the
  // rim, and the path stays in that plane: free.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = -2, .y = 0, .z = 1 + kTolerance}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2}
  ));
}

TEST(FindFirstHit, CylinderBoundingSurfaceWindowAndSink) {
  // Down the wall from {x, 0, 3}, beside the high rim. A start tol / 8 inside the wall's rest gap rests on it; 3 tol /
  // 16 inside does not, and the rim search hits it where the true distance is tol, Sqrt(1 - (13 / 16)^2) tol above the
  // cap's plane. A path that sinks toward the wall by tol / 128 during the sweep still rests; one that sinks by tol /
  // 32 does not. Just above the rest gap (tol / 128) and sinking by tol / 64, the start rests too, where the sharp
  // grown wall stopped it at 1 s; tol / 8 above, the path never reaches the grown wall.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const Vec3 down = {.x = 0, .y = 0, .z = -1};
  const double rest_x = 0.5 + kTolerance;  // m
  ExpectFree(
      FindHit(inputs, {.start = {.x = rest_x - kTolerance / 8, .y = 0, .z = 3}, .velocity = down, .duration = 4})
  );
  const FirstHit deeper =
      FindHit(inputs, {.start = {.x = rest_x - 3 * kTolerance / 16, .y = 0, .z = 3}, .velocity = down, .duration = 4});
  EXPECT_EQ(deeper.solid_index, 0U);
  EXPECT_EQ(deeper.at_hit.face, SolidFace::kHighRim);
  // The search stops within tol / 1024 of the rest gap, where the gap closes at 0.584 m/s: within tol / 512 s.
  EXPECT_NEAR(deeper.elapsed, 2 - std::sqrt(1 - (13.0 / 16) * (13.0 / 16)) * kTolerance, kTolerance / 512);
  const Vec3 rest_start = {.x = rest_x, .y = 0, .z = 3};
  ExpectFree(
      FindHit(inputs, {.start = rest_start, .velocity = {.x = -kTolerance / 512, .y = 0, .z = -1}, .duration = 4})
  );
  const FirstHit sinking =
      FindHit(inputs, {.start = rest_start, .velocity = {.x = -kTolerance / 128, .y = 0, .z = -1}, .duration = 4});
  EXPECT_EQ(sinking.solid_index, 0U);
  EXPECT_EQ(sinking.at_hit.face, SolidFace::kHighRim);
  const Vec3 quick_sink = {.x = -kTolerance / 128, .y = 0, .z = -2};
  ExpectFree(FindHit(
      inputs,
      {.start = {.x = rest_x + kTolerance / 128, .y = 0, .z = 3}, .velocity = quick_sink, .duration = 2}
  ));
  ExpectFree(
      FindHit(inputs, {.start = {.x = rest_x + kTolerance / 8, .y = 0, .z = 3}, .velocity = quick_sink, .duration = 2})
  );
  // A wall of radius 4 m: the wall's rate is divided by the distance from the axis, so sinking by tol / 128 still
  // rests there; the product alone would read four times that, a sink of tol / 32.
  const std::array wide_solids = {
      ToSolid(Cylinder{.center = {}, .axis = kAxisZ, .radius = 4, .half_height = 1}, kNoMotion, 1),
  };
  const SolidContactInputs wide_inputs = BuildInputs(wide_solids, 4);
  ExpectFree(FindHit(
      wide_inputs,
      {
          .start = {.x = 4 + kTolerance, .y = 0, .z = 3},
          .velocity = {.x = -kTolerance / 512, .y = 0, .z = -1},
          .duration = 4,
      }
  ));
}

TEST(FindFirstHit, CylinderWallThatGivesTheDistance_KeepsItsOwnTouchingRule) {
  // Beside the wall, tol / 128 above its rest gap, sinking by tol / 64 over 2 s while rising along it: the wall gives
  // the distance, so the plain closed form decides (plan decision P16 alone) and the grown wall is reached at 1 s, at
  // the height of the cap's plane, still on the wall proper.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(
      inputs,
      {
          .start = {.x = 0.5 + kTolerance + kTolerance / 128, .y = 0, .z = 0},
          .velocity = {.x = -kTolerance / 128, .y = 0, .z = 1},
          .duration = 2,
      }
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 1, 1e-9);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
}

TEST(FindFirstHit, CylinderTangentToWall_FreeOrHitNearTwo) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = -2, .y = 0.5 + kTolerance, .z = 0.25}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2}
  );
  if (first_hit.solid_index == kNoSolid) {
    ExpectFree(first_hit);
    return;
  }
  EXPECT_NEAR(first_hit.elapsed, 2, 0x1p-20);
  EXPECT_NEAR(first_hit.at_hit.normal_out_of_solid.x, 0, 0x1p-20);
}

TEST(FindFirstHit, CylinderStartOneToleranceOutsideRest_HitsAfterGapOverSpeed) {
  // At distance 2 tol above the cap, or outside the wall, the gap is tol: the closed form gives tol / 1 s through the
  // cap, and tol / 0.7 s through the wall at 0.7 m/s. Through the wall the time is a tiny difference of large numbers
  // unless the root and its constant term are written as the method paper writes them, so it is checked to the
  // relative tolerance of tol itself.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit through_cap = FindHit(
      inputs,
      {.start = {.x = 0.1, .y = 0, .z = 1 + 2 * kTolerance}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1}
  );
  EXPECT_EQ(through_cap.solid_index, 0U);
  ExpectNearRelative(through_cap.elapsed, kTolerance, kTolerance);
  EXPECT_EQ(through_cap.at_hit.face, SolidFace::kHighZ);
  const FirstHit through_wall = FindHit(
      inputs,
      {
          .start = {.x = -(0.5 + 2 * kTolerance), .y = 0, .z = 0.25},
          .velocity = {.x = 0.7, .y = 0, .z = 0},
          .duration = 1,
      }
  );
  EXPECT_EQ(through_wall.solid_index, 0U);
  ExpectNearRelative(through_wall.elapsed, kTolerance / 0.7, kTolerance);
  EXPECT_EQ(through_wall.at_hit.face, SolidFace::kCurved);
}

TEST(FindFirstHit, CylinderHitExactlyAtEnd_IsHit) {
  // The duration set to the hit time the same path finds with a longer duration: the end counts, and a duration a
  // little shorter is free. Once through the wall, once through the cap.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const std::array long_paths = {
      Path{.start = {.x = -2, .y = 0.3, .z = 0.25}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 4},
      Path{.start = {.x = 0.1, .y = 0.2, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4},
  };
  for (const Path& long_path : long_paths) {
    const double hit_elapsed = FindHit(inputs, long_path).elapsed;
    Path path_ending_at_hit = long_path;
    path_ending_at_hit.duration = hit_elapsed;
    const FirstHit first_hit = FindHit(inputs, path_ending_at_hit);
    EXPECT_EQ(first_hit.solid_index, 0U);
    EXPECT_EQ(first_hit.elapsed, hit_elapsed);
    Path path_ending_before_hit = long_path;
    path_ending_before_hit.duration = hit_elapsed * (1 - 0x1p-40);
    ExpectFree(FindHit(inputs, path_ending_before_hit));
  }
}

TEST(FindFirstHit, CylinderStartJustInsideRest_PushingInHitsAtZero) {
  // Half a tolerance above the low cap, below it: gap -tol / 2. Sliding along the cap is free; pushing in is a hit.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = 0.1, .y = 0.1, .z = -(1 + kTolerance / 2)};
  ExpectFree(FindHit(inputs, {.start = start, .velocity = {.x = 1, .y = 0, .z = kTolerance / 4}, .duration = 1}));
  const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kLowZ);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance / 2);
}

TEST(FindFirstHit, CylinderStartAtCentre_HitsAtZeroWithDegenerateNormal) {
  // At the centre of a cylinder longer than wide the wall is nearer than the caps, and the point is on the axis: the
  // degenerate normal, minus the cylinder's own x axis, which ToSolid builds as {0, -1, 0} for the axis z.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(inputs, {.start = {}, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 1, .z = 0}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, -0.5);
}

TEST(FindFirstHitOfSweep, CylinderBesideRimRestingOnWall_FreeAlsoAfterAResponse) {
  // 3 / 16 tol above the cap's plane at the wall's rest gap, beside the rim 1.0174 tol away, falling along the wall:
  // the sweep tries the wall first and rests on it, so it is free, touched or not (plan decision P21, item 16 b); as
  // built the touched rule read the rim inside its window first and hit it at 0. tol / 32 inside the wall's rest gap
  // and tol / 8 above the cap's plane, the rim inside its own touching window: free too (was a hit at 0).
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 down = {.x = 0, .y = 0, .z = -1};
  const TouchedSolid touched = {.is_touched = true, .approach_allowance = 0x1p-46};
  const Sweep above_seam = {
      .start = {.x = 0.5 + kTolerance, .y = 0, .z = 1 + 3.0 / 16 * kTolerance},
      .velocity_for_this_step = down,
      .duration = 1,
  };
  FirstHit first_hit;
  ASSERT_EQ(FindFirstHitOfSweep(inputs, above_seam, touched, &first_hit, nullptr), ContactError::kNone);
  ExpectFree(first_hit);
  const Sweep inside_rest_gap = {
      .start = {.x = 0.5 + kTolerance - kTolerance / 32, .y = 0, .z = 1 + kTolerance / 8},
      .velocity_for_this_step = down,
      .duration = 1,
  };
  ASSERT_EQ(FindFirstHitOfSweep(inputs, inside_rest_gap, TouchedSolid{}, &first_hit, nullptr), ContactError::kNone);
  ExpectFree(first_hit);
}

TEST(FindFirstHit, StartElapsed_CylinderTouchingRuleUsesLaterPose) {
  // A cylinder rising at 1 m/s: half a second into the step its high cap is at z = 1.5, so a particle at rest at
  // z = 1.5 + tol rests on it and the cap pushes into it: hit at 0 on kHighZ. Posed at the start of the step, the
  // start would be half a metre above the cap.
  const std::array solids = {
      BuildCylinder({}, kAxisZ, {.linear_velocity = {.x = 0, .y = 0, .z = 1}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0.1, .y = 0, .z = 1.5 + kTolerance}, .velocity = kAtRest, .start_elapsed = 0.5, .duration = 0.5}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, StartElapsed_CylinderRimCornerUsesLaterPose) {
  // A cylinder moving at {1, 0, 0} m/s: half a second into the step its axis is at x = 0.5, so a particle at rest at
  // {1 + 0.9 tol, 0, 1 + 0.5 tol} is in its rim corner, 0.9 tol beyond the wall and 0.5 tol beyond the cap, and the rim
  // moves into it: the rim search hits it once the wall distance is Sqrt(0.75) tol, (0.9 - Sqrt(0.75)) tol s later.
  // Posed at the start of the step, the start would be half a metre outside the wall and the hit about half a second
  // later.
  const std::array solids = {
      BuildCylinder({}, kAxisZ, {.linear_velocity = {.x = 1, .y = 0, .z = 0}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {
      .start = {.x = 1 + 0.9 * kTolerance, .y = 0, .z = 1 + 0.5 * kTolerance},
      .velocity = kAtRest,
      .start_elapsed = 0.5,
      .duration = 0.5,
  };
  const FirstHit first_hit = FindHit(inputs, path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, (0.9 - std::sqrt(0.75)) * kTolerance, kTolerance / 1024);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
}

TEST(FindFirstHit, StartElapsed_CylinderRimCornerMovingWithCylinderIsFree) {
  // The same start, moving with the cylinder and away from its rim.
  const std::array solids = {
      BuildCylinder({}, kAxisZ, {.linear_velocity = {.x = 1, .y = 0, .z = 0}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {
      .start = {.x = 1 + 0.9 * kTolerance, .y = 0, .z = 1 + 0.5 * kTolerance},
      .velocity = {.x = 1, .y = 0, .z = 0.5},
      .start_elapsed = 0.5,
      .duration = 0.5,
  };
  ExpectFree(FindHit(inputs, path));
}

TEST(FindFirstHit, StartElapsed_CylinderDeepStartUsesLaterPose) {
  // A cylinder moving at {0, 0, -1} m/s from z = 1: one second into the step its centre is at the particle, so a hit
  // at 0 with the distance -0.5. Posed at the start of the step, the start would rest on the low cap and move away
  // from it, which is free.
  const std::array solids = {
      BuildCylinder(
          {.x = 0, .y = 0, .z = 1},
          kAxisZ,
          {.linear_velocity = {.x = 0, .y = 0, .z = -1}, .angular_velocity = {}}
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(inputs, {.start = {}, .velocity = kAtRest, .start_elapsed = 1, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.distance, -0.5);
}

// One face of a cylinder of radius 0.5 m and half height 1 m, and a path that hits it, in the cylinder's own axes.
struct CylinderFaceCase {
  SolidFace face = SolidFace::kHighZ;
  Vec3 own_start{};  // m, own axes, from the centre.
  Vec3 own_velocity{};  // m/s, own axes, measured from the cylinder.
  double hit_elapsed = 0;  // s.
  Vec3 own_normal_out_of_solid{};  // No unit, unit vector, own axes, at the hit.
  double normal_tolerance = 0;  // No unit. How far each axis of the normal may be from the expected one.
  double distance_at_hit = 0;  // m.
  double distance_tolerance = 0x1p-40;  // m. How far the distance at the hit may be from distance_at_hit.
};

// The six faces: the wall from two sides, both caps, and both rims entered through the sharp corner of the grown
// cylinder, where the entries into the grown wall and between the grown caps tie at 1 s, Sqrt(2) tol from the rim; the
// rim search goes on along the rim's normal to tol, (1 - Sqrt(0.5)) tol s later.
std::array<CylinderFaceCase, 6> BuildCylinderFaceCases() {
  const double rim_component = std::numbers::sqrt2 / 2;
  // A rim normal points from the rim to a hit about tol away, so a rounding of the turned coordinates of about
  // 1e-15 m turns it by about 1e-15 / tol, near 1e-6: the direction itself is that uncertain, not the code. The
  // review measured a worst error of 2.07e-6 over 2,000 random turns of these cases; 1e-5 is about 5 times that.
  constexpr double kRimNormalTolerance = 1e-5;
  return {
      {
          {
              .face = SolidFace::kCurved,
              .own_start = {.x = -2, .y = 0, .z = 0.25},
              .own_velocity = {.x = 1, .y = 0, .z = 0},
              .hit_elapsed = 1.5 - kTolerance,
              .own_normal_out_of_solid = {.x = -1, .y = 0, .z = 0},
              .normal_tolerance = 1e-8,
              .distance_at_hit = kTolerance,
          },
          {
              .face = SolidFace::kCurved,
              .own_start = {.x = 0, .y = 2, .z = -0.5},
              .own_velocity = {.x = 0, .y = -1, .z = 0.125},
              .hit_elapsed = 1.5 - kTolerance,
              .own_normal_out_of_solid = {.x = 0, .y = 1, .z = 0},
              .normal_tolerance = 1e-8,
              .distance_at_hit = kTolerance,
          },
          {
              .face = SolidFace::kHighZ,
              .own_start = {.x = 0.1, .y = 0.2, .z = 3},
              .own_velocity = {.x = 0, .y = 0, .z = -1},
              .hit_elapsed = 2 - kTolerance,
              .own_normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
              .normal_tolerance = 1e-8,
              .distance_at_hit = kTolerance,
          },
          {
              .face = SolidFace::kLowZ,
              .own_start = {.x = -0.2, .y = 0.1, .z = -3},
              .own_velocity = {.x = 0, .y = 0, .z = 1},
              .hit_elapsed = 2 - kTolerance,
              .own_normal_out_of_solid = {.x = 0, .y = 0, .z = -1},
              .normal_tolerance = 1e-8,
              .distance_at_hit = kTolerance,
          },
          {
              .face = SolidFace::kHighRim,
              .own_start = {.x = 1.5 + kTolerance, .y = 0, .z = 2 + kTolerance},
              .own_velocity = {.x = -1, .y = 0, .z = -1},
              .hit_elapsed = 1 + (1 - rim_component) * kTolerance,
              .own_normal_out_of_solid = {.x = rim_component, .y = 0, .z = rim_component},
              .normal_tolerance = kRimNormalTolerance,
              .distance_at_hit = kTolerance,
              // Where the rim search stopped, up to tol / 1024 beyond tol.
              .distance_tolerance = kTolerance / 1024,
          },
          {
              .face = SolidFace::kLowRim,
              .own_start = {.x = 0, .y = -(1.5 + kTolerance), .z = -(2 + kTolerance)},
              .own_velocity = {.x = 0, .y = 1, .z = 1},
              .hit_elapsed = 1 + (1 - rim_component) * kTolerance,
              .own_normal_out_of_solid = {.x = 0, .y = -rim_component, .z = -rim_component},
              .normal_tolerance = kRimNormalTolerance,
              .distance_at_hit = kTolerance,
              // Where the rim search stopped, up to tol / 1024 beyond tol.
              .distance_tolerance = kTolerance / 1024,
          },
      },
  };
}

// Fails the calling test unless each face case hits its face, with a cylinder of radius 0.5 m and half height 1 m at
// {1, 2, 3} along axis, moving, a quarter of a second into the step. Each case's start and velocity are written in
// world axes with the cylinder's own axes, so every orientation expects the same hit, up to the rounding of the turn.
void ExpectEachCylinderFaceHit(Vec3 axis) {
  const SolidMotion motion = {.linear_velocity = {.x = 0.5, .y = -0.25, .z = 1}, .angular_velocity = {}};
  const Vec3 center = {.x = 1, .y = 2, .z = 3};
  const std::array solids = {BuildCylinder(center, axis, motion)};
  ASSERT_EQ(CheckSolid(solids[0], 3.25), ContactError::kNone);
  const SolidContactInputs inputs = BuildInputs(solids, 3.25);
  const Mat3 axes = solids[0].pose_at_step_start.axes;
  const Vec3 center_at_start = center + 0.25 * motion.linear_velocity;
  for (const CylinderFaceCase& face_case : BuildCylinderFaceCases()) {
    const Path path = {
        .start = center_at_start + axes * face_case.own_start,
        .velocity = motion.linear_velocity + axes * face_case.own_velocity,
        .start_elapsed = 0.25,
        .duration = 3,
    };
    const FirstHit first_hit = FindHit(inputs, path);
    EXPECT_EQ(first_hit.solid_index, 0U) << static_cast<int>(face_case.face);
    ExpectNearRelative(first_hit.elapsed, face_case.hit_elapsed, 1);
    EXPECT_EQ(first_hit.at_hit.face, face_case.face);
    ExpectVec3Near(
        first_hit.at_hit.normal_out_of_solid,
        axes * face_case.own_normal_out_of_solid,
        face_case.normal_tolerance
    );
    EXPECT_NEAR(first_hit.at_hit.distance, face_case.distance_at_hit, face_case.distance_tolerance)
        << static_cast<int>(face_case.face);
  }
}

TEST(FindFirstHit, MovingCylinderEachFace_HitsThatFace) {
  ExpectEachCylinderFaceHit(kAxisZ);
}

TEST(FindFirstHit, TiltedMovingCylinderEachFace_HitsThatFace) {
  ExpectEachCylinderFaceHit(ToColumn(kTiltedAxes, 2));
}

// --- The rim search ---

// The half sizes of the test cylinder, radius 0.5 m and half height 1 m.
constexpr Vec3 kCylinderHalfSize = {.x = 0.5, .y = 0.5, .z = 1};

// What the rim search returned for one own path, and how many advances it made.
struct RimSearch {
  HitTime hit_time{};
  int advance_count = -1;
};

// Runs the rim search on a path in a cylinder's own axes from the entry the two intervals give, as FindCylinderHit
// does; a path that does not enter the grown cylinder within the duration is free with no advance.
RimSearch SearchRimFromEntry(OwnPath own_path, Vec3 half_size, double duration) {
  const TimeInterval inside_wall = ComputeTimeInsideGrownWall(own_path, half_size, kTolerance);
  const TimeInterval between_caps = ComputeTimeBetweenGrownCaps(own_path, half_size, kTolerance);
  RimSearch search;
  search.advance_count = 0;
  if (inside_wall.is_empty || between_caps.is_empty) {
    return search;
  }
  const double entry_elapsed = std::max(inside_wall.entry_elapsed, between_caps.entry_elapsed);  // s
  const bool leaves_wall_first = inside_wall.has_exit && inside_wall.exit_elapsed < entry_elapsed;
  const bool leaves_caps_first = between_caps.has_exit && between_caps.exit_elapsed < entry_elapsed;
  if (leaves_wall_first || leaves_caps_first || !(entry_elapsed <= duration)) {
    return search;
  }
  search.hit_time = FindCylinderRimHit(own_path, half_size, kTolerance, entry_elapsed, duration, &search.advance_count);
  return search;
}

// The cap distance at a rim search's hit, in the test cylinder's own axes.
double ComputeCapDistanceAtHit(OwnPath own_path, HitTime hit_time) {
  const Vec3 hit_point = own_path.start + own_path.velocity * hit_time.elapsed;
  return std::abs(hit_point.z) - kCylinderHalfSize.z;
}

// One path of the rim search's tests: the own path, the sweep's duration and the advances expected.
struct RimSearchCase {
  OwnPath own_path{};
  double duration = 0;  // s.
  int advance_count = 0;
};

TEST(FindCylinderRimHit, AdvanceCounts) {
  // The own paths of the first-hit tests, from the entry of the two intervals: each advance is the gap over the speed
  // at which it closes, so a path along the rim's normal needs one and a path down the wall two (plan decision P21).
  const auto expect_advances = [](RimSearchCase search_case) {
    const RimSearch search = SearchRimFromEntry(search_case.own_path, kCylinderHalfSize, search_case.duration);
    EXPECT_TRUE(search.hit_time.has_hit);
    EXPECT_EQ(search.advance_count, search_case.advance_count) << search_case.own_path.start.x;
    return search;
  };
  const Vec3 down = {.x = 0, .y = 0, .z = -1};
  expect_advances({
      .own_path = {.start = {.x = 0.5 + kTolerance / 2, .y = 0, .z = 3}, .velocity = down},
      .duration = 4,
      .advance_count = 2,
  });
  expect_advances({
      .own_path =
          {
              .start = {.x = 1.8500000009313227, .y = 0, .z = 1.4500000009313225},
              .velocity = {.x = -3, .y = 0, .z = -1},
          },
      .duration = 1,
      .advance_count = 2,
  });
  expect_advances({
      .own_path = {.start = {.x = 1.5, .y = 0, .z = 2}, .velocity = {.x = -1, .y = 0, .z = -1}},
      .duration = 2,
      .advance_count = 1,
  });
  expect_advances({
      .own_path =
          {
              .start = {.x = 0.5 + 0.9 * kTolerance, .y = 0, .z = 1 + 0.9 * kTolerance},
              .velocity = {.x = -1, .y = 0, .z = -1},
          },
      .duration = 1,
      .advance_count = 1,
  });
  const RimSearch deeper = expect_advances({
      .own_path = {.start = {.x = 0.5 + 13.0 / 16 * kTolerance, .y = 0, .z = 3}, .velocity = down},
      .duration = 4,
      .advance_count = 2,
  });
  // Within tol / 1024 of the rest gap, closing at 0.584 m/s: within tol / 512 s of the exact time.
  EXPECT_NEAR(deeper.hit_time.elapsed, 2 - std::sqrt(1 - (13.0 / 16) * (13.0 / 16)) * kTolerance, kTolerance / 512);
  // Sinking toward the wall at tol / 128 m/s from its rest gap: the path crosses the wall's grown plane 0.016 tol
  // inside, so the gap closes slowly at the end, and 4 advances reach the rim where the cap distance is 0.1763 tol.
  const OwnPath sinking = {
      .start = {.x = 0.5 + kTolerance, .y = 0, .z = 3},
      .velocity = {.x = -kTolerance / 128, .y = 0, .z = -1},
  };
  const RimSearch sinking_search = expect_advances({.own_path = sinking, .duration = 4, .advance_count = 4});
  EXPECT_NEAR(ComputeCapDistanceAtHit(sinking, sinking_search.hit_time), 0.1763 * kTolerance, kTolerance / 512);
  // An entry on the wall proper is the hit itself: no advance, the entry's time bit for bit.
  const OwnPath onto_wall = {.start = {.x = -2, .y = 0, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}};
  const RimSearch wall_search = expect_advances({.own_path = onto_wall, .duration = 2, .advance_count = 0});
  EXPECT_EQ(
      wall_search.hit_time.elapsed,
      ComputeTimeInsideGrownWall(onto_wall, kCylinderHalfSize, kTolerance).entry_elapsed
  );
}

TEST(FindCylinderRimHit, GapRateTooSmallToSquare_IsFreeNotStalled) {
  // From the rim corner 0.9 tol beyond the wall and the cap, closing at 1.4e-170 m/s: the advance is the
  // quotient, 1.8e160 s, far beyond the sweep, so the path is free with no advance. Through ComputeSecondOrderAdvance
  // the square of the gap rate underflows to 0, the advance is 0, and the search would report a hit 0.27 tol outside
  // at its limit. (At 1.4e-160 m/s the square is a subnormal number, and that form still advances.)
  const OwnPath own_path = {
      .start = {.x = 0.5 + 0.9 * kTolerance, .y = 0, .z = 1 + 0.9 * kTolerance},
      .velocity = {.x = -1e-170, .y = 0, .z = -1e-170},
  };
  int advance_count = -1;
  const HitTime hit_time = FindCylinderRimHit(own_path, kCylinderHalfSize, kTolerance, 0, 1, &advance_count);
  EXPECT_FALSE(hit_time.has_hit);
  EXPECT_EQ(hit_time.elapsed, 0);
  EXPECT_EQ(advance_count, 0);
}

TEST(FindCylinderRimHit, GapThatWouldNotCloseWithinTheSweep_IsFree) {
  // tol / 2048 outside the round rim's rest gap, below the hit gap tol / 1024, closing along the rim's normal at
  // 1e-3 tol/s: at that rate, which only slows, the gap would close after 0.49 s. Within a sweep of 0.25 s it does not:
  // free. Within 1 s it does: a hit where it stands.
  const Vec3 rim_normal = {.x = std::sqrt(0.5), .y = 0, .z = std::sqrt(0.5)};
  const OwnPath own_path = {
      .start = Vec3{.x = 0.5, .y = 0, .z = 1} + (kTolerance + kTolerance / 2048) * rim_normal,
      .velocity = -1e-3 * kTolerance * rim_normal,
  };
  int advance_count = -1;
  const HitTime short_sweep = FindCylinderRimHit(own_path, kCylinderHalfSize, kTolerance, 0, 0.25, &advance_count);
  EXPECT_FALSE(short_sweep.has_hit);
  EXPECT_EQ(advance_count, 0);
  const HitTime long_sweep = FindCylinderRimHit(own_path, kCylinderHalfSize, kTolerance, 0, 1, &advance_count);
  EXPECT_TRUE(long_sweep.has_hit);
  EXPECT_EQ(long_sweep.elapsed, 0);
  EXPECT_EQ(advance_count, 0);
}

TEST(ComputeIsRestingOnBoundingSurface, WindowAndSink) {
  // A gap from -tol / 8 to tol / 16, both limits included, that sinks by at most tol / 64 during the sweep. With the
  // tolerance 0, the window is the gap 0 alone and the path must not sink at all.
  EXPECT_TRUE(ComputeIsRestingOnBoundingSurface(kTolerance, -kTolerance / 8, 0, 1));
  EXPECT_FALSE(ComputeIsRestingOnBoundingSurface(kTolerance, -kTolerance / 8 - 0x1p-60, 0, 1));
  EXPECT_TRUE(ComputeIsRestingOnBoundingSurface(kTolerance, kTolerance / 16, 0, 1));
  EXPECT_FALSE(ComputeIsRestingOnBoundingSurface(kTolerance, kTolerance / 16 + 0x1p-60, 0, 1));
  EXPECT_TRUE(ComputeIsRestingOnBoundingSurface(kTolerance, 0, -kTolerance / 64, 1));
  EXPECT_FALSE(ComputeIsRestingOnBoundingSurface(kTolerance, 0, -kTolerance / 64, 1 + 0x1p-52));
  EXPECT_TRUE(ComputeIsRestingOnBoundingSurface(kTolerance, 0, 5, 1));
  EXPECT_TRUE(ComputeIsRestingOnBoundingSurface(0, 0, 0, 1));
  EXPECT_FALSE(ComputeIsRestingOnBoundingSurface(0, 0, -0x1p-60, 1));
  EXPECT_FALSE(ComputeIsRestingOnBoundingSurface(0, 0x1p-60, 0, 1));
}

// --- A turning solid ---

// A box of half size 0.5 m centred at the origin at the start of the step, with world axes.
Solid BuildTurningBox(SolidMotion motion) {
  return ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, motion, 1);
}

// The world position of an own point of a solid at its pose elapsed seconds into the step.
Vec3 ToWorldPoint(const Solid& solid, double elapsed, Vec3 own_point) {
  const SolidPose pose = ComputeSolidPose(solid, elapsed);
  return pose.origin + pose.axes * own_point;
}

// s. The time the search may stop before the true hit: it stops at a gap of at most tol / 4, which a face closing
// at about 0.33 m/s covers in 0.76 tol s.
constexpr double kTurningHitTimeTolerance = 0x1p-30;

TEST(FindFirstHit, TurningBoxReachesRestingParticle_AtTrigonometricAngle) {
  // Measured from the box the particle goes round at radius 0.6 m, at the angle 0.3 - t from the own x axis, so face
  // kHighX reaches it when 0.6 cos(t - 0.3) = 0.5 + tol.
  const std::array solids = {BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}})};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = 0.6 * std::cos(0.3), .y = 0.6 * std::sin(0.3), .z = 0};
  const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = kAtRest, .duration = 1});
  const double expected_elapsed = 0.3 + std::acos((0.5 + kTolerance) / 0.6);  // s, 0.8857
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, expected_elapsed, kTurningHitTimeTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
}

TEST(FindFirstHit, TurningBoxNotReachedInStep_IsFree) {
  // Turning the other way, the particle's angle from the own x axis grows from 0.3 to 0.5 rad in 0.2 s: the distance
  // to face kHighX falls from 0.073 m to 0.6 cos(0.5) - 0.5 = 0.027 m and never reaches tol.
  const std::array solids = {BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = -1}})};
  const SolidContactInputs inputs = BuildInputs(solids, 0.2);
  const Vec3 start = {.x = 0.6 * std::cos(0.3), .y = 0.6 * std::sin(0.3), .z = 0};
  ExpectFree(FindHit(inputs, {.start = start, .velocity = kAtRest, .duration = 0.2}));
}

// Fails the calling test unless a hit of the stepping search sits at the surface of the grown solid: its distance is
// in [tol - tol / 64, (1 + kTurningSolidHitGapShare) tol], the rounding allowance below and the stopping gap above.
void ExpectAtGrownSurface(DistanceToSolid at_hit) {
  EXPECT_GE(at_hit.distance, kTolerance - kTolerance / 64);
  EXPECT_LE(at_hit.distance, (1 + kTurningSolidHitGapShare) * kTolerance);
}

// The answer of the stepping search toward solid 0, called directly, with the advances it made.
struct TurningSearch {
  ContactError error = ContactError::kNone;
  FirstHit first_hit{};
  int advance_count = 0;
};

// Runs FindHitOnTurningSolid toward solid 0 along one path.
TurningSearch SearchTurningSolid(const SolidContactInputs& inputs, Path path) {
  const Sweep sweep = {
      .start = path.start,
      .velocity_for_this_step = path.velocity,
      .start_elapsed = path.start_elapsed,
      .duration = path.duration,
  };
  TurningSearch search;
  search.error = FindHitOnTurningSolid(inputs, 0, sweep, &search.first_hit, &search.advance_count);
  return search;
}

TEST(FindFirstHit, GrazingTurningBox_HitWhereFaceTurnsOntoParticle) {
  // Two tolerances from the middle of face kHighX, the particle's distance falls as (0.5 + 2 tol) cos(t) - 0.5, so
  // the face reaches it at acos((0.5 + tol) / (0.5 + 2 tol)) s. The second-order advance follows the gap's curvature
  // and gets there in one advance; the first-order advance alone, about 2 tol s each, needed 43,154.
  const std::array solids = {BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}})};
  const SolidContactInputs inputs = BuildInputs(solids, 0.1);
  const double radius = 0.5 + 2 * kTolerance;  // m
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = radius, .y = 0, .z = 0}, .velocity = kAtRest, .duration = 0.1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, std::acos((0.5 + kTolerance) / radius), kTurningHitTimeTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
  ExpectAtGrownSurface(first_hit.at_hit);
}

TEST(FindFirstHit, TurningBoxSlowApproachWithinAdvanceLimit_IsHit) {
  // At rest at radius r = 0.5 + 3 * 2^-9 m on the own x axis: face kHighX closes on the particle as r cos(t) - 0.5,
  // starting at rate 0, and reaches it at acos((0.5 + tol) / r) = 0.152 s, at 0.077 m/s against a bound of 0.506 m/s.
  // The first-order advance alone needed 113 advances; the search now takes 2. A gap of tol / 4 closing at 0.077 m/s
  // takes 3.2 tol seconds, so the hit may come that much early.
  const std::array solids = {BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}})};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const double radius = 0.505859375;  // m
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = radius, .y = 0, .z = 0}, .velocity = kAtRest, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, std::acos((0.5 + kTolerance) / radius), 4 * kTurningHitTimeTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
}

TEST(FindFirstHit, TurningBoxSlowApproachBeyondFirstOrderLimit_IsHit) {
  // The same at radius 0.5 + 3 * 2^-10 m: the face reaches the particle at 0.108 s, slowly enough that the first-order
  // advance alone would need 157 advances, more than kMaxAdvanceCount. The second-order advance needs 2.
  const std::array solids = {BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}})};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const double radius = 0.5029296875;  // m
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = radius, .y = 0, .z = 0}, .velocity = kAtRest, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, std::acos((0.5 + kTolerance) / radius), 4 * kTurningHitTimeTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
  ExpectAtGrownSurface(first_hit.at_hit);
}

// The remaining close call. On the turn axis, above the top face of a box of half size 0.5 m turning at {0, 0, 1}
// rad/s, a particle at rest at {0, 0, z} keeps the own point {0, 0, z} exactly: the distance is z - 0.5, the gap
// rate exactly 0 and K = z, so every advance is 2 g / Sqrt(2 K g) = Sqrt(2 g / z) with g = z - 0.5 - tol, whatever
// Sin gives. The count of advances over a sweep of 1 s is the first integer above 1 / Sqrt(2 g / z).

TEST(FindFirstHit, TurningBoxHoverOnSpinAxisWithinAdvanceLimit_IsFree) {
  // z = 0.5 + 2^-15: 1 / Sqrt(2 g / z) = 90.5, so 91 advances reach the end of the sweep. The first-order advance
  // alone, g / z each, would need about 16,400.
  const std::array solids = {BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}})};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {.start = {.x = 0, .y = 0, .z = 0.5 + 0x1p-15}, .velocity = kAtRest, .duration = 1};
  ExpectFree(FindHit(inputs, path));
  const TurningSearch search = SearchTurningSolid(inputs, path);
  EXPECT_EQ(search.error, ContactError::kNone);
  EXPECT_EQ(search.advance_count, 91);
}

TEST(FindFirstHit, TurningBoxHoverOnSpinAxisBeyondAdvanceLimit_IsCloseCall) {
  // z = 0.5 + 2^-17: 1 / Sqrt(2 g / z) = 181.02, far from an integer, so 182 advances would be needed whatever the
  // rounding, more than kMaxAdvanceCount.
  const std::array solids = {BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}})};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  FirstHit first_hit;
  ContactCloseCall close_call = ContactCloseCall::kNone;
  const ContactError error =
      FindFirstHit(inputs, {.x = 0, .y = 0, .z = 0.5 + 0x1p-17}, kAtRest, 0, 1, &first_hit, &close_call);
  EXPECT_EQ(error, ContactError::kTooCloseToCall);
  EXPECT_EQ(close_call, ContactCloseCall::kSlowApproachToTurningSolid);
  EXPECT_EQ(first_hit.solid_index, 0U);
}

TEST(FindFirstHit, HeadOnSlowlyTurningBox_HitsJustBeforeGapOverSpeed) {
  // A box moving at {1, 0, 0} m/s toward a particle at rest at x = 3 m, turning slowly about the x axis, which leaves
  // the particle's distance unchanged: the gap closes at 1 m/s and reaches 0 at 2.5 - tol s. The bound,
  // 1 + 0.01 * 3 = 1.03 m/s, is within 3 % of that rate, so an advance longer than the gap over the bound, or a bound
  // that left out the box's velocity, would pass the hit. The search stops within tol / 4 of the grown face.
  const std::array solids = {
      BuildTurningBox({.linear_velocity = {.x = 1, .y = 0, .z = 0}, .angular_velocity = {.x = 0.01, .y = 0, .z = 0}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 3);
  const FirstHit first_hit = FindHit(inputs, {.start = {.x = 3, .y = 0, .z = 0}, .velocity = kAtRest, .duration = 3});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_GE(first_hit.elapsed, 2.5 - (1 + kTurningSolidHitGapShare) * kTolerance - 1e-15);
  EXPECT_LE(first_hit.elapsed, 2.5 - kTolerance + 1e-15);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
}

TEST(FindFirstHit, RestingOnTurningBox_IsHitAtZero) {
  // At rest on face kHighX (distance tol exactly, gap 0) and moving along it: a turning solid has no touching rule.
  // The same box without the turn frees the same path by the touching rule.
  const Vec3 start = {.x = 0.5 + kTolerance, .y = 0, .z = 0};
  const Vec3 along_face = {.x = 0, .y = 0, .z = 1};
  const std::array turning = {BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}})};
  const FirstHit first_hit =
      FindHit(BuildInputs(turning, 0.1), {.start = start, .velocity = along_face, .duration = 0.1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
  const std::array still = {BuildTurningBox(kNoMotion)};
  ExpectFree(FindHit(BuildInputs(still, 0.1), {.start = start, .velocity = along_face, .duration = 0.1}));
}

TEST(FindFirstHit, CylinderTurningAcrossAxis_ReachesRestingParticleAtTrigonometricAngle) {
  // A cylinder along x turning about z: the particle sits 1.2 - t rad from the axis, at 0.6 sin(1.2 - t) m from it and
  // 0.6 cos(1.2 - t) < 1 m along it, so the wall reaches it when 0.6 sin(1.2 - t) = 0.5 + tol.
  const std::array solids = {
      BuildCylinder(
          {},
          {.x = 1, .y = 0, .z = 0},
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}}
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = 0.6 * std::cos(1.2), .y = 0.6 * std::sin(1.2), .z = 0};
  const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = kAtRest, .duration = 1});
  const double expected_elapsed = 1.2 - std::asin((0.5 + kTolerance) / 0.6);  // s
  EXPECT_NEAR(expected_elapsed, 0.21488921385421111, 1e-15);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, expected_elapsed, kTurningHitTimeTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
}

TEST(FindFirstHit, TurningPlateCommitted_SearchUsesCommittedHalfSpace) {
  // The plate of CommittedPlate_UsesOnlyCommittedFace, turning about an axis in its plane: a turning solid, searched
  // with the distance a committed particle sees (plan decision P3). From inside the thickness at z = -0.005 both forms
  // hit at 0, an uncommitted particle on kLowZ at -0.005, a particle committed to kAlongNormal on kHighZ at -0.015.
  const std::array solids = {
      ToSolid(
          Plate{.center = {}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.01},
          {.linear_velocity = {}, .angular_velocity = {.x = 1, .y = 0, .z = 0}},
          5
      ),
  };
  SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {.start = {.x = 0, .y = 0, .z = -0.005}, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 1};
  const FirstHit uncommitted = FindHit(inputs, path);
  EXPECT_EQ(uncommitted.elapsed, 0);
  EXPECT_EQ(uncommitted.at_hit.face, SolidFace::kLowZ);
  EXPECT_EQ(uncommitted.at_hit.distance, 0.005 - 0.01);
  inputs.plate_side = {.plate_id = 5, .side = SolidSide::kAlongNormal};
  const FirstHit committed = FindHit(inputs, path);
  EXPECT_EQ(committed.solid_index, 0U);
  EXPECT_EQ(committed.elapsed, 0);
  EXPECT_EQ(committed.at_hit.face, SolidFace::kHighZ);
  EXPECT_EQ(committed.at_hit.distance, -0.005 - 0.01);
}

// The floor through the origin with normal {0, 0, 1}, spinning about its normal at angular_velocity_z rad/s.
Solid BuildSpinningFloor(double angular_velocity_z) {
  return ToSolid(
      FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
      {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = angular_velocity_z}},
      1
  );
}

// A plate of half thickness 0.01 m through the origin with normal {0, 0, 1}, turning at angular_velocity.
Solid BuildPlateWithNormalZ(Vec3 angular_velocity) {
  return ToSolid(
      Plate{.center = {}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.01},
      {.linear_velocity = {}, .angular_velocity = angular_velocity},
      1
  );
}

TEST(FindFirstHit, SpinningFlatSolid_IsClosedForm) {
  // A flat solid is unbounded sideways, so a spin about its normal leaves its shape as it is (plan decision P17): it
  // is not a turning solid, and the closed form gives the bits of the floor without the spin. 1 rad/s keeps the turn
  // within kMaxTurnPerStep, so a spin counted as a turn shows here as another hit time, not as a refused solid.
  const Path path = {.start = {.x = 0.3, .y = 0, .z = 0.5}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1};
  const std::array spinning = {BuildSpinningFloor(1)};
  const FirstHit first_hit = FindHit(BuildInputs(spinning, 1), path);
  const std::array still = {BuildSpinningFloor(0)};
  const FirstHit still_hit = FindHit(BuildInputs(still, 1), path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0.5 - kTolerance);
  EXPECT_EQ(first_hit.elapsed, still_hit.elapsed);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
}

TEST(FindFirstHit, RestingOnSpinningFlatSolid_TouchingRuleFrees) {
  // At rest on the spinning floor (distance tol) and sliding along it: a solid that does not turn keeps its touching
  // rule (plan decision P2). Counted as turning, it would be hit at 0.
  const std::array solids = {BuildSpinningFloor(5)};
  const Path path = {
      .start = {.x = 0.3, .y = 0, .z = kTolerance},
      .velocity = {.x = 1, .y = 0, .z = 0},
      .duration = 0.1,
  };
  ExpectFree(FindHit(BuildInputs(solids, 0.1), path));
}

TEST(FindFirstHit, SpinningPlateAboutNormal_IsClosedForm) {
  // A plate is unbounded sideways too, so its spin about the normal is not a turn either (1 rad/s, as above).
  const Path path = {.start = {.x = 0.3, .y = 0, .z = 0.5}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1};
  const std::array spinning = {BuildPlateWithNormalZ({.x = 0, .y = 0, .z = 1})};
  const FirstHit first_hit = FindHit(BuildInputs(spinning, 1), path);
  const std::array still = {BuildPlateWithNormalZ({})};
  const FirstHit still_hit = FindHit(BuildInputs(still, 1), path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, (0.5 - 0.01) - kTolerance);
  EXPECT_EQ(first_hit.elapsed, still_hit.elapsed);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
}

TEST(FindFirstHit, RestingOnSpinningPlate_TouchingRuleFrees) {
  const std::array solids = {BuildPlateWithNormalZ({.x = 0, .y = 0, .z = 5})};
  const Path path = {
      .start = {.x = 0.3, .y = 0, .z = 0.01 + kTolerance},
      .velocity = {.x = 1, .y = 0, .z = 0},
      .duration = 0.1,
  };
  ExpectFree(FindHit(BuildInputs(solids, 0.1), path));
}

TEST(FindFirstHit, PlateTurningAboutAxisInItsPlane_IsTurning) {
  // A turn about an axis in the plate's plane tilts its faces: a turning solid, which has no touching rule (plan
  // decision P2), so the same resting particle is hit at 0.
  const std::array solids = {BuildPlateWithNormalZ({.x = 1, .y = 0, .z = 0})};
  const Path path = {
      .start = {.x = 0.3, .y = 0, .z = 0.01 + kTolerance},
      .velocity = {.x = 1, .y = 0, .z = 0},
      .duration = 0.1,
  };
  const FirstHit first_hit = FindHit(BuildInputs(solids, 0.1), path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
}

// The tilted normal {0, sin 0.3, cos 0.3}, on which a spin about the normal leaves a part across it of a few rounding
// units at most rates (9e-16 rad/s at 5 rad/s), and the two unit directions {1, 0, 0} and {0, cos 0.3, -sin 0.3} in
// its plane.
Vec3 ComputeTiltedNormal() {
  return {.x = 0, .y = std::sin(0.3), .z = std::cos(0.3)};
}
Vec3 ComputeAlongTiltedPlane() {
  return {.x = 0, .y = std::cos(0.3), .z = -std::sin(0.3)};
}

// The flat solid through the origin with normal ComputeTiltedNormal(), spinning about that normal at spin_rate rad/s.
Solid BuildSpinningTiltedFloor(double spin_rate) {
  return ToSolid(
      FlatSolid{.point_on_surface = {}, .normal_out_of_solid = ComputeTiltedNormal()},
      {.linear_velocity = {}, .angular_velocity = spin_rate * ComputeTiltedNormal()},
      1
  );
}

TEST(FindFirstHit, FallingOnFastSpinningTiltedFloor_IsClosedForm) {
  // 100 rad/s about the tilted normal in a step of 0.1 s, 10 rad of spin: the part across the normal that rounding
  // leaves is far below kNegligibleTurnShare, so the floor does not turn, and the closed form gives the bits of the
  // still floor: a hit about (0.5 - tol) / 10 s in.
  const Path path = {.start = 0.5 * ComputeTiltedNormal(), .velocity = -10 * ComputeTiltedNormal(), .duration = 0.1};
  const std::array spinning = {BuildSpinningTiltedFloor(100)};
  const std::array still = {BuildSpinningTiltedFloor(0)};
  const FirstHit spinning_hit = FindHit(BuildInputs(spinning, 0.1), path);
  const FirstHit still_hit = FindHit(BuildInputs(still, 0.1), path);
  EXPECT_EQ(spinning_hit.solid_index, 0U);
  EXPECT_EQ(spinning_hit.elapsed, still_hit.elapsed);
  EXPECT_NEAR(spinning_hit.elapsed, (0.5 - kTolerance) / 10, 1e-15);
  EXPECT_EQ(spinning_hit.at_hit.face, SolidFace::kHighZ);
}

TEST(FindFirstHit, RestingOnFastSpinningTiltedFloor_TouchingRuleFrees) {
  // At rest one tolerance above the tilted floor spinning at 100 rad/s: the touching rule frees it, with no close call.
  // Counted as turning on its rounding bits, the floor would hit it at 0 in every sweep.
  const std::array solids = {BuildSpinningTiltedFloor(100)};
  FirstHit first_hit;
  ContactCloseCall close_call = ContactCloseCall::kTooManyHits;
  const Vec3 start = Vec3{.x = 0.3, .y = 0, .z = 0} + kTolerance * ComputeTiltedNormal();
  const ContactError error = FindFirstHit(BuildInputs(solids, 0.1), start, kAtRest, 0, 0.1, &first_hit, &close_call);
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_EQ(close_call, ContactCloseCall::kNone);
  ExpectFree(first_hit);
}

TEST(FindFirstHit, SlidingOnFastSpinningTiltedFloor_TouchingRuleFrees) {
  // Sliding along the spinning tilted floor at 3 m/s in each of two directions of its plane, one tolerance above it,
  // over 0.1 s and over 1/240 s: free by the touching rule, never a close call (the stepping search gave close calls
  // on every such path).
  const std::array solids = {BuildSpinningTiltedFloor(100)};
  const Vec3 start = Vec3{.x = 0.3, .y = 0, .z = 0} + kTolerance * ComputeTiltedNormal();
  for (const Vec3 direction : {Vec3{.x = 1, .y = 0, .z = 0}, ComputeAlongTiltedPlane()}) {
    for (const double step_duration : {0.1, 1.0 / 240}) {
      FirstHit first_hit;
      ContactCloseCall close_call = ContactCloseCall::kTooManyHits;
      const ContactError error = FindFirstHit(
          BuildInputs(solids, step_duration),
          start,
          3 * direction,
          0,
          step_duration,
          &first_hit,
          &close_call
      );
      EXPECT_EQ(error, ContactError::kNone) << step_duration;
      EXPECT_EQ(close_call, ContactCloseCall::kNone) << step_duration;
      ExpectFree(first_hit);
    }
  }
}

TEST(FindFirstHit, TiltedPlateTurningAboutAxisInItsPlane_IsTurning) {
  // The tilted plate turning at 1 rad/s about {1, 0, 0}, an axis in its plane: a real turn, so the resting particle
  // is hit at 0 (a turning solid has no touching rule).
  const std::array solids = {
      ToSolid(
          Plate{.center = {}, .normal = ComputeTiltedNormal(), .half_thickness = 0.01},
          {.linear_velocity = {}, .angular_velocity = {.x = 1, .y = 0, .z = 0}},
          1
      ),
  };
  const Path path = {
      .start = Vec3{.x = 0.3, .y = 0, .z = 0} + (0.01 + kTolerance) * ComputeTiltedNormal(),
      .velocity = {.x = 1, .y = 0, .z = 0},
      .duration = 0.1,
  };
  const FirstHit first_hit = FindHit(BuildInputs(solids, 0.1), path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
}

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

TEST(FindFirstHit, TurningSolidCloseCallAfterEarlierHit_IsHit) {
  // Solid 0 is the box of TurningBoxHoverOnSpinAxisBeyondAdvanceLimit with its hovering particle; solid 1, a flat
  // solid facing down from z = 0.6 and moving down at 1 m/s, reaches the particle first, at (0.6 - z) - tol s. The
  // solids that do not turn are searched first, and the box only up to that time: free in 19 advances. Searched over
  // the whole sweep, the box would end the call with a close call.
  const double start_height = 0.5 + 0x1p-17;  // m
  const std::array solids = {
      BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}}),
      ToSolid(
          FlatSolid{.point_on_surface = {.x = 0, .y = 0, .z = 0.6}, .normal_out_of_solid = {.x = 0, .y = 0, .z = -1}},
          {.linear_velocity = {.x = 0, .y = 0, .z = -1}, .angular_velocity = {}},
          2
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = start_height}, .velocity = kAtRest, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 1U);
  EXPECT_EQ(first_hit.elapsed, (0.6 - start_height) - kTolerance);
  EXPECT_NEAR(first_hit.elapsed, 0.09999236967414615, 1e-17);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  const TurningSearch capped_search = SearchTurningSolid(
      inputs,
      {.start = {.x = 0, .y = 0, .z = start_height}, .velocity = kAtRest, .duration = first_hit.elapsed}
  );
  EXPECT_EQ(capped_search.error, ContactError::kNone);
  EXPECT_EQ(capped_search.first_hit.solid_index, kNoSolid);
  EXPECT_EQ(capped_search.advance_count, 19);
}

TEST(FindFirstHit, TurningSolidHitBeforeClosedFormHit_Wins) {
  // Solid 0, a flat solid facing down from z = 0.95 and moving down at 1 m/s, would reach the particle of
  // TurningBoxReachesRestingParticle at 0.95 - tol s; solid 1, the turning box, reaches it first, at 0.8857 s. The
  // box's search, capped at the flat solid's hit, still finds it, and the earlier hit wins.
  const std::array solids = {
      ToSolid(
          FlatSolid{.point_on_surface = {.x = 0, .y = 0, .z = 0.95}, .normal_out_of_solid = {.x = 0, .y = 0, .z = -1}},
          {.linear_velocity = {.x = 0, .y = 0, .z = -1}, .angular_velocity = {}},
          1
      ),
      ToSolid(
          Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}},
          2
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = 0.6 * std::cos(0.3), .y = 0.6 * std::sin(0.3), .z = 0};
  const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = kAtRest, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 1U);
  EXPECT_NEAR(first_hit.elapsed, 0.3 + std::acos((0.5 + kTolerance) / 0.6), kTurningHitTimeTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
}

TEST(FindFirstHit, TurningSolidTiesClosedFormSolidAtZero_LowerIndexWins) {
  // Solid 0, a turning box, holds the particle at rest on its face kHighX (distance tol): a hit at 0 by the stepping
  // search, which runs second. Solid 1, a flat solid filling x <= 0.6, holds the same particle 0.1 m inside: a hit at 0
  // by its closed form, which runs first. Equal times keep the lower index whatever the order of the search.
  const std::array solids = {
      BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}}),
      ToSolid(
          FlatSolid{.point_on_surface = {.x = 0.6, .y = 0, .z = 0}, .normal_out_of_solid = {.x = 1, .y = 0, .z = 0}},
          kNoMotion,
          2
      ),
  };
  const FirstHit first_hit = FindHit(
      BuildInputs(solids, 0.1),
      {.start = {.x = 0.5 + kTolerance, .y = 0, .z = 0}, .velocity = kAtRest, .duration = 0.1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
}

TEST(ComputeEarlierHit, EqualTimes_LowerIndexWins) {
  const FirstHit kept = {.elapsed = 0.5, .solid_index = 1, .at_hit = {}};
  EXPECT_EQ(ComputeEarlierHit(kept, {.elapsed = 0.5, .solid_index = 0, .at_hit = {}}).solid_index, 0U);
  EXPECT_EQ(ComputeEarlierHit(kept, {.elapsed = 0.5, .solid_index = 2, .at_hit = {}}).solid_index, 1U);
  EXPECT_EQ(ComputeEarlierHit(kept, {.elapsed = 0.25, .solid_index = 2, .at_hit = {}}).solid_index, 2U);
  EXPECT_EQ(ComputeEarlierHit(kept, FirstHit{}).solid_index, 1U);
  EXPECT_EQ(ComputeEarlierHit(FirstHit{}, kept).solid_index, 1U);
}

TEST(ComputeSecondOrderAdvance, LargeRateTinyGap_RootWithoutCancellation) {
  // gdot = 1, K = 1 and g = 2^-60: the root is gdot + Sqrt(gdot^2 + 2 g K) = 2 exactly in doubles. The form
  // 2 g / (Sqrt(...) - gdot) subtracts equal numbers there and gives an infinite advance.
  EXPECT_EQ(ComputeSecondOrderAdvance({.gap = 0x1p-60, .gap_rate = 1, .gap_curvature_bound = 1}, 10), 2);
  EXPECT_EQ(ComputeSecondOrderAdvance({.gap = 0x1p-32, .gap_rate = 1, .gap_curvature_bound = 1}, 10), 2 + 0x1p-32);
}

TEST(ComputeSecondOrderAdvance, NoCurvature_GapOverClosingSpeedOrWholeSweep) {
  // K = 0: a closing gap is reached at g / -gdot; a gap that does not close takes the whole sweep.
  EXPECT_EQ(ComputeSecondOrderAdvance({.gap = 1, .gap_rate = -2, .gap_curvature_bound = 0}, 10), 0.5);
  EXPECT_EQ(ComputeSecondOrderAdvance({.gap = 1, .gap_rate = 0, .gap_curvature_bound = 0}, 10), 10);
  EXPECT_EQ(ComputeSecondOrderAdvance({.gap = 1, .gap_rate = 3, .gap_curvature_bound = 0}, 10), 10);
}

TEST(ComputeSecondOrderAdvance, UnderflowedRoot_GivesNoAdvance) {
  // 2 K g = 2^-1199 underflows to 0 with gdot = 0: no second-order advance, so the first-order one stands.
  EXPECT_EQ(ComputeSecondOrderAdvance({.gap = 0x1p-600, .gap_rate = 0, .gap_curvature_bound = 0x1p-600}, 10), 0);
}

TEST(HasConvexDistance, TurningShapesYes_DomainBoxNo) {
  EXPECT_TRUE(HasConvexDistance(SolidShape::kFlatSolid));
  EXPECT_TRUE(HasConvexDistance(SolidShape::kPlate));
  EXPECT_TRUE(HasConvexDistance(SolidShape::kBox));
  EXPECT_TRUE(HasConvexDistance(SolidShape::kSphere));
  EXPECT_TRUE(HasConvexDistance(SolidShape::kCylinder));
  EXPECT_FALSE(HasConvexDistance(SolidShape::kDomainBox));
}

TEST(FindHitOnTurningSolid, TurningRoom_NotConvexSoFirstOrder) {
  // A test-only room: the domain box's distance, the smallest gap to its six closed sides, turning at {0, 0, pi / 2}
  // rad/s (CheckSolid would not accept it; the search is called directly). That distance is not convex, so the
  // second-order model of one wall says nothing about the next: with it the search answered free after 2 advances,
  // with the path ending 0.5 m beyond a wall. The first-order advance alone finds the wall at 0.6693805413275088 s.
  Solid room;
  room.shape = SolidShape::kDomainBox;
  room.solid_id = 1;
  room.half_size = {.x = 1, .y = 1, .z = 1};
  room.closed = {.left = true, .right = true, .front = true, .back = true, .bottom = true, .top = true};
  room.motion.angular_velocity = {.x = 0, .y = 0, .z = std::numbers::pi / 2};
  const std::array solids = {room};
  const TurningSearch search = SearchTurningSolid(
      BuildInputs(solids, 1),
      {.start = {.x = 0.5, .y = 0.7, .z = 0}, .velocity = {.x = 1, .y = -1, .z = 0}, .duration = 1}
  );
  EXPECT_EQ(search.error, ContactError::kNone);
  EXPECT_EQ(search.first_hit.solid_index, 0U);
  EXPECT_NEAR(search.first_hit.elapsed, 0.66938054133, kTurningHitTimeTolerance);
  ExpectAtGrownSurface(search.first_hit.at_hit);
}

// A box of half size 0.5 m centred at the origin at the start of the step, moving at {1, 0, 0} m/s and turning at
// {0, 0, 1} rad/s.
Solid BuildMovingTurningBox() {
  return BuildTurningBox({.linear_velocity = {.x = 1, .y = 0, .z = 0}, .angular_velocity = {.x = 0, .y = 0, .z = 1}});
}

TEST(FindFirstHit, StartElapsed_TurningBoxHitAtZeroUsesLaterPose) {
  // Half a second in, the particle sits half a tolerance from face kHighX: a hit at 0. At the pose of the step's
  // start it would be 0.44 m away, and the box would not reach it in the quarter second of the sweep.
  const std::array solids = {BuildMovingTurningBox()};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = ToWorldPoint(solids[0], 0.5, {.x = 0.5 + kTolerance / 2, .y = 0, .z = 0});
  const FirstHit first_hit =
      FindHit(inputs, {.start = start, .velocity = kAtRest, .start_elapsed = 0.5, .duration = 0.25});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
}

TEST(FindFirstHit, StartElapsed_TurningBoxLeftBehindIsFree) {
  // The particle rests on face kLowX of the box as posed at the step's start, but half a second in the box has moved
  // 0.5 m away from it, and it keeps moving away: free. Posed at the step's start, it would be hit at 0.
  const std::array solids = {BuildMovingTurningBox()};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {
      .start = {.x = -(0.5 + kTolerance / 2), .y = 0, .z = 0},
      .velocity = kAtRest,
      .start_elapsed = 0.5,
      .duration = 0.25,
  };
  ExpectFree(FindHit(inputs, path));
}

TEST(FindFirstHit, StartElapsed_TurningBoxSearchUsesLaterPoses) {
  // The particle of TurningBoxReachesRestingParticle placed in the box's axes a quarter second in and moving with the
  // box: measured from the box it goes round the same circle, so the hit comes at the same time since the sweep's
  // start. Posed at the step's start, or at the sweep's time without start_elapsed, the samples would be wrong.
  const std::array solids = {BuildMovingTurningBox()};
  const SolidContactInputs inputs = BuildInputs(solids, 1.5);
  const Vec3 start = ToWorldPoint(solids[0], 0.25, {.x = 0.6 * std::cos(0.3), .y = 0.6 * std::sin(0.3), .z = 0});
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = start, .velocity = solids[0].motion.linear_velocity, .start_elapsed = 0.25, .duration = 1}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 0.3 + std::acos((0.5 + kTolerance) / 0.6), kTurningHitTimeTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
}

TEST(FindFirstHit, StartElapsed_TurningCylinderHitAtZeroUsesLaterPose) {
  // A cylinder along x moving at {0, 1, 0} m/s and turning about z. Half a second in, the particle sits half a
  // tolerance outside its wall: a hit at 0. At the pose of the step's start it would be 0.56 m from the wall, and the
  // sweep of a tenth of a second could not close that.
  const std::array solids = {
      BuildCylinder(
          {},
          {.x = 1, .y = 0, .z = 0},
          {.linear_velocity = {.x = 0, .y = 1, .z = 0}, .angular_velocity = {.x = 0, .y = 0, .z = 1}}
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = ToWorldPoint(solids[0], 0.5, {.x = 0, .y = 0.5 + kTolerance / 2, .z = 0.25});
  const FirstHit first_hit =
      FindHit(inputs, {.start = start, .velocity = kAtRest, .start_elapsed = 0.5, .duration = 0.1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
}

// --- Generated sweeps ---

// Fails the calling test unless the path stays farther than tol from the solid, checked at 64 times, until the hit
// (or over the whole duration when free), and a hit sits at distance tol up to rounding. Assumes start_elapsed 0.
void ExpectFirstHitWhereDistanceReachesTolerance(const Solid& solid, Path path, FirstHit first_hit, int case_index) {
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
    ExpectFirstHitWhereDistanceReachesTolerance(
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

// m. Far above the rounding of a distance at unit scale, far below the contact tolerance.
constexpr double kReferenceAllowance = 1e-13;

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
void ExpectFirstHitAgreesWithReference(
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
    ExpectFirstHitAgreesWithReference(solid, largest_distance_at_hit, path, first_hit, case_index);
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
  // tol / 1024 beyond tol (plan decision P21).
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
RimEdgeCounts CountRimEdgeMistakes(const Solid& solid, int path_count) {
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
    const RimSearch search = SearchRimFromEntry(own_path, solid.half_size, path.duration);
    const bool is_rim_entry = search.hit_time.has_hit && first_hit.at_hit.face != SolidFace::kCurved
        && first_hit.at_hit.face != SolidFace::kLowZ && first_hit.at_hit.face != SolidFace::kHighZ;
    const int count_index = is_rim_entry ? search.advance_count : kMaxRimAdvanceCount + 1;
    ++counts.hits_per_advance_count.at(static_cast<std::size_t>(count_index));
  }
  return counts;
}

// Prints one cylinder's counts of the rim edge sweeps, for the report.
void PrintRimEdgeCounts(const Solid& cylinder, const RimEdgeCounts& counts) {
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
  // The model of plan decision P21 on the same draws, for the two cylinders along z: 1, 2 or 3 advances in 7,189,
  // 32,918 and 9,893 paths; the cylinder's own axes turn the start, which may move a few between neighbours.
  constexpr std::array<int, 3> kModelHitsPerAdvanceCount = {7189, 32918, 9893};
  for (std::size_t cylinder_index = 0; cylinder_index < cylinders.size(); ++cylinder_index) {
    const Solid& cylinder = cylinders.at(cylinder_index);
    const RimEdgeCounts counts = CountRimEdgeMistakes(cylinder, 50000);
    EXPECT_EQ(counts.mistakes, 0) << "radius " << cylinder.half_size.x;
    // Nearly every path goes deep into the cylinder, so the check is about hits that must not be missed.
    EXPECT_GT(counts.must_hit, 45000) << "radius " << cylinder.half_size.x;
    PrintRimEdgeCounts(cylinder, counts);
    const bool is_along_z = cylinder_index < 2;
    for (std::size_t advance_count = 1; is_along_z && advance_count <= 3; ++advance_count) {
      const int model_count = kModelHitsPerAdvanceCount.at(advance_count - 1);
      EXPECT_NEAR(counts.hits_per_advance_count.at(advance_count), model_count, 5) << advance_count;
    }
  }
}

// The kinds of path past a cylinder's rim of plan decision P21, item 15 (a).
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
  TouchedSolid touched{};
};

// A uniform draw between low and high.
double DrawBetween(CaseGenerator& generator, double low, double high) {
  return low + (high - low) * generator.Next();
}

// A unit vector from three NextSigned, drawn again while shorter than 0.1.
Vec3 DrawUnitDirection(CaseGenerator& generator) {
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
Vec3 DrawPointBesideRim(
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
double DrawRestGapOffset(CaseGenerator& generator) {
  const int choice = static_cast<int>(std::floor(3 * generator.Next()));
  const double share = generator.NextSigned();
  const double exponent = std::floor(DrawBetween(generator, 2, 31));
  const double units = std::floor(DrawBetween(generator, -8, 9));
  if (choice == 0) {
    return 0;
  }
  return choice == 1 ? share * std::exp2(-exponent) : units * 0x1p-44;
}

// A slant of a path off a plane: 0, 0, 1e-12, 1e-6 or 1e-3, by floor(5 Next), times NextSigned.
double DrawSlant(CaseGenerator& generator) {
  constexpr std::array<double, 5> kSlants = {0, 0, 1e-12, 1e-6, 1e-3};
  const double slant = kSlants.at(static_cast<std::size_t>(std::floor(5 * generator.Next())));
  return slant * generator.NextSigned();
}

// Builds one path past the rim of a cylinder of the given half size, in its own axes. Draws, in this order: the kind
// (0.35 through the rim, 0.15 down the wall, 0.15 along the cap, 0.15 a graze, 0.1 from the corner, 0.1 anywhere, by
// one Next); the kind's own draws, as in plan decision P21's model (the speed 2^(6 Next - 3) m/s, path lengths tol
// 2^(1 + 11 Next)); and whether the touched rule applies, Next < 0.3, with an allowance of 2^-46 times the sum of the
// velocity's absolute coordinates.
RimPath BuildRimPath(CaseGenerator& generator, Vec3 half_size) {
  const double radius = half_size.x;  // m
  const double half_height = half_size.z;  // m
  const double kind_draw = generator.Next();
  RimPath path;
  Vec3 radial{};
  double cap_sign = 1;
  const auto draw_speed = [&generator] {
    return std::exp2(DrawBetween(generator, -3, 3));
  };  // m/s
  if (kind_draw < 0.35) {
    path.kind = RimPathKind::kThroughRim;
    const double wall_share = DrawBetween(generator, -2.5, 2.5);
    const double cap_share = DrawBetween(generator, -2.5, 2.5);
    const Vec3 target = DrawPointBesideRim(
        generator,
        half_size,
        {.wall_share = wall_share, .cap_share = cap_share},
        &radial,
        &cap_sign
    );
    const Vec3 direction = DrawUnitDirection(generator);
    const double length = kTolerance * std::exp2(DrawBetween(generator, 1, 12));  // m
    const double speed = draw_speed();
    path.own_path = {.start = target - length * direction, .velocity = speed * direction};
    path.duration = 2 * length / speed;
  } else if (kind_draw < 0.5) {
    path.kind = RimPathKind::kDownWall;
    const double offset = DrawRestGapOffset(generator);
    const Vec3 start = DrawPointBesideRim(
        generator,
        half_size,
        {.wall_share = 1 + offset, .cap_share = DrawBetween(generator, 2, 2000)},
        &radial,
        &cap_sign
    );
    const double speed = draw_speed();
    const double slant = DrawSlant(generator);
    path.own_path = {.start = start, .velocity = speed * (slant * radial - Vec3{.x = 0, .y = 0, .z = cap_sign})};
    const double length = std::abs(start.z) - half_height + std::min(half_height, 100 * kTolerance);  // m
    path.duration = 1.5 * length / speed;
  } else if (kind_draw < 0.65) {
    path.kind = RimPathKind::kAlongCap;
    const double offset = DrawRestGapOffset(generator);
    const Vec3 start = DrawPointBesideRim(
        generator,
        half_size,
        {.wall_share = DrawBetween(generator, 2, 2000), .cap_share = 1 + offset},
        &radial,
        &cap_sign
    );
    const double speed = draw_speed();
    const double slant = DrawSlant(generator);
    const double sideways = DrawBetween(generator, -0.5, 0.5);
    const Vec3 across = {.x = -radial.y, .y = radial.x, .z = 0};
    path.own_path = {
        .start = start,
        .velocity = speed * (sideways * across - radial + Vec3{.x = 0, .y = 0, .z = cap_sign * slant}),
    };
    const double length = (std::hypot(start.x, start.y) - radius) + std::min(radius, 100 * kTolerance);  // m
    path.duration = 1.5 * length / speed;
  } else if (kind_draw < 0.8) {
    path.kind = RimPathKind::kGraze;
    const double angle = DrawBetween(generator, 0, std::numbers::pi / 2);  // rad
    const bool is_rounding_offset = generator.Next() < 0.5;
    const double rounding_offset = std::floor(DrawBetween(generator, -64, 65)) * 0x1p-46;
    const double share_offset = DrawBetween(generator, -0.125, 0.125);
    const double offset = is_rounding_offset ? rounding_offset : share_offset;
    const RimPointShares shares = {
        .wall_share = (1 + offset) * std::cos(angle),
        .cap_share = (1 + offset) * std::sin(angle),
    };
    const Vec3 target = DrawPointBesideRim(generator, half_size, shares, &radial, &cap_sign);
    const Vec3 rim_normal = std::cos(angle) * radial + Vec3{.x = 0, .y = 0, .z = cap_sign * std::sin(angle)};
    Vec3 direction{};
    while (math::Length(direction) < 0.1) {
      const Vec3 drawn = DrawUnitDirection(generator);
      direction = drawn - math::Dot(drawn, rim_normal) * rim_normal;
    }
    direction = math::Normalized(direction);
    const double length = kTolerance * std::exp2(DrawBetween(generator, 1, 12));  // m
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
    const Vec3 start = DrawPointBesideRim(
        generator,
        half_size,
        {.wall_share = wall_share, .cap_share = cap_share},
        &radial,
        &cap_sign
    );
    const Vec3 direction = DrawUnitDirection(generator);
    const double speed = draw_speed();
    path.own_path = {.start = start, .velocity = speed * direction};
    path.duration = kTolerance * std::exp2(DrawBetween(generator, 0, 8)) / speed;
  } else {
    path.kind = RimPathKind::kAnywhere;
    Vec3 start{};
    do {
      start = {
          .x = DrawBetween(generator, -3, 3) * radius,
          .y = DrawBetween(generator, -3, 3) * radius,
          .z = DrawBetween(generator, -3, 3) * half_height,
      };
    } while (ComputeDistanceToCylinder(start, half_size, kIdentity).distance <= 2 * kTolerance);
    const Vec3 target = {
        .x = DrawBetween(generator, -1.2, 1.2) * radius,
        .y = DrawBetween(generator, -1.2, 1.2) * radius,
        .z = DrawBetween(generator, -1.2, 1.2) * half_height,
    };
    const double speed_factor = 1 + generator.Next();  // 1/s
    path.own_path = {.start = start, .velocity = speed_factor * (target - start)};
    path.duration = 1;
  }
  const Vec3 velocity = path.own_path.velocity;
  const double allowance = 0x1p-46 * (std::abs(velocity.x) + std::abs(velocity.y) + std::abs(velocity.z));  // m/s
  path.touched =
      generator.Next() < 0.3 ? TouchedSolid{.is_touched = true, .approach_allowance = allowance} : TouchedSolid{};
  return path;
}

// What the paths past a cylinder's rim gave.
struct RimPathCounts {
  int paths = 0;
  int deep_starts = 0;  // Starts deeper than tol inside, not swept.
  int hits = 0;
  int free_by_touching_rule = 0;  // Free by the touching rule on the distance (plan decisions P2 and P16).
  int free_resting = 0;  // Free because the start rests on a bounding surface (plan decision P21, item 1).
  int free_other = 0;
  int near_limit =
      0;  // Starts within rounding of a limit of the bounding surfaces' rule, where either answer is right.
  int mistakes = 0;
  std::array<int, kMaxRimAdvanceCount + 1> searches_per_advance_count{};
  double least_resting_distance = 1;  // m, the smallest true distance along a resting path.
};

// Whether a start beside a cylinder's rim rests on its wall or its cap's plane by plan decision P21's numbers, written
// here (-tol / 8, tol / 16, tol / 64), on gaps and rates computed apart from the code: in world axes, with std::hypot.
// Near a limit, within 2^-20 tol, rounding may decide either way, and is_near_limit says so.
struct RestingStart {
  bool is_beside_rim = false;
  bool is_resting = false;
  bool is_near_limit = false;
};

// Classifies one start by the decision's numbers (RestingStart).
RestingStart ClassifyRestingStart(const Solid& cylinder, Path path) {
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
  const RimSearch search = SearchRimFromEntry(own_path, cylinder.half_size, path.duration);
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
// code (ClassifyRestingStart), before the touching rule as the sweep tries it: a start that rests on a bounding
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
  const RestingStart resting = ClassifyRestingStart(cylinder, path);
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
std::pair<int, int> PrintRimPathCounts(const RimPathCounts& counts) {
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
  // Next before each path, in the six kinds of plan decision P21's model: through the rim's neighbourhood, down the
  // wall and along the cap at their rest gaps, grazing the round rim, from the rim corner, and from anywhere, a third
  // with the touched rule. The model on its own draws: no missed hit, no hit farther than tol / 1024 plus rounding, at
  // most 5 advances, 2.0 on average, and the smallest true gap along a resting path -0.126 tol.
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
  const auto [searches, advances] = PrintRimPathCounts(counts);
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
      ExpectFirstHitAgreesWithReference(solid, kTolerance, path, first_hit, offset_units);
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
bool IsOutsideGrownSolidThroughout(const Solid& solid, Path path, double end_elapsed, int* sample_count) {
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

// The lower-bound reference (IsOutsideGrownSolidThroughout) as a SweepReference, for the test contact tolerance.
bool IsOutsideByLowerBound(const SolidContactInputs& inputs, Path path, double end_elapsed, int* sample_count) {
  return IsOutsideGrownSolidThroughout(inputs.solids[0], path, end_elapsed, sample_count);
}

constexpr SweepReference kLowerBoundReference = {
    .is_outside_throughout = IsOutsideByLowerBound,
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
void CountTurningSweep(
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
  const TurningSearch search = SearchTurningSolid(inputs, path);
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
TurningSweepCounts CountGeneratedTurningSweeps(const Solid& solid) {
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
    CountTurningSweep(
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
void ExpectTurningSweepCounts(const TurningSweepCounts& counts, SweepCriteria criteria, const char* set_name) {
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
    const TurningSweepCounts counts = CountGeneratedTurningSweeps(solid);
    // Measured: no close call (first order alone: 32 to 56 per solid).
    ExpectTurningSweepCounts(counts, {.largest_close_call_count = 1}, solid_name);
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
    CountTurningSweep(BuildInputs(solids, 1.5), path, kLowerBoundReference, &counts);
  }
  // Measured: no close call, since the approach uses nearly the whole bound.
  ExpectTurningSweepCounts(counts, {.largest_close_call_count = 2}, "nearly_tight_bound");
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
Path AimAtOwnPoint(const Solid& solid, PathAim aim) {
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
PathAim DrawEdgeOrCornerAim(Vec3 half_size, CaseGenerator* generator) {
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
PathAim DrawRimAim(CaseGenerator* generator) {
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
bool StartsAwayFromSolid(const Solid& solid, Path path) {
  return ComputeDistanceToSolid(solid, ComputeSolidPose(solid, path.start_elapsed), path.start).distance
      > 2 * kTolerance;
}

// A long box of half sizes {0.5, 4, 0.5} with world axes, centred at the origin at the start of the step, moving at
// linear_velocity and turning a quarter turn per second about z: kMaxTurnPerStep in a step of 1 s.
Solid BuildQuarterTurnLongBox(Vec3 linear_velocity) {
  return ToSolid(
      Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 4, .z = 0.5}},
      {.linear_velocity = linear_velocity, .angular_velocity = {.x = 0, .y = 0, .z = std::numbers::pi / 2}},
      1
  );
}

// A path that starts r0 m from the origin of a BuildQuarterTurnLongBox moving radially at box_speed m/s, along the
// own direction {sin phi0, -cos phi0, 0} at the pose of start_elapsed, and moves outward along that line at
// outward_speed m/s relative to the box, to the end of the step. The box's arm sweeps onto it while it moves out,
// so the gap closes at nearly the full bound, and the particle is farther from the origin at the end than at the
// start: the far end of r_max is the one that bounds the late advances.
struct OutwardPath {
  double start_elapsed = 0;  // s.
  double start_radius = 0;  // m, r0.
  double outward_speed = 0;  // m/s, relative to the box.
  double start_angle = 0;  // rad, phi0.
  double box_speed = 0;  // m/s, along the same direction; negative is toward the origin's other side.
};

// Returns the box and the path an OutwardPath describes.
std::pair<Solid, Path> BuildOutwardPath(OutwardPath outward) {
  const Vec3 own_direction = {.x = std::sin(outward.start_angle), .y = -std::cos(outward.start_angle), .z = 0};
  const Mat3 axes_at_start =
      math::Rotation({.x = 0, .y = 0, .z = std::numbers::pi / 2 * outward.start_elapsed}) * kIdentity;
  const Vec3 direction = axes_at_start * own_direction;
  const Solid solid = BuildQuarterTurnLongBox(outward.box_speed * direction);
  const Vec3 origin_at_start = ComputeSolidPose(solid, outward.start_elapsed).origin;
  const Path path = {
      .start = origin_at_start + outward.start_radius * direction,
      .velocity = solid.motion.linear_velocity + outward.outward_speed * direction,
      .start_elapsed = outward.start_elapsed,
      .duration = 1 - outward.start_elapsed,
  };
  return {solid, path};
}

TEST(FindFirstHit, TurningLongBoxOutwardPath_FarEndOfRadiusBoundsAdvance) {
  // From 1 m out at the angle phi0 = 1.6316666666666666, moving outward at 3 m/s: the path ends 4 m from the origin,
  // and the arm closes on it at nearly 1.57 * r(t) m/s. A bound built from the start's distance alone would let an
  // advance carry the particle 6.4 cm into the box; the true bound stops it 1.1 tol from face kHighX at
  // 0.9562428207707441 s, after 32 first-order advances (the review's mirror). The search now takes 6 advances and
  // stops at 0.9562428207946203 s (SecondOrderSearchStoppingTimes_PinAdvanceCounts); the bound is still its floor.
  const std::array solids = {BuildQuarterTurnLongBox({})};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {
      .start = {.x = 0.9981479728118984, .y = 0.060832757388580135, .z = 0},
      .velocity = {.x = 2.994443918435695, .y = 0.1824982721657404, .z = 0},
      .duration = 1,
  };
  const FirstHit first_hit = FindHit(inputs, path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 0.9562428207707441, kTurningHitTimeTolerance);
  EXPECT_GE(first_hit.at_hit.distance, kTolerance - kReferenceAllowance);
  EXPECT_LE(first_hit.at_hit.distance, (1 + kTurningSolidHitGapShare) * kTolerance);
}

TEST(FindFirstHit, StartElapsed_MovingTurningBox_FarEndOriginAtMatchingTime) {
  // The same box moving at 10 m/s against the outward direction, a fifth of a second into the step; the particle starts
  // 1 m out at phi0 = 1.4475 and moves out at 2 m/s relative to the box for the 0.8 s left. The far end of r_max
  // must use the origin at start_elapsed + duration: the origin at duration alone is 2 m nearer the end of the path,
  // and that smaller bound carries the particle 93 micrometres into the box. Hit at 0.7981334038 s.
  const auto [solid, path] = BuildOutwardPath({
      .start_elapsed = 0.2,
      .start_radius = 1,
      .outward_speed = 2,
      .start_angle = 1.4475,
      .box_speed = -10,
  });
  const std::array solids = {solid};
  const FirstHit first_hit = FindHit(BuildInputs(solids, 1), path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 0.7981334038435, kTurningHitTimeTolerance);
  EXPECT_GE(first_hit.at_hit.distance, kTolerance - kReferenceAllowance);
  EXPECT_LE(first_hit.at_hit.distance, (1 + kTurningSolidHitGapShare) * kTolerance);
}

TEST(FindFirstHit, TurningLongBoxOutwardPathFromNearOrigin_FarEndOfRadiusBoundsCurvature) {
  // From 0.6 m out at the angle 1.68125 rad, moving outward at 3 m/s: the path ends 3.6 m from the origin, six times
  // farther than it starts, and the arm closes on it near the end. K, the bound on the gap's curvature, must use that
  // far end: built from the start's distance alone, it let the second advance carry the particle 3.8 cm into the box.
  const std::array solids = {BuildQuarterTurnLongBox({})};
  const double start_angle = 1.68125;  // rad
  const Vec3 direction = {.x = std::sin(start_angle), .y = -std::cos(start_angle), .z = 0};
  const FirstHit first_hit =
      FindHit(BuildInputs(solids, 1), {.start = 0.6 * direction, .velocity = 3 * direction, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 0.98010536766574, kTurningHitTimeTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
  ExpectAtGrownSurface(first_hit.at_hit);
}

TEST(FindFirstHit, TurningLongBoxSlidePastEndFace_CoriolisAndNearEndBoundCurvature) {
  // Just beyond the end face kHighY of the quarter-turn arm, at own y = 4.05, the particle moves with the face plus
  // 8 m/s along own x, so it slides past the face while the face's normal turns with it: both terms of K, the
  // Coriolis term counted twice and the centripetal term with the near end of r_max, are needed. Measured on the
  // mirror: K from the end's distance alone stops 1.8 mm inside, the Coriolis term once 8.7 mm, without it 2.5 cm.
  const std::array solids = {BuildQuarterTurnLongBox({})};
  const double turn_rate = std::numbers::pi / 2;  // rad/s
  const Path path = {
      .start = {.x = -0.4, .y = 4.05, .z = 0},
      .velocity = {.x = 8 - 4.05 * turn_rate, .y = -0.4 * turn_rate, .z = 0},
      .duration = 1,
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(inputs, path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 0.0815739451161, kTurningHitTimeTolerance);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighY);
  ExpectAtGrownSurface(first_hit.at_hit);
  EXPECT_EQ(SearchTurningSolid(inputs, path).advance_count, 3);
}

TEST(FindFirstHit, TurningLongBoxFaceSweepsOntoParticle_FirstOrderAdvanceIsTheFloor) {
  // At rest 0.4 m in front of face kLowX of the quarter-turn arm, 3 m along it: the face comes on nearly head on, so
  // the gap closes at almost the bound while K, built for the whole sweep, is large. Then the first-order advance,
  // g / bound, is the longer one, and the search keeps it: 3 advances. The second-order advance alone takes 4.
  const std::array solids = {BuildQuarterTurnLongBox({})};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const TurningSearch search = SearchTurningSolid(
      inputs,
      {.start = {.x = -0.5 - kTolerance - 0.4, .y = 3, .z = 0}, .velocity = kAtRest, .duration = 1}
  );
  EXPECT_EQ(search.error, ContactError::kNone);
  EXPECT_EQ(search.first_hit.solid_index, 0U);
  EXPECT_EQ(search.first_hit.at_hit.face, SolidFace::kLowX);
  EXPECT_NEAR(search.first_hit.elapsed, 0.08348195268941995, 1e-13);
  EXPECT_EQ(search.advance_count, 3);
}

TEST(FindFirstHit, SecondOrderSearchStoppingTimes_PinAdvanceCounts) {
  // The search's cost, pinned so that a change of method shows here: each case stops at the sample of a given
  // advance, and another number of advances, or another advance length, would stop elsewhere in the last tol / 4 of
  // gap before the hit. 1e-13 s allows for the last bits of Sin on another platform. The first-order advance alone
  // needed 30, 25, 113, 157 (a close call), 32 and more advances for these cases.
  constexpr double kStoppingTimeTolerance = 1e-13;  // s
  const std::array boxes = {BuildTurningBox({.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}})};
  const std::array cylinders = {
      BuildCylinder(
          {},
          {.x = 1, .y = 0, .z = 0},
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}}
      ),
  };
  const std::array long_boxes = {BuildQuarterTurnLongBox({})};
  const auto [moving_long_box, moving_long_box_path] = BuildOutwardPath({
      .start_elapsed = 0.2,
      .start_radius = 1,
      .outward_speed = 2,
      .start_angle = 1.4475,
      .box_speed = -10,
  });
  const std::array moving_long_boxes = {moving_long_box};
  struct PinnedCase {
    const char* name = "";
    SolidContactInputs inputs{};
    Path path{};
    int advance_count = 0;
    double stopping_elapsed = 0;  // s
  };
  const std::array pinned_cases = {
      PinnedCase{
          .name = "TurningBoxReachesRestingParticle",
          .inputs = BuildInputs(boxes, 1),
          .path =
              {
                  .start = {.x = 0.6 * std::cos(0.3), .y = 0.6 * std::sin(0.3), .z = 0},
                  .velocity = kAtRest,
                  .duration = 1,
              },
          .advance_count = 3,
          .stopping_elapsed = 0.8856855405066031,
      },
      PinnedCase{
          .name = "CylinderTurningAcrossAxis",
          .inputs = BuildInputs(cylinders, 1),
          .path =
              {
                  .start = {.x = 0.6 * std::cos(1.2), .y = 0.6 * std::sin(1.2), .z = 0},
                  .velocity = kAtRest,
                  .duration = 1,
              },
          .advance_count = 3,
          .stopping_elapsed = 0.21488921385342513,
      },
      PinnedCase{
          .name = "TurningBoxSlowApproachWithinAdvanceLimit",
          .inputs = BuildInputs(boxes, 1),
          .path = {.start = {.x = 0.505859375, .y = 0, .z = 0}, .velocity = kAtRest, .duration = 1},
          .advance_count = 2,
          .stopping_elapsed = 0.15235117303165863,
      },
      PinnedCase{
          .name = "TurningBoxSlowApproachBeyondFirstOrderLimit",
          .inputs = BuildInputs(boxes, 1),
          .path = {.start = {.x = 0.5029296875, .y = 0, .z = 0}, .velocity = kAtRest, .duration = 1},
          .advance_count = 2,
          .stopping_elapsed = 0.1079898627864529,
      },
      PinnedCase{
          .name = "TurningLongBoxOutwardPath",
          .inputs = BuildInputs(long_boxes, 1),
          .path =
              {
                  .start = {.x = 0.9981479728118984, .y = 0.060832757388580135, .z = 0},
                  .velocity = {.x = 2.994443918435695, .y = 0.1824982721657404, .z = 0},
                  .duration = 1,
              },
          .advance_count = 6,
          .stopping_elapsed = 0.9562428207946203,
      },
      PinnedCase{
          .name = "StartElapsed_MovingTurningBox",
          .inputs = BuildInputs(moving_long_boxes, 1),
          .path = moving_long_box_path,
          .advance_count = 5,
          .stopping_elapsed = 0.7981334038631699,
      },
  };
  for (const PinnedCase& pinned_case : pinned_cases) {
    const FirstHit first_hit = FindHit(pinned_case.inputs, pinned_case.path);
    EXPECT_NEAR(first_hit.elapsed, pinned_case.stopping_elapsed, kStoppingTimeTolerance) << pinned_case.name;
    const TurningSearch search = SearchTurningSolid(pinned_case.inputs, pinned_case.path);
    EXPECT_EQ(search.advance_count, pinned_case.advance_count) << pinned_case.name;
    EXPECT_EQ(search.first_hit.elapsed, first_hit.elapsed) << pinned_case.name;
  }
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
    } while (!StartsAwayFromSolid(solid_and_path.first, solid_and_path.second));
    const std::array solids = {solid_and_path.first};
    CountTurningSweep(BuildInputs(solids, 1), solid_and_path.second, kLowerBoundReference, &counts);
  }
  // Measured: no close call (first order alone: 118).
  ExpectTurningSweepCounts(counts, {.largest_close_call_count = 3}, "outward");
}

TEST(FindFirstHit, GeneratedTurningBoxEdgeAndCornerSweeps_AgreeWithLowerBoundReference) {
  // 4000 paths aimed within 2 tol of an edge or a corner of the grown box of half sizes {0.5, 0.25, 1}, which moves at
  // {0.25, -0.5, 0.125} m/s and turns at {0.48, -0.64, 0.3} rad/s (DrawEdgeOrCornerAim, drawn again until
  // StartsAwayFromSolid).
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
      path = AimAtOwnPoint(solid, DrawEdgeOrCornerAim(half_size, &generator));
    } while (!StartsAwayFromSolid(solid, path));
    CountTurningSweep(inputs, path, kLowerBoundReference, &counts);
  }
  // Measured: no close call (first order alone: 805, mostly paths that pass the edge or the corner sideways within a
  // few tolerances).
  ExpectTurningSweepCounts(counts, {.largest_close_call_count = 4}, "box_edges_and_corners");
}

TEST(FindFirstHit, GeneratedTiltingCylinderRimSweeps_AgreeWithLowerBoundReference) {
  // 4000 paths aimed within 2 tol of the grown rim circle of a cylinder of radius 0.5 m and half height 0.75 m along
  // the tilted axis, moving at {0.25, -0.5, 0.125} m/s and turning at {0.7, 0.4, -0.5} rad/s, which tilts its axis
  // (DrawRimAim, drawn again until StartsAwayFromSolid).
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
      path = AimAtOwnPoint(solid, DrawRimAim(&generator));
    } while (!StartsAwayFromSolid(solid, path));
    CountTurningSweep(inputs, path, kLowerBoundReference, &counts);
  }
  // Measured: no close call (first order alone: 291).
  ExpectTurningSweepCounts(counts, {.largest_close_call_count = 4}, "cylinder_rims");
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
    CountTurningSweep(inputs, {.start = start, .velocity = velocity, .duration = 1}, kLowerBoundReference, &counts);
  }
  // Measured: no close call; 5.9 advances on average, 18 at the 99th percentile, 55 at most.
  ExpectTurningSweepCounts(counts, {.largest_close_call_count = 3}, "end_face_slide");
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
    } while (!StartsAwayFromSolid(solid_and_path.first, solid_and_path.second));
    const std::array solids = {solid_and_path.first};
    CountTurningSweep(BuildInputs(solids, 1), solid_and_path.second, kLowerBoundReference, &counts);
  }
  // Measured: no close call; 5.5 advances on average, 7 at most.
  ExpectTurningSweepCounts(counts, {.largest_close_call_count = 3}, "inward");
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
bool IsOutsideBoxThroughout(const SolidContactInputs& inputs, Path path, double end_elapsed, int* sample_count) {
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
  // the face. Reference: IsOutsideBoxThroughout, allowance tol / 64.
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
      .is_outside_throughout = IsOutsideBoxThroughout,
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
    CountTurningSweep(
        inputs,
        {.start = start, .velocity = velocity, .duration = kStepDuration},
        reference,
        &counts_per_kind.at(kind)
    );
  }
  // Measured: no close call of any kind; 5.5, 6.3 and 7.9 advances on average, 36, 32 and 41 at the 99th percentile,
  // 119 at most. The first-order advance alone gave close calls on most of these paths.
  for (std::size_t kind = 0; kind < 3; ++kind) {
    ExpectTurningSweepCounts(
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
