#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>

#include "fluidity/base/mat3.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_test_cases.h"
#include "fluidity/contact/solid_contact.h"
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

TEST(FindFirstHit, CloseCall_LeavesFirstHitUnchanged) {
  // A particle at rest two tolerances from a box turning at 1 rad/s: the face sweeps toward it slower than any safe
  // search allows, so the step is too close to call, and the first hit keeps what it held.
  const std::array solids = {
      ToSolid(
          Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}},
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}},
          1
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 0.1);
  FirstHit first_hit = {.elapsed = 7, .solid_index = 3, .at_hit = {}};
  ContactCloseCall close_call = ContactCloseCall::kNone;
  const ContactError error =
      FindFirstHit(inputs, {.x = 0.5 + 2 * kTolerance, .y = 0, .z = 0}, kAtRest, 0, 0.1, &first_hit, &close_call);
  EXPECT_EQ(error, ContactError::kTooCloseToCall);
  EXPECT_EQ(close_call, ContactCloseCall::kSlowApproachToTurningSolid);
  EXPECT_EQ(first_hit.elapsed, 7);
  EXPECT_EQ(first_hit.solid_index, 3U);
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
  // from 0.45 s on, whatever the rounding.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {
      .start = {.x = 1.8500000009313227, .y = 0, .z = 1.4500000009313225},
      .velocity = {.x = -3, .y = 0, .z = -1},
      .duration = 1,
  };
  const FirstHit first_hit = FindHit(inputs, path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 0.45, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  EXPECT_GE(first_hit.at_hit.distance, kTolerance - 0x1p-40);
  EXPECT_LE(first_hit.at_hit.distance, std::numbers::sqrt2 * kTolerance + 0x1p-40);
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
  // at 2 - tol s. There the wall and cap distances are tol / 2 and tol: the point faces the high rim, at the true
  // distance Sqrt(1.25) tol, the sharp corner of the grown shape (plan P15). The rim move is MoveParticle's.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0.5 + kTolerance / 2, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  ExpectVec3Near(first_hit.at_hit.normal_out_of_solid, Vec3{.x = 1, .y = 0, .z = 2} * (1 / std::sqrt(5.0)), 1e-8);
  EXPECT_NEAR(first_hit.at_hit.distance, std::sqrt(1.25) * kTolerance, 0x1p-40);
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
  // in its rim corner (plan P15). Approaching the rim is a hit at 0; moving away is free.
  const Vec3 corner_start = {.x = 0.5 + 0.9 * kTolerance, .y = 0, .z = 1 + 0.9 * kTolerance};
  const FirstHit approaching =
      FindHit(inputs, {.start = corner_start, .velocity = {.x = -1, .y = 0, .z = 0}, .duration = 1});
  EXPECT_EQ(approaching.solid_index, 0U);
  EXPECT_EQ(approaching.elapsed, 0);
  EXPECT_EQ(approaching.at_hit.face, SolidFace::kHighRim);
  EXPECT_NEAR(approaching.at_hit.distance, 0.9 * std::numbers::sqrt2 * kTolerance, 0x1p-40);
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

TEST(FindFirstHit, CylinderTurningAcrossItsAxis_IsCloseCall) {
  // A cylinder along x turning about z: its axis tilts, so it is a turning solid, which this code cannot sweep yet.
  const std::array solids = {
      BuildCylinder(
          {},
          {.x = 1, .y = 0, .z = 0},
          {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}}
      ),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  FirstHit first_hit = {.elapsed = 7, .solid_index = 3, .at_hit = {}};
  ContactCloseCall close_call = ContactCloseCall::kNone;
  const ContactError error = FindFirstHit(
      inputs,
      {.x = 0.6 * std::cos(1.2), .y = 0.6 * std::sin(1.2), .z = 0},
      kAtRest,
      0,
      1,
      &first_hit,
      &close_call
  );
  EXPECT_EQ(error, ContactError::kTooCloseToCall);
  EXPECT_EQ(close_call, ContactCloseCall::kSlowApproachToTurningSolid);
  EXPECT_EQ(first_hit.elapsed, 7);
  EXPECT_EQ(first_hit.solid_index, 3U);
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

TEST(FindFirstHit, CylinderTurningAcrossAboveBound_IsCloseCall) {
  // (tol / 32) * 2 * 3 = 0.1875 tol > tol / 8: a turning solid.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 32)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  FirstHit first_hit;
  ContactCloseCall close_call = ContactCloseCall::kNone;
  const ContactError error =
      FindFirstHit(inputs, {.x = -4, .y = 0, .z = 0}, {.x = 1, .y = 0, .z = 0}, 0, 2, &first_hit, &close_call);
  EXPECT_EQ(error, ContactError::kTooCloseToCall);
  EXPECT_EQ(close_call, ContactCloseCall::kSlowApproachToTurningSolid);
}

TEST(FindFirstHit, CylinderTurningAcrossAboveBound_ShortSweepIsCloseCall) {
  // The bound counts the whole step, not the sweep: a sweep of a quarter of the step from half a second in is still
  // a turning solid, although (tol / 32) * 0.5 * 3 = 0.047 tol would be below tol / 8.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 32)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  FirstHit first_hit;
  ContactCloseCall close_call = ContactCloseCall::kNone;
  const ContactError error =
      FindFirstHit(inputs, {.x = -4, .y = 0, .z = 0}, {.x = 1, .y = 0, .z = 0}, 0.5, 0.5, &first_hit, &close_call);
  EXPECT_EQ(error, ContactError::kTooCloseToCall);
  EXPECT_EQ(close_call, ContactCloseCall::kSlowApproachToTurningSolid);
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

TEST(FindFirstHit, CylinderGrownDiscEdgeMetExactly_IsHit) {
  // Down the axis direction at exactly the grown radius 0.5 + tol: the squared radius at the cap equals R * R, bit
  // for bit, and the edge counts. The hit faces the high rim at Sqrt(2) tol.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0.5 + kTolerance, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  EXPECT_NEAR(first_hit.at_hit.distance, std::numbers::sqrt2 * kTolerance, 0x1p-40);
}

TEST(FindFirstHit, CylinderGrownWallTopMetExactly_IsHit) {
  // Across the axis at exactly the grown half height 1 + tol: the height at the wall equals H, and the edge counts.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = -2, .y = 0, .z = 1 + kTolerance}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  ExpectNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
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
  // {1 + 0.9 tol, 0, 1 + 0.9 tol} is in its rim corner and the rim moves into it: hit at 0. Posed at the start of the
  // step, the start would be half a metre outside the wall and the hit about half a second later.
  const std::array solids = {
      BuildCylinder({}, kAxisZ, {.linear_velocity = {.x = 1, .y = 0, .z = 0}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {
      .start = {.x = 1 + 0.9 * kTolerance, .y = 0, .z = 1 + 0.9 * kTolerance},
      .velocity = kAtRest,
      .start_elapsed = 0.5,
      .duration = 0.5,
  };
  const FirstHit first_hit = FindHit(inputs, path);
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
}

TEST(FindFirstHit, StartElapsed_CylinderRimCornerMovingWithCylinderIsFree) {
  // The same start, moving with the cylinder and away from its rim.
  const std::array solids = {
      BuildCylinder({}, kAxisZ, {.linear_velocity = {.x = 1, .y = 0, .z = 0}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {
      .start = {.x = 1 + 0.9 * kTolerance, .y = 0, .z = 1 + 0.9 * kTolerance},
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
};

// The six faces: the wall from two sides, both caps, and both rims entered through the sharp corner of the grown
// cylinder, where the entries into the grown wall and between the grown caps tie at 1 s and the true distance is
// Sqrt(2) tol.
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
              .hit_elapsed = 1,
              .own_normal_out_of_solid = {.x = rim_component, .y = 0, .z = rim_component},
              .normal_tolerance = kRimNormalTolerance,
              .distance_at_hit = std::numbers::sqrt2 * kTolerance,
          },
          {
              .face = SolidFace::kLowRim,
              .own_start = {.x = 0, .y = -(1.5 + kTolerance), .z = -(2 + kTolerance)},
              .own_velocity = {.x = 0, .y = 1, .z = 1},
              .hit_elapsed = 1,
              .own_normal_out_of_solid = {.x = 0, .y = -rim_component, .z = -rim_component},
              .normal_tolerance = kRimNormalTolerance,
              .distance_at_hit = std::numbers::sqrt2 * kTolerance,
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
    EXPECT_NEAR(first_hit.at_hit.distance, face_case.distance_at_hit, 0x1p-40) << static_cast<int>(face_case.face);
  }
}

TEST(FindFirstHit, MovingCylinderEachFace_HitsThatFace) {
  ExpectEachCylinderFaceHit(kAxisZ);
}

TEST(FindFirstHit, TiltedMovingCylinderEachFace_HitsThatFace) {
  ExpectEachCylinderFaceHit(ToColumn(kTiltedAxes, 2));
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
// largest_distance_at_hit (tol for a sphere, Sqrt(2) tol for a cylinder, whose grown shape has sharp rims). A hit
// is never behind the start nor after the duration.
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
  // spinning about its own axis, which leaves the shape. Its first hits are on the grown cylinder with sharp rims, so
  // a hit is up to Sqrt(2) tol from the true rim.
  const Vec3 axis = ToColumn(kTiltedAxes, 2);
  const std::array solids = {
      ToSolid(
          Cylinder{.center = {.x = 0.25, .y = 0.5, .z = -0.125}, .axis = axis, .radius = 0.5, .half_height = 0.75},
          {.linear_velocity = {.x = 0.25, .y = -0.5, .z = 0.125}, .angular_velocity = 3 * axis},
          1
      ),
  };
  ASSERT_EQ(CheckSolid(solids[0], 1.5), ContactError::kNone);
  const int hit_count = CheckGeneratedRoundSweeps(solids[0], std::numbers::sqrt2 * kTolerance);
  EXPECT_GT(hit_count, 100);
  EXPECT_LT(hit_count, 900);
}

// Counts the paths through the grown rim edge of one cylinder that FindFirstHit gets wrong, against the slow
// reference. Draw order per path: azimuth 2 pi Next(); angle in the plane of the axis 0.05 + (pi / 2 - 0.1) Next();
// distance before the edge 0.5 + 1.5 Next(); the cap, high when Next() < 0.5; an offset of the line from the edge of
// -8 to 8 units of 2^-53, as floor(17 Next()) - 8; speed 0.5 + 1.5 Next(). In the cylinder's own axes the path passes
// within that offset of the grown rim circle {R cos(azimuth), R sin(azimuth), +-H}, going toward the axis and the
// middle plane, for three times the time it takes to reach the edge. Returns the number of mistakes; *must_hit_count
// counts the paths the reference says must be hit (their true distance goes below tol - kReferenceAllowance).
int CountRimEdgeMistakes(const Solid& solid, int path_count, int* must_hit_count) {
  const std::array solids = {solid};
  // The longest sweep is 3 * 2 / 0.5 = 12 s.
  const SolidContactInputs inputs = BuildInputs(solids, 12);
  const Mat3 axes = solid.pose_at_step_start.axes;
  const double grown_radius = solid.half_size.x + kTolerance;
  const double grown_half_height = solid.half_size.z + kTolerance;
  CaseGenerator generator;
  int mistake_count = 0;
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
    if (must_hit) {
      ++*must_hit_count;
    }
    if (first_hit.solid_index == kNoSolid) {
      mistake_count += must_hit ? 1 : 0;
      continue;
    }
    const double distance_at_hit = ComputeDistanceAlongPath(solid, path, first_hit.elapsed);
    const bool is_at_grown_surface = distance_at_hit >= kTolerance - kReferenceAllowance
        && distance_at_hit <= std::numbers::sqrt2 * kTolerance + kReferenceAllowance;
    const bool is_first = first_hit.elapsed == 0
        || ComputeSmallestDistanceAlongPath(solid, path, first_hit.elapsed) >= kTolerance - kReferenceAllowance;
    mistake_count += is_at_grown_surface && is_first ? 0 : 1;
  }
  return mistake_count;
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
  for (const Solid& cylinder : cylinders) {
    int must_hit_count = 0;
    EXPECT_EQ(CountRimEdgeMistakes(cylinder, 50000, &must_hit_count), 0) << "radius " << cylinder.half_size.x;
    // Nearly every path goes deep into the cylinder, so the check is about hits that must not be missed.
    EXPECT_GT(must_hit_count, 45000) << "radius " << cylinder.half_size.x;
  }
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

}  // namespace
}  // namespace fluidity::contact
