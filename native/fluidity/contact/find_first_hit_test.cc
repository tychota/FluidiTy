#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

}  // namespace
}  // namespace fluidity::contact
