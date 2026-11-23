#include "fluidity/contact/solid_contact_steps.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <string>

#include "fluidity/base/mat3.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/solid_contact_steps_find_first_hit_cases.h"

namespace fluidity::contact {
namespace {
constexpr Vec3 kAtRest = {.x = 0, .y = 0, .z = 0};  // m/s.

// The ground: the flat solid z <= 0, which does not move.
Solid BuildGround(std::uint32_t solid_id) {
  return ToSolid(
      FlatSolid{.point_on_surface = {}, .normal_out_of_solid = {.x = 0, .y = 0, .z = 1}},
      kNoMotion,
      solid_id
  );
}

// Fails the calling test unless |actual - expected| <= 2^-44 * max(|actual|, |expected|, reference): the relative
// tolerance of the contact tests, a few hundred rounding units of the larger value: room for the rounding of a closed
// form, far below what a wrong formula gives. reference is the largest input of the test (1 for times of
// about a second).
void CheckNearRelative(double actual, double expected, double reference) {
  const double scale = std::max({std::abs(actual), std::abs(expected), reference});
  EXPECT_LE(std::abs(actual - expected), 0x1p-44 * scale) << actual << " vs " << expected;
}

// Fails the calling test unless the path is free: no solid, and the other fields at their defaults.
void CheckFree(FirstHit first_hit) {
  const DistanceToSolid defaults{};
  EXPECT_EQ(first_hit.solid_index, kNoSolid);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.distance, defaults.distance);
  EXPECT_EQ(first_hit.at_hit.face, defaults.face);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, defaults.normal_out_of_solid, 0);
}

// Fails the calling test unless a distance reported at a rim hit is where the rim search stops: at least tol and at
// most tol + tol / 1024 (kRimHitGapShare), up to rounding.
void CheckRimHitDistance(double distance) {
  EXPECT_GE(distance, kTolerance - 0x1p-40);
  EXPECT_LE(distance, kTolerance + kTolerance / 1024 + 0x1p-40);
}

// --- The examples of the method paper ---

TEST(FindFirstHit, NoSolids_IsFree) {
  const SolidContactInputs inputs = BuildInputs({}, 1);
  CheckFree(FindHit(inputs, {.start = {}, .velocity = {.x = 1, .y = 2, .z = 3}, .start_elapsed = 0, .duration = 1}));
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
  CheckNearRelative(first_hit.elapsed, 0.1 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
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
  CheckFree(FindHit(
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
  // Plate through the origin, normal +z, half thickness 0.01; the start is inside it at z = -0.005, moving at +1 m/s.
  // Seen by an uncommitted particle the plate has two faces and the nearer is kLowZ: distance -0.005, a start deeper
  // than tol inside, so a hit at 0 on kLowZ. Committed to kAlongNormal, the plate is the half space below its kHighZ
  // face (method paper <sec-plate-sides>): distance -0.005 - 0.01 = -0.015, also a hit at 0, but on kHighZ with normal
  // +z. Not free in the committed case: the committed half space contains the whole thickness, so a committed particle
  // can never be freer than an uncommitted one, and only the face tells the two apart.
  const std::array solids = {
      ToSolid(Plate{.center = {}, .normal = {.x = 0, .y = 0, .z = 1}, .half_thickness = 0.01}, kNoMotion, 5),
  };
  SolidContactInputs inputs = BuildInputs(solids, 1);
  const Path path = {.start = {.x = 0, .y = 0, .z = -0.005}, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 1};

  const FirstHit uncommitted = FindHit(inputs, path);
  EXPECT_EQ(uncommitted.solid_index, 0U);
  EXPECT_EQ(uncommitted.elapsed, 0);
  EXPECT_EQ(uncommitted.at_hit.face, SolidFace::kLowZ);
  CheckVec3IsNear(uncommitted.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = -1}, 0);
  EXPECT_EQ(uncommitted.at_hit.distance, 0.005 - 0.01);

  inputs.plate_side = {.plate_id = 5, .side = SolidSide::kAlongNormal};
  const FirstHit committed = FindHit(inputs, path);
  EXPECT_EQ(committed.solid_index, 0U);
  EXPECT_EQ(committed.elapsed, 0);
  EXPECT_EQ(committed.at_hit.face, SolidFace::kHighZ);
  CheckVec3IsNear(committed.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = 1}, 0);
  EXPECT_EQ(committed.at_hit.distance, -0.005 - 0.01);
}

TEST(FindFirstHit, DomainBoxOpenTop_ExitIsFree) {
  const std::array solids = {ToSolid(DomainBox{.low_corner = {}, .high_corner = {.x = 1, .y = 1, .z = 1}}, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  CheckFree(
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = 1, .y = 0, .z = 0}, 0);
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
  CheckFree(FindHit(
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
  CheckFree(FindHit(
      inputs,
      {.start = {.x = -1, .y = 0.5 + kTolerance, .z = 0}, .velocity = {.x = 1, .y = 1, .z = 0}, .duration = 1}
  ));
}

TEST(FindFirstHit, SlidingOnTouchedFace_IsNotAHit) {
  // Resting at z = tol (gap 0) and sinking at tol / 4 m/s for 1 s: the path goes at most tol / 4 below the start, so
  // it cannot cross the surface. The touching rule frees it.
  const std::array solids = {BuildGround(1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  CheckFree(FindHit(
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = 1}, 0);
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
    CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, face_case.normal_out_of_solid, 0);
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
    CheckNearRelative(first_hit.elapsed, (1 - kTolerance) / 2, 1);
    EXPECT_EQ(first_hit.at_hit.face, face_case.face);
    CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, face_case.normal_out_of_solid, 0);
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
    CheckNearRelative(first_hit.elapsed, (1 - kTolerance) / 2, 1);
    EXPECT_EQ(first_hit.at_hit.face, face_case.face);
    CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, face_case.normal_out_of_solid, 1e-15);
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
  CheckNearRelative(first_hit.elapsed, (1 - kTolerance) / 2, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, normal, 1e-15);
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
    CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, -face_case.normal_out_of_solid, 0);
    EXPECT_EQ(first_hit.at_hit.distance, kTolerance) << static_cast<int>(face_case.face);
  }
}

TEST(FindFirstHit, DomainBoxCorner_RestingOnTwoSides_IsFree) {
  // Resting one tolerance from side_left and side_front, sinking toward both at tol / 4 m/s and rising: each closed
  // side is its own flat face with its own touching rule, so neither is hit.
  const std::array solids = {ToSolid(DomainBox{.low_corner = {}, .high_corner = {.x = 1, .y = 1, .z = 1}}, 1)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 velocity = {.x = -kTolerance / 4, .y = -kTolerance / 4, .z = 0.25};
  CheckFree(
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 1, .z = 0}, 0);
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
  CheckFree(FindHit(
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
  CheckFree(FindHit(inputs, {.start = {}, .velocity = {.x = 3, .y = 4, .z = 0}, .duration = 1}));
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
  CheckFree(FindHit(
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
  CheckFree(FindHit(
      inputs,
      {.start = {.x = 0, .y = 0, .z = 0.5 + kTolerance}, .velocity = {.x = 2, .y = 0, .z = 0}, .duration = 1}
  ));
}

TEST(FindFirstHit, BoxRestingOnEdge_RestsOnTheOtherFacePlane) {
  // One tolerance from both kHighX and kHighZ: their gaps tie at 0 and the tie keeps kHighX, which gives the distance.
  // Sinking along the x face is free. Pushing into kHighX is free too: the start rests on the plane of kHighZ, which
  // the sweep tries first (method paper <sec-touching-rule>; with the touching rule on the distance first, a hit at 0
  // on kHighX). Pushing in while sinking into the top's plane by tol / 32 does not rest on it, so the distance's face
  // rules: a hit at 0 on kHighX.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = 0.5 + kTolerance, .y = 0, .z = 0.5 + kTolerance};
  CheckFree(FindHit(inputs, {.start = start, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 1}));
  CheckFree(FindHit(inputs, {.start = start, .velocity = {.x = -1, .y = 0, .z = 0}, .duration = 1}));
  const FirstHit first_hit =
      FindHit(inputs, {.start = start, .velocity = {.x = -1, .y = 0, .z = -kTolerance / 32}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighX);
}

TEST(FindFirstHitOfSweep, BoxBesideItLevelWithItsTop_FreeAlsoAfterAResponse) {
  // 33 / 32 tol beyond kHighX, level with the top's plane, moving inward after a response: the sweep tries the top's
  // plane first and rests on it, so it is free (read first, the touched rule would take kHighX, 1.03 tol away, as
  // approached: a hit at 0 and a dead stop).
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Sweep sweep = {
      .start = {.x = 0.5 + 33.0 / 32 * kTolerance, .y = 0, .z = 0.5 + kTolerance},
      .velocity_for_this_step = {.x = -1, .y = 0, .z = 0},
      .duration = 1,
  };
  const TouchedRule touched = {.is_touched = true, .approach_allowance = 0x1p-46};
  FirstHit first_hit;
  ASSERT_EQ(FindFirstHitOfSweep(inputs, sweep, touched, &first_hit, nullptr), ContactError::kNone);
  CheckFree(first_hit);
}

TEST(FindFirstHit, BoxDownItsGrownSidePlanePastTheEdge_IsFree) {
  // Down the plane of kHighX at its rest gap, past the top edge: kHighZ gives the distance, and the start rests on the
  // plane of kHighX, which the whole box lies behind, so the sweep is free (method paper <sec-touching-rule>). Without
  // that rule the path enters the grown box through the plane of kHighZ at 2.5 - tol s and stops dead at the box's
  // side.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  CheckFree(FindHit(
      inputs,
      {.start = {.x = 0.5 + kTolerance, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4}
  ));
}

TEST(FindFirstHit, BoxBoundingSurfaceWindowAndSink) {
  // Down the plane of kHighX from {x, 0, 3}. A start tol / 8 inside that plane's rest gap rests on it; 3 tol / 16
  // inside does not, and kHighZ is entered at 2.5 - tol s exactly. Sinking toward the plane by tol / 128 during the
  // sweep still rests; by tol / 32 it does not. Just above the rest gap (tol / 128) and sinking by tol / 64, the start
  // rests too, where the plane of kHighZ was entered at 1.25 - tol / 2 s.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const Vec3 down = {.x = 0, .y = 0, .z = -1};
  const double rest_x = 0.5 + kTolerance;  // m
  const auto expect_top_hit = [](FirstHit first_hit, double elapsed) {
    EXPECT_EQ(first_hit.solid_index, 0U);
    EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
    EXPECT_EQ(first_hit.elapsed, elapsed);
  };
  CheckFree(
      FindHit(inputs, {.start = {.x = rest_x - kTolerance / 8, .y = 0, .z = 3}, .velocity = down, .duration = 4})
  );
  expect_top_hit(
      FindHit(inputs, {.start = {.x = rest_x - 3 * kTolerance / 16, .y = 0, .z = 3}, .velocity = down, .duration = 4}),
      2.5 - kTolerance
  );
  const Vec3 rest_start = {.x = rest_x, .y = 0, .z = 3};
  CheckFree(
      FindHit(inputs, {.start = rest_start, .velocity = {.x = -kTolerance / 512, .y = 0, .z = -1}, .duration = 4})
  );
  expect_top_hit(
      FindHit(inputs, {.start = rest_start, .velocity = {.x = -kTolerance / 128, .y = 0, .z = -1}, .duration = 4}),
      2.5 - kTolerance
  );
  CheckFree(FindHit(
      inputs,
      {
          .start = {.x = rest_x + kTolerance / 128, .y = 0, .z = 3},
          .velocity = {.x = -kTolerance / 128, .y = 0, .z = -2},
          .duration = 2,
      }
  ));
}

TEST(FindFirstHit, BoxFaceThatGivesTheDistance_KeepsItsOwnTouchingRule) {
  // tol / 128 above the top's rest gap, sinking by tol / 128 over 1 s while moving along it: kHighZ gives the distance,
  // so the plain closed form decides (the touching rule of method paper <sec-touching-rule> alone) and its plane is
  // entered at 1 s exactly.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit = FindHit(
      inputs,
      {
          .start = {.x = 0, .y = 0, .z = 0.5 + kTolerance + kTolerance / 128},
          .velocity = {.x = 0.25, .y = 0, .z = -kTolerance / 128},
          .duration = 1,
      }
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  EXPECT_EQ(first_hit.elapsed, 1);
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
  CheckFree(FindHit(
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
  CheckFree(FindHit(inputs, {.start = {.x = 0, .y = 0, .z = 1}, .velocity = {.x = 3, .y = 4, .z = 0}, .duration = 10}));
}

TEST(FindFirstHit, PathParallelAboveBoxFace_IsFree) {
  // Above the top face (gap 0.25 - tol > 0) and moving along it: that face is never reached, although the path
  // crosses the x range of the box.
  const std::array solids = {
      ToSolid(Box{.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}}, kNoMotion, 1),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  CheckFree(
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
  CheckFree(FindHit(inputs, {.start = start, .velocity = {.x = 1, .y = 1, .z = 0}, .duration = 1}));
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = -1}, 0);
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
  EXPECT_EQ(close_call, ContactCloseCall::kHoverNearTurningSolid);
  EXPECT_EQ(first_hit.solid_index, 1U);
  EXPECT_EQ(first_hit.elapsed, 0);
  EXPECT_EQ(first_hit.at_hit.distance, 0);
}

TEST(FindFirstHit, SphereHeadOn_HitsAtGapOverSpeed) {
  // From 2 m in front of the centre at 1 m/s, the grown sphere of radius 0.5 + tol is reached at 1.5 - tol.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  CheckNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

TEST(FindFirstHit, SphereGrazing_HitsAtQuadraticRoot) {
  // 0.3 m off the centre line: x meets -Sqrt((0.5 + tol)^2 - 0.09), near -0.4, so the normal is near {-0.8, 0.6, 0}.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -2, .y = 0.3, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  CheckNearRelative(first_hit.elapsed, 2 - std::sqrt((0.5 + kTolerance) * (0.5 + kTolerance) - 0.09), 1);
  CheckNearRelative(first_hit.elapsed, 1.5999999988358469, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -0.8, .y = 0.6, .z = 0}, 1e-8);
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
    CheckFree(first_hit);
    return;
  }
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 2, 0x1p-20);
  EXPECT_NEAR(first_hit.at_hit.normal_out_of_solid.x, 0, 0x1p-20);
}

TEST(FindFirstHit, SphereMiss_IsFree) {
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  CheckFree(
      FindHit(inputs, {.start = {.x = -2, .y = 0.6, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2})
  );
}

TEST(FindFirstHit, SphereMovingAway_IsFree) {
  // Dot(start, velocity) = 2 >= 0: the path does not approach the centre.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  CheckFree(
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0}, .velocity = {.x = -1, .y = 0, .z = 0}, .duration = 2})
  );
}

TEST(FindFirstHit, SphereTouchingTangentPath_IsNotAHit) {
  // At distance tol (gap 0) and moving along the tangent plane: rate 0, so the touching rule frees it (method paper
  // <sec-round-starts>).
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  CheckFree(FindHit(
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
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
  CheckNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

TEST(FindFirstHit, SpinningSphere_IsClosedForm) {
  // A spin about the centre does not change the shape (method paper <sec-slide-and-spin>): the head-on time of the
  // sphere at rest, at the start of the step and a quarter of a second into it, when the spin has turned the sphere's
  // axes by 1.25 rad.
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
    CheckNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
    EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
    CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
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
    CheckNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
    EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
    CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, direction, 1e-8);
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
  CheckNearRelative(first_hit.elapsed, kTolerance / 0.7, kTolerance);
}

TEST(FindFirstHit, SphereStartJustInsideRest_PushingInHitsAtZero) {
  // At distance tol / 2 the gap is -tol / 2, inside the touching window.
  const std::array solids = {BuildSphere({}, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = 0, .y = 0, .z = 0.5 + kTolerance / 2};
  CheckFree(FindHit(inputs, {.start = start, .velocity = {.x = 1, .y = 0, .z = -kTolerance / 4}, .duration = 1}));
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
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
  CheckFree(FindHit(inputs, path_ending_before_hit));
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = 1}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, kTolerance);
}

TEST(FindFirstHit, StartElapsed_SphereRisingWithParticleIsFree) {
  // The same start, now rising with the sphere and sliding sideways: measured from the sphere it moves along the
  // tangent plane, so the touching rule frees it.
  const std::array solids = {
      BuildSphere({}, {.linear_velocity = {.x = 0, .y = 0, .z = 1}, .angular_velocity = {}}),
  };
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  CheckFree(FindHit(
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 0);
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
  CheckNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

// --- A cylinder ---

TEST(FindFirstHit, CylinderWall_HitsAtRadialGap) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0.25}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  CheckNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
  EXPECT_NEAR(first_hit.at_hit.distance, kTolerance, 0x1p-40);
}

TEST(FindFirstHit, CylinderCap_HitsAtHeightGap) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 0.1, .y = 0.2, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4});
  EXPECT_EQ(first_hit.solid_index, 0U);
  CheckNearRelative(first_hit.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = 1}, 1e-8);
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
  CheckNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
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
  CheckNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
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
  CheckRimHitDistance(first_hit.at_hit.distance);
}

TEST(FindFirstHit, CylinderParallelToAxisOutside_IsFree) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  CheckFree(
      FindHit(inputs, {.start = {.x = 0.6, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4})
  );
}

TEST(FindFirstHit, CylinderParallelToAxisInGrownRadius_HitsRim) {
  // The path never meets the wall (no motion across the axis) and enters the grown disc at {0.5 + tol / 2, 0, 1 + tol}
  // at 2 - tol s, beside the high rim, Sqrt(1.25) tol from it: the sharp corner of the grown shape. The rim search goes
  // on to where the true distance is tol, Sqrt(0.75) tol above the cap's plane, in 2 advances (method paper
  // <sec-round-first-hit>).
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit = FindHit(
      inputs,
      {.start = {.x = 0.5 + kTolerance / 2, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4}
  );
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 2 - std::sqrt(0.75) * kTolerance, kTolerance / 1024);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = 0.5, .y = 0, .z = std::sqrt(0.75)}, 1e-5);
  CheckRimHitDistance(first_hit.at_hit.distance);
}

TEST(FindFirstHit, CylinderRimHeadOnAtFortyFiveDegrees_HitsAtOneTolerance) {
  // Along the rim's own normal {1, 0, 1} / Sqrt(2): the entry is the corner {0.5 + tol, 0, 1 + tol}, Sqrt(2) tol from
  // the rim, and one advance of the gap over its rate lands at tol, at 1 - Sqrt(0.5) tol s (the sharp corner alone
  // would give 1 - tol s).
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = 1.5, .y = 0, .z = 2}, .velocity = {.x = -1, .y = 0, .z = -1}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_NEAR(first_hit.elapsed, 1 - std::sqrt(0.5) * kTolerance, kTolerance / 1024);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = std::sqrt(0.5), .y = 0, .z = std::sqrt(0.5)}, 1e-5);
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
    CheckFree(FindHit(inputs, {.start = start, .velocity = velocity, .duration = 0.5}));
  }
  const double offset = 22.0 / 32 * kTolerance;  // m
  const Vec3 start = {.x = 0.75 + offset, .y = 0, .z = 0.75 + offset};
  const FirstHit first_hit = FindHit(inputs, {.start = start, .velocity = velocity, .duration = 0.5});
  EXPECT_EQ(first_hit.solid_index, 0U);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighRim);
  CheckRimHitDistance(first_hit.at_hit.distance);
}

TEST(FindFirstHit, CylinderParallelToCapsAbove_IsFree) {
  // The path is inside the grown wall from 1.5 - tol s, but never between the grown caps' planes: free.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  CheckFree(
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 1.5}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 4})
  );
}

TEST(FindFirstHit, CylinderTouchingWallTangentPath_IsNotAHit) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  CheckFree(FindHit(
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
  CheckFree(FindHit(inputs, {.start = touching_start, .velocity = {.x = 1, .y = 0, .z = 1}, .duration = 1}));
  // Wall and cap distances 0.9 tol: inside the grown sharp cylinder, but at the true distance 0.9 Sqrt(2) tol > tol,
  // in its rim corner. The entry is 0 and the rim search decides: along the rim's normal the true distance reaches tol
  // after (0.9 - Sqrt(0.5)) tol s, one advance. Along {-1, 0, 0} the start rests on the cap's plane, 0.1 tol inside
  // its rest gap, and does not sink: free (method paper <sec-touching-rule>). Moving away is free.
  const Vec3 corner_start = {.x = 0.5 + 0.9 * kTolerance, .y = 0, .z = 1 + 0.9 * kTolerance};
  const FirstHit approaching =
      FindHit(inputs, {.start = corner_start, .velocity = {.x = -1, .y = 0, .z = -1}, .duration = 1});
  EXPECT_EQ(approaching.solid_index, 0U);
  EXPECT_NEAR(approaching.elapsed, (0.9 - std::sqrt(0.5)) * kTolerance, kTolerance / 1024);
  EXPECT_EQ(approaching.at_hit.face, SolidFace::kHighRim);
  CheckVec3IsNear(approaching.at_hit.normal_out_of_solid, {.x = std::sqrt(0.5), .y = 0, .z = std::sqrt(0.5)}, 1e-5);
  CheckRimHitDistance(approaching.at_hit.distance);
  CheckFree(FindHit(inputs, {.start = corner_start, .velocity = {.x = -1, .y = 0, .z = 0}, .duration = 1}));
  CheckFree(FindHit(inputs, {.start = corner_start, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 1}));
}

TEST(FindFirstHit, CylinderRimCornerStartAlongTangent_IsFree) {
  // Wall and cap distances both exactly 0.875 tol, so the rim normal has equal x and z parts and the path along
  // {1, 0, -1} has rate exactly 0: it neither approaches the rim nor leaves it, and the true distance, convex along the
  // path, only grows. Free; only an approach (rate < 0) is a hit.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 corner_start = {.x = 0.5 + 0.875 * kTolerance, .y = 0, .z = 1 + 0.875 * kTolerance};
  CheckFree(FindHit(inputs, {.start = corner_start, .velocity = {.x = 1, .y = 0, .z = -1}, .duration = 1}));
}

TEST(FindFirstHit, SpinningCylinderAboutAxis_IsClosedForm) {
  // A spin about the cylinder's own axis does not change its shape (method paper <sec-slide-and-spin>): the wall time
  // of the cylinder at rest, at the start of the step and a quarter of a second into it.
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
    CheckNearRelative(first_hit.elapsed, 1.5 - kTolerance, 1);
    EXPECT_EQ(first_hit.at_hit.face, SolidFace::kCurved);
    CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = -1, .y = 0, .z = 0}, 1e-8);
  }
}

TEST(FindFirstHit, TiltedSpinningCylinder_IsClosedForm) {
  // A spin about a tilted axis leaves a part across the axis of a few rounding units, far below the bound of method
  // paper <sec-slide-and-spin>, so the cylinder is still swept by the closed form: the cap is hit 2 - tol after a start
  // 3 m along the axis.
  const Vec3 axis = ToColumn(kTiltedAxes, 2);
  const std::array solids = {BuildCylinder({}, axis, {.linear_velocity = {}, .angular_velocity = 5 * axis})};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit first_hit = FindHit(inputs, {.start = 3 * axis, .velocity = -1 * axis, .duration = 4});
  EXPECT_EQ(first_hit.solid_index, 0U);
  CheckNearRelative(first_hit.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(first_hit.at_hit.face, SolidFace::kHighZ);
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, axis, 1e-8);
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
  // In a step of 2 s the turn moves the surface by at most (tol / 64) * 2 * 3 = 0.09375 tol <= tol / 8 (method paper
  // <sec-slide-and-spin>): the wall is hit at 2 - tol by the closed form.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 64)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -4, .y = 0, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 2});
  EXPECT_EQ(first_hit.solid_index, 0U);
  CheckNearRelative(first_hit.elapsed, 2 - kTolerance, 1);
}

TEST(FindFirstHit, CylinderTurningAcrossAtBound_IsClosedForm) {
  // In a step of 1 s at tol / 24 rad/s the bound's product (tol / 24) * 1 * 3 rounds to exactly tol / 8: the bound
  // itself still counts as not turning (method paper <sec-slide-and-spin>: it turns only above it).
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 24)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const FirstHit first_hit =
      FindHit(inputs, {.start = {.x = -3, .y = 0, .z = 0}, .velocity = {.x = 1, .y = 0, .z = 0}, .duration = 1});
  EXPECT_EQ(first_hit.solid_index, 0U);
  CheckNearRelative(first_hit.elapsed, 1 - kTolerance, 1);
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
  CheckFree(FindHit(inputs, {.start = {.x = -(2 + kTolerance), .y = 0, .z = 0}, .velocity = kAtRest, .duration = 2}));
}

TEST(FindFirstHit, CylinderTurningAcrossAtBound_RestingParticleIsFree) {
  // (tol / 24) * 1 * 3 rounds to exactly tol / 8, which still counts as not turning: the touching rule frees it.
  const std::array solids = {BuildCylinderTurningAcross(kTolerance / 24)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  CheckFree(FindHit(inputs, {.start = {.x = -(2 + kTolerance), .y = 0, .z = 0}, .velocity = kAtRest, .duration = 1}));
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
  CheckNearRelative(first_hit.elapsed, 0.5 - kTolerance, 1);
}

TEST(FindFirstHit, CylinderPathAlongAxis_HitsEachCap) {
  // On the axis with no motion across it, the path is inside the grown wall all along: it enters between the grown
  // caps' planes, through the high cap from above and the low cap from below.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  const FirstHit from_above =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4});
  EXPECT_EQ(from_above.solid_index, 0U);
  CheckNearRelative(from_above.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(from_above.at_hit.face, SolidFace::kHighZ);
  const FirstHit from_below =
      FindHit(inputs, {.start = {.x = 0, .y = 0, .z = -3}, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 4});
  EXPECT_EQ(from_below.solid_index, 0U);
  CheckNearRelative(from_below.elapsed, 2 - kTolerance, 1);
  EXPECT_EQ(from_below.at_hit.face, SolidFace::kLowZ);
  CheckVec3IsNear(from_below.at_hit.normal_out_of_solid, {.x = 0, .y = 0, .z = -1}, 1e-8);
}

TEST(FindFirstHit, CylinderMovingAwayFromCapOrWall_IsFree) {
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  CheckFree(
      FindHit(inputs, {.start = {.x = 0.1, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = 1}, .duration = 4})
  );
  CheckFree(
      FindHit(inputs, {.start = {.x = 0.1, .y = 0, .z = -3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4})
  );
  CheckFree(
      FindHit(inputs, {.start = {.x = -2, .y = 0, .z = 0}, .velocity = {.x = -1, .y = 0, .z = 0}, .duration = 4})
  );
}

TEST(FindFirstHit, CylinderDownItsGrownWallPastTheRim_IsFree) {
  // Down the axis direction at exactly the grown radius 0.5 + tol, beside the high rim: the start rests on the wall
  // taken as unbounded along the axis, which the whole cylinder lies behind, and does not sink toward it, so the sweep
  // is free (method paper <sec-touching-rule>). The sharp grown edge alone would stop it Sqrt(2) tol from the rim.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 4);
  CheckFree(FindHit(
      inputs,
      {.start = {.x = 0.5 + kTolerance, .y = 0, .z = 3}, .velocity = {.x = 0, .y = 0, .z = -1}, .duration = 4}
  ));
}

TEST(FindFirstHit, CylinderAlongItsGrownCapPlanePastTheRim_IsFree) {
  // Across the axis at exactly the grown half height 1 + tol: the start rests on the plane of the high cap, beside the
  // rim, and the path stays in that plane: free.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 2);
  CheckFree(FindHit(
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
  CheckFree(
      FindHit(inputs, {.start = {.x = rest_x - kTolerance / 8, .y = 0, .z = 3}, .velocity = down, .duration = 4})
  );
  const FirstHit deeper =
      FindHit(inputs, {.start = {.x = rest_x - 3 * kTolerance / 16, .y = 0, .z = 3}, .velocity = down, .duration = 4});
  EXPECT_EQ(deeper.solid_index, 0U);
  EXPECT_EQ(deeper.at_hit.face, SolidFace::kHighRim);
  // The search stops within tol / 1024 of the rest gap, where the gap closes at 0.584 m/s: within tol / 512 s.
  EXPECT_NEAR(deeper.elapsed, 2 - std::sqrt(1 - (13.0 / 16) * (13.0 / 16)) * kTolerance, kTolerance / 512);
  const Vec3 rest_start = {.x = rest_x, .y = 0, .z = 3};
  CheckFree(
      FindHit(inputs, {.start = rest_start, .velocity = {.x = -kTolerance / 512, .y = 0, .z = -1}, .duration = 4})
  );
  const FirstHit sinking =
      FindHit(inputs, {.start = rest_start, .velocity = {.x = -kTolerance / 128, .y = 0, .z = -1}, .duration = 4});
  EXPECT_EQ(sinking.solid_index, 0U);
  EXPECT_EQ(sinking.at_hit.face, SolidFace::kHighRim);
  const Vec3 quick_sink = {.x = -kTolerance / 128, .y = 0, .z = -2};
  CheckFree(FindHit(
      inputs,
      {.start = {.x = rest_x + kTolerance / 128, .y = 0, .z = 3}, .velocity = quick_sink, .duration = 2}
  ));
  CheckFree(
      FindHit(inputs, {.start = {.x = rest_x + kTolerance / 8, .y = 0, .z = 3}, .velocity = quick_sink, .duration = 2})
  );
  // A wall of radius 4 m: the wall's rate is divided by the distance from the axis, so sinking by tol / 128 still
  // rests there; the product alone would read four times that, a sink of tol / 32.
  const std::array wide_solids = {
      ToSolid(Cylinder{.center = {}, .axis = kAxisZ, .radius = 4, .half_height = 1}, kNoMotion, 1),
  };
  const SolidContactInputs wide_inputs = BuildInputs(wide_solids, 4);
  CheckFree(FindHit(
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
  // the distance, so the plain closed form decides (the touching rule of method paper <sec-round-starts> alone) and the
  // grown wall is reached at 1 s, at the height of the cap's plane, still on the wall proper.
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
    CheckFree(first_hit);
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
  CheckNearRelative(through_cap.elapsed, kTolerance, kTolerance);
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
  CheckNearRelative(through_wall.elapsed, kTolerance / 0.7, kTolerance);
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
    CheckFree(FindHit(inputs, path_ending_before_hit));
  }
}

TEST(FindFirstHit, CylinderStartJustInsideRest_PushingInHitsAtZero) {
  // Half a tolerance above the low cap, below it: gap -tol / 2. Sliding along the cap is free; pushing in is a hit.
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 start = {.x = 0.1, .y = 0.1, .z = -(1 + kTolerance / 2)};
  CheckFree(FindHit(inputs, {.start = start, .velocity = {.x = 1, .y = 0, .z = kTolerance / 4}, .duration = 1}));
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
  CheckVec3IsNear(first_hit.at_hit.normal_out_of_solid, {.x = 0, .y = 1, .z = 0}, 0);
  EXPECT_EQ(first_hit.at_hit.distance, -0.5);
}

TEST(FindFirstHitOfSweep, CylinderBesideRimRestingOnWall_FreeAlsoAfterAResponse) {
  // 3 / 16 tol above the cap's plane at the wall's rest gap, beside the rim 1.0174 tol away, falling along the wall:
  // the sweep tries the wall first and rests on it, so it is free, touched or not (method paper <sec-touching-rule>);
  // as built the touched rule read the rim inside its window first and hit it at 0. tol / 32 inside the wall's rest gap
  // and tol / 8 above the cap's plane, the rim inside its own touching window: free too (was a hit at 0).
  const std::array solids = {BuildCylinder({}, kAxisZ, kNoMotion)};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const Vec3 down = {.x = 0, .y = 0, .z = -1};
  const TouchedRule touched = {.is_touched = true, .approach_allowance = 0x1p-46};
  const Sweep above_seam = {
      .start = {.x = 0.5 + kTolerance, .y = 0, .z = 1 + 3.0 / 16 * kTolerance},
      .velocity_for_this_step = down,
      .duration = 1,
  };
  FirstHit first_hit;
  ASSERT_EQ(FindFirstHitOfSweep(inputs, above_seam, touched, &first_hit, nullptr), ContactError::kNone);
  CheckFree(first_hit);
  const Sweep inside_rest_gap = {
      .start = {.x = 0.5 + kTolerance - kTolerance / 32, .y = 0, .z = 1 + kTolerance / 8},
      .velocity_for_this_step = down,
      .duration = 1,
  };
  ASSERT_EQ(FindFirstHitOfSweep(inputs, inside_rest_gap, TouchedRule{}, &first_hit, nullptr), ContactError::kNone);
  CheckFree(first_hit);
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
  CheckFree(FindHit(inputs, path));
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
  // 1e-15 m turns it by about 1e-15 / tol, near 1e-6: the direction itself is that uncertain, not the code. Measured:
  // a worst error of 2.07e-6 over 2,000 random turns of these cases; 1e-5 is about 5 times that.
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
void CheckEachCylinderFaceHit(Vec3 axis) {
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
    CheckNearRelative(first_hit.elapsed, face_case.hit_elapsed, 1);
    EXPECT_EQ(first_hit.at_hit.face, face_case.face);
    CheckVec3IsNear(
        first_hit.at_hit.normal_out_of_solid,
        axes * face_case.own_normal_out_of_solid,
        face_case.normal_tolerance
    );
    EXPECT_NEAR(first_hit.at_hit.distance, face_case.distance_at_hit, face_case.distance_tolerance)
        << static_cast<int>(face_case.face);
  }
}

TEST(FindFirstHit, MovingCylinderEachFace_HitsThatFace) {
  CheckEachCylinderFaceHit(kAxisZ);
}

TEST(FindFirstHit, TiltedMovingCylinderEachFace_HitsThatFace) {
  CheckEachCylinderFaceHit(ToColumn(kTiltedAxes, 2));
}

// --- The rim search ---

// The half sizes of the test cylinder, radius 0.5 m and half height 1 m.
constexpr Vec3 kCylinderHalfSize = {.x = 0.5, .y = 0.5, .z = 1};

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
  // at which it closes, so a path along the rim's normal needs one and a path down the wall two (method paper
  // <sec-round-first-hit>).
  const auto expect_advances = [](RimSearchCase search_case) {
    const RimSearch search = FindRimFromEntry(search_case.own_path, kCylinderHalfSize, search_case.duration);
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
  CheckFree(FindHit(inputs, {.start = start, .velocity = kAtRest, .duration = 0.2}));
}

// Fails the calling test unless a hit of the stepping search sits at the surface of the grown solid: its distance is
// in [tol - tol / 64, (1 + kTurningSolidHitGapShare) tol], the rounding allowance below and the stopping gap above.
void CheckAtGrownSurface(DistanceToSolid at_hit) {
  EXPECT_GE(at_hit.distance, kTolerance - kTolerance / 64);
  EXPECT_LE(at_hit.distance, (1 + kTurningSolidHitGapShare) * kTolerance);
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
  CheckAtGrownSurface(first_hit.at_hit);
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
  CheckAtGrownSurface(first_hit.at_hit);
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
  CheckFree(FindHit(inputs, path));
  const TurningSearch search = FindTurningSolid(inputs, path);
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
  EXPECT_EQ(close_call, ContactCloseCall::kHoverNearTurningSolid);
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
  CheckFree(FindHit(BuildInputs(still, 0.1), {.start = start, .velocity = along_face, .duration = 0.1}));
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
  // with the distance a committed particle sees (method paper <sec-plate-sides>). From inside the thickness at z =
  // -0.005 both forms hit at 0, an uncommitted particle on kLowZ at -0.005, a particle committed to kAlongNormal on
  // kHighZ at -0.015.
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
  // A flat solid is unbounded sideways, so a spin about its normal leaves its shape as it is (method paper
  // <sec-not-turning>): it is not a turning solid, and the closed form gives the bits of the floor without the spin. 1
  // rad/s keeps the turn within kMaxTurnPerStep, so a spin counted as a turn shows here as another hit time, not as a
  // solid CheckSolid fails on.
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
  // rule (method paper <sec-touching-rule>). Counted as turning, it would be hit at 0.
  const std::array solids = {BuildSpinningFloor(5)};
  const Path path = {
      .start = {.x = 0.3, .y = 0, .z = kTolerance},
      .velocity = {.x = 1, .y = 0, .z = 0},
      .duration = 0.1,
  };
  CheckFree(FindHit(BuildInputs(solids, 0.1), path));
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
  CheckFree(FindHit(BuildInputs(solids, 0.1), path));
}

TEST(FindFirstHit, PlateTurningAboutAxisInItsPlane_IsTurning) {
  // A turn about an axis in the plate's plane tilts its faces: a turning solid, which has no touching rule (method
  // paper <sec-turning-solid>), so the same resting particle is hit at 0.
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
  CheckFree(first_hit);
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
      CheckFree(first_hit);
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
  const TurningSearch capped_search = FindTurningSolid(
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
  // second-order model of one side says nothing about the next: with it the search answered free after 2 advances,
  // with the path ending 0.5 m beyond a side. The first-order advance alone finds the side at 0.6693805413275088 s.
  Solid room;
  room.shape = SolidShape::kDomainBox;
  room.solid_id = 1;
  room.half_size = {.x = 1, .y = 1, .z = 1};
  room.closed = {.left = true, .right = true, .front = true, .back = true, .bottom = true, .top = true};
  room.motion.angular_velocity = {.x = 0, .y = 0, .z = std::numbers::pi / 2};
  const std::array solids = {room};
  const TurningSearch search = FindTurningSolid(
      BuildInputs(solids, 1),
      {.start = {.x = 0.5, .y = 0.7, .z = 0}, .velocity = {.x = 1, .y = -1, .z = 0}, .duration = 1}
  );
  EXPECT_EQ(search.error, ContactError::kNone);
  EXPECT_EQ(search.first_hit.solid_index, 0U);
  EXPECT_NEAR(search.first_hit.elapsed, 0.66938054133, kTurningHitTimeTolerance);
  CheckAtGrownSurface(search.first_hit.at_hit);
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
  CheckFree(FindHit(inputs, path));
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

TEST(FindFirstHit, TurningLongBoxOutwardPath_FarEndOfRadiusBoundsAdvance) {
  // From 1 m out at the angle phi0 = 1.6316666666666666, moving outward at 3 m/s: the path ends 4 m from the origin,
  // and the arm closes on it at nearly 1.57 * r(t) m/s. A bound built from the start's distance alone would let an
  // advance carry the particle 6.4 cm into the box; the true bound stops it 1.1 tol from face kHighX at
  // 0.9562428207707441 s, after 32 first-order advances (a mirror scene, measured). The search now takes 6 advances and
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
  CheckAtGrownSurface(first_hit.at_hit);
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
  CheckAtGrownSurface(first_hit.at_hit);
  EXPECT_EQ(FindTurningSolid(inputs, path).advance_count, 3);
}

TEST(FindFirstHit, TurningLongBoxFaceSweepsOntoParticle_FirstOrderAdvanceIsTheFloor) {
  // At rest 0.4 m in front of face kLowX of the quarter-turn arm, 3 m along it: the face comes on nearly head on, so
  // the gap closes at almost the bound while K, built for the whole sweep, is large. Then the first-order advance,
  // g / bound, is the longer one, and the search keeps it: 3 advances. The second-order advance alone takes 4.
  const std::array solids = {BuildQuarterTurnLongBox({})};
  const SolidContactInputs inputs = BuildInputs(solids, 1);
  const TurningSearch search = FindTurningSolid(
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
    const TurningSearch search = FindTurningSolid(pinned_case.inputs, pinned_case.path);
    EXPECT_EQ(search.advance_count, pinned_case.advance_count) << pinned_case.name;
    EXPECT_EQ(search.first_hit.elapsed, first_hit.elapsed) << pinned_case.name;
  }
}

}  // namespace
}  // namespace fluidity::contact
