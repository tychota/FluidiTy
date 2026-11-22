#include "fluidity/contact/solid_contact.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/left_out_solids_sets.h"

namespace fluidity::contact {
namespace {

// The reads of one solid during one move: how many times it was posed and its distance read.
struct SolidReads {
  long pose_reads = 0;
  long distance_reads = 0;
};

// Runs MoveParticle on one scene and returns the reads of the solid with id `solid_id`.
SolidReads ComputeReadsOfSolid(const MoveParticleCase& test_case, std::uint32_t solid_id, ContactError* error) {
  solid_read_counts = {};
  SolidContactResult result;
  *error = MoveParticle(ToSolidContactInputs(test_case), &result, nullptr);
  return {
      .pose_reads = solid_read_counts.pose_reads[solid_id],
      .distance_reads = solid_read_counts.distance_reads[solid_id],
  };
}

// The scene of BuildLeftOutBoundaryCases named `name`, the `occurrence`-th of that name (0 or 1 for the two sides of a
// boundary).
MoveParticleCase FindBoundaryCase(std::string_view name, int occurrence) {
  int seen = 0;
  for (const MoveParticleCase& test_case : BuildLeftOutBoundaryCases()) {
    if (test_case.name == name && seen++ == occurrence) {
      return test_case;
    }
  }
  ADD_FAILURE() << "no boundary case " << name;
  return {};
}

TEST(LeftOutSolids, FarParticle_ReadsEachOutOfRangeSolidOnce) {
  // The far set's particles above z = 0 and its 32 solids below z = -0.5: a particle at rest at z = 0.5 reads each
  // solid once, in the start's depth check, and the domain box of the crowded set is read more than once.
  const std::vector<Solid> solids = BuildLeftOutSet(LeftOutSet::kFar).front().solids;
  const MoveParticleCase far_particle = {
      .name = "FarParticle",
      .solids = solids,
      .inputs = left_out_sets::BuildSetInputs({.x = 0.3, .y = 0.3, .z = 0.5}, left_out_sets::kAtRest, 0.01, 0.3),
  };
  solid_read_counts = {};
  SolidContactResult result;
  ASSERT_EQ(MoveParticle(ToSolidContactInputs(far_particle), &result, nullptr), ContactError::kNone);
  for (const Solid& solid : solids) {
    EXPECT_EQ(solid_read_counts.pose_reads[solid.solid_id], 1) << "solid " << solid.solid_id;
    EXPECT_EQ(solid_read_counts.distance_reads[solid.solid_id], 1) << "solid " << solid.solid_id;
  }
}

TEST(LeftOutSolids, FarDomainBox_IsAlwaysRead) {
  ContactError error = ContactError::kNone;
  const SolidReads reads = ComputeReadsOfSolid(FindBoundaryCase("FarDomainBox", 0), 1, &error);
  EXPECT_EQ(error, ContactError::kNone);
  EXPECT_GT(reads.distance_reads, 1);
}

// Checks one shape's boundary: the start at the out-of-range distance reads the solid more than once, the start one
// double farther reads it once.
void CheckBoundaryOfShape(std::string_view name) {
  SCOPED_TRACE(std::string(name));
  ContactError inside_error = ContactError::kNone;
  ContactError outside_error = ContactError::kNone;
  const SolidReads inside = ComputeReadsOfSolid(FindBoundaryCase(name, 0), 1, &inside_error);
  const SolidReads outside = ComputeReadsOfSolid(FindBoundaryCase(name, 1), 1, &outside_error);
  EXPECT_EQ(inside_error, ContactError::kNone);
  EXPECT_EQ(outside_error, ContactError::kNone);
  EXPECT_GT(inside.distance_reads, 1);
  EXPECT_EQ(outside.distance_reads, 1);
  EXPECT_EQ(outside.pose_reads, 1);
}

TEST(LeftOutSolids, FlatSolidAtBoundary_ReadOnlyWithin) {
  CheckBoundaryOfShape("Flat");
}

TEST(LeftOutSolids, PlateAtBoundary_ReadOnlyWithin) {
  CheckBoundaryOfShape("Plate");
}

TEST(LeftOutSolids, BoxAtBoundary_ReadOnlyWithin) {
  CheckBoundaryOfShape("Box");
}

TEST(LeftOutSolids, SphereAtBoundary_ReadOnlyWithin) {
  CheckBoundaryOfShape("Sphere");
}

TEST(LeftOutSolids, SpinningCylinderAtBoundary_ReadOnlyWithin) {
  // The turn's term: min(Length(angular_velocity) T, 2) * Length(half_size), 0.4 * 0.15 m here, counts in the bound.
  CheckBoundaryOfShape("SpinningCylinder");
}

TEST(LeftOutSolids, CylinderAtBoundary_ReadOnlyWithin) {
  CheckBoundaryOfShape("Cylinder");
}

TEST(LeftOutSolids, TurningBoxAtBoundary_ReadOnlyWithin) {
  // A box turning at 20 rad/s about z: its distance, the largest face-plane gap, is only bounded through its true
  // distance, so the window and the stepping search's gap count twice (kBoxDistanceShare).
  CheckBoundaryOfShape("TurningBox");
}

TEST(LeftOutSolids, ArrivingSphere_IsReadAndHits) {
  // Only the sphere's own travel, 0.1 m in the step, brings it to the particle at rest: it must be read, and it hits.
  ContactError error = ContactError::kNone;
  const SolidReads reads = ComputeReadsOfSolid(FindBoundaryCase("ArrivingSphere", 0), 1, &error);
  EXPECT_GT(reads.distance_reads, 1);
  SolidContactResult result;
  const MoveParticleCase scene = FindBoundaryCase("ArrivingSphere", 0);
  ASSERT_EQ(MoveParticle(ToSolidContactInputs(scene), &result, nullptr), ContactError::kNone);
  EXPECT_EQ(result.kind, ContactKind::kHit);
}

TEST(LeftOutSolids, TurningPaddle_IsReadAndSweepsIntoParticle) {
  // Only the turn brings the paddle's tip to the particle at rest: it must be read, and it hits the particle.
  ContactError error = ContactError::kNone;
  const SolidReads reads = ComputeReadsOfSolid(FindBoundaryCase("TurningPaddle", 0), 1, &error);
  EXPECT_GT(reads.distance_reads, 1);
  SolidContactResult result;
  ContactFailure failure;
  const MoveParticleCase scene = FindBoundaryCase("TurningPaddle", 0);
  const ContactError move_error = MoveParticle(ToSolidContactInputs(scene), &result, &failure);
  const bool is_hit_or_close_call = move_error == ContactError::kTooCloseToCall
      || (move_error == ContactError::kNone && result.kind == ContactKind::kHit);
  EXPECT_TRUE(is_hit_or_close_call) << static_cast<int>(move_error);
}

TEST(LeftOutSolids, TurningFloorFarFromAxis_RisesIntoParticle) {
  // The floor turns about the y axis; 10 m from it the plane rises 0.1 m in the step under a particle at rest 0.05 m
  // above. The rule counts the particle's arm from the plane's origin, so the floor is read: the particle is hit and
  // carried up with it to about 0.1 m.
  SolidContactResult result;
  const MoveParticleCase scene = FindBoundaryCase("TurningFloorFarFromAxis", 0);
  ASSERT_EQ(MoveParticle(ToSolidContactInputs(scene), &result, nullptr), ContactError::kNone);
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_NEAR(result.position.z, 0.1, 1e-6);
}

TEST(LeftOutSolids, FastSpinningTiltedCylinder_KeepsHoverNearTurningSolidCloseCall) {
  // 3 m away, the turning cylinder's stepping search runs out of advances: the rule's step 4 keeps it read, so the
  // move gives the close call the search gives, naming the cylinder.
  SolidContactResult result;
  ContactFailure failure;
  const MoveParticleCase scene = FindBoundaryCase("FastSpinningTiltedCylinder", 0);
  EXPECT_EQ(MoveParticle(ToSolidContactInputs(scene), &result, &failure), ContactError::kTooCloseToCall);
  EXPECT_EQ(failure.close_call, ContactCloseCall::kHoverNearTurningSolid);
  EXPECT_EQ(failure.solid_id, 1U);
}

TEST(LeftOutSolids, LongStartMove_ReadsEverySolidForTheStartResponse) {
  // A V of two flat solids moves the particle off both faces, 32 tol to its rest point; a third flat solid 1.03125 tol
  // beyond that point was left out at the start, 33 tol away. Read again from the moved start, it is active and
  // approached: the start response hits it once, as with every solid read (EverySet_SameBitsAsReadingEverySolid
  // compares the bits). Left out, it would give no hit and kind kNone.
  const MoveParticleCase scene = FindBoundaryCase("LongStartMoveIntoLeftOutWall", 0);
  const SolidContactInputs inputs = ToSolidContactInputs(scene);
  constexpr std::uint32_t kWallIndex = 2;
  const SolidPose wall_pose = inputs.solids[kWallIndex].pose_at_step_start;
  const DistanceToSolid wall_at_start =
      ComputeDistanceForParticle(inputs, kWallIndex, wall_pose, inputs.start_position, inputs.plate_side);
  EXPECT_TRUE(ComputeIsSolidOutOfRange(inputs, kWallIndex, wall_at_start.distance, ComputeParticleReach(inputs)));
  SolidContactResult result;
  ASSERT_EQ(MoveParticle(inputs, &result, nullptr), ContactError::kNone);
  EXPECT_EQ(result.kind, ContactKind::kHit);
  EXPECT_EQ(result.hit_count, 1);
  ContactError error = ContactError::kNone;
  EXPECT_GT(ComputeReadsOfSolid(scene, 3, &error).distance_reads, 1);
}

TEST(LeftOutSolids, PushedIntoFarSphere_SphereReadAfterTheResponse) {
  // The sphere is out of range for the particle at rest; the wall pushes it at 20 m/s toward the sphere, so after that
  // response the move reads every solid again, meets the sphere, and the particle is squeezed between the two (naming
  // the wall, the solid its stuck velocity approaches fastest). Leaving the sphere out after the response would carry
  // the particle into it.
  ContactError error = ContactError::kNone;
  const SolidReads reads = ComputeReadsOfSolid(FindBoundaryCase("PushedIntoFarSphere", 0), 2, &error);
  EXPECT_GT(reads.distance_reads, 1);
  SolidContactResult result;
  ContactFailure failure;
  const MoveParticleCase scene = FindBoundaryCase("PushedIntoFarSphere", 0);
  EXPECT_EQ(MoveParticle(ToSolidContactInputs(scene), &result, &failure), ContactError::kTooCloseToCall);
  EXPECT_EQ(failure.close_call, ContactCloseCall::kSqueezedBetweenSolids);
  EXPECT_EQ(failure.solid_id, 1U);
}

}  // namespace
}  // namespace fluidity::contact
