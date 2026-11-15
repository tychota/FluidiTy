#include "fluidity/contact/solid_shapes.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>

#include "fluidity/base/mat3.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_test_cases.h"

namespace fluidity::contact {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// A solid that neither moves nor turns.
constexpr SolidMotion kNoMotion = SolidMotion();

Vec3 ToColumnX(Mat3 axes) {
  return {.x = axes.xx, .y = axes.yx, .z = axes.zx};
}

Vec3 ToColumnY(Mat3 axes) {
  return {.x = axes.xy, .y = axes.yy, .z = axes.zy};
}

Vec3 ToColumnZ(Mat3 axes) {
  return {.x = axes.xz, .y = axes.yz, .z = axes.zz};
}

Mat3 BuildAxesFromColumns(Vec3 column_x, Vec3 column_y, Vec3 column_z) {
  return {
      .xx = column_x.x,
      .xy = column_y.x,
      .xz = column_z.x,
      .yx = column_x.y,
      .yy = column_y.y,
      .yz = column_z.y,
      .zx = column_x.z,
      .zy = column_y.z,
      .zz = column_z.z,
  };
}

void ExpectSameVec3(Vec3 actual, Vec3 expected) {
  EXPECT_EQ(actual.x, expected.x) << "axis x";
  EXPECT_EQ(actual.y, expected.y) << "axis y";
  EXPECT_EQ(actual.z, expected.z) << "axis z";
}

void ExpectSameMat3(Mat3 actual, Mat3 expected) {
  ExpectSameVec3(ToColumnX(actual), ToColumnX(expected));
  ExpectSameVec3(ToColumnY(actual), ToColumnY(expected));
  ExpectSameVec3(ToColumnZ(actual), ToColumnZ(expected));
}

// A box at {1, 2, 3} with half sizes 0.5 that does not move: every check passes.
Solid BuildValidBox() {
  const Box box = {
      .center = {.x = 1, .y = 2, .z = 3},
      .axes = kIdentity,
      .half_size = {.x = 0.5, .y = 0.5, .z = 0.5},
  };
  return ToSolid(box, kNoMotion, 7);
}

// The domain box from {0, 0, 0} to {1, 1, 1}, open at the top.
Solid BuildValidDomainBox() {
  const DomainBox domain_box = {
      .low_corner = {.x = 0, .y = 0, .z = 0},
      .high_corner = {.x = 1, .y = 1, .z = 1},
  };
  return ToSolid(domain_box, 1);
}

Solid BuildSphere(double radius, SolidMotion motion) {
  const Sphere sphere = {.center = {.x = 0, .y = 0, .z = 0}, .radius = radius};
  return ToSolid(sphere, motion, 3);
}

// A cylinder standing along world z: radius 0.5, half height 1.
Solid BuildUprightCylinder(SolidMotion motion) {
  const Cylinder cylinder = {
      .center = {.x = 0, .y = 0, .z = 0},
      .axis = {.x = 0, .y = 0, .z = 1},
      .radius = 0.5,
      .half_height = 1,
  };
  return ToSolid(cylinder, motion, 4);
}

// A cylinder lying along world x: radius 0.5, half height 1. Its own z axis is {1, 0, 0}.
Solid BuildCylinderAlongX(SolidMotion motion) {
  const Cylinder cylinder = {
      .center = {.x = 0, .y = 0, .z = 0},
      .axis = {.x = 1, .y = 0, .z = 0},
      .radius = 0.5,
      .half_height = 1,
  };
  return ToSolid(cylinder, motion, 9);
}

SolidMotion BuildTurning(Vec3 angular_velocity) {
  return {.linear_velocity = {.x = 0, .y = 0, .z = 0}, .angular_velocity = angular_velocity};
}

TEST(ToSolid, FlatSolid_ZAxisIsNormalAndAxesAreARotation) {
  const Vec3 normal = math::Normalized({.x = 1, .y = 2, .z = 3});
  const FlatSolid flat_solid = {.point_on_surface = {.x = 4, .y = 5, .z = 6}, .normal_out_of_solid = normal};

  const Solid solid = ToSolid(flat_solid, kNoMotion, 2);

  EXPECT_EQ(solid.shape, SolidShape::kFlatSolid);
  EXPECT_EQ(solid.solid_id, 2U);
  ExpectSameVec3(solid.pose_at_step_start.origin, flat_solid.point_on_surface);
  ExpectSameVec3(ToColumnZ(solid.pose_at_step_start.axes), normal);
  // The normal's smallest part is on x, so x is the world axis crossed with the normal.
  const Vec3 expected_column_x = math::Normalized(math::Cross({.x = 1, .y = 0, .z = 0}, normal));
  ExpectSameVec3(ToColumnX(solid.pose_at_step_start.axes), expected_column_x);
  EXPECT_NEAR(math::Determinant(solid.pose_at_step_start.axes), 1, 1e-15);
}

TEST(ToSolid, Plate_HalfSizeZIsHalfThickness) {
  const Plate plate = {
      .center = {.x = 0, .y = 0, .z = 1},
      .normal = {.x = 0, .y = 0, .z = 1},
      .half_thickness = 0.25,
  };

  const Solid solid = ToSolid(plate, kNoMotion, 5);

  EXPECT_EQ(solid.shape, SolidShape::kPlate);
  ExpectSameVec3(solid.half_size, {.x = 0, .y = 0, .z = 0.25});
  ExpectSameVec3(solid.pose_at_step_start.origin, plate.center);
  ExpectSameVec3(ToColumnZ(solid.pose_at_step_start.axes), plate.normal);
}

TEST(ToSolid, DomainBox_OriginIsCentreAndHalfSizeIsHalfExtent) {
  const DomainBox domain_box = {
      .low_corner = {.x = 0, .y = 0, .z = 0},
      .high_corner = {.x = 2, .y = 4, .z = 6},
  };

  const Solid solid = ToSolid(domain_box, 1);

  EXPECT_EQ(solid.shape, SolidShape::kDomainBox);
  ExpectVec3Near(solid.pose_at_step_start.origin, {.x = 1, .y = 2, .z = 3}, 0);
  ExpectVec3Near(solid.half_size, {.x = 1, .y = 2, .z = 3}, 0);
  ExpectSameMat3(solid.pose_at_step_start.axes, kIdentity);
  EXPECT_FALSE(solid.closed.top);
  EXPECT_TRUE(solid.closed.bottom);
}

TEST(ToSolid, Sphere_HalfSizeIsRadiusAndAxesAreIdentity) {
  const Sphere sphere = {.center = {.x = 1, .y = 2, .z = 3}, .radius = 0.5};

  const Solid solid = ToSolid(sphere, kNoMotion, 6);

  EXPECT_EQ(solid.shape, SolidShape::kSphere);
  ExpectSameVec3(solid.pose_at_step_start.origin, {.x = 1, .y = 2, .z = 3});
  ExpectSameVec3(solid.half_size, {.x = 0.5, .y = 0.5, .z = 0.5});
  ExpectSameMat3(solid.pose_at_step_start.axes, kIdentity);
}

TEST(ToSolid, Cylinder_ZAxisIsAxisAndHalfSizeIsRadiusRadiusHalfHeight) {
  const Cylinder cylinder = {
      .center = {.x = 0, .y = 0, .z = 0},
      .axis = {.x = 1, .y = 0, .z = 0},
      .radius = 0.5,
      .half_height = 1,
  };

  const Solid solid = ToSolid(cylinder, kNoMotion, 8);

  EXPECT_EQ(solid.shape, SolidShape::kCylinder);
  // The axis has parts 0 on y and z; the tie keeps y, and Cross({0, 1, 0}, {1, 0, 0}) = {0, 0, -1}.
  ExpectSameVec3(ToColumnX(solid.pose_at_step_start.axes), {.x = 0, .y = 0, .z = -1});
  ExpectSameVec3(ToColumnY(solid.pose_at_step_start.axes), {.x = 0, .y = 1, .z = 0});
  ExpectSameVec3(ToColumnZ(solid.pose_at_step_start.axes), {.x = 1, .y = 0, .z = 0});
  ExpectSameVec3(solid.half_size, {.x = 0.5, .y = 0.5, .z = 1});
}

TEST(ToSolidFace, EveryDomainSide_MatchesSolidFaceOrder) {
  static_assert(ToSolidFace(DomainSide::kLeft) == SolidFace::kLowX);
  static_assert(ToSolidFace(DomainSide::kRight) == SolidFace::kHighX);
  static_assert(ToSolidFace(DomainSide::kFront) == SolidFace::kLowY);
  static_assert(ToSolidFace(DomainSide::kBack) == SolidFace::kHighY);
  static_assert(ToSolidFace(DomainSide::kBottom) == SolidFace::kLowZ);
  static_assert(ToSolidFace(DomainSide::kTop) == SolidFace::kHighZ);
  EXPECT_EQ(ToSolidFace(DomainSide::kTop), SolidFace::kHighZ);
}

TEST(PlateSide, SamePlateAndSide_AreEqual) {
  static_assert(sizeof(PlateSide) == 8);
  const PlateSide first = {.plate_id = 1, .side = SolidSide::kAlongNormal};
  const PlateSide second = {.plate_id = 1, .side = SolidSide::kAlongNormal};
  const PlateSide other_side = {.plate_id = 1, .side = SolidSide::kAgainstNormal};

  EXPECT_EQ(first, second);
  EXPECT_NE(first, other_side);
}

TEST(Solid, AnyValue_IsTriviallyCopyable) {
  EXPECT_TRUE(std::is_trivially_copyable_v<Solid>);
}

TEST(ComputeShapeTurningAngularVelocity, Cylinder_IsPartAcrossItsAxis) {
  const SolidMotion turning = {
      .linear_velocity = {.x = 0, .y = 0, .z = 0},
      .angular_velocity = {.x = 2, .y = 0, .z = 100},
  };

  ExpectSameVec3(ComputeShapeTurningAngularVelocity(BuildUprightCylinder(turning)), {.x = 2, .y = 0, .z = 0});
}

TEST(ComputeShapeTurningAngularVelocity, Sphere_IsZero) {
  const SolidMotion turning = {
      .linear_velocity = {.x = 0, .y = 0, .z = 0},
      .angular_velocity = {.x = 2, .y = 0, .z = 100},
  };

  ExpectSameVec3(ComputeShapeTurningAngularVelocity(BuildSphere(0.5, turning)), {.x = 0, .y = 0, .z = 0});
}

TEST(ComputeShapeTurningAngularVelocity, Box_IsFullAngularVelocity) {
  Solid solid = BuildValidBox();
  solid.motion.angular_velocity = {.x = 2, .y = 0, .z = 100};

  ExpectSameVec3(ComputeShapeTurningAngularVelocity(solid), {.x = 2, .y = 0, .z = 100});
}

TEST(ComputeShapeTurningAngularVelocity, CylinderAlongXSpinningAboutX_IsZero) {
  const Solid solid = BuildCylinderAlongX(BuildTurning({.x = 100, .y = 0, .z = 0}));

  ExpectSameVec3(ComputeShapeTurningAngularVelocity(solid), {.x = 0, .y = 0, .z = 0});
}

TEST(ComputeShapeTurningAngularVelocity, CylinderAlongXTurningAboutZ_IsFullTurn) {
  // Removing the world z part instead of the part along the cylinder's own axis would give zero here.
  const Solid solid = BuildCylinderAlongX(BuildTurning({.x = 0, .y = 0, .z = 2}));

  ExpectSameVec3(ComputeShapeTurningAngularVelocity(solid), {.x = 0, .y = 0, .z = 2});
}

TEST(CheckSolid, ValidBox_IsNone) {
  EXPECT_EQ(CheckSolid(BuildValidBox(), 0.01), ContactError::kNone);
}

TEST(CheckSolid, EveryShapeBuiltByToSolid_IsNone) {
  const FlatSolid flat_solid = {
      .point_on_surface = {.x = 0, .y = 0, .z = 0},
      .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
  };
  const Plate plate = {.center = {.x = 0, .y = 0, .z = 0}, .normal = {.x = 0, .y = 1, .z = 0}, .half_thickness = 0.1};

  EXPECT_EQ(CheckSolid(ToSolid(flat_solid, kNoMotion, 1), 0.01), ContactError::kNone);
  EXPECT_EQ(CheckSolid(ToSolid(plate, kNoMotion, 2), 0.01), ContactError::kNone);
  EXPECT_EQ(CheckSolid(BuildValidDomainBox(), 0.01), ContactError::kNone);
  EXPECT_EQ(CheckSolid(BuildSphere(0.5, kNoMotion), 0.01), ContactError::kNone);
  EXPECT_EQ(CheckSolid(BuildUprightCylinder(kNoMotion), 0.01), ContactError::kNone);
}

TEST(CheckSolid, NonFiniteOrigin_IsBadInput) {
  Solid solid = BuildValidBox();
  solid.pose_at_step_start.origin.y = kNaN;

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, NonFiniteMotion_IsBadInput) {
  Solid solid = BuildValidBox();
  solid.motion.angular_velocity.x = std::numeric_limits<double>::infinity();

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, AxesNotARotation_IsBadInput) {
  const Vec3 unit_x = {.x = 1, .y = 0, .z = 0};
  const Vec3 unit_y = {.x = 0, .y = 1, .z = 0};
  const Vec3 unit_z = {.x = 0, .y = 0, .z = 1};
  Solid reflected = BuildValidBox();
  reflected.pose_at_step_start.axes = BuildAxesFromColumns(unit_y, unit_x, unit_z);
  Solid scaled = BuildValidBox();
  scaled.pose_at_step_start.axes = BuildAxesFromColumns(unit_x * 2, unit_y, unit_z);

  EXPECT_EQ(CheckSolid(reflected, 0.01), ContactError::kBadInput);
  EXPECT_EQ(CheckSolid(scaled, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, AxesNotOrthogonal_IsBadInput) {
  // Unit columns with a positive determinant (about 0.707), but y and z are 45 degrees apart.
  const double half_root_two = std::numbers::sqrt2 / 2;
  Solid solid = BuildValidBox();
  solid.pose_at_step_start.axes = BuildAxesFromColumns(
      {.x = 1, .y = 0, .z = 0},
      {.x = 0, .y = 1, .z = 0},
      {.x = 0, .y = half_root_two, .z = half_root_two}
  );

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, ColumnLengthJustInsideTolerance_IsNone) {
  // Squared length 1 + 2^-41, half the tolerance of 2^-40.
  const double column_length = math::Sqrt(1 + 0x1p-41);
  Solid solid = BuildValidBox();
  solid.pose_at_step_start.axes =
      BuildAxesFromColumns({.x = column_length, .y = 0, .z = 0}, {.x = 0, .y = 1, .z = 0}, {.x = 0, .y = 0, .z = 1});

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kNone);
}

TEST(CheckSolid, ColumnLengthJustOutsideTolerance_IsBadInput) {
  // Squared length 1 + 2^-39, twice the tolerance of 2^-40.
  const double column_length = math::Sqrt(1 + 0x1p-39);
  Solid solid = BuildValidBox();
  solid.pose_at_step_start.axes =
      BuildAxesFromColumns({.x = column_length, .y = 0, .z = 0}, {.x = 0, .y = 1, .z = 0}, {.x = 0, .y = 0, .z = 1});

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, ZeroHalfSizeBox_IsBadInput) {
  Solid solid = BuildValidBox();
  solid.half_size.y = 0;

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, IdZero_IsBadInput) {
  Solid solid = BuildValidBox();
  solid.solid_id = 0;

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, MovingDomainBox_IsBadInput) {
  Solid sliding = BuildValidDomainBox();
  sliding.motion.linear_velocity.x = 1;
  Solid turning = BuildValidDomainBox();
  turning.motion.angular_velocity.z = 0.1;

  EXPECT_EQ(CheckSolid(sliding, 0.01), ContactError::kBadInput);
  EXPECT_EQ(CheckSolid(turning, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, DomainBoxWithTurnedAxes_IsBadInput) {
  // A true rotation, a quarter turn about z, so only the domain box's alignment rule refuses it.
  Solid solid = BuildValidDomainBox();
  solid.pose_at_step_start.axes =
      BuildAxesFromColumns({.x = 0, .y = 1, .z = 0}, {.x = -1, .y = 0, .z = 0}, {.x = 0, .y = 0, .z = 1});

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, DomainBoxWithNoClosedSide_IsBadInput) {
  Solid solid = BuildValidDomainBox();
  solid.closed = {.left = false, .right = false, .front = false, .back = false, .bottom = false, .top = false};

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, NegativeFrictionScale_IsBadInput) {
  Solid solid = BuildValidBox();
  solid.friction_scale = -0.5;

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, NonPositiveStepDuration_IsBadInput) {
  EXPECT_EQ(CheckSolid(BuildValidBox(), 0), ContactError::kBadInput);
  EXPECT_EQ(CheckSolid(BuildValidBox(), kNaN), ContactError::kBadInput);
}

TEST(CheckSolid, UnknownShape_IsBadInput) {
  Solid solid = BuildValidBox();
  solid.shape = static_cast<SolidShape>(200);

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, TwoRadiansInOneStep_IsSolidTurnsTooFar) {
  Solid solid = BuildValidBox();
  solid.motion.angular_velocity = {.x = 0, .y = 0, .z = 2};
  ContactCloseCall close_call = ContactCloseCall::kNone;

  EXPECT_EQ(CheckSolid(solid, 1, &close_call), ContactError::kTooCloseToCall);
  EXPECT_EQ(close_call, ContactCloseCall::kSolidTurnsTooFar);
  EXPECT_EQ(CheckSolid(solid, 1), ContactError::kTooCloseToCall);
}

TEST(CheckSolid, QuarterTurnExactly_IsNone) {
  Solid solid = BuildValidBox();
  solid.motion.angular_velocity = {.x = 0, .y = 0, .z = kMaxTurnPerStep};
  // Writes kNone over a stale value: the output always says whether the failure was a close call.
  ContactCloseCall close_call = ContactCloseCall::kTooManyHits;

  EXPECT_EQ(CheckSolid(solid, 1, &close_call), ContactError::kNone);
  EXPECT_EQ(close_call, ContactCloseCall::kNone);
}

TEST(CheckSolid, ZeroRadiusSphere_IsBadInput) {
  EXPECT_EQ(CheckSolid(BuildSphere(0, kNoMotion), 0.01), ContactError::kBadInput);
  EXPECT_EQ(CheckSolid(BuildSphere(-1, kNoMotion), 0.01), ContactError::kBadInput);
  EXPECT_EQ(CheckSolid(BuildSphere(kNaN, kNoMotion), 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, SphereFirstHalfSizeDiffers_IsBadInput) {
  Solid solid = BuildSphere(0.5, kNoMotion);
  solid.half_size = {.x = 0.6, .y = 0.5, .z = 0.5};

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, PlateNonPositiveHalfThickness_IsBadInput) {
  const Plate zero_thickness = {.center = {.x = 0, .y = 0, .z = 0}, .normal = {.x = 0, .y = 0, .z = 1}};
  const Plate negative_thickness = {
      .center = {.x = 0, .y = 0, .z = 0},
      .normal = {.x = 0, .y = 0, .z = 1},
      .half_thickness = -0.1,
  };

  EXPECT_EQ(CheckSolid(ToSolid(zero_thickness, kNoMotion, 2), 0.01), ContactError::kBadInput);
  EXPECT_EQ(CheckSolid(ToSolid(negative_thickness, kNoMotion, 2), 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, CylinderNonPositiveRadius_IsBadInput) {
  Solid zero_radius = BuildUprightCylinder(kNoMotion);
  zero_radius.half_size.x = 0;
  zero_radius.half_size.y = 0;
  Solid negative_radius = BuildUprightCylinder(kNoMotion);
  negative_radius.half_size.x = -0.5;
  negative_radius.half_size.y = -0.5;

  EXPECT_EQ(CheckSolid(zero_radius, 0.01), ContactError::kBadInput);
  EXPECT_EQ(CheckSolid(negative_radius, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, SphereUnequalHalfSizes_IsBadInput) {
  Solid solid = BuildSphere(0.5, kNoMotion);
  solid.half_size = {.x = 0.5, .y = 0.5, .z = 0.6};

  EXPECT_EQ(CheckSolid(solid, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, CylinderNegativeHalfHeight_IsBadInput) {
  Solid negative_half_height = BuildUprightCylinder(kNoMotion);
  negative_half_height.half_size.z = -1;
  Solid unequal_radii = BuildUprightCylinder(kNoMotion);
  unequal_radii.half_size.y = 0.6;

  EXPECT_EQ(CheckSolid(negative_half_height, 0.01), ContactError::kBadInput);
  EXPECT_EQ(CheckSolid(unequal_radii, 0.01), ContactError::kBadInput);
}

TEST(CheckSolid, CylinderTumblingTwoRadiansInOneStep_IsSolidTurnsTooFar) {
  const SolidMotion tumbling = {
      .linear_velocity = {.x = 0, .y = 0, .z = 0},
      .angular_velocity = {.x = 2, .y = 0, .z = 0},
  };
  ContactCloseCall close_call = ContactCloseCall::kNone;

  EXPECT_EQ(CheckSolid(BuildUprightCylinder(tumbling), 1, &close_call), ContactError::kTooCloseToCall);
  EXPECT_EQ(close_call, ContactCloseCall::kSolidTurnsTooFar);
}

TEST(CheckSolid, SpinningFastAboutOwnAxis_IsNone) {
  // 100 rad/s for 0.1 s is 10 rad, far above a quarter turn, but a spin does not change the shape.
  const SolidMotion spinning = {
      .linear_velocity = {.x = 0, .y = 0, .z = 0},
      .angular_velocity = {.x = 0, .y = 0, .z = 100},
  };

  EXPECT_EQ(CheckSolid(BuildSphere(0.5, spinning), 0.1), ContactError::kNone);
  EXPECT_EQ(CheckSolid(BuildUprightCylinder(spinning), 0.1), ContactError::kNone);
}

TEST(CheckSolid, SpinningFastAboutTiltedAxis_IsNone) {
  // A cylinder along world x spinning at 100 rad/s about x for 0.1 s: 10 rad of spin, no change of shape.
  EXPECT_EQ(CheckSolid(BuildCylinderAlongX(BuildTurning({.x = 100, .y = 0, .z = 0})), 0.1), ContactError::kNone);
}

// A box at the origin with world axes, half sizes 1 and the given motion.
Solid BuildUnitBox(SolidMotion motion) {
  const Box box = {
      .center = {.x = 0, .y = 0, .z = 0},
      .axes = kIdentity,
      .half_size = {.x = 1, .y = 1, .z = 1},
  };
  return ToSolid(box, motion, 11);
}

TEST(ComputeSolidPose, QuarterTurnAboutZ_SendsSolidXToWorldY) {
  const Solid solid = BuildUnitBox(BuildTurning({.x = 0, .y = 0, .z = std::numbers::pi / 2}));

  const SolidPose pose = ComputeSolidPose(solid, 1);

  ExpectVec3Near(ToColumnX(pose.axes), {.x = 0, .y = 1, .z = 0}, 1e-15);
  ExpectVec3Near(ToColumnY(pose.axes), {.x = -1, .y = 0, .z = 0}, 1e-15);
  ExpectSameVec3(pose.origin, {.x = 0, .y = 0, .z = 0});
}

TEST(ComputeSolidPose, TiltedSolidTurning_TurnsAboutWorldAxisNotOwnAxis) {
  // Start axes: a quarter turn about world x, so the own y axis is world z and the own z axis is world -y. A
  // quarter turn about world z then sends the own x axis to world y. Turning about the own z axis instead (axes0
  // times the turn) would send it to world z.
  const Box box = {
      .center = {.x = 0, .y = 0, .z = 0},
      .axes = BuildAxesFromColumns({.x = 1, .y = 0, .z = 0}, {.x = 0, .y = 0, .z = 1}, {.x = 0, .y = -1, .z = 0}),
      .half_size = {.x = 1, .y = 1, .z = 1},
  };
  const Solid solid = ToSolid(box, BuildTurning({.x = 0, .y = 0, .z = std::numbers::pi}), 12);

  const SolidPose pose = ComputeSolidPose(solid, 0.5);

  ExpectVec3Near(ToColumnX(pose.axes), {.x = 0, .y = 1, .z = 0}, 1e-15);
  ExpectVec3Near(ToColumnZ(pose.axes), {.x = 1, .y = 0, .z = 0}, 1e-15);
}

TEST(ComputeSolidPose, ZeroElapsed_IsStartPose) {
  const Box box = {
      .center = {.x = 1, .y = 2, .z = 3},
      .axes = BuildAxesFromColumns({.x = 1, .y = 0, .z = 0}, {.x = 0, .y = 0, .z = 1}, {.x = 0, .y = -1, .z = 0}),
      .half_size = {.x = 1, .y = 1, .z = 1},
  };
  const SolidMotion motion = {
      .linear_velocity = {.x = 0.5, .y = -1, .z = 2},
      .angular_velocity = {.x = 1, .y = 2, .z = 3},
  };
  const Solid solid = ToSolid(box, motion, 12);

  const SolidPose pose = ComputeSolidPose(solid, 0);

  ExpectSameVec3(pose.origin, solid.pose_at_step_start.origin);
  ExpectSameMat3(pose.axes, solid.pose_at_step_start.axes);
}

TEST(ComputeSolidPose, Translating_MovesOriginOnly) {
  const Plate plate = {
      .center = {.x = 1, .y = 2, .z = 3},
      .normal = math::Normalized({.x = 1, .y = 2, .z = 3}),
      .half_thickness = 0.25,
  };
  const SolidMotion translating = {
      .linear_velocity = {.x = 0.5, .y = 0, .z = 0},
      .angular_velocity = {.x = 0, .y = 0, .z = 0},
  };
  const Solid solid = ToSolid(plate, translating, 5);

  const SolidPose pose = ComputeSolidPose(solid, 2);

  ExpectSameVec3(pose.origin, {.x = 2, .y = 2, .z = 3});
  ExpectSameMat3(pose.axes, solid.pose_at_step_start.axes);
}

TEST(ComputeSolidVelocity, Turning_IsLinearPlusAngularCrossArm) {
  const SolidMotion motion = {
      .linear_velocity = {.x = 1, .y = 0, .z = 0},
      .angular_velocity = {.x = 0, .y = 0, .z = 2},
  };
  const Solid solid = BuildUnitBox(motion);

  // {1, 0, 0} + Cross({0, 0, 2}, {0, 3, 0}) = {1, 0, 0} + {-6, 0, 0}.
  const Vec3 velocity = ComputeSolidVelocity(solid, solid.pose_at_step_start, {.x = 0, .y = 3, .z = 0});

  ExpectSameVec3(velocity, {.x = -5, .y = 0, .z = 0});
}

TEST(ComputeSolidVelocity, MovedPose_ArmIsFromPoseOrigin) {
  const SolidMotion motion = {
      .linear_velocity = {.x = 1, .y = 0, .z = 0},
      .angular_velocity = {.x = 0, .y = 0, .z = 2},
  };
  const Solid solid = BuildUnitBox(motion);
  const SolidPose moved_pose = {.origin = {.x = 0, .y = 1, .z = 0}, .axes = kIdentity};

  // The arm is {0, 3, 0} - {0, 1, 0} = {0, 2, 0}: {1, 0, 0} + {-4, 0, 0}.
  const Vec3 velocity = ComputeSolidVelocity(solid, moved_pose, {.x = 0, .y = 3, .z = 0});

  ExpectSameVec3(velocity, {.x = -3, .y = 0, .z = 0});
}

TEST(ComputeDistanceToSolid, FlatSolidAboveAndBelow_IsSignedHeight) {
  const Vec3 normal = math::Normalized({.x = 0, .y = 3, .z = 4});
  const Vec3 point_on_surface = {.x = 1, .y = 2, .z = 3};
  const FlatSolid flat_solid = {.point_on_surface = point_on_surface, .normal_out_of_solid = normal};
  const Solid solid = ToSolid(flat_solid, kNoMotion, 2);

  const DistanceToSolid above =
      ComputeDistanceToSolid(solid, solid.pose_at_step_start, point_on_surface + 0.5 * normal);
  const DistanceToSolid below =
      ComputeDistanceToSolid(solid, solid.pose_at_step_start, point_on_surface - 0.25 * normal);

  // A tilted normal rounds the point and its turn into the own axes: a few rounding units of 3.
  EXPECT_NEAR(above.distance, 0.5, 1e-15);
  EXPECT_EQ(above.face, SolidFace::kHighZ);
  ExpectSameVec3(above.normal_out_of_solid, normal);
  EXPECT_NEAR(below.distance, -0.25, 1e-15);
  EXPECT_EQ(below.face, SolidFace::kHighZ);
  ExpectSameVec3(below.normal_out_of_solid, normal);
}

TEST(ComputeDistanceToSolid, MovedPose_UsesGivenPose) {
  const FlatSolid flat_solid = {.point_on_surface = {.x = 0, .y = 0, .z = 0}};
  const Solid solid = ToSolid(flat_solid, kNoMotion, 2);
  const SolidPose raised_pose = {.origin = {.x = 0, .y = 0, .z = 1}, .axes = solid.pose_at_step_start.axes};

  const DistanceToSolid distance = ComputeDistanceToSolid(solid, raised_pose, {.x = 0, .y = 0, .z = 1.5});

  EXPECT_EQ(distance.distance, 0.5);
}

// A plate in the plane z = 0 with normal {0, 0, 1} and half thickness 0.1.
Solid BuildFlatLyingPlate() {
  const Plate plate = {
      .center = {.x = 0, .y = 0, .z = 0},
      .normal = {.x = 0, .y = 0, .z = 1},
      .half_thickness = 0.1,
  };
  return ToSolid(plate, kNoMotion, 5);
}

TEST(ComputeDistanceToSolid, PlateEitherSide_IsGapToNearerFace) {
  const Solid solid = BuildFlatLyingPlate();
  const SolidPose pose = solid.pose_at_step_start;

  const DistanceToSolid above = ComputeDistanceToSolid(solid, pose, {.x = 0, .y = 0, .z = 0.3});
  const DistanceToSolid below = ComputeDistanceToSolid(solid, pose, {.x = 0, .y = 0, .z = -0.3});
  const DistanceToSolid middle = ComputeDistanceToSolid(solid, pose, {.x = 0, .y = 0, .z = 0});

  // 0.3 - 0.1 in double is one rounding unit below 0.2.
  EXPECT_NEAR(above.distance, 0.2, 1e-16);
  EXPECT_EQ(above.face, SolidFace::kHighZ);
  ExpectSameVec3(above.normal_out_of_solid, {.x = 0, .y = 0, .z = 1});
  EXPECT_NEAR(below.distance, 0.2, 1e-16);
  EXPECT_EQ(below.face, SolidFace::kLowZ);
  ExpectSameVec3(below.normal_out_of_solid, {.x = 0, .y = 0, .z = -1});
  // On the middle plane p.z = 0 counts as the kHighZ side.
  EXPECT_EQ(middle.distance, -0.1);
  EXPECT_EQ(middle.face, SolidFace::kHighZ);
  ExpectSameVec3(middle.normal_out_of_solid, {.x = 0, .y = 0, .z = 1});
}

TEST(ComputeDistanceToSolid, BoxOutsideOneFace_IsGapToThatFace) {
  const Solid solid = BuildUnitBox(kNoMotion);

  const DistanceToSolid distance = ComputeDistanceToSolid(solid, solid.pose_at_step_start, {.x = 3, .y = 0, .z = 0});

  EXPECT_EQ(distance.distance, 2);
  EXPECT_EQ(distance.face, SolidFace::kHighX);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = 1, .y = 0, .z = 0});
}

TEST(ComputeDistanceToSolid, BoxInside_IsMinusDepthOfNearestFace) {
  const Solid solid = BuildUnitBox(kNoMotion);

  const DistanceToSolid distance =
      ComputeDistanceToSolid(solid, solid.pose_at_step_start, {.x = 0.9, .y = 0, .z = 0.5});

  // 0.9 - 1 in double is a rounding unit of 0.9 away from -0.1.
  EXPECT_NEAR(distance.distance, -0.1, 1e-16);
  EXPECT_EQ(distance.face, SolidFace::kHighX);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = 1, .y = 0, .z = 0});
}

TEST(ComputeDistanceToSolid, BoxCornerTie_TakesFirstFaceInOrder) {
  const Solid solid = BuildUnitBox(kNoMotion);

  // The kLowX and kLowY gaps are both 1.
  const DistanceToSolid distance = ComputeDistanceToSolid(solid, solid.pose_at_step_start, {.x = -2, .y = -2, .z = 0});

  EXPECT_EQ(distance.distance, 1);
  EXPECT_EQ(distance.face, SolidFace::kLowX);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = -1, .y = 0, .z = 0});
}

TEST(ComputeDistanceToSolid, BoxOnMiddlePlaneOfNearestAxis_TakesLowFace) {
  // Half size 0.5 on x and 1 elsewhere: at the centre the x faces are nearest, both 0.5 away. The tie takes the
  // first face in face order, kLowX, whose normal is the own -x axis.
  const Box box = {
      .center = {.x = 0, .y = 0, .z = 0},
      .axes = kIdentity,
      .half_size = {.x = 0.5, .y = 1, .z = 1},
  };
  const Solid solid = ToSolid(box, kNoMotion, 11);

  const DistanceToSolid distance = ComputeDistanceToSolid(solid, solid.pose_at_step_start, {.x = 0, .y = 0, .z = 0});

  EXPECT_EQ(distance.distance, -0.5);
  EXPECT_EQ(distance.face, SolidFace::kLowX);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = -1, .y = 0, .z = 0});
}

TEST(ComputeDistanceToSolid, TurnedBox_UsesSolidAxes) {
  // A quarter turn about z: the own x axis is world y and the own y axis is world -x. The world point {0, 3, 0}
  // is the own point {3, 0, 0}.
  const Box box = {
      .center = {.x = 0, .y = 0, .z = 0},
      .axes = BuildAxesFromColumns({.x = 0, .y = 1, .z = 0}, {.x = -1, .y = 0, .z = 0}, {.x = 0, .y = 0, .z = 1}),
      .half_size = {.x = 1, .y = 1, .z = 1},
  };
  const Solid solid = ToSolid(box, kNoMotion, 11);

  const DistanceToSolid distance = ComputeDistanceToSolid(solid, solid.pose_at_step_start, {.x = 0, .y = 3, .z = 0});

  EXPECT_EQ(distance.distance, 2);
  EXPECT_EQ(distance.face, SolidFace::kHighX);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = 0, .y = 1, .z = 0});
}

TEST(ComputeDistanceToSolid, DomainBoxInside_IsGapToNearestClosedSide) {
  const Solid solid = BuildValidDomainBox();

  const DistanceToSolid distance =
      ComputeDistanceToSolid(solid, solid.pose_at_step_start, {.x = 0.5, .y = 0.5, .z = 0.1});

  // 0.5 + (0.1 - 0.5) rounds by a unit of 0.4.
  EXPECT_NEAR(distance.distance, 0.1, 1e-16);
  EXPECT_EQ(distance.face, SolidFace::kLowZ);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = 0, .y = 0, .z = 1});
}

TEST(ComputeDistanceToSolid, DomainBoxOpenTop_IgnoresTop) {
  const Solid solid = BuildValidDomainBox();

  // Left, right, front and back tie at 0.5; the open top, 0.05 away, does not count.
  const DistanceToSolid distance =
      ComputeDistanceToSolid(solid, solid.pose_at_step_start, {.x = 0.5, .y = 0.5, .z = 0.95});

  EXPECT_EQ(distance.distance, 0.5);
  EXPECT_EQ(distance.face, SolidFace::kLowX);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = 1, .y = 0, .z = 0});
}

TEST(ComputeDistanceToSolid, DomainBoxClosedTop_IsGapToTop) {
  Solid solid = BuildValidDomainBox();
  solid.closed.top = true;

  const DistanceToSolid distance =
      ComputeDistanceToSolid(solid, solid.pose_at_step_start, {.x = 0.5, .y = 0.5, .z = 0.95});

  // 0.5 - (0.95 - 0.5): 0.95 itself is rounded, a unit of 0.95 away.
  EXPECT_NEAR(distance.distance, 0.05, 1e-16);
  EXPECT_EQ(distance.face, SolidFace::kHighZ);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = 0, .y = 0, .z = -1});
}

TEST(ComputeDistanceToSolid, DomainBoxTie_TakesLeftBeforeFront) {
  const Solid solid = BuildValidDomainBox();

  // Left and front are both 0.2 away; bottom is 0.5 away.
  const DistanceToSolid distance =
      ComputeDistanceToSolid(solid, solid.pose_at_step_start, {.x = 0.2, .y = 0.2, .z = 0.5});

  EXPECT_NEAR(distance.distance, 0.2, 1e-16);
  EXPECT_EQ(distance.face, SolidFace::kLowX);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = 1, .y = 0, .z = 0});
}

// One point and the face, distance and normal ComputeDistanceToSolid should give for it.
struct FaceCase {
  Vec3 point{};  // m, world.
  double distance = 0;  // m.
  SolidFace face = SolidFace::kHighZ;
  Vec3 normal_out_of_solid{};  // No unit, world.
};

void ExpectFaceCase(const Solid& solid, const FaceCase& face_case) {
  const DistanceToSolid distance = ComputeDistanceToSolid(solid, solid.pose_at_step_start, face_case.point);
  EXPECT_EQ(distance.distance, face_case.distance);
  EXPECT_EQ(distance.face, face_case.face);
  ExpectSameVec3(distance.normal_out_of_solid, face_case.normal_out_of_solid);
}

TEST(ComputeDistanceToSolid, BoxEachFace_ReportsItsFaceAndNormal) {
  // Half sizes {1, 2, 3}: each point is 3 beyond the middle of one face, so every gap is exactly 3 and a mixed-up
  // axis, half size, sign or normal changes the answer.
  const Box box = {
      .center = {.x = 0, .y = 0, .z = 0},
      .axes = kIdentity,
      .half_size = {.x = 1, .y = 2, .z = 3},
  };
  const Solid solid = ToSolid(box, kNoMotion, 11);
  const std::array<FaceCase, 6> face_cases = {
      {
          {
              .point = {.x = -4, .y = 0, .z = 0},
              .distance = 3,
              .face = SolidFace::kLowX,
              .normal_out_of_solid = {.x = -1, .y = 0, .z = 0},
          },
          {
              .point = {.x = 4, .y = 0, .z = 0},
              .distance = 3,
              .face = SolidFace::kHighX,
              .normal_out_of_solid = {.x = 1, .y = 0, .z = 0},
          },
          {
              .point = {.x = 0, .y = -5, .z = 0},
              .distance = 3,
              .face = SolidFace::kLowY,
              .normal_out_of_solid = {.x = 0, .y = -1, .z = 0},
          },
          {
              .point = {.x = 0, .y = 5, .z = 0},
              .distance = 3,
              .face = SolidFace::kHighY,
              .normal_out_of_solid = {.x = 0, .y = 1, .z = 0},
          },
          {
              .point = {.x = 0, .y = 0, .z = -6},
              .distance = 3,
              .face = SolidFace::kLowZ,
              .normal_out_of_solid = {.x = 0, .y = 0, .z = -1},
          },
          {
              .point = {.x = 0, .y = 0, .z = 6},
              .distance = 3,
              .face = SolidFace::kHighZ,
              .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
          },
      },
  };

  for (const FaceCase& face_case : face_cases) {
    SCOPED_TRACE(static_cast<int>(face_case.face));
    ExpectFaceCase(solid, face_case);
  }
}

TEST(ComputeDistanceToSolid, DomainBoxEachClosedSide_ReportsItsSide) {
  // From {0, 0, 0} to {2, 4, 6}, every side closed: centre {1, 2, 3}, half sizes {1, 2, 3}. Each point is 0.25
  // from one side and at least 1.75 from the others; 0.25 and the coordinates are exact in binary.
  const DomainBox domain_box = {
      .low_corner = {.x = 0, .y = 0, .z = 0},
      .high_corner = {.x = 2, .y = 4, .z = 6},
      .closed = {.left = true, .right = true, .front = true, .back = true, .bottom = true, .top = true},
  };
  const Solid solid = ToSolid(domain_box, 1);
  const std::array<FaceCase, 6> face_cases = {
      {
          {
              .point = {.x = 0.25, .y = 2, .z = 3},
              .distance = 0.25,
              .face = SolidFace::kLowX,
              .normal_out_of_solid = {.x = 1, .y = 0, .z = 0},
          },
          {
              .point = {.x = 1.75, .y = 2, .z = 3},
              .distance = 0.25,
              .face = SolidFace::kHighX,
              .normal_out_of_solid = {.x = -1, .y = 0, .z = 0},
          },
          {
              .point = {.x = 1, .y = 0.25, .z = 3},
              .distance = 0.25,
              .face = SolidFace::kLowY,
              .normal_out_of_solid = {.x = 0, .y = 1, .z = 0},
          },
          {
              .point = {.x = 1, .y = 3.75, .z = 3},
              .distance = 0.25,
              .face = SolidFace::kHighY,
              .normal_out_of_solid = {.x = 0, .y = -1, .z = 0},
          },
          {
              .point = {.x = 1, .y = 2, .z = 0.25},
              .distance = 0.25,
              .face = SolidFace::kLowZ,
              .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
          },
          {
              .point = {.x = 1, .y = 2, .z = 5.75},
              .distance = 0.25,
              .face = SolidFace::kHighZ,
              .normal_out_of_solid = {.x = 0, .y = 0, .z = -1},
          },
      },
  };

  for (const FaceCase& face_case : face_cases) {
    SCOPED_TRACE(static_cast<int>(face_case.face));
    ExpectFaceCase(solid, face_case);
  }
}

// m for distances, no unit for normals. The brief's 1e-15: the round solids' worst case here is one rounding unit
// of 2 (4.4e-16), from the square root, a division and the turn through the axes.
constexpr double kRoundSolidTolerance = 1e-15;

void ExpectFaceCaseNear(const Solid& solid, const FaceCase& face_case) {
  const DistanceToSolid distance = ComputeDistanceToSolid(solid, solid.pose_at_step_start, face_case.point);
  EXPECT_NEAR(distance.distance, face_case.distance, kRoundSolidTolerance);
  EXPECT_EQ(distance.face, face_case.face);
  ExpectVec3Near(distance.normal_out_of_solid, face_case.normal_out_of_solid, kRoundSolidTolerance);
}

// A sphere at {1, 2, 3} with radius 0.5 that does not move.
Solid BuildOffsetSphere() {
  const Sphere sphere = {.center = {.x = 1, .y = 2, .z = 3}, .radius = 0.5};
  return ToSolid(sphere, kNoMotion, 6);
}

// A cylinder that does not move, centred at the origin, with the given axis, radius and half height.
Solid BuildStillCylinder(Vec3 axis, double radius, double half_height) {
  const Cylinder cylinder = {
      .center = {.x = 0, .y = 0, .z = 0},
      .axis = axis,
      .radius = radius,
      .half_height = half_height,
  };
  return ToSolid(cylinder, kNoMotion, 8);
}

// The cylinder of most round-solid tests: along world z, radius 1, half height 0.5. ToSolid gives it the axes
// x {0, -1, 0}, y {1, 0, 0}, z {0, 0, 1}, so its own x axis is not world x.
Solid BuildWideCylinder() {
  return BuildStillCylinder({.x = 0, .y = 0, .z = 1}, 1, 0.5);
}

TEST(ComputeDistanceToSolid, SphereOutside_IsDistanceMinusRadius) {
  const Solid solid = BuildOffsetSphere();

  // {0.375, 0, 0.5} from the centre is 0.625 long, a 3-4-5 triangle scaled by 0.125.
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 1, .y = 2, .z = 4},
          .distance = 0.5,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
      }
  );
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 1.375, .y = 2, .z = 3.5},
          .distance = 0.125,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 0.6, .y = 0, .z = 0.8},
      }
  );
}

TEST(ComputeDistanceToSolid, SphereInside_IsNegative) {
  ExpectFaceCaseNear(
      BuildOffsetSphere(),
      {
          .point = {.x = 1, .y = 2, .z = 3.25},
          .distance = -0.25,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
      }
  );
}

TEST(ComputeDistanceToSolid, SphereCentre_NormalIsMinusSolidXAxis) {
  // Every direction is nearest at the centre. A sphere's axes are kIdentity, so minus its x axis is {-1, 0, 0}.
  ExpectFaceCaseNear(
      BuildOffsetSphere(),
      {
          .point = {.x = 1, .y = 2, .z = 3},
          .distance = -0.5,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = -1, .y = 0, .z = 0},
      }
  );
}

TEST(ComputeDistanceToSolid, TurnedSphereCentre_NormalIsMinusPoseXAxis) {
  // A spinning sphere's pose turns. After a quarter turn about z its x axis is world y, so the degenerate normal
  // is {0, -1, 0}: the rule follows the pose, not the world.
  const Solid solid = BuildOffsetSphere();
  const SolidPose turned_pose = {
      .origin = solid.pose_at_step_start.origin,
      .axes = BuildAxesFromColumns({.x = 0, .y = 1, .z = 0}, {.x = -1, .y = 0, .z = 0}, {.x = 0, .y = 0, .z = 1}),
  };

  const DistanceToSolid distance = ComputeDistanceToSolid(solid, turned_pose, {.x = 1, .y = 2, .z = 3});

  EXPECT_EQ(distance.distance, -0.5);
  EXPECT_EQ(distance.face, SolidFace::kCurved);
  ExpectSameVec3(distance.normal_out_of_solid, {.x = 0, .y = -1, .z = 0});
}

TEST(ComputeDistanceToSolid, TurnedSphere_NormalIsInWorldAxes) {
  // The same quarter turn: the world direction {0, 0.6, 0.8} from the centre is the own direction {0.6, 0, 0.8}.
  // Leaving the normal in own axes would give {0.6, 0, 0.8}.
  const Solid solid = BuildOffsetSphere();
  const SolidPose turned_pose = {
      .origin = solid.pose_at_step_start.origin,
      .axes = BuildAxesFromColumns({.x = 0, .y = 1, .z = 0}, {.x = -1, .y = 0, .z = 0}, {.x = 0, .y = 0, .z = 1}),
  };

  const DistanceToSolid distance = ComputeDistanceToSolid(solid, turned_pose, {.x = 1, .y = 2.375, .z = 3.5});

  EXPECT_NEAR(distance.distance, 0.125, kRoundSolidTolerance);
  EXPECT_EQ(distance.face, SolidFace::kCurved);
  ExpectVec3Near(distance.normal_out_of_solid, {.x = 0, .y = 0.6, .z = 0.8}, kRoundSolidTolerance);
}

TEST(ComputeDistanceToSolid, CylinderWall_IsRadialGap) {
  // wall_distance 2, cap_distance -0.25.
  ExpectFaceCaseNear(
      BuildWideCylinder(),
      {
          .point = {.x = 3, .y = 0, .z = 0.25},
          .distance = 2,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 1, .y = 0, .z = 0},
      }
  );
}

TEST(ComputeDistanceToSolid, CylinderCaps_AreHeightGaps) {
  const Solid solid = BuildWideCylinder();

  // wall_distance -0.5 for both; cap_distance 1.5 above, 1 below.
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0.5, .y = 0, .z = 2},
          .distance = 1.5,
          .face = SolidFace::kHighZ,
          .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
      }
  );
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0, .y = 0.5, .z = -1.5},
          .distance = 1,
          .face = SolidFace::kLowZ,
          .normal_out_of_solid = {.x = 0, .y = 0, .z = -1},
      }
  );
}

TEST(ComputeDistanceToSolid, CylinderRims_AreDistanceToTheCircle) {
  const Solid solid = BuildWideCylinder();

  // wall_distance 0.375 and cap_distance 0.5 for both: the distance to the rim circle is 0.625.
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 1.375, .y = 0, .z = 1},
          .distance = 0.625,
          .face = SolidFace::kHighRim,
          .normal_out_of_solid = {.x = 0.6, .y = 0, .z = 0.8},
      }
  );
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0, .y = -1.375, .z = -1},
          .distance = 0.625,
          .face = SolidFace::kLowRim,
          .normal_out_of_solid = {.x = 0, .y = -0.6, .z = -0.8},
      }
  );
  // wall_distance 0.5 is the larger gap here, so trying the wall before the rim would wrongly take the wall.
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 1.5, .y = 0, .z = 0.875},
          .distance = 0.625,
          .face = SolidFace::kHighRim,
          .normal_out_of_solid = {.x = 0.8, .y = 0, .z = 0.6},
      }
  );
}

TEST(ComputeDistanceToSolid, CylinderInside_NearerOfWallAndCap) {
  const Solid solid = BuildWideCylinder();

  // {0.9, 0, 0}: wall_distance -0.1, cap_distance -0.5. {0, 0, 0.375}: wall_distance -1, cap_distance -0.125.
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0.9, .y = 0, .z = 0},
          .distance = -0.1,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 1, .y = 0, .z = 0},
      }
  );
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0, .y = 0, .z = 0.375},
          .distance = -0.125,
          .face = SolidFace::kHighZ,
          .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
      }
  );
}

TEST(ComputeDistanceToSolid, CylinderInsideTie_TakesWall) {
  // wall_distance and cap_distance are both -0.25.
  ExpectFaceCaseNear(
      BuildWideCylinder(),
      {
          .point = {.x = 0.75, .y = 0, .z = 0.25},
          .distance = -0.25,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 1, .y = 0, .z = 0},
      }
  );
}

TEST(ComputeDistanceToSolid, CylinderMiddlePlaneInCapRegion_TakesHighCap) {
  // At the centre of the wide cylinder, wall_distance -1 and cap_distance -0.5: the caps are nearer and tie, and
  // p.z = 0 takes the high cap.
  ExpectFaceCaseNear(
      BuildWideCylinder(),
      {
          .point = {.x = 0, .y = 0, .z = 0},
          .distance = -0.5,
          .face = SolidFace::kHighZ,
          .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
      }
  );
}

TEST(ComputeDistanceToSolid, CylinderOnAxis_NormalIsMinusSolidXAxis) {
  // Radius 0.25, half height 2: at the centre, wall_distance -0.25 beats cap_distance -2, and the axis has no
  // radial direction. ToSolid gives an upright cylinder the x axis {0, -1, 0}, so the normal is {0, 1, 0}.
  const Solid solid = BuildStillCylinder({.x = 0, .y = 0, .z = 1}, 0.25, 2);
  ASSERT_EQ(ToColumnX(solid.pose_at_step_start.axes).y, -1);

  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0, .y = 0, .z = 0},
          .distance = -0.25,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 0, .y = 1, .z = 0},
      }
  );
}

TEST(ComputeDistanceToSolid, CylinderRegionBoundaries_AreContinuous) {
  const Solid solid = BuildWideCylinder();

  // {1, 0, 0.75}: wall_distance 0, cap_distance 0.25, on the edge of the rim region: the cap.
  // {1.25, 0, 0.5}: wall_distance 0.25, cap_distance 0, on the other edge: the wall. Both are 0.25 away.
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 1, .y = 0, .z = 0.75},
          .distance = 0.25,
          .face = SolidFace::kHighZ,
          .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
      }
  );
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 1.25, .y = 0, .z = 0.5},
          .distance = 0.25,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 1, .y = 0, .z = 0},
      }
  );
  // One rounding unit to each side of each border: the region changes, the distance does not. A region whose
  // distance were off by any amount above the tolerance would show here.
  const auto expect_distance_and_face = [&solid](Vec3 point, SolidFace face) {
    const DistanceToSolid distance = ComputeDistanceToSolid(solid, solid.pose_at_step_start, point);
    EXPECT_NEAR(distance.distance, 0.25, kRoundSolidTolerance);
    EXPECT_EQ(distance.face, face);
  };
  expect_distance_and_face({.x = std::nextafter(1.0, 2.0), .y = 0, .z = 0.75}, SolidFace::kHighRim);
  expect_distance_and_face({.x = std::nextafter(1.0, 0.0), .y = 0, .z = 0.75}, SolidFace::kHighZ);
  expect_distance_and_face({.x = 1.25, .y = 0, .z = std::nextafter(0.5, 1.0)}, SolidFace::kHighRim);
  expect_distance_and_face({.x = 1.25, .y = 0, .z = std::nextafter(0.5, 0.0)}, SolidFace::kCurved);
}

TEST(ComputeDistanceToSolid, CylinderExactlyOnRim_TakesWall) {
  // wall_distance and cap_distance are both 0. The rim needs both gaps positive, and the wall wins the tie with
  // the cap. A rim test written with >= would divide 0 by 0 here and give a NaN normal.
  ExpectFaceCaseNear(
      BuildWideCylinder(),
      {
          .point = {.x = 1, .y = 0, .z = 0.5},
          .distance = 0,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 1, .y = 0, .z = 0},
      }
  );
}

TEST(ComputeDistanceToSolid, TurnedCylinder_UsesSolidAxes) {
  // Axis {1, 0, 0}, with ToSolid's axes x {0, 0, -1}, y {0, 1, 0}, z {1, 0, 0}. Both points are 3 from the axis and
  // 0.25 along it. The second one is the own point {3, 0, 0.25}: its world normal {0, 0, -1} differs from the own
  // normal {1, 0, 0}, so a normal left in the solid's own axes fails. The first one cannot tell, because the own
  // y axis is world y.
  const Solid solid = BuildStillCylinder({.x = 1, .y = 0, .z = 0}, 1, 0.5);

  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0.25, .y = 3, .z = 0},
          .distance = 2,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 0, .y = 1, .z = 0},
      }
  );
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0.25, .y = 0, .z = -3},
          .distance = 2,
          .face = SolidFace::kCurved,
          .normal_out_of_solid = {.x = 0, .y = 0, .z = -1},
      }
  );
}

TEST(ComputeDistanceToSolid, TiltedCylinderEachFace_ReportsItsFaceAndNormal) {
  // Axis {0, 0.6, 0.8}, centre {0.5, -0.25, 1}, radius 1, half height 0.5. ToSolid gives the own axes
  // x {0, -0.8, 0.6}, y {1, 0, 0}, z {0, 0.6, 0.8}. Each point is the centre plus own coordinates along those
  // axes, so the hand numbers of the upright cases hold, and every normal has parts on two or three world axes.
  const Cylinder cylinder = {
      .center = {.x = 0.5, .y = -0.25, .z = 1},
      .axis = {.x = 0, .y = 0.6, .z = 0.8},
      .radius = 1,
      .half_height = 0.5,
  };
  const Solid solid = ToSolid(cylinder, kNoMotion, 8);
  const Vec3 own_x_axis = {.x = 0, .y = -0.8, .z = 0.6};
  const Vec3 own_y_axis = {.x = 1, .y = 0, .z = 0};
  const Vec3 own_z_axis = {.x = 0, .y = 0.6, .z = 0.8};
  ExpectVec3Near(ToColumnX(solid.pose_at_step_start.axes), own_x_axis, kRoundSolidTolerance);
  ExpectVec3Near(ToColumnY(solid.pose_at_step_start.axes), own_y_axis, kRoundSolidTolerance);
  const auto to_world = [&cylinder, own_x_axis, own_y_axis, own_z_axis](Vec3 own_point) {
    return cylinder.center + own_point.x * own_x_axis + own_point.y * own_y_axis + own_point.z * own_z_axis;
  };
  const std::array<FaceCase, 7> face_cases = {
      {
          {
              .point = to_world({.x = 3, .y = 0, .z = 0.25}),
              .distance = 2,
              .face = SolidFace::kCurved,
              .normal_out_of_solid = own_x_axis,
          },
          {
              .point = to_world({.x = 0, .y = -0.9, .z = 0}),
              .distance = -0.1,
              .face = SolidFace::kCurved,
              .normal_out_of_solid = -own_y_axis,
          },
          {
              .point = to_world({.x = 0.5, .y = 0, .z = 2}),
              .distance = 1.5,
              .face = SolidFace::kHighZ,
              .normal_out_of_solid = own_z_axis,
          },
          {
              .point = to_world({.x = 0, .y = 0.5, .z = -1.5}),
              .distance = 1,
              .face = SolidFace::kLowZ,
              .normal_out_of_solid = -own_z_axis,
          },
          {
              .point = to_world({.x = 1.375, .y = 0, .z = 1}),
              .distance = 0.625,
              .face = SolidFace::kHighRim,
              .normal_out_of_solid = 0.6 * own_x_axis + 0.8 * own_z_axis,
          },
          {
              .point = to_world({.x = 0, .y = -1.375, .z = -1}),
              .distance = 0.625,
              .face = SolidFace::kLowRim,
              .normal_out_of_solid = -0.6 * own_y_axis - 0.8 * own_z_axis,
          },
          {
              .point = to_world({.x = -1.5, .y = 0, .z = -0.875}),
              .distance = 0.625,
              .face = SolidFace::kLowRim,
              .normal_out_of_solid = -0.8 * own_x_axis - 0.6 * own_z_axis,
          },
      },
  };

  for (const FaceCase& face_case : face_cases) {
    SCOPED_TRACE(static_cast<int>(face_case.face));
    ExpectFaceCaseNear(solid, face_case);
  }
}

TEST(ComputeDistanceToSolid, PaperCylinderExamples_MatchThePaper) {
  // The method paper's numbers: radius 0.5, half height 0.25. (0.8, 0, 0.65): wall_distance 0.3, cap_distance
  // 0.4, rim at 0.5. (0.1, 0, 0.2): wall_distance -0.4, cap_distance -0.05, under the top cap.
  const Solid solid = BuildStillCylinder({.x = 0, .y = 0, .z = 1}, 0.5, 0.25);

  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0.8, .y = 0, .z = 0.65},
          .distance = 0.5,
          .face = SolidFace::kHighRim,
          .normal_out_of_solid = {.x = 0.6, .y = 0, .z = 0.8},
      }
  );
  ExpectFaceCaseNear(
      solid,
      {
          .point = {.x = 0.1, .y = 0, .z = 0.2},
          .distance = -0.05,
          .face = SolidFace::kHighZ,
          .normal_out_of_solid = {.x = 0, .y = 0, .z = 1},
      }
  );
}

TEST(ComputeDistanceToSolid, SphereNaNPoint_IsNaNDistance) {
  const Vec3 nan_point = {.x = kNaN, .y = 0, .z = 0};

  EXPECT_TRUE(std::isnan(ComputeDistanceToSolid(BuildOffsetSphere(), {}, nan_point).distance));
}

TEST(ComputeDistanceToSolid, CylinderNaNPoint_IsNaNDistance) {
  // A NaN must not be lost in the cylinder's max and min of the two gaps.
  const Vec3 nan_point = {.x = kNaN, .y = 0, .z = 0};

  EXPECT_TRUE(std::isnan(ComputeDistanceToSolid(BuildWideCylinder(), {}, nan_point).distance));
}

TEST(ComputeDistanceToCylinder, NaNHeight_IsNaNDistance) {
  // Only cap_distance is NaN; wall_distance is 1. A max that drops a NaN would give the distance 1.
  const Vec3 half_size = {.x = 1, .y = 1, .z = 0.5};

  EXPECT_TRUE(std::isnan(ComputeDistanceToCylinder({.x = 2, .y = 0, .z = kNaN}, half_size, kIdentity).distance));
}

TEST(ComputeDistanceToCylinder, NaNRadialPart_IsNaNDistance) {
  // Only wall_distance is NaN; cap_distance is 1.5.
  const Vec3 half_size = {.x = 1, .y = 1, .z = 0.5};

  EXPECT_TRUE(std::isnan(ComputeDistanceToCylinder({.x = kNaN, .y = 0, .z = 2}, half_size, kIdentity).distance));
}

TEST(ComputeSolidVelocity, TurningSphere_IsAngularCrossArm) {
  // Cross({0, 0, 2}, {0.5, 0, 0}) = {0, 1, 0}: the full angular velocity counts, though a sphere's shape never turns.
  const Solid solid = BuildSphere(0.5, BuildTurning({.x = 0, .y = 0, .z = 2}));

  const Vec3 velocity = ComputeSolidVelocity(solid, solid.pose_at_step_start, {.x = 0.5, .y = 0, .z = 0});

  ExpectSameVec3(velocity, {.x = 0, .y = 1, .z = 0});
}

}  // namespace
}  // namespace fluidity::contact
