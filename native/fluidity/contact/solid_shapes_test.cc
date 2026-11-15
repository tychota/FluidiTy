#include "fluidity/contact/solid_shapes.h"

#include <gtest/gtest.h>

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

}  // namespace
}  // namespace fluidity::contact
