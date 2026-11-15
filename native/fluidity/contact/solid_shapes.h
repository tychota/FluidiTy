// The shapes of solids, their pose and motion during one step, the check that a solid is usable, and the distance
// from a point to a solid.
//
// Purpose: describe every solid that particles cannot enter in one form, Solid, that the contact code reads,
//   check it once per step before any particle is moved, and give its pose, its surface velocity and the distance
//   to it at any time within the step.
//
// Idea: each shape is described in its own axes, so one formula per shape serves any position and turn. A solid's
//   pose places those axes in the world: an origin and a rotation whose columns are the solid's own x, y and z axes.
//   Its motion is a linear and an angular velocity, constant during the step. The user describes a shape in the
//   words of that shape (a flat solid by a point and a normal, a box by its centre and half sizes), and ToSolid turns
//   it into a Solid. The shapes, with p a point in the solid's own axes:
//     flat solid  fills p.z <= 0. Its only face is kHighZ.
//     plate       fills |p.z| <= half_thickness, unbounded sideways. kHighZ is the face on its kAlongNormal side.
//     box         fills |p.x| <= half_size.x, |p.y| <= half_size.y and |p.z| <= half_size.z.
//     domain box  the box the particles live in. Particles are inside; each closed side is an unbounded flat face
//                 whose normal points into the box. Fixed and aligned with the world axes.
//     sphere      fills |p| <= radius around its origin. Its axes are kIdentity: they matter only through its turn,
//                 which moves its surface but not its shape.
//     cylinder    fills Sqrt(p.x^2 + p.y^2) <= radius and |p.z| <= half_height. Its own z axis is its axis.
//   A flat solid, a plate and a cylinder build their axes from one direction d (the normal, or the axis): the z
//   column is d, the x column is Cross(e, d) normalised, with e the world axis on which d has its smallest part
//   (the first in x, y, z order on ties), and the y column is Cross(z column, x column).
//   Example: a cylinder along {1, 0, 0} gets x column {0, 0, -1}, y column {0, 1, 0} and z column {1, 0, 0}.
//   The distance from a point to a solid is positive outside and negative inside. It comes with the normal and the
//   name of the face that gives it.
//
// Words:
//   pose        where a solid is: the origin of its own axes and the rotation that gives those axes in the world
//   motion      how a solid moves during one step: a linear velocity of its origin and an angular velocity about it
//   solid face  one part of a solid's surface, named in the solid's own axes (SolidFace)
//   plate side  one of the two sides of one plate, as a stable identity (PlateSide); particles on different sides
//               of a plate never exchange values
//   wall        the round surface of a cylinder, around its axis (face kCurved)
//   cap         one of the two flat ends of a cylinder (faces kLowZ and kHighZ)
//   rim         the circle where a cylinder's wall meets a cap (faces kLowRim and kHighRim)
//   spin        a turn about a sphere's centre or a cylinder's axis: it moves the surface but not the shape
//   degenerate normal  the normal chosen where several directions are equally near: at a sphere's centre, and
//                      on a cylinder's axis where the wall is nearer than the caps. Minus the solid's own x axis
// Owns:    SolidShape, SolidFace, SolidSide, PlateSide, FlatSolid, Plate, Box, Sphere, Cylinder, ClosedSides,
//          DomainBox, SolidMotion, SolidPose, Solid, kNoSolid, kMaxTurnPerStep, ToSolid,
//          ComputeShapeTurningAngularVelocity, CheckSolid, DistanceToSolid, ComputeSolidPose, ComputeSolidVelocity,
//          ComputeDistanceToSphere, ComputeDistanceToCylinder, ComputeDistanceToSolid, DomainSide, ToSolidFace
// Uses:    Vec3, math::Cross, math::Dot, math::Length, math::Normalized, math::IsFinite (base/vec3.h),
//          Mat3, kIdentity, math::Determinant, math::Transpose, math::Rotation (base/mat3.h), Axis,
//          math::UnitVector (base/axis.h),
//          math::Abs, math::Max, math::Sqrt (base/scalar_math.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h),
//          ContactError, ContactCloseCall (contact/contact_error.h)
// Used by: every contact function, the solids module, which builds the solids of a scene, and the step
// Read:    docs/method-paper/fluidity-method.typ <sec-solid-shapes>, <eq:solid-distance>, <sec-round-solids>
//          Hu 2018, doi:10.1145/3197517.3201293

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"

namespace fluidity::contact {

// The kind of shape a Solid describes. Values keep their numbers: a new shape is added last.
enum class SolidShape : std::uint8_t {
  kFlatSolid,  // Fills one side of a plane.
  kPlate,  // A thin solid with two sides, unbounded sideways.
  kBox,  // Six flat faces.
  kDomainBox,  // The box the particles live in; the particles are inside.
  kSphere,  // Round, around its origin.
  kCylinder,  // Finite and solid, with a wall and two flat caps.
};
static_assert(static_cast<std::uint8_t>(SolidShape::kDomainBox) == 3, "a new shape is added after kDomainBox");

// One part of a solid's surface, in the solid's own axes. When two faces give the same distance, the first in this
// order is reported. Values keep their numbers: a new face is added last.
enum class SolidFace : std::uint8_t {
  kLowX,  // The face at the low end of the own x axis.
  kHighX,  // The face at the high end of the own x axis.
  kLowY,  // The face at the low end of the own y axis.
  kHighY,  // The face at the high end of the own y axis.
  kLowZ,  // The face at the low end of the own z axis; a cylinder's low cap.
  kHighZ,  // The face at the high end of the own z axis; a cylinder's high cap.
  kCurved,  // The round surface of a sphere, or the wall of a cylinder.
  kLowRim,  // The round edge where a cylinder's wall meets its low cap.
  kHighRim,  // The round edge where a cylinder's wall meets its high cap.
};
static_assert(static_cast<std::uint8_t>(SolidFace::kHighZ) == 5, "a new face is added after kHighZ");
static_assert(static_cast<std::uint8_t>(SolidFace::kCurved) == 6, "the round faces keep their numbers");
static_assert(static_cast<std::uint8_t>(SolidFace::kLowRim) == 7, "the round faces keep their numbers");
static_assert(static_cast<std::uint8_t>(SolidFace::kHighRim) == 8, "the round faces keep their numbers");

// Which side of a plate a point or a particle is on. The value is the sign of the own z coordinate on that side.
enum class SolidSide : std::int8_t {
  kNoPlate = 0,  // Not on a side of a plate: the solid is not a plate, or the point is inside the plate.
  kAgainstNormal = -1,  // On the side the plate's normal points away from.
  kAlongNormal = +1,  // On the side the plate's normal points to.
};

// The stable identity of one side of one plate. It is compared, never computed with: particles may exchange values
// through the grid only when their plate sides are equal (same_side).
struct PlateSide {
  std::uint32_t plate_id = 0;  // solid_id of the plate. 0 means no plate.
  SolidSide side = SolidSide::kNoPlate;
  friend constexpr bool operator==(PlateSide, PlateSide) = default;  // same_side is this equality.
};
static_assert(sizeof(PlateSide) == 8, "a plate side is 8 bytes in every particle array");

// A solid that fills one side of a plane.
struct FlatSolid {
  Vec3 point_on_surface{};  // m, world, at the start of the step.
  Vec3 normal_out_of_solid{.x = 0, .y = 0, .z = 1};  // No unit, unit vector, world.
};

// A thin solid with two sides, unbounded sideways. A thin solid of finite size is a Box.
struct Plate {
  Vec3 center{};  // m, world. Any point of the middle plane.
  Vec3 normal{.x = 0, .y = 0, .z = 1};  // No unit, unit vector, world. Points to the kAlongNormal side.
  double half_thickness = 0;  // m, positive.
};

// A solid with six flat faces.
struct Box {
  Vec3 center{};  // m, world.
  Mat3 axes = kIdentity;  // No unit, a rotation. Columns are the box's x, y, z axes in world.
  Vec3 half_size{};  // m, positive, along the box's axes.
};

// A round solid.
struct Sphere {
  Vec3 center{};  // m, world.
  double radius = 0;  // m, positive.
};

// A finite solid cylinder with two flat caps.
struct Cylinder {
  Vec3 center{};  // m, world. Midpoint of the axis.
  // No unit, unit vector, world. The solid's own z axis; the caps are at own z = -half_height and +half_height.
  Vec3 axis{.x = 0, .y = 0, .z = 1};
  double radius = 0;  // m, positive.
  double half_height = 0;  // m, positive, along the axis.
};

// Which sides of the domain box particles cannot cross. true means closed. By default only the top is open.
struct ClosedSides {
  bool left = true;  // side_left, -X.
  bool right = true;  // side_right, +X.
  bool front = true;  // side_front, -Y.
  bool back = true;  // side_back, +Y.
  bool bottom = true;  // side_bottom, -Z.
  bool top = false;  // side_top, +Z.
};

// The box the particles live in. It is fixed and aligned with the world axes.
struct DomainBox {
  Vec3 low_corner{};  // m, world. Meets side_left (-X), side_front (-Y) and side_bottom (-Z).
  Vec3 high_corner{};  // m, world. Meets side_right (+X), side_back (+Y) and side_top (+Z).
  ClosedSides closed{};
};

// How a solid moves during one step. Constant during the step.
struct SolidMotion {
  Vec3 linear_velocity{};  // m/s, world, of the solid's origin.
  Vec3 angular_velocity{};  // rad/s, world, about the solid's origin. Its length is the rate of the turn.
};

// Where a solid is: the origin and the axes of its own frame.
struct SolidPose {
  Vec3 origin{};  // m, world.
  Mat3 axes = kIdentity;  // No unit, a rotation. Columns are the solid's x, y, z axes in world.
};

// What the contact code reads: the shape, its pose at the start of the step and its motion during the step.
struct Solid {
  SolidShape shape = SolidShape::kFlatSolid;
  std::uint32_t solid_id = 0;  // Stable, nonzero, unique in the scene.
  SolidPose pose_at_step_start{};
  SolidMotion motion{};
  // m. Box and domain box: along the own axes. Plate: z is half_thickness, x and y are 0. Sphere: {radius, radius,
  // radius}. Cylinder: {radius, radius, half_height}. Flat solid: unused, {0, 0, 0}.
  Vec3 half_size{};
  ClosedSides closed{};  // Domain box only.
  double friction_scale = 1;  // No unit, zero or more. Multiplies the particle's friction at this solid.
  bool reports_reaction = false;  // true for driven and free solids: their hits fill the solid's reaction.
};

// "No solid", as an index into the solids array.
inline constexpr std::uint32_t kNoSolid = 0xFFFFFFFFU;

// rad. The largest turn of a solid's shape during one step: a quarter turn, pi / 2. A solid that turns further
// asks for a shorter step.
inline constexpr double kMaxTurnPerStep = 1.5707963267948966;

// Builds the Solid of a flat solid. Its origin is the point on the surface and its own z axis is the normal.
//
// Method: the x axis is Cross(e, normal) normalised, e the world axis on which the normal has its smallest part
//   (the first in x, y, z order on ties); the y axis is Cross(normal, x axis). The normal is not normalised here: a
//   normal that is not a unit vector gives axes that CheckSolid refuses.
// Example: normal {0, 0, 1}. Its parts on x and y tie at 0, so e is x, and Cross({1, 0, 0}, {0, 0, 1}) gives the
//   x axis {0, -1, 0}. The axes are x {0, -1, 0}, y {1, 0, 0}, z {0, 0, 1}.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU inline Solid ToSolid(FlatSolid shape, SolidMotion motion, std::uint32_t solid_id) {
  const Vec3 normal = shape.normal_out_of_solid;
  // Crossing with the world axis farthest from the normal keeps the cross product far from zero.
  Axis smallest_part_axis = Axis::kX;
  double smallest_part = math::Abs(normal.x);
  if (math::Abs(normal.y) < smallest_part) {
    smallest_part_axis = Axis::kY;
    smallest_part = math::Abs(normal.y);
  }
  if (math::Abs(normal.z) < smallest_part) {
    smallest_part_axis = Axis::kZ;
  }
  const Vec3 column_x = math::Normalized(math::Cross(math::UnitVector(smallest_part_axis), normal));
  const Vec3 column_y = math::Cross(normal, column_x);

  Solid solid;
  solid.shape = SolidShape::kFlatSolid;
  solid.solid_id = solid_id;
  solid.pose_at_step_start.origin = shape.point_on_surface;
  solid.pose_at_step_start.axes = {
      .xx = column_x.x,
      .xy = column_y.x,
      .xz = normal.x,
      .yx = column_x.y,
      .yy = column_y.y,
      .yz = normal.y,
      .zx = column_x.z,
      .zy = column_y.z,
      .zz = normal.z,
  };
  solid.motion = motion;
  return solid;
}

// Builds the Solid of a plate. Its origin is the center, its own z axis is the normal and half_size is
// {0, 0, half_thickness}. The axes follow the rule of the flat solid.
//
// Example: center {0, 0, 1}, normal {0, 0, 1}, half_thickness 0.25: the plate fills 0.75 <= z <= 1.25 and its
//   kHighZ face, at z = 1.25, looks to the kAlongNormal side.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU inline Solid ToSolid(Plate shape, SolidMotion motion, std::uint32_t solid_id) {
  const FlatSolid middle_plane = {.point_on_surface = shape.center, .normal_out_of_solid = shape.normal};
  Solid solid = ToSolid(middle_plane, motion, solid_id);
  solid.shape = SolidShape::kPlate;
  solid.half_size = {.x = 0, .y = 0, .z = shape.half_thickness};
  return solid;
}

// Builds the Solid of a box. Its origin is the center; its axes and half sizes are the box's.
//
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU constexpr Solid ToSolid(Box shape, SolidMotion motion, std::uint32_t solid_id) {
  Solid solid;
  solid.shape = SolidShape::kBox;
  solid.solid_id = solid_id;
  solid.pose_at_step_start.origin = shape.center;
  solid.pose_at_step_start.axes = shape.axes;
  solid.motion = motion;
  solid.half_size = shape.half_size;
  return solid;
}

// Builds the Solid of the domain box: origin at its centre, axes kIdentity, half size half its extent, no motion.
//
// Example: low corner {0, 0, 0}, high corner {2, 4, 6}: origin {1, 2, 3}, half_size {1, 2, 3}.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU constexpr Solid ToSolid(DomainBox shape, std::uint32_t solid_id) {
  Solid solid;
  solid.shape = SolidShape::kDomainBox;
  solid.solid_id = solid_id;
  solid.pose_at_step_start.origin = (shape.low_corner + shape.high_corner) * 0.5;
  solid.half_size = (shape.high_corner - shape.low_corner) * 0.5;
  solid.closed = shape.closed;
  return solid;
}

// Builds the Solid of a sphere: origin at its center, axes kIdentity, half_size {radius, radius, radius}.
//
// Read: docs/method-paper/fluidity-method.typ <sec-round-solids>
FLUIDITY_CPU_AND_GPU constexpr Solid ToSolid(Sphere shape, SolidMotion motion, std::uint32_t solid_id) {
  Solid solid;
  solid.shape = SolidShape::kSphere;
  solid.solid_id = solid_id;
  solid.pose_at_step_start.origin = shape.center;
  solid.motion = motion;
  solid.half_size = {.x = shape.radius, .y = shape.radius, .z = shape.radius};
  return solid;
}

// Builds the Solid of a cylinder: origin at its center, own z axis along its axis (the other two by the rule of the
// flat solid), half_size {radius, radius, half_height}.
//
// Example: axis {1, 0, 0}, radius 0.5, half_height 1: x column {0, 0, -1}, y column {0, 1, 0}, z column {1, 0, 0},
//   half_size {0.5, 0.5, 1}.
// Read: docs/method-paper/fluidity-method.typ <sec-round-solids>
FLUIDITY_CPU_AND_GPU inline Solid ToSolid(Cylinder shape, SolidMotion motion, std::uint32_t solid_id) {
  const FlatSolid axis_frame = {.point_on_surface = shape.center, .normal_out_of_solid = shape.axis};
  Solid solid = ToSolid(axis_frame, motion, solid_id);
  solid.shape = SolidShape::kCylinder;
  solid.half_size = {.x = shape.radius, .y = shape.radius, .z = shape.half_height};
  return solid;
}

// Computes the part of a solid's angular velocity that changes its shape, rad/s, world.
//
// Method: a spin moves the surface of a sphere or a cylinder but not its shape. A sphere gives {0, 0, 0}. A
//   cylinder gives the part across its own z axis e_z, the z column of its axes at the start of the step. Every
//   other shape gives its full angular velocity.
// Units: rad/s, world.
// Example: a cylinder along z turning at {2, 0, 100} rad/s gives {2, 0, 0}; a sphere turning at {2, 0, 100} rad/s
//   gives {0, 0, 0}; a box gives {2, 0, 100}.
// Read: docs/method-paper/fluidity-method.typ <sec-round-solids>
FLUIDITY_CPU_AND_GPU inline Vec3 ComputeShapeTurningAngularVelocity(const Solid& solid) {
  const Vec3 angular_velocity = solid.motion.angular_velocity;
  if (solid.shape == SolidShape::kSphere) {
    return Vec3{};
  }
  if (solid.shape == SolidShape::kCylinder) {
    // Method paper <sec-round-solids>:  omega_across = omega - Dot(omega, e_z) e_z
    const Mat3 axes = solid.pose_at_step_start.axes;
    const Vec3 own_z_axis = {.x = axes.xz, .y = axes.yz, .z = axes.zz};
    const double along_axis_rate = math::Dot(angular_velocity, own_z_axis);
    return angular_velocity - along_axis_rate * own_z_axis;
  }
  return angular_velocity;
}

// Checks that one solid can be used for a step of step_duration seconds, and says why not.
//
// Method: checks, in this order: every value is finite and step_duration is positive; the axes are a rotation
//   (each column's squared length within 2^-40 of 1, each pair's |Dot| at most 2^-40, determinant positive); the
//   sizes fit the shape; the id is not 0 and friction_scale is zero or more; a domain box does not move, has
//   exactly the world axes (kIdentity, as ToSolid writes them) and has a closed side; the turn of the shape during the
//   step, |ComputeShapeTurningAngularVelocity| * step_duration, is at most kMaxTurnPerStep. So a sphere never turns too
//   far, and a cylinder may spin about its own axis at any rate.
// Units: seconds for step_duration.
// Example: a box turning at {0, 0, 2} rad/s with step_duration 1 s turns 2 rad: kTooCloseToCall, and *close_call
//   is kSolidTurnsTooFar. With step_duration 0.5 s it turns 1 rad: kNone. A cylinder along z spinning at
//   {0, 0, 100} rad/s with step_duration 0.1 s: kNone.
// Fails: kBadInput        a value is not finite, step_duration is not positive, the axes are not a rotation, a size
//                         is not positive, a sphere's three half sizes differ, a cylinder's half_size.x and
//                         half_size.y differ, the id is 0, friction_scale is negative, the shape is unknown, or a
//                         domain box moves, has axes other than kIdentity or has no closed side
//        kTooCloseToCall  the shape turns more than kMaxTurnPerStep during the step; *close_call is
//                         kSolidTurnsTooFar. A shorter step fixes it.
//        When close_call is not null, *close_call is always written: kNone unless the error is kTooCloseToCall.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>, <sec-round-solids>
FLUIDITY_CPU_AND_GPU inline ContactError
CheckSolid(const Solid& solid, double step_duration, ContactCloseCall* close_call) {
  // Steps: 1 finite values  2 axes form a rotation  3 sizes fit the shape  4 id, friction and domain box
  //   5 turn of the shape during the step
  if (close_call != nullptr) {
    *close_call = ContactCloseCall::kNone;
  }
  const SolidPose& pose = solid.pose_at_step_start;
  const SolidMotion& motion = solid.motion;

  // Step 1: a value that is not finite usually comes from a scene file or a driven path that went wrong.
  const bool pose_is_finite = math::IsFinite(pose.origin) && math::IsFinite(pose.axes);
  const bool motion_is_finite = math::IsFinite(motion.linear_velocity) && math::IsFinite(motion.angular_velocity);
  const bool sizes_are_finite = math::IsFinite(solid.half_size) && math::IsFinite(solid.friction_scale);
  const bool step_duration_is_positive = math::IsFinite(step_duration) && step_duration > 0;
  if (!pose_is_finite || !motion_is_finite || !sizes_are_finite || !step_duration_is_positive) {
    return ContactError::kBadInput;
  }

  // Step 2: every formula of the contact code turns a point into the solid's own axes with the transpose, which
  // undoes the turn only for a rotation. A mirror image would also turn every normal inside out.
  // A rotation built in double precision is off by a few rounding units, far below this tolerance.
  constexpr double kRotationTolerance = 0x1p-40;
  const Vec3 column_x = {.x = pose.axes.xx, .y = pose.axes.yx, .z = pose.axes.zx};
  const Vec3 column_y = {.x = pose.axes.xy, .y = pose.axes.yy, .z = pose.axes.zy};
  const Vec3 column_z = {.x = pose.axes.xz, .y = pose.axes.yz, .z = pose.axes.zz};
  const double column_x_length_error = math::Abs(math::Dot(column_x, column_x) - 1);
  const double column_y_length_error = math::Abs(math::Dot(column_y, column_y) - 1);
  const double column_z_length_error = math::Abs(math::Dot(column_z, column_z) - 1);
  const double largest_length_error =
      math::Max(math::Max(column_x_length_error, column_y_length_error), column_z_length_error);
  const double largest_column_dot = math::Max(
      math::Max(math::Abs(math::Dot(column_x, column_y)), math::Abs(math::Dot(column_x, column_z))),
      math::Abs(math::Dot(column_y, column_z))
  );
  const bool is_rotation = largest_length_error <= kRotationTolerance && largest_column_dot <= kRotationTolerance
      && math::Determinant(pose.axes) > 0;
  if (!is_rotation) {
    return ContactError::kBadInput;
  }

  // Step 3: ToSolid writes a sphere's and a cylinder's equal half sizes from one radius, so they compare exactly.
  // An unknown shape keeps sizes_fit_shape false: the value was not written by ToSolid.
  const Vec3 half_size = solid.half_size;
  bool sizes_fit_shape = false;
  switch (solid.shape) {
    case SolidShape::kFlatSolid:
      sizes_fit_shape = true;
      break;
    case SolidShape::kPlate:
      sizes_fit_shape = half_size.z > 0;
      break;
    case SolidShape::kBox:
    case SolidShape::kDomainBox:
      sizes_fit_shape = half_size.x > 0 && half_size.y > 0 && half_size.z > 0;
      break;
    case SolidShape::kSphere:
      sizes_fit_shape = half_size.x > 0 && half_size.x == half_size.y && half_size.y == half_size.z;
      break;
    case SolidShape::kCylinder:
      sizes_fit_shape = half_size.x > 0 && half_size.x == half_size.y && half_size.z > 0;
      break;
  }
  if (!sizes_fit_shape) {
    return ContactError::kBadInput;
  }

  // Step 4: id 0 means "no solid" in failures and plate sides, so a solid cannot carry it. The contact code treats
  // the domain box as fixed and aligned with the world axes, and a domain box with every side open would stop
  // nothing.
  const bool friction_scale_is_valid = solid.friction_scale >= 0;
  if (solid.solid_id == 0 || !friction_scale_is_valid) {
    return ContactError::kBadInput;
  }
  const Vec3 linear = motion.linear_velocity;
  const Vec3 angular = motion.angular_velocity;
  const bool is_still =
      linear.x == 0 && linear.y == 0 && linear.z == 0 && angular.x == 0 && angular.y == 0 && angular.z == 0;
  const ClosedSides closed = solid.closed;
  const bool has_closed_side =
      closed.left || closed.right || closed.front || closed.back || closed.bottom || closed.top;
  const Mat3 axes = pose.axes;
  const bool has_world_axes = axes.xx == 1 && axes.xy == 0 && axes.xz == 0 && axes.yx == 0 && axes.yy == 1
      && axes.yz == 0 && axes.zx == 0 && axes.zy == 0 && axes.zz == 1;
  const bool is_usable_domain_box = is_still && has_world_axes && has_closed_side;
  if (solid.shape == SolidShape::kDomainBox && !is_usable_domain_box) {
    return ContactError::kBadInput;
  }

  // Step 5: the contact code follows a turning solid only up to a quarter turn per step.
  const double turn_during_step = math::Length(ComputeShapeTurningAngularVelocity(solid)) * step_duration;  // rad
  if (turn_during_step <= kMaxTurnPerStep) {
    return ContactError::kNone;
  }
  if (close_call != nullptr) {
    *close_call = ContactCloseCall::kSolidTurnsTooFar;
  }
  return ContactError::kTooCloseToCall;
}

// Checks one solid as the three-argument form does, without saying why a check was too close to call.
//
// Example: a box turning at {0, 0, 2} rad/s with step_duration 1 s: kTooCloseToCall.
// Fails: as the three-argument form.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU inline ContactError CheckSolid(const Solid& solid, double step_duration) {
  return CheckSolid(solid, step_duration, nullptr);
}

// The distance from a point to a solid, and the face that gives it.
struct DistanceToSolid {
  double distance = 0;  // m. Positive outside the solid, negative inside.
  Vec3 normal_out_of_solid{};  // No unit, unit vector, world. The normal of the face that gives the distance.
  SolidFace face = SolidFace::kHighZ;
};

// Computes a solid's pose elapsed seconds after the start of the step.
//
// Method: the origin moves at the linear velocity. The axes turn by angular_velocity * elapsed about the world
//   direction of the angular velocity: axes = math::Rotation(angular_velocity * elapsed) * axes at the step start.
//   The full angular velocity counts, so the axes of a spinning sphere or cylinder turn although its shape does not.
//   Same bits on CPU and GPU when the angular velocity is zero; otherwise the Sin inside math::Rotation may differ
//   in the last bits. At elapsed 0 the pose is the start pose exactly.
// Units: seconds for elapsed.
// Example: a box with world axes turning at {0, 0, pi / 2} rad/s gives, after 1 s, the x column {0, 1, 0} within
//   rounding. Origin {1, 2, 3} moving at {0.5, 0, 0} m/s gives origin {2, 2, 3} after 2 s, axes unchanged.
// Requires: elapsed is finite and zero or more; it is at most the step duration.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU inline SolidPose ComputeSolidPose(const Solid& solid, double elapsed) {
  assert(math::IsFinite(elapsed) && elapsed >= 0);
  const SolidPose start = solid.pose_at_step_start;
  const SolidMotion motion = solid.motion;
  // Method paper <sec-solid-shapes>:  o(t) = o + v_lin t,  A(t) = Rot(omega t) A
  // The turn multiplies from the left, because the angular velocity is a world direction, not an own axis.
  return {
      .origin = start.origin + motion.linear_velocity * elapsed,
      .axes = math::Rotation(motion.angular_velocity * elapsed) * start.axes,
  };
}

// Computes the velocity of the solid's surface at a world point: the velocity of a point fixed to the solid.
//
// Method: linear_velocity + Cross(angular_velocity, point - pose.origin). The full angular velocity counts, spin
//   included. Same bits on CPU and GPU: only + - * in a fixed order.
// Units: m/s, world. The point is in m, world, at the time of the pose.
// Example: linear velocity {1, 0, 0} m/s, angular velocity {0, 0, 2} rad/s, pose origin {0, 0, 0}, point
//   {0, 3, 0}: {1, 0, 0} + {-6, 0, 0} = {-5, 0, 0} m/s.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU constexpr Vec3 ComputeSolidVelocity(const Solid& solid, SolidPose pose, Vec3 point) {
  // Method paper <sec-solid-shapes>:  u = v_lin + omega x (x - o(t))
  const Vec3 arm = point - pose.origin;
  return solid.motion.linear_velocity + math::Cross(solid.motion.angular_velocity, arm);
}

// Computes the distance from a point to a sphere, as ComputeDistanceToSolid does for a sphere: |p| - radius, face
// kCurved, normal p / |p|, written in world axes with solid_axes.
//
// Method: same bits on CPU and GPU: only + - * /, Sqrt and a comparison. At the centre, Dot(p, p) = 0 as computed,
//   every direction is nearest and the normal is the degenerate normal, minus the x column of solid_axes.
// Units: m. own_point is in the sphere's own axes, from its centre.
// Example: radius 0.5, solid_axes kIdentity, own point {0.375, 0, 0.5}: distance 0.125, normal {0.6, 0, 0.8}.
// Requires: solid_axes is a rotation.
// Read: docs/method-paper/fluidity-method.typ <eq:sphere-distance>
FLUIDITY_CPU_AND_GPU inline DistanceToSolid ComputeDistanceToSphere(Vec3 own_point, double radius, Mat3 solid_axes) {
  // Method paper <eq:sphere-distance>:  d = |p| - radius,  n = p / |p|
  const double center_distance_squared = math::Dot(own_point, own_point);
  const double center_distance = math::Sqrt(center_distance_squared);
  const double distance = center_distance - radius;
  // At the centre every direction is nearest: the degenerate normal is the kLowX normal, as a box's tie gives.
  // The test is on the computed square, so the CPU and the GPU take the same branch.
  if (center_distance_squared == 0) {
    const Vec3 own_x_axis = {.x = solid_axes.xx, .y = solid_axes.yx, .z = solid_axes.zx};
    return {.distance = distance, .normal_out_of_solid = -own_x_axis, .face = SolidFace::kCurved};
  }
  const Vec3 own_normal = {
      .x = own_point.x / center_distance,
      .y = own_point.y / center_distance,
      .z = own_point.z / center_distance,
  };
  return {.distance = distance, .normal_out_of_solid = solid_axes * own_normal, .face = SolidFace::kCurved};
}

// Computes the distance from a point to a cylinder, as ComputeDistanceToSolid does for a cylinder, with the normal
// in world axes (written with solid_axes) and the face of the region: rim, wall or cap.
//
// Method: wall_distance = Sqrt(p.x^2 + p.y^2) - radius, cap_distance = |p.z| - half_height, and the distance
//   Sqrt(max(wall_distance, 0)^2 + max(cap_distance, 0)^2) + min(max(wall_distance, cap_distance), 0) in every
//   region. The regions are tried in the order rim, wall, cap; ComputeDistanceToSolid states them and their ties.
//   Same bits on CPU and GPU: only + - * /, Abs, Sqrt and comparisons. A NaN gap gives a NaN distance.
// Units: m. own_point is in the cylinder's own axes, from its centre; its own z axis is the cylinder's axis.
//   half_size is {radius, radius, half_height}, as a Solid holds it.
// Example: half_size {0.5, 0.5, 0.25}, solid_axes kIdentity, own point {0.8, 0, 0.65}: gaps 0.3 and 0.4,
//   distance 0.5, face kHighRim, normal {0.6, 0, 0.8}.
// Requires: solid_axes is a rotation.
// Read: docs/method-paper/fluidity-method.typ <eq:cylinder-distance>
FLUIDITY_CPU_AND_GPU inline DistanceToSolid ComputeDistanceToCylinder(Vec3 own_point, Vec3 half_size, Mat3 solid_axes) {
  // Method paper <eq:cylinder-distance>:  d_wall = rho - radius,  d_cap = |p_z| - half_height,
  //   d = Sqrt(max(d_wall, 0)^2 + max(d_cap, 0)^2) + min(max(d_wall, d_cap), 0)
  const double axis_distance_squared = own_point.x * own_point.x + own_point.y * own_point.y;
  const double axis_distance = math::Sqrt(axis_distance_squared);  // rho
  const double wall_distance = axis_distance - half_size.x;
  const double cap_distance = math::Abs(own_point.z) - half_size.z;
  // One expression for every region, so both sides of a region boundary give the same distance. A NaN in either
  // gap reaches the distance through these comparisons; math::Max would drop it, and a NaN point would get the
  // distance 0.
  const double wall_distance_outside = wall_distance < 0 ? 0 : wall_distance;
  const double cap_distance_outside = cap_distance < 0 ? 0 : cap_distance;
  const double larger_distance = wall_distance < cap_distance ? cap_distance : wall_distance;
  const double larger_distance_inside = larger_distance > 0 ? 0 : larger_distance;
  const double distance =
      math::Sqrt(wall_distance_outside * wall_distance_outside + cap_distance_outside * cap_distance_outside)
      + larger_distance_inside;
  // sigma_z: the cap on the point's side. The middle plane p.z = 0 counts as the high side, as for a plate.
  const bool is_high_side = own_point.z >= 0;
  const double cap_sign = is_high_side ? 1 : -1;
  const Vec3 own_x_axis = {.x = solid_axes.xx, .y = solid_axes.yx, .z = solid_axes.zx};
  const Vec3 own_z_axis = {.x = solid_axes.xz, .y = solid_axes.yz, .z = solid_axes.zz};

  // The branches choose only the normal and the face, in the order rim, wall, cap.
  const bool is_beyond_wall = wall_distance > 0;
  const bool is_beyond_cap = cap_distance > 0;
  if (is_beyond_wall && is_beyond_cap) {
    // Method paper <eq:cylinder-distance>:  n = (d_wall e_rho + d_cap sigma_z e_z) / d
    // Beyond the wall, axis_distance > radius > 0, so e_rho has a direction.
    const Vec3 own_normal = {
        .x = wall_distance * (own_point.x / axis_distance) / distance,
        .y = wall_distance * (own_point.y / axis_distance) / distance,
        .z = cap_distance * cap_sign / distance,
    };
    const SolidFace rim_face = is_high_side ? SolidFace::kHighRim : SolidFace::kLowRim;
    return {.distance = distance, .normal_out_of_solid = solid_axes * own_normal, .face = rim_face};
  }
  // When both gaps are zero or less, a tie between the wall and a cap goes to the wall.
  if (wall_distance >= cap_distance) {
    // On the axis e_rho has no direction: the degenerate normal, as at a sphere's centre. The wall wins there only
    // when -radius >= cap_distance, so only inside a cylinder at least as long as it is wide.
    if (axis_distance_squared == 0) {
      return {.distance = distance, .normal_out_of_solid = -own_x_axis, .face = SolidFace::kCurved};
    }
    const Vec3 own_normal = {.x = own_point.x / axis_distance, .y = own_point.y / axis_distance, .z = 0};
    return {.distance = distance, .normal_out_of_solid = solid_axes * own_normal, .face = SolidFace::kCurved};
  }
  if (is_high_side) {
    return {.distance = distance, .normal_out_of_solid = own_z_axis, .face = SolidFace::kHighZ};
  }
  return {.distance = distance, .normal_out_of_solid = -own_z_axis, .face = SolidFace::kLowZ};
}

// Computes the distance from a world point to a solid at a given pose, positive outside and negative inside, with
// the normal and the face that give it.
//
// Method: turns the point into the solid's own axes, p = Transpose(pose.axes) * (point - pose.origin), then
//   flat solid  distance p.z, face kHighZ, normal the own +z axis.
//   plate       distance |p.z| - half_thickness. Face kHighZ with the own +z axis when p.z >= 0, else kLowZ
//               with the own -z axis.
//   box         distance max over the own axes of |p.axis| - half_size.axis: the gap to the nearest face plane,
//               not the straight-line distance outside an edge or a corner. On the axis that gives it, the low
//               face when p.axis <= 0, else the high face; the normal is that face's own axis, pointing out.
//   domain box  distance min over the closed sides of the gap from the point to that side. The normal points
//               into the box.
//   sphere      distance |p| - radius, the true distance to the surface. Face kCurved, normal p / |p|.
//   cylinder    the true distance to the surface, from the two gaps wall_distance = Sqrt(p.x^2 + p.y^2) - radius
//               and cap_distance = |p.z| - half_height, by one expression for every region:
//               Sqrt(max(wall_distance, 0)^2 + max(cap_distance, 0)^2) + min(max(wall_distance, cap_distance), 0).
//               The region gives the face and the normal. The regions are tried in the order rim, wall, cap.
//               Rim: both gaps are positive. The face is kHighRim when p.z >= 0, else kLowRim. The normal points
//               from the nearest point of the rim circle to the point.
//               Wall: wall_distance >= cap_distance. The face is kCurved. The normal points straight away from the
//               axis.
//               Cap: every other point. The face is the cap on the point's side, kHighZ when p.z >= 0, else kLowZ.
//               The normal is the own +z or -z axis.
//   For the flat faces the normal is a column of pose.axes or its negative, so writing it back in world axes adds
//   no rounding; a round normal is turned back with pose.axes, which rounds. Same bits on CPU and GPU for a given
//   pose: only + - * /, Abs, Sqrt and comparisons. A NaN in the point or the pose gives a NaN distance.
// Ties: when two faces of a box or a domain box give the same distance, the first in the order kLowX, kHighX,
//   kLowY, kHighY, kLowZ, kHighZ is reported, so on the middle plane of a box's axis (p.axis = 0) the low face is
//   reported. A plate's middle plane, p.z = 0, gives kHighZ by the plate's own rule, and so do a cylinder's caps.
//   Inside a cylinder, a tie between the wall and a cap goes to the wall. Where every direction is nearest, at a
//   sphere's centre and on a cylinder's axis in the wall region, the degenerate normal is minus the x column of
//   pose.axes, the normal of kLowX, as a box's tie gives.
// Example: a box at the origin with world axes and half sizes {1, 1, 1}. The point {3, 0, 0}: distance 2, face
//   kHighX, normal {1, 0, 0}. The point {0.9, 0, 0.5}: -0.1, kHighX. The point {-2, -2, 0}: 1, kLowX (a tie
//   with kLowY). The domain box from {0, 0, 0} to {1, 1, 1} with an open top and the point {0.5, 0.5, 0.95}:
//   0.5, kLowX (four sides tie), normal {1, 0, 0}.
//   A sphere at {1, 2, 3} with radius 0.5 and the point {1.375, 2, 3.5}: 0.125, kCurved, normal {0.6, 0, 0.8}.
//   A cylinder along z at the origin with radius 0.5 and half height 0.25. The point {0.8, 0, 0.65}: gaps 0.3 and
//   0.4, distance 0.5, kHighRim, normal {0.6, 0, 0.8}. The point {0.1, 0, 0.2}: gaps -0.4 and -0.05, distance
//   -0.05, kHighZ, normal {0, 0, 1}.
// Requires: pose.axes is a rotation (the pose of a solid that passed CheckSolid, from ComputeSolidPose). The shape
//   is one of SolidShape's values. A domain box has a closed side.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>, <eq:solid-distance>, <eq:sphere-distance>,
//       <eq:cylinder-distance>
FLUIDITY_CPU_AND_GPU inline DistanceToSolid ComputeDistanceToSolid(const Solid& solid, SolidPose pose, Vec3 point) {
  // Steps: 1 turn the point into the solid's own axes  2 distance of the shape
  // Step 1: the axes are a rotation (CheckSolid), so the transpose undoes the turn.
  // Method paper <sec-solid-shapes>:  p = A^T (x - o)
  const Vec3 own_point = math::Transpose(pose.axes) * (point - pose.origin);
  const Vec3 own_x_axis = {.x = pose.axes.xx, .y = pose.axes.yx, .z = pose.axes.zx};
  const Vec3 own_y_axis = {.x = pose.axes.xy, .y = pose.axes.yy, .z = pose.axes.zy};
  const Vec3 own_z_axis = {.x = pose.axes.xz, .y = pose.axes.yz, .z = pose.axes.zz};
  const Vec3 half_size = solid.half_size;

  // Step 2: one formula per shape.
  switch (solid.shape) {
    case SolidShape::kFlatSolid:
      // Method paper <eq:solid-distance>:  d = p_z
      return {.distance = own_point.z, .normal_out_of_solid = own_z_axis, .face = SolidFace::kHighZ};
    case SolidShape::kPlate: {
      // Method paper <eq:solid-distance>:  d = |p_z| - half_thickness
      // The middle plane itself, p.z = 0, counts as the kHighZ side.
      const double distance = math::Abs(own_point.z) - half_size.z;
      if (own_point.z >= 0) {
        return {.distance = distance, .normal_out_of_solid = own_z_axis, .face = SolidFace::kHighZ};
      }
      return {.distance = distance, .normal_out_of_solid = -own_z_axis, .face = SolidFace::kLowZ};
    }
    case SolidShape::kBox: {
      // Method paper <eq:solid-distance>:  d = max over k of (|p_k| - half_size_k)
      // The gap to the face plane of one own axis, on the point's side of the box.
      const auto compute_axis_gap =
          [](double own_coordinate, double axis_half_size, Vec3 own_axis, SolidFace low_face) -> DistanceToSolid {
        const double gap = math::Abs(own_coordinate) - axis_half_size;
        // At own_coordinate = 0 both faces of the axis give the same gap; the tie keeps the low face, the first.
        if (own_coordinate <= 0) {
          return {.distance = gap, .normal_out_of_solid = -own_axis, .face = low_face};
        }
        // Each high face follows its low face in SolidFace.
        const auto high_face = static_cast<SolidFace>(static_cast<std::uint8_t>(low_face) + 1);
        return {.distance = gap, .normal_out_of_solid = own_axis, .face = high_face};
      };
      // A later axis wins only with a strictly larger gap, so a tie keeps the earlier face.
      DistanceToSolid largest = compute_axis_gap(own_point.x, half_size.x, own_x_axis, SolidFace::kLowX);
      const DistanceToSolid along_y = compute_axis_gap(own_point.y, half_size.y, own_y_axis, SolidFace::kLowY);
      if (along_y.distance > largest.distance) {
        largest = along_y;
      }
      const DistanceToSolid along_z = compute_axis_gap(own_point.z, half_size.z, own_z_axis, SolidFace::kLowZ);
      if (along_z.distance > largest.distance) {
        largest = along_z;
      }
      return largest;
    }
    case SolidShape::kDomainBox: {
      // Method paper <eq:solid-distance>:  d = min over closed sides of (half_size_k - sigma p_k)
      // sigma is -1 for a low side and +1 for a high side. Each side's normal points into the box.
      // A later side wins only with a strictly smaller gap, so a tie keeps the earlier side.
      DistanceToSolid nearest;
      bool has_nearest = false;
      const auto keep_if_nearer = [&nearest,
                                   &has_nearest](bool is_closed, double gap, Vec3 normal_into_box, SolidFace face) {
        const bool is_nearer = !has_nearest || gap < nearest.distance;
        if (is_closed && is_nearer) {
          nearest = {.distance = gap, .normal_out_of_solid = normal_into_box, .face = face};
          has_nearest = true;
        }
      };
      const ClosedSides closed = solid.closed;
      keep_if_nearer(closed.left, half_size.x + own_point.x, own_x_axis, SolidFace::kLowX);
      keep_if_nearer(closed.right, half_size.x - own_point.x, -own_x_axis, SolidFace::kHighX);
      keep_if_nearer(closed.front, half_size.y + own_point.y, own_y_axis, SolidFace::kLowY);
      keep_if_nearer(closed.back, half_size.y - own_point.y, -own_y_axis, SolidFace::kHighY);
      keep_if_nearer(closed.bottom, half_size.z + own_point.z, own_z_axis, SolidFace::kLowZ);
      keep_if_nearer(closed.top, half_size.z - own_point.z, -own_z_axis, SolidFace::kHighZ);
      // CheckSolid refuses a domain box with every side open.
      assert(has_nearest);
      return nearest;
    }
    case SolidShape::kSphere:
      return ComputeDistanceToSphere(own_point, half_size.x, pose.axes);
    case SolidShape::kCylinder:
      return ComputeDistanceToCylinder(own_point, half_size, pose.axes);
  }
  // An unknown shape has no formula, so reaching this line is a caller bug: CheckSolid refuses it.
  assert(false && "ComputeDistanceToSolid has no formula for this shape");
  return DistanceToSolid{};
}

// The Blender names of the six sides of the domain box, in SolidFace order.
enum class DomainSide : std::uint8_t {
  kLeft,  // side_left, -X: SolidFace::kLowX.
  kRight,  // side_right, +X: SolidFace::kHighX.
  kFront,  // side_front, -Y: SolidFace::kLowY.
  kBack,  // side_back, +Y: SolidFace::kHighY.
  kBottom,  // side_bottom, -Z: SolidFace::kLowZ.
  kTop,  // side_top, +Z: SolidFace::kHighZ.
};

// Converts a domain side to the face of the domain box it names.
//
// Example: ToSolidFace(DomainSide::kBottom) = SolidFace::kLowZ.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU constexpr SolidFace ToSolidFace(DomainSide side) {
  switch (side) {
    case DomainSide::kLeft:
      return SolidFace::kLowX;
    case DomainSide::kRight:
      return SolidFace::kHighX;
    case DomainSide::kFront:
      return SolidFace::kLowY;
    case DomainSide::kBack:
      return SolidFace::kHighY;
    case DomainSide::kBottom:
      return SolidFace::kLowZ;
    case DomainSide::kTop:
      break;
  }
  return SolidFace::kHighZ;
}

}  // namespace fluidity::contact
