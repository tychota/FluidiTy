// The shapes of solids, their pose and motion during one step, and the check that a solid is usable.
//
// Purpose: describe every solid that particles cannot enter in one form, Solid, that the contact code reads, and
//   check it once per step before any particle is moved.
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
// Owns:    SolidShape, SolidFace, SolidSide, PlateSide, FlatSolid, Plate, Box, Sphere, Cylinder, ClosedSides,
//          DomainBox, SolidMotion, SolidPose, Solid, kNoSolid, kMaxTurnPerStep, ToSolid,
//          ComputeShapeTurningAngularVelocity, CheckSolid, DomainSide, ToSolidFace
// Uses:    Vec3, math::Cross, math::Dot, math::Length, math::Normalized, math::IsFinite (base/vec3.h),
//          Mat3, kIdentity, math::Determinant (base/mat3.h), Axis, math::UnitVector (base/axis.h),
//          math::Abs, math::Max (base/scalar_math.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h),
//          ContactError, ContactCloseCall (contact/contact_error.h)
// Used by: every contact function, the solids module, which builds the solids of a scene, and the step
// Read:    docs/method-paper/fluidity-method.typ <sec-solid-shapes>, <sec-round-solids>
//          Hu 2018, doi:10.1145/3197517.3201293

#pragma once

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
