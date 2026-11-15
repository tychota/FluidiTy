// The private steps of one particle's move through the solids: the distance the particle sees to one solid, and the
// first hit of a straight path on one solid.
//
// Purpose: give FindFirstHit, and later the rest of a particle's move, the functions that evaluate one solid for one
//   particle, so one switch chooses each shape's formula and each shape keeps its own rules.
//
// Idea: a solid that does not turn only translates. Measured from the solid, the particle then moves in a straight
//   line at its velocity minus the solid's, and its distance to each flat face plane changes linearly with time. The
//   gap is that distance minus the contact tolerance: a sweep runs against the solid grown by the tolerance, so the
//   hit is the moment the gap reaches zero. A flat solid, a plate and a box are convex: the path is inside the grown
//   solid while it is behind every face plane at once. It enters at the latest of the times it crosses the planes it
//   starts in front of, if that comes no later than the earliest time it leaves a plane it starts behind. Each closed
//   side of the domain box is a flat face of its own. Before any of this, the touching rule settles a start that
//   already rests on the surface.
//   Example: a box of half size 0.5 m at the origin and a path from {-1, 0.1, 0.2} at {2, 0, 0} m/s. The path crosses
//   the plane of kLowX at (0.5 - tol) / 2 s and leaves the plane of kHighX at (1.5 + tol) / 2 s; the y and z planes
//   are neither crossed nor left. The hit is on kLowX at 0.25 - tol / 2 s.
//   A sphere and a cylinder are round, so the gap to them does not change linearly. Measured from the solid, the
//   path meets the grown sphere where its distance to the centre is radius + tol: the root of a quadratic in time. The
//   cylinder is grown with sharp edges: it is the part shared by its grown wall, unbounded along the axis, and the
//   space between its grown caps' planes, so the path is inside it while it is inside both, and enters it at the
//   later of the two entries, as it enters a box at the latest crossing into a face plane. A spin about the sphere's
//   centre or the cylinder's axis moves the surface but not the shape, so it does not make the solid turn here.
//   Example: a sphere of radius 0.5 m at the origin and a path from {-2, 0, 0} at {1, 0, 0} m/s: the hit is at
//   1.5 - tol s, normal {-1, 0, 0}.
//
// Words:
//   touching rule  how a sweep treats a start whose gap is in [-tol, 0] for the surface that gives the distance: the
//                  particle rests on that surface, which is hit only when the path pushes into it by more than tol
//                  during the sweep. The solid lies behind that surface's plane, so a path that stays within tol of
//                  the plane cannot enter it. On a sphere or a cylinder the solid is convex and its true distance is
//                  a convex function along the path, which gives the same rule
//   rim corner     the part of a cylinder grown with sharp edges that is farther than tol from the rim: a point
//                  beyond the wall and beyond a cap, by at most tol from each
// Owns:    Sweep, FacePlane, StartCase, OwnPath, HitTime, ComputeCoordinate, ComputeColumn, ToFacePlane,
//          ComputeFaceDistance, ComputeDistanceForParticle, ComputeStartCase, ComputeOwnPath, ToFirstHit,
//          FindEntryThroughFacePlanes, FindHitOnConvexSolid, FindHitOnDomainBox, FindSphereHit, TimeInterval,
//          ComputeTimeInsideGrownWall, ComputeTimeBetweenGrownCaps, FindCylinderHit, FindFirstHitOnSolid
// Uses:    Vec3, math::Dot, math::Length (base/vec3.h), Mat3, kIdentity, math::Transpose (base/mat3.h),
//          Axis (base/axis.h), math::Abs, math::Sqrt (base/scalar_math.h), FLUIDITY_CPU_AND_GPU
//          (base/cpu_and_gpu.h), ContactError (contact/contact_error.h), Solid, SolidShape, SolidFace, SolidSide,
//          PlateSide, SolidPose, ClosedSides, DistanceToSolid, kNoSolid, ComputeSolidPose, ComputeDistanceToSolid,
//          ComputeDistanceToSphere, ComputeDistanceToCylinder, ComputeShapeTurningAngularVelocity
//          (contact/solid_shapes.h), SolidContactInputs, FirstHit (contact/solid_contact_types.h)
// Used by: contact/solid_contact.h, whose FindFirstHit calls FindFirstHitOnSolid for each solid
// Read:    docs/method-paper/fluidity-method.typ <sec-first-hit>, <eq:hit-time>, <proc-find-first-hit>,
//          <sec-round-solids>, <eq:sphere-hit-time>, <eq:cylinder-hit-time>, <proc-sphere-first-hit>,
//          <proc-cylinder-first-hit>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// One straight path of a sweep: start + velocity_for_this_step * t for 0 <= t <= duration, with every solid posed
// start_elapsed + t seconds into the step.
struct Sweep {
  Vec3 start{};  // m, world.
  Vec3 velocity_for_this_step{};  // m/s, world.
  double start_elapsed = 0;  // s since the start of the step: the part of the step already used.
  double duration = 0;  // s, zero or more.
};

// The plane of one flat face in the solid's own axes. With p a point in those axes, from the solid's origin, the
// distance from p to the face is sign_out_of_solid * p.axis - offset: positive on the side the face looks to.
struct FacePlane {
  SolidFace face = SolidFace::kHighZ;
  Axis axis = Axis::kZ;  // The own axis the face plane is at right angles to.
  double sign_out_of_solid = 1;  // No unit, +1 or -1: the face looks along +axis or -axis.
  double offset = 0;  // m. How far the plane is from the origin, along sign_out_of_solid times the own axis.
};

// How a sweep starts against one surface, by the gap at its start and the touching rule.
enum class StartCase : std::uint8_t {
  kOutside,  // The gap is positive: the start is outside the grown solid, and the closed form decides.
  kRestingAndFree,  // The touching rule: the particle rests on the surface and the path does not push into it.
  kHitAtStart,  // The path pushes into the surface it rests on, or starts deeper than tol inside: a hit at time 0.
};

// A sweep measured from a solid that does not turn, in the solid's own axes at the pose of the sweep's start.
struct OwnPath {
  Vec3 start{};  // m, own axes, from the solid's origin.
  Vec3 velocity{};  // m/s, own axes: velocity_for_this_step minus the solid's linear velocity.
};

// When a closed form in a solid's own axes finds the first hit of a sweep, or that the sweep is free.
struct HitTime {
  bool has_hit = false;  // false: the sweep is free.
  double elapsed = 0;  // s, since the start of the sweep, zero or more. 0 when has_hit is false.
};

// The times a path measured from a solid spends inside one part of its grown shape, from entry to exit, or never.
struct TimeInterval {
  bool is_empty = true;  // true: the path is never inside during the sweep.
  double entry_elapsed = 0;  // s, since the start of the sweep, zero or more.
  bool has_exit = false;  // false: once inside, the path never leaves.
  double exit_elapsed = 0;  // s, since the start of the sweep, at least entry_elapsed up to rounding.
};

// Returns the coordinate of vector along one axis.
FLUIDITY_CPU_AND_GPU constexpr double ComputeCoordinate(Vec3 vector, Axis axis) {
  switch (axis) {
    case Axis::kX:
      return vector.x;
    case Axis::kY:
      return vector.y;
    case Axis::kZ:
      break;
  }
  return vector.z;
}

// Returns the column of axes for one own axis: that axis in world axes.
FLUIDITY_CPU_AND_GPU constexpr Vec3 ComputeColumn(Mat3 axes, Axis axis) {
  switch (axis) {
    case Axis::kX:
      return {.x = axes.xx, .y = axes.yx, .z = axes.zx};
    case Axis::kY:
      return {.x = axes.xy, .y = axes.yy, .z = axes.zy};
    case Axis::kZ:
      break;
  }
  return {.x = axes.xz, .y = axes.yz, .z = axes.zz};
}

// Converts the side a particle is committed to into the one face plane it sees of that plate (plan decision P3): kHighZ
// at own z = +half_thickness for kAlongNormal, kLowZ at own z = -half_thickness for kAgainstNormal.
//
// Units: m for half_thickness.
// Example: kAgainstNormal, half_thickness 0.25: face kLowZ, sign_out_of_solid -1, offset 0.25, so a point at own z = -1
//   is at distance 0.75.
// Requires: side is kAlongNormal or kAgainstNormal.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU constexpr FacePlane ToFacePlane(SolidSide side, double half_thickness) {
  assert(side == SolidSide::kAlongNormal || side == SolidSide::kAgainstNormal);
  const bool is_along_normal = side == SolidSide::kAlongNormal;
  return {
      .face = is_along_normal ? SolidFace::kHighZ : SolidFace::kLowZ,
      .axis = Axis::kZ,
      .sign_out_of_solid = is_along_normal ? 1.0 : -1.0,
      .offset = half_thickness,
  };
}

// Computes the distance from a world point to one face plane at a given pose of its solid, with the face's normal.
//
// Method: p = Transpose(pose.axes) * (point - pose.origin), as ComputeDistanceToSolid turns it; the distance is
//   sign_out_of_solid * p.axis - offset and the normal is sign_out_of_solid times that column of pose.axes. Multiplying
//   by +1 or -1 is exact, so for a face of a flat solid, a plate, a box or a domain box this is the distance
//   ComputeDistanceToSolid gives for that face, bit for bit. Same bits on CPU and GPU: only + - * and a switch.
// Units: m, world, at the time of the pose.
// Example: a box at the origin with world axes and half size 0.5; face kLowX {kLowX, kX, -1, 0.5} and the point
//   {-1, 0.1, 0.2}: distance 0.5, normal {-1, 0, 0}.
// Requires: pose.axes is a rotation.
// Read: docs/method-paper/fluidity-method.typ <eq:solid-distance>
FLUIDITY_CPU_AND_GPU constexpr DistanceToSolid ComputeFaceDistance(SolidPose pose, Vec3 point, FacePlane face_plane) {
  // Method paper <sec-solid-shapes>:  p = A^T (x - o)
  const Vec3 own_point = math::Transpose(pose.axes) * (point - pose.origin);
  return {
      .distance = face_plane.sign_out_of_solid * ComputeCoordinate(own_point, face_plane.axis) - face_plane.offset,
      .normal_out_of_solid = face_plane.sign_out_of_solid * ComputeColumn(pose.axes, face_plane.axis),
      .face = face_plane.face,
  };
}

// Computes the distance from a world point to one solid as a particle committed to plate_side sees it: a plate the
// particle is committed to is the half space behind its committed face (plan decision P3); every other solid gives
// ComputeDistanceToSolid.
//
// Method: when plate_side.plate_id is the solid's solid_id, the solid is that plate and the distance is
//   s * p.z - half_thickness, s = +1 for kAlongNormal and -1 for kAgainstNormal, with face kHighZ or kLowZ and normal
//   s times the own z axis (ComputeFaceDistance of ToFacePlane). A start inside the thickness is then inside by more
//   than half the thickness on one side. plate_id 0 names no solid, since CheckSolid refuses the id 0. Same bits on
//   CPU and GPU as ComputeDistanceToSolid.
// Units: m, world, at the time of the pose.
// Example: a plate of id 5 through the origin with normal {0, 0, 1} and half thickness 0.01, and the point
//   {0, 0, -0.005}. Committed to {5, kAlongNormal}: -0.015, kHighZ, normal {0, 0, 1}. Not committed: -0.005, kLowZ,
//   normal {0, 0, -1}.
// Requires: solid_index < inputs.solid_count. When plate_side.plate_id is this solid's solid_id, the solid is a
//   plate and plate_side.side is kAlongNormal or kAgainstNormal. pose.axes is a rotation.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>, <eq:solid-distance>
FLUIDITY_CPU_AND_GPU inline DistanceToSolid ComputeDistanceForParticle(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    SolidPose pose,
    Vec3 point,
    PlateSide plate_side
) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  const bool is_committed_to_this_plate = plate_side.plate_id == solid.solid_id;
  if (!is_committed_to_this_plate) {
    return ComputeDistanceToSolid(solid, pose, point);
  }
  assert(solid.shape == SolidShape::kPlate);
  return ComputeFaceDistance(pose, point, ToFacePlane(plate_side.side, solid.half_size.z));
}

// Computes how a sweep starts against one surface of a solid that does not turn, from the distance at the start and
// the path's velocity measured from the solid, the two in the same axes.
//
// Method: with tol = inputs.contact_tolerance, gap = at_start.distance - tol and
//   rate = Dot(at_start.normal_out_of_solid, relative_velocity) (plan decisions P2 and P16):
//   gap > 0           kOutside.
//   gap < -tol        kHitAtStart: deeper inside than one tolerance.
//   otherwise         the touching rule: kHitAtStart when rate * duration < -tol, kRestingAndFree when it is -tol or
//                     more. The limit is free: such a path ends on the surface, not inside it.
//   The rule is the same for a flat face and a round surface. Same bits on CPU and GPU: only + - * and comparisons.
// Units: m for the distance, m/s for the velocity, s for the duration.
// Example: tol 2^-30, at_start {tol, {0, 0, 1}, kHighZ} (the rest gap): the relative velocity {5, 0, -tol / 4} for
//   1 s gives kRestingAndFree; {5, 0, -1} gives kHitAtStart. With at_start.distance 2 tol: kOutside.
// Requires: at_start is the distance of the surface that gives the distance to the solid at the sweep's start: a face
//   of the solid, or one closed side of a domain box. relative_velocity is in the axes of at_start's normal.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>, <sec-round-solids>
FLUIDITY_CPU_AND_GPU inline StartCase
ComputeStartCase(const SolidContactInputs& inputs, DistanceToSolid at_start, Vec3 relative_velocity, double duration) {
  const double contact_tolerance = inputs.contact_tolerance;
  const double start_gap = at_start.distance - contact_tolerance;  // m
  if (start_gap > 0) {
    return StartCase::kOutside;
  }
  // MoveParticle never sweeps from deeper than one tolerance inside; this answer only keeps a caller's error safe.
  if (start_gap < -contact_tolerance) {
    return StartCase::kHitAtStart;
  }
  // The touching rule: the solid lies behind this surface's plane (a round solid behind its tangent plane, since it
  // is convex), and along the path the gap is at least start_gap + rate * t >= start_gap - tol while
  // rate * duration >= -tol, so the path never goes below distance 0 by more than the window absorbs.
  // Method paper <eq:hit-time>:  rate = n . (v - v_lin)
  const double gap_rate = math::Dot(at_start.normal_out_of_solid, relative_velocity);  // m/s
  const bool pushes_into_surface = gap_rate * duration < -contact_tolerance;
  return pushes_into_surface ? StartCase::kHitAtStart : StartCase::kRestingAndFree;
}

// Computes how a sweep starts against one surface of one solid that does not turn, from the distance at the start, in
// world axes.
//
// Method: the form that takes a relative velocity, with velocity_for_this_step - linear_velocity and the sweep's
//   duration.
// Units: m for the distance, m/s for the velocities, s for the duration.
// Example: a fixed flat solid with normal {0, 0, 1}, tol 2^-30 and at_start.distance tol (the rest gap): the velocity
//   {5, 0, -tol / 4} for 1 s gives kRestingAndFree; {5, 0, -1} gives kHitAtStart. With at_start.distance 2 tol:
//   kOutside.
// Requires: solid_index < inputs.solid_count, and the solid does not turn. at_start is the distance of the surface
//   that gives the distance to the solid at the sweep's start: a face of the solid, or one closed side of a domain box.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>
FLUIDITY_CPU_AND_GPU inline StartCase
ComputeStartCase(const SolidContactInputs& inputs, std::uint32_t solid_index, Sweep sweep, DistanceToSolid at_start) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  const Vec3 relative_velocity = sweep.velocity_for_this_step - solid.motion.linear_velocity;
  return ComputeStartCase(inputs, at_start, relative_velocity, sweep.duration);
}

// Computes a sweep measured from one solid that does not turn, in the solid's own axes at the pose of start_elapsed.
//
// Method: with pose = ComputeSolidPose(solid, sweep.start_elapsed), start = Transpose(pose.axes) * (sweep.start -
//   pose.origin) and velocity = Transpose(pose.axes) * (velocity_for_this_step - linear_velocity). The start is the
//   point ComputeDistanceToSolid turns the sweep's start into at that pose, bit for bit, so a distance computed from it
//   in own axes equals the one measured in world axes. Same bits on CPU and GPU when the angular velocity is zero; a
//   spin makes ComputeSolidPose call Sin.
// Units: m and m/s, own axes.
// Example: a solid with world axes at {1, 0, 0} moving at {0.5, 0, 0} m/s, start_elapsed 2 s: the origin is at
//   {2, 0, 0}, so the start {3, 1, 0} moving at {1, 0, 0} m/s gives start {1, 1, 0} and velocity {0.5, 0, 0}.
// Requires: solid_index < inputs.solid_count.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>, <sec-round-solids>
FLUIDITY_CPU_AND_GPU inline OwnPath
ComputeOwnPath(const SolidContactInputs& inputs, std::uint32_t solid_index, Sweep sweep) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  const SolidPose pose = ComputeSolidPose(solid, sweep.start_elapsed);
  // Method paper <sec-solid-shapes>:  p = A^T (x - o)
  const Mat3 to_own_axes = math::Transpose(pose.axes);
  return {
      .start = to_own_axes * (sweep.start - pose.origin),
      .velocity = to_own_axes * (sweep.velocity_for_this_step - solid.motion.linear_velocity),
  };
}

// Converts the time a closed form found into the first hit on one solid, with the distance, normal and face at the
// hit position.
//
// Method: a free sweep gives FirstHit{} (solid_index kNoSolid). Otherwise at_hit is ComputeDistanceToSolid at the
//   position sweep.start + velocity_for_this_step * elapsed, at the pose of start_elapsed + elapsed: the face comes
//   from the region of the hit position, so a cylinder hit in its rim corner reports a rim (plan decision P15). A
//   hit at 0 gives the distance at the start, the one the start case read.
// Units: s for the time; m and m/s, world, for the sweep.
// Example: a fixed sphere of radius 0.5 m at the origin, the sweep from {-2, 0, 0} at {1, 0, 0} m/s and a hit at
//   1.5 - tol s: at_hit is tol, normal {-1, 0, 0}, face kCurved, up to rounding.
// Requires: solid_index < inputs.solid_count; hit_time is that solid's closed form for this sweep.
// Read: docs/method-paper/fluidity-method.typ <sec-round-solids>, <proc-find-first-hit>
FLUIDITY_CPU_AND_GPU inline FirstHit
ToFirstHit(const SolidContactInputs& inputs, std::uint32_t solid_index, Sweep sweep, HitTime hit_time) {
  assert(solid_index < inputs.solid_count);
  if (!hit_time.has_hit) {
    return FirstHit{};
  }
  const Solid& solid = inputs.solids[solid_index];
  const Vec3 hit_position = sweep.start + sweep.velocity_for_this_step * hit_time.elapsed;
  const SolidPose hit_pose = ComputeSolidPose(solid, sweep.start_elapsed + hit_time.elapsed);
  return {
      .elapsed = hit_time.elapsed,
      .solid_index = solid_index,
      .at_hit = ComputeDistanceToSolid(solid, hit_pose, hit_position),
  };
}

// Finds where a straight path enters a convex solid that does not turn, given the planes of its flat faces: the latest
// crossing into a face plane, if no plane is left before it and it comes within the duration.
//
// Method: poses the solid at sweep.start_elapsed and turns the start and the velocity relative to the solid into its
//   own axes. For each face plane, gap = sign_out_of_solid * p.axis - offset - tol and rate = sign_out_of_solid *
//   u.axis: gap > 0 and rate >= 0  the path never reaches this plane: no hit. gap > 0 and rate < 0   the path crosses
//   into the plane at gap / -rate; entry is the latest of these. gap <= 0 and rate > 0  the path leaves through the
//   plane at -gap / rate; exit is the earliest of these. The path hits when some plane is crossed, entry <= exit and
//   entry <= duration: both limits count as a hit. The face reported is the one that gives the entry, and at_hit is
//   ComputeFaceDistance of that face at the hit position start + velocity_for_this_step * entry, at the pose of
//   start_elapsed + entry: one tolerance up to rounding. Same bits on CPU and GPU: only + - * / and comparisons.
// Units: m, m/s and s, world.
// Ties: equal entry times keep the face that comes first in the array, which lists faces in face order.
// Example: the box of the file header's example, planes kLowX to kHighZ: entry (0.5 - tol) / 2 through kLowX, exit
//   (1.5 + tol) / 2 through kHighX: a hit at 0.25 - tol / 2 s on kLowX, at distance tol.
// Requires: solid_index < inputs.solid_count, and the solid does not turn. face_planes points to face_plane_count
//   planes, at least one, whose half spaces behind them make up the solid. The start is outside the grown solid.
// Read: docs/method-paper/fluidity-method.typ <eq:hit-time>, <sec-first-hit>
FLUIDITY_CPU_AND_GPU inline FirstHit FindEntryThroughFacePlanes(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    Sweep sweep,
    const FacePlane* face_planes,
    int face_plane_count
) {
  assert(solid_index < inputs.solid_count);
  assert(face_planes != nullptr && face_plane_count > 0);
  // Steps: 1 the path in the solid's own axes  2 entry and exit over the face planes  3 the hit
  // Step 1: the solid only translates, so measured from it the path is a straight line at the relative velocity.
  const Solid& solid = inputs.solids[solid_index];
  const double contact_tolerance = inputs.contact_tolerance;
  const OwnPath own_path = ComputeOwnPath(inputs, solid_index, sweep);
  const Vec3 own_start = own_path.start;
  const Vec3 own_velocity = own_path.velocity;

  // Step 2: the gaps are computed as ComputeFaceDistance computes the distance, minus tol, so a plane's gap here has
  // the bits ComputeStartCase saw for the same face.
  // Method paper <eq:hit-time>:  g(t) = g(0) + rate t,  t_hit = g(0) / -rate
  int entry_index = -1;
  double entry_elapsed = 0;  // s
  bool has_exit = false;
  double exit_elapsed = 0;  // s
  for (int plane_index = 0; plane_index < face_plane_count; ++plane_index) {
    const FacePlane face_plane = face_planes[plane_index];
    const double sign_out_of_solid = face_plane.sign_out_of_solid;
    const double start_gap =
        sign_out_of_solid * ComputeCoordinate(own_start, face_plane.axis) - face_plane.offset - contact_tolerance;
    const double gap_rate = sign_out_of_solid * ComputeCoordinate(own_velocity, face_plane.axis);  // m/s
    if (start_gap > 0) {
      // The solid lies behind this plane, so a path that never reaches the plane never reaches the solid.
      if (gap_rate >= 0) {
        return FirstHit{};
      }
      const double crossing_elapsed = start_gap / -gap_rate;  // s
      // A later plane wins only when strictly later, so equal entry times keep the first face.
      if (entry_index < 0 || crossing_elapsed > entry_elapsed) {
        entry_index = plane_index;
        entry_elapsed = crossing_elapsed;
      }
    } else if (gap_rate > 0) {
      const double leaving_elapsed = -start_gap / gap_rate;  // s
      if (!has_exit || leaving_elapsed < exit_elapsed) {
        has_exit = true;
        exit_elapsed = leaving_elapsed;
      }
    }
  }

  // Step 3: entry == exit is a path that meets the grown solid at one point, an edge or a corner: a hit.
  const bool crosses_a_plane = entry_index >= 0;
  const bool enters_before_leaving = !has_exit || entry_elapsed <= exit_elapsed;
  const bool enters_within_sweep = entry_elapsed <= sweep.duration;
  if (!crosses_a_plane || !enters_before_leaving || !enters_within_sweep) {
    return FirstHit{};
  }
  const Vec3 hit_position = sweep.start + sweep.velocity_for_this_step * entry_elapsed;
  const SolidPose hit_pose = ComputeSolidPose(solid, sweep.start_elapsed + entry_elapsed);
  return {
      .elapsed = entry_elapsed,
      .solid_index = solid_index,
      .at_hit = ComputeFaceDistance(hit_pose, hit_position, face_planes[entry_index]),
  };
}

// Finds the first hit of a straight path on a flat solid, a plate or a box that does not turn: the touching rule on
// the surface that gives the distance, then the entry through the solid's face planes.
//
// Method: at_start = ComputeDistanceForParticle at the start, at the pose of sweep.start_elapsed, with the particle's
//   plate side. ComputeStartCase of at_start: kHitAtStart gives a hit at 0 with at_start; kRestingAndFree gives no
//   hit, for the whole solid, which lies behind the surface the particle rests on; kOutside gives
//   FindEntryThroughFacePlanes. The planes' largest distance is the distance at_start holds, bit for bit, so a start
//   outside by one test is outside by the other.
// Units: m, m/s and s, world.
// Example: a fixed flat solid with normal {0, 0, 1} and a start at z = tol moving at {5, 0, -1} for 1 s: a hit at 0,
//   distance tol, face kHighZ.
// Requires: as FindEntryThroughFacePlanes, without the condition on the start; the planes are those of the solid as
//   the particle's plate side sees it.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>, <proc-find-first-hit>
FLUIDITY_CPU_AND_GPU inline FirstHit FindHitOnConvexSolid(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    Sweep sweep,
    const FacePlane* face_planes,
    int face_plane_count
) {
  assert(solid_index < inputs.solid_count);
  const SolidPose pose = ComputeSolidPose(inputs.solids[solid_index], sweep.start_elapsed);
  const DistanceToSolid at_start =
      ComputeDistanceForParticle(inputs, solid_index, pose, sweep.start, inputs.plate_side);
  switch (ComputeStartCase(inputs, solid_index, sweep, at_start)) {
    case StartCase::kOutside:
      return FindEntryThroughFacePlanes(inputs, solid_index, sweep, face_planes, face_plane_count);
    case StartCase::kRestingAndFree:
      return FirstHit{};
    case StartCase::kHitAtStart:
      break;
  }
  return {.elapsed = 0, .solid_index = solid_index, .at_hit = at_start};
}

// Finds the first hit of a straight path on the domain box: each closed side is a flat face of its own, with its own
// touching rule, and the earliest hit over the sides wins.
//
// Method: for each closed side in face order, at_start = ComputeFaceDistance of that side at the start, at the pose
//   of sweep.start_elapsed (the distance into the box, as ComputeDistanceToSolid gives for that side), then
//   ComputeStartCase: kHitAtStart is a hit at 0 on that side, kRestingAndFree no hit on it, kOutside
//   FindEntryThroughFacePlanes of that one plane. A particle in a corner rests on two or three sides at once, so
//   every side needs its own touching rule, not only the nearest one.
// Units: m, m/s and s, world.
// Ties: equal times keep the side that comes first in face order.
// Example: the domain box [0, 1]^3 with an open top and a start {tol, tol, 0.5} moving at {-tol / 4, -tol / 4, 0.25}
//   for 1 s: no hit (it rests on side_left and side_front). At {0, -1, 0.25}: a hit at 0 on kLowY, normal {0, 1, 0}.
// Requires: solid_index < inputs.solid_count, and that solid is a domain box with a closed side.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>, <proc-find-first-hit>
FLUIDITY_CPU_AND_GPU inline FirstHit
FindHitOnDomainBox(const SolidContactInputs& inputs, std::uint32_t solid_index, Sweep sweep) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  assert(solid.shape == SolidShape::kDomainBox);
  const SolidPose pose = ComputeSolidPose(solid, sweep.start_elapsed);
  const Vec3 half_size = solid.half_size;
  const ClosedSides closed = solid.closed;
  FirstHit earliest;
  // A later side wins only when strictly earlier, so a tie keeps the side that comes first.
  const auto keep_if_earlier = [&](bool is_closed, FacePlane side_plane) {
    if (!is_closed) {
      return;
    }
    const DistanceToSolid at_start = ComputeFaceDistance(pose, sweep.start, side_plane);
    FirstHit side_hit;
    switch (ComputeStartCase(inputs, solid_index, sweep, at_start)) {
      case StartCase::kOutside:
        side_hit = FindEntryThroughFacePlanes(inputs, solid_index, sweep, &side_plane, 1);
        break;
      case StartCase::kRestingAndFree:
        break;
      case StartCase::kHitAtStart:
        side_hit = {.elapsed = 0, .solid_index = solid_index, .at_hit = at_start};
        break;
    }
    const bool is_hit = side_hit.solid_index != kNoSolid;
    const bool is_earlier = earliest.solid_index == kNoSolid || side_hit.elapsed < earliest.elapsed;
    if (is_hit && is_earlier) {
      earliest = side_hit;
    }
  };
  // Each side's distance is into the box: p.axis + half_size from a low side, half_size - p.axis from a high side.
  keep_if_earlier(
      closed.left,
      {.face = SolidFace::kLowX, .axis = Axis::kX, .sign_out_of_solid = 1, .offset = -half_size.x}
  );
  keep_if_earlier(
      closed.right,
      {.face = SolidFace::kHighX, .axis = Axis::kX, .sign_out_of_solid = -1, .offset = -half_size.x}
  );
  keep_if_earlier(
      closed.front,
      {.face = SolidFace::kLowY, .axis = Axis::kY, .sign_out_of_solid = 1, .offset = -half_size.y}
  );
  keep_if_earlier(
      closed.back,
      {.face = SolidFace::kHighY, .axis = Axis::kY, .sign_out_of_solid = -1, .offset = -half_size.y}
  );
  keep_if_earlier(
      closed.bottom,
      {.face = SolidFace::kLowZ, .axis = Axis::kZ, .sign_out_of_solid = 1, .offset = -half_size.z}
  );
  keep_if_earlier(
      closed.top,
      {.face = SolidFace::kHighZ, .axis = Axis::kZ, .sign_out_of_solid = -1, .offset = -half_size.z}
  );
  return earliest;
}

// Finds the first hit of a straight path on a sphere, measured from the sphere in its own axes: the start rules, then
// the first root of the path on the sphere grown by the tolerance.
//
// Method: with p0 = own_path.start, u = own_path.velocity, tol = inputs.contact_tolerance, R = radius + tol:
//   1 at_start = ComputeDistanceToSphere(p0, radius, kIdentity), the bits ComputeDistanceToSolid gives in world axes,
//     and ComputeStartCase of it with u (plan decision P16): kHitAtStart is a hit at 0, kRestingAndFree is free.
//   2 kOutside, so the gap g0 = |p0| - radius - tol is positive. A = Dot(u, u), B = Dot(p0, u), C = g0 * (|p0| + R),
//     which equals Dot(p0, p0) - R^2 but takes its sign from the gap, so it is positive. Free when A == 0, when
//     B >= 0 (the path does not approach the centre) or when B^2 - A * C < 0 (it misses the grown sphere).
//   3 t = C / (-B + Sqrt(B^2 - A * C)), the smaller root written without cancellation: -B > 0 and the square root
//     add, so a path that grazes the grown sphere, where B^2 - A * C is a rounding of zero, still gets a time, and
//     C > 0 makes it positive, never behind the start. A hit when t <= duration.
//   Same bits on CPU and GPU: only + - * /, Sqrt and comparisons.
// Units: m and m/s, own axes; s for duration and the result.
// Example: radius 0.5, tol 2^-30, p0 {-2, 0.3, 0}, u {1, 0, 0}, duration 2: A = 1, B = -2, and the hit is at
//   2 - Sqrt((0.5 + tol)^2 - 0.09) = 1.5999999988358469 s.
// Requires: solid_index < inputs.solid_count, and that solid is a sphere. own_path is a sweep measured from it
//   (ComputeOwnPath).
// Read: docs/method-paper/fluidity-method.typ <proc-sphere-first-hit>, <eq:sphere-hit-time>
FLUIDITY_CPU_AND_GPU inline HitTime
FindSphereHit(const SolidContactInputs& inputs, std::uint32_t solid_index, OwnPath own_path, double duration) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  assert(solid.shape == SolidShape::kSphere);
  // Steps (method paper <proc-sphere-first-hit>): 1 read the start  2 find the root  3 keep the hit
  // Step 1: the start rules read the true distance, as for a flat face.
  const double radius = solid.half_size.x;  // m
  const double contact_tolerance = inputs.contact_tolerance;
  const Vec3 start = own_path.start;
  const Vec3 velocity = own_path.velocity;
  const DistanceToSolid at_start = ComputeDistanceToSphere(start, radius, kIdentity);
  switch (ComputeStartCase(inputs, at_start, velocity, duration)) {
    case StartCase::kOutside:
      break;
    case StartCase::kRestingAndFree:
      return HitTime{};
    case StartCase::kHitAtStart:
      return {.has_hit = true, .elapsed = 0};
  }

  // Step 2: the constant term comes from the gap the start case read, so it is positive here, as that gap is.
  // Method paper <eq:sphere-hit-time>:  A = w . w,  B = q . w,  C = g0 (|q| + R)
  const double center_distance = math::Sqrt(math::Dot(start, start));  // m, as ComputeDistanceToSphere computes it.
  const double start_gap = at_start.distance - contact_tolerance;  // m, positive.
  const double grown_radius = radius + contact_tolerance;  // m
  const double speed_squared = math::Dot(velocity, velocity);  // m^2/s^2
  const double start_dot_velocity = math::Dot(start, velocity);  // m^2/s, negative while approaching the centre.
  const double squared_distance_excess = start_gap * (center_distance + grown_radius);  // m^2
  const bool approaches_center = speed_squared != 0 && start_dot_velocity < 0;
  if (!approaches_center) {
    return HitTime{};
  }
  const double discriminant = start_dot_velocity * start_dot_velocity - speed_squared * squared_distance_excess;
  if (discriminant < 0) {
    return HitTime{};
  }

  // Step 3: the textbook (-B - Sqrt(B^2 - A C)) / A would subtract two near-equal numbers on a head-on path.
  // Method paper <eq:sphere-hit-time>:  t_hit = C / (-B + Sqrt(B^2 - A C))
  const double hit_elapsed = squared_distance_excess / (-start_dot_velocity + math::Sqrt(discriminant));  // s
  if (hit_elapsed > duration) {
    return HitTime{};
  }
  return {.has_hit = true, .elapsed = hit_elapsed};
}

// Computes when a path, measured from a cylinder in its own axes, is inside the cylinder's wall grown by the
// tolerance, taken as unbounded along the axis: the times its distance from the axis is at most R = radius + tol.
//
// Method: with p0 = own_path.start, u = own_path.velocity, rho0 = Sqrt(p0.x^2 + p0.y^2) and wall_gap = rho0 - radius
//   - tol, which is ComputeDistanceToCylinder's wall distance minus tol, bit for bit. The squared distance from the
//   axis minus R^2 is A t^2 + 2 B t + C with A = u.x^2 + u.y^2, B = p0.x u.x + p0.y u.y and C = wall_gap * (rho0 + R),
//   which equals rho0^2 - R^2 but takes its sign from the gap. Each root is written so that nothing cancels:
//   wall_gap > 0  outside: empty when A == 0, B >= 0 (no approach to the axis) or B^2 - A C < 0 (a miss); otherwise
//                 entry C / (-B + Sqrt(B^2 - A C)), positive, and exit (-B + Sqrt(B^2 - A C)) / A.
//   otherwise     inside from the start: entry 0; no exit when A == 0; otherwise the larger root, (-B + Sqrt(B^2 -
//                 A C)) / A when B <= 0 and -C / (B + Sqrt(B^2 - A C)) when B > 0, zero or more since C <= 0.
//   Same bits on CPU and GPU: only + - * /, Sqrt and comparisons.
// Units: m and m/s, own axes; m for half_size and contact_tolerance; s for the times.
// Example: radius 0.5, tol 2^-30, p0 {-2, 0, 1.5}, u {1, 0, -0.5}: inside from 1.5 - tol s to 2.5 + tol s.
// Read: docs/method-paper/fluidity-method.typ <eq:cylinder-hit-time>, <proc-cylinder-first-hit>
FLUIDITY_CPU_AND_GPU inline TimeInterval
ComputeTimeInsideGrownWall(OwnPath own_path, Vec3 half_size, double contact_tolerance) {
  const Vec3 start = own_path.start;
  const Vec3 velocity = own_path.velocity;
  const double radius = half_size.x;  // m
  // The expressions of ComputeDistanceToCylinder, so the side of the grown wall agrees with the distance's gap.
  const double axis_distance = math::Sqrt(start.x * start.x + start.y * start.y);  // m, rho0
  const double wall_gap = axis_distance - radius - contact_tolerance;  // m
  // Method paper <eq:cylinder-hit-time>:  A = w_x^2 + w_y^2,  B = q_x w_x + q_y w_y,
  //   C = (rho0 - radius - tol) (rho0 + R)
  const double grown_radius = radius + contact_tolerance;  // m
  const double speed_across_axis_squared = velocity.x * velocity.x + velocity.y * velocity.y;  // m^2/s^2
  const double start_dot_velocity_across_axis = start.x * velocity.x + start.y * velocity.y;  // m^2/s
  const double squared_distance_excess = wall_gap * (axis_distance + grown_radius);  // m^2
  const double discriminant = start_dot_velocity_across_axis * start_dot_velocity_across_axis
      - speed_across_axis_squared * squared_distance_excess;
  const bool is_beyond_grown_wall = wall_gap > 0;
  if (is_beyond_grown_wall) {
    const bool approaches_axis = speed_across_axis_squared != 0 && start_dot_velocity_across_axis < 0;
    if (!approaches_axis || discriminant < 0) {
      return TimeInterval{};
    }
    // Method paper <eq:cylinder-hit-time>:  t_wall = C / (-B + Sqrt(B^2 - A C))
    const double root_sum = -start_dot_velocity_across_axis + math::Sqrt(discriminant);  // m^2/s, positive.
    return {
        .is_empty = false,
        .entry_elapsed = squared_distance_excess / root_sum,
        .has_exit = true,
        .exit_elapsed = root_sum / speed_across_axis_squared,
    };
  }
  // Inside the grown wall already: C <= 0, so the discriminant is at least B^2 and the larger root is zero or more.
  if (speed_across_axis_squared == 0) {
    return {.is_empty = false, .entry_elapsed = 0, .has_exit = false, .exit_elapsed = 0};
  }
  const double root_term = math::Sqrt(discriminant);  // m^2/s
  const double exit_elapsed = start_dot_velocity_across_axis <= 0
      ? (-start_dot_velocity_across_axis + root_term) / speed_across_axis_squared
      : -squared_distance_excess / (start_dot_velocity_across_axis + root_term);
  return {.is_empty = false, .entry_elapsed = 0, .has_exit = true, .exit_elapsed = exit_elapsed};
}

// Computes when a path, measured from a cylinder in its own axes, is between the planes of the cylinder's caps grown
// by the tolerance: the times |z| is at most H = half_height + tol.
//
// Method: with p0 = own_path.start, u = own_path.velocity and cap_gap = |p0.z| - half_height - tol, which is
//   ComputeDistanceToCylinder's cap distance minus tol, bit for bit:
//   cap_gap > 0  beyond the cap on the side s0 = sign(p0.z): empty unless the path moves toward it, -s0 * u.z > 0;
//                otherwise entry cap_gap / (-s0 * u.z), positive, and exit (|p0.z| + half_height + tol) / (-s0 * u.z),
//                where it reaches the far grown cap.
//   otherwise    between the grown caps from the start: entry 0; no exit when u.z == 0; otherwise the time it
//                reaches the grown cap it moves toward, -(s * p0.z - half_height - tol) / |u.z| with s = sign(u.z),
//                zero or more since that gap is at most cap_gap.
//   Same bits on CPU and GPU: only + - * /, Abs and comparisons.
// Units: m and m/s, own axes; m for half_size and contact_tolerance; s for the times.
// Example: half height 1, tol 2^-30, p0 {0.1, 0.2, 3}, u {0, 0, -1}: between the grown caps from 2 - tol s to
//   4 + tol s.
// Read: docs/method-paper/fluidity-method.typ <eq:cylinder-hit-time>, <proc-cylinder-first-hit>
FLUIDITY_CPU_AND_GPU inline TimeInterval
ComputeTimeBetweenGrownCaps(OwnPath own_path, Vec3 half_size, double contact_tolerance) {
  const double start_height = own_path.start.z;  // m
  const double height_rate = own_path.velocity.z;  // m/s
  const double half_height = half_size.z;  // m
  // The expression of ComputeDistanceToCylinder, so the side of the grown cap agrees with the distance's gap.
  const double cap_gap = math::Abs(start_height) - half_height - contact_tolerance;  // m
  const bool is_beyond_grown_cap = cap_gap > 0;
  if (is_beyond_grown_cap) {
    // sigma_0: the start is beyond the high cap or the low one; cap_gap > 0 keeps start_height away from 0.
    const double cap_side_sign = start_height > 0 ? 1.0 : -1.0;
    const double speed_toward_cap = -cap_side_sign * height_rate;  // m/s
    const bool approaches_cap = speed_toward_cap > 0;
    if (!approaches_cap) {
      return TimeInterval{};
    }
    // Method paper <eq:cylinder-hit-time>:  t_cap = (d_cap - tol) / (-sigma_0 w_z)
    return {
        .is_empty = false,
        .entry_elapsed = cap_gap / speed_toward_cap,
        .has_exit = true,
        .exit_elapsed = (math::Abs(start_height) + half_height + contact_tolerance) / speed_toward_cap,
    };
  }
  if (height_rate == 0) {
    return {.is_empty = false, .entry_elapsed = 0, .has_exit = false, .exit_elapsed = 0};
  }
  const double moving_side_sign = height_rate > 0 ? 1.0 : -1.0;
  const double gap_to_cap_ahead = moving_side_sign * start_height - half_height - contact_tolerance;  // m, <= 0
  return {
      .is_empty = false,
      .entry_elapsed = 0,
      .has_exit = true,
      .exit_elapsed = -gap_to_cap_ahead / math::Abs(height_rate),
  };
}

// Finds the first hit of a straight path on a cylinder, measured from the cylinder in its own axes: the start rules,
// then the entry into the cylinder grown with sharp edges.
//
// Method (plan decisions P15 and P16), with p0 = own_path.start, u = own_path.velocity, tol =
//   inputs.contact_tolerance:
//   1 at_start = ComputeDistanceToCylinder(p0, half_size, kIdentity), the bits ComputeDistanceToSolid gives in world
//     axes, and ComputeStartCase of it with u: kHitAtStart is a hit at 0, kRestingAndFree is free. A kOutside start
//     with wall_distance - tol <= 0 and cap_distance - tol <= 0, computed as the distance computes them, is in a rim
//     corner: a hit at 0 when rate = Dot(at_start's normal, u) < 0, free otherwise. Both gaps come from the
//     distance's own expressions, so a rim corner has both distances in (0, tol] exactly.
//   2 The grown cylinder is the part shared by the grown unbounded wall and the space between the grown caps' planes,
//     so the path is inside it while it is inside both: ComputeTimeInsideGrownWall and ComputeTimeBetweenGrownCaps.
//     The entry is the later of the two entries; the path hits when neither interval is empty, the entry comes no
//     later than either exit and no later than duration. Both limits count, so a path that meets the sharp rim
//     edge exactly is a hit. This is the entry and exit test of a box (FindEntryThroughFacePlanes): a path that goes
//     into the grown cylinder through the rim edge has its exits far after its entry, so rounding of the entry
//     times cannot make it miss; only a path that grazes the edge sits on the limit, where a hit and a miss are both
//     safe.
//   Same bits on CPU and GPU: only + - * /, Abs, Sqrt and comparisons.
// Units: m and m/s, own axes; s for duration and the result.
// Ties: equal entries through the wall and a cap give the same time; the face comes from the hit position.
// Example: radius 0.5, half height 1, tol 2^-30, p0 {0.5 + tol / 2, 0, 3}, u {0, 0, -1}: inside the grown wall from
//   the start on (no motion across the axis), between the grown caps from 2 - tol s: a hit at 2 - tol s, at
//   {0.5 + tol / 2, 0, 1 + tol}, which faces the high rim at Sqrt(1.25) tol. A start {0.5 + 0.9 tol, 0, 1 + 0.9 tol} is
//   in the rim corner: u {-1, 0, 0} is a hit at 0.
// Requires: solid_index < inputs.solid_count, and that solid is a cylinder whose axis keeps its direction. own_path is
//   a sweep measured from it (ComputeOwnPath).
// Read: docs/method-paper/fluidity-method.typ <proc-cylinder-first-hit>, <eq:cylinder-hit-time>
FLUIDITY_CPU_AND_GPU inline HitTime
FindCylinderHit(const SolidContactInputs& inputs, std::uint32_t solid_index, OwnPath own_path, double duration) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  assert(solid.shape == SolidShape::kCylinder);
  // Steps (method paper <proc-cylinder-first-hit>): 1 read the start  2 the times inside the wall and between the
  //   caps  3 the entry into both
  // Step 1: the start rules read the true distance, as for a flat face.
  const Vec3 half_size = solid.half_size;
  const double contact_tolerance = inputs.contact_tolerance;
  const Vec3 start = own_path.start;
  const DistanceToSolid at_start = ComputeDistanceToCylinder(start, half_size, kIdentity);
  switch (ComputeStartCase(inputs, at_start, own_path.velocity, duration)) {
    case StartCase::kOutside:
      break;
    case StartCase::kRestingAndFree:
      return HitTime{};
    case StartCase::kHitAtStart:
      return {.has_hit = true, .elapsed = 0};
  }
  // A positive gap inside the grown sharp cylinder: the intervals below would both start at 0, but the true distance,
  // convex along the path, stays above tol while rate >= 0. These are the expressions of ComputeDistanceToCylinder,
  // so a positive gap with neither gap positive means both distances are positive: the rim region.
  const double axis_distance = math::Sqrt(start.x * start.x + start.y * start.y);  // m, rho0
  const double wall_gap = axis_distance - half_size.x - contact_tolerance;  // m
  const double cap_gap = math::Abs(start.z) - half_size.z - contact_tolerance;  // m
  const bool is_in_rim_corner = wall_gap <= 0 && cap_gap <= 0;
  if (is_in_rim_corner) {
    const double gap_rate = math::Dot(at_start.normal_out_of_solid, own_path.velocity);  // m/s
    const bool approaches_rim = gap_rate < 0;
    return approaches_rim ? HitTime{.has_hit = true, .elapsed = 0} : HitTime{};
  }

  // Step 2: the start is beyond the grown wall, a grown cap or both, so the entry below is positive.
  const TimeInterval inside_wall = ComputeTimeInsideGrownWall(own_path, half_size, contact_tolerance);
  const TimeInterval between_caps = ComputeTimeBetweenGrownCaps(own_path, half_size, contact_tolerance);
  if (inside_wall.is_empty || between_caps.is_empty) {
    return HitTime{};
  }

  // Step 3: entry == exit is a path that meets the grown cylinder at one point, on its rim edge: a hit.
  // The later entry; a NaN, which only a broken precondition could give, compares false and is never a hit.
  const double entry_elapsed = inside_wall.entry_elapsed < between_caps.entry_elapsed ? between_caps.entry_elapsed
                                                                                      : inside_wall.entry_elapsed;  // s
  const bool leaves_wall_first = inside_wall.has_exit && inside_wall.exit_elapsed < entry_elapsed;
  const bool leaves_caps_first = between_caps.has_exit && between_caps.exit_elapsed < entry_elapsed;
  const bool enters_within_sweep = entry_elapsed <= duration;
  if (leaves_wall_first || leaves_caps_first || !enters_within_sweep) {
    return HitTime{};
  }
  return {.has_hit = true, .elapsed = entry_elapsed};
}

// Finds the first hit of a straight path on one solid: the steps "solid that does not turn" and "turning solid" of
// FindFirstHit for that solid.
//
// Method: with shape_turning = ComputeShapeTurningAngularVelocity(solid), the part of the angular velocity that
//   changes the shape. A flat solid, a plate or a box turns when shape_turning is not exactly {0, 0, 0} (plan
//   decision P7). A sphere never turns: shape_turning is {0, 0, 0}, since a spin about its centre leaves its shape.
//   A cylinder turns when Length(shape_turning) * step_duration * Length(half_size) > tol / 8 (plan decision P17):
//   its axis then tilts enough to move its surface by more than tol / 8 during the step. Below that, which includes
//   a spin about a tilted axis that rounding leaves with a tiny part across it, the closed form at the pose of
//   start_elapsed holds. One switch over the shape:
//   flat solid  FindHitOnConvexSolid with its one face plane, kHighZ at own z = 0.
//   plate       FindHitOnConvexSolid with kLowZ and kHighZ at own z = -half_thickness and +half_thickness, or only
//               the committed face (ToFacePlane) when inputs.plate_side names this plate.
//   box         FindHitOnConvexSolid with its six face planes, in face order.
//   domain box  FindHitOnDomainBox.
//   sphere      FindSphereHit of the sweep in the sphere's own axes (ComputeOwnPath), then ToFirstHit.
//   cylinder    FindCylinderHit likewise.
//   A solid that turns is kTooCloseToCall: this code has no search toward a turning solid yet.
//   *hit_on_solid has solid_index kNoSolid when the path is free. Same bits on CPU and GPU when the angular velocity
//   is zero: only + - * /, Abs, Sqrt and comparisons, since ComputeSolidPose of a solid that does not move its axes
//   needs no Sin. A spinning sphere or cylinder poses its axes with Sin, which may differ in the last bits.
// Units: s for start_elapsed and duration; m and m/s, world, for the path.
// Example: a box of half size 0.5 m at the origin and the path from {-1, 0.1, 0.2} at {2, 0, 0} m/s for 1 s: elapsed
//   0.25 - tol / 2 on kLowX. A sphere of radius 0.5 m at the origin and the path from {-2, 0, 0} at {1, 0, 0} m/s for
//   2 s: elapsed 1.5 - tol on kCurved.
// Requires: solid_index < inputs.solid_count; hit_on_solid is not null. The solid's shape is one of SolidShape's
//   values.
// Fails: kBadInput        the solid is of an unknown shape; a debug build stops at an assertion first.
//        kTooCloseToCall  the solid turns; the only close call is kSlowApproachToTurningSolid.
//        On failure *hit_on_solid is unchanged.
// Read: docs/method-paper/fluidity-method.typ <proc-find-first-hit>, <sec-first-hit>, <sec-round-solids>
//
// clang-tidy: start_elapsed and duration follow the path, as contact design 3.4 orders FindFirstHit's parameters; the
// swappable-parameters check flags the two doubles.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU inline ContactError FindFirstHitOnSolid(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    Vec3 start,
    Vec3 velocity_for_this_step,
    double start_elapsed,
    double duration,
    FirstHit* hit_on_solid
) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  assert(solid_index < inputs.solid_count);
  assert(hit_on_solid != nullptr);
  // Steps (method paper <proc-find-first-hit>): 1 a solid that does not turn  2 a turning solid
  // Each shape's branch applies its own touching rule, so a new shape brings its own rule with its own case.
  const Solid& solid = inputs.solids[solid_index];
  const Sweep sweep = {
      .start = start,
      .velocity_for_this_step = velocity_for_this_step,
      .start_elapsed = start_elapsed,
      .duration = duration,
  };
  const Vec3 shape_turning = ComputeShapeTurningAngularVelocity(solid);
  const bool shape_turns = shape_turning.x != 0 || shape_turning.y != 0 || shape_turning.z != 0;
  const Vec3 half_size = solid.half_size;
  switch (solid.shape) {
    case SolidShape::kFlatSolid: {
      // Step 2: the closed forms below hold only while the faces keep their direction.
      if (shape_turns) {
        return ContactError::kTooCloseToCall;
      }
      // Step 1: one face plane.
      const FacePlane surface = {.face = SolidFace::kHighZ, .axis = Axis::kZ, .sign_out_of_solid = 1, .offset = 0};
      *hit_on_solid = FindHitOnConvexSolid(inputs, solid_index, sweep, &surface, 1);
      return ContactError::kNone;
    }
    case SolidShape::kPlate: {
      if (shape_turns) {
        return ContactError::kTooCloseToCall;
      }
      // Step 1: a committed particle sees only the committed face (plan decision P3). For every start this gives the
      // two-face result: a start outside the committed half space is beyond the committed face, which both forms
      // enter first, and every other start is settled by the start rule on the P3 distance. It states P3 here.
      const PlateSide plate_side = inputs.plate_side;
      if (plate_side.plate_id == solid.solid_id) {
        const FacePlane committed_face = ToFacePlane(plate_side.side, half_size.z);
        *hit_on_solid = FindHitOnConvexSolid(inputs, solid_index, sweep, &committed_face, 1);
        return ContactError::kNone;
      }
      // A C array: GPU code cannot call std::array's member functions (nvcc without --expt-relaxed-constexpr).
      // NOLINTNEXTLINE(modernize-avoid-c-arrays): see the comment above.
      const FacePlane plate_faces[] = {
          {.face = SolidFace::kLowZ, .axis = Axis::kZ, .sign_out_of_solid = -1, .offset = half_size.z},
          {.face = SolidFace::kHighZ, .axis = Axis::kZ, .sign_out_of_solid = 1, .offset = half_size.z},
      };
      *hit_on_solid = FindHitOnConvexSolid(inputs, solid_index, sweep, plate_faces, 2);
      return ContactError::kNone;
    }
    case SolidShape::kBox: {
      if (shape_turns) {
        return ContactError::kTooCloseToCall;
      }
      // Step 1: six face planes, in face order, so equal entry times keep the first face.
      // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as above.
      const FacePlane box_faces[] = {
          {.face = SolidFace::kLowX, .axis = Axis::kX, .sign_out_of_solid = -1, .offset = half_size.x},
          {.face = SolidFace::kHighX, .axis = Axis::kX, .sign_out_of_solid = 1, .offset = half_size.x},
          {.face = SolidFace::kLowY, .axis = Axis::kY, .sign_out_of_solid = -1, .offset = half_size.y},
          {.face = SolidFace::kHighY, .axis = Axis::kY, .sign_out_of_solid = 1, .offset = half_size.y},
          {.face = SolidFace::kLowZ, .axis = Axis::kZ, .sign_out_of_solid = -1, .offset = half_size.z},
          {.face = SolidFace::kHighZ, .axis = Axis::kZ, .sign_out_of_solid = 1, .offset = half_size.z},
      };
      *hit_on_solid = FindHitOnConvexSolid(inputs, solid_index, sweep, box_faces, 6);
      return ContactError::kNone;
    }
    case SolidShape::kDomainBox:
      // Step 1: CheckSolid refuses a domain box that moves, so it never turns.
      assert(!shape_turns);
      *hit_on_solid = FindHitOnDomainBox(inputs, solid_index, sweep);
      return ContactError::kNone;
    case SolidShape::kSphere: {
      // Step 1: a spin about the centre moves the surface but not the shape (plan decision P17), so a sphere never
      // turns and always has its closed form, the touching rule included.
      assert(!shape_turns);
      const HitTime hit_time = FindSphereHit(inputs, solid_index, ComputeOwnPath(inputs, solid_index, sweep), duration);
      *hit_on_solid = ToFirstHit(inputs, solid_index, sweep, hit_time);
      return ContactError::kNone;
    }
    case SolidShape::kCylinder: {
      // Step 2: a turn across the axis tilts it. Rounding leaves a tiny part across the axis even in a spin about a
      // tilted axis, so the test is a bound on how far the tilt moves the surface, not an exact zero.
      // Method paper <sec-round-solids>:  |omega_across| T |half_size| <= tol / 8
      const double surface_movement_from_tilt =
          math::Length(shape_turning) * inputs.step_duration * math::Length(half_size);  // m
      if (surface_movement_from_tilt > inputs.contact_tolerance / 8) {
        return ContactError::kTooCloseToCall;
      }
      // Step 1: the wall, the caps and the rims, with the touching rule and the rim corner.
      const HitTime hit_time =
          FindCylinderHit(inputs, solid_index, ComputeOwnPath(inputs, solid_index, sweep), duration);
      *hit_on_solid = ToFirstHit(inputs, solid_index, sweep, hit_time);
      return ContactError::kNone;
    }
  }
  // An unknown shape has no closed form, so reaching this line is a caller bug: CheckSolid refuses it.
  assert(false && "FindFirstHitOnSolid has no closed form for this shape");
  return ContactError::kBadInput;
}

}  // namespace fluidity::contact
