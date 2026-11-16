// The private steps of one particle's move through the solids: the distance the particle sees to one solid, the
// first hit of a straight path on one solid, and what MoveParticle does around each sweep.
//
// Purpose: give FindFirstHit and MoveParticle the functions that evaluate one solid for one particle, so one switch
//   chooses each shape's formula and each shape keeps its own rules; and give MoveParticle its steps: the input
//   check, the start rules, one hit with its joint response over every surface the particle is on, the move off
//   several near surfaces, the settle and the end check.
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
//   A turning solid's gap is not linear in time and its faces do not keep their direction, so no closed form holds.
//   The gap cannot shrink faster than the particle's speed relative to the solid, which a bound caps, and its rate
//   cannot change faster than a second bound, K; the stepping search advances along the path by the longer of two
//   times over which the gap provably stays positive: the gap over the first bound, and, when the shape's distance
//   is convex, the root of the gap's lower model g + gdot s - K s^2 / 2. So it never passes the moment the path
//   reaches the grown solid, and it stops once the gap is a quarter of a tolerance. A flat solid or a plate spinning
//   about its normal is not a turning solid: both are unbounded sideways, so the spin leaves the shape unchanged.
//   Example: a box of half size 0.5 m at the origin turning at {0, 0, 1} rad/s and a particle at rest at
//   0.6 * {cos 0.3, sin 0.3, 0} m: the bound is 0.6 m/s, K is 0.6 m/s^2, and after 3 advances face kHighX is hit at
//   0.8857 s.
//   Before the first sweep, the start rules classify the start against every solid: too deep inside, or on the wrong
//   side of a committed plate, fails; closer than tol to some surfaces, the particle moves once to the rest gap of all
//   of them; inside, or approaching, a surface is also a hit at time 0. One move, not one per solid in turn: in a
//   wedge sharper than a right angle each solid's move pushes the start toward the other.
//   Example: a particle 0.0005 m inside a fixed floor, with a start overlap tolerance of 0.001 m, starts the step at
//   z = tol, with the response and kind kStartedInsideSolid. At rest at the apex of the floor and a face at 60 degrees
//   to it, it starts at {Sqrt(3) tol, 0, tol}, tol from both.
//   The move off several surfaces is the smallest change, from the start, that brings every near surface to the rest
//   gap, solved again from each iterate with the distances and normals read there, because a round surface's normal
//   turns as the point moves (MoveToRestGapOfNearSurfaces, SolveSmallestCorrection). Every shape's distance is convex,
//   so each iterate is at the rest gap of every surface it read and no farther from the start than the one before.
//   Example: in the cusp under a ball of radius 0.5 m resting on a floor, a start tol / 4 from both moves in 5
//   iterates to the rest point 2 Sqrt(0.5 tol) = 46341 tol from the contact.
//   At each hit, at the start or along the path, both velocities get one joint response over the active surfaces,
//   every surface the particle is within tol + tol / 16 of, read afresh, and the surface hit: one surface gives the
//   single-surface response; several give the smallest frictionless change that approaches none of them, then the
//   friction of each used surface from its push. In a groove the particle keeps the part along the common line, in a
//   cusp or a corner it stops, and between moving solids it takes the velocity that approaches none, unless they close
//   on it (a squeeze). After the first response of a step every solid has the touching rule of a touched solid, with
//   the largest allowance of the responses so far.
//   Example: at the apex of a 60 degree wedge, a particle moving at {-1, 1, -1} m/s, friction 0, gets {0, 1, 0}, where
//   the floor's response {-1, 1, 0} followed by the other face's {-0.25, 1, -0.433} would approach the floor again.
//   At the end of the step the settle moves the particle once more off every surface of every solid it is closer to
//   than tol, and the end check then asks every solid to be at least tol / 2 away.
//   Example: a particle sliding along a fixed floor that a slightly tilted flat solid pushes down by 0.75 tol ends at
//   z = tol: the settle lifts it.
//
// Words:
//   settle         at the end of the step, the one move of the particle off every surface of every solid it is closer
//                  to than tol, back to the rest gap of all of them
//   wedge angle    of two surfaces, 180 degrees minus the angle between their normals: an acute wedge has normals more
//                  than a right angle apart, Dot(n_A, n_B) < 0
//   shortfall      of a surface at a point, tol - d: how far the point is from that surface's rest gap
//   tangent contact  two convex solids touching at a point or along a line (a ball resting on a floor, two touching
//                  balls), around which the free space is a cusp, whose angle shrinks to 0
//   move cap       kMaxRestGapMoveInTolerances * tol, the longest move off several near surfaces at once
//   slide          the rest of the step spent moving along the surfaces the particle is on, with the velocity the
//                  joint response left
//   joint response  the response of one hit over every active surface at once (ApplyJointResponse): the
//                  single-surface response for one, else the smallest frictionless change, the friction of each used
//                  surface and, when needed, a second frictionless change
//   active surface  a surface of any solid that the particle is within tol + tol / 16 of at a hit, read afresh, or the
//                  surface the sweep hit; exact copies count once
//   direction      active surfaces whose normals agree within kSameNormalAxisDifference: one target of the solver
//   push           of a surface in a joint response, m/s: the multiplier of its direction in the frictionless change,
//                  how far the change goes along its normal
//   friction limit  m/s: friction times friction_scale times the push, the most sideways speed a surface's friction
//                  can take in one response
//   pushing surface  the surface that acts for a direction: among those that push, the roughest, then a reacting
//                  one, then the first in the sorted order
//   sliver         a pair of active surfaces whose normals are within kVelocityPairLimitShare of opposite (a wedge
//                  under 1.35e-6 rad): the frictionless change cannot be computed there, and the particle sticks to
//                  the surface it approaches fastest
//   squeeze        solids closing on the particle so that no velocity leaves them all, or an escape faster than
//                  kMaxSqueezeOutSpeedFactor times the speeds involved: kSqueezedBetweenSolids, naming the solid
//                  approached fastest
//   end check      the last test of a particle's move: every solid at least tol / 2 away at its pose at the end of
//                  the step, else the call is too close to call
//   touching rule  how a sweep treats a start whose gap is in [-tol, 0] for the surface that gives the distance: the
//                  particle rests on that surface, which is hit only when the path pushes into it by more than tol
//                  during the sweep. The solid lies behind that surface's plane, so a path that stays within tol of
//                  the plane cannot enter it. On a sphere or a cylinder the solid is convex and its true distance is
//                  a convex function along the path, which gives the same rule
//   rim corner     the part of a cylinder grown with sharp edges that is farther than tol from the rim: a point
//                  beyond the wall and beyond a cap, by at most tol from each
//   turning solid  a solid whose shape turns during the step: a box with an angular velocity other than {0, 0, 0},
//                  a flat solid or a plate with an angular velocity whose part across the normal is more than
//                  rounding (kNegligibleTurnShare of it), or a cylinder whose axis tilts beyond plan decision P17's
//                  bound. It has no touching rule
//   second-order advance  the advance of the stepping search toward a solid whose distance is convex: the positive
//                  root of g + gdot s - K s^2 / 2, gdot the gap rate and K the gap curvature bound
// Owns:    Sweep, FacePlane, StartCase, OwnPath, HitTime, ComputeCoordinate, ComputeColumn, ToFacePlane,
//          ComputeIsClosedSide, ComputeFaceDistance, ComputeDistanceForParticle, ComputeStartCase, ComputeOwnPath,
//          ToFirstHit, FindEntryThroughFacePlanes, FindHitOnConvexSolid, FindHitOnDomainBox, FindSphereHit,
//          TimeInterval, ComputeTimeInsideGrownWall, ComputeTimeBetweenGrownCaps, FindCylinderHit, HasConvexDistance,
//          GapLowerModel, ComputeSecondOrderAdvance, FindHitOnTurningSolid, ComputeIsTurningSolid, FindFirstHitOnSolid,
//          ComputeEarlierHit, FindEarliestHitWithTurningSolids, TouchedSolid, FindFirstHitOfSweep, ParticleMove,
//          CheckSolidContactInputs, ToSideAtHit, ApplyReactionChange, CheckHitCountBelowLimit, MoveToRestGap,
//          CheckStartOnCommittedSide, ComputeIsApproachingSolid, NearSurfaces, FindSurfacesOfSolid, SubsetEquations,
//          SubsetMultipliers, SolveSubsetMultipliers, ComputeSubsetChange, CorrectionSubset, ComputeCorrectionSlack,
//          SolveCorrectionOfSubset, ComputeIsWithinTargets, kMaxCorrectionSubsets, ComputeNextSubset,
//          ComputeIsWithinSlack, FindShortestCorrection, FindCorrectionOfLargestFrictionSum, SolveSmallestCorrection,
//          ComputeIsAtLeastFromEverySolid, BuildFixedCopyOfSurface, FindCrossingOfMove, kMaxTrackedSurfaces,
//          ConstrainingSurface, ConstrainingSurfaces, ComputeIsConstraining, ApplyConstrainingSurface,
//          ComputeIsSameNormal, ApplyRestGapTarget, ComputeSweptGrowth, FindEarliestCrossing, ApplyCrossedSurfaces,
//          RestGapTargets, BuildRestGapTargets, RestGapMove, NearSolids, FindNearSolids, ComputeMoveOffOneNearSolid,
//          MoveToRestGapOfNearSurfaces, ActiveSurface, ActiveSurfaces, ComputeIsExactCopy, ComputeIsSortedBefore,
//          ApplyActiveSurface, ApplyDirections, FindActiveSurfaces, ApplyCloseCall, JointTargets,
//          ComputeIsPushingBefore, BuildJointTargets, ComputeIsWithinSlack, ComputeFastestApproachedSolid, JointChange,
//          ApplyCharge, JointFriction, ComputeFrictionStep, kFrictionOrders, ComputeMeanOverOrders,
//          ComputeJointFriction, ApplyStick, ApplySecondSolve, ComputeJointChange, ApplyJointCharges,
//          ApplyJointResponse, ApplyHit, CheckStartOfSolid, ComputeIsStartInsideSolid, ComputeIsInArrayOrderBefore,
//          StartResponse, FindStartResponse, ApplyStartRules, ApplySettleAtEnd, CheckEndOutsideSolids
// Uses:    Vec3, math::Dot, math::Cross, math::Length, math::IsFinite (base/vec3.h), Mat3, kIdentity,
//          math::Transpose, math::Determinant (base/mat3.h), Axis (base/axis.h), math::Abs, math::Sqrt, math::Max
//          (base/scalar_math.h), PerParticleVelocity (base/per_particle_velocity.h), FLUIDITY_CPU_AND_GPU
//          (base/cpu_and_gpu.h), ContactError, ContactCloseCall, ContactFailure (contact/contact_error.h), Solid,
//          SolidShape, SolidFace, SolidSide, PlateSide, SolidPose, ClosedSides, DomainSide, DistanceToSolid, kNoSolid,
//          ComputeSolidPose, ComputeSolidVelocity, ComputeDistanceToSolid, ComputeDistanceToSphere,
//          ComputeDistanceToCylinder, ComputeShapeTurningAngularVelocity, kNegligibleTurnShare
//          (contact/solid_shapes.h), SolidContactInputs, SolidContactResult, SolidReaction, ContactKind, FirstHit,
//          kMaxHitsPerStep, kMaxAdvanceCount, kTurningSolidHitGapShare, kTouchedSolidRestingGapShare,
//          kMaxRestGapMoveInTolerances, kMaxRestGapIterations, kMaxConstrainingSurfaces, kSameNormalAxisDifference,
//          kMaxCorrectionRefinements, kVelocityPairLimitShare, kVelocityTripleLimit, kMaxSqueezeOutSpeedFactor,
//          kMaxUsedDirections, kMaxFrictionOrders, CorrectedQuantity, CorrectionTargets, SmallestCorrection,
//          kApproachLeftAfterResponseShare, ComputeContactResponse, ComputeLargestApproachLeftAfterResponse
//          (contact/solid_contact_types.h)
// Used by: contact/solid_contact.h, whose FindFirstHit calls FindFirstHitOnSolid for each solid and whose MoveParticle
//          calls the steps of a particle's move
// Read:    docs/method-paper/fluidity-method.typ <sec-first-hit>, <eq:hit-time>, <proc-find-first-hit>,
//          <sec-round-solids>, <eq:sphere-hit-time>, <eq:cylinder-hit-time>, <proc-sphere-first-hit>,
//          <proc-cylinder-first-hit>, <proc-move-particle>, <fig-move-off-near-surfaces>, <sec-contact-response>,
//          <sec-contact-tolerance>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
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

// What a sweep knows about the solids from the responses of the particle's move so far: whether one was made in this
// step, which gives every solid the touching rule of a touched solid, and the approach allowance that rule uses.
struct TouchedSolid {
  bool is_touched = false;
  double approach_allowance = 0;  // m/s, zero or more.
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

// Converts one side of the domain box into its face plane, whose distance is measured into the box: own p.axis +
// half_size from a low side, half_size - p.axis from a high side, as ComputeDistanceToSolid gives for that side.
//
// Units: m for half_size.
// Example: kBottom with half_size {0.5, 0.5, 0.5}: face kLowZ, axis kZ, sign_out_of_solid +1, offset -0.5, so the
//   centre of the box is 0.5 from it and its normal is {0, 0, 1}, into the box.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>, <eq:solid-distance>
FLUIDITY_CPU_AND_GPU constexpr FacePlane ToFacePlane(DomainSide side, Vec3 half_size) {
  switch (side) {
    case DomainSide::kLeft:
      return {.face = SolidFace::kLowX, .axis = Axis::kX, .sign_out_of_solid = 1, .offset = -half_size.x};
    case DomainSide::kRight:
      return {.face = SolidFace::kHighX, .axis = Axis::kX, .sign_out_of_solid = -1, .offset = -half_size.x};
    case DomainSide::kFront:
      return {.face = SolidFace::kLowY, .axis = Axis::kY, .sign_out_of_solid = 1, .offset = -half_size.y};
    case DomainSide::kBack:
      return {.face = SolidFace::kHighY, .axis = Axis::kY, .sign_out_of_solid = -1, .offset = -half_size.y};
    case DomainSide::kBottom:
      return {.face = SolidFace::kLowZ, .axis = Axis::kZ, .sign_out_of_solid = 1, .offset = -half_size.z};
    case DomainSide::kTop:
      break;
  }
  return {.face = SolidFace::kHighZ, .axis = Axis::kZ, .sign_out_of_solid = -1, .offset = -half_size.z};
}

// Computes whether one side of the domain box is closed, so it stops particles.
//
// Example: the default ClosedSides: kBottom is closed, kTop is open.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsClosedSide(ClosedSides closed, DomainSide side) {
  switch (side) {
    case DomainSide::kLeft:
      return closed.left;
    case DomainSide::kRight:
      return closed.right;
    case DomainSide::kFront:
      return closed.front;
    case DomainSide::kBack:
      return closed.back;
    case DomainSide::kBottom:
      return closed.bottom;
    case DomainSide::kTop:
      break;
  }
  return closed.top;
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
//   gap > limit       kOutside. The limit is 0, or kTouchedSolidRestingGapShare * tol = tol / 16 once the particle
//                     had a response in this step (touched.is_touched), for every solid.
//   gap < -tol        kHitAtStart: deeper inside than one tolerance.
//   otherwise         the touching rule. Before any response: kHitAtStart when rate * duration < -tol,
//                     kRestingAndFree when it is -tol or more; the limit is free, such a path ends on the surface, not
//                     inside it. After one: kHitAtStart when rate < -touched.approach_allowance, the path really
//                     approaching; kRestingAndFree otherwise, since an approach within the allowance is what the
//                     responses so far left by rounding (FindFirstHitOfSweep gives the proof).
//   Why a touched solid differs: after a hit the particle stands at the rest gap only up to rounding, possibly a few
//   rounding units outside, and its velocity keeps a rounding-sized approach (method paper <sec-contact-response>).
//   With the limit 0 such a start would be kOutside, and the closed form would hit the same solid again from the two
//   rounding-sized numbers. What the allowance costs: a free path may approach at up to the allowance, so its gap falls
//   by at most approach_allowance * duration, and over the whole step by at most tol / 16 (FindFirstHitOfSweep).
//   The rule is the same for a flat face and a round surface. Same bits on CPU and GPU: only + - * and comparisons.
// Units: m for the distance, m/s for the velocity and the allowance, s for the duration.
// Example: tol 2^-30, at_start {tol, {0, 0, 1}, kHighZ} (the rest gap): the relative velocity {5, 0, -tol / 4} for
//   1 s gives kRestingAndFree; {5, 0, -1} gives kHitAtStart. With at_start.distance 2 tol: kOutside. Touched with an
//   allowance of 2^-44 m/s: at_start.distance tol + 2^-60 m and the velocity {5, 0, -2^-55} give kRestingAndFree, where
//   a solid not touched gives kOutside; {5, 0, -tol / 4} gives kHitAtStart.
// Requires: at_start is the distance of the surface that gives the distance to the solid at the sweep's start: a face
//   of the solid, or one closed side of a domain box. relative_velocity is in the axes of at_start's normal.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>, <sec-round-solids>
FLUIDITY_CPU_AND_GPU inline StartCase ComputeStartCase(
    const SolidContactInputs& inputs,
    DistanceToSolid at_start,
    Vec3 relative_velocity,
    double duration,
    TouchedSolid touched
) {
  const double contact_tolerance = inputs.contact_tolerance;
  const double start_gap = at_start.distance - contact_tolerance;  // m
  const double resting_gap_limit = touched.is_touched ? kTouchedSolidRestingGapShare * contact_tolerance : 0;  // m
  if (start_gap > resting_gap_limit) {
    return StartCase::kOutside;
  }
  // Deeper than one tolerance inside. MoveParticle never sweeps from here: the start rules move the start to the rest
  // gap of every near surface at once, and a touched solid is approached by at most tol / 16 in the step. A sweep from
  // here, which only a broken precondition could give, is hit at 0, so the particle never ends inside.
  if (start_gap < -contact_tolerance) {
    return StartCase::kHitAtStart;
  }
  // The touching rule: the solid lies behind this surface's plane (a round solid behind its tangent plane, since it
  // is convex), and along the path the gap is at least start_gap + rate * t. Not touched before: rate * duration >=
  // -tol keeps it at least start_gap - tol, so the path never goes below distance 0 by more than the window absorbs.
  // Touched: rate >= -allowance keeps it at least start_gap - allowance * duration >= start_gap - tol / 16.
  // Method paper <eq:hit-time>:  rate = n . (v - v_lin)
  const double gap_rate = math::Dot(at_start.normal_out_of_solid, relative_velocity);  // m/s
  const bool pushes_into_surface =
      touched.is_touched ? gap_rate < -touched.approach_allowance : gap_rate * duration < -contact_tolerance;
  return pushes_into_surface ? StartCase::kHitAtStart : StartCase::kRestingAndFree;
}

// Computes how a sweep starts against one surface of one solid that does not turn, from the distance at the start, in
// world axes.
//
// Method: the form that takes a relative velocity, with velocity_for_this_step minus the velocity of the solid's
//   surface at the sweep's start, ComputeSolidVelocity at the pose of start_elapsed, and the sweep's duration. A flat
//   solid or a plate spinning about its normal does not turn, and its spin's omega x r has no part along the normal, so
//   the rate along the normal is the one measured from linear_velocity alone up to rounding: plan decision P2's closed
//   form is unchanged, and without spin ComputeSolidVelocity is linear_velocity bit for bit. The full velocity is used
//   because the start rules' approach test and the joint response measure from it: a spin of 100 m/s at the particle
//   rounds omega x r by about 1e-14 m/s along the normal, and a rate measured from linear_velocity alone could then see
//   an approach beyond the touched rule's allowance that the response had left within it, and hit the spinning solid
//   at 0 by rounding.
// Units: m for the distance, m/s for the velocities, s for the duration.
// Example: a fixed flat solid with normal {0, 0, 1}, tol 2^-30 and at_start.distance tol (the rest gap): the velocity
//   {5, 0, -tol / 4} for 1 s gives kRestingAndFree; {5, 0, -1} gives kHitAtStart. With at_start.distance 2 tol:
//   kOutside.
// Requires: solid_index < inputs.solid_count, and the solid does not turn. at_start is the distance of the surface
//   that gives the distance to the solid at the sweep's start: a face of the solid, or one closed side of a domain box.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>
FLUIDITY_CPU_AND_GPU inline StartCase ComputeStartCase(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    Sweep sweep,
    DistanceToSolid at_start,
    TouchedSolid touched
) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  const Vec3 solid_velocity = ComputeSolidVelocity(solid, ComputeSolidPose(solid, sweep.start_elapsed), sweep.start);
  const Vec3 relative_velocity = sweep.velocity_for_this_step - solid_velocity;
  return ComputeStartCase(inputs, at_start, relative_velocity, sweep.duration, touched);
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
//   plate side. ComputeStartCase of at_start, with touched: kHitAtStart gives a hit
//   at 0 with at_start; kRestingAndFree gives no hit, for the whole solid, which lies behind the surface the particle
//   rests on; kOutside gives FindEntryThroughFacePlanes. The planes' largest distance is the distance at_start holds,
//   bit for bit, so a start outside by one test is outside by the other.
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
    int face_plane_count,
    TouchedSolid touched
) {
  assert(solid_index < inputs.solid_count);
  const SolidPose pose = ComputeSolidPose(inputs.solids[solid_index], sweep.start_elapsed);
  const DistanceToSolid at_start =
      ComputeDistanceForParticle(inputs, solid_index, pose, sweep.start, inputs.plate_side);
  switch (ComputeStartCase(inputs, solid_index, sweep, at_start, touched)) {
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
// Method: for each closed side in face order, at_start = ComputeFaceDistance of that side's plane (ToFacePlane) at the
//   start, at the pose of sweep.start_elapsed (the distance into the box, as ComputeDistanceToSolid gives for that
//   side), then ComputeStartCase with touched: kHitAtStart is a hit at 0 on
//   that side, kRestingAndFree no hit on it, kOutside FindEntryThroughFacePlanes of that one plane. A particle in a
//   corner rests on two or three sides at once, so every side needs its own touching rule, not only the nearest one.
// Units: m, m/s and s, world.
// Ties: equal times keep the side that comes first in face order.
// Example: the domain box [0, 1]^3 with an open top and a start {tol, tol, 0.5} moving at {-tol / 4, -tol / 4, 0.25}
//   for 1 s: no hit (it rests on side_left and side_front). At {0, -1, 0.25}: a hit at 0 on kLowY, normal {0, 1, 0}.
// Requires: solid_index < inputs.solid_count, and that solid is a domain box with a closed side.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>, <proc-find-first-hit>
FLUIDITY_CPU_AND_GPU inline FirstHit
FindHitOnDomainBox(const SolidContactInputs& inputs, std::uint32_t solid_index, Sweep sweep, TouchedSolid touched) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  assert(solid.shape == SolidShape::kDomainBox);
  const SolidPose pose = ComputeSolidPose(solid, sweep.start_elapsed);
  FirstHit earliest;
  // A C array: GPU code cannot call std::array's member functions (nvcc without --expt-relaxed-constexpr).
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): see the comment above.
  const DomainSide sides_in_face_order[] = {
      DomainSide::kLeft,
      DomainSide::kRight,
      DomainSide::kFront,
      DomainSide::kBack,
      DomainSide::kBottom,
      DomainSide::kTop,
  };
  for (const DomainSide side : sides_in_face_order) {
    if (!ComputeIsClosedSide(solid.closed, side)) {
      continue;
    }
    const FacePlane side_plane = ToFacePlane(side, solid.half_size);
    const DistanceToSolid at_start = ComputeFaceDistance(pose, sweep.start, side_plane);
    FirstHit side_hit;
    switch (ComputeStartCase(inputs, solid_index, sweep, at_start, touched)) {
      case StartCase::kOutside:
        side_hit = FindEntryThroughFacePlanes(inputs, solid_index, sweep, &side_plane, 1);
        break;
      case StartCase::kRestingAndFree:
        break;
      case StartCase::kHitAtStart:
        side_hit = {.elapsed = 0, .solid_index = solid_index, .at_hit = at_start};
        break;
    }
    // A later side wins only when strictly earlier, so a tie keeps the side that comes first.
    const bool is_hit = side_hit.solid_index != kNoSolid;
    const bool is_earlier = earliest.solid_index == kNoSolid || side_hit.elapsed < earliest.elapsed;
    if (is_hit && is_earlier) {
      earliest = side_hit;
    }
  }
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
FLUIDITY_CPU_AND_GPU inline HitTime FindSphereHit(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    OwnPath own_path,
    double duration,
    TouchedSolid touched
) {
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
  switch (ComputeStartCase(inputs, at_start, velocity, duration, touched)) {
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
FLUIDITY_CPU_AND_GPU inline HitTime FindCylinderHit(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    OwnPath own_path,
    double duration,
    TouchedSolid touched
) {
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
  switch (ComputeStartCase(inputs, at_start, own_path.velocity, duration, touched)) {
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
    // A cylinder the particle already touched: an approach within the allowance is the response's rounding.
    const bool approaches_rim = gap_rate < -touched.approach_allowance;
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

// Says whether a shape's distance is convex in the solid's own axes, with the normal its distance reports a
// subgradient of it: the condition for the second-order advance of the stepping search (plan decision P18).
//
// Method: with p the point in the solid's own axes, true for
//   flat solid  p.z, linear.
//   plate       |p.z| - half_thickness, and s * p.z - half_thickness for a committed particle (plan decision P3).
//   box         the largest of |p_k| - half_size_k over the own axes.
//   sphere      |p| - radius, the true distance to a convex solid, negative inside.
//   cylinder    the true distance to a convex solid, negative inside (the cylinder's distance formula).
//   False for the domain box: its distance, the smallest gap to a closed side, is not convex, and a turning room
//   searched with the second-order advance would let the path pass a wall. This is the one place that chooses the
//   advance: a new shape adds its case here, and a shape whose distance is not convex (a mesh, the smallest of its
//   triangle distances) returns false and keeps the first-order advance.
// Example: HasConvexDistance(SolidShape::kBox) is true; HasConvexDistance(SolidShape::kDomainBox) is false.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>
FLUIDITY_CPU_AND_GPU constexpr bool HasConvexDistance(SolidShape shape) {
  switch (shape) {
    case SolidShape::kFlatSolid:
    case SolidShape::kPlate:
    case SolidShape::kBox:
    case SolidShape::kSphere:
    case SolidShape::kCylinder:
      return true;
    case SolidShape::kDomainBox:
      break;
  }
  return false;
}

// The lower model of a turning solid's gap after one sample of the stepping search: for every s >= 0,
// g(t + s) >= gap + gap_rate * s - gap_curvature_bound * s^2 / 2 (FindHitOnTurningSolid proves it).
struct GapLowerModel {
  double gap = 0;  // m, positive: the distance at the sample minus the contact tolerance.
  double gap_rate = 0;  // m/s. How fast the gap changes at the sample; negative while it closes.
  double gap_curvature_bound = 0;  // m/s^2, zero or more. The fastest the gap rate can change during the sweep.
};

// Computes the second-order advance of the stepping search: the positive root of the gap's lower model, the time over
// which the model, and so the gap, stays at least 0.
//
// Method: with g = gap, gdot = gap_rate and K = gap_curvature_bound, the root of g + gdot s - K s^2 / 2 = 0 written
//   so that no two terms of opposite sign meet (plan decision P18):
//   gdot <= 0  2 g / (Sqrt(gdot^2 + 2 K g) - gdot): the denominator adds two terms >= 0.
//   gdot > 0   (gdot + Sqrt(gdot^2 + 2 K g)) / K: the numerator adds two positive terms.
//   The sum under the root adds two terms >= 0, so rounding cannot make it negative. K == 0 with gdot >= 0 is the one
//   zero denominator: the gap cannot shrink, and the advance is whole_sweep_duration, which ends the search free. A
//   root term of 0 with K > 0, which only underflow gives, returns 0: no second-order advance, the first-order one
//   stands. Same bits on CPU and GPU: only + - * /, Sqrt and comparisons.
// Units: s for whole_sweep_duration and the result.
// Example: g = 2^-17 - 2^-30 m, gdot = 0 and K = 0.5 + 2^-17 m/s^2 (the hover on a box's turn axis): Sqrt(2 g / K),
//   about 0.0055 s. g = 0.1, gdot = -1, K = 0: 0.1 s, the gap over the speed at which it closes.
// Requires: gap > 0 and gap_curvature_bound >= 0.
// Read: docs/method-paper/fluidity-method.typ <eq:second-order-advance>, <sec-first-hit>
FLUIDITY_CPU_AND_GPU inline double ComputeSecondOrderAdvance(GapLowerModel gap_model, double whole_sweep_duration) {
  assert(gap_model.gap > 0 && gap_model.gap_curvature_bound >= 0);
  const double gap = gap_model.gap;  // m
  const double gap_rate = gap_model.gap_rate;  // m/s
  const double gap_curvature_bound = gap_model.gap_curvature_bound;  // m/s^2
  const bool gap_can_shrink = gap_curvature_bound > 0 || gap_rate < 0;
  if (!gap_can_shrink) {
    return whole_sweep_duration;
  }
  // Method paper <eq:second-order-advance>:  s = 2 g / (Sqrt(gdot^2 + 2 K g) - gdot)  (gdot <= 0),
  //   s = (gdot + Sqrt(gdot^2 + 2 K g)) / K  (gdot > 0)
  const double root_term = math::Sqrt(gap_rate * gap_rate + 2 * gap_curvature_bound * gap);  // m/s
  if (gap_rate > 0) {
    return (gap_rate + root_term) / gap_curvature_bound;
  }
  return root_term > 0 ? 2 * gap / (root_term - gap_rate) : 0;
}

// Finds the first hit of a straight path on one turning solid by the stepping search: from the sweep's start, each
// advance is a time over which the gap provably stays positive, so no advance can pass the moment the path reaches
// the solid grown by the tolerance.
//
// Method (method paper <sec-first-hit>), with tol = inputs.contact_tolerance, x(t) = sweep.start +
//   velocity_for_this_step * t, v_rel = velocity_for_this_step - linear_velocity, and the gap g(t) =
//   ComputeDistanceForParticle(x(t)) - tol, at the pose of start_elapsed + t and with the particle's plate side:
//   1 r_max is the larger of the distances from x(0) to the solid's origin at start_elapsed and from x(duration) to
//     it at start_elapsed + duration. bound = Length(v_rel) + Length(angular_velocity) * r_max: the gap changes no
//     faster than that. K = Length(angular_velocity) * (2 * Length(v_rel) + Length(angular_velocity) * r_max), the
//     gap curvature bound: the gap rate changes no faster than that. Both proofs are in the body.
//   2 From t = 0: a hit at t when g(t) <= kTurningSolidHitGapShare * tol, with at_hit the distance there; otherwise
//     t <- t + advance, and the path is free once t >= duration. The first-order advance is g / bound. When
//     HasConvexDistance(solid.shape), the second-order advance is the positive root of g + gdot s - K s^2 / 2, with
//     the gap rate gdot = Dot(normal, velocity_for_this_step - ComputeSolidVelocity at x(t)) and the normal of the
//     sample, and the advance is the larger of the two; otherwise it is the first-order advance alone. A hit after
//     at least one advance is at a distance in [tol, 1.25 tol] up to rounding; a hit at 0 reports the start's
//     distance, at most 1.25 tol and possibly below tol.
//   3 When kMaxAdvanceCount advances end neither way, the search cannot answer: the close call.
//   Cost: the first-order advance closes the gap by the share rate / bound of it, rate the speed at which the gap
//   really closes, so alone it needs about ln(4 g / tol) * bound / rate advances: 30 in the example below. The
//   second-order advance follows the true gap rate, and lands near d = tol when the path approaches: 3 advances in
//   the example. On the 30,000 generated paths of the tests (turning boxes, plates, flat solids, cylinders, a
//   quarter-turn arm and a paddle) the mean is 3.5 to 8 advances per set, the 99th percentile at most 41, and there
//   is no close call; first order alone gave close calls on up to 80 % of a set. A close call remains for a path that
//   hovers near the solid or grazes it, with almost no motion along the normal, where each advance is only about
//   Sqrt(2 g / K): a particle at rest 2^-17 m above a box's face, on its turn axis, needs 182 advances.
//   There is no touching rule: a turning face does not stay in its plane, so the touching rule's plane argument fails.
//   A particle resting on the solid, at distance tol, has g(0) = 0 and is hit at 0 in every step; the response and
//   the slide in the solid's axes then carry it with the solid (plan decisions P2 and P7). The gap is read only
//   through ComputeDistanceForParticle and may come from any face: the first-order advance fits any distance that
//   changes no faster than the point it is measured at, the second-order advance a convex one whose normal is a
//   subgradient (HasConvexDistance). Same bits on CPU and GPU cannot be promised: ComputeSolidPose turns the axes
//   with Sin. Every other operation is + - * /, Sqrt or a comparison.
// Units: m, m/s and s, world.
// Example: a box of half size 0.5 m at the origin turning at {0, 0, 1} rad/s, tol 2^-30 m, and a particle at rest at
//   0.6 * {cos 0.3, sin 0.3, 0} m for 1 s: bound 0.6 m/s and K 0.6 m/s^2; after 3 advances a hit on kHighX at
//   0.8857 s, 1.4e-10 s before 0.3 + acos((0.5 + tol) / 0.6), the time the true distance reaches tol.
// Requires: solid_index < inputs.solid_count; hit_on_solid is not null (advance_count may be null).
//   sweep.start_elapsed and sweep.duration are finite and zero or more, and their sum is at most inputs.step_duration
//   up to rounding. inputs.contact_tolerance is zero or more.
// Fails: kTooCloseToCall  more than kMaxAdvanceCount advances: the path hovers near the solid or grazes it, with
//                         too little motion along the normal (see Cost). The caller shortens the step and tries again.
//        On failure *hit_on_solid is unchanged. *advance_count, when advance_count is not null, holds the advances
//        made on every return.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>, <proc-find-first-hit>
FLUIDITY_CPU_AND_GPU inline ContactError FindHitOnTurningSolid(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    Sweep sweep,
    FirstHit* hit_on_solid,
    int* advance_count
) {
  assert(solid_index < inputs.solid_count);
  assert(hit_on_solid != nullptr);
  assert(math::IsFinite(sweep.start_elapsed) && sweep.start_elapsed >= 0);
  assert(math::IsFinite(sweep.duration) && sweep.duration >= 0);
  assert(inputs.contact_tolerance >= 0);
  // Steps (method paper <proc-find-first-hit>): 1 the bounds on how fast the gap and its rate change  2 advance until
  //   a hit, the end of the sweep or the last advance
  const Solid& solid = inputs.solids[solid_index];
  const double contact_tolerance = inputs.contact_tolerance;
  const Vec3 velocity_for_this_step = sweep.velocity_for_this_step;
  const Vec3 relative_velocity = velocity_for_this_step - solid.motion.linear_velocity;  // m/s, world
  const double relative_speed = math::Length(relative_velocity);  // m/s
  const double turn_rate = math::Length(solid.motion.angular_velocity);  // rad/s
  // The count is written as it grows, so every return leaves the advances made in it.
  int unread_advance_count = 0;
  int* const advances_made = advance_count != nullptr ? advance_count : &unread_advance_count;
  *advances_made = 0;

  // Step 1: why no gap shrinks faster than the bound. Relative to the solid the particle is at the own point
  // p(t) = Transpose(A(t)) (x(t) - o(t)), A the axes and o the origin, and it moves at
  // p'(t) = Transpose(A(t)) (relative_velocity - angular_velocity x (x(t) - o(t))). A(t) is a rotation, so that speed
  // is at most |relative_velocity| + |angular_velocity| |x(t) - o(t)|. Every distance ComputeDistanceForParticle
  // gives changes no faster than p moves: a plane's distance, the largest of a box's plane distances, and the true
  // distance to a cylinder each grow by at most the length of a move of p. Finally x(t) - o(t) = (x(0) - o(0)) +
  // relative_velocity * t is a straight line in t, whose length is convex in t and so largest at one end of the
  // sweep, r_max. Each end needs the origin at its own time: a moving origin enters only there.
  // Method paper <sec-first-hit>:  bound = |v - v_lin| + |omega| r_max
  const Vec3 end_position = sweep.start + velocity_for_this_step * sweep.duration;
  const Vec3 origin_at_start = ComputeSolidPose(solid, sweep.start_elapsed).origin;
  const Vec3 origin_at_end = ComputeSolidPose(solid, sweep.start_elapsed + sweep.duration).origin;
  const double largest_distance_to_origin =
      math::Max(math::Length(sweep.start - origin_at_start), math::Length(end_position - origin_at_end));  // m, r_max
  const double gap_rate_bound = relative_speed + turn_rate * largest_distance_to_origin;  // m/s
  // Why the gap's rate changes no faster than K. With A' = [omega]x A and o' = linear_velocity, both constant in the
  // step, the own point's acceleration is p'' = Transpose(A) (-2 omega x v_rel + omega x (omega x (x - o))), the two
  // terms papers call Coriolis and centripetal. Its length is at most |omega| (2 |v_rel| + |omega| |x - o|), and
  // |x - o| <= r_max at every time of the sweep, as above. No small-angle step is taken, so this holds for any turn,
  // kMaxTurnPerStep and beyond. r_max must cover both ends of the sweep, not the sample's time: an advance reaches
  // times where |x - o| is larger, and a K built from the start alone, or from the end alone, let advances carry
  // particles centimetres or millimetres into a quarter-turn arm.
  // Method paper <sec-first-hit>:  K = |omega| (2 |v - v_lin| + |omega| r_max)
  const double gap_curvature_bound = turn_rate * (2 * relative_speed + turn_rate * largest_distance_to_origin);
  const bool takes_second_order_advance = HasConvexDistance(solid.shape);

  // Step 2: why an advance never passes a crossing. First order: from a sample at t with gap g > 0, for every s in
  // [t, t + g / bound] the gap is at least g - bound * (s - t) >= 0. Second order, for a convex distance d: the normal
  // the distance reports, written in own axes, is a unit subgradient of d, so d(p(s)) >= d(p(t)) + n_own . (p(s) -
  // p(t)) for every s, inside the solid too. Shape by shape: a flat solid and a committed plate are linear, n_own
  // their gradient; a plate's middle plane p.z = 0 reports +z, inside [-1, 1], the subdifferential of |p.z|; a box
  // reports one face of largest gap, a subgradient of the max also on an edge or a corner where faces tie, and at
  // p_k = 0 its low face, inside [-1, 1] again; a cylinder outside reports the gradient of its true distance, the
  // rim's included; inside, the wall or the cap whose gap is the larger, and on the axis the degenerate normal, inside
  // the unit disc of subgradients of the distance from the axis. With n_own . p'(t) = gdot and |n_own . p''| <= K:
  //   g(s) >= g(t) + gdot (s - t) - K (s - t)^2 / 2,
  // whose one positive root is a time the gap stays at least 0 for. The larger of two safe advances is safe. Either
  // way the path stays at distance tol or more over the whole advance, its end included, so it cannot reach the grown
  // solid between two samples, still less pass through the solid. The first sample with g <= tol / 4 therefore comes
  // no later than the moment the path first reaches the grown solid, and after at least one advance its distance is
  // in [tol, 1.25 tol] (a second-order advance usually lands near tol). An advance that reaches duration covers the
  // rest of the sweep, which is then free. Rounding: a computed distance is off by about 32 rounding units of the
  // largest coordinate, at most 1/128 of tol, which holds 4096 of them (contact/contact_tolerance.h); gdot is off by
  // a few rounding units of |v| + |v_lin| + |omega| r_max, which over one advance moves the model by far less; the
  // bound, K, the root and the division change an advance by a few rounding units of itself. So the path stays farther
  // than about tol - tol / 64 before the hit. A bound of 0 means the gap cannot change: the advance is infinite and the
  // path is free.
  double elapsed = 0;  // s, since the start of the sweep.
  while (true) {
    const Vec3 position = sweep.start + velocity_for_this_step * elapsed;
    const SolidPose pose = ComputeSolidPose(solid, sweep.start_elapsed + elapsed);
    const DistanceToSolid at_position =
        ComputeDistanceForParticle(inputs, solid_index, pose, position, inputs.plate_side);
    const double gap = at_position.distance - contact_tolerance;  // m
    if (gap <= kTurningSolidHitGapShare * contact_tolerance) {
      *hit_on_solid = {.elapsed = elapsed, .solid_index = solid_index, .at_hit = at_position};
      return ContactError::kNone;
    }
    // A path that hovers near the solid or grazes it, with little motion along the normal runs out of advances here
    // (Cost in the docstring).
    if (*advances_made == kMaxAdvanceCount) {
      return ContactError::kTooCloseToCall;
    }
    const double first_order_advance = gap / gap_rate_bound;  // s
    double advance = first_order_advance;  // s
    if (takes_second_order_advance) {
      // The gap rate reads the surface velocity at the particle, not the solid's linear velocity alone: the turn moves
      // the face under the particle.
      // Method paper <sec-first-hit>:  gdot = n . (v - u(x)),  u(x) = v_lin + omega x (x - o)
      const Vec3 surface_velocity = ComputeSolidVelocity(solid, pose, position);
      const GapLowerModel gap_model = {
          .gap = gap,
          .gap_rate = math::Dot(at_position.normal_out_of_solid, velocity_for_this_step - surface_velocity),
          .gap_curvature_bound = gap_curvature_bound,
      };
      const double second_order_advance = ComputeSecondOrderAdvance(gap_model, sweep.duration);  // s
      // A comparison, not math::Max: a NaN second-order advance leaves the first-order advance in place.
      if (second_order_advance > first_order_advance) {
        advance = second_order_advance;
      }
    }
    elapsed += advance;
    ++*advances_made;
    if (elapsed >= sweep.duration) {
      *hit_on_solid = FirstHit{};
      return ContactError::kNone;
    }
  }
}

// Computes whether one solid is a turning solid: its shape turns during the step, so its first hit comes from the
// stepping search, not from a closed form.
//
// Method: with shape_turning = ComputeShapeTurningAngularVelocity(solid), the part of the angular velocity that
//   changes the shape. A flat solid, a plate or a box turns when shape_turning is not exactly {0, 0, 0} (plan
//   decision P7): for a flat solid and a plate it is the part across the normal, so a spin about the normal is not a
//   turn (plan decision P17), and it is {0, 0, 0} when that part is no longer than kNegligibleTurnShare of the
//   angular velocity, which is all a spin about a tilted normal leaves by rounding. A sphere never turns: shape_turning
//   is {0, 0, 0}. A cylinder turns when Length(shape_turning) * step_duration * Length(half_size) > tol / 8 (plan
//   decision P17): its axis then tilts enough to move its surface by more than tol / 8 during the step. Below that,
//   which includes a spin about a tilted axis that rounding leaves with a tiny part across it, the closed form at the
//   pose of start_elapsed holds. A domain box never moves. Same bits on CPU and GPU: only + - * and Sqrt.
// Example: a box turning at {0, 0, 1} rad/s turns; a flat solid with normal {0, 0, 1} turning at {0, 0, 5} rad/s does
//   not, and at {1, 0, 0} rad/s it does.
// Requires: solid_index < inputs.solid_count.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>, <sec-round-solids>
FLUIDITY_CPU_AND_GPU inline bool ComputeIsTurningSolid(const SolidContactInputs& inputs, std::uint32_t solid_index) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  const Vec3 shape_turning = ComputeShapeTurningAngularVelocity(solid);
  if (solid.shape != SolidShape::kCylinder) {
    // Plan decision P7: a turn of any size other than exactly zero.
    return shape_turning.x != 0 || shape_turning.y != 0 || shape_turning.z != 0;
  }
  // Plan decision P17: rounding leaves a tiny part across a cylinder's axis even in a spin about a tilted axis, so
  // its test is a bound on how far the tilt moves the surface, not an exact zero.
  // Method paper <sec-round-solids>:  |omega_across| T |half_size| <= tol / 8
  const double surface_movement_from_tilt =
      math::Length(shape_turning) * inputs.step_duration * math::Length(solid.half_size);  // m
  return surface_movement_from_tilt > inputs.contact_tolerance / 8;
}

// Finds the first hit of a straight path on one solid: the steps "solid that does not turn" and "turning solid" of
// FindFirstHit for that solid.
//
// Method: ComputeIsTurningSolid says whether the solid's shape turns during the step: a box whose angular velocity is
//   not exactly {0, 0, 0}, a flat solid or a plate whose angular velocity has a part across its normal (beyond
//   rounding), or a cylinder whose axis tilts beyond plan decision P17's bound; never a sphere or a domain box. One
//   switch over the shape gives a turning solid FindHitOnTurningSolid, the same search for every shape that can turn,
//   and every other solid its closed form:
//   flat solid  FindHitOnConvexSolid with its one face plane, kHighZ at own z = 0.
//   plate       FindHitOnConvexSolid with kLowZ and kHighZ at own z = -half_thickness and +half_thickness, or only
//               the committed face (ToFacePlane) when inputs.plate_side names this plate.
//   box         FindHitOnConvexSolid with its six face planes, in face order.
//   domain box  FindHitOnDomainBox.
//   sphere      FindSphereHit of the sweep in the sphere's own axes (ComputeOwnPath), then ToFirstHit.
//   cylinder    FindCylinderHit likewise.
//   touched says whether the particle had a response in the step, which gives every solid the touched rule
//   (FindFirstHitOfSweep): the closed forms' touching rule reads it (ComputeStartCase); the stepping search has no
//   touching rule and ignores it.
//   *hit_on_solid has solid_index kNoSolid when the path is free. Same bits on CPU and GPU when the angular velocity
//   is zero: only + - * /, Abs, Sqrt and comparisons, since ComputeSolidPose of a solid that does not move its axes
//   needs no Sin. A turning solid, and a spinning sphere or cylinder, pose their axes with Sin, which may differ in the
//   last bits.
// Units: s for start_elapsed and duration; m and m/s, world, for the path.
// Example: a box of half size 0.5 m at the origin and the path from {-1, 0.1, 0.2} at {2, 0, 0} m/s for 1 s: elapsed
//   0.25 - tol / 2 on kLowX. A sphere of radius 0.5 m at the origin and the path from {-2, 0, 0} at {1, 0, 0} m/s for
//   2 s: elapsed 1.5 - tol on kCurved. The same box turning at {0, 0, 1} rad/s and a particle at rest at
//   0.6 * {cos 0.3, sin 0.3, 0} m for 1 s: elapsed 0.8857 s on kHighX, by the stepping search.
// Requires: solid_index < inputs.solid_count; hit_on_solid is not null. The solid's shape is one of SolidShape's
//   values.
// Fails: kBadInput        the solid is of an unknown shape; a debug build stops at an assertion first.
//        kTooCloseToCall  the stepping search toward a turning solid needs more than kMaxAdvanceCount advances; the
//                         only close call is kSlowApproachToTurningSolid.
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
    TouchedSolid touched,
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
  const Vec3 half_size = solid.half_size;
  const bool is_turning_solid = ComputeIsTurningSolid(inputs, solid_index);
  switch (solid.shape) {
    case SolidShape::kFlatSolid: {
      // Step 2: the closed forms below hold only while the faces keep their direction.
      if (is_turning_solid) {
        return FindHitOnTurningSolid(inputs, solid_index, sweep, hit_on_solid, nullptr);
      }
      // Step 1: one face plane.
      const FacePlane surface = {.face = SolidFace::kHighZ, .axis = Axis::kZ, .sign_out_of_solid = 1, .offset = 0};
      *hit_on_solid = FindHitOnConvexSolid(inputs, solid_index, sweep, &surface, 1, touched);
      return ContactError::kNone;
    }
    case SolidShape::kPlate: {
      if (is_turning_solid) {
        return FindHitOnTurningSolid(inputs, solid_index, sweep, hit_on_solid, nullptr);
      }
      // Step 1: a committed particle sees only the committed face (plan decision P3). For every start this gives the
      // two-face result: a start outside the committed half space is beyond the committed face, which both forms
      // enter first, and every other start is settled by the start rule on the P3 distance. It states P3 here.
      const PlateSide plate_side = inputs.plate_side;
      if (plate_side.plate_id == solid.solid_id) {
        const FacePlane committed_face = ToFacePlane(plate_side.side, half_size.z);
        *hit_on_solid = FindHitOnConvexSolid(inputs, solid_index, sweep, &committed_face, 1, touched);
        return ContactError::kNone;
      }
      // A C array: GPU code cannot call std::array's member functions (nvcc without --expt-relaxed-constexpr).
      // NOLINTNEXTLINE(modernize-avoid-c-arrays): see the comment above.
      const FacePlane plate_faces[] = {
          {.face = SolidFace::kLowZ, .axis = Axis::kZ, .sign_out_of_solid = -1, .offset = half_size.z},
          {.face = SolidFace::kHighZ, .axis = Axis::kZ, .sign_out_of_solid = 1, .offset = half_size.z},
      };
      *hit_on_solid = FindHitOnConvexSolid(inputs, solid_index, sweep, plate_faces, 2, touched);
      return ContactError::kNone;
    }
    case SolidShape::kBox: {
      if (is_turning_solid) {
        return FindHitOnTurningSolid(inputs, solid_index, sweep, hit_on_solid, nullptr);
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
      *hit_on_solid = FindHitOnConvexSolid(inputs, solid_index, sweep, box_faces, 6, touched);
      return ContactError::kNone;
    }
    case SolidShape::kDomainBox:
      // Step 1: CheckSolid refuses a domain box that moves, so it never turns.
      assert(!is_turning_solid);
      *hit_on_solid = FindHitOnDomainBox(inputs, solid_index, sweep, touched);
      return ContactError::kNone;
    case SolidShape::kSphere: {
      // Step 1: a spin about the centre moves the surface but not the shape (plan decision P17), so a sphere never
      // turns and always has its closed form, the touching rule included.
      assert(!is_turning_solid);
      const HitTime hit_time =
          FindSphereHit(inputs, solid_index, ComputeOwnPath(inputs, solid_index, sweep), duration, touched);
      *hit_on_solid = ToFirstHit(inputs, solid_index, sweep, hit_time);
      return ContactError::kNone;
    }
    case SolidShape::kCylinder: {
      // Step 2: an axis that tilts beyond plan decision P17's bound.
      if (is_turning_solid) {
        return FindHitOnTurningSolid(inputs, solid_index, sweep, hit_on_solid, nullptr);
      }
      // Step 1: the wall, the caps and the rims, with the touching rule and the rim corner.
      const HitTime hit_time =
          FindCylinderHit(inputs, solid_index, ComputeOwnPath(inputs, solid_index, sweep), duration, touched);
      *hit_on_solid = ToFirstHit(inputs, solid_index, sweep, hit_time);
      return ContactError::kNone;
    }
  }
  // An unknown shape has no closed form, so reaching this line is a caller bug: CheckSolid refuses it.
  assert(false && "FindFirstHitOnSolid has no closed form for this shape");
  return ContactError::kBadInput;
}

// Computes which of two first hits FindFirstHit keeps: the earlier one, and of two equally early hits the one on the
// lower solid index, whatever order the solids were searched in.
//
// Method: candidate wins when it is a hit and kept is free, when it is strictly earlier, or when it is equally early
//   with a lower solid_index; otherwise kept stays. A free candidate (solid_index kNoSolid) never wins. Same bits on
//   CPU and GPU: only comparisons.
// Example: kept {elapsed 0.5, solid_index 1} and candidate {elapsed 0.5, solid_index 0}: the candidate. Candidate
//   {elapsed 0.5, solid_index 2}: kept.
// Read: docs/method-paper/fluidity-method.typ <proc-find-first-hit>
FLUIDITY_CPU_AND_GPU constexpr FirstHit ComputeEarlierHit(FirstHit kept, FirstHit candidate) {
  const bool is_hit = candidate.solid_index != kNoSolid;
  const bool has_kept_hit = kept.solid_index != kNoSolid;
  const bool is_earlier = !has_kept_hit || candidate.elapsed < kept.elapsed;
  const bool is_tie_with_lower_index =
      has_kept_hit && candidate.elapsed == kept.elapsed && candidate.solid_index < kept.solid_index;
  return is_hit && (is_earlier || is_tie_with_lower_index) ? candidate : kept;
}

// Finds the earliest hit of a sweep over the solids that do not turn and the turning solids: the step "turning solid"
// of FindFirstHit, after its step "solid that does not turn" has filled *earliest.
//
// Method: each turning solid (ComputeIsTurningSolid) in array order is searched by FindFirstHitOnSolid with the
//   sweep's duration capped at earliest->elapsed when *earliest is a hit. A hit at or after that time could not win, so
//   the search stops there; a search that reaches the cap is free and the kept hit stands, since the path stayed at
//   distance tol or more up to it, and a close call beyond the cap is never reached. ComputeEarlierHit keeps the
//   earlier hit, and of two equally early hits the one on the lower index, although the turning solids are searched
//   after solids of any index. The solids that do not turn are skipped here.
// Units: m, m/s and s, world.
// Ties: equal times keep the lower solid index.
// Example: solid 0 a box turning at {0, 0, 1} rad/s with a particle at rest 2^-17 m above its top face on the turn
//   axis, and *earliest the hit of a flat solid 1 at 0.0999924 s: the box is searched up to that time, free after 19
//   advances, and *earliest stays.
// Requires: earliest is not null and holds the earliest hit of the solids that do not turn (FirstHit{} when free).
//   The requirements of FindFirstHitOnSolid for every solid.
// Fails: kTooCloseToCall  a turning solid's search needs more than kMaxAdvanceCount advances before the cap
//                         (FindHitOnTurningSolid: Cost).
//        *earliest is then FirstHit{} with solid_index the failing solid's index, so the caller can name it.
// Read: docs/method-paper/fluidity-method.typ <proc-find-first-hit>, <sec-first-hit>
FLUIDITY_CPU_AND_GPU inline ContactError
FindEarliestHitWithTurningSolids(const SolidContactInputs& inputs, Sweep sweep, FirstHit* earliest) {
  assert(earliest != nullptr);
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    if (!ComputeIsTurningSolid(inputs, solid_index)) {
      continue;
    }
    const bool has_earlier_hit = earliest->solid_index != kNoSolid;
    const double search_duration = has_earlier_hit ? earliest->elapsed : sweep.duration;  // s
    FirstHit hit_on_solid;
    // A turning solid has no touching rule, so whether the step had a response does not matter.
    const TouchedSolid not_touched;
    const ContactError error = FindFirstHitOnSolid(
        inputs,
        solid_index,
        sweep.start,
        sweep.velocity_for_this_step,
        sweep.start_elapsed,
        search_duration,
        not_touched,
        &hit_on_solid
    );
    // A turning solid fails only with a close call, which names this solid.
    if (error != ContactError::kNone) {
      *earliest = {.elapsed = 0, .solid_index = solid_index, .at_hit = {}};
      return error;
    }
    *earliest = ComputeEarlierHit(*earliest, hit_on_solid);
  }
  return ContactError::kNone;
}

// Finds the first hit of one sweep over every solid, as FindFirstHit does, with what the responses of the particle's
// move so far say: after the first response of the step every solid gets the touching rule of a touched solid.
//
// Method: first every solid that does not turn, in array order, by FindFirstHitOnSolid with touched; then the turning
//   solids (FindEarliestHitWithTurningSolids), each searched only up to the earliest hit found so far. Nothing skips a
//   solid: a solid is hit again when the path really approaches it at the start, or later in the sweep when the start
//   is farther than tol / 16 from its rest gap. With touched.is_touched false this is FindFirstHit, bit for bit.
//   Why every solid gets the touched rule after a response (plan decision P20, item 9, which replaces the list of
//   solids met that P2 kept), in four steps:
//   1 Before any response the velocities are the inputs, and the plain touching rule holds (plan decision P12).
//   2 Every sweep after a response starts where the latest response was made, at its time, with the velocity it
//     returned: a hit is followed by its response at the same position and pose, and the start's response by the
//     first sweep. A solid whose start gap is within the touched rule's window [-tol, tol / 16] there is within
//     tol + tol / 16, so FindActiveSurfaces read it as an active surface of that response, with the same distance
//     function, position and pose (an exact copy dropped from the list has the same normal, solid velocity and so
//     target as the one kept). That response left the velocity approaching every active surface by at most its
//     allowance (ComputeJointChange: one surface, the bound of ComputeLargestApproachLeftAfterResponse; several, the
//     test of step 7 or a solve's answer within its slack), and touched.approach_allowance, the largest allowance of
//     the responses so far, is at least it. One known exception: a surface whose normal agrees with its direction's
//     within kSameNormalAxisDifference but not bit for bit, in a change that uses another direction too, is missed by
//     up to 2^-40 of the change; the sweep then sees an approach and hits it at 0, one more hit, never a missed one.
//     A solid outside the window takes its closed form, whatever touched says.
//   3 What replaces the allowance of each listed entry: one number per velocity, the largest so far. Response j
//     asserts (one active surface) or checks (several, kSqueezedBetweenSolids) b_j * t_left_j <= tol / 64, t_left_j the
//     time left at it. Between response k and the next, a free path approaches any solid at up to max over j <= k of
//     b_j, for at most t_left_k <= t_left_j, so by at most tol / 64. Responses come at most at kMaxHitsPerStep times,
//     so the particle comes closer to any solid by at most tol / 16 in the step.
//   4 The resting window: a particle the touched rule lets rest starts at a gap of at most tol / 16, the window of the
//     active surfaces, and ends the step no deeper than tol / 16 below where it rested. What is proved is that bound,
//     not d >= 0: before the first response the plain rule lets a path end on a surface, d near 0, and the touched
//     stretches after it may take the particle up to tol / 16 past the surface, which the settle lifts back to tol
//     (ApplySettleAtEnd). So, at the extreme of MoveParticle's Requires (speeds times the time left near 2^40 tol), a
//     non-committed plate thinner than tol / 8 could have its middle passed, and one thinner than tol / 16 be passed
//     through, after which the settle lifts the particle to the far side. Neither has been constructed (the review's
//     300,000 crowded scenes with plates down to tol / 4 thick kept every sweep at 0.99945 tol or more) nor proved
//     impossible: a limit under plan decision P3.
// Units: m, m/s and s, world.
// Ties: equal hit times keep the lower solid index, then the face that comes first in face order.
// Example: a fixed flat solid with normal {0.6, 0, 0.8} through the origin, a particle at the rest gap after a hit
//   there, 2.8e-18 m outside it by rounding, moving at the response's {-0.32, 0, 0.24} m/s, whose approach -2^-55 m/s
//   is rounding: touched, with an allowance of 2^-44 m/s, the path is free; not touched, the closed form hits the solid
//   again.
// Requires: as FindFirstHit.
// Fails: as FindFirstHit. On kTooCloseToCall, *first_hit is FirstHit{} with solid_index the turning solid that could
//   not be searched; *close_call, when not null, is always written.
// Read: docs/method-paper/fluidity-method.typ <proc-find-first-hit>, <sec-first-hit>
FLUIDITY_CPU_AND_GPU inline ContactError FindFirstHitOfSweep(
    const SolidContactInputs& inputs,
    Sweep sweep,
    TouchedSolid touched,
    FirstHit* first_hit,
    ContactCloseCall* close_call
) {
  assert(first_hit != nullptr);
  assert(inputs.solids != nullptr || inputs.solid_count == 0);
  assert(math::IsFinite(sweep.start) && math::IsFinite(sweep.velocity_for_this_step));
  assert(math::IsFinite(sweep.start_elapsed) && sweep.start_elapsed >= 0);
  assert(math::IsFinite(sweep.duration) && sweep.duration >= 0);
  // With duration >= 0 this also keeps start_elapsed within the step.
  assert(sweep.duration <= inputs.step_duration - sweep.start_elapsed);
  assert(math::IsFinite(inputs.contact_tolerance) && inputs.contact_tolerance >= 0);
  if (close_call != nullptr) {
    *close_call = ContactCloseCall::kNone;
  }
  // Steps 1 and 2 of FindFirstHit run in FindFirstHitOnSolid, one solid at a time; step 3 keeps the earliest hit
  // here. The solids that do not turn go first, so every turning solid is searched only up to the earliest hit.
  FirstHit earliest;
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    assert(CheckSolid(inputs.solids[solid_index], inputs.step_duration) == ContactError::kNone);
    if (ComputeIsTurningSolid(inputs, solid_index)) {
      continue;
    }
    FirstHit hit_on_solid;
    const ContactError error = FindFirstHitOnSolid(
        inputs,
        solid_index,
        sweep.start,
        sweep.velocity_for_this_step,
        sweep.start_elapsed,
        sweep.duration,
        touched,
        &hit_on_solid
    );
    // A solid that does not turn fails only for an unknown shape, a caller bug that CheckSolid catches first.
    if (error != ContactError::kNone) {
      return error;
    }
    // Step 3: solids come in index order here, so a later one wins only when strictly earlier.
    earliest = ComputeEarlierHit(earliest, hit_on_solid);
  }
  // The turning solids: a close call is passed on with its reason, always the stepping search running out of
  // advances (a hover or a graze), and its solid. The step
  // is shortened and tried again.
  const ContactError turning_error = FindEarliestHitWithTurningSolids(inputs, sweep, &earliest);
  if (turning_error == ContactError::kTooCloseToCall) {
    if (close_call != nullptr) {
      *close_call = ContactCloseCall::kSlowApproachToTurningSolid;
    }
    *first_hit = earliest;
    return turning_error;
  }
  if (turning_error != ContactError::kNone) {
    return turning_error;
  }
  *first_hit = earliest;
  return ContactError::kNone;
}

// One particle's move through the solids while MoveParticle works on it: the result so far, the part of the step
// used, and what the responses so far left by rounding.
struct ParticleMove {
  SolidContactResult result{};  // The position and velocities so far; the record of the first hit; the reaction.
  double elapsed = 0;  // s since the start of the step: the part of the step already used.
  // A joint response was made in this step: every solid then has the touched rule (FindFirstHitOfSweep).
  bool has_response = false;
  // m/s. The largest approach allowance of the responses so far for velocity.for_this_step, 0 before any: what they may
  // have left along any active normal by rounding. It is carried, not recomputed: the bound is proved from the
  // velocity a response received, and one from the velocity it returned can be far smaller than what is left, for
  // example after a slip just below the stick limit.
  double this_step_approach_allowance = 0;
  // m/s. The same for velocity.for_next_step. Each velocity has its own: a response to a for_this_step at rest has an
  // allowance of 0, while it leaves rounding in a moving for_next_step (-2^-55 m/s after a response to {-2, 0, -2} m/s
  // at normal {0.6, 0, 0.8}), which compared with 0 would be an approach. The sweeps never read it: for_next_step does
  // not move the particle.
  double next_step_approach_allowance = 0;
};

// Checks the values one particle's move reads, and says kBadInput when one is out of range.
//
// Method: in this order: the start and both velocities are finite; step_duration is finite and positive; friction is
//   finite and zero or more; start_overlap_tolerance is finite and zero or more; contact_tolerance is finite and
//   positive; solids is not null unless solid_count is 0; the solid_ids are strictly ascending; and the plate side is
//   {0, kNoPlate}, or names the solid_id of a plate among the solids with the side kAlongNormal or kAgainstNormal (plan
//   decision P3). The solids themselves are not checked here: the step calls CheckSolid once per step (plan decision
//   P9).
// Example: step_duration 0, a friction of -1, a contact_tolerance of 0, solid_ids {2, 1} or the plate side {99,
//   kAlongNormal} with no solid of id 99 each give kBadInput.
// Fails: kBadInput  a value above is not finite or out of range. A scene or particle array that went wrong upstream;
//                   the step stops.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline ContactError CheckSolidContactInputs(const SolidContactInputs& inputs) {
  const PerParticleVelocity<Vec3> velocity = inputs.velocity;
  const bool motion_is_finite = math::IsFinite(inputs.start_position) && math::IsFinite(velocity.for_next_step)
      && math::IsFinite(velocity.for_this_step);
  const bool step_duration_is_positive = math::IsFinite(inputs.step_duration) && inputs.step_duration > 0;
  const bool friction_is_valid = math::IsFinite(inputs.friction) && inputs.friction >= 0;
  const bool overlap_tolerance_is_valid =
      math::IsFinite(inputs.start_overlap_tolerance) && inputs.start_overlap_tolerance >= 0;
  const bool contact_tolerance_is_valid = math::IsFinite(inputs.contact_tolerance) && inputs.contact_tolerance > 0;
  const bool solids_are_given = inputs.solids != nullptr || inputs.solid_count == 0;
  const bool values_are_valid = motion_is_finite && step_duration_is_positive && friction_is_valid
      && overlap_tolerance_is_valid && contact_tolerance_is_valid && solids_are_given;
  if (!values_are_valid) {
    return ContactError::kBadInput;
  }
  // Ascending ids give the order of solids and of equal hit times on every processor.
  for (std::uint32_t solid_index = 1; solid_index < inputs.solid_count; ++solid_index) {
    if (inputs.solids[solid_index].solid_id <= inputs.solids[solid_index - 1].solid_id) {
      return ContactError::kBadInput;
    }
  }
  // Plan decision P3: a committed particle sees its plate as a half space, so the side must name a plate.
  const PlateSide plate_side = inputs.plate_side;
  const bool is_committed = plate_side.plate_id != 0;
  if (!is_committed) {
    return plate_side.side == SolidSide::kNoPlate ? ContactError::kNone : ContactError::kBadInput;
  }
  const bool names_a_side = plate_side.side == SolidSide::kAlongNormal || plate_side.side == SolidSide::kAgainstNormal;
  if (!names_a_side) {
    return ContactError::kBadInput;
  }
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    if (solid.solid_id == plate_side.plate_id) {
      return solid.shape == SolidShape::kPlate ? ContactError::kNone : ContactError::kBadInput;
    }
  }
  return ContactError::kBadInput;
}

// Converts the face of a hit into the side of the plate that was hit: kAlongNormal for kHighZ, kAgainstNormal for
// kLowZ, and kNoPlate when the solid is not a plate.
//
// Example: a plate hit on face kLowZ: kAgainstNormal. A box hit on kHighZ: kNoPlate.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU constexpr SolidSide ToSideAtHit(const Solid& solid, SolidFace face) {
  if (solid.shape != SolidShape::kPlate) {
    return SolidSide::kNoPlate;
  }
  return face == SolidFace::kHighZ ? SolidSide::kAlongNormal : SolidSide::kAgainstNormal;
}

// Adds one change of velocity.for_next_step, made against one solid, to the reaction of a particle's move (plan
// decision P5), and fails when a second reacting solid would need the one record.
//
// Method: a solid without reports_reaction, and a change of exactly {0, 0, 0}, change nothing: a zero change adds
//   nothing to the push on the solid, so it neither claims the record nor conflicts with it (an exact test, the same
//   branch on CPU and GPU). The first reacting solid with a change fills the reaction: its index, the position of the
//   change and the change itself, bit for bit. Later changes against the same solid are added to the change;
//   hit_position stays the first. A change against a different reacting solid fails, because one record cannot hold
//   both.
// Units: m, world, for the position; m/s, world, for the change.
// Example: two changes {-1, 0, 0} and {0, 0, 2} m/s against the same reacting solid sum to {-1, 0, 2} m/s, with the
//   position of the first. A change {0, 0, 0} against a second reacting solid leaves the reaction as it is.
// Requires: solid_index < inputs.solid_count; reaction and failure are not null.
// Fails: kTooCloseToCall  a second, different solid with reports_reaction changed velocity.for_next_step in this
//                         step. *failure is kSecondReactingSolid with that solid's solid_id. A shorter step separates
//                         two hits, but not two reacting solids that one joint response pushes: that fails at every
//                         step length.
//        *failure is written only on failure.
// Read: docs/method-paper/fluidity-method.typ <sec-contact-response>, <proc-move-particle>
//
// clang-tidy: the position and the change are a point in m and a velocity in m/s, both Vec3; the swappable-parameters
// check flags the two.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU inline ContactError ApplyReactionChange(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    Vec3 change_position,
    Vec3 velocity_for_next_step_change,
    SolidReaction* reaction,
    ContactFailure* failure
) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  assert(solid_index < inputs.solid_count);
  assert(reaction != nullptr && failure != nullptr);
  const Solid& solid = inputs.solids[solid_index];
  const Vec3 change = velocity_for_next_step_change;
  const bool is_zero_change = change.x == 0 && change.y == 0 && change.z == 0;
  if (!solid.reports_reaction || is_zero_change) {
    return ContactError::kNone;
  }
  if (reaction->solid_index == kNoSolid) {
    *reaction = {
        .solid_index = solid_index,
        .hit_position = change_position,
        .velocity_for_next_step_change = change,
    };
    return ContactError::kNone;
  }
  if (reaction->solid_index != solid_index) {
    *failure = {
        .error = ContactError::kTooCloseToCall,
        .close_call = ContactCloseCall::kSecondReactingSolid,
        .particle_id = 0,
        .solid_id = solid.solid_id,
        .magnitude = 0,
    };
    return ContactError::kTooCloseToCall;
  }
  reaction->velocity_for_next_step_change = reaction->velocity_for_next_step_change + change;
  return ContactError::kNone;
}

// Checks that one more hit, on one solid, stays within kMaxHitsPerStep after the hits of result, and says
// kTooCloseToCall when it would be the fifth.
//
// Method: result.hit_count == kMaxHitsPerStep fails: a hit is one joint response, however many surfaces it reads, so
//   a particle that meets solids at a fifth separate moment of one step is in a gap too narrow for this step length.
//   The start's hit counts too (plan decision P4); it is one, so the start alone never reaches the limit.
// Example: hit_count 3 passes; hit_count 4 fails, naming the solid of the fifth hit.
// Requires: solid_index < inputs.solid_count; result.hit_count <= kMaxHitsPerStep; failure is not null.
// Fails: kTooCloseToCall  a fifth hit: *failure is kTooManyHits with that solid's solid_id. A shorter step has fewer
//                         hits along its sweeps.
//        *failure is written only on failure.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline ContactError CheckHitCountBelowLimit(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    const SolidContactResult& result,
    ContactFailure* failure
) {
  assert(solid_index < inputs.solid_count);
  assert(result.hit_count <= kMaxHitsPerStep);
  assert(failure != nullptr);
  if (result.hit_count < kMaxHitsPerStep) {
    return ContactError::kNone;
  }
  *failure = {
      .error = ContactError::kTooCloseToCall,
      .close_call = ContactCloseCall::kTooManyHits,
      .particle_id = 0,
      .solid_id = inputs.solids[solid_index].solid_id,
      .magnitude = 0,
  };
  return ContactError::kTooCloseToCall;
}

// Computes the point moved along the normal of one solid to distance contact_tolerance from it: the rest gap.
//
// Method: one switch over the shape.
//   flat solid, plate, box, sphere, cylinder  with the distance d and the normal n that ComputeDistanceForParticle
//     gives at the point and the pose, with the particle's plate side, the result is point + (tol - d) * n. For a flat
//     face this moves the point straight to the face plane at tol; for a sphere, a cylinder's wall or its rim, along
//     the line to the nearest surface point, which keeps that point; for a box, to tol from the face that gives the
//     distance, whose plane is then the farthest of the box's, so the box's distance is tol. It moves a point that is
//     farther than tol back toward the solid too.
//   domain box  each closed side in face order whose distance into the box is below tol moves the point along that
//     side's normal to tol from it. A particle in a corner is closer than tol to two or three sides, and the distance
//     to the domain box is the smallest of theirs, so one move along the nearest side would leave it closer than tol.
//     The sides keep the world axes, so a move along one changes no other side's distance. A side at tol or farther
//     does not move the point.
//   The result is at distance tol up to rounding. The caller calls it only for d < tol. Same bits on CPU and GPU as
//   ComputeDistanceForParticle.
// Units: m, world, at the time of the pose.
// Example: a fixed flat solid with normal {0, 0, 1} through the origin and the point {0.5, 0, tol / 4}: {0.5, 0, tol}.
//   The domain box [0, 1]^3 and the point {tol / 4, 0.5, 0.3 tol}: {tol, 0.5, tol}.
// Requires: solid_index < inputs.solid_count; pose is that solid's pose (ComputeSolidPose).
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <sec-contact-tolerance>
FLUIDITY_CPU_AND_GPU inline Vec3
MoveToRestGap(const SolidContactInputs& inputs, std::uint32_t solid_index, SolidPose pose, Vec3 point) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  const double contact_tolerance = inputs.contact_tolerance;
  switch (solid.shape) {
    case SolidShape::kFlatSolid:
    case SolidShape::kPlate:
    case SolidShape::kBox:
    case SolidShape::kSphere:
    case SolidShape::kCylinder:
      break;
    case SolidShape::kDomainBox: {
      Vec3 moved_point = point;
      // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in FindHitOnDomainBox.
      const DomainSide sides_in_face_order[] = {
          DomainSide::kLeft,
          DomainSide::kRight,
          DomainSide::kFront,
          DomainSide::kBack,
          DomainSide::kBottom,
          DomainSide::kTop,
      };
      for (const DomainSide side : sides_in_face_order) {
        if (!ComputeIsClosedSide(solid.closed, side)) {
          continue;
        }
        const DistanceToSolid at_side = ComputeFaceDistance(pose, moved_point, ToFacePlane(side, solid.half_size));
        if (at_side.distance < contact_tolerance) {
          moved_point = moved_point + (contact_tolerance - at_side.distance) * at_side.normal_out_of_solid;
        }
      }
      return moved_point;
    }
  }
  // An unknown shape stops at ComputeDistanceToSolid's assertion: CheckSolid refuses it.
  const DistanceToSolid at_point = ComputeDistanceForParticle(inputs, solid_index, pose, point, inputs.plate_side);
  return point + (contact_tolerance - at_point.distance) * at_point.normal_out_of_solid;
}

// Checks that a particle committed to a plate does not start more than one contact tolerance beyond that plate's
// other face, and says kOnWrongSideOfPlate when it does (plan decision P3).
//
// Method: only for the plate inputs.plate_side names. With s = +1 for kAlongNormal and -1 for kAgainstNormal and p the
//   start in the plate's own axes at the pose, the distance beyond the other face is -s * p.z - half_thickness:
//   ComputeFaceDistance of the other face's plane (ToFacePlane of the other side), positive on the side the particle
//   must not be on. The start fails when that distance is above tol, s * p.z < -(half_thickness + tol). Within tol of
//   the other face, or inside the thickness, the start is only inside the committed half space, and the start rules
//   push it to the committed side. Every other solid passes. Same bits on CPU and GPU: only + - * and comparisons.
// Units: m, world, at the time of the pose.
// Example: a fixed plate of id 5 through the origin with normal {0, 0, 1} and half thickness 0.01, committed to
//   {5, kAlongNormal}: a start at z = -0.02 fails with magnitude 0.01; z = -0.001 passes.
// Requires: solid_index < inputs.solid_count; pose is that solid's pose; inputs passed CheckSolidContactInputs;
//   failure is not null.
// Fails: kOnWrongSideOfPlate  the start is beyond the other face of its committed plate by more than tol. *failure
//                             names the plate's solid_id, with the distance beyond that face as the magnitude (plan
//                             decision P6). A particle that crossed its plate in an earlier step, or a plate side that
//                             went wrong upstream; the step stops.
//        *failure is written only on failure.
// Read: docs/method-paper/fluidity-method.typ <sec-solid-shapes>, <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline ContactError CheckStartOnCommittedSide(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    SolidPose pose,
    Vec3 start,
    ContactFailure* failure
) {
  assert(solid_index < inputs.solid_count);
  assert(failure != nullptr);
  const Solid& solid = inputs.solids[solid_index];
  const PlateSide plate_side = inputs.plate_side;
  const bool is_committed_to_this_plate = plate_side.plate_id == solid.solid_id;
  if (!is_committed_to_this_plate) {
    return ContactError::kNone;
  }
  assert(solid.shape == SolidShape::kPlate);
  const SolidSide other_side =
      plate_side.side == SolidSide::kAlongNormal ? SolidSide::kAgainstNormal : SolidSide::kAlongNormal;
  // Method paper <sec-solid-shapes>:  beyond the other face by -(s p_z + half_thickness)
  const double distance_beyond_other_face =
      ComputeFaceDistance(pose, start, ToFacePlane(other_side, solid.half_size.z)).distance;  // m
  if (distance_beyond_other_face <= inputs.contact_tolerance) {
    return ContactError::kNone;
  }
  *failure = {
      .error = ContactError::kOnWrongSideOfPlate,
      .close_call = ContactCloseCall::kNone,
      .particle_id = 0,
      .solid_id = solid.solid_id,
      .magnitude = distance_beyond_other_face,
  };
  return ContactError::kOnWrongSideOfPlate;
}

// Computes whether either velocity of a particle's move approaches one solid along a normal, at the particle's
// position and the pose of move.elapsed, by more than the responses so far may have left by rounding.
//
// Method: with u = ComputeSolidVelocity at the position and pose, each velocity's approach speed is
//   Dot(normal_out_of_solid, velocity - u), as ApplyJointResponse records it. A velocity approaches when its approach
//   speed is below minus its own allowance so far: move.this_step_approach_allowance for for_this_step,
//   move.next_step_approach_allowance for for_next_step. Both are 0 before any
//   response, since the velocities are then the inputs, and after one the rounding it may have left along any normal
//   is not an approach. Each velocity needs its own: a response to a for_this_step at rest has an allowance of 0, and
//   compared with it, the -2^-55 m/s that the same response leaves in a for_next_step of {-2, 0, -2} m/s would be an
//   approach to a parallel second solid, a spurious hit. Each allowance is at least the rounding of the solid's spin,
//   kApproachLeftAfterResponseShare * |omega x r|_1 (0 without angular velocity): a flat solid, a plate, a sphere or a
//   cylinder spinning about its normal or axis does not turn, and omega x r has no part along its normal, but computed
//   it has a few rounding units of |omega x r| there, about 1e-14 m/s at 100 m/s, which before any response would be
//   an approach and a spurious hit at 0 on a spinning floor the particle only rests on. Same bits on CPU and GPU when
//   the solid's angular velocity is zero: only + - * and comparisons.
// Units: m/s, world.
// Example: a fixed floor with normal {0, 0, 1}, before any response: velocities {1, 0, -1} and {0, 0, 0} approach (the
//   first does); {1, 0, 2} and {0, 0, 0} do not. After a response with allowance 2^-44 m/s, {0, 0, -2^-50} does not.
//   A floor spinning about its normal at 100 m/s at the particle, before any response: an approach of -1e-14 m/s is
//   rounding, within 2^-46 * 100 m/s, and does not approach.
// Requires: solid_index < inputs.solid_count.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <sec-contact-response>
FLUIDITY_CPU_AND_GPU inline bool ComputeIsApproachingSolid(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    Vec3 normal_out_of_solid,
    const ParticleMove& move
) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  const SolidContactResult& result = move.result;
  const SolidPose pose = ComputeSolidPose(solid, move.elapsed);
  const Vec3 solid_velocity = ComputeSolidVelocity(solid, pose, result.position);
  // The spin's part of the solid velocity, whose rounding along the normal is no approach.
  const Vec3 spin_velocity = math::Cross(solid.motion.angular_velocity, result.position - pose.origin);  // m/s
  const double spin_rounding = kApproachLeftAfterResponseShare
      * (math::Abs(spin_velocity.x) + math::Abs(spin_velocity.y) + math::Abs(spin_velocity.z));  // m/s
  const double this_step_allowance = math::Max(move.this_step_approach_allowance, spin_rounding);  // m/s
  const double next_step_allowance = math::Max(move.next_step_approach_allowance, spin_rounding);  // m/s
  // Method paper <eq:contact-response>:  a = n . (v - u)
  const double next_step_approach = math::Dot(normal_out_of_solid, result.velocity.for_next_step - solid_velocity);
  const double this_step_approach = math::Dot(normal_out_of_solid, result.velocity.for_this_step - solid_velocity);
  return next_step_approach < -next_step_allowance || this_step_approach < -this_step_allowance;
}

// Surfaces of one solid at one point, each with its distance, normal and face (FindSurfacesOfSolid).
struct NearSurfaces {
  // In face order. Six is every side of a domain box; another shape has one.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in ParticleMove.
  DistanceToSolid surfaces[6]{};
  int count = 0;
};

// Finds every surface of one solid at a point: the one surface that gives the solid's distance, or each closed side of
// a domain box. The start rules, the move off near surfaces and the settle read each on its own.
//
// Method: one switch over the shape.
//   flat solid, plate, box, sphere, cylinder  at_point, the surface that gives the distance. Its distance is convex,
//     with the normal a subgradient (plan decision P18), which the move off near surfaces relies on.
//   domain box  every closed side in face order, its distance into the box being ComputeFaceDistance of ToFacePlane: a
//     particle in a corner is near two or three sides, each a flat face of its own, as the sweep treats them
//     (FindHitOnDomainBox). at_point is the nearest of them. Each side's distance is linear, so convex too.
//   The list of a solid is the same at every point, in length and order, so a surface keeps its place in it from one
//   point to the next. Same bits on CPU and GPU: only + - * and comparisons.
// Units: m, world, at the time of the pose.
// Example: the domain box [0, 1]^3 with an open top and the point {tol / 4, 0.5, 0.3 tol}: kLowX at tol / 4, kHighX at
//   1 - tol / 4, kLowY at 0.5, kHighY at 0.5 and kLowZ at 0.3 tol. A fixed floor and the point {0, 0, tol / 2}: kHighZ
//   at tol / 2.
// Requires: solid_index < inputs.solid_count; pose is that solid's pose; at_point is ComputeDistanceForParticle at the
//   point and the pose, with the particle's plate side.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <sec-solid-shapes>
FLUIDITY_CPU_AND_GPU inline NearSurfaces FindSurfacesOfSolid(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    SolidPose pose,
    Vec3 point,
    DistanceToSolid at_point
) {
  assert(solid_index < inputs.solid_count);
  const Solid& solid = inputs.solids[solid_index];
  NearSurfaces surfaces;
  switch (solid.shape) {
    case SolidShape::kFlatSolid:
    case SolidShape::kPlate:
    case SolidShape::kBox:
    case SolidShape::kSphere:
    case SolidShape::kCylinder:
      break;
    case SolidShape::kDomainBox: {
      // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in FindHitOnDomainBox.
      const DomainSide sides_in_face_order[] = {
          DomainSide::kLeft,
          DomainSide::kRight,
          DomainSide::kFront,
          DomainSide::kBack,
          DomainSide::kBottom,
          DomainSide::kTop,
      };
      for (const DomainSide side : sides_in_face_order) {
        if (!ComputeIsClosedSide(solid.closed, side)) {
          continue;
        }
        surfaces.surfaces[surfaces.count] = ComputeFaceDistance(pose, point, ToFacePlane(side, solid.half_size));
        ++surfaces.count;
      }
      return surfaces;
    }
  }
  surfaces.surfaces[0] = at_point;
  surfaces.count = 1;
  return surfaces;
}

// The equalities of one subset of surfaces that SolveSmallestCorrection meets exactly: count unit normals n_k and their
// right sides b_k, for multipliers m with Dot(n_k, sum over j of m_j n_j) = b_k.
struct SubsetEquations {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in ParticleMove.
  Vec3 normals[3]{};  // No unit, unit vectors, world.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in ParticleMove.
  double right_sides[3]{};  // m or m/s.
  int count = 0;  // 1 to 3.
};

// The multipliers of one subset's equalities, or none when the subset cannot be solved.
struct SubsetMultipliers {
  bool is_solved = false;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in ParticleMove.
  double values[3]{};  // m or m/s, in the subset's order.
};

// Solves one subset's equalities for their multipliers, by Cramer's rule on the equations of the normals' dots.
//
// Method (plan decision P19, item 2), with the dots G_ij = Dot(n_i, n_j) and the right sides b:
//   one surface    m = b: the normal is a unit vector, so its own dot is 1 up to rounding.
//   two surfaces   c = Dot(n_a, n_b), m_a = (b_a - c b_b) / (1 - c^2) and m_b = (b_b - c b_a) / (1 - c^2). None when
//                  1 - c^2 <= 0 as computed: parallel or opposite normals, which no pair of equalities can meet.
//   three          m_k = Determinant of G with column k replaced by b, over Determinant(G), each expanded along its
//                  first line (math::Determinant). None when Dot(n_1, Cross(n_2, n_3)) == 0 as computed: three normals
//                  in one plane, where Determinant(G), its square, vanishes.
//   For a velocity, a pair is also not used when 1 + c < kVelocityPairLimitShare, and a triple when one of its pairs is
//   or when |Dot(n_1, Cross(n_2, n_3))| < kVelocityTripleLimit: the refinements could not bring the residual within
//   the velocity's allowance. A position needs no such limit (kVelocityPairLimitShare). Same bits on CPU and GPU:
//   only + - * / and comparisons.
// Units: m or m/s for the right sides and the multipliers.
// Example: n_a {0, 0, 1}, n_b {Sqrt(3) / 2, 0, -1 / 2} (a 60 degree wedge) and b {1, 1}: c = -1 / 2, m = {2, 2}.
// Requires: equations.count is 1 to 3.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline SubsetMultipliers
SolveSubsetMultipliers(const SubsetEquations& equations, CorrectedQuantity quantity) {
  assert(equations.count >= 1 && equations.count <= 3);
  const bool is_velocity = quantity == CorrectedQuantity::kVelocity;
  SubsetMultipliers multipliers;
  if (equations.count == 1) {
    multipliers.is_solved = true;
    multipliers.values[0] = equations.right_sides[0];
    return multipliers;
  }
  // The dots of the normals: one pair, or the three pairs of a triple.
  const Vec3 normal_one = equations.normals[0];
  const Vec3 normal_two = equations.normals[1];
  const double one_two_dot = math::Dot(normal_one, normal_two);
  if (equations.count == 2) {
    const double right_one = equations.right_sides[0];
    const double right_two = equations.right_sides[1];
    // Method paper <proc-move-particle>:  m_a = (b_a - c b_b) / (1 - c^2),  m_b = (b_b - c b_a) / (1 - c^2)
    const double gram_determinant = 1 - one_two_dot * one_two_dot;
    const bool is_below_velocity_limit = is_velocity && 1 + one_two_dot < kVelocityPairLimitShare;
    if (gram_determinant <= 0 || is_below_velocity_limit) {
      return multipliers;
    }
    multipliers.is_solved = true;
    multipliers.values[0] = (right_one - one_two_dot * right_two) / gram_determinant;
    multipliers.values[1] = (right_two - one_two_dot * right_one) / gram_determinant;
    return multipliers;
  }
  const Vec3 normal_three = equations.normals[2];
  const double one_three_dot = math::Dot(normal_one, normal_three);
  const double two_three_dot = math::Dot(normal_two, normal_three);
  const double triple_product = math::Dot(normal_one, math::Cross(normal_two, normal_three));
  const bool has_pair_below_velocity_limit = 1 + one_two_dot < kVelocityPairLimitShare
      || 1 + one_three_dot < kVelocityPairLimitShare || 1 + two_three_dot < kVelocityPairLimitShare;
  const bool is_below_velocity_limit =
      is_velocity && (has_pair_below_velocity_limit || math::Abs(triple_product) < kVelocityTripleLimit);
  if (triple_product == 0 || is_below_velocity_limit) {
    return multipliers;
  }
  // Method paper <proc-move-particle>:  G m = b,  m_k = det(G with column k replaced by b) / det(G)
  const Mat3 dots = {
      .xx = math::Dot(normal_one, normal_one),
      .xy = one_two_dot,
      .xz = one_three_dot,
      .yx = one_two_dot,
      .yy = math::Dot(normal_two, normal_two),
      .yz = two_three_dot,
      .zx = one_three_dot,
      .zy = two_three_dot,
      .zz = math::Dot(normal_three, normal_three),
  };
  const double dots_determinant = math::Determinant(dots);
  const double right_x = equations.right_sides[0];
  const double right_y = equations.right_sides[1];
  const double right_z = equations.right_sides[2];
  Mat3 with_right_sides = dots;
  with_right_sides.xx = right_x;
  with_right_sides.yx = right_y;
  with_right_sides.zx = right_z;
  multipliers.values[0] = math::Determinant(with_right_sides) / dots_determinant;
  with_right_sides = dots;
  with_right_sides.xy = right_x;
  with_right_sides.yy = right_y;
  with_right_sides.zy = right_z;
  multipliers.values[1] = math::Determinant(with_right_sides) / dots_determinant;
  with_right_sides = dots;
  with_right_sides.xz = right_x;
  with_right_sides.yz = right_y;
  with_right_sides.zz = right_z;
  multipliers.values[2] = math::Determinant(with_right_sides) / dots_determinant;
  multipliers.is_solved = true;
  return multipliers;
}

// Computes the change of one subset: the sum over k of multipliers[k] * normals[k], added in the subset's order.
//
// Example: normals {0, 0, 1} and {Sqrt(3) / 2, 0, -1 / 2}, multipliers {2, 2}: {Sqrt(3), 0, 1}.
// Requires: equations.count is 1 to 3.
FLUIDITY_CPU_AND_GPU inline Vec3 ComputeSubsetChange(const SubsetEquations& equations, SubsetMultipliers multipliers) {
  assert(equations.count >= 1 && equations.count <= 3);
  Vec3 change{};
  for (int surface_index = 0; surface_index < equations.count; ++surface_index) {
    change = change + multipliers.values[surface_index] * equations.normals[surface_index];
  }
  return change;
}

// One subset of up to three surfaces, by index into a CorrectionTargets, ascending.
struct CorrectionSubset {
  int count = 0;  // 1 to 3.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in ParticleMove.
  int surfaces[3]{};
};

// Computes how much a candidate change may miss each target by: the slack asked for, and for a velocity at least the
// change's own allowance.
//
// Method: a position gets slack. A velocity gets the larger of slack and kApproachLeftAfterResponseShare *
//   (targets.speeds_before_change + |w.x| + |w.y| + |w.z|), the bound of ComputeLargestApproachLeftAfterResponse
//   counted over every speed the change involves, its own size included: a change can be far faster than the
//   velocity it changes (a wedge closing on a particle), and the rounding of composing it grows with it. The caller
//   records that allowance, so whatever approach a candidate leaves within it is covered by the touched rule. Same bits
//   on CPU and GPU: only + * Abs and comparisons.
// Units: m or m/s.
// Example: a velocity, slack 0, speeds_before_change 3 m/s and the change {1, 0, -1} m/s: 5 * 2^-46 m/s.
FLUIDITY_CPU_AND_GPU inline double
ComputeCorrectionSlack(const CorrectionTargets& targets, double slack, CorrectedQuantity quantity, Vec3 change) {
  if (quantity == CorrectedQuantity::kPosition) {
    return slack;
  }
  const double change_axes_sum = math::Abs(change.x) + math::Abs(change.y) + math::Abs(change.z);  // m/s
  // Method paper <proc-move-particle>:  b = 2^-46 (|v|_1 + |w|_1 + sum |u_i|_1)
  const double change_allowance = kApproachLeftAfterResponseShare * (targets.speeds_before_change + change_axes_sum);
  return math::Max(slack, change_allowance);
}

// Solves one subset of surfaces as equalities for the change, solved again on its residual for a velocity, or says that
// it has none.
//
// Method: SolveSubsetMultipliers on the subset's normals and targets; none when it cannot be solved, or when a
//   multiplier is negative or not finite: a negative one would pull the change into a surface, which the smallest
//   change of another subset then does better. The change is ComputeSubsetChange. For a velocity, while some residual
//   b_k - Dot(n_k, change) exceeds a quarter of the candidate's slack (ComputeCorrectionSlack) in size and fewer than
//   kMaxCorrectionRefinements solves were added, the subset is solved again with the residuals as right sides and the
//   result added to the change and the multipliers.
//   The candidate is not yet checked against the surfaces outside the subset (SolveSmallestCorrection does).
// Units: m or m/s.
// Example: the 60 degree wedge of SolveSubsetMultipliers, targets {1, 1}: the change {Sqrt(3), 0, 1}, multipliers
//   {2, 2}. With targets {1, -1} the subset of both surfaces gives m_b < 0 and no candidate.
// Requires: subset indexes surfaces below targets.count; slack is zero or more.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline SmallestCorrection SolveCorrectionOfSubset(
    const CorrectionTargets& targets,
    CorrectionSubset subset,
    double slack,
    CorrectedQuantity quantity
) {
  assert(subset.count >= 1 && subset.count <= 3);
  assert(slack >= 0);
  SubsetEquations equations;
  equations.count = subset.count;
  for (int member_index = 0; member_index < subset.count; ++member_index) {
    const int surface_index = subset.surfaces[member_index];
    assert(surface_index >= 0 && surface_index < targets.count);
    equations.normals[member_index] = targets.normals[surface_index];
    equations.right_sides[member_index] = targets.targets[surface_index];
  }
  SmallestCorrection candidate;
  SubsetMultipliers multipliers = SolveSubsetMultipliers(equations, quantity);
  if (!multipliers.is_solved) {
    return candidate;
  }
  for (int member_index = 0; member_index < subset.count; ++member_index) {
    const double multiplier = multipliers.values[member_index];
    // A NaN fails the first test: it is not zero or more.
    if (!(multiplier >= 0) || !math::IsFinite(multiplier)) {
      return candidate;
    }
  }
  Vec3 change = ComputeSubsetChange(equations, multipliers);
  // A velocity's residual must fit its allowance; a position's next iterate corrects it.
  int refinement_count = 0;
  const bool solves_residual = quantity == CorrectedQuantity::kVelocity;
  while (solves_residual && refinement_count < kMaxCorrectionRefinements) {
    SubsetEquations residual_equations = equations;
    const double candidate_slack = ComputeCorrectionSlack(targets, slack, quantity, change);  // m/s
    bool is_within_slack = true;
    for (int member_index = 0; member_index < subset.count; ++member_index) {
      const double residual = equations.right_sides[member_index] - math::Dot(equations.normals[member_index], change);
      residual_equations.right_sides[member_index] = residual;
      is_within_slack = is_within_slack && math::Abs(residual) <= candidate_slack / 4;
    }
    if (is_within_slack) {
      break;
    }
    // The same normals as the first solve, which passed the limits, so this solve succeeds.
    const SubsetMultipliers refinement = SolveSubsetMultipliers(residual_equations, quantity);
    change = change + ComputeSubsetChange(residual_equations, refinement);
    for (int member_index = 0; member_index < subset.count; ++member_index) {
      multipliers.values[member_index] = multipliers.values[member_index] + refinement.values[member_index];
    }
    ++refinement_count;
  }
  candidate.is_found = true;
  candidate.change = change;
  candidate.used_count = subset.count;
  for (int member_index = 0; member_index < subset.count; ++member_index) {
    candidate.used_surfaces[member_index] = subset.surfaces[member_index];
    candidate.multipliers[member_index] = multipliers.values[member_index];
  }
  candidate.refinement_count = refinement_count;
  return candidate;
}

// Computes whether a candidate change meets every target within the slack: Dot(n_i, change) >= target_i - slack, less
// for a position the rounding of composing the change, 2^-49 times the sum of the sizes of its multipliers, and for a
// velocity within at least the change's own allowance (ComputeCorrectionSlack).
//
// Method: composing sum m_k n_k rounds each axis by about one rounding unit of each term, so the dot with a unit normal
//   moves by at most about 8 rounding units of sum |m_k|: 2^-49 = 8 * 2^-52. In a cusp the multipliers are far larger
//   than the change (they grow as 1 / (1 + c)), and without this share the exact solution of a pair would miss its own
//   targets by their rounding. Nothing bounds this share: for a pair whose normals are nearly opposite, or a triple
//   whose normals are nearly in one plane, the multipliers are huge, and a candidate may miss a target by many times
//   the slack (a fuzz of 60000 cases found misses of up to 527 slacks). So an accepted position is not promised to meet
//   its targets; the next iterate corrects it, and MoveToRestGapOfNearSurfaces keeps a point only after reading the
//   true distances. A velocity adds no such share: its slack grows with the change instead, and the joint response
//   records it as the change's allowance. A NaN dot fails.
// Example: targets {1, 1} on the 60 degree wedge and the change {Sqrt(3), 0, 1}: both dots are 1 up to rounding,
//   within any slack of a few rounding units.
// Requires: candidate.is_found.
FLUIDITY_CPU_AND_GPU inline bool ComputeIsWithinTargets(
    const CorrectionTargets& targets,
    const SmallestCorrection& candidate,
    double slack,
    CorrectedQuantity quantity
) {
  assert(candidate.is_found);
  double multiplier_sizes = 0;
  for (int member_index = 0; member_index < candidate.used_count; ++member_index) {
    multiplier_sizes = multiplier_sizes + math::Abs(candidate.multipliers[member_index]);
  }
  const double composing_slack = quantity == CorrectedQuantity::kPosition ? 0x1p-49 * multiplier_sizes : 0;
  const double candidate_slack = ComputeCorrectionSlack(targets, slack, quantity, candidate.change);
  for (int surface_index = 0; surface_index < targets.count; ++surface_index) {
    const double along_normal = math::Dot(targets.normals[surface_index], candidate.change);
    const double least_along_normal = targets.targets[surface_index] - candidate_slack - composing_slack;
    if (!(along_normal >= least_along_normal)) {
      return false;
    }
  }
  return true;
}

// The most subsets of one, two or three among kMaxSlideDirections surfaces: n + n (n - 1) / 2 + n (n - 1) (n - 2) / 6,
// 8 + 28 + 56 = 92 at n = 8.
inline constexpr int kMaxCorrectionSubsets = kMaxSlideDirections + kMaxSlideDirections * (kMaxSlideDirections - 1) / 2
    + kMaxSlideDirections * (kMaxSlideDirections - 1) * (kMaxSlideDirections - 2) / 6;
static_assert(kMaxCorrectionSubsets == 92, "the documentation counts 92 subsets of at most three among eight");

// Computes the subset that SolveSmallestCorrection tries after subset among surface_count surfaces: the next one of the
// same size in ascending order of its indices, then the first of the next size, or a subset of count 0 after the last
// triple (or after the last subset when there are fewer than three surfaces).
//
// Example: among four surfaces, {1, 3} is followed by {2, 3}, {2, 3} by {0, 1, 2}, and {1, 2, 3} by none.
// Requires: subset.count is 1 to 3 and its indices ascend below surface_count.
FLUIDITY_CPU_AND_GPU constexpr CorrectionSubset ComputeNextSubset(CorrectionSubset subset, int surface_count) {
  const int count = subset.count;
  // The last member that can still move up: member k may reach surface_count - count + k.
  int member_index = count - 1;
  while (member_index >= 0 && subset.surfaces[member_index] == surface_count - count + member_index) {
    --member_index;
  }
  CorrectionSubset next = subset;
  if (member_index >= 0) {
    ++next.surfaces[member_index];
    for (int later_index = member_index + 1; later_index < count; ++later_index) {
      next.surfaces[later_index] = next.surfaces[later_index - 1] + 1;
    }
    return next;
  }
  const int next_count = count + 1;
  if (next_count > 3 || next_count > surface_count) {
    return {.count = 0, .surfaces = {0, 0, 0}};
  }
  return {.count = next_count, .surfaces = {0, 1, 2}};
}

// Computes whether a velocity approaches no surface of targets by more than the slack: every target is at most slack.
// A NaN target approaches.
FLUIDITY_CPU_AND_GPU inline bool ComputeIsWithinSlack(const CorrectionTargets& targets, double slack) {
  for (int direction_index = 0; direction_index < targets.count; ++direction_index) {
    if (!(targets.targets[direction_index] <= slack)) {
      return false;
    }
  }
  return true;
}

// Finds the candidate SolveSmallestCorrection tries first: among the subsets of one, two or three surfaces whose
// change meets every target within the slack (SolveCorrectionOfSubset, ComputeIsWithinTargets), the one with the
// smallest Length(change), the first in subset order on a tie; none found when no subset meets them.
//
// Requires: as SolveSmallestCorrection.
FLUIDITY_CPU_AND_GPU inline SmallestCorrection
FindShortestCorrection(const CorrectionTargets& targets, double slack, CorrectedQuantity quantity) {
  SmallestCorrection smallest;
  double smallest_length = 0;  // m or m/s
  CorrectionSubset subset = {.count = 1, .surfaces = {0, 0, 0}};
  for (int subset_index = 0; subset_index < kMaxCorrectionSubsets && subset.count != 0; ++subset_index) {
    const CorrectionSubset tried = subset;
    subset = ComputeNextSubset(subset, targets.count);
    const SmallestCorrection candidate = SolveCorrectionOfSubset(targets, tried, slack, quantity);
    if (!candidate.is_found || !ComputeIsWithinTargets(targets, candidate, slack, quantity)) {
      continue;
    }
    const double candidate_length = math::Length(candidate.change);
    if (!smallest.is_found || candidate_length < smallest_length) {
      smallest = candidate;
      smallest_length = candidate_length;
    }
  }
  return smallest;
}

// Finds, for a velocity, the candidate with the largest sum of friction limits among those no longer than the
// shortest plus its slack: sum over k of targets.frictions[used_surfaces[k]] * multipliers[k], the first in subset
// order on equal sums (plan decision P20, item 3).
//
// Method: a second pass over the same subsets as FindShortestCorrection, solved again, so no list of candidates is
//   kept. The shortest itself qualifies, so one is always found.
// Requires: shortest is FindShortestCorrection of these targets and slack for a velocity, and was found.
FLUIDITY_CPU_AND_GPU inline SmallestCorrection
FindCorrectionOfLargestFrictionSum(const CorrectionTargets& targets, double slack, const SmallestCorrection& shortest) {
  assert(shortest.is_found);
  const CorrectedQuantity quantity = CorrectedQuantity::kVelocity;
  const double length_limit =
      math::Length(shortest.change) + ComputeCorrectionSlack(targets, slack, quantity, shortest.change);
  SmallestCorrection kept;
  double kept_friction_sum = 0;  // m/s
  CorrectionSubset subset = {.count = 1, .surfaces = {0, 0, 0}};
  for (int subset_index = 0; subset_index < kMaxCorrectionSubsets && subset.count != 0; ++subset_index) {
    const CorrectionSubset tried = subset;
    subset = ComputeNextSubset(subset, targets.count);
    const SmallestCorrection candidate = SolveCorrectionOfSubset(targets, tried, slack, quantity);
    const bool is_admissible = candidate.is_found && ComputeIsWithinTargets(targets, candidate, slack, quantity)
        && math::Length(candidate.change) <= length_limit;
    if (!is_admissible) {
      continue;
    }
    // Method paper <proc-move-particle>:  sum over k of mu_k lambda_k
    double friction_sum = 0;  // m/s
    for (int member_index = 0; member_index < candidate.used_count; ++member_index) {
      friction_sum =
          friction_sum + targets.frictions[candidate.used_surfaces[member_index]] * candidate.multipliers[member_index];
    }
    if (!kept.is_found || friction_sum > kept_friction_sum) {
      kept = candidate;
      kept_friction_sum = friction_sum;
    }
  }
  return kept;
}

// Solves for the smallest change w with Dot(n_i, w) >= target_i for every surface, up to the slack: the move of a
// point off several near surfaces at once, or the frictionless change of a velocity that approaches several.
//
// Method (plan decision P19, item 2): the problem is convex, so its exact answer meets some subset of at most three
//   surfaces as equalities (three independent normals fix a point in space) and every other surface as an inequality.
//   So the solver tries each single surface, each pair and each triple, in that order, each in ascending indices
//   (ComputeNextSubset: 4 + 6 + 4 = 14 subsets of four surfaces, at most kMaxCorrectionSubsets = 92 of eight), as
//   SolveCorrectionOfSubset solves them; it keeps a candidate that meets every target within the slack
//   (ComputeIsWithinTargets) and has the smallest Length(change), the first in that order on a tie. A position has no
//   empty subset: the move off near surfaces calls it only when a surface is short of the rest gap. Fixed loops, no
//   allocation, only + - * /, Abs and Sqrt: the same bits on CPU and GPU.
//   A velocity differs in two ways (plan decision P20, item 3).
//   No change: when every target is at most the slack passed, the answer is no change, used_count 0, so a velocity
//   that approaches no surface beyond its allowance is never changed by rounding-sized targets. The slack passed, not
//   the candidate's own 2^-46 (speeds): beside a floor sliding at 100 m/s, a slow real approach of 1e-13 m/s to an
//   upright is removed, as one surface's response removes it.
//   Ties: where normals depend on each other (three faces sharing one line, four through a point, coincident faces) the
//   same smallest change has several sets of multipliers, and the friction, which reads them, would differ by which
//   one rounding picked (15,059 of 20,000 coplanar triples jumped, by up to 0.70 of the speed, for a 1e-13 change of
//   the velocity). So a second pass over the same subsets keeps, among the candidates no longer than the shortest plus
//   its slack (ComputeCorrectionSlack), the one with the largest sum of frictions[k] * multipliers[k], the first in
//   subset order on equal sums: the largest value of a linear function over the admissible multipliers, which varies
//   continuously with the velocity (0 jumps over the same 20,000). Two passes and not a kept list: 92 candidates
//   would not fit a device stack cheaply. Cost: each subset is solved twice, at most 2 * kMaxCorrectionSubsets = 184
//   solves of one velocity.
//   Example of why a pair: from a start on both faces of a 60 degree wedge, moving the floor's shortfall along its
//   normal leaves the other face short by half of it; the pair meets both at once.
// Units: m for a position, m/s for a velocity: the targets, the slack and the change.
// Ties: a position keeps the first of equal lengths in the order above; a velocity, the second pass above.
// Example: slack 2^-40. Normals {0, 0, 1} and {Sqrt(3) / 2, 0, -1 / 2} (a 60 degree wedge), targets {1, 1}: change
//   {Sqrt(3), 0, 1}, multipliers {2, 2}; targets {1, 0}: change {1 / Sqrt(3), 0, 1}, multipliers {4 / 3, 2 / 3}.
//   Normals {0, 0, 1} and {1, 0, 0}, targets {1, -1}: the floor alone, {0, 0, 1}. The three axes, targets {1, 1, 1}:
//   {1, 1, 1}.
//   A velocity {1, 0, -1} on the floor alone, target 1, slack 2^-46: change {0, 0, 1}; target 2^-50: no change, used
//   count 0.
// Requires: targets.count is 1 to kMaxConstrainingSurfaces for a position and 1 to kMaxSlideDirections for a velocity,
//   each normal a unit vector; slack is zero or more.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline SmallestCorrection
SolveSmallestCorrection(const CorrectionTargets& targets, double slack, CorrectedQuantity quantity) {
  assert(targets.count >= 1 && targets.count <= kMaxSlideDirections);
  assert(quantity == CorrectedQuantity::kVelocity || targets.count <= kMaxConstrainingSurfaces);
  assert(slack >= 0);
  for (int surface_index = 0; surface_index < targets.count; ++surface_index) {
    assert(math::Abs(math::Dot(targets.normals[surface_index], targets.normals[surface_index]) - 1) <= 0x1p-40);
  }
  const bool is_velocity = quantity == CorrectedQuantity::kVelocity;
  if (is_velocity && ComputeIsWithinSlack(targets, slack)) {
    SmallestCorrection no_change;
    no_change.is_found = true;
    return no_change;
  }
  const SmallestCorrection shortest = FindShortestCorrection(targets, slack, quantity);
  if (!is_velocity || !shortest.is_found) {
    return shortest;
  }
  return FindCorrectionOfLargestFrictionSum(targets, slack, shortest);
}

// Computes whether every solid is at least least_distance from a point, at its pose pose_elapsed seconds into the step,
// by ComputeDistanceForParticle: for a domain box that is every closed side, since its distance is the nearest side's.
//
// Method: a NaN distance is not at least anything, so it gives false. Same bits on CPU and GPU when every angular
//   velocity is zero.
// Units: m, world; s for pose_elapsed.
// Example: a fixed floor through the origin and the point {0, 0, tol}: true for least_distance tol, false for 2 tol.
// Requires: inputs.solids points to inputs.solid_count solids; pose_elapsed is within the step.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline bool ComputeIsAtLeastFromEverySolid(
    const SolidContactInputs& inputs,
    double pose_elapsed,
    Vec3 point,
    double least_distance
) {
  assert(inputs.solids != nullptr || inputs.solid_count == 0);
  assert(pose_elapsed >= 0 && pose_elapsed <= inputs.step_duration);
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const SolidPose pose = ComputeSolidPose(inputs.solids[solid_index], pose_elapsed);
    const double distance = ComputeDistanceForParticle(inputs, solid_index, pose, point, inputs.plate_side).distance;
    if (!(distance >= least_distance)) {
      return false;
    }
  }
  return true;
}

// Builds a copy of one solid that keeps one surface of it fixed at one pose, so that the closed forms of the sweep
// (FindFirstHitOnSolid) can test a straight move of a point against that surface as it stands.
//
// Method: the copy's pose at the start of the step is pose and it has no motion, so at every time of a sweep it stands
//   where the solid stood at pose; without motion it is never a turning solid, so it always has its closed form. One
//   switch over the shape:
//   flat solid, plate, box, sphere, cylinder  the whole solid: it has one surface.
//   domain box  only the closed side whose face is surface_face stays closed, so the copy is that one side.
// Example: the domain box [0, 1]^3 and kLowZ: a domain box closed only at side_bottom.
// Requires: surface_face is the face of one of the solid's surfaces (FindSurfacesOfSolid).
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline Solid BuildFixedCopyOfSurface(const Solid& solid, SolidPose pose, SolidFace surface_face) {
  Solid fixed_copy = solid;
  fixed_copy.pose_at_step_start = pose;
  fixed_copy.motion = SolidMotion{};
  switch (solid.shape) {
    case SolidShape::kFlatSolid:
    case SolidShape::kPlate:
    case SolidShape::kBox:
    case SolidShape::kSphere:
    case SolidShape::kCylinder:
      break;
    case SolidShape::kDomainBox:
      fixed_copy.closed = {
          .left = surface_face == SolidFace::kLowX,
          .right = surface_face == SolidFace::kHighX,
          .front = surface_face == SolidFace::kLowY,
          .back = surface_face == SolidFace::kHighY,
          .bottom = surface_face == SolidFace::kLowZ,
          .top = surface_face == SolidFace::kHighZ,
      };
      break;
  }
  // The copy keeps one of the solid's own closed sides, so it is a domain box that stops particles.
  assert(!fixed_copy.closed.left || solid.closed.left);
  assert(!fixed_copy.closed.right || solid.closed.right);
  assert(!fixed_copy.closed.front || solid.closed.front);
  assert(!fixed_copy.closed.back || solid.closed.back);
  assert(!fixed_copy.closed.bottom || solid.closed.bottom);
  assert(!fixed_copy.closed.top || solid.closed.top);
  return fixed_copy;
}

// Finds where a straight move of a point first comes within grown_by of one surface of a solid, with the solid fixed at
// one pose: the crossing check of a move off near surfaces.
//
// Method: FindFirstHitOnSolid of move, a sweep of start + change * t for 0 <= t <= 1 s at the time 0 of the fixed copy
//   (BuildFixedCopyOfSurface), not touched, and grown_by as the contact tolerance: the same closed forms,
//   touching rule included, as every sweep, against the surface grown by grown_by. The hit's elapsed time is the share
//   of the move at which it first comes within grown_by. A move longer than the thinnest solid could otherwise pass
//   through one between two points that are both outside it.
// Units: m, world; the sweep's velocity is the move per second, so the hit time is a share of the move.
// Example: a plate across the start of a sharp wedge, its near face 100 tol out, a move of 3820 tol through it and a
//   surface grown by tol: a hit at about 99 / 3820 of the move.
// Requires: solid_index < inputs.solid_count; pose is that solid's pose; surface_face is the face of one of its
//   surfaces; move.start_elapsed is 0 and move.duration is 1; grown_by is positive.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline HitTime FindCrossingOfMove(
    const SolidContactInputs& inputs,
    std::uint32_t solid_index,
    SolidPose pose,
    SolidFace surface_face,
    Sweep move,
    double grown_by
) {
  assert(solid_index < inputs.solid_count);
  assert(move.start_elapsed == 0 && move.duration == 1);
  assert(grown_by > 0);
  const Solid fixed_copy = BuildFixedCopyOfSurface(inputs.solids[solid_index], pose, surface_face);
  SolidContactInputs fixed_inputs = inputs;
  fixed_inputs.solids = &fixed_copy;
  fixed_inputs.solid_count = 1;
  fixed_inputs.contact_tolerance = grown_by;
  FirstHit hit;
  const TouchedSolid not_touched;
  const ContactError error = FindFirstHitOnSolid(
      fixed_inputs,
      0,
      move.start,
      move.velocity_for_this_step,
      move.start_elapsed,
      move.duration,
      not_touched,
      &hit
  );
  // A fixed copy never turns, so it has a closed form; only an unknown shape fails, which CheckSolid stops first.
  assert(error == ContactError::kNone);
  if (error != ContactError::kNone || hit.solid_index == kNoSolid) {
    return HitTime{};
  }
  return {.has_hit = true, .elapsed = hit.elapsed};
}

// The most surfaces one move off near surfaces keeps track of: those near an iterate, those that constrained the
// iterate before, and those a move crossed. Surfaces with the same normal share one of the kMaxConstrainingSurfaces
// targets, so there can be more surfaces than targets (five stacked floors and an upright: six surfaces, two
// targets); sixteen is far more than any scene that has a move.
inline constexpr int kMaxTrackedSurfaces = 4 * kMaxConstrainingSurfaces;

// One surface that constrains a move off near surfaces: its solid, and its place in FindSurfacesOfSolid's list.
struct ConstrainingSurface {
  std::uint32_t solid_index = kNoSolid;  // Index into solids, not the solid_id.
  int surface_index = 0;  // Index into the solid's list of FindSurfacesOfSolid, the same at every point.
};

// The surfaces that constrain a move off near surfaces: those near the iterate or constraining the iterate before,
// then those the move crossed.
struct ConstrainingSurfaces {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in ParticleMove.
  ConstrainingSurface surfaces[kMaxTrackedSurfaces]{};
  int count = 0;
  // More than kMaxTrackedSurfaces surfaces, or more than kMaxConstrainingSurfaces normals that differ, would have
  // constrained the move: no move is found.
  bool has_too_many = false;
  // The solid latest in array order of every surface that constrained the move so far, kNoSolid before any.
  std::uint32_t latest_solid_index = kNoSolid;
};

// Computes whether one surface of one solid is among the surfaces that constrain a move.
//
// Example: a list holding {solid 1, surface 0}: true for solid 1, surface 0; false for solid 1, surface 1.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsConstraining(const ConstrainingSurfaces& constraining, std::uint32_t solid_index, int surface_index) {
  for (int listed_index = 0; listed_index < constraining.count; ++listed_index) {
    const ConstrainingSurface listed = constraining.surfaces[listed_index];
    if (listed.solid_index == solid_index && listed.surface_index == surface_index) {
      return true;
    }
  }
  return false;
}

// Applies one more surface to the surfaces that constrain a move: it joins the list and may become the latest solid,
// or, beyond kMaxTrackedSurfaces, marks the list as having too many.
//
// Example: an empty list and {solid 2, surface 0}: count 1, latest solid 2.
// Requires: constraining is not null.
FLUIDITY_CPU_AND_GPU inline void
ApplyConstrainingSurface(ConstrainingSurfaces* constraining, ConstrainingSurface surface) {
  assert(constraining != nullptr);
  const bool is_later_solid =
      constraining->latest_solid_index == kNoSolid || surface.solid_index > constraining->latest_solid_index;
  if (is_later_solid) {
    constraining->latest_solid_index = surface.solid_index;
  }
  if (constraining->count == kMaxTrackedSurfaces) {
    constraining->has_too_many = true;
    return;
  }
  constraining->surfaces[constraining->count] = surface;
  ++constraining->count;
}

// Computes whether two unit normals are the same up to rounding: every axis within kSameNormalAxisDifference.
//
// Example: {0, 0, 1} and {0, 0, 1}: true; {0, 0, 1} and {2^-30, 0, 1}: false.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsSameNormal(Vec3 first_normal, Vec3 second_normal) {
  const Vec3 difference = first_normal - second_normal;
  return math::Abs(difference.x) <= kSameNormalAxisDifference && math::Abs(difference.y) <= kSameNormalAxisDifference
      && math::Abs(difference.z) <= kSameNormalAxisDifference;
}

// Applies one surface's target to the targets of a move off near surfaces: a surface with the same normal as one
// already there raises that target to the larger of the two; another normal adds a target. Returns false when it
// would be the fifth normal that differs, and then changes nothing.
//
// Method: for one normal n, Dot(n, w) >= max(t_1, t_2) meets both targets, so two surfaces with the same normal need
//   one target: the more demanding. Normals equal up to kSameNormalAxisDifference on every axis share it, missing the
//   other surface's target by at most Sqrt(3) 2^-40 |w|, below 2^-17 tol within the move cap
//   (kSameNormalAxisDifference). Coincident or stacked surfaces (a ground lying on the domain's bottom, floors stacked
//   a fraction of tol apart) so count once toward kMaxConstrainingSurfaces.
// Example: targets {n {0, 0, 1}, t 0.5 tol} and a surface {n {0, 0, 1}, t 0.75 tol}: {n {0, 0, 1}, t 0.75 tol}.
// Requires: targets is not null.
FLUIDITY_CPU_AND_GPU inline bool ApplyRestGapTarget(CorrectionTargets* targets, Vec3 normal, double target) {
  assert(targets != nullptr);
  for (int target_index = 0; target_index < targets->count; ++target_index) {
    if (ComputeIsSameNormal(targets->normals[target_index], normal)) {
      targets->targets[target_index] = math::Max(targets->targets[target_index], target);
      return true;
    }
  }
  if (targets->count == kMaxConstrainingSurfaces) {
    return false;
  }
  targets->normals[targets->count] = normal;
  targets->targets[targets->count] = target;
  ++targets->count;
  return true;
}

// Computes how far one surface is grown when a move off near surfaces is swept against it, or 0 when it is not swept.
//
// Method: a surface near the start point (d < tol) is not swept: the move leaves it. A surface that does not yet
//   constrain the move is grown by tol: the move must not bring the point within tol of it unseen. A surface that
//   already constrains the move, but was not near the start point, is grown by tol / 2: its target is a linearisation
//   at an earlier iterate, whose half space need not hold the start point, so a straight move could cut a corner of a
//   convex solid although both its ends are outside it. A move that ends at the rest gap of such a surface stays
//   farther than tol / 2 from it, so this sweep finds only a move that comes nearer on the way.
// Units: m.
// Example: tol 2^-30, a surface 3 tol from the start point that does not constrain the move: 2^-30. The same surface
//   among the constraining ones: 2^-31. A surface tol / 2 from the start point: 0.
FLUIDITY_CPU_AND_GPU constexpr double ComputeSweptGrowth(
    const SolidContactInputs& inputs,
    const ConstrainingSurfaces& constraining,
    ConstrainingSurface surface,
    double start_distance
) {
  const double contact_tolerance = inputs.contact_tolerance;
  if (start_distance < contact_tolerance) {
    return 0;
  }
  const bool is_constraining = ComputeIsConstraining(constraining, surface.solid_index, surface.surface_index);
  return is_constraining ? contact_tolerance / 2 : contact_tolerance;
}

// Finds the share of a move off near surfaces at which it first comes within the swept growth of a surface
// (ComputeSweptGrowth, FindCrossingOfMove), over every surface of every solid, read at the start point.
//
// Units: m, world; s for pose_elapsed; the hit time is a share of the move, 0 to 1.
// Example: the floor and a 0.03 degree side constrain the move 3820 tol out along their bisector, and a plate 100 tol
//   out lies across it: a hit at about 0.026 of the move.
// Requires: pose_elapsed is within the step.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline HitTime FindEarliestCrossing(
    const SolidContactInputs& inputs,
    double pose_elapsed,
    Sweep move,
    const ConstrainingSurfaces& constraining
) {
  HitTime earliest;
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const SolidPose pose = ComputeSolidPose(inputs.solids[solid_index], pose_elapsed);
    const DistanceToSolid at_start =
        ComputeDistanceForParticle(inputs, solid_index, pose, move.start, inputs.plate_side);
    const NearSurfaces surfaces = FindSurfacesOfSolid(inputs, solid_index, pose, move.start, at_start);
    for (int surface_index = 0; surface_index < surfaces.count; ++surface_index) {
      const DistanceToSolid surface = surfaces.surfaces[surface_index];
      const ConstrainingSurface listed = {.solid_index = solid_index, .surface_index = surface_index};
      const double grown_by = ComputeSweptGrowth(inputs, constraining, listed, surface.distance);  // m
      if (grown_by == 0) {
        continue;
      }
      const HitTime crossing = FindCrossingOfMove(inputs, solid_index, pose, surface.face, move, grown_by);
      const bool is_earlier = crossing.has_hit && (!earliest.has_hit || crossing.elapsed < earliest.elapsed);
      earliest = is_earlier ? crossing : earliest;
    }
  }
  return earliest;
}

// Applies the crossing check of one move off near surfaces: finds the share of the move at which it first crosses a
// surface (FindEarliestCrossing), and the surfaces not yet constraining that it crosses at that share join the
// constraining surfaces.
//
// Method: every solid in array order, every surface in its list at the start point (FindSurfacesOfSolid), each swept
//   against the surface grown as ComputeSweptGrowth says. Nothing skips a solid: a thin solid is caught however far it
//   is from both ends of the move. The caller shortens the move to the returned share.
// Units: m, world; s for pose_elapsed; the hit time is a share of the move, 0 to 1.
// Example: the floor and a 0.03 degree side constrain the move 3820 tol out along their bisector, and a plate 100 tol
//   out lies across it: a hit at about 0.026 of the move, and the plate joins.
// Requires: constraining is not null; pose_elapsed is within the step.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline HitTime ApplyCrossedSurfaces(
    const SolidContactInputs& inputs,
    double pose_elapsed,
    Vec3 start_point,
    Vec3 change,
    ConstrainingSurfaces* constraining
) {
  assert(constraining != nullptr);
  const Sweep move = {.start = start_point, .velocity_for_this_step = change, .start_elapsed = 0, .duration = 1};
  const HitTime earliest = FindEarliestCrossing(inputs, pose_elapsed, move, *constraining);
  if (!earliest.has_hit) {
    return earliest;
  }
  const ConstrainingSurfaces before = *constraining;
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const SolidPose pose = ComputeSolidPose(inputs.solids[solid_index], pose_elapsed);
    const DistanceToSolid at_start =
        ComputeDistanceForParticle(inputs, solid_index, pose, start_point, inputs.plate_side);
    const NearSurfaces surfaces = FindSurfacesOfSolid(inputs, solid_index, pose, start_point, at_start);
    for (int surface_index = 0; surface_index < surfaces.count; ++surface_index) {
      const DistanceToSolid surface = surfaces.surfaces[surface_index];
      const bool is_constraining = ComputeIsConstraining(before, solid_index, surface_index);
      if (is_constraining || surface.distance < inputs.contact_tolerance) {
        continue;
      }
      const HitTime crossing =
          FindCrossingOfMove(inputs, solid_index, pose, surface.face, move, inputs.contact_tolerance);
      if (crossing.has_hit && crossing.elapsed <= earliest.elapsed) {
        ApplyConstrainingSurface(constraining, {.solid_index = solid_index, .surface_index = surface_index});
      }
    }
  }
  return earliest;
}

// The targets of one iterate of a move off near surfaces, and the surfaces they come from.
struct RestGapTargets {
  CorrectionTargets targets{};  // One per normal that differs among the constraining surfaces (ApplyRestGapTarget).
  ConstrainingSurfaces surfaces{};
};

// Builds the targets of one iterate of the move off near surfaces: each constraining surface's shortfall at the
// iterate, written as a least change from the start point along its normal there.
//
// Method, with p0 the start point, x the iterate and tol the contact tolerance: every solid in array order, every
//   surface in its list (FindSurfacesOfSolid), with its distance d and normal n at x. A surface constrains this
//   iterate when d < tol (near now) or when it constrained the one before (else the move could go back into it). Its
//   target is the linearisation d + Dot(n, y - x) >= tol for the next iterate y, written for w = y - p0:
//   Dot(n, w) >= tol - d + Dot(n, x - p0); surfaces with the same normal share one target, the largest
//   (ApplyRestGapTarget). A surface that is neither, and that the next move comes within tol of, is caught by that
//   move's crossing check (ApplyCrossedSurfaces). A fifth normal that differs, or more than kMaxTrackedSurfaces
//   surfaces, marks surfaces.has_too_many; the scan goes on so that surfaces.latest_solid_index names the latest.
// Units: m, world; s for pose_elapsed.
// Example: the floor and the 60 degree side through the origin, p0 = x = {0, 0, 0}: both are near, d = 0, targets
//   {tol, tol}. The unit domain box and a ground lying on its bottom, x = {0.5, 0.5, tol / 2}: one target, tol / 2.
// Requires: pose_elapsed is within the step.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline RestGapTargets BuildRestGapTargets(
    const SolidContactInputs& inputs,
    double pose_elapsed,
    Vec3 start_point,
    Vec3 iterate,
    const ConstrainingSurfaces& previous
) {
  assert(pose_elapsed >= 0 && pose_elapsed <= inputs.step_duration);
  const double contact_tolerance = inputs.contact_tolerance;
  const Vec3 move_so_far = iterate - start_point;  // m
  RestGapTargets built;
  built.surfaces.latest_solid_index = previous.latest_solid_index;
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const SolidPose pose = ComputeSolidPose(inputs.solids[solid_index], pose_elapsed);
    const NearSurfaces at_iterate = FindSurfacesOfSolid(
        inputs,
        solid_index,
        pose,
        iterate,
        ComputeDistanceForParticle(inputs, solid_index, pose, iterate, inputs.plate_side)
    );
    for (int surface_index = 0; surface_index < at_iterate.count; ++surface_index) {
      const DistanceToSolid surface = at_iterate.surfaces[surface_index];
      const bool is_near_now = surface.distance < contact_tolerance;
      const bool was_constraining = ComputeIsConstraining(previous, solid_index, surface_index);
      if (!is_near_now && !was_constraining) {
        continue;
      }
      ApplyConstrainingSurface(&built.surfaces, {.solid_index = solid_index, .surface_index = surface_index});
      // Method paper <proc-move-particle>:  n . w >= tol - d(x) + n . (x - p0)
      const Vec3 normal = surface.normal_out_of_solid;
      const double target = contact_tolerance - surface.distance + math::Dot(normal, move_so_far);  // m
      // A fifth normal: the scan goes on only to find the latest solid, which a failure names.
      if (!ApplyRestGapTarget(&built.targets, normal, target)) {
        built.surfaces.has_too_many = true;
      }
    }
  }
  return built;
}

// Where a move off near surfaces put a point, or that it found none within the move cap.
struct RestGapMove {
  bool is_found = false;  // false: no point at the rest gap of every near surface within the cap was found.
  Vec3 moved_point{};  // m, world. The point itself when no surface was near; unspecified when none was found.
  int iteration_count = 0;  // Iterates of the move off several surfaces, 0 when none was needed.
  // When none was found, the solid latest in array order of every surface that constrained the move.
  std::uint32_t latest_solid_index = kNoSolid;
};

// The solids a point is closer to than the contact tolerance: how many, and the last of them in array order.
struct NearSolids {
  int count = 0;
  std::uint32_t last_solid_index = kNoSolid;  // Index into solids, kNoSolid when count is 0.
};

// Finds the solids a point is closer to than the contact tolerance, at their pose pose_elapsed seconds into the step.
//
// Example: the floor and the 60 degree side through the origin, the point {0, 0, tol / 2}: count 2, last solid 1.
// Requires: pose_elapsed is within the step.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline NearSolids
FindNearSolids(const SolidContactInputs& inputs, double pose_elapsed, Vec3 point) {
  assert(pose_elapsed >= 0 && pose_elapsed <= inputs.step_duration);
  NearSolids near;
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const SolidPose pose = ComputeSolidPose(inputs.solids[solid_index], pose_elapsed);
    const double distance = ComputeDistanceForParticle(inputs, solid_index, pose, point, inputs.plate_side).distance;
    if (distance < inputs.contact_tolerance) {
      ++near.count;
      near.last_solid_index = solid_index;
    }
  }
  return near;
}

// Computes the move off the one solid a point is near, along that solid's own normals (MoveToRestGap), when it leaves
// the point at the rest gap of every solid without crossing another.
//
// Method: MoveToRestGap of that solid, as one solid's start rule and settle always moved: along its normal, or off each
//   closed side of a domain box in turn. The move is kept when every solid is then at least tol - tol / 1024 away
//   (ComputeIsAtLeastFromEverySolid) and the move crosses no surface of another solid (ApplyCrossedSurfaces);
//   otherwise is_found is false and MoveToRestGapOfNearSurfaces moves off all the surfaces at once. It is not limited
//   by the move cap: it is the move of plan decision P12, at most start_overlap_tolerance + tol at the start.
// Units: m, world; s for pose_elapsed.
// Example: a fixed floor and the point {0.5, 0, tol / 4}: {0.5, 0, tol}, the same bits as MoveToRestGap.
// Requires: near.count is 1 and near.last_solid_index names the near solid; pose_elapsed is within the step.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <sec-contact-tolerance>
FLUIDITY_CPU_AND_GPU inline RestGapMove
ComputeMoveOffOneNearSolid(const SolidContactInputs& inputs, double pose_elapsed, Vec3 point, NearSolids near) {
  assert(near.count == 1 && near.last_solid_index < inputs.solid_count);
  const std::uint32_t solid_index = near.last_solid_index;
  const double contact_tolerance = inputs.contact_tolerance;
  const SolidPose pose = ComputeSolidPose(inputs.solids[solid_index], pose_elapsed);
  const Vec3 moved_point = MoveToRestGap(inputs, solid_index, pose, point);
  RestGapMove one_solid_move;
  const double least_distance = contact_tolerance - contact_tolerance / 1024;  // m
  if (!ComputeIsAtLeastFromEverySolid(inputs, pose_elapsed, moved_point, least_distance)) {
    return one_solid_move;
  }
  // The near surfaces of that solid do not count as crossed: the move leaves them.
  ConstrainingSurfaces moved_off;
  const DistanceToSolid at_point = ComputeDistanceForParticle(inputs, solid_index, pose, point, inputs.plate_side);
  const NearSurfaces surfaces = FindSurfacesOfSolid(inputs, solid_index, pose, point, at_point);
  for (int surface_index = 0; surface_index < surfaces.count; ++surface_index) {
    if (surfaces.surfaces[surface_index].distance < contact_tolerance) {
      ApplyConstrainingSurface(&moved_off, {.solid_index = solid_index, .surface_index = surface_index});
    }
  }
  if (ApplyCrossedSurfaces(inputs, pose_elapsed, point, moved_point - point, &moved_off).has_hit) {
    return one_solid_move;
  }
  one_solid_move.is_found = true;
  one_solid_move.moved_point = moved_point;
  return one_solid_move;
}

// Computes the point at the rest gap of every surface a point is near, at the solids' pose pose_elapsed seconds into
// the step, as near to the point as the solver finds: one move off several near surfaces at once (plan decision P19,
// item 1). The start rules and the settle call it with every solid.
//
// Method, with p0 = point and tol = inputs.contact_tolerance:
//   1 No surface closer than tol (FindNearSolids): p0 itself.
//   2 One near solid: its own move (ComputeMoveOffOneNearSolid), kept when it leaves every solid at the rest gap and
//     crosses none. Why not one move per solid in turn: in a wedge sharper than a right angle, each solid's move to its
//     rest gap pushes the point toward the other (at rest on both faces at 60 degrees the floor's move leaves the other
//     face tol / 2 short, and every start failed, 2000 of 2000 at 60 degrees and at 30).
//   3 Otherwise, sequential linearisation from x_0 = p0: x_(k+1) = p0 + w, w the smallest change with
//     Dot(n_i, w) >= tol - d_i(x_k) + Dot(n_i, x_k - p0) for every constraining surface i (BuildRestGapTargets,
//     SolveSmallestCorrection for a position with slack tol / 1024), the distances and normals read afresh at x_k by
//     the per-solid functions alone. A change longer than the cap kMaxRestGapMoveInTolerances * tol is shortened to
//     it, which keeps the first, overshooting iterate of a deep cusp near; the next iterate repairs it. The segment
//     from p0 to each iterate is swept against the other surfaces (ApplyCrossedSurfaces): a move that crosses one
//     stops where it first comes within the growth it is swept with (ComputeSweptGrowth), and that surface constrains
//     the next iterate, linearised there. Dropping the crossed iterate instead would linearise a curved pair at the
//     same point again, and in a cusp beside a wall those targets have no answer although the rest point is far from
//     the wall. The iterates stop when one moves less than tol / 64 and every solid is at least tol - tol / 1024 from
//     it, or after kMaxRestGapIterations; a fifth normal that differs among the constraining surfaces (surfaces with
//     the same normal share one target, ApplyRestGapTarget), more than kMaxTrackedSurfaces surfaces, or a subset with
//     no answer, ends the move with none found.
//   4 The last iterate is kept when every solid is at least tol - tol / 128 from it, by its true distances; otherwise
//     none is found. It is within the cap of p0, since every change was shortened to it.
//   What the iterates guarantee for convex shapes (every shape of module 1: the domain box per side): each surface's
//   distance satisfies d_i(y) >= d_i(x) + Dot(n_i(x), y - x) for every y, the subgradient inequality of plan decision
//   P18. So an iterate x_(k+1) that the solver found and the cap did not shorten meets, for each surface that
//   constrained it, d_i(x_(k+1)) >= tol - slack, the slack being tol / 1024 plus the rounding of composing the change
//   (ComputeIsWithinTargets), which nothing bounds when the multipliers are huge. Such an iterate meets its own
//   linearisation up to that slack, so w = x_(k+1) - p0 meets the targets of the next one, and the smallest w is no
//   longer: from the first unshortened iterate on, the iterates come no farther from p0, up to the slack. A shortened
//   iterate (the cap, or a crossing) meets neither, so before it nothing is promised: a deep start can end beyond the
//   rest point (measured: from 0.005 x* and nearer at R / tol = 2^29, 8 iterates end at 1.1256 x*, both distances tol).
//   The last test reads the true distances, so a point returned is at the rest gap of every surface whatever the
//   shapes and whatever the slack. Two flat faces: exact at the first iterate, tol /
//   sin(a / 2) along the bisector of a wedge of angle a from a start on both (2 tol at 60 degrees). The cusp under a
//   ball of radius R resting on a floor: the first linearisation from deep in the cusp overshoots out along the floor
//   (the two tangent planes meet near half the distance), and the next come back to the rest point {2 Sqrt(R tol), 0,
//   tol} from the contact, both distances exactly tol: 5 iterates from tol / 4 short of both at R / tol = 2^29, within
//   1e-4 of it after 8 from 0.1 and 0.01 x* at R / tol = 2^40 (1.4e-4 from 0.02 x*, and up to 1.3e-4 when a wall
//   beside the cusp takes one iterate for its crossing). Why a cap: the point is displaced without a matching velocity,
//   so the move stays small against a cell (kMaxRestGapMoveInTolerances). Why no angle limit: positions need no
//   conditioning (kVelocityPairLimitShare), and a ball resting on a floor makes every angle down to 0. Cost: one
//   distance per solid when no surface is near (almost every particle); otherwise, per iterate, one read of every
//   surface for the targets, one closed-form sweep for the crossing check of every surface not constraining, and one
//   distance per solid for the stop test.
// Units: m, world; s for pose_elapsed.
// Example: the floor and the 60 degree side through the origin, p0 = {0, 0, 0}: {Sqrt(3) tol, 0, tol}, 1 iterate and a
//   second that moves no more. A ball of radius 0.5 resting on the floor and p0 {Sqrt(0.5 tol), 0, tol / 4}:
//   {2 Sqrt(0.5 tol), 0, tol} = {46340.95 tol, 0, tol} after 5 iterates.
// Requires: inputs.solids points to inputs.solid_count solids that passed CheckSolid; point is finite; pose_elapsed is
//   within the step.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <fig-move-off-near-surfaces>
FLUIDITY_CPU_AND_GPU inline RestGapMove
MoveToRestGapOfNearSurfaces(const SolidContactInputs& inputs, double pose_elapsed, Vec3 point) {
  assert(inputs.solids != nullptr || inputs.solid_count == 0);
  assert(math::IsFinite(point));
  assert(pose_elapsed >= 0 && pose_elapsed <= inputs.step_duration);
  // Steps: 1 no near surface  2 one near solid  3 iterate  4 keep the last iterate
  const double contact_tolerance = inputs.contact_tolerance;
  const double move_cap = kMaxRestGapMoveInTolerances * contact_tolerance;  // m
  // m. An iterate that moves no more stops at this distance from every solid; the last one is kept at the second.
  const double stop_distance = contact_tolerance - contact_tolerance / 1024;
  const double rest_gap_distance = contact_tolerance - contact_tolerance / 128;
  RestGapMove move;
  // Step 1: almost every particle is near no solid, and pays one distance per solid.
  const NearSolids near = FindNearSolids(inputs, pose_elapsed, point);
  if (near.count == 0) {
    move.is_found = true;
    move.moved_point = point;
    return move;
  }
  // Step 2: one solid moves the point as it always did, the same bits, with no cap.
  if (near.count == 1) {
    const RestGapMove one_solid_move = ComputeMoveOffOneNearSolid(inputs, pose_elapsed, point, near);
    if (one_solid_move.is_found) {
      return one_solid_move;
    }
  }
  // Step 3: each iterate meets the linearisation at the one before, measured from the point.
  ConstrainingSurfaces constraining;
  Vec3 iterate = point;
  for (int iteration = 1; iteration <= kMaxRestGapIterations; ++iteration) {
    move.iteration_count = iteration;
    const RestGapTargets built = BuildRestGapTargets(inputs, pose_elapsed, point, iterate, constraining);
    constraining = built.surfaces;
    // More than four directions of near surfaces at once (kStartsInNarrowGap).
    if (constraining.has_too_many) {
      move.latest_solid_index = constraining.latest_solid_index;
      return move;
    }
    const SmallestCorrection correction =
        SolveSmallestCorrection(built.targets, contact_tolerance / 1024, CorrectedQuantity::kPosition);
    // No subset meets every target: two faces facing each other closer than 2 tol, a pocket too small, or a cusp
    // within rounding of its contact point, where the two normals are opposite to the last bit.
    if (!correction.is_found) {
      move.latest_solid_index = constraining.latest_solid_index;
      return move;
    }
    Vec3 change = correction.change;
    const double change_length = math::Length(change);  // m
    if (change_length > move_cap) {
      change = (move_cap / change_length) * change;
    }
    // A move that crosses a surface stops where it first comes within the growth it is swept with (ComputeSweptGrowth),
    // and that surface constrains the next iterate, linearised there, near it, where the linearisation of a curved pair
    // is good.
    const HitTime crossing = ApplyCrossedSurfaces(inputs, pose_elapsed, point, change, &constraining);
    if (constraining.has_too_many) {
      move.latest_solid_index = constraining.latest_solid_index;
      return move;
    }
    const double move_share = crossing.has_hit ? crossing.elapsed : 1;
    const Vec3 next_iterate = point + move_share * change;
    const double step_length = math::Length(next_iterate - iterate);  // m
    iterate = next_iterate;
    const bool is_still = step_length <= contact_tolerance / 64;
    if (is_still && ComputeIsAtLeastFromEverySolid(inputs, pose_elapsed, iterate, stop_distance)) {
      break;
    }
  }
  // Step 4: the true distances decide, whatever the shapes. The slack of a cusp's iterate, tol / 1024 plus the rounding
  // of its large multipliers (about 2 tol / 1024 at R / tol = 2^40), stays below tol / 128.
  // The last iterate is within the cap of the point, since every change was shortened to it.
  if (!ComputeIsAtLeastFromEverySolid(inputs, pose_elapsed, iterate, rest_gap_distance)) {
    move.latest_solid_index = constraining.latest_solid_index;
    return move;
  }
  move.is_found = true;
  move.moved_point = iterate;
  return move;
}

// One active surface of a joint response: a surface the particle is on, or the one the sweep hit.
struct ActiveSurface {
  std::uint32_t solid_index = kNoSolid;  // Index into solids, not the solid_id.
  SolidFace face = SolidFace::kHighZ;  // A domain box's side, or the face of the solid's distance.
  // No unit, unit vector, world: at the particle's position; the hit's own normal for the surface the sweep hit.
  Vec3 normal_out_of_solid{};
  Vec3 solid_velocity{};  // m/s, world: ComputeSolidVelocity at the particle, at the pose of move.elapsed.
  double friction = 0;  // No unit: inputs.friction * friction_scale of the solid.
  bool reports_reaction = false;  // The solid's reports_reaction.
  int direction_index = 0;  // The direction it shares with the surfaces of the same normal (ActiveSurfaces).
};

// The active surfaces of one hit, sorted by value, and the directions they face: surfaces whose normals agree within
// kSameNormalAxisDifference on every axis share one direction, which the solver reads as one surface.
struct ActiveSurfaces {
  // Sorted by value (ComputeIsSortedBefore), exact copies dropped (ComputeIsExactCopy).
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  ActiveSurface surfaces[kMaxTrackedSurfaces]{};
  int count = 0;
  // The normal of each direction: that of its first surface in the sorted order.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  Vec3 directions[kMaxSlideDirections]{};
  int direction_count = 0;
  // More than kMaxTrackedSurfaces surfaces, or more than kMaxSlideDirections directions: one response cannot read them
  // at once, and the hit is kTooManyHits.
  bool has_too_many = false;
};

// Computes whether two active surfaces are the same bit for bit in everything a joint response reads of them: normal,
// solid velocity, friction and reports_reaction. The solid may differ.
//
// Method: an exact copy has the same targets, the same friction limit and the same charge, so it adds nothing to the
//   response; kept, seventeen coincident fixed floors would take seventeen places. Normals that agree only within
//   kSameNormalAxisDifference, or velocities one rounding unit apart, are not copies: both stay, and the pushing
//   surface rule decides between them (BuildJointTargets). Nor is a coincident surface that differs only in friction
//   or reports_reaction: it stays, and its |u|_1 enters the speeds of every slack, so a smoother copy of a moving
//   solid changes the result by rounding (16 of 506,634 runs of the review's probe, at most 2.2e-14 of the speed).
// Example: two fixed floors through the origin, friction scale 1, neither reacting: copies. The same with one moving
//   along itself at 1 m/s: not copies.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsExactCopy(const ActiveSurface& first, const ActiveSurface& second) {
  const Vec3 first_normal = first.normal_out_of_solid;
  const Vec3 second_normal = second.normal_out_of_solid;
  const bool is_same_normal =
      first_normal.x == second_normal.x && first_normal.y == second_normal.y && first_normal.z == second_normal.z;
  const Vec3 first_velocity = first.solid_velocity;
  const Vec3 second_velocity = second.solid_velocity;
  const bool is_same_velocity = first_velocity.x == second_velocity.x && first_velocity.y == second_velocity.y
      && first_velocity.z == second_velocity.z;
  const bool is_same_friction = first.friction == second.friction;
  const bool is_same_reaction = first.reports_reaction == second.reports_reaction;
  return is_same_normal && is_same_velocity && is_same_friction && is_same_reaction;
}

// Computes whether one active surface comes before another in the sorted order: by normal x, y, z, then solid velocity
// x, y, z, then friction, each compared by <, then a reacting solid before one that does not react, then solid
// index.
//
// Method (plan decision P20, item 1): every sum and every tie of the joint response reads the surfaces in this order,
//   which their values decide, so the velocities are the same bits for every order of the solids array (measured: 0
//   differences over every permutation of 119,999 corners; in array order, 554 of 39,999 differ, by at most 1.7e-13 of
//   the speed). Two surfaces equal in every value key differ only in reports_reaction (else one is an exact copy of
//   the other, dropped): the reacting one comes first, so a stick on such coincident surfaces is charged to the
//   reacting solid whatever the array order (P5). The solid index then never decides between two kept surfaces of
//   different solids; it stays as the last key so that the order is total.
// Example: normals {0, 0, 1} and {1, 0, 0}: {0, 0, 1} first, its x is smaller. Two floors with normal {0, 0, 1}, one
//   fixed and one moving at {1, 0, 0}: the fixed one first.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsSortedBefore(const ActiveSurface& first, const ActiveSurface& second) {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  const double first_keys[] = {
      first.normal_out_of_solid.x,
      first.normal_out_of_solid.y,
      first.normal_out_of_solid.z,
      first.solid_velocity.x,
      first.solid_velocity.y,
      first.solid_velocity.z,
      first.friction,
  };
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  const double second_keys[] = {
      second.normal_out_of_solid.x,
      second.normal_out_of_solid.y,
      second.normal_out_of_solid.z,
      second.solid_velocity.x,
      second.solid_velocity.y,
      second.solid_velocity.z,
      second.friction,
  };
  for (int key_index = 0; key_index < 7; ++key_index) {
    if (first_keys[key_index] < second_keys[key_index]) {
      return true;
    }
    if (second_keys[key_index] < first_keys[key_index]) {
      return false;
    }
  }
  if (first.reports_reaction != second.reports_reaction) {
    return first.reports_reaction;
  }
  return first.solid_index < second.solid_index;
}

// Applies one surface to the active surfaces: dropped when it is an exact copy of a listed one, else inserted at its
// place in the sorted order.
//
// Method: an insertion into at most kMaxTrackedSurfaces places: the listed surfaces that it comes before
//   (ComputeIsSortedBefore) move up by one. A seventeenth surface that is not a copy sets has_too_many. Copies are
//   tested before the limit, so 17 or 40 coincident fixed floors take one place.
// Example: the list {floor {0, 0, 1}} and a side {-1, 0, 0}: {side, floor}. Then a second fixed floor {0, 0, 1}: no
//   change.
// Requires: active is not null; directions are not yet assigned (ApplyDirections runs after every surface).
FLUIDITY_CPU_AND_GPU inline void ApplyActiveSurface(ActiveSurfaces* active, const ActiveSurface& surface) {
  assert(active != nullptr);
  for (int listed_index = 0; listed_index < active->count; ++listed_index) {
    if (ComputeIsExactCopy(active->surfaces[listed_index], surface)) {
      return;
    }
  }
  if (active->count == kMaxTrackedSurfaces) {
    active->has_too_many = true;
    return;
  }
  int insert_index = active->count;
  while (insert_index > 0 && ComputeIsSortedBefore(surface, active->surfaces[insert_index - 1])) {
    active->surfaces[insert_index] = active->surfaces[insert_index - 1];
    --insert_index;
  }
  active->surfaces[insert_index] = surface;
  ++active->count;
}

// Applies the directions to the sorted active surfaces: each surface joins the first direction whose normal is the
// same within kSameNormalAxisDifference (ComputeIsSameNormal), else opens one with its own normal.
//
// Method: in the sorted order, so a direction's normal is that of its first surface by value, whatever the array order.
//   A ninth direction sets has_too_many.
// Example: a floor {0, 0, 1}, a ground {2e-13, 0, 1} and a side {1, 0, 0}: two directions, {0, 0, 1} and {1, 0, 0}.
// Requires: active is not null.
FLUIDITY_CPU_AND_GPU inline void ApplyDirections(ActiveSurfaces* active) {
  assert(active != nullptr);
  for (int surface_index = 0; surface_index < active->count; ++surface_index) {
    ActiveSurface& surface = active->surfaces[surface_index];
    int direction_index = 0;
    while (direction_index < active->direction_count
           && !ComputeIsSameNormal(active->directions[direction_index], surface.normal_out_of_solid)) {
      ++direction_index;
    }
    if (direction_index == active->direction_count) {
      if (active->direction_count == kMaxSlideDirections) {
        active->has_too_many = true;
        return;
      }
      active->directions[direction_index] = surface.normal_out_of_solid;
      ++active->direction_count;
    }
    surface.direction_index = direction_index;
  }
}

// Finds the active surfaces of one hit (plan decision P20, item 1): every surface of every solid that the particle at
// its position is within tol + tol / 16 of, read afresh at the pose of move.elapsed, and the surface the sweep hit,
// whatever its distance, with the hit's own normal and face.
//
// Method: every solid in array order, every surface in its list (FindSurfacesOfSolid), with its distance and normal at
//   the position now. A surface within the window counts whether or not a velocity approaches it, touched or not: at
//   the rest point of a wedge sharper than 2^-21 rad the floor is at exactly tol. A surface farther than tol + tol / 16
//   is one the particle is not on: a ceiling 0.1 m above a floor touched earlier must not stop it. The surface the
//   sweep hit (same solid; for a domain box the same face) counts whatever its distance, since a turning solid is hit
//   up to 1.25 tol away, and with the hit's normal, along which the hit measured the approach. Each surface carries its
//   solid's velocity at the particle, friction times its friction_scale, and reports_reaction. Exact copies are dropped
//   and the rest sorted (ApplyActiveSurface), then the directions are assigned (ApplyDirections). Cost: one distance
//   per solid.
// Units: m, m/s, world.
// Example: a fixed floor and a particle at {0.5, 0, tol}: one surface, normal {0, 0, 1}. At {0.5, 0, 0.02}: none.
//   Eight coincident fixed floors and an upright flat solid, the particle at their corner: two surfaces, the upright
//   and one floor.
// Requires: move.elapsed is within the step; hit is null at the start, else the hit that moved the particle here.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline ActiveSurfaces
FindActiveSurfaces(const SolidContactInputs& inputs, const ParticleMove& move, const FirstHit* hit) {
  const Vec3 position = move.result.position;
  const double active_distance = inputs.contact_tolerance + kTouchedSolidRestingGapShare * inputs.contact_tolerance;
  ActiveSurfaces active;
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    const SolidPose pose = ComputeSolidPose(solid, move.elapsed);
    const DistanceToSolid at_position =
        ComputeDistanceForParticle(inputs, solid_index, pose, position, inputs.plate_side);
    const NearSurfaces surfaces = FindSurfacesOfSolid(inputs, solid_index, pose, position, at_position);
    const Vec3 solid_velocity = ComputeSolidVelocity(solid, pose, position);
    const bool is_hit_solid = hit != nullptr && hit->solid_index == solid_index;
    const bool has_sides = solid.shape == SolidShape::kDomainBox;
    for (int surface_index = 0; surface_index < surfaces.count; ++surface_index) {
      const DistanceToSolid surface = surfaces.surfaces[surface_index];
      const bool is_hit_surface = is_hit_solid && (!has_sides || surface.face == hit->at_hit.face);
      const bool is_within_window = surface.distance <= active_distance;
      if (!is_hit_surface && !is_within_window) {
        continue;
      }
      const DistanceToSolid read = is_hit_surface ? hit->at_hit : surface;
      const ActiveSurface active_surface = {
          .solid_index = solid_index,
          .face = read.face,
          .normal_out_of_solid = read.normal_out_of_solid,
          .solid_velocity = solid_velocity,
          .friction = inputs.friction * solid.friction_scale,
          .reports_reaction = solid.reports_reaction,
          .direction_index = 0,
      };
      ApplyActiveSurface(&active, active_surface);
    }
  }
  ApplyDirections(&active);
  return active;
}

// Applies a close call to a particle's move: *failure becomes kTooCloseToCall with that close call, naming one solid,
// magnitude 0, and the error is returned.
//
// Requires: solid_index < inputs.solid_count; failure is not null.
FLUIDITY_CPU_AND_GPU inline ContactError ApplyCloseCall(
    const SolidContactInputs& inputs,
    ContactCloseCall close_call,
    std::uint32_t solid_index,
    ContactFailure* failure
) {
  assert(solid_index < inputs.solid_count);
  assert(failure != nullptr);
  *failure = {
      .error = ContactError::kTooCloseToCall,
      .close_call = close_call,
      .particle_id = 0,
      .solid_id = inputs.solids[solid_index].solid_id,
      .magnitude = 0,
  };
  return ContactError::kTooCloseToCall;
}

// The targets one velocity has against the active surfaces: per direction for the solver, and per surface.
struct JointTargets {
  // Per direction: the normal, the largest target of its surfaces, the friction of its pushing surface; and the
  // speeds |v|_1 + sum of |u_i|_1 over the sorted surfaces.
  CorrectionTargets targets{};
  // m/s. Per active surface: -Dot(n_i, v - u_i), the approach removed along its own normal from its own solid.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  double own_targets[kMaxTrackedSurfaces]{};
  // Per direction: the index into the active surfaces of its pushing surface.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  int pushing_surfaces[kMaxSlideDirections]{};
};

// Computes whether one surface pushes in its direction before the one kept so far: it has the larger friction, then it
// reports a reaction where the kept one does not. Equal in both: the kept one, first in the sorted order, stays.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsPushingBefore(const ActiveSurface& surface, const ActiveSurface& kept) {
  if (surface.friction != kept.friction) {
    return surface.friction > kept.friction;
  }
  return surface.reports_reaction && !kept.reports_reaction;
}

// Builds the targets of one velocity against the active surfaces, and each direction's pushing surface (plan decision
// P20, items 3 and 4).
//
// Method: speeds = |v|_1 + sum of |u_i|_1, in the sorted order. Each surface's own target is -Dot(n_i, v - u_i),
//   measured along its own normal from its own solid; a direction's target is the largest of its surfaces'. The
//   pushing surface of a direction carries its friction and its reaction: among its surfaces whose own target is
//   within kSameNormalAxisDifference * (|v|_1 + |u_i|_1) of the direction's (they push; the bound is what normals
//   kSameNormalAxisDifference apart can differ by), the one with the largest friction, then one whose solid reports a
//   reaction, then the first in the sorted order (ComputeIsPushingBefore). So a coincident surface as rough or
//   smoother, with the same solid velocity, changes nothing up to rounding (with a moving solid, its speed enters the
//   slacks: at most 2.2e-14 of the speed, ComputeIsExactCopy), and a rougher one acts in place of the smoother. Same
//   bits on CPU and GPU when no solid turns: only + - *, Abs and comparisons.
// Units: m/s, world.
// Example: a fixed floor of friction 0.2 and a coincident fixed floor of friction 0.4, the velocity {1, 0, -1}: one
//   direction, target 1 m/s, the second floor pushes, frictions[0] = 0.4.
// Requires: active.direction_count is 1 to kMaxSlideDirections and every surface has its direction.
FLUIDITY_CPU_AND_GPU inline JointTargets BuildJointTargets(const ActiveSurfaces& active, Vec3 velocity) {
  assert(active.direction_count >= 1 && active.direction_count <= kMaxSlideDirections);
  JointTargets built;
  CorrectionTargets& targets = built.targets;
  targets.count = active.direction_count;
  const double velocity_axes_sum = math::Abs(velocity.x) + math::Abs(velocity.y) + math::Abs(velocity.z);  // m/s
  double speeds = velocity_axes_sum;  // m/s
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  bool has_target[kMaxSlideDirections]{};
  for (int surface_index = 0; surface_index < active.count; ++surface_index) {
    const ActiveSurface& surface = active.surfaces[surface_index];
    const Vec3 solid_velocity = surface.solid_velocity;
    speeds = speeds + (math::Abs(solid_velocity.x) + math::Abs(solid_velocity.y) + math::Abs(solid_velocity.z));
    // Method paper <proc-move-particle>:  t_i = -n_i . (v - u_i)
    const double own_target = -math::Dot(surface.normal_out_of_solid, velocity - solid_velocity);  // m/s
    built.own_targets[surface_index] = own_target;
    const int direction_index = surface.direction_index;
    if (!has_target[direction_index] || own_target > targets.targets[direction_index]) {
      targets.targets[direction_index] = own_target;
    }
    has_target[direction_index] = true;
  }
  targets.speeds_before_change = speeds;
  for (int direction_index = 0; direction_index < active.direction_count; ++direction_index) {
    targets.normals[direction_index] = active.directions[direction_index];
    int pushing_index = -1;
    for (int surface_index = 0; surface_index < active.count; ++surface_index) {
      const ActiveSurface& surface = active.surfaces[surface_index];
      if (surface.direction_index != direction_index) {
        continue;
      }
      const Vec3 solid_velocity = surface.solid_velocity;
      const double solid_axes_sum =
          math::Abs(solid_velocity.x) + math::Abs(solid_velocity.y) + math::Abs(solid_velocity.z);
      const double target_rounding = kSameNormalAxisDifference * (velocity_axes_sum + solid_axes_sum);  // m/s
      const bool pushes = built.own_targets[surface_index] >= targets.targets[direction_index] - target_rounding;
      if (pushes && (pushing_index < 0 || ComputeIsPushingBefore(surface, active.surfaces[pushing_index]))) {
        pushing_index = surface_index;
      }
    }
    // The surface whose target the direction kept always pushes, unless that target is NaN: then the first one acts.
    if (pushing_index < 0) {
      pushing_index = 0;
      while (active.surfaces[pushing_index].direction_index != direction_index) {
        ++pushing_index;
      }
    }
    built.pushing_surfaces[direction_index] = pushing_index;
    targets.frictions[direction_index] = active.surfaces[pushing_index].friction;
  }
  return built;
}

// Computes the solid a velocity approaches fastest: that of the active surface with the largest own target, the
// lowest solid index on a tie (plan decision P20, item 10). A squeeze names it.
//
// Example: a fixed floor and a ceiling coming down at 1 m/s, the velocity {0, 0, 0}: the ceiling (own target 1 m/s
//   against 0).
// Requires: active.count >= 1; built is BuildJointTargets of the velocity.
FLUIDITY_CPU_AND_GPU inline std::uint32_t
ComputeFastestApproachedSolid(const ActiveSurfaces& active, const JointTargets& built) {
  assert(active.count >= 1);
  int fastest_index = 0;
  for (int surface_index = 1; surface_index < active.count; ++surface_index) {
    const double own_target = built.own_targets[surface_index];
    const double fastest_target = built.own_targets[fastest_index];
    const bool is_faster = own_target > fastest_target;
    const bool is_tie_at_smaller_index = own_target == fastest_target
        && active.surfaces[surface_index].solid_index < active.surfaces[fastest_index].solid_index;
    if (is_faster || is_tie_at_smaller_index) {
      fastest_index = surface_index;
    }
  }
  return active.surfaces[fastest_index].solid_index;
}

// What one joint response does to one velocity: the velocity it returns, its approach allowance, and the changes it
// charges to solids (for velocity.for_next_step, the reaction).
struct JointChange {
  Vec3 velocity{};  // m/s, world.
  // m/s. What the response may have left along any active normal by rounding: the largest of the allowance so far and
  // that of each solve made.
  double approach_allowance = 0;
  int charge_count = 0;  // At most 2 * kMaxUsedDirections: a push and a friction share per used direction.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  std::uint32_t charged_solids[2 * kMaxUsedDirections]{};
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  Vec3 charges[2 * kMaxUsedDirections]{};  // m/s, world: the change charged to each solid.
  // The solid the velocity passed approaches fastest (ComputeFastestApproachedSolid), named by a squeeze that comes
  // after the change: the check of the allowance over the time left (ApplyJointResponse).
  std::uint32_t approached_solid_index = kNoSolid;
};

// Applies one charge to a joint change: the solid and the change of velocity it is charged with.
FLUIDITY_CPU_AND_GPU inline void ApplyCharge(JointChange* change, std::uint32_t solid_index, Vec3 charge) {
  assert(change != nullptr);
  assert(change->charge_count < 2 * kMaxUsedDirections);
  change->charged_solids[change->charge_count] = solid_index;
  change->charges[change->charge_count] = charge;
  ++change->charge_count;
}

// The friction of one joint response over its used directions (plan decision P20, item 5).
struct JointFriction {
  Vec3 velocity{};  // m/s, world: x1, the frictionless velocity v0 with the friction change F.
  Vec3 change{};  // m/s, world: F.
  // m/s, world: each used direction's share of F, in the order of the solver's used directions.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  Vec3 shares[kMaxUsedDirections]{};
  // One used direction with friction: x1 is ComputeContactResponse itself, and the whole change goes to its solid.
  bool is_single_response = false;
  // The used pushing surfaces move differently: F may have a part along a used normal, so a second solve follows.
  bool moves_differently = false;
};

// Computes one surface's friction step on a velocity y: its sideways part relative to the solid, shortened by the
// friction limit, whole when the limit reaches it (plan decision P20, item 5).
//
// Method: t = y - u; t = t - Dot(n, t) n; the step is -t when friction_limit >= Length(t), else
//   -(friction_limit / Length(t)) t. It never reverses the sideways velocity.
// Units: m/s, world.
// Example: y {1, 0, 0}, a fixed floor {0, 0, 1}, limit 0.25 m/s: {-0.25, 0, 0}. Limit 2 m/s: {-1, 0, 0}.
FLUIDITY_CPU_AND_GPU inline Vec3
ComputeFrictionStep(Vec3 velocity, const ActiveSurface& pushing, double friction_limit) {
  const Vec3 normal = pushing.normal_out_of_solid;
  const Vec3 relative = velocity - pushing.solid_velocity;
  const Vec3 sideways = relative - math::Dot(normal, relative) * normal;  // m/s
  const double sideways_speed = math::Length(sideways);  // m/s
  if (friction_limit >= sideways_speed) {
    return -sideways;
  }
  return -(friction_limit / sideways_speed) * sideways;
}

// The orders in which two or three used directions apply their friction one after the other: for three every order;
// for two, orders 0 and 2, whose first two members are {0, 1} and {1, 0}.
// NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
inline constexpr int kFrictionOrders[kMaxFrictionOrders][kMaxUsedDirections] = {
    {0, 1, 2},
    {0, 2, 1},
    {1, 0, 2},
    {1, 2, 0},
    {2, 0, 1},
    {2, 1, 0},
};

// Computes the mean over every order of the used directions of their own frictions applied one after the other from
// v0, for pushing surfaces that move differently (plan decision P20, item 5).
//
// Method: for each order (2 or 6), y = v0, and for each used direction in the order y = y + ComputeFrictionStep(y),
//   that step being the order's share of the direction. F is the sum over the orders of y - v0 times 1 / order count,
//   each share likewise. The mean does not depend on any order. Each step stops at sticking to its own solid at that
//   moment, but a later step of the other surface moves the velocity on, and the mean keeps both: a known limit of
//   this approximation, not a property of friction, is that a surface's own friction can push the particle past
//   sticking to its solid, reversing the sliding relative to it (the review's probe: 1,331 of 24,401 responses that
//   used the mean, 5.5 %, by up to 0.22 and at p99 0.072 of the summed speeds; pinned by
//   MeanOverOrdersBetweenMovingSolids_FrictionCanPassStick_KnownLimit). The exact answer for solids that move
//   differently is a question for the owner. When the solid velocities differ by rounding only it agrees with the stick
//   or slip of one solid velocity to 1.7e-14 of the speed (20,000 wedges). It approximates the exact answer of the same
//   friction model: median 0.046, p90 0.19, p99 0.37 and at worst 0.66 of the largest speed over 2,418 draws with
//   friction up to 2 and solids up to 1 m/s apart; more passes do not help. Adding every surface's own friction from
//   the same v0 instead reverses the sliding when two velocities are one rounding unit apart.
// Units: m/s, world.
// Example: a floor moving at {0, 1, 0} and a fixed side with normal {1, 0, 0}, v0 {0, 1, 0}, both limits 0.2 m/s. The
//   floor first: its step is 0, the side's {0, -0.2, 0}, y {0, 0.8, 0}. The side first: {0, 0.8, 0}, then the floor
//   takes the 0.2 m/s it slides relative to the floor, y {0, 1, 0}. The mean: F {0, -0.1, 0}, x1 {0, 0.9, 0}.
// Requires: used_count is 2 or 3; pushing and friction_limits hold used_count entries.
FLUIDITY_CPU_AND_GPU inline JointFriction ComputeMeanOverOrders(
    Vec3 frictionless_velocity,
    const ActiveSurface* pushing,
    const double* friction_limits,
    int used_count
) {
  assert(used_count == 2 || used_count == 3);
  const int order_count = used_count == 2 ? 2 : kMaxFrictionOrders;
  JointFriction friction;
  friction.moves_differently = true;
  Vec3 total{};  // m/s
  for (int order_index = 0; order_index < order_count; ++order_index) {
    // For two used directions, orders 0 and 2: {0, 1} and {1, 0}.
    const int table_index = used_count == 2 ? 2 * order_index : order_index;
    Vec3 stepped = frictionless_velocity;  // m/s
    for (int member_index = 0; member_index < used_count; ++member_index) {
      const int used_index = kFrictionOrders[table_index][member_index];
      const Vec3 step = ComputeFrictionStep(stepped, pushing[used_index], friction_limits[used_index]);
      stepped = stepped + step;
      friction.shares[used_index] = friction.shares[used_index] + step;
    }
    total = total + (stepped - frictionless_velocity);
  }
  const double order_share = 1.0 / order_count;  // No unit.
  friction.change = order_share * total;
  for (int used_index = 0; used_index < used_count; ++used_index) {
    friction.shares[used_index] = order_share * friction.shares[used_index];
  }
  friction.velocity = frictionless_velocity + friction.change;
  return friction;
}

// Computes the friction of one joint response from the frictionless velocity v0, the pushes and each used direction's
// pushing surface (plan decision P20, item 5).
//
// Method, with L_k = friction_k * push_k each used direction's friction limit:
//   every L_k zero  F = 0, x1 = v0.
//   one used direction, with friction  x1 = ComputeContactResponse(v, u_k, n_k, friction_k), the single-surface
//     response itself, bit for bit, and F = x1 - ComputeContactResponse(v, u_k, n_k, 0), which only a second solve
//     reads. Without friction (L_k zero) the case above applies: x1 = v0 = v + w from the solver, which equals the
//     single-surface response only up to rounding.
//   several, every u_k the same bit for bit  the surfaces act as one: with s = v0 - u and L = sum of L_k in used order,
//     x1 = u when L >= Length(s) (stick), else v0 + (L / Length(s)) (u - v0) (slip); F = x1 - v0, shared as
//     (L_k / L) F. The exact answer of friction with these pushes.
//   several, moving differently  ComputeMeanOverOrders.
// Units: m/s, world.
// Example: a fixed floor and a fixed face at 30 degrees to it, v {-1, 1, -1}, friction 0.5: pushes 2.732 and 2 m/s, so
//   L = 0.5 (2.732 + 2) = 2.366 m/s against the 1 m/s left along the apex: x1 = {0, 0, 0}, a stick. At 60 degrees
//   with v {-0.2, 1, -1}: L = 0.6732 m/s, x1 = {0, 0.3268, 0}.
// Requires: correction.used_count is 1 to kMaxUsedDirections; built is BuildJointTargets of velocity.
FLUIDITY_CPU_AND_GPU inline JointFriction ComputeJointFriction(
    const ActiveSurfaces& active,
    const JointTargets& built,
    const SmallestCorrection& correction,
    Vec3 velocity
) {
  const int used_count = correction.used_count;
  assert(used_count >= 1 && used_count <= kMaxUsedDirections);
  const Vec3 frictionless_velocity = velocity + correction.change;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  ActiveSurface pushing[kMaxUsedDirections]{};
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  double friction_limits[kMaxUsedDirections]{};
  bool has_friction = false;
  bool has_one_solid_velocity = true;
  for (int used_index = 0; used_index < used_count; ++used_index) {
    const int direction_index = correction.used_surfaces[used_index];
    pushing[used_index] = active.surfaces[built.pushing_surfaces[direction_index]];
    // Method paper <proc-move-particle>:  L_k = mu_k lambda_k
    friction_limits[used_index] = pushing[used_index].friction * correction.multipliers[used_index];
    has_friction = has_friction || friction_limits[used_index] != 0;
    const Vec3 first_velocity = pushing[0].solid_velocity;
    const Vec3 solid_velocity = pushing[used_index].solid_velocity;
    has_one_solid_velocity = has_one_solid_velocity && solid_velocity.x == first_velocity.x
        && solid_velocity.y == first_velocity.y && solid_velocity.z == first_velocity.z;
  }
  JointFriction friction;
  if (!has_friction) {
    friction.velocity = frictionless_velocity;
    return friction;
  }
  if (used_count == 1) {
    const ActiveSurface& surface = pushing[0];
    const Vec3 normal = surface.normal_out_of_solid;
    friction.velocity = ComputeContactResponse(velocity, surface.solid_velocity, normal, surface.friction);
    friction.change = friction.velocity - ComputeContactResponse(velocity, surface.solid_velocity, normal, 0);
    friction.shares[0] = friction.change;
    friction.is_single_response = true;
    return friction;
  }
  if (!has_one_solid_velocity) {
    return ComputeMeanOverOrders(frictionless_velocity, pushing, friction_limits, used_count);
  }
  const Vec3 solid_velocity = pushing[0].solid_velocity;
  const double sideways_speed = math::Length(frictionless_velocity - solid_velocity);  // m/s
  double friction_limit_sum = 0;  // m/s
  for (int used_index = 0; used_index < used_count; ++used_index) {
    friction_limit_sum = friction_limit_sum + friction_limits[used_index];
  }
  // Method paper <proc-move-particle>:  x1 = u if L >= |v0 - u|, else v0 + (L / |v0 - u|) (u - v0)
  if (friction_limit_sum >= sideways_speed) {
    friction.velocity = solid_velocity;
  } else {
    friction.velocity =
        frictionless_velocity + (friction_limit_sum / sideways_speed) * (solid_velocity - frictionless_velocity);
  }
  friction.change = friction.velocity - frictionless_velocity;
  for (int used_index = 0; used_index < used_count; ++used_index) {
    friction.shares[used_index] = (friction_limits[used_index] / friction_limit_sum) * friction.change;
  }
  return friction;
}

// Computes the stick of a velocity that no change leaves every active surface (plan decision P20, item 7): it takes
// the solid velocity of the active surface it approaches fastest, and any direction that velocity still approaches
// beyond the slack is a squeeze.
//
// Method: the surface with the largest own target (a tie: the first in the sorted order); the velocity becomes its
//   solid velocity, one charge, that velocity minus the velocity passed, to its solid. With the slack
//   Max(allowance so far, 2^-46 (speeds + |change|_1)), a target at the new velocity above it is
//   kSqueezedBetweenSolids, naming the solid that velocity approaches fastest. A pair below kVelocityPairLimitShare (a
//   sliver: a wedge under 1.35e-6 rad) or two solids closing on the particle come here.
// Units: m/s, world.
// Requires: active.count >= 1; built is BuildJointTargets of velocity; change and squeezing_solid_index are not null.
// Fails: kTooCloseToCall  a squeeze; *squeezing_solid_index is written only then.
FLUIDITY_CPU_AND_GPU inline ContactError ApplyStick(
    const ActiveSurfaces& active,
    const JointTargets& built,
    Vec3 velocity,
    double approach_allowance_so_far,
    JointChange* change,
    std::uint32_t* squeezing_solid_index
) {
  assert(active.count >= 1);
  assert(change != nullptr && squeezing_solid_index != nullptr);
  int fastest_index = 0;
  for (int surface_index = 1; surface_index < active.count; ++surface_index) {
    if (built.own_targets[surface_index] > built.own_targets[fastest_index]) {
      fastest_index = surface_index;
    }
  }
  const ActiveSurface& stuck = active.surfaces[fastest_index];
  const Vec3 stuck_velocity = stuck.solid_velocity;
  const Vec3 stick_change = stuck_velocity - velocity;
  const double change_axes_sum = math::Abs(stick_change.x) + math::Abs(stick_change.y) + math::Abs(stick_change.z);
  const double slack = math::Max(
      approach_allowance_so_far,
      kApproachLeftAfterResponseShare * (built.targets.speeds_before_change + change_axes_sum)
  );
  const JointTargets after_stick = BuildJointTargets(active, stuck_velocity);
  if (!ComputeIsWithinSlack(after_stick.targets, slack)) {
    *squeezing_solid_index = ComputeFastestApproachedSolid(active, after_stick);
    return ContactError::kTooCloseToCall;
  }
  change->velocity = stuck_velocity;
  change->approach_allowance = math::Max(change->approach_allowance, slack);
  ApplyCharge(change, stuck.solid_index, stick_change);
  return ContactError::kNone;
}

// Computes the second frictionless solve of a joint response, from v + F (plan decision P20, item 6): the pushes are
// found afresh, so friction lifts a particle off a surface only when it overcomes what pressed it there.
//
// Method: z = v + F with its targets (BuildJointTargets). No target above the allowance so far: z, with the allowance
//   Max(allowance so far, 2^-46 speeds). Else SolveSmallestCorrection; none found: the stick from z (ApplyStick);
//   found: z + w2, faster than kMaxSqueezeOutSpeedFactor * largest_speed is kSqueezedBetweenSolids, and each push of w2
//   is charged to its direction's pushing solid at z. The second solve's pushes add no friction.
// Units: m/s, world.
// Requires: change and squeezing_solid_index are not null; change holds the friction shares already.
// Fails: kTooCloseToCall  a squeeze; *squeezing_solid_index is written only then.
//
// clang-tidy: the allowance so far and the largest speed are both m/s doubles; the swappable-parameters check flags
// them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU inline ContactError ApplySecondSolve(
    const ActiveSurfaces& active,
    Vec3 velocity_with_friction,
    double approach_allowance_so_far,
    double largest_speed,
    JointChange* change,
    std::uint32_t* squeezing_solid_index
) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  assert(change != nullptr && squeezing_solid_index != nullptr);
  const JointTargets built = BuildJointTargets(active, velocity_with_friction);
  const double speeds = built.targets.speeds_before_change;  // m/s
  if (ComputeIsWithinSlack(built.targets, approach_allowance_so_far)) {
    change->velocity = velocity_with_friction;
    change->approach_allowance = math::Max(
        change->approach_allowance,
        math::Max(approach_allowance_so_far, kApproachLeftAfterResponseShare * speeds)
    );
    return ContactError::kNone;
  }
  const SmallestCorrection correction =
      SolveSmallestCorrection(built.targets, approach_allowance_so_far, CorrectedQuantity::kVelocity);
  if (!correction.is_found) {
    return ApplyStick(active, built, velocity_with_friction, approach_allowance_so_far, change, squeezing_solid_index);
  }
  const Vec3 changed_velocity = velocity_with_friction + correction.change;
  if (math::Length(changed_velocity) > kMaxSqueezeOutSpeedFactor * largest_speed) {
    *squeezing_solid_index = ComputeFastestApproachedSolid(active, built);
    return ContactError::kTooCloseToCall;
  }
  change->velocity = changed_velocity;
  const double second_allowance =
      ComputeCorrectionSlack(built.targets, approach_allowance_so_far, CorrectedQuantity::kVelocity, correction.change);
  change->approach_allowance = math::Max(change->approach_allowance, second_allowance);
  for (int used_index = 0; used_index < correction.used_count; ++used_index) {
    const int direction_index = correction.used_surfaces[used_index];
    const ActiveSurface& pushing = active.surfaces[built.pushing_surfaces[direction_index]];
    // Method paper <proc-move-particle>:  push_k = lambda_k n_k
    ApplyCharge(change, pushing.solid_index, correction.multipliers[used_index] * active.directions[direction_index]);
  }
  return ContactError::kNone;
}

// Computes what one joint response does to one velocity over the active surfaces (plan decision P20, items 2 to 8,
// and 10).
//
// Method, with v the velocity, b the allowance so far (0 before the first response of the step) and u_i, n_i each
//   active surface's solid velocity and normal:
//   1 One active surface: ComputeContactResponse(v, u, n, friction), the single-surface response of Task 7, bit for
//     bit; one charge, the whole change, to its solid; allowance Max(b, ComputeLargestApproachLeftAfterResponse(v, u)).
//     No solver, no cap.
//   2 The targets and pushing surfaces (BuildJointTargets).
//   3 No target above b: v, bit for bit, no charge, allowance Max(b, 2^-46 speeds).
//   4 SolveSmallestCorrection with slack b gives w = sum push_k n_k over at most three directions; none found: the
//     stick (ApplyStick). v0 = v + w faster than kMaxSqueezeOutSpeedFactor * largest_speed is kSqueezedBetweenSolids.
//   5 and 6 The friction from v0 (ComputeJointFriction): x1 = v0 + F.
//   7 One used direction or one solid velocity: with slack = Max(b, 2^-46 (speeds + |x1 - v|_1)), when no target at
//     x1 is above it, x1 is the result, with that allowance; the charges are the whole change x1 - v to the pushing
//     solid of a single response, else push_k n_k plus its share of F to each used direction's pushing solid.
//   8 Otherwise (the test failed, or the used solids move differently), each friction share to its pushing solid, and
//     the second solve from v + F (ApplySecondSolve). The allowance is also at least the first solve's,
//     2^-46 (speeds + |w|_1).
//   Why one response over all surfaces and not one per surface in turn: the result of the sequence depended on its
//   order, which the rounding of the start point decided (16.0 % of 100,000 draws one rounding unit apart jumped, by
//   up to 0.42 of the speed). Here every formula is a sum or a largest value over the sorted surfaces.
//   Same bits on CPU and GPU when no solid turns: only + - * /, Abs, Sqrt and comparisons.
// Units: m/s, world.
// Example: a fixed floor and a fixed face at 60 degrees to it, friction 0, v {-1, 1, -1}: {0, 1, 0}, along the apex,
//   where the responses one surface after the other gave the floor's {-1, 1, 0}, then the side's {-0.25, 1, -0.433},
//   which approaches the floor again. With friction 0.5 and v {-0.2, 1, -1}: {0, 0.3268, 0}.
// Requires: active.count is at most kMaxTrackedSurfaces and not has_too_many; largest_speed is the largest Length of
//   both velocities before the response and of every active solid velocity; change and squeezing_solid_index are not
//   null.
// Fails: kTooCloseToCall  a squeeze: the escape faster than the cap, or a stick that still approaches a surface.
//                         *squeezing_solid_index is the solid the failing velocity approaches fastest
//                         (ComputeFastestApproachedSolid); it is written only on failure, *change is then
//                         unspecified.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <sec-contact-response>
//
// clang-tidy: as ApplySecondSolve.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU inline ContactError ComputeJointChange(
    const ActiveSurfaces& active,
    Vec3 velocity,
    double approach_allowance_so_far,
    double largest_speed,
    JointChange* change,
    std::uint32_t* squeezing_solid_index
) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  assert(!active.has_too_many);
  assert(change != nullptr && squeezing_solid_index != nullptr);
  // Steps: 1 one surface  2 targets  3 no change  4 solve  5, 6 friction  7 test  8 second solve
  *change = {};
  change->velocity = velocity;
  change->approach_allowance = approach_allowance_so_far;
  if (active.count == 0) {
    return ContactError::kNone;
  }
  // Step 1: Task 7's response, bit for bit.
  if (active.count == 1) {
    const ActiveSurface& surface = active.surfaces[0];
    change->velocity =
        ComputeContactResponse(velocity, surface.solid_velocity, surface.normal_out_of_solid, surface.friction);
    change->approach_allowance =
        math::Max(approach_allowance_so_far, ComputeLargestApproachLeftAfterResponse(velocity, surface.solid_velocity));
    change->approached_solid_index = surface.solid_index;
    ApplyCharge(change, surface.solid_index, change->velocity - velocity);
    return ContactError::kNone;
  }
  // Step 2: one target per direction, each surface along its own normal.
  const JointTargets built = BuildJointTargets(active, velocity);
  const double speeds = built.targets.speeds_before_change;  // m/s
  change->approached_solid_index = ComputeFastestApproachedSolid(active, built);
  // Step 3: plan decision P20, item 3, the slack passed and not the candidate's own.
  if (ComputeIsWithinSlack(built.targets, approach_allowance_so_far)) {
    change->approach_allowance = math::Max(approach_allowance_so_far, kApproachLeftAfterResponseShare * speeds);
    return ContactError::kNone;
  }
  // Step 4: the frictionless change, and the escape's speed.
  const SmallestCorrection correction =
      SolveSmallestCorrection(built.targets, approach_allowance_so_far, CorrectedQuantity::kVelocity);
  if (!correction.is_found) {
    return ApplyStick(active, built, velocity, approach_allowance_so_far, change, squeezing_solid_index);
  }
  assert(correction.used_count >= 1);
  const Vec3 frictionless_velocity = velocity + correction.change;
  if (math::Length(frictionless_velocity) > kMaxSqueezeOutSpeedFactor * largest_speed) {
    *squeezing_solid_index = change->approached_solid_index;
    return ContactError::kTooCloseToCall;
  }
  // Steps 5 and 6: the friction over the used directions.
  const JointFriction friction = ComputeJointFriction(active, built, correction, velocity);
  // Step 7: the result approaches no active surface beyond its allowance.
  const Vec3 total_change = friction.velocity - velocity;
  const double total_axes_sum = math::Abs(total_change.x) + math::Abs(total_change.y) + math::Abs(total_change.z);
  const double result_slack =
      math::Max(approach_allowance_so_far, kApproachLeftAfterResponseShare * (speeds + total_axes_sum));
  const bool is_result = !friction.moves_differently
      && ComputeIsWithinSlack(BuildJointTargets(active, friction.velocity).targets, result_slack);
  if (is_result) {
    change->velocity = friction.velocity;
    change->approach_allowance = result_slack;
    if (friction.is_single_response) {
      const int direction_index = correction.used_surfaces[0];
      ApplyCharge(change, active.surfaces[built.pushing_surfaces[direction_index]].solid_index, total_change);
      return ContactError::kNone;
    }
    for (int used_index = 0; used_index < correction.used_count; ++used_index) {
      const int direction_index = correction.used_surfaces[used_index];
      // Method paper <proc-move-particle>:  push_k = lambda_k n_k, plus the share of F
      const Vec3 push = correction.multipliers[used_index] * active.directions[direction_index];
      ApplyCharge(
          change,
          active.surfaces[built.pushing_surfaces[direction_index]].solid_index,
          push + friction.shares[used_index]
      );
    }
    return ContactError::kNone;
  }
  // Step 8: the friction shares go to the first solve's pushing solids; the second solve starts from v + F.
  for (int used_index = 0; used_index < correction.used_count; ++used_index) {
    const int direction_index = correction.used_surfaces[used_index];
    ApplyCharge(
        change,
        active.surfaces[built.pushing_surfaces[direction_index]].solid_index,
        friction.shares[used_index]
    );
  }
  const ContactError second_error = ApplySecondSolve(
      active,
      velocity + friction.change,
      approach_allowance_so_far,
      largest_speed,
      change,
      squeezing_solid_index
  );
  if (second_error != ContactError::kNone) {
    return second_error;
  }
  const double first_allowance =
      ComputeCorrectionSlack(built.targets, 0, CorrectedQuantity::kVelocity, correction.change);
  change->approach_allowance = math::Max(change->approach_allowance, first_allowance);
  return ContactError::kNone;
}

// Applies the charges of one joint change of velocity.for_next_step to the reaction, in ascending solid index, equal
// indexes in their order, at the particle's position (plan decision P20, item 12).
//
// Method: an insertion order over at most 2 * kMaxUsedDirections charges, then ApplyReactionChange for each. In
//   ascending solid index, so two reacting solids pushed by one response fail with kSecondReactingSolid naming the
//   higher index, whatever the order of the used directions.
// Requires: move and failure are not null.
// Fails: kTooCloseToCall  kSecondReactingSolid (ApplyReactionChange). *failure is written only on failure.
FLUIDITY_CPU_AND_GPU inline ContactError ApplyJointCharges(
    const SolidContactInputs& inputs,
    const JointChange& change,
    ParticleMove* move,
    ContactFailure* failure
) {
  assert(move != nullptr && failure != nullptr);
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  int charge_order[2 * kMaxUsedDirections]{};
  for (int charge_index = 0; charge_index < change.charge_count; ++charge_index) {
    int insert_index = charge_index;
    while (insert_index > 0
           && change.charged_solids[charge_order[insert_index - 1]] > change.charged_solids[charge_index]) {
      charge_order[insert_index] = charge_order[insert_index - 1];
      --insert_index;
    }
    charge_order[insert_index] = charge_index;
  }
  for (int order_index = 0; order_index < change.charge_count; ++order_index) {
    const int charge_index = charge_order[order_index];
    const ContactError reaction_error = ApplyReactionChange(
        inputs,
        change.charged_solids[charge_index],
        move->result.position,
        change.charges[charge_index],
        &move->result.reaction,
        failure
    );
    if (reaction_error != ContactError::kNone) {
      return reaction_error;
    }
  }
  return ContactError::kNone;
}

// Applies one hit to a particle's move, at its position: the record when it is the first, both velocities through the
// joint response over the active surfaces, the reaction, the allowances, and one more hit (plan decision P20).
//
// Method (method paper <proc-move-particle>, step 4):
//   1 More active surfaces or directions than one response reads, or a fifth hit, is kTooManyHits naming the record's
//     solid (CheckHitCountBelowLimit).
//   2 largest_speed: the largest Length of both velocities before the response and of every active solid velocity.
//   3 The first hit fills the record from record_surface and the velocities before: kind kHit, hit_fraction =
//     elapsed / step_duration, its normal, both approach speeds Dot(normal, velocity - u), its solid's index, its face
//     and, for a plate, the side.
//   4 velocity.for_next_step: ComputeJointChange with move->next_step_approach_allowance; its charges to the reaction
//     (ApplyJointCharges).
//   5 velocity.for_this_step: ComputeJointChange with move->this_step_approach_allowance. Its allowance b times the
//     time left must be at most tol / 64: with one active surface that is the bound of
//     ComputeLargestApproachLeftAfterResponse and an assertion (MoveParticle: Requires); with several it is checked,
//     and above it is kSqueezedBetweenSolids naming the solid for_this_step approached fastest: the escape is too fast
//     for the time left, which a shorter step cures.
//   6 Both allowances are raised to the response's, has_response becomes true, hit_count grows by one.
//   Why the touched rule holds after it for every solid, with the largest allowance so far: FindFirstHitOfSweep.
// Units: s for times; m and m/s, world.
// Example: the particle at {1 - tol, 0, 0} moving at {1, 0, 0} m/s against the fixed flat solid of surface x = 1 and
//   normal {-1, 0, 0}, 0.1 - tol s into a step of 0.2 s: both velocities {0, 0, 0}, approach speeds -1 m/s,
//   hit_fraction (0.1 - tol) / 0.2, allowance 2^-46 m/s, hit_count 1.
// Requires: record_surface.solid_index < inputs.solid_count; active was read at move's position and elapsed time;
//   move and failure are not null. The speeds involved, summed over their axes, times the time left are at most
//   2^40 tol.
// Fails: kTooCloseToCall  kTooManyHits (step 1), kSecondReactingSolid (a second reacting solid charged), or
//                         kSqueezedBetweenSolids (ComputeJointChange, or step 5).
//        *failure is written only on failure; *move is then unspecified.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <sec-contact-response>, <eq:contact-response>
FLUIDITY_CPU_AND_GPU inline ContactError ApplyJointResponse(
    const SolidContactInputs& inputs,
    const ActiveSurfaces& active,
    const ActiveSurface& record_surface,
    ParticleMove* move,
    ContactFailure* failure
) {
  assert(record_surface.solid_index < inputs.solid_count);
  assert(move != nullptr && failure != nullptr);
  SolidContactResult& result = move->result;
  // Steps: 1 limits  2 largest speed  3 record  4 for_next_step  5 for_this_step  6 allowances
  // Step 1: one response reads every surface at once, so more than it can read is too many.
  if (active.has_too_many) {
    return ApplyCloseCall(inputs, ContactCloseCall::kTooManyHits, record_surface.solid_index, failure);
  }
  const ContactError hit_limit_error = CheckHitCountBelowLimit(inputs, record_surface.solid_index, result, failure);
  if (hit_limit_error != ContactError::kNone) {
    return hit_limit_error;
  }
  // Step 2: the speeds before the response bound the escape of both velocities.
  const PerParticleVelocity<Vec3> before = result.velocity;
  double largest_speed = math::Max(math::Length(before.for_next_step), math::Length(before.for_this_step));  // m/s
  for (int surface_index = 0; surface_index < active.count; ++surface_index) {
    largest_speed = math::Max(largest_speed, math::Length(active.surfaces[surface_index].solid_velocity));
  }
  // Step 3: the record describes the first hit only.
  if (result.hit_count == 0) {
    const Vec3 normal = record_surface.normal_out_of_solid;
    const Vec3 solid_velocity = record_surface.solid_velocity;
    result.kind = ContactKind::kHit;
    result.hit_fraction = move->elapsed / inputs.step_duration;
    result.normal_out_of_solid = normal;
    result.approach_speed = {
        .for_next_step = math::Dot(normal, before.for_next_step - solid_velocity),
        .for_this_step = math::Dot(normal, before.for_this_step - solid_velocity),
    };
    result.solid_index = record_surface.solid_index;
    result.face = record_surface.face;
    result.side_at_hit = ToSideAtHit(inputs.solids[record_surface.solid_index], record_surface.face);
  }
  // Step 4: only for_next_step counts in the reaction, because it is the velocity the particle keeps.
  std::uint32_t squeezing_solid_index = kNoSolid;
  JointChange next_step_change;
  const ContactError next_step_error = ComputeJointChange(
      active,
      before.for_next_step,
      move->next_step_approach_allowance,
      largest_speed,
      &next_step_change,
      &squeezing_solid_index
  );
  if (next_step_error != ContactError::kNone) {
    return ApplyCloseCall(inputs, ContactCloseCall::kSqueezedBetweenSolids, squeezing_solid_index, failure);
  }
  const ContactError charge_error = ApplyJointCharges(inputs, next_step_change, move, failure);
  if (charge_error != ContactError::kNone) {
    return charge_error;
  }
  // Step 5: for_this_step moves the particle, so its allowance over the time left is bounded.
  JointChange this_step_change;
  const ContactError this_step_error = ComputeJointChange(
      active,
      before.for_this_step,
      move->this_step_approach_allowance,
      largest_speed,
      &this_step_change,
      &squeezing_solid_index
  );
  if (this_step_error != ContactError::kNone) {
    return ApplyCloseCall(inputs, ContactCloseCall::kSqueezedBetweenSolids, squeezing_solid_index, failure);
  }
  const double time_left = inputs.step_duration - move->elapsed;  // s
  const double largest_approach_gain = kTouchedSolidRestingGapShare / kMaxHitsPerStep * inputs.contact_tolerance;  // m
  const bool is_within_gain = this_step_change.approach_allowance * time_left <= largest_approach_gain;
  if (active.count > 1 && !is_within_gain) {
    const std::uint32_t approached_solid_index = this_step_change.approached_solid_index;
    return ApplyCloseCall(inputs, ContactCloseCall::kSqueezedBetweenSolids, approached_solid_index, failure);
  }
  assert(is_within_gain);
  // Step 6: one number per velocity covers every response so far.
  result.velocity = {
      .for_next_step = next_step_change.velocity,
      .for_this_step = this_step_change.velocity,
  };
  move->next_step_approach_allowance =
      math::Max(move->next_step_approach_allowance, next_step_change.approach_allowance);
  move->this_step_approach_allowance =
      math::Max(move->this_step_approach_allowance, this_step_change.approach_allowance);
  move->has_response = true;
  ++result.hit_count;
  return ContactError::kNone;
}

// Applies one hit of a sweep to a particle's move: moves the particle to the hit, then one joint response over every
// surface it is on there (plan decision P20, item 11).
//
// Method (method paper <proc-move-particle>, step 4):
//   1 A fifth hit fails (CheckHitCountBelowLimit).
//   2 The position becomes position + velocity_for_this_step * first_hit.elapsed and the elapsed time grows by
//     first_hit.elapsed: the expressions FindFirstHit evaluates the hit with, so the solid is posed at the same time.
//   3 The active surfaces there, the hit surface with the hit's normal (FindActiveSurfaces), and ApplyJointResponse
//     with the hit surface as the record's.
// Units: s for times; m and m/s, world.
// Example: the particle at {0.9, 0, 0} moving at {1, 0, 0} m/s, with the flat solid of surface x = 1 and normal
//   {-1, 0, 0} hit at 0.1 - tol s: position {1 - tol, 0, 0}, then the response of ApplyJointResponse's example.
// Requires: first_hit is a hit of FindFirstHitOfSweep on move's position and velocity_for_this_step, from
//   move->elapsed; move and failure are not null. As ApplyJointResponse for the speeds.
// Fails: kTooCloseToCall  a fifth hit (kTooManyHits, with the hit solid's solid_id), or as ApplyJointResponse.
//                         A shorter step has fewer hits.
//        *failure is written only on failure; *move is then unspecified.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline ContactError
ApplyHit(const SolidContactInputs& inputs, FirstHit first_hit, ParticleMove* move, ContactFailure* failure) {
  assert(first_hit.solid_index < inputs.solid_count);
  assert(move != nullptr && failure != nullptr);
  SolidContactResult& result = move->result;
  // Step 1: a particle that keeps hitting solids in one step is in a gap too narrow for this step length.
  const ContactError hit_limit_error = CheckHitCountBelowLimit(inputs, first_hit.solid_index, result, failure);
  if (hit_limit_error != ContactError::kNone) {
    return hit_limit_error;
  }
  // Step 2: these are FindFirstHit's own expressions, so the position and the pose are those of the hit.
  result.position = result.position + result.velocity.for_this_step * first_hit.elapsed;
  move->elapsed = move->elapsed + first_hit.elapsed;
  // Step 3: one response over every surface the particle is on, the hit surface included.
  const ActiveSurfaces active = FindActiveSurfaces(inputs, *move, &first_hit);
  const Solid& solid = inputs.solids[first_hit.solid_index];
  const ActiveSurface record_surface = {
      .solid_index = first_hit.solid_index,
      .face = first_hit.at_hit.face,
      .normal_out_of_solid = first_hit.at_hit.normal_out_of_solid,
      .solid_velocity = ComputeSolidVelocity(solid, ComputeSolidPose(solid, move->elapsed), result.position),
      .friction = inputs.friction * solid.friction_scale,
      .reports_reaction = solid.reports_reaction,
      .direction_index = 0,
  };
  return ApplyJointResponse(inputs, active, record_surface, move, failure);
}

// Checks the start of a particle's move against one solid, at the start as given and the pose of time 0: the checks
// of the start rules that fail (method paper <proc-move-particle>, step 2; plan decisions P3, P4 and P12).
//
// Method, with d the distance ComputeDistanceForParticle gives at the start, with the particle's plate side:
//   1 A particle committed to this plate and beyond its other face by more than tol fails
//     (CheckStartOnCommittedSide), before the depth: it is not inside its half space but on the wrong side.
//   2 d < -start_overlap_tolerance fails: too deep inside. The depth is read before any move: the move off near
//     surfaces could otherwise carry a start that is too deep out of the solid and hide it.
//   Same bits on CPU and GPU when the solid's angular velocity is zero.
// Units: m, world.
// Example: a fixed floor with normal {0, 0, 1} and start_overlap_tolerance 0.001: a start at {0, 0, -0.0005} passes; at
//   {0, 0, -0.01} it fails with magnitude 0.01.
// Requires: solid_index < inputs.solid_count; inputs passed CheckSolidContactInputs; failure is not null.
// Fails: kOnWrongSideOfPlate  as CheckStartOnCommittedSide.
//        kStartsInsideSolid   d < -start_overlap_tolerance: *failure names the solid_id, with the depth -d as the
//                             magnitude (plan decision P6). A particle placed inside a solid, or left there by an
//                             earlier step; the step stops.
//        *failure is written only on failure.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <sec-solid-shapes>, <sec-contact-tolerance>
FLUIDITY_CPU_AND_GPU inline ContactError
CheckStartOfSolid(const SolidContactInputs& inputs, std::uint32_t solid_index, Vec3 start, ContactFailure* failure) {
  assert(solid_index < inputs.solid_count);
  assert(failure != nullptr);
  // Steps: 1 wrong side of the plate  2 too deep inside
  const Solid& solid = inputs.solids[solid_index];
  const SolidPose start_pose = ComputeSolidPose(solid, 0);
  // Step 1: the specific diagnosis comes before the depth (plan decision P3).
  const ContactError side_error = CheckStartOnCommittedSide(inputs, solid_index, start_pose, start, failure);
  if (side_error != ContactError::kNone) {
    return side_error;
  }
  // Step 2: deeper than the user allows is a scene error, and the step stops.
  const double start_distance =
      ComputeDistanceForParticle(inputs, solid_index, start_pose, start, inputs.plate_side).distance;  // m
  if (start_distance < -inputs.start_overlap_tolerance) {
    *failure = {
        .error = ContactError::kStartsInsideSolid,
        .close_call = ContactCloseCall::kNone,
        .particle_id = 0,
        .solid_id = solid.solid_id,
        .magnitude = -start_distance,
    };
    return ContactError::kStartsInsideSolid;
  }
  return ContactError::kNone;
}

// Computes whether the start of a particle's move, as given, was inside one solid: its distance at the pose of time 0,
// with the particle's plate side, below 0 (plan decision P4).
//
// Requires: solid_index < inputs.solid_count.
FLUIDITY_CPU_AND_GPU inline bool
ComputeIsStartInsideSolid(const SolidContactInputs& inputs, std::uint32_t solid_index, Vec3 start) {
  assert(solid_index < inputs.solid_count);
  const SolidPose start_pose = ComputeSolidPose(inputs.solids[solid_index], 0);
  return ComputeDistanceForParticle(inputs, solid_index, start_pose, start, inputs.plate_side).distance < 0;
}

// Computes whether one surface comes before another in array order: by solid index, then by face (SolidFace order).
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsInArrayOrderBefore(const ActiveSurface& first, const ActiveSurface& second) {
  if (first.solid_index != second.solid_index) {
    return first.solid_index < second.solid_index;
  }
  return static_cast<int>(first.face) < static_cast<int>(second.face);
}

// Whether the start has a response, whether it was inside a solid, and the surface whose record the response writes.
struct StartResponse {
  bool has_response = false;
  bool is_inside = false;  // The start as given was inside some solid: the kind becomes kStartedInsideSolid.
  ActiveSurface record_surface{};  // The surface whose record the response writes, when has_response.
};

// Finds the record of the start's response (plan decision P20, item 10): the first, in array order (solid, then face),
// of the active surfaces whose solid the start was inside or that a velocity approaches (ComputeIsApproachingSolid);
// when none is, but the start was inside some solid, the surface that gives the first such solid's distance, read at
// the moved start.
//
// Method: the start has a response when it was inside a solid as given (d < 0), or when either velocity approaches an
//   active surface at the moved start, below 0 or below the rounding of a spinning solid's surface velocity. A solid
//   the start was inside but that is farther than tol + tol / 16 after the move is not an active surface; the response
//   then reads the surfaces that are, possibly none, and still counts as a hit with kind kStartedInsideSolid.
// Example: two fixed parallel floors at z = 0 and z = tol / 2 and a start at z = tol / 4, at rest: inside the one at
//   tol / 2, moved to z = 1.5 tol; that floor is the only active surface and the record's.
// Requires: move's position is the moved start and move.elapsed is 0; start is the start as given.
FLUIDITY_CPU_AND_GPU inline StartResponse FindStartResponse(
    const SolidContactInputs& inputs,
    const ActiveSurfaces& active,
    Vec3 start,
    const ParticleMove& move
) {
  StartResponse start_response;
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    start_response.is_inside = start_response.is_inside || ComputeIsStartInsideSolid(inputs, solid_index, start);
  }
  for (int surface_index = 0; surface_index < active.count; ++surface_index) {
    const ActiveSurface& surface = active.surfaces[surface_index];
    const bool is_inside = ComputeIsStartInsideSolid(inputs, surface.solid_index, start);
    const bool is_approached =
        ComputeIsApproachingSolid(inputs, surface.solid_index, surface.normal_out_of_solid, move);
    const bool is_first =
        !start_response.has_response || ComputeIsInArrayOrderBefore(surface, start_response.record_surface);
    if ((is_inside || is_approached) && is_first) {
      start_response.has_response = true;
      start_response.record_surface = surface;
    }
  }
  if (start_response.has_response || !start_response.is_inside) {
    return start_response;
  }
  const Vec3 position = move.result.position;
  std::uint32_t solid_index = 0;
  while (!ComputeIsStartInsideSolid(inputs, solid_index, start)) {
    ++solid_index;
  }
  const Solid& solid = inputs.solids[solid_index];
  const SolidPose pose = ComputeSolidPose(solid, 0);
  const DistanceToSolid at_position =
      ComputeDistanceForParticle(inputs, solid_index, pose, position, inputs.plate_side);
  start_response.has_response = true;
  start_response.record_surface = {
      .solid_index = solid_index,
      .face = at_position.face,
      .normal_out_of_solid = at_position.normal_out_of_solid,
      .solid_velocity = ComputeSolidVelocity(solid, pose, position),
      .friction = inputs.friction * solid.friction_scale,
      .reports_reaction = solid.reports_reaction,
      .direction_index = 0,
  };
  return start_response;
}

// Applies the start rules of every solid to a particle's move: step 2 of MoveParticle (plan decisions P4 and P12, both
// amended by P19 and P20).
//
// Method:
//   1 The checks that fail, every solid in array order, at the start as given and the pose of time 0
//     (CheckStartOfSolid): the wrong side of the committed plate, then the depth. The first solid that fails decides.
//   2 One move of the start off every near surface of every solid at once (MoveToRestGapOfNearSurfaces at time 0),
//     whatever the velocity: with one near solid along its normal, as P12 always moved it. Why one move and not one per
//     solid in turn: in a wedge sharper than a right angle each solid's move pushes the start toward the other, so a
//     start resting on both faces failed at every step length (2000 of 2000 at 60 and at 30 degrees; 8 % of the
//     generated starts at 85 degrees, 49 % at 60, 98 % at 5), and in the cusp under a ball resting on a floor the angle
//     goes to 0. No move found is kStartsInNarrowGap.
//   3 The active surfaces at the moved start (FindActiveSurfaces), read there and not at the start as given: the
//     sweep that follows reads the normal at the moved start, and the response must leave no approach along that one.
//     A surface near the start as given but farther than tol + tol / 16 after the move is not in the response: of two
//     parallel floors tol / 4 apart, only the higher one, which the particle rests on, responds and is charged.
//   4 The start has a response when it was inside a solid as given, or a velocity approaches an active surface
//     (FindStartResponse): one joint response at time 0 (ApplyJointResponse), one hit however many surfaces, and kind
//     kStartedInsideSolid when the start was inside. Otherwise the move was silent: no hit, no record, and the sweeps
//     keep the plain touching rule (plan decision P12: a touching particle that leaves is free).
// Units: m, m/s and s, world.
// Example: a fixed floor and an upright flat solid at x = 1, a particle at {1 - tol / 2, 0, tol / 2} moving at
//   {1, 0, -1} m/s, friction 0: the start moves to {1 - tol, 0, tol}; one response over both: kind kHit, hit_count 1,
//   both velocities {0, 0, 0}. A particle at rest at the apex of the fixed floor and a fixed face at 60 degrees moves
//   to {Sqrt(3) tol, 0, tol}, silently.
// Requires: inputs passed CheckSolidContactInputs; move and failure are not null; move's position is the start, its
//   elapsed time 0 and it has no response yet. As ApplyJointResponse for the speeds.
// Fails: kOnWrongSideOfPlate, kStartsInsideSolid  as CheckStartOfSolid, for the first solid that fails.
//        kStartsInNarrowGap  the start has no point at the rest gap of its near surfaces within the move cap
//                            (MoveToRestGapOfNearSurfaces). *failure names the solid latest in array order of every
//                            surface that constrained the move, magnitude 0. A scene error that no sweep can lead to:
//                            two faces facing each other closer than 2 tol, a start within rounding of the contact
//                            point of a ball resting on a floor, a pocket too small for a point at tol from each side,
//                            or a start deep at the apex of a flat wedge sharper than about 2^-21 rad: the grown solids
//                            overlap there, so a sweep stops a moving particle before it. Also a start near surfaces
//                            facing more than four different ways, which one move cannot read at once, although a
//                            point may fit (surfaces with the same normal count once, up to kMaxTrackedSurfaces = 16
//                            surfaces in all). A known limit, beside the start near the contact point: a wall within
//                            the rounding window of a cusp's rest point, ulp(R) / (2 Sqrt(tol / R)) (0.0014 tol at R =
//                            0.5 m, 0.25 tol at 16 m, 128 tol at 1 km), where the ball's rounded distance moves the
//                            linearised vertex past the wall and no subset has an answer although the rest point fits.
//                            Only a start placed there reaches it, and a shorter step does not help.
//        kTooCloseToCall     as ApplyJointResponse: kSecondReactingSolid, kSqueezedBetweenSolids, or kTooManyHits only
//                            when more than 8 directions or 16 surfaces are active (the start alone never has too
//                            many hits: it is one).
//        *failure is written only on failure; *move is then unspecified.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <fig-move-off-near-surfaces>
FLUIDITY_CPU_AND_GPU inline ContactError
ApplyStartRules(const SolidContactInputs& inputs, ParticleMove* move, ContactFailure* failure) {
  assert(move != nullptr && failure != nullptr);
  assert(!move->has_response && move->elapsed == 0);
  // Steps: 1 the checks that fail  2 one move off the near surfaces  3 the active surfaces  4 the response
  const Vec3 start = move->result.position;
  // Step 1: at the start as given, so the move cannot hide a start that is too deep.
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const ContactError error = CheckStartOfSolid(inputs, solid_index, start, failure);
    if (error != ContactError::kNone) {
      return error;
    }
  }
  // Step 2: a scene error when no point of the gap is at the rest gap of every near surface.
  const RestGapMove start_move = MoveToRestGapOfNearSurfaces(inputs, 0, start);
  if (!start_move.is_found) {
    assert(start_move.latest_solid_index < inputs.solid_count);
    *failure = {
        .error = ContactError::kStartsInNarrowGap,
        .close_call = ContactCloseCall::kNone,
        .particle_id = 0,
        .solid_id = inputs.solids[start_move.latest_solid_index].solid_id,
        .magnitude = 0,
    };
    return ContactError::kStartsInNarrowGap;
  }
  move->result.position = start_move.moved_point;
  // Step 3: plan decision P20, item 10: at the moved start.
  const ActiveSurfaces active = FindActiveSurfaces(inputs, *move, nullptr);
  // Step 4: a start inside, or an approach, is one hit at time 0; a particle that leaves was moved silently.
  const StartResponse start_response = FindStartResponse(inputs, active, start, *move);
  if (!start_response.has_response) {
    return ContactError::kNone;
  }
  const ContactError response_error = ApplyJointResponse(inputs, active, start_response.record_surface, move, failure);
  if (response_error != ContactError::kNone) {
    return response_error;
  }
  if (start_response.is_inside) {
    move->result.kind = ContactKind::kStartedInsideSolid;
  }
  return ContactError::kNone;
}

// Applies the settle at the end of a particle's move: one move off every near surface of every solid, touched or not,
// at the solids' pose at the end of the step (MoveToRestGapOfNearSurfaces), to the rest gap of all of them (plan
// decision P19, item 7).
//
// Method: the distances and normals are read afresh at the end pose, not the normals recorded at the hits, because a
//   moving solid's surface has moved and a turning one's normal has turned since. Every solid, not only the touched
//   ones: one move off several surfaces leaves out a near solid it does not read, and could push the particle into it
//   (in the cusp under a ball touching a wall, the settle off the touched ball alone ended 0.43 tol from the untouched
//   wall); and a sweep may bring the particle within tol of a solid it never touched, which the settle lifts back to
//   the rest gap, as plan decision P12 promises. When no move is found within the cap (two surfaces closer than 2 tol
//   at the end, a solid that closed on the particle), the particle stays where it is and the end check decides
//   (CheckEndOutsideSolids). The settle only moves the particle away from every near surface: a point it returns is at
//   least tol - tol / 128 from every solid, so the sweeps' bound (any solid approached by at most tol / 16 in the step
//   after a response, FindFirstHitOfSweep) is untouched.
//   Cost: a particle near no solid at the end pays one distance per solid here, as many as the end check reads after
//   it; a particle that moves pays the iterates of MoveToRestGapOfNearSurfaces.
// Units: m, world.
// Example: a particle that ends a slide along a fixed floor with normal {0, 0, 1} at z = tol / 4 ends at z = tol. At
//   {Sqrt(3) / 2 tol, 0, tol / 2}, tol / 2 from the fixed floor and from a fixed face at 60 degrees, it ends at
//   {Sqrt(3) tol, 0, tol}, touched or not.
// Requires: move is not null.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline void ApplySettleAtEnd(const SolidContactInputs& inputs, ParticleMove* move) {
  assert(move != nullptr);
  const Vec3 end_position = move->result.position;
  // Every solid that does not turn ends at d >= tol - tol / 16 up to rounding: the start rules leave it at the rest gap
  // or outside it, a sweep stops at its grown surface, and the touched rule lets the particle approach it by at most
  // tol / 16 in the step (FindFirstHitOfSweep). Deeper than -2 tol means a hit at a wrong pose or position, which the
  // move below would silently repair, so a debug build stops. The loop holds only assertions, so a release build drops
  // it. A turning solid is left out: its stepping search stops between tol and 1.25 tol, but no bound is proved for it.
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    assert(
        ComputeIsTurningSolid(inputs, solid_index)
        || ComputeDistanceForParticle(
               inputs,
               solid_index,
               ComputeSolidPose(inputs.solids[solid_index], inputs.step_duration),
               end_position,
               inputs.plate_side
           )
                .distance
            >= -2 * inputs.contact_tolerance
    );
  }
  const RestGapMove settle = MoveToRestGapOfNearSurfaces(inputs, inputs.step_duration, end_position);
  if (settle.is_found) {
    move->result.position = settle.moved_point;
  }
}

// Checks that the end position of a particle's move is at least half a contact tolerance from every solid, at each
// solid's pose at the end of the step, and names the first solid that is closer.
//
// Method: the end check of method paper <sec-contact-tolerance>. For each solid in array order, the distance
//   ComputeDistanceForParticle gives at the end position and the pose of step_duration, with the particle's plate
//   side; the first one that is not at least tol / 2 fails, a NaN distance included. This check keeps the module's
//   promise that a particle never ends a step inside a solid, with room to spare: a distance evaluated later, by any
//   module and with its own rounding, is off by at most tol / 128 (contact/contact_tolerance.h), so it is still
//   positive. The particle rests at tol, twice the limit, so the check fails only when a solid closes on it during the
//   step faster than the hits and the settle could keep it at tol.
// Units: m, world.
// Example: a fixed floor with normal {0, 0, 1} through the origin and the end {0, 0, 0.25 tol}: fails, magnitude
//   0.25 tol. An end at z = tol / 2 passes.
// Requires: failure is not null.
// Fails: kTooCloseToCall  a solid is closer than tol / 2 at the end of the step: *failure is kEndsTooCloseToSolid with
//                         that solid's solid_id and the distance d as the magnitude (plan decision P6). A shorter step
//                         gives the hits a chance to keep the particle out.
//        *failure is written only on failure.
// Read: docs/method-paper/fluidity-method.typ <sec-contact-tolerance>, <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline ContactError
CheckEndOutsideSolids(const SolidContactInputs& inputs, Vec3 end_position, ContactFailure* failure) {
  assert(failure != nullptr);
  const double least_end_distance = inputs.contact_tolerance / 2;  // m
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    const Solid& solid = inputs.solids[solid_index];
    const SolidPose end_pose = ComputeSolidPose(solid, inputs.step_duration);
    const double end_distance =
        ComputeDistanceForParticle(inputs, solid_index, end_pose, end_position, inputs.plate_side).distance;
    const bool is_far_enough = end_distance >= least_end_distance;
    if (!is_far_enough) {
      *failure = {
          .error = ContactError::kTooCloseToCall,
          .close_call = ContactCloseCall::kEndsTooCloseToSolid,
          .particle_id = 0,
          .solid_id = solid.solid_id,
          .magnitude = end_distance,
      };
      return ContactError::kTooCloseToCall;
    }
  }
  return ContactError::kNone;
}

}  // namespace fluidity::contact
