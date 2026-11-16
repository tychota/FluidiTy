// Solid contact for one particle: the first hit of a straight path against the solids, and the particle's move for
// one step.
//
// Purpose: find when a particle's straight path first comes within the contact tolerance of a solid, and which solid
//   and face it meets; and move one particle for one step without letting it end inside a solid. The types they read
//   and return, and the response at a hit, are in contact/solid_contact_types.h; the work on one solid and the steps
//   of the move are in contact/solid_contact_steps.h.
//
// Idea: during a step a particle moves in a straight line, the sweep. Every sweep runs against the
//   solids grown by the contact tolerance, so the hit is the first time the gap, the distance minus the tolerance,
//   reaches zero. For a solid that does not turn the gap to a face changes linearly with time, so the hit time is the
//   gap divided by the speed at which it closes. A particle that already rests on a surface is not hit by it again
//   unless the path pushes into it (the touching rule, contact/solid_contact_steps.h). Toward a turning solid the
//   sweep steps forward instead, each advance no longer than the gap allows (the stepping search).
//   Example: a flat solid with surface x = 1 and normal {-1, 0, 0}, and a path from x = 0.9 at {1, 0, 0} m/s: gap
//   0.1 - tol, closed at 1 m/s, so the hit is at 0.1 - tol s.
//   A particle's move strings sweeps together. At a hit the particle stops one tolerance short of the surface, its
//   speed into the solid is removed and friction slows its sideways speed (the response); then it sweeps again from
//   there for the time left, along the surface. Papers call the sweep continuous collision detection.
//   Example: the same flat solid, friction 0, and a step of 0.2 s: the particle stops at x = 1 - tol at 0.1 - tol s,
//   with speed 0, and stays there for the rest of the step.
//
// Words:
//   sweep           the straight path of a particle during all or part of a step, start + velocity * t for
//                   0 <= t <= duration, tested against the solids grown by the contact tolerance
//   gap             m. The distance from a particle to a solid minus the contact tolerance: zero at a hit, negative
//                   inside the grown solid. Not the rest gap, which is the distance kept at rest (tol): at rest the
//                   gap is 0
// Owns:    FindFirstHit, MoveParticle
// Uses:    Vec3 (base/vec3.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), ContactError, ContactCloseCall,
//          ContactFailure (contact/contact_error.h), Solid, kNoSolid (contact/solid_shapes.h), SolidContactInputs,
//          SolidContactResult, FirstHit (contact/solid_contact_types.h), Sweep, TouchedSolid, FindFirstHitOfSweep,
//          ParticleMove, CheckSolidContactInputs, ApplyStartRules, ApplyHit, ApplySettleAtEnd,
//          CheckEndOutsideSolids (contact/solid_contact_steps.h)
// Used by: the MoveParticles stage
// Read:    docs/method-paper/fluidity-method.typ <sec-first-hit>, <eq:hit-time>, <proc-find-first-hit>,
//          <fig-sweep-hit-slide>, <proc-move-particle>, <sec-contact-response>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// Finds the first time the straight path start + velocity_for_this_step * t, 0 <= t <= duration, comes within the
// contact tolerance of a solid, with the solids posed start_elapsed + t seconds into the step, and says why not when
// that cannot be answered safely.
//
// Method: first every solid that does not turn, in array order; then every turning solid, in array order, each
//   searched only up to the earliest hit found so far; then the earliest hit (method paper <proc-find-first-hit>):
//   1 Solid that does not turn: the gap to each face plane changes linearly, gap(t) = gap(0) + rate * t with
//     rate = Dot(normal, velocity_for_this_step - linear_velocity). The touching rule comes first, on the surface
//     that gives the distance at the start, with the rate measured from the surface's velocity there
//     (ComputeSolidVelocity, which a spin adds to linear_velocity): a gap in [-tol, 0] is a hit at 0 when
//     rate * duration < -tol and no hit otherwise; a gap below -tol is a hit at 0. Otherwise a flat solid, a plate or a
//     box is hit where the path enters the grown solid through its face planes, the latest crossing into a plane if no
//     plane is left before it, with the entering face; each closed side of the domain box is a flat face of its own. A
//     plate seen by a particle committed to one of its sides (inputs.plate_side) is the half space behind its committed
//     face, so it has that one face (plan decision P3). A sphere and a cylinder use their true distance and normal in
//     world axes for the touching rule (plan decision P16), as a flat face does, then a closed form in their own axes:
//     the first root of the path on the sphere grown by tol, or the entry into the cylinder grown by tol with sharp
//     edges, the later of its entries into the grown wall and between the grown caps' planes when it comes before both
//     exits (plan decision P15). On the wall or a cap proper that entry is the hit; beside a rim the rim search goes on
//     from it on the true distance, the gap over the speed at which it closes, until the gap is at most tol / 1024
//     (plan decision P21). Before the closed form, a start beside a cylinder's rim that rests on the wall or the cap's
//     plane, gap in [-tol / 8, tol / 16] and sinking at most tol / 64 during the sweep, is free: the whole cylinder
//     lies behind that bounding surface. A hit counts when its time is at most duration.
//   2 Turning solid: a box whose angular velocity is not exactly {0, 0, 0} (plan decision P7), a flat solid or a plate
//     whose angular velocity has a part across its normal longer than kNegligibleTurnShare of it, or a cylinder whose
//     axis tilts by more than plan decision P17's bound. Its gap is not linear in time, so the stepping search advances
//     from t = 0. With bound = Length(velocity_for_this_step - linear_velocity) + Length(angular_velocity) * r_max,
//     r_max the larger distance from the path's two ends to the solid's origin at those times, the gap cannot shrink
//     faster than bound, so the first-order advance g(t) / bound passes no crossing. A shape whose distance is convex
//     (every turning shape here) also has the second-order advance (plan decision P18): the positive root of
//     g + gdot s - K s^2 / 2, with gdot the gap rate at the sample and K = Length(angular_velocity) *
//     (2 * Length(velocity_for_this_step - linear_velocity) + Length(angular_velocity) * r_max), which bounds how
//     fast the gap rate changes; the larger of the two advances is taken. The search stops with a hit when
//     g(t) <= tol / 4 (kTurningSolidHitGapShare), after at least one advance at a distance in [tol, 1.25 tol], and
//     with no hit once t >= duration; a few advances usually suffice. There is no touching rule: a particle resting on
//     a turning solid is hit at 0. A spin about a sphere's centre, a cylinder's axis or the normal of a flat solid or
//     a plate is not a turn. The search's duration is capped at the earliest hit of the solids searched before it: a
//     later hit could not win, and a search that reaches the cap is free.
//   3 The smallest hit time wins, and equal times keep the lower solid index whatever the order of the search.
//     *first_hit holds its time, the solid's index and at_hit: the distance, normal and face at the hit position (one
//     tolerance from the entering face up to rounding; between tol and tol + tol / 1024 from a cylinder's rim, whose
//     face it then reports; between tol and 1.25 tol from a turning solid), or, for a hit at 0, the distance at the
//     start.
//     A free path gives solid_index kNoSolid, elapsed 0 and at_hit at its defaults.
//   The inputs are not checked here (plan decision P9). Same bits on CPU and GPU when every solid's angular velocity
//   is zero: only + - * /, Abs, Sqrt and comparisons. A turning or spinning solid is posed with Sin, which may differ
//   in the last bits, so a turning solid's hit time may differ too.
// Units: m and m/s, world, for the path; s for start_elapsed, duration and the result's elapsed, which counts from
//   start_elapsed.
// Ties: equal hit times keep the lower solid index, then the face that comes first in face order.
// Example: a fixed flat solid with surface x = 1 and normal {-1, 0, 0}, contact tolerance 2^-30 m, and the path from
//   {0.9, 0, 0} at {1, 0, 0} m/s for 0.2 s: elapsed (1 - 0.9) - 2^-30 = 0.0999999990686774 s, solid_index 0, face
//   kHighZ, normal {-1, 0, 0}, distance 2^-30 m up to rounding. The same path starting at {1 - 2^-30, 0, 0} moving at
//   {0, 0, 1} m/s rests on the surface and is free; at {1, 0, 0} m/s it pushes in and is hit at 0. A fixed sphere of
//   radius 0.5 m at the origin and the path from {-2, 0.3, 0} at {1, 0, 0} m/s for 2 s: elapsed
//   2 - Sqrt((0.5 + 2^-30)^2 - 0.09) = 1.5999999988358469 s, face kCurved, normal near {-0.8, 0.6, 0}. A box of half
//   size 0.5 m at the origin turning at {0, 0, 1} rad/s and a particle at rest at 0.6 * {cos 0.3, sin 0.3, 0} m for
//   1 s: after 3 advances, elapsed 0.8857 s on kHighX, 1.4e-10 s before 0.3 + acos((0.5 + 2^-30) / 0.6).
// Requires: first_hit is not null. inputs.solids points to inputs.solid_count solids (it may be null when the count
//   is 0), each of which passed CheckSolid for inputs.step_duration. start and velocity_for_this_step are finite.
//   start_elapsed and duration are finite and zero or more, and duration is at most inputs.step_duration -
//   start_elapsed as computed (the time left in the step), so every pose stays within the step up to one rounding
//   unit: start_elapsed + (step_duration - start_elapsed) can exceed step_duration by one.
//   inputs.contact_tolerance is finite and zero or more. When inputs.plate_side.plate_id is the solid_id of a
//   solid, that solid is a plate and the side is kAlongNormal or kAgainstNormal (an id that names no solid sees
//   every plate with both faces; MoveParticle refuses it).
// Fails: kTooCloseToCall  the search toward a turning solid needs more than kMaxAdvanceCount advances before the
//                         earliest hit of the other solids: the path hovers near it or grazes it, with too little
//                         motion along its normal, which is rare (FindHitOnTurningSolid: Cost). *close_call is
//                         kSlowApproachToTurningSolid, and *first_hit is FirstHit{} but for solid_index, the index of
//                         that turning solid, so the caller can name it; the caller shortens the step and tries again.
//        kBadInput        a solid is of an unknown shape; a debug build stops at an assertion first. *first_hit is
//                         unchanged.
//        When close_call is not null, *close_call is always written: kNone unless the error is kTooCloseToCall.
// Read: docs/method-paper/fluidity-method.typ <sec-first-hit>, <eq:hit-time>, <proc-find-first-hit>,
//       <fig-sweep-hit-slide>, <sec-round-solids>
//
// clang-tidy: start_elapsed and duration follow the path, as contact design 3.4 orders the parameters; the
// swappable-parameters check flags the two doubles.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU inline ContactError FindFirstHit(
    const SolidContactInputs& inputs,
    Vec3 start,
    Vec3 velocity_for_this_step,
    double start_elapsed,
    double duration,
    FirstHit* first_hit,
    ContactCloseCall* close_call
) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  // FindFirstHitOfSweep asserts the preconditions.
  // The sweep with no earlier hit in the step: no solid gets the touching rule of a touched solid.
  const Sweep sweep = {
      .start = start,
      .velocity_for_this_step = velocity_for_this_step,
      .start_elapsed = start_elapsed,
      .duration = duration,
  };
  const TouchedSolid not_touched;
  return FindFirstHitOfSweep(inputs, sweep, not_touched, first_hit, close_call);
}

// Finds the first hit as the seven-argument form does, without saying why a call was too close to call.
//
// Example: the example of the seven-argument form gives the same first hit.
// Fails: as the seven-argument form.
// Read: docs/method-paper/fluidity-method.typ <proc-find-first-hit>
// NOLINTBEGIN(bugprone-easily-swappable-parameters): as the seven-argument form.
FLUIDITY_CPU_AND_GPU inline ContactError FindFirstHit(
    const SolidContactInputs& inputs,
    Vec3 start,
    Vec3 velocity_for_this_step,
    double start_elapsed,
    double duration,
    FirstHit* first_hit
) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  return FindFirstHit(inputs, start, velocity_for_this_step, start_elapsed, duration, first_hit, nullptr);
}

// Moves one particle for one step and stops it at solid surfaces: it sweeps to the first hit, responds, and sweeps
// again from the hit with the time left, until the path is free, the step is used or the hit limit is passed.
//
// Method (method paper <proc-move-particle>):
//   1 Check inputs (CheckSolidContactInputs).
//   2 Classify the start (ApplyStartRules, plan decisions P3, P4, P12, P19 and P20), at the pose of time 0, in three
//     parts. First the checks that fail, every solid in array order at the start as given; the first that fails
//     decides: a particle beyond the other face of its committed plate by more than tol, then a start deeper inside
//     than start_overlap_tolerance. Then one move of the start off every surface closer than tol, d < tol, of every
//     solid at once, whatever its velocity (MoveToRestGapOfNearSurfaces): to the point at the rest gap of all of them
//     that the solver finds nearest the start, within a move cap of 2^22 tol; with one near solid, along its normal to
//     the rest gap; a domain box's closed sides closer than tol each count as a surface of their own (a corner has two
//     or three). So a start in a wedge, a corner or the cusp under a ball resting on a floor rests at tol from every
//     face, where moving off one face at a time would push it toward the other. No such point within the cap is
//     kStartsInNarrowGap. Then the start has a response when it was inside a solid as given, d < 0, or when either
//     velocity approaches an active surface at the moved start (below 0, or below the rounding of a spinning solid's
//     surface velocity; a solid passed at a bounding surface's level has none): one joint response at time 0, as in
//     step 4, with the active surfaces read at the moved start, one hit however many surfaces, the record with
//     hit_fraction 0, and kind kStartedInsideSolid when the start was inside a solid. Otherwise the move is silent:
//     nothing else changes, and a touching particle that leaves is free. A turning solid gets the same rules, then step
//     4's departure below.
//   3 Find the first hit (FindFirstHitOfSweep) from the current position along velocity.for_this_step, with the solids
//     posed at the time already used and duration = step_duration - elapsed, computed afresh before every sweep. After
//     the first response of the step every solid that does not turn has the touching rule of a touched solid, on every
//     surface, after the bounding surfaces a start may rest on: a start gap up to tol / 16 rests on it, and it is hit
//     at 0 only when the path approaches it faster than the largest allowance of the responses so far. So rounding does
//     not make a solid hit again, while a solid that is really approached again, by a push from another solid or
//     because it moves, is hit. With no hit the particle advances by velocity.for_this_step times that duration, and
//     the hits end.
//   4 The joint response (ApplyHit, ApplyJointResponse, plan decision P20): the particle moves to the hit and nothing
//     moves it further, a rim hit included, which the rim search leaves within tol / 1024 of the rest gap (plan
//     decision P21). Both velocities get one response over every active surface: the surfaces of every solid within
//     tol + tol / 16 of the particle, read afresh, except a solid it passes at a bounding surface's level (level with
//     one that does not give the solid's distance, plan decision P21, item 16), and the surface hit, with the hit's
//     normal. Exact copies are dropped, and the rest sorted by value, so the response is the same bits in every order
//     of the solids array. One active
//     surface: the single-surface response, ComputeContactResponse with the solid velocity and friction times its
//     friction_scale. Several: the smallest frictionless change that leaves the velocity approaching none of them
//     (SolveSmallestCorrection), then friction from each used surface's push, the friction limit friction * push: with
//     one used direction and friction the single-surface response itself; with one solid velocity the limits add, and
//     the particle sticks or slips toward that velocity; with solids moving differently, the mean over the orders of
//     their own frictions; then, when the result approaches an active surface or the solids move differently, a second
//     frictionless change from the velocity plus the friction. Two fixed faces keep the part along their common line,
//     so a particle in a groove slides on, and one pushed into a cusp or an acute wedge stops; in a wedge of 120
//     degrees, friction 0, {-1, 1, -1} gives {0, 1, 0}. No change found (a pair within 2^-40 of opposite, or solids
//     closing): the particle sticks to the surface it approaches fastest. The first hit fills the record, each push and
//     friction share of velocity.for_next_step counts in its own solid's reaction (plan decision P5), and the
//     response's allowance raises the move's, one number per velocity. The sweep starts again at step 3 with the new
//     velocity.for_this_step. On one convex solid that does not turn (a sphere, a cylinder, a box) that sweep is free
//     of it: the response leaves the path tangent to the surface or leaving it, measured from the surface's velocity,
//     the touching rule reads that same approach (plan decision P16), and a convex solid lies behind its tangent plane,
//     so the straight path cannot enter it again in the step; a lone round solid takes one hit, far from
//     kMaxHitsPerStep. Departure from the method paper's step 4, for now: a turning solid has no touching rule,
//     so a particle it hit, or one that starts within 1.25 tol of it, is hit again at time 0 by every later sweep and
//     the move ends in kTooManyHits instead of sliding in the solid's axes. A shorter step does not help; only a hit
//     exactly at the end of the step succeeds.
//   5 Settle and check the end (ApplySettleAtEnd, CheckEndOutsideSolids): one move off every surface closer than tol
//     of every solid at the end pose, to the rest gap of all of them (MoveToRestGapOfNearSurfaces); when there is none
//     the particle stays. Then every solid must be at least tol / 2 away: the end check stays the last word, so a
//     particle never ends a step inside a solid. Unlike the response, the move off near surfaces at the start and the
//     settle read the solids in array order, so a reordered array or an exact copy can change the position by a unit
//     in the last place (MoveToRestGapOfNearSurfaces, Order), a known limit.
//   Without a hit the end position is start_position + velocity.for_this_step * step_duration, one multiply and one
//   add per axis, and both velocities come back unchanged. Same bits on CPU and GPU when every solid's angular
//   velocity is zero: only + - * /, Abs, Sqrt, Length and comparisons.
// Units: m, m/s and s, world.
// Order: solids in array order, ascending solid_id; equal hit times keep the lower index (FindFirstHit).
// Example: a fixed flat solid with surface x = 1 and normal {-1, 0, 0}, contact tolerance 2^-30 m, friction 0, and a
//   particle at {0.9, 0, 0} moving at {1, 0, 0} m/s (both velocities) for 0.2 s: kind kHit, hit_fraction
//   (0.1 - 2^-30) / 0.2 = 0.4999999953433871, face kHighZ, normal {-1, 0, 0}, approach speeds -1 m/s, end
//   {1 - 2^-30, 0, 0} = {0.9999999990686774, 0, 0} m, both velocities {0, 0, 0}, hit_count 1. A fixed floor
//   with normal {0, 0, 1} and friction 0.5: a particle at {0.2, 0.5, 0.1} moving at {3, 0, -1} m/s for 0.2 s hits it
//   at 0.1 - 2^-30 s, slips on at {2.5, 0, 0} m/s and ends at x = 0.75 - 2^-31, z = 2^-30. The fixed floor with
//   friction 0 and a particle at {0, 0, 2^-32} moving away at {1, 0, 2} m/s for 0.1 s: moved silently to z = 2^-30,
//   it ends at {0.1, 0, 0.2 + 2^-30} with kind kNone and both velocities unchanged; at {0, 0, -0.0005} and at rest,
//   with start_overlap_tolerance 0.001, it moves to z = 2^-30 with kind kStartedInsideSolid and stays there.
// Requires: result is not null. Each solid passed CheckSolid for inputs.step_duration (plan decision P9). At every
//   hit on one active surface, the sum of the absolute axes of velocity.for_this_step and of the solid velocity, times
//   the time left, is at most 2^40 tol (asserted), so that its allowance, 2^-46 times that sum, times the time left is
//   at most tol / 64, and the particle approaches any solid by at most tol / 16 over the step (with several active
//   surfaces the same bound is checked: kSqueezedBetweenSolids). With the contact tolerance of ComputeContactTolerance,
//   at least 2^-40 of the largest coordinate, it holds whenever those speeds times the step stay within the largest
//   coordinate.
// Fails: kBadInput            a value is not finite or out of range, the solid_ids are not strictly ascending, or the
//                             plate side names no plate (CheckSolidContactInputs).
//        kOnWrongSideOfPlate  the start is beyond the other face of its committed plate by more than tol; the
//                             failure names the plate, with that distance (CheckStartOnCommittedSide). A scene
//                             error; the step stops.
//        kStartsInsideSolid   the start is deeper inside a solid than start_overlap_tolerance; the failure names it,
//                             with the depth (CheckStartOfSolid). A scene error; the step stops.
//        kStartsInNarrowGap   the start has no point at the rest gap of every surface it is near within the move cap
//                             (MoveToRestGapOfNearSurfaces): two solids facing each other less than 2 tol apart, a
//                             start within rounding of the contact point of a ball resting on a floor, a wall within
//                             the rounding window of such a contact's rest point (a known limit, about 128 tol at
//                             R = 1024 m), a pocket too small for a point at tol from each of its sides, near surfaces
//                             facing more than four different ways (one move reads at most four directions) or more
//                             than 16 surfaces (kMaxTrackedSurfaces), or a start deep at the apex of a flat wedge
//                             sharper than about 2^-21 rad. The failure names the near solid latest in array order,
//                             magnitude 0. A scene error that no sweep can lead to, since the grown solids overlap
//                             there, except the cases where a point fits (the known limit, the directions and the
//                             count); the step stops.
//        kTooCloseToCall      no safe answer for this step; *failure's close_call says why: kTooManyHits (a fifth hit
//                             along the path, or more than 8 directions or 16 surfaces active at one hit; the start
//                             alone never has too many hits, it is one), kSqueezedBetweenSolids (no velocity leaves
//                             every active surface and the stick still approaches one, the escape is faster than
//                             kMaxSqueezeOutSpeedFactor times the speeds involved, or its rounding over the time left
//                             is above tol / 64; each names the solid the failing velocity approaches fastest),
//                             kSecondReactingSolid (one step charges two solids with reports_reaction, naming the
//                             higher index when one response pushes both), kEndsTooCloseToSolid (the end check), or
//                             kSlowApproachToTurningSolid (passed on from the sweep, with that turning solid's id). The
//                             step shortens the step and tries again, which helps when the failure comes from the
//                             sweeps.
//        Failures a shorter step cannot fix: the scene errors kOnWrongSideOfPlate, kStartsInsideSolid and
//        kStartsInNarrowGap; the close calls that the start alone decides: two reacting solids pushed by the start's
//        response, and a start within 1.25 tol of a turning solid (until its slide is built, plan Task 13); more than 8
//        directions or 16 surfaces active at once; a particle in the crease of two reacting solids that pushes into
//        both (a question for the owner under P5); and the squeezes that the geometry and the speeds decide, solids
//        closing on the particle or an escape above the speed cap (one face closing on a fixed one at under
//        0.8953 degrees, two at under 1.7906). A particle that zig-zags into the neck of two touching balls can need a
//        step shortened in proportion to Sqrt(R tol) / speed to keep within four hits, and four moving faces within a
//        few tol of each other but not at one rest point are met one after the other (13 of 283,929 such corners fail
//        at every step length). The step module decides what a squeeze that lasts does. Only the squeeze whose rounding
//        over the time left is too large is cured by a shorter step. On failure *result is unchanged. When failure is
//        not null, *failure is always written: ContactFailure{} on success; on failure the error, the close call, the
//        solid's solid_id (never its index) and the magnitude of plan decision P6. particle_id stays 0.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <fig-sweep-hit-slide>, <sec-contact-response>,
//       <sec-contact-tolerance>
FLUIDITY_CPU_AND_GPU inline ContactError
MoveParticle(const SolidContactInputs& inputs, SolidContactResult* result, ContactFailure* failure) {
  assert(result != nullptr);
  // Steps (method paper <proc-move-particle>): 1 check inputs  2 classify the start  3 find the first hit
  //   4 joint response  5 settle and check the end
  // Every step writes its failure here; *failure gets it at the end, and stays ContactFailure{} on success.
  ContactFailure move_failure;
  ParticleMove move;
  move.result.position = inputs.start_position;
  move.result.velocity = inputs.velocity;
  // Step 1: a bad value comes from upstream code, and the step stops.
  ContactError error = CheckSolidContactInputs(inputs);
  if (error != ContactError::kNone) {
    move_failure.error = error;
  }
  // Step 2: a start too deep inside a solid, on the wrong side of its plate, or in a gap with no point at the rest gap
  // of every near surface, is a scene error, and the step stops. A second reacting solid at the start is a close call
  // that the start alone decides. Otherwise the start has moved once to the rest gap of every near surface, and a start
  // inside a solid or approaching an active surface is one hit at time 0.
  if (error == ContactError::kNone) {
    error = ApplyStartRules(inputs, &move, &move_failure);
  }

  // Steps 3 and 4, one sweep per pass. The duration is computed afresh from the elapsed time: a time left reduced by
  // each hit time would round above step_duration - elapsed and break FindFirstHit's precondition.
  while (error == ContactError::kNone) {
    const double duration = inputs.step_duration - move.elapsed;  // s
    // A hit at the very end of the step leaves no time; elapsed may exceed step_duration by one rounding unit.
    if (duration <= 0) {
      break;
    }
    // Step 3: the sweep fails only for a turning solid it cannot search safely (a close call, which names that solid),
    // or a bad shape. After a response every solid has the touched rule, so rounding is not a new hit.
    FirstHit first_hit;
    ContactCloseCall close_call = ContactCloseCall::kNone;
    const Vec3 velocity_for_this_step = move.result.velocity.for_this_step;
    const Sweep sweep = {
        .start = move.result.position,
        .velocity_for_this_step = velocity_for_this_step,
        .start_elapsed = move.elapsed,
        .duration = duration,
    };
    const TouchedSolid touched = {
        .is_touched = move.has_response,
        .approach_allowance = move.this_step_approach_allowance,
    };
    error = FindFirstHitOfSweep(inputs, sweep, touched, &first_hit, &close_call);
    if (error != ContactError::kNone) {
      move_failure.error = error;
      move_failure.close_call = close_call;
      if (error == ContactError::kTooCloseToCall) {
        move_failure.solid_id = inputs.solids[first_hit.solid_index].solid_id;
      }
      break;
    }
    if (first_hit.solid_index == kNoSolid) {
      move.result.position = move.result.position + velocity_for_this_step * duration;
      break;
    }
    // Step 4: one joint response over every surface the particle is on; the next pass sweeps again.
    error = ApplyHit(inputs, first_hit, &move, &move_failure);
  }

  // Step 5: the settle brings the particle back to the rest gap of every near surface; the end check keeps the module's
  // promise that a particle never ends a step inside a solid (CheckEndOutsideSolids), whatever the settle found.
  if (error == ContactError::kNone) {
    ApplySettleAtEnd(inputs, &move);
    error = CheckEndOutsideSolids(inputs, move.result.position, &move_failure);
  }
  if (failure != nullptr) {
    *failure = move_failure;
  }
  if (error == ContactError::kNone) {
    *result = move.result;
  }
  return error;
}

}  // namespace fluidity::contact
