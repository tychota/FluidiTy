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
// Uses:    Vec3, math::IsFinite (base/vec3.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), ContactError,
//          ContactCloseCall, ContactFailure (contact/contact_error.h), Solid, kNoSolid, CheckSolid
//          (contact/solid_shapes.h), SolidContactInputs, SolidContactResult, FirstHit, kMaxAdvanceCount,
//          kTurningSolidHitGapShare (contact/solid_contact_types.h), FindFirstHitOnSolid, ParticleMove,
//          CheckSolidContactInputs, CheckStartOutsideSolids, ApplyHit, ApplySettleAtEnd, CheckEndOutsideSolids
//          (contact/solid_contact_steps.h)
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
// Method: each solid in array order, then the earliest hit (method paper <proc-find-first-hit>):
//   1 Solid that does not turn: the gap to each face plane changes linearly, gap(t) = gap(0) + rate * t with
//     rate = Dot(normal, velocity_for_this_step - linear_velocity). The touching rule comes first, on the surface
//     that gives the distance at the start: a gap in [-tol, 0] is a hit at 0 when rate * duration < -tol and no hit
//     otherwise; a gap below -tol is a hit at 0. Otherwise a flat solid, a plate or a box is hit where the path
//     enters the grown solid through its face planes, the latest crossing into a plane if no plane is left before
//     it, with the entering face; each closed side of the domain box is a flat face of its own. A plate seen by a
//     particle committed to one of its sides (inputs.plate_side) is the half space behind its committed face, so it
//     has that one face (plan decision P3). A sphere and a cylinder use their true distance for the touching rule
//     (plan decision P16), then a closed form: the first root of the path on the sphere grown by tol, or the entry
//     into the cylinder grown by tol with sharp edges, the later of its entries into the grown wall and between the
//     grown caps' planes when it comes before both exits (plan decision P15). A hit counts when its time is at most
//     duration.
//   2 Turning solid: a flat solid, a plate or a box whose angular velocity is not exactly {0, 0, 0} (plan decision
//     P7), or a cylinder whose axis tilts by more than plan decision P17's bound. Its gap is not linear in time, so
//     the stepping search advances from t = 0 by g(t) / bound, with bound = Length(velocity_for_this_step -
//     linear_velocity) + Length(angular_velocity) * r_max and r_max the larger distance from the path's two ends to
//     the solid's origin at those times: the gap cannot shrink faster, so no advance passes a crossing. It stops with
//     a hit when g(t) <= tol / 4 (kTurningSolidHitGapShare), after at least one advance at a distance in
//     [tol, 1.25 tol], and with no hit once t >= duration. It needs about ln(4 g / tol) * bound / rate advances, rate
//     the speed at which the gap really closes. There is no touching rule: a particle resting on a turning solid is
//     hit at 0. A spin about a sphere's centre or a cylinder's axis is not a turn.
//   3 The smallest hit time wins. *first_hit holds its time, the solid's index and at_hit: the distance, normal and
//     face at the hit position (one tolerance from the entering face up to rounding; between tol and Sqrt(2) tol from
//     a cylinder's rim, whose face it then reports; between tol and 1.25 tol from a turning solid), or, for a hit at
//     0, the distance at the start. A free path gives
//     solid_index kNoSolid, elapsed 0 and at_hit at its defaults.
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
//   1 s: after 30 advances, elapsed 0.8857 s on kHighX, 4e-10 s before 0.3 + acos((0.5 + 2^-30) / 0.6).
// Requires: first_hit is not null. inputs.solids points to inputs.solid_count solids (it may be null when the count
//   is 0), each of which passed CheckSolid for inputs.step_duration. start and velocity_for_this_step are finite.
//   start_elapsed and duration are finite and zero or more, and duration is at most inputs.step_duration -
//   start_elapsed as computed (the time left in the step), so every pose stays within the step up to one rounding
//   unit: start_elapsed + (step_duration - start_elapsed) can exceed step_duration by one.
//   inputs.contact_tolerance is finite and zero or more. When inputs.plate_side.plate_id is the solid_id of a
//   solid, that solid is a plate and the side is kAlongNormal or kAgainstNormal (an id that names no solid sees
//   every plate with both faces; MoveParticle refuses it).
// Fails: kTooCloseToCall  the search toward a turning solid needs more than kMaxAdvanceCount advances: the path
//                         closes on it at a small share of the bound, or passes near it or grazes it, which is common
//                         near a turning solid (FindHitOnTurningSolid: Cost). *close_call is
//                         kSlowApproachToTurningSolid; the caller shortens the step and tries again.
//        kBadInput        a solid is of an unknown shape; a debug build stops at an assertion first.
//        When close_call is not null, *close_call is always written: kNone unless the error is kTooCloseToCall. On
//        failure *first_hit is unchanged.
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
  assert(first_hit != nullptr);
  assert(inputs.solids != nullptr || inputs.solid_count == 0);
  assert(math::IsFinite(start) && math::IsFinite(velocity_for_this_step));
  assert(math::IsFinite(start_elapsed) && start_elapsed >= 0 && math::IsFinite(duration) && duration >= 0);
  // With duration >= 0 this also keeps start_elapsed within the step.
  assert(duration <= inputs.step_duration - start_elapsed);
  assert(math::IsFinite(inputs.contact_tolerance) && inputs.contact_tolerance >= 0);
  if (close_call != nullptr) {
    *close_call = ContactCloseCall::kNone;
  }
  // Steps 1 and 2 run in FindFirstHitOnSolid, one solid at a time; step 3 keeps the earliest hit here.
  FirstHit earliest;
  for (std::uint32_t solid_index = 0; solid_index < inputs.solid_count; ++solid_index) {
    assert(CheckSolid(inputs.solids[solid_index], inputs.step_duration) == ContactError::kNone);
    FirstHit hit_on_solid;
    const ContactError error =
        FindFirstHitOnSolid(inputs, solid_index, start, velocity_for_this_step, start_elapsed, duration, &hit_on_solid);
    if (error != ContactError::kNone) {
      // FindFirstHitOnSolid fails with a close call only for a turning solid, so the reason is always the slow
      // approach. The step is shortened and tried again.
      if (error == ContactError::kTooCloseToCall && close_call != nullptr) {
        *close_call = ContactCloseCall::kSlowApproachToTurningSolid;
      }
      return error;
    }
    // Step 3: a later solid wins only when strictly earlier, so equal times keep the lower index.
    const bool is_hit = hit_on_solid.solid_index != kNoSolid;
    const bool is_earlier = earliest.solid_index == kNoSolid || hit_on_solid.elapsed < earliest.elapsed;
    if (is_hit && is_earlier) {
      earliest = hit_on_solid;
    }
  }
  *first_hit = earliest;
  return ContactError::kNone;
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
//   2 Classify the start: a start inside a solid at time 0, d < 0, fails; a start within one tolerance, 0 <= d < tol,
//     sweeps from where it is, and the touching rule decides whether its path pushes into that solid.
//   3 Find the first hit (FindFirstHit) from the current position along velocity.for_this_step, with the solids posed
//     at the time already used and duration = step_duration - elapsed, computed afresh before every sweep. With no
//     hit the particle advances by velocity.for_this_step times that duration, and the hits end.
//   4 Respond (ApplyHit): the particle moves to the hit, both velocities get ComputeContactResponse with the solid
//     velocity at the hit and friction times the solid's friction_scale, the first hit fills the record, and the
//     change of velocity.for_next_step counts in the reaction of a solid with reports_reaction (plan decision P5).
//   5 Slide: the solid and its normal join the touched surfaces and the sweep starts again at step 3 with the new
//     velocity.for_this_step. A solid that does not turn is not hit again by a path along its surface (the touching
//     rule). A turning solid has no touching rule, so a particle it hit is hit again at time 0 by every later sweep
//     and the move ends in kTooManyHits.
//   6 Settle and check the end (ApplySettleAtEnd, CheckEndOutsideSolids): each touched surface closer than tol at the
//     end pose moves the particle back to the rest gap, then every solid must be at least tol / 2 away.
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
//   at 0.1 - 2^-30 s, slips on at {2.5, 0, 0} m/s and ends at x = 0.75 - 2^-31, z = 2^-30.
// Requires: result is not null. Each solid passed CheckSolid for inputs.step_duration (plan decision P9).
// Fails: kBadInput          a value is not finite or out of range, the solid_ids are not strictly ascending, or the
//                           plate side names no plate (CheckSolidContactInputs).
//        kStartsInsideSolid the start is inside a solid; the failure names it, with the depth
//        (CheckStartOutsideSolids). kTooCloseToCall    the step is too long for a safe answer; the step shortens it and
//        tries again.
//                           *failure's close_call says why: kTooManyHits (a fifth hit), kSecondReactingSolid (a
//                           second solid with reports_reaction), kEndsTooCloseToSolid (the end check), or
//                           kSlowApproachToTurningSolid (passed on from FindFirstHit, with solid_id 0).
//        On failure *result is unchanged. When failure is not null, *failure is always written: ContactFailure{} on
//        success; on failure the error, the close call, the solid's solid_id (never its index) and the magnitude of
//        plan decision P6. particle_id stays 0.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <fig-sweep-hit-slide>, <sec-contact-response>,
//       <sec-contact-tolerance>
FLUIDITY_CPU_AND_GPU inline ContactError
MoveParticle(const SolidContactInputs& inputs, SolidContactResult* result, ContactFailure* failure) {
  assert(result != nullptr);
  // Steps (method paper <proc-move-particle>): 1 check inputs  2 classify the start  3 find the first hit
  //   4 respond  5 slide  6 settle and check the end
  // Every step writes its failure here; *failure gets it at the end, and stays ContactFailure{} on success.
  ContactFailure move_failure;
  // Step 1: a bad value comes from upstream code, and the step stops.
  ContactError error = CheckSolidContactInputs(inputs);
  if (error != ContactError::kNone) {
    move_failure.error = error;
  }
  // Step 2: inside a solid at the start of the step is a scene error, and the step stops.
  if (error == ContactError::kNone) {
    error = CheckStartOutsideSolids(inputs, &move_failure);
  }

  // Steps 3 to 5, one sweep per pass. The duration is computed afresh from the elapsed time: a time left reduced by
  // each hit time would round above step_duration - elapsed and break FindFirstHit's precondition.
  ParticleMove move;
  move.result.position = inputs.start_position;
  move.result.velocity = inputs.velocity;
  while (error == ContactError::kNone) {
    const double duration = inputs.step_duration - move.elapsed;  // s
    // A hit at the very end of the step leaves no time; elapsed may exceed step_duration by one rounding unit.
    if (duration <= 0) {
      break;
    }
    // Step 3: FindFirstHit fails only for a turning solid it cannot search safely (a close call), or a bad shape.
    FirstHit first_hit;
    ContactCloseCall close_call = ContactCloseCall::kNone;
    const Vec3 velocity_for_this_step = move.result.velocity.for_this_step;
    error = FindFirstHit(
        inputs,
        move.result.position,
        velocity_for_this_step,
        move.elapsed,
        duration,
        &first_hit,
        &close_call
    );
    if (error != ContactError::kNone) {
      move_failure.error = error;
      move_failure.close_call = close_call;
      break;
    }
    if (first_hit.solid_index == kNoSolid) {
      move.result.position = move.result.position + velocity_for_this_step * duration;
      break;
    }
    // Steps 4 and 5: the response, the record, the reaction and the touched surface; the next pass sweeps again.
    error = ApplyHit(inputs, first_hit, &move, &move_failure);
  }

  // Step 6: the settle keeps the rest gap to the touched surfaces; the end check keeps the module's promise that a
  // particle never ends a step inside a solid (CheckEndOutsideSolids).
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
