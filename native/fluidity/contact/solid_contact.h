// Solid contact for one particle: the first hit of a straight path against the solids.
//
// Purpose: find when a particle's straight path first comes within the contact tolerance of a solid, and which solid
//   and face it meets. The types it reads and returns, and the response at a hit, are in
//   contact/solid_contact_types.h; the work on one solid is in contact/solid_contact_steps.h.
//
// Idea: during a step a particle moves in a straight line, the sweep. Every sweep runs against the
//   solids grown by the contact tolerance, so the hit is the first time the gap, the distance minus the tolerance,
//   reaches zero. For a solid that does not turn the gap to a face changes linearly with time, so the hit time is the
//   gap divided by the speed at which it closes. A particle that already rests on a surface is not hit by it again
//   unless the path pushes into it (the touching rule, contact/solid_contact_steps.h).
//   Example: a flat solid with surface x = 1 and normal {-1, 0, 0}, and a path from x = 0.9 at {1, 0, 0} m/s: gap
//   0.1 - tol, closed at 1 m/s, so the hit is at 0.1 - tol s.
//
// Words:
//   sweep           the straight path of a particle during all or part of a step, start + velocity * t for
//                   0 <= t <= duration, tested against the solids grown by the contact tolerance
//   gap             m. The distance from a particle to a solid minus the contact tolerance: zero at a hit, negative
//                   inside the grown solid. Not the rest gap, which is the distance kept at rest (tol): at rest the
//                   gap is 0
// Owns:    FindFirstHit
// Uses:    Vec3, math::IsFinite (base/vec3.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), ContactError,
//          ContactCloseCall (contact/contact_error.h), Solid, kNoSolid, CheckSolid (contact/solid_shapes.h),
//          SolidContactInputs, FirstHit (contact/solid_contact_types.h), FindFirstHitOnSolid
//          (contact/solid_contact_steps.h)
// Used by: the MoveParticles stage, through the contact functions that move one particle
// Read:    docs/method-paper/fluidity-method.typ <sec-first-hit>, <eq:hit-time>, <proc-find-first-hit>,
//          <fig-sweep-hit-slide>

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
//   2 Turning solid: this code has no search toward a turning solid yet, so a flat solid, a plate or a box whose
//     angular velocity is not exactly {0, 0, 0}, or a cylinder whose axis tilts by more than plan decision P17's
//     bound, makes the call too close to call. A spin about a sphere's centre or a cylinder's axis is not a turn.
//   3 The smallest hit time wins. *first_hit holds its time, the solid's index and at_hit: the distance, normal and
//     face at the hit position (one tolerance from the entering face up to rounding; between tol and Sqrt(2) tol from
//     a cylinder's rim, whose face it then reports), or, for a hit at 0, the distance at the start. A free path gives
//     solid_index kNoSolid, elapsed 0 and at_hit at its defaults.
//   The inputs are not checked here (plan decision P9). Same bits on CPU and GPU when every solid's angular velocity
//   is zero: only + - * /, Abs, Sqrt and comparisons.
// Units: m and m/s, world, for the path; s for start_elapsed, duration and the result's elapsed, which counts from
//   start_elapsed.
// Ties: equal hit times keep the lower solid index, then the face that comes first in face order.
// Example: a fixed flat solid with surface x = 1 and normal {-1, 0, 0}, contact tolerance 2^-30 m, and the path from
//   {0.9, 0, 0} at {1, 0, 0} m/s for 0.2 s: elapsed (1 - 0.9) - 2^-30 = 0.0999999990686774 s, solid_index 0, face
//   kHighZ, normal {-1, 0, 0}, distance 2^-30 m up to rounding. The same path starting at {1 - 2^-30, 0, 0} moving at
//   {0, 0, 1} m/s rests on the surface and is free; at {1, 0, 0} m/s it pushes in and is hit at 0. A fixed sphere of
//   radius 0.5 m at the origin and the path from {-2, 0.3, 0} at {1, 0, 0} m/s for 2 s: elapsed
//   2 - Sqrt((0.5 + 2^-30)^2 - 0.09) = 1.5999999988358469 s, face kCurved, normal near {-0.8, 0.6, 0}.
// Requires: first_hit is not null. inputs.solids points to inputs.solid_count solids (it may be null when the count
//   is 0), each of which passed CheckSolid for inputs.step_duration. start and velocity_for_this_step are finite.
//   start_elapsed and duration are finite and zero or more, and duration is at most inputs.step_duration -
//   start_elapsed as computed (the time left in the step), so every pose stays within the step.
//   inputs.contact_tolerance is finite and zero or more. When inputs.plate_side.plate_id is the solid_id of a
//   solid, that solid is a plate and the side is kAlongNormal or kAgainstNormal (an id that names no solid sees
//   every plate with both faces; MoveParticle refuses it).
// Fails: kTooCloseToCall  a solid turns: *close_call is kSlowApproachToTurningSolid. A shorter step does not fix it
//                         yet; the search toward a turning solid will.
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

}  // namespace fluidity::contact
