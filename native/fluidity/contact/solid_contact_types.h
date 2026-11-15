// The types of one particle's move through the solids, and what a hit does to the particle's velocity.
//
// Purpose: declare what one particle's move through the solids reads and returns, and change a particle's velocity
//   where it meets a solid's surface, so it does not move into the solid and loses some sideways speed to friction.
//   The contact functions that move a particle and their private steps (contact/solid_contact.h,
//   contact/solid_contact_steps.h) both build on this header, which needs neither.
//
// Idea: at a surface, a velocity splits into a part along the normal and a part along the surface, both measured
//   from the moving solid. A solid blocks motion into it and lets motion away from it pass, so only an approaching
//   velocity changes. Its speed into the solid is removed. Then friction takes friction times that removed speed off
//   the sideways speed. When that would take all the sideways speed or more, the particle sticks: it takes the
//   solid's velocity. Otherwise it slips with less sideways speed. Papers call this Coulomb friction.
//   Example: a fixed flat solid with normal {0, 0, 1} and a velocity {3, 0, -1} m/s. The approach speed is -1 m/s and
//   the sideways speed is 3 m/s. Friction 0.5 leaves {2.5, 0, 0} m/s; friction 3 or more leaves {0, 0, 0} m/s.
//
// Words:
//   hit             a moment in a step when a particle meets a solid, along its path or at its start, and its
//                   velocity gets the response; at most kMaxHitsPerStep per step. FirstHit is the earliest hit along
//                   one straight path
//   approach speed  m/s. The particle's speed along the normal out of the solid, measured from the moving solid:
//                   Dot(normal_out_of_solid, velocity - solid_velocity). Negative means approaching
//   sideways speed  m/s. The particle's speed along the surface, measured from the moving solid
//   friction        no unit, zero or more. How much sideways speed a solid takes off per unit of speed into it.
//                   Papers: the Coulomb coefficient
//   stick           the particle takes the solid's velocity, because friction takes all its sideways speed
//   slip            the particle keeps moving along the surface with less sideways speed
//   reaction        the change a solid makes to a particle's velocity for the next step, which the solid needs to
//                   compute the force on it
//   approach left after the response  the approach speed that rounding leaves in a velocity the response returned:
//                   zero up to a few rounding units, of either sign
// Owns:    ContactKind, kMaxHitsPerStep, kMaxAdvanceCount, kTurningSolidHitGapShare, SolidContactInputs, SolidReaction,
//          SolidContactResult, FirstHit, kApproachLeftAfterResponseShare, ComputeLargestApproachLeftAfterResponse,
//          ComputeContactResponse
// Uses:    Vec3, math::Dot, math::Length, math::IsFinite (base/vec3.h), math::Abs (base/scalar_math.h),
//          PerParticleVelocity (base/per_particle_velocity.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), Solid,
//          PlateSide, SolidFace, SolidSide, DistanceToSolid, kNoSolid (contact/solid_shapes.h)
// Used by: contact/solid_contact.h and contact/solid_contact_steps.h, and through them the MoveParticles stage
// Read:    docs/method-paper/fluidity-method.typ <sec-contact-response>, <eq:contact-response>,
//          <fig-contact-response>
//          Stomakhin 2013, doi:10.1145/2461912.2461948

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// What happened to a particle during one step's move.
enum class ContactKind : std::uint8_t {
  kNone,  // The particle met no solid.
  kHit,  // The path met a solid during the step.
  kStartedInsideSolid,  // The start was slightly inside a solid and was moved out to the contact tolerance.
};

// The most hits one step allows. A particle in a corner rests on three faces, one per axis; one more hit is the
// slide that leaves the corner.
inline constexpr int kMaxHitsPerStep = 4;
// The most advances of the stepping search toward a turning solid in one sweep, a cap on its cost. Each advance is
// the larger of the gap over the fastest the gap can shrink and the second-order advance, which follows the true
// rate at which the gap closes: 3 in the method paper's example; first order needed 30. A path that hovers near the
// solid with almost no motion along the normal needs more and ends in a close call, which is rare
// (FindHitOnTurningSolid states measured counts).
inline constexpr int kMaxAdvanceCount = 128;
// No unit. The stepping search toward a turning solid stops with a hit once the gap is at most this share of the
// contact tolerance, so a hit after at least one advance sits between tol and 1.25 tol from the surface (a hit at 0
// reports the start's distance, at most 1.25 tol). A quarter gives the search a range of tol / 4 to land in, while
// the hit stays far from the touching rule's window [-tol, 0] and from the end check at tol / 2. A power of two
// keeps the test exact.
inline constexpr double kTurningSolidHitGapShare = 0.25;

// What one particle's move through the solids reads for one step.
struct SolidContactInputs {
  Vec3 start_position{};  // m, world, at the start of the step.
  PerParticleVelocity<Vec3> velocity{};  // m/s, world, before the response. for_this_step moves the particle.
  const Solid* solids = nullptr;  // solid_count solids, in ascending solid_id order.
  std::uint32_t solid_count = 0;
  double step_duration = 0;  // s, positive.
  double friction = 0;  // No unit, zero or more. Papers: Coulomb coefficient.
  double start_overlap_tolerance = 0;  // m, zero or more. The deepest start inside a solid that is moved out.
  double contact_tolerance = 0;  // m, positive. From ComputeContactTolerance.
  PlateSide plate_side{};  // The side this particle is committed to, or {0, kNoPlate}.
};

// What a solid needs to compute the force on it from one particle's move.
struct SolidReaction {
  std::uint32_t solid_index = kNoSolid;  // Index into solids, not the solid_id. kNoSolid: no reacting solid was hit.
  Vec3 hit_position{};  // m, world, where the first change of velocity.for_next_step against this solid happened.
  // m/s, world. velocity.for_next_step after minus before, summed over every change this solid caused.
  Vec3 velocity_for_next_step_change{};
};
static_assert(sizeof(SolidReaction) == 56, "the step's memory estimate charges 56 bytes per particle for a reaction");

// What one particle's move through the solids returns for one step.
struct SolidContactResult {
  Vec3 position{};  // m, world, at the end of the step.
  PerParticleVelocity<Vec3> velocity{};  // m/s, world, after the response.
  ContactKind kind = ContactKind::kNone;
  double hit_fraction = 0;  // No unit, 0 to 1. When the first hit happens, as a share of the step. 0 without a hit.
  Vec3 normal_out_of_solid{};  // No unit, unit vector, world, at the first hit. Zero without a hit.
  PerParticleVelocity<double> approach_speed{};  // m/s, at the first hit. Negative means approaching.
  std::uint32_t solid_index = kNoSolid;  // Index into solids of the first hit's solid, not its solid_id.
  SolidFace face = SolidFace::kHighZ;  // The face of the first hit.
  SolidSide side_at_hit = SolidSide::kNoPlate;  // The side of the plate that was hit, when it is a plate.
  std::uint8_t hit_count = 0;  // Hits during the step, 0 to kMaxHitsPerStep.
  SolidReaction reaction{};
};

// The first hit along one straight path, or none.
struct FirstHit {
  double elapsed = 0;  // s, since the start of the sweep.
  std::uint32_t solid_index = kNoSolid;  // Index into solids, not the solid_id. kNoSolid: the path is free.
  DistanceToSolid at_hit{};  // The distance, normal and face at the hit.
};

// No unit. The share of the speeds involved that bounds the approach speed left after the response: 64 rounding units
// of 2^-52, about 9 times the at most 7 that the derivation of ComputeLargestApproachLeftAfterResponse gives.
inline constexpr double kApproachLeftAfterResponseShare = 0x1p-46;

// Computes the largest size that Dot(normal_out_of_solid, result - solid_velocity), computed in doubles, can have when
// result is ComputeContactResponse(velocity, solid_velocity, normal_out_of_solid, friction), for any unit normal and
// friction.
//
// Method: kApproachLeftAfterResponseShare * A, with A = |velocity.x| + |velocity.y| + |velocity.z| +
//   |solid_velocity.x| + |solid_velocity.y| + |solid_velocity.z|: each vector's three in x, y, z order, then the two
//   sums. A sum of absolute axes, not a
//   Length: it squares nothing, so it neither underflows for tiny speeds nor overflows for large ones, and it is at
//   least S = Length(velocity) + Length(solid_velocity), so a bound in S holds in A.
//   Derivation, with e = 2^-52: one operation rounds by at most e / 2 of its result, and a dot product with a unit
//   normal by at most 2 e of the length of its other vector (its subtraction and its three products and two sums).
//   1 The first removal leaves w with w - solid_velocity the sideways part of velocity - solid_velocity, so
//     |w - solid_velocity| <= S and |w| <= 2 S, up to terms of order e S.
//   2 The second removal recomputes that approach with an error of at most 2 e S. Its subtraction rounds each axis of
//     w by at most e / 2 of that axis, so the approach by at most e / 2 * |w| <= e S. At most 3 e S of approach
//     stays. With friction 0 that is the result.
//   3 A slip returns w + fraction * (solid_velocity - w), fraction in [0, 1]: the exact blend keeps (1 - fraction)
//     times the approach left, at most 3 e S, and its subtraction, product and sum round by at most e / 2 * S,
//     e / 2 * S and e / 2 * 2 S along the normal: at most 2 e S more, 5 e S in all. A stick returns solid_velocity,
//     whose approach computes to exactly 0.
//   4 The caller's own Dot of the result adds at most 2 e S. Total: at most 7 e S <= 7 e A, below the 64 e A of the
//     bound.
//   The normal's length, within 2^-41 of 1, changes these figures by a negligible share. Below 2^-1022 an operation
//   rounds by up to 2^-1075 absolute instead; the about 40 operations of the response and the Dot add at most
//   2^-1069, far below 57 e A for any A >= 2^-1000. A Python model of the code found at most 0.72 e A over 150,000
//   approaching cases with A from 2^-1000 to 2^500 m/s, slips and frictions just below the stick limit included.
// Units: m/s, for the two velocities and the result.
// Example: velocity {-2, 0, -2}, a fixed solid: A = 4 m/s and the bound is 2^-44 = 5.7e-14 m/s. With normal
//   {0.6, 0, 0.8} and friction 0, the response leaves an approach of -2^-55 = -2.8e-17 m/s. The same velocity times
//   2^-830 (about 1.4e-250 m/s) leaves -2^-885 against a bound of 2^-874, where a bound built from Length would be 0,
//   because the squares underflow.
// Requires: velocity and solid_velocity are finite, and A is 0 or between 2^-1000 (about 1e-301) and 2^500 (about
//   3e150) m/s: above the speeds where rounding becomes absolute, and below those where the response's Length
//   overflows its squares.
// Read: docs/method-paper/fluidity-method.typ <sec-contact-response>
FLUIDITY_CPU_AND_GPU inline double ComputeLargestApproachLeftAfterResponse(Vec3 velocity, Vec3 solid_velocity) {
  assert(math::IsFinite(velocity));
  assert(math::IsFinite(solid_velocity));
  // Method paper <sec-contact-response>:  A = sum of |v_i| + sum of |u_i|
  const double velocity_axes_sum = math::Abs(velocity.x) + math::Abs(velocity.y) + math::Abs(velocity.z);
  const double solid_velocity_axes_sum =
      math::Abs(solid_velocity.x) + math::Abs(solid_velocity.y) + math::Abs(solid_velocity.z);
  const double speeds_involved = velocity_axes_sum + solid_velocity_axes_sum;  // m/s
  assert(speeds_involved == 0 || speeds_involved >= 0x1p-1000);
  assert(speeds_involved <= 0x1p500);
  return kApproachLeftAfterResponseShare * speeds_involved;
}

// Computes one velocity after it meets a solid's surface: no speed into the solid up to rounding, and the sideways
// speed reduced by friction.
//
// Method: with approach_speed = Dot(normal_out_of_solid, velocity - solid_velocity):
//   1 approach_speed >= 0: the velocity leaves or slides along the surface and comes back unchanged, bit for bit.
//   2 velocity - approach_speed * normal_out_of_solid removes the approach. The approach left by rounding is
//     computed from that velocity and removed once more.
//   3 friction 0: that velocity is the result. Otherwise sideways_speed = Length(that velocity - solid_velocity) and
//     into_solid_speed = -approach_speed. The particle sticks, and the result is solid_velocity exactly, when
//     friction >= sideways_speed / into_solid_speed; with no sideways speed any positive friction sticks. Otherwise it
//     slips: that velocity + (friction * into_solid_speed / sideways_speed) * (solid_velocity - that velocity).
//   Afterwards the approach speed Dot(normal_out_of_solid, result - solid_velocity) is zero only up to rounding: at
//   most ComputeLargestApproachLeftAfterResponse(velocity, solid_velocity) in size, and possibly slightly negative,
//   still approaching. A caller that tests whether a returned velocity still approaches compares with that bound,
//   not with 0. A stick leaves exactly 0.
//   Same bits on CPU and GPU: only + - * / and Length, which base/ promises the same bits for.
// Units: m/s, world, for the three velocities. friction has no unit; normal_out_of_solid has none.
// Example: normal {0, 0, 1}, a fixed solid, velocity {3, 0, -1}: approach speed -1, sideways speed 3. Friction 0.5
//   gives {2.5, 0, 0}; friction 3 or 4 gives {0, 0, 0}. Velocity {0, 0, -2} at a solid moving at {0, 0, 0.5}
//   gives {0, 0, 0.5}. Velocity {1, 0, 2} comes back unchanged. Normal {0.6, 0, 0.8}, a fixed solid, velocity
//   {-2, 0, -2}, friction 0: the result's approach is -2^-55 m/s, within the bound of 5.7e-14 m/s.
// Requires: velocity and solid_velocity are finite. normal_out_of_solid is a unit vector: its squared length is
//   within 2^-40 of 1. friction is zero or more; +infinity is allowed and always sticks.
// Read: docs/method-paper/fluidity-method.typ <eq:contact-response>, <fig-contact-response>
//       Stomakhin 2013, doi:10.1145/2461912.2461948
FLUIDITY_CPU_AND_GPU inline Vec3
ComputeContactResponse(Vec3 velocity, Vec3 solid_velocity, Vec3 normal_out_of_solid, double friction) {
  assert(math::IsFinite(velocity));
  assert(math::IsFinite(solid_velocity));
  assert(math::Abs(math::Dot(normal_out_of_solid, normal_out_of_solid) - 1) <= 0x1p-40);
  assert(friction >= 0);
  // Steps: 1 keep a velocity that does not approach  2 remove the approach  3 friction
  // Step 1: the approach is measured from the moving solid.
  // Method paper <eq:contact-response>:  a = n . (v - u)
  const double approach_speed = math::Dot(normal_out_of_solid, velocity - solid_velocity);
  if (approach_speed >= 0) {
    return velocity;
  }

  // Step 2: the approach is removed from the world velocity, not from the velocity measured from the solid, which
  // would then be added back. Near a fast solid that round trip would erase a small sideways velocity:
  // (1e-9 - 1e8) + 1e8 is 0, since doubles near 1e8 are 1.5e-8 apart. The second removal takes off most of what the
  // rounding of the first one left; what still stays is bounded by ComputeLargestApproachLeftAfterResponse.
  // Method paper <eq:contact-response>:  v' = v - a n
  Vec3 velocity_along_surface = velocity - approach_speed * normal_out_of_solid;
  const double approach_left_by_rounding = math::Dot(normal_out_of_solid, velocity_along_surface - solid_velocity);
  velocity_along_surface = velocity_along_surface - approach_left_by_rounding * normal_out_of_solid;
  if (friction == 0) {
    return velocity_along_surface;
  }

  // Step 3: friction takes friction * into_solid_speed off the sideways speed. The stick test compares friction with
  // the quotient sideways_speed / into_solid_speed, as the method paper's equation does, so a friction equal to that
  // double sticks; the product form friction * into_solid_speed >= sideways_speed rounds differently at the limit.
  // Overflow is no reason for the division: an infinite product would still compare correctly. With no sideways
  // speed the quotient is 0, so any positive friction sticks.
  // Method paper <eq:contact-response>:  stick when mu >= s_side / (-a)
  const double into_solid_speed = -approach_speed;  // m/s, positive.
  const double sideways_speed = math::Length(velocity_along_surface - solid_velocity);
  const bool friction_takes_all_sideways_speed = friction >= sideways_speed / into_solid_speed;
  if (friction_takes_all_sideways_speed) {
    return solid_velocity;
  }
  // Method paper <eq:contact-response>:  v'' = v' + (mu (-a) / s_side) (u - v')
  const double friction_fraction = friction * into_solid_speed / sideways_speed;
  return velocity_along_surface + friction_fraction * (solid_velocity - velocity_along_surface);
}

}  // namespace fluidity::contact
