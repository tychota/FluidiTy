// The work MoveParticles does for one particle, written once so the CPU loop and a GPU thread run the same code.
//
// Purpose: give both halves of MoveParticles (contact/move_particles.h) one description of a call, one move of one
//   particle with the check of its speeds that guards it, one way to write that particle's outputs, and what the loop
//   reduces over the particles: the order of failures and the counts.
//
// Idea: a call visits every particle of a range once, moves it and writes its outputs at its index; a particle that
//   fails writes nothing. Particles are independent, so the visits may come in any order and in any number of
//   threads. The loop keeps the earliest failure in one total order, the lower particle id then the lower index
//   (ComputeIsEarlierFailure), with a flag that says whether any particle failed, and adds the counts as integers, so
//   the reported failure and the counts do not depend on that order. No floating-point value is summed across
//   particles. Before each move the particle's speeds are checked against the largest speeds whose response rounding
//   MoveParticle can bound (CheckParticleSpeeds), which MoveParticle does not check itself.
//   Example: three particles with ids 7, 3 and 5 at indexes 0, 1 and 2, where 7 and 5 fail: (5, 2) comes before
//   (7, 0), so the call reports particle 5 whatever order the threads finish in.
//
// Owns:    MoveParticlesCall, kLargestInputSpeedSum, CheckParticleSpeeds, ComputeParticleMoveAt,
//          ApplyParticleOutputsAt, ComputeIsEarlierFailure, MoveParticleCounts, ComputeParticleCounts
// Uses:    Vec3 (base/vec3.h), math::Abs, math::IsFinite (base/scalar_math.h), PerParticleVelocity
//          (base/per_particle_velocity.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), ContactError,
//          ContactCloseCall, ContactFailure (contact/contact_error.h), Solid, PlateSide (contact/solid_shapes.h),
//          SolidContactInputs, SolidContactResult, SolidReaction, ParticleContact, ToParticleContact, ContactKind
//          (contact/solid_contact_types.h), MoveParticle (contact/solid_contact.h)
// Used by: contact/move_particles_cpu.cc, contact/move_particles_gpu.cu
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>, <proc-move-particle>

#pragma once

#include <cstddef>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// Everything one MoveParticles call reads and writes, as plain pointers and values on the processor that runs it: the
// CPU half fills it from MoveParticlesInputs, MovedParticles and ContactBuffers (contact/move_particles.h) after their
// checks, and passes it whole to the per-particle functions below, or to the GPU half by value. Every array holds
// particle_count values at the same indexes, except solids.
struct MoveParticlesCall {
  std::size_t particle_count = 0;
  const std::uint64_t* particle_id = nullptr;  // Stable ids, unique in the call, below 2^64 - 1.
  const Vec3* start_position = nullptr;  // m, world.
  PerParticleVelocity<const Vec3*> velocity{};  // m/s, world, before the response.
  const double* friction = nullptr;  // No unit, zero or more.
  const PlateSide* plate_side = nullptr;  // Null: no particle is committed to a plate.
  const Solid* solids = nullptr;  // solid_count solids in ascending solid_id order: the copy in ContactBuffers.
  // The outputs, written at the index of each particle that moves; a particle that fails leaves its index as it was.
  Vec3* position = nullptr;  // m, world, at the end of the step.
  Vec3* velocity_for_next_step = nullptr;  // m/s, world, after the response.
  ParticleContact* record = nullptr;
  SolidReaction* reaction = nullptr;  // Null when no solid reports a reaction.
  // One failure per particle for a reduction that runs after the loop, as the GPU's does; null when the loop reduces
  // as it goes, as the CPU's does.
  ContactFailure* failure_per_particle = nullptr;
  double step_duration = 0;  // s, positive.
  double start_overlap_tolerance = 0;  // m, zero or more.
  double contact_tolerance = 0;  // m, positive.
  std::uint32_t solid_count = 0;  // Below kNoSolid.
};
// The GPU half receives the call by value from the CPU half, which another compiler built: one layout for both.
static_assert(sizeof(MoveParticlesCall) == 136, "the CPU half and the GPU half read one layout of MoveParticlesCall");
static_assert(offsetof(MoveParticlesCall, step_duration) == 104, "one layout of MoveParticlesCall");
static_assert(offsetof(MoveParticlesCall, solid_count) == 128, "one layout of MoveParticlesCall");

// m/s. The largest sum of the absolute axes of a particle's velocity, of a solid's linear velocity (m/s) and of its
// angular velocity (rad/s) that MoveParticles accepts. ComputeLargestApproachLeftAfterResponse requires the speeds
// involved at a hit, the particle's and the solid's surface velocity summed over their axes, to be at most 2^500 m/s,
// below the speeds whose squares overflow; MoveParticle asserts it and does not check it. 2^480 leaves a factor 2^20
// for what the responses of a step and a spin's lever arm add: each response on one surface adds at most twice the
// solid's speed, and four hits at most eight times it. No physical scene comes within 2^400 of it.
inline constexpr double kLargestInputSpeedSum = 0x1p480;

// Checks that both velocities of particle `index` are at most kLargestInputSpeedSum, summed over their axes: the part
// of MoveParticle's Requires on the speeds that MoveParticle does not check for itself. MoveParticle checks the bound
// of a response's rounding against the time left at every hit (kMovesTooFastForStep) and that every value is finite
// (kBadInput); the solids' speeds are checked once per call. Changes nothing but *failure.
//
// Method: a velocity that is not finite passes, for MoveParticle to report with its own failure. Same bits on CPU and
//   GPU: + and comparisons only.
// Units: m/s, world.
// Example: for_this_step {1e150, 0, 0} m/s passes; {2^481, 0, 0} m/s fails with magnitude 2^481.
// Requires: index < call.particle_count; the call passed the CPU half's checks.
// Fails: kBadInput  a velocity's sum of absolute axes is above kLargestInputSpeedSum (magnitude: that sum, m/s): a
// value
//                   gone wrong upstream, which no shorter step cures. *failure gets the error, the particle's id with
//                   has_particle_id, solid_id 0 and the magnitude; it is not written on success.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <sec-contact-response>
FLUIDITY_CPU_AND_GPU inline ContactError
CheckParticleSpeeds(const MoveParticlesCall& call, std::size_t index, ContactFailure* failure) {
  const auto compute_axes_sum = [](Vec3 vector) -> double {
    return math::Abs(vector.x) + math::Abs(vector.y) + math::Abs(vector.z);
  };
  const double next_step_speed_sum = compute_axes_sum(call.velocity.for_next_step[index]);  // m/s
  const double this_step_speed_sum = compute_axes_sum(call.velocity.for_this_step[index]);  // m/s
  // An infinite or NaN sum is left to MoveParticle, which reports a value that is not finite as kBadInput.
  const auto is_too_fast = [](double speed_sum) -> bool {
    return speed_sum > kLargestInputSpeedSum && math::IsFinite(speed_sum);
  };
  const bool is_next_step_too_fast = is_too_fast(next_step_speed_sum);
  const bool is_this_step_too_fast = is_too_fast(this_step_speed_sum);
  if (!is_next_step_too_fast && !is_this_step_too_fast) {
    return ContactError::kNone;
  }
  failure->error = ContactError::kBadInput;
  failure->close_call = ContactCloseCall::kNone;
  failure->has_particle_id = true;
  failure->particle_id = call.particle_id[index];
  failure->solid_id = 0;
  failure->magnitude = is_next_step_too_fast ? next_step_speed_sum : this_step_speed_sum;
  return ContactError::kBadInput;
}

// Moves particle `index` of the call for one step, without writing any output: checks its speeds (CheckParticleSpeeds),
// then runs MoveParticle on its inputs and the call's solids.
//
// Method: the inputs of MoveParticle are the particle's values at `index`, the call's solids and scalars, and its
//   plate side, or {0, kNoPlate} when call.plate_side is null. The failure, when there is one, gets the particle's id.
//   Two calls on the same call and index give the same bits: MoveParticle reads nothing else.
// Example: a particle at {0.9, 0, 0} moving at {1, 0, 0} m/s for 0.2 s toward the fixed flat solid x >= 1, friction
//   0: *result is MoveParticle's result for those inputs, ending at {1 - tol, 0, 0} with kind kHit.
// Requires: index < call.particle_count; result and failure are not null; the call passed the CPU half's checks.
// Fails: as CheckParticleSpeeds, then as MoveParticle. *failure is always written: ContactFailure{} on success, the
//        failure with the particle's id and has_particle_id otherwise. *result is written on success only.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>
FLUIDITY_CPU_AND_GPU inline ContactError ComputeParticleMoveAt(
    const MoveParticlesCall& call,
    std::size_t index,
    SolidContactResult* result,
    ContactFailure* failure
) {
  *failure = ContactFailure{};
  const ContactError speeds_error = CheckParticleSpeeds(call, index, failure);
  if (speeds_error != ContactError::kNone) {
    return speeds_error;
  }
  SolidContactInputs inputs;
  inputs.start_position = call.start_position[index];
  inputs.velocity = {
      .for_next_step = call.velocity.for_next_step[index],
      .for_this_step = call.velocity.for_this_step[index],
  };
  inputs.solids = call.solids;
  inputs.solid_count = call.solid_count;
  inputs.step_duration = call.step_duration;
  inputs.friction = call.friction[index];
  inputs.start_overlap_tolerance = call.start_overlap_tolerance;
  inputs.contact_tolerance = call.contact_tolerance;
  inputs.plate_side = call.plate_side == nullptr ? PlateSide{} : call.plate_side[index];
  const ContactError error = MoveParticle(inputs, result, failure);
  if (error != ContactError::kNone) {
    failure->has_particle_id = true;
    failure->particle_id = call.particle_id[index];
  }
  return error;
}

// Writes the outputs of particle `index` of the call from the result of its move: the end position,
// velocity.for_next_step, the contact record and, when call.reaction is not null, the reaction. The loop calls it
// after ComputeParticleMoveAt succeeds; for_this_step after the move is not kept.
//
// Example: the particle of ComputeParticleMoveAt's example: position[index] = {1 - tol, 0, 0},
//   velocity_for_next_step[index] = {0, 0, 0}, record[index].kind = kHit, reaction[index].solid_index = kNoSolid.
// Requires: index < call.particle_count; result is ComputeParticleMoveAt's for that index; the outputs hold
//   particle_count values and overlap no input.
FLUIDITY_CPU_AND_GPU inline void
ApplyParticleOutputsAt(const MoveParticlesCall& call, std::size_t index, const SolidContactResult& result) {
  call.position[index] = result.position;
  call.velocity_for_next_step[index] = result.velocity.for_next_step;
  call.record[index] = ToParticleContact(result);
  if (call.reaction != nullptr) {
    call.reaction[index] = result.reaction;
  }
}

// Computes whether the failure of the particle (particle_id, index) comes before that of (other_particle_id,
// other_index) in the order whose first failing particle a call reports: the lower particle id, then the lower index.
// A total order, so the reported failure is the same whatever order the threads finish in, even with equal ids. The
// loop keeps a flag of its own for whether any particle failed, so no id stands for "no failure".
// On the GPU the order is two integer minima: the smallest id among the failing particles, then the smallest index
// among the failing particles with that id.
//
// Example: (9, 7) comes before (50, 2) and before (9, 8).
// clang-tidy: two (id, index) pairs, in the order the comparison reads them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsEarlierFailure(
    std::uint64_t particle_id,
    std::size_t index,
    std::uint64_t other_particle_id,
    std::size_t other_index
) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  if (particle_id != other_particle_id) {
    return particle_id < other_particle_id;
  }
  return index < other_index;
}

// What one call counts over its particles; the fields of MoveParticlesResult (contact/move_particles.h), which lives
// in a header the GPU cannot include.
struct MoveParticleCounts {
  std::uint64_t hit_count = 0;  // Particles with kind kHit.
  std::uint64_t started_inside_count = 0;  // Particles with kind kStartedInsideSolid.
};
static_assert(sizeof(MoveParticleCounts) == 16, "the CPU half reads the GPU's counts byte for byte");
static_assert(offsetof(MoveParticleCounts, started_inside_count) == 8, "one layout of MoveParticleCounts");

// Computes what one particle adds to the counts: 1 to hit_count for kHit, 1 to started_inside_count for
// kStartedInsideSolid, nothing for kNone.
//
// Example: kHit gives {1, 0}; kNone gives {0, 0}.
FLUIDITY_CPU_AND_GPU constexpr MoveParticleCounts ComputeParticleCounts(ContactKind kind) {
  return {
      .hit_count = kind == ContactKind::kHit ? 1U : 0U,
      .started_inside_count = kind == ContactKind::kStartedInsideSolid ? 1U : 0U,
  };
}

}  // namespace fluidity::contact
