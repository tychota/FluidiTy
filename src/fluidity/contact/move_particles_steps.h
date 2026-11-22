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
//   MoveParticle can bound (CheckParticleSpeeds), and its start and path against the largest coordinate its rounding
//   bounds hold for (CheckParticleCoordinates), which MoveParticle does not check itself.
//   Example: three particles with ids 7, 3 and 5 at indexes 0, 1 and 2, where 7 and 5 fail: (5, 2) comes before
//   (7, 0), so the call reports particle 5 whatever order the threads finish in.
//
//   The GPU cannot keep a running minimum in one variable without making the threads wait for each other, so its loop
//   writes each particle's failure beside it (ApplyParticleVisitAt) and reduces after the loop: the smallest failing
//   id, then the smallest index among the failing particles with that id, then that particle's failure
//   (FirstFailureMinima, ComputeIsFailureWithParticleId, ComputeFirstFailure). Minima and integer sums give the same
//   answer in every order, so the GPU may update them with atomic operations.
//
// Owns:    MoveParticlesCall, kLargestInputSpeedSum, CheckParticleSpeeds, kLargestCoordinateInTolerances,
//          CheckParticleCoordinates, ComputeParticleMoveAt,
//          ApplyParticleOutputsAt, ComputeIsEarlierFailure, MoveParticleCounts, ComputeParticleCounts,
//          ParticleVisit, ApplyParticleVisitAt, FirstFailureMinima, ComputeIsFailureWithParticleId,
//          ComputeFirstFailure
// Uses:    Vec3 (base/vec3.h), math::Abs, math::IsFinite, math::Max (base/scalar_math.h), PerParticleVelocity
//          (base/per_particle_velocity.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), ContactError,
//          ContactCloseCall, ContactFailure (contact/contact_error.h), Solid, PlateSide (contact/solid_shapes.h),
//          SolidContactInputs, SolidContactResult, SolidReaction, ParticleContact, ToParticleContact, ContactKind
//          (contact/solid_contact_types.h), MoveParticle (contact/solid_contact.h)
// Used by: contact/move_particles_cpu.cc, contact/move_particles_gpu.cu
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>, <proc-move-particle>

#pragma once

#include <cassert>
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
// solid's speed, and four hits at most eight times it. Not proved for joint responses: one may return up to
// kMaxSqueezeOutSpeedFactor = 64 times the speeds involved, so four in a row could in principle grow a speed by 2^24,
// past the factor; no scene that does so is known. No physical scene comes within 2^400 of the bound.
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
//                   value gone wrong upstream, which no shorter step cures. *failure gets the error, the particle's id
//                   with has_particle_id, solid_id 0 and the magnitude; it is not written on success.
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
  failure->can_shorter_step_cure = false;
  failure->particle_id = call.particle_id[index];
  failure->solid_id = 0;
  failure->magnitude = is_next_step_too_fast ? next_step_speed_sum : this_step_speed_sum;
  return ContactError::kBadInput;
}

// No unit, in contact tolerances. The largest coordinate MoveParticles accepts, on any axis, for a particle's start and
// its path at velocity.for_this_step over the step, and for a solid's origin over the step with its reach: the
// premise of MoveParticle on coordinates (contact/solid_contact.h, Requires), which it does not check. Its rounding
// bounds rest on it: a hit within tol / 64 of the rest gap, a distance within tol / 128, and the room of the rule that
// leaves out far solids. The tolerance of ComputeContactTolerance, at least 2^-40 of the scene's largest coordinate,
// gives it for every solid and every particle start inside the domain box.
inline constexpr double kLargestCoordinateInTolerances = 0x1p40;

// Checks that particle `index` starts and moves within MoveParticle's premise on coordinates: on each axis, the
// start's coordinate plus its travel at velocity.for_this_step over the step, |start| + |velocity| * step_duration, at
// most kLargestCoordinateInTolerances * contact_tolerance. The straight path at that velocity stays within it; a path
// that a response sends farther is required and not checked. A path aimed out of the domain box fails here only when
// it would end beyond the bound, past the room the box's reach leaves (its half diagonal against its half size on
// that axis), though its side would stop it first. Changes nothing but *failure.
//
// Method: one comparison per axis. A value that is not finite passes, for MoveParticle to report with its own
//   failure. Same bits on CPU and GPU: + * and comparisons only.
// Units: m, m/s and s, world.
// Example: tol 2^-30 m, so the bound is 1024 m, and a step of 0.125 s: a particle at {1000, 0, 0} moving at
//   {192, 0, 0} m/s reaches 1024 m and passes; at {-1000, 0, 0} moving at {0, 0, 0} it passes; at {1000, 0, 0} moving
//   at {193, 0, 0} m/s it reaches 1024.125 m and fails with magnitude 1024.125.
// Requires: index < call.particle_count; the call passed the CPU half's checks.
// Fails: kBadInput  an axis's |start| + |velocity| * step_duration is above the bound (magnitude: the largest such sum,
//                   m): a particle far outside the scene its tolerance was computed for, which no shorter step cures.
//                   *failure gets the error, the particle's id with has_particle_id, solid_id 0 and the magnitude; it
//                   is not written on success.
// Read: docs/method-paper/fluidity-method.typ <proc-move-particle>, <sec-contact-tolerance>
FLUIDITY_CPU_AND_GPU inline ContactError
CheckParticleCoordinates(const MoveParticlesCall& call, std::size_t index, ContactFailure* failure) {
  const Vec3 start = call.start_position[index];
  const Vec3 velocity = call.velocity.for_this_step[index];
  const double step_duration = call.step_duration;
  const double largest_coordinate = kLargestCoordinateInTolerances * call.contact_tolerance;  // m
  const double x_extent = math::Abs(start.x) + (math::Abs(velocity.x) * step_duration);  // m
  const double y_extent = math::Abs(start.y) + (math::Abs(velocity.y) * step_duration);  // m
  const double z_extent = math::Abs(start.z) + (math::Abs(velocity.z) * step_duration);  // m
  const double largest_extent = math::Max(math::Max(x_extent, y_extent), z_extent);  // m
  // An infinite extent is left to MoveParticle, which reports a value that is not finite as kBadInput; math::Max drops
  // a NaN, which MoveParticle reports the same way.
  const bool is_beyond_scale = largest_extent > largest_coordinate && math::IsFinite(largest_extent);
  if (!is_beyond_scale) {
    return ContactError::kNone;
  }
  failure->error = ContactError::kBadInput;
  failure->close_call = ContactCloseCall::kNone;
  failure->has_particle_id = true;
  failure->can_shorter_step_cure = false;
  failure->particle_id = call.particle_id[index];
  failure->solid_id = 0;
  failure->magnitude = largest_extent;
  return ContactError::kBadInput;
}

// Moves particle `index` of the call for one step, without writing any output: checks its speeds (CheckParticleSpeeds)
// and its coordinates (CheckParticleCoordinates), then runs MoveParticle on its inputs and the call's solids.
//
// Method: the inputs of MoveParticle are the particle's values at `index`, the call's solids and scalars, and its
//   plate side, or {0, kNoPlate} when call.plate_side is null. The failure, when there is one, gets the particle's id.
//   Two calls on the same call and index give the same bits: MoveParticle reads nothing else.
// Example: a particle at {0.9, 0, 0} moving at {1, 0, 0} m/s for 0.2 s toward the fixed flat solid x >= 1, friction
//   0: *result is MoveParticle's result for those inputs, ending at {1 - tol, 0, 0} with kind kHit.
// Requires: index < call.particle_count; result and failure are not null; the call passed the CPU half's checks.
// Fails: as CheckParticleSpeeds, then as CheckParticleCoordinates, then as MoveParticle. *failure is always written:
//        ContactFailure{} on success, the failure with the particle's id and has_particle_id otherwise. *result is
//        written on success only.
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
  const ContactError coordinates_error = CheckParticleCoordinates(call, index, failure);
  if (coordinates_error != ContactError::kNone) {
    return coordinates_error;
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

// What one visit of a particle leaves for a reduction that runs after the loop: whether it failed, the id of a failing
// particle, and what a moved particle adds to the counts.
struct ParticleVisit {
  bool has_failure = false;
  std::uint64_t particle_id = 0;  // The failing particle's id; 0 when it moved.
  MoveParticleCounts counts{};  // ComputeParticleCounts of its kind when it moved; {0, 0} when it failed.
};

// Visits particle `index` for a loop that reduces after it, as the GPU's does: moves it (ComputeParticleMoveAt),
// writes its failure to call.failure_per_particle[index], ContactFailure{} when it moved, and its outputs when it
// moved (ApplyParticleOutputsAt). The GPU runs it once per thread and item; CPU tests run it over every thread block
// and thread.
//
// Example: a particle with id 9 that starts 0.01 m inside a floor with start_overlap_tolerance 0 gives {has_failure
//   true, particle_id 9, counts {0, 0}}, and failure_per_particle[index] holds kStartsInsideSolid with particle id 9.
//   A particle that hits the floor gives {false, 0, {1, 0}} and its outputs.
// Requires: index < call.particle_count; call.failure_per_particle holds particle_count failures; the call passed the
//   CPU half's checks.
FLUIDITY_CPU_AND_GPU inline ParticleVisit ApplyParticleVisitAt(const MoveParticlesCall& call, std::size_t index) {
  assert(call.failure_per_particle != nullptr);
  SolidContactResult result;
  ContactFailure failure;
  const ContactError error = ComputeParticleMoveAt(call, index, &result, &failure);
  // The returned error is the one source of "failed": the stored failure carries it, so the flag of the reduction and
  // the failures its index search reads (ComputeIsFailureWithParticleId) always agree.
  if (error != ContactError::kNone) {
    failure.error = error;
    call.failure_per_particle[index] = failure;
    return {.has_failure = true, .particle_id = failure.particle_id, .counts = {}};
  }
  call.failure_per_particle[index] = ContactFailure{};
  ApplyParticleOutputsAt(call, index, result);
  return {.has_failure = false, .particle_id = 0, .counts = ComputeParticleCounts(result.kind)};
}

// The state of the search for a call's first failure, in the order of ComputeIsEarlierFailure, as two integer minima
// and a flag: each can only go down (the minima) or up (the flag), so the threads may update it in any order and it
// ends the same. The flag is kept apart, so no id or index stands for "no failure". The GPU holds it in the one-value
// first-failure buffer until the last step writes the failure over it, so it fits there.
// Example: particles with ids 50 at index 5 and 9 at index 700 fail: {9, 700, 1}. None fails: {2^64 - 1, 2^64 - 1, 0}.
struct FirstFailureMinima {
  std::uint64_t smallest_failing_id = 0xFFFFFFFFFFFFFFFF;  // The smallest particle id among the failing particles.
  // The smallest index among the failing particles with that id.
  std::uint64_t smallest_failing_index = 0xFFFFFFFFFFFFFFFF;
  std::uint32_t has_failure = 0;  // 1 when a particle failed. An integer, so the GPU can set it with an atomic or.
};
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t), "an index fits in smallest_failing_index");
static_assert(sizeof(FirstFailureMinima) <= sizeof(ContactFailure), "the minima fit in the first-failure buffer");
static_assert(alignof(FirstFailureMinima) <= alignof(ContactFailure), "the minima fit in the first-failure buffer");

// Computes whether particle `index` failed and has the id `particle_id`: the second minimum of FirstFailureMinima
// takes the indexes of these particles only. Reads call.particle_id and call.failure_per_particle, which the visit
// wrote; whether the particle failed comes from its error, never from its id.
// Example: after the visits of FirstFailureMinima's example, (index 700, id 9) gives true, (5, 9) false, (700, 50)
//   false.
// Requires: index < call.particle_count; every particle was visited (ApplyParticleVisitAt).
FLUIDITY_CPU_AND_GPU inline bool
ComputeIsFailureWithParticleId(const MoveParticlesCall& call, std::size_t index, std::uint64_t particle_id) {
  return call.particle_id[index] == particle_id && call.failure_per_particle[index].error != ContactError::kNone;
}

// Computes the failure a call reports from the final minima: the failure of particle minima.smallest_failing_index,
// or ContactFailure{} when no particle failed. A flag without an index, which ApplyParticleVisitAt rules out, gives
// kRunOnCallFailed instead of a read past the failures.
// Example: minima {9, 700, 1} gives failure_per_particle[700]; {2^64 - 1, 2^64 - 1, 0} gives ContactFailure{};
//   {9, 2^64 - 1, 1} gives kRunOnCallFailed.
// Requires: the minima are final: every particle was visited and every failing particle with the smallest failing id
//   was compared.
FLUIDITY_CPU_AND_GPU inline ContactFailure
ComputeFirstFailure(const MoveParticlesCall& call, FirstFailureMinima minima) {
  if (minima.has_failure == 0) {
    return ContactFailure{};
  }
  // Release builds drop asserts, so the bound is checked here: an index past the failures would read GPU memory
  // that is not the call's.
  if (minima.smallest_failing_index >= call.particle_count) {
    return ContactFailure{.error = ContactError::kRunOnCallFailed};
  }
  return call.failure_per_particle[minima.smallest_failing_index];
}

}  // namespace fluidity::contact
