// The MoveParticles stage: one particle move through the solids for every particle of one material in one range of
// cell blocks, on the processor the run uses.
//
// Purpose: give the step one call that moves a range of particles for one step and stops them at solid surfaces, with
//   its buffers built once from the run's memory, and one failure to report when a particle cannot be moved.
//
// Idea: MoveParticle (contact/solid_contact.h) moves one particle and reads nothing but its inputs, so particles are
//   independent: a CPU loop or one GPU thread per particle runs it, and the order does not change any particle's
//   result. MoveParticles first checks the call once: buffer sizes, pointers, overlaps, the scalars and every solid
//   (CheckSolid). Then it visits every particle once, moves it and writes its outputs (contact/move_particles_steps.h),
//   and keeps the first failure in one total order, the lower particle id then the lower index, and integer counts, so
//   the failure it reports does not depend on how threads share the work. The outputs belong to the step's try: after
//   a failure their contents are unspecified, and the step discards the whole try, so it never reads them. A failed
//   call check writes nothing. Contact adds no floating-point value across particles: the reaction of each particle is
//   written beside it, and the solids module sums them in a fixed order.
//   The pointers of the inputs and outputs are how this module stays free of the particles module: the step fills
//   MoveParticlesInputs from the particle arrays (ids, positions, both velocities, plate sides) and the materials'
//   friction, and MovedParticles from the arrays it writes the step's new positions and velocities to, plus a record
//   array and, with a reacting solid, a reaction array that it keeps as working buffers. One call per material and
//   range of cell blocks; gas markers are moved with friction 0, like any other material.
//   Example: three particles against the fixed floor z <= 0, one moving above it, one hitting it, one starting 0.01 m
//   inside it with start_overlap_tolerance 0.02 m: the result is {hit_count 1, started_inside_count 1}.
//
// Words:
//   contact record  ParticleContact: what one particle's move met in one step, kept per particle
//   first failure   the failure a call reports: the one with the lowest particle id, then the lowest index
// Owns:    ContactBuffers, BuildContactBuffers, MoveParticlesInputs, MovedParticles, MoveParticlesResult,
//          MoveParticles, BuildContactMemoryEstimate, BuildContactStackEstimate
// Uses:    Buffer, BufferError (base/buffer.h), RunMemory (base/run_memory.h), RunOn (base/run_on.h),
//          MemoryEstimateEntry (base/memory_count.h), PerParticleVelocity (base/per_particle_velocity.h), Vec3
//          (base/vec3.h), CPU-only guard (base/cpu_only.h), ContactFailure (contact/contact_error.h), Solid, PlateSide
//          (contact/solid_shapes.h), ParticleContact, SolidReaction (contact/solid_contact_types.h),
//          MoveParticleCounts (contact/move_particles_steps.h)
// Implementation uses: CheckCanRunOn (base/check_can_run_on.h), CheckSolid (contact/solid_shapes.h), the per-particle
//          functions (contact/move_particles_steps.h) and, in a build with GPU code, the GPU half
//          (contact/move_particles_gpu.h)
// Used by: the step (stage MoveParticles), the materials and the solids, which read the records and reactions
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>, <proc-move-particle>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/move_particles_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// What one MoveParticles call counts. Integer sums, the same in every order of the particles.
struct MoveParticlesResult {
  std::uint64_t hit_count = 0;  // Particles with kind == kHit.
  std::uint64_t started_inside_count = 0;  // Particles with kind == kStartedInsideSolid.
};
// The GPU writes the counts as MoveParticleCounts and the CPU reads them as this struct.
static_assert(sizeof(MoveParticlesResult) == sizeof(MoveParticleCounts), "one layout for the counts");
static_assert(
    offsetof(MoveParticlesResult, started_inside_count) == offsetof(MoveParticleCounts, started_inside_count)
);

struct MoveParticlesInputs;
struct MovedParticles;

// The buffers MoveParticles works in, built once for the run on one processor from the run's memory: a copy of the
// solids and the arrays of the failure reduction.
//
// Move-only. Every buffer is a piece of the RunMemory it was built from (base/run_memory.h), which must outlive it.
// Their contents mean nothing between calls, except solids_copy().
class ContactBuffers {
 public:
  ContactBuffers(const ContactBuffers&) = delete;
  ContactBuffers& operator=(const ContactBuffers&) = delete;
  ContactBuffers(ContactBuffers&&) noexcept = default;
  ContactBuffers& operator=(ContactBuffers&&) noexcept = default;
  ~ContactBuffers() = default;

  // The processor the buffers live on, and the processor every pointer of a call must point to.
  [[nodiscard]] RunOn run_on() const {
    return run_on_;
  }

  // The most particles one call may move.
  [[nodiscard]] std::size_t max_particle_count() const {
    return failure_per_particle_.size();
  }

  // The most solids one call may read.
  [[nodiscard]] std::size_t max_solid_count() const {
    return solids_copy_.size();
  }

  // The solids the last MoveParticles call that passed its call checks copied here, on run_on(): solids.size()
  // entries, in the call's order; empty before the first such call. Valid until the next call. A GPU pointer for
  // RunOn::kGpu, so a GPU function can read the solids without copying them again.
  [[nodiscard]] std::span<const Solid> solids_copy() const {
    return {solids_copy_.data(), solids_copy_count_};
  }

 private:
  friend std::expected<ContactBuffers, ContactFailure>
  BuildContactBuffers(RunOn run_on, std::size_t max_particle_count, std::size_t max_solid_count, RunMemory& run_memory);
  friend std::expected<MoveParticlesResult, ContactFailure>
  MoveParticles(const MoveParticlesInputs& inputs, const MovedParticles& moved, ContactBuffers& buffers);

  ContactBuffers() = default;

  Buffer<Solid> solids_copy_;  // max_solid_count solids.
  Buffer<ContactFailure> failure_per_particle_;  // max_particle_count failures, for the GPU's reduction.
  Buffer<MoveParticleCounts> call_result_;  // One: the counts of the GPU's reduction, a MoveParticlesResult.
  Buffer<ContactFailure> first_failure_;  // One: the failure the GPU's reduction keeps.
  std::size_t solids_copy_count_ = 0;  // Solids of the last call in solids_copy_.
  RunOn run_on_ = RunOn::kCpu;
};

// Builds the buffers of MoveParticles for up to max_particle_count particles and max_solid_count solids per call, as
// four pieces of `run_memory`, in the order of BuildContactMemoryEstimate's entries.
//
// Method: CheckCanRunOn(run_on) first, then BuildBuffer(run_on, size, run_memory) for the solids copy
//   (max_solid_count), the failure per particle (max_particle_count), the call result (1) and the first failure (1).
//   A RunMemory of ComputeRunMemoryBytes(BuildContactMemoryEstimate(...), counts, MemoryWhere::kRunOn) bytes holds
//   them to the byte. A size of 0 takes no piece. On the GPU, step 3 then raises the GPU's per-thread stack to what the
//   particle visits need (BuildContactStackEstimate), which the driver takes at once outside run_memory, so a GPU too
//   full for it fails here, not at the first MoveParticles call. The stack belongs to the GPU's context for the rest
//   of the process.
// Example: BuildContactBuffers(RunOn::kCpu, 1000, 4, run_memory) asks for 800 + 32,000 + 16 + 32 bytes, as pieces of
//   1,024, 32,000, 256 and 256 bytes; max_particle_count() is 1000 and max_solid_count() is 4.
// Requires: run_on == run_memory.run_on() when run_on can run, checked by assert in debug builds.
// Fails: kRunOnCallFailed  CheckCanRunOn(run_on) failed: the GPU in a build without GPU code or on a machine without
//                          a usable GPU. Nothing is taken from run_memory.
//        kOutOfMemory      a buffer does not fit in what run_memory has left; magnitude is the bytes that buffer asked
//                          for. The pieces already taken stay in the block, since a piece is never freed alone, and the
//                          step stops. On the GPU also: the GPU cannot give the particle visits their stack (step 3);
//                          magnitude is that stack's bytes, BuildContactStackEstimate's entry.
//        No buffers are returned on failure.
// Read: docs/method-paper/fluidity-method.typ <sec-run-memory>
std::expected<ContactBuffers, ContactFailure>
BuildContactBuffers(RunOn run_on, std::size_t max_particle_count, std::size_t max_solid_count, RunMemory& run_memory);

// What one MoveParticles call reads: borrowed arrays of particle_count values each, at the same indexes, on the
// processor the buffers were built for, except `solids`, which are in CPU memory.
struct MoveParticlesInputs {
  std::size_t particle_count = 0;  // Particles of one material in one range of cell blocks.
  const std::uint64_t* particle_id =
      nullptr;  // Stable, unique in the call, below 2^64 - 1. Names a failure's particle.
  const Vec3* start_position = nullptr;  // m, world, at the start of the step.
  PerParticleVelocity<const Vec3*> velocity{};  // m/s, world, before the response.
  const double* friction = nullptr;  // No unit, zero or more, per particle (a mixture blends its ingredients; gas 0).
  const PlateSide* plate_side = nullptr;  // Null: no particle of this material is committed to a plate.
  std::span<const Solid> solids;  // CPU memory, ascending solid_id. 200 bytes per solid.
  double step_duration = 0;  // s, positive.
  double start_overlap_tolerance = 0;  // m, zero or more. The deepest start inside a solid that is moved out.
  double contact_tolerance = 0;  // m, positive. From ComputeContactTolerance.
};

// Where one MoveParticles call writes, at the indexes of its inputs, on the same processor: outputs of one try, which
// the step commits only when the whole try succeeds. After a failed call their contents are unspecified.
struct MovedParticles {
  Vec3* position = nullptr;  // m, world, at the end of the step.
  Vec3* velocity_for_next_step = nullptr;  // m/s, world, after the response.
  ParticleContact* record = nullptr;  // The contact record of each particle.
  SolidReaction* reaction = nullptr;  // May be null only when no solid has reports_reaction.
};

// Moves every particle of one range for one step through the solids (MoveParticle for each), and writes their end
// positions, their velocity.for_next_step, their contact records and their reactions, or reports why it cannot.
//
// Method (method paper <sec-contact>):
//   1 The call checks, before any particle is read, in this order: particle_count or solids.size() above the buffers'
//     maxima; a required pointer null with particle_count > 0 (particle_id, start_position, both velocities, friction,
//     position, velocity_for_next_step, record), or reaction null while a solid has reports_reaction; two of the
//     arrays overlapping, an output with an input or with another output (plate_side and reaction count when not
//     null, and the solids count with every array); step_duration not finite and positive, start_overlap_tolerance
//     not finite and zero or more, contact_tolerance not finite and positive; solids.size() not below kNoSolid, or
//     solid_ids not strictly ascending; then, for each solid in array order, its linear and angular velocity summed
//     over their axes above kLargestInputSpeedSum, CheckSolid for step_duration, and its origin over the step plus its
//     reach (ComputeContactScale) above kLargestCoordinateInTolerances * contact_tolerance on some axis. A failed check
//     writes nothing, and solids_copy() keeps its last value.
//   2 The solids are copied into the buffers (solids_copy()).
//   3 One visit per particle, on run_on(): both velocities at most kLargestInputSpeedSum (CheckParticleSpeeds), on
//     each axis the start plus the travel at for_this_step at most kLargestCoordinateInTolerances * contact_tolerance
//     (CheckParticleCoordinates), then MoveParticle, then its outputs at its index; a particle that fails writes
//     nothing and the visits go on. The first failure in the order (particle_id, index) is kept, and the kinds are
//     counted. On the CPU one loop in index order; on the GPU one thread per particle.
//   Same bits on CPU and GPU when every solid's angular velocity is zero (MoveParticle); with a turning or spinning
//   solid the floating-point outputs are promised to agree within the bounds of method paper <sec-contact> (a quarter
//   of tol for positions) and the discrete ones exactly; neither is measured on a GPU yet (docs/gpu-runs.md).
// Units: m, m/s and s, world.
// Ties: two failing particles with the same particle_id, against the Requires: the lower index, on both processors.
// Example: one particle at {0.9, 0, 0} moving at {1, 0, 0} m/s for 0.2 s toward the fixed flat solid of surface x = 1,
//   friction 0, contact tolerance 2^-30 m: result {hit_count 1, started_inside_count 0}, position {1 - 2^-30, 0, 0},
//   velocity_for_next_step {0, 0, 0}, record kind kHit with hit_fraction 0.4999999953433871. Particles with ids 50 at
//   index 2 and 9 at index 7 that both start 0.01 m deep in a floor with start_overlap_tolerance 0: kStartsInsideSolid
//   with particle_id 9.
// Requires: inputs.solids, inputs and moved stay unchanged during the call. Particle ids are unique in the call and
//   below 2^64 - 1: the particles module owns that invariant (the solids module also sums reactions in id order), and
//   contact does not check it. The pointers point to run_on() memory with particle_count values each (null pointers
//   and overlaps are checked, sizes cannot be).
// Fails: kBuffersTooSmall  particle_count above max_particle_count() or solids.size() above max_solid_count().
//        kBadInput         a required pointer is null, reaction is null with a reacting solid, two arrays overlap, a
//                          scalar is out of range, there are kNoSolid solids or more, or the solid_ids do not ascend;
//                          a solid moves or turns faster than kLargestInputSpeedSum, fails CheckSolid with kBadInput
//                          or reaches beyond 2^40 contact tolerances (solid_id: that solid's; magnitude: that
//                          coordinate, m, for the last); or the first failing particle has a velocity above
//                          kLargestInputSpeedSum (CheckParticleSpeeds: magnitude that sum), starts or moves beyond
//                          2^40 contact tolerances (CheckParticleCoordinates: magnitude that coordinate, m), or fails
//                          MoveParticle with kBadInput.
//        kTooCloseToCall   a solid turns more than kMaxTurnPerStep during the step (close_call kSolidTurnsTooFar, its
//                          solid_id); or as MoveParticle for the first failing particle, kMovesTooFastForStep included.
//                          The step shortens the step and tries again.
//        kStartsInsideSolid, kOnWrongSideOfPlate, kStartsInNarrowGap  as MoveParticle, for the first failing particle.
//                          Scene errors; the step stops.
//        kRunOnCallFailed  a call to the GPU failed; the run stops.
//        A failure of one particle has has_particle_id true and its particle_id; a failure of the call's checks has
//        has_particle_id false (particle_id 0 is a valid id, so the flag tells them apart). After any failure but a
//        failed call check the outputs are unspecified: the particles visited before and after the failing one may
//        have been written.
// Read: docs/method-paper/fluidity-method.typ <sec-contact>, <proc-move-particle>
std::expected<MoveParticlesResult, ContactFailure>
MoveParticles(const MoveParticlesInputs& inputs, const MovedParticles& moved, ContactBuffers& buffers);

// Builds the estimate entry of the stack the GPU's particle visits need, which BuildContactBuffers takes on the GPU
// outside the run's memory: "contact: GPU stack of the particle visits", kGivenCount, kWholeRun,
// MemoryWhere::kDriverStack, the visit function's local bytes per thread (bytes_each) for every thread the GPU can hold
// at once (given_count: threads per multiprocessor x multiprocessors). For RunOn::kCpu the entry counts nothing.
//
// Method: the driver gives each resident thread the per-thread stack, so the count is the GPU's resident threads, not
//   the particles. The bytes per thread come from the compiled function (cudaFuncGetAttributes).
// Example: CUDA 13.4, a 36-multiprocessor sm_120 card: 55,296 threads x 12,784 bytes, about 707 MB. RunOn::kCpu: 0.
// Fails: kRunOnCallFailed  RunOn::kGpu cannot run (CheckCanRunOn), or a query of the GPU failed.
// Read: docs/method-paper/fluidity-method.typ <sec-memory-estimate>
std::expected<MemoryEstimateEntry, ContactFailure> BuildContactStackEstimate(RunOn run_on);

// Builds the memory estimate of BuildContactBuffers (base/memory_count.h): one entry per buffer, in the order it
// builds them, each kGivenCount, kWholeRun, MemoryWhere::kRunOn, one copy: "contact: solids copy" (max_solid_count x
// sizeof(Solid)), "contact: failure per particle" (max_particle_count x sizeof(ContactFailure)), "contact: call
// result" (1 x sizeof(MoveParticlesResult)), "contact: first failure" (1 x sizeof(ContactFailure)). The records and
// reactions are the step's working buffers and its estimate counts them.
//
// Example: 1000 particles and 4 solids: ComputeMemoryBytes gives 1000 x 32 + 4 x 200 + 16 + 32 = 32,848 bytes, and
//   ComputeRunMemoryBytes with MemoryWhere::kRunOn gives 32,000 + 1,024 + 256 + 256 = 33,536 bytes.
// Read: docs/method-paper/fluidity-method.typ <sec-memory-estimate>
std::vector<MemoryEstimateEntry>
BuildContactMemoryEstimate(std::size_t max_particle_count, std::size_t max_solid_count);

}  // namespace fluidity::contact
