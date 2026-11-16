// The GPU half of MoveParticles, private to contact/: the declaration of what contact/move_particles_gpu.cu defines.
//
// Purpose: let contact/move_particles_cpu.cc reach the GPU loop without including a CUDA header.
//
// Idea: the CPU half has checked the call, copied the solids to the GPU and filled a MoveParticlesCall whose pointers
//   are GPU memory. The GPU half runs the per-particle functions of contact/move_particles_steps.h, one thread per
//   particle, and reduces the failures and the counts with integer operations into two one-value buffers, which the
//   CPU half then reads. It exists only in a build with GPU code, so contact/move_particles_cpu.cc names it only under
//   `#if defined(FLUIDITY_HAS_GPU)`. This file uses the C++20 subset and includes no CPU-only header.
//
// Owns:    MoveParticlesOneThreadPerParticleOnGpu
// Uses:    RunOnCallStatus (base/run_on_call.h), ContactFailure (contact/contact_error.h), MoveParticlesCall,
//          MoveParticleCounts (contact/move_particles_steps.h)
// Used by: contact/move_particles_cpu.cc, and contact/move_particles_gpu.cu, which defines it
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>

#pragma once

#include "fluidity/base/run_on_call.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/move_particles_steps.h"

namespace fluidity::contact {

// Moves every particle of `call` on the GPU, one thread each: ComputeParticleMoveAt, then ApplyParticleOutputsAt on
// success, and the particle's failure (ContactFailure{} on success) in call.failure_per_particle. Then the reduction:
// the first failure in the order of ComputeIsEarlierFailure, as two integer minima (the smallest particle_id among the
// failing particles, then the smallest index among the failing particles with that id), goes to *first_failure, or
// ContactFailure{} when none failed, and the kinds' counts, integer sums, to *counts (both GPU memory, one value each).
// Whether a particle failed is read from its error, never from its id. call.particle_count is 1 or more.
// Returns kDone when *first_failure and *counts hold the answer, kCallFailed when a GPU call failed.
RunOnCallStatus MoveParticlesOneThreadPerParticleOnGpu(
    const MoveParticlesCall& call,
    MoveParticleCounts* counts,
    ContactFailure* first_failure
);

}  // namespace fluidity::contact
