// The GPU half of MoveParticles, private to contact/: the declarations of what contact/move_particles_gpu.cu defines.
//
// Purpose: let contact/move_particles_cpu.cc reach the GPU loop without including a CUDA header.
//
// Idea: the CPU half has checked the call, copied the solids to the GPU and filled a MoveParticlesCall whose pointers
//   are GPU memory. The GPU half runs the per-particle functions of contact/move_particles_steps.h, one thread per
//   particle, and reduces the failures and the counts with integer operations into two one-value buffers, which the
//   CPU half then reads. It exists only in a build with GPU code, so contact/move_particles_cpu.cc names it only under
//   `#if defined(FLUIDITY_HAS_GPU)`. This file uses the C++20 subset and includes no CPU-only header.
//
// Owns:    MoveParticlesOneThreadPerParticleOnGpu, MoveParticlesOnGpuInLaunchSize, VisitStackOnGpu,
//          ComputeVisitStackOnGpu, ReserveVisitStackOnGpu
// Uses:    RunOnCallStatus, LaunchSize (base/run_on_call.h), ContactFailure (contact/contact_error.h),
//          MoveParticlesCall, MoveParticleCounts (contact/move_particles_steps.h)
// Used by: contact/move_particles_cpu.cc, contact/move_particles.gpu.test.cc, and contact/move_particles_gpu.cu, which
//          defines them
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>

#pragma once

#include <cstdint>

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
//
// Method: MoveParticlesOnGpuInLaunchSize with ComputeLaunchSize(call.particle_count).
// Returns kDone when *first_failure and *counts hold the answer, kCallFailed when a GPU call failed.
RunOnCallStatus MoveParticlesOneThreadPerParticleOnGpu(
    const MoveParticlesCall& call,
    MoveParticleCounts* counts,
    ContactFailure* first_failure
);

// The loop and the reduction of MoveParticlesOneThreadPerParticleOnGpu, with the thread blocks and their size given,
// so the GPU tests can run one call in several launch sizes and show that the answer does not depend on them.
//
// Method: four GPU functions in one stream, then a wait for the GPU:
//   1 one thread sets *counts to {0, 0} and the search for the first failure (FirstFailureMinima) to its start;
//   2 the visits: each thread runs ApplyParticleVisitAt over its items (grid stride), takes the atomic minimum of a
//     failing particle's id and sets the flag with an atomic or, and adds a moved particle's counts with atomic
//     additions;
//   3 when a particle failed, each thread takes the atomic minimum of the indexes of its failing items whose id is the
//     smallest failing id (ComputeIsFailureWithParticleId);
//   4 one thread writes ComputeFirstFailure to *first_failure.
//   Integer minima, an or and integer sums give the same values in every order of the threads, so the answer is the
//   same in every launch size and on every run. The search lives in the bytes of *first_failure (it fits:
//   contact/move_particles_steps.h), so the call allocates nothing.
// Example: 1000 particles where ids 50 at index 5 and 9 at index 700 fail: {1, 1}, {3, 64} and {4, 256} thread blocks
//   and threads all give the failure of particle 9.
// Requires: call.particle_count is 1 or more; launch_size has 1 to kMaxThreadBlockCount thread blocks of 1 to
//   kThreadBlockSize threads, the launch bound of the visits; every pointer is GPU memory.
// Fails: kCallFailed  a launch or the GPU work failed; *counts and *first_failure are then unspecified.
RunOnCallStatus MoveParticlesOnGpuInLaunchSize(
    const MoveParticlesCall& call,
    LaunchSize launch_size,
    MoveParticleCounts* counts,
    ContactFailure* first_failure
);

// The stack the GPU function that visits the particles needs on the current GPU.
struct VisitStackOnGpu {
  std::uint64_t bytes_per_thread = 0;  // The function's local memory per thread, as the compiler sized it.
  std::uint64_t resident_threads = 0;  // Threads per multiprocessor times multiprocessors: all that can run at once.
};

// Reads the stack of the visit function on the current GPU: its local bytes per thread (cudaFuncGetAttributes,
// localSizeBytes) and the threads the GPU can hold at once (maxThreadsPerMultiProcessor x multiProcessorCount).
// Example: CUDA 13.4 on a 36-multiprocessor sm_120 card: {12,784, 55,296}, about 707 MB in all.
// Fails: kCallFailed  a query failed; *stack is then unchanged.
RunOnCallStatus ComputeVisitStackOnGpu(VisitStackOnGpu* stack);

// Raises the per-thread stack of the current GPU to at least bytes_per_thread (cudaDeviceSetLimit with
// cudaLimitStackSize), so the visits' first launch never has to grow it. The CUDA manuals: the stack "will be
// immediately resized", and a failure to provide the memory returns cudaErrorMemoryAllocation. The limit belongs to
// the GPU's context, so every later GPU function of the process gets at least this stack; the driver keeps it until
// the process ends or the limit is set again. A stack already large enough is left as it is.
// Fails: kOutOfMemory  the GPU could not provide the stack; the error is cleared, so the next GPU call does not see it
//        kCallFailed   another query or the setting failed, or the GPU gave a smaller stack than asked
RunOnCallStatus ReserveVisitStackOnGpu(std::uint64_t bytes_per_thread);

}  // namespace fluidity::contact
