// The GPU half of MoveParticles, private to contact/: the functions declared in contact/move_particles_gpu.h.
//
// Four thin GPU functions over the shared per-particle functions of contact/move_particles_steps.h, which CPU tests
// run over every thread block and thread: start the reduction, visit every particle, find the smallest failing index
// among the failing particles with the smallest failing id, write the first failure. The only operations between
// threads are integer atomics (minimum, or, addition) on 64-bit and 32-bit integers: their final values do not depend
// on the order the threads apply them, so the discrete answer is the same on every run. No floating-point value is
// summed. Every call uses the default stream, so the four functions run in order and each sees what the one before it
// wrote. Launches, the wait and statuses go through base/gpu_launch.cuh.
//
// Rounding: the move of each particle is the CPU's code compiled for the GPU with --fmad=false (no fused multiply-add),
// as the CPU builds forbid contraction, so it gives the same bits as the CPU when no solid turns or spins
// (contact/solid_contact.h, MoveParticle).
// Read: docs/method-paper/fluidity-method.typ <sec-contact>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/move_particles_gpu.h"
#include "fluidity/contact/move_particles_steps.h"

namespace fluidity::contact {
namespace {

// The CUDA atomics take unsigned long long for 64 bits; std::uint64_t is unsigned long on Linux and unsigned long long
// on Windows, both 64 bits wide.
static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t), "the 64-bit atomics act on the 64-bit fields");

// The 64-bit integer at `value` as the CUDA atomics take it.
__device__ unsigned long long* ToAtomicInteger(std::uint64_t* value) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the same 64 bits, as the atomics' type.
  return reinterpret_cast<unsigned long long*>(value);
}

// Step 1, one thread: the counts start at {0, 0} and the search at no failure.
__global__ void StartReduction(MoveParticleCounts* counts, FirstFailureMinima* minima) {
  *counts = MoveParticleCounts{};
  *minima = FirstFailureMinima{};
}

// Step 2: every particle visited once, its failure kept beside it; the smallest failing id, the flag and the counts
// updated with atomics. A count of 0 adds nothing, so only particles that hit or started inside touch the counts.
// The move of one particle is long: the CUDA 13.4 build gives this function up to 255 registers per thread and about
// 11 to 13 KB of stack, which the driver provides at launch. The launch bound (thread blocks of at most
// kThreadBlockSize threads, at least one resident per multiprocessor) makes the compiler fit one thread block into a
// multiprocessor's 65,536 registers, so a launch never fails for lack of registers, and lets it use them all: with
// the thread count alone, sm_86 and sm_89 cut to 40 registers to fit more thread blocks and spilled 25 KB per thread.
__global__ void __launch_bounds__(
    kThreadBlockSize,
    1
) VisitEveryParticle(MoveParticlesCall call, MoveParticleCounts* counts, FirstFailureMinima* minima) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < call.particle_count; index += stride.item_step) {
    const ParticleVisit visit = ApplyParticleVisitAt(call, index);
    if (visit.has_failure) {
      atomicMin(ToAtomicInteger(&minima->smallest_failing_id), visit.particle_id);
      atomicOr(&minima->has_failure, 1U);
      continue;
    }
    if (visit.counts.hit_count != 0) {
      atomicAdd(ToAtomicInteger(&counts->hit_count), visit.counts.hit_count);
    }
    if (visit.counts.started_inside_count != 0) {
      atomicAdd(ToAtomicInteger(&counts->started_inside_count), visit.counts.started_inside_count);
    }
  }
}

// Step 3: the smallest index among the failing particles with the smallest failing id. Step 2 has finished, so the
// smallest failing id and the flag are final; this function changes only the index.
__global__ void __launch_bounds__(
    kThreadBlockSize
) FindSmallestFailingIndex(MoveParticlesCall call, FirstFailureMinima* minima) {
  // Without a failure there is nothing to find, and the threads read no failure.
  if (minima->has_failure == 0) {
    return;
  }
  const std::uint64_t smallest_failing_id = minima->smallest_failing_id;
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < call.particle_count; index += stride.item_step) {
    if (ComputeIsFailureWithParticleId(call, index, smallest_failing_id)) {
      atomicMin(ToAtomicInteger(&minima->smallest_failing_index), index);
    }
  }
}

// Step 4, one thread: the reported failure. `minima` and `first_failure` are the same bytes, so the minima are read
// whole before the failure is written over them.
__global__ void
WriteFirstFailure(MoveParticlesCall call, const FirstFailureMinima* minima, ContactFailure* first_failure) {
  const FirstFailureMinima final_minima = *minima;
  *first_failure = ComputeFirstFailure(call, final_minima);
}

// The launch of a GPU function that one thread runs.
constexpr LaunchSize kOneThread = {.thread_block_count = 1, .thread_block_size = 1};

}  // namespace

RunOnCallStatus MoveParticlesOneThreadPerParticleOnGpu(
    const MoveParticlesCall& call,
    MoveParticleCounts* counts,
    ContactFailure* first_failure
) {
  return MoveParticlesOnGpuInLaunchSize(call, ComputeLaunchSize(call.particle_count), counts, first_failure);
}

RunOnCallStatus MoveParticlesOnGpuInLaunchSize(
    const MoveParticlesCall& call,
    LaunchSize launch_size,
    MoveParticleCounts* counts,
    ContactFailure* first_failure
) {
  // Steps: 1 start the reduction  2 visit every particle  3 the smallest failing index  4 the first failure, then wait
  // The search lives in the first-failure buffer until step 4 writes the failure over it (move_particles_steps.h
  // asserts that it fits), so the call needs no memory of its own.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the buffer's bytes hold the search until step 4.
  auto* const minima = reinterpret_cast<FirstFailureMinima*>(first_failure);
  // Step 1.
  if (const RunOnCallStatus status = ApplyGpuFunction(StartReduction, kOneThread, counts, minima);
      status != RunOnCallStatus::kDone) {
    return status;
  }
  // Step 2.
  if (const RunOnCallStatus status = ApplyGpuFunction(VisitEveryParticle, launch_size, call, counts, minima);
      status != RunOnCallStatus::kDone) {
    return status;
  }
  // Step 3.
  if (const RunOnCallStatus status = ApplyGpuFunction(FindSmallestFailingIndex, launch_size, call, minima);
      status != RunOnCallStatus::kDone) {
    return status;
  }
  // Step 4.
  if (const RunOnCallStatus status = ApplyGpuFunction(WriteFirstFailure, kOneThread, call, minima, first_failure);
      status != RunOnCallStatus::kDone) {
    return status;
  }
  // The wait reports an error of any of the four functions while it ran.
  return CheckGpuWork();
}

RunOnCallStatus ComputeVisitStackOnGpu(VisitStackOnGpu* stack) {
  cudaFuncAttributes attributes{};
  if (const cudaError_t error = cudaFuncGetAttributes(&attributes, VisitEveryParticle); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  int device = 0;
  if (const cudaError_t error = cudaGetDevice(&device); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  int threads_per_multiprocessor = 0;
  if (const cudaError_t error =
          cudaDeviceGetAttribute(&threads_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, device);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  int multiprocessor_count = 0;
  if (const cudaError_t error = cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  stack->bytes_per_thread = attributes.localSizeBytes;
  stack->resident_threads =
      static_cast<std::uint64_t>(threads_per_multiprocessor) * static_cast<std::uint64_t>(multiprocessor_count);
  return RunOnCallStatus::kDone;
}

RunOnCallStatus ReserveVisitStackOnGpu(std::uint64_t bytes_per_thread) {
  // Steps: 1 the stack now  2 raise it  3 check what the driver gave
  // Step 1.
  std::size_t current_bytes = 0;
  if (const cudaError_t error = cudaDeviceGetLimit(&current_bytes, cudaLimitStackSize); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (current_bytes >= bytes_per_thread) {
    return RunOnCallStatus::kDone;
  }
  // Step 2: the driver takes the memory now, so a GPU too full to hold it says so here, at build time.
  const cudaError_t set_error = cudaDeviceSetLimit(cudaLimitStackSize, static_cast<std::size_t>(bytes_per_thread));
  if (set_error == cudaErrorMemoryAllocation) {
    // Not a sticky error: reading it clears it, so the next GPU call does not report it again.
    static_cast<void>(cudaGetLastError());
    return RunOnCallStatus::kOutOfMemory;
  }
  if (set_error != cudaSuccess) {
    return ToRunOnCallStatus(set_error);
  }
  // Step 3: the driver may round the value; it must not be below what the visits need.
  std::size_t given_bytes = 0;
  if (const cudaError_t error = cudaDeviceGetLimit(&given_bytes, cudaLimitStackSize); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  return given_bytes >= bytes_per_thread ? RunOnCallStatus::kDone : RunOnCallStatus::kCallFailed;
}

}  // namespace fluidity::contact
