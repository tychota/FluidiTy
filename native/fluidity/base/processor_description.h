// What a processor reports about itself: for now, how much memory it has and how much is free.
//
// Purpose: let a caller compare the memory a run needs with the memory the chosen processor has, before it builds
//   the run's buffers.
//
// Idea: the CPU answer comes from the operating software of the machine (on Apple computers, Linux or Windows):
//   the physical memory and the part of it that programs can take now. The GPU answer comes from the CUDA runtime
//   for the current GPU. Free memory changes all the time, so the answer is a reading at one moment, not a
//   promise.
//
// Owns:    MemoryAvailable, ReadMemoryAvailable
// Uses:    RunOn (base/run_on.h), BufferError (base/buffer.h), CheckCanRunOn (base/check_can_run_on.h); its
//          implementation uses the private GPU call (base/processor_description_gpu.h) in a build with GPU code
// Used by: the step's memory plan, the command line
// Read:    docs/method-paper/fluidity-method.typ <sec-buffer>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"

namespace fluidity {

// The memory of one processor at one moment.
struct MemoryAvailable {
  std::uint64_t total_bytes = 0;  // Bytes. All the memory of the processor. 0 means unknown.
  std::uint64_t free_bytes = 0;  // Bytes. The part a program could take now, at most total_bytes. 0 when unknown.
};

// Reads the total and the free memory of the processor `run_on`.
//
// Method: for RunOn::kCpu, asks the operating software for the physical memory and the memory programs can take
//   now (Apple computers: free and inactive pages; Linux: MemAvailable of /proc/meminfo, which ignores a container's
//   memory limit; Windows: available physical memory). When that query fails, total_bytes and free_bytes are 0, meaning
//   unknown; it is not an error. For RunOn::kGpu, calls CheckCanRunOn(RunOn::kGpu) first, then asks the CUDA runtime
//   about the current GPU.
// Example: a machine with 32 GiB of which 20 GiB can be taken gives {34359738368, 21474836480}.
// Fails: kRunOnCallFailed RunOn::kGpu was asked for and CheckCanRunOn failed (also in a build without GPU code),
//                         or the CUDA runtime could not answer
//        Never fails for RunOn::kCpu. On failure no reading is returned.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>
std::expected<MemoryAvailable, BufferError> ReadMemoryAvailable(RunOn run_on);

}  // namespace fluidity
