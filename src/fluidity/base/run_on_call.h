// How a call to the GPU half of an operation ended, and how many thread blocks a GPU function starts.
//
// Purpose: give every GPU half (`<name>_gpu.cu`) one status to return and one rule for the size of a launch, so the
//   CPU half of every operation turns a GPU failure into its error the same way.
//
// Idea: a GPU half returns kDone, or kOutOfMemory when an allocation failed, or kSizeMismatch when a check that the GPU
//   half made found the inputs out of range before it wrote anything (a total too large, a destination out of range),
//   or kCallFailed for any other failed GPU call. The CPU half turns the status into its own error; ToBufferError
//   (base/buffer.h) is that mapping for BufferError. A GPU function runs as thread blocks of kThreadBlockSize threads,
//   at most kMaxThreadBlockCount thread blocks; each thread handles the items i, i + (number of threads), and so on, so
//   a fixed number of thread blocks covers any count.
//   Example: ComputeLaunchSize(1000) gives 4 thread blocks of 256 threads; ComputeLaunchSize(10^9) gives 65,535
//   thread blocks, and each thread handles about 60 items.
//
// Words:
//   thread block  the threads of one GPU launch unit, which can share fast memory and wait for each other; never
//                 "block" or "group" alone in that sense
// Owns:    RunOnCallStatus, kThreadBlockSize, kMaxThreadBlockCount, LaunchSize, ComputeLaunchSize
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h)
// Used by: every private GPU half (`<name>_gpu.h`, `<name>_gpu.cu`), base/gpu_launch.cuh, ToBufferError
//          (base/buffer.h)
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

#include <cstddef>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"

namespace fluidity {

// How a call to a GPU half ended. The CPU half turns it into its error: ToBufferError for BufferError.
enum class RunOnCallStatus : std::uint8_t {
  kDone,  // The outputs hold the answer.
  kOutOfMemory,  // An allocation of the call failed. Nothing was written. Only allocations return it.
  kSizeMismatch,  // A GPU half's check found the inputs out of range (a total above 2^32 - 1). Nothing written.
  kCallFailed,  // Another GPU call failed. The outputs are unspecified.
};

// Threads in one thread block of a simple GPU function: one thread per item, as many items as threads.
inline constexpr unsigned int kThreadBlockSize = 256;

// The most thread blocks a simple GPU function starts. The GPU accepts up to 2^31 - 1 thread blocks along x; fewer
// are started, and each thread loops over several items, so a count of any size works.
inline constexpr unsigned int kMaxThreadBlockCount = 65535;

// The shape of one launch of a GPU function.
struct LaunchSize {
  unsigned int thread_block_count = 0;  // Thread blocks; at least 1 for a launch.
  unsigned int thread_block_size = 0;  // Threads in each thread block.
};

// The launch of a simple GPU function over `item_count` items: one thread per item in thread blocks of
// kThreadBlockSize, at most kMaxThreadBlockCount thread blocks. item_count 0 gives 0 thread blocks, a launch the GPU
// fails on: a caller with no items does not launch.
// Example: 1 item gives 1 thread block; 256 give 1; 257 give 2; 10^9 give 65,535.
FLUIDITY_CPU_AND_GPU constexpr LaunchSize ComputeLaunchSize(std::size_t item_count) {
  const std::size_t needed = item_count / kThreadBlockSize + (item_count % kThreadBlockSize != 0 ? 1 : 0);
  const std::size_t thread_block_count = needed < kMaxThreadBlockCount ? needed : kMaxThreadBlockCount;
  return {.thread_block_count = static_cast<unsigned int>(thread_block_count), .thread_block_size = kThreadBlockSize};
}

}  // namespace fluidity
