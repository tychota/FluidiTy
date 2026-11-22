// The choice of processor, and the reasons a choice cannot run.
//
// Purpose: give every shared type one small value that says where an operation runs, and give the CPU and the GPU
//   code one list of reasons the GPU cannot be used.
//
// Idea: the caller chooses the CPU or the GPU when the simulation is built. Nothing changes the choice later, and
//   nothing moves work from the GPU to the CPU without saying so. The two enums here are plain values with no
//   functions, so a shared header can carry them as members and a GPU file can include this file.
//   The functions that answer whether a choice can run, and the texts for the reasons, are in the CPU-only header
//   base/check_can_run_on.h.
//
// Words:
//   run on  the processor that runs an operation: the CPU or the GPU
// Owns:    RunOn, RunOnError
// Uses:    nothing
// Used by: CheckCanRunOn (base/check_can_run_on.h) and every other function that takes a RunOn, every shared type
//          that carries the choice
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

#include <cstdint>

namespace fluidity {

// The processor that runs an operation. The caller makes this choice; no function in the code base makes it.
enum class RunOn : std::uint8_t {
  kCpu,
  kGpu,
};

// The reasons a choice of processor cannot run here.
enum class RunOnError : std::uint8_t {
  kNone,  // No error. CheckCanRunOn never returns it as an error.
  kGpuNotBuilt,  // This build has no GPU code.
  kNoGpuFound,  // The build has GPU code but the machine has no usable GPU.
  kGpuCallFailed,  // A GPU call returned an error.
  kGpuMathDiffers,  // The shared math gave different numbers on the GPU.
};

}  // namespace fluidity
