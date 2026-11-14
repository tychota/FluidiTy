// The GPU half of ReadMemoryAvailable, private to base/: the declaration of what
// base/processor_description_gpu.cu defines.
//
// Purpose: let base/processor_description_cpu.cc read the GPU's memory without including a CUDA header.
//
// Idea: one function wraps cudaMemGetInfo. It exists only in a build with GPU code, so the CPU file names it only
//   under `#if defined(FLUIDITY_HAS_GPU)`. This file uses the C++20 subset and includes no CPU-only header.
//   Callers use base/processor_description.h.
//
// Owns:    ReadGpuMemoryAvailable
// Uses:    RunOnCallStatus (base/run_on_call.h)
// Used by: base/processor_description_cpu.cc, and base/processor_description_gpu.cu, which defines it
// Read:    docs/method-paper/fluidity-method.typ <sec-buffer>

#pragma once

#include <cstdint>

#include "fluidity/base/run_on_call.h"

namespace fluidity {

// Reads the total and the free memory of the current GPU, in bytes (cudaMemGetInfo). On failure returns how the call
// failed and changes neither output.
RunOnCallStatus ReadGpuMemoryAvailable(std::uint64_t* total_bytes, std::uint64_t* free_bytes);

}  // namespace fluidity
