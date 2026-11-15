// The GPU half of ReadMemoryAvailable and ReadProcessorDescription, private to base/: the declaration of what
// base/processor_description_gpu.cu defines.
//
// Purpose: let base/processor_description_cpu.cc read the GPU's memory, name and versions without including a CUDA
//   header.
//
// Idea: one function wraps cudaMemGetInfo, and one wraps the CUDA calls that name the GPU and its versions. Both exist
//   only in a build with GPU code, so the CPU file names them only under `#if defined(FLUIDITY_HAS_GPU)`. This file
//   uses the C++20 subset and includes no CPU-only header. Callers use base/processor_description.h.
//
// Owns:    ReadGpuMemoryAvailable, ReadGpuProcessorDescription
// Uses:    RunOnCallStatus (base/run_on_call.h)
// Used by: base/processor_description_cpu.cc, and base/processor_description_gpu.cu, which defines them
// Read:    docs/method-paper/fluidity-method.typ <sec-memory-estimate>

#pragma once

#include <cstddef>
#include <cstdint>

#include "fluidity/base/run_on_call.h"

namespace fluidity {

// Reads the total and the free memory of the current GPU, in bytes (cudaMemGetInfo), after making sure the GPU context
// exists (cudaFree(nullptr)), so the context and the GPU code loaded so far are already outside "free". On failure
// returns how the call failed and changes neither output.
RunOnCallStatus ReadGpuMemoryAvailable(std::uint64_t* total_bytes, std::uint64_t* free_bytes);

// Reads the name of the current GPU, the driver version and the CUDA runtime version into three text buffers of
// `text_size` chars each, as null-terminated text (cudaGetDeviceProperties, cudaDriverGetVersion,
// cudaRuntimeGetVersion). Versions are written as "major.minor". A query that fails leaves its buffer empty, so
// every buffer is a valid string on return. `text_size` must be at least 1.
void ReadGpuProcessorDescription(char* model, char* driver, char* runtime, std::size_t text_size);

}  // namespace fluidity
