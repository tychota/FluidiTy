// The GPU half of Buffer, private to base/: the declarations of what base/buffer_gpu.cu defines.
//
// Purpose: let base/buffer_cpu.cc reach the CUDA runtime without including a CUDA header.
//
// Idea: each function wraps one CUDA runtime call on raw bytes and returns a RunOnCallStatus; the caller turns it into
//   a BufferError with ToBufferError. An allocation that fails for any reason is kOutOfMemory; a copy, a fill or a free
//   that fails is kCallFailed, whatever its CUDA error (ToRunOnCallStatus of base/gpu_launch.cuh). They exist only in a
//   build with GPU code, so base/buffer_cpu.cc names them only under `#if defined(FLUIDITY_HAS_GPU)`. This file uses
//   the C++20 subset and includes no CPU-only header, so base/buffer_gpu.cu, which defines the functions, includes it
//   too. Callers use base/buffer.h.
//
// Owns:    AllocateGpuBytes, FreeGpuBytes, WriteGpuBytes, ReadGpuBytes, CopyGpuBytes, ZeroGpuBytes,
//          AllocateLockedBytes, FreeLockedBytes
// Uses:    RunOnCallStatus (base/run_on_call.h)
// Used by: base/buffer_cpu.cc, and base/buffer_gpu.cu, which defines them
// Read:    docs/method-paper/fluidity-method.typ <sec-buffer>

#pragma once

#include <cstddef>

#include "fluidity/base/run_on_call.h"

namespace fluidity {

// Allocates `byte_count` bytes of GPU memory (cudaMalloc) and sets *bytes to them. `byte_count` is above 0. On
// failure returns kOutOfMemory and sets *bytes to null.
RunOnCallStatus AllocateGpuBytes(std::size_t byte_count, void** bytes);

// Frees GPU memory from AllocateGpuBytes (cudaFree). Returns kCallFailed when the call failed.
RunOnCallStatus FreeGpuBytes(void* bytes);

// Copies `byte_count` bytes from CPU memory `from` to GPU memory `to` (cudaMemcpy). Returns when `from` may be
// reused; the GPU may still be finishing the copy, and later GPU calls see it complete. Fails with kCallFailed.
RunOnCallStatus WriteGpuBytes(const void* from, std::size_t byte_count, void* to);

// Copies `byte_count` bytes from GPU memory `from` to CPU memory `to` (cudaMemcpy). Returns when `to` holds them.
// Fails with kCallFailed.
RunOnCallStatus ReadGpuBytes(const void* from, std::size_t byte_count, void* to);

// Copies `byte_count` bytes from GPU memory `from` to GPU memory `to`, which do not overlap (cudaMemcpy), and waits
// for the GPU, so an error of the copy itself is reported here. Fails with kCallFailed.
RunOnCallStatus CopyGpuBytes(const void* from, std::size_t byte_count, void* to);

// Sets `byte_count` bytes of GPU memory to zero (cudaMemset) and waits for the GPU, so an error of the fill itself
// is reported here. Fails with kCallFailed.
RunOnCallStatus ZeroGpuBytes(void* bytes, std::size_t byte_count);

// Allocates `byte_count` bytes of CPU memory locked for copies (cudaHostAlloc) and sets *bytes to them.
// `byte_count` is above 0. On failure returns kOutOfMemory and sets *bytes to null.
RunOnCallStatus AllocateLockedBytes(std::size_t byte_count, void** bytes);

// Frees CPU memory from AllocateLockedBytes (cudaFreeHost). Returns kCallFailed when the call failed.
RunOnCallStatus FreeLockedBytes(void* bytes);

}  // namespace fluidity
