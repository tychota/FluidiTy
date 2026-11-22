// The untyped byte functions under Buffer, private to base/. Callers use base/buffer.h.
//
// Purpose: keep every processor-specific call of Buffer in one CPU file, base/buffer_cpu.cc, so base/buffer.h
//   names no GPU call and the templates stay free of `#if`.
//
// Idea: the templates of base/buffer.h check what needs the value type (sizes, slices, the byte count) and then
//   call one of these functions with raw bytes and the buffer's processor. Each function does the rest once for
//   every value type: it allocates, frees or copies with memcpy on the CPU, or with the CUDA runtime in a build with
//   GPU code, and it fails on two byte ranges that overlap. BufferError and CpuMemory are only declared here (their
//   underlying type is fixed, so the declaration is complete); base/buffer.h defines their values.
//
// Owns:    BufferBytes, BuildBufferBytes, FreeBufferBytes, WriteBufferBytes, ReadBufferBytes, CopyBufferBytes
// Uses:    RunOn (base/run_on.h), CPU-only guard (base/cpu_only.h)
// Implementation uses: CheckCanRunOn (base/check_can_run_on.h) and, in a build with GPU code, the private GPU
//          half (base/buffer_gpu.h), which calls the CUDA runtime
// Used by: base/buffer.h; defined in base/buffer_cpu.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-buffer>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstddef>
#include <cstdint>
#include <expected>

#include "fluidity/base/run_on.h"

namespace fluidity {

// Defined, with their values, in base/buffer.h.
enum class BufferError : std::uint8_t;
enum class CpuMemory : std::uint8_t;

// Memory built by BuildBufferBytes.
struct BufferBytes {
  void* bytes = nullptr;  // Null when the byte count is 0.
  // kLockedForCopies only when locked memory was really obtained; FreeBufferBytes needs it to free the same way.
  CpuMemory cpu_memory_used{};
};

// Builds `byte_count` bytes on `run_on`, every byte zero, starting at a multiple of kBufferAlignmentBytes.
//
// Method: for RunOn::kGpu, first calls CheckCanRunOn(RunOn::kGpu). A byte count of 0 allocates nothing. On the CPU,
//   `cpu_memory` kLockedForCopies gives locked memory when a GPU is usable, otherwise ordinary memory.
// Fails: kOutOfMemory     the processor could not provide the memory
//        kRunOnCallFailed CheckCanRunOn failed for the GPU, or the GPU call that fills the memory with zeros failed
//        On failure nothing stays allocated.
std::expected<BufferBytes, BufferError> BuildBufferBytes(RunOn run_on, std::size_t byte_count, CpuMemory cpu_memory);

// Frees memory from BuildBufferBytes, the way it was allocated. Null does nothing. A failure of the GPU call cannot
// be reported and is ignored.
void FreeBufferBytes(RunOn run_on, CpuMemory cpu_memory_used, void* bytes);

// Copies `byte_count` bytes from CPU memory `from` into buffer memory `to` on `run_on`.
// Fails: kSizeMismatch    the two byte ranges overlap; nothing is written
//        kRunOnCallFailed the GPU call failed; the bytes of `to` are then unspecified, a mix of old and new
std::expected<void, BufferError> WriteBufferBytes(const void* from, std::size_t byte_count, RunOn run_on, void* to);

// Copies `byte_count` bytes from buffer memory `from` on `run_on` into CPU memory `to`.
// Fails: kSizeMismatch    the two byte ranges overlap; nothing is written
//        kRunOnCallFailed the GPU call failed; the bytes of `to` are then unspecified, a mix of old and new
std::expected<void, BufferError> ReadBufferBytes(RunOn run_on, const void* from, std::size_t byte_count, void* to);

// Copies `byte_count` bytes between two buffer memories on `run_on`.
// Fails: kSizeMismatch    the two byte ranges overlap; nothing is written
//        kRunOnCallFailed the GPU call failed; the bytes of `to` are then unspecified, a mix of old and new
std::expected<void, BufferError> CopyBufferBytes(RunOn run_on, const void* from, std::size_t byte_count, void* to);

}  // namespace fluidity
