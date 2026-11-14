// An array of fixed size that lives on the processor the simulation runs on.
//
// Purpose: give every module one way to hold its arrays on the CPU or on the GPU, and one way to move values
//   between CPU memory and those arrays.
//
// Idea: a buffer is built once for a processor (RunOn) and a size, and never grows. Its memory starts as zero
//   bytes and is freed when the buffer is destroyed. Only WriteBuffer and ReadBuffer move values between CPU
//   memory and a buffer, so every copy to or from the GPU is visible at the call. CopyBuffer copies between two
//   buffers on the same processor. A slice is a run of `count` items starting at index `first`; each copy has a
//   form that takes one. Every function checks sizes, slices and overlap before it touches memory. The span
//   parameters take a std::vector or a std::array directly: the value type comes from the buffer.
//   Example: a buffer of 5 doubles holds {1, 2, 3, 4, 5}, and `const std::array<double, 2> values = {20, 30};`.
//   WriteBuffer(values, buffer, 1) makes the buffer {1, 20, 30, 4, 5}. WriteBuffer(values, buffer, 4) returns
//   kSizeMismatch and changes nothing. (A braced list such as {20, 30} does not convert to a std::span.)
//   Two layers: the typed functions of this file check sizes and slices, the only checks that need the value type.
//   Each then calls an untyped byte function of base/buffer_bytes.h, which picks the processor and copies with
//   memcpy or the CUDA runtime, so the templates hold no `#if` and one source file chooses the processor.
//
//     Buffer<Value> functions          base/buffer.h        check sizes and slices
//       |
//       v
//     byte functions                   base/buffer_bytes.h  pick the processor (defined in base/buffer_cpu.cc)
//       |                    |
//       v                    v
//     memcpy                 CUDA runtime
//     base/buffer_cpu.cc     base/buffer_gpu.cu
//
// Words:
//   buffer             an array of fixed size on the processor chosen by the caller
//   slice              `count` items of a buffer starting at index `first`
//   locked for copies  CPU memory that the operating system never moves or swaps out, so the GPU copies it
//                      directly; papers and the CUDA manual call it page-locked or pinned memory
// Owns:    Buffer, BufferValue, BufferError, CpuMemory, BuildBuffer, WriteBuffer, ReadBuffer, CopyBuffer,
//          ToMessage(BufferError), ToName(BufferError), ToBufferError, kBufferAlignmentBytes, CheckSliceIsInside
// Uses:    RunOn (base/run_on.h), RunOnCallStatus (base/run_on_call.h), CPU-only guard (base/cpu_only.h), the private
//          byte functions (base/buffer_bytes.h)
// Used by: every module that keeps per-cell, per-face or per-particle arrays
// Read:    docs/method-paper/fluidity-method.typ <sec-buffer>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "fluidity/base/buffer_bytes.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"

namespace fluidity {

// Why a buffer function failed. Every function of base/ that works on buffers returns it, and a module maps it into
// its own failure (AGENTS.md, "One error enum per module").
enum class BufferError : std::uint8_t {
  kNone,  // No error. No function returns it as an error.
  kOutOfMemory,  // The chosen processor could not hold the array. Only allocations return it; nothing was written.
  // CheckCanRunOn failed for the chosen processor, or a call to it failed. The reason is not kept here: the outer
  // boundary calls CheckCanRunOn(run_on) before any work, which returns the RunOnError and its message.
  kRunOnCallFailed,
  // Sizes do not fit: a span and a buffer, or two buffers, have different sizes; a slice is out of range; two ranges
  // overlap; or a count or a total is above what the function can index (each function's Fails says which). The
  // last case can come from a scene that is too large, not only from a bug in the caller.
  kSizeMismatch,
};

// Converts how a GPU half ended (base/run_on_call.h) to its BufferError: kDone gives kNone, kOutOfMemory gives
// kOutOfMemory, kSizeMismatch gives kSizeMismatch and kCallFailed gives kRunOnCallFailed. The one mapping every GPU
// half's status goes through. An out-of-range value gives kRunOnCallFailed.
BufferError ToBufferError(RunOnCallStatus status);

// Converts an error to a plain sentence that says what the user can do. Never empty; each error has its own text.
// An out-of-range value gives the text of an unknown error.
std::string_view ToMessage(BufferError error);

// Converts an error to its code: the enumerator without the k, in snake_case, such as "size_mismatch". The command
// line and the Blender add-on use it, so a code never changes once published. An out-of-range value gives "unknown".
std::string_view ToName(BufferError error);

// The kind of CPU memory a CPU buffer asks for. GPU buffers ignore it.
enum class CpuMemory : std::uint8_t {
  kOrdinary,  // Ordinary memory.
  // Locked for copies, for CPU buffers copied to the GPU every step (a batched run). Copies are faster, but the
  // machine has less of this memory than of ordinary memory. In a build without GPU code, or when no GPU is
  // usable, the buffer gets ordinary memory instead: without a GPU nothing gains from locking.
  kLockedForCopies,
};

// Bytes. Every CPU buffer's data() is a multiple of this, so a cache line of 64 bytes never holds the end of one
// buffer and the start of another. Locked memory starts at a page, whose size is a multiple of 64 bytes, and GPU
// memory starts at a multiple of 256 bytes, so the promise holds for every buffer with memory. Page alignment of
// locked memory is observed, not documented; a locked address that is not a multiple of 64 bytes is freed and the
// buffer gets ordinary memory instead.
inline constexpr std::size_t kBufferAlignmentBytes = 64;

// A type a buffer can hold: copying its bytes copies the value (trivially copyable), so values move between the
// processors with a plain byte copy, and its alignment is at most kBufferAlignmentBytes. Every value is written in
// the C++20 subset that GPU code can use.
template <typename Value>
concept BufferValue = std::is_trivially_copyable_v<Value> && alignof(Value) <= kBufferAlignmentBytes;

template <BufferValue Value>
class Buffer;

// Builds a buffer of `size` values on `run_on`, every byte zero.
//
// Method: checks the byte count, then, for RunOn::kGpu, calls CheckCanRunOn(RunOn::kGpu), whose answer is computed
//   once per process; then allocates GPU memory and fills it with zero bytes. For RunOn::kCpu, allocates CPU memory
//   aligned to kBufferAlignmentBytes and fills it with zero bytes. With CpuMemory::kLockedForCopies and a usable
//   GPU, the CPU memory is locked for copies; otherwise it is ordinary. Size 0 allocates nothing and data() is null.
// Example: BuildBuffer<double>(RunOn::kCpu, 1000) gives size() == 1000, run_on() == kCpu and 1000 zeros.
// Fails: kOutOfMemory     size * sizeof(Value) is above the largest std::ptrdiff_t, the largest size any object can
//                         have (checked first, so the product never wraps around), or the processor could not
//                         provide the memory
//        kRunOnCallFailed RunOn::kGpu was asked for and CheckCanRunOn failed (also in a build without GPU code),
//                         or the call that fills the GPU memory with zeros failed
//        On failure nothing stays allocated and no buffer is returned.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>
template <BufferValue Value>
std::expected<Buffer<Value>, BufferError>
BuildBuffer(RunOn run_on, std::size_t size, CpuMemory cpu_memory = CpuMemory::kOrdinary);

// An array of `size()` values on the processor `run_on()`. The values are contiguous, start at a multiple of
// kBufferAlignmentBytes, and are zero bytes when the buffer is built.
//
// Move-only: copying is a compile error; CopyBuffer copies the values. A moved-from buffer has size 0, null data,
// and keeps its run_on(). The size never changes: a new size is a new buffer. The destructor frees the memory.
// When freeing GPU or locked memory fails there, the failure cannot be reported and is ignored: the GPU is then
// already broken, and the next GPU call reports it.
template <BufferValue Value>
class Buffer {
 public:
  // An empty CPU buffer: size 0, null data. BuildBuffer makes buffers that hold values.
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept
      : values_(std::exchange(other.values_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        run_on_(other.run_on_),
        cpu_memory_used_(other.cpu_memory_used_) {
  }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      FreeBufferBytes(run_on_, cpu_memory_used_, values_);
      values_ = std::exchange(other.values_, nullptr);
      size_ = std::exchange(other.size_, 0);
      run_on_ = other.run_on_;
      cpu_memory_used_ = other.cpu_memory_used_;
    }
    return *this;
  }
  ~Buffer() {
    FreeBufferBytes(run_on_, cpu_memory_used_, values_);
  }

  // The number of values. 0 for an empty or moved-from buffer.
  [[nodiscard]] std::size_t size() const {
    return size_;
  }

  // The processor the values live on.
  [[nodiscard]] RunOn run_on() const {
    return run_on_;
  }

  // The first value: a CPU pointer for RunOn::kCpu, a GPU pointer for RunOn::kGpu. Use it on run_on() only.
  // May be null when size() is 0.
  [[nodiscard]] Value* data() {
    return values_;
  }
  [[nodiscard]] const Value* data() const {
    return values_;
  }

 private:
  template <BufferValue BuiltValue>
  friend std::expected<Buffer<BuiltValue>, BufferError>
  BuildBuffer(RunOn run_on, std::size_t size, CpuMemory cpu_memory);

  // Takes ownership of `values`, built by BuildBufferBytes as `cpu_memory_used` says.
  Buffer(Value* values, std::size_t size, RunOn run_on, CpuMemory cpu_memory_used)
      : values_(values), size_(size), run_on_(run_on), cpu_memory_used_(cpu_memory_used) {
  }

  Value* values_ = nullptr;
  std::size_t size_ = 0;
  RunOn run_on_ = RunOn::kCpu;
  // Which CPU memory holds the values. kLockedForCopies only when locked memory was really obtained, so the
  // destructor frees it the same way. Not read for a GPU buffer.
  CpuMemory cpu_memory_used_ = CpuMemory::kOrdinary;
};

// BuildBuffer: see its declaration above Buffer.
template <BufferValue Value>
std::expected<Buffer<Value>, BufferError> BuildBuffer(RunOn run_on, std::size_t size, CpuMemory cpu_memory) {
  // The byte count must not wrap around, or the allocation would get a smaller size. No object, on either
  // processor, can be larger than the largest std::ptrdiff_t, because a difference of two pointers into it must fit
  // there; such a size is out of memory before any allocation is tried.
  constexpr std::size_t kLargestByteCount = std::numeric_limits<std::ptrdiff_t>::max();
  if (size > kLargestByteCount / sizeof(Value)) {
    return std::unexpected(BufferError::kOutOfMemory);
  }
  const std::expected<BufferBytes, BufferError> built = BuildBufferBytes(run_on, size * sizeof(Value), cpu_memory);
  if (!built.has_value()) {
    return std::unexpected(built.error());
  }
  return Buffer<Value>(static_cast<Value*>(built->bytes), size, run_on, built->cpu_memory_used);
}

// Checks that `count` items starting at index `first` lie inside a buffer of `size` items. Changes nothing.
//
// Method: compares without any sum, so a huge `first` cannot wrap around and pass.
// Example: size 5: (first 3, count 2) is inside; (first 5, count 0) is inside; (first 4, count 2) is not.
// Fails: kSizeMismatch  first > size, or count > size - first
constexpr std::expected<void, BufferError> CheckSliceIsInside(std::size_t first, std::size_t count, std::size_t size) {
  const bool first_is_inside = first <= size;
  if (!first_is_inside || count > size - first) {
    return std::unexpected(BufferError::kSizeMismatch);
  }
  return {};
}

// Copies `values` into the buffer, starting at index `first`: a write to the GPU for a GPU buffer.
//
// Method: checks the slice, then that `values` does not overlap the slice (a span into the same buffer), then
//   copies values.size() items. An empty span at first == buffer.size() succeeds and copies nothing. `values` may
//   be a std::vector or a std::array: the value type comes from the buffer.
// Example: buffer {1, 2, 3, 4, 5}, values {20, 30}, first 1: the buffer becomes {1, 20, 30, 4, 5}.
// Fails: kSizeMismatch    first + values.size() is beyond buffer.size(), or `values` overlaps the slice
//        kRunOnCallFailed the copy to the GPU failed
//        A failed check writes nothing. After a failed GPU call the target's values in the slice are
//        unspecified, a mix of old and new; values outside the slice are unchanged.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>
template <BufferValue Value>
std::expected<void, BufferError>
WriteBuffer(std::type_identity_t<std::span<const Value>> values, Buffer<Value>& buffer, std::size_t first) {
  if (const std::expected<void, BufferError> slice = CheckSliceIsInside(first, values.size(), buffer.size());
      !slice.has_value()) {
    return slice;
  }
  return WriteBufferBytes(values.data(), values.size_bytes(), buffer.run_on(), buffer.data() + first);
}

// Copies `values` into the whole buffer: a write to the GPU for a GPU buffer.
//
// Example: buffer of 3, values {1, 2, 3}: the buffer holds {1, 2, 3}.
// Fails: kSizeMismatch    values.size() differs from buffer.size(), or `values` overlaps the buffer
//        kRunOnCallFailed the copy to the GPU failed
//        A failed check writes nothing. After a failed GPU call the target's values in the slice are
//        unspecified, a mix of old and new; values outside the slice are unchanged.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>
template <BufferValue Value>
std::expected<void, BufferError>
WriteBuffer(std::type_identity_t<std::span<const Value>> values, Buffer<Value>& buffer) {
  if (values.size() != buffer.size()) {
    return std::unexpected(BufferError::kSizeMismatch);
  }
  return WriteBuffer(values, buffer, 0);
}

// Copies values.size() items of the buffer, starting at index `first`, into `values`: a read from the GPU for a GPU
// buffer.
//
// Method: checks the slice, then that `values` does not overlap the slice (a span into the same buffer), then
//   copies. An empty span at first == buffer.size() succeeds and copies nothing.
// Example: buffer {1, 2, 3, 4, 5}, first 2, a span of 3: the span holds {3, 4, 5}.
// Fails: kSizeMismatch    first + values.size() is beyond buffer.size(), or `values` overlaps the slice
//        kRunOnCallFailed the copy from the GPU failed
//        A failed check writes nothing. After a failed GPU call the target's values in the slice are
//        unspecified, a mix of old and new; values outside the slice are unchanged.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>
template <BufferValue Value>
std::expected<void, BufferError>
ReadBuffer(const Buffer<Value>& buffer, std::size_t first, std::type_identity_t<std::span<Value>> values) {
  if (const std::expected<void, BufferError> slice = CheckSliceIsInside(first, values.size(), buffer.size());
      !slice.has_value()) {
    return slice;
  }
  return ReadBufferBytes(buffer.run_on(), buffer.data() + first, values.size_bytes(), values.data());
}

// Copies the whole buffer into `values`: a read from the GPU for a GPU buffer.
//
// Example: buffer {1, 2, 3}, a span of 3: the span holds {1, 2, 3}.
// Fails: kSizeMismatch    values.size() differs from buffer.size(), or `values` overlaps the buffer
//        kRunOnCallFailed the copy from the GPU failed
//        A failed check writes nothing. After a failed GPU call the target's values in the slice are
//        unspecified, a mix of old and new; values outside the slice are unchanged.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>
template <BufferValue Value>
std::expected<void, BufferError>
ReadBuffer(const Buffer<Value>& buffer, std::type_identity_t<std::span<Value>> values) {
  if (values.size() != buffer.size()) {
    return std::unexpected(BufferError::kSizeMismatch);
  }
  return ReadBuffer(buffer, 0, values);
}

// Copies `count` items of `from`, starting at index `first`, into `to`, starting at index `to_first`. The copy runs
// on the processor both buffers live on, so values never pass through the other processor.
//
// Method: checks both slices, then that the slices do not overlap, then copies. Overlapping slices of one buffer
//   are not copied: a plain byte copy between them would mix old and new values, so the call returns
//   kSizeMismatch. Two separate slices of one buffer are copied. count 0 succeeds and copies nothing.
// Example: from {1, 2, 3, 4}, first 1, count 2, to {0, 0, 0}, to_first 0: `to` becomes {2, 3, 0}.
//   From and to the same buffer {1, 2, 3, 4} with first 0, count 3, to_first 1: kSizeMismatch, nothing changes.
// Requires: from.run_on() == to.run_on(), checked by assert in debug builds; WriteBuffer and ReadBuffer cross
//   between processors.
// Fails: kSizeMismatch    a slice is beyond its buffer's size, or the two slices overlap in one buffer
//        kRunOnCallFailed the copy on the GPU failed
//        A failed check writes nothing. After a failed GPU call the target's values in the slice are
//        unspecified, a mix of old and new; values outside the slice are unchanged.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>
template <BufferValue Value>
std::expected<void, BufferError>
CopyBuffer(const Buffer<Value>& from, std::size_t first, std::size_t count, Buffer<Value>& to, std::size_t to_first) {
  // Requires: one processor. A copy between processors is a bug in the caller: WriteBuffer and ReadBuffer cross.
  assert(from.run_on() == to.run_on());
  if (const std::expected<void, BufferError> slice = CheckSliceIsInside(first, count, from.size());
      !slice.has_value()) {
    return slice;
  }
  if (const std::expected<void, BufferError> slice = CheckSliceIsInside(to_first, count, to.size());
      !slice.has_value()) {
    return slice;
  }
  // Both slices are inside their buffers, so count * sizeof(Value) cannot wrap around. CopyBufferBytes refuses
  // overlapping byte ranges.
  return CopyBufferBytes(to.run_on(), from.data() + first, count * sizeof(Value), to.data() + to_first);
}

// Copies every value of `from` into `to`, on the processor both buffers live on.
//
// Method: checks the sizes, then copies. A buffer copied onto itself (same object, size above 0) is an overlap and
//   returns kSizeMismatch.
// Example: from {1, -2, 3.5}, to {0, 0, 0}: `to` becomes {1, -2, 3.5}.
// Requires: from.run_on() == to.run_on(), checked by assert in debug builds.
// Fails: kSizeMismatch    the sizes differ, or `from` and `to` are the same buffer
//        kRunOnCallFailed the copy on the GPU failed
//        A failed check writes nothing. After a failed GPU call the target's values in the slice are
//        unspecified, a mix of old and new; values outside the slice are unchanged.
// Read: docs/method-paper/fluidity-method.typ <sec-buffer>
template <BufferValue Value>
std::expected<void, BufferError> CopyBuffer(const Buffer<Value>& from, Buffer<Value>& to) {
  assert(from.run_on() == to.run_on());
  if (from.size() != to.size()) {
    return std::unexpected(BufferError::kSizeMismatch);
  }
  return CopyBuffer(from, 0, from.size(), to, 0);
}

}  // namespace fluidity
