#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <new>
#include <string_view>

#include "fluidity/base/buffer.h"
#include "fluidity/base/buffer_bytes.h"
#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"

#if defined(FLUIDITY_HAS_GPU)
#include "fluidity/base/buffer_gpu.h"
#endif

namespace fluidity {
namespace {

// Whether the byte ranges [first, first + byte_count) and [second, second + byte_count) share a byte. Compares
// addresses as integers: comparing pointers into two different arrays with < is not defined in C++.
bool ByteRangesOverlap(const void* first, const void* second, std::size_t byte_count) {
  const auto first_address = std::bit_cast<std::uintptr_t>(first);
  const auto second_address = std::bit_cast<std::uintptr_t>(second);
  return first_address < second_address + byte_count && second_address < first_address + byte_count;
}

// Ordinary CPU memory: aligned, zero-filled. Null when the machine cannot provide it.
void* BuildOrdinaryBytes(std::size_t byte_count) {
  void* const bytes = ::operator new(byte_count, std::align_val_t{kBufferAlignmentBytes}, std::nothrow);
  if (bytes != nullptr) {
    std::memset(bytes, 0, byte_count);
  }
  return bytes;
}

#if defined(FLUIDITY_HAS_GPU)

// GPU memory, zero-filled.
std::expected<BufferBytes, BufferError> BuildGpuBytes(std::size_t byte_count) {
  void* bytes = nullptr;
  if (const RunOnCallStatus status = AllocateGpuBytes(byte_count, &bytes); status != RunOnCallStatus::kDone) {
    return std::unexpected(ToBufferError(status));
  }
  if (const RunOnCallStatus status = ZeroGpuBytes(bytes, byte_count); status != RunOnCallStatus::kDone) {
    // The fill failed after the allocation worked: the driver is failing. Free so nothing stays allocated.
    static_cast<void>(FreeGpuBytes(bytes));
    return std::unexpected(ToBufferError(status));
  }
  return BufferBytes{.bytes = bytes, .cpu_memory_used = CpuMemory::kOrdinary};
}

// Locked CPU memory, zero-filled, or null bytes when the address it got is not a multiple of kBufferAlignmentBytes.
std::expected<BufferBytes, BufferError> BuildLockedBytes(std::size_t byte_count) {
  void* bytes = nullptr;
  if (const RunOnCallStatus status = AllocateLockedBytes(byte_count, &bytes); status != RunOnCallStatus::kDone) {
    // Locked memory is scarcer than ordinary memory. The caller asked for it, so the failure is reported rather
    // than silently replaced.
    return std::unexpected(ToBufferError(status));
  }
  // cudaHostAlloc has always returned page-aligned memory, but its manual does not promise it: this is observed,
  // not documented. A misaligned address would break the alignment promise of Buffer, so the caller falls back.
  if (std::bit_cast<std::uintptr_t>(bytes) % kBufferAlignmentBytes != 0) {
    static_cast<void>(FreeLockedBytes(bytes));
    return BufferBytes{};
  }
  std::memset(bytes, 0, byte_count);
  return BufferBytes{.bytes = bytes, .cpu_memory_used = CpuMemory::kLockedForCopies};
}

#endif

}  // namespace

std::expected<BufferBytes, BufferError> BuildBufferBytes(RunOn run_on, std::size_t byte_count, CpuMemory cpu_memory) {
  // Steps: 1 check the processor  2 GPU memory  3 locked CPU memory  4 ordinary CPU memory

  // Step 1: the GPU must be usable before anything is asked of it, even for 0 bytes.
  if (run_on == RunOn::kGpu && !CheckCanRunOn(RunOn::kGpu).has_value()) {
    // No GPU code, no GPU, or a GPU whose math differs: the caller can choose the CPU or fix the machine.
    return std::unexpected(BufferError::kRunOnCallFailed);
  }
  if (byte_count == 0) {
    return BufferBytes{};
  }

#if defined(FLUIDITY_HAS_GPU)
  // Step 2: GPU memory.
  if (run_on == RunOn::kGpu) {
    return BuildGpuBytes(byte_count);
  }
  // Step 3: locked CPU memory, only when asked for and a GPU can use it.
  if (cpu_memory == CpuMemory::kLockedForCopies && CheckCanRunOn(RunOn::kGpu).has_value()) {
    const std::expected<BufferBytes, BufferError> locked = BuildLockedBytes(byte_count);
    if (!locked.has_value() || locked->bytes != nullptr) {
      return locked;
    }
  }
#else
  // Without GPU code there is no GPU to lock memory for, so the request gets ordinary memory.
  static_cast<void>(cpu_memory);
#endif

  // Step 4: ordinary CPU memory.
  void* const bytes = BuildOrdinaryBytes(byte_count);
  if (bytes == nullptr) {
    return std::unexpected(BufferError::kOutOfMemory);
  }
  return BufferBytes{.bytes = bytes, .cpu_memory_used = CpuMemory::kOrdinary};
}

void FreeBufferBytes(RunOn run_on, CpuMemory cpu_memory_used, void* bytes) {
  if (bytes == nullptr) {
    return;
  }
#if defined(FLUIDITY_HAS_GPU)
  if (run_on == RunOn::kGpu) {
    static_cast<void>(FreeGpuBytes(bytes));
    return;
  }
  if (cpu_memory_used == CpuMemory::kLockedForCopies) {
    static_cast<void>(FreeLockedBytes(bytes));
    return;
  }
#else
  // Without GPU code every buffer with memory is an ordinary CPU buffer.
  static_cast<void>(run_on);
  static_cast<void>(cpu_memory_used);
#endif
  ::operator delete(bytes, std::align_val_t{kBufferAlignmentBytes});
}

std::expected<void, BufferError> WriteBufferBytes(const void* from, std::size_t byte_count, RunOn run_on, void* to) {
  if (byte_count == 0) {
    return {};
  }
  if (ByteRangesOverlap(from, to, byte_count)) {
    // The span points into the buffer's own slice: a byte copy would mix old and new values.
    return std::unexpected(BufferError::kSizeMismatch);
  }
#if defined(FLUIDITY_HAS_GPU)
  if (run_on == RunOn::kGpu) {
    if (const RunOnCallStatus status = WriteGpuBytes(from, byte_count, to); status != RunOnCallStatus::kDone) {
      return std::unexpected(ToBufferError(status));
    }
    return {};
  }
#else
  static_cast<void>(run_on);
#endif
  std::memcpy(to, from, byte_count);
  return {};
}

std::expected<void, BufferError> ReadBufferBytes(RunOn run_on, const void* from, std::size_t byte_count, void* to) {
  if (byte_count == 0) {
    return {};
  }
  if (ByteRangesOverlap(from, to, byte_count)) {
    // The span points into the buffer's own slice: a byte copy would mix old and new values.
    return std::unexpected(BufferError::kSizeMismatch);
  }
#if defined(FLUIDITY_HAS_GPU)
  if (run_on == RunOn::kGpu) {
    if (const RunOnCallStatus status = ReadGpuBytes(from, byte_count, to); status != RunOnCallStatus::kDone) {
      return std::unexpected(ToBufferError(status));
    }
    return {};
  }
#else
  static_cast<void>(run_on);
#endif
  std::memcpy(to, from, byte_count);
  return {};
}

std::expected<void, BufferError> CopyBufferBytes(RunOn run_on, const void* from, std::size_t byte_count, void* to) {
  if (byte_count == 0) {
    return {};
  }
  if (ByteRangesOverlap(from, to, byte_count)) {
    // Two overlapping slices of one buffer: a byte copy would mix old and new values.
    return std::unexpected(BufferError::kSizeMismatch);
  }
#if defined(FLUIDITY_HAS_GPU)
  if (run_on == RunOn::kGpu) {
    if (const RunOnCallStatus status = CopyGpuBytes(from, byte_count, to); status != RunOnCallStatus::kDone) {
      return std::unexpected(ToBufferError(status));
    }
    return {};
  }
#else
  static_cast<void>(run_on);
#endif
  std::memcpy(to, from, byte_count);
  return {};
}

std::string_view ToMessage(BufferError error) {
  switch (error) {
    case BufferError::kNone:
      return "There is no error.";
    case BufferError::kOutOfMemory:
      return "The chosen processor does not have enough free memory for the array. Use a smaller scene, free memory, "
             "or choose the other processor.";
    case BufferError::kRunOnCallFailed:
      return "The chosen processor cannot run here, or a call to it failed. Check the GPU and its driver, or choose "
             "the CPU.";
    case BufferError::kSizeMismatch:
      return "Sizes do not fit: two arrays differ in size, a range lies outside its array, or a count is above what "
             "the operation can handle. Check the sizes passed in; a scene too large for that limit gives this too.";
  }
  return "An unknown buffer error happened.";
}

BufferError ToBufferError(RunOnCallStatus status) {
  switch (status) {
    case RunOnCallStatus::kDone:
      return BufferError::kNone;
    case RunOnCallStatus::kOutOfMemory:
      return BufferError::kOutOfMemory;
    case RunOnCallStatus::kSizeMismatch:
      return BufferError::kSizeMismatch;
    case RunOnCallStatus::kCallFailed:
      return BufferError::kRunOnCallFailed;
  }
  return BufferError::kRunOnCallFailed;
}

std::string_view ToName(BufferError error) {
  switch (error) {
    case BufferError::kNone:
      return "none";
    case BufferError::kOutOfMemory:
      return "out_of_memory";
    case BufferError::kRunOnCallFailed:
      return "run_on_call_failed";
    case BufferError::kSizeMismatch:
      return "size_mismatch";
  }
  return "unknown";
}

}  // namespace fluidity
