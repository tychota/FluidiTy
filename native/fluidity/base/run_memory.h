// One reserved block of memory on one processor, which hands out buffers as pieces of itself.
//
// Purpose: let a run take all its planned memory at the start, in one reservation per processor, and build every
//   buffer it keeps from it, so the memory the run takes equals the memory it planned.
//
// Idea: BuildRunMemory reserves one block of the bytes a memory estimate gives (ComputeRunMemoryBytes of
//   base/memory_count.h) and fills it with zero bytes once. BuildBuffer with a RunMemory then hands out the next
//   bytes of the block as a piece, left to right: each piece starts at a multiple of 256 bytes from the start of the
//   block and takes its byte count rounded up to 256 (ComputeRunMemoryPieceBytes). Two rules keep it simple. A piece
//   is never freed alone: destroying a buffer built from a block frees nothing, and its bytes stay taken. The block
//   outlives its pieces: it frees all its memory at once when destroyed. One reservation cannot fragment memory, and
//   on the GPU it makes one synchronizing cudaMalloc instead of one per buffer.
//   Example: a block of 2,048 bytes; buffers of 10, 100 and 3 doubles (80, 800 and 24 bytes) take the pieces at
//   offsets 0, 256 and 1,280, so handed_out_bytes() is 1,536 and unused_bytes() is 512. A fourth buffer of 100
//   doubles needs 1,024 bytes and returns kRunMemoryTooSmall.
//
// Words:
//   RunMemory       one block of memory reserved on one processor for a run, which hands out pieces
//   piece           the bytes of one buffer inside a RunMemory's block
//   reserved bytes  the size of the block, asked once from the processor
//   handed out      bytes of the block given to pieces so far, including their rounding; never given back
// Owns:    RunMemory, BuildRunMemory, BuildBuffer(RunOn, std::size_t, RunMemory&)
// Uses:    Buffer, BufferError, CpuMemory (base/buffer.h), BufferBytes (base/buffer_bytes.h),
//          RunOn (base/run_on.h), CPU-only guard (base/cpu_only.h)
// Implementation uses: ComputeRunMemoryPieceBytes (base/memory_count.h), BuildBufferBytes and FreeBufferBytes
//          (base/buffer_bytes.h)
// Used by: the step, which reserves one RunMemory per processor; every module's buffer builders
// Read:    docs/method-paper/fluidity-method.typ <sec-run-memory>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cassert>
#include <cstddef>
#include <expected>
#include <limits>
#include <memory>

#include "fluidity/base/buffer.h"
#include "fluidity/base/buffer_bytes.h"
#include "fluidity/base/run_on.h"

namespace fluidity {

// Builds a RunMemory (below): reserves `byte_count` bytes on `run_on`, every byte zero.
//
// Method: one BuildBufferBytes call, which for RunOn::kGpu first calls CheckCanRunOn(RunOn::kGpu). On the GPU the
//   block starts at a multiple of 256 bytes (cudaMalloc); on the CPU at a multiple of kBufferAlignmentBytes (64).
//   With CpuMemory::kLockedForCopies and a usable GPU, a CPU block is locked for copies; otherwise it is ordinary
//   memory, as for BuildBuffer. The block is filled with zero bytes here, once; pieces are never handed out twice, so
//   every piece starts as zero bytes without a fill of its own. Byte count 0 reserves nothing; only empty buffers
//   can then be built from it. A byte count from ComputeRunMemoryBytes is a multiple of 256, so its pieces can use
//   every byte.
// Example: BuildRunMemory(RunOn::kCpu, 4096) gives reserved_bytes() == 4096, handed_out_bytes() == 0.
// Fails: kOutOfMemory     byte_count is above the largest std::ptrdiff_t, or the processor could not provide the
//                         memory
//        kRunOnCallFailed RunOn::kGpu was asked for and CheckCanRunOn failed (also in a build without GPU code), or
//                         the call that fills the GPU memory with zeros failed
//        kOutOfMemory also when the small CPU allocation that holds the live-piece count fails
//        On failure nothing stays reserved and no RunMemory is returned.
// Read: docs/method-paper/fluidity-method.typ <sec-run-memory>
std::expected<RunMemory, BufferError>
BuildRunMemory(RunOn run_on, std::size_t byte_count, CpuMemory cpu_memory = CpuMemory::kOrdinary);

// One block of reserved_bytes() bytes on run_on(), handed out left to right as pieces to the buffers built from it.
//
// Move-only: copying is a compile error. Moving keeps every piece valid, because the live-piece count the pieces point
// at lives in a heap allocation the RunMemory owns. A moved-from RunMemory reserves 0 bytes and has handed out
// nothing; no buffer may be built from it. The destructor frees the whole block at once. When freeing GPU or locked
// memory fails there, the failure cannot be reported and is ignored, as for a Buffer.
// Requires: the block outlives its buffers (live_piece_count() == 0 when the RunMemory is destroyed or assigned
//   over), checked by assert in debug builds. If it does not, their values are gone, but destroying them is safe:
//   the RunMemory then leaves its count allocation behind (8 bytes, never freed) for the late pieces to count down.
class RunMemory {
 public:
  RunMemory(const RunMemory&) = delete;
  RunMemory& operator=(const RunMemory&) = delete;
  RunMemory(RunMemory&& other) noexcept;
  RunMemory& operator=(RunMemory&& other) noexcept;
  ~RunMemory();

  // The processor the block lives on.
  [[nodiscard]] RunOn run_on() const {
    return run_on_;
  }

  // Bytes. The size of the block, as BuildRunMemory was asked. 0 for a moved-from RunMemory.
  [[nodiscard]] std::size_t reserved_bytes() const {
    return reserved_bytes_;
  }

  // Bytes. What the pieces have taken so far, each rounded up to a multiple of 256; destroyed pieces still count,
  // because their bytes are never handed out again. Also the offset where the next piece starts.
  [[nodiscard]] std::size_t handed_out_bytes() const {
    return handed_out_bytes_;
  }

  // Bytes. reserved_bytes() - handed_out_bytes(): what later pieces can still take, up to rounding: a remainder
  // smaller than one piece's rounding cannot be handed out. Example: 1,000 bytes left hold a piece of 768 bytes but
  // not one of 1,000, which takes 1,024.
  [[nodiscard]] std::size_t unused_bytes() const {
    return reserved_bytes() - handed_out_bytes();
  }

  // The pieces built from this block and not yet destroyed or emptied by a move.
  [[nodiscard]] std::size_t live_piece_count() const {
    return live_piece_count_ == nullptr ? 0 : *live_piece_count_;
  }

 private:
  friend std::expected<RunMemory, BufferError>
  BuildRunMemory(RunOn run_on, std::size_t byte_count, CpuMemory cpu_memory);
  template <BufferValue Value>
  friend std::expected<Buffer<Value>, BufferError> BuildBuffer(RunOn run_on, std::size_t size, RunMemory& run_memory);

  // Takes ownership of `block` (reserved_bytes bytes, built by BuildBufferBytes) and of `live_piece_count`.
  RunMemory(BufferBytes block, std::size_t reserved_bytes, RunOn run_on, std::unique_ptr<std::size_t> live_piece_count);

  // Hands out the next ComputeRunMemoryPieceBytes(byte_count) bytes of the block and counts one more live piece.
  // byte_count is above 0. Fails with kRunMemoryTooSmall, taking nothing, when they do not fit.
  std::expected<void*, BufferError> HandOutPiece(std::size_t byte_count);

  // Frees the block. With pieces still alive, asserts in a debug build, and in every build leaves the live-piece count
  // allocation behind, so that the pieces destroyed later count down into memory that still exists.
  void FreeBlock();

  BufferBytes block_{};
  std::size_t reserved_bytes_ = 0;
  std::size_t handed_out_bytes_ = 0;  // Bytes, a multiple of 256: where the next piece starts.
  RunOn run_on_ = RunOn::kCpu;
  // Buffers built from the block and not yet destroyed or emptied by a move. On the heap, so moving the RunMemory
  // never moves it: every piece points at it. Null only in a moved-from RunMemory.
  std::unique_ptr<std::size_t> live_piece_count_;
};

// Builds a buffer of `size` values on `run_on` as the next piece of `run_memory`, every byte zero.
//
// Method: takes the next ComputeRunMemoryPieceBytes(size * sizeof(Value)) bytes of the block. The buffer owns no
//   memory: destroying it frees nothing and its bytes stay handed out. Size 0 takes nothing, is not a piece, and
//   data() is null. No call reaches the processor, so this never returns kOutOfMemory or kRunOnCallFailed.
// Example: in a fresh block of 4,096 bytes, BuildBuffer<double>(RunOn::kCpu, 10, run_memory) gives size() == 10, 10
//   zeros at the start of the block, and handed_out_bytes() == 256; then 100 doubles start 256 bytes further on.
// Requires: run_on == run_memory.run_on(), checked by assert in debug builds.
// Requires: run_memory is not moved-from, checked by assert in debug builds.
// Fails: kRunMemoryTooSmall  the rounded bytes are above run_memory.unused_bytes(), or size * sizeof(Value) does not
//                            fit in std::size_t. The run reserves exactly what its estimate gives, so this usually
//                            means the estimate missed an array; raised limits can also cause it while the run goes
//                            on. Nothing is taken, the block is unchanged and no buffer is returned.
// Read: docs/method-paper/fluidity-method.typ <sec-run-memory>
template <BufferValue Value>
std::expected<Buffer<Value>, BufferError> BuildBuffer(RunOn run_on, std::size_t size, RunMemory& run_memory) {
  // Requires: one processor. A block on one processor asked for a buffer on the other is a bug in the caller.
  assert(run_on == run_memory.run_on());
  // Requires: a RunMemory that still owns a block. A moved-from one has no count for its pieces to point at.
  const bool run_memory_is_not_moved_from = run_memory.live_piece_count_ != nullptr;
  assert(run_memory_is_not_moved_from);
  static_cast<void>(run_memory_is_not_moved_from);
  if (size == 0) {
    return Buffer<Value>(nullptr, 0, run_on, CpuMemory::kOrdinary);
  }
  // A product that would wrap around is replaced by the largest byte count, which no block can hold, so the piece is
  // refused as too large instead of taking a small wrapped size.
  constexpr std::size_t kLargestSize = std::numeric_limits<std::size_t>::max();
  const bool byte_count_fits = size <= kLargestSize / sizeof(Value);
  const std::size_t byte_count = byte_count_fits ? size * sizeof(Value) : kLargestSize;
  const std::expected<void*, BufferError> piece = run_memory.HandOutPiece(byte_count);
  if (!piece.has_value()) {
    return std::unexpected(piece.error());
  }
  return Buffer<Value>(
      static_cast<Value*>(*piece),
      size,
      run_on,
      run_memory.block_.cpu_memory_used,
      run_memory.live_piece_count_.get()
  );
}

}  // namespace fluidity
