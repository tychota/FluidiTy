#include <cassert>
#include <cstddef>
#include <expected>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include "fluidity/base/buffer.h"
#include "fluidity/base/buffer_bytes.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"

namespace fluidity {

RunMemory::RunMemory(
    BufferBytes block,
    std::size_t reserved_bytes,
    RunOn run_on,
    std::unique_ptr<std::size_t> live_piece_count
)
    : block_(block), reserved_bytes_(reserved_bytes), run_on_(run_on), live_piece_count_(std::move(live_piece_count)) {
}

RunMemory::RunMemory(RunMemory&& other) noexcept
    : block_(std::exchange(other.block_, BufferBytes{})),
      reserved_bytes_(std::exchange(other.reserved_bytes_, 0)),
      handed_out_bytes_(std::exchange(other.handed_out_bytes_, 0)),
      run_on_(other.run_on_),
      live_piece_count_(std::move(other.live_piece_count_)) {
}

RunMemory& RunMemory::operator=(RunMemory&& other) noexcept {
  if (this != &other) {
    FreeBlock();
    block_ = std::exchange(other.block_, BufferBytes{});
    reserved_bytes_ = std::exchange(other.reserved_bytes_, 0);
    handed_out_bytes_ = std::exchange(other.handed_out_bytes_, 0);
    run_on_ = other.run_on_;
    live_piece_count_ = std::move(other.live_piece_count_);
  }
  return *this;
}

RunMemory::~RunMemory() {
  FreeBlock();
}

void RunMemory::FreeBlock() {
  // Requires: every piece was destroyed or emptied by a move first.
  const bool no_piece_is_alive = live_piece_count() == 0;
  assert(no_piece_is_alive);
  if (!no_piece_is_alive) {
    // A release build goes on. The late pieces will still count themselves down when destroyed, so their count must
    // outlive this RunMemory: it is left behind on purpose, 8 bytes never freed, instead of a write into freed memory.
    const std::size_t* const count_left_behind = live_piece_count_.release();
    static_cast<void>(count_left_behind);
  }
  FreeBufferBytes(run_on_, block_.cpu_memory_used, block_.bytes);
  block_ = BufferBytes{};
}

std::expected<void*, BufferError> RunMemory::HandOutPiece(std::size_t byte_count) {
  const std::size_t piece_bytes = ComputeRunMemoryPieceBytes(byte_count);
  // A moved-from RunMemory reserves 0 bytes, so no piece fits there and its missing count is never touched.
  if (piece_bytes > unused_bytes()) {
    // Not a precondition: what fits depends on the plan made elsewhere and on limits raised while the run goes on.
    // The module turns it into its own out-of-memory failure.
    return std::unexpected(BufferError::kRunMemoryTooSmall);
  }
  // handed_out_bytes_ is a sum of pieces, each a multiple of 256, so the piece starts at a multiple of 256 from the
  // start of the block.
  void* const piece = static_cast<std::byte*>(block_.bytes) + handed_out_bytes_;
  handed_out_bytes_ += piece_bytes;
  ++*live_piece_count_;
  return piece;
}

std::expected<RunMemory, BufferError> BuildRunMemory(RunOn run_on, std::size_t byte_count, CpuMemory cpu_memory) {
  // No object, on either processor, can be larger than the largest std::ptrdiff_t (as in BuildBuffer).
  constexpr std::size_t kLargestByteCount = std::numeric_limits<std::ptrdiff_t>::max();
  if (byte_count > kLargestByteCount) {
    return std::unexpected(BufferError::kOutOfMemory);
  }
  // The live-piece count lives on the heap so that moving the RunMemory never moves it: every piece keeps a pointer to
  // it.
  std::unique_ptr<std::size_t> live_piece_count(new (std::nothrow) std::size_t{0});
  if (live_piece_count == nullptr) {
    return std::unexpected(BufferError::kOutOfMemory);
  }
  // One allocation and one fill with zero bytes for the whole block: the only call to the processor.
  const std::expected<BufferBytes, BufferError> block = BuildBufferBytes(run_on, byte_count, cpu_memory);
  if (!block.has_value()) {
    return std::unexpected(block.error());
  }
  return RunMemory(*block, byte_count, run_on, std::move(live_piece_count));
}

}  // namespace fluidity
