// Buffers of one value per place of every cell block, bound to one grid version.
//
// Purpose: give every module one type for a per-cell, per-face, per-corner or per-block array, so a value is found
//   by the same place index the grid's tables use, and a function can tell which grid version the values were laid
//   out for.
//
// Idea: a grid has 64 cell places, 240 face places and 125 corner places in each cell block (grid/grid_ids.h). Values
//   are built once with room for `max_cell_blocks` blocks, and never grow. Each new grid version is bound to them by
//   ApplyGridVersion, which only writes down the version and the block count: the values keep their size, and their
//   contents become unspecified until the module that owns them fills them again. A bind runs on the CPU, copies
//   nothing from the GPU and allocates nothing. A function that takes values and a grid compares grid_version() and
//   block_count() with the grid's and fails with kWrongGridVersion when they differ. A grid piece has the whole grid's
//   version, so values loaded for one piece are bound again at every load.
//   What a bind checks: only that the grid's block count fits the room. The values remember the grid's version number
//   and block count, not the grid object, so two grid objects with the same version number cannot be told apart:
//   values bound to one pass CheckGridVersion against the other. The caller must therefore know which grid its values
//   belong to, and must not keep two grids at once with the same version number when it relies on that check.
//   Example: values with room for 4 blocks, bound to a grid of 3 blocks: block_count() is 3, and a CellValues holds
//   3 x 64 = 192 meaningful values. Bound to a grid of 2 blocks next, block_count() is 2 and the 192 values
//   are unspecified.
//
// Words:
//   place  the position of a value inside a cell block, as the grid's tables count them
//   room   the number of blocks the values were built for (`max_cell_blocks`)
//   bound  the values hold the version and the block count of one grid
// Owns:    GridValues, CellValues, FaceValues, CornerValues, CellBlockValues, BuildCellValues, BuildFaceValues,
//          BuildCornerValues, BuildCellBlockValues, ApplyGridVersion, ReadValues, WriteValues, CheckGridVersion,
//          GridValuesAccess (private to this header)
// Uses:    CPU-only guard (base/cpu_only.h), Buffer, BufferValue, BuildBuffer, ReadBuffer, WriteBuffer
//          (base/buffer.h), CheckCanRunOn (base/check_can_run_on.h), RunMemory (base/run_memory.h), RunOn
//          (base/run_on.h), GridFailure (grid/grid_error.h), GridVersion and the places per block
//          (grid/grid_ids.h), ToGridFailure (grid/grid_buffers_internal.h), StaggeredGrid and kMaxCellBlocksLimit
//          (grid/staggered_grid.h)
// Used by: every module that keeps a value for each cell, face, corner or block of a grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-run-memory>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard
// before any library header.
#include "fluidity/base/cpu_only.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <type_traits>
#include <utility>

#include "fluidity/base/buffer.h"
#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {

class GridValuesAccess;

// One value per place of every block: `kPlacesPerBlock` is 64 (cells), 240 (faces), 125 (corners) or 1 (the block
// itself). Built once with room for max_cell_blocks() blocks, then bound to each grid version. Places
// [0, block_count() * kPlacesPerBlock) are meaningful; they are indexed by the place index of the bound grid
// (CellPlaceIndex, FacePlaceIndex, CornerPlaceIndex or the block index). Move-only. A moved-from object has room 0, is
// bound to no grid, and keeps its run_on().
template <BufferValue Value, int kPlacesPerBlock>
class GridValues {
  static_assert(kPlacesPerBlock >= 1, "a block holds at least one place");

 public:
  GridValues(const GridValues&) = delete;
  GridValues& operator=(const GridValues&) = delete;
  GridValues(GridValues&& other) noexcept
      : buffer_(std::move(other.buffer_)),
        grid_version_(std::exchange(other.grid_version_, GridVersion{.value = 0})),
        block_count_(std::exchange(other.block_count_, 0)) {
  }
  GridValues& operator=(GridValues&& other) noexcept {
    if (this != &other) {
      buffer_ = std::move(other.buffer_);
      grid_version_ = std::exchange(other.grid_version_, GridVersion{.value = 0});
      block_count_ = std::exchange(other.block_count_, 0);
    }
    return *this;
  }
  ~GridValues() = default;

  // The version of the grid the values are bound to. GridVersion{0}: bound to none.
  [[nodiscard]] GridVersion grid_version() const {
    return grid_version_;
  }

  // The bound extent: places [0, block_count() * kPlacesPerBlock) are meaningful. 0 when bound to none.
  [[nodiscard]] std::uint32_t block_count() const {
    return block_count_;
  }

  // The room, in blocks, fixed when the values were built.
  [[nodiscard]] std::uint32_t max_cell_blocks() const {
    return static_cast<std::uint32_t>(buffer_.size() / kPlacesPerBlock);
  }

  [[nodiscard]] RunOn run_on() const {
    return buffer_.run_on();
  }

  // The first value: valid on run_on() only.
  [[nodiscard]] Value* data() {
    return buffer_.data();
  }
  [[nodiscard]] const Value* data() const {
    return buffer_.data();
  }

  // The buffer (base/buffer.h), of size max_cell_blocks() * kPlacesPerBlock. The ordered sums of base/ take it with
  // count block_count() * kPlacesPerBlock.
  [[nodiscard]] Buffer<Value>& buffer() {
    return buffer_;
  }
  [[nodiscard]] const Buffer<Value>& buffer() const {
    return buffer_;
  }

 private:
  friend class GridValuesAccess;

  explicit GridValues(Buffer<Value> buffer) : buffer_(std::move(buffer)) {
  }

  Buffer<Value> buffer_;
  GridVersion grid_version_ = {.value = 0};
  std::uint32_t block_count_ = 0;
};

// One value per cell place, meaningful where the cell is active (CellState::kActive).
template <BufferValue Value>
using CellValues = GridValues<Value, kCellPlacesPerBlock>;
// One value per face place (FacePlaceIndex).
template <BufferValue Value>
using FaceValues = GridValues<Value, kFacePlacesPerBlock>;
// One value per corner place (CornerPlaceIndex).
template <BufferValue Value>
using CornerValues = GridValues<Value, kCornerPlacesPerBlock>;
// One value per cell block (CellBlockIndex).
template <BufferValue Value>
using CellBlockValues = GridValues<Value, 1>;

// The one way a value buffer is made or bound. Private to this header: the builders and ApplyGridVersion below call
// it, and nothing else does.
class GridValuesAccess {
 public:
  // Values with room for `max_cell_blocks` blocks on `run_on`, bound to no grid, as one piece of `run_memory`.
  // Fails: kRunOnCallFailed CheckCanRunOn(run_on) failed; nothing is taken from run_memory
  //        kBadInput        max_cell_blocks is 0 or above kMaxCellBlocksLimit
  //        kOutOfMemory     the piece does not fit in what run_memory has left; the magnitude is the bytes asked
  //                         for, max_cell_blocks * kPlacesPerBlock * sizeof(Value)
  template <BufferValue Value, int kPlacesPerBlock>
  static std::expected<GridValues<Value, kPlacesPerBlock>, GridFailure>
  BuildGridValues(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory) {
    // Step 1: a processor that cannot run is reported before the block is touched.
    if (!CheckCanRunOn(run_on).has_value()) {
      return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
    }
    // Requires: one processor for the values and the block.
    assert(run_on == run_memory.run_on());

    // Step 2: a room of 0 is a size never filled in; one above the limit has place indexes that do not fit 32 bits.
    if (max_cell_blocks == 0 || max_cell_blocks > kMaxCellBlocksLimit) {
      return std::unexpected(GridFailure{.error = GridError::kBadInput});
    }

    // Step 3: one piece of the block. The limit keeps the product below 2^32 x 240, so it cannot wrap.
    const std::uint64_t place_count = std::uint64_t{max_cell_blocks} * kPlacesPerBlock;
    const std::uint64_t asked_bytes = place_count * sizeof(Value);
    std::expected<Buffer<Value>, BufferError> built =
        BuildBuffer<Value>(run_on, static_cast<std::size_t>(place_count), run_memory);
    if (!built.has_value()) {
      return std::unexpected(ToGridFailure(built.error(), asked_bytes));
    }
    return GridValues<Value, kPlacesPerBlock>(std::move(*built));
  }

  // Writes down the grid's version and block count. Reads only grid.version() and grid.counts().cell_blocks.
  template <BufferValue Value, int kPlacesPerBlock>
  static std::expected<void, GridFailure>
  BindGridValues(GridValues<Value, kPlacesPerBlock>& values, const StaggeredGrid& grid) {
    // Requires: one processor. Nothing is copied, so a mismatch is a bug in the caller, not an error value.
    assert(values.run_on() == grid.run_on());
    // A grid with more blocks than the room leaves the values as they were.
    const std::uint32_t grid_block_count = grid.counts().cell_blocks;
    if (grid_block_count > values.max_cell_blocks()) {
      return std::unexpected(GridFailure{.error = GridError::kTooManyCellBlocks});
    }
    values.grid_version_ = grid.version();
    values.block_count_ = grid_block_count;
    return {};
  }

  // Values built for exactly the grid's blocks, and at least one, bound to the grid.
  template <BufferValue Value, int kPlacesPerBlock>
  static std::expected<GridValues<Value, kPlacesPerBlock>, GridFailure>
  BuildBoundGridValues(const StaggeredGrid& grid, RunMemory& run_memory) {
    // An empty grid has no block, and values of room 0 do not exist: it gets room for 1.
    const std::uint32_t room = std::max<std::uint32_t>(grid.counts().cell_blocks, 1);
    std::expected<GridValues<Value, kPlacesPerBlock>, GridFailure> values =
        BuildGridValues<Value, kPlacesPerBlock>(grid.run_on(), room, run_memory);
    if (!values.has_value()) {
      return values;
    }
    if (const std::expected<void, GridFailure> bound = BindGridValues(*values, grid); !bound.has_value()) {
      return std::unexpected(bound.error());
    }
    return values;
  }
};

// Builds values with room for `max_cell_blocks` blocks on `run_on`, bound to no grid, as one piece of `run_memory`.
// The only allocation of a value buffer.
//
// Method: CheckCanRunOn(run_on) first, then the room, then one memory piece of run_memory of
//   ComputeRunMemoryPieceBytes(max_cell_blocks x places per block x sizeof(Value)) bytes. Every byte of the piece
//   is zero.
//   Every Build...Values below is this one function for its places per block: 64, 240, 125 and 1.
// Example: BuildFaceValues<double>(RunOn::kCpu, 1000, run_memory) takes 1,000 x 240 x 8 = 1,920,000 bytes of a fresh
//   run_memory, and its max_cell_blocks() is 1000, its block_count() 0 and its grid_version().value 0.
// Requires: run_on == run_memory.run_on() when run_on can run, checked by assert in debug builds.
// Fails: kRunOnCallFailed CheckCanRunOn(run_on) failed (a GPU asked for in a build or on a machine without one);
//                         nothing is taken from run_memory
//        kBadInput        max_cell_blocks is 0 or above kMaxCellBlocksLimit
//        kOutOfMemory     the piece does not fit in what run_memory has left; the failure's magnitude is the bytes
//                         asked for, before rounding. Nothing is taken. A run plans its memory from its estimate, so
//                         this is a planning error.
//        On failure no values are returned.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-run-memory>
template <BufferValue Value>
std::expected<CellValues<Value>, GridFailure>
BuildCellValues(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory) {
  return GridValuesAccess::BuildGridValues<Value, kCellPlacesPerBlock>(run_on, max_cell_blocks, run_memory);
}

// As BuildCellValues, with 240 places per block.
template <BufferValue Value>
std::expected<FaceValues<Value>, GridFailure>
BuildFaceValues(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory) {
  return GridValuesAccess::BuildGridValues<Value, kFacePlacesPerBlock>(run_on, max_cell_blocks, run_memory);
}

// As BuildCellValues, with 125 places per block.
template <BufferValue Value>
std::expected<CornerValues<Value>, GridFailure>
BuildCornerValues(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory) {
  return GridValuesAccess::BuildGridValues<Value, kCornerPlacesPerBlock>(run_on, max_cell_blocks, run_memory);
}

// As BuildCellValues, with 1 place per block.
template <BufferValue Value>
std::expected<CellBlockValues<Value>, GridFailure>
BuildCellBlockValues(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory) {
  return GridValuesAccess::BuildGridValues<Value, 1>(run_on, max_cell_blocks, run_memory);
}

// Binds `values` to `grid` (a whole grid or a grid piece's grid): grid_version() and block_count() become the grid's.
//
// What it checks: only that the grid's block count fits the room. It writes down the grid's version number, not the
//   grid: a later check of the values against a grid compares version numbers, so two grid objects with the same
//   version number are not told apart, and values bound to one pass for the other. The caller must make sure the grid
//   it binds is the grid it later uses, and that no two grids it keeps at once share a version number. A grid at
//   version 0 (moved from, or its buffers taken back) binds, and leaves the values bound to no grid.
// Method: reads only grid.version() and grid.counts().cell_blocks, which every grid keeps on the CPU, and writes them
//   down. It allocates, launches and copies nothing, on either processor. The contents of the values become
//   unspecified: no value is touched, and the owning module fills them again.
// Example: values with room for 8 blocks and a grid of 3 blocks at version 5: block_count() becomes 3 and
//   grid_version().value becomes 5. A grid of 9 blocks fails with kTooManyCellBlocks and changes nothing.
// Requires: values.run_on() == grid.run_on(), checked by assert in debug builds.
// Fails: kTooManyCellBlocks  the grid has more blocks than values.max_cell_blocks(); `values` is unchanged
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
template <BufferValue Value, int kPlacesPerBlock>
std::expected<void, GridFailure>
ApplyGridVersion(GridValues<Value, kPlacesPerBlock>& values, const StaggeredGrid& grid) {
  return GridValuesAccess::BindGridValues(values, grid);
}

// Builds values for exactly grid.counts().cell_blocks blocks on grid.run_on() and binds them to `grid`. Allocates; for
// tests, tools and values that are built once.
//
// Method: the room form of BuildCellValues with the grid's processor and block count, then ApplyGridVersion. A grid of
//   no block gets room for 1 block, since values of room 0 do not exist.
// Example: a grid of 9 blocks at version 3 gives max_cell_blocks() == 9, block_count() == 9, grid_version().value == 3,
//   and takes ComputeRunMemoryPieceBytes(9 x 64 x sizeof(Value)) bytes of run_memory.
// Requires: grid.run_on() == run_memory.run_on(), checked by assert in debug builds.
// Fails: as the room form, which says each cause: kRunOnCallFailed, kBadInput and kOutOfMemory
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-run-memory>
template <BufferValue Value>
std::expected<CellValues<Value>, GridFailure> BuildCellValues(const StaggeredGrid& grid, RunMemory& run_memory) {
  return GridValuesAccess::BuildBoundGridValues<Value, kCellPlacesPerBlock>(grid, run_memory);
}

// As the grid form of BuildCellValues, with 240 places per block.
template <BufferValue Value>
std::expected<FaceValues<Value>, GridFailure> BuildFaceValues(const StaggeredGrid& grid, RunMemory& run_memory) {
  return GridValuesAccess::BuildBoundGridValues<Value, kFacePlacesPerBlock>(grid, run_memory);
}

// As the grid form of BuildCellValues, with 125 places per block.
template <BufferValue Value>
std::expected<CornerValues<Value>, GridFailure> BuildCornerValues(const StaggeredGrid& grid, RunMemory& run_memory) {
  return GridValuesAccess::BuildBoundGridValues<Value, kCornerPlacesPerBlock>(grid, run_memory);
}

// As the grid form of BuildCellValues, with 1 place per block.
template <BufferValue Value>
std::expected<CellBlockValues<Value>, GridFailure>
BuildCellBlockValues(const StaggeredGrid& grid, RunMemory& run_memory) {
  return GridValuesAccess::BuildBoundGridValues<Value, 1>(grid, run_memory);
}

// Copies the bound part of `values` into CPU memory: a read from the GPU for GPU values. The only way values
// leave the GPU.
//
// Method: checks the bind, then the span's size, then copies the first block_count() x places per block entries with
//   the slice form of ReadBuffer. A moved-from or never-bound object is bound to no grid.
// Example: CellValues bound to a grid of 2 blocks need a span of 128 entries; a span of 127 is kBadInput.
// Requires: `cpu_values` is CPU memory.
// Fails: kWrongGridVersion  the values are bound to no grid (grid_version().value is 0)
//        kBadInput          cpu_values.size() differs from block_count() x places per block, or `cpu_values`
//                           overlaps the values (the caller's mistake)
//        kRunOnCallFailed   the copy from the GPU failed
//        A failed check writes nothing: the span and the values are unchanged. After kRunOnCallFailed the span
//        is unspecified.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
template <BufferValue Value, int kPlacesPerBlock>
std::expected<void, GridFailure>
ReadValues(const GridValues<Value, kPlacesPerBlock>& values, std::type_identity_t<std::span<Value>> cpu_values) {
  if (values.grid_version().value == 0) {
    return std::unexpected(GridFailure{.error = GridError::kWrongGridVersion});
  }
  const std::uint64_t bound_place_count = std::uint64_t{values.block_count()} * kPlacesPerBlock;
  if (cpu_values.size() != bound_place_count) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  const std::expected<void, BufferError> read = ReadBuffer(values.buffer(), 0, cpu_values);
  if (!read.has_value()) {
    // The size was checked above, so the buffer's size error can only be an overlap: the caller's span.
    if (read.error() == BufferError::kSizeMismatch) {
      return std::unexpected(GridFailure{.error = GridError::kBadInput});
    }
    return std::unexpected(ToGridFailure(read.error(), 0));
  }
  return {};
}

// Copies CPU memory into the bound part of `*values`: a write to the GPU for GPU values. The only way values
// enter the GPU.
//
// Method: as ReadValues, with the slice form of WriteBuffer. Places past the bound part are not touched.
// Example: CellValues bound to a grid of 2 blocks take 128 entries.
// Requires: `cpu_values` is CPU memory; `values` is not null.
// Fails: as ReadValues; after kRunOnCallFailed the bound part is unspecified.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
template <BufferValue Value, int kPlacesPerBlock>
std::expected<void, GridFailure>
WriteValues(std::type_identity_t<std::span<const Value>> cpu_values, GridValues<Value, kPlacesPerBlock>* values) {
  assert(values != nullptr);
  if (values->grid_version().value == 0) {
    return std::unexpected(GridFailure{.error = GridError::kWrongGridVersion});
  }
  const std::uint64_t bound_place_count = std::uint64_t{values->block_count()} * kPlacesPerBlock;
  if (cpu_values.size() != bound_place_count) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  const std::expected<void, BufferError> written = WriteBuffer(cpu_values, values->buffer(), 0);
  if (!written.has_value()) {
    // The size was checked above, so the buffer's size error can only be an overlap: the caller's span.
    if (written.error() == BufferError::kSizeMismatch) {
      return std::unexpected(GridFailure{.error = GridError::kBadInput});
    }
    return std::unexpected(ToGridFailure(written.error(), 0));
  }
  return {};
}

// Checks that `version` is the version of `grid`. Changes nothing.
//
// Example: a grid at version 3 and values bound to version 2: kWrongGridVersion.
// Fails: kWrongGridVersion  version.value differs from grid.version().value
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
std::expected<void, GridFailure> CheckGridVersion(const StaggeredGrid& grid, GridVersion version);

}  // namespace fluidity::grid
