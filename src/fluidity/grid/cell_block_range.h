// Ranges of cell blocks and the border blocks a range needs around it, on either processor.
//
// Purpose: cut a grid into ranges of consecutive blocks, and find, for any set of blocks, the blocks outside it that a
//   stage writing values kept in the set may read, so a run can work on one range at a time.
//
// Idea: a stage works on a range, a run of consecutive blocks in the fixed order. It writes only values kept in the
//   range's blocks, so every value has one writer. It may read values kept in the range and in its border blocks:
//   the blocks outside the range that its readers read, on the way or as a result, from a place kept in the range
//   (border_block_steps.h). A grid piece of the range and its border answers every such reader as the whole grid
//   does, which defines the border. A pass that runs several stages on one load stacks borders: the border of its last
//   stage is found around the range, then the border of the stage before around the range plus that border, and so
//   on. The buffer form of FindBorderBlocks does one layer per call in memory that was built once, so a run in batches
//   finds the layers of every range without allocating: the marks of the forward form are one byte per block. On the
//   GPU it runs one thread for each place of the set, then one thread for each block, and only the count comes back.
//   Example: a grid of 10 blocks cut with at most 4 blocks per range gives the ranges {0, 4}, {4, 4} and {8, 2}. In a
//   run in one piece the range is every block and its border is empty.
//
// Words:
//   range         a run of consecutive cell blocks, in the fixed order, that one stage writes
//   border block  a block outside a range that the stage may read
//   border width  how far a stage's readers read (BorderWidth)
//   border marks  one byte per block of the grid: 1 for a block that a reader read, or a bigger block of one
//   layer         the blocks one stage of a stacked border adds around the blocks of the stages after it
// Owns:    RangeWithBorder, BuildRangeWithBorder, BuildRanges, FindBorderBlocks (vector form and buffer form),
//          BorderSearchBuffers, BuildBorderSearchBuffers, BuildBorderSearchStackEstimate
// Uses:    CPU-only guard (base/cpu_only.h), BorderWidth (grid/border_block_steps.h), StaggeredGrid
//          (grid/staggered_grid.h), GridPiece (grid/grid_piece.h, included for the callers of this header),
//          GridFailure (grid/grid_error.h), CellBlockIndex, CellBlockRange (grid/grid_ids.h),
//          Buffer (base/buffer.h), RunMemory (base/run_memory.h), OrderBuffers (base/stable_order.h), RunOn
//          (base/run_on.h), GridBuildStatus (grid/grid_build_status.h), MemoryEstimateEntry (base/memory_count.h)
// Implementation uses: BlockSet, ComputeIsInBlockSet, ApplyCellPlaceReadMarks, ApplyFacePlaceReadMarks,
//          ComputeBorderSoFarFailureKey, ApplySetPlaceReadMarks, ComputeListedMark (grid/border_block_steps.h),
//          CheckCanRunOn (base/check_can_run_on.h), BuildBorderSearchMemoryEstimate (grid/grid_memory_estimate.h), the
//          GPU half (grid/cell_block_range_gpu.h)
// Used by: the stages that run on one range of blocks, the grid pieces of a run in batches, and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>, <sec-face-search>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {

// One range and the blocks around it that a stage on the range may read.
struct RangeWithBorder {
  CellBlockRange range;  // A stage writes only values kept in these blocks.
  // Ascending, each once, outside the range: FindBorderBlocks of the range's blocks at kTouchingCells.
  std::vector<CellBlockIndex> border_blocks;
};

// Cuts the grid into consecutive ranges of at most max_blocks_per_range blocks, in order: every block is in exactly
// one range, and every range but the last has max_blocks_per_range blocks. A grid of no block gives no range.
// Method: reads counts() only, so it works on either processor.
// Example: 10 blocks and max_blocks_per_range 4 give {0, 4}, {4, 4} and {8, 2}.
// Fails: kBadInput  max_blocks_per_range is 0: no range can hold a block, so no cut covers the grid. The result then
//                   holds the failure and no list.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
std::expected<std::vector<CellBlockRange>, GridFailure>
BuildRanges(const StaggeredGrid& grid, std::uint32_t max_blocks_per_range);

// Finds the border of a range at kTouchingCells: the blocks outside it that a stage writing values kept in it may
// read, when the stage reads, from each place it writes, the places next to it (BorderWidth::kTouchingCells). This is
// the width of the transfer step, which adds into each face piece of the range from the particles of the cells
// that can list it.
// Method: FindBorderBlocks of the range's blocks, with the width kTouchingCells, by the forward form.
// Work: that of the vector form of FindBorderBlocks, for the places of the range's blocks.
// Allocates the marks (one byte per block of the grid, freed on return) and the returned list, sized once: two
// allocations, or one when the border is empty.
// Example: the grid of the file header of border_block_steps.h and the range {0, 1} give the border blocks 1 and 2.
// Requires: grid.run_on() == RunOn::kCpu, and range inside the grid's blocks (debug assertions). A range outside the
//   grid is a precondition here because this function has no failure value, but a status (kBadInput) in BuildRanges,
//   BuildGridPiece and the buffer form of FindBorderBlocks.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>
RangeWithBorder BuildRangeWithBorder(const StaggeredGrid& grid, CellBlockRange range);

// Finds the blocks outside `blocks` that a stage writing values kept in `blocks` may read, at `width`, in ascending
// index, each once. `blocks` need not be a run, so borders stack: the border of the stage before is
// FindBorderBlocks of `blocks` plus the border found so far, at its own width.
//
// Method: the readers run forward from every place kept in the set. ApplyCellPlaceReadMarks and
//   ApplyFacePlaceReadMarks (border_block_steps.h) mark every block that a reader of the width reads from one place,
//   an active cell, a face piece or a corner place, on the way or as a result, with its bigger blocks; then one pass
//   over the marks, in ascending block index, lists the marked blocks outside the set. The readers of each width are
//   listed at BorderWidth; the border also holds every bigger block, up to size level 0, of each block of the set and
//   of each block read, which the readers walk through. The proofs are at ComputeIsBorderBlock, which says the same
//   from a block's side and gives the same set. A stage calls only those readers, and only from places kept in the
//   set; a position for FindNearbyFaces must be one that FindCellAt gives to a cell of the set.
// Work: for each of the 64 cell places and 240 face places of each block of the set, one marking function, which
//   costs up to about 10^5 table reads (cell place) or 4 x 10^4 (face place) whatever the grid's size, and then one
//   read of each of the grid's block marks. So it grows with the set, and with the grid only by that one pass over
//   one byte per block, not with the grid times its size levels.
// Allocates the marks (one byte per block of the grid, freed on return) and the returned list, sized once: two
// allocations, or one when the border is empty. The form that takes BorderSearchBuffers below allocates nothing.
// Order: ascending block index, so the result is the same in every call on the same grid.
// Example: one size level, cells 1 m wide, cells {3,3,0}, {4,3,0} and {6,6,0} in the blocks 0, 1 and 2: the border of
//   {0} at kTouchingCells is {1, 2}. Block 2 holds the +Y face piece of {4,3,0}, kept at plane 0 of block 2. The
//   border of {2} is {0, 1}: the cell_low of that face piece is {4,3,0}, and {3,3,0} touches it, so its particles can
//   list the face piece. With 2 size levels and the size-level-0 cells {0..3}^3 all active but {0..2}^3, which are
//   split, the border of the size-level-1 block (0,0,0), block 1, holds block 0, through which FindCellAt walks.
// Requires: grid.run_on() == RunOn::kCpu (debug assertion).
// Fails: kBadInput  `blocks` is not ascending, repeats a block, or holds an index at or above the grid's block count.
//        The result then holds the failure and no list.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>, <sec-face-search>
std::expected<std::vector<CellBlockIndex>, GridFailure>
FindBorderBlocks(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width);

// The room of the buffer form of FindBorderBlocks: the border marks of the forward form, one byte per block, and, for a
// grid on the GPU, the arrays that list the marked blocks. Built once for a limit on the number of blocks, then used
// for any number of calls on grids of at most that many blocks, one call at a time. Move-only; a moved-from object has
// max_cell_blocks() 0.
// Its arrays are pieces of the RunMemory it was built from, which must outlive it (base/run_memory.h).
class BorderSearchBuffers {
 public:
  // An empty object: max_cell_blocks() is 0, so no call on a grid with blocks takes it (a grid of 0 blocks with the
  // range {0, 0} does). BuildBorderSearchBuffers makes the real ones.
  BorderSearchBuffers() = default;
  BorderSearchBuffers(const BorderSearchBuffers&) = delete;
  BorderSearchBuffers& operator=(const BorderSearchBuffers&) = delete;
  BorderSearchBuffers(BorderSearchBuffers&& other) noexcept
      : marks_(std::move(other.marks_)),
        sums_(std::move(other.sums_)),
        order_buffers_(std::move(other.order_buffers_)),
        status_(std::move(other.status_)),
        max_cell_blocks_(std::exchange(other.max_cell_blocks_, 0)),
        run_on_(other.run_on_) {
  }
  BorderSearchBuffers& operator=(BorderSearchBuffers&& other) noexcept {
    if (this != &other) {
      marks_ = std::move(other.marks_);
      sums_ = std::move(other.sums_);
      order_buffers_ = std::move(other.order_buffers_);
      status_ = std::move(other.status_);
      max_cell_blocks_ = std::exchange(other.max_cell_blocks_, 0);
      run_on_ = other.run_on_;
    }
    return *this;
  }
  ~BorderSearchBuffers() = default;

  // The most blocks of a grid that a call can search. 0 for an empty or moved-from object.
  [[nodiscard]] std::uint32_t max_cell_blocks() const {
    return max_cell_blocks_;
  }

  // The processor the arrays live on.
  [[nodiscard]] RunOn run_on() const {
    return run_on_;
  }

  // The border marks: max_cell_blocks() entries of one byte. Their contents mean nothing between calls: each call sets
  // the marks of its grid's blocks to 0 first.
  [[nodiscard]] const Buffer<std::uint8_t>& marks() const {
    return marks_;
  }

 private:
  friend std::expected<BorderSearchBuffers, GridFailure>
  BuildBorderSearchBuffers(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory);
  friend std::expected<std::uint32_t, GridFailure> FindBorderBlocks(
      const StaggeredGrid& grid,
      CellBlockRange range,
      const Buffer<CellBlockIndex>& border_so_far,
      std::uint32_t border_so_far_count,
      BorderWidth width,
      BorderSearchBuffers& search_buffers,
      Buffer<CellBlockIndex>& border_blocks
  );

  Buffer<std::uint8_t> marks_;  // One byte per block, on both processors.
  // GPU only, as the estimate lists them (BuildBorderSearchMemoryEstimate): one word per block plus one, the 0/1 words
  // of the listed marks and then their exclusive prefix sums; the working buffers of that sum; one GridBuildStatus.
  Buffer<std::uint32_t> sums_;
  std::optional<OrderBuffers> order_buffers_;
  Buffer<GridBuildStatus> status_;
  std::uint32_t max_cell_blocks_ = 0;
  RunOn run_on_ = RunOn::kCpu;
};

// Builds the room of border search for grids of at most max_cell_blocks blocks, as pieces of `run_memory`: one piece
// for each entry of BuildBorderSearchMemoryEstimate(run_on, max_cell_blocks), in its order. This is the one
// allocation of border search.
// Example: on the CPU, 1,000 blocks take one piece of 1,000 bytes, which a run memory hands out as 1,024.
// Requires: run_on == run_memory.run_on(), checked by assert in debug builds. The check of CheckCanRunOn(run_on)
//   comes before that assert, so asking for a GPU that is not there fails with kRunOnCallFailed whatever the
//   run memory.
// Fails: kRunOnCallFailed  CheckCanRunOn(run_on) failed (a GPU asked for in a build without GPU code, or without a
//                          usable GPU); nothing is taken
//        kBadInput         max_cell_blocks is 0 or above kMaxCellBlocksLimit; nothing is taken
//        kOutOfMemory      a piece does not fit in run_memory; the failure's magnitude is the bytes of the array that
//                          did not fit. On the GPU also: the GPU cannot give the stack of the marking function (the
//                          driver's per-thread stack, raised here), and the magnitude is its bytes
//                          (BuildBorderSearchStackEstimate)
//        kInternalFault    the arrays and the estimate's entries do not match: no input causes it, only a change to
//                          the estimate or to this function that leaves the other behind
//        On failure no buffers are returned. Pieces taken before a failure stay in the block, which frees them
//        with itself.
// Read: docs/method-paper/fluidity-method.typ <sec-grid-memory>
std::expected<BorderSearchBuffers, GridFailure>
BuildBorderSearchBuffers(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory);

// Finds one layer of a stacked border, on the processor of the grid: the blocks outside `range` that are in
// `border_so_far` or in the border, at `width`, of `range` plus `border_so_far`. Writes the first
// border_blocks.size() of them to `border_blocks`, in ascending index, each once, and returns how many there are in
// all. A count above border_blocks.size() means the buffer was too small; nothing beyond it is written. The call
// allocates nothing.
//
// Method: the forward form, in the border marks of `search_buffers`. 1 Set the marks of the grid's blocks to 0, so
//   that they hold only what this call's places read: a mark left from an earlier call would add a block to the list
//   silently. The marking functions read no mark (border_block_steps.h), so threads never read what others write; a
//   GPU form must keep it so, and a marking function writes the value 1 for every block it reads, tested against none.
//   2 Check the entries of `border_so_far`. 3 For each block of `range`, then of `border_so_far`, run
//   ApplyCellPlaceReadMarks for each of its 64 cell places and ApplyFacePlaceReadMarks for each of its 240 face places
//   at `width` (ApplySetPlaceReadMarks, for one place of the set). 4 Walk the marks in index order and list each marked
//   block outside `range`. Every block of the set is marked through its own cell places, so the list holds
//   `border_so_far` without a merge. On the GPU the steps are launches of one stream: 1 one thread resets the status
//   record and one thread for each block clears its mark; 2 one thread for each entry offers its failure key to the
//   record; 3 one thread for each place of the set marks, and stops at once when the record holds a key, so the marks
//   are written only with the value 1, with plain stores, and no thread reads a mark; the places run in launches of at
//   most kSetBlocksPerMarkingLaunch blocks of the set each (2,048), one after the other, which changes no mark; 4 after
//   the last launch has finished, one thread for each block writes its listed mark (0 when the record holds a key), an
//   exclusive prefix sum gives each listed block its place, and one thread for each block writes the ascending list and
//   the total into the record, which is the only thing read back. The prefix sum keeps index order, so the list is the
//   CPU's.
// Work: for each place of `range` plus `border_so_far`, one marking function, as the vector form of FindBorderBlocks,
//   whose Work line gives the cost; and, whatever the set, one write and one read of each of the grid's block marks.
// Example: the grid of the file header of border_block_steps.h, range {0, 1} (block 0) and no border so far give the
//   list {1, 2} and the count 2. A stacked border is one call per layer, with two output buffers used in turn: the
//   list of one call is the border so far of the next. Call it with the width of the last stage first and that of the
//   first stage last; the stage that a call serves reads only from places of `range` and of the lists of the calls
//   before it.
// Requires: every buffer on grid.run_on() (debug assertion). The grid and the range are those of every layer.
// Fails: kBadInput          border_so_far is not ascending, holds a block of `range` or an index at or above the
//                           grid's block count; border_blocks is border_so_far (the same buffer); border_so_far_count
//                           is above border_so_far.size(); `range` is not inside the grid's blocks. Nothing is written
//                           on the CPU; on the GPU the entries are found bad by the pass itself, which writes no block.
//        kTooManyCellBlocks search_buffers.max_cell_blocks() is below the grid's block count. Nothing is written. It is
//                           checked before the entries of border_so_far, so a call that fails both ways gives this
//                           error on either processor.
//        kTooManyCellBlocks also: `range` plus border_so_far_count has more than kMaxSetBlocks blocks
//                           (14,128,181), so a place of the set would not fit the 32-bit place index. Checked on both
//                           processors by ComputeIsSetWithinPlaceLimit, before the entries of border_so_far and before
//                           anything is written.
//        kRunOnCallFailed   GPU only: a GPU call failed. A launch that did not start waits for the GPU's pending work
//                           before the call returns, so no earlier launch of the call is still writing.
//        kInternalFault     GPU only: the prefix sum failed on a size, or the status record read back contradicts the
//                           steps that wrote it: a failure key of a step other than the check of border_so_far, or a
//                           count above the block count (ToBorderBlockCount, grid/cell_block_range_internal.h)
//        On failure border_blocks holds no result; after kRunOnCallFailed the marks and the arrays of
//        `search_buffers` are unspecified, as every buffer of a failed GPU call. If a launch of the marking fails after
//        earlier ones marked, nothing is written to the output (no list, no count), and the next call clears the
//        marks first.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>, <sec-face-search>
std::expected<std::uint32_t, GridFailure> FindBorderBlocks(
    const StaggeredGrid& grid,
    CellBlockRange range,
    const Buffer<CellBlockIndex>& border_so_far,
    std::uint32_t border_so_far_count,
    BorderWidth width,
    BorderSearchBuffers& search_buffers,
    Buffer<CellBlockIndex>& border_blocks
);

// Builds the estimate entry of the stack the GPU's border search needs: "grid: GPU stack of the border search",
// kGivenCount, kWholeRun, MemoryWhere::kDriverStack, the local bytes per thread of its largest GPU function
// (bytes_each) for every thread the GPU can hold at once (given_count: threads per multiprocessor x multiprocessors).
// For RunOn::kCpu the entry counts nothing. A plan that runs the border search on the GPU adds this entry to the
// estimates of the search buffers.
//
// Method: the thread that runs the marking functions keeps their results as locals, more than the driver's default
//   stack of 1,024 B per thread, and the driver gives each resident thread that stack, so the count is the GPU's
//   resident threads, not the places of a set. The bytes per thread come from the compiled functions
//   (cudaFuncGetAttributes). BuildBorderSearchBuffers on the GPU reserves the stack, so a GPU too full for it fails
//   there with kOutOfMemory, not in a later search. The driver keeps one stack, sized for the largest per-thread need
//   of any GPU function launched, so this entry added to another module's driver-stack entry (contact's) counts more
//   than the driver takes: the sum is an upper bound, not the bytes taken.
// Example: 55,296 resident threads of a 36-multiprocessor sm_120 card, times the bytes per thread of the compiled
//   function (kBorderSearchStackBytes at most, grid/cell_block_range_gpu.h). RunOn::kCpu: 0.
// Fails: kRunOnCallFailed  RunOn::kGpu cannot run (CheckCanRunOn), or a query of the GPU failed.
// Read: docs/method-paper/fluidity-method.typ <sec-grid-memory>
std::expected<MemoryEstimateEntry, GridFailure> BuildBorderSearchStackEstimate(RunOn run_on);

}  // namespace fluidity::grid
