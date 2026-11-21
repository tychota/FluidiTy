// The settings and counts of a grid, the buffers a grid is built into, the built grid, and how its builder buffers
// are shared.
//
// Purpose: declare the values a caller chooses before it builds a grid (the room and the processor), the buffers
//   built once for those choices, the build of one grid version into them and the grid that owns them then, the counts
//   a built grid reports, and the choice of how many sets of builder buffers a run keeps.
//
// Idea: a grid lives in arrays with room for a fixed number of cell blocks, chosen once for the run: the grid
//   buffers. Building a grid also needs temporary arrays, the builder buffers. They hold nothing between builds, so
//   two grids that are never built at the same time can share one set of them: the caller owns the set and lends it
//   to each set of grid buffers. Every array is one memory piece of the run's reserved block (base/run_memory.h) and
//   one entry of the grid's memory estimate (grid/grid_memory_estimate.h), which counts the builder buffers once or
//   once per grid, as BuilderBuffersSharing says. The builders take their memory pieces in the order of those
//   entries, so the bytes they take equal the estimate. They do not read an entry's `where`: on the CPU the kRunOn
//   and the kCpu entries all come from the one RunMemory passed, so a CPU caller reserves both in one CPU block.
//   A build fills the grid buffers with the tables of one grid version and allocates nothing; the buffers then belong
//   to the StaggeredGrid it returns, until TakeBuffers gives them back for the next version.
//   Example: the step keeps a committed grid and a try grid. With kOneForAllGrids they share one set of builder
//   buffers, and the estimate counts that set once: on the CPU, 2 x 460 + 512 = 1,432 bytes per block.
//
// Words:
//   cell block       4 x 4 x 4 cells kept together
//   builder buffers  the temporary arrays a grid build uses: cell ids, sort keys, two orders and the working
//                    buffers of the stable order
//   room             the number of cell blocks an array has memory for: max_cell_blocks
//   memory piece     the bytes of one buffer inside a RunMemory's block (base/run_memory.h)
// Owns:    kMaxCellBlocksLimit, GridSettings, GridInputs, GridCounts, BuilderBuffersSharing, GridBuilderBuffers,
//          BuildGridBuilderBuffers, GridBuilderBuffersSize, ComputeGridBuilderBuffersSize, GridBuffers,
//          BuildGridBuffers, StaggeredGrid, BuildGrid, TakeBuffers, ReadGrid, ComputeGridCounts, CheckSizeStepRule
// Uses:    CPU-only guard (base/cpu_only.h), Buffer (base/buffer.h), MemoryEstimateEntry (base/memory_count.h),
//          RunMemory (base/run_memory.h), RunOn (base/run_on.h), GridBuffersArrays, GridBuilderBuffersArrays,
//          BuildGridBuilderBuffersFromEntries, BuildGridBuffersFromEntries, BuildGridFromBuffers, ComputeHasLoanOf
//          (grid/grid_buffers_internal.h), GridFailure (grid/grid_error.h), CellId, GridVersion, CellBlockRange,
//          kMaxSizeLevels (grid/grid_ids.h), GridFrame, GridTables (grid/grid_tables.h)
// Implementation uses: ReadBuffer (base/buffer.h), CheckCanRunOn (base/check_can_run_on.h), ComputeMemoryBytes
//          (base/memory_count.h), BuildOrderBuffers (base/stable_order.h), BuildGridMemoryEstimate, ToMemoryCounts
//          (grid/grid_memory_estimate.h), BuildSortedCellList, CheckCellListInputs and the checks of a cell list
//          (grid/grid_cell_checks.h), FindSortedCellAt, ComputeSplitCellCountToAdd, ComputeBigCellAtSizeLevel,
//          ToBlockSortKey, FindBlockNeighbours, ComputeBlockFaultKey, ComputeCoveredCellState, ComputeFacePieceKind,
//          ComputeFailureCells (grid/grid_build_steps.h), ToCellPlaceIndex (grid/grid_ids.h),
//          FindCellBlock (grid/grid_tables.h); and, when GPU code is compiled in, the GPU path of BuildGrid and
//          ComputeGridCounts: ApplyListChecksInParallel, ApplyCallsInOrder, ToInputCells,
//          ToGridFailure(const GridBuildStatus&) (grid/grid_cell_checks.h), BuildStableOrder,
//          ComputeExclusivePrefixSum (base/stable_order.h), ToBufferError (base/buffer.h), ComputeSortCount
//          (grid/grid_build_steps.h) and the GPU half of P7 to P12 (grid/staggered_grid_gpu.h)
// Used by: the grid's builders and memory estimate, the step, which chooses the room, owns the buffers, counts a list
//          to plan it and builds its grids, every module that reads a grid's tables, and cell resize, which checks the
//          lists it splits
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-grid-memory>, <sec-run-memory>,
//          <proc-build-grid>, <sec-parallel-build>, <proc-count-grid>, <sec-face-pieces>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// The largest max_cell_blocks: floor((2^32 - 1) / 240) = 17,895,697. The face places bind: with 240 per block, every
// face place index fits 32 bits and stays below kNoIndex (240 x 17,895,697 = 4,294,967,280). The cell places, 64 per
// block, then fit the 32-bit orders and prefix sums of base/, and the corner places, 125 per block, fit 32 bits too.
inline constexpr std::uint32_t kMaxCellBlocksLimit = 17895697;
static_assert(std::uint64_t{kFacePlacesPerBlock} * kMaxCellBlocksLimit <= kNoIndex, "face places fit 32 bits");
static_assert(std::uint64_t{kCornerPlacesPerBlock} * kMaxCellBlocksLimit <= kNoIndex, "corner places fit 32 bits");
static_assert(std::uint64_t{kCellPlacesPerBlock} * kMaxCellBlocksLimit <= kNoIndex, "cell places fit 32 bits");
static_assert(std::uint64_t{kFacePlacesPerBlock} * (kMaxCellBlocksLimit + 1) > kNoIndex, "the largest such room");

// What a caller chooses once for a grid's buffers.
struct GridSettings {
  // Room, 1..kMaxCellBlocksLimit. The default 0 is no room: a build that takes these settings fails with kBadInput.
  // A run's room comes from its plan, never from a constant.
  std::uint32_t max_cell_blocks = 0;
  RunOn run_on = RunOn::kCpu;  // The caller's choice. Never changed silently.
};

// What one build or check of a list of active cells needs. The list is the only description of the grid.
// When active_cell_buffer is set it is the list, its first active_cell_buffer_count entries, and active_cells must be
// empty; otherwise active_cells is the list. An empty list is a valid grid with no cells.
// Example: a CPU caller passes {.frame = frame, .active_cells = cells, .version = {1}}; a GPU run passes a buffer of
//   ids on the GPU and its count instead.
struct GridInputs {
  GridFrame frame;
  std::span<const CellId> active_cells;  // A list in CPU memory, any order.
  const Buffer<CellId>* active_cell_buffer = nullptr;  // Or a list in a buffer, any order.
  std::uint32_t active_cell_buffer_count = 0;  // Entries of *active_cell_buffer used.
  GridVersion version;  // 1 or more: the version of the grid before plus 1.
};

// What a built grid knows about its size, kept on the CPU. Everything the memory estimate needs from a grid.
struct GridCounts {
  std::uint32_t cell_blocks;
  std::uint64_t active_cells_per_size_level[kMaxSizeLevels];  // NOLINT(modernize-avoid-c-arrays): as GridBuildStatus.
  std::uint64_t active_cells;
  std::uint64_t face_pieces;  // Places whose kind is not kNotAPiece.
};

// How many sets of builder buffers the "builder buffers" entries of BuildGridMemoryEstimate count.
enum class BuilderBuffersSharing : std::uint8_t {
  kOnePerGrid,  // One set of builder buffers for each of the grid_count sets of grid buffers.
  kOneForAllGrids,  // One set lent to all of them.
};

// The temporary arrays of grid builds and checks on one processor, for grids of at most max_cell_blocks
// blocks: the "grid: builder buffers: ..." entries of BuildGridMemoryEstimate. Holds nothing between calls.
// Owned by the caller and lent to every GridBuffers built with it: those must not outlive it, it is neither moved,
// assigned to nor destroyed while they exist, and one build or check uses it at a time. Move-only.
//
// Every array is a memory piece of the RunMemory it was built from, which must outlive it. lent_count() counts the
// GridBuffers that hold a loan of these builder buffers; moving, assigning to or destroying them while it is above 0
// is a bug in the caller, checked by assert in debug builds. The count lives in a small CPU allocation of its own (8
// bytes, outside every RunMemory and the memory estimate), which moves with the arrays, so a GridBuffers returns its
// loan to the count, never to the object. A release build goes on after such a bug. Builder buffers destroyed or
// assigned to while lent leave their count behind, never freed; builder buffers moved while lent pass it on to the
// object they move into. Either way the late GridBuffers count down into memory that still exists; their
// builder_buffers pointer then names an object that was destroyed, emptied by a move, or given other arrays, as with a
// RunMemory destroyed before its memory pieces.
// Moved-from builder buffers have max_cell_blocks() 0, byte_count() 0 and lent_count() 0, and keep their run_on().
class GridBuilderBuffers {
 public:
  GridBuilderBuffers(const GridBuilderBuffers&) = delete;
  GridBuilderBuffers& operator=(const GridBuilderBuffers&) = delete;
  GridBuilderBuffers(GridBuilderBuffers&& other) noexcept;
  GridBuilderBuffers& operator=(GridBuilderBuffers&& other) noexcept;
  ~GridBuilderBuffers();

  // Room: the most cell blocks of a grid these buffers can build. 0 after a move.
  [[nodiscard]] std::uint32_t max_cell_blocks() const {
    return max_cell_blocks_;
  }

  // The processor the arrays live on.
  [[nodiscard]] RunOn run_on() const {
    return run_on_;
  }

  [[nodiscard]] std::size_t byte_count() const;  // Bytes of its arrays; 0 after a move.

  // The GridBuffers built with these builder buffers and not yet destroyed or assigned to; a move of a GridBuffers
  // passes its loan on. 0 after a move of the builder buffers.
  [[nodiscard]] std::size_t lent_count() const {
    return lent_count_ == nullptr ? 0 : *lent_count_;
  }

 private:
  friend std::expected<GridBuilderBuffers, GridFailure> BuildGridBuilderBuffersFromEntries(
      RunOn run_on,
      std::uint32_t max_cell_blocks,
      std::span<const MemoryEstimateEntry> estimate,
      RunMemory& run_memory
  );
  friend std::expected<GridBuffers, GridFailure> BuildGridBuffersFromEntries(
      const GridSettings& settings,
      GridBuilderBuffers& builder_buffers,
      std::span<const MemoryEstimateEntry> estimate,
      RunMemory& run_memory
  );
  friend GridBuilderBuffersArrays& ToGridBuilderBuffersArrays(GridBuilderBuffers& builder_buffers);
  friend const GridBuilderBuffersArrays& ToGridBuilderBuffersArrays(const GridBuilderBuffers& builder_buffers);
  friend bool ComputeHasLoanOf(const GridBuffers& buffers, const GridBuilderBuffers& builder_buffers);

  GridBuilderBuffers() = default;

  // Frees the lent count, or, when loans are still out, leaves it behind for the late GridBuffers to count down.
  void FreeLentCount();

  GridBuilderBuffersArrays arrays_;
  std::uint32_t max_cell_blocks_ = 0;
  RunOn run_on_ = RunOn::kCpu;
  // GridBuffers holding a loan. On the heap, so a GridBuffers can count down into it even after a bug destroyed
  // these builder buffers first. Null only in moved-from builder buffers.
  std::unique_ptr<std::size_t> lent_count_;
};

// Builds them as memory pieces of run_memory, in the order of the estimate's entries.
//
// Method: CheckCanRunOn(run_on) first, then the room, then CheckGridArrayEntries (grid/grid_buffers_internal.h) on
//   BuildGridMemoryEstimate(run_on, 1, max_cell_blocks), then one memory piece for each "grid: builder buffers: ..."
//   entry, of exactly that entry's bytes: BuildBuffer for an array, BuildOrderBuffers for the order buffers. An entry's
//   `where` is not read: on the CPU every entry comes from run_memory, so a CPU caller reserves the kRunOn and the kCpu
//   entries in one CPU block. A RunMemory of ComputeRunMemoryBytes of those entries holds them to the byte.
// Example: BuildGridBuilderBuffers(RunOn::kCpu, 1000, run_memory) takes one memory piece of 512,000 bytes, the cell
//   ids; byte_count() is 512,000.
// Requires: run_on == run_memory.run_on() when run_on can run, checked by assert in debug builds.
// Fails: kRunOnCallFailed  CheckCanRunOn(run_on) failed; nothing is taken from run_memory.
//        kBadInput         max_cell_blocks is 0 or above kMaxCellBlocksLimit; nothing is taken.
//        kInternalFault    the estimate and the builders disagree on the arrays; nothing is taken.
//        kOutOfMemory      an array does not fit in what run_memory has left; magnitude is its bytes. Also when the
//                          8 bytes of the lent count cannot be allocated, before any memory piece is taken.
//        The memory pieces already taken stay in the block, since a memory piece is never freed alone. No builder
//        buffers are returned on failure.
// Read: docs/method-paper/fluidity-method.typ <sec-grid-memory>, <sec-run-memory>
std::expected<GridBuilderBuffers, GridFailure>
BuildGridBuilderBuffers(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory);

// Exact bytes of a GridBuilderBuffers: bytes_per_cell_block * max_cell_blocks + fixed_bytes. These are the bytes of
// the arrays, without the rounding of their memory pieces, so they are not what a RunMemory must reserve for them: on
// the GPU the builder buffers are five memory pieces, each rounded up to 256 bytes. At 1,000 blocks on the GPU the
// size gives 1,924,110 bytes and the memory pieces take 1,924,864. A reservation uses ComputeRunMemoryBytes of the
// estimate's entries (base/memory_count.h).
struct GridBuilderBuffersSize {
  std::uint32_t bytes_per_cell_block = 0;
  std::uint64_t fixed_bytes = 0;
};

// The sum of the "grid: builder buffers: ..." entries of BuildGridMemoryEstimate: one definition for both.
//
// Method: every builder entry grows linearly with the room, so the sum at 0 blocks is fixed_bytes and the sum at 1
//   block minus the sum at 0 blocks is bytes_per_cell_block. The bytes of the arrays, not of their memory
//   pieces (GridBuilderBuffersSize).
// Example: RunOn::kCpu gives {512, 0}, the cell ids. RunOn::kGpu gives {1920, 4110}: cell ids and sort keys
//   (512 + 512), two orders (256 + 4 each) and the order buffers (64 x 6 + 6 + 4,096).
// Read: docs/method-paper/fluidity-method.typ <sec-grid-memory>, <tab-grid-memory>
GridBuilderBuffersSize ComputeGridBuilderBuffersSize(RunOn run_on);

// The arrays one grid version is built into, with room for max_cell_blocks() blocks: the entries of
// BuildGridMemoryEstimate that are not builder buffers. They are the five tables and the piece map, and on the GPU also
// the build status record; a GPU grid keeps nothing in CPU memory. They use the GridBuilderBuffers lent by
// BuildGridBuffers for every build and check, and never own it. They belong to no grid, so their contents mean nothing.
//
// Move-only. Every array is a memory piece of the RunMemory it was built from, which must outlive it. The loan of the
// builder buffers ends when the GridBuffers is destroyed or assigned to, and moves with it. A moved-from GridBuffers
// has room 0 (max_cell_blocks() == 0), no arrays and no builder buffers, and keeps its run_on().
class GridBuffers {
 public:
  GridBuffers(const GridBuffers&) = delete;
  GridBuffers& operator=(const GridBuffers&) = delete;
  GridBuffers(GridBuffers&& other) noexcept;
  GridBuffers& operator=(GridBuffers&& other) noexcept;
  ~GridBuffers();

  // Room: the most cell blocks a grid built into these buffers may have. 0 after a move.
  [[nodiscard]] std::uint32_t max_cell_blocks() const {
    return max_cell_blocks_;
  }

  // The processor the arrays live on.
  [[nodiscard]] RunOn run_on() const {
    return run_on_;
  }

 private:
  friend std::expected<GridBuffers, GridFailure> BuildGridBuffersFromEntries(
      const GridSettings& settings,
      GridBuilderBuffers& builder_buffers,
      std::span<const MemoryEstimateEntry> estimate,
      RunMemory& run_memory
  );
  friend GridBuffersArrays& ToGridBuffersArrays(GridBuffers& buffers);
  friend const GridBuffersArrays& ToGridBuffersArrays(const GridBuffers& buffers);
  friend bool ComputeHasLoanOf(const GridBuffers& buffers, const GridBuilderBuffers& builder_buffers);

  GridBuffers() = default;

  // Ends the loan of the builder buffers, if any: one fewer in their lent count.
  void ReturnLoan();

  GridBuffersArrays arrays_;
  std::uint32_t max_cell_blocks_ = 0;
  RunOn run_on_ = RunOn::kCpu;
  std::size_t* lent_count_ = nullptr;  // The lent count of the builder buffers, while this holds a loan.
};

// Builds a grid's own arrays (five tables, the piece map and, on the GPU, the status record) for
// settings.max_cell_blocks blocks on settings.run_on, as memory pieces of run_memory. The buffers use
// builder_buffers for every build and check.
//
// Method: CheckCanRunOn(settings.run_on) first, then the room and the lent builder buffers' room, then
//   CheckGridArrayEntries on BuildGridMemoryEstimate(settings.run_on, 1, settings.max_cell_blocks), then one memory
//   piece for each entry that is not a builder buffers entry, of exactly that entry's bytes, in the order of the
//   entries. As for BuildGridBuilderBuffers, an entry's `where` is not read: on the CPU a caller reserves the kRunOn
//   and the kCpu entries in one CPU block. On success the builder buffers' lent count grows by one.
// Example: CPU settings of 1,000 blocks take memory pieces of 8,000, 108,000, 36,000, 64,000, 240,000 and 4,000
//   bytes, 460 bytes per block; with the 512,000 bytes of their builder buffers, 972 bytes per block.
// Requires: builder_buffers.run_on() == settings.run_on == run_memory.run_on() when that processor can run, checked
//   by assert in debug builds.
// Requires: builder_buffers outlives the returned buffers and is neither moved, assigned to nor destroyed while they
//   exist, checked by assert in debug builds through its lent count.
// Fails: kRunOnCallFailed, kBadInput (room 0 or above the limit, or builder_buffers.max_cell_blocks() below
//        settings.max_cell_blocks), each before any memory piece is taken; kInternalFault and kOutOfMemory as above.
//        On failure builder_buffers is unchanged, and the memory pieces already taken stay in the block.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-grid-memory>, <sec-run-memory>
std::expected<GridBuffers, GridFailure>
BuildGridBuffers(const GridSettings& settings, GridBuilderBuffers& builder_buffers, RunMemory& run_memory);

// One built grid version: the grid buffers that hold its tables, its frame, its version and its counts. A grid keeps
// pressure at cell centers and each velocity part on the faces it crosses; also called MAC grid, Harlow and Welch 1965.
// Its tables live on the processor of its buffers only, and its version and counts are kept on the CPU.
//
// Move-only; owns its GridBuffers. BuildGrid makes one, and TakeBuffers gives its buffers back for the next version.
// A moved-from grid, or one whose buffers were taken back, has version 0, zero counts and no tables.
class StaggeredGrid {
 public:
  StaggeredGrid(const StaggeredGrid&) = delete;
  StaggeredGrid& operator=(const StaggeredGrid&) = delete;
  StaggeredGrid(StaggeredGrid&& other) noexcept;
  StaggeredGrid& operator=(StaggeredGrid&& other) noexcept;
  ~StaggeredGrid() = default;

  // 1 or more for a built grid.
  [[nodiscard]] GridVersion version() const {
    return version_;
  }

  [[nodiscard]] const GridCounts& counts() const {
    return counts_;
  }

  // Every block of the grid: {0, counts().cell_blocks}.
  [[nodiscard]] CellBlockRange all_blocks() const {
    return {.first_block = 0, .block_count = counts_.cell_blocks};
  }

  // The tables on `run_on`: those of the grid when run_on is run_on(). The other processor holds no table, so for it
  // the result has version 0, no block and null arrays.
  // Example: tables(RunOn::kCpu) of a CPU grid of 2 blocks has block_count 2; tables(RunOn::kGpu) of it has version 0.
  [[nodiscard]] GridTables tables(RunOn run_on) const;

  // The processor of the buffers.
  [[nodiscard]] RunOn run_on() const {
    return buffers_.run_on();
  }

  // The room of the buffers, at least counts().cell_blocks.
  [[nodiscard]] std::uint32_t max_cell_blocks() const {
    return buffers_.max_cell_blocks();
  }

 private:
  friend StaggeredGrid
  BuildGridFromBuffers(GridBuffers&& buffers, GridFrame frame, GridVersion version, const GridCounts& counts);
  friend GridBuffers TakeBuffers(StaggeredGrid&& grid);
  friend std::expected<StaggeredGrid, GridFailure> ReadGrid(const StaggeredGrid& grid, GridBuffers& cpu_buffers);

  StaggeredGrid(GridBuffers&& buffers, GridFrame frame, GridVersion version, const GridCounts& counts);

  GridBuffers buffers_;
  GridFrame frame_{};
  GridVersion version_{};
  GridCounts counts_{};
};

// Builds the grid of `inputs` into `buffers`: the tables of one grid version, method paper <proc-build-grid>. On
// success the buffers move into the returned grid and `buffers` is left moved-from. On failure no grid is returned,
// and `buffers` keeps its arrays and its lent builder buffers, so it can build again.
//
// Method: the input checks and steps 1 to 4 as BuildSortedCellList (grid/grid_cell_checks.h), siblings required, which
//   leave the n active cells sorted in the lent builder buffers' cell ids. Step 5 appends the split cells that
//   ComputeSplitCellCountToAdd and ComputeBigCellAtSizeLevel give, sorts the active and the split cells each by their
//   block, and walks both in the fixed order of their blocks, writing each block id once. Step 6 runs
//   FindBlockNeighbours for every block; then ComputeBlockFaultKey, step 7 of the failure key (<sec-parallel-build>),
//   checks every block. Step 7 marks every cell place kAbsent, each active cell kActive and each split cell kSplit at
//   its place (ToCellPlaceIndex), then gives each place still kAbsent its ComputeCoveredCellState. Step 8 writes
//   ComputeFacePieceKind at every face place and counts the face pieces. Every table entry comes from those shared
//   functions (grid/grid_build_steps.h, grid/grid_ids.h), the ones the GPU build calls, so the two give the same
//   bytes. On GPU buffers the same steps run in parallel, <proc-build-grid-in-parallel> P1 to P12, through the same
//   per-item rules; the status record, which holds the counts or the first failure, is the one value copied back to
//   the CPU, and no table leaves the GPU. Allocates nothing on either processor.
// Order: each step runs over every position before the next, so the failure reported is the first by the failure key
//   (<sec-parallel-build>): the GPU's smallest key.
// Example: with 3 size levels, size level 0 {0,0,0} beside the 8 size-level-1 cells of size level 0 {1,0,0} gives
//   version 1, 2 blocks (size level 0 (0,0,0), then size level 1 (0,0,0)), 9 active cells and 41 face pieces. The 9
//   size-level-0 cells {4 i, 0, 0}, each in a block of its own, into buffers for 8 blocks are kTooManyCellBlocks.
// Requires: as CheckSizeStepRule(inputs, buffers).
// Fails: kBadInput, kCoordOutOfRange, kCellsOverlap, kSizeStepRuleBroken  as CheckSizeStepRule(inputs, buffers)
//        kSiblingCellMissing  a big cell is split and one of its small cells is not present: cell = that small cell, or
//                             CellId{0} beyond the coordinate range; other_cell = its big cell
//                             (grid/grid_cell_checks.h, CheckSortedCells, says which cell when the missing one is
//                             eighth 0)
//        kTooManyCellBlocks   more than 64 * buffers.max_cell_blocks() cells, before any id is read; the active and
//                             split cells above that number (step 5); more blocks than buffers.max_cell_blocks() (step
//                             5). cell and other_cell zero: the GPU cannot name the first cell beyond the room without
//                             working memory beyond it. The caller raises the room.
//        kRunOnCallFailed     a GPU call failed: the buffers' contents are unspecified, and the run stops
//        kInternalFault       a state no input causes, checked in every build: a table array whose size is not the
//                             room's, a block of size level 1 or more without its bigger block (step 7 of the failure
//                             key), a cell whose block step 5 did not write, or a failure whose cells disagree with
//                             it; on the GPU also a builder array too small for the room, a size error from a stable
//                             order, a prefix sum or a read of base/, more active and split cells than the sort by
//                             block orders (kSortCountFaultKey), or a status record that contradicts itself or, without
//                             a failure, counts more blocks than the room or other active cells than the list; zero
//                             cells
//        After a failed input check nothing is written; after a later failure the builder buffers' cell ids and the
//        tables are unspecified.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <sec-parallel-build>,
//       <sec-cell-blocks>, <sec-face-pieces>
std::expected<StaggeredGrid, GridFailure> BuildGrid(const GridInputs& inputs, GridBuffers& buffers);

// Takes back the buffers of a grid that is no longer needed, for the next BuildGrid. Allocates nothing.
//
// Method: the buffers move out of the grid with their arrays and their loan of builder buffers. The grid is left with
//   version 0, zero counts and no tables, and every GridTables taken from it is invalid.
// Example: BuildGrid of version 1 into buffers, TakeBuffers(std::move(grid)), then BuildGrid of version 2 into the
//   same buffers: the tables of version 2 lie in the same memory.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
GridBuffers TakeBuffers(StaggeredGrid&& grid);

// Copies the used part of the five tables of `grid` into `cpu_buffers` and returns a grid on the CPU with the same
// frame, version and counts: the one explicit copy of a grid's tables, for tests, tools and a caller that accepts a
// transfer from the GPU. Allocates nothing. On success `cpu_buffers` moves into the returned grid and is left
// moved-from; on failure it keeps its arrays and its loan and can build again.
//
// Method: one read of base/ for each part of a table, from the grid's arrays to the front of the buffers' arrays: the
//   block ids, 27 neighbours per block, the bigger block of each block, the 8 smaller blocks of each block, 64 cell
//   states and 240 face piece kinds per block, 456 bytes per block. Only the first counts().cell_blocks blocks of
//   each array are read, so the bytes are 456 per block of the grid, never of the room. The bigger blocks and the
//   smaller blocks are two reads, since the layout of bigger_and_smaller_blocks depends on the room
//   (ComputeSmallerBlocksStart): the smaller blocks go where the room of `cpu_buffers` puts them, not where the room
//   of `grid` has them, so the two rooms may differ. The piece map is no part of a grid and is not copied. The loan
//   of builder buffers stays with `cpu_buffers`: a copy builds nothing, so the returned grid's buffers keep the loan
//   they had, and TakeBuffers then BuildGrid work on them as on buffers a build filled. The reads stop at the first
//   failure. A GPU grid goes through the same reads, which then cross from the GPU; a failed read from the GPU and a
//   destination whose arrays do not have its room's size are the two failures no run on the CPU reaches.
// Example: a grid of 2 blocks in a room of 252, read into buffers of room 4,096, gives a grid with the same version
//   and counts whose first 2 blocks hold the same tables; max_cell_blocks() of the result is 4,096.
// Fails: kBadInput           `grid` holds no tables (version 0: moved from, or its buffers were taken back);
//                            `cpu_buffers` is moved from (room 0) or is not on the CPU, checked in every build before
//                            anything is written
//        kTooManyCellBlocks  cpu_buffers.max_cell_blocks() is below grid.counts().cell_blocks; cell and other_cell
//                            zero
//        kRunOnCallFailed    a read from the GPU failed
//        kInternalFault      a state no input causes, checked in every build: a table array of `cpu_buffers` whose
//                            size is not its room's, or a size error from a read of base/ after the sizes were checked
//        After a failed check nothing is written. After a failed read the tables of `cpu_buffers` are unspecified.
//        After a success the part of each table beyond the first counts().cell_blocks blocks is unspecified: it may
//        hold the entries of an earlier grid.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-buffer>
std::expected<StaggeredGrid, GridFailure> ReadGrid(const StaggeredGrid& grid, GridBuffers& cpu_buffers);

// Counts the cell blocks, the active cells and the face pieces that a list of active cells in CPU memory would give,
// without building the grid: what a run's plan needs to choose the room before any buffers exist.
//
// Method: method paper <proc-count-grid>. (1) The checks of a build except the length and the room: CheckGridInputs,
//   the ids in input order, then a CPU copy of the ids sorted into the fixed order and checked for overlaps, siblings
//   and the size step rule. (2) The active cells, per size level, in one pass. (3) The face pieces, one active cell at
//   a time, by binary searches in the sorted copy: is the place of the cell's own size level beside a side active, and
//   is it split. The low side counts one face piece unless the place is split; the high side counts one unless the
//   place is active or split. A place inside a bigger cell and an absent place give the same count, so the count needs
//   no lookup of the bigger cell: at most 9 searches per cell. (4) The blocks: the split cells appended to the copy,
//   all ids sorted by their block, and the runs of one block counted. This is a second statement of the face rule of
//   <sec-face-pieces>, from the places beside each cell instead of from built tables. The test
//   ComputeGridCounts.RandomGrids_EqualGeometricChecker holds it to the face rule. It builds no table because tables
//   need buffers sized by the block count, the number asked for; sized for the worst case, one block per cell, they
//   would cost about 1.1 kB per active cell.
// Memory: one CPU allocation of n + floor(n / 7) ids for n active cells, 8 bytes each: 8 bytes per active or split
//   cell, since a split cell has 8 small cells and so there are at most n / 7 of them; at most 9.2 bytes per active
//   cell. Two sorts and four linear passes, O(n log n).
// Example: with 3 size levels, size level 0 {0,0,0} beside the 8 size-level-1 cells of size level 0 {1,0,0} gives 2
//   blocks, active_cells_per_size_level {1, 8, 0, ...}, 9 active cells and 41 face pieces: the big cell's 5 whole
//   sides, the small cells' 24 outer sides (4 of them shared with the big cell) and the 12 between them. The copy
//   holds 9 + 1 ids, 80 bytes.
// Requires: the list is in CPU memory: the span, or a CPU active_cell_buffer. An empty list has no processor. Checked
//   by assert in debug builds.
// Fails: kBadInput           as CheckGridInputs; more than 2^32 - 1 cells; then the first bad id in input order, as
//                            CheckSizeStepRule's span form
//        kCoordOutOfRange, kCellsOverlap, kSiblingCellMissing, kSizeStepRuleBroken  as a build of the list, with the
//                            same cells: the first failure by step, then by position
//        kOutOfMemory        the copy could not be allocated; magnitude = its bytes, 8 (n + floor(n / 7)); zero cells
//        kInternalFault      more split cells than the copy has room for, more than 2^32 - 1 blocks, or the cells
//                            named for a failure disagree with it; zero cells. No input causes it.
//        Never kTooManyCellBlocks: there is no room yet. An empty list gives zero counts and allocates nothing.
// Read: docs/method-paper/fluidity-method.typ <proc-count-grid>, <sec-face-pieces>, <sec-cell-blocks>
std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs);

// Counts the cell blocks, the active cells and the face pieces of the list of `inputs` as BuildGrid would, by running
// the steps of that build into `buffers` and reading the counts from what it wrote. Allocates nothing. The buffers
// hold no grid before or after; their arrays and loan stay, and their contents become unspecified.
//
// Method: the steps of BuildGrid, siblings required, so the counts are those of the grid BuildGrid would return. Unlike
//   the one-argument form, which has no room, this form needs the room of `buffers`. On GPU buffers the counts come
//   from the status record of the build, the one value copied back to the CPU.
//   An empty list gives zero counts, as the one-argument form does.
// Example: the 9 size-level-0 cells {4 i, 0, 0}, each in a block of its own, into buffers for 8 blocks are
//   kTooManyCellBlocks with cell and other_cell zero, while the one-argument form counts 9 cell blocks.
// Requires: as CheckSizeStepRule(inputs, buffers).
// Fails: as BuildGrid for the same inputs and buffers, including kBadInput when `buffers` is moved from (a build moved
//        it into a grid, or it was moved to another object), kTooManyCellBlocks when the grid does not fit the
//        buffers (cell and other_cell zero) and kRunOnCallFailed when a GPU call failed. A failed call leaves `buffers`
//        usable, except after kRunOnCallFailed, when the run stops.
// Read: docs/method-paper/fluidity-method.typ <proc-count-grid>, <proc-build-grid>
std::expected<GridCounts, GridFailure> ComputeGridCounts(const GridInputs& inputs, GridBuffers& buffers);

// Checks a list of active cells in CPU memory: its ids, overlaps and the size step rule, but not the siblings, since
// cell resize checks its lists while it splits cells and some siblings do not exist yet.
//
// Method: the frame first (finite origin, positive biggest cell width, 1 to 16 size levels). Then method paper
//   <proc-build-grid> steps 1, 2 and 4: every id in input order, then a CPU copy of the ids sorted into the fixed order
//   and checked for overlaps, then for the size step rule, each step over every position before the next. The failure
//   reported is the first by step, then by position: the one the GPU's smallest failure key gives.
// Example: with 3 size levels, size level 0 {0,0,0} and size level 2 {4,4,4} touch at a corner: kSizeStepRuleBroken,
//   cell size level 2 {4,4,4}, other_cell size level 0 {0,0,0}. Size level 0 {0,0,0} beside size level 1 {2,0,0}
//   passes, and so do 7 of the 8 small cells of one big cell. The list {a, a, b} where b has bit 63 clear is
//   kBadInput naming b: step 1 comes before the overlap of step 2.
// Requires: active_cells is in CPU memory.
// Fails: kBadInput           a frame value as above, or more than 2^32 - 1 cells, which no 32-bit position indexes;
//                            then the first id in input order
//                            with bit 63 clear, bits 61..62 set, or a size level of size_level_count or more: cell =
//                            that id as given
//        kCoordOutOfRange    the first id in input order with a coordinate of kMaxCoord + 1: cell = that id
//        kCellsOverlap       the first overlap in the fixed order: cell = the smaller or repeated cell, other_cell =
//                            the bigger cell that holds it, or the same cell
//        kSizeStepRuleBroken the first break in the fixed order: cell = the smaller cell, other_cell = the active cell
//                            two or more size levels bigger that touches it
//        kInternalFault      the failure found and the cells named for it disagree; zero cells. No input causes it.
//        kOutOfMemory        the CPU copy of the ids could not be allocated; magnitude = its bytes, 8 per cell.
//        Allocates one CPU copy of the ids, 8 B per cell, and writes nothing else.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <sec-parallel-build>
std::expected<void, GridFailure> CheckSizeStepRule(GridFrame frame, std::span<const CellId> active_cells);

// Checks the list of `inputs` as the span form does, on buffers.run_on(), allocating nothing on either processor: the
// lent builder buffers' cell ids hold the sorted copy. The buffers hold no grid before or after, and their contents are
// unspecified afterwards.
//
// Method: the input checks come first, in this order, and never take part in the order of failures: the frame, the
//   version and the shape of the list (two lists, a count above its buffer), then the lent builder buffers, then the
//   length, more than 64 * buffers.max_cell_blocks() cells, before any id is read. Then steps 1, 2 and 4 of
//   <proc-build-grid> as the span form. On the GPU the same steps run as <proc-build-grid-in-parallel> steps P1 to P4
//   and P6, through the same per-item rules (BuildSortedCellList, grid/grid_cell_checks.h), so the result is the CPU's;
//   the status record is the one value copied back to the CPU.
// Example: grid buffers for 1 block and a list of 65 cells whose first id has bit 63 clear: kTooManyCellBlocks with
//   cell and other_cell zero, not kBadInput. The same list with version 0 is kBadInput.
// Requires: the list is on buffers.run_on(), a span in CPU memory, except an empty list, which has no processor;
//   checked by assert in debug builds. The lent builder buffers still exist: builder buffers that were destroyed
//   cannot be detected.
// Fails: kBadInput           a frame value as for the span form; version 0; both lists given; active_cell_buffer_count
//                            above the size of active_cell_buffer, or above 0 without one; a list that lies in the
//                            lent builder buffers' cell ids; buffers without lent builder buffers, with builder
//                            buffers other than the lent ones, or lent ones with less room than the buffers. Each of
//                            these writes nothing. Then a bad id as for the span form.
//        kTooManyCellBlocks  more than 64 * buffers.max_cell_blocks() cells, a list of any length; cell and other_cell
//                            zero; nothing written
//        kRunOnCallFailed    a GPU call failed: the buffers' contents are unspecified, and the run stops
//        kCoordOutOfRange, kCellsOverlap, kSizeStepRuleBroken as for the span form
//        kInternalFault      as for the span form, or a size error from a copy, an order or a read of base/ after the
//                            sizes were checked, or on the GPU a builder array too small for the list or a status
//                            record that contradicts itself (a key without an error, an error without a key, a key of
//                            no step 1 to 7); zero cells
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <sec-parallel-build>, <proc-build-grid-in-parallel>
std::expected<void, GridFailure> CheckSizeStepRule(const GridInputs& inputs, GridBuffers& buffers);

}  // namespace fluidity::grid
