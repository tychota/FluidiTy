// The arrays inside the grid's buffers and the entries that size them, for the grid module's own code and tests.
//
// Purpose: name every array that GridBuffers and GridBuilderBuffers (grid/staggered_grid.h) hold and the entry of
//   BuildGridMemoryEstimate that sizes it, so the estimate and the builders spell each entry once, and the grid's
//   builders and checks get at the arrays in one way, while other modules see only the two classes.
//
// Idea: each array is one memory piece of a RunMemory and one entry of BuildGridMemoryEstimate. The entry's name, from
//   the constants below, says which array it is, and its bytes say how many items the array holds. Before a builder
//   takes any memory piece, CheckGridArrayEntries checks that the estimate names each array of the processor exactly
//   once and nothing else, so an entry the builders do not know, or an array left without an entry, is an internal
//   fault before any memory piece is taken. The arrays have room for max_cell_blocks cell blocks and their contents
//   mean nothing until a build fills them. A GridBuffers also points at the GridBuilderBuffers lent to it, which it
//   never owns.
//   Example: CPU grid buffers for 1,000 blocks hold 1,000 block ids, 27,000 same-size-level neighbour blocks,
//   9,000 bigger and smaller blocks, 64,000 cell states, 240,000 face piece kinds and 1,000 piece map entries; their
//   CPU builder buffers hold 64,000 cell ids.
//
// Words:
//   cell block       4 x 4 x 4 cells kept together
//   builder buffers  the temporary arrays a grid build uses: cell ids, sort keys, two orders and the working
//                    buffers of the stable order
//   memory piece     the bytes of one buffer inside a RunMemory's block (base/run_memory.h)
//   piece map        for each block of a grid piece, the index of the same block in the whole grid
//   internal fault   a state the grid's own code should never reach; never reported as bad input
// Owns:    the entry names of BuildGridMemoryEstimate and of BuildBorderSearchMemoryEstimate, the item bytes of the
//          grid's arrays, GridArray, FindGridArray,
//          ComputeIsBuilderBuffersArray, ComputeIsGridArrayOnProcessor, CheckGridArrayEntries, ToGridFailure,
//          GridBuffersArrays, GridBuilderBuffersArrays, ToGridBuffersArrays, ToGridBuilderBuffersArrays,
//          ComputeHasLoanOf, ComputeSmallerBlocksStart, BuildGridFromBuffers,
//          BuildGridBuilderBuffersFromEntries, BuildGridBuffersFromEntries
// Uses:    CPU-only guard (base/cpu_only.h), Buffer, BufferError (base/buffer.h), MemoryEstimateEntry
//          (base/memory_count.h), RunMemory (base/run_memory.h), RunOn (base/run_on.h), OrderBuffers
//          (base/stable_order.h), GridBuildStatus (grid/grid_build_status.h), GridFailure (grid/grid_error.h),
//          CellBlockId, CellId, GridVersion (grid/grid_ids.h), CellState, FacePieceKind, GridFrame (grid/grid_tables.h)
// Used by: grid/grid_memory_estimate.cc, which spells its entries with these names; grid/staggered_grid.h, which
//          holds these arrays; the grid's builders, checks and tests
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-grid-memory>, <tab-grid-memory>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

class GridBuffers;
class GridBuilderBuffers;
class StaggeredGrid;
struct GridCounts;
struct GridSettings;

// The name of each entry of BuildGridMemoryEstimate, one spelling for the estimate and the builders.
inline constexpr std::string_view kBlockIdsEntryName = "grid: block ids";
inline constexpr std::string_view kSameSizeLevelNeighbourBlocksEntryName = "grid: same-size-level neighbour blocks";
inline constexpr std::string_view kBiggerAndSmallerBlocksEntryName = "grid: bigger and smaller blocks";
inline constexpr std::string_view kCellStatesEntryName = "grid: cell states";
inline constexpr std::string_view kFacePieceKindsEntryName = "grid: face piece kinds";
inline constexpr std::string_view kPieceMapEntryName = "grid: piece map";
inline constexpr std::string_view kBuildStatusEntryName = "grid: build status";
inline constexpr std::string_view kBuilderCellIdsEntryName = "grid: builder buffers: cell ids";
inline constexpr std::string_view kBuilderSortKeysEntryName = "grid: builder buffers: sort keys";
inline constexpr std::string_view kBuilderOrderEntryName = "grid: builder buffers: order";
inline constexpr std::string_view kBuilderSecondOrderEntryName = "grid: builder buffers: second order";
inline constexpr std::string_view kBuilderOrderBuffersEntryName = "grid: builder buffers: order buffers";
// The entries of BuildBorderSearchMemoryEstimate, the arrays of BorderSearchBuffers.
inline constexpr std::string_view kBorderMarksEntryName = "grid: border marks";
inline constexpr std::string_view kBorderSumsEntryName = "grid: border sums";
inline constexpr std::string_view kBorderOrderBuffersEntryName = "grid: border order buffers";
inline constexpr std::string_view kBorderStatusEntryName = "grid: border status";

// Bytes of one item of each array, method paper <tab-grid-memory>, shared by the estimate and the builders, which
// check each array's item type against them at compile time. A block id is 8 bytes, a block index 4.
inline constexpr std::uint32_t kBlockIdBytes = 8;
inline constexpr std::uint32_t kBlockIndexBytes = 4;
// 27 per block: the blocks of the same size level around a block, the block itself included.
inline constexpr std::uint32_t kSameSizeLevelNeighbourBlocksBytes = 27 * kBlockIndexBytes;
// 9 per block: one bigger block and 8 smaller blocks, laid out as ComputeSmallerBlocksStart states.
inline constexpr std::uint32_t kBiggerAndSmallerBlocksBytes = (1 + 8) * kBlockIndexBytes;
inline constexpr std::uint32_t kCellStateBytes = 1;
inline constexpr std::uint32_t kFacePieceKindBytes = 1;
inline constexpr std::uint32_t kPieceMapBytes = kBlockIndexBytes;
inline constexpr std::uint32_t kCellIdBytes = 8;
inline constexpr std::uint32_t kSortKeyBytes = 8;
inline constexpr std::uint32_t kOrderItemBytes = 4;  // A 32-bit index.

// The arrays of a grid's buffers and builder buffers, one for each entry of BuildGridMemoryEstimate.
enum class GridArray : std::uint8_t {
  kBlockIds,
  kSameSizeLevelNeighbourBlocks,
  kBiggerAndSmallerBlocks,
  kCellStates,
  kFacePieceKinds,
  kPieceMap,
  kBuildStatus,  // GPU only.
  kBuilderCellIds,
  kBuilderSortKeys,  // GPU only.
  kBuilderOrder,  // GPU only.
  kBuilderSecondOrder,  // GPU only.
  kBuilderOrderBuffers,  // GPU only.
};

// Finds the array an entry name names, by the constants above. Empty for a name no builder knows.
std::optional<GridArray> FindGridArray(std::string_view entry_name);

// Computes whether the array belongs to GridBuilderBuffers (true) or to GridBuffers (false).
bool ComputeIsBuilderBuffersArray(GridArray array);

// Computes whether the builders build the array on run_on: every array on the GPU; on the CPU the five tables, the
// piece map and the builder buffers' cell ids, which the CPU builder sorts in place.
bool ComputeIsGridArrayOnProcessor(GridArray array, RunOn run_on);

// Checks the estimate of one grid on run_on at max_cell_blocks before a builder takes any memory piece: every entry
// names an array of run_on, each such array has exactly one entry, every entry lives on run_on (kRunOn, or kCpu on
// the CPU), and every entry's bytes are a whole number of the array's items (for the order buffers, of the form
// bytes_per_item x items + fixed_bytes). Changes nothing.
//
// Example: CheckGridArrayEntries(BuildGridMemoryEstimate(RunOn::kGpu, 1, 1000), RunOn::kGpu, 1000) succeeds; the CPU
//   estimate checked for RunOn::kGpu fails, since it has no sort keys.
// Fails: kInternalFault  an entry no builder knows, an array without an entry or with two, an entry on the wrong
//                        processor, or bytes that are not whole items. No input causes it: only a change to the
//                        estimate or to the builders that leaves the other behind.
// Read: docs/method-paper/fluidity-method.typ <sec-grid-memory>
std::expected<void, GridFailure>
CheckGridArrayEntries(std::span<const MemoryEstimateEntry> estimate, RunOn run_on, std::uint32_t max_cell_blocks);

// Converts a buffer failure to a grid failure, the module's one mapping: kOutOfMemory and kRunMemoryTooSmall give
// kOutOfMemory with asked_bytes as magnitude; kSizeMismatch gives kInternalFault, since the grid's room limit keeps
// every count a buffer function checks in range and no build can produce it; kRunOnCallFailed, and any other value,
// give kRunOnCallFailed. A caller's span is not a build: ReadValues and WriteValues (grid/grid_values.h) report their
// own kSizeMismatch, a span that overlaps the values, as kBadInput and do not call this for it.
GridFailure ToGridFailure(BufferError error, std::uint64_t asked_bytes);

// The arrays of one grid version, one per non-builder entry of BuildGridMemoryEstimate, for max_cell_blocks blocks.
// On the CPU build_status is empty. A moved-from set holds empty buffers and a null builder_buffers.
struct GridBuffersArrays {
  Buffer<CellBlockId> block_ids;  // One per block, in the fixed order.
  // 27 per block: the blocks of the same size level around each block, itself included.
  Buffer<std::uint32_t> same_size_level_neighbour_blocks;
  // 9 per block, in the layout ComputeSmallerBlocksStart states.
  Buffer<std::uint32_t> bigger_and_smaller_blocks;
  Buffer<CellState> cell_states;  // 64 per block, one per cell place.
  Buffer<FacePieceKind> face_piece_kinds;  // 240 per block, one per face place.
  // One per block: the index in the whole grid of each block of a grid piece. A whole grid's build leaves it unwritten.
  Buffer<CellBlockIndex> piece_map;
  Buffer<GridBuildStatus> build_status;  // On the GPU one record; empty on the CPU.
  GridBuilderBuffers* builder_buffers = nullptr;  // Lent by the caller of BuildGridBuffers; never owned.
};

// Computes the index in bigger_and_smaller_blocks where the smaller blocks start, and so states the array's layout:
// first the bigger block of each of the room's max_cell_blocks blocks, then from this index on the 8 smaller blocks of
// each block, by block and then by eighth, whatever the block count of the grid. The one statement of the layout, for
// the build and for StaggeredGrid::tables.
// Example: a room of 1,000 blocks keeps the bigger blocks at 0..999 and the 8 smaller blocks of block b at
//   1,000 + 8 b to 1,000 + 8 b + 7.
constexpr std::size_t ComputeSmallerBlocksStart(std::uint32_t max_cell_blocks) {
  return max_cell_blocks;
}

// The builder buffers, one per "grid: builder buffers: ..." entry of BuildGridMemoryEstimate, for max_cell_blocks
// blocks. The CPU builder sorts the cell ids in place, so on the CPU every array but cell_ids is empty and there are
// no order buffers.
struct GridBuilderBuffersArrays {
  Buffer<CellId> cell_ids;  // 64 per block.
  Buffer<std::uint64_t> sort_keys;  // GPU only: 64 per block.
  Buffer<std::uint32_t> order;  // GPU only: 64 per block, plus 1 for the total of a prefix sum.
  Buffer<std::uint32_t> second_order;  // GPU only: as order.
  std::optional<OrderBuffers> order_buffers;  // GPU only: for 64 per block plus 1 items.
};

// The arrays of `buffers`. The grid module's code gets at them only this way.
GridBuffersArrays& ToGridBuffersArrays(GridBuffers& buffers);
const GridBuffersArrays& ToGridBuffersArrays(const GridBuffers& buffers);

// The arrays of the buffers of `grid`, for the grid piece that reads its own piece map.
const GridBuffersArrays& ToGridBuffersArrays(const StaggeredGrid& grid);

// The arrays of `builder_buffers`. The grid module's code gets at them only this way.
GridBuilderBuffersArrays& ToGridBuilderBuffersArrays(GridBuilderBuffers& builder_buffers);
const GridBuilderBuffersArrays& ToGridBuilderBuffersArrays(const GridBuilderBuffers& builder_buffers);

// Computes whether `buffers` hold a loan counted by the lent count of `builder_buffers`: the count the buffers return
// their loan to is not null and is the one builder_buffers own. It fails for other builder buffers, and for lent ones
// that were moved from or assigned to, since the count moves or is left behind with the loan. It reads both objects,
// so builder_buffers must still exist.
bool ComputeHasLoanOf(const GridBuffers& buffers, const GridBuilderBuffers& builder_buffers);

// Builds a grid from buffers that a build has filled: the one place a StaggeredGrid is made. The buffers move into the
// grid, which keeps the frame, the version and the counts.
// Example: BuildGrid, once the tables of version 3 are written, calls it with that version and the counts it found.
// Requires: version 1 or more; counts.cell_blocks at most buffers.max_cell_blocks(); the buffers hold the tables of
//   that version. Checked by assert in debug builds, the first two.
StaggeredGrid
BuildGridFromBuffers(GridBuffers&& buffers, GridFrame frame, GridVersion version, const GridCounts& counts);

// BuildGridBuilderBuffers and BuildGridBuffers (grid/staggered_grid.h) with the estimate given, so a test can pass
// one the builders do not know. The public forms pass BuildGridMemoryEstimate(run_on, 1, max_cell_blocks). Same
// checks, in the same order; then CheckGridArrayEntries, whose kInternalFault comes before any memory piece is taken.
std::expected<GridBuilderBuffers, GridFailure> BuildGridBuilderBuffersFromEntries(
    RunOn run_on,
    std::uint32_t max_cell_blocks,
    std::span<const MemoryEstimateEntry> estimate,
    RunMemory& run_memory
);
std::expected<GridBuffers, GridFailure> BuildGridBuffersFromEntries(
    const GridSettings& settings,
    GridBuilderBuffers& builder_buffers,
    std::span<const MemoryEstimateEntry> estimate,
    RunMemory& run_memory
);

}  // namespace fluidity::grid
