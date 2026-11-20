#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {
namespace {

// The item bytes of a buffer type, so each array's own item type is compared with its entry's item bytes below.
template <typename BufferType>
inline constexpr std::size_t kItemBytesOf = 0;
template <BufferValue Value>
inline constexpr std::size_t kItemBytesOf<Buffer<Value>> = sizeof(Value);

// Where an entry of the estimate meets its array: the array's item type has the bytes the entry counts (method paper
// <tab-grid-memory>). A type that changes size without its entry stops the build here.
static_assert(kItemBytesOf<decltype(GridBuffersArrays::block_ids)> == kBlockIdBytes);
static_assert(
    kSameSizeLevelNeighbourBlocksBytes
    == 27 * kItemBytesOf<decltype(GridBuffersArrays::same_size_level_neighbour_blocks)>
);
static_assert(kBiggerAndSmallerBlocksBytes == 9 * kItemBytesOf<decltype(GridBuffersArrays::bigger_and_smaller_blocks)>);
static_assert(kItemBytesOf<decltype(GridBuffersArrays::cell_states)> == kCellStateBytes);
static_assert(kItemBytesOf<decltype(GridBuffersArrays::face_piece_kinds)> == kFacePieceKindBytes);
static_assert(kItemBytesOf<decltype(GridBuffersArrays::piece_map)> == kPieceMapBytes);
static_assert(kItemBytesOf<decltype(GridBuffersArrays::build_status)> == sizeof(GridBuildStatus));
static_assert(kItemBytesOf<decltype(GridBuilderBuffersArrays::cell_ids)> == kCellIdBytes);
static_assert(kItemBytesOf<decltype(GridBuilderBuffersArrays::sort_keys)> == kSortKeyBytes);
static_assert(kItemBytesOf<decltype(GridBuilderBuffersArrays::order)> == kOrderItemBytes);
static_assert(kItemBytesOf<decltype(GridBuilderBuffersArrays::second_order)> == kOrderItemBytes);

// Every array, in the order of the enum.
constexpr std::array kGridArrays = {
    GridArray::kBlockIds,
    GridArray::kSameSizeLevelNeighbourBlocks,
    GridArray::kBiggerAndSmallerBlocks,
    GridArray::kCellStates,
    GridArray::kFacePieceKinds,
    GridArray::kPieceMap,
    GridArray::kBuildStatus,
    GridArray::kBuilderCellIds,
    GridArray::kBuilderSortKeys,
    GridArray::kBuilderOrder,
    GridArray::kBuilderSecondOrder,
    GridArray::kBuilderOrderBuffers,
};
static_assert(static_cast<std::size_t>(GridArray::kBuilderOrderBuffers) + 1 == kGridArrays.size());

GridFailure ToInternalFault() {
  return {.error = GridError::kInternalFault};
}

// The bytes of one item of an array: what its entry's bytes must be a whole number of. For the order buffers, one
// byte: their entry is checked against the form bytes_per_item x items + fixed_bytes instead.
std::size_t ComputeItemBytes(GridArray array) {
  switch (array) {
    case GridArray::kBlockIds:
      return kBlockIdBytes;
    case GridArray::kSameSizeLevelNeighbourBlocks:
    case GridArray::kBiggerAndSmallerBlocks:
    case GridArray::kPieceMap:
      return kBlockIndexBytes;
    case GridArray::kCellStates:
      return kCellStateBytes;
    case GridArray::kFacePieceKinds:
      return kFacePieceKindBytes;
    case GridArray::kBuildStatus:
      return sizeof(GridBuildStatus);
    case GridArray::kBuilderCellIds:
      return kCellIdBytes;
    case GridArray::kBuilderSortKeys:
      return kSortKeyBytes;
    case GridArray::kBuilderOrder:
    case GridArray::kBuilderSecondOrder:
      return kOrderItemBytes;
    case GridArray::kBuilderOrderBuffers:
      break;
  }
  return 1;
}

// A room of 1..kMaxCellBlocksLimit blocks. 0 is the default of GridSettings, so it is no room at all.
bool IsRoomInRange(std::uint32_t max_cell_blocks) {
  return max_cell_blocks >= 1 && max_cell_blocks <= kMaxCellBlocksLimit;
}

// The counts of a grid that fills its room, as the estimate's entries read them.
MemoryCounts ToRoomCounts(std::uint32_t max_cell_blocks) {
  GridCounts counts = {};
  counts.cell_blocks = max_cell_blocks;
  return ToMemoryCounts(counts, 0);
}

// Bytes of the one array an entry of a one-grid estimate names: it has one copy, so this is its item count times its
// bytes each, without the rounding of its memory piece.
std::uint64_t ComputeEntryBytes(const MemoryEstimateEntry& entry, const MemoryCounts& counts) {
  return ComputeMemoryBytes(std::span<const MemoryEstimateEntry>(&entry, 1), counts);
}

// The item count of the order buffers whose bytes are byte_count, or none when byte_count is not of the form
// bytes_per_item x items + fixed_bytes. Compares before it subtracts, so nothing wraps around.
std::optional<std::uint64_t> FindOrderBuffersItemCount(RunOn run_on, std::uint64_t byte_count) {
  const OrderBuffersSize order_buffers_size = ComputeOrderBuffersSize(run_on);
  if (byte_count < order_buffers_size.fixed_bytes) {
    return std::nullopt;
  }
  const std::uint64_t per_item_bytes = byte_count - order_buffers_size.fixed_bytes;
  if (per_item_bytes % order_buffers_size.bytes_per_item != 0) {
    return std::nullopt;
  }
  return per_item_bytes / order_buffers_size.bytes_per_item;
}

// Builds `array` as one memory piece of exactly byte_count bytes. CheckGridArrayEntries has checked that the bytes
// are whole items.
template <BufferValue Value>
std::expected<void, GridFailure>
BuildArray(RunOn run_on, std::uint64_t byte_count, RunMemory& run_memory, Buffer<Value>& array) {
  std::expected<Buffer<Value>, BufferError> built =
      BuildBuffer<Value>(run_on, static_cast<std::size_t>(byte_count / sizeof(Value)), run_memory);
  if (!built.has_value()) {
    return std::unexpected(ToGridFailure(built.error(), byte_count));
  }
  array = std::move(*built);
  return {};
}

// Builds the order buffers of byte_count bytes as one memory piece.
std::expected<void, GridFailure> BuildOrderBuffersArray(
    RunOn run_on,
    std::uint64_t byte_count,
    RunMemory& run_memory,
    GridBuilderBuffersArrays& arrays
) {
  const std::optional<std::uint64_t> item_count = FindOrderBuffersItemCount(run_on, byte_count);
  if (!item_count.has_value()) {
    // CheckGridArrayEntries has checked the form; reaching here means it and this function disagree.
    return std::unexpected(ToInternalFault());
  }
  std::expected<OrderBuffers, BufferError> built =
      BuildOrderBuffers(run_on, static_cast<std::size_t>(*item_count), run_memory);
  if (!built.has_value()) {
    return std::unexpected(ToGridFailure(built.error(), byte_count));
  }
  arrays.order_buffers = std::move(*built);
  return {};
}

// Builds one builder buffers array of byte_count bytes into `arrays`. An array of GridBuffers is an internal fault:
// the builder never passes one.
std::expected<void, GridFailure> BuildBuilderBuffersArray(
    GridArray array,
    std::uint64_t byte_count,
    RunOn run_on,
    RunMemory& run_memory,
    GridBuilderBuffersArrays& arrays
) {
  switch (array) {
    case GridArray::kBuilderCellIds:
      return BuildArray(run_on, byte_count, run_memory, arrays.cell_ids);
    case GridArray::kBuilderSortKeys:
      return BuildArray(run_on, byte_count, run_memory, arrays.sort_keys);
    case GridArray::kBuilderOrder:
      return BuildArray(run_on, byte_count, run_memory, arrays.order);
    case GridArray::kBuilderSecondOrder:
      return BuildArray(run_on, byte_count, run_memory, arrays.second_order);
    case GridArray::kBuilderOrderBuffers:
      return BuildOrderBuffersArray(run_on, byte_count, run_memory, arrays);
    case GridArray::kBlockIds:
    case GridArray::kSameSizeLevelNeighbourBlocks:
    case GridArray::kBiggerAndSmallerBlocks:
    case GridArray::kCellStates:
    case GridArray::kFacePieceKinds:
    case GridArray::kPieceMap:
    case GridArray::kBuildStatus:
      break;
  }
  return std::unexpected(ToInternalFault());
}

// Builds one array of a grid's own buffers of byte_count bytes into `arrays`. A builder buffers array is an internal
// fault: the builder never passes one.
std::expected<void, GridFailure> BuildGridBuffersArray(
    GridArray array,
    std::uint64_t byte_count,
    RunOn run_on,
    RunMemory& run_memory,
    GridBuffersArrays& arrays
) {
  switch (array) {
    case GridArray::kBlockIds:
      return BuildArray(run_on, byte_count, run_memory, arrays.block_ids);
    case GridArray::kSameSizeLevelNeighbourBlocks:
      return BuildArray(run_on, byte_count, run_memory, arrays.same_size_level_neighbour_blocks);
    case GridArray::kBiggerAndSmallerBlocks:
      return BuildArray(run_on, byte_count, run_memory, arrays.bigger_and_smaller_blocks);
    case GridArray::kCellStates:
      return BuildArray(run_on, byte_count, run_memory, arrays.cell_states);
    case GridArray::kFacePieceKinds:
      return BuildArray(run_on, byte_count, run_memory, arrays.face_piece_kinds);
    case GridArray::kPieceMap:
      return BuildArray(run_on, byte_count, run_memory, arrays.piece_map);
    case GridArray::kBuildStatus:
      return BuildArray(run_on, byte_count, run_memory, arrays.build_status);
    case GridArray::kBuilderCellIds:
    case GridArray::kBuilderSortKeys:
    case GridArray::kBuilderOrder:
    case GridArray::kBuilderSecondOrder:
    case GridArray::kBuilderOrderBuffers:
      break;
  }
  return std::unexpected(ToInternalFault());
}

// Whether an entry lives where the builders take it from: kRunOn always, kCpu only on the CPU, where it shares the
// one CPU block; kDriverStack never.
bool IsEntryOnProcessor(const MemoryEstimateEntry& entry, RunOn run_on) {
  return entry.where == MemoryWhere::kRunOn || (entry.where == MemoryWhere::kCpu && run_on == RunOn::kCpu);
}

// The sum of the builder buffers entries of a one-grid estimate at max_cell_blocks.
std::uint64_t ComputeBuilderBuffersEntryBytes(RunOn run_on, std::uint32_t max_cell_blocks) {
  const MemoryCounts counts = ToRoomCounts(max_cell_blocks);
  std::uint64_t byte_sum = 0;
  for (const MemoryEstimateEntry& entry : BuildGridMemoryEstimate(run_on, 1, max_cell_blocks)) {
    const std::optional<GridArray> array = FindGridArray(entry.name);
    if (array.has_value() && ComputeIsBuilderBuffersArray(*array)) {
      byte_sum += ComputeEntryBytes(entry, counts);
    }
  }
  return byte_sum;
}

// The failure keys' steps that the build checks after the sorted list (<sec-parallel-build>, "The first failure"): the
// two checks of <proc-build-grid> step 5. Step 7 comes from ComputeBlockFaultKey.
constexpr int kCellPlacesStep = 5;  // The active and split cells against the room's cell places.
constexpr int kBlockCountStep = 6;  // The blocks against the room.

// Names the failure of a key of steps 5 to 7 with ComputeFailureCells, as the GPU names its smallest key. These steps
// name no cell, so no list is passed.
GridFailure ToGridFailureOfBuildKey(std::uint64_t failure_key) {
  const CellCheck cell_check = ComputeFailureCells(GridFrame(), nullptr, 0, nullptr, 0, failure_key);
  return {.error = cell_check.error, .cell = cell_check.cell, .other_cell = cell_check.other_cell};
}

// Says whether every table array has the size of the room, as BuildGridBuffers built it from the estimate.
bool ComputeHasTablesOfRoom(const GridBuffersArrays& arrays, std::uint32_t max_cell_blocks) {
  const std::size_t room = max_cell_blocks;
  return arrays.block_ids.size() == room && arrays.same_size_level_neighbour_blocks.size() == 27 * room
      && arrays.bigger_and_smaller_blocks.size() == 9 * room && arrays.cell_states.size() == kCellPlacesPerBlock * room
      && arrays.face_piece_kinds.size() == kFacePlacesPerBlock * room;
}

// Writes `state` at the place of each cell in the block that step 5 wrote for it: <proc-build-grid> step 7, the place
// from ToCellPlaceIndex. Returns false, at the first such cell, when a cell's block is not among the block ids.
bool ApplyStateOfCells(
    std::span<const CellId> cells,
    CellState state,
    const GridTables& tables,
    CellState* cell_states
) {
  return std::ranges::all_of(cells, [state, &tables, cell_states](CellId cell) {
    const CellBlockIndex block = FindSortedBlock(tables.block_ids, tables.block_count, ToCellBlockId(cell));
    if (block.value == kNoIndex) {
      return false;
    }
    cell_states[ToCellPlaceIndex(block, ToCellCoord(cell)).value] = state;
    return true;
  });
}

// Writes, after the active cells at the start of `cells`, the split cells each one adds: <proc-build-grid> step 5.
// `cells` has room for exactly the split cells ComputeSplitCellCountToAdd counts, which the caller summed.
void ApplySplitCellsAfterActiveCells(std::span<CellId> cells, std::uint32_t active_count) {
  std::size_t added_position = active_count;
  for (std::uint32_t position = 0; position < active_count; ++position) {
    const CellId cell = cells[position];
    const int size_level_number = static_cast<int>(ToSizeLevel(cell));
    const int added_count = ComputeSplitCellCountToAdd(cell);
    for (int added = 1; added <= added_count; ++added) {
      cells[added_position] = ComputeBigCellAtSizeLevel(cell, static_cast<SizeLevel>(size_level_number - added));
      ++added_position;
    }
  }
}

// Writes the block of every cell once into block_ids, in the fixed order: <proc-build-grid> step 5. Both lists are in
// the fixed block order (ApplyFixedBlockOrder), so taking the earlier of the two next cells visits every block in that
// order, and a block differs from the last one written exactly when it is new. The keys are the fixed order, so the
// block ids need no sort of their own. Returns the block count, or nothing when block_ids has no room for them all.
std::optional<std::uint32_t> ApplyBlockIdsOfCells(
    std::span<const CellId> active_cells,
    std::span<const CellId> split_cells,
    std::span<CellBlockId> block_ids
) {
  std::uint32_t block_count = 0;
  std::size_t active_position = 0;
  std::size_t split_position = 0;
  while (active_position < active_cells.size() || split_position < split_cells.size()) {
    const bool is_active_done = active_position == active_cells.size();
    const bool is_split_done = split_position == split_cells.size();
    const bool is_split_next = is_active_done
        || (!is_split_done && ComputeIsEarlierBlock(split_cells[split_position], active_cells[active_position]));
    const CellId cell = is_split_next ? split_cells[split_position++] : active_cells[active_position++];
    const CellBlockId block = ToCellBlockId(cell);
    const bool is_new_block = block_count == 0 || block_ids[block_count - 1].value != block.value;
    if (!is_new_block) {
      continue;
    }
    if (block_count == block_ids.size()) {
      return std::nullopt;
    }
    block_ids[block_count] = block;
    ++block_count;
  }
  return block_count;
}

}  // namespace

std::optional<GridArray> FindGridArray(std::string_view entry_name) {
  constexpr std::array<std::string_view, kGridArrays.size()> kEntryNames = {
      kBlockIdsEntryName,
      kSameSizeLevelNeighbourBlocksEntryName,
      kBiggerAndSmallerBlocksEntryName,
      kCellStatesEntryName,
      kFacePieceKindsEntryName,
      kPieceMapEntryName,
      kBuildStatusEntryName,
      kBuilderCellIdsEntryName,
      kBuilderSortKeysEntryName,
      kBuilderOrderEntryName,
      kBuilderSecondOrderEntryName,
      kBuilderOrderBuffersEntryName,
  };
  for (std::size_t index = 0; index < kEntryNames.size(); ++index) {
    if (kEntryNames[index] == entry_name) {
      return kGridArrays[index];
    }
  }
  return std::nullopt;
}

bool ComputeIsBuilderBuffersArray(GridArray array) {
  switch (array) {
    case GridArray::kBuilderCellIds:
    case GridArray::kBuilderSortKeys:
    case GridArray::kBuilderOrder:
    case GridArray::kBuilderSecondOrder:
    case GridArray::kBuilderOrderBuffers:
      return true;
    case GridArray::kBlockIds:
    case GridArray::kSameSizeLevelNeighbourBlocks:
    case GridArray::kBiggerAndSmallerBlocks:
    case GridArray::kCellStates:
    case GridArray::kFacePieceKinds:
    case GridArray::kPieceMap:
    case GridArray::kBuildStatus:
      break;
  }
  return false;
}

bool ComputeIsGridArrayOnProcessor(GridArray array, RunOn run_on) {
  if (run_on == RunOn::kGpu) {
    return true;
  }
  switch (array) {
    case GridArray::kBlockIds:
    case GridArray::kSameSizeLevelNeighbourBlocks:
    case GridArray::kBiggerAndSmallerBlocks:
    case GridArray::kCellStates:
    case GridArray::kFacePieceKinds:
    case GridArray::kPieceMap:
    case GridArray::kBuilderCellIds:
      return true;
    case GridArray::kBuildStatus:
    case GridArray::kBuilderSortKeys:
    case GridArray::kBuilderOrder:
    case GridArray::kBuilderSecondOrder:
    case GridArray::kBuilderOrderBuffers:
      break;
  }
  return false;
}

std::expected<void, GridFailure>
CheckGridArrayEntries(std::span<const MemoryEstimateEntry> estimate, RunOn run_on, std::uint32_t max_cell_blocks) {
  // Steps: 1 every entry names a known array of run_on, once, where the builders take it, in whole items  2 every
  // array of run_on has its entry
  // Step 1.
  const MemoryCounts counts = ToRoomCounts(max_cell_blocks);
  std::array<std::uint32_t, kGridArrays.size()> entry_count_per_array = {};
  for (const MemoryEstimateEntry& entry : estimate) {
    const std::optional<GridArray> array = FindGridArray(entry.name);
    if (!array.has_value() || !ComputeIsGridArrayOnProcessor(*array, run_on) || !IsEntryOnProcessor(entry, run_on)) {
      return std::unexpected(ToInternalFault());
    }
    std::uint32_t& entry_count = entry_count_per_array[static_cast<std::size_t>(*array)];
    ++entry_count;
    if (entry_count > 1) {
      return std::unexpected(ToInternalFault());
    }
    const std::uint64_t byte_count = ComputeEntryBytes(entry, counts);
    const bool is_whole_items = *array == GridArray::kBuilderOrderBuffers
        ? FindOrderBuffersItemCount(run_on, byte_count).has_value()
        : byte_count % ComputeItemBytes(*array) == 0;
    if (!is_whole_items) {
      return std::unexpected(ToInternalFault());
    }
  }
  // Step 2.
  for (const GridArray array : kGridArrays) {
    const bool has_entry = entry_count_per_array[static_cast<std::size_t>(array)] == 1;
    if (ComputeIsGridArrayOnProcessor(array, run_on) && !has_entry) {
      return std::unexpected(ToInternalFault());
    }
  }
  return {};
}

GridFailure ToGridFailure(BufferError error, std::uint64_t asked_bytes) {
  switch (error) {
    case BufferError::kOutOfMemory:
    case BufferError::kRunMemoryTooSmall:
      return {.error = GridError::kOutOfMemory, .magnitude = asked_bytes};
    case BufferError::kSizeMismatch:
      // The room limit keeps every count below what a buffer function checks: no build produces it.
      return ToInternalFault();
    case BufferError::kNone:
    case BufferError::kRunOnCallFailed:
      break;
  }
  return {.error = GridError::kRunOnCallFailed};
}

GridBuilderBuffers::GridBuilderBuffers(GridBuilderBuffers&& other) noexcept
    : arrays_(std::move(other.arrays_)),
      max_cell_blocks_(std::exchange(other.max_cell_blocks_, 0)),
      run_on_(other.run_on_),
      lent_count_(std::move(other.lent_count_)) {
  // Requires: not lent. A GridBuffers holding a loan points at `other`, which is about to hold nothing. The count
  // moves along, so a release build that breaks this still counts the loans back into live memory.
  const bool moved_from_is_not_lent = lent_count() == 0;
  assert(moved_from_is_not_lent);
  static_cast<void>(moved_from_is_not_lent);
  other.arrays_.order_buffers.reset();
}

GridBuilderBuffers& GridBuilderBuffers::operator=(GridBuilderBuffers&& other) noexcept {
  // Requires: neither side lent. Assigning onto lent builder buffers would change the room the borrowers checked.
  const bool assigned_to_is_not_lent = lent_count() == 0;
  const bool moved_from_is_not_lent = other.lent_count() == 0;
  assert(assigned_to_is_not_lent);
  assert(moved_from_is_not_lent);
  static_cast<void>(assigned_to_is_not_lent);
  static_cast<void>(moved_from_is_not_lent);
  if (this != &other) {
    arrays_ = std::move(other.arrays_);
    other.arrays_.order_buffers.reset();
    max_cell_blocks_ = std::exchange(other.max_cell_blocks_, 0);
    run_on_ = other.run_on_;
    FreeLentCount();
    lent_count_ = std::move(other.lent_count_);
  }
  return *this;
}

GridBuilderBuffers::~GridBuilderBuffers() {
  // Requires: not lent. A GridBuffers that outlives its builder buffers would point at arrays that are gone.
  const bool destroyed_is_not_lent = lent_count() == 0;
  assert(destroyed_is_not_lent);
  static_cast<void>(destroyed_is_not_lent);
  FreeLentCount();
}

void GridBuilderBuffers::FreeLentCount() {
  if (lent_count() != 0) {
    // A release build goes on after a broken Requires. The late GridBuffers will still count themselves down, so the
    // count must outlive these builder buffers: it is left behind on purpose, 8 bytes never freed, instead of a write
    // into freed memory (as RunMemory does with its live-piece count).
    const std::size_t* const count_left_behind = lent_count_.release();
    static_cast<void>(count_left_behind);
  }
  lent_count_.reset();
}

std::size_t GridBuilderBuffers::byte_count() const {
  const std::size_t order_buffers_bytes = arrays_.order_buffers.has_value() ? arrays_.order_buffers->byte_count() : 0;
  return (arrays_.cell_ids.size() * sizeof(CellId)) + (arrays_.sort_keys.size() * sizeof(std::uint64_t))
      + (arrays_.order.size() * sizeof(std::uint32_t)) + (arrays_.second_order.size() * sizeof(std::uint32_t))
      + order_buffers_bytes;
}

GridBuffers::GridBuffers(GridBuffers&& other) noexcept
    : arrays_(std::move(other.arrays_)),
      max_cell_blocks_(std::exchange(other.max_cell_blocks_, 0)),
      run_on_(other.run_on_),
      lent_count_(std::exchange(other.lent_count_, nullptr)) {
  // The loan moves with the arrays: the lent count is unchanged.
  other.arrays_.builder_buffers = nullptr;
}

GridBuffers& GridBuffers::operator=(GridBuffers&& other) noexcept {
  if (this != &other) {
    ReturnLoan();
    arrays_ = std::move(other.arrays_);
    other.arrays_.builder_buffers = nullptr;
    lent_count_ = std::exchange(other.lent_count_, nullptr);
    max_cell_blocks_ = std::exchange(other.max_cell_blocks_, 0);
    run_on_ = other.run_on_;
  }
  return *this;
}

GridBuffers::~GridBuffers() {
  ReturnLoan();
}

void GridBuffers::ReturnLoan() {
  // Through the count, not the builder buffers: the count is still there even when a bug destroyed them first.
  if (lent_count_ != nullptr) {
    --*lent_count_;
    lent_count_ = nullptr;
  }
  arrays_.builder_buffers = nullptr;
}

GridBuffersArrays& ToGridBuffersArrays(GridBuffers& buffers) {
  return buffers.arrays_;
}

const GridBuffersArrays& ToGridBuffersArrays(const GridBuffers& buffers) {
  return buffers.arrays_;
}

GridBuilderBuffersArrays& ToGridBuilderBuffersArrays(GridBuilderBuffers& builder_buffers) {
  return builder_buffers.arrays_;
}

const GridBuilderBuffersArrays& ToGridBuilderBuffersArrays(const GridBuilderBuffers& builder_buffers) {
  return builder_buffers.arrays_;
}

bool ComputeHasLoanOf(const GridBuffers& buffers, const GridBuilderBuffers& builder_buffers) {
  return buffers.lent_count_ != nullptr && buffers.lent_count_ == builder_buffers.lent_count_.get();
}

std::expected<GridBuilderBuffers, GridFailure>
BuildGridBuilderBuffers(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory) {
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(run_on, 1, max_cell_blocks);
  return BuildGridBuilderBuffersFromEntries(run_on, max_cell_blocks, estimate, run_memory);
}

std::expected<GridBuilderBuffers, GridFailure> BuildGridBuilderBuffersFromEntries(
    RunOn run_on,
    std::uint32_t max_cell_blocks,
    std::span<const MemoryEstimateEntry> estimate,
    RunMemory& run_memory
) {
  // Steps: 1 the processor  2 the room  3 the entries  4 one memory piece per builder buffers entry, in the order of
  // the estimate
  // Step 1: a processor that cannot run is reported before the block is touched, so a GPU asked for in a build
  // without GPU code fails here whatever block was passed.
  if (!CheckCanRunOn(run_on).has_value()) {
    return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
  }
  // Requires: one processor (BuildBuffer from a RunMemory asserts it too).
  assert(run_on == run_memory.run_on());

  // Step 2: a room of 0 comes from settings never filled in; one above the limit would overflow the 32-bit cell place
  // counts of the orders. The caller's plan must choose a room in range.
  if (!IsRoomInRange(max_cell_blocks)) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }

  // Step 3: an estimate and builders that disagree are a fault in FluidiTy, found before any memory piece is taken.
  if (const std::expected<void, GridFailure> checked = CheckGridArrayEntries(estimate, run_on, max_cell_blocks);
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }

  // Step 4: the memory pieces come from the estimate's own entries, so the arrays and the estimate cannot differ. On
  // failure the memory pieces already taken stay in the block; the arrays built so far are destroyed here.
  GridBuilderBuffers builder_buffers;
  builder_buffers.run_on_ = run_on;
  // The lent count lives on the heap so that it outlives the builder buffers if a caller breaks the lending rule.
  builder_buffers.lent_count_.reset(new (std::nothrow) std::size_t{0});
  if (builder_buffers.lent_count_ == nullptr) {
    return std::unexpected(GridFailure{.error = GridError::kOutOfMemory, .magnitude = sizeof(std::size_t)});
  }
  const MemoryCounts counts = ToRoomCounts(max_cell_blocks);
  for (const MemoryEstimateEntry& entry : estimate) {
    // Step 3 found every name.
    const std::optional<GridArray> array = FindGridArray(entry.name);
    if (!array.has_value()) {
      return std::unexpected(ToInternalFault());
    }
    if (!ComputeIsBuilderBuffersArray(*array)) {
      continue;
    }
    const std::expected<void, GridFailure> built =
        BuildBuilderBuffersArray(*array, ComputeEntryBytes(entry, counts), run_on, run_memory, builder_buffers.arrays_);
    if (!built.has_value()) {
      return std::unexpected(built.error());
    }
  }
  builder_buffers.max_cell_blocks_ = max_cell_blocks;
  return builder_buffers;
}

GridBuilderBuffersSize ComputeGridBuilderBuffersSize(RunOn run_on) {
  const std::uint64_t bytes_at_no_room = ComputeBuilderBuffersEntryBytes(run_on, 0);
  const std::uint64_t bytes_at_one_block = ComputeBuilderBuffersEntryBytes(run_on, 1);
  // Every entry counts at least as much at one block as at none, so the difference never needs the clamp; it is there
  // so an estimate that broke this could not wrap around.
  const std::uint64_t bytes_per_cell_block =
      bytes_at_one_block > bytes_at_no_room ? bytes_at_one_block - bytes_at_no_room : 0;
  return {
      .bytes_per_cell_block = static_cast<std::uint32_t>(bytes_per_cell_block),
      .fixed_bytes = bytes_at_no_room,
  };
}

std::expected<GridBuffers, GridFailure>
BuildGridBuffers(const GridSettings& settings, GridBuilderBuffers& builder_buffers, RunMemory& run_memory) {
  const std::vector<MemoryEstimateEntry> estimate =
      BuildGridMemoryEstimate(settings.run_on, 1, settings.max_cell_blocks);
  return BuildGridBuffersFromEntries(settings, builder_buffers, estimate, run_memory);
}

std::expected<GridBuffers, GridFailure> BuildGridBuffersFromEntries(
    const GridSettings& settings,
    GridBuilderBuffers& builder_buffers,
    std::span<const MemoryEstimateEntry> estimate,
    RunMemory& run_memory
) {
  // Steps: 1 the processor  2 the room and the lent builder buffers  3 the entries  4 one memory piece per entry of
  // the grid's own arrays, in the order of the estimate  5 the loan
  // Step 1: as BuildGridBuilderBuffers, before the block or the builder buffers are touched.
  const RunOn run_on = settings.run_on;
  if (!CheckCanRunOn(run_on).has_value()) {
    return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
  }
  // Requires: one processor for the buffers, the builder buffers and the block.
  assert(builder_buffers.run_on() == run_on);
  assert(run_memory.run_on() == run_on);

  // Step 2: a room of 0 is settings never filled in. Builder buffers with less room cannot build a grid that fills
  // these buffers, and moved-from ones have room 0: the caller builds them for at least this room. Builder buffers
  // never have more than kMaxCellBlocksLimit, so this check also stops a room above the limit, and no separate test
  // of the limit is needed here.
  if (settings.max_cell_blocks == 0) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  if (builder_buffers.max_cell_blocks() < settings.max_cell_blocks) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  // Builder buffers with room always have a lent count: the builder sets the room only after the count exists, and
  // every move carries the room and the count together. Without one this code is wrong, not the caller's input.
  if (builder_buffers.lent_count_ == nullptr) {
    return std::unexpected(ToInternalFault());
  }

  // Step 3: as BuildGridBuilderBuffers.
  if (const std::expected<void, GridFailure> checked =
          CheckGridArrayEntries(estimate, run_on, settings.max_cell_blocks);
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }

  // Step 4: as BuildGridBuilderBuffers, for the entries that are not builder buffers.
  GridBuffers buffers;
  buffers.run_on_ = run_on;
  const MemoryCounts counts = ToRoomCounts(settings.max_cell_blocks);
  for (const MemoryEstimateEntry& entry : estimate) {
    const std::optional<GridArray> array = FindGridArray(entry.name);
    if (!array.has_value()) {
      return std::unexpected(ToInternalFault());
    }
    if (ComputeIsBuilderBuffersArray(*array)) {
      continue;
    }
    const std::expected<void, GridFailure> built =
        BuildGridBuffersArray(*array, ComputeEntryBytes(entry, counts), run_on, run_memory, buffers.arrays_);
    if (!built.has_value()) {
      return std::unexpected(built.error());
    }
  }

  // Step 5: the loan is taken only on success, so a failure leaves the builder buffers as they were. Step 2 found
  // the count.
  buffers.arrays_.builder_buffers = &builder_buffers;
  buffers.lent_count_ = builder_buffers.lent_count_.get();
  ++*buffers.lent_count_;
  buffers.max_cell_blocks_ = settings.max_cell_blocks;
  return buffers;
}

StaggeredGrid::StaggeredGrid(GridBuffers&& buffers, GridFrame frame, GridVersion version, const GridCounts& counts)
    : buffers_(std::move(buffers)), frame_(frame), version_(version), counts_(counts) {
}

StaggeredGrid::StaggeredGrid(StaggeredGrid&& other) noexcept
    : buffers_(std::move(other.buffers_)),
      frame_(other.frame_),
      version_(std::exchange(other.version_, GridVersion{})),
      counts_(std::exchange(other.counts_, GridCounts{})) {
}

StaggeredGrid& StaggeredGrid::operator=(StaggeredGrid&& other) noexcept {
  if (this != &other) {
    buffers_ = std::move(other.buffers_);
    frame_ = other.frame_;
    version_ = std::exchange(other.version_, GridVersion{});
    counts_ = std::exchange(other.counts_, GridCounts{});
  }
  return *this;
}

GridTables StaggeredGrid::tables(RunOn run_on) const {
  // The tables live on the buffers' processor only; a grid moved from or taken back holds none.
  const bool holds_tables = run_on == buffers_.run_on() && version_.value != 0;
  if (!holds_tables) {
    return {};
  }
  const GridBuffersArrays& arrays = ToGridBuffersArrays(buffers_);
  const std::uint32_t* const bigger_and_smaller_blocks = arrays.bigger_and_smaller_blocks.data();
  return {
      .frame = frame_,
      .version = version_,
      .block_count = counts_.cell_blocks,
      .block_ids = arrays.block_ids.data(),
      .same_size_level_blocks = arrays.same_size_level_neighbour_blocks.data(),
      .bigger_block = bigger_and_smaller_blocks,
      .smaller_blocks = bigger_and_smaller_blocks + ComputeSmallerBlocksStart(buffers_.max_cell_blocks()),
      .cell_states = arrays.cell_states.data(),
      .face_piece_kinds = arrays.face_piece_kinds.data(),
  };
}

StaggeredGrid
BuildGridFromBuffers(GridBuffers&& buffers, GridFrame frame, GridVersion version, const GridCounts& counts) {
  // Requires: a built grid's version and a block count within the room.
  assert(version.value != 0);
  assert(counts.cell_blocks <= buffers.max_cell_blocks());
  return {std::move(buffers), frame, version, counts};
}

GridBuffers TakeBuffers(StaggeredGrid&& grid) {
  GridBuffers buffers = std::move(grid.buffers_);
  grid.version_ = GridVersion{};
  grid.counts_ = GridCounts{};
  return buffers;
}

std::expected<StaggeredGrid, GridFailure> BuildGrid(const GridInputs& inputs, GridBuffers& buffers) {
  // Steps of <proc-build-grid>: 1 to 4 the input checks, the ids and the sorted list  5 the split cells and the
  // blocks  6 the neighbour tables  7 the cell states  8 the face piece kinds
  // Steps 1 to 4: the n active cells, sorted in the fixed order at the start of the builder buffers' cell ids.
  const std::expected<std::uint32_t, GridFailure> sorted_count =
      BuildSortedCellList(inputs, buffers, SiblingCheck::kRequire);
  if (!sorted_count.has_value()) {
    return std::unexpected(sorted_count.error());
  }
  const std::uint32_t active_count = *sorted_count;
  GridBuffersArrays& arrays = ToGridBuffersArrays(buffers);
  const std::uint32_t room = buffers.max_cell_blocks();
  // BuildGridBuffers built every table for the room from the estimate's entries; other sizes mean the two disagree.
  if (!ComputeHasTablesOfRoom(arrays, room)) {
    return std::unexpected(ToInternalFault());
  }
  Buffer<CellId>& cell_ids = ToGridBuilderBuffersArrays(*arrays.builder_buffers).cell_ids;

  // Step 5: the split cells, then the blocks. Once the siblings passed, each split cell is the big cell of exactly one
  // eighth-0 small cell, so the cells that ComputeSplitCellCountToAdd counts are each added once.
  std::uint64_t split_count = 0;
  for (std::uint32_t position = 0; position < active_count; ++position) {
    split_count += static_cast<std::uint64_t>(ComputeSplitCellCountToAdd(cell_ids.data()[position]));
  }
  const std::uint64_t cell_count = active_count + split_count;
  // The active and split cells need more cell places than the room has: the caller raises the room.
  if (cell_count > std::uint64_t{kCellPlacesPerBlock} * room) {
    return std::unexpected(ToGridFailureOfBuildKey(ToFailureKey(kCellPlacesStep, 0, 0)));
  }
  // The lent builder buffers hold 64 cell ids per block of a room at least the buffers' (CheckLentBuilderBuffers).
  if (cell_ids.size() < cell_count) {
    return std::unexpected(ToInternalFault());
  }
  const std::span<CellId> cells(cell_ids.data(), static_cast<std::size_t>(cell_count));
  ApplySplitCellsAfterActiveCells(cells, active_count);
  // The two parts stay apart, so step 7 knows which cells are active and which are split.
  const std::span<CellId> active_cells = cells.first(active_count);
  const std::span<CellId> split_cells = cells.subspan(active_count);
  ApplyFixedBlockOrder(active_cells);
  ApplyFixedBlockOrder(split_cells);
  CellBlockId* const block_ids = arrays.block_ids.data();
  const std::optional<std::uint32_t> written_block_count =
      ApplyBlockIdsOfCells(active_cells, split_cells, std::span<CellBlockId>(block_ids, room));
  // More blocks than the room holds: the caller raises the room. No cell is named, as on the GPU.
  if (!written_block_count.has_value()) {
    return std::unexpected(ToGridFailureOfBuildKey(ToFailureKey(kBlockCountStep, 0, 0)));
  }
  const std::uint32_t block_count = *written_block_count;

  // Step 6: the neighbour tables of every block. Then ComputeBlockFaultKey over the blocks, step 7 of the failure key
  // (<sec-parallel-build>, "The first failure").
  std::uint32_t* const same_size_level_blocks = arrays.same_size_level_neighbour_blocks.data();
  std::uint32_t* const bigger_block = arrays.bigger_and_smaller_blocks.data();
  std::uint32_t* const smaller_blocks = bigger_block + ComputeSmallerBlocksStart(room);
  for (std::uint32_t block = 0; block < block_count; ++block) {
    FindBlockNeighbours(block_ids, block_count, {block}, same_size_level_blocks, bigger_block, smaller_blocks);
  }
  for (std::uint32_t block = 0; block < block_count; ++block) {
    const std::uint64_t fault_key = ComputeBlockFaultKey(block_ids, bigger_block, {block});
    // A block of size level 1 or more lies in a split cell whose block step 5 wrote; without it, the build is wrong.
    if (fault_key != kNoFailureKey) {
      return std::unexpected(ToGridFailureOfBuildKey(fault_key));
    }
  }

  // Step 7: every place kAbsent, the active and the split cells at their places, then the places covered by an active
  // cell one size level bigger. ComputeCoveredCellState reads only kActive marks, which this last pass never writes.
  CellState* const cell_states = arrays.cell_states.data();
  const GridTables tables = {
      .frame = inputs.frame,
      .version = inputs.version,
      .block_count = block_count,
      .block_ids = block_ids,
      .same_size_level_blocks = same_size_level_blocks,
      .bigger_block = bigger_block,
      .smaller_blocks = smaller_blocks,
      .cell_states = cell_states,
      .face_piece_kinds = arrays.face_piece_kinds.data(),
  };
  const std::size_t cell_place_count = std::size_t{kCellPlacesPerBlock} * block_count;
  std::fill_n(cell_states, cell_place_count, CellState::kAbsent);
  // Step 5 wrote the block of every cell, so FindSortedBlock finds each; a miss is a fault in the build.
  const bool are_cells_placed = ApplyStateOfCells(active_cells, CellState::kActive, tables, cell_states)
      && ApplyStateOfCells(split_cells, CellState::kSplit, tables, cell_states);
  if (!are_cells_placed) {
    return std::unexpected(ToInternalFault());
  }
  for (std::size_t place = 0; place < cell_place_count; ++place) {
    if (cell_states[place] == CellState::kAbsent) {
      cell_states[place] = ComputeCoveredCellState(tables, {static_cast<std::uint32_t>(place)});
    }
  }

  // Step 8: the kind of every face place, from the two cell places beside it.
  GridCounts counts{};
  FacePieceKind* const face_piece_kinds = arrays.face_piece_kinds.data();
  const std::size_t face_place_count = std::size_t{kFacePlacesPerBlock} * block_count;
  for (std::size_t face = 0; face < face_place_count; ++face) {
    const FacePieceKind kind = ComputeFacePieceKind(tables, {static_cast<std::uint32_t>(face)});
    face_piece_kinds[face] = kind;
    if (kind != FacePieceKind::kNotAPiece) {
      ++counts.face_pieces;
    }
  }

  counts.cell_blocks = block_count;
  counts.active_cells = active_count;
  for (const CellId cell : active_cells) {
    ++counts.active_cells_per_size_level[static_cast<std::size_t>(ToSizeLevel(cell))];
  }
  return BuildGridFromBuffers(std::move(buffers), inputs.frame, inputs.version, counts);
}

}  // namespace fluidity::grid
