#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/cell_block_range_internal.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

#if defined(FLUIDITY_HAS_GPU)
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/cell_block_range_gpu.h"
#endif

namespace fluidity::grid {
namespace {

// Checks the sorted list of a call: ascending with no repeat, and every index below the grid's block count. A failed
// list is bad input because the walk's binary search needs the order and no block beyond the grid can be asked.
std::expected<void, GridFailure> CheckSortedBlocks(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks) {
  bool is_ascending = true;
  for (std::size_t entry = 1; entry < blocks.size(); ++entry) {
    is_ascending = is_ascending && blocks[entry - 1].value < blocks[entry].value;
  }
  const bool is_below_count = blocks.empty() || blocks.back().value < grid.counts().cell_blocks;
  if (!is_ascending || !is_below_count) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  return {};
}

// Runs the readers of `width` forward from every cell place and face place kept in `block`.
void ApplyMarksFromBlock(const GridTables& tables, std::uint32_t block, BorderWidth width, std::uint8_t* marks) {
  for (std::uint32_t place_in_block = 0; place_in_block < static_cast<std::uint32_t>(kCellPlacesPerBlock);
       ++place_in_block) {
    const CellPlaceIndex cell = {.value = (block * static_cast<std::uint32_t>(kCellPlacesPerBlock)) + place_in_block};
    ApplyCellPlaceReadMarks(tables, cell, width, marks);
  }
  for (std::uint32_t place_in_block = 0; place_in_block < static_cast<std::uint32_t>(kFacePlacesPerBlock);
       ++place_in_block) {
    const FacePlaceIndex face = {.value = (block * static_cast<std::uint32_t>(kFacePlacesPerBlock)) + place_in_block};
    ApplyFacePlaceReadMarks(tables, face, width, marks);
  }
}

// Finds the border of `set` at `width` by running the readers forward from every place kept in the set, then one pass
// over the marks in ascending index, which lists the marked blocks outside the set. Allocates one mark byte per block
// of the grid, freed on return.
// Work: for each place of the set, one marking function (border_block_steps.h), whose work does not depend on the
// grid's size; then two passes over the marks, one to count and one to list, with one set test for each marked block.
// The set's size drives the marking, so the cost does not grow with the grid times its size levels.
std::vector<CellBlockIndex> FindBorderByMarking(const GridTables& tables, const BlockSet& set, BorderWidth width) {
  std::vector<std::uint8_t> marks(tables.block_count, 0);
  for (std::uint32_t entry = 0; entry < set.range.block_count; ++entry) {
    ApplyMarksFromBlock(tables, set.range.first_block + entry, width, marks.data());
  }
  for (std::uint32_t entry = 0; entry < set.sorted_count; ++entry) {
    ApplyMarksFromBlock(tables, set.sorted_blocks[entry].value, width, marks.data());
  }
  // The list is sized once, so a call allocates the marks and, when the border is not empty, the list: two times.
  std::size_t border_block_count = 0;
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    border_block_count += marks[block] != 0 && !ComputeIsInBlockSet(set, CellBlockIndex{block}) ? 1U : 0U;
  }
  std::vector<CellBlockIndex> border_blocks;
  border_blocks.reserve(border_block_count);
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    if (marks[block] != 0 && !ComputeIsInBlockSet(set, CellBlockIndex{block})) {
      border_blocks.push_back(CellBlockIndex{block});
    }
  }
  return border_blocks;
}

// Checks the entries of the border blocks that a call of the buffer form was given: ascending with no repeat, each
// below the grid's block count and none in the range. A block of the range cannot be border so far, since the list is
// what lies outside it. Each entry asks the rule that the GPU pass asks with one thread for each entry.
std::expected<void, GridFailure>
CheckBorderSoFar(const StaggeredGrid& grid, CellBlockRange range, std::span<const CellBlockIndex> border_so_far) {
  const std::uint32_t block_count = grid.counts().cell_blocks;
  for (std::uint32_t entry = 0; entry < border_so_far.size(); ++entry) {
    if (ComputeBorderSoFarFailureKey(range, border_so_far.data(), block_count, entry) != kNoFailureKey) {
      return std::unexpected(GridFailure{.error = GridError::kBadInput});
    }
  }
  return {};
}

// Builds one array of the border search buffers as one memory piece of exactly byte_count bytes, which must be the
// bytes of `item_count` items: an entry that says otherwise is an internal fault. An array already built is one too,
// for an estimate that lists it twice.
template <BufferValue Value>
std::expected<void, GridFailure>
BuildBorderArray(std::uint64_t byte_count, std::uint64_t item_count, RunMemory& run_memory, Buffer<Value>& array) {
  if (byte_count != item_count * sizeof(Value) || array.size() != 0) {
    return std::unexpected(GridFailure{.error = GridError::kInternalFault});
  }
  std::expected<Buffer<Value>, BufferError> built =
      BuildBuffer<Value>(run_memory.run_on(), static_cast<std::size_t>(item_count), run_memory);
  if (!built.has_value()) {
    return std::unexpected(ToGridFailure(built.error(), byte_count));
  }
  array = std::move(*built);
  return {};
}

// clang-tidy: a count of bytes and a count of items; the swappable-parameters check flags the two integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
// Builds the order buffers of the prefix sum over `item_count` items as one memory piece of exactly byte_count bytes.
std::expected<void, GridFailure> BuildBorderOrderBuffers(
    std::uint64_t byte_count,
    std::uint64_t item_count,
    RunMemory& run_memory,
    std::optional<OrderBuffers>& order_buffers
) {
  const OrderBuffersSize size = ComputeOrderBuffersSize(run_memory.run_on());
  const std::uint64_t expected_bytes = (std::uint64_t{size.bytes_per_item} * item_count) + size.fixed_bytes;
  if (byte_count != expected_bytes || order_buffers.has_value()) {
    return std::unexpected(GridFailure{.error = GridError::kInternalFault});
  }
  std::expected<OrderBuffers, BufferError> built =
      BuildOrderBuffers(run_memory.run_on(), static_cast<std::size_t>(item_count), run_memory);
  if (!built.has_value()) {
    return std::unexpected(ToGridFailure(built.error(), byte_count));
  }
  order_buffers = std::move(*built);
  return {};
}
// NOLINTEND(bugprone-easily-swappable-parameters)

#if defined(FLUIDITY_HAS_GPU)

// What the GPU pass of one call works on: the pointers are GPU memory of the grid's processor.
struct GpuBorderSearch {
  GridTables tables{};
  CellBlockRange range{};
  const CellBlockIndex* border_so_far = nullptr;
  std::uint32_t border_so_far_count = 0;
  BorderWidth width = BorderWidth::kTouchingCells;
  std::uint32_t block_count = 0;
  std::uint8_t* marks = nullptr;
  Buffer<std::uint32_t>* sums = nullptr;  // block_count + 1 words.
  OrderBuffers* order_buffers = nullptr;
  Buffer<GridBuildStatus>* status = nullptr;
  Buffer<CellBlockIndex>* border_blocks = nullptr;
};

// The failure of a GPU-half call that did not start: kRunOnCallFailed, through the mapping of base/.
GridFailure ToCallFailure(RunOnCallStatus call_status) {
  return ToGridFailure(ToBufferError(call_status), 0);
}

// Runs the GPU pass over the marks of `search` and returns the count of the list, or the failure.
// Steps: 1 reset the status and the marks  2 check the border so far  3 mark from every place of the set  4 write the
// listed marks  5 prefix sum  6 write the list and the count  7 read the status
// Each step is one or more launches, in the one stream, so each sees what the step before wrote and the marking
// launch has finished before the scan of the marks starts. Only the status record comes back.
std::expected<std::uint32_t, GridFailure> FindBorderBlocksOnGpu(const GpuBorderSearch& search) {
  GridBuildStatus* const status = search.status->data();
  std::uint32_t* const sums = search.sums->data();
  // Step 1.
  if (const RunOnCallStatus call_status = ApplyBorderStatusResetOnGpu(status); call_status != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(call_status));
  }
  if (const RunOnCallStatus call_status = ResetBorderMarksOnGpu(search.marks, search.block_count);
      call_status != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(call_status));
  }
  // Step 2.
  if (const RunOnCallStatus call_status = CheckBorderSoFarOnGpu(
          search.range,
          search.border_so_far,
          search.border_so_far_count,
          search.block_count,
          status
      );
      call_status != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(call_status));
  }
  // Step 3: one launch for each chunk of the set, each waited for. The threads stop at once when step 2 failed. A chunk
  // that fails ends the call here: the marks of the earlier chunks give no list, and the next call clears them.
  // The caller checked ComputeIsSetWithinPlaceLimit, so the sum fits 32 bits.
  const auto set_block_count =
      static_cast<std::uint32_t>(std::uint64_t{search.range.block_count} + search.border_so_far_count);
  const std::uint32_t chunk_count = ComputeMarkingChunkCount(set_block_count, kSetBlocksPerMarkingLaunch);
  for (std::uint32_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    if (const RunOnCallStatus call_status = WriteBorderReadMarksOnGpu(
            search.tables,
            search.range,
            search.border_so_far,
            search.border_so_far_count,
            ComputeMarkingChunk(set_block_count, chunk_index, kSetBlocksPerMarkingLaunch),
            search.width,
            status,
            search.marks
        );
        call_status != RunOnCallStatus::kDone) {
      return std::unexpected(ToCallFailure(call_status));
    }
  }
  // Step 4: after a failure of step 2 every word is 0, so the prefix sum adds no word of an earlier call.
  if (const RunOnCallStatus call_status =
          WriteListedMarksOnGpu(search.marks, search.range, search.block_count, status, sums);
      call_status != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(call_status));
  }
  // Step 5: in place, the total one past the last block. The sizes were checked (the room, the buffers built for it),
  // so an error here is a size error that no input causes, or a failed GPU call, which the mapping tells apart.
  if (const std::expected<void, BufferError> summed =
          ComputeExclusivePrefixSum(*search.sums, search.block_count, *search.sums, *search.order_buffers);
      !summed.has_value()) {
    // Steps 1 to 4 may still be running when the prefix sum's first launch fails.
    static_cast<void>(WaitForBorderSearchOnGpu());
    return std::unexpected(ToGridFailure(summed.error(), 0));
  }
  // Step 6.
  if (const RunOnCallStatus call_status = WriteBorderBlocksOnGpu(
          sums,
          search.block_count,
          static_cast<std::uint32_t>(search.border_blocks->size()),
          search.border_blocks->data(),
          status
      );
      call_status != RunOnCallStatus::kDone) {
    return std::unexpected(ToCallFailure(call_status));
  }
  // Step 7: the count, or the failure of step 2, is all that comes back.
  GridBuildStatus read_status{};
  if (const std::expected<void, BufferError> read =
          ReadBuffer(*search.status, 0, std::span<GridBuildStatus>(&read_status, 1));
      !read.has_value()) {
    return std::unexpected(ToGridFailure(read.error(), sizeof(GridBuildStatus)));
  }
  return ToBorderBlockCount(read_status, search.block_count);
}

#endif

}  // namespace

std::expected<std::uint32_t, GridFailure> ToBorderBlockCount(const GridBuildStatus& status, std::uint32_t block_count) {
  if (status.failure_key != kNoFailureKey) {
    // The id step checks the border so far, which the caller made. A key of another step contradicts the steps that
    // wrote the record: no input gives one.
    const bool is_bad_border = ToFailureStep(status.failure_key) == kIdStep;
    return std::unexpected(GridFailure{.error = is_bad_border ? GridError::kBadInput : GridError::kInternalFault});
  }
  // The total is a count of blocks of the grid: a record that says otherwise contradicts the steps that wrote it.
  if (status.result_count > block_count) {
    return std::unexpected(GridFailure{.error = GridError::kInternalFault});
  }
  return status.result_count;
}

std::expected<std::vector<CellBlockRange>, GridFailure>
BuildRanges(const StaggeredGrid& grid, std::uint32_t max_blocks_per_range) {
  // A cut into ranges of no block would never get to the last block: the caller asked for no cut at all.
  if (max_blocks_per_range == 0) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  std::vector<CellBlockRange> ranges;
  const std::uint32_t block_count = grid.counts().cell_blocks;
  for (std::uint32_t first_block = 0; first_block < block_count; first_block += max_blocks_per_range) {
    const std::uint32_t blocks_left = block_count - first_block;
    ranges.push_back({.first_block = first_block, .block_count = std::min(max_blocks_per_range, blocks_left)});
  }
  return ranges;
}

RangeWithBorder BuildRangeWithBorder(const StaggeredGrid& grid, CellBlockRange range) {
  assert(grid.run_on() == RunOn::kCpu);
  assert(range.first_block <= grid.counts().cell_blocks);
  assert(range.block_count <= grid.counts().cell_blocks - range.first_block);
  const BlockSet set = {.range = range, .sorted_blocks = nullptr, .sorted_count = 0};
  return {
      .range = range,
      .border_blocks = FindBorderByMarking(grid.tables(RunOn::kCpu), set, BorderWidth::kTouchingCells),
  };
}

std::expected<std::vector<CellBlockIndex>, GridFailure>
FindBorderBlocks(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width) {
  const std::expected<void, GridFailure> check = CheckSortedBlocks(grid, blocks);
  if (!check.has_value()) {
    return std::unexpected(check.error());
  }
  const BlockSet set = {
      .range = {.first_block = 0, .block_count = 0},
      .sorted_blocks = blocks.data(),
      .sorted_count = static_cast<std::uint32_t>(blocks.size()),
  };
  return FindBorderByMarking(grid.tables(RunOn::kCpu), set, width);
}

std::expected<BorderSearchBuffers, GridFailure>
BuildBorderSearchBuffers(RunOn run_on, std::uint32_t max_cell_blocks, RunMemory& run_memory) {
  // Steps: 1 the processor  2 the room  3 one memory piece per entry of the estimate, in its order  4 every array
  //   5 on the GPU, the stack of the marking function
  // Step 1: a processor that cannot run is reported before the block is touched.
  if (!CheckCanRunOn(run_on).has_value()) {
    return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
  }
  // Requires: one processor (BuildBuffer from a RunMemory asserts it too).
  assert(run_on == run_memory.run_on());

  // Step 2: a room of 0 is a limit never filled in; one above the limit would overflow the 32-bit counts of the lists.
  if (max_cell_blocks == 0 || max_cell_blocks > kMaxCellBlocksLimit) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }

  // Step 3: the pieces come from the estimate's own entries, so the arrays and the estimate cannot differ. On failure
  // the pieces already taken stay in the block, and the arrays built so far are destroyed here.
  BorderSearchBuffers search_buffers;
  search_buffers.run_on_ = run_on;
  const std::uint64_t block_count = max_cell_blocks;
  for (const MemoryEstimateEntry& entry : BuildBorderSearchMemoryEstimate(run_on, max_cell_blocks)) {
    const std::uint64_t byte_count = entry.given_count * entry.bytes_each;
    std::expected<void, GridFailure> built = std::unexpected(GridFailure{.error = GridError::kInternalFault});
    if (entry.name == kBorderMarksEntryName) {
      built = BuildBorderArray(byte_count, block_count, run_memory, search_buffers.marks_);
    } else if (entry.name == kBorderSumsEntryName) {
      built = BuildBorderArray(byte_count, block_count + 1, run_memory, search_buffers.sums_);
    } else if (entry.name == kBorderOrderBuffersEntryName) {
      built = BuildBorderOrderBuffers(byte_count, block_count + 1, run_memory, search_buffers.order_buffers_);
    } else if (entry.name == kBorderStatusEntryName) {
      built = BuildBorderArray(byte_count, 1, run_memory, search_buffers.status_);
    }
    if (!built.has_value()) {
      return std::unexpected(built.error());
    }
  }

  // Step 4: an estimate that left an array out is a fault in FluidiTy, not an input.
  const bool has_marks = search_buffers.marks_.size() == block_count;
  const bool has_gpu_arrays = search_buffers.sums_.size() == block_count + 1
      && search_buffers.order_buffers_.has_value() && search_buffers.status_.size() == 1;
  if (!has_marks || (run_on == RunOn::kGpu && !has_gpu_arrays)) {
    return std::unexpected(GridFailure{.error = GridError::kInternalFault});
  }

#if defined(FLUIDITY_HAS_GPU)
  // Step 5: the marking function's stack, taken now so that a GPU too full for it fails here, with the bytes it asked
  // for, rather than in the first search as an empty list.
  if (run_on == RunOn::kGpu) {
    std::uint64_t bytes_per_thread = 0;
    std::uint64_t resident_threads = 0;
    if (ComputeLargestBorderSearchStackOnGpu(&bytes_per_thread) != RunOnCallStatus::kDone
        || ComputeResidentThreadsOnGpu(&resident_threads) != RunOnCallStatus::kDone) {
      return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
    }
    const RunOnCallStatus reserved = ReserveBorderSearchStackOnGpu(bytes_per_thread);
    if (reserved == RunOnCallStatus::kOutOfMemory) {
      return std::unexpected(
          GridFailure{.error = GridError::kOutOfMemory, .magnitude = bytes_per_thread * resident_threads}
      );
    }
    if (reserved != RunOnCallStatus::kDone) {
      return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
    }
  }
#endif
  search_buffers.max_cell_blocks_ = max_cell_blocks;
  return search_buffers;
}

std::expected<MemoryEstimateEntry, GridFailure> BuildBorderSearchStackEstimate(RunOn run_on) {
  MemoryEstimateEntry entry = {
      .name = "grid: GPU stack of the border search",
      .counted_per = CountedPer::kGivenCount,
      .given_count = 0,
      .bytes_each = 0,
      .copies = 1,
      .kept_for = KeptFor::kWholeRun,
      .where = MemoryWhere::kDriverStack,
  };
  if (run_on == RunOn::kCpu) {
    return entry;
  }
  if (!CheckCanRunOn(run_on).has_value()) {
    return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
  }
#if defined(FLUIDITY_HAS_GPU)
  std::uint64_t bytes_per_thread = 0;
  std::uint64_t resident_threads = 0;
  if (ComputeLargestBorderSearchStackOnGpu(&bytes_per_thread) != RunOnCallStatus::kDone
      || ComputeResidentThreadsOnGpu(&resident_threads) != RunOnCallStatus::kDone) {
    return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
  }
  entry.given_count = resident_threads;
  entry.bytes_each = static_cast<std::uint32_t>(bytes_per_thread);
  return entry;
#else
  // Without GPU code CheckCanRunOn never accepts the GPU, so no caller reaches this.
  return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
#endif
}

std::expected<std::uint32_t, GridFailure> FindBorderBlocks(
    const StaggeredGrid& grid,
    CellBlockRange range,
    const Buffer<CellBlockIndex>& border_so_far,
    std::uint32_t border_so_far_count,
    BorderWidth width,
    BorderSearchBuffers& search_buffers,
    Buffer<CellBlockIndex>& border_blocks
) {
  // Steps: 1 the shape of the arguments  2 the room and the set's places  3 the border so far  4 the marks  5 the list
  // Requires: every buffer on the processor of the grid, which the caller chose when it built them.
  assert(border_so_far.run_on() == grid.run_on());
  assert(border_blocks.run_on() == grid.run_on());
  assert(search_buffers.run_on() == grid.run_on());

  // Step 1: the search writes border_blocks while it reads the border so far, so one buffer for both would change
  // the set it searches. The two buffers are different objects, or two pieces, which never overlap.
  const std::uint32_t block_count = grid.counts().cell_blocks;
  const bool is_range_inside_grid =
      range.first_block <= block_count && range.block_count <= block_count - range.first_block;
  const bool is_same_buffer = &border_blocks == &border_so_far
      || (border_blocks.data() != nullptr && border_blocks.data() == border_so_far.data());
  if (is_same_buffer || border_so_far_count > border_so_far.size() || !is_range_inside_grid) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }

  // Step 2: marks for fewer blocks than the grid has would be written past. The GPU pass cannot tell a bad border so
  // far before it has cleared the marks, so the room is checked first on both processors, and a call that fails both
  // ways gives the same error on both.
  if (search_buffers.max_cell_blocks() < block_count) {
    return std::unexpected(GridFailure{.error = GridError::kTooManyCellBlocks});
  }

  // Step 2 (b): the places of the set are 32-bit numbers (ApplySetPlaceReadMarks): a set of more blocks than
  // kMaxSetBlocks would wrap them onto low places and miss border blocks. Both processors ask the same function, before
  // any mark is written. A valid call has a range of at most the grid's blocks, so only a long border so far gets here.
  if (!ComputeIsSetWithinPlaceLimit(range.block_count, border_so_far_count)) {
    return std::unexpected(GridFailure{.error = GridError::kTooManyCellBlocks});
  }

  // Step 3: the entries must be ascending blocks outside the range, or the list would not hold them in order. A GPU
  // grid runs the whole pass from here, steps 3 to 5 in its own launches: one thread for each entry asks the rule that
  // the CPU form below asks of each entry in turn.
  if (grid.run_on() == RunOn::kGpu) {
#if defined(FLUIDITY_HAS_GPU)
    return FindBorderBlocksOnGpu({
        .tables = grid.tables(RunOn::kGpu),
        .range = range,
        .border_so_far = border_so_far.data(),
        .border_so_far_count = border_so_far_count,
        .width = width,
        .block_count = block_count,
        .marks = search_buffers.marks_.data(),
        .sums = &search_buffers.sums_,
        .order_buffers = &*search_buffers.order_buffers_,
        .status = &search_buffers.status_,
        .border_blocks = &border_blocks,
    });
#else
    // Without GPU code no GPU grid or GPU search buffers can be built, so no caller reaches this.
    return std::unexpected(GridFailure{.error = GridError::kRunOnCallFailed});
#endif
  }
  const std::span<const CellBlockIndex> border_so_far_entries(border_so_far.data(), border_so_far_count);
  if (const std::expected<void, GridFailure> checked = CheckBorderSoFar(grid, range, border_so_far_entries);
      !checked.has_value()) {
    return std::unexpected(checked.error());
  }

  // Step 4: the marks start at 0 for every block, also for the blocks of the set, so that they hold only what this
  // call's places read: a mark left from an earlier call would add a block to the list silently. A marking function
  // reads no mark, so no call depends on the order of the marks. The blocks of the set are marked through their own
  // cell places.
  const GridTables tables = grid.tables(RunOn::kCpu);
  std::uint8_t* const marks = search_buffers.marks_.data();
  std::fill_n(marks, block_count, std::uint8_t{0});
  const std::uint64_t set_block_count = std::uint64_t{range.block_count} + border_so_far_count;
  const std::uint64_t set_place_count = set_block_count * kPlacesPerSetBlock;
  for (std::uint64_t set_place = 0; set_place < set_place_count; ++set_place) {
    ApplySetPlaceReadMarks(
        tables,
        range,
        border_so_far.data(),
        border_so_far_count,
        static_cast<std::uint32_t>(set_place),
        width,
        marks
    );
  }

  // Step 5: the marked blocks outside the range, in index order. The first border_blocks.size() are written.
  std::uint32_t border_block_count = 0;
  for (std::uint32_t block = 0; block < block_count; ++block) {
    if (ComputeListedMark(marks, range, block) == 0) {
      continue;
    }
    if (border_block_count < border_blocks.size()) {
      border_blocks.data()[border_block_count] = CellBlockIndex{block};
    }
    ++border_block_count;
  }
  return border_block_count;
}

}  // namespace fluidity::grid
