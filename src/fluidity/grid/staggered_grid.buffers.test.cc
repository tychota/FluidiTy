#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_error_message.h"
#include "fluidity/grid/grid_memory_estimate.h"

namespace fluidity::grid {
namespace {

// The counts of a grid of cell_blocks blocks, as a memory estimate reads them.
MemoryCounts BuildCounts(std::uint32_t cell_blocks) {
  GridCounts counts = {};
  counts.cell_blocks = cell_blocks;
  return ToMemoryCounts(counts, 0);
}

// Bytes a CPU run reserves for `estimate`: its kRunOn and its kCpu entries both live in CPU memory.
std::size_t ComputeCpuRunMemoryBytes(const std::vector<MemoryEstimateEntry>& estimate, std::uint32_t max_cell_blocks) {
  const MemoryCounts counts = BuildCounts(max_cell_blocks);
  return ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kRunOn)
      + ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kCpu);
}

// A CPU RunMemory of exactly the CPU estimate of grid_count grids at max_cell_blocks.
std::expected<RunMemory, BufferError> BuildCpuRunMemoryOfEstimate(
    std::uint32_t max_cell_blocks,
    std::uint32_t grid_count = 1,
    BuilderBuffersSharing builder_buffers_sharing = BuilderBuffersSharing::kOnePerGrid
) {
  const std::vector<MemoryEstimateEntry> estimate =
      BuildGridMemoryEstimate(RunOn::kCpu, grid_count, max_cell_blocks, builder_buffers_sharing);
  return BuildRunMemory(RunOn::kCpu, ComputeCpuRunMemoryBytes(estimate, max_cell_blocks));
}

// The estimate without the entry named `name`.
std::vector<MemoryEstimateEntry>
BuildEstimateWithout(std::vector<MemoryEstimateEntry> estimate, std::string_view name) {
  std::erase_if(estimate, [name](const MemoryEstimateEntry& entry) {
    return entry.name == name;
  });
  return estimate;
}

// An entry no builder knows, as a later change to the estimate could add one.
MemoryEstimateEntry BuildUnknownEntry() {
  return {
      .name = "grid: entry no builder knows",
      .counted_per = CountedPer::kCellBlock,
      .given_count = 0,
      .bytes_each = 4,
      .copies = 1,
      .kept_for = KeptFor::kWholeRun,
      .where = MemoryWhere::kRunOn,
  };
}

// Bytes of the values of one buffer.
template <typename Value>
std::uint64_t ComputeBufferBytes(const Buffer<Value>& buffer) {
  return buffer.size() * sizeof(Value);
}

// Bytes of the arrays of one set of grid buffers, the lent builder buffers left out.
std::uint64_t ComputeGridBuffersBytes(const GridBuffers& buffers) {
  const GridBuffersArrays& arrays = ToGridBuffersArrays(buffers);
  return ComputeBufferBytes(arrays.block_ids) + ComputeBufferBytes(arrays.same_size_level_neighbour_blocks)
      + ComputeBufferBytes(arrays.bigger_and_smaller_blocks) + ComputeBufferBytes(arrays.cell_states)
      + ComputeBufferBytes(arrays.face_piece_kinds) + ComputeBufferBytes(arrays.piece_map)
      + ComputeBufferBytes(arrays.build_status);
}

// One array of a CPU grid: its bytes and its first byte, named as its estimate entry.
struct NamedArray {
  std::string_view name;
  std::uint64_t bytes = 0;
  const void* first_byte = nullptr;
};

template <typename Value>
NamedArray ToNamedArray(std::string_view name, const Buffer<Value>& buffer) {
  return {.name = name, .bytes = ComputeBufferBytes(buffer), .first_byte = buffer.data()};
}

// The arrays of a CPU grid in the order of the CPU estimate's entries: the five tables, the builder buffers' cell ids,
// the piece map.
std::vector<NamedArray> BuildCpuArrays(const GridBuilderBuffers& builder_buffers, const GridBuffers& buffers) {
  const GridBuffersArrays& arrays = ToGridBuffersArrays(buffers);
  const GridBuilderBuffersArrays& builder_arrays = ToGridBuilderBuffersArrays(builder_buffers);
  return {
      ToNamedArray("grid: block ids", arrays.block_ids),
      ToNamedArray("grid: same-size-level neighbour blocks", arrays.same_size_level_neighbour_blocks),
      ToNamedArray("grid: bigger and smaller blocks", arrays.bigger_and_smaller_blocks),
      ToNamedArray("grid: cell states", arrays.cell_states),
      ToNamedArray("grid: face piece kinds", arrays.face_piece_kinds),
      ToNamedArray("grid: builder buffers: cell ids", builder_arrays.cell_ids),
      ToNamedArray("grid: piece map", arrays.piece_map),
  };
}

// Checks that the arrays are the estimate's entries, name for name and byte for byte, and that their memory pieces
// lie in the block in the estimate's order.
void CheckArraysEqualEntries(
    const std::vector<NamedArray>& arrays,
    const std::vector<MemoryEstimateEntry>& estimate,
    const MemoryCounts& counts
) {
  ASSERT_EQ(arrays.size(), estimate.size());
  for (std::size_t index = 0; index < estimate.size(); ++index) {
    EXPECT_EQ(arrays[index].name, estimate[index].name);
    const std::span<const MemoryEstimateEntry> entry(&estimate[index], 1);
    EXPECT_EQ(arrays[index].bytes, ComputeMemoryBytes(entry, counts)) << estimate[index].name;
  }
  // The builder buffers come first in the block, as the caller builds them first; the rest follow the estimate.
  // Cell ids are entry 5, so the order in the block is entries 5, 0, 1, 2, 3, 4, 6.
  const std::vector<std::size_t> order_in_block = {5, 0, 1, 2, 3, 4, 6};
  for (std::size_t place = 1; place < order_in_block.size(); ++place) {
    const NamedArray& before = arrays[order_in_block[place - 1]];
    const NamedArray& after = arrays[order_in_block[place]];
    EXPECT_TRUE(std::less<const void*>{}(before.first_byte, after.first_byte)) << before.name << ", " << after.name;
  }
}

// Builds one CPU grid of max_cell_blocks blocks with its builder buffers in a CPU RunMemory of exactly its estimate,
// and checks each array against its entry, then the whole block.
void CheckCpuArraysEqualTheEstimate(std::uint32_t max_cell_blocks) {
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, max_cell_blocks);
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(max_cell_blocks);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, max_cell_blocks, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const GridSettings settings = {.max_cell_blocks = max_cell_blocks, .run_on = RunOn::kCpu};
  const std::expected<GridBuffers, GridFailure> buffers = BuildGridBuffers(settings, *builder_buffers, *run_memory);
  ASSERT_TRUE(buffers.has_value());

  CheckArraysEqualEntries(BuildCpuArrays(*builder_buffers, *buffers), estimate, BuildCounts(max_cell_blocks));
  EXPECT_EQ(run_memory->handed_out_bytes(), run_memory->reserved_bytes());
}

TEST(BuildGridBuffers, CpuBuffers_HoldTheEstimatedBytes) {
  // A CPU grid takes 456 bytes of tables, 4 of piece map and 512 of builder cell ids per block: 972.
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1000);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = 1000, .run_on = RunOn::kCpu}, *builder_buffers, *run_memory);
  ASSERT_TRUE(buffers.has_value());
  EXPECT_EQ(buffers->max_cell_blocks(), 1000U);
  EXPECT_EQ(buffers->run_on(), RunOn::kCpu);
  EXPECT_EQ(builder_buffers->byte_count() + ComputeGridBuffersBytes(*buffers), 1000U * 972U);
}

TEST(BuildGridBuffers, CpuArrays_HoldThePaperItemCounts) {
  // Method paper <tab-grid-memory>, per block, divided by the bytes of one item: block ids 8 / 8 = 1, same-size-level
  // neighbour blocks 108 / 4 = 27, bigger and smaller blocks 36 / 4 = 9, cell states 64 / 1 = 64, face piece kinds
  // 240 / 1 = 240, piece map 4 / 4 = 1, builder cell ids 512 / 8 = 64. At 1,000 blocks, a thousand times each.
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1000);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = 1000, .run_on = RunOn::kCpu}, *builder_buffers, *run_memory);
  ASSERT_TRUE(buffers.has_value());
  const GridBuffersArrays& arrays = ToGridBuffersArrays(*buffers);
  EXPECT_EQ(arrays.block_ids.size(), 1000U);
  EXPECT_EQ(arrays.same_size_level_neighbour_blocks.size(), 27000U);
  EXPECT_EQ(arrays.bigger_and_smaller_blocks.size(), 9000U);
  EXPECT_EQ(arrays.cell_states.size(), 64000U);
  EXPECT_EQ(arrays.face_piece_kinds.size(), 240000U);
  EXPECT_EQ(arrays.piece_map.size(), 1000U);
  EXPECT_EQ(arrays.build_status.size(), 0U);
  const GridBuilderBuffersArrays& builder_arrays = ToGridBuilderBuffersArrays(*builder_buffers);
  EXPECT_EQ(builder_arrays.cell_ids.size(), 64000U);
  EXPECT_EQ(builder_arrays.sort_keys.size(), 0U);
  EXPECT_EQ(builder_arrays.order.size(), 0U);
  EXPECT_EQ(builder_arrays.second_order.size(), 0U);
  EXPECT_FALSE(builder_arrays.order_buffers.has_value());
}

TEST(BuildGridBuffers, ThreeRooms_TakeOneMemoryPiecePerEntryInOrder) {
  // One block, where every array is far below its 256-byte rounding; 1,000 blocks; 50,000 blocks (48.6 MB).
  CheckCpuArraysEqualTheEstimate(1);
  CheckCpuArraysEqualTheEstimate(1000);
  CheckCpuArraysEqualTheEstimate(50000);
}

TEST(BuildGridBuffers, TwoBuffersShareBuilderBuffers_HoldTheSharedEstimate) {
  const std::vector<MemoryEstimateEntry> estimate =
      BuildGridMemoryEstimate(RunOn::kCpu, 2, 1000, BuilderBuffersSharing::kOneForAllGrids);
  std::expected<RunMemory, BufferError> run_memory =
      BuildCpuRunMemoryOfEstimate(1000, 2, BuilderBuffersSharing::kOneForAllGrids);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const GridSettings settings = {.max_cell_blocks = 1000, .run_on = RunOn::kCpu};
  const std::expected<GridBuffers, GridFailure> committed = BuildGridBuffers(settings, *builder_buffers, *run_memory);
  ASSERT_TRUE(committed.has_value());
  const std::expected<GridBuffers, GridFailure> try_buffers = BuildGridBuffers(settings, *builder_buffers, *run_memory);
  ASSERT_TRUE(try_buffers.has_value());

  // 2 x 460 + 512 = 1,432 bytes per block.
  const std::uint64_t held_bytes =
      builder_buffers->byte_count() + ComputeGridBuffersBytes(*committed) + ComputeGridBuffersBytes(*try_buffers);
  EXPECT_EQ(held_bytes, ComputeMemoryBytes(estimate, BuildCounts(1000)));
  EXPECT_EQ(ToGridBuffersArrays(*committed).builder_buffers, &*builder_buffers);
  EXPECT_EQ(ToGridBuffersArrays(*try_buffers).builder_buffers, &*builder_buffers);
  EXPECT_EQ(builder_buffers->lent_count(), 2U);
}

// Builds grid_count CPU grid buffers of 1,000 blocks and their builder buffers, shared as `builder_buffers_sharing`
// says, in a CPU RunMemory of exactly their estimate, and checks that the block is used to the byte.
void CheckBuildersHandOutTheEstimate(BuilderBuffersSharing builder_buffers_sharing, std::uint32_t grid_count) {
  std::expected<RunMemory, BufferError> run_memory =
      BuildCpuRunMemoryOfEstimate(1000, grid_count, builder_buffers_sharing);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  std::vector<GridBuffers> grid_buffers;
  for (std::uint32_t grid = 0; grid < grid_count; ++grid) {
    std::expected<GridBuffers, GridFailure> buffers =
        BuildGridBuffers({.max_cell_blocks = 1000, .run_on = RunOn::kCpu}, *builder_buffers, *run_memory);
    ASSERT_TRUE(buffers.has_value());
    grid_buffers.push_back(std::move(*buffers));
  }
  EXPECT_EQ(run_memory->handed_out_bytes(), run_memory->reserved_bytes()) << grid_count;
  EXPECT_EQ(builder_buffers->lent_count(), grid_count);
}

TEST(BuildGridBuffers, IntoRunMemory_HandsOutExactlyTheEstimate) {
  CheckBuildersHandOutTheEstimate(BuilderBuffersSharing::kOnePerGrid, 1);
  CheckBuildersHandOutTheEstimate(BuilderBuffersSharing::kOneForAllGrids, 2);
}

TEST(BuildGridBuffers, RunMemoryOneMemoryPieceShort_IsOutOfMemory) {
  // The last memory piece is the piece map: 4 x 1,000 = 4,000 bytes, rounded to 4,096. A block of the estimate less
  // 256 bytes leaves only 3,840 for it, so it fails, and every memory piece before it stays taken.
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, 1000);
  std::expected<RunMemory, BufferError> run_memory =
      BuildRunMemory(RunOn::kCpu, ComputeCpuRunMemoryBytes(estimate, 1000) - 256);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = 1000, .run_on = RunOn::kCpu}, *builder_buffers, *run_memory);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error().error, GridError::kOutOfMemory);
  EXPECT_EQ(buffers.error().magnitude, 4000U);
  EXPECT_EQ(run_memory->handed_out_bytes(), run_memory->reserved_bytes() - 3840U);
  EXPECT_EQ(builder_buffers->lent_count(), 0U);
}

TEST(BuildGridBuilderBuffers, RunMemoryWithoutCellIds_IsOutOfMemory) {
  // A block of the 1,000-block estimate without its cell ids entry: the other entries take 460,544 bytes, fewer than
  // the cell ids' 8 x 64 x 1,000 = 512,000, the first memory piece. Nothing is taken.
  const std::vector<MemoryEstimateEntry> estimate =
      BuildEstimateWithout(BuildGridMemoryEstimate(RunOn::kCpu, 1, 1000), kBuilderCellIdsEntryName);
  std::expected<RunMemory, BufferError> run_memory =
      BuildRunMemory(RunOn::kCpu, ComputeCpuRunMemoryBytes(estimate, 1000));
  ASSERT_TRUE(run_memory.has_value());
  EXPECT_EQ(run_memory->reserved_bytes(), 460544U);
  const std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_FALSE(builder_buffers.has_value());
  EXPECT_EQ(builder_buffers.error().error, GridError::kOutOfMemory);
  EXPECT_EQ(builder_buffers.error().magnitude, 512000U);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

TEST(BuildGridBuffers, LentBuilderBuffersTooSmall_IsBadInput) {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1000);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 500, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const std::size_t handed_out_bytes = run_memory->handed_out_bytes();
  const std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = 1000, .run_on = RunOn::kCpu}, *builder_buffers, *run_memory);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error().error, GridError::kBadInput);
  EXPECT_EQ(run_memory->handed_out_bytes(), handed_out_bytes);
  // The lent builder buffers are untouched: 8 x 64 x 500 bytes of cell ids, and no loan.
  EXPECT_EQ(builder_buffers->max_cell_blocks(), 500U);
  EXPECT_EQ(builder_buffers->byte_count(), 256000U);
  EXPECT_EQ(builder_buffers->lent_count(), 0U);
}

TEST(BuildGridBuilderBuffers, ZeroBlocks_IsBadInput) {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 0, *run_memory);
  ASSERT_FALSE(builder_buffers.has_value());
  EXPECT_EQ(builder_buffers.error().error, GridError::kBadInput);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

TEST(BuildGridBuilderBuffers, AboveLimit_IsBadInput) {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, kMaxCellBlocksLimit + 1, *run_memory);
  ASSERT_FALSE(builder_buffers.has_value());
  EXPECT_EQ(builder_buffers.error().error, GridError::kBadInput);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

TEST(BuildGridBuffers, ZeroBlocks_IsBadInput) {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers = BuildGridBuilderBuffers(RunOn::kCpu, 1, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const std::size_t handed_out_bytes = run_memory->handed_out_bytes();
  const GridSettings no_room;
  const std::expected<GridBuffers, GridFailure> buffers = BuildGridBuffers(no_room, *builder_buffers, *run_memory);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error().error, GridError::kBadInput);
  EXPECT_EQ(run_memory->handed_out_bytes(), handed_out_bytes);
}

TEST(BuildGridBuffers, AboveLimit_IsBadInput) {
  // Builder buffers never have room above the limit, so the check of the lent room is the one that fails here.
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers = BuildGridBuilderBuffers(RunOn::kCpu, 1, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const std::size_t handed_out_bytes = run_memory->handed_out_bytes();
  const std::expected<GridBuffers, GridFailure> buffers = BuildGridBuffers(
      {.max_cell_blocks = kMaxCellBlocksLimit + 1, .run_on = RunOn::kCpu},
      *builder_buffers,
      *run_memory
  );
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error().error, GridError::kBadInput);
  EXPECT_EQ(run_memory->handed_out_bytes(), handed_out_bytes);
}

TEST(CheckGridArrayEntries, CpuAndGpuEstimates_NameEveryArrayOnce) {
  // The check the builders run first, on the estimate of each processor: the GPU estimate is checked here on a
  // machine without a GPU.
  for (const RunOn run_on : {RunOn::kCpu, RunOn::kGpu}) {
    for (const std::uint32_t max_cell_blocks : {1U, 1000U, kMaxCellBlocksLimit}) {
      const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(run_on, 1, max_cell_blocks);
      EXPECT_TRUE(CheckGridArrayEntries(estimate, run_on, max_cell_blocks).has_value()) << max_cell_blocks;
    }
  }
}

TEST(CheckGridArrayEntries, UnknownEntry_IsInternalFault) {
  std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kGpu, 1, 1000);
  estimate.push_back(BuildUnknownEntry());
  const std::expected<void, GridFailure> checked = CheckGridArrayEntries(estimate, RunOn::kGpu, 1000);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, GridError::kInternalFault);
}

TEST(CheckGridArrayEntries, ArrayWithoutEntry_IsInternalFault) {
  const std::vector<MemoryEstimateEntry> estimate =
      BuildEstimateWithout(BuildGridMemoryEstimate(RunOn::kGpu, 1, 1000), kBuildStatusEntryName);
  const std::expected<void, GridFailure> checked = CheckGridArrayEntries(estimate, RunOn::kGpu, 1000);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, GridError::kInternalFault);
}

TEST(CheckGridArrayEntries, EntryTwice_IsInternalFault) {
  std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, 1000);
  estimate.push_back(estimate.front());
  const std::expected<void, GridFailure> checked = CheckGridArrayEntries(estimate, RunOn::kCpu, 1000);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, GridError::kInternalFault);
}

TEST(CheckGridArrayEntries, GpuEntriesOnTheCpu_AreInternalFault) {
  // The GPU estimate has sort keys, orders and a status record, which the CPU builders do not build.
  const std::expected<void, GridFailure> checked =
      CheckGridArrayEntries(BuildGridMemoryEstimate(RunOn::kGpu, 1, 1000), RunOn::kCpu, 1000);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, GridError::kInternalFault);
}

TEST(CheckGridArrayEntries, BytesThatAreNotWholeItems_AreInternalFault) {
  // 4 bytes per block of 8-byte block ids: half an item.
  std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, 1);
  estimate.front().bytes_each = 4;
  const std::expected<void, GridFailure> checked = CheckGridArrayEntries(estimate, RunOn::kCpu, 1);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, GridError::kInternalFault);
}

TEST(BuildGridBuilderBuffers, EstimateWithoutCellIds_IsInternalFault) {
  // The private form with the estimate given is how a test reaches the fault: no public input can.
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1000);
  ASSERT_TRUE(run_memory.has_value());
  const std::vector<MemoryEstimateEntry> estimate =
      BuildEstimateWithout(BuildGridMemoryEstimate(RunOn::kCpu, 1, 1000), kBuilderCellIdsEntryName);
  const std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffersFromEntries(RunOn::kCpu, 1000, estimate, *run_memory);
  ASSERT_FALSE(builder_buffers.has_value());
  EXPECT_EQ(builder_buffers.error().error, GridError::kInternalFault);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

TEST(BuildGridBuffers, EstimateWithAnUnknownEntry_IsInternalFault) {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1000);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const std::size_t handed_out_bytes = run_memory->handed_out_bytes();
  std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, 1000);
  estimate.insert(estimate.begin() + 1, BuildUnknownEntry());
  const std::expected<GridBuffers, GridFailure> buffers = BuildGridBuffersFromEntries(
      {.max_cell_blocks = 1000, .run_on = RunOn::kCpu},
      *builder_buffers,
      estimate,
      *run_memory
  );
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error().error, GridError::kInternalFault);
  EXPECT_EQ(run_memory->handed_out_bytes(), handed_out_bytes);
  EXPECT_EQ(builder_buffers->lent_count(), 0U);
}

TEST(ToGridFailure, SizeMismatch_IsInternalFault) {
  // No grid build produces it: the room limit keeps every count a buffer function checks in range.
  EXPECT_EQ(ToGridFailure(BufferError::kSizeMismatch, 0).error, GridError::kInternalFault);
}

TEST(ToGridFailure, RunMemoryTooSmall_IsOutOfMemoryWithTheBytesAsked) {
  const GridFailure failure = ToGridFailure(BufferError::kRunMemoryTooSmall, 4000);
  EXPECT_EQ(failure.error, GridError::kOutOfMemory);
  EXPECT_EQ(failure.magnitude, 4000U);
}

TEST(ComputeGridBuilderBuffersSize, Cpu_EqualsByteCount) {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1000);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const GridBuilderBuffersSize size = ComputeGridBuilderBuffersSize(RunOn::kCpu);
  // The CPU builder buffers are the cell ids alone: 8 x 64 = 512 bytes per block, nothing fixed.
  EXPECT_EQ(builder_buffers->byte_count(), (std::uint64_t{1000} * size.bytes_per_cell_block) + size.fixed_bytes);
  EXPECT_EQ(builder_buffers->byte_count(), 512000U);
}

TEST(ComputeGridBuilderBuffersSize, EitherProcessor_EqualsBuilderEntries) {
  for (const RunOn run_on : {RunOn::kCpu, RunOn::kGpu}) {
    std::uint64_t builder_entry_bytes = 0;
    for (const MemoryEstimateEntry& entry : BuildGridMemoryEstimate(run_on, 1, 1000)) {
      if (entry.name.starts_with("grid: builder buffers: ")) {
        builder_entry_bytes += ComputeMemoryBytes(std::span<const MemoryEstimateEntry>(&entry, 1), BuildCounts(1000));
      }
    }
    const GridBuilderBuffersSize size = ComputeGridBuilderBuffersSize(run_on);
    EXPECT_EQ((std::uint64_t{1000} * size.bytes_per_cell_block) + size.fixed_bytes, builder_entry_bytes);
  }
}

TEST(ComputeGridBuilderBuffersSize, Gpu_IsBelowTheMemoryPieces) {
  // At 1,000 blocks the GPU builder buffers hold 1,000 x 1,920 + 4,110 = 1,924,110 bytes in five memory pieces, each
  // rounded up to 256: 512,000 + 512,000 + 256,256 + 256,256 + 388,352 = 1,924,864 bytes to reserve.
  const GridBuilderBuffersSize size = ComputeGridBuilderBuffersSize(RunOn::kGpu);
  EXPECT_EQ((std::uint64_t{1000} * size.bytes_per_cell_block) + size.fixed_bytes, 1924110U);
  std::vector<MemoryEstimateEntry> builder_entries;
  for (const MemoryEstimateEntry& entry : BuildGridMemoryEstimate(RunOn::kGpu, 1, 1000)) {
    if (entry.name.starts_with("grid: builder buffers: ")) {
      builder_entries.push_back(entry);
    }
  }
  EXPECT_EQ(ComputeRunMemoryBytes(builder_entries, BuildCounts(1000), MemoryWhere::kRunOn), 1924864U);
}

TEST(GridBuffers, MovedFrom_HasNoRoom) {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1000);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = 1000, .run_on = RunOn::kCpu}, *builder_buffers, *run_memory);
  ASSERT_TRUE(buffers.has_value());
  const GridBuffers moved_to = std::move(*buffers);
  EXPECT_EQ(moved_to.max_cell_blocks(), 1000U);
  // NOLINTNEXTLINE(bugprone-use-after-move): a moved-from GridBuffers is specified to have room 0.
  EXPECT_EQ(buffers->max_cell_blocks(), 0U);
  EXPECT_EQ(ComputeGridBuffersBytes(*buffers), 0U);
  EXPECT_EQ(ToGridBuffersArrays(*buffers).builder_buffers, nullptr);
  // The loan moved with the arrays.
  EXPECT_EQ(builder_buffers->lent_count(), 1U);
}

TEST(GridBuilderBuffers, MovedFrom_HoldsNoBytes) {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1000);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const GridBuilderBuffers moved_to = std::move(*builder_buffers);
  EXPECT_EQ(moved_to.byte_count(), 512000U);
  // NOLINTNEXTLINE(bugprone-use-after-move): moved-from builder buffers are specified to hold 0 bytes.
  EXPECT_EQ(builder_buffers->byte_count(), 0U);
  EXPECT_EQ(builder_buffers->max_cell_blocks(), 0U);
}

TEST(GridBuilderBuffers, LoansEnded_CanBeMovedAndAssigned) {
  // The allowed sequence: lend twice, end each loan by destroying, by assigning over, and by moving the grid buffers,
  // then move the builder buffers. The sanitizers of a checked build see every step.
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1000, 2);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const GridSettings settings = {.max_cell_blocks = 1000, .run_on = RunOn::kCpu};
  std::optional<GridBuffers> committed;
  std::optional<GridBuffers> try_buffers;
  {
    std::expected<GridBuffers, GridFailure> built = BuildGridBuffers(settings, *builder_buffers, *run_memory);
    ASSERT_TRUE(built.has_value());
    committed.emplace(std::move(*built));
  }
  {
    std::expected<GridBuffers, GridFailure> built = BuildGridBuffers(settings, *builder_buffers, *run_memory);
    ASSERT_TRUE(built.has_value());
    try_buffers.emplace(std::move(*built));
  }
  EXPECT_EQ(builder_buffers->lent_count(), 2U);
  // Assigning the try buffers over the committed ones ends the committed loan; the try loan moves.
  *committed = std::move(*try_buffers);
  EXPECT_EQ(builder_buffers->lent_count(), 1U);
  try_buffers.reset();
  EXPECT_EQ(builder_buffers->lent_count(), 1U);
  committed.reset();
  EXPECT_EQ(builder_buffers->lent_count(), 0U);
  const GridBuilderBuffers moved_to = std::move(*builder_buffers);
  EXPECT_EQ(moved_to.lent_count(), 0U);
  EXPECT_EQ(moved_to.max_cell_blocks(), 1000U);
}

TEST(GridBuilderBuffers, MovedThenLent_CountOnTheNewOwner) {
  // The count moves with the arrays: lending the builder buffers after a move counts on the object that holds them
  // now, and assigning other builder buffers over them once the loan has ended frees the old count. The sanitizers of
  // a checked build see every step.
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1, 2);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> built = BuildGridBuilderBuffers(RunOn::kCpu, 1, *run_memory);
  ASSERT_TRUE(built.has_value());
  GridBuilderBuffers builder_buffers = std::move(*built);
  // NOLINTNEXTLINE(bugprone-use-after-move): moved-from builder buffers are specified to have no loans.
  EXPECT_EQ(built->lent_count(), 0U);
  {
    const std::expected<GridBuffers, GridFailure> buffers =
        BuildGridBuffers({.max_cell_blocks = 1, .run_on = RunOn::kCpu}, builder_buffers, *run_memory);
    ASSERT_TRUE(buffers.has_value());
    EXPECT_EQ(builder_buffers.lent_count(), 1U);
  }
  EXPECT_EQ(builder_buffers.lent_count(), 0U);
  std::expected<GridBuilderBuffers, GridFailure> other = BuildGridBuilderBuffers(RunOn::kCpu, 1, *run_memory);
  ASSERT_TRUE(other.has_value());
  builder_buffers = std::move(*other);
  EXPECT_EQ(builder_buffers.lent_count(), 0U);
  const std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = 1, .run_on = RunOn::kCpu}, builder_buffers, *run_memory);
  ASSERT_TRUE(buffers.has_value());
  EXPECT_EQ(builder_buffers.lent_count(), 1U);
}

TEST(GridBuilderBuffers, AssignedIntoMovedFrom_LendWithTheAssignedCount) {
  // How a caller rebuilds builder buffers it moved out of: the assignment brings the count with the arrays, so the
  // loan counts on it. Without the count the lending would be an internal fault.
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1, 2);
  ASSERT_TRUE(run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> built = BuildGridBuilderBuffers(RunOn::kCpu, 1, *run_memory);
  ASSERT_TRUE(built.has_value());
  const GridBuilderBuffers moved_to = std::move(*built);
  std::expected<GridBuilderBuffers, GridFailure> other = BuildGridBuilderBuffers(RunOn::kCpu, 1, *run_memory);
  ASSERT_TRUE(other.has_value());
  // NOLINTNEXTLINE(bugprone-use-after-move): assigning into moved-from builder buffers is the point of the test.
  *built = std::move(*other);
  const std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = 1, .run_on = RunOn::kCpu}, *built, *run_memory);
  ASSERT_TRUE(buffers.has_value()) << ToName(buffers.error().error);
  EXPECT_EQ(built->lent_count(), 1U);
  EXPECT_EQ(moved_to.lent_count(), 0U);
}

#if !defined(NDEBUG)
// What each death test runs. A braced statement inside EXPECT_DEATH cannot hold a comma outside parentheses, so
// each one is its own function. A failed build returns early, and EXPECT_DEATH then fails for want of a stop.

// Builds CPU builder buffers of one block into `builder_buffers` and lends them to new grid buffers.
std::optional<GridBuffers>
BuildLentGridBuffers(RunMemory& run_memory, std::optional<GridBuilderBuffers>& builder_buffers) {
  std::expected<GridBuilderBuffers, GridFailure> built = BuildGridBuilderBuffers(RunOn::kCpu, 1, run_memory);
  if (!built.has_value()) {
    return std::nullopt;
  }
  builder_buffers.emplace(std::move(*built));
  std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = 1, .run_on = RunOn::kCpu}, *builder_buffers, run_memory);
  if (!buffers.has_value()) {
    return std::nullopt;
  }
  return std::move(*buffers);
}

void ApplyMoveOfLentBuilderBuffers() {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1);
  if (!run_memory.has_value()) {
    return;
  }
  std::optional<GridBuilderBuffers> builder_buffers;
  const std::optional<GridBuffers> buffers = BuildLentGridBuffers(*run_memory, builder_buffers);
  if (!buffers.has_value() || !builder_buffers.has_value()) {
    return;
  }
  const GridBuilderBuffers moved_to = std::move(*builder_buffers);
}

void ApplyAssignmentOntoLentBuilderBuffers() {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1, 2);
  if (!run_memory.has_value()) {
    return;
  }
  std::optional<GridBuilderBuffers> builder_buffers;
  const std::optional<GridBuffers> buffers = BuildLentGridBuffers(*run_memory, builder_buffers);
  std::expected<GridBuilderBuffers, GridFailure> other = BuildGridBuilderBuffers(RunOn::kCpu, 1, *run_memory);
  if (!buffers.has_value() || !builder_buffers.has_value() || !other.has_value()) {
    return;
  }
  *builder_buffers = std::move(*other);
}

void ApplyDestructionOfLentBuilderBuffers() {
  std::expected<RunMemory, BufferError> run_memory = BuildCpuRunMemoryOfEstimate(1);
  if (!run_memory.has_value()) {
    return;
  }
  std::optional<GridBuilderBuffers> builder_buffers;
  const std::optional<GridBuffers> buffers = BuildLentGridBuffers(*run_memory, builder_buffers);
  if (!buffers.has_value()) {
    return;
  }
  builder_buffers.reset();
}
#endif

// In a release build `assert` compiles to nothing, so each death test reports itself skipped there.

TEST(GridBuilderBuffersDeathTest, MovedWhileLent_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(ApplyMoveOfLentBuilderBuffers(), "moved_from_is_not_lent");
#endif
}

TEST(GridBuilderBuffersDeathTest, AssignedToWhileLent_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(ApplyAssignmentOntoLentBuilderBuffers(), "assigned_to_is_not_lent");
#endif
}

TEST(GridBuilderBuffersDeathTest, DestroyedWhileLent_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(ApplyDestructionOfLentBuilderBuffers(), "destroyed_is_not_lent");
#endif
}

}  // namespace
}  // namespace fluidity::grid
