// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Finds the borders of ranges on the GPU with the buffer form of FindBorderBlocks (one thread for each place of the
// set, then one thread for each block) and expects the CPU's list and count, entry for entry, by bits. Every grid is
// built on both processors from one list. A comparison that fails names the layer, the field and both values. The lists
// and counts read back go to WriteResultDigest, so the repeat run shows the GPU gave the same bytes twice. The marks
// are plain bytes that threads write at once, so the tests that run the same search twice, with the buffers reused, are
// the ones that would show a result that depends on thread timing.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/processor_description.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/cell_block_range_gpu.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

// The random grids searched, and the blocks of a range of each kind of grid.
constexpr int kGridCount = 10;
constexpr std::uint32_t kRangeBlocks = 4;
constexpr std::uint32_t kNestedBallRangeBlocks = 16;

// One layer of a stacked border as a comparison sees it: the count and the entries that were written.
struct BorderLayerResult {
  std::uint32_t count = 0;
  std::vector<std::uint32_t> blocks;
};

void ApplyWordsOf(const BorderLayerResult& result, ResultWords& words) {
  words.Add("count", result.count);
  std::size_t entry = 0;
  for (const std::uint32_t block : result.blocks) {
    words.AddEntry("blocks", entry++, block);
  }
}

BorderLayerResult ToLayerResult(std::span<const CellBlockIndex> layer) {
  BorderLayerResult result;
  result.count = static_cast<std::uint32_t>(layer.size());
  for (const CellBlockIndex block : layer) {
    result.blocks.push_back(block.value);
  }
  return result;
}

// Finds every layer of the stacked border of `range` on the grid of `rig`, each on the list of the layer before.
std::vector<BorderLayerResult>
FindStackedLayers(BorderSearchRig& rig, CellBlockRange range, std::span<const BorderWidth> layer_widths) {
  std::vector<BorderLayerResult> layers;
  std::vector<CellBlockIndex> border_so_far;
  for (const BorderWidth width : layer_widths) {
    border_so_far = FindLayerOrFail(rig, range, border_so_far, width);
    layers.push_back(ToLayerResult(border_so_far));
  }
  return layers;
}

// Finds the layers of every range on both processors, compares them by bits, and digests the GPU's.
void CheckLayersOfRanges(
    const GridOnBothProcessors& grid,
    std::span<const CellBlockRange> ranges,
    std::span<const BorderWidth> layer_widths,
    const std::string& name
) {
  std::expected<BorderSearchRig, GridFailure> on_cpu = BuildBorderSearchRig(grid.on_cpu.grid);
  std::expected<BorderSearchRig, GridFailure> on_gpu = BuildBorderSearchRig(grid.on_gpu.grid);
  ASSERT_TRUE(on_cpu.has_value());
  ASSERT_TRUE(on_gpu.has_value());
  std::vector<BorderLayerResult> cpu_layers;
  std::vector<BorderLayerResult> gpu_layers;
  for (const CellBlockRange range : ranges) {
    std::vector<BorderLayerResult> cpu_range_layers = FindStackedLayers(*on_cpu, range, layer_widths);
    std::vector<BorderLayerResult> gpu_range_layers = FindStackedLayers(*on_gpu, range, layer_widths);
    std::ranges::move(cpu_range_layers, std::back_inserter(cpu_layers));
    std::ranges::move(gpu_range_layers, std::back_inserter(gpu_layers));
  }
  CheckSameResults(
      std::span<const BorderLayerResult>(cpu_layers),
      std::span<const BorderLayerResult>(gpu_layers),
      name
  );
  WriteDigestOfResults("border search on the GPU, " + name, std::span<const BorderLayerResult>(gpu_layers));
}

// The random grids, the tested ranges of each (first, middle, last, whole grid), four layers.
TEST(FindBorderBlocks, OnGpuRandomGrids_SameBitsAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  for (std::size_t grid = 0; grid < grids.size(); ++grid) {
    SCOPED_TRACE(grid);
    const std::vector<CellBlockRange> ranges = BuildTestedRanges(grids[grid].on_cpu.grid, kRangeBlocks);
    CheckLayersOfRanges(grids[grid], ranges, kLayerWidths, "random grid " + std::to_string(grid));
  }
}

// The nested-ball grid, ranges of 16 blocks: one layer at each width, then two stacked layers (touching, four widths).
TEST(FindBorderBlocks, OnGpuNestedBallRanges_SameBitsAsCpu) {
  const std::optional<GridOnBothProcessors> grid = BuildGridOnBothProcessors(ToUnitFrame(4), BuildNestedBallCells());
  ASSERT_TRUE(grid.has_value());
  const std::vector<CellBlockRange> ranges = BuildTestedRanges(grid->on_cpu.grid, kNestedBallRangeBlocks);
  const std::array<BorderWidth, 1> touching = {BorderWidth::kTouchingCells};
  const std::array<BorderWidth, 1> four_widths = {BorderWidth::kFourCellWidths};
  CheckLayersOfRanges(*grid, ranges, touching, "nested ball, touching");
  CheckLayersOfRanges(*grid, ranges, four_widths, "nested ball, four widths");
  CheckLayersOfRanges(*grid, ranges, kNestedBallLayerWidths, "nested ball, stacked");
}

// The count and the entries written by one call whose output buffer holds only `room` entries.
BorderLayerResult FindWithShortOutput(const StaggeredGrid& grid, CellBlockRange range, std::size_t room) {
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(grid);
  EXPECT_TRUE(rig.has_value());
  std::expected<Buffer<CellBlockIndex>, BufferError> short_buffer = BuildBuffer<CellBlockIndex>(grid.run_on(), room);
  EXPECT_TRUE(short_buffer.has_value());
  if (!rig.has_value() || !short_buffer.has_value()) {
    return {};
  }
  const std::expected<std::uint32_t, GridFailure> count = FindBorderBlocks(
      grid,
      range,
      rig->so_far_buffer,
      0,
      BorderWidth::kFourCellWidths,
      rig->search_buffers,
      *short_buffer
  );
  EXPECT_TRUE(count.has_value());
  std::vector<CellBlockIndex> written(room);
  EXPECT_TRUE(ReadBuffer<CellBlockIndex>(*short_buffer, 0, written).has_value());
  BorderLayerResult result = ToLayerResult(written);
  result.count = count.value_or(0);
  return result;
}

// An output buffer of 3 entries: the count is the whole border's, and the entries are the first three.
TEST(FindBorderBlocks, OnGpuShortOutput_SameCountAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  std::vector<BorderLayerResult> on_cpu;
  std::vector<BorderLayerResult> on_gpu;
  for (const GridOnBothProcessors& grid : grids) {
    const CellBlockRange range = {.first_block = 0, .block_count = 2};
    on_cpu.push_back(FindWithShortOutput(grid.on_cpu.grid, range, 3));
    on_gpu.push_back(FindWithShortOutput(grid.on_gpu.grid, range, 3));
  }
  CheckSameResults(
      std::span<const BorderLayerResult>(on_cpu),
      std::span<const BorderLayerResult>(on_gpu),
      "short output"
  );
  EXPECT_GT(on_cpu.front().count, 3U) << "the border must be longer than the output, or the case shows nothing";
  WriteDigestOfResults("border search on the GPU, short output", std::span<const BorderLayerResult>(on_gpu));
}

// The error of one call with a bad border so far, or 0 when it succeeded.
GridError FindErrorOfBadList(const StaggeredGrid& grid, std::span<const CellBlockIndex> bad_list, bool is_same_buffer) {
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(grid);
  EXPECT_TRUE(rig.has_value());
  if (!rig.has_value() || !WriteBuffer<CellBlockIndex>(bad_list, rig->so_far_buffer, 0).has_value()) {
    return GridError::kInternalFault;
  }
  const CellBlockRange range = {.first_block = 2, .block_count = 2};
  Buffer<CellBlockIndex>& output = is_same_buffer ? rig->so_far_buffer : rig->border_buffer;
  const std::expected<std::uint32_t, GridFailure> count = FindBorderBlocks(
      grid,
      range,
      rig->so_far_buffer,
      static_cast<std::uint32_t>(bad_list.size()),
      BorderWidth::kTouchingCells,
      rig->search_buffers,
      output
  );
  return count.has_value() ? GridError::kNone : count.error().error;
}

// Expects the GPU grid and the CPU grid to fail with kBadInput on `bad_list`, given as the border so far.
void CheckBadListFailsOnBothProcessors(const GridOnBothProcessors& grid, std::span<const CellBlockIndex> bad_list) {
  const GridError on_cpu = FindErrorOfBadList(grid.on_cpu.grid, bad_list, false);
  const GridError on_gpu = FindErrorOfBadList(grid.on_gpu.grid, bad_list, false);
  EXPECT_EQ(on_cpu, GridError::kBadInput);
  EXPECT_EQ(on_gpu, on_cpu);
}

// A border so far that is not ascending, holds a block of the range, or holds a block beyond the grid, and one buffer
// given as both input and output: the GPU's error is the CPU's.
TEST(FindBorderBlocks, OnGpuFailures_EqualCpuFailures) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(1);
  ASSERT_EQ(grids.size(), 1U);
  const std::uint32_t block_count = grids.front().on_cpu.grid.counts().cell_blocks;
  ASSERT_GE(block_count, 8U);
  const std::vector<std::vector<CellBlockIndex>> bad_lists = {
      {CellBlockIndex{5}, CellBlockIndex{4}},
      {CellBlockIndex{1}, CellBlockIndex{3}},
      {CellBlockIndex{1}, CellBlockIndex{block_count}},
      {CellBlockIndex{5}, CellBlockIndex{5}},
  };
  for (const std::vector<CellBlockIndex>& bad_list : bad_lists) {
    SCOPED_TRACE(bad_list.back().value);
    CheckBadListFailsOnBothProcessors(grids.front(), bad_list);
  }
  const std::array<CellBlockIndex, 1> good_list = {CellBlockIndex{5}};
  EXPECT_EQ(FindErrorOfBadList(grids.front().on_cpu.grid, good_list, true), GridError::kBadInput);
  EXPECT_EQ(FindErrorOfBadList(grids.front().on_gpu.grid, good_list, true), GridError::kBadInput);
}

// Expects one call on the GPU to allocate nothing on the CPU: the calls are counted apart.
void CheckCallAllocatesNothing(BorderSearchRig& rig, CellBlockRange range, BorderWidth width) {
  const std::uint64_t before = ComputeAllocationCount();
  const std::expected<std::uint32_t, GridFailure> count =
      FindBorderBlocks(rig.grid, range, rig.so_far_buffer, 0, width, rig.search_buffers, rig.border_buffer);
  const std::uint64_t after = ComputeAllocationCount();
  EXPECT_TRUE(count.has_value());
  EXPECT_EQ(after - before, 0U) << range.first_block;
}

// Each call on the GPU allocates nothing on the CPU: the rig is built before the counted calls.
TEST(FindBorderBlocks, OnGpu_AllocatesNothing) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(1);
  ASSERT_EQ(grids.size(), 1U);
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(grids.front().on_gpu.grid);
  ASSERT_TRUE(rig.has_value());
  const std::array<CellBlockRange, 3> ranges = {
      CellBlockRange{.first_block = 0, .block_count = 1},
      CellBlockRange{.first_block = 3, .block_count = 2},
      CellBlockRange{.first_block = 0, .block_count = rig->grid.counts().cell_blocks},
  };
  for (const CellBlockRange range : ranges) {
    CheckCallAllocatesNothing(*rig, range, BorderWidth::kTouchingCells);
    CheckCallAllocatesNothing(*rig, range, BorderWidth::kFourCellWidths);
  }
}

// The layers of every tested range of `grid`, on the buffers of `rig`.
std::vector<BorderLayerResult> FindLayersOfTestedRanges(BorderSearchRig& rig) {
  std::vector<BorderLayerResult> layers;
  for (const CellBlockRange range : BuildTestedRanges(rig.grid, kRangeBlocks)) {
    std::ranges::move(FindStackedLayers(rig, range, kLayerWidths), std::back_inserter(layers));
  }
  return layers;
}

// The same layers found twice with the same search buffers, the second time after other searches have left their marks:
// the bytes of every list are equal.
TEST(FindBorderBlocks, OnGpuTwice_GivesIdenticalBytes) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(2);
  ASSERT_EQ(grids.size(), 2U);
  std::vector<BorderLayerResult> first;
  std::vector<BorderLayerResult> second;
  for (const GridOnBothProcessors& grid : grids) {
    std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(grid.on_gpu.grid);
    ASSERT_TRUE(rig.has_value());
    std::ranges::move(FindLayersOfTestedRanges(*rig), std::back_inserter(first));
    std::ranges::move(FindLayersOfTestedRanges(*rig), std::back_inserter(second));
  }
  ASSERT_FALSE(first.empty());
  const std::vector<std::uint64_t> first_words = ToResultWords(std::span<const BorderLayerResult>(first));
  const std::vector<std::uint64_t> second_words = ToResultWords(std::span<const BorderLayerResult>(second));
  EXPECT_EQ(first_words, second_words);
  WriteDigestOfResults("border search on the GPU, twice", std::span<const BorderLayerResult>(second));
}

// The memory counts of a room of max_cell_blocks blocks, as the estimate reads them.
MemoryCounts ToRoomMemoryCounts(std::uint32_t max_cell_blocks) {
  GridCounts room_counts = {};
  room_counts.cell_blocks = max_cell_blocks;
  return ToMemoryCounts(room_counts, 0);
}

// At 1,000 blocks: reserving the estimate of the GPU search buffers takes at least its reserved bytes from the GPU's
// free bytes; building the search buffers into it takes no more.
TEST(BuildBorderSearchMemoryEstimate, MatchesAllocatedGpuBytes) {
  constexpr std::uint32_t kThousandBlocks = 1000;
  const std::vector<MemoryEstimateEntry> estimate = BuildBorderSearchMemoryEstimate(RunOn::kGpu, kThousandBlocks);
  const std::size_t byte_count =
      ComputeRunMemoryBytes(estimate, ToRoomMemoryCounts(kThousandBlocks), MemoryWhere::kRunOn);
  const std::expected<MemoryAvailable, BufferError> before_reserve = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(before_reserve.has_value());
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, byte_count);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<MemoryAvailable, BufferError> after_reserve = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(after_reserve.has_value());
  EXPECT_GE(before_reserve->free_bytes - after_reserve->free_bytes, run_memory->reserved_bytes());
  const std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(RunOn::kGpu, kThousandBlocks, *run_memory);
  ASSERT_TRUE(search_buffers.has_value());
  const std::expected<MemoryAvailable, BufferError> after_build = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_TRUE(after_build.has_value());
  EXPECT_EQ(after_build->free_bytes, after_reserve->free_bytes);
  const std::array<std::uint64_t, 2> words = {run_memory->reserved_bytes(), run_memory->handed_out_bytes()};
  WriteResultDigest("BuildBorderSearchBuffers on the GPU, 1,000 blocks", std::span<const std::uint64_t>(words));
}

TEST(BuildBorderSearchBuffers, OnGpuIntoRunMemory_HandsOutExactlyTheEstimate) {
  constexpr std::uint32_t kThousandBlocks = 1000;
  const std::vector<MemoryEstimateEntry> estimate = BuildBorderSearchMemoryEstimate(RunOn::kGpu, kThousandBlocks);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(
      RunOn::kGpu,
      ComputeRunMemoryBytes(estimate, ToRoomMemoryCounts(kThousandBlocks), MemoryWhere::kRunOn)
  );
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(RunOn::kGpu, kThousandBlocks, *run_memory);
  ASSERT_TRUE(search_buffers.has_value());
  EXPECT_EQ(search_buffers->max_cell_blocks(), kThousandBlocks);
  EXPECT_EQ(search_buffers->run_on(), RunOn::kGpu);
  EXPECT_EQ(run_memory->handed_out_bytes(), run_memory->reserved_bytes());
}

// The largest local memory of the GPU functions is at most the bytes of the Stack sentence of cell_block_range_gpu.h,
// and above the driver's default of 1,024 bytes, so the entry of BuildBorderSearchStackEstimate is needed.
TEST(FindBorderBlocks, OnGpuFunctions_KeepTheStatedStack) {
  std::uint64_t bytes_per_thread = 0;
  ASSERT_EQ(ComputeLargestBorderSearchStackOnGpu(&bytes_per_thread), RunOnCallStatus::kDone);
  EXPECT_LE(bytes_per_thread, kBorderSearchStackBytes);
  EXPECT_GT(bytes_per_thread, 1024U);
  const std::array<std::uint64_t, 1> words = {bytes_per_thread};
  WriteResultDigest("border search stack bytes per thread on the GPU", std::span<const std::uint64_t>(words));
}

// The stack entry: the driver's stack for every resident thread, in the memory of the run.
TEST(BuildBorderSearchStackEstimate, OnGpu_IsTheStackForEveryResidentThread) {
  const std::expected<MemoryEstimateEntry, GridFailure> entry = BuildBorderSearchStackEstimate(RunOn::kGpu);
  ASSERT_TRUE(entry.has_value());
  std::uint64_t resident_threads = 0;
  ASSERT_EQ(ComputeResidentThreadsOnGpu(&resident_threads), RunOnCallStatus::kDone);
  EXPECT_EQ(entry->where, MemoryWhere::kDriverStack);
  EXPECT_EQ(entry->given_count, resident_threads);
  EXPECT_GT(entry->bytes_each, 0U);
  EXPECT_LE(entry->bytes_each, kBorderSearchStackBytes);
}

}  // namespace
}  // namespace fluidity::grid
