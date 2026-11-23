#include "fluidity/grid/grid_memory_estimate.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {
namespace {

// The counts of a grid of cell_blocks blocks, as a memory estimate reads them.
MemoryCounts BuildCounts(std::uint32_t cell_blocks) {
  GridCounts counts = {};
  counts.cell_blocks = cell_blocks;
  return ToMemoryCounts(counts, 0);
}

// What one entry of an estimate must say at 1,000 blocks. An empty given_count means the entry is counted per
// something and states none.
struct ExpectedEntry {
  std::string_view name;
  CountedPer counted_per;
  std::uint32_t bytes_each;
  std::uint64_t given_count;
  MemoryWhere where;
};

constexpr std::uint64_t kOrderBytes = 4ULL * ((64 * 1000) + 1);  // 4 bytes for each of 64 x 1,000 + 1 items.
constexpr std::uint64_t kOrderBuffersBytes = (6ULL * ((64 * 1000) + 1)) + 4096;  // {6, 4096} of the GPU.

void CheckEntry(const MemoryEstimateEntry& entry, const ExpectedEntry& expected) {
  EXPECT_EQ(entry.name, expected.name);
  const auto fields = std::tie(entry.counted_per, entry.bytes_each, entry.given_count, entry.where, entry.copies);
  const std::uint32_t one_copy = 1;
  const auto expected_fields =
      std::tie(expected.counted_per, expected.bytes_each, expected.given_count, expected.where, one_copy);
  EXPECT_EQ(fields, expected_fields) << entry.name;
  EXPECT_EQ(entry.kept_for, KeptFor::kWholeRun) << entry.name;
}

void CheckEntries(const std::vector<MemoryEstimateEntry>& estimate, const std::vector<ExpectedEntry>& expected) {
  ASSERT_EQ(estimate.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    CheckEntry(estimate[index], expected[index]);
  }
}

// An entry counted per cell block, cell place or face place.
ExpectedEntry
BuildExpectedPerItemEntry(std::string_view name, CountedPer counted_per, std::uint32_t bytes_each, MemoryWhere where) {
  return {.name = name, .counted_per = counted_per, .bytes_each = bytes_each, .given_count = 0, .where = where};
}

// An entry of its own item count, in the run's memory.
ExpectedEntry BuildExpectedGivenEntry(std::string_view name, std::uint64_t given_count, std::uint32_t bytes_each) {
  return {
      .name = name,
      .counted_per = CountedPer::kGivenCount,
      .bytes_each = bytes_each,
      .given_count = given_count,
      .where = MemoryWhere::kRunOn,
  };
}

// The bytes one entry takes as one memory piece per copy, from its own item count.
std::uint64_t ComputeEntryPieceBytes(const MemoryEstimateEntry& entry, const MemoryCounts& counts) {
  std::uint64_t item_count = entry.given_count;
  switch (entry.counted_per) {
    case CountedPer::kCellBlock:
      item_count = counts.cell_blocks;
      break;
    case CountedPer::kCellPlace:
      item_count = counts.cell_places;
      break;
    case CountedPer::kFacePlace:
      item_count = counts.face_places;
      break;
    case CountedPer::kCornerPlace:
      item_count = counts.corner_places;
      break;
    case CountedPer::kParticle:
      item_count = counts.particles;
      break;
    case CountedPer::kGivenCount:
      break;
  }
  return entry.copies * ComputeRunMemoryPieceBytes(item_count * entry.bytes_each);
}

std::uint64_t ComputeEntryPieceBytesSum(const std::vector<MemoryEstimateEntry>& estimate, const MemoryCounts& counts) {
  std::uint64_t sum_of_pieces = 0;
  for (const MemoryEstimateEntry& entry : estimate) {
    sum_of_pieces += ComputeEntryPieceBytes(entry, counts);
  }
  return sum_of_pieces;
}

void CheckEveryEntryHasCopies(const std::vector<MemoryEstimateEntry>& estimate, std::uint32_t copies) {
  for (const MemoryEstimateEntry& entry : estimate) {
    EXPECT_EQ(entry.copies, copies) << entry.name;
  }
}

void CheckSharedCopies(const std::vector<MemoryEstimateEntry>& estimate, std::size_t builder_buffers_entry_count) {
  std::size_t counted_once = 0;
  for (const MemoryEstimateEntry& entry : estimate) {
    const bool is_builder_buffers_entry = entry.name.starts_with("grid: builder buffers:");
    EXPECT_EQ(entry.copies, is_builder_buffers_entry ? 1U : 2U) << entry.name;
    counted_once += is_builder_buffers_entry ? 1 : 0;
  }
  EXPECT_EQ(counted_once, builder_buffers_entry_count);
}

void CheckTwoGridsDoubleEveryByte(RunOn run_on) {
  const std::vector<MemoryEstimateEntry> one_grid = BuildGridMemoryEstimate(run_on, 1, 1000);
  const std::vector<MemoryEstimateEntry> two_grids = BuildGridMemoryEstimate(run_on, 2, 1000);
  const MemoryCounts counts = BuildCounts(1000);
  CheckEveryEntryHasCopies(two_grids, 2);
  EXPECT_EQ(
      ComputeMemoryBytes(two_grids, counts, MemoryWhere::kRunOn),
      2 * ComputeMemoryBytes(one_grid, counts, MemoryWhere::kRunOn)
  );
  EXPECT_EQ(
      ComputeMemoryBytes(two_grids, counts, MemoryWhere::kCpu),
      2 * ComputeMemoryBytes(one_grid, counts, MemoryWhere::kCpu)
  );
}

void CheckNoCpuCopyName(RunOn run_on) {
  for (const MemoryEstimateEntry& entry : BuildGridMemoryEstimate(run_on, 1, 1000)) {
    EXPECT_FALSE(entry.name.ends_with("(CPU copy)")) << entry.name;
  }
}

TEST(BuildGridMemoryEstimate, CpuGrid_IsNineHundredSeventyTwoBytesPerBlock) {
  // Per block: 8 + 108 + 36 + 64 + 240 = 456 in the run's memory, then 8 x 64 + 4 = 516 in CPU memory.
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, 1000);
  const MemoryCounts counts = BuildCounts(1000);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts), 1000U * 972U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kRunOn), 1000U * 456U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kCpu), 1000U * 516U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kDriverStack), 0U);
}

TEST(BuildGridMemoryEstimate, CpuGrid_ListsTheTablesThenTheBuilderBuffers) {
  CheckEntries(
      BuildGridMemoryEstimate(RunOn::kCpu, 1, 1000),
      {
          BuildExpectedPerItemEntry("grid: block ids", CountedPer::kCellBlock, 8, MemoryWhere::kRunOn),
          BuildExpectedPerItemEntry(
              "grid: same-size-level neighbour blocks",
              CountedPer::kCellBlock,
              108,
              MemoryWhere::kRunOn
          ),
          BuildExpectedPerItemEntry("grid: bigger and smaller blocks", CountedPer::kCellBlock, 36, MemoryWhere::kRunOn),
          BuildExpectedPerItemEntry("grid: cell states", CountedPer::kCellPlace, 1, MemoryWhere::kRunOn),
          BuildExpectedPerItemEntry("grid: face piece kinds", CountedPer::kFacePlace, 1, MemoryWhere::kRunOn),
          BuildExpectedPerItemEntry("grid: builder buffers: cell ids", CountedPer::kCellPlace, 8, MemoryWhere::kCpu),
          BuildExpectedPerItemEntry("grid: piece map", CountedPer::kCellBlock, 4, MemoryWhere::kCpu),
      }
  );
}

TEST(BuildGridMemoryEstimate, GpuGrid_ListsTheTablesThenTheBuilderBuffersThenTheStatus) {
  CheckEntries(
      BuildGridMemoryEstimate(RunOn::kGpu, 1, 1000),
      {
          BuildExpectedPerItemEntry("grid: block ids", CountedPer::kCellBlock, 8, MemoryWhere::kRunOn),
          BuildExpectedPerItemEntry(
              "grid: same-size-level neighbour blocks",
              CountedPer::kCellBlock,
              108,
              MemoryWhere::kRunOn
          ),
          BuildExpectedPerItemEntry("grid: bigger and smaller blocks", CountedPer::kCellBlock, 36, MemoryWhere::kRunOn),
          BuildExpectedPerItemEntry("grid: cell states", CountedPer::kCellPlace, 1, MemoryWhere::kRunOn),
          BuildExpectedPerItemEntry("grid: face piece kinds", CountedPer::kFacePlace, 1, MemoryWhere::kRunOn),
          BuildExpectedPerItemEntry("grid: builder buffers: cell ids", CountedPer::kCellPlace, 8, MemoryWhere::kRunOn),
          BuildExpectedPerItemEntry("grid: builder buffers: sort keys", CountedPer::kCellPlace, 8, MemoryWhere::kRunOn),
          BuildExpectedGivenEntry("grid: builder buffers: order", kOrderBytes, 1),
          BuildExpectedGivenEntry("grid: builder buffers: second order", kOrderBytes, 1),
          BuildExpectedGivenEntry("grid: builder buffers: order buffers", kOrderBuffersBytes, 1),
          BuildExpectedPerItemEntry("grid: piece map", CountedPer::kCellBlock, 4, MemoryWhere::kRunOn),
          BuildExpectedGivenEntry("grid: build status", 1, sizeof(GridBuildStatus)),
      }
  );
}

TEST(BuildGridMemoryEstimate, GpuRun_KeepsNothingOnTheCpu) {
  const OrderBuffersSize size = ComputeOrderBuffersSize(RunOn::kGpu);
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kGpu, 1, 1000);
  const MemoryCounts counts = BuildCounts(1000);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kCpu), 0U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kDriverStack), 0U);
  // Per block 456 tables + 1,536 builder buffers + 4 piece map + 64 x bytes_per_item of the order buffers, then once
  // fixed_bytes + bytes_per_item (the extra item of the order buffers) + 8 (the extra items of the two orders) and
  // one status record.
  const std::uint64_t per_block_bytes = 456U + 1536U + 4U + (64U * size.bytes_per_item);
  const std::uint64_t expected_bytes =
      (1000U * per_block_bytes) + size.fixed_bytes + size.bytes_per_item + 8U + sizeof(GridBuildStatus);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kRunOn), expected_bytes);
  // With {6, 4096} and a status of 192 bytes: 1000 x 2,380 + 4,096 + 6 + 8 + 192.
  EXPECT_EQ(size.bytes_per_item, 6U);
  EXPECT_EQ(size.fixed_bytes, 4096U);
  EXPECT_EQ(expected_bytes, 2384302U);
}

TEST(BuildGridMemoryEstimate, CpuRun_NoEntryIsACpuCopy) {
  CheckNoCpuCopyName(RunOn::kCpu);
}

TEST(BuildGridMemoryEstimate, GpuRun_NoEntryIsACpuCopy) {
  CheckNoCpuCopyName(RunOn::kGpu);
}

TEST(BuildGridMemoryEstimate, CpuTwoGrids_DoublesEveryEntry) {
  CheckTwoGridsDoubleEveryByte(RunOn::kCpu);
}

TEST(BuildGridMemoryEstimate, GpuTwoGrids_DoublesEveryEntry) {
  CheckTwoGridsDoubleEveryByte(RunOn::kGpu);
}

TEST(BuildGridMemoryEstimate, CpuSharedBuilderBuffers_CountedOnce) {
  const MemoryCounts counts = BuildCounts(1000);
  const std::vector<MemoryEstimateEntry> estimate =
      BuildGridMemoryEstimate(RunOn::kCpu, 2, 1000, BuilderBuffersSharing::kOneForAllGrids);
  CheckSharedCopies(estimate, 1);
  // Two grids of 456 bytes per block in the run's memory; in CPU memory two piece maps of 4 and one array of cell ids
  // of 512 bytes per block.
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kRunOn), 2U * 456U * 1000U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kCpu), (2U * 4U * 1000U) + (512U * 1000U));
}

TEST(BuildGridMemoryEstimate, GpuSharedBuilderBuffers_CountedOnce) {
  const MemoryCounts counts = BuildCounts(1000);
  const std::vector<MemoryEstimateEntry> estimate =
      BuildGridMemoryEstimate(RunOn::kGpu, 2, 1000, BuilderBuffersSharing::kOneForAllGrids);
  CheckSharedCopies(estimate, 5);
  // Two grids of 460 bytes per block share one set of 1,920 bytes per block, with 4,096 + 6 + 8 once and two
  // status records.
  const std::uint64_t expected_bytes =
      (1000U * ((2U * 460U) + 1920U)) + 4096U + 6U + 8U + (2U * sizeof(GridBuildStatus));
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kRunOn), expected_bytes);
  EXPECT_EQ(expected_bytes, (2840U * 1000U) + 4110U + 384U);
}

TEST(BuildGridMemoryEstimate, GpuEntries_EachIsOneMemoryPiece) {
  // Protects the rounding contract only: the reservation is the sum, over the entries, of copies x the piece bytes of
  // one copy. The helper rebuilds that sum from the entry's own fields, so this test passes for any list of entries.
  // The literal tests below (8,448 and 2,385,664 bytes) are what pin the entries themselves.
  for (const std::uint32_t max_cell_blocks : {1U, 7U, 1000U}) {
    const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kGpu, 1, max_cell_blocks);
    const MemoryCounts counts = BuildCounts(max_cell_blocks);
    EXPECT_EQ(ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kRunOn), ComputeEntryPieceBytesSum(estimate, counts))
        << max_cell_blocks;
  }
}

TEST(BuildGridMemoryEstimate, GpuOneBlock_ReservesEachArrayRoundedTo256) {
  // Pieces at 1 block: the five tables 5 x 256, cell ids and sort keys 512 + 512, the two orders 4 x 65 = 260 bytes
  // each, so 512 + 512, the order buffers 6 x 65 + 4,096 = 4,486, so 4,608, the piece map 256 and the status 256.
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kGpu, 1, 1);
  const MemoryCounts counts = BuildCounts(1);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kRunOn), 6682U);
  EXPECT_EQ(
      ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kRunOn),
      (5U * 256U) + 512U + 512U + 512U + 512U + 4608U + 256U + 256U
  );
  EXPECT_EQ(ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kRunOn), 8448U);
}

TEST(BuildGridMemoryEstimate, GpuThousandBlocks_ReservesEachArrayRoundedTo256) {
  // Pieces at 1,000 blocks: ids 8,000 -> 8,192; neighbours 108,000 -> 108,032; bigger and smaller 36,000 -> 36,096;
  // cell states 64,000; face piece kinds 240,000 -> 240,128; cell ids and sort keys 512,000 each; each order
  // 4 x 64,001 = 256,004 -> 256,256; order buffers 6 x 64,001 + 4,096 = 388,102 -> 388,352; piece map 4,000 -> 4,096;
  // status 192 -> 256.
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kGpu, 1, 1000);
  const MemoryCounts counts = BuildCounts(1000);
  const std::uint64_t expected_bytes =
      8192U + 108032U + 36096U + 64000U + 240128U + 512000U + 512000U + 256256U + 256256U + 388352U + 4096U + 256U;
  EXPECT_EQ(ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kRunOn), expected_bytes);
  EXPECT_EQ(expected_bytes, 2385664U);
}

TEST(BuildGridMemoryEstimate, CpuThousandBlocks_ReservesEachArrayRoundedTo256) {
  // In the run's memory the five tables: 8,192 + 108,032 + 36,096 + 64,000 + 240,128 = 456,448. In CPU memory the
  // cell ids 512,000 and the piece map 4,000 -> 4,096: 516,096.
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, 1000);
  const MemoryCounts counts = BuildCounts(1000);
  EXPECT_EQ(ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kRunOn), 456448U);
  EXPECT_EQ(ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kCpu), 516096U);
}

const MemoryEstimateEntry& FindEntryByName(const std::vector<MemoryEstimateEntry>& estimate, std::string_view name) {
  const auto found = std::ranges::find(estimate, name, &MemoryEstimateEntry::name);
  if (found == estimate.end()) {
    ADD_FAILURE() << "no entry named " << name;
    std::abort();
  }
  return *found;
}

TEST(BuildGridMemoryEstimate, LargestRoom_KeepsEveryCountExact) {
  // At the limit, n = 17,895,697 blocks: 64 x 17,895,697 + 1 = 1,145,324,609 items, below 2^32. Each order holds
  // 4 x 1,145,324,609 = 4,581,298,436 bytes; the order buffers 6 x 1,145,324,609 + 4,096 = 6,871,951,750 bytes. One
  // grid takes 1,484 bytes per block (tables 456, cell ids 512, sort keys 512, piece map 4), the two orders and the
  // order buffers 14 x (64 n + 1) + 4,096, and 192 for the status: 2,380 n + 4,302 = 42,591,763,162 bytes, far from
  // the 2^64 - 1 at which the sums saturate.
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kGpu, 1, kMaxCellBlocksLimit);
  EXPECT_EQ(FindEntryByName(estimate, "grid: builder buffers: order").given_count, 4581298436ULL);
  EXPECT_EQ(FindEntryByName(estimate, "grid: builder buffers: second order").given_count, 4581298436ULL);
  EXPECT_EQ(FindEntryByName(estimate, "grid: builder buffers: order buffers").given_count, 6871951750ULL);
  const std::uint64_t total_bytes = ComputeMemoryBytes(estimate, BuildCounts(kMaxCellBlocksLimit), MemoryWhere::kRunOn);
  EXPECT_EQ(total_bytes, 42591763162ULL);
  EXPECT_LT(total_bytes, std::numeric_limits<std::uint64_t>::max());
}

TEST(BuildGridMemoryEstimate, NoGrids_OnePerGridCountsNothing) {
  // Every entry has copies 0, so no entry counts a byte on either processor.
  for (const RunOn run_on : {RunOn::kCpu, RunOn::kGpu}) {
    const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(run_on, 0, 10);
    CheckEveryEntryHasCopies(estimate, 0);
    EXPECT_EQ(ComputeMemoryBytes(estimate, BuildCounts(10)), 0U);
  }
}

TEST(BuildGridMemoryEstimate, NoGrids_OneForAllGridsStillCountsTheBuilderBuffers) {
  // The one shared set keeps its copy: on the GPU at 10 blocks 10 x 1,920 + 4,110 = 23,310 bytes; on the CPU the
  // cell ids, 10 x 512 = 5,120 bytes, in CPU memory. Nothing else counts.
  const MemoryCounts counts = BuildCounts(10);
  const std::vector<MemoryEstimateEntry> gpu_estimate =
      BuildGridMemoryEstimate(RunOn::kGpu, 0, 10, BuilderBuffersSharing::kOneForAllGrids);
  EXPECT_EQ(ComputeMemoryBytes(gpu_estimate, counts, MemoryWhere::kRunOn), 23310U);
  EXPECT_EQ(ComputeMemoryBytes(gpu_estimate, counts, MemoryWhere::kCpu), 0U);
  const std::vector<MemoryEstimateEntry> cpu_estimate =
      BuildGridMemoryEstimate(RunOn::kCpu, 0, 10, BuilderBuffersSharing::kOneForAllGrids);
  EXPECT_EQ(ComputeMemoryBytes(cpu_estimate, counts, MemoryWhere::kRunOn), 0U);
  EXPECT_EQ(ComputeMemoryBytes(cpu_estimate, counts, MemoryWhere::kCpu), 5120U);
}

TEST(ToMemoryCounts, NineBlocks_CountsPlacesPerBlock) {
  GridCounts counts = {};
  counts.cell_blocks = 9;
  const MemoryCounts memory_counts = ToMemoryCounts(counts, 12);
  EXPECT_EQ(memory_counts.cell_blocks, 9U);
  EXPECT_EQ(memory_counts.cell_places, 576U);
  EXPECT_EQ(memory_counts.face_places, 2160U);
  EXPECT_EQ(memory_counts.corner_places, 1125U);
  EXPECT_EQ(memory_counts.particles, 12U);
}

TEST(GridBuildStatus, IsTriviallyCopyable) {
  EXPECT_TRUE(std::is_trivially_copyable_v<GridBuildStatus>);
  EXPECT_EQ(kNoFailureKey, 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(sizeof(GridBuildStatus), 192U);
  EXPECT_EQ(offsetof(GridBuildStatus, face_pieces), 184U);
}

}  // namespace
}  // namespace fluidity::grid
