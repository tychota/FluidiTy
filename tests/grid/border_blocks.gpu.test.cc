// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on a machine with a GPU.
//
// Asks the border block test (ComputeIsBorderBlock, grid/border_block_steps.h) of every block of the random grids on
// the GPU, for a few block sets per grid and both border widths, and expects the CPU's answer bit for bit. The set's
// sorted list lives in GPU memory, which the runner copies. The CPU side must not be trivial: over the test both
// answers occur at both widths, and some block is in the wide border but not in the near one. The helpers that
// ComputeIsBorderBlock reaches are listed in grid_functions_runner_gpu.h. Every GPU result goes to WriteResultDigest
// under a unique name. A block next to the set can cost up to about 3 million table reads and moves at kTouchingCells
// and 10 million at kFourCellWidths, and a block asks every block under it, so this test may take long on the GPU.
// The second test runs the forward form (ApplyCellPlaceReadMarks, ApplyFacePlaceReadMarks), which the GPU border pass
// will run, one thread per place of a range, and expects the CPU's marks bit for bit and their union to be the border.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_functions_answers.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

constexpr int kGridCount = 5;
constexpr std::uint32_t kRangeBlockCount = 8;
constexpr std::uint64_t kSeedOffset = 60000;

// A block set as a stage gets it: a range of consecutive blocks and a sorted list, either of them possibly empty.
struct BlockSetCase {
  CellBlockRange range;
  std::vector<CellBlockIndex> sorted_blocks;
};

// How often each answer occurred on the CPU, by width, over the whole test.
struct AnswerCounts {
  int border_at_touching = 0;
  int not_border_at_touching = 0;
  int border_at_wide = 0;
  int not_border_at_wide = 0;
  int wide_only = 0;  // Border blocks at kFourCellWidths that are not border blocks at kTouchingCells.
};

// Draws an ascending list of blocks: each block with probability 1 in `one_in`, and at least one block.
std::vector<CellBlockIndex>
FindDrawnAscendingBlocks(std::mt19937_64& generator, std::uint32_t block_count, std::uint64_t one_in) {
  std::vector<CellBlockIndex> blocks;
  for (std::uint32_t block = 0; block < block_count; ++block) {
    if (ToUniformIndex(generator(), one_in) == 0) {
      blocks.push_back({.value = block});
    }
  }
  if (blocks.empty()) {
    blocks.push_back({.value = static_cast<std::uint32_t>(ToUniformIndex(generator(), block_count))});
  }
  return blocks;
}

// The three block sets of one grid, drawn from `generator`: a range of 8 blocks in the middle, a sparse sorted
// list, and a short range with a sparse list of the blocks after it.
std::vector<BlockSetCase> FindChosenBlockSets(std::mt19937_64& generator, std::uint32_t block_count) {
  const std::uint32_t range_size = block_count < kRangeBlockCount ? block_count : kRangeBlockCount;
  const BlockSetCase middle_range = {
      .range = {.first_block = (block_count - range_size) / 2, .block_count = range_size},
      .sorted_blocks = {},
  };
  const BlockSetCase sparse_list = {
      .range = {.first_block = 0, .block_count = 0},
      .sorted_blocks = FindDrawnAscendingBlocks(generator, block_count, 6),
  };
  const std::uint32_t short_range_size = range_size / 2 > 0 ? range_size / 2 : 1;
  BlockSetCase mixed = {.range = {.first_block = 0, .block_count = short_range_size}, .sorted_blocks = {}};
  for (const CellBlockIndex block : FindDrawnAscendingBlocks(generator, block_count, 10)) {
    if (block.value >= short_range_size) {
      mixed.sorted_blocks.push_back(block);
    }
  }
  return {middle_range, sparse_list, std::move(mixed)};
}

// One query per block and width: every block of the grid at kTouchingCells, then every block at kFourCellWidths.
std::vector<BorderBlockQuery> BuildQueries(std::uint32_t block_count) {
  std::vector<BorderBlockQuery> queries;
  queries.reserve(2 * static_cast<std::size_t>(block_count));
  for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
    for (std::uint32_t block = 0; block < block_count; ++block) {
      queries.push_back({.block = {.value = block}, .width = width});
    }
  }
  return queries;
}

// Answers the queries on the CPU against the set, in order.
std::vector<BorderBlockResult> ComputeBorderBlockResultsOnCpu(
    const GridTables& tables,
    const BlockSet& set,
    std::span<const BorderBlockQuery> queries
) {
  std::vector<BorderBlockResult> results;
  results.reserve(queries.size());
  for (const BorderBlockQuery& query : queries) {
    results.push_back(ComputeBorderBlockResult(tables, set, query));
  }
  return results;
}

// Adds the CPU's answers of one set to the counts. The first half of the answers is kTouchingCells, the second half
// kFourCellWidths, block by block, as BuildQueries orders them.
void ApplyAnswerCounts(std::span<const BorderBlockResult> on_cpu, AnswerCounts& counts) {
  const std::size_t block_count = on_cpu.size() / 2;
  for (std::size_t block = 0; block < block_count; ++block) {
    const bool is_border_at_touching = on_cpu[block].is_border != 0;
    const bool is_border_at_wide = on_cpu[block_count + block].is_border != 0;
    (is_border_at_touching ? counts.border_at_touching : counts.not_border_at_touching) += 1;
    (is_border_at_wide ? counts.border_at_wide : counts.not_border_at_wide) += 1;
    if (is_border_at_wide && !is_border_at_touching) {
      counts.wide_only += 1;
    }
  }
}

// The answers of every block for one set of one grid: the GPU's bits are the CPU's. The CPU's answers are counted.
void CheckBorderBlocksOfSet(
    const GridOnBothProcessors& grid,
    const BlockSetCase& set_case,
    const std::string& name,
    AnswerCounts& counts
) {
  const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
  const std::vector<BorderBlockQuery> queries = BuildQueries(cpu_tables.block_count);
  const BlockSet cpu_set = {
      .range = set_case.range,
      .sorted_blocks = set_case.sorted_blocks.data(),
      .sorted_count = static_cast<std::uint32_t>(set_case.sorted_blocks.size()),
  };
  const std::vector<BorderBlockResult> on_cpu = ComputeBorderBlockResultsOnCpu(cpu_tables, cpu_set, queries);
  ApplyAnswerCounts(on_cpu, counts);

  std::vector<BorderBlockResult> on_gpu(queries.size());
  const RunOnCallStatus status = ComputeBorderBlockResultsOnGpu(
      grid.on_gpu.grid.tables(RunOn::kGpu),
      set_case.range,
      set_case.sorted_blocks,
      queries.data(),
      static_cast<int>(queries.size()),
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  CheckSameResults(std::span<const BorderBlockResult>(on_cpu), std::span<const BorderBlockResult>(on_gpu), name);
  WriteDigestOfResults("border blocks on the GPU, " + name, std::span<const BorderBlockResult>(on_gpu));
}

// The answers of every block for the three sets of one grid: the GPU's bits are the CPU's.
void CheckBorderBlocksOfGrid(const GridOnBothProcessors& grid, std::size_t grid_entry, AnswerCounts& counts) {
  const std::uint32_t block_count = grid.on_cpu.grid.tables(RunOn::kCpu).block_count;
  ASSERT_GT(block_count, 0U);
  std::mt19937_64 generator(kSeedOffset + grid_entry);
  const std::vector<BlockSetCase> sets = FindChosenBlockSets(generator, block_count);
  for (std::size_t set_entry = 0; set_entry < sets.size(); ++set_entry) {
    SCOPED_TRACE(set_entry);
    const std::string name = "border blocks, grid " + std::to_string(grid_entry) + ", set " + std::to_string(set_entry);
    CheckBorderBlocksOfSet(grid, sets[set_entry], name, counts);
  }
}

// Expects both answers at both widths over the CPU's answers, and a block in the wide border but not in the near one,
// so the comparison with the GPU is not one of answers that are all the same.
void CheckAnswersAreNotTrivial(const AnswerCounts& counts) {
  EXPECT_GT(counts.border_at_touching, 0);
  EXPECT_GT(counts.not_border_at_touching, 0);
  EXPECT_GT(counts.border_at_wide, 0);
  EXPECT_GT(counts.not_border_at_wide, 0);
  EXPECT_GT(counts.wide_only, 0);
}

// Whether each block of 5 random grids is a border block of 3 sets, at both widths: the GPU's bits are the CPU's, and
// on the CPU both answers occur at both widths, one block being in the wide border and not in the near one.
TEST(ComputeIsBorderBlock, OnGpu_SameBitsAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  AnswerCounts counts;
  for (std::size_t grid_entry = 0; grid_entry < grids.size(); ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    CheckBorderBlocksOfGrid(grids[grid_entry], grid_entry, counts);
  }
  CheckAnswersAreNotTrivial(counts);
}

// One query per cell place and face place of `blocks`, at `width`: the places a GPU border pass runs one thread for.
std::vector<ReadMarksQuery> BuildReadMarksQueries(std::span<const CellBlockIndex> blocks, BorderWidth width) {
  std::vector<ReadMarksQuery> queries;
  queries.reserve(blocks.size() * (kCellPlacesPerBlock + kFacePlacesPerBlock));
  for (const CellBlockIndex block : blocks) {
    for (std::uint32_t place_in_block = 0; place_in_block < kCellPlacesPerBlock; ++place_in_block) {
      queries.push_back({
          .place = (block.value * kCellPlacesPerBlock) + place_in_block,
          .kind = MarkedPlaceKind::kCellPlace,
          .width = width,
      });
    }
    for (std::uint32_t place_in_block = 0; place_in_block < kFacePlacesPerBlock; ++place_in_block) {
      queries.push_back({
          .place = (block.value * kFacePlacesPerBlock) + place_in_block,
          .kind = MarkedPlaceKind::kFacePlace,
          .width = width,
      });
    }
  }
  return queries;
}

// The marks of every place of the middle range of 8 blocks of one grid, at `width`: the GPU's bits are the CPU's, and
// the blocks marked by the GPU outside the range are the border that FindBorderBlocks finds on the CPU.
void CheckReadMarksOfGrid(const GridOnBothProcessors& grid, std::size_t grid_entry, BorderWidth width) {
  const StaggeredGrid& cpu_grid = grid.on_cpu.grid;
  const GridTables cpu_tables = cpu_grid.tables(RunOn::kCpu);
  ASSERT_LE(cpu_tables.block_count, static_cast<std::uint32_t>(kMaxMarkedBlocks));
  const std::uint32_t range_size =
      cpu_tables.block_count < kRangeBlockCount ? cpu_tables.block_count : kRangeBlockCount;
  std::vector<CellBlockIndex> range_blocks;
  range_blocks.reserve(range_size);
  for (std::uint32_t entry = 0; entry < range_size; ++entry) {
    range_blocks.push_back({.value = ((cpu_tables.block_count - range_size) / 2) + entry});
  }
  const std::vector<ReadMarksQuery> queries = BuildReadMarksQueries(range_blocks, width);
  std::vector<ReadMarksResult> on_cpu;
  on_cpu.reserve(queries.size());
  for (const ReadMarksQuery& query : queries) {
    on_cpu.push_back(ComputeReadMarksResult(cpu_tables, query));
  }
  std::vector<ReadMarksResult> on_gpu(queries.size());
  const RunOnCallStatus status = ComputeReadMarksResultsOnGpu(
      grid.on_gpu.grid.tables(RunOn::kGpu),
      queries.data(),
      static_cast<int>(queries.size()),
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  const std::string name =
      "read marks, grid " + std::to_string(grid_entry) + ", width " + std::to_string(static_cast<int>(width));
  CheckSameResults(std::span<const ReadMarksResult>(on_cpu), std::span<const ReadMarksResult>(on_gpu), name);
  WriteDigestOfResults("read marks on the GPU, " + name, std::span<const ReadMarksResult>(on_gpu));
  std::vector<std::uint32_t> marked_outside;
  for (std::uint32_t block = 0; block < cpu_tables.block_count; ++block) {
    const bool is_in_range = block >= range_blocks.front().value && block <= range_blocks.back().value;
    const bool is_marked = std::ranges::any_of(on_gpu, [block](const ReadMarksResult& result) {
      return result.marks[block] != 0;
    });
    if (is_marked && !is_in_range) {
      marked_outside.push_back(block);
    }
  }
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border =
      FindBorderBlocks(cpu_grid, range_blocks, width);
  ASSERT_TRUE(border.has_value());
  std::vector<std::uint32_t> border_numbers;
  border_numbers.reserve(border->size());
  for (const CellBlockIndex block : *border) {
    border_numbers.push_back(block.value);
  }
  EXPECT_EQ(marked_outside, border_numbers);
}

// The forward form of the border (ApplyCellPlaceReadMarks, ApplyFacePlaceReadMarks), one thread per place of a range
// of 8 blocks of 5 random grids, at both widths: the GPU's marks are the CPU's bit for bit, and their union outside
// the range is the border the CPU finds block by block.
TEST(ApplyCellPlaceReadMarks, OnGpu_SameMarksAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  for (std::size_t grid_entry = 0; grid_entry < grids.size(); ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
      CheckReadMarksOfGrid(grids[grid_entry], grid_entry, width);
    }
  }
}

}  // namespace
}  // namespace fluidity::grid
