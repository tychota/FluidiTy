// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on a machine with a GPU.
//
// Asks FindCellAt on the GPU for 10,000 positions of random grids and expects the cell places of the CPU, in both forms
// of the function: with and without a starting block. The tables of the GPU grid are built on the GPU, so this also
// shows that they answer as the CPU's do. Every GPU result goes to WriteResultDigest under a unique name.

#include "fluidity/grid/find_cell_at.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_functions_answers.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

constexpr int kGridCount = 10;
constexpr int kPositionsPerGrid = 1000;

// The 1,000 queries of one grid: positions drawn from one std::mt19937_64 of seed grid_entry + 20261201, each axis
// -0.5 + 4 u with u from ToUnitDouble, so the grid's cells fill 0..3 m and about 42 in 100 positions lie inside them.
// The starting block of query i is the block of the cell the CPU finds at position i - 1, and none for the first query
// and after a position outside the cells.
std::vector<FindCellAtQuery> BuildFindCellAtQueries(const GridTables& cpu_tables, int grid_entry) {
  constexpr std::uint64_t kFirstSeed = 20261201;
  std::mt19937_64 generator(
      kFirstSeed + static_cast<std::uint64_t>(grid_entry)
  );  // NOLINT(bugprone-random-generator-seed)
  std::vector<FindCellAtQuery> queries;
  queries.reserve(kPositionsPerGrid);
  CellBlockIndex start_block = {.value = kNoIndex};
  for (int draw = 0; draw < kPositionsPerGrid; ++draw) {
    const double position_x = -0.5 + (4.0 * ToUnitDouble(generator()));
    const double position_y = -0.5 + (4.0 * ToUnitDouble(generator()));
    const double position_z = -0.5 + (4.0 * ToUnitDouble(generator()));
    const Vec3 position = {.x = position_x, .y = position_y, .z = position_z};
    queries.push_back({.position = position, .near = start_block});
    const CellPlaceIndex found = FindCellAt(cpu_tables, position);
    start_block = found.value == kNoIndex ? CellBlockIndex{.value = kNoIndex} : ToCellBlockIndex(found);
  }
  return queries;
}

// Answers the queries on the CPU, in order.
std::vector<FoundCells> ComputeFoundCellsOnCpu(const GridTables& tables, std::span<const FindCellAtQuery> queries) {
  std::vector<FoundCells> results;
  results.reserve(queries.size());
  for (const FindCellAtQuery& query : queries) {
    results.push_back(FindCellsAtPosition(tables, query));
  }
  return results;
}

// Counts the results whose position found a cell.
int ComputeFoundCount(std::span<const FoundCells> results) {
  return static_cast<int>(std::ranges::count_if(results, [](const FoundCells& found) {
    return found.without_start.value != kNoIndex;
  }));
}

// 10 random grids of 1,000 positions each: both forms of FindCellAt give the CPU's places on the GPU. The CPU counts
// the positions that find a cell and those that find none, so a drop in the coverage fails the test.
TEST(FindCellAt, OnGpu_EqualsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  int found_count = 0;
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    const GridOnBothProcessors& grid = grids.at(static_cast<std::size_t>(grid_entry));
    const std::vector<FindCellAtQuery> queries =
        BuildFindCellAtQueries(grid.on_cpu.grid.tables(RunOn::kCpu), grid_entry);
    const std::vector<FoundCells> on_cpu = ComputeFoundCellsOnCpu(grid.on_cpu.grid.tables(RunOn::kCpu), queries);
    found_count += ComputeFoundCount(on_cpu);

    std::vector<FoundCells> on_gpu(queries.size());
    const RunOnCallStatus status = FindCellsAtOnGpu(
        grid.on_gpu.grid.tables(RunOn::kGpu),
        queries.data(),
        static_cast<int>(queries.size()),
        on_gpu.data()
    );
    ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
    CheckSameResults(std::span<const FoundCells>(on_cpu), std::span<const FoundCells>(on_gpu), "FindCellAt, position");
    WriteDigestOfResults(
        "FindCellAt on the GPU, both forms, grid " + std::to_string(grid_entry),
        std::span<const FoundCells>(on_gpu)
    );
  }
  const int position_count = kGridCount * kPositionsPerGrid;
  EXPECT_GT(found_count, position_count / 4);
  EXPECT_LT(found_count, position_count - (position_count / 10));
}

// A call with no query starts no GPU function and is done: it asks the GPU for nothing, so it cannot fail on an
// empty launch.
TEST(FindCellAt, OnGpuWithNoQuery_IsDone) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(1);
  ASSERT_EQ(grids.size(), 1U);
  EXPECT_EQ(
      FindCellsAtOnGpu(grids.front().on_gpu.grid.tables(RunOn::kGpu), nullptr, 0, nullptr),
      RunOnCallStatus::kDone
  ) << ReadLastRunnerFailure();
}

}  // namespace
}  // namespace fluidity::grid
