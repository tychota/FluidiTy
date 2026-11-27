// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on a machine with a GPU.
//
// Asks FindNewCellNeighbors on the GPU for the hand-worked new cells of the near-small grid and of the range-end grid
// and expects the facts of the CPU, which are the hand-computed ones. The tables of the GPU grids are built on the GPU,
// so this also shows that they answer as the CPU's do. Every GPU result goes to WriteResultDigest under a unique name.

#include <gtest/gtest.h>

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/new_cell_rule.h"
#include "fluidity/grid/testing/grid_functions_answers.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"
#include "fluidity/grid/testing/new_cell_rule_cases.h"

namespace fluidity::grid {
namespace {

// Checks the hand cases of one grid: the CPU's answers are the hand-computed facts, and the GPU's are the CPU's, bit
// for bit.
void CheckHandCasesOnGpu(
    const std::vector<CellId>& cells,
    const std::vector<HandCase>& hand_cases,
    std::string_view name
) {
  const std::optional<GridOnBothProcessors> grid = BuildGridOnBothProcessors(ToHandGridFrame(), cells);
  ASSERT_TRUE(grid.has_value());
  const GridTables cpu_tables = grid->on_cpu.grid.tables(RunOn::kCpu);
  std::vector<CellId> queries;
  std::vector<NewCellNeighbors> on_cpu;
  for (const HandCase& hand_case : hand_cases) {
    queries.push_back(hand_case.new_cell);
    on_cpu.push_back(FindNewCellNeighborsOf(cpu_tables, hand_case.new_cell));
    EXPECT_EQ(on_cpu.back().touches_much_smaller_cell, hand_case.expected.touches_much_smaller_cell);
    EXPECT_EQ(on_cpu.back().touches_bigger_cell, hand_case.expected.touches_bigger_cell);
  }
  std::vector<NewCellNeighbors> on_gpu(queries.size());
  const RunOnCallStatus status = FindNewCellNeighborsOnGpu(
      grid->on_gpu.grid.tables(RunOn::kGpu),
      queries.data(),
      static_cast<int>(queries.size()),
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  CheckSameResults(
      std::span<const NewCellNeighbors>(on_cpu),
      std::span<const NewCellNeighbors>(on_gpu),
      "FindNewCellNeighbors, new cell"
  );
  WriteDigestOfResults(name, std::span<const NewCellNeighbors>(on_gpu));
}

// The 12 hand cases of the near-small grid and the 2 of the range-end grid give the CPU's facts on the GPU.
TEST(FindNewCellNeighbors, HandCasesOnGpu_EqualCpu) {
  CheckHandCasesOnGpu(BuildNearSmallCells(), BuildNearSmallCases(), "FindNewCellNeighbors on the GPU, near-small grid");
  CheckHandCasesOnGpu(BuildRangeEndCells(), BuildRangeEndCases(), "FindNewCellNeighbors on the GPU, range-end grid");
}

}  // namespace
}  // namespace fluidity::grid
