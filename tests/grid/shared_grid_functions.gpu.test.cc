// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Asks the shared grid functions that no other GPU test of the grid starts by name on the GPU, and expects the CPU's
// results bit for bit: the formulas without a table (the complete weighted spread, the reference speed, the cell widths
// by size level, the sort count and the small cells of a big cell), what the tables say about a cell place (its centre,
// its id and the place of that id) and about a face place (the distance between the centres of its cells, its id by the
// tables and by its parts, and the place of that id). Also reads the attributes of every runner's GPU function. The
// functions that these reach are listed in grid_functions_runner_gpu.h. Every GPU result goes to WriteResultDigest
// under a unique name.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
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
constexpr int kFormulaQueryCount = 4000;
constexpr std::uint64_t kFormulaSeed = 20261122;
constexpr std::uint64_t kRoomCellBlocks = 64;  // The cell places of one block, as ComputeSortCount counts the room.

// Draws a number in [low, high) from the generator.
double FindDrawnBetween(std::mt19937_64& generator, double low, double high) {
  return low + ((high - low) * ToUnitDouble(generator()));
}

// The 4,000 queries of the formulas: frames of 1 to 16 size levels whose biggest width and origin are drawn (so most
// widths are not dyadic), cell widths and speeds, build counts, and the cells of the random grids as big cells with
// each of their eighths in turn. From one std::mt19937_64 of seed kFormulaSeed.
std::vector<FormulaQuery> BuildFormulaQueries(const std::vector<CellId>& big_cells) {
  std::mt19937_64 generator(kFormulaSeed);  // NOLINT(bugprone-random-generator-seed)
  std::vector<FormulaQuery> queries;
  queries.reserve(kFormulaQueryCount);
  for (int draw = 0; draw < kFormulaQueryCount; ++draw) {
    const GridFrame frame = {
        .origin = {.x = FindDrawnBetween(generator, -10, 10), .y = FindDrawnBetween(generator, -10, 10), .z = 0.5},
        .biggest_cell_width = FindDrawnBetween(generator, 0.01, 8),
        .size_level_count =
            static_cast<std::uint8_t>(1 + ToUniformIndex(generator(), static_cast<std::uint64_t>(kMaxSizeLevels))),
    };
    queries.push_back({
        .frame = frame,
        .cell_width = FindDrawnBetween(generator, 0.001, 4),
        .largest_liquid_speed = FindDrawnBetween(generator, 0, 20),
        .gravity_size = FindDrawnBetween(generator, 0, 20),
        .active_count = static_cast<std::uint32_t>(ToUniformIndex(generator(), 100000)),
        .max_cell_blocks = static_cast<std::uint32_t>(1 + ToUniformIndex(generator(), 2000)),
        .big_cell = big_cells[ToUniformIndex(generator(), big_cells.size())],
        .cell_eighth = draw % 8,
    });
  }
  return queries;
}

// Answers the queries of the formulas on the CPU, in order.
std::vector<FormulaResult> ComputeFormulaResultsOnCpu(const GridTables& tables, std::span<const FormulaQuery> queries) {
  std::vector<FormulaResult> results;
  results.reserve(queries.size());
  for (const FormulaQuery& query : queries) {
    results.push_back(ComputeFormulaResult(tables, query));
  }
  return results;
}

// Expects the CPU's result of the formulas for one query to say what the formulas mean, so a comparison with the GPU is
// not one of two wrong answers: the widest cell is the frame's width, the sort count is at most the room, the reference
// speed is at least the liquid speed, and a small cell is one size level below its big cell and inside it.
void CheckFormulaResultOnCpu(const FormulaQuery& query, const FormulaResult& result) {
  const auto big_size_level = static_cast<int>(ToSizeLevel(query.big_cell));
  const auto small_size_level = static_cast<int>(ToSizeLevel(result.small_cell));
  EXPECT_EQ(result.widths.width[0], query.frame.biggest_cell_width);
  EXPECT_EQ(result.widths.size_level_count, query.frame.size_level_count);
  EXPECT_GE(result.reference_speed, query.largest_liquid_speed);
  EXPECT_LE(result.sort_count, kRoomCellBlocks * query.max_cell_blocks);
  EXPECT_EQ(small_size_level, big_size_level + 1);
  EXPECT_EQ(ComputeBigCell(result.small_cell).value, query.big_cell.value);
}

// Expects the CPU's results of the formulas to say what the formulas mean, one query after the other.
void CheckFormulaResultsOnCpu(std::span<const FormulaQuery> queries, std::span<const FormulaResult> results) {
  ASSERT_EQ(queries.size(), results.size());
  for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
    SCOPED_TRACE(query_index);
    CheckFormulaResultOnCpu(queries[query_index], results[query_index]);
  }
}

// The formulas that take no table, over 4,000 drawn queries: the GPU's bits are the CPU's.
TEST(SharedGridFormulas, OnGpu_SameBitsAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(1);
  ASSERT_EQ(grids.size(), 1U);
  const std::vector<FormulaQuery> queries = BuildFormulaQueries(grids.at(0).cells);
  const std::vector<FormulaResult> on_cpu =
      ComputeFormulaResultsOnCpu(grids.at(0).on_cpu.grid.tables(RunOn::kCpu), queries);
  CheckFormulaResultsOnCpu(queries, on_cpu);

  std::vector<FormulaResult> on_gpu(queries.size());
  const RunOnCallStatus status = ComputeFormulasOnGpu(
      grids.at(0).on_gpu.grid.tables(RunOn::kGpu),
      queries.data(),
      static_cast<int>(queries.size()),
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  CheckSameResults(std::span<const FormulaResult>(on_cpu), std::span<const FormulaResult>(on_gpu), "formulas");
  WriteDigestOfResults("shared grid formulas on the GPU, 4,000 queries", std::span<const FormulaResult>(on_gpu));
}

// Answers ComputeCellPlaceFacts on the CPU for each place, in order.
std::vector<CellPlaceFacts>
ComputeCellPlaceFactsOnCpu(const GridTables& tables, std::span<const CellPlaceIndex> places) {
  std::vector<CellPlaceFacts> results;
  results.reserve(places.size());
  for (const CellPlaceIndex place : places) {
    results.push_back(ComputeCellPlaceFacts(tables, place));
  }
  return results;
}

// Answers ComputeFacePlaceFacts on the CPU for each place, in order.
std::vector<FacePlaceFacts>
ComputeFacePlaceFactsOnCpu(const GridTables& tables, std::span<const FacePlaceIndex> places) {
  std::vector<FacePlaceFacts> results;
  results.reserve(places.size());
  for (const FacePlaceIndex place : places) {
    results.push_back(ComputeFacePlaceFacts(tables, place));
  }
  return results;
}

// Expects the CPU's facts of one active cell to say that the id is the cell of the list and its place is the place
// asked.
void CheckCellPlaceFactsOnCpu(const CellPlaceFacts& facts, CellId cell, CellPlaceIndex place) {
  EXPECT_EQ(facts.id.value, cell.value);
  EXPECT_EQ(facts.place_of_id.value, place.value);
}

// Expects the CPU's facts of one face piece to say that the two ids are equal, the distance is above 0 and the id leads
// back to the place.
void CheckFacePlaceFactsOnCpu(const FacePlaceFacts& facts, FacePlaceIndex place) {
  EXPECT_NE(facts.id.value, 0U);
  EXPECT_EQ(facts.id.value, facts.id_from_parts.value);
  EXPECT_GT(facts.center_distance, 0.0);
  EXPECT_EQ(facts.place_of_id.value, place.value);
}

// The facts of every active cell of one grid: the GPU's bits are the CPU's, and on the CPU each id is its cell of the
// list and leads back to its place.
void CheckCellPlaceFactsOfGrid(const GridOnBothProcessors& grid, int grid_entry) {
  const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
  const std::vector<CellPlaceIndex> places = FindActiveCellPlaces(cpu_tables, grid.cells);
  ASSERT_FALSE(places.empty());
  const std::vector<CellPlaceFacts> on_cpu = ComputeCellPlaceFactsOnCpu(cpu_tables, places);
  for (std::size_t entry = 0; entry < places.size(); ++entry) {
    SCOPED_TRACE(entry);
    CheckCellPlaceFactsOnCpu(on_cpu[entry], grid.cells[entry], places[entry]);
  }
  std::vector<CellPlaceFacts> on_gpu(places.size());
  const RunOnCallStatus status = ComputeCellPlaceFactsOnGpu(
      grid.on_gpu.grid.tables(RunOn::kGpu),
      places.data(),
      static_cast<int>(places.size()),
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  CheckSameResults(
      std::span<const CellPlaceFacts>(on_cpu),
      std::span<const CellPlaceFacts>(on_gpu),
      "cell place facts, grid " + std::to_string(grid_entry)
  );
  WriteDigestOfResults(
      "cell place facts on the GPU, every active cell of grid " + std::to_string(grid_entry),
      std::span<const CellPlaceFacts>(on_gpu)
  );
}

// The facts of every face piece of one grid: the GPU's bits are the CPU's, and on the CPU the ids, the distance and the
// place of the id are what the tables say.
void CheckFacePlaceFactsOfGrid(const GridOnBothProcessors& grid, int grid_entry) {
  const GridTables cpu_tables = grid.on_cpu.grid.tables(RunOn::kCpu);
  const std::vector<FacePlaceIndex> places = FindFacePiecePlaces(cpu_tables);
  ASSERT_FALSE(places.empty());
  const std::vector<FacePlaceFacts> on_cpu = ComputeFacePlaceFactsOnCpu(cpu_tables, places);
  for (std::size_t entry = 0; entry < places.size(); ++entry) {
    SCOPED_TRACE(entry);
    CheckFacePlaceFactsOnCpu(on_cpu[entry], places[entry]);
  }
  std::vector<FacePlaceFacts> on_gpu(places.size());
  const RunOnCallStatus status = ComputeFacePlaceFactsOnGpu(
      grid.on_gpu.grid.tables(RunOn::kGpu),
      places.data(),
      static_cast<int>(places.size()),
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  CheckSameResults(
      std::span<const FacePlaceFacts>(on_cpu),
      std::span<const FacePlaceFacts>(on_gpu),
      "face place facts, grid " + std::to_string(grid_entry)
  );
  WriteDigestOfResults(
      "face place facts on the GPU, every face piece of grid " + std::to_string(grid_entry),
      std::span<const FacePlaceFacts>(on_gpu)
  );
}

// The centre, the id and the place of that id for every active cell of 10 random grids: the GPU's bits are the CPU's.
TEST(ComputeCellPlaceFacts, OnGpu_SameBitsAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    CheckCellPlaceFactsOfGrid(grids.at(static_cast<std::size_t>(grid_entry)), grid_entry);
  }
}

// The cell distance, the id by the tables and by the parts, and the place of the id for every face piece of 10 random
// grids: the GPU's bits are the CPU's.
TEST(ComputeFacePlaceFacts, OnGpu_SameBitsAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  for (int grid_entry = 0; grid_entry < kGridCount; ++grid_entry) {
    SCOPED_TRACE(grid_entry);
    CheckFacePlaceFactsOfGrid(grids.at(static_cast<std::size_t>(grid_entry)), grid_entry);
  }
}

// Every runner's GPU function loads for this GPU and can start with the thread blocks the runners use, which the
// attributes show. The local memory and the registers of each are written to the notes of the run; they say what the
// binary asks for, not whether the stack is enough: the other tests start every function and show that.
TEST(GridRunners, OnGpu_FunctionsLoadWithTheirBlockSize) {
  std::vector<RunnerFunctionInfo> functions;
  ASSERT_EQ(ReadRunnerFunctionInfo(&functions), RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  EXPECT_EQ(functions.size(), 14U);
  for (const RunnerFunctionInfo& function : functions) {
    SCOPED_TRACE(function.name);
    EXPECT_GE(function.most_threads_per_block, static_cast<int>(kThreadBlockSize));
    RecordProperty(function.name + "_local_bytes", std::to_string(function.local_bytes));
    RecordProperty(function.name + "_registers", function.register_count);
    WriteResultNote(
        "runner " + function.name + ": " + std::to_string(function.local_bytes) + " bytes of local memory, "
        + std::to_string(function.register_count) + " registers, at most "
        + std::to_string(function.most_threads_per_block) + " threads per block"
    );
  }
}

}  // namespace
}  // namespace fluidity::grid
