// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Asks the three steps that write a grid piece (ToPiecePlace, ToPieceNeighbour, ComputePieceFaceKind,
// grid/grid_piece_steps.h) for every face place of every block of real pieces of 10 random grids, on the GPU, and
// expects the CPU's answer bit for bit. The pieces are built on the CPU, so the border blocks and the piece map they
// give are the real lists; the runner copies them to the GPU. Every GPU result goes to WriteDigestOfResults under a
// unique name.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_functions_answers.h"
#include "fluidity/grid/testing/grid_functions_gpu_cases.h"
#include "fluidity/grid/testing/grid_functions_result_words.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

constexpr int kGridCount = 10;
constexpr std::uint32_t kRangeBlocks = 4;
constexpr int kNeighbourEntriesPerBlock = 36;  // 27 of the same size level, 1 bigger, 8 smaller.

// The whole grid's neighbour entry number `entry` of block `whole_block`: the 27, then the bigger block, then the 8
// smaller blocks.
std::uint32_t FindNeighbourEntry(const GridTables& whole, std::uint32_t whole_block, int entry) {
  if (entry < 27) {
    return whole.same_size_level_blocks[(std::size_t{whole_block} * 27) + static_cast<std::size_t>(entry)];
  }
  if (entry == 27) {
    return whole.bigger_block[whole_block];
  }
  return whole.smaller_blocks[(std::size_t{whole_block} * 8) + static_cast<std::size_t>(entry - 28)];
}

// One query per face place of each block of the piece: its place, a neighbour entry of the block (every entry turn by
// turn) and a loaded item that takes every item in turn.
std::vector<PieceStepQuery> BuildQueries(const GridTables& whole, std::span<const CellBlockIndex> map) {
  std::vector<PieceStepQuery> queries;
  queries.reserve(map.size() * kFacePlacesPerBlock);
  for (std::uint32_t piece_block = 0; piece_block < map.size(); ++piece_block) {
    for (int place = 0; place < kFacePlacesPerBlock; ++place) {
      queries.push_back({
          .loaded_item =
              static_cast<std::uint32_t>((std::size_t{piece_block} * kFacePlacesPerBlock + place) % map.size()),
          .whole_neighbour = FindNeighbourEntry(whole, map[piece_block].value, place % kNeighbourEntriesPerBlock),
          .piece_block = piece_block,
          .face_place_in_block = static_cast<std::uint32_t>(place),
      });
    }
  }
  return queries;
}

// Which of the two answers that tell a piece from the whole grid the CPU gave somewhere: a face place outside the
// piece, and a neighbour that is not loaded. A test that saw neither compared answers that are the same everywhere.
struct CpuAnswersSeen {
  bool has_outside_piece_kind = false;
  bool has_no_index_neighbour = false;
};

// The steps for every face place of the piece of one range: the GPU's bits are the CPU's. Records in `seen` what the
// CPU's answers held.
void CheckStepsOfRange(
    const GridOnBothProcessors& grid,
    CellBlockRange range,
    const std::string& name,
    CpuAnswersSeen& seen
) {
  const StaggeredGrid& whole_grid = grid.on_cpu.grid;
  const RangeWithBorder range_with_border = BuildRangeWithBorder(whole_grid, range);
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(whole_grid.counts().cell_blocks);
  ASSERT_TRUE(room.has_value());
  const std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(whole_grid, range, range_with_border.border_blocks, room->buffers);
  ASSERT_TRUE(piece.has_value());
  const GridTables whole = whole_grid.tables(RunOn::kCpu);
  const PieceStepSet set = {
      .range = range,
      .border_blocks = range_with_border.border_blocks.data(),
      .border_block_count = static_cast<std::uint32_t>(range_with_border.border_blocks.size()),
      .map = piece->block_in_whole_grid().data(),
      .loaded_block_count = static_cast<std::uint32_t>(piece->block_in_whole_grid().size()),
  };
  const std::vector<PieceStepQuery> queries = BuildQueries(whole, piece->block_in_whole_grid());
  std::vector<PieceStepResult> on_cpu;
  on_cpu.reserve(queries.size());
  for (const PieceStepQuery& query : queries) {
    on_cpu.push_back(ComputePieceStepResult(whole, set, query));
  }
  constexpr auto kOutsidePieceNumber = static_cast<std::uint32_t>(FacePieceKind::kOutsidePiece);
  seen.has_outside_piece_kind =
      seen.has_outside_piece_kind || std::ranges::any_of(on_cpu, [](const PieceStepResult& result) {
        return result.face_kind == kOutsidePieceNumber;
      });
  seen.has_no_index_neighbour =
      seen.has_no_index_neighbour || std::ranges::any_of(on_cpu, [](const PieceStepResult& result) {
        return result.piece_neighbour == kNoIndex;
      });
  std::vector<PieceStepResult> on_gpu(queries.size());
  const RunOnCallStatus status = ComputePieceStepResultsOnGpu(
      grid.on_gpu.grid.tables(RunOn::kGpu),
      set,
      queries.data(),
      static_cast<int>(queries.size()),
      on_gpu.data()
  );
  ASSERT_EQ(status, RunOnCallStatus::kDone) << ReadLastRunnerFailure();
  CheckSameResults(std::span<const PieceStepResult>(on_cpu), std::span<const PieceStepResult>(on_gpu), name);
  WriteDigestOfResults("piece steps on the GPU, " + name, std::span<const PieceStepResult>(on_gpu));
}

// The piece map, the neighbour entries and the face kinds of the pieces of 10 random grids, on the first, middle and
// last range and on the range of the whole grid: the GPU's bits are the CPU's.
TEST(BuildGridPiece, StepsOnGpu_SameBitsAsCpu) {
  const std::vector<GridOnBothProcessors> grids = BuildRandomGridsOnBothProcessors(kGridCount);
  ASSERT_EQ(grids.size(), static_cast<std::size_t>(kGridCount));
  CpuAnswersSeen seen;
  for (std::size_t grid_entry = 0; grid_entry < grids.size(); ++grid_entry) {
    for (const CellBlockRange range : BuildTestedRanges(grids[grid_entry].on_cpu.grid, kRangeBlocks)) {
      const std::string name =
          "grid " + std::to_string(grid_entry) + ", range from block " + std::to_string(range.first_block);
      SCOPED_TRACE(name);
      CheckStepsOfRange(grids[grid_entry], range, name, seen);
    }
  }
  EXPECT_TRUE(seen.has_outside_piece_kind);
  EXPECT_TRUE(seen.has_no_index_neighbour);
}

}  // namespace
}  // namespace fluidity::grid
