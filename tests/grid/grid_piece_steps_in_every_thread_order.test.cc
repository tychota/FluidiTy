#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/grid_piece_internal.h"
#include "fluidity/grid/grid_piece_steps.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_piece_cases.h"
#include "fluidity/grid/testing/grid_piece_replay.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"
#include "fluidity/grid/testing/thread_replay.h"

namespace fluidity::grid {
namespace {

// The launches the replay runs each step in: one item for each thread, forward and in reverse, and 7 threads, forward
// and in reverse, so each thread has several items and the order of threads shows.
constexpr std::array<ReplayLaunch, 4> kReplayLaunches = {
    ReplayLaunch{.order = LaunchOrder::kForward, .thread_count = 0},
    ReplayLaunch{.order = LaunchOrder::kReverse, .thread_count = 0},
    ReplayLaunch{.order = LaunchOrder::kForward, .thread_count = 7},
    ReplayLaunch{.order = LaunchOrder::kReverse, .thread_count = 7},
};

// What one build of a piece is asked for: the range, the border list and the room of the buffers.
struct PieceBuildInputs {
  CellBlockRange range;
  std::span<const CellBlockIndex> border;
  std::uint32_t max_cell_blocks;  // Blocks the buffers have room for.
};

// The end of a build on the GPU, replayed: the host's checks of the range and of the room, then the launches of the
// steps in `launch`, then the reading of the status, with the code the GPU build runs (CheckPieceRangeAndRoom,
// ToPieceBuildSummary). A failed check gives its failure before any launch, as on the GPU.
std::expected<PieceBuildSummary, GridFailure>
ComputeReplayedBuild(const StaggeredGrid& whole, const PieceBuildInputs& inputs, ReplayLaunch launch) {
  const GridTables whole_tables = whole.tables(RunOn::kCpu);
  const std::expected<void, GridFailure> checked =
      CheckPieceRangeAndRoom(whole_tables.block_count, inputs.range, inputs.border.size(), inputs.max_cell_blocks);
  if (!checked.has_value()) {
    return std::unexpected(checked.error());
  }
  const PieceReplayResult replay = ComputeGridPieceReplay(whole_tables, inputs.range, inputs.border, launch);
  return ToPieceBuildSummary(replay.status, replay.loaded_count, static_cast<std::uint32_t>(inputs.border.size()));
}

// Checks that two failures have the same error, cells and magnitude.
void CheckSameFailure(const GridFailure& actual, const GridFailure& expected) {
  EXPECT_EQ(actual.error, expected.error);
  EXPECT_EQ(actual.cell.value, expected.cell.value);
  EXPECT_EQ(actual.other_cell.value, expected.other_cell.value);
  EXPECT_EQ(actual.magnitude, expected.magnitude);
}

// Checks that the replayed build and the CPU builder give the same result for the inputs: the same failure in every
// field, or the same counts and start of the range.
void CheckReplayedBuildEqualsCpuBuild(const StaggeredGrid& whole, const PieceBuildInputs& inputs, ReplayLaunch launch) {
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(inputs.max_cell_blocks);
  ASSERT_TRUE(room.has_value());
  const std::expected<GridPiece, GridFailure> piece = BuildGridPiece(whole, inputs.range, inputs.border, room->buffers);
  const std::expected<PieceBuildSummary, GridFailure> replayed = ComputeReplayedBuild(whole, inputs, launch);
  ASSERT_EQ(replayed.has_value(), piece.has_value());
  if (!piece.has_value()) {
    CheckSameFailure(replayed.error(), piece.error());
    return;
  }
  EXPECT_EQ(ToGridCountFields(replayed->counts), ToGridCountFields(piece->grid().counts()));
  EXPECT_EQ(replayed->range_start_in_piece, piece->range().first_block);
}

// Checks that one replay of the build of the piece of `range` and `border` has the tables and the piece map that the
// CPU builder gives, byte for byte, and the counts and the start of the range through the host's reading of the status.
void CheckReplayEqualsCpuPiece(
    const StaggeredGrid& whole,
    CellBlockRange range,
    std::span<const CellBlockIndex> border,
    ReplayLaunch launch
) {
  std::optional<PieceBuffersRoom> room =
      BuildPieceBuffersRoom(range.block_count + static_cast<std::uint32_t>(border.size()));
  ASSERT_TRUE(room.has_value());
  const std::expected<GridPiece, GridFailure> piece = BuildGridPiece(whole, range, border, room->buffers);
  ASSERT_TRUE(piece.has_value());
  const PieceReplayResult replay = ComputeGridPieceReplay(whole.tables(RunOn::kCpu), range, border, launch);
  ASSERT_EQ(replay.status.failure_key, kNoFailureKey);
  const std::optional<std::string> difference = FindFirstTableDifference(*piece, replay.tables);
  EXPECT_FALSE(difference.has_value()) << difference.value_or("");
  const PieceBuildInputs inputs = {
      .range = range,
      .border = border,
      .max_cell_blocks = range.block_count + static_cast<std::uint32_t>(border.size()),
  };
  CheckReplayedBuildEqualsCpuBuild(whole, inputs, launch);
}

// The steps replayed on the CPU, in every thread order, give the piece of the CPU builder: the tables, the map and
// the counts byte for byte, and the first block of the range, for every range of each grid and its near border and its
// stacked border.
TEST(BuildGridPiece, StepsReplayedInEveryThreadOrder_EqualCpuPiece) {
  for (const GridUnderTest& under_test : BuildGridsUnderTest(kEqualityGridCount, RangeChoice::kTestedRanges)) {
    for (const CellBlockRange range : under_test.ranges) {
      SCOPED_TRACE(testing::Message() << under_test.name << ", range from block " << range.first_block);
      const StackedBorder stacked = FindStackedBorder(*under_test.grid, range);
      for (const ReplayLaunch launch : kReplayLaunches) {
        CheckReplayEqualsCpuPiece(*under_test.grid, range, stacked.first_layer, launch);
        CheckReplayEqualsCpuPiece(*under_test.grid, range, stacked.border, launch);
      }
    }
  }
}

// The failure key of a replay for a border list that the CPU builder fails with kBadInput, and whether any array of the
// piece was written.
struct FailedReplay {
  int failure_step = 0;
  bool is_anything_written = false;
};

// Says whether every byte of `array` is 0xFF, what no step wrote.
template <typename Value>
bool ComputeIsUnwritten(const std::vector<Value>& array) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(array.data());
  for (std::size_t byte = 0; byte < array.size() * sizeof(Value); ++byte) {
    if (bytes[byte] != 0xFF) {
      return false;
    }
  }
  return true;
}

FailedReplay FindFailureOfReplay(
    const StaggeredGrid& whole,
    CellBlockRange range,
    std::span<const CellBlockIndex> border,
    ReplayLaunch launch
) {
  const PieceReplayResult replay = ComputeGridPieceReplay(whole.tables(RunOn::kCpu), range, border, launch);
  const PieceTables& tables = replay.tables;
  const bool is_all_unwritten = ComputeIsUnwritten(tables.block_in_whole) && ComputeIsUnwritten(tables.block_ids)
      && ComputeIsUnwritten(tables.cell_states) && ComputeIsUnwritten(tables.face_piece_kinds)
      && ComputeIsUnwritten(tables.same_size_level_blocks);
  return {.failure_step = ToFailureStep(replay.status.failure_key), .is_anything_written = !is_all_unwritten};
}

// Checks that the CPU builder fails with kBadInput for `border`, that the replay fails at step 1 of the failure keys
// (the border) in every launch and writes nothing in the arrays, and that the replayed build ends in the CPU's failure.
void CheckBadBorderFailsLikeCpuBuilder(const StaggeredGrid& whole, std::span<const CellBlockIndex> border) {
  const CellBlockRange range = {.first_block = 4, .block_count = 3};
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(10);
  ASSERT_TRUE(room.has_value());
  const std::expected<GridPiece, GridFailure> piece = BuildGridPiece(whole, range, border, room->buffers);
  ASSERT_FALSE(piece.has_value());
  EXPECT_EQ(piece.error().error, GridError::kBadInput);
  for (const ReplayLaunch launch : kReplayLaunches) {
    const FailedReplay failed = FindFailureOfReplay(whole, range, border, launch);
    EXPECT_EQ(failed.failure_step, kPieceBorderStep);
    EXPECT_FALSE(failed.is_anything_written);
    const PieceBuildInputs inputs = {.range = range, .border = border, .max_cell_blocks = 10};
    CheckReplayedBuildEqualsCpuBuild(whole, inputs, launch);
  }
}

// A border list that is not ascending, one that holds a block of the range, one that repeats a block and one with a
// block beyond the grid.
TEST(BuildGridPiece, StepsReplayedWithBadBorder_FailLikeCpuBuilder) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(10);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<std::vector<CellBlockIndex>> bad_borders = {
      ToBlocks({2, 1, 8}),
      ToBlocks({1, 5, 8}),
      ToBlocks({1, 8, 8}),
      ToBlocks({1, 2, 10}),
  };
  for (const std::vector<CellBlockIndex>& border : bad_borders) {
    SCOPED_TRACE(border.back().value);
    CheckBadBorderFailsLikeCpuBuilder(test_grid->grid, border);
  }
}

// A range past the grid, a room too small, both at once, and a bad border list with too little room. The checks of the
// range and of the room come first on both processors, in the order range, room, border list.
TEST(BuildGridPiece, StepsReplayedWithBadRangeOrRoom_FailLikeCpuBuilder) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(10);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::vector<CellBlockIndex> border = ToBlocks({1, 2, 8});
  const std::vector<CellBlockIndex> bad_border = ToBlocks({2, 1, 8});
  struct Case {
    PieceBuildInputs inputs;
    GridError expected_error = GridError::kNone;
  };
  const auto make_case = [](CellBlockRange range,
                            std::span<const CellBlockIndex> border_list,
                            std::uint32_t max_cell_blocks,
                            GridError expected_error) {
    return Case{
        .inputs = {.range = range, .border = border_list, .max_cell_blocks = max_cell_blocks},
        .expected_error = expected_error,
    };
  };
  const std::array<Case, 5> cases = {
      make_case({.first_block = 8, .block_count = 3}, border, 10, GridError::kBadInput),
      make_case({.first_block = 4, .block_count = 3}, border, 5, GridError::kTooManyCellBlocks),
      make_case({.first_block = 9, .block_count = 3}, border, 5, GridError::kBadInput),
      make_case({.first_block = 4, .block_count = 3}, bad_border, 5, GridError::kTooManyCellBlocks),
      make_case({.first_block = 4, .block_count = 3}, border, 6, GridError::kNone),
  };
  for (const Case& current : cases) {
    SCOPED_TRACE(
        testing::Message() << "range from block " << current.inputs.range.first_block << ", room "
                           << current.inputs.max_cell_blocks
    );
    std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(current.inputs.max_cell_blocks);
    ASSERT_TRUE(room.has_value());
    const std::expected<GridPiece, GridFailure> piece =
        BuildGridPiece(test_grid->grid, current.inputs.range, current.inputs.border, room->buffers);
    EXPECT_EQ(piece.has_value() ? GridError::kNone : piece.error().error, current.expected_error);
    for (const ReplayLaunch launch : kReplayLaunches) {
      CheckReplayedBuildEqualsCpuBuild(test_grid->grid, current.inputs, launch);
    }
  }
}

}  // namespace
}  // namespace fluidity::grid
