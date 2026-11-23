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
#include "fluidity/grid/grid_piece_steps.h"
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

// The counts of the record of a replay, as a GridCounts: the blocks, the active cells of each size level, the active
// cells (their sum) and the face pieces.
GridCounts ToCountsOfReplay(const PieceReplayResult& replay) {
  GridCounts counts{};
  counts.cell_blocks = replay.loaded_count;
  for (std::size_t size_level = 0; size_level < kMaxSizeLevels; ++size_level) {
    counts.active_cells_per_size_level[size_level] = replay.status.active_cells_per_size_level[size_level];
    counts.active_cells += replay.status.active_cells_per_size_level[size_level];
  }
  counts.face_pieces = replay.status.face_pieces;
  return counts;
}

// Checks that one replay of the build of the piece of `range` and `border` has the tables, the piece map, the counts
// and the start of the range that the CPU builder gives, byte for byte.
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
  EXPECT_EQ(ToGridCountFields(ToCountsOfReplay(replay)), ToGridCountFields(piece->grid().counts()));
  EXPECT_EQ(replay.status.result_count, piece->range().first_block);
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

// Checks that the CPU builder fails with kBadInput for `border` and that the replay fails at step 1 of the failure keys
// (the border) in every launch and writes nothing in the arrays.
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

}  // namespace
}  // namespace fluidity::grid
