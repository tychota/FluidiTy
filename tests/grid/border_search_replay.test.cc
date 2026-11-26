#include "fluidity/grid/testing/border_search_replay.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/cell_block_range_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"
#include "fluidity/grid/testing/thread_replay.h"

namespace fluidity::grid {
namespace {

// The random grids replayed, and the blocks of a range of each kind of grid.
constexpr std::size_t kReplayGridCount = 10;
constexpr std::uint32_t kRangeBlocks = 4;
constexpr std::uint32_t kNestedBallRangeBlocks = 16;

// A small marking launch, so the sets of these grids take several launches as a large set does.
constexpr std::uint32_t kSmallSetBlocksPerLaunch = 3;
// The grid of many blocks, each holding two cells, and the range that is more than one marking launch of the GPU.
constexpr std::int32_t kManyBlocksGridBlockCount = 4097;
constexpr std::uint32_t kManyBlocksRangeBlocks = 3000;

// The launches every layer is replayed with: the GPU's own launch in forward and in reverse order, and a launch of
// 1,000 threads, so each thread walks several items as the grid-stride loop of a GPU function does.
constexpr std::array<ReplayLaunch, 3> kReplayLaunches = {
    ReplayLaunch{.order = LaunchOrder::kForward, .thread_count = 0},
    ReplayLaunch{.order = LaunchOrder::kReverse, .thread_count = 0},
    ReplayLaunch{.order = LaunchOrder::kReverse, .thread_count = 1000},
};

// The nested-ball grid, built once per program.
const TestGrid& FindNestedBallGrid() {
  static const TestGrid built = [] {
    const std::vector<CellId> cells = BuildNestedBallCells();
    std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(4), cells));
    EXPECT_TRUE(test_grid.has_value());
    return std::move(*test_grid);
  }();
  return built;
}

std::vector<std::uint32_t> ToBlockNumberList(std::span<const CellBlockIndex> blocks) {
  std::vector<std::uint32_t> numbers;
  for (const CellBlockIndex block : blocks) {
    numbers.push_back(block.value);
  }
  return numbers;
}

// The list a replay wrote: its first `count` entries, or the whole list when the count is above the room.
std::vector<std::uint32_t> ToWrittenBlockNumbers(const BorderReplayResult& replay) {
  const std::size_t written = std::min<std::size_t>(replay.count, replay.border_blocks.size());
  return ToBlockNumberList(std::span<const CellBlockIndex>(replay.border_blocks.data(), written));
}

// The status record that a replay left: its failure key and the count it wrote, as the GPU's last launch writes them.
GridBuildStatus ToReplayStatus(const BorderReplayResult& replay) {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.failure_key = replay.failure_key;
  status.result_count = replay.count;
  return status;
}

// Replays one call with `inputs` in `launch` and expects the CPU form's list and count, entry for entry.
void CheckReplayEqualsCpuList(
    const GridTables& tables,
    const BorderReplayInputs& inputs,
    ReplayLaunch launch,
    std::span<const CellBlockIndex> cpu_layer
) {
  SCOPED_TRACE(
      testing::Message() << "order " << static_cast<int>(launch.order) << ", threads " << launch.thread_count
                         << ", blocks per marking launch " << inputs.set_blocks_per_marking_launch
  );
  const BorderReplayResult replay = ComputeBorderSearchReplay(tables, inputs, launch, MarksAtStart::kCleared);
  EXPECT_EQ(replay.failure_key, kNoFailureKey);
  const std::expected<std::uint32_t, GridFailure> read_count =
      ToBorderBlockCount(ToReplayStatus(replay), tables.block_count);
  ASSERT_TRUE(read_count.has_value());
  EXPECT_EQ(*read_count, cpu_layer.size());
  EXPECT_EQ(replay.count, cpu_layer.size());
  EXPECT_EQ(replay.prefix_total, cpu_layer.size());
  EXPECT_EQ(ToWrittenBlockNumbers(replay), ToBlockNumberList(cpu_layer));
}

// Replays one layer of a stacked border in every launch, at the GPU's own marking launch size and again split into
// launches of 3 blocks of the set (in the first and the last launch shape), and expects the CPU form's list and count.
void CheckReplayEqualsCpuLayer(
    const StaggeredGrid& grid,
    CellBlockRange range,
    std::span<const CellBlockIndex> border_so_far,
    BorderWidth width,
    std::span<const CellBlockIndex> cpu_layer
) {
  const GridTables tables = grid.tables(RunOn::kCpu);
  BorderReplayInputs inputs = {
      .range = range,
      .border_so_far = border_so_far,
      .width = width,
      .border_block_room = static_cast<std::uint32_t>(cpu_layer.size()),
  };
  for (const ReplayLaunch launch : kReplayLaunches) {
    CheckReplayEqualsCpuList(tables, inputs, launch, cpu_layer);
  }
  inputs.set_blocks_per_marking_launch = kSmallSetBlocksPerLaunch;
  CheckReplayEqualsCpuList(tables, inputs, kReplayLaunches.front(), cpu_layer);
  CheckReplayEqualsCpuList(tables, inputs, kReplayLaunches.back(), cpu_layer);
}

// Replays every layer of the stacked border of `range`, the layer j on the list of the layer j - 1, and expects the CPU
// form's. Returns whether some layer was found with its marks left as found a different list, so a caller can show
// that the clearing of the marks matters.
bool CheckStackedLayers(BorderSearchRig& rig, CellBlockRange range, std::span<const BorderWidth> layer_widths) {
  bool is_stale_marks_visible = false;
  std::vector<CellBlockIndex> border_so_far;
  for (std::size_t layer = 0; layer < layer_widths.size(); ++layer) {
    SCOPED_TRACE(testing::Message() << "layer " << layer);
    const BorderWidth width = layer_widths[layer];
    const std::vector<CellBlockIndex> cpu_layer = FindLayerOrFail(rig, range, border_so_far, width);
    CheckReplayEqualsCpuLayer(rig.grid, range, border_so_far, width, cpu_layer);
    const BorderReplayInputs stale_inputs = {
        .range = range,
        .border_so_far = border_so_far,
        .width = width,
        .border_block_room = rig.grid.counts().cell_blocks,
    };
    const BorderReplayResult stale = ComputeBorderSearchReplay(
        rig.grid.tables(RunOn::kCpu),
        stale_inputs,
        kReplayLaunches[0],
        MarksAtStart::kLeftAsFound
    );
    is_stale_marks_visible = is_stale_marks_visible || ToWrittenBlockNumbers(stale) != ToBlockNumberList(cpu_layer);
    border_so_far = cpu_layer;
  }
  return is_stale_marks_visible;
}

// Every kind of range of the first random grids, at four layers, in every launch: the replay is the CPU form's list
// and count, and marks left from an earlier call show up in some list.
TEST(FindBorderBlocks, EveryThreadBlockAndThread_EqualsCpuStackedLayers) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_GE(built.grids.size(), kReplayGridCount);
  bool is_stale_marks_visible = false;
  for (std::size_t grid = 0; grid < kReplayGridCount; ++grid) {
    const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
    std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(staggered_grid);
    ASSERT_TRUE(rig.has_value());
    for (const CellBlockRange range : BuildTestedRanges(staggered_grid, kRangeBlocks)) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
      is_stale_marks_visible = CheckStackedLayers(*rig, range, kLayerWidths) || is_stale_marks_visible;
    }
  }
  EXPECT_TRUE(is_stale_marks_visible) << "no list showed the marks left as found, so the clearing is not tested";
}

// The nested-ball grid, ranges of 16 blocks, two layers (touching, four widths).
TEST(FindBorderBlocks, EveryThreadOnNestedBall_EqualsCpuStackedLayers) {
  const TestGrid& nested_ball = FindNestedBallGrid();
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(nested_ball.grid);
  ASSERT_TRUE(rig.has_value());
  for (const CellBlockRange range : BuildNestedBallTestedRanges(nested_ball.grid, kNestedBallRangeBlocks)) {
    SCOPED_TRACE(testing::Message() << "nested ball, range from block " << range.first_block);
    CheckStackedLayers(*rig, range, kNestedBallLayerWidths);
  }
}

// A grid of 4,097 blocks, a range of 3,000 of them (more than the 2,048 blocks of one marking launch of the GPU) and a
// second layer on top of it: the replay, which splits the marking at the GPU's own chunk size (and at 3 blocks), equals
// the CPU form's lists.
TEST(FindBorderBlocks, EveryThreadOnMoreBlocksThanOneLaunch_EqualsCpuStackedLayers) {
  const std::vector<CellId> cells = BuildTouchingCellsInManyBlocks(kManyBlocksGridBlockCount);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(test_grid.has_value());
  ASSERT_EQ(test_grid->grid.counts().cell_blocks, static_cast<std::uint32_t>(kManyBlocksGridBlockCount));
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const CellBlockRange range = {.first_block = 0, .block_count = kManyBlocksRangeBlocks};
  ASSERT_GT(range.block_count, kSetBlocksPerMarkingLaunch);
  const std::vector<CellBlockIndex> first_layer = FindLayerOrFail(*rig, range, {}, BorderWidth::kTouchingCells);
  ASSERT_FALSE(first_layer.empty()) << "the border must hold a block, or the comparison shows nothing";
  const std::array<BorderWidth, 2> layer_widths = {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths};
  CheckStackedLayers(*rig, range, layer_widths);
}

// Expects the status record a failed replay left to read as bad input, the way the search on the GPU reads it.
void CheckReplayStatusIsBadInput(const BorderReplayResult& replay, std::uint32_t block_count) {
  const std::expected<std::uint32_t, GridFailure> read_count = ToBorderBlockCount(ToReplayStatus(replay), block_count);
  ASSERT_FALSE(read_count.has_value());
  EXPECT_EQ(read_count.error().error, GridError::kBadInput);
}

// Expects the replay of one bad border so far, in every launch, to fail as the CPU form's call does: a count of 0, no
// list written, and a failure key.
void CheckBadBorderSoFarFails(
    const StaggeredGrid& grid,
    CellBlockRange range,
    std::span<const CellBlockIndex> bad_list
) {
  const std::uint32_t block_count = grid.counts().cell_blocks;
  for (const ReplayLaunch launch : kReplayLaunches) {
    const BorderReplayInputs inputs = {
        .range = range,
        .border_so_far = bad_list,
        .width = BorderWidth::kTouchingCells,
        .border_block_room = block_count,
    };
    const BorderReplayResult replay =
        ComputeBorderSearchReplay(grid.tables(RunOn::kCpu), inputs, launch, MarksAtStart::kCleared);
    EXPECT_NE(replay.failure_key, kNoFailureKey);
    CheckReplayStatusIsBadInput(replay, block_count);
    EXPECT_EQ(replay.count, 0U);
    EXPECT_EQ(replay.prefix_total, 0U);
    EXPECT_TRUE(std::ranges::all_of(replay.border_blocks, [](CellBlockIndex block) {
      return block.value == kNoIndex;
    }));
  }
}

// Expects the CPU form to fail with kBadInput on `bad_list` and the replay, in every launch, to fail too.
void CheckBadListFailsInBothForms(
    BorderSearchRig& rig,
    CellBlockRange range,
    std::span<const CellBlockIndex> bad_list
) {
  ASSERT_TRUE(WriteBuffer<CellBlockIndex>(bad_list, rig.so_far_buffer, 0).has_value());
  const std::expected<std::uint32_t, GridFailure> cpu_count = FindBorderBlocks(
      rig.grid,
      range,
      rig.so_far_buffer,
      static_cast<std::uint32_t>(bad_list.size()),
      BorderWidth::kTouchingCells,
      rig.search_buffers,
      rig.border_buffer
  );
  ASSERT_FALSE(cpu_count.has_value());
  EXPECT_EQ(cpu_count.error().error, GridError::kBadInput);
  CheckBadBorderSoFarFails(rig.grid, range, bad_list);
}

// A border so far that is not ascending, holds a block of the range, or holds a block beyond the grid fails in the
// replay as it does in the CPU form: no list, a count of 0, and the failure key of the first bad entry.
TEST(FindBorderBlocks, EveryThreadOnBadBorderSoFar_EqualsCpuFailures) {
  const StaggeredGrid& grid = FindRandomGridsWithCheckers().grids.at(0).grid;
  const std::uint32_t block_count = grid.counts().cell_blocks;
  ASSERT_GE(block_count, 8U);
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(grid);
  ASSERT_TRUE(rig.has_value());
  const CellBlockRange range = {.first_block = 2, .block_count = 2};
  const std::vector<std::vector<CellBlockIndex>> bad_lists = {
      {CellBlockIndex{5}, CellBlockIndex{4}},
      {CellBlockIndex{1}, CellBlockIndex{3}},
      {CellBlockIndex{1}, CellBlockIndex{block_count}},
      {CellBlockIndex{5}, CellBlockIndex{5}},
  };
  for (const std::vector<CellBlockIndex>& bad_list : bad_lists) {
    SCOPED_TRACE(ToBlockNumberList(bad_list).back());
    CheckBadListFailsInBothForms(*rig, range, bad_list);
  }
}

}  // namespace
}  // namespace fluidity::grid
