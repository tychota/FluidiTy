#include "fluidity/grid/grid_cell_checks.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// The room of the CPU buffers that every random grid fits: at most 252 cell blocks each.
constexpr std::uint32_t kRandomGridRoom = 4096;

// A frame with its origin at 0 and biggest cells 1 m wide.
GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

std::vector<std::uint64_t> ToValues(std::span<const CellId> cells) {
  std::vector<std::uint64_t> values;
  values.reserve(cells.size());
  for (const CellId cell : cells) {
    values.push_back(cell.value);
  }
  return values;
}

// The list in the fixed order, sorted with ComesBefore: the test's own sort, apart from the sort keys.
std::vector<CellId> ToSortedCells(std::vector<CellId> cells) {
  std::ranges::sort(cells, [](CellId first, CellId second) {
    return ComesBefore(first, second);
  });
  return cells;
}

int ToSizeLevelNumber(CellId cell) {
  return static_cast<int>(ToSizeLevel(cell));
}

// floor(coord / 2^size_level_steps) by integer division and a correction for a negative remainder: the test's own
// arithmetic, apart from the shift the grid uses.
//
// clang-tidy: a coordinate and a count of size levels; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::int32_t ToFloorOfHalving(std::int32_t coord, int size_level_steps) {
  const auto divisor = static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(size_level_steps));
  std::int64_t quotient = coord / divisor;
  if (coord % divisor != 0 && coord < 0) {
    quotient -= 1;
  }
  return static_cast<std::int32_t>(quotient);
}

// The big cell, one size level up, that holds `cell`, by the test's own floor.
CellId ToBigCellByDivision(CellId cell) {
  const CellCoord coord = ToCellCoord(cell);
  return ToCell(
      ToSizeLevelNumber(cell) - 1,
      ToFloorOfHalving(coord.x, 1),
      ToFloorOfHalving(coord.y, 1),
      ToFloorOfHalving(coord.z, 1)
  );
}

// The 8 small cells of a big cell, by the test's own arithmetic: 2 c and 2 c + 1 on each axis.
std::vector<CellId> ToSmallCells(CellId big_cell) {
  const CellCoord coord = ToCellCoord(big_cell);
  const int small_size_level = ToSizeLevelNumber(big_cell) + 1;
  std::vector<CellId> small_cells;
  for (std::int32_t dz = 0; dz < 2; ++dz) {
    for (std::int32_t dy = 0; dy < 2; ++dy) {
      for (std::int32_t dx = 0; dx < 2; ++dx) {
        small_cells.push_back(ToCell(small_size_level, (2 * coord.x) + dx, (2 * coord.y) + dy, (2 * coord.z) + dz));
      }
    }
  }
  return small_cells;
}

// A CPU RunMemory with grid buffers and the builder buffers lent to them. Kept behind a pointer, since lent builder
// buffers must not move; the members are destroyed in reverse order, the grid buffers first.
struct CpuGridBuffers {
  std::unique_ptr<RunMemory> run_memory;
  std::unique_ptr<GridBuilderBuffers> builder_buffers;
  std::unique_ptr<GridBuffers> buffers;
};

// Builds CPU grid buffers of max_cell_blocks blocks in a RunMemory of exactly their estimate. Null, with a test
// failure, when a builder fails.
std::unique_ptr<CpuGridBuffers> BuildCpuGridBuffers(std::uint32_t max_cell_blocks) {
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, max_cell_blocks);
  GridCounts room_counts = {};
  room_counts.cell_blocks = max_cell_blocks;
  const MemoryCounts memory_counts = ToMemoryCounts(room_counts, 0);
  // A CPU run keeps its kRunOn and its kCpu entries in one CPU block.
  const std::size_t byte_count = ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kRunOn)
      + ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kCpu);
  auto built = std::make_unique<CpuGridBuffers>();
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, byte_count);
  if (!run_memory.has_value()) {
    ADD_FAILURE() << "BuildRunMemory failed";
    return nullptr;
  }
  built->run_memory = std::make_unique<RunMemory>(std::move(*run_memory));
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, max_cell_blocks, *built->run_memory);
  if (!builder_buffers.has_value()) {
    ADD_FAILURE() << "BuildGridBuilderBuffers failed";
    return nullptr;
  }
  built->builder_buffers = std::make_unique<GridBuilderBuffers>(std::move(*builder_buffers));
  const GridSettings settings = {.max_cell_blocks = max_cell_blocks, .run_on = RunOn::kCpu};
  std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers(settings, *built->builder_buffers, *built->run_memory);
  if (!buffers.has_value()) {
    ADD_FAILURE() << "BuildGridBuffers failed";
    return nullptr;
  }
  built->buffers = std::make_unique<GridBuffers>(std::move(*buffers));
  return built;
}

// The inputs of a CPU list of version 1.
GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

// The first `count` cell ids of builder buffers, as values: the part a check of `count` cells uses.
std::vector<std::uint64_t> ReadLentCellIds(const GridBuilderBuffers& builder_buffers, std::size_t count) {
  const Buffer<CellId>& cell_ids = ToGridBuilderBuffersArrays(builder_buffers).cell_ids;
  return ToValues(std::span<const CellId>(cell_ids.data(), std::min(count, cell_ids.size())));
}

void ExpectFailure(
    const std::expected<void, GridFailure>& result,
    GridError error,
    CellId cell = CellId{0},
    CellId other_cell = CellId{0}
) {
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, error);
  EXPECT_EQ(result.error().cell.value, cell.value);
  EXPECT_EQ(result.error().other_cell.value, other_cell.value);
  EXPECT_EQ(result.error().magnitude, 0U);
}

// Every field of a result, as one comparable value: whether it passed, then the failure's error, cells and magnitude
// (all zero on success).
std::tuple<bool, GridError, std::uint64_t, std::uint64_t, std::uint64_t> ToResultFields(
    const std::expected<void, GridFailure>& result
) {
  const GridFailure failure = result.has_value() ? GridFailure{} : result.error();
  return {result.has_value(), failure.error, failure.cell.value, failure.other_cell.value, failure.magnitude};
}

void ExpectSameResult(
    const std::expected<void, GridFailure>& actual,
    const std::expected<void, GridFailure>& expected
) {
  EXPECT_EQ(ToResultFields(actual), ToResultFields(expected));
}

// Size level 0 {0,0,0}, which covers [0, 1) m on each axis, and a size-level-2 cell 0.25 m wide just beyond its high
// side: {4,4,4} covers [1, 1.25) on each axis and touches it at the corner (1, 1, 1); {4,4,0} along the edge x = y = 1;
// {4,0,0} on the side x = 1. Each pair is two size levels apart.
std::vector<CellId> BuildTwoSizeLevelsApart(CellId small_cell) {
  return {ToCell(0, 0, 0, 0), small_cell};
}

std::array<std::vector<CellId>, 3> BuildEveryTwoSizeLevelsApartList() {
  return {
      BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 4)),
      BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 0)),
      BuildTwoSizeLevelsApart(ToCell(2, 4, 0, 0)),
  };
}

// The 7 size-level-1 cells of size level 0 {0,0,0} without its eighth 5, size level 1 {1,0,1}.
std::vector<CellId> BuildSevenOfEightSmallCells() {
  std::vector<CellId> cells = ToSmallCells(ToCell(0, 0, 0, 0));
  cells.erase(cells.begin() + 5);
  return cells;
}

// Size level 0 {0,0,0} split, its eighth 0, size level 1 {0,0,0}, missing, and each of its 7 other size-level-1 cells
// split into its 8 size-level-2 cells: 56 active cells.
std::vector<CellId> BuildMissingCellBesideSplitSiblings() {
  std::vector<CellId> cells;
  const std::vector<CellId> size_level_one_cells = ToSmallCells(ToCell(0, 0, 0, 0));
  for (std::size_t cell_eighth = 1; cell_eighth < 8; ++cell_eighth) {
    const std::vector<CellId> small_cells = ToSmallCells(size_level_one_cells.at(cell_eighth));
    cells.insert(cells.end(), small_cells.begin(), small_cells.end());
  }
  return cells;
}

// `count` size-level-0 cells along x from 0: a valid list of one size level.
std::vector<CellId> BuildBigCellsAlongX(std::int32_t count) {
  std::vector<CellId> cells;
  cells.reserve(static_cast<std::size_t>(count));
  for (std::int32_t x = 0; x < count; ++x) {
    cells.push_back(ToCell(0, x, 0, 0));
  }
  return cells;
}

// ---------------------------------------------------------------------------------------------------------------------
// The span form
// ---------------------------------------------------------------------------------------------------------------------

// Size level 2 {4,4,4} is the smaller cell of the pair and names size level 0 {0,0,0}, two size levels bigger.
TEST(CheckSizeStepRule, CornerTouchTwoSizeLevelsApart_IsSizeStepRuleBroken) {
  const std::vector<CellId> cells = BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 4));
  ExpectFailure(
      CheckSizeStepRule(ToUnitFrame(3), cells),
      GridError::kSizeStepRuleBroken,
      ToCell(2, 4, 4, 4),
      ToCell(0, 0, 0, 0)
  );
}

TEST(CheckSizeStepRule, EdgeTouchTwoSizeLevelsApart_IsSizeStepRuleBroken) {
  const std::vector<CellId> cells = BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 0));
  ExpectFailure(
      CheckSizeStepRule(ToUnitFrame(3), cells),
      GridError::kSizeStepRuleBroken,
      ToCell(2, 4, 4, 0),
      ToCell(0, 0, 0, 0)
  );
}

TEST(CheckSizeStepRule, FaceTouchTwoSizeLevelsApart_IsSizeStepRuleBroken) {
  const std::vector<CellId> cells = BuildTwoSizeLevelsApart(ToCell(2, 4, 0, 0));
  ExpectFailure(
      CheckSizeStepRule(ToUnitFrame(3), cells),
      GridError::kSizeStepRuleBroken,
      ToCell(2, 4, 0, 0),
      ToCell(0, 0, 0, 0)
  );
}

// Size level 1 {2,0,0} covers [1, 1.5) x [0, 0.5) x [0, 0.5): it touches size level 0 {0,0,0} on the side x = 1, one
// size level apart. Its 7 siblings are missing, which this check does not look at.
TEST(CheckSizeStepRule, OneSizeLevelApart_Passes) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), ToCell(1, 2, 0, 0)};
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(3), cells).has_value());
}

TEST(CheckSizeStepRule, SevenOfEightSmallCells_Passes) {
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(3), BuildSevenOfEightSmallCells()).has_value());
}

// The hidden missing sibling is a sibling failure, which this check leaves out; the rest is a valid grid.
TEST(CheckSizeStepRule, MissingCellBesideSplitSiblings_Passes) {
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(3), BuildMissingCellBesideSplitSiblings()).has_value());
}

// Input {a, a, bad1, bad2}: step 1 fails at input indexes 2 and 3, keys (1, 2, 0) and (1, 3, 0); the overlap of a with
// itself is step 2, key (2, 1, 0). The smallest key is (1, 2, 0): bad1, named as given, with bit 63 clear.
TEST(CheckSizeStepRule, BadIdAfterRepeat_IsBadInputOfFirstBadId) {
  const CellId cell = ToCell(0, 0, 0, 0);
  const CellId first_bad_id = {.value = cell.value & ~(std::uint64_t{1} << 63U)};
  const CellId second_bad_id = {.value = 0};
  const std::vector<CellId> cells = {cell, cell, first_bad_id, second_bad_id};
  ExpectFailure(CheckSizeStepRule(ToUnitFrame(3), cells), GridError::kBadInput, first_bad_id);
}

// Two ids with bit 61 set keep ordinary sort keys, so sorting puts Y = size level 0 {0,0,0} before X = size level 0
// {10,0,0}. Step 1 runs in input order, before the sort: input index 0, X, has key (1, 0, 0) and is named as given.
std::vector<CellId> BuildBadIdsThatSortingReorders() {
  constexpr std::uint64_t kBitSixtyOne = std::uint64_t{1} << 61U;
  return {
      CellId{.value = ToCell(0, 10, 0, 0).value | kBitSixtyOne},
      CellId{.value = ToCell(0, 0, 0, 0).value | kBitSixtyOne},
  };
}

TEST(CheckSizeStepRule, BadIdsThatSortingReorders_IsBadInputOfFirstInInputOrder) {
  const std::vector<CellId> cells = BuildBadIdsThatSortingReorders();
  ExpectFailure(CheckSizeStepRule(ToUnitFrame(3), cells), GridError::kBadInput, cells.at(0));
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  ExpectFailure(
      CheckSizeStepRule(ToInputs(ToUnitFrame(3), cells), *grid_buffers->buffers),
      GridError::kBadInput,
      cells.at(0)
  );
}

// kMaxCoord + 1 is the one coordinate outside the range that an id can hold.
TEST(CheckSizeStepRule, CoordBeyondLimit_IsCoordOutOfRange) {
  const CellId beyond_limit = ToCell(0, kMaxCoord + 1, 0, 0);
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0), beyond_limit};
  ExpectFailure(CheckSizeStepRule(ToUnitFrame(3), cells), GridError::kCoordOutOfRange, beyond_limit);
}

TEST(CheckSizeStepRule, BigCellsAtBothCoordLimits_Pass) {
  const std::vector<CellId> cells = {
      ToCell(0, kMaxCoord, kMaxCoord, kMaxCoord),
      ToCell(0, -kMaxCoord, -kMaxCoord, -kMaxCoord),
      ToCell(0, kMaxCoord, -kMaxCoord, 0),
  };
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(3), cells).has_value());
}

// Size level 2 {kMaxCoord, 0, 0} covers x in [65535.75, 65536) m; size level 0 {65536, 0, 0} begins at x = 65536 m, on
// its high side, two size levels bigger. The place beside the small cell, x = 2^18, lies beyond the range at size
// level 2 and is found at size level 0.
TEST(CheckSizeStepRule, SmallCellAtHighCoordLimitBesideBigCell_IsSizeStepRuleBroken) {
  const std::vector<CellId> cells = {ToCell(0, 65536, 0, 0), ToCell(2, kMaxCoord, 0, 0)};
  ExpectFailure(
      CheckSizeStepRule(ToUnitFrame(3), cells),
      GridError::kSizeStepRuleBroken,
      ToCell(2, kMaxCoord, 0, 0),
      ToCell(0, 65536, 0, 0)
  );
}

TEST(CheckSizeStepRule, BadFrame_IsBadInput) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  GridFrame origin_not_finite = ToUnitFrame(3);
  origin_not_finite.origin.y = std::numeric_limits<double>::quiet_NaN();
  GridFrame width_zero = ToUnitFrame(3);
  width_zero.biggest_cell_width = 0;
  GridFrame width_not_finite = ToUnitFrame(3);
  width_not_finite.biggest_cell_width = std::numeric_limits<double>::infinity();
  for (const GridFrame frame : {origin_not_finite, width_zero, width_not_finite, ToUnitFrame(0), ToUnitFrame(17)}) {
    ExpectFailure(CheckSizeStepRule(frame, cells), GridError::kBadInput);
  }
}

// The frame of `size_level_count` size levels whose smallest width, the biggest halved size_level_count - 1 times, is
// smallest_width m.
GridFrame ToFrameOfSmallestWidth(int size_level_count, double smallest_width) {
  GridFrame frame = ToUnitFrame(size_level_count);
  frame.biggest_cell_width = std::ldexp(smallest_width, size_level_count - 1);
  return frame;
}

// The smallest width is checked against 2^-1022, the smallest normal double: exactly 2^-1022 passes, one halving below
// is kBadInput, for 1, 2, 8 and 16 size levels. With 16 size levels that is biggest cells of 2^-1007 m and 2^-1008 m.
TEST(CheckSizeStepRule, SmallestWidthBelowNormalRange_IsBadInput) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  constexpr double kSmallestNormal = std::numeric_limits<double>::min();
  for (const int size_level_count : {1, 2, 8, 16}) {
    SCOPED_TRACE(testing::Message() << "size levels " << size_level_count);
    EXPECT_TRUE(CheckSizeStepRule(ToFrameOfSmallestWidth(size_level_count, kSmallestNormal), cells).has_value());
    ExpectFailure(
        CheckSizeStepRule(ToFrameOfSmallestWidth(size_level_count, kSmallestNormal / 2), cells),
        GridError::kBadInput
    );
  }
}

TEST(CheckSizeStepRule, EmptyList_Passes) {
  EXPECT_TRUE(CheckSizeStepRule(ToUnitFrame(1), std::span<const CellId>()).has_value());
}

// ---------------------------------------------------------------------------------------------------------------------
// The input checks of the buffers form
// ---------------------------------------------------------------------------------------------------------------------

TEST(GridInputs, TwoLists_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  const Buffer<CellId> cell_buffer = BuildBufferHolding(RunOn::kCpu, cells);
  GridInputs inputs = ToInputs(ToUnitFrame(1), cells);
  inputs.active_cell_buffer = &cell_buffer;
  inputs.active_cell_buffer_count = 1;
  ExpectFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

TEST(GridInputs, BufferCountAboveSize_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const Buffer<CellId> cell_buffer =
      BuildBufferHolding(RunOn::kCpu, std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(0, 1, 0, 0)});
  GridInputs inputs = ToInputs(ToUnitFrame(1), {});
  inputs.active_cell_buffer = &cell_buffer;
  inputs.active_cell_buffer_count = 3;
  ExpectFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

TEST(GridInputs, CountWithoutBuffer_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  GridInputs inputs = ToInputs(ToUnitFrame(1), {});
  inputs.active_cell_buffer_count = 1;
  ExpectFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

TEST(GridInputs, VersionZero_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  GridInputs inputs = ToInputs(ToUnitFrame(1), cells);
  inputs.version = {.value = 0};
  ExpectFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

TEST(GridInputs, BufferList_IsCheckedAsTheSpan) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  // Only the first 2 entries are the list: the third, two size levels from the first, is not read.
  const Buffer<CellId> cell_buffer =
      BuildBufferHolding(RunOn::kCpu, std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(1, 2, 0, 0), ToCell(2, 4, 4, 4)});
  GridInputs inputs = ToInputs(ToUnitFrame(3), {});
  inputs.active_cell_buffer = &cell_buffer;
  inputs.active_cell_buffer_count = 2;
  EXPECT_TRUE(CheckSizeStepRule(inputs, *grid_buffers->buffers).has_value());
  inputs.active_cell_buffer_count = 3;
  ExpectFailure(
      CheckSizeStepRule(inputs, *grid_buffers->buffers),
      GridError::kSizeStepRuleBroken,
      ToCell(2, 4, 4, 4),
      ToCell(0, 0, 0, 0)
  );
}

// The list is the lent builder buffers' own cell ids, which the sorted copy would overwrite while reading them.
// A first check leaves the valid list {size level 0 {0,0,0}, size level 1 {2,0,0}} sorted in the cell ids, so the
// aliased list holds valid ids and only the alias check can stop it (without it, base's copy would see the overlap).
TEST(CheckSizeStepRule, ListInLentCellIds_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = {ToCell(1, 2, 0, 0), ToCell(0, 0, 0, 0)};
  ASSERT_TRUE(CheckSizeStepRule(ToInputs(ToUnitFrame(3), cells), *grid_buffers->buffers).has_value());
  const Buffer<CellId>& cell_ids = ToGridBuilderBuffersArrays(*grid_buffers->builder_buffers).cell_ids;
  GridInputs inputs = ToInputs(ToUnitFrame(3), {});
  inputs.active_cell_buffer = &cell_ids;
  inputs.active_cell_buffer_count = 2;
  ExpectFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

// The arrays of the grid buffers point at no builder buffers, then at builder buffers that were never lent to them, of
// a smaller room and of the same room. The cell ids of the lent ones stay all zero, as built: nothing is written.
TEST(CheckSizeStepRule, BuilderBuffersNotTheLentOnes_IsBadInput) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(4);
  const std::unique_ptr<CpuGridBuffers> smaller_grid_buffers = BuildCpuGridBuffers(1);
  const std::unique_ptr<CpuGridBuffers> other_grid_buffers = BuildCpuGridBuffers(4);
  const bool are_all_built =
      grid_buffers != nullptr && smaller_grid_buffers != nullptr && other_grid_buffers != nullptr;
  ASSERT_TRUE(are_all_built);
  const std::vector<CellId> cells = {ToCell(1, 2, 0, 0), ToCell(0, 0, 0, 0)};
  const GridInputs inputs = ToInputs(ToUnitFrame(3), cells);
  GridBuffersArrays& arrays = ToGridBuffersArrays(*grid_buffers->buffers);
  GridBuilderBuffers* const lent_builder_buffers = arrays.builder_buffers;
  const std::vector<std::uint64_t> cell_ids_before = ReadLentCellIds(*lent_builder_buffers, cells.size());

  const std::array<GridBuilderBuffers*, 3> replacements = {
      nullptr,
      smaller_grid_buffers->builder_buffers.get(),
      other_grid_buffers->builder_buffers.get(),
  };
  for (GridBuilderBuffers* const builder_buffers : replacements) {
    arrays.builder_buffers = builder_buffers;
    ExpectFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
    EXPECT_EQ(ReadLentCellIds(*lent_builder_buffers, cells.size()), cell_ids_before);
  }

  // With the lent builder buffers back, the same call passes and writes the sorted copy.
  arrays.builder_buffers = lent_builder_buffers;
  EXPECT_TRUE(CheckSizeStepRule(inputs, *grid_buffers->buffers).has_value());
  EXPECT_EQ(ReadLentCellIds(*lent_builder_buffers, cells.size()).at(0), ToCell(0, 0, 0, 0).value);
}

// Room for 1 block is 64 cell places; 65 cells cannot fit, whatever they are.
TEST(CheckSizeStepRule, MoreCellsThanPlaces_IsTooManyCellBlocks) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = BuildBigCellsAlongX(65);
  ExpectFailure(
      CheckSizeStepRule(ToInputs(ToUnitFrame(1), cells), *grid_buffers->buffers),
      GridError::kTooManyCellBlocks
  );
}

TEST(CheckSizeStepRule, AsManyCellsAsPlaces_Passes) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  const std::vector<CellId> cells = BuildBigCellsAlongX(64);
  EXPECT_TRUE(CheckSizeStepRule(ToInputs(ToUnitFrame(1), cells), *grid_buffers->buffers).has_value());
}

// The length is an input check: it comes before step 1 reads the bad id at input index 0. The version comes before
// the length.
TEST(CheckSizeStepRule, TooLongListWithBadId_IsTooManyCellBlocks) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  std::vector<CellId> cells = BuildBigCellsAlongX(65);
  cells.at(0) = {.value = cells.at(0).value & ~(std::uint64_t{1} << 63U)};
  GridInputs inputs = ToInputs(ToUnitFrame(1), cells);
  ExpectFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kTooManyCellBlocks);
  inputs.version = {.value = 0};
  ExpectFailure(CheckSizeStepRule(inputs, *grid_buffers->buffers), GridError::kBadInput);
}

TEST(CheckSizeStepRule, IntoBuffers_EqualsSpanForm) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(kRandomGridRoom);
  ASSERT_NE(grid_buffers, nullptr);
  for (const std::vector<CellId>& cells : BuildEveryTwoSizeLevelsApartList()) {
    const std::expected<void, GridFailure> span_result = CheckSizeStepRule(ToUnitFrame(3), cells);
    ASSERT_FALSE(span_result.has_value());
    ExpectSameResult(CheckSizeStepRule(ToInputs(ToUnitFrame(3), cells), *grid_buffers->buffers), span_result);
  }
  for (const std::vector<CellId>& cells : BuildShuffledRandomGrids()) {
    const std::expected<void, GridFailure> span_result = CheckSizeStepRule(ToUnitFrame(4), cells);
    EXPECT_TRUE(span_result.has_value());
    ExpectSameResult(CheckSizeStepRule(ToInputs(ToUnitFrame(4), cells), *grid_buffers->buffers), span_result);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// BuildSortedCellList
// ---------------------------------------------------------------------------------------------------------------------

// Runs BuildSortedCellList and drops the count, for ExpectFailure.
std::expected<void, GridFailure>
CheckThroughSortedCellList(GridFrame frame, std::span<const CellId> cells, GridBuffers& buffers) {
  const std::expected<std::uint32_t, GridFailure> built =
      BuildSortedCellList(ToInputs(frame, cells), buffers, SiblingCheck::kRequire);
  if (!built.has_value()) {
    return std::unexpected(built.error());
  }
  return {};
}

TEST(BuildSortedCellList, ShuffledList_LeavesFixedCellOrder) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(kRandomGridRoom);
  ASSERT_NE(grid_buffers, nullptr);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    const std::vector<CellId>& cells = BuildShuffledRandomGrids().at(grid);
    const std::expected<std::uint32_t, GridFailure> built =
        BuildSortedCellList(ToInputs(ToUnitFrame(4), cells), *grid_buffers->buffers, SiblingCheck::kRequire);
    ASSERT_TRUE(built.has_value()) << grid;
    ASSERT_EQ(*built, cells.size()) << grid;
    const std::vector<std::uint64_t> sorted_values = ReadLentCellIds(*grid_buffers->builder_buffers, cells.size());
    EXPECT_EQ(sorted_values, ToValues(ToSortedCells(cells))) << grid;
  }
}

// Sorted, the first cell is size level 2 {2,0,0}, eighth 0 of size level 1 {1,0,0}; that cell is eighth 1 of size
// level 0 {0,0,0}. The siblings of {1,0,0} are all present; its walk then reaches size level 0 {0,0,0}, whose eighth 0,
// size level 1 {0,0,0}, is neither listed nor split into listed cells.
TEST(BuildSortedCellList, MissingCellBesideSplitSiblings_IsSiblingCellMissingOfThatCell) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(64);
  ASSERT_NE(grid_buffers, nullptr);
  ExpectFailure(
      CheckThroughSortedCellList(ToUnitFrame(3), BuildMissingCellBesideSplitSiblings(), *grid_buffers->buffers),
      GridError::kSiblingCellMissing,
      ToCell(1, 0, 0, 0),
      ToCell(0, 0, 0, 0)
  );
}

// 3 size levels, {size level 0 {0,0,0}, size level 2 {4,4,4}}. Sorted, {4,4,4} is at position 1 and fails two
// steps there. Step 3: its big cell is size level 1 {2,2,2}; eighth 0 is {4,4,4} itself, eighth 1 is size level 2
// {5,4,4} (x = 2 * 2 + 1), which is not listed: key (3, 1, 1). Step 4: the corner touch with {0,0,0} in direction 0:
// key (4, 1, 0). The smaller key is step 3's.
TEST(BuildSortedCellList, MissingSiblingAndSizeStepBreak_IsSiblingCellMissing) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(1);
  ASSERT_NE(grid_buffers, nullptr);
  ExpectFailure(
      CheckThroughSortedCellList(ToUnitFrame(3), BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 4)), *grid_buffers->buffers),
      GridError::kSiblingCellMissing,
      ToCell(2, 5, 4, 4),
      ToCell(1, 2, 2, 2)
  );
}

// Size level 0 {0,0,0} split, its size-level-1 cell {1,0,0} split too, and size level 2 {2,0,0}, eighth 0 of {1,0,0},
// left out. No active cell lies at the low corner of {1,0,0}, so {1,0,0} is not present among the siblings of
// {0,0,0}, and the walk from size level 1 {0,0,0}, at position 0, names it: {1,0,0} is eighth 1 (x odd), so it is the
// first cell above the one left out that is not eighth 0.
TEST(BuildSortedCellList, EighthZeroCellLeftOut_NamesFirstBiggerCellNotAtEighthZero) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(64);
  ASSERT_NE(grid_buffers, nullptr);
  std::vector<CellId> cells = ToSmallCells(ToCell(0, 0, 0, 0));
  cells.erase(cells.begin() + 1);
  for (const CellId small_cell : ToSmallCells(ToCell(1, 1, 0, 0))) {
    if (small_cell.value != ToCell(2, 2, 0, 0).value) {
      cells.push_back(small_cell);
    }
  }
  ExpectFailure(
      CheckThroughSortedCellList(ToUnitFrame(3), cells, *grid_buffers->buffers),
      GridError::kSiblingCellMissing,
      ToCell(1, 1, 0, 0),
      ToCell(0, 0, 0, 0)
  );
}

// 2 size levels. The 8 size-level-1 cells of size level 0 {-131071, 0, 0}, at x = -262142 and -262141, pass. The 4
// size-level-1 cells at x = -kMaxCoord = -262143, y and z 0 or 1, lie in size level 0 {-131072, 0, 0}, whose other 4
// small cells would be at x = -262144, outside the range. Sorted, x = -262143, y = z = 0 comes first; its big cell's
// eighth 0 is the first small cell not present, and no id names it.
TEST(BuildSortedCellList, SmallCellAtLowCoordLimit_IsSiblingCellMissingWithNoCell) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(64);
  ASSERT_NE(grid_buffers, nullptr);
  std::vector<CellId> cells = ToSmallCells(ToCell(0, -131071, 0, 0));
  ASSERT_TRUE(CheckThroughSortedCellList(ToUnitFrame(2), cells, *grid_buffers->buffers).has_value());
  for (std::int32_t z = 0; z < 2; ++z) {
    for (std::int32_t y = 0; y < 2; ++y) {
      cells.push_back(ToCell(1, -kMaxCoord, y, z));
    }
  }
  ExpectFailure(
      CheckThroughSortedCellList(ToUnitFrame(2), cells, *grid_buffers->buffers),
      GridError::kSiblingCellMissing,
      CellId{0},
      ToCell(0, -131072, 0, 0)
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// The random grids, each changed in one way. The expected cells come from the change itself and from whole-unit boxes
// of the test support. Each list is also checked in a second order, ApplyFixedShuffle of seed + 2000.
// ---------------------------------------------------------------------------------------------------------------------

constexpr int kRandomGridSizeLevels = 4;

// The first cell of a list from index `start` on, wrapping around, for which `is_wanted` holds; none when no cell does.
template <typename CellPredicate>
std::optional<std::size_t> FindCellFrom(const std::vector<CellId>& cells, std::size_t start, CellPredicate is_wanted) {
  for (std::size_t step = 0; step < cells.size(); ++step) {
    const std::size_t index = (start + step) % cells.size();
    if (is_wanted(cells.at(index))) {
      return index;
    }
  }
  return std::nullopt;
}

// A cell that has a big cell: size level 1 or more.
bool ComputeHasBigCell(CellId cell) {
  return ToSizeLevelNumber(cell) >= 1;
}

// Says whether a cell has an odd coordinate on some axis, so it is not the eighth-0 small cell of its big cell.
bool ComputeIsAwayFromBigCellCorner(CellId cell) {
  const CellCoord coord = ToCellCoord(cell);
  const auto is_odd = [](std::int32_t axis_coord) {
    return ToFloorOfHalving(axis_coord, 1) * 2 != axis_coord;
  };
  return is_odd(coord.x) || is_odd(coord.y) || is_odd(coord.z);
}

// Checks `cells` and the same cells in a second order with `check`, and expects the failure each time.
template <typename CheckFunction>
void ExpectFailureInAnyOrder(
    std::vector<CellId> cells,
    std::uint64_t seed,
    CheckFunction check,
    GridError error,
    CellId cell,
    CellId other_cell
) {
  SCOPED_TRACE(seed);
  ExpectFailure(check(cells), error, cell, other_cell);
  ApplyFixedShuffle(cells, seed + 2000);
  ExpectFailure(check(cells), error, cell, other_cell);
}

// A cell listed twice is an overlap of that cell with itself, and the only failure of the list.
TEST(CheckSizeStepRule, RandomGridWithRepeatedCell_IsCellsOverlapOfThatCell) {
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    std::vector<CellId> cells = BuildShuffledRandomGrids().at(seed - 1);
    const CellId repeated_cell = cells.at(seed % cells.size());
    cells.push_back(repeated_cell);
    ExpectFailureInAnyOrder(
        cells,
        seed,
        [](const std::vector<CellId>& list) {
          return CheckSizeStepRule(ToUnitFrame(kRandomGridSizeLevels), list);
        },
        GridError::kCellsOverlap,
        repeated_cell,
        repeated_cell
    );
  }
}

// The big cell B of a listed cell is added. Every listed cell inside B overlaps it; in the fixed order (z, then y, then
// x of the low corner) the first of them is the one active cell at B's low corner, found here by whole-unit boxes. B
// itself sorts just before it and holds no overlap, since every cell bigger than B is split, not listed.
TEST(CheckSizeStepRule, RandomGridWithBigCellAdded_IsCellsOverlapAtItsLowCorner) {
  int changed_grid_count = 0;
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    std::vector<CellId> cells = BuildShuffledRandomGrids().at(seed - 1);
    const std::optional<std::size_t> small_index = FindCellFrom(cells, seed % cells.size(), ComputeHasBigCell);
    if (!small_index.has_value()) {
      continue;
    }
    const CellId big_cell = ToBigCellByDivision(cells.at(*small_index));
    const UnitBox big_box = ToUnitBox(big_cell, kRandomGridSizeLevels);
    const auto at_big_corner = std::ranges::find_if(cells, [&big_box](CellId cell) {
      return ToUnitBox(cell, kRandomGridSizeLevels).low == big_box.low;
    });
    ASSERT_NE(at_big_corner, cells.end()) << seed;
    const CellId expected_cell = *at_big_corner;
    cells.push_back(big_cell);
    ExpectFailureInAnyOrder(
        cells,
        seed,
        [](const std::vector<CellId>& list) {
          return CheckSizeStepRule(ToUnitFrame(kRandomGridSizeLevels), list);
        },
        GridError::kCellsOverlap,
        expected_cell,
        big_cell
    );
    ++changed_grid_count;
  }
  EXPECT_GE(changed_grid_count, 190);
}

// Says whether two boxes of cells that do not overlap share at least one point: a side, an edge or a corner. On every
// axis the two ranges then meet or overlap.
bool ComputeIsBoxTouching(const UnitBox& first, const UnitBox& second) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const bool is_meeting = first.low.at(axis) <= second.low.at(axis) + second.width
        && second.low.at(axis) <= first.low.at(axis) + first.width;
    if (!is_meeting) {
      return false;
    }
  }
  return true;
}

// Says whether whole-unit box `outer` holds box `inner`.
bool ComputeIsBoxInside(const UnitBox& inner, const UnitBox& outer) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const bool is_inside = inner.low.at(axis) >= outer.low.at(axis)
        && inner.low.at(axis) + inner.width <= outer.low.at(axis) + outer.width;
    if (!is_inside) {
      return false;
    }
  }
  return true;
}

// The listed cell two or more size levels bigger than `cell` that covers a place of `cell`'s size level beside it, for
// the first direction that has one: (dz, dy, dx), each -1, 0 or +1, in the order dz, then dy, then dx. Found by
// whole-unit boxes over every listed cell; a place lies wholly inside a bigger cell or wholly outside it.
std::optional<CellId> FindSizeStepBreakByBoxes(const std::vector<CellId>& cells, CellId cell) {
  const UnitBox box = ToUnitBox(cell, kRandomGridSizeLevels);
  for (std::int64_t dz = -1; dz <= 1; ++dz) {
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        if (dz == 0 && dy == 0 && dx == 0) {
          continue;
        }
        const UnitBox place = {
            .low =
                {box.low.at(0) + (dx * box.width), box.low.at(1) + (dy * box.width), box.low.at(2) + (dz * box.width)},
            .width = box.width,
        };
        for (const CellId other_cell : cells) {
          const bool is_two_bigger = ToSizeLevelNumber(other_cell) <= ToSizeLevelNumber(cell) - 2;
          if (is_two_bigger && ComputeIsBoxInside(place, ToUnitBox(other_cell, kRandomGridSizeLevels))) {
            return other_cell;
          }
        }
      }
    }
  }
  return std::nullopt;
}

// A cell c of size level 1 or 2 that touches a cell one size level bigger is replaced by its 8 small cells, which
// keeps the siblings together and adds no overlap. Only those small cells can break the rule: every other cell keeps
// its neighbours, or gains smaller ones. The first of them in the fixed order that touches a cell two size levels
// bigger is named, with the bigger cell of its first such direction.
TEST(CheckSizeStepRule, RandomGridWithCellSplitBesideBiggerCell_IsSizeStepRuleBroken) {
  int changed_grid_count = 0;
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    const std::vector<CellId> sorted_cells = ToSortedCells(BuildShuffledRandomGrids().at(seed - 1));
    const auto split_cell = std::ranges::find_if(sorted_cells, [&sorted_cells](CellId cell) {
      const int size_level = ToSizeLevelNumber(cell);
      if (size_level < 1 || size_level > kRandomGridSizeLevels - 2) {
        return false;
      }
      const UnitBox box = ToUnitBox(cell, kRandomGridSizeLevels);
      return std::ranges::any_of(sorted_cells, [size_level, &box](CellId other_cell) {
        const bool is_one_bigger = ToSizeLevelNumber(other_cell) == size_level - 1;
        return is_one_bigger && ComputeIsBoxTouching(box, ToUnitBox(other_cell, kRandomGridSizeLevels));
      });
    });
    if (split_cell == sorted_cells.end()) {
      continue;
    }
    std::vector<CellId> cells;
    std::ranges::copy_if(sorted_cells, std::back_inserter(cells), [split_cell](CellId cell) {
      return cell.value != split_cell->value;
    });
    const std::vector<CellId> small_cells = ToSortedCells(ToSmallCells(*split_cell));
    cells.insert(cells.end(), small_cells.begin(), small_cells.end());
    std::optional<CellId> broken_cell;
    std::optional<CellId> bigger_cell;
    for (const CellId small_cell : small_cells) {
      bigger_cell = FindSizeStepBreakByBoxes(cells, small_cell);
      if (bigger_cell.has_value()) {
        broken_cell = small_cell;
        break;
      }
    }
    if (!broken_cell.has_value() || !bigger_cell.has_value()) {
      ADD_FAILURE() << "no break found for seed " << seed;
      continue;
    }
    ExpectFailureInAnyOrder(
        cells,
        seed,
        [](const std::vector<CellId>& list) {
          return CheckSizeStepRule(ToUnitFrame(kRandomGridSizeLevels), list);
        },
        GridError::kSizeStepRuleBroken,
        *broken_cell,
        *bigger_cell
    );
    ++changed_grid_count;
  }
  EXPECT_GE(changed_grid_count, 190);
}

// The cell and the big cell a missing sibling names when `left_out` is the one cell left out of a valid list, by the
// rule stated at CheckSortedCells: the first cell above it, of size level 1 or more, that is not eighth 0 of its big
// cell, when `left_out` is eighth 0 and such a cell exists; otherwise `left_out` itself. Worked out with the test's own
// halving and parity.
std::pair<CellId, CellId> ToExpectedMissingCells(CellId left_out) {
  if (ComputeIsAwayFromBigCellCorner(left_out)) {
    return {left_out, ToBigCellByDivision(left_out)};
  }
  for (CellId bigger_cell = ToBigCellByDivision(left_out); ComputeHasBigCell(bigger_cell);
       bigger_cell = ToBigCellByDivision(bigger_cell)) {
    if (ComputeIsAwayFromBigCellCorner(bigger_cell)) {
      return {bigger_cell, ToBigCellByDivision(bigger_cell)};
    }
  }
  return {left_out, ToBigCellByDivision(left_out)};
}

bool ComputeIsAtBigCellCorner(CellId cell) {
  return !ComputeIsAwayFromBigCellCorner(cell);
}

// Leaves out of a random grid one cell of size level 1 or more for which `is_wanted` holds, and expects the failure
// ToExpectedMissingCells gives, in two input orders. Returns the cell left out, or none when the grid has none.
std::optional<CellId> ExpectLeftOutCellNamed(std::uint64_t seed, bool (*is_wanted)(CellId), GridBuffers& buffers) {
  std::vector<CellId> cells = BuildShuffledRandomGrids().at(seed - 1);
  const std::optional<std::size_t> left_out_index = FindCellFrom(cells, seed % cells.size(), [is_wanted](CellId cell) {
    return ComputeHasBigCell(cell) && is_wanted(cell);
  });
  if (!left_out_index.has_value()) {
    return std::nullopt;
  }
  const CellId left_out = cells.at(*left_out_index);
  cells.erase(cells.begin() + static_cast<std::ptrdiff_t>(*left_out_index));
  const auto [expected_cell, expected_other_cell] = ToExpectedMissingCells(left_out);
  ExpectFailureInAnyOrder(
      cells,
      seed,
      [&buffers](const std::vector<CellId>& list) {
        return CheckThroughSortedCellList(ToUnitFrame(kRandomGridSizeLevels), list, buffers);
      },
      GridError::kSiblingCellMissing,
      expected_cell,
      expected_other_cell
  );
  return left_out;
}

// One cell of size level 1 or more is left out, twice per grid: once a cell that is eighth 0 of its big cell and once
// one that is not. The named cells follow ToExpectedMissingCells; both kinds, and both answers for eighth-0 cells (the
// cell itself and a bigger cell), must occur.
TEST(BuildSortedCellList, RandomGridMissingOneCell_NamesTheCellByTheEighthZeroRule) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(kRandomGridRoom);
  ASSERT_NE(grid_buffers, nullptr);
  // Counts: cells away from the big cell's corner left out, eighth-0 cells left out, and of those how many name the
  // cell itself and how many a bigger cell.
  std::array<int, 4> counts = {};
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    const std::optional<CellId> away_left_out =
        ExpectLeftOutCellNamed(seed, ComputeIsAwayFromBigCellCorner, *grid_buffers->buffers);
    counts.at(0) += static_cast<int>(away_left_out.has_value());
    const std::optional<CellId> corner_left_out =
        ExpectLeftOutCellNamed(seed, ComputeIsAtBigCellCorner, *grid_buffers->buffers);
    if (corner_left_out.has_value()) {
      ++counts.at(1);
      const bool is_named_itself = ToExpectedMissingCells(*corner_left_out).first.value == corner_left_out->value;
      ++counts.at(is_named_itself ? 2 : 3);
    }
  }
  const bool has_every_kind = counts.at(0) >= 190 && counts.at(1) >= 190 && counts.at(2) >= 1 && counts.at(3) >= 1;
  EXPECT_TRUE(has_every_kind) << counts.at(0) << " " << counts.at(1) << " " << counts.at(2) << " " << counts.at(3);
}

}  // namespace
}  // namespace fluidity::grid
