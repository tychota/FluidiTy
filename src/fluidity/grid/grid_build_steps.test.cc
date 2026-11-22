#include "fluidity/grid/grid_build_steps.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

constexpr std::uint64_t kAllOnes = ~std::uint64_t{0};
constexpr std::uint64_t kTwoToThe15 = std::uint64_t{1} << 15U;
constexpr std::uint64_t kTwoToThe34 = std::uint64_t{1} << 34U;
constexpr std::uint64_t kTwoToThe38 = std::uint64_t{1} << 38U;
constexpr std::uint64_t kTwoToThe40 = std::uint64_t{1} << 40U;
constexpr std::uint64_t kLargestPosition = (std::uint64_t{1} << 48U) - 1;

// 2^exponent as a coordinate, shifted as an unsigned value.
constexpr std::int32_t ComputePowerOfTwo(int exponent) {
  return static_cast<std::int32_t>(1U << static_cast<unsigned>(exponent));
}

constexpr std::int32_t kTwoToThe15Coord = ComputePowerOfTwo(15);
constexpr std::int32_t kTwoToThe16 = ComputePowerOfTwo(16);
constexpr std::int32_t kTwoToThe17 = ComputePowerOfTwo(17);
constexpr std::int32_t kTwoToThe18 = ComputePowerOfTwo(18);

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

// floor(coord / 2^size_level_steps), by integer division and a correction for a negative remainder: the test's own
// arithmetic, independent of the shift the grid uses.
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

// The cell of size level bigger_size_level that holds `cell`, by the test's own floor.
CellId ToBigCellByDivision(CellId cell, int bigger_size_level) {
  const int size_level_steps = static_cast<int>(ToSizeLevel(cell)) - bigger_size_level;
  const CellCoord coord = ToCellCoord(cell);
  return ToCell(
      bigger_size_level,
      ToFloorOfHalving(coord.x, size_level_steps),
      ToFloorOfHalving(coord.y, size_level_steps),
      ToFloorOfHalving(coord.z, size_level_steps)
  );
}

// The 8 small cells of a big cell, by eighth: x + 2 y + 4 z of the high halves.
void AppendSmallCells(CellId big_cell, std::vector<CellId>& cells) {
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
}

// The list in the fixed order, sorted with ComesBefore (std::sort is test-only here).
std::vector<CellId> ToSortedCells(std::vector<CellId> cells) {
  std::ranges::sort(cells, [](CellId first, CellId second) {
    return ComesBefore(first, second);
  });
  return cells;
}

std::vector<std::uint64_t> ToValues(const std::vector<CellId>& cells) {
  std::vector<std::uint64_t> values;
  values.reserve(cells.size());
  for (const CellId cell : cells) {
    values.push_back(cell.value);
  }
  return values;
}

std::uint32_t ToCount(const std::vector<CellId>& cells) {
  return static_cast<std::uint32_t>(cells.size());
}

// The position of `cell` in a list, found by a linear scan, or kNoIndex.
std::uint32_t FindByScan(const std::vector<CellId>& cells, CellId cell) {
  for (std::size_t index = 0; index < cells.size(); ++index) {
    if (cells[index].value == cell.value) {
      return static_cast<std::uint32_t>(index);
    }
  }
  return kNoIndex;
}

// Orders indexes into `cells` by the three passes of a sort key, least significant first, each pass stable, as
// BuildStableOrder does; std::stable_sort stands in for it in this test.
template <typename SortKeyFunction>
std::vector<std::size_t> ToStableOrderByPasses(const std::vector<CellId>& cells, SortKeyFunction sort_key) {
  std::vector<std::size_t> order(cells.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    std::ranges::stable_sort(order, [&cells, &sort_key, pass](std::size_t first, std::size_t second) {
      return sort_key(cells[first], pass) < sort_key(cells[second], pass);
    });
  }
  return order;
}

// Random cell ids at size levels 0..max_size_level, with coordinates in -kMaxCoord..kMaxCoord; the first 100 have one
// coordinate at a limit: axis index % 3, -kMaxCoord for an even index / 3 and kMaxCoord for an odd one.
//
// clang-tidy: a seed, a size level and a count; the swappable-parameters check flags the three integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<CellId> BuildRandomCellIds(std::uint64_t seed, int max_size_level, std::size_t count) {
  std::mt19937_64 random_numbers(seed);
  const std::uint64_t coord_count = (2 * static_cast<std::uint64_t>(kMaxCoord)) + 1;
  std::vector<CellId> cells;
  cells.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint64_t size_level_choices = static_cast<std::uint64_t>(max_size_level) + 1;
    const auto size_level = static_cast<int>(ToUniformIndex(random_numbers(), size_level_choices));
    std::array<std::int32_t, 3> coord{};
    for (std::int32_t& axis_coord : coord) {
      axis_coord = static_cast<std::int32_t>(ToUniformIndex(random_numbers(), coord_count)) - kMaxCoord;
    }
    if (index < 100) {
      coord.at(index % 3) = (index / 3) % 2 == 0 ? -kMaxCoord : kMaxCoord;
    }
    cells.push_back(ToCell(size_level, coord.at(0), coord.at(1), coord.at(2)));
  }
  return cells;
}

// The smallest failure key of one check over every position of a sorted list, or kNoFailureKey.
template <typename CheckFunction>
std::uint64_t ComputeSmallestFailureKey(const std::vector<CellId>& sorted_cells, int step, CheckFunction check) {
  std::uint64_t smallest_key = ~std::uint64_t{0};
  for (std::uint32_t position = 0; position < ToCount(sorted_cells); ++position) {
    const CellCheck cell_check = check(sorted_cells.data(), ToCount(sorted_cells), position);
    if (cell_check.error != GridError::kNone) {
      smallest_key = std::min(smallest_key, ToFailureKey(step, position, cell_check.detail));
    }
  }
  return smallest_key;
}

void ExpectSameCheck(const CellCheck& actual, const CellCheck& expected) {
  EXPECT_EQ(actual.error, expected.error);
  EXPECT_EQ(actual.cell.value, expected.cell.value);
  EXPECT_EQ(actual.other_cell.value, expected.other_cell.value);
  EXPECT_EQ(actual.detail, expected.detail);
}

// The 7 size-level-1 cells of size-level-0 cell {-1,0,0} without its eighth 5, size level 1 {-1,0,1}: eighth 5 is
// x high, y low, z high, so x = 2 (-1) + 1 = -1, y = 0, z = 2 (0) + 1 = 1.
std::vector<CellId> BuildSevenOfEightSiblings() {
  std::vector<CellId> cells;
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    if (cell_eighth != 5) {
      cells.push_back(ComputeSmallCell(ToCell(0, -1, 0, 0), cell_eighth));
    }
  }
  return ToSortedCells(cells);
}

// Size-level-0 cell {0,0,0} split, its eighth 0 (size level 1 {0,0,0}) missing, each of its 7 other size-level-1
// cells split into its 8 size-level-2 cells: 56 active cells.
std::vector<CellId> BuildMissingCellBesideSplitSiblings() {
  std::vector<CellId> cells;
  for (int cell_eighth = 1; cell_eighth < 8; ++cell_eighth) {
    AppendSmallCells(ComputeSmallCell(ToCell(0, 0, 0, 0), cell_eighth), cells);
  }
  return ToSortedCells(cells);
}

// Size-level-0 cell {-1,-1,-1} split down to size level 15 along its eighth-0 chain: at each size level 1..14 the 7
// cells of eighths 1..7 are active and eighth 0 is split; the 8 cells of size level 15 are active. 7 x 14 + 8 = 106
// cells. When missing_size_level_one_eighth is 1..7, that size-level-1 cell is left out.
std::vector<CellId> BuildChainToSizeLevelFifteen(int missing_size_level_one_eighth) {
  std::vector<CellId> cells;
  CellId split_cell = ToCell(0, -1, -1, -1);
  for (int size_level = 1; size_level <= 15; ++size_level) {
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      const bool is_split = cell_eighth == 0 && size_level < 15;
      const bool is_left_out = size_level == 1 && cell_eighth == missing_size_level_one_eighth;
      if (!is_split && !is_left_out) {
        cells.push_back(ComputeSmallCell(split_cell, cell_eighth));
      }
    }
    split_cell = ComputeSmallCell(split_cell, 0);
  }
  return ToSortedCells(cells);
}

// The 200 random grids of the test support, each list sorted into the fixed order. Built once for the whole file,
// since several tests read them; seed s is entry s - 1.
const std::vector<std::vector<CellId>>& BuildSortedRandomGrids() {
  static const std::vector<std::vector<CellId>> sorted_grids = [] {
    std::vector<std::vector<CellId>> grids;
    grids.reserve(kRandomGridCount);
    for (const std::vector<CellId>& shuffled_cells : BuildShuffledRandomGrids()) {
      grids.push_back(ToSortedCells(shuffled_cells));
    }
    return grids;
  }();
  return sorted_grids;
}

const std::vector<CellId>& BuildSortedRandomGrid(std::uint64_t seed) {
  return BuildSortedRandomGrids().at(seed - 1);
}

// ---------------------------------------------------------------------------------------------------------------------
// Sort keys
// ---------------------------------------------------------------------------------------------------------------------

TEST(ToCellSortKey, ThreePasses_GiveComesBeforeOrder) {
  const std::vector<CellId> cells = BuildRandomCellIds(20261114, 3, 10000);
  const std::vector<std::size_t> order = ToStableOrderByPasses(cells, ToCellSortKey);
  std::vector<CellId> sorted_by_keys;
  sorted_by_keys.reserve(cells.size());
  for (const std::size_t index : order) {
    sorted_by_keys.push_back(cells[index]);
  }
  EXPECT_EQ(ToValues(sorted_by_keys), ToValues(ToSortedCells(cells)));
}

TEST(ToCellSortKey, CoordLimits_FitInThirtyEightBits) {
  // biased(kMaxCoord, 0) = (2^18 - 1 + 2^18) << 15 = 2^34 - 2^15; pass 0 adds the size level 0 in the low 4 bits.
  EXPECT_EQ(ToCellSortKey(ToCell(0, kMaxCoord, 0, 0), 0), (kTwoToThe34 - kTwoToThe15) << 4U);
  // biased(-kMaxCoord, 0) = (-(2^18 - 1) + 2^18) << 15 = 2^15.
  EXPECT_EQ(ToCellSortKey(ToCell(0, -kMaxCoord, 0, 0), 0), kTwoToThe15 << 4U);
  // Size level 15: biased(c, 15) = c + 2^33, at most 2^33 + 2^18 - 1, so pass 0 is at most 2^37 + 2^22 - 16 + 15.
  EXPECT_LT(ToCellSortKey(ToCell(15, kMaxCoord, 0, 0), 0), kTwoToThe38);
  EXPECT_LT(ToCellSortKey(ToCell(15, -kMaxCoord, 0, 0), 0), kTwoToThe38);
}

// Coordinates per axis: both limits, the three around 0, and -2^l and 2^l, whose low corners, 2^l x 2^(15 - l) = 2^15
// widths of size level 15, are those of size-level-0 cells -1 and 1, so every size level ties at those corners.
std::vector<CellId> BuildEverySizeLevelAtCoordLimits() {
  std::vector<CellId> cells;
  for (int size_level = 0; size_level < kMaxSizeLevels; ++size_level) {
    // A set, so size level 0, where 2^0 = 1, lists each cell once.
    const std::set<std::int32_t> axis_values =
        {-kMaxCoord, -ComputePowerOfTwo(size_level), -1, 0, 1, ComputePowerOfTwo(size_level), kMaxCoord};
    for (const std::int32_t z : axis_values) {
      for (const std::int32_t y : axis_values) {
        for (const std::int32_t x : axis_values) {
          cells.push_back(ToCell(size_level, x, y, z));
        }
      }
    }
  }
  return cells;
}

TEST(ToCellSortKey, EverySizeLevelAtCoordLimits_GiveComesBeforeOrder) {
  const std::vector<CellId> cells = BuildEverySizeLevelAtCoordLimits();
  const std::vector<std::size_t> order = ToStableOrderByPasses(cells, ToCellSortKey);
  std::vector<CellId> sorted_by_keys;
  sorted_by_keys.reserve(order.size());
  for (const std::size_t index : order) {
    sorted_by_keys.push_back(cells[index]);
  }
  EXPECT_EQ(ToValues(sorted_by_keys), ToValues(ToSortedCells(cells)));
}

// The header's bound: biased < 2^34, so passes 1 and 2 are below 2^34 and pass 0, biased x 16 + 15, below 2^38.
void ExpectKeysBelowProvedBounds(CellId cell) {
  EXPECT_LT(ToCellSortKey(cell, 0), kTwoToThe38);
  EXPECT_LT(ToCellSortKey(cell, 1), kTwoToThe34);
  EXPECT_LT(ToCellSortKey(cell, 2), kTwoToThe34);
  EXPECT_LT(ToBlockSortKey(cell, 0), kTwoToThe38);
  EXPECT_LT(ToBlockSortKey(cell, 1), kTwoToThe34);
  EXPECT_LT(ToBlockSortKey(cell, 2), kTwoToThe34);
}

TEST(ToCellSortKey, EverySizeLevelAtCoordLimits_StaysBelowTheProvedBounds) {
  for (const CellId cell : BuildEverySizeLevelAtCoordLimits()) {
    ExpectKeysBelowProvedBounds(cell);
  }
}

TEST(ToCellSortKey, InvalidId_SortsLast) {
  const CellId face_piece_value = {.value = ToFacePieceId(Axis::kZ, SizeLevel{3}, {.x = 1, .y = 2, .z = 3}).value};
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    EXPECT_EQ(ToCellSortKey(CellId{0}, pass), kAllOnes);
    EXPECT_EQ(ToCellSortKey(face_piece_value, pass), kAllOnes);
  }
}

TEST(ToBlockSortKey, LowestBlockAtSizeLevelZero_IsZero) {
  // Block corner 4 floor(-(2^18 - 1) / 4) = -2^18, plus 2^18: 0 on every axis, and size level 0 in pass 0.
  const CellId lowest_cell = ToCell(0, -kMaxCoord, -kMaxCoord, -kMaxCoord);
  const CellId next_cell = ToCell(0, -kMaxCoord + 4, -kMaxCoord + 4, -kMaxCoord + 4);
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    EXPECT_EQ(ToBlockSortKey(lowest_cell, pass), 0U);
    EXPECT_LE(ToBlockSortKey(lowest_cell, pass), ToBlockSortKey(next_cell, pass));
  }
}

// Every in-range cell of one block has the key of the block in each pass.
void ExpectEqualKeysInBlock(int size_level, CellCoord block_coord) {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 4; ++z) {
    for (std::int32_t y = 0; y < 4; ++y) {
      for (std::int32_t x = 0; x < 4; ++x) {
        const CellCoord coord = {
            .x = (4 * block_coord.x) + x,
            .y = (4 * block_coord.y) + y,
            .z = (4 * block_coord.z) + z,
        };
        if (ComputeIsCoordInRange(coord)) {
          cells.push_back(ToCellId(static_cast<SizeLevel>(size_level), coord));
        }
      }
    }
  }
  ASSERT_FALSE(cells.empty());
  for (const CellId cell : cells) {
    for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
      EXPECT_EQ(ToBlockSortKey(cell, pass), ToBlockSortKey(cells.front(), pass))
          << size_level << ": " << block_coord.x << " " << block_coord.y << " " << block_coord.z;
    }
  }
}

TEST(ToBlockSortKey, CellsOfOneBlock_HaveEqualKeys) {
  for (const int size_level : {0, 3, 15}) {
    ExpectEqualKeysInBlock(size_level, {.x = -1, .y = 2, .z = -3});
    // The block at the low limit holds cells -2^18 + 1..-2^18 + 3 on each axis; the one at the high limit
    // 2^18 - 4..2^18 - 1.
    ExpectEqualKeysInBlock(size_level, {.x = -kTwoToThe16, .y = -kTwoToThe16, .z = -kTwoToThe16});
    ExpectEqualKeysInBlock(size_level, {.x = kTwoToThe16 - 1, .y = kTwoToThe16 - 1, .z = kTwoToThe16 - 1});
  }
}

TEST(ToBlockSortKey, InvalidId_SortsLast) {
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    EXPECT_EQ(ToBlockSortKey(CellId{0}, pass), kAllOnes);
  }
}

TEST(ToBlockSortKey, ThreePasses_GiveComesBeforeOrderOfBlocks) {
  std::vector<CellId> cells = BuildRandomCellIds(20261115, 15, 10000);
  const std::vector<CellId> limit_cells = BuildEverySizeLevelAtCoordLimits();
  cells.insert(cells.end(), limit_cells.begin(), limit_cells.end());
  const std::vector<std::size_t> order = ToStableOrderByPasses(cells, ToBlockSortKey);
  std::vector<std::uint64_t> blocks_by_keys;
  blocks_by_keys.reserve(order.size());
  std::vector<CellBlockId> blocks;
  blocks.reserve(cells.size());
  for (const std::size_t index : order) {
    blocks_by_keys.push_back(ToCellBlockId(cells[index]).value);
  }
  for (const CellId cell : cells) {
    blocks.push_back(ToCellBlockId(cell));
  }
  std::ranges::sort(blocks, [](CellBlockId first, CellBlockId second) {
    return ComesBefore(first, second);
  });
  std::vector<std::uint64_t> sorted_blocks;
  sorted_blocks.reserve(blocks.size());
  for (const CellBlockId block : blocks) {
    sorted_blocks.push_back(block.value);
  }
  EXPECT_EQ(blocks_by_keys, sorted_blocks);
}

// ---------------------------------------------------------------------------------------------------------------------
// Ids and searches
// ---------------------------------------------------------------------------------------------------------------------

TEST(CheckCellId, IdWithoutCellMark_IsBadInput) {
  const CellId face_piece_value = {.value = ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 0, .y = 0, .z = 0}).value};
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), CellId{0}), GridError::kBadInput);
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), face_piece_value), GridError::kBadInput);
}

TEST(CheckCellId, AxisBitsSet_IsBadInput) {
  // Bits 61..62 are zero in every cell id; a value with either set is no cell id.
  const std::uint64_t cell_value = ToCell(1, 0, 0, 0).value;
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), CellId{cell_value | (std::uint64_t{1} << 61U)}), GridError::kBadInput);
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), CellId{cell_value | (std::uint64_t{1} << 62U)}), GridError::kBadInput);
}

TEST(CheckCellId, SizeLevelBeyondFrame_IsBadInput) {
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), ToCell(3, 0, 0, 0)), GridError::kBadInput);
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), ToCell(15, 0, 0, 0)), GridError::kBadInput);
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), ToCell(2, 0, 0, 0)), GridError::kNone);
  EXPECT_EQ(CheckCellId(ToUnitFrame(16), ToCell(15, 0, 0, 0)), GridError::kNone);
}

TEST(CheckCellId, OnePastHighCoordLimit_IsCoordOutOfRange) {
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), ToCell(2, kMaxCoord + 1, 0, 0)), GridError::kCoordOutOfRange);
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), ToCell(0, 0, kMaxCoord + 1, 0)), GridError::kCoordOutOfRange);
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), ToCell(1, 0, 0, kMaxCoord + 1)), GridError::kCoordOutOfRange);
}

TEST(CheckCellId, BadSizeLevelAndBadCoord_IsBadInput) {
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), ToCell(5, kMaxCoord + 1, 0, 0)), GridError::kBadInput);
}

TEST(CheckCellId, CoordLimits_AreValid) {
  for (const int size_level : {0, 2}) {
    for (const std::int32_t z : {-kMaxCoord, 0, kMaxCoord}) {
      for (const std::int32_t y : {-kMaxCoord, 0, kMaxCoord}) {
        for (const std::int32_t x : {-kMaxCoord, 0, kMaxCoord}) {
          EXPECT_EQ(CheckCellId(ToUnitFrame(3), ToCell(size_level, x, y, z)), GridError::kNone);
        }
      }
    }
  }
}

TEST(FindSortedCell, EveryListedCell_GivesItsPosition) {
  const std::vector<CellId> sorted_cells = ToSortedCells(BuildEverySizeLevelAtCoordLimits());
  for (std::uint32_t position = 0; position < ToCount(sorted_cells); ++position) {
    EXPECT_EQ(FindSortedCell(sorted_cells.data(), ToCount(sorted_cells), sorted_cells[position]), position);
  }
}

TEST(FindSortedCell, UnlistedCell_IsNoIndex) {
  const std::vector<CellId> sorted_cells = ToSortedCells(BuildEverySizeLevelAtCoordLimits());
  EXPECT_EQ(FindSortedCell(sorted_cells.data(), ToCount(sorted_cells), ToCell(3, 2, 2, 2)), kNoIndex);
  EXPECT_EQ(FindSortedCell(sorted_cells.data(), ToCount(sorted_cells), ToCell(15, -kMaxCoord + 1, 0, 0)), kNoIndex);
  EXPECT_EQ(FindSortedCell(nullptr, 0, ToCell(0, 0, 0, 0)), kNoIndex);
}

TEST(FindSortedBlock, EveryListedBlock_GivesItsIndex) {
  std::vector<CellBlockId> blocks;
  for (const CellId cell : BuildEverySizeLevelAtCoordLimits()) {
    blocks.push_back(ToCellBlockId(cell));
  }
  std::ranges::sort(blocks, [](CellBlockId first, CellBlockId second) {
    return ComesBefore(first, second);
  });
  const auto [unique_end, blocks_end] = std::ranges::unique(blocks, [](CellBlockId first, CellBlockId second) {
    return first.value == second.value;
  });
  blocks.erase(unique_end, blocks_end);
  const auto block_count = static_cast<std::uint32_t>(blocks.size());
  for (std::uint32_t index = 0; index < block_count; ++index) {
    EXPECT_EQ(FindSortedBlock(blocks.data(), block_count, blocks[index]).value, index);
  }
  // Size-level-3 block (4, 4, 4) holds the cells 16..19; no cell of the list lies there.
  EXPECT_EQ(FindSortedBlock(blocks.data(), block_count, ToCellBlockId(ToCell(3, 16, 16, 16))).value, kNoIndex);
}

TEST(FindSortedCellAt, CoordOutsideRange_IsNoIndex) {
  // Packing {-kMaxCoord - 1, 0, 0} keeps 19 bits of -1, 2^19 - 1: the id of size-level-0 cell {kMaxCoord + 1, 0, 0}.
  // The list holds that id too, so a search by the packed id would find it.
  const CellId wrapped_cell = ToCell(0, -kMaxCoord - 1, 0, 0);
  ASSERT_EQ(wrapped_cell.value, ToCell(0, kMaxCoord + 1, 0, 0).value);
  const std::vector<CellId> sorted_cells = ToSortedCells({ToCell(1, 1, 0, 0), wrapped_cell});
  const CellCoord outside_coord = {.x = -kMaxCoord - 1, .y = 0, .z = 0};
  EXPECT_EQ(FindSortedCellAt(sorted_cells.data(), ToCount(sorted_cells), SizeLevel{0}, outside_coord), kNoIndex);
}

TEST(FindSortedCellAt, CoordInRange_FindsTheCell) {
  const std::vector<CellId> sorted_cells = ToSortedCells({ToCell(1, 1, 0, 0), ToCell(4, -kMaxCoord, kMaxCoord, -7)});
  const CellCoord limit_coord = {.x = -kMaxCoord, .y = kMaxCoord, .z = -7};
  const std::uint32_t expected_position = FindByScan(sorted_cells, ToCell(4, -kMaxCoord, kMaxCoord, -7));
  EXPECT_EQ(FindSortedCellAt(sorted_cells.data(), ToCount(sorted_cells), SizeLevel{4}, limit_coord), expected_position);
}

TEST(ComputeBigCellAtSizeLevel, TwoSizeLevelsUp_HalvesTwice) {
  // floor(-5 / 4) = -2, floor(6 / 4) = 1, floor(7 / 4) = 1.
  EXPECT_EQ(ComputeBigCellAtSizeLevel(ToCell(3, -5, 6, 7), SizeLevel{1}).value, ToCell(1, -2, 1, 1).value);
}

TEST(ComputeBigCellAtSizeLevel, OwnSizeLevel_GivesTheCell) {
  EXPECT_EQ(
      ComputeBigCellAtSizeLevel(ToCell(7, -kMaxCoord, 3, kMaxCoord), SizeLevel{7}).value,
      ToCell(7, -kMaxCoord, 3, kMaxCoord).value
  );
}

TEST(ComputeBigCellAtSizeLevel, CoordLimitsAtEverySizeLevel_GiveTheFloor) {
  // From size level 15 to size level 0: floor((2^18 - 1) / 2^15) = 7 and floor(-(2^18 - 1) / 2^15) = -8.
  EXPECT_EQ(
      ComputeBigCellAtSizeLevel(ToCell(15, kMaxCoord, -kMaxCoord, -1), SizeLevel{0}).value,
      ToCell(0, 7, -8, -1).value
  );
  for (int size_level = 0; size_level < kMaxSizeLevels; ++size_level) {
    for (int bigger_size_level = 0; bigger_size_level <= size_level; ++bigger_size_level) {
      const CellId cell = ToCell(size_level, kMaxCoord, -kMaxCoord, -3);
      EXPECT_EQ(
          ComputeBigCellAtSizeLevel(cell, static_cast<SizeLevel>(bigger_size_level)).value,
          ToBigCellByDivision(cell, bigger_size_level).value
      ) << size_level
        << " to " << bigger_size_level;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Overlaps: method paper <proc-build-grid> step 2
// ---------------------------------------------------------------------------------------------------------------------

TEST(CheckOverlapAt, RepeatedCell_NamesBothCells) {
  const CellId repeated_cell = ToCell(1, -1, 2, 3);
  const std::vector<CellId> sorted_cells = ToSortedCells({ToCell(1, 0, 2, 3), repeated_cell, repeated_cell});
  const std::uint32_t second_position = FindByScan(sorted_cells, repeated_cell) + 1;
  ExpectSameCheck(
      CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), second_position),
      {.error = GridError::kCellsOverlap, .cell = repeated_cell, .other_cell = repeated_cell, .detail = 0}
  );
  ExpectSameCheck(CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), second_position - 1), {});
}

TEST(CheckOverlapAt, CellInsideActiveBigCell_NamesBothCells) {
  // Size level 2 {-3, 1, 2} lies in size level 0 {floor(-3 / 4), floor(1 / 4), floor(2 / 4)} = {-1, 0, 0}.
  const CellId small_cell = ToCell(2, -3, 1, 2);
  const CellId big_cell = ToCell(0, -1, 0, 0);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, big_cell});
  ExpectSameCheck(
      CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kCellsOverlap, .cell = small_cell, .other_cell = big_cell, .detail = 0}
  );
  ExpectSameCheck(CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, big_cell)), {});
}

TEST(CheckOverlapAt, CellInsideTwoActiveBiggerCells_NamesTheNearest) {
  // Size level 2 {-3, 1, 2} lies in size level 1 {-2, 0, 1} and in size level 0 {-1, 0, 0}; one size level up first.
  const CellId small_cell = ToCell(2, -3, 1, 2);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, -1, 0, 0), ToCell(1, -2, 0, 1)});
  ExpectSameCheck(
      CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kCellsOverlap, .cell = small_cell, .other_cell = ToCell(1, -2, 0, 1), .detail = 0}
  );
}

TEST(CheckOverlapAt, CellAtCoordLimitsInsideSizeLevelZeroCell_NamesBothCells) {
  // Size level 15 {kMaxCoord, -kMaxCoord, 0} lies in size level 0 {7, -8, 0}: floor(+-(2^18 - 1) / 2^15).
  const CellId small_cell = ToCell(15, kMaxCoord, -kMaxCoord, 0);
  const CellId big_cell = ToCell(0, 7, -8, 0);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, big_cell});
  ExpectSameCheck(
      CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kCellsOverlap, .cell = small_cell, .other_cell = big_cell, .detail = 0}
  );
}

TEST(CheckOverlapAt, RandomGrids_Pass) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<CellId>& sorted_cells = BuildSortedRandomGrid(seed);
    EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 2, CheckOverlapAt), kNoFailureKey) << seed;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Siblings: method paper <proc-build-grid> step 3
// ---------------------------------------------------------------------------------------------------------------------

TEST(CheckSiblingsAt, SevenOfEight_NamesTheMissing) {
  const std::vector<CellId> sorted_cells = BuildSevenOfEightSiblings();
  for (std::uint32_t position = 0; position < ToCount(sorted_cells); ++position) {
    ExpectSameCheck(
        CheckSiblingsAt(sorted_cells.data(), ToCount(sorted_cells), position),
        {
            .error = GridError::kSiblingCellMissing,
            .cell = ToCell(1, -1, 0, 1),
            .other_cell = ToCell(0, -1, 0, 0),
            .detail = 5,
        }
    );
  }
}

TEST(CheckSiblingsAt, MissingCellBesideSplitSiblings_NamesTheMissing) {
  const std::vector<CellId> sorted_cells = BuildMissingCellBesideSplitSiblings();
  ASSERT_EQ(sorted_cells.size(), 56U);
  // The first cell in the fixed order that is eighth 0 of its size-level-1 cell: every coordinate even.
  std::uint32_t first_eighth_zero_position = kNoIndex;
  for (std::uint32_t position = 0; position < ToCount(sorted_cells); ++position) {
    const CellCoord coord = ToCellCoord(sorted_cells[position]);
    if (coord.x % 2 == 0 && coord.y % 2 == 0 && coord.z % 2 == 0) {
      first_eighth_zero_position = position;
      break;
    }
  }
  const std::uint64_t smallest_key = ComputeSmallestFailureKey(sorted_cells, 3, CheckSiblingsAt);
  ASSERT_NE(smallest_key, kNoFailureKey);
  EXPECT_EQ(ToFailurePosition(smallest_key), first_eighth_zero_position);
  // Walk step 1 checks the 8 size-level-1 cells of size level 0 {0,0,0}; its eighth 0 is the missing one.
  ExpectSameCheck(
      CheckSiblingsAt(sorted_cells.data(), ToCount(sorted_cells), first_eighth_zero_position),
      {
          .error = GridError::kSiblingCellMissing,
          .cell = ToCell(1, 0, 0, 0),
          .other_cell = ToCell(0, 0, 0, 0),
          .detail = (1U << 3U) | 0U,
      }
  );
}

TEST(CheckSiblingsAt, SplitSiblings_Pass) {
  std::vector<CellId> cells = BuildMissingCellBesideSplitSiblings();
  AppendSmallCells(ToCell(1, 0, 0, 0), cells);
  const std::vector<CellId> sorted_cells = ToSortedCells(cells);
  ASSERT_EQ(sorted_cells.size(), 64U);
  EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 3, CheckSiblingsAt), kNoFailureKey);
}

TEST(CheckSiblingsAt, ChainDownToSizeLevelFifteen_Passes) {
  const std::vector<CellId> sorted_cells = BuildChainToSizeLevelFifteen(0);
  ASSERT_EQ(sorted_cells.size(), 106U);
  EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 3, CheckSiblingsAt), kNoFailureKey);
}

TEST(CheckSiblingsAt, MissingSiblingAtTopOfLongChain_IsFoundFromTheChainEnd) {
  // Size level 1 eighth 7 of size level 0 {-1,-1,-1} is {-1,-1,-1} at size level 1 (2 (-1) + 1 on each axis).
  const std::vector<CellId> sorted_cells = BuildChainToSizeLevelFifteen(7);
  const CellCheck expected_check = {
      .error = GridError::kSiblingCellMissing,
      .cell = ToCell(1, -1, -1, -1),
      .other_cell = ToCell(0, -1, -1, -1),
      .detail = 7,
  };
  // From the end of the eighth-0 chain, size level 15 {-2^15, ...}: walk steps 0..13 pass, walk step 14 checks size
  // level 0 {-1,-1,-1}, so detail = (14 << 3) | 7 = 119.
  const CellId chain_end = ToCell(15, -kTwoToThe15Coord, -kTwoToThe15Coord, -kTwoToThe15Coord);
  ExpectSameCheck(
      CheckSiblingsAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, chain_end)),
      {
          .error = expected_check.error,
          .cell = expected_check.cell,
          .other_cell = expected_check.other_cell,
          .detail = 119,
      }
  );
  // An active size-level-1 sibling, eighth 1 = {-1,-2,-2}, finds it at its first walk step.
  ExpectSameCheck(
      CheckSiblingsAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, ToCell(1, -1, -2, -2))),
      expected_check
  );
}

// The 4 cells of size level small_size_level at x = -kMaxCoord and y, z = 0 or 1. They lie in the big cell at x =
// floor(-(2^18 - 1) / 2) = -2^17, whose eighth-0 small cells have x = -2^18, outside the range.
std::vector<CellId> BuildCellsAtLowCoordLimit(int small_size_level) {
  std::vector<CellId> cells;
  for (const std::int32_t z : {0, 1}) {
    for (const std::int32_t y : {0, 1}) {
      cells.push_back(ToCell(small_size_level, -kMaxCoord, y, z));
    }
  }
  return ToSortedCells(cells);
}

TEST(CheckSiblingsAt, SiblingBeyondLowCoordLimit_NamesNoCellAndTheBigCell) {
  // No id names the small cell at -2^18, so the failure names CellId{0} and the big cell, at every size level.
  for (int small_size_level = 1; small_size_level < kMaxSizeLevels; ++small_size_level) {
    const std::vector<CellId> sorted_cells = BuildCellsAtLowCoordLimit(small_size_level);
    for (std::uint32_t position = 0; position < ToCount(sorted_cells); ++position) {
      ExpectSameCheck(
          CheckSiblingsAt(sorted_cells.data(), ToCount(sorted_cells), position),
          {
              .error = GridError::kSiblingCellMissing,
              .cell = CellId{0},
              .other_cell = ToCell(small_size_level - 1, -kTwoToThe17, 0, 0),
              .detail = 0,
          }
      );
    }
  }
}

// Size level 0 {2^(18 - l) - 1}^3, l = small_size_level, split along its eighth-7 path down to size level l: at each
// size level j the path cell is 2^(18 - l + j) - 1 on every axis, so the 8 cells of size level l are 2^18 - 2 and
// 2^18 - 1 = kMaxCoord. Above size level l the 7 other small cells of each split cell are active. With
// missing_eighth 0..7, that cell of size level l is left out.
std::vector<CellId> BuildPathToHighCoordLimit(int small_size_level, int missing_eighth) {
  const std::int32_t start_coord = ComputePowerOfTwo(18 - small_size_level) - 1;
  CellId path_cell = ToCell(0, start_coord, start_coord, start_coord);
  std::vector<CellId> cells;
  for (int size_level = 1; size_level <= small_size_level; ++size_level) {
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      const bool is_path = cell_eighth == 7 && size_level < small_size_level;
      const bool is_left_out = size_level == small_size_level && cell_eighth == missing_eighth;
      if (!is_path && !is_left_out) {
        cells.push_back(ComputeSmallCell(path_cell, cell_eighth));
      }
    }
    path_cell = ComputeSmallCell(path_cell, 7);
  }
  return ToSortedCells(cells);
}

TEST(CheckSiblingsAt, SiblingsAtHighCoordLimit_Pass) {
  for (int small_size_level = 1; small_size_level < kMaxSizeLevels; ++small_size_level) {
    const std::vector<CellId> sorted_cells = BuildPathToHighCoordLimit(small_size_level, -1);
    ASSERT_NE(FindByScan(sorted_cells, ToCell(small_size_level, kMaxCoord, kMaxCoord, kMaxCoord)), kNoIndex);
    EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 3, CheckSiblingsAt), kNoFailureKey) << small_size_level;
  }
}

TEST(CheckSiblingsAt, MissingSiblingAtHighCoordLimit_NamesTheMissing) {
  // Eighth 7 is {kMaxCoord, kMaxCoord, kMaxCoord}. Its eighth-0 small cell, 2^19 - 2, is outside the range, so the
  // search for a split cell stops there; at size level 15 it stops because no size level is smaller. Its sibling
  // eighth 0, {2^18 - 2}^3, finds it at walk step 0: detail 7.
  for (int small_size_level = 1; small_size_level < kMaxSizeLevels; ++small_size_level) {
    const std::vector<CellId> sorted_cells = BuildPathToHighCoordLimit(small_size_level, 7);
    const CellId eighth_zero_cell = ToCell(small_size_level, kMaxCoord - 1, kMaxCoord - 1, kMaxCoord - 1);
    ExpectSameCheck(
        CheckSiblingsAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, eighth_zero_cell)),
        {
            .error = GridError::kSiblingCellMissing,
            .cell = ToCell(small_size_level, kMaxCoord, kMaxCoord, kMaxCoord),
            .other_cell = ToCell(small_size_level - 1, kTwoToThe17 - 1, kTwoToThe17 - 1, kTwoToThe17 - 1),
            .detail = 7,
        }
    );
  }
}

// Splits `cell` in `cells` into its 8 small cells.
void ApplySplit(std::vector<CellId>& cells, CellId cell) {
  std::erase_if(cells, [cell](CellId listed_cell) {
    return listed_cell.value == cell.value;
  });
  AppendSmallCells(cell, cells);
}

TEST(CheckSiblingsAt, SizeLevelZeroCellAtMinusEight_HoldsSizeLevelFifteenAwayFromItsLowCorner) {
  // Size level 0 {-8, 0, 0} covers size-level-14 x -2^17..-2^17 + 2^14 - 1. Of those only -2^17 cannot be split (its
  // small cells would reach -2^18), so the path of eighth-7 cells, x = 2 x + 1 at each split, reaches size level 15:
  // x = -15, -29, ..., -2^17 + 2^14 - 1 at size level 14, whose small cells are in range. The size-step closure
  // splits only cells of size level 13 or bigger, which can all be split here.
  std::vector<CellId> cells = {ToCell(0, -8, 0, 0)};
  CellId path_cell = ToCell(0, -8, 0, 0);
  for (int size_level = 0; size_level < 15; ++size_level) {
    ApplySplit(cells, path_cell);
    path_cell = ComputeSmallCell(path_cell, 7);
  }
  const std::vector<CellId> sorted_cells = ToSortedCells(BuildSizeStepClosure(cells, kMaxSizeLevels));
  EXPECT_EQ(ToCellCoord(path_cell).x, -(7 * kTwoToThe15Coord) - 1);
  EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 2, CheckOverlapAt), kNoFailureKey);
  EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 3, CheckSiblingsAt), kNoFailureKey);
  EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 4, CheckSizeStepAt), kNoFailureKey);
}

TEST(CheckSiblingsAt, RandomGrids_Pass) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<CellId>& sorted_cells = BuildSortedRandomGrid(seed);
    EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 3, CheckSiblingsAt), kNoFailureKey) << seed;
  }
}

// Says whether an active cell lies inside `cell` or is `cell`, by unit boxes. With kSameLowCorner, only an active cell
// whose low corner is that of `cell` counts: one at the end of the eighth-0 chain of `cell`.
enum class CornerRule : std::uint8_t {
  kAnyCorner,
  kSameLowCorner,
};
bool ComputeHoldsActiveCell(
    const std::vector<CellId>& active_cells,
    CellId cell,
    int size_level_count,
    CornerRule corner_rule
) {
  const UnitBox outer_box = ToUnitBox(cell, size_level_count);
  return std::ranges::any_of(active_cells, [&](CellId active_cell) {
    const UnitBox inner_box = ToUnitBox(active_cell, size_level_count);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const bool is_inside = inner_box.low.at(axis) >= outer_box.low.at(axis)
          && inner_box.low.at(axis) + inner_box.width <= outer_box.low.at(axis) + outer_box.width;
      const bool is_corner_kept =
          corner_rule == CornerRule::kAnyCorner || inner_box.low.at(axis) == outer_box.low.at(axis);
      if (!is_inside || !is_corner_kept) {
        return false;
      }
    }
    return true;
  });
}

// A named cell has no active cell at the end of its eighth-0 chain, while its big cell holds one. When a removed cell
// ended the chain of a split cell, that split cell is named too: its chain is broken.
void ExpectNamedCellAbsent(const std::vector<CellId>& cells, const CellCheck& cell_check, std::uint64_t seed) {
  EXPECT_EQ(cell_check.error, GridError::kSiblingCellMissing);
  EXPECT_FALSE(ComputeHoldsActiveCell(cells, cell_check.cell, 4, CornerRule::kSameLowCorner)) << seed;
  EXPECT_TRUE(ComputeHoldsActiveCell(cells, cell_check.other_cell, 4, CornerRule::kAnyCorner)) << seed;
}

// Leaves out of one random grid its first cell of size level 1 or more (a size-level-0 cell has no siblings to miss)
// and checks the siblings at every position.
void ExpectMissingCellNamedOnlyWithAbsentCells(std::uint64_t seed) {
  std::vector<CellId> cells = BuildSortedRandomGrid(seed);
  const auto removed = std::ranges::find_if(cells, [](CellId cell) {
    return ToSizeLevel(cell) != SizeLevel{0};
  });
  if (removed == cells.end()) {
    return;
  }
  const CellId removed_cell = *removed;
  cells.erase(removed);
  bool is_removed_cell_named = false;
  for (std::uint32_t position = 0; position < ToCount(cells); ++position) {
    const CellCheck cell_check = CheckSiblingsAt(cells.data(), ToCount(cells), position);
    if (cell_check.error == GridError::kNone) {
      continue;
    }
    ExpectNamedCellAbsent(cells, cell_check, seed);
    is_removed_cell_named = is_removed_cell_named || cell_check.cell.value == removed_cell.value;
  }
  EXPECT_TRUE(is_removed_cell_named) << seed;
}

TEST(CheckSiblingsAt, RandomGridMissingOneCell_NamesTheCellAndOnlyAbsentCells) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    ExpectMissingCellNamedOnlyWithAbsentCells(seed);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Size step rule: method paper <proc-build-grid> step 4
// ---------------------------------------------------------------------------------------------------------------------

TEST(CheckSizeStepAt, CornerTouchTwoSizeLevelsApart_NamesDirectionZero) {
  // Size level 0 {0,0,0} covers size-level-2 places 0..3; the place (-1, -1, -1) from {4,4,4} is (3, 3, 3).
  const CellId small_cell = ToCell(2, 4, 4, 4);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, 0, 0, 0)});
  ExpectSameCheck(
      CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kSizeStepRuleBroken, .cell = small_cell, .other_cell = ToCell(0, 0, 0, 0), .detail = 0}
  );
}

TEST(CheckSizeStepAt, BigCellOfTheTwo_SeesNothing) {
  const std::vector<CellId> sorted_cells = ToSortedCells({ToCell(2, 4, 4, 4), ToCell(0, 0, 0, 0)});
  ExpectSameCheck(
      CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, ToCell(0, 0, 0, 0))),
      {}
  );
}

TEST(CheckSizeStepAt, OneSizeLevelApart_Passes) {
  // Size level 1 {2,0,0} lies in size level 0 {1,0,0}; its -X place, size level 1 {1,0,0}, lies in size level 0
  // {0,0,0}, one size level bigger, which the rule allows.
  const std::vector<CellId> sorted_cells = ToSortedCells({ToCell(1, 2, 0, 0), ToCell(0, 0, 0, 0)});
  EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 4, CheckSizeStepAt), kNoFailureKey);
}

TEST(CheckSizeStepAt, ThreeSizeLevelsApartBelowZero_NamesTheFirstDirection) {
  // Size level 3 {-8,-8,-8} lies in size level 0 {-1,-1,-1}. Size level 0 {-2,-1,-1} covers size-level-3 places
  // x -16..-9, y and z -8..-1. A place with dz or dy = -1 has z or y = -9, outside; the first place inside is
  // (dz, dy, dx) = (0, 0, -1): 9 (dz + 1) + 3 (dy + 1) + (dx + 1) = 12, before the left-out middle 13.
  const CellId small_cell = ToCell(3, -8, -8, -8);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, -2, -1, -1)});
  ExpectSameCheck(
      CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kSizeStepRuleBroken, .cell = small_cell, .other_cell = ToCell(0, -2, -1, -1), .detail = 12}
  );
}

TEST(CheckSizeStepAt, CellAtLowCoordLimit_SeesNoCellBeyond) {
  // A packed id of the place beyond, size level 2 {-kMaxCoord - 1, 0, 0}, keeps x = 2^18, which lies in size level 0
  // {2^18 / 4, 0, 0} = {65536, 0, 0}: 2^18 + 2^18 - 1 = 524,287 size-level-2 widths from the cell.
  const CellId small_cell = ToCell(2, -kMaxCoord, 0, 0);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, 65536, 0, 0)});
  ExpectSameCheck(
      CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {}
  );
}

TEST(CheckSizeStepAt, CellAtHighCoordLimit_SeesTheBigCellBeside) {
  // The +X place of size level 2 {kMaxCoord, 0, 0} is x = 2^18, in size level 0 x = 2^18 / 4 = 65536, while the cell
  // itself lies in size level 0 x = floor((2^18 - 1) / 4) = 65535. Direction (0, 0, +1) is 9 + 3 + 2 - 1 = 13. The
  // directions before it with dx = +1 have dy or dz = -1, whose places lie in size level 0 y or z = -1.
  const CellId small_cell = ToCell(2, kMaxCoord, 0, 0);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, 65536, 0, 0)});
  ExpectSameCheck(
      CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kSizeStepRuleBroken, .cell = small_cell, .other_cell = ToCell(0, 65536, 0, 0), .detail = 13}
  );
}

TEST(CheckSizeStepAt, SizeLevelFifteenAtHighCoordLimit_SeesTheSizeLevelZeroCellBeyond) {
  // Size level 15 {kMaxCoord}^3 lies in size level 0 {7,7,7}. Size level 0 {8,7,7} covers size-level-15 places x
  // 2^18..2^18 + 2^15 - 1, y and z 7 x 2^15..2^18 - 1. The first direction whose place it covers is
  // (dz, dy, dx) = (-1, -1, +1): place (2^18, 2^18 - 2, 2^18 - 2), direction index 0 + 0 + 2 = 2.
  const CellId small_cell = ToCell(15, kMaxCoord, kMaxCoord, kMaxCoord);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, 8, 7, 7)});
  ExpectSameCheck(
      CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kSizeStepRuleBroken, .cell = small_cell, .other_cell = ToCell(0, 8, 7, 7), .detail = 2}
  );
}

TEST(CheckSizeStepAt, RandomGrids_Pass) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<CellId>& sorted_cells = BuildSortedRandomGrid(seed);
    EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 4, CheckSizeStepAt), kNoFailureKey) << seed;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Failure keys: method paper <sec-parallel-build>, "The first failure"
// ---------------------------------------------------------------------------------------------------------------------

TEST(ToFailureKey, StepOutranksPosition) {
  EXPECT_GT(ToFailureKey(2, 0, 0), ToFailureKey(1, kTwoToThe40, 255));
}

TEST(ToFailureKey, PositionOutranksDetail) {
  EXPECT_LT(ToFailureKey(3, 1, 255), ToFailureKey(3, 2, 0));
}

TEST(ToFailureKey, PackedFields_ReadBack) {
  // (2 << 56) | (5 << 8) | 0 = 144115188075857152, the example of grid/grid_build_status.h.
  EXPECT_EQ(ToFailureKey(2, 5, 0), 144115188075857152U);
  const std::uint64_t largest_key = ToFailureKey(6, kLargestPosition, 255);
  EXPECT_EQ(ToFailureStep(largest_key), 6);
  EXPECT_EQ(ToFailurePosition(largest_key), kLargestPosition);
  EXPECT_EQ(ToFailureDetail(largest_key), 255U);
  EXPECT_LT(largest_key, kNoFailureKey);
  EXPECT_EQ(ToFailureStep(ToFailureKey(1, 0, 0)), 1);
  EXPECT_EQ(ToFailurePosition(ToFailureKey(1, 0, 0)), 0U);
}

// The detail is the low 8 bits alone: neither the position nor the step leaks into it.
TEST(ToFailureDetail, PackedKeys_GiveTheLowEightBits) {
  EXPECT_EQ(ToFailureDetail(ToFailureKey(3, 7, 9)), 9U);
  EXPECT_EQ(ToFailureDetail(kSortCountFaultKey), 1U);
  EXPECT_EQ(ToFailureDetail(ToFailureKey(5, 0, 0)), 0U);
  EXPECT_EQ(ToFailureDetail(ToFailureKey(7, kLargestPosition, 0)), 0U);
  EXPECT_EQ(ToFailureDetail(ToFailureKey(4, 1, 25)), 25U);
  EXPECT_EQ(ToFailureDetail(kNoFailureKey), 255U);
}

TEST(ComputeFailureCells, KeyOfEachStep_GivesTheChecksCells) {
  // Step 2: the repeated cell at its second position.
  const CellId repeated_cell = ToCell(1, -1, 2, 3);
  const std::vector<CellId> overlap_cells = ToSortedCells({ToCell(1, 0, 2, 3), repeated_cell, repeated_cell});
  const std::uint32_t overlap_position = FindByScan(overlap_cells, repeated_cell) + 1;
  ExpectSameCheck(
      ComputeFailureCells(
          ToUnitFrame(3),
          nullptr,
          0,
          overlap_cells.data(),
          ToCount(overlap_cells),
          ToFailureKey(2, overlap_position, 0)
      ),
      {.error = GridError::kCellsOverlap, .cell = repeated_cell, .other_cell = repeated_cell, .detail = 0}
  );
  // Step 3: seven of eight siblings, at position 3, missing eighth 5 at walk step 0.
  const std::vector<CellId> sibling_cells = BuildSevenOfEightSiblings();
  ExpectSameCheck(
      ComputeFailureCells(
          ToUnitFrame(3),
          nullptr,
          0,
          sibling_cells.data(),
          ToCount(sibling_cells),
          ToFailureKey(3, 3, 5)
      ),
      {
          .error = GridError::kSiblingCellMissing,
          .cell = ToCell(1, -1, 0, 1),
          .other_cell = ToCell(0, -1, 0, 0),
          .detail = 5,
      }
  );
  // Step 4: the corner touch, size level 2 {4,4,4} after size level 0 {0,0,0}, at position 1, direction 0.
  const std::vector<CellId> size_step_cells = ToSortedCells({ToCell(2, 4, 4, 4), ToCell(0, 0, 0, 0)});
  ExpectSameCheck(
      ComputeFailureCells(
          ToUnitFrame(3),
          nullptr,
          0,
          size_step_cells.data(),
          ToCount(size_step_cells),
          ToFailureKey(4, 1, 0)
      ),
      {
          .error = GridError::kSizeStepRuleBroken,
          .cell = ToCell(2, 4, 4, 4),
          .other_cell = ToCell(0, 0, 0, 0),
          .detail = 0,
      }
  );
}

TEST(ComputeFailureCells, RoomSteps_GiveTooManyCellBlocks) {
  const std::vector<CellId> sorted_cells = BuildSevenOfEightSiblings();
  for (const int step : {5, 6}) {
    ExpectSameCheck(
        ComputeFailureCells(
            ToUnitFrame(3),
            nullptr,
            0,
            sorted_cells.data(),
            ToCount(sorted_cells),
            ToFailureKey(step, 0, 0)
        ),
        {.error = GridError::kTooManyCellBlocks, .cell = CellId{0}, .other_cell = CellId{0}, .detail = 0}
    );
  }
}

TEST(ComputeFailureCells, KeyWithoutFailureAtItsPosition_IsInternalFault) {
  // A valid list: no check fails, so a key of step 2, 3 or 4 contradicts the list.
  const std::vector<CellId> sorted_cells = BuildPathToHighCoordLimit(1, -1);
  const CellCheck internal_fault = {.error = GridError::kInternalFault};
  for (const int step : {2, 3, 4}) {
    ExpectSameCheck(
        ComputeFailureCells(
            ToUnitFrame(3),
            nullptr,
            0,
            sorted_cells.data(),
            ToCount(sorted_cells),
            ToFailureKey(step, 0, 0)
        ),
        internal_fault
    );
  }
}

TEST(ComputeFailureCells, PositionPastTheList_IsInternalFault) {
  const std::vector<CellId> sorted_cells = BuildSevenOfEightSiblings();
  ExpectSameCheck(
      ComputeFailureCells(
          ToUnitFrame(3),
          nullptr,
          0,
          sorted_cells.data(),
          ToCount(sorted_cells),
          ToFailureKey(3, ToCount(sorted_cells), 0)
      ),
      {.error = GridError::kInternalFault}
  );
}

TEST(ComputeFailureCells, StepOutsideTheChecks_IsInternalFault) {
  const std::vector<CellId> sorted_cells = BuildSevenOfEightSiblings();
  for (const std::uint64_t failure_key : {ToFailureKey(0, 0, 0), ToFailureKey(8, 0, 0), kNoFailureKey}) {
    ExpectSameCheck(
        ComputeFailureCells(ToUnitFrame(3), nullptr, 0, sorted_cells.data(), ToCount(sorted_cells), failure_key),
        {.error = GridError::kInternalFault}
    );
  }
}

// Step 7, a block without its bigger block, is a fault in the build: no cell is named, whatever the position.
TEST(ComputeFailureCells, StepSeven_IsInternalFaultWithNoCell) {
  const std::vector<CellId> sorted_cells = BuildSevenOfEightSiblings();
  for (const std::uint64_t block : {0U, 3U}) {
    ExpectSameCheck(
        ComputeFailureCells(
            ToUnitFrame(3),
            nullptr,
            0,
            sorted_cells.data(),
            ToCount(sorted_cells),
            ToFailureKey(7, block, 0)
        ),
        {.error = GridError::kInternalFault, .cell = CellId{0}, .other_cell = CellId{0}, .detail = 0}
    );
  }
}

// An input list in input order: a valid id, three bad ids and a valid one, checked against a frame of 3 size levels.
std::vector<CellId> BuildInputWithBadIds() {
  return {
      ToCell(2, -kMaxCoord, kMaxCoord, 0),
      ToCell(0, kMaxCoord + 1, 0, 0),
      ToCell(3, 0, 0, 0),
      CellId{0},
      ToCell(1, 5, -6, 7),
  };
}

TEST(ComputeFailureCells, IdStepKey_GivesTheInputIdAndItsCheck) {
  // Input index 1 is kCoordOutOfRange, 2 a size level beyond the frame, 3 no cell id: each named as given.
  const std::vector<CellId> input_cells = BuildInputWithBadIds();
  const std::vector<CellId> sorted_cells = ToSortedCells(input_cells);
  for (std::uint32_t input_index = 1; input_index <= 3; ++input_index) {
    const CellId input_cell = input_cells.at(input_index);
    ExpectSameCheck(
        ComputeFailureCells(
            ToUnitFrame(3),
            input_cells.data(),
            ToCount(input_cells),
            sorted_cells.data(),
            ToCount(sorted_cells),
            ToFailureKey(1, input_index, 0)
        ),
        {.error = CheckCellId(ToUnitFrame(3), input_cell), .cell = input_cell}
    );
  }
  // The expected errors, worked out from the ids: 2^19 - 1 stored on x, size level 3 of 3 size levels, bit 63 clear.
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), input_cells.at(1)), GridError::kCoordOutOfRange);
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), input_cells.at(2)), GridError::kBadInput);
  EXPECT_EQ(CheckCellId(ToUnitFrame(3), input_cells.at(3)), GridError::kBadInput);
}

TEST(ComputeFailureCells, IdStepKeyAtValidId_IsInternalFault) {
  const std::vector<CellId> input_cells = BuildInputWithBadIds();
  for (const std::uint32_t input_index : {0U, 4U}) {
    ExpectSameCheck(
        ComputeFailureCells(
            ToUnitFrame(3),
            input_cells.data(),
            ToCount(input_cells),
            nullptr,
            0,
            ToFailureKey(1, input_index, 0)
        ),
        {.error = GridError::kInternalFault}
    );
  }
}

TEST(ComputeFailureCells, IdStepKeyPastTheInput_IsInternalFault) {
  const std::vector<CellId> input_cells = BuildInputWithBadIds();
  ExpectSameCheck(
      ComputeFailureCells(
          ToUnitFrame(3),
          input_cells.data(),
          ToCount(input_cells),
          nullptr,
          0,
          ToFailureKey(1, ToCount(input_cells), 0)
      ),
      {.error = GridError::kInternalFault}
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// Split cells: method paper <proc-build-grid> step 3, <proc-build-grid-in-parallel> step P7
// ---------------------------------------------------------------------------------------------------------------------

// The split cells of a valid list, by brute force: every cell of a bigger size level whose unit box holds an active
// cell's low corner. The random grids lie in 0..3 m, so every unit is 0 or more and plain division is the floor.
std::set<std::uint64_t> BuildSplitCellsByUnitBoxes(const std::vector<CellId>& cells, int size_level_count) {
  std::set<std::uint64_t> split_cells;
  for (const CellId cell : cells) {
    const UnitBox box = ToUnitBox(cell, size_level_count);
    for (int bigger_size_level = 0; bigger_size_level < static_cast<int>(ToSizeLevel(cell)); ++bigger_size_level) {
      const auto width_exponent = static_cast<unsigned>(size_level_count - 1 - bigger_size_level);
      const auto width = static_cast<std::int64_t>(std::uint64_t{1} << width_exponent);
      const CellId split_cell = ToCell(
          bigger_size_level,
          static_cast<std::int32_t>(box.low.at(0) / width),
          static_cast<std::int32_t>(box.low.at(1) / width),
          static_cast<std::int32_t>(box.low.at(2) / width)
      );
      split_cells.insert(split_cell.value);
    }
  }
  return split_cells;
}

TEST(ComputeSplitCellCountToAdd, RandomGrids_AddEachSplitCellOnce) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<CellId>& cells = BuildSortedRandomGrid(seed);
    std::vector<std::uint64_t> added_cells;
    for (const CellId cell : cells) {
      const int size_level = static_cast<int>(ToSizeLevel(cell));
      const int added_count = ComputeSplitCellCountToAdd(cell);
      for (int size_level_steps = 1; size_level_steps <= added_count; ++size_level_steps) {
        added_cells.push_back(
            ComputeBigCellAtSizeLevel(cell, static_cast<SizeLevel>(size_level - size_level_steps)).value
        );
      }
    }
    const std::set<std::uint64_t> distinct_added_cells(added_cells.begin(), added_cells.end());
    EXPECT_EQ(distinct_added_cells.size(), added_cells.size()) << seed;
    EXPECT_EQ(distinct_added_cells, BuildSplitCellsByUnitBoxes(cells, 4)) << seed;
    // Every split cell has 8 small cells, active or split, so the split cells are at most n / 7 of n active cells:
    // the bound the sort count of method paper <sec-parallel-build> rests on.
    EXPECT_LE(added_cells.size(), cells.size() / 7) << seed;
  }
}

TEST(ComputeSplitCellCountToAdd, EighthZeroChainFromSizeLevelFifteen_AddsFifteen) {
  // Every coordinate of the chain 0 is even at every size level; so is -2^15 at size level 15, -2^14 at 14, ..., -2
  // at size level 1.
  EXPECT_EQ(ComputeSplitCellCountToAdd(ToCell(15, 0, 0, 0)), 15);
  EXPECT_EQ(ComputeSplitCellCountToAdd(ToCell(15, -kTwoToThe15Coord, -kTwoToThe15Coord, -kTwoToThe15Coord)), 15);
}

TEST(ComputeSplitCellCountToAdd, CellsAtCoordLimits_StopAtTheFirstOtherEighth) {
  // kMaxCoord is odd: eighth 1, 2 or 4 of its big cell, so nothing is added.
  EXPECT_EQ(ComputeSplitCellCountToAdd(ToCell(15, kMaxCoord, 0, 0)), 0);
  EXPECT_EQ(ComputeSplitCellCountToAdd(ToCell(3, 0, -kMaxCoord, 0)), 0);
  // -2^18 + 2 at size level 1 is even: it adds size level 0 -2^17 + 1, which has no big cell.
  EXPECT_EQ(ComputeSplitCellCountToAdd(ToCell(1, -kTwoToThe18 + 2, 0, 0)), 1);
  // Size level 3 {2,0,0} is eighth 0 of size level 2 {1,0,0}, which is eighth 1 of size level 1 {0,0,0}.
  EXPECT_EQ(ComputeSplitCellCountToAdd(ToCell(3, 2, 0, 0)), 1);
  EXPECT_EQ(ComputeSplitCellCountToAdd(ToCell(0, 0, 0, 0)), 0);
}

// The exclusive prefix sums of ComputeSplitCellCountToAdd over `cells`, the total at the end, by the test's own loop.
std::vector<std::uint32_t> ToSplitCellSums(const std::vector<CellId>& cells) {
  std::vector<std::uint32_t> sums(cells.size() + 1, 0);
  for (std::size_t position = 0; position < cells.size(); ++position) {
    sums[position + 1] = sums[position] + static_cast<std::uint32_t>(ComputeSplitCellCountToAdd(cells[position]));
  }
  return sums;
}

// The cell ids after ApplySplitCellWritesAt ran for every position of `positions`, one after the other: `cells`, then
// room for the split cells and two places more, all first holding the stale id.
//
// clang-tidy: the sums and the positions; the swappable-parameters check flags the two lists of integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::vector<CellId> ComputeCellIdsWithSplitCells(
    const std::vector<CellId>& cells,
    const std::vector<std::uint32_t>& sums,
    const std::vector<std::uint32_t>& positions
) {
  std::vector<CellId> cell_ids(cells.size() + sums.back() + 2, ToCell(3, 1000, 1000, 1000));
  std::ranges::copy(cells, cell_ids.begin());
  for (const std::uint32_t position : positions) {
    ApplySplitCellWritesAt(cell_ids.data(), sums.data(), ToCount(cells), position);
  }
  return cell_ids;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Expects `cell_ids` to hold `cells`, then each split cell of `cells` once, then the two stale ids.
void ExpectSplitCellsAppended(
    const std::vector<CellId>& cell_ids,
    const std::vector<CellId>& cells,
    std::uint64_t seed
) {
  const auto active_end = cell_ids.begin() + static_cast<std::ptrdiff_t>(cells.size());
  const auto used_end = cell_ids.end() - 2;
  const std::vector<std::uint64_t> appended_values = ToValues(std::vector<CellId>(active_end, used_end));
  const std::set<std::uint64_t> distinct_values(appended_values.begin(), appended_values.end());
  EXPECT_EQ(distinct_values.size(), appended_values.size()) << seed;
  EXPECT_EQ(distinct_values, BuildSplitCellsByUnitBoxes(cells, 4)) << seed;
  EXPECT_EQ(ToValues(std::vector<CellId>(cell_ids.begin(), active_end)), ToValues(cells)) << seed;
  const std::vector<std::uint64_t> stale_values = {
      ToCell(3, 1000, 1000, 1000).value,
      ToCell(3, 1000, 1000, 1000).value,
  };
  EXPECT_EQ(ToValues(std::vector<CellId>(used_end, cell_ids.end())), stale_values) << seed;
}

// Puts positions in an order drawn from `seed`, the same on every machine: the shuffle of ApplyFixedShuffle.
void ApplyFixedPositionShuffle(std::vector<std::uint32_t>& positions, std::uint64_t seed) {
  std::mt19937_64 random_numbers(seed);
  for (std::size_t index = positions.size(); index > 1; --index) {
    std::swap(positions[index - 1], positions[ToUniformIndex(random_numbers(), index)]);
  }
}

// Each position writes its own places, so the threads of P7 (c) may run in any order: here first to last, last to
// first and one seeded shuffle per grid. The places after the split cells keep the stale id.
TEST(ApplySplitCellWritesAt, RandomGrids_AppendExactlyTheSplitCells) {
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    const std::vector<CellId>& cells = BuildSortedRandomGrid(seed);
    const std::vector<std::uint32_t> sums = ToSplitCellSums(cells);
    std::vector<std::uint32_t> positions(cells.size());
    for (std::uint32_t position = 0; position < ToCount(cells); ++position) {
      positions[position] = position;
    }
    ExpectSplitCellsAppended(ComputeCellIdsWithSplitCells(cells, sums, positions), cells, seed);
    std::ranges::reverse(positions);
    ExpectSplitCellsAppended(ComputeCellIdsWithSplitCells(cells, sums, positions), cells, seed);
    ApplyFixedPositionShuffle(positions, seed + 2000);
    ExpectSplitCellsAppended(ComputeCellIdsWithSplitCells(cells, sums, positions), cells, seed);
  }
}

// P8 orders only the sort count's places, so cells beyond it, which the n / 7 bound rules out, are an internal fault,
// the key of step 5 with detail 1; the room's key of step 5, detail 0, is smaller and wins when both hold.
TEST(ComputeCellPlacesFailureKey, CellsBeyondSortCount_IsTheInternalFaultKey) {
  static_assert(kSortCountFaultKey == ToFailureKey(kCellPlacesStep, 0, 1));
  EXPECT_EQ(ComputeCellPlacesFailureKey(16, 64, 15), kSortCountFaultKey);
  EXPECT_EQ(ComputeCellPlacesFailureKey(16, 64, 16), kNoFailureKey);
  EXPECT_EQ(ComputeCellPlacesFailureKey(72, 64, 15), ToFailureKey(kCellPlacesStep, 0, 0));
  // Every later step stops on the fault with the one rule of an earlier step.
  EXPECT_TRUE(ComputeHasEarlierStepFailed(kSortCountFaultKey, kBlockCountStep));
  EXPECT_TRUE(ComputeHasEarlierStepFailed(kSortCountFaultKey, kBlockFaultStep));
}

// The fault names no cell; the room's key of step 5 stays kTooManyCellBlocks; a detail of 2 or more is no key of
// step 5.
TEST(ComputeFailureCells, CellPlacesKeys_GiveTheirErrors) {
  const auto to_error = [](std::uint64_t failure_key) {
    return ComputeFailureCells(GridFrame(), nullptr, 0, nullptr, 0, failure_key);
  };
  ExpectSameCheck(to_error(kSortCountFaultKey), {.error = GridError::kInternalFault});
  ExpectSameCheck(to_error(ToFailureKey(kCellPlacesStep, 0, 0)), {.error = GridError::kTooManyCellBlocks});
  ExpectSameCheck(to_error(ToFailureKey(kCellPlacesStep, 0, 2)), {.error = GridError::kInternalFault});
}

TEST(ComputeRoomFailureKey, OneAboveRoom_IsTheStepsKey) {
  EXPECT_EQ(ComputeRoomFailureKey(513, 512, kCellPlacesStep), ToFailureKey(kCellPlacesStep, 0, 0));
  EXPECT_EQ(ComputeRoomFailureKey(512, 512, kCellPlacesStep), kNoFailureKey);
  EXPECT_EQ(ComputeRoomFailureKey(9, 8, kBlockCountStep), ToFailureKey(kBlockCountStep, 0, 0));
}

// 64 x kMaxCellBlocksLimit = 1,145,324,608 places: the largest sort count, which fits 32 bits although n + n / 7 does
// not always.
TEST(ComputeSortCount, LargestRoom_FitsThirtyTwoBits) {
  constexpr std::uint32_t kLargestRoomCellPlaces = 1145324608;
  static_assert(std::uint64_t{kCellPlacesPerBlock} * kMaxCellBlocksLimit == kLargestRoomCellPlaces);
  // n + n / 7 is 1,308,942,409 here, above the room's cell places.
  EXPECT_EQ(ComputeSortCount(kLargestRoomCellPlaces, kMaxCellBlocksLimit), kLargestRoomCellPlaces);
  // n + n / 7 is 4,908,534,051 here, above 2^32.
  EXPECT_EQ(ComputeSortCount(4294967295U, kMaxCellBlocksLimit), kLargestRoomCellPlaces);
  EXPECT_EQ(ComputeSortCount(14, 100), 16U);
  EXPECT_EQ(ComputeSortCount(9, 100), 10U);
  // n / 6 would give 7 here.
  EXPECT_EQ(ComputeSortCount(6, 100), 6U);
  EXPECT_EQ(ComputeSortCount(100, 1), 64U);
}

// Six size-level-0 cells in the blocks [A, A, B, C, C, C], A = (0,0,0), B = (1,0,0), C = (2,0,0), then CellId{0},
// stored in another order than sorted so that every read goes through `order`.
struct SixSortedCells {
  std::array<CellId, 7> cell_ids;
  std::array<std::uint32_t, 7> order;  // Sorted entry j is cell_ids[order[j]].
};

SixSortedCells BuildSixSortedCells() {
  const std::array<CellId, 7> sorted = {
      ToCell(0, 0, 0, 0),
      ToCell(0, 1, 0, 0),
      ToCell(0, 4, 0, 0),
      ToCell(0, 8, 0, 0),
      ToCell(0, 9, 1, 0),
      ToCell(0, 10, 0, 3),
      CellId{0},
  };
  SixSortedCells six = {
      .cell_ids = std::array<CellId, 7>(),
      .order = std::array<std::uint32_t, 7>{2, 5, 3, 4, 6, 0, 1},
  };
  for (std::size_t position = 0; position < sorted.size(); ++position) {
    six.cell_ids.at(six.order.at(position)) = sorted.at(position);
  }
  return six;
}

TEST(ComputeBlockStartMark, SixSortedCells_MarkEachRunOnce) {
  const SixSortedCells six = BuildSixSortedCells();
  std::array<std::uint32_t, 7> marks{};
  std::array<std::uint32_t, 8> sums{};
  for (std::uint32_t position = 0; position < 7; ++position) {
    marks.at(position) = ComputeBlockStartMark(six.cell_ids.data(), six.order.data(), position);
    sums.at(position + 1) = sums.at(position) + marks.at(position);
  }
  EXPECT_EQ(marks, (std::array<std::uint32_t, 7>{1, 0, 1, 1, 0, 0, 0}));
  EXPECT_EQ(sums, (std::array<std::uint32_t, 8>{0, 1, 1, 2, 3, 3, 3, 3}));
  std::array<std::uint32_t, 6> block_indexes{};
  for (std::uint32_t position = 0; position < 6; ++position) {
    block_indexes.at(position) =
        ComputeBlockIndexOfSortedEntry(six.cell_ids.data(), six.order.data(), sums.data(), position);
  }
  EXPECT_EQ(block_indexes, (std::array<std::uint32_t, 6>{0, 0, 1, 2, 2, 2}));
}

// The id 0 at position 0 is no cell, so not even the first entry marks it.
TEST(ComputeBlockStartMark, IdZeroFirst_IsNotMarked) {
  const std::array<CellId, 1> cell_ids = {CellId{0}};
  const std::array<std::uint32_t, 1> order = {0};
  EXPECT_EQ(ComputeBlockStartMark(cell_ids.data(), order.data(), 0), 0U);
}

TEST(ApplyBlockIdWriteAt, SixSortedCells_WriteEachBlockOnceInOrder) {
  const SixSortedCells six = BuildSixSortedCells();
  const std::array<std::uint32_t, 8> sums = {0, 1, 1, 2, 3, 3, 3, 3};
  const CellBlockId stale_block = ToCellBlockId(ToCell(3, 28, 28, 28));
  std::array<CellBlockId, 4> block_ids = {stale_block, stale_block, stale_block, stale_block};
  for (std::uint32_t visited = 0; visited < 7; ++visited) {
    ApplyBlockIdWriteAt(block_ids.data(), six.cell_ids.data(), six.order.data(), sums.data(), 6 - visited);
  }
  EXPECT_EQ(block_ids.at(0).value, ToCellBlockId(ToCell(0, 0, 0, 0)).value);
  EXPECT_EQ(block_ids.at(1).value, ToCellBlockId(ToCell(0, 4, 0, 0)).value);
  EXPECT_EQ(block_ids.at(2).value, ToCellBlockId(ToCell(0, 8, 0, 0)).value);
  EXPECT_EQ(block_ids.at(3).value, stale_block.value);
}

// ---------------------------------------------------------------------------------------------------------------------
// Neighbour tables, cell states and face piece kinds: method paper <proc-build-grid> steps 6 to 8
// ---------------------------------------------------------------------------------------------------------------------

constexpr std::uint32_t kSameSizeLevelEntriesPerBlock = 27;
constexpr std::uint32_t kSmallerBlocksPerBlock = 8;
constexpr std::uint32_t kSelfEntry = 13;  // The offset (0, 0, 0): ((0 + 1) * 3 + (0 + 1)) * 3 + (0 + 1).

// The tables of one grid in test vectors; ToGridTables points a GridTables into them.
struct TableVectors {
  GridFrame frame{};
  std::vector<CellBlockId> block_ids;
  std::vector<std::uint32_t> same_size_level_blocks;
  std::vector<std::uint32_t> bigger_block;
  std::vector<std::uint32_t> smaller_blocks;
  std::vector<CellState> cell_states;
  std::vector<FacePieceKind> face_piece_kinds;

  [[nodiscard]] GridTables ToGridTables() const {
    return {
        .frame = frame,
        .version = GridVersion{1},
        .block_count = static_cast<std::uint32_t>(block_ids.size()),
        .block_ids = block_ids.data(),
        .same_size_level_blocks = same_size_level_blocks.data(),
        .bigger_block = bigger_block.data(),
        .smaller_blocks = smaller_blocks.data(),
        .cell_states = cell_states.data(),
        .face_piece_kinds = face_piece_kinds.data(),
    };
  }
};

// Tables for the given blocks with no neighbour, every cell place kAbsent and every face place kNotAPiece.
TableVectors BuildEmptyTables(GridFrame frame, std::vector<CellBlockId> block_ids) {
  const std::size_t block_count = block_ids.size();
  return {
      .frame = frame,
      .block_ids = std::move(block_ids),
      .same_size_level_blocks = std::vector<std::uint32_t>(block_count * kSameSizeLevelEntriesPerBlock, kNoIndex),
      .bigger_block = std::vector<std::uint32_t>(block_count, kNoIndex),
      .smaller_blocks = std::vector<std::uint32_t>(block_count * kSmallerBlocksPerBlock, kNoIndex),
      .cell_states = std::vector<CellState>(block_count * kCellPlacesPerBlock, CellState::kAbsent),
      .face_piece_kinds = std::vector<FacePieceKind>(block_count * kFacePlacesPerBlock, FacePieceKind::kNotAPiece),
  };
}

// The block of size_level at a block coordinate, by the bits of the cell id at that coordinate (grid_ids.h).
CellBlockId ToBlock(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return {.value = ToCell(size_level, x, y, z).value};
}

// Cell place (x, y, z) of a block: 64 b + (4 z + y) * 4 + x.
//
// clang-tidy: a block index and three local coordinates; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::uint32_t ToCellPlace(std::uint32_t block, int x, int y, int z) {
  return (block * 64) + static_cast<std::uint32_t>((((z * 4) + y) * 4) + x);
}

// Face place of a block on an axis, local (x, y, z), the coordinate along the axis being the plane 0..4: 240 b + 80
// axis + (z * 4 + y) * 5 + x on X, (z * 5 + y) * 4 + x on Y, (z * 4 + y) * 4 + x on Z.
//
// clang-tidy: a block index, an axis and three local coordinates; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FacePlaceIndex ToFacePlace(std::uint32_t block, Axis axis, int x, int y, int z) {
  int place_on_axis = 0;
  if (axis == Axis::kX) {
    place_on_axis = (((z * 4) + y) * 5) + x;
  } else if (axis == Axis::kY) {
    place_on_axis = (((z * 5) + y) * 4) + x;
  } else {
    place_on_axis = (((z * 4) + y) * 4) + x;
  }
  const auto axis_number = static_cast<std::uint32_t>(axis);
  return {.value = (block * 240) + (axis_number * 80) + static_cast<std::uint32_t>(place_on_axis)};
}

// Writes the 27, 1 and 8 entries of every block with FindBlockNeighbours.
void ApplyFoundNeighbours(TableVectors& tables) {
  const auto block_count = static_cast<std::uint32_t>(tables.block_ids.size());
  for (std::uint32_t block = 0; block < block_count; ++block) {
    FindBlockNeighbours(
        tables.block_ids.data(),
        block_count,
        CellBlockIndex{block},
        tables.same_size_level_blocks.data(),
        tables.bigger_block.data(),
        tables.smaller_blocks.data()
    );
  }
}

// The two-size fixture: size level 0 {0,0,0} active and its neighbour {1,0,0} split into its 8 size-level-1 cells.
// Block 0 is size level 0 (0,0,0), which holds both big cells; block 1 is size level 1 (0,0,0), which holds the small
// cells at x 2..3, y 0..1, z 0..1. Both blocks have the low corner 0, so the bigger block comes first. The neighbour
// tables are written by hand: no block has a same-size-level neighbour, block 1's bigger block is 0, and block 1 is
// block 0's smaller block of eighth 0. Every state is written by hand except the places of block 1 inside the big
// cell, x 0..1, y 0..1, z 0..1, which are left kAbsent here.
TableVectors BuildTwoSizeFixtureWithoutCover() {
  TableVectors tables = BuildEmptyTables(ToUnitFrame(2), {ToBlock(0, 0, 0, 0), ToBlock(1, 0, 0, 0)});
  tables.same_size_level_blocks.at((0 * kSameSizeLevelEntriesPerBlock) + kSelfEntry) = 0;
  tables.same_size_level_blocks.at((1 * kSameSizeLevelEntriesPerBlock) + kSelfEntry) = 1;
  tables.bigger_block.at(1) = 0;
  tables.smaller_blocks.at((0 * kSmallerBlocksPerBlock) + 0) = 1;
  tables.cell_states.at(ToCellPlace(0, 0, 0, 0)) = CellState::kActive;
  tables.cell_states.at(ToCellPlace(0, 1, 0, 0)) = CellState::kSplit;
  for (int z = 0; z <= 1; ++z) {
    for (int y = 0; y <= 1; ++y) {
      for (int x = 2; x <= 3; ++x) {
        tables.cell_states.at(ToCellPlace(1, x, y, z)) = CellState::kActive;
      }
    }
  }
  return tables;
}

// The two-size fixture with the places of block 1 inside the big cell marked kInsideBiggerCell by hand.
TableVectors BuildTwoSizeFixture() {
  TableVectors tables = BuildTwoSizeFixtureWithoutCover();
  for (int z = 0; z <= 1; ++z) {
    for (int y = 0; y <= 1; ++y) {
      for (int x = 0; x <= 1; ++x) {
        tables.cell_states.at(ToCellPlace(1, x, y, z)) = CellState::kInsideBiggerCell;
      }
    }
  }
  return tables;
}

TEST(FindBlockNeighbours, TwoSizeFixture_LinksBiggerAndSmaller) {
  TableVectors tables = BuildEmptyTables(ToUnitFrame(2), {ToBlock(0, 0, 0, 0), ToBlock(1, 0, 0, 0)});
  ApplyFoundNeighbours(tables);

  // Block 1, size level 1 (0,0,0), lies in size level 0 (floor(0 / 2), ...) = (0,0,0): block 0.
  EXPECT_EQ(tables.bigger_block.at(1), 0U);
  EXPECT_EQ(tables.bigger_block.at(0), kNoIndex);
  // Eighth 0 of block 0 is size level 1 (2 * 0 + 0, ...) = (0,0,0): block 1. Eighths 1..7 are at 1 on some axis.
  const std::vector<std::uint32_t> block_zero_smaller_blocks(
      tables.smaller_blocks.begin(),
      tables.smaller_blocks.begin() + kSmallerBlocksPerBlock
  );
  const std::vector<std::uint32_t> expected_smaller_blocks = {
      1,
      kNoIndex,
      kNoIndex,
      kNoIndex,
      kNoIndex,
      kNoIndex,
      kNoIndex,
      kNoIndex,
  };
  EXPECT_EQ(block_zero_smaller_blocks, expected_smaller_blocks);
  EXPECT_EQ(tables.same_size_level_blocks.at((0 * kSameSizeLevelEntriesPerBlock) + kSelfEntry), 0U);
  EXPECT_EQ(tables.same_size_level_blocks.at((1 * kSameSizeLevelEntriesPerBlock) + kSelfEntry), 1U);
}

// The size-level-1 block (0,0,0) alone has no bigger block: step 7, position 0, which ComputeFailureCells names as an
// internal fault with no cell. With the size-level-0 block (0,0,0) before it, block 1's bigger block is block 0, and
// block 0, of size level 0, never has one: no fault. No cell list reaches the first case through a build, since the
// sibling check stops it, so the test is on the function.
TEST(ComputeBlockFaultKey, BlockWithoutItsBiggerBlock_IsStepSeven) {
  TableVectors lone_tables = BuildEmptyTables(ToUnitFrame(2), {ToBlock(1, 0, 0, 0)});
  ApplyFoundNeighbours(lone_tables);
  const std::uint64_t lone_key =
      ComputeBlockFaultKey(lone_tables.block_ids.data(), lone_tables.bigger_block.data(), CellBlockIndex{0});
  EXPECT_EQ(lone_key, ToFailureKey(7, 0, 0));
  ExpectSameCheck(
      ComputeFailureCells(ToUnitFrame(2), nullptr, 0, nullptr, 0, lone_key),
      {.error = GridError::kInternalFault, .cell = CellId{0}, .other_cell = CellId{0}, .detail = 0}
  );

  TableVectors two_size_tables = BuildEmptyTables(ToUnitFrame(2), {ToBlock(0, 0, 0, 0), ToBlock(1, 0, 0, 0)});
  ApplyFoundNeighbours(two_size_tables);
  for (std::uint32_t block = 0; block < 2; ++block) {
    EXPECT_EQ(
        ComputeBlockFaultKey(
            two_size_tables.block_ids.data(),
            two_size_tables.bigger_block.data(),
            CellBlockIndex{block}
        ),
        kNoFailureKey
    );
  }
}

TEST(FindBlockNeighbours, BlockAtCoordLimit_HasNoBlockBeyond) {
  // Size level 0 {-kMaxCoord,0,0} is in block (floor(-(2^18 - 1) / 4), 0, 0) = (-2^16, 0, 0), and {kMaxCoord,0,0} in
  // block (2^16 - 1, 0, 0). Each lies at one end of the range, so neither is a neighbour of the other.
  const CellBlockId low_block = ToCellBlockId(ToCell(0, -kMaxCoord, 0, 0));
  const CellBlockId high_block = ToCellBlockId(ToCell(0, kMaxCoord, 0, 0));
  EXPECT_EQ(low_block.value, ToBlock(0, -kTwoToThe16, 0, 0).value);
  EXPECT_EQ(high_block.value, ToBlock(0, kTwoToThe16 - 1, 0, 0).value);
  TableVectors tables = BuildEmptyTables(ToUnitFrame(1), {low_block, high_block});
  ApplyFoundNeighbours(tables);

  for (std::uint32_t entry = 0; entry < kSameSizeLevelEntriesPerBlock; ++entry) {
    const std::uint32_t expected_low = entry == kSelfEntry ? 0 : kNoIndex;
    const std::uint32_t expected_high = entry == kSelfEntry ? 1 : kNoIndex;
    EXPECT_EQ(tables.same_size_level_blocks.at(entry), expected_low) << entry;
    EXPECT_EQ(tables.same_size_level_blocks.at(kSameSizeLevelEntriesPerBlock + entry), expected_high) << entry;
  }
}

TEST(FindBlockNeighbours, BlockCoordBeyondRange_IsNotMadeIntoAnId) {
  // A block coordinate of -kMaxCoord - 1 would pack as kMaxCoord + 1: ((-kMaxCoord - 1) + kMaxCoord) mod 2^19 = 2^19 -
  // 1. The list holds the block at kMaxCoord + 1 that the packing would name, so only the range test keeps it out.
  const CellBlockId lowest_block = ToBlock(0, -kMaxCoord, 0, 0);
  const CellBlockId one_past_highest_block = ToBlock(0, kMaxCoord + 1, 0, 0);
  TableVectors tables = BuildEmptyTables(ToUnitFrame(1), {lowest_block, one_past_highest_block});
  ApplyFoundNeighbours(tables);

  // Entry 12 is the offset (-1, 0, 0): ((0 + 1) * 3 + (0 + 1)) * 3 + (-1 + 1).
  EXPECT_EQ(tables.same_size_level_blocks.at(12), kNoIndex);
  EXPECT_EQ(tables.same_size_level_blocks.at(kSelfEntry), 0U);
}

TEST(FindBlockNeighbours, SizeLevelFifteenBlock_HasNoSmallerBlocks) {
  TableVectors tables = BuildEmptyTables(ToUnitFrame(16), {ToBlock(14, 0, 0, 0), ToBlock(15, 0, 0, 0)});
  ApplyFoundNeighbours(tables);

  EXPECT_EQ(tables.bigger_block.at(1), 0U);
  EXPECT_EQ(tables.smaller_blocks.at(0), 1U);
  for (std::uint32_t cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    EXPECT_EQ(tables.smaller_blocks.at(kSmallerBlocksPerBlock + cell_eighth), kNoIndex) << cell_eighth;
  }
}

TEST(FindBlockNeighbours, NegativeBlocks_FindNeighboursAcrossZero) {
  // Size level 1 blocks (-1,-1,-1) and (0,0,0) touch at a corner: offset (+1, +1, +1), entry 26, from the first, and
  // (-1, -1, -1), entry 0, from the second. They lie in size level 0 blocks (floor(-1 / 2), ...) = (-1,-1,-1) and
  // (0,0,0). In the fixed order: size level 0 (-1,-1,-1), size level 1 (-1,-1,-1), size level 0 (0,0,0), size level 1
  // (0,0,0), since the size-level-1 block (-1,-1,-1) starts at -4 size-level-1 widths, the corner of the size-level-0
  // block (-1,-1,-1) is at -8.
  TableVectors tables = BuildEmptyTables(
      ToUnitFrame(2),
      {ToBlock(0, -1, -1, -1), ToBlock(1, -1, -1, -1), ToBlock(0, 0, 0, 0), ToBlock(1, 0, 0, 0)}
  );
  ApplyFoundNeighbours(tables);

  EXPECT_EQ(tables.same_size_level_blocks.at((1 * kSameSizeLevelEntriesPerBlock) + 26), 3U);
  EXPECT_EQ(tables.same_size_level_blocks.at((3 * kSameSizeLevelEntriesPerBlock) + 0), 1U);
  EXPECT_EQ(tables.same_size_level_blocks.at((0 * kSameSizeLevelEntriesPerBlock) + 26), 2U);
  EXPECT_EQ(tables.bigger_block.at(1), 0U);
  EXPECT_EQ(tables.bigger_block.at(3), 2U);
  // Size level 1 block (-1,-1,-1) is 2 (-1) + 1 on each axis: eighth 7 of size level 0 block (-1,-1,-1).
  EXPECT_EQ(tables.smaller_blocks.at((0 * kSmallerBlocksPerBlock) + 7), 1U);
  EXPECT_EQ(tables.smaller_blocks.at((2 * kSmallerBlocksPerBlock) + 0), 3U);
}

TEST(ComputeCoveredCellState, PlaceInsideActiveBigCell_IsInsideBiggerCell) {
  // Block 1 (size level 1, block coordinate 0, even: the low half) place (1, 0, 1) lies in the bigger block's place
  // (0 + floor(1 / 2), 0, 0 + floor(1 / 2)) = (0, 0, 0): size level 0 {0,0,0}, active.
  const TableVectors tables = BuildTwoSizeFixtureWithoutCover();
  EXPECT_EQ(
      ComputeCoveredCellState(tables.ToGridTables(), CellPlaceIndex{ToCellPlace(1, 1, 0, 1)}),
      CellState::kInsideBiggerCell
  );
}

TEST(ComputeCoveredCellState, PlaceInsideNoActiveBigCell_IsAbsent) {
  // Block 1 place (0, 2, 0) lies in the bigger block's place (0, 1, 0), which holds no cell.
  const TableVectors tables = BuildTwoSizeFixtureWithoutCover();
  EXPECT_EQ(
      ComputeCoveredCellState(tables.ToGridTables(), CellPlaceIndex{ToCellPlace(1, 0, 2, 0)}),
      CellState::kAbsent
  );
}

TEST(ComputeCoveredCellState, CoverAlreadyMarkedInsideBiggerCell_IsAbsent) {
  // A size-level-2 block (0,0,0), block 2, whose bigger block is block 1. Its place (0, 0, 0) lies in block 1's place
  // (0, 0, 0), marked kInsideBiggerCell: an active cell two size levels bigger covers nothing one size level down, so
  // the answer does not depend on whether that mark was written before.
  TableVectors tables = BuildTwoSizeFixture();
  tables.frame = ToUnitFrame(3);
  tables.block_ids.push_back(ToBlock(2, 0, 0, 0));
  tables.same_size_level_blocks.resize(std::size_t{3} * kSameSizeLevelEntriesPerBlock, kNoIndex);
  tables.same_size_level_blocks.at((2 * kSameSizeLevelEntriesPerBlock) + kSelfEntry) = 2;
  tables.bigger_block.push_back(1);
  tables.smaller_blocks.resize(std::size_t{3} * kSmallerBlocksPerBlock, kNoIndex);
  tables.smaller_blocks.at((1 * kSmallerBlocksPerBlock) + 0) = 2;
  tables.cell_states.resize(std::size_t{3} * kCellPlacesPerBlock, CellState::kAbsent);
  tables.face_piece_kinds.resize(std::size_t{3} * kFacePlacesPerBlock, FacePieceKind::kNotAPiece);

  EXPECT_EQ(
      ComputeCoveredCellState(tables.ToGridTables(), CellPlaceIndex{ToCellPlace(2, 0, 0, 0)}),
      CellState::kAbsent
  );
}

TEST(ComputeCoveredCellState, SizeLevelZeroPlace_IsAbsent) {
  const TableVectors tables = BuildTwoSizeFixtureWithoutCover();
  EXPECT_EQ(
      ComputeCoveredCellState(tables.ToGridTables(), CellPlaceIndex{ToCellPlace(0, 2, 0, 0)}),
      CellState::kAbsent
  );
}

// Size level 0 {-1,0,0} active and the 8 size-level-1 cells of size level 0 {0,0,0}. Blocks in the fixed order: 0 is
// size level 0 (-1,0,0), low corner x = -4 widths of a size-level-0 cell; 1 is size level 0 (0,0,0); 2 is size level
// 1 (0,0,0), at the same low corner as block 1 and after it. Neighbour tables by hand: blocks 0 and 1 are X neighbours
// (entries 14 and 12), block 2 has no size-level-1 neighbour, its bigger block is 1, and it is block 1's eighth 0.
TableVectors BuildBigCellBesideSmallCellsAcrossZero() {
  TableVectors tables =
      BuildEmptyTables(ToUnitFrame(2), {ToBlock(0, -1, 0, 0), ToBlock(0, 0, 0, 0), ToBlock(1, 0, 0, 0)});
  for (std::uint32_t block = 0; block < 3; ++block) {
    tables.same_size_level_blocks.at((block * kSameSizeLevelEntriesPerBlock) + kSelfEntry) = block;
  }
  tables.same_size_level_blocks.at((0 * kSameSizeLevelEntriesPerBlock) + 14) = 1;
  tables.same_size_level_blocks.at((1 * kSameSizeLevelEntriesPerBlock) + 12) = 0;
  tables.bigger_block.at(2) = 1;
  tables.smaller_blocks.at((1 * kSmallerBlocksPerBlock) + 0) = 2;
  // Size level 0 x = -1 is 4 (-1) + 3: local 3 of block 0.
  tables.cell_states.at(ToCellPlace(0, 3, 0, 0)) = CellState::kActive;
  tables.cell_states.at(ToCellPlace(1, 0, 0, 0)) = CellState::kSplit;
  for (int z = 0; z <= 1; ++z) {
    for (int y = 0; y <= 1; ++y) {
      for (int x = 0; x <= 1; ++x) {
        tables.cell_states.at(ToCellPlace(2, x, y, z)) = CellState::kActive;
      }
    }
  }
  return tables;
}

TEST(ComputeCellStateBeside, NoNeighbourBlockButBigCellBeside_IsInsideBiggerCell) {
  // Local (-1, 0, 0) of block 2 is size level 1 x = -1. Block 2 is the low half of block 1 on X, so the place lies
  // beyond the low side of block 1: in block 0, local 3, size level 0 x = -1, active.
  const TableVectors tables = BuildBigCellBesideSmallCellsAcrossZero();
  EXPECT_EQ(ComputeCellStateBeside(tables.ToGridTables(), CellBlockIndex{2}, -1, 0, 0), CellState::kInsideBiggerCell);
}

TEST(ComputeCellStateBeside, NoNeighbourBlockAndNoBigCellBeside_IsAbsent) {
  // Local (0, -1, 0) of block 2 lies beyond the low Y side of block 1, where no block is.
  const TableVectors tables = BuildBigCellBesideSmallCellsAcrossZero();
  EXPECT_EQ(ComputeCellStateBeside(tables.ToGridTables(), CellBlockIndex{2}, 0, -1, 0), CellState::kAbsent);
}

TEST(ComputeCellStateBeside, HighSideInsideBiggerBlock_IsCoveredThere) {
  // Local (4, 0, 0) of block 2 is size level 1 x = 4: block 2 is the low half of block 1, so the place lies in block 1
  // itself, local 2 + ... = floor(4 / 2) = 2, which holds no active cell.
  const TableVectors tables = BuildBigCellBesideSmallCellsAcrossZero();
  EXPECT_EQ(ComputeCellStateBeside(tables.ToGridTables(), CellBlockIndex{2}, 4, 0, 0), CellState::kAbsent);
}

TEST(ComputeCellStateBeside, NeighbourBlock_GivesItsStoredState) {
  // Local (-1, 0, 0) of block 1 is local 3 of block 0: active. Local (4, 0, 0) of block 0 is local 0 of block 1: split.
  const TableVectors tables = BuildBigCellBesideSmallCellsAcrossZero();
  EXPECT_EQ(ComputeCellStateBeside(tables.ToGridTables(), CellBlockIndex{1}, -1, 0, 0), CellState::kActive);
  EXPECT_EQ(ComputeCellStateBeside(tables.ToGridTables(), CellBlockIndex{0}, 4, 0, 0), CellState::kSplit);
}

TEST(ComputeCellStateBeside, SizeLevelZeroWithoutNeighbour_IsAbsent) {
  // Local (-1, 0, 0) of block 0 is size level 0 x = -5, in no block; size level 0 has no bigger cell.
  const TableVectors tables = BuildBigCellBesideSmallCellsAcrossZero();
  EXPECT_EQ(ComputeCellStateBeside(tables.ToGridTables(), CellBlockIndex{0}, -1, 0, 0), CellState::kAbsent);
}

TEST(ComputeFacePieceKind, TwoSizeFixture_GivesFourQuarterFacePieces) {
  const TableVectors tables = BuildTwoSizeFixture();
  const GridTables grid_tables = tables.ToGridTables();

  // Plane 2 of block 1 on X: local cell 1 (kInsideBiggerCell) below, local cell 2 above, active at y 0..1, z 0..1.
  int between_cells_count = 0;
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < 4; ++y) {
      const FacePieceKind kind = ComputeFacePieceKind(grid_tables, ToFacePlace(1, Axis::kX, 2, y, z));
      const bool is_beside_active_cell = y <= 1 && z <= 1;
      EXPECT_EQ(kind, is_beside_active_cell ? FacePieceKind::kBetweenCells : FacePieceKind::kNotAPiece) << y << z;
      between_cells_count += kind == FacePieceKind::kBetweenCells ? 1 : 0;
    }
  }
  EXPECT_EQ(between_cells_count, 4);
  // Plane 1 of block 0 on X lies between the active big cell and the split one.
  EXPECT_EQ(ComputeFacePieceKind(grid_tables, ToFacePlace(0, Axis::kX, 1, 0, 0)), FacePieceKind::kNotAPiece);
}

// Size level 0 {3,0,0}, local 3 of block (0,0,0), alone or with {4,0,0}, local 0 of block (1,0,0). Neighbour tables
// by hand: with both, the two blocks are X neighbours.
TableVectors BuildCellBeforeBlockSide() {
  TableVectors tables = BuildEmptyTables(ToUnitFrame(1), {ToBlock(0, 0, 0, 0)});
  tables.same_size_level_blocks.at(kSelfEntry) = 0;
  tables.cell_states.at(ToCellPlace(0, 3, 0, 0)) = CellState::kActive;
  return tables;
}

TableVectors BuildCellsAcrossBlockSide() {
  TableVectors tables = BuildEmptyTables(ToUnitFrame(1), {ToBlock(0, 0, 0, 0), ToBlock(0, 1, 0, 0)});
  tables.same_size_level_blocks.at(kSelfEntry) = 0;
  tables.same_size_level_blocks.at(kSameSizeLevelEntriesPerBlock + kSelfEntry) = 1;
  tables.same_size_level_blocks.at(14) = 1;
  tables.same_size_level_blocks.at(kSameSizeLevelEntriesPerBlock + 12) = 0;
  tables.cell_states.at(ToCellPlace(0, 3, 0, 0)) = CellState::kActive;
  tables.cell_states.at(ToCellPlace(1, 0, 0, 0)) = CellState::kActive;
  return tables;
}

TEST(ComputeFacePieceKind, PlaneFourWithHighBlock_IsNotAPiece) {
  const TableVectors tables = BuildCellsAcrossBlockSide();
  const GridTables grid_tables = tables.ToGridTables();
  EXPECT_EQ(ComputeFacePieceKind(grid_tables, ToFacePlace(0, Axis::kX, 4, 0, 0)), FacePieceKind::kNotAPiece);
  EXPECT_EQ(ComputeFacePieceKind(grid_tables, ToFacePlace(1, Axis::kX, 0, 0, 0)), FacePieceKind::kBetweenCells);
  // With {3,0,0} alone, plane 4 is the cell's high side at the grid's edge.
  const TableVectors lone_tables = BuildCellBeforeBlockSide();
  EXPECT_EQ(
      ComputeFacePieceKind(lone_tables.ToGridTables(), ToFacePlace(0, Axis::kX, 4, 0, 0)),
      FacePieceKind::kGridEdgeHigh
  );
}

TEST(ComputeFacePieceKind, LowSideBesideAbsentPlace_IsGridEdgeLow) {
  // Plane 3 of block 0 on X is the low side of the active local cell 3, beside the absent local cell 2.
  const TableVectors tables = BuildCellBeforeBlockSide();
  EXPECT_EQ(
      ComputeFacePieceKind(tables.ToGridTables(), ToFacePlace(0, Axis::kX, 3, 0, 0)),
      FacePieceKind::kGridEdgeLow
  );
}

TEST(ComputeFacePieceKind, PlacesOnYAndZ_FollowTheirLayouts) {
  // Size level 0 {3,0,0} alone: its low Y side is plane 0 of axis Y at local (3, 0, 0), place 240 * 0 + 80 + (0 * 5 +
  // 0) * 4 + 3 = 83; its high Z side is plane 1 of axis Z at (3, 0, 1), place 160 + (1 * 4 + 0) * 4 + 3 = 179.
  const TableVectors tables = BuildCellBeforeBlockSide();
  const GridTables grid_tables = tables.ToGridTables();
  EXPECT_EQ(ToFacePlace(0, Axis::kY, 3, 0, 0).value, 83U);
  EXPECT_EQ(ToFacePlace(0, Axis::kZ, 3, 0, 1).value, 179U);
  EXPECT_EQ(ComputeFacePieceKind(grid_tables, FacePlaceIndex{83}), FacePieceKind::kGridEdgeLow);
  EXPECT_EQ(ComputeFacePieceKind(grid_tables, FacePlaceIndex{179}), FacePieceKind::kGridEdgeHigh);
  // Plane 1 of axis Y beside the cell, at local (3, 1, 0): place 80 + (0 * 5 + 1) * 4 + 3 = 87, the cell's high Y side.
  EXPECT_EQ(ComputeFacePieceKind(grid_tables, FacePlaceIndex{87}), FacePieceKind::kGridEdgeHigh);
}

// ---------------------------------------------------------------------------------------------------------------------
// The three steps together, against a direct lookup and against the geometric checker
// ---------------------------------------------------------------------------------------------------------------------

// The split cells of a valid list: every cell of a bigger size level that holds an active cell, by the test's floor.
std::set<std::uint64_t> BuildSplitCells(const std::vector<CellId>& active_cells) {
  std::set<std::uint64_t> split_cells;
  for (const CellId cell : active_cells) {
    for (int bigger_size_level = 0; bigger_size_level < static_cast<int>(ToSizeLevel(cell)); ++bigger_size_level) {
      split_cells.insert(ToBigCellByDivision(cell, bigger_size_level).value);
    }
  }
  return split_cells;
}

// The block coordinate of a cell coordinate, by the test's floor: floor(c / 4) on each axis.
CellCoord ToBlockCoordByDivision(CellCoord cell_coord) {
  return {
      .x = ToFloorOfHalving(cell_coord.x, 2),
      .y = ToFloorOfHalving(cell_coord.y, 2),
      .z = ToFloorOfHalving(cell_coord.z, 2),
  };
}

CellCoord ToBlockCoord(CellBlockId block) {
  return ToCellCoord(CellId{.value = block.value});
}

int ToSizeLevelNumber(CellBlockId block) {
  return static_cast<int>(ToSizeLevel(CellId{.value = block.value}));
}

// The block of a cell, by the test's floor.
CellBlockId ToBlockOfCell(CellId cell) {
  const CellCoord block_coord = ToBlockCoordByDivision(ToCellCoord(cell));
  return ToBlock(static_cast<int>(ToSizeLevel(cell)), block_coord.x, block_coord.y, block_coord.z);
}

// The tables of a valid list of active cells, built with the three steps under test: block ids by the test (every
// block that holds an active or split cell, in the fixed order by ComesBefore), neighbours by FindBlockNeighbours,
// active and split marks by the test, then ComputeCoveredCellState at every other place and ComputeFacePieceKind at
// every face place.
TableVectors BuildTablesOfCells(GridFrame frame, const std::vector<CellId>& active_cells) {
  const std::set<std::uint64_t> split_cells = BuildSplitCells(active_cells);
  std::vector<CellId> active_and_split_cells = active_cells;
  for (const std::uint64_t split_cell : split_cells) {
    active_and_split_cells.push_back(CellId{split_cell});
  }
  std::set<std::uint64_t> block_values;
  for (const CellId cell : active_and_split_cells) {
    block_values.insert(ToBlockOfCell(cell).value);
  }
  std::vector<CellBlockId> block_ids;
  block_ids.reserve(block_values.size());
  for (const std::uint64_t block_value : block_values) {
    block_ids.push_back(CellBlockId{block_value});
  }
  std::ranges::sort(block_ids, [](CellBlockId first, CellBlockId second) {
    return ComesBefore(first, second);
  });
  std::map<std::uint64_t, std::uint32_t> block_index_by_id;
  for (std::size_t block = 0; block < block_ids.size(); ++block) {
    block_index_by_id[block_ids.at(block).value] = static_cast<std::uint32_t>(block);
  }

  TableVectors tables = BuildEmptyTables(frame, block_ids);
  ApplyFoundNeighbours(tables);
  for (const CellId cell : active_and_split_cells) {
    const CellCoord cell_coord = ToCellCoord(cell);
    const CellCoord block_coord = ToBlockCoordByDivision(cell_coord);
    const std::uint32_t place = ToCellPlace(
        block_index_by_id.at(ToBlockOfCell(cell).value),
        cell_coord.x - (4 * block_coord.x),
        cell_coord.y - (4 * block_coord.y),
        cell_coord.z - (4 * block_coord.z)
    );
    tables.cell_states.at(place) = split_cells.contains(cell.value) ? CellState::kSplit : CellState::kActive;
  }
  for (std::uint32_t place = 0; place < tables.cell_states.size(); ++place) {
    const CellState state = tables.cell_states.at(place);
    if (state != CellState::kActive && state != CellState::kSplit) {
      tables.cell_states.at(place) = ComputeCoveredCellState(tables.ToGridTables(), CellPlaceIndex{place});
    }
  }
  for (std::uint32_t face = 0; face < tables.face_piece_kinds.size(); ++face) {
    tables.face_piece_kinds.at(face) = ComputeFacePieceKind(tables.ToGridTables(), FacePlaceIndex{face});
  }
  return tables;
}

// A face piece as (id, kind). The face pieces of a grid are compared as multisets, so a face piece kept twice fails.
using FacePieceWithKind = std::pair<std::uint64_t, FacePieceKind>;

// The face pieces of the tables: every face place that is not kNotAPiece, with the id of its axis, its block's size
// level and the cell place on its high side, 4 b + the local coordinate, the plane along the axis. Plane 4 of a block
// and plane 0 of the block above give one id, so the id cannot say which block keeps the face piece; it also expects
// that a face piece on plane 4 has no block of the same size level on its high side, where plane 0 would keep it.
std::multiset<FacePieceWithKind> BuildFacePiecesOfTables(const TableVectors& tables) {
  const std::vector<std::uint64_t> block_list = [&tables] {
    std::vector<std::uint64_t> values;
    values.reserve(tables.block_ids.size());
    for (const CellBlockId block : tables.block_ids) {
      values.push_back(block.value);
    }
    return values;
  }();
  const std::set<std::uint64_t> block_values(block_list.begin(), block_list.end());
  std::multiset<FacePieceWithKind> pieces;
  for (std::uint32_t face = 0; face < tables.face_piece_kinds.size(); ++face) {
    const FacePieceKind kind = tables.face_piece_kinds.at(face);
    if (kind == FacePieceKind::kNotAPiece) {
      continue;
    }
    const CellBlockId block = tables.block_ids.at(face / 240);
    const auto axis_number = static_cast<std::size_t>((face % 240) / 80);
    // x varies fastest, then y, then z; along the axis the plane takes 5 values, on the other two axes 4.
    std::array<std::int32_t, 3> local_coord{};
    auto rest = static_cast<std::int32_t>(face % 80);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const std::int32_t extent = axis == axis_number ? 5 : 4;
      local_coord.at(axis) = rest % extent;
      rest /= extent;
    }
    const CellCoord block_coord = ToBlockCoord(block);
    const CellCoord cell_high_coord = {
        .x = (4 * block_coord.x) + local_coord.at(0),
        .y = (4 * block_coord.y) + local_coord.at(1),
        .z = (4 * block_coord.z) + local_coord.at(2),
    };
    const auto axis = static_cast<Axis>(axis_number);
    const auto size_level = static_cast<SizeLevel>(ToSizeLevelNumber(block));
    if (local_coord.at(axis_number) == 4) {
      std::array<std::int32_t, 3> high_block_coord = {block_coord.x, block_coord.y, block_coord.z};
      high_block_coord.at(axis_number) += 1;
      const CellBlockId high_block =
          ToBlock(ToSizeLevelNumber(block), high_block_coord.at(0), high_block_coord.at(1), high_block_coord.at(2));
      EXPECT_FALSE(block_values.contains(high_block.value)) << "face place " << face << " is kept on plane 4";
    }
    pieces.insert({ToFacePieceId(axis, size_level, cell_high_coord).value, kind});
  }
  return pieces;
}

// The face pieces of the geometric checker, the kind from its cells: both sides kBetweenCells, only the high side
// kGridEdgeLow, only the low side kGridEdgeHigh.
std::multiset<FacePieceWithKind> BuildFacePiecesOfChecker(GridFrame frame, const std::vector<CellId>& active_cells) {
  const GeometricGrid grid(frame, active_cells);
  std::multiset<FacePieceWithKind> pieces;
  for (const GeometricFacePiece& piece : grid.FacePieces()) {
    FacePieceKind kind = FacePieceKind::kBetweenCells;
    if (!piece.cell_low.has_value()) {
      kind = FacePieceKind::kGridEdgeLow;
    } else if (!piece.cell_high.has_value()) {
      kind = FacePieceKind::kGridEdgeHigh;
    }
    pieces.insert({piece.id.value, kind});
  }
  return pieces;
}

// The tables of the 200 random grids, built once for the whole file; seed s is entry s - 1.
const std::vector<TableVectors>& BuildRandomGridTables() {
  static const std::vector<TableVectors> grid_tables = [] {
    std::vector<TableVectors> tables;
    tables.reserve(200);
    for (std::uint64_t seed = 1; seed <= 200; ++seed) {
      tables.push_back(BuildTablesOfCells(ToUnitFrame(4), BuildSortedRandomGrid(seed)));
    }
    return tables;
  }();
  return grid_tables;
}

// The index of a block in sorted block ids by a map, or kNoIndex: a lookup apart from the binary search.
class BlockLookup {
 public:
  explicit BlockLookup(const std::vector<CellBlockId>& block_ids) {
    for (std::size_t block = 0; block < block_ids.size(); ++block) {
      block_index_by_id_[block_ids.at(block).value] = static_cast<std::uint32_t>(block);
    }
  }

  [[nodiscard]] std::uint32_t FindBlock(int size_level, CellCoord block_coord) const {
    const auto found = block_index_by_id_.find(ToBlock(size_level, block_coord.x, block_coord.y, block_coord.z).value);
    return found == block_index_by_id_.end() ? kNoIndex : found->second;
  }

 private:
  std::map<std::uint64_t, std::uint32_t> block_index_by_id_;
};

// Expects the 27 same-size-level entries of one block to be the blocks the lookup finds at b + (dx, dy, dz).
void ExpectSameSizeLevelEntriesFound(const TableVectors& tables, const BlockLookup& lookup, std::uint32_t block) {
  const CellCoord block_coord = ToBlockCoord(tables.block_ids.at(block));
  const int size_level = ToSizeLevelNumber(tables.block_ids.at(block));
  std::uint32_t entry = 0;
  for (std::int32_t dz = -1; dz <= 1; ++dz) {
    for (std::int32_t dy = -1; dy <= 1; ++dy) {
      for (std::int32_t dx = -1; dx <= 1; ++dx) {
        const CellCoord neighbour_coord = {.x = block_coord.x + dx, .y = block_coord.y + dy, .z = block_coord.z + dz};
        EXPECT_EQ(
            tables.same_size_level_blocks.at((block * kSameSizeLevelEntriesPerBlock) + entry),
            lookup.FindBlock(size_level, neighbour_coord)
        ) << block
          << " " << entry;
        ++entry;
      }
    }
  }
}

// Expects the bigger block of one block to be the lookup's at floor(b / 2), size level l - 1, and none at size
// level 0; and its smaller block of eighth e the lookup's at 2 b + (e % 2, (e / 2) % 2, e / 4), size level l + 1.
void ExpectBiggerAndSmallerEntriesFound(const TableVectors& tables, const BlockLookup& lookup, std::uint32_t block) {
  const CellCoord block_coord = ToBlockCoord(tables.block_ids.at(block));
  const int size_level = ToSizeLevelNumber(tables.block_ids.at(block));
  std::uint32_t expected_bigger = kNoIndex;
  if (size_level >= 1) {
    const CellCoord bigger_coord = {
        .x = ToFloorOfHalving(block_coord.x, 1),
        .y = ToFloorOfHalving(block_coord.y, 1),
        .z = ToFloorOfHalving(block_coord.z, 1),
    };
    expected_bigger = lookup.FindBlock(size_level - 1, bigger_coord);
  }
  EXPECT_EQ(tables.bigger_block.at(block), expected_bigger) << block;
  for (std::int32_t cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    const CellCoord smaller_coord = {
        .x = (2 * block_coord.x) + (cell_eighth % 2),
        .y = (2 * block_coord.y) + ((cell_eighth / 2) % 2),
        .z = (2 * block_coord.z) + (cell_eighth / 4),
    };
    EXPECT_EQ(
        tables.smaller_blocks.at((block * kSmallerBlocksPerBlock) + static_cast<std::uint32_t>(cell_eighth)),
        lookup.FindBlock(size_level + 1, smaller_coord)
    ) << block
      << " " << cell_eighth;
  }
}

TEST(FindBlockNeighbours, RandomGrids_MatchADirectLookup) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    SCOPED_TRACE(seed);
    const TableVectors& tables = BuildRandomGridTables().at(seed - 1);
    const BlockLookup lookup(tables.block_ids);
    for (std::uint32_t block = 0; block < tables.block_ids.size(); ++block) {
      ExpectSameSizeLevelEntriesFound(tables, lookup, block);
      ExpectBiggerAndSmallerEntriesFound(tables, lookup, block);
    }
  }
}

TEST(ComputeCoveredCellState, RandomGrids_MatchActiveCellsOneSizeLevelBigger) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<std::uint64_t> active_list = ToValues(BuildSortedRandomGrid(seed));
    const std::set<std::uint64_t> active_values(active_list.begin(), active_list.end());
    const std::set<std::uint64_t> split_cells = BuildSplitCells(BuildSortedRandomGrid(seed));
    const TableVectors& tables = BuildRandomGridTables().at(seed - 1);
    for (std::uint32_t place = 0; place < tables.cell_states.size(); ++place) {
      const CellBlockId block = tables.block_ids.at(place / 64);
      const int size_level = ToSizeLevelNumber(block);
      const CellCoord block_coord = ToBlockCoord(block);
      const auto place_in_block = static_cast<std::int32_t>(place % 64);
      const CellId cell = ToCell(
          size_level,
          (4 * block_coord.x) + (place_in_block % 4),
          (4 * block_coord.y) + ((place_in_block / 4) % 4),
          (4 * block_coord.z) + (place_in_block / 16)
      );
      CellState expected_state = CellState::kAbsent;
      if (active_values.contains(cell.value)) {
        expected_state = CellState::kActive;
      } else if (split_cells.contains(cell.value)) {
        expected_state = CellState::kSplit;
      } else if (size_level >= 1 && active_values.contains(ToBigCellByDivision(cell, size_level - 1).value)) {
        expected_state = CellState::kInsideBiggerCell;
      }
      EXPECT_EQ(tables.cell_states.at(place), expected_state) << seed << " " << place;
    }
  }
}

TEST(ComputeFacePieceKind, RandomGrids_GiveTheCheckersFacePieces) {
  // 40 of the 200 grids: the checker takes most of this file's time, and the grid's build repeats the comparison on
  // the built grid for all 200.
  for (std::uint64_t seed = 1; seed <= 40; ++seed) {
    const TableVectors& tables = BuildRandomGridTables().at(seed - 1);
    EXPECT_EQ(BuildFacePiecesOfTables(tables), BuildFacePiecesOfChecker(ToUnitFrame(4), BuildSortedRandomGrid(seed)))
        << seed;
  }
}

// A list moved by `shift` size-level-0 cells on every axis: a cell of size level l moves by shift 2^l of its own.
std::vector<CellId> ToShiftedCells(const std::vector<CellId>& cells, std::int32_t shift) {
  std::vector<CellId> shifted_cells;
  shifted_cells.reserve(cells.size());
  for (const CellId cell : cells) {
    const int size_level = static_cast<int>(ToSizeLevel(cell));
    const std::int32_t cell_shift = shift * ComputePowerOfTwo(size_level);
    const CellCoord coord = ToCellCoord(cell);
    shifted_cells.push_back(ToCell(size_level, coord.x + cell_shift, coord.y + cell_shift, coord.z + cell_shift));
  }
  return ToSortedCells(shifted_cells);
}

TEST(ComputeFacePieceKind, RandomGridsAcrossZero_GiveTheCheckersFacePieces) {
  // Moved by -2 size-level-0 cells, the box 0..3 becomes -2..1 on every axis: blocks of negative and positive
  // coordinates, with the block side at 0 inside the box at every size level.
  for (std::uint64_t seed = 1; seed <= 20; ++seed) {
    const std::vector<CellId> cells = ToShiftedCells(BuildSortedRandomGrid(seed), -2);
    const TableVectors tables = BuildTablesOfCells(ToUnitFrame(4), cells);
    EXPECT_EQ(BuildFacePiecesOfTables(tables), BuildFacePiecesOfChecker(ToUnitFrame(4), cells)) << seed;
  }
}

TEST(ComputeFacePieceKind, ThreeSizeLevelsMeeting_GiveTheCheckersFacePieces) {
  // Size level 0 {0,0,0} active; {1,0,0} split, and its size-level-1 eighth 1, {3,0,0}, split again into size level 2
  // x 6..7, y 0..1, z 0..1. Size level 2 meets size level 1, which meets size level 0.
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    const CellId small_cell = ComputeSmallCell(ToCell(0, 1, 0, 0), cell_eighth);
    if (cell_eighth == 1) {
      AppendSmallCells(small_cell, cells);
    } else {
      cells.push_back(small_cell);
    }
  }
  cells = ToSortedCells(cells);
  const TableVectors tables = BuildTablesOfCells(ToUnitFrame(3), cells);
  const std::multiset<FacePieceWithKind> pieces = BuildFacePiecesOfTables(tables);

  EXPECT_EQ(pieces, BuildFacePiecesOfChecker(ToUnitFrame(3), cells));
  // The low X side of size level 2 {6,0,0} lies on size level 1 {2,0,0}, active: kBetweenCells at size level 2. The
  // place x = 5 lies in the size-level-2 block (1,0,0), local 1, covered by the size-level-1 cell 2.
  const FacePieceId small_low_side = ToFacePieceId(Axis::kX, SizeLevel{2}, {.x = 6, .y = 0, .z = 0});
  EXPECT_TRUE(pieces.contains({small_low_side.value, FacePieceKind::kBetweenCells}));
  // Its high X side, x = 8 at size level 2, lies beyond every block of size level 2 and 1: the grid's edge.
  const FacePieceId small_high_side = ToFacePieceId(Axis::kX, SizeLevel{2}, {.x = 8, .y = 0, .z = 0});
  EXPECT_TRUE(pieces.contains({small_high_side.value, FacePieceKind::kGridEdgeHigh}));
}

TEST(ComputeFacePieceKind, SizeLevelZeroCellsAtCoordLimits_GiveTheCheckersFacePieces) {
  std::vector<CellId> cells = {ToCell(0, -kMaxCoord, 0, 0), ToCell(0, kMaxCoord, 0, 0)};
  AppendSmallCells(ToCell(0, 0, 0, 0), cells);
  cells = ToSortedCells(cells);
  const TableVectors tables = BuildTablesOfCells(ToUnitFrame(2), cells);
  const std::multiset<FacePieceWithKind> pieces = BuildFacePiecesOfTables(tables);

  EXPECT_EQ(pieces, BuildFacePiecesOfChecker(ToUnitFrame(2), cells));
  // 6 sides for each lone cell and 36 for the 8 small cells, which touch nothing else.
  EXPECT_EQ(pieces.size(), 48U);
}

TEST(ComputeFacePieceKind, BigCellBeyondHighCoordLimit_IsBetweenCells) {
  // Size level 0 {2^17 - 1, 0, 0} split into size level 1 x = 2^18 - 2 and 2^18 - 1 = kMaxCoord, and size level 0
  // {2^17, 0, 0} active beside it. The high X side of size level 1 x = kMaxCoord is the place x = 2^18, outside the
  // range at size level 1 and inside size level 0 x = 2^17: a face piece between cells, at size level 1.
  std::vector<CellId> cells = {ToCell(0, kTwoToThe17, 0, 0)};
  AppendSmallCells(ToCell(0, kTwoToThe17 - 1, 0, 0), cells);
  cells = ToSortedCells(cells);
  const TableVectors tables = BuildTablesOfCells(ToUnitFrame(2), cells);
  const std::multiset<FacePieceWithKind> pieces = BuildFacePiecesOfTables(tables);

  EXPECT_EQ(pieces, BuildFacePiecesOfChecker(ToUnitFrame(2), cells));
  const FacePieceId beyond_limit = ToFacePieceId(Axis::kX, SizeLevel{1}, {.x = kMaxCoord + 1, .y = 0, .z = 0});
  EXPECT_TRUE(pieces.contains({beyond_limit.value, FacePieceKind::kBetweenCells}));
}

TEST(ComputeFacePieceKind, SmallCellsAtLowCoordLimit_GiveTheCheckersFacePieces) {
  // Size level 0 {-2^17, 0, 0} active; {-2^17 + 1, 0, 0} split into size level 1 x = -2^18 + 2 and -2^18 + 3. The big
  // cell covers size level 1 x = -2^18 + 1 = -kMaxCoord and x = -2^18, outside the range; nothing lies beyond it.
  std::vector<CellId> cells = {ToCell(0, -kTwoToThe17, 0, 0)};
  AppendSmallCells(ToCell(0, -kTwoToThe17 + 1, 0, 0), cells);
  cells = ToSortedCells(cells);
  const TableVectors tables = BuildTablesOfCells(ToUnitFrame(2), cells);
  const std::multiset<FacePieceWithKind> pieces = BuildFacePiecesOfTables(tables);

  EXPECT_EQ(pieces, BuildFacePiecesOfChecker(ToUnitFrame(2), cells));
  const FacePieceId big_low_side = ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = -kTwoToThe17, .y = 0, .z = 0});
  EXPECT_TRUE(pieces.contains({big_low_side.value, FacePieceKind::kGridEdgeLow}));
}

// ---------------------------------------------------------------------------------------------------------------------
// The per-item rules of the tables in parallel (<proc-build-grid-in-parallel> P9 to P12)
// ---------------------------------------------------------------------------------------------------------------------

// Block 1 of the two-size blocks writes its 36 entries, bigger block 0 among them, and offers no key; block 0 is of
// size level 0 and offers none either.
TEST(ApplyBlockNeighboursWriteAt, BlockWithItsBiggerBlock_OffersNoKey) {
  TableVectors two_size_tables = BuildEmptyTables(ToUnitFrame(2), {ToBlock(0, 0, 0, 0), ToBlock(1, 0, 0, 0)});
  std::ranges::fill(two_size_tables.bigger_block, 7U);
  for (const std::uint32_t block : {1U, 0U}) {
    const std::uint64_t failure_key = ApplyBlockNeighboursWriteAt(
        two_size_tables.block_ids.data(),
        2,
        CellBlockIndex{block},
        two_size_tables.same_size_level_blocks.data(),
        two_size_tables.bigger_block.data(),
        two_size_tables.smaller_blocks.data()
    );
    EXPECT_EQ(failure_key, kNoFailureKey) << block;
  }
  EXPECT_EQ(two_size_tables.bigger_block, (std::vector<std::uint32_t>{kNoIndex, 0}));
  EXPECT_EQ(two_size_tables.smaller_blocks.at(0), 1U);
  EXPECT_EQ(two_size_tables.same_size_level_blocks.at(kSameSizeLevelEntriesPerBlock + kSelfEntry), 1U);
}

// The size-level-1 block alone writes kNoIndex as its bigger block and offers the key of step 7 at index 0.
TEST(ApplyBlockNeighboursWriteAt, BlockWithoutItsBiggerBlock_OffersStepSeven) {
  TableVectors lone_tables = BuildEmptyTables(ToUnitFrame(2), {ToBlock(1, 0, 0, 0)});
  lone_tables.bigger_block.at(0) = 7;
  const std::uint64_t lone_key = ApplyBlockNeighboursWriteAt(
      lone_tables.block_ids.data(),
      1,
      CellBlockIndex{0},
      lone_tables.same_size_level_blocks.data(),
      lone_tables.bigger_block.data(),
      lone_tables.smaller_blocks.data()
  );
  EXPECT_EQ(lone_key, ToFailureKey(kBlockFaultStep, 0, 0));
  EXPECT_EQ(lone_tables.bigger_block.at(0), kNoIndex);
}

// The sorted entries of the sort by block over the 8 size-level-1 cells of size level 0 {0,0,0}, the split cell it adds
// at place 8 and the id 0 of the tail at place 9: the order a stable order of the block keys gives, the marks and their
// exclusive sums.
struct SplitCellSortedEntries {
  std::vector<CellId> cell_ids;
  std::vector<std::uint32_t> order;
  std::vector<std::uint32_t> sums;
};

SplitCellSortedEntries BuildSplitCellSortedEntries() {
  SplitCellSortedEntries entries;
  AppendSmallCells(ToCell(0, 0, 0, 0), entries.cell_ids);
  entries.cell_ids.push_back(ToCell(0, 0, 0, 0));
  entries.cell_ids.push_back(CellId{0});
  const auto count = static_cast<std::uint32_t>(entries.cell_ids.size());
  for (std::uint32_t place = 0; place < count; ++place) {
    entries.order.push_back(place);
  }
  std::ranges::stable_sort(entries.order, [&entries](std::uint32_t first, std::uint32_t second) {
    return ComputeIsEarlierBlock(entries.cell_ids.at(first), entries.cell_ids.at(second));
  });
  std::uint32_t mark_sum = 0;
  for (std::uint32_t position = 0; position < count; ++position) {
    entries.sums.push_back(mark_sum);
    mark_sum += ComputeBlockStartMark(entries.cell_ids.data(), entries.order.data(), position);
  }
  entries.sums.push_back(mark_sum);
  return entries;
}

// Every entry, run last to first, over places that hold a stale kInsideBiggerCell: the split cell is entry 0, in block
// 0, and writes kSplit at place 0; each active cell writes kActive at its local place in block 1 and gives size level
// 1; the id 0 writes nothing and gives -1, as the split cell does. Every other place keeps the stale state.
TEST(ApplySortedCellStateAt, SortedEntries_MarkEachCellAtItsPlace) {
  const SplitCellSortedEntries entries = BuildSplitCellSortedEntries();
  ASSERT_EQ(entries.sums.back(), 2U);
  std::vector<CellState> cell_states(std::size_t{2} * kCellPlacesPerBlock, CellState::kInsideBiggerCell);
  std::map<int, int> entries_per_result;
  for (std::uint32_t position = 10; position-- > 0;) {
    const int result = ApplySortedCellStateAt(
        cell_states.data(),
        entries.cell_ids.data(),
        entries.order.data(),
        entries.sums.data(),
        8,
        position
    );
    ++entries_per_result[result];
  }
  EXPECT_EQ(entries_per_result, (std::map<int, int>{{-1, 2}, {1, 8}}));
  std::vector<CellState> expected_states(std::size_t{2} * kCellPlacesPerBlock, CellState::kInsideBiggerCell);
  expected_states.at(ToCellPlace(0, 0, 0, 0)) = CellState::kSplit;
  for (int z = 0; z <= 1; ++z) {
    for (int y = 0; y <= 1; ++y) {
      for (int x = 0; x <= 1; ++x) {
        expected_states.at(ToCellPlace(1, x, y, z)) = CellState::kActive;
      }
    }
  }
  EXPECT_TRUE(cell_states == expected_states);
}

// The two-size fixture without its cover: every place's state after the cover, from ComputeCellStateAfterCover, is the
// fixture with its cover. The 8 places of block 1 inside the active big cell give kInsideBiggerCell; the active, split
// and absent places give their own state.
TEST(ComputeCellStateAfterCover, TwoSizeFixture_GivesTheFixtureWithItsCover) {
  const TableVectors tables = BuildTwoSizeFixtureWithoutCover();
  const GridTables grid_tables = tables.ToGridTables();
  std::vector<CellState> states_after_cover;
  states_after_cover.reserve(std::size_t{2} * kCellPlacesPerBlock);
  for (std::uint32_t place = 0; place < 2 * kCellPlacesPerBlock; ++place) {
    states_after_cover.push_back(ComputeCellStateAfterCover(grid_tables, CellPlaceIndex{place}));
  }
  EXPECT_TRUE(states_after_cover == BuildTwoSizeFixture().cell_states);
}

// The two launches of P10 (c) over the fixture without its cover, each run last to first, the answers in words that
// held stale values: the first writes every place's answer and no state; the second copies them, giving the fixture
// with its cover.
TEST(ApplyCoverAnswerAt, TwoLaunches_GiveTheFixtureWithItsCover) {
  TableVectors tables = BuildTwoSizeFixtureWithoutCover();
  const GridTables grid_tables = tables.ToGridTables();
  const std::vector<CellState> states_before = tables.cell_states;
  std::vector<std::uint64_t> cover_answers(std::size_t{2} * kCellPlacesPerBlock, ~std::uint64_t{0});
  for (std::uint32_t place = 2 * kCellPlacesPerBlock; place-- > 0;) {
    ApplyCoverAnswerAt(grid_tables, cover_answers.data(), CellPlaceIndex{place});
  }
  EXPECT_TRUE(tables.cell_states == states_before);
  for (std::uint32_t place = 2 * kCellPlacesPerBlock; place-- > 0;) {
    ApplyCoverAnswerCopyAt(cover_answers.data(), tables.cell_states.data(), CellPlaceIndex{place});
  }
  EXPECT_TRUE(tables.cell_states == BuildTwoSizeFixture().cell_states);
}

// The calls run in order up to the first that does not start, and not one after it.
TEST(ApplyCallsInOrder, CallThatDidNotStart_StopsTheCallsAfterIt) {
  std::vector<int> made_calls;
  const auto make_call = [&made_calls](int call, RunOnCallStatus call_status) {
    return [&made_calls, call, call_status] {
      made_calls.push_back(call);
      return call_status;
    };
  };
  const RunOnCallStatus stopped = ApplyCallsInOrder(
      make_call(1, RunOnCallStatus::kDone),
      make_call(2, RunOnCallStatus::kCallFailed),
      make_call(3, RunOnCallStatus::kDone)
  );
  EXPECT_EQ(stopped, RunOnCallStatus::kCallFailed);
  EXPECT_EQ(made_calls, (std::vector<int>{1, 2}));
  made_calls.clear();
  const RunOnCallStatus done =
      ApplyCallsInOrder(make_call(4, RunOnCallStatus::kDone), make_call(5, RunOnCallStatus::kDone));
  EXPECT_EQ(done, RunOnCallStatus::kDone);
  EXPECT_EQ(made_calls, (std::vector<int>{4, 5}));
}

// The two-size fixture, every face place run last to first: each writes ComputeFacePieceKind and gives 1 for a face
// piece, 41 in all, the count of the grid of size level 0 {0,0,0} beside the 8 small cells of size level 0 {1,0,0}.
TEST(ApplyFacePieceKindWriteAt, TwoSizeFixture_GivesFortyOneFacePieces) {
  TableVectors tables = BuildTwoSizeFixture();
  std::ranges::fill(tables.face_piece_kinds, FacePieceKind::kOutsidePiece);
  const GridTables grid_tables = tables.ToGridTables();
  std::uint32_t face_piece_count = 0;
  for (std::uint32_t face = 2 * kFacePlacesPerBlock; face-- > 0;) {
    face_piece_count += ApplyFacePieceKindWriteAt(grid_tables, tables.face_piece_kinds.data(), FacePlaceIndex{face});
  }
  EXPECT_EQ(face_piece_count, 41U);
  for (std::uint32_t face = 0; face < 2 * kFacePlacesPerBlock; ++face) {
    EXPECT_EQ(tables.face_piece_kinds.at(face), ComputeFacePieceKind(grid_tables, FacePlaceIndex{face})) << face;
  }
}

// Without a failure the record gets error kNone and active_cells, the sum of the counts per size level.
TEST(ApplyBuildStatusResult, NoFailure_SumsTheSizeLevels) {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.active_cells = 99;
  status.active_cells_per_size_level[0] = 1;
  status.active_cells_per_size_level[1] = 8;
  status.active_cells_per_size_level[15] = 3;
  ApplyBuildStatusResult(ToUnitFrame(16), nullptr, 0, nullptr, 0, &status);
  EXPECT_EQ(status.error, GridError::kNone);
  EXPECT_EQ(status.active_cells, 12U);
  EXPECT_EQ(status.cell.value, 0U);
}

// The key of step 7 gives kInternalFault with zero cells, and the counts stay as they were.
TEST(ApplyBuildStatusResult, StepSevenKey_IsInternalFaultWithCountsLeft) {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.failure_key = ToFailureKey(kBlockFaultStep, 0, 0);
  status.active_cells = 99;
  status.active_cells_per_size_level[0] = 1;
  status.cell = ToCell(0, 1, 2, 3);
  ApplyBuildStatusResult(ToUnitFrame(2), nullptr, 0, nullptr, 0, &status);
  EXPECT_EQ(status.error, GridError::kInternalFault);
  EXPECT_EQ(status.cell.value, 0U);
  EXPECT_EQ(status.other_cell.value, 0U);
  EXPECT_EQ(status.active_cells, 99U);
}

// ---------------------------------------------------------------------------------------------------------------------
// The check of a list on the GPU (<proc-build-grid-in-parallel> P1 to P6), replayed on the CPU
// ---------------------------------------------------------------------------------------------------------------------

// The order in which the replay runs the threads of one launch.
enum class LaunchOrder : std::uint8_t {
  kForward,  // Thread block 0, thread 0 first.
  kReverse,  // The last thread of the last thread block first.
};

constexpr std::array<LaunchOrder, 2> kLaunchOrders = {LaunchOrder::kForward, LaunchOrder::kReverse};
constexpr std::array<SiblingCheck, 2> kSiblingChecks = {SiblingCheck::kRequire, SiblingCheck::kSkip};

// The room of the CPU buffers the replay tests check lists in: 262,144 cell places, more than any list here.
constexpr std::uint32_t kReplayRoom = 4096;

// How the replay runs the threads of every launch.
struct ReplayLaunch {
  LaunchOrder order = LaunchOrder::kForward;
  // Threads of a launch. 0 is the GPU's own: the thread blocks of ComputeLaunchSize(item count) times their size, one
  // item per thread for any list here. A smaller number gives each thread several items, as the GPU does for a list of
  // more than 65,535 x 256 cells.
  std::size_t thread_count = 0;
};

// Runs apply_item(index) for every item below item_count of one launch sized for launch_item_count items, one thread
// after the other in launch.order, each thread over its items as the grid-stride loop of a GPU function walks them:
// index = thread, thread + thread count, and so on, where thread = thread block x thread block size + thread in the
// thread block. apply_item returns false where the GPU function returns, which ends that thread's loop. A launch sized
// for 0 items does not start. The GPU functions of P9 to P11 are sized for the room and act on the items below a count
// they read from the status record, so their two counts differ.
//
// clang-tidy: two counts of items; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <typename ApplyItem>
void ApplyEveryThreadOfSizedLaunch(
    std::size_t launch_item_count,
    std::size_t item_count,
    ReplayLaunch launch,
    ApplyItem apply_item
) {
  if (launch_item_count == 0) {
    return;
  }
  const LaunchSize launch_size = ComputeLaunchSize(launch_item_count);
  const std::size_t launch_thread_count =
      static_cast<std::size_t>(launch_size.thread_block_count) * launch_size.thread_block_size;
  const std::size_t thread_count = launch.thread_count == 0 ? launch_thread_count : launch.thread_count;
  for (std::size_t visited = 0; visited < thread_count; ++visited) {
    const std::size_t thread = launch.order == LaunchOrder::kForward ? visited : thread_count - 1 - visited;
    for (std::size_t index = thread; index < item_count; index += thread_count) {
      if (!apply_item(index)) {
        break;
      }
    }
  }
}

// NOLINTEND(bugprone-easily-swappable-parameters)

// Runs apply_item over a launch sized for its items, as ApplyEveryThreadOfSizedLaunch says.
template <typename ApplyItem>
void ApplyEveryThreadOfLaunch(std::size_t item_count, ReplayLaunch launch, ApplyItem apply_item) {
  ApplyEveryThreadOfSizedLaunch(item_count, item_count, launch, apply_item);
}

// The GPU's working arrays of a check or a build, its cell ids and its block ids, built on the CPU by the replay. CPU
// builder buffers hold only the cell ids of the CPU check, and the replay keeps its own, so the CPU check never writes
// what the replay compares.
struct ReplayArrays {
  std::uint32_t max_cell_blocks = 0;  // The room the arrays are sized for.
  Buffer<CellId> cell_ids;
  Buffer<std::uint64_t> sort_keys;
  Buffer<std::uint32_t> order;
  Buffer<std::uint32_t> second_order;
  std::optional<OrderBuffers> order_buffers;
  Buffer<CellBlockId> block_ids;
  Buffer<std::uint32_t> same_size_level_blocks;  // 27 per block.
  // 1 per block, the bigger blocks, then 8 per block, the smaller blocks, from ComputeSmallerBlocksStart of the room.
  Buffer<std::uint32_t> bigger_and_smaller_blocks;
  Buffer<CellState> cell_states;  // 64 per block.
  Buffer<FacePieceKind> face_piece_kinds;  // 240 per block.
  GridBuildStatus status{};
  // The first sort count cell ids just before P7 (d), the tail fill, of the last build replay.
  std::vector<CellId> cells_before_tail_fill;
  int failed_prefix_sum_count = 0;  // Prefix sums of the last build replay that returned an error.
  // Places of P10 (c) and P11 whose block, bigger block or a neighbour of either named a block at or past the block
  // count: the replay counts them and skips the place, where the GPU would read past its tables.
  int stale_table_read_count = 0;
};

// The entries after the room's part of each replay array, filled with a fixed pattern that no step writes: a write
// past an array of the room on the GPU lands there in the replay, and the test sees it without a sanitizer.
constexpr std::size_t kGuardCount = 16;

// Builds the working arrays for a room of max_cell_blocks blocks, as BuildGridBuilderBuffers and BuildGridBuffers size
// them on the GPU, each followed by kGuardCount guard entries: 64 cell ids, sort keys and order entries per block, plus
// 1 order entry for the total of a prefix sum, and the five tables of the room. The guards hold their pattern
// (ApplyGuardValues).
ReplayArrays BuildReplayArrays(std::uint32_t max_cell_blocks) {
  const std::size_t place_count = std::size_t{kCellPlacesPerBlock} * max_cell_blocks;
  ReplayArrays arrays;
  arrays.max_cell_blocks = max_cell_blocks;
  std::expected<Buffer<CellId>, BufferError> cell_ids = BuildBuffer<CellId>(RunOn::kCpu, place_count + kGuardCount);
  std::expected<Buffer<std::uint64_t>, BufferError> sort_keys =
      BuildBuffer<std::uint64_t>(RunOn::kCpu, place_count + kGuardCount);
  std::expected<Buffer<std::uint32_t>, BufferError> order =
      BuildBuffer<std::uint32_t>(RunOn::kCpu, place_count + 1 + kGuardCount);
  std::expected<Buffer<std::uint32_t>, BufferError> second_order =
      BuildBuffer<std::uint32_t>(RunOn::kCpu, place_count + 1 + kGuardCount);
  std::expected<OrderBuffers, BufferError> order_buffers = BuildOrderBuffers(RunOn::kCpu, place_count + 1);
  std::expected<Buffer<CellBlockId>, BufferError> block_ids =
      BuildBuffer<CellBlockId>(RunOn::kCpu, max_cell_blocks + kGuardCount);
  const std::size_t room = max_cell_blocks;
  std::expected<Buffer<std::uint32_t>, BufferError> same_size_level_blocks =
      BuildBuffer<std::uint32_t>(RunOn::kCpu, (kSameSizeLevelEntriesPerBlock * room) + kGuardCount);
  std::expected<Buffer<std::uint32_t>, BufferError> bigger_and_smaller_blocks =
      BuildBuffer<std::uint32_t>(RunOn::kCpu, ((1 + kSmallerBlocksPerBlock) * room) + kGuardCount);
  std::expected<Buffer<CellState>, BufferError> cell_states =
      BuildBuffer<CellState>(RunOn::kCpu, place_count + kGuardCount);
  std::expected<Buffer<FacePieceKind>, BufferError> face_piece_kinds =
      BuildBuffer<FacePieceKind>(RunOn::kCpu, (std::size_t{kFacePlacesPerBlock} * room) + kGuardCount);
  const bool is_built = cell_ids.has_value() && sort_keys.has_value() && order.has_value() && second_order.has_value()
      && order_buffers.has_value() && block_ids.has_value() && same_size_level_blocks.has_value()
      && bigger_and_smaller_blocks.has_value() && cell_states.has_value() && face_piece_kinds.has_value();
  EXPECT_TRUE(is_built);
  if (is_built) {
    arrays.same_size_level_blocks = std::move(*same_size_level_blocks);
    arrays.bigger_and_smaller_blocks = std::move(*bigger_and_smaller_blocks);
    arrays.cell_states = std::move(*cell_states);
    arrays.face_piece_kinds = std::move(*face_piece_kinds);
    arrays.cell_ids = std::move(*cell_ids);
    arrays.sort_keys = std::move(*sort_keys);
    arrays.order = std::move(*order);
    arrays.second_order = std::move(*second_order);
    arrays.order_buffers = std::move(*order_buffers);
    arrays.block_ids = std::move(*block_ids);
  }
  return arrays;
}

ReplayArrays BuildReplayArrays() {
  return BuildReplayArrays(kReplayRoom);
}

// The guard patterns: values no step writes and no list of these tests holds.
CellId ToGuardCellId() {
  return ToCell(4, 2000, 2000, 2000);
}
constexpr std::uint64_t kGuardSortKey = 0x5A5A5A5A5A5A5A5AU;
constexpr std::uint32_t kGuardOrderEntry = 0xA5A5A5A5U;  // Also the guard of the two neighbour tables.
constexpr auto kGuardCellState = static_cast<CellState>(0xA5U);  // No value of the enum.
constexpr auto kGuardFacePieceKind = static_cast<FacePieceKind>(0x5AU);  // No value of the enum.

// The guard entries of one array: the last kGuardCount.
template <typename Value>
std::span<Value> ToGuardEntries(Buffer<Value>& buffer) {
  return std::span<Value>(buffer.data(), buffer.size()).last(kGuardCount);
}

template <typename Value>
std::span<const Value> ToGuardEntries(const Buffer<Value>& buffer) {
  return std::span<const Value>(buffer.data(), buffer.size()).last(kGuardCount);
}

// Writes the guard patterns after the room's part of every array.
void ApplyGuardValues(ReplayArrays& arrays) {
  std::ranges::fill(ToGuardEntries(arrays.cell_ids), ToGuardCellId());
  std::ranges::fill(ToGuardEntries(arrays.sort_keys), kGuardSortKey);
  std::ranges::fill(ToGuardEntries(arrays.order), kGuardOrderEntry);
  std::ranges::fill(ToGuardEntries(arrays.second_order), kGuardOrderEntry);
  std::ranges::fill(ToGuardEntries(arrays.block_ids), ToCellBlockId(ToGuardCellId()));
  std::ranges::fill(ToGuardEntries(arrays.same_size_level_blocks), kGuardOrderEntry);
  std::ranges::fill(ToGuardEntries(arrays.bigger_and_smaller_blocks), kGuardOrderEntry);
  std::ranges::fill(ToGuardEntries(arrays.cell_states), kGuardCellState);
  std::ranges::fill(ToGuardEntries(arrays.face_piece_kinds), kGuardFacePieceKind);
}

// Says whether every guard entry of one array holds `guard`.
template <typename Value>
bool ComputeAreGuardEntries(const Buffer<Value>& buffer, Value guard) {
  return std::ranges::all_of(ToGuardEntries(buffer), [guard](Value entry) {
    return entry == guard;
  });
}

// Says whether every guard entry still holds its pattern: no write left the room's part of an array.
bool ComputeAreGuardsUnchanged(const ReplayArrays& arrays) {
  const CellId guard_cell = ToGuardCellId();
  const CellBlockId guard_block = ToCellBlockId(guard_cell);
  return std::ranges::all_of(
             ToGuardEntries(arrays.cell_ids),
             [guard_cell](CellId cell) {
               return cell.value == guard_cell.value;
             }
         )
      && std::ranges::all_of(
             ToGuardEntries(arrays.sort_keys),
             [](std::uint64_t key) {
               return key == kGuardSortKey;
             }
      )
      && std::ranges::all_of(
             ToGuardEntries(arrays.order),
             [](std::uint32_t entry) {
               return entry == kGuardOrderEntry;
             }
      )
      && std::ranges::all_of(
             ToGuardEntries(arrays.second_order),
             [](std::uint32_t entry) {
               return entry == kGuardOrderEntry;
             }
      )
      && std::ranges::all_of(
             ToGuardEntries(arrays.block_ids),
             [guard_block](CellBlockId block) {
               return block.value == guard_block.value;
             }
      )
      && ComputeAreGuardEntries(arrays.same_size_level_blocks, kGuardOrderEntry)
      && ComputeAreGuardEntries(arrays.bigger_and_smaller_blocks, kGuardOrderEntry)
      && ComputeAreGuardEntries(arrays.cell_states, kGuardCellState)
      && ComputeAreGuardEntries(arrays.face_piece_kinds, kGuardFacePieceKind);
}

// A cell id that no list of these tests holds, written where a result is expected, so a result never written shows.
CellId ToStaleCellId() {
  return ToCell(3, 1000, 1000, 1000);
}

// Writes large values over every working array, the cell ids and the tables, and a status record that already holds
// the smallest key there is, as an earlier check could leave them. A replay that read any of them before writing it
// would differ from the CPU: the stale key would win the minimum, a stale sort key or order would misplace a cell, a
// stale cell id would stay in the sorted list or become a block, a stale neighbour entry, kSplit state or
// kOutsidePiece kind would stay in a table, kSplit even stopping P10 (c) at that place. The guards keep their patterns.
void ApplyStaleValues(ReplayArrays& arrays) {
  std::fill_n(arrays.cell_ids.data(), arrays.cell_ids.size(), ToStaleCellId());
  std::fill_n(arrays.sort_keys.data(), arrays.sort_keys.size(), ~std::uint64_t{0} - 1);
  std::fill_n(arrays.order.data(), arrays.order.size(), kNoIndex - 1);
  std::fill_n(arrays.second_order.data(), arrays.second_order.size(), kNoIndex - 1);
  std::fill_n(arrays.block_ids.data(), arrays.block_ids.size(), ToCellBlockId(ToStaleCellId()));
  std::fill_n(arrays.same_size_level_blocks.data(), arrays.same_size_level_blocks.size(), kNoIndex - 1);
  std::fill_n(arrays.bigger_and_smaller_blocks.data(), arrays.bigger_and_smaller_blocks.size(), kNoIndex - 1);
  std::fill_n(arrays.cell_states.data(), arrays.cell_states.size(), CellState::kSplit);
  std::fill_n(arrays.face_piece_kinds.data(), arrays.face_piece_kinds.size(), FacePieceKind::kOutsidePiece);
  ApplyGuardValues(arrays);
  arrays.status = ComputeResetBuildStatus();
  arrays.status.failure_key = ToFailureKey(kIdStep, 0, 0);
  arrays.status.error = GridError::kBadInput;
  arrays.status.cell = ToCell(0, 1, 2, 3);
}

// Writes 0xFFFFFFFF words over the sort keys, both orders and the whole status record, its counts included, and the
// stale id over the cell ids: a prefix sum that read two such words would pass 2^32 - 1 and fail. The guards keep their
// patterns.
void ApplyStaleAllOnesWords(ReplayArrays& arrays) {
  ApplyStaleValues(arrays);
  std::fill_n(arrays.sort_keys.data(), arrays.sort_keys.size(), ~std::uint64_t{0});
  std::fill_n(arrays.order.data(), arrays.order.size(), ~std::uint32_t{0});
  std::fill_n(arrays.second_order.data(), arrays.second_order.size(), ~std::uint32_t{0});
  ApplyGuardValues(arrays);
  arrays.status.cell_blocks = ~std::uint32_t{0};
  arrays.status.split_cells = ~std::uint32_t{0};
  arrays.status.result_count = ~std::uint32_t{0};
  arrays.status.active_cells = ~std::uint64_t{0};
  std::ranges::fill(arrays.status.active_cells_per_size_level, ~std::uint64_t{0});
  arrays.status.face_pieces = ~std::uint64_t{0};
}

// Replays <proc-build-grid-in-parallel> P1 to P6 on the CPU, with the per-item rules the GPU functions of
// grid/staggered_grid_gpu.cu call, in the same sequence as BuildSortedCellList's GPU path: every thread of every launch
// as `launch` says, the atomic minimum replaced by a plain minimum, BuildStableOrder on the CPU between the passes. The
// sorted list goes to arrays.cell_ids. Returns false, after a test failure, when BuildStableOrder fails.
bool ApplyListCheckStepsByReplay(
    GridFrame frame,
    std::span<const CellId> input_cells,
    SiblingCheck sibling_check,
    ReplayLaunch launch,
    ReplayArrays& arrays
) {
  if (!arrays.order_buffers.has_value()) {
    ADD_FAILURE() << "the replay has no order buffers";
    return false;
  }
  OrderBuffers& order_buffers = *arrays.order_buffers;
  const auto count = static_cast<std::uint32_t>(input_cells.size());
  const CellId* const input = input_cells.data();
  CellId* const sorted_cells = arrays.cell_ids.data();
  GridBuildStatus& status = arrays.status;
  const auto offer_key = [&status](std::uint64_t failure_key) {
    status.failure_key = std::min(status.failure_key, failure_key);
  };
  // P1: the reset, one thread.
  status = ComputeResetBuildStatus();
  // P2: the ids, one thread per input index.
  ApplyEveryThreadOfLaunch(count, launch, [&](std::size_t index) {
    offer_key(ComputeCellIdFailureKey(frame, input, index));
    return true;
  });
  // P3: three passes, alternating the two orders as the GPU path does, then the copy in sorted order.
  const std::array<const Buffer<std::uint32_t>*, kSortKeyPassCount> previous_orders = {
      nullptr,
      &arrays.order,
      &arrays.second_order,
  };
  const std::array<Buffer<std::uint32_t>*, kSortKeyPassCount> pass_orders = {
      &arrays.order,
      &arrays.second_order,
      &arrays.order,
  };
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    ApplyEveryThreadOfLaunch(count, launch, [&](std::size_t index) {
      arrays.sort_keys.data()[index] = ToCellSortKey(input[index], pass);
      return true;
    });
    const auto pass_index = static_cast<std::size_t>(pass);
    const std::expected<void, BufferError> ordered = BuildStableOrder(
        arrays.sort_keys,
        count,
        previous_orders.at(pass_index),
        *pass_orders.at(pass_index),
        order_buffers
    );
    if (!ordered.has_value()) {
      ADD_FAILURE() << "BuildStableOrder failed";
      return false;
    }
  }
  ApplyEveryThreadOfLaunch(count, launch, [&](std::size_t position) {
    sorted_cells[position] = input[arrays.order.data()[position]];
    return true;
  });
  // P4, P5 when siblings are required, P6: a thread returns once an earlier step's key is there.
  for (const int step : {kOverlapStep, kSiblingStep, kSizeStepRuleStep}) {
    if (step == kSiblingStep && sibling_check == SiblingCheck::kSkip) {
      continue;
    }
    ApplyEveryThreadOfLaunch(count, launch, [&](std::size_t position) {
      if (ComputeHasEarlierStepFailed(status.failure_key, step)) {
        return false;
      }
      offer_key(ComputeSortedCellFailureKey(sorted_cells, count, static_cast<std::uint32_t>(position), step));
      return true;
    });
  }
  return true;
}

// Replays P1 to P6 and the failure half of P12, the GPU path of BuildSortedCellList, as ApplyListCheckStepsByReplay
// says. Returns ToGridFailure of the status record it wrote.
GridFailure ApplyGpuStepsByReplay(
    GridFrame frame,
    std::span<const CellId> input_cells,
    SiblingCheck sibling_check,
    ReplayLaunch launch,
    ReplayArrays& arrays
) {
  if (!ApplyListCheckStepsByReplay(frame, input_cells, sibling_check, launch, arrays)) {
    return {.error = GridError::kInternalFault};
  }
  // P12, the failure half, one thread; then the CPU half's reading of the record.
  const auto count = static_cast<std::uint32_t>(input_cells.size());
  ApplyFailureCells(frame, input_cells.data(), count, arrays.cell_ids.data(), count, &arrays.status);
  return ToGridFailure(arrays.status);
}

// How a build replay differs from the GPU's sequence: a copy with one part changed, so a test shows the part matters.
enum class ReplayChange : std::uint8_t {
  kNone,  // The GPU's sequence.
  kWithoutTailFill,  // P7 (d), the tail fill, left out.
  kSplitCountsReturnEarly,  // A thread of P7 (a) returns after an earlier step's failure instead of writing 0.
  // P7 (b) checks the cells against one place less than they need in place of the sort count, the state the n / 7
  // bound rules out.
  kSortCountBelowCells,
};

// Runs ComputeExclusivePrefixSum in place over the first `count` words of `sums`, as the GPU path does between the
// launches of P7 and P8. An error is counted in arrays.failed_prefix_sum_count and returns false: the CPU half maps a
// size error of base/ to kInternalFault.
bool ApplyPrefixSumOfReplay(std::uint32_t count, ReplayArrays& arrays) {
  if (!arrays.order_buffers.has_value()) {
    return false;
  }
  const std::expected<void, BufferError> summed =
      ComputeExclusivePrefixSum(arrays.second_order, count, arrays.second_order, *arrays.order_buffers);
  if (!summed.has_value()) {
    ++arrays.failed_prefix_sum_count;
    return false;
  }
  return true;
}

// The key of P7 (b) in the replay: ComputeCellPlacesFailureKey against the room's cell places and the sort count, or,
// for ReplayChange::kSortCountBelowCells, against one place less than the cells need in place of the sort count.
//
// clang-tidy: three counts; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::uint64_t ComputeCellPlacesKeyOfReplay(
    std::uint32_t active_count,
    std::uint32_t split_cells,
    std::uint32_t max_cell_blocks,
    ReplayChange change
) {
  const std::uint64_t room_cell_places = std::uint64_t{kCellPlacesPerBlock} * max_cell_blocks;
  const std::uint64_t cell_count = std::uint64_t{active_count} + split_cells;
  const bool is_bound_lowered = change == ReplayChange::kSortCountBelowCells && cell_count > 0;
  const std::uint32_t sort_count = ComputeSortCount(active_count, max_cell_blocks);
  const auto checked_sort_count = static_cast<std::uint32_t>(is_bound_lowered ? cell_count - 1 : sort_count);
  return ComputeCellPlacesFailureKey(cell_count, room_cell_places, checked_sort_count);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Replays P7 and P8 on the sorted list of `active_count` cells in arrays.cell_ids, for a room of max_cell_blocks, with
// the per-item rules the GPU functions call, every thread of every launch as `launch` says, each prefix sum and stable
// order on the CPU between the launches. Returns false when a prefix sum or a stable order fails.
//
// clang-tidy: a count and a room; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool ApplyBlockStepsByReplay(
    std::uint32_t active_count,
    std::uint32_t max_cell_blocks,
    ReplayLaunch launch,
    ReplayChange change,
    ReplayArrays& arrays
) {
  if (!arrays.order_buffers.has_value()) {
    ADD_FAILURE() << "the replay has no order buffers";
    return false;
  }
  OrderBuffers& order_buffers = *arrays.order_buffers;
  CellId* const cell_ids = arrays.cell_ids.data();
  std::uint32_t* const sums = arrays.second_order.data();
  const std::uint32_t* const order = arrays.order.data();
  GridBuildStatus& status = arrays.status;
  const auto offer_key = [&status](std::uint64_t failure_key) {
    status.failure_key = std::min(status.failure_key, failure_key);
  };
  const auto has_earlier_failure = [&status](int step) {
    return ComputeHasEarlierStepFailed(status.failure_key, step);
  };
  // Step P7 (a): the count of each active cell, or 0 after an earlier failure, then the prefix sum in place.
  ApplyEveryThreadOfLaunch(active_count, launch, [&](std::size_t position) {
    if (has_earlier_failure(kCellPlacesStep)) {
      if (change == ReplayChange::kSplitCountsReturnEarly) {
        return false;
      }
      sums[position] = 0;
      return true;
    }
    sums[position] = static_cast<std::uint32_t>(ComputeSplitCellCountToAdd(cell_ids[position]));
    return true;
  });
  if (!ApplyPrefixSumOfReplay(active_count, arrays)) {
    return false;
  }
  // Step P7 (b), one thread: the cells against the room's cell places, then against the sort count.
  const std::uint32_t sort_count = ComputeSortCount(active_count, max_cell_blocks);
  status.split_cells = sums[active_count];
  offer_key(ComputeCellPlacesKeyOfReplay(active_count, status.split_cells, max_cell_blocks, change));
  // Step P7 (c).
  ApplyEveryThreadOfLaunch(active_count, launch, [&](std::size_t position) {
    if (has_earlier_failure(kBlockCountStep)) {
      return false;
    }
    ApplySplitCellWritesAt(cell_ids, sums, active_count, static_cast<std::uint32_t>(position));
    return true;
  });
  // Step P7 (d): the tail fill, over the CPU's sort count, with the used count from the status record.
  arrays.cells_before_tail_fill.assign(cell_ids, cell_ids + sort_count);
  if (change != ReplayChange::kWithoutTailFill) {
    ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t place) {
      if (has_earlier_failure(kBlockCountStep)) {
        return false;
      }
      ApplySortTailFillAt(cell_ids, active_count + status.split_cells, static_cast<std::uint32_t>(place));
      return true;
    });
  }
  // Step P8: three passes of the block keys, alternating the orders as P3 does, the final order in arrays.order.
  const std::array<const Buffer<std::uint32_t>*, kSortKeyPassCount> previous_orders = {
      nullptr,
      &arrays.order,
      &arrays.second_order,
  };
  const std::array<Buffer<std::uint32_t>*, kSortKeyPassCount> pass_orders = {
      &arrays.order,
      &arrays.second_order,
      &arrays.order,
  };
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t place) {
      if (has_earlier_failure(kBlockCountStep)) {
        return false;
      }
      arrays.sort_keys.data()[place] = ToBlockSortKey(cell_ids[place], pass);
      return true;
    });
    const auto pass_index = static_cast<std::size_t>(pass);
    const std::expected<void, BufferError> ordered = BuildStableOrder(
        arrays.sort_keys,
        sort_count,
        previous_orders.at(pass_index),
        *pass_orders.at(pass_index),
        order_buffers
    );
    if (!ordered.has_value()) {
      ADD_FAILURE() << "BuildStableOrder failed";
      return false;
    }
  }
  // Step P8: the marks, or 0 after an earlier failure, step 5 included, then the prefix sum in place.
  ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t position) {
    sums[position] = has_earlier_failure(kBlockCountStep)
        ? 0U
        : ComputeBlockStartMark(cell_ids, order, static_cast<std::uint32_t>(position));
    return true;
  });
  if (!ApplyPrefixSumOfReplay(sort_count, arrays)) {
    return false;
  }
  // Step P8: the block count against the room, one thread, then the scatter of the block ids.
  status.cell_blocks = sums[sort_count];
  offer_key(ComputeRoomFailureKey(status.cell_blocks, max_cell_blocks, kBlockCountStep));
  ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t position) {
    if (has_earlier_failure(kBlockFaultStep)) {
      return false;
    }
    ApplyBlockIdWriteAt(arrays.block_ids.data(), cell_ids, order, sums, static_cast<std::uint32_t>(position));
    return true;
  });
  return true;
}

// The tables of the replay arrays for the block count of their status record, at most the room, as a GPU function of
// P10 (c) or P11 sees them.
GridTables ToReplayTables(GridFrame frame, GridVersion version, const ReplayArrays& arrays) {
  const std::uint32_t* const bigger_block = arrays.bigger_and_smaller_blocks.data();
  return {
      .frame = frame,
      .version = version,
      .block_count = std::min(arrays.status.cell_blocks, arrays.max_cell_blocks),
      .block_ids = arrays.block_ids.data(),
      .same_size_level_blocks = arrays.same_size_level_blocks.data(),
      .bigger_block = bigger_block,
      .smaller_blocks = bigger_block + ComputeSmallerBlocksStart(arrays.max_cell_blocks),
      .cell_states = arrays.cell_states.data(),
      .face_piece_kinds = arrays.face_piece_kinds.data(),
  };
}

// Says whether an entry is a block of the tables or kNoIndex.
bool ComputeIsBlockEntryInRange(const GridTables& tables, std::uint32_t entry) {
  return entry == kNoIndex || entry < tables.block_count;
}

// Says whether every block entry that a place of `block` may read through P10 (c) or P11 names a block of the tables or
// kNoIndex: the block's bigger block and 27 neighbours, and, when the bigger block is a block, its 27 neighbours.
bool ComputeAreTableEntriesInRange(const GridTables& tables, std::uint32_t block) {
  const auto are_neighbours_in_range = [&tables](std::uint32_t of_block) {
    const std::span<const std::uint32_t> neighbours(
        tables.same_size_level_blocks + (std::size_t{kSameSizeLevelEntriesPerBlock} * of_block),
        kSameSizeLevelEntriesPerBlock
    );
    return std::ranges::all_of(neighbours, [&tables](std::uint32_t entry) {
      return ComputeIsBlockEntryInRange(tables, entry);
    });
  };
  const std::uint32_t bigger_block = tables.bigger_block[block];
  if (!ComputeIsBlockEntryInRange(tables, bigger_block) || !are_neighbours_in_range(block)) {
    return false;
  }
  return bigger_block == kNoIndex || are_neighbours_in_range(bigger_block);
}

// The replay's instrument for stale reads: two blocks with every entry kNoIndex are in range; one stale entry, at 2 or
// more with 2 blocks, makes block 0 out of range in each of the three places it may read: its bigger block, its own
// neighbours with no bigger block, and the neighbours of its bigger block, block 1, while its own are in range.
TEST(ComputeAreTableEntriesInRange, StaleEntryInEachPlace_IsOutOfRange) {
  const TableVectors in_range = BuildEmptyTables(ToUnitFrame(2), {ToBlock(0, 0, 0, 0), ToBlock(1, 0, 0, 0)});
  EXPECT_TRUE(ComputeAreTableEntriesInRange(in_range.ToGridTables(), 0));
  TableVectors stale_bigger = in_range;
  stale_bigger.bigger_block.at(0) = 2;
  EXPECT_FALSE(ComputeAreTableEntriesInRange(stale_bigger.ToGridTables(), 0));
  TableVectors stale_own_neighbour = in_range;
  stale_own_neighbour.same_size_level_blocks.at(0) = kNoIndex - 1;
  EXPECT_FALSE(ComputeAreTableEntriesInRange(stale_own_neighbour.ToGridTables(), 0));
  TableVectors stale_bigger_neighbour = in_range;
  stale_bigger_neighbour.bigger_block.at(0) = 1;
  stale_bigger_neighbour.same_size_level_blocks.at(kSameSizeLevelEntriesPerBlock) = 2;
  EXPECT_FALSE(ComputeAreTableEntriesInRange(stale_bigger_neighbour.ToGridTables(), 0));
  EXPECT_FALSE(ComputeAreTableEntriesInRange(stale_bigger_neighbour.ToGridTables(), 1));
}

// Replays P9 to P11 into the tables of `arrays`, for a room of arrays.max_cell_blocks blocks, after P8 left the block
// ids, the final order in arrays.order and the sums of the marks in arrays.second_order: every thread of every launch
// as `launch` says, each launch sized for the room (or, for P10 (b), the sort count) as the GPU half sizes it, the
// threads acting on the items below the counts of the status record. The atomic minimum and adds become plain ones.
//
// clang-tidy: two counts; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void ApplyTableStepsByReplay(
    GridFrame frame,
    std::uint32_t active_count,
    std::uint32_t sort_count,
    ReplayLaunch launch,
    ReplayArrays& arrays
) {
  GridBuildStatus& status = arrays.status;
  const std::uint32_t room = arrays.max_cell_blocks;
  const GridTables tables = ToReplayTables(frame, GridVersion{1}, arrays);
  const std::uint32_t block_count = tables.block_count;
  const auto has_earlier_failure = [&status](int step) {
    return ComputeHasEarlierStepFailed(status.failure_key, step);
  };
  // P9: one thread per block of the room; a block of size level 1 or more without its bigger block offers step 7.
  std::uint32_t* const same_size_level_blocks = arrays.same_size_level_blocks.data();
  std::uint32_t* const bigger_block = arrays.bigger_and_smaller_blocks.data();
  std::uint32_t* const smaller_blocks = bigger_block + ComputeSmallerBlocksStart(room);
  ApplyEveryThreadOfSizedLaunch(room, block_count, launch, [&](std::size_t block) {
    if (has_earlier_failure(kBlockFaultStep)) {
      return false;
    }
    const std::uint64_t failure_key = ApplyBlockNeighboursWriteAt(
        arrays.block_ids.data(),
        block_count,
        CellBlockIndex{static_cast<std::uint32_t>(block)},
        same_size_level_blocks,
        bigger_block,
        smaller_blocks
    );
    status.failure_key = std::min(status.failure_key, failure_key);
    return true;
  });
  // P10 (a), (b) and (c), then P11: any key stops them.
  const std::size_t room_cell_places = std::size_t{kCellPlacesPerBlock} * room;
  CellState* const cell_states = arrays.cell_states.data();
  const std::size_t used_cell_places = std::size_t{kCellPlacesPerBlock} * block_count;
  ApplyEveryThreadOfSizedLaunch(room_cell_places, used_cell_places, launch, [&](std::size_t place) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    ApplyAbsentStateAt(cell_states, CellPlaceIndex{static_cast<std::uint32_t>(place)});
    return true;
  });
  ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t position) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    const int size_level = ApplySortedCellStateAt(
        cell_states,
        arrays.cell_ids.data(),
        arrays.order.data(),
        arrays.second_order.data(),
        active_count,
        static_cast<std::uint32_t>(position)
    );
    if (size_level >= 0) {
      ++status.active_cells_per_size_level[static_cast<std::size_t>(size_level)];
    }
    return true;
  });
  // Which blocks read only blocks of the tables or kNoIndex, found once after P9: nothing after it changes a block
  // entry.
  std::vector<bool> is_block_in_range(block_count);
  for (std::uint32_t block = 0; block < block_count; ++block) {
    is_block_in_range[block] = ComputeAreTableEntriesInRange(tables, block);
  }
  // P10 (c): the answers into the sort keys, free after P8, then the copy into the states.
  std::uint64_t* const cover_answers = arrays.sort_keys.data();
  ApplyEveryThreadOfSizedLaunch(room_cell_places, used_cell_places, launch, [&](std::size_t place) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    if (!is_block_in_range[place / kCellPlacesPerBlock]) {
      ++arrays.stale_table_read_count;
      return true;
    }
    ApplyCoverAnswerAt(tables, cover_answers, CellPlaceIndex{static_cast<std::uint32_t>(place)});
    return true;
  });
  ApplyEveryThreadOfSizedLaunch(room_cell_places, used_cell_places, launch, [&](std::size_t place) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    ApplyCoverAnswerCopyAt(cover_answers, cell_states, CellPlaceIndex{static_cast<std::uint32_t>(place)});
    return true;
  });
  const std::size_t room_face_places = std::size_t{kFacePlacesPerBlock} * room;
  const std::size_t used_face_places = std::size_t{kFacePlacesPerBlock} * block_count;
  ApplyEveryThreadOfSizedLaunch(room_face_places, used_face_places, launch, [&](std::size_t face) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    if (!is_block_in_range[face / kFacePlacesPerBlock]) {
      ++arrays.stale_table_read_count;
      return true;
    }
    status.face_pieces += ApplyFacePieceKindWriteAt(
        tables,
        arrays.face_piece_kinds.data(),
        FacePlaceIndex{static_cast<std::uint32_t>(face)}
    );
    return true;
  });
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Replays a build on the GPU, <proc-build-grid-in-parallel> P1 to P12: the input checks of P1 with CheckCellListInputs
// on `buffers`, which gives the room, then the steps of ApplyListCheckStepsByReplay with the siblings required,
// ApplyBlockStepsByReplay, ApplyTableStepsByReplay and ApplyBuildStatusResult. A failed prefix sum or stable order
// gives kInternalFault, as the CPU half's mapping of a size error does. Returns the failure the status names.
GridFailure ApplyGpuBuildStepsByReplay(
    const GridInputs& inputs,
    const GridBuffers& buffers,
    ReplayLaunch launch,
    ReplayChange change,
    ReplayArrays& arrays
) {
  arrays.failed_prefix_sum_count = 0;
  arrays.stale_table_read_count = 0;
  arrays.cells_before_tail_fill.clear();
  if (const std::expected<void, GridFailure> inputs_checked = CheckCellListInputs(inputs, buffers);
      !inputs_checked.has_value()) {
    return inputs_checked.error();
  }
  const std::uint32_t max_cell_blocks = buffers.max_cell_blocks();
  // The guards lie just after the room's part of each array only when the arrays are sized for this room.
  if (arrays.max_cell_blocks != max_cell_blocks) {
    ADD_FAILURE() << "the replay arrays are sized for another room";
    return {.error = GridError::kInternalFault};
  }
  const std::span<const CellId> input_cells = inputs.active_cells;
  if (!ApplyListCheckStepsByReplay(inputs.frame, input_cells, SiblingCheck::kRequire, launch, arrays)) {
    return {.error = GridError::kInternalFault};
  }
  const auto count = static_cast<std::uint32_t>(input_cells.size());
  if (!ApplyBlockStepsByReplay(count, max_cell_blocks, launch, change, arrays)) {
    return {.error = GridError::kInternalFault};
  }
  ApplyTableStepsByReplay(inputs.frame, count, ComputeSortCount(count, max_cell_blocks), launch, arrays);
  ApplyBuildStatusResult(inputs.frame, input_cells.data(), count, arrays.cell_ids.data(), count, &arrays.status);
  return ToGridFailure(arrays.status);
}

// Every field of a failure, as one comparable value.
std::tuple<GridError, std::uint64_t, std::uint64_t, std::uint64_t> ToFailureFields(const GridFailure& failure) {
  return {failure.error, failure.cell.value, failure.other_cell.value, failure.magnitude};
}

// What one replay gave: the failure and the first n cell ids.
struct ReplayResult {
  GridFailure failure;
  std::vector<std::uint64_t> sorted_values;
};

// Replays the GPU check of `cells` in both launch orders, each after ApplyStaleValues, and returns the results in the
// order of kLaunchOrders.
std::array<ReplayResult, 2> ComputeReplaysInBothOrders(
    GridFrame frame,
    std::span<const CellId> cells,
    SiblingCheck sibling_check,
    std::size_t thread_count,
    ReplayArrays& arrays
) {
  std::array<ReplayResult, 2> results;
  for (std::size_t order_index = 0; order_index < kLaunchOrders.size(); ++order_index) {
    ApplyStaleValues(arrays);
    const ReplayLaunch launch = {.order = kLaunchOrders.at(order_index), .thread_count = thread_count};
    results.at(order_index).failure = ApplyGpuStepsByReplay(frame, cells, sibling_check, launch, arrays);
    results.at(order_index).sorted_values =
        ToValues(std::vector<CellId>(arrays.cell_ids.data(), arrays.cell_ids.data() + cells.size()));
  }
  return results;
}

// Says whether a failure is one of step 1, the ids: the replay's sorted list is then never used and is not compared.
bool ComputeIsIdFailure(const GridFailure& failure) {
  return failure.error == GridError::kBadInput || failure.error == GridError::kCoordOutOfRange;
}

// Expects each replay to give cpu_failure and, unless the ids failed, expected_sorted. `context` names the list and
// the sibling check.
void ExpectReplaysEqualCpu(
    const std::array<ReplayResult, 2>& replays,
    const GridFailure& cpu_failure,
    const std::vector<std::uint64_t>& expected_sorted,
    const std::string& context
) {
  for (std::size_t order_index = 0; order_index < replays.size(); ++order_index) {
    const std::string order_context = context + (order_index == 0 ? ", forward" : ", reverse");
    EXPECT_EQ(ToFailureFields(replays.at(order_index).failure), ToFailureFields(cpu_failure)) << order_context;
    if (!ComputeIsIdFailure(cpu_failure)) {
      EXPECT_EQ(replays.at(order_index).sorted_values, expected_sorted) << order_context;
    }
  }
}

// Replays the GPU check of `cells` for each of `sibling_checks`, in both launch orders and with `thread_count` threads
// (0 for the GPU's own), on the replay's own cell ids, then runs BuildSortedCellList on the CPU buffers. Expects the
// same failure, and, unless the ids failed, the sorted list of ComesBefore, which is also the CPU check's when it
// passes. `name` says which list failed.
void ExpectReplayEqualsCpu(
    GridFrame frame,
    std::span<const CellId> cells,
    std::span<const SiblingCheck> sibling_checks,
    std::size_t thread_count,
    GridBuffers& buffers,
    ReplayArrays& arrays,
    const std::string& name
) {
  const Buffer<CellId>& cpu_cell_ids =
      ToGridBuilderBuffersArrays(*ToGridBuffersArrays(buffers).builder_buffers).cell_ids;
  const GridInputs inputs = {.frame = frame, .active_cells = cells, .version = {.value = 1}};
  const std::vector<std::uint64_t> expected_sorted = ToValues(ToSortedCells({cells.begin(), cells.end()}));
  for (const SiblingCheck sibling_check : sibling_checks) {
    const std::array<ReplayResult, 2> replays =
        ComputeReplaysInBothOrders(frame, cells, sibling_check, thread_count, arrays);
    const std::expected<std::uint32_t, GridFailure> cpu_result = BuildSortedCellList(inputs, buffers, sibling_check);
    const std::string context = name + (sibling_check == SiblingCheck::kRequire ? ", siblings" : ", no siblings");
    if (cpu_result.has_value()) {
      const std::vector<CellId> cpu_sorted(cpu_cell_ids.data(), cpu_cell_ids.data() + *cpu_result);
      EXPECT_EQ(ToValues(cpu_sorted), expected_sorted) << context;
    }
    ExpectReplaysEqualCpu(
        replays,
        cpu_result.has_value() ? GridFailure{} : cpu_result.error(),
        expected_sorted,
        context
    );
  }
}

// Size level 0 {0,0,0} and a size-level-2 cell beyond its high side that touches it at a corner, along an edge and on
// a side: the three lists two size levels apart of the CPU check's tests.
std::array<std::vector<CellId>, 3> BuildEveryTwoSizeLevelsApartList() {
  return {
      std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(2, 4, 4, 4)},
      std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(2, 4, 4, 0)},
      std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(2, 4, 0, 0)},
  };
}

// A CPU grid of no cells in a room of max_cell_blocks blocks, whose buffers TakeEmptyGridBuffers hands to a test.
std::expected<TestGrid, GridFailure> BuildReplayBuffersOwner(std::uint32_t max_cell_blocks) {
  const GridInputs no_cells = {.frame = ToUnitFrame(4), .active_cells = {}, .version = {.value = 1}};
  return BuildTestGrid(no_cells, {.max_cell_blocks = max_cell_blocks});
}

std::expected<TestGrid, GridFailure> BuildReplayBuffersOwner() {
  return BuildReplayBuffersOwner(kReplayRoom);
}

// The mutated lists of the replay: seeds 1..kReplayedMutatedGridCount. They hold every combination of earliest failing
// step, sibling check and one or several failing items in that step that seeds 1..500 hold, so the other 400 would
// add about 5 s to a debug run and no new case; the gpu tests check all 500 on hardware.
constexpr std::uint64_t kReplayedMutatedGridCount = 100;

// Adds to errors_seen the error the CPU check gives `cells` with each sibling check, kNone when it passes.
void ApplyCpuErrors(
    GridFrame frame,
    std::span<const CellId> cells,
    GridBuffers& buffers,
    std::set<GridError>& errors_seen
) {
  const GridInputs inputs = {.frame = frame, .active_cells = cells, .version = {.value = 1}};
  for (const SiblingCheck sibling_check : kSiblingChecks) {
    const std::expected<std::uint32_t, GridFailure> cpu_result = BuildSortedCellList(inputs, buffers, sibling_check);
    errors_seen.insert(cpu_result.has_value() ? GridError::kNone : cpu_result.error().error);
  }
}

// The random grids are valid: no item offers a key, so they show the sorted list and that no per-item rule fails, with
// siblings required, which runs every step that skipping them runs, and P5 too. The lists that fail, two size levels
// apart, BuildSizeStepBreakWithSiblings (siblings pass, then the size step rule fails) and mutated, show the first
// failure, with and without siblings, where the order of the threads could matter.
TEST(BuildSortedCellList, EveryThreadBlockAndThread_SameFirstFailureAsCpu) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  constexpr std::array<SiblingCheck, 1> kSiblingsRequired = {SiblingCheck::kRequire};
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    const std::vector<CellId>& cells = BuildShuffledRandomGrids().at(grid);
    ExpectReplayEqualsCpu(
        ToUnitFrame(4),
        cells,
        kSiblingsRequired,
        0,
        buffers,
        arrays,
        "random grid " + std::to_string(grid)
    );
  }
  for (const std::vector<CellId>& cells : BuildEveryTwoSizeLevelsApartList()) {
    ExpectReplayEqualsCpu(ToUnitFrame(3), cells, kSiblingChecks, 0, buffers, arrays, "two size levels apart");
  }
  ExpectReplayEqualsCpu(
      ToUnitFrame(3),
      BuildSizeStepBreakWithSiblings(),
      kSiblingChecks,
      0,
      buffers,
      arrays,
      "size step break with siblings"
  );
  std::set<GridError> errors_seen;
  for (std::uint64_t seed = 1; seed <= kReplayedMutatedGridCount; ++seed) {
    const std::vector<CellId> cells = BuildMutatedGridCells(seed);
    ExpectReplayEqualsCpu(
        ToUnitFrame(4),
        cells,
        kSiblingChecks,
        0,
        buffers,
        arrays,
        "mutated list " + std::to_string(seed)
    );
    ApplyCpuErrors(ToUnitFrame(4), cells, buffers, errors_seen);
  }
  for (const GridError error : {
           GridError::kBadInput,
           GridError::kCoordOutOfRange,
           GridError::kCellsOverlap,
           GridError::kSiblingCellMissing,
           GridError::kSizeStepRuleBroken,
       }) {
    EXPECT_TRUE(errors_seen.contains(error)) << static_cast<int>(error);
  }
}

// The list of BuildSizeStepBreakWithSiblings passes the siblings, so with siblings required the size step rule fails:
// size level 2 {4,0,0}, at position 1, beside size level 0 {0,0,0} in direction 12.
TEST(BuildSortedCellList, SizeStepBreakWithSiblings_IsSizeStepRuleBrokenEitherWay) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  const std::vector<CellId> cells = BuildSizeStepBreakWithSiblings();
  const GridFailure expected = {
      .error = GridError::kSizeStepRuleBroken,
      .cell = ToCell(2, 4, 0, 0),
      .other_cell = ToCell(0, 0, 0, 0),
  };
  for (const SiblingCheck sibling_check : kSiblingChecks) {
    const std::expected<std::uint32_t, GridFailure> result = BuildSortedCellList(
        {.frame = ToUnitFrame(3), .active_cells = cells, .version = {.value = 1}},
        buffers,
        sibling_check
    );
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(ToFailureFields(result.error()), ToFailureFields(expected));
  }
}

// The size-level-2 cell {188,188,188} touches the box's last cell, size level 0 {46,46,46}, at the corner (47, 47, 47)
// m: sorted last, it fails the size step rule, or first its siblings, at position 103,823. Each list runs with the
// GPU's own launch, one item per thread, and with 1,000 threads, about 104 items per thread, the shape the GPU takes
// for a list of more than 65,535 x 256 cells.
TEST(BuildSortedCellList, LargeListAfterStaleValues_SameFirstFailureAsCpu) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  const std::vector<CellId> failing_cells = BuildLargeBoxOfCells({ToCell(2, 188, 188, 188)});
  const std::vector<CellId> valid_cells = BuildLargeBoxOfCells({});
  for (const std::vector<CellId>* cells : {&failing_cells, &valid_cells}) {
    for (const std::size_t thread_count : {std::size_t{0}, std::size_t{1000}}) {
      ExpectReplayEqualsCpu(
          ToUnitFrame(3),
          *cells,
          kSiblingChecks,
          thread_count,
          buffers,
          arrays,
          "large box of " + std::to_string(cells->size()) + ", threads " + std::to_string(thread_count)
      );
    }
  }
  // The failure is the one the CPU names, not the stale record's.
  const std::expected<std::uint32_t, GridFailure> skipped = BuildSortedCellList(
      {.frame = ToUnitFrame(3), .active_cells = failing_cells, .version = {.value = 1}},
      buffers,
      SiblingCheck::kSkip
  );
  ASSERT_FALSE(skipped.has_value());
  EXPECT_EQ(skipped.error().error, GridError::kSizeStepRuleBroken);
  EXPECT_EQ(skipped.error().cell.value, ToCell(2, 188, 188, 188).value);
}

// With 1 and 7 threads every thread has several items, so a thread of P4 to P6 returns after an item of its own when
// an earlier step failed: the lists that fail, in both launch orders.
TEST(BuildSortedCellList, SeveralItemsPerThread_SameFirstFailureAsCpu) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  for (const std::size_t thread_count : {std::size_t{1}, std::size_t{7}}) {
    const std::string threads = ", threads " + std::to_string(thread_count);
    for (const std::vector<CellId>& cells : BuildEveryTwoSizeLevelsApartList()) {
      ExpectReplayEqualsCpu(ToUnitFrame(3), cells, kSiblingChecks, thread_count, buffers, arrays, "apart" + threads);
    }
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
      ExpectReplayEqualsCpu(
          ToUnitFrame(4),
          BuildMutatedGridCells(seed),
          kSiblingChecks,
          thread_count,
          buffers,
          arrays,
          "mutated list " + std::to_string(seed) + threads
      );
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The build on the GPU up to the block ids (<proc-build-grid-in-parallel> P1 to P8), replayed on the CPU
// ---------------------------------------------------------------------------------------------------------------------

// Which arrays a build replay starts from.
enum class ReplayStart : std::uint8_t {
  kAfterStaleValues,  // ApplyStaleValues first.
  kAfterPreviousList,  // The arrays as the replay of the list before left them, cells of another grid included.
};

// The `count` values from `first`, copied into a vector.
template <typename Value>
std::vector<Value> ToVector(const Value* first, std::size_t count) {
  return std::vector<Value>(first, first + count);
}

// The used part of the five tables of `tables`, block_count blocks of each, copied into vectors.
TableVectors ToTableVectors(const GridTables& tables) {
  const std::size_t block_count = tables.block_count;
  return {
      .frame = tables.frame,
      .block_ids = ToVector(tables.block_ids, block_count),
      .same_size_level_blocks = ToVector(tables.same_size_level_blocks, kSameSizeLevelEntriesPerBlock * block_count),
      .bigger_block = ToVector(tables.bigger_block, block_count),
      .smaller_blocks = ToVector(tables.smaller_blocks, kSmallerBlocksPerBlock * block_count),
      .cell_states = ToVector(tables.cell_states, kCellPlacesPerBlock * block_count),
      .face_piece_kinds = ToVector(tables.face_piece_kinds, kFacePlacesPerBlock * block_count),
  };
}

// What one build replay gave: the failure, on success the counts and the used tables, and the two counts the status
// record holds after P8.
struct BuildReplayResult {
  GridFailure failure;
  std::optional<TableVectors> tables;  // On success only.
  GridCounts counts{};  // On success only: the counts of the status record.
  std::uint32_t cell_blocks = 0;
  std::uint32_t split_cells = 0;
  int failed_prefix_sum_count = 0;
};

// The counts of a build's status record, as the CPU half of a build on the GPU reads them on success.
GridCounts ToCountsOfStatus(const GridBuildStatus& status) {
  GridCounts counts{};
  counts.cell_blocks = status.cell_blocks;
  std::ranges::copy(status.active_cells_per_size_level, &counts.active_cells_per_size_level[0]);
  counts.active_cells = status.active_cells;
  counts.face_pieces = status.face_pieces;
  return counts;
}

// Replays the build of `inputs` on the room of `buffers` into `arrays` (ApplyGpuBuildStepsByReplay), collects what it
// gave and expects every guard of the arrays unchanged: no step wrote past an array of the room.
BuildReplayResult ComputeBuildReplay(
    const GridInputs& inputs,
    const GridBuffers& buffers,
    ReplayLaunch launch,
    ReplayChange change,
    ReplayArrays& arrays
) {
  BuildReplayResult result;
  result.failure = ApplyGpuBuildStepsByReplay(inputs, buffers, launch, change, arrays);
  result.cell_blocks = arrays.status.cell_blocks;
  result.split_cells = arrays.status.split_cells;
  result.failed_prefix_sum_count = arrays.failed_prefix_sum_count;
  EXPECT_TRUE(ComputeAreGuardsUnchanged(arrays)) << "a write left the room's part of a replay array";
  EXPECT_EQ(arrays.stale_table_read_count, 0) << "P10 (c) or P11 read a block past the block count";
  if (result.failure.error == GridError::kNone && result.cell_blocks <= arrays.max_cell_blocks) {
    result.tables = ToTableVectors(ToReplayTables(inputs.frame, inputs.version, arrays));
    result.counts = ToCountsOfStatus(arrays.status);
  }
  return result;
}

// The split cells and the blocks of a list, by the test's own floor, for a list that passed steps 1 to 4.
struct ExpectedBlocks {
  std::uint32_t split_cells = 0;
  std::uint32_t cell_blocks = 0;
};

ExpectedBlocks ComputeExpectedBlocks(std::span<const CellId> cells) {
  const std::vector<CellId> active_cells(cells.begin(), cells.end());
  const std::set<std::uint64_t> split_cells = BuildSplitCells(active_cells);
  std::set<std::uint64_t> blocks;
  for (const CellId cell : active_cells) {
    blocks.insert(ToBlockOfCell(cell).value);
  }
  for (const std::uint64_t split_cell : split_cells) {
    blocks.insert(ToBlockOfCell(CellId{.value = split_cell}).value);
  }
  return {
      .split_cells = static_cast<std::uint32_t>(split_cells.size()),
      .cell_blocks = static_cast<std::uint32_t>(blocks.size()),
  };
}

// What the CPU build of a list gave, and what the status counts of its replay must be.
struct CpuBuildOfList {
  GridFailure failure;
  std::optional<TableVectors> tables;  // On success only: the used part of the grid's tables.
  GridCounts counts{};  // On success only.
  bool is_list_too_long = false;  // The input checks of P1 failed it: the replay never reached the GPU steps.
  ExpectedBlocks status_counts;  // The split cells and blocks the status record must hold otherwise.
};

// Runs BuildGrid on the CPU buffers, keeps the version check, the counts and the used tables of the grid, and takes
// the buffers back. The status counts: after a failure of steps 1 to 4, zero; otherwise the split cells of the list,
// and the blocks of the list unless step 5 failed, when the marks are 0.
CpuBuildOfList ComputeCpuBuildOfList(const GridInputs& inputs, GridBuffers& buffers) {
  CpuBuildOfList cpu_build;
  std::expected<StaggeredGrid, GridFailure> cpu_grid = BuildGrid(inputs, buffers);
  if (cpu_grid.has_value()) {
    EXPECT_EQ(cpu_grid->version().value, inputs.version.value);
    cpu_build.tables = ToTableVectors(cpu_grid->tables(RunOn::kCpu));
    cpu_build.counts = cpu_grid->counts();
    buffers = TakeBuffers(std::move(*cpu_grid));
  } else {
    cpu_build.failure = cpu_grid.error();
  }
  const std::uint64_t room_cell_places = std::uint64_t{kCellPlacesPerBlock} * buffers.max_cell_blocks();
  cpu_build.is_list_too_long = inputs.active_cells.size() > room_cell_places;
  const bool has_passed_sorted_checks =
      cpu_build.failure.error == GridError::kNone || cpu_build.failure.error == GridError::kTooManyCellBlocks;
  if (!has_passed_sorted_checks || cpu_build.is_list_too_long) {
    return cpu_build;
  }
  cpu_build.status_counts = ComputeExpectedBlocks(inputs.active_cells);
  if (inputs.active_cells.size() + cpu_build.status_counts.split_cells > room_cell_places) {
    cpu_build.status_counts.cell_blocks = 0;
  }
  return cpu_build;
}

// Says whether two copies of used tables hold the same bytes: the block count and every entry of the five tables.
bool ComputeIsSameTableVectors(const std::optional<TableVectors>& first, const std::optional<TableVectors>& second) {
  if (!first.has_value() || !second.has_value()) {
    return first.has_value() == second.has_value();
  }
  return ComputeIsSameUsedTables(first->ToGridTables(), second->ToGridTables());
}

// Expects one build replay to give the CPU's failure, no failed prefix sum, on success the CPU's counts and the same
// bytes in the used part of the five tables, and, when the GPU steps ran, the status counts after P8.
void ExpectBuildReplayEqualsCpuBuild(
    const BuildReplayResult& replay,
    const CpuBuildOfList& cpu_build,
    const std::string& context
) {
  SCOPED_TRACE(context);
  EXPECT_EQ(ToFailureFields(replay.failure), ToFailureFields(cpu_build.failure));
  EXPECT_EQ(replay.failed_prefix_sum_count, 0);
  EXPECT_TRUE(ComputeIsSameTableVectors(replay.tables, cpu_build.tables));
  EXPECT_EQ(ToGridCountFields(replay.counts), ToGridCountFields(cpu_build.counts));
  // The input checks of P1 stop a list too long before any GPU step, so the record keeps no count of it.
  const ExpectedBlocks replay_status_counts = {.split_cells = replay.split_cells, .cell_blocks = replay.cell_blocks};
  const ExpectedBlocks expected_status_counts =
      cpu_build.is_list_too_long ? replay_status_counts : cpu_build.status_counts;
  EXPECT_EQ(replay_status_counts.split_cells, expected_status_counts.split_cells);
  EXPECT_EQ(replay_status_counts.cell_blocks, expected_status_counts.cell_blocks);
}

// Replays the build of `cells` in both launch orders with `thread_count` threads (0 for the GPU's own), the forward
// order from `start` and the reverse one after ApplyStaleValues, then runs the CPU build (ComputeCpuBuildOfList), and
// expects each replay to equal it (ExpectBuildReplayEqualsCpuBuild). On success the status block count is the CPU's.
//
// clang-tidy: a room and a thread count; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ExpectBuildReplayEqualsCpu(
    GridFrame frame,
    std::span<const CellId> cells,
    std::size_t thread_count,
    ReplayStart start,
    GridBuffers& buffers,
    ReplayArrays& arrays,
    const std::string& name
) {
  const GridInputs inputs = {.frame = frame, .active_cells = cells, .version = {.value = 1}};
  std::array<BuildReplayResult, 2> replays;
  for (std::size_t order_index = 0; order_index < kLaunchOrders.size(); ++order_index) {
    if (order_index == 1 || start == ReplayStart::kAfterStaleValues) {
      ApplyStaleValues(arrays);
    }
    const ReplayLaunch launch = {.order = kLaunchOrders.at(order_index), .thread_count = thread_count};
    replays.at(order_index) = ComputeBuildReplay(inputs, buffers, launch, ReplayChange::kNone, arrays);
  }
  const CpuBuildOfList cpu_build = ComputeCpuBuildOfList(inputs, buffers);
  ExpectBuildReplayEqualsCpuBuild(replays.at(0), cpu_build, name + ", forward");
  ExpectBuildReplayEqualsCpuBuild(replays.at(1), cpu_build, name + ", reverse");
}

// Replays the full build of random grid 1 into `arrays` after ApplyStaleValues, then the build of the 14 lone cells
// {4 i, 0, 0} with `change`, both on the room of `buffers`, and returns the second.
BuildReplayResult ComputeLoneCellsAfterGridOne(ReplayChange change, const GridBuffers& buffers, ReplayArrays& arrays) {
  const GridInputs grid_one_inputs = {
      .frame = ToUnitFrame(4),
      .active_cells = BuildShuffledRandomGrids().at(0),
      .version = {.value = 1},
  };
  const std::vector<CellId> lone_cells = BuildLoneCellsAlongX(14);
  const GridInputs lone_inputs = {.frame = ToUnitFrame(4), .active_cells = lone_cells, .version = {.value = 1}};
  const ReplayLaunch launch = {.order = LaunchOrder::kForward, .thread_count = 0};
  ApplyStaleValues(arrays);
  const BuildReplayResult grid_one_build =
      ComputeBuildReplay(grid_one_inputs, buffers, launch, ReplayChange::kNone, arrays);
  EXPECT_EQ(grid_one_build.failure.error, GridError::kNone);
  return ComputeBuildReplay(lone_inputs, buffers, launch, change, arrays);
}

// Expects places 14 and 15 of the lone cells' build to hold cell ids just before step (d) and the id 0 after it.
void ExpectTailStaleThenFilled(const ReplayArrays& arrays) {
  ASSERT_EQ(arrays.cells_before_tail_fill.size(), 16U);
  const std::array<std::uint64_t, 2> cell_marks_before = {
      arrays.cells_before_tail_fill.at(14).value & kIdCellMark,
      arrays.cells_before_tail_fill.at(15).value & kIdCellMark,
  };
  EXPECT_EQ(cell_marks_before, (std::array<std::uint64_t, 2>{kIdCellMark, kIdCellMark}));
  const std::array<std::uint64_t, 2> values_after = {
      arrays.cell_ids.data()[14].value,
      arrays.cell_ids.data()[15].value,
  };
  EXPECT_EQ(values_after, (std::array<std::uint64_t, 2>{0, 0}));
}

// Just before step (d) the tail places 14 and 15 hold cells of random grid 1, the build before; the fill turns them
// into the id 0, so the 14 lone cells give 14 blocks, the CPU's. The copy without the fill shows those cells become
// blocks: they share one block, so 15.
TEST(ApplySortTailFillAt, UsedBuffers_HoldNoStaleCell) {
  constexpr std::uint32_t kRoom = 100;
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner(kRoom);
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays(kRoom);
  ASSERT_TRUE(arrays.order_buffers.has_value());

  const BuildReplayResult lone_build = ComputeLoneCellsAfterGridOne(ReplayChange::kNone, buffers, arrays);
  ExpectTailStaleThenFilled(arrays);
  const std::vector<CellId> lone_cells = BuildLoneCellsAlongX(14);
  const CpuBuildOfList cpu_build =
      ComputeCpuBuildOfList({.frame = ToUnitFrame(4), .active_cells = lone_cells, .version = {.value = 1}}, buffers);
  EXPECT_EQ(lone_build.cell_blocks, 14U);
  ExpectBuildReplayEqualsCpuBuild(lone_build, cpu_build, "lone cells after random grid 1");

  const BuildReplayResult without_fill = ComputeLoneCellsAfterGridOne(ReplayChange::kWithoutTailFill, buffers, arrays);
  EXPECT_EQ(without_fill.cell_blocks, 15U);
}

// Replays the build of `cells` on the room of `buffers` in both launch orders, each after ApplyStaleAllOnesWords, and
// expects no failed prefix sum and the CPU build's failure, whose error is `error`.
void ExpectTrueErrorAfterAllOnesWords(
    const std::vector<CellId>& cells,
    GridBuffers& buffers,
    GridError error,
    ReplayArrays& arrays
) {
  const GridInputs inputs = {.frame = ToUnitFrame(4), .active_cells = cells, .version = {.value = 1}};
  std::array<BuildReplayResult, 2> replays;
  for (std::size_t order_index = 0; order_index < kLaunchOrders.size(); ++order_index) {
    ApplyStaleAllOnesWords(arrays);
    const ReplayLaunch launch = {.order = kLaunchOrders.at(order_index), .thread_count = 0};
    replays.at(order_index) = ComputeBuildReplay(inputs, buffers, launch, ReplayChange::kNone, arrays);
  }
  const CpuBuildOfList cpu_build = ComputeCpuBuildOfList(inputs, buffers);
  EXPECT_EQ(cpu_build.failure.error, error);
  ExpectBuildReplayEqualsCpuBuild(
      replays.at(0),
      cpu_build,
      "error " + std::to_string(static_cast<int>(error)) + ", forward"
  );
  ExpectBuildReplayEqualsCpuBuild(
      replays.at(1),
      cpu_build,
      "error " + std::to_string(static_cast<int>(error)) + ", reverse"
  );
}

// Every working array starts as 0xFFFFFFFF words. P7 (a) and the marks of P8 write 0 after an earlier failure, so both
// prefix sums add zeros and the build names the true failure, never kInternalFault.
TEST(BuildGrid, StaleLargeValuesWithFailingList_ReportsTheTrueError) {
  std::expected<TestGrid, GridFailure> owner_of_100 = BuildReplayBuffersOwner(100);
  GridBuffers buffers_of_100 = TakeEmptyGridBuffers(owner_of_100);
  std::expected<TestGrid, GridFailure> owner_of_1 = BuildReplayBuffersOwner(1);
  GridBuffers buffers_of_1 = TakeEmptyGridBuffers(owner_of_1);
  ReplayArrays arrays = BuildReplayArrays(100);
  ReplayArrays arrays_of_1 = BuildReplayArrays(1);
  ASSERT_TRUE(arrays.order_buffers.has_value());
  ASSERT_TRUE(arrays_of_1.order_buffers.has_value());
  std::vector<CellId> repeated_cell = BuildLoneCellsAlongX(9);
  repeated_cell.push_back(ToCell(0, 8, 0, 0));
  ExpectTrueErrorAfterAllOnesWords(repeated_cell, buffers_of_100, GridError::kCellsOverlap, arrays);
  ExpectTrueErrorAfterAllOnesWords(BuildSevenOfEightSiblings(), buffers_of_100, GridError::kSiblingCellMissing, arrays);
  ExpectTrueErrorAfterAllOnesWords(
      BuildSmallCellsOfEightBigCells(),
      buffers_of_1,
      GridError::kTooManyCellBlocks,
      arrays_of_1
  );
}

// After P3 the order P7 (a) writes into holds a permutation of 0..n - 1, whose sum passes 2^32 - 1 from 92,683 cells.
// The box of 103,825 cells fails its siblings at the last position; a copy of P7 (a) that returns there instead of
// writing 0 leaves that permutation, and the first prefix sum fails. The GPU's sequence names the CPU's failure.
TEST(BuildGrid, SplitCountsThatReturnEarly_FailThePrefixSum) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  const std::vector<CellId> failing_cells = BuildLargeBoxOfCells({ToCell(2, 188, 188, 188)});
  const GridInputs inputs = {.frame = ToUnitFrame(3), .active_cells = failing_cells, .version = {.value = 1}};

  const ReplayLaunch launch = {.order = LaunchOrder::kForward, .thread_count = 0};
  ApplyStaleAllOnesWords(arrays);
  const BuildReplayResult as_on_gpu = ComputeBuildReplay(inputs, buffers, launch, ReplayChange::kNone, arrays);
  ApplyStaleAllOnesWords(arrays);
  const BuildReplayResult returning_early =
      ComputeBuildReplay(inputs, buffers, launch, ReplayChange::kSplitCountsReturnEarly, arrays);
  const std::expected<GridCounts, GridFailure> cpu_counts = ComputeGridCounts(inputs, buffers);
  ASSERT_FALSE(cpu_counts.has_value());
  EXPECT_EQ(cpu_counts.error().error, GridError::kSiblingCellMissing);
  EXPECT_EQ(as_on_gpu.failed_prefix_sum_count, 0);
  EXPECT_EQ(ToFailureFields(as_on_gpu.failure), ToFailureFields(cpu_counts.error()));
  EXPECT_EQ(returning_early.failed_prefix_sum_count, 1);
  EXPECT_EQ(returning_early.failure.error, GridError::kInternalFault);
}

// A failing list and the valid grid whose build the arrays hold before it.
struct ListAfterOtherGrid {
  std::vector<CellId> other_cells;  // A valid list that fits the room.
  std::vector<CellId> failing_cells;
  std::size_t thread_count = 0;  // 0 for the GPU's own.
};

// Replays and checks the build of lists.other_cells, then that of lists.failing_cells from the arrays it left
// (ExpectBuildReplayEqualsCpu), so the failing list starts from buffers that held another grid.
void ExpectFailingListAfterOtherGrid(
    const ListAfterOtherGrid& lists,
    GridBuffers& buffers,
    ReplayArrays& arrays,
    const std::string& name
) {
  ExpectBuildReplayEqualsCpu(
      ToUnitFrame(4),
      lists.other_cells,
      lists.thread_count,
      ReplayStart::kAfterStaleValues,
      buffers,
      arrays,
      name + ", the grid before"
  );
  ExpectBuildReplayEqualsCpu(
      ToUnitFrame(4),
      lists.failing_cells,
      lists.thread_count,
      ReplayStart::kAfterPreviousList,
      buffers,
      arrays,
      name
  );
}

// Active and split cells that fit the room but, with P7 (b) checking against one place less than they need, not the
// sort count: the check offers kSortCountFaultKey, which reads as kInternalFault with no cell; every later step stops,
// so the marks are 0 and no block id is written; no write leaves the arrays.
TEST(BuildGrid, CellsBeyondSortCount_IsInternalFaultWithNoBlockWritten) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  const GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  const GridInputs inputs = {
      .frame = ToUnitFrame(4),
      .active_cells = BuildShuffledRandomGrids().at(0),
      .version = {.value = 1},
  };
  ApplyStaleValues(arrays);
  const BuildReplayResult replay = ComputeBuildReplay(
      inputs,
      buffers,
      {.order = LaunchOrder::kForward, .thread_count = 0},
      ReplayChange::kSortCountBelowCells,
      arrays
  );
  EXPECT_EQ(arrays.status.failure_key, kSortCountFaultKey);
  EXPECT_EQ(ToFailureFields(replay.failure), ToFailureFields({.error = GridError::kInternalFault}));
  EXPECT_GT(replay.split_cells, 0U);
  EXPECT_EQ(replay.cell_blocks, 0U);
  EXPECT_EQ(arrays.block_ids.data()[0].value, ToCellBlockId(ToStaleCellId()).value);
}

// The lists of the gpu tests BuildGrid.OnGpu_EqualsCpuTablesBitForBit and BuildGrid.OnGpuFailures_EqualCpuFailures,
// replayed through P12. The empty list, one cell, the two-size fixture and the random grids pass every step: their
// version, counts and the used part of the five tables show P7 to P12 give the CPU's grid byte for byte. The mutated
// lists, seeds 1..kReplayedMutatedGridCount as in the check's replay (all 500 would take the test program past 45 s in
// the debug preset; the gpu test runs them all), fail at steps 1 to 4, and show every later step stops. The room
// checks: 9 blocks into room for 8 (step 6), 513 cells into the same (the length check of P1), and 72 active and split
// cells into room for 1 (step 5, after the split cells are counted). Each replay starts from arrays that held another
// grid, and the CPU build runs in CPU buffers that held one too.
TEST(BuildGrid, EveryThreadBlockAndThread_EqualsCpuTables) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  ApplyStaleValues(arrays);
  const std::vector<CellId> one_cell = {ToCell(0, 1, -2, 3)};
  const std::vector<CellId> two_size_cells = BuildTwoSizeFixtureCells();
  for (const auto& [frame, cells, name] : {
           std::tuple{ToUnitFrame(3), std::span<const CellId>(), std::string("the empty list")},
           std::tuple{ToUnitFrame(3), std::span<const CellId>(one_cell), std::string("one cell")},
           std::tuple{ToUnitFrame(3), std::span<const CellId>(two_size_cells), std::string("the two-size fixture")},
       }) {
    ExpectBuildReplayEqualsCpu(frame, cells, 0, ReplayStart::kAfterPreviousList, buffers, arrays, name);
  }
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    ExpectBuildReplayEqualsCpu(
        ToUnitFrame(4),
        BuildShuffledRandomGrids().at(grid),
        0,
        ReplayStart::kAfterPreviousList,
        buffers,
        arrays,
        "random grid " + std::to_string(grid)
    );
  }
  for (std::uint64_t seed = 1; seed <= kReplayedMutatedGridCount; ++seed) {
    ExpectBuildReplayEqualsCpu(
        ToUnitFrame(4),
        BuildMutatedGridCells(seed),
        0,
        ReplayStart::kAfterPreviousList,
        buffers,
        arrays,
        "mutated list " + std::to_string(seed)
    );
  }

  std::expected<TestGrid, GridFailure> owner_of_8 = BuildReplayBuffersOwner(8);
  GridBuffers buffers_of_8 = TakeEmptyGridBuffers(owner_of_8);
  std::expected<TestGrid, GridFailure> owner_of_1 = BuildReplayBuffersOwner(1);
  GridBuffers buffers_of_1 = TakeEmptyGridBuffers(owner_of_1);
  ReplayArrays arrays_of_8 = BuildReplayArrays(8);
  ReplayArrays arrays_of_1 = BuildReplayArrays(1);
  ASSERT_TRUE(arrays_of_8.order_buffers.has_value());
  ASSERT_TRUE(arrays_of_1.order_buffers.has_value());
  const std::vector<CellId> eight_blocks = BuildLoneCellsAlongX(8);
  const std::vector<CellId> one_block = {ToCell(0, 0, 0, 0), ToCell(0, 1, 0, 0), ToCell(0, 2, 0, 0)};
  for (const std::size_t thread_count : {std::size_t{0}, std::size_t{1}, std::size_t{7}}) {
    const std::string threads = ", threads " + std::to_string(thread_count);
    ExpectFailingListAfterOtherGrid(
        {.other_cells = eight_blocks, .failing_cells = BuildLoneCellsAlongX(9), .thread_count = thread_count},
        buffers_of_8,
        arrays_of_8,
        "nine blocks into eight" + threads
    );
    ExpectFailingListAfterOtherGrid(
        {
            .other_cells = eight_blocks,
            .failing_cells = BuildLoneCellsAlongX((64 * 8) + 1),
            .thread_count = thread_count,
        },
        buffers_of_8,
        arrays_of_8,
        "513 cells into eight blocks" + threads
    );
    ExpectFailingListAfterOtherGrid(
        {.other_cells = one_block, .failing_cells = BuildSmallCellsOfEightBigCells(), .thread_count = thread_count},
        buffers_of_1,
        arrays_of_1,
        "72 cells into one block" + threads
    );
  }
}

// Says whether every value of the room's part of a replay table still holds `stale`, the value ApplyStaleValues wrote.
template <typename Value>
bool ComputeIsRoomPartStale(const Buffer<Value>& table, std::size_t room_count, Value stale) {
  return std::ranges::all_of(std::span<const Value>(table.data(), room_count), [stale](Value value) {
    return value == stale;
  });
}

// Replays P9 to P12 after a failure of step 6, the block count, in arrays whose states are a stale kAbsent and whose
// neighbour tables hold the stale kNoIndex - 1, with `launch`: the block count of the record names the whole room. P9
// stops on the key of step 6, and P10 and P11 stop on any key, so no stale entry is read (a P10 (c) that went on would
// read the stale neighbours and be counted), no state or face piece kind is written and no guard changes.
void ExpectNoTableReadAfterBlockCountFailure(ReplayLaunch launch, ReplayArrays& arrays) {
  const std::size_t room = arrays.max_cell_blocks;
  ApplyStaleValues(arrays);
  std::fill_n(arrays.cell_states.data(), kCellPlacesPerBlock * room, CellState::kAbsent);
  arrays.status = ComputeResetBuildStatus();
  arrays.status.failure_key = ToFailureKey(kBlockCountStep, 0, 0);
  arrays.status.cell_blocks = arrays.max_cell_blocks + 1;
  arrays.stale_table_read_count = 0;
  ApplyTableStepsByReplay(ToUnitFrame(2), 0, 0, launch, arrays);
  EXPECT_EQ(arrays.stale_table_read_count, 0);
  EXPECT_TRUE(ComputeIsRoomPartStale(arrays.cell_states, kCellPlacesPerBlock * room, CellState::kAbsent));
  EXPECT_TRUE(
      ComputeIsRoomPartStale(arrays.face_piece_kinds, kFacePlacesPerBlock * room, FacePieceKind::kOutsidePiece)
  );
  EXPECT_TRUE(ComputeIsRoomPartStale(arrays.bigger_and_smaller_blocks, 9 * room, kNoIndex - 1));
  EXPECT_TRUE(ComputeAreGuardsUnchanged(arrays));
}

// After a failure of step 6 no step of P9 to P11 reads or writes a table: both launch orders, the GPU's own launch and
// 1 and 7 threads.
TEST(BuildGrid, ReplayAfterBlockCountFailure_ReadsNoStaleTable) {
  ReplayArrays arrays = BuildReplayArrays(8);
  ASSERT_TRUE(arrays.order_buffers.has_value());
  for (const LaunchOrder order : kLaunchOrders) {
    for (const std::size_t thread_count : {std::size_t{0}, std::size_t{1}, std::size_t{7}}) {
      SCOPED_TRACE(
          std::string(order == LaunchOrder::kForward ? "forward" : "reverse") + ", threads "
          + std::to_string(thread_count)
      );
      ExpectNoTableReadAfterBlockCountFailure({.order = order, .thread_count = thread_count}, arrays);
    }
  }
}

// Replays P9 to P12 alone on the size-level-1 blocks (0,0,0) and (1,0,0), written by hand in arrays that hold stale
// values, with `launch`, and expects the key of step 7 at block 0, kInternalFault with no cell, both bigger blocks
// written kNoIndex, every cell state and face piece kind still stale, no count added and every guard unchanged.
void ExpectBlocksWithoutBiggerBlockFault(ReplayLaunch launch, ReplayArrays& arrays) {
  const std::size_t room = arrays.max_cell_blocks;
  ApplyStaleValues(arrays);
  arrays.status = ComputeResetBuildStatus();
  arrays.status.cell_blocks = 2;
  arrays.block_ids.data()[0] = ToBlock(1, 0, 0, 0);
  arrays.block_ids.data()[1] = ToBlock(1, 1, 0, 0);
  ApplyTableStepsByReplay(ToUnitFrame(2), 0, 0, launch, arrays);
  ApplyBuildStatusResult(ToUnitFrame(2), nullptr, 0, nullptr, 0, &arrays.status);
  const std::tuple<std::uint64_t, GridError, std::uint64_t, std::uint64_t> record = {
      arrays.status.failure_key,
      arrays.status.error,
      arrays.status.active_cells,
      arrays.status.face_pieces,
  };
  EXPECT_EQ(record, std::make_tuple(ToFailureKey(kBlockFaultStep, 0, 0), GridError::kInternalFault, 0U, 0U));
  EXPECT_EQ(ToFailureFields(ToGridFailure(arrays.status)), ToFailureFields({.error = GridError::kInternalFault}));
  const std::array<std::uint32_t, 2> bigger_blocks = {
      arrays.bigger_and_smaller_blocks.data()[0],
      arrays.bigger_and_smaller_blocks.data()[1],
  };
  EXPECT_EQ(bigger_blocks, (std::array<std::uint32_t, 2>{kNoIndex, kNoIndex}));
  EXPECT_TRUE(ComputeIsRoomPartStale(arrays.cell_states, kCellPlacesPerBlock * room, CellState::kSplit));
  EXPECT_TRUE(
      ComputeIsRoomPartStale(arrays.face_piece_kinds, kFacePlacesPerBlock * room, FacePieceKind::kOutsidePiece)
  );
  EXPECT_TRUE(ComputeAreGuardsUnchanged(arrays));
}

// The replay of P9 to P12 alone, from block ids written by hand: the size-level-1 blocks (0,0,0) and (1,0,0) with no
// size-level-0 block, which no valid list gives, since the siblings check stops it. Both blocks offer the key of step
// 7; the smaller, at block 0, wins in either launch order, since a key of step 7 does not stop a thread of P9. P10 and
// P11 then do nothing, so every cell state and face piece kind keeps its stale value; P12 names kInternalFault with no
// cell. Both launch orders, the GPU's own launch and 1 and 7 threads; no write leaves the arrays.
TEST(BuildGrid, ReplayWithBlockWithoutItsBiggerBlock_IsInternalFault) {
  ReplayArrays arrays = BuildReplayArrays(8);
  ASSERT_TRUE(arrays.order_buffers.has_value());
  for (const LaunchOrder order : kLaunchOrders) {
    for (const std::size_t thread_count : {std::size_t{0}, std::size_t{1}, std::size_t{7}}) {
      SCOPED_TRACE(
          std::string(order == LaunchOrder::kForward ? "forward" : "reverse") + ", threads "
          + std::to_string(thread_count)
      );
      ExpectBlocksWithoutBiggerBlockFault({.order = order, .thread_count = thread_count}, arrays);
    }
  }
}

// A status record holding `error` and the cells size level 1 {2,0,0} and size level 0 {1,0,0} under `failure_key`.
GridBuildStatus ToStatusRecord(std::uint64_t failure_key, GridError error) {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.failure_key = failure_key;
  status.error = error;
  status.cell = ToCell(1, 2, 0, 0);
  status.other_cell = ToCell(0, 1, 0, 0);
  return status;
}

// A key with no error, an error with no key, and keys of steps 0, 8 and 9, just outside the steps 1 to 7: each is a
// record the GPU cannot write, and each gives kInternalFault with no cell.
TEST(BuildSortedCellList, StatusThatContradictsItself_IsInternalFault) {
  const std::array<GridBuildStatus, 5> records = {
      ToStatusRecord(ToFailureKey(kOverlapStep, 5, 0), GridError::kNone),
      ToStatusRecord(kNoFailureKey, GridError::kCellsOverlap),
      ToStatusRecord(ToFailureKey(0, 3, 0), GridError::kCellsOverlap),
      ToStatusRecord(ToFailureKey(8, 0, 0), GridError::kCellsOverlap),
      ToStatusRecord(ToFailureKey(9, 0, 0), GridError::kCellsOverlap),
  };
  for (std::size_t record = 0; record < records.size(); ++record) {
    EXPECT_EQ(ToFailureFields(ToGridFailure(records.at(record))), ToFailureFields({.error = GridError::kInternalFault}))
        << record;
  }
}

// A key of step 5 with a detail of 2 or more is no key a build offers: the record contradicts itself, whatever its
// error. The sort count fault, step 5 with detail 1, carries kInternalFault and names no cell.
TEST(BuildSortedCellList, StatusOfStepFiveWithUnknownDetail_IsInternalFault) {
  const GridBuildStatus unknown_detail =
      ToStatusRecord(ToFailureKey(kCellPlacesStep, 0, 2), GridError::kTooManyCellBlocks);
  EXPECT_EQ(ToFailureFields(ToGridFailure(unknown_detail)), ToFailureFields({.error = GridError::kInternalFault}));
  const GridBuildStatus sort_count_fault = ToStatusRecord(kSortCountFaultKey, GridError::kInternalFault);
  EXPECT_EQ(ToFailureFields(ToGridFailure(sort_count_fault)), ToFailureFields({.error = GridError::kInternalFault}));
  const GridBuildStatus room = ToStatusRecord(ToFailureKey(kCellPlacesStep, 0, 0), GridError::kTooManyCellBlocks);
  EXPECT_EQ(ToGridFailure(room).error, GridError::kTooManyCellBlocks);
}

// Step 7, a block without its bigger block, is a fault of the build: whatever cells the record holds, none is reported.
TEST(BuildSortedCellList, StatusOfStepSeven_IsInternalFaultWithNoCell) {
  const GridBuildStatus status = ToStatusRecord(ToFailureKey(7, 2, 0), GridError::kInternalFault);
  EXPECT_EQ(ToFailureFields(ToGridFailure(status)), ToFailureFields({.error = GridError::kInternalFault}));
}

// Steps 1 and 6, the ends of the steps a check or a build names cells for, give the record's error and cells.
TEST(BuildSortedCellList, StatusOfFirstAndSixthStep_GivesTheRecordsFailure) {
  const GridBuildStatus id_status = ToStatusRecord(ToFailureKey(kIdStep, 4, 0), GridError::kBadInput);
  const GridFailure id_failure = ToGridFailure(id_status);
  EXPECT_EQ(id_failure.error, GridError::kBadInput);
  EXPECT_EQ(id_failure.cell.value, ToCell(1, 2, 0, 0).value);
  const GridBuildStatus room_status = ToStatusRecord(ToFailureKey(6, 0, 0), GridError::kTooManyCellBlocks);
  EXPECT_EQ(ToGridFailure(room_status).error, GridError::kTooManyCellBlocks);
}

TEST(BuildSortedCellList, StatusWithKeyAndError_GivesTheErrorAndItsCells) {
  GridBuildStatus status = ComputeResetBuildStatus();
  EXPECT_EQ(ToFailureFields(ToGridFailure(status)), ToFailureFields({}));
  status.failure_key = ToFailureKey(kSizeStepRuleStep, 1, 0);
  status.error = GridError::kSizeStepRuleBroken;
  status.cell = ToCell(2, 4, 4, 4);
  status.other_cell = ToCell(0, 0, 0, 0);
  const GridFailure expected = {
      .error = GridError::kSizeStepRuleBroken,
      .cell = ToCell(2, 4, 4, 4),
      .other_cell = ToCell(0, 0, 0, 0),
  };
  EXPECT_EQ(ToFailureFields(ToGridFailure(status)), ToFailureFields(expected));
}

}  // namespace
}  // namespace fluidity::grid
