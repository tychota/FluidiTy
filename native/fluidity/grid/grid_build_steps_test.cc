#include "fluidity/grid/grid_build_steps.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/grid_test_support.h"

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
  EXPECT_EQ(largest_key & 0xFFU, 255U);
  EXPECT_LT(largest_key, kNoFailureKey);
  EXPECT_EQ(ToFailureStep(ToFailureKey(1, 0, 0)), 1);
  EXPECT_EQ(ToFailurePosition(ToFailureKey(1, 0, 0)), 0U);
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

}  // namespace
}  // namespace fluidity::grid
