#include "fluidity/grid/grid_build_steps.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
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

// The 200 random grids, each list shuffled and then sorted into the fixed order. Built once for the whole file, since
// several tests read them; seed s is entry s - 1.
const std::vector<std::vector<CellId>>& BuildSortedRandomGrids() {
  static const std::vector<std::vector<CellId>> sorted_grids = [] {
    std::vector<std::vector<CellId>> grids;
    grids.reserve(200);
    for (std::uint64_t seed = 1; seed <= 200; ++seed) {
      std::vector<CellId> cells = BuildRandomGridCells(seed, 4, 3);
      ApplyFixedShuffle(cells, seed + 1000);
      grids.push_back(ToSortedCells(cells));
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

}  // namespace
}  // namespace fluidity::grid
