#include "fluidity/grid/grid_build_steps.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_build_steps_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

constexpr std::uint64_t kAllOnes = ~std::uint64_t{0};

constexpr std::uint64_t kTwoToThe15 = std::uint64_t{1} << 15U;

constexpr std::uint64_t kTwoToThe40 = std::uint64_t{1} << 40U;

constexpr std::uint64_t kLargestPosition = (std::uint64_t{1} << 48U) - 1;

constexpr std::int32_t kTwoToThe15Coord = ComputePowerOfTwo(15);

constexpr std::int32_t kTwoToThe16 = ComputePowerOfTwo(16);

constexpr std::int32_t kTwoToThe17 = ComputePowerOfTwo(17);

constexpr std::int32_t kTwoToThe18 = ComputePowerOfTwo(18);

// The 8 small cells of a big cell, by eighth: x + 2 y + 4 z of the high halves.
void ApplyAppendSmallCells(CellId big_cell, std::vector<CellId>& cells) {
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
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

void CheckSameCheck(const CellCheck& actual, const CellCheck& expected) {
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
    ApplyAppendSmallCells(ComputeSmallCell(ToCell(0, 0, 0, 0), cell_eighth), cells);
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

// ---------------------------------------------------------------------------------------------------------------------
// Sort keys
// ---------------------------------------------------------------------------------------------------------------------

TEST(ToCellSortKey, CoordLimits_FitInThirtyEightBits) {
  // biased(kMaxCoord, 0) = (2^18 - 1 + 2^18) << 15 = 2^34 - 2^15; pass 0 adds the size level 0 in the low 4 bits.
  EXPECT_EQ(ToCellSortKey(ToCell(0, kMaxCoord, 0, 0), 0), (kTwoToThe34 - kTwoToThe15) << 4U);
  // biased(-kMaxCoord, 0) = (-(2^18 - 1) + 2^18) << 15 = 2^15.
  EXPECT_EQ(ToCellSortKey(ToCell(0, -kMaxCoord, 0, 0), 0), kTwoToThe15 << 4U);
  // Size level 15: biased(c, 15) = c + 2^33, at most 2^33 + 2^18 - 1, so pass 0 is at most 2^37 + 2^22 - 16 + 15.
  EXPECT_LT(ToCellSortKey(ToCell(15, kMaxCoord, 0, 0), 0), kTwoToThe38);
  EXPECT_LT(ToCellSortKey(ToCell(15, -kMaxCoord, 0, 0), 0), kTwoToThe38);
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
void CheckEqualKeysInBlock(int size_level, CellCoord block_coord) {
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
    CheckEqualKeysInBlock(size_level, {.x = -1, .y = 2, .z = -3});
    // The block at the low limit holds cells -2^18 + 1..-2^18 + 3 on each axis; the one at the high limit
    // 2^18 - 4..2^18 - 1.
    CheckEqualKeysInBlock(size_level, {.x = -kTwoToThe16, .y = -kTwoToThe16, .z = -kTwoToThe16});
    CheckEqualKeysInBlock(size_level, {.x = kTwoToThe16 - 1, .y = kTwoToThe16 - 1, .z = kTwoToThe16 - 1});
  }
}

TEST(ToBlockSortKey, InvalidId_SortsLast) {
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    EXPECT_EQ(ToBlockSortKey(CellId{0}, pass), kAllOnes);
  }
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
  CheckSameCheck(
      CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), second_position),
      {.error = GridError::kCellsOverlap, .cell = repeated_cell, .other_cell = repeated_cell, .detail = 0}
  );
  CheckSameCheck(CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), second_position - 1), {});
}

TEST(CheckOverlapAt, CellInsideActiveBigCell_NamesBothCells) {
  // Size level 2 {-3, 1, 2} lies in size level 0 {floor(-3 / 4), floor(1 / 4), floor(2 / 4)} = {-1, 0, 0}.
  const CellId small_cell = ToCell(2, -3, 1, 2);
  const CellId big_cell = ToCell(0, -1, 0, 0);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, big_cell});
  CheckSameCheck(
      CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kCellsOverlap, .cell = small_cell, .other_cell = big_cell, .detail = 0}
  );
  CheckSameCheck(CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, big_cell)), {});
}

TEST(CheckOverlapAt, CellInsideTwoActiveBiggerCells_NamesTheNearest) {
  // Size level 2 {-3, 1, 2} lies in size level 1 {-2, 0, 1} and in size level 0 {-1, 0, 0}; one size level up first.
  const CellId small_cell = ToCell(2, -3, 1, 2);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, -1, 0, 0), ToCell(1, -2, 0, 1)});
  CheckSameCheck(
      CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kCellsOverlap, .cell = small_cell, .other_cell = ToCell(1, -2, 0, 1), .detail = 0}
  );
}

TEST(CheckOverlapAt, CellAtCoordLimitsInsideSizeLevelZeroCell_NamesBothCells) {
  // Size level 15 {kMaxCoord, -kMaxCoord, 0} lies in size level 0 {7, -8, 0}: floor(+-(2^18 - 1) / 2^15).
  const CellId small_cell = ToCell(15, kMaxCoord, -kMaxCoord, 0);
  const CellId big_cell = ToCell(0, 7, -8, 0);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, big_cell});
  CheckSameCheck(
      CheckOverlapAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kCellsOverlap, .cell = small_cell, .other_cell = big_cell, .detail = 0}
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// Siblings: method paper <proc-build-grid> step 3
// ---------------------------------------------------------------------------------------------------------------------

TEST(CheckSiblingsAt, SevenOfEight_NamesTheMissing) {
  const std::vector<CellId> sorted_cells = BuildSevenOfEightSiblings();
  for (std::uint32_t position = 0; position < ToCount(sorted_cells); ++position) {
    CheckSameCheck(
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
  CheckSameCheck(
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
  ApplyAppendSmallCells(ToCell(1, 0, 0, 0), cells);
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
  CheckSameCheck(
      CheckSiblingsAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, chain_end)),
      {
          .error = expected_check.error,
          .cell = expected_check.cell,
          .other_cell = expected_check.other_cell,
          .detail = 119,
      }
  );
  // An active size-level-1 sibling, eighth 1 = {-1,-2,-2}, finds it at its first walk step.
  CheckSameCheck(
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
      CheckSameCheck(
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
    CheckSameCheck(
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
  ApplyAppendSmallCells(cell, cells);
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

// ---------------------------------------------------------------------------------------------------------------------
// Size step rule: method paper <proc-build-grid> step 4
// ---------------------------------------------------------------------------------------------------------------------

TEST(CheckSizeStepAt, CornerTouchTwoSizeLevelsApart_NamesDirectionZero) {
  // Size level 0 {0,0,0} covers size-level-2 places 0..3; the place (-1, -1, -1) from {4,4,4} is (3, 3, 3).
  const CellId small_cell = ToCell(2, 4, 4, 4);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, 0, 0, 0)});
  CheckSameCheck(
      CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kSizeStepRuleBroken, .cell = small_cell, .other_cell = ToCell(0, 0, 0, 0), .detail = 0}
  );
}

TEST(CheckSizeStepAt, BigCellOfTheTwo_SeesNothing) {
  const std::vector<CellId> sorted_cells = ToSortedCells({ToCell(2, 4, 4, 4), ToCell(0, 0, 0, 0)});
  CheckSameCheck(
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
  CheckSameCheck(
      CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kSizeStepRuleBroken, .cell = small_cell, .other_cell = ToCell(0, -2, -1, -1), .detail = 12}
  );
}

TEST(CheckSizeStepAt, CellAtLowCoordLimit_SeesNoCellBeyond) {
  // A packed id of the place beyond, size level 2 {-kMaxCoord - 1, 0, 0}, keeps x = 2^18, which lies in size level 0
  // {2^18 / 4, 0, 0} = {65536, 0, 0}: 2^18 + 2^18 - 1 = 524,287 size-level-2 widths from the cell.
  const CellId small_cell = ToCell(2, -kMaxCoord, 0, 0);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, 65536, 0, 0)});
  CheckSameCheck(CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)), {});
}

TEST(CheckSizeStepAt, CellAtHighCoordLimit_SeesTheBigCellBeside) {
  // The +X place of size level 2 {kMaxCoord, 0, 0} is x = 2^18, in size level 0 x = 2^18 / 4 = 65536, while the cell
  // itself lies in size level 0 x = floor((2^18 - 1) / 4) = 65535. Direction (0, 0, +1) is 9 + 3 + 2 - 1 = 13. The
  // directions before it with dx = +1 have dy or dz = -1, whose places lie in size level 0 y or z = -1.
  const CellId small_cell = ToCell(2, kMaxCoord, 0, 0);
  const std::vector<CellId> sorted_cells = ToSortedCells({small_cell, ToCell(0, 65536, 0, 0)});
  CheckSameCheck(
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
  CheckSameCheck(
      CheckSizeStepAt(sorted_cells.data(), ToCount(sorted_cells), FindByScan(sorted_cells, small_cell)),
      {.error = GridError::kSizeStepRuleBroken, .cell = small_cell, .other_cell = ToCell(0, 8, 7, 7), .detail = 2}
  );
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
  CheckSameCheck(
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
  CheckSameCheck(
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
  CheckSameCheck(
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
    CheckSameCheck(
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
    CheckSameCheck(
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
  CheckSameCheck(
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
    CheckSameCheck(
        ComputeFailureCells(ToUnitFrame(3), nullptr, 0, sorted_cells.data(), ToCount(sorted_cells), failure_key),
        {.error = GridError::kInternalFault}
    );
  }
}

// Step 7, a block without its bigger block, is a fault in the build: no cell is named, whatever the position.
TEST(ComputeFailureCells, StepSeven_IsInternalFaultWithNoCell) {
  const std::vector<CellId> sorted_cells = BuildSevenOfEightSiblings();
  for (const std::uint64_t block : {0U, 3U}) {
    CheckSameCheck(
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
    CheckSameCheck(
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
    CheckSameCheck(
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
  CheckSameCheck(
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
  CheckSameCheck(to_error(kSortCountFaultKey), {.error = GridError::kInternalFault});
  CheckSameCheck(to_error(ToFailureKey(kCellPlacesStep, 0, 0)), {.error = GridError::kTooManyCellBlocks});
  CheckSameCheck(to_error(ToFailureKey(kCellPlacesStep, 0, 2)), {.error = GridError::kInternalFault});
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

constexpr std::uint32_t kSelfEntry = 13;  // The offset (0, 0, 0): ((0 + 1) * 3 + (0 + 1)) * 3 + (0 + 1).

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
  CheckSameCheck(
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

TEST(ComputeFacePieceKind, ThreeSizeLevelsMeeting_GiveTheCheckersFacePieces) {
  // Size level 0 {0,0,0} active; {1,0,0} split, and its size-level-1 eighth 1, {3,0,0}, split again into size level 2
  // x 6..7, y 0..1, z 0..1. Size level 2 meets size level 1, which meets size level 0.
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    const CellId small_cell = ComputeSmallCell(ToCell(0, 1, 0, 0), cell_eighth);
    if (cell_eighth == 1) {
      ApplyAppendSmallCells(small_cell, cells);
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
  ApplyAppendSmallCells(ToCell(0, 0, 0, 0), cells);
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
  ApplyAppendSmallCells(ToCell(0, kTwoToThe17 - 1, 0, 0), cells);
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
  ApplyAppendSmallCells(ToCell(0, -kTwoToThe17 + 1, 0, 0), cells);
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
  ApplyAppendSmallCells(ToCell(0, 0, 0, 0), entries.cell_ids);
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

// Replays the build of `cells` on the room of `buffers` in both launch orders, each after ApplyStaleAllOnesWords, and
// expects no failed prefix sum and the CPU build's failure, whose error is `error`.
void CheckTrueErrorAfterAllOnesWords(
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
  CheckBuildReplayEqualsCpuBuild(
      replays.at(0),
      cpu_build,
      "error " + std::to_string(static_cast<int>(error)) + ", forward"
  );
  CheckBuildReplayEqualsCpuBuild(
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
  CheckTrueErrorAfterAllOnesWords(repeated_cell, buffers_of_100, GridError::kCellsOverlap, arrays);
  CheckTrueErrorAfterAllOnesWords(BuildSevenOfEightSiblings(), buffers_of_100, GridError::kSiblingCellMissing, arrays);
  CheckTrueErrorAfterAllOnesWords(
      BuildSmallCellsOfEightBigCells(),
      buffers_of_1,
      GridError::kTooManyCellBlocks,
      arrays_of_1
  );
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
void CheckNoTableReadAfterBlockCountFailure(ReplayLaunch launch, ReplayArrays& arrays) {
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
      CheckNoTableReadAfterBlockCountFailure({.order = order, .thread_count = thread_count}, arrays);
    }
  }
}

// Replays P9 to P12 alone on the size-level-1 blocks (0,0,0) and (1,0,0), written by hand in arrays that hold stale
// values, with `launch`, and expects the key of step 7 at block 0, kInternalFault with no cell, both bigger blocks
// written kNoIndex, every cell state and face piece kind still stale, no count added and every guard unchanged.
void CheckBlocksWithoutBiggerBlockFault(ReplayLaunch launch, ReplayArrays& arrays) {
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
      CheckBlocksWithoutBiggerBlockFault({.order = order, .thread_count = thread_count}, arrays);
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
