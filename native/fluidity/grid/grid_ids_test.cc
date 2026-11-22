#include "fluidity/grid/grid_ids.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <tuple>
#include <type_traits>

#include "fluidity/base/axis.h"
#include "fluidity/grid/grid_test_support.h"

namespace fluidity::grid {
namespace {

// The 27 coordinates with -kMaxCoord, 0 or kMaxCoord on each axis.
constexpr std::array<CellCoord, 27> ToLimitCoords() {
  constexpr std::array<std::int32_t, 3> kAxisValues = {-kMaxCoord, 0, kMaxCoord};
  std::array<CellCoord, 27> limit_coords{};
  for (std::size_t index = 0; index < limit_coords.size(); ++index) {
    limit_coords.at(index) = {
        .x = kAxisValues.at(index % 3),
        .y = kAxisValues.at((index / 3) % 3),
        .z = kAxisValues.at(index / 9),
    };
  }
  return limit_coords;
}

constexpr std::array<CellCoord, 27> kLimitCoords = ToLimitCoords();

constexpr bool ComputeIsSameCoord(CellCoord first, CellCoord second) {
  return first.x == second.x && first.y == second.y && first.z == second.z;
}

constexpr bool ComputeIsRoundTrip(SizeLevel size_level, CellCoord cell_coord) {
  const CellId cell = ToCellId(size_level, cell_coord);
  return ToSizeLevel(cell) == size_level && ComputeIsSameCoord(ToCellCoord(cell), cell_coord);
}

static_assert(ComputeIsRoundTrip(SizeLevel{15}, {.x = -kMaxCoord, .y = 0, .z = kMaxCoord}));

TEST(ToCellId, RoundTrip_AtCoordLimits) {
  for (const SizeLevel size_level : {SizeLevel{0}, SizeLevel{15}}) {
    for (const CellCoord cell_coord : kLimitCoords) {
      EXPECT_NE(ToCellId(size_level, cell_coord).value, 0U);
      EXPECT_TRUE(ComputeIsRoundTrip(size_level, cell_coord))
          << static_cast<int>(size_level) << ": " << cell_coord.x << " " << cell_coord.y << " " << cell_coord.z;
    }
  }
}

static_assert(ComputeIsRoundTrip(SizeLevel{4}, {.x = kMaxCoord + 1, .y = -kMaxCoord, .z = 0}));

TEST(ToCellId, OnePastTheLimit_KeepsItsCoordinate) {
  const CellCoord cell_coord = {.x = kMaxCoord + 1, .y = -kMaxCoord, .z = 7};
  const CellCoord read_coord = ToCellCoord(ToCellId(SizeLevel{4}, cell_coord));
  EXPECT_EQ(read_coord.x, kMaxCoord + 1);
  EXPECT_EQ(read_coord.y, -kMaxCoord);
  EXPECT_EQ(read_coord.z, 7);
}

// Bit 63 set, size level 5 in bits 57..60, and each coordinate c stored as c + kMaxCoord in 19 bits.
constexpr std::uint64_t kExpectedCellIdValue = (std::uint64_t{1} << 63U) | (std::uint64_t{5} << 57U)
    | (std::uint64_t{(1U << 19U) - 1U} << 38U) | (std::uint64_t{kMaxCoord} << 19U) | std::uint64_t{0};
static_assert(ToCellId(SizeLevel{5}, {.x = -kMaxCoord, .y = 0, .z = kMaxCoord + 1}).value == kExpectedCellIdValue);

TEST(ToCellId, BitLayout_MatchesTheHeader) {
  const CellId cell = ToCellId(SizeLevel{5}, {.x = -kMaxCoord, .y = 0, .z = kMaxCoord + 1});
  EXPECT_EQ(cell.value, kExpectedCellIdValue);
}

constexpr CellCoord kHighSideOfLastCell = {.x = kMaxCoord + 1, .y = -kMaxCoord, .z = 0};
static_assert(ToFacePieceId(Axis::kX, SizeLevel{3}, kHighSideOfLastCell).value != 0);

TEST(ToFacePieceId, HighSideOfCellAtCoordLimit_RoundTrips) {
  const FacePieceId piece = ToFacePieceId(Axis::kX, SizeLevel{3}, kHighSideOfLastCell);
  EXPECT_NE(piece.value, 0U);
  EXPECT_NE(piece.value, ToFacePieceId(Axis::kY, SizeLevel{3}, kHighSideOfLastCell).value);
  // Axis X stored as 1 in bits 61..62, size level 3 in bits 57..60, then z, y and x as in a cell id.
  const std::uint64_t expected_value = (std::uint64_t{1} << 61U) | (std::uint64_t{3} << 57U)
      | (std::uint64_t{kMaxCoord} << 38U) | (std::uint64_t{0} << 19U) | std::uint64_t{(1U << 19U) - 1U};
  EXPECT_EQ(piece.value, expected_value);
}

constexpr std::uint64_t ToKeptAxis(FacePieceId piece) {
  return (piece.value >> 61U) & 3U;
}
static_assert(ToKeptAxis(ToFacePieceId(Axis::kZ, SizeLevel{0}, {.x = 0, .y = 0, .z = 0})) == 3);

TEST(ToFacePieceId, EveryValidAxis_IsKeptAsAxisPlusOne) {
  // Bit 63 stays clear, so no face piece id carries the mark of a cell.
  const CellCoord cell_high_coord = {.x = -1, .y = 2, .z = -kMaxCoord};
  const FacePieceId x_piece = ToFacePieceId(Axis::kX, SizeLevel{7}, cell_high_coord);
  const FacePieceId y_piece = ToFacePieceId(Axis::kY, SizeLevel{7}, cell_high_coord);
  const FacePieceId z_piece = ToFacePieceId(Axis::kZ, SizeLevel{7}, cell_high_coord);
  EXPECT_EQ(ToKeptAxis(x_piece), 1U);
  EXPECT_EQ(ToKeptAxis(y_piece), 2U);
  EXPECT_EQ(ToKeptAxis(z_piece), 3U);
  EXPECT_EQ(x_piece.value >> 63U, 0U);
  EXPECT_EQ(y_piece.value >> 63U, 0U);
  EXPECT_EQ(z_piece.value >> 63U, 0U);
}

static_assert(
    ToFacePieceId(Axis::kZ, SizeLevel{0}, {.x = -kMaxCoord, .y = -kMaxCoord, .z = -kMaxCoord}).value
    != ToFacePieceId(Axis::kY, SizeLevel{0}, {.x = -kMaxCoord, .y = -kMaxCoord, .z = -kMaxCoord}).value
);

TEST(ToFacePieceId, EveryAxisAndCoordLimit_GivesItsOwnId) {
  // 3 axes x 2 size levels x 27 low corners: 162 ids, every one different and none 0.
  std::set<std::uint64_t> values;
  for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
    for (const SizeLevel size_level : {SizeLevel{0}, SizeLevel{15}}) {
      for (const CellCoord cell_high_coord : kLimitCoords) {
        values.insert(ToFacePieceId(axis, size_level, cell_high_coord).value);
      }
    }
  }
  EXPECT_EQ(values.size(), 162U);
  EXPECT_EQ(values.count(0), 0U);
}

constexpr bool ComputeIsBigCellOfNegativeCoordsRight() {
  const CellCoord big_coord = ToCellCoord(ComputeBigCell(ToCellId(SizeLevel{2}, {.x = -1, .y = -2, .z = 3})));
  return ComputeIsSameCoord(big_coord, {.x = -1, .y = -1, .z = 1});
}
static_assert(ComputeIsBigCellOfNegativeCoordsRight());

TEST(ComputeBigCell, NegativeCoords_RoundTowardMinusInfinity) {
  const CellId big_cell = ComputeBigCell(ToCellId(SizeLevel{2}, {.x = -1, .y = -2, .z = 3}));
  EXPECT_EQ(ToSizeLevel(big_cell), SizeLevel{1});
  const CellCoord big_coord = ToCellCoord(big_cell);
  EXPECT_EQ(big_coord.x, -1);
  EXPECT_EQ(big_coord.y, -1);
  EXPECT_EQ(big_coord.z, 1);
}

TEST(ComputeBigCell, CellsAtCoordLimits_GiveTheHalvedFloor) {
  // floor(-262143 / 2) = -131072 and floor(262144 / 2) = 131072.
  const CellCoord big_coord =
      ToCellCoord(ComputeBigCell(ToCellId(SizeLevel{15}, {.x = -kMaxCoord, .y = kMaxCoord + 1, .z = kMaxCoord})));
  EXPECT_EQ(big_coord.x, -131072);
  EXPECT_EQ(big_coord.y, 131072);
  EXPECT_EQ(big_coord.z, 131071);
}

constexpr CellId kBigCellOfEighths = ToCellId(SizeLevel{6}, {.x = -3, .y = 0, .z = 5});
static_assert(ComputeBigCell(ComputeSmallCell(kBigCellOfEighths, 6)).value == kBigCellOfEighths.value);
static_assert(ComputeCellEighth(ComputeSmallCell(kBigCellOfEighths, 6)) == 6);

void ExpectEveryEighthGivesBack(CellId big_cell) {
  const auto small_size_level = static_cast<SizeLevel>(static_cast<int>(ToSizeLevel(big_cell)) + 1);
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    const CellId small_cell = ComputeSmallCell(big_cell, cell_eighth);
    EXPECT_EQ(ComputeBigCell(small_cell).value, big_cell.value) << cell_eighth;
    EXPECT_EQ(ComputeCellEighth(small_cell), cell_eighth);
    EXPECT_EQ(ToSizeLevel(small_cell), small_size_level);
  }
}

TEST(ComputeSmallCell, EveryEighth_GivesBackItsBigCell) {
  ExpectEveryEighthGivesBack(kBigCellOfEighths);
  // Negative coordinates, and the doubled coordinate -kMaxCoord + 1 near the low limit.
  ExpectEveryEighthGivesBack(ToCellId(SizeLevel{0}, {.x = -kMaxCoord / 2, .y = 0, .z = -1}));
}

TEST(ComputeSmallCell, EighthBits_AreXThenYThenZ) {
  // Eighth 6 = 0 + 2 * 1 + 4 * 1: the low x half, the high y and z halves.
  const CellCoord small_coord = ToCellCoord(ComputeSmallCell(kBigCellOfEighths, 6));
  EXPECT_EQ(small_coord.x, -6);
  EXPECT_EQ(small_coord.y, 1);
  EXPECT_EQ(small_coord.z, 11);
}

static_assert(
    ToCellBlockId(ToCellId(SizeLevel{1}, {.x = -4, .y = -4, .z = -4})).value
    == ToCellBlockId(ToCellId(SizeLevel{1}, {.x = -1, .y = -1, .z = -1})).value
);
static_assert(
    ToCellBlockId(ToCellId(SizeLevel{1}, {.x = -1, .y = -1, .z = -1})).value
    != ToCellBlockId(ToCellId(SizeLevel{1}, {.x = 0, .y = 0, .z = 0})).value
);

TEST(ToCellBlockId, CellsOfOneBlock_ShareIt) {
  const CellBlockId low_block = ToCellBlockId(ToCellId(SizeLevel{1}, {.x = -4, .y = -4, .z = -4}));
  const CellBlockId high_block = ToCellBlockId(ToCellId(SizeLevel{1}, {.x = 0, .y = 0, .z = 0}));
  EXPECT_NE(low_block.value, high_block.value);
  // Every cell of -4..3 on each axis: in the low block exactly when every axis is negative, in the high block exactly
  // when none is.
  for (std::int32_t index = 0; index < 8 * 8 * 8; ++index) {
    const CellCoord cell_coord = {.x = (index % 8) - 4, .y = ((index / 8) % 8) - 4, .z = (index / 64) - 4};
    const CellBlockId block = ToCellBlockId(ToCellId(SizeLevel{1}, cell_coord));
    const bool is_every_axis_negative = cell_coord.x < 0 && cell_coord.y < 0 && cell_coord.z < 0;
    const bool is_no_axis_negative = cell_coord.x >= 0 && cell_coord.y >= 0 && cell_coord.z >= 0;
    EXPECT_EQ(block.value == low_block.value, is_every_axis_negative) << index;
    EXPECT_EQ(block.value == high_block.value, is_no_axis_negative) << index;
  }
}

// The block of the size-level-15 cell at (x, 0, 0).
CellBlockId ToBlockOfSmallestCellAt(std::int32_t x) {
  return ToCellBlockId(ToCellId(SizeLevel{15}, {.x = x, .y = 0, .z = 0}));
}

TEST(ToCellBlockId, CellAtLowCoordLimit_GivesTheBlockOfItsFloorQuarter) {
  // floor(-262143 / 4) = -65536: that block holds the cells -262144..-262141, so -kMaxCoord..-kMaxCoord + 2 of the
  // range.
  const CellBlockId lowest_block = ToBlockOfSmallestCellAt(-kMaxCoord);
  EXPECT_NE(lowest_block.value, 0U);
  EXPECT_EQ(lowest_block.value, ToBlockOfSmallestCellAt(-kMaxCoord + 2).value);
  EXPECT_NE(lowest_block.value, ToBlockOfSmallestCellAt(-kMaxCoord + 3).value);
  EXPECT_TRUE(ComesBefore(lowest_block, ToBlockOfSmallestCellAt(-kMaxCoord + 3)));
}

TEST(ToCellBlockId, CellAtHighCoordLimit_GivesTheBlockOfItsFloorQuarter) {
  // floor(262143 / 4) = 65535: that block holds kMaxCoord - 3..kMaxCoord, and kMaxCoord + 1 starts block 65536.
  const CellBlockId highest_block = ToBlockOfSmallestCellAt(kMaxCoord);
  EXPECT_EQ(highest_block.value, ToBlockOfSmallestCellAt(kMaxCoord - 3).value);
  EXPECT_NE(highest_block.value, ToBlockOfSmallestCellAt(kMaxCoord - 4).value);
  EXPECT_NE(highest_block.value, ToBlockOfSmallestCellAt(kMaxCoord + 1).value);
  EXPECT_TRUE(ComesBefore(ToBlockOfSmallestCellAt(-kMaxCoord), highest_block));
  EXPECT_TRUE(ComesBefore(highest_block, ToBlockOfSmallestCellAt(kMaxCoord + 1)));
  EXPECT_FALSE(ComesBefore(highest_block, ToBlockOfSmallestCellAt(-kMaxCoord)));
}

// The blocks of size levels 1 and 2 whose low corner is the origin: the bigger one comes first.
static_assert(ComesBefore(
    ToCellBlockId(ToCellId(SizeLevel{1}, {.x = 0, .y = 0, .z = 0})),
    ToCellBlockId(ToCellId(SizeLevel{2}, {.x = 0, .y = 0, .z = 0}))
));

TEST(ComesBefore, BiggerBlockAtSameCorner_ComesFirst) {
  const CellBlockId big_block = ToCellBlockId(ToCellId(SizeLevel{1}, {.x = -4, .y = 4, .z = 0}));
  const CellBlockId small_block = ToCellBlockId(ToCellId(SizeLevel{2}, {.x = -8, .y = 8, .z = 0}));
  EXPECT_TRUE(ComesBefore(big_block, small_block));
  EXPECT_FALSE(ComesBefore(small_block, big_block));
}

static_assert(
    ComesBefore(ToCellId(SizeLevel{3}, {.x = 5, .y = 5, .z = -1}), ToCellId(SizeLevel{3}, {.x = -5, .y = -5, .z = 0}))
);

TEST(ComesBefore, LowerZ_ComesFirst) {
  // z decides before y and x, and a bigger cell with a lower z still comes first.
  const CellId low_cell = ToCellId(SizeLevel{3}, {.x = 5, .y = 5, .z = -1});
  const CellId high_cell = ToCellId(SizeLevel{3}, {.x = -5, .y = -5, .z = 0});
  EXPECT_TRUE(ComesBefore(low_cell, high_cell));
  EXPECT_FALSE(ComesBefore(high_cell, low_cell));
  const CellBlockId low_block = ToCellBlockId(ToCellId(SizeLevel{0}, {.x = 100, .y = 100, .z = -1}));
  const CellBlockId high_block = ToCellBlockId(ToCellId(SizeLevel{0}, {.x = -100, .y = -100, .z = 0}));
  EXPECT_TRUE(ComesBefore(low_block, high_block));
  EXPECT_FALSE(ComesBefore(high_block, low_block));
}

constexpr CellId kLowestBigCell = ToCellId(SizeLevel{0}, {.x = -kMaxCoord, .y = 0, .z = 0});
constexpr CellId kLowestSmallCell = ToCellId(SizeLevel{15}, {.x = -kMaxCoord, .y = 0, .z = 0});
constexpr CellId kHighestBigCell = ToCellId(SizeLevel{0}, {.x = kMaxCoord, .y = 0, .z = 0});
static_assert(ComesBefore(kLowestBigCell, kLowestSmallCell));

TEST(ComesBefore, CellsAtCoordLimits_OrderByLowCorner) {
  EXPECT_TRUE(ComesBefore(kLowestBigCell, kLowestSmallCell));
  EXPECT_TRUE(ComesBefore(kLowestSmallCell, kHighestBigCell));
  EXPECT_TRUE(ComesBefore(kLowestBigCell, kHighestBigCell));
  EXPECT_FALSE(ComesBefore(kLowestSmallCell, kLowestBigCell));
  EXPECT_FALSE(ComesBefore(kHighestBigCell, kLowestSmallCell));
  EXPECT_FALSE(ComesBefore(kHighestBigCell, kHighestBigCell));
  EXPECT_FALSE(ComesBefore(kLowestSmallCell, kLowestSmallCell));
}

// The fixed order written out directly: the low corner in widths of a size-level-15 cell by z, then y, then x, then
// the smaller size level value first. Multiplication of 64-bit integers, with no id bits read.
using FixedOrderKey = std::tuple<std::int64_t, std::int64_t, std::int64_t, int>;

FixedOrderKey ToFixedOrderKey(int size_level, CellCoord low_corner_in_cells) {
  const auto widths_per_cell = static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(15 - size_level));
  return {
      low_corner_in_cells.z * widths_per_cell,
      low_corner_in_cells.y * widths_per_cell,
      low_corner_in_cells.x * widths_per_cell,
      size_level
  };
}

// A coordinate from -kMaxCoord to kMaxCoord + 1, or near one end of that range half of the time, so ties of the low
// corner at the limits come up.
std::int32_t ToRandomCoord(std::uint64_t raw) {
  const std::uint64_t span = 2 * static_cast<std::uint64_t>(kMaxCoord) + 2;
  const std::int32_t coord = static_cast<std::int32_t>(ToUniformIndex(raw, span)) - kMaxCoord;
  switch (ToUniformIndex(raw >> 40U, 4)) {
    case 0:
      return -kMaxCoord + static_cast<std::int32_t>(ToUniformIndex(raw >> 20U, 8));
    case 1:
      return kMaxCoord + 1 - static_cast<std::int32_t>(ToUniformIndex(raw >> 20U, 8));
    default:
      return coord;
  }
}

struct RandomCell {
  int size_level;
  CellCoord cell_coord;
};

RandomCell ToRandomCell(std::mt19937_64& generator) {
  const auto size_level = static_cast<int>(ToUniformIndex(generator(), kMaxSizeLevels));
  const std::int32_t x = ToRandomCoord(generator());
  const std::int32_t y = ToRandomCoord(generator());
  const std::int32_t z = ToRandomCoord(generator());
  return {.size_level = size_level, .cell_coord = {.x = x, .y = y, .z = z}};
}

TEST(ComesBefore, RandomCells_MatchTheDirectComparison) {
  // A fixed seed: the test checks the same pairs on every run.
  std::mt19937_64 generator(20261114);  // NOLINT(bugprone-random-generator-seed)
  int checked_pair_count = 0;
  for (int pair = 0; pair < 200000; ++pair) {
    const RandomCell first = ToRandomCell(generator);
    RandomCell second = ToRandomCell(generator);
    if (pair % 4 == 0) {
      // A quarter of the pairs share the low corner: the same cell, or a cell of a smaller size level. A low corner
      // that the smaller size level cannot name is skipped.
      const auto size_level_steps = static_cast<int>(ToUniformIndex(generator(), 16 - first.size_level));
      const auto cells_per_cell =
          static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(size_level_steps));
      const std::int64_t x = first.cell_coord.x * cells_per_cell;
      const std::int64_t y = first.cell_coord.y * cells_per_cell;
      const std::int64_t z = first.cell_coord.z * cells_per_cell;
      const auto is_in_range = [](std::int64_t coord) {
        return coord >= -kMaxCoord && coord <= kMaxCoord;
      };
      if (!is_in_range(x) || !is_in_range(y) || !is_in_range(z)) {
        continue;
      }
      second.size_level = first.size_level + size_level_steps;
      second.cell_coord = {
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
          .z = static_cast<std::int32_t>(z),
      };
    }
    const CellId one_cell = ToCellId(SizeLevel{static_cast<std::uint8_t>(first.size_level)}, first.cell_coord);
    const CellId other_cell = ToCellId(SizeLevel{static_cast<std::uint8_t>(second.size_level)}, second.cell_coord);
    const FixedOrderKey first_key = ToFixedOrderKey(first.size_level, first.cell_coord);
    const FixedOrderKey second_key = ToFixedOrderKey(second.size_level, second.cell_coord);
    ASSERT_EQ(ComesBefore(one_cell, other_cell), first_key < second_key) << pair;
    ASSERT_EQ(ComesBefore(other_cell, one_cell), second_key < first_key) << pair;
    ++checked_pair_count;
  }
  EXPECT_GT(checked_pair_count, 150000);
}

// The low corner of the block of a cell, in cells of its size level: four times the floor of a quarter.
CellCoord ToBlockLowCorner(CellCoord cell_coord) {
  const auto low_corner = [](std::int32_t coord) {
    std::int32_t quarter = coord / 4;
    // Division of a negative coordinate rounds toward zero; the floor is one lower when there is a remainder.
    if (coord < 0 && coord % 4 != 0) {
      --quarter;
    }
    return 4 * quarter;
  };
  return {.x = low_corner(cell_coord.x), .y = low_corner(cell_coord.y), .z = low_corner(cell_coord.z)};
}

TEST(ComesBefore, RandomBlocks_MatchTheDirectComparison) {
  // A fixed seed: the test checks the same pairs on every run.
  std::mt19937_64 generator(20261115);  // NOLINT(bugprone-random-generator-seed)
  for (int pair = 0; pair < 200000; ++pair) {
    const RandomCell first = ToRandomCell(generator);
    const RandomCell second = ToRandomCell(generator);
    const CellBlockId one_block =
        ToCellBlockId(ToCellId(SizeLevel{static_cast<std::uint8_t>(first.size_level)}, first.cell_coord));
    const CellBlockId other_block =
        ToCellBlockId(ToCellId(SizeLevel{static_cast<std::uint8_t>(second.size_level)}, second.cell_coord));
    const FixedOrderKey first_key = ToFixedOrderKey(first.size_level, ToBlockLowCorner(first.cell_coord));
    const FixedOrderKey second_key = ToFixedOrderKey(second.size_level, ToBlockLowCorner(second.cell_coord));
    ASSERT_EQ(ComesBefore(one_block, other_block), first_key < second_key) << pair;
    ASSERT_EQ(ComesBefore(other_block, one_block), second_key < first_key) << pair;
    // Two blocks are equal in the order only when they are the same block.
    ASSERT_EQ(first_key == second_key, one_block.value == other_block.value) << pair;
  }
}

static_assert(ComputeIsCoordInRange({.x = kMaxCoord, .y = -kMaxCoord, .z = 0}));
static_assert(!ComputeIsCoordInRange({.x = kMaxCoord + 1, .y = 0, .z = 0}));

TEST(ComputeIsCoordInRange, OnePastEitherLimit_IsOutside) {
  EXPECT_TRUE(ComputeIsCoordInRange({.x = kMaxCoord, .y = -kMaxCoord, .z = 0}));
  EXPECT_TRUE(ComputeIsCoordInRange({.x = -kMaxCoord, .y = kMaxCoord, .z = -kMaxCoord}));
  EXPECT_FALSE(ComputeIsCoordInRange({.x = kMaxCoord + 1, .y = 0, .z = 0}));
  EXPECT_FALSE(ComputeIsCoordInRange({.x = 0, .y = -kMaxCoord - 1, .z = 0}));
  EXPECT_FALSE(ComputeIsCoordInRange({.x = 0, .y = 0, .z = -kMaxCoord - 1}));
}

static_assert(ToCellBlockIndex(CellPlaceIndex{130}).value == 2);
static_assert(ToCellBlockIndex(FacePlaceIndex{479}).value == 1);
static_assert(ToCellBlockIndex(CornerPlaceIndex{249}).value == 1);

TEST(ToCellBlockIndex, PlaceIndexes_DivideByPlacesPerBlock) {
  EXPECT_EQ(ToCellBlockIndex(CellPlaceIndex{130}).value, 2U);
  EXPECT_EQ(ToCellBlockIndex(FacePlaceIndex{479}).value, 1U);
  EXPECT_EQ(ToCellBlockIndex(CellPlaceIndex{63}).value, 0U);
  EXPECT_EQ(ToCellBlockIndex(FacePlaceIndex{480}).value, 2U);
  EXPECT_EQ(ToCellBlockIndex(CornerPlaceIndex{124}).value, 0U);
  EXPECT_EQ(ToCellBlockIndex(CornerPlaceIndex{249}).value, 1U);
  EXPECT_EQ(ToCellBlockIndex(CornerPlaceIndex{250}).value, 2U);
}

static_assert(ToCellPlaceIndex(CellBlockIndex{3}, {.x = 5, .y = -1, .z = 2}).value == 237);

// Local coordinates by the test's own floor: {5, -1, 2} lies in block coordinate (1, -1, 0) at local (1, 3, 2), so in
// block 3 its place is 192 + (4 x 2 + 3) x 4 + 1 = 237. The lowest coordinate -kMaxCoord = -262,143 is local 1, and
// kMaxCoord is local 3. The cells {-kMaxCoord, kMaxCoord, 0} and {kMaxCoord, -kMaxCoord, 3} lie in two different
// blocks; given block index 0 for each, as if each block were the first, they are at (4 x 0 + 3) x 4 + 1 = 13 and
// (4 x 3 + 1) x 4 + 3 = 55.
TEST(ToCellPlaceIndex, CellCoord_GivesLocalPlaceInItsBlock) {
  EXPECT_EQ(ToCellPlaceIndex(CellBlockIndex{3}, {.x = 5, .y = -1, .z = 2}).value, 237U);
  EXPECT_EQ(ToCellPlaceIndex(CellBlockIndex{0}, {.x = -kMaxCoord, .y = kMaxCoord, .z = 0}).value, 13U);
  EXPECT_EQ(ToCellPlaceIndex(CellBlockIndex{0}, {.x = kMaxCoord, .y = -kMaxCoord, .z = 3}).value, 55U);
  EXPECT_EQ(ToCellPlaceIndex(CellBlockIndex{1}, {.x = 0, .y = 0, .z = 0}).value, 64U);
}

static_assert(std::is_trivially_copyable_v<CellBlockRange>);
static_assert(sizeof(CellBlockRange) == 8);

TEST(CellBlockRange, IsTriviallyCopyable) {
  EXPECT_TRUE(std::is_trivially_copyable_v<CellBlockRange>);
  EXPECT_EQ(sizeof(CellBlockRange), 8U);
}

}  // namespace
}  // namespace fluidity::grid
