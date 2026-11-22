#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>

#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

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

}  // namespace
}  // namespace fluidity::grid
