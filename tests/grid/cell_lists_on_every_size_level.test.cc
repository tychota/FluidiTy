#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_build_steps_cases.h"

namespace fluidity::grid {
namespace {

// The header's bound: biased < 2^34, so passes 1 and 2 are below 2^34 and pass 0, biased x 16 + 15, below 2^38.
void CheckKeysBelowProvedBounds(CellId cell) {
  EXPECT_LT(ToCellSortKey(cell, 0), kTwoToThe38);
  EXPECT_LT(ToCellSortKey(cell, 1), kTwoToThe34);
  EXPECT_LT(ToCellSortKey(cell, 2), kTwoToThe34);
  EXPECT_LT(ToBlockSortKey(cell, 0), kTwoToThe38);
  EXPECT_LT(ToBlockSortKey(cell, 1), kTwoToThe34);
  EXPECT_LT(ToBlockSortKey(cell, 2), kTwoToThe34);
}

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

TEST(ToCellSortKey, EverySizeLevelAtCoordLimits_StaysBelowTheProvedBounds) {
  for (const CellId cell : BuildEverySizeLevelAtCoordLimits()) {
    CheckKeysBelowProvedBounds(cell);
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

}  // namespace
}  // namespace fluidity::grid
