#include "fluidity/grid/border_block_steps.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

// The tables of the plane-four grid, written by hand: one size level, cells 1 m wide, the active cells {3,3,0} in
// block 0 at (0,0,0), {4,3,0} in block 1 at (1,0,0) and {6,6,0} in block 2 at (1,1,0).
struct PlaneFourTables {
  std::array<CellBlockId, 3> block_ids;
  std::vector<std::uint32_t> same_size_level_blocks = std::vector<std::uint32_t>(std::size_t{3} * 27, kNoIndex);
  std::vector<std::uint32_t> bigger_block = std::vector<std::uint32_t>(3, kNoIndex);
  std::vector<std::uint32_t> smaller_blocks = std::vector<std::uint32_t>(std::size_t{3} * 8, kNoIndex);
  std::vector<CellState> cell_states = std::vector<CellState>(std::size_t{3} * 64, CellState::kAbsent);
  std::vector<FacePieceKind> face_piece_kinds =
      std::vector<FacePieceKind>(std::size_t{3} * 240, FacePieceKind::kNotAPiece);
};

// The entry of a block's 27 neighbours for an offset of one block along x, y or z: 13 is the block itself.
constexpr std::size_t kEntryOfBlockItself = 13;
constexpr std::size_t kEntryAlongX = 14;
constexpr std::size_t kEntryAlongY = 16;
constexpr std::size_t kEntryAlongXAndY = 17;

PlaneFourTables BuildPlaneFourTables() {
  PlaneFourTables owner = {
      .block_ids = {
          ToCellBlockId(ToCell(0, 3, 3, 0)),
          ToCellBlockId(ToCell(0, 4, 3, 0)),
          ToCellBlockId(ToCell(0, 6, 6, 0)),
      },
  };
  // Each block is its own neighbour, and its neighbour on the high side of each axis is the next one named.
  for (std::size_t block = 0; block < 3; ++block) {
    owner.same_size_level_blocks.at((block * 27) + kEntryOfBlockItself) = static_cast<std::uint32_t>(block);
  }
  const auto make_neighbours = [&owner](std::size_t low_block, std::size_t high_block, std::size_t entry_toward_high) {
    // The entry of the opposite offset is 26 minus the entry.
    owner.same_size_level_blocks.at((low_block * 27) + entry_toward_high) = static_cast<std::uint32_t>(high_block);
    owner.same_size_level_blocks.at((high_block * 27) + (26 - entry_toward_high)) =
        static_cast<std::uint32_t>(low_block);
  };
  make_neighbours(0, 1, kEntryAlongX);
  make_neighbours(1, 2, kEntryAlongY);
  make_neighbours(0, 2, kEntryAlongXAndY);
  // The active cells: {3,3,0} is local (3,3,0) of block 0, {4,3,0} is local (0,3,0) of block 1 and {6,6,0} is local
  // (2,2,0) of block 2. A cell place is (z * 4 + y) * 4 + x.
  owner.cell_states.at((0 * 64) + 15) = CellState::kActive;
  owner.cell_states.at((1 * 64) + 12) = CellState::kActive;
  owner.cell_states.at((2 * 64) + 10) = CellState::kActive;
  // The face pieces of {4,3,0} that matter. X places are (z * 4 + y) * 5 + x, Y places 80 + (z * 5 + y) * 4 + x.
  // Its -X side is plane 0 of block 1, between it and {3,3,0}; plane 4 of block 0 is no piece, as block 1 exists.
  const std::size_t block_1_faces = std::size_t{1} * 240;
  const std::size_t block_2_faces = std::size_t{2} * 240;
  owner.face_piece_kinds.at(block_1_faces + 15) = FacePieceKind::kBetweenCells;
  owner.face_piece_kinds.at(block_1_faces + 16) = FacePieceKind::kGridEdgeHigh;
  owner.face_piece_kinds.at(block_1_faces + 80 + 12) = FacePieceKind::kGridEdgeLow;
  // Its +Y side is plane 4 of block 1, which is no piece because block 2 exists, and plane 0 of block 2, local x 0.
  owner.face_piece_kinds.at(block_2_faces + 80 + 0) = FacePieceKind::kGridEdgeHigh;
  // The cell {3,3,0} has pieces of its own in block 0.
  owner.face_piece_kinds.at(0 * 240 + 18) = FacePieceKind::kGridEdgeLow;
  owner.face_piece_kinds.at(0 * 240 + 80 + 15) = FacePieceKind::kGridEdgeLow;
  owner.face_piece_kinds.at(0 * 240 + 80 + 19) = FacePieceKind::kGridEdgeHigh;
  return owner;
}

GridTables ToTables(const PlaneFourTables& owner) {
  return {
      .frame = ToUnitFrame(1),
      .version = {.value = 1},
      .block_count = 3,
      .block_ids = owner.block_ids.data(),
      .same_size_level_blocks = owner.same_size_level_blocks.data(),
      .bigger_block = owner.bigger_block.data(),
      .smaller_blocks = owner.smaller_blocks.data(),
      .cell_states = owner.cell_states.data(),
      .face_piece_kinds = owner.face_piece_kinds.data(),
  };
}

TEST(ComputeIsInBlockSet, RangeOrSortedList_IsFound) {
  const std::array<CellBlockIndex, 2> sorted_blocks = {CellBlockIndex{7}, CellBlockIndex{9}};
  const BlockSet set = {
      .range = {.first_block = 2, .block_count = 3},
      .sorted_blocks = sorted_blocks.data(),
      .sorted_count = 2,
  };
  for (const std::uint32_t block : {2U, 3U, 4U, 7U, 9U}) {
    EXPECT_TRUE(ComputeIsInBlockSet(set, {.value = block})) << block;
  }
  for (const std::uint32_t block : {0U, 1U, 5U, 6U, 8U, 10U}) {
    EXPECT_FALSE(ComputeIsInBlockSet(set, {.value = block})) << block;
  }
}

TEST(ComputeIsInBlockSet, EmptySet_HoldsNothing) {
  const BlockSet set = {.range = {.first_block = 0, .block_count = 0}, .sorted_blocks = nullptr, .sorted_count = 0};
  EXPECT_FALSE(ComputeIsInBlockSet(set, {.value = 0}));
}

TEST(ComputeIsBorderBlock, PlaneFourFixture_MarksBlockC) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  const BlockSet set = {.range = {.first_block = 0, .block_count = 1}, .sorted_blocks = nullptr, .sorted_count = 0};
  EXPECT_FALSE(ComputeIsBorderBlock(tables, set, {.value = 0}, BorderWidth::kTouchingCells));
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 1}, BorderWidth::kTouchingCells));
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 2}, BorderWidth::kTouchingCells));
}

TEST(ComputeIsBorderBlock, SetOfTheLastBlock_MarksTheBlocksOfTheCellsNearItsFacePiece) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  const BlockSet set = {.range = {.first_block = 2, .block_count = 1}, .sorted_blocks = nullptr, .sorted_count = 0};
  // Block 2 keeps the +Y face piece of {4,3,0}, whose cell_low is in block 1. The cell {3,3,0} of block 0 touches that
  // cell, so its particles can list the face piece: transfer reads block 0.
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 0}, BorderWidth::kTouchingCells));
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 1}, BorderWidth::kTouchingCells));
  EXPECT_FALSE(ComputeIsBorderBlock(tables, set, {.value = 2}, BorderWidth::kTouchingCells));
}

TEST(ApplyCellPlaceReadMarks, CellOfBlockA_MarksEveryBlock) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  // {3,3,0} touches {4,3,0} of block 1, whose +Y face piece is plane 0 of block 2.
  ApplyCellPlaceReadMarks(tables, {.value = 15}, BorderWidth::kTouchingCells, marks.data());
  EXPECT_EQ(marks, (std::array<std::uint8_t, 3>{1, 1, 1}));
}

TEST(ApplyCellPlaceReadMarks, PlaceOfNoCell_MarksOnlyItsBlock) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  ApplyCellPlaceReadMarks(tables, {.value = 64 + 0}, BorderWidth::kTouchingCells, marks.data());
  EXPECT_EQ(marks, (std::array<std::uint8_t, 3>{0, 1, 0}));
}

TEST(ApplyFacePlaceReadMarks, FacePieceOfBlockC_MarksTheBlocksOfTheCellsNearIt) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  // The +Y face piece of {4,3,0}: its cell_low is in block 1, and {3,3,0} of block 0 touches that cell.
  ApplyFacePlaceReadMarks(tables, {.value = (2 * 240) + 80}, BorderWidth::kTouchingCells, marks.data());
  EXPECT_EQ(marks, (std::array<std::uint8_t, 3>{1, 1, 1}));
}

TEST(ApplyFacePlaceReadMarks, PlaceOfNoFacePiece_MarksNothing) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  ApplyFacePlaceReadMarks(tables, {.value = (2 * 240) + 81}, BorderWidth::kFourCellWidths, marks.data());
  EXPECT_EQ(marks, (std::array<std::uint8_t, 3>{0, 0, 0}));
}

TEST(ComputeIsBorderBlock, SetOfTheLastBlock_HasBothOthersAtFourCellWidths) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  const std::array<CellBlockIndex, 1> sorted_blocks = {CellBlockIndex{2}};
  const BlockSet set = {
      .range = {.first_block = 0, .block_count = 0},
      .sorted_blocks = sorted_blocks.data(),
      .sorted_count = 1,
  };
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 0}, BorderWidth::kFourCellWidths));
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 1}, BorderWidth::kFourCellWidths));
  EXPECT_FALSE(ComputeIsBorderBlock(tables, set, {.value = 2}, BorderWidth::kFourCellWidths));
}

}  // namespace
}  // namespace fluidity::grid
