#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/border_block_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// One mark per block of a grid: 1 for a block read, 0 for a block not read.
using BlockMarks = std::vector<std::uint8_t>;

// The blocks read in every grid of the set `built`, found from the geometry once per program for each set.
const std::vector<GeometricReadBlocks>& FindGeometricReadBlocks(const RandomGridsWithCheckers& built) {
  static std::map<const RandomGridsWithCheckers*, std::vector<GeometricReadBlocks>> read_blocks_of_sets;
  std::vector<GeometricReadBlocks>& found = read_blocks_of_sets[&built];
  if (found.empty()) {
    found.reserve(built.grids.size());
    for (std::size_t grid = 0; grid < built.grids.size(); ++grid) {
      found.push_back(BuildGeometricReadBlocks(built.checkers.at(grid), built.grids.at(grid).grid.tables(RunOn::kCpu)));
    }
  }
  return found;
}

// Checks that the border of each set of grid `grid` of `built` at `width` is the one FindBorderGeometrically finds.
void CheckBordersEqualGeometricBorders(const RandomGridsWithCheckers& built, std::size_t grid, BorderWidth width) {
  const GridBorderCase& grid_case = FindGridBorderCase(built, grid);
  for (std::size_t set_number = 0; set_number < grid_case.sets.size(); ++set_number) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set_number);
    const std::vector<CellBlockIndex>& border = width == BorderWidth::kTouchingCells
        ? grid_case.touching_borders.at(set_number)
        : grid_case.wide_borders.at(set_number);
    const std::vector<CellBlockIndex> geometric_border =
        FindBorderGeometrically(FindGeometricReadBlocks(built).at(grid), grid_case.sets.at(set_number), width);
    EXPECT_EQ(ToBlockNumbers(border), ToBlockNumbers(geometric_border));
  }
}

TEST(FindBorderBlocks, TouchingCells_EqualsGeometricReadSet) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    CheckBordersEqualGeometricBorders(built, grid, BorderWidth::kTouchingCells);
  }
}

TEST(FindBorderBlocks, FourCellWidths_EqualsGeometricReadSet) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    CheckBordersEqualGeometricBorders(built, grid, BorderWidth::kFourCellWidths);
  }
}

// The same on the framed random grids, whose cells lie at negative and distant coordinates in other frames.
TEST(FindBorderBlocks, FramedRandomGridsAtBothWidths_EqualGeometricReadSet) {
  const RandomGridsWithCheckers& built = FindFramedRandomGridsWithCheckers();
  ASSERT_EQ(built.grids.size(), kFramedRandomGridCount);
  for (std::size_t grid = 0; grid < built.grids.size(); ++grid) {
    CheckBordersEqualGeometricBorders(built, grid, BorderWidth::kTouchingCells);
    CheckBordersEqualGeometricBorders(built, grid, BorderWidth::kFourCellWidths);
  }
}

// The borders of one block alone at kTouchingCells and kFourCellWidths, by the forward form: the marks of its cell
// places, which do not depend on the width, are made once and shared by both widths.
std::array<BlockNumbers, 2> FindBlockBordersByReadMarks(const GridTables& tables, std::uint32_t block) {
  BlockMarks cell_marks(tables.block_count, 0);
  for (std::uint32_t place_in_block = 0; place_in_block < kCellPlacesPerBlock; ++place_in_block) {
    const CellPlaceIndex cell = {.value = (block * kCellPlacesPerBlock) + place_in_block};
    ApplyCellPlaceReadMarks(tables, cell, BorderWidth::kTouchingCells, cell_marks.data());
  }
  std::array<BlockNumbers, 2> borders;
  for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
    BlockMarks marks = cell_marks;
    for (std::uint32_t place_in_block = 0; place_in_block < kFacePlacesPerBlock; ++place_in_block) {
      const FacePlaceIndex face = {.value = (block * kFacePlacesPerBlock) + place_in_block};
      ApplyFacePlaceReadMarks(tables, face, width, marks.data());
    }
    BlockNumbers& border = borders.at(width == BorderWidth::kTouchingCells ? 0 : 1);
    for (std::uint32_t marked = 0; marked < tables.block_count; ++marked) {
      if (marks.at(marked) != 0 && marked != block) {
        border.insert(marked);
      }
    }
  }
  return borders;
}

// Checks, for every block of grid `grid` of `built` alone, that its border at each width, by the forward form, is the
// set of blocks it reads by the geometry (FindBorderGeometrically).
void CheckEveryBlockAloneEqualsGeometricBorder(const RandomGridsWithCheckers& built, std::size_t grid) {
  const GridTables tables = built.grids.at(grid).grid.tables(RunOn::kCpu);
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", block " << block);
    const std::vector<CellBlockIndex> set = {{.value = block}};
    const std::array<BlockNumbers, 2> borders = FindBlockBordersByReadMarks(tables, block);
    const GeometricReadBlocks& read_blocks = FindGeometricReadBlocks(built).at(grid);
    EXPECT_EQ(borders.at(0), ToBlockNumbers(FindBorderGeometrically(read_blocks, set, BorderWidth::kTouchingCells)));
    EXPECT_EQ(borders.at(1), ToBlockNumbers(FindBorderGeometrically(read_blocks, set, BorderWidth::kFourCellWidths)));
  }
}

// The border of each block alone, found forward from the places its readers read (ApplyCellPlaceReadMarks and
// ApplyFacePlaceReadMarks), is the set of blocks it reads by the geometry, for every block of every random grid at
// both widths. The border of a set is the union of the borders of its blocks with their bigger blocks, less the set,
// in both computations, so this covers every set; BlockByBlockForm_EqualsForwardForm ties the other form to it.
TEST(FindBorderBlocks, EveryBlockAlone_EqualsGeometricReadSet) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_EQ(FindGeometricReadBlocks(built).size(), kRandomGridCount);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    CheckEveryBlockAloneEqualsGeometricBorder(built, grid);
  }
}

// The same on every block of the framed random grids.
TEST(FindBorderBlocks, EveryBlockAloneOfFramedRandomGrids_EqualsGeometricReadSet) {
  const RandomGridsWithCheckers& built = FindFramedRandomGridsWithCheckers();
  ASSERT_EQ(FindGeometricReadBlocks(built).size(), kFramedRandomGridCount);
  for (std::size_t grid = 0; grid < built.grids.size(); ++grid) {
    CheckEveryBlockAloneEqualsGeometricBorder(built, grid);
  }
}

// Finds a few face pieces kept in the blocks of a set: about kFacePiecesPerSet, evenly spread over the list of all.
std::vector<FacePlaceIndex> FindSpreadFacePiecesKeptIn(const GridTables& tables, const BlockNumbers& set) {
  constexpr std::size_t kFacePiecesPerSet = 6;
  std::vector<FacePlaceIndex> face_pieces;
  for (const std::uint32_t block : set) {
    for (std::uint32_t place_in_block = 0; place_in_block < kFacePlacesPerBlock; ++place_in_block) {
      const FacePlaceIndex face = {.value = (block * kFacePlacesPerBlock) + place_in_block};
      if (tables.face_piece_kinds[face.value] != FacePieceKind::kNotAPiece) {
        face_pieces.push_back(face);
      }
    }
  }
  const std::size_t step = std::max<std::size_t>(1, face_pieces.size() / kFacePiecesPerSet);
  std::vector<FacePlaceIndex> spread_face_pieces;
  spread_face_pieces.reserve((face_pieces.size() / step) + 1);
  for (std::size_t entry = 0; entry < face_pieces.size(); entry += step) {
    spread_face_pieces.push_back(face_pieces.at(entry));
  }
  return spread_face_pieces;
}

// Checks that every face piece of a size level within one of the start's whose centre lies within 4 widths of `face`
// in grid `grid` of `built` is kept in a block of `held_blocks`.
void CheckFacesWithinFourWidthsAreHeld(
    const RandomGridsWithCheckers& built,
    std::size_t grid,
    FacePlaceIndex face,
    const BlockNumbers& held_blocks
) {
  const GridTables tables = built.grids.at(grid).grid.tables(RunOn::kCpu);
  const int start_size_level = static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(face)));
  for (const GeometricFacePiece& near_piece : built.checkers.at(grid).FacesWithin(ToFacePieceId(tables, face), 4.0)) {
    // A face search finds face pieces of the size levels one below, equal to and one above the start's only.
    const int size_level_difference = static_cast<int>(near_piece.size_level) - start_size_level;
    if (size_level_difference < -1 || size_level_difference > 1) {
      continue;
    }
    const FacePlaceIndex near_place = FindFacePlace(tables, near_piece.id);
    ASSERT_NE(near_place.value, kNoIndex);
    const std::uint32_t near_block = ToCellBlockIndex(near_place).value;
    EXPECT_TRUE(held_blocks.contains(near_block))
        << "face " << face.value << " has a face within 4 widths in block " << near_block;
  }
}

// For a few face pieces kept in each set (FacesWithin looks at every face piece of the grid in each call), every face
// within 4 widths is kept in the set or its border.
void CheckFourCellWidthsHoldEveryFaceWithinFourWidths(const RandomGridsWithCheckers& built) {
  for (std::size_t grid = 0; grid < built.grids.size(); ++grid) {
    const GridTables tables = built.grids.at(grid).grid.tables(RunOn::kCpu);
    const GridBorderCase& grid_case = FindGridBorderCase(built, grid);
    for (std::size_t set_number = 0; set_number < grid_case.sets.size(); ++set_number) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set_number);
      const BlockNumbers set = ToBlockNumbers(grid_case.sets.at(set_number));
      BlockNumbers held_blocks = ToBlockNumbers(grid_case.wide_borders.at(set_number));
      held_blocks.insert(set.begin(), set.end());
      for (const FacePlaceIndex face : FindSpreadFacePiecesKeptIn(tables, set)) {
        CheckFacesWithinFourWidthsAreHeld(built, grid, face, held_blocks);
      }
    }
  }
}

TEST(FindBorderBlocks, FourCellWidths_HoldsEveryFaceWithinFourWidths) {
  CheckFourCellWidthsHoldEveryFaceWithinFourWidths(FindRandomGridsWithCheckers());
}

TEST(FindBorderBlocks, FramedRandomGridsAtFourCellWidths_HoldEveryFaceWithinFourWidths) {
  CheckFourCellWidthsHoldEveryFaceWithinFourWidths(FindFramedRandomGridsWithCheckers());
}

}  // namespace
}  // namespace fluidity::grid
