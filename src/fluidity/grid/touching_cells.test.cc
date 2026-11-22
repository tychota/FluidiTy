#include "fluidity/grid/touching_cells.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <map>
#include <span>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/touching_cells_cases.h"

namespace fluidity::grid {
namespace {

// A frame with its origin at 0 and biggest cells 1 m wide: every width and area in it is a power of two, so the tests
// compare sums of areas exactly.
constexpr GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// The inputs of a CPU list of version 1.
GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

// Appends the 8 small cells of `big_cell`, in cell eighth order.
void ApplyAppendSmallCells(CellId big_cell, std::vector<CellId>& cells) {
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
}

// A default-made result is an empty list whose every entry is kNoIndex, as a reader's entries past count are.
static_assert(TouchingCells().count == 0 && TouchingCells().cells[0].value == kNoIndex);

static_assert(TouchingCells().cells[kMaxTouchingCells - 1].value == kNoIndex);

static_assert(CellFacePieces().count == 0 && CellFacePieces().faces[0].value == kNoIndex);

static_assert(CellFacePieces().faces[kMaxCellFacePieces - 1].value == kNoIndex);

// The places of the two-size fixture (2 size levels): block 0 is the size-level-0 block (0,0,0), block 1 the
// size-level-1 block (0,0,0).
GridTables ToTablesOf(const std::expected<TestGrid, GridFailure>& test_grid) {
  return test_grid->grid.tables(RunOn::kCpu);
}

// The worked example of FindTouchingCells: the big cell, place 0, touches the four size-level-1 cells with x = 2, at
// local (2, y, z) of block 1, and no other cell.
TEST(FindTouchingCells, TwoSizeFixture_BigCellTouchesFour) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const TouchingCells touching = FindTouchingCells(ToTablesOf(test_grid), CellPlaceIndex{0});
  const std::vector<std::uint32_t> expected_places = {64 + 2, 64 + 4 + 2, 64 + 16 + 2, 64 + 20 + 2};
  EXPECT_EQ(ToListedPlaces(touching.cells, touching.count), expected_places);
  EXPECT_TRUE(ComputeHasNoIndexPastCount(touching.cells, touching.count));
}

// The worked example of FindFacePieces: the big cell's 5 whole sides, kept in block 0, and the 4 quarters of its +X
// side, kept at size level 1 at X places 2, 2 + 5, 2 + 20 and 2 + 25 of block 1.
TEST(FindFacePieces, TwoSizeFixture_BigCellHasNine) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const CellFacePieces face_pieces = FindFacePieces(ToTablesOf(test_grid), CellPlaceIndex{0});
  // -X at X place 0, -Y and +Y at Y places 80 and 80 + 4, -Z and +Z at Z places 160 and 160 + 16.
  const std::vector<std::uint32_t> expected_places = {
      0,
      80,
      80 + 4,
      160,
      160 + 16,
      240 + 2,
      240 + 2 + 5,
      240 + 2 + 20,
      240 + 2 + 25,
  };
  EXPECT_EQ(ToListedPlaces(face_pieces.faces, face_pieces.count), expected_places);
  EXPECT_TRUE(ComputeHasNoIndexPastCount(face_pieces.faces, face_pieces.count));
}

// The size-level-0 cell {0,0,0} with each of the 26 size-level-0 cells around it split, 2 size levels: 1 + 26 x 8 cells
// across 8 blocks of each size level, so the places around the cell lie in every neighbour direction of its block.
std::vector<CellId> BuildCellWithEveryNeighbourSplit() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (std::int32_t z = -1; z <= 1; ++z) {
    for (std::int32_t y = -1; y <= 1; ++y) {
      for (std::int32_t x = -1; x <= 1; ++x) {
        if (x != 0 || y != 0 || z != 0) {
          ApplyAppendSmallCells(ToCell(0, x, y, z), cells);
        }
      }
    }
  }
  return cells;
}

// The bound of kMaxTouchingCells is reached: 6 sides of 4 small cells, 12 edges of 2, 8 corners of 1, the checker's
// cells in ascending index order.
TEST(FindTouchingCells, AllNeighboursSplit_Has56) {
  const std::vector<CellId> cells = BuildCellWithEveryNeighbourSplit();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const GeometricGrid checker(ToUnitFrame(2), cells);
  const CellPlaceIndex cell = FindCellPlace(tables, ToCell(0, 0, 0, 0));
  const TouchingCells touching = FindTouchingCells(tables, cell);
  EXPECT_EQ(touching.count, kMaxTouchingCells);
  EXPECT_EQ(ToListedPlaces(touching.cells, touching.count), ToCheckerTouchingPlaces(tables, checker, cells.at(0)));
}

// The size-level-0 cell {0,0,0} with its 6 side neighbours split, 2 size levels: 1 + 6 x 8 cells.
std::vector<CellId> BuildCellWithEverySideNeighbourSplit() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int axis = 0; axis < 3; ++axis) {
    for (const std::int32_t side_step : {-1, 1}) {
      const std::int32_t x = axis == 0 ? side_step : 0;
      const std::int32_t y = axis == 1 ? side_step : 0;
      const std::int32_t z = axis == 2 ? side_step : 0;
      ApplyAppendSmallCells(ToCell(0, x, y, z), cells);
    }
  }
  return cells;
}

// The bound of kMaxCellFacePieces is reached: each of the 6 sides is 4 quarters of 0.25 m2, the checker's face pieces
// in ascending index order. The low sides' quarters are the high sides of small cells at local 3 of their block, kept
// at plane 0 of the block above.
TEST(FindFacePieces, AllSidesBesideSplitCells_Has24) {
  const std::vector<CellId> cells = BuildCellWithEverySideNeighbourSplit();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const GeometricGrid checker(ToUnitFrame(2), cells);
  const CellPlaceIndex cell = FindCellPlace(tables, ToCell(0, 0, 0, 0));
  const CellFacePieces face_pieces = FindFacePieces(tables, cell);
  EXPECT_EQ(face_pieces.count, kMaxCellFacePieces);
  const std::vector<FacePieceId> checker_pieces = ToCheckerFacePiecesByCell(checker).at(cells.at(0).value);
  EXPECT_EQ(ToListedPlaces(face_pieces.faces, face_pieces.count), ToAscendingFacePlaces(tables, checker_pieces));
  EXPECT_EQ(ComputeCellSurfaceSums(tables, cell, face_pieces).side_areas, (std::array<double, 6>{1, 1, 1, 1, 1, 1}));
}

// A copy of the used part of a grid's tables, which a test may change: to stand for a grid piece that did not load some
// blocks, or for tables that contradict themselves.
struct CopiedTables {
  GridTables tables;
  std::vector<CellBlockId> block_ids;
  std::vector<std::uint32_t> same_size_level_blocks;
  std::vector<std::uint32_t> bigger_block;
  std::vector<std::uint32_t> smaller_blocks;
  std::vector<CellState> cell_states;
  std::vector<FacePieceKind> face_piece_kinds;
};

// Copies the used part of `tables` and points the copy's GridTables at the copied arrays.
void CopyTables(const GridTables& tables, CopiedTables& copy) {
  const std::size_t block_count = tables.block_count;
  copy.block_ids.assign(tables.block_ids, tables.block_ids + block_count);
  copy.same_size_level_blocks.assign(tables.same_size_level_blocks, tables.same_size_level_blocks + (27 * block_count));
  copy.bigger_block.assign(tables.bigger_block, tables.bigger_block + block_count);
  copy.smaller_blocks.assign(tables.smaller_blocks, tables.smaller_blocks + (8 * block_count));
  copy.cell_states.assign(tables.cell_states, tables.cell_states + (kCellPlacesPerBlock * block_count));
  copy.face_piece_kinds.assign(tables.face_piece_kinds, tables.face_piece_kinds + (kFacePlacesPerBlock * block_count));
  copy.tables = tables;
  copy.tables.block_ids = copy.block_ids.data();
  copy.tables.same_size_level_blocks = copy.same_size_level_blocks.data();
  copy.tables.bigger_block = copy.bigger_block.data();
  copy.tables.smaller_blocks = copy.smaller_blocks.data();
  copy.tables.cell_states = copy.cell_states.data();
  copy.tables.face_piece_kinds = copy.face_piece_kinds.data();
}

// In a grid piece, a face piece kept in a block the grid piece did not load is not listed. The cell {0,0,0} beside 6
// split cells, as a grid piece that did not load the size-level-1 block (0,0,0): the -X side's quarters are the high
// sides of the small cells at x = -1, local 3 of the size-level-1 block (-1,0,0), and kept at plane 0 of that missing
// block. In the grid piece, that block's +X neighbour is "none" and its plane 4 stays kNotAPiece, so the side gives
// none: 24 - 4 = 20.
TEST(FindFacePieces, FacePiecesInABlockNotLoaded_AreLeftOut) {
  const std::vector<CellId> cells = BuildCellWithEverySideNeighbourSplit();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  CopiedTables grid_piece;
  CopyTables(ToTablesOf(test_grid), grid_piece);
  const CellBlockIndex low_block = FindCellBlock(grid_piece.tables, ToCellBlockId(ToCell(1, -1, 0, 0)));
  ASSERT_NE(low_block.value, kNoIndex);
  constexpr std::size_t kHighXEntry = 13 + 1;
  grid_piece.same_size_level_blocks.at((27 * std::size_t{low_block.value}) + kHighXEntry) = kNoIndex;
  const CellPlaceIndex cell = FindCellPlace(grid_piece.tables, ToCell(0, 0, 0, 0));
  const CellFacePieces whole_grid_face_pieces = FindFacePieces(ToTablesOf(test_grid), cell);
  const CellFacePieces grid_piece_face_pieces = FindFacePieces(grid_piece.tables, cell);
  EXPECT_EQ(grid_piece_face_pieces.count, kMaxCellFacePieces - 4);
  // The face pieces left are the whole grid's other 20, in the same order.
  std::vector<std::uint32_t> expected_places;
  for (const std::uint32_t face_place : ToListedPlaces(whole_grid_face_pieces.faces, whole_grid_face_pieces.count)) {
    const bool is_low_x_side = ToAxis(FacePlaceIndex{face_place}) == Axis::kX
        && FindCellHigh(ToTablesOf(test_grid), FacePlaceIndex{face_place}).value == cell.value;
    if (!is_low_x_side) {
      expected_places.push_back(face_place);
    }
  }
  EXPECT_EQ(ToListedPlaces(grid_piece_face_pieces.faces, grid_piece_face_pieces.count), expected_places);
}

// Tables that contradict themselves: in the grid of AllNeighboursSplit_Has56, the small cell size level 1 {2,0,0}
// beside the cell is marked split, as the size step rule forbids. Only active small places are listed, so it is left
// out and 55 cells remain, the other 55 of the whole grid in the same order.
TEST(FindTouchingCells, SmallPlaceNotActive_IsLeftOut) {
  const std::vector<CellId> cells = BuildCellWithEveryNeighbourSplit();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  CopiedTables contradicting;
  CopyTables(ToTablesOf(test_grid), contradicting);
  const CellPlaceIndex small_place = FindCellPlace(contradicting.tables, ToCell(1, 2, 0, 0));
  ASSERT_NE(small_place.value, kNoIndex);
  contradicting.cell_states.at(small_place.value) = CellState::kSplit;
  const CellPlaceIndex cell = FindCellPlace(contradicting.tables, ToCell(0, 0, 0, 0));
  const TouchingCells whole_grid_touching = FindTouchingCells(ToTablesOf(test_grid), cell);
  const TouchingCells contradicting_touching = FindTouchingCells(contradicting.tables, cell);
  std::vector<std::uint32_t> expected_places = ToListedPlaces(whole_grid_touching.cells, whole_grid_touching.count);
  std::erase(expected_places, small_place.value);
  EXPECT_EQ(contradicting_touching.count, kMaxTouchingCells - 1);
  EXPECT_EQ(ToListedPlaces(contradicting_touching.cells, contradicting_touching.count), expected_places);
}

// Removes the smaller block of eighth 0 of the size-level-0 block (0,0,0) from copied tables, which then contradict
// themselves: its split places keep kSplit without a block for their small cells. Gives the block removed. In the grids
// around the cell {0,0,0} below, eighth 0 holds the cell and its 7 neighbours with coordinates 0 or 1.
std::uint32_t ApplySmallBlockRemoval(CopiedTables& contradicting) {
  const CellBlockIndex biggest_block = FindCellBlock(contradicting.tables, ToCellBlockId(ToCell(0, 0, 0, 0)));
  EXPECT_NE(biggest_block.value, kNoIndex);
  std::uint32_t& small_block_entry = contradicting.smaller_blocks.at(8 * std::size_t{biggest_block.value});
  const std::uint32_t removed_block = small_block_entry;
  small_block_entry = kNoIndex;
  return removed_block;
}

// A split place without its smaller block gives no small cells, and nothing outside the arrays is read. In the grid of
// AllNeighboursSplit_Has56 the 7 neighbours in eighth 0 lose their small cells next to the cell: 3 sides of 4, 3 edges
// of 2 and 1 corner, 19, so 37 of the 56 remain.
TEST(FindTouchingCells, SplitPlaceWithoutSmallBlock_GivesNoSmallCells) {
  const std::vector<CellId> cells = BuildCellWithEveryNeighbourSplit();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  CopiedTables contradicting;
  CopyTables(ToTablesOf(test_grid), contradicting);
  const std::uint32_t removed_block = ApplySmallBlockRemoval(contradicting);
  const CellPlaceIndex cell = FindCellPlace(contradicting.tables, ToCell(0, 0, 0, 0));
  const TouchingCells whole_grid_touching = FindTouchingCells(ToTablesOf(test_grid), cell);
  const TouchingCells contradicting_touching = FindTouchingCells(contradicting.tables, cell);
  std::vector<std::uint32_t> expected_places = ToListedPlaces(whole_grid_touching.cells, whole_grid_touching.count);
  std::erase_if(expected_places, [removed_block](std::uint32_t place) {
    return ToCellBlockIndex(CellPlaceIndex{place}).value == removed_block;
  });
  EXPECT_EQ(contradicting_touching.count, kMaxTouchingCells - 19);
  EXPECT_EQ(ToListedPlaces(contradicting_touching.cells, contradicting_touching.count), expected_places);
}

// A split place without its smaller block gives no face pieces, and nothing outside the arrays is read. In the grid of
// AllSidesBesideSplitCells_Has24 the cell's +X, +Y and +Z neighbours lie in eighth 0, so their 12 quarters go and the
// 12 of the low sides remain.
TEST(FindFacePieces, SplitPlaceWithoutSmallBlock_GivesNoFacePieces) {
  const std::vector<CellId> cells = BuildCellWithEverySideNeighbourSplit();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  CopiedTables contradicting;
  CopyTables(ToTablesOf(test_grid), contradicting);
  ApplySmallBlockRemoval(contradicting);
  const CellPlaceIndex cell = FindCellPlace(contradicting.tables, ToCell(0, 0, 0, 0));
  const CellFacePieces whole_grid_face_pieces = FindFacePieces(ToTablesOf(test_grid), cell);
  const CellFacePieces contradicting_face_pieces = FindFacePieces(contradicting.tables, cell);
  // The face pieces left are those of the low sides, of which the cell is cell_high.
  std::vector<std::uint32_t> expected_places;
  for (const std::uint32_t face_place : ToListedPlaces(whole_grid_face_pieces.faces, whole_grid_face_pieces.count)) {
    if (FindCellHigh(ToTablesOf(test_grid), FacePlaceIndex{face_place}).value == cell.value) {
      expected_places.push_back(face_place);
    }
  }
  EXPECT_EQ(contradicting_face_pieces.count, kMaxCellFacePieces - 12);
  EXPECT_EQ(ToListedPlaces(contradicting_face_pieces.faces, contradicting_face_pieces.count), expected_places);
}

// Only a split place across a side without a face piece gives face pieces. In the two-size fixture the big cell's +X
// neighbour, place 1, is marked kActive while the big cell's whole +X side, X place 1 of block 0, stays kNotAPiece:
// tables that contradict themselves. The +X side then gives none, and the 5 other sides remain.
TEST(FindFacePieces, ActivePlaceAcrossSideWithoutFacePiece_GivesNone) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  CopiedTables contradicting;
  CopyTables(ToTablesOf(test_grid), contradicting);
  ASSERT_EQ(contradicting.cell_states.at(1), CellState::kSplit);
  ASSERT_EQ(contradicting.face_piece_kinds.at(1), FacePieceKind::kNotAPiece);
  contradicting.cell_states.at(1) = CellState::kActive;
  const CellFacePieces face_pieces = FindFacePieces(contradicting.tables, CellPlaceIndex{0});
  const std::vector<std::uint32_t> expected_places = {0, 80, 80 + 4, 160, 160 + 16};
  EXPECT_EQ(ToListedPlaces(face_pieces.faces, face_pieces.count), expected_places);
}

// In the two-size fixture: place 1, split, beside the big cell; place 2, absent; place 64 + 1, inside the big cell,
// whose +X side is the face piece at X place 2 of block 1. Each has an active cell or a face piece next to it, so only
// the check of the place's own state gives an empty list.
constexpr std::array<CellPlaceIndex, 3> kPlacesWithoutAnActiveCell = {
    CellPlaceIndex{1},
    CellPlaceIndex{2},
    CellPlaceIndex{64 + 1},
};

// A place that holds no active cell has no touching cells, and the whole array is kNoIndex.
TEST(FindTouchingCells, PlaceWithoutAnActiveCell_HasNone) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  for (const CellPlaceIndex place : kPlacesWithoutAnActiveCell) {
    SCOPED_TRACE(testing::Message() << "place " << place.value);
    const TouchingCells touching = FindTouchingCells(ToTablesOf(test_grid), place);
    EXPECT_EQ(touching.count, 0);
    EXPECT_TRUE(ComputeHasNoIndexPastCount(touching.cells, 0));
  }
}

// A place that holds no active cell has no face pieces, and the whole array is kNoIndex.
TEST(FindFacePieces, PlaceWithoutAnActiveCell_HasNone) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  for (const CellPlaceIndex place : kPlacesWithoutAnActiveCell) {
    SCOPED_TRACE(testing::Message() << "place " << place.value);
    const CellFacePieces face_pieces = FindFacePieces(ToTablesOf(test_grid), place);
    EXPECT_EQ(face_pieces.count, 0);
    EXPECT_TRUE(ComputeHasNoIndexPastCount(face_pieces.faces, 0));
  }
}

// kNoIndex and a place of block 2, one past the two-size fixture's 2 blocks, name no place: nothing is read there.
constexpr std::array<CellPlaceIndex, 2> kIndicesOfNoPlace = {
    CellPlaceIndex{kNoIndex},
    CellPlaceIndex{2 * kCellPlacesPerBlock},
};

// An index of no place has no touching cells.
TEST(FindTouchingCells, IndexOfNoPlace_HasNone) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  for (const CellPlaceIndex place : kIndicesOfNoPlace) {
    SCOPED_TRACE(testing::Message() << "place " << place.value);
    const TouchingCells touching = FindTouchingCells(ToTablesOf(test_grid), place);
    EXPECT_EQ(touching.count, 0);
    EXPECT_TRUE(ComputeHasNoIndexPastCount(touching.cells, 0));
  }
}

// In the tables of no grid, block_count 0 and null arrays, place 0 has no touching cells: nothing is read.
TEST(FindTouchingCells, TablesOfNoGrid_HasNone) {
  const GridTables no_grid_tables;
  const TouchingCells in_no_grid = FindTouchingCells(no_grid_tables, CellPlaceIndex{0});
  EXPECT_EQ(in_no_grid.count, 0);
  EXPECT_TRUE(ComputeHasNoIndexPastCount(in_no_grid.cells, 0));
}

// Tables whose block_count is 1 while their arrays still hold block 1 of the two-size fixture: place 64 + 2 is kActive
// in the arrays but lies at block_count, so it is no place of the tables and is not read.
GridTables ToTablesCutToOneBlock(const std::expected<TestGrid, GridFailure>& test_grid) {
  GridTables tables = ToTablesOf(test_grid);
  tables.block_count = 1;
  return tables;
}

// A place of the block at block_count has no touching cells, though the arrays hold an active cell there.
TEST(FindTouchingCells, PlaceAtBlockCount_HasNone) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesCutToOneBlock(test_grid);
  ASSERT_EQ(tables.cell_states[64 + 2], CellState::kActive);
  const TouchingCells touching = FindTouchingCells(tables, CellPlaceIndex{64 + 2});
  EXPECT_EQ(touching.count, 0);
  EXPECT_TRUE(ComputeHasNoIndexPastCount(touching.cells, 0));
}

// An index of no place has no face pieces.
TEST(FindFacePieces, IndexOfNoPlace_HasNone) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  for (const CellPlaceIndex place : kIndicesOfNoPlace) {
    SCOPED_TRACE(testing::Message() << "place " << place.value);
    const CellFacePieces face_pieces = FindFacePieces(ToTablesOf(test_grid), place);
    EXPECT_EQ(face_pieces.count, 0);
    EXPECT_TRUE(ComputeHasNoIndexPastCount(face_pieces.faces, 0));
  }
}

// In the tables of no grid, block_count 0 and null arrays, place 0 has no face pieces: nothing is read.
TEST(FindFacePieces, TablesOfNoGrid_HasNone) {
  const GridTables no_grid_tables;
  const CellFacePieces in_no_grid = FindFacePieces(no_grid_tables, CellPlaceIndex{0});
  EXPECT_EQ(in_no_grid.count, 0);
  EXPECT_TRUE(ComputeHasNoIndexPastCount(in_no_grid.faces, 0));
}

// A place of the block at block_count has no face pieces, though the arrays hold an active cell there.
TEST(FindFacePieces, PlaceAtBlockCount_HasNone) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesCutToOneBlock(test_grid);
  ASSERT_EQ(tables.cell_states[64 + 2], CellState::kActive);
  const CellFacePieces face_pieces = FindFacePieces(tables, CellPlaceIndex{64 + 2});
  EXPECT_EQ(face_pieces.count, 0);
  EXPECT_TRUE(ComputeHasNoIndexPastCount(face_pieces.faces, 0));
}

// A hand-made grid for the readers: its name, its frame's size level count and its active cells.
struct NamedGrid {
  const char* name;
  int size_level_count;
  std::vector<CellId> cells;
};

// Grids the random grids never reach: negative block coordinates, and cells at both coordinate limits, where the
// coordinates of a place beyond the block wrap and plane 4 lies at kMaxCoord + 1. 2^17 - 1 is the largest coordinate a
// size-level-0 cell may have and still be split.
std::vector<NamedGrid> BuildGridsAtNegativeAndLimitCoords() {
  constexpr auto kLargestSplitCoord = static_cast<std::int32_t>((1U << 17U) - 1U);
  std::vector<NamedGrid> grids;
  std::vector<CellId> negative_two_size_cells = {ToCell(0, -1, -1, -1)};
  ApplyAppendSmallCells(ToCell(0, 0, -1, -1), negative_two_size_cells);
  grids.push_back({.name = "two-size fixture at -1", .size_level_count = 2, .cells = negative_two_size_cells});
  grids.push_back(
      {.name = "every neighbour split", .size_level_count = 2, .cells = BuildCellWithEveryNeighbourSplit()}
  );
  grids.push_back({
      .name = "two cells at the high limit",
      .size_level_count = 1,
      .cells = {ToCell(0, kMaxCoord - 1, kMaxCoord, kMaxCoord), ToCell(0, kMaxCoord, kMaxCoord, kMaxCoord)},
  });
  grids.push_back({
      .name = "two cells at the low limit",
      .size_level_count = 1,
      .cells = {ToCell(0, -kMaxCoord, -kMaxCoord, -kMaxCoord), ToCell(0, -kMaxCoord + 1, -kMaxCoord + 1, -kMaxCoord)},
  });
  std::vector<CellId> high_split_cells = {ToCell(0, kLargestSplitCoord - 1, 0, 0)};
  ApplyAppendSmallCells(ToCell(0, kLargestSplitCoord, 0, 0), high_split_cells);
  grids.push_back({.name = "small cells at the high limit", .size_level_count = 2, .cells = high_split_cells});
  std::vector<CellId> low_split_cells = {ToCell(0, -kLargestSplitCoord + 1, 0, 0)};
  ApplyAppendSmallCells(ToCell(0, -kLargestSplitCoord, 0, 0), low_split_cells);
  grids.push_back({.name = "small cells at the low split limit", .size_level_count = 2, .cells = low_split_cells});
  return grids;
}

// Both readers at every active cell of the grids at negative and limit coordinates match the checker, as the random
// grids do. The grids are small, so the test takes a fraction of a second.
TEST(GridTables, GridsAtNegativeAndLimitCoords_MatchGeometricChecker) {
  for (const NamedGrid& named_grid : BuildGridsAtNegativeAndLimitCoords()) {
    SCOPED_TRACE(named_grid.name);
    const GridFrame frame = ToUnitFrame(named_grid.size_level_count);
    const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, named_grid.cells));
    ASSERT_TRUE(test_grid.has_value());
    const GeometricGrid checker(frame, named_grid.cells);
    LargestCounts largest_counts;
    const ActiveCellReaderMistakes mistakes =
        FindActiveCellReaderMistakes(ToTablesOf(test_grid), checker, named_grid.cells, largest_counts);
    CheckNoMistake("FindTouchingCells: the checker's places", mistakes.touching_cells_not_the_checkers);
    CheckNoMistake("FindTouchingCells: strictly ascending", mistakes.touching_cells_out_of_order);
    CheckNoMistake("FindTouchingCells: kNoIndex from count on", mistakes.touching_entries_past_count_not_no_index);
    CheckNoMistake("FindFacePieces: the checker's places", mistakes.face_pieces_not_the_checkers);
    CheckNoMistake("FindFacePieces: the checker's ids as a set", mistakes.face_piece_ids_not_the_checkers);
    CheckNoMistake("FindFacePieces: strictly ascending", mistakes.face_pieces_out_of_order);
    CheckNoMistake("FindFacePieces: kNoIndex from count on", mistakes.face_entries_past_count_not_no_index);
    CheckNoMistake("every side covered once", mistakes.sides_not_covered_once);
    CheckNoMistake("surface closed", mistakes.surfaces_not_closed);
    EXPECT_GT(largest_counts.touching_cells, 0);
  }
}

}  // namespace
}  // namespace fluidity::grid
