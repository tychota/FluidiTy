#include "fluidity/grid/cell_corners.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/cell_corners_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// The corner places of a result, as plain numbers.
template <typename CornerArray>
std::vector<std::uint32_t> ToCornerValues(const CornerArray& corners) {
  std::vector<std::uint32_t> corner_values;
  for (const CornerPlaceIndex corner : corners) {
    corner_values.push_back(corner.value);
  }
  return corner_values;
}

// Says whether every entry of a result's array is kNoIndex.
template <typename PlaceArray>
bool ComputeIsAllNoIndex(const PlaceArray& places) {
  return std::ranges::all_of(places, [](const auto place) {
    return place.value == kNoIndex;
  });
}

// A default-made result has every entry kNoIndex, as a reader gives for a place without a cell or a face piece.
static_assert(CellCorners().corners[0].value == kNoIndex && CellCorners().corners[7].value == kNoIndex);

static_assert(FaceCorners().corners[0].value == kNoIndex && FaceCorners().corners[3].value == kNoIndex);

static_assert(CellsAroundEdge().cells[0].value == kNoIndex && CellsAroundEdge().cells[3].value == kNoIndex);

static_assert(!CellsAroundEdge().is_same_size);

// Example 1 of the header: the corner of {3,0,0} on its high x side, at {4,0,0} m, is lattice point (4, 0, 0) of
// block 0 and (0, 0, 0) of block 1, and is stored in block 1, place 125. With {3,0,0} alone it is place 4.
TEST(FindCornerPlaces, CornerOnHighSideOfBlock_IsStoredInHighBlock) {
  const std::vector<CellId> two_cells = {ToCell(0, 3, 0, 0), ToCell(0, 4, 0, 0)};
  const std::expected<TestGrid, GridFailure> two_cell_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), two_cells));
  ASSERT_TRUE(two_cell_grid.has_value());
  const GridTables two_cell_tables = ToTablesOf(two_cell_grid);
  const CellPlaceIndex low_cell = FindCellPlace(two_cell_tables, two_cells.at(0));
  const CellPlaceIndex high_cell = FindCellPlace(two_cell_tables, two_cells.at(1));
  EXPECT_EQ(FindCornerPlaces(two_cell_tables, low_cell).corners[1].value, 125U);
  // The same point is the low corner, corner 0, of {4,0,0}.
  EXPECT_EQ(FindCornerPlaces(two_cell_tables, high_cell).corners[0].value, 125U);
  EXPECT_TRUE(ComputeIsSameVec3(ComputeCornerPosition(two_cell_tables, CornerPlaceIndex{125}), {4, 0, 0}));

  const std::vector<CellId> lone_cell = {ToCell(0, 3, 0, 0)};
  const std::expected<TestGrid, GridFailure> lone_cell_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), lone_cell));
  ASSERT_TRUE(lone_cell_grid.has_value());
  const GridTables lone_cell_tables = ToTablesOf(lone_cell_grid);
  const CellCorners lone_corners = FindCornerPlaces(lone_cell_tables, FindCellPlace(lone_cell_tables, lone_cell.at(0)));
  EXPECT_EQ(lone_corners.corners[1].value, 4U);
  EXPECT_TRUE(ComputeIsSameVec3(ComputeCornerPosition(lone_cell_tables, CornerPlaceIndex{4}), {4, 0, 0}));
}

// The corner places of the point {1, 0.5, 0.5} m as the four small cells around it on the big cell's side of the
// two-size fixture give it: {2, 0..1, 0..1}, each by the corner on its low x side whose y and z bits point at it.
std::vector<std::uint32_t> ToSmallCellPlacesOfPoint(const GridTables& tables) {
  std::vector<std::uint32_t> places_of_point;
  for (std::int32_t y = 0; y <= 1; ++y) {
    for (std::int32_t z = 0; z <= 1; ++z) {
      const CellCorners small_corners = FindCornerPlaces(tables, FindCellPlace(tables, ToCell(1, 2, y, z)));
      const int corner_number = ((1 - y) * 2) + ((1 - z) * 4);
      places_of_point.push_back(small_corners.corners[corner_number].value);
    }
  }
  return places_of_point;
}

// Example 2 of the header, in the two-size fixture: block 1 is the size-level-1 block, covering 0 to 2 m.
TEST(FindCornerPlaces, CornerInMiddleOfBigFace_BelongsToSmallCellsOnly) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  EXPECT_EQ(ToSmallCellPlacesOfPoint(tables), (std::vector<std::uint32_t>{157, 157, 157, 157}));
  // The big cell's 8 corners are lattice points (0 or 2, 0 or 2, 0 or 2) of block 1, all in 125..249, none 157:
  // {0,0,0} m is 125, though no small cell touches it; {1,0,0} m is 127; {1,1,1} m is 187.
  const std::vector<std::uint32_t> big_cell_corners = {125, 127, 135, 137, 175, 177, 185, 187};
  EXPECT_EQ(ToCornerValues(FindCornerPlaces(tables, CellPlaceIndex{0}).corners), big_cell_corners);
  // The big cell's corner at {1,0,0} m is the low corner of the small cell {2,0,0}.
  EXPECT_EQ(FindCornerPlaces(tables, FindCellPlace(tables, ToCell(1, 2, 0, 0))).corners[0].value, 127U);
  EXPECT_TRUE(ComputeIsSameVec3(ComputeCornerPosition(tables, CornerPlaceIndex{125}), {0, 0, 0}));
  EXPECT_TRUE(ComputeIsSameVec3(ComputeCornerPosition(tables, CornerPlaceIndex{157}), {1, 0.5, 0.5}));
}

// The example of FindCornerPlaces for a face piece, in the two-size fixture: the face piece at X place 2 of block 1,
// beside the small cell {2,0,0}, has the corners of that cell's low x side, lattice points (2, 0..1, 0..1) of block 1;
// the big cell's whole -X side, X place 0 of block 0, has its corners at (0, 0 or 2, 0 or 2) of block 1.
TEST(FindCornerPlaces, FacePieceBesideBigCell_HasTheSmallCellsCorners) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const std::vector<std::uint32_t> quarter_corners = {127, 132, 152, 157};
  EXPECT_EQ(ToCornerValues(FindCornerPlaces(tables, FacePlaceIndex{240 + 2}).corners), quarter_corners);
  const std::vector<std::uint32_t> whole_side_corners = {125, 135, 175, 185};
  EXPECT_EQ(ToCornerValues(FindCornerPlaces(tables, FacePlaceIndex{0}).corners), whole_side_corners);
}

// Corner bits of the point {4,4,0} m in the cells {3,3,0}, {4,3,0}, {3,4,0} and {4,4,0}: high x for x = 3, high y for
// y = 3.
constexpr std::array<int, 4> kCornerOfEdgePoint = {1 + 2, 2, 1, 0};

// The corner place of the point {4,4,0} m as each of `cells`, the first of those four, gives it.
std::vector<std::uint32_t> ToPlacesOfEdgePoint(const GridTables& tables, const std::vector<CellId>& cells) {
  std::vector<std::uint32_t> places_of_point;
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const CellPlaceIndex place = FindCellPlace(tables, cells.at(cell));
    places_of_point.push_back(FindCornerPlaces(tables, place).corners[kCornerOfEdgePoint.at(cell)].value);
  }
  return places_of_point;
}

// Four blocks around the edge x = y = 4 m: the point {4,4,0} m is stored in the last of them in the fixed order,
// block 3, at its lattice point (0, 0, 0). Without the cell {4,4,0} it is stored in block 2, the block of {3,4,0}, at
// its lattice point (4, 0, 0).
TEST(FindCornerPlaces, FourBlocksAroundAnEdge_LastBlockInFixedOrderStoresIt) {
  const std::vector<CellId> four_cells = {
      ToCell(0, 3, 3, 0),
      ToCell(0, 4, 3, 0),
      ToCell(0, 3, 4, 0),
      ToCell(0, 4, 4, 0),
  };
  const std::expected<TestGrid, GridFailure> four_cell_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), four_cells));
  ASSERT_TRUE(four_cell_grid.has_value());
  const GridTables four_cell_tables = ToTablesOf(four_cell_grid);
  // The cells lie in blocks 0, 1, 2 and 3, in the fixed order.
  std::vector<std::uint32_t> cell_places;
  cell_places.reserve(four_cells.size());
  for (const CellId cell : four_cells) {
    cell_places.push_back(FindCellPlace(four_cell_tables, cell).value);
  }
  EXPECT_EQ(cell_places, (std::vector<std::uint32_t>{0 + 3 + 12, 64 + 0 + 12, 128 + 3 + 0, 192 + 0 + 0}));
  EXPECT_EQ(ToPlacesOfEdgePoint(four_cell_tables, four_cells), (std::vector<std::uint32_t>{375, 375, 375, 375}));

  const std::vector<CellId> three_cells = {four_cells.at(0), four_cells.at(1), four_cells.at(2)};
  const std::expected<TestGrid, GridFailure> three_cell_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), three_cells));
  ASSERT_TRUE(three_cell_grid.has_value());
  const GridTables three_cell_tables = ToTablesOf(three_cell_grid);
  EXPECT_EQ(ToPlacesOfEdgePoint(three_cell_tables, three_cells), (std::vector<std::uint32_t>{254, 254, 254}));
  EXPECT_TRUE(ComputeIsSameVec3(ComputeCornerPosition(three_cell_tables, CornerPlaceIndex{254}), {4, 4, 0}));
}

// The size-level-0 cells {0..7}^3, 1 size level: 8 blocks.
std::vector<CellId> BuildUniformCells() {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 8; ++z) {
    for (std::int32_t y = 0; y < 8; ++y) {
      for (std::int32_t x = 0; x < 8; ++x) {
        cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  return cells;
}

// The face pieces of a grid and the face corners among them that differ from the corner of a cell beside them.
struct FaceCornerCounts {
  int face_pieces = 0;
  int corners_not_the_cell_corners = 0;
};

// Compares, for every face piece of the tables, its 4 corners with the corners of its cell_low on that cell's high side
// and of its cell_high on that cell's low side.
FaceCornerCounts CountFaceCornersNotTheCellCorners(const GridTables& tables) {
  FaceCornerCounts counts;
  for (std::uint32_t face_value = 0; face_value < tables.block_count * kFacePlacesPerBlock; ++face_value) {
    const FacePlaceIndex face = {.value = face_value};
    if (tables.face_piece_kinds[face_value] == FacePieceKind::kNotAPiece) {
      continue;
    }
    ++counts.face_pieces;
    const FaceCorners face_corners = FindCornerPlaces(tables, face);
    for (const bool is_cell_low : {true, false}) {
      const CellPlaceIndex cell = is_cell_low ? FindCellLow(tables, face) : FindCellHigh(tables, face);
      if (cell.value == kNoIndex) {
        continue;
      }
      const CellCorners cell_corners = FindCornerPlaces(tables, cell);
      for (int face_corner_number = 0; face_corner_number < 4; ++face_corner_number) {
        const int cell_corner_number = ToCellCornerOfFaceCorner(
            ToAxis(face),
            face_corner_number,
            is_cell_low ? SideOfCell::kHigh : SideOfCell::kLow
        );
        const bool is_same_place =
            face_corners.corners[face_corner_number].value == cell_corners.corners[cell_corner_number].value;
        counts.corners_not_the_cell_corners += is_same_place ? 0 : 1;
      }
    }
  }
  return counts;
}

// Every face piece of a uniform grid has as its 4 corners the corners of its cell_low on that cell's high side and the
// corners of its cell_high on that cell's low side.
TEST(FindCornerPlaces, FaceCorners_AreTheCellCornersOnThatSide) {
  const std::vector<CellId> cells = BuildUniformCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const FaceCornerCounts counts = CountFaceCornersNotTheCellCorners(ToTablesOf(test_grid));
  EXPECT_EQ(counts.corners_not_the_cell_corners, 0);
  // 3 axes x 9 planes x 8 x 8.
  EXPECT_EQ(counts.face_pieces, 3 * 9 * 64);
}

// Counts the edges of a size-level-0 cell, for every ordered pair of axes and every quarter, whose FindCellsAroundEdge
// result is not the four cells at the test's own coordinates, all of the same size.
int CountUniformEdgeMistakes(const GridTables& tables, CellCoord cell_coord) {
  int mistake_count = 0;
  const CellPlaceIndex cell = FindCellPlace(tables, ToCellId(SizeLevel{0}, cell_coord));
  for (const std::array<Axis, 2>& axes : kAxisPairs) {
    for (int quarter = 0; quarter < 4; ++quarter) {
      const CellsAroundEdge around = FindCellsAroundEdge(tables, cell, axes.at(0), axes.at(1), quarter);
      const std::array<CellCoord, 4> coords = ToCoordsAroundEdge(cell_coord, axes.at(0), axes.at(1), quarter);
      EdgeValues expected_values = {0, 0, 0, 0, 1};
      for (std::size_t around_number = 0; around_number < coords.size(); ++around_number) {
        expected_values.at(around_number) =
            FindCellPlace(tables, ToCellId(SizeLevel{0}, coords.at(around_number))).value;
      }
      mistake_count += ToEdgeValues(around) == expected_values ? 0 : 1;
    }
  }
  return mistake_count;
}

// In a uniform grid every edge of a cell inside it has four cells of its size, here for a cell at local (3, 3, 3) of
// block 0, whose high places lie in other blocks, and one at local (0, 0, 0) of block 7, whose low places do.
TEST(FindCellsAroundEdge, UniformGrid_FindsFourSameSize) {
  const std::vector<CellId> cells = BuildUniformCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  EXPECT_EQ(CountUniformEdgeMistakes(tables, {.x = 3, .y = 3, .z = 3}), 0);
  EXPECT_EQ(CountUniformEdgeMistakes(tables, {.x = 4, .y = 4, .z = 4}), 0);
}

// A split place around the edge holds smaller cells: no place is given for it and is_same_size is false. In the
// two-size fixture the big cell's +X neighbour is split; in the second grid only the place across both axes is.
TEST(FindCellsAroundEdge, BesideSplitCell_IsNotSameSize) {
  const std::vector<CellId> fixture_cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> fixture_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), fixture_cells));
  ASSERT_TRUE(fixture_grid.has_value());
  const CellsAroundEdge beside_split =
      FindCellsAroundEdge(ToTablesOf(fixture_grid), CellPlaceIndex{0}, Axis::kX, Axis::kY, 1);
  // cells[2], {0,-1,0}, and cells[3], {1,-1,0}, lie in no block and inside no cell: absent.
  EXPECT_EQ(ToEdgeValues(beside_split), (EdgeValues{0, kNoIndex, kNoIndex, kNoIndex, 0}));

  const std::vector<CellId> diagonal_cells = BuildSplitCellAcrossEdge();
  const std::expected<TestGrid, GridFailure> diagonal_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), diagonal_cells));
  ASSERT_TRUE(diagonal_grid.has_value());
  const GridTables tables = ToTablesOf(diagonal_grid);
  const CellsAroundEdge split_across_both =
      FindCellsAroundEdge(tables, FindCellPlace(tables, ToCell(0, 0, 0, 0)), Axis::kX, Axis::kY, 3);
  const EdgeValues expected_values = {
      FindCellPlace(tables, ToCell(0, 0, 0, 0)).value,
      FindCellPlace(tables, ToCell(0, 1, 0, 0)).value,
      FindCellPlace(tables, ToCell(0, 0, 1, 0)).value,
      kNoIndex,
      0,
  };
  EXPECT_EQ(ToEdgeValues(split_across_both), expected_values);
}

// A place inside a bigger cell is not of the same size, whether its block of the cell's size level exists or not. In
// the two-size fixture the place left of the small cell {2,0,0} is kInsideBiggerCell in block 1; beside the big cell on
// the low side, the place left of the small cell {0,0,0} has no size-level-1 block.
TEST(FindCellsAroundEdge, BesideBiggerCell_IsNotSameSize) {
  const std::vector<CellId> fixture_cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> fixture_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), fixture_cells));
  ASSERT_TRUE(fixture_grid.has_value());
  const GridTables fixture_tables = ToTablesOf(fixture_grid);
  const CellPlaceIndex small_cell = FindCellPlace(fixture_tables, ToCell(1, 2, 0, 0));
  const CellsAroundEdge inside_big = FindCellsAroundEdge(fixture_tables, small_cell, Axis::kX, Axis::kY, 0);
  EXPECT_EQ(ToEdgeValues(inside_big), (EdgeValues{small_cell.value, kNoIndex, kNoIndex, kNoIndex, 0}));

  const std::vector<CellId> low_big_cells = BuildBigCellOnLowSide();
  const std::expected<TestGrid, GridFailure> low_big_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), low_big_cells));
  ASSERT_TRUE(low_big_grid.has_value());
  const GridTables tables = ToTablesOf(low_big_grid);
  ASSERT_EQ(FindCellBlock(tables, ToCellBlockId(ToCell(1, -1, 0, 0))).value, kNoIndex);
  const CellPlaceIndex corner_cell = FindCellPlace(tables, ToCell(1, 0, 0, 0));
  // Quarter 2: the low x side and the high y side.
  const CellsAroundEdge no_block = FindCellsAroundEdge(tables, corner_cell, Axis::kX, Axis::kY, 2);
  const EdgeValues expected_values = {
      corner_cell.value,
      kNoIndex,
      FindCellPlace(tables, ToCell(1, 0, 1, 0)).value,
      kNoIndex,
      0,
  };
  EXPECT_EQ(ToEdgeValues(no_block), expected_values);
}

// Absent places are not another size: at the grid's edge the places are kNoIndex and is_same_size stays true. A lone
// cell; and the small cell {3,1,1} of the two-size fixture toward high x and y, where no size-level-1 block lies and
// the size-level-0 places that would cover the places are absent.
TEST(FindCellsAroundEdge, AtGridEdge_IsNoIndexAndSameSize) {
  const std::vector<CellId> lone_cell = {ToCell(0, 0, 0, 0)};
  const std::expected<TestGrid, GridFailure> lone_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), lone_cell));
  ASSERT_TRUE(lone_grid.has_value());
  constexpr EdgeValues kLoneValues = {0, kNoIndex, kNoIndex, kNoIndex, 1};
  for (int quarter = 0; quarter < 4; ++quarter) {
    const CellsAroundEdge around =
        FindCellsAroundEdge(ToTablesOf(lone_grid), CellPlaceIndex{0}, Axis::kY, Axis::kZ, quarter);
    EXPECT_EQ(ToEdgeValues(around), kLoneValues) << "quarter " << quarter;
  }

  const std::vector<CellId> fixture_cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> fixture_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), fixture_cells));
  ASSERT_TRUE(fixture_grid.has_value());
  const GridTables tables = ToTablesOf(fixture_grid);
  const CellPlaceIndex small_cell = FindCellPlace(tables, ToCell(1, 3, 1, 1));
  const CellsAroundEdge beyond_blocks = FindCellsAroundEdge(tables, small_cell, Axis::kX, Axis::kY, 3);
  EXPECT_EQ(ToEdgeValues(beyond_blocks), (EdgeValues{small_cell.value, kNoIndex, kNoIndex, kNoIndex, 1}));
}

// In the two-size fixture: place 1, split; place 2, absent; place 64 + 1, inside the big cell.
constexpr std::array<CellPlaceIndex, 3> kPlacesWithoutAnActiveCell = {
    CellPlaceIndex{1},
    CellPlaceIndex{2},
    CellPlaceIndex{64 + 1},
};

// kNoIndex and a place of block 2, one past the two-size fixture's 2 blocks, name no place.
constexpr std::array<std::uint32_t, 2> kIndexesOfNoPlace = {kNoIndex, 2 * kCellPlacesPerBlock};

// A cell place that holds no active cell has no corners.
TEST(FindCornerPlaces, CellPlaceWithoutActiveCell_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  for (const CellPlaceIndex place : kPlacesWithoutAnActiveCell) {
    EXPECT_TRUE(ComputeIsAllNoIndex(FindCornerPlaces(ToTablesOf(test_grid), place).corners)) << "place " << place.value;
  }
}

// A face place that holds no face piece has no corners: X place 1 of block 0, between the big cell and the split cell.
TEST(FindCornerPlaces, FacePlaceWithoutFacePiece_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  ASSERT_EQ(tables.face_piece_kinds[1], FacePieceKind::kNotAPiece);
  EXPECT_TRUE(ComputeIsAllNoIndex(FindCornerPlaces(tables, FacePlaceIndex{1}).corners));
}

// Tables whose block_count is 1 while their arrays still hold block 1 of the two-size fixture.
GridTables ToTablesCutToOneBlock(const std::expected<TestGrid, GridFailure>& test_grid) {
  GridTables tables = ToTablesOf(test_grid);
  tables.block_count = 1;
  return tables;
}

// An index of no place, kNoIndex or a place of a block at or beyond block_count, has no corners: nothing is read
// there. In the tables cut to one block, the active small cell at place 64 + 2 and the face piece at X place 2 of
// block 1 lie at block_count.
TEST(FindCornerPlaces, IndexOfNoPlace_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const GridTables cut_tables = ToTablesCutToOneBlock(test_grid);
  ASSERT_EQ(cut_tables.cell_states[64 + 2], CellState::kActive);
  ASSERT_EQ(cut_tables.face_piece_kinds[240 + 2], FacePieceKind::kBetweenCells);
  const std::array<CellCorners, 3> cell_results = {
      FindCornerPlaces(tables, CellPlaceIndex{kIndexesOfNoPlace[0]}),
      FindCornerPlaces(tables, CellPlaceIndex{kIndexesOfNoPlace[1]}),
      FindCornerPlaces(cut_tables, CellPlaceIndex{64 + 2}),
  };
  EXPECT_TRUE(std::ranges::all_of(cell_results, [](const CellCorners& result) {
    return ComputeIsAllNoIndex(result.corners);
  }));
  const std::array<FaceCorners, 3> face_results = {
      FindCornerPlaces(tables, FacePlaceIndex{kNoIndex}),
      FindCornerPlaces(tables, FacePlaceIndex{480}),
      FindCornerPlaces(cut_tables, FacePlaceIndex{240 + 2}),
  };
  EXPECT_TRUE(std::ranges::all_of(face_results, [](const FaceCorners& result) {
    return ComputeIsAllNoIndex(result.corners);
  }));
}

// In the tables of no grid, block_count 0 and null arrays, place 0 has no corners: nothing is read.
TEST(FindCornerPlaces, TablesOfNoGrid_IsNoIndex) {
  const GridTables no_grid_tables;
  EXPECT_TRUE(ComputeIsAllNoIndex(FindCornerPlaces(no_grid_tables, CellPlaceIndex{0}).corners));
  EXPECT_TRUE(ComputeIsAllNoIndex(FindCornerPlaces(no_grid_tables, FacePlaceIndex{0}).corners));
}

bool ComputeIsInfiniteEverywhere(Vec3 position) {
  return position.x == math::kPositiveInfinity && position.y == math::kPositiveInfinity
      && position.z == math::kPositiveInfinity;
}

// A corner index of no place has no position: +infinity on each axis, and nothing is read. Index 250 is the first
// corner of block 2 of the two-size fixture, and 125 lies at block_count in the tables cut to one block.
TEST(ComputeCornerPosition, IndexOfNoPlace_IsInfinite) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  EXPECT_TRUE(ComputeIsInfiniteEverywhere(ComputeCornerPosition(ToTablesOf(test_grid), CornerPlaceIndex{kNoIndex})));
  EXPECT_TRUE(ComputeIsInfiniteEverywhere(ComputeCornerPosition(ToTablesOf(test_grid), CornerPlaceIndex{250})));
  EXPECT_TRUE(
      ComputeIsInfiniteEverywhere(ComputeCornerPosition(ToTablesCutToOneBlock(test_grid), CornerPlaceIndex{125}))
  );
  EXPECT_TRUE(ComputeIsInfiniteEverywhere(ComputeCornerPosition(GridTables{}, CornerPlaceIndex{0})));
}

// In a frame whose origin and width are not powers of two, the position is origin + m h_l by one product and one sum,
// m the lattice coordinate at the block's size level; the same point at the other size level, m 2 and h_l / 2, gives
// the same double. In the two-size fixture, corner place 127 is lattice point (2, 0, 0) of block 1, size level 1, and
// the big cell's corner {1, 0, 0} at size level 0.
TEST(ComputeCornerPosition, OffsetFrame_IsOriginPlusLatticeCoordTimesWidth) {
  const GridFrame frame = {
      .origin = {.x = -2.1, .y = 0.3, .z = 8.7},
      .biggest_cell_width = 0.7,
      .size_level_count = 2,
  };
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const double small_width = ComputeCellWidth(frame, SizeLevel{1});
  const Vec3 position = ComputeCornerPosition(tables, CornerPlaceIndex{127});
  EXPECT_EQ(position.x, frame.origin.x + (2.0 * small_width));
  EXPECT_EQ(position.x, frame.origin.x + (1.0 * frame.biggest_cell_width));
  EXPECT_EQ(position.y, frame.origin.y + (0.0 * small_width));
  EXPECT_EQ(position.z, frame.origin.z + (0.0 * small_width));
  // Place 187, lattice point (2, 2, 2): the big cell's corner {1, 1, 1}.
  const Vec3 far_position = ComputeCornerPosition(tables, CornerPlaceIndex{187});
  EXPECT_EQ(far_position.y, frame.origin.y + (2.0 * small_width));
  EXPECT_EQ(far_position.z, frame.origin.z + (1.0 * frame.biggest_cell_width));
}

// The empty result of FindCellsAroundEdge as plain numbers: every place kNoIndex and is_same_size false.
constexpr EdgeValues kEmptyEdgeValues = {kNoIndex, kNoIndex, kNoIndex, kNoIndex, 0};

// In the tables of no grid and at an index of no place, FindCellsAroundEdge gives an empty result: every entry kNoIndex
// and is_same_size false. So does a place that holds no active cell.
TEST(FindCellsAroundEdge, PlaceWithoutActiveCell_IsEmpty) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  for (const CellPlaceIndex place : kPlacesWithoutAnActiveCell) {
    const CellsAroundEdge around = FindCellsAroundEdge(ToTablesOf(test_grid), place, Axis::kX, Axis::kY, 3);
    EXPECT_EQ(ToEdgeValues(around), kEmptyEdgeValues) << "place " << place.value;
  }
}

TEST(FindCellsAroundEdge, IndexOfNoPlace_IsEmpty) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  for (const std::uint32_t index : kIndexesOfNoPlace) {
    const CellsAroundEdge around =
        FindCellsAroundEdge(ToTablesOf(test_grid), CellPlaceIndex{index}, Axis::kX, Axis::kY, 0);
    EXPECT_EQ(ToEdgeValues(around), kEmptyEdgeValues) << "index " << index;
  }
  const CellsAroundEdge at_block_count =
      FindCellsAroundEdge(ToTablesCutToOneBlock(test_grid), CellPlaceIndex{64 + 2}, Axis::kX, Axis::kY, 0);
  EXPECT_EQ(ToEdgeValues(at_block_count), kEmptyEdgeValues);
}

TEST(FindCellsAroundEdge, TablesOfNoGrid_IsEmpty) {
  const GridTables no_grid_tables;
  const CellsAroundEdge around = FindCellsAroundEdge(no_grid_tables, CellPlaceIndex{0}, Axis::kX, Axis::kY, 0);
  EXPECT_EQ(ToEdgeValues(around), kEmptyEdgeValues);
}

// Two equal axes, an axis value of 3 and a quarter outside 0..3 name no edge: an empty result, nothing read.
TEST(FindCellsAroundEdge, NoEdgeNamed_IsEmpty) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const auto no_axis = static_cast<Axis>(3);
  const std::array<CellsAroundEdge, 5> results = {
      FindCellsAroundEdge(tables, CellPlaceIndex{0}, Axis::kX, Axis::kX, 0),
      FindCellsAroundEdge(tables, CellPlaceIndex{0}, no_axis, Axis::kY, 0),
      FindCellsAroundEdge(tables, CellPlaceIndex{0}, Axis::kZ, no_axis, 0),
      FindCellsAroundEdge(tables, CellPlaceIndex{0}, Axis::kX, Axis::kY, 4),
      FindCellsAroundEdge(tables, CellPlaceIndex{0}, Axis::kX, Axis::kY, -1),
  };
  for (const CellsAroundEdge& around : results) {
    EXPECT_EQ(ToEdgeValues(around), kEmptyEdgeValues);
  }
}

// Grids at both coordinate limits, where a corner lies at kMaxCoord + 1 and, with 16 size levels, a block coordinate
// beyond the range of a block id is met.
std::vector<NamedGrid> BuildGridsAtCoordLimits() {
  constexpr auto kLargestSplitCoord = static_cast<std::int32_t>((1U << 17U) - 1U);
  std::vector<NamedGrid> grids;
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
  AppendSmallCells(ToCell(0, kLargestSplitCoord, 0, 0), high_split_cells);
  grids.push_back({.name = "small cells at the high limit", .size_level_count = 2, .cells = high_split_cells});
  std::vector<CellId> low_split_cells = {ToCell(0, -kLargestSplitCoord + 1, 0, 0)};
  AppendSmallCells(ToCell(0, -kLargestSplitCoord, 0, 0), low_split_cells);
  grids.push_back({.name = "small cells at the low split limit", .size_level_count = 2, .cells = low_split_cells});
  grids.push_back({
      .name = "lone cell at the high limit, 16 size levels",
      .size_level_count = kMaxSizeLevels,
      .cells = {ToCell(0, kMaxCoord, kMaxCoord, kMaxCoord)},
  });
  // A full search from size level 15 would scale the high corner of the cell at the limit, 2^18, to 2^33, in the block
  // 2^31 or 2^31 - 1: outside the range of a block id, where 2^31 would pack into the id of the size-level-15 block
  // (0,0,0). The bounded search stops at size level 2, where the block coordinate is at most kMaxCoord + 1: outside
  // the range too, and never the coordinate of a block of the grid. The range check of the reader is therefore a
  // guard for block ids that contradict themselves, which no grid here reaches.
  std::vector<CellId> deep_and_far_cells = BuildCellsDownToSmallestAtOrigin();
  deep_and_far_cells.push_back(ToCell(0, kMaxCoord, kMaxCoord, kMaxCoord));
  grids.push_back({
      .name = "smallest cells at the origin and a cell at the high limit, 16 size levels",
      .size_level_count = kMaxSizeLevels,
      .cells = deep_and_far_cells,
  });
  grids.push_back({
      .name = "lone cell at the low limit, 16 size levels",
      .size_level_count = kMaxSizeLevels,
      .cells = {ToCell(0, -kMaxCoord, -kMaxCoord, -kMaxCoord)},
  });
  return grids;
}

// A frame whose origin and width are not powers of two, where a corner position rounds.
constexpr GridFrame ToOffsetFrame(int size_level_count) {
  return {
      .origin = {.x = -2.1, .y = 0.3, .z = 8.7},
      .biggest_cell_width = 0.7,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

// Counts the corners of active cells whose ComputeCornerPosition differs, in any bit, from origin + (c + bit) h_l
// computed at the cell's own size level l: the same double from every cell that reaches a corner.
int CountCornerPositionsNotTheCells(const GridTables& tables, const std::vector<CellId>& cells) {
  int mistake_count = 0;
  for (const CellId cell_id : cells) {
    const CellCorners cell_corners = FindCornerPlaces(tables, FindCellPlace(tables, cell_id));
    const double cell_width = ComputeCellWidth(tables.frame, ToSizeLevel(cell_id));
    const CellCoord cell_coord = ToCellCoord(cell_id);
    for (int corner_number = 0; corner_number < 8; ++corner_number) {
      const auto corner_bits = static_cast<unsigned>(corner_number);
      // The corner's integer coordinate at the cell's size level, c + 0 or 1, kept signed.
      const std::int64_t corner_x = std::int64_t{cell_coord.x} + static_cast<std::int64_t>(corner_bits & 1U);
      const std::int64_t corner_y = std::int64_t{cell_coord.y} + static_cast<std::int64_t>((corner_bits >> 1U) & 1U);
      const std::int64_t corner_z = std::int64_t{cell_coord.z} + static_cast<std::int64_t>((corner_bits >> 2U) & 1U);
      const Vec3 expected_position = {
          .x = tables.frame.origin.x + (static_cast<double>(corner_x) * cell_width),
          .y = tables.frame.origin.y + (static_cast<double>(corner_y) * cell_width),
          .z = tables.frame.origin.z + (static_cast<double>(corner_z) * cell_width),
      };
      const Vec3 position = ComputeCornerPosition(tables, cell_corners.corners[corner_number]);
      mistake_count += ComputeIsSameVec3(position, expected_position) ? 0 : 1;
    }
  }
  return mistake_count;
}

// Checks the readers on hand-made grids: against the checker in the unit frame, where every position is exact, and,
// built again in the offset frame, every corner position bit for bit against each cell that reaches it.
void CheckHandMadeGrids(const std::vector<NamedGrid>& grids) {
  for (const NamedGrid& named_grid : grids) {
    SCOPED_TRACE(named_grid.name);
    const GridFrame frame = ToUnitFrame(named_grid.size_level_count);
    const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, named_grid.cells));
    ASSERT_TRUE(test_grid.has_value());
    const GeometricGrid checker(frame, named_grid.cells);
    CheckNoCornerOrEdgeMistake(FindCornerAndEdgeMistakes(ToTablesOf(test_grid), checker, named_grid.cells, kAxisPairs));
    const GridFrame offset_frame = ToOffsetFrame(named_grid.size_level_count);
    const std::expected<TestGrid, GridFailure> offset_grid = BuildTestGrid(ToInputs(offset_frame, named_grid.cells));
    ASSERT_TRUE(offset_grid.has_value());
    EXPECT_EQ(CountCornerPositionsNotTheCells(ToTablesOf(offset_grid), named_grid.cells), 0);
  }
}

// The readers at every active cell and face piece of the hand-made grids around the origin match the checker, as in
// the random grids, and give each corner one position, bit for bit, in a frame that rounds. A fraction of a second.
TEST(GridTables, GridsAroundOrigin_EveryCornerOrEdgeReaderMatchesGeometricChecker) {
  CheckHandMadeGrids(BuildGridsAroundOrigin());
}

// The same at both coordinate limits.
TEST(GridTables, GridsAtCoordLimits_EveryCornerOrEdgeReaderMatchesGeometricChecker) {
  CheckHandMadeGrids(BuildGridsAtCoordLimits());
}

// The corner of the big cell {0,0,0} at {1,0,0} m is stored two size levels down, in the size-level-2 block (1,0,0),
// block 2, at its lattice point (0, 0, 0): place 250. The size-level-1 block (0,0,0) holds the point too, at (2, 0, 0),
// so a search that started one size level down would give 125 + 2.
TEST(FindCornerPlaces, CornerTwoSizeLevelsDown_IsStoredInSmallestBlock) {
  const std::vector<CellId> cells = BuildCornerStoredTwoSizeLevelsDown();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  EXPECT_EQ(FindCornerPlaces(tables, FindCellPlace(tables, ToCell(0, 0, 0, 0))).corners[1].value, 250U);
  EXPECT_EQ(FindCornerPlaces(tables, FindCellPlace(tables, ToCell(1, 2, 0, 0))).corners[0].value, 250U);
}

// Every corner place of every active cell and face piece of the tables, in place order, as plain numbers.
std::vector<std::uint32_t> ToEveryCornerPlace(const GridTables& tables) {
  std::vector<std::uint32_t> corner_places;
  for (std::uint32_t cell_value = 0; cell_value < tables.block_count * kCellPlacesPerBlock; ++cell_value) {
    for (const CornerPlaceIndex corner : FindCornerPlaces(tables, CellPlaceIndex{cell_value}).corners) {
      corner_places.push_back(corner.value);
    }
  }
  for (std::uint32_t face_value = 0; face_value < tables.block_count * kFacePlacesPerBlock; ++face_value) {
    for (const CornerPlaceIndex corner : FindCornerPlaces(tables, FacePlaceIndex{face_value}).corners) {
      corner_places.push_back(corner.value);
    }
  }
  return corner_places;
}

// Tables whose frame claims 255 size levels give the places of the built grid: the search never starts beyond size
// level 15, where a size level of 16 or 17 would pack into the id of a size-level-0 or -1 block. In the grid of the
// smallest cells at the origin, the size-level-14 and -15 cells reach that guard.
TEST(FindCornerPlaces, FrameBeyondLastSizeLevel_GivesTheBuiltGridsPlaces) {
  const std::vector<CellId> cells = BuildCellsDownToSmallestAtOrigin();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(kMaxSizeLevels), cells));
  ASSERT_TRUE(test_grid.has_value());
  GridTables contradicting_tables = ToTablesOf(test_grid);
  contradicting_tables.frame.size_level_count = 255;
  EXPECT_EQ(ToEveryCornerPlace(contradicting_tables), ToEveryCornerPlace(ToTablesOf(test_grid)));
}

// Tables whose frame claims 1 size level, below the size level of their small cells: each search starts at the cell's
// or face piece's own size level, so every corner still has a place in the tables, and the big cell's corners lie in
// block 0, lattice points (0 or 1, 0 or 1, 0 or 1).
TEST(FindCornerPlaces, FrameOfOneSizeLevel_GivesPlacesInsideTheTables) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  GridTables contradicting_tables = ToTablesOf(test_grid);
  contradicting_tables.frame.size_level_count = 1;
  const std::uint32_t corner_place_count = contradicting_tables.block_count * kCornerPlacesPerBlock;
  std::vector<std::uint32_t> corners_of_active_cells;
  for (const CellId cell_id : cells) {
    for (const CornerPlaceIndex corner :
         FindCornerPlaces(contradicting_tables, FindCellPlace(contradicting_tables, cell_id)).corners) {
      corners_of_active_cells.push_back(corner.value);
    }
  }
  EXPECT_TRUE(std::ranges::all_of(corners_of_active_cells, [corner_place_count](std::uint32_t corner) {
    return corner < corner_place_count;
  }));
  const std::vector<std::uint32_t> big_cell_corners = {0, 1, 5, 6, 25, 26, 30, 31};
  EXPECT_EQ(ToCornerValues(FindCornerPlaces(contradicting_tables, CellPlaceIndex{0}).corners), big_cell_corners);
}

}  // namespace
}  // namespace fluidity::grid
