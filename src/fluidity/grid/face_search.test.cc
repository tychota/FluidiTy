#include "fluidity/grid/face_search.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <set>
#include <span>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {
namespace {

// A default-made result is an empty list whose every entry is kNoIndex.
static_assert(CellsNearFace().count == 0 && CellsNearFace().cells[0].value == kNoIndex);

static_assert(CellsNearFace().cells[kMaxCellsNearFace - 1].value == kNoIndex);

static_assert(FaceSearch().block_count == 0 && FaceSearch().face.value == kNoIndex);

static_assert(FaceSearch().blocks[kMaxFaceSearchBlocks - 1].value == kNoIndex);

// The worked example of FindCellsNearFace: in the two-size fixture, the face piece between the big cell and the small
// cell {2,0,0}, X place 2 of block 1, has the big cell, the small cell and its 7 siblings near it: all 9 cells.
TEST(FindCellsNearFace, TwoSizeFixture_GivesEveryCell) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const CellsNearFace cells_near_face = FindCellsNearFace(ToTablesOf(test_grid), FacePlaceIndex{240 + 2});
  const std::vector<std::uint32_t> expected_places =
      {0, 64 + 2, 64 + 3, 64 + 6, 64 + 7, 64 + 18, 64 + 19, 64 + 22, 64 + 23};
  EXPECT_EQ(ToListedPlaces(cells_near_face.cells, cells_near_face.count), expected_places);
  EXPECT_TRUE(ComputeHasNoIndexPastCount(cells_near_face.cells, cells_near_face.count));
}

// The largest count of cells near a face known: 82, at the face piece between the two cells of
// BuildSameSizePairWithEveryNeighbourSplit, and no face piece of that grid gives more.
TEST(FindCellsNearFace, SameSizePairWithEveryNeighbourSplit_Gives82) {
  const std::vector<CellId> cells = BuildSameSizePairWithEveryNeighbourSplit();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const FacePlaceIndex face = FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 1, .y = 0, .z = 0}));
  ASSERT_NE(face.value, kNoIndex);
  const int pair_count = FindCellsNearFace(tables, face).count;
  RecordProperty("largest_cells_near_face_known", pair_count);
  EXPECT_EQ(pair_count, 82);
  EXPECT_EQ(ComputeLargestCellsNearFace(tables), 82);
  EXPECT_LE(pair_count, kMaxCellsNearFace);
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

// The cells near a face of a list stopped at kMaxCellsNearFace, and how many different cells there are in all.
struct StoppedCellsNearFace {
  std::vector<std::uint32_t> stopped_places;  // Ascending.
  std::size_t every_count = 0;
};

// Finds the cells a list stopped at kMaxCellsNearFace holds, in ascending order: cell_low, cell_high, every touching
// cell of cell_low and those of cell_high but the last 2 in its order; and the count of every different cell.
StoppedCellsNearFace FindCellsNearFaceStoppedAtBound(
    CellPlaceIndex cell_low,
    CellPlaceIndex cell_high,
    const std::array<TouchingCells, 2>& touching_of_low_and_high
) {
  std::set<std::uint32_t> every_place = {cell_low.value, cell_high.value};
  std::set<std::uint32_t> stopped_places = every_place;
  const TouchingCells& low_touching = touching_of_low_and_high.at(0);
  const TouchingCells& high_touching = touching_of_low_and_high.at(1);
  for (int entry = 0; entry < low_touching.count; ++entry) {
    every_place.insert(low_touching.cells[entry].value);
    stopped_places.insert(low_touching.cells[entry].value);
  }
  for (int entry = 0; entry < high_touching.count; ++entry) {
    every_place.insert(high_touching.cells[entry].value);
    if (entry < high_touching.count - 2) {
      stopped_places.insert(high_touching.cells[entry].value);
    }
  }
  return {.stopped_places = {stopped_places.begin(), stopped_places.end()}, .every_count = every_place.size()};
}

// Two far cells, each with its 26 neighbours split, 2 size levels: size level 0 {1,1,1} and {100,2,2}.
std::vector<CellId> BuildTwoFarCellsWithEveryNeighbourSplit() {
  std::vector<CellId> cells = BuildCellWithEveryNeighbourSplit(1, 1, 1);
  const std::vector<CellId> far_cells = BuildCellWithEveryNeighbourSplit(100, 2, 2);
  cells.insert(cells.end(), far_cells.begin(), far_cells.end());
  return cells;
}

// Makes the copied tables of BuildTwoFarCellsWithEveryNeighbourSplit contradict themselves. In the size-level-1 block
// (0,0,0), whose places (2..3, 2..3, 2..3) lie inside {1,1,1}, the X place at plane 4, local (4, 2, 2), is marked
// kBetweenCells, and the block's +X neighbour link is pointed at the size-level-0 block (25,0,0), where {100,2,2} sits
// at local (0, 2, 2). Gives that face place; kNoIndex, with a failure, when a block is missing.
FacePlaceIndex ApplyFarCellLink(CopiedTables& contradicting) {
  const CellBlockIndex small_block = FindCellBlock(contradicting.tables, ToCellBlockId(ToCell(1, 0, 0, 0)));
  const CellBlockIndex far_block = FindCellBlock(contradicting.tables, ToCellBlockId(ToCell(0, 100, 0, 0)));
  if (small_block.value == kNoIndex || far_block.value == kNoIndex) {
    ADD_FAILURE() << "a block of the two far cells is missing";
    return {.value = kNoIndex};
  }
  // X place (z 4 + y) 5 + x of local (4, 2, 2).
  const FacePlaceIndex face = {.value = (small_block.value * kFacePlacesPerBlock) + ((((2 * 4) + 2) * 5) + 4)};
  EXPECT_EQ(contradicting.face_piece_kinds.at(face.value), FacePieceKind::kNotAPiece);
  contradicting.face_piece_kinds.at(face.value) = FacePieceKind::kBetweenCells;
  constexpr std::size_t kHighXEntry = 13 + 1;
  contradicting.same_size_level_blocks.at((27 * std::size_t{small_block.value}) + kHighXEntry) = far_block.value;
  return face;
}

// Tables that contradict themselves past the bound (ApplyFarCellLink): cell_low is {1,1,1}, through the bigger block,
// and cell_high is {100,2,2}: 2 cells whose 56 touching cells each are all different, 114 in all. The list stops at
// 112 and leaves out the last 2 touching cells of cell_high.
TEST(FindCellsNearFace, ContradictingTablesPastTheBound_StopAt112) {
  const std::vector<CellId> cells = BuildTwoFarCellsWithEveryNeighbourSplit();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  CopiedTables contradicting;
  CopyTables(ToTablesOf(test_grid), contradicting);
  const FacePlaceIndex face = ApplyFarCellLink(contradicting);
  ASSERT_NE(face.value, kNoIndex);
  const CellPlaceIndex cell_low = FindCellPlace(contradicting.tables, ToCell(0, 1, 1, 1));
  const CellPlaceIndex cell_high = FindCellPlace(contradicting.tables, ToCell(0, 100, 2, 2));
  ASSERT_EQ(FindCellLow(contradicting.tables, face).value, cell_low.value);
  ASSERT_EQ(FindCellHigh(contradicting.tables, face).value, cell_high.value);
  const std::array<TouchingCells, 2> touching_of_low_and_high = {
      FindTouchingCells(contradicting.tables, cell_low),
      FindTouchingCells(contradicting.tables, cell_high),
  };
  const StoppedCellsNearFace stopped = FindCellsNearFaceStoppedAtBound(cell_low, cell_high, touching_of_low_and_high);
  ASSERT_EQ(stopped.every_count, 114U);
  const CellsNearFace cells_near_face = FindCellsNearFace(contradicting.tables, face);
  EXPECT_EQ(cells_near_face.count, kMaxCellsNearFace);
  EXPECT_EQ(ToListedPlaces(cells_near_face.cells, cells_near_face.count), stopped.stopped_places);
}

// A place without a face piece, an index of no place and the tables of no grid give count 0, every entry kNoIndex.
// In the two-size fixture, X place 1 of block 0 lies between the big cell and the split cell: kNotAPiece.
TEST(FindCellsNearFace, PlaceWithoutFacePiece_HasNone) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const CellsNearFace cells_near_face = FindCellsNearFace(ToTablesOf(test_grid), FacePlaceIndex{1});
  EXPECT_EQ(cells_near_face.count, 0);
  EXPECT_TRUE(ComputeHasNoIndexPastCount(cells_near_face.cells, 0));
}

// kNoIndex and a place of block 2, one past the two-size fixture's 2 blocks, name no place: nothing is read there.
TEST(FindCellsNearFace, IndexOfNoPlace_HasNone) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  for (const FacePlaceIndex face : {FacePlaceIndex{kNoIndex}, FacePlaceIndex{2 * kFacePlacesPerBlock}}) {
    SCOPED_TRACE(testing::Message() << "face " << face.value);
    const CellsNearFace cells_near_face = FindCellsNearFace(ToTablesOf(test_grid), face);
    EXPECT_EQ(cells_near_face.count, 0);
    EXPECT_TRUE(ComputeHasNoIndexPastCount(cells_near_face.cells, 0));
  }
}

// In the tables of no grid, block_count 0 and null arrays, face place 0 has no cells near it: nothing is read.
TEST(FindCellsNearFace, TablesOfNoGrid_HasNone) {
  const GridTables no_grid_tables;
  const CellsNearFace cells_near_face = FindCellsNearFace(no_grid_tables, FacePlaceIndex{0});
  EXPECT_EQ(cells_near_face.count, 0);
  EXPECT_TRUE(ComputeHasNoIndexPastCount(cells_near_face.cells, 0));
}

// The worked example of the face search: in a grid of one size level, width 1 m, radius 1.5 around an X face piece
// gives the 3 x 3 x 3 X face pieces centred on it, itself included. The box -2..2 puts the face piece at x = 0, on the
// border of blocks -1 and 0 on every axis.
TEST(FindNextFace, UniformGridRadiusOneAndAHalf_Finds27) {
  const std::vector<CellId> cells = BuildUniformBox();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const FacePlaceIndex face = FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 0, .y = 0, .z = 0}));
  ASSERT_NE(face.value, kNoIndex);
  const std::vector<std::uint32_t> walked_faces = FindWalkedFaces(tables, face, 1.5);
  std::vector<std::uint32_t> expected_faces;
  for (std::int32_t z = -1; z <= 1; ++z) {
    for (std::int32_t y = -1; y <= 1; ++y) {
      for (std::int32_t x = -1; x <= 1; ++x) {
        const FacePieceId face_piece = ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = x, .y = y, .z = z});
        expected_faces.push_back(FindFacePlace(tables, face_piece).value);
      }
    }
  }
  std::ranges::sort(expected_faces);
  EXPECT_EQ(walked_faces.size(), 27U);
  EXPECT_EQ(walked_faces, expected_faces);
}

// The face piece at plane 4 of the size-level-1 block (1,0,0), between {7,0,0} and {4,0,0}, finds the size-level-0
// face piece between {4,0,0} and {5,0,0} at radius 4: it lies in a bigger block that is the bigger block of no
// size-level-1 block of the grid.
TEST(FindNextFace, BiggerBlockBehindAbsentNeighbour_IsRead) {
  const std::vector<CellId> cells = BuildBiggerBlockBehindAbsentNeighbour();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const FacePlaceIndex face = FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{1}, {.x = 8, .y = 0, .z = 0}));
  const FacePlaceIndex bigger_face =
      FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 5, .y = 0, .z = 0}));
  ASSERT_NE(face.value, kNoIndex);
  ASSERT_NE(bigger_face.value, kNoIndex);
  // Plane 4 of its block: X place (z 4 + y) 5 + 4.
  ASSERT_EQ((face.value % kFacePlacesPerBlock) % 5, 4U);
  const std::vector<std::uint32_t> walked_faces = FindWalkedFaces(tables, face, 4.0);
  EXPECT_TRUE(std::ranges::contains(walked_faces, bigger_face.value));
}

// In a grid piece that lacks the start's bigger block, the 27 positions of size level l - 1 are found by their ids, and
// the walk is that of the whole grid. The start of BiggerBlockBehindAbsentNeighbour_IsRead loses its bigger block and
// still finds the size-level-0 face piece behind the absent neighbour.
TEST(FindNextFace, BiggerBlockNotInTables_IsFoundById) {
  const std::vector<CellId> cells = BuildBiggerBlockBehindAbsentNeighbour();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const FacePlaceIndex face = FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{1}, {.x = 8, .y = 0, .z = 0}));
  ASSERT_NE(face.value, kNoIndex);
  const std::vector<std::uint32_t> whole_grid_faces = FindWalkedFaces(tables, face, 4.0);
  CopiedTables grid_piece;
  CopyTables(tables, grid_piece);
  grid_piece.bigger_block.at(ToCellBlockIndex(face).value) = kNoIndex;
  const FacePlaceIndex bigger_face =
      FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 5, .y = 0, .z = 0}));
  EXPECT_TRUE(std::ranges::contains(whole_grid_faces, bigger_face.value));
  EXPECT_EQ(FindWalkedFaces(grid_piece.tables, face, 4.0), whole_grid_faces);
}

// Size level 0 {3,0,0} active beside size level 0 {4,0,0} split, 2 size levels: the size-level-1 cells of {4,0,0}
// lie in the size-level-1 block (2,0,0), under the size-level-0 block (1,0,0), which holds no face piece.
std::vector<CellId> BuildCellBesideSplitBlock() {
  std::vector<CellId> cells = {ToCell(0, 3, 0, 0)};
  AppendSmallCells(ToCell(0, 4, 0, 0), cells);
  return cells;
}

// In a grid piece whose link to a block of the start's size level is "none", the 8 positions of size level l + 1 under
// that position are found by their ids. From the -X side of {3,0,0}, with the link of block (0,0,0) to block (1,0,0)
// cleared, the walk still finds the size-level-1 face piece between {8,0,0} and {9,0,0}, and is that of the whole grid.
TEST(FindNextFace, SmallerBlocksUnderUnlinkedPosition_AreFoundById) {
  const std::vector<CellId> cells = BuildCellBesideSplitBlock();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const FacePlaceIndex face = FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 3, .y = 0, .z = 0}));
  const FacePlaceIndex smaller_face =
      FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{1}, {.x = 9, .y = 0, .z = 0}));
  ASSERT_NE(face.value, kNoIndex);
  ASSERT_NE(smaller_face.value, kNoIndex);
  const std::vector<std::uint32_t> whole_grid_faces = FindWalkedFaces(tables, face, 4.0);
  EXPECT_TRUE(std::ranges::contains(whole_grid_faces, smaller_face.value));
  CopiedTables grid_piece;
  CopyTables(tables, grid_piece);
  constexpr std::size_t kHighXEntry = 13 + 1;
  grid_piece.same_size_level_blocks.at((27 * std::size_t{ToCellBlockIndex(face).value}) + kHighXEntry) = kNoIndex;
  EXPECT_EQ(FindWalkedFaces(grid_piece.tables, face, 4.0), whole_grid_faces);
}

// A radius above kMaxFaceSearchRadius is the limit: radius 6 and +infinity walk the radius-4 face pieces, and the
// search keeps radius 4. In the box of 12 x 12 x 12 size-level-0 cells, radius 4 around a middle face piece gives
// 7 x 7 x 7 = 343 face pieces, where an unclamped radius 6 would give 11 x 11 x 11.
TEST(BuildFaceSearch, RadiusAboveLimit_IsTreatedAsTheLimit) {
  const std::vector<CellId> cells = BuildBoxOfCells(-6, 5);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const FacePlaceIndex face = FindFacePlace(tables, ToFacePieceId(Axis::kZ, SizeLevel{0}, {.x = 0, .y = 0, .z = 0}));
  ASSERT_NE(face.value, kNoIndex);
  const std::vector<std::uint32_t> limit_faces = FindWalkedFaces(tables, face, kMaxFaceSearchRadius);
  EXPECT_EQ(limit_faces.size(), 343U);
  for (const double radius : {6.0, std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(testing::Message() << "radius " << radius);
    EXPECT_EQ(BuildFaceSearch(tables, face, radius).radius, kMaxFaceSearchRadius);
    EXPECT_EQ(FindWalkedFaces(tables, face, radius), limit_faces);
  }
}

// A radius that is not above 0 finds nothing, not even the start: 0, -1, -infinity and NaN are all radius 0.
TEST(BuildFaceSearch, RadiusNotAboveZero_FindsNothing) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const std::array<double, 4> radii = {
      0.0,
      -1.0,
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
  };
  for (const double radius : radii) {
    SCOPED_TRACE(testing::Message() << "radius " << radius);
    EXPECT_EQ(BuildFaceSearch(tables, FacePlaceIndex{240 + 2}, radius).radius, 0.0);
    EXPECT_TRUE(FindWalkedFaces(tables, FacePlaceIndex{240 + 2}, radius).empty());
  }
}

// A start that is no face piece gives a search of no block: a place of kind kNotAPiece, kNoIndex, and a place of a
// block beyond block_count.
TEST(BuildFaceSearch, StartWithoutFacePiece_FindsNothing) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const std::array<FacePlaceIndex, 3> starts = {
      FacePlaceIndex{1},
      FacePlaceIndex{kNoIndex},
      FacePlaceIndex{2 * kFacePlacesPerBlock},
  };
  for (const FacePlaceIndex start : starts) {
    SCOPED_TRACE(testing::Message() << "start " << start.value);
    const FaceSearch search = BuildFaceSearch(ToTablesOf(test_grid), start, 4.0);
    EXPECT_EQ(search.block_count, 0);
    EXPECT_TRUE(ComputeHasNoIndexPastCount(search.blocks, 0));
    EXPECT_TRUE(FindWalkedFaces(ToTablesOf(test_grid), start, 4.0).empty());
  }
}

// In the tables of no grid, block_count 0 and null arrays, a search from face place 0 has no block and finds nothing.
TEST(BuildFaceSearch, TablesOfNoGrid_FindsNothing) {
  const GridTables no_grid_tables;
  EXPECT_EQ(BuildFaceSearch(no_grid_tables, FacePlaceIndex{0}, 4.0).block_count, 0);
  EXPECT_TRUE(FindWalkedFaces(no_grid_tables, FacePlaceIndex{0}, 4.0).empty());
}

// A null search gives kNoIndex.
TEST(FindNextFace, NullSearch_GivesNoFace) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  EXPECT_EQ(FindNextFace(ToTablesOf(test_grid), nullptr).value, kNoIndex);
}

// A finished walk gives kNoIndex on every later call.
TEST(FindNextFace, FinishedSearch_StaysFinished) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  FaceSearch search = BuildFaceSearch(tables, FacePlaceIndex{240 + 2}, 1.5);
  int found_count = 0;
  while (FindNextFace(tables, &search).value != kNoIndex && found_count <= kFacePlacesPerBlock) {
    ++found_count;
  }
  EXPECT_GT(found_count, 0);
  for (int call = 0; call < 3; ++call) {
    EXPECT_EQ(FindNextFace(tables, &search).value, kNoIndex);
  }
}

// A search with counters outside their ranges gives kNoIndex: next_block below 0, next_place below 0 or above 80.
TEST(FindNextFace, CountersOutOfRange_GiveNoFace) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  for (const std::array<int, 2> counters : {std::array{-1, 0}, std::array{0, -1}, std::array{0, 81}}) {
    SCOPED_TRACE(testing::Message() << "next_block " << counters.at(0) << ", next_place " << counters.at(1));
    FaceSearch search = BuildFaceSearch(tables, FacePlaceIndex{240 + 2}, 1.5);
    search.next_block = counters.at(0);
    search.next_place = counters.at(1);
    EXPECT_EQ(FindNextFace(tables, &search).value, kNoIndex);
  }
}

// A search whose start is no face piece of these tables gives kNoIndex, whatever blocks it lists: a search from the
// two-size fixture's face piece at X place 2 of block 1, with its start moved to X place 1 of block 0, kNotAPiece.
TEST(FindNextFace, StartNotAFacePiece_GivesNoFace) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  FaceSearch search = BuildFaceSearch(tables, FacePlaceIndex{240 + 2}, 4.0);
  ASSERT_GT(search.block_count, 0);
  search.face = FacePlaceIndex{1};
  EXPECT_EQ(FindNextFace(tables, &search).value, kNoIndex);
}

// A search that lists a block two size levels from its start's skips that block: in the three-size grid, a search from
// the big cell's -X face piece, size level 0, with every size-level-2 block added to its list, gives the face pieces of
// the search as made, none of size level 2.
TEST(FindNextFace, SearchWithBlockTwoSizeLevelsAway_SkipsIt) {
  const std::vector<CellId> cells = BuildThreeSizeCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = ToTablesOf(test_grid);
  const FacePlaceIndex face = FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 0, .y = 0, .z = 0}));
  ASSERT_NE(face.value, kNoIndex);
  const std::vector<std::uint32_t> made_faces = FindWalkedFaces(tables, face, 4.0);
  FaceSearch search = BuildFaceSearch(tables, face, 4.0);
  int added_block_count = 0;
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    if (ToSizeLevel(tables, CellBlockIndex{block}) == SizeLevel{2}) {
      search.blocks[search.block_count] = CellBlockIndex{block};
      ++search.block_count;
      ++added_block_count;
    }
  }
  ASSERT_GT(added_block_count, 0);
  EXPECT_EQ(FindGivenFaces(tables, search), made_faces);
}

// A search made for other tables reads only places of these tables: a search from the box -2..2, whose blocks go up to
// index 7, walked on the two-size fixture's 2 blocks, with its start set to a face piece there. Block indexes at or
// beyond 2 are skipped, a block_count above the array is read as 270, and every face piece given is the fixture's.
TEST(FindNextFace, SearchOfOtherTables_ReadsOnlyTheseTables) {
  const std::vector<CellId> fixture_cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> fixture_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), fixture_cells));
  ASSERT_TRUE(fixture_grid.has_value());
  const GridTables fixture_tables = ToTablesOf(fixture_grid);
  const std::vector<CellId> box_cells = BuildUniformBox();
  const std::expected<TestGrid, GridFailure> box_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), box_cells));
  ASSERT_TRUE(box_grid.has_value());
  const GridTables box_tables = ToTablesOf(box_grid);
  const FacePieceId box_face_piece = ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 0, .y = 0, .z = 0});
  FaceSearch search = BuildFaceSearch(box_tables, FindFacePlace(box_tables, box_face_piece), 4.0);
  ASSERT_GT(search.block_count, 2);
  // The start becomes a face piece of the fixture: X place 2 of block 1.
  search.face = FacePlaceIndex{240 + 2};
  search.block_count = kMaxFaceSearchBlocks + 5;
  const std::vector<std::uint32_t> given_faces = FindGivenFaces(fixture_tables, search);
  EXPECT_FALSE(given_faces.empty());
  // Read as 270, the block count gives the walk of the same search with block_count 270: nothing past `blocks` is read.
  FaceSearch capped_search = search;
  capped_search.block_count = kMaxFaceSearchBlocks;
  EXPECT_EQ(given_faces, FindGivenFaces(fixture_tables, capped_search));
  const auto is_face_piece_of_fixture = [&fixture_tables](std::uint32_t face) {
    const bool is_in_tables = face < fixture_tables.block_count * static_cast<std::uint32_t>(kFacePlacesPerBlock);
    return is_in_tables && fixture_tables.face_piece_kinds[face] != FacePieceKind::kNotAPiece;
  };
  EXPECT_TRUE(std::ranges::all_of(given_faces, is_face_piece_of_fixture));
}

}  // namespace
}  // namespace fluidity::grid
