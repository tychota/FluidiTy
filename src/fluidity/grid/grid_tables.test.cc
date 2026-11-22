#include "fluidity/grid/grid_tables.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// A frame with its origin at 0 and biggest cells 1 m wide: every width, centre and position in it is a double with
// few significant bits, so the tests compare them exactly.
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

// Hands each of the 200 random grids, with its checker and its list, to read_grid(tables, checker, cells), under a
// trace that names the grid's entry. A grid that failed to build fails the calling test, and no grid is handed on.
template <typename ReadGrid>
void ApplyToRandomGrids(const ReadGrid& read_grid) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_EQ(built.grids.size(), kRandomGridCount);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid);
    read_grid(
        built.grids.at(grid).grid.tables(RunOn::kCpu),
        built.checkers.at(grid),
        BuildShuffledRandomGrids().at(grid)
    );
  }
}

// The face places of a grid that hold a face piece, in place order.
std::vector<FacePlaceIndex> FindFacePiecePlaces(const GridTables& tables) {
  std::vector<FacePlaceIndex> face_piece_places;
  const std::uint64_t face_place_count = std::uint64_t{tables.block_count} * kFacePlacesPerBlock;
  for (std::uint64_t face_place = 0; face_place < face_place_count; ++face_place) {
    if (tables.face_piece_kinds[face_place] != FacePieceKind::kNotAPiece) {
      face_piece_places.push_back({.value = static_cast<std::uint32_t>(face_place)});
    }
  }
  return face_piece_places;
}

// The width in m of a cell of the random grids' frame (1 m biggest cells, 4 size levels), from its unit box.
double ToUnitFrameCellWidth(CellId cell) {
  const UnitBox box = ToUnitBox(cell, kRandomGridSizeLevelCount);
  constexpr double kUnitWidth = 1.0 / 8.0;  // A cell of size level 3.
  return static_cast<double>(box.width) * kUnitWidth;
}

// The id of the active cell at a place, or CellId{0} for kNoIndex: what FindCellLow and FindCellHigh name.
CellId ToCellIdOrZero(const GridTables& tables, CellPlaceIndex cell) {
  return cell.value == kNoIndex ? CellId{0} : ToCellId(tables, cell);
}

// The checker's cell on one side of a face piece, or CellId{0} at the grid's edge.
CellId ToCheckerCellOrZero(const std::optional<CellId>& cell) {
  return cell.has_value() ? *cell : CellId{0};
}

static_assert(ComputeCellWidth(ToUnitFrame(16), SizeLevel{15}) == 0x1p-15);

TEST(ComputeCellWidth, SizeLevelFifteen_IsExact) {
  EXPECT_EQ(ComputeCellWidth(ToUnitFrame(16), SizeLevel{15}), 0x1p-15);
}

TEST(ComputeCellWidth, WidthNotAPowerOfTwo_IsHalvedPerSizeLevel) {
  GridFrame frame = ToUnitFrame(3);
  frame.biggest_cell_width = 3.0;
  EXPECT_EQ(ComputeCellWidth(frame, SizeLevel{0}), 3.0);
  EXPECT_EQ(ComputeCellWidth(frame, SizeLevel{2}), 0.75);
}

static_assert(ToAxis(FacePlaceIndex{79}) == Axis::kX);
static_assert(ToAxis(FacePlaceIndex{80}) == Axis::kY);
static_assert(ToAxis(FacePlaceIndex{240 + 160}) == Axis::kZ);

TEST(ToAxis, EachThirdOfABlock_IsOneAxis) {
  EXPECT_EQ(ToAxis(FacePlaceIndex{240 + 0}), Axis::kX);
  EXPECT_EQ(ToAxis(FacePlaceIndex{240 + 159}), Axis::kY);
  EXPECT_EQ(ToAxis(FacePlaceIndex{240 + 239}), Axis::kZ);
}

// What the readers give for some face pieces of a grid, each as a sorted collection, and the sum of their areas.
struct FacePieceMeasures {
  std::multiset<FacePieceKind> kinds;
  std::multiset<double> areas;
  std::multiset<double> center_distances;
  std::set<std::uint32_t> low_places;  // FindCellLow.
  std::set<SizeLevel> size_levels;  // Of the blocks that keep them.
  std::set<std::tuple<double, double, double>> positions;  // ComputeFacePosition, x, y, z.
  double area_sum = 0;
};

FacePieceMeasures ToFacePieceMeasures(const GridTables& tables, const std::vector<FacePlaceIndex>& faces) {
  FacePieceMeasures measures;
  for (const FacePlaceIndex face : faces) {
    measures.kinds.insert(tables.face_piece_kinds[face.value]);
    measures.areas.insert(ComputeFaceArea(tables, face));
    measures.center_distances.insert(ComputeCellCenterDistance(tables, face));
    measures.low_places.insert(FindCellLow(tables, face).value);
    measures.size_levels.insert(ToSizeLevel(tables, ToCellBlockIndex(face)));
    const Vec3 position = ComputeFacePosition(tables, face);
    measures.positions.insert({position.x, position.y, position.z});
    measures.area_sum += ComputeFaceArea(tables, face);
  }
  return measures;
}

// One cell in a grid of one size level: its 6 sides are face pieces at the grid's edge, 1 m2 each, half a width from
// the cell's centre.
TEST(ComputeFaceArea, OneBigCell_HasSixEdgeFacePieces) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const FacePieceMeasures measures = ToFacePieceMeasures(tables, FindFacePiecePlaces(tables));
  const std::multiset<FacePieceKind> expected_kinds = {
      FacePieceKind::kGridEdgeLow,
      FacePieceKind::kGridEdgeLow,
      FacePieceKind::kGridEdgeLow,
      FacePieceKind::kGridEdgeHigh,
      FacePieceKind::kGridEdgeHigh,
      FacePieceKind::kGridEdgeHigh,
  };
  EXPECT_EQ(measures.kinds, expected_kinds);
  EXPECT_EQ(measures.areas, std::multiset<double>({1.0, 1.0, 1.0, 1.0, 1.0, 1.0}));
  EXPECT_EQ(measures.center_distances, std::multiset<double>({0.5, 0.5, 0.5, 0.5, 0.5, 0.5}));
}

// The edge face pieces name the one cell on their inner side and no cell beyond.
TEST(FindCellLow, GridEdgeLow_IsNoIndex) {
  const std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  std::multiset<std::pair<std::uint32_t, std::uint32_t>> low_and_high_places;
  for (const FacePlaceIndex face : FindFacePiecePlaces(tables)) {
    low_and_high_places.insert({FindCellLow(tables, face).value, FindCellHigh(tables, face).value});
  }
  // 3 kGridEdgeLow pieces with the cell on their high side, 3 kGridEdgeHigh pieces with it on their low side.
  const std::multiset<std::pair<std::uint32_t, std::uint32_t>> expected_places = {
      {kNoIndex, 0},
      {kNoIndex, 0},
      {kNoIndex, 0},
      {0, kNoIndex},
      {0, kNoIndex},
      {0, kNoIndex},
  };
  EXPECT_EQ(low_and_high_places, expected_places);
}

// The X face pieces between cells of two sizes whose cell_low is the cell `big_cell`.
std::vector<FacePlaceIndex> FindQuarterPlaces(const GridTables& tables, CellId big_cell) {
  std::vector<FacePlaceIndex> quarter_places;
  for (const FacePlaceIndex face : FindFacePiecePlaces(tables)) {
    const bool is_between_x =
        tables.face_piece_kinds[face.value] == FacePieceKind::kBetweenCells && ToAxis(face) == Axis::kX;
    if (is_between_x && ToCellIdOrZero(tables, FindCellLow(tables, face)).value == big_cell.value) {
      quarter_places.push_back(face);
    }
  }
  return quarter_places;
}

// The worked example of the header: the big cell's +X side is 4 face pieces at size level 1, each a quarter of the
// side, each with the big cell as its cell_low.
TEST(FindCellLow, SplitCellBesideBigCell_HasFourQuarterFacePieces) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const std::vector<FacePlaceIndex> quarter_places = FindQuarterPlaces(tables, ToCell(0, 0, 0, 0));
  const FacePieceMeasures measures = ToFacePieceMeasures(tables, quarter_places);
  EXPECT_EQ(measures.areas, std::multiset<double>({0.25, 0.25, 0.25, 0.25}));
  EXPECT_EQ(measures.center_distances, std::multiset<double>({0.75, 0.75, 0.75, 0.75}));
  EXPECT_EQ(measures.area_sum, 1.0);
  EXPECT_EQ(measures.low_places, std::set<std::uint32_t>({0}));
  EXPECT_EQ(measures.size_levels, std::set<SizeLevel>({SizeLevel{1}}));
  const std::set<std::tuple<double, double, double>> expected_positions = {
      {1.0, 0.25, 0.25},
      {1.0, 0.25, 0.75},
      {1.0, 0.75, 0.25},
      {1.0, 0.75, 0.75},
  };
  EXPECT_EQ(measures.positions, expected_positions);
}

// The centre distance of the face piece at X place 2 of block 1 in the two-size fixture, beside its big cell, for
// biggest cells of width significand * 2^exponent: h_0 / 2 + h_0 / 4, of which the exact value is 3 significand *
// 2^(exponent - 2).
double ComputeDistanceBesideBigCell(std::uint64_t significand, int exponent) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  GridFrame frame = ToUnitFrame(2);
  frame.biggest_cell_width = std::ldexp(static_cast<double>(significand), exponent);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  EXPECT_TRUE(test_grid.has_value());
  if (!test_grid.has_value()) {
    return 0;
  }
  return ComputeCellCenterDistance(test_grid->grid.tables(RunOn::kCpu), FacePlaceIndex{kFacePlacesPerBlock + 2});
}

// h_0 = (2^51 - 1) 2^-51 m, a significand of 51 bits: 3 times it needs 53 bits, so the distance is the exact
// 3 (2^51 - 1) 2^-53 m.
TEST(ComputeCellCenterDistance, BesideABiggerCellOfA51BitWidth_IsExact) {
  constexpr std::uint64_t kSignificand = (std::uint64_t{1} << 51U) - 1;
  // 3 kSignificand < 2^53, so the conversion to double is exact.
  const double exact_distance = std::ldexp(static_cast<double>(3 * kSignificand), -53);
  EXPECT_EQ(ComputeDistanceBesideBigCell(kSignificand, -51), exact_distance);
}

// h_0 = (2^52 - 1) 2^-52 m, a significand of 52 bits, one more than the exact case: 3 times it is odd and needs 54
// bits, so the distance is not exact. The exact 3 (2^52 - 1) 2^-54 m lies halfway between two doubles; the one sum
// and the conversion of the integer 3 (2^52 - 1) to double both round it to the even one.
TEST(ComputeCellCenterDistance, BesideABiggerCellOfA52BitWidth_IsCorrectlyRounded) {
  constexpr std::uint64_t kSignificand = (std::uint64_t{1} << 52U) - 1;
  constexpr std::uint64_t kExactScaledDistance = 3 * kSignificand;
  const double distance = ComputeDistanceBesideBigCell(kSignificand, -52);
  const double nearest_distance = std::ldexp(static_cast<double>(kExactScaledDistance), -54);
  EXPECT_EQ(distance, nearest_distance);
  // Scaled by 2^54 the distance is a whole number below 2^54, held exactly by the integer.
  EXPECT_NE(static_cast<std::uint64_t>(std::ldexp(distance, 54)), kExactScaledDistance);
}

// h_0 = (2^53 - 1) 2^-53 m, a significand of 53 bits: 3 times it needs 55 bits, so the distance is not exact. It is
// the double nearest to 3 (2^53 - 1) 2^-55 m, which the conversion of the integer 3 (2^53 - 1) to double gives.
TEST(ComputeCellCenterDistance, BesideABiggerCellOfA53BitWidth_IsCorrectlyRounded) {
  constexpr std::uint64_t kSignificand = (std::uint64_t{1} << 53U) - 1;
  constexpr std::uint64_t kExactScaledDistance = 3 * kSignificand;
  const double distance = ComputeDistanceBesideBigCell(kSignificand, -53);
  const double nearest_distance = std::ldexp(static_cast<double>(kExactScaledDistance), -55);
  EXPECT_EQ(distance, nearest_distance);
  // Scaled by 2^55 the distance is a whole number below 2^55, held exactly by the integer.
  EXPECT_NE(static_cast<std::uint64_t>(std::ldexp(distance, 55)), kExactScaledDistance);
}

// X place 1 of block 0 lies between the active big cell and the split cell {1,0,0}: no face piece, so no cells and no
// distance, though an active cell lies on its low side.
TEST(FindCellLow, PlaceWithoutAFacePiece_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const FacePlaceIndex face = {.value = 1};
  ASSERT_EQ(tables.face_piece_kinds[face.value], FacePieceKind::kNotAPiece);
  EXPECT_EQ(FindCellLow(tables, face).value, kNoIndex);
  EXPECT_EQ(FindCellHigh(tables, face).value, kNoIndex);
  EXPECT_EQ(ComputeCellCenterDistance(tables, face), 0.0);
}

// Bit 63 marks a cell id; the same bits without it are no cell id, though the grid holds the cell they would name.
TEST(FindCellPlace, ValueWithoutTheCellMark_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const CellId small_cell = ToCell(1, 2, 0, 0);
  ASSERT_EQ(FindCellPlace(tables, small_cell).value, 64U + 2U);
  constexpr std::uint64_t kAllButTheCellMark = ~(std::uint64_t{1} << 63U);
  EXPECT_EQ(FindCellPlace(tables, CellId{.value = small_cell.value & kAllButTheCellMark}).value, kNoIndex);
}

// X place 0 of block 1 lies between the place x = -1 of size level 1, where no block is, and a place inside the big
// cell: no face piece, though the big cell, an active cell, covers its high side.
TEST(FindCellHigh, PlaceWithoutAFacePiece_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const FacePlaceIndex face = {.value = kFacePlacesPerBlock + 0};
  ASSERT_EQ(tables.face_piece_kinds[face.value], FacePieceKind::kNotAPiece);
  EXPECT_EQ(FindCellHigh(tables, face).value, kNoIndex);
  EXPECT_EQ(FindCellLow(tables, face).value, kNoIndex);
}

// Bits 61..62 mark a face piece id; a cell id with them set is no cell id, though its coordinates name a cell.
TEST(FindCellPlace, ValueWithAxisBits_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const CellId small_cell = ToCell(1, 2, 0, 0);
  constexpr std::uint64_t kAxisBitOne = std::uint64_t{1} << 61U;
  EXPECT_EQ(FindCellPlace(tables, CellId{.value = small_cell.value | kAxisBitOne}).value, kNoIndex);
}

// Size level 1 {4,0,0} lies in the size-level-1 block (1,0,0), which the grid does not have.
TEST(FindCellPlace, CellOfAnAbsentBlock_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  EXPECT_EQ(FindCellPlace(tables, ToCell(1, 4, 0, 0)).value, kNoIndex);
}

// The id of the -X edge face piece of the lone cell {0,0,0} with its axis bits cleared is no face piece id. The grid
// has 100 blocks, one per lone cell {4 i, 0, 0}: an axis read from bits 0 would be 255, and its face places would lie
// 85 blocks further on, among face pieces of other blocks.
TEST(FindFacePlace, IdWithoutAxisBits_IsNoIndex) {
  const std::vector<CellId> cells = BuildLoneCellsAlongX(100);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  ASSERT_EQ(tables.block_count, 100U);
  const FacePieceId face_piece = ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 0, .y = 0, .z = 0});
  ASSERT_EQ(FindFacePlace(tables, face_piece).value, 0U);
  constexpr std::uint64_t kAllButTheAxisBits = ~(std::uint64_t{3} << 61U);
  EXPECT_EQ(FindFacePlace(tables, FacePieceId{.value = face_piece.value & kAllButTheAxisBits}).value, kNoIndex);
}

// The X face piece of size level 1 at x = 6 would be the low side of a place in the absent block (1,0,0), at its local
// 2: not on the block's low side, so plane 4 of the block below does not keep it.
TEST(FindFacePlace, IdInAnAbsentBlockOffItsLowSide_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  // At local 0 of that absent block the edge face piece is kept at plane 4 of block 1.
  const FacePieceId edge_face_piece = ToFacePieceId(Axis::kX, SizeLevel{1}, {.x = 4, .y = 0, .z = 0});
  ASSERT_EQ(FindFacePlace(tables, edge_face_piece).value, kFacePlacesPerBlock + 4U);
  EXPECT_EQ(FindFacePlace(tables, ToFacePieceId(Axis::kX, SizeLevel{1}, {.x = 6, .y = 0, .z = 0})).value, kNoIndex);
}

TEST(ToSizeLevel, EachBlock_GivesTheSizeLevelOfItsId) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  ASSERT_EQ(tables.block_count, 2U);
  EXPECT_EQ(ToSizeLevel(tables, CellBlockIndex{0}), SizeLevel{0});
  EXPECT_EQ(ToSizeLevel(tables, CellBlockIndex{1}), SizeLevel{1});
}

// Size level 0 {-2^17, 0, 0} beside the 8 size-level-1 cells of {-2^17 + 1, 0, 0}: the size-level-1 block (-2^16, 0,
// 0) has its local x = 0 at -2^18, outside the range, inside the big cell. No id is made for that place.
TEST(ToCellId, PlaceAtTheLowLimitInsideABiggerCell_IsZero) {
  constexpr std::int32_t kBigX = -131072;  // -2^17.
  std::vector<CellId> cells = {ToCell(0, kBigX, 0, 0)};
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(ToCell(0, kBigX + 1, 0, 0), cell_eighth));
  }
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const CellBlockIndex small_block = FindCellBlock(tables, ToCellBlockId(ToCell(1, kBigX * 2 + 2, 0, 0)));
  ASSERT_NE(small_block.value, kNoIndex);
  const CellPlaceIndex limit_place = {.value = small_block.value * kCellPlacesPerBlock};
  EXPECT_EQ(tables.cell_states[limit_place.value], CellState::kInsideBiggerCell);
  EXPECT_EQ(ToCellId(tables, limit_place).value, 0U);
  // The face piece between the big cell and the small cell 2^18 + 2 names the big cell on its low side.
  const FacePlaceIndex between_face = {.value = (small_block.value * kFacePlacesPerBlock) + 2};
  EXPECT_EQ(tables.face_piece_kinds[between_face.value], FacePieceKind::kBetweenCells);
  EXPECT_EQ(ToCellIdOrZero(tables, FindCellLow(tables, between_face)).value, ToCell(0, kBigX, 0, 0).value);
}

// The blocks of a grid that FindCellBlock does not find at their own index.
std::size_t CountMisplacedBlocks(const GridTables& tables) {
  std::size_t misplaced_count = 0;
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    misplaced_count += FindCellBlock(tables, tables.block_ids[block]).value == block ? 0 : 1;
  }
  return misplaced_count;
}

// FindCellBlock on one grid: every block at its own index, and a block below the smallest size level nowhere.
void ExpectBlocksFound(const GridTables& tables) {
  SCOPED_TRACE("FindCellBlock");
  EXPECT_EQ(CountMisplacedBlocks(tables), 0U);
  EXPECT_EQ(FindCellBlock(tables, ToCellBlockId(ToCell(kRandomGridSizeLevelCount, 0, 0, 0))).value, kNoIndex);
}

// What the round trip of cell ids found in one grid.
struct CellRoundTrips {
  std::size_t wrong_active_cell_count = 0;  // Listed cells not found at a kActive place with their own id.
  std::size_t wrong_place_count = 0;  // kActive or kSplit places whose id does not give the place back.
  std::size_t active_place_count = 0;
};

CellRoundTrips CountCellRoundTrips(const GridTables& tables, const std::vector<CellId>& cells) {
  CellRoundTrips round_trips;
  for (const CellId cell : cells) {
    const CellPlaceIndex place = FindCellPlace(tables, cell);
    const bool is_found = place.value != kNoIndex && tables.cell_states[place.value] == CellState::kActive;
    round_trips.wrong_active_cell_count += is_found && ToCellId(tables, place).value == cell.value ? 0 : 1;
  }
  const std::uint64_t cell_place_count = std::uint64_t{tables.block_count} * kCellPlacesPerBlock;
  for (std::uint64_t place_value = 0; place_value < cell_place_count; ++place_value) {
    const CellPlaceIndex place = {.value = static_cast<std::uint32_t>(place_value)};
    const CellState state = tables.cell_states[place_value];
    round_trips.active_place_count += state == CellState::kActive ? 1 : 0;
    const bool holds_cell = state == CellState::kActive || state == CellState::kSplit;
    const bool is_round_trip = FindCellPlace(tables, ToCellId(tables, place)).value == place.value;
    round_trips.wrong_place_count += holds_cell && !is_round_trip ? 1 : 0;
  }
  return round_trips;
}

// FindCellPlace and ToCellId on one grid: every listed cell is found at a kActive place with its own id, and every
// kActive or kSplit place round-trips.
void ExpectCellRoundTrips(const GridTables& tables, const std::vector<CellId>& cells) {
  SCOPED_TRACE("FindCellPlace, ToCellId");
  const CellRoundTrips round_trips = CountCellRoundTrips(tables, cells);
  EXPECT_EQ(round_trips.wrong_active_cell_count, 0U);
  EXPECT_EQ(round_trips.wrong_place_count, 0U);
  EXPECT_EQ(round_trips.active_place_count, cells.size());
}

// The places of a grid that hold no cell but give an id other than 0.
std::size_t CountNonzeroIdsWithoutACell(const GridTables& tables) {
  std::size_t nonzero_count = 0;
  const std::uint64_t cell_place_count = std::uint64_t{tables.block_count} * kCellPlacesPerBlock;
  for (std::uint64_t place_value = 0; place_value < cell_place_count; ++place_value) {
    const CellState state = tables.cell_states[place_value];
    const bool holds_cell = state == CellState::kActive || state == CellState::kSplit;
    const CellPlaceIndex place = {.value = static_cast<std::uint32_t>(place_value)};
    nonzero_count += !holds_cell && ToCellId(tables, place).value != 0 ? 1 : 0;
  }
  return nonzero_count;
}

// The small cells of eighth 7 of the listed cells of size level 0 to 2 that FindCellPlace finds: they lie inside an
// active cell, so none is a cell of the grid.
std::size_t CountFoundSmallCellsOfActiveCells(const GridTables& tables, const std::vector<CellId>& cells) {
  std::size_t found_count = 0;
  for (const CellId cell : cells) {
    const bool can_split = static_cast<int>(ToSizeLevel(cell)) < kRandomGridSizeLevelCount - 1;
    found_count += can_split && FindCellPlace(tables, ComputeSmallCell(cell, 7)).value != kNoIndex ? 1 : 0;
  }
  return found_count;
}

// ToCellId and FindCellPlace on one grid: a place inside a bigger cell or absent gives CellId{0}, and the id of a cell
// that is not in the grid is not found.
void ExpectNoIdsWithoutACell(const GridTables& tables, const std::vector<CellId>& cells) {
  SCOPED_TRACE("ToCellId of a place without a cell");
  EXPECT_EQ(CountNonzeroIdsWithoutACell(tables), 0U);
  EXPECT_EQ(CountFoundSmallCellsOfActiveCells(tables, cells), 0U);
  EXPECT_EQ(FindCellPlace(tables, CellId{0}).value, kNoIndex);
}

// The checker's face pieces not found at a face place whose id is the checker's.
std::size_t CountWrongFacePiecesFound(const GridTables& tables, const GeometricGrid& checker) {
  std::size_t wrong_count = 0;
  for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
    const FacePlaceIndex face = FindFacePlace(tables, face_piece.id);
    const bool is_found = face.value != kNoIndex;
    wrong_count += is_found && ToFacePieceId(tables, face).value == face_piece.id.value ? 0 : 1;
  }
  return wrong_count;
}

// The face pieces of a grid whose id does not give their place back.
std::size_t CountFacePlacesNotRoundTripping(const GridTables& tables, const std::vector<FacePlaceIndex>& faces) {
  std::size_t wrong_count = 0;
  for (const FacePlaceIndex face : faces) {
    wrong_count += FindFacePlace(tables, ToFacePieceId(tables, face)).value == face.value ? 0 : 1;
  }
  return wrong_count;
}

// FindFacePlace and ToFacePieceId on one grid: every checker face piece is found at a place whose id is the checker's,
// every face piece place round-trips, and no other place holds a face piece.
void ExpectFacePieceRoundTrips(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("FindFacePlace, ToFacePieceId");
  const std::vector<FacePlaceIndex> face_piece_places = FindFacePiecePlaces(tables);
  EXPECT_EQ(CountWrongFacePiecesFound(tables, checker), 0U);
  EXPECT_EQ(CountFacePlacesNotRoundTripping(tables, face_piece_places), 0U);
  EXPECT_EQ(face_piece_places.size(), checker.FacePieces().size());
}

// Values drawn around the ids of a grid, the same on every machine: each id of `ids` with one of its 64 bits flipped,
// drawn from std::mt19937_64, whose raw output the C++ standard fixes, and as many raw 64-bit numbers.
std::vector<std::uint64_t> BuildValuesAroundIds(const std::vector<std::uint64_t>& ids, std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::vector<std::uint64_t> values;
  values.reserve(2 * ids.size());
  for (const std::uint64_t id : ids) {
    values.push_back(id ^ (std::uint64_t{1} << ToUniformIndex(generator(), 64)));
    values.push_back(generator());
  }
  return values;
}

// What a lookup made of fuzzed values: how many it accepted, and how many of those did not name their place back.
struct FuzzedLookups {
  std::size_t accepted_count = 0;
  std::size_t wrong_count = 0;
};

FuzzedLookups CountFuzzedCellLookups(const GridTables& tables, const std::vector<CellId>& cells, std::uint64_t seed) {
  std::vector<std::uint64_t> ids;
  ids.reserve(cells.size());
  for (const CellId cell : cells) {
    ids.push_back(cell.value);
  }
  FuzzedLookups lookups;
  for (const std::uint64_t value : BuildValuesAroundIds(ids, seed)) {
    const CellPlaceIndex place = FindCellPlace(tables, CellId{.value = value});
    if (place.value != kNoIndex) {
      ++lookups.accepted_count;
      lookups.wrong_count += ToCellId(tables, place).value == value ? 0 : 1;
    }
  }
  return lookups;
}

// FindCellPlace on fuzzed values of one grid: any value it accepts gives that same value back through ToCellId. Values
// with one bit flipped cover a cleared bit 63, set axis bits, absent blocks and places without a cell. Gives the count
// of accepted values.
std::size_t ExpectFuzzedCellLookups(const GridTables& tables, const std::vector<CellId>& cells, std::uint64_t seed) {
  SCOPED_TRACE("FindCellPlace on fuzzed values");
  const FuzzedLookups lookups = CountFuzzedCellLookups(tables, cells, seed);
  EXPECT_EQ(lookups.wrong_count, 0U);
  return lookups.accepted_count;
}

FuzzedLookups CountFuzzedFaceLookups(const GridTables& tables, const GeometricGrid& checker, std::uint64_t seed) {
  std::vector<std::uint64_t> ids;
  ids.reserve(checker.FacePieces().size());
  for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
    ids.push_back(face_piece.id.value);
  }
  FuzzedLookups lookups;
  for (const std::uint64_t value : BuildValuesAroundIds(ids, seed)) {
    const FacePlaceIndex face = FindFacePlace(tables, FacePieceId{.value = value});
    if (face.value != kNoIndex) {
      ++lookups.accepted_count;
      lookups.wrong_count += ToFacePieceId(tables, face).value == value ? 0 : 1;
    }
  }
  return lookups;
}

// FindFacePlace on fuzzed values of one grid: any value it accepts gives that same value back through ToFacePieceId.
// Values with one bit flipped cover a set bit 63, wrong axis bits, absent blocks, places off a block's low side and
// places without a face piece. Gives the count of accepted values.
std::size_t ExpectFuzzedFaceLookups(const GridTables& tables, const GeometricGrid& checker, std::uint64_t seed) {
  SCOPED_TRACE("FindFacePlace on fuzzed values");
  const FuzzedLookups lookups = CountFuzzedFaceLookups(tables, checker, seed);
  EXPECT_EQ(lookups.wrong_count, 0U);
  return lookups.accepted_count;
}

// Any value a lookup accepts names its place back, over the 200 random grids: one build of the grids for both lookups.
TEST(GridTables, RandomGrids_AnyAcceptedValueNamesItsPlaceBack) {
  std::size_t accepted_cell_count = 0;
  std::size_t accepted_face_count = 0;
  std::uint64_t seed = 0;
  ApplyToRandomGrids([&](const GridTables& tables, const GeometricGrid& checker, const std::vector<CellId>& cells) {
    ++seed;
    accepted_cell_count += ExpectFuzzedCellLookups(tables, cells, seed);
    accepted_face_count += ExpectFuzzedFaceLookups(tables, checker, seed + 1000);
  });
  // Some flipped values name other cells and face pieces of the grids, so the round trips were tried.
  EXPECT_GT(accepted_cell_count, 0U);
  EXPECT_GT(accepted_face_count, 0U);
}

// The plane 0..4 of a face place along its axis, by the layout of GridTables.
std::uint32_t ToPlane(FacePlaceIndex face) {
  const std::uint32_t place_on_axis = (face.value % kFacePlacesPerBlock) % 80;
  if (ToAxis(face) == Axis::kX) {
    return place_on_axis % 5;
  }
  if (ToAxis(face) == Axis::kY) {
    return (place_on_axis / 4) % 5;
  }
  return place_on_axis / 16;
}

// The id a face place without a face piece would have, by the test's own arithmetic: its axis, its block's size level
// and the coordinate of the cell place on its high side; FacePieceId{0} when that coordinate is outside the range.
FacePieceId ToIdOfFacePlace(const GridTables& tables, FacePlaceIndex face) {
  const std::uint32_t block = face.value / kFacePlacesPerBlock;
  const auto place_on_axis = static_cast<std::int32_t>((face.value % kFacePlacesPerBlock) % 80);
  const Axis axis = ToAxis(face);
  std::int32_t local_x = place_on_axis % 4;
  std::int32_t local_y = (place_on_axis / 4) % 4;
  std::int32_t local_z = place_on_axis / 16;
  if (axis == Axis::kX) {
    local_x = place_on_axis % 5;
    local_y = (place_on_axis / 5) % 4;
    local_z = place_on_axis / 20;
  } else if (axis == Axis::kY) {
    local_y = (place_on_axis / 4) % 5;
    local_z = place_on_axis / 20;
  }
  const CellId block_as_cell = {.value = tables.block_ids[block].value};
  const CellCoord block_coord = ToCellCoord(block_as_cell);
  const CellCoord high_coord = {
      .x = (4 * block_coord.x) + local_x,
      .y = (4 * block_coord.y) + local_y,
      .z = (4 * block_coord.z) + local_z,
  };
  if (high_coord.x < -kMaxCoord || high_coord.y < -kMaxCoord || high_coord.z < -kMaxCoord) {
    return FacePieceId{0};
  }
  return ToFacePieceId(axis, ToSizeLevel(block_as_cell), high_coord);
}

// What the places without a face piece of one grid gave.
struct PlacesWithoutAFacePiece {
  std::size_t nonzero_id_count = 0;  // ToFacePieceId gave an id other than 0.
  std::size_t found_here_count = 0;  // FindFacePlace of the place's own id gave the place itself.
  std::size_t found_elsewhere_count = 0;  // It gave another place.
  // It gave another place that is not plane 0 above the place's plane 4.
  std::size_t found_not_above_plane_four_count = 0;
};

PlacesWithoutAFacePiece CountPlacesWithoutAFacePiece(const GridTables& tables) {
  PlacesWithoutAFacePiece counts;
  const std::uint64_t face_place_count = std::uint64_t{tables.block_count} * kFacePlacesPerBlock;
  for (std::uint64_t face_place = 0; face_place < face_place_count; ++face_place) {
    if (tables.face_piece_kinds[face_place] != FacePieceKind::kNotAPiece) {
      continue;
    }
    const FacePlaceIndex face = {.value = static_cast<std::uint32_t>(face_place)};
    counts.nonzero_id_count += ToFacePieceId(tables, face).value == 0 ? 0 : 1;
    const FacePlaceIndex found = FindFacePlace(tables, ToIdOfFacePlace(tables, face));
    const bool is_elsewhere = found.value != kNoIndex && found.value != face.value;
    const bool is_above_plane_four = ToPlane(face) == 4 && ToPlane(found) == 0;
    counts.found_here_count += found.value == face.value ? 1 : 0;
    counts.found_elsewhere_count += is_elsewhere ? 1 : 0;
    counts.found_not_above_plane_four_count += is_elsewhere && !is_above_plane_four ? 1 : 0;
  }
  return counts;
}

// ToFacePieceId and FindFacePlace on the places of one grid without a face piece: the id 0, and their own id names no
// face piece, except on plane 4 when the block above keeps the face piece of that id at its plane 0. Gives the count of
// such plane-4 places.
std::size_t ExpectPlacesWithoutAFacePiece(const GridTables& tables) {
  SCOPED_TRACE("ToFacePieceId of a place without a face piece");
  const PlacesWithoutAFacePiece counts = CountPlacesWithoutAFacePiece(tables);
  EXPECT_EQ(counts.nonzero_id_count, 0U);
  EXPECT_EQ(counts.found_here_count, 0U);
  EXPECT_EQ(counts.found_not_above_plane_four_count, 0U);
  EXPECT_EQ(FindFacePlace(tables, FacePieceId{0}).value, kNoIndex);
  return counts.found_elsewhere_count;
}

// The ids of cells, blocks and face pieces round-trip through their places over the 200 random grids, and places
// without a cell or a face piece give no id: one build of the grids for every lookup.
TEST(GridTables, RandomGrids_EveryIdRoundTripsThroughItsPlace) {
  std::size_t plane_four_below_a_face_piece_count = 0;
  ApplyToRandomGrids([&](const GridTables& tables, const GeometricGrid& checker, const std::vector<CellId>& cells) {
    ExpectBlocksFound(tables);
    ExpectCellRoundTrips(tables, cells);
    ExpectNoIdsWithoutACell(tables, cells);
    ExpectFacePieceRoundTrips(tables, checker);
    plane_four_below_a_face_piece_count += ExpectPlacesWithoutAFacePiece(tables);
  });
  // Plane 4 below a face piece kept at plane 0 above occurs, so that exception was seen.
  EXPECT_GT(plane_four_below_a_face_piece_count, 0U);
}

// The position the checker gives a face piece: the high cell's coordinate times the width along the axis, and half a
// width more across.
Vec3 ToCheckerFacePosition(const GeometricFacePiece& face_piece) {
  const double width = 1.0 / static_cast<double>(std::uint32_t{1} << static_cast<unsigned>(face_piece.size_level));
  return {
      .x = (face_piece.cell_high_coord.x + (face_piece.axis == Axis::kX ? 0.0 : 0.5)) * width,
      .y = (face_piece.cell_high_coord.y + (face_piece.axis == Axis::kY ? 0.0 : 0.5)) * width,
      .z = (face_piece.cell_high_coord.z + (face_piece.axis == Axis::kZ ? 0.0 : 0.5)) * width,
  };
}

bool ComputeIsSameVec3(Vec3 first, Vec3 second) {
  return first.x == second.x && first.y == second.y && first.z == second.z;
}

// The checker's face pieces whose found place fails `is_right(tables, face, face_piece)`, or that are not found.
template <typename IsRight>
std::size_t CountWrongFacePieces(const GridTables& tables, const GeometricGrid& checker, const IsRight& is_right) {
  std::size_t wrong_count = 0;
  for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
    const FacePlaceIndex face = FindFacePlace(tables, face_piece.id);
    wrong_count += face.value != kNoIndex && is_right(face, face_piece) ? 0 : 1;
  }
  return wrong_count;
}

// ComputeFacePosition on one grid: the centre of every face piece is on the plane of its axis, in the middle of the
// side of its smaller cell.
void ExpectFacePositions(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("ComputeFacePosition");
  const std::size_t wrong_count =
      CountWrongFacePieces(tables, checker, [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ComputeIsSameVec3(ComputeFacePosition(tables, face), ToCheckerFacePosition(face_piece));
      });
  EXPECT_EQ(wrong_count, 0U);
}

// ComputeFaceArea on one grid: the checker's area of every face piece.
void ExpectFaceAreas(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("ComputeFaceArea");
  const std::size_t wrong_count =
      CountWrongFacePieces(tables, checker, [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ComputeFaceArea(tables, face) == face_piece.area;
      });
  EXPECT_EQ(wrong_count, 0U);
}

// The checker's face pieces whose cell on one side is one size level bigger than the face piece.
std::size_t CountBiggerCellsBeside(const GeometricGrid& checker, bool (*is_bigger)(const GeometricFacePiece&)) {
  std::size_t bigger_count = 0;
  for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
    bigger_count += is_bigger(face_piece) ? 1 : 0;
  }
  return bigger_count;
}

bool ComputeIsLowCellBigger(const GeometricFacePiece& face_piece) {
  return face_piece.cell_low.has_value() && ToSizeLevel(*face_piece.cell_low) != face_piece.size_level;
}

bool ComputeIsHighCellBigger(const GeometricFacePiece& face_piece) {
  return face_piece.cell_high.has_value() && ToSizeLevel(*face_piece.cell_high) != face_piece.size_level;
}

// FindCellLow on one grid: the checker's cell_low of every face piece. Gives the count of bigger low cells.
std::size_t ExpectCellLows(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("FindCellLow");
  const std::size_t wrong_count =
      CountWrongFacePieces(tables, checker, [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ToCellIdOrZero(tables, FindCellLow(tables, face)).value
            == ToCheckerCellOrZero(face_piece.cell_low).value;
      });
  EXPECT_EQ(wrong_count, 0U);
  return CountBiggerCellsBeside(checker, ComputeIsLowCellBigger);
}

// FindCellHigh on one grid: the checker's cell_high of every face piece. Gives the count of bigger high cells.
std::size_t ExpectCellHighs(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("FindCellHigh");
  const std::size_t wrong_count =
      CountWrongFacePieces(tables, checker, [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ToCellIdOrZero(tables, FindCellHigh(tables, face)).value
            == ToCheckerCellOrZero(face_piece.cell_high).value;
      });
  EXPECT_EQ(wrong_count, 0U);
  return CountBiggerCellsBeside(checker, ComputeIsHighCellBigger);
}

// The checker's distance between the centres of a face piece's cells: half the width of each cell it has.
double ToCheckerCenterDistance(const GeometricFacePiece& face_piece) {
  const double low_half_width = face_piece.cell_low.has_value() ? ToUnitFrameCellWidth(*face_piece.cell_low) / 2 : 0;
  const double high_half_width = face_piece.cell_high.has_value() ? ToUnitFrameCellWidth(*face_piece.cell_high) / 2 : 0;
  return low_half_width + high_half_width;
}

// ComputeCellCenterDistance on one grid: half the width of each cell beside every face piece, along its axis; half a
// width at the grid's edge.
void ExpectCenterDistances(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("ComputeCellCenterDistance");
  const std::size_t wrong_count =
      CountWrongFacePieces(tables, checker, [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ComputeCellCenterDistance(tables, face) == ToCheckerCenterDistance(face_piece);
      });
  EXPECT_EQ(wrong_count, 0U);
}

// The middle of a cell's unit box, in m, in the random grids' frame.
Vec3 ToBoxCenter(CellId cell) {
  constexpr double kUnitWidth = 1.0 / 8.0;  // A cell of size level 3.
  const UnitBox box = ToUnitBox(cell, kRandomGridSizeLevelCount);
  const double half_width = static_cast<double>(box.width) * kUnitWidth / 2;
  return {
      .x = (static_cast<double>(box.low.at(0)) * kUnitWidth) + half_width,
      .y = (static_cast<double>(box.low.at(1)) * kUnitWidth) + half_width,
      .z = (static_cast<double>(box.low.at(2)) * kUnitWidth) + half_width,
  };
}

// The listed cells not found, or whose place has a centre other than the middle of their unit box.
std::size_t CountWrongCellCenters(const GridTables& tables, const std::vector<CellId>& cells) {
  std::size_t wrong_count = 0;
  for (const CellId cell : cells) {
    const CellPlaceIndex place = FindCellPlace(tables, cell);
    const bool is_found = place.value != kNoIndex;
    wrong_count += is_found && ComputeIsSameVec3(ComputeCellCenter(tables, place), ToBoxCenter(cell)) ? 0 : 1;
  }
  return wrong_count;
}

// ComputeCellCenter on one grid: the centre of every active cell is the middle of its unit box.
void ExpectCellCenters(const GridTables& tables, const std::vector<CellId>& cells) {
  SCOPED_TRACE("ComputeCellCenter");
  EXPECT_EQ(CountWrongCellCenters(tables, cells), 0U);
}

// Every reader of positions, areas, distances and the cells beside a face piece matches the geometric checker over the
// 200 random grids, exactly: one build of the grids for every reader.
TEST(GridTables, RandomGrids_EveryReaderMatchesGeometricChecker) {
  std::size_t bigger_low_count = 0;
  std::size_t bigger_high_count = 0;
  ApplyToRandomGrids([&](const GridTables& tables, const GeometricGrid& checker, const std::vector<CellId>& cells) {
    ExpectCellCenters(tables, cells);
    ExpectFacePositions(tables, checker);
    ExpectFaceAreas(tables, checker);
    bigger_low_count += ExpectCellLows(tables, checker);
    bigger_high_count += ExpectCellHighs(tables, checker);
    ExpectCenterDistances(tables, checker);
  });
  // A bigger cell on each side occurs, so the way through the bigger block was compared.
  EXPECT_GT(bigger_low_count, 0U);
  EXPECT_GT(bigger_high_count, 0U);
}

// The frame's origin moves every centre and every face position by the same vector.
TEST(ComputeCellCenter, MovedOrigin_MovesTheCentre) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  GridFrame frame = ToUnitFrame(2);
  frame.origin = {.x = -2.0, .y = 0.5, .z = 8.0};
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(frame, cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  const Vec3 big_center = ComputeCellCenter(tables, FindCellPlace(tables, ToCell(0, 0, 0, 0)));
  EXPECT_EQ(big_center.x, -1.5);
  EXPECT_EQ(big_center.y, 1.0);
  EXPECT_EQ(big_center.z, 8.5);
  const Vec3 small_center = ComputeCellCenter(tables, FindCellPlace(tables, ToCell(1, 3, 1, 0)));
  EXPECT_EQ(small_center.x, -0.25);
  EXPECT_EQ(small_center.y, 1.25);
  EXPECT_EQ(small_center.z, 8.25);
  const Vec3 face_position = ComputeFacePosition(
      tables,
      FindFacePlace(tables, ToFacePieceId(Axis::kY, SizeLevel{1}, {.x = 3, .y = 1, .z = 0}))
  );
  EXPECT_EQ(face_position.x, -0.25);
  EXPECT_EQ(face_position.y, 1.0);
  EXPECT_EQ(face_position.z, 8.25);
}

}  // namespace
}  // namespace fluidity::grid
