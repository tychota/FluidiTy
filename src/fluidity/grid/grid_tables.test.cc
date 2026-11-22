#include "fluidity/grid/grid_tables.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
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
#include "fluidity/grid/testing/grid_tables_cases.h"

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

// The inputs of a CPU list of version 1.
GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
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
