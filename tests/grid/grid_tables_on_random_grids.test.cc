#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/grid_tables_cases.h"

namespace fluidity::grid {
namespace {

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

// The width in m of a cell of the random grids' frame (1 m biggest cells, 4 size levels), from its unit box.
double ToUnitFrameCellWidth(CellId cell) {
  const UnitBox box = ToUnitBox(cell, kRandomGridSizeLevelCount);
  constexpr double kUnitWidth = 1.0 / 8.0;  // A cell of size level 3.
  return static_cast<double>(box.width) * kUnitWidth;
}

// The checker's cell on one side of a face piece, or CellId{0} at the grid's edge.
CellId ToCheckerCellOrZero(const std::optional<CellId>& cell) {
  return cell.has_value() ? *cell : CellId{0};
}

// The blocks of a grid that FindCellBlock does not find at their own index.
std::size_t ComputeCountMisplacedBlocks(const GridTables& tables) {
  std::size_t misplaced_count = 0;
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    misplaced_count += FindCellBlock(tables, tables.block_ids[block]).value == block ? 0 : 1;
  }
  return misplaced_count;
}

// FindCellBlock on one grid: every block at its own index, and a block below the smallest size level nowhere.
void CheckBlocksFound(const GridTables& tables) {
  SCOPED_TRACE("FindCellBlock");
  EXPECT_EQ(ComputeCountMisplacedBlocks(tables), 0U);
  EXPECT_EQ(FindCellBlock(tables, ToCellBlockId(ToCell(kRandomGridSizeLevelCount, 0, 0, 0))).value, kNoIndex);
}

// What the round trip of cell ids found in one grid.
struct CellRoundTrips {
  std::size_t wrong_active_cell_count = 0;  // Listed cells not found at a kActive place with their own id.
  std::size_t wrong_place_count = 0;  // kActive or kSplit places whose id does not give the place back.
  std::size_t active_place_count = 0;
};

CellRoundTrips ComputeCountCellRoundTrips(const GridTables& tables, const std::vector<CellId>& cells) {
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
void CheckCellRoundTrips(const GridTables& tables, const std::vector<CellId>& cells) {
  SCOPED_TRACE("FindCellPlace, ToCellId");
  const CellRoundTrips round_trips = ComputeCountCellRoundTrips(tables, cells);
  EXPECT_EQ(round_trips.wrong_active_cell_count, 0U);
  EXPECT_EQ(round_trips.wrong_place_count, 0U);
  EXPECT_EQ(round_trips.active_place_count, cells.size());
}

// The places of a grid that hold no cell but give an id other than 0.
std::size_t ComputeCountNonzeroIdsWithoutACell(const GridTables& tables) {
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
std::size_t ComputeCountFoundSmallCellsOfActiveCells(const GridTables& tables, const std::vector<CellId>& cells) {
  std::size_t found_count = 0;
  for (const CellId cell : cells) {
    const bool can_split = static_cast<int>(ToSizeLevel(cell)) < kRandomGridSizeLevelCount - 1;
    found_count += can_split && FindCellPlace(tables, ComputeSmallCell(cell, 7)).value != kNoIndex ? 1 : 0;
  }
  return found_count;
}

// ToCellId and FindCellPlace on one grid: a place inside a bigger cell or absent gives CellId{0}, and the id of a cell
// that is not in the grid is not found.
void CheckNoIdsWithoutACell(const GridTables& tables, const std::vector<CellId>& cells) {
  SCOPED_TRACE("ToCellId of a place without a cell");
  EXPECT_EQ(ComputeCountNonzeroIdsWithoutACell(tables), 0U);
  EXPECT_EQ(ComputeCountFoundSmallCellsOfActiveCells(tables, cells), 0U);
  EXPECT_EQ(FindCellPlace(tables, CellId{0}).value, kNoIndex);
}

// The checker's face pieces not found at a face place whose id is the checker's.
std::size_t ComputeCountWrongFacePiecesFound(const GridTables& tables, const GeometricGrid& checker) {
  std::size_t wrong_count = 0;
  for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
    const FacePlaceIndex face = FindFacePlace(tables, face_piece.id);
    const bool is_found = face.value != kNoIndex;
    wrong_count += is_found && ToFacePieceId(tables, face).value == face_piece.id.value ? 0 : 1;
  }
  return wrong_count;
}

// The face pieces of a grid whose id does not give their place back.
std::size_t ComputeCountFacePlacesNotRoundTripping(const GridTables& tables, const std::vector<FacePlaceIndex>& faces) {
  std::size_t wrong_count = 0;
  for (const FacePlaceIndex face : faces) {
    wrong_count += FindFacePlace(tables, ToFacePieceId(tables, face)).value == face.value ? 0 : 1;
  }
  return wrong_count;
}

// FindFacePlace and ToFacePieceId on one grid: every checker face piece is found at a place whose id is the checker's,
// every face piece place round-trips, and no other place holds a face piece.
void CheckFacePieceRoundTrips(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("FindFacePlace, ToFacePieceId");
  const std::vector<FacePlaceIndex> face_piece_places = FindFacePiecePlaces(tables);
  EXPECT_EQ(ComputeCountWrongFacePiecesFound(tables, checker), 0U);
  EXPECT_EQ(ComputeCountFacePlacesNotRoundTripping(tables, face_piece_places), 0U);
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

FuzzedLookups
ComputeCountFuzzedCellLookups(const GridTables& tables, const std::vector<CellId>& cells, std::uint64_t seed) {
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
std::size_t CheckFuzzedCellLookups(const GridTables& tables, const std::vector<CellId>& cells, std::uint64_t seed) {
  SCOPED_TRACE("FindCellPlace on fuzzed values");
  const FuzzedLookups lookups = ComputeCountFuzzedCellLookups(tables, cells, seed);
  EXPECT_EQ(lookups.wrong_count, 0U);
  return lookups.accepted_count;
}

FuzzedLookups
ComputeCountFuzzedFaceLookups(const GridTables& tables, const GeometricGrid& checker, std::uint64_t seed) {
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
std::size_t CheckFuzzedFaceLookups(const GridTables& tables, const GeometricGrid& checker, std::uint64_t seed) {
  SCOPED_TRACE("FindFacePlace on fuzzed values");
  const FuzzedLookups lookups = ComputeCountFuzzedFaceLookups(tables, checker, seed);
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
    accepted_cell_count += CheckFuzzedCellLookups(tables, cells, seed);
    accepted_face_count += CheckFuzzedFaceLookups(tables, checker, seed + 1000);
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

PlacesWithoutAFacePiece ComputeCountPlacesWithoutAFacePiece(const GridTables& tables) {
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
std::size_t CheckPlacesWithoutAFacePiece(const GridTables& tables) {
  SCOPED_TRACE("ToFacePieceId of a place without a face piece");
  const PlacesWithoutAFacePiece counts = ComputeCountPlacesWithoutAFacePiece(tables);
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
    CheckBlocksFound(tables);
    CheckCellRoundTrips(tables, cells);
    CheckNoIdsWithoutACell(tables, cells);
    CheckFacePieceRoundTrips(tables, checker);
    plane_four_below_a_face_piece_count += CheckPlacesWithoutAFacePiece(tables);
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
std::size_t
ComputeCountWrongFacePieces(const GridTables& tables, const GeometricGrid& checker, const IsRight& is_right) {
  std::size_t wrong_count = 0;
  for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
    const FacePlaceIndex face = FindFacePlace(tables, face_piece.id);
    wrong_count += face.value != kNoIndex && is_right(face, face_piece) ? 0 : 1;
  }
  return wrong_count;
}

// ComputeFacePosition on one grid: the centre of every face piece is on the plane of its axis, in the middle of the
// side of its smaller cell.
void CheckFacePositions(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("ComputeFacePosition");
  const std::size_t wrong_count = ComputeCountWrongFacePieces(
      tables,
      checker,
      [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ComputeIsSameVec3(ComputeFacePosition(tables, face), ToCheckerFacePosition(face_piece));
      }
  );
  EXPECT_EQ(wrong_count, 0U);
}

// ComputeFaceArea on one grid: the checker's area of every face piece.
void CheckFaceAreas(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("ComputeFaceArea");
  const std::size_t wrong_count = ComputeCountWrongFacePieces(
      tables,
      checker,
      [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ComputeFaceArea(tables, face) == face_piece.area;
      }
  );
  EXPECT_EQ(wrong_count, 0U);
}

// The checker's face pieces whose cell on one side is one size level bigger than the face piece.
std::size_t ComputeCountBiggerCellsBeside(const GeometricGrid& checker, bool (*is_bigger)(const GeometricFacePiece&)) {
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
std::size_t CheckCellLows(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("FindCellLow");
  const std::size_t wrong_count = ComputeCountWrongFacePieces(
      tables,
      checker,
      [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ToCellIdOrZero(tables, FindCellLow(tables, face)).value
            == ToCheckerCellOrZero(face_piece.cell_low).value;
      }
  );
  EXPECT_EQ(wrong_count, 0U);
  return ComputeCountBiggerCellsBeside(checker, ComputeIsLowCellBigger);
}

// FindCellHigh on one grid: the checker's cell_high of every face piece. Gives the count of bigger high cells.
std::size_t CheckCellHighs(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("FindCellHigh");
  const std::size_t wrong_count = ComputeCountWrongFacePieces(
      tables,
      checker,
      [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ToCellIdOrZero(tables, FindCellHigh(tables, face)).value
            == ToCheckerCellOrZero(face_piece.cell_high).value;
      }
  );
  EXPECT_EQ(wrong_count, 0U);
  return ComputeCountBiggerCellsBeside(checker, ComputeIsHighCellBigger);
}

// The checker's distance between the centres of a face piece's cells: half the width of each cell it has.
double ToCheckerCenterDistance(const GeometricFacePiece& face_piece) {
  const double low_half_width = face_piece.cell_low.has_value() ? ToUnitFrameCellWidth(*face_piece.cell_low) / 2 : 0;
  const double high_half_width = face_piece.cell_high.has_value() ? ToUnitFrameCellWidth(*face_piece.cell_high) / 2 : 0;
  return low_half_width + high_half_width;
}

// ComputeCellCenterDistance on one grid: half the width of each cell beside every face piece, along its axis; half a
// width at the grid's edge.
void CheckCenterDistances(const GridTables& tables, const GeometricGrid& checker) {
  SCOPED_TRACE("ComputeCellCenterDistance");
  const std::size_t wrong_count = ComputeCountWrongFacePieces(
      tables,
      checker,
      [&tables](FacePlaceIndex face, const GeometricFacePiece& face_piece) {
        return ComputeCellCenterDistance(tables, face) == ToCheckerCenterDistance(face_piece);
      }
  );
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
std::size_t ComputeCountWrongCellCenters(const GridTables& tables, const std::vector<CellId>& cells) {
  std::size_t wrong_count = 0;
  for (const CellId cell : cells) {
    const CellPlaceIndex place = FindCellPlace(tables, cell);
    const bool is_found = place.value != kNoIndex;
    wrong_count += is_found && ComputeIsSameVec3(ComputeCellCenter(tables, place), ToBoxCenter(cell)) ? 0 : 1;
  }
  return wrong_count;
}

// ComputeCellCenter on one grid: the centre of every active cell is the middle of its unit box.
void CheckCellCenters(const GridTables& tables, const std::vector<CellId>& cells) {
  SCOPED_TRACE("ComputeCellCenter");
  EXPECT_EQ(ComputeCountWrongCellCenters(tables, cells), 0U);
}

// Every reader of positions, areas, distances and the cells beside a face piece matches the geometric checker over the
// 200 random grids, exactly: one build of the grids for every reader.
TEST(GridTables, RandomGrids_EveryReaderMatchesGeometricChecker) {
  std::size_t bigger_low_count = 0;
  std::size_t bigger_high_count = 0;
  ApplyToRandomGrids([&](const GridTables& tables, const GeometricGrid& checker, const std::vector<CellId>& cells) {
    CheckCellCenters(tables, cells);
    CheckFacePositions(tables, checker);
    CheckFaceAreas(tables, checker);
    bigger_low_count += CheckCellLows(tables, checker);
    bigger_high_count += CheckCellHighs(tables, checker);
    CheckCenterDistances(tables, checker);
  });
  // A bigger cell on each side occurs, so the way through the bigger block was compared.
  EXPECT_GT(bigger_low_count, 0U);
  EXPECT_GT(bigger_high_count, 0U);
}

}  // namespace
}  // namespace fluidity::grid
