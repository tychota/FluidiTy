#include "fluidity/grid/grid_tables.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// A frame with its origin at 0 and biggest cells 1 m wide.
constexpr GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

// A frame whose width is no power of two and whose origin is not 0. A face search counts in quarter widths from the
// origin, so it gives the same face pieces in this frame as in the unit frame.
constexpr GridFrame ToOffsetFrame(int size_level_count) {
  return {
      .origin = {.x = -1.3, .y = 2.9, .z = 0.1},
      .biggest_cell_width = 0.7,
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

GridTables ToTablesOf(const std::expected<TestGrid, GridFailure>& test_grid) {
  return test_grid->grid.tables(RunOn::kCpu);
}

// Appends the 8 small cells of `big_cell`, in cell eighth order.
void AppendSmallCells(CellId big_cell, std::vector<CellId>& cells) {
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
}

// The listed places of a result's array, entries 0..count - 1, as plain numbers.
template <typename PlaceArray>
std::vector<std::uint32_t> ToListedPlaces(const PlaceArray& places, int count) {
  std::vector<std::uint32_t> listed_places;
  listed_places.reserve(static_cast<std::size_t>(count));
  for (int entry = 0; entry < count; ++entry) {
    listed_places.push_back(places[entry].value);
  }
  return listed_places;
}

// Says whether every entry from `count` on is kNoIndex: the bytes past the list are fixed.
template <typename PlaceArray>
bool ComputeHasNoIndexPastCount(const PlaceArray& places, int count) {
  for (auto entry = static_cast<std::size_t>(count); entry < std::size(places); ++entry) {
    if (places[entry].value != kNoIndex) {
      return false;
    }
  }
  return true;
}

bool ComputeIsStrictlyAscending(const std::vector<std::uint32_t>& places) {
  return std::ranges::adjacent_find(places, std::ranges::greater_equal{}) == places.end();
}

// The face pieces FindNextFace gives from `search` until kNoIndex, in order. After more calls than the face places of
// 270 blocks the walk stops and ends with kNoIndex, so a walk that never ends fails a comparison and never hangs.
std::vector<std::uint32_t> FindGivenFaces(const GridTables& tables, FaceSearch search) {
  std::vector<std::uint32_t> given_faces;
  constexpr int kMaxCalls = (kMaxFaceSearchBlocks * kFacePlacesPerBlock) + 1;
  for (int call = 0; call < kMaxCalls; ++call) {
    const FacePlaceIndex next_face = FindNextFace(tables, &search);
    if (next_face.value == kNoIndex) {
      return given_faces;
    }
    given_faces.push_back(next_face.value);
  }
  given_faces.push_back(kNoIndex);
  return given_faces;
}

// The face pieces a face search from `face` walks, in the order FindNextFace gives them (FindGivenFaces).
std::vector<std::uint32_t>
FindWalkedFaces(const GridTables& tables, FacePlaceIndex face, double radius_in_cell_widths) {
  return FindGivenFaces(tables, BuildFaceSearch(tables, face, radius_in_cell_widths));
}

// The checker's face pieces within the radius of `face` whose size level is within one of its own, as places of the
// tables in ascending order. The filter is made here, in the test: FacesWithin itself takes every size level.
std::vector<std::uint32_t> FindCheckerFacesWithin(
    const GridTables& tables,
    const GeometricGrid& checker,
    const GeometricFacePiece& face_piece,
    double radius_in_cell_widths
) {
  std::vector<std::uint32_t> face_places;
  for (const GeometricFacePiece& within : checker.FacesWithin(face_piece.id, radius_in_cell_widths)) {
    const int size_level_difference = static_cast<int>(within.size_level) - static_cast<int>(face_piece.size_level);
    if (size_level_difference >= -1 && size_level_difference <= 1) {
      face_places.push_back(FindFacePlace(tables, within.id).value);
    }
  }
  std::ranges::sort(face_places);
  return face_places;
}

// The cells near a face piece by its definition, from the checker alone: its cell_low and cell_high and the checker's
// touching cells of each, as places of the tables in ascending order.
std::vector<std::uint32_t>
FindCheckerCellsNearFace(const GridTables& tables, const GeometricGrid& checker, const GeometricFacePiece& face_piece) {
  std::set<std::uint32_t> places;
  for (const std::optional<CellId>& cell : {face_piece.cell_low, face_piece.cell_high}) {
    if (!cell.has_value()) {
      continue;
    }
    places.insert(FindCellPlace(tables, *cell).value);
    for (const CellId touching_cell : checker.TouchingCells(*cell)) {
      places.insert(FindCellPlace(tables, touching_cell).value);
    }
  }
  return {places.begin(), places.end()};
}

// The touching cells of each active cell of one grid by FindTouchingCells, which its own tests compare with the
// checker, found once per cell: entry p holds the ascending touching places of cell place p, with p itself first.
class TouchingCellsByPlace {
 public:
  explicit TouchingCellsByPlace(const GridTables& tables)
      : tables_(tables), cells_by_place_(std::size_t{tables.block_count} * kCellPlacesPerBlock) {
  }

  // The cell and its touching cells, found on the first call for that cell.
  const std::vector<std::uint32_t>& FindCellAndTouchingCells(CellPlaceIndex cell) {
    std::vector<std::uint32_t>& places = cells_by_place_.at(cell.value);
    if (places.empty()) {
      const TouchingCells touching = FindTouchingCells(tables_, cell);
      places = ToListedPlaces(touching.cells, touching.count);
      places.insert(std::ranges::upper_bound(places, cell.value), cell.value);
    }
    return places;
  }

 private:
  GridTables tables_;
  std::vector<std::vector<std::uint32_t>> cells_by_place_;
};

// The cells near a face piece by its definition: the two cells from the checker and the touching cells of each from
// FindTouchingCells, as places in ascending order.
std::vector<std::uint32_t> FindCellsNearFaceByTouchingCells(
    const GridTables& tables,
    TouchingCellsByPlace& touching_cells,
    const GeometricFacePiece& face_piece
) {
  std::vector<std::uint32_t> places;
  for (const std::optional<CellId>& cell : {face_piece.cell_low, face_piece.cell_high}) {
    if (!cell.has_value()) {
      continue;
    }
    const std::vector<std::uint32_t>& cell_places =
        touching_cells.FindCellAndTouchingCells(FindCellPlace(tables, *cell));
    std::vector<std::uint32_t> merged_places;
    std::ranges::set_union(places, cell_places, std::back_inserter(merged_places));
    places = std::move(merged_places);
  }
  return places;
}

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

// A hand-made grid for the readers: its name, its frame and its active cells.
struct NamedGrid {
  const char* name;
  GridFrame frame;
  std::vector<CellId> cells;
};

// The size-level-0 cell {x, y, z} with each of the 26 size-level-0 cells around it split, 2 size levels.
std::vector<CellId> BuildCellWithEveryNeighbourSplit(std::int32_t x, std::int32_t y, std::int32_t z) {
  std::vector<CellId> cells = {ToCell(0, x, y, z)};
  for (std::int32_t offset_z = -1; offset_z <= 1; ++offset_z) {
    for (std::int32_t offset_y = -1; offset_y <= 1; ++offset_y) {
      for (std::int32_t offset_x = -1; offset_x <= 1; ++offset_x) {
        if (offset_x != 0 || offset_y != 0 || offset_z != 0) {
          AppendSmallCells(ToCell(0, x + offset_x, y + offset_y, z + offset_z), cells);
        }
      }
    }
  }
  return cells;
}

// Size level 0 {3,0,0} split into the size-level-1 cells of block (1,0,0), beside the size-level-0 cells {4..6,0,0}
// of block (1,0,0), 2 size levels. The face pieces between {7,y,z} of size level 1 and {4,0,0} are kept at plane 4 of
// the size-level-1 block (1,0,0), as no size-level-1 block lies above it. The size-level-0 face piece between {4,0,0}
// and {5,0,0}, 1 m = 2 widths away, lies in the size-level-0 block (1,0,0), which is the bigger block of no
// size-level-1 block of the grid: only the position of the bigger block leads to it.
std::vector<CellId> BuildBiggerBlockBehindAbsentNeighbour() {
  std::vector<CellId> cells = {ToCell(0, 4, 0, 0), ToCell(0, 5, 0, 0), ToCell(0, 6, 0, 0)};
  AppendSmallCells(ToCell(0, 3, 0, 0), cells);
  return cells;
}

// Three size levels: size level 0 {0,0,0}; the size-level-1 cells of {1,0,0} with x = 2 active, and those with x = 3
// split into size-level-2 cells, which touch no size-level-0 cell. Face pieces of size level 2 lie within 1.5 widths of
// the big cell's face pieces and are left out.
std::vector<CellId> BuildThreeSizeCells() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    const CellId small_cell = ComputeSmallCell(ToCell(0, 1, 0, 0), cell_eighth);
    if (ToCellCoord(small_cell).x == 2) {
      cells.push_back(small_cell);
    } else {
      AppendSmallCells(small_cell, cells);
    }
  }
  return cells;
}

// The size-level-0 cells of the box low..high on each axis.
//
// clang-tidy: the two ends of one range; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<CellId> BuildBoxOfCells(std::int32_t low, std::int32_t high) {
  std::vector<CellId> cells;
  for (std::int32_t z = low; z <= high; ++z) {
    for (std::int32_t y = low; y <= high; ++y) {
      for (std::int32_t x = low; x <= high; ++x) {
        cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  return cells;
}

// The size-level-0 cells of the box -2..2 on each axis: face pieces at block borders in every direction, and at the
// grid's edge.
std::vector<CellId> BuildUniformBox() {
  return BuildBoxOfCells(-2, 2);
}

// Two size-level-0 cells of one size beside each other, {0,0,0} and {1,0,0}, with all 34 size-level-0 places around
// the pair split, 2 size levels: the largest count of cells near a face known, 82, at the X face piece between them.
std::vector<CellId> BuildSameSizePairWithEveryNeighbourSplit() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0), ToCell(0, 1, 0, 0)};
  for (std::int32_t z = -1; z <= 1; ++z) {
    for (std::int32_t y = -1; y <= 1; ++y) {
      for (std::int32_t x = -1; x <= 2; ++x) {
        const bool is_pair_cell = y == 0 && z == 0 && (x == 0 || x == 1);
        if (!is_pair_cell) {
          AppendSmallCells(ToCell(0, x, y, z), cells);
        }
      }
    }
  }
  return cells;
}

// Grids the random grids never give: face pieces at block borders, between two sizes, at the grid's edge, at negative
// coordinates and at both coordinate limits, a bigger block behind an absent neighbour, three size levels, and a frame
// with a width that is no power of two and an origin. 2^17 - 1 is the largest coordinate a size-level-0 cell may have
// and still be split.
std::vector<NamedGrid> BuildHandGrids() {
  constexpr auto kLargestSplitCoord = static_cast<std::int32_t>((1U << 17U) - 1U);
  std::vector<NamedGrid> grids;
  grids.push_back({.name = "two-size fixture", .frame = ToUnitFrame(2), .cells = BuildTwoSizeFixtureCells()});
  std::vector<CellId> negative_two_size_cells = {ToCell(0, -1, -1, -1)};
  AppendSmallCells(ToCell(0, 0, -1, -1), negative_two_size_cells);
  grids.push_back({.name = "two-size fixture at -1", .frame = ToUnitFrame(2), .cells = negative_two_size_cells});
  grids.push_back({
      .name = "every neighbour split",
      .frame = ToUnitFrame(2),
      .cells = BuildCellWithEveryNeighbourSplit(0, 0, 0),
  });
  grids.push_back({
      .name = "same-size pair with every neighbour split",
      .frame = ToUnitFrame(2),
      .cells = BuildSameSizePairWithEveryNeighbourSplit(),
  });
  grids.push_back({.name = "uniform box across blocks", .frame = ToUnitFrame(1), .cells = BuildUniformBox()});
  grids.push_back({
      .name = "two cells at the high limit",
      .frame = ToUnitFrame(1),
      .cells = {ToCell(0, kMaxCoord - 1, kMaxCoord, kMaxCoord), ToCell(0, kMaxCoord, kMaxCoord, kMaxCoord)},
  });
  grids.push_back({
      .name = "two cells at the low limit",
      .frame = ToUnitFrame(1),
      .cells = {ToCell(0, -kMaxCoord, -kMaxCoord, -kMaxCoord), ToCell(0, -kMaxCoord + 1, -kMaxCoord, -kMaxCoord)},
  });
  std::vector<CellId> high_split_cells = {ToCell(0, kLargestSplitCoord - 1, 0, 0)};
  AppendSmallCells(ToCell(0, kLargestSplitCoord, 0, 0), high_split_cells);
  grids.push_back({.name = "small cells at the high split limit", .frame = ToUnitFrame(2), .cells = high_split_cells});
  std::vector<CellId> low_split_cells = {ToCell(0, -kLargestSplitCoord + 1, 0, 0)};
  AppendSmallCells(ToCell(0, -kLargestSplitCoord, 0, 0), low_split_cells);
  grids.push_back({.name = "small cells at the low split limit", .frame = ToUnitFrame(2), .cells = low_split_cells});
  grids.push_back({
      .name = "bigger block behind an absent neighbour",
      .frame = ToUnitFrame(2),
      .cells = BuildBiggerBlockBehindAbsentNeighbour(),
  });
  grids.push_back({.name = "three sizes", .frame = ToUnitFrame(3), .cells = BuildThreeSizeCells()});
  grids.push_back({.name = "three sizes, offset frame", .frame = ToOffsetFrame(3), .cells = BuildThreeSizeCells()});
  return grids;
}

// FindCellsNearFace at every face piece of the hand-made grids is its definition from the checker alone: cell_low,
// cell_high and the checker's touching cells of each, in ascending index order.
TEST(FindCellsNearFace, HandGrids_MatchCheckerCells) {
  for (const NamedGrid& named_grid : BuildHandGrids()) {
    SCOPED_TRACE(named_grid.name);
    const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(named_grid.frame, named_grid.cells));
    ASSERT_TRUE(test_grid.has_value());
    const GridTables tables = ToTablesOf(test_grid);
    const GeometricGrid checker(named_grid.frame, named_grid.cells);
    std::size_t mismatch_count = 0;
    for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
      const CellsNearFace cells_near_face = FindCellsNearFace(tables, FindFacePlace(tables, face_piece.id));
      const bool is_match = ToListedPlaces(cells_near_face.cells, cells_near_face.count)
          == FindCheckerCellsNearFace(tables, checker, face_piece);
      mismatch_count += is_match ? 0 : 1;
    }
    EXPECT_EQ(mismatch_count, 0U);
  }
}

// Computes the largest count FindCellsNearFace gives over every face piece of the tables.
int ComputeLargestCellsNearFace(const GridTables& tables) {
  int largest_count = 0;
  const std::uint32_t face_place_count = tables.block_count * static_cast<std::uint32_t>(kFacePlacesPerBlock);
  for (std::uint32_t face_place = 0; face_place < face_place_count; ++face_place) {
    if (tables.face_piece_kinds[face_place] != FacePieceKind::kNotAPiece) {
      largest_count = std::max(largest_count, FindCellsNearFace(tables, FacePlaceIndex{face_place}).count);
    }
  }
  return largest_count;
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

// A grid piece made from a whole grid's tables by keeping some blocks: the kept blocks in their order, with new
// indexes, and every link to a dropped block "none", as a grid piece's tables have them.
struct GridPieceTables {
  GridTables tables;
  std::vector<CellBlockId> block_ids;
  std::vector<std::uint32_t> same_size_level_blocks;
  std::vector<std::uint32_t> bigger_block;
  std::vector<std::uint32_t> smaller_blocks;
  std::vector<CellState> cell_states;
  std::vector<FacePieceKind> face_piece_kinds;
  // By block of the whole grid: its index in the grid piece, or kNoIndex when dropped.
  std::vector<std::uint32_t> grid_piece_blocks;
};

// Copies the kept blocks of `whole` into `grid_piece`, by is_kept per block, and points its GridTables at them.
void CopyGridPiece(const GridTables& whole, const std::vector<bool>& is_kept, GridPieceTables& grid_piece) {
  grid_piece.grid_piece_blocks.assign(whole.block_count, kNoIndex);
  std::uint32_t kept_count = 0;
  for (std::uint32_t block = 0; block < whole.block_count; ++block) {
    if (is_kept.at(block)) {
      grid_piece.grid_piece_blocks.at(block) = kept_count;
      ++kept_count;
    }
  }
  const auto to_piece_block = [&grid_piece](std::uint32_t whole_block) {
    return whole_block == kNoIndex ? kNoIndex : grid_piece.grid_piece_blocks.at(whole_block);
  };
  for (std::uint32_t block = 0; block < whole.block_count; ++block) {
    if (!is_kept.at(block)) {
      continue;
    }
    const std::size_t first = block;
    grid_piece.block_ids.push_back(whole.block_ids[first]);
    for (std::size_t entry = 0; entry < 27; ++entry) {
      grid_piece.same_size_level_blocks.push_back(to_piece_block(whole.same_size_level_blocks[(27 * first) + entry]));
    }
    grid_piece.bigger_block.push_back(to_piece_block(whole.bigger_block[first]));
    for (std::size_t entry = 0; entry < 8; ++entry) {
      grid_piece.smaller_blocks.push_back(to_piece_block(whole.smaller_blocks[(8 * first) + entry]));
    }
    const std::size_t first_cell = first * kCellPlacesPerBlock;
    grid_piece.cell_states.insert(
        grid_piece.cell_states.end(),
        whole.cell_states + first_cell,
        whole.cell_states + first_cell + kCellPlacesPerBlock
    );
    const std::size_t first_face = first * kFacePlacesPerBlock;
    grid_piece.face_piece_kinds.insert(
        grid_piece.face_piece_kinds.end(),
        whole.face_piece_kinds + first_face,
        whole.face_piece_kinds + first_face + kFacePlacesPerBlock
    );
  }
  grid_piece.tables = whole;
  grid_piece.tables.block_count = kept_count;
  grid_piece.tables.block_ids = grid_piece.block_ids.data();
  grid_piece.tables.same_size_level_blocks = grid_piece.same_size_level_blocks.data();
  grid_piece.tables.bigger_block = grid_piece.bigger_block.data();
  grid_piece.tables.smaller_blocks = grid_piece.smaller_blocks.data();
  grid_piece.tables.cell_states = grid_piece.cell_states.data();
  grid_piece.tables.face_piece_kinds = grid_piece.face_piece_kinds.data();
}

// Converts a face place of the whole grid to its place in the grid piece; kNoIndex when its block was dropped.
std::uint32_t ToGridPieceFace(const GridPieceTables& grid_piece, std::uint32_t whole_face) {
  const std::uint32_t piece_block = grid_piece.grid_piece_blocks.at(whole_face / kFacePlacesPerBlock);
  if (piece_block == kNoIndex) {
    return kNoIndex;
  }
  return (piece_block * kFacePlacesPerBlock) + (whole_face % kFacePlacesPerBlock);
}

// Converts the whole grid's walk to the piece: the face pieces of the kept blocks, at their places in the piece, in
// the same ascending order, as the grid piece's search must give them.
std::vector<std::uint32_t>
ToGridPieceFaces(const GridPieceTables& grid_piece, const std::vector<std::uint32_t>& whole_faces) {
  std::vector<std::uint32_t> grid_piece_faces;
  for (const std::uint32_t whole_face : whole_faces) {
    const std::uint32_t grid_piece_face = ToGridPieceFace(grid_piece, whole_face);
    if (grid_piece_face != kNoIndex) {
      grid_piece_faces.push_back(grid_piece_face);
    }
  }
  return grid_piece_faces;
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

// The number of small random grids the bound test builds, and their shape: 3 size levels in a box of 2 x 2 x 2
// size-level-0 cells, at most 512 cells each. 10,000 such grids took 35 s in the debug preset (18 s to build them,
// 17 s for FindCellsNearFace at their face pieces), so the test builds 1,000.
constexpr std::uint64_t kSmallRandomGridCount = 1'000;
constexpr int kSmallRandomGridSizeLevelCount = 3;
constexpr int kSmallRandomGridBiggestCellsPerAxis = 2;

// FindCellsNearFace at every face piece of 1,000 small random grids stays within kMaxCellsNearFace. The largest count
// found is recorded as the property largest_cells_near_face.
TEST(FindCellsNearFace, SmallRandomGrids_NeverExceedTheBound) {
  const GridFrame frame = ToUnitFrame(kSmallRandomGridSizeLevelCount);
  // One set of buffers, given back after each grid: a small grid has at most 1 + 1 + 8 blocks.
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = 16});
  ASSERT_TRUE(owner.has_value());
  GridBuffers buffers = TakeBuffers(std::move(owner->grid));
  int largest_count = 0;
  for (std::uint64_t seed = 1; seed <= kSmallRandomGridCount; ++seed) {
    const std::vector<CellId> cells =
        BuildRandomGridCells(seed, kSmallRandomGridSizeLevelCount, kSmallRandomGridBiggestCellsPerAxis);
    std::expected<StaggeredGrid, GridFailure> built = BuildGrid(ToInputs(frame, cells), buffers);
    ASSERT_TRUE(built.has_value()) << "seed " << seed;
    largest_count = std::max(largest_count, ComputeLargestCellsNearFace(built->tables(RunOn::kCpu)));
    buffers = TakeBuffers(std::move(*built));
  }
  RecordProperty("largest_cells_near_face", largest_count);
  EXPECT_LE(largest_count, kMaxCellsNearFace);
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

// The face search at every face piece of the hand-made grids, at radii 1.5, 2.5 and 4, gives the checker's face pieces
// of size levels within one, in ascending index order.
TEST(FindNextFace, HandGrids_MatchFilteredChecker) {
  for (const NamedGrid& named_grid : BuildHandGrids()) {
    SCOPED_TRACE(named_grid.name);
    const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(named_grid.frame, named_grid.cells));
    ASSERT_TRUE(test_grid.has_value());
    const GridTables tables = ToTablesOf(test_grid);
    const GeometricGrid checker(named_grid.frame, named_grid.cells);
    for (const double radius : {1.5, 2.5, 4.0}) {
      SCOPED_TRACE(testing::Message() << "radius " << radius);
      std::size_t mismatch_count = 0;
      for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
        const FacePlaceIndex face = FindFacePlace(tables, face_piece.id);
        const bool is_match =
            FindWalkedFaces(tables, face, radius) == FindCheckerFacesWithin(tables, checker, face_piece, radius);
        mismatch_count += is_match ? 0 : 1;
      }
      EXPECT_EQ(mismatch_count, 0U);
    }
  }
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

// The size-level-0 cells of the box 0..7 on each axis, those of x 4..7, y 0..3, z 0..3 split, 2 size levels: 8
// size-level-0 blocks, of which (1,0,0) holds only split places, over the size-level-1 blocks (2..3, 0..1, 0..1).
std::vector<CellId> BuildBoxWithOneSplitBlock() {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z <= 7; ++z) {
    for (std::int32_t y = 0; y <= 7; ++y) {
      for (std::int32_t x = 0; x <= 7; ++x) {
        const bool is_in_split_block = x >= 4 && y <= 3 && z <= 3;
        if (is_in_split_block) {
          AppendSmallCells(ToCell(0, x, y, z), cells);
        } else {
          cells.push_back(ToCell(0, x, y, z));
        }
      }
    }
  }
  return cells;
}

// How the walks from the face pieces of one block compare between a grid piece and its whole grid.
struct GridPieceWalkComparison {
  std::size_t mismatch_count = 0;  // Walks not the whole grid's walk restricted to the kept blocks.
  std::size_t faces_under_dropped_block = 0;  // Walked face pieces in the smaller blocks of the dropped block.
};

// Compares, from every face piece of `start_block`, at radii 1.5 and 4, the grid piece's walk with the whole grid's
// walk restricted to the kept blocks, and counts the walked face pieces under `dropped_block`.
GridPieceWalkComparison ComputeGridPieceWalkComparison(
    const GridTables& whole,
    const GridPieceTables& grid_piece,
    CellBlockIndex start_block,
    CellBlockIndex dropped_block
) {
  GridPieceWalkComparison comparison;
  const std::uint32_t first_face = start_block.value * static_cast<std::uint32_t>(kFacePlacesPerBlock);
  for (std::uint32_t whole_face = first_face; whole_face < first_face + kFacePlacesPerBlock; ++whole_face) {
    const std::uint32_t grid_piece_face = ToGridPieceFace(grid_piece, whole_face);
    if (grid_piece_face == kNoIndex || whole.face_piece_kinds[whole_face] == FacePieceKind::kNotAPiece) {
      continue;
    }
    for (const double radius : {1.5, 4.0}) {
      const std::vector<std::uint32_t> grid_piece_faces =
          FindWalkedFaces(grid_piece.tables, FacePlaceIndex{grid_piece_face}, radius);
      const std::vector<std::uint32_t> whole_faces = FindWalkedFaces(whole, FacePlaceIndex{whole_face}, radius);
      comparison.mismatch_count += grid_piece_faces == ToGridPieceFaces(grid_piece, whole_faces) ? 0 : 1;
      const auto is_under_dropped_block = [&whole, dropped_block](std::uint32_t face) {
        return whole.bigger_block[face / kFacePlacesPerBlock] == dropped_block.value;
      };
      comparison.faces_under_dropped_block +=
          static_cast<std::size_t>(std::ranges::count_if(whole_faces, is_under_dropped_block));
    }
  }
  return comparison;
}

// A grid piece of BuildBoxWithOneSplitBlock without the size-level-0 block (1,0,0): one dropped neighbour, on +X of
// block (0,0,0), among linked ones (on +Y, +Z and the others), whose smaller blocks the grid piece holds. From every
// face piece of block (0,0,0), at radii 1.5 and 4, the search gives the whole grid's walk restricted to the kept
// blocks: the size-level-1 face pieces under the dropped position are found by their ids, at that position only.
TEST(FindNextFace, GridPieceWithOneNeighbourDropped_GivesTheKeptFaces) {
  const std::vector<CellId> cells = BuildBoxWithOneSplitBlock();
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(2), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables whole = ToTablesOf(test_grid);
  const CellBlockIndex dropped_block = FindCellBlock(whole, ToCellBlockId(ToCell(0, 4, 0, 0)));
  const CellBlockIndex start_block = FindCellBlock(whole, ToCellBlockId(ToCell(0, 0, 0, 0)));
  ASSERT_NE(dropped_block.value, kNoIndex);
  ASSERT_NE(start_block.value, kNoIndex);
  std::vector<bool> is_kept(whole.block_count, true);
  is_kept.at(dropped_block.value) = false;
  GridPieceTables grid_piece;
  CopyGridPiece(whole, is_kept, grid_piece);
  const GridPieceWalkComparison comparison =
      ComputeGridPieceWalkComparison(whole, grid_piece, start_block, dropped_block);
  EXPECT_EQ(comparison.mismatch_count, 0U);
  // Face pieces of the smaller blocks under the dropped position were among those found.
  EXPECT_GT(comparison.faces_under_dropped_block, 0U);
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

// What one grid's face pieces got wrong, property by property: a count of face pieces or of walks each.
struct FaceReaderMistakes {
  std::size_t cells_near_face_not_the_definition = 0;  // Every face piece, touching cells from FindTouchingCells.
  std::size_t cells_near_face_not_the_checkers = 0;  // The chosen face pieces, touching cells from the checker.
  std::size_t cells_near_face_out_of_order = 0;
  std::size_t cells_near_face_entries_past_count_not_no_index = 0;
  std::size_t walks_not_the_checkers = 0;  // The chosen face pieces at each radius, against the filtered checker.
  std::size_t walks_out_of_order = 0;
  std::size_t search_blocks_out_of_order = 0;  // Not strictly ascending, or not kNoIndex from block_count on.
  // In the grid piece, not the whole grid's walk in the kept blocks.
  std::size_t grid_piece_walks_not_the_whole_grids = 0;
  std::size_t grid_piece_cells_near_face_past_the_bound = 0;  // In the grid piece, more than kMaxCellsNearFace.
};

// What the random grids met, to show the tests reached the cases they are about.
struct FaceReaderCases {
  int largest_cells_near_face = 0;
  // Walked face pieces of size level l - 1 in a block that is the bigger block of no same-size-level neighbour that
  // the start's block has: only the positions of the bigger block lead to them.
  std::size_t faces_behind_absent_neighbours = 0;
  std::size_t faces_two_size_levels_away = 0;  // Checker face pieces within the radius that the filter left out.
  std::size_t grid_piece_walks = 0;  // Walks compared in the grid pieces.
};

// What one random grid gave: its mistakes and the cases it met.
struct FaceReaderCheck {
  FaceReaderMistakes mistakes;
  FaceReaderCases cases_met;
};

// The share of blocks a random grid piece keeps, in tenths, by the grid's seed modulo 3: 3, 5 or 8.
constexpr std::array<std::uint64_t, 3> kKeptTenths = {3, 5, 8};

// Copies a random grid piece of `tables`: each block kept with the share kKeptTenths of `grid_piece_seed`, drawn from
// one std::mt19937_64 of `grid_piece_seed`, and the blocks of the face places `starts` always kept.
void CopyRandomGridPiece(
    const GridTables& tables,
    const std::vector<FacePlaceIndex>& starts,
    std::uint64_t grid_piece_seed,
    GridPieceTables& grid_piece
) {
  std::mt19937_64 random_numbers(grid_piece_seed);
  const std::uint64_t kept_tenths = kKeptTenths.at(grid_piece_seed % kKeptTenths.size());
  std::vector<bool> is_kept(tables.block_count);
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    is_kept.at(block) = ToUniformIndex(random_numbers(), 10) < kept_tenths;
  }
  for (const FacePlaceIndex start : starts) {
    is_kept.at(ToCellBlockIndex(start).value) = true;
  }
  CopyGridPiece(tables, is_kept, grid_piece);
}

// The chosen face pieces of one grid, as indexes into the checker's FacePieces(): the first, the last and three
// between; per axis the first between two sizes, the first at the grid's edge, and the first kept at plane 4 beside a
// bigger cell_high.
std::vector<std::size_t> FindChosenFacePieces(const GridTables& tables, const GeometricGrid& checker) {
  const std::vector<GeometricFacePiece>& face_pieces = checker.FacePieces();
  const std::size_t piece_count = face_pieces.size();
  std::set<std::size_t> chosen = {0, piece_count / 4, piece_count / 2, (3 * piece_count) / 4, piece_count - 1};
  std::array<bool, 3> has_between_two_sizes{};
  std::array<bool, 3> has_grid_edge{};
  std::array<bool, 3> has_plane_four{};
  for (std::size_t index = 0; index < piece_count; ++index) {
    const GeometricFacePiece& face_piece = face_pieces.at(index);
    const auto axis = static_cast<std::size_t>(face_piece.axis);
    const bool is_grid_edge = !face_piece.cell_low.has_value() || !face_piece.cell_high.has_value();
    if (is_grid_edge && !has_grid_edge.at(axis)) {
      has_grid_edge.at(axis) = true;
      chosen.insert(index);
    }
    if (is_grid_edge) {
      continue;
    }
    const bool is_between_two_sizes = ToSizeLevel(*face_piece.cell_low) != ToSizeLevel(*face_piece.cell_high);
    if (is_between_two_sizes && !has_between_two_sizes.at(axis)) {
      has_between_two_sizes.at(axis) = true;
      chosen.insert(index);
    }
    // The plane along the axis of the face place: x of (z 4 + y) 5 + x, y of (z 5 + y) 4 + x, z of (z 4 + y) 4 + x.
    const std::uint32_t place_on_axis = (FindFacePlace(tables, face_piece.id).value % kFacePlacesPerBlock) % 80;
    const std::array<std::uint32_t, 3> planes = {place_on_axis % 5, (place_on_axis / 4) % 5, place_on_axis / 16};
    const bool is_plane_four_beside_bigger =
        planes.at(axis) == 4 && ToSizeLevel(*face_piece.cell_high) < face_piece.size_level;
    if (is_plane_four_beside_bigger && !has_plane_four.at(axis)) {
      has_plane_four.at(axis) = true;
      chosen.insert(index);
    }
  }
  return {chosen.begin(), chosen.end()};
}

// Counts the walked face pieces of size level l - 1 that lie in a block no same-size-level neighbour of the start's
// block has as its bigger block.
std::size_t ComputeFacesBehindAbsentNeighbours(
    const GridTables& tables,
    FacePlaceIndex start,
    const std::vector<std::uint32_t>& walked_faces
) {
  const std::uint32_t start_block = ToCellBlockIndex(start).value;
  const auto start_size_level = static_cast<int>(ToSizeLevel(tables, CellBlockIndex{start_block}));
  std::set<std::uint32_t> neighbours_bigger_blocks;
  for (std::size_t entry = 0; entry < 27; ++entry) {
    const std::uint32_t neighbour = tables.same_size_level_blocks[(27 * std::size_t{start_block}) + entry];
    if (neighbour != kNoIndex) {
      neighbours_bigger_blocks.insert(tables.bigger_block[neighbour]);
    }
  }
  std::size_t face_count = 0;
  for (const std::uint32_t face : walked_faces) {
    const CellBlockIndex block = ToCellBlockIndex(FacePlaceIndex{face});
    const bool is_bigger = static_cast<int>(ToSizeLevel(tables, block)) == start_size_level - 1;
    face_count += is_bigger && !neighbours_bigger_blocks.contains(block.value) ? 1 : 0;
  }
  return face_count;
}

// Which face pieces of a random grid FindCellsNearFace is checked at against its definition.
enum class FacePiecesChecked : std::uint8_t {
  kChosen,  // Only the chosen face pieces (FindChosenFacePieces).
  kEvery,  // Every face piece as well.
};

// Checks FindCellsNearFace at every face piece of one random grid against its definition.
FaceReaderCheck FindEveryFaceMistakes(const GridTables& tables, const GeometricGrid& checker) {
  FaceReaderMistakes mistakes;
  FaceReaderCases cases_met;
  TouchingCellsByPlace touching_cells(tables);
  for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
    const CellsNearFace cells_near_face = FindCellsNearFace(tables, FindFacePlace(tables, face_piece.id));
    const std::vector<std::uint32_t> listed = ToListedPlaces(cells_near_face.cells, cells_near_face.count);
    mistakes.cells_near_face_not_the_definition +=
        listed == FindCellsNearFaceByTouchingCells(tables, touching_cells, face_piece) ? 0 : 1;
    mistakes.cells_near_face_out_of_order += ComputeIsStrictlyAscending(listed) ? 0 : 1;
    mistakes.cells_near_face_entries_past_count_not_no_index +=
        ComputeHasNoIndexPastCount(cells_near_face.cells, cells_near_face.count) ? 0 : 1;
    cases_met.largest_cells_near_face = std::max(cases_met.largest_cells_near_face, cells_near_face.count);
  }
  return {.mistakes = mistakes, .cases_met = cases_met};
}

// Checks both readers on one random grid: FindCellsNearFace at every face piece when asked, and at the chosen face
// pieces against the checker alone; the face search at the chosen face pieces at radii 1.5, 2.5 and 4; and both
// readers from the chosen face pieces in a random grid piece of `grid_piece_seed`, against the whole grid.
FaceReaderCheck FindFaceReaderMistakes(
    const GridTables& tables,
    const GeometricGrid& checker,
    FacePiecesChecked face_pieces_checked,
    std::uint64_t grid_piece_seed
) {
  FaceReaderMistakes mistakes;
  FaceReaderCases cases_met;
  const std::vector<GeometricFacePiece>& face_pieces = checker.FacePieces();
  if (face_pieces_checked == FacePiecesChecked::kEvery) {
    const FaceReaderCheck every_face_check = FindEveryFaceMistakes(tables, checker);
    mistakes = every_face_check.mistakes;
    cases_met = every_face_check.cases_met;
  }
  std::vector<FacePlaceIndex> starts;
  for (const std::size_t index : FindChosenFacePieces(tables, checker)) {
    starts.push_back(FindFacePlace(tables, face_pieces.at(index).id));
  }
  GridPieceTables grid_piece;
  CopyRandomGridPiece(tables, starts, grid_piece_seed, grid_piece);
  for (const std::size_t index : FindChosenFacePieces(tables, checker)) {
    const GeometricFacePiece& face_piece = face_pieces.at(index);
    const FacePlaceIndex face = FindFacePlace(tables, face_piece.id);
    const FacePlaceIndex grid_piece_face = {.value = ToGridPieceFace(grid_piece, face.value)};
    mistakes.grid_piece_cells_near_face_past_the_bound +=
        FindCellsNearFace(grid_piece.tables, grid_piece_face).count > kMaxCellsNearFace ? 1 : 0;
    const CellsNearFace cells_near_face = FindCellsNearFace(tables, face);
    mistakes.cells_near_face_not_the_checkers += ToListedPlaces(cells_near_face.cells, cells_near_face.count)
            == FindCheckerCellsNearFace(tables, checker, face_piece)
        ? 0
        : 1;
    for (const double radius : {1.5, 2.5, 4.0}) {
      const std::vector<std::uint32_t> walked_faces = FindWalkedFaces(tables, face, radius);
      const std::vector<std::uint32_t> checker_faces = FindCheckerFacesWithin(tables, checker, face_piece, radius);
      mistakes.walks_not_the_checkers += walked_faces == checker_faces ? 0 : 1;
      mistakes.walks_out_of_order += ComputeIsStrictlyAscending(walked_faces) ? 0 : 1;
      const FaceSearch search = BuildFaceSearch(tables, face, radius);
      const std::vector<std::uint32_t> blocks = ToListedPlaces(search.blocks, search.block_count);
      const bool are_blocks_in_order =
          ComputeIsStrictlyAscending(blocks) && ComputeHasNoIndexPastCount(search.blocks, search.block_count);
      mistakes.search_blocks_out_of_order += are_blocks_in_order ? 0 : 1;
      cases_met.faces_behind_absent_neighbours += ComputeFacesBehindAbsentNeighbours(tables, face, walked_faces);
      cases_met.faces_two_size_levels_away += checker.FacesWithin(face_piece.id, radius).size() - checker_faces.size();
      const std::vector<std::uint32_t> grid_piece_faces = FindWalkedFaces(grid_piece.tables, grid_piece_face, radius);
      mistakes.grid_piece_walks_not_the_whole_grids +=
          grid_piece_faces == ToGridPieceFaces(grid_piece, walked_faces) ? 0 : 1;
      ++cases_met.grid_piece_walks;
    }
  }
  return {.mistakes = mistakes, .cases_met = cases_met};
}

// Checks that no face piece or walk of `mistake_count` went wrong, under a trace that names the property.
void CheckNoMistake(const char* property, std::size_t mistake_count) {
  SCOPED_TRACE(property);
  EXPECT_EQ(mistake_count, 0U);
}

// The random grids, spread by size, at whose every face piece FindCellsNearFace is checked. All 200 took 13 s more in
// the debug preset; the chosen face pieces of every grid are checked as well.
constexpr std::size_t kEveryFacePieceGridCount = 25;

// Both readers on the 200 random grids. FindCellsNearFace at every face piece of 25 of them, spread by size, is
// cell_low, cell_high and their touching cells, ascending, kNoIndex past count; at the chosen face pieces of all 200 it
// is the checker's own. The face search from the chosen face pieces at radii 1.5, 2.5 and 4 gives the checker's face
// pieces of size levels within one, ascending, from blocks listed in ascending order. In a random grid piece of each
// grid, keeping 3, 5 or 8 tenths of its blocks and the chosen face pieces' blocks, the search from each chosen face
// piece gives the whole grid's walk restricted to the kept blocks, and FindCellsNearFace stays within its bound. One
// TEST checks all of these, because each TEST runs in a process of its own and would build the 200 grids again.
TEST(GridTables, RandomGrids_FaceReadersMatchGeometricChecker) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_EQ(built.grids.size(), kRandomGridCount);
  FaceReaderCases cases_met;
  const std::vector<std::size_t> every_piece_grids = FindRandomGridsSpreadBySize(kEveryFacePieceGridCount);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid);
    const FacePiecesChecked face_pieces_checked =
        std::ranges::contains(every_piece_grids, grid) ? FacePiecesChecked::kEvery : FacePiecesChecked::kChosen;
    const FaceReaderCheck check = FindFaceReaderMistakes(
        built.grids.at(grid).grid.tables(RunOn::kCpu),
        built.checkers.at(grid),
        face_pieces_checked,
        grid + 7000
    );
    const FaceReaderMistakes& mistakes = check.mistakes;
    cases_met.largest_cells_near_face =
        std::max(cases_met.largest_cells_near_face, check.cases_met.largest_cells_near_face);
    cases_met.faces_behind_absent_neighbours += check.cases_met.faces_behind_absent_neighbours;
    cases_met.faces_two_size_levels_away += check.cases_met.faces_two_size_levels_away;
    cases_met.grid_piece_walks += check.cases_met.grid_piece_walks;
    CheckNoMistake("FindCellsNearFace: its definition", mistakes.cells_near_face_not_the_definition);
    CheckNoMistake("FindCellsNearFace: the checker's cells", mistakes.cells_near_face_not_the_checkers);
    CheckNoMistake("FindCellsNearFace: strictly ascending", mistakes.cells_near_face_out_of_order);
    CheckNoMistake(
        "FindCellsNearFace: kNoIndex from count on",
        mistakes.cells_near_face_entries_past_count_not_no_index
    );
    CheckNoMistake("FindNextFace: the filtered checker's face pieces", mistakes.walks_not_the_checkers);
    CheckNoMistake("FindNextFace: strictly ascending", mistakes.walks_out_of_order);
    CheckNoMistake("BuildFaceSearch: blocks strictly ascending", mistakes.search_blocks_out_of_order);
    CheckNoMistake(
        "grid piece: FindNextFace gives the kept face pieces",
        mistakes.grid_piece_walks_not_the_whole_grids
    );
    CheckNoMistake(
        "grid piece: FindCellsNearFace within the bound",
        mistakes.grid_piece_cells_near_face_past_the_bound
    );
  }
  RecordProperty("largest_cells_near_face_in_random_grids", cases_met.largest_cells_near_face);
  EXPECT_LE(cases_met.largest_cells_near_face, kMaxCellsNearFace);
  EXPECT_GT(cases_met.faces_behind_absent_neighbours, 0U);
  EXPECT_GT(cases_met.faces_two_size_levels_away, 0U);
  EXPECT_GT(cases_met.grid_piece_walks, 0U);
}

}  // namespace
}  // namespace fluidity::grid
