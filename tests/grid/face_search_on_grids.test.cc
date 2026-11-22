#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <iterator>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/face_search_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {
namespace {

// A frame whose width is no power of two and whose origin is not 0. A face search counts in quarter widths from the
// origin, so it gives the same face pieces in this frame as in the unit frame.
constexpr GridFrame ToOffsetFrame(int size_level_count) {
  return {
      .origin = {.x = -1.3, .y = 2.9, .z = 0.1},
      .biggest_cell_width = 0.7,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

bool ComputeIsStrictlyAscending(const std::vector<std::uint32_t>& places) {
  return std::ranges::adjacent_find(places, std::ranges::greater_equal{}) == places.end();
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

// A hand-made grid for the readers: its name, its frame and its active cells.
struct NamedGrid {
  const char* name;
  GridFrame frame;
  std::vector<CellId> cells;
};

// Grids the random grids never give: face pieces at block borders, between two sizes, at the grid's edge, at negative
// coordinates and at both coordinate limits, a bigger block behind an absent neighbour, three size levels, and a frame
// with a width that is no power of two and an origin. 2^17 - 1 is the largest coordinate a size-level-0 cell may have
// and still be split.
std::vector<NamedGrid> BuildHandGrids() {
  constexpr auto kLargestSplitCoord = static_cast<std::int32_t>((1U << 17U) - 1U);
  std::vector<NamedGrid> grids;
  grids.push_back({.name = "two-size fixture", .frame = ToUnitFrame(2), .cells = BuildTwoSizeFixtureCells()});
  std::vector<CellId> negative_two_size_cells = {ToCell(0, -1, -1, -1)};
  ApplyAppendSmallCells(ToCell(0, 0, -1, -1), negative_two_size_cells);
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
  ApplyAppendSmallCells(ToCell(0, kLargestSplitCoord, 0, 0), high_split_cells);
  grids.push_back({.name = "small cells at the high split limit", .frame = ToUnitFrame(2), .cells = high_split_cells});
  std::vector<CellId> low_split_cells = {ToCell(0, -kLargestSplitCoord + 1, 0, 0)};
  ApplyAppendSmallCells(ToCell(0, -kLargestSplitCoord, 0, 0), low_split_cells);
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

// The size-level-0 cells of the box 0..7 on each axis, those of x 4..7, y 0..3, z 0..3 split, 2 size levels: 8
// size-level-0 blocks, of which (1,0,0) holds only split places, over the size-level-1 blocks (2..3, 0..1, 0..1).
std::vector<CellId> BuildBoxWithOneSplitBlock() {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z <= 7; ++z) {
    for (std::int32_t y = 0; y <= 7; ++y) {
      for (std::int32_t x = 0; x <= 7; ++x) {
        const bool is_in_split_block = x >= 4 && y <= 3 && z <= 3;
        if (is_in_split_block) {
          ApplyAppendSmallCells(ToCell(0, x, y, z), cells);
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
