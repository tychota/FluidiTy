#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <random>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {
namespace {

using BlockNumbers = std::set<std::uint32_t>;

BlockNumbers ToBlockNumbers(std::span<const CellBlockIndex> blocks) {
  BlockNumbers numbers;
  for (const CellBlockIndex block : blocks) {
    numbers.insert(block.value);
  }
  return numbers;
}

std::vector<CellBlockIndex> ToBlockList(const BlockNumbers& numbers) {
  std::vector<CellBlockIndex> blocks;
  for (const std::uint32_t number : numbers) {
    blocks.push_back({.value = number});
  }
  return blocks;
}

bool ComputeIsAscendingWithoutRepeat(std::span<const CellBlockIndex> blocks) {
  return std::ranges::adjacent_find(
             blocks,
             [](CellBlockIndex first, CellBlockIndex second) {
               return first.value >= second.value;
             }
         )
      == blocks.end();
}

// The border of a set, which every check below asks for. It fails the calling test when the call fails.
std::vector<CellBlockIndex>
FindBorderOrFail(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width) {
  const std::expected<std::vector<CellBlockIndex>, GridFailure> border = FindBorderBlocks(grid, blocks, width);
  EXPECT_TRUE(border.has_value());
  return border.has_value() ? *border : std::vector<CellBlockIndex>{};
}

// The cut of a grid into ranges of at most max_blocks_per_range blocks. It fails the calling test when the call fails.
std::vector<CellBlockRange> BuildRangesOrFail(const StaggeredGrid& grid, std::uint32_t max_blocks_per_range) {
  const std::expected<std::vector<CellBlockRange>, GridFailure> ranges = BuildRanges(grid, max_blocks_per_range);
  EXPECT_TRUE(ranges.has_value());
  return ranges.has_value() ? *ranges : std::vector<CellBlockRange>{};
}

// One mark per block of a grid: 1 for a block read, 0 for a block not read.
using BlockMarks = std::vector<std::uint8_t>;

// What each block's places read, by width: per block, the block of every place a reader returned or listed, run
// from every active cell, face piece and corner place kept in the block. A block's own mark is set.
struct ReaderReadBlocks {
  std::vector<BlockMarks> touching_cells;
  std::vector<BlockMarks> four_cell_widths;
};

// Runs the kTouchingCells readers that start from an active cell: FindTouchingCells, FindFacePieces of it and of each
// touching cell, FindCornerPlaces, FindCellsAroundEdge of its 12 edges and FindCellAt from its centre.
void ApplyCellReaders(const GridTables& tables, CellPlaceIndex cell, BlockMarks& blocks_read) {
  blocks_read.at(ToCellBlockIndex(cell).value) = 1;
  const auto insert_face_pieces = [&tables, &blocks_read](CellPlaceIndex read_cell) {
    const CellFacePieces face_pieces = FindFacePieces(tables, read_cell);
    for (int entry = 0; entry < face_pieces.count; ++entry) {
      blocks_read.at(ToCellBlockIndex(face_pieces.faces[entry]).value) = 1;
    }
  };
  insert_face_pieces(cell);
  const TouchingCells touching = FindTouchingCells(tables, cell);
  for (int entry = 0; entry < touching.count; ++entry) {
    blocks_read.at(ToCellBlockIndex(touching.cells[entry]).value) = 1;
    insert_face_pieces(touching.cells[entry]);
  }
  for (const CornerPlaceIndex corner : FindCornerPlaces(tables, cell).corners) {
    blocks_read.at(ToCellBlockIndex(corner).value) = 1;
  }
  for (const auto& [first_axis, second_axis] :
       {std::array{Axis::kX, Axis::kY}, std::array{Axis::kX, Axis::kZ}, std::array{Axis::kY, Axis::kZ}}) {
    for (int quarter = 0; quarter < 4; ++quarter) {
      for (const CellPlaceIndex around : FindCellsAroundEdge(tables, cell, first_axis, second_axis, quarter).cells) {
        if (around.value != kNoIndex) {
          blocks_read.at(ToCellBlockIndex(around).value) = 1;
        }
      }
    }
  }
  const CellPlaceIndex found = FindCellAt(tables, ComputeCellCenter(tables, cell), ToCellBlockIndex(cell));
  blocks_read.at(ToCellBlockIndex(found).value) = 1;
}

// Runs FindNearbyFaces on one axis for a position inside the cell, off its centre so the window is not symmetric.
void ApplyNearbyFacesReader(const GridTables& tables, CellPlaceIndex cell, Axis axis, BlockMarks& blocks_read) {
  const Vec3 center = ComputeCellCenter(tables, cell);
  const double cell_width = ComputeCellWidth(tables.frame, ToSizeLevel(tables, ToCellBlockIndex(cell)));
  const Vec3 position = {
      .x = center.x + (0.3 * cell_width),
      .y = center.y - (0.2 * cell_width),
      .z = center.z + (0.4 * cell_width),
  };
  GridError error = GridError::kNone;
  const NearbyFaceList list = FindNearbyFaces(tables, position, axis, ToCellBlockIndex(cell), &error);
  EXPECT_EQ(error, GridError::kNone);
  for (int entry = 0; entry < list.count; ++entry) {
    blocks_read.at(ToCellBlockIndex(list.faces[entry]).value) = 1;
  }
}

// Runs the readers that start from a face piece but the face search: FindCellLow, FindCellHigh and FindCornerPlaces at
// kTouchingCells, and FindCellsNearFace at kFourCellWidths.
void ApplyFaceReaders(
    const GridTables& tables,
    FacePlaceIndex face,
    BlockMarks& blocks_read,
    BlockMarks& wide_blocks_read
) {
  blocks_read.at(ToCellBlockIndex(face).value) = 1;
  for (const CellPlaceIndex face_cell : {FindCellLow(tables, face), FindCellHigh(tables, face)}) {
    if (face_cell.value != kNoIndex) {
      blocks_read.at(ToCellBlockIndex(face_cell).value) = 1;
    }
  }
  for (const CornerPlaceIndex corner : FindCornerPlaces(tables, face).corners) {
    blocks_read.at(ToCellBlockIndex(corner).value) = 1;
  }
  const CellsNearFace cells_near_face = FindCellsNearFace(tables, face);
  for (int entry = 0; entry < cells_near_face.count; ++entry) {
    wide_blocks_read.at(ToCellBlockIndex(cells_near_face.cells[entry]).value) = 1;
  }
}

// Finds the block list of BuildFaceSearch from `face` at the largest radius, in its own order.
std::vector<std::uint32_t> FindFaceSearchBlocks(const GridTables& tables, FacePlaceIndex face) {
  const FaceSearch search = BuildFaceSearch(tables, face, kMaxFaceSearchRadius);
  std::vector<std::uint32_t> blocks;
  blocks.reserve(static_cast<std::size_t>(search.block_count));
  for (int entry = 0; entry < search.block_count; ++entry) {
    blocks.push_back(search.blocks[entry].value);
  }
  return blocks;
}

// Walks FindNextFace from `face` at the largest radius and adds the block of each face piece it gives.
void ApplyFaceWalk(const GridTables& tables, FacePlaceIndex face, BlockMarks& wide_blocks_read) {
  FaceSearch search = BuildFaceSearch(tables, face, kMaxFaceSearchRadius);
  for (FacePlaceIndex found = FindNextFace(tables, &search); found.value != kNoIndex;
       found = FindNextFace(tables, &search)) {
    wide_blocks_read.at(ToCellBlockIndex(found).value) = 1;
  }
}

// Runs the cell readers from every active cell of `block`, and FindNearbyFaces from every 4th active cell of the grid,
// counted by `active_cell_number`, on one axis in turn.
void ApplyCellReadersOfBlock(
    const GridTables& tables,
    std::uint32_t block,
    int& active_cell_number,
    BlockMarks& blocks_read
) {
  for (std::uint32_t place_in_block = 0; place_in_block < kCellPlacesPerBlock; ++place_in_block) {
    const CellPlaceIndex cell = {.value = (block * kCellPlacesPerBlock) + place_in_block};
    if (tables.cell_states[cell.value] != CellState::kActive) {
      continue;
    }
    ApplyCellReaders(tables, cell, blocks_read);
    if (active_cell_number % 4 == 0) {
      ApplyNearbyFacesReader(tables, cell, static_cast<Axis>((active_cell_number / 4) % 3), blocks_read);
    }
    ++active_cell_number;
  }
}

// Runs the face readers from every face piece of `block`; BuildFaceSearch from the first face piece of each axis,
// whose lists must be equal; and FindNextFace from the first face piece of the axis numbered block mod 3.
void ApplyFaceReadersOfBlock(
    const GridTables& tables,
    std::uint32_t block,
    BlockMarks& blocks_read,
    BlockMarks& wide_blocks_read
) {
  std::array<bool, 3> has_searched_axis = {false, false, false};
  std::vector<std::uint32_t> first_search_blocks;
  for (std::uint32_t place_in_block = 0; place_in_block < kFacePlacesPerBlock; ++place_in_block) {
    const FacePlaceIndex face = {.value = (block * kFacePlacesPerBlock) + place_in_block};
    if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
      continue;
    }
    ApplyFaceReaders(tables, face, blocks_read, wide_blocks_read);
    const auto axis_index = static_cast<std::size_t>(ToAxis(face));
    if (has_searched_axis.at(axis_index)) {
      continue;
    }
    has_searched_axis.at(axis_index) = true;
    const std::vector<std::uint32_t> search_blocks = FindFaceSearchBlocks(tables, face);
    for (const std::uint32_t search_block : search_blocks) {
      wide_blocks_read.at(search_block) = 1;
    }
    if (first_search_blocks.empty()) {
      first_search_blocks = search_blocks;
    }
    EXPECT_EQ(search_blocks, first_search_blocks) << "block " << block << ", axis " << axis_index;
    if (axis_index == block % 3) {
      ApplyFaceWalk(tables, face, wide_blocks_read);
    }
  }
}

// Runs every reader from every place of every block once, with two readers run from fewer places. FindNearbyFaces
// runs from every 4th active cell, on one axis in turn: it lists face pieces of the cell and of its touching cells,
// which FindFacePieces gives from every cell. BuildFaceSearch runs from the first face piece of each axis in each
// block, and FindNextFace walks from the one of the axis numbered block mod 3: BuildFaceSearch chooses its blocks
// from the position of the start's block alone (its Method), and the lists of a block's axes are checked equal here.
// A corner place's one reader, ComputeCornerPosition, reads its own block's id and the frame.
ReaderReadBlocks BuildReaderReadBlocks(const GridTables& tables) {
  ReaderReadBlocks read_blocks = {
      .touching_cells = std::vector<BlockMarks>(tables.block_count, BlockMarks(tables.block_count, 0)),
      .four_cell_widths = std::vector<BlockMarks>(tables.block_count, BlockMarks(tables.block_count, 0)),
  };
  int active_cell_number = 0;
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    BlockMarks& blocks_read = read_blocks.touching_cells.at(block);
    BlockMarks& wide_blocks_read = read_blocks.four_cell_widths.at(block);
    blocks_read.at(block) = 1;
    ApplyCellReadersOfBlock(tables, block, active_cell_number, blocks_read);
    ApplyFaceReadersOfBlock(tables, block, blocks_read, wide_blocks_read);
    for (std::size_t read_block = 0; read_block < blocks_read.size(); ++read_block) {
      wide_blocks_read.at(read_block) |= blocks_read.at(read_block);
    }
  }
  return read_blocks;
}

// The random grids on which the readers run from every place: the first 100. This is the costly part of the checks.
constexpr std::size_t kReaderGridCount = 100;

// The blocks read by the readers of each of the first kReaderGridCount random grids, found once per program.
const std::vector<ReaderReadBlocks>& FindReaderReadBlocks() {
  static const std::vector<ReaderReadBlocks> read_blocks_of_grids = [] {
    const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
    std::vector<ReaderReadBlocks> found;
    found.reserve(kReaderGridCount);
    for (std::size_t grid = 0; grid < kReaderGridCount; ++grid) {
      found.push_back(BuildReaderReadBlocks(built.grids.at(grid).grid.tables(RunOn::kCpu)));
    }
    return found;
  }();
  return read_blocks_of_grids;
}

// The chosen sets of one random grid and their borders at both widths.
struct GridBorderCase {
  std::vector<std::vector<CellBlockIndex>> sets;
  std::vector<std::vector<CellBlockIndex>> touching_borders;  // FindBorderBlocks of each set at kTouchingCells.
  std::vector<std::vector<CellBlockIndex>> wide_borders;  // FindBorderBlocks of each set at kFourCellWidths.
};

// The sets and borders of every random grid, found once per program.
const std::vector<GridBorderCase>& FindGridBorderCases() {
  static const std::vector<GridBorderCase> cases = [] {
    const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
    std::vector<GridBorderCase> found;
    found.reserve(built.grids.size());
    for (std::size_t grid = 0; grid < built.grids.size(); ++grid) {
      const StaggeredGrid& staggered_grid = built.grids.at(grid).grid;
      GridBorderCase grid_case = {
          .sets = FindChosenBlockSets(grid, staggered_grid),
          .touching_borders = {},
          .wide_borders = {},
      };
      for (const std::vector<CellBlockIndex>& blocks : grid_case.sets) {
        grid_case.touching_borders.push_back(FindBorderOrFail(staggered_grid, blocks, BorderWidth::kTouchingCells));
        grid_case.wide_borders.push_back(FindBorderOrFail(staggered_grid, blocks, BorderWidth::kFourCellWidths));
      }
      found.push_back(std::move(grid_case));
    }
    return found;
  }();
  return cases;
}

// The blocks read in every random grid, found from the geometry once per program.
const std::vector<GeometricReadBlocks>& FindGeometricReadBlocks() {
  static const std::vector<GeometricReadBlocks> read_blocks_of_grids = [] {
    const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
    std::vector<GeometricReadBlocks> found;
    found.reserve(built.grids.size());
    for (std::size_t grid = 0; grid < built.grids.size(); ++grid) {
      found.push_back(BuildGeometricReadBlocks(built.checkers.at(grid), built.grids.at(grid).grid.tables(RunOn::kCpu)));
    }
    return found;
  }();
  return read_blocks_of_grids;
}

// The blocks that the places kept in `blocks` read, by `read_blocks_of_width`, outside `blocks`.
BlockNumbers
FindReadOutside(const std::vector<BlockMarks>& read_blocks_of_width, std::span<const CellBlockIndex> blocks) {
  const BlockNumbers set = ToBlockNumbers(blocks);
  BlockNumbers read_outside;
  for (const std::uint32_t block : set) {
    const BlockMarks& marks = read_blocks_of_width.at(block);
    for (std::uint32_t read_block = 0; read_block < marks.size(); ++read_block) {
      if (marks.at(read_block) != 0 && !set.contains(read_block)) {
        read_outside.insert(read_block);
      }
    }
  }
  return read_outside;
}

// Checks that the border of each set of random grid `grid` at `width` is the one FindBorderGeometrically finds.
void CheckBordersEqualGeometricBorders(std::size_t grid, BorderWidth width) {
  const GridBorderCase& grid_case = FindGridBorderCases().at(grid);
  for (std::size_t set_number = 0; set_number < grid_case.sets.size(); ++set_number) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set_number);
    const std::vector<CellBlockIndex>& border = width == BorderWidth::kTouchingCells
        ? grid_case.touching_borders.at(set_number)
        : grid_case.wide_borders.at(set_number);
    const std::vector<CellBlockIndex> geometric_border =
        FindBorderGeometrically(FindGeometricReadBlocks().at(grid), grid_case.sets.at(set_number), width);
    EXPECT_EQ(ToBlockNumbers(border), ToBlockNumbers(geometric_border));
  }
}

TEST(FindBorderBlocks, TouchingCells_EqualsGeometricReadSet) {
  ASSERT_EQ(FindGridBorderCases().size(), kRandomGridCount);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    CheckBordersEqualGeometricBorders(grid, BorderWidth::kTouchingCells);
  }
}

TEST(FindBorderBlocks, FourCellWidths_EqualsGeometricReadSet) {
  ASSERT_EQ(FindGridBorderCases().size(), kRandomGridCount);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    CheckBordersEqualGeometricBorders(grid, BorderWidth::kFourCellWidths);
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

// Checks, for every block of random grid `grid` alone, that its border at each width, by the forward form, is the
// set of blocks it reads by the geometry (FindBorderGeometrically).
void CheckEveryBlockAloneEqualsGeometricBorder(std::size_t grid) {
  const GridTables tables = FindRandomGridsWithCheckers().grids.at(grid).grid.tables(RunOn::kCpu);
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", block " << block);
    const std::vector<CellBlockIndex> set = {{.value = block}};
    const std::array<BlockNumbers, 2> borders = FindBlockBordersByReadMarks(tables, block);
    const GeometricReadBlocks& read_blocks = FindGeometricReadBlocks().at(grid);
    EXPECT_EQ(borders.at(0), ToBlockNumbers(FindBorderGeometrically(read_blocks, set, BorderWidth::kTouchingCells)));
    EXPECT_EQ(borders.at(1), ToBlockNumbers(FindBorderGeometrically(read_blocks, set, BorderWidth::kFourCellWidths)));
  }
}

// The border of each block alone, found forward from the places its readers read (ApplyCellPlaceReadMarks and
// ApplyFacePlaceReadMarks), is the set of blocks it reads by the geometry, for every block of every random grid at
// both widths. The border of a set is the union of the borders of its blocks with their bigger blocks, less the set,
// in both computations, so this covers every set; BlockByBlockForm_EqualsForwardForm ties the other form to it.
TEST(FindBorderBlocks, EveryBlockAlone_EqualsGeometricReadSet) {
  ASSERT_EQ(FindGeometricReadBlocks().size(), kRandomGridCount);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    CheckEveryBlockAloneEqualsGeometricBorder(grid);
  }
}

// The random grids on which the two forms are compared: their chosen sets, and every block alone of the first ones.
constexpr std::size_t kTwoFormsGridCount = 60;
constexpr std::size_t kBlockAloneGridCount = 10;

// The border of a set by the form that asks each block from its own side (ComputeIsBorderBlock): the form of
// FindBorderBlocks that takes an output span, which has no working memory. It fails the calling test when the call
// fails or the span is too short.
std::vector<CellBlockIndex>
FindBorderBlockByBlockOrFail(const StaggeredGrid& grid, std::span<const CellBlockIndex> blocks, BorderWidth width) {
  std::vector<CellBlockIndex> border(grid.counts().cell_blocks);
  const std::expected<std::uint32_t, GridFailure> count = FindBorderBlocks(grid, blocks, width, border);
  EXPECT_TRUE(count.has_value());
  const std::uint32_t border_count = count.value_or(0);
  EXPECT_LE(border_count, border.size());
  border.resize(std::min<std::size_t>(border_count, border.size()));
  return border;
}

// Checks that the forward form (FindBorderByReadMarks) gives, for each chosen set of random grid `grid`, the borders
// the form that asks each block from its own side finds.
void CheckForwardFormOnChosenSets(std::size_t grid) {
  const GridTables tables = FindRandomGridsWithCheckers().grids.at(grid).grid.tables(RunOn::kCpu);
  const GridBorderCase& grid_case = FindGridBorderCases().at(grid);
  for (std::size_t set_number = 0; set_number < grid_case.sets.size(); ++set_number) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set_number);
    const std::vector<CellBlockIndex>& set = grid_case.sets.at(set_number);
    const StaggeredGrid& staggered_grid = FindRandomGridsWithCheckers().grids.at(grid).grid;
    for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
      EXPECT_EQ(
          ToBlockNumbers(FindBorderByReadMarks(tables, set, width)),
          ToBlockNumbers(FindBorderBlockByBlockOrFail(staggered_grid, set, width))
      );
    }
  }
}

// Checks that the forward form and the form that asks each block from its own side give the same border for every
// block alone of random grid `grid`.
void CheckForwardFormOnEveryBlockAlone(std::size_t grid) {
  const StaggeredGrid& staggered_grid = FindRandomGridsWithCheckers().grids.at(grid).grid;
  const GridTables tables = staggered_grid.tables(RunOn::kCpu);
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", block " << block);
    const std::vector<CellBlockIndex> set = {{.value = block}};
    for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
      EXPECT_EQ(
          ToBlockNumbers(FindBorderByReadMarks(tables, set, width)),
          ToBlockNumbers(FindBorderBlockByBlockOrFail(staggered_grid, set, width))
      );
    }
  }
}

// The form that asks each block from its own side (FindBorderBlocks with an output span, through
// ComputeIsBorderBlock) and the forward form, in which every place of the set marks what its readers read
// (FindBorderByReadMarks, through ApplyCellPlaceReadMarks and ApplyFacePlaceReadMarks), give the same border: for
// every chosen set of the first 60 random grids, and for every block alone of the first 10, at both widths.
TEST(FindBorderBlocks, BlockByBlockForm_EqualsForwardForm) {
  ASSERT_EQ(FindGridBorderCases().size(), kRandomGridCount);
  for (std::size_t grid = 0; grid < kTwoFormsGridCount; ++grid) {
    CheckForwardFormOnChosenSets(grid);
  }
  for (std::size_t grid = 0; grid < kBlockAloneGridCount; ++grid) {
    CheckForwardFormOnEveryBlockAlone(grid);
  }
}

// Checks that a border is ascending, repeats no block and holds no block of `set`.
void CheckBorderIsAscendingOutsideSet(const BlockNumbers& set, const std::vector<CellBlockIndex>& border) {
  EXPECT_TRUE(ComputeIsAscendingWithoutRepeat(border));
  EXPECT_TRUE(std::ranges::none_of(border, [&set](CellBlockIndex block) {
    return set.contains(block.value);
  }));
}

TEST(FindBorderBlocks, RandomSets_GiveAscendingBlocksOutsideTheSet) {
  const std::vector<GridBorderCase>& cases = FindGridBorderCases();
  ASSERT_EQ(cases.size(), kRandomGridCount);
  for (std::size_t grid = 0; grid < cases.size(); ++grid) {
    const GridBorderCase& grid_case = cases.at(grid);
    for (std::size_t set_number = 0; set_number < grid_case.sets.size(); ++set_number) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set_number);
      const BlockNumbers set = ToBlockNumbers(grid_case.sets.at(set_number));
      CheckBorderIsAscendingOutsideSet(set, grid_case.touching_borders.at(set_number));
      CheckBorderIsAscendingOutsideSet(set, grid_case.wide_borders.at(set_number));
    }
  }
}

// Checks that every block of `read_outside` is in `border`.
void CheckBorderHoldsBlocksRead(const BlockNumbers& read_outside, const std::vector<CellBlockIndex>& border) {
  const BlockNumbers border_numbers = ToBlockNumbers(border);
  for (const std::uint32_t block : read_outside) {
    EXPECT_TRUE(border_numbers.contains(block)) << "block " << block << " is read and not in the border";
  }
}

// Checks, for each set of random grid `grid`, that the places the readers of each width return outside the set lie in
// its border.
void CheckReadersReturnPlacesOfBorders(std::size_t grid) {
  const GridBorderCase& grid_case = FindGridBorderCases().at(grid);
  const ReaderReadBlocks& reader_read_blocks = FindReaderReadBlocks().at(grid);
  for (std::size_t set_number = 0; set_number < grid_case.sets.size(); ++set_number) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set_number);
    const std::vector<CellBlockIndex>& blocks = grid_case.sets.at(set_number);
    CheckBorderHoldsBlocksRead(
        FindReadOutside(reader_read_blocks.touching_cells, blocks),
        grid_case.touching_borders.at(set_number)
    );
    CheckBorderHoldsBlocksRead(
        FindReadOutside(reader_read_blocks.four_cell_widths, blocks),
        grid_case.wide_borders.at(set_number)
    );
  }
}

// Every place that a reader of the width returns or lists, run from every place kept in the blocks of a set, is kept
// in the set or its border. The border holds more: the blocks the readers pass through, which the piece test
// (border_blocks_in_grid_pieces.test.cc) shows are needed. The first 100 random grids, 4 sets each.
TEST(FindBorderBlocks, EveryReaderFromEveryPlace_ReturnsPlacesOfSetPlusBorder) {
  ASSERT_EQ(FindReaderReadBlocks().size(), kReaderGridCount);
  for (std::size_t grid = 0; grid < kReaderGridCount; ++grid) {
    ASSERT_EQ(FindGridBorderCases().at(grid).sets.size(), 4U);
    CheckReadersReturnPlacesOfBorders(grid);
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
// in random grid `grid` is kept in a block of `held_blocks`.
void CheckFacesWithinFourWidthsAreHeld(std::size_t grid, FacePlaceIndex face, const BlockNumbers& held_blocks) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
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
TEST(FindBorderBlocks, FourCellWidths_HoldsEveryFaceWithinFourWidths) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  const std::vector<GridBorderCase>& cases = FindGridBorderCases();
  ASSERT_EQ(cases.size(), kRandomGridCount);
  for (std::size_t grid = 0; grid < cases.size(); ++grid) {
    const GridTables tables = built.grids.at(grid).grid.tables(RunOn::kCpu);
    const GridBorderCase& grid_case = cases.at(grid);
    for (std::size_t set_number = 0; set_number < grid_case.sets.size(); ++set_number) {
      SCOPED_TRACE(testing::Message() << "random grid " << grid << ", set " << set_number);
      const BlockNumbers set = ToBlockNumbers(grid_case.sets.at(set_number));
      BlockNumbers held_blocks = ToBlockNumbers(grid_case.wide_borders.at(set_number));
      held_blocks.insert(set.begin(), set.end());
      for (const FacePlaceIndex face : FindSpreadFacePiecesKeptIn(tables, set)) {
        CheckFacesWithinFourWidthsAreHeld(grid, face, held_blocks);
      }
    }
  }
}

// Checks the four layers of stacked borders around `range` in random grid `grid`, for stages of widths (touching,
// touching, four widths, touching), found from the last stage backwards: every place that the stage running on the
// blocks loaded so far returns outside them lies in the next layer.
void CheckStackedLayersAreRead(std::size_t grid, CellBlockRange range) {
  constexpr std::array<BorderWidth, 4> kStageWidths = {
      BorderWidth::kTouchingCells,
      BorderWidth::kTouchingCells,
      BorderWidth::kFourCellWidths,
      BorderWidth::kTouchingCells,
  };
  SCOPED_TRACE(testing::Message() << "random grid " << grid << ", range from block " << range.first_block);
  const StaggeredGrid& staggered_grid = FindRandomGridsWithCheckers().grids.at(grid).grid;
  const ReaderReadBlocks& reader_read_blocks = FindReaderReadBlocks().at(grid);
  BlockNumbers loaded = ToBlockNumbers(ToRangeBlocks(range));
  for (std::size_t layer = 0; layer < kStageWidths.size(); ++layer) {
    const BorderWidth width = kStageWidths.at(kStageWidths.size() - 1 - layer);
    const std::vector<CellBlockIndex> stage_blocks = ToBlockList(loaded);
    const BlockNumbers next_layer = ToBlockNumbers(FindBorderOrFail(staggered_grid, stage_blocks, width));
    const std::vector<BlockMarks>& read_blocks_of_width =
        width == BorderWidth::kTouchingCells ? reader_read_blocks.touching_cells : reader_read_blocks.four_cell_widths;
    SCOPED_TRACE(testing::Message() << "layer " << layer);
    CheckBorderHoldsBlocksRead(FindReadOutside(read_blocks_of_width, stage_blocks), ToBlockList(next_layer));
    loaded.insert(next_layer.begin(), next_layer.end());
  }
}

// A pass of four stages of widths (touching, touching, four widths, touching) on the ranges of 4 blocks of 50 random
// grids. The layers are found from the last stage backwards: L1 = border(R, touching), L2 = border(R + L1, four
// widths), and so on. Every place that the stage running on R + L1 + ... + L(k-1) returns outside the blocks it runs
// on lies in Lk.
TEST(FindBorderBlocks, StackedBorders_HoldEveryRead) {
  constexpr std::size_t kStackedGridCount = 50;
  ASSERT_GE(FindReaderReadBlocks().size(), kStackedGridCount);
  for (std::size_t grid = 0; grid < kStackedGridCount; ++grid) {
    for (const CellBlockRange range : BuildRangesOrFail(FindRandomGridsWithCheckers().grids.at(grid).grid, 4)) {
      CheckStackedLayersAreRead(grid, range);
    }
  }
}

// The active cells kept in the blocks of a range.
std::vector<CellPlaceIndex> FindActiveCellsOfRange(const GridTables& tables, CellBlockRange range) {
  std::vector<CellPlaceIndex> range_cells;
  const std::uint32_t first_place = range.first_block * kCellPlacesPerBlock;
  for (std::uint32_t place = first_place; place < first_place + (range.block_count * kCellPlacesPerBlock); ++place) {
    if (tables.cell_states[place] == CellState::kActive) {
      range_cells.push_back({.value = place});
    }
  }
  return range_cells;
}

// Checks that every face FindNearbyFaces lists, on each axis, for `position` in `cell` is kept in the range or its
// border.
void CheckNearbyFacesAreHeld(
    const GridTables& tables,
    const RangeWithBorder& range_with_border,
    CellPlaceIndex cell,
    Vec3 position
) {
  const CellBlockRange range = range_with_border.range;
  const BlockNumbers border = ToBlockNumbers(range_with_border.border_blocks);
  for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
    GridError error = GridError::kNone;
    const NearbyFaceList list = FindNearbyFaces(tables, position, axis, ToCellBlockIndex(cell), &error);
    ASSERT_EQ(error, GridError::kNone);
    for (int entry = 0; entry < list.count; ++entry) {
      const std::uint32_t face_block = ToCellBlockIndex(list.faces[entry]).value;
      const bool is_in_range = face_block - range.first_block < range.block_count;
      EXPECT_TRUE(is_in_range || border.contains(face_block)) << "face block " << face_block;
    }
  }
}

// Every face FindNearbyFaces lists for a position in a cell of a range is kept in the range or in its border at
// kTouchingCells: 100 positions in each of 50 random grids cut into ranges of 3 blocks.
TEST(FindNearbyFaces, RangeWithBorder_HoldsEveryFace) {
  constexpr std::size_t kNearbyGridCount = 50;
  constexpr int kPositionsPerGrid = 100;
  int positions_checked = 0;
  for (std::size_t grid = 0; grid < kNearbyGridCount; ++grid) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid);
    const StaggeredGrid& staggered_grid = FindRandomGridsWithCheckers().grids.at(grid).grid;
    const GridTables tables = staggered_grid.tables(RunOn::kCpu);
    const std::vector<CellBlockRange> cut = BuildRangesOrFail(staggered_grid, 3);
    std::vector<RangeWithBorder> ranges;
    ranges.reserve(cut.size());
    for (const CellBlockRange range : cut) {
      ranges.push_back(BuildRangeWithBorder(staggered_grid, range));
    }
    std::mt19937_64 generator(60000 + grid);
    for (int position_number = 0; position_number < kPositionsPerGrid; ++position_number) {
      const RangeWithBorder& range_with_border = ranges.at(ToUniformIndex(generator(), ranges.size()));
      const std::vector<CellPlaceIndex> range_cells = FindActiveCellsOfRange(tables, range_with_border.range);
      if (range_cells.empty()) {
        continue;
      }
      const CellPlaceIndex cell = range_cells.at(ToUniformIndex(generator(), range_cells.size()));
      const Vec3 center = ComputeCellCenter(tables, cell);
      const double cell_width = ComputeCellWidth(tables.frame, ToSizeLevel(tables, ToCellBlockIndex(cell)));
      // A position inside the cell: its centre moved by less than half its width on each axis.
      const Vec3 position = {
          .x = center.x + ((ToUnitDouble(generator()) - 0.5) * cell_width),
          .y = center.y + ((ToUnitDouble(generator()) - 0.5) * cell_width),
          .z = center.z + ((ToUnitDouble(generator()) - 0.5) * cell_width),
      };
      CheckNearbyFacesAreHeld(tables, range_with_border, cell, position);
      ++positions_checked;
    }
  }
  EXPECT_GT(positions_checked, 0);
}

}  // namespace
}  // namespace fluidity::grid
