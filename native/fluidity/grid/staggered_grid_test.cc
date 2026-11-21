#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/allocation_counter_test_support.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/grid_test_support.h"

namespace fluidity::grid {
namespace {

// A frame with its origin at 0 and biggest cells 1 m wide.
GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

// The room of the buffers the random grids are built into, one after the other: they have at most 252 blocks.
constexpr std::uint32_t kRandomGridRoom = 4096;

CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// The inputs of a CPU list of version 1.
GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

// The 8 small cells of a big cell, by the test's own arithmetic: 2 c and 2 c + 1 on each axis, in eighth order.
std::vector<CellId> ToSmallCells(CellId big_cell) {
  const CellCoord coord = ToCellCoord(big_cell);
  const int small_size_level = static_cast<int>(ToSizeLevel(big_cell)) + 1;
  std::vector<CellId> small_cells;
  for (std::int32_t dz = 0; dz < 2; ++dz) {
    for (std::int32_t dy = 0; dy < 2; ++dy) {
      for (std::int32_t dx = 0; dx < 2; ++dx) {
        small_cells.push_back(ToCell(small_size_level, (2 * coord.x) + dx, (2 * coord.y) + dy, (2 * coord.z) + dz));
      }
    }
  }
  return small_cells;
}

// Size-level-0 cells {4 i, 0, 0} for i = 0..count - 1: each alone in a block of its own.
std::vector<CellId> BuildCellsInBlocksOfTheirOwn(std::int32_t count) {
  std::vector<CellId> cells;
  cells.reserve(static_cast<std::size_t>(count));
  for (std::int32_t index = 0; index < count; ++index) {
    cells.push_back(ToCell(0, 4 * index, 0, 0));
  }
  return cells;
}

void ExpectFailure(
    const std::expected<StaggeredGrid, GridFailure>& result,
    GridError error,
    CellId cell = CellId{0},
    CellId other_cell = CellId{0}
) {
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, error);
  EXPECT_EQ(result.error().cell.value, cell.value);
  EXPECT_EQ(result.error().other_cell.value, other_cell.value);
  EXPECT_EQ(result.error().magnitude, 0U);
}

// floor(coord / 4) and coord - 4 floor(coord / 4), by integer division and a correction for a negative remainder: the
// test's own arithmetic, apart from the shifts the grid uses.
std::int32_t ToBlockCoord(std::int32_t coord) {
  const std::int32_t quotient = coord / 4;
  return (coord % 4 != 0 && coord < 0) ? quotient - 1 : quotient;
}

std::int32_t ToLocalCoord(std::int32_t coord) {
  return coord - (4 * ToBlockCoord(coord));
}

// floor(coord / 2), as ToBlockCoord.
std::int32_t ToFloorOfHalf(std::int32_t coord) {
  const std::int32_t quotient = coord / 2;
  return (coord % 2 != 0 && coord < 0) ? quotient - 1 : quotient;
}

// A cell place as the test names it: size level, then x, y and z at that size level.
using PlaceKey = std::tuple<int, std::int32_t, std::int32_t, std::int32_t>;

PlaceKey ToPlaceKey(CellId cell) {
  const CellCoord coord = ToCellCoord(cell);
  return {static_cast<int>(ToSizeLevel(cell)), coord.x, coord.y, coord.z};
}

// The index of each block of a grid, found by its id: the test's own lookup, apart from the grid's binary search.
std::map<std::uint64_t, std::uint32_t> BuildBlockIndexes(const GridTables& tables) {
  std::map<std::uint64_t, std::uint32_t> block_indexes;
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    block_indexes[tables.block_ids[block].value] = block;
  }
  return block_indexes;
}

// The state of every cell place of a grid, by the place's size level and coordinate. A block id holds its block
// coordinate as a cell id holds a cell coordinate, so the place of local (x, y, z) in block b is 4 b + local.
std::map<PlaceKey, CellState> BuildPlaceStates(const GridTables& tables) {
  std::map<PlaceKey, CellState> place_states;
  for (std::uint32_t block = 0; block < tables.block_count; ++block) {
    const CellId block_as_cell = {.value = tables.block_ids[block].value};
    const CellCoord block_coord = ToCellCoord(block_as_cell);
    const int size_level = static_cast<int>(ToSizeLevel(block_as_cell));
    for (int place_in_block = 0; place_in_block < kCellPlacesPerBlock; ++place_in_block) {
      const PlaceKey place = {
          size_level,
          (4 * block_coord.x) + (place_in_block % 4),
          (4 * block_coord.y) + ((place_in_block / 4) % 4),
          (4 * block_coord.z) + (place_in_block / 16),
      };
      const std::size_t place_index =
          (std::size_t{block} * kCellPlacesPerBlock) + static_cast<std::size_t>(place_in_block);
      place_states[place] = tables.cell_states[place_index];
    }
  }
  return place_states;
}

// The face place of a local place of block `block` along `axis`, the coordinate along the axis being the plane 0..4:
// the layout of GridTables (grid/grid_tables.h), written out by the test.
std::uint32_t ToFacePlace(Axis axis, std::uint32_t block, std::array<std::int32_t, 3> local) {
  const std::int32_t x = local.at(0);
  const std::int32_t y = local.at(1);
  const std::int32_t z = local.at(2);
  std::int32_t place_on_axis = 0;
  if (axis == Axis::kX) {
    place_on_axis = (((z * 4) + y) * 5) + x;
  } else if (axis == Axis::kY) {
    place_on_axis = (((z * 5) + y) * 4) + x;
  } else {
    place_on_axis = (((z * 4) + y) * 4) + x;
  }
  return (block * kFacePlacesPerBlock) + (80 * static_cast<std::uint32_t>(axis))
      + static_cast<std::uint32_t>(place_on_axis);
}

// The kind the method paper gives a face piece with these cells beside it (<sec-face-pieces>).
FacePieceKind ToExpectedKind(const GeometricFacePiece& face_piece) {
  if (face_piece.cell_low.has_value() && face_piece.cell_high.has_value()) {
    return FacePieceKind::kBetweenCells;
  }
  return face_piece.cell_high.has_value() ? FacePieceKind::kGridEdgeLow : FacePieceKind::kGridEdgeHigh;
}

// The block tables of the fixture of OneSplitCell_GivesTheWorkedTables, in a room of 8 blocks.
void ExpectOneSplitCellBlockTables(const GridTables& tables) {
  const std::vector<std::uint64_t> block_ids = {tables.block_ids[0].value, tables.block_ids[1].value};
  EXPECT_EQ(block_ids, (std::vector<std::uint64_t>{ToCell(0, 0, 0, 0).value, ToCell(1, 0, 0, 0).value}));
  EXPECT_EQ(
      std::vector<std::uint32_t>(tables.bigger_block, tables.bigger_block + 2),
      (std::vector<std::uint32_t>{kNoIndex, 0})
  );
  EXPECT_EQ(tables.smaller_blocks, tables.bigger_block + 8);
  std::vector<std::uint32_t> expected_smaller_blocks(16, kNoIndex);
  expected_smaller_blocks.at(0) = 1;
  EXPECT_EQ(std::vector<std::uint32_t>(tables.smaller_blocks, tables.smaller_blocks + 16), expected_smaller_blocks);
}

// The cell states of the fixture of OneSplitCell_GivesTheWorkedTables.
void ExpectOneSplitCellStates(const GridTables& tables) {
  std::map<std::pair<int, CellState>, int> state_counts;
  for (int place = 0; place < 2 * kCellPlacesPerBlock; ++place) {
    ++state_counts[{place / kCellPlacesPerBlock, tables.cell_states[place]}];
  }
  const std::map<std::pair<int, CellState>, int> expected_state_counts = {
      {{0, CellState::kAbsent}, 62},
      {{0, CellState::kActive}, 1},
      {{0, CellState::kSplit}, 1},
      {{1, CellState::kAbsent}, 48},
      {{1, CellState::kInsideBiggerCell}, 8},
      {{1, CellState::kActive}, 8},
  };
  EXPECT_EQ(state_counts, expected_state_counts);
  // Block 0: place 0 the active cell, place 1 the split cell. Block 1: size level 1 place (1, 1, 1), at 21, lies inside
  // the big cell; (2, 1, 1), at 22, is a small cell.
  const std::vector<CellState> chosen_states = {
      tables.cell_states[0],
      tables.cell_states[1],
      tables.cell_states[64 + 21],
      tables.cell_states[64 + 22],
  };
  const std::vector<CellState> expected_chosen_states = {
      CellState::kActive,
      CellState::kSplit,
      CellState::kInsideBiggerCell,
      CellState::kActive,
  };
  EXPECT_EQ(chosen_states, expected_chosen_states);
}

// ---------------------------------------------------------------------------------------------------------------------
// Fixtures with their numbers worked out
// ---------------------------------------------------------------------------------------------------------------------

// Size level 0 {0,0,0} beside the 8 size-level-1 cells of size level 0 {1,0,0}, 3 size levels. Blocks in the fixed
// order: 0 is size level 0 (0,0,0), holding the active cell at place 0 and the split cell {1,0,0} at place 1; 1 is
// size level 1 (0,0,0), whose cells x = 0..3 hold the small cells at x = 2..3, y and z 0..1. Block 1's bigger block is
// 0; block 0's smaller block of eighth 0 is 1, its other eighths none. Block 0: 1 active, 1 split, 62 absent places.
// Block 1: 8 active; the 8 places x = 0..1, y and z 0..1 lie inside the active big cell; 48 absent. Face pieces: the
// big cell's 5 whole sides, the 4 quarters between it and the small cells, 12 sides between small cells and their 20
// other outer sides: 41. Active cells 9: 1 of size level 0, 8 of size level 1. In a room of 8 blocks the array of
// bigger and smaller blocks holds the 8 bigger blocks first, so block 0's smaller blocks start at entry 8.
TEST(BuildGrid, OneSplitCell_GivesTheWorkedTables) {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  const std::vector<CellId> small_cells = ToSmallCells(ToCell(0, 1, 0, 0));
  cells.insert(cells.end(), small_cells.begin(), small_cells.end());
  // A room of 8 blocks, more than the 2 used, so the smaller blocks start after the room's 8 bigger blocks.
  const std::expected<TestGrid, GridFailure> test_grid =
      BuildTestGrid(ToInputs(ToUnitFrame(3), cells), {.max_cell_blocks = 8});
  ASSERT_TRUE(test_grid.has_value());
  const StaggeredGrid& grid = test_grid->grid;
  EXPECT_EQ(grid.version().value, 1U);
  GridCounts expected_counts{.cell_blocks = 2, .active_cells_per_size_level = {}, .active_cells = 9, .face_pieces = 41};
  expected_counts.active_cells_per_size_level[0] = 1;
  expected_counts.active_cells_per_size_level[1] = 8;
  EXPECT_EQ(ToGridCountFields(grid.counts()), ToGridCountFields(expected_counts));

  const GridTables tables = grid.tables(RunOn::kCpu);
  ASSERT_EQ(tables.block_count, 2U);
  ExpectOneSplitCellBlockTables(tables);
  ExpectOneSplitCellStates(tables);
}

TEST(BuildGrid, EmptyList_GivesEmptyGrid) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), {}));
  ASSERT_TRUE(test_grid.has_value());
  EXPECT_EQ(test_grid->grid.version().value, 1U);
  EXPECT_EQ(ToGridCountFields(test_grid->grid.counts()), ToGridCountFields(GridCounts{}));
  EXPECT_EQ(test_grid->grid.all_blocks().first_block, 0U);
  EXPECT_EQ(test_grid->grid.all_blocks().block_count, 0U);
}

// The 7 size-level-1 cells of size level 0 {0,0,0} without eighth 5, size level 1 {1,0,1}: eighth 5 is not eighth 0,
// so the failure names it and its big cell.
TEST(BuildGrid, LoneSmallCell_IsSiblingCellMissing) {
  std::vector<CellId> cells = ToSmallCells(ToCell(0, 0, 0, 0));
  cells.erase(cells.begin() + 5);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ExpectFailure(
      BuildGrid(ToInputs(ToUnitFrame(3), cells), buffers),
      GridError::kSiblingCellMissing,
      ToCell(1, 1, 0, 1),
      ToCell(0, 0, 0, 0)
  );
}

// 9 blocks into a room of 8: step 5 of <proc-build-grid> counts the blocks against the room. The failed build leaves
// the buffers with their arrays, and they then build the first 8 cells.
TEST(BuildGrid, AboveBuffersRoom_IsTooManyCellBlocks) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(9);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ExpectFailure(BuildGrid(ToInputs(ToUnitFrame(3), cells), buffers), GridError::kTooManyCellBlocks);
  EXPECT_EQ(buffers.max_cell_blocks(), 8U);
  const std::expected<StaggeredGrid, GridFailure> grid =
      BuildGrid(ToInputs(ToUnitFrame(3), std::span<const CellId>(cells).first(8)), buffers);
  ASSERT_TRUE(grid.has_value());
  EXPECT_EQ(grid->counts().cell_blocks, 8U);
}

// 64 x 8 + 1 = 513 size-level-0 cells {i, 0, 0} into a room of 8 blocks: the length check, before any id is read.
std::vector<CellId> BuildMoreCellsThanEightBlocksHold() {
  constexpr std::int32_t kCellCount = (64 * 8) + 1;
  std::vector<CellId> cells;
  cells.reserve(kCellCount);
  for (std::int32_t index = 0; index < kCellCount; ++index) {
    cells.push_back(ToCell(0, index, 0, 0));
  }
  return cells;
}

TEST(BuildGrid, MoreCellsThanPlaces_IsTooManyCellBlocks) {
  const std::vector<CellId> cells = BuildMoreCellsThanEightBlocksHold();
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ExpectFailure(BuildGrid(ToInputs(ToUnitFrame(3), cells), buffers), GridError::kTooManyCellBlocks);
}

// The same list with an id whose bit 63 is clear and one at kMaxCoord + 1: the length check still comes first, so the
// bad ids are never read.
TEST(BuildGrid, TooLongListWithBadId_IsTooManyCellBlocks) {
  std::vector<CellId> cells = BuildMoreCellsThanEightBlocksHold();
  cells.at(3) = CellId{.value = cells.at(3).value & ~(std::uint64_t{1} << 63U)};
  cells.at(7) = ToCell(0, kMaxCoord + 1, 0, 0);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(3), {}), {.max_cell_blocks = 8});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ExpectFailure(BuildGrid(ToInputs(ToUnitFrame(3), cells), buffers), GridError::kTooManyCellBlocks);
}

// The 64 size-level-1 cells of the 8 size-level-0 cells {0..1, 0..1, 0..1} fill the 64 cell places of one block, so
// the length check passes. Each size-level-1 cell at even coordinates is eighth 0 and adds its size-level-0 cell: 8
// split cells, 72 cells above the 64 places. This is step 5 of <proc-build-grid>, found once the split cells are known.
TEST(BuildGrid, SplitCellsOverflowThePlaces_IsTooManyCellBlocks) {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 2; ++z) {
    for (std::int32_t y = 0; y < 2; ++y) {
      for (std::int32_t x = 0; x < 2; ++x) {
        const std::vector<CellId> small_cells = ToSmallCells(ToCell(0, x, y, z));
        cells.insert(cells.end(), small_cells.begin(), small_cells.end());
      }
    }
  }
  ASSERT_EQ(cells.size(), 64U);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(2), {}), {.max_cell_blocks = 1});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ExpectFailure(BuildGrid(ToInputs(ToUnitFrame(2), cells), buffers), GridError::kTooManyCellBlocks);
}

// The 64 size-level-0 cells {0..3, 0..3, 0..3} fill the 64 cell places of one block, with no split cell: 64 cells, not
// above 64 x 1, so a room of 1 builds. 1 block, 64 active cells, and 3 axes x 5 planes x 16 = 240 face pieces.
TEST(BuildGrid, CellPlacesExactlyFull_Builds) {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 4; ++z) {
    for (std::int32_t y = 0; y < 4; ++y) {
      for (std::int32_t x = 0; x < 4; ++x) {
        cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(2), {}), {.max_cell_blocks = 1});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  const std::expected<StaggeredGrid, GridFailure> grid = BuildGrid(ToInputs(ToUnitFrame(2), cells), buffers);
  ASSERT_TRUE(grid.has_value());
  GridCounts expected_counts{
      .cell_blocks = 1,
      .active_cells_per_size_level = {},
      .active_cells = 64,
      .face_pieces = 240,
  };
  expected_counts.active_cells_per_size_level[0] = 64;
  EXPECT_EQ(ToGridCountFields(grid->counts()), ToGridCountFields(expected_counts));
}

// In a room of 1 block, whose builder buffers hold 64 cell ids: the 56 size-level-0 cells of block (0,0,0) with z <= 2,
// or z = 3 and y <= 1, then the 8 size-level-1 cells of the 57th place, size level 0 {0,2,3}. That is 64 active cells,
// which pass the length check. The small cell {0,4,6} is eighth 0 and adds its split cell {0,2,3}: 65 cells, one
// above the 64 places, so step 5 fails with kTooManyCellBlocks and no cell, before any block is counted.
TEST(BuildGrid, OneCellAboveThePlaces_IsTooManyCellBlocks) {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 4; ++z) {
    for (std::int32_t y = 0; y < 4; ++y) {
      for (std::int32_t x = 0; x < 4; ++x) {
        const bool is_listed = z <= 2 || y <= 1;
        if (is_listed) {
          cells.push_back(ToCell(0, x, y, z));
        }
      }
    }
  }
  const std::vector<CellId> small_cells = ToSmallCells(ToCell(0, 0, 2, 3));
  cells.insert(cells.end(), small_cells.begin(), small_cells.end());
  ASSERT_EQ(cells.size(), 64U);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(ToUnitFrame(2), {}), {.max_cell_blocks = 1});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ExpectFailure(BuildGrid(ToInputs(ToUnitFrame(2), cells), buffers), GridError::kTooManyCellBlocks);
}

TEST(BuildTestGrid, DefaultSettings_SizedToTheList) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(9);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(test_grid.has_value());
  EXPECT_EQ(test_grid->grid.max_cell_blocks(), 9U);
  EXPECT_EQ(test_grid->grid.counts().cell_blocks, 9U);
}

TEST(StaggeredGrid, CpuGrid_HasNoGpuTables) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(2);
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), cells));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables gpu_tables = test_grid->grid.tables(RunOn::kGpu);
  EXPECT_EQ(gpu_tables.version.value, 0U);
  EXPECT_EQ(gpu_tables.block_ids, nullptr);
  EXPECT_EQ(gpu_tables.same_size_level_blocks, nullptr);
  EXPECT_EQ(gpu_tables.bigger_block, nullptr);
  EXPECT_EQ(gpu_tables.smaller_blocks, nullptr);
  EXPECT_EQ(gpu_tables.cell_states, nullptr);
  EXPECT_EQ(gpu_tables.face_piece_kinds, nullptr);
}

// ---------------------------------------------------------------------------------------------------------------------
// The random grids
// ---------------------------------------------------------------------------------------------------------------------

// The version a random grid is built as: grid + 2, so each grid has its own version above 1.
GridVersion ToRandomGridVersion(std::size_t grid) {
  return {.value = static_cast<std::uint64_t>(grid) + 2};
}

// Builds the 200 random grids in turn, as version grid + 2, into one set of buffers for kRandomGridRoom blocks, hands
// each built grid to read_grid(grid, built) and takes the buffers back. Each test builds the grids it reads in its own
// process, since the gate runs every test as its own process. A grid that fails to build fails the calling test and is
// not handed on.
template <typename ReadGrid>
void ApplyToRandomGrids(const ReadGrid& read_grid) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = kRandomGridRoom});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    SCOPED_TRACE(grid);
    GridInputs inputs = ToInputs(frame, BuildShuffledRandomGrids().at(grid));
    inputs.version = ToRandomGridVersion(grid);
    std::expected<StaggeredGrid, GridFailure> built = BuildGrid(inputs, buffers);
    EXPECT_TRUE(built.has_value());
    if (!built.has_value()) {
      continue;
    }
    read_grid(grid, *built);
    buffers = TakeBuffers(std::move(*built));
  }
}

// The cells sorted with ComesBefore.
std::vector<CellId> BuildCellsInFixedOrder(std::vector<CellId> cells) {
  std::ranges::sort(cells, [](CellId first, CellId second) {
    return ComesBefore(first, second);
  });
  return cells;
}

// Expects grid i, built from its shuffled list and from its sorted list, to carry version i + 2 both times: the grids
// and their tables.
void ExpectRandomGridVersion(const StaggeredGrid& shuffled, const StaggeredGrid& sorted, std::size_t grid) {
  EXPECT_EQ(shuffled.version().value, grid + 2);
  EXPECT_EQ(shuffled.tables(RunOn::kCpu).version.value, grid + 2);
  EXPECT_EQ(sorted.version().value, grid + 2);
  EXPECT_EQ(sorted.tables(RunOn::kCpu).version.value, grid + 2);
}

// Each of the 200 lists is shuffled; the same list sorted with ComesBefore gives the same bytes in every table. Grid i
// is built as version i + 2 both times, and the grids and their tables carry that version. The shuffled lists are
// built into one set of buffers and the sorted lists into a second, each taken back after each grid.
TEST(BuildGrid, AnyInputOrder_GivesSameTables) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = kRandomGridRoom});
  GridBuffers sorted_buffers = TakeEmptyGridBuffers(owner);
  ApplyToRandomGrids([&](std::size_t grid, const StaggeredGrid& shuffled) {
    const std::vector<CellId> sorted_cells = BuildCellsInFixedOrder(BuildShuffledRandomGrids().at(grid));
    GridInputs inputs = ToInputs(frame, sorted_cells);
    inputs.version = ToRandomGridVersion(grid);
    std::expected<StaggeredGrid, GridFailure> sorted = BuildGrid(inputs, sorted_buffers);
    ASSERT_TRUE(sorted.has_value());
    ExpectRandomGridVersion(shuffled, *sorted, grid);
    EXPECT_TRUE(ComputeIsSameUsedTables(shuffled.tables(RunOn::kCpu), sorted->tables(RunOn::kCpu)));
    EXPECT_EQ(ToGridCountFields(shuffled.counts()), ToGridCountFields(sorted->counts()));
    sorted_buffers = TakeBuffers(std::move(*sorted));
  });
}

// Where the checker's face pieces sit in the tables, over all random grids.
struct FacePlacesFound {
  int kept_at_plane_zero_of_high_block = 0;  // On a block's high side, with the high block present.
  int kept_at_plane_four = 0;  // On a block's high side, with no block of that size level beyond.
};

// The index of the block of size_level at a block coordinate, or kNoIndex: the test's own lookup.
std::uint32_t FindBlockIndex(
    const std::map<std::uint64_t, std::uint32_t>& block_indexes,
    int size_level,
    std::array<std::int32_t, 3> block_coord
) {
  const CellId block_as_cell = ToCell(size_level, block_coord.at(0), block_coord.at(1), block_coord.at(2));
  const auto found_block = block_indexes.find(block_as_cell.value);
  return found_block == block_indexes.end() ? kNoIndex : found_block->second;
}

// Finds the face place that keeps one checker face piece. It is at the place of the cell on the face piece's high side,
// plane = local coordinate along its axis, when that place's block exists. Otherwise the high place is local 0 of a
// block that does not exist, and the face piece is at plane 4 of the block below. When the high block exists and the
// face piece is at its plane 0, plane 4 of the block below, if any, must be kNotAPiece; that place is returned as well.
struct FacePiecePlaces {
  std::uint32_t kept = kNoIndex;  // Where the face piece is kept.
  // Plane 4 of the block below, when the face piece is at plane 0 above.
  std::uint32_t plane_four_left_empty = kNoIndex;
  bool is_at_plane_four = false;  // Kept at plane 4: no block of its size level lies on the high side.
};

FacePiecePlaces
FindFacePiecePlaces(const std::map<std::uint64_t, std::uint32_t>& block_indexes, const GeometricFacePiece& face_piece) {
  const auto axis_index = static_cast<std::size_t>(face_piece.axis);
  const int size_level = static_cast<int>(face_piece.size_level);
  const std::array<std::int32_t, 3> high_coord = {
      face_piece.cell_high_coord.x,
      face_piece.cell_high_coord.y,
      face_piece.cell_high_coord.z,
  };
  std::array<std::int32_t, 3> block_coord{};
  std::array<std::int32_t, 3> local{};
  for (std::size_t index = 0; index < 3; ++index) {
    block_coord.at(index) = ToBlockCoord(high_coord.at(index));
    local.at(index) = ToLocalCoord(high_coord.at(index));
  }
  std::array<std::int32_t, 3> low_block_coord = block_coord;
  low_block_coord.at(axis_index) -= 1;
  std::array<std::int32_t, 3> plane_four_local = local;
  plane_four_local.at(axis_index) = 4;
  const bool is_on_block_high_side = local.at(axis_index) == 0;
  const std::uint32_t high_block = FindBlockIndex(block_indexes, size_level, block_coord);
  const std::uint32_t low_block =
      is_on_block_high_side ? FindBlockIndex(block_indexes, size_level, low_block_coord) : kNoIndex;
  if (high_block != kNoIndex) {
    const std::uint32_t plane_four =
        low_block == kNoIndex ? kNoIndex : ToFacePlace(face_piece.axis, low_block, plane_four_local);
    return {
        .kept = ToFacePlace(face_piece.axis, high_block, local),
        .plane_four_left_empty = plane_four,
        .is_at_plane_four = false,
    };
  }
  // No block holds the high place, so it lies beyond a block's high side, and the block below keeps the face piece.
  EXPECT_TRUE(is_on_block_high_side);
  if (low_block == kNoIndex) {
    ADD_FAILURE() << "no block keeps face piece " << face_piece.id.value;
    return {};
  }
  return {
      .kept = ToFacePlace(face_piece.axis, low_block, plane_four_local),
      .plane_four_left_empty = kNoIndex,
      .is_at_plane_four = true,
  };
}

// Expects the face piece kinds of one built grid to be the checker's: each checker face piece at exactly one place
// (FindFacePiecePlaces), with the kind its cells give, and kNotAPiece at every other place.
void ExpectCheckerFacePieceKinds(const GridTables& tables, const GeometricGrid& checker, FacePlacesFound& found) {
  const std::map<std::uint64_t, std::uint32_t> block_indexes = BuildBlockIndexes(tables);
  std::map<std::uint32_t, FacePieceKind> expected_kinds;
  for (const GeometricFacePiece& face_piece : checker.FacePieces()) {
    const FacePiecePlaces places = FindFacePiecePlaces(block_indexes, face_piece);
    if (places.plane_four_left_empty != kNoIndex) {
      ++found.kept_at_plane_zero_of_high_block;
      // Plane 4 of the block below is the same place in space; the face piece is kept once, above.
      expected_kinds.emplace(places.plane_four_left_empty, FacePieceKind::kNotAPiece);
    }
    if (places.is_at_plane_four) {
      ++found.kept_at_plane_four;
    }
    const bool is_new_place = expected_kinds.emplace(places.kept, ToExpectedKind(face_piece)).second;
    EXPECT_TRUE(is_new_place) << "two face pieces at face place " << places.kept;
  }
  std::size_t wrong_kind_count = 0;
  const std::size_t face_place_count = std::size_t{tables.block_count} * kFacePlacesPerBlock;
  for (std::size_t face_place = 0; face_place < face_place_count; ++face_place) {
    const auto expected = expected_kinds.find(static_cast<std::uint32_t>(face_place));
    const FacePieceKind expected_kind = expected == expected_kinds.end() ? FacePieceKind::kNotAPiece : expected->second;
    wrong_kind_count += tables.face_piece_kinds[face_place] == expected_kind ? 0 : 1;
  }
  EXPECT_EQ(wrong_kind_count, 0U);
}

// All 200 random grids; the shared rules' own test compares on 40 seeds.
TEST(BuildGrid, FacePieceKinds_MatchGeometricChecker) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  FacePlacesFound found;
  ApplyToRandomGrids([&](std::size_t grid, const StaggeredGrid& built) {
    const GeometricGrid checker(frame, BuildShuffledRandomGrids().at(grid));
    ExpectCheckerFacePieceKinds(built.tables(RunOn::kCpu), checker, found);
    EXPECT_EQ(built.counts().face_pieces, checker.FacePieces().size());
  });
  // Both ways of keeping a face piece on a block's high side occur, so both were compared.
  EXPECT_GT(found.kept_at_plane_zero_of_high_block, 0);
  EXPECT_GT(found.kept_at_plane_four, 0);
}

// Expects the cell states of one built grid from the list alone: kActive exactly at the listed cells, kSplit exactly
// at the bigger cells that hold one (found by halving, the test's own arithmetic), kInsideBiggerCell exactly at the
// places whose cell one size level bigger is active, kAbsent elsewhere.
void ExpectCheckerCellStates(const GridTables& tables, const std::vector<CellId>& cells) {
  std::set<PlaceKey> active_places;
  std::set<PlaceKey> split_places;
  for (const CellId cell : cells) {
    active_places.insert(ToPlaceKey(cell));
    auto [size_level, x, y, z] = ToPlaceKey(cell);
    while (size_level > 0) {
      --size_level;
      x = ToFloorOfHalf(x);
      y = ToFloorOfHalf(y);
      z = ToFloorOfHalf(z);
      split_places.insert({size_level, x, y, z});
    }
  }
  std::size_t wrong_state_count = 0;
  std::size_t active_count = 0;
  std::size_t split_count = 0;
  for (const auto& [place, state] : BuildPlaceStates(tables)) {
    const auto [size_level, x, y, z] = place;
    const PlaceKey covering_place = {size_level - 1, ToFloorOfHalf(x), ToFloorOfHalf(y), ToFloorOfHalf(z)};
    CellState expected_state = CellState::kAbsent;
    if (active_places.contains(place)) {
      expected_state = CellState::kActive;
      ++active_count;
    } else if (split_places.contains(place)) {
      expected_state = CellState::kSplit;
      ++split_count;
    } else if (size_level > 0 && active_places.contains(covering_place)) {
      expected_state = CellState::kInsideBiggerCell;
    }
    if (state != expected_state) {
      ++wrong_state_count;
    }
  }
  EXPECT_EQ(wrong_state_count, 0U);
  // Every listed cell and every split cell has its place in a block of the grid.
  EXPECT_EQ(active_count, active_places.size());
  EXPECT_EQ(split_count, split_places.size());
}

// Expects the kSplit places of one built grid to be exactly the cells the shared rules add: ComputeSplitCellCountToAdd
// and ComputeBigCellAtSizeLevel over the list.
void ExpectAddedSplitCellsMarkedSplit(const GridTables& tables, const std::vector<CellId>& cells) {
  std::set<PlaceKey> added_places;
  for (const CellId cell : cells) {
    const int size_level = static_cast<int>(ToSizeLevel(cell));
    for (int added = 1; added <= ComputeSplitCellCountToAdd(cell); ++added) {
      added_places.insert(ToPlaceKey(ComputeBigCellAtSizeLevel(cell, static_cast<SizeLevel>(size_level - added))));
    }
  }
  std::set<PlaceKey> split_places;
  for (const auto& [place, state] : BuildPlaceStates(tables)) {
    if (state == CellState::kSplit) {
      split_places.insert(place);
    }
  }
  EXPECT_EQ(added_places, split_places);
}

// All 200 random grids.
TEST(BuildGrid, CellStates_MatchGeometricChecker) {
  ApplyToRandomGrids([](std::size_t grid, const StaggeredGrid& built) {
    ExpectCheckerCellStates(built.tables(RunOn::kCpu), BuildShuffledRandomGrids().at(grid));
  });
}

// All 200 random grids: the split cells of step 5 are exactly the places marked kSplit.
TEST(BuildGrid, AddedSplitCells_AreThePlacesMarkedSplit) {
  ApplyToRandomGrids([](std::size_t grid, const StaggeredGrid& built) {
    ExpectAddedSplitCellsMarkedSplit(built.tables(RunOn::kCpu), BuildShuffledRandomGrids().at(grid));
  });
}

// The addresses of the arrays of grid buffers, to see that a build moves none of them.
std::array<const void*, 6> ToArrayAddresses(const GridBuffers& buffers) {
  const GridBuffersArrays& arrays = ToGridBuffersArrays(buffers);
  return {
      arrays.block_ids.data(),
      arrays.same_size_level_neighbour_blocks.data(),
      arrays.bigger_and_smaller_blocks.data(),
      arrays.cell_states.data(),
      arrays.face_piece_kinds.data(),
      arrays.piece_map.data(),
  };
}

// What building the random grids in turn into one set of buffers found.
struct BuildsInTurn {
  std::size_t failed_build_count = 0;
  std::size_t different_grid_count = 0;  // Grids whose counts or used tables differ from the reference grid's.
  std::size_t moved_buffers_count = 0;  // Builds after which an array of the buffers lies elsewhere.
};

// Builds the random grid of each entry into `buffers`, compares it with its reference grid (same position) and takes
// the buffers back, in turn. Allocates nothing: it only counts what it finds.
BuildsInTurn ApplyBuildsInTurn(
    const std::vector<std::size_t>& grids,
    const std::vector<TestGrid>& reference_grids,
    GridBuffers& buffers
) {
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  const std::array<const void*, 6> array_addresses = ToArrayAddresses(buffers);
  BuildsInTurn builds;
  for (std::size_t position = 0; position < reference_grids.size(); ++position) {
    std::expected<StaggeredGrid, GridFailure> built =
        BuildGrid(ToInputs(frame, BuildShuffledRandomGrids()[grids[position]]), buffers);
    if (!built.has_value()) {
      ++builds.failed_build_count;
      continue;
    }
    builds.different_grid_count += ComputeIsSameGrid(*built, reference_grids[position].grid) ? 0 : 1;
    buffers = TakeBuffers(std::move(*built));
    builds.moved_buffers_count += ToArrayAddresses(buffers) == array_addresses ? 0 : 1;
  }
  return builds;
}

// 30 random grids spread by size (FindRandomGridsSpreadBySize: at evenly spaced ranks of the size order, from the grid
// with the most cells to the one with the fewest), built one after the other into buffers for 4,096 blocks and taken
// back each time, with the buffers' arrays never moving. The reference grids are built first, each with its own memory;
// the counted part builds, compares and takes back only. The build is meant to write only into buffers sized once for
// the room; this test checks it over small, middling and large grids, so an allocation that only some sizes reach still
// shows, and building 30 in turn checks that taking the buffers back and building again allocates nothing.
TEST(BuildGrid, IntoBuffers_AllocatesNothing) {
  constexpr std::size_t kGridCount = 30;
  const GridFrame frame = ToUnitFrame(kRandomGridSizeLevelCount);
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(kGridCount);
  const std::vector<TestGrid> reference_grids = BuildReferenceGrids(grids);
  ASSERT_EQ(reference_grids.size(), kGridCount);
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(ToInputs(frame, {}), {.max_cell_blocks = kRandomGridRoom});
  GridBuffers buffers = TakeEmptyGridBuffers(owner);

  const std::uint64_t allocation_count_before = ComputeAllocationCount();
  const BuildsInTurn builds = ApplyBuildsInTurn(grids, reference_grids, buffers);
  const std::uint64_t allocation_count_after = ComputeAllocationCount();

  EXPECT_EQ(allocation_count_after - allocation_count_before, 0U);
  EXPECT_EQ(builds.failed_build_count, 0U);
  EXPECT_EQ(builds.different_grid_count, 0U);
  EXPECT_EQ(builds.moved_buffers_count, 0U);
}

// A request whose aligned block would need more than SIZE_MAX bytes fails as its form says, counted, rather than
// wrapping the size to a small block: null from the nothrow form, std::bad_alloc from the throwing one.
TEST(ComputeAllocationCount, AlignedRequestNearSizeMax_FailsWithoutWrapping) {
  // Read through a volatile, so the compiler sees no constant size to warn about.
  const volatile std::size_t near_size_max = std::numeric_limits<std::size_t>::max() - 8;
  const std::size_t byte_count = near_size_max;
  constexpr std::align_val_t kAlignment{64};
  const std::uint64_t allocation_count_before = ComputeAllocationCount();
  void* const memory = ::operator new(byte_count, kAlignment, std::nothrow);
  EXPECT_EQ(memory, nullptr);
  EXPECT_THROW(::operator delete(::operator new(byte_count, kAlignment), kAlignment), std::bad_alloc);
  EXPECT_EQ(ComputeAllocationCount() - allocation_count_before, 2U);
}

}  // namespace
}  // namespace fluidity::grid
