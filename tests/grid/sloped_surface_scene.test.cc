// Runs the whole grid module on one realistic scene: a box of liquid with a sloped surface (the file header of
// grid/testing/sloped_surface_scene.h), a sheet of small cells through a box of big ones.
//
// The checks, in order: the grid's counts equal ComputeGridCounts and the counts written out below; the grid is cut
// into ranges of 8 blocks; for every range the stacked border of four layers (touching, four widths, touching,
// touching) is found with the buffer form of FindBorderBlocks and the grid piece is built from the range and all
// layers; every reader of the touching width and of the four-cell width answers in the piece as on the whole grid from
// every place of the range; and at the 1,000 positions, each taken in the piece of the range that holds its cell, the
// weights of the three axes have the whole grid's bits and add up to 1.

#include "fluidity/grid/testing/sloped_surface_scene.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_error_message.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/nearby_face_weights_cases.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"
#include "fluidity/grid/testing/touching_cells_cases.h"

namespace fluidity::grid {
namespace {

constexpr std::array<Axis, 3> kAxes = {Axis::kX, Axis::kY, Axis::kZ};

// The scene, built once per program: the cells, the grid, its ranges and the positions.
struct Scene {
  std::vector<CellId> cells;
  TestGrid test_grid;
  std::vector<CellBlockRange> ranges;
  std::vector<Vec3> positions;
};

const Scene& FindScene() {
  static const Scene scene = [] {
    std::vector<CellId> cells = BuildSlopedSurfaceCells();
    std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(BuildSlopedSurfaceFrame(), cells));
    EXPECT_TRUE(test_grid.has_value());
    std::expected<std::vector<CellBlockRange>, GridFailure> ranges =
        BuildRanges(test_grid->grid, kSlopedSurfaceRangeBlocks);
    EXPECT_TRUE(ranges.has_value());
    return Scene{
        .cells = std::move(cells),
        .test_grid = std::move(*test_grid),
        .ranges = std::move(*ranges),
        .positions = BuildSlopedSurfacePositions(),
    };
  }();
  return scene;
}

// The counts of the scene, computed once and read against the scene by hand. See CheckGridCounts for how they follow.
constexpr std::uint32_t kExpectedCellBlocks = 218;
constexpr std::uint64_t kExpectedActiveCells = 9584;
constexpr std::array<std::uint64_t, kSlopedSurfaceSizeLevelCount> kExpectedCellsPerSizeLevel = {288, 720, 8576};
constexpr std::size_t kExpectedRangeCount = 28;
constexpr std::size_t kExpectedRangesWithPositions = 27;
constexpr std::uint64_t kExpectedMarkedCells = 7680;

// The largest border of a range of 8 blocks, over the ranges of the scene: one layer, with no blocks found before it,
// at the touching width and at the four-cell-width width. A change in the border rule shows here first. A range of 8
// blocks gives about half of the 218 blocks for the touching width and more for the wider one. Both numbers are
// pinned from the border search on this scene, as a regression pin and not as a bound: a change to either is a change
// of the border rule to look at.
constexpr std::size_t kExpectedLargestNearBorder = 97;
constexpr std::size_t kExpectedLargestWideBorder = 174;

// The most blocks a piece of the scene holds: its range of 8 blocks and its stacked border of four layers. The stacked
// total, not the first layer, sets the size of the piece's memory; the widest piece here holds all 218 blocks.
constexpr std::uint32_t kExpectedLargestPieceBlocks = 218;

// Checks one count of the grid against the number expected, naming it on failure.
void CheckCount(const std::string& name, std::uint64_t actual, std::uint64_t expected) {
  EXPECT_EQ(actual, expected) << name;
}

// Counts the size level 2 cells of the box that the scene marks.
std::uint64_t ComputeMarkedCellCount() {
  std::uint64_t marked_count = 0;
  for (std::int32_t cell_z = 0; cell_z < kSlopedSurfaceCellsAlongZ * 4; ++cell_z) {
    for (std::int32_t cell_y = 0; cell_y < kSlopedSurfaceCellsAlongY * 4; ++cell_y) {
      for (std::int32_t cell_x = 0; cell_x < kSlopedSurfaceCellsAlongX * 4; ++cell_x) {
        marked_count += ComputeIsSlopedSurfaceCellMarked(cell_x, cell_z) ? 1 : 0;
      }
    }
  }
  return marked_count;
}

// Checks the grid's counts against ComputeGridCounts and against the numbers worked out from the scene.
//
// How the expected numbers follow from the scene. Each of the 16 layers of y of the box has 480 marked size level 2
// cells: 64 columns of x, 0.25 m wide, with 8 cells marked along z in the strip of 2 m about the plane, fewer where
// the strip leaves the box above z = 8 m (for x over 12 m). That is 7,680 marked cells, which the 8,576 active
// size level 2 cells cover: whole sets of 8 siblings, and the sets the size-step rule split further. The cells of the
// three size levels fill the box exactly: 288 x 64 + 720 x 8 + 8,576 = 32,768 = 512 x 64 size level 2 cells. The
// 218 blocks hold 64 places each, so the 8,576 size level 2 cells need at least 134 blocks, the 720 size level 1 cells
// at least 12 and the 288 size level 0 cells at least 5: 218 is more than those 151, as the cells of a sheet leave
// places of a block unused where it bends at the edge of the box.
void CheckGridCounts(const Scene& scene) {
  const std::expected<GridCounts, GridFailure> computed =
      ComputeGridCounts(ToInputs(BuildSlopedSurfaceFrame(), scene.cells));
  ASSERT_TRUE(computed.has_value());
  const GridCounts& counts = scene.test_grid.grid.counts();
  CheckCount("cell blocks, computed", counts.cell_blocks, computed->cell_blocks);
  CheckCount("active cells, computed", counts.active_cells, computed->active_cells);
  CheckCount("face pieces, computed", counts.face_pieces, computed->face_pieces);
  CheckCount("cell blocks", counts.cell_blocks, kExpectedCellBlocks);
  CheckCount("active cells", counts.active_cells, kExpectedActiveCells);
  CheckCount("ranges", scene.ranges.size(), kExpectedRangeCount);
  CheckCount("marked size level 2 cells", ComputeMarkedCellCount(), kExpectedMarkedCells);
  std::uint64_t volume_in_smallest_cells = 0;
  for (std::size_t size_level = 0; size_level < kSlopedSurfaceSizeLevelCount; ++size_level) {
    const std::string name = "active cells of size level " + std::to_string(size_level);
    const std::uint64_t cells = counts.active_cells_per_size_level[size_level];
    CheckCount(name + ", computed", cells, computed->active_cells_per_size_level[size_level]);
    CheckCount(name, cells, kExpectedCellsPerSizeLevel.at(size_level));
    const std::uint64_t smallest_cells_per_cell = std::uint64_t{1}
        << (3 * (kSlopedSurfaceSizeLevelCount - 1 - size_level));
    volume_in_smallest_cells += cells * smallest_cells_per_cell;
  }
  CheckCount("volume in the smallest cells", volume_in_smallest_cells, std::uint64_t{512} * 64);
}

// The place of a piece's face or cell in the whole grid: its block through the piece map, its place in the block kept.
// A block the piece map does not hold fails the check and gives kNoIndex.
FacePlaceIndex ToWholeFace(FacePlaceIndex piece_face, std::span<const CellBlockIndex> piece_map) {
  const auto places_per_block = static_cast<std::uint32_t>(kFacePlacesPerBlock);
  const std::size_t piece_block = piece_face.value / places_per_block;
  EXPECT_LT(piece_block, piece_map.size());
  if (piece_block >= piece_map.size()) {
    return {.value = kNoIndex};
  }
  const std::uint32_t whole_block = piece_map[piece_block].value;
  return {.value = (whole_block * places_per_block) + (piece_face.value % places_per_block)};
}

CellPlaceIndex ToWholeCell(CellPlaceIndex piece_cell, std::span<const CellBlockIndex> piece_map) {
  const auto places_per_block = static_cast<std::uint32_t>(kCellPlacesPerBlock);
  const std::size_t piece_block = piece_cell.value / places_per_block;
  EXPECT_LT(piece_block, piece_map.size());
  if (piece_block >= piece_map.size()) {
    return {.value = kNoIndex};
  }
  const std::uint32_t whole_block = piece_map[piece_block].value;
  return {.value = (whole_block * places_per_block) + (piece_cell.value % places_per_block)};
}

// Says whether two doubles have the same bits, or two vectors or matrices of doubles have the same bits in every entry.
bool ComputeHasSameBits(double first, double second) {
  return std::bit_cast<std::uint64_t>(first) == std::bit_cast<std::uint64_t>(second);
}

bool ComputeHasSameBits(Vec3 first, Vec3 second) {
  return ComputeHasSameBits(first.x, second.x) && ComputeHasSameBits(first.y, second.y)
      && ComputeHasSameBits(first.z, second.z);
}

bool ComputeHasSameBits(const Mat3& first, const Mat3& second) {
  const bool is_same_in_top_row = ComputeHasSameBits(first.xx, second.xx) && ComputeHasSameBits(first.xy, second.xy)
      && ComputeHasSameBits(first.xz, second.xz);
  const bool is_same_in_middle_row = ComputeHasSameBits(first.yx, second.yx) && ComputeHasSameBits(first.yy, second.yy)
      && ComputeHasSameBits(first.yz, second.yz);
  const bool is_same_in_bottom_row = ComputeHasSameBits(first.zx, second.zx) && ComputeHasSameBits(first.zy, second.zy)
      && ComputeHasSameBits(first.zz, second.zz);
  return is_same_in_top_row && is_same_in_middle_row && is_same_in_bottom_row;
}

// Checks that a number of the piece's weights has the same bits as the whole grid's, naming it on failure.
template <typename Value>
void CheckSameBits(const char* name, const Value& in_piece, const Value& on_whole_grid) {
  EXPECT_TRUE(ComputeHasSameBits(in_piece, on_whole_grid)) << name;
}

// Checks that two weights of a face have the same bits in every number.
void CheckFaceWeightsHaveSameBits(const FaceWeight& in_piece, const FaceWeight& on_whole_grid) {
  CheckSameBits("face weight", in_piece.weight, on_whole_grid.weight);
  CheckSameBits("face offset", in_piece.offset, on_whole_grid.offset);
  CheckSameBits("face offset from mean", in_piece.offset_from_mean, on_whole_grid.offset_from_mean);
  CheckSameBits("face weight change", in_piece.weight_change, on_whole_grid.weight_change);
}

// The tables a position is asked in: the whole grid's, the piece's, and the piece map.
struct WeightsInBoth {
  GridTables whole;
  GridTables piece;
  std::span<const CellBlockIndex> piece_map;
};

// Checks that the sums of a position made in a piece have the same bits as the whole grid's.
void CheckSumsHaveSameBits(const NearbyFaceWeights& in_piece, const NearbyFaceWeights& on_whole_grid) {
  CheckSameBits("cell width", in_piece.list.cell_width, on_whole_grid.list.cell_width);
  CheckSameBits("position", in_piece.position, on_whole_grid.position);
  CheckSameBits("weight sum", in_piece.weight_sum, on_whole_grid.weight_sum);
  CheckSameBits("mean offset", in_piece.mean_offset, on_whole_grid.mean_offset);
  CheckSameBits("raw weight change sum", in_piece.raw_weight_change_sum, on_whole_grid.raw_weight_change_sum);
  CheckSameBits("weighted spread", in_piece.weighted_spread, on_whole_grid.weighted_spread);
  CheckSameBits("inverse weighted spread", in_piece.inverse_weighted_spread, on_whole_grid.inverse_weighted_spread);
}

// Checks that the faces of a position's list in a piece are the whole grid's faces, the piece's places read through
// its map, and that each face's weight has the same bits.
void CheckFacesAreWholeGrids(
    const WeightsInBoth& tables,
    const NearbyFaceWeights& in_piece,
    const NearbyFaceWeights& on_whole_grid
) {
  for (int entry = 0; entry < on_whole_grid.list.count; ++entry) {
    EXPECT_EQ(ToWholeFace(in_piece.list.faces[entry], tables.piece_map).value, on_whole_grid.list.faces[entry].value);
    CheckFaceWeightsHaveSameBits(
        ComputeFaceWeight(tables.piece, in_piece, entry),
        ComputeFaceWeight(tables.whole, on_whole_grid, entry)
    );
  }
}

// Checks that the weights of a position made in a piece are the whole grid's: the same cell, list, mask, fit and
// completeness, the same bits in every sum and every face's weight, the piece's places read through its map.
void CheckWeightsAreWholeGrids(
    const WeightsInBoth& tables,
    const NearbyFaceWeights& in_piece,
    const NearbyFaceWeights& on_whole_grid
) {
  ASSERT_EQ(in_piece.list.count, on_whole_grid.list.count);
  EXPECT_EQ(ToWholeCell(in_piece.list.cell, tables.piece_map).value, on_whole_grid.list.cell.value);
  EXPECT_EQ(in_piece.list.axis, on_whole_grid.list.axis);
  EXPECT_TRUE(std::ranges::equal(in_piece.kept.bits, on_whole_grid.kept.bits));
  EXPECT_EQ(in_piece.fit, on_whole_grid.fit);
  EXPECT_EQ(in_piece.is_complete, on_whole_grid.is_complete);
  CheckSumsHaveSameBits(in_piece, on_whole_grid);
  CheckFacesAreWholeGrids(tables, in_piece, on_whole_grid);
}

// Checks the weights of the positions whose cell lies in the range at `range_entry`: on each of the three axes the
// piece gives the whole grid's weights by bits, and the weights of the piece add up to 1 within kWeightSumTolerance.
// Returns how many positions the range held.
std::size_t CheckPositionsOfRange(
    const Scene& scene,
    const GridPiece& piece,
    std::span<const std::size_t> range_of_position,
    std::size_t range_entry
) {
  const WeightsInBoth tables = {
      .whole = scene.test_grid.grid.tables(RunOn::kCpu),
      .piece = piece.grid().tables(RunOn::kCpu),
      .piece_map = piece.block_in_whole_grid(),
  };
  std::size_t position_count = 0;
  for (std::size_t entry = 0; entry < scene.positions.size(); ++entry) {
    if (range_of_position[entry] != range_entry) {
      continue;
    }
    ++position_count;
    const Vec3 position = scene.positions[entry];
    for (const Axis axis : kAxes) {
      SCOPED_TRACE(testing::Message() << "position " << entry << ", axis " << static_cast<int>(axis));
      const NearbyFaceWeights on_whole_grid = FindWeightsOfAllFaces(tables.whole, position, axis);
      const NearbyFaceWeights in_piece = FindWeightsOfAllFaces(tables.piece, position, axis);
      EXPECT_EQ(on_whole_grid.list.cell_width, kSlopedSurfaceSmallestCellWidth);
      CheckWeightsAreWholeGrids(tables, in_piece, on_whole_grid);
      EXPECT_NEAR(ComputeKeptWeightTotal(tables.piece, in_piece), 1.0, kWeightSumTolerance);
    }
  }
  return position_count;
}

// What the loop over the ranges collects: the largest borders, and how many ranges held positions.
struct RangeTotals {
  std::size_t largest_near_border = 0;
  std::size_t largest_wide_border = 0;
  std::uint32_t largest_piece_blocks = 0;
  std::size_t ranges_with_positions = 0;
  std::size_t positions_checked = 0;
  std::size_t pieces_smaller_than_grid = 0;  // Ranges whose piece holds fewer blocks than the whole grid.
};

// What the check of each range needs: the scene, the range of each position, the memory of the border search and of the
// pieces, used for every range in turn, and the totals.
struct RangeCheckContext {
  const Scene& scene;
  std::span<const std::size_t> range_of_position;
  BorderSearchRig& rig;
  PieceBuffersRoom& room;
  RangeTotals totals = {};
};

// Builds the stacked border of four layers of one range with the buffer form, builds the piece, and checks the readers
// and the positions of the range against the whole grid. A range of 8 blocks also gives the sizes of its single-layer
// borders to the totals.
void CheckRange(RangeCheckContext& context, std::size_t range_entry) {
  const Scene& scene = context.scene;
  const CellBlockRange range = scene.ranges[range_entry];
  SCOPED_TRACE(testing::Message() << "range " << range_entry << " from block " << range.first_block);
  const std::vector<CellBlockIndex> no_blocks;
  if (range.block_count == kSlopedSurfaceRangeBlocks) {
    const std::size_t near_border = FindLayerOrFail(context.rig, range, no_blocks, BorderWidth::kTouchingCells).size();
    const std::size_t wide_border = FindLayerOrFail(context.rig, range, no_blocks, BorderWidth::kFourCellWidths).size();
    context.totals.largest_near_border = std::max(context.totals.largest_near_border, near_border);
    context.totals.largest_wide_border = std::max(context.totals.largest_wide_border, wide_border);
  }
  std::vector<CellBlockIndex> stacked_border;
  for (const BorderWidth width : kLayerWidths) {
    stacked_border = FindLayerOrFail(context.rig, range, stacked_border, width);
  }
  std::expected<GridPiece, GridFailure> piece =
      BuildGridPiece(scene.test_grid.grid, range, stacked_border, context.room.buffers);
  ASSERT_TRUE(piece.has_value()) << ToName(piece.error().error);
  const GridTables whole = scene.test_grid.grid.tables(RunOn::kCpu);
  const LoadedPieceTables loaded = {
      .tables = piece->grid().tables(RunOn::kCpu),
      .block_in_whole = piece->block_in_whole_grid(),
  };
  const std::optional<std::string> difference =
      FindFirstReaderDifference(whole, loaded, ToRangeBlocks(range), BorderWidth::kTouchingCells);
  EXPECT_FALSE(difference.has_value()) << difference.value_or("");
  const std::optional<std::string> difference_at_four_widths =
      FindFirstReaderDifference(whole, loaded, ToRangeBlocks(range), BorderWidth::kFourCellWidths);
  EXPECT_FALSE(difference_at_four_widths.has_value()) << difference_at_four_widths.value_or("");
  context.totals.largest_piece_blocks =
      std::max(context.totals.largest_piece_blocks, piece->grid().counts().cell_blocks);
  const bool is_piece_smaller_than_grid =
      piece->grid().counts().cell_blocks < scene.test_grid.grid.counts().cell_blocks;
  context.totals.pieces_smaller_than_grid += is_piece_smaller_than_grid ? 1 : 0;
  const std::size_t position_count = CheckPositionsOfRange(scene, *piece, context.range_of_position, range_entry);
  context.totals.positions_checked += position_count;
  context.totals.ranges_with_positions += position_count > 0 ? 1 : 0;
  context.room.buffers = TakeBuffers(std::move(*piece));
}

// Checks the border sizes and the coverage that the loop over the ranges collected.
void CheckTotals(const RangeTotals& totals) {
  testing::Test::RecordProperty("largest_near_border", static_cast<int>(totals.largest_near_border));
  testing::Test::RecordProperty("largest_wide_border", static_cast<int>(totals.largest_wide_border));
  testing::Test::RecordProperty("largest_piece_blocks", static_cast<int>(totals.largest_piece_blocks));
  testing::Test::RecordProperty("ranges_with_positions", static_cast<int>(totals.ranges_with_positions));
  testing::Test::RecordProperty("pieces_smaller_than_grid", static_cast<int>(totals.pieces_smaller_than_grid));
  // A piece as big as the whole grid is compared with itself, so the comparison means something only for the pieces
  // that lack blocks.
  EXPECT_GT(totals.pieces_smaller_than_grid, 0U);
  CheckCount("largest near border", totals.largest_near_border, kExpectedLargestNearBorder);
  CheckCount("largest wide border", totals.largest_wide_border, kExpectedLargestWideBorder);
  CheckCount("largest piece blocks", totals.largest_piece_blocks, kExpectedLargestPieceBlocks);
  CheckCount("ranges with positions", totals.ranges_with_positions, kExpectedRangesWithPositions);
  CheckCount("positions checked", totals.positions_checked, kSlopedSurfacePositionCount);
}

// Checks every range of the scene, and returns the totals. The memory of the border search and of the pieces is built
// once and used for every range in turn.
RangeTotals CheckEveryRange(const Scene& scene) {
  const std::vector<std::size_t> range_of_position =
      FindRangeOfEachPosition(scene.test_grid.grid.tables(RunOn::kCpu), scene.ranges, scene.positions);
  EXPECT_EQ(std::ranges::count(range_of_position, scene.ranges.size()), 0) << "a position lies outside every range";
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(scene.test_grid.grid);
  std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(scene.test_grid.grid.counts().cell_blocks);
  EXPECT_TRUE(rig.has_value());
  EXPECT_TRUE(room.has_value());
  if (!rig.has_value() || !room.has_value()) {
    return {};
  }
  RangeCheckContext context = {
      .scene = scene,
      .range_of_position = range_of_position,
      .rig = *rig,
      .room = *room,
  };
  for (std::size_t range_entry = 0; range_entry < scene.ranges.size(); ++range_entry) {
    CheckRange(context, range_entry);
  }
  return context.totals;
}

// The whole module on the scene: the counts; then each range of 8 blocks with its stacked border of four layers, its
// piece, the readers from its places and the positions it holds; then the largest borders and the coverage.
TEST(StaggeredGrid, SlopedSurfaceInBox_PassesEveryCheck) {
  const Scene& scene = FindScene();
  CheckGridCounts(scene);
  CheckTotals(CheckEveryRange(scene));
}

// The cells of the scene of size level `size_level`, every k-th of them in list order so that about `count` are taken.
// clang-tidy: a size level and a count; the swappable-parameters check flags the two numbers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<CellId> FindChosenCellsOfSizeLevel(const Scene& scene, int size_level, std::size_t count) {
  std::vector<CellId> of_size_level;
  for (const CellId cell : scene.cells) {
    if (static_cast<int>(ToSizeLevel(cell)) == size_level) {
      of_size_level.push_back(cell);
    }
  }
  const std::size_t step = std::max<std::size_t>(1, of_size_level.size() / count);
  std::vector<CellId> chosen;
  for (std::size_t entry = 0; entry < of_size_level.size(); entry += step) {
    chosen.push_back(of_size_level.at(entry));
  }
  return chosen;
}

// Checks that the nearby faces of each axis at `position`, all kept, are listed for the cell at `place` and give
// weights that add up to 1.
void CheckNearbyWeightsAddUpInCell(const GridTables& tables, Vec3 position, CellPlaceIndex place) {
  for (const Axis axis : kAxes) {
    GridError error = GridError::kInternalFault;
    const NearbyFaceList list = FindNearbyFaces(tables, position, axis, CellBlockIndex{kNoIndex}, &error);
    EXPECT_EQ(error, GridError::kNone);
    const NearbyFaceWeights weights = ComputeNearbyFaceWeights(tables, list, position, ToMaskOfAll(list));
    EXPECT_EQ(list.cell.value, place.value);
    EXPECT_NEAR(ComputeKeptWeightTotal(tables, weights), 1.0, kWeightSumTolerance);
  }
}

// Checks that FindCellAt at three positions inside a cell gives that cell, and that the nearby faces of each axis, all
// kept, give weights that add up to 1.
void CheckPositionsInCellAnswerAsTheCell(const GridTables& tables, CellId cell_id) {
  const CellPlaceIndex place = FindCellPlace(tables, cell_id);
  ASSERT_NE(place.value, kNoIndex);
  const Vec3 center = ComputeCellCenter(tables, place);
  const double width = ComputeCellWidth(tables.frame, ToSizeLevel(tables, ToCellBlockIndex(place)));
  for (const double fraction : {-0.4, 0.0, 0.4}) {
    const Vec3 position = {
        .x = center.x + (fraction * width),
        .y = center.y + (fraction * width),
        .z = center.z + (fraction * width),
    };
    EXPECT_EQ(FindCellAt(tables, position).value, place.value) << "cell " << cell_id.value << ", " << fraction;
    CheckNearbyWeightsAddUpInCell(tables, position, place);
  }
}

// Cells of size levels 0, 1 and 2 of the scene, about 70 of each, against the geometric checker and against the
// cell they were made from: the touching cells and the face pieces are the checker's, and positions inside each cell
// give that cell and weights that add up to 1. The scene's 1,000 positions all lie in cells of size level 2.
TEST(StaggeredGrid, SlopedSurfaceInBox_CellsOfEverySizeLevelAnswerAsTheChecker) {
  const Scene& scene = FindScene();
  const GridTables tables = scene.test_grid.grid.tables(RunOn::kCpu);
  std::vector<CellId> chosen;
  for (int size_level = 0; size_level < kSlopedSurfaceSizeLevelCount; ++size_level) {
    const std::vector<CellId> of_size_level = FindChosenCellsOfSizeLevel(scene, size_level, 70);
    EXPECT_GE(of_size_level.size(), 10U) << "size level " << size_level;
    chosen.insert(chosen.end(), of_size_level.begin(), of_size_level.end());
  }
  const GeometricGrid checker(BuildSlopedSurfaceFrame(), scene.cells);
  LargestCounts largest_counts;
  const ActiveCellReaderMistakes mistakes = FindActiveCellReaderMistakes(tables, checker, chosen, largest_counts);
  CheckNoMistake("FindTouchingCells: the checker's places", mistakes.touching_cells_not_the_checkers);
  CheckNoMistake("FindTouchingCells: strictly ascending", mistakes.touching_cells_out_of_order);
  CheckNoMistake("FindFacePieces: the checker's places", mistakes.face_pieces_not_the_checkers);
  CheckNoMistake("FindFacePieces: the checker's ids", mistakes.face_piece_ids_not_the_checkers);
  CheckNoMistake("EveryCellSide_IsCoveredOnce", mistakes.sides_not_covered_once);
  CheckNoMistake("ClosedSurface_AreasCancel", mistakes.surfaces_not_closed);
  EXPECT_GT(largest_counts.touching_cells, 26);
  for (const CellId cell_id : chosen) {
    CheckPositionsInCellAnswerAsTheCell(tables, cell_id);
  }
}

}  // namespace
}  // namespace fluidity::grid
