#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/tree_sum.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_piece_cases.h"
#include "fluidity/grid/testing/grid_piece_support.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

constexpr int kPositionCount = 100;
constexpr int kMostDraws = 20 * kPositionCount;  // Draws of positions allowed for finding kPositionCount in range.
constexpr std::array<std::uint32_t, 3> kCutBlocks = {1, 3, 7};

// Builds the piece of `range` and `border` into the buffers of `room` and fails the calling test when it cannot.
std::optional<GridPiece> BuildPieceOrFail(
    const StaggeredGrid& grid,
    CellBlockRange range,
    std::span<const CellBlockIndex> border,
    PieceBuffersRoom& room
) {
  std::expected<GridPiece, GridFailure> piece = BuildGridPiece(grid, range, border, room.buffers);
  EXPECT_TRUE(piece.has_value());
  if (!piece.has_value()) {
    return std::nullopt;
  }
  return std::move(*piece);
}

// Gives the buffers of the piece back to `room`.
void ApplyTakeBack(GridPiece&& piece, PieceBuffersRoom& room) {
  room.buffers = TakeBuffers(std::move(piece));
}

// What a check of one piece gets: the whole grid, the piece and the border layers it was built from.
using PieceCheck = void (*)(const StaggeredGrid& whole, const GridPiece& piece, const StackedBorder& stacked);

// Builds the piece of each range of `under_test` with its stacked border, into buffers that are used again for each
// piece, and runs `check_piece` on it.
void CheckEachPiece(const GridUnderTest& under_test, PieceCheck check_piece) {
  std::optional<PieceBuffersRoom> room_holder = BuildPieceBuffersRoom(under_test.grid->counts().cell_blocks);
  ASSERT_TRUE(room_holder.has_value());
  PieceBuffersRoom& room = *room_holder;
  for (const CellBlockRange range : under_test.ranges) {
    SCOPED_TRACE(testing::Message() << under_test.name << ", range from block " << range.first_block);
    const StackedBorder stacked = FindStackedBorder(*under_test.grid, range);
    std::optional<GridPiece> piece = BuildPieceOrFail(*under_test.grid, range, stacked.border, room);
    ASSERT_TRUE(piece.has_value());
    check_piece(*under_test.grid, *piece, stacked);
    ApplyTakeBack(std::move(*piece), room);
  }
}

// Checks that the range of the piece is a run of its blocks that maps to the range of the whole grid.
void CheckRangeIsRunOfPiece(const GridPiece& piece, CellBlockRange range) {
  EXPECT_EQ(piece.range_in_whole_grid().first_block, range.first_block);
  EXPECT_EQ(piece.range_in_whole_grid().block_count, range.block_count);
  EXPECT_EQ(piece.range().block_count, range.block_count);
  const std::span<const CellBlockIndex> map = piece.block_in_whole_grid();
  ASSERT_GE(map.size(), std::size_t{piece.range().first_block} + piece.range().block_count);
  for (std::uint32_t block = 0; block < piece.range().block_count; ++block) {
    EXPECT_EQ(map[piece.range().first_block + block].value, range.first_block + block);
  }
}

// Checks that the piece has the tables of BuildPieceTables for its blocks, byte for byte, and that its range is a run
// that maps to range_in_whole_grid.
void CheckPieceEqualsReference(const StaggeredGrid& whole, const GridPiece& piece, const StackedBorder& stacked) {
  const std::vector<CellBlockIndex> loaded_blocks = BuildMergedBlocks(ToRangeBlocks(stacked.range), stacked.border);
  const PieceTables expected = BuildPieceTables(whole.tables(RunOn::kCpu), loaded_blocks);
  const std::optional<std::string> difference = FindFirstTableDifference(piece, expected);
  EXPECT_FALSE(difference.has_value()) << difference.value_or("");
  CheckRangeIsRunOfPiece(piece, stacked.range);
}

// The pieces of every range of every grid have the tables of BuildPieceTables for their blocks, byte for byte, and
// their range is a run that maps to range_in_whole_grid. The buffers are used again for each piece.
TEST(BuildGridPiece, RealPieces_EqualPieceTables) {
  for (const GridUnderTest& under_test : BuildGridsUnderTest(kEqualityGridCount, RangeChoice::kEveryRange)) {
    CheckEachPiece(under_test, CheckPieceEqualsReference);
  }
}

// The tables and the piece map of `piece`, as the readers take them.
LoadedPieceTables ToLoadedPieceTables(const GridPiece& piece) {
  return {.tables = piece.grid().tables(RunOn::kCpu), .block_in_whole = piece.block_in_whole_grid()};
}

// Checks that every reader of the width, from every place of the range, answers in the piece as on the whole grid: the
// range at kTouchingCells, and the range plus the first layer at kFourCellWidths.
void CheckPieceAnswersAsWholeGrid(const StaggeredGrid& whole, const GridPiece& piece, const StackedBorder& stacked) {
  const LoadedPieceTables loaded_tables = ToLoadedPieceTables(piece);
  const std::optional<std::string> touching_difference = FindFirstReaderDifference(
      whole.tables(RunOn::kCpu),
      loaded_tables,
      ToRangeBlocks(stacked.range),
      BorderWidth::kTouchingCells
  );
  EXPECT_FALSE(touching_difference.has_value()) << touching_difference.value_or("");
  const std::vector<CellBlockIndex> set_with_first_layer =
      BuildMergedBlocks(ToRangeBlocks(stacked.range), stacked.first_layer);
  const std::optional<std::string> wide_difference = FindFirstReaderDifference(
      whole.tables(RunOn::kCpu),
      loaded_tables,
      set_with_first_layer,
      BorderWidth::kFourCellWidths
  );
  EXPECT_FALSE(wide_difference.has_value()) << wide_difference.value_or("");
}

// Every reader of the width, from every place of the range, answers in the real piece as on the whole grid.
TEST(BuildGridPiece, RealPieces_AnswerAsTheWholeGrid) {
  for (const GridUnderTest& under_test : BuildGridsUnderTest(kReaderGridCount, RangeChoice::kTestedRanges)) {
    CheckEachPiece(under_test, CheckPieceAnswersAsWholeGrid);
  }
}

// The whole grid's place of a place of the piece, both of places_per_block places per block.
std::uint32_t
ToWholePlace(std::span<const CellBlockIndex> map, std::uint32_t piece_place, std::uint32_t places_per_block) {
  return (map[piece_place / places_per_block].value * places_per_block) + (piece_place % places_per_block);
}

// Says whether two doubles have the same bits.
bool ComputeIsSameBits(double first, double second) {
  return std::bit_cast<std::uint64_t>(first) == std::bit_cast<std::uint64_t>(second);
}

// Says whether two vectors have the same bits on each axis.
bool ComputeIsSameVec3Bits(Vec3 first, Vec3 second) {
  return ComputeIsSameBits(first.x, second.x) && ComputeIsSameBits(first.y, second.y)
      && ComputeIsSameBits(first.z, second.z);
}

// The active cells of the blocks of `range` in the whole grid: their places.
std::vector<CellPlaceIndex> FindActiveCellsOfRange(const GridTables& whole, CellBlockRange range) {
  std::vector<CellPlaceIndex> cells;
  for (std::uint32_t block = range.first_block; block < range.first_block + range.block_count; ++block) {
    for (std::uint32_t place = 0; place < kCellPlacesPerBlock; ++place) {
      const std::uint32_t cell = (block * kCellPlacesPerBlock) + place;
      if (whole.cell_states[cell] == CellState::kActive) {
        cells.push_back({.value = cell});
      }
    }
  }
  return cells;
}

// The candidate faces of one position and axis, on the whole grid and in the piece.
struct ListsOfOnePosition {
  NearbyFaceList whole;
  NearbyFaceList piece;
};

// Checks that the piece's list maps to the whole grid's through the piece map: the cell, the cell width and every face
// place.
void CheckListsMapThroughPieceMap(std::span<const CellBlockIndex> map, const ListsOfOnePosition& lists) {
  for (int entry = 0; entry < lists.whole.count; ++entry) {
    EXPECT_EQ(ToWholePlace(map, lists.piece.faces[entry].value, kFacePlacesPerBlock), lists.whole.faces[entry].value);
  }
  EXPECT_EQ(ToWholePlace(map, lists.piece.cell.value, kCellPlacesPerBlock), lists.whole.cell.value);
  EXPECT_TRUE(ComputeIsSameBits(lists.piece.cell_width, lists.whole.cell_width));
}

// Finds the lists of a position and axis on the whole grid and in the piece, and checks that they map to each other.
// Gives nothing when a search failed or the lists differ in length, which fails the test.
std::optional<ListsOfOnePosition>
FindSameLists(const GridTables& whole, const LoadedPieceTables& piece, Vec3 position, Axis axis) {
  GridError whole_error = GridError::kNone;
  GridError piece_error = GridError::kNone;
  const CellBlockIndex no_start = {.value = kNoIndex};
  const ListsOfOnePosition lists = {
      .whole = FindNearbyFaces(whole, position, axis, no_start, &whole_error),
      .piece = FindNearbyFaces(piece.tables, position, axis, no_start, &piece_error),
  };
  const bool is_found = whole_error == GridError::kNone && piece_error == GridError::kNone;
  EXPECT_TRUE(is_found);
  EXPECT_EQ(lists.piece.count, lists.whole.count);
  if (!is_found || lists.piece.count != lists.whole.count) {
    return std::nullopt;
  }
  CheckListsMapThroughPieceMap(piece.block_in_whole, lists);
  return lists;
}

// Says whether two matrices have the same bits in every entry.
bool ComputeIsSameMat3Bits(const Mat3& first, const Mat3& second) {
  const std::array<double, 9> first_entries = {
      first.xx,
      first.xy,
      first.xz,
      first.yx,
      first.yy,
      first.yz,
      first.zx,
      first.zy,
      first.zz,
  };
  const std::array<double, 9> second_entries = {
      second.xx,
      second.xy,
      second.xz,
      second.yx,
      second.yy,
      second.yz,
      second.zx,
      second.zy,
      second.zz,
  };
  return std::ranges::equal(first_entries, second_entries, ComputeIsSameBits);
}

// Checks that the sums and the spreads of the two weights of a position have the same bits.
void CheckSameWeightSums(const NearbyFaceWeights& whole, const NearbyFaceWeights& piece) {
  EXPECT_EQ(std::memcmp(&piece.kept, &whole.kept, sizeof(NearbyFaceMask)), 0);
  EXPECT_TRUE(ComputeIsSameBits(piece.weight_sum, whole.weight_sum));
  EXPECT_TRUE(ComputeIsSameVec3Bits(piece.mean_offset, whole.mean_offset));
  EXPECT_TRUE(ComputeIsSameVec3Bits(piece.raw_weight_change_sum, whole.raw_weight_change_sum));
}

// Checks that every field of the two weights of a position has the same bits.
void CheckSameWeightFields(const NearbyFaceWeights& whole, const NearbyFaceWeights& piece) {
  CheckSameWeightSums(whole, piece);
  EXPECT_TRUE(ComputeIsSameMat3Bits(piece.weighted_spread, whole.weighted_spread));
  EXPECT_TRUE(ComputeIsSameMat3Bits(piece.inverse_weighted_spread, whole.inverse_weighted_spread));
  EXPECT_TRUE(piece.fit == whole.fit && piece.is_complete == whole.is_complete);
}

// Checks that one kept face has the same weight, offsets and weight change in the piece and on the whole grid.
void CheckSameFaceWeight(const FaceWeight& whole, const FaceWeight& piece) {
  EXPECT_TRUE(ComputeIsSameBits(piece.weight, whole.weight));
  EXPECT_TRUE(ComputeIsSameVec3Bits(piece.offset, whole.offset));
  EXPECT_TRUE(ComputeIsSameVec3Bits(piece.offset_from_mean, whole.offset_from_mean));
  EXPECT_TRUE(ComputeIsSameVec3Bits(piece.weight_change, whole.weight_change));
}

// Checks that, for one position and axis, the list and the weights of all kept faces are the whole grid's: the lists
// map through the piece map, and every field of the weights has the same bits.
void CheckSameWeights(const GridTables& whole, const LoadedPieceTables& piece, Vec3 position, Axis axis) {
  const std::optional<ListsOfOnePosition> lists = FindSameLists(whole, piece, position, axis);
  if (!lists.has_value()) {
    return;
  }
  const NearbyFaceWeights whole_weights =
      ComputeNearbyFaceWeights(whole, lists->whole, position, ToMaskOfAll(lists->whole));
  const NearbyFaceWeights piece_weights =
      ComputeNearbyFaceWeights(piece.tables, lists->piece, position, ToMaskOfAll(lists->piece));
  CheckSameWeightFields(whole_weights, piece_weights);
  for (int entry = 0; entry < lists->whole.count; ++entry) {
    CheckSameFaceWeight(
        ComputeKeptFaceWeight(whole, whole_weights, entry),
        ComputeKeptFaceWeight(piece.tables, piece_weights, entry)
    );
  }
}

// A position drawn inside the box of a cell, from `generator`.
Vec3 FindDrawnPositionInCell(const GridTables& whole, CellPlaceIndex cell, std::mt19937_64& generator) {
  const Vec3 center = ComputeCellCenter(whole, cell);
  const double width = ComputeCellWidth(whole.frame, ToSizeLevel(whole, ToCellBlockIndex(cell)));
  return {
      .x = center.x + ((ToUnitDouble(generator()) - 0.5) * width),
      .y = center.y + ((ToUnitDouble(generator()) - 0.5) * width),
      .z = center.z + ((ToUnitDouble(generator()) - 0.5) * width),
  };
}

// Says whether FindCellAt gives the position to a cell of the blocks of `range`.
bool ComputeIsInRangeCell(const GridTables& whole, Vec3 position, CellBlockRange range) {
  const CellPlaceIndex found = FindCellAt(whole, position);
  if (found.value == kNoIndex) {
    return false;
  }
  const std::uint32_t block = ToCellBlockIndex(found).value;
  return block >= range.first_block && block - range.first_block < range.block_count;
}

// Checks the weights at 100 positions that FindCellAt gives to cells of the range, on every axis. Positions are drawn
// until 100 are checked, within kMostDraws draws.
void CheckPieceGivesSameWeights(const StaggeredGrid& whole_grid, const GridPiece& piece, const StackedBorder& stacked) {
  const GridTables whole = whole_grid.tables(RunOn::kCpu);
  const std::vector<CellPlaceIndex> cells = FindActiveCellsOfRange(whole, stacked.range);
  if (cells.empty()) {
    return;
  }
  const LoadedPieceTables loaded_tables = ToLoadedPieceTables(piece);
  std::mt19937_64 generator(stacked.range.first_block + 1000U);
  int checked_positions = 0;
  for (int draw = 0; draw < kMostDraws && checked_positions < kPositionCount; ++draw) {
    const CellPlaceIndex cell = cells.at(ToUniformIndex(generator(), cells.size()));
    const Vec3 position = FindDrawnPositionInCell(whole, cell, generator);
    if (!ComputeIsInRangeCell(whole, position, stacked.range)) {
      continue;
    }
    ++checked_positions;
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      CheckSameWeights(whole, loaded_tables, position, axis);
    }
  }
  EXPECT_EQ(checked_positions, kPositionCount);
}

// 100 positions that FindCellAt gives to cells of the range, with every face kept and every axis: the lists map to the
// whole grid's through the piece map, and every weight is the same bits.
TEST(BuildGridPiece, SameWeightsAsWholeGrid) {
  for (const GridUnderTest& under_test : BuildGridsUnderTest(kReaderGridCount, RangeChoice::kTestedRanges)) {
    CheckEachPiece(under_test, CheckPieceGivesSameWeights);
  }
}

// The partial of one block of `tables`: the ordered sum, over the block's active cells, of the height of the cell's
// centre times its width.
double ComputeBlockPartial(const GridTables& tables, std::uint32_t block) {
  double partial = 0;
  const double cell_width = ComputeCellWidth(tables.frame, ToSizeLevel(tables, CellBlockIndex{block}));
  for (std::uint32_t place = 0; place < kCellPlacesPerBlock; ++place) {
    const std::uint32_t cell = (block * kCellPlacesPerBlock) + place;
    if (tables.cell_states[cell] == CellState::kActive) {
      partial += ComputeCellCenter(tables, {.value = cell}).z * cell_width;
    }
  }
  return partial;
}

// The partials of the range's blocks of `piece`, placed at range_in_whole_grid in `placed_partials`.
void ApplyPlacedPartials(const GridPiece& piece, std::vector<double>& placed_partials) {
  const GridTables piece_tables = piece.grid().tables(RunOn::kCpu);
  for (std::uint32_t block = 0; block < piece.range().block_count; ++block) {
    placed_partials.at(piece.range_in_whole_grid().first_block + block) =
        ComputeBlockPartial(piece_tables, piece.range().first_block + block);
  }
}

// Computes the total of the block partials of every range of one cut, each computed in its grid piece and placed at
// range_in_whole_grid.
double ComputeTotalOfPieces(const StaggeredGrid& grid, std::uint32_t cut_blocks, PieceBuffersRoom& room) {
  std::vector<double> placed_partials(grid.counts().cell_blocks, 0.0);
  for (const CellBlockRange range : BuildRanges(grid, cut_blocks).value_or(std::vector<CellBlockRange>{})) {
    const RangeWithBorder range_with_border = BuildRangeWithBorder(grid, range);
    std::optional<GridPiece> piece = BuildPieceOrFail(grid, range, range_with_border.border_blocks, room);
    if (!piece.has_value()) {
      return 0.0;
    }
    ApplyPlacedPartials(*piece, placed_partials);
    ApplyTakeBack(std::move(*piece), room);
  }
  return math::TreeSum(placed_partials.data(), placed_partials.size());
}

// Computes the total of the block partials of the whole grid.
double ComputeTotalOfWholeGrid(const GridTables& whole) {
  std::vector<double> whole_partials;
  whole_partials.reserve(whole.block_count);
  for (std::uint32_t block = 0; block < whole.block_count; ++block) {
    whole_partials.push_back(ComputeBlockPartial(whole, block));
  }
  return math::TreeSum(whole_partials.data(), whole_partials.size());
}

// The total of the block partials of the whole grid, and the total of the same partials computed in the pieces of
// each cut and placed at range_in_whole_grid, are the same bits.
TEST(BuildGridPiece, BlockSumsInPieceOrder_EqualWholeGrid) {
  for (const GridUnderTest& under_test : BuildGridsUnderTest(kReaderGridCount, RangeChoice::kTestedRanges)) {
    const StaggeredGrid& grid = *under_test.grid;
    const double whole_total = ComputeTotalOfWholeGrid(grid.tables(RunOn::kCpu));
    std::optional<PieceBuffersRoom> room = BuildPieceBuffersRoom(grid.counts().cell_blocks);
    ASSERT_TRUE(room.has_value());
    for (const std::uint32_t cut_blocks : kCutBlocks) {
      SCOPED_TRACE(testing::Message() << under_test.name << ", at most " << cut_blocks << " blocks per range");
      EXPECT_TRUE(ComputeIsSameBits(ComputeTotalOfPieces(grid, cut_blocks, *room), whole_total));
    }
  }
}

// The weight of the entry `position_in_list` of an ordered sum: a whole number from 1 to 1,024 taken from a fixed hash
// of the position (Knuth's multiplicative hash, 2,654,435,761 times the position, the product divided by 1,024, modulo
// 1,024), so the weights neither repeat in a short period nor shrink with the distance from the end of the list. A
// double holds every weight exactly.
double ComputeOrderedSumWeight(std::size_t position_in_list) {
  const std::uint64_t product = std::uint64_t{2654435761U} * position_in_list;
  return static_cast<double>(((product / 1024U) % 1024U) + 1U);
}

// The sum of `partials` made one after the other, each partial times the weight of its position in the list: unlike a
// sum of a set, it gives other bits when two partials swap places anywhere in the list, whatever the length. (A sum
// that halves the total before each addition forgets a partial 53 places back, so it cannot see a swap far from the end
// of a long list.)
double ComputeOrderedWeightedSum(std::span<const double> partials) {
  double total = 0;
  for (std::size_t position_in_list = 0; position_in_list < partials.size(); ++position_in_list) {
    total += ComputeOrderedSumWeight(position_in_list) * partials[position_in_list];
  }
  return total;
}

// Checks that the sum of the partials of all the blocks of the piece, border included, taken in the piece's order, has
// the same bits as the sum of the partials of the same blocks of the whole grid taken in the whole grid's order. The
// sum depends on the order, so a piece that lists the range first, or any other order than the whole grid's, fails.
void CheckBlockOrderGivesWholeGridSum(
    const StaggeredGrid& whole,
    const GridPiece& piece,
    const StackedBorder& stacked
) {
  const GridTables piece_tables = piece.grid().tables(RunOn::kCpu);
  const GridTables whole_tables = whole.tables(RunOn::kCpu);
  std::vector<double> piece_partials;
  piece_partials.reserve(piece_tables.block_count);
  for (std::uint32_t block = 0; block < piece_tables.block_count; ++block) {
    piece_partials.push_back(ComputeBlockPartial(piece_tables, block));
  }
  std::vector<double> whole_partials;
  whole_partials.reserve(piece_tables.block_count);
  for (const CellBlockIndex block : BuildMergedBlocks(ToRangeBlocks(stacked.range), stacked.border)) {
    whole_partials.push_back(ComputeBlockPartial(whole_tables, block.value));
  }
  EXPECT_TRUE(ComputeIsSameBits(ComputeOrderedWeightedSum(piece_partials), ComputeOrderedWeightedSum(whole_partials)));
}

// An ordered sum over the blocks of each piece, border included, has the bits of the same sum over the same blocks of
// the whole grid.
TEST(BuildGridPiece, OrderedSumOverPieceBlocks_EqualsWholeGridSum) {
  for (const GridUnderTest& under_test : BuildGridsUnderTest(kReaderGridCount, RangeChoice::kEveryRange)) {
    CheckEachPiece(under_test, CheckBlockOrderGivesWholeGridSum);
  }
}

// Says whether the border of `stacked` has a block before the range and a block after it in the whole grid's order.
bool ComputeHasBorderOnBothSides(const StackedBorder& stacked) {
  const bool is_before = std::ranges::any_of(stacked.border, [&stacked](CellBlockIndex block) {
    return block.value < stacked.range.first_block;
  });
  const bool is_after = std::ranges::any_of(stacked.border, [&stacked](CellBlockIndex block) {
    return block.value >= stacked.range.first_block + stacked.range.block_count;
  });
  return is_before && is_after;
}

// Some tested range has border blocks both before and after it, so the order test above can tell a piece that lists
// the range first from the whole grid's order.
TEST(BuildGridPiece, TestedRanges_IncludeBorderOnBothSides) {
  int ranges_with_border_on_both_sides = 0;
  for (const GridUnderTest& under_test : BuildGridsUnderTest(kReaderGridCount, RangeChoice::kEveryRange)) {
    for (const CellBlockRange range : under_test.ranges) {
      if (ComputeHasBorderOnBothSides(FindStackedBorder(*under_test.grid, range))) {
        ++ranges_with_border_on_both_sides;
      }
    }
  }
  EXPECT_GT(ranges_with_border_on_both_sides, 0);
}

// Checks that two pieces have the same grid, piece map and ranges.
void CheckSamePieces(const GridPiece& span_piece, const GridPiece& buffer_piece, CellBlockRange range) {
  EXPECT_TRUE(ComputeIsSameGrid(span_piece.grid(), buffer_piece.grid()));
  EXPECT_TRUE(
      std::ranges::equal(
          span_piece.block_in_whole_grid(),
          buffer_piece.block_in_whole_grid(),
          {},
          &CellBlockIndex::value,
          &CellBlockIndex::value
      )
  );
  EXPECT_EQ(buffer_piece.range().first_block, span_piece.range().first_block);
  EXPECT_EQ(buffer_piece.range().block_count, span_piece.range().block_count);
  EXPECT_EQ(buffer_piece.range_in_whole_grid().first_block, range.first_block);
}

// Checks that the piece built from the buffer form's border is the piece of the span form: the same tables, map and
// counts.
void CheckBufferPieceEqualsSpanPiece(const StaggeredGrid& grid, CellBlockRange range, BorderSearchRig& rig) {
  const std::vector<CellBlockIndex> border = FindLayerOrFail(rig, range, {}, BorderWidth::kFourCellWidths);
  std::optional<PieceBuffersRoom> span_room = BuildPieceBuffersRoom(grid.counts().cell_blocks);
  std::optional<PieceBuffersRoom> buffer_room = BuildPieceBuffersRoom(grid.counts().cell_blocks);
  ASSERT_TRUE(span_room.has_value());
  ASSERT_TRUE(buffer_room.has_value());
  const std::optional<GridPiece> span_piece = BuildPieceOrFail(grid, range, border, *span_room);
  ASSERT_TRUE(span_piece.has_value());
  const std::expected<GridPiece, GridFailure> buffer_piece =
      BuildGridPiece(grid, range, rig.border_buffer, static_cast<std::uint32_t>(border.size()), buffer_room->buffers);
  ASSERT_TRUE(buffer_piece.has_value());
  CheckSamePieces(*span_piece, *buffer_piece, range);
}

// The border from the buffer form of FindBorderBlocks gives the piece of the span form: the same tables, map and
// counts.
TEST(BuildGridPiece, BufferForm_EqualsSpanForm) {
  for (const GridUnderTest& under_test : BuildGridsUnderTest(kReaderGridCount, RangeChoice::kTestedRanges)) {
    std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(*under_test.grid);
    ASSERT_TRUE(rig.has_value());
    for (const CellBlockRange range : under_test.ranges) {
      SCOPED_TRACE(testing::Message() << under_test.name << ", range from block " << range.first_block);
      CheckBufferPieceEqualsSpanPiece(*under_test.grid, range, *rig);
    }
  }
}

}  // namespace
}  // namespace fluidity::grid
