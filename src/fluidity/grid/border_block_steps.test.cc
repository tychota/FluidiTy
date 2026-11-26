#include "fluidity/grid/border_block_steps.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/cell_block_range.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/border_search_replay.h"
#include "fluidity/grid/testing/cell_block_range_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"
#include "fluidity/grid/testing/thread_replay.h"

namespace fluidity::grid {
namespace {

// The tables of the plane-four grid, written by hand: one size level, cells 1 m wide, the active cells {3,3,0} in
// block 0 at (0,0,0), {4,3,0} in block 1 at (1,0,0) and {6,6,0} in block 2 at (1,1,0).
struct PlaneFourTables {
  std::array<CellBlockId, 3> block_ids;
  std::vector<std::uint32_t> same_size_level_blocks = std::vector<std::uint32_t>(std::size_t{3} * 27, kNoIndex);
  std::vector<std::uint32_t> bigger_block = std::vector<std::uint32_t>(3, kNoIndex);
  std::vector<std::uint32_t> smaller_blocks = std::vector<std::uint32_t>(std::size_t{3} * 8, kNoIndex);
  std::vector<CellState> cell_states = std::vector<CellState>(std::size_t{3} * 64, CellState::kAbsent);
  std::vector<FacePieceKind> face_piece_kinds =
      std::vector<FacePieceKind>(std::size_t{3} * 240, FacePieceKind::kNotAPiece);
};

// The entry of a block's 27 neighbours for an offset of one block along x, y or z: 13 is the block itself.
constexpr std::size_t kEntryOfBlockItself = 13;
constexpr std::size_t kEntryAlongX = 14;
constexpr std::size_t kEntryAlongY = 16;
constexpr std::size_t kEntryAlongXAndY = 17;

PlaneFourTables BuildPlaneFourTables() {
  PlaneFourTables owner = {
      .block_ids = {
          ToCellBlockId(ToCell(0, 3, 3, 0)),
          ToCellBlockId(ToCell(0, 4, 3, 0)),
          ToCellBlockId(ToCell(0, 6, 6, 0)),
      },
  };
  // Each block is its own neighbour, and its neighbour on the high side of each axis is the next one named.
  for (std::size_t block = 0; block < 3; ++block) {
    owner.same_size_level_blocks.at((block * 27) + kEntryOfBlockItself) = static_cast<std::uint32_t>(block);
  }
  const auto make_neighbours = [&owner](std::size_t low_block, std::size_t high_block, std::size_t entry_toward_high) {
    // The entry of the opposite offset is 26 minus the entry.
    owner.same_size_level_blocks.at((low_block * 27) + entry_toward_high) = static_cast<std::uint32_t>(high_block);
    owner.same_size_level_blocks.at((high_block * 27) + (26 - entry_toward_high)) =
        static_cast<std::uint32_t>(low_block);
  };
  make_neighbours(0, 1, kEntryAlongX);
  make_neighbours(1, 2, kEntryAlongY);
  make_neighbours(0, 2, kEntryAlongXAndY);
  // The active cells: {3,3,0} is local (3,3,0) of block 0, {4,3,0} is local (0,3,0) of block 1 and {6,6,0} is local
  // (2,2,0) of block 2. A cell place is (z * 4 + y) * 4 + x.
  owner.cell_states.at((0 * 64) + 15) = CellState::kActive;
  owner.cell_states.at((1 * 64) + 12) = CellState::kActive;
  owner.cell_states.at((2 * 64) + 10) = CellState::kActive;
  // The face pieces of {4,3,0} that matter. X places are (z * 4 + y) * 5 + x, Y places 80 + (z * 5 + y) * 4 + x.
  // Its -X side is plane 0 of block 1, between it and {3,3,0}; plane 4 of block 0 is no piece, as block 1 exists.
  const std::size_t block_1_faces = std::size_t{1} * 240;
  const std::size_t block_2_faces = std::size_t{2} * 240;
  owner.face_piece_kinds.at(block_1_faces + 15) = FacePieceKind::kBetweenCells;
  owner.face_piece_kinds.at(block_1_faces + 16) = FacePieceKind::kGridEdgeHigh;
  owner.face_piece_kinds.at(block_1_faces + 80 + 12) = FacePieceKind::kGridEdgeLow;
  // Its +Y side is plane 4 of block 1, which is no piece because block 2 exists, and plane 0 of block 2, local x 0.
  owner.face_piece_kinds.at(block_2_faces + 80 + 0) = FacePieceKind::kGridEdgeHigh;
  // The cell {3,3,0} has pieces of its own in block 0.
  owner.face_piece_kinds.at(0 * 240 + 18) = FacePieceKind::kGridEdgeLow;
  owner.face_piece_kinds.at(0 * 240 + 80 + 15) = FacePieceKind::kGridEdgeLow;
  owner.face_piece_kinds.at(0 * 240 + 80 + 19) = FacePieceKind::kGridEdgeHigh;
  return owner;
}

GridTables ToTables(const PlaneFourTables& owner) {
  return {
      .frame = ToUnitFrame(1),
      .version = {.value = 1},
      .block_count = 3,
      .block_ids = owner.block_ids.data(),
      .same_size_level_blocks = owner.same_size_level_blocks.data(),
      .bigger_block = owner.bigger_block.data(),
      .smaller_blocks = owner.smaller_blocks.data(),
      .cell_states = owner.cell_states.data(),
      .face_piece_kinds = owner.face_piece_kinds.data(),
  };
}

TEST(ComputeIsInBlockSet, RangeOrSortedList_IsFound) {
  const std::array<CellBlockIndex, 2> sorted_blocks = {CellBlockIndex{7}, CellBlockIndex{9}};
  const BlockSet set = {
      .range = {.first_block = 2, .block_count = 3},
      .sorted_blocks = sorted_blocks.data(),
      .sorted_count = 2,
  };
  for (const std::uint32_t block : {2U, 3U, 4U, 7U, 9U}) {
    EXPECT_TRUE(ComputeIsInBlockSet(set, {.value = block})) << block;
  }
  for (const std::uint32_t block : {0U, 1U, 5U, 6U, 8U, 10U}) {
    EXPECT_FALSE(ComputeIsInBlockSet(set, {.value = block})) << block;
  }
}

TEST(ComputeIsInBlockSet, EmptySet_HoldsNothing) {
  const BlockSet set = {.range = {.first_block = 0, .block_count = 0}, .sorted_blocks = nullptr, .sorted_count = 0};
  EXPECT_FALSE(ComputeIsInBlockSet(set, {.value = 0}));
}

TEST(ComputeIsBorderBlock, PlaneFourFixture_MarksBlockC) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  const BlockSet set = {.range = {.first_block = 0, .block_count = 1}, .sorted_blocks = nullptr, .sorted_count = 0};
  EXPECT_FALSE(ComputeIsBorderBlock(tables, set, {.value = 0}, BorderWidth::kTouchingCells));
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 1}, BorderWidth::kTouchingCells));
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 2}, BorderWidth::kTouchingCells));
}

TEST(ComputeIsBorderBlock, SetOfTheLastBlock_MarksTheBlocksOfTheCellsNearItsFacePiece) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  const BlockSet set = {.range = {.first_block = 2, .block_count = 1}, .sorted_blocks = nullptr, .sorted_count = 0};
  // Block 2 keeps the +Y face piece of {4,3,0}, whose cell_low is in block 1. The cell {3,3,0} of block 0 touches that
  // cell, so its particles can list the face piece: transfer reads block 0.
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 0}, BorderWidth::kTouchingCells));
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 1}, BorderWidth::kTouchingCells));
  EXPECT_FALSE(ComputeIsBorderBlock(tables, set, {.value = 2}, BorderWidth::kTouchingCells));
}

TEST(ApplyCellPlaceReadMarks, CellOfBlockA_MarksEveryBlock) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  // {3,3,0} touches {4,3,0} of block 1, whose +Y face piece is plane 0 of block 2.
  ApplyCellPlaceReadMarks(tables, {.value = 15}, BorderWidth::kTouchingCells, marks.data());
  EXPECT_EQ(marks, (std::array<std::uint8_t, 3>{1, 1, 1}));
}

TEST(ApplyCellPlaceReadMarks, PlaceOfNoCell_MarksOnlyItsBlock) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  ApplyCellPlaceReadMarks(tables, {.value = 64 + 0}, BorderWidth::kTouchingCells, marks.data());
  EXPECT_EQ(marks, (std::array<std::uint8_t, 3>{0, 1, 0}));
}

TEST(ApplyFacePlaceReadMarks, FacePieceOfBlockC_MarksTheBlocksOfTheCellsNearIt) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  // The +Y face piece of {4,3,0}: its cell_low is in block 1, and {3,3,0} of block 0 touches that cell.
  ApplyFacePlaceReadMarks(tables, {.value = (2 * 240) + 80}, BorderWidth::kTouchingCells, marks.data());
  EXPECT_EQ(marks, (std::array<std::uint8_t, 3>{1, 1, 1}));
}

TEST(ApplyFacePlaceReadMarks, PlaceOfNoFacePiece_MarksNothing) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  ApplyFacePlaceReadMarks(tables, {.value = (2 * 240) + 81}, BorderWidth::kFourCellWidths, marks.data());
  EXPECT_EQ(marks, (std::array<std::uint8_t, 3>{0, 0, 0}));
}

TEST(ComputeIsBorderBlock, SetOfTheLastBlock_HasBothOthersAtFourCellWidths) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  const std::array<CellBlockIndex, 1> sorted_blocks = {CellBlockIndex{2}};
  const BlockSet set = {
      .range = {.first_block = 0, .block_count = 0},
      .sorted_blocks = sorted_blocks.data(),
      .sorted_count = 1,
  };
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 0}, BorderWidth::kFourCellWidths));
  EXPECT_TRUE(ComputeIsBorderBlock(tables, set, {.value = 1}, BorderWidth::kFourCellWidths));
  EXPECT_FALSE(ComputeIsBorderBlock(tables, set, {.value = 2}, BorderWidth::kFourCellWidths));
}

// The marks of the places of the set, run in `places` order into zeroed marks.
std::array<std::uint8_t, 3>
ComputeMarksOfPlaces(const GridTables& tables, const std::vector<std::uint32_t>& places, BorderWidth width) {
  const std::array<CellBlockIndex, 1> border_so_far = {CellBlockIndex{2}};
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  for (const std::uint32_t set_place : places) {
    ApplySetPlaceReadMarks(
        tables,
        {.first_block = 0, .block_count = 1},
        border_so_far.data(),
        1,
        set_place,
        width,
        marks.data()
    );
  }
  return marks;
}

// The marks of the same set by the loops of the CPU form: every cell place and face place of block 0, then of block 2.
std::array<std::uint8_t, 3> ComputeMarksOfBlocksByLoops(const GridTables& tables, BorderWidth width) {
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  for (const std::uint32_t block : {0U, 2U}) {
    for (std::uint32_t place_in_block = 0; place_in_block < kCellPlacesPerBlock; ++place_in_block) {
      ApplyCellPlaceReadMarks(tables, {.value = (block * kCellPlacesPerBlock) + place_in_block}, width, marks.data());
    }
    for (std::uint32_t place_in_block = 0; place_in_block < kFacePlacesPerBlock; ++place_in_block) {
      ApplyFacePlaceReadMarks(tables, {.value = (block * kFacePlacesPerBlock) + place_in_block}, width, marks.data());
    }
  }
  return marks;
}

// The launch of the GPU's own size, threads in launch order.
constexpr ReplayLaunch kGpuLaunch = {.order = LaunchOrder::kForward, .thread_count = 0};

TEST(ApplySetPlaceReadMarks, PlacesInAnyOrder_GiveTheSameMarks) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  // The set is block 0 (the range) and block 2 (the border so far): 608 places.
  std::vector<std::uint32_t> in_order(std::size_t{2} * kPlacesPerSetBlock);
  // A loop, because the standard library of Xcode 16.4 has no std::ranges::iota and clang-tidy rejects std::iota.
  for (std::size_t place = 0; place < in_order.size(); ++place) {
    in_order[place] = static_cast<std::uint32_t>(place);
  }
  std::vector<std::uint32_t> in_reverse = in_order;
  std::ranges::reverse(in_reverse);
  std::vector<std::uint32_t> odd_before_even;
  for (const std::uint32_t parity : {1U, 0U}) {
    for (std::uint32_t place = parity; place < in_order.size(); place += 2) {
      odd_before_even.push_back(place);
    }
  }
  for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
    const std::array<std::uint8_t, 3> by_loops = ComputeMarksOfBlocksByLoops(tables, width);
    EXPECT_EQ(ComputeMarksOfPlaces(tables, in_order, width), by_loops);
    EXPECT_EQ(ComputeMarksOfPlaces(tables, in_reverse, width), by_loops);
    EXPECT_EQ(ComputeMarksOfPlaces(tables, odd_before_even, width), by_loops);
  }
}

TEST(ApplySetPlaceReadMarks, PlaceBeyondTheSet_MarksNothing) {
  const PlaneFourTables owner = BuildPlaneFourTables();
  const GridTables tables = ToTables(owner);
  std::array<std::uint8_t, 3> marks = {0, 0, 0};
  ApplySetPlaceReadMarks(
      tables,
      {.first_block = 0, .block_count = 1},
      nullptr,
      0,
      kPlacesPerSetBlock,
      BorderWidth::kTouchingCells,
      marks.data()
  );
  EXPECT_EQ(marks, (std::array<std::uint8_t, 3>{0, 0, 0}));
}

TEST(ComputeBorderSoFarFailureKey, EntryNotAboveTheOneBefore_OffersItsPosition) {
  const std::array<CellBlockIndex, 3> border_so_far = {CellBlockIndex{2}, CellBlockIndex{4}, CellBlockIndex{3}};
  const CellBlockRange range = {.first_block = 0, .block_count = 1};
  EXPECT_EQ(ComputeBorderSoFarFailureKey(range, border_so_far.data(), 5, 0), kNoFailureKey);
  EXPECT_EQ(ComputeBorderSoFarFailureKey(range, border_so_far.data(), 5, 1), kNoFailureKey);
  EXPECT_EQ(ComputeBorderSoFarFailureKey(range, border_so_far.data(), 5, 2), ToFailureKey(1, 2, 0));
  // A block of the range, and a block at the grid's block count, fail too.
  const std::array<CellBlockIndex, 2> bad_blocks = {CellBlockIndex{0}, CellBlockIndex{5}};
  EXPECT_EQ(ComputeBorderSoFarFailureKey(range, bad_blocks.data(), 5, 0), ToFailureKey(1, 0, 0));
  EXPECT_EQ(ComputeBorderSoFarFailureKey(range, bad_blocks.data(), 5, 1), ToFailureKey(1, 1, 0));
}

TEST(ComputeListedMark, MarkedBlockOutsideTheRange_IsOne) {
  const std::array<std::uint8_t, 5> marks = {1, 1, 1, 1, 0};
  const CellBlockRange range = {.first_block = 1, .block_count = 2};
  EXPECT_EQ(ComputeListedMark(marks.data(), range, 0), 1U);
  EXPECT_EQ(ComputeListedMark(marks.data(), range, 1), 0U);
  EXPECT_EQ(ComputeListedMark(marks.data(), range, 2), 0U);
  EXPECT_EQ(ComputeListedMark(marks.data(), range, 3), 1U);
  EXPECT_EQ(ComputeListedMark(marks.data(), range, 4), 0U);
}

TEST(ApplyBorderBlockWriteAt, ListedBlockPastTheRoom_IsCountedNotWritten) {
  const std::array<std::uint32_t, 5> sums = {0, 1, 1, 1, 2};
  std::array<CellBlockIndex, 2> border_blocks = {CellBlockIndex{kNoIndex}, CellBlockIndex{kNoIndex}};
  for (std::uint32_t block = 0; block < 4; ++block) {
    ApplyBorderBlockWriteAt(sums.data(), block, 1, border_blocks.data());
  }
  EXPECT_EQ(border_blocks[0].value, 0U);
  EXPECT_EQ(border_blocks[1].value, kNoIndex);
}

TEST(FindBorderBlocks, StaleWordsWithBadBorderSoFar_PrefixSumAddsZeros) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), BuildPlaneFourCells()));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  ASSERT_EQ(tables.block_count, 3U);
  // The marks start as 0xFF bytes and the sums as 0xFFFFFFFF words inside the replay; the border so far {2, 1} is
  // not ascending.
  const std::array<CellBlockIndex, 2> border_so_far = {CellBlockIndex{2}, CellBlockIndex{1}};
  const BorderReplayInputs inputs = {
      .range = {.first_block = 0, .block_count = 1},
      .border_so_far = border_so_far,
      .width = BorderWidth::kTouchingCells,
      .border_block_room = 3,
  };
  const BorderReplayResult replay = ComputeBorderSearchReplay(tables, inputs, kGpuLaunch, MarksAtStart::kCleared);
  EXPECT_EQ(replay.listed_words, (std::vector<std::uint32_t>{0, 0, 0}));
  EXPECT_EQ(replay.prefix_total, 0U);
  EXPECT_EQ(replay.count, 0U);
  EXPECT_EQ(replay.failure_key, ToFailureKey(1, 1, 0));
  EXPECT_TRUE(std::ranges::all_of(replay.border_blocks, [](CellBlockIndex block) {
    return block.value == kNoIndex;
  }));
  // The CPU form of the same call fails with kBadInput.
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  ASSERT_TRUE(WriteBuffer<CellBlockIndex>(border_so_far, rig->so_far_buffer, 0).has_value());
  const std::expected<std::uint32_t, GridFailure> count = FindBorderBlocks(
      test_grid->grid,
      {.first_block = 0, .block_count = 1},
      rig->so_far_buffer,
      2,
      BorderWidth::kTouchingCells,
      rig->search_buffers,
      rig->border_buffer
  );
  ASSERT_FALSE(count.has_value());
  EXPECT_EQ(count.error().error, GridError::kBadInput);
}

// The limit of a set, as arithmetic: 304 places for each block, and 2^32 / 304 blocks, whose last place is a 32-bit
// number, while one block more has a place that wraps.
static_assert(kPlacesPerSetBlock == 304);
static_assert(kMaxSetBlocks == 14'128'181);
static_assert(std::uint64_t{kMaxSetBlocks} * kPlacesPerSetBlock == 4'294'967'024ULL);
constexpr std::uint64_t kPlaceCountOf32Bits = 4'294'967'296ULL;
static_assert(std::uint64_t{kMaxSetBlocks} * kPlacesPerSetBlock <= kPlaceCountOf32Bits);
static_assert((std::uint64_t{kMaxSetBlocks} + 1) * kPlacesPerSetBlock > kPlaceCountOf32Bits);
// The grid's own block limit is above it, which is why the set needs its own check.
static_assert(kMaxCellBlocksLimit > kMaxSetBlocks);
static_assert(kSetBlocksPerMarkingLaunch == 2048);

TEST(ComputeIsSetWithinPlaceLimit, CountsJustBelowAtAndAboveTheLimit_FitUpToTheLimit) {
  EXPECT_TRUE(ComputeIsSetWithinPlaceLimit(0, 0));
  EXPECT_TRUE(ComputeIsSetWithinPlaceLimit(kMaxSetBlocks - 1, 0));
  EXPECT_TRUE(ComputeIsSetWithinPlaceLimit(kMaxSetBlocks, 0));
  EXPECT_FALSE(ComputeIsSetWithinPlaceLimit(kMaxSetBlocks + 1, 0));
  // The range and the border so far add up: 1 block of range and the rest as entries.
  EXPECT_TRUE(ComputeIsSetWithinPlaceLimit(1, kMaxSetBlocks - 1));
  EXPECT_FALSE(ComputeIsSetWithinPlaceLimit(1, kMaxSetBlocks));
  EXPECT_FALSE(ComputeIsSetWithinPlaceLimit(0, std::uint64_t{kMaxSetBlocks} + 1));
  EXPECT_TRUE(ComputeIsSetWithinPlaceLimit(kMaxCellBlocksLimit - 4'000'000, 0));
  EXPECT_FALSE(ComputeIsSetWithinPlaceLimit(kMaxCellBlocksLimit, 0));
}

TEST(ComputeIsSetWithinPlaceLimit, CountsThatWouldWrapIn32Bits_DoNotFit) {
  const std::uint64_t largest_count = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(ComputeIsSetWithinPlaceLimit(largest_count, largest_count));
  EXPECT_FALSE(ComputeIsSetWithinPlaceLimit(largest_count, 2));
}

TEST(ComputeMarkingChunk, ChunksOfTwoOverFiveBlocks_HoldEveryPlaceOnceInOrder) {
  constexpr std::uint32_t kBlocks = 5;
  constexpr std::uint32_t kBlocksPerChunk = 2;
  ASSERT_EQ(ComputeMarkingChunkCount(kBlocks, kBlocksPerChunk), 3U);
  std::uint32_t next_place = 0;
  for (std::uint32_t chunk_index = 0; chunk_index < 3; ++chunk_index) {
    const SetPlaceChunk chunk = ComputeMarkingChunk(kBlocks, chunk_index, kBlocksPerChunk);
    EXPECT_EQ(chunk.first_set_place, next_place);
    EXPECT_EQ(chunk.set_place_count, (chunk_index == 2 ? 1U : 2U) * kPlacesPerSetBlock);
    next_place += chunk.set_place_count;
  }
  EXPECT_EQ(next_place, kBlocks * kPlacesPerSetBlock);
  const SetPlaceChunk beyond = ComputeMarkingChunk(kBlocks, 3, kBlocksPerChunk);
  EXPECT_EQ(beyond.set_place_count, 0U);
}

TEST(ComputeMarkingChunk, SetsAroundTheChunkSize_NeedTheRightCount) {
  EXPECT_EQ(ComputeMarkingChunkCount(0, kSetBlocksPerMarkingLaunch), 0U);
  EXPECT_EQ(ComputeMarkingChunkCount(1, kSetBlocksPerMarkingLaunch), 1U);
  EXPECT_EQ(ComputeMarkingChunkCount(kSetBlocksPerMarkingLaunch, kSetBlocksPerMarkingLaunch), 1U);
  EXPECT_EQ(ComputeMarkingChunkCount(kSetBlocksPerMarkingLaunch + 1, kSetBlocksPerMarkingLaunch), 2U);
  // The largest set: its last chunk ends at its last place, which is a 32-bit number.
  const std::uint32_t chunk_count = ComputeMarkingChunkCount(kMaxSetBlocks, kSetBlocksPerMarkingLaunch);
  const SetPlaceChunk last = ComputeMarkingChunk(kMaxSetBlocks, chunk_count - 1, kSetBlocksPerMarkingLaunch);
  EXPECT_EQ(
      std::uint64_t{last.first_set_place} + last.set_place_count,
      std::uint64_t{kMaxSetBlocks} * kPlacesPerSetBlock
  );
}

// The block numbers a replay wrote: its first `count` entries.
std::vector<std::uint32_t> ToWrittenList(const BorderReplayResult& replay) {
  std::vector<std::uint32_t> numbers;
  for (std::uint32_t entry = 0; entry < replay.count && entry < replay.border_blocks.size(); ++entry) {
    numbers.push_back(replay.border_blocks[entry].value);
  }
  return numbers;
}

// The set of the plane-four grid, block 0 as the range and block 2 as the border so far, replayed with one block for
// each marking launch (two launches), then with the GPU's own chunk: the same list and count, in both widths and in
// both thread orders.
void CheckOneBlockLaunchesEqualOneLaunch(const GridTables& tables, BorderWidth width, ReplayLaunch launch) {
  const std::array<CellBlockIndex, 1> border_so_far = {CellBlockIndex{2}};
  BorderReplayInputs inputs = {
      .range = {.first_block = 0, .block_count = 1},
      .border_so_far = border_so_far,
      .width = width,
      .border_block_room = 3,
  };
  const BorderReplayResult one_launch = ComputeBorderSearchReplay(tables, inputs, launch, MarksAtStart::kCleared);
  inputs.set_blocks_per_marking_launch = 1;
  const BorderReplayResult two_launches = ComputeBorderSearchReplay(tables, inputs, launch, MarksAtStart::kCleared);
  EXPECT_GT(one_launch.count, 0U);
  EXPECT_EQ(two_launches.count, one_launch.count);
  EXPECT_EQ(two_launches.listed_words, one_launch.listed_words);
  EXPECT_EQ(ToWrittenList(two_launches), ToWrittenList(one_launch));
}

TEST(ComputeBorderSearchReplay, ChunksOfOneBlock_GiveTheListOfOneLaunch) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), BuildPlaneFourCells()));
  ASSERT_TRUE(test_grid.has_value());
  const GridTables tables = test_grid->grid.tables(RunOn::kCpu);
  for (const BorderWidth width : {BorderWidth::kTouchingCells, BorderWidth::kFourCellWidths}) {
    for (const LaunchOrder order : kLaunchOrders) {
      CheckOneBlockLaunchesEqualOneLaunch(tables, width, {.order = order, .thread_count = 0});
    }
  }
}

// A set of no block (an empty range, no border so far) runs no marking launch: the list is empty on the replay and on
// the CPU form.
TEST(FindBorderBlocks, EmptySet_FindsNoBlock) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), BuildPlaneFourCells()));
  ASSERT_TRUE(test_grid.has_value());
  const BorderReplayInputs inputs = {
      .range = {.first_block = 1, .block_count = 0},
      .border_so_far = {},
      .width = BorderWidth::kTouchingCells,
      .border_block_room = 3,
  };
  const BorderReplayResult replay =
      ComputeBorderSearchReplay(test_grid->grid.tables(RunOn::kCpu), inputs, kGpuLaunch, MarksAtStart::kCleared);
  EXPECT_EQ(replay.count, 0U);
  EXPECT_EQ(replay.failure_key, kNoFailureKey);
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  const std::vector<CellBlockIndex> cpu_layer =
      FindLayerOrFail(*rig, {.first_block = 1, .block_count = 0}, {}, BorderWidth::kFourCellWidths);
  EXPECT_TRUE(cpu_layer.empty());
}

// A border so far of kMaxSetBlocks + 1 entries is a set that no 32-bit place index can hold: both processors fail the
// call with kTooManyCellBlocks, before the entries are read. At kMaxSetBlocks - 1 entries and one block of range the
// set fits, so the call goes on to the entries, and the first one is a block of the range: kBadInput.
TEST(FindBorderBlocks, SetAboveThePlaceLimit_IsTooManyCellBlocks) {
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(1), BuildPlaneFourCells()));
  ASSERT_TRUE(test_grid.has_value());
  std::expected<BorderSearchRig, GridFailure> rig = BuildBorderSearchRig(test_grid->grid);
  ASSERT_TRUE(rig.has_value());
  std::expected<Buffer<CellBlockIndex>, BufferError> long_list =
      BuildBuffer<CellBlockIndex>(RunOn::kCpu, std::size_t{kMaxSetBlocks} + 1);
  ASSERT_TRUE(long_list.has_value());
  const std::array<CellBlockIndex, 1> block_of_the_range = {CellBlockIndex{0}};
  ASSERT_TRUE(WriteBuffer<CellBlockIndex>(block_of_the_range, *long_list, 0).has_value());
  const CellBlockRange range = {.first_block = 0, .block_count = 1};
  const std::expected<std::uint32_t, GridFailure> above = FindBorderBlocks(
      test_grid->grid,
      range,
      *long_list,
      kMaxSetBlocks,
      BorderWidth::kTouchingCells,
      rig->search_buffers,
      rig->border_buffer
  );
  ASSERT_FALSE(above.has_value());
  EXPECT_EQ(above.error().error, GridError::kTooManyCellBlocks);
  const std::expected<std::uint32_t, GridFailure> at_limit = FindBorderBlocks(
      test_grid->grid,
      range,
      *long_list,
      kMaxSetBlocks - 1,
      BorderWidth::kTouchingCells,
      rig->search_buffers,
      rig->border_buffer
  );
  ASSERT_FALSE(at_limit.has_value());
  EXPECT_EQ(at_limit.error().error, GridError::kBadInput);
}

}  // namespace
}  // namespace fluidity::grid
