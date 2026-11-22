#include "fluidity/grid/grid_values.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"

namespace fluidity::grid {
namespace {

constexpr std::size_t kTwoMebibytes = std::size_t{2} * 1024 * 1024;

RunMemory BuildCpuRunMemory(std::size_t byte_count) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, byte_count);
  EXPECT_TRUE(run_memory.has_value());
  return std::move(*run_memory);
}

// A CPU grid of `block_count` blocks, each holding one cell, at the version the inputs carry.
TestGrid BuildGridOfBlocks(std::int32_t block_count, GridVersion version) {
  const std::vector<CellId> cells = BuildCellsInBlocksOfTheirOwn(block_count);
  GridInputs inputs = ToInputs(ToUnitFrame(3), cells);
  inputs.version = version;
  std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(inputs);
  EXPECT_TRUE(test_grid.has_value());
  return std::move(*test_grid);
}

// Writes 0, 1, 2, ... into the bound part, then reads it back.
template <int kPlacesPerBlock>
void CheckWriteThenReadReturnsTheSameValues(GridValues<double, kPlacesPerBlock>& values) {
  std::vector<double> written(std::size_t{values.block_count()} * kPlacesPerBlock);
  for (std::size_t index = 0; index < written.size(); ++index) {
    written[index] = 0.5 + static_cast<double>(index);
  }
  ASSERT_TRUE(WriteValues<double>(written, &values).has_value());
  std::vector<double> read(written.size(), -1.0);
  ASSERT_TRUE(ReadValues(values, std::span<double>(read)).has_value());
  EXPECT_EQ(read, written);
}

// The piece the room form takes for 1,000 blocks: `expected_place_count` doubles, rounded up to 256 bytes. The caller
// writes the count as a literal, so a wrong number of places per block in a name fails here.
template <int kPlacesPerBlock>
void CheckTakesOnePieceOfPlacesTimesSize(
    const std::expected<GridValues<double, kPlacesPerBlock>, GridFailure>& built,
    const RunMemory& run_memory,
    std::size_t expected_place_count
) {
  ASSERT_TRUE(built.has_value());
  EXPECT_EQ(run_memory.handed_out_bytes(), ComputeRunMemoryPieceBytes(expected_place_count * sizeof(double)));
  EXPECT_EQ(built->max_cell_blocks(), 1000U);
  EXPECT_EQ(built->block_count(), 0U);
  EXPECT_EQ(built->grid_version().value, 0U);
  EXPECT_EQ(built->buffer().size(), expected_place_count);
}

constexpr std::uint32_t kRoomOfEightBlocks = 8;
constexpr std::int32_t kGridOfThreeBlocks = 3;
constexpr double kUntouched = -7.0;

// Values with room for 8 blocks, bound to `grid` (3 blocks), every place of the room set to `kUntouched`.
template <int kPlacesPerBlock>
void CheckBindsToTheGridAndFillsTheRoom(
    std::expected<GridValues<double, kPlacesPerBlock>, GridFailure>& built,
    const StaggeredGrid& grid
) {
  ASSERT_TRUE(built.has_value());
  ASSERT_TRUE(ApplyGridVersion(*built, grid).has_value());
  ASSERT_EQ(built->block_count(), static_cast<std::uint32_t>(kGridOfThreeBlocks));
  ASSERT_EQ(built->max_cell_blocks(), kRoomOfEightBlocks);
  std::fill_n(built->data(), built->buffer().size(), kUntouched);
}

// Reads with a span of exactly the 3 bound blocks: the first 3 blocks come back, however large the room.
template <int kPlacesPerBlock>
void CheckReadCoversExactlyTheBoundBlocks(GridValues<double, kPlacesPerBlock>& values) {
  const std::size_t bound_place_count = std::size_t{kGridOfThreeBlocks} * kPlacesPerBlock;
  for (std::size_t index = 0; index < bound_place_count; ++index) {
    values.data()[index] = 0.5 + static_cast<double>(index);
  }
  std::vector<double> read(bound_place_count, -1.0);
  ASSERT_TRUE(ReadValues(values, std::span<double>(read)).has_value());
  for (std::size_t index = 0; index < bound_place_count; ++index) {
    EXPECT_EQ(read[index], 0.5 + static_cast<double>(index));
  }
}

// A span sized for the whole room is not the bound extent: it fails and the span keeps what it held.
template <int kPlacesPerBlock>
void CheckReadSizedForTheRoomIsBadInput(GridValues<double, kPlacesPerBlock>& values) {
  std::vector<double> read(std::size_t{kRoomOfEightBlocks} * kPlacesPerBlock, -1.0);
  const std::expected<void, GridFailure> result = ReadValues(values, std::span<double>(read));
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kBadInput);
  EXPECT_EQ(read, std::vector<double>(read.size(), -1.0));
}

// Writes the 3 bound blocks: the places past them keep what they held.
template <int kPlacesPerBlock>
void CheckWriteLeavesThePlacesPastTheBoundBlocks(GridValues<double, kPlacesPerBlock>& values) {
  const std::size_t bound_place_count = std::size_t{kGridOfThreeBlocks} * kPlacesPerBlock;
  const std::vector<double> written(bound_place_count, 2.5);
  ASSERT_TRUE(WriteValues<double>(written, &values).has_value());
  for (std::size_t index = 0; index < values.buffer().size(); ++index) {
    EXPECT_EQ(values.data()[index], index < bound_place_count ? 2.5 : kUntouched);
  }
}

// A span sized for the whole room is not the bound extent: it fails and no value changes.
template <int kPlacesPerBlock>
void CheckWriteSizedForTheRoomIsBadInput(GridValues<double, kPlacesPerBlock>& values) {
  const std::vector<double> written(std::size_t{kRoomOfEightBlocks} * kPlacesPerBlock, 2.5);
  const std::expected<void, GridFailure> result = WriteValues<double>(written, &values);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kBadInput);
  for (std::size_t index = 0; index < values.buffer().size(); ++index) {
    EXPECT_EQ(values.data()[index], kUntouched);
  }
}

// Values with room for 8 blocks, bound to a grid of 3, set up for one of the checks above.
struct FourValuesOfRoomEight {
  TestGrid test_grid;
  RunMemory run_memory;
  std::expected<CellValues<double>, GridFailure> cell_values;
  std::expected<FaceValues<double>, GridFailure> face_values;
  std::expected<CornerValues<double>, GridFailure> corner_values;
  std::expected<CellBlockValues<double>, GridFailure> block_values;
};

FourValuesOfRoomEight BuildFourValuesOfRoomEight() {
  FourValuesOfRoomEight four = {
      .test_grid = BuildGridOfBlocks(kGridOfThreeBlocks, {.value = 1}),
      .run_memory = BuildCpuRunMemory(kTwoMebibytes),
      .cell_values = std::unexpected(GridFailure{}),
      .face_values = std::unexpected(GridFailure{}),
      .corner_values = std::unexpected(GridFailure{}),
      .block_values = std::unexpected(GridFailure{}),
  };
  four.cell_values = BuildCellValues<double>(RunOn::kCpu, kRoomOfEightBlocks, four.run_memory);
  four.face_values = BuildFaceValues<double>(RunOn::kCpu, kRoomOfEightBlocks, four.run_memory);
  four.corner_values = BuildCornerValues<double>(RunOn::kCpu, kRoomOfEightBlocks, four.run_memory);
  four.block_values = BuildCellBlockValues<double>(RunOn::kCpu, kRoomOfEightBlocks, four.run_memory);
  CheckBindsToTheGridAndFillsTheRoom(four.cell_values, four.test_grid.grid);
  CheckBindsToTheGridAndFillsTheRoom(four.face_values, four.test_grid.grid);
  CheckBindsToTheGridAndFillsTheRoom(four.corner_values, four.test_grid.grid);
  CheckBindsToTheGridAndFillsTheRoom(four.block_values, four.test_grid.grid);
  return four;
}

TEST(ReadValues, RoomAboveTheGrid_CoversExactlyTheBoundBlocks) {
  FourValuesOfRoomEight four = BuildFourValuesOfRoomEight();
  CheckReadCoversExactlyTheBoundBlocks(*four.cell_values);
  CheckReadCoversExactlyTheBoundBlocks(*four.face_values);
  CheckReadCoversExactlyTheBoundBlocks(*four.corner_values);
  CheckReadCoversExactlyTheBoundBlocks(*four.block_values);
}

TEST(ReadValues, SpanSizedForTheRoom_IsBadInput) {
  FourValuesOfRoomEight four = BuildFourValuesOfRoomEight();
  CheckReadSizedForTheRoomIsBadInput(*four.cell_values);
  CheckReadSizedForTheRoomIsBadInput(*four.face_values);
  CheckReadSizedForTheRoomIsBadInput(*four.corner_values);
  CheckReadSizedForTheRoomIsBadInput(*four.block_values);
}

TEST(WriteValues, RoomAboveTheGrid_LeavesThePlacesPastTheBoundBlocksUntouched) {
  FourValuesOfRoomEight four = BuildFourValuesOfRoomEight();
  CheckWriteLeavesThePlacesPastTheBoundBlocks(*four.cell_values);
  CheckWriteLeavesThePlacesPastTheBoundBlocks(*four.face_values);
  CheckWriteLeavesThePlacesPastTheBoundBlocks(*four.corner_values);
  CheckWriteLeavesThePlacesPastTheBoundBlocks(*four.block_values);
}

TEST(WriteValues, SpanSizedForTheRoom_IsBadInput) {
  FourValuesOfRoomEight four = BuildFourValuesOfRoomEight();
  CheckWriteSizedForTheRoomIsBadInput(*four.cell_values);
  CheckWriteSizedForTheRoomIsBadInput(*four.face_values);
  CheckWriteSizedForTheRoomIsBadInput(*four.corner_values);
  CheckWriteSizedForTheRoomIsBadInput(*four.block_values);
}

// The span starts one entry into the values, so it overlaps them without being the same range.
TEST(ReadValues, SpanOverlappingTheValues_IsBadInputWithNothingChanged) {
  FourValuesOfRoomEight four = BuildFourValuesOfRoomEight();
  CellValues<double>& values = *four.cell_values;
  const std::size_t bound_place_count = std::size_t{kGridOfThreeBlocks} * kCellPlacesPerBlock;
  for (std::size_t index = 0; index < values.buffer().size(); ++index) {
    values.data()[index] = 0.5 + static_cast<double>(index);
  }
  const std::vector<double> before(values.data(), values.data() + values.buffer().size());
  const std::expected<void, GridFailure> result =
      ReadValues(values, std::span<double>(values.data() + 1, bound_place_count));
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kBadInput);
  EXPECT_EQ(std::vector<double>(values.data(), values.data() + values.buffer().size()), before);
}

TEST(WriteValues, SpanOverlappingTheValues_IsBadInputWithNothingChanged) {
  FourValuesOfRoomEight four = BuildFourValuesOfRoomEight();
  CellValues<double>& values = *four.cell_values;
  const std::size_t bound_place_count = std::size_t{kGridOfThreeBlocks} * kCellPlacesPerBlock;
  for (std::size_t index = 0; index < values.buffer().size(); ++index) {
    values.data()[index] = 0.5 + static_cast<double>(index);
  }
  const std::vector<double> before(values.data(), values.data() + values.buffer().size());
  const std::expected<void, GridFailure> result =
      WriteValues<double>(std::span<const double>(values.data() + 1, bound_place_count), &values);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kBadInput);
  EXPECT_EQ(std::vector<double>(values.data(), values.data() + values.buffer().size()), before);
}

TEST(ApplyGridVersion, OtherGrid_FollowsTheGrid) {
  const TestGrid first_grid = BuildGridOfBlocks(2, {.value = 7});
  const TestGrid second_grid = BuildGridOfBlocks(5, {.value = 8});
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  std::expected<CellValues<double>, GridFailure> values = BuildCellValues<double>(RunOn::kCpu, 8, run_memory);
  ASSERT_TRUE(values.has_value());
  ASSERT_TRUE(ApplyGridVersion(*values, first_grid.grid).has_value());
  EXPECT_EQ(values->grid_version().value, 7U);
  EXPECT_EQ(values->block_count(), 2U);
  ASSERT_TRUE(ApplyGridVersion(*values, second_grid.grid).has_value());
  EXPECT_EQ(values->grid_version().value, 8U);
  EXPECT_EQ(values->block_count(), 5U);
}

TEST(ApplyGridVersion, MoreBlocksThanRoom_IsTooManyCellBlocks) {
  const TestGrid small_grid = BuildGridOfBlocks(3, {.value = 4});
  const TestGrid big_grid = BuildGridOfBlocks(9, {.value = 5});
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  std::expected<CellValues<double>, GridFailure> values = BuildCellValues<double>(RunOn::kCpu, 8, run_memory);
  ASSERT_TRUE(values.has_value());
  ASSERT_TRUE(ApplyGridVersion(*values, small_grid.grid).has_value());
  const std::expected<void, GridFailure> bound = ApplyGridVersion(*values, big_grid.grid);
  ASSERT_FALSE(bound.has_value());
  EXPECT_EQ(bound.error().error, GridError::kTooManyCellBlocks);
  EXPECT_EQ(values->grid_version().value, 4U);
  EXPECT_EQ(values->block_count(), 3U);
}

TEST(ReadValues, NotBound_IsWrongGridVersion) {
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  const std::expected<CellValues<double>, GridFailure> values = BuildCellValues<double>(RunOn::kCpu, 8, run_memory);
  ASSERT_TRUE(values.has_value());
  std::vector<double> read;
  const std::expected<void, GridFailure> result = ReadValues(*values, std::span<double>(read));
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kWrongGridVersion);
}

TEST(ReadValues, SpanOneEntryShort_IsBadInput) {
  const TestGrid test_grid = BuildGridOfBlocks(2, {.value = 1});
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  const std::expected<CellValues<double>, GridFailure> values = BuildCellValues<double>(test_grid.grid, run_memory);
  ASSERT_TRUE(values.has_value());
  std::vector<double> read((2 * kCellPlacesPerBlock) - 1);
  const std::expected<void, GridFailure> result = ReadValues(*values, std::span<double>(read));
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kBadInput);
}

TEST(WriteValues, SpanOneEntryLong_IsBadInput) {
  const TestGrid test_grid = BuildGridOfBlocks(2, {.value = 1});
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  std::expected<CellBlockValues<double>, GridFailure> values = BuildCellBlockValues<double>(test_grid.grid, run_memory);
  ASSERT_TRUE(values.has_value());
  const std::vector<double> written(3, 1.0);
  const std::expected<void, GridFailure> result = WriteValues<double>(written, &*values);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kBadInput);
}

TEST(WriteValues, ThenReadValues_ReturnsTheSameValues) {
  const TestGrid test_grid = BuildGridOfBlocks(3, {.value = 2});
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  std::expected<CellValues<double>, GridFailure> cell_values = BuildCellValues<double>(test_grid.grid, run_memory);
  std::expected<FaceValues<double>, GridFailure> face_values = BuildFaceValues<double>(test_grid.grid, run_memory);
  std::expected<CornerValues<double>, GridFailure> corner_values =
      BuildCornerValues<double>(test_grid.grid, run_memory);
  std::expected<CellBlockValues<double>, GridFailure> block_values =
      BuildCellBlockValues<double>(test_grid.grid, run_memory);
  ASSERT_TRUE(cell_values.has_value());
  ASSERT_TRUE(face_values.has_value());
  ASSERT_TRUE(corner_values.has_value());
  ASSERT_TRUE(block_values.has_value());
  CheckWriteThenReadReturnsTheSameValues(*cell_values);
  CheckWriteThenReadReturnsTheSameValues(*face_values);
  CheckWriteThenReadReturnsTheSameValues(*corner_values);
  CheckWriteThenReadReturnsTheSameValues(*block_values);
}

TEST(WriteValues, PastTheBoundPart_LeavesValuesUntouched) {
  const TestGrid test_grid = BuildGridOfBlocks(2, {.value = 1});
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  std::expected<CellBlockValues<double>, GridFailure> values = BuildCellBlockValues<double>(RunOn::kCpu, 4, run_memory);
  ASSERT_TRUE(values.has_value());
  ASSERT_TRUE(ApplyGridVersion(*values, test_grid.grid).has_value());
  const std::vector<double> written = {1.5, 2.5};
  ASSERT_TRUE(WriteValues<double>(written, &*values).has_value());
  EXPECT_EQ(values->data()[0], 1.5);
  EXPECT_EQ(values->data()[1], 2.5);
  EXPECT_EQ(values->data()[2], 0.0);
  EXPECT_EQ(values->data()[3], 0.0);
}

TEST(CheckGridVersion, OtherVersion_IsWrongGridVersion) {
  const TestGrid test_grid = BuildGridOfBlocks(1, {.value = 3});
  EXPECT_TRUE(CheckGridVersion(test_grid.grid, {.value = 3}).has_value());
  const std::expected<void, GridFailure> result = CheckGridVersion(test_grid.grid, {.value = 2});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, GridError::kWrongGridVersion);
}

TEST(BuildCellValues, ZeroBlocks_IsBadInput) {
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  const std::expected<CellValues<double>, GridFailure> values = BuildCellValues<double>(RunOn::kCpu, 0, run_memory);
  ASSERT_FALSE(values.has_value());
  EXPECT_EQ(values.error().error, GridError::kBadInput);
  EXPECT_EQ(run_memory.handed_out_bytes(), 0U);
}

TEST(BuildCellValues, AboveTheBlockLimit_IsBadInput) {
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  const std::expected<CellValues<double>, GridFailure> values =
      BuildCellValues<double>(RunOn::kCpu, kMaxCellBlocksLimit + 1, run_memory);
  ASSERT_FALSE(values.has_value());
  EXPECT_EQ(values.error().error, GridError::kBadInput);
}

TEST(BuildCellValues, GridWithoutBlocks_HasRoomForOne) {
  const std::vector<CellId> no_cells;
  const std::expected<TestGrid, GridFailure> test_grid = BuildTestGrid(ToInputs(ToUnitFrame(3), no_cells));
  ASSERT_TRUE(test_grid.has_value());
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  const std::expected<CellValues<double>, GridFailure> values = BuildCellValues<double>(test_grid->grid, run_memory);
  ASSERT_TRUE(values.has_value());
  EXPECT_EQ(values->max_cell_blocks(), 1U);
  EXPECT_EQ(values->block_count(), 0U);
  EXPECT_EQ(values->grid_version().value, 1U);
}

TEST(GridValues, MovedFrom_IsBoundToNoneWithNoRoom) {
  const TestGrid test_grid = BuildGridOfBlocks(2, {.value = 6});
  RunMemory run_memory = BuildCpuRunMemory(kTwoMebibytes);
  std::expected<CellValues<double>, GridFailure> built = BuildCellValues<double>(test_grid.grid, run_memory);
  ASSERT_TRUE(built.has_value());
  const CellValues<double> moved_to = std::move(*built);
  EXPECT_EQ(moved_to.grid_version().value, 6U);
  EXPECT_EQ(moved_to.block_count(), 2U);
  EXPECT_EQ(built->grid_version().value, 0U);
  EXPECT_EQ(built->block_count(), 0U);
  EXPECT_EQ(built->max_cell_blocks(), 0U);
}

TEST(BuildFaceValues, IntoRunMemory_TakesOnePieceOfPlacesTimesSize) {
  RunMemory cell_memory = BuildCpuRunMemory(kTwoMebibytes);
  RunMemory face_memory = BuildCpuRunMemory(kTwoMebibytes);
  RunMemory corner_memory = BuildCpuRunMemory(kTwoMebibytes);
  RunMemory block_memory = BuildCpuRunMemory(kTwoMebibytes);
  const std::expected<CellValues<double>, GridFailure> cell_values =
      BuildCellValues<double>(RunOn::kCpu, 1000, cell_memory);
  const std::expected<FaceValues<double>, GridFailure> face_values =
      BuildFaceValues<double>(RunOn::kCpu, 1000, face_memory);
  const std::expected<CornerValues<double>, GridFailure> corner_values =
      BuildCornerValues<double>(RunOn::kCpu, 1000, corner_memory);
  const std::expected<CellBlockValues<double>, GridFailure> block_values =
      BuildCellBlockValues<double>(RunOn::kCpu, 1000, block_memory);
  // 1,000 blocks of 64, 240, 125 and 1 places, written as numbers so that a wrong alias fails.
  CheckTakesOnePieceOfPlacesTimesSize(cell_values, cell_memory, 64'000);
  CheckTakesOnePieceOfPlacesTimesSize(face_values, face_memory, 240'000);
  CheckTakesOnePieceOfPlacesTimesSize(corner_values, corner_memory, 125'000);
  CheckTakesOnePieceOfPlacesTimesSize(block_values, block_memory, 1'000);
  EXPECT_EQ(cell_memory.handed_out_bytes(), 512'000U);
  EXPECT_EQ(face_memory.handed_out_bytes(), 1'920'000U);
  EXPECT_EQ(corner_memory.handed_out_bytes(), 1'000'192U);
  EXPECT_EQ(block_memory.handed_out_bytes(), 8'192U);
}

TEST(BuildFaceValues, ForAGrid_TakesOnePieceOfItsBlocksTimesPlacesTimesSize) {
  const TestGrid test_grid = BuildGridOfBlocks(9, {.value = 1});
  RunMemory cell_memory = BuildCpuRunMemory(kTwoMebibytes);
  RunMemory face_memory = BuildCpuRunMemory(kTwoMebibytes);
  RunMemory corner_memory = BuildCpuRunMemory(kTwoMebibytes);
  RunMemory block_memory = BuildCpuRunMemory(kTwoMebibytes);
  ASSERT_TRUE(BuildCellValues<double>(test_grid.grid, cell_memory).has_value());
  ASSERT_TRUE(BuildFaceValues<double>(test_grid.grid, face_memory).has_value());
  ASSERT_TRUE(BuildCornerValues<double>(test_grid.grid, corner_memory).has_value());
  ASSERT_TRUE(BuildCellBlockValues<double>(test_grid.grid, block_memory).has_value());
  EXPECT_EQ(
      cell_memory.handed_out_bytes(),
      ComputeRunMemoryPieceBytes(std::size_t{9} * kCellPlacesPerBlock * sizeof(double))
  );
  EXPECT_EQ(
      face_memory.handed_out_bytes(),
      ComputeRunMemoryPieceBytes(std::size_t{9} * kFacePlacesPerBlock * sizeof(double))
  );
  EXPECT_EQ(
      corner_memory.handed_out_bytes(),
      ComputeRunMemoryPieceBytes(std::size_t{9} * kCornerPlacesPerBlock * sizeof(double))
  );
  EXPECT_EQ(block_memory.handed_out_bytes(), ComputeRunMemoryPieceBytes(9 * sizeof(double)));
}

// 1,000 blocks of 64 cell places of 8 bytes are 512,000 bytes, and the run memory holds 256 bytes. Release builds run
// this too: the failure is a returned value, not an assert.
TEST(BuildCellValues, RunMemoryTooSmall_IsOutOfMemory) {
  RunMemory run_memory = BuildCpuRunMemory(256);
  const std::expected<CellValues<double>, GridFailure> values = BuildCellValues<double>(RunOn::kCpu, 1000, run_memory);
  ASSERT_FALSE(values.has_value());
  EXPECT_EQ(values.error().error, GridError::kOutOfMemory);
  EXPECT_EQ(values.error().magnitude, 512000U);
  EXPECT_EQ(run_memory.handed_out_bytes(), 0U);
}

}  // namespace
}  // namespace fluidity::grid
