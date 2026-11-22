#include "fluidity/grid/grid_values.h"

#include <gtest/gtest.h>

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

// The piece the room form takes: places x 1,000 blocks x 8 bytes, rounded up to 256.
template <int kPlacesPerBlock>
void CheckTakesOnePieceOfPlacesTimesSize(
    const std::expected<GridValues<double, kPlacesPerBlock>, GridFailure>& built,
    const RunMemory& run_memory
) {
  ASSERT_TRUE(built.has_value());
  EXPECT_EQ(
      run_memory.handed_out_bytes(),
      ComputeRunMemoryPieceBytes(std::size_t{1000} * kPlacesPerBlock * sizeof(double))
  );
  EXPECT_EQ(built->max_cell_blocks(), 1000U);
  EXPECT_EQ(built->block_count(), 0U);
  EXPECT_EQ(built->grid_version().value, 0U);
  EXPECT_EQ(built->buffer().size(), std::size_t{1000} * kPlacesPerBlock);
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
  CheckTakesOnePieceOfPlacesTimesSize(cell_values, cell_memory);
  CheckTakesOnePieceOfPlacesTimesSize(face_values, face_memory);
  CheckTakesOnePieceOfPlacesTimesSize(corner_values, corner_memory);
  CheckTakesOnePieceOfPlacesTimesSize(block_values, block_memory);
  EXPECT_EQ(face_memory.handed_out_bytes(), 1'920'000U);
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
