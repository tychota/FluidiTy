#include "fluidity/grid/grid_piece_internal.h"

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece_steps.h"

namespace fluidity::grid {
namespace {

// The status of a good build of 3 blocks with a border of 2 entries, 1 below the range: 2 active cells in the first
// size and 1 in the second, and 5 face pieces.
GridBuildStatus ToGoodStatus() {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.result_count = 1;
  status.active_cells_per_size_level[0] = 2;
  status.active_cells_per_size_level[1] = 1;
  status.face_pieces = 5;
  return status;
}

TEST(ToPieceBuildSummary, GoodStatus_GivesCountsAndRangeStart) {
  const std::expected<PieceBuildSummary, GridFailure> summary = ToPieceBuildSummary(ToGoodStatus(), 3, 2);
  ASSERT_TRUE(summary.has_value());
  EXPECT_EQ(summary->counts.cell_blocks, 3U);
  EXPECT_EQ(summary->counts.active_cells, 3U);
  EXPECT_EQ(summary->counts.active_cells_per_size_level[0], 2U);
  EXPECT_EQ(summary->counts.active_cells_per_size_level[1], 1U);
  EXPECT_EQ(summary->counts.face_pieces, 5U);
  EXPECT_EQ(summary->range_start_in_piece, 1U);
}

TEST(ToPieceBuildSummary, KeyOfTheBorderStep_IsBadInput) {
  GridBuildStatus status = ToGoodStatus();
  status.failure_key = ToFailureKey(kPieceBorderStep, 4, 0);
  const std::expected<PieceBuildSummary, GridFailure> summary = ToPieceBuildSummary(status, 3, 2);
  ASSERT_FALSE(summary.has_value());
  EXPECT_EQ(summary.error().error, GridError::kBadInput);
}

TEST(ToPieceBuildSummary, KeyOfTheFaultStep_IsInternalFault) {
  GridBuildStatus status = ToGoodStatus();
  status.failure_key = ToFailureKey(kPieceFaultStep, 1, 7);
  const std::expected<PieceBuildSummary, GridFailure> summary = ToPieceBuildSummary(status, 3, 2);
  ASSERT_FALSE(summary.has_value());
  EXPECT_EQ(summary.error().error, GridError::kInternalFault);
}

TEST(ToPieceBuildSummary, MoreActiveCellsThanThePlacesHold_IsInternalFault) {
  GridBuildStatus status = ToGoodStatus();
  status.active_cells_per_size_level[0] = (3 * kCellPlacesPerBlock) + 1;
  const std::expected<PieceBuildSummary, GridFailure> summary = ToPieceBuildSummary(status, 3, 2);
  ASSERT_FALSE(summary.has_value());
  EXPECT_EQ(summary.error().error, GridError::kInternalFault);
}

TEST(ToPieceBuildSummary, MoreFacePiecesThanThePlacesHold_IsInternalFault) {
  GridBuildStatus status = ToGoodStatus();
  status.face_pieces = (3 * kFacePlacesPerBlock) + 1;
  const std::expected<PieceBuildSummary, GridFailure> summary = ToPieceBuildSummary(status, 3, 2);
  ASSERT_FALSE(summary.has_value());
  EXPECT_EQ(summary.error().error, GridError::kInternalFault);
}

TEST(ToPieceBuildSummary, RangeStartAboveTheBorder_IsInternalFault) {
  GridBuildStatus status = ToGoodStatus();
  status.result_count = 3;
  const std::expected<PieceBuildSummary, GridFailure> summary = ToPieceBuildSummary(status, 3, 2);
  ASSERT_FALSE(summary.has_value());
  EXPECT_EQ(summary.error().error, GridError::kInternalFault);
}

TEST(CheckPieceRangeAndRoom, FittingRangeWithBorder_Passes) {
  EXPECT_TRUE(CheckPieceRangeAndRoom(10, {.first_block = 4, .block_count = 3}, 3, 6).has_value());
}

TEST(CheckPieceRangeAndRoom, RangePastTheLastBlock_IsBadInput) {
  const std::expected<void, GridFailure> checked =
      CheckPieceRangeAndRoom(10, {.first_block = 8, .block_count = 3}, 0, 10);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, GridError::kBadInput);
}

TEST(CheckPieceRangeAndRoom, RoomBelowRangeWithBorder_IsTooManyCellBlocks) {
  const std::expected<void, GridFailure> checked =
      CheckPieceRangeAndRoom(10, {.first_block = 4, .block_count = 3}, 3, 5);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, GridError::kTooManyCellBlocks);
}

TEST(CheckPieceRangeAndRoom, RangePastTheLastBlockInSmallRoom_IsBadInput) {
  const std::expected<void, GridFailure> checked =
      CheckPieceRangeAndRoom(10, {.first_block = 9, .block_count = 3}, 3, 5);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, GridError::kBadInput);
}

}  // namespace
}  // namespace fluidity::grid
