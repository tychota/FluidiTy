#include "fluidity/grid/cell_block_range_internal.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <expected>

#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"

namespace fluidity::grid {
namespace {

TEST(ToBorderBlockCount, NoKey_IsTheCount) {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.result_count = 3;
  const std::expected<std::uint32_t, GridFailure> count = ToBorderBlockCount(status, 10);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 3U);
}

TEST(ToBorderBlockCount, CountEqualToTheBlockCount_IsTheCount) {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.result_count = 10;
  const std::expected<std::uint32_t, GridFailure> count = ToBorderBlockCount(status, 10);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 10U);
}

TEST(ToBorderBlockCount, KeyOfTheIdStep_IsBadInput) {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.failure_key = ToFailureKey(kIdStep, 2, 0);
  const std::expected<std::uint32_t, GridFailure> count = ToBorderBlockCount(status, 10);
  ASSERT_FALSE(count.has_value());
  EXPECT_EQ(count.error().error, GridError::kBadInput);
}

TEST(ToBorderBlockCount, KeyOfAnotherStep_IsInternalFault) {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.failure_key = ToFailureKey(kOverlapStep, 2, 0);
  const std::expected<std::uint32_t, GridFailure> count = ToBorderBlockCount(status, 10);
  ASSERT_FALSE(count.has_value());
  EXPECT_EQ(count.error().error, GridError::kInternalFault);
}

TEST(ToBorderBlockCount, CountAboveTheBlockCount_IsInternalFault) {
  GridBuildStatus status = ComputeResetBuildStatus();
  status.result_count = 11;
  const std::expected<std::uint32_t, GridFailure> count = ToBorderBlockCount(status, 10);
  ASSERT_FALSE(count.has_value());
  EXPECT_EQ(count.error().error, GridError::kInternalFault);
}

}  // namespace
}  // namespace fluidity::grid
