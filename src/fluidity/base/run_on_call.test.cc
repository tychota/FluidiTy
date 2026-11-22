#include "fluidity/base/run_on_call.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>

namespace fluidity {
namespace {

TEST(ComputeLaunchSize, OneItem_IsOneThreadBlock) {
  static_assert(ComputeLaunchSize(1).thread_block_count == 1);
  EXPECT_EQ(ComputeLaunchSize(1).thread_block_count, 1U);
}

TEST(ComputeLaunchSize, FullThreadBlock_IsOneThreadBlock) {
  EXPECT_EQ(ComputeLaunchSize(kThreadBlockSize).thread_block_count, 1U);
}

TEST(ComputeLaunchSize, OneItemPastAThreadBlock_IsTwoThreadBlocks) {
  EXPECT_EQ(ComputeLaunchSize(kThreadBlockSize + 1).thread_block_count, 2U);
}

TEST(ComputeLaunchSize, AnyCount_HasThreadBlocksOfTheFixedSize) {
  EXPECT_EQ(ComputeLaunchSize(1000).thread_block_size, kThreadBlockSize);
}

TEST(ComputeLaunchSize, HugeCount_StopsAtTheMaxThreadBlockCount) {
  EXPECT_EQ(ComputeLaunchSize(std::numeric_limits<std::size_t>::max()).thread_block_count, kMaxThreadBlockCount);
}

TEST(ComputeLaunchSize, NoItem_IsNoThreadBlock) {
  EXPECT_EQ(ComputeLaunchSize(0).thread_block_count, 0U);
}

}  // namespace
}  // namespace fluidity
