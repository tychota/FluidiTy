// Test-only: the GPU pass of the border search, replayed on the CPU.
//
// Purpose: run the steps of the GPU pass of FindBorderBlocks (grid/cell_block_range_gpu.h) on CPU buffers, thread by
//   thread, so a test can compare their result with the CPU form of the buffer search without a GPU.
//
// Idea: each GPU function is a thin loop over a shared per-item function of grid/border_block_steps.h. The replay runs
//   those functions for every thread block and thread of the launch, in launch order or in reverse, with the atomic
//   minimum of the failure key as a plain minimum, and the prefix sum on CPU buffers between steps 4 and 6, as the GPU
//   does. The marks, the words of the prefix sum and the status record start as 0xFF bytes, so a step that left out a
//   write shows: step 1 clears the marks, step 4 writes every word, and the prefix sum reads only what they wrote. The
//   replay can also skip the clearing of the marks, to show that the clearing matters: marks left by an earlier call
//   put every block in the list.
//
// Words:
//   replay  the GPU functions run one thread after the other on the CPU
// Owns:    MarksAtStart, BorderReplayInputs, BorderReplayResult, ComputeBorderSearchReplay
// Uses:    CPU-only guard (base/cpu_only.h), ReplayLaunch, ApplyEveryThreadOfLaunch, ApplyEveryThreadOfSizedLaunch
//          (grid/testing/thread_replay.h), BorderWidth, ComputeBorderSoFarFailureKey,
//          ApplySetPlaceReadMarks, ComputeListedMark, ApplyBorderBlockWriteAt, kSetBlocksPerMarkingLaunch,
//          ComputeMarkingShardCount, ComputeMarkingShard, SetPlaceShard
//          (grid/border_block_steps.h), ComputeResetBuildStatus (grid/grid_build_steps.h), Buffer (base/buffer.h),
//          RunOn (base/run_on.h), OrderBuffers (base/stable_order.h), GridBuildStatus (grid/grid_build_status.h),
//          CellBlockIndex, kNoIndex (grid/grid_ids.h), GridTables (grid/grid_tables.h)
// Used by: grid/border_block_steps.test.cc and tests/grid/border_search_replay.test.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/thread_replay.h"

namespace fluidity::grid {

// Whether step 1 of the replay clears the marks.
enum class MarksAtStart : std::uint8_t {
  kCleared,  // Step 1 sets every mark to 0, as the GPU pass does.
  kLeftAsFound,  // The marks keep their 0xFF bytes: what a call that forgot to clear them would see.
};

// What one replayed call is given.
struct BorderReplayInputs {
  CellBlockRange range;
  std::span<const CellBlockIndex> border_so_far;
  BorderWidth width = BorderWidth::kTouchingCells;
  std::uint32_t border_block_room = 0;  // Entries the list may hold.
  // Blocks of the set that one replayed marking launch runs places for: the shards of ComputeMarkingShard. The GPU's
  // own is kSetBlocksPerMarkingLaunch; a small one makes a small set take several launches, as a large set does.
  std::uint32_t set_blocks_per_marking_launch = kSetBlocksPerMarkingLaunch;
};

// What one replayed call leaves behind.
struct BorderReplayResult {
  std::vector<CellBlockIndex> border_blocks;  // border_block_room entries; 0xFFFFFFFF where nothing was written.
  std::vector<std::uint32_t> listed_words;  // The words step 4 wrote, one for each block.
  std::uint32_t prefix_total = 0;  // sums[block_count] after the prefix sum.
  std::uint32_t count = 0;  // The count step 6 leaves in the status record.
  std::uint64_t failure_key = kNoFailureKey;  // The failure key of the status record after step 2.
};

// Runs steps 1 to 6 of the GPU pass over `tables` for `inputs`, each launch with `launch`. A failed prefix sum fails
// the calling test.
// Requires: tables.block_count is the block count of the grid whose tables these are.
inline BorderReplayResult ComputeBorderSearchReplay(
    const GridTables& tables,
    const BorderReplayInputs& inputs,
    ReplayLaunch launch,
    MarksAtStart marks_at_start
) {
  constexpr std::uint8_t kStaleByte = 0xFF;
  const std::uint32_t block_count = tables.block_count;
  const auto border_so_far_count = static_cast<std::uint32_t>(inputs.border_so_far.size());
  std::vector<std::uint8_t> marks(block_count, kStaleByte);
  std::expected<Buffer<std::uint32_t>, BufferError> sums = BuildBuffer<std::uint32_t>(RunOn::kCpu, block_count + 1);
  std::expected<OrderBuffers, BufferError> order_buffers = BuildOrderBuffers(RunOn::kCpu, block_count + 1);
  BorderReplayResult result;
  result.border_blocks.assign(inputs.border_block_room, CellBlockIndex{kNoIndex});
  EXPECT_TRUE(sums.has_value());
  EXPECT_TRUE(order_buffers.has_value());
  if (!sums.has_value() || !order_buffers.has_value()) {
    return result;
  }
  std::fill_n(sums->data(), block_count + 1, ~std::uint32_t{0});
  GridBuildStatus status{};
  std::memset(&status, kStaleByte, sizeof(status));

  // Step 1: one thread resets the status; one thread for each block clears its mark.
  status = ComputeResetBuildStatus();
  if (marks_at_start == MarksAtStart::kCleared) {
    ApplyEveryThreadOfLaunch(block_count, launch, [&](std::size_t block) {
      marks[block] = 0;
      return true;
    });
  }
  // Step 2: one thread for each entry offers its key; the atomic minimum is a plain minimum.
  ApplyEveryThreadOfLaunch(border_so_far_count, launch, [&](std::size_t entry) {
    const std::uint64_t failure_key = ComputeBorderSoFarFailureKey(
        inputs.range,
        inputs.border_so_far.data(),
        block_count,
        static_cast<std::uint32_t>(entry)
    );
    status.failure_key = std::min(status.failure_key, failure_key);
    return true;
  });
  // Step 3: one launch for each shard of the set, one thread for each place of the shard; after a failure of step 2 a
  // thread stops at once.
  const std::uint32_t set_block_count = inputs.range.block_count + border_so_far_count;
  const std::uint32_t shard_count = ComputeMarkingShardCount(set_block_count, inputs.set_blocks_per_marking_launch);
  for (std::uint32_t shard_index = 0; shard_index < shard_count; ++shard_index) {
    const SetPlaceShard shard = ComputeMarkingShard(set_block_count, shard_index, inputs.set_blocks_per_marking_launch);
    ApplyEveryThreadOfLaunch(shard.set_place_count, launch, [&](std::size_t item) {
      if (status.failure_key != kNoFailureKey) {
        return false;
      }
      ApplySetPlaceReadMarks(
          tables,
          inputs.range,
          inputs.border_so_far.data(),
          border_so_far_count,
          shard.first_set_place + static_cast<std::uint32_t>(item),
          inputs.width,
          marks.data()
      );
      return true;
    });
  }
  // Step 4, after step 3 has finished: one thread for each block writes its listed mark, 0 after a failure.
  ApplyEveryThreadOfLaunch(block_count, launch, [&](std::size_t block) {
    const bool has_failed = status.failure_key != kNoFailureKey;
    sums->data()[block] =
        has_failed ? 0U : ComputeListedMark(marks.data(), inputs.range, static_cast<std::uint32_t>(block));
    return true;
  });
  result.listed_words.assign(sums->data(), sums->data() + block_count);
  // Step 5: the prefix sum, in place, between the launches.
  const std::expected<void, BufferError> summed = ComputeExclusivePrefixSum(*sums, block_count, *sums, *order_buffers);
  EXPECT_TRUE(summed.has_value());
  result.prefix_total = sums->data()[block_count];
  // Step 6: one thread for each block writes itself when it is listed; thread 0 writes the total.
  ApplyEveryThreadOfSizedLaunch(block_count == 0 ? 1 : block_count, block_count, launch, [&](std::size_t block) {
    ApplyBorderBlockWriteAt(
        sums->data(),
        static_cast<std::uint32_t>(block),
        inputs.border_block_room,
        result.border_blocks.data()
    );
    return true;
  });
  status.result_count = sums->data()[block_count];
  result.count = status.result_count;
  result.failure_key = status.failure_key;
  return result;
}

}  // namespace fluidity::grid
