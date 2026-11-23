// Test-only: the replay of a GPU launch on the CPU, thread by thread, which the replay tests of the grid share.
//
// Purpose: run the per-item function of a GPU function for every thread block and thread of its launch, in launch order
//   or in reverse, so a test compares the steps of a GPU pass with the CPU form without a GPU.
//
// Idea: a GPU function is a grid-stride loop: thread t of T threads visits the items t, t + T, and so on, and
//   stops where the function returns. The replay visits the threads one after the other, in launch order or in reverse,
//   and each thread's items in that order, so any dependence on the order of the threads shows.
//
// Owns:    LaunchOrder, kLaunchOrders, ReplayLaunch, ApplyEveryThreadOfSizedLaunch, ApplyEveryThreadOfLaunch
// Uses:    CPU-only guard (base/cpu_only.h), ComputeLaunchSize (base/run_on_call.h)
// Used by: grid/testing/grid_build_steps_cases.h, grid/testing/border_search_replay.h
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "fluidity/base/run_on_call.h"

namespace fluidity::grid {

// The order in which the replay runs the threads of one launch.
enum class LaunchOrder : std::uint8_t {
  kForward,  // Thread block 0, thread 0 first.
  kReverse,  // The last thread of the last thread block first.
};

constexpr std::array<LaunchOrder, 2> kLaunchOrders = {LaunchOrder::kForward, LaunchOrder::kReverse};

// How the replay runs the threads of every launch.
struct ReplayLaunch {
  LaunchOrder order = LaunchOrder::kForward;
  // Threads of a launch. 0 is the GPU's own: the thread blocks of ComputeLaunchSize(item count) times their size, one
  // item per thread for any list here. A smaller number gives each thread several items, as the GPU does for a list of
  // more than 65,535 x 256 cells.
  std::size_t thread_count = 0;
};

// Runs apply_item(index) for every item below item_count of one launch sized for launch_item_count items, one thread
// after the other in launch.order, each thread over its items as the grid-stride loop of a GPU function walks them:
// index = thread, thread + thread count, and so on, where thread = thread block x thread block size + thread in the
// thread block. apply_item returns false where the GPU function returns, which ends that thread's loop. A launch sized
// for 0 items does not start. The GPU functions of P9 to P11 are sized for the room and act on the items below a count
// they read from the status record, so their two counts differ.
//
// clang-tidy: two counts of items; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <typename ApplyItem>
void ApplyEveryThreadOfSizedLaunch(
    std::size_t launch_item_count,
    std::size_t item_count,
    ReplayLaunch launch,
    ApplyItem apply_item
) {
  if (launch_item_count == 0) {
    return;
  }
  const LaunchSize launch_size = ComputeLaunchSize(launch_item_count);
  const std::size_t launch_thread_count =
      static_cast<std::size_t>(launch_size.thread_block_count) * launch_size.thread_block_size;
  const std::size_t thread_count = launch.thread_count == 0 ? launch_thread_count : launch.thread_count;
  for (std::size_t visited = 0; visited < thread_count; ++visited) {
    const std::size_t thread = launch.order == LaunchOrder::kForward ? visited : thread_count - 1 - visited;
    for (std::size_t index = thread; index < item_count; index += thread_count) {
      if (!apply_item(index)) {
        break;
      }
    }
  }
}

// NOLINTEND(bugprone-easily-swappable-parameters)

// Runs apply_item over a launch sized for its items, as ApplyEveryThreadOfSizedLaunch says.
template <typename ApplyItem>
void ApplyEveryThreadOfLaunch(std::size_t item_count, ReplayLaunch launch, ApplyItem apply_item) {
  ApplyEveryThreadOfSizedLaunch(item_count, item_count, launch, apply_item);
}

}  // namespace fluidity::grid
