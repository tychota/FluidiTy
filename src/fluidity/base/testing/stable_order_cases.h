// Test-only: the helpers that the unit tests and the functional tests of stable_order share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    BuildCpuBufferFilledWith, ComputeReferenceOrder, ComputeOrderOnCpu
// Uses:    the headers it includes
// Used by: src/fluidity/base/stable_order.test.cc and tests/base/stable_order_against_std_sort.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/base/testing/base_support.h"

namespace fluidity {

// A CPU buffer of `size` copies of `value`.
inline Buffer<std::uint32_t> BuildCpuBufferFilledWith(std::size_t size, std::uint32_t value) {
  const std::vector<std::uint32_t> values(size, value);
  return BuildBufferHolding(RunOn::kCpu, values);
}

// The order std::stable_sort gives the indexes of `keys`, starting from `start` (index order when empty).
inline std::vector<std::uint32_t>
ComputeReferenceOrder(const std::vector<std::uint64_t>& keys, const std::vector<std::uint32_t>& start) {
  std::vector<std::uint32_t> order(keys.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = start.empty() ? static_cast<std::uint32_t>(index) : start[index];
  }
  std::ranges::stable_sort(order, [&keys](std::uint32_t first, std::uint32_t second) {
    return keys[first] < keys[second];
  });
  return order;
}

// BuildStableOrder on CPU buffers holding `keys`, with no previous order and order buffers built for keys.size()
// items. Fails the calling test when a call fails.
inline std::vector<std::uint32_t> ComputeOrderOnCpu(const std::vector<std::uint64_t>& keys) {
  const Buffer<std::uint64_t> key_buffer = BuildBufferHolding(RunOn::kCpu, keys);
  Buffer<std::uint32_t> order = BuildCpuBufferFilledWith(keys.size(), 0);
  std::expected<OrderBuffers, BufferError> buffers = BuildOrderBuffers(RunOn::kCpu, keys.size());
  EXPECT_TRUE(buffers.has_value());
  if (!buffers.has_value()) {
    return {};
  }
  const std::expected<void, BufferError> built = BuildStableOrder(key_buffer, keys.size(), nullptr, order, *buffers);
  EXPECT_TRUE(built.has_value()) << ToName(built.error());
  return ReadAll(order);
}

}  // namespace fluidity
