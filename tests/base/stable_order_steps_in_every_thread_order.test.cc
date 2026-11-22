// Runs the GPU stable order, prefix sum and group starts on the CPU, with the per-thread steps that the GPU functions
// call, and compares the results with plain references. Each GPU function is run as its launch would cover it: every
// thread block and every thread, phase by phase, where a phase is the code between two waits of the thread block
// (__syncthreads); every thread of a thread block finishes a phase before any starts the next. The thread blocks and
// threads run once in launch order and once in reverse, so a result that depended on the order of the threads would
// show. The host loops (passes, layers) follow base/stable_order_gpu.cu.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "fluidity/base/stable_order_digit.h"
#include "fluidity/base/stable_order_steps.h"
#include "fluidity/base/testing/base_support.h"

namespace fluidity {
namespace {

// The passes of a 64-bit key, one per byte.
constexpr std::uint32_t kKeyPassCount = sizeof(std::uint64_t);

// The order in which the simulated thread blocks and threads of one launch run.
enum class ThreadOrder : std::uint8_t {
  kLaunchOrder,  // Thread block 0, thread 0 first.
  kReverse,  // The last thread of the last thread block first.
};

// Index `visit` of `count`, in `order`.
std::size_t ToVisited(std::size_t visit, std::size_t count, ThreadOrder order) {
  return order == ThreadOrder::kLaunchOrder ? visit : count - 1 - visit;
}

// The fast memory of one thread block, as the GPU function declares it.
struct ThreadBlockMemory {
  std::array<std::uint8_t, kItemsPerChunk> digits{};
  std::array<std::uint32_t, kItemsPerChunk> items{};

  ChunkItems ToChunkItems(std::uint32_t item_count) {
    return {.digits = digits.data(), .items = items.data(), .item_count = item_count};
  }
};

// The first phase of CountChunkDigits and PlaceChunkItems for one thread block: every thread copies its places of the
// chunk.
template <typename Key>
std::uint32_t RunReadChunk(PassSource<Key> source, std::size_t chunk, ThreadBlockMemory& memory, ThreadOrder order) {
  const std::uint32_t item_count = ComputeChunkItemCount(source.count, chunk);
  for (std::size_t visit = 0; visit < kChunkThreadBlockSize; ++visit) {
    const auto thread = static_cast<std::uint32_t>(ToVisited(visit, kChunkThreadBlockSize, order));
    for (std::uint32_t place = thread; place < item_count; place += kChunkThreadBlockSize) {
      CopyChunkItem(source, chunk, place, memory.ToChunkItems(item_count));
    }
  }
  return item_count;
}

// CountChunkDigits over every thread block: chunk_counts[d * chunk_count + chunk].
template <typename Key>
void RunCountChunkDigits(PassSource<Key> source, std::vector<std::uint32_t>& chunk_counts, ThreadOrder order) {
  const std::size_t chunk_count = ComputeChunkCount(source.count);
  for (std::size_t visit = 0; visit < chunk_count; ++visit) {
    const std::size_t chunk = ToVisited(visit, chunk_count, order);
    ThreadBlockMemory memory;
    const std::uint32_t item_count = RunReadChunk(source, chunk, memory, order);
    for (std::size_t thread_visit = 0; thread_visit < kChunkThreadBlockSize; ++thread_visit) {
      const auto digit = static_cast<std::uint32_t>(ToVisited(thread_visit, kChunkThreadBlockSize, order));
      chunk_counts.at(digit * chunk_count + chunk) = ComputeChunkDigitCount(memory.ToChunkItems(item_count), digit);
    }
  }
}

// PlaceChunkItems over every thread block.
template <typename Key>
void RunPlaceChunkItems(
    PassSource<Key> source,
    const std::vector<std::uint32_t>& first_places,
    std::vector<std::uint32_t>& next_sequence,
    ThreadOrder order
) {
  const std::size_t chunk_count = ComputeChunkCount(source.count);
  for (std::size_t visit = 0; visit < chunk_count; ++visit) {
    const std::size_t chunk = ToVisited(visit, chunk_count, order);
    ThreadBlockMemory memory;
    const std::uint32_t item_count = RunReadChunk(source, chunk, memory, order);
    for (std::size_t thread_visit = 0; thread_visit < kChunkThreadBlockSize; ++thread_visit) {
      const auto digit = static_cast<std::uint32_t>(ToVisited(thread_visit, kChunkThreadBlockSize, order));
      const std::uint32_t first_place = first_places.at(digit * chunk_count + chunk);
      CopyChunkItemsOfDigit(memory.ToChunkItems(item_count), digit, next_sequence.data(), first_place);
    }
  }
}

// SumChunks over every thread block: totals[chunk] is the sum of the chunk.
template <typename Value>
void RunSumChunks(PrefixSumValues<Value> values, std::uint64_t* totals, ThreadOrder order) {
  const std::size_t chunk_count = ComputeChunkCount(values.count);
  for (std::size_t visit = 0; visit < chunk_count; ++visit) {
    const std::size_t chunk = ToVisited(visit, chunk_count, order);
    std::array<std::uint64_t, kChunkThreadBlockSize> thread_sums{};
    for (std::size_t thread_visit = 0; thread_visit < kChunkThreadBlockSize; ++thread_visit) {
      const auto thread = static_cast<std::uint32_t>(ToVisited(thread_visit, kChunkThreadBlockSize, order));
      thread_sums[thread] = ComputeThreadValues(values, ComputeThreadFirstIndex(chunk, thread)).sum;
    }
    for (std::uint32_t distance = kChunkThreadBlockSize / 2; distance > 0; distance /= 2) {
      for (std::size_t thread_visit = 0; thread_visit < kChunkThreadBlockSize; ++thread_visit) {
        const auto thread = static_cast<std::uint32_t>(ToVisited(thread_visit, kChunkThreadBlockSize, order));
        ApplyHalvingRound(thread_sums.data(), {.thread = thread, .distance = distance});
      }
    }
    totals[chunk] = thread_sums[0];
  }
}

// WriteChunkPrefixSums over every thread block. Each round runs as on the GPU: every thread reads its addend, then
// every thread adds it.
template <typename Value, typename Sum>
void RunWriteChunkPrefixSums(
    PrefixSumValues<Value> values,
    const std::uint64_t* chunk_offsets,
    Sum* sums,
    ThreadOrder order
) {
  const std::size_t chunk_count = ComputeChunkCount(values.count);
  for (std::size_t visit = 0; visit < chunk_count; ++visit) {
    const std::size_t chunk = ToVisited(visit, chunk_count, order);
    std::array<ThreadValues, kChunkThreadBlockSize> thread_values{};
    std::array<std::uint64_t, kChunkThreadBlockSize> running_sums{};
    for (std::size_t thread_visit = 0; thread_visit < kChunkThreadBlockSize; ++thread_visit) {
      const auto thread = static_cast<std::uint32_t>(ToVisited(thread_visit, kChunkThreadBlockSize, order));
      thread_values[thread] = ComputeThreadValues(values, ComputeThreadFirstIndex(chunk, thread));
      running_sums[thread] = thread_values[thread].sum;
    }
    for (std::uint32_t distance = 1; distance < kChunkThreadBlockSize; distance *= 2) {
      std::array<std::uint64_t, kChunkThreadBlockSize> addends{};
      for (std::size_t thread_visit = 0; thread_visit < kChunkThreadBlockSize; ++thread_visit) {
        const auto thread = static_cast<std::uint32_t>(ToVisited(thread_visit, kChunkThreadBlockSize, order));
        addends[thread] = ComputeRunningSumAddend(running_sums.data(), {.thread = thread, .distance = distance});
      }
      for (std::size_t thread_visit = 0; thread_visit < kChunkThreadBlockSize; ++thread_visit) {
        const auto thread = static_cast<std::uint32_t>(ToVisited(thread_visit, kChunkThreadBlockSize, order));
        running_sums[thread] += addends[thread];
      }
    }
    const std::uint64_t chunk_offset = chunk_offsets != nullptr ? chunk_offsets[chunk] : 0;
    for (std::size_t thread_visit = 0; thread_visit < kChunkThreadBlockSize; ++thread_visit) {
      const auto thread = static_cast<std::uint32_t>(ToVisited(thread_visit, kChunkThreadBlockSize, order));
      const std::uint64_t first_running_sum = chunk_offset + running_sums[thread] - thread_values[thread].sum;
      CopyThreadPrefixSums(
          thread_values[thread],
          first_running_sum,
          PrefixSums<Sum>{.sums = sums, .count = values.count},
          ComputeThreadFirstIndex(chunk, thread)
      );
    }
  }
}

// The exclusive prefix sum of `values` in place, as ComputeGpuExclusivePrefixSum runs it without the total check:
// the way up, then the way down, in layer totals of exactly ComputePrefixSumLayerEntryCount entries. Returns the
// total.
std::uint64_t RunPrefixSum(std::vector<std::uint32_t>& values, ThreadOrder order) {
  const PrefixSumLayers layers = BuildPrefixSumLayers(values.size());
  std::vector<std::uint64_t> layer_totals(ComputePrefixSumLayerEntryCount(values.size()));
  EXPECT_EQ(layers.first_entry[layers.top_layer] + 1, layer_totals.size()) << "values " << values.size();
  RunSumChunks(
      PrefixSumValues<std::uint32_t>{.values = values.data(), .count = values.size()},
      layer_totals.data(),
      order
  );
  for (int layer = 2; layer <= layers.top_layer; ++layer) {
    RunSumChunks(
        PrefixSumValues<std::uint64_t>{
            .values = layer_totals.data() + layers.first_entry[layer - 1],
            .count = layers.entry_count[layer - 1],
        },
        layer_totals.data() + layers.first_entry[layer],
        order
    );
  }
  for (int layer = layers.top_layer - 1; layer >= 1; --layer) {
    const std::uint64_t* const chunk_offsets =
        layer + 1 < layers.top_layer ? layer_totals.data() + layers.first_entry[layer + 1] : nullptr;
    std::uint64_t* const layer_entries = layer_totals.data() + layers.first_entry[layer];
    RunWriteChunkPrefixSums(
        PrefixSumValues<std::uint64_t>{.values = layer_entries, .count = layers.entry_count[layer]},
        chunk_offsets,
        layer_entries,
        order
    );
  }
  const std::uint64_t total = layer_totals[layers.first_entry[layers.top_layer]];
  const std::uint64_t* const chunk_offsets =
      layers.top_layer > 1 ? layer_totals.data() + layers.first_entry[1] : nullptr;
  RunWriteChunkPrefixSums(
      PrefixSumValues<std::uint32_t>{.values = values.data(), .count = values.size()},
      chunk_offsets,
      values.data(),
      order
  );
  return total;
}

// The order of `keys` as PlaceByDigits in base/stable_order_gpu.cu runs it: `pass_count` passes, starting from
// `previous_order` (null for the items in ascending index), alternating between two sequences so that the last pass
// writes the result.
template <typename Key>
std::vector<std::uint32_t> RunPasses(
    const std::vector<Key>& keys,
    const std::uint32_t* previous_order,
    std::uint32_t pass_count,
    ThreadOrder order
) {
  const std::size_t count = keys.size();
  const std::size_t chunk_count = ComputeChunkCount(count);
  std::vector<std::uint32_t> result(count);
  std::vector<std::uint32_t> second_sequence(count);
  std::vector<std::uint32_t> chunk_counts(std::size_t{kDigitValueCount} * chunk_count);
  const std::uint32_t* sequence = previous_order;
  for (std::uint32_t pass = 0; pass < pass_count; ++pass) {
    const bool writes_result = (pass_count - 1 - pass) % 2 == 0;
    std::vector<std::uint32_t>& next_sequence = writes_result ? result : second_sequence;
    const PassSource<Key> source = {.keys = keys.data(), .sequence = sequence, .count = count, .pass = pass};
    RunCountChunkDigits(source, chunk_counts, order);
    const std::uint64_t total = RunPrefixSum(chunk_counts, order);
    EXPECT_EQ(total, count);
    RunPlaceChunkItems(source, chunk_counts, next_sequence, order);
    sequence = next_sequence.data();
  }
  return result;
}

// The order of `keys` as BuildGpuStableOrder runs it: all eight passes, no previous order.
std::vector<std::uint32_t> RunStableOrder(const std::vector<std::uint64_t>& keys, ThreadOrder order) {
  return RunPasses(keys, nullptr, kKeyPassCount, order);
}

// The order std::stable_sort gives `starting_order`, a list of the indexes of `keys`, by key: equal keys keep their
// order in `starting_order`.
template <typename Key>
std::vector<std::uint32_t>
ComputeReferenceOrder(const std::vector<Key>& keys, std::vector<std::uint32_t> starting_order) {
  std::ranges::stable_sort(starting_order, [&keys](std::uint32_t first, std::uint32_t second) {
    return keys[first] < keys[second];
  });
  return starting_order;
}

// The order std::stable_sort gives the indexes of `keys`.
template <typename Key>
std::vector<std::uint32_t> ComputeReferenceOrder(const std::vector<Key>& keys) {
  std::vector<std::uint32_t> ascending_indexes(keys.size());
  // A loop, because the standard library of Xcode 16.4 has no std::ranges::iota and clang-tidy rejects std::iota.
  for (std::size_t index = 0; index < ascending_indexes.size(); ++index) {
    ascending_indexes[index] = static_cast<std::uint32_t>(index);
  }
  return ComputeReferenceOrder(keys, std::move(ascending_indexes));
}

// The counts compared: around one digit's 256 values and one chunk's 1,024 places, and one large count.
constexpr std::array<std::size_t, 8> kComparedCounts = {1, 2, 255, 257, 1023, 1024, 1025, 70001};

TEST(RunStableOrder, EveryThreadInLaunchOrder_EqualsStdStableSort) {
  for (const std::size_t count : kComparedCounts) {
    const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(count);
    EXPECT_EQ(RunStableOrder(keys, ThreadOrder::kLaunchOrder), ComputeReferenceOrder(keys)) << "count " << count;
  }
}

TEST(RunStableOrder, EveryThreadInReverse_EqualsStdStableSort) {
  for (const std::size_t count : kComparedCounts) {
    const std::vector<std::uint64_t> keys = BuildKeysDifferingInTopByte(count);
    EXPECT_EQ(RunStableOrder(keys, ThreadOrder::kReverse), ComputeReferenceOrder(keys)) << "count " << count;
  }
}

TEST(RunPasses, PreviousOrder_KeepsItsOrderAmongEqualKeys) {
  for (const std::size_t count : kComparedCounts) {
    const std::vector<std::uint64_t> keys = BuildKeysWithManyTies(count);
    // A previous order that is not the ascending indexes: the indexes from the last to the first.
    std::vector<std::uint32_t> previous_order(count);
    for (std::size_t place = 0; place < count; ++place) {
      previous_order[place] = static_cast<std::uint32_t>(count - 1 - place);
    }
    const std::vector<std::uint32_t> expected = ComputeReferenceOrder(keys, previous_order);
    for (const ThreadOrder order : {ThreadOrder::kLaunchOrder, ThreadOrder::kReverse}) {
      EXPECT_EQ(RunPasses(keys, previous_order.data(), kKeyPassCount, order), expected) << "count " << count;
    }
  }
}

TEST(RunPasses, DestinationPassCountBelowEight_EqualsStdStableSort) {
  // The passes of BuildGroupsByDestination: 32-bit destinations and only the passes their largest value needs, an
  // odd count (1, 3) and an even one (2), so either sequence may hold the last pass.
  for (const std::uint32_t destination_count : {200U, 300U, 70000U}) {
    const std::uint32_t pass_count = ComputeDestinationPassCount(destination_count);
    for (const std::size_t count : kComparedCounts) {
      std::vector<std::uint32_t> destinations(count);
      for (std::size_t item = 0; item < count; ++item) {
        destinations[item] = static_cast<std::uint32_t>((item * 7919) % destination_count);
      }
      for (const ThreadOrder order : {ThreadOrder::kLaunchOrder, ThreadOrder::kReverse}) {
        EXPECT_EQ(RunPasses(destinations, nullptr, pass_count, order), ComputeReferenceOrder(destinations))
            << "destination_count " << destination_count << ", count " << count;
      }
    }
  }
}

TEST(RunPrefixSum, EveryThreadInBothOrders_EqualsTheRunningSum) {
  for (const std::size_t count : {1U, 4U, 1023U, 1024U, 1025U, 1048577U}) {
    std::vector<std::uint32_t> values(count);
    for (std::size_t index = 0; index < count; ++index) {
      values[index] = static_cast<std::uint32_t>((index * 7919) % 1000);
    }
    std::vector<std::uint32_t> expected(count);
    std::exclusive_scan(values.begin(), values.end(), expected.begin(), 0U);
    for (const ThreadOrder order : {ThreadOrder::kLaunchOrder, ThreadOrder::kReverse}) {
      std::vector<std::uint32_t> sums = values;
      const std::uint64_t total = RunPrefixSum(sums, order);
      EXPECT_EQ(sums, expected) << "count " << count;
      EXPECT_EQ(total, std::accumulate(values.begin(), values.end(), std::uint64_t{0})) << "count " << count;
    }
  }
}

}  // namespace
}  // namespace fluidity
