// The per-thread steps of the GPU stable order and of its prefix sum, written once so CPU tests can run them.
//
// Purpose: give the GPU half of base/stable_order.h every computation one GPU thread does, as functions that CPU tests
//   call for every thread block and thread in any order, so the arithmetic of the GPU functions is tested without a
//   GPU; and give the CPU half the layer sizes the GPU uses, so the order buffers are sized from the same counts.
//
// Idea: a counting pass cuts the sequence into chunks of kItemsPerChunk places, one thread block of
//   kChunkThreadBlockSize threads per chunk. Each thread copies a few places of its chunk (the item and its digit) into
//   the thread block's fast memory; after the thread block waits for all of them, thread d counts the items of digit d,
//   and later places them one after another from the first place of its digit. A prefix sum adds each chunk in 64 bits:
//   each thread copies and sums kItemsPerThread values, the thread block adds the thread sums in halving rounds for the
//   chunk total, or in doubling rounds for the running sums (papers call these a reduction and a scan), and each thread
//   writes the sums of its values from its running sum. Between two rounds every thread of the thread block waits for
//   the others, which the GPU function does and the CPU tests reproduce by running a whole round before the next.
//   Integers only, so any order of the threads within a round gives the same bytes.
//   Example: a chunk holds items {7, 2, 9} with digits {1, 0, 1}: thread 1 counts 2 items, thread 0 counts 1; with
//   first places 0 for digit 0 and 1 for digit 1, item 2 goes to place 0, items 7 and 9 to places 1 and 2.
//
// Words:
//   chunk thread block  the thread block of kChunkThreadBlockSize threads that handles one chunk; thread d handles
//                       digit d
// Owns:    kChunkThreadBlockSize, kItemsPerThread, kMaxPrefixSumLayerCount, PassSource, ChunkItems,
//          ComputeChunkItemCount, CopyChunkItem, ComputeChunkDigitCount, CopyChunkItemsOfDigit, ThreadValues,
//          PrefixSumValues, PrefixSums, ComputeThreadFirstIndex, ComputeThreadValues, ThreadBlockRound,
//          ApplyHalvingRound, ComputeRunningSumAddend, CopyThreadPrefixSums, SortedDestinations, ComputeGroupStart,
//          ComputeDestinationPassCount, PrefixSumLayers, BuildPrefixSumLayers
// Uses:    the digit rule and chunk sizes (base/stable_order_digit.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h)
// Used by: base/stable_order_gpu.cu, base/stable_order_steps_test.cc
// Read:    docs/method-paper/fluidity-method.typ <proc-stable-order>

#pragma once

#include <cstddef>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/stable_order_digit.h"

namespace fluidity {

// Threads in the thread block of one chunk; thread d handles digit d.
inline constexpr std::uint32_t kChunkThreadBlockSize = kDigitValueCount;

// Places of a chunk that one thread copies and sums.
inline constexpr std::uint32_t kItemsPerThread = kItemsPerChunk / kChunkThreadBlockSize;
static_assert(kItemsPerThread * kChunkThreadBlockSize == kItemsPerChunk);

// The most layers a prefix sum can need: each layer has at most 1/1024 of the entries of the one below, so even 2^64
// values need 7 layers.
inline constexpr int kMaxPrefixSumLayerCount = 8;

// What a counting pass reads: the keys, the starting sequence (null for the indexes in ascending order), the number
// of places and the pass, which picks the digit (base/stable_order_digit.h).
template <typename Key>
struct PassSource {
  const Key* keys = nullptr;  // Indexed by item.
  const std::uint32_t* sequence = nullptr;  // `count` items, or null for 0, 1, ..., count - 1.
  std::size_t count = 0;  // Places in the sequence.
  std::uint32_t pass = 0;  // Below 8.
};

// The fast memory of one chunk thread block: the item at each place of its chunk and the item's digit, kItemsPerChunk
// entries each, and how many places of the chunk hold an item.
struct ChunkItems {
  std::uint8_t* digits = nullptr;
  std::uint32_t* items = nullptr;
  std::uint32_t item_count = 0;  // ComputeChunkItemCount of the chunk; set once the chunk is copied.
};

// The number of places in chunk `chunk` of a sequence of `count` places: kItemsPerChunk, or fewer for the last chunk.
// chunk is below ComputeChunkCount(count).
// Example: count 1,025: chunk 0 has 1,024 places, chunk 1 has 1.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t ComputeChunkItemCount(std::size_t count, std::size_t chunk) {
  const std::size_t remaining = count - chunk * kItemsPerChunk;
  return remaining < kItemsPerChunk ? static_cast<std::uint32_t>(remaining) : kItemsPerChunk;
}

// Copies place `place` of chunk `chunk` into the thread block's fast memory: chunk_items.items[place] is the item at
// that place of the sequence, chunk_items.digits[place] its digit at source.pass. place is below the chunk's item
// count.
template <typename Key>
FLUIDITY_CPU_AND_GPU void
CopyChunkItem(PassSource<Key> source, std::size_t chunk, std::uint32_t place, ChunkItems chunk_items) {
  const std::size_t position = chunk * kItemsPerChunk + place;
  const std::uint32_t item =
      source.sequence != nullptr ? source.sequence[position] : static_cast<std::uint32_t>(position);
  chunk_items.items[place] = item;
  chunk_items.digits[place] = static_cast<std::uint8_t>(ComputeKeyDigit(source.keys[item], source.pass));
}

// Step 2 for one thread: the number of the chunk's places whose digit is `digit`.
// Example: digits {1, 0, 1}: digit 1 gives 2.
FLUIDITY_CPU_AND_GPU inline std::uint32_t ComputeChunkDigitCount(ChunkItems chunk_items, std::uint32_t digit) {
  std::uint32_t digit_count = 0;
  for (std::uint32_t place = 0; place < chunk_items.item_count; ++place) {
    if (chunk_items.digits[place] == digit) {
      ++digit_count;
    }
  }
  return digit_count;
}

// Step 4 for one thread: walks the chunk's places front to back and copies each item with digit `digit` to
// next_sequence[first_place], next_sequence[first_place + 1], and so on. Only this thread writes the places of this
// digit in this chunk.
// Example: items {7, 2, 9}, digits {1, 0, 1}, digit 1, first_place 1: next_sequence[1] = 7, next_sequence[2] = 9.
FLUIDITY_CPU_AND_GPU inline void CopyChunkItemsOfDigit(
    ChunkItems chunk_items,
    std::uint32_t digit,
    std::uint32_t* next_sequence,
    std::uint32_t first_place
) {
  std::uint32_t next_place = first_place;
  for (std::uint32_t place = 0; place < chunk_items.item_count; ++place) {
    if (chunk_items.digits[place] == digit) {
      next_sequence[next_place] = chunk_items.items[place];
      ++next_place;
    }
  }
}

// The values a prefix sum reads: `count` unsigned integers, 32 bits for the values and 64 bits for the chunk totals
// of a layer.
template <typename Value>
struct PrefixSumValues {
  const Value* values = nullptr;
  std::size_t count = 0;
};

// The kItemsPerThread values of one thread, widened to 64 bits (0 past the end), and their sum.
struct ThreadValues {
  // A C array: GPU code cannot call std::array's member functions (nvcc without --expt-relaxed-constexpr).
  std::uint64_t own_values[kItemsPerThread] = {};  // NOLINT(modernize-avoid-c-arrays): see the comment above.
  std::uint64_t sum = 0;
};

// The index of the first value that thread `thread` of the thread block of chunk `chunk` handles:
// chunk * 1,024 + thread * 4.
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputeThreadFirstIndex(std::size_t chunk, std::uint32_t thread) {
  return chunk * kItemsPerChunk + std::size_t{thread} * kItemsPerThread;
}

// The values thread_first, ..., thread_first + kItemsPerThread - 1 of `values`, 0 past values.count, and their sum.
// Exact: a sum of four values below 2^64 / 4 each cannot overflow, and the callers' values are far smaller.
// Example: values {3, 0, 2, 5, 1}, thread_first 4: own values {1, 0, 0, 0}, sum 1.
template <typename Value>
FLUIDITY_CPU_AND_GPU ThreadValues ComputeThreadValues(PrefixSumValues<Value> values, std::size_t thread_first) {
  ThreadValues thread_values;
  for (std::uint32_t offset = 0; offset < kItemsPerThread; ++offset) {
    const std::size_t index = thread_first + offset;
    thread_values.own_values[offset] = index < values.count ? static_cast<std::uint64_t>(values.values[index]) : 0;
    thread_values.sum += thread_values.own_values[offset];
  }
  return thread_values;
}

// One thread in one round of its thread block: the thread's index in the thread block and the distance of the round.
struct ThreadBlockRound {
  std::uint32_t thread = 0;  // Below kChunkThreadBlockSize.
  std::uint32_t distance = 0;  // The halving rounds use 128, 64, ..., 1; the doubling rounds 1, 2, ..., 128.
};

// One halving round of the chunk total for one thread: thread t below the distance adds sums[t + distance] to
// sums[t]. After the rounds 128, 64, ..., 1, sums[0] is the total of the thread block.
// Example: sums {1, 2, 3, 4}, distance 2: threads 0 and 1 give {4, 6, 3, 4}.
FLUIDITY_CPU_AND_GPU inline void ApplyHalvingRound(std::uint64_t* thread_sums, ThreadBlockRound round) {
  if (round.thread < round.distance) {
    thread_sums[round.thread] += thread_sums[round.thread + round.distance];
  }
}

// The value one thread adds in one doubling round of the running sums: running_sums[t - distance], or 0 for t below
// the distance. Every thread reads its addend, the thread block waits, then every thread adds it; after the rounds
// 1, 2, ..., 128, running_sums[t] is the sum of the thread sums 0 to t.
// Example: running sums {1, 2, 3}, distance 1: addends {0, 1, 2}.
FLUIDITY_CPU_AND_GPU inline std::uint64_t
ComputeRunningSumAddend(const std::uint64_t* running_sums, ThreadBlockRound round) {
  return round.thread >= round.distance ? running_sums[round.thread - round.distance] : 0;
}

// Where a prefix sum writes: `count` sums, 32 bits for the values and 64 bits for the running sums of a layer.
template <typename Sum>
struct PrefixSums {
  Sum* sums = nullptr;
  std::size_t count = 0;
};

// Writes the exclusive prefix sums of one thread's values: output.sums[thread_first + k] is first_running_sum plus the
// own values before k, for each index below output.count. For a 32-bit Sum the caller has checked that the total, and
// so every running sum, fits.
// Example: own values {3, 0, 2, 5}, first running sum 10: sums {10, 13, 13, 15}.
template <typename Sum>
FLUIDITY_CPU_AND_GPU void CopyThreadPrefixSums(
    ThreadValues thread_values,
    std::uint64_t first_running_sum,
    PrefixSums<Sum> output,
    std::size_t thread_first
) {
  std::uint64_t running_sum = first_running_sum;
  for (std::uint32_t offset = 0; offset < kItemsPerThread; ++offset) {
    const std::size_t index = thread_first + offset;
    if (index < output.count) {
      output.sums[index] = static_cast<Sum>(running_sum);
    }
    running_sum += thread_values.own_values[offset];
  }
}

// The destinations of the items in group order (ascending), as BuildGroupsByDestination reads them.
struct SortedDestinations {
  const std::uint32_t* values = nullptr;
  std::size_t count = 0;
};

// The start of group `group`: the number of sorted destinations below `group`, found by binary search.
// Example: sorted destinations {0, 0, 1, 2, 2}: group 1 starts at 2, group 3 at 5.
FLUIDITY_CPU_AND_GPU inline std::uint32_t ComputeGroupStart(SortedDestinations destinations, std::size_t group) {
  // Invariant: every entry before `below_end` is below `group`; no entry from `not_below_first` on is.
  std::size_t below_end = 0;
  std::size_t not_below_first = destinations.count;
  while (below_end < not_below_first) {
    const std::size_t middle = below_end + (not_below_first - below_end) / 2;
    if (destinations.values[middle] < group) {
      below_end = middle + 1;
    } else {
      not_below_first = middle;
    }
  }
  return static_cast<std::uint32_t>(below_end);
}

// The passes a destination needs: the bytes of destination_count - 1, at least one. Higher bytes are zero for every
// item, so their passes would move nothing. destination_count is at least 1.
// Example: 256 destinations need 1 pass; 257 need 2; 2^20 need 3.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t ComputeDestinationPassCount(std::uint32_t destination_count) {
  std::uint32_t pass_count = 1;
  for (std::uint32_t largest = destination_count - 1; (largest >> kDigitBitCount) != 0; largest >>= kDigitBitCount) {
    ++pass_count;
  }
  return pass_count;
}

// Where each layer of a prefix sum lives in the layer totals. Layer 1 holds the totals of the chunks of the values,
// layer k + 1 the totals of the chunks of layer k; the top layer holds one total. Index 0 stands for the values.
struct PrefixSumLayers {
  // C arrays: GPU code cannot call std::array's member functions (nvcc without --expt-relaxed-constexpr).
  std::size_t first_entry[kMaxPrefixSumLayerCount + 1] = {};  // NOLINT(modernize-avoid-c-arrays): see above.
  std::size_t entry_count[kMaxPrefixSumLayerCount + 1] = {};  // NOLINT(modernize-avoid-c-arrays): see above.
  int top_layer = 0;
};

// Builds the layers of a prefix sum over `value_count` values, value_count 1 or more. They use
// ComputePrefixSumLayerEntryCount(value_count) totals, the number the CPU half sizes the order buffers with.
// Example: 1,025 values: layer 1 holds 2 totals from entry 0, layer 2 holds 1 from entry 2; the top layer is 2.
FLUIDITY_CPU_AND_GPU constexpr PrefixSumLayers BuildPrefixSumLayers(std::size_t value_count) {
  PrefixSumLayers layers;
  layers.entry_count[0] = value_count;
  std::size_t next_first_entry = 0;
  int layer = 0;
  do {
    ++layer;
    layers.first_entry[layer] = next_first_entry;
    layers.entry_count[layer] = ComputeChunkCount(layers.entry_count[layer - 1]);
    next_first_entry += layers.entry_count[layer];
  } while (layers.entry_count[layer] > 1);
  layers.top_layer = layer;
  return layers;
}

}  // namespace fluidity
