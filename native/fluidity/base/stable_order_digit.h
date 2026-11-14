// The digit of a key and the chunk sizes of the stable order, written once for both processors.
//
// Purpose: give the CPU and the GPU halves of base/stable_order.h one rule for the digit an item is placed by,
//   and one set of sizes for the GPU's temporary arrays, so the two halves and the size of the order buffers cannot
//   drift apart.
//
// Idea: the stable order places items by one byte of their key at a time, lowest byte first. ComputeKeyDigit picks that
//   byte. The GPU cuts a sequence into chunks of kItemsPerChunk places, one thread block of kDigitValueCount threads
//   per chunk, and adds prefix sums chunk by chunk, layer above layer; ComputePrefixSumLayerEntryCount gives the number
//   of chunk totals all those layers hold, which the CPU half uses to size the GPU order buffers.
//   Example: key 0x0102 has digit 0x02 at pass 0, 0x01 at pass 1 and 0 at every later pass.
//
// Words:
//   key    an unsigned integer that an item is ordered by
//   digit  one byte of a key; pass 0 reads the lowest byte
//   chunk  kItemsPerChunk consecutive places of a sequence, handled by one thread block on the GPU
// Owns:    kDigitBitCount, kDigitValueCount, kItemsPerChunk, ComputeKeyDigit, ComputeChunkCount,
//          ComputePrefixSumLayerEntryCount
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h)
// Used by: base/stable_order_cpu.cc, base/stable_order_gpu.cu
// Read:    docs/method-paper/fluidity-method.typ <proc-stable-order>

#pragma once

#include <cstddef>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"

namespace fluidity {

// Bits in one digit: one byte, so a 64-bit key takes eight counting passes.
inline constexpr std::uint32_t kDigitBitCount = 8;

// The number of values a digit can take. On the GPU it is also the number of threads in the thread block of a chunk:
// thread d handles the items with digit d.
inline constexpr std::uint32_t kDigitValueCount = 1U << kDigitBitCount;

// Places in one GPU chunk. Each of the kDigitValueCount threads of its thread block reads four of them into shared
// memory.
inline constexpr std::uint32_t kItemsPerChunk = 1024;

// The digit of `key` at pass `pass`: byte `pass` of the key, counted from the lowest. `pass` is below 8.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t ComputeKeyDigit(std::uint64_t key, std::uint32_t pass) {
  return static_cast<std::uint32_t>((key >> (kDigitBitCount * pass)) & (kDigitValueCount - 1));
}

// The number of chunks that cover `item_count` places: item_count / kItemsPerChunk, rounded up. 0 for 0 places.
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputeChunkCount(std::size_t item_count) {
  return item_count / kItemsPerChunk + (item_count % kItemsPerChunk != 0 ? 1 : 0);
}

// The number of chunk totals the layers of a GPU prefix sum over `value_count` values hold. Layer 1 holds one
// total per chunk of the values, layer 2 one per chunk of layer 1, and so on up to the layer that holds a single
// total. Example: 1,025 values give 2 totals in layer 1 and 1 in layer 2, so 3. 0 values need no layer and give 0;
// 1 to 1,024 values give 1.
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputePrefixSumLayerEntryCount(std::size_t value_count) {
  std::size_t entry_count = 0;
  std::size_t layer_value_count = value_count;
  while (layer_value_count > 0) {
    const std::size_t chunk_count = ComputeChunkCount(layer_value_count);
    entry_count += chunk_count;
    layer_value_count = chunk_count > 1 ? chunk_count : 0;
  }
  return entry_count;
}

}  // namespace fluidity
