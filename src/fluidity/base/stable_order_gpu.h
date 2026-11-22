// The GPU half of the stable order, private to base/: the declarations of what base/stable_order_gpu.cu defines.
//
// Purpose: let base/stable_order_cpu.cc reach the GPU sort, grouping and prefix sum without including a CUDA header.
//
// Idea: the CPU half checks sizes, splits the bytes of the order buffers into GpuOrderBufferAreas, and calls one
//   function here with raw GPU pointers. Each function returns a RunOnCallStatus: kDone, kSizeMismatch when a check
//   made on the GPU found the inputs out of range before anything was written (a total above 2^32 - 1, a destination
//   out of range), or a failed GPU call. They exist only in a build with GPU code, so base/stable_order_cpu.cc names
//   them only under `#if defined(FLUIDITY_HAS_GPU)`. This file uses the C++20 subset and includes no CPU-only header,
//   so base/stable_order_gpu.cu includes it too. Callers use base/stable_order.h.
//
// Owns:    GpuOrderBufferAreas, BuildGpuStableOrder, BuildGpuGroupsByDestination, ComputeGpuExclusivePrefixSum
// Uses:    RunOnCallStatus (base/run_on_call.h)
// Used by: base/stable_order_cpu.cc, and base/stable_order_gpu.cu, which defines them
// Read:    docs/method-paper/fluidity-method.typ <proc-stable-order>

#pragma once

#include <cstddef>
#include <cstdint>

#include "fluidity/base/run_on_call.h"

namespace fluidity {

// The arrays of an OrderBuffers on the GPU, each in GPU memory, for up to max_count items. The CPU half sizes them
// from base/stable_order_digit.h, so they hold:
struct GpuOrderBufferAreas {
  std::uint32_t* second_sequence = nullptr;  // max_count entries: the sequence between two passes.
  // kDigitValueCount * ComputeChunkCount(max_count) entries: the count, then the first place, of each digit in each
  // chunk, digit by digit and chunk by chunk within a digit.
  std::uint32_t* chunk_counts = nullptr;
  // ComputePrefixSumLayerEntryCount(value_count) entries for any prefix sum of at most
  // max(max_count, kDigitValueCount * ComputeChunkCount(max_count)) values: the chunk totals of every layer.
  std::uint64_t* layer_totals = nullptr;
  std::uint32_t* destination_flag = nullptr;  // One entry: set to 1 when a destination is out of range.
};

// Writes to order[0..count) the indexes of keys[0..count) by ascending key, ties in the order of previous_order[]
// (index order when previous_order is null). All eight passes run. count is 1 or more and at most the max_count
// of the order buffers; order is neither previous_order nor one of their areas.
RunOnCallStatus BuildGpuStableOrder(
    const std::uint64_t* keys,
    std::uint32_t count,
    const std::uint32_t* previous_order,
    std::uint32_t* order,
    const GpuOrderBufferAreas& areas
);

// Writes item_order[0..item_count) and first_of_group[0..destination_count] as BuildGroupsByDestination promises.
// Checks the destinations first: one that is destination_count or more gives kSizeMismatch and writes nothing.
// item_count may be 0, which writes zeros to first_of_group.
RunOnCallStatus BuildGpuGroupsByDestination(
    const std::uint32_t* destination,
    std::uint32_t item_count,
    std::uint32_t destination_count,
    std::uint32_t* first_of_group,
    std::uint32_t* item_order,
    const GpuOrderBufferAreas& areas
);

// Writes sums[0..count] as ComputeExclusivePrefixSum promises; `sums` may equal `values`. Computes the total first:
// a total above 2^32 - 1 gives kSizeMismatch and writes nothing. count may be 0, which writes sums[0] = 0.
RunOnCallStatus ComputeGpuExclusivePrefixSum(
    const std::uint32_t* values,
    std::uint32_t count,
    std::uint32_t* sums,
    const GpuOrderBufferAreas& areas
);

}  // namespace fluidity
