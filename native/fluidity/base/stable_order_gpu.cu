// The GPU half of the stable order, private to base/: the functions declared in base/stable_order_gpu.h.
//
// Choice: a least-significant-digit radix sort written here, eight bits per pass, not a library sort. The toolkit's
// documentation does not promise that its sorts give a stable and identical result on every toolkit version, and an
// unstable sort followed by a fix-up is not allowed. The numbered steps match Procedure <proc-stable-order>.
//
// One counting pass: the sequence is cut into chunks of kItemsPerChunk places, one thread block of kDigitValueCount
// threads per chunk. The thread block reads the chunk's items and digits into shared memory. Thread d walks the whole
// chunk from front to back and counts the items with digit d; every thread reads the same entry at the same time, which
// the GPU is expected to serve to all of them in one read (not measured yet). The counts go to a table, digit by digit
// and chunk by chunk within a digit, so that its exclusive prefix sum gives, for each digit and chunk, the place of the
// chunk's first item with that digit: every item with a smaller digit, then the items with that digit in earlier
// chunks. A second walk by thread d places its items one after another from there. A place is therefore the first place
// of the digit in the chunk plus the number of earlier items in the chunk with that digit: a formula of the keys, never
// a racing counter. The only atomic operation is the atomicOr of the destination check, whose result (1 when any
// destination is out of range) is the same whatever the order of the threads.
//
// Prefix sums: chunk by chunk. Each chunk of kItemsPerChunk values is added in 64 bits by one thread block, and its
// total goes to the layer above; that layer is added the same way, until a layer holds one total. Going back down, each
// layer becomes the running sums of the layer below, and each chunk adds its running sum to the exclusive prefix sums
// of its own values. This was chosen over a single up-sweep and down-sweep tree over all values because each layer is a
// plain launch over independent chunks, and because the total is known after the way up, before any sum is written: a
// total above 2^32 - 1 then writes nothing. Integer additions are exact, so the order of additions inside a chunk does
// not change any result.
//
// Every call uses the default stream, so the launches run in the order they are made. Each public function waits for
// the GPU at its end, so an error of the GPU work is reported by the call that caused it. Each GPU function is a thin
// wrapper around the per-thread steps of base/stable_order_steps.h, which CPU tests run for every thread block and
// thread; launches, waits and statuses go through base/gpu_launch.cuh.
// Read: docs/method-paper/fluidity-method.typ <proc-stable-order>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/stable_order_digit.h"
#include "fluidity/base/stable_order_gpu.h"
#include "fluidity/base/stable_order_steps.h"

namespace fluidity {
namespace {

// The passes of a 64-bit key, one per byte.
constexpr std::uint32_t kKeyPassCount = sizeof(std::uint64_t);

// The launch of a GPU function with one thread block per chunk.
LaunchSize ComputeChunkLaunchSize(std::size_t chunk_count) {
  return {.thread_block_count = static_cast<unsigned int>(chunk_count), .thread_block_size = kChunkThreadBlockSize};
}

// Reads this thread block's chunk of the sequence into shared memory and sets chunk_items.item_count. Every thread of
// the thread block must call it: it ends with the thread block's wait.
template <typename Key>
__device__ void ReadChunk(PassSource<Key> source, ChunkItems& chunk_items) {
  chunk_items.item_count = ComputeChunkItemCount(source.count, blockIdx.x);
  for (std::uint32_t place = threadIdx.x; place < chunk_items.item_count; place += kChunkThreadBlockSize) {
    CopyChunkItem(source, blockIdx.x, place, chunk_items);
  }
  __syncthreads();
}

// Step 2 for one chunk: thread d writes to chunk_counts[d * chunk_count + chunk] how many items of the chunk have
// digit d.
template <typename Key>
__global__ void CountChunkDigits(PassSource<Key> source, std::size_t chunk_count, std::uint32_t* chunk_counts) {
  __shared__ std::uint8_t digits[kItemsPerChunk];
  __shared__ std::uint32_t items[kItemsPerChunk];
  ChunkItems chunk_items = {.digits = digits, .items = items, .item_count = 0};
  ReadChunk(source, chunk_items);
  const std::uint32_t digit = threadIdx.x;
  chunk_counts[digit * chunk_count + blockIdx.x] = ComputeChunkDigitCount(chunk_items, digit);
}

// Step 4 for one chunk: thread d puts each item of the chunk with digit d at the next place from
// first_places[d * chunk_count + chunk].
template <typename Key>
__global__ void PlaceChunkItems(
    PassSource<Key> source,
    std::size_t chunk_count,
    const std::uint32_t* first_places,
    std::uint32_t* next_sequence
) {
  __shared__ std::uint8_t digits[kItemsPerChunk];
  __shared__ std::uint32_t items[kItemsPerChunk];
  ChunkItems chunk_items = {.digits = digits, .items = items, .item_count = 0};
  ReadChunk(source, chunk_items);
  const std::uint32_t digit = threadIdx.x;
  CopyChunkItemsOfDigit(chunk_items, digit, next_sequence, first_places[digit * chunk_count + blockIdx.x]);
}

// The way up of a prefix sum: totals[chunk] = the sum of chunk `chunk` of the values, in 64 bits.
template <typename Value>
__global__ void SumChunks(PrefixSumValues<Value> values, std::uint64_t* totals) {
  __shared__ std::uint64_t thread_sums[kChunkThreadBlockSize];
  const std::uint32_t thread = threadIdx.x;
  thread_sums[thread] = ComputeThreadValues(values, ComputeThreadFirstIndex(blockIdx.x, thread)).sum;
  __syncthreads();
  // Integer additions are exact, so this tree gives the same total as any other order.
  for (std::uint32_t distance = kChunkThreadBlockSize / 2; distance > 0; distance /= 2) {
    ApplyHalvingRound(thread_sums, {.thread = thread, .distance = distance});
    __syncthreads();
  }
  if (thread == 0) {
    totals[blockIdx.x] = thread_sums[0];
  }
}

// The way down of a prefix sum, for one chunk: sums[i] = chunk_offsets[chunk] + the values of the chunk before i.
// chunk_offsets null means 0 (a single chunk). Each thread reads its values before the first wait and writes after
// the last, and a thread block touches only its own chunk, so `sums` may be `values`.
template <typename Value, typename Sum>
__global__ void WriteChunkPrefixSums(PrefixSumValues<Value> values, const std::uint64_t* chunk_offsets, Sum* sums) {
  __shared__ std::uint64_t running_sums[kChunkThreadBlockSize];
  const std::uint32_t thread = threadIdx.x;
  const std::size_t thread_first = ComputeThreadFirstIndex(blockIdx.x, thread);
  const ThreadValues thread_values = ComputeThreadValues(values, thread_first);
  running_sums[thread] = thread_values.sum;
  __syncthreads();
  // After these rounds running_sums[t] is the sum of the thread sums 0 to t. Each round reads, waits, then writes,
  // so no thread reads an entry another thread is changing.
  for (std::uint32_t distance = 1; distance < kChunkThreadBlockSize; distance *= 2) {
    const std::uint64_t addend = ComputeRunningSumAddend(running_sums, {.thread = thread, .distance = distance});
    __syncthreads();
    running_sums[thread] += addend;
    __syncthreads();
  }
  const std::uint64_t chunk_offset = chunk_offsets != nullptr ? chunk_offsets[blockIdx.x] : 0;
  const std::uint64_t first_running_sum = chunk_offset + running_sums[thread] - thread_values.sum;
  CopyThreadPrefixSums(
      thread_values,
      first_running_sum,
      PrefixSums<Sum>{.sums = sums, .count = values.count},
      thread_first
  );
}

// The way up: fills every layer of `layers` with chunk totals, the top layer with the total of all values.
RunOnCallStatus SumLayersUp(const std::uint32_t* values, const PrefixSumLayers& layers, std::uint64_t* layer_totals) {
  if (const RunOnCallStatus status = ApplyGpuFunction(
          SumChunks<std::uint32_t>,
          ComputeChunkLaunchSize(layers.entry_count[1]),
          PrefixSumValues<std::uint32_t>{.values = values, .count = layers.entry_count[0]},
          layer_totals + layers.first_entry[1]
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  for (int layer = 2; layer <= layers.top_layer; ++layer) {
    if (const RunOnCallStatus status = ApplyGpuFunction(
            SumChunks<std::uint64_t>,
            ComputeChunkLaunchSize(layers.entry_count[layer]),
            PrefixSumValues<std::uint64_t>{
                .values = layer_totals + layers.first_entry[layer - 1],
                .count = layers.entry_count[layer - 1],
            },
            layer_totals + layers.first_entry[layer]
        );
        status != RunOnCallStatus::kDone) {
      return status;
    }
  }
  return RunOnCallStatus::kDone;
}

// The way down: turns layers top_layer - 1 to 1 into running sums, in place, then writes the exclusive prefix sums
// of the values to sums[0..value_count). The top layer keeps the total. `sums` may be `values`.
RunOnCallStatus WriteSumsDown(
    const std::uint32_t* values,
    const PrefixSumLayers& layers,
    std::uint64_t* layer_totals,
    std::uint32_t* sums
) {
  for (int layer = layers.top_layer - 1; layer >= 1; --layer) {
    // The layer just below the top is a single chunk, whose running sums start at 0.
    const std::uint64_t* const chunk_offsets =
        layer + 1 < layers.top_layer ? layer_totals + layers.first_entry[layer + 1] : nullptr;
    std::uint64_t* const layer_entries = layer_totals + layers.first_entry[layer];
    if (const RunOnCallStatus status = ApplyGpuFunction(
            WriteChunkPrefixSums<std::uint64_t, std::uint64_t>,
            ComputeChunkLaunchSize(layers.entry_count[layer + 1]),
            PrefixSumValues<std::uint64_t>{.values = layer_entries, .count = layers.entry_count[layer]},
            chunk_offsets,
            layer_entries
        );
        status != RunOnCallStatus::kDone) {
      return status;
    }
  }
  const std::uint64_t* const chunk_offsets = layers.top_layer > 1 ? layer_totals + layers.first_entry[1] : nullptr;
  return ApplyGpuFunction(
      WriteChunkPrefixSums<std::uint32_t, std::uint32_t>,
      ComputeChunkLaunchSize(layers.entry_count[1]),
      PrefixSumValues<std::uint32_t>{.values = values, .count = layers.entry_count[0]},
      chunk_offsets,
      sums
  );
}

// Procedure <proc-stable-order> over `pass_count` passes: writes to order[0..count) the indexes by ascending key,
// ties in the order of previous_order (index order when null). count is 1 or more. Waits for the GPU.
template <typename Key>
RunOnCallStatus PlaceByDigits(
    const Key* keys,
    std::uint32_t count,
    const std::uint32_t* previous_order,
    std::uint32_t* order,
    std::uint32_t pass_count,
    const GpuOrderBufferAreas& areas
) {
  const std::size_t chunk_count = ComputeChunkCount(count);
  const LaunchSize launch_size = ComputeChunkLaunchSize(chunk_count);
  const PrefixSumLayers table_layers = BuildPrefixSumLayers(std::size_t{kDigitValueCount} * chunk_count);

  // Step 1: the starting sequence; null stands for the indexes in ascending order.
  const std::uint32_t* sequence = previous_order;
  // Step 5: one pass per byte. No pass is skipped on the GPU; the CPU's skipping does not change the result.
  for (std::uint32_t pass = 0; pass < pass_count; ++pass) {
    // The passes alternate between `order` and the second sequence, chosen so that the last pass writes `order`.
    const bool writes_order = (pass_count - 1 - pass) % 2 == 0;
    std::uint32_t* const next_sequence = writes_order ? order : areas.second_sequence;
    const PassSource<Key> source = {.keys = keys, .sequence = sequence, .count = count, .pass = pass};

    // Step 2: count the digits of each chunk.
    if (const RunOnCallStatus status =
            ApplyGpuFunction(CountChunkDigits<Key>, launch_size, source, chunk_count, areas.chunk_counts);
        status != RunOnCallStatus::kDone) {
      return status;
    }

    // Step 3: the exclusive prefix sum of the table gives the first place of each digit in each chunk. Its total is
    // count, below 2^32, so it fits and needs no check.
    if (const RunOnCallStatus status = SumLayersUp(areas.chunk_counts, table_layers, areas.layer_totals);
        status != RunOnCallStatus::kDone) {
      return status;
    }
    if (const RunOnCallStatus status =
            WriteSumsDown(areas.chunk_counts, table_layers, areas.layer_totals, areas.chunk_counts);
        status != RunOnCallStatus::kDone) {
      return status;
    }

    // Step 4: place the items of each chunk.
    if (const RunOnCallStatus status =
            ApplyGpuFunction(PlaceChunkItems<Key>, launch_size, source, chunk_count, areas.chunk_counts, next_sequence);
        status != RunOnCallStatus::kDone) {
      return status;
    }
    sequence = next_sequence;
  }
  // Step 6: the last pass wrote `order`. The wait reports an error of any launch above.
  return CheckGpuWork();
}

// Sets *flag to 1 when any destination[0..item_count) is destination_count or more. atomicOr makes the many writes
// one after another instead of a data race, and its result, 1 if any thread found a destination out of range, is
// the same whatever the order of the threads.
__global__ void FlagDestinationsOutOfRange(
    const std::uint32_t* destination,
    std::size_t item_count,
    std::uint32_t destination_count,
    std::uint32_t* flag
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t item = stride.first_item; item < item_count; item += stride.item_step) {
    if (destination[item] >= destination_count) {
      atomicOr(flag, 1U);
    }
  }
}

// sorted_destination[place] = destination[item_order[place]]: the destinations in group order, ascending.
__global__ void ReadSortedDestinations(
    const std::uint32_t* destination,
    const std::uint32_t* item_order,
    std::size_t item_count,
    std::uint32_t* sorted_destination
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t place = stride.first_item; place < item_count; place += stride.item_step) {
    sorted_destination[place] = destination[item_order[place]];
  }
}

// first_of_group[group] = ComputeGroupStart(destinations, group) for group in [0, group_start_count).
__global__ void
FindGroupStarts(SortedDestinations destinations, std::size_t group_start_count, std::uint32_t* first_of_group) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t group = stride.first_item; group < group_start_count; group += stride.item_step) {
    first_of_group[group] = ComputeGroupStart(destinations, group);
  }
}

}  // namespace

RunOnCallStatus BuildGpuStableOrder(
    const std::uint64_t* keys,
    std::uint32_t count,
    const std::uint32_t* previous_order,
    std::uint32_t* order,
    const GpuOrderBufferAreas& areas
) {
  return PlaceByDigits(keys, count, previous_order, order, kKeyPassCount, areas);
}

RunOnCallStatus BuildGpuGroupsByDestination(
    const std::uint32_t* destination,
    std::uint32_t item_count,
    std::uint32_t destination_count,
    std::uint32_t* first_of_group,
    std::uint32_t* item_order,
    const GpuOrderBufferAreas& areas
) {
  // Groups steps 3 to 5 of Procedure <proc-groups-by-destination>; steps 1 and 2 ran in base/stable_order_cpu.cc.
  if (item_count > 0) {
    // Groups step 3: every destination must name a group, before anything is written.
    if (const cudaError_t error = cudaMemset(areas.destination_flag, 0, sizeof(std::uint32_t)); error != cudaSuccess) {
      return ToRunOnCallStatus(error);
    }
    if (const RunOnCallStatus status = ApplyGpuFunction(
            FlagDestinationsOutOfRange,
            ComputeLaunchSize(item_count),
            destination,
            item_count,
            destination_count,
            areas.destination_flag
        );
        status != RunOnCallStatus::kDone) {
      return status;
    }
    std::uint32_t flag = 0;
    // The copy waits for the flag to be final.
    if (const cudaError_t error = cudaMemcpy(&flag, areas.destination_flag, sizeof(flag), cudaMemcpyDeviceToHost);
        error != cudaSuccess) {
      return ToRunOnCallStatus(error);
    }
    if (flag != 0) {
      // A destination names no group: nothing was written.
      return RunOnCallStatus::kSizeMismatch;
    }

    // Groups step 4: one stable order with the destination as the key.
    if (const RunOnCallStatus status = PlaceByDigits(
            destination,
            item_count,
            nullptr,
            item_order,
            ComputeDestinationPassCount(destination_count),
            areas
        );
        status != RunOnCallStatus::kDone) {
      return status;
    }
    // The second sequence is free once the order is final; it now holds the destinations in group order.
    if (const RunOnCallStatus status = ApplyGpuFunction(
            ReadSortedDestinations,
            ComputeLaunchSize(item_count),
            destination,
            item_order,
            item_count,
            areas.second_sequence
        );
        status != RunOnCallStatus::kDone) {
      return status;
    }
  }

  // Groups step 5: each group start by binary search. With no item every start is 0.
  const std::size_t group_start_count = std::size_t{destination_count} + 1;
  if (const RunOnCallStatus status = ApplyGpuFunction(
          FindGroupStarts,
          ComputeLaunchSize(group_start_count),
          SortedDestinations{.values = areas.second_sequence, .count = item_count},
          group_start_count,
          first_of_group
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  return CheckGpuWork();
}

RunOnCallStatus ComputeGpuExclusivePrefixSum(
    const std::uint32_t* values,
    std::uint32_t count,
    std::uint32_t* sums,
    const GpuOrderBufferAreas& areas
) {
  if (count == 0) {
    // No values: the total sums[0] is 0.
    if (const cudaError_t error = cudaMemset(sums, 0, sizeof(std::uint32_t)); error != cudaSuccess) {
      return ToRunOnCallStatus(error);
    }
    return CheckGpuWork();
  }
  const PrefixSumLayers layers = BuildPrefixSumLayers(count);
  if (const RunOnCallStatus status = SumLayersUp(values, layers, areas.layer_totals);
      status != RunOnCallStatus::kDone) {
    return status;
  }
  // The total, before any sum is written. The copy waits for the way up.
  std::uint64_t total = 0;
  if (const cudaError_t error = cudaMemcpy(
          &total,
          areas.layer_totals + layers.first_entry[layers.top_layer],
          sizeof(total),
          cudaMemcpyDeviceToHost
      );
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (total > std::numeric_limits<std::uint32_t>::max()) {
    // The sums would not fit in 32 bits: nothing was written.
    return RunOnCallStatus::kSizeMismatch;
  }
  if (const RunOnCallStatus status = WriteSumsDown(values, layers, areas.layer_totals, sums);
      status != RunOnCallStatus::kDone) {
    return status;
  }
  const auto total_sum = static_cast<std::uint32_t>(total);
  if (const cudaError_t error = cudaMemcpy(sums + count, &total_sum, sizeof(total_sum), cudaMemcpyHostToDevice);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  return CheckGpuWork();
}

}  // namespace fluidity
