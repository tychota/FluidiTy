// The GPU half of the ordered sums, private to base/: the reductions declared in base/ordered_sums_gpu.h.
//
// Each reduction adds the padded tree of math::TreeSum one layer at a time: one GPU call per layer, each thread
// computing pair sums with the shared formula of base/tree_sum_layers.h, which CPU tests run pair by pair. A layer
// reads the sums of the layer below and writes half as many, so the two layer arrays of the sum buffers take turns. The
// numbered steps match Procedure <proc-tree-sum>; Step 1 (no values) is handled by the caller,
// base/ordered_sums_cpu.cc. No atomic additions and no library reduction are used: their order of additions is not
// fixed. Every call uses the default stream, so the layers run in order, and the final copy to the CPU waits for
// them. Launches, waits and statuses go through base/gpu_launch.cuh.
// Read: docs/method-paper/fluidity-method.typ <proc-tree-sum>

#include <cuda_runtime.h>

#include <cstddef>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/ordered_sums_gpu.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/tree_sum.h"
#include "fluidity/base/tree_sum_layers.h"

namespace fluidity {
namespace {

// The pair rule of math::TreeSum: steps 2 and 3 for one pair.
struct AddPair {
  __device__ static double ComputePair(LayerSides sides, std::size_t pair) {
    return ComputeLayerPairSum(sides, pair);
  }
};

// The pair rule of the largest magnitude.
struct KeepLargerMagnitude {
  __device__ static double ComputePair(LayerSides sides, std::size_t pair) {
    return ComputeLayerLargerMagnitude(sides, pair);
  }
};

// One layer: pair_sums[pair] = PairRule::ComputePair(sides, pair) for every pair below pair_count.
template <typename PairRule>
__global__ void CombineOneLayer(LayerSides sides, double* pair_sums, std::size_t pair_count) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t pair = stride.first_item; pair < pair_count; pair += stride.item_step) {
    pair_sums[pair] = PairRule::ComputePair(sides, pair);
  }
}

// products[index] = first[index] * second[index], each rounded once. Kept apart from the additions, which read the
// products from memory; --fmad=false would forbid fusing them anyway.
__global__ void MultiplyPairs(const double* first, const double* second, std::size_t count, double* products) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < count; index += stride.item_step) {
    products[index] = first[index] * second[index];
  }
}

// Combines `count` sides of GPU memory layer by layer with PairRule, until one is left, and copies it to *result.
// count is at least 1; for count above 1, `layer_values` holds ComputeLayerValueCount(count) doubles of GPU memory.
// Returns kDone, or the status of the first call that failed; *result is written on success only.
template <typename PairRule>
RunOnCallStatus CombineInLayers(const double* sides, std::size_t count, double* layer_values, double* result) {
  // Steps 2 and 3 run inside CombineOneLayer, for every pair of a layer.
  std::size_t layer = 1;
  std::size_t side_count = count;
  while (side_count > 1) {
    const std::size_t pair_count = ComputePairCount(side_count);
    double* const pair_sums = layer_values + ComputeLayerArrayOffset(count, layer);
    if (const RunOnCallStatus status = ApplyGpuFunction(
            CombineOneLayer<PairRule>,
            ComputeLaunchSize(pair_count),
            LayerSides{.values = sides, .count = side_count},
            pair_sums,
            pair_count
        );
        status != RunOnCallStatus::kDone) {
      return status;
    }
    sides = pair_sums;
    side_count = pair_count;
    ++layer;
  }

  // Step 4: the side that covers every index. The copy waits for every layer to finish, so an error while a layer
  // ran reports here.
  double top = 0.0;
  if (const cudaError_t error = cudaMemcpy(&top, sides, sizeof(double), cudaMemcpyDeviceToHost); error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  *result = top;
  return RunOnCallStatus::kDone;
}

}  // namespace

RunOnCallStatus ComputeGpuTreeSum(const double* values, std::size_t count, GpuSumBufferAreas areas, double* result) {
  return CombineInLayers<AddPair>(values, count, areas.layer_values, result);
}

RunOnCallStatus ComputeGpuTreeDot(
    const double* first,
    const double* second,
    std::size_t count,
    GpuSumBufferAreas areas,
    double* result
) {
  if (const RunOnCallStatus status =
          ApplyGpuFunction(MultiplyPairs, ComputeLaunchSize(count), first, second, count, areas.products);
      status != RunOnCallStatus::kDone) {
    return status;
  }
  return CombineInLayers<AddPair>(areas.products, count, areas.layer_values, result);
}

RunOnCallStatus
ComputeGpuLargestMagnitude(const double* values, std::size_t count, GpuSumBufferAreas areas, double* result) {
  double largest = 0.0;
  if (const RunOnCallStatus status = CombineInLayers<KeepLargerMagnitude>(values, count, areas.layer_values, &largest);
      status != RunOnCallStatus::kDone) {
    return status;
  }
  // With one value no layer runs and the value comes back with its sign; this takes its magnitude. For more values
  // the layers already did, and taking it again changes nothing.
  *result = math::LargerMagnitude(largest, 0.0);
  return RunOnCallStatus::kDone;
}

}  // namespace fluidity
