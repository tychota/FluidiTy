// The tree sum one layer at a time, one pair sum per call, as the GPU computes it.
//
// Purpose: give the GPU half of base/ordered_sums.h the formula of one pair sum and the sizes of its layer arrays,
//   written once, so CPU tests can run every layer pair by pair and compare with math::TreeSum, and SumBuffers is
//   sized from the same counts the GPU uses.
//
// Idea: layer 1 adds pairs of values, layer 2 pairs of layer 1 sums, and so on, until one sum is left (see
//   base/tree_sum.h for the words). Pair `pair` of a layer of `side_count` sides is sides[2 * pair] plus
//   sides[2 * pair + 1]; when the right side is past the end it covers only padding, so the pair is the left side
//   plus +0.0. The pair sums of one layer do not depend on each other, so any order of the pairs, or all at once,
//   gives the same layer. A layer writes into one of two arrays that take turns: layers 1, 3, 5, ... into the first,
//   which holds the ComputePairCount(count) sums of layer 1; layers 2, 4, ... into the second.
//   Example: 5 values make layers of 3, 2 and 1 sums; the two arrays hold 3 + 2 = 5 doubles.
//
// Owns:    ComputePairCount, ComputeLayerValueCount, ComputeLayerArrayOffset, LayerSides, ComputeLayerPairSum,
//          ComputeLayerLargerMagnitude
// Uses:    math::LargerMagnitude (base/tree_sum.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h)
// Used by: base/ordered_sums_gpu.cu, base/ordered_sums_cpu.cc (the size of the sum buffers),
//          base/tree_sum_layers.test.cc
// Read:    docs/method-paper/fluidity-method.typ <proc-tree-sum>

#pragma once

#include <cstddef>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/tree_sum.h"

namespace fluidity {

// The number of pair sums a layer of `side_count` sides makes: side_count / 2, rounded up, without overflow.
// Example: 5 sides make 3 pair sums; 1 side makes 1 (the side plus padding); 0 sides make 0.
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputePairCount(std::size_t side_count) {
  return side_count / 2 + side_count % 2;
}

// The doubles the two layer arrays hold for a sum of `count` values: the sums of layer 1, then those of layer 2.
// Every later layer is smaller than the one written into the same array before it. 0 for count 0 or 1, which need
// no layer. At most count + 1, so it cannot overflow.
// Example: 5 values give 3 + 2 = 5; 1,000 values give 500 + 250 = 750.
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputeLayerValueCount(std::size_t count) {
  if (count < 2) {
    return 0;
  }
  return ComputePairCount(count) + ComputePairCount(ComputePairCount(count));
}

// Where layer `layer` (1 for the first) of a sum of `count` values writes its pair sums, in doubles from the start
// of the layer arrays: odd layers write the first array, at 0; even layers the second, after the
// ComputePairCount(count) sums of layer 1.
// Example: 5 values: layer 1 writes at 0, layer 2 at 3, layer 3 at 0 again.
FLUIDITY_CPU_AND_GPU constexpr std::size_t ComputeLayerArrayOffset(std::size_t count, std::size_t layer) {
  return layer % 2 == 1 ? 0 : ComputePairCount(count);
}

// The sides one layer reads: the values, or the pair sums of the layer below. Not owned.
struct LayerSides {
  const double* values = nullptr;  // `count` sides in memory of the processor that runs the layer.
  std::size_t count = 0;  // The number of sides.
};

// Pair sum `pair` of a layer: sides[2 * pair] + sides[2 * pair + 1], or sides[2 * pair] + 0.0 when the right side
// is at or past sides.count (padding). `pair` is below ComputePairCount(sides.count).
// Method: Procedure <proc-tree-sum>, steps 2 and 3, for one pair.
// Example: sides {1, 2, 3}: pair 0 is 3; pair 1 is 3 + 0.0 = 3.
FLUIDITY_CPU_AND_GPU inline double ComputeLayerPairSum(LayerSides sides, std::size_t pair) {
  const std::size_t left_index = 2 * pair;
  const std::size_t right_index = left_index + 1;
  if (right_index < sides.count) {
    // Step 2: the right side covers real values.
    return sides.values[left_index] + sides.values[right_index];
  }
  // Step 3: the right side covers only padding, whose sum is +0.0. Adding it turns -0.0 into +0.0, as on the CPU.
  return sides.values[left_index] + 0.0;
}

// Pair `pair` of a layer of the largest magnitude: math::LargerMagnitude of its two sides, or of the left side and
// +0.0 when the right side is padding. Any order gives the same answer; the layers reuse the code of the sum.
// Example: sides {-3, 2, -5}: pair 0 is 3; pair 1 is 5.
FLUIDITY_CPU_AND_GPU inline double ComputeLayerLargerMagnitude(LayerSides sides, std::size_t pair) {
  const std::size_t left_index = 2 * pair;
  const std::size_t right_index = left_index + 1;
  if (right_index < sides.count) {
    return math::LargerMagnitude(sides.values[left_index], sides.values[right_index]);
  }
  return math::LargerMagnitude(sides.values[left_index], 0.0);
}

}  // namespace fluidity
