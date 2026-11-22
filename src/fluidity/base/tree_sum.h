// Sums of many doubles in one fixed order, for code that both the CPU and the GPU run.
//
// Purpose: give every solver's convergence test and every conservation total a sum whose bits depend only on the
//   values and their indexes, so they are the same on every run and on both processors (unless the result is NaN,
//   whose bits IEEE 754 leaves open).
//
// Idea: a computer rounds the result of each addition, so adding the same doubles in another order can change the
//   result. TreeSum fixes the order. It pads the values with zeros up to the next power of two and adds them in a
//   binary tree. In ordinary tree words, each node is its left child plus its right child, and the values are at
//   the bottom; this code calls a node a pair sum and its children its left and right sides. Each pair sum is
//   rounded once. The padding is never kept in memory; a pair whose right side covers only padding adds +0.0.
//   IEEE 754 defines the rounded result of every addition exactly, so the CPU and the GPU, which add the same
//   pairs, give the same bits, unless the result is NaN, whose bits IEEE 754 leaves open.
//   Example: {1e16, 1, -1e16, 1}. Left to right gives 1.0. The tree gives (1e16 + 1) + (-1e16 + 1) = 0.0, because
//   near 1e16 the doubles are 2 apart and each pair rounds to plus or minus 1e16. The exact sum is 2. Neither
//   answer is more right; what matters is that the tree's order never changes.
//
// Words:
//   pair sum   one sum of the tree: its left side plus its right side, where a side is a value or a pair sum of
//              the layer below
//   layer      the pair sums at one height of the tree: layer 1 adds pairs of values, layer 2 adds pairs of layer 1
//              sums, and so on
//   padding    the zeros that fill the values up to the next power of two; never kept in memory
//   magnitude  the absolute value of a number
// Owns:    math::TreeSumOf, math::TreeSum, math::LargerMagnitude
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), math::Abs (base/scalar_math.h)
// Used by: base/ordered_sums.h (ComputeTreeSum, ComputeTreeDot, ComputeLargestMagnitude), and GPU code that sums
//          a few values in one thread
// Read:    docs/method-paper/fluidity-method.typ <sec-fixed-order-sums>

#pragma once

#include <concepts>
#include <cstddef>
#include <limits>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"

namespace fluidity::math {

// Computes the sum of value_at(0), ..., value_at(count - 1) in the fixed order of the padded binary tree.
//
// Method: follows Procedure <proc-tree-sum>. Calls value_at once per index, in index order, and keeps each finished
//   left side until its right side is ready, so it adds exactly the pairs of the tree, left side first. Allocates
//   nothing and does not recurse, so GPU code may call it; it keeps 64 doubles on the stack. value_at(index)
//   returns a double that is rounded before it is added: with value_at = first[index] * second[index] the product
//   is rounded once, then added, never fused with the addition (see TreeSum below and base/ordered_sums.h).
// Order: the pairs depend only on `count`. Same bits on every run, and the same bits on the CPU and the GPU,
//   unless the result is NaN, whose bits IEEE 754 leaves open.
// Example: TreeSumOf(3, [](std::size_t index) { return index + 1.0; }) adds (1 + 2) + (3 + 0.0) = 6.
//   count 0 gives +0.0 and calls value_at never.
// Read: docs/method-paper/fluidity-method.typ <proc-tree-sum>
template <typename ValueAt>
  requires std::invocable<ValueAt&, std::size_t>
FLUIDITY_CPU_AND_GPU double TreeSumOf(std::size_t count, ValueAt value_at) noexcept {
  // Steps: 1 no values  2 pairs with real values on the right  3 pairs with padding on the right  4 the top
  //
  // Why this adds exactly the pairs of the padded tree. Index means an index of the padded array. Layer r of the
  // tree has one side for each run of 2^r indexes [j * 2^r, (j + 1) * 2^r), and side j of layer r + 1 is side 2j plus
  // side 2j + 1 of layer r.
  //   Reading. After `read_count` values have been read, left_side_waiting[r] holds, for each bit r set in
  //   read_count, the side of layer r that covers [high, high + 2^r), where high is read_count with bits 0 to r
  //   cleared. These sides cover [0, read_count) exactly, and each is complete: it covers only values already
  //   read. Reading value `index` == read_count gives a side of layer 0. Adding 1 to read_count carries through
  //   its lowest run of set bits 0, ..., t - 1. For each such r, the side just built covers the 2^r indexes right
  //   after left_side_waiting[r], so the two are sides 2j and 2j + 1 of one pair, and their sum, left first, is
  //   the side of layer r + 1. Bit t of read_count is clear, so the sum waits at t. That is the invariant for
  //   read_count + 1.
  //   Finishing. Once every value is read, let `low` be the lowest set bit of count. left_side_waiting[low] covers
  //   [count - 2^low, count): it holds the last value. Climbing from layer `low`, keep the side j of layer r that
  //   holds the last value, j = (count - 1) >> r. If j is odd, its left neighbour j - 1 covers only real values,
  //   and it is the waiting side of bit r: for r above `low`, count and count - 1 have the same bit r. If j is
  //   even, its right neighbour j + 1 starts at or after count, so it covers only padding, and every sum of
  //   padding is +0.0 + +0.0 = +0.0. At r == low, j is even, because bit `low` of count - 1 is clear. When j is 0
  //   the side covers every index: it is the top. The pairs not computed, made of padding alone, are all +0.0,
  //   and each is used as written above, so the result has the bits of the padded tree added layer by layer.
  //   Size. read_count is at most count - 1 when a value is read, below 2^64 - 1, so its run of set bits stops at
  //   bit 63 or below, and 64 waiting sums are enough.

  // Step 1: no values.
  if (count == 0) {
    return 0.0;
  }

  constexpr int kMaxLayerCount = std::numeric_limits<std::size_t>::digits;
  // A C array, not std::array: GPU code cannot call std::array's member functions, which are CPU functions
  // (nvcc without --expt-relaxed-constexpr).
  double left_side_waiting[kMaxLayerCount] = {};  // NOLINT(modernize-avoid-c-arrays): see the comment above.

  // Step 2, while reading: every pair whose two sides cover only real values.
  for (std::size_t index = 0; index < count; ++index) {
    double side = value_at(index);
    unsigned int layer = 0;
    while (((index >> layer) & 1U) != 0) {
      // -ffp-contract=off or /fp:precise (CPU) and --fmad=false (GPU) keep a product returned by value_at rounded
      // before this addition: a fused multiply-add would round the two together and give other bits.
      side = left_side_waiting[layer] + side;
      ++layer;
    }
    left_side_waiting[layer] = side;
  }

  unsigned int layer = 0;
  while (((count >> layer) & 1U) == 0) {
    ++layer;
  }
  double side = left_side_waiting[layer];
  std::size_t side_index = (count - 1) >> layer;
  while (side_index != 0) {
    if ((side_index & 1U) != 0) {
      // Step 2, while finishing: the right side reaches past the last value but covers some real values.
      side = left_side_waiting[layer] + side;
    } else {
      // Step 3: the right side covers only padding. Adding +0.0 keeps every number's bits, except -0.0
      // becomes +0.0 (and a NaN may change bits).
      side = side + 0.0;
    }
    side_index >>= 1U;
    ++layer;
  }

  // Step 4: the side that covers every index. With one value, the value itself: no pair was added.
  return side;
}

// Computes the sum of values[0..count) in the fixed order of the padded binary tree.
//
// Method: TreeSumOf over the values; see it for the order. Same bits on every run, and on the CPU and the GPU
//   unless the result is NaN, whose bits IEEE 754 leaves open.
//   The CPU builds its library with -ffp-contract=off (/fp:precise on MSVC) and the GPU with --fmad=false, so no
//   addition is fused.
// Order: values are added as Procedure <proc-tree-sum> says: pairs of neighbours, then pairs of pair sums.
// Example: TreeSum({1e16, 1, -1e16, 1}, 4) = 0.0, while a left-to-right loop gives 1.0.
//   Seven values give the bits of the same seven followed by 0.0.
//   Zeros: the one value -0.0 gives -0.0, since no addition happens. Padding follows IEEE addition, where
//   -0.0 + 0.0 is +0.0: {-0.0, -0.0, -0.0} is padded to four and gives (-0.0 + -0.0) + (-0.0 + 0.0) = +0.0, while
//   {-0.0, -0.0} needs no padding and gives -0.0.
//   count 0 gives +0.0, and `values` may then be null. Otherwise `values` points to at least `count` doubles.
// Read: docs/method-paper/fluidity-method.typ <sec-fixed-order-sums>
FLUIDITY_CPU_AND_GPU inline double TreeSum(const double* values, std::size_t count) noexcept {
  return TreeSumOf(count, [values](std::size_t index) {
    return values[index];
  });
}

// Computes the larger of the magnitudes (absolute values) of two values, or NaN when either is NaN.
//
// Method: same bits on CPU and GPU, since comparisons round nothing. math::Max returns the other value when one is
//   NaN; here a NaN on either side wins, so a running maximum can never hide a NaN. Which NaN comes back, and its
//   bits, are not promised.
// Ties: both zeros give +0.0.
// Example: LargerMagnitude(-3, 2) = 3. LargerMagnitude(1, NaN) is NaN, and so is LargerMagnitude(NaN, 1).
// Read: docs/method-paper/fluidity-method.typ <sec-fixed-order-sums>
FLUIDITY_CPU_AND_GPU constexpr double LargerMagnitude(double first, double second) noexcept {
  const double first_magnitude = Abs(first);
  const double second_magnitude = Abs(second);
  // Only a NaN compares unequal to itself. Each side is tested, because a NaN fails every comparison.
  const bool first_is_nan = !(first_magnitude == first_magnitude);
  if (first_is_nan) {
    return first_magnitude;
  }
  const bool second_is_nan = !(second_magnitude == second_magnitude);
  if (second_is_nan) {
    return second_magnitude;
  }
  if (second_magnitude > first_magnitude) {
    return second_magnitude;
  }
  return first_magnitude;
}

}  // namespace fluidity::math
