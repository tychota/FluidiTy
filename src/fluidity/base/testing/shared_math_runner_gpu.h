// The shared math evaluated for one input, and its GPU runner, for scalar_math.shared.gpu.test.cc, private to base/.
//
// Purpose: run every shared math function on the GPU with fixed inputs, so the test can compare each result with the
//   CPU's: bit for bit for the functions that promise the same bits, within a tolerance for the others.
//
// Idea: ComputeSharedMathResults computes, for one input, one result per shared function. It is marked
//   FLUIDITY_CPU_AND_GPU, so the GPU function and the CPU loop of the test call the same code. The results come in two
//   arrays: same_bits for Floor, Abs, Sqrt, Min, Max, IsFinite, Length, Normalized, TreeSum, TreeSumOf and
//   LargerMagnitude, which IEEE 754 rounds exactly; last_bits for Cbrt, Exp, Expm1, Log, Pow, Sin, Cos and Rotation,
//   which call each processor's own library. ComputeSharedMathOnGpu copies the inputs to the GPU, runs one thread per
//   input, and copies the results back. Each Sqrt of a negative value gives NaN on both processors, whose bits may
//   differ.
//
// Owns:    kSharedMathInputCount, kSameBitsResultCount, kLastBitsResultCount, SharedMathInputs, SharedMathResult,
//          BuildSharedMathInput, ComputeSharedMathResults, ComputeSharedMathOnGpu
// Uses:    Vec3 (base/vec3.h), Mat3 and math::Rotation (base/mat3.h), the scalar math (base/scalar_math.h),
//          math::TreeSum, math::TreeSumOf and math::LargerMagnitude (base/tree_sum.h), RunOnCallStatus
//          (base/run_on_call.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h)
// Used by: scalar_math.shared.gpu.test.cc, and testing/shared_math_runner_gpu.cu, which defines ComputeSharedMathOnGpu
// Read:    docs/method-paper/fluidity-method.typ <sec-conventions>

#pragma once

#include <cstddef>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/tree_sum.h"
#include "fluidity/base/vec3.h"

namespace fluidity {

// The number of fixed inputs; one GPU thread per input.
inline constexpr int kSharedMathInputCount = 64;

// Results that must have the same bits on both processors, or both be NaN: Floor, Abs, Sqrt, Min, Max, IsFinite (1
// or 0) twice, Length, the three axes of Normalized, TreeSum and TreeSumOf of seven values, and LargerMagnitude.
inline constexpr int kSameBitsResultCount = 14;

// Results that may differ in the last bits: Cbrt, Exp, Expm1, Log, Pow, Sin, Cos and the nine entries of Rotation.
inline constexpr int kLastBitsResultCount = 16;

// One fixed input. No unit: the test compares numbers, not quantities.
struct SharedMathInputs {
  double value = 0.0;  // From about -36 to 36, both signs.
  double other = 0.0;  // From about -20 to 20, both signs.
  Vec3 vector{};  // Never zero: z is at least 1.
};

// The results of one input, in the order of the counts above.
struct SharedMathResult {
  // C arrays, so the struct is copied between the processors as plain bytes.
  double same_bits[kSameBitsResultCount];  // NOLINT(modernize-avoid-c-arrays): see the comment above.
  double last_bits[kLastBitsResultCount];  // NOLINT(modernize-avoid-c-arrays): see the comment above.
};

// The input of index `input_index`, from a small integer formula, the same on both processors.
FLUIDITY_CPU_AND_GPU constexpr SharedMathInputs BuildSharedMathInput(int input_index) {
  const double value = (((input_index * 37) % 101) - 50) * 0.73 + input_index * 1e-3;
  const double other = (((input_index * 53) % 97) - 48) * 0.41;
  return {.value = value, .other = other, .vector = {.x = value, .y = other, .z = 1.0 + input_index * 0.1}};
}

// Every shared math function of one input, as SharedMathResult describes.
FLUIDITY_CPU_AND_GPU inline SharedMathResult ComputeSharedMathResults(SharedMathInputs input) {
  SharedMathResult results{};
  // value * kOverflowFactor * 10.0 overflows to infinity for any |value| above about 0.18 (all inputs but the
  // near-zero ones), so IsFinite sees an infinity made from a finite input.
  constexpr double kOverflowFactor = 1e308;
  const double positive_value = math::Abs(input.value) + 0.25;
  const Vec3 normalized = math::Normalized(input.vector);
  results.same_bits[0] = math::Floor(input.value);
  results.same_bits[1] = math::Abs(input.value);
  results.same_bits[2] = math::Sqrt(input.value);
  results.same_bits[3] = math::Min(input.value, input.other);
  results.same_bits[4] = math::Max(input.value, input.other);
  results.same_bits[5] = math::IsFinite(input.value) ? 1.0 : 0.0;
  results.same_bits[6] = math::IsFinite(input.value * kOverflowFactor * 10.0) ? 1.0 : 0.0;
  results.same_bits[7] = math::Length(input.vector);
  results.same_bits[8] = normalized.x;
  results.same_bits[9] = normalized.y;
  results.same_bits[10] = normalized.z;
  // Seven values whose sum rounds differently in another order: the tree's order is what is compared. A C array:
  // GPU code cannot call std::array's member functions.
  constexpr std::size_t kTreeValueCount = 7;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): see the comment above.
  const double tree_values[kTreeValueCount] = {
      input.value,
      1e16,
      input.other,
      -1e16,
      input.value * input.other,
      -0.0,
      0.5,
  };
  results.same_bits[11] = math::TreeSum(tree_values, kTreeValueCount);
  results.same_bits[12] = math::TreeSumOf(kTreeValueCount, [&tree_values, input](std::size_t index) {
    return tree_values[index] * input.other;
  });
  results.same_bits[13] = math::LargerMagnitude(input.value, input.other);
  const Mat3 rotation = math::Rotation(input.vector * 0.1);
  results.last_bits[0] = math::Cbrt(input.value);
  results.last_bits[1] = math::Exp(input.value);
  results.last_bits[2] = math::Expm1(input.value * 1e-3);
  results.last_bits[3] = math::Log(positive_value);
  results.last_bits[4] = math::Pow(positive_value, input.other * 0.1);
  results.last_bits[5] = math::Sin(input.value);
  results.last_bits[6] = math::Cos(input.value);
  results.last_bits[7] = rotation.xx;
  results.last_bits[8] = rotation.xy;
  results.last_bits[9] = rotation.xz;
  results.last_bits[10] = rotation.yx;
  results.last_bits[11] = rotation.yy;
  results.last_bits[12] = rotation.yz;
  results.last_bits[13] = rotation.zx;
  results.last_bits[14] = rotation.zy;
  results.last_bits[15] = rotation.zz;
  return results;
}

// Computes ComputeSharedMathResults(inputs[i]) on the GPU for i below `count`, into the CPU array `results`.
// Writes `results` once the GPU work is done; when a later step fails (freeing the GPU memory), the status says so
// and `results` may already hold the values.
// Fails: kOutOfMemory  the GPU had no memory for the inputs or the results
//        kCallFailed   another GPU call failed
RunOnCallStatus ComputeSharedMathOnGpu(const SharedMathInputs* inputs, int count, SharedMathResult* results);

}  // namespace fluidity
