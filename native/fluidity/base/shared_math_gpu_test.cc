// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Launches every shared math function in GPU code and compares each result with the CPU's. The functions that
// promise the same bits must give the same bits (or NaN on both). The others are compared within kLastBitsTolerance,
// and the test prints the largest difference it saw for each, so the hardware run can calibrate the tolerance.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "fluidity/base/run_on_call.h"
#include "fluidity/base/shared_math_test_gpu.h"
#include "fluidity/base/test_support.h"

namespace fluidity {
namespace {

// A few rounding units of a value near 1, both as a floor and per unit of the value, as scalar_math_test.cc uses
// for the same functions on one processor. Not yet measured across processors: the hardware run reports the
// largest difference.
constexpr Tolerance kLastBitsTolerance = {.absolute_floor = 1e-15, .relative_term = 1e-15};

constexpr std::array<const char*, kSameBitsResultCount> kSameBitsNames = {
    "Floor",
    "Abs",
    "Sqrt",
    "Min",
    "Max",
    "IsFinite",
    "IsFinite of an overflow",
    "Length",
    "Normalized x",
    "Normalized y",
    "Normalized z",
    "TreeSum",
    "TreeSumOf",
    "LargerMagnitude",
};

constexpr std::array<const char*, kLastBitsResultCount> kLastBitsNames = {
    "Cbrt",
    "Exp",
    "Expm1",
    "Log",
    "Pow",
    "Sin",
    "Cos",
    "Rotation xx",
    "Rotation xy",
    "Rotation xz",
    "Rotation yx",
    "Rotation yy",
    "Rotation yz",
    "Rotation zx",
    "Rotation zy",
    "Rotation zz",
};

// The results of every input on the GPU and on the CPU.
struct ResultsOnBoth {
  std::vector<SharedMathResult> on_gpu;
  std::vector<SharedMathResult> on_cpu;
};

// Computes every input on both processors. Fails the calling test when the GPU half fails.
ResultsOnBoth ComputeOnBoth() {
  std::vector<SharedMathInputs> inputs(kSharedMathInputCount);
  for (int input_index = 0; input_index < kSharedMathInputCount; ++input_index) {
    inputs[input_index] = BuildSharedMathInput(input_index);
  }
  ResultsOnBoth results{.on_gpu = std::vector<SharedMathResult>(kSharedMathInputCount), .on_cpu = {}};
  const RunOnCallStatus status = ComputeSharedMathOnGpu(inputs.data(), kSharedMathInputCount, results.on_gpu.data());
  EXPECT_EQ(status, RunOnCallStatus::kDone);
  for (const SharedMathInputs& input : inputs) {
    results.on_cpu.push_back(ComputeSharedMathResults(input));
  }
  return results;
}

TEST(SharedMath, SameBitsFunctionsOnGpu_GiveTheCpuBits) {
  const ResultsOnBoth results = ComputeOnBoth();
  for (int input_index = 0; input_index < kSharedMathInputCount; ++input_index) {
    for (int result_index = 0; result_index < kSameBitsResultCount; ++result_index) {
      EXPECT_TRUE(CheckSameBitsOrBothNaN(
          results.on_cpu[input_index].same_bits[result_index],
          results.on_gpu[input_index].same_bits[result_index]
      )) << kSameBitsNames[result_index]
         << ", input " << input_index;
    }
  }
}

TEST(SharedMath, LastBitsFunctionsOnGpu_AgreeWithinTheTolerance) {
  const ResultsOnBoth results = ComputeOnBoth();
  for (int result_index = 0; result_index < kLastBitsResultCount; ++result_index) {
    double largest_difference = 0.0;
    for (int input_index = 0; input_index < kSharedMathInputCount; ++input_index) {
      const double on_cpu = results.on_cpu[input_index].last_bits[result_index];
      const double on_gpu = results.on_gpu[input_index].last_bits[result_index];
      largest_difference = std::max(largest_difference, std::abs(on_cpu - on_gpu));
      EXPECT_TRUE(CheckWithinTolerance(on_cpu, on_gpu, kLastBitsTolerance))
          << kLastBitsNames[result_index] << ", input " << input_index;
    }
    // Read by the owner after the hardware run, to calibrate kLastBitsTolerance.
    std::cout << kLastBitsNames[result_index] << ": largest |cpu - gpu| = " << largest_difference << "\n";
  }
}

}  // namespace
}  // namespace fluidity
