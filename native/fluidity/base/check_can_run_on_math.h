// The fixed computation of the GPU math check, written once for both processors.
//
// Purpose: give the GPU math check one function that computes the results of one input, so the GPU function and
//   the CPU loop call the same code and cannot drift apart.
//
// Idea: the check builds 32 inputs from a small integer formula and computes, for each, a cross, a dot, a fixed
//   rotation times a vector, a determinant, a square root, a floor and a length. These use only operations that
//   IEEE 754 defines as correctly rounded, so the two processors must agree to the last bit.
//   The inputs are picked so that a fused multiply-add on either processor, or flush-to-zero on the CPU, changes a
//   result. Each value is an odd whole number of 27 bits divided by 2^26, a double between 1.5 and 2. The product
//   of two such values needs 54 bits, one more than a double holds, so the product is rounded. A fused multiply-add
//   skips that rounding and gives other bits.
//   Every fourth input is scaled by 2^-520, so its products fall below 2^-1022 and are subnormal: flush-to-zero
//   would replace them by zero. No result is a NaN or an infinity, since a NaN's bits may differ between processors.
//
// Words:
//   subnormal           a double smaller than 2^-1022, the smallest normal double; it keeps fewer digits
//   fused multiply-add  first * second + third rounded once instead of twice
// Owns:    kMathCheckInputCount, MathCheckInputs, MathCheckResult, BuildMathCheckValue, BuildMathCheckInputs,
//          ComputeMathCheckResult
// Uses:    Vec3 and math::Cross, math::Dot, math::Length (base/vec3.h), Mat3 and math::Determinant (base/mat3.h),
//          math::Sqrt and math::Floor (base/scalar_math.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h)
// Used by: the GPU math check (base/check_can_run_on_gpu.cu)
// Read:    docs/method-paper/fluidity-method.typ <proc-check-can-run-on>

#pragma once

#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"

namespace fluidity {

// The number of fixed inputs. One GPU thread computes one input.
inline constexpr int kMathCheckInputCount = 32;

// One fixed input: two vectors. No unit: the check compares bits, not quantities.
struct MathCheckInputs {
  Vec3 first;
  Vec3 second;
};

// The results of one input, eleven doubles. No unit.
struct MathCheckResult {
  Vec3 cross;  // math::Cross(first, second).
  double dot;  // math::Dot(first, second).
  Vec3 rotated_first;  // The fixed rotation times first.
  double determinant;  // math::Determinant of the matrix with rows first, second, rotated_first.
  double square_root_of_dot;  // math::Sqrt(dot). The dot is positive for every input.
  double floor_of_scaled_dot;  // math::Floor(dot * 2^40).
  double length_of_difference;  // math::Length(first - second).
};

// Builds value number `part`, from 0 to 5, of fixed input number `input_index`: an odd whole number below 2^27,
// divided by 2^26.
//
// Method: the value is (2^27 - 1 - 2 k) / 2^26, where k = ((6 * input_index + part) * 2654435761) mod 2^24. The
//   multiplier is Knuth's multiplicative hash constant, used only because it spreads consecutive numbers, so the
//   values fall between 1.5 and 2 without a pattern. The whole number is below 2^53, so it converts to a double
//   exactly, and the division by a power of two is exact.
// Example: BuildMathCheckValue(0, 0) = (2^27 - 1) / 2^26 = 2 - 2^-26, about 1.999999985.
FLUIDITY_CPU_AND_GPU constexpr double BuildMathCheckValue(int input_index, int part) noexcept {
  constexpr std::uint64_t kPartCount = 6;
  constexpr std::uint64_t kSpreadMultiplier = 2654435761;
  constexpr std::uint64_t kSpreadRange = std::uint64_t{1} << 24U;
  constexpr std::uint64_t kOddValueLimit = std::uint64_t{1} << 27U;
  constexpr double kOddValueScale = 0x1p-26;

  const std::uint64_t spread_input =
      kPartCount * static_cast<std::uint64_t>(input_index) + static_cast<std::uint64_t>(part);
  const std::uint64_t spread = (spread_input * kSpreadMultiplier) % kSpreadRange;
  // Below 2^27 and odd, so the product of two values is odd and needs 54 bits.
  const std::uint64_t odd_whole_number = kOddValueLimit - 1 - 2 * spread;
  return static_cast<double>(odd_whole_number) * kOddValueScale;
}

// Builds fixed input number `input_index`, from 0 to kMathCheckInputCount - 1. The same index gives the same bits
// on every processor and every run.
//
// Method: first holds values 0, 1, 2 and second holds values 3, 4, 5 of BuildMathCheckValue. An odd input_index
//   negates first.z, so signs vary. When input_index mod 4 is 3, every value is multiplied by 2^-520, which is
//   exact: the values stay normal, near 2^-519, and their products, near 2^-1038, are subnormal.
// Example: BuildMathCheckInputs(0).first.x = 2 - 2^-26. BuildMathCheckInputs(3).first.x is about 1.55 * 2^-520.
FLUIDITY_CPU_AND_GPU constexpr MathCheckInputs BuildMathCheckInputs(int input_index) noexcept {
  constexpr double kSubnormalProductScale = 0x1p-520;

  const bool negates_first_z = input_index % 2 == 1;
  const bool has_subnormal_products = input_index % 4 == 3;
  const double first_z_sign = negates_first_z ? -1.0 : 1.0;
  const double scale = has_subnormal_products ? kSubnormalProductScale : 1.0;
  return {
      .first =
          {
              .x = BuildMathCheckValue(input_index, 0) * scale,
              .y = BuildMathCheckValue(input_index, 1) * scale,
              .z = first_z_sign * BuildMathCheckValue(input_index, 2) * scale,
          },
      .second = {
          .x = BuildMathCheckValue(input_index, 3) * scale,
          .y = BuildMathCheckValue(input_index, 4) * scale,
          .z = BuildMathCheckValue(input_index, 5) * scale,
      },
  };
}

// Computes the results of one input with the shared vector and matrix helpers. The GPU function and the CPU loop
// of the GPU math check both call it.
//
// Method: only add, subtract, multiply, square root and floor, each correctly rounded. The fixed rotation
//   turns about Z by the angle whose cosine is 0.6 and whose sine is 0.8; its entries are the doubles nearest 0.6
//   and 0.8, so it is a rotation to within rounding, and the check needs only that it is fixed.
// Order: as each helper gives it: x, y, z for sums, matrix row x first for the determinant.
// Example: for BuildMathCheckInputs(0), dot = 0x1.39c6a26df9f00p+3, about 9.806.
FLUIDITY_CPU_AND_GPU inline MathCheckResult ComputeMathCheckResult(MathCheckInputs inputs) noexcept {
  constexpr Mat3 kFixedRotation = {
      .xx = 0.6,
      .xy = -0.8,
      .xz = 0,
      .yx = 0.8,
      .yy = 0.6,
      .yz = 0,
      .zx = 0,
      .zy = 0,
      .zz = 1,
  };
  // 2^40 leaves about eight bits after the binary point of a dot near 10, so the floor depends on low bits.
  constexpr double kFloorScale = 0x1p40;

  const Vec3 first = inputs.first;
  const Vec3 second = inputs.second;
  const Vec3 rotated_first = kFixedRotation * first;
  const Mat3 rows_first_second_rotated = {
      .xx = first.x,
      .xy = first.y,
      .xz = first.z,
      .yx = second.x,
      .yy = second.y,
      .yz = second.z,
      .zx = rotated_first.x,
      .zy = rotated_first.y,
      .zz = rotated_first.z,
  };
  const double dot = math::Dot(first, second);
  return {
      .cross = math::Cross(first, second),
      .dot = dot,
      .rotated_first = rotated_first,
      .determinant = math::Determinant(rows_first_second_rotated),
      .square_root_of_dot = math::Sqrt(dot),
      .floor_of_scaled_dot = math::Floor(dot * kFloorScale),
      .length_of_difference = math::Length(first - second),
  };
}

}  // namespace fluidity
