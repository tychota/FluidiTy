// Math functions on one double, for formulas that both the CPU and the GPU run.
//
// Purpose: give every shared header one set of plain-named math functions that compile for the CPU and the GPU.
//
// Idea: the functions of <cmath> live in namespace std. The CUDA compiler supplies GPU versions of many of them,
//   but this file does not rely on that, so one spelling works in every shared header. Each function here calls
//   the C++ library on the CPU and the CUDA math library on the GPU, so a formula is written once with math::Sqrt,
//   math::Exp and the others.
//   The functions come in two groups.
//   Same bits on CPU and GPU: Floor, Abs, Sqrt, Min, Max, IsFinite and IsNan. IEEE 754 fixes their result exactly: Sqrt
//     is correctly rounded, and both processors follow IEEE 754 for a double square root. A formula built only
//     from these and from add, subtract, multiply and divide also gives the same bits, as math::Length does.
//     A result that is NaN is the exception: IEEE 754 leaves the bits of a NaN open, so each processor may give
//     its own NaN, as Sqrt(-1) does.
//   May differ in the last bits between CPU and GPU: Cbrt, Exp, Expm1, Log, Pow, Sin and Cos. Each processor
//     calls its own library, and IEEE 754 does not require these to be correctly rounded. Any formula that uses
//     one of them, such as math::Rotation, is compared between the CPU and the GPU with a tolerance.
//
// Words:
//   NaN               "not a number", the double that 0.0 / 0.0 gives. It compares unequal to every double.
//   correctly rounded the exact answer rounded once to the nearest double
//   rounding unit     the gap between a double and the next larger double
// Owns:    math::Floor, math::Abs, math::Sqrt, math::Cbrt, math::Exp, math::Expm1, math::Log, math::Pow,
//          math::Sin, math::Cos, math::Min, math::Max, math::IsFinite(double), math::IsNan,
//          math::kPositiveInfinity
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), the CUDA math library under the CUDA compiler
// Used by: base/vec3.h, base/mat3.h, every shared header with a formula
// Read:    docs/method-paper/fluidity-method.typ <sec-conventions>

#pragma once

#include "fluidity/base/cpu_and_gpu.h"

// Each function below has two paths. The `__CUDACC__` path is taken by the CUDA compiler in both of its passes.
// It calls the CUDA math library (`::floor`, `::sqrt` and the others), which the CUDA compiler declares in every
// file: the GPU pass gets the GPU version and the CPU pass gets the C library's. The other path is taken by every
// other compiler and calls the <cmath> functions in namespace std.
#if !defined(__CUDACC__)
#include <cmath>
#endif
#include <limits>

namespace fluidity::math {

// Positive infinity, the double above every finite double. A constant evaluated on the CPU at compile time, so GPU
// code reads it as a plain double; GPU code could not call std::numeric_limits<double>::infinity() itself.
// Example: Min(kPositiveInfinity, 2) = 2; IsFinite(kPositiveInfinity) is false.
inline constexpr double kPositiveInfinity = std::numeric_limits<double>::infinity();

// Computes the largest whole number that is not above the value.
//
// Method: same bits on CPU and GPU, except for a NaN. The result is exact, so there is no rounding to differ.
// Example: Floor(2.75) = 2. Floor(-0.5) = -1, not 0.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Floor(double value) noexcept {
#if defined(__CUDACC__)
  return ::floor(value);
#else
  return std::floor(value);
#endif
}

// Computes the absolute value: the value with its sign made positive.
//
// Method: same bits on CPU and GPU. Both processors run the same comparisons, which round nothing.
//   Both zeros give +0. A NaN comes back unchanged.
// Example: Abs(-2.5) = 2.5. Abs(-0.0) = +0.0.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU constexpr double Abs(double value) noexcept {
  if (value < 0) {
    return -value;
  }
  // Turns -0 into +0. A comparison cannot tell the two zeros apart, so the zero is written out.
  if (value == 0) {
    return 0.0;
  }
  return value;
}

// Computes the square root of a value that is zero or more.
//
// Method: same bits on CPU and GPU, except for a NaN. IEEE 754 defines the square root as correctly rounded, and
//   both processors follow it for doubles: the GPU always rounds a double square root to nearest, whatever the
//   compiler flags.
// Example: Sqrt(9) = 3 exactly. Sqrt(2) = 1.4142135623730951, the double nearest the true root.
//   A negative value gives NaN, whose bits may differ between the processors.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Sqrt(double value) noexcept {
#if defined(__CUDACC__)
  return ::sqrt(value);
#else
  return std::sqrt(value);
#endif
}

// Computes the cube root. Unlike Pow(value, 1.0 / 3), it accepts a negative value.
//
// Method: may differ in the last bits between CPU and GPU.
// Example: Cbrt(27) is 3 and Cbrt(-8) is -2, each within a few rounding units.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Cbrt(double value) noexcept {
#if defined(__CUDACC__)
  return ::cbrt(value);
#else
  return std::cbrt(value);
#endif
}

// Computes e raised to the value, where e = 2.718281828459045 is the base of the natural logarithm.
//
// Method: may differ in the last bits between CPU and GPU.
// Example: Exp(0) = 1. Exp(1) is e within a few rounding units.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Exp(double value) noexcept {
#if defined(__CUDACC__)
  return ::exp(value);
#else
  return std::exp(value);
#endif
}

// Computes exp(value) - 1 without losing the small part when the value is near zero.
//
// Method: may differ in the last bits between CPU and GPU.
// Example: Expm1(1e-10) = 1.00000000005e-10. Exp(1e-10) - 1 would keep only about seven of these digits,
//   because Exp(1e-10) is so close to 1 that most digits of the difference are lost in the rounding of the 1.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Expm1(double value) noexcept {
#if defined(__CUDACC__)
  return ::expm1(value);
#else
  return std::expm1(value);
#endif
}

// Computes the natural logarithm, the inverse of Exp, of a positive value.
//
// Method: may differ in the last bits between CPU and GPU.
// Example: Log(1) = 0. Log(e) is 1 within a few rounding units. Log(0) is minus infinity; a negative value
//   gives NaN.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Log(double value) noexcept {
#if defined(__CUDACC__)
  return ::log(value);
#else
  return std::log(value);
#endif
}

// Computes base raised to the exponent.
//
// Method: may differ in the last bits between CPU and GPU.
// Example: Pow(2, 10) is 1024 and Pow(16, 0.25) is 2, each within a few rounding units.
//   A negative base with an exponent that is not a whole number gives NaN.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Pow(double base, double exponent) noexcept {
#if defined(__CUDACC__)
  return ::pow(base, exponent);
#else
  return std::pow(base, exponent);
#endif
}

// Computes the sine of an angle.
//
// Method: may differ in the last bits between CPU and GPU.
// Units: the angle is in radians.
// Example: Sin(pi / 6) is 0.5 within a few rounding units.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Sin(double angle) noexcept {
#if defined(__CUDACC__)
  return ::sin(angle);
#else
  return std::sin(angle);
#endif
}

// Computes the cosine of an angle.
//
// Method: may differ in the last bits between CPU and GPU.
// Units: the angle is in radians.
// Example: Cos(pi / 3) is 0.5 within a few rounding units.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Cos(double angle) noexcept {
#if defined(__CUDACC__)
  return ::cos(angle);
#else
  return std::cos(angle);
#endif
}

// Computes the smaller of two values. A NaN loses: with one NaN the other value is returned, and with two NaNs
// the result is NaN. This is what fmin does on both processors.
//
// Method: same bits on CPU and GPU. Both processors run the same comparisons, which round nothing.
// Ties: +0 and -0 compare equal, so Min(+0.0, -0.0) returns the first, +0.0.
// Example: Min(3, -1) = -1. Min(NaN, 2) = 2.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU constexpr double Min(double first, double second) noexcept {
  // Only a NaN compares unequal to itself. Both NaN tests come before the ordering test because, with the second
  // value going straight to the ordering test, MSVC's constant evaluator returned the NaN for Min(2.0, NaN).
  const bool first_is_nan = !(first == first);
  if (first_is_nan) {
    return second;
  }
  const bool second_is_nan = !(second == second);
  if (second_is_nan) {
    return first;
  }
  if (second < first) {
    return second;
  }
  return first;
}

// Computes the larger of two values. A NaN loses: with one NaN the other value is returned, and with two NaNs
// the result is NaN. This is what fmax does on both processors.
//
// Method: same bits on CPU and GPU. Both processors run the same comparisons, which round nothing.
// Ties: +0 and -0 compare equal, so Max(+0.0, -0.0) returns the first, +0.0.
// Example: Max(3, -1) = 3. Max(NaN, 2) = 2.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU constexpr double Max(double first, double second) noexcept {
  // Only a NaN compares unequal to itself. The tests run in the same order as in Min, which MSVC's constant evaluator
  // needs: see the comment there.
  const bool first_is_nan = !(first == first);
  if (first_is_nan) {
    return second;
  }
  const bool second_is_nan = !(second == second);
  if (second_is_nan) {
    return first;
  }
  if (second > first) {
    return second;
  }
  return first;
}

// Says whether a value is an ordinary number: false for plus or minus infinity and for NaN, true otherwise.
//
// Method: same bits on CPU and GPU. The answer depends only on the bits of the value.
// Example: IsFinite(1e308) is true. IsFinite(1e308 * 10) is false, because the product overflows to infinity.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline bool IsFinite(double value) noexcept {
#if defined(__CUDACC__)
  return ::isfinite(value);
#else
  return std::isfinite(value);
#endif
}

// Says whether a value is NaN.
//
// Method: same answer on CPU and GPU. Only a NaN compares unequal to itself, as Min and Max test it, so the answer
//   needs no library call and is usable in a constant expression.
// Example: IsNan(0.0 / 0.0) is true. IsNan(kPositiveInfinity) is false.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU constexpr bool IsNan(double value) noexcept {
  return !(value == value);
}

}  // namespace fluidity::math
