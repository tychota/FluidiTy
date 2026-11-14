// Functions that the tests of base/ share, so no test file keeps its own copy.
//
// Purpose: give every test one way to compare doubles bit for bit, one form of the method paper's tolerance and one
//   check for "same bits, or both NaN".
//
// Idea: a test that demands equal bits compares the 64 bits of two doubles (ToBits), because == treats +0 and -0
//   as equal and a NaN as unequal to itself. CheckSameBitsOrBothNaN accepts two NaNs whatever their bits, since IEEE
//   754 leaves the bits of a NaN open. CheckWithinTolerance is the comparison of a CPU and a GPU result that the
//   method paper states, with the CPU result as the reference: |reference - compared| <= absolute floor + relative
//   term * |reference|.
//   Example: CheckWithinTolerance(1.0, 1.0 + 2e-16, {.absolute_floor = 1e-15, .relative_term = 1e-15}) passes.
//
// Owns:    ToBits, Tolerance, CheckWithinTolerance, CheckSameBitsOrBothNaN
// Uses:    CPU-only guard (base/cpu_only.h), GoogleTest
// Used by: the tests of base/ (the library fluidity_base_test_support)
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace fluidity {

// Converts a double to its 64 bits, so +0.0 and -0.0 differ and every bit is compared.
std::uint64_t ToBits(double value);

// The allowed difference between a reference result (the CPU's) and a compared one (the GPU's): absolute_floor +
// relative_term * |reference|.
struct Tolerance {
  double absolute_floor = 0.0;  // Same unit as the values; the allowed difference near zero.
  double relative_term = 0.0;  // No unit; the allowed difference per unit of |reference|.
};

// Checks |reference - compared| <= tolerance.absolute_floor + tolerance.relative_term * |reference| (method paper,
// <sec-run-on>). Two equal infinities of the same sign pass; a NaN on either side fails. The failure message gives
// both values and the difference.
// Example: reference 2.0, compared 2.0 + 1e-15, {1e-15, 1e-15}: allowed 3e-15, passes.
::testing::AssertionResult CheckWithinTolerance(double reference, double compared, Tolerance tolerance);

// Checks that two doubles have the same 64 bits, or are both NaN (IEEE 754 leaves the bits of a NaN open).
// Example: (+0.0, -0.0) fails; (NaN, -NaN) passes; (1.0, 1.0) passes.
::testing::AssertionResult CheckSameBitsOrBothNaN(double expected, double actual);

}  // namespace fluidity
