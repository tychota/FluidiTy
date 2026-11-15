// Values and checks that the contact tests share.
//
// Purpose: give every contact test one contact tolerance, one way to compare two vectors axis by axis, and one
//   generator of repeatable test cases.
//
// Idea: an analytic contact test passes a fixed contact tolerance, so its expected values can be written by hand,
//   including the rest gap: a particle in contact rests one tolerance outside the surface. ExpectVec3Near fails the
//   calling test when any axis differs by more than the tolerance it is given.
//   Example: ExpectVec3Near({1, 2, 3}, {1, 2, 3 + 1e-12}, 1e-9) passes.
//   A generated test set draws its numbers from one CaseGenerator in a fixed order, so every run and every platform
//   tests the same cases.
//
// Owns:    kTestContactTolerance, ExpectVec3Near, CaseGenerator
// Uses:    Vec3 (base/vec3.h), CPU-only guard (base/cpu_only.h), GoogleTest
// Used by: the tests of contact/
// Read:    docs/method-paper/fluidity-method.typ <sec-contact-tolerance>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <cstdint>

#include "fluidity/base/vec3.h"

namespace fluidity::contact {

// m. The contact tolerance of every analytic contact test, unless the test is about the tolerance itself. It is
// small enough to stay below any length a test checks and large enough to sit far above rounding at unit scale.
inline constexpr double kTestContactTolerance = 0x1p-30;

// Fails the calling test when an axis of actual differs from the same axis of expected by more than tolerance.
//
// Units: the unit of the two vectors; tolerance is in the same unit.
// Example: ExpectVec3Near({1, 0, 0}, {1, 0, 1e-3}, 1e-6) fails on axis z.
// Read: docs/method-paper/fluidity-method.typ <sec-contact-tolerance>
inline void ExpectVec3Near(Vec3 actual, Vec3 expected, double tolerance) {
  EXPECT_NEAR(actual.x, expected.x, tolerance) << "axis x";
  EXPECT_NEAR(actual.y, expected.y, tolerance) << "axis y";
  EXPECT_NEAR(actual.z, expected.z, tolerance) << "axis z";
}

// Repeatable numbers for generated test cases: integer steps only, so every compiler and platform draws the same
// sequence. Each test set starts a new generator and states the order of its draws.
//
// Method: Next() sets state = state * 6364136223846793005 + 1442695040888963407, wrapping at 2^64, and returns the top
//   53 bits of state as a double in [0, 1): (state >> 11) * 2^-53, exact. NextSigned() is 2 * Next() - 1, in [-1, 1).
// Example: two generators built the same way return the same numbers in the same order.
class CaseGenerator {
 public:
  double Next() {
    state_ = state_ * 6364136223846793005U + 1442695040888963407U;
    return static_cast<double>(state_ >> 11U) * 0x1p-53;
  }
  double NextSigned() {
    return 2 * Next() - 1;
  }

 private:
  std::uint64_t state_ = 0x2545F4914F6CDD1DU;
};

}  // namespace fluidity::contact
