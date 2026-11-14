#include "fluidity/base/test_support.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <ios>

namespace fluidity {

std::uint64_t ToBits(double value) {
  return std::bit_cast<std::uint64_t>(value);
}

::testing::AssertionResult CheckWithinTolerance(double reference, double compared, Tolerance tolerance) {
  const double difference = std::abs(reference - compared);
  const double allowed = tolerance.absolute_floor + tolerance.relative_term * std::abs(reference);
  // An infinity on either side agrees only with the same infinity. The tolerance cannot decide: two equal
  // infinities differ by NaN, and an infinite reference makes the allowed difference infinite.
  const bool either_is_infinite = std::isinf(reference) || std::isinf(compared);
  if (either_is_infinite) {
    if (reference == compared) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << std::hexfloat << "reference " << reference << " and compared " << compared
                                         << " are not the same infinity";
  }
  // Written so that a NaN difference fails: every comparison with NaN is false.
  if (difference <= allowed) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << std::hexfloat << "reference " << reference << " and compared " << compared
                                       << " differ by " << difference << ", more than the allowed " << allowed;
}

::testing::AssertionResult CheckSameBitsOrBothNaN(double expected, double actual) {
  if (std::isnan(expected) && std::isnan(actual)) {
    return ::testing::AssertionSuccess();
  }
  if (ToBits(expected) == ToBits(actual)) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << std::hexfloat << "expected " << expected << " but got " << actual
                                       << ", which has other bits";
}

}  // namespace fluidity
