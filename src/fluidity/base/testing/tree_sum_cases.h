// Test-only: the helpers that the unit tests and the functional tests of tree_sum share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ComputePaddedTreeSum, BuildMixedValuesWithNegativeZeros
// Uses:    the headers it includes
// Used by: src/fluidity/base/tree_sum.test.cc and tests/base/tree_sum_on_every_count.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cstddef>
#include <span>
#include <vector>

namespace fluidity {

// The padded tree written the plain way, to compare with math::TreeSum: copies the values into an array padded
// with +0.0 to the next power of two, then adds neighbours in pairs, layer after layer, in place.
inline double ComputePaddedTreeSum(std::span<const double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::size_t place_count = 1;
  while (place_count < values.size()) {
    place_count *= 2;
  }
  std::vector<double> sides(place_count, 0.0);
  for (std::size_t index = 0; index < values.size(); ++index) {
    sides[index] = values[index];
  }
  for (std::size_t side_count = place_count; side_count > 1; side_count /= 2) {
    for (std::size_t pair = 0; pair < side_count / 2; ++pair) {
      sides[pair] = sides[2 * pair] + sides[2 * pair + 1];
    }
  }
  return sides[0];
}

// The mixed values of the test support, with every tenth value -0.0, so the padding rule is exercised too.
inline std::vector<double> BuildMixedValuesWithNegativeZeros(std::size_t count) {
  std::vector<double> values = BuildMixedValues(count);
  for (std::size_t index = 9; index < count; index += 10) {
    values[index] = -0.0;
  }
  return values;
}

}  // namespace fluidity
