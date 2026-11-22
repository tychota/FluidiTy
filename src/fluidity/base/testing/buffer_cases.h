// Test-only: the helpers that the unit tests and the functional tests of buffer on a GPU share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    BuildTestValues
// Uses:    the headers it includes
// Used by: src/fluidity/base/buffer.gpu.test.cc and tests/base/buffer_on_gpu.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cstddef>
#include <vector>

namespace fluidity {

// Values with a negative zero, a subnormal and ordinary numbers, so a round trip is checked on every bit kind.
inline std::vector<double> BuildTestValues(std::size_t count) {
  std::vector<double> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = (static_cast<double>(index) - 500.0) / 7.0;
  }
  values[0] = -0.0;
  values[1] = 4.9406564584124654e-324;
  return values;
}

}  // namespace fluidity
