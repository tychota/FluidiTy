#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"

namespace fluidity {
namespace {

TEST(WriteBuffer, ThenReadBuffer_ReturnsTheSameValues) {
  std::vector<double> written(1000);
  for (std::size_t index = 0; index < written.size(); ++index) {
    // Values with every bit pattern kind that matters: negative zero, a subnormal and ordinary numbers.
    written[index] = (static_cast<double>(index) - 500.0) / 7.0;
  }
  written[0] = -0.0;
  written[1] = std::numeric_limits<double>::denorm_min();
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, written);
  EXPECT_EQ(ToBits(ReadAll(buffer)), ToBits(written));
}

}  // namespace
}  // namespace fluidity
