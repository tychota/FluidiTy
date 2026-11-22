#include "fluidity/base/check_can_run_on_math.h"

#include <gtest/gtest.h>

#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"

namespace fluidity {
namespace {

// The expected bits below were computed apart from this code, with Python floats, which round each add, subtract,
// multiply and square root separately, as the numerical flags require.
TEST(ComputeMathCheckResult, FirstInput_HasPinnedBits) {
  const MathCheckResult result = ComputeMathCheckResult(BuildMathCheckInputs(0));
  EXPECT_EQ(ToBits(result.cross.x), ToBits(0x1.d23ae165cd6e8p-1));
  EXPECT_EQ(ToBits(result.cross.y), ToBits(-0x1.dbef740b9add0p-1));
  EXPECT_EQ(ToBits(result.cross.z), ToBits(-0x1.20845da329180p-5));
  EXPECT_EQ(ToBits(result.dot), ToBits(0x1.39c6a26df9f00p+3));
  EXPECT_EQ(ToBits(result.rotated_first.x), ToBits(-0x1.40d6e4999999cp-2));
  EXPECT_EQ(ToBits(result.determinant), ToBits(-0x1.71fa865cc52b4p+1));
  EXPECT_EQ(ToBits(result.square_root_of_dot), ToBits(0x1.90d0d547e58d8p+1));
  EXPECT_EQ(ToBits(result.floor_of_scaled_dot), ToBits(0x1.39c6a26df9e00p+43));
  EXPECT_EQ(ToBits(result.length_of_difference), ToBits(0x1.f7a96189ffe31p-2));
}

TEST(ComputeMathCheckResult, ScaledInput_HasPinnedSubnormalBits) {
  const MathCheckResult result = ComputeMathCheckResult(BuildMathCheckInputs(3));
  EXPECT_EQ(ToBits(result.cross.x), ToBits(0x0.000178f7436c4p-1022));
  EXPECT_EQ(ToBits(result.dot), ToBits(0x0.0000c2f83f86fp-1022));
  EXPECT_EQ(ToBits(result.square_root_of_dot), ToBits(0x1.bed22c12333e8p-520));
  EXPECT_EQ(ToBits(result.length_of_difference), ToBits(0x1.ae3e093bd25d2p-519));
}

}  // namespace
}  // namespace fluidity
