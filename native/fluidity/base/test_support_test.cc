#include "fluidity/base/test_support.h"

#include <gtest/gtest.h>

#include <limits>

namespace fluidity {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr Tolerance kTolerance = {.absolute_floor = 1e-15, .relative_term = 1e-15};

TEST(ToBits, TwoZeros_Differ) {
  EXPECT_NE(ToBits(0.0), ToBits(-0.0));
}

TEST(ToBits, One_IsTheIeeeBits) {
  EXPECT_EQ(ToBits(1.0), 0x3FF0000000000000ULL);
}

TEST(CheckWithinTolerance, DifferenceBelowTheAllowed_Passes) {
  EXPECT_TRUE(CheckWithinTolerance(2.0, 2.0 + 2e-15, kTolerance));
}

TEST(CheckWithinTolerance, DifferenceAboveTheAllowed_Fails) {
  EXPECT_FALSE(CheckWithinTolerance(2.0, 2.0 + 1e-14, kTolerance));
}

TEST(CheckWithinTolerance, ValueNearZero_UsesTheAbsoluteFloor) {
  EXPECT_TRUE(CheckWithinTolerance(0.0, 5e-16, kTolerance));
}

TEST(CheckWithinTolerance, EqualInfinities_Pass) {
  EXPECT_TRUE(
      CheckWithinTolerance(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), kTolerance)
  );
}

TEST(CheckWithinTolerance, OppositeInfinities_Fail) {
  EXPECT_FALSE(CheckWithinTolerance(
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      kTolerance
  ));
}

TEST(CheckWithinTolerance, NaN_Fails) {
  EXPECT_FALSE(CheckWithinTolerance(kNaN, kNaN, kTolerance));
}

TEST(CheckSameBitsOrBothNaN, EqualValues_Pass) {
  EXPECT_TRUE(CheckSameBitsOrBothNaN(1.5, 1.5));
}

TEST(CheckSameBitsOrBothNaN, TwoZeros_Fail) {
  EXPECT_FALSE(CheckSameBitsOrBothNaN(0.0, -0.0));
}

TEST(CheckSameBitsOrBothNaN, TwoNaNsWithOtherBits_Pass) {
  EXPECT_TRUE(CheckSameBitsOrBothNaN(kNaN, -kNaN));
}

TEST(CheckSameBitsOrBothNaN, NaNAgainstANumber_Fails) {
  EXPECT_FALSE(CheckSameBitsOrBothNaN(kNaN, 1.0));
}

}  // namespace
}  // namespace fluidity
