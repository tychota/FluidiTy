#include <gtest/gtest.h>

#include <random>

#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/weight_curve.h"

namespace fluidity::grid {
namespace {

TEST(ComputeWeightCurve, AnyPosition_ThreeNeighboursSumToOne) {
  // A fixed seed: the test checks the same offsets on every run.
  std::mt19937_64 generator(20261114);  // NOLINT(bugprone-random-generator-seed)
  for (int draw = 0; draw < 1000; ++draw) {
    const double offset = ToUnitDouble(generator()) - 0.5;
    const WeightCurveValue below = ComputeWeightCurve(offset - 1.0);
    const WeightCurveValue middle = ComputeWeightCurve(offset);
    const WeightCurveValue above = ComputeWeightCurve(offset + 1.0);
    const double weight_sum = below.weight + middle.weight + above.weight;
    const double slope_sum = below.slope + middle.slope + above.slope;
    EXPECT_NEAR(weight_sum, 1.0, 4e-16) << offset;
    EXPECT_NEAR(slope_sum, 0.0, 4e-16) << offset;
  }
}

TEST(ComputeWeightCurve, AnyDistance_IsNeverNegative) {
  for (int step = -400; step <= 400; ++step) {
    EXPECT_GE(ComputeWeightCurve(0.005 * step).weight, 0.0) << step;
  }
}

}  // namespace
}  // namespace fluidity::grid
