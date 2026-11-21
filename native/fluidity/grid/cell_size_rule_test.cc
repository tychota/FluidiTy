#include "fluidity/grid/cell_size_rule.h"

#include <gtest/gtest.h>

#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

#include "fluidity/base/scalar_math.h"
#include "fluidity/grid/cell_size_rule_test_cases.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {
namespace {

// The frame of most tests: biggest cells 1 m wide and 5 size levels, widths 1, 0.5, 0.25, 0.125 and 0.0625 m.
constexpr GridFrame kFrame = {.origin = {}, .biggest_cell_width = 1.0, .size_level_count = 5};

// Default settings with the smallest size level 4, so w_min = 0.0625 m.
constexpr CellSizeSettings kSettings = {.smallest_size_level = SizeLevel{4}};

constexpr double kReferenceSpeed = 1.0;  // m/s.

// The wanted cell width of `facts` in kFrame with kSettings and kReferenceSpeed.
double ComputeWidthInTestFrame(const CellSizeFacts& facts) {
  return ComputeWantedCellWidth(facts, kReferenceSpeed, ComputeCellWidthsBySizeLevel(kFrame), kSettings);
}

// The numbers the test-local copy of the rule reads, written as literals.
struct TestRuleNumbers {
  double smallest_width;  // m.
  double biggest_width;  // m.
  double surface_reach_in_cell_widths;  // No unit.
  double solid_reach_in_cell_widths;  // No unit.
  double shear_width_factor;  // No unit.
};

// kFrame with kSettings: every number a power of two.
constexpr TestRuleNumbers kDyadicNumbers = {
    .smallest_width = 0.0625,
    .biggest_width = 1.0,
    .surface_reach_in_cell_widths = 8.0,
    .solid_reach_in_cell_widths = 2.0,
    .shear_width_factor = 0.25,
};

// kOddFrame with kOddSettings: no number a power of two, so two groupings of one product or quotient differ in bits.
constexpr TestRuleNumbers kOddNumbers = {
    .smallest_width = 0.01875,
    .biggest_width = 0.3,
    .surface_reach_in_cell_widths = 3.0,
    .solid_reach_in_cell_widths = 5.0,
    .shear_width_factor = 0.3,
};
constexpr GridFrame kOddFrame = {.origin = {}, .biggest_cell_width = 0.3, .size_level_count = 5};
constexpr CellSizeSettings kOddSettings = {
    .smallest_size_level = SizeLevel{4},
    .surface_reach_in_cell_widths = 3.0,
    .solid_reach_in_cell_widths = 5.0,
    .shear_width_factor = 0.3,
};

// The rule of <eq:wanted-cell-width> with the product by the cell size factor left out, restated without the
// header's code: the widths and the settings are literals (TestRuleNumbers), Max and Min are std::fmax and std::fmin,
// and infinity comes from std::numeric_limits. It keeps the order of operations of the method paper and nothing else
// of the header. With kOddNumbers, a header that groups a product or a quotient differently gives other bits for
// some fact sets; with either numbers, so does one that compares at the reach the other way or clamps in another order.
double ComputeTestRuleWithoutFactor(const CellSizeFacts& facts, double reference_speed, TestRuleNumbers numbers) {
  const double surface_reach = numbers.surface_reach_in_cell_widths * numbers.smallest_width;
  if (facts.touches_air_pocket || facts.wants_smallest_cells) {
    return numbers.smallest_width;
  }
  if (facts.depth_below_surface < surface_reach) {
    return numbers.smallest_width;
  }
  const double solid_reach = numbers.solid_reach_in_cell_widths * numbers.smallest_width;
  if (facts.distance_to_solid < solid_reach) {
    return numbers.smallest_width;
  }
  const double depth_quotient = facts.depth_below_surface / surface_reach;
  const double depth_width = numbers.smallest_width * std::fmax(1.0, depth_quotient);
  const double distance_quotient = facts.distance_to_solid / solid_reach;
  const double solid_width = numbers.smallest_width * std::fmax(1.0, distance_quotient);
  double shear_width = std::numeric_limits<double>::infinity();
  if (facts.velocity_change_size > 0.0) {
    const double shear_speed = numbers.shear_width_factor * reference_speed;
    shear_width = shear_speed / facts.velocity_change_size;
  }
  const double smallest_term = std::fmin(std::fmin(depth_width, solid_width), shear_width);
  return std::fmax(numbers.smallest_width, std::fmin(smallest_term, numbers.biggest_width));
}

// Finds the first of the 10,000 fact sets, with factor 1, whose wanted cell width in `frame` with `settings` differs
// in bits from the test-local copy of the rule with `numbers`; -1 when none does.
int FindFirstFactSetWithOtherBits(GridFrame frame, const CellSizeSettings& settings, TestRuleNumbers numbers) {
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(frame);
  for (int fact_set_index = 0; fact_set_index < kTestFactSetCount; ++fact_set_index) {
    const CellSizeFacts facts = BuildTestCellSizeFacts(fact_set_index, 1.0);
    const double reference_speed = ComputeTestReferenceSpeed(fact_set_index);
    const double wanted_width = ComputeWantedCellWidth(facts, reference_speed, widths, settings);
    const double expected = ComputeTestRuleWithoutFactor(facts, reference_speed, numbers);
    if (std::bit_cast<std::uint64_t>(wanted_width) != std::bit_cast<std::uint64_t>(expected)) {
      return fact_set_index;
    }
  }
  return -1;
}

// How many of the 10,000 fact sets, with factor 1, get each kind of width in kFrame with kSettings.
struct WidthKindCounts {
  int smallest = 0;
  int between = 0;
  int biggest = 0;
};

WidthKindCounts ComputeWidthKindCounts() {
  WidthKindCounts counts;
  for (int fact_set_index = 0; fact_set_index < kTestFactSetCount; ++fact_set_index) {
    const CellSizeFacts facts = BuildTestCellSizeFacts(fact_set_index, 1.0);
    const double wanted_width = ComputeWantedCellWidth(
        facts,
        ComputeTestReferenceSpeed(fact_set_index),
        ComputeCellWidthsBySizeLevel(kFrame),
        kSettings
    );
    if (wanted_width == 0.0625) {
      ++counts.smallest;
    } else if (wanted_width == 1.0) {
      ++counts.biggest;
    } else {
      ++counts.between;
    }
  }
  return counts;
}

TEST(ComputeCellWidthsBySizeLevel, EverySizeLevel_EqualsComputeCellWidth) {
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kFrame);
  EXPECT_EQ(widths.size_level_count, 5);
  for (int size_level = 0; size_level < kMaxSizeLevels; ++size_level) {
    const double expected = size_level < 5 ? ComputeCellWidth(kFrame, static_cast<SizeLevel>(size_level)) : 0.0;
    EXPECT_EQ(std::bit_cast<std::uint64_t>(widths.width[size_level]), std::bit_cast<std::uint64_t>(expected))
        << size_level;
  }
  EXPECT_EQ(widths.width[4], 0.0625);
}

TEST(ComputeWantedCellWidth, SurfaceCell_IsSmallest) {
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 0.0}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 0.0, .distance_to_solid = 10.0}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 0.0, .cell_size_factor = 4.0}), 0.0625);
}

TEST(ComputeWantedCellWidth, UniformTranslation_NoShearSplit) {
  const CellSizeFacts facts = {.velocity_change_size = 0.0};
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kFrame);
  const double wanted_width = ComputeWantedCellWidth(facts, 5.0, widths, kSettings);
  EXPECT_EQ(wanted_width, 1.0);
  EXPECT_EQ(ComputeWantedSizeLevel(wanted_width, widths), SizeLevel{0});
}

TEST(ComputeWantedCellWidth, NothingWithinReach_IsBiggestWidth) {
  EXPECT_EQ(ComputeWidthInTestFrame({.velocity_change_size = 1e-300}), 1.0);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = DBL_MAX, .velocity_change_size = 1e-300}), 1.0);
}

TEST(ComputeWantedCellWidth, DepthBeyondReach_GrowsLinearly) {
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0}), 0.125);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 0.25}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 100.0}), 1.0);
}

TEST(ComputeWantedCellWidth, SolidBeyondReach_GrowsLinearly) {
  EXPECT_EQ(ComputeWidthInTestFrame({.distance_to_solid = 0.5}), 0.25);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0, .distance_to_solid = 0.5}), 0.125);
}

TEST(ComputeWantedCellWidth, ShearTerm_IsShearFactorTimesSpeedOverVelocityChange) {
  constexpr GridFrame kHalfMetreFrame = {.origin = {}, .biggest_cell_width = 0.5, .size_level_count = 6};
  constexpr CellSizeSettings kSmallestFiveSettings = {.smallest_size_level = SizeLevel{5}};
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kHalfMetreFrame);
  EXPECT_EQ(widths.width[5], 0.015625);
  const CellSizeFacts facts = {.velocity_change_size = math::Sqrt(2.0) * 10.0};
  const double wanted_width = ComputeWantedCellWidth(facts, kReferenceSpeed, widths, kSmallestFiveSettings);
  // 0.25 * 1 / (10 sqrt(2)) = sqrt(2) / 80.
  EXPECT_NEAR(wanted_width, 0.0176776695296637, 1e-15 * 0.0176776695296637);
  EXPECT_EQ(ComputeWantedSizeLevel(wanted_width, widths), SizeLevel{5});
}

TEST(ComputeWantedCellWidth, TouchesAirPocketOrWantsSmallestCells_IsSmallest) {
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 100.0, .touches_air_pocket = true}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 100.0, .wants_smallest_cells = true}), 0.0625);
}

TEST(ComputeWantedCellWidth, MergeTestFlags_ChangeNothing) {
  EXPECT_EQ(
      ComputeWidthInTestFrame({.depth_below_surface = 1.0, .holds_particles_that_keep_their_count = true}),
      0.125
  );
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0, .touches_air_gas_or_solid = true}), 0.125);
}

TEST(ComputeWantedCellWidth, FactorOne_SameBitsAsNoFactor) {
  ASSERT_EQ(ComputeCellWidthsBySizeLevel(kOddFrame).width[4], kOddNumbers.smallest_width);
  EXPECT_EQ(FindFirstFactSetWithOtherBits(kFrame, kSettings, kDyadicNumbers), -1);
  EXPECT_EQ(FindFirstFactSetWithOtherBits(kOddFrame, kOddSettings, kOddNumbers), -1);
  // In kFrame the fact sets reach the smallest width (8,601 times), widths between (1,396) and the biggest (3), so
  // every part of the rule is read.
  const WidthKindCounts counts = ComputeWidthKindCounts();
  EXPECT_EQ(counts.smallest, 8601);
  EXPECT_EQ(counts.between, 1396);
  EXPECT_EQ(counts.biggest, 3);
}

TEST(ComputeWantedCellWidth, FactorHalf_HalvesDeepWidth) {
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0, .cell_size_factor = 0.5}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.distance_to_solid = 0.5, .cell_size_factor = 0.5}), 0.125);
}

TEST(ComputeWantedCellWidth, FactorTwo_DoublesDeepWidth) {
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0, .cell_size_factor = 2.0}), 0.25);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 100.0, .cell_size_factor = 2.0}), 1.0);
}

TEST(ComputeWantedCellWidth, FactorTwoNearSurface_IsSmallest) {
  // 0.25 m is 4 smallest widths, within the 8; 0.5 m is exactly 8, not within.
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 0.25, .cell_size_factor = 2.0}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 0.5, .cell_size_factor = 2.0}), 0.125);
}

TEST(ComputeWantedCellWidth, NegativeDepthOrDistance_IsSmallest) {
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = -0.03}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.distance_to_solid = -0.03}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = -0.03, .cell_size_factor = 4.0}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.distance_to_solid = -0.03, .cell_size_factor = 4.0}), 0.0625);
}

TEST(ComputeWantedCellWidth, JustBelowSurfaceReachInOddFrame_IsSmallest) {
  // 3 * 0.01875 m rounds, so the comparison at the surface reach is tested where its rounding matters.
  const double surface_reach = 3.0 * kOddNumbers.smallest_width;
  const CellSizeFacts facts = {.depth_below_surface = std::nextafter(surface_reach, 0.0), .cell_size_factor = 2.0};
  const double wanted_width =
      ComputeWantedCellWidth(facts, kReferenceSpeed, ComputeCellWidthsBySizeLevel(kOddFrame), kOddSettings);
  EXPECT_EQ(wanted_width, kOddNumbers.smallest_width);
}

TEST(ComputeWantedCellWidth, DepthAtSurfaceReachNotAPowerOfTwo_IsScaled) {
  // Biggest cells 3 m wide and 7 size levels: w_min = 3/64 m. A surface reach of 0.7 smallest widths makes the reach
  // 0.7 * 3/64 m, a rounded product. A depth equal to it is not within the reach, so the factor of 2 scales w_min;
  // comparing depth / w_min with 0.7 instead would round to "within" here.
  constexpr GridFrame kThreeMetreFrame = {.origin = {}, .biggest_cell_width = 3.0, .size_level_count = 7};
  constexpr CellSizeSettings kShortReachSettings = {
      .smallest_size_level = SizeLevel{6},
      .surface_reach_in_cell_widths = 0.7,
  };
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kThreeMetreFrame);
  ASSERT_EQ(widths.width[6], 0.046875);
  const CellSizeFacts facts = {.depth_below_surface = 0.7 * 0.046875, .cell_size_factor = 2.0};
  EXPECT_EQ(ComputeWantedCellWidth(facts, kReferenceSpeed, widths, kShortReachSettings), 0.09375);
}

TEST(ComputeWantedCellWidth, DistanceAtSolidReachNotAPowerOfTwo_IsScaled) {
  // The solid twin of DepthAtSurfaceReachNotAPowerOfTwo_IsScaled: w_min = 3/64 m and a solid reach of 0.7 smallest
  // widths, a rounded product. A distance equal to it is not within the solid reach, so the factor of 2 scales
  // w_min; comparing distance / w_min with 0.7 instead would round to "within" here.
  constexpr GridFrame kThreeMetreFrame = {.origin = {}, .biggest_cell_width = 3.0, .size_level_count = 7};
  constexpr CellSizeSettings kShortSolidReachSettings = {
      .smallest_size_level = SizeLevel{6},
      .solid_reach_in_cell_widths = 0.7,
  };
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kThreeMetreFrame);
  ASSERT_EQ(widths.width[6], 0.046875);
  const CellSizeFacts facts = {.distance_to_solid = 0.7 * 0.046875, .cell_size_factor = 2.0};
  EXPECT_EQ(ComputeWantedCellWidth(facts, kReferenceSpeed, widths, kShortSolidReachSettings), 0.09375);
}

TEST(ComputeWantedCellWidth, ConstantExpression_GivesTheRuntimeWidth) {
  // Depth 1 m in kFrame: 0.125 m, computed by the compiler.
  static_assert(
      ComputeWantedCellWidth(
          {.depth_below_surface = 1.0},
          kReferenceSpeed,
          ComputeCellWidthsBySizeLevel(kFrame),
          kSettings
      )
      == 0.125
  );
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0}), 0.125);
}

TEST(ComputeWantedCellWidth, InfiniteVelocityChangeSize_IsSmallest) {
  EXPECT_EQ(ComputeWidthInTestFrame({.velocity_change_size = math::kPositiveInfinity}), 0.0625);
}

TEST(ComputeWantedCellWidth, InfiniteReferenceSpeed_GivesNoShearLimit) {
  const CellSizeFacts facts = {.velocity_change_size = 10.0};
  const double wanted_width =
      ComputeWantedCellWidth(facts, math::kPositiveInfinity, ComputeCellWidthsBySizeLevel(kFrame), kSettings);
  EXPECT_EQ(wanted_width, 1.0);
}

TEST(ComputeWantedCellWidth, VelocityChangeSizeZeroOrBelow_GivesNoShearTerm) {
  EXPECT_EQ(ComputeWidthInTestFrame({.velocity_change_size = -0.0}), 1.0);
  EXPECT_EQ(ComputeWidthInTestFrame({.velocity_change_size = -1.0}), 1.0);
}

TEST(ComputeWantedCellWidth, SolidWithinReach_IsSmallestWhateverTheFactor) {
  // The solid reach is 2 * 0.0625 = 0.125 m.
  EXPECT_EQ(ComputeWidthInTestFrame({.distance_to_solid = 0.0, .cell_size_factor = 4.0}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.distance_to_solid = 0.10, .cell_size_factor = 4.0}), 0.0625);
}

TEST(ComputeWantedCellWidth, SolidAtOrBeyondReach_IsScaled) {
  // 0.125 m is exactly the solid reach, not within it: 0.0625 * 1 * 2. 0.5 m: 0.0625 * 4 * 4, clamped to 1.
  EXPECT_EQ(ComputeWidthInTestFrame({.distance_to_solid = 0.125, .cell_size_factor = 2.0}), 0.125);
  EXPECT_EQ(ComputeWidthInTestFrame({.distance_to_solid = 0.5, .cell_size_factor = 4.0}), 1.0);
}

TEST(ComputeWantedCellWidth, AnyNanInput_IsSmallest) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // Each case would give a width above w_min without its NaN: depth 1 m and factor 4 give 0.5 m.
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = nan, .cell_size_factor = 4.0}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.distance_to_solid = nan, .cell_size_factor = 4.0}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0, .velocity_change_size = nan}), 0.0625);
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0, .cell_size_factor = nan}), 0.0625);
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kFrame);
  const CellSizeFacts facts = {.depth_below_surface = 1.0, .velocity_change_size = 10.0};
  EXPECT_EQ(ComputeWantedCellWidth(facts, nan, widths, kSettings), 0.0625);
  // Even where the shear term would not read the reference speed, with a velocity change size of 0.
  EXPECT_EQ(ComputeWantedCellWidth({.depth_below_surface = 1.0}, nan, widths, kSettings), 0.0625);
}

TEST(ComputeWantedCellWidth, InfiniteVelocityChangeAndReferenceSpeed_IsSmallest) {
  // Outside the precondition: infinity / infinity in the shear term, a NaN that Min would drop.
  const CellSizeFacts facts = {.velocity_change_size = math::kPositiveInfinity};
  const double wanted_width =
      ComputeWantedCellWidth(facts, math::kPositiveInfinity, ComputeCellWidthsBySizeLevel(kFrame), kSettings);
  EXPECT_EQ(wanted_width, 0.0625);
}

TEST(ComputeWantedCellWidth, FactorZeroOrBelow_IsSmallest) {
  // With a finite term the product is 0 or negative; with every term infinite, 0 * infinity is NaN or the product is
  // -infinity. Each gives w_min.
  for (const double factor : {0.0, -0.0, -2.0}) {
    EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0, .cell_size_factor = factor}), 0.0625) << factor;
    EXPECT_EQ(ComputeWidthInTestFrame({.cell_size_factor = factor}), 0.0625) << factor;
  }
}

TEST(ComputeWantedCellWidth, InfiniteFactor_IsBiggestWidth) {
  EXPECT_EQ(ComputeWidthInTestFrame({.depth_below_surface = 1.0, .cell_size_factor = math::kPositiveInfinity}), 1.0);
}

TEST(ComputeWantedCellWidth, InfiniteFactorWithShearTermZero_IsSmallest) {
  // An infinite velocity change size makes the shear term 0, and 0 * infinity is NaN.
  const CellSizeFacts facts = {
      .velocity_change_size = math::kPositiveInfinity,
      .cell_size_factor = math::kPositiveInfinity,
  };
  EXPECT_EQ(ComputeWidthInTestFrame(facts), 0.0625);
}

TEST(ComputeWantedCellWidth, ZeroReferenceSpeedWithVelocityChange_IsSmallest) {
  // Zero gravity and a liquid at rest: the reference speed is 0, so the shear term is 0 for any change above 0.
  const double reference_speed = ComputeReferenceSpeed(0.0, 0.0, 1.0);
  const CellSizeFacts facts = {.velocity_change_size = 0.37};
  EXPECT_EQ(ComputeWantedCellWidth(facts, reference_speed, ComputeCellWidthsBySizeLevel(kFrame), kSettings), 0.0625);
}

TEST(ComputeWantedCellWidth, ZeroReferenceSpeedWithoutVelocityChange_HasNoShearTerm) {
  // A velocity change size of 0 skips the shear term, so no 0 / 0 is computed: the far cell gets the biggest width.
  const double reference_speed = ComputeReferenceSpeed(0.0, 0.0, 1.0);
  const CellSizeFacts facts = {.velocity_change_size = 0.0};
  EXPECT_EQ(ComputeWantedCellWidth(facts, reference_speed, ComputeCellWidthsBySizeLevel(kFrame), kSettings), 1.0);
}

TEST(ComputeWantedSizeLevel, WidthEqualToASizeLevel_GivesThatSizeLevel) {
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kFrame);
  EXPECT_EQ(ComputeWantedSizeLevel(0.125, widths), SizeLevel{3});
  EXPECT_EQ(ComputeWantedSizeLevel(std::nextafter(0.125, 0.0), widths), SizeLevel{4});
  EXPECT_EQ(ComputeWantedSizeLevel(0.2, widths), SizeLevel{3});
  EXPECT_EQ(ComputeWantedSizeLevel(5.0, widths), SizeLevel{0});
}

TEST(ComputeWantedSizeLevel, BelowSmallestWidth_GivesSmallestSizeLevel) {
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kFrame);
  EXPECT_EQ(ComputeWantedSizeLevel(0.01, widths), SizeLevel{4});
}

TEST(ComputeWantedSizeLevel, NanWidth_GivesSmallestSizeLevel) {
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kFrame);
  EXPECT_EQ(ComputeWantedSizeLevel(std::numeric_limits<double>::quiet_NaN(), widths), SizeLevel{4});
}

TEST(ComputeReferenceSpeed, SlowLiquid_UsesGravityWaveSpeed) {
  EXPECT_EQ(
      std::bit_cast<std::uint64_t>(ComputeReferenceSpeed(0.1, 9.81, 1.0)),
      std::bit_cast<std::uint64_t>(math::Sqrt(9.81))
  );
  EXPECT_EQ(ComputeReferenceSpeed(5.0, 9.81, 1.0), 5.0);
  // The square root of the product, not the product of two square roots, which differs in the last bit here.
  const double gravity_times_width = 9.81 * 5.0;
  EXPECT_EQ(
      std::bit_cast<std::uint64_t>(ComputeReferenceSpeed(0.1, 9.81, 5.0)),
      std::bit_cast<std::uint64_t>(math::Sqrt(gravity_times_width))
  );
}

TEST(ComputeReferenceSpeed, ZeroGravityLiquidAtRest_IsZero) {
  EXPECT_EQ(ComputeReferenceSpeed(0.0, 0.0, 1.0), 0.0);
}

TEST(ComputeReferenceSpeed, NanLiquidSpeed_UsesGravityWaveSpeed) {
  EXPECT_EQ(ComputeReferenceSpeed(std::numeric_limits<double>::quiet_NaN(), 4.0, 1.0), 2.0);
}

TEST(CellSizeSettings, Defaults_AreTheMethodPapersValues) {
  constexpr CellSizeSettings kDefaults{};
  static_assert(kDefaults.surface_reach_in_cell_widths == 8.0);
  static_assert(kDefaults.solid_reach_in_cell_widths == 2.0);
  static_assert(kDefaults.shear_width_factor == 0.25);
  SUCCEED();
}

TEST(CellSizeSettings, DefaultSmallestSizeLevel_IsTheBiggestCellsUntilARunSetsIt) {
  constexpr CellSizeSettings kDefaults{};
  static_assert(kDefaults.smallest_size_level == SizeLevel{0});
  // Left at 0, w_min is the biggest width, so even a surface cell is 1 m wide.
  const double wanted_width = ComputeWantedCellWidth(
      {.depth_below_surface = 0.0},
      kReferenceSpeed,
      ComputeCellWidthsBySizeLevel(kFrame),
      kDefaults
  );
  EXPECT_EQ(wanted_width, 1.0);
}

TEST(ComputeTableSizeLevelCount, CountOutsideOneToSixteen_IsClamped) {
  static_assert(ComputeTableSizeLevelCount(5) == 5);
  static_assert(ComputeTableSizeLevelCount(0) == 1);
  static_assert(ComputeTableSizeLevelCount(-3) == 1);
  static_assert(ComputeTableSizeLevelCount(16) == 16);
  static_assert(ComputeTableSizeLevelCount(20) == 16);
  SUCCEED();
}

TEST(ComputeCellWidthsBySizeLevel, CountAboveSixteen_FillsSixteenWidths) {
  constexpr GridFrame kTooManyFrame = {.origin = {}, .biggest_cell_width = 1.0, .size_level_count = 20};
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kTooManyFrame);
  EXPECT_EQ(widths.size_level_count, 16);
  EXPECT_EQ(widths.width[15], 0x1p-15);
}

TEST(ComputeCellWidthsBySizeLevel, CountZero_FillsTheBiggestWidthOnly) {
  constexpr GridFrame kNoSizeLevelFrame = {.origin = {}, .biggest_cell_width = 1.0, .size_level_count = 0};
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kNoSizeLevelFrame);
  EXPECT_EQ(widths.size_level_count, 1);
  EXPECT_EQ(widths.width[0], 1.0);
  EXPECT_EQ(widths.width[1], 0.0);
}

TEST(ComputeWantedCellWidth, SmallestSizeLevelPastTheTable_ReadsTheLastSizeLevel) {
  for (const int asked_size_level : {5, 7, 15, 255}) {
    const CellSizeSettings settings = {.smallest_size_level = static_cast<SizeLevel>(asked_size_level)};
    const double wanted_width = ComputeWantedCellWidth(
        {.depth_below_surface = 0.0},
        kReferenceSpeed,
        ComputeCellWidthsBySizeLevel(kFrame),
        settings
    );
    EXPECT_EQ(wanted_width, 0.0625) << asked_size_level;
  }
}

TEST(ComputeWantedSizeLevel, CountOutsideOneToSixteen_IsClamped) {
  CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(kFrame);
  widths.size_level_count = 0;
  EXPECT_EQ(ComputeWantedSizeLevel(0.01, widths), SizeLevel{0});
  widths.size_level_count = 40;
  EXPECT_EQ(ComputeWantedSizeLevel(0.01, widths), SizeLevel{5});
}

}  // namespace
}  // namespace fluidity::grid
