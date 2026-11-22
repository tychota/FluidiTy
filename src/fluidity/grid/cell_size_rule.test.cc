#include "fluidity/grid/cell_size_rule.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <vector>

#include "fluidity/base/run_on.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_size_rule_cases.h"
#include "fluidity/grid/testing/grid_support.h"

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

CellId ToTestCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// A committed grid of `cells`, built on the CPU in `frame`.
std::expected<TestGrid, GridFailure> BuildCommittedGridInFrame(GridFrame frame, std::span<const CellId> cells) {
  return BuildTestGrid({.frame = frame, .active_cells = cells, .version = {.value = 1}});
}

// A committed grid of `cells`, built on the CPU in a frame of origin 0, biggest cells 1 m wide and size_level_count
// size levels.
std::expected<TestGrid, GridFailure> BuildCommittedGrid(int size_level_count, std::span<const CellId> cells) {
  const GridFrame frame = {
      .origin = {},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
  return BuildCommittedGridInFrame(frame, cells);
}

// The 8 small cells of `big_cell`.
std::vector<CellId> BuildSmallCells(CellId big_cell) {
  std::vector<CellId> small_cells;
  small_cells.reserve(8);
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    small_cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
  return small_cells;
}

// The place of `cell` in `committed`, which must hold it.
std::uint32_t FindTestCellPlace(const GridTables& committed, CellId cell) {
  const CellPlaceIndex place = FindCellPlace(committed, cell);
  EXPECT_NE(place.value, kNoIndex);
  return place.value;
}

// The two-size fixture in a frame of 3 size levels: size level 0 {0,0,0} beside the 8 size-level-1 cells of size level
// 0 {1,0,0}. From (-0.5, 0.5, 0.5) the searched position (0.5, 0.5, 0.5) finds the big cell, whose centre is 1 m away.
// From (2.25, 0.25, 0.25) the searched position (1.25, 0.25, 0.25) finds the small cell {2,0,0}, whose centre is 1 m
// away; the small cell {3,0,0}, whose centre is 0.5 m away, holds no searched position. No other searched position
// finds a cell.
TEST(ComputeNewCellSizeLevel, NextToBigCell_IsOneSizeLevelSmaller) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(3, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  EXPECT_EQ(ComputeNewCellSizeLevel(tables, {.x = -0.5, .y = 0.5, .z = 0.5}, SizeLevel{2}), SizeLevel{1});
  EXPECT_EQ(ComputeNewCellSizeLevel(tables, {.x = 2.25, .y = 0.25, .z = 0.25}, SizeLevel{2}), SizeLevel{2});
}

TEST(ComputeNewCellSizeLevel, AtSmallestSizeLevel_IsCapped) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(3, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  EXPECT_EQ(ComputeNewCellSizeLevel(tables, {.x = 2.25, .y = 0.25, .z = 0.25}, SizeLevel{1}), SizeLevel{1});
}

// No searched position lies in an active cell: far from the cells, and in a grid of no cells.
TEST(ComputeNewCellSizeLevel, NoCellNearby_IsSmallestSizeLevel) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(3, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  const Vec3 far_position = {.x = 10.0, .y = 10.0, .z = 10.0};
  EXPECT_EQ(ComputeNewCellSizeLevel(tables, far_position, SizeLevel{2}), SizeLevel{2});
  EXPECT_EQ(ComputeNewCellSizeLevel(tables, far_position, SizeLevel{1}), SizeLevel{1});
  const std::expected<TestGrid, GridFailure> empty = BuildCommittedGrid(3, {});
  ASSERT_TRUE(empty.has_value());
  EXPECT_EQ(ComputeNewCellSizeLevel(empty->grid.tables(RunOn::kCpu), far_position, SizeLevel{2}), SizeLevel{2});
}

// The tables of no grid have a frame of no size level, which reads as one size level, as ComputeTableSizeLevelCount
// reads it: size level 0 whatever smallest_size_level asks.
TEST(ComputeNewCellSizeLevel, TablesOfNoGrid_IsSizeLevelZero) {
  const Vec3 far_position = {.x = 10.0, .y = 10.0, .z = 10.0};
  EXPECT_EQ(ComputeNewCellSizeLevel(GridTables{}, far_position, SizeLevel{2}), SizeLevel{0});
  EXPECT_EQ(ComputeNewCellSizeLevel(GridTables{}, far_position, static_cast<SizeLevel>(200)), SizeLevel{0});
}

// A NaN or an infinite axis puts every searched position outside the grid.
TEST(ComputeNewCellSizeLevel, PositionNotFinite_IsSmallestSizeLevel) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(3, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  for (const double value :
       {std::numeric_limits<double>::quiet_NaN(), math::kPositiveInfinity, -math::kPositiveInfinity}) {
    EXPECT_EQ(ComputeNewCellSizeLevel(tables, {.x = value, .y = 0.5, .z = 0.5}, SizeLevel{1}), SizeLevel{1}) << value;
  }
}

// Size level 0 {0,0,0} split, its small cell {0,1,1} split, that one's {0,2,2} split and that one's {0,4,4} split, in
// a frame of 5 size levels, closed under the size step rule.
std::vector<CellId> BuildNestedSplitCells() {
  const std::array<CellId, 4> split_cells = {
      ToTestCell(0, 0, 0, 0),
      ToTestCell(1, 0, 1, 1),
      ToTestCell(2, 0, 2, 2),
      ToTestCell(3, 0, 4, 4),
  };
  std::vector<CellId> cells;
  for (std::size_t split_index = 0; split_index < split_cells.size(); ++split_index) {
    const bool has_split_small_cell = split_index + 1 < split_cells.size();
    for (const CellId small_cell : BuildSmallCells(split_cells.at(split_index))) {
      const bool is_split = has_split_small_cell && small_cell.value == split_cells.at(split_index + 1).value;
      if (!is_split) {
        cells.push_back(small_cell);
      }
    }
  }
  return BuildSizeStepClosure(cells, 5);
}

// The size step rule is kept with the cell found only. 5 size levels: size level 0 {0,0,0} split, its small cell
// {0,1,1} split, that one's {0,2,2} split and that one's {0,4,4} split, closed under the size step rule. From
// (-0.01, 0.53, 0.53), 0.01 m outside the grid's -X side, the one searched position inside the grid, (0.99, 0.53,
// 0.53), finds the size-level-1 cell {1,1,1}, so the new cell gets size level 2. Yet the cell across the face x = 0,
// 0.04 m from the position, is of size level 4: no searched position lies in it. A new cell there would be two size
// levels bigger than a cell it touches, so the step that adds cells closes the size step rule over its new list.
TEST(ComputeNewCellSizeLevel, SmallerCellMissedByTheSearch_CanBreakTheSizeStepRule) {
  const std::vector<CellId> cells = BuildNestedSplitCells();
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(5, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  const Vec3 position = {.x = -0.01, .y = 0.53, .z = 0.53};
  EXPECT_EQ(FindNearestCommittedCell(tables, position).value, FindTestCellPlace(tables, ToTestCell(1, 1, 1, 1)));
  EXPECT_EQ(ComputeNewCellSizeLevel(tables, position, SizeLevel{4}), SizeLevel{2});
  const CellPlaceIndex touching = FindCellAt(tables, {.x = 0.0, .y = 0.53, .z = 0.53});
  ASSERT_NE(touching.value, kNoIndex);
  EXPECT_EQ(ToSizeLevel(ToCellId(tables, touching)), SizeLevel{4});
}

// The two-size fixture in a frame of 2 size levels, so the small cells are of the frame's last size level.
TEST(ComputeNewCellSizeLevel, SmallestSizeLevelPastTheFrame_ReadsTheLastSizeLevel) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(2, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  EXPECT_EQ(ComputeNewCellSizeLevel(tables, {.x = 2.25, .y = 0.25, .z = 0.25}, SizeLevel{5}), SizeLevel{1});
  EXPECT_EQ(ComputeNewCellSizeLevel(tables, {.x = 10.0, .y = 10.0, .z = 10.0}, SizeLevel{5}), SizeLevel{1});
}

TEST(FindNearestCommittedCell, TwoCellsEquallyNear_FirstInFixedOrderWins) {
  const std::array<CellId, 2> cells = {ToTestCell(0, 0, 0, 0), ToTestCell(0, 2, 0, 0)};
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(1, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  // Exactly 1.0 m from both centres: the cell first in the fixed order.
  EXPECT_EQ(
      FindNearestCommittedCell(tables, {.x = 1.5, .y = 0.5, .z = 0.5}).value,
      FindTestCellPlace(tables, cells[0])
  );
  // 1.25 m from {0,0,0} and 0.75 m from {2,0,0}.
  EXPECT_EQ(
      FindNearestCommittedCell(tables, {.x = 1.75, .y = 0.5, .z = 0.5}).value,
      FindTestCellPlace(tables, cells[1])
  );
  EXPECT_EQ(FindNearestCommittedCell(tables, {.x = 10.0, .y = 10.0, .z = 10.0}).value, kNoIndex);
}

// Size level 0 {0,0,0} and the 8 size-level-1 cells of size level 0 {2,-1,0}, 2 size levels. From (1.75, 0.5, 0.625)
// the searched position (2.75, -0.5, 0.625) finds the small cell {5,-1,1}, before (0.75, 0.5, 0.625) finds the big
// cell. Both centres lie at the squared distance 1.578125 m2. The big cell's low corner has the smaller z, so it comes
// first in the fixed order, though the search finds it second. Cells of one size are found in their fixed order, so
// only two sizes tell the fixed order from the search order.
TEST(FindNearestCommittedCell, TieFoundFirstButLaterInFixedOrder_LosesToTheEarlierCell) {
  std::vector<CellId> cells = {ToTestCell(0, 0, 0, 0)};
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(ToTestCell(0, 2, -1, 0), cell_eighth));
  }
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(2, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  EXPECT_EQ(
      FindNearestCommittedCell(tables, {.x = 1.75, .y = 0.5, .z = 0.625}).value,
      FindTestCellPlace(tables, cells[0])
  );
}

// Size level 0 {0,0,0} and the 8 size-level-1 cells of size level 0 {2,1,0}, 2 size levels. From (1.875, 0.375, 0.375)
// the big cell's centre lies at (-1.375, 0.125, 0.125), squared distance 1.921875 m2, and the small cell {5,2,0}'s at
// (0.875, 0.875, -0.125), 1.546875 m2: the small cell is nearer, though the sum of its axes, 1.875 m, is the larger.
TEST(FindNearestCommittedCell, NearerInStraightLine_WinsOverSmallerSumOfAxes) {
  std::vector<CellId> cells = {ToTestCell(0, 0, 0, 0)};
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(ToTestCell(0, 2, 1, 0), cell_eighth));
  }
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(2, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  EXPECT_EQ(
      FindNearestCommittedCell(tables, {.x = 1.875, .y = 0.375, .z = 0.375}).value,
      FindTestCellPlace(tables, ToTestCell(1, 5, 2, 0))
  );
}

// One cell, found only from a searched position across a corner: every offset -1, or every offset +1.
TEST(FindNearestCommittedCell, CellOnlyAcrossACorner_IsFound) {
  const std::array<CellId, 1> cells = {ToTestCell(0, 0, 0, 0)};
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(1, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  const std::uint32_t cell_place = FindTestCellPlace(tables, cells[0]);
  EXPECT_EQ(FindNearestCommittedCell(tables, {.x = -0.5, .y = -0.5, .z = -0.5}).value, cell_place);
  EXPECT_EQ(FindNearestCommittedCell(tables, {.x = 1.5, .y = 1.5, .z = 1.5}).value, cell_place);
}

// Inside {1,0,0}, 0.25 m from its centre and 0.75 m from the centre of {0,0,0}, which a searched position also finds.
TEST(FindNearestCommittedCell, PositionInsideACell_FindsThatCell) {
  const std::array<CellId, 2> cells = {ToTestCell(0, 0, 0, 0), ToTestCell(0, 1, 0, 0)};
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(1, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  EXPECT_EQ(
      FindNearestCommittedCell(tables, {.x = 1.25, .y = 0.5, .z = 0.5}).value,
      FindTestCellPlace(tables, cells[1])
  );
}

// Size level 0 {0,0,0} and the 8 size-level-1 cells of size level 0 {2,0,-1}, 2 size levels. From (1, 0.25, -0.625)
// the big cell's centre (0.5, 0.5, 0.5) and the small cell {4,0,-2}'s centre (2.25, 0.25, -0.75) both lie at the
// squared distance 1.578125 m2. The small cell's low corner has z = -1, below the big cell's 0, so the small cell comes
// first in the fixed order and wins. Without z the big cell would be nearer (0.3125 against 1.5625 m2).
TEST(FindNearestCommittedCell, TieWithSmallCellFirstInFixedOrder_SmallCellWins) {
  std::vector<CellId> cells = BuildSmallCells(ToTestCell(0, 2, 0, -1));
  cells.push_back(ToTestCell(0, 0, 0, 0));
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(2, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  EXPECT_EQ(
      FindNearestCommittedCell(tables, {.x = 1.0, .y = 0.25, .z = -0.625}).value,
      FindTestCellPlace(tables, ToTestCell(1, 4, 0, -2))
  );
}

// A NaN or an infinite axis puts every searched position outside the grid.
TEST(FindNearestCommittedCell, PositionNotFinite_IsNoIndex) {
  const std::vector<CellId> cells = BuildTwoSizeFixtureCells();
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGrid(3, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  for (const double value :
       {std::numeric_limits<double>::quiet_NaN(), math::kPositiveInfinity, -math::kPositiveInfinity}) {
    EXPECT_EQ(FindNearestCommittedCell(tables, {.x = value, .y = 0.5, .z = 0.5}).value, kNoIndex) << value;
    EXPECT_EQ(FindNearestCommittedCell(tables, {.x = 0.5, .y = value, .z = 0.5}).value, kNoIndex) << value;
    EXPECT_EQ(FindNearestCommittedCell(tables, {.x = 0.5, .y = 0.5, .z = value}).value, kNoIndex) << value;
  }
}

// Cells {0,0,0} and {2,0,0} of one size level, and the position 1.75, 0.5 and 0.5 biggest widths from the origin:
// 1.25 widths from the centre of {0,0,0} and 0.75 from the centre of {2,0,0}, which is the nearer.
void ExpectNearerCellInFrame(GridFrame frame) {
  const std::array<CellId, 2> cells = {ToTestCell(0, 0, 0, 0), ToTestCell(0, 2, 0, 0)};
  const std::expected<TestGrid, GridFailure> committed = BuildCommittedGridInFrame(frame, cells);
  ASSERT_TRUE(committed.has_value());
  const GridTables tables = committed->grid.tables(RunOn::kCpu);
  const double width = frame.biggest_cell_width;
  const Vec3 position = {
      .x = frame.origin.x + (1.75 * width),
      .y = frame.origin.y + (0.5 * width),
      .z = frame.origin.z + (0.5 * width),
  };
  EXPECT_EQ(FindNearestCommittedCell(tables, position).value, FindTestCellPlace(tables, cells[1]));
}

// An origin and a width that are not powers of two: every centre and the position round, and the nearer cell
// still wins.
TEST(FindNearestCommittedCell, OriginAndWidthNotPowersOfTwo_FindsTheNearerCell) {
  ExpectNearerCellInFrame(
      {.origin = {.x = -2.3, .y = 0.7, .z = 1000.1}, .biggest_cell_width = 0.3, .size_level_count = 1}
  );
}

// Biggest cells 1e160 m wide: a squared distance in metres would overflow to +infinity for both cells.
TEST(FindNearestCommittedCell, HugeBiggestCellWidth_FindsTheNearerCell) {
  ExpectNearerCellInFrame({.origin = {}, .biggest_cell_width = 1e160, .size_level_count = 1});
}

// Biggest cells 1e-170 m wide: a squared distance in metres would underflow to 0 for both cells.
TEST(FindNearestCommittedCell, TinyBiggestCellWidth_FindsTheNearerCell) {
  ExpectNearerCellInFrame({.origin = {}, .biggest_cell_width = 1e-170, .size_level_count = 1});
}

}  // namespace
}  // namespace fluidity::grid
