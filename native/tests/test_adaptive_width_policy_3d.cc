#include <array>
#include <cmath>
#include <limits>

#include "absl/status/status.h"
#include "fluidity/host/adaptive_width_policy_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr AdaptiveWidthDemand3D kDemand{.maximum_width = 5.0};
constexpr AdaptiveCoarsenHaloSample3D kSafeSample{.demand = kDemand};

TEST(AdaptiveWidthPolicy3D, RefinesImmediatelyToTightestPhysicalOrCameraDemand) {
  for (const AdaptiveWidthDemand3D demand : {
           AdaptiveWidthDemand3D{.physical_width = 0.75, .camera_width = 4.0, .maximum_width = 5.0},
           AdaptiveWidthDemand3D{.physical_width = 4.0, .camera_width = 0.75, .maximum_width = 5.0},
       }) {
    const auto decision = DecideAdaptiveWidth3D(1.0, demand, {}, false, false, 0.125, 0.0, 1.0);
    ASSERT_TRUE(decision.ok()) << decision.status();
    EXPECT_EQ(decision->change, AdaptiveWidthChange3D::kRefine);
    EXPECT_DOUBLE_EQ(decision->eligible_duration, 0.0);
  }
}

TEST(AdaptiveWidthPolicy3D, CountsOnlyAcceptedDurationAndUsesInclusiveMargin) {
  const std::array halo{AdaptiveCoarsenHaloSample3D{.demand = {.maximum_width = 2.5}}};
  const auto first = DecideAdaptiveWidth3D(1.0, kDemand, halo, true, true, 0.0, 0.25, 0.5);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->change, AdaptiveWidthChange3D::kKeep);
  EXPECT_DOUBLE_EQ(first->eligible_duration, 0.25);
  const auto replay = DecideAdaptiveWidth3D(1.0, kDemand, halo, true, true, first->eligible_duration, 0.0, 0.5);
  ASSERT_TRUE(replay.ok()) << replay.status();
  EXPECT_EQ(replay->change, AdaptiveWidthChange3D::kKeep);
  EXPECT_DOUBLE_EQ(replay->eligible_duration, 0.25);
  const auto second = DecideAdaptiveWidth3D(1.0, kDemand, halo, true, true, first->eligible_duration, 0.25, 0.5);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(second->change, AdaptiveWidthChange3D::kCoarsen);
  EXPECT_DOUBLE_EQ(second->eligible_duration, 0.0);
}

TEST(AdaptiveWidthPolicy3D, ResetsHysteresisForPhysicalVetoesAndIncompleteHalo) {
  const auto expect_veto = [](AdaptiveCoarsenHaloSample3D sample, AdaptiveCoarsenVeto3D veto) {
    const std::array halo{kSafeSample, sample};
    const auto result = DecideAdaptiveWidth3D(1.0, kDemand, halo, true, true, 0.25, 0.25, 0.5);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->change, AdaptiveWidthChange3D::kKeep);
    EXPECT_EQ(result->veto, veto);
    EXPECT_DOUBLE_EQ(result->eligible_duration, 0.0);
  };
  expect_veto({.demand = {.support_width = 2.0, .maximum_width = 5.0}}, AdaptiveCoarsenVeto3D::kWidth);
  expect_veto({.demand = kDemand, .touches_cut_or_contact = true}, AdaptiveCoarsenVeto3D::kCutOrContact);
  expect_veto({.demand = kDemand, .support_certified = false}, AdaptiveCoarsenVeto3D::kUncertifiedSupport);
  expect_veto({.demand = kDemand, .cavity_width = 7.99}, AdaptiveCoarsenVeto3D::kUnresolvedFeature);
  expect_veto({.demand = kDemand, .sheet_width = 7.99}, AdaptiveCoarsenVeto3D::kUnresolvedFeature);
  expect_veto({.demand = kDemand, .gap_width = 7.99}, AdaptiveCoarsenVeto3D::kUnresolvedFeature);

  const std::array halo{kSafeSample};
  const auto incomplete = DecideAdaptiveWidth3D(1.0, kDemand, halo, true, false, 0.25, 0.25, 0.5);
  ASSERT_TRUE(incomplete.ok()) << incomplete.status();
  EXPECT_EQ(incomplete->veto, AdaptiveCoarsenVeto3D::kIncompleteHalo);
  EXPECT_DOUBLE_EQ(incomplete->eligible_duration, 0.0);
}

TEST(AdaptiveWidthPolicy3D, RejectsNonfiniteAndInvalidClockInputs) {
  constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
  const std::array halo{kSafeSample};
  EXPECT_EQ(
      DecideAdaptiveWidth3D(1.0, kDemand, halo, true, true, 0.0, kNan, 0.5).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      DecideAdaptiveWidth3D(1.0, kDemand, halo, true, true, 0.75, 0.25, 0.5).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  const std::array invalid_halo{AdaptiveCoarsenHaloSample3D{.demand = kDemand, .sheet_width = kNan}};
  EXPECT_EQ(
      DecideAdaptiveWidth3D(1.0, kDemand, invalid_halo, true, true, 0.0, 0.25, 0.5).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

}  // namespace
}  // namespace fluidity
