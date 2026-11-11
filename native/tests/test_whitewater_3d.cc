#include <cmath>

#include "fluidity/host/whitewater_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(Whitewater3D, FrozenDragIsStableAtZeroTinyAndStiffRatesAndSubdivides) {
  const WhitewaterKinematics3D initial{.position = {1, 2, 3}, .velocity = {3, 4, 5}};
  const Vec3 carrier{0.3, -0.2, 0.1};
  const Vec3 acceleration{0, -9.81, 0};
  for (const double rate : {0.0, 1e-15, 1e-7, 3.0, 1e6}) {
    auto full = IntegrateWhitewaterDrag3D(initial, carrier, acceleration, rate, 0.2);
    auto half = IntegrateWhitewaterDrag3D(initial, carrier, acceleration, rate, 0.1);
    ASSERT_TRUE(full.ok()) << full.status();
    ASSERT_TRUE(half.ok());
    auto split = IntegrateWhitewaterDrag3D(*half, carrier, acceleration, rate, 0.1);
    ASSERT_TRUE(split.ok());
    EXPECT_NEAR(full->position.x, split->position.x, 2e-14);
    EXPECT_NEAR(full->position.y, split->position.y, 2e-14);
    EXPECT_NEAR(full->velocity.y, split->velocity.y, 2e-13);
  }
  auto ballistic = IntegrateWhitewaterDrag3D(initial, carrier, acceleration, 0, 0.2);
  ASSERT_TRUE(ballistic.ok());
  EXPECT_NEAR(ballistic->position.y, 2 + 4 * 0.2 - 0.5 * 9.81 * 0.04, 1e-14);
  EXPECT_EQ(
      IntegrateWhitewaterDrag3D(initial, carrier, acceleration, -1, 0.2).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}
TEST(Whitewater3D, DeterministicEmissionCarriesRemainderWithoutLosingOppositeBursts) {
  WhitewaterEmissionState3D state;
  const WhitewaterBirthEvent3D first{
      .time_seconds = 0.01,
      .position = {},
      .velocity = {4, 0, 0},
      .normal = {1, 0, 0},
      .volume_time_m3_seconds = 0.14
  };
  auto a = EmitWhitewaterEvent3D(first, 10, 1, state, 100);
  ASSERT_TRUE(a.ok()) << a.status();
  EXPECT_EQ(a->points.size(), 1);
  EXPECT_NEAR(a->state.remainder, 0.4, 1e-14);
  auto second = first;
  second.time_seconds = 0.02;
  second.velocity.x = -4;
  auto b = EmitWhitewaterEvent3D(second, 10, 1, a->state, 100);
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(b->points.size(), 1);
  EXPECT_EQ(a->points[0].velocity.x, 4);
  EXPECT_EQ(b->points[0].velocity.x, -4);
  EXPECT_EQ(a->points[0].birth_time_seconds, 0.01);
  EXPECT_EQ(b->points[0].birth_time_seconds, 0.02);
  EXPECT_NE(a->points[0].id, b->points[0].id);
  EXPECT_EQ(EmitWhitewaterEvent3D(first, 10, 1, state, 0).status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(state.next_id, 1);
  EXPECT_EQ(state.remainder, 0);
}
TEST(Whitewater3D, ClassificationUsesHalfRadiusHysteresis) {
  EXPECT_EQ(ClassifyWhitewater3D(WhitewaterKind3D::kFoam, 0.14, 0.1).value(), WhitewaterKind3D::kFoam);
  EXPECT_EQ(ClassifyWhitewater3D(WhitewaterKind3D::kFoam, 0.16, 0.1).value(), WhitewaterKind3D::kSpray);
  EXPECT_EQ(ClassifyWhitewater3D(WhitewaterKind3D::kSpray, 0.06, 0.1).value(), WhitewaterKind3D::kSpray);
  EXPECT_EQ(ClassifyWhitewater3D(WhitewaterKind3D::kSpray, 0.04, 0.1).value(), WhitewaterKind3D::kFoam);
  EXPECT_EQ(ClassifyWhitewater3D(WhitewaterKind3D::kBubble, -0.06, 0.1).value(), WhitewaterKind3D::kBubble);
  EXPECT_EQ(ClassifyWhitewater3D(WhitewaterKind3D::kBubble, -0.04, 0.1).value(), WhitewaterKind3D::kFoam);
}

TEST(Whitewater3D, SourceStatisticsUseSpecificEnergyRelativeVelocityAndOutwardCrests) {
  const WhitewaterSourceCell3D center{
      .index = {0, 0, 0},
      .velocity = {2, 0, 0},
      .normal = {1, 0, 0},
      .curvature_per_m = 3
  };
  const std::array neighbors{
      WhitewaterSourceCell3D{.index = {1, 0, 0}, .velocity = {-2, 0, 0}, .normal = {1, 0, 0}, .curvature_per_m = 0}
  };
  const WhitewaterSourceOptions3D options{
      .spacing_m = 0.1,
      .radius_m = 0.1,
      .kinetic = {0, 2},
      .trapped_air = {0, 4},
      .crest = {0, 3},
      .trapped_air_weight = 2,
      .crest_weight = 3,
  };
  auto result = ComputeWhitewaterSource3D(center, neighbors, options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->specific_energy_m2_per_s2, 2);
  EXPECT_EQ(result->trapped_air_m_per_s, 4);
  EXPECT_EQ(result->crest_per_m, 3);
  EXPECT_EQ(result->potential, 5);
  auto inward = center;
  inward.velocity.x = -2;
  result = ComputeWhitewaterSource3D(inward, neighbors, options);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->crest_per_m, 0);
  auto invalid = center;
  invalid.normal.x = std::nan("");
  EXPECT_EQ(ComputeWhitewaterSource3D(invalid, neighbors, options).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(Whitewater3D, SourceRejectsMissingCarrierUnorderedThresholdsAndDuplicateNeighbors) {
  auto center = WhitewaterSourceCell3D{.index = {}, .velocity = {}, .normal = {1, 0, 0}, .curvature_per_m = 0};
  auto options = WhitewaterSourceOptions3D{
      .spacing_m = 0.1,
      .radius_m = 0.1,
      .kinetic = {0, 1},
      .trapped_air = {0, 1},
      .crest = {0, 1},
      .trapped_air_weight = 1,
      .crest_weight = 1,
  };
  center.carrier_valid = false;
  EXPECT_EQ(ComputeWhitewaterSource3D(center, {}, options).status().code(), absl::StatusCode::kFailedPrecondition);
  center.carrier_valid = true;
  options.kinetic.high = 0;
  EXPECT_EQ(ComputeWhitewaterSource3D(center, {}, options).status().code(), absl::StatusCode::kInvalidArgument);
  options.kinetic.high = 1;
  auto neighbor = center;
  neighbor.index.x = -1;
  const std::array duplicate{neighbor, neighbor};
  EXPECT_EQ(ComputeWhitewaterSource3D(center, duplicate, options).status().code(), absl::StatusCode::kInvalidArgument);
  auto result = ComputeWhitewaterSource3D(center, std::span(&center, 1), options);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->potential, 0);
}
}  // namespace
}  // namespace fluidity
