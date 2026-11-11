#include <cmath>
#include <limits>

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

TEST(Whitewater3D, RelativeVelocityOverflowIsNotMaskedByZeroKineticEnergy) {
  const WhitewaterSourceCell3D center{.index = {}, .velocity = {}, .normal = {1, 0, 0}, .curvature_per_m = 0};
  const WhitewaterSourceCell3D neighbor{
      .index = {1, 1, 0},
      .velocity = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0},
      .normal = {1, 0, 0},
      .curvature_per_m = 0,
  };
  const WhitewaterSourceOptions3D options{
      .spacing_m = 0.1,
      .radius_m = 0.1,
      .kinetic = {0, 1},
      .trapped_air = {0, 1},
      .crest = {0, 1},
      .trapped_air_weight = 1,
      .crest_weight = 1,
  };
  EXPECT_EQ(
      ComputeWhitewaterSource3D(center, std::span(&neighbor, 1), options).status().code(),
      absl::StatusCode::kOutOfRange
  );
}

TEST(Whitewater3D, BirthUsesQuadratureOccupancyAndAcceptedDuration) {
  WhitewaterOccupiedCell3D cell{
      .sample = {.index = {-1, 2, 0}, .velocity = {2, 0, 0}, .normal = {1, 0, 0}, .curvature_per_m = 3},
      .occupied_volume_m3 = 0.0625,
  };
  const WhitewaterSourceOptions3D options{
      .spacing_m = 0.5,
      .radius_m = 0.5,
      .kinetic = {0, 2},
      .trapped_air = {0, 1},
      .crest = {0, 3},
      .trapped_air_weight = 0,
      .crest_weight = 2,
  };
  const WhitewaterSourceStep3D step{
      .start_time_seconds = 0.25,
      .end_time_seconds = 0.375,
      .quadrature_time_seconds = 0.3125
  };
  const auto births = ComputeWhitewaterBirths3D(std::span(&cell, 1), {1, 2, 3}, step, options);
  ASSERT_TRUE(births.ok()) << births.status();
  ASSERT_EQ(births->size(), 1);
  const auto& birth = births->front();
  EXPECT_EQ(birth.source_cell.x, -1);
  EXPECT_DOUBLE_EQ(birth.event.time_seconds, 0.3125);
  EXPECT_DOUBLE_EQ(birth.event.position.x, 0.75);
  EXPECT_DOUBLE_EQ(birth.event.position.y, 3.25);
  EXPECT_DOUBLE_EQ(birth.event.velocity.x, 2);
  EXPECT_DOUBLE_EQ(birth.event.volume_time_m3_seconds, 0.015625);
  EXPECT_EQ(
      ComputeWhitewaterBirths3D(std::span(&cell, 1), {}, step, options, 0).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  cell.sample.velocity = {};
  auto still = ComputeWhitewaterBirths3D(std::span(&cell, 1), {}, step, options);
  ASSERT_TRUE(still.ok());
  EXPECT_TRUE(still->empty());
  cell.sample.carrier_valid = false;
  EXPECT_EQ(
      ComputeWhitewaterBirths3D(std::span(&cell, 1), {}, step, options).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(Whitewater3D, CellRemaindersRemainSeparateAcrossIntervalsWithGlobalIds) {
  std::array births{
      AcceptedWhitewaterBirth3D{
          .accepted_step_ordinal = 1,
          .birth =
              {
                  .source_cell = {-2, 0, 0},
                  .event =
                      {.time_seconds = 0.01,
                       .position = {-1, 0, 0},
                       .velocity = {2, 0, 0},
                       .normal = {1, 0, 0},
                       .volume_time_m3_seconds = 0.6},
              },
      },
      AcceptedWhitewaterBirth3D{
          .accepted_step_ordinal = 1,
          .birth =
              {
                  .source_cell = {1, 0, 0},
                  .event =
                      {.time_seconds = 0.01,
                       .position = {1, 0, 0},
                       .velocity = {-2, 0, 0},
                       .normal = {-1, 0, 0},
                       .volume_time_m3_seconds = 0.6},
              },
      },
  };
  auto first = EmitWhitewaterBirthsThroughStep3D(births, {}, 2, 1, 1, 10);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_TRUE(first->points.empty());
  ASSERT_EQ(first->checkpoint.remainders.size(), 2);
  EXPECT_DOUBLE_EQ(first->checkpoint.remainders[0].fractional_count, 0.6);
  EXPECT_DOUBLE_EQ(first->checkpoint.remainders[1].fractional_count, 0.6);
  for (auto& birth : births) {
    birth.accepted_step_ordinal = 3;
    birth.birth.event.time_seconds = 0.03;
  }
  auto second = EmitWhitewaterBirthsThroughStep3D(births, first->checkpoint, 3, 1, 1, 10);
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_EQ(second->points.size(), 2);
  EXPECT_EQ(second->points[0].id, 1);
  EXPECT_EQ(second->points[1].id, 2);
  EXPECT_DOUBLE_EQ(second->points[0].velocity.x, 2);
  EXPECT_DOUBLE_EQ(second->points[1].velocity.x, -2);
  EXPECT_DOUBLE_EQ(second->points[1].birth_time_seconds, 0.03);
  EXPECT_EQ(second->checkpoint.next_point_id, 3);
  EXPECT_EQ(second->checkpoint.last_accepted_step_ordinal, 3);
  EXPECT_NEAR(second->checkpoint.remainders[0].fractional_count, 0.2, 1e-15);
  EXPECT_NEAR(second->checkpoint.remainders[1].fractional_count, 0.2, 1e-15);
  EXPECT_EQ(
      EmitWhitewaterBirthsThroughStep3D(births, first->checkpoint, 3, 1, 1, 1).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  EXPECT_EQ(
      EmitWhitewaterBirthsThroughStep3D(births, second->checkpoint, 4, 1, 1, 10).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(first->checkpoint.next_point_id, 1);
  EXPECT_DOUBLE_EQ(first->checkpoint.remainders[0].fractional_count, 0.6);
}
}  // namespace
}  // namespace fluidity
