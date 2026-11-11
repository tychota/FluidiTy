#include <array>
#include <cstdint>
#include <limits>

#include "fluidity/host/liquid_resampling_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr auto kCubic = LiquidUniformResamplingBasis3D::kCubicNormal;
constexpr auto kQuadratic = LiquidUniformResamplingBasis3D::kQuadratic;

std::array<LiquidParticleState3D, 2> CounterstreamingChildren() {
  return {
      LiquidParticleState3D{
          .id = 3,
          .tag = 17,
          .world_position = {-0.5, 0.0, 0.0},
          .world_velocity = {2.0, -1.0, 0.0},
          .velocity_slopes = {{1.0, 2.0, 0.0}, {-3.0, 1.0, 0.0}, {}},
          .reference_volume = 1.0,
          .volume_ratio = 2.0,
          .side_tag = -42,
          .pic_initialized = true,
      },
      LiquidParticleState3D{
          .id = 2,
          .tag = 17,
          .world_position = {0.5, 0.0, 0.0},
          .world_velocity = {-2.0, 1.0, 0.0},
          .velocity_slopes = {{2.0, -1.0, 0.0}, {1.0, -2.0, 0.0}, {}},
          .reference_volume = 3.0,
          .volume_ratio = 3.0,
          .side_tag = -42,
          .pic_initialized = true,
      },
  };
}

struct Certification {
  int calls = 0;
  int refuse_call = -1;
  std::array<double, 3> widths{};
  std::array<std::uint64_t, 3> ids{};
};

absl::Status
Certify(const LiquidParticleState3D& particle, double width, LiquidUniformResamplingBasis3D, void* context) {
  auto& certification = *static_cast<Certification*>(context);
  const int index = certification.calls++;
  certification.widths[index] = width;
  certification.ids[index] = particle.id;
  if (index == certification.refuse_call) {
    return absl::FailedPreconditionError("geometry or EOS refuses particle");
  }
  return absl::OkStatus();
}

TEST(LiquidUniformMerge3D, ConservesCounterstreamingMomentsAndDissipatesEnergy) {
  const auto children = CounterstreamingChildren();
  Certification certification;
  auto result = MergeUniformLiquidParticles3D(children, 10, 1.0, kCubic, Certify, &certification);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(certification.calls, 3);
  EXPECT_EQ(certification.widths, (std::array{1.0, 1.0, 2.0}));
  EXPECT_EQ(certification.ids, (std::array<std::uint64_t, 3>{3, 2, 10}));
  EXPECT_EQ(result->next_particle_id, 11);
  EXPECT_GT(result->covariance_gap_lower_bound, 0.0);
  EXPECT_LE(result->covariance_gap_lower_bound, 0.5);

  const auto& merged = result->particle;
  const LiquidParticleState3D expected{
      .id = 10,
      .tag = 17,
      .velocity_slopes = {{-3.0 / 8.0, 1.0 / 8.0, 0.0}, {1.0 / 4.0, -1.0 / 8.0, 0.0}, {}},
      .mass = 2.0,
      .reference_volume = 4.0,
      .volume_ratio = 11.0 / 4.0,
      .side_tag = -42,
      .pic_initialized = true,
  };
  EXPECT_EQ(merged, expected);
  EXPECT_DOUBLE_EQ(merged.reference_volume * merged.volume_ratio, 11.0);
  const double angular_momentum = merged.mass * (merged.velocity_slopes.y_row.x - merged.velocity_slopes.x_row.y);
  EXPECT_DOUBLE_EQ(angular_momentum, 1.0 / 4.0);
  const auto slope_x = merged.velocity_slopes.x_row;
  const auto slope_y = merged.velocity_slopes.y_row;
  const double merged_energy = merged.mass / 2.0
      * (4.0 / 3.0 * (slope_x.x * slope_x.x + slope_y.y * slope_y.y) + slope_x.y * slope_x.y + slope_y.x * slope_y.x);
  EXPECT_DOUBLE_EQ(merged_energy, 55.0 / 192.0);
  EXPECT_LT(merged_energy, 205.0 / 24.0);
}

TEST(LiquidUniformMerge3D, UsesMassWeightsForCentersAndCrossMoments) {
  auto children = CounterstreamingChildren();
  children[1].mass = 3.0;
  Certification certification;
  const auto result = MergeUniformLiquidParticles3D(children, 10, 1.0, kCubic, Certify, &certification);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->particle.mass, 4.0);
  EXPECT_DOUBLE_EQ(result->particle.world_position.x, 0.25);
  EXPECT_DOUBLE_EQ(result->particle.world_velocity.x, -1.0);
  EXPECT_DOUBLE_EQ(result->particle.world_velocity.y, 0.5);
  EXPECT_DOUBLE_EQ(result->particle.velocity_slopes.x_row.x, -0.125);
  EXPECT_DOUBLE_EQ(result->particle.velocity_slopes.y_row.x, 0.375);
  EXPECT_DOUBLE_EQ(result->particle.reference_volume * result->particle.volume_ratio, 11.0);
}

TEST(LiquidUniformMerge3D, RejectsNonpositiveOrUncertainRowCovariance) {
  auto children = CounterstreamingChildren();
  Certification certification;
  children[0].world_position.x = -1.0;
  children[1].world_position.x = 1.0;
  EXPECT_EQ(
      MergeUniformLiquidParticles3D(children, 10, 1.0, kCubic, Certify, &certification).status().code(),
      absl::StatusCode::kFailedPrecondition
  );

  for (double coordinate : {0.5, 0.55}) {
    children[0].world_position = {-coordinate, -coordinate, -coordinate};
    children[1].world_position = {coordinate, coordinate, coordinate};
    EXPECT_EQ(
        MergeUniformLiquidParticles3D(children, 10, 1.0, kQuadratic, Certify, &certification).status().code(),
        absl::StatusCode::kFailedPrecondition
    );
  }
  EXPECT_EQ(certification.calls, 0);
}

TEST(LiquidUniformMerge3D, RejectsMismatchesAndPropagatesAdmissibilityRefusal) {
  auto children = CounterstreamingChildren();
  Certification certification;
  children[1].tag++;
  EXPECT_EQ(
      MergeUniformLiquidParticles3D(children, 10, 1.0, kCubic, Certify, &certification).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  children = CounterstreamingChildren();
  children[1].side_tag++;
  EXPECT_EQ(
      MergeUniformLiquidParticles3D(children, 10, 1.0, kCubic, Certify, &certification).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  children = CounterstreamingChildren();
  children[1].affine_enabled = false;
  EXPECT_EQ(
      MergeUniformLiquidParticles3D(children, 10, 1.0, kCubic, Certify, &certification).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  children = CounterstreamingChildren();
  children[1].id = children[0].id;
  EXPECT_EQ(
      MergeUniformLiquidParticles3D(children, 10, 1.0, kCubic, Certify, &certification).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  children = CounterstreamingChildren();
  constexpr auto max_id = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(
      MergeUniformLiquidParticles3D(children, max_id, 1.0, kCubic, Certify, &certification).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(certification.calls, 0);

  for (int refuse_call : {0, 2}) {
    certification = {.refuse_call = refuse_call};
    auto rejected = MergeUniformLiquidParticles3D(children, 10, 1.0, kCubic, Certify, &certification);
    EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(certification.calls, refuse_call + 1);
    EXPECT_EQ(children, CounterstreamingChildren());
  }
  certification = {};
  auto last = MergeUniformLiquidParticles3D(children, max_id - 1, 1.0, kCubic, Certify, &certification);
  ASSERT_TRUE(last.ok()) << last.status();
  EXPECT_EQ(last->particle.id, max_id - 1);
  EXPECT_EQ(last->next_particle_id, max_id);
}

}  // namespace
}  // namespace fluidity
