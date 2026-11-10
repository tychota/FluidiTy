#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "../src/liquid_isolated_drop_3d.h"
#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace fluidity::liquid_step_internal {
namespace {

constexpr double kCellWidth = 0.025;
constexpr double kPhysicalRadius = 0.125;
constexpr double kSeedCovarianceScale =
    kPhysicalRadius * kPhysicalRadius / (5.177536231884058 * kCellWidth * kCellWidth);

std::vector<LiquidParticleState3D> DropSeed() {
  std::vector<LiquidParticleState3D> particles;
  for (int z = -5; z < 5; ++z) {
    for (int y = -5; y < 5; ++y) {
      for (int x = -5; x < 5; ++x) {
        const double x_offset = x + 0.5;
        const double y_offset = y + 0.5;
        const double z_offset = z + 0.5;
        if (x_offset * x_offset + y_offset * y_offset + z_offset * z_offset < 25) {
          particles.push_back({
              .id = particles.size() + 1,
              .world_position =
                  {
                      kCellWidth * x_offset / std::sqrt(1.03),
                      kCellWidth * y_offset * 1.03,
                      kCellWidth * z_offset / std::sqrt(1.03),
                  },
              .mass = 0.01482107041435402,
              .reference_volume = 1.4821070414354021e-05,
          });
        }
      }
    }
  }
  return particles;
}

TEST(IsolatedDrop3D, AuthoredSeedRetainsWetCentersCornersAndCurvature) {
  const auto particles = DropSeed();
  ASSERT_EQ(particles.size(), 552);
  const auto fit = PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale);
  ASSERT_TRUE(fit.ok()) << fit.status();

  int wet_centers = 0;
  int wet_corners = 0;
  int air_links = 0;
  double minimum_curvature = std::numeric_limits<double>::infinity();
  double maximum_curvature = 0;
  constexpr std::array<Vec3, 6> kNeighbors{
      Vec3{-kCellWidth, 0, 0},
      Vec3{kCellWidth, 0, 0},
      Vec3{0, -kCellWidth, 0},
      Vec3{0, kCellWidth, 0},
      Vec3{0, 0, -kCellWidth},
      Vec3{0, 0, kCellWidth},
  };
  for (int z = -7; z <= 7; ++z) {
    for (int y = -7; y <= 7; ++y) {
      for (int x = -7; x <= 7; ++x) {
        const auto corner = fit->QuadraticGridLevelSet({kCellWidth * x, kCellWidth * y, kCellWidth * z});
        ASSERT_TRUE(corner.ok()) << corner.status();
        wet_corners += *corner < 0;
        const Vec3 center{kCellWidth * (x + 0.5), kCellWidth * (y + 0.5), kCellWidth * (z + 0.5)};
        const auto level = fit->QuadraticGridLevelSet(center);
        ASSERT_TRUE(level.ok()) << level.status();
        if (*level >= 0) {
          continue;
        }
        ++wet_centers;
        for (const Vec3 offset : kNeighbors) {
          const Vec3 neighbor{center.x + offset.x, center.y + offset.y, center.z + offset.z};
          const auto neighbor_level = fit->QuadraticGridLevelSet(neighbor);
          ASSERT_TRUE(neighbor_level.ok()) << neighbor_level.status();
          if (*neighbor_level >= 0) {
            const auto curvature = fit->AirLinkCurvature(center, neighbor);
            ASSERT_TRUE(curvature.ok()) << curvature.status();
            ++air_links;
            minimum_curvature = std::min(minimum_curvature, *curvature);
            maximum_curvature = std::max(maximum_curvature, *curvature);
          }
        }
      }
    }
  }
  EXPECT_EQ(wet_centers, 528);
  EXPECT_EQ(wet_corners, 511);
  EXPECT_EQ(air_links, 472);
  EXPECT_NEAR(minimum_curvature, 15.561424751112174, 5e-10);
  EXPECT_NEAR(maximum_curvature, 16.94204543523493, 5e-10);
}

TEST(IsolatedDrop3D, UsesMassAndFixedSeedScaleForRotatedDrop) {
  auto particles = DropSeed();
  const auto transform = [](Vec3 point) {
    return Vec3{0.02 + 0.6 * point.x - 0.8 * point.y, 0.03 + 0.8 * point.x + 0.6 * point.y, -0.04 + point.z};
  };
  for (auto& particle : particles) {
    particle.volume_ratio = particle.world_position.y > 0 ? 1.2 : 0.9;
    particle.world_position = transform(particle.world_position);
  }
  const auto fit = PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale);
  ASSERT_TRUE(fit.ok()) << fit.status();
  const auto pole = fit->QuadraticGridLevelSet(transform({0, 0.12875, 0}));
  ASSERT_TRUE(pole.ok()) << pole.status();
  EXPECT_NEAR(*pole, 0, 2e-15);
  const auto curvature = fit->AirLinkCurvature(transform({0, 0.12, 0}), transform({0, 0.14, 0}));
  ASSERT_TRUE(curvature.ok()) << curvature.status();
  EXPECT_NEAR(*curvature, 16.9744, 5e-10);
}

TEST(IsolatedDrop3D, ParticleDistanceDistinguishesTheBetaBandAndMedialAxis) {
  const auto particles = DropSeed();
  const auto fit = PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale);
  ASSERT_TRUE(fit.ok()) << fit.status();
  const auto center_distance = fit->ParticleSignedDistance({0, 0, 0});
  const auto inner_distance = fit->ParticleSignedDistance({0, 0.103, 0});
  const auto outer_distance = fit->ParticleSignedDistance({0, 0.14, 0});
  const auto inner_level = fit->QuadraticGridLevelSet({0, 0.103, 0});
  ASSERT_TRUE(center_distance.ok()) << center_distance.status();
  ASSERT_TRUE(inner_distance.ok()) << inner_distance.status();
  ASSERT_TRUE(outer_distance.ok()) << outer_distance.status();
  ASSERT_TRUE(inner_level.ok()) << inner_level.status();
  EXPECT_NEAR(*center_distance, -0.12316615977053665, 2e-15);
  EXPECT_NEAR(*inner_distance, -0.02575, 2e-15);
  EXPECT_NEAR(*outer_distance, 0.01125, 2e-15);
  EXPECT_LT(*inner_distance, -kCellWidth);
  EXPECT_GE(*inner_level, -kCellWidth);
  EXPECT_EQ(fit->AirLinkCurvature({0, 0, 0}, {0, 0.1, 0}).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(IsolatedDrop3D, RejectsMalformedAndDisconnectedInputs) {
  auto particles = DropSeed();
  particles.back().mass = 0;
  EXPECT_EQ(
      PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  particles = DropSeed();
  particles.back().world_position = particles.front().world_position;
  EXPECT_EQ(
      PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  particles = DropSeed();
  particles.back().world_position.x += 1;
  EXPECT_EQ(
      PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(IsolatedDrop3D, RejectsConnectedPlanarCovariance) {
  const std::array particles{
      LiquidParticleState3D{.world_position = {0, 0, 0}},
      LiquidParticleState3D{.world_position = {kCellWidth, 0, 0}},
      LiquidParticleState3D{.world_position = {0, kCellWidth, 0}},
      LiquidParticleState3D{.world_position = {kCellWidth, kCellWidth, 0}},
  };
  EXPECT_EQ(
      PrepareIsolatedDrop3D(particles, kCellWidth, kPhysicalRadius, kSeedCovarianceScale).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

}  // namespace
}  // namespace fluidity::liquid_step_internal
