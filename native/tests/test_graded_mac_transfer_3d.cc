#include <array>
#include <cmath>
#include <vector>

#include "fluidity/host/graded_mac_transfer_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

auto MixedTopology() {
  std::array<AdaptiveMacCellKey3D, 9> leaves{};
  leaves[0] = {.lower_corner = {-2, 0, 0}, .width_exponent = 1};
  std::size_t ordinal = 1;
  for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        leaves[ordinal++] = {.lower_corner = {x, y, z}, .width_exponent = 0};
      }
    }
  }
  return AdaptiveMacTopology3D::Create(leaves, 1);
}

TEST(GradedMacTransfer3D, ConservesComponentMassAndMomentumAtSignedJunction) {
  const auto topology = MixedTopology();
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto transfer = GradedMacTransfer3D::Create(*topology, {0, 0, 0});
  ASSERT_TRUE(transfer.ok()) << transfer.status();
  std::vector<LiquidAdaptiveFaceState3D> faces(transfer->faces().size());
  const std::array particles{LiquidParticleState3D{
      .world_position = {-0.25, 0.75, 1.25},
      .world_velocity = {1.5, -2, 0.125},
      .velocity_slopes = {{1, 2, 3}, {2, -1, 3}, {1, 1, 1}},
      .mass = 3,
  }};
  const auto deposits = transfer->Deposit(particles, faces);
  ASSERT_TRUE(deposits.ok()) << deposits.status();
  std::array<double, 3> mass{};
  std::array<double, 3> momentum{};
  for (std::size_t slot = 0; slot < deposits->size(); ++slot) {
    const auto axis = static_cast<std::size_t>(transfer->faces()[slot].axis);
    mass[axis] += (*deposits)[slot].deposited_mass;
    momentum[axis] += (*deposits)[slot].deposited_pure_momentum;
  }
  for (double component_mass : mass) {
    EXPECT_NEAR(component_mass, 3, 1e-12);
  }
  EXPECT_NEAR(momentum[0], 4.5, 1e-12);
  EXPECT_NEAR(momentum[1], -6, 1e-12);
  EXPECT_NEAR(momentum[2], 0.375, 1e-12);
  const auto replayed = transfer->Deposit(particles, faces);
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  for (std::size_t slot = 0; slot < deposits->size(); ++slot) {
    EXPECT_DOUBLE_EQ((*deposits)[slot].deposited_mass, (*replayed)[slot].deposited_mass);
    EXPECT_DOUBLE_EQ((*deposits)[slot].deposited_pure_momentum, (*replayed)[slot].deposited_pure_momentum);
  }
}

TEST(GradedMacTransfer3D, ReproducesAffineVelocityOnActualSubfaceAnchors) {
  const auto topology = MixedTopology();
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto transfer = GradedMacTransfer3D::Create(*topology, {4, -2, 8});
  ASSERT_TRUE(transfer.ok()) << transfer.status();
  std::vector<LiquidAdaptiveFaceState3D> faces(transfer->faces().size());
  const Vec3 particle_position{3.75, -1.25, 9.25};
  for (std::size_t slot = 0; slot < faces.size(); ++slot) {
    const auto& geometry = transfer->faces()[slot];
    const auto anchor = geometry.world_anchor;
    const auto axis = static_cast<double>(geometry.axis);
    faces[slot] = {
        .carrier_velocity = axis + 2 * (anchor.x - particle_position.x) - 3 * (anchor.y - particle_position.y)
            + 0.25 * (anchor.z - particle_position.z),
        .validity = LiquidFaceValidity3D::kValid,
    };
  }
  const auto gathered = transfer->Gather({.world_position = particle_position}, faces);
  ASSERT_TRUE(gathered.ok()) << gathered.status();
  EXPECT_TRUE(gathered->affine_enabled);
  EXPECT_NEAR(gathered->velocity.x, 0, 1e-12);
  EXPECT_NEAR(gathered->velocity.y, 1, 1e-12);
  EXPECT_NEAR(gathered->velocity.z, 2, 1e-12);
  for (const auto slopes :
       {gathered->velocity_slopes.x_row, gathered->velocity_slopes.y_row, gathered->velocity_slopes.z_row}) {
    EXPECT_NEAR(slopes.x, 2, 1e-12);
    EXPECT_NEAR(slopes.y, -3, 1e-12);
    EXPECT_NEAR(slopes.z, 0.25, 1e-12);
  }
  faces.front().validity = LiquidFaceValidity3D::kLowMass;
  const auto low_mass = transfer->Gather({.world_position = particle_position}, faces);
  ASSERT_TRUE(low_mass.ok()) << low_mass.status();
  EXPECT_FALSE(low_mass->affine_enabled);
  EXPECT_DOUBLE_EQ(low_mass->velocity_slopes.x_row.x, 0);
}

TEST(GradedMacTransfer3D, RefusesUnrepresentableTranslatedSubfaceCenter) {
  const std::array keys{AdaptiveMacCellKey3D{.lower_corner = {}, .width_exponent = 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 2);
  ASSERT_TRUE(topology.ok()) << topology.status();
  EXPECT_EQ(
      GradedMacTransfer3D::Create(*topology, {std::ldexp(1., 53), 0, 0}).status().code(),
      absl::StatusCode::kOutOfRange
  );
}
TEST(GradedMacTransfer3D, RankLossUsesPicAndMissingComponentFailsClosed) {
  const std::array keys{AdaptiveMacCellKey3D{.lower_corner = {}, .width_exponent = 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto transfer = GradedMacTransfer3D::Create(*topology, {});
  ASSERT_TRUE(transfer.ok()) << transfer.status();
  std::vector<LiquidAdaptiveFaceState3D> faces(transfer->faces().size());
  for (std::size_t slot = 0; slot < faces.size(); ++slot) {
    faces[slot] = {
        .carrier_velocity = 2 + static_cast<double>(transfer->faces()[slot].axis),
        .validity = LiquidFaceValidity3D::kPredicted,
        .side_tag = 8,
    };
  }
  const LiquidParticleState3D particle{.world_position = {0.5, 0.5, 0.5}, .side_tag = 8};
  const auto gathered = transfer->Gather(particle, faces);
  ASSERT_TRUE(gathered.ok()) << gathered.status();
  EXPECT_FALSE(gathered->affine_enabled);
  EXPECT_DOUBLE_EQ(gathered->velocity.x, 2);
  EXPECT_DOUBLE_EQ(gathered->velocity.y, 3);
  EXPECT_DOUBLE_EQ(gathered->velocity.z, 4);
  EXPECT_DOUBLE_EQ(gathered->velocity_slopes.x_row.x, 0);
  for (std::size_t slot = 0; slot < faces.size(); ++slot) {
    if (transfer->faces()[slot].axis == FaceNormalAxis::kX) {
      faces[slot].side_tag = 9;
    }
  }
  EXPECT_EQ(transfer->Gather(particle, faces).status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(transfer->Deposit(std::span(&particle, 1), faces).status().code(), absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace fluidity
