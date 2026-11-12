#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
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

auto Bits(Vec3 values) {
  return std::array{
      std::bit_cast<std::uint64_t>(values.x),
      std::bit_cast<std::uint64_t>(values.y),
      std::bit_cast<std::uint64_t>(values.z),
  };
}

std::vector<LiquidAdaptiveFaceState3D> MakeIndexedControlCarriers(std::span<const GradedMacFace3D> faces) {
  std::vector<LiquidAdaptiveFaceState3D> carriers;
  carriers.reserve(faces.size());
  for (const auto& face : faces) {
    const auto [x, y, z] = face.world_anchor;
    carriers.push_back({
        .carrier_velocity = x * y * 0.125 - y * z * 0.0625 + 0.25 * x + static_cast<double>(face.axis),
        .validity = LiquidFaceValidity3D::kValid,
        .structurally_eligible = z != 8,
    });
  }
  return carriers;
}

TEST(GradedMacTransfer3D, DistantSlotsPreserveCanonicalReductionBitsOnBothSidesOfJunction) {
  const auto local_topology = MixedTopology();
  ASSERT_TRUE(local_topology.ok()) << local_topology.status();
  std::vector<AdaptiveMacCellKey3D> leaves;
  for (const auto& leaf : local_topology->cells()) {
    leaves.push_back(leaf.key);
  }
  for (int index = 0; index < 32; ++index) {
    leaves.push_back({.lower_corner = {-64 + index * 4, 0, -24}, .width_exponent = 0});
    leaves.push_back({.lower_corner = {-64 + index * 4, 0, 24}, .width_exponent = 0});
  }
  const auto padded_topology = AdaptiveMacTopology3D::Create(leaves, 1);
  ASSERT_TRUE(padded_topology.ok()) << padded_topology.status();
  const Vec3 origin{10, -4, 8};
  const auto local = GradedMacTransfer3D::Create(*local_topology, origin);
  const auto padded = GradedMacTransfer3D::Create(*padded_topology, origin);
  ASSERT_TRUE(local.ok()) << local.status();
  ASSERT_TRUE(padded.ok()) << padded.status();
  auto local_carriers = MakeIndexedControlCarriers(local->faces());
  auto padded_carriers = MakeIndexedControlCarriers(padded->faces());
  const std::array particles{
      LiquidParticleState3D{
          .world_position = {9.75, -3.25, 9.25},
          .world_velocity = {0.75, -1.125, 4.25},
          .velocity_slopes = {{0.5, -3, 1}, {-1, 2, 3}, {4, 3, 2}},
          .mass = 0.375,
      },
      LiquidParticleState3D{
          .world_position = {10.25, -3.25, 9.25},
          .world_velocity = {-1.125, 2.25, -4},
          .velocity_slopes = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}},
          .mass = 2.625,
      },
  };
  for (const auto& particle : particles) {
    const auto local_result = local->Gather(particle, local_carriers);
    const auto padded_result = padded->Gather(particle, padded_carriers);
    ASSERT_TRUE(local_result.ok()) << local_result.status();
    ASSERT_TRUE(padded_result.ok()) << padded_result.status();
    EXPECT_EQ(local_result->affine_enabled, padded_result->affine_enabled);
    EXPECT_EQ(Bits(local_result->velocity), Bits(padded_result->velocity));
    EXPECT_EQ(Bits(local_result->velocity_slopes.x_row), Bits(padded_result->velocity_slopes.x_row));
    EXPECT_EQ(Bits(local_result->velocity_slopes.y_row), Bits(padded_result->velocity_slopes.y_row));
    EXPECT_EQ(Bits(local_result->velocity_slopes.z_row), Bits(padded_result->velocity_slopes.z_row));
  }
  const auto local_deposits = local->Deposit(particles, local_carriers);
  const auto padded_deposits = padded->Deposit(particles, padded_carriers);
  ASSERT_TRUE(local_deposits.ok()) << local_deposits.status();
  ASSERT_TRUE(padded_deposits.ok()) << padded_deposits.status();
  std::size_t local_slot = 0;
  for (std::size_t slot = 0; slot < padded->faces().size(); ++slot) {
    const auto& face = padded->faces()[slot];
    MacFaceDeposit3D expected{};
    if (face.world_anchor.z >= 8 && face.world_anchor.z <= 10) {
      ASSERT_LT(local_slot, local->faces().size());
      EXPECT_EQ(face.axis, local->faces()[local_slot].axis);
      EXPECT_EQ(Bits(face.world_anchor), Bits(local->faces()[local_slot].world_anchor));
      expected = (*local_deposits)[local_slot++];
    }
    EXPECT_EQ(
        std::bit_cast<std::uint64_t>((*padded_deposits)[slot].deposited_mass),
        std::bit_cast<std::uint64_t>(expected.deposited_mass)
    );
    EXPECT_EQ(
        std::bit_cast<std::uint64_t>((*padded_deposits)[slot].deposited_pure_momentum),
        std::bit_cast<std::uint64_t>(expected.deposited_pure_momentum)
    );
  }
  EXPECT_EQ(local_slot, local->faces().size());
  const std::size_t geometry_bytes = padded->faces().size() * sizeof(GradedMacFace3D)
      + padded_topology->cells().size() * (2 * sizeof(Vec3) + sizeof(double));
  EXPECT_GT(padded->reserved_bytes(), geometry_bytes);

  // Leave one supporting sample on each component to exercise the PIC path
  // while many otherwise-valid slots are masked or spatially pruned.
  const auto& particle = particles[1];
  const auto keep_one_per_axis = [&](std::span<const GradedMacFace3D> faces, auto& carriers) {
    std::array<bool, 3> kept{};
    for (std::size_t slot = 0; slot < faces.size(); ++slot) {
      const auto& face = faces[slot];
      const auto axis = static_cast<std::size_t>(face.axis);
      const Vec3 anchor = face.world_anchor;
      const bool nearby = std::abs(anchor.x - particle.world_position.x) <= 1
          && std::abs(anchor.y - particle.world_position.y) <= 1 && std::abs(anchor.z - particle.world_position.z) <= 1;
      if (nearby && carriers[slot].structurally_eligible && !kept[axis]) {
        kept[axis] = true;
      } else {
        carriers[slot].validity = LiquidFaceValidity3D::kMissing;
        carriers[slot].structurally_eligible = false;
      }
    }
    EXPECT_EQ(kept, (std::array{true, true, true}));
  };
  keep_one_per_axis(local->faces(), local_carriers);
  keep_one_per_axis(padded->faces(), padded_carriers);
  const auto local_pic = local->Gather(particle, local_carriers);
  const auto padded_pic = padded->Gather(particle, padded_carriers);
  ASSERT_TRUE(local_pic.ok()) << local_pic.status();
  ASSERT_TRUE(padded_pic.ok()) << padded_pic.status();
  EXPECT_FALSE(local_pic->affine_enabled);
  EXPECT_FALSE(padded_pic->affine_enabled);
  EXPECT_EQ(Bits(local_pic->velocity), Bits(padded_pic->velocity));
  EXPECT_EQ(Bits(local_pic->velocity_slopes.x_row), Bits(padded_pic->velocity_slopes.x_row));
  EXPECT_EQ(Bits(local_pic->velocity_slopes.y_row), Bits(padded_pic->velocity_slopes.y_row));
  EXPECT_EQ(Bits(local_pic->velocity_slopes.z_row), Bits(padded_pic->velocity_slopes.z_row));

  for (std::size_t slot = 0; slot < padded->faces().size(); ++slot) {
    if (padded->faces()[slot].axis == FaceNormalAxis::kX) {
      padded_carriers[slot].side_tag = 8;
    }
  }
  EXPECT_EQ(padded->Gather(particle, padded_carriers).status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(
      padded->Deposit(std::span(&particle, 1), padded_carriers).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(GradedMacTransfer3D, TestsCarrierAtRoundedSupportBoundary) {
  std::vector<AdaptiveMacCellKey3D> leaves;
  // The x-normal anchor at world x=13 begins the second canonical block.
  for (int x = -13; x <= 20; ++x) {
    leaves.push_back({.lower_corner = {x, 0, 0}, .width_exponent = 0});
  }
  const auto topology = AdaptiveMacTopology3D::Create(leaves, 1);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto transfer = GradedMacTransfer3D::Create(*topology, {10, 0, 0});
  ASSERT_TRUE(transfer.ok()) << transfer.status();
  auto carriers = MakeIndexedControlCarriers(transfer->faces());
  const auto boundary_face = std::ranges::find_if(transfer->faces(), [](const GradedMacFace3D& face) {
    return face.axis == FaceNormalAxis::kX && face.world_anchor.x == 13;
  });
  ASSERT_NE(boundary_face, transfer->faces().end());
  const std::size_t boundary_slot = static_cast<std::size_t>(boundary_face - transfer->faces().begin());
  carriers[boundary_slot].carrier_velocity = std::numeric_limits<double>::quiet_NaN();
  LiquidParticleState3D particle{.world_position = {11, 0.5, 0.5}};
  ASSERT_TRUE(transfer->Gather(particle, carriers).ok());
  particle.world_position.x = std::nextafter(11., 10.);
  ASSERT_TRUE(transfer->Gather(particle, carriers).ok());
  particle.world_position.x = std::nextafter(11., 12.);
  EXPECT_EQ(transfer->Gather(particle, carriers).status().code(), absl::StatusCode::kInvalidArgument);
}
}  // namespace
}  // namespace fluidity
