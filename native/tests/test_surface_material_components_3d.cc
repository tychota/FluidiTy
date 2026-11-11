#include <array>
#include <limits>

#include "fluidity/host/surface_material_components_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(SurfaceMaterialComponents3D, GroupsTransitiveSphereOverlapWithExactMaterialInventory) {
  const std::array particles{
      LiquidParticleState3D{.id = 9, .world_position = {8, 0, 0}, .reference_volume = 2, .volume_ratio = 0.5},
      LiquidParticleState3D{.id = 7, .world_position = {3, 0, 0}, .reference_volume = 2, .volume_ratio = 1.5},
      LiquidParticleState3D{.id = 5, .world_position = {0, 0, 0}, .reference_volume = 1, .volume_ratio = 0.75},
      LiquidParticleState3D{.id = 2, .world_position = {1.5, 0, 0}, .reference_volume = 4, .volume_ratio = 1.25},
  };
  auto components = PartitionSurfaceSphereUnion3D(particles, 1);
  ASSERT_TRUE(components.ok()) << components.status();
  ASSERT_EQ(components->size(), 2);
  const auto& connected = (*components)[0];
  EXPECT_EQ(connected.particle_indices, (std::vector<std::size_t>{3, 2, 1}));
  ASSERT_EQ(connected.members.size(), 3);
  EXPECT_EQ(connected.members[0].particle_id, 2);
  EXPECT_DOUBLE_EQ(connected.members[0].reference_volume_m3, 4);
  EXPECT_DOUBLE_EQ(connected.target_volume_m3, 8.75);
  EXPECT_EQ((*components)[1].particle_indices, (std::vector<std::size_t>{0}));
  EXPECT_DOUBLE_EQ((*components)[1].target_volume_m3, 1);
}

TEST(SurfaceMaterialComponents3D, TangencyDoesNotConnectMaterialInteriors) {
  std::array particles{
      LiquidParticleState3D{.id = 1, .world_position = {0, 0, 0}},
      LiquidParticleState3D{.id = 2, .world_position = {0, 2, 0}},
  };
  auto tangent = PartitionSurfaceSphereUnion3D(particles, 1);
  ASSERT_TRUE(tangent.ok());
  EXPECT_EQ(tangent->size(), 2);
  particles[1].world_position.y = 1.999;
  auto overlap = PartitionSurfaceSphereUnion3D(particles, 1);
  ASSERT_TRUE(overlap.ok());
  EXPECT_EQ(overlap->size(), 1);
}

TEST(SurfaceMaterialComponents3D, RejectsDuplicateIdsUnsupportedSidesAndExcessCapacity) {
  std::array particles{LiquidParticleState3D{.id = 1}, LiquidParticleState3D{.id = 1}};
  EXPECT_EQ(PartitionSurfaceSphereUnion3D(particles, 1).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(PartitionSurfaceSphereUnion3D(particles, 1, 1).status().code(), absl::StatusCode::kResourceExhausted);
  particles[1].id = 2;
  particles[1].side_tag = 3;
  EXPECT_EQ(PartitionSurfaceSphereUnion3D(particles, 1).status().code(), absl::StatusCode::kUnimplemented);
  particles[1].side_tag = 0;
  particles[1].volume_ratio = std::numeric_limits<double>::max();
  particles[1].reference_volume = 2;
  EXPECT_EQ(PartitionSurfaceSphereUnion3D(particles, 1).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(SurfaceMaterialComponents3D, DropletHysteresisUsesPhysicalDiameterAndPriorRepresentation) {
  using Kind = SurfaceComponentRepresentation3D;
  const auto classify = [](Kind previous, double diameter_m) {
    return ClassifySurfaceComponent3D(previous, diameter_m, 0.1);
  };
  EXPECT_EQ(classify(Kind::kMesh, 0.19).value(), Kind::kDroplet);
  EXPECT_EQ(classify(Kind::kMesh, 0.2).value(), Kind::kMesh);
  EXPECT_EQ(classify(Kind::kMesh, 0.25).value(), Kind::kMesh);
  EXPECT_EQ(classify(Kind::kDroplet, 0.25).value(), Kind::kDroplet);
  EXPECT_EQ(classify(Kind::kDroplet, 0.3).value(), Kind::kDroplet);
  EXPECT_EQ(classify(Kind::kDroplet, 0.31).value(), Kind::kMesh);
  EXPECT_EQ(classify(Kind::kMesh, -1).status().code(), absl::StatusCode::kInvalidArgument);
}
}  // namespace
}  // namespace fluidity
