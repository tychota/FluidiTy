#include <array>

#include "fluidity/host/surface_component_lineage_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(SurfaceComponentLineage3D, SplitKeepsIdentityOnLargestOverlapAndPreservesEachMaterialTarget) {
  constexpr std::array members{
      SurfaceComponentMember3D{.particle_id = 1, .reference_volume_m3 = 1},
      SurfaceComponentMember3D{.particle_id = 2, .reference_volume_m3 = 3},
      SurfaceComponentMember3D{.particle_id = 3, .reference_volume_m3 = 2},
  };
  const std::array previous{SurfaceComponentHistory3D{.component_id = 7, .members = members}};
  const std::array current{
      SurfaceComponentMaterial3D{.members = std::span(members).first(1), .target_volume_m3 = 1.25},
      SurfaceComponentMaterial3D{.members = std::span(members).subspan(1), .target_volume_m3 = 4.75},
  };
  auto lineage = MatchSurfaceComponentLineage3D(previous, current, 8);
  ASSERT_TRUE(lineage.ok()) << lineage.status();
  ASSERT_EQ(lineage->components.size(), 2);
  EXPECT_EQ(lineage->components[0].component_id, 8);
  EXPECT_EQ(lineage->components[1].component_id, 7);
  EXPECT_EQ(lineage->next_component_id, 9);
  EXPECT_TRUE(lineage->retired_component_ids.empty());
  EXPECT_DOUBLE_EQ(lineage->components[0].target_volume_m3, 1.25);
  EXPECT_DOUBLE_EQ(lineage->components[1].target_volume_m3, 4.75);
  ASSERT_EQ(lineage->components[0].parents.size(), 1);
  ASSERT_EQ(lineage->components[1].parents.size(), 1);
  EXPECT_EQ(lineage->components[0].parents[0].parent_component_id, 7);
  EXPECT_DOUBLE_EQ(lineage->components[0].parents[0].shared_reference_volume_m3, 1);
  EXPECT_DOUBLE_EQ(lineage->components[1].parents[0].shared_reference_volume_m3, 5);
}

TEST(SurfaceComponentLineage3D, MergeRetiresOneIdentityAndReportsBothParentsInOrder) {
  constexpr std::array members{
      SurfaceComponentMember3D{.particle_id = 1, .reference_volume_m3 = 2},
      SurfaceComponentMember3D{.particle_id = 2, .reference_volume_m3 = 2},
  };
  const std::array previous{
      SurfaceComponentHistory3D{.component_id = 4, .members = std::span(members).last(1)},
      SurfaceComponentHistory3D{.component_id = 9, .members = std::span(members).first(1)},
  };
  const std::array current{SurfaceComponentMaterial3D{.members = members, .target_volume_m3 = 3.5}};
  auto lineage = MatchSurfaceComponentLineage3D(previous, current, 10);
  ASSERT_TRUE(lineage.ok()) << lineage.status();
  ASSERT_EQ(lineage->components.size(), 1);
  EXPECT_EQ(lineage->components[0].component_id, 4);
  EXPECT_EQ(lineage->next_component_id, 10);
  EXPECT_EQ(lineage->retired_component_ids, (std::vector<std::uint64_t>{9}));
  ASSERT_EQ(lineage->components[0].parents.size(), 2);
  EXPECT_EQ(lineage->components[0].parents[0].parent_component_id, 4);
  EXPECT_EQ(lineage->components[0].parents[1].parent_component_id, 9);
  EXPECT_DOUBLE_EQ(lineage->components[0].target_volume_m3, 3.5);
}

TEST(SurfaceComponentLineage3D, DisjointBirthAndDeathNeverReuseAnIdentity) {
  constexpr std::array members{
      SurfaceComponentMember3D{.particle_id = 1, .reference_volume_m3 = 2},
      SurfaceComponentMember3D{.particle_id = 2, .reference_volume_m3 = 2},
  };
  const std::array previous{SurfaceComponentHistory3D{.component_id = 4, .members = std::span(members).first(1)}};
  const std::array current{
      SurfaceComponentMaterial3D{.members = std::span(members).last(1), .target_volume_m3 = 2},
  };
  auto lineage = MatchSurfaceComponentLineage3D(previous, current, 5);
  ASSERT_TRUE(lineage.ok()) << lineage.status();
  EXPECT_EQ(lineage->components[0].component_id, 5);
  EXPECT_TRUE(lineage->components[0].parents.empty());
  EXPECT_EQ(lineage->retired_component_ids, (std::vector<std::uint64_t>{4}));
  EXPECT_EQ(lineage->next_component_id, 6);
}

TEST(SurfaceComponentLineage3D, RejectsAliasedMembershipChangedReferenceVolumeAndCapacity) {
  constexpr std::array members{
      SurfaceComponentMember3D{.particle_id = 1, .reference_volume_m3 = 1},
      SurfaceComponentMember3D{.particle_id = 2, .reference_volume_m3 = 1},
      SurfaceComponentMember3D{.particle_id = 3, .reference_volume_m3 = 1},
  };
  const std::array previous{SurfaceComponentHistory3D{.component_id = 1, .members = members}};
  const std::array aliased{
      SurfaceComponentMaterial3D{.members = std::span(members).first(2), .target_volume_m3 = 2},
      SurfaceComponentMaterial3D{.members = std::span(members).subspan(1), .target_volume_m3 = 2},
  };
  EXPECT_EQ(MatchSurfaceComponentLineage3D(previous, aliased, 2).status().code(), absl::StatusCode::kInvalidArgument);
  auto changed = members;
  changed[1].reference_volume_m3 = 1.5;
  const std::array current{SurfaceComponentMaterial3D{.members = changed, .target_volume_m3 = 3.5}};
  EXPECT_EQ(
      MatchSurfaceComponentLineage3D(previous, current, 2).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(
      MatchSurfaceComponentLineage3D(previous, current, 2, 2).status().code(),
      absl::StatusCode::kResourceExhausted
  );
}
}  // namespace
}  // namespace fluidity
