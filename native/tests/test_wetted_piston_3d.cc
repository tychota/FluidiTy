#include "wetted_piston_3d_cases.h"

namespace fluidity {
namespace {
TEST(WettedPiston3D, PlanarPressurePatchesUseWetCentroidAndSelectedCell) {
  auto workspace = WettedPistonWorkspace3D::Create(wetted_piston_test::kCapacity);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  wetted_piston_test::Fixture fixture;
  fixture.tank.floor_y = 0.125;
  fixture.tank.low_z = 0.25;
  fixture.tank.high_z = 0.875;
  std::vector<RigidPressurePatch3D> patches;
  const auto wet = fixture.Run(
      [&patches](const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& scratch)
          -> absl::StatusOr<WettedPistonView3D> {
        const auto result = ComputeWettedPiston3D(inputs, scratch);
        if (!result.ok()) {
          return result.status();
        }
        const auto built = BuildPlanarWettedPistonPressurePatches3D(inputs, *result, 17);
        if (!built.ok()) {
          return built.status();
        }
        patches = *built;
        return *result;
      },
      LiquidHorizontalPlaneSurface3D{0.625},
      {},
      *workspace
  );
  ASSERT_TRUE(wet.ok()) << wet.status();
  ASSERT_EQ(patches.size(), 1);
  EXPECT_EQ(patches.front().pressure_cell_index, 0);
  EXPECT_DOUBLE_EQ(patches.front().wetted_area, 0.3125);
  EXPECT_DOUBLE_EQ(patches.front().world_centroid.y, 0.375);
  EXPECT_DOUBLE_EQ(patches.front().world_centroid.z, 0.5625);
  const RigidBody3D body{.body_id = 17, .mass = 1.0, .principal_inertia = {1.0, 1.0, 1.0}};
  const std::array pressure{8.0};
  const auto loaded = AccumulateRigidBodyPressure3D(body, pressure, 0.0, patches);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_DOUBLE_EQ(loaded->force.x, -2.5);
  EXPECT_DOUBLE_EQ(loaded->torque.y, -1.40625);
  EXPECT_DOUBLE_EQ(loaded->torque.z, 0.9375);
}

TEST(WettedPiston3D, ClipsHorizontalPlaneAndKeepsBothFluxSigns) {
  wetted_piston_test::PlaneClippingAndSignedFlux(ComputeWettedPiston3D, 0);
}
TEST(WettedPiston3D, EnclosesThinOverlappingCapAndAttachesOnce) {
  wetted_piston_test::OverlappingThinCapEnclosesAnalyticArea(ComputeWettedPiston3D, 0);
}
TEST(WettedPiston3D, CertifiesTangencyAndRejectsUncertainPositiveArea) {
  wetted_piston_test::TangencyAndUncertifiableThinWall(ComputeWettedPiston3D, 0);
}
TEST(WettedPiston3D, RejectsDisconnectedAndUnownedWetRows) {
  wetted_piston_test::RefusesUnsupportedOwnershipAndKeepsPublication(ComputeWettedPiston3D);
}
TEST(WettedPiston3D, UsesSignedHalfOpenPistonCellAndChecksQueryCapacity) {
  wetted_piston_test::PreservesSignedLatticeAndChecksCapacity(ComputeWettedPiston3D, 0);
}
}  // namespace
}  // namespace fluidity
