#include <array>
#include <vector>

#include "fluidity/host/graded_mac_transfer_3d.h"
#include "fluidity/host/motion_carrier_3d.h"
#include "graded_tank_fixture.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(MotionCarrier3D, ReconstructsInitialConstantVelocityAndRejectsMissingHalo) {
  LiquidState3D seed;
  seed.geometry.cell_width = 0.1;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(1);
  ASSERT_TRUE(particles.ok());
  ASSERT_TRUE(particles->Resize(1).ok());
  (*particles)[0] = {
      .id = 1,
      .world_position = {-0.05, 0.05, 0.15},
      .world_velocity = {1, -2, 3},
  };
  seed.particles = std::move(*particles);
  seed.next_particle_id = 2;
  auto carrier = FrozenMotionCarrier3D::Create(seed, {}, {});
  ASSERT_TRUE(carrier.ok()) << carrier.status();
  const auto sampled = carrier->Sample({-0.04, 0.06, 0.14});
  ASSERT_TRUE(sampled.ok()) << sampled.status();
  EXPECT_NEAR(sampled->velocity.x, 1, 1e-10);
  EXPECT_NEAR(sampled->velocity.y, -2, 1e-10);
  EXPECT_NEAR(sampled->velocity.z, 3, 1e-10);
  EXPECT_EQ(carrier->Sample({20, 20, 20}).status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(carrier->Sample({-0.05, 0.05, 0.15}, 42).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(MotionCarrier3D, EmptyCarrierCannotBecomeAValidZero) {
  LiquidState3D empty;
  auto carrier = FrozenMotionCarrier3D::Create(empty, {}, {});
  ASSERT_TRUE(carrier.ok()) << carrier.status();
  EXPECT_EQ(carrier->Sample({}).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(MotionCarrier3D, SamplesAcceptedGradedFacesAtTheInterfaceAndOpenTop) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto populated = test::PopulateGradedTankFixture3D(state, scene);
  ASSERT_TRUE(populated.ok()) << populated;
  state.clock = {.time = 1.0 / 24, .frame_ordinal = 1, .accepted_step_ordinal = 1, .epoch = 1};
  std::vector<AdaptiveMacCellKey3D> keys;
  for (const auto& cell : state.adaptive_cells.values()) {
    keys.push_back(cell.key);
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto geometry = GradedMacTransfer3D::Create(*topology, state.geometry.world_origin);
  ASSERT_TRUE(geometry.ok()) << geometry.status();
  const auto affine = [](Vec3 p) {
    return Vec3{1 + p.x + 2 * p.y - p.z, -2 + p.x - p.y + p.z, 3 + 2 * p.x - p.y + 2 * p.z};
  };
  for (std::size_t slot = 0; slot < geometry->faces().size(); ++slot) {
    const auto& face = geometry->faces()[slot];
    const auto velocity = affine(face.world_anchor);
    const std::array components{velocity.x, velocity.y, velocity.z};
    state.adaptive_faces[slot].carrier_velocity = components[static_cast<std::size_t>(face.axis)];
  }
  const auto carrier = FrozenMotionCarrier3D::Create(state, scene, {});
  ASSERT_TRUE(carrier.ok()) << carrier.status();
  for (const Vec3 position :
       {Vec3{0.99, 0.49, 0.49}, Vec3{1.01, 0.51, 0.51}, Vec3{1.1, 1.0, 0.4}, Vec3{1.1, 1.125, 0.4}}) {
    const auto sampled = carrier->Sample(position);
    ASSERT_TRUE(sampled.ok()) << sampled.status();
    const auto expected = affine(position);
    EXPECT_NEAR(sampled->velocity.x, expected.x, 1e-11);
    EXPECT_NEAR(sampled->velocity.y, expected.y, 1e-11);
    EXPECT_NEAR(sampled->velocity.z, expected.z, 1e-11);
    EXPECT_EQ(sampled->continued, position.y >= 1.0);
    EXPECT_FALSE(sampled->rank_fallback);
  }
  EXPECT_EQ(carrier->Sample({1.1, 1.251, 0.4}).status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(carrier->Sample({1.1, 0.5, 0.4}, 42).status().code(), absl::StatusCode::kFailedPrecondition);

  for (std::size_t slot = 0; slot < geometry->faces().size(); ++slot) {
    if (geometry->faces()[slot].axis == FaceNormalAxis::kX) {
      state.adaptive_faces[slot].validity = LiquidFaceValidity3D::kLowMass;
    }
  }
  const auto missing = FrozenMotionCarrier3D::Create(state, scene, {});
  ASSERT_TRUE(missing.ok()) << missing.status();
  EXPECT_EQ(missing->Sample({1.1, 0.5, 0.4}).status().code(), absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace fluidity
