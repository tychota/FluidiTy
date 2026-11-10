#include "fluidity/host/motion_carrier_3d.h"
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

}  // namespace
}  // namespace fluidity
