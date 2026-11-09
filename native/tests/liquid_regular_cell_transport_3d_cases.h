#pragma once

#include <array>
#include <cmath>
#include <limits>

#include "fluidity/host/liquid_substep_backend_3d.h"
#include "gtest/gtest.h"

namespace fluidity::test_internal {
struct RegularCellUpdateFixture3D {
  std::array<LiquidParticleState3D, 2> particles{};
  std::array<LiquidBoundaryParticleGather3D, 2> gathered{};
  std::array<double, 2> distances{-0.5, -0.5};
  std::array<LiquidRegularCellTransport3D, 2> transport{};
  LiquidParticleUpdateInputs3D inputs{gathered, distances, {}, 2, 0.125, 1, 0, 0, transport};

  RegularCellUpdateFixture3D() {
    particles[0].world_position = {2.5, 3, 3.5};
    particles[0].world_velocity = {1, 2, 3};
    particles[0].volume_ratio = 1.25;
    particles[0].pic_initialized = true;
    particles[1] = particles[0];
    gathered[0].new_pic_velocity = {0.25, 0.5, -0.5};
    gathered[0].particle_minus_old_pic_velocity = {0.5, 0.25, 0.5};
    gathered[0].fitted_velocity_slopes = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    gathered[0].velocity_gradient.x_row.x = 1;
    gathered[1] = gathered[0];
    transport[0] = {
        .active = true,
        .position_in_cell_widths = {0.25, 0.5, 0.75},
        .low_face_normal_velocity = {-0.5, 1, 2},
        .high_face_normal_velocity = {3.5, -5, 0},
    };
    transport[1].position_in_cell_widths.x = std::numeric_limits<double>::quiet_NaN();
  }
};

template <typename Update>
void CheckRegularCellTransport(Update update) {
  RegularCellUpdateFixture3D fixture;
  const auto initial = fixture.particles;
  // At h=2 the normal derivatives are (2,-3,-1). The local velocity is
  // (0.5,-2,0.5), while the inactive entry uses the expanding PIC+residual field.
  ASSERT_TRUE(update(fixture.inputs, fixture.particles).ok());
  const auto& active = fixture.particles[0];
  EXPECT_DOUBLE_EQ(active.world_position.x, 2.5625);
  EXPECT_DOUBLE_EQ(active.world_position.y, 2.75);
  EXPECT_DOUBLE_EQ(active.world_position.z, 3.5625);
  EXPECT_NEAR(active.volume_ratio, 1.25 * std::exp(-0.25), 2e-14);
  EXPECT_NEAR(active.world_velocity.x, 0.745, 2e-14);
  EXPECT_NEAR(active.world_velocity.y, 0.7475, 2e-14);
  EXPECT_NEAR(active.world_velocity.z, -0.005, 2e-14);
  const auto& slopes = active.velocity_slopes;
  EXPECT_EQ((std::array{slopes.x_row.x, slopes.x_row.y, slopes.x_row.z}), (std::array{1., 2., 3.}));
  EXPECT_EQ((std::array{slopes.y_row.x, slopes.y_row.y, slopes.y_row.z}), (std::array{4., 5., 6.}));
  EXPECT_EQ((std::array{slopes.z_row.x, slopes.z_row.y, slopes.z_row.z}), (std::array{7., 8., 9.}));
  EXPECT_NEAR(fixture.particles[1].world_position.x, 2.593125, 2e-14);
  EXPECT_NEAR(fixture.particles[1].volume_ratio, 1.25 * std::exp(0.125), 2e-14);
  const auto inactive = fixture.particles[1];

  fixture.particles = initial;
  // Reversing the Z derivative makes the trace exactly zero.
  fixture.transport[0].high_face_normal_velocity.z = 4;
  ASSERT_TRUE(update(fixture.inputs, fixture.particles).ok());
  EXPECT_DOUBLE_EQ(fixture.particles[0].world_position.z, 3.9375);
  EXPECT_DOUBLE_EQ(fixture.particles[0].volume_ratio, 1.25);

  // The next dispatch on the same backend must not reuse a staged selection.
  fixture.particles = initial;
  fixture.inputs.regular_cell_transport = {};
  ASSERT_TRUE(update(fixture.inputs, fixture.particles).ok());
  EXPECT_EQ(fixture.particles[0], inactive);
  EXPECT_EQ(fixture.particles[1], inactive);
}

template <typename Update>
void CheckRegularCellTransportContact(Update update) {
  RegularCellUpdateFixture3D fixture;
  fixture.inputs.obstacle = AxisAlignedLiquidTank3D{
      .start_low_x = 0,
      .high_x = 6,
      .floor_y = 0,
      .low_z = 0,
      .high_z = 6,
      .piston_velocity_x = 0,
  };
  fixture.particles[0].world_position.y = 0.125;
  fixture.transport[0].position_in_cell_widths.y = 0.0625;
  fixture.transport[0].low_face_normal_velocity.y = 0;
  fixture.transport[0].high_face_normal_velocity.y = -32;
  // The selected velocity reaches the floor after 1/16 s. The outgoing stored
  // Y velocity survives contact and the selected trace is 2-16-1 = -15/s.
  ASSERT_TRUE(update(fixture.inputs, fixture.particles).ok());
  EXPECT_DOUBLE_EQ(fixture.particles[0].world_position.x, 2.5625);
  EXPECT_DOUBLE_EQ(fixture.particles[0].world_position.y, 0);
  EXPECT_DOUBLE_EQ(fixture.particles[0].world_position.z, 3.5625);
  EXPECT_NEAR(fixture.particles[0].world_velocity.y, 0.7475, 2e-14);
  EXPECT_NEAR(fixture.particles[0].volume_ratio, 1.25 * std::exp(-1.875), 2e-14);
}

template <typename Update>
void CheckMalformedRegularCellTransport(Update update) {
  RegularCellUpdateFixture3D fixture;
  const auto initial = fixture.particles;
  fixture.inputs.regular_cell_transport = std::span(fixture.transport).first(1);
  EXPECT_EQ(update(fixture.inputs, fixture.particles).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(fixture.particles, initial);
  fixture.particles = initial;
  fixture.inputs.regular_cell_transport = fixture.transport;

  const auto reject_late_sample = [&](LiquidRegularCellTransport3D sample, absl::StatusCode code) {
    fixture.transport[1] = sample;
    const auto status = update(fixture.inputs, fixture.particles);
    EXPECT_EQ(status.code(), code);
    EXPECT_TRUE(status.message().empty());
    EXPECT_EQ(fixture.particles, initial);
    fixture.particles = initial;
  };
  auto malformed = fixture.transport[0];
  malformed.position_in_cell_widths.x = -0.01;
  reject_late_sample(malformed, absl::StatusCode::kInvalidArgument);
  malformed = fixture.transport[0];
  malformed.position_in_cell_widths.z = 1.01;
  reject_late_sample(malformed, absl::StatusCode::kInvalidArgument);
  malformed = fixture.transport[0];
  malformed.low_face_normal_velocity.y = std::numeric_limits<double>::quiet_NaN();
  reject_late_sample(malformed, absl::StatusCode::kInvalidArgument);
  malformed = fixture.transport[0];
  malformed.high_face_normal_velocity.z = std::numeric_limits<double>::infinity();
  reject_late_sample(malformed, absl::StatusCode::kInvalidArgument);
  malformed = fixture.transport[0];
  malformed.low_face_normal_velocity.x = -std::numeric_limits<double>::max();
  malformed.high_face_normal_velocity.x = std::numeric_limits<double>::max();
  reject_late_sample(malformed, absl::StatusCode::kOutOfRange);

  // The sampled velocity stays small and the divergence stays finite, but J
  // cannot represent the exponential growth. Keep the physical retry status.
  malformed = fixture.transport[0];
  malformed.position_in_cell_widths.x = 0;
  malformed.high_face_normal_velocity.x = 16000;
  reject_late_sample(malformed, absl::StatusCode::kAborted);
}

template <typename Update>
void CheckAliasedRegularCellTransport(Update update) {
  RegularCellUpdateFixture3D fixture;
  union {
    LiquidParticleState3D particle{};
    LiquidRegularCellTransport3D transport;
  } aliased;
  const auto initial = aliased.particle;
  fixture.inputs.gathered = std::span(fixture.gathered).first(1);
  fixture.inputs.liquid_distances = std::span(fixture.distances).first(1);
  fixture.inputs.regular_cell_transport = std::span(&aliased.transport, 1);
  EXPECT_EQ(update(fixture.inputs, std::span(&aliased.particle, 1)).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(aliased.particle, initial);
}
}  // namespace fluidity::test_internal
