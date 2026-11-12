#include "fluidity/host/gas_checkpoint_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(GasCheckpoint3D, RestoresCellPressureScalarsAndApicHistory) {
  const auto domain = CreateGasUniformDomain3D({
      .geometry = {{0.0, 0.0, 0.0}, 1.0},
      .extent = {1, 1, 1},
      .open_boundary = {},
      .ambient = {.carrier = 1.2, .enthalpy = 354637.5},
      .ambient_pressure = 101325.0,
      .initial_velocity = {},
  });
  ASSERT_TRUE(domain.ok()) << domain.status();
  GasCheckpoint3D checkpoint{
      .frame = 1,
      .time = 0.05,
      .accepted_steps = 2,
      .state =
          {
              .cells = domain->cells,
              .markers = domain->markers,
              .face_velocities = domain->face_velocities,
              .fuel_burn_rate = {0.125},
          },
  };
  checkpoint.state.cells[0].absolute_pressure = 101500.0;
  checkpoint.state.cells[0].density.fuel = 0.25;
  checkpoint.state.markers[0].velocity_slopes.x_row.y = 0.375;
  checkpoint.state.face_velocities[0] = 0.5;
  LiquidDigest3D scene_digest{};
  scene_digest[4] = 23;
  const auto encoded = EncodeGasCheckpoint3D(checkpoint, scene_digest);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeGasCheckpoint3D(*encoded, *domain, scene_digest);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->frame, 1);
  EXPECT_EQ(decoded->time, 0.05);
  EXPECT_EQ(decoded->accepted_steps, 2);
  EXPECT_EQ(decoded->state.cells[0].absolute_pressure, 101500.0);
  EXPECT_EQ(decoded->state.cells[0].density.fuel, 0.25);
  EXPECT_EQ(decoded->state.markers[0].velocity_slopes.x_row.y, 0.375);
  EXPECT_EQ(decoded->state.face_velocities[0], 0.5);
  EXPECT_EQ(decoded->state.fuel_burn_rate[0], 0.125);
  EXPECT_EQ(EncodeGasCheckpoint3D(*decoded, scene_digest).value(), *encoded);
  ++scene_digest[4];
  EXPECT_EQ(
      DecodeGasCheckpoint3D(*encoded, *domain, scene_digest).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_FALSE(DecodeGasCheckpoint3D(std::span(*encoded).first(8), *domain, scene_digest).ok());
}

}  // namespace
}  // namespace fluidity
