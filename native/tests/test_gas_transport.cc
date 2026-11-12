#include <array>

#include "fluidity/host/gas_transport.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(GasTransport, BudgetsIncludeSharedInteriorFluxAndSignedExteriorFlux) {
  const std::array cells{
      GasTransportCell{.volume = 1.0, .density = {1.0, 2.0, 0.0, 10.0}, .source_rate = {0.0, 1.0, 0.0, 2.0}},
      GasTransportCell{.volume = 2.0, .density = {2.0, 1.0, 0.0, 20.0}, .source_rate = {}},
  };
  const std::array faces{
      GasTransportFace{.tail = std::nullopt, .head = 0, .volume_flux = 0.4},
      GasTransportFace{.tail = 0, .head = 1, .volume_flux = 0.3},
      GasTransportFace{.tail = 1, .head = std::nullopt, .volume_flux = 0.2},
  };
  const auto result = TransportGasScalars(0.5, cells, faces, {1.0, 0.0, 0.0, 5.0});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NEAR(result->density[0].fuel, 2.2, 1e-14);
  EXPECT_NEAR(result->density[1].fuel, 1.1, 1e-14);
  EXPECT_NEAR(result->density[0].carrier, 1.05, 1e-14);
  EXPECT_NEAR(result->density[1].carrier, 1.975, 1e-14);
  EXPECT_NEAR(result->outward_rate.fuel, 0.2, 1e-14);
  EXPECT_NEAR(result->outward_rate.enthalpy, 2.0, 1e-14);
  EXPECT_NEAR(result->density[0].enthalpy + 2.0 * result->density[1].enthalpy, 50.0, 1e-14);
}

TEST(GasTransport, RejectsCombinedOutflowThatExceedsTheDonorVolume) {
  const std::array cells{GasTransportCell{.volume = 1.0, .density = {1.0, 0.0, 0.0, 10.0}, .source_rate = {}}};
  const std::array faces{
      GasTransportFace{.tail = 0, .head = std::nullopt, .volume_flux = 0.6},
      GasTransportFace{.tail = std::nullopt, .head = 0, .volume_flux = -0.6},
  };
  const auto result = TransportGasScalars(1.0, cells, faces, {1.0, 0.0, 0.0, 10.0});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace fluidity
