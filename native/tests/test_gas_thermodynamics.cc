#include <array>

#include "fluidity/host/gas_thermodynamics.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(GasThermodynamics, BalancesClosedHeatByVolumeAndKeepsComponentsIndependent) {
  const std::array cells{
      GasHeatingCell{.volume = 1.0, .heat_rate = 1000.0, .component = 0},
      GasHeatingCell{.volume = 3.0, .heat_rate = 0.0, .component = 0},
      GasHeatingCell{.volume = 2.0, .heat_rate = 1000.0, .component = 1},
      GasHeatingCell{.volume = 2.0, .heat_rate = 1000.0, .component = 2},
  };
  const std::array components{
      GasThermodynamicComponent{.absolute_pressure = 100000.0, .open_to_ambient = false},
      GasThermodynamicComponent{.absolute_pressure = 101000.0, .open_to_ambient = false},
      GasThermodynamicComponent{.absolute_pressure = 101325.0, .open_to_ambient = true},
  };

  const auto result = AdvanceGasHeating(0.1, cells, components);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NEAR(result->component_pressure[0], 100010.0, 1e-9);
  EXPECT_NEAR(result->component_pressure[1], 101040.0, 1e-9);
  EXPECT_EQ(result->component_pressure[2], 101325.0);
  EXPECT_NEAR(result->target_divergence[0] + 3.0 * result->target_divergence[1], 0.0, 1e-16);
  EXPECT_EQ(result->target_divergence[2], 0.0);
  EXPECT_NEAR(result->target_divergence[3], 400.0 / (1.4 * 101325.0), 1e-16);
  EXPECT_NEAR(result->enthalpy_pressure_work_rate[0], 100.0, 1e-12);
  EXPECT_NEAR(result->enthalpy_pressure_work_rate[1], 400.0, 1e-12);
  EXPECT_EQ(result->enthalpy_pressure_work_rate[2], 0.0);
}

TEST(GasThermodynamics, RejectsInvalidMembershipAndNonpositiveUpdatedPressure) {
  const std::array components{GasThermodynamicComponent{.absolute_pressure = 10.0, .open_to_ambient = false}};
  const std::array unknown_component{GasHeatingCell{.volume = 1.0, .heat_rate = 0.0, .component = 1}};
  const std::array extreme_cooling{GasHeatingCell{.volume = 1.0, .heat_rate = -100.0, .component = 0}};
  EXPECT_EQ(AdvanceGasHeating(1.0, unknown_component, components).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(AdvanceGasHeating(1.0, extreme_cooling, components).status().code(), absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace fluidity
