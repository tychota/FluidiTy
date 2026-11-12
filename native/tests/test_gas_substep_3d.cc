#include <array>
#include <cmath>

#include "fluidity/host/gas_substep_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(GasSubstep3D, FixedCellHeatingUsesProjectedFluxOrSealedPressureWork) {
  const std::array blocks{BlockIndex3D{0, 0, 0}};
  const auto grid = SparseMacGrid3D::Create({{0.0, 0.0, 0.0}, 1.0}, blocks, blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const double ambient_pressure = 101325.0;
  const double heat_rate = 35000.0;
  const double step_duration = 0.1;
  const GasScalars ambient{.carrier = 1.2, .enthalpy = 3.5 * ambient_pressure};
  const std::array cells{
      GasCell3D{.cell_index = {0, 0, 0}, .volume = 1.0, .density = ambient, .absolute_pressure = ambient_pressure},
  };
  const std::array sources{GasCellSource3D{.rates = {.enthalpy = heat_rate}}};
  std::array<MacPressureFace3D, 6> faces{};
  const std::array indices{
      FaceIndex3D{FaceNormalAxis::kX, {0, 0, 0}},
      FaceIndex3D{FaceNormalAxis::kX, {1, 0, 0}},
      FaceIndex3D{FaceNormalAxis::kY, {0, 0, 0}},
      FaceIndex3D{FaceNormalAxis::kY, {0, 1, 0}},
      FaceIndex3D{FaceNormalAxis::kZ, {0, 0, 0}},
      FaceIndex3D{FaceNormalAxis::kZ, {0, 0, 1}},
  };
  for (std::size_t ordinal = 0; ordinal < faces.size(); ++ordinal) {
    faces[ordinal].face_index = indices[ordinal];
    faces[ordinal].pressure_link = SealedMacPressureLink3D{};
  }

  const auto sealed = AdvanceGasSubstep3D(*grid, step_duration, cells, faces, sources, ambient, ambient_pressure, {});
  ASSERT_TRUE(sealed.ok()) << sealed.status();
  EXPECT_NEAR(sealed->cells[0].absolute_pressure - ambient_pressure, 0.4 * heat_rate * step_duration, 1e-8);
  EXPECT_NEAR(sealed->cells[0].density.enthalpy, 3.5 * sealed->cells[0].absolute_pressure, 1e-8);

  auto reacting_cells = cells;
  reacting_cells[0].density.fuel = 0.02;
  auto reacting_sources = sources;
  reacting_sources[0].burn_rate = 4.0;
  reacting_sources[0].heat_per_fuel = 2000000.0;
  reacting_sources[0].soot_yield = 0.1;
  const auto reacted =
      AdvanceGasSubstep3D(*grid, step_duration, reacting_cells, faces, reacting_sources, ambient, ambient_pressure, {});
  ASSERT_TRUE(reacted.ok()) << reacted.status();
  const double burned_fuel = 0.02 * (1.0 - std::exp(-0.4));
  EXPECT_NEAR(reacted->cells[0].density.fuel, 0.02 - burned_fuel, 1e-14);
  EXPECT_NEAR(reacted->cells[0].density.soot, 0.1 * burned_fuel, 1e-14);
  EXPECT_NEAR(reacted->fuel_burn_rate[0] * step_duration, burned_fuel, 1e-14);
  EXPECT_NEAR(
      reacted->cells[0].absolute_pressure - ambient_pressure,
      0.4 * (heat_rate * step_duration + 2000000.0 * burned_fuel),
      1e-8
  );

  faces[1].open_area = 1.0;
  faces[1].pressure_gradient_distance = 0.5;
  faces[1].pressure_link = DirichletMacPressureLink3D{.exterior_pressure = 0.0};
  const auto open = AdvanceGasSubstep3D(*grid, step_duration, cells, faces, sources, ambient, ambient_pressure, {});
  ASSERT_TRUE(open.ok()) << open.status();
  const double expansion = step_duration * open->projection.corrected_face_velocities[1];
  const double analytic_expansion = 0.4 * heat_rate * step_duration / (1.4 * ambient_pressure);
  EXPECT_NEAR(expansion, analytic_expansion, 0.25 * analytic_expansion);
  const double pressure_defect = open->projection.cell_constraint_defects[0];
  EXPECT_NEAR(
      open->cells[0].density.enthalpy - ambient.enthalpy,
      -step_duration * ambient.enthalpy * pressure_defect,
      1e-8
  );
  EXPECT_NEAR(
      open->cells[0].density.enthalpy - ambient.enthalpy,
      step_duration * (heat_rate - open->outward_rate.enthalpy),
      1e-8
  );
  EXPECT_NEAR(open->cells[0].density.carrier, ambient.carrier * (1.0 - expansion), 1e-14);
  EXPECT_EQ(open->cells[0].absolute_pressure, ambient_pressure);
}

}  // namespace
}  // namespace fluidity
