#include <array>
#include <cmath>
#include <vector>

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

TEST(GasSubstep3D, ApicPreservesUniformDraftAndAdvectsMarker) {
  const std::array blocks{BlockIndex3D{0, 0, 0}, BlockIndex3D{1, 0, 0}, BlockIndex3D{0, 1, 0}, BlockIndex3D{0, 0, 1}};
  const auto grid = SparseMacGrid3D::Create({{0.0, 0.0, 0.0}, 1.0}, blocks, blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const GasScalars ambient{.carrier = 1.2, .enthalpy = 354637.5};
  const std::array cells{
      GasCell3D{.cell_index = {2, 2, 2}, .volume = 1.0, .density = ambient, .absolute_pressure = 101325.0},
  };
  const std::array sources{GasCellSource3D{}};
  const std::array indices{
      FaceIndex3D{FaceNormalAxis::kX, {2, 2, 2}},
      FaceIndex3D{FaceNormalAxis::kX, {3, 2, 2}},
      FaceIndex3D{FaceNormalAxis::kY, {2, 2, 2}},
      FaceIndex3D{FaceNormalAxis::kY, {2, 3, 2}},
      FaceIndex3D{FaceNormalAxis::kZ, {2, 2, 2}},
      FaceIndex3D{FaceNormalAxis::kZ, {2, 2, 3}},
  };
  std::array<MacPressureFace3D, 6> faces{};
  for (std::size_t i = 0; i < faces.size(); ++i) {
    faces[i].face_index = indices[i];
    faces[i].pressure_link = SealedMacPressureLink3D{};
  }
  for (std::size_t i = 0; i < 2; ++i) {
    faces[i].open_area = 1.0;
    faces[i].pressure_gradient_distance = 0.5;
    faces[i].pressure_link = DirichletMacPressureLink3D{.exterior_pressure = 0.0};
  }
  std::vector<double> draft(grid->face_count());
  for (std::size_t slot = 0; slot < draft.size(); ++slot) {
    const auto face = grid->view().FaceIndex(slot);
    ASSERT_TRUE(face.ok()) << face.status();
    draft[slot] = face->axis == FaceNormalAxis::kX ? 0.2 : 0.0;
  }
  const std::array markers{GasApicMarker3D{
      .world_position = {2.5, 2.5, 2.5},
      .weight = 50.0,
      .velocity = {0.2, 0.0, 0.0},
      .velocity_slopes = {},
  }};
  const auto step = AdvanceGasApicSubstep3D(*grid, 0.1, cells, faces, sources, ambient, 101325.0, {}, markers, draft);
  ASSERT_TRUE(step.ok()) << step.status();
  ASSERT_EQ(step->markers.size(), 1);
  EXPECT_NEAR(step->markers[0].world_position.x, 2.52, 1e-14);
  EXPECT_NEAR(step->markers[0].velocity.x, 0.2, 1e-14);
  EXPECT_EQ(step->markers[0].weight, 50.0);
  EXPECT_NEAR(step->gas.cells[0].density.carrier, ambient.carrier, 1e-14);

  auto outlet_markers = markers;
  outlet_markers[0].world_position.x = 2.999;
  const auto outlet =
      AdvanceGasApicSubstep3D(*grid, 0.1, cells, faces, sources, ambient, 101325.0, {}, outlet_markers, draft);
  ASSERT_TRUE(outlet.ok()) << outlet.status();
  EXPECT_TRUE(outlet->markers.empty());

  faces[1].open_area = 0.0;
  faces[1].pressure_link = SealedMacPressureLink3D{};
  const auto wall =
      AdvanceGasApicSubstep3D(*grid, 0.1, cells, faces, sources, ambient, 101325.0, {}, outlet_markers, draft);
  ASSERT_TRUE(wall.ok()) << wall.status();
  ASSERT_EQ(wall->markers.size(), 1);
  EXPECT_EQ(wall->markers[0].world_position.x, outlet_markers[0].world_position.x);
  EXPECT_EQ(wall->markers[0].velocity.x, 0.0);
}

TEST(GasSubstep3D, UniformDomainOwnsStencilsAndAdvancesAcrossCellBoundary) {
  GasUniformDomainOptions3D options{
      .geometry = {{0.0, 0.0, 0.0}, 0.25},
      .extent = {2, 2, 2},
      .open_boundary = {true, true, false, false, false, false},
      .ambient = {.carrier = 1.2, .enthalpy = 354637.5},
      .ambient_pressure = 101325.0,
      .initial_velocity = {0.2, 0.0, 0.0},
  };
  const auto domain = CreateGasUniformDomain3D(options);
  ASSERT_TRUE(domain.ok()) << domain.status();
  ASSERT_EQ(domain->cells.size(), 8);
  EXPECT_EQ(domain->faces.size(), 36);
  ASSERT_EQ(domain->markers.size(), 8);
  EXPECT_EQ(domain->markers[0].world_position.x, 0.125);
  auto markers = domain->markers;
  markers[0].world_position.x = 0.24;
  const std::vector<GasCellSource3D> sources(domain->cells.size());
  const auto step = AdvanceGasApicSubstep3D(
      domain->grid,
      0.1,
      domain->cells,
      domain->faces,
      sources,
      options.ambient,
      options.ambient_pressure,
      {},
      markers,
      domain->face_velocities
  );
  ASSERT_TRUE(step.ok()) << step.status();
  ASSERT_EQ(step->markers.size(), 8);
  EXPECT_NEAR(step->markers[0].world_position.x, 0.26, 1e-14);
  EXPECT_NEAR(step->gas.cells[0].density.carrier, 1.2, 1e-14);
}
}  // namespace
}  // namespace fluidity
