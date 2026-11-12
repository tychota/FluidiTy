#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#include "../src/mac_pressure_3d_multigrid.h"
#include "../src/mac_pressure_3d_system.h"
#include "../src/mac_pressure_3d_workspace_internal.h"
#include "../src/mac_viscous_pressure_3d.h"
#include "absl/status/status.h"
#include "fluidity/host/mac_pressure_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "fluidity/host/viscosity.h"
#include "gtest/gtest.h"
#include "mac_pressure_3d_fixture.h"
#include "mac_pressure_acoustic_cases.h"
#include "mac_viscous_traction_cases.h"

namespace fluidity {
namespace {

TEST(MacPressure3D, BackwardEulerStandingWaveMatchesAnalyticMode) {
  auto workspace = MacPressureWorkspace3D::Create(16, 15);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  pressure_test::ExpectBackwardEulerAcousticColumn(
      pressure_test::AcousticInitialCondition::kStandingWave,
      [&](const auto& system) {
        return workspace->ProjectPrepared(system);
      }
  );
}

TEST(MacPressure3D, BackwardEulerPressurePulseMatchesModalOracle) {
  auto workspace = MacPressureWorkspace3D::Create(16, 15);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  pressure_test::ExpectBackwardEulerAcousticColumn(
      pressure_test::AcousticInitialCondition::kLocalizedPulse,
      [&](const auto& system) {
        return workspace->ProjectPrepared(system);
      }
  );
}

TEST(MacPressure3D, WaterHammerValveClosureMatchesAcousticOracle) {
  auto workspace = MacPressureWorkspace3D::Create(64, 63);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  pressure_test::ExpectBackwardEulerAcousticColumn(
      pressure_test::AcousticInitialCondition::kValveClosure,
      [&](const auto& system) {
        return workspace->ProjectPrepared(system);
      }
  );
}

TEST(MacPressure3D, CompliantPistonConvergesToEosCompression) {
  auto workspace = MacPressureWorkspace3D::Create(1, 6);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  pressure_test::ExpectCompliantPistonCompression([&](const MacPressureInputs3D& inputs) {
    return ProjectMacPressure3D(
        inputs.mac_grid.view(),
        inputs.step_duration,
        {inputs.pressure_cells, inputs.pressure_faces},
        {},
        *workspace
    );
  });
}

using pressure_internal::EvaluateMacPressure3D;
using pressure_internal::PrepareMacPressureSystem3D;

constexpr UniformMacGeometry3D kUnitGeometry{{0.0, 0.0, 0.0}, 1.0};
constexpr std::array kUnitBlock{BlockIndex3D{0, 0, 0}};
constexpr double kInfinity = std::numeric_limits<double>::infinity();

std::array<MacPressureFace3D, 6> SealedFacesAtX(std::int32_t x) {
  return {{
      {.face_index = {FaceNormalAxis::kX, {x, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kX, {x + 1, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kY, {x, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kY, {x, 1, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kZ, {x, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kZ, {x, 0, 1}}, .pressure_link = SealedMacPressureLink3D{}},
  }};
}

MacPressureCell3D IncompressibleCellAtX(std::int32_t x) {
  return {.cell_index = {x, 0, 0}, .fluid_volume = 1.0, .bulk_modulus = kInfinity};
}

TEST(MacPressure3D, CoupledViscosityBalancesMomentumAndCompliantFlux) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto faces = SealedFacesAtX(0);
  for (std::size_t face = 0; face < 2; ++face) {
    faces[face].open_area = 1.0;
    faces[face].inverse_density = 1.0;
    faces[face].pressure_gradient_distance = 1.0;
  }
  faces[0].predicted_normal_velocity = -1.0;
  faces[0].pressure_link = DirichletMacPressureLink3D{2.0};
  faces[1].predicted_normal_velocity = 8.0;
  faces[1].pressure_link = DirichletMacPressureLink3D{5.0};
  const std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1.0,
      .target_divergence = 3.0,
      .bulk_modulus = 2.0,
      .previous_pressure = 2.0,
  }};
  const std::array viscosity{MacViscosityCell3D{.cell_index = {0, 0, 0}, .dynamic_viscosity = 1.0}};
  const auto strain = AssembleFullCellMacStrain3D(grid->view(), viscosity);
  ASSERT_TRUE(strain.ok()) << strain.status();
  const auto result = ProjectMacViscousPressure3D({*grid, 1.0, cells, faces}, *strain);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NEAR(result->cell_pressures[0], 4.0, 1e-9);
  EXPECT_NEAR(result->face_velocities[0], 1.0, 1e-9);
  EXPECT_NEAR(result->face_velocities[1], 3.0, 1e-9);
  EXPECT_NEAR(result->cell_constraint_defects[0], 0.0, 1e-9);
  EXPECT_LT(result->max_momentum_residual, 1e-9);
}

TEST(MacPressure3D, SolvesAllIndependentFixtureCases) {
  std::vector<pressure_test::MacPressureFixtureCase3D> cases;
  ASSERT_NO_FATAL_FAILURE(pressure_test::LoadMacPressureFixtureCases3D(cases));
  ASSERT_EQ(cases.size(), 8);
  for (const auto& fixture_case : cases) {
    SCOPED_TRACE(fixture_case.name);
    const auto projection =
        ProjectMacPressure3D({fixture_case.grid, fixture_case.step_duration, fixture_case.cells, fixture_case.faces});
    ASSERT_TRUE(projection.ok()) << projection.status();
    pressure_test::ExpectMacPressureFixtureProjection3D(
        *projection,
        fixture_case.expected_projection,
        fixture_case.absolute_tolerance,
        fixture_case.relative_tolerance
    );
  }
}

using viscous_traction_test::AffinePressure;
using viscous_traction_test::AffineProblem;
using viscous_traction_test::MakeAffineProblem;
using viscous_traction_test::MakeRectangularAffineProblem;
using viscous_traction_test::SortFaces;

TEST(MacPressure3D, CoupledViscosityRecoversPlanarExtensionTractionAtHighContrast) {
  viscous_traction_test::ExpectPlanarExtensionTraction([](const MacPressureInputs3D& inputs,
                                                          std::span<const MacStrainRow3D> strain) {
    return ProjectMacViscousPressure3D(inputs, strain);
  });
}

TEST(MacPressure3D, CoupledViscosityRecoversMovingWallCouetteTraction) {
  viscous_traction_test::ExpectMovingWallCouetteTraction([](const MacPressureInputs3D& inputs,
                                                            std::span<const MacStrainRow3D> strain) {
    return ProjectMacViscousPressure3D(inputs, strain);
  });
}

TEST(MacPressure3D, CoupledViscosityQualifiesPeriodicCouetteTractions) {
  for (const std::size_t resolution : {16, 32}) {
    SCOPED_TRACE(resolution);
    const double h = 1.0 / static_cast<double>(resolution);
    const double volume = h * h;  // Unit-depth extrusion.
    for (const double dt : {0.005, 0.0025}) {
      SCOPED_TRACE(dt);
      for (const double ratio : {1.0, 10.0, 1000.0}) {
        SCOPED_TRACE(ratio);
        const auto viscosity = [&](std::size_t y) {
          return y < resolution / 2 ? 1.0 : ratio;
        };
        const auto x_slot = [&](std::size_t x, std::size_t y) {
          return resolution * y + x;
        };
        const auto y_slot = [&](std::size_t x, std::size_t y) {
          return resolution * (resolution + y - 1) + x;
        };
        const double exact_shear = 2.0 / (1.0 + 1.0 / ratio);
        std::vector<MacPressureCell3D> cells;
        std::vector<pressure_internal::ExplicitMacPressureFace3D> faces;
        std::vector<MacStrainRow3D> strain;
        for (std::size_t y = 0; y < resolution; ++y) {
          const double sample_y = (static_cast<double>(y) + 0.5) * h;
          const double speed = exact_shear * (std::min(sample_y, 0.5) + std::max(sample_y - 0.5, 0.0) / ratio);
          for (std::size_t x = 0; x < resolution; ++x) {
            const CellIndex3D cell_index{static_cast<int>(x), static_cast<int>(y), 0};
            cells.push_back({.cell_index = cell_index, .fluid_volume = volume, .bulk_modulus = kInfinity});
            faces.push_back({
                .face_slot = x_slot(x, y),
                .tail_cell_ordinal = x_slot((x + resolution - 1) % resolution, y),
                .head_cell_ordinal = x_slot(x, y),
                .open_area = h,
                .inverse_density = 1.0,
                .pressure_gradient_distance = h,
                .predicted_normal_velocity = speed,
            });
            strain.push_back({
                .low_cell = cell_index,
                .first_axis = FaceNormalAxis::kX,
                .second_axis = FaceNormalAxis::kX,
                .terms = {{{x_slot(x, y), -1.0 / h}, {x_slot((x + 1) % resolution, y), 1.0 / h}}},
                .term_count = 2,
                .quadrature_weight = 2.0 * viscosity(y) * volume,
            });
            MacStrainRow3D normal{
                .low_cell = cell_index,
                .first_axis = FaceNormalAxis::kY,
                .second_axis = FaceNormalAxis::kY,
                .terms = {},
                .term_count = 0,
                .quadrature_weight = 2.0 * viscosity(y) * volume,
            };
            if (y > 0) {
              normal.terms[normal.term_count++] = {y_slot(x, y), -1.0 / h};
            }
            if (y + 1 < resolution) {
              normal.terms[normal.term_count++] = {y_slot(x, y + 1), 1.0 / h};
            }
            strain.push_back(normal);
            if (y == 0 || y + 1 == resolution) {
              const bool upper = y != 0;
              auto row = MakeMacTangentialWallStrain3D(
                  cell_index,
                  FaceNormalAxis::kX,
                  FaceNormalAxis::kY,
                  x_slot(x, y),
                  h,
                  viscosity(y),
                  upper ? 1.0 : 0.0,
                  upper
              );
              ASSERT_TRUE(row.ok()) << row.status();
              row->quadrature_weight /= h;  // Convert the half-cell row to unit depth.
              strain.push_back(*row);
            }
          }
        }
        for (std::size_t y = 1; y < resolution; ++y) {
          const double shear_viscosity = 2.0 / (1.0 / viscosity(y - 1) + 1.0 / viscosity(y));
          for (std::size_t x = 0; x < resolution; ++x) {
            faces.push_back({
                .face_slot = y_slot(x, y),
                .tail_cell_ordinal = x_slot(x, y - 1),
                .head_cell_ordinal = x_slot(x, y),
                .open_area = h,
                .inverse_density = 1.0,
                .pressure_gradient_distance = h,
            });
            const double derivative = 0.5 / h;
            strain.push_back({
                .low_cell = {static_cast<int>(x), static_cast<int>(y - 1), 0},
                .first_axis = FaceNormalAxis::kX,
                .second_axis = FaceNormalAxis::kY,
                .terms = {{
                    {x_slot(x, y - 1), -derivative},
                    {x_slot(x, y), derivative},
                    {y_slot((x + resolution - 1) % resolution, y), -derivative},
                    {y_slot(x, y), derivative},
                }},
                .term_count = 4,
                .quadrature_weight = 4.0 * shear_viscosity * volume,
            });
          }
        }
        const auto projection = pressure_internal::ProjectExplicitMacViscousPressure3D(dt, cells, faces, strain);
        ASSERT_TRUE(projection.ok()) << projection.status();
        double max_normal_error = 0.0;
        double max_shear_error = 0.0;
        std::array<double, 2> normal_error_squared{};
        std::array<double, 2> shear_error_squared{};
        std::array<std::size_t, 2> normal_samples{};
        std::array<std::size_t, 2> shear_samples{};
        for (const auto& row : strain) {
          double value = row.prescribed_strain;
          for (const auto& term : std::span(row.terms).first(row.term_count)) {
            value += term.coefficient * projection->face_velocities[term.face_slot];
          }
          if (row.first_axis != row.second_axis) {
            const double sample_volume = row.term_count == 1 ? 0.5 * volume : volume;
            const double traction = row.quadrature_weight * value / (2.0 * sample_volume);
            const double error = std::abs(traction - exact_shear) / 2.0;
            max_shear_error = std::max(max_shear_error, error);
            const auto y = static_cast<std::size_t>(row.low_cell.y);
            if (row.term_count == 4 && (y == resolution / 2 - 2 || y == resolution / 2)) {
              const std::size_t side = y < resolution / 2 ? 0 : 1;
              shear_error_squared[side] += error * error;
              ++shear_samples[side];
            }
          } else if (row.first_axis == FaceNormalAxis::kY) {
            const auto y = static_cast<std::size_t>(row.low_cell.y);
            const auto cell = x_slot(static_cast<std::size_t>(row.low_cell.x), y);
            // The pressure pin is zero relative to the uniform analytic reference S.
            const double normal_error = 2.0 * viscosity(y) * value - projection->cell_pressures[cell];
            const double error = std::abs(normal_error) / 2.0;
            max_normal_error = std::max(max_normal_error, error);
            if (y == resolution / 2 - 1 || y == resolution / 2) {
              const std::size_t side = y < resolution / 2 ? 0 : 1;
              normal_error_squared[side] += error * error;
              ++normal_samples[side];
            }
          }
        }
        double normal_rms = 0.0;
        double shear_rms = 0.0;
        for (std::size_t side = 0; side < 2; ++side) {
          ASSERT_EQ(normal_samples[side], resolution);
          ASSERT_EQ(shear_samples[side], resolution);
          normal_rms = std::max(normal_rms, std::sqrt(normal_error_squared[side] / static_cast<double>(resolution)));
          shear_rms = std::max(shear_rms, std::sqrt(shear_error_squared[side] / static_cast<double>(resolution)));
        }
        EXPECT_LE(normal_rms, 0.05);
        EXPECT_LE(shear_rms, 0.05);
        EXPECT_LE(max_normal_error, 0.10);
        EXPECT_LE(max_shear_error, 0.10);
        std::cout << "periodic_couette resolution=" << resolution << " dt=" << dt << " viscosity_ratio=" << ratio
                  << " normal_rms=" << normal_rms << " tangential_rms=" << shear_rms
                  << " normal_max=" << max_normal_error << " tangential_max=" << max_shear_error << '\n';
      }
    }
  }
}

TEST(MacPressure3D, CoupledViscosityPreservesMovingWallFluxAndOrientedJumps) {
  auto problem = MakeAffineProblem(0, 2, false);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  for (auto& face : problem.faces) {
    if (face.face_index.axis != FaceNormalAxis::kX) {
      continue;
    }
    if (face.open_area == 0.0) {
      face.wall_normal_velocity = 1.0;
    } else {
      face.pressure_link = InternalMacPressureLink3D{2.0};
      face.predicted_normal_velocity = 0.0;
    }
  }
  std::vector<MacViscosityCell3D> material;
  for (const auto& cell : problem.cells) {
    material.push_back({.cell_index = cell.cell_index, .dynamic_viscosity = 3.0});
  }
  const auto strain = AssembleFullCellMacStrain3D(grid->view(), material);
  ASSERT_TRUE(strain.ok()) << strain.status();
  const auto result = ProjectMacViscousPressure3D({*grid, 1.0, problem.cells, problem.faces}, *strain);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->cell_pressures.front(), 0.0);
  for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
    EXPECT_NEAR(result->cell_pressures[cell], AffinePressure(problem.cells[cell].cell_index), 1e-4);
    EXPECT_NEAR(result->cell_constraint_defects[cell], 0.0, 2e-5);
  }
  for (std::size_t face = 0; face < problem.faces.size(); ++face) {
    const bool x_normal = problem.faces[face].face_index.axis == FaceNormalAxis::kX;
    EXPECT_NEAR(result->face_velocities[face], x_normal ? 1.0 : 0.0, 2e-5);
  }
}

TEST(MacPressure3D, CoupledViscosityUsesGaugeShiftedPressureHistory) {
  for (const bool anchored : {false, true}) {
    SCOPED_TRACE(anchored);
    auto problem = MakeAffineProblem(0, 2, anchored);
    const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
    ASSERT_TRUE(grid.ok()) << grid.status();
    SortFaces(problem, *grid);
    std::vector<MacViscosityCell3D> material;
    for (const auto& cell : problem.cells) {
      material.push_back({.cell_index = cell.cell_index, .dynamic_viscosity = 1.0});
    }
    const auto strain = AssembleFullCellMacStrain3D(grid->view(), material);
    ASSERT_TRUE(strain.ok()) << strain.status();
    const MacPressureInputs3D inputs{*grid, 1.0, problem.cells, problem.faces};
    const auto cold = ProjectMacViscousPressure3D(inputs, *strain);
    ASSERT_TRUE(cold.ok()) << cold.status();
    ASSERT_GT(cold->pressure_iterations, 0u);
    const double gauge_shift = anchored ? 0.0 : 32.0;
    for (std::size_t index = 0; index < problem.cells.size(); ++index) {
      problem.cells[index].previous_pressure = cold->cell_pressures[index] + gauge_shift;
    }
    const auto warm = ProjectMacViscousPressure3D(inputs, *strain);
    ASSERT_TRUE(warm.ok()) << warm.status();
    EXPECT_EQ(warm->pressure_iterations, 0u);
    EXPECT_DOUBLE_EQ(warm->initial_constraint_residual_l2, cold->initial_constraint_residual_l2);
    for (std::size_t index = 0; index < problem.cells.size(); ++index) {
      EXPECT_NEAR(warm->cell_pressures[index], cold->cell_pressures[index], 1e-12);
    }
    for (std::size_t face = 0; face < problem.faces.size(); ++face) {
      EXPECT_NEAR(warm->face_velocities[face], cold->face_velocities[face], 1e-12);
    }
  }
}

TEST(MacPressure3D, CoupledPressureHistoryKeepsWeakJumpScaleAndClearsZeroForcing) {
  constexpr double kJump = 1e-8;
  auto problem = MakeAffineProblem(0, 2, false);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  for (auto& face : problem.faces) {
    face.predicted_normal_velocity = 0.0;
    face.wall_normal_velocity = 0.0;
    if (auto* internal = std::get_if<InternalMacPressureLink3D>(&face.pressure_link)) {
      internal->pressure_jump = face.face_index.axis == FaceNormalAxis::kX ? kJump : 0.0;
    }
  }
  std::vector<MacViscosityCell3D> material;
  for (const auto& cell : problem.cells) {
    material.push_back({.cell_index = cell.cell_index, .dynamic_viscosity = 1.0});
  }
  const auto strain = AssembleFullCellMacStrain3D(grid->view(), material);
  ASSERT_TRUE(strain.ok()) << strain.status();
  const MacPressureInputs3D inputs{*grid, 1.0, problem.cells, problem.faces};
  const auto cold = ProjectMacViscousPressure3D(inputs, *strain);
  ASSERT_TRUE(cold.ok()) << cold.status();
  ASSERT_GT(cold->pressure_iterations, 0u);
  ASSERT_GT(cold->initial_constraint_residual_l2, 0.0);
  for (std::size_t index = 0; index < problem.cells.size(); ++index) {
    auto& cell = problem.cells[index];
    const double exact = kJump * cell.cell_index.x;
    EXPECT_NEAR(cold->cell_pressures[index], exact, 1e-11);
    cell.previous_pressure = 32.0 + exact;
  }
  const auto warm = ProjectMacViscousPressure3D(inputs, *strain);
  ASSERT_TRUE(warm.ok()) << warm.status();
  EXPECT_EQ(warm->pressure_iterations, 0u);
  EXPECT_DOUBLE_EQ(warm->initial_constraint_residual_l2, cold->initial_constraint_residual_l2);
  EXPECT_LE(warm->constraint_residual_l2, 1e-3 * warm->initial_constraint_residual_l2);

  for (auto& face : problem.faces) {
    if (auto* internal = std::get_if<InternalMacPressureLink3D>(&face.pressure_link)) {
      internal->pressure_jump = 0.0;
    }
  }
  const auto unforced = ProjectMacViscousPressure3D(inputs, *strain);
  ASSERT_TRUE(unforced.ok()) << unforced.status();
  EXPECT_EQ(unforced->pressure_iterations, 0u);
  EXPECT_DOUBLE_EQ(unforced->initial_constraint_residual_l2, 0.0);
  for (const double pressure : unforced->cell_pressures) {
    EXPECT_DOUBLE_EQ(pressure, 0.0);
  }
  for (const double velocity : unforced->face_velocities) {
    EXPECT_DOUBLE_EQ(velocity, 0.0);
  }
}

TEST(MacPressure3D, BoundedOriginalSystemMatchesOwningAndPreservesArenaOnFailure) {
  auto problem = MakeAffineProblem(0, 2, true);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto owning = PrepareMacPressureSystem3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(owning.ok()) << owning.status();
  const auto bytes =
      pressure_internal::QueryBoundedMacPressureSystemBytes3D(problem.cells.size(), problem.faces.size());
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  ASSERT_EQ(*bytes % sizeof(std::max_align_t), 0u);
  auto backing = LiquidRecords3D<std::max_align_t>::Create(*bytes / sizeof(std::max_align_t) + 1);
  ASSERT_TRUE(backing.ok()) << backing.status();
  ASSERT_TRUE(backing->Resize(backing->capacity()).ok());
  pressure_internal::MacPressureScratch3D scratch(backing->values());
  const auto sentinel = scratch.Take<std::max_align_t>(1);
  ASSERT_TRUE(sentinel.ok()) << sentinel.status();
  auto sentinel_bytes = std::as_writable_bytes(*sentinel);
  sentinel_bytes.front() = std::byte{43};
  const auto entry_cursor = scratch.used();

  const auto rejected = pressure_internal::PrepareBoundedMacPressureSystem3D(
      grid->view(),
      1.0,
      problem.cells,
      std::span(problem.faces).first(problem.faces.size() - 1),
      scratch
  );
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(scratch.used(), entry_cursor);
  EXPECT_EQ(sentinel_bytes.front(), std::byte{43});

  const auto borrowed =
      pressure_internal::PrepareBoundedMacPressureSystem3D(grid->view(), 1.0, problem.cells, problem.faces, scratch);
  ASSERT_TRUE(borrowed.ok()) << borrowed.status();
  EXPECT_GT(scratch.used(), entry_cursor);
  EXPECT_EQ(sentinel_bytes.front(), std::byte{43});
  EXPECT_TRUE(std::ranges::equal(borrowed->integrated_rhs(), owning->integrated_rhs()));
  EXPECT_TRUE(std::ranges::equal(borrowed->rhs_absolute_term_sums(), owning->rhs_absolute_term_sums()));
  EXPECT_TRUE(std::ranges::equal(borrowed->component_ordinals(), owning->component_ordinals()));
  EXPECT_TRUE(std::ranges::equal(borrowed->gauge_cell_ordinals(), owning->gauge_cell_ordinals()));
  ASSERT_EQ(borrowed->links().size(), owning->links().size());
  for (std::size_t face = 0; face < borrowed->links().size(); ++face) {
    const auto& actual = borrowed->links()[face];
    const auto& expected = owning->links()[face];
    EXPECT_EQ(actual.face_slot, expected.face_slot);
    EXPECT_EQ(actual.tail_cell_ordinal, expected.tail_cell_ordinal);
    EXPECT_EQ(actual.head_cell_ordinal, expected.head_cell_ordinal);
    EXPECT_DOUBLE_EQ(actual.open_area, expected.open_area);
    EXPECT_DOUBLE_EQ(actual.conductance, expected.conductance);
    EXPECT_DOUBLE_EQ(actual.prescribed_wall_flux, expected.prescribed_wall_flux);
  }
}

TEST(MacPressure3D, CoupledPressureMatchesAnalyticColumnAcrossGcrRestart) {
  constexpr int kCellCount = 64;
  const auto reference_pressure = [](int index) {
    if (index < 0 || index >= kCellCount) {
      return 0.0;
    }
    const double x = static_cast<double>(index + 1);
    return 0.1 * std::sin(0.32 * x) + 0.05 * std::cos(1.73 * x) + 0.01 * (index % 5);
  };
  auto problem = MakeRectangularAffineProblem(0, {kCellCount, 1, 1}, true);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  for (auto& face : problem.faces) {
    if (face.face_index.axis != FaceNormalAxis::kX) {
      face.open_area = 0.0;
      face.predicted_normal_velocity = 0.0;
      face.wall_normal_velocity = 0.0;
      face.pressure_link = SealedMacPressureLink3D{};
      continue;
    }
    const int head = face.face_index.lattice_index.x;
    face.open_area = 1.0;
    face.inverse_density = 1.0;
    face.pressure_gradient_distance = 1.0;
    face.predicted_normal_velocity = reference_pressure(head) - reference_pressure(head - 1);
    face.pressure_link = head == 0 || head == kCellCount ? decltype(face.pressure_link){DirichletMacPressureLink3D{0.0}}
                                                         : decltype(face.pressure_link){InternalMacPressureLink3D{0.0}};
  }
  std::vector<MacViscosityCell3D> material;
  for (const auto& cell : problem.cells) {
    material.push_back({.cell_index = cell.cell_index, .dynamic_viscosity = 0.2});
  }
  const auto strain = AssembleFullCellMacStrain3D(grid->view(), material);
  ASSERT_TRUE(strain.ok()) << strain.status();
  const auto solution = ProjectMacViscousPressure3D({*grid, 1.0, problem.cells, problem.faces}, *strain);
  ASSERT_TRUE(solution.ok()) << solution.status();
  std::cout << "coupled_gcr_restart_iterations=" << solution->pressure_iterations << '\n';
  EXPECT_GT(solution->pressure_iterations, 32u);
  for (std::size_t index = 0; index < problem.cells.size(); ++index) {
    EXPECT_NEAR(solution->cell_pressures[index], reference_pressure(problem.cells[index].cell_index.x), 1e-3);
  }
  for (const double speed : solution->face_velocities) {
    EXPECT_NEAR(speed, 0.0, 2e-5);
  }
}

TEST(MacPressure3D, CoupledCartesianWallMatchesAnalyticTractionAndWork) {
  constexpr double kDynamicViscosity = 10.0;
  constexpr double kDensity = 1.0;
  constexpr double kWallSpeed = 0.25;
  constexpr double kExactStrain = -0.2;
  constexpr double kExactTraction = 2.0 * kDynamicViscosity * kExactStrain;
  for (const double width : {1.0, 0.5}) {
    for (const double step_duration : {0.02, 0.01}) {
      for (const double wall_offset_fraction : {0.125, 0.25, 0.375}) {
        SCOPED_TRACE(::testing::Message() << width << ' ' << step_duration << ' ' << wall_offset_fraction);
        const double area = width * width;
        const double liquid_length = (1.0 - wall_offset_fraction) * width;
        const double liquid_volume = area * liquid_length;
        const double exact_outlet_speed = kWallSpeed + kExactStrain * liquid_length;
        const double pressure_gradient_distance = 0.5 * width;
        auto problem = MakeAffineProblem(0, 1, true);
        const auto grid = SparseMacGrid3D::Create({{}, width}, problem.blocks, problem.blocks.size());
        ASSERT_TRUE(grid.ok()) << grid.status();
        SortFaces(problem, *grid);
        ASSERT_EQ(problem.cells.size(), 1u);
        auto& cell = problem.cells[0];
        cell.fluid_volume = liquid_volume;
        cell.target_divergence = kExactStrain;
        cell.previous_pressure = 0.0;
        std::size_t outlet_ordinal = problem.faces.size();
        for (std::size_t ordinal = 0; ordinal < problem.faces.size(); ++ordinal) {
          auto& face = problem.faces[ordinal];
          face.open_area = 0.0;
          face.wall_normal_velocity = 0.0;
          face.predicted_normal_velocity = 0.0;
          face.pressure_link = SealedMacPressureLink3D{};
          if (face.face_index.axis != FaceNormalAxis::kX) {
            continue;
          }
          if (face.face_index.lattice_index.x == 0) {
            face.wall_normal_velocity = kWallSpeed;
            face.prescribed_wall_flux = area * kWallSpeed;
          } else {
            face.open_area = area;
            face.inverse_density = 1.0 / kDensity;
            face.pressure_gradient_distance = pressure_gradient_distance;
            face.predicted_normal_velocity = exact_outlet_speed;
            face.pressure_link = DirichletMacPressureLink3D{0.0};
            outlet_ordinal = ordinal;
          }
        }
        ASSERT_LT(outlet_ordinal, problem.faces.size());
        const auto wall_slot = grid->view().FaceSlot({FaceNormalAxis::kX, {0, 0, 0}});
        const auto outlet_slot = grid->view().FaceSlot({FaceNormalAxis::kX, {1, 0, 0}});
        ASSERT_TRUE(wall_slot.ok()) << wall_slot.status();
        ASSERT_TRUE(outlet_slot.ok()) << outlet_slot.status();
        const double inverse_length = 1.0 / liquid_length;
        const MacStrainRow3D row{
            .low_cell = cell.cell_index,
            .first_axis = FaceNormalAxis::kX,
            .second_axis = FaceNormalAxis::kX,
            .terms =
                {{{.face_slot = *wall_slot, .coefficient = -inverse_length},
                  {.face_slot = *outlet_slot, .coefficient = inverse_length}}},
            .term_count = 2,
            .quadrature_weight = 2.0 * kDynamicViscosity * liquid_volume,
        };
        const auto result =
            ProjectMacViscousPressure3D({*grid, step_duration, problem.cells, problem.faces}, std::span(&row, 1));
        ASSERT_TRUE(result.ok()) << result.status();
        const double outlet_speed = result->face_velocities[outlet_ordinal];
        const double measured_strain = (outlet_speed - kWallSpeed) / liquid_length;
        const double measured_traction = 2.0 * kDynamicViscosity * measured_strain;
        const double pressure = result->cell_pressures[0];
        EXPECT_NEAR(outlet_speed, exact_outlet_speed, 1e-10);
        EXPECT_NEAR(measured_traction, kExactTraction, 1e-9);
        EXPECT_NEAR(pressure, kExactTraction, 1e-9);
        EXPECT_NEAR(result->inward_prescribed_wall_flux_m3_per_s, area * kWallSpeed, 1e-14);

        const double inertia = area * kDensity * pressure_gradient_distance;
        const double change_in_kinetic_energy =
            0.5 * inertia * (outlet_speed * outlet_speed - exact_outlet_speed * exact_outlet_speed);
        const double velocity_change = outlet_speed - exact_outlet_speed;
        const double backward_euler_defect = 0.5 * inertia * velocity_change * velocity_change;
        const double strain_dissipation = step_duration * row.quadrature_weight * measured_strain * measured_strain;
        const double wall_work = -step_duration * area * measured_traction * kWallSpeed;
        const double pressure_work = step_duration * area * pressure * outlet_speed;
        EXPECT_GT(wall_work, 0.0);
        EXPECT_NEAR(wall_work, -step_duration * area * kExactTraction * kWallSpeed, 1e-11);
        EXPECT_NEAR(
            change_in_kinetic_energy + backward_euler_defect + strain_dissipation,
            wall_work + pressure_work,
            1e-11
        );
      }
    }
  }
}

TEST(MacPressure3D, SolvesNegativeCoordinateMultilevelAffineFieldWithAllSixBoundaries) {
  for (const bool anchored : {false, true}) {
    SCOPED_TRACE(anchored);
    auto problem = MakeAffineProblem(-3, 4, anchored);
    const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
    ASSERT_TRUE(grid.ok()) << grid.status();
    SortFaces(problem, *grid);
    const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces});
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_GE(result->diagnostics.multigrid_level_count, 2);
    EXPECT_GT(result->diagnostics.iteration_count, 0);
    EXPECT_LE(result->diagnostics.max_volume_scaled_residual, 2e-5);
    EXPECT_LE(result->diagnostics.max_cell_constraint_defect, 2e-5);
    double rhs_norm = 0.0;
    for (double rhs : result->integrated_rhs) {
      rhs_norm = std::hypot(rhs_norm, rhs);
    }
    EXPECT_LE(result->diagnostics.true_residual_l2, 2e-6 + 2e-6 * rhs_norm);
    for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
      const double expected = AffinePressure(problem.cells[cell].cell_index) + (anchored ? 0.0 : 21.0);
      EXPECT_NEAR(result->cell_pressures[cell], expected, 5e-5);
      EXPECT_NEAR(result->true_residuals[cell], -result->cell_constraint_defects[cell], 1e-12);
    }
    for (double velocity : result->corrected_face_velocities) {
      EXPECT_NEAR(velocity, 0.0, 5e-5);
    }
    RecordProperty(
        anchored ? "anchored_true_residual_l2" : "gauged_true_residual_l2",
        std::to_string(result->diagnostics.true_residual_l2)
    );
    RecordProperty(
        anchored ? "anchored_max_defect" : "gauged_max_defect",
        std::to_string(result->diagnostics.max_cell_constraint_defect)
    );
  }
}

TEST(MacPressure3D, ShiftsPreviousPressureByEachNeumannPinBeforeIterating) {
  auto problem = MakeAffineProblem(-3, 4, false);
  for (auto& cell : problem.cells) {
    cell.previous_pressure = AffinePressure(cell.cell_index) + 100.0;
  }
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces}, {.max_iterations = 1});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->diagnostics.iteration_count, 0);
  EXPECT_DOUBLE_EQ(result->cell_pressures.front(), 0.0);
  EXPECT_DOUBLE_EQ(result->cell_pressures.back(), 21.0);
}

TEST(MacPressure3D, RetainsAnchoredPreviousPressureMean) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1.0,
      .target_divergence = 0.5,
      .bulk_modulus = 2.0,
      .previous_pressure = 4.0,
  }};
  const auto result = ProjectMacPressure3D({*grid, 1.0, cells, SealedFacesAtX(0)});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->gauge_cell_ordinals.empty());
  EXPECT_DOUBLE_EQ(result->cell_pressures.front(), 5.0);
}

TEST(MacPressure3D, InteriorMovingWallFluxEntersRhsAndPhysicalDefect) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1.0,
      .bulk_modulus = 2.0,
      .previous_pressure = 4.0,
      .integrated_inward_wall_flux = 0.5,
  }};
  const auto faces = SealedFacesAtX(0);
  const auto prepared = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  const auto old = EvaluateMacPressure3D(*prepared, std::array{4.0});
  ASSERT_TRUE(old.ok()) << old.status();
  EXPECT_DOUBLE_EQ(old->true_residuals.front(), 0.5);
  EXPECT_DOUBLE_EQ(old->cell_constraint_defects.front(), -0.5);
  const auto result = ProjectMacPressure3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->integrated_rhs.front(), 2.5);
  EXPECT_DOUBLE_EQ(result->cell_pressures.front(), 5.0);
  EXPECT_DOUBLE_EQ(result->cell_constraint_defects.front(), 0.0);

  cells.front().bulk_modulus = kInfinity;
  const auto closed = ProjectMacPressure3D({*grid, 1.0, cells, faces});
  EXPECT_EQ(closed.status().code(), absl::StatusCode::kFailedPrecondition);
}
TEST(MacPressure3D, ExhaustsOneIterationWithoutPublishingOrMutatingInputs) {
  auto problem = MakeAffineProblem(-3, 4, false);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces}, {.max_iterations = 1});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kAborted);
  EXPECT_EQ(problem.cells.front().previous_pressure, 0.0);
  EXPECT_EQ(problem.cells.back().previous_pressure, 0.0);
  const auto converged = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(converged.ok()) << converged.status();
  EXPECT_GT(converged->diagnostics.iteration_count, 1);
}

TEST(MacPressure3D, TinyCellMustPassLocalGateEvenWhenInitialGlobalGatePasses) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1e-6,
      .target_divergence = 0.1,
      .bulk_modulus = kInfinity,
  }};
  auto faces = SealedFacesAtX(0);
  faces[1].open_area = 1.0;
  faces[1].inverse_density = 1.0;
  faces[1].pressure_gradient_distance = 1.0;
  faces[1].pressure_link = DirichletMacPressureLink3D{0.0};
  const auto prepared = PrepareMacPressureSystem3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  const auto initial = EvaluateMacPressure3D(*prepared, std::array{0.0});
  ASSERT_TRUE(initial.ok()) << initial.status();
  EXPECT_LT(initial->diagnostics.true_residual_l2, 2e-6);
  EXPECT_NEAR(initial->diagnostics.max_volume_scaled_residual, 0.1, 1e-15);
  const auto result = ProjectMacPressure3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_GT(result->diagnostics.iteration_count, 0);
  EXPECT_NEAR(result->cell_pressures.front(), 1e-7, 2e-11);
  EXPECT_LE(result->diagnostics.max_volume_scaled_residual, 2e-5);
  EXPECT_LE(result->diagnostics.max_cell_constraint_defect, 2e-5);
}

TEST(MacPressure3D, RejectsIterationBudgetsOutsideFixedBound) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{IncompressibleCellAtX(0)};
  const auto faces = SealedFacesAtX(0);
  for (const auto iterations : {std::size_t{0}, std::size_t{257}}) {
    const auto result = ProjectMacPressure3D({*grid, 1.0, cells, faces}, {.max_iterations = iterations});
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

double GraphEnergy(const pressure_internal::MacPressureLevel3D& level, std::span<const double> pressure) {
  double energy = 0.0;
  for (std::size_t unknown = 0; unknown < pressure.size(); ++unknown) {
    energy += level.anchor_diagonal[unknown] * pressure[unknown] * pressure[unknown];
  }
  for (const auto& edge : level.edges) {
    const double difference = pressure[edge.lower_unknown_ordinal] - pressure[edge.higher_unknown_ordinal];
    energy += edge.conductance * difference * difference;
  }
  return energy;
}

TEST(MacPressure3D, SharedHierarchyPreservesGalerkinEnergyAndBoundsTerminalBlocks) {
  auto problem = MakeAffineProblem(-4, 8, true);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto hierarchy = pressure_internal::BuildMacPressureHierarchy3D(*system);
  ASSERT_TRUE(hierarchy.ok()) << hierarchy.status();
  ASSERT_GE(hierarchy->levels.size(), 3);
  for (std::size_t ordinal = 0; ordinal + 1 < hierarchy->levels.size(); ++ordinal) {
    const auto& fine = hierarchy->levels[ordinal];
    const auto& coarse = hierarchy->levels[ordinal + 1];
    std::vector<double> coarse_pressure(coarse.coordinates.size());
    for (std::size_t unknown = 0; unknown < coarse_pressure.size(); ++unknown) {
      coarse_pressure[unknown] = static_cast<double>(unknown % 7) - 3.0;
    }
    std::vector<double> fine_pressure(fine.coordinates.size());
    for (std::size_t unknown = 0; unknown < fine_pressure.size(); ++unknown) {
      fine_pressure[unknown] = coarse_pressure[fine.aggregate_ordinals[unknown]];
    }
    EXPECT_DOUBLE_EQ(GraphEnergy(fine, fine_pressure), GraphEnergy(coarse, coarse_pressure));
    std::vector<int> mirror_counts(fine.edges.size(), 0);
    for (std::size_t unknown = 0; unknown < fine.coordinates.size(); ++unknown) {
      for (std::size_t offset = fine.row_offsets[unknown]; offset < fine.row_offsets[unknown + 1]; ++offset) {
        const auto neighbor = fine.neighbors[offset];
        const auto& edge = fine.edges[neighbor.edge_ordinal];
        EXPECT_TRUE(
            (edge.lower_unknown_ordinal == unknown && edge.higher_unknown_ordinal == neighbor.unknown_ordinal)
            || (edge.higher_unknown_ordinal == unknown && edge.lower_unknown_ordinal == neighbor.unknown_ordinal)
        );
        ++mirror_counts[neighbor.edge_ordinal];
      }
    }
    for (const auto count : mirror_counts) {
      EXPECT_EQ(count, 2);
    }
  }
  for (const auto& block : hierarchy->levels.back().terminal_blocks) {
    EXPECT_LE(block.unknown_ordinals.size(), 32);
  }
  const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_GE(result->diagnostics.multigrid_level_count, 3);
  EXPECT_GT(result->diagnostics.iteration_count, 0);
  for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
    EXPECT_NEAR(result->cell_pressures[cell], AffinePressure(problem.cells[cell].cell_index), 5e-5);
  }
}

TEST(MacPressure3D, SignedCoordinatePlateauUsesLinkedPairCoarsening) {
  auto problem = MakeAffineProblem(-1, 2, true);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto hierarchy = pressure_internal::BuildMacPressureHierarchy3D(*system);
  ASSERT_TRUE(hierarchy.ok()) << hierarchy.status();
  ASSERT_EQ(hierarchy->levels.size(), 2);
  const auto& fine = hierarchy->levels.front();
  EXPECT_LT(hierarchy->levels.back().coordinates.size(), fine.coordinates.size());
  for (std::size_t unknown = 0; unknown < fine.coordinates.size(); ++unknown) {
    bool has_linked_partner = false;
    for (std::size_t offset = fine.row_offsets[unknown]; offset < fine.row_offsets[unknown + 1]; ++offset) {
      has_linked_partner |=
          fine.aggregate_ordinals[unknown] == fine.aggregate_ordinals[fine.neighbors[offset].unknown_ordinal];
    }
    EXPECT_TRUE(has_linked_partner);
  }
}

TEST(MacPressure3D, EliminatedArticulationPinRetainsConductanceInSeparateSingletons) {
  auto problem = MakeAffineProblem(0, 2, false);
  problem.cells.resize(3);  // (0,0,0), (1,0,0), (0,1,0): the lowest cell joins two arms.
  const auto contains = [&](CellIndex3D index) {
    return std::ranges::any_of(problem.cells, [&](const auto& cell) {
      return cell.cell_index.x == index.x && cell.cell_index.y == index.y && cell.cell_index.z == index.z;
    });
  };
  std::erase_if(problem.faces, [&](auto& face) {
    const auto index = face.face_index.lattice_index;
    const auto axis = face.face_index.axis;
    const CellIndex3D head{index.x, index.y, index.z};
    const CellIndex3D tail{
        index.x - (axis == FaceNormalAxis::kX ? 1 : 0),
        index.y - (axis == FaceNormalAxis::kY ? 1 : 0),
        index.z - (axis == FaceNormalAxis::kZ ? 1 : 0),
    };
    const bool has_tail = contains(tail);
    const bool has_head = contains(head);
    if (!has_tail && !has_head) {
      return true;
    }
    if (!has_tail || !has_head) {
      face.open_area = 0.0;
      face.predicted_normal_velocity = 0.0;
      face.pressure_link = SealedMacPressureLink3D{};
    }
    return false;
  });
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const auto system = PrepareMacPressureSystem3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(system.ok()) << system.status();
  const auto hierarchy = pressure_internal::BuildMacPressureHierarchy3D(*system);
  ASSERT_TRUE(hierarchy.ok()) << hierarchy.status();
  EXPECT_EQ(hierarchy->unknown_to_cell, (std::vector<std::size_t>{1, 2}));
  EXPECT_EQ(hierarchy->cell_to_unknown.front(), pressure_internal::kNoPressureUnknown3D);
  ASSERT_EQ(hierarchy->levels.size(), 1);
  EXPECT_TRUE(hierarchy->levels.front().edges.empty());
  EXPECT_EQ(hierarchy->levels.front().anchor_diagonal, (std::vector{1.0, 1.0}));
  EXPECT_EQ(hierarchy->levels.front().terminal_blocks.size(), 2);
  const auto result = ProjectMacPressure3D({*grid, 1.0, problem.cells, problem.faces});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->cell_pressures, (std::vector{0.0, 1.0, 2.0}));
}

TEST(MacPressure3D, ReplacesFalseConvergedFp32ResidualAndRestartsWithinSameBudget) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1e-6,
      .target_divergence = 1e6,
      .bulk_modulus = 1e-6 / 1.1,
  }};
  const auto faces = SealedFacesAtX(0);
  const auto exhausted = ProjectMacPressure3D({*grid, 1.0, cells, faces}, {.max_iterations = 1});
  EXPECT_EQ(exhausted.status().code(), absl::StatusCode::kAborted);
  const auto result = ProjectMacPressure3D({*grid, 1.0, cells, faces});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_GE(result->diagnostics.iteration_count, 2);
  EXPECT_NEAR(result->cell_pressures.front(), 1.0 / 1.1, 2e-11);
  EXPECT_LE(result->diagnostics.max_volume_scaled_residual, 2e-5);
  EXPECT_LE(result->diagnostics.max_cell_constraint_defect, 2e-5);
}

TEST(MacPressure3D, RejectsCoefficientsOutsideFineRangeAndPreservesPreparedRangeChecks) {
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto cells = std::array{IncompressibleCellAtX(0)};
  auto faces = SealedFacesAtX(0);
  faces[1].open_area = 1.0;
  faces[1].pressure_gradient_distance = 1.0;
  faces[1].predicted_normal_velocity = 1.0;
  faces[1].pressure_link = DirichletMacPressureLink3D{0.0};
  for (const double inverse_density : {1e-100, 1e39}) {
    faces[1].inverse_density = inverse_density;
    const auto result = ProjectMacPressure3D({*grid, 1.0, cells, faces});
    EXPECT_EQ(result.status().code(), absl::StatusCode::kOutOfRange);
  }
  cells[0].fluid_volume = std::numeric_limits<double>::denorm_min();
  cells[0].target_divergence = 0.25;
  const auto result = ProjectMacPressure3D({*grid, 1.0, cells, SealedFacesAtX(0)});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kOutOfRange);
}
}  // namespace
}  // namespace fluidity
