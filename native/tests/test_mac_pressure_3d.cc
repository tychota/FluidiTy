#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <tuple>
#include <vector>

#include "../src/mac_pressure_3d_multigrid.h"
#include "../src/mac_pressure_3d_system.h"
#include "../src/mac_viscous_pressure_3d.h"
#include "absl/status/status.h"
#include "fluidity/host/mac_pressure_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "fluidity/host/viscosity.h"
#include "gtest/gtest.h"
#include "mac_pressure_3d_fixture.h"
#include "mac_pressure_acoustic_cases.h"

namespace fluidity {
namespace {

TEST(MacPressure3D, BackwardEulerStandingWaveMatchesAnalyticMode) {
  auto workspace = MacPressureWorkspace3D::Create(16, 15);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  pressure_test::ExpectBackwardEulerAcousticColumn(
      pressure_test::AcousticInitialPressure::kStandingWave,
      [&](const auto& system) {
        return workspace->ProjectPrepared(system);
      }
  );
}

TEST(MacPressure3D, BackwardEulerPressurePulseMatchesModalOracle) {
  auto workspace = MacPressureWorkspace3D::Create(16, 15);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  pressure_test::ExpectBackwardEulerAcousticColumn(
      pressure_test::AcousticInitialPressure::kLocalizedPulse,
      [&](const auto& system) {
        return workspace->ProjectPrepared(system);
      }
  );
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

std::int32_t FixtureOwnerAxis(std::int32_t index) {
  return index / 4 - (index % 4 < 0 ? 1 : 0);
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

// Build an independent affine solution: each predicted face velocity is the
// analytic pressure slope, so correction leaves zero flux on every open face.
struct AffineProblem {
  std::vector<MacPressureCell3D> cells;
  std::vector<MacPressureFace3D> faces;
  std::vector<BlockIndex3D> blocks;
};

double AffinePressure(CellIndex3D index) {
  return index.x + 2.0 * index.y + 4.0 * index.z;
}

AffineProblem MakeRectangularAffineProblem(std::int32_t minimum, std::array<int, 3> extents, bool anchored) {
  AffineProblem problem;
  std::set<std::tuple<int, int, int, int>> face_keys;
  for (int z = minimum; z < minimum + extents[2]; ++z) {
    for (int y = minimum; y < minimum + extents[1]; ++y) {
      for (int x = minimum; x < minimum + extents[0]; ++x) {
        problem.cells.push_back({
            .cell_index = {x, y, z},
            .fluid_volume = 1.0,
            .bulk_modulus = kInfinity,
        });
        for (int axis = 0; axis < 3; ++axis) {
          for (int side = 0; side < 2; ++side) {
            face_keys.emplace(axis, x + (axis == 0 ? side : 0), y + (axis == 1 ? side : 0), z + (axis == 2 ? side : 0));
          }
        }
      }
    }
  }
  std::set<std::tuple<int, int, int>> block_keys;
  for (const auto& [axis, x, y, z] : face_keys) {
    const std::array coordinates{x, y, z};
    const bool low = coordinates[axis] == minimum;
    const bool high = coordinates[axis] == minimum + extents[axis];
    const bool boundary = low || high;
    MacPressureFace3D face{
        .face_index = {static_cast<FaceNormalAxis>(axis), {x, y, z}},
        .open_area = boundary && !anchored ? 0.0 : 1.0,
        .inverse_density = 1.0,
        .pressure_gradient_distance = 1.0,
        .predicted_normal_velocity = boundary && !anchored ? 0.0 : static_cast<double>(1 << axis),
        .pressure_link = InternalMacPressureLink3D{0.0},
    };
    if (boundary && anchored) {
      const CellIndex3D exterior{
          x - (axis == 0 && low ? 1 : 0),
          y - (axis == 1 && low ? 1 : 0),
          z - (axis == 2 && low ? 1 : 0),
      };
      face.pressure_link = DirichletMacPressureLink3D{AffinePressure(exterior)};
    } else if (boundary) {
      face.pressure_link = SealedMacPressureLink3D{};
    }
    problem.faces.push_back(face);
    block_keys.emplace(FixtureOwnerAxis(x), FixtureOwnerAxis(y), FixtureOwnerAxis(z));
  }
  for (const auto& [x, y, z] : block_keys) {
    problem.blocks.push_back({x, y, z});
  }
  return problem;
}

AffineProblem MakeAffineProblem(std::int32_t minimum, std::int32_t extent, bool anchored) {
  return MakeRectangularAffineProblem(minimum, {extent, extent, extent}, anchored);
}

void SortFaces(AffineProblem& problem, const SparseMacGrid3D& grid) {
  std::ranges::sort(problem.faces, [&](const auto& left, const auto& right) {
    return *grid.FaceSlot(left.face_index) < *grid.FaceSlot(right.face_index);
  });
}

TEST(MacPressure3D, CoupledViscosityRecoversPlanarExtensionTractionAtHighContrast) {
  for (const int resolution : {16, 32}) {
    SCOPED_TRACE(resolution);
    const double cell_width = 1.0 / resolution;
    for (const double step_duration : {0.005, 0.0025}) {
      SCOPED_TRACE(step_duration);
      for (const double viscosity_ratio : {1.0, 10.0, 1000.0}) {
        SCOPED_TRACE(viscosity_ratio);
        auto problem = MakeRectangularAffineProblem(0, {resolution, resolution, 1}, false);
        const auto grid = SparseMacGrid3D::Create({{0.0, 0.0, 0.0}, cell_width}, problem.blocks, problem.blocks.size());
        ASSERT_TRUE(grid.ok()) << grid.status();
        SortFaces(problem, *grid);
        std::vector<MacViscosityCell3D> material;
        for (auto& cell : problem.cells) {
          cell.fluid_volume = cell_width * cell_width * cell_width;
          const double viscosity = cell.cell_index.x < resolution / 2 ? 1.0 : viscosity_ratio;
          material.push_back({.cell_index = cell.cell_index, .dynamic_viscosity = viscosity});
        }
        for (auto& face : problem.faces) {
          const auto world_position = grid->FaceWorldAnchor(face.face_index);
          ASSERT_TRUE(world_position.ok()) << world_position.status();
          const std::array extension_velocity{world_position->x - 0.5, 0.5 - world_position->y, 0.0};
          const double normal_velocity = extension_velocity[static_cast<std::size_t>(face.face_index.axis)];
          if (face.open_area == 0.0) {
            face.wall_normal_velocity = normal_velocity;
          } else {
            face.open_area = cell_width * cell_width;
            face.pressure_gradient_distance = cell_width;
            face.predicted_normal_velocity = normal_velocity;
          }
        }
        const auto strain = AssembleFullCellMacStrain3D(grid->view(), material);
        ASSERT_TRUE(strain.ok()) << strain.status();
        const auto projection =
            ProjectMacViscousPressure3D({*grid, step_duration, problem.cells, problem.faces}, *strain);
        ASSERT_TRUE(projection.ok()) << projection.status();
        std::vector<double> face_velocity(grid->face_count());
        for (std::size_t face = 0; face < problem.faces.size(); ++face) {
          const auto slot = grid->FaceSlot(problem.faces[face].face_index);
          ASSERT_TRUE(slot.ok()) << slot.status();
          face_velocity[*slot] = projection->face_velocities[face];
        }
        std::array<double, 2> normalized_normal_error_squared{};
        double max_normal_error = 0.0;
        std::array<std::size_t, 2> normal_sample_count{};
        for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
          EXPECT_NEAR(projection->cell_constraint_defects[cell], 0.0, 2e-5);
          const int cell_x = problem.cells[cell].cell_index.x;
          if (cell_x != resolution / 2 - 1 && cell_x != resolution / 2) {
            continue;
          }
          const auto& normal_row = (*strain)[3 * cell];
          const auto& low = normal_row.terms[0];
          const auto& high = normal_row.terms[1];
          const double normal_strain =
              low.coefficient * face_velocity[low.face_slot] + high.coefficient * face_velocity[high.face_slot];
          const double pressure = 4.0 + projection->cell_pressures[cell];
          const double traction = 2.0 * material[cell].dynamic_viscosity * normal_strain - pressure;
          const double normalized_error = (traction + 2.0) / 2.0;
          const std::size_t side = cell_x < resolution / 2 ? 0 : 1;
          normalized_normal_error_squared[side] += normalized_error * normalized_error;
          max_normal_error = std::max(max_normal_error, std::abs(normalized_error));
          ++normal_sample_count[side];
        }
        double normal_rms = 0.0;
        for (std::size_t side = 0; side < normal_sample_count.size(); ++side) {
          ASSERT_EQ(normal_sample_count[side], static_cast<std::size_t>(resolution));
          const double side_rms = std::sqrt(normalized_normal_error_squared[side] / normal_sample_count[side]);
          normal_rms = std::max(normal_rms, side_rms);
        }
        double max_tangential_error = 0.0;
        std::array<std::size_t, 2> shear_sample_count{};
        for (const auto& row : *strain) {
          const int low_x = row.low_cell.x;
          if ((low_x != resolution / 2 - 2 && low_x != resolution / 2) || row.first_axis != FaceNormalAxis::kX
              || row.second_axis != FaceNormalAxis::kY) {
            continue;
          }
          // Use same-phase edge quadrature one cell to either side of the interface.
          double shear = 0.0;
          for (const auto& term : row.terms) {
            shear += term.coefficient * face_velocity[term.face_slot];
          }
          const double normalized_traction =
              row.quadrature_weight * shear / (4.0 * cell_width * cell_width * cell_width);
          max_tangential_error = std::max(max_tangential_error, std::abs(normalized_traction));
          const std::size_t side = low_x < resolution / 2 ? 0 : 1;
          ++shear_sample_count[side];
        }
        for (const std::size_t sample_count : shear_sample_count) {
          ASSERT_EQ(sample_count, static_cast<std::size_t>(resolution - 1));
        }
        EXPECT_LE(normal_rms, 0.05);
        EXPECT_LE(max_normal_error, 0.10);
        EXPECT_LE(max_tangential_error, 0.01);
        std::cout << "extension resolution=" << resolution << " dt=" << step_duration
                  << " viscosity_ratio=" << viscosity_ratio << " normal_rms=" << normal_rms
                  << " normal_max=" << max_normal_error << " tangential_max=" << max_tangential_error
                  << " pressure_iterations=" << projection->pressure_iterations << '\n';
      }
    }
  }
}

TEST(MacPressure3D, CoupledViscosityRecoversMovingWallCouetteTraction) {
  constexpr double kStressScale = 2.0;
  for (const int resolution : {16, 32}) {
    SCOPED_TRACE(resolution);
    const double width = 1.0 / resolution;
    const double volume = width * width * width;
    for (const double step_duration : {0.005, 0.0025}) {
      SCOPED_TRACE(step_duration);
      for (const double viscosity_ratio : {1.0, 10.0, 1000.0}) {
        SCOPED_TRACE(viscosity_ratio);
        // An x-invariant strip uses exact inlet/outlet traces in place of periodic links.
        auto problem = MakeRectangularAffineProblem(0, {2, resolution, 1}, false);
        const auto grid = SparseMacGrid3D::Create({{0.0, 0.0, 0.0}, width}, problem.blocks, problem.blocks.size());
        ASSERT_TRUE(grid.ok()) << grid.status();
        SortFaces(problem, *grid);
        const double expected_shear = 2.0 / (1.0 + 1.0 / viscosity_ratio);
        std::vector<MacViscosityCell3D> material;
        for (auto& cell : problem.cells) {
          cell.fluid_volume = volume;
          const double viscosity = cell.cell_index.y < resolution / 2 ? 1.0 : viscosity_ratio;
          material.push_back({.cell_index = cell.cell_index, .dynamic_viscosity = viscosity});
        }
        for (auto& face : problem.faces) {
          const auto position = grid->FaceWorldAnchor(face.face_index);
          ASSERT_TRUE(position.ok()) << position.status();
          double velocity = 0.0;
          if (face.face_index.axis == FaceNormalAxis::kX) {
            velocity =
                expected_shear * (std::min(position->y, 0.5) + std::max(position->y - 0.5, 0.0) / viscosity_ratio);
          }
          if (face.open_area == 0.0) {
            face.wall_normal_velocity = velocity;
          } else {
            face.open_area = width * width;
            face.pressure_gradient_distance = width;
            face.predicted_normal_velocity = velocity;
          }
        }
        auto strain = AssembleFullCellMacStrain3D(grid->view(), material);
        ASSERT_TRUE(strain.ok()) << strain.status();
        for (const bool upper_wall : {false, true}) {
          const int wall_cell_y = upper_wall ? resolution - 1 : 0;
          const auto slot = grid->FaceSlot({FaceNormalAxis::kX, {1, wall_cell_y, 0}});
          ASSERT_TRUE(slot.ok()) << slot.status();
          const auto row = MakeMacTangentialWallStrain3D(
              {1, wall_cell_y, 0},
              FaceNormalAxis::kX,
              FaceNormalAxis::kY,
              *slot,
              width,
              upper_wall ? viscosity_ratio : 1.0,
              upper_wall ? 1.0 : 0.0,
              upper_wall
          );
          ASSERT_TRUE(row.ok()) << row.status();
          strain->push_back(*row);
        }
        // The prescribed inlet and outlet also carry the analytic shear
        // traction. Half-cell rows balance the transverse momentum there.
        for (const bool right_boundary : {false, true}) {
          const int boundary_cell_x = right_boundary ? 1 : 0;
          for (int y_face = 1; y_face < resolution; ++y_face) {
            const double viscosity_low = y_face - 1 < resolution / 2 ? 1.0 : viscosity_ratio;
            const double viscosity_high = y_face < resolution / 2 ? 1.0 : viscosity_ratio;
            const double viscosity = 2.0 / (1.0 / viscosity_low + 1.0 / viscosity_high);
            const auto slot = grid->FaceSlot({FaceNormalAxis::kY, {boundary_cell_x, y_face, 0}});
            ASSERT_TRUE(slot.ok()) << slot.status();
            const auto row = MakeMacTangentialWallStrain3D(
                {boundary_cell_x, y_face, 0},
                FaceNormalAxis::kY,
                FaceNormalAxis::kX,
                *slot,
                width,
                viscosity,
                0.0,
                right_boundary,
                expected_shear / viscosity
            );
            ASSERT_TRUE(row.ok()) << row.status();
            strain->push_back(*row);
          }
        }
        const auto projection =
            ProjectMacViscousPressure3D({*grid, step_duration, problem.cells, problem.faces}, *strain);
        ASSERT_TRUE(projection.ok()) << projection.status();
        std::vector<double> face_velocity(grid->face_count());
        for (std::size_t face = 0; face < problem.faces.size(); ++face) {
          const auto slot = grid->FaceSlot(problem.faces[face].face_index);
          ASSERT_TRUE(slot.ok()) << slot.status();
          face_velocity[*slot] = projection->face_velocities[face];
        }
        double max_normal_error = 0.0;
        for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
          const auto& normal_row = (*strain)[3 * cell + 1];
          const auto& low = normal_row.terms[0];
          const auto& high = normal_row.terms[1];
          const double normal_strain =
              low.coefficient * face_velocity[low.face_slot] + high.coefficient * face_velocity[high.face_slot];
          const double traction_error =
              (2.0 * material[cell].dynamic_viscosity * normal_strain - projection->cell_pressures[cell])
              / kStressScale;
          max_normal_error = std::max(max_normal_error, std::abs(traction_error));
        }
        double max_shear_error = 0.0;
        std::size_t shear_samples = 0;
        for (const auto& row : *strain) {
          if (row.first_axis != FaceNormalAxis::kX || row.second_axis != FaceNormalAxis::kY) {
            continue;
          }
          double shear = row.prescribed_strain;
          for (const auto& term : std::span(row.terms).first(row.term_count)) {
            shear += term.coefficient * face_velocity[term.face_slot];
          }
          const double strain_volume = row.term_count == 1 ? 0.5 * volume : volume;
          const double traction = row.quadrature_weight * shear / (2.0 * strain_volume);
          max_shear_error = std::max(max_shear_error, std::abs(traction - expected_shear) / kStressScale);
          ++shear_samples;
        }
        ASSERT_EQ(shear_samples, static_cast<std::size_t>(3 * resolution - 1));
        EXPECT_LE(max_normal_error, 0.05);
        EXPECT_LE(max_shear_error, 0.05);
        std::cout << "couette resolution=" << resolution << " dt=" << step_duration
                  << " viscosity_ratio=" << viscosity_ratio << " normal_max=" << max_normal_error
                  << " tangential_max=" << max_shear_error << '\n';
      }
    }
  }
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
