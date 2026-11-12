#pragma once

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

#include "fluidity/host/mac_pressure_3d.h"
#include "fluidity/host/viscosity.h"
#include "gtest/gtest.h"

namespace fluidity::viscous_traction_test {

inline std::int32_t FixtureOwnerAxis(std::int32_t index) {
  return index / 4 - (index % 4 < 0 ? 1 : 0);
}

// Build an independent affine solution: each predicted face velocity is the
// analytic pressure slope, so correction leaves zero flux on every open face.
struct AffineProblem {
  std::vector<MacPressureCell3D> cells;
  std::vector<MacPressureFace3D> faces;
  std::vector<BlockIndex3D> blocks;
};

inline double AffinePressure(CellIndex3D index) {
  return index.x + 2.0 * index.y + 4.0 * index.z;
}

inline AffineProblem MakeRectangularAffineProblem(std::int32_t minimum, std::array<int, 3> extents, bool anchored) {
  AffineProblem problem;
  std::set<std::tuple<int, int, int, int>> face_keys;
  for (int z = minimum; z < minimum + extents[2]; ++z) {
    for (int y = minimum; y < minimum + extents[1]; ++y) {
      for (int x = minimum; x < minimum + extents[0]; ++x) {
        problem.cells.push_back({
            .cell_index = {x, y, z},
            .fluid_volume = 1.0,
            .bulk_modulus = std::numeric_limits<double>::infinity(),
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

inline AffineProblem MakeAffineProblem(std::int32_t minimum, std::int32_t extent, bool anchored) {
  return MakeRectangularAffineProblem(minimum, {extent, extent, extent}, anchored);
}

inline void SortFaces(AffineProblem& problem, const SparseMacGrid3D& grid) {
  std::ranges::sort(problem.faces, [&](const auto& left, const auto& right) {
    return *grid.FaceSlot(left.face_index) < *grid.FaceSlot(right.face_index);
  });
}

template <typename Project>
void ExpectPlanarExtensionTraction(Project project) {
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
            project(MacPressureInputs3D{*grid, step_duration, problem.cells, problem.faces}, *strain);
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
                  << " pressure_iterations=" << projection->pressure_iterations << std::endl;
      }
    }
  }
}

template <typename Project>
void ExpectMovingWallCouetteTraction(Project project) {
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
            project(MacPressureInputs3D{*grid, step_duration, problem.cells, problem.faces}, *strain);
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
                  << " tangential_max=" << max_shear_error << std::endl;
      }
    }
  }
}

}  // namespace fluidity::viscous_traction_test
