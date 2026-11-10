#pragma once

#include <array>
#include <iostream>
#include <vector>

#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "gtest/gtest.h"

namespace fluidity::boundary_test {

inline constexpr LiquidStateCapacities3D kTranslatedTankCapacities{2, 8, 8, 384, 0};

template <typename Freeze>
void ExpectOpenTankRejectsUnresolvedTranslatedLattice(Freeze freeze) {
  constexpr double kUnresolvedOriginX = 0x1p53;
  constexpr double kResolvedOriginX = 16;
  constexpr std::array blocks{BlockIndex3D{0, 0, 0}, BlockIndex3D{1, 0, 0}};
  const auto resolved_grid = SparseMacGrid3D::Create({{kResolvedOriginX, 0, 0}, 1}, blocks, blocks.size());
  const auto unresolved_grid = SparseMacGrid3D::Create({{kUnresolvedOriginX, 0, 0}, 1}, blocks, blocks.size());
  ASSERT_TRUE(resolved_grid.ok()) << resolved_grid.status();
  ASSERT_TRUE(unresolved_grid.ok()) << unresolved_grid.status();
  auto workspace = LiquidBoundaryWorkspace3D::Create(kTranslatedTankCapacities);
  ASSERT_TRUE(workspace.ok()) << workspace.status();

  const std::array centers{
      LiquidCenterDistance3D{{0, 0, 0}, -1},
      LiquidCenterDistance3D{{1, 0, 0}, -1},
      LiquidCenterDistance3D{{3, 0, 0}, -1},
  };
  std::vector<LiquidCornerDistance3D> corners;
  for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 5; ++x) {
        corners.push_back({{x, y, z}, -1});
      }
    }
  }
  for (const auto& center : centers) {
    const std::array candidates{center.cell_index};
    const int cell_x = center.cell_index.x;
    for (const double piston_velocity_x : {0.0, 0.125}) {
      SCOPED_TRACE(testing::Message() << "cell_x=" << cell_x << " piston_velocity_x=" << piston_velocity_x);
      const AxisAlignedLiquidTank3D resolved_tank{kResolvedOriginX, kResolvedOriginX + 4, -1, -1, 2, piston_velocity_x};
      const LiquidBoundaryGeometryViewInputs3D resolved_inputs{
          resolved_grid->view(),
          candidates,
          centers,
          corners,
          resolved_tank,
          1,
          1000,
          0,
      };
      const auto published = freeze(resolved_inputs, *workspace);
      ASSERT_TRUE(published.ok()) << published.status();
      ASSERT_EQ(published->cells.size(), 1);
      ASSERT_EQ(published->faces.size(), 6);
      EXPECT_DOUBLE_EQ(published->cells.front().fluid_volume, 1);
      for (const auto& face : published->faces) {
        const bool x_face = face.face_index.axis == FaceNormalAxis::kX;
        const int face_x = face.face_index.lattice_index.x;
        const bool sealed = x_face && (face_x == 0 || face_x == 4);
        EXPECT_DOUBLE_EQ(face.open_area, sealed ? 0 : 1);
        EXPECT_EQ(face.prescribed_wall_flux, x_face && face_x == 0 ? piston_velocity_x : 0);
      }
      const std::vector saved_faces(published->faces.begin(), published->faces.end());

      const AxisAlignedLiquidTank3D unresolved_tank{
          kUnresolvedOriginX,
          kUnresolvedOriginX + 4,
          -1,
          -1,
          2,
          piston_velocity_x,
      };
      const LiquidBoundaryGeometryViewInputs3D unresolved_inputs{
          unresolved_grid->view(),
          candidates,
          centers,
          corners,
          unresolved_tank,
          1,
          1000,
          0,
      };
      const auto rejected = freeze(unresolved_inputs, *workspace);
      EXPECT_EQ(rejected.status().code(), absl::StatusCode::kOutOfRange);
      if (rejected.ok()) {
        std::cout << "unresolved_tank_cell_x=" << cell_x << " selected_cells=" << rejected->cells.size();
        for (const auto& face : rejected->faces) {
          if (face.face_index.axis == FaceNormalAxis::kX) {
            std::cout << " x" << face.face_index.lattice_index.x << "_area=" << face.open_area << " x"
                      << face.face_index.lattice_index.x << "_flux=" << face.prescribed_wall_flux.value_or(0);
          }
        }
        std::cout << '\n';
        continue;
      }
      EXPECT_DOUBLE_EQ(published->cells.front().fluid_volume, 1);
      EXPECT_EQ(published->cells.front().cell_index.x, cell_x);
      EXPECT_DOUBLE_EQ(published->center_distances.front().signed_distance, -1);
      for (std::size_t ordinal = 0; ordinal < saved_faces.size(); ++ordinal) {
        EXPECT_EQ(published->faces[ordinal].face_slot, saved_faces[ordinal].face_slot);
        EXPECT_DOUBLE_EQ(published->faces[ordinal].open_area, saved_faces[ordinal].open_area);
        EXPECT_DOUBLE_EQ(published->faces[ordinal].blocked_area, saved_faces[ordinal].blocked_area);
        EXPECT_EQ(published->faces[ordinal].prescribed_wall_flux, saved_faces[ordinal].prescribed_wall_flux);
      }
    }
  }
}

template <typename Freeze>
void ExpectOpenTankPreservesLatticeResiduals(Freeze freeze) {
  struct Fixture {
    double origin_x;
    double cell_width;
    int cell_x;
    double tank_low_x;
    double tank_high_x;
    double expected_in_tank_x_extent;
  };
  constexpr double kCoarseOriginX = 0x1p52;
  constexpr std::array fixtures{
      // Exact low face is origin+1.25, strictly inside. Adding the origin
      // in binary64 rounds that face onto the piston at origin+1.
      Fixture{kCoarseOriginX, 1.25, 1, kCoarseOriginX + 1, kCoarseOriginX + 8, 1.25},
      // Moving the same low wall to origin+2 leaves only the last half meter
      // of the cell, including its exact high x face at origin+2.5.
      Fixture{kCoarseOriginX, 1.25, 1, kCoarseOriginX + 2, kCoarseOriginX + 8, 0.5},
      // At the high face, exact 3*h is below its binary64 rounded product.
      // The translation by 0.25 is exact after multiplication rounds.
      Fixture{0.25, 0.1, 2, -1, 0x1.199999999999ap-1, 0.1},
  };
  auto workspace = LiquidBoundaryWorkspace3D::Create(kTranslatedTankCapacities);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(testing::Message() << "origin_x=" << fixture.origin_x << " cell_width=" << fixture.cell_width);
    const auto grid =
        SparseMacGrid3D::Create({{fixture.origin_x, 0, 0}, fixture.cell_width}, std::array{BlockIndex3D{0, 0, 0}}, 1);
    ASSERT_TRUE(grid.ok()) << grid.status();
    const std::array candidates{CellIndex3D{fixture.cell_x, 0, 0}};
    const std::array centers{LiquidCenterDistance3D{candidates.front(), -1}};
    std::vector<LiquidCornerDistance3D> corners;
    for (int z = 0; z < 2; ++z) {
      for (int y = 0; y < 2; ++y) {
        for (int x = fixture.cell_x; x <= fixture.cell_x + 1; ++x) {
          corners.push_back({{x, y, z}, -1});
        }
      }
    }
    const AxisAlignedLiquidTank3D tank{fixture.tank_low_x, fixture.tank_high_x, -1, -1, 2, 0.125};
    const LiquidBoundaryGeometryViewInputs3D inputs{
        grid->view(),
        candidates,
        centers,
        corners,
        tank,
        1,
        1000,
        0,
    };
    const auto frozen = freeze(inputs, *workspace);
    ASSERT_TRUE(frozen.ok()) << frozen.status();
    ASSERT_EQ(frozen->cells.size(), 1);
    ASSERT_EQ(frozen->faces.size(), 6);
    const double full_face_area = fixture.cell_width * fixture.cell_width;
    const double expected_volume = full_face_area * fixture.expected_in_tank_x_extent;
    EXPECT_DOUBLE_EQ(frozen->cells.front().fluid_volume, expected_volume);
    for (const auto& face : frozen->faces) {
      const bool x_face = face.face_index.axis == FaceNormalAxis::kX;
      const bool low_x_face = x_face && face.face_index.lattice_index.x == fixture.cell_x;
      const bool piston_face = low_x_face && fixture.expected_in_tank_x_extent < fixture.cell_width;
      double expected_open_area = fixture.cell_width * fixture.expected_in_tank_x_extent;
      if (x_face) {
        expected_open_area = piston_face ? 0 : full_face_area;
      }
      EXPECT_DOUBLE_EQ(face.open_area, expected_open_area)
          << "axis=" << static_cast<int>(face.face_index.axis) << " face_x=" << face.face_index.lattice_index.x;
      EXPECT_EQ(face.prescribed_wall_flux, piston_face ? full_face_area * 0.125 : 0);
    }
  }
}

}  // namespace fluidity::boundary_test
