#pragma once

#include <array>
#include <tuple>

#include "fluidity/host/liquid_substep_backend_3d.h"
#include "gtest/gtest.h"

namespace fluidity::tank_staging_test {

// No particle contributes candidate support. High x and z faces cross signed
// block boundaries. The surface needs an exterior air center and derivatives.
inline void ExpectPlaneSupport(const LiquidSubstepBackend3D& operations, LiquidStepOptions3D options) {
  const LiquidState3D initial;
  LiquidSceneInputs3D scene;
  scene.field = LiquidHorizontalPlaneSurface3D{0.75};
  scene.obstacle = AxisAlignedLiquidTank3D{-1, 0, -1, 3, 4, 0};
  scene.world_acceleration = {};
  scene.surface_tension = 0.125;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, operations);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  const auto& state = scheduler->committed();
  EXPECT_EQ(state.clock.epoch, 1);
  EXPECT_TRUE(state.particles.values().empty());
  constexpr std::array expected_cells{CellIndex3D{-1, -1, 3}, CellIndex3D{-1, 0, 3}};
  ASSERT_EQ(state.cells.size(), expected_cells.size());
  for (std::size_t index = 0; index < expected_cells.size(); ++index) {
    const auto expected = expected_cells[index];
    const auto& cell = state.cells[index];
    EXPECT_EQ(
        std::tie(cell.cell_index.x, cell.cell_index.y, cell.cell_index.z),
        std::tie(expected.x, expected.y, expected.z)
    );
    EXPECT_EQ(cell.occupancy, LiquidCellOccupancy3D::kLiquid);
    EXPECT_DOUBLE_EQ(cell.previous_pressure, 0);
  }
  constexpr std::array expected_blocks{
      BlockIndex3D{-1, -1, 0},
      BlockIndex3D{0, -1, 0},
      BlockIndex3D{-1, 0, 0},
      BlockIndex3D{0, 0, 0},
      BlockIndex3D{-1, -1, 1},
      BlockIndex3D{-1, 0, 1},
  };
  ASSERT_EQ(state.blocks.size(), expected_blocks.size());
  ASSERT_EQ(state.faces.size(), expected_blocks.size() * 192);
  for (std::size_t index = 0; index < expected_blocks.size(); ++index) {
    const auto expected = expected_blocks[index];
    const auto actual = state.blocks[index].block_index;
    EXPECT_EQ(std::tie(actual.x, actual.y, actual.z), std::tie(expected.x, expected.y, expected.z));
  }
  for (const auto& face : state.faces.values()) {
    EXPECT_EQ(face.side_tag, 0);
  }
}

inline void ExpectWallSamples(const LiquidSubstepBackend3D& operations) {
  const BlockIndex3D block{};
  const auto grid = SparseMacGridView3D::Create({{}, 1}, {&block, 1});
  ASSERT_TRUE(grid.ok()) << grid.status();
  std::array<LiquidBoundaryFaceTag3D, 192> tags{};
  std::array<LiquidPrescribedWallSample3D, 192> walls{};
  struct FaceOracle {
    FaceIndex3D face;
    Vec3 normal;
    double piston_velocity_x;
    bool eligible;
  };
  for (const double piston_x : {0.875, 1.125}) {
    const AxisAlignedLiquidTank3D tank{piston_x, 3, 1, 1, 3, 0.125};
    const auto sampled = operations.sample_faces(operations.context, *grid, tank, 0, tags, walls);
    ASSERT_TRUE(sampled.ok()) << sampled;
    const std::array oracles{
        FaceOracle{{FaceNormalAxis::kX, {1, 1, 1}}, {1, 0, 0}, 0.125, piston_x < 1},
        FaceOracle{{FaceNormalAxis::kX, {3, 1, 1}}, {-1, 0, 0}, 0, false},
        FaceOracle{{FaceNormalAxis::kY, {1, 1, 1}}, {0, 1, 0}, 0, false},
        FaceOracle{{FaceNormalAxis::kX, {3, 3, 1}}, {-1, 0, 0}, 0, false},
        FaceOracle{{FaceNormalAxis::kZ, {1, 1, 1}}, {0, 0, 1}, 0, false},
        FaceOracle{{FaceNormalAxis::kZ, {1, 1, 3}}, {0, 0, -1}, 0, false},
        FaceOracle{{FaceNormalAxis::kX, {1, 3, 1}}, {1, 0, 0}, 0.125, piston_x < 1},
    };
    for (const auto& oracle : oracles) {
      const auto slot = grid->FaceSlot(oracle.face);
      ASSERT_TRUE(slot.ok());
      EXPECT_EQ(tags[*slot].side_tag, 0);
      EXPECT_EQ(tags[*slot].structurally_eligible, oracle.eligible);
      const auto& wall = walls[*slot];
      EXPECT_TRUE(wall.available);
      EXPECT_EQ(wall.side_tag, 0);
      EXPECT_EQ(wall.unit_normal.x, oracle.normal.x);
      EXPECT_EQ(wall.unit_normal.y, oracle.normal.y);
      EXPECT_EQ(wall.unit_normal.z, oracle.normal.z);
      EXPECT_EQ(wall.wall_velocity.x, oracle.piston_velocity_x);
      EXPECT_EQ(wall.wall_velocity.y, 0);
      EXPECT_EQ(wall.wall_velocity.z, 0);
    }
    EXPECT_EQ(
        operations.sample_faces(operations.context, *grid, tank, 1, {}, walls).code(),
        absl::StatusCode::kInvalidArgument
    );
  }
}

inline void ExpectTranslatedFaceUsesLatticeResidual(const LiquidSubstepBackend3D& operations) {
  constexpr double origin_x = 0x1p52;
  const BlockIndex3D block{};
  const auto grid = SparseMacGridView3D::Create({{origin_x, 0, 0}, 1.25}, {&block, 1});
  ASSERT_TRUE(grid.ok()) << grid.status();
  struct Oracle {
    FaceIndex3D face;
    double piston_offset_from_origin;
  };
  constexpr std::array oracles{
      Oracle{{FaceNormalAxis::kX, {1, 0, 0}}, 1},
      Oracle{{FaceNormalAxis::kY, {2, 1, 0}}, 3},
  };
  std::array<LiquidBoundaryFaceTag3D, 192> tags{};
  std::array<LiquidPrescribedWallSample3D, 192> walls{};
  for (const auto& oracle : oracles) {
    const double piston_x = origin_x + oracle.piston_offset_from_origin;
    const auto rounded_anchor = grid->FaceWorldAnchor(oracle.face);
    ASSERT_TRUE(rounded_anchor.ok());
    EXPECT_DOUBLE_EQ(rounded_anchor->x, piston_x);
    const auto slot = grid->FaceSlot(oracle.face);
    ASSERT_TRUE(slot.ok());
    const AxisAlignedLiquidTank3D tank{piston_x, origin_x + 6, 0, -2, 3, 0.125};
    const auto sampled = operations.sample_faces(operations.context, *grid, tank, 0, tags, walls);
    ASSERT_TRUE(sampled.ok()) << sampled;
    // The normal and transverse x anchors are 1.25 m and 3.125 m above
    // origin_x, placing each inside its piston despite the rounded anchor.
    EXPECT_TRUE(tags[*slot].structurally_eligible);
    EXPECT_TRUE(walls[*slot].available);
    EXPECT_DOUBLE_EQ(walls[*slot].unit_normal.x, 1);
    EXPECT_DOUBLE_EQ(walls[*slot].wall_velocity.x, 0.125);
  }
}

inline void ExpectParticleUpdateUsesBothVelocityFields(const LiquidSubstepBackend3D& operations) {
  LiquidParticleState3D particle;
  particle.world_position = {1, 1.125, 1.5};
  particle.pic_initialized = true;
  LiquidBoundaryParticleGather3D gathered;
  gathered.new_pic_velocity = {-0.375, -0.5, 0.25};
  gathered.particle_minus_old_pic_velocity = {-0.25, -0.25, 0.5};
  gathered.velocity_gradient.x_row.x = 0.25;
  constexpr double liquid_distance = -0.25;
  const auto updated = operations.update_particles(
      operations.context,
      {
          .gathered = {&gathered, 1},
          .liquid_distances = {&liquid_distance, 1},
          .obstacle = AxisAlignedLiquidTank3D{0.875, 3, 1, 1, 3, 0.125},
          .cell_width = 1,
          .step_duration = 0.25,
          .particle_energy_blend = 1,
          .friction_coefficient = 0,
          .penetration_tolerance = 0,
      },
      {&particle, 1}
  );
  ASSERT_TRUE(updated.ok()) << updated;
  // The advecting PIC velocity reaches both planes at the step end. The
  // retained stored velocity still carries its independent z residual.
  EXPECT_DOUBLE_EQ(particle.world_position.x, 0.90625);
  EXPECT_DOUBLE_EQ(particle.world_position.y, 1);
  EXPECT_DOUBLE_EQ(particle.world_position.z, 1.5625);
  EXPECT_DOUBLE_EQ(particle.world_velocity.x, 0.125);
  EXPECT_DOUBLE_EQ(particle.world_velocity.y, 0);
  EXPECT_DOUBLE_EQ(particle.world_velocity.z, 0.745);
}

}  // namespace fluidity::tank_staging_test
