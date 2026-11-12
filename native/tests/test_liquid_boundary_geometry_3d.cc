#include <limits>
#include <numbers>
#include <numeric>

#include "fluidity/base/sampled_solid_liquid_geometry_3d.h"
#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "fluidity/host/plastic_solid_boundary_3d.h"
#include "fluidity/host/rigid_body_3d.h"
#include "liquid_boundary_geometry_3d_cases.h"
namespace fluidity::boundary_test {
namespace {
constexpr auto kOwningFreeze =
    static_cast<absl::StatusOr<FrozenLiquidBoundaryGeometry3D> (*)(const LiquidBoundaryGeometryInputs3D&)>(
        FreezeLiquidBoundaryGeometry3D
    );
constexpr auto kOwningLinks = static_cast<
    absl::StatusOr<LiquidPressureProblem3D> (*)(const FrozenLiquidBoundaryGeometry3D&, const LiquidPressureFields3D&)>(
    BuildLiquidPressureLinks3D
);
}  // namespace

TEST(LiquidBoundaryGeometry3D, SampledObliqueSolidSharesApertureAcrossCells) {
  boundary_internal::SampledSolidLiquidCell first{.cell_width = 1.0};
  boundary_internal::SampledSolidLiquidCell second{.cell_width = 1.0};
  for (int corner = 0; corner < 8; ++corner) {
    const double x = corner & 1;
    const double y = (corner >> 1) & 1;
    first.liquid_distance[corner] = y - 0.75;
    second.liquid_distance[corner] = y - 0.75;
    first.solid_distance[corner] = (x + y - 1.25) / std::numbers::sqrt2;
    second.solid_distance[corner] = (x + y - 0.25) / std::numbers::sqrt2;
  }
  const auto left = boundary_internal::MeasureSampledSolidLiquidCell(first);
  const auto right = boundary_internal::MeasureSampledSolidLiquidCell(second);
  ASSERT_TRUE(left.ok());
  ASSERT_TRUE(right.ok());
  EXPECT_NEAR(left->wet_volume, 0.125, 2e-14);
  EXPECT_NEAR(right->wet_volume, 0.71875, 2e-14);
  EXPECT_NEAR(left->open_area[1], 0.75, 2e-14);
  EXPECT_EQ(left->open_area[1], right->open_area[0]);
  EXPECT_NEAR(left->open_area[4], 0.28125, 2e-14);
  EXPECT_EQ(left->open_area[0], 0.0);
}

TEST(LiquidBoundaryGeometry3D, SampledSolidCornerKeepsPiecewiseLinearMeasure) {
  boundary_internal::SampledSolidLiquidCell cell{.cell_width = 2.0};
  cell.liquid_distance.fill(-1);
  cell.solid_distance.fill(-1);
  cell.solid_distance[7] = 1;

  const auto measures = boundary_internal::MeasureSampledSolidLiquidCell(cell);
  ASSERT_TRUE(measures.ok());
  EXPECT_NEAR(measures->wet_volume, 1.0, 2e-14);
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(measures->open_area[2 * axis], 0.0);
    EXPECT_NEAR(measures->open_area[2 * axis + 1], 1.0, 2e-14);
  }
}

TEST(LiquidBoundaryGeometry3D, SampledSolidFaceWeightsIntegrateAffineWallFlux) {
  const auto weights = boundary_internal::MeasureSampledSolidFaceWeights(1.0, {-0.5, 0.5, -0.5, 0.5});
  ASSERT_TRUE(weights.ok());
  constexpr std::array expected_weights{11.0 / 48, 1.0 / 48, 7.0 / 48, 5.0 / 48};
  constexpr std::array normal_velocities{2.0, 5.0, -2.0, 1.0};
  for (std::size_t corner = 0; corner < weights->size(); ++corner) {
    EXPECT_NEAR((*weights)[corner], expected_weights[corner], 2e-14);
  }
  EXPECT_NEAR(std::accumulate(weights->begin(), weights->end(), 0.0), 0.5, 2e-14);
  EXPECT_NEAR(std::inner_product(weights->begin(), weights->end(), normal_velocities.begin(), 0.0), 0.375, 2e-14);

  const auto closed = boundary_internal::MeasureSampledSolidFaceWeights(2.0, {0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(closed.ok());
  EXPECT_NEAR(std::accumulate(closed->begin(), closed->end(), 0.0), 4.0, 2e-14);
  const auto open = boundary_internal::MeasureSampledSolidFaceWeights(1.0, {1.0, 1.0, 1.0, 1.0});
  ASSERT_TRUE(open.ok());
  EXPECT_EQ(std::accumulate(open->begin(), open->end(), 0.0), 0.0);
}

TEST(LiquidBoundaryGeometry3D, SnowParticleSamplesDriveCutCellFreeze) {
  const auto grid = *SparseMacGrid3D::Create({{}, 1.0}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  const std::array cells{CellIndex3D{0, 0, 0}};
  const std::array centers{
      LiquidCenterDistance3D{{0, 0, -1}, 1},
      LiquidCenterDistance3D{{0, -1, 0}, 1},
      LiquidCenterDistance3D{{-1, 0, 0}, 1},
      LiquidCenterDistance3D{{0, 0, 0}, -1},
      LiquidCenterDistance3D{{1, 0, 0}, 1},
      LiquidCenterDistance3D{{0, 1, 0}, 1},
      LiquidCenterDistance3D{{0, 0, 1}, 1},
  };
  std::array<LiquidCornerDistance3D, 8> corners{};
  for (int corner = 0; corner < 8; ++corner) {
    corners[corner] = {{corner & 1, (corner >> 1) & 1, (corner >> 2) & 1}, -1.0};
  }
  const std::array particles{
      PlasticSolidCheckpointParticle3D{
          .particle_id = 7,
          .body_id = 4,
          .material_id = 1,
          .transfer =
              {
                  .sorted_world_position = {},
                  .trial_world_position = {},
                  .particle_mass = 1,
                  .particle_velocity = {2, 0, 0},
                  .particle_velocity_slopes = {.x_row = {0, 4, 0}},
              },
      },
  };
  const auto sampled = SamplePlasticSolidBoundary3D(grid.view(), corners, particles, 4, 0.5);
  ASSERT_TRUE(sampled.ok()) << sampled.status();
  EXPECT_DOUBLE_EQ(sampled->corners.values()[0].signed_distance, -0.5);
  EXPECT_DOUBLE_EQ(sampled->corners.values()[2].wall_velocity.x, 6.0);
  EXPECT_EQ(sampled->particle_ids.values()[2], 7);
  const auto frozen = FreezeLiquidBoundaryGeometry3D({
      .mac_grid = grid,
      .cell_indices = cells,
      .center_distances = centers,
      .corner_distances = corners,
      .obstacle = {},
      .step_duration = 0.01,
      .liquid_density = 1.0,
      .surface_tension = 0.0,
      .sampled_solid = sampled->View(),
  });
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  EXPECT_NEAR(frozen->cells().front().fluid_volume, 1.0 - 0.125 / std::sqrt(6.0), 2e-14);
  const auto face = std::ranges::find(frozen->faces(), FaceNormalAxis::kX, [](const FrozenLiquidFace3D& f) {
    return f.face_index.axis;
  });
  ASSERT_NE(face, frozen->faces().end());
  ASSERT_TRUE(face->prescribed_wall_flux.has_value());
  const auto loads = StagePlasticSolidPressureFace3D(grid.view(), *sampled, *face, 13.0, 5.0, 0.01);
  ASSERT_TRUE(loads.ok()) << loads.status();
  double impulse = 0.0;
  double pressure_work = 0.0;
  for (const auto& load : *loads) {
    EXPECT_EQ(load.particle_id, 7);
    EXPECT_DOUBLE_EQ(load.impulse.y, 0.0);
    EXPECT_DOUBLE_EQ(load.impulse.z, 0.0);
    impulse += load.impulse.x;
    pressure_work += load.impulse.x * (2.0 + 4.0 * load.world_offset.y);
  }
  EXPECT_NEAR(impulse, 0.08 * face->blocked_area, 2e-14);
  EXPECT_NEAR(pressure_work, 0.08 * *face->prescribed_wall_flux, 2e-14);
  const std::array fields{LiquidPressureCellFields3D{0, std::numeric_limits<double>::infinity(), 0}};
  const std::vector<double> predicted_velocities(frozen->faces().size(), 0);
  const auto links = BuildLiquidPressureLinks3D(*frozen, {fields, predicted_velocities, 0});
  ASSERT_TRUE(links.ok()) << links.status();
  const auto projected = ProjectMacPressure3D({grid, 0.01, links->cells, links->faces});
  ASSERT_TRUE(projected.ok()) << projected.status();
  ASSERT_EQ(projected->cell_pressures.size(), 1);
  const FrozenLiquidBoundaryGeometryView3D geometry{
      .mac_grid = grid.view(),
      .cells = frozen->cells(),
      .faces = frozen->faces(),
      .step_duration = 0.01,
  };
  const auto batch = StagePlasticSolidPressureLoads3D(geometry, *sampled, projected->cell_pressures, 0);
  ASSERT_TRUE(batch.ok()) << batch.status();
  ASSERT_EQ(batch->size(), 12);
  double batch_work = 0;
  for (const auto& load : batch->values()) {
    EXPECT_EQ(load.particle_id, 7);
    batch_work += load.impulse.x * (2.0 + 4.0 * load.world_offset.y);
  }
  EXPECT_LT(batch_work, 0);
  EXPECT_NEAR(batch_work, -0.01 * projected->cell_pressures[0] * *face->prescribed_wall_flux, 2e-12);
  EXPECT_EQ(
      StagePlasticSolidPressureLoads3D(geometry, *sampled, {}, 0).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  auto stale_face = *face;
  stale_face.prescribed_wall_flux = *stale_face.prescribed_wall_flux + 0.1;
  EXPECT_EQ(
      StagePlasticSolidPressureFace3D(grid.view(), *sampled, stale_face, 13.0, 5.0, 0.01).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(LiquidBoundaryGeometry3D, PressureLoadedSnowTrialAdvancesWithoutChangingAcceptedState) {
  const auto grid = *SparseMacGrid3D::Create({{}, 1.0}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  const SnowParameters3D snow{2, 3, 0.025, 0.01, 4};
  const DrySolidParticle3D particle{
      .world_position = {2, 2, 2},
      .mass = 2,
      .reference_volume = 0.1,
      .elastic_deformation = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
  };
  const auto checkpoint = CapturePlasticSolidCheckpointParticle3D(7, 4, 1, snow, particle);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const std::array accepted{*checkpoint};
  const PlasticSolidPressureImpulse3D impulse{.particle_id = 7, .impulse = {0.2, 0, 0}, .world_offset = {0, 0.25, 0}};
  const std::array loads{impulse};
  auto trial = StagePressureLoadedSnowParticles3D(accepted, 4, 1, snow, 1.0, loads);
  ASSERT_TRUE(trial.ok()) << trial.status();
  ASSERT_EQ(trial->size(), 1);
  EXPECT_NEAR(trial->front().world_velocity.x, 0.1, 2e-14);
  EXPECT_NEAR(trial->front().velocity_slopes.x_row.y, 0.1, 2e-14);

  const DrySolidStep3D step{.duration = 0.01, .acoustic_cfl = 0.25};
  ASSERT_TRUE(AdvanceDrySolid3D(grid, snow, step, *trial).ok());
  EXPECT_NEAR(trial->front().world_position.x, 2.001, 2e-14);
  EXPECT_NEAR(trial->front().elastic_deformation.x_row.y, 0.001, 2e-14);

  auto unknown = impulse;
  unknown.particle_id = 9;
  const std::array invalid_loads{impulse, unknown};
  EXPECT_EQ(
      StagePressureLoadedSnowParticles3D(accepted, 4, 1, snow, 1.0, invalid_loads).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_DOUBLE_EQ(accepted.front().transfer.particle_velocity.x, 0.0);
  EXPECT_DOUBLE_EQ(accepted.front().transfer.sorted_world_position.x, 2.0);
}

TEST(LiquidBoundaryGeometry3D, AnalyticPlatenExcludesOverlappingSnowReactionArea) {
  boundary_internal::SampledSolidLiquidCell cell{.cell_width = 1.0};
  for (int corner = 0; corner < 8; ++corner) {
    cell.liquid_distance[corner] = -1.0;
    cell.solid_distance[corner] = (corner & 1) - 0.5;
  }
  const boundary_internal::ClippingPlane fluid_side{.normal = {0, 1, 0}, .plane_coordinate = 0.75};
  const auto clipped = boundary_internal::MeasureSampledSolidLiquidCell(cell, &fluid_side);
  ASSERT_TRUE(clipped.ok());
  EXPECT_NEAR(clipped->wet_volume, 0.375, 2e-14);
  EXPECT_NEAR(clipped->open_area[4], 0.375, 2e-14);
  EXPECT_EQ(clipped->open_area[3], 0.0);

  const auto snow_weights = boundary_internal::MeasureSampledSolidFaceWeights(1.0, {-0.5, 0.5, -0.5, 0.5}, &fluid_side);
  ASSERT_TRUE(snow_weights.ok());
  constexpr std::array corner_velocity{2.0, 6.0, 2.0, 6.0};
  EXPECT_NEAR(std::accumulate(snow_weights->begin(), snow_weights->end(), 0.0), 0.375, 2e-14);
  EXPECT_NEAR(
      std::inner_product(snow_weights->begin(), snow_weights->end(), corner_velocity.begin(), 0.0),
      1.125,
      2e-14
  );
}

TEST(LiquidBoundaryGeometry3D, ComposedPlatenPressureFluxKeepsSnowReactionSeparate) {
  const auto grid = *SparseMacGrid3D::Create({{}, 1.0}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  const std::array cells{CellIndex3D{0, 0, 0}};
  const std::array centers{LiquidCenterDistance3D{{0, 0, 0}, -1}};
  std::array<LiquidCornerDistance3D, 8> liquid_corners{};
  PlasticSolidBoundarySamples3D snow{
      .corners = *LiquidRecords3D<SolidCornerDistance3D>::Create(8),
      .particle_ids = *LiquidRecords3D<std::uint64_t>::Create(8),
      .particle_offsets = *LiquidRecords3D<Vec3>::Create(8),
  };
  ASSERT_TRUE(snow.corners.Resize(8).ok());
  ASSERT_TRUE(snow.particle_ids.Resize(8).ok());
  ASSERT_TRUE(snow.particle_offsets.Resize(8).ok());
  for (int corner = 0; corner < 8; ++corner) {
    const CellCornerIndex3D index{corner & 1, (corner >> 1) & 1, (corner >> 2) & 1};
    liquid_corners[corner] = {index, -1.0};
    snow.corners.values()[corner] = {index, index.x - 0.5, {0, 0, 2.0 + 4.0 * index.x}};
    snow.particle_ids.values()[corner] = 7;
  }
  const double q = 1.0 / std::sqrt(2.0);
  const LiquidObstacle3D platen = TranslatingHalfspace3D{{0, -q, -q}, -0.75 * q, {0, 0.5, 0.5}};
  const auto frozen = FreezeLiquidBoundaryGeometry3D({
      .mac_grid = grid,
      .cell_indices = cells,
      .center_distances = centers,
      .corner_distances = liquid_corners,
      .obstacle = platen,
      .step_duration = 0.01,
      .liquid_density = 1,
      .surface_tension = 0,
      .sampled_solid = snow.View(),
  });
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  EXPECT_NEAR(frozen->cells()[0].fluid_volume, 0.140625, 2e-14);
  const auto face = std::ranges::find_if(frozen->faces(), [](const FrozenLiquidFace3D& candidate) {
    return candidate.face_index.axis == FaceNormalAxis::kZ && candidate.face_index.lattice_index.z == 0;
  });
  ASSERT_NE(face, frozen->faces().end());
  ASSERT_TRUE(face->prescribed_wall_flux.has_value());
  EXPECT_NEAR(face->open_area, 0.375, 2e-14);
  EXPECT_NEAR(*face->prescribed_wall_flux, 1.25, 2e-14);
  const auto loads = StagePlasticSolidPressureFace3D(grid.view(), snow, *face, 13, 5, 0.01, platen);
  ASSERT_TRUE(loads.ok()) << loads.status();
  double impulse = 0;
  double work = 0;
  for (std::size_t i = 0; i < loads->size(); ++i) {
    impulse += (*loads)[i].impulse.z;
    work += (*loads)[i].impulse.z * snow.corners.values()[i].wall_velocity.z;
  }
  EXPECT_NEAR(impulse, 0.03, 2e-14);
  EXPECT_NEAR(work, 0.09, 2e-14);
}

TEST(LiquidBoundaryGeometry3D, SampledSolidPlateauIsClosed) {
  boundary_internal::SampledSolidLiquidCell cell{.cell_width = 1.0};
  cell.liquid_distance.fill(-1);
  const auto closed = boundary_internal::MeasureSampledSolidLiquidCell(cell);
  ASSERT_TRUE(closed.ok());
  EXPECT_EQ(closed->wet_volume, 0.0);
  for (const double open_area : closed->open_area) {
    EXPECT_EQ(open_area, 0.0);
  }
  cell.solid_distance[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(boundary_internal::MeasureSampledSolidLiquidCell(cell).ok());
}

TEST(LiquidBoundaryGeometry3D, OrientedBoxSubtractsWetVolumeAndBlocksFace) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  const auto grid = cases.Grid();
  const auto samples = cases.Sample([](double, double y, double) {
    return y - 0.75;
  });
  const LiquidBoundaryGeometryInputs3D inputs{
      grid,
      samples.candidates,
      samples.centers,
      samples.corners,
      {},
      0.01,
      1.0,
      0.0,
  };
  const OrientedFloatingBoxGeometry3D box{
      .pose =
          {
              .center = {0.0, 0.75, 0.5},
              .orientation = {.w = std::sqrt(0.9), .vector = {0.0, std::sqrt(0.1), 0.0}},
          },
      .body_half_extent = {0.25, 0.25, 0.25},
      .liquid_height = 0.75,
  };
  const auto frozen = FreezeOrientedFloatingBoxLiquidBoundaryGeometry3D(inputs, box);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  // The x=0 face and y=0.75 surface each bisect the yawed cube.
  EXPECT_NEAR(frozen->cells().front().fluid_volume, 0.71875, 1e-13);
  const auto low_x = std::ranges::find_if(frozen->faces(), [](const auto& face) {
    return face.face_index.axis == FaceNormalAxis::kX && face.face_index.lattice_index.x == -2;
  });
  ASSERT_NE(low_x, frozen->faces().end());
  EXPECT_NEAR(low_x->blocked_area, 0.40625, 1e-13);
  EXPECT_NEAR(low_x->open_area, 0.59375, 1e-13);
  const auto high_y = std::ranges::find_if(frozen->faces(), [](const auto& face) {
    return face.face_index.axis == FaceNormalAxis::kY && face.face_index.lattice_index.y == -1;
  });
  ASSERT_NE(high_y, frozen->faces().end());
  EXPECT_NEAR(high_y->open_area, 0.875, 1e-13);

  // The same link is a free-surface link when the waterplane is above the
  // MAC anchor, provided the head cell center is still in air.
  const auto higher_samples = cases.Sample([](double, double y, double) {
    return y - 1.25;
  });
  auto higher_inputs = inputs;
  higher_inputs.cell_indices = higher_samples.candidates;
  higher_inputs.center_distances = higher_samples.centers;
  higher_inputs.corner_distances = higher_samples.corners;
  auto higher_box = box;
  higher_box.pose.center.y = 1.25;
  higher_box.liquid_height = 1.25;
  const auto higher = FreezeOrientedFloatingBoxLiquidBoundaryGeometry3D(higher_inputs, higher_box);
  ASSERT_TRUE(higher.ok()) << higher.status();
  const auto higher_waterplane = std::ranges::find_if(higher->faces(), [](const auto& face) {
    return face.face_index.axis == FaceNormalAxis::kY && face.face_index.lattice_index.y == -1;
  });
  ASSERT_NE(higher_waterplane, higher->faces().end());
  EXPECT_NEAR(higher_waterplane->open_area, 0.875, 1e-13);

  const LiquidBoundaryGeometryViewInputs3D tank_inputs{
      grid.view(),
      samples.candidates,
      samples.centers,
      samples.corners,
      AxisAlignedLiquidTank3D{-0.75, 0.875, 0.125, 0.125, 0.875, 0.0},
      0.01,
      1.0,
      0.0,
  };
  auto workspace = LiquidBoundaryWorkspace3D::Create({.blocks = 8, .cells = 8, .faces = 48});
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const auto in_tank = workspace->FreezeOrientedFloatingBox(tank_inputs, box);
  ASSERT_TRUE(in_tank.ok()) << in_tank.status();
  ASSERT_EQ(in_tank->cells.size(), 1);
  EXPECT_NEAR(in_tank->cells.front().fluid_volume, 0.37890625, 1e-13);
  const auto tank_low_x = std::ranges::find_if(in_tank->faces, [](const auto& face) {
    return face.face_index.axis == FaceNormalAxis::kX && face.face_index.lattice_index.x == -2;
  });
  ASSERT_NE(tank_low_x, in_tank->faces.end());
  EXPECT_NEAR(tank_low_x->open_area, 0.3125, 1e-13);

  auto overlaps_tank = box;
  overlaps_tank.pose.center.z = 0.25;
  const auto rejected = workspace->FreezeOrientedFloatingBox(tank_inputs, overlaps_tank);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(rejected.status().message().empty());
  EXPECT_NEAR(in_tank->cells.front().fluid_volume, 0.37890625, 1e-13);
}
TEST(LiquidBoundaryGeometry3D, FiniteBoxClipsPlanarLiquidVolumeAndApertures) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  const auto grid = cases.Grid();
  const auto samples = cases.Sample([](double, double y, double) {
    return y - 0.75;
  });
  const LiquidBoundaryGeometryInputs3D inputs{
      grid,
      samples.candidates,
      samples.centers,
      samples.corners,
      {},
      0.01,
      1000.0,
      0.0,
  };
  const AxisAlignedFloatingBoxGeometry3D box{
      .world_low = {0.0, 0.25, 0.25},
      .world_high = {0.5, 1.0, 0.75},
      .liquid_height = 0.75,
  };
  const auto frozen = FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, box);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  EXPECT_DOUBLE_EQ(frozen->cells().front().fluid_volume, 0.625);
  ASSERT_EQ(frozen->faces().size(), 6);
  const auto low_x = std::ranges::find_if(frozen->faces(), [](const auto& face) {
    return face.face_index.axis == FaceNormalAxis::kX && face.face_index.lattice_index.x == -2;
  });
  ASSERT_NE(low_x, frozen->faces().end());
  EXPECT_DOUBLE_EQ(low_x->open_area, 0.625);
  EXPECT_DOUBLE_EQ(low_x->blocked_area, 0.375);
  EXPECT_DOUBLE_EQ(low_x->wall_normal_velocity, 0.0);

  auto interior = box;
  interior.world_low = {0.25, 0.25, 0.0};
  interior.world_high = {0.75, 0.5, 1.0};
  const auto connected = FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, interior);
  ASSERT_TRUE(connected.ok()) << connected.status();
  EXPECT_DOUBLE_EQ(connected->cells().front().fluid_volume, 0.625);

  auto split = box;
  split.world_low = {0.25, 0.0, 0.0};
  split.world_high = {0.5, 1.0, 1.0};
  EXPECT_EQ(
      FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, split).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  auto workspace = LiquidBoundaryWorkspace3D::Create({.blocks = 8, .cells = 8, .faces = 48});
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const LiquidBoundaryGeometryViewInputs3D view{
      grid.view(),
      samples.candidates,
      samples.centers,
      samples.corners,
      {},
      0.01,
      1000.0,
      0.0,
  };
  const auto accepted = workspace->FreezeFloatingBox(view, box);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  const auto rejected = workspace->FreezeFloatingBox(view, split);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(rejected.status().message().empty());
  EXPECT_DOUBLE_EQ(accepted->cells.front().fluid_volume, 0.625);
  auto overlap = inputs;
  overlap.obstacle = TranslatingHalfspace3D{{0.0, 1.0, 0.0}, 0.125, {}};
  EXPECT_EQ(
      FreezeFloatingBoxLiquidBoundaryGeometry3D(overlap, box).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}
TEST(LiquidBoundaryGeometry3D, DisjointBoxAndTankDriveReciprocalSolvedPressure) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  const auto grid = cases.Grid();
  const auto samples = cases.Sample([](double, double y, double) {
    return y - 0.75;
  });
  const AxisAlignedLiquidTank3D tank{0.125, 0.875, 0.125, 0.125, 0.875, 0.0};
  const LiquidBoundaryGeometryInputs3D inputs{
      grid,
      samples.candidates,
      samples.centers,
      samples.corners,
      tank,
      0.125,
      1.0,
      0.0,
  };
  const AxisAlignedFloatingBoxGeometry3D box{
      .world_low = {0.25, 0.25, 0.25},
      .world_high = {0.5, 0.875, 0.5},
      .liquid_height = 0.75,
  };
  const auto frozen = FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, box);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  EXPECT_DOUBLE_EQ(frozen->cells().front().fluid_volume, 0.3203125);
  auto problem = cases.Links(*frozen);
  ASSERT_EQ(problem.cells.size(), 1);
  const auto open_count = std::ranges::count_if(problem.faces, [](const auto& face) {
    return face.open_area > 0.0;
  });
  EXPECT_EQ(open_count, 1);
  for (const auto& face : problem.faces) {
    if (face.open_area > 0.0) {
      EXPECT_EQ(face.face_index.axis, FaceNormalAxis::kY);
      EXPECT_DOUBLE_EQ(face.open_area, 0.5625);
    }
  }

  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.375, 0.5625, 0.375},
      .linear_velocity = {0.0, 0.5, 0.0},
  };
  const auto patches = BuildFloatingBoxPressurePatches3D(
      body.body_id,
      box.world_low,
      box.world_high,
      box.liquid_height,
      grid.geometry(),
      samples.candidates
  );
  ASSERT_TRUE(patches.ok()) << patches.status();
  const auto flux = ComputeRigidBodyInwardWallFlux3D(body, problem.cells.size(), *patches);
  ASSERT_TRUE(flux.ok()) << flux.status();
  EXPECT_DOUBLE_EQ(flux->front(), -0.03125);
  problem.cells.front().integrated_inward_wall_flux = flux->front();
  const auto solved = ProjectMacPressure3D({grid, inputs.step_duration, problem.cells, problem.faces});
  ASSERT_TRUE(solved.ok()) << solved.status();
  EXPECT_DOUBLE_EQ(solved->integrated_rhs.front(), -0.03125);
  EXPECT_NEAR(solved->cell_pressures.front(), -1.0 / 9.0, 1e-7);
  const auto loaded = AccumulateRigidBodyPressure3D(body, solved->cell_pressures, 0.0, *patches);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_NEAR(loaded->force.y, -1.0 / 144.0, 1e-8);
  EXPECT_DOUBLE_EQ(loaded->force.y * body.linear_velocity.y + solved->cell_pressures.front() * flux->front(), 0.0);

  auto touching = box;
  touching.world_low.x = tank.start_low_x;
  EXPECT_EQ(
      FreezeFloatingBoxLiquidBoundaryGeometry3D(inputs, touching).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}
TEST(LiquidBoundaryGeometry3D, OpenTankClipsVolumeAndUsesPistonOnlyFlux) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.OpenTankClipsVolumeAndUsesPistonOnlyFlux();
}
TEST(LiquidBoundaryGeometry3D, OpenTankPreservesNarrowAndLiquidClippedMeasures) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.OpenTankPreservesNarrowAndLiquidClippedMeasures();
}
TEST(LiquidBoundaryGeometry3D, AnalyticPlaneFixtureAllAxesSignedIndicesAndTinyOpening) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.AnalyticPlaneFixtureAllAxesSignedIndicesAndTinyOpening();
}
TEST(LiquidBoundaryGeometry3D, ObliqueAndSimultaneousLiquidSolidClips) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ObliqueAndSimultaneousLiquidSolidClips();
}
TEST(LiquidBoundaryGeometry3D, ThinSlabInvisibleToCornersAndSplitCellRefusal) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ThinSlabInvisibleToCornersAndSplitCellRefusal();
}
TEST(LiquidBoundaryGeometry3D, SampledSolidFreezeSharesOnePressureFace) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.SampledSolidFreezeSharesOnePressureFace();
}

TEST(LiquidBoundaryGeometry3D, SharedFaceHasOneOwnerAndMatchingTriangulation) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.SharedFaceHasOneOwnerAndMatchingTriangulation();
}
TEST(LiquidBoundaryGeometry3D, PredictorDensitySetsSharedAndFreeSurfaceMobility) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  const auto grid = cases.Grid();
  auto samples = cases.IsolatedLiquid(-1.0);
  samples.candidates.push_back({-1, -2, -2});
  for (auto& center : samples.centers) {
    const auto cell = center.cell_index;
    if (cell.x == -1 && cell.y == -2 && cell.z == -2) {
      center.signed_distance = -1.0;
    }
  }
  constexpr double kStepDuration = 0.01;
  const auto frozen = cases.Freeze(grid, samples, {}, 0.0, kStepDuration);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 2U);
  std::array fields{
      LiquidPressureCellFields3D{0.0, 100.0, 3.0, 2.0},
      LiquidPressureCellFields3D{0.0, 100.0, 1.0, 6.0},
  };
  const std::vector<double> velocity(frozen->faces().size());
  const auto links = BuildLiquidPressureLinks3D(*frozen, {fields, velocity, 0.0});
  ASSERT_TRUE(links.ok()) << links.status();
  const auto projection = ProjectMacPressure3D({grid, kStepDuration, links->cells, links->faces});
  ASSERT_TRUE(projection.ok()) << projection.status();
  std::size_t shared_faces = 0;
  std::size_t surface_faces = 0;
  for (std::size_t face = 0; face < links->faces.size(); ++face) {
    const auto& link = links->faces[face];
    if (std::holds_alternative<InternalMacPressureLink3D>(link.pressure_link)) {
      ++shared_faces;
      EXPECT_EQ(link.inverse_density, 0.25);
      const double pressure_difference = projection->cell_pressures[0] - projection->cell_pressures[1];
      EXPECT_GT(pressure_difference, 1.0);
      EXPECT_NEAR(projection->corrected_face_velocities[face], kStepDuration * 0.25 * pressure_difference, 1e-12);
    } else if (std::holds_alternative<DirichletMacPressureLink3D>(link.pressure_link)) {
      ++surface_faces;
      const auto lattice = link.face_index.lattice_index;
      const int liquid_x = link.face_index.axis == FaceNormalAxis::kX && lattice.x == 0 ? -1 : lattice.x;
      EXPECT_EQ(link.inverse_density, liquid_x == -2 ? 0.5 : 1.0 / 6.0);
    }
  }
  EXPECT_EQ(shared_faces, 1U);
  EXPECT_EQ(surface_faces, 10U);
  fields[1].density = 0.0;
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(*frozen, {fields, velocity, 0.0}).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidBoundaryGeometry3D, PartitionsLiquidStressVolumesAcrossCellOctants) {
  const UniformMacGeometry3D geometry{{3.0, 1.0, 0.0}, 1.0};
  const CellIndex3D cell{-3, -1, 0};
  const std::array distances{-0.5, 0.5, 0.5, 1.5, -0.5, 0.5, 0.5, 1.5};
  const auto liquid = MeasureLiquidCellOctants3D(geometry, cell, {}, distances);
  ASSERT_TRUE(liquid.ok()) << liquid.status();
  constexpr std::array kExpected{0.0625, 0.0, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0};
  for (std::size_t octant = 0; octant < kExpected.size(); ++octant) {
    EXPECT_NEAR((*liquid)[octant], kExpected[octant], 1e-15);
  }

  const TranslatingHalfspace3D wall{{0.0, 1.0, 0.0}, 0.25, {}};
  const auto partial = MeasureLiquidCellOctants3D(geometry, cell, wall, distances);
  ASSERT_TRUE(partial.ok()) << partial.status();
  EXPECT_NEAR((*partial)[0], 0.015625, 1e-15);
  EXPECT_NEAR((*partial)[4], 0.015625, 1e-15);

  const AxisAlignedLiquidTank3D tank{0.25, 0.75, 0.0, 0.0, 1.0, 0.0};
  const std::array full{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
  const auto tank_liquid = MeasureLiquidCellOctants3D(geometry, cell, tank, full);
  ASSERT_TRUE(tank_liquid.ok()) << tank_liquid.status();
  for (const double volume : *tank_liquid) {
    EXPECT_NEAR(volume, 0.0625, 1e-15);
  }

  const std::array<double, 8> plateau{};
  const auto empty = MeasureLiquidCellOctants3D(geometry, cell, {}, plateau);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(*empty, plateau);
  const std::array nonaffine{-0.2, 0.8, -0.4, 0.2, -0.5, 0.6, -0.8, 0.9};
  const auto partial_liquid = MeasureLiquidCellOctants3D(geometry, cell, {}, nonaffine);
  ASSERT_TRUE(partial_liquid.ok()) << partial_liquid.status();
  const auto gas = MeasureLiquidAirCell3D({geometry, cell, {}, nonaffine});
  ASSERT_TRUE(gas.ok()) << gas.status();
  const double liquid_volume = std::accumulate(partial_liquid->begin(), partial_liquid->end(), 0.0);
  EXPECT_NEAR(liquid_volume + gas->gas_volume, 1.0, 1e-15);

  auto invalid = distances;
  invalid[0] = std::numeric_limits<double>::infinity();
  EXPECT_EQ(
      MeasureLiquidCellOctants3D(geometry, cell, {}, invalid).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidBoundaryGeometry3D, ClipsGasVolumeAndSharedFaceAreas) {
  LiquidAirCellMeasureInputs3D inputs{
      .geometry = {{0.0, 0.0, 0.0}, 1.0},
      .cell_index = {0, 0, 0},
      .obstacle = {},
      .liquid_corner_distances = {-0.25, 0.75, -0.25, 0.75, -0.25, 0.75, -0.25, 0.75},
  };
  const auto gas = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(gas.ok()) << gas.status();
  EXPECT_NEAR(gas->gas_volume, 0.75, 1e-15);
  EXPECT_EQ(gas->gas_face_areas, (std::array{0.0, 1.0, 0.75, 0.75, 0.75, 0.75}));
  inputs.cell_index.y = 1;
  const auto neighbor = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(neighbor.ok()) << neighbor.status();
  EXPECT_EQ(gas->gas_face_areas[3], neighbor->gas_face_areas[2]);

  inputs.cell_index.y = 0;
  inputs.obstacle = TranslatingHalfspace3D{{0.0, 1.0, 0.0}, 0.5, {}};
  const auto halfspace = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(halfspace.ok()) << halfspace.status();
  EXPECT_NEAR(halfspace->gas_volume, 0.375, 1e-15);
  EXPECT_EQ(halfspace->gas_face_areas, (std::array{0.0, 0.5, 0.0, 0.75, 0.375, 0.375}));
  inputs.obstacle = AxisAlignedLiquidTank3D{0.5, 1.0, 0.0, 0.0, 1.0, 0.0};
  const auto tank = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(tank.ok()) << tank.status();
  EXPECT_NEAR(tank->gas_volume, 0.5, 1e-15);
  EXPECT_EQ(tank->gas_face_areas, (std::array{0.0, 0.0, 0.0, 0.5, 0.0, 0.0}));
}

TEST(LiquidBoundaryGeometry3D, GasConnectivityRequiresSharedTriangleOpening) {
  LiquidAirCellMeasureInputs3D inputs{
      .geometry = {{0.0, 0.0, 0.0}, 1.0},
      .cell_index = {0, 0, 0},
      .obstacle = {},
      .liquid_corner_distances = {-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},
  };
  EXPECT_EQ(MeasureLiquidAirCell3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
  inputs.liquid_corner_distances[0] = 0.0;
  inputs.liquid_corner_distances[7] = 0.0;
  EXPECT_EQ(MeasureLiquidAirCell3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
  inputs.liquid_corner_distances[0] = 1.0;
  const auto connected = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(connected.ok()) << connected.status();
  EXPECT_GT(connected->gas_volume, 0.0);
  EXPECT_GT(connected->gas_face_areas[0], 0.0);
  EXPECT_GT(connected->gas_face_areas[1], 0.0);
}

TEST(LiquidBoundaryGeometry3D, GasMeasureRejectsSlabSplitAndIgnoresZeroPlateaus) {
  LiquidAirCellMeasureInputs3D inputs{
      .geometry = {{0.0, 0.0, 0.0}, 1.0},
      .cell_index = {0, 0, 0},
      .obstacle = TranslatingSlab3D{{1.0, 0.0, 0.0}, 0.5, 0.1, {}},
      .liquid_corner_distances = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
  };
  EXPECT_EQ(MeasureLiquidAirCell3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
  inputs.obstacle = {};
  inputs.liquid_corner_distances.fill(0.0);
  const auto plateau = MeasureLiquidAirCell3D(inputs);
  ASSERT_TRUE(plateau.ok()) << plateau.status();
  EXPECT_EQ(plateau->gas_volume, 0.0);
  EXPECT_EQ(plateau->gas_face_areas, (std::array<double, 6>{}));
}

TEST(LiquidBoundaryGeometry3D, CavityGasPressureAddsSignedCapillaryJumpOnce) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.CavityGasPressureAddsSignedCapillaryJumpOnce();
}

TEST(LiquidBoundaryGeometry3D, InternalCapillaryBoundScalesBeforeSquaring) {
  const double maximum = std::numeric_limits<double>::max();
  const auto large_density = InternalLiquidCapillaryStepBound3D(maximum, maximum, 1.0, 0.5);
  ASSERT_TRUE(large_density.ok()) << large_density.status();
  EXPECT_DOUBLE_EQ(*large_density, std::sqrt(maximum / std::numbers::pi));
  const auto thin_cell = InternalLiquidCapillaryStepBound3D(1.0, 1.0, 1.0e-200, 1.0);
  ASSERT_TRUE(thin_cell.ok()) << thin_cell.status();
  EXPECT_NEAR(*thin_cell / 1.0e-300, std::sqrt(0.5 / std::numbers::pi), 1.0e-15);
  const auto no_tension = InternalLiquidCapillaryStepBound3D(1.0, 1.0, 1.0, 0.0);
  ASSERT_TRUE(no_tension.ok()) << no_tension.status();
  EXPECT_EQ(*no_tension, std::numeric_limits<double>::infinity());
  EXPECT_EQ(InternalLiquidCapillaryStepBound3D(0.0, 1.0, 1.0, 1.0).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidBoundaryGeometry3D, SharedLiquidInterfaceUsesSeriesResistanceAndOneJump) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.SharedLiquidInterfaceUsesSeriesResistanceAndOneJump();
}

TEST(LiquidBoundaryGeometry3D, ThetaFixtureAndSignedPressureMapping) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ThetaFixtureAndSignedPressureMapping();
}
TEST(LiquidBoundaryGeometry3D, PlaneAndSphereCenteredCurvature) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.PlaneAndSphereCenteredCurvature();
}
TEST(LiquidBoundaryGeometry3D, CapillaryLiftOnceAndCorrectionMismatchControl) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.CapillaryLiftOnceAndCorrectionMismatchControl();
}
TEST(LiquidBoundaryGeometry3D, RejectsMissingSamplesOwnersOrderingAndInvalidObstacle) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.RejectsMissingSamplesOwnersOrderingAndInvalidObstacle();
}
TEST(LiquidBoundaryGeometry3D, CapillaryBoundZeroSigmaAndDegenerateDerivativeChecks) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.CapillaryBoundZeroSigmaAndDegenerateDerivativeChecks();
}
TEST(LiquidBoundaryGeometry3D, FrozenSnapshotPreservesWallFluxAndSeparatesPressureHistory) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.FrozenSnapshotPreservesWallFluxAndSeparatesPressureHistory();
}
TEST(LiquidBoundaryGeometry3D, InternalLinkUsesFullFrozenAreaAndZeroJump) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.InternalLinkUsesFullFrozenAreaAndZeroJump();
}
TEST(LiquidBoundaryGeometry3D, RejectsPressureFieldsMissingAirAndNegativeOmittedNeighbor) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.RejectsPressureFieldsMissingAirAndNegativeOmittedNeighbor();
}
TEST(LiquidBoundaryGeometry3D, RejectsUnrepresentableMeasuresAndInvalidPhysicalInputs) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.RejectsUnrepresentableMeasuresAndInvalidPhysicalInputs();
}
TEST(LiquidBoundaryGeometry3D, InteriorPositiveSigmaDoesNotRequireUnusedCurvature) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.InteriorPositiveSigmaDoesNotRequireUnusedCurvature();
}
TEST(LiquidBoundaryGeometry3D, SealedLowFaceAtMinimumSignedIndexNeedsNoExteriorSample) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.SealedLowFaceAtMinimumSignedIndexNeedsNoExteriorSample();
}
TEST(LiquidBoundaryGeometry3D, ResolvesRepresentableTinyLiquidTetrahedron) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ResolvesRepresentableTinyLiquidTetrahedron();
}
TEST(LiquidBoundaryGeometry3D, RejectsCapillaryLiftThatUnderflowsToZero) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.RejectsCapillaryLiftThatUnderflowsToZero();
}
TEST(LiquidBoundaryGeometry3D, ZeroCornerPlateauHasNoStrictlyNegativeLiquidVolume) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.ZeroCornerPlateauHasNoStrictlyNegativeLiquidVolume();
}
TEST(LiquidBoundaryGeometry3D, MixedZeroPlateauCountsOnlyStrictlyNegativeTetrahedra) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.MixedZeroPlateauCountsOnlyStrictlyNegativeTetrahedra();
}
TEST(LiquidBoundaryGeometry3D, PhysicalVolumeSurvivesUnderflowOfNormalizedCubicMeasure) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.PhysicalVolumeSurvivesUnderflowOfNormalizedCubicMeasure();
}
TEST(LiquidBoundaryGeometry3D, PhysicalFaceAreaSurvivesUnderflowOfNormalizedSquareMeasure) {
  GeometryCases cases(kOwningFreeze, kOwningLinks);
  cases.PhysicalFaceAreaSurvivesUnderflowOfNormalizedSquareMeasure();
}
}  // namespace fluidity::boundary_test
