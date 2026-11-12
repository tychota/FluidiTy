#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "../src/graded_liquid_boundary_geometry_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(GradedLiquidBoundaryGeometry3D, ClipsCanonicalSignedCoarseFineWorldSubfaces) {
  constexpr std::array<AdaptiveMacCellKey3D, 5> keys{{
      {{-2, 0, 0}, 1},
      {{0, 0, 0}, 0},
      {{0, 1, 0}, 0},
      {{0, 0, 1}, 0},
      {{0, 1, 1}, 0},
  }};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto geometry = FreezeGradedPlanarLiquidGeometry3D(
      *topology,
      {3, -4, 8},
      {.unit_normal = {0.6, 0.8, 0}, .start_plane_coordinate = -1, .wall_velocity = {0.5, 0, 0}},
      {.unit_normal = {0, 0, 1}, .plane_coordinate_m = 8.75},
      0.25
  );
  ASSERT_TRUE(geometry.ok()) << geometry.status();
  ASSERT_EQ(geometry->leaves().size(), keys.size());
  EXPECT_DOUBLE_EQ(geometry->leaves()[0].world_lower_m.x, 1);
  EXPECT_DOUBLE_EQ(geometry->leaves()[0].world_upper_m.x, 3);
  EXPECT_NEAR(geometry->leaves()[0].fluid_volume_m3, 1.125, 1e-14);
  EXPECT_NEAR(geometry->leaves()[1].fluid_volume_m3, 0.625, 1e-14);
  EXPECT_NEAR(geometry->leaves()[2].fluid_volume_m3, 0.75, 1e-14);
  EXPECT_DOUBLE_EQ(geometry->leaves()[3].fluid_volume_m3, 0);
  EXPECT_DOUBLE_EQ(geometry->leaves()[4].fluid_volume_m3, 0);

  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(ownership.ok()) << ownership.status();
  ASSERT_GE(topology->interior_faces().size(), 4);
  constexpr std::array open_areas{0.375, 0.75, 0.0, 0.0};
  constexpr std::array solid_areas{0.375, 0.0, 0.0, 0.0};
  for (std::size_t ordinal = 0; ordinal < open_areas.size(); ++ordinal) {
    const auto& topology_face = topology->interior_faces()[ordinal];
    const auto& face = geometry->subfaces()[ownership->interior_face_slots[ordinal]];
    EXPECT_EQ(topology_face.tail_cell_ordinal, 0);
    EXPECT_EQ(topology_face.head_cell_ordinal, ordinal + 1);
    EXPECT_EQ(face.face_index.axis, FaceNormalAxis::kX);
    EXPECT_FALSE(face.exterior);
    EXPECT_DOUBLE_EQ(face.world_lower_m.x, 3);
    EXPECT_DOUBLE_EQ(face.world_upper_m.x, 3);
    EXPECT_DOUBLE_EQ(face.full_area_m2, 1);
    EXPECT_NEAR(face.wet_open_area_m2, open_areas[ordinal], 1e-14);
    if (open_areas[ordinal] > 0) {
      ASSERT_TRUE(face.wet_open_centroid_m);
      EXPECT_DOUBLE_EQ(face.wet_open_centroid_m->x, 3.0);
      EXPECT_NEAR(face.wet_open_centroid_m->y, ordinal == 0 ? -3.25 : -2.5, 1e-14);
      EXPECT_NEAR(face.wet_open_centroid_m->z, 8.375, 1e-14);
    } else {
      EXPECT_FALSE(face.wet_open_centroid_m);
    }
    EXPECT_NEAR(face.wet_solid_area_m2, solid_areas[ordinal], 1e-14);
    EXPECT_NEAR(face.wall_normal_flux_m3_per_s, 0.5 * solid_areas[ordinal], 1e-14);
  }
  const std::vector<GradedLiquidPressureCellFields3D> cells(keys.size());
  const std::vector<GradedLiquidPressureSubfaceFields3D> faces(geometry->subfaces().size());
  EXPECT_EQ(
      PrepareGradedPlanarLiquidPressure3D(*geometry, cells, faces).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}

TEST(GradedLiquidBoundaryGeometry3D, AdmitsKnownAirOutsideSignedCoarseFinePressureSubset) {
  constexpr std::array keys{AdaptiveMacCellKey3D{{-2, -2, 0}, 1}, AdaptiveMacCellKey3D{{-2, 0, 0}, 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto freeze = [&](double liquid_height, double wall_height, double outward_y = 1) {
    return FreezeGradedPlanarLiquidGeometry3D(
        *topology,
        {},
        {.unit_normal = {0, 1, 0}, .start_plane_coordinate = wall_height},
        {.unit_normal = {0, outward_y, 0}, .plane_coordinate_m = liquid_height},
        0.25
    );
  };
  const auto geometry = freeze(0, -10);
  ASSERT_TRUE(geometry.ok()) << geometry.status();
  ASSERT_FALSE(geometry->leaves()[0].known_air);
  ASSERT_TRUE(geometry->leaves()[1].known_air);
  std::array<LiquidAdaptiveCellState3D, 2> accepted{{
      {.key = keys[0], .inventory = {.mass_kg = 1, .reference_volume_m3 = 1, .current_volume_m3 = 1}},
      {.key = keys[1], .occupancy = LiquidCellOccupancy3D::kAir},
  }};
  std::array<GradedLiquidPressureCellFields3D, 2> cells{};
  const std::vector<GradedLiquidPressureSubfaceFields3D> faces(geometry->subfaces().size());
  const auto subset = PrepareGradedWetPressureSubset3D(*geometry, accepted, cells, faces);
  ASSERT_TRUE(subset.ok()) << subset.status();
  EXPECT_EQ(subset->leaf_ordinals, std::vector<std::size_t>{0});
  ASSERT_EQ(subset->system.cells().size(), 1);
  EXPECT_DOUBLE_EQ(subset->system.cells()[0].fluid_volume, 8);
  EXPECT_LT(subset->system.links().size(), faces.size());
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(ownership.ok()) << ownership.status();
  ASSERT_EQ(ownership->interior_face_slots.size(), 1);
  const auto shared = std::ranges::find(
      subset->system.links(),
      ownership->interior_face_slots[0],
      &pressure_internal::FrozenMacPressureLink3D::face_slot
  );
  ASSERT_NE(shared, subset->system.links().end());
  EXPECT_EQ(shared->tail_cell_ordinal, 0);
  EXPECT_FALSE(shared->head_cell_ordinal);
  EXPECT_DOUBLE_EQ(shared->open_area, 0);
  const auto stationary = pressure_internal::EvaluateMacPressure3D(subset->system, std::array{0.0});
  ASSERT_TRUE(stationary.ok()) << stationary.status();
  EXPECT_EQ(stationary->corrected_face_velocities.size(), subset->system.links().size());
  EXPECT_DOUBLE_EQ(stationary->cell_constraint_defects[0], 0);
  auto unmasked_faces = faces;
  unmasked_faces[ownership->interior_face_slots[0]].predicted_normal_velocity_m_per_s = -1;
  EXPECT_EQ(
      PrepareGradedWetPressureSubset3D(*geometry, accepted, cells, unmasked_faces).status().code(),
      absl::StatusCode::kFailedPrecondition
  );

  const auto reversed_geometry = freeze(0, -10, -1);
  ASSERT_TRUE(reversed_geometry.ok()) << reversed_geometry.status();
  auto reversed_leaves = accepted;
  std::swap(reversed_leaves[0].occupancy, reversed_leaves[1].occupancy);
  std::swap(reversed_leaves[0].inventory, reversed_leaves[1].inventory);
  const auto reversed = PrepareGradedWetPressureSubset3D(*reversed_geometry, reversed_leaves, cells, faces);
  ASSERT_TRUE(reversed.ok()) << reversed.status();
  EXPECT_EQ(reversed->leaf_ordinals, std::vector<std::size_t>{1});
  EXPECT_DOUBLE_EQ(reversed->system.cells()[0].fluid_volume, 1);
  const auto reversed_shared = std::ranges::find(
      reversed->system.links(),
      ownership->interior_face_slots[0],
      &pressure_internal::FrozenMacPressureLink3D::face_slot
  );
  ASSERT_NE(reversed_shared, reversed->system.links().end());
  EXPECT_FALSE(reversed_shared->tail_cell_ordinal);
  EXPECT_EQ(reversed_shared->head_cell_ordinal, 0);

  const auto wetting = freeze(0.5, -10);
  ASSERT_TRUE(wetting.ok()) << wetting.status();
  const auto drying = freeze(-2, -10);
  ASSERT_TRUE(drying.ok()) << drying.status();
  const auto blocked = freeze(0, 1);
  ASSERT_TRUE(blocked.ok()) << blocked.status();
  ASSERT_FALSE(blocked->leaves()[1].known_air);
  for (const auto* unsupported : {&*wetting, &*drying, &*blocked}) {
    EXPECT_EQ(
        PrepareGradedWetPressureSubset3D(*unsupported, accepted, cells, faces).status().code(),
        absl::StatusCode::kFailedPrecondition
    );
  }
  cells[1].target_divergence_s_inv = 1;
  EXPECT_EQ(
      PrepareGradedWetPressureSubset3D(*geometry, accepted, cells, faces).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  cells[1] = {};
  accepted[1].inventory.enthalpy_j = -1;
  EXPECT_EQ(
      PrepareGradedWetPressureSubset3D(*geometry, accepted, cells, faces).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_DOUBLE_EQ(accepted[1].inventory.enthalpy_j, -1);
}

TEST(GradedLiquidBoundaryGeometry3D, EdgeAndCornerContactBlockNoPositiveMeasure) {
  constexpr std::array keys{AdaptiveMacCellKey3D{{0, 0, 0}, 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const double edge_normal = std::sqrt(0.5);
  const double corner_normal = std::sqrt(1.0 / 3.0);
  const std::array normals{Vec3{edge_normal, edge_normal, 0}, Vec3{corner_normal, corner_normal, corner_normal}};
  for (const Vec3 normal : normals) {
    const auto geometry = FreezeGradedPlanarLiquidGeometry3D(
        *topology,
        {},
        {.unit_normal = normal, .start_plane_coordinate = 0, .wall_velocity = {1, -1, 0.25}},
        {.unit_normal = {0, 0, 1}, .plane_coordinate_m = 1},
        0.5
    );
    ASSERT_TRUE(geometry.ok()) << geometry.status();
    EXPECT_NEAR(geometry->leaves()[0].fluid_volume_m3, 1, 1e-14);
    for (const auto& face : geometry->subfaces()) {
      const bool surface_plateau = face.face_index.axis == FaceNormalAxis::kZ && face.world_lower_m.z == 1;
      EXPECT_NEAR(face.wet_open_area_m2, surface_plateau ? 0 : 1, 1e-14);
      EXPECT_EQ(face.wet_open_centroid_m.has_value(), !surface_plateau);
      EXPECT_DOUBLE_EQ(face.wet_solid_area_m2, 0);
      EXPECT_DOUBLE_EQ(face.wall_normal_flux_m3_per_s, 0);
    }
  }
}

TEST(GradedLiquidBoundaryGeometry3D, FlatCoplanarWallOwnsOnlyItsWetArea) {
  constexpr std::array keys{AdaptiveMacCellKey3D{{0, 0, 0}, 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto geometry = FreezeGradedPlanarLiquidGeometry3D(
      *topology,
      {-2, 4, 1},
      {.unit_normal = {1, 0, 0}, .start_plane_coordinate = -2, .wall_velocity = {-0.25, 0, 0}},
      {.unit_normal = {0, 0, 1}, .plane_coordinate_m = 1.5},
      0.25
  );
  ASSERT_TRUE(geometry.ok()) << geometry.status();
  EXPECT_NEAR(geometry->leaves()[0].fluid_volume_m3, 0.5, 1e-14);
  const auto& low_x = geometry->subfaces()[0];
  EXPECT_EQ(low_x.face_index.axis, FaceNormalAxis::kX);
  EXPECT_DOUBLE_EQ(low_x.world_lower_m.x, -2);
  EXPECT_DOUBLE_EQ(low_x.wet_open_area_m2, 0);
  EXPECT_FALSE(low_x.wet_open_centroid_m);
  EXPECT_DOUBLE_EQ(low_x.wet_solid_area_m2, 0.5);
  EXPECT_DOUBLE_EQ(low_x.wall_normal_flux_m3_per_s, -0.125);
  EXPECT_DOUBLE_EQ(geometry->subfaces()[1].wet_open_area_m2, 0.5);
  EXPECT_DOUBLE_EQ(geometry->subfaces()[1].wall_normal_flux_m3_per_s, 0);
}

TEST(GradedLiquidBoundaryGeometry3D, MeasuresTriangularWetApertureCentroid) {
  constexpr std::array keys{AdaptiveMacCellKey3D{{0, 0, 0}, 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const double normal_component = std::sqrt(0.5);
  const auto geometry = FreezeGradedPlanarLiquidGeometry3D(
      *topology,
      {},
      {.unit_normal = {normal_component, normal_component, 0},
       .start_plane_coordinate = 1.5 * normal_component,
       .wall_velocity = {}},
      {.unit_normal = {0, 0, 1}, .plane_coordinate_m = 1.0},
      0.25
  );
  ASSERT_TRUE(geometry.ok()) << geometry.status();
  const auto face = std::ranges::find_if(geometry->subfaces(), [](const auto& subface) {
    return subface.face_index.axis == FaceNormalAxis::kZ && subface.world_lower_m.z == 0;
  });
  ASSERT_NE(face, geometry->subfaces().end());
  EXPECT_NEAR(face->wet_open_area_m2, 0.125, 1e-14);
  ASSERT_TRUE(face->wet_open_centroid_m);
  EXPECT_NEAR(face->wet_open_centroid_m->x, 5.0 / 6.0, 1e-14);
  EXPECT_NEAR(face->wet_open_centroid_m->y, 5.0 / 6.0, 1e-14);
  EXPECT_DOUBLE_EQ(face->wet_open_centroid_m->z, 0);
}

TEST(GradedLiquidBoundaryGeometry3D, ComposesSignedWallIncidenceAndIndependentSource) {
  constexpr std::array<AdaptiveMacCellKey3D, 2> keys{{
      {{-1, 0, 0}, 0},
      {{0, 0, 0}, 0},
  }};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto geometry = FreezeGradedPlanarLiquidGeometry3D(
      *topology,
      {},
      {.unit_normal = {0, 1, 0}, .start_plane_coordinate = 0.5, .wall_velocity = {0.5, 0.25, 0}},
      {.unit_normal = {0, 0, 1}, .plane_coordinate_m = 2},
      0.25
  );
  ASSERT_TRUE(geometry.ok()) << geometry.status();
  std::array<GradedLiquidPressureCellFields3D, 2> cells{};
  std::vector<GradedLiquidPressureSubfaceFields3D> faces(geometry->subfaces().size());
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(ownership.ok()) << ownership.status();
  const auto interior_slot = ownership->interior_face_slots[0];
  faces[interior_slot].inverse_density_m3_per_kg = 1;
  EXPECT_DOUBLE_EQ(geometry->subfaces()[interior_slot].wall_normal_flux_m3_per_s, 0.25);
  EXPECT_EQ(
      PrepareGradedPlanarLiquidPressure3D(*geometry, cells, faces).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  for (auto& cell : cells) {
    cell.target_divergence_s_inv = -0.5;
  }
  const auto system = PrepareGradedPlanarLiquidPressure3D(*geometry, cells, faces);
  ASSERT_TRUE(system.ok()) << system.status();
  EXPECT_DOUBLE_EQ(system->cells()[0].integrated_target_flux, -0.25);
  EXPECT_DOUBLE_EQ(system->cells()[1].integrated_target_flux, -0.25);
  const auto& link = system->links()[interior_slot];
  EXPECT_EQ(link.tail_cell_ordinal, 0);
  EXPECT_EQ(link.head_cell_ordinal, 1);
  EXPECT_DOUBLE_EQ(link.prescribed_wall_flux, 0.25);
  EXPECT_DOUBLE_EQ(link.open_area, 0.5);
  const std::array zero_pressure{0.0, 0.0};
  const auto projection = pressure_internal::EvaluateMacPressure3D(*system, zero_pressure);
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_NEAR(projection->cell_constraint_defects[0], 0, 1e-14);
  EXPECT_NEAR(projection->cell_constraint_defects[1], 0, 1e-14);

  // An exterior pressure is caller-owned and keeps only the measured wet opening.
  const auto low_x_slot = ownership->exterior_faces.front().face_slot;
  faces[low_x_slot].inverse_density_m3_per_kg = 1;
  faces[low_x_slot].exterior_pressure = GradedLiquidPressureExterior3D{.pressure_pa = 2, .distance_m = 0.5};
  const auto anchored = PrepareGradedPlanarLiquidPressure3D(*geometry, cells, faces);
  ASSERT_TRUE(anchored.ok()) << anchored.status();
  EXPECT_DOUBLE_EQ(anchored->links()[low_x_slot].open_area, 0.5);
  EXPECT_TRUE(anchored->gauge_cell_ordinals().empty());

  const std::array<LiquidAdaptiveCellState3D, 2> accepted{{
      {.key = keys[0], .inventory = {.mass_kg = 0.5, .reference_volume_m3 = 0.5, .current_volume_m3 = 0.5}},
      {.key = keys[1], .inventory = {.mass_kg = 0.5, .reference_volume_m3 = 0.5, .current_volume_m3 = 0.5}},
  }};
  const auto subset = PrepareGradedWetPressureSubset3D(*geometry, accepted, cells, faces);
  ASSERT_TRUE(subset.ok()) << subset.status();
  EXPECT_EQ(subset->leaf_ordinals, (std::vector<std::size_t>{0, 1}));
  const auto anchored_projection = pressure_internal::EvaluateMacPressure3D(*anchored, zero_pressure);
  const auto subset_projection = pressure_internal::EvaluateMacPressure3D(subset->system, zero_pressure);
  ASSERT_TRUE(anchored_projection.ok()) << anchored_projection.status();
  ASSERT_TRUE(subset_projection.ok()) << subset_projection.status();
  EXPECT_EQ(anchored_projection->corrected_face_velocities, subset_projection->corrected_face_velocities);
  EXPECT_EQ(anchored_projection->true_residuals, subset_projection->true_residuals);
}

TEST(GradedLiquidBoundaryGeometry3D, MeasuresResolvedWorldExtentBeforeCheckingNominalGraphMetric) {
  constexpr std::array keys{AdaptiveMacCellKey3D{{0, 0, 0}, 0}};
  constexpr double nominal_width_m = 0.1;
  const auto topology = AdaptiveMacTopology3D::Create(keys, nominal_width_m);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto geometry = FreezeGradedPlanarLiquidGeometry3D(
      *topology,
      {1e12, 0, 0},
      {.unit_normal = {0, 1, 0}, .start_plane_coordinate = -1},
      {.unit_normal = {0, 1, 0}, .plane_coordinate_m = 1},
      1.0
  );
  ASSERT_TRUE(geometry.ok()) << geometry.status();
  const auto& leaf = geometry->leaves()[0];
  const double x_extent_m = leaf.world_upper_m.x - leaf.world_lower_m.x;
  EXPECT_NE(x_extent_m, nominal_width_m);
  EXPECT_DOUBLE_EQ(leaf.fluid_volume_m3, x_extent_m * nominal_width_m * nominal_width_m);
  const std::array<GradedLiquidPressureCellFields3D, 1> cells{};
  const std::vector<GradedLiquidPressureSubfaceFields3D> faces(geometry->subfaces().size());
  EXPECT_EQ(
      PrepareGradedPlanarLiquidPressure3D(*geometry, cells, faces).status().code(),
      absl::StatusCode::kOutOfRange
  );
}

TEST(GradedLiquidBoundaryGeometry3D, RefusesWorldCoordinatesThatCollapseLeafBounds) {
  constexpr std::array keys{AdaptiveMacCellKey3D{{0, 0, 0}, 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1.0);
  ASSERT_TRUE(topology.ok()) << topology.status();
  EXPECT_EQ(
      FreezeGradedPlanarLiquidGeometry3D(
          *topology,
          {std::numeric_limits<double>::max(), 0, 0},
          {.unit_normal = {1, 0, 0}, .start_plane_coordinate = 0},
          {.unit_normal = {0, 0, 1}, .plane_coordinate_m = 1},
          1.0
      )
          .status()
          .code(),
      absl::StatusCode::kOutOfRange
  );
}

}  // namespace
}  // namespace fluidity
