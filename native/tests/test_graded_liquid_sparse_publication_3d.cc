#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "../src/graded_liquid_sparse_publication_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(GradedLiquidSparsePublication3D, PublishesSparsePressureByCanonicalSlotOnlyAfterPreflight) {
  constexpr std::array keys{AdaptiveMacCellKey3D{{-2, -2, 0}, 1}, AdaptiveMacCellKey3D{{-2, 0, 0}, 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(ownership.ok()) << ownership.status();
  ASSERT_EQ(ownership->interior_face_slots.size(), 1);
  const auto geometry = FreezeGradedPlanarLiquidGeometry3D(
      *topology,
      {},
      {.unit_normal = {0, 1, 0}, .start_plane_coordinate = -10},
      {.unit_normal = {0, -1, 0}, .plane_coordinate_m = 0},
      0.25
  );
  ASSERT_TRUE(geometry.ok()) << geometry.status();
  const std::size_t face_count = geometry->subfaces().size();
  auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(keys.size());
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(face_count);
  ASSERT_TRUE(cells.ok() && faces.ok());
  ASSERT_TRUE(cells->Resize(keys.size()).ok() && faces->Resize(face_count).ok());
  (*cells)[0] = {.key = keys[0], .occupancy = LiquidCellOccupancy3D::kAir, .previous_pressure = 5};
  (*cells)[1] = {
      .key = keys[1],
      .inventory = {.mass_kg = 1, .reference_volume_m3 = 1, .current_volume_m3 = 1},
      .previous_pressure = -3,
  };
  for (std::size_t slot = 0; slot < face_count; ++slot) {
    (*faces)[slot] = {
        .carrier_velocity = geometry->subfaces()[slot].wet_open_area_m2 > 0 ? 0 : static_cast<double>(slot) + 1,
        .validity = LiquidFaceValidity3D::kValid,
        .side_tag = -static_cast<std::int64_t>(slot),
        .structurally_eligible = slot % 2 == 0,
    };
  }
  LiquidState3D trial{
      .clock = {.time = 1, .frame_ordinal = 1, .accepted_step_ordinal = 1, .epoch = 1},
      .adaptive_cells = std::move(*cells),
      .adaptive_faces = std::move(*faces),
  };
  const auto outlet = std::ranges::find_if(ownership->exterior_faces, [](const auto& face) {
    return face.cell_ordinal == 1 && face.face_index.axis == FaceNormalAxis::kX && !face.cell_is_tail;
  });
  ASSERT_NE(outlet, ownership->exterior_faces.end());
  const std::size_t outlet_slot = outlet->face_slot;
  trial.adaptive_faces[outlet_slot].validity = LiquidFaceValidity3D::kLowMass;
  trial.adaptive_faces[outlet_slot].structurally_eligible = true;
  trial.adaptive_faces[outlet_slot].carrier_velocity = 0.5;
  const auto before = CloneLiquidState3D(trial, {.cells = keys.size(), .faces = face_count});
  ASSERT_TRUE(before.ok()) << before.status();

  std::array<GradedLiquidPressureCellFields3D, 2> pressure_cells{};
  pressure_cells[1].target_divergence_s_inv = 0.125;
  std::vector<GradedLiquidPressureSubfaceFields3D> pressure_faces(face_count);
  pressure_faces[outlet_slot] = {
      .inverse_density_m3_per_kg = 1,
      .predicted_normal_velocity_m_per_s = 0.5,
      .exterior_pressure = GradedLiquidPressureExterior3D{.pressure_pa = 2, .distance_m = 0.5},
  };
  const auto subset =
      PrepareGradedWetPressureSubset3D(*geometry, trial.adaptive_cells.values(), pressure_cells, pressure_faces);
  ASSERT_TRUE(subset.ok()) << subset.status();
  const auto links = subset->system.links();
  const auto outlet_link =
      std::ranges::find(links, outlet_slot, &pressure_internal::FrozenMacPressureLink3D::face_slot);
  ASSERT_NE(outlet_link, links.end());
  EXPECT_NE(static_cast<std::size_t>(outlet_link - links.begin()), outlet_slot);
  auto workspace = MacPressureWorkspace3D::Create(subset->system.cells().size(), links.size());
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const auto solved = workspace->ProjectPrepared(subset->system);
  ASSERT_TRUE(solved.ok()) << solved.status();
  const GradedSparsePressureResultView3D values{solved->cell_pressures, solved->corrected_face_velocities};

  const auto reject = [&](const absl::StatusOr<GradedSparsePressureResultView3D>& projection,
                          const GradedWetPressureSubset3D& graph,
                          LiquidClock3D clock) {
    const auto status = PublishGradedSparsePressure3D(trial, clock, *geometry, graph, projection);
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(trial == *before);
    return status.code();
  };
  EXPECT_EQ(reject(absl::AbortedError("pressure retry"), *subset, trial.clock), absl::StatusCode::kAborted);
  auto stale_clock = trial.clock;
  ++stale_clock.epoch;
  EXPECT_EQ(reject(values, *subset, stale_clock), absl::StatusCode::kFailedPrecondition);
  auto wrong_row = *subset;
  wrong_row.leaf_ordinals[0] = 0;
  EXPECT_EQ(reject(values, wrong_row, trial.clock), absl::StatusCode::kInvalidArgument);

  const std::array one_row{MacPressureCell3D{
      .cell_index = keys[1].lower_corner,
      .fluid_volume = 1,
      .target_divergence = 0.125,
      .bulk_modulus = std::numeric_limits<double>::infinity(),
  }};
  const std::array wrong_link{pressure_internal::ExplicitMacPressureBoundaryFace3D{
      .face_slot = outlet_slot + 1,
      .cell_ordinal = 0,
      .cell_is_tail = false,
      .full_face_area = 1,
      .open_area = 1,
      .inverse_density = 1,
      .pressure_gradient_distance = 0.5,
      .predicted_normal_velocity = 0.5,
      .prescribed_wall_flux = 0,
      .exterior_pressure = 2,
  }};
  const auto wrong_system = pressure_internal::PrepareExplicitMacPressureSystem3D(0.25, one_row, {}, wrong_link);
  ASSERT_TRUE(wrong_system.ok()) << wrong_system.status();
  const GradedWetPressureSubset3D wrong_slot{.system = *wrong_system, .leaf_ordinals = {1}};
  EXPECT_EQ(
      reject(
          GradedSparsePressureResultView3D{values.cell_pressures, values.corrected_face_velocities.first(1)},
          wrong_slot,
          trial.clock
      ),
      absl::StatusCode::kInvalidArgument
  );

  auto nonfinite =
      std::vector<double>(values.corrected_face_velocities.begin(), values.corrected_face_velocities.end());
  nonfinite.back() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
      reject(GradedSparsePressureResultView3D{values.cell_pressures, nonfinite}, *subset, trial.clock),
      absl::StatusCode::kInvalidArgument
  );
  const auto aliased_pressure = std::span(&trial.adaptive_cells[1].previous_pressure, 1);
  EXPECT_EQ(
      reject(
          GradedSparsePressureResultView3D{aliased_pressure, values.corrected_face_velocities},
          *subset,
          trial.clock
      ),
      absl::StatusCode::kInvalidArgument
  );

  auto no_carrier = CloneLiquidState3D(trial, {.cells = keys.size(), .faces = face_count});
  ASSERT_TRUE(no_carrier.ok()) << no_carrier.status();
  no_carrier->adaptive_faces[outlet_slot].validity = LiquidFaceValidity3D::kMissing;
  const auto missing_before = CloneLiquidState3D(*no_carrier, {.cells = keys.size(), .faces = face_count});
  ASSERT_TRUE(missing_before.ok()) << missing_before.status();
  EXPECT_EQ(
      PublishGradedSparsePressure3D(*no_carrier, no_carrier->clock, *geometry, *subset, values).code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_TRUE(*no_carrier == *missing_before);

  const auto published = PublishGradedSparsePressure3D(trial, trial.clock, *geometry, *subset, values);
  ASSERT_TRUE(published.ok()) << published;
  EXPECT_DOUBLE_EQ(trial.adaptive_cells[0].previous_pressure, 0);
  EXPECT_DOUBLE_EQ(trial.adaptive_cells[1].previous_pressure, 3.25);
  EXPECT_DOUBLE_EQ(trial.adaptive_cells[1].inventory.mass_kg, 1);
  EXPECT_EQ(trial.clock, before->clock);
  for (std::size_t slot = 0; slot < face_count; ++slot) {
    const auto& face = trial.adaptive_faces[slot];
    EXPECT_EQ(face.side_tag, before->adaptive_faces[slot].side_tag);
    EXPECT_EQ(face.structurally_eligible, before->adaptive_faces[slot].structurally_eligible);
    if (slot == outlet_slot) {
      EXPECT_EQ(face.validity, LiquidFaceValidity3D::kLowMass);
      EXPECT_DOUBLE_EQ(face.carrier_velocity, -0.125);
    } else if (geometry->subfaces()[slot].wet_open_area_m2 > 0) {
      EXPECT_EQ(face.validity, before->adaptive_faces[slot].validity);
      EXPECT_DOUBLE_EQ(face.carrier_velocity, 0);
    } else {
      EXPECT_EQ(face.validity, LiquidFaceValidity3D::kMissing);
      EXPECT_DOUBLE_EQ(face.carrier_velocity, 0);
    }
  }
}

}  // namespace
}  // namespace fluidity
