#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include "fluidity/host/mixed_contact_3d.h"
#if defined(FLUIDITY_M5_CUDA_SCENES)
#include "fluidity/host/cuda_mac_pressure_3d.h"
#endif

namespace {

using fluidity::BlockIndex3D;
using fluidity::Mat3;
using fluidity::SparseMacGrid3D;

constexpr Mat3 kIdentity{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

bool TestPlanarMixedContactDetection() {
  fluidity::LiquidParticleState3D water{
      .id = 7,
      .world_position = {-0.0625, 0, 0},
      .world_velocity = {2, 0, 0},
      .mass = 6
  };
  fluidity::PlasticSolidCheckpointParticle3D snow{
      .particle_id = 3,
      .body_id = 9,
      .material_id = 1,
      .transfer = {.trial_world_position = {0.3125, 0, 0}, .particle_mass = 2, .particle_velocity = {-2, 0, 0}},
      .reference_volume = 0.001,
      .elastic_deformation = kIdentity,
  };
  constexpr std::array contributors{
      fluidity::ContactFaceParticle3D{3, 9, 2, 1, {-2, 0, 0}},
      fluidity::ContactFaceParticle3D{7, 11, 6, 1, {2, 0, 0}},
  };
  const fluidity::DetectedContactFace3D pair{3, contributors, 9, 11, {1, 0, 0}, 0};
  const auto reconstructed =
      fluidity::ReconstructAxisAlignedMixedContactFace3D({&water, 1}, 11, {&snow, 1}, pair, 0.0625, 0.0625);
  if (!reconstructed.ok() || reconstructed->first_interface_world_position.x != 0.25
      || reconstructed->second_plane_start_coordinate != 0) {
    return false;
  }
  const fluidity::DetectedContactFace3D reversed{3, contributors, 11, 9, {-1, 0, 0}, 0};
  const auto reversed_fronts =
      fluidity::ReconstructAxisAlignedMixedContactFace3D({&water, 1}, 11, {&snow, 1}, reversed, 0.0625, 0.0625);
  if (!reversed_fronts.ok() || reversed_fronts->first_interface_world_position.x != 0
      || reversed_fronts->second_plane_start_coordinate != -0.25) {
    return false;
  }
  auto another_snow = snow;
  another_snow.particle_id = 4;
  const std::array tied_snow{snow, another_snow};
  const std::array tied_contributors{
      contributors[0],
      fluidity::ContactFaceParticle3D{4, 9, 2, 1, {-1, 0, 0}},
      contributors[1],
  };
  const fluidity::DetectedContactFace3D ambiguous{3, tied_contributors, 9, 11, {1, 0, 0}, 0};
  if (fluidity::ReconstructAxisAlignedMixedContactFace3D({&water, 1}, 11, tied_snow, ambiguous, 0.0625, 0.0625)
          .status()
          .code()
      != absl::StatusCode::kFailedPrecondition) {
    return false;
  }
  const auto face = *reconstructed;
  const auto missed = fluidity::ApplyPlanarMixedContactFaces3D({&water, 1}, 11, {&snow, 1}, {&face, 1}, 0.03125);
  if (!missed.ok() || !missed->empty() || water.world_velocity.x != 2 || snow.transfer.particle_velocity.x != -2) {
    return false;
  }
  auto invalid = face;
  invalid.pair.face_slot = 6;
  invalid.first_interface_world_position.x = -0.25;
  const std::array late_overlap{face, invalid};
  const auto rejected = fluidity::ApplyPlanarMixedContactFaces3D({&water, 1}, 11, {&snow, 1}, late_overlap, 0.125);
  if (rejected.ok() || water.world_velocity.x != 2 || snow.transfer.particle_velocity.x != -2) {
    return false;
  }
  auto endpoint_water = water;
  auto endpoint_snow = snow;
  const auto endpoint =
      fluidity::ApplyPlanarMixedContactFaces3D({&endpoint_water, 1}, 11, {&endpoint_snow, 1}, {&face, 1}, 0.0625);
  if (!endpoint.ok() || endpoint->size() != 1 || endpoint->front().contact_fraction != 1) {
    return false;
  }
  const auto rejected_motion =
      fluidity::AdvancePlanarMixedContact3D({&water, 1}, 11, {&snow, 1}, {&face, 1}, 0.125, 0.10);
  if (rejected_motion.ok() || water.world_position.x != -0.0625 || water.world_velocity.x != 2
      || snow.transfer.trial_world_position.x != 0.3125 || snow.transfer.particle_velocity.x != -2) {
    return false;
  }
  // A frozen unit cell with one open outlet must discharge 2 m³/s.
  // At unit face area this fixes the outlet speed independently of the solver.
  constexpr std::array blocks{BlockIndex3D{0, 0, 0}};
  const auto grid = SparseMacGrid3D::Create({{-1, 0, 0}, 1}, blocks, blocks.size());
  if (!grid.ok()) {
    return false;
  }
  constexpr std::array cells{fluidity::MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1,
      .target_divergence = 2,
      .bulk_modulus = std::numeric_limits<double>::infinity(),
  }};
  using fluidity::FaceNormalAxis;
  const fluidity::FaceIndex3D outlet{FaceNormalAxis::kX, {1, 0, 0}};
  std::array pressure_faces{
      fluidity::MacPressureFace3D{.face_index = {FaceNormalAxis::kX, {0, 0, 0}}},
      fluidity::MacPressureFace3D{
          .face_index = outlet,
          .open_area = 1,
          .inverse_density = 1,
          .pressure_gradient_distance = 1,
          .pressure_link = fluidity::DirichletMacPressureLink3D{0},
      },
      fluidity::MacPressureFace3D{.face_index = {FaceNormalAxis::kY, {0, 0, 0}}},
      fluidity::MacPressureFace3D{.face_index = {FaceNormalAxis::kY, {0, 1, 0}}},
      fluidity::MacPressureFace3D{.face_index = {FaceNormalAxis::kZ, {0, 0, 0}}},
      fluidity::MacPressureFace3D{.face_index = {FaceNormalAxis::kZ, {0, 0, 1}}},
  };
  for (auto& pressure_face : pressure_faces) {
    if (pressure_face.open_area == 0) {
      pressure_face.pressure_link = fluidity::SealedMacPressureLink3D{};
    }
  }
  const auto outlet_slot = grid->FaceSlot(outlet);
  if (!outlet_slot.ok()) {
    return false;
  }
  std::sort(pressure_faces.begin(), pressure_faces.end(), [&](const auto& first, const auto& second) {
    return *grid->FaceSlot(first.face_index) < *grid->FaceSlot(second.face_index);
  });
  const auto outlet_face = std::find_if(pressure_faces.begin(), pressure_faces.end(), [](const auto& pressure_face) {
    return pressure_face.open_area > 0;
  });
  const auto outlet_ordinal = static_cast<std::size_t>(outlet_face - pressure_faces.begin());
  const fluidity::MacPressureInputs3D pressure_inputs{*grid, 0.125, cells, pressure_faces};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto projection = fluidity::CudaProjectMacPressure3D(pressure_inputs);
#else
  const auto projection = fluidity::ProjectMacPressure3D(pressure_inputs);
#endif
  if (!projection.ok()) {
    std::fprintf(stderr, "M5 pressure-contact fixture: %s\n", projection.status().ToString().c_str());
    return false;
  }
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto& pressure = projection->projection;
#else
  const auto& pressure = *projection;
#endif
  const double outlet_normal_velocity = pressure.corrected_face_velocities[outlet_ordinal];
  if (std::abs(outlet_normal_velocity - 2) > 1e-12 || std::abs(pressure.cell_pressures.front() - 16) > 1e-12
      || pressure.diagnostics.max_cell_constraint_defect > 1e-12) {
    return false;
  }
  const auto project =
      +[](const fluidity::MacPressureInputs3D& inputs) -> absl::StatusOr<fluidity::MacPressureProjection3D> {
#if defined(FLUIDITY_M5_CUDA_SCENES)
    auto projected = fluidity::CudaProjectMacPressure3D(inputs);
    if (!projected.ok()) {
      return projected.status();
    }
    return std::move(projected->projection);
#else
    return fluidity::ProjectMacPressure3D(inputs);
#endif
  };
  auto reciprocal_cells = cells;
  reciprocal_cells.front().target_divergence = 0;
  const auto low_wall = std::find_if(pressure_faces.begin(), pressure_faces.end(), [](const auto& pressure_face) {
    return pressure_face.face_index.axis == FaceNormalAxis::kX && pressure_face.face_index.lattice_index.x == 0;
  });
  if (low_wall == pressure_faces.end()) {
    return false;
  }
  const auto low_wall_ordinal = static_cast<std::size_t>(low_wall - pressure_faces.begin());
  const fluidity::MacPressureInputs3D reciprocal_inputs{*grid, 0.125, reciprocal_cells, pressure_faces};
  const auto light_snow =
      fluidity::SolvePlanarSnowPressureTrial3D(reciprocal_inputs, low_wall_ordinal, 0.25, 10, project);
  if (!light_snow.ok() || std::abs(light_snow->solved_wall_normal_velocity - 2) > 1e-12
      || std::abs(light_snow->liquid_projection.cell_pressures.front() - 16) > 1e-12
      || std::abs(light_snow->liquid_projection.corrected_face_velocities[outlet_ordinal] - 2) > 1e-12
      || light_snow->liquid_projection.diagnostics.max_cell_constraint_defect > 1e-12
      || std::abs(light_snow->pressure_impulse_on_snow + 2) > 1e-12
      || std::abs(light_snow->pressure_work_on_snow + 4) > 1e-12 || low_wall->wall_normal_velocity != 0) {
    return false;
  }
  auto reflected_faces = pressure_faces;
  auto& reflected_outlet = reflected_faces[low_wall_ordinal];
  auto& reflected_wall = reflected_faces[outlet_ordinal];
  reflected_outlet = *outlet_face;
  reflected_outlet.face_index = low_wall->face_index;
  reflected_wall = *low_wall;
  reflected_wall.face_index = outlet;
  const fluidity::MacPressureInputs3D reflected_inputs{*grid, 0.125, reciprocal_cells, reflected_faces};
  const auto reflected = fluidity::SolvePlanarSnowPressureTrial3D(reflected_inputs, outlet_ordinal, 2, -3, project);
  if (!reflected.ok() || std::abs(reflected->solved_wall_normal_velocity + 2) > 1e-12
      || std::abs(reflected->pressure_impulse_on_snow - 2) > 1e-12
      || std::abs(reflected->pressure_work_on_snow + 4) > 1e-12
      || std::abs(reflected->liquid_projection.corrected_face_velocities[low_wall_ordinal] + 2) > 1e-12
      || fluidity::SolvePlanarSnowPressureTrial3D(reciprocal_inputs, outlet_ordinal, 2, 3, project).status().code()
          != absl::StatusCode::kInvalidArgument) {
    return false;
  }
  std::array connected_cells{reciprocal_cells.front(), reciprocal_cells.front()};
  connected_cells.back().cell_index.x = 1;
  std::vector connected_faces(pressure_faces.begin(), pressure_faces.end());
  connected_faces[outlet_ordinal].pressure_link = fluidity::InternalMacPressureLink3D{0};
  auto far_outlet = *outlet_face;
  far_outlet.face_index.lattice_index.x = 2;
  connected_faces.push_back(far_outlet);
  for (const auto axis : {FaceNormalAxis::kY, FaceNormalAxis::kZ}) {
    for (const int side : {0, 1}) {
      connected_faces.push_back({
          .face_index = {axis, {1, axis == FaceNormalAxis::kY ? side : 0, axis == FaceNormalAxis::kZ ? side : 0}},
          .pressure_link = fluidity::SealedMacPressureLink3D{},
      });
    }
  }
  std::sort(connected_faces.begin(), connected_faces.end(), [&](const auto& first, const auto& second) {
    return *grid->FaceSlot(first.face_index) < *grid->FaceSlot(second.face_index);
  });
  const auto x_face_ordinal = [&](int x) {
    const auto face = std::find_if(connected_faces.begin(), connected_faces.end(), [x](const auto& candidate) {
      return candidate.face_index.axis == FaceNormalAxis::kX && candidate.face_index.lattice_index.x == x;
    });
    return static_cast<std::size_t>(face - connected_faces.begin());
  };
  const auto connected_low = x_face_ordinal(0);
  const auto connected_middle = x_face_ordinal(1);
  const auto connected_high = x_face_ordinal(2);
  const fluidity::MacPressureInputs3D connected_inputs{*grid, 0.125, connected_cells, connected_faces};
  const auto connected = fluidity::SolvePlanarSnowPressureTrial3D(connected_inputs, connected_low, 2, 4, project);
  constexpr double kConnectedTolerance = 2e-6;
  if (!connected.ok() || std::abs(connected->solved_wall_normal_velocity - 2) > kConnectedTolerance
      || std::abs(connected->liquid_projection.cell_pressures[0] - 32) > kConnectedTolerance
      || std::abs(connected->liquid_projection.cell_pressures[1] - 16) > kConnectedTolerance
      || std::abs(connected->liquid_projection.corrected_face_velocities[connected_middle] - 2) > kConnectedTolerance
      || std::abs(connected->liquid_projection.corrected_face_velocities[connected_high] - 2) > kConnectedTolerance
      || std::abs(connected->pressure_impulse_on_snow + 4) > kConnectedTolerance) {
    return false;
  }
  auto crossing_snow = snow;
  crossing_snow.transfer.trial_world_position = {-1.125, 0.25, 0.25};
  crossing_snow.transfer.particle_velocity = {4, 0, 0};
  std::array crossing_liquid{water, water};
  crossing_liquid[0].world_position = {-0.125, 0.25, 0.25};
  crossing_liquid[0].world_velocity = {};
  crossing_liquid[0].volume_ratio = 0.95;
  crossing_liquid[1].id = 8;
  crossing_liquid[1].world_position = {0.875, 0.25, 0.25};
  const std::array crossing_contributors{fluidity::SnowPressureContributor3D{3, 1}};
  const auto escaped = fluidity::AdvancePlanarSnowPressure3D(
      connected_inputs,
      connected_low,
      crossing_contributors,
      crossing_liquid,
      {&crossing_snow, 1},
      0.3,
      project
  );
  if (escaped.status().message() != "planar pressure liquid transport leaves adjacent pressure support"
      || crossing_liquid[0].world_position.x != -0.125 || crossing_liquid[0].world_velocity.x != 0
      || crossing_liquid[1].world_position.x != 0.875 || crossing_snow.transfer.trial_world_position.x != -1.125
      || crossing_snow.transfer.particle_velocity.x != 4) {
    return false;
  }
  const auto crossing = fluidity::AdvancePlanarSnowPressure3D(
      connected_inputs,
      connected_low,
      crossing_contributors,
      {crossing_liquid.data(), 1},
      {&crossing_snow, 1},
      0.3,
      project
  );
  if (!crossing.ok() || std::abs(crossing_liquid[0].world_position.x - 0.125) > kConnectedTolerance
      || std::abs(crossing_liquid[0].world_velocity.x - 2) > kConnectedTolerance
      || std::abs(crossing_liquid[0].volume_ratio - 0.95) > kConnectedTolerance
      || std::abs(crossing_snow.transfer.trial_world_position.x + 0.875) > kConnectedTolerance
      || std::abs(crossing_snow.transfer.particle_velocity.x - 2) > kConnectedTolerance) {
    return false;
  }
  connected_faces[connected_low] = far_outlet;
  connected_faces[connected_low].face_index = low_wall->face_index;
  connected_faces[connected_high] = *low_wall;
  connected_faces[connected_high].face_index = far_outlet.face_index;
  const auto reverse_connected =
      fluidity::SolvePlanarSnowPressureTrial3D(connected_inputs, connected_high, 2, -4, project);
  if (!reverse_connected.ok() || std::abs(reverse_connected->solved_wall_normal_velocity + 2) > kConnectedTolerance
      || std::abs(reverse_connected->liquid_projection.cell_pressures[0] - 16) > kConnectedTolerance
      || std::abs(reverse_connected->liquid_projection.cell_pressures[1] - 32) > kConnectedTolerance
      || std::abs(reverse_connected->pressure_impulse_on_snow - 4) > kConnectedTolerance) {
    return false;
  }
  connected_faces[connected_middle].open_area = 0;
  connected_faces[connected_middle].pressure_link = fluidity::SealedMacPressureLink3D{};
  if (fluidity::SolvePlanarSnowPressureTrial3D(connected_inputs, connected_middle, 2, 4, project).status().code()
      != absl::StatusCode::kInvalidArgument) {
    return false;
  }
  std::array pressure_snow{snow, snow, snow};
  pressure_snow[0].transfer.particle_mass = 0.125;
  pressure_snow[0].transfer.particle_velocity = {12, 2, 3};
  pressure_snow[1].particle_id = 4;
  pressure_snow[1].transfer.particle_mass = 0.0625;
  pressure_snow[1].transfer.particle_velocity = {16, 4, 5};
  pressure_snow[2].particle_id = 5;
  pressure_snow[2].body_id = 17;
  const std::array traction{
      fluidity::SnowPressureContributor3D{3, 0.5},
      fluidity::SnowPressureContributor3D{4, 0.5},
  };
  const auto rejected_traction =
      fluidity::ApplyPlanarSnowPressure3D(reciprocal_inputs, outlet_ordinal, traction, pressure_snow, project);
  if (rejected_traction.status().code() != absl::StatusCode::kInvalidArgument
      || pressure_snow[0].transfer.particle_velocity.x != 12 || pressure_snow[1].transfer.particle_velocity.x != 16) {
    return false;
  }
  auto invalid_traction = traction;
  invalid_traction[1].particle_id = 5;
  const auto wrong_body = fluidity::ApplyPlanarSnowPressure3D(
      reciprocal_inputs,
      low_wall_ordinal,
      invalid_traction,
      pressure_snow,
      project
  );
  if (wrong_body.status().code() != absl::StatusCode::kInvalidArgument
      || pressure_snow[0].transfer.particle_velocity.x != 12 || pressure_snow[2].transfer.particle_velocity.x != -2) {
    return false;
  }
  auto closed_faces = pressure_faces;
  auto& closed_outlet = closed_faces[outlet_ordinal];
  closed_outlet.open_area = 0;
  closed_outlet.pressure_link = fluidity::SealedMacPressureLink3D{};
  const fluidity::MacPressureInputs3D closed_inputs{*grid, 0.125, reciprocal_cells, closed_faces};
  const auto failed_probe =
      fluidity::ApplyPlanarSnowPressure3D(closed_inputs, low_wall_ordinal, traction, pressure_snow, project);
  if (failed_probe.status().code() != absl::StatusCode::kFailedPrecondition
      || pressure_snow[0].transfer.particle_velocity.x != 12 || pressure_snow[1].transfer.particle_velocity.x != 16) {
    return false;
  }
  const auto applied_traction =
      fluidity::ApplyPlanarSnowPressure3D(reciprocal_inputs, low_wall_ordinal, traction, pressure_snow, project);
  const double first_velocity = pressure_snow[0].transfer.particle_velocity.x;
  const double second_velocity = pressure_snow[1].transfer.particle_velocity.x;
  const double transferred_momentum = 0.125 * (first_velocity - 12) + 0.0625 * (second_velocity - 16);
  if (!applied_traction.ok() || std::abs(first_velocity - 4) > 1e-12 || std::abs(second_velocity) > 1e-12
      || std::abs(transferred_momentum - applied_traction->pressure_impulse_on_snow) > 1e-12
      || std::abs(applied_traction->solved_wall_normal_velocity - 2) > 1e-12
      || std::abs(applied_traction->pressure_work_on_snow + 4) > 1e-12
      || pressure_snow[0].transfer.particle_velocity.y != 2 || pressure_snow[1].transfer.particle_velocity.z != 5
      || pressure_snow[2].transfer.particle_velocity.x != -2
      || pressure_snow[1].transfer.trial_world_position.x != snow.transfer.trial_world_position.x
      || pressure_snow[1].log_plastic_jacobian != snow.log_plastic_jacobian) {
    return false;
  }
  auto advected_water = water;
  advected_water.world_position = {-0.75, 0.25, 0.25};
  advected_water.world_velocity = {};
  advected_water.volume_ratio = 0.95;
  auto advected_snow = snow;
  advected_snow.transfer.trial_world_position = {-1.125, 0.25, 0.25};
  advected_snow.transfer.particle_velocity = {3, 3, 0};
  const std::array single_traction{fluidity::SnowPressureContributor3D{3, 1}};
  const auto rejected_advection = fluidity::AdvancePlanarSnowPressure3D(
      reciprocal_inputs,
      low_wall_ordinal,
      single_traction,
      {&advected_water, 1},
      {&advected_snow, 1},
      0.3,
      project
  );
  if (rejected_advection.status().message() != "planar mixed contact path exceeds motion bound"
      || advected_water.world_position.x != -0.75 || advected_water.world_velocity.x != 0
      || advected_water.volume_ratio != 0.95 || advected_snow.transfer.trial_world_position.x != -1.125
      || advected_snow.transfer.particle_velocity.x != 3) {
    return false;
  }
  advected_snow.transfer.particle_velocity.y = 0;
  const auto accepted_advection = fluidity::AdvancePlanarSnowPressure3D(
      reciprocal_inputs,
      low_wall_ordinal,
      single_traction,
      {&advected_water, 1},
      {&advected_snow, 1},
      0.3,
      project
  );
  if (!accepted_advection.ok() || std::abs(advected_water.world_position.x + 0.5) > 1e-12
      || std::abs(advected_snow.transfer.trial_world_position.x + 0.875) > 1e-12
      || std::abs(advected_water.world_velocity.x - 2) > 1e-12
      || std::abs(advected_snow.transfer.particle_velocity.x - 2) > 1e-12
      || std::abs(advected_water.volume_ratio - 0.95) > 1e-12 || advected_snow.transfer.sorted_world_position.x != 0) {
    return false;
  }
  auto moving_snow = snow;
  moving_snow.transfer.particle_mass = 0.375;
  moving_snow.transfer.trial_world_position = {-0.875, 0.25, 0.25};
  moving_snow.transfer.particle_velocity = {4, 0, 0};
  auto moving_liquid = water;
  moving_liquid.world_position = {-0.5, 0.25, 0.25};
  moving_liquid.volume_ratio = 0.95;
  const auto freeze = [&] {
    return fluidity::FreezePlanarSnowPressureCell3D(
        *grid,
        {0, 0, 0},
        FaceNormalAxis::kX,
        true,
        {&moving_snow, 1},
        0.125,
        1,
        0
    );
  };
  const auto first_frozen = freeze();
  if (!first_frozen.ok()) {
    std::fprintf(stderr, "M5 moving snow first_frozen: %s\n", first_frozen.status().ToString().c_str());
    return false;
  }
  if (first_frozen->wall_plane_coordinate != -0.75 || first_frozen->cell.fluid_volume != 0.75) {
    return false;
  }
  const fluidity::MacPressureInputs3D first_cut_inputs{
      *grid,
      0.125,
      {&first_frozen->cell, 1},
      first_frozen->faces,
  };
  const auto first_cut = fluidity::AdvancePlanarSnowPressure3D(
      first_cut_inputs,
      first_frozen->wall_face_ordinal,
      single_traction,
      {&moving_liquid, 1},
      {&moving_snow, 1},
      0.3,
      project,
      first_frozen->wall_plane_coordinate
  );
  if (!first_cut.ok()) {
    std::fprintf(stderr, "M5 moving snow first_cut: %s\n", first_cut.status().ToString().c_str());
    return false;
  }
  if (std::abs(first_cut->solved_wall_normal_velocity - 2) > 1e-12
      || std::abs(first_cut->liquid_projection.cell_pressures.front() - 6) > 1e-12
      || std::abs(moving_snow.transfer.trial_world_position.x + 0.625) > 1e-12
      || std::abs(moving_liquid.world_position.x + 0.25) > 1e-12) {
    return false;
  }
  const auto second_frozen = freeze();
  if (!second_frozen.ok()) {
    std::fprintf(stderr, "M5 moving snow second_frozen: %s\n", second_frozen.status().ToString().c_str());
    return false;
  }
  if (std::abs(second_frozen->wall_plane_coordinate + 0.5) > 1e-12
      || std::abs(second_frozen->cell.fluid_volume - 0.5) > 1e-12) {
    return false;
  }
  const fluidity::MacPressureInputs3D second_cut_inputs{
      *grid,
      0.125,
      {&second_frozen->cell, 1},
      second_frozen->faces,
  };
  const auto inconsistent = fluidity::AdvancePlanarSnowPressure3D(
      second_cut_inputs,
      second_frozen->wall_face_ordinal,
      single_traction,
      {&moving_liquid, 1},
      {&moving_snow, 1},
      0.3,
      project,
      -0.625
  );
  if (inconsistent.status().code() != absl::StatusCode::kInvalidArgument
      || std::abs(moving_liquid.world_position.x + 0.25) > 1e-12
      || std::abs(moving_snow.transfer.trial_world_position.x + 0.625) > 1e-12) {
    return false;
  }
  const auto second_cut = fluidity::AdvancePlanarSnowPressure3D(
      second_cut_inputs,
      second_frozen->wall_face_ordinal,
      single_traction,
      {&moving_liquid, 1},
      {&moving_snow, 1},
      0.3,
      project,
      second_frozen->wall_plane_coordinate
  );
  if (!second_cut.ok()) {
    std::fprintf(stderr, "M5 moving snow second_cut: %s\n", second_cut.status().ToString().c_str());
    return false;
  }
  constexpr double kMovingWallTolerance = 2e-7;
  if (std::abs(second_cut->solved_wall_normal_velocity - 1.2) > kMovingWallTolerance
      || std::abs(second_cut->liquid_projection.cell_pressures.front() - 2.4) > kMovingWallTolerance
      || std::abs(moving_snow.transfer.trial_world_position.x + 0.475) > kMovingWallTolerance
      || std::abs(moving_liquid.world_position.x + 0.1) > kMovingWallTolerance
      || std::abs(moving_liquid.volume_ratio - 0.95) > kMovingWallTolerance) {
    return false;
  }
  fluidity::LiquidState3D accepted_liquid;
  accepted_liquid.geometry = grid->geometry();
  accepted_liquid.next_particle_id = 8;
  const auto prepare = []<typename Record>(fluidity::LiquidRecords3D<Record>& records, std::size_t count) {
    auto allocated = fluidity::LiquidRecords3D<Record>::Create(count);
    if (!allocated.ok() || !allocated->Resize(count).ok()) {
      return false;
    }
    records = std::move(*allocated);
    return true;
  };
  if (!prepare(accepted_liquid.particles, 1) || !prepare(accepted_liquid.blocks, 1)
      || !prepare(accepted_liquid.cells, 1) || !prepare(accepted_liquid.faces, first_frozen->faces.size())) {
    return false;
  }
  accepted_liquid.particles[0] = water;
  accepted_liquid.particles[0].world_position = {-0.5, 0.25, 0.25};
  accepted_liquid.particles[0].world_velocity = {};
  accepted_liquid.particles[0].volume_ratio = 0.95;
  accepted_liquid.cells[0].cell_index = first_frozen->cell.cell_index;
  for (std::size_t ordinal = 0; ordinal < first_frozen->faces.size(); ++ordinal) {
    const auto index = first_frozen->faces[ordinal].face_index;
    const auto slot = grid->FaceSlot(index);
    if (!slot.ok()) {
      return false;
    }
    accepted_liquid.faces[ordinal].face_index = index;
    accepted_liquid.faces[ordinal].face_slot = *slot;
  }
  auto transaction = fluidity::LiquidFrameTransaction3D::Create(accepted_liquid, {1, 1, 1, 6});
  if (!transaction.ok() || !transaction->BeginFrame(0.125).ok()) {
    return false;
  }
  auto committed_snow = moving_snow;
  committed_snow.transfer.trial_world_position.x = -0.875;
  committed_snow.transfer.particle_velocity = {4, 0, 0};
  const auto rejected_attempt = fluidity::AttemptPlanarSnowPressureStep3D(
      *transaction,
      first_cut_inputs,
      first_frozen->wall_face_ordinal,
      single_traction,
      {&committed_snow, 1},
      0.20,
      project,
      first_frozen->wall_plane_coordinate
  );
  if (!rejected_attempt.ok() || rejected_attempt->outcome.disposition != fluidity::LiquidAttemptDisposition3D::kRetryDt
      || rejected_attempt->outcome.retry_duration != 0.0625 || rejected_attempt->pressure.has_value()
      || transaction->committed() != accepted_liquid || committed_snow.transfer.trial_world_position.x != -0.875
      || committed_snow.transfer.particle_velocity.x != 4) {
    return false;
  }
  auto half_cut_inputs = first_cut_inputs;
  half_cut_inputs.step_duration = *rejected_attempt->outcome.retry_duration;
  const auto accepted_attempt = fluidity::AttemptPlanarSnowPressureStep3D(
      *transaction,
      half_cut_inputs,
      first_frozen->wall_face_ordinal,
      single_traction,
      {&committed_snow, 1},
      0.20,
      project,
      first_frozen->wall_plane_coordinate
  );
  const auto& committed = transaction->committed();
  if (!accepted_attempt.ok() || accepted_attempt->outcome.disposition != fluidity::LiquidAttemptDisposition3D::kAccepted
      || !accepted_attempt->pressure.has_value() || committed.clock.time != 0.0625
      || committed.clock.accepted_step_ordinal != 1 || committed.clock.epoch != 1
      || std::abs(committed.cells[0].previous_pressure - 12) > 1e-12
      || std::abs(committed.faces[first_frozen->wall_face_ordinal].carrier_velocity - 2) > 1e-12
      || std::abs(committed.particles[0].world_position.x + 0.375) > 1e-12
      || std::abs(committed_snow.transfer.trial_world_position.x + 0.75) > 1e-12
      || std::abs(committed_snow.transfer.particle_velocity.x - 2) > 1e-12) {
    return false;
  }
  auto frame_snow = committed_snow;
  frame_snow.transfer.trial_world_position.x = -0.875;
  frame_snow.transfer.particle_velocity.x = 4;
  fluidity::PlanarSnowPressureFrameSettings3D frame_settings{
      .wall_cell = {0, 0, 0},
      .wall_axis = FaceNormalAxis::kX,
      .wall_on_low_face = true,
      .surface_offset = 0.125,
      .inverse_liquid_density = 1,
      .outlet_pressure = 0,
      .target_time = 0.125,
      .nominal_step_duration = 0.125,
      .maximum_path_length = 0.20,
  };
  const auto frame = fluidity::AdvancePlanarSnowPressureFrame3D(
      *grid,
      accepted_liquid,
      {&frame_snow, 1},
      single_traction,
      frame_settings,
      project
  );
  if (!frame.ok()) {
    std::fprintf(stderr, "M5 moving snow frame: %s\n", frame.status().ToString().c_str());
    return false;
  }
  const auto& frame_clock = frame->liquid.clock;
  constexpr double kFrameTolerance = 2e-6;
  if (frame->rejected_attempts != 1 || frame_clock.time != 0.125 || frame_clock.frame_ordinal != 1
      || frame_clock.substep_ordinal != 0 || frame_clock.accepted_step_ordinal != 2 || frame_clock.epoch != 2
      || std::abs(frame->liquid.cells[0].previous_pressure - 60.0 / 11.0) > kFrameTolerance
      || std::abs(frame->liquid.particles[0].world_position.x - (-0.375 + 0.75 / 11.0)) > kFrameTolerance
      || std::abs(frame->snow[0].transfer.trial_world_position.x - (-0.75 + 0.75 / 11.0)) > kFrameTolerance
      || std::abs(frame->snow[0].transfer.particle_velocity.x - 12.0 / 11.0) > kFrameTolerance) {
    return false;
  }
  frame_settings.maximum_path_length = 0.01;
  const auto exhausted = fluidity::AdvancePlanarSnowPressureFrame3D(
      *grid,
      accepted_liquid,
      {&frame_snow, 1},
      single_traction,
      frame_settings,
      project
  );
  if (exhausted.status().code() != absl::StatusCode::kFailedPrecondition || accepted_liquid.clock.time != 0
      || accepted_liquid.particles[0].world_position.x != -0.5 || frame_snow.transfer.trial_world_position.x != -0.875
      || frame_snow.transfer.particle_velocity.x != 4) {
    return false;
  }
  // The fixture transfers the outlet speed to one liquid particle with unit weight.
  // Stage that transfer with the mixed advance and publish the two particles together.
  water.world_velocity.x = 0;
  const auto advance_pressure_trial = [&](double maximum_path_length
                                      ) -> absl::StatusOr<std::vector<fluidity::PlanarMixedContactReceipt3D>> {
    auto trial_water = water;
    auto trial_snow = snow;
    trial_water.world_velocity.x = outlet_normal_velocity;
    const std::array pressure_contributors{
        contributors[0],
        fluidity::ContactFaceParticle3D{7, 11, 6, 1, trial_water.world_velocity},
    };
    const fluidity::DetectedContactFace3D pressure_pair{*outlet_slot, pressure_contributors, 9, 11, {1, 0, 0}, 0};
    const auto pressure_contact = fluidity::ReconstructAxisAlignedMixedContactFace3D(
        {&trial_water, 1},
        11,
        {&trial_snow, 1},
        pressure_pair,
        0.0625,
        0.0625
    );
    if (!pressure_contact.ok()) {
      return pressure_contact.status();
    }
    auto trial = fluidity::AdvancePlanarMixedContact3D(
        {&trial_water, 1},
        11,
        {&trial_snow, 1},
        {&*pressure_contact, 1},
        pressure_inputs.step_duration,
        maximum_path_length
    );
    if (trial.ok()) {
      water = trial_water;
      snow = trial_snow;
    }
    return trial;
  };
  const auto rejected_pressure_trial = advance_pressure_trial(0.10);
  if (rejected_pressure_trial.status().message() != "planar mixed contact path exceeds motion bound"
      || water.world_position.x != -0.0625 || water.world_velocity.x != 0
      || snow.transfer.trial_world_position.x != 0.3125 || snow.transfer.particle_velocity.x != -2) {
    return false;
  }
  const auto hit = advance_pressure_trial(0.25);
  return hit.ok() && hit->size() == 1 && hit->front().face_slot == *outlet_slot
      && std::abs(hit->front().contact_fraction - 0.5) < 1e-12
      && std::abs(hit->front().contact.impulse_on_first.x - 6) < 1e-12 && std::abs(water.world_velocity.x - 1) < 1e-12
      && std::abs(snow.transfer.particle_velocity.x - 1) < 1e-12 && std::abs(water.world_position.x - 0.125) < 1e-12
      && std::abs(snow.transfer.trial_world_position.x - 0.25) < 1e-12 && snow.transfer.sorted_world_position.x == 0;
}

bool TestMixedContactAcceptance() {
  fluidity::LiquidParticleState3D water{
      .id = 7,
      .world_position = {2, 3, 4},
      .world_velocity = {2, 0, 0},
      .mass = 6,
      .volume_ratio = 0.95,
  };
  fluidity::PlasticSolidCheckpointParticle3D snow{
      .particle_id = 3,
      .body_id = 9,
      .material_id = 1,
      .transfer =
          {
              .trial_world_position = {1, 2, 3},
              .particle_mass = 2,
              .particle_velocity = {-2, 3, 0},
          },
      .reference_volume = 0.001,
      .elastic_deformation = kIdentity,
      .log_plastic_jacobian = std::log(0.8),
  };
  constexpr std::array contributors{
      fluidity::ContactFaceParticle3D{3, 9, 2, 0.5, {-2, 3, 0}},
      fluidity::ContactFaceParticle3D{7, 11, 6, 0.5, {2, 0, 0}},
  };
  std::array faces{
      fluidity::DetectedContactFace3D{3, contributors, 9, 11, {1, 0, 0}, 0.25},
      fluidity::DetectedContactFace3D{6, contributors, 9, 99, {1, 0, 0}, 0.25},
  };
  water.world_velocity.z = -0.0;
  if (!fluidity::ApplyMixedContactFaces3D({&water, 1}, 11, {&snow, 1}, {}).ok()
      || !std::signbit(water.world_velocity.z)) {
    return false;
  }
  auto extreme_water = water;
  extreme_water.world_velocity.x = -1.7e308;
  auto extreme_snow = snow;
  auto extreme_contributors = contributors;
  extreme_contributors[0].world_face_velocity = {-4e307, 0, 0};
  extreme_contributors[1].world_face_velocity = {4e307, 0, 0};
  const fluidity::DetectedContactFace3D extreme_face{3, extreme_contributors, 9, 11, {1, 0, 0}, 0};
  const auto overflow =
      fluidity::ApplyMixedContactFaces3D({&extreme_water, 1}, 11, {&extreme_snow, 1}, {&extreme_face, 1});
  if (overflow.status().code() != absl::StatusCode::kOutOfRange
      || overflow.status().message() != "unrepresentable stored mixed-contact velocity"
      || extreme_water.world_velocity.x != -1.7e308
      || extreme_snow.transfer.particle_velocity.x != snow.transfer.particle_velocity.x) {
    return false;
  }
  const auto old_water = water;
  const auto old_snow = snow;
  if (fluidity::ApplyMixedContactFaces3D({&water, 1}, 11, {&snow, 1}, faces).ok() || !(water == old_water)
      || snow.transfer.particle_velocity.x != old_snow.transfer.particle_velocity.x
      || snow.transfer.particle_velocity.y != old_snow.transfer.particle_velocity.y) {
    return false;
  }
  faces[1].second_body_id = 11;
  const auto accepted = fluidity::ApplyMixedContactFaces3D({&water, 1}, 11, {&snow, 1}, faces);
  return accepted.ok() && accepted->size() == 2 && std::abs(water.world_velocity.x - 1) < 1e-12
      && std::abs(snow.transfer.particle_velocity.x - 1) < 1e-12
      && std::abs(6 * water.world_velocity.y + 2 * snow.transfer.particle_velocity.y - 6) < 1e-12
      && water.world_position.x == old_water.world_position.x && water.volume_ratio == old_water.volume_ratio
      && snow.log_plastic_jacobian == old_snow.log_plastic_jacobian
      && snow.transfer.trial_world_position.x == old_snow.transfer.trial_world_position.x;
}

}  // namespace

int main() {
  if (!TestPlanarMixedContactDetection()) {
    std::fprintf(stderr, "M5 planar pressure and swept contact fixture failed\n");
    return 1;
  }
  if (!TestMixedContactAcceptance()) {
    std::fprintf(stderr, "M5 mixed contact rollback fixture failed\n");
    return 1;
  }
  return 0;
}
