#include "fluidity/host/liquid_substep_3d.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numbers>
#include <optional>
#include <tuple>

#include "absl/cleanup/cleanup.h"
#include "liquid_boundary_backend.h"
#include "liquid_substep_3d_internal.h"
namespace fluidity {
namespace {
#define STEP_TRY(expression) \
  do { \
    const auto status = (expression); \
    if (!status.ok()) { \
      return status; \
    } \
  } while (false)
absl::Status Error(absl::StatusCode code) {
  return absl::Status(code, {});
}
// Check the complete owned payload before publishing any reservation or growth.
absl::StatusOr<std::size_t> ReservationBytes(
    LiquidStateCapacities3D capacities,
    std::size_t source_count,
    std::size_t constituent_material_count,
    std::size_t phase_pair_count,
    std::size_t workspace_bytes,
    std::size_t staging_bytes,
    bool cavity_enabled
) {
  std::size_t total = 0;
  const auto add = [&](std::size_t count, std::size_t stride) -> absl::Status {
    if (count > (static_cast<std::size_t>(PTRDIFF_MAX) - total) / stride) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    total += count * stride;
    return absl::OkStatus();
  };
  STEP_TRY(add(capacities.particles, 3 * sizeof(LiquidParticleState3D)));
  STEP_TRY(add(capacities.blocks, 3 * sizeof(LiquidBlockState3D)));
  STEP_TRY(add(capacities.cells, 3 * sizeof(LiquidCellState3D)));
  STEP_TRY(add(capacities.faces, 3 * sizeof(LiquidFaceState3D)));
  STEP_TRY(add(capacities.sources, 3 * sizeof(LiquidSourceState3D)));
  STEP_TRY(add(source_count, sizeof(LiquidEmissionSource3D)));
  STEP_TRY(add(constituent_material_count, sizeof(LiquidConstituentMaterial3D)));
  STEP_TRY(add(phase_pair_count, sizeof(LiquidPhasePair3D)));
  STEP_TRY(add(workspace_bytes, 1));
  STEP_TRY(add(staging_bytes, 1));
  if (cavity_enabled) {
    STEP_TRY(add(capacities.cells, 3 * (sizeof(LiquidCavityGasState3D) + sizeof(LiquidCavityComponentState3D))));
  }
  return total;
}
bool Complete(const LiquidSubstepBackend3D& backend) {
  return backend.sample_field && backend.sample_faces && backend.initialize && backend.prepare_publication
      && backend.commit_publication && backend.discard_publication && backend.prepare_growth && backend.commit_growth
      && backend.discard_growth && backend.topology && backend.deposit && backend.normalize
      && backend.predict_missing_face && backend.forces && backend.freeze && backend.links && backend.project
      && backend.gather && backend.update_particles;
}
auto CellKey(CellIndex3D index) {
  return std::tuple(index.z, index.y, index.x);
}

// Select the projected new history only for particles in a frozen pressure cell.
// An unresolved containing cell has to be reported in scalar particle order.
std::optional<bool> UsesProjectedGatherHistory(const FrozenLiquidBoundaryGeometryView3D& frozen, Vec3 position) {
  const auto cell = liquid_step_internal::TankContainingCell(frozen.mac_grid.geometry(), position);
  if (!cell) {
    return std::nullopt;
  }
  const auto cell_key = CellKey(*cell);
  const auto found = std::ranges::lower_bound(frozen.cells, cell_key, {}, [](const auto& pressure_cell) {
    return CellKey(pressure_cell.cell_index);
  });
  return found != frozen.cells.end() && CellKey(found->cell_index) == cell_key;
}
auto FaceKey(FaceIndex3D index) {
  return std::tuple(index.axis, index.lattice_index.z, index.lattice_index.y, index.lattice_index.x);
}

using liquid_step_internal::BalanceSealedLiquidDrift3D;
using liquid_step_internal::LiquidVolumeDriftRate3D;
using liquid_step_internal::Staging;

// Check physical compatibility before adding any sensor correction.
absl::Status ApplyIncompressibleDrift(
    SparseMacGridView3D grid,
    double step_duration,
    LiquidSceneDrift3D controls,
    Staging& staging,
    MacPressureWorkspace3D& workspace,
    LiquidPressureProblemView3D& problem
) {
  const auto components = workspace.ClassifyPhysicalComponents(grid, step_duration, problem);
  if (!components.ok()) {
    return components.status();
  }
  auto& corrected = *staging.drift_pressure_cells;
  auto& sealed = *staging.sealed_drift_cells;
  STEP_TRY(corrected.Resize(problem.cells.size()));
  std::ranges::copy(problem.cells, corrected.values().begin());
  const auto add_correction = [&](std::size_t cell, double correction) {
    if (correction != 0.0) {
      corrected[cell].target_divergence += correction;
    }
    return std::isfinite(corrected[cell].target_divergence) ? absl::OkStatus() : Error(absl::StatusCode::kOutOfRange);
  };
  for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
    if (std::ranges::binary_search(components->sealed_roots, components->roots[cell])) {
      continue;
    }
    const auto correction =
        LiquidVolumeDriftRate3D((*staging.drift_sensors)[cell], 0.0, controls.correction_time, controls.maximum_rate);
    if (!correction.ok()) {
      return correction.status();
    }
    STEP_TRY(add_correction(cell, *correction));
  }
  STEP_TRY(sealed.Resize(problem.cells.size()));
  for (const std::size_t root : components->sealed_roots) {
    std::size_t sensed_count = 0;
    for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
      const auto& sensor = (*staging.drift_sensors)[cell];
      if (components->roots[cell] != root || sensor.sampled_reference_volume == 0.0) {
        continue;
      }
      const double raw_rate = -std::log(sensor.mean_volume_ratio) / controls.correction_time;
      if (!std::isfinite(raw_rate)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      sealed[sensed_count++] = {
          .cell_ordinal = cell,
          .fluid_volume = problem.cells[cell].fluid_volume,
          .raw_rate = raw_rate,
      };
    }
    const auto sensed_cells = sealed.values().first(sensed_count);
    STEP_TRY(BalanceSealedLiquidDrift3D(sensed_cells, controls.maximum_rate));
    for (const auto& cell : sensed_cells) {
      STEP_TRY(add_correction(cell.cell_ordinal, cell.corrected_rate));
    }
  }
  problem.cells = corrected.values();
  return absl::OkStatus();
}

bool HasConstituentViscosity(const LiquidSceneInputs3D& scene) {
  return std::ranges::any_of(scene.constituent_materials, [](const auto& material) {
    return material.dynamic_viscosity_pa_s.has_value();
  });
}

bool CanApplyViscosity(const LiquidSceneInputs3D& scene) {
  return scene.dynamic_viscosity > 0.0 || std::ranges::any_of(scene.constituent_materials, [](const auto& material) {
           return material.dynamic_viscosity_pa_s.value_or(0.0) > 0.0;
         });
}
bool CanStageRegularCellTransport(const LiquidSceneInputs3D& scene, const AxisAlignedLiquidTank3D* tank) {
  return tank && tank->piston_velocity_x == 0 && std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field)
      && scene.sources.empty() && scene.target_divergence == 0 && std::isinf(scene.bulk_modulus) && !scene.eos
      && (!scene.drift || scene.drift->maximum_rate == 0.0) && scene.surface_tension == 0 && !CanApplyViscosity(scene);
}

// Continue projected changes only when the new donors are a strict subset of
// the old donors and both histories describe the same open, untagged surface.
bool CanContinueIsolatedDropProjection(
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& projected_history
) {
  const auto usable = [](LiquidFaceValidity3D validity) {
    return validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted;
  };
  bool removed_donor = false;
  for (std::size_t slot = 0; slot < projected_history.validity.size(); ++slot) {
    const auto& old_tag = old_history.face_tags[slot];
    const auto& projected_tag = projected_history.face_tags[slot];
    if (old_tag.side_tag != projected_tag.side_tag
        || old_tag.structurally_eligible != projected_tag.structurally_eligible
        || old_history.wall_samples[slot].available || projected_history.wall_samples[slot].available) {
      return false;
    }
    const bool old_donor = usable(old_history.validity[slot]);
    const bool projected_donor = usable(projected_history.validity[slot]);
    if (projected_donor && !old_donor) {
      return false;
    }
    removed_donor = removed_donor || (old_donor && !projected_donor);
  }
  return removed_donor;
}

Vec3 ContinueProjectedChange(Vec3 full_old, Vec3 projected, Vec3 restricted_old) {
  return {
      full_old.x + (projected.x - restricted_old.x),
      full_old.y + (projected.y - restricted_old.y),
      full_old.z + (projected.z - restricted_old.z),
  };
}

Mat3 ContinueProjectedChange(Mat3 full_old, Mat3 projected, Mat3 restricted_old) {
  return {
      ContinueProjectedChange(full_old.x_row, projected.x_row, restricted_old.x_row),
      ContinueProjectedChange(full_old.y_row, projected.y_row, restricted_old.y_row),
      ContinueProjectedChange(full_old.z_row, projected.z_row, restricted_old.z_row),
  };
}

// The restricted and full old gathers are copied before reusing workspace.
// The projected result remains borrowed until the next gather.
absl::StatusOr<LiquidBoundaryGatherView3D> GatherIsolatedDropHistories(
    const LiquidSubstepBackend3D& backend,
    LiquidSubstepBackend3D::Gather& gather,
    const LiquidBoundaryGatherViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace,
    std::span<LiquidBoundaryParticleGather3D> restricted_old,
    std::span<LiquidBoundaryParticleGather3D> full_old
) {
  auto old_inputs = inputs;
  old_inputs.new_history.face_velocities = inputs.old_history.face_velocities;
  const auto restricted = gather(backend.context, old_inputs, workspace);
  if (!restricted.ok()) {
    return restricted.status();
  }
  if (restricted->particles.size() != inputs.particles.size()) {
    return Error(absl::StatusCode::kInternal);
  }
  std::ranges::copy(restricted->particles, restricted_old.begin());
  old_inputs.new_history.validity = inputs.old_history.validity;
  const auto full = gather(backend.context, old_inputs, workspace);
  if (!full.ok()) {
    return full.status();
  }
  if (full->particles.size() != inputs.particles.size()) {
    return Error(absl::StatusCode::kInternal);
  }
  std::ranges::copy(full->particles, full_old.begin());
  auto projected = gather(backend.context, inputs, workspace);
  if (projected.ok() && projected->particles.size() != inputs.particles.size()) {
    return Error(absl::StatusCode::kInternal);
  }
  return projected;
}

// Publish the corrected record only after checking all continued components.
absl::Status CorrectIsolatedDropProjectedChange(
    const LiquidBoundaryParticleGather3D& full_old,
    const LiquidBoundaryParticleGather3D& restricted_old,
    LiquidBoundaryParticleGather3D result,
    LiquidBoundaryParticleGather3D& corrected
) {
  result.grid_velocity_increment =
      ContinueProjectedChange(Vec3{}, result.new_pic_velocity, restricted_old.new_pic_velocity);
  result.new_pic_velocity =
      ContinueProjectedChange(full_old.new_pic_velocity, result.new_pic_velocity, restricted_old.new_pic_velocity);
  result.fitted_velocity_slopes = ContinueProjectedChange(
      full_old.fitted_velocity_slopes,
      result.fitted_velocity_slopes,
      restricted_old.fitted_velocity_slopes
  );
  result.velocity_gradient =
      ContinueProjectedChange(full_old.velocity_gradient, result.velocity_gradient, restricted_old.velocity_gradient);
  using liquid_step_internal::Finite;
  if (!Finite(result.grid_velocity_increment) || !Finite(result.new_pic_velocity)) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  for (const auto row : {&Mat3::x_row, &Mat3::y_row, &Mat3::z_row}) {
    if (!Finite(result.fitted_velocity_slopes.*row) || !Finite(result.velocity_gradient.*row)) {
      return Error(absl::StatusCode::kOutOfRange);
    }
  }
  corrected = result;
  return absl::OkStatus();
}

absl::StatusOr<LiquidBoundaryGatherView3D> GatherIsolatedDropProjectedChange(
    const LiquidSubstepBackend3D& backend,
    const LiquidBoundaryGatherViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace,
    LiquidBoundaryParticleGather3D& corrected
) {
  LiquidBoundaryParticleGather3D restricted_old;
  LiquidBoundaryParticleGather3D full_old;
  auto projected = GatherIsolatedDropHistories(
      backend,
      *backend.gather,
      inputs,
      workspace,
      std::span(&restricted_old, 1),
      std::span(&full_old, 1)
  );
  if (!projected.ok()) {
    return projected.status();
  }
  STEP_TRY(CorrectIsolatedDropProjectedChange(full_old, restricted_old, projected->particles[0], corrected));
  projected->particles = std::span(&corrected, 1);
  return projected;
}

// The frozen faces and pressure links share the owned-slot order. A sealed
// face supplies a boundary value independently of its dormant carrier sample.
std::optional<double> RegularTransportFaceVelocity(
    const FrozenLiquidBoundaryGeometryView3D& frozen,
    std::span<const MacPressureFace3D> links,
    const liquid_step_internal::Staging& staging,
    FaceIndex3D face_index,
    std::int64_t particle_side_tag,
    const boundary_internal::TankLatticeAxis& normal_lattice,
    double wall_coordinate
) {
  const auto slot = frozen.mac_grid.FaceSlot(face_index);
  if (!slot.ok()) {
    return std::nullopt;
  }
  const auto face = std::ranges::lower_bound(frozen.faces, *slot, {}, &FrozenLiquidFace3D::face_slot);
  if (face == frozen.faces.end() || face->face_slot != *slot) {
    return std::nullopt;
  }
  const auto ordinal = static_cast<std::size_t>(face - frozen.faces.begin());
  if (ordinal >= links.size()) {
    return std::nullopt;
  }
  const auto& link = links[ordinal];
  const bool matching_link = FaceKey(link.face_index) == FaceKey(face_index) && link.open_area == face->open_area;
  if (face->prescribed_wall_flux.value_or(0) != 0) {
    return std::nullopt;
  }
  if (!matching_link || link.prescribed_wall_flux.value_or(0) != 0) {
    return std::nullopt;
  }
  const double full_face_area = normal_lattice.cell_width * normal_lattice.cell_width;
  if (face->open_area == full_face_area && face->blocked_area == 0) {
    const auto& tag = staging.new_tags[*slot];
    const auto validity = staging.validity[*slot];
    const bool usable = validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted;
    const double velocity = staging.new_velocities[*slot];
    if (tag.structurally_eligible && tag.side_tag == particle_side_tag && usable && std::isfinite(velocity)) {
      return velocity;
    }
    return std::nullopt;
  }
  const bool fully_sealed = face->open_area == 0 && face->blocked_area == full_face_area;
  const bool stationary = face->wall_normal_velocity == 0 && link.wall_normal_velocity == 0;
  if (!fully_sealed || !stationary || !std::isfinite(wall_coordinate)) {
    return std::nullopt;
  }
  const auto wall_offset_sign = boundary_internal::TankPlaneOffsetSign(wall_coordinate, normal_lattice);
  if (wall_offset_sign.ok() && *wall_offset_sign == 0) {
    return 0;
  }
  return std::nullopt;
}

// Select only a frozen pressure cell with six usable normal samples.
// Use the same checked lattice arithmetic for the cell and local fractions.
LiquidRegularCellTransport3D RegularTransportSample(
    const FrozenLiquidBoundaryGeometryView3D& frozen,
    const LiquidPressureProblemView3D& links,
    const liquid_step_internal::Staging& staging,
    const AxisAlignedLiquidTank3D& tank,
    const LiquidParticleState3D& particle
) {
  const auto geometry = frozen.mac_grid.geometry();
  const auto cell = liquid_step_internal::TankContainingCell(geometry, particle.world_position);
  if (!cell) {
    return {};
  }
  const auto cell_key = CellKey(*cell);
  const auto pressure_cell = std::ranges::lower_bound(frozen.cells, cell_key, {}, [](const auto& frozen_cell) {
    return CellKey(frozen_cell.cell_index);
  });
  if (pressure_cell == frozen.cells.end() || CellKey(pressure_cell->cell_index) != cell_key) {
    return {};
  }
  const auto cell_ordinal = static_cast<std::size_t>(pressure_cell - frozen.cells.begin());
  if (cell_ordinal >= links.cells.size() || CellKey(links.cells[cell_ordinal].cell_index) != cell_key) {
    return {};
  }
  constexpr std::array components{&Vec3::x, &Vec3::y, &Vec3::z};
  constexpr std::array axes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
  constexpr std::array face_components{&FaceLatticeIndex3D::x, &FaceLatticeIndex3D::y, &FaceLatticeIndex3D::z};
  const Vec3 low_wall{tank.start_low_x, tank.floor_y, tank.low_z};
  const Vec3 high_wall{tank.high_x, std::numeric_limits<double>::infinity(), tank.high_z};
  LiquidRegularCellTransport3D sample;
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    const auto component = components[axis];
    const auto face_component = face_components[axis];
    const FaceIndex3D low_face{axes[axis], {cell->x, cell->y, cell->z}};
    const auto low_plane_index = low_face.lattice_index.*face_component;
    FaceIndex3D high_face = low_face;
    ++(high_face.lattice_index.*face_component);
    const boundary_internal::TankLatticeAxis low_lattice{
        geometry.world_origin.*component,
        static_cast<double>(low_plane_index),
        geometry.cell_width,
    };
    auto high_lattice = low_lattice;
    high_lattice.lattice_index += 1;
    const auto fraction = boundary_internal::TankClipCoordinate(particle.world_position.*component, low_lattice);
    if (!fraction.ok()) {
      return {};
    }
    const auto low_velocity = RegularTransportFaceVelocity(
        frozen,
        links.faces,
        staging,
        low_face,
        particle.side_tag,
        low_lattice,
        low_wall.*component
    );
    const auto high_velocity = RegularTransportFaceVelocity(
        frozen,
        links.faces,
        staging,
        high_face,
        particle.side_tag,
        high_lattice,
        high_wall.*component
    );
    if (!low_velocity || !high_velocity) {
      return {};
    }
    sample.position_in_cell_widths.*component = *fraction;
    sample.low_face_normal_velocity.*component = *low_velocity;
    sample.high_face_normal_velocity.*component = *high_velocity;
  }
  sample.active = true;
  return sample;
}
struct CavityStepScratch {
  LiquidRecords3D<LiquidAirVolumeCell3D> previous_gas;
  LiquidRecords3D<std::size_t> previous_labels;
  LiquidRecords3D<LiquidCavityLineageState3D> previous_lineage;
  LiquidRecords3D<LiquidAirVolumeCell3D> candidate_gas;
  LiquidRecords3D<std::size_t> candidate_labels;
  LiquidRecords3D<std::size_t> queue;
  LiquidRecords3D<LiquidAirComponent3D> components;
  LiquidRecords3D<LiquidAirComponentOverlap3D> overlaps;
  LiquidRecords3D<std::optional<LiquidCavityInventory3D>> inventories;
  LiquidRecords3D<LiquidCavityLineageCandidate3D> candidates;
  LiquidRecords3D<LiquidCavityBoundaryComponent3D> boundary_components;
  LiquidRecords3D<LiquidAirCellPressure3D> pressures;
  std::size_t reserved_bytes = 0;

  static absl::StatusOr<CavityStepScratch> Create(std::size_t capacity) {
    CavityStepScratch owner;
    const auto reserve = [&]<typename Record>(LiquidRecords3D<Record>& records) -> absl::Status {
      constexpr std::size_t stride = sizeof(Record);
      if (capacity > (SIZE_MAX - owner.reserved_bytes) / stride) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      auto storage = LiquidRecords3D<Record>::Create(capacity);
      if (!storage.ok()) {
        return storage.status();
      }
      STEP_TRY(storage->Resize(capacity));
      records = std::move(*storage);
      owner.reserved_bytes += capacity * stride;
      return absl::OkStatus();
    };
    STEP_TRY(reserve(owner.previous_gas));
    STEP_TRY(reserve(owner.previous_labels));
    STEP_TRY(reserve(owner.previous_lineage));
    STEP_TRY(reserve(owner.candidate_gas));
    STEP_TRY(reserve(owner.candidate_labels));
    STEP_TRY(reserve(owner.queue));
    STEP_TRY(reserve(owner.components));
    STEP_TRY(reserve(owner.overlaps));
    STEP_TRY(reserve(owner.inventories));
    STEP_TRY(reserve(owner.candidates));
    STEP_TRY(reserve(owner.boundary_components));
    STEP_TRY(reserve(owner.pressures));
    return owner;
  }

  absl::StatusOr<LiquidStepCavityBoundaries3D> Prepare(
      const LiquidSceneCavity3D& settings,
      const LiquidBoundaryGeometryViewInputs3D& samples,
      LiquidState3D& state,
      std::optional<LiquidCavityStepPreparation3D>& measurement
  ) {
    std::span<const FaceIndex3D> outlets;
    if (settings.atmosphere_outlets) {
      const auto supplied = settings.atmosphere_outlets(settings.context, samples);
      if (!supplied.ok()) {
        return supplied.status();
      }
      outlets = *supplied;
    }
    struct BirthEquilibrium {
      const LiquidSceneCavity3D& settings;
      double surface_tension;
    } equilibrium{settings, samples.surface_tension};
    const auto default_initialize = [](void* context, const LiquidAirComponent3D& component) {
      const auto& birth = *static_cast<const BirthEquilibrium*>(context);
      return InitializeSubmergedLiquidCavity3D(
          component.gas_volume,
          birth.settings.atmospheric_absolute_pressure,
          birth.settings.birth_hydrostatic_gauge_pressure,
          birth.surface_tension
      );
    };
    // The transaction supplies the last accepted snapshot, including on resume
    // and capacity replay. Gas passages are only needed for the current labeler.
    for (std::size_t index = 0; index < state.cavity_gas.size(); ++index) {
      const auto& saved = state.cavity_gas[index];
      previous_gas[index] = {.cell_index = saved.cell_index, .gas_volume = saved.gas_volume};
      previous_labels[index] = saved.component;
    }
    for (std::size_t index = 0; index < state.cavity_components.size(); ++index) {
      const auto& saved = state.cavity_components[index];
      auto& lineage = previous_lineage[index];
      lineage = {.id = saved.id};
      if (saved.id != 0) {
        lineage.inventory = LiquidCavityInventory3D{saved.initial_volume, saved.initial_absolute_pressure};
      }
    }
    const LiquidCavityStepInputs3D inputs{
        .geometry = samples.mac_grid.geometry(),
        .obstacle = samples.obstacle,
        .sampled_cells = samples.cell_indices,
        .corners = samples.corner_distances,
        .centers = samples.center_distances,
        .atmosphere_outlets = outlets,
        .previous_snapshot =
            {
                previous_gas.values().first(state.cavity_gas.size()),
                previous_labels.values().first(state.cavity_gas.size()),
                state.cavity_components.size(),
            },
        .previous_lineage = previous_lineage.values().first(state.cavity_components.size()),
        .next_id = state.next_cavity_id,
        .atmospheric_absolute_pressure = settings.atmospheric_absolute_pressure,
        .liquid_density = samples.liquid_density,
        .exterior = !settings.atmosphere_outlets && std::holds_alternative<std::monostate>(samples.obstacle)
            ? LiquidAirExterior3D::kUnobstructedBoundingBox
            : LiquidAirExterior3D::kExplicitOutlets,
        .initialization_context = settings.initialize ? settings.context : &equilibrium,
        .initialize = settings.initialize ? settings.initialize : +default_initialize,
    };
    const LiquidCavityStepScratch3D scratch{
        .gas_cells = candidate_gas.values(),
        .labels = candidate_labels.values(),
        .queue = queue.values(),
        .components = components.values(),
        .overlaps = overlaps.values(),
        .inventories = inventories.values(),
        .lineage = candidates.values(),
        .boundary_components = boundary_components.values(),
        .air_pressures = pressures.values(),
    };
    const auto prepared = PrepareLiquidCavityStep3D(inputs, scratch);
    if (!prepared.ok()) {
      return prepared.status();
    }
    STEP_TRY(state.cavity_gas.Resize(prepared->gas_cell_count));
    STEP_TRY(state.cavity_components.Resize(prepared->component_count));
    for (std::size_t index = 0; index < prepared->gas_cell_count; ++index) {
      const auto& cell = candidate_gas[index];
      state.cavity_gas[index] = {cell.cell_index, cell.gas_volume, candidate_labels[index]};
    }
    for (std::size_t index = 0; index < prepared->component_count; ++index) {
      const auto& lineage = candidates[index].state;
      auto& saved = state.cavity_components[index];
      saved = {.id = lineage.id};
      if (lineage.inventory) {
        saved.initial_volume = lineage.inventory->initial_volume;
        saved.initial_absolute_pressure = lineage.inventory->initial_absolute_pressure;
      }
    }
    state.next_cavity_id = prepared->lineage.next_id;
    measurement = *prepared;
    return LiquidStepCavityBoundaries3D{
        .air_pressures = pressures.values().first(prepared->boundaries.air_pressure_count),
        .maximum_step_duration = prepared->boundaries.maximum_step_duration,
    };
  }
};

}  // namespace
struct LiquidSubstepScheduler3D::Impl {
  LiquidFrameTransaction3D transaction;
  LiquidStepWorkspace3D workspace;
  liquid_step_internal::Staging staging;
  LiquidRecords3D<LiquidEmissionSource3D> sources;
  LiquidRecords3D<LiquidConstituentMaterial3D> constituent_materials;
  LiquidRecords3D<LiquidPhasePair3D> phase_pairs;
  LiquidSceneInputs3D scene;
  std::optional<CavityStepScratch> cavity;
  LiquidStepOptions3D options;
  LiquidSubstepBackend3D backend;
  std::optional<ResolvedOceanBoundary3D> ocean_boundary;
  std::optional<liquid_step_internal::OceanInletParticles3D> ocean_inlet_particles;
  LiquidStepDiagnostics3D diagnostics;
  LiquidStageFaultHook3D hook;
  double target_time = 0;
  bool frame_active = false;
  Impl(
      LiquidFrameTransaction3D transaction_owner,
      LiquidStepWorkspace3D workspace_owner,
      liquid_step_internal::Staging staging_owner
  )
      : transaction(std::move(transaction_owner)),
        workspace(std::move(workspace_owner)),
        staging(std::move(staging_owner)) {
  }
  std::size_t reservation_bytes = 0;
  absl::Status Enter(LiquidPhysicalStage3D stage, double step_duration) {
    diagnostics.stages[diagnostics.stage_count++] = stage;
    if (hook.before_stage) {
      return hook.before_stage(hook.context, stage, step_duration);
    }
    return absl::OkStatus();
  }
  void SetWorkPhase(LiquidWorkPhaseObserver3D observer, LiquidWorkPhase3D phase) {
    if (observer.enter) {
      observer.enter(observer.context, phase);
    }
    if (backend.set_work_phase) {
      backend.set_work_phase(backend.context, phase);
    }
  }
  absl::Status Execute(
      double step_duration,
      LiquidState3D& trial,
      LiquidAcceptedStepMeasurement3D& accepted_measurement,
      LiquidWorkPhaseObserver3D work_phase
  ) {
    auto& boundary = workspace.boundary();
    const double old_time = transaction.committed().clock.time;
    const double new_time = old_time + step_duration;
    const auto obstacle = liquid_step_internal::ObstacleAtTime(scene.obstacle, old_time);
    auto new_obstacle = liquid_step_internal::ObstacleAtTime(scene.obstacle, new_time);
    const MacDepositQuantization3D quantization{
        options.mass_per_quantum,
        options.momentum_per_quantum,
        options.capacities.particles,
    };
    STEP_TRY(Enter(LiquidPhysicalStage3D::kEmit, step_duration));
    if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
      STEP_TRY(boundary_internal::ArithmeticStatus(boundary_internal::CheckTankStep(*tank, step_duration), true));
      // Contact and frozen geometry use the start pose and this exact duration.
      new_obstacle = liquid_step_internal::ObstacleAtTime(obstacle, step_duration);
    }
    STEP_TRY(liquid_step_internal::Emit(scene, step_duration, trial));
    if (ocean_inlet_particles) {
      STEP_TRY(ocean_inlet_particles->Apply(*ocean_boundary, obstacle, old_time, trial));
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kActivate, step_duration));
    STEP_TRY(liquid_step_internal::PrepareField(scene.field, obstacle, trial, staging, backend, cavity.has_value()));
    STEP_TRY(backend.topology(
        backend.context,
        trial.particles.values(),
        staging.candidates.values(),
        transaction.committed(),
        workspace
    ));
    STEP_TRY(workspace.ApplyTrialTopology(trial));
    const auto topology = workspace.prepared();
    if (!topology.ok()) {
      return topology.status();
    }
    const auto grid = topology->grid;
    STEP_TRY(liquid_step_internal::PrepareTransfers(trial, grid, staging));
    STEP_TRY(backend.sample_faces(backend.context, grid, obstacle, 0, staging.tags.values(), {}));
    SetWorkPhase(work_phase, LiquidWorkPhase3D::kDepositNormalization);
    STEP_TRY(Enter(LiquidPhysicalStage3D::kDeposit, step_duration));
    const auto deposits = backend.deposit(
        backend.context,
        {grid, staging.particles.values(), staging.tags.values(), quantization},
        boundary
    );
    if (!deposits.ok()) {
      return deposits.status();
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kOldSnapshot, step_duration));
    const auto normalized =
        backend.normalize(backend.context, deposits->face_quanta, quantization, options.minimum_face_mass, boundary);
    if (!normalized.ok()) {
      return normalized.status();
    }
    std::ranges::copy(normalized->face_velocities, staging.old_velocities.values().begin());
    std::ranges::copy(normalized->validity, staging.validity.values().begin());
    const LiquidBoundaryVelocityHistory3D old_history{
        .velocity_sample_time = old_time,
        .wall_sample_time = old_time,
        .face_velocities = staging.old_velocities.values(),
        .validity = normalized->validity,
        .face_tags = staging.tags.values(),
        .wall_samples = staging.old_walls.values(),
    };
    SetWorkPhase(work_phase, LiquidWorkPhase3D::kForcesGeometryPressure);
    STEP_TRY(Enter(LiquidPhysicalStage3D::kForces, step_duration));
    STEP_TRY(backend.forces(
        backend.context,
        grid,
        staging.old_velocities.values(),
        scene.world_acceleration,
        step_duration,
        staging.new_velocities.values()
    ));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kWaves, step_duration));
    // Both backends publish predicted velocities into host staging. Pressure
    // consumes the relaxed staging values, and old history remains untouched.
    const auto relax_face = [&](std::size_t slot) -> absl::Status {
      const auto face = grid.FaceIndex(slot);
      if (!face.ok()) {
        return face.status();
      }
      const auto anchor = grid.FaceWorldAnchor(*face);
      if (!anchor.ok()) {
        return anchor.status();
      }
      const auto velocity =
          ocean_boundary->RelaxFace(face->axis, *anchor, staging.new_velocities[slot], new_time, step_duration);
      if (!velocity.ok()) {
        return velocity.status();
      }
      staging.new_velocities[slot] = *velocity;
      return absl::OkStatus();
    };
    if (ocean_boundary) {
      for (std::size_t slot = 0; slot < grid.face_count(); ++slot) {
        const auto validity = normalized->validity[slot];
        if (validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted) {
          STEP_TRY(relax_face(slot));
        }
      }
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kBodyContact, step_duration));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kInterface, step_duration));
    std::optional<double> exact_curvature;
    const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field);
    const bool analytic_sphere_uses_capillarity = sphere && scene.surface_tension > 0;
    if (analytic_sphere_uses_capillarity) {
      const double outward_curvature = 2 / sphere->radius;
      if (std::isfinite(outward_curvature)) {
        exact_curvature = outward_curvature;
      }
    }
    const LiquidBoundaryGeometryViewInputs3D geometry_inputs{
        .mac_grid = grid,
        .cell_indices = staging.candidates.values(),
        .center_distances = staging.centers.values(),
        .corner_distances = staging.corners.values(),
        .obstacle = obstacle,
        .step_duration = step_duration,
        .liquid_density = scene.liquid_density,
        .surface_tension = scene.surface_tension,
        .exact_outward_interface_curvature = exact_curvature,
    };
    LiquidStepCavityBoundaries3D cavity_boundaries;
    if (cavity || scene.cavity_provider.prepare) {
      const auto prepared = cavity
          ? cavity->Prepare(*scene.cavity, geometry_inputs, trial, accepted_measurement.cavity)
          : scene.cavity_provider.prepare(scene.cavity_provider.context, old_time, geometry_inputs);
      if (!prepared.ok()) {
        return prepared.status();
      }
      cavity_boundaries = *prepared;
      const double bound = cavity_boundaries.maximum_step_duration;
      if (!(bound > 0.0)) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
      if (std::isfinite(bound)) {
        accepted_measurement.step_start_cavity_bound_seconds = bound;
        if (step_duration > bound) {
          return Error(absl::StatusCode::kAborted);
        }
      }
    }
    const auto frozen = backend.freeze(backend.context, geometry_inputs, boundary);
    if (!frozen.ok()) {
      // Geometry uses FailedPrecondition for both step_duration certification and incompatible
      // geometry. Recheck only the former at the conservative capillary duration;
      // an incompatible geometry remains fatal, regardless of attempted step_duration.
      if (frozen.status().code() == absl::StatusCode::kFailedPrecondition && scene.surface_tension > 0) {
        const double h = trial.geometry.cell_width;
        const double capillary_duration =
            std::sqrt(scene.liquid_density * h * h * h / (4 * std::numbers::pi * scene.surface_tension));
        if (std::isfinite(capillary_duration) && capillary_duration > 0 && step_duration > capillary_duration) {
          auto certified_inputs = geometry_inputs;
          certified_inputs.step_duration = capillary_duration;
          const auto certified = backend.freeze(backend.context, certified_inputs, boundary);
          if (certified.ok() && step_duration > certified->capillary_step_bound) {
            return Error(absl::StatusCode::kAborted);
          }
        }
      }
      return frozen.status();
    }
    if (frozen->capillary_step_bound != std::numeric_limits<double>::infinity()) {
      // An unresolved sphere without an open air link has no capillary
      // contribution. Require a representable curvature when a link uses it.
      if (analytic_sphere_uses_capillarity && !exact_curvature.has_value()) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      accepted_measurement.step_start_capillary_bound_seconds = frozen->capillary_step_bound;
    }
    const auto* tank_at_step_start = std::get_if<AxisAlignedLiquidTank3D>(&obstacle);
    if (tank_at_step_start) {
      accepted_measurement.piston_pose = LiquidPistonStepPose3D{
          .step_start_low_x_m = tank_at_step_start->start_low_x,
          .step_end_low_x_m = std::get<AxisAlignedLiquidTank3D>(new_obstacle).start_low_x,
      };
    }
    std::optional<LiquidEos3D> eos;
    if (scene.eos) {
      eos = LiquidEos3D{scene.liquid_density, scene.eos->reference_sound_speed, scene.eos->law};
    }
    double minimum_predictor_density = scene.liquid_density;
    STEP_TRY(staging.pressure_fields.Resize(frozen->cells.size()));
    STEP_TRY(staging.incident_velocities.Resize(frozen->faces.size()));
    const auto particles = trial.particles.values();
    const auto first_explicit = std::ranges::find_if(particles, [](const LiquidParticleState3D& particle) {
      return particle.constituent_count != 0;
    });
    const bool uses_composition_density = staging.composition_density && first_explicit != particles.end();
    const bool uses_composition_viscosity = uses_composition_density && staging.composition_viscosity.has_value();
    const double sensor_viscosity = uses_composition_viscosity ? scene.dynamic_viscosity : 0.0;
    std::span<const LiquidInternalInterface3D> internal_interfaces;
    if (uses_composition_density) {
      STEP_TRY(staging.composition_density->Resize(frozen->cells.size()));
      const auto second_phase = std::ranges::find_if(particles, [&](const LiquidParticleState3D& particle) {
        return particle.constituent_count != 0 && particle.phase_id != first_explicit->phase_id;
      });
      if (second_phase == particles.end()) {
        STEP_TRY(liquid_step_internal::SampleLiquidCellCompositionDensity3D(
            grid.geometry(),
            frozen->cells,
            particles,
            scene.constituent_materials,
            scene.liquid_density,
            staging.composition_density->values(),
            sensor_viscosity
        ));
      } else {
        const auto* pair = FindLiquidPhasePair3D(scene.phase_pairs, first_explicit->phase_id, second_phase->phase_id);
        const auto* planar_tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle);
        const bool supported_exterior = std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field)
            || (std::holds_alternative<LiquidHorizontalPlaneSurface3D>(scene.field) && planar_tank
                && planar_tank->piston_velocity_x == 0.0);
        if (!pair || !staging.phase_crossings || !supported_exterior || staging.drift_sensors) {
          return Error(absl::StatusCode::kFailedPrecondition);
        }
        STEP_TRY(staging.phase_crossings->Resize(frozen->faces.size()));
        const auto count = liquid_step_internal::PrepareLiquidPhaseCrossings3D(
            *frozen,
            particles,
            scene.constituent_materials,
            *pair,
            staging.composition_density->values(),
            staging.phase_crossings->values(),
            sensor_viscosity
        );
        if (!count.ok()) {
          return count.status();
        }
        internal_interfaces = staging.phase_crossings->values().first(*count);
      }
    }
    if (staging.drift_sensors) {
      STEP_TRY(staging.drift_sensors->Resize(frozen->cells.size()));
      STEP_TRY(liquid_step_internal::SampleLiquidCellVolumeRatios3D(
          grid.geometry(),
          frozen->cells,
          trial.particles.values(),
          0,
          staging.drift_sensors->values()
      ));
    }
    for (std::size_t index = 0; index < frozen->cells.size(); ++index) {
      const auto cells = trial.cells.values();
      const auto key = frozen->cells[index].cell_index;
      const auto found = std::lower_bound(cells.begin(), cells.end(), key, [](const auto& cell, auto value) {
        return CellKey(cell.cell_index) < CellKey(value);
      });
      if (found == cells.end() || CellKey(found->cell_index) != CellKey(key)) {
        return Error(absl::StatusCode::kInternal);
      }
      double target_divergence = scene.target_divergence;
      double predicted_pressure = found->previous_pressure;
      double tangent_bulk_modulus = scene.bulk_modulus;
      std::optional<double> predicted_density;
      if (uses_composition_density) {
        const auto& sensed = (*staging.composition_density)[index];
        if (sensed.sampled_volume_m3 > 0.0) {
          predicted_density = sensed.sampled_mass_kg / sensed.sampled_volume_m3;
          if (!std::isfinite(*predicted_density) || *predicted_density <= 0.0) {
            return Error(absl::StatusCode::kOutOfRange);
          }
        } else if (sensed.extrapolated_density > 0.0) {
          predicted_density = sensed.extrapolated_density;
        }
      }
      if (eos) {
        const auto pressure = liquid_step_internal::AdvectLiquidPressureHistory3D(
            transaction.committed(),
            key,
            step_duration,
            scene.ambient_pressure
        );
        if (!pressure.ok()) {
          return pressure.status();
        }
        predicted_pressure = *pressure;
        auto local_eos = *eos;
        local_eos.reference_density = predicted_density.value_or(scene.liquid_density);
        const auto predictor = EvaluateLiquidEosAtPressure3D(local_eos, predicted_pressure);
        if (!predictor.ok()) {
          return predictor.status();
        }
        tangent_bulk_modulus = predictor->bulk_modulus;
        predicted_density = predictor->density;
        if (staging.drift_sensors) {
          const auto correction = liquid_step_internal::LiquidVolumeDriftRate3D(
              (*staging.drift_sensors)[index],
              predictor->log_volume_ratio,
              scene.drift->correction_time,
              scene.drift->maximum_rate
          );
          if (!correction.ok()) {
            return correction.status();
          }
          if (*correction != 0.0) {
            target_divergence += *correction;
          }
          if (!std::isfinite(target_divergence)) {
            return Error(absl::StatusCode::kOutOfRange);
          }
        }
      }
      if (predicted_density) {
        minimum_predictor_density = std::min(minimum_predictor_density, *predicted_density);
      }
      staging.pressure_fields[index] = {
          .target_divergence = target_divergence,
          .bulk_modulus = tangent_bulk_modulus,
          .previous_pressure = predicted_pressure,
          .density = predicted_density,
      };
    }
    if ((eos || uses_composition_density) && accepted_measurement.step_start_capillary_bound_seconds) {
      // Freeze used reference density. Tighten its active bound when the
      // pressure operator sees lighter liquid at the advected predictor.
      const double density_scale = std::sqrt(minimum_predictor_density) / std::sqrt(scene.liquid_density);
      auto& capillary_bound = *accepted_measurement.step_start_capillary_bound_seconds;
      capillary_bound *= density_scale;
      if (!(capillary_bound > 0.0)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      if (step_duration > capillary_bound) {
        return Error(absl::StatusCode::kAborted);
      }
    }
    for (std::size_t index = 0; index < frozen->faces.size(); ++index) {
      const auto& face = frozen->faces[index];
      const auto slot = face.face_slot;
      if (face.open_area > 0
          && (normalized->validity[slot] == LiquidFaceValidity3D::kMissing
              || normalized->validity[slot] == LiquidFaceValidity3D::kLowMass)) {
        STEP_TRY(backend.sample_faces(
            backend.context,
            grid,
            obstacle,
            staging.tags[slot].side_tag,
            {},
            staging.old_walls.values()
        ));
        const LiquidMissingFacePredictionInputs3D prediction_inputs{
            .grid = grid,
            .old_history = old_history,
            .face_slot = slot,
            .side_tag = staging.tags[slot].side_tag,
            .world_acceleration = scene.world_acceleration,
            .step_duration = step_duration,
        };
        const auto predicted_face_velocity = backend.predict_missing_face(backend.context, prediction_inputs, boundary);
        if (!predicted_face_velocity.ok()) {
          return predicted_face_velocity.status();
        }
        // Preserve old validity: pressure prediction must not erase a low-mass
        // retention fallback. The backend applies acceleration once to continuation.
        staging.new_velocities[slot] = *predicted_face_velocity;
        if (ocean_boundary) {
          STEP_TRY(relax_face(slot));
        }
      }
      staging.incident_velocities[index] = staging.new_velocities[slot];
    }
    // Evaluate the signed wall flux at the frozen pose. Pressure link assembly
    // consumes the face-order borrow before the wetted-piston workspace is reused.
    std::span<const std::optional<double>> wetted_piston_face_fluxes;
    if (staging.wetted_piston.has_value()) {
      const WettedPistonInputs3D wetted_piston_inputs{
          .field = scene.field,
          .particles = trial.particles.values(),
          .frozen = *frozen,
          .tank = *tank_at_step_start,
      };
      const auto wetted = backend.wetted_piston(backend.context, wetted_piston_inputs, *staging.wetted_piston);
      if (!wetted.ok()) {
        return wetted.status();
      }
      if (wetted->face_fluxes.size() != frozen->faces.size()) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
      wetted_piston_face_fluxes = wetted->face_fluxes;
    }
    auto links = backend.links(
        backend.context,
        *frozen,
        {
            .cells = staging.pressure_fields.values(),
            .predicted_face_velocities = staging.incident_velocities.values(),
            .ambient_pressure = scene.ambient_pressure,
            .wetted_piston_face_fluxes = wetted_piston_face_fluxes,
            .air_cell_pressures = cavity_boundaries.air_pressures,
            .internal_interfaces = internal_interfaces,
        },
        boundary
    );
    if (!links.ok()) {
      return links.status();
    }
    if (staging.isolated_drop) {
      STEP_TRY(liquid_step_internal::SetIsolatedDropPressureLifts(
          grid.geometry(),
          scene.surface_tension,
          scene.ambient_pressure,
          staging,
          *links
      ));
    }
    diagnostics.lower_theta_regularization_count = links->lower_theta_regularization_count;
    STEP_TRY(Enter(LiquidPhysicalStage3D::kViscosity, step_duration));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kPressure, step_duration));
    STEP_TRY(trial.cells.Resize(frozen->cells.size()));
    // An empty, successfully frozen geometry has no constraint rows. All other
    // stages still execute, including transfer/forces and backend particle update.
    if (!links->cells.empty()) {
      auto pressure_problem = *links;
      if (staging.drift_pressure_cells) {
        STEP_TRY(
            ApplyIncompressibleDrift(grid, step_duration, *scene.drift, staging, workspace.pressure(), pressure_problem)
        );
      }
      pressure_problem.seed_uniform_dirichlet_rest_pressure = exact_curvature.has_value();
      std::span<const double> projected_cell_pressures;
      std::span<const double> projected_face_velocities;
      MacPressureDiagnostics3D pressure_diagnostics;
      MacPressureFrozenMeasurement3D frozen_measurement;
      std::span<double> cell_dynamic_viscosity;
      if (uses_composition_viscosity) {
        if (pressure_problem.cells.size() != frozen->cells.size()) {
          return absl::InternalError("viscosity staging lost frozen cells");
        }
        STEP_TRY(staging.composition_viscosity->Resize(pressure_problem.cells.size()));
        cell_dynamic_viscosity = staging.composition_viscosity->values();
        for (std::size_t index = 0; index < pressure_problem.cells.size(); ++index) {
          if (CellKey(pressure_problem.cells[index].cell_index) != CellKey(frozen->cells[index].cell_index)) {
            return absl::InternalError("viscosity staging lost frozen cell order");
          }
          const auto& sensor = (*staging.composition_density)[index];
          const double viscosity = sensor.sampled_volume_m3 > 0.0
              ? sensor.sampled_viscosity_volume / sensor.sampled_volume_m3
              : sensor.extrapolated_viscosity;
          if (!std::isfinite(viscosity) || viscosity < 0.0) {
            return absl::OutOfRangeError("sampled liquid viscosity is unrepresentable");
          }
          cell_dynamic_viscosity[index] = viscosity;
        }
      }
      const bool active_viscosity = cell_dynamic_viscosity.empty()
          ? scene.dynamic_viscosity > 0.0
          : std::ranges::any_of(cell_dynamic_viscosity, [](double viscosity) {
              return viscosity > 0.0;
            });
      if (!active_viscosity) {
        const auto projection = backend.project(
            backend.context,
            grid,
            step_duration,
            pressure_problem,
            options.pressure_solver_options,
            workspace.pressure()
        );
        if (!projection.ok()) {
          return projection.status();
        }
        projected_cell_pressures = projection->cell_pressures;
        projected_face_velocities = projection->corrected_face_velocities;
        pressure_diagnostics = projection->diagnostics;
        frozen_measurement = projection->frozen_measurement;
      } else {
        auto result = backend.project_viscous(
            backend.context,
            geometry_inputs,
            pressure_problem,
            scene.dynamic_viscosity,
            cell_dynamic_viscosity,
            options.pressure_solver_options,
            workspace.pressure()
        );
        if (!result.ok()) {
          return result.status();
        }
        const auto& coupled = *result;
        accepted_measurement.viscous_max_momentum_residual_kg_m_per_s = coupled.max_momentum_residual;
        projected_cell_pressures = coupled.cell_pressures;
        projected_face_velocities = coupled.face_velocities;
        pressure_diagnostics = {
            .iteration_count = coupled.pressure_iterations,
            .true_residual_l2 = coupled.constraint_residual_l2,
            .max_volume_scaled_residual = coupled.max_cell_constraint_defect,
            .max_cell_constraint_defect = coupled.max_cell_constraint_defect,
        };
        frozen_measurement.integrated_rhs_l2_m3_per_s = coupled.initial_constraint_residual_l2;
        frozen_measurement.inward_prescribed_wall_flux_m3_per_s = coupled.inward_prescribed_wall_flux_m3_per_s;
        for (const auto& cell : pressure_problem.cells) {
          frozen_measurement.pressure_cell_volume_m3 += cell.fluid_volume;
        }
      }
      if (projected_cell_pressures.size() != frozen->cells.size()
          || projected_face_velocities.size() != frozen->faces.size()) {
        return absl::InternalError("pressure projection extents do not match frozen geometry");
      }
      if (eos) {
        for (const double pressure : projected_cell_pressures) {
          const auto accepted_eos = EvaluateLiquidEosAtPressure3D(*eos, pressure);
          if (!accepted_eos.ok()) {
            return accepted_eos.status();
          }
        }
      }
      diagnostics.pressure_diagnostics = pressure_diagnostics;
      accepted_measurement.pressure = pressure_diagnostics;
      accepted_measurement.pressure_rhs_l2_m3_per_s = frozen_measurement.integrated_rhs_l2_m3_per_s;
      accepted_measurement.step_start_pressure_cell_volume_m3 = frozen_measurement.pressure_cell_volume_m3;
      if (accepted_measurement.piston_pose.has_value()) {
        accepted_measurement.step_start_inward_piston_flux_m3_per_s =
            frozen_measurement.inward_prescribed_wall_flux_m3_per_s;
      }
      for (std::size_t index = 0; index < frozen->cells.size(); ++index) {
        trial.cells[index] = {
            frozen->cells[index].cell_index,
            LiquidCellOccupancy3D::kLiquid,
            projected_cell_pressures[index],
        };
      }
      for (std::size_t index = 0; index < frozen->faces.size(); ++index) {
        const auto slot = frozen->faces[index].face_slot;
        staging.new_velocities[slot] = projected_face_velocities[index];
        if (staging.validity[slot] == LiquidFaceValidity3D::kMissing && frozen->faces[index].open_area > 0) {
          staging.validity[slot] = LiquidFaceValidity3D::kPredicted;
        }
      }
    }
    SetWorkPhase(work_phase, LiquidWorkPhase3D::kGatherContactAcceptance);
    STEP_TRY(Enter(LiquidPhysicalStage3D::kNewSnapshot, step_duration));
    STEP_TRY(backend.sample_faces(backend.context, grid, new_obstacle, 0, staging.new_tags.values(), {}));
    for (std::size_t index = 0; index < trial.faces.size(); ++index) {
      trial.faces[index].carrier_velocity = staging.new_velocities[index];
      trial.faces[index].validity = staging.validity[index];
      trial.faces[index].side_tag = staging.new_tags[index].side_tag;
      trial.faces[index].structurally_eligible = staging.new_tags[index].structurally_eligible;
    }
    std::span<const LiquidFaceValidity3D> projected_gather_validity;
    if (staging.projected_gather_validity && !links->cells.empty()) {
      auto validity = staging.projected_gather_validity->values();
      // Preserve the committed carrier and old history. In the new history,
      // unprojected usable faces must use continuation from projected donors.
      for (std::size_t slot = 0; slot < validity.size(); ++slot) {
        const auto carrier_validity = staging.validity[slot];
        const bool usable =
            carrier_validity == LiquidFaceValidity3D::kValid || carrier_validity == LiquidFaceValidity3D::kPredicted;
        const bool unprojected_donor = staging.new_tags[slot].structurally_eligible && usable;
        validity[slot] = unprojected_donor ? LiquidFaceValidity3D::kMissing : carrier_validity;
      }
      for (const auto& face : frozen->faces) {
        validity[face.face_slot] = staging.validity[face.face_slot];
      }
      projected_gather_validity = validity;
    }
    const LiquidBoundaryVelocityHistory3D new_history{
        .velocity_sample_time = new_time,
        .wall_sample_time = new_time,
        .face_velocities = staging.new_velocities.values(),
        .validity = staging.validity.values(),
        .face_tags = staging.new_tags.values(),
        .wall_samples = staging.new_walls.values(),
    };
    STEP_TRY(Enter(LiquidPhysicalStage3D::kGather, step_duration));
    // Thin-slab stencil nodes can be shared by opposite incident walls. Reuse
    // the two frozen wall histories only within a consecutive run on one side.
    std::optional<std::int64_t> sampled_wall_side;
    std::span<const LiquidBoundaryParticleGather3D> pending_gathers;
    std::size_t attempted_run_end = 0;
    bool attempted_run_uses_projected_history = false;
    std::size_t field_chunk_end = 0;
    bool field_chunk_ready = false;
    bool continue_isolated_projection = false;
    for (std::size_t index = 0; index < trial.particles.size(); ++index) {
      const auto& particle = trial.particles[index];
      const auto side = particle.side_tag;
      if (sampled_wall_side != side) {
        STEP_TRY(backend.sample_faces(backend.context, grid, obstacle, side, {}, staging.old_walls.values()));
        STEP_TRY(backend.sample_faces(backend.context, grid, new_obstacle, side, {}, staging.new_walls.values()));
        sampled_wall_side = side;
        if (staging.isolated_drop && !projected_gather_validity.empty()) {
          auto projected_history = new_history;
          projected_history.validity = projected_gather_validity;
          continue_isolated_projection = CanContinueIsolatedDropProjection(old_history, projected_history);
        }
      }
      bool use_projected_history = false;
      if (!projected_gather_validity.empty()) {
        if (index < attempted_run_end) {
          use_projected_history = attempted_run_uses_projected_history;
        } else {
          const auto selected = UsesProjectedGatherHistory(*frozen, particle.world_position);
          if (!selected.has_value()) {
            return Error(absl::StatusCode::kOutOfRange);
          }
          use_projected_history = *selected;
        }
      }
      LiquidBoundaryGatherViewInputs3D gather_inputs{
          .mac_grid = grid,
          .particles = staging.particles.values().subspan(index, 1),
          .old_time = old_time,
          .new_time = new_time,
          .old_history = old_history,
          .new_history = new_history,
      };
      if (use_projected_history) {
        gather_inputs.new_history.validity = projected_gather_validity;
      }
      const bool continues_projected_change = use_projected_history && continue_isolated_projection;
      if (backend.gather_run && index >= attempted_run_end) {
        attempted_run_end = index + 1;
        attempted_run_uses_projected_history = use_projected_history;
        while (attempted_run_end < trial.particles.size() && trial.particles[attempted_run_end].side_tag == side) {
          if (!projected_gather_validity.empty()) {
            const auto& next = trial.particles[attempted_run_end];
            const auto selected = UsesProjectedGatherHistory(*frozen, next.world_position);
            // A speculative lookup must not report an error before earlier gathers or field samples.
            if (!selected.has_value() || *selected != use_projected_history) {
              break;
            }
          }
          ++attempted_run_end;
        }
        const auto run_count = attempted_run_end - index;
        if (run_count > 1) {
          auto run_inputs = gather_inputs;
          run_inputs.particles = staging.particles.values().subspan(index, run_count);
          const auto run = continues_projected_change
              ? GatherIsolatedDropHistories(
                    backend,
                    *backend.gather_run,
                    run_inputs,
                    boundary,
                    staging.isolated_drop_restricted_gathers->values().subspan(index, run_count),
                    staging.gathered.values().subspan(index, run_count)
                )
              : backend.gather_run(backend.context, run_inputs, boundary);
          if (run.ok()) {
            if (run->particles.size() != run_count) {
              return Error(absl::StatusCode::kInternal);
            }
            pending_gathers = run->particles;
          } else if (run.status().code() == absl::StatusCode::kInternal) {
            return run.status();
          }
          // Replay a reusable rejection so an earlier scalar field error wins.
        }
      }
      const bool used_grouped_gather = !pending_gathers.empty();
      if (used_grouped_gather) {
        const auto& gathered = pending_gathers.front();
        if (continues_projected_change) {
          STEP_TRY(CorrectIsolatedDropProjectedChange(
              staging.gathered[index],
              (*staging.isolated_drop_restricted_gathers)[index],
              gathered,
              staging.gathered[index]
          ));
        } else {
          staging.gathered[index] = gathered;
        }
        diagnostics.old_pic_fallback_count += gathered.old_pic_fallback;
        diagnostics.new_pic_fallback_count += gathered.new_pic_fallback;
        pending_gathers = pending_gathers.subspan(1);
      } else {
        const auto gather = continues_projected_change
            ? GatherIsolatedDropProjectedChange(backend, gather_inputs, boundary, staging.gathered[index])
            : backend.gather(backend.context, gather_inputs, boundary);
        if (!gather.ok()) {
          return gather.status();
        }
        staging.gathered[index] = gather->particles[0];
        diagnostics.old_pic_fallback_count += gather->old_pic_fallback_particle_count;
        diagnostics.new_pic_fallback_count += gather->new_pic_fallback_particle_count;
      }
      if (staging.isolated_drop) {
        const auto distance = staging.isolated_drop->ParticleSignedDistance(particle.world_position);
        if (!distance.ok()) {
          return distance.status();
        }
        staging.particle_distances[index] = *distance;
        continue;
      }
      if (used_grouped_gather && backend.sample_gathered_fields && index >= field_chunk_end) {
        const auto field_count = std::min(kLiquidFieldBatchCapacity3D, attempted_run_end - index);
        field_chunk_end = index + field_count;
        field_chunk_ready = false;
        if (field_count > 1) {
          std::array<Vec3, kLiquidFieldBatchCapacity3D> field_positions;
          for (std::size_t offset = 0; offset < field_count; ++offset) {
            field_positions[offset] = trial.particles[index + offset].world_position;
          }
          const auto distances = staging.particle_distances.values().subspan(index, field_count);
          const auto status = backend.sample_gathered_fields(
              backend.context,
              scene.field,
              trial.particles.values(),
              std::span(field_positions).first(field_count),
              distances
          );
          if (status.code() == absl::StatusCode::kInternal) {
            return status;
          }
          // Batch validation can see a later query failure before evaluating
          // an earlier query. Replay this chunk in scalar order on rejection.
          field_chunk_ready = status.ok() && std::ranges::all_of(distances, [](double distance) {
                                return std::isfinite(distance);
                              });
        }
      }
      if (field_chunk_ready && index < field_chunk_end) {
        continue;
      }
      const auto distance =
          backend.sample_field(backend.context, scene.field, trial.particles.values(), particle.world_position);
      if (!distance.ok()) {
        return distance.status();
      }
      if (!std::isfinite(*distance)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      staging.particle_distances[index] = *distance;
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kParticleUpdate, step_duration));
    if (staging.regular_transport) {
      const auto& tank = std::get<AxisAlignedLiquidTank3D>(obstacle);
      for (std::size_t index = 0; index < trial.particles.size(); ++index) {
        (*staging.regular_transport)[index] =
            RegularTransportSample(*frozen, *links, staging, tank, trial.particles[index]);
      }
    }
    const LiquidParticleUpdateInputs3D particle_update_inputs{
        .gathered = staging.gathered.values(),
        .liquid_distances = staging.particle_distances.values(),
        .obstacle = obstacle,
        .cell_width = trial.geometry.cell_width,
        .step_duration = step_duration,
        .particle_energy_blend = options.particle_energy_blend,
        .friction_coefficient = scene.friction_coefficient,
        .penetration_tolerance = scene.penetration_tolerance,
        .regular_cell_transport = staging.regular_transport ? staging.regular_transport->values()
                                                            : std::span<const LiquidRegularCellTransport3D>{},
    };
    STEP_TRY(backend.update_particles(backend.context, particle_update_inputs, trial.particles.values()));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kWhitewater, step_duration));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kValidate, step_duration));
    const double wall_displacement_bound = step_duration
        * liquid_step_internal::MaximumComponentMagnitude(liquid_step_internal::WallVelocity(scene.obstacle));
    if (wall_displacement_bound > 0.5 * trial.geometry.cell_width) {
      return Error(absl::StatusCode::kAborted);
    }
    const auto retained = transaction.committed().blocks.values();
    for (auto& block : trial.blocks.values()) {
      const auto key = [](BlockIndex3D v) {
        return std::tuple(v.z, v.y, v.x);
      };
      const auto found =
          std::lower_bound(retained.begin(), retained.end(), block.block_index, [&](const auto& old, auto index) {
            return key(old.block_index) < key(index);
          });
      if (found != retained.end() && key(found->block_index) == key(block.block_index)) {
        if (block.activation_age == UINT32_MAX) {
          return Error(absl::StatusCode::kOutOfRange);
        }
        ++block.activation_age;
      }
    }
    STEP_TRY(ValidateLiquidState3D(trial));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kAccept, step_duration));
    return absl::OkStatus();
  }
  absl::StatusOr<double> ScheduledDuration() const {
    const auto& state = transaction.committed();
    const double remaining = target_time - state.clock.time;
    double speed = liquid_step_internal::MaximumComponentMagnitude(liquid_step_internal::WallVelocity(scene.obstacle));
    for (const auto& particle : state.particles.values()) {
      speed = std::max(speed, liquid_step_internal::MaximumComponentMagnitude(particle.world_velocity));
    }
    for (const auto& source : scene.sources) {
      speed = std::max(
          speed,
          liquid_step_internal::MaximumComponentMagnitude(source.emitted_particle_template.world_velocity)
      );
    }
    double maximum = std::min(options.maximum_step_duration, remaining);
    const double acceleration = liquid_step_internal::MaximumComponentMagnitude(scene.world_acceleration);
    // Bound v*step_duration + a*step_duration^2, including this step's full acceleration before advection.
    const double half_width = 0.5 * state.geometry.cell_width;
    if (speed > 0) {
      maximum = std::min(maximum, half_width / speed);
    }
    if (acceleration > 0) {
      const double root = std::hypot(speed, 2 * std::sqrt(acceleration) * std::sqrt(half_width));
      maximum = std::min(maximum, 2 * half_width / (speed + root));
    }
    if (scene.surface_tension > 0) {
      const double h = state.geometry.cell_width;
      maximum = std::min(
          maximum,
          std::sqrt(scene.liquid_density * h * h * h / (4 * std::numbers::pi * scene.surface_tension))
      );
    }
    if (!std::isfinite(maximum) || maximum <= 0) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    const double count = std::ceil(remaining / maximum);
    const double duration = remaining / count;
    if (!std::isfinite(count) || duration <= 0 || state.clock.time + duration == state.clock.time) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    return duration;
  }
};
LiquidSubstepScheduler3D::LiquidSubstepScheduler3D(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}
LiquidSubstepScheduler3D::LiquidSubstepScheduler3D(LiquidSubstepScheduler3D&&) noexcept = default;
LiquidSubstepScheduler3D& LiquidSubstepScheduler3D::operator=(LiquidSubstepScheduler3D&&) noexcept = default;
LiquidSubstepScheduler3D::~LiquidSubstepScheduler3D() = default;
absl::StatusOr<LiquidSubstepScheduler3D> LiquidSubstepScheduler3D::Create(
    const LiquidState3D& initial,
    const LiquidSceneInputs3D& scene,
    LiquidStepOptions3D options
) {
  return Create(initial, scene, options, CpuLiquidSubstepBackend3D());
}
absl::StatusOr<LiquidSubstepScheduler3D> LiquidSubstepScheduler3D::Create(
    const LiquidState3D& initial,
    const LiquidSceneInputs3D& scene,
    LiquidStepOptions3D options,
    const LiquidSubstepBackend3D& backend
) {
  const bool needs_lattice_plane = std::holds_alternative<LiquidHorizontalPlaneSurface3D>(scene.field);
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.obstacle);
  const bool needs_wetted_piston = tank && tank->piston_velocity_x != 0;
  const bool needs_isolated_drop = std::holds_alternative<LiquidIsolatedCovarianceDrop3D>(scene.field);
  if (!Complete(backend) || (needs_lattice_plane ? !backend.sample_lattice_plane : !backend.sample_field_batch)
      || (needs_wetted_piston && !backend.wetted_piston)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  STEP_TRY(liquid_step_internal::ValidateScene(initial, scene, options));
  if (CanApplyViscosity(scene) && !backend.project_viscous) {
    return absl::UnimplementedError("backend does not support coupled viscosity");
  }
  if (CanApplyViscosity(scene) && needs_wetted_piston && !needs_lattice_plane) {
    return absl::UnimplementedError("viscous projection requires certified moving-wall quadratures");
  }
  const auto bytes = QueryLiquidWorkspaceBytes3D(options.capacities);
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto staging = liquid_step_internal::Staging::Create(
      options.capacities,
      CanStageRegularCellTransport(scene, tank),
      needs_wetted_piston,
      needs_isolated_drop,
      scene.drift && scene.drift->maximum_rate > 0.0,
      scene.drift && scene.drift->maximum_rate > 0.0 && !scene.eos,
      !scene.constituent_materials.empty(),
      !scene.phase_pairs.empty(),
      HasConstituentViscosity(scene)
  );
  if (!staging.ok()) {
    return staging.status();
  }
  const auto capacities = options.capacities;
  const auto reservation_bytes = ReservationBytes(
      capacities,
      scene.sources.size(),
      scene.constituent_materials.size(),
      scene.phase_pairs.size(),
      bytes->total_bytes,
      staging->reserved_bytes,
      scene.cavity.has_value()
  );
  if (!reservation_bytes.ok()) {
    return reservation_bytes.status();
  }
  if ((!scene.cavity && initial.next_cavity_id != 0)
      || (scene.cavity && initial.clock.accepted_step_ordinal != 0 && initial.next_cavity_id == 0)) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  std::optional<CavityStepScratch> cavity;
  std::optional<LiquidState3D> cavity_seed;
  if (scene.cavity) {
    if (scene.cavity_provider.prepare || !std::isfinite(scene.cavity->atmospheric_absolute_pressure)
        || scene.cavity->atmospheric_absolute_pressure <= 0.0
        || !std::isfinite(scene.cavity->birth_hydrostatic_gauge_pressure)
        || scene.cavity->birth_hydrostatic_gauge_pressure < 0.0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    const auto cavity_capacity = std::max(capacities.cells, options.capacity_replay_reservations.cells);
    auto scratch = CavityStepScratch::Create(cavity_capacity);
    if (!scratch.ok()) {
      return scratch.status();
    }
    if (scratch->reserved_bytes > SIZE_MAX - *reservation_bytes) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    cavity = std::move(*scratch);
    if (initial.next_cavity_id == 0) {
      const LiquidStateCapacities3D seed_capacities{
          initial.particles.size(),
          initial.blocks.size(),
          initial.cells.size(),
          initial.faces.size(),
          initial.sources.size(),
      };
      auto seed = CloneLiquidState3D(initial, seed_capacities);
      if (!seed.ok()) {
        return seed.status();
      }
      auto gas = LiquidRecords3D<LiquidCavityGasState3D>::Create(capacities.cells);
      auto components = LiquidRecords3D<LiquidCavityComponentState3D>::Create(capacities.cells);
      if (!gas.ok()) {
        return gas.status();
      }
      if (!components.ok()) {
        return components.status();
      }
      seed->cavity_gas = std::move(*gas);
      seed->cavity_components = std::move(*components);
      seed->next_cavity_id = 1;
      const auto obstacle = liquid_step_internal::ObstacleAtTime(scene.obstacle, initial.clock.time);
      // Frame zero has no activated grid. The host field sampler supplies the
      // initial geometry independently of the physical backend's publication.
      STEP_TRY(
          liquid_step_internal::PrepareField(scene.field, obstacle, *seed, *staging, CpuLiquidSubstepBackend3D(), true)
      );
      const auto grid = SparseMacGridView3D::Create(initial.geometry, {});
      if (!grid.ok()) {
        return grid.status();
      }
      const LiquidBoundaryGeometryViewInputs3D samples{
          .mac_grid = *grid,
          .cell_indices = staging->candidates.values(),
          .center_distances = staging->centers.values(),
          .corner_distances = staging->corners.values(),
          .obstacle = obstacle,
          .step_duration = 0.0,
          .liquid_density = scene.liquid_density,
          .surface_tension = scene.surface_tension,
      };
      std::optional<LiquidCavityStepPreparation3D> measurement;
      const auto prepared = cavity->Prepare(*scene.cavity, samples, *seed, measurement);
      if (!prepared.ok()) {
        return prepared.status();
      }
      cavity_seed = std::move(*seed);
    }
  }
  const auto& initial_version = cavity_seed ? *cavity_seed : initial;
  auto transaction = LiquidFrameTransaction3D::Create(initial_version, capacities, scene.cavity.has_value());
  if (!transaction.ok()) {
    return transaction.status();
  }
  auto workspace = LiquidStepWorkspace3D::Create(capacities, initial.geometry);
  if (!workspace.ok()) {
    return workspace.status();
  }
  auto sources = LiquidRecords3D<LiquidEmissionSource3D>::Create(scene.sources.size());
  if (!sources.ok()) {
    return sources.status();
  }
  STEP_TRY(sources->Resize(scene.sources.size()));
  std::ranges::copy(scene.sources, sources->values().begin());
  auto materials = LiquidRecords3D<LiquidConstituentMaterial3D>::Create(scene.constituent_materials.size());
  if (!materials.ok()) {
    return materials.status();
  }
  STEP_TRY(materials->Resize(scene.constituent_materials.size()));
  std::ranges::copy(scene.constituent_materials, materials->values().begin());
  auto phase_pairs = LiquidRecords3D<LiquidPhasePair3D>::Create(scene.phase_pairs.size());
  if (!phase_pairs.ok()) {
    return phase_pairs.status();
  }
  STEP_TRY(phase_pairs->Resize(scene.phase_pairs.size()));
  std::ranges::copy(scene.phase_pairs, phase_pairs->values().begin());
  std::unique_ptr<Impl> impl(new (std::nothrow)
                                 Impl(std::move(*transaction), std::move(*workspace), std::move(*staging)));
  if (!impl) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  impl->reservation_bytes = *reservation_bytes + (cavity ? cavity->reserved_bytes : 0);
  impl->cavity = std::move(cavity);
  impl->sources = std::move(*sources);
  impl->scene = scene;
  impl->scene.sources = impl->sources.values();
  impl->constituent_materials = std::move(*materials);
  impl->scene.constituent_materials = impl->constituent_materials.values();
  impl->phase_pairs = std::move(*phase_pairs);
  impl->scene.phase_pairs = impl->phase_pairs.values();
  impl->options = options;
  if (options.ocean_boundary) {
    auto ocean_boundary = ResolvedOceanBoundary3D::Resolve(*options.ocean_boundary);
    if (!ocean_boundary.ok()) {
      return ocean_boundary.status();
    }
    impl->ocean_boundary = std::move(*ocean_boundary);
    if (options.ocean_boundary->inlet_particle_spacing_m > 0) {
      if (!std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field)) {
        return absl::InvalidArgumentError("ocean particle level control needs a particle-derived liquid field");
      }
      auto inlet = liquid_step_internal::OceanInletParticles3D::Create(*options.ocean_boundary, scene.liquid_density);
      if (!inlet.ok()) {
        return inlet.status();
      }
      if (inlet->reserved_bytes() > SIZE_MAX - impl->reservation_bytes) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      impl->reservation_bytes += inlet->reserved_bytes();
      impl->ocean_inlet_particles = std::move(*inlet);
    }
  }
  STEP_TRY(backend.initialize(backend.context, initial_version));
  impl->backend = backend;
  return LiquidSubstepScheduler3D(std::move(impl));
}
const LiquidState3D& LiquidSubstepScheduler3D::committed() const {
  return impl_->transaction.committed();
}
const LiquidStepDiagnostics3D& LiquidSubstepScheduler3D::diagnostics() const {
  return impl_->diagnostics;
}
std::size_t LiquidSubstepScheduler3D::reserved_bytes() const {
  return impl_->reservation_bytes;
}
void LiquidSubstepScheduler3D::SetStageFaultHookForTesting(LiquidStageFaultHook3D hook) {
  impl_->hook = hook;
}
absl::Status LiquidSubstepScheduler3D::BeginFrame(double target) {
  STEP_TRY(impl_->transaction.BeginFrame(target));
  impl_->target_time = target;
  impl_->frame_active = true;
  impl_->diagnostics = {};
  return absl::OkStatus();
}
absl::StatusOr<LiquidAttemptOutcome3D>
LiquidSubstepScheduler3D::Attempt(double step_duration, LiquidWorkPhaseObserver3D work_phase) {
  auto& impl = *impl_;
  auto& diagnostics = impl.diagnostics;
  if (!impl.frame_active || !std::isfinite(step_duration) || step_duration <= 0
      || step_duration > impl.target_time - committed().clock.time) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  impl.SetWorkPhase(work_phase, LiquidWorkPhase3D::kEmitActivationField);
  const absl::Cleanup end_phase = [&] {
    impl.SetWorkPhase(work_phase, LiquidWorkPhase3D::kCount);
  };
  auto trial = impl.transaction.ResetTrialFromCommitted();
  if (!trial.ok()) {
    return trial.status();
  }
  ++diagnostics.attempts;
  diagnostics.accepted_measurement.reset();
  diagnostics.stage_count = 0;
  diagnostics.attempted_duration = step_duration;
  diagnostics.pressure_diagnostics = {};
  diagnostics.old_pic_fallback_count = 0;
  diagnostics.new_pic_fallback_count = 0;
  diagnostics.lower_theta_regularization_count = 0;
  diagnostics.failure_code = absl::StatusCode::kOk;
  diagnostics.failed_stage = LiquidPhysicalStage3D::kCount;
  LiquidAcceptedStepMeasurement3D accepted_measurement;
  auto status = impl.Execute(step_duration, **trial, accepted_measurement, work_phase);
  if (status.ok()) {
    const auto clock = impl.transaction.PreviewAcceptedClock(step_duration);
    status = clock.ok() ? impl.backend.prepare_publication(impl.backend.context, **trial, *clock) : clock.status();
  }
  auto disposition = LiquidAttemptDisposition3D::kAccepted;
  if (!status.ok()) {
    diagnostics.failure_code = status.code();
    diagnostics.failed_stage = diagnostics.stages[diagnostics.stage_count - 1];
    disposition = LiquidAttemptDisposition3D::kFatal;
    if (status.code() == absl::StatusCode::kResourceExhausted) {
      disposition = LiquidAttemptDisposition3D::kCapacity;
    }
    // Only explicitly step-dependent failures use Aborted. Geometry constraint,
    // source compatibility, invalid range and backend errors are never step_duration retries.
    if (status.code() == absl::StatusCode::kAborted) {
      disposition = LiquidAttemptDisposition3D::kRetryDt;
    }
  }
  auto outcome = impl.transaction.FinishAttempt(disposition, step_duration);
  if (!outcome.ok()) {
    impl.backend.discard_publication(impl.backend.context);
    diagnostics.failure_code = outcome.status().code();
    return outcome.status();
  }
  if (outcome->disposition == LiquidAttemptDisposition3D::kAccepted) {
    impl.backend.commit_publication(impl.backend.context);
    ++diagnostics.accepted_attempts;
    accepted_measurement.accepted_clock = committed().clock;
    accepted_measurement.accepted_dt_seconds = step_duration;
    diagnostics.accepted_measurement = accepted_measurement;
    diagnostics.stages[diagnostics.stage_count++] = LiquidPhysicalStage3D::kAdapt;
  }
  if (outcome->disposition != LiquidAttemptDisposition3D::kAccepted) {
    impl.backend.discard_publication(impl.backend.context);
  }
  if (outcome->retry_duration) {
    ++diagnostics.dt_retries;
  }
  return outcome;
}
absl::Status LiquidSubstepScheduler3D::FinishFrame() {
  STEP_TRY(impl_->transaction.FinishFrame());
  impl_->frame_active = false;
  return absl::OkStatus();
}
absl::Status LiquidSubstepScheduler3D::AdvanceFrame(double target, LiquidProgressObserver3D observer) {
  STEP_TRY(BeginFrame(target));
  std::optional<double> retry;
  while (committed().clock.time < target) {
    auto duration = impl_->ScheduledDuration();
    if (!duration.ok()) {
      return duration.status();
    }
    if (retry) {
      duration = *retry;
    }
    const auto start_clock = committed().clock;
    const auto previous_attempt_count = diagnostics().attempts;
    const auto work_start =
        observer.observe_attempt ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto outcome = Attempt(*duration, observer.work_phase);
    if (observer.observe_attempt && diagnostics().attempts != previous_attempt_count) {
      const std::chrono::duration<double> work_elapsed = std::chrono::steady_clock::now() - work_start;
      observer.observe_attempt(
          observer.context,
          {
              .start_clock = start_clock,
              .attempt_in_frame = diagnostics().attempts,
              .capacity_pass = diagnostics().capacity_replays,
              .attempted_dt_seconds = *duration,
              .work_wall_seconds = work_elapsed.count(),
              .disposition = outcome.ok() ? outcome->disposition : LiquidAttemptDisposition3D::kFatal,
              .retry_dt_seconds = outcome.ok() ? outcome->retry_duration : std::nullopt,
              .failure_code = outcome.ok() ? diagnostics().failure_code : outcome.status().code(),
          }
      );
    }
    if (!outcome.ok()) {
      return outcome.status();
    }
    if (outcome->disposition == LiquidAttemptDisposition3D::kAccepted && observer.observe != nullptr) {
      STEP_TRY(observer.observe(observer.context, committed(), diagnostics(), false));
    }
    retry = outcome->retry_duration;
    if (outcome->disposition == LiquidAttemptDisposition3D::kFatal) {
      return Error(impl_->diagnostics.failure_code);
    }
    if (outcome->disposition == LiquidAttemptDisposition3D::kRetryDt && !retry) {
      return Error(absl::StatusCode::kAborted);
    }
    if (outcome->disposition == LiquidAttemptDisposition3D::kCapacity) {
      if (impl_->diagnostics.capacity_replays != 0) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      const auto growth = impl_->options.capacity_replay_reservations;
      const auto current = impl_->options.capacities;
      if (growth == LiquidStateCapacities3D{} || growth == current) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      auto staging = liquid_step_internal::Staging::Create(
          growth,
          impl_->staging.regular_transport.has_value(),
          impl_->staging.wetted_piston.has_value(),
          impl_->staging.isolated_drop_pressure_faces.has_value(),
          impl_->staging.drift_sensors.has_value(),
          impl_->staging.drift_pressure_cells.has_value(),
          impl_->staging.composition_density.has_value(),
          impl_->staging.phase_crossings.has_value(),
          impl_->staging.composition_viscosity.has_value()
      );
      if (!staging.ok()) {
        return staging.status();
      }
      auto workspace = LiquidStepWorkspace3D::Create(growth, committed().geometry);
      if (!workspace.ok()) {
        return workspace.status();
      }
      const auto reservation_bytes = ReservationBytes(
          growth,
          impl_->sources.size(),
          impl_->constituent_materials.size(),
          impl_->phase_pairs.size(),
          workspace->usage().reserved_bytes,
          staging->reserved_bytes,
          impl_->cavity.has_value()
      );
      if (!reservation_bytes.ok()) {
        return reservation_bytes.status();
      }
      const auto inlet_bytes = impl_->ocean_inlet_particles ? impl_->ocean_inlet_particles->reserved_bytes() : 0;
      const auto cavity_bytes = impl_->cavity ? impl_->cavity->reserved_bytes : 0;
      if (inlet_bytes > SIZE_MAX - *reservation_bytes || cavity_bytes > SIZE_MAX - *reservation_bytes - inlet_bytes) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      const auto prepared_growth =
          impl_->backend.prepare_growth(impl_->backend.context, growth, impl_->transaction.frame_start());
      if (!prepared_growth.ok()) {
        impl_->backend.discard_growth(impl_->backend.context);
        return prepared_growth;
      }
      const auto restarted = impl_->transaction.RestartFrameWithCapacity(growth);
      if (!restarted.ok()) {
        impl_->backend.discard_growth(impl_->backend.context);
        return restarted;
      }
      impl_->backend.commit_growth(impl_->backend.context);
      impl_->reservation_bytes = *reservation_bytes + inlet_bytes + cavity_bytes;
      impl_->workspace = std::move(*workspace);
      impl_->staging = std::move(*staging);
      impl_->options.capacities = growth;
      ++impl_->diagnostics.capacity_replays;
      if (observer.observe != nullptr) {
        STEP_TRY(observer.observe(observer.context, committed(), diagnostics(), true));
      }
    }
  }
  return FinishFrame();
}
#undef STEP_TRY
}  // namespace fluidity
