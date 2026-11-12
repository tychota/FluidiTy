#include "fluidity/host/liquid_substep_3d.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numbers>
#include <numeric>
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
    std::size_t workspace_bytes,
    std::size_t staging_bytes
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
  STEP_TRY(add(workspace_bytes, 1));
  STEP_TRY(add(staging_bytes, 1));
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

bool CanStageRegularCellTransport(const LiquidSceneInputs3D& scene, const AxisAlignedLiquidTank3D* tank) {
  return tank && tank->piston_velocity_x == 0 && std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field)
      && scene.sources.empty() && scene.target_divergence == 0 && std::isinf(scene.bulk_modulus)
      && scene.surface_tension == 0;
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
}  // namespace
struct LiquidSubstepScheduler3D::Impl {
  LiquidFrameTransaction3D transaction;
  LiquidStepWorkspace3D workspace;
  liquid_step_internal::Staging staging;
  LiquidRecords3D<LiquidEmissionSource3D> sources;
  LiquidSceneInputs3D scene;
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
      LiquidWorkPhaseObserver3D work_phase,
      RigidBody3D* piston_load,
      bool dynamic_piston,
      bool coupled_piston,
      const LiquidPistonReactionUpdate3D* update_with_reaction
  ) {
    auto& boundary = workspace.boundary();
    const double old_time = transaction.committed().clock.time;
    const double new_time = old_time + step_duration;
    auto obstacle = liquid_step_internal::ObstacleAtTime(scene.obstacle, old_time);
    if (dynamic_piston) {
      auto& tank = std::get<AxisAlignedLiquidTank3D>(obstacle);
      tank.start_low_x = piston_load->center.x;
      tank.piston_velocity_x = piston_load->linear_velocity.x;
    }
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
    STEP_TRY(liquid_step_internal::PrepareField(scene.field, obstacle, trial, staging, backend));
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
    STEP_TRY(staging.pressure_fields.Resize(frozen->cells.size()));
    STEP_TRY(staging.incident_velocities.Resize(frozen->faces.size()));
    for (std::size_t index = 0; index < frozen->cells.size(); ++index) {
      const auto cells = trial.cells.values();
      const auto key = frozen->cells[index].cell_index;
      const auto found = std::lower_bound(cells.begin(), cells.end(), key, [](const auto& cell, auto value) {
        return CellKey(cell.cell_index) < CellKey(value);
      });
      if (found == cells.end() || CellKey(found->cell_index) != CellKey(key)) {
        return Error(absl::StatusCode::kInternal);
      }
      staging.pressure_fields[index] = {scene.target_divergence, scene.bulk_modulus, found->previous_pressure};
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
    std::optional<WettedPistonInputs3D> piston_inputs;
    std::optional<WettedPistonView3D> piston_wetted;
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
      piston_inputs.emplace(wetted_piston_inputs);
      piston_wetted = *wetted;
    }
    auto links = backend.links(
        backend.context,
        *frozen,
        {
            .cells = staging.pressure_fields.values(),
            .predicted_face_velocities = staging.incident_velocities.values(),
            .ambient_pressure = scene.ambient_pressure,
            .wetted_piston_face_fluxes = wetted_piston_face_fluxes,
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
    std::optional<double> coupled_velocity;
    if (coupled_piston && links->cells.empty()) {
      return Error(absl::StatusCode::kFailedPrecondition);
    }
    if (!links->cells.empty()) {
      auto pressure_problem = *links;
      pressure_problem.seed_uniform_dirichlet_rest_pressure = exact_curvature.has_value();
      auto projection = backend.project(
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
      if (coupled_piston) {
        auto& scratch = *staging.pressure_face_scratch;
        STEP_TRY(scratch.Resize(links->faces.size()));
        const auto coupled_faces = scratch.values();
        std::ranges::copy(links->faces, coupled_faces.begin());
        pressure_problem.faces = coupled_faces;

        // Receiver area is aggregated once in the wetted geometry. Attached
        // physical patches contribute their area through the same receiver.
        const auto set_wall_speed = [&](double speed) -> absl::Status {
          for (std::size_t face_index = 0; face_index < coupled_faces.size(); ++face_index) {
            if (piston_wetted->face_fluxes[face_index].has_value()) {
              auto& face = coupled_faces[face_index];
              face.wall_normal_velocity = speed;
              face.prescribed_wall_flux = 0.0;
            }
          }
          for (const auto& patch : piston_wetted->patches) {
            if (patch.receiver_area.upper > 0.0) {
              if (!patch.receiver_face_ordinal || *patch.receiver_face_ordinal >= coupled_faces.size()
                  || !piston_wetted->face_fluxes[*patch.receiver_face_ordinal].has_value()) {
                return Error(absl::StatusCode::kFailedPrecondition);
              }
              const double area = std::midpoint(patch.receiver_area.lower, patch.receiver_area.upper);
              coupled_faces[*patch.receiver_face_ordinal].prescribed_wall_flux = area * speed;
            }
          }
          return absl::OkStatus();
        };
        auto pressure_only_body = *piston_load;
        pressure_only_body.force = {};
        pressure_only_body.torque = {};
        const auto load_at = [&](std::span<const double> pressures) {
          return AccumulatePlanarWettedPistonPressure3D(
              *piston_inputs,
              *piston_wetted,
              pressures,
              scene.ambient_pressure,
              pressure_only_body
          );
        };
        const auto initial_load = load_at(projection->cell_pressures);
        if (!initial_load.ok()) {
          return initial_load.status();
        }
        const double initial_speed = piston_load->linear_velocity.x;
        const double speed_increment = std::copysign(0.125 * grid.geometry().cell_width / step_duration, initial_speed);
        const double probe_speed = initial_speed + speed_increment;
        const double probe_delta = probe_speed - initial_speed;
        if (!std::isfinite(probe_speed) || !std::isfinite(probe_delta) || probe_delta == 0.0) {
          return Error(absl::StatusCode::kOutOfRange);
        }
        STEP_TRY(set_wall_speed(probe_speed));
        projection = backend.project(
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
        const auto probe_load = load_at(projection->cell_pressures);
        if (!probe_load.ok()) {
          return probe_load.status();
        }

        // Frozen linear pressure response gives the scalar Schur complement
        // for the axial body degree of freedom, regardless of pressure-cell count.
        const double added_mass = -step_duration * (probe_load->force.x - initial_load->force.x) / probe_delta;
        const double effective_mass = piston_load->mass + added_mass;
        const double total_force = piston_load->force.x + initial_load->force.x;
        const double speed = std::fma(step_duration, total_force / effective_mass, initial_speed);
        if (added_mass < 0.0 || !std::isfinite(speed) || !std::isfinite(effective_mass)) {
          return Error(absl::StatusCode::kOutOfRange);
        }
        auto& tank = std::get<AxisAlignedLiquidTank3D>(obstacle);
        tank.piston_velocity_x = speed;
        STEP_TRY(boundary_internal::ArithmeticStatus(boundary_internal::CheckTankStep(tank, step_duration), true));
        new_obstacle = liquid_step_internal::ObstacleAtTime(obstacle, step_duration);
        accepted_measurement.piston_pose->step_end_low_x_m =
            std::get<AxisAlignedLiquidTank3D>(new_obstacle).start_low_x;
        STEP_TRY(set_wall_speed(speed));
        projection = backend.project(
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
        coupled_velocity = speed;
      }
      if (piston_load != nullptr) {
        const auto loaded = AccumulatePlanarWettedPistonPressure3D(
            *piston_inputs,
            *piston_wetted,
            projection->cell_pressures,
            scene.ambient_pressure,
            *piston_load
        );
        if (!loaded.ok()) {
          return loaded.status();
        }
        *piston_load = *loaded;
      }
      diagnostics.pressure_diagnostics = projection->diagnostics;
      accepted_measurement.pressure = projection->diagnostics;
      accepted_measurement.pressure_rhs_l2_m3_per_s = projection->frozen_measurement.integrated_rhs_l2_m3_per_s;
      accepted_measurement.step_start_pressure_cell_volume_m3 = projection->frozen_measurement.pressure_cell_volume_m3;
      if (accepted_measurement.piston_pose.has_value()) {
        accepted_measurement.step_start_inward_piston_flux_m3_per_s =
            projection->frozen_measurement.inward_prescribed_wall_flux_m3_per_s;
      }
      for (std::size_t index = 0; index < frozen->cells.size(); ++index) {
        trial.cells[index] = {
            frozen->cells[index].cell_index,
            LiquidCellOccupancy3D::kLiquid,
            projection->cell_pressures[index],
        };
      }
      for (std::size_t index = 0; index < frozen->faces.size(); ++index) {
        const auto slot = frozen->faces[index].face_slot;
        staging.new_velocities[slot] = projection->corrected_face_velocities[index];
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
    double piston_contact_impulse = 0.0;
    if (dynamic_piston && trial.particles.size() != 0) {
      const auto impulse = update_with_reaction == nullptr
          ? UpdateCpuLiquidParticlesWithPistonReaction3D(particle_update_inputs, trial.particles.values())
          : (*update_with_reaction)(particle_update_inputs, trial.particles.values());
      if (!impulse.ok()) {
        return impulse.status();
      }
      piston_contact_impulse = *impulse;
    } else {
      STEP_TRY(backend.update_particles(backend.context, particle_update_inputs, trial.particles.values()));
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kWhitewater, step_duration));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kValidate, step_duration));
    const double wall_displacement_bound =
        step_duration * liquid_step_internal::MaximumComponentMagnitude(liquid_step_internal::WallVelocity(obstacle));
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
    if (dynamic_piston) {
      const double axial_acceleration = piston_load->force.x / piston_load->mass;
      const double next_velocity = std::fma(step_duration, axial_acceleration, piston_load->linear_velocity.x)
          + piston_contact_impulse / piston_load->mass;
      if (!std::isfinite(next_velocity)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      piston_load->center.x = std::get<AxisAlignedLiquidTank3D>(new_obstacle).start_low_x;
      if (coupled_velocity
          && std::abs(next_velocity - *coupled_velocity) > 1e-8 * std::max(1.0, std::abs(next_velocity))) {
        return Error(absl::StatusCode::kFailedPrecondition);
      }
      piston_load->linear_velocity.x = coupled_velocity.value_or(next_velocity);
      piston_load->force = {};
      piston_load->torque = {};
    }
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
  const auto bytes = QueryLiquidWorkspaceBytes3D(options.capacities);
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto staging = liquid_step_internal::Staging::Create(
      options.capacities,
      CanStageRegularCellTransport(scene, tank),
      needs_wetted_piston,
      needs_isolated_drop
  );
  if (!staging.ok()) {
    return staging.status();
  }
  const auto capacities = options.capacities;
  const auto reservation_bytes =
      ReservationBytes(capacities, scene.sources.size(), bytes->total_bytes, staging->reserved_bytes);
  if (!reservation_bytes.ok()) {
    return reservation_bytes.status();
  }
  auto transaction = LiquidFrameTransaction3D::Create(initial, capacities);
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
  std::unique_ptr<Impl> impl(new (std::nothrow)
                                 Impl(std::move(*transaction), std::move(*workspace), std::move(*staging)));
  if (!impl) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  impl->reservation_bytes = *reservation_bytes;
  impl->sources = std::move(*sources);
  impl->scene = scene;
  impl->scene.sources = impl->sources.values();
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
  STEP_TRY(backend.initialize(backend.context, initial));
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
  return AttemptImpl(step_duration, work_phase, nullptr);
}

absl::StatusOr<LiquidPlanarPistonAttempt3D>
LiquidSubstepScheduler3D::AttemptWithPlanarPistonPressureLoad(double step_duration, RigidBody3D piston) {
  if (!impl_->staging.wetted_piston.has_value()
      || !std::holds_alternative<LiquidHorizontalPlaneSurface3D>(impl_->scene.field)) {
    return absl::FailedPreconditionError("piston load requires a moving planar piston scene");
  }
  const auto validation = ValidateRigidBody3D(piston);
  if (!validation.ok()) {
    return validation;
  }
  const auto outcome = AttemptImpl(step_duration, {}, &piston);
  if (!outcome.ok()) {
    return outcome.status();
  }
  return LiquidPlanarPistonAttempt3D{
      .outcome = *outcome,
      .loaded_piston = outcome->disposition == LiquidAttemptDisposition3D::kAccepted
          ? std::optional<RigidBody3D>{piston}
          : std::nullopt,
  };
}

absl::StatusOr<LiquidPlanarPistonAttempt3D>
LiquidSubstepScheduler3D::AttemptWithDynamicPlanarPiston(double step_duration, RigidBody3D piston) {
  return AttemptDynamicPistonImpl(step_duration, piston, false);
}

absl::StatusOr<LiquidPlanarPistonAttempt3D> LiquidSubstepScheduler3D::AttemptWithDynamicPlanarPiston(
    double step_duration,
    RigidBody3D piston,
    LiquidPistonReactionUpdate3D update_with_reaction
) {
  return AttemptDynamicPistonImpl(step_duration, piston, false, &update_with_reaction);
}

absl::StatusOr<LiquidPlanarPistonAttempt3D>
LiquidSubstepScheduler3D::AttemptWithCoupledPlanarPiston(double step_duration, RigidBody3D piston) {
  if (!committed().particles.values().empty() || std::isfinite(impl_->scene.bulk_modulus)
      || impl_->scene.surface_tension != 0.0) {
    return absl::FailedPreconditionError("coupled piston requires a particle-free incompressible planar scene");
  }
  return AttemptDynamicPistonImpl(step_duration, piston, true);
}

absl::StatusOr<LiquidPlanarPistonAttempt3D> LiquidSubstepScheduler3D::AttemptDynamicPistonImpl(
    double step_duration,
    RigidBody3D piston,
    bool coupled,
    const LiquidPistonReactionUpdate3D* update_with_reaction
) {
  if (!impl_->staging.wetted_piston.has_value()
      || !std::holds_alternative<LiquidHorizontalPlaneSurface3D>(impl_->scene.field) || !impl_->scene.sources.empty()
      || impl_->ocean_boundary) {
    return absl::FailedPreconditionError("dynamic piston requires a moving planar piston scene without sources");
  }
  if (update_with_reaction == nullptr && !committed().particles.values().empty()
      && impl_->backend.update_particles != CpuLiquidSubstepBackend3D().update_particles) {
    return absl::FailedPreconditionError("dynamic piston particles require the CPU contact update");
  }
  const auto validation = ValidateRigidBody3D(piston);
  if (!validation.ok()) {
    return validation;
  }
  if (piston.linear_velocity.y != 0.0 || piston.linear_velocity.z != 0.0 || piston.angular_velocity.x != 0.0
      || piston.angular_velocity.y != 0.0 || piston.angular_velocity.z != 0.0) {
    return absl::InvalidArgumentError("dynamic piston requires axial motion");
  }
  const auto outcome = AttemptImpl(step_duration, {}, &piston, true, coupled, update_with_reaction);
  if (!outcome.ok()) {
    return outcome.status();
  }
  return LiquidPlanarPistonAttempt3D{
      .outcome = *outcome,
      .loaded_piston = outcome->disposition == LiquidAttemptDisposition3D::kAccepted
          ? std::optional<RigidBody3D>{piston}
          : std::nullopt,
  };
}

absl::StatusOr<LiquidAttemptOutcome3D> LiquidSubstepScheduler3D::AttemptImpl(
    double step_duration,
    LiquidWorkPhaseObserver3D work_phase,
    RigidBody3D* piston_load,
    bool dynamic_piston,
    bool coupled_piston,
    const LiquidPistonReactionUpdate3D* update_with_reaction
) {
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
  auto status = impl.Execute(
      step_duration,
      **trial,
      accepted_measurement,
      work_phase,
      piston_load,
      dynamic_piston,
      coupled_piston,
      update_with_reaction
  );
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
          impl_->staging.isolated_drop_restricted_gathers.has_value()
      );
      if (!staging.ok()) {
        return staging.status();
      }
      auto workspace = LiquidStepWorkspace3D::Create(growth, committed().geometry);
      if (!workspace.ok()) {
        return workspace.status();
      }
      const auto reservation_bytes =
          ReservationBytes(growth, impl_->sources.size(), workspace->usage().reserved_bytes, staging->reserved_bytes);
      if (!reservation_bytes.ok()) {
        return reservation_bytes.status();
      }
      const auto inlet_bytes = impl_->ocean_inlet_particles ? impl_->ocean_inlet_particles->reserved_bytes() : 0;
      if (inlet_bytes > SIZE_MAX - *reservation_bytes) {
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
      impl_->reservation_bytes = *reservation_bytes + inlet_bytes;
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
