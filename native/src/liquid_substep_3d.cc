#include "fluidity/host/liquid_substep_3d.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <tuple>

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
}  // namespace
struct LiquidSubstepScheduler3D::Impl {
  LiquidFrameTransaction3D transaction;
  LiquidStepWorkspace3D workspace;
  liquid_step_internal::Staging staging;
  LiquidRecords3D<LiquidEmissionSource3D> sources;
  LiquidSceneInputs3D scene;
  LiquidStepOptions3D options;
  LiquidSubstepBackend3D backend;
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
  absl::Status
  Execute(double step_duration, LiquidState3D& trial, LiquidAcceptedStepMeasurement3D& accepted_measurement) {
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
    STEP_TRY(Enter(LiquidPhysicalStage3D::kBodyContact, step_duration));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kInterface, step_duration));
    const LiquidBoundaryGeometryViewInputs3D geometry_inputs{
        .mac_grid = grid,
        .cell_indices = staging.candidates.values(),
        .center_distances = staging.centers.values(),
        .corner_distances = staging.corners.values(),
        .obstacle = obstacle,
        .step_duration = step_duration,
        .liquid_density = scene.liquid_density,
        .surface_tension = scene.surface_tension,
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
      accepted_measurement.step_start_capillary_bound_seconds = frozen->capillary_step_bound;
    }
    if (const auto* tank_at_step_start = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
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
      }
      staging.incident_velocities[index] = staging.new_velocities[slot];
    }
    const auto links = backend.links(
        backend.context,
        *frozen,
        {staging.pressure_fields.values(), staging.incident_velocities.values(), scene.ambient_pressure},
        boundary
    );
    if (!links.ok()) {
      return links.status();
    }
    diagnostics.lower_theta_regularization_count = links->lower_theta_regularization_count;
    STEP_TRY(Enter(LiquidPhysicalStage3D::kViscosity, step_duration));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kPressure, step_duration));
    STEP_TRY(trial.cells.Resize(frozen->cells.size()));
    // An empty, successfully frozen geometry has no constraint rows. All other
    // stages still execute, including transfer/forces and backend particle update.
    if (!links->cells.empty()) {
      const auto projection = backend.project(
          backend.context,
          grid,
          step_duration,
          *links,
          options.pressure_solver_options,
          workspace.pressure()
      );
      if (!projection.ok()) {
        return projection.status();
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
    STEP_TRY(Enter(LiquidPhysicalStage3D::kNewSnapshot, step_duration));
    STEP_TRY(backend.sample_faces(backend.context, grid, new_obstacle, 0, staging.new_tags.values(), {}));
    for (std::size_t index = 0; index < trial.faces.size(); ++index) {
      trial.faces[index].carrier_velocity = staging.new_velocities[index];
      trial.faces[index].validity = staging.validity[index];
      trial.faces[index].side_tag = staging.new_tags[index].side_tag;
      trial.faces[index].structurally_eligible = staging.new_tags[index].structurally_eligible;
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
    // A thin slab can share original stencil nodes across sides. Each gather
    // receives distinct old/new prescribed wall views for THIS particle's side.
    // Numerical donor order remains canonical owned-slot order, independent of
    // particle groups. Copy each result before reusing the bounded gather bank.
    for (std::size_t index = 0; index < trial.particles.size(); ++index) {
      const auto side = trial.particles[index].side_tag;
      STEP_TRY(backend.sample_faces(backend.context, grid, obstacle, side, {}, staging.old_walls.values()));
      STEP_TRY(backend.sample_faces(backend.context, grid, new_obstacle, side, {}, staging.new_walls.values()));
      const auto gather = backend.gather(
          backend.context,
          {
              .mac_grid = grid,
              .particles = staging.particles.values().subspan(index, 1),
              .old_time = old_time,
              .new_time = new_time,
              .old_history = old_history,
              .new_history = new_history,
          },
          boundary
      );
      if (!gather.ok()) {
        return gather.status();
      }
      staging.gathered[index] = gather->particles[0];
      diagnostics.old_pic_fallback_count += gather->old_pic_fallback_particle_count;
      diagnostics.new_pic_fallback_count += gather->new_pic_fallback_particle_count;
      const auto distance = backend.sample_field(
          backend.context,
          scene.field,
          trial.particles.values(),
          trial.particles[index].world_position
      );
      if (!distance.ok()) {
        return distance.status();
      }
      if (!std::isfinite(*distance)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      staging.particle_distances[index] = *distance;
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kParticleUpdate, step_duration));
    const LiquidParticleUpdateInputs3D particle_update_inputs{
        .gathered = staging.gathered.values(),
        .liquid_distances = staging.particle_distances.values(),
        .obstacle = obstacle,
        .cell_width = trial.geometry.cell_width,
        .step_duration = step_duration,
        .particle_energy_blend = options.particle_energy_blend,
        .friction_coefficient = scene.friction_coefficient,
        .penetration_tolerance = scene.penetration_tolerance,
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
  if (!Complete(backend) || (needs_lattice_plane ? !backend.sample_lattice_plane : !backend.sample_field_batch)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  STEP_TRY(liquid_step_internal::ValidateScene(initial, scene, options));
  const auto bytes = QueryLiquidWorkspaceBytes3D(options.capacities);
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto staging = liquid_step_internal::Staging::Create(options.capacities);
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
absl::StatusOr<LiquidAttemptOutcome3D> LiquidSubstepScheduler3D::Attempt(double step_duration) {
  auto& impl = *impl_;
  auto& diagnostics = impl.diagnostics;
  if (!impl.frame_active || !std::isfinite(step_duration) || step_duration <= 0
      || step_duration > impl.target_time - committed().clock.time) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
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
  auto status = impl.Execute(step_duration, **trial, accepted_measurement);
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
    const auto outcome = Attempt(*duration);
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
      auto staging = liquid_step_internal::Staging::Create(growth);
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
      impl_->reservation_bytes = *reservation_bytes;
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
