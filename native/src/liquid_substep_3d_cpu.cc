#include <algorithm>
#include <cmath>

#include "liquid_boundary_backend.h"
#include "liquid_substep_3d_internal.h"
namespace fluidity {
namespace {
absl::Status ApplyForces(
    void*,
    SparseMacGridView3D grid,
    std::span<const double> old_velocities,
    Vec3 acceleration,
    double step_duration,
    std::span<double> predicted
) {
  if (old_velocities.size() != grid.face_count() || predicted.size() != grid.face_count()) {
    return liquid_step_internal::Error(absl::StatusCode::kInvalidArgument);
  }
  for (std::size_t index = 0; index < predicted.size(); ++index) {
    const auto face = grid.FaceIndex(index);
    if (!face.ok()) {
      return face.status();
    }
    predicted[index] =
        old_velocities[index] + step_duration * liquid_step_internal::Component(acceleration, face->axis);
    if (!std::isfinite(predicted[index])) {
      return liquid_step_internal::Error(absl::StatusCode::kOutOfRange);
    }
  }
  return absl::OkStatus();
}
absl::Status
UpdateParticles(void*, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
  if (inputs.gathered.size() != particles.size() || inputs.liquid_distances.size() != particles.size()) {
    return liquid_step_internal::Error(absl::StatusCode::kInvalidArgument);
  }
  for (std::size_t index = 0; index < particles.size(); ++index) {
    auto& particle = particles[index];
    const auto& gather = inputs.gathered[index];
    const double divergence =
        gather.velocity_gradient.x_row.x + gather.velocity_gradient.y_row.y + gather.velocity_gradient.z_row.z;
    const bool fallback = gather.old_pic_fallback || gather.new_pic_fallback || gather.low_mass_fallback;
    const double alpha = particle.pic_initialized && !fallback ? 0.99 * inputs.particle_energy_blend : 0;
    const double solid_distance = liquid_step_internal::SolidDistance(inputs.obstacle, particle.world_position);
    if (std::isnan(solid_distance)) {
      return liquid_step_internal::Error(absl::StatusCode::kOutOfRange);
    }
    const bool surface_expansion = divergence > 0 && inputs.liquid_distances[index] <= 0
        && inputs.liquid_distances[index] >= -inputs.cell_width && solid_distance > inputs.cell_width;
    const double beta = surface_expansion ? inputs.particle_energy_blend : 0;
    const auto stored = liquid_step_internal::Add(
        gather.new_pic_velocity,
        liquid_step_internal::Scale(gather.particle_minus_old_pic_velocity, alpha)
    );
    const auto advecting = liquid_step_internal::Add(
        gather.new_pic_velocity,
        liquid_step_internal::Scale(gather.particle_minus_old_pic_velocity, beta * alpha)
    );
    const auto contact = ResolveLiquidWallContactBounded3D({
        .start_world_position = particle.world_position,
        .stored_world_velocity = stored,
        .advecting_world_velocity = advecting,
        .obstacle = inputs.obstacle,
        .step_duration = inputs.step_duration,
        .friction_coefficient = inputs.friction_coefficient,
        .penetration_tolerance = inputs.penetration_tolerance,
    });
    if (!contact.ok()) {
      return contact.status();
    }
    const Vec3 displacement{
        contact->world_position.x - particle.world_position.x,
        contact->world_position.y - particle.world_position.y,
        contact->world_position.z - particle.world_position.z,
    };
    if (liquid_step_internal::MaximumComponentMagnitude(displacement) > 0.5 * inputs.cell_width) {
      return liquid_step_internal::Error(absl::StatusCode::kAborted);
    }
    const double volume_ratio = particle.volume_ratio * std::exp(inputs.step_duration * divergence);
    if (!std::isfinite(volume_ratio) || volume_ratio <= 0) {
      return liquid_step_internal::Error(absl::StatusCode::kAborted);
    }
    particle.world_position = contact->world_position;
    particle.world_velocity = contact->stored_world_velocity;
    particle.volume_ratio = volume_ratio;
    particle.velocity_slopes = gather.fitted_velocity_slopes;
    particle.affine_enabled = gather.affine_enabled;
    particle.pic_initialized = true;
  }
  return absl::OkStatus();
}
}  // namespace
LiquidSubstepBackend3D CpuLiquidSubstepBackend3D() {
  return {
      .context = nullptr,
      .sample_field =
          [](void*, const LiquidSceneField3D& field, std::span<const LiquidParticleState3D> particles, Vec3 position
          ) -> absl::StatusOr<double> {
        const double distance = liquid_step_internal::FieldDistance(field, particles, position);
        if (!std::isfinite(distance)) {
          return liquid_step_internal::Error(absl::StatusCode::kOutOfRange);
        }
        return distance;
      },
      .sample_faces =
          [](void*,
             SparseMacGridView3D grid,
             const LiquidObstacle3D& obstacle,
             std::int64_t side,
             std::span<LiquidBoundaryFaceTag3D> tags,
             std::span<LiquidPrescribedWallSample3D> walls) {
            const auto count = grid.face_count();
            if ((!tags.empty() && tags.size() != count) || (!walls.empty() && walls.size() != count)) {
              return liquid_step_internal::Error(absl::StatusCode::kInvalidArgument);
            }
            if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
              if (side != 0) {
                return liquid_step_internal::Error(absl::StatusCode::kInvalidArgument);
              }
              const auto pose = boundary_internal::ArithmeticStatus(boundary_internal::CheckTankStep(*tank, 0), true);
              if (!pose.ok()) {
                return pose;
              }
            }
            if (!tags.empty()) {
              const auto status = liquid_step_internal::FillFaceTags(grid, obstacle, tags);
              if (!status.ok()) {
                return status;
              }
            }
            return liquid_step_internal::FillWalls(grid, obstacle, side, walls);
          },
      .initialize =
          [](void*, const LiquidState3D&) {
            return absl::OkStatus();
          },
      .prepare_publication =
          [](void*, const LiquidState3D&, LiquidClock3D) {
            return absl::OkStatus();
          },
      .commit_publication =
          [](void*) noexcept {
          },
      .discard_publication =
          [](void*) noexcept {
          },
      .prepare_growth =
          [](void*, LiquidStateCapacities3D, const LiquidState3D&) {
            return absl::OkStatus();
          },
      .commit_growth =
          [](void*) noexcept {
          },
      .discard_growth =
          [](void*) noexcept {
          },
      .topology =
          [](void*,
             std::span<const LiquidParticleState3D> particles,
             std::span<const CellIndex3D> pressure_candidates,
             const LiquidState3D& committed,
             LiquidStepWorkspace3D& workspace) {
            return workspace.PrepareTrialTopology(particles, pressure_candidates, &committed);
          },
      .deposit =
          [](void*, const LiquidBoundaryDepositViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
            return DepositCompatibleMacQuanta3D(inputs, workspace);
          },
      .normalize =
          [](void*,
             std::span<const MacFaceDepositQuanta3D> quanta,
             MacDepositQuantization3D quantization,
             double minimum_face_mass,
             LiquidBoundaryWorkspace3D& workspace) {
            return NormalizeLiquidMacFaces3D(quanta, quantization, minimum_face_mass, workspace);
          },
      .predict_missing_face = [](void*,
                                 const LiquidMissingFacePredictionInputs3D& inputs,
                                 LiquidBoundaryWorkspace3D& workspace) -> absl::StatusOr<double> {
        const auto continued_old_velocity =
            workspace.ContinueFace(inputs.grid, inputs.old_history, inputs.face_slot, inputs.side_tag);
        if (!continued_old_velocity.ok()) {
          return continued_old_velocity.status();
        }
        const auto face = inputs.grid.FaceIndex(inputs.face_slot);
        if (!face.ok()) {
          return face.status();
        }
        const double predicted_face_velocity = *continued_old_velocity
            + inputs.step_duration * liquid_step_internal::Component(inputs.world_acceleration, face->axis);
        if (!std::isfinite(predicted_face_velocity)) {
          return liquid_step_internal::Error(absl::StatusCode::kOutOfRange);
        }
        return predicted_face_velocity;
      },
      .forces = ApplyForces,
      .freeze =
          [](void*, const LiquidBoundaryGeometryViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
            return FreezeLiquidBoundaryGeometry3D(inputs, workspace);
          },
      .links =
          [](void*,
             const FrozenLiquidBoundaryGeometryView3D& geometry,
             const LiquidPressureFields3D& fields,
             LiquidBoundaryWorkspace3D& workspace) {
            return BuildLiquidPressureLinks3D(geometry, fields, workspace);
          },
      .project =
          [](void*,
             SparseMacGridView3D grid,
             double step_duration,
             const LiquidPressureProblemView3D& problem,
             MacPressureSolverOptions3D options,
             MacPressureWorkspace3D& workspace) {
            return ProjectMacPressure3D(grid, step_duration, problem, options, workspace);
          },
      .gather =
          [](void*, const LiquidBoundaryGatherViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
            return GatherLiquidBoundaryParticles3D(inputs, workspace);
          },
      .update_particles = UpdateParticles,
  };
}
}  // namespace fluidity
