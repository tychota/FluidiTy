#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>

#include "liquid_boundary_backend.h"
#include "liquid_substep_3d_internal.h"
namespace fluidity {
namespace liquid_step_internal {
absl::Status ValidateParticleUpdateRecords(
    const LiquidParticleUpdateInputs3D& inputs,
    std::span<const LiquidParticleState3D> particles
) {
  constexpr std::array rows{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto& particle = particles[index];
    const auto& gather = inputs.gathered[index];
    if (!Finite(particle.world_position) || !std::isfinite(particle.volume_ratio) || particle.volume_ratio <= 0
        || !std::isfinite(inputs.liquid_distances[index]) || !Finite(gather.new_pic_velocity)
        || !Finite(gather.particle_minus_old_pic_velocity)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    for (const auto row : rows) {
      if (!Finite(gather.velocity_gradient.*row) || !Finite(gather.fitted_velocity_slopes.*row)) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace liquid_step_internal
namespace {
using liquid_step_internal::Error;
using liquid_step_internal::Finite;

bool Overlap(std::span<const std::byte> first, std::span<const std::byte> second) {
  if (first.empty() || second.empty()) {
    return false;
  }
  const std::less<const std::byte*> before;
  return before(first.data(), second.data() + second.size()) && before(second.data(), first.data() + first.size());
}

template <typename Sample>
absl::Status CheckBatch(std::span<const Sample> samples, std::span<double> distances) {
  if (samples.size() != distances.size() || Overlap(std::as_bytes(samples), std::as_bytes(distances))) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (samples.size() > kLiquidFieldBatchCapacity3D) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  return absl::OkStatus();
}

// Keep the entire output private until every sample has been checked.
template <typename Sample, typename Evaluate>
absl::Status PublishDistances(std::span<const Sample> samples, std::span<double> distances, Evaluate evaluate) {
  std::array<double, kLiquidFieldBatchCapacity3D> staged_distances;
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const auto distance = evaluate(samples[index]);
    if (!distance.ok()) {
      return distance.status();
    }
    staged_distances[index] = *distance;
  }
  std::ranges::copy_n(staged_distances.begin(), distances.size(), distances.begin());
  return absl::OkStatus();
}

absl::Status SampleWorldBatch(
    void*,
    const LiquidSceneField3D& field,
    std::span<const LiquidParticleState3D> particles,
    std::span<const Vec3> positions,
    std::span<double> distances
) {
  const auto extent = CheckBatch(positions, distances);
  if (!extent.ok()) {
    return extent;
  }
  if (Overlap(std::as_bytes(distances), std::as_bytes(particles))
      || Overlap(std::as_bytes(distances), std::as_bytes(std::span(&field, 1)))) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&field)) {
    if (!Finite(sphere->world_center) || !std::isfinite(sphere->radius) || sphere->radius <= 0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  } else if (const auto* particle_union = std::get_if<LiquidParticleSphereUnion3D>(&field)) {
    if (!std::isfinite(particle_union->radius) || particle_union->radius <= 0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  } else {
    const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&field);
    if (!plane || !std::isfinite(plane->world_surface_height_y)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  for (const auto& particle : particles) {
    if (!Finite(particle.world_position)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  if (!std::ranges::all_of(positions, Finite)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  return PublishDistances(positions, distances, [&](Vec3 position) -> absl::StatusOr<double> {
    const double distance = liquid_step_internal::FieldDistance(field, particles, position);
    if (!std::isfinite(distance)) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    return distance;
  });
}

absl::Status SampleLatticePlaneBatch(
    void*,
    const LiquidHorizontalPlaneSurface3D& plane,
    UniformMacGeometry3D geometry,
    std::span<const double> lattice_ordinates_y,
    std::span<double> distances
) {
  const auto extent = CheckBatch(lattice_ordinates_y, distances);
  if (!extent.ok()) {
    return extent;
  }
  if (Overlap(std::as_bytes(distances), std::as_bytes(std::span(&plane, 1)))
      || !std::isfinite(plane.world_surface_height_y) || !std::isfinite(geometry.world_origin.y)
      || !std::isfinite(geometry.cell_width) || geometry.cell_width <= 0) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (!std::ranges::all_of(lattice_ordinates_y, [](double ordinate) {
        return std::isfinite(ordinate);
      })) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  return PublishDistances(lattice_ordinates_y, distances, [&](double lattice_y) -> absl::StatusOr<double> {
    const auto distance = boundary_internal::HorizontalPlaneLatticeDistance(
        plane.world_surface_height_y,
        {geometry.world_origin.y, lattice_y, geometry.cell_width}
    );
    if (!distance.ok()) {
      return boundary_internal::ArithmeticStatus(distance.status(), true);
    }
    return *distance;
  });
}

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
UpdateParticle(const LiquidParticleUpdateInputs3D& inputs, std::size_t index, LiquidParticleState3D& particle) {
  const auto& gather = inputs.gathered[index];
  const bool fallback = gather.old_pic_fallback || gather.new_pic_fallback || gather.low_mass_fallback;
  const double alpha = particle.pic_initialized && !fallback ? 0.99 * inputs.particle_energy_blend : 0;
  const auto stored = liquid_step_internal::Add(
      gather.new_pic_velocity,
      liquid_step_internal::Scale(gather.particle_minus_old_pic_velocity, alpha)
  );
  Vec3 advecting;
  double divergence;
  if (!inputs.regular_cell_transport.empty() && inputs.regular_cell_transport[index].active) {
    const auto field =
        internal::EvaluateRegularCellTransport3D(inputs.regular_cell_transport[index], inputs.cell_width);
    if (!field.ok()) {
      return boundary_internal::ArithmeticStatus(field.status(), true);
    }
    advecting = field->advecting_world_velocity;
    divergence = field->divergence;
  } else {
    divergence = gather.velocity_gradient.x_row.x + gather.velocity_gradient.y_row.y + gather.velocity_gradient.z_row.z;
    const double solid_distance = liquid_step_internal::SolidDistance(inputs.obstacle, particle.world_position);
    if (std::isnan(solid_distance)) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    const bool surface_expansion = divergence > 0 && inputs.liquid_distances[index] <= 0
        && inputs.liquid_distances[index] >= -inputs.cell_width && solid_distance > inputs.cell_width;
    const double beta = surface_expansion ? inputs.particle_energy_blend : 0;
    advecting = liquid_step_internal::Add(
        gather.new_pic_velocity,
        liquid_step_internal::Scale(gather.particle_minus_old_pic_velocity, beta * alpha)
    );
  }
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
  return absl::OkStatus();
}

absl::Status
UpdateParticles(void*, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
  if (inputs.gathered.size() != particles.size() || inputs.liquid_distances.size() != particles.size()) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const auto transport = inputs.regular_cell_transport;
  if (!transport.empty()) {
    const auto particle_bytes = std::as_bytes(particles);
    if (transport.size() != particles.size() || Overlap(particle_bytes, std::as_bytes(transport))
        || Overlap(particle_bytes, std::as_bytes(inputs.gathered))
        || Overlap(particle_bytes, std::as_bytes(inputs.liquid_distances))) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    const auto record_status = liquid_step_internal::ValidateParticleUpdateRecords(inputs, particles);
    if (!record_status.ok()) {
      return record_status;
    }
    // Check the same arithmetic in particle order before publishing a sampled
    // update. A late malformed sample must not preempt an earlier retry.
    for (std::size_t index = 0; index < particles.size(); ++index) {
      auto checked = particles[index];
      const auto status = UpdateParticle(inputs, index, checked);
      if (!status.ok()) {
        return status;
      }
    }
  }
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto status = UpdateParticle(inputs, index, particles[index]);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<MacViscousPressureProjection3D> liquid_step_internal::ProjectFullCellViscosity(
    SparseMacGridView3D grid,
    double step_duration,
    const LiquidPressureProblemView3D& problem,
    double dynamic_viscosity,
    std::span<const double> cell_dynamic_viscosity,
    MacPressureSolverOptions3D options,
    ProjectViscousPressure3D project
) {
  const double width = grid.geometry().cell_width;
  const double full_area = width * width;
  const double full_volume = full_area * width;
  std::vector<MacViscosityCell3D> material;
  material.reserve(problem.cells.size());
  if (!cell_dynamic_viscosity.empty() && cell_dynamic_viscosity.size() != problem.cells.size()) {
    return absl::InvalidArgumentError("viscosity coefficients differ from the frozen pressure cells");
  }
  for (std::size_t index = 0; index < problem.cells.size(); ++index) {
    const auto& cell = problem.cells[index];
    // The cut-volume accumulator sums tetrahedra even for full cells.
    constexpr double kFullCellRoundoff = 64 * std::numeric_limits<double>::epsilon();
    if (std::abs(cell.fluid_volume / full_volume - 1.0) > kFullCellRoundoff) {
      return absl::UnimplementedError("viscous substep requires full-cell strain quadratures");
    }
    const double viscosity = cell_dynamic_viscosity.empty() ? dynamic_viscosity : cell_dynamic_viscosity[index];
    material.push_back({.cell_index = cell.cell_index, .dynamic_viscosity = viscosity});
  }
  for (const auto& face : problem.faces) {
    if (face.open_area != 0.0 && face.open_area != full_area) {
      return absl::UnimplementedError("viscous substep requires full-area or sealed faces");
    }
  }
  const auto strain = AssembleFullCellMacStrain3D(grid, material);
  if (!strain.ok()) {
    return strain.status();
  }
  const auto blocks = grid.active_block_indices();
  const auto owning_grid = SparseMacGrid3D::Create(grid.geometry(), blocks, blocks.size());
  if (!owning_grid.ok()) {
    return owning_grid.status();
  }
  return project({*owning_grid, step_duration, problem.cells, problem.faces}, *strain, options);
}

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
      .wetted_piston =
          [](void*, const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace) {
            return ComputeWettedPiston3D(inputs, workspace);
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
      .project_viscous =
          [](void*,
             SparseMacGridView3D grid,
             double step_duration,
             const LiquidPressureProblemView3D& problem,
             double dynamic_viscosity,
             std::span<const double> cell_dynamic_viscosity,
             MacPressureSolverOptions3D options) {
            return liquid_step_internal::ProjectFullCellViscosity(
                grid,
                step_duration,
                problem,
                dynamic_viscosity,
                cell_dynamic_viscosity,
                options,
                ProjectMacViscousPressure3D
            );
          },
      .gather =
          [](void*, const LiquidBoundaryGatherViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
            return GatherLiquidBoundaryParticles3D(inputs, workspace);
          },
      .update_particles = UpdateParticles,
      .sample_field_batch = SampleWorldBatch,
      .sample_lattice_plane = SampleLatticePlaneBatch,
  };
}
}  // namespace fluidity
