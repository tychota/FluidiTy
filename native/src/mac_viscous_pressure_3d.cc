#include "mac_viscous_pressure_3d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "mac_pressure_3d_arithmetic.h"
#include "mac_pressure_3d_multigrid.h"
#include "mac_pressure_3d_workspace_internal.h"
#include "viscosity_velocity_block_3d.h"

namespace fluidity {
namespace {

using pressure_internal::PreparedMacPressureView3D;

using SolveViscousVelocityView3D = absl::FunctionRef<absl::StatusOr<viscosity_internal::VelocityBlockViewSolution3D>(
    const MacViscousVelocityBlockInputs3D&,
    std::span<const double>,
    MacViscousVelocityBlockOptions3D
)>;

absl::StatusOr<viscosity_internal::VelocityBlockScratch3D> PrepareCpuVelocityScratch(
    pressure_internal::MacPressureScratch3D& scratch,
    std::size_t row_count,
    std::size_t face_count
) {
  const auto rows = scratch.Take<MacStrainRow3D>(row_count);
  if (!rows.ok()) {
    return rows.status();
  }
  const auto storage = scratch.Take<double>(pressure_internal::kCoupledVelocityScratchFields3D * face_count);
  if (!storage.ok()) {
    return storage.status();
  }
  return viscosity_internal::VelocityBlockScratch3D{
      .weighted_rows = *rows,
      .diagonal = storage->subspan(0, face_count),
      .velocity = storage->subspan(face_count, face_count),
      .residual = storage->subspan(2 * face_count, face_count),
      .preconditioned = storage->subspan(3 * face_count, face_count),
      .direction = storage->subspan(4 * face_count, face_count),
      .applied = storage->subspan(5 * face_count, face_count),
  };
}

constexpr std::size_t kPressureIterationBudget = 256;

// Remap sparse grid slots to compact incident face ordinals for the velocity solve.
absl::Status RemapStrain(const PreparedMacPressureView3D& system, std::span<MacStrainRow3D> rows) {
  const auto links = system.links();
  for (auto& row : rows) {
    if (row.term_count == 0 || row.term_count > row.terms.size()) {
      return absl::InvalidArgumentError("invalid coupled strain row extent");
    }
    if (!std::isfinite(row.quadrature_weight) || row.quadrature_weight < 0.0 || !std::isfinite(row.prescribed_strain)) {
      return absl::InvalidArgumentError("invalid coupled strain quadrature");
    }
    for (auto& term : std::span(row.terms).first(row.term_count)) {
      const auto found =
          std::ranges::lower_bound(links, term.face_slot, {}, &pressure_internal::FrozenMacPressureLink3D::face_slot);
      if (found == links.end() || found->face_slot != term.face_slot || !std::isfinite(term.coefficient)) {
        return absl::InvalidArgumentError("invalid coupled strain face or coefficient");
      }
      term.face_slot = static_cast<std::size_t>(found - links.begin());
    }
  }
  return absl::OkStatus();
}

// Apply D and C in integrated flux units. Add the wall and source terms only
// for the physical candidate, never for a pressure search direction.
absl::Status AccumulateConstraint(
    const PreparedMacPressureView3D& system,
    std::span<const double> pressure,
    std::span<const double> velocity,
    bool physical,
    std::span<pressure_internal::arithmetic::CompensatedSum> flux,
    std::span<double> result
) {
  const auto cells = system.cells();
  const auto links = system.links();
  std::ranges::fill(flux, pressure_internal::arithmetic::CompensatedSum{});
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    if (physical) {
      flux[cell].Add(-cells[cell].integrated_target_flux);
    }
    if (cells[cell].compliance > 0.0) {
      const double previous_pressure = physical ? cells[cell].previous_pressure : 0.0;
      const double pressure_difference = pressure[cell] - previous_pressure;
      const double compliance_flux = cells[cell].compliance * pressure_difference;
      if (!pressure_internal::arithmetic::IsRepresentablePressureProduct(
              cells[cell].compliance,
              pressure_difference,
              compliance_flux
          )) {
        return absl::OutOfRangeError("coupled compliance flux is unrepresentable");
      }
      flux[cell].Add(compliance_flux);
    }
  }
  for (std::size_t face = 0; face < links.size(); ++face) {
    const auto& link = links[face];
    const double open_flux = link.open_area * velocity[face];
    if (!pressure_internal::arithmetic::IsRepresentablePressureProduct(link.open_area, velocity[face], open_flux)) {
      return absl::OutOfRangeError("coupled face flux is unrepresentable");
    }
    pressure_internal::arithmetic::ForEachIncidentCell(link, [&](std::size_t cell, double outward_sign) {
      flux[cell].Add(outward_sign * open_flux);
      if (physical) {
        flux[cell].Add(outward_sign * link.prescribed_wall_flux);
      }
    });
  }
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    result[cell] = flux[cell].Total();
    if (!std::isfinite(result[cell])) {
      return absl::OutOfRangeError("coupled cell flux is unrepresentable");
    }
  }
  return absl::OkStatus();
}

double Dot(std::span<const double> left, std::span<const double> right) {
  return std::inner_product(left.begin(), left.end(), right.begin(), 0.0);
}

absl::StatusOr<double> CoupledConstraintNorm(std::span<const double> integrated_fluxes) {
  double norm = 0.0;
  for (const double flux : integrated_fluxes) {
    norm = std::hypot(norm, flux);
  }
  if (!std::isfinite(norm)) {
    return absl::OutOfRangeError("coupled residual norm is unrepresentable");
  }
  return norm;
}

absl::StatusOr<MacViscousPressureProjection3D> SolveCoupled(
    const PreparedMacPressureView3D& system,
    double dt,
    std::span<const MacStrainRow3D> strain_rows,
    std::span<double> face_buffers,
    std::size_t max_iterations,
    std::optional<SolveViscousVelocityView3D> solve_velocity_block,
    pressure_internal::MacPressureScratch3D& scratch
) {
  const auto rows = scratch.Take<MacStrainRow3D>(strain_rows.size());
  if (!rows.ok()) {
    return rows.status();
  }
  std::ranges::copy(strain_rows, rows->begin());
  const auto remap_status = RemapStrain(system, *rows);
  if (!remap_status.ok()) {
    return remap_status;
  }
  const auto links = system.links();
  const auto cells = system.cells();
  const auto wall_normal_velocities = face_buffers.first(links.size());
  const auto inertia = face_buffers.subspan(links.size(), links.size());
  const auto base_momentum = face_buffers.subspan(2 * links.size(), links.size());
  const auto velocity_diagonal = face_buffers.subspan(3 * links.size(), links.size());
  const auto momentum = face_buffers.subspan(4 * links.size(), links.size());
  const auto cell_buffers = scratch.Take<double>(pressure_internal::kCoupledCellScratchFields3D * cells.size());
  if (!cell_buffers.ok()) {
    return cell_buffers.status();
  }
  const auto pressure_diagonal = cell_buffers->first(cells.size());
  const auto zero_pressure = cell_buffers->subspan(cells.size(), cells.size());
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    pressure_diagonal[cell] = cells[cell].compliance;
  }

  for (std::size_t face = 0; face < links.size(); ++face) {
    const auto& link = links[face];
    if (link.open_area == 0.0) {
      continue;
    }
    const double step_area = dt * link.open_area;
    inertia[face] = step_area / link.pressure_to_velocity_scale;
    if (!std::isnormal(inertia[face]) || inertia[face] <= 0.0) {
      return absl::OutOfRangeError("coupled face inertia is unrepresentable");
    }
    double pressure_lift = 0.0;
    if (const auto* jump = std::get_if<InternalMacPressureLink3D>(&link.pressure_link)) {
      pressure_lift = jump->pressure_jump;
    } else if (const auto* exterior = std::get_if<DirichletMacPressureLink3D>(&link.pressure_link)) {
      pressure_lift = link.tail_cell_ordinal ? -exterior->exterior_pressure : exterior->exterior_pressure;
    }
    base_momentum[face] = inertia[face] * link.predicted_normal_velocity + step_area * pressure_lift;
  }

  // Use the diagonal of H in the pressure preconditioner. The inviscid
  // diagonal misses the mobility reduction on the stiff side of an interface.
  std::ranges::copy(inertia, velocity_diagonal.begin());
  for (const auto& row : *rows) {
    for (const auto& term : std::span(row.terms).first(row.term_count)) {
      if (inertia[term.face_slot] > 0.0) {
        velocity_diagonal[term.face_slot] += dt * row.quadrature_weight * term.coefficient * term.coefficient;
      }
    }
  }
  for (std::size_t face = 0; face < links.size(); ++face) {
    if (inertia[face] == 0.0) {
      continue;
    }
    const auto& link = links[face];
    const double effective_conductance = link.conductance * (inertia[face] / velocity_diagonal[face]);
    if (!pressure_internal::arithmetic::IsFinitePositive(effective_conductance)) {
      return absl::OutOfRangeError("coupled preconditioner conductance is unrepresentable");
    }
    pressure_internal::arithmetic::ForEachIncidentCell(link, [&](std::size_t cell, double) {
      pressure_diagonal[cell] += effective_conductance;
    });
  }

  // Eliminate prescribed velocities in E before solving H u = momentum.
  for (const auto& row : *rows) {
    if (row.quadrature_weight == 0.0) {
      continue;
    }
    const auto terms = std::span(row.terms).first(row.term_count);
    double prescribed_strain = row.prescribed_strain;
    for (const auto& term : terms) {
      if (inertia[term.face_slot] == 0.0) {
        prescribed_strain += term.coefficient * wall_normal_velocities[term.face_slot];
      }
    }
    if (prescribed_strain == 0.0) {
      continue;
    }
    for (const auto& term : terms) {
      if (inertia[term.face_slot] > 0.0) {
        base_momentum[term.face_slot] -= dt * row.quadrature_weight * prescribed_strain * term.coefficient;
      }
    }
  }
  if (!std::ranges::all_of(base_momentum, [](double value) {
        return std::isfinite(value);
      })) {
    return absl::OutOfRangeError("coupled base momentum is unrepresentable");
  }

  const auto pinned_buffer = scratch.Take<bool>(cells.size());
  if (!pinned_buffer.ok()) {
    return pinned_buffer.status();
  }
  const auto pinned = *pinned_buffer;
  for (const auto cell : system.gauge_cell_ordinals()) {
    pinned[cell] = true;
  }
  const MacViscousVelocityBlockInputs3D velocity_block{*rows, inertia, dt};
  viscosity_internal::VelocityBlockScratch3D cpu_scratch;
  if (!solve_velocity_block.has_value()) {
    const auto prepared = PrepareCpuVelocityScratch(scratch, rows->size(), links.size());
    if (!prepared.ok()) {
      return prepared.status();
    }
    cpu_scratch = *prepared;
  }
  const auto solve_velocity = [&](std::span<const double> pressure, bool include_base) {
    for (std::size_t face = 0; face < links.size(); ++face) {
      const auto& link = links[face];
      const double tail = link.tail_cell_ordinal ? pressure[*link.tail_cell_ordinal] : 0.0;
      const double head = link.head_cell_ordinal ? pressure[*link.head_cell_ordinal] : 0.0;
      momentum[face] = include_base ? base_momentum[face] : 0.0;
      if (link.open_area > 0.0) {
        momentum[face] += dt * link.open_area * (tail - head);
      }
    }
    if (solve_velocity_block.has_value()) {
      return (*solve_velocity_block)(velocity_block, momentum, {});
    }
    return viscosity_internal::SolveVelocityBlockWithScratch3D(velocity_block, momentum, {}, cpu_scratch);
  };
  auto initial_pressure = pressure_internal::InitialMacPressure3D(system);
  if (!initial_pressure.ok()) {
    return initial_pressure.status();
  }
  auto pressure = std::move(*initial_pressure);
  const bool starts_with_history = std::ranges::any_of(pressure, [](double value) {
    return value != 0.0;
  });
  const auto flux_buffer = scratch.Take<pressure_internal::arithmetic::CompensatedSum>(cells.size());
  if (!flux_buffer.ok()) {
    return flux_buffer.status();
  }
  const auto flux_sums = *flux_buffer;
  std::vector<double> residual(cells.size());
  double original_rhs_norm = 0.0;
  if (starts_with_history) {
    // Measure the RHS with the same velocity block at zero pressure. A
    // converged history is allowed to have a small residual without shrinking
    // the tolerance or changing the recorded physical forcing scale.
    const auto base_velocity = solve_velocity(zero_pressure, true);
    if (!base_velocity.ok()) {
      return base_velocity.status();
    }
    const auto base_flux =
        AccumulateConstraint(system, zero_pressure, base_velocity->velocity, true, flux_sums, residual);
    if (!base_flux.ok()) {
      return base_flux;
    }
    const auto base_norm = CoupledConstraintNorm(residual);
    if (!base_norm.ok()) {
      return base_norm.status();
    }
    original_rhs_norm = *base_norm;
    if (original_rhs_norm == 0.0) {
      std::ranges::fill(pressure, 0.0);
    }
  }
  const std::size_t basis_capacity = std::min(pressure_internal::kCoupledPressureRestartLength3D, max_iterations);
  const std::size_t basis_extent = basis_capacity * cells.size();
  std::span<double> directions;
  std::span<double> responses;
  std::size_t basis_count = 0;
  pressure_internal::MacPressureStoppingCriteria3D stopping{};
  for (std::size_t iteration = 0; iteration <= max_iterations; ++iteration) {
    auto velocity = solve_velocity(pressure, true);
    if (!velocity.ok()) {
      return velocity.status();
    }
    const auto physical_flux = AccumulateConstraint(system, pressure, velocity->velocity, true, flux_sums, residual);
    if (!physical_flux.ok()) {
      return physical_flux;
    }
    const auto constraint_norm = CoupledConstraintNorm(residual);
    if (!constraint_norm.ok()) {
      return constraint_norm.status();
    }
    const double residual_norm = *constraint_norm;
    if (iteration == 0) {
      if (!starts_with_history) {
        original_rhs_norm = residual_norm;
      }
      const auto criteria = pressure_internal::MacPressureStoppingCriteria(original_rhs_norm, links);
      if (!criteria.ok()) {
        return criteria.status();
      }
      stopping = *criteria;
    }
    double max_defect = 0.0;
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      const double defect = residual[cell] / cells[cell].fluid_volume;
      if (!std::isfinite(defect)) {
        return absl::OutOfRangeError("coupled pressure defect is unrepresentable");
      }
      max_defect = std::max(max_defect, std::abs(defect));
    }
    if (max_defect <= stopping.volume_scaled_limit && residual_norm <= stopping.integrated_residual_limit) {
      for (std::size_t cell = 0; cell < cells.size(); ++cell) {
        residual[cell] /= cells[cell].fluid_volume;
      }
      std::vector<double> accepted_velocity(velocity->velocity.begin(), velocity->velocity.end());
      for (std::size_t face = 0; face < links.size(); ++face) {
        if (inertia[face] == 0.0) {
          accepted_velocity[face] = wall_normal_velocities[face];
        }
      }
      return MacViscousPressureProjection3D{
          .cell_pressures = std::move(pressure),
          .face_velocities = std::move(accepted_velocity),
          .cell_constraint_defects = std::move(residual),
          .pressure_iterations = iteration,
          .constraint_residual_l2 = residual_norm,
          .initial_constraint_residual_l2 = original_rhs_norm,
          .max_cell_constraint_defect = max_defect,
          .max_momentum_residual = velocity->max_residual,
      };
    }
    if (iteration == max_iterations) {
      break;
    }

    // GCR retains measured responses rather than assuming that a tolerance-based
    // inner solve is an exact linear inverse. Recompute the physical residual
    // from the coupled candidate after each pressure update.
    if (directions.empty()) {
      const auto basis = scratch.Take<double>(2 * basis_extent);
      if (!basis.ok()) {
        return basis.status();
      }
      directions = basis->first(basis_extent);
      responses = basis->subspan(basis_extent);
    }
    if (basis_count == basis_capacity) {
      basis_count = 0;
    }
    const auto direction = directions.subspan(basis_count * cells.size(), cells.size());
    const auto response = responses.subspan(basis_count * cells.size(), cells.size());
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      if (pinned[cell]) {
        residual[cell] = 0.0;
        direction[cell] = 0.0;
      } else {
        if (!pressure_internal::arithmetic::IsFinitePositive(pressure_diagonal[cell])) {
          return absl::OutOfRangeError("coupled preconditioner diagonal is unrepresentable");
        }
        direction[cell] = residual[cell] / pressure_diagonal[cell];
      }
    }
    const auto response_velocity = solve_velocity(direction, false);
    if (!response_velocity.ok()) {
      return response_velocity.status();
    }
    const auto response_flux =
        AccumulateConstraint(system, direction, response_velocity->velocity, false, flux_sums, response);
    if (!response_flux.ok()) {
      return response_flux;
    }
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      if (pinned[cell]) {
        response[cell] = 0.0;
      }
    }
    for (std::size_t previous = 0; previous < basis_count; ++previous) {
      const auto previous_response = responses.subspan(previous * cells.size(), cells.size());
      const auto previous_direction = directions.subspan(previous * cells.size(), cells.size());
      const double overlap = Dot(response, previous_response);
      for (std::size_t cell = 0; cell < cells.size(); ++cell) {
        response[cell] -= overlap * previous_response[cell];
        direction[cell] -= overlap * previous_direction[cell];
      }
    }
    const double norm = std::sqrt(Dot(response, response));
    if (!std::isfinite(norm) || norm <= 0.0) {
      return absl::AbortedError("coupled pressure direction has no usable response");
    }
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      response[cell] /= norm;
      direction[cell] /= norm;
    }
    const double step = Dot(residual, response);
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      pressure[cell] -= step * direction[cell];
      if (!std::isfinite(pressure[cell])) {
        return absl::OutOfRangeError("coupled pressure iterate is unrepresentable");
      }
    }
    ++basis_count;
  }
  return absl::AbortedError("coupled pressure iteration budget exhausted");
}

absl::StatusOr<MacViscousPressureProjection3D> ProjectUniformWithVelocityView(
    const pressure_internal::UniformViscousPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    std::optional<SolveViscousVelocityView3D> solve_velocity_block,
    pressure_internal::MacPressureScratch3D& scratch
) {
  const auto system = pressure_internal::PrepareBoundedMacPressureSystem3D(
      inputs.mac_grid,
      inputs.step_duration,
      inputs.pressure_cells,
      inputs.pressure_faces,
      scratch
  );
  if (!system.ok()) {
    return system.status();
  }
  const auto face_buffers =
      scratch.Take<double>(pressure_internal::kCoupledFaceScratchFields3D * inputs.pressure_faces.size());
  if (!face_buffers.ok()) {
    return face_buffers.status();
  }
  const auto wall_normal_velocities = face_buffers->first(inputs.pressure_faces.size());
  pressure_internal::arithmetic::CompensatedSum inward_prescribed_wall_flux;
  for (std::size_t ordinal = 0; ordinal < inputs.pressure_faces.size(); ++ordinal) {
    const auto& face = inputs.pressure_faces[ordinal];
    wall_normal_velocities[ordinal] = face.wall_normal_velocity;
    pressure_internal::arithmetic::AccumulateInwardExplicitWallFlux(
        system->links()[ordinal],
        face.prescribed_wall_flux.has_value(),
        inward_prescribed_wall_flux
    );
  }
  auto result = SolveCoupled(
      *system,
      inputs.step_duration,
      strain_rows,
      *face_buffers,
      options.max_iterations,
      solve_velocity_block,
      scratch
  );
  if (result.ok()) {
    result->inward_prescribed_wall_flux_m3_per_s = inward_prescribed_wall_flux.Total();
  }
  return result;
}

absl::StatusOr<LiquidRecords3D<std::max_align_t>> CreateCoupledScratch(
    std::size_t cell_count,
    std::size_t face_count,
    std::size_t strain_row_count,
    bool prepare_uniform_system
) {
  const auto bytes = pressure_internal::QueryCoupledMacPressureScratchBytes3D(
      cell_count,
      face_count,
      strain_row_count,
      prepare_uniform_system
  );
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto storage = LiquidRecords3D<std::max_align_t>::Create(*bytes / sizeof(std::max_align_t));
  if (!storage.ok()) {
    return storage.status();
  }
  const auto resize = storage->Resize(storage->capacity());
  if (!resize.ok()) {
    return resize;
  }
  return storage;
}

absl::StatusOr<MacViscousPressureProjection3D> ProjectUniformWithReservation(
    const pressure_internal::UniformViscousPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    std::optional<SolveViscousVelocityView3D> solve_velocity_block,
    MacPressureWorkspace3D* workspace
) {
  if (options.max_iterations == 0 || options.max_iterations > kPressureIterationBudget) {
    return absl::InvalidArgumentError("invalid coupled pressure iteration budget");
  }
  const auto cell_count = inputs.pressure_cells.size();
  const auto face_count = inputs.pressure_faces.size();
  if (workspace != nullptr) {
    using Access = pressure_internal::CoupledPressureWorkspaceAccess3D;
    if (!Access::Fits(*workspace, cell_count, face_count)
        || strain_rows.size() > cell_count * pressure_internal::kMaximumUniformStrainRowsPerCell3D) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    pressure_internal::MacPressureScratch3D scratch(Access::Storage(*workspace));
    auto result = ProjectUniformWithVelocityView(inputs, strain_rows, options, solve_velocity_block, scratch);
    Access::RecordUsage(*workspace, scratch);
    return result;
  }

  auto storage = CreateCoupledScratch(cell_count, face_count, strain_rows.size(), true);
  if (!storage.ok()) {
    return storage.status();
  }
  pressure_internal::MacPressureScratch3D scratch(storage->values());
  return ProjectUniformWithVelocityView(inputs, strain_rows, options, solve_velocity_block, scratch);
}

pressure_internal::UniformViscousPressureInputs3D BorrowUniformInputs(const MacPressureInputs3D& inputs) {
  return {inputs.mac_grid.view(), inputs.step_duration, inputs.pressure_cells, inputs.pressure_faces};
}

absl::StatusOr<MacViscousPressureProjection3D> ProjectCpuUniform(
    const pressure_internal::UniformViscousPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D* workspace
) {
  return ProjectUniformWithReservation(inputs, strain_rows, options, std::nullopt, workspace);
}

absl::StatusOr<MacViscousPressureProjection3D> ProjectUniformWithOwningVelocity(
    const pressure_internal::UniformViscousPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    pressure_internal::SolveViscousVelocity3D solve_velocity_block,
    MacPressureWorkspace3D* workspace
) {
  std::optional<MacViscousVelocityBlockSolution3D> owned_velocity;
  const auto solve = [&](const MacViscousVelocityBlockInputs3D& block,
                         std::span<const double> rhs,
                         MacViscousVelocityBlockOptions3D controls
                     ) -> absl::StatusOr<viscosity_internal::VelocityBlockViewSolution3D> {
    auto result = solve_velocity_block(block, rhs, controls);
    if (!result.ok()) {
      return result.status();
    }
    owned_velocity.emplace(std::move(*result));
    return viscosity_internal::VelocityBlockViewSolution3D{
        .velocity = owned_velocity->velocity,
        .iterations = owned_velocity->iterations,
        .max_residual = owned_velocity->max_residual,
    };
  };
  return ProjectUniformWithReservation(inputs, strain_rows, options, SolveViscousVelocityView3D(solve), workspace);
}
}  // namespace

absl::StatusOr<MacViscousPressureProjection3D> ProjectMacViscousPressure3D(
    const MacPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options
) {
  return ProjectCpuUniform(BorrowUniformInputs(inputs), strain_rows, options, nullptr);
}

namespace pressure_internal {
absl::StatusOr<MacViscousPressureProjection3D> ProjectMacViscousPressureWithVelocitySolver3D(
    const MacPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    SolveViscousVelocity3D solve_velocity_block
) {
  return ProjectUniformWithOwningVelocity(
      BorrowUniformInputs(inputs),
      strain_rows,
      options,
      solve_velocity_block,
      nullptr
  );
}

absl::StatusOr<MacViscousPressureProjection3D> ProjectWorkspaceMacViscousPressure3D(
    const UniformViscousPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace
) {
  return ProjectCpuUniform(inputs, strain_rows, options, &workspace);
}

absl::StatusOr<MacViscousPressureProjection3D> ProjectWorkspaceMacViscousPressureWithVelocitySolver3D(
    const UniformViscousPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace,
    SolveViscousVelocity3D solve_velocity_block
) {
  return ProjectUniformWithOwningVelocity(inputs, strain_rows, options, solve_velocity_block, &workspace);
}

absl::StatusOr<MacViscousPressureProjection3D> ProjectExplicitMacViscousPressure3D(
    double step_duration,
    std::span<const MacPressureCell3D> cells,
    std::span<const ExplicitMacPressureFace3D> faces,
    std::span<const MacStrainRow3D> strain_rows
) {
  const auto system = PrepareExplicitMacPressureSystem3D(step_duration, cells, faces);
  if (!system.ok()) {
    return system.status();
  }
  auto storage = CreateCoupledScratch(cells.size(), faces.size(), strain_rows.size(), false);
  if (!storage.ok()) {
    return storage.status();
  }
  MacPressureScratch3D arena(storage->values());
  const auto face_buffers = arena.Take<double>(kCoupledFaceScratchFields3D * faces.size());
  if (!face_buffers.ok()) {
    return face_buffers.status();
  }
  return SolveCoupled(
      BorrowMacPressureSystem3D(*system),
      step_duration,
      strain_rows,
      *face_buffers,
      kPressureIterationBudget,
      std::nullopt,
      arena
  );
}
}  // namespace pressure_internal

}  // namespace fluidity
