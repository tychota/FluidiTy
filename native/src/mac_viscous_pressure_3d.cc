#include "mac_viscous_pressure_3d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "mac_pressure_3d_arithmetic.h"
#include "mac_pressure_3d_multigrid.h"

namespace fluidity {
namespace {

using pressure_internal::PreparedMacPressureSystem3D;

constexpr std::size_t kPressureIterationBudget = 256;
constexpr std::size_t kRestartLength = 32;

// Remap sparse grid slots to compact incident face ordinals for the velocity solve.
absl::StatusOr<std::vector<MacStrainRow3D>>
RemapStrain(const PreparedMacPressureSystem3D& system, std::span<const MacStrainRow3D> strain_rows) {
  const auto links = system.links();
  std::vector<MacStrainRow3D> rows(strain_rows.begin(), strain_rows.end());
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
  return rows;
}

// Apply D and C in integrated flux units. Add the wall and source terms only
// for the physical candidate, never for a pressure search direction.
absl::Status AccumulateConstraint(
    const PreparedMacPressureSystem3D& system,
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

// Solve on frozen links. Explicit graphs have no closed faces and an empty
// wall velocity view. Uniform graphs supply one prescribed value per link.
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
    const PreparedMacPressureSystem3D& system,
    double dt,
    std::span<const MacStrainRow3D> strain_rows,
    std::span<const double> wall_normal_velocities,
    std::size_t max_iterations,
    pressure_internal::SolveViscousVelocity3D solve_velocity_block
) {
  const auto rows = RemapStrain(system, strain_rows);
  if (!rows.ok()) {
    return rows.status();
  }
  const auto links = system.links();
  const auto cells = system.cells();
  std::vector<double> inertia(links.size());
  std::vector<double> base_momentum(links.size());
  std::vector<double> pressure_diagonal(cells.size());
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
  std::vector<double> velocity_diagonal = inertia;
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

  std::vector<bool> pinned(cells.size());
  for (const auto cell : system.gauge_cell_ordinals()) {
    pinned[cell] = true;
  }
  const MacViscousVelocityBlockInputs3D velocity_block{*rows, inertia, dt};
  std::vector<double> momentum(links.size());
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
    return solve_velocity_block(velocity_block, momentum, {});
  };
  auto initial_pressure = pressure_internal::InitialMacPressure3D(system);
  if (!initial_pressure.ok()) {
    return initial_pressure.status();
  }
  auto pressure = std::move(*initial_pressure);
  const bool starts_with_history = std::ranges::any_of(pressure, [](double value) {
    return value != 0.0;
  });
  std::vector<pressure_internal::arithmetic::CompensatedSum> flux_sums(cells.size());
  std::vector<double> residual(cells.size());
  double original_rhs_norm = 0.0;
  if (starts_with_history) {
    // Measure the RHS with the same velocity block at zero pressure. A
    // converged history is allowed to have a small residual without shrinking
    // the tolerance or changing the recorded physical forcing scale.
    const std::vector<double> zero_pressure(cells.size());
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
  const std::size_t basis_capacity = std::min(kRestartLength, max_iterations);
  const std::size_t basis_extent = basis_capacity * cells.size();
  std::vector<double> directions;
  std::vector<double> responses;
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
      for (std::size_t face = 0; face < links.size(); ++face) {
        if (inertia[face] == 0.0) {
          velocity->velocity[face] = wall_normal_velocities[face];
        }
      }
      return MacViscousPressureProjection3D{
          .cell_pressures = std::move(pressure),
          .face_velocities = std::move(velocity->velocity),
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
      directions.resize(basis_extent);
      responses.resize(basis_extent);
    }
    if (basis_count == basis_capacity) {
      basis_count = 0;
    }
    const auto direction = std::span(directions).subspan(basis_count * cells.size(), cells.size());
    const auto response = std::span(responses).subspan(basis_count * cells.size(), cells.size());
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
      const auto previous_response = std::span(responses).subspan(previous * cells.size(), cells.size());
      const auto previous_direction = std::span(directions).subspan(previous * cells.size(), cells.size());
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

}  // namespace

absl::StatusOr<MacViscousPressureProjection3D> ProjectMacViscousPressure3D(
    const MacPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options
) {
  return pressure_internal::ProjectMacViscousPressureWithVelocitySolver3D(
      inputs,
      strain_rows,
      options,
      SolveMacViscousVelocityBlock3D
  );
}

namespace pressure_internal {
absl::StatusOr<MacViscousPressureProjection3D> ProjectMacViscousPressureWithVelocitySolver3D(
    const MacPressureInputs3D& inputs,
    std::span<const MacStrainRow3D> strain_rows,
    MacPressureSolverOptions3D options,
    SolveViscousVelocity3D solve_velocity_block
) {
  if (options.max_iterations == 0 || options.max_iterations > kPressureIterationBudget) {
    return absl::InvalidArgumentError("invalid coupled pressure iteration budget");
  }
  const auto system = pressure_internal::PrepareMacPressureSystem3D(inputs);
  if (!system.ok()) {
    return system.status();
  }
  std::vector<double> wall_normal_velocities;
  wall_normal_velocities.reserve(inputs.pressure_faces.size());
  arithmetic::CompensatedSum inward_prescribed_wall_flux;
  for (std::size_t ordinal = 0; ordinal < inputs.pressure_faces.size(); ++ordinal) {
    const auto& face = inputs.pressure_faces[ordinal];
    wall_normal_velocities.push_back(face.wall_normal_velocity);
    arithmetic::AccumulateInwardExplicitWallFlux(
        system->links()[ordinal],
        face.prescribed_wall_flux.has_value(),
        inward_prescribed_wall_flux
    );
  }
  auto result = SolveCoupled(
      *system,
      inputs.step_duration,
      strain_rows,
      wall_normal_velocities,
      options.max_iterations,
      solve_velocity_block
  );
  if (result.ok()) {
    result->inward_prescribed_wall_flux_m3_per_s = inward_prescribed_wall_flux.Total();
  }
  return result;
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
  return SolveCoupled(
      *system,
      step_duration,
      strain_rows,
      {},
      kPressureIterationBudget,
      SolveMacViscousVelocityBlock3D
  );
}
}  // namespace pressure_internal

}  // namespace fluidity
