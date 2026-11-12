#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "fluidity/host/viscosity.h"
#include "mac_pressure_3d_arithmetic.h"
#include "mac_pressure_3d_system.h"

namespace fluidity {
namespace {

using pressure_internal::PreparedMacPressureSystem3D;

constexpr double kConstraintTolerance = 2e-5;
constexpr std::size_t kPressureIterationBudget = 256;
constexpr std::size_t kRestartLength = 32;

// Remap sparse grid slots to compact incident face ordinals for the velocity solve.
absl::StatusOr<std::vector<MacStrainRow3D>>
RemapStrain(const PreparedMacPressureSystem3D& system, std::span<const MacStrainRow3D> strain_rows) {
  const auto links = system.links();
  std::vector<MacStrainRow3D> rows(strain_rows.begin(), strain_rows.end());
  for (auto& row : rows) {
    if (row.term_count != 2 && row.term_count != 4) {
      return absl::InvalidArgumentError("invalid coupled strain row extent");
    }
    if (!std::isfinite(row.quadrature_weight) || row.quadrature_weight < 0.0) {
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
absl::StatusOr<std::vector<double>> AccumulateConstraint(
    const PreparedMacPressureSystem3D& system,
    std::span<const double> pressure,
    std::span<const double> velocity,
    bool physical
) {
  const auto cells = system.cells();
  const auto links = system.links();
  std::vector<pressure_internal::arithmetic::CompensatedSum> flux(cells.size());
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
  std::vector<double> result(cells.size());
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    result[cell] = flux[cell].Total();
    if (!std::isfinite(result[cell])) {
      return absl::OutOfRangeError("coupled cell flux is unrepresentable");
    }
  }
  return result;
}

double Dot(std::span<const double> left, std::span<const double> right) {
  return std::inner_product(left.begin(), left.end(), right.begin(), 0.0);
}

}  // namespace

absl::StatusOr<MacViscousPressureProjection3D>
ProjectMacViscousPressure3D(const MacPressureInputs3D& inputs, std::span<const MacStrainRow3D> strain_rows) {
  const auto system = pressure_internal::PrepareMacPressureSystem3D(inputs);
  if (!system.ok()) {
    return system.status();
  }
  const auto rows = RemapStrain(*system, strain_rows);
  if (!rows.ok()) {
    return rows.status();
  }
  const auto links = system->links();
  const auto cells = system->cells();
  const double dt = inputs.step_duration;
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
    double prescribed_strain = 0.0;
    for (const auto& term : terms) {
      if (inertia[term.face_slot] == 0.0) {
        prescribed_strain += term.coefficient * inputs.pressure_faces[term.face_slot].wall_normal_velocity;
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
  for (const auto cell : system->gauge_cell_ordinals()) {
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
    return SolveMacViscousVelocityBlock3D(velocity_block, momentum);
  };
  std::vector<double> pressure(cells.size());
  std::vector<std::vector<double>> directions;
  std::vector<std::vector<double>> responses;
  double initial_residual_norm = 0.0;
  for (std::size_t iteration = 0; iteration <= kPressureIterationBudget; ++iteration) {
    auto velocity = solve_velocity(pressure, true);
    if (!velocity.ok()) {
      return velocity.status();
    }
    auto physical_flux = AccumulateConstraint(*system, pressure, velocity->velocity, true);
    if (!physical_flux.ok()) {
      return physical_flux.status();
    }
    auto& residual = *physical_flux;
    double residual_norm = 0.0;
    for (const double flux : residual) {
      residual_norm = std::hypot(residual_norm, flux);
    }
    if (!std::isfinite(residual_norm)) {
      return absl::OutOfRangeError("coupled residual norm is unrepresentable");
    }
    if (iteration == 0) {
      initial_residual_norm = residual_norm;
    }
    double max_defect = 0.0;
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      const double defect = residual[cell] / cells[cell].fluid_volume;
      if (!std::isfinite(defect)) {
        return absl::OutOfRangeError("coupled pressure defect is unrepresentable");
      }
      max_defect = std::max(max_defect, std::abs(defect));
    }
    if (max_defect <= kConstraintTolerance && residual_norm <= 2e-6 + 2e-6 * initial_residual_norm) {
      for (std::size_t cell = 0; cell < cells.size(); ++cell) {
        residual[cell] /= cells[cell].fluid_volume;
      }
      for (std::size_t face = 0; face < links.size(); ++face) {
        if (inertia[face] == 0.0) {
          velocity->velocity[face] = inputs.pressure_faces[face].wall_normal_velocity;
        }
      }
      return MacViscousPressureProjection3D{
          .cell_pressures = std::move(pressure),
          .face_velocities = std::move(velocity->velocity),
          .cell_constraint_defects = std::move(residual),
          .pressure_iterations = iteration,
          .max_momentum_residual = velocity->max_residual,
      };
    }
    if (iteration == kPressureIterationBudget) {
      break;
    }

    // GCR retains measured responses rather than assuming that a tolerance-based
    // inner solve is an exact linear inverse. Recompute the physical residual
    // from the coupled candidate after each pressure update.
    if (directions.size() == kRestartLength) {
      directions.clear();
      responses.clear();
    }
    std::vector<double> direction(cells.size());
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      if (pinned[cell]) {
        residual[cell] = 0.0;
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
    auto response_flux = AccumulateConstraint(*system, direction, response_velocity->velocity, false);
    if (!response_flux.ok()) {
      return response_flux.status();
    }
    auto& response = *response_flux;
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
      if (pinned[cell]) {
        response[cell] = 0.0;
      }
    }
    for (std::size_t previous = 0; previous < responses.size(); ++previous) {
      const double overlap = Dot(response, responses[previous]);
      for (std::size_t cell = 0; cell < cells.size(); ++cell) {
        response[cell] -= overlap * responses[previous][cell];
        direction[cell] -= overlap * directions[previous][cell];
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
    directions.push_back(std::move(direction));
    responses.push_back(std::move(response));
  }
  return absl::AbortedError("coupled pressure iteration budget exhausted");
}

}  // namespace fluidity
