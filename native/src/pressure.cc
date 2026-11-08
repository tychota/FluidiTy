#include "fluidity/host/pressure.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/host/grid.h"
#include "fluidity/host/small_solve.h"
#include "fluidity/host/views.h"

namespace fluidity {
namespace {

static_assert(kMaxPressureCells <= kMaxDenseUnknowns);

using FaceLink = PressureWorkspace::FaceLink;
constexpr double kCompatibilityTolerance = 1e-12;

bool IsFinite(double value) {
  return std::isfinite(value);
}

bool IsFinitePositive(double value) {
  return value > 0.0 && std::isfinite(value);
}

// Subnormal matrix coefficients can lose relative precision before pressure lifts.
// A larger diagonal contribution can hide that loss from the SPD checks.
bool IsPositiveNormal(double value) {
  return value >= std::numeric_limits<double>::min() && std::isfinite(value);
}

// The inputs are finite and positive. Evaluate with normal-sized fractions so
// an intermediate product cannot lose range before division restores it.
double MultiplyDividePositive(double first_factor, double second_factor, double divisor) {
  int first_exponent;
  int second_exponent;
  int divisor_exponent;
  const double first_fraction = std::frexp(first_factor, &first_exponent);
  const double second_fraction = std::frexp(second_factor, &second_exponent);
  const double divisor_fraction = std::frexp(divisor, &divisor_exponent);
  const double fraction = first_fraction * second_fraction / divisor_fraction;
  return std::scalbn(fraction, first_exponent + second_exponent - divisor_exponent);
}

absl::Status ValidatePressureViews(
    const PressureInputsView& inputs,
    const PressureOutputsView& outputs,
    const PressureWorkspace& workspace,
    std::size_t cell_count,
    std::size_t face_count
) {
  const auto face_extents = CheckExtents(
      face_count,
      {
          inputs.predicted_face_velocities.size(),
          inputs.face_open_fractions.size(),
          inputs.face_inverse_densities.size(),
          inputs.pressure_gradient_distances.size(),
          inputs.wall_face_velocities.size(),
          inputs.dirichlet_face_mask.size(),
          inputs.dirichlet_face_pressures.size(),
          inputs.face_pressure_jumps.size(),
          outputs.corrected_face_velocities.size(),
      }
  );
  if (!face_extents.ok()) {
    return face_extents;
  }
  const auto cell_extents = CheckExtents(
      cell_count,
      {
          inputs.target_cell_divergence.size(),
          inputs.cell_fluid_volumes.size(),
          inputs.cell_bulk_moduli.size(),
          inputs.previous_cell_pressures.size(),
          outputs.cell_pressures.size(),
          outputs.original_rhs.size(),
          outputs.original_residuals.size(),
          outputs.cell_constraint_defects.size(),
          outputs.gauge_cell_indices.size(),
      }
  );
  if (!cell_extents.ok()) {
    return cell_extents;
  }
  if (const auto status = ValidateExtent(outputs.original_matrix.size(), cell_count * cell_count); !status.ok()) {
    return status;
  }

  return CheckWritableViews(
      {
          std::as_bytes(outputs.cell_pressures),
          std::as_bytes(outputs.corrected_face_velocities),
          std::as_bytes(outputs.original_matrix),
          std::as_bytes(outputs.original_rhs),
          std::as_bytes(outputs.original_residuals),
          std::as_bytes(outputs.cell_constraint_defects),
          std::as_bytes(outputs.gauge_cell_indices),
          std::as_bytes(std::span(&outputs.gauge_count, 1)),
          std::as_bytes(std::span(&workspace, 1)),
      },
      {
          std::as_bytes(std::span(&inputs, 1)),
          std::as_bytes(inputs.predicted_face_velocities),
          std::as_bytes(inputs.face_open_fractions),
          std::as_bytes(inputs.face_inverse_densities),
          std::as_bytes(inputs.pressure_gradient_distances),
          std::as_bytes(inputs.wall_face_velocities),
          std::as_bytes(inputs.dirichlet_face_mask),
          std::as_bytes(inputs.dirichlet_face_pressures),
          std::as_bytes(inputs.face_pressure_jumps),
          std::as_bytes(inputs.target_cell_divergence),
          std::as_bytes(inputs.cell_fluid_volumes),
          std::as_bytes(inputs.cell_bulk_moduli),
          std::as_bytes(inputs.previous_cell_pressures),
      }
  );
}

absl::Status ValidatePressureCoefficients(const PressureInputsView& inputs) {
  const std::array finite_fields{
      inputs.predicted_face_velocities,
      inputs.wall_face_velocities,
      inputs.dirichlet_face_pressures,
      inputs.face_pressure_jumps,
      inputs.target_cell_divergence,
      inputs.previous_cell_pressures,
  };
  for (const auto field : finite_fields) {
    if (!std::ranges::all_of(field, IsFinite)) {
      return absl::InvalidArgumentError("pressure velocity, pressure, jump and source fields must be finite");
    }
  }
  const auto valid_aperture = [](double aperture) {
    return aperture >= 0.0 && aperture <= 1.0;
  };
  if (!std::ranges::all_of(inputs.face_open_fractions, valid_aperture)) {
    return absl::InvalidArgumentError("pressure aperture must be in [0, 1]");
  }
  for (const auto field :
       {inputs.face_inverse_densities, inputs.pressure_gradient_distances, inputs.cell_fluid_volumes}) {
    if (!std::ranges::all_of(field, IsFinitePositive)) {
      return absl::InvalidArgumentError("pressure mobility, distance and fluid volume must be finite and positive");
    }
  }
  const auto valid_bulk_modulus = [](double modulus) {
    return modulus > 0.0;
  };
  if (!std::ranges::all_of(inputs.cell_bulk_moduli, valid_bulk_modulus)) {
    return absl::InvalidArgumentError("pressure bulk modulus must be positive");
  }
  const auto binary_mask = [](std::uint8_t value) {
    return value <= 1;
  };
  if (!std::ranges::all_of(inputs.dirichlet_face_mask, binary_mask)) {
    return absl::InvalidArgumentError("pressure Dirichlet mask must contain 0 or 1");
  }
  return absl::OkStatus();
}

bool IsInternal(const FaceLink& face) {
  return face.tail >= 0 && face.head >= 0;
}

// Store the same face geometry and coefficients for assembly and velocity correction.
absl::StatusOr<FaceLink> BuildFaceLink(const PressureInputsView& inputs, std::size_t face_index) {
  const auto geometry = FaceGeometry(inputs.grid, face_index);
  if (!geometry.ok()) {
    return geometry.status();
  }
  const double aperture = inputs.face_open_fractions[face_index];
  FaceLink face{
      .tail = geometry->tail_cell_index,
      .head = geometry->head_cell_index,
      .open_area = geometry->area * aperture,
      .wall_area = 0.0,
      .pressure_to_velocity_scale = 0.0,
      .conductance = 0.0,
      .signed_boundary_pressure = 0.0,
      .has_pressure_coupling = false,
  };
  const bool is_dirichlet = inputs.dirichlet_face_mask[face_index] == 1;
  if (is_dirichlet && IsInternal(face)) {
    return absl::InvalidArgumentError("pressure Dirichlet face must be on the grid boundary");
  }
  if (is_dirichlet && inputs.face_pressure_jumps[face_index] != 0.0) {
    return absl::InvalidArgumentError("pressure jump must be zero on a Dirichlet face");
  }

  face.wall_area = geometry->area - face.open_area;
  const bool supports_pressure_coupling = IsInternal(face) || is_dirichlet;
  face.has_pressure_coupling = aperture > 0.0 && supports_pressure_coupling;
  if (!face.has_pressure_coupling) {
    return face;
  }
  if (face.open_area == 0.0) {
    return absl::OutOfRangeError("active pressure open area rounded to zero");
  }
  const double mobility = inputs.face_inverse_densities[face_index];
  const double distance = inputs.pressure_gradient_distances[face_index];
  face.pressure_to_velocity_scale = MultiplyDividePositive(inputs.dt, mobility, distance);
  if (!IsFinitePositive(face.pressure_to_velocity_scale)) {
    return absl::OutOfRangeError("pressure velocity scale is outside the positive representable range");
  }
  // Use the correction coefficient in the matrix so both phases share the face response.
  face.conductance = face.open_area * face.pressure_to_velocity_scale;
  if (!IsPositiveNormal(face.conductance)) {
    return absl::OutOfRangeError("active pressure conductance must be a positive normal double");
  }
  if (is_dirichlet) {
    // The missing pressure contributes with the opposite incidence of the adjacent cell.
    const double outward_sign = face.tail >= 0 ? 1.0 : -1.0;
    face.signed_boundary_pressure = outward_sign * inputs.dirichlet_face_pressures[face_index];
  }
  return face;
}

double IntegratedFaceFlux(const FaceLink& face, double fluid_velocity, double wall_velocity) {
  return face.open_area * fluid_velocity + face.wall_area * wall_velocity;
}

void AccumulateOutwardFaceFlux(const FaceLink& face, double oriented_face_flux, std::span<double> outward_cell_fluxes) {
  if (face.tail >= 0) {
    outward_cell_fluxes[static_cast<std::size_t>(face.tail)] += oriented_face_flux;
  }
  if (face.head >= 0) {
    outward_cell_fluxes[static_cast<std::size_t>(face.head)] -= oriented_face_flux;
  }
}

// Assemble A = R diag(k) R^T + C and b in integrated flux units.
absl::Status AssembleOriginalSystem(
    const PressureInputsView& inputs,
    std::span<bool> cell_has_pressure_anchor,
    PressureWorkspace& workspace
) {
  const std::size_t cell_count = cell_has_pressure_anchor.size();
  auto original_matrix = std::span(workspace.original_matrix).first(cell_count * cell_count);
  auto original_rhs = std::span(workspace.original_rhs).first(cell_count);
  std::ranges::fill(original_matrix, 0.0);

  // Keep the three incidence products separate. Balance source and predicted flux
  // before adding the pressure lifts so cancellation preserves small contributions.
  std::array<double, kMaxPressureCells> predicted_outward_fluxes{};
  std::array<double, kMaxPressureCells> jump_flux_lifts{};
  std::array<double, kMaxPressureCells> dirichlet_flux_lifts{};

  for (std::size_t face_index = 0; face_index < inputs.predicted_face_velocities.size(); ++face_index) {
    const auto face = BuildFaceLink(inputs, face_index);
    if (!face.ok()) {
      return face.status();
    }
    workspace.faces[face_index] = *face;
    const double face_conductance = face->conductance;
    const double predicted_face_flux = IntegratedFaceFlux(
        *face,
        inputs.predicted_face_velocities[face_index],
        inputs.wall_face_velocities[face_index]
    );
    AccumulateOutwardFaceFlux(*face, predicted_face_flux, predicted_outward_fluxes);
    AccumulateOutwardFaceFlux(*face, face_conductance * inputs.face_pressure_jumps[face_index], jump_flux_lifts);
    // A Dirichlet column has only one incidence, which squares its sign in the lift.
    AccumulateOutwardFaceFlux(*face, face_conductance * face->signed_boundary_pressure, dirichlet_flux_lifts);

    if (face_conductance == 0.0) {
      continue;
    }
    if (IsInternal(*face)) {
      const auto tail_cell_index = static_cast<std::size_t>(face->tail);
      const auto head_cell_index = static_cast<std::size_t>(face->head);
      original_matrix[tail_cell_index * cell_count + tail_cell_index] += face_conductance;
      original_matrix[head_cell_index * cell_count + head_cell_index] += face_conductance;
      original_matrix[tail_cell_index * cell_count + head_cell_index] -= face_conductance;
      original_matrix[head_cell_index * cell_count + tail_cell_index] -= face_conductance;
    } else {
      const auto boundary_cell_index = static_cast<std::size_t>(face->tail >= 0 ? face->tail : face->head);
      original_matrix[boundary_cell_index * cell_count + boundary_cell_index] += face_conductance;
      cell_has_pressure_anchor[boundary_cell_index] = true;
    }
  }

  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    const double cell_bulk_modulus = inputs.cell_bulk_moduli[cell_index];
    double bulk_modulus_times_dt = std::numeric_limits<double>::infinity();
    double cell_compliance = 0.0;
    if (std::isfinite(cell_bulk_modulus)) {
      bulk_modulus_times_dt = cell_bulk_modulus * inputs.dt;
      if (!IsFinitePositive(bulk_modulus_times_dt)) {
        return absl::OutOfRangeError("pressure bulk modulus times dt is outside the representable range");
      }
      cell_compliance = inputs.cell_fluid_volumes[cell_index] / bulk_modulus_times_dt;
      if (!IsPositiveNormal(cell_compliance)) {
        return absl::OutOfRangeError("finite-modulus pressure compliance must be a positive normal double");
      }
    }
    workspace.bulk_modulus_times_dt[cell_index] = bulk_modulus_times_dt;
    cell_has_pressure_anchor[cell_index] = cell_has_pressure_anchor[cell_index] || cell_compliance > 0.0;
    original_matrix[cell_index * cell_count + cell_index] += cell_compliance;
    const double integrated_target_flux =
        inputs.cell_fluid_volumes[cell_index] * inputs.target_cell_divergence[cell_index];
    original_rhs[cell_index] = integrated_target_flux - predicted_outward_fluxes[cell_index]
        - jump_flux_lifts[cell_index] + dirichlet_flux_lifts[cell_index]
        + cell_compliance * inputs.previous_cell_pressures[cell_index];
  }

  if (!std::ranges::all_of(original_matrix, IsFinite) || !std::ranges::all_of(original_rhs, IsFinite)) {
    return absl::OutOfRangeError("pressure assembly is outside the representable range");
  }
  return absl::OkStatus();
}

std::size_t FindComponentRootCellIndex(std::span<const std::size_t> parent_cell_indices, std::size_t cell_index) {
  while (parent_cell_indices[cell_index] != cell_index) {
    cell_index = parent_cell_indices[cell_index];
  }
  return cell_index;
}

// Only positive internal conductance connects cells. Union toward the smaller root
// so each unanchored component's root is also its deterministic gauge index.
absl::StatusOr<std::size_t> SelectGauges(
    std::span<const FaceLink> faces,
    std::span<const bool> cell_has_pressure_anchor,
    std::span<const double> original_rhs,
    std::span<std::size_t> gauge_cell_indices
) {
  std::array<std::size_t, kMaxPressureCells> parent_cell_index_storage;
  const auto parent_cell_indices = std::span(parent_cell_index_storage).first(original_rhs.size());
  std::iota(parent_cell_indices.begin(), parent_cell_indices.end(), std::size_t{0});
  for (const FaceLink& face : faces) {
    if (face.conductance > 0.0 && IsInternal(face)) {
      const std::size_t tail_root_cell_index =
          FindComponentRootCellIndex(parent_cell_indices, static_cast<std::size_t>(face.tail));
      const std::size_t head_root_cell_index =
          FindComponentRootCellIndex(parent_cell_indices, static_cast<std::size_t>(face.head));
      parent_cell_indices[std::max(tail_root_cell_index, head_root_cell_index)] =
          std::min(tail_root_cell_index, head_root_cell_index);
    }
  }

  struct ComponentBalance {
    bool has_pressure_anchor = false;
    double rhs_sum = 0.0;
    double absolute_rhs_sum = 0.0;
  };
  std::array<ComponentBalance, kMaxPressureCells> component_balances;
  for (std::size_t cell_index = 0; cell_index < original_rhs.size(); ++cell_index) {
    const std::size_t component_root_cell_index = FindComponentRootCellIndex(parent_cell_indices, cell_index);
    parent_cell_indices[cell_index] = component_root_cell_index;
    component_balances[component_root_cell_index].has_pressure_anchor =
        component_balances[component_root_cell_index].has_pressure_anchor || cell_has_pressure_anchor[cell_index];
  }
  for (std::size_t cell_index = 0; cell_index < original_rhs.size(); ++cell_index) {
    auto& component_balance = component_balances[parent_cell_indices[cell_index]];
    if (!component_balance.has_pressure_anchor) {
      component_balance.rhs_sum += original_rhs[cell_index];
      component_balance.absolute_rhs_sum += std::abs(original_rhs[cell_index]);
    }
  }

  std::size_t gauge_count = 0;
  for (std::size_t component_root_cell_index = 0; component_root_cell_index < original_rhs.size();
       ++component_root_cell_index) {
    const auto& component_balance = component_balances[component_root_cell_index];
    if (parent_cell_indices[component_root_cell_index] != component_root_cell_index
        || component_balance.has_pressure_anchor) {
      continue;
    }
    if (!std::isfinite(component_balance.rhs_sum) || !std::isfinite(component_balance.absolute_rhs_sum)) {
      return absl::OutOfRangeError("pressure component balance is outside the representable range");
    }
    const double rhs_compatibility_scale =
        std::max(component_balance.absolute_rhs_sum, std::numeric_limits<double>::min());
    if (std::abs(component_balance.rhs_sum) > kCompatibilityTolerance * rhs_compatibility_scale) {
      return absl::FailedPreconditionError("incompatible Neumann pressure component");
    }
    gauge_cell_indices[gauge_count++] = component_root_cell_index;
  }
  return gauge_count;
}

// A zero pin removes both the row and column to preserve symmetry. Keep the
// original system intact so its residual includes the removed equation.
absl::Status SolveGaugedSystem(std::size_t cell_count, std::size_t gauge_count, PressureWorkspace& workspace) {
  auto matrix = std::span(workspace.solve_matrix).first(cell_count * cell_count);
  auto rhs = std::span(workspace.solve_rhs).first(cell_count);
  std::ranges::copy(std::span(workspace.original_matrix).first(matrix.size()), matrix.begin());
  std::ranges::copy(std::span(workspace.original_rhs).first(cell_count), rhs.begin());
  for (const std::size_t gauge : std::span(workspace.gauge_cell_indices).first(gauge_count)) {
    std::ranges::fill(matrix.subspan(gauge * cell_count, cell_count), 0.0);
    for (std::size_t row = 0; row < cell_count; ++row) {
      matrix[row * cell_count + gauge] = 0.0;
    }
    matrix[gauge * cell_count + gauge] = 1.0;
    rhs[gauge] = 0.0;
  }
  auto pressure = std::span(workspace.cell_pressures).first(cell_count);
  return SolveSmallSpd(matrix, rhs, pressure, workspace.solve_workspace);
}

// Evaluate the correction with the same orientation as R^T p. Sum the corrected
// fluid and wall flux independently of the original system for the physical defect.
absl::Status CorrectVelocityAndAccumulateFlux(const PressureInputsView& inputs, PressureWorkspace& workspace) {
  // Reuse defect storage for the integrated flux until all corrected faces have contributed.
  auto corrected_outward_flux = std::span(workspace.cell_constraint_defects).first(inputs.cell_fluid_volumes.size());
  std::ranges::fill(corrected_outward_flux, 0.0);
  for (std::size_t face_index = 0; face_index < inputs.predicted_face_velocities.size(); ++face_index) {
    const FaceLink& face = workspace.faces[face_index];
    double corrected_face_velocity = inputs.predicted_face_velocities[face_index];
    if (face.has_pressure_coupling) {
      const double tail_pressure = face.tail >= 0 ? workspace.cell_pressures[static_cast<std::size_t>(face.tail)] : 0.0;
      const double head_pressure = face.head >= 0 ? workspace.cell_pressures[static_cast<std::size_t>(face.head)] : 0.0;
      const double pressure_difference = tail_pressure - head_pressure + inputs.face_pressure_jumps[face_index];
      corrected_face_velocity +=
          face.pressure_to_velocity_scale * (pressure_difference - face.signed_boundary_pressure);
    }
    if (!std::isfinite(corrected_face_velocity)) {
      return absl::OutOfRangeError("corrected pressure velocity is outside the representable range");
    }
    workspace.corrected_face_velocities[face_index] = corrected_face_velocity;
    const double flux = IntegratedFaceFlux(face, corrected_face_velocity, inputs.wall_face_velocities[face_index]);
    AccumulateOutwardFaceFlux(face, flux, corrected_outward_flux);
  }
  return absl::OkStatus();
}

absl::Status EvaluateResidualAndDefect(const PressureInputsView& inputs, PressureWorkspace& workspace) {
  const std::size_t cell_count = inputs.cell_fluid_volumes.size();
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    double matrix_pressure = 0.0;
    for (std::size_t column = 0; column < cell_count; ++column) {
      const double pressure_coefficient = workspace.original_matrix[cell * cell_count + column];
      matrix_pressure = std::fma(pressure_coefficient, workspace.cell_pressures[column], matrix_pressure);
    }
    const double residual = workspace.original_rhs[cell] - matrix_pressure;

    const double bulk_modulus_times_dt = workspace.bulk_modulus_times_dt[cell];
    double compliance_defect = 0.0;
    if (std::isfinite(bulk_modulus_times_dt)) {
      compliance_defect =
          (workspace.cell_pressures[cell] - inputs.previous_cell_pressures[cell]) / bulk_modulus_times_dt;
    }
    const double corrected_outward_flux = workspace.cell_constraint_defects[cell];
    const double corrected_divergence = corrected_outward_flux / inputs.cell_fluid_volumes[cell];
    const double defect = corrected_divergence - inputs.target_cell_divergence[cell] + compliance_defect;
    if (!std::isfinite(residual) || !std::isfinite(defect)) {
      return absl::OutOfRangeError("pressure residual or constraint defect is outside the representable range");
    }
    workspace.original_residuals[cell] = residual;
    workspace.cell_constraint_defects[cell] = defect;
  }
  return absl::OkStatus();
}

void PublishOutputs(const PressureWorkspace& workspace, std::size_t gauge_count, PressureOutputsView outputs) {
  const std::size_t cell_count = outputs.cell_pressures.size();
  std::ranges::copy(std::span(workspace.cell_pressures).first(cell_count), outputs.cell_pressures.begin());
  std::ranges::copy(
      std::span(workspace.corrected_face_velocities).first(outputs.corrected_face_velocities.size()),
      outputs.corrected_face_velocities.begin()
  );
  std::ranges::copy(
      std::span(workspace.original_matrix).first(outputs.original_matrix.size()),
      outputs.original_matrix.begin()
  );
  std::ranges::copy(std::span(workspace.original_rhs).first(cell_count), outputs.original_rhs.begin());
  std::ranges::copy(std::span(workspace.original_residuals).first(cell_count), outputs.original_residuals.begin());
  std::ranges::copy(
      std::span(workspace.cell_constraint_defects).first(cell_count),
      outputs.cell_constraint_defects.begin()
  );
  std::ranges::copy(std::span(workspace.gauge_cell_indices).first(gauge_count), outputs.gauge_cell_indices.begin());
  outputs.gauge_count = gauge_count;
}

}  // namespace

absl::Status
SolvePressure2D(const PressureInputsView& inputs, PressureOutputsView outputs, PressureWorkspace& workspace) {
  const auto cell_count = CellCount(inputs.grid);
  if (!cell_count.ok()) {
    return cell_count.status();
  }
  if (*cell_count > kMaxPressureCells) {
    return absl::UnimplementedError("tiny pressure oracle supports at most 64 cells");
  }
  const auto face_count = FaceCount(inputs.grid);
  if (!face_count.ok()) {
    return face_count.status();
  }
  if (!IsFinitePositive(inputs.dt)) {
    return absl::InvalidArgumentError("pressure dt must be finite and positive");
  }
  if (const auto status = ValidatePressureViews(inputs, outputs, workspace, *cell_count, *face_count); !status.ok()) {
    return status;
  }
  if (const auto status = ValidatePressureCoefficients(inputs); !status.ok()) {
    return status;
  }

  std::array<bool, kMaxPressureCells> anchored_cells{};
  auto anchors = std::span(anchored_cells).first(*cell_count);
  if (const auto status = AssembleOriginalSystem(inputs, anchors, workspace); !status.ok()) {
    return status;
  }
  const auto gauge_count = SelectGauges(
      std::span(workspace.faces).first(*face_count),
      anchors,
      std::span(workspace.original_rhs).first(*cell_count),
      workspace.gauge_cell_indices
  );
  if (!gauge_count.ok()) {
    return gauge_count.status();
  }
  if (const auto status = SolveGaugedSystem(*cell_count, *gauge_count, workspace); !status.ok()) {
    return status;
  }
  if (const auto status = CorrectVelocityAndAccumulateFlux(inputs, workspace); !status.ok()) {
    return status;
  }
  if (const auto status = EvaluateResidualAndDefect(inputs, workspace); !status.ok()) {
    return status;
  }
  PublishOutputs(workspace, *gauge_count, outputs);
  return absl::OkStatus();
}

}  // namespace fluidity
