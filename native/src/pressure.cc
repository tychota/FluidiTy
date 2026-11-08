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
          inputs.velocity.size(),
          inputs.aperture.size(),
          inputs.mobility.size(),
          inputs.distance.size(),
          inputs.wall_velocity.size(),
          inputs.dirichlet_mask.size(),
          inputs.dirichlet_pressure.size(),
          inputs.jump.size(),
          outputs.velocity.size(),
      }
  );
  if (!face_extents.ok()) {
    return face_extents;
  }
  const auto cell_extents = CheckExtents(
      cell_count,
      {
          inputs.source.size(),
          inputs.fluid_volume.size(),
          inputs.bulk_modulus.size(),
          inputs.previous_pressure.size(),
          outputs.pressure.size(),
          outputs.rhs.size(),
          outputs.residual.size(),
          outputs.constraint_defect.size(),
          outputs.gauges.size(),
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
          std::as_bytes(outputs.pressure),
          std::as_bytes(outputs.velocity),
          std::as_bytes(outputs.original_matrix),
          std::as_bytes(outputs.rhs),
          std::as_bytes(outputs.residual),
          std::as_bytes(outputs.constraint_defect),
          std::as_bytes(outputs.gauges),
          std::as_bytes(std::span(&outputs.gauge_count, 1)),
          std::as_bytes(std::span(&workspace, 1)),
      },
      {
          std::as_bytes(std::span(&inputs, 1)),
          std::as_bytes(inputs.velocity),
          std::as_bytes(inputs.aperture),
          std::as_bytes(inputs.mobility),
          std::as_bytes(inputs.distance),
          std::as_bytes(inputs.wall_velocity),
          std::as_bytes(inputs.dirichlet_mask),
          std::as_bytes(inputs.dirichlet_pressure),
          std::as_bytes(inputs.jump),
          std::as_bytes(inputs.source),
          std::as_bytes(inputs.fluid_volume),
          std::as_bytes(inputs.bulk_modulus),
          std::as_bytes(inputs.previous_pressure),
      }
  );
}

absl::Status ValidatePressureCoefficients(const PressureInputsView& inputs) {
  const std::array finite_fields{
      inputs.velocity,
      inputs.wall_velocity,
      inputs.dirichlet_pressure,
      inputs.jump,
      inputs.source,
      inputs.previous_pressure,
  };
  for (const auto field : finite_fields) {
    if (!std::ranges::all_of(field, IsFinite)) {
      return absl::InvalidArgumentError("pressure velocity, pressure, jump and source fields must be finite");
    }
  }
  const auto valid_aperture = [](double aperture) {
    return aperture >= 0.0 && aperture <= 1.0;
  };
  if (!std::ranges::all_of(inputs.aperture, valid_aperture)) {
    return absl::InvalidArgumentError("pressure aperture must be in [0, 1]");
  }
  for (const auto field : {inputs.mobility, inputs.distance, inputs.fluid_volume}) {
    if (!std::ranges::all_of(field, IsFinitePositive)) {
      return absl::InvalidArgumentError("pressure mobility, distance and fluid volume must be finite and positive");
    }
  }
  const auto valid_bulk_modulus = [](double modulus) {
    return modulus > 0.0;
  };
  if (!std::ranges::all_of(inputs.bulk_modulus, valid_bulk_modulus)) {
    return absl::InvalidArgumentError("pressure bulk modulus must be positive");
  }
  const auto binary_mask = [](std::uint8_t value) {
    return value <= 1;
  };
  if (!std::ranges::all_of(inputs.dirichlet_mask, binary_mask)) {
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
  const double aperture = inputs.aperture[face_index];
  FaceLink face{
      .tail = geometry->tail,
      .head = geometry->head,
      .open_area = geometry->area * aperture,
      .wall_area = 0.0,
      .velocity_scale = 0.0,
      .conductance = 0.0,
      .signed_boundary_pressure = 0.0,
      .active = false,
  };
  const bool is_dirichlet = inputs.dirichlet_mask[face_index] == 1;
  if (is_dirichlet && IsInternal(face)) {
    return absl::InvalidArgumentError("pressure Dirichlet face must be on the grid boundary");
  }
  if (is_dirichlet && inputs.jump[face_index] != 0.0) {
    return absl::InvalidArgumentError("pressure jump must be zero on a Dirichlet face");
  }

  face.wall_area = geometry->area - face.open_area;
  const bool has_pressure_coupling = IsInternal(face) || is_dirichlet;
  face.active = aperture > 0.0 && has_pressure_coupling;
  if (!face.active) {
    return face;
  }
  if (face.open_area == 0.0) {
    return absl::OutOfRangeError("active pressure open area rounded to zero");
  }
  const double mobility = inputs.mobility[face_index];
  const double distance = inputs.distance[face_index];
  face.velocity_scale = MultiplyDividePositive(inputs.dt, mobility, distance);
  if (!IsFinitePositive(face.velocity_scale)) {
    return absl::OutOfRangeError("pressure velocity scale is outside the positive representable range");
  }
  // Use the correction coefficient in the matrix so both phases share the face response.
  face.conductance = face.open_area * face.velocity_scale;
  if (!IsPositiveNormal(face.conductance)) {
    return absl::OutOfRangeError("active pressure conductance must be a positive normal double");
  }
  if (is_dirichlet) {
    // The missing pressure contributes with the opposite incidence of the adjacent cell.
    const double outward_sign = face.tail >= 0 ? 1.0 : -1.0;
    face.signed_boundary_pressure = outward_sign * inputs.dirichlet_pressure[face_index];
  }
  return face;
}

double IntegratedFaceFlux(const FaceLink& face, double fluid_velocity, double wall_velocity) {
  return face.open_area * fluid_velocity + face.wall_area * wall_velocity;
}

void AccumulateOutward(const FaceLink& face, double face_value, std::span<double> cell_values) {
  if (face.tail >= 0) {
    cell_values[static_cast<std::size_t>(face.tail)] += face_value;
  }
  if (face.head >= 0) {
    cell_values[static_cast<std::size_t>(face.head)] -= face_value;
  }
}

// Assemble A = R diag(k) R^T + C and b in integrated flux units.
absl::Status
AssembleOriginalSystem(const PressureInputsView& inputs, std::span<bool> anchored_cells, PressureWorkspace& workspace) {
  const std::size_t cell_count = anchored_cells.size();
  auto matrix = std::span(workspace.original_matrix).first(cell_count * cell_count);
  auto rhs = std::span(workspace.rhs).first(cell_count);
  std::ranges::fill(matrix, 0.0);

  // Keep the three incidence products separate. Balance source and predicted flux
  // before adding the pressure lifts so cancellation preserves small contributions.
  std::array<double, kMaxPressureCells> predicted_outward_flux{};
  std::array<double, kMaxPressureCells> jump_flux_lift{};
  std::array<double, kMaxPressureCells> dirichlet_flux_lift{};

  for (std::size_t face_index = 0; face_index < inputs.velocity.size(); ++face_index) {
    const auto face = BuildFaceLink(inputs, face_index);
    if (!face.ok()) {
      return face.status();
    }
    workspace.faces[face_index] = *face;
    const double conductance = face->conductance;
    const double flux = IntegratedFaceFlux(*face, inputs.velocity[face_index], inputs.wall_velocity[face_index]);
    AccumulateOutward(*face, flux, predicted_outward_flux);
    AccumulateOutward(*face, conductance * inputs.jump[face_index], jump_flux_lift);
    // A Dirichlet column has only one incidence, which squares its sign in the lift.
    AccumulateOutward(*face, conductance * face->signed_boundary_pressure, dirichlet_flux_lift);

    if (conductance == 0.0) {
      continue;
    }
    if (IsInternal(*face)) {
      const auto tail = static_cast<std::size_t>(face->tail);
      const auto head = static_cast<std::size_t>(face->head);
      matrix[tail * cell_count + tail] += conductance;
      matrix[head * cell_count + head] += conductance;
      matrix[tail * cell_count + head] -= conductance;
      matrix[head * cell_count + tail] -= conductance;
    } else {
      const auto cell = static_cast<std::size_t>(face->tail >= 0 ? face->tail : face->head);
      matrix[cell * cell_count + cell] += conductance;
      anchored_cells[cell] = true;
    }
  }

  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    const double bulk_modulus = inputs.bulk_modulus[cell];
    double bulk_time = std::numeric_limits<double>::infinity();
    double compliance = 0.0;
    if (std::isfinite(bulk_modulus)) {
      bulk_time = bulk_modulus * inputs.dt;
      if (!IsFinitePositive(bulk_time)) {
        return absl::OutOfRangeError("pressure bulk modulus times dt is outside the representable range");
      }
      compliance = inputs.fluid_volume[cell] / bulk_time;
      if (!IsPositiveNormal(compliance)) {
        return absl::OutOfRangeError("finite-modulus pressure compliance must be a positive normal double");
      }
    }
    workspace.bulk_time[cell] = bulk_time;
    anchored_cells[cell] = anchored_cells[cell] || compliance > 0.0;
    matrix[cell * cell_count + cell] += compliance;
    const double integrated_source = inputs.fluid_volume[cell] * inputs.source[cell];
    rhs[cell] = integrated_source - predicted_outward_flux[cell] - jump_flux_lift[cell] + dirichlet_flux_lift[cell]
        + compliance * inputs.previous_pressure[cell];
  }

  if (!std::ranges::all_of(matrix, IsFinite) || !std::ranges::all_of(rhs, IsFinite)) {
    return absl::OutOfRangeError("pressure assembly is outside the representable range");
  }
  return absl::OkStatus();
}

std::size_t FindComponent(std::span<const std::size_t> parents, std::size_t cell) {
  while (parents[cell] != cell) {
    cell = parents[cell];
  }
  return cell;
}

// Only positive internal conductance connects cells. Union toward the smaller root
// so each unanchored component's root is also its deterministic gauge index.
absl::StatusOr<std::size_t> SelectGauges(
    std::span<const FaceLink> faces,
    std::span<const bool> anchored_cells,
    std::span<const double> rhs,
    std::span<std::size_t> gauges
) {
  std::array<std::size_t, kMaxPressureCells> parents;
  const auto components = std::span(parents).first(rhs.size());
  std::iota(components.begin(), components.end(), std::size_t{0});
  for (const FaceLink& face : faces) {
    if (face.conductance > 0.0 && IsInternal(face)) {
      const std::size_t tail_root = FindComponent(components, static_cast<std::size_t>(face.tail));
      const std::size_t head_root = FindComponent(components, static_cast<std::size_t>(face.head));
      components[std::max(tail_root, head_root)] = std::min(tail_root, head_root);
    }
  }

  struct ComponentBalance {
    bool anchored = false;
    double net_rhs = 0.0;
    double rhs_norm = 0.0;
  };
  std::array<ComponentBalance, kMaxPressureCells> balances;
  for (std::size_t cell = 0; cell < rhs.size(); ++cell) {
    const std::size_t root = FindComponent(components, cell);
    components[cell] = root;
    balances[root].anchored = balances[root].anchored || anchored_cells[cell];
  }
  for (std::size_t cell = 0; cell < rhs.size(); ++cell) {
    auto& balance = balances[components[cell]];
    if (!balance.anchored) {
      balance.net_rhs += rhs[cell];
      balance.rhs_norm += std::abs(rhs[cell]);
    }
  }

  std::size_t gauge_count = 0;
  for (std::size_t root = 0; root < rhs.size(); ++root) {
    const auto& balance = balances[root];
    if (components[root] != root || balance.anchored) {
      continue;
    }
    if (!std::isfinite(balance.net_rhs) || !std::isfinite(balance.rhs_norm)) {
      return absl::OutOfRangeError("pressure component balance is outside the representable range");
    }
    const double scale = std::max(balance.rhs_norm, std::numeric_limits<double>::min());
    if (std::abs(balance.net_rhs) > kCompatibilityTolerance * scale) {
      return absl::FailedPreconditionError("incompatible Neumann pressure component");
    }
    gauges[gauge_count++] = root;
  }
  return gauge_count;
}

// A zero pin removes both the row and column to preserve symmetry. Keep the
// original system intact so its residual includes the removed equation.
absl::Status SolveGaugedSystem(std::size_t cell_count, std::size_t gauge_count, PressureWorkspace& workspace) {
  auto matrix = std::span(workspace.solve_matrix).first(cell_count * cell_count);
  auto rhs = std::span(workspace.solve_rhs).first(cell_count);
  std::ranges::copy(std::span(workspace.original_matrix).first(matrix.size()), matrix.begin());
  std::ranges::copy(std::span(workspace.rhs).first(cell_count), rhs.begin());
  for (const std::size_t gauge : std::span(workspace.gauges).first(gauge_count)) {
    std::ranges::fill(matrix.subspan(gauge * cell_count, cell_count), 0.0);
    for (std::size_t row = 0; row < cell_count; ++row) {
      matrix[row * cell_count + gauge] = 0.0;
    }
    matrix[gauge * cell_count + gauge] = 1.0;
    rhs[gauge] = 0.0;
  }
  auto pressure = std::span(workspace.pressure).first(cell_count);
  return SolveSmallSpd(matrix, rhs, pressure, workspace.solve_workspace);
}

// Evaluate the correction with the same orientation as R^T p. Sum the corrected
// fluid and wall flux independently of the original system for the physical defect.
absl::Status CorrectVelocityAndAccumulateFlux(const PressureInputsView& inputs, PressureWorkspace& workspace) {
  // Reuse defect storage for the integrated flux until all corrected faces have contributed.
  auto corrected_outward_flux = std::span(workspace.constraint_defect).first(inputs.fluid_volume.size());
  std::ranges::fill(corrected_outward_flux, 0.0);
  for (std::size_t face_index = 0; face_index < inputs.velocity.size(); ++face_index) {
    const FaceLink& face = workspace.faces[face_index];
    double velocity = inputs.velocity[face_index];
    if (face.active) {
      const double tail_pressure = face.tail >= 0 ? workspace.pressure[static_cast<std::size_t>(face.tail)] : 0.0;
      const double head_pressure = face.head >= 0 ? workspace.pressure[static_cast<std::size_t>(face.head)] : 0.0;
      const double pressure_difference = tail_pressure - head_pressure + inputs.jump[face_index];
      velocity += face.velocity_scale * (pressure_difference - face.signed_boundary_pressure);
    }
    if (!std::isfinite(velocity)) {
      return absl::OutOfRangeError("corrected pressure velocity is outside the representable range");
    }
    workspace.velocity[face_index] = velocity;
    const double flux = IntegratedFaceFlux(face, velocity, inputs.wall_velocity[face_index]);
    AccumulateOutward(face, flux, corrected_outward_flux);
  }
  return absl::OkStatus();
}

absl::Status EvaluateResidualAndDefect(const PressureInputsView& inputs, PressureWorkspace& workspace) {
  const std::size_t cell_count = inputs.fluid_volume.size();
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    double matrix_pressure = 0.0;
    for (std::size_t column = 0; column < cell_count; ++column) {
      const double pressure_coefficient = workspace.original_matrix[cell * cell_count + column];
      matrix_pressure = std::fma(pressure_coefficient, workspace.pressure[column], matrix_pressure);
    }
    const double residual = workspace.rhs[cell] - matrix_pressure;

    const double bulk_time = workspace.bulk_time[cell];
    double compliance_defect = 0.0;
    if (std::isfinite(bulk_time)) {
      compliance_defect = (workspace.pressure[cell] - inputs.previous_pressure[cell]) / bulk_time;
    }
    const double corrected_outward_flux = workspace.constraint_defect[cell];
    const double corrected_divergence = corrected_outward_flux / inputs.fluid_volume[cell];
    const double defect = corrected_divergence - inputs.source[cell] + compliance_defect;
    if (!std::isfinite(residual) || !std::isfinite(defect)) {
      return absl::OutOfRangeError("pressure residual or constraint defect is outside the representable range");
    }
    workspace.residual[cell] = residual;
    workspace.constraint_defect[cell] = defect;
  }
  return absl::OkStatus();
}

void PublishOutputs(const PressureWorkspace& workspace, std::size_t gauge_count, PressureOutputsView outputs) {
  const std::size_t cell_count = outputs.pressure.size();
  std::ranges::copy(std::span(workspace.pressure).first(cell_count), outputs.pressure.begin());
  std::ranges::copy(std::span(workspace.velocity).first(outputs.velocity.size()), outputs.velocity.begin());
  std::ranges::copy(
      std::span(workspace.original_matrix).first(outputs.original_matrix.size()),
      outputs.original_matrix.begin()
  );
  std::ranges::copy(std::span(workspace.rhs).first(cell_count), outputs.rhs.begin());
  std::ranges::copy(std::span(workspace.residual).first(cell_count), outputs.residual.begin());
  std::ranges::copy(std::span(workspace.constraint_defect).first(cell_count), outputs.constraint_defect.begin());
  std::ranges::copy(std::span(workspace.gauges).first(gauge_count), outputs.gauges.begin());
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
      std::span(workspace.rhs).first(*cell_count),
      workspace.gauges
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
