#include "fluidity/host/viscosity.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

#include "fluidity/host/views.h"

namespace fluidity {
namespace {

// Keep the exponent separate so a representable final product can survive
// large or small intermediate factors and can share a power-of-two shift.
struct BinaryProduct {
  double fraction;
  int exponent;

  double Value(int exponent_shift = 0) const {
    return std::scalbn(fraction, exponent + exponent_shift);
  }
};

BinaryProduct FactorProduct(double first_factor, double second_factor, double third_factor, double divisor = 1.0) {
  if (first_factor == 0.0 || second_factor == 0.0 || third_factor == 0.0) {
    return {0.0, 0};
  }
  int first_exponent;
  int second_exponent;
  int third_exponent;
  int divisor_exponent;
  const double first_fraction = std::frexp(first_factor, &first_exponent);
  const double second_fraction = std::frexp(second_factor, &second_exponent);
  const double third_fraction = std::frexp(third_factor, &third_exponent);
  const double divisor_fraction = std::frexp(divisor, &divisor_exponent);
  const double fraction = (first_fraction * (second_fraction * third_fraction)) / divisor_fraction;
  const int exponent = first_exponent + second_exponent + third_exponent - divisor_exponent;
  return {fraction, exponent};
}

double ScaledProduct(double first_factor, double second_factor, double third_factor, double divisor = 1.0) {
  return FactorProduct(first_factor, second_factor, third_factor, divisor).Value();
}

auto ViscosityCellKey(CellIndex3D cell) {
  return std::tuple(cell.z, cell.y, cell.x);
}

CellIndex3D NextViscosityCell(CellIndex3D cell, std::size_t axis) {
  std::array coordinates{&cell.x, &cell.y, &cell.z};
  ++*coordinates[axis];
  return cell;
}

FaceIndex3D ViscosityFace(std::size_t axis, CellIndex3D cell) {
  return {.axis = static_cast<FaceNormalAxis>(axis), .lattice_index = {cell.x, cell.y, cell.z}};
}

const MacViscosityCell3D* FindViscosityCell(std::span<const MacViscosityCell3D> cells, CellIndex3D index) {
  const auto key = ViscosityCellKey(index);
  const auto found = std::ranges::lower_bound(cells, key, {}, [](const MacViscosityCell3D& cell) {
    return ViscosityCellKey(cell.cell_index);
  });
  if (found == cells.end() || ViscosityCellKey(found->cell_index) != key) {
    return nullptr;
  }
  return &*found;
}

// Scaling by the smallest viscosity keeps the reciprocal reduction bounded.
double HarmonicEdgeViscosity(std::array<double, 4> viscosities) {
  const double minimum = *std::ranges::min_element(viscosities);
  if (minimum == 0.0) {
    return 0.0;
  }
  double reciprocal_sum = 0.0;
  for (const double viscosity : viscosities) {
    reciprocal_sum += minimum / viscosity;
  }
  return minimum * (4.0 / reciprocal_sum);
}

double StrainWeight3D(double viscosity, double spacing, int multiplicity_exponent) {
  int spacing_exponent;
  const double spacing_fraction = std::frexp(spacing, &spacing_exponent);
  const BinaryProduct weighted_area = FactorProduct(viscosity, spacing, spacing);
  return std::scalbn(
      weighted_area.fraction * spacing_fraction,
      weighted_area.exponent + spacing_exponent + multiplicity_exponent
  );
}

// Every row is assembled in a private vector. A failed lookup cannot expose
// a stencil with an unowned face slot.
absl::Status SetStrainTerms(
    const SparseMacGridView3D& grid,
    std::span<const FaceIndex3D> faces,
    double difference_coefficient,
    MacStrainRow3D& row
) {
  for (std::size_t term_index = 0; term_index < faces.size(); ++term_index) {
    const auto slot = grid.FaceSlot(faces[term_index]);
    if (!slot.ok()) {
      return slot.status();
    }
    const double coefficient = term_index % 2 == 0 ? -difference_coefficient : difference_coefficient;
    row.terms[term_index] = {.face_slot = *slot, .coefficient = coefficient};
  }
  row.term_count = faces.size();
  return absl::OkStatus();
}

struct BinaryExponentRange {
  void Include(BinaryProduct term) {
    if (term.fraction != 0.0) {
      const int term_exponent = term.exponent + std::ilogb(term.fraction);
      min_exponent = std::min(min_exponent, term_exponent);
      max_exponent = std::max(max_exponent, term_exponent);
    }
  }

  bool Empty() const {
    return min_exponent > max_exponent;
  }

  int min_exponent = std::numeric_limits<int>::max();
  int max_exponent = std::numeric_limits<int>::min();
};

absl::Status
ValidateStepViews(const ViscosityInputsView& inputs, ViscosityOutputsView outputs, ViscosityWorkspace& workspace) {
  const std::size_t columns = inputs.face_velocities.size();
  if (columns == 0) {
    return absl::InvalidArgumentError("viscosity requires at least one velocity unknown");
  }
  if (columns > kMaxDenseUnknowns) {
    return absl::UnimplementedError("viscosity supports at most 64 velocity unknowns");
  }
  const auto coefficients = CheckedExtentProduct(inputs.strain_row_count, columns);
  if (!coefficients.ok()) {
    return coefficients.status();
  }
  const auto coefficient_bytes = CheckedExtentProduct(*coefficients, sizeof(double));
  if (!coefficient_bytes.ok()) {
    return coefficient_bytes.status();
  }
  const auto column_extents =
      CheckExtents(columns, {inputs.face_inertias.size(), outputs.updated_face_velocities.size()});
  if (!column_extents.ok()) {
    return column_extents;
  }
  const auto row_extents =
      CheckExtents(inputs.strain_row_count, {inputs.strain_quadrature_weights.size(), inputs.boundary_strain.size()});
  if (!row_extents.ok()) {
    return row_extents;
  }
  const auto strain_extent = ValidateExtent(inputs.strain_operator.size(), *coefficients);
  if (!strain_extent.ok()) {
    return strain_extent;
  }

  return CheckWritableViews(
      {std::as_bytes(outputs.updated_face_velocities),
       std::as_bytes(std::span(&outputs.diagnostics, 1)),
       std::as_bytes(std::span(&workspace, 1))},
      {std::as_bytes(std::span(&inputs, 1)),
       std::as_bytes(inputs.face_velocities),
       std::as_bytes(inputs.face_inertias),
       std::as_bytes(inputs.strain_operator),
       std::as_bytes(inputs.strain_quadrature_weights),
       std::as_bytes(inputs.boundary_strain)}
  );
}

absl::Status ValidateStepParameters(const ViscosityInputsView& inputs) {
  if (!std::isfinite(inputs.dt) || inputs.dt < 0.0) {
    return absl::InvalidArgumentError("viscosity dt must be finite and nonnegative");
  }
  if (!std::isfinite(inputs.explicit_stability_margin) || inputs.explicit_stability_margin < 0.0
      || inputs.explicit_stability_margin >= 1.0) {
    return absl::InvalidArgumentError("viscosity safety must be in [0, 1)");
  }
  switch (inputs.mode) {
    case ViscosityMode::kAuto:
    case ViscosityMode::kExplicit:
    case ViscosityMode::kImplicit:
      break;
    default:
      return absl::InvalidArgumentError("invalid viscosity mode");
  }

  for (std::size_t face = 0; face < inputs.face_velocities.size(); ++face) {
    if (!std::isfinite(inputs.face_velocities[face])) {
      return absl::InvalidArgumentError("face velocity must be finite");
    }
    if (!std::isfinite(inputs.face_inertias[face]) || inputs.face_inertias[face] <= 0.0) {
      return absl::InvalidArgumentError("face inertia must be finite and positive");
    }
  }
  for (std::size_t row = 0; row < inputs.strain_row_count; ++row) {
    if (!std::isfinite(inputs.strain_quadrature_weights[row]) || inputs.strain_quadrature_weights[row] < 0.0) {
      return absl::InvalidArgumentError("strain quadrature must be finite and nonnegative");
    }
    if (!std::isfinite(inputs.boundary_strain[row])) {
      return absl::InvalidArgumentError("wall strain must be finite");
    }
  }
  for (const double coefficient : inputs.strain_operator) {
    if (!std::isfinite(coefficient)) {
      return absl::InvalidArgumentError("strain operator must be finite");
    }
  }
  return absl::OkStatus();
}

// Add one strain row's weighted outer product to the lower triangle and its
// prescribed wall strain contribution to the load.
absl::Status AccumulateStrainRow(
    std::span<const double> strain_coefficients,
    double quadrature,
    double wall_strain,
    std::span<double> stiffness,
    std::span<double> wall_load
) {
  const std::size_t columns = strain_coefficients.size();
  for (std::size_t face_row = 0; face_row < columns; ++face_row) {
    const double row_coefficient = strain_coefficients[face_row];
    if (row_coefficient == 0.0) {
      continue;
    }
    if (wall_strain != 0.0) {
      const double wall_load_contribution = ScaledProduct(quadrature, row_coefficient, wall_strain);
      if (!std::isnormal(wall_load_contribution)) {
        return absl::OutOfRangeError("wall load term is outside normal double range");
      }
      wall_load[face_row] += wall_load_contribution;
    }
    for (std::size_t face_column = 0; face_column <= face_row; ++face_column) {
      const double column_coefficient = strain_coefficients[face_column];
      if (column_coefficient == 0.0) {
        continue;
      }
      const double stiffness_coefficient_contribution = ScaledProduct(quadrature, row_coefficient, column_coefficient);
      if (!std::isnormal(stiffness_coefficient_contribution)) {
        return absl::OutOfRangeError("viscous stiffness term is outside normal double range");
      }
      stiffness[face_row * columns + face_column] += stiffness_coefficient_contribution;
    }
  }
  return absl::OkStatus();
}

// The reductions can overflow or cancel to subnormal values even when all
// individual terms were normal. Check the completed entries before mirroring.
absl::Status FinalizeViscousOperator(std::span<double> stiffness, std::span<const double> wall_load) {
  const std::size_t columns = wall_load.size();
  for (std::size_t row = 0; row < columns; ++row) {
    if (wall_load[row] != 0.0 && !std::isnormal(wall_load[row])) {
      return absl::OutOfRangeError("assembled wall load is outside normal double range");
    }
    for (std::size_t column = 0; column <= row; ++column) {
      const double coefficient = stiffness[row * columns + column];
      if (coefficient != 0.0 && !std::isnormal(coefficient)) {
        return absl::OutOfRangeError("assembled viscous stiffness is outside normal double range");
      }
      stiffness[column * columns + row] = coefficient;
    }
  }
  return absl::OkStatus();
}

// Both integrators use the same symmetric E^T W E. Reduce the lower triangle
// and wall load in strain row order before finalizing the dense operator.
absl::Status
AssembleViscousOperator(const ViscosityInputsView& inputs, std::span<double> stiffness, std::span<double> wall_load) {
  const std::size_t columns = inputs.face_velocities.size();
  std::ranges::fill(stiffness, 0.0);
  std::ranges::fill(wall_load, 0.0);
  for (std::size_t strain_row = 0; strain_row < inputs.strain_row_count; ++strain_row) {
    const double quadrature = inputs.strain_quadrature_weights[strain_row];
    if (quadrature == 0.0) {
      continue;
    }
    const auto strain_coefficients = inputs.strain_operator.subspan(strain_row * columns, columns);
    const double wall_strain = inputs.boundary_strain[strain_row];
    const auto accumulated = AccumulateStrainRow(strain_coefficients, quadrature, wall_strain, stiffness, wall_load);
    if (!accumulated.ok()) {
      return accumulated;
    }
  }
  return FinalizeViscousOperator(stiffness, wall_load);
}

double MassScaledStiffness(double coefficient, double row_sqrt_mass, double column_sqrt_mass) {
  return ScaledProduct(coefficient, 1.0 / row_sqrt_mass, 1.0 / column_sqrt_mass);
}

double UpperAdjacent(double value) {
  return std::nextafter(value, std::numeric_limits<double>::infinity());
}

// Bound W E²/M using outward-rounded mantissas. This avoids overflowing E²
// or losing a contribution below the normal range before division by M.
double UpperTraceContribution(double quadrature, double strain_coefficient, double face_inertia) {
  int quadrature_exponent;
  int strain_exponent;
  int inertia_exponent;
  const double quadrature_fraction = std::frexp(quadrature, &quadrature_exponent);
  const double strain_fraction = std::frexp(std::abs(strain_coefficient), &strain_exponent);
  const double inertia_fraction = std::frexp(face_inertia, &inertia_exponent);
  const double squared_strain_fraction = UpperAdjacent(strain_fraction * strain_fraction);
  const double weighted_strain_fraction = UpperAdjacent(quadrature_fraction * squared_strain_fraction);
  const double trace_fraction = UpperAdjacent(weighted_strain_fraction / inertia_fraction);
  const int trace_exponent = quadrature_exponent + 2 * strain_exponent - inertia_exponent;
  return UpperAdjacent(std::scalbn(trace_fraction, trace_exponent));
}

// The exact mass-scaled stiffness is PSD. Its trace bounds its spectrum
// without depending on cancellation or the computed square roots of inertia.
double ComputeSpectralBound(const ViscosityInputsView& inputs) {
  const std::size_t columns = inputs.face_inertias.size();
  double spectral_bound = 0.0;
  for (std::size_t row = 0; row < inputs.strain_row_count; ++row) {
    const double quadrature = inputs.strain_quadrature_weights[row];
    if (quadrature == 0.0) {
      continue;
    }
    for (std::size_t column = 0; column < columns; ++column) {
      const double strain_coefficient = inputs.strain_operator[row * columns + column];
      if (strain_coefficient == 0.0) {
        continue;
      }
      const double trace_contribution =
          UpperTraceContribution(quadrature, strain_coefficient, inputs.face_inertias[column]);
      spectral_bound = UpperAdjacent(spectral_bound + trace_contribution);
    }
  }
  return spectral_bound;
}

absl::Status ApplyForwardEuler(
    const ViscosityInputsView& inputs,
    std::span<const double> stiffness,
    std::span<const double> wall_load,
    std::span<double> updated_face_velocities
) {
  const std::size_t columns = inputs.face_velocities.size();
  for (std::size_t row = 0; row < columns; ++row) {
    const double face_mass = inputs.face_inertias[row];
    double velocity_increment = ScaledProduct(inputs.dt, wall_load[row], 1.0, face_mass);
    // Accumulate in velocity units so an extreme force can still make a
    // representable contribution after scaling by dt and the face inertia.
    for (std::size_t column = 0; column < columns; ++column) {
      const double coefficient = stiffness[row * columns + column];
      const double viscous_increment = ScaledProduct(inputs.dt, coefficient, inputs.face_velocities[column], face_mass);
      velocity_increment -= viscous_increment;
    }
    if (!std::isfinite(velocity_increment)) {
      return absl::OutOfRangeError("explicit viscous increment is nonfinite");
    }
    updated_face_velocities[row] = inputs.face_velocities[row] + velocity_increment;
  }
  return absl::OkStatus();
}

// Assemble I + dt S in place, preserving the symmetry of the mass-scaled system.
// Its largest diagonal sets the scale of attenuation through the solve.
absl::StatusOr<double>
BuildImplicitSystem(double step_duration, std::span<double> system_matrix, std::span<const double> sqrt_mass) {
  const std::size_t columns = sqrt_mass.size();
  double largest_diagonal = 1.0;
  for (std::size_t row = 0; row < columns; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      const double stiffness_coefficient = system_matrix[row * columns + column];
      const double mass_scaled_stiffness_coefficient =
          MassScaledStiffness(stiffness_coefficient, sqrt_mass[row], sqrt_mass[column]);
      const double step_scaled_stiffness_coefficient = step_duration * mass_scaled_stiffness_coefficient;
      if (stiffness_coefficient != 0.0 && !std::isnormal(step_scaled_stiffness_coefficient)) {
        return absl::OutOfRangeError("implicit viscous stiffness is outside normal double range");
      }
      double system_coefficient = step_scaled_stiffness_coefficient;
      if (row == column) {
        system_coefficient += 1.0;
        largest_diagonal = std::max(largest_diagonal, system_coefficient);
      }
      system_matrix[row * columns + column] = system_coefficient;
      system_matrix[column * columns + row] = system_coefficient;
    }
  }
  return largest_diagonal;
}

// Scale all right-hand-side terms and solve coordinates by one power of two.
// Choose the shift from the actual products, including terms below the double
// range before shifting. Bias toward larger RHS values to leave room for decay.
absl::StatusOr<int> ChooseImplicitRhsShift(
    const ViscosityInputsView& inputs,
    std::span<const double> wall_load,
    std::span<const double> sqrt_mass,
    double largest_diagonal
) {
  BinaryExponentRange term_exponents;
  for (std::size_t face = 0; face < sqrt_mass.size(); ++face) {
    term_exponents.Include(FactorProduct(sqrt_mass[face], inputs.face_velocities[face], 1.0));
    term_exponents.Include(FactorProduct(inputs.dt, wall_load[face], 1.0, sqrt_mass[face]));
  }
  if (term_exponents.Empty()) {
    return 0;
  }

  constexpr int kMinNormalExponent = std::numeric_limits<double>::min_exponent - 1;
  // Leave eight exponent bits for additions within a system of at most 64 unknowns.
  constexpr int kMaxRhsTermExponent = std::numeric_limits<double>::max_exponent - 1 - 8;
  const int minimum_rhs_shift = kMinNormalExponent - term_exponents.min_exponent;
  const int maximum_rhs_shift = kMaxRhsTermExponent - term_exponents.max_exponent;
  if (minimum_rhs_shift > maximum_rhs_shift) {
    return absl::OutOfRangeError("implicit right-hand side terms have no common normal scale");
  }
  const int rhs_exponent_midpoint = std::midpoint(term_exponents.min_exponent, term_exponents.max_exponent);
  const int preferred_shift = std::ilogb(largest_diagonal) / 2 - rhs_exponent_midpoint;
  return std::clamp(preferred_shift, minimum_rhs_shift, maximum_rhs_shift);
}

absl::Status StageImplicitRhs(
    const ViscosityInputsView& inputs,
    std::span<const double> wall_load,
    std::span<const double> sqrt_mass,
    int rhs_shift,
    std::span<double> solve_rhs
) {
  for (std::size_t face = 0; face < sqrt_mass.size(); ++face) {
    int sqrt_mass_exponent;
    const double sqrt_mass_fraction = std::frexp(sqrt_mass[face], &sqrt_mass_exponent);
    // The checked product window also keeps this multiplier in range because
    // the mass fraction lies in [0.5, 1). Fusing the sum retains the low bits
    // when the weighted velocity and wall impulse nearly cancel.
    const double exponent_scaled_velocity = std::scalbn(inputs.face_velocities[face], sqrt_mass_exponent + rhs_shift);
    const double scaled_wall_impulse = FactorProduct(inputs.dt, wall_load[face], 1.0, sqrt_mass[face]).Value(rhs_shift);
    solve_rhs[face] = std::fma(sqrt_mass_fraction, exponent_scaled_velocity, scaled_wall_impulse);
    if (solve_rhs[face] != 0.0 && !std::isnormal(solve_rhs[face])) {
      return absl::OutOfRangeError("scaled implicit right-hand side is outside normal double range");
    }
  }
  return absl::OkStatus();
}

// A rounded-to-zero solve component can hide a physical velocity. Check its
// equation in a row-local scale so tiny couplings survive the residual calculation.
bool ZeroComponentSatisfiesEquation(
    std::span<const double> equation,
    double rhs,
    std::span<const double> weighted_velocity
) {
  BinaryExponentRange term_exponents;
  term_exponents.Include({rhs, 0});
  for (std::size_t column = 0; column < equation.size(); ++column) {
    term_exponents.Include(FactorProduct(equation[column], weighted_velocity[column], 1.0));
  }
  if (term_exponents.Empty()) {
    return true;
  }

  const int residual_shift = -term_exponents.max_exponent;
  double scaled_residual = -std::scalbn(rhs, residual_shift);
  for (std::size_t column = 0; column < equation.size(); ++column) {
    scaled_residual += FactorProduct(equation[column], weighted_velocity[column], 1.0).Value(residual_shift);
  }
  const double tolerance = 8.0 * static_cast<double>(equation.size()) * std::numeric_limits<double>::epsilon();
  return std::abs(scaled_residual) <= tolerance;
}

absl::Status RestoreImplicitVelocity(
    std::span<const double> system_matrix,
    std::span<const double> solve_rhs,
    std::span<const double> sqrt_mass,
    int rhs_shift,
    std::span<double> scaled_solution
) {
  const std::size_t face_count = sqrt_mass.size();
  for (std::size_t face_index = 0; face_index < face_count; ++face_index) {
    const double scaled_mass_weighted_velocity = scaled_solution[face_index];
    if (scaled_mass_weighted_velocity == 0.0) {
      const auto system_row = system_matrix.subspan(face_index * face_count, face_count);
      if (!ZeroComponentSatisfiesEquation(system_row, solve_rhs[face_index], scaled_solution)) {
        return absl::OutOfRangeError("implicit solve lost a velocity component at the common scale");
      }
    } else if (!std::isnormal(scaled_mass_weighted_velocity)) {
      return absl::OutOfRangeError("implicit solve component is outside normal double range");
    }
  }

  for (std::size_t face_index = 0; face_index < face_count; ++face_index) {
    const double scaled_mass_weighted_velocity = scaled_solution[face_index];
    const double velocity =
        FactorProduct(scaled_mass_weighted_velocity, 1.0, 1.0, sqrt_mass[face_index]).Value(-rhs_shift);
    if (!std::isfinite(velocity) || (scaled_mass_weighted_velocity != 0.0 && velocity == 0.0)) {
      return absl::OutOfRangeError("implicit face velocity is outside representable double range");
    }
    scaled_solution[face_index] = velocity;
  }
  return absl::OkStatus();
}

// Solve in sqrt(M) velocity coordinates with an additional common power of
// two for range. The matrix is independent of that common scale, so the
// conditioning screen still sees the mass-scaled backward Euler equation.
absl::Status ApplyBackwardEuler(
    const ViscosityInputsView& inputs,
    std::span<double> system_matrix,
    std::span<const double> wall_load,
    std::span<const double> sqrt_mass,
    std::span<double> solve_rhs,
    std::span<double> implicit_solution_storage,
    SmallSolveWorkspace& solve
) {
  const auto largest_diagonal = BuildImplicitSystem(inputs.dt, system_matrix, sqrt_mass);
  if (!largest_diagonal.ok()) {
    return largest_diagonal.status();
  }
  const auto rhs_shift = ChooseImplicitRhsShift(inputs, wall_load, sqrt_mass, *largest_diagonal);
  if (!rhs_shift.ok()) {
    return rhs_shift.status();
  }
  const auto staged = StageImplicitRhs(inputs, wall_load, sqrt_mass, *rhs_shift, solve_rhs);
  if (!staged.ok()) {
    return staged;
  }
  const auto solved = SolveSmallSpd(system_matrix, solve_rhs, implicit_solution_storage, solve);
  if (!solved.ok()) {
    return solved;
  }
  return RestoreImplicitVelocity(system_matrix, solve_rhs, sqrt_mass, *rhs_shift, implicit_solution_storage);
}

absl::Status ComputeEnergyAndWallWork(
    const ViscosityInputsView& inputs,
    std::span<const double> updated_face_velocities,
    std::span<const double> wall_load,
    std::span<const double> sqrt_mass,
    ViscosityDiagnostics& diagnostics
) {
  // Forward Euler takes work and material dissipation at the old velocity, so
  // its energy balance subtracts half the squared mass norm of the increment.
  // Backward Euler uses the new velocity and adds that same increment energy.
  const bool explicit_step = diagnostics.method == ViscosityMethod::kExplicit;
  for (std::size_t face = 0; face < updated_face_velocities.size(); ++face) {
    if (!std::isfinite(updated_face_velocities[face])) {
      return absl::OutOfRangeError("updated viscous velocity is nonfinite");
    }
    const double old_weighted_velocity = sqrt_mass[face] * inputs.face_velocities[face];
    const double new_weighted_velocity = sqrt_mass[face] * updated_face_velocities[face];
    const double velocity_increment = updated_face_velocities[face] - inputs.face_velocities[face];
    const bool unweighted_increment_fits = std::isfinite(velocity_increment);
    // Subtract first to retain small changes. If opposite large velocities
    // overflow the subtraction, apply the mass metric before subtracting.
    const double weighted_increment = unweighted_increment_fits ? sqrt_mass[face] * velocity_increment
                                                                : new_weighted_velocity - old_weighted_velocity;

    const double work_velocity = explicit_step ? inputs.face_velocities[face] : updated_face_velocities[face];

    diagnostics.energy_before = std::fma(0.5 * old_weighted_velocity, old_weighted_velocity, diagnostics.energy_before);
    diagnostics.energy_after = std::fma(0.5 * new_weighted_velocity, new_weighted_velocity, diagnostics.energy_after);
    diagnostics.numerical_dissipation =
        std::fma(0.5 * weighted_increment, weighted_increment, diagnostics.numerical_dissipation);
    diagnostics.wall_work += ScaledProduct(inputs.dt, work_velocity, wall_load[face]);
  }
  if (!std::isfinite(diagnostics.energy_before) || !std::isfinite(diagnostics.energy_after)) {
    return absl::OutOfRangeError("viscosity kinetic energy is nonfinite");
  }
  if (!std::isfinite(diagnostics.wall_work) || !std::isfinite(diagnostics.numerical_dissipation)) {
    return absl::OutOfRangeError("viscosity wall work or increment energy is nonfinite");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<MacStrainExtents> MacStrainExtents2D(const Grid2D& grid) {
  const auto face_count = FaceCount(grid);
  if (!face_count.ok()) {
    return face_count.status();
  }
  if (*face_count > kMaxDenseUnknowns) {
    return absl::UnimplementedError("MAC strain supports at most 64 velocity unknowns");
  }

  // With at most 64 faces, the grid has at most 32 cells. There are two normal
  // rows per cell and fewer shear rows than cells.
  const std::size_t cell_columns = grid.nx;
  const std::size_t cell_rows = grid.ny;
  const std::size_t normal_rows = 2 * cell_columns * cell_rows;
  const std::size_t shear_rows = (cell_columns - 1) * (cell_rows - 1);
  const std::size_t strain_rows = normal_rows + shear_rows;
  const auto coefficients = CheckedExtentProduct(strain_rows, *face_count);
  if (!coefficients.ok()) {
    return coefficients.status();
  }
  return MacStrainExtents{
      .strain_row_count = strain_rows,
      .face_velocity_count = *face_count,
      .coefficient_count = *coefficients,
  };
}

absl::Status AssembleMacStrain2D(
    const Grid2D& grid,
    double dynamic_viscosity,
    std::span<double> strain_operator,
    std::span<double> strain_quadrature_weights
) {
  const auto shape = MacStrainExtents2D(grid);
  if (!shape.ok()) {
    return shape.status();
  }
  if (!std::isfinite(dynamic_viscosity) || dynamic_viscosity < 0.0) {
    return absl::InvalidArgumentError("dynamic viscosity must be finite and nonnegative");
  }
  const auto strain_extent = ValidateExtent(strain_operator.size(), shape->coefficient_count);
  if (!strain_extent.ok()) {
    return strain_extent;
  }
  const auto weight_extent = ValidateExtent(strain_quadrature_weights.size(), shape->strain_row_count);
  if (!weight_extent.ok()) {
    return weight_extent;
  }
  const auto disjoint = CheckWritableViews(
      {std::as_bytes(strain_operator), std::as_bytes(strain_quadrature_weights)},
      {std::as_bytes(std::span(&grid, 1))}
  );
  if (!disjoint.ok()) {
    return disjoint;
  }

  const std::size_t cell_columns = grid.nx;
  const std::size_t cell_rows = grid.ny;
  const std::size_t cell_count = cell_columns * cell_rows;
  const std::size_t normal_row_count = 2 * cell_count;
  const bool has_shear_rows = shape->strain_row_count > normal_row_count;
  const double normal_derivative = 1.0 / grid.h;
  const double shear_derivative = 0.5 * normal_derivative;
  if (!std::isnormal(normal_derivative) || (has_shear_rows && !std::isnormal(shear_derivative))) {
    return absl::OutOfRangeError("MAC strain difference coefficient is outside normal double range");
  }

  // Fold the tensor multiplicity into a spacing factor before multiplication.
  // This also handles a finite mu whose double or quadruple would overflow.
  const double normal_weight = ScaledProduct(dynamic_viscosity, 2.0 * grid.h, grid.h);
  const double shear_weight = has_shear_rows ? ScaledProduct(dynamic_viscosity, 4.0 * grid.h, grid.h) : 0.0;
  if (dynamic_viscosity > 0.0 && !std::isnormal(normal_weight)) {
    return absl::OutOfRangeError("normal strain quadrature is outside normal double range");
  }
  if (dynamic_viscosity > 0.0 && has_shear_rows && !std::isnormal(shear_weight)) {
    return absl::OutOfRangeError("shear strain quadrature is outside normal double range");
  }

  // Every coefficient and index extent has been checked. Fill the caller's
  // dense views directly now that the remaining stencil writes cannot fail.
  std::ranges::fill(strain_operator, 0.0);
  std::ranges::fill(strain_quadrature_weights.first(normal_row_count), normal_weight);
  std::ranges::fill(strain_quadrature_weights.subspan(normal_row_count), shear_weight);

  const std::size_t x_face_stride = cell_columns + 1;
  const std::size_t y_face_offset = x_face_stride * cell_rows;
  for (std::size_t row = 0; row < cell_rows; ++row) {
    for (std::size_t column = 0; column < cell_columns; ++column) {
      const std::size_t cell = row * cell_columns + column;
      const std::size_t left_face = row * x_face_stride + column;
      const std::size_t bottom_face = y_face_offset + cell;
      const std::size_t x_normal_row_offset = cell * shape->face_velocity_count;
      const std::size_t y_normal_row_offset = (cell_count + cell) * shape->face_velocity_count;
      strain_operator[x_normal_row_offset + left_face] = -normal_derivative;
      strain_operator[x_normal_row_offset + left_face + 1] = normal_derivative;
      strain_operator[y_normal_row_offset + bottom_face] = -normal_derivative;
      strain_operator[y_normal_row_offset + bottom_face + cell_columns] = normal_derivative;
    }
  }

  for (std::size_t row = 1; row < cell_rows; ++row) {
    for (std::size_t column = 1; column < cell_columns; ++column) {
      const std::size_t shear_vertex_index = (row - 1) * (cell_columns - 1) + column - 1;
      const std::size_t shear_row_offset = (normal_row_count + shear_vertex_index) * shape->face_velocity_count;
      const std::size_t upper_x_face = row * x_face_stride + column;
      const std::size_t right_y_face = y_face_offset + row * cell_columns + column;
      const std::size_t lower_x_face = upper_x_face - x_face_stride;
      const std::size_t left_y_face = right_y_face - 1;
      strain_operator[shear_row_offset + lower_x_face] = -shear_derivative;
      strain_operator[shear_row_offset + upper_x_face] = shear_derivative;
      strain_operator[shear_row_offset + left_y_face] = -shear_derivative;
      strain_operator[shear_row_offset + right_y_face] = shear_derivative;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<MacStrainRow3D>>
AssembleFullCellMacStrain3D(const SparseMacGridView3D& grid, std::span<const MacViscosityCell3D> cells) {
  std::vector<MacStrainRow3D> rows;
  const auto maximum_rows = CheckedExtentProduct(cells.size(), 6);
  if (!maximum_rows.ok() || *maximum_rows > rows.max_size()) {
    return absl::OutOfRangeError("viscosity strain row extent is unrepresentable");
  }
  for (std::size_t index = 0; index < cells.size(); ++index) {
    const auto& cell = cells[index];
    if (index > 0 && ViscosityCellKey(cells[index - 1].cell_index) >= ViscosityCellKey(cell.cell_index)) {
      return absl::InvalidArgumentError("viscosity cells must be unique and ordered");
    }
    if (!std::isfinite(cell.dynamic_viscosity) || cell.dynamic_viscosity < 0.0) {
      return absl::InvalidArgumentError("viscosity must be finite and nonnegative");
    }
    constexpr auto kMaximumIndex = std::numeric_limits<std::int32_t>::max();
    if (cell.cell_index.x == kMaximumIndex || cell.cell_index.y == kMaximumIndex
        || cell.cell_index.z == kMaximumIndex) {
      return absl::OutOfRangeError("viscosity cell has an unrepresentable high face");
    }
  }
  if (cells.empty()) {
    return rows;
  }

  const double spacing = grid.geometry().cell_width;
  const double normal_coefficient = 1.0 / spacing;
  const double shear_coefficient = 0.5 * normal_coefficient;
  if (!std::isnormal(normal_coefficient)) {
    return absl::OutOfRangeError("normal strain coefficient is outside normal double range");
  }
  rows.reserve(*maximum_rows);
  for (const auto& cell : cells) {
    const double quadrature_weight = StrainWeight3D(cell.dynamic_viscosity, spacing, 1);
    if (cell.dynamic_viscosity > 0.0 && !std::isnormal(quadrature_weight)) {
      return absl::OutOfRangeError("normal strain quadrature is outside normal double range");
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto normal = static_cast<FaceNormalAxis>(axis);
      const std::array faces{
          ViscosityFace(axis, cell.cell_index),
          ViscosityFace(axis, NextViscosityCell(cell.cell_index, axis)),
      };
      MacStrainRow3D row{
          .low_cell = cell.cell_index,
          .first_axis = normal,
          .second_axis = normal,
          .terms = {},
          .term_count = 0,
          .quadrature_weight = quadrature_weight,
      };
      const auto status = SetStrainTerms(grid, faces, normal_coefficient, row);
      if (!status.ok()) {
        return status;
      }
      rows.push_back(row);
    }
  }

  for (const auto& cell : cells) {
    for (std::size_t first_axis = 0; first_axis < 2; ++first_axis) {
      const auto first_index = NextViscosityCell(cell.cell_index, first_axis);
      const auto* first = FindViscosityCell(cells, first_index);
      if (first == nullptr) {
        continue;
      }
      for (std::size_t second_axis = first_axis + 1; second_axis < 3; ++second_axis) {
        const auto second_index = NextViscosityCell(cell.cell_index, second_axis);
        const auto diagonal_index = NextViscosityCell(first_index, second_axis);
        const auto* second = FindViscosityCell(cells, second_index);
        const auto* diagonal = FindViscosityCell(cells, diagonal_index);
        if (second == nullptr || diagonal == nullptr) {
          continue;
        }
        const double edge_viscosity = HarmonicEdgeViscosity(
            {cell.dynamic_viscosity, first->dynamic_viscosity, second->dynamic_viscosity, diagonal->dynamic_viscosity}
        );
        if (!std::isfinite(edge_viscosity)) {
          return absl::OutOfRangeError("edge viscosity is unrepresentable");
        }
        const double quadrature_weight = StrainWeight3D(edge_viscosity, spacing, 2);
        if (!std::isnormal(shear_coefficient) || (edge_viscosity > 0.0 && !std::isnormal(quadrature_weight))) {
          return absl::OutOfRangeError("shear strain coefficient or quadrature is outside normal double range");
        }
        const std::array faces{
            ViscosityFace(first_axis, first_index),
            ViscosityFace(first_axis, diagonal_index),
            ViscosityFace(second_axis, second_index),
            ViscosityFace(second_axis, diagonal_index),
        };
        MacStrainRow3D row{
            .low_cell = cell.cell_index,
            .first_axis = static_cast<FaceNormalAxis>(first_axis),
            .second_axis = static_cast<FaceNormalAxis>(second_axis),
            .terms = {},
            .term_count = 0,
            .quadrature_weight = quadrature_weight,
        };
        const auto status = SetStrainTerms(grid, faces, shear_coefficient, row);
        if (!status.ok()) {
          return status;
        }
        rows.push_back(row);
      }
    }
  }
  return rows;
}

absl::Status
ViscousStep(const ViscosityInputsView& inputs, ViscosityOutputsView outputs, ViscosityWorkspace& workspace) {
  const auto views = ValidateStepViews(inputs, outputs, workspace);
  if (!views.ok()) {
    return views;
  }
  const auto parameters = ValidateStepParameters(inputs);
  if (!parameters.ok()) {
    return parameters;
  }

  const std::size_t columns = inputs.face_velocities.size();
  auto stiffness = std::span(workspace.matrix).first(columns * columns);
  auto wall_load = std::span(workspace.wall_load).first(columns);
  const auto assembled = AssembleViscousOperator(inputs, stiffness, wall_load);
  if (!assembled.ok()) {
    return assembled;
  }
  auto sqrt_mass = std::span(workspace.sqrt_mass).first(columns);
  for (std::size_t face = 0; face < columns; ++face) {
    sqrt_mass[face] = std::sqrt(inputs.face_inertias[face]);
  }
  const double spectral_bound = ComputeSpectralBound(inputs);
  // Round toward rejection on both sides of the stability comparison.
  // Zero dt and zero stiffness are exact identities.
  const double stability_margin = inputs.explicit_stability_margin;
  const double explicit_limit = stability_margin == 0.0 ? 2.0 : 2.0 * std::nextafter(1.0 - stability_margin, 0.0);
  const bool explicit_safe =
      inputs.dt == 0.0 || spectral_bound == 0.0 || UpperAdjacent(inputs.dt * spectral_bound) <= explicit_limit;
  if (inputs.mode == ViscosityMode::kExplicit && !explicit_safe) {
    return absl::FailedPreconditionError("explicit viscosity exceeds its stability bound");
  }
  const bool use_explicit = inputs.mode != ViscosityMode::kImplicit && explicit_safe;
  ViscosityDiagnostics diagnostics{
      .method = use_explicit ? ViscosityMethod::kExplicit : ViscosityMethod::kImplicit,
      .spectral_bound = spectral_bound,
      .energy_before = 0.0,
      .energy_after = 0.0,
      .wall_work = 0.0,
      .numerical_dissipation = 0.0,
  };

  auto next_velocity = std::span(workspace.next_velocity).first(columns);
  if (inputs.dt == 0.0 || spectral_bound == 0.0) {
    std::ranges::copy(inputs.face_velocities, next_velocity.begin());
  } else if (use_explicit) {
    const auto stepped = ApplyForwardEuler(inputs, stiffness, wall_load, next_velocity);
    if (!stepped.ok()) {
      return stepped;
    }
  } else {
    auto solve_rhs = std::span(workspace.solve_rhs).first(columns);
    const auto stepped =
        ApplyBackwardEuler(inputs, stiffness, wall_load, sqrt_mass, solve_rhs, next_velocity, workspace.solve);
    if (!stepped.ok()) {
      return stepped;
    }
  }

  const auto measured = ComputeEnergyAndWallWork(inputs, next_velocity, wall_load, sqrt_mass, diagnostics);
  if (!measured.ok()) {
    return measured;
  }

  std::ranges::copy(next_velocity, outputs.updated_face_velocities.begin());
  outputs.diagnostics = diagnostics;
  return absl::OkStatus();
}

}  // namespace fluidity
