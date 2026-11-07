#include "fluidity/host/viscosity.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>

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

struct BinaryExponentRange {
  void Include(BinaryProduct term) {
    if (term.fraction != 0.0) {
      const int exponent = term.exponent + std::ilogb(term.fraction);
      smallest = std::min(smallest, exponent);
      largest = std::max(largest, exponent);
    }
  }

  bool Empty() const {
    return smallest > largest;
  }

  int smallest = std::numeric_limits<int>::max();
  int largest = std::numeric_limits<int>::min();
};

absl::Status
ValidateStepViews(const ViscosityInputsView& inputs, ViscosityOutputsView outputs, ViscosityWorkspace& workspace) {
  const std::size_t columns = inputs.velocity.size();
  if (columns == 0) {
    return absl::InvalidArgumentError("viscosity requires at least one velocity unknown");
  }
  if (columns > kMaxDenseUnknowns) {
    return absl::UnimplementedError("viscosity supports at most 64 velocity unknowns");
  }
  const auto coefficients = CheckedExtentProduct(inputs.rows, columns);
  if (!coefficients.ok()) {
    return coefficients.status();
  }
  const auto coefficient_bytes = CheckedExtentProduct(*coefficients, sizeof(double));
  if (!coefficient_bytes.ok()) {
    return coefficient_bytes.status();
  }
  const auto column_extents = CheckExtents(columns, {inputs.mass.size(), outputs.velocity.size()});
  if (!column_extents.ok()) {
    return column_extents;
  }
  const auto row_extents = CheckExtents(inputs.rows, {inputs.weight.size(), inputs.boundary_strain.size()});
  if (!row_extents.ok()) {
    return row_extents;
  }
  const auto strain_extent = ValidateExtent(inputs.strain.size(), *coefficients);
  if (!strain_extent.ok()) {
    return strain_extent;
  }

  return CheckWritableViews(
      {std::as_bytes(outputs.velocity),
       std::as_bytes(std::span(&outputs.diagnostics, 1)),
       std::as_bytes(std::span(&workspace, 1))},
      {std::as_bytes(std::span(&inputs, 1)),
       std::as_bytes(inputs.velocity),
       std::as_bytes(inputs.mass),
       std::as_bytes(inputs.strain),
       std::as_bytes(inputs.weight),
       std::as_bytes(inputs.boundary_strain)}
  );
}

absl::Status ValidateStepParameters(const ViscosityInputsView& inputs) {
  if (!std::isfinite(inputs.dt) || inputs.dt < 0.0) {
    return absl::InvalidArgumentError("viscosity dt must be finite and nonnegative");
  }
  if (!std::isfinite(inputs.safety) || inputs.safety < 0.0 || inputs.safety >= 1.0) {
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

  for (std::size_t face = 0; face < inputs.velocity.size(); ++face) {
    if (!std::isfinite(inputs.velocity[face])) {
      return absl::InvalidArgumentError("face velocity must be finite");
    }
    if (!std::isfinite(inputs.mass[face]) || inputs.mass[face] <= 0.0) {
      return absl::InvalidArgumentError("face inertia must be finite and positive");
    }
  }
  for (std::size_t row = 0; row < inputs.rows; ++row) {
    if (!std::isfinite(inputs.weight[row]) || inputs.weight[row] < 0.0) {
      return absl::InvalidArgumentError("strain quadrature must be finite and nonnegative");
    }
    if (!std::isfinite(inputs.boundary_strain[row])) {
      return absl::InvalidArgumentError("wall strain must be finite");
    }
  }
  for (const double coefficient : inputs.strain) {
    if (!std::isfinite(coefficient)) {
      return absl::InvalidArgumentError("strain operator must be finite");
    }
  }
  return absl::OkStatus();
}

// Assemble one lower triangle of E^T W E and mirror it so both integrators
// use the same symmetric stiffness. Accumulate the wall lift in the same row order.
absl::Status
AssembleViscousOperator(const ViscosityInputsView& inputs, std::span<double> stiffness, std::span<double> wall_load) {
  const std::size_t columns = inputs.velocity.size();
  std::ranges::fill(stiffness, 0.0);
  std::ranges::fill(wall_load, 0.0);
  for (std::size_t strain_row = 0; strain_row < inputs.rows; ++strain_row) {
    const double quadrature = inputs.weight[strain_row];
    if (quadrature == 0.0) {
      continue;
    }
    const double wall_strain = inputs.boundary_strain[strain_row];
    const auto strain = inputs.strain.subspan(strain_row * columns, columns);
    for (std::size_t face_row = 0; face_row < columns; ++face_row) {
      const double row_coefficient = strain[face_row];
      if (row_coefficient == 0.0) {
        continue;
      }
      if (wall_strain != 0.0) {
        const double contribution = ScaledProduct(quadrature, row_coefficient, wall_strain);
        if (!std::isnormal(contribution)) {
          return absl::OutOfRangeError("wall load term is outside normal double range");
        }
        wall_load[face_row] += contribution;
      }
      for (std::size_t face_column = 0; face_column <= face_row; ++face_column) {
        const double column_coefficient = strain[face_column];
        if (column_coefficient == 0.0) {
          continue;
        }
        const double contribution = ScaledProduct(quadrature, row_coefficient, column_coefficient);
        if (!std::isnormal(contribution)) {
          return absl::OutOfRangeError("viscous stiffness term is outside normal double range");
        }
        stiffness[face_row * columns + face_column] += contribution;
      }
    }
  }

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

double MassScaledStiffness(double coefficient, double row_sqrt_mass, double column_sqrt_mass) {
  return ScaledProduct(coefficient, 1.0 / row_sqrt_mass, 1.0 / column_sqrt_mass);
}

// The infinity norm bounds the largest eigenvalue of the symmetric mass-scaled
// operator. A single row maximum determines the stability limit for all faces.
absl::StatusOr<double> ComputeSpectralBound(std::span<const double> stiffness, std::span<const double> sqrt_mass) {
  const std::size_t columns = sqrt_mass.size();
  double spectral_bound = 0.0;
  for (std::size_t row = 0; row < columns; ++row) {
    double row_sum = 0.0;
    for (std::size_t column = 0; column < columns; ++column) {
      const double coefficient = stiffness[row * columns + column];
      const double scaled_coefficient = MassScaledStiffness(coefficient, sqrt_mass[row], sqrt_mass[column]);
      if (coefficient != 0.0 && !std::isnormal(scaled_coefficient)) {
        return absl::OutOfRangeError("mass-scaled viscous stiffness is outside normal double range");
      }
      row_sum += std::abs(scaled_coefficient);
    }
    if (!std::isfinite(row_sum)) {
      return absl::OutOfRangeError("viscosity spectral row sum is nonfinite");
    }
    spectral_bound = std::max(spectral_bound, row_sum);
  }
  return spectral_bound;
}

absl::Status ApplyForwardEuler(
    const ViscosityInputsView& inputs,
    std::span<const double> stiffness,
    std::span<const double> wall_load,
    std::span<double> next_velocity
) {
  const std::size_t columns = inputs.velocity.size();
  for (std::size_t row = 0; row < columns; ++row) {
    const double face_mass = inputs.mass[row];
    double velocity_increment = ScaledProduct(inputs.dt, wall_load[row], 1.0, face_mass);
    // Accumulate in velocity units so an extreme force can still make a
    // representable contribution after scaling by dt and the face inertia.
    for (std::size_t column = 0; column < columns; ++column) {
      const double coefficient = stiffness[row * columns + column];
      const double viscous_increment = ScaledProduct(inputs.dt, coefficient, inputs.velocity[column], face_mass);
      velocity_increment -= viscous_increment;
    }
    if (!std::isfinite(velocity_increment)) {
      return absl::OutOfRangeError("explicit viscous increment is nonfinite");
    }
    next_velocity[row] = inputs.velocity[row] + velocity_increment;
  }
  return absl::OkStatus();
}

// Assemble I + dt S in place, preserving the symmetry of the mass-scaled system.
// Its largest diagonal sets the scale of attenuation through the solve.
absl::StatusOr<double>
BuildImplicitSystem(double dt, std::span<double> system_matrix, std::span<const double> sqrt_mass) {
  const std::size_t columns = sqrt_mass.size();
  double largest_diagonal = 1.0;
  for (std::size_t row = 0; row < columns; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      const double stiffness = system_matrix[row * columns + column];
      const double mass_scaled = MassScaledStiffness(stiffness, sqrt_mass[row], sqrt_mass[column]);
      const double step_coefficient = dt * mass_scaled;
      if (stiffness != 0.0 && !std::isnormal(step_coefficient)) {
        return absl::OutOfRangeError("implicit viscous stiffness is outside normal double range");
      }
      double system_coefficient = step_coefficient;
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
    term_exponents.Include(FactorProduct(sqrt_mass[face], inputs.velocity[face], 1.0));
    term_exponents.Include(FactorProduct(inputs.dt, wall_load[face], 1.0, sqrt_mass[face]));
  }
  if (term_exponents.Empty()) {
    return 0;
  }

  constexpr int kMinNormalExponent = std::numeric_limits<double>::min_exponent - 1;
  // Leave eight exponent bits for additions within a system of at most 64 unknowns.
  constexpr int kMaxRhsTermExponent = std::numeric_limits<double>::max_exponent - 1 - 8;
  const int smallest_shift = kMinNormalExponent - term_exponents.smallest;
  const int largest_shift = kMaxRhsTermExponent - term_exponents.largest;
  if (smallest_shift > largest_shift) {
    return absl::OutOfRangeError("implicit right-hand side terms have no common normal scale");
  }
  const int term_center = std::midpoint(term_exponents.smallest, term_exponents.largest);
  const int preferred_shift = std::ilogb(largest_diagonal) / 2 - term_center;
  return std::clamp(preferred_shift, smallest_shift, largest_shift);
}

absl::Status StageImplicitRhs(
    const ViscosityInputsView& inputs,
    std::span<const double> wall_load,
    std::span<const double> sqrt_mass,
    int rhs_shift,
    std::span<double> solve_rhs
) {
  for (std::size_t face = 0; face < sqrt_mass.size(); ++face) {
    const double weighted_velocity = FactorProduct(sqrt_mass[face], inputs.velocity[face], 1.0).Value(rhs_shift);
    const double wall_impulse = FactorProduct(inputs.dt, wall_load[face], 1.0, sqrt_mass[face]).Value(rhs_shift);
    solve_rhs[face] = weighted_velocity + wall_impulse;
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

  const int residual_shift = -term_exponents.largest;
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
    std::span<double> next_velocity
) {
  const std::size_t columns = sqrt_mass.size();
  for (std::size_t face = 0; face < columns; ++face) {
    const double weighted_velocity = next_velocity[face];
    if (weighted_velocity == 0.0) {
      const auto equation = system_matrix.subspan(face * columns, columns);
      if (!ZeroComponentSatisfiesEquation(equation, solve_rhs[face], next_velocity)) {
        return absl::OutOfRangeError("implicit solve lost a velocity component at the common scale");
      }
    } else if (!std::isnormal(weighted_velocity)) {
      return absl::OutOfRangeError("implicit solve component is outside normal double range");
    }
  }

  for (std::size_t face = 0; face < columns; ++face) {
    const double weighted_velocity = next_velocity[face];
    const double velocity = FactorProduct(weighted_velocity, 1.0, 1.0, sqrt_mass[face]).Value(-rhs_shift);
    if (!std::isfinite(velocity) || (weighted_velocity != 0.0 && velocity == 0.0)) {
      return absl::OutOfRangeError("implicit face velocity is outside representable double range");
    }
    next_velocity[face] = velocity;
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
    std::span<double> next_velocity,
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
  const auto solved = SolveSmallSpd(system_matrix, solve_rhs, next_velocity, solve);
  if (!solved.ok()) {
    return solved;
  }
  return RestoreImplicitVelocity(system_matrix, solve_rhs, sqrt_mass, *rhs_shift, next_velocity);
}

absl::Status ComputeEnergyAndWallWork(
    const ViscosityInputsView& inputs,
    std::span<const double> next_velocity,
    std::span<const double> wall_load,
    std::span<const double> sqrt_mass,
    ViscosityDiagnostics& diagnostics
) {
  // Forward Euler takes work and material dissipation at the old velocity, so
  // its energy balance subtracts half the squared mass norm of the increment.
  // Backward Euler uses the new velocity and adds that same increment energy.
  const bool explicit_step = diagnostics.method == ViscosityMethod::kExplicit;
  for (std::size_t face = 0; face < next_velocity.size(); ++face) {
    if (!std::isfinite(next_velocity[face])) {
      return absl::OutOfRangeError("updated viscous velocity is nonfinite");
    }
    const double old_weighted_velocity = sqrt_mass[face] * inputs.velocity[face];
    const double new_weighted_velocity = sqrt_mass[face] * next_velocity[face];
    const double velocity_increment = next_velocity[face] - inputs.velocity[face];
    const double weighted_increment = std::isfinite(velocity_increment) ? sqrt_mass[face] * velocity_increment
                                                                        : new_weighted_velocity - old_weighted_velocity;
    const double work_velocity = explicit_step ? inputs.velocity[face] : next_velocity[face];

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

absl::StatusOr<StrainShape> MacStrainShape2D(const Grid2D& grid) {
  const auto faces = FaceCount(grid);
  if (!faces.ok()) {
    return faces.status();
  }
  if (*faces > kMaxDenseUnknowns) {
    return absl::UnimplementedError("MAC strain supports at most 64 velocity unknowns");
  }

  // With at most 64 faces, the grid has at most 32 cells. There are two normal
  // rows per cell and fewer shear rows than cells.
  const std::size_t cell_columns = grid.nx;
  const std::size_t cell_rows = grid.ny;
  const std::size_t normal_rows = 2 * cell_columns * cell_rows;
  const std::size_t shear_rows = (cell_columns - 1) * (cell_rows - 1);
  const std::size_t strain_rows = normal_rows + shear_rows;
  const auto coefficients = CheckedExtentProduct(strain_rows, *faces);
  if (!coefficients.ok()) {
    return coefficients.status();
  }
  return StrainShape{
      .rows = strain_rows,
      .columns = *faces,
      .coefficients = *coefficients,
  };
}

absl::Status MacStrain2D(const Grid2D& grid, double mu, std::span<double> strain, std::span<double> weight) {
  const auto shape = MacStrainShape2D(grid);
  if (!shape.ok()) {
    return shape.status();
  }
  if (!std::isfinite(mu) || mu < 0.0) {
    return absl::InvalidArgumentError("dynamic viscosity must be finite and nonnegative");
  }
  const auto strain_extent = ValidateExtent(strain.size(), shape->coefficients);
  if (!strain_extent.ok()) {
    return strain_extent;
  }
  const auto weight_extent = ValidateExtent(weight.size(), shape->rows);
  if (!weight_extent.ok()) {
    return weight_extent;
  }
  const auto disjoint =
      CheckWritableViews({std::as_bytes(strain), std::as_bytes(weight)}, {std::as_bytes(std::span(&grid, 1))});
  if (!disjoint.ok()) {
    return disjoint;
  }

  const std::size_t cell_columns = grid.nx;
  const std::size_t cell_rows = grid.ny;
  const std::size_t cells = cell_columns * cell_rows;
  const std::size_t normal_rows = 2 * cells;
  const bool has_shear = shape->rows > normal_rows;
  const double normal_derivative = 1.0 / grid.h;
  const double shear_derivative = 0.5 * normal_derivative;
  if (!std::isnormal(normal_derivative) || (has_shear && !std::isnormal(shear_derivative))) {
    return absl::OutOfRangeError("MAC strain difference coefficient is outside normal double range");
  }

  // Fold the tensor multiplicity into a spacing factor before multiplication.
  // This also handles a finite mu whose double or quadruple would overflow.
  const double normal_weight = ScaledProduct(mu, 2.0 * grid.h, grid.h);
  const double shear_weight = has_shear ? ScaledProduct(mu, 4.0 * grid.h, grid.h) : 0.0;
  if (mu > 0.0 && !std::isnormal(normal_weight)) {
    return absl::OutOfRangeError("normal strain quadrature is outside normal double range");
  }
  if (mu > 0.0 && has_shear && !std::isnormal(shear_weight)) {
    return absl::OutOfRangeError("shear strain quadrature is outside normal double range");
  }

  // Every coefficient and index extent has been checked. Fill the caller's
  // dense views directly now that the remaining stencil writes cannot fail.
  std::ranges::fill(strain, 0.0);
  std::ranges::fill(weight.first(normal_rows), normal_weight);
  std::ranges::fill(weight.subspan(normal_rows), shear_weight);

  const std::size_t x_face_stride = cell_columns + 1;
  const std::size_t y_face_offset = x_face_stride * cell_rows;
  for (std::size_t row = 0; row < cell_rows; ++row) {
    for (std::size_t column = 0; column < cell_columns; ++column) {
      const std::size_t cell = row * cell_columns + column;
      const std::size_t left_face = row * x_face_stride + column;
      const std::size_t bottom_face = y_face_offset + cell;
      const std::size_t x_strain_offset = cell * shape->columns;
      const std::size_t y_strain_offset = (cells + cell) * shape->columns;
      strain[x_strain_offset + left_face] = -normal_derivative;
      strain[x_strain_offset + left_face + 1] = normal_derivative;
      strain[y_strain_offset + bottom_face] = -normal_derivative;
      strain[y_strain_offset + bottom_face + cell_columns] = normal_derivative;
    }
  }

  for (std::size_t row = 1; row < cell_rows; ++row) {
    for (std::size_t column = 1; column < cell_columns; ++column) {
      const std::size_t vertex = (row - 1) * (cell_columns - 1) + column - 1;
      const std::size_t shear_offset = (normal_rows + vertex) * shape->columns;
      const std::size_t upper_x_face = row * x_face_stride + column;
      const std::size_t right_y_face = y_face_offset + row * cell_columns + column;
      const std::size_t lower_x_face = upper_x_face - x_face_stride;
      const std::size_t left_y_face = right_y_face - 1;
      strain[shear_offset + lower_x_face] = -shear_derivative;
      strain[shear_offset + upper_x_face] = shear_derivative;
      strain[shear_offset + left_y_face] = -shear_derivative;
      strain[shear_offset + right_y_face] = shear_derivative;
    }
  }
  return absl::OkStatus();
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

  const std::size_t columns = inputs.velocity.size();
  auto stiffness = std::span(workspace.matrix).first(columns * columns);
  auto wall_load = std::span(workspace.wall_load).first(columns);
  const auto assembled = AssembleViscousOperator(inputs, stiffness, wall_load);
  if (!assembled.ok()) {
    return assembled;
  }
  auto sqrt_mass = std::span(workspace.sqrt_mass).first(columns);
  for (std::size_t face = 0; face < columns; ++face) {
    sqrt_mass[face] = std::sqrt(inputs.mass[face]);
  }
  const auto computed_bound = ComputeSpectralBound(stiffness, sqrt_mass);
  if (!computed_bound.ok()) {
    return computed_bound.status();
  }

  // The stability test applies to the entire coupled operator. A product
  // above the finite explicit limit can only select the implicit branch.
  const double explicit_limit = 2.0 * (1.0 - inputs.safety);
  const double spectral_bound = *computed_bound;
  const double scaled_step = inputs.dt * spectral_bound;
  const bool explicit_safe = scaled_step <= explicit_limit;
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
    std::ranges::copy(inputs.velocity, next_velocity.begin());
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

  std::ranges::copy(next_velocity, outputs.velocity.begin());
  outputs.diagnostics = diagnostics;
  return absl::OkStatus();
}

}  // namespace fluidity
