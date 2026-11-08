#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>

#include "pressure_bridge.h"

namespace fluidity::cuda_internal {
namespace {

struct PressureFaceCoefficients {
  double open_area;
  double wall_area;
  double pressure_to_velocity_scale;  // dt * inverse density / distance, or zero on inactive faces.
  double signed_dirichlet_pressure;  // Exterior pressure times the adjacent cell's incidence.
};

struct PressureKernelWorkspace {
  PressureFaceCoefficients face_coefficients[kMaxCudaPressureFaces];
  double cell_bulk_modulus_times_step[kMaxCudaPressureCells];
  bool cell_has_pressure_anchor[kMaxCudaPressureCells];
};

__device__ bool IsPositiveNormal(double pressure_coefficient) {
  return pressure_coefficient >= DBL_MIN && isfinite(pressure_coefficient);
}

__device__ double IntegratedFaceFlux(
    const PressureFaceCoefficients& face_coefficients,
    double fluid_face_velocity,
    double wall_face_velocity
) {
  return face_coefficients.open_area * fluid_face_velocity + face_coefficients.wall_area * wall_face_velocity;
}

__device__ void
AccumulateOutward(const PressureFaceInput& pressure_face, double oriented_face_flux, double* outward_cell_fluxes) {
  if (pressure_face.tail_cell_index >= 0) {
    outward_cell_fluxes[pressure_face.tail_cell_index] += oriented_face_flux;
  }
  if (pressure_face.head_cell_index >= 0) {
    outward_cell_fluxes[pressure_face.head_cell_index] -= oriented_face_flux;
  }
}

// Assemble R diag(conductance) R^T + compliance in integrated flux units.
// Keep the predicted flux and both pressure lifts separate until forming b.
__device__ PressureKernelError AssembleOriginalSystem(
    const PreparedPressureInputs& pressure_inputs,
    PressureKernelWorkspace& pressure_workspace,
    PressureKernelResult& pressure_result
) {
  const std::size_t cell_count = pressure_inputs.cell_count;
  double predicted_outward_fluxes[kMaxCudaPressureCells]{};
  double jump_flux_lifts[kMaxCudaPressureCells]{};
  double dirichlet_flux_lifts[kMaxCudaPressureCells]{};

  for (std::size_t face_index = 0; face_index < pressure_inputs.face_count; ++face_index) {
    const auto& pressure_face = pressure_inputs.faces[face_index];
    auto& face_coefficients = pressure_workspace.face_coefficients[face_index];
    const double open_face_area = pressure_face.face_area * pressure_face.open_area_fraction;
    face_coefficients.open_area = open_face_area;
    face_coefficients.wall_area = pressure_face.face_area - open_face_area;
    const double predicted_face_flux =
        IntegratedFaceFlux(face_coefficients, pressure_face.predicted_face_velocity, pressure_face.wall_face_velocity);
    AccumulateOutward(pressure_face, predicted_face_flux, predicted_outward_fluxes);

    const bool has_tail_cell = pressure_face.tail_cell_index >= 0;
    const bool has_head_cell = pressure_face.head_cell_index >= 0;
    const bool is_internal_face = has_tail_cell && has_head_cell;
    const bool has_pressure_coupling = is_internal_face || pressure_face.has_dirichlet_pressure;
    if (pressure_face.open_area_fraction == 0.0 || !has_pressure_coupling) {
      continue;
    }
    const double face_pressure_to_velocity_scale =
        pressure_inputs.step_duration * pressure_face.inverse_density / pressure_face.pressure_gradient_distance;
    const double face_conductance = open_face_area * face_pressure_to_velocity_scale;
    if (!IsPositiveNormal(face_conductance)) {
      return PressureKernelError::kNonrepresentableArithmetic;
    }
    face_coefficients.pressure_to_velocity_scale = face_pressure_to_velocity_scale;
    const double jump_flux_lift = face_conductance * pressure_face.oriented_pressure_jump;
    AccumulateOutward(pressure_face, jump_flux_lift, jump_flux_lifts);
    if (is_internal_face) {
      const auto tail_cell_index = static_cast<std::size_t>(pressure_face.tail_cell_index);
      const auto head_cell_index = static_cast<std::size_t>(pressure_face.head_cell_index);
      pressure_result.original_matrix[tail_cell_index * cell_count + tail_cell_index] += face_conductance;
      pressure_result.original_matrix[head_cell_index * cell_count + head_cell_index] += face_conductance;
      pressure_result.original_matrix[tail_cell_index * cell_count + head_cell_index] -= face_conductance;
      pressure_result.original_matrix[head_cell_index * cell_count + tail_cell_index] -= face_conductance;
    } else {
      const auto boundary_cell_index =
          static_cast<std::size_t>(has_tail_cell ? pressure_face.tail_cell_index : pressure_face.head_cell_index);
      face_coefficients.signed_dirichlet_pressure =
          has_tail_cell ? pressure_face.dirichlet_pressure : -pressure_face.dirichlet_pressure;
      const double dirichlet_flux_lift = face_conductance * face_coefficients.signed_dirichlet_pressure;
      AccumulateOutward(pressure_face, dirichlet_flux_lift, dirichlet_flux_lifts);
      pressure_result.original_matrix[boundary_cell_index * cell_count + boundary_cell_index] += face_conductance;
      pressure_workspace.cell_has_pressure_anchor[boundary_cell_index] = true;
    }
  }

  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    const auto& pressure_cell = pressure_inputs.cells[cell_index];
    double cell_compliance = 0.0;
    if (isfinite(pressure_cell.bulk_modulus)) {
      const double cell_bulk_modulus_times_step = pressure_cell.bulk_modulus * pressure_inputs.step_duration;
      if (cell_bulk_modulus_times_step <= 0.0 || !isfinite(cell_bulk_modulus_times_step)) {
        return PressureKernelError::kNonrepresentableArithmetic;
      }
      cell_compliance = pressure_cell.fluid_volume / cell_bulk_modulus_times_step;
      if (!IsPositiveNormal(cell_compliance)) {
        return PressureKernelError::kNonrepresentableArithmetic;
      }
      pressure_workspace.cell_bulk_modulus_times_step[cell_index] = cell_bulk_modulus_times_step;
      pressure_workspace.cell_has_pressure_anchor[cell_index] = true;
    }
    pressure_result.original_matrix[cell_index * cell_count + cell_index] += cell_compliance;
    const double integrated_target_flux = pressure_cell.fluid_volume * pressure_cell.target_cell_divergence;
    const double previous_pressure_flux_lift = cell_compliance * pressure_cell.previous_pressure;
    const double cell_rhs = integrated_target_flux - predicted_outward_fluxes[cell_index] - jump_flux_lifts[cell_index]
        + dirichlet_flux_lifts[cell_index] + previous_pressure_flux_lift;
    if (!isfinite(cell_rhs)) {
      return PressureKernelError::kNonrepresentableArithmetic;
    }
    pressure_result.original_rhs[cell_index] = cell_rhs;
  }
  for (std::size_t matrix_entry_index = 0; matrix_entry_index < cell_count * cell_count; ++matrix_entry_index) {
    if (!isfinite(pressure_result.original_matrix[matrix_entry_index])) {
      return PressureKernelError::kNonrepresentableArithmetic;
    }
  }
  return PressureKernelError::kNone;
}

__device__ std::size_t FindComponentRootCellIndex(const std::size_t* parent_cell_indices, std::size_t cell_index) {
  while (parent_cell_indices[cell_index] != cell_index) {
    cell_index = parent_cell_indices[cell_index];
  }
  return cell_index;
}

// Negative off-diagonals correspond to positive internal face conductances.
// Merge toward the lower root to identify each component's lowest cell index.
__device__ PressureKernelError SelectGauges(
    std::size_t cell_count,
    const PressureKernelWorkspace& pressure_workspace,
    PressureKernelResult& pressure_result
) {
  std::size_t parent_cell_indices[kMaxCudaPressureCells];
  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    parent_cell_indices[cell_index] = cell_index;
  }
  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    for (std::size_t neighbor_cell_index = cell_index + 1; neighbor_cell_index < cell_count; ++neighbor_cell_index) {
      if (pressure_result.original_matrix[cell_index * cell_count + neighbor_cell_index] < 0.0) {
        const std::size_t cell_root_index = FindComponentRootCellIndex(parent_cell_indices, cell_index);
        const std::size_t neighbor_root_cell_index =
            FindComponentRootCellIndex(parent_cell_indices, neighbor_cell_index);
        const std::size_t lower_root_cell_index =
            cell_root_index < neighbor_root_cell_index ? cell_root_index : neighbor_root_cell_index;
        const std::size_t upper_root_cell_index =
            cell_root_index < neighbor_root_cell_index ? neighbor_root_cell_index : cell_root_index;
        parent_cell_indices[upper_root_cell_index] = lower_root_cell_index;
      }
    }
  }
  bool component_has_pressure_anchor[kMaxCudaPressureCells]{};
  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    const std::size_t component_root_cell_index = FindComponentRootCellIndex(parent_cell_indices, cell_index);
    parent_cell_indices[cell_index] = component_root_cell_index;
    component_has_pressure_anchor[component_root_cell_index] |= pressure_workspace.cell_has_pressure_anchor[cell_index];
  }

  double component_rhs_sums[kMaxCudaPressureCells]{};
  double component_absolute_rhs_sums[kMaxCudaPressureCells]{};
  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    const std::size_t component_root_cell_index = parent_cell_indices[cell_index];
    if (!component_has_pressure_anchor[component_root_cell_index]) {
      const double cell_rhs = pressure_result.original_rhs[cell_index];
      component_rhs_sums[component_root_cell_index] += cell_rhs;
      component_absolute_rhs_sums[component_root_cell_index] += fabs(cell_rhs);
    }
  }
  constexpr double kRelativeCompatibilityTolerance = 64.0 * DBL_EPSILON;
  for (std::size_t root_cell_index = 0; root_cell_index < cell_count; ++root_cell_index) {
    if (parent_cell_indices[root_cell_index] != root_cell_index || component_has_pressure_anchor[root_cell_index]) {
      continue;
    }
    const double component_rhs_sum = component_rhs_sums[root_cell_index];
    const double component_absolute_rhs_sum = component_absolute_rhs_sums[root_cell_index];
    if (!isfinite(component_rhs_sum) || !isfinite(component_absolute_rhs_sum)) {
      return PressureKernelError::kNonrepresentableArithmetic;
    }
    if (fabs(component_rhs_sum) > kRelativeCompatibilityTolerance * component_absolute_rhs_sum) {
      return PressureKernelError::kIncompatibleComponent;
    }
    pressure_result.gauge_cell_indices[pressure_result.gauge_count++] = root_cell_index;
  }
  return PressureKernelError::kNone;
}

// Factor the matrix with a unit diagonal and zero off-diagonals at each gauge.
// The original matrix and RHS remain available for the unpinned residual.
__device__ PressureKernelError SolveGaugedSystem(std::size_t cell_count, PressureKernelResult& pressure_result) {
  bool cell_is_gauge[kMaxCudaPressureCells]{};
  for (std::size_t gauge_slot = 0; gauge_slot < pressure_result.gauge_count; ++gauge_slot) {
    cell_is_gauge[pressure_result.gauge_cell_indices[gauge_slot]] = true;
  }
  double lower_cholesky_factor[kMaxCudaPressureCells][kMaxCudaPressureCells];
  for (std::size_t row_index = 0; row_index < cell_count; ++row_index) {
    for (std::size_t column_index = 0; column_index <= row_index; ++column_index) {
      const bool is_diagonal = row_index == column_index;
      double reduced_matrix_entry = pressure_result.original_matrix[row_index * cell_count + column_index];
      if (cell_is_gauge[row_index] || cell_is_gauge[column_index]) {
        reduced_matrix_entry = is_diagonal ? 1.0 : 0.0;
      }
      for (std::size_t inner_index = 0; inner_index < column_index; ++inner_index) {
        reduced_matrix_entry -=
            lower_cholesky_factor[row_index][inner_index] * lower_cholesky_factor[column_index][inner_index];
      }
      if (!isfinite(reduced_matrix_entry)) {
        return PressureKernelError::kNonrepresentableArithmetic;
      }
      if (is_diagonal && reduced_matrix_entry <= 0.0) {
        return PressureKernelError::kUnsafeCholeskyPivot;
      }
      const double factor_entry = is_diagonal
          ? sqrt(reduced_matrix_entry)
          : reduced_matrix_entry / lower_cholesky_factor[column_index][column_index];
      if (!isfinite(factor_entry)) {
        return PressureKernelError::kNonrepresentableArithmetic;
      }
      lower_cholesky_factor[row_index][column_index] = factor_entry;
    }
  }

  double forward_substitution_values[kMaxCudaPressureCells];
  for (std::size_t row_index = 0; row_index < cell_count; ++row_index) {
    double substitution_rhs = cell_is_gauge[row_index] ? 0.0 : pressure_result.original_rhs[row_index];
    for (std::size_t column_index = 0; column_index < row_index; ++column_index) {
      substitution_rhs -= lower_cholesky_factor[row_index][column_index] * forward_substitution_values[column_index];
    }
    const double forward_substitution_value = substitution_rhs / lower_cholesky_factor[row_index][row_index];
    if (!isfinite(forward_substitution_value)) {
      return PressureKernelError::kNonrepresentableArithmetic;
    }
    forward_substitution_values[row_index] = forward_substitution_value;
  }
  for (std::size_t remaining_rows = cell_count; remaining_rows > 0; --remaining_rows) {
    const std::size_t row_index = remaining_rows - 1;
    double substitution_rhs = forward_substitution_values[row_index];
    for (std::size_t column_index = row_index + 1; column_index < cell_count; ++column_index) {
      substitution_rhs -= lower_cholesky_factor[column_index][row_index] * pressure_result.cell_pressures[column_index];
    }
    const double solved_cell_pressure = substitution_rhs / lower_cholesky_factor[row_index][row_index];
    if (!isfinite(solved_cell_pressure)) {
      return PressureKernelError::kNonrepresentableArithmetic;
    }
    pressure_result.cell_pressures[row_index] = solved_cell_pressure;
  }
  return PressureKernelError::kNone;
}

// Reuse the assembled face coefficients and sum physical fluxes independently
// from A*p so the defect also checks the corrected velocities and wall flux.
__device__ PressureKernelError CorrectVelocityAndEvaluateDefects(
    const PreparedPressureInputs& pressure_inputs,
    const PressureKernelWorkspace& pressure_workspace,
    PressureKernelResult& pressure_result
) {
  double corrected_outward_fluxes[kMaxCudaPressureCells]{};
  for (std::size_t face_index = 0; face_index < pressure_inputs.face_count; ++face_index) {
    const auto& pressure_face = pressure_inputs.faces[face_index];
    const auto& face_coefficients = pressure_workspace.face_coefficients[face_index];
    double corrected_face_velocity = pressure_face.predicted_face_velocity;
    if (face_coefficients.pressure_to_velocity_scale > 0.0) {
      const double tail_cell_pressure =
          pressure_face.tail_cell_index >= 0 ? pressure_result.cell_pressures[pressure_face.tail_cell_index] : 0.0;
      const double head_cell_pressure =
          pressure_face.head_cell_index >= 0 ? pressure_result.cell_pressures[pressure_face.head_cell_index] : 0.0;
      const double oriented_pressure_difference =
          tail_cell_pressure - head_cell_pressure + pressure_face.oriented_pressure_jump;
      const double velocity_correction = face_coefficients.pressure_to_velocity_scale
          * (oriented_pressure_difference - face_coefficients.signed_dirichlet_pressure);
      corrected_face_velocity += velocity_correction;
    }
    if (!isfinite(corrected_face_velocity)) {
      return PressureKernelError::kNonrepresentableArithmetic;
    }
    pressure_result.corrected_face_velocities[face_index] = corrected_face_velocity;
    const double corrected_face_flux =
        IntegratedFaceFlux(face_coefficients, corrected_face_velocity, pressure_face.wall_face_velocity);
    AccumulateOutward(pressure_face, corrected_face_flux, corrected_outward_fluxes);
  }

  const std::size_t cell_count = pressure_inputs.cell_count;
  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    double original_matrix_pressure_product = 0.0;
    for (std::size_t column_index = 0; column_index < cell_count; ++column_index) {
      const double pressure_coefficient = pressure_result.original_matrix[cell_index * cell_count + column_index];
      original_matrix_pressure_product =
          fma(pressure_coefficient, pressure_result.cell_pressures[column_index], original_matrix_pressure_product);
    }
    const double original_residual = pressure_result.original_rhs[cell_index] - original_matrix_pressure_product;
    const auto& pressure_cell = pressure_inputs.cells[cell_index];
    const double corrected_cell_divergence = corrected_outward_fluxes[cell_index] / pressure_cell.fluid_volume;
    double compliance_divergence = 0.0;
    if (isfinite(pressure_cell.bulk_modulus)) {
      const double cell_pressure_change = pressure_result.cell_pressures[cell_index] - pressure_cell.previous_pressure;
      compliance_divergence = cell_pressure_change / pressure_workspace.cell_bulk_modulus_times_step[cell_index];
    }
    const double cell_constraint_defect =
        corrected_cell_divergence - pressure_cell.target_cell_divergence + compliance_divergence;
    if (!isfinite(original_residual) || !isfinite(cell_constraint_defect)) {
      return PressureKernelError::kNonrepresentableArithmetic;
    }
    pressure_result.original_residuals[cell_index] = original_residual;
    pressure_result.cell_constraint_defects[cell_index] = cell_constraint_defect;
  }
  return PressureKernelError::kNone;
}

__global__ void SolvePressureKernel(
    const PreparedPressureInputs* device_pressure_inputs,
    PressureKernelResult* device_pressure_result
) {
#if defined(__CUDA_ARCH__)
  device_pressure_result->compiled_architecture = __CUDA_ARCH__;
#endif
  PressureKernelWorkspace pressure_workspace{};
  auto& pressure_result = *device_pressure_result;
  const auto& pressure_inputs = *device_pressure_inputs;
  pressure_result.error = AssembleOriginalSystem(pressure_inputs, pressure_workspace, pressure_result);
  if (pressure_result.error != PressureKernelError::kNone) {
    return;
  }
  pressure_result.error = SelectGauges(pressure_inputs.cell_count, pressure_workspace, pressure_result);
  if (pressure_result.error != PressureKernelError::kNone) {
    return;
  }
  pressure_result.error = SolveGaugedSystem(pressure_inputs.cell_count, pressure_result);
  if (pressure_result.error != PressureKernelError::kNone) {
    return;
  }
  pressure_result.error = CorrectVelocityAndEvaluateDefects(pressure_inputs, pressure_workspace, pressure_result);
}

}  // namespace

int LaunchPressure(
    const PreparedPressureInputs* device_pressure_inputs,
    PressureKernelResult* device_pressure_result,
    void* stream
) {
  void* pressure_kernel_arguments[] = {&device_pressure_inputs, &device_pressure_result};
  // Check the launch's own status without consuming the caller's last CUDA error.
  return cudaLaunchKernel(
      SolvePressureKernel,
      dim3(1),
      dim3(1),
      pressure_kernel_arguments,
      0,
      static_cast<cudaStream_t>(stream)
  );
}

}  // namespace fluidity::cuda_internal
