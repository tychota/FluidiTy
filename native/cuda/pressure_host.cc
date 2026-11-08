#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

#include "absl/status/status.h"
#include "fluidity/host/cuda_pressure.h"
#include "fluidity/host/grid.h"
#include "fluidity/host/views.h"
#include "operator_runtime.h"
#include "pressure_bridge.h"

namespace fluidity {
namespace {

using cuda_internal::PreparedPressureInputs;
using cuda_internal::PressureKernelError;
using cuda_internal::PressureKernelResult;

absl::Status ValidatePressureViews(
    const PressureInputsView& pressure_inputs,
    const PressureOutputsView& pressure_outputs,
    const CudaOperatorMeasurement& kernel_measurement,
    std::size_t cell_count,
    std::size_t face_count
) {
  const auto face_extent_status = CheckExtents(
      face_count,
      {
          pressure_inputs.predicted_face_velocities.size(),
          pressure_inputs.face_open_fractions.size(),
          pressure_inputs.face_inverse_densities.size(),
          pressure_inputs.pressure_gradient_distances.size(),
          pressure_inputs.wall_face_velocities.size(),
          pressure_inputs.dirichlet_face_mask.size(),
          pressure_inputs.dirichlet_face_pressures.size(),
          pressure_inputs.face_pressure_jumps.size(),
          pressure_outputs.corrected_face_velocities.size(),
      }
  );
  if (!face_extent_status.ok()) {
    return face_extent_status;
  }
  const auto cell_extent_status = CheckExtents(
      cell_count,
      {
          pressure_inputs.target_cell_divergence.size(),
          pressure_inputs.cell_fluid_volumes.size(),
          pressure_inputs.cell_bulk_moduli.size(),
          pressure_inputs.previous_cell_pressures.size(),
          pressure_outputs.pressure.size(),
          pressure_outputs.rhs.size(),
          pressure_outputs.residual.size(),
          pressure_outputs.constraint_defect.size(),
          pressure_outputs.gauge_cell_indices.size(),
      }
  );
  if (!cell_extent_status.ok()) {
    return cell_extent_status;
  }
  const auto matrix_extent_status = ValidateExtent(pressure_outputs.original_matrix.size(), cell_count * cell_count);
  if (!matrix_extent_status.ok()) {
    return matrix_extent_status;
  }

  return CheckWritableViews(
      {
          std::as_bytes(pressure_outputs.pressure),
          std::as_bytes(pressure_outputs.corrected_face_velocities),
          std::as_bytes(pressure_outputs.original_matrix),
          std::as_bytes(pressure_outputs.rhs),
          std::as_bytes(pressure_outputs.residual),
          std::as_bytes(pressure_outputs.constraint_defect),
          std::as_bytes(pressure_outputs.gauge_cell_indices),
          std::as_bytes(std::span(&pressure_outputs.gauge_count, 1)),
          std::as_bytes(std::span(&kernel_measurement, 1)),
      },
      {
          std::as_bytes(std::span(&pressure_inputs, 1)),
          std::as_bytes(pressure_inputs.predicted_face_velocities),
          std::as_bytes(pressure_inputs.face_open_fractions),
          std::as_bytes(pressure_inputs.face_inverse_densities),
          std::as_bytes(pressure_inputs.pressure_gradient_distances),
          std::as_bytes(pressure_inputs.wall_face_velocities),
          std::as_bytes(pressure_inputs.dirichlet_face_mask),
          std::as_bytes(pressure_inputs.dirichlet_face_pressures),
          std::as_bytes(pressure_inputs.face_pressure_jumps),
          std::as_bytes(pressure_inputs.target_cell_divergence),
          std::as_bytes(pressure_inputs.cell_fluid_volumes),
          std::as_bytes(pressure_inputs.cell_bulk_moduli),
          std::as_bytes(pressure_inputs.previous_cell_pressures),
      }
  );
}

bool IsFinitePositive(double physical_value) {
  return physical_value > 0.0 && std::isfinite(physical_value);
}

// Validate physical domains and package geometry. The kernel derives all
// pressure coefficients and solves from these fields.
absl::Status
PreparePressureInputs(const PressureInputsView& pressure_inputs, PreparedPressureInputs& prepared_pressure_inputs) {
  for (std::size_t face_index = 0; face_index < prepared_pressure_inputs.face_count; ++face_index) {
    const auto face_geometry = FaceGeometry(pressure_inputs.grid, face_index);
    if (!face_geometry.ok()) {
      return face_geometry.status();
    }
    const double open_area_fraction = pressure_inputs.face_open_fractions[face_index];
    const bool valid_open_area_fraction = open_area_fraction >= 0.0 && open_area_fraction <= 1.0;
    if (!valid_open_area_fraction) {
      return absl::InvalidArgumentError("CUDA pressure aperture must be in [0, 1]");
    }
    const double inverse_density = pressure_inputs.face_inverse_densities[face_index];
    const double pressure_gradient_distance = pressure_inputs.pressure_gradient_distances[face_index];
    if (!IsFinitePositive(inverse_density) || !IsFinitePositive(pressure_gradient_distance)) {
      return absl::InvalidArgumentError("CUDA pressure inverse density and distance must be finite and positive");
    }
    const double predicted_face_velocity = pressure_inputs.predicted_face_velocities[face_index];
    const double wall_face_velocity = pressure_inputs.wall_face_velocities[face_index];
    const double dirichlet_pressure = pressure_inputs.dirichlet_face_pressures[face_index];
    const double oriented_pressure_jump = pressure_inputs.face_pressure_jumps[face_index];
    if (!std::isfinite(predicted_face_velocity) || !std::isfinite(wall_face_velocity)
        || !std::isfinite(dirichlet_pressure) || !std::isfinite(oriented_pressure_jump)) {
      return absl::InvalidArgumentError("CUDA pressure face velocities, boundary pressures and jumps must be finite");
    }
    const auto dirichlet_face_mask = pressure_inputs.dirichlet_face_mask[face_index];
    if (dirichlet_face_mask > 1) {
      return absl::InvalidArgumentError("CUDA pressure Dirichlet mask must contain 0 or 1");
    }
    const bool has_dirichlet_pressure = dirichlet_face_mask == 1;
    const bool is_internal_face = face_geometry->tail >= 0 && face_geometry->head >= 0;
    if (has_dirichlet_pressure && is_internal_face) {
      return absl::InvalidArgumentError("CUDA pressure Dirichlet face must be on the grid boundary");
    }
    if (has_dirichlet_pressure && oriented_pressure_jump != 0.0) {
      return absl::InvalidArgumentError("CUDA pressure jump must be zero on a Dirichlet face");
    }
    prepared_pressure_inputs.faces[face_index] = {
        .tail_cell_index = face_geometry->tail,
        .head_cell_index = face_geometry->head,
        .face_area = face_geometry->area,
        .predicted_face_velocity = predicted_face_velocity,
        .open_area_fraction = open_area_fraction,
        .inverse_density = inverse_density,
        .pressure_gradient_distance = pressure_gradient_distance,
        .wall_face_velocity = wall_face_velocity,
        .dirichlet_pressure = dirichlet_pressure,
        .oriented_pressure_jump = oriented_pressure_jump,
        .has_dirichlet_pressure = has_dirichlet_pressure,
    };
  }
  for (std::size_t cell_index = 0; cell_index < prepared_pressure_inputs.cell_count; ++cell_index) {
    const double target_cell_divergence = pressure_inputs.target_cell_divergence[cell_index];
    const double previous_cell_pressure = pressure_inputs.previous_cell_pressures[cell_index];
    if (!std::isfinite(target_cell_divergence) || !std::isfinite(previous_cell_pressure)) {
      return absl::InvalidArgumentError("CUDA pressure target divergence and previous pressure must be finite");
    }
    const double cell_fluid_volume = pressure_inputs.cell_fluid_volumes[cell_index];
    if (!IsFinitePositive(cell_fluid_volume)) {
      return absl::InvalidArgumentError("CUDA pressure fluid volume must be finite and positive");
    }
    const double cell_bulk_modulus = pressure_inputs.cell_bulk_moduli[cell_index];
    if (!(cell_bulk_modulus > 0.0)) {
      return absl::InvalidArgumentError("CUDA pressure bulk modulus must be positive");
    }
    prepared_pressure_inputs.cells[cell_index] = {
        .target_cell_divergence = target_cell_divergence,
        .fluid_volume = cell_fluid_volume,
        .bulk_modulus = cell_bulk_modulus,
        .previous_pressure = previous_cell_pressure,
    };
  }
  return absl::OkStatus();
}

absl::Status CollectPressure(
    cuda_internal::CudaOperatorRuntime& runtime,
    const PreparedPressureInputs& prepared_pressure_inputs,
    PressureKernelResult& staged_pressure_result,
    CudaOperatorMeasurement& staged_kernel_measurement
) {
  if (auto status = runtime.Initialize(); !status.ok()) {
    return status;
  }
  PreparedPressureInputs* device_pressure_inputs = nullptr;
  if (auto status = runtime.Upload(std::span(&prepared_pressure_inputs, 1), device_pressure_inputs); !status.ok()) {
    return status;
  }
  PressureKernelResult* device_pressure_result = nullptr;
  if (auto status = runtime.AllocateZeroed(1, device_pressure_result); !status.ok()) {
    return status;
  }
  if (auto status = runtime.RecordKernelStart(); !status.ok()) {
    return status;
  }
  const auto launch_error = static_cast<cudaError_t>(
      cuda_internal::LaunchPressure(device_pressure_inputs, device_pressure_result, runtime.stream())
  );
  if (auto status = cuda_internal::CheckCuda(launch_error, "CUDA pressure launch"); !status.ok()) {
    return status;
  }
  if (auto status = runtime.SynchronizeAndMeasureKernel(); !status.ok()) {
    return status;
  }
  if (auto status = runtime.Download(device_pressure_result, std::span(&staged_pressure_result, 1)); !status.ok()) {
    return status;
  }
  switch (staged_pressure_result.error) {
    case PressureKernelError::kNone:
      break;
    case PressureKernelError::kNonrepresentableArithmetic:
      return absl::OutOfRangeError("CUDA pressure arithmetic is outside the representable range");
    case PressureKernelError::kIncompatibleComponent:
      return absl::FailedPreconditionError("incompatible CUDA Neumann pressure component");
    case PressureKernelError::kUnsafeCholeskyPivot:
      return absl::FailedPreconditionError("CUDA pressure Cholesky pivot is not positive");
    default:
      return absl::InternalError("unknown CUDA pressure kernel error");
  }
  const auto completed_kernel_measurement =
      runtime.ValidateKernelMeasurement(staged_pressure_result.compiled_architecture);
  if (!completed_kernel_measurement.ok()) {
    return completed_kernel_measurement.status();
  }
  staged_kernel_measurement = *completed_kernel_measurement;
  return absl::OkStatus();
}

}  // namespace

absl::Status CudaSolvePressure2D(
    const PressureInputsView& pressure_inputs,
    PressureOutputsView pressure_outputs,
    CudaOperatorMeasurement& kernel_measurement
) {
  const auto cell_count = CellCount(pressure_inputs.grid);
  if (!cell_count.ok()) {
    return cell_count.status();
  }
  if (*cell_count > cuda_internal::kMaxCudaPressureCells) {
    return absl::ResourceExhaustedError("CUDA pressure cell capacity exceeded");
  }
  const auto face_count = FaceCount(pressure_inputs.grid);
  if (!face_count.ok()) {
    return face_count.status();
  }
  if (*face_count > cuda_internal::kMaxCudaPressureFaces) {
    return absl::ResourceExhaustedError("CUDA pressure face capacity exceeded");
  }
  if (!IsFinitePositive(pressure_inputs.dt)) {
    return absl::InvalidArgumentError("CUDA pressure step duration must be finite and positive");
  }
  const auto view_status =
      ValidatePressureViews(pressure_inputs, pressure_outputs, kernel_measurement, *cell_count, *face_count);
  if (!view_status.ok()) {
    return view_status;
  }

  // Keep the upload and download storage alive through synchronization and cleanup.
  PreparedPressureInputs prepared_pressure_inputs{
      .cell_count = *cell_count,
      .face_count = *face_count,
      .step_duration = pressure_inputs.dt,
      .faces = {},
      .cells = {},
  };
  const auto preparation_status = PreparePressureInputs(pressure_inputs, prepared_pressure_inputs);
  if (!preparation_status.ok()) {
    return preparation_status;
  }
  PressureKernelResult staged_pressure_result{};
  CudaOperatorMeasurement staged_kernel_measurement;
  absl::Status pressure_status;
  absl::Status cleanup_status;
  {
    cuda_internal::CudaOperatorRuntime runtime(cleanup_status);
    pressure_status =
        CollectPressure(runtime, prepared_pressure_inputs, staged_pressure_result, staged_kernel_measurement);
  }
  pressure_status.Update(cleanup_status);
  if (!pressure_status.ok()) {
    return pressure_status;
  }

  std::copy_n(staged_pressure_result.cell_pressures, *cell_count, pressure_outputs.pressure.begin());
  std::copy_n(
      staged_pressure_result.corrected_face_velocities,
      *face_count,
      pressure_outputs.corrected_face_velocities.begin()
  );
  std::copy_n(
      staged_pressure_result.original_matrix,
      *cell_count * *cell_count,
      pressure_outputs.original_matrix.begin()
  );
  std::copy_n(staged_pressure_result.original_rhs, *cell_count, pressure_outputs.rhs.begin());
  std::copy_n(staged_pressure_result.original_residuals, *cell_count, pressure_outputs.residual.begin());
  std::copy_n(staged_pressure_result.cell_constraint_defects, *cell_count, pressure_outputs.constraint_defect.begin());
  std::copy_n(
      staged_pressure_result.gauge_cell_indices,
      staged_pressure_result.gauge_count,
      pressure_outputs.gauge_cell_indices.begin()
  );
  pressure_outputs.gauge_count = staged_pressure_result.gauge_count;
  kernel_measurement = staged_kernel_measurement;
  return absl::OkStatus();
}

}  // namespace fluidity
