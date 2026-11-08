#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <type_traits>

#include "mac_pressure_3d_bridge.h"

namespace fluidity::cuda_internal {
namespace {

// Serial deterministic device arithmetic is deliberate for this bounded parity
// slice. One launch performs at most one iteration, never the whole budget.
__device__ bool RangeFailure(PressureDevice3D device) {
  device.state->error = PressureError3D::kRange;
  return false;
}
__device__ bool FineValue(PressureDevice3D device, double value, float& rounded) {
  if (!isfinite(value) || fabs(value) > FLT_MAX) {
    return RangeFailure(device);
  }
  rounded = static_cast<float>(value);
  if (!isfinite(rounded) || (value != 0.0 && rounded == 0.0F)) {
    return RangeFailure(device);
  }
  return true;
}
// Match the original evaluator's exact-subnormal rule. Normalized mantissas
// keep the explicit FMA exactness residual from underflowing itself.
__device__ bool Product(PressureDevice3D device, double coefficient, double quantity, double& product) {
  product = coefficient * quantity;
  if (!isfinite(product)) {
    return RangeFailure(device);
  }
  if (coefficient == 0.0 || quantity == 0.0 || fabs(product) >= DBL_MIN) {
    return true;
  }
  if (product == 0.0) {
    return RangeFailure(device);
  }
  int coefficient_exponent = 0;
  int quantity_exponent = 0;
  const double coefficient_mantissa = frexp(coefficient, &coefficient_exponent);
  const double quantity_mantissa = frexp(quantity, &quantity_exponent);
  const double normalized_product = scalbn(product, -(coefficient_exponent + quantity_exponent));
  if (fma(coefficient_mantissa, quantity_mantissa, -normalized_product) != 0.0) {
    return RangeFailure(device);
  }
  return true;
}
__device__ void Add(PressureSum3D& sum, double term) {
  const double next = sum.sum + term;
  if (fabs(sum.sum) >= fabs(term)) {
    sum.correction += (sum.sum - next) + term;
  } else {
    sum.correction += (term - next) + sum.sum;
  }
  sum.sum = next;
}
__device__ double Total(const PressureSum3D& sum) {
  return sum.sum + sum.correction;
}
__device__ void Add(PressureRow3D& row, double term) {
  Add(row.flux, term);
  Add(row.absolute_terms, fabs(term));
}
__device__ void AddIncident(PressureDevice3D device, PressureRow3D* rows, const PressureLink3D& link, double flux) {
  if (link.tail_cell_ordinal < device.cell_count) {
    Add(rows[link.tail_cell_ordinal], flux);
  }
  if (link.head_cell_ordinal < device.cell_count) {
    Add(rows[link.head_cell_ordinal], -flux);
  }
}
// Recompute the full original operator including gauge rows and separately
// accumulate physical flux. Neither physical rows nor correction read the RHS.
__device__ bool Evaluate(PressureDevice3D device) {
  const auto count = device.cell_count;
  double* candidate = device.output + count;
  double* residual = candidate + count;
  double* defect = residual + count;
  double* velocity = defect + count;
  auto* operator_rows = device.rows;
  auto* physical_rows = device.rows + count;
  for (std::size_t i = 0; i < count; ++i) {
    candidate[i] = device.output[i];
  }
  for (std::size_t i = 0; i < device.unknown_count; ++i) {
    const auto cell = device.indices[device.unknown_map_offset + i];
    candidate[cell] += static_cast<double>(device.fine_scratch[i]);
    if (!isfinite(candidate[cell])) {
      return RangeFailure(device);
    }
  }
  for (std::size_t i = 0; i < count; ++i) {
    operator_rows[i] = {};
    physical_rows[i] = {};
    double flux = 0.0;
    if (!Product(device, device.cells[i].compliance, candidate[i], flux)) {
      return false;
    }
    Add(operator_rows[i], flux);
    Add(physical_rows[i], -device.cells[i].integrated_target_flux);
    if (device.cells[i].compliance > 0.0) {
      const double change = candidate[i] - device.cells[i].previous_pressure;
      if (!Product(device, device.cells[i].compliance, change, flux)) {
        return false;
      }
      Add(physical_rows[i], flux);
    }
  }
  for (std::size_t face = 0; face < device.face_count; ++face) {
    const auto link = device.links[face];
    if (link.conductance != 0.0) {
      const double tail = link.tail_cell_ordinal < count ? candidate[link.tail_cell_ordinal] : 0.0;
      const double head = link.head_cell_ordinal < count ? candidate[link.head_cell_ordinal] : 0.0;
      double flux = 0.0;
      if (!Product(device, link.conductance, tail - head, flux)) {
        return false;
      }
      AddIncident(device, operator_rows, link, flux);
    }
    double corrected = link.predicted_normal_velocity;
    if (link.open_area > 0.0) {
      const double tail = link.tail_cell_ordinal < count ? candidate[link.tail_cell_ordinal] : link.exterior_pressure;
      const double head = link.head_cell_ordinal < count ? candidate[link.head_cell_ordinal] : link.exterior_pressure;
      const double lifted_difference = tail - head + link.jump;
      double correction = 0.0;
      if (!Product(device, link.pressure_to_velocity_scale, lifted_difference, correction)) {
        return false;
      }
      corrected += correction;
    }
    if (!isfinite(corrected)) {
      return RangeFailure(device);
    }
    velocity[face] = corrected;
    double fluid_flux = 0.0;
    double wall_flux = 0.0;
    if (!Product(device, link.open_area, corrected, fluid_flux)) {
      return false;
    }
    if (!Product(device, link.blocked_area, link.wall_normal_velocity, wall_flux)) {
      return false;
    }
    // Preserve each row's prepared face order and fluid-before-wall incidence.
    if (link.tail_cell_ordinal < count) {
      Add(physical_rows[link.tail_cell_ordinal], fluid_flux);
      Add(physical_rows[link.tail_cell_ordinal], wall_flux);
    }
    if (link.head_cell_ordinal < count) {
      Add(physical_rows[link.head_cell_ordinal], -fluid_flux);
      Add(physical_rows[link.head_cell_ordinal], -wall_flux);
    }
  }
  auto& state = *device.state;
  state.residual_l2 = 0.0;
  state.maximum_scaled_residual = 0.0;
  state.maximum_defect = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    residual[i] = device.cells[i].integrated_rhs - Total(operator_rows[i].flux);
    const double scaled = fabs(residual[i]) / device.cells[i].fluid_volume;
    defect[i] = Total(physical_rows[i].flux) / device.cells[i].fluid_volume;
    const double identity = device.cells[i].fluid_volume * defect[i] + residual[i];
    const double absolute_terms = device.cells[i].rhs_absolute_term_sum + Total(operator_rows[i].absolute_terms)
        + Total(physical_rows[i].absolute_terms);
    if (!isfinite(scaled) || !isfinite(defect[i]) || !isfinite(identity) || !isfinite(absolute_terms)) {
      return RangeFailure(device);
    }
    if (fabs(identity) > (64.0 * DBL_EPSILON) * absolute_terms) {
      state.error = PressureError3D::kIdentity;
      return false;
    }
    state.residual_l2 = hypot(state.residual_l2, residual[i]);
    state.maximum_scaled_residual = fmax(state.maximum_scaled_residual, scaled);
    state.maximum_defect = fmax(state.maximum_defect, fabs(defect[i]));
  }
  if (!isfinite(state.residual_l2)) {
    return RangeFailure(device);
  }
  state.converged = state.residual_l2 <= device.integrated_limit && state.maximum_scaled_residual <= device.local_limit
      && state.maximum_defect <= device.local_limit;
  return true;
}
template <typename Scalar>
__device__ bool
ApplyLevel(PressureDevice3D device, const PressureLevel3D& level, const Scalar* pressure, Scalar* flux) {
  for (std::size_t i = 0; i < level.count; ++i) {
    Scalar row_flux = static_cast<Scalar>(device.coefficients[level.anchor_offset + i]) * pressure[i];
    for (auto offset = device.indices[level.row_offset + i]; offset < device.indices[level.row_offset + i + 1];
         ++offset) {
      const auto neighbor = device.indices[level.neighbor_offset + offset];
      const auto edge = device.indices[level.neighbor_edge_offset + offset];
      const Scalar difference = pressure[i] - pressure[neighbor];
      const Scalar edge_flux = static_cast<Scalar>(device.coefficients[level.edge_offset + edge]) * difference;
      row_flux += edge_flux;
    }
    if (!isfinite(row_flux)) {
      return RangeFailure(device);
    }
    flux[i] = row_flux;
  }
  return true;
}
template <typename Scalar>
__device__ bool
Terminal(PressureDevice3D device, const PressureLevel3D& level, const Scalar* residual, Scalar* correction) {
  for (std::size_t b = 0; b < level.block_count; ++b) {
    const auto block = device.blocks[level.block_offset + b];
    double block_correction[32];
    const auto* unknowns = device.indices + block.unknown_offset;
    const auto* factor = device.coefficients + block.factor_offset;
    for (std::size_t row = 0; row < block.count; ++row) {
      double flux = residual[unknowns[row]];
      for (std::size_t column = 0; column < row; ++column) {
        flux -= factor[row * block.count + column] * block_correction[column];
      }
      block_correction[row] = flux / factor[row * block.count + row];
    }
    for (std::size_t reverse = block.count; reverse > 0; --reverse) {
      const auto row = reverse - 1;
      double flux = block_correction[row];
      for (std::size_t column = row + 1; column < block.count; ++column) {
        flux -= factor[column * block.count + row] * block_correction[column];
      }
      block_correction[row] = flux / factor[row * block.count + row];
      if (!isfinite(block_correction[row])) {
        return RangeFailure(device);
      }
      if constexpr (std::is_same_v<Scalar, float>) {
        if (!FineValue(device, block_correction[row], correction[unknowns[row]])) {
          return false;
        }
      } else {
        correction[unknowns[row]] = block_correction[row];
      }
    }
  }
  return true;
}
// Each level has one Jacobi sweep on descent and one on ascent. Restriction
// visits fine unknowns in ordinal order, avoiding floating-point atomics.
template <typename Scalar>
__device__ bool
Descend(PressureDevice3D device, std::size_t ordinal, const Scalar* residual, Scalar* correction, Scalar* flux) {
  const auto level = device.levels[ordinal];
  constexpr Scalar weight = Scalar{2} / Scalar{3};
  for (std::size_t i = 0; i < level.count; ++i) {
    correction[i] = weight * (residual[i] / static_cast<Scalar>(device.coefficients[level.diagonal_offset + i]));
  }
  if (!ApplyLevel(device, level, correction, flux)) {
    return false;
  }
  const auto coarse = device.levels[ordinal + 1];
  auto* coarse_residual = device.coarse_scratch + coarse.scratch_offset;
  for (std::size_t i = 0; i < coarse.count; ++i) {
    coarse_residual[i] = 0.0;
  }
  for (std::size_t i = 0; i < level.count; ++i) {
    const Scalar remaining = residual[i] - flux[i];
    coarse_residual[device.indices[level.aggregate_offset + i]] += static_cast<double>(remaining);
  }
  for (std::size_t i = 0; i < coarse.count; ++i) {
    if (!isfinite(coarse_residual[i])) {
      return RangeFailure(device);
    }
  }
  return true;
}
template <typename Scalar>
__device__ bool
Ascend(PressureDevice3D device, std::size_t ordinal, const Scalar* residual, Scalar* correction, Scalar* flux) {
  const auto level = device.levels[ordinal];
  const auto coarse = device.levels[ordinal + 1];
  const auto* coarse_correction = device.coarse_scratch + device.level_unknown_count + coarse.scratch_offset;
  for (std::size_t i = 0; i < level.count; ++i) {
    const double coarse_pressure = coarse_correction[device.indices[level.aggregate_offset + i]];
    if constexpr (std::is_same_v<Scalar, float>) {
      float fine_pressure = 0.0F;
      if (!FineValue(device, coarse_pressure, fine_pressure)) {
        return false;
      }
      correction[i] += fine_pressure;
    } else {
      correction[i] += coarse_pressure;
    }
  }
  if (!ApplyLevel(device, level, correction, flux)) {
    return false;
  }
  constexpr Scalar weight = Scalar{2} / Scalar{3};
  for (std::size_t i = 0; i < level.count; ++i) {
    const Scalar remaining = residual[i] - flux[i];
    correction[i] += weight * (remaining / static_cast<Scalar>(device.coefficients[level.diagonal_offset + i]));
    if (!isfinite(correction[i])) {
      return RangeFailure(device);
    }
  }
  return true;
}
__device__ bool Vcycle(PressureDevice3D device, const float* residual, float* correction, float* flux) {
  if (device.level_count == 1) {
    return Terminal(device, device.levels[0], residual, correction);
  }
  if (!Descend(device, 0, residual, correction, flux)) {
    return false;
  }
  auto* coarse_residual = device.coarse_scratch;
  auto* coarse_correction = coarse_residual + device.level_unknown_count;
  auto* coarse_flux = coarse_correction + device.level_unknown_count;
  for (std::size_t ordinal = 1; ordinal + 1 < device.level_count; ++ordinal) {
    const auto offset = device.levels[ordinal].scratch_offset;
    if (!Descend(device, ordinal, coarse_residual + offset, coarse_correction + offset, coarse_flux + offset)) {
      return false;
    }
  }
  const auto terminal = device.levels[device.level_count - 1];
  if (!Terminal(
          device,
          terminal,
          coarse_residual + terminal.scratch_offset,
          coarse_correction + terminal.scratch_offset
      )) {
    return false;
  }
  for (std::size_t ordinal = device.level_count - 1; ordinal > 1; --ordinal) {
    const auto offset = device.levels[ordinal - 1].scratch_offset;
    if (!Ascend(device, ordinal - 1, coarse_residual + offset, coarse_correction + offset, coarse_flux + offset)) {
      return false;
    }
  }
  return Ascend(device, 0, residual, correction, flux);
}
__device__ bool Dot(PressureDevice3D device, const float* left, const float* right, double& sum) {
  sum = 0.0;
  for (std::size_t i = 0; i < device.unknown_count; ++i) {
    const double product = static_cast<double>(left[i]) * static_cast<double>(right[i]);
    sum += product;
  }
  return isfinite(sum) ? true : RangeFailure(device);
}
__device__ bool CopyTrueResidual(PressureDevice3D device) {
  auto* residual = device.fine_scratch + device.unknown_count;
  const auto* true_residual = device.output + 2 * device.cell_count;
  for (std::size_t i = 0; i < device.unknown_count; ++i) {
    if (!FineValue(device, true_residual[device.indices[device.unknown_map_offset + i]], residual[i])) {
      return false;
    }
  }
  return true;
}
__device__ bool RecursivePasses(PressureDevice3D device, const float* residual) {
  double norm = 0.0;
  double maximum = 0.0;
  for (std::size_t i = 0; i < device.unknown_count; ++i) {
    const double flux = residual[i];
    norm = hypot(norm, flux);
    const auto cell = device.indices[device.unknown_map_offset + i];
    maximum = fmax(maximum, fabs(flux) / device.cells[cell].fluid_volume);
  }
  return norm <= device.integrated_limit && maximum <= device.local_limit;
}
__global__ void InitializeMacPressure3DKernel(PressureDevice3D device) {
#if defined(__CUDA_ARCH__)
  device.state->compiled_architecture = __CUDA_ARCH__;
#endif
  for (std::size_t i = 0; i < device.cell_count; ++i) {
    device.output[i] = device.cells[i].initial_pressure;
  }
  if (!Evaluate(device) || device.state->converged) {
    return;
  }
  if (device.unknown_count == 0) {
    device.state->error = PressureError3D::kExhausted;
    return;
  }
  device.state->restart = true;
  CopyTrueResidual(device);
}
__global__ void IterateMacPressure3DKernel(PressureDevice3D device) {
#if defined(__CUDA_ARCH__)
  device.state->compiled_architecture = __CUDA_ARCH__;
#endif
  auto& state = *device.state;
  if (state.error != PressureError3D::kNone || state.converged) {
    return;
  }
  ++state.iterations;
  const auto count = device.unknown_count;
  auto* correction = device.fine_scratch;
  auto* residual = correction + count;
  auto* direction = residual + count;
  auto* direction_flux = direction + count;
  auto* preconditioned = direction_flux + count;
  auto* preconditioned_flux = preconditioned + count;
  if (!Vcycle(device, residual, preconditioned, preconditioned_flux)) {
    return;
  }
  double preconditioned_residual_inner_product = 0.0;
  if (!Dot(device, residual, preconditioned, preconditioned_residual_inner_product)) {
    return;
  }
  if (preconditioned_residual_inner_product <= 0.0) {
    state.error = PressureError3D::kCurvature;
    return;
  }
  if (state.restart) {
    for (std::size_t i = 0; i < count; ++i) {
      direction[i] = preconditioned[i];
    }
  } else {
    float beta = 0.0F;
    if (!FineValue(
            device,
            preconditioned_residual_inner_product / state.previous_preconditioned_residual_inner_product,
            beta
        )) {
      return;
    }
    for (std::size_t i = 0; i < count; ++i) {
      const float previous_direction = beta * direction[i];
      direction[i] = preconditioned[i] + previous_direction;
    }
  }
  if (!ApplyLevel(device, device.levels[0], direction, direction_flux)) {
    return;
  }
  double curvature = 0.0;
  if (!Dot(device, direction, direction_flux, curvature)) {
    return;
  }
  if (curvature <= 0.0) {
    state.error = PressureError3D::kCurvature;
    return;
  }
  float alpha = 0.0F;
  if (!FineValue(device, preconditioned_residual_inner_product / curvature, alpha)) {
    return;
  }
  for (std::size_t i = 0; i < count; ++i) {
    const float pressure_increment = alpha * direction[i];
    const float flux_increment = alpha * direction_flux[i];
    correction[i] += pressure_increment;
    residual[i] -= flux_increment;
    if (!isfinite(correction[i]) || !isfinite(residual[i])) {
      RangeFailure(device);
      return;
    }
  }
  state.restart = false;
  state.previous_preconditioned_residual_inner_product = preconditioned_residual_inner_product;
  const bool last = state.iterations == device.max_iterations;
  if (last || RecursivePasses(device, residual)) {
    if (!Evaluate(device) || state.converged) {
      return;
    }
    if (last) {
      state.error = PressureError3D::kExhausted;
      return;
    }
    for (std::size_t i = 0; i < device.cell_count; ++i) {
      device.output[i] = device.output[device.cell_count + i];
    }
    for (std::size_t i = 0; i < count; ++i) {
      correction[i] = 0.0F;
    }
    if (!CopyTrueResidual(device)) {
      return;
    }
    state.restart = true;
  }
}
}  // namespace

int LaunchMacPressure3D(PressureDevice3D device, bool initialize, void* stream) {
  void* arguments[] = {&device};
  const void* kernel = initialize ? reinterpret_cast<const void*>(InitializeMacPressure3DKernel)
                                  : reinterpret_cast<const void*>(IterateMacPressure3DKernel);
  return cudaLaunchKernel(kernel, dim3(1), dim3(1), arguments, 0, static_cast<cudaStream_t>(stream));
}
}  // namespace fluidity::cuda_internal
