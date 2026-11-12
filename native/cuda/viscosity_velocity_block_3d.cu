#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

#include "viscosity_velocity_block_3d.cuh"

namespace fluidity::cuda_internal {
namespace {

// The reference device solve follows CPU slot and row order for deterministic
// weighted strain, transpose scatter, and PCG reductions.
__device__ void Apply(ViscousVelocityDevice3D device, const double* velocity) {
  for (std::size_t face = 0; face < device.face_count; ++face) {
    device.applied[face] = device.faces[face].inertia * velocity[face];
  }
  for (std::size_t row = 0; row < device.row_count; ++row) {
    const std::size_t first = device.row_offsets[row];
    const std::size_t end = device.row_offsets[row + 1];
    double weighted_strain = 0.0;
    for (std::size_t term_index = first; term_index < end; ++term_index) {
      const auto term = device.terms[term_index];
      weighted_strain += term.weighted_coefficient * velocity[term.face_slot];
    }
    for (std::size_t term_index = first; term_index < end; ++term_index) {
      const auto term = device.terms[term_index];
      device.applied[term.face_slot] += term.weighted_coefficient * weighted_strain;
    }
  }
}

__device__ double Precondition(ViscousVelocityDevice3D device) {
  double product = 0.0;
  for (std::size_t face = 0; face < device.face_count; ++face) {
    const auto data = device.faces[face];
    const double residual = device.residual[face];
    const double correction = data.inertia == 0.0 ? 0.0 : residual / data.diagonal;
    device.preconditioned[face] = correction;
    product += residual * correction;
  }
  return product;
}

__global__ void Solve(ViscousVelocityDevice3D device) {
  auto& state = *device.state;
  state = {};
#if defined(__CUDA_ARCH__)
  state.compiled_architecture = __CUDA_ARCH__;
#endif
  for (std::size_t face = 0; face < device.face_count; ++face) {
    device.velocity[face] = 0.0;
    device.residual[face] = device.faces[face].rhs;
  }
  if (device.max_rhs == 0.0) {
    return;
  }
  double residual_product = Precondition(device);
  for (std::size_t face = 0; face < device.face_count; ++face) {
    device.direction[face] = device.preconditioned[face];
  }
  while (state.iterations < device.max_iterations) {
    if (!isfinite(residual_product) || residual_product <= 0.0) {
      state.error = ViscousVelocityError3D::kRange;
      return;
    }
    Apply(device, device.direction);
    double curvature = 0.0;
    for (std::size_t face = 0; face < device.face_count; ++face) {
      if (!isfinite(device.applied[face])) {
        state.error = ViscousVelocityError3D::kRange;
        return;
      }
      curvature += device.direction[face] * device.applied[face];
    }
    if (!isfinite(curvature) || curvature <= 0.0) {
      state.error = ViscousVelocityError3D::kCurvature;
      return;
    }
    const double step = residual_product / curvature;
    if (!isfinite(step) || fabs(step) < DBL_MIN) {
      state.error = ViscousVelocityError3D::kRange;
      return;
    }
    double max_residual = 0.0;
    for (std::size_t face = 0; face < device.face_count; ++face) {
      device.velocity[face] += step * device.direction[face];
      device.residual[face] -= step * device.applied[face];
      if (!isfinite(device.velocity[face]) || !isfinite(device.residual[face])) {
        state.error = ViscousVelocityError3D::kRange;
        return;
      }
      max_residual = fmax(max_residual, fabs(device.residual[face]));
    }
    ++state.iterations;
    const bool check_true_residual = max_residual <= device.tolerance || state.iterations == device.max_iterations;
    if (check_true_residual) {
      Apply(device, device.velocity);
      max_residual = 0.0;
      for (std::size_t face = 0; face < device.face_count; ++face) {
        device.residual[face] = device.faces[face].rhs - device.applied[face];
        if (!isfinite(device.residual[face])) {
          state.error = ViscousVelocityError3D::kRange;
          return;
        }
        max_residual = fmax(max_residual, fabs(device.residual[face]));
      }
      if (max_residual <= device.tolerance) {
        state.max_residual = max_residual;
        return;
      }
    }
    const double next_product = Precondition(device);
    const double conjugacy = check_true_residual ? 0.0 : next_product / residual_product;
    for (std::size_t face = 0; face < device.face_count; ++face) {
      device.direction[face] = device.preconditioned[face] + conjugacy * device.direction[face];
    }
    residual_product = next_product;
  }
  state.error = ViscousVelocityError3D::kExhausted;
}

}  // namespace

cudaError_t LaunchViscousVelocityBlock3D(ViscousVelocityDevice3D device, cudaStream_t stream) {
  Solve<<<1, 1, 0, stream>>>(device);
  return cudaGetLastError();
}

}  // namespace fluidity::cuda_internal
