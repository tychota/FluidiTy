#include <cuda_runtime.h>

#include <cstddef>

#include "transfer_3d_bridge.h"

namespace fluidity::cuda_internal {
namespace {

constexpr unsigned int kThreadsPerBlock = 128;
constexpr unsigned int kMaxBlockCount = 65535;

// All launch paths use a positive extent. Grid-stride loops also cover large
// gather and clear extents without narrowing the work count to a launch dimension.
unsigned int LaunchBlockCount(std::size_t work_count) {
  const std::size_t block_count = (work_count - 1) / kThreadsPerBlock + 1;
  return block_count > kMaxBlockCount ? kMaxBlockCount : static_cast<unsigned int>(block_count);
}

__device__ std::size_t GlobalThreadIndex() {
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ std::size_t GridThreadCount() {
  return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

__device__ void RecordCompiledArchitecture(MacTransferKernelReport3D* kernel_report) {
#if defined(__CUDA_ARCH__)
  if (GlobalThreadIndex() == 0) {
    kernel_report->compiled_architecture = __CUDA_ARCH__;
  }
#endif
}

__device__ bool IsFinite(Vec3 vector) {
  return isfinite(vector.x) && isfinite(vector.y) && isfinite(vector.z);
}

__device__ bool IsScaledContributionInRange(double scaled_contribution, double scaled_contribution_magnitude_limit) {
  return isfinite(scaled_contribution) && fabs(scaled_contribution) <= scaled_contribution_magnitude_limit;
}

__device__ bool DepositParticleQuanta3D(
    const DeviceMacDepositParticle3D& particle,
    DeviceMacDepositQuantization3D scales,
    DeviceMacFaceQuantaAccumulator3D* face_accumulators
) {
  const double particle_velocity_components[]{
      particle.particle_velocity.x,
      particle.particle_velocity.y,
      particle.particle_velocity.z
  };
  const Vec3 particle_velocity_slope_rows[]{
      particle.particle_velocity_slopes.x_row,
      particle.particle_velocity_slopes.y_row,
      particle.particle_velocity_slopes.z_row,
  };
  // Traverse each normal component in the same sample order as the CPU.
  for (std::size_t face_normal_axis_ordinal = 0; face_normal_axis_ordinal < kMacComponentCount3D;
       ++face_normal_axis_ordinal) {
    const Vec3 particle_velocity_slope_row = particle_velocity_slope_rows[face_normal_axis_ordinal];
    const double particle_normal_velocity = particle_velocity_components[face_normal_axis_ordinal];
    for (const auto& sample : particle.stencil.face_samples_by_normal[face_normal_axis_ordinal]) {
      const double affine_normal_velocity_offset =
          (particle_velocity_slope_row.x * sample.face_offset_from_particle.x
           + particle_velocity_slope_row.y * sample.face_offset_from_particle.y)
          + particle_velocity_slope_row.z * sample.face_offset_from_particle.z;
      const double particle_normal_velocity_at_face = particle_normal_velocity + affine_normal_velocity_offset;
      const double mass_contribution = sample.weight * particle.particle_mass;
      const double pure_momentum_contribution = mass_contribution * particle_normal_velocity_at_face;
      if (!isfinite(mass_contribution) || !isfinite(pure_momentum_contribution)) {
        return false;
      }
      const double scaled_mass_contribution = mass_contribution / scales.mass_per_quantum;
      const double scaled_pure_momentum_contribution = pure_momentum_contribution / scales.pure_momentum_per_quantum;
      const bool scaled_mass_in_range =
          IsScaledContributionInRange(scaled_mass_contribution, scales.scaled_contribution_magnitude_limit);
      const bool scaled_pure_momentum_in_range =
          IsScaledContributionInRange(scaled_pure_momentum_contribution, scales.scaled_contribution_magnitude_limit);
      if (!scaled_mass_in_range || !scaled_pure_momentum_in_range) {
        return false;
      }
      const long long quantized_mass_contribution = __double2ll_rn(scaled_mass_contribution);
      const long long quantized_pure_momentum_contribution = __double2ll_rn(scaled_pure_momentum_contribution);
      auto& face_accumulator = face_accumulators[sample.face_slot];
      // A particle contributes once per owned face. The per-contribution guard
      // bounds the sum of absolute signed quanta by 2^62 in each channel.
      atomicAdd(&face_accumulator.mass_quanta_sum, static_cast<unsigned long long>(quantized_mass_contribution));
      atomicAdd(
          &face_accumulator.pure_momentum_quanta_sum,
          static_cast<unsigned long long>(quantized_pure_momentum_contribution)
      );
    }
  }
  return true;
}

__global__ void DepositMacQuanta3DKernel(
    const DeviceMacDepositParticle3D* particles,
    std::size_t particle_count,
    DeviceMacDepositQuantization3D scales,
    DeviceMacFaceQuantaAccumulator3D* face_accumulators,
    MacTransferKernelReport3D* kernel_report
) {
  RecordCompiledArchitecture(kernel_report);
  for (std::size_t particle_index = GlobalThreadIndex(); particle_index < particle_count;
       particle_index += GridThreadCount()) {
    if (!DepositParticleQuanta3D(particles[particle_index], scales, face_accumulators)) {
      atomicExch(&kernel_report->arithmetic_error, static_cast<int>(MacTransferArithmeticError3D::kOutOfRange));
    }
  }
}

__global__ void ClearDepositMacQuanta3DKernel(
    DeviceMacFaceQuantaAccumulator3D* face_accumulators,
    std::size_t face_count,
    MacTransferKernelReport3D* kernel_report
) {
  RecordCompiledArchitecture(kernel_report);
  for (std::size_t face_slot = GlobalThreadIndex(); face_slot < face_count; face_slot += GridThreadCount()) {
    face_accumulators[face_slot] = {};
  }
}

__device__ void AddScaled(Vec3& sum, Vec3 vector, double scale) {
  sum.x += vector.x * scale;
  sum.y += vector.y * scale;
  sum.z += vector.z * scale;
}

__device__ bool GatherParticle3D(
    const DeviceMacParticleStencil3D& stencil,
    DeviceMacFaceVelocityFields3D face_fields,
    double cell_width,
    DeviceMacParticleGatherResult3D& result
) {
  double pic_velocities[kMacComponentCount3D];
  double grid_velocity_increments[kMacComponentCount3D];
  Vec3 fitted_velocity_slope_rows[kMacComponentCount3D];
  Vec3 velocity_gradient_rows[kMacComponentCount3D];
  for (std::size_t axis_index = 0; axis_index < kMacComponentCount3D; ++axis_index) {
    double old_pic_velocity = 0.0;
    double new_pic_velocity = 0.0;
    Vec3 normalized_offset_velocity_moment{};
    Vec3 velocity_gradient_row{};
    // Keep the host stencil's (z,y,x) sum order within each component.
    for (const auto& sample : stencil.face_samples_by_normal[axis_index]) {
      const double old_face_velocity = face_fields.old_face_velocities[sample.face_slot];
      const double new_face_velocity = face_fields.new_face_velocities[sample.face_slot];
      const double weighted_new_face_velocity = sample.weight * new_face_velocity;
      old_pic_velocity += sample.weight * old_face_velocity;
      new_pic_velocity += weighted_new_face_velocity;
      const Vec3 normalized_face_displacement{
          .x = sample.face_offset_from_particle.x / cell_width,
          .y = sample.face_offset_from_particle.y / cell_width,
          .z = sample.face_offset_from_particle.z / cell_width,
      };
      AddScaled(normalized_offset_velocity_moment, normalized_face_displacement, weighted_new_face_velocity);
      AddScaled(velocity_gradient_row, sample.weight_gradient, new_face_velocity);
    }
    // The normalized complete-support second moment is I/4. Use the
    // normalized displacement to restore slopes without forming h squared.
    const Vec3 fitted_velocity_slope_row{
        .x = (4.0 * normalized_offset_velocity_moment.x) / cell_width,
        .y = (4.0 * normalized_offset_velocity_moment.y) / cell_width,
        .z = (4.0 * normalized_offset_velocity_moment.z) / cell_width,
    };
    const double grid_velocity_increment = new_pic_velocity - old_pic_velocity;
    if (!isfinite(new_pic_velocity) || !isfinite(grid_velocity_increment) || !IsFinite(fitted_velocity_slope_row)
        || !IsFinite(velocity_gradient_row)) {
      return false;
    }
    pic_velocities[axis_index] = new_pic_velocity;
    grid_velocity_increments[axis_index] = grid_velocity_increment;
    fitted_velocity_slope_rows[axis_index] = fitted_velocity_slope_row;
    velocity_gradient_rows[axis_index] = velocity_gradient_row;
  }
  result = {
      .new_pic_velocity = {pic_velocities[0], pic_velocities[1], pic_velocities[2]},
      .grid_velocity_increment =
          {grid_velocity_increments[0], grid_velocity_increments[1], grid_velocity_increments[2]},
      .fitted_velocity_slopes =
          {fitted_velocity_slope_rows[0], fitted_velocity_slope_rows[1], fitted_velocity_slope_rows[2]},
      .velocity_gradient = {velocity_gradient_rows[0], velocity_gradient_rows[1], velocity_gradient_rows[2]},
  };
  return true;
}

__global__ void GatherMac3DKernel(
    const DeviceMacParticleStencil3D* particle_stencils,
    std::size_t particle_count,
    DeviceMacFaceVelocityFields3D face_fields,
    double cell_width,
    DeviceMacParticleGatherResult3D* particle_results,
    MacTransferKernelReport3D* kernel_report
) {
  RecordCompiledArchitecture(kernel_report);
  for (std::size_t particle_index = GlobalThreadIndex(); particle_index < particle_count;
       particle_index += GridThreadCount()) {
    if (!GatherParticle3D(
            particle_stencils[particle_index],
            face_fields,
            cell_width,
            particle_results[particle_index]
        )) {
      atomicExch(&kernel_report->arithmetic_error, static_cast<int>(MacTransferArithmeticError3D::kOutOfRange));
    }
  }
}

}  // namespace

int LaunchDeposit3D(
    const DeviceMacDepositParticle3D* particles,
    std::size_t particle_count,
    DeviceMacDepositQuantization3D scales,
    DeviceMacFaceQuantaAccumulator3D* face_accumulators,
    std::size_t face_count,
    MacTransferKernelReport3D* kernel_report,
    void* stream
) {
  if (particle_count == 0) {
    void* arguments[]{&face_accumulators, &face_count, &kernel_report};
    return cudaLaunchKernel(
        ClearDepositMacQuanta3DKernel,
        dim3(LaunchBlockCount(face_count)),
        dim3(kThreadsPerBlock),
        arguments,
        0,
        static_cast<cudaStream_t>(stream)
    );
  }
  void* arguments[]{&particles, &particle_count, &scales, &face_accumulators, &kernel_report};
  return cudaLaunchKernel(
      DepositMacQuanta3DKernel,
      dim3(LaunchBlockCount(particle_count)),
      dim3(kThreadsPerBlock),
      arguments,
      0,
      static_cast<cudaStream_t>(stream)
  );
}

int LaunchGather3D(
    const DeviceMacParticleStencil3D* particle_stencils,
    std::size_t particle_count,
    DeviceMacFaceVelocityFields3D face_fields,
    double cell_width,
    DeviceMacParticleGatherResult3D* particle_results,
    MacTransferKernelReport3D* kernel_report,
    void* stream
) {
  void* arguments[]{&particle_stencils, &particle_count, &face_fields, &cell_width, &particle_results, &kernel_report};
  return cudaLaunchKernel(
      GatherMac3DKernel,
      dim3(LaunchBlockCount(particle_count)),
      dim3(kThreadsPerBlock),
      arguments,
      0,
      static_cast<cudaStream_t>(stream)
  );
}

}  // namespace fluidity::cuda_internal
