#include <cuda_runtime.h>

#include <cfloat>

#include "transfer_bridge.h"

namespace fluidity::cuda_internal {
namespace {

__device__ double Dot(Vec2 lhs, Vec2 rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y;
}

// The finite magnitude test also rejects NaN and infinity. Division by a valid
// finite scale preserves nonfiniteness in the unscaled contribution.
__device__ bool IsQuantizable(double scaled_contribution) {
  return fabs(scaled_contribution) <= 0x1p52;
}

__device__ DepositError DepositComponent(
    const DepositParticle& particle,
    std::size_t velocity_component_index,
    DepositScales scales,
    DepositFaceAccumulator* face_accumulators,
    std::size_t face_count
) {
  const bool is_x_normal = velocity_component_index == 0;
  const Vec2 velocity_fit_slope_row = is_x_normal
      ? Vec2{particle.velocity_fit_slopes.xx, particle.velocity_fit_slopes.xy}
      : Vec2{particle.velocity_fit_slopes.yx, particle.velocity_fit_slopes.yy};
  const Vec2 kirchhoff_stress_row = is_x_normal ? Vec2{particle.kirchhoff_stress.xx, particle.kirchhoff_stress.xy}
                                                : Vec2{particle.kirchhoff_stress.yx, particle.kirchhoff_stress.yy};
  const double particle_velocity_component = is_x_normal ? particle.velocity.x : particle.velocity.y;
  const auto& component_slots = particle.stencil.component_slots[velocity_component_index];

  // Remove the weighted first moment from the affine velocity in the same order
  // as the CPU transfer. The original weight gradients determine stress force.
  Vec2 weighted_face_offset_mean{};
  for (const auto& slot : component_slots) {
    weighted_face_offset_mean = weighted_face_offset_mean + slot.particle_to_face_offset * slot.basis_weight;
  }
  for (const auto& slot : component_slots) {
    if (slot.face_slot >= face_count) {
      return DepositError::kInvalidFaceIndex;
    }
    const Vec2 centered_face_offset = slot.particle_to_face_offset - weighted_face_offset_mean;
    const double deposited_velocity = particle_velocity_component + Dot(velocity_fit_slope_row, centered_face_offset);
    const double mass_contribution = slot.basis_weight * particle.mass;
    const double pure_momentum_contribution = mass_contribution * deposited_velocity;
    const double stress_force_contribution =
        -particle.reference_volume * Dot(kirchhoff_stress_row, slot.basis_gradient);
    const double scaled_mass_contribution = mass_contribution / scales.mass_per_quantum;
    const double scaled_pure_momentum_contribution = pure_momentum_contribution / scales.pure_momentum_per_quantum;
    const double scaled_stress_force_contribution = stress_force_contribution / scales.stress_force_per_quantum;
    if (!IsQuantizable(scaled_mass_contribution) || !IsQuantizable(scaled_pure_momentum_contribution)
        || !IsQuantizable(scaled_stress_force_contribution)) {
      return DepositError::kOutOfRange;
    }

    const long long quantized_mass_contribution = __double2ll_rn(scaled_mass_contribution);
    const long long quantized_pure_momentum_contribution = __double2ll_rn(scaled_pure_momentum_contribution);
    const long long quantized_stress_force_contribution = __double2ll_rn(scaled_stress_force_contribution);
    // Each particle contributes at most one term of magnitude 2^52 to this face.
    // Across at most 1024 particles the signed sum is representable and the
    // unsigned modular sum has the same bits, regardless of atomic order.
    auto& face_accumulator = face_accumulators[slot.face_slot];
    atomicAdd(&face_accumulator.mass_quanta_sum, static_cast<unsigned long long>(quantized_mass_contribution));
    atomicAdd(
        &face_accumulator.pure_momentum_quanta_sum,
        static_cast<unsigned long long>(quantized_pure_momentum_contribution)
    );
    atomicAdd(
        &face_accumulator.stress_force_quanta_sum,
        static_cast<unsigned long long>(quantized_stress_force_contribution)
    );
  }
  return DepositError::kNone;
}

__global__ void DepositKernel(
    const DepositParticle* particles,
    std::size_t particle_count,
    DepositScales scales,
    DepositFaceAccumulator* face_accumulators,
    std::size_t face_count,
    DepositKernelState* state
) {
  const std::size_t particle_index = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__)
  if (particle_index == 0) {
    state->compiled_architecture = __CUDA_ARCH__;
  }
#endif
  if (particle_index >= particle_count) {
    return;
  }

  for (std::size_t velocity_component_index = 0; velocity_component_index < kTransferComponents;
       ++velocity_component_index) {
    const auto error =
        DepositComponent(particles[particle_index], velocity_component_index, scales, face_accumulators, face_count);
    if (error != DepositError::kNone) {
      atomicMax(&state->error, static_cast<int>(error));
      return;
    }
  }
}

struct GatherComponentResult {
  double pic_velocity;
  double grid_velocity_increment;
  Vec2 velocity_fit_slope_row;
  Vec2 velocity_gradient_row;
};

// Scale the symmetric moment before comparing its smaller and larger eigenvalues.
// For a positive largest eigenvalue, det = smaller * larger gives the rank test
// without subtracting nearly equal values to recover the smaller eigenvalue.
__device__ GatherError SolveGatherAffineMoment(
    double centered_moment_xx,
    double centered_moment_xy,
    double centered_moment_yy,
    Vec2 centered_offset_velocity_moment,
    Vec2& velocity_fit_slope_row
) {
  const double moment_scale = fmax(centered_moment_xx, centered_moment_yy);
  if (moment_scale == 0.0) {
    return GatherError::kUnsafeAffineMoment;
  }
  const double normalized_moment_xx = centered_moment_xx / moment_scale;
  const double normalized_moment_xy = centered_moment_xy / moment_scale;
  const double normalized_moment_yy = centered_moment_yy / moment_scale;
  const double normalized_moment_determinant =
      fma(normalized_moment_xx, normalized_moment_yy, -normalized_moment_xy * normalized_moment_xy);
  const double largest_normalized_eigenvalue = 0.5
      * (normalized_moment_xx + normalized_moment_yy
         + hypot(normalized_moment_xx - normalized_moment_yy, 2.0 * normalized_moment_xy));
  constexpr double kMinimumAffineMomentEigenvalueRatio = 2.0 * DBL_EPSILON;
  const double normalized_determinant_threshold =
      kMinimumAffineMomentEigenvalueRatio * largest_normalized_eigenvalue * largest_normalized_eigenvalue;
  if (!isfinite(normalized_moment_determinant) || !isfinite(normalized_determinant_threshold)) {
    return GatherError::kNonfiniteArithmetic;
  }
  if (normalized_moment_determinant <= normalized_determinant_threshold) {
    return GatherError::kUnsafeAffineMoment;
  }

  const Vec2 scaled_centered_offset_velocity_moment{
      .x = centered_offset_velocity_moment.x / moment_scale,
      .y = centered_offset_velocity_moment.y / moment_scale,
  };
  const Vec2 velocity_fit_slope_numerator{
      .x = normalized_moment_yy * scaled_centered_offset_velocity_moment.x
          - normalized_moment_xy * scaled_centered_offset_velocity_moment.y,
      .y = normalized_moment_xx * scaled_centered_offset_velocity_moment.y
          - normalized_moment_xy * scaled_centered_offset_velocity_moment.x,
  };
  velocity_fit_slope_row = {
      .x = velocity_fit_slope_numerator.x / normalized_moment_determinant,
      .y = velocity_fit_slope_numerator.y / normalized_moment_determinant,
  };
  if (!isfinite(velocity_fit_slope_row.x) || !isfinite(velocity_fit_slope_row.y)) {
    return GatherError::kNonfiniteArithmetic;
  }
  return GatherError::kNone;
}

__device__ GatherError GatherComponent(
    const TransferSlot (&component_slots)[kTransferSlots],
    GatherFaceFields face_fields,
    GatherComponentResult& component_result
) {
  double old_pic_velocity = 0.0;
  double new_pic_velocity = 0.0;
  Vec2 weighted_face_offset_mean{};
  Vec2 velocity_gradient_row{};
  // Sum both PIC fields in the same order so identical inputs give exactly zero
  // increment. The true derivative uses the unmodified basis gradients.
  for (const auto& slot : component_slots) {
    if (slot.face_slot >= face_fields.face_count) {
      return GatherError::kInvalidFaceIndex;
    }
    const double old_face_velocity_sample = face_fields.old_face_velocities[slot.face_slot];
    const double new_face_velocity_sample = face_fields.new_face_velocities[slot.face_slot];
    old_pic_velocity += slot.basis_weight * old_face_velocity_sample;
    new_pic_velocity += slot.basis_weight * new_face_velocity_sample;
    weighted_face_offset_mean = weighted_face_offset_mean + slot.particle_to_face_offset * slot.basis_weight;
    velocity_gradient_row = velocity_gradient_row + slot.basis_gradient * new_face_velocity_sample;
  }
  const double grid_velocity_increment = new_pic_velocity - old_pic_velocity;
  if (!isfinite(grid_velocity_increment) || !isfinite(velocity_gradient_row.x) || !isfinite(velocity_gradient_row.y)) {
    return GatherError::kNonfiniteArithmetic;
  }

  // Form the full centered second moment and the offset-velocity cross moment.
  // Centering the offsets makes the weighted constant velocity term vanish.
  double centered_moment_xx = 0.0;
  double centered_moment_xy = 0.0;
  double centered_moment_yy = 0.0;
  Vec2 centered_offset_velocity_moment{};
  for (const auto& slot : component_slots) {
    const Vec2 centered_face_offset = slot.particle_to_face_offset - weighted_face_offset_mean;
    const Vec2 weighted_centered_face_offset = centered_face_offset * slot.basis_weight;
    const double new_face_velocity_sample = face_fields.new_face_velocities[slot.face_slot];
    const double weighted_new_face_velocity_sample = slot.basis_weight * new_face_velocity_sample;
    centered_moment_xx += weighted_centered_face_offset.x * centered_face_offset.x;
    centered_moment_xy += weighted_centered_face_offset.x * centered_face_offset.y;
    centered_moment_yy += weighted_centered_face_offset.y * centered_face_offset.y;
    centered_offset_velocity_moment =
        centered_offset_velocity_moment + centered_face_offset * weighted_new_face_velocity_sample;
  }
  const bool finite_centered_moment =
      isfinite(centered_moment_xx) && isfinite(centered_moment_xy) && isfinite(centered_moment_yy);
  if (!finite_centered_moment || !isfinite(centered_offset_velocity_moment.x)
      || !isfinite(centered_offset_velocity_moment.y)) {
    return GatherError::kNonfiniteArithmetic;
  }

  Vec2 velocity_fit_slope_row;
  const auto velocity_fit_error = SolveGatherAffineMoment(
      centered_moment_xx,
      centered_moment_xy,
      centered_moment_yy,
      centered_offset_velocity_moment,
      velocity_fit_slope_row
  );
  if (velocity_fit_error != GatherError::kNone) {
    return velocity_fit_error;
  }
  component_result = {
      .pic_velocity = new_pic_velocity,
      .grid_velocity_increment = grid_velocity_increment,
      .velocity_fit_slope_row = velocity_fit_slope_row,
      .velocity_gradient_row = velocity_gradient_row,
  };
  return GatherError::kNone;
}

__global__ void GatherKernel(
    const TransferStencil* particle_stencils,
    std::size_t particle_count,
    GatherFaceFields face_fields,
    GatherParticleResult* particle_results,
    GatherKernelState* gather_kernel_state
) {
  const std::size_t particle_index = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__)
  if (particle_index == 0) {
    gather_kernel_state->compiled_architecture = __CUDA_ARCH__;
  }
#endif
  if (particle_index >= particle_count) {
    return;
  }

  const auto& particle_stencil = particle_stencils[particle_index];
  GatherComponentResult component_results[kTransferComponents];
  for (std::size_t component_index = 0; component_index < kTransferComponents; ++component_index) {
    const auto gather_error = GatherComponent(
        particle_stencil.component_slots[component_index],
        face_fields,
        component_results[component_index]
    );
    if (gather_error != GatherError::kNone) {
      atomicOr(&gather_kernel_state->error_bits, static_cast<unsigned int>(gather_error));
      return;
    }
  }
  const auto& x_velocity_result = component_results[0];
  const auto& y_velocity_result = component_results[1];
  particle_results[particle_index] = {
      .pic_velocity = {x_velocity_result.pic_velocity, y_velocity_result.pic_velocity},
      .grid_velocity_increment = {x_velocity_result.grid_velocity_increment, y_velocity_result.grid_velocity_increment},
      .velocity_fit_slopes =
          {
              .xx = x_velocity_result.velocity_fit_slope_row.x,
              .xy = x_velocity_result.velocity_fit_slope_row.y,
              .yx = y_velocity_result.velocity_fit_slope_row.x,
              .yy = y_velocity_result.velocity_fit_slope_row.y,
          },
      .velocity_gradient =
          {
              .xx = x_velocity_result.velocity_gradient_row.x,
              .xy = x_velocity_result.velocity_gradient_row.y,
              .yx = y_velocity_result.velocity_gradient_row.x,
              .yy = y_velocity_result.velocity_gradient_row.y,
          },
  };
}

}  // namespace

int LaunchDeposit(
    const DepositParticle* particles,
    std::size_t particle_count,
    DepositScales scales,
    DepositFaceAccumulator* face_accumulators,
    std::size_t face_count,
    DepositKernelState* state,
    void* stream
) {
  constexpr unsigned int kThreadsPerBlock = 128;
  const auto particle_blocks = static_cast<unsigned int>((particle_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const unsigned int block_count = particle_blocks == 0 ? 1 : particle_blocks;
  void* arguments[] = {&particles, &particle_count, &scales, &face_accumulators, &face_count, &state};
  return cudaLaunchKernel(
      DepositKernel,
      dim3(block_count),
      dim3(kThreadsPerBlock),
      arguments,
      0,
      static_cast<cudaStream_t>(stream)
  );
}

int LaunchGather(
    const TransferStencil* particle_stencils,
    std::size_t particle_count,
    GatherFaceFields face_fields,
    GatherParticleResult* particle_results,
    GatherKernelState* gather_kernel_state,
    void* stream
) {
  constexpr unsigned int kThreadsPerBlock = 128;
  const auto block_count = static_cast<unsigned int>((particle_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  void* arguments[] = {&particle_stencils, &particle_count, &face_fields, &particle_results, &gather_kernel_state};
  return cudaLaunchKernel(
      GatherKernel,
      dim3(block_count),
      dim3(kThreadsPerBlock),
      arguments,
      0,
      static_cast<cudaStream_t>(stream)
  );
}

}  // namespace fluidity::cuda_internal
