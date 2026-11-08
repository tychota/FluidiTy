#include <cuda_runtime.h>

#include "transfer_bridge.h"

namespace fluidity::cuda_internal {
namespace {

__device__ double Dot(Vec2 lhs, Vec2 rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y;
}

// The finite magnitude test also rejects NaN and infinity. Division by a valid
// finite scale preserves nonfiniteness in the unscaled contribution.
__device__ bool IsQuantizable(double scaled) {
  return fabs(scaled) <= 0x1p52;
}

__device__ DepositError DepositComponent(
    const DepositParticle& particle,
    std::size_t component,
    DepositScales scales,
    DepositFaceAccumulator* face_accumulators,
    std::size_t face_count
) {
  const bool is_x_normal = component == 0;
  const Vec2 affine_row =
      is_x_normal ? Vec2{particle.affine.xx, particle.affine.xy} : Vec2{particle.affine.yx, particle.affine.yy};
  const Vec2 stress_row =
      is_x_normal ? Vec2{particle.stress.xx, particle.stress.xy} : Vec2{particle.stress.yx, particle.stress.yy};
  const double velocity = is_x_normal ? particle.velocity.x : particle.velocity.y;
  const auto& slots = particle.stencil.components[component];

  // Remove the weighted first moment from the affine velocity in the same order
  // as the CPU transfer. The original weight gradients determine stress force.
  Vec2 mean_offset{};
  for (const auto& slot : slots) {
    mean_offset = mean_offset + slot.offset * slot.weight;
  }
  for (const auto& slot : slots) {
    if (slot.face_index >= face_count) {
      return DepositError::kInvalidFaceIndex;
    }
    const Vec2 centered_offset = slot.offset - mean_offset;
    const double deposited_velocity = velocity + Dot(affine_row, centered_offset);
    const double mass_contribution = slot.weight * particle.mass;
    const double pure_momentum_contribution = mass_contribution * deposited_velocity;
    const double stress_force_contribution = -particle.reference_volume * Dot(stress_row, slot.gradient);
    const double scaled_mass_contribution = mass_contribution / scales.mass;
    const double scaled_pure_momentum_contribution = pure_momentum_contribution / scales.pure_momentum;
    const double scaled_stress_force_contribution = stress_force_contribution / scales.stress_force;
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
    auto& face_accumulator = face_accumulators[slot.face_index];
    atomicAdd(&face_accumulator.mass_sum, static_cast<unsigned long long>(quantized_mass_contribution));
    atomicAdd(
        &face_accumulator.pure_momentum_sum,
        static_cast<unsigned long long>(quantized_pure_momentum_contribution)
    );
    atomicAdd(&face_accumulator.stress_force_sum, static_cast<unsigned long long>(quantized_stress_force_contribution));
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

  for (std::size_t component = 0; component < kTransferComponents; ++component) {
    const auto error = DepositComponent(particles[particle_index], component, scales, face_accumulators, face_count);
    if (error != DepositError::kNone) {
      atomicMax(&state->error, static_cast<int>(error));
      return;
    }
  }
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

}  // namespace fluidity::cuda_internal
