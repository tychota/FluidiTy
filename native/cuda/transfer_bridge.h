#pragma once

#include <cstddef>

#include "fluidity/base/types.h"

#if !defined(__CUDACC__)
#include "fluidity/host/grid.h"
#endif

namespace fluidity::cuda_internal {

inline constexpr std::size_t kMaxTransferParticles = 1024;
inline constexpr std::size_t kMaxTransferFaces = 4096;
inline constexpr std::size_t kTransferComponents = 2;
inline constexpr std::size_t kTransferSlots = 9;

struct TransferSlot {
  std::size_t face_slot;  // Flat MAC face slot in x-normal then y-normal face order.
  double basis_weight;
  Vec2 basis_gradient;  // Derivative with respect to particle position, in inverse meters.
  Vec2 particle_to_face_offset;  // Face position minus particle position, in meters.
};

struct TransferStencil {
  // X-normal then y-normal. Each component has y varying fastest within each x slot.
  TransferSlot component_slots[kTransferComponents][kTransferSlots];
};

#if !defined(__CUDACC__)
// Prepares a complete stencil after FaceCount validates the grid and the caller
// bounds its face count. Propagates invalid support and arithmetic errors.
absl::Status
PrepareTransferStencil(const Grid2D& mac_grid, Vec2 particle_world_position, TransferStencil& particle_stencil);
#endif

struct DepositParticle {
  TransferStencil stencil;
  double mass;
  Vec2 velocity;
  Mat2 velocity_fit_slopes;
  double reference_volume;
  Mat2 kirchhoff_stress;
};

struct DepositScales {
  double mass;
  double pure_momentum;
  double stress_force;
};

struct DepositFaceAccumulator {
  // The device adds signed contributions modulo 2^64 using unsigned atomics.
  unsigned long long mass_quanta_sum;
  unsigned long long pure_momentum_quanta_sum;
  unsigned long long stress_force_quanta_sum;
};

enum class DepositError : int {
  kNone,
  kOutOfRange,
  kInvalidFaceIndex,
};

struct DepositKernelState {
  int compiled_architecture;
  int error;  // DepositError value, updated atomically by failing threads.
};

// Launches one thread per particle and writes the image architecture even for an
// empty batch. Face accumulators and state start at zero and all buffers outlive
// stream completion. Returns the immediate CUDA runtime launch code.
int LaunchDeposit(
    const DepositParticle* particles,
    std::size_t particle_count,
    DepositScales scales,
    DepositFaceAccumulator* face_accumulators,
    std::size_t face_count,
    DepositKernelState* state,
    void* stream
);

struct GatherFaceFields {
  const double* old_face_velocities;
  const double* new_face_velocities;
  std::size_t face_count;
};

struct GatherParticleResult {
  Vec2 pic_velocity;
  Vec2 grid_velocity_increment;
  Mat2 velocity_fit_slopes;
  Mat2 velocity_gradient;
};

enum class GatherError : unsigned int {
  kNone = 0,
  kNonfiniteArithmetic = 1U << 0,
  kUnsafeAffineMoment = 1U << 1,
  kInvalidFaceIndex = 1U << 2,
};

struct GatherKernelState {
  int compiled_architecture;
  unsigned int error_bits;  // GatherError bits, combined atomically across particles.
};

// Launches one thread per particle to gather both velocity components. Particle
// count is positive. State starts at zero and all buffers outlive stream
// completion. Returns the immediate CUDA runtime launch code.
int LaunchGather(
    const TransferStencil* particle_stencils,
    std::size_t particle_count,
    GatherFaceFields face_fields,
    GatherParticleResult* particle_results,
    GatherKernelState* gather_kernel_state,
    void* stream
);

}  // namespace fluidity::cuda_internal
