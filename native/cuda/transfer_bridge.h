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
  std::size_t face_index;
  double weight;
  Vec2 gradient;  // Derivative with respect to particle position, in inverse meters.
  Vec2 offset;  // Face position minus particle position, in meters.
};

struct TransferStencil {
  // X-normal then y-normal. Each component has y varying fastest within each x slot.
  TransferSlot components[kTransferComponents][kTransferSlots];
};

#if !defined(__CUDACC__)
// Prepares a complete stencil after FaceCount validates the grid and the caller
// bounds its face count. Propagates invalid support and arithmetic errors.
absl::Status PrepareTransferStencil(const Grid2D& grid, Vec2 position, TransferStencil& stencil);
#endif

struct DepositParticle {
  TransferStencil stencil;
  double mass;
  Vec2 velocity;
  Mat2 affine;
  double reference_volume;
  Mat2 stress;
};

struct DepositScales {
  double mass;
  double pure_momentum;
  double stress_force;
};

struct DepositFaceAccumulator {
  // The device adds signed contributions modulo 2^64 using unsigned atomics.
  unsigned long long mass_sum;
  unsigned long long pure_momentum_sum;
  unsigned long long stress_force_sum;
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

}  // namespace fluidity::cuda_internal
