#pragma once

#include <cstddef>

#include "fluidity/base/mac_types_3d.h"

namespace fluidity::cuda_internal {

inline constexpr std::size_t kMacComponentCount3D = 3;
inline constexpr std::size_t kMacSupportSize3D = 27;

struct DeviceMacParticleStencil3D {
  // X, Y, Z normal components, each with all owned samples in (z,y,x) order.
  MacFaceStencilSample3D face_samples_by_normal[kMacComponentCount3D][kMacSupportSize3D];
};

struct DeviceMacDepositParticle3D {
  DeviceMacParticleStencil3D stencil;
  double particle_mass;
  Vec3 particle_velocity;
  Mat3 particle_velocity_slopes;
};

struct DeviceMacDepositQuantization3D {
  double mass_per_quantum;
  double pure_momentum_per_quantum;
  double scaled_contribution_magnitude_limit;
};

struct DeviceMacFaceQuantaAccumulator3D {
  // CUDA supplies unsigned 64-bit atomic addition. The host decodes the
  // modulo-2^64 sums after a successful kernel, under the signed 2^62 bound.
  unsigned long long mass_quanta_sum;
  unsigned long long pure_momentum_quanta_sum;
};

struct DeviceMacFaceVelocityFields3D {
  const double* old_face_velocities;
  const double* new_face_velocities;
};

struct DeviceMacParticleGatherResult3D {
  Vec3 new_pic_velocity;
  Vec3 grid_velocity_increment;
  Mat3 fitted_velocity_slopes;
  Mat3 velocity_gradient;
};

enum class MacTransferArithmeticError3D : int {
  kNone,
  kOutOfRange,
};

struct MacTransferKernelReport3D {
  int compiled_architecture;
  int arithmetic_error;  // TransferError3D written atomically by failing lanes.
};

// Launches deposit on completely resolved stencils or clears all destination
// accumulators on device for an empty batch. State and nonempty-batch face
// accumulators start at zero. Returns the immediate CUDA launch status.
int LaunchDeposit3D(
    const DeviceMacDepositParticle3D* particles,
    std::size_t particle_count,
    DeviceMacDepositQuantization3D scales,
    DeviceMacFaceQuantaAccumulator3D* face_accumulators,
    std::size_t face_count,
    MacTransferKernelReport3D* kernel_report,
    void* stream
);

// Launches gather for a positive particle count on complete, resolved stencils.
// State starts at zero. Buffers for both launches outlive stream completion.
// Returns the immediate CUDA launch status.
int LaunchGather3D(
    const DeviceMacParticleStencil3D* particle_stencils,
    std::size_t particle_count,
    DeviceMacFaceVelocityFields3D face_fields,
    double cell_width,
    DeviceMacParticleGatherResult3D* particle_results,
    MacTransferKernelReport3D* kernel_report,
    void* stream
);

}  // namespace fluidity::cuda_internal
