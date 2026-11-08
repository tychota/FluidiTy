#pragma once

#include <span>

#include "absl/status/status.h"
#include "fluidity/host/cuda_measurement.h"
#include "fluidity/host/transfers_3d.h"

namespace fluidity {

// Executes DepositMacQuanta3D's complete-support liquid deposit on CUDA device
// 0, with the same extents, units, scale and capacity bounds, and input rules.
// The measurement must be disjoint from both spans. An empty batch clears a
// nonempty destination on the device. If both spans are empty, no launch is
// needed and measurement is preserved. Otherwise success publishes the faces
// and CUDA event measurement together. The CUDA event interval covers device work. Host validation, device
// arithmetic and CUDA runtime failures preserve both outputs.
absl::Status CudaDepositMacQuanta3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    MacDepositQuantization3D quantization,
    std::span<MacFaceDepositQuanta3D> face_deposits,
    CudaOperatorMeasurement& measurement
);

// Executes GatherMac3D on CUDA device 0 with the same complete-support input,
// extent, unit and ordering rules. The measurement must be disjoint from all
// spans. Empty batches still validate the views and fields, and preserve the
// measurement. For a nonempty batch, success publishes the particle results
// and CUDA event measurement together. Host validation, device arithmetic
// and CUDA runtime failures preserve both outputs.
absl::Status CudaGatherMac3D(
    const SparseMacGrid3D& mac_grid,
    std::span<const MacTransferParticle3D> particles,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    std::span<MacParticleGatherResult3D> particle_gather_results,
    CudaOperatorMeasurement& measurement
);

}  // namespace fluidity
