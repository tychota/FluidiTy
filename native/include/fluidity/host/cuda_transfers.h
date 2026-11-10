#pragma once

#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "fluidity/host/cuda_measurement.h"
#include "fluidity/host/transfers.h"

namespace fluidity {

struct CudaDepositScales {
  double mass;  // Kilograms per quantum.
  double pure_momentum;  // Kilogram meters per second per quantum.
  double stress_force;  // Newtons per quantum.
};

struct CudaDepositViews {
  std::span<std::int64_t> mass_quanta;  // Quantized mass sums at each face.
  std::span<std::int64_t> pure_momentum_quanta;  // Quantized pure momentum sums at each face.
  std::span<std::int64_t> stress_force_quanta;  // Quantized stress force sums at each face.
};

// Deposits mass (kg), pure normal momentum (kg*m/s) from centered APIC, and
// stress force (N) at unit depth on CUDA device 0. Reference volumes are finite nonnegative m^3, and stresses
// are finite row-major Kirchhoff stresses in Pa, both in particle order. Every
// particle requires all nine nodes on each lattice, including zero-weight nodes.
// Scales are positive normal powers of two in the corresponding channel units.
// Divide each contribution by its scale and round to nearest-even before integer
// accumulation. Outputs have exactly FaceCount entries in FaceGeometry order.
// Outputs and measurement must be disjoint from each other and all inputs. An empty
// batch still launches and sets every face output to zero. All outputs change only
// on success.
// Returns kInvalidArgument for invalid inputs, kOutOfRange for nonfinite arithmetic
// or scaled magnitude above 2^52, and kResourceExhausted beyond 1024 particles or
// 4096 faces or on host or device allocation failure. An unusable device or image
// returns kFailedPrecondition. Other CUDA errors return kInternal. Propagates grid
// errors.
absl::Status CudaP2G(
    const Grid2D& mac_grid,
    ParticleView2D particle_inputs,
    std::span<const double> particle_reference_volumes,
    std::span<const Mat2> particle_kirchhoff_stresses,
    CudaDepositScales quantization_scales,
    CudaDepositViews deposit_outputs,
    CudaOperatorMeasurement& kernel_measurement
);

struct CudaGatherViews {
  std::span<Vec2> pic_velocities;  // PIC velocity in m/s.
  std::span<Vec2> grid_velocity_increments;  // New PIC minus old PIC in m/s.
  std::span<Mat2> velocity_fit_slopes;  // Centered affine fit C in 1/s.
  std::span<Mat2> velocity_gradients;  // True interpolation gradient L in 1/s.
};

// Gathers particle-in-cell (PIC) velocity and the new-minus-old PIC increment (m/s), the centered
// affine fit C and the true velocity gradient L (1/s) at world positions in m.
// C and L have one Cartesian row per velocity component. Each particle needs all
// nine nodes on both staggered lattices, including zero-weight nodes. Both finite
// face fields have exactly FaceCount entries in FaceGeometry order and may alias.
// Each output has one entry per particle. Outputs and kernel measurement must
// be disjoint from all inputs and each other and only change on success. An empty
// batch validates inputs without calling CUDA or changing the measurement.
// Nonempty batches run on CUDA device 0. Invalid inputs return kInvalidArgument,
// nonfinite arithmetic returns kOutOfRange, and a numerically singular moment or
// unusable CUDA device or image returns kFailedPrecondition. Capacity is 1024
// particles and 4096 faces. Exceeding either bound or failing to allocate host or
// device storage returns kResourceExhausted. Other CUDA errors return kInternal.
// Propagates grid errors.
absl::Status CudaG2P(
    const Grid2D& mac_grid,
    std::span<const Vec2> particle_world_positions,
    std::span<const double> old_face_velocities,
    std::span<const double> new_face_velocities,
    CudaGatherViews gather_outputs,
    CudaOperatorMeasurement& kernel_measurement
);

}  // namespace fluidity
