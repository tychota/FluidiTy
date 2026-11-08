#pragma once

#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "fluidity/host/cuda_measurement.h"
#include "fluidity/host/transfers.h"

namespace fluidity {

struct CudaDepositScales {
  double mass;
  double momentum;
  double force;
};

struct CudaDepositViews {
  std::span<std::int64_t> mass;  // Quantized mass sums at each face.
  std::span<std::int64_t> momentum;  // Quantized pure momentum sums at each face.
  std::span<std::int64_t> force;  // Quantized stress force sums at each face.
};

// Deposits mass (kg), centered APIC momentum (kg*m/s), and stress force (N) at unit
// depth on CUDA device 0. Reference volumes are finite nonnegative m^3, and stresses
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
    const Grid2D& grid,
    ParticleView2D particles,
    std::span<const double> reference_volumes,
    std::span<const Mat2> stresses,
    CudaDepositScales scales,
    CudaDepositViews outputs,
    CudaOperatorMeasurement& measurement
);

}  // namespace fluidity
