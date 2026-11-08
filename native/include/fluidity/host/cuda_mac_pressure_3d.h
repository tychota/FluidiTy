#pragma once

#include "fluidity/host/cuda_measurement.h"
#include "fluidity/host/mac_pressure_3d.h"

namespace fluidity {

struct CudaMacPressureProjection3D {
  MacPressureProjection3D projection;
  // CUDA-event elapsed time on the solver stream, from warm-start evaluation
  // through the final iteration/status synchronization. Includes orchestration
  // gaps and status copies; excludes preparation, uploads and output download.
  CudaOperatorMeasurement measurement;
};

// Project the bounded owned-face system on CUDA device 0. Inputs remain
// unchanged and errors publish no projection. Status and numerical contracts
// match ProjectMacPressure3D; unavailable hardware returns kFailedPrecondition.
absl::StatusOr<CudaMacPressureProjection3D>
CudaProjectMacPressure3D(const MacPressureInputs3D& inputs, MacPressureSolverOptions3D options = {});
}  // namespace fluidity
