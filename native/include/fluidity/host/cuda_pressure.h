#pragma once

#include "absl/status/status.h"
#include "fluidity/host/cuda_measurement.h"
#include "fluidity/host/pressure.h"

namespace fluidity {

// Assembles and solves the integrated pressure system on CUDA device 0 and
// corrects open internal and Dirichlet faces. Uses the CPU views' units and
// MAC/cell ordering. Supports grids with at most 8 cells and 25 faces.
// For n cells and f faces, velocity has f entries, original_matrix has n*n,
// and all other output spans have n. Read-only spans may alias. Writable spans,
// gauge_count and kernel_measurement must be disjoint from the input view,
// its spans and each other. The first gauge_count slots receive the lowest cell
// index of each compatible unanchored component, in ascending order. The other
// gauge slots keep their original values.
//
// Returns kInvalidArgument for invalid inputs, extents or aliasing,
// kResourceExhausted for capacity or allocation failure, kOutOfRange for
// unrepresentable geometry/arithmetic (including non-normal active conductance
// or compliance), and kFailedPrecondition for an unusable device, incompatible
// component or unsafe SPD solve. Other CUDA runtime errors return kInternal.
// Every output, including the measurement, is preserved on failure.
absl::Status CudaSolvePressure2D(
    const PressureInputsView& pressure_inputs,
    PressureOutputsView pressure_outputs,
    CudaOperatorMeasurement& kernel_measurement
);

}  // namespace fluidity
