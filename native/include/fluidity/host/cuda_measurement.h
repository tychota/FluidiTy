#pragma once

namespace fluidity {

struct CudaOperatorMeasurement {
  int compiled_architecture;  // CUDA image architecture in __CUDA_ARCH__ encoding.
  // Elapsed milliseconds between CUDA events on the operator stream.
  // The operator API defines which work the interval covers.
  float kernel_ms;
};

}  // namespace fluidity
