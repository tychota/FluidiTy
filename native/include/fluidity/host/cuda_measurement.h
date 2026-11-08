#pragma once

namespace fluidity {

struct CudaOperatorMeasurement {
  int compiled_architecture;  // CUDA image architecture in __CUDA_ARCH__ encoding.
  float kernel_ms;  // Milliseconds between CUDA events bracketing the operator kernel.
};

}  // namespace fluidity
