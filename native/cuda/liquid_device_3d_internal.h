#pragma once

#include <cuda_runtime_api.h>

#include "absl/status/statusor.h"

namespace fluidity::cuda_internal {

// Translate setup failures before an arena or poison ledger exists. Runtime
// operations on an existing arena still use its sticky failure handling.
inline absl::Status LiquidCudaDeviceSetupStatus3D(cudaError_t result) {
  switch (result) {
    case cudaSuccess:
      return absl::OkStatus();
    case cudaErrorInvalidDevice:
      return absl::InvalidArgumentError("");
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorInitializationError:
    case cudaErrorSystemDriverMismatch:
      return absl::UnavailableError("");
    case cudaErrorMemoryAllocation:
      return absl::ResourceExhaustedError("");
    default:
      return absl::InternalError("");
  }
}

// Validate an ordinal and the qualified capability without changing the current
// device or launching work. A machine with no usable devices is Unavailable.
inline absl::StatusOr<cudaDeviceProp> QueryLiquidCudaDevice3D(int device_ordinal) {
  if (device_ordinal < 0) {
    return absl::InvalidArgumentError("");
  }
  int device_count = 0;
  auto status = LiquidCudaDeviceSetupStatus3D(cudaGetDeviceCount(&device_count));
  if (!status.ok()) {
    return status;
  }
  if (device_count == 0) {
    return absl::UnavailableError("");
  }
  if (device_ordinal >= device_count) {
    return absl::InvalidArgumentError("");
  }
  cudaDeviceProp properties{};
  status = LiquidCudaDeviceSetupStatus3D(cudaGetDeviceProperties(&properties, device_ordinal));
  if (!status.ok()) {
    return status;
  }
  if (properties.major != 12 || properties.minor != 0) {
    return absl::FailedPreconditionError("");
  }
  return properties;
}

}  // namespace fluidity::cuda_internal
