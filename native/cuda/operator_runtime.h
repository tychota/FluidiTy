#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/host/cuda_measurement.h"
#include "fluidity/host/views.h"

namespace fluidity::cuda_internal {

inline absl::Status CheckCuda(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) {
    return absl::OkStatus();
  }

  const std::string message = std::string(operation) + ": " + cudaGetErrorName(error);
  switch (error) {
    case cudaErrorNoDevice:
    case cudaErrorInvalidDevice:
    case cudaErrorDevicesUnavailable:
    case cudaErrorInsufficientDriver:
    case cudaErrorInitializationError:
    case cudaErrorSystemDriverMismatch:
    case cudaErrorNoKernelImageForDevice:
    case cudaErrorInvalidDeviceFunction:
    case cudaErrorUnsupportedPtxVersion:
      return absl::FailedPreconditionError(message);
    case cudaErrorMemoryAllocation:
      return absl::ResourceExhaustedError(message);
    default:
      return absl::InternalError(message);
  }
}

// Owns one call's device buffers, stream and events. Callers keep host upload
// storage alive through destruction and combine cleanup_status with the execution
// status before publishing any staged data.
class CudaOperatorRuntime {
 public:
  explicit CudaOperatorRuntime(absl::Status& cleanup_status) : cleanup_status_(cleanup_status) {
  }
  CudaOperatorRuntime(const CudaOperatorRuntime&) = delete;
  CudaOperatorRuntime& operator=(const CudaOperatorRuntime&) = delete;

  ~CudaOperatorRuntime() noexcept {
    // Early errors can occur while uploads or a kernel are still in flight.
    if (stream_ != nullptr) {
      cleanup_status_.Update(CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize (cleanup)"));
    }
    if (kernel_end_ != nullptr) {
      cleanup_status_.Update(CheckCuda(cudaEventDestroy(kernel_end_), "cudaEventDestroy (end)"));
    }
    if (kernel_start_ != nullptr) {
      cleanup_status_.Update(CheckCuda(cudaEventDestroy(kernel_start_), "cudaEventDestroy (start)"));
    }
    if (stream_ != nullptr) {
      cleanup_status_.Update(CheckCuda(cudaStreamDestroy(stream_), "cudaStreamDestroy"));
    }
    for (std::size_t remaining_buffers = buffer_count_; remaining_buffers > 0; --remaining_buffers) {
      cleanup_status_.Update(CheckCuda(cudaFree(buffers_[remaining_buffers - 1]), "cudaFree"));
    }
    if (previous_device_ >= 0) {
      cleanup_status_.Update(CheckCuda(cudaSetDevice(previous_device_), "cudaSetDevice (restore)"));
    }
  }

  absl::Status Initialize() {
    constexpr int device_ordinal = 0;
    int device_count = 0;
    if (auto status = CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"); !status.ok()) {
      return status;
    }
    if (device_count == 0) {
      return absl::FailedPreconditionError("no CUDA device is available");
    }
    int current_device = -1;
    if (auto status = CheckCuda(cudaGetDevice(&current_device), "cudaGetDevice"); !status.ok()) {
      return status;
    }
    if (current_device != device_ordinal) {
      if (auto status = CheckCuda(cudaSetDevice(device_ordinal), "cudaSetDevice"); !status.ok()) {
        return status;
      }
      previous_device_ = current_device;
    }

    cudaDeviceProp properties{};
    if (auto status = CheckCuda(cudaGetDeviceProperties(&properties, device_ordinal), "cudaGetDeviceProperties");
        !status.ok()) {
      return status;
    }
    device_architecture_ = properties.major * 100 + properties.minor * 10;
    if (auto status =
            CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        !status.ok()) {
      return status;
    }
    if (auto status = CheckCuda(cudaEventCreate(&kernel_start_), "cudaEventCreate (start)"); !status.ok()) {
      return status;
    }
    return CheckCuda(cudaEventCreate(&kernel_end_), "cudaEventCreate (end)");
  }

  cudaStream_t stream() const {
    return stream_;
  }

  // Zero-element buffers have a null device pointer and issue no runtime calls.
  template <typename T>
  absl::Status Allocate(std::size_t element_count, T*& device_buffer) {
    device_buffer = nullptr;
    if (element_count == 0) {
      return absl::OkStatus();
    }
    if (buffer_count_ == buffers_.size()) {
      return absl::ResourceExhaustedError("CUDA operator buffer capacity exceeded");
    }
    const auto byte_count = CheckedExtentProduct(element_count, sizeof(T));
    if (!byte_count.ok()) {
      return byte_count.status();
    }
    void* allocation = nullptr;
    const auto status = CheckCuda(cudaMalloc(&allocation, *byte_count), "cudaMalloc");
    if (!status.ok()) {
      return status;
    }
    buffers_[buffer_count_++] = allocation;
    device_buffer = static_cast<T*>(allocation);
    return absl::OkStatus();
  }

  template <typename T>
  absl::Status Upload(std::span<const T> host_values, T*& device_buffer) {
    const auto status = Allocate(host_values.size(), device_buffer);
    if (!status.ok() || host_values.empty()) {
      return status;
    }
    return CheckCuda(
        cudaMemcpyAsync(device_buffer, host_values.data(), host_values.size_bytes(), cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync (host to device)"
    );
  }

  // The supplied element_count must match the allocation.
  template <typename T>
  absl::Status AllocateZeroed(std::size_t element_count, T*& device_buffer) {
    const auto status = Allocate(element_count, device_buffer);
    if (!status.ok() || element_count == 0) {
      return status;
    }
    return CheckCuda(cudaMemsetAsync(device_buffer, 0, element_count * sizeof(T), stream_), "cudaMemsetAsync");
  }

  absl::Status RecordKernelStart() {
    return CheckCuda(cudaEventRecord(kernel_start_, stream_), "cudaEventRecord (start)");
  }

  // Completes all work on the stream before any result is read by the host.
  absl::Status SynchronizeAndMeasureKernel() {
    if (auto status = CheckCuda(cudaEventRecord(kernel_end_, stream_), "cudaEventRecord (end)"); !status.ok()) {
      return status;
    }
    if (auto status = CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); !status.ok()) {
      return status;
    }
    return CheckCuda(cudaEventElapsedTime(&kernel_ms_, kernel_start_, kernel_end_), "cudaEventElapsedTime");
  }

  // Call only after SynchronizeAndMeasureKernel has synchronized the stream.
  template <typename T>
  absl::Status Download(const T* device_buffer, std::span<T> host_values) {
    if (host_values.empty()) {
      return absl::OkStatus();
    }
    return CheckCuda(
        cudaMemcpy(host_values.data(), device_buffer, host_values.size_bytes(), cudaMemcpyDeviceToHost),
        "cudaMemcpy (device to host)"
    );
  }

  // The marker must be written by the kernel image that just executed.
  absl::StatusOr<CudaOperatorMeasurement> ValidateKernelMeasurement(int compiled_architecture) const {
    if (compiled_architecture != device_architecture_) {
      return absl::FailedPreconditionError("CUDA operator image architecture differs from device architecture");
    }
    if (!std::isfinite(kernel_ms_) || kernel_ms_ < 0.0F) {
      return absl::InternalError("CUDA operator event timing is invalid");
    }
    return CudaOperatorMeasurement{compiled_architecture, kernel_ms_};
  }

 private:
  absl::Status& cleanup_status_;
  int previous_device_ = -1;
  int device_architecture_ = 0;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t kernel_start_ = nullptr;
  cudaEvent_t kernel_end_ = nullptr;
  std::array<void*, 16> buffers_{};
  std::size_t buffer_count_ = 0;
  float kernel_ms_ = -1.0F;
};

}  // namespace fluidity::cuda_internal
