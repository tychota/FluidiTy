#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "fluidity/host/cuda_probe.h"
#include "probe_bridge.h"

namespace fluidity {
namespace {

absl::Status CheckCuda(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) {
    return absl::OkStatus();
  }

  const std::string message = std::string(operation) + ": " + cudaGetErrorName(error);
  switch (error) {
    case cudaErrorInvalidDevice:
      return absl::InvalidArgumentError(message);
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorInitializationError:
    case cudaErrorSystemDriverMismatch:
      return absl::UnavailableError(message);
    case cudaErrorMemoryAllocation:
      return absl::ResourceExhaustedError(message);
    case cudaErrorNoKernelImageForDevice:
    case cudaErrorInvalidDeviceFunction:
    case cudaErrorUnsupportedPtxVersion:
      return absl::FailedPreconditionError(message);
    default:
      return absl::InternalError(message);
  }
}

// Keep cleanup local to each execution and capture errors before publishing the report.
class ProbeResources {
 public:
  explicit ProbeResources(absl::Status& cleanup_status) : cleanup_status_(cleanup_status) {
  }
  ProbeResources(const ProbeResources&) = delete;
  ProbeResources& operator=(const ProbeResources&) = delete;

  ~ProbeResources() noexcept {
    if (stream != nullptr) {
      cleanup_status_.Update(CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy"));
    }
    if (device_output != nullptr) {
      cleanup_status_.Update(CheckCuda(cudaFree(device_output), "cudaFree"));
    }
    if (previous_device >= 0) {
      cleanup_status_.Update(CheckCuda(cudaSetDevice(previous_device), "cudaSetDevice (restore)"));
    }
  }

  int previous_device = -1;
  cudaStream_t stream = nullptr;
  void* device_output = nullptr;

 private:
  absl::Status& cleanup_status_;
};

// Stage runtime codes and copied values until the architecture and cleanup checks succeed.
absl::Status CollectProbe(int device_index, ProbeReport& staged_report, absl::Status& cleanup_status) {
  int device_count = 0;
  if (auto status = CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"); !status.ok()) {
    return status;
  }
  if (device_count == 0) {
    return absl::UnavailableError("No CUDA devices available");
  }
  if (device_index >= device_count) {
    return absl::InvalidArgumentError("CUDA probe device ordinal is outside the device count");
  }

  int current_device = 0;
  if (auto status = CheckCuda(cudaGetDevice(&current_device), "cudaGetDevice"); !status.ok()) {
    return status;
  }
  ProbeResources resources(cleanup_status);
  if (current_device != device_index) {
    if (auto status = CheckCuda(cudaSetDevice(device_index), "cudaSetDevice"); !status.ok()) {
      return status;
    }
    resources.previous_device = current_device;
  }

  cudaDeviceProp properties{};
  if (auto status = CheckCuda(cudaGetDeviceProperties(&properties, device_index), "cudaGetDeviceProperties");
      !status.ok()) {
    return status;
  }
  if (auto status = CheckCuda(cudaDriverGetVersion(&staged_report.driver_version), "cudaDriverGetVersion");
      !status.ok()) {
    return status;
  }
  if (auto status = CheckCuda(cudaRuntimeGetVersion(&staged_report.runtime_version), "cudaRuntimeGetVersion");
      !status.ok()) {
    return status;
  }

  // Reserve a sentinel just beyond the samples and one separate slot for the device image's architecture.
  constexpr std::size_t sample_count = ProbeReport::kSampleCount;
  std::array<Vec2, sample_count + 2> copied_probe_payload{};
  constexpr std::size_t buffer_bytes = sizeof(Vec2) * copied_probe_payload.size();
  if (auto status =
          CheckCuda(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
      !status.ok()) {
    return status;
  }
  if (auto status = CheckCuda(cudaMalloc(&resources.device_output, buffer_bytes), "cudaMalloc"); !status.ok()) {
    return status;
  }
  if (auto status =
          CheckCuda(cudaMemsetAsync(resources.device_output, 0, buffer_bytes, resources.stream), "cudaMemsetAsync");
      !status.ok()) {
    return status;
  }

  staged_report.launch_result =
      LaunchProbeKernel(static_cast<Vec2*>(resources.device_output), sample_count, resources.stream);
  if (auto status = CheckCuda(static_cast<cudaError_t>(staged_report.launch_result), "LaunchProbeKernel");
      !status.ok()) {
    return status;
  }
  staged_report.synchronize_result = cudaStreamSynchronize(resources.stream);
  if (auto status = CheckCuda(static_cast<cudaError_t>(staged_report.synchronize_result), "cudaStreamSynchronize");
      !status.ok()) {
    return status;
  }
  staged_report.copy_result =
      cudaMemcpy(copied_probe_payload.data(), resources.device_output, buffer_bytes, cudaMemcpyDeviceToHost);
  if (auto status = CheckCuda(static_cast<cudaError_t>(staged_report.copy_result), "cudaMemcpy"); !status.ok()) {
    return status;
  }

  // Compare the marker written by the executing image to the queried capability encoding.
  const int device_architecture = properties.major * 100 + properties.minor * 10;
  const double compiled_architecture = copied_probe_payload[sample_count + 1].x;
  if (compiled_architecture != device_architecture) {
    return absl::FailedPreconditionError(
        "CUDA probe compiled architecture " + std::to_string(compiled_architecture)
        + " differs from device architecture " + std::to_string(device_architecture)
    );
  }
  staged_report.device_name = properties.name;
  staged_report.compute_major = properties.major;
  staged_report.compute_minor = properties.minor;
  staged_report.compiled_architecture = static_cast<int>(compiled_architecture);
  staged_report.sample_count = sample_count;
  std::copy_n(
      copied_probe_payload.begin(),
      staged_report.samples_with_sentinel.size(),
      staged_report.samples_with_sentinel.begin()
  );
  return absl::OkStatus();
}

}  // namespace

absl::Status RunCudaProbe(int device_index, ProbeReport* report) {
  if (report == nullptr) {
    return absl::InvalidArgumentError("CUDA probe requires a report");
  }
  if (device_index < 0) {
    return absl::InvalidArgumentError("CUDA probe device ordinal must be nonnegative");
  }

  ProbeReport staged_report;
  absl::Status cleanup_status;
  absl::Status status = CollectProbe(device_index, staged_report, cleanup_status);
  status.Update(cleanup_status);
  if (!status.ok()) {
    return status;
  }
  *report = std::move(staged_report);
  return absl::OkStatus();
}

}  // namespace fluidity
