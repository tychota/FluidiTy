#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <concepts>
#include <cstddef>
#include <new>
#include <ratio>
#include <span>
#include <type_traits>

#include "absl/status/status.h"
#include "operator_runtime.h"

namespace fluidity::graph_spike {

using cuda_internal::CheckCuda;

// Owns one nonblocking test stream. Declare it after the graph and buffers so
// teardown also fences the stream before those resources are released on early exit.
class CudaStream {
 public:
  CudaStream() = default;
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  ~CudaStream() {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

  // Creates the stream outside capture. Call once per instance.
  absl::Status Initialize() {
    return CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  }

  // Returns the owned stream after successful initialization.
  cudaStream_t handle() const {
    return stream_;
  }

  // Waits for queued work and reports asynchronous runtime errors. Call before
  // reading download staging or reusing upload staging.
  absl::Status Synchronize() const {
    return CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }

 private:
  cudaStream_t stream_ = nullptr;
};

// Owns a fixed-size pinned staging buffer and a matching device allocation.
// Allocate before capture and keep both addresses alive through graph destruction.
template <typename BufferElement, std::size_t ElementCount>
  requires(std::is_trivial_v<BufferElement> && ElementCount > 0)
class PinnedDeviceBuffer {
 public:
  PinnedDeviceBuffer() = default;
  PinnedDeviceBuffer(const PinnedDeviceBuffer&) = delete;
  PinnedDeviceBuffer& operator=(const PinnedDeviceBuffer&) = delete;

  ~PinnedDeviceBuffer() {
    if (device_ != nullptr) {
      cudaFree(device_);
    }
    if (host_ != nullptr) {
      cudaFreeHost(host_);
    }
  }

  // Allocates both buffers and value-initializes host staging. On error the
  // buffer cannot be used. Call once per instance.
  absl::Status Allocate() {
    void* host_allocation = nullptr;
    if (auto status = CheckCuda(cudaMallocHost(&host_allocation, kByteCount), "cudaMallocHost"); !status.ok()) {
      return status;
    }
    host_ = new (host_allocation) BufferElement[ElementCount]{};

    void* device_allocation = nullptr;
    const auto status = CheckCuda(cudaMalloc(&device_allocation, kByteCount), "cudaMalloc");
    if (status.ok()) {
      device_ = static_cast<BufferElement*>(device_allocation);
    }
    return status;
  }

  // Returns host staging after allocation. Access only with stream work fenced.
  std::span<BufferElement, ElementCount> host_staging() const {
    return std::span<BufferElement, ElementCount>(host_, ElementCount);
  }

  // Returns the device address after allocation for use in kernel launches.
  BufferElement* device_data() const {
    return device_;
  }

  // Enqueues a full upload from pinned host staging.
  absl::Status Upload(cudaStream_t stream) const {
    return CheckCuda(
        cudaMemcpyAsync(device_, host_, kByteCount, cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync (host to device)"
    );
  }

  // Enqueues a full download into pinned host staging.
  absl::Status Download(cudaStream_t stream) const {
    return CheckCuda(
        cudaMemcpyAsync(host_, device_, kByteCount, cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync (device to host)"
    );
  }

  // Enqueues a zero byte fill of the entire device buffer.
  absl::Status ZeroDevice(cudaStream_t stream) const {
    return CheckCuda(cudaMemsetAsync(device_, 0, kByteCount, stream), "cudaMemsetAsync");
  }

 private:
  static constexpr std::size_t kByteCount = sizeof(BufferElement) * ElementCount;
  BufferElement* host_ = nullptr;
  BufferElement* device_ = nullptr;
};

// Owns a captured graph and its executable. Callers fence all queued work
// before destruction and keep captured buffers alive until after destruction.
class CudaGraphReplay {
 public:
  CudaGraphReplay() = default;
  CudaGraphReplay(const CudaGraphReplay&) = delete;
  CudaGraphReplay& operator=(const CudaGraphReplay&) = delete;

  ~CudaGraphReplay() {
    if (executable_ != nullptr) {
      cudaGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
    }
  }

  // Captures and instantiates enqueue_work on a fenced nonblocking stream.
  // The callback queues work on that stream, returns a status, and does not
  // throw. Failed capture leaves the owner empty and can be retried. An
  // already instantiated owner returns FailedPrecondition.
  template <typename EnqueueWork>
    requires std::same_as<std::invoke_result_t<EnqueueWork&>, absl::Status>
  absl::Status Capture(cudaStream_t stream, EnqueueWork enqueue_work) {
    if (executable_ != nullptr) {
      return absl::FailedPreconditionError("CUDA Graph executable already exists");
    }
    const auto capture_start = std::chrono::steady_clock::now();
    if (auto status = CheckCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
        !status.ok()) {
      return status;
    }

    // End capture even when the callback rejects the sequence. CUDA can
    // return a graph containing the operations that the callback queued.
    absl::Status status = enqueue_work();
    cudaGraph_t captured_graph = nullptr;
    status.Update(CheckCuda(cudaStreamEndCapture(stream, &captured_graph), "cudaStreamEndCapture"));

    cudaGraphExec_t captured_executable = nullptr;
    if (status.ok()) {
      if (captured_graph == nullptr) {
        return absl::InternalError("CUDA stream capture returned no graph");
      }
      status = CheckCuda(cudaGraphInstantiate(&captured_executable, captured_graph), "cudaGraphInstantiate");
    }
    if (!status.ok()) {
      if (captured_executable != nullptr) {
        status.Update(CheckCuda(cudaGraphExecDestroy(captured_executable), "cudaGraphExecDestroy"));
      }
      if (captured_graph != nullptr) {
        status.Update(CheckCuda(cudaGraphDestroy(captured_graph), "cudaGraphDestroy"));
      }
      return status;
    }

    const auto capture_end = std::chrono::steady_clock::now();
    capture_us_ = std::chrono::duration<double, std::micro>(capture_end - capture_start).count();
    graph_ = captured_graph;
    executable_ = captured_executable;
    return absl::OkStatus();
  }

  // Queues the executable or returns FailedPrecondition if none is available.
  // The caller is responsible for fencing before accessing captured host buffers.
  absl::Status Launch(cudaStream_t stream) const {
    if (executable_ == nullptr) {
      return absl::FailedPreconditionError("no CUDA Graph executable is available");
    }
    return CheckCuda(cudaGraphLaunch(executable_, stream), "cudaGraphLaunch");
  }

  // Returns host capture and instantiation duration in microseconds, or zero
  // before the first successful capture.
  double capture_us() const {
    return capture_us_;
  }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  double capture_us_ = 0.0;
};

}  // namespace fluidity::graph_spike
