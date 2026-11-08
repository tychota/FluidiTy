#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstring>

#include "fluidity/host/cuda_probe.h"
#include "gtest/gtest.h"
#include "probe_bridge.h"

namespace {

enum class Fault {
  kNone,
  kNoDevice,
  kInsufficientDriver,
  kAllocation,
  kLaunch,
  kSynchronize,
  kCopy,
  kArchitecture,
  kFree,
};

// These link-time doubles supply the CUDA calls to the actual host wrapper without a driver.
Fault fault = Fault::kNone;
fluidity::Vec2 device_output[19]{};
int current_device = 0;
bool allocation_live = false;
bool stream_live = false;
int stream_token = 0;

}  // namespace

cudaError_t CUDARTAPI cudaGetDeviceCount(int* count) {
  if (fault == Fault::kInsufficientDriver) {
    return cudaErrorInsufficientDriver;
  }
  *count = fault == Fault::kNoDevice ? 0 : 2;
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDevice(int* device) {
  *device = current_device;
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetDevice(int device) {
  current_device = device;
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp* properties, int) {
  *properties = {};
  constexpr char name[] = "CUDA API double";
  std::memcpy(properties->name, name, sizeof(name));
  properties->major = 12;
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int* version) {
  *version = 13030;
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* version) {
  *version = 13030;
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int) {
  stream_live = true;
  *stream = reinterpret_cast<cudaStream_t>(&stream_token);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t) {
  stream_live = false;
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMalloc(void** output, std::size_t bytes) {
  if (fault == Fault::kAllocation || bytes > sizeof(device_output)) {
    return cudaErrorMemoryAllocation;
  }
  allocation_live = true;
  *output = device_output;
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaFree(void*) {
  allocation_live = false;
  return fault == Fault::kFree ? cudaErrorUnknown : cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemsetAsync(void* output, int value, std::size_t bytes, cudaStream_t) {
  std::memset(output, value, bytes);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t) {
  return fault == Fault::kSynchronize ? cudaErrorIllegalAddress : cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpy(void* output, const void* input, std::size_t bytes, cudaMemcpyKind) {
  if (fault == Fault::kCopy) {
    return cudaErrorUnknown;
  }
  std::memcpy(output, input, bytes);
  return cudaSuccess;
}

const char* CUDARTAPI cudaGetErrorName(cudaError_t) {
  return "injected CUDA error";
}

namespace fluidity {

int LaunchProbeKernel(Vec2* output, std::size_t count, void*) {
  if (fault == Fault::kLaunch) {
    return cudaErrorLaunchFailure;
  }
  output[count + 1].x = fault == Fault::kArchitecture ? 900.0 : 1200.0;
  return cudaSuccess;
}

namespace {

class CudaProbeErrors : public testing::Test {
 protected:
  void SetUp() override {
    fault = Fault::kNone;
    current_device = 0;
    allocation_live = false;
    stream_live = false;
  }

  // Use nondefault values to detect both whole-report replacement and individual field writes.
  void ExpectFailure(int device_index, absl::StatusCode code) {
    ProbeReport report{
        .device_name = "previous report",
        .compute_major = 8,
        .compute_minor = 9,
        .driver_version = 111,
        .runtime_version = 222,
        .compiled_architecture = 890,
        .element_count = 3,
        .launch_result = 10,
        .copy_result = 20,
        .synchronize_result = 30,
    };
    report.values.fill(Vec2{64.0, -128.0});
    const absl::Status status = RunCudaProbe(device_index, &report);
    EXPECT_EQ(status.code(), code) << status;
    EXPECT_EQ(report.device_name, "previous report");
    EXPECT_EQ(report.compute_major, 8);
    EXPECT_EQ(report.compute_minor, 9);
    EXPECT_EQ(report.driver_version, 111);
    EXPECT_EQ(report.runtime_version, 222);
    EXPECT_EQ(report.compiled_architecture, 890);
    EXPECT_EQ(report.element_count, 3);
    EXPECT_EQ(report.launch_result, 10);
    EXPECT_EQ(report.copy_result, 20);
    EXPECT_EQ(report.synchronize_result, 30);
    for (const Vec2 value : report.values) {
      EXPECT_EQ(value.x, 64.0);
      EXPECT_EQ(value.y, -128.0);
    }
    EXPECT_EQ(current_device, 0);
    EXPECT_FALSE(allocation_live);
    EXPECT_FALSE(stream_live);
  }
};

TEST_F(CudaProbeErrors, RejectsNullReport) {
  const absl::Status status = RunCudaProbe(0, nullptr);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
}

TEST_F(CudaProbeErrors, RejectsNegativeOrdinal) {
  ExpectFailure(-1, absl::StatusCode::kInvalidArgument);
}

TEST_F(CudaProbeErrors, RejectsOrdinalAtDeviceCount) {
  ExpectFailure(2, absl::StatusCode::kInvalidArgument);
}

TEST_F(CudaProbeErrors, RejectsNoDevice) {
  fault = Fault::kNoDevice;
  ExpectFailure(0, absl::StatusCode::kUnavailable);
}

TEST_F(CudaProbeErrors, RejectsInsufficientDriver) {
  fault = Fault::kInsufficientDriver;
  ExpectFailure(0, absl::StatusCode::kUnavailable);
}

TEST_F(CudaProbeErrors, AllocationFailurePreservesReport) {
  fault = Fault::kAllocation;
  ExpectFailure(1, absl::StatusCode::kResourceExhausted);
}

TEST_F(CudaProbeErrors, ImmediateLaunchFailurePreservesReport) {
  fault = Fault::kLaunch;
  ExpectFailure(1, absl::StatusCode::kInternal);
}

TEST_F(CudaProbeErrors, SynchronizationFailurePreservesReport) {
  fault = Fault::kSynchronize;
  ExpectFailure(1, absl::StatusCode::kInternal);
}

TEST_F(CudaProbeErrors, CopyFailurePreservesReport) {
  fault = Fault::kCopy;
  ExpectFailure(1, absl::StatusCode::kInternal);
}

TEST_F(CudaProbeErrors, ArchitectureMismatchPreservesReport) {
  fault = Fault::kArchitecture;
  ExpectFailure(1, absl::StatusCode::kFailedPrecondition);
}

TEST_F(CudaProbeErrors, CleanupFailurePreservesReport) {
  fault = Fault::kFree;
  ExpectFailure(1, absl::StatusCode::kInternal);
}

}  // namespace
}  // namespace fluidity
