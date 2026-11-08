#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "fluidity/base/types.h"

namespace fluidity {

struct ProbeReport {
  static constexpr std::size_t kElementCount = 17;

  std::string device_name;
  // Compute capability reported by the device, as separate major and minor numbers.
  int compute_major = 0;
  int compute_minor = 0;
  // CUDA API version encoding, for example 13030 means 13.3.
  int driver_version = 0;
  int runtime_version = 0;
  // The value of __CUDA_ARCH__ in the executed device image.
  int compiled_architecture = 0;
  std::size_t element_count = 0;
  // Runtime return codes. Zero means success and -1 means the operation has yet to run.
  int launch_result = -1;
  int copy_result = -1;
  int synchronize_result = -1;
  // The last slot is the zero sentinel after the kElementCount results.
  std::array<Vec2, kElementCount + 1> values{};
};

// Runs a fixed-size Vec2 arithmetic probe on a CUDA device ordinal. Returns InvalidArgument for an invalid
// ordinal or null report, Unavailable when CUDA cannot initialize, ResourceExhausted on allocation failure,
// FailedPrecondition when the device image and device architecture differ, and Internal for other CUDA failures.
// On success, publishes the samples in thread-index order after synchronizing, copying, and cleaning up.
// The caller's previously selected CUDA device is restored on success. On failure, the report is unchanged.
absl::Status RunCudaProbe(int device_index, ProbeReport* report);

}  // namespace fluidity
