#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "../cuda/liquid_arena_3d_internal.h"

namespace fluidity::simulate {
namespace {
int completed_snapshot_count = 0;
bool delay_next_device_to_host_copy = false;

// Only the snapshot queue uses this event-query adapter. Report a synthetic
// failure after the third actual completion, when the third frame has not
// yet reached the cache writer.
cudaError_t QuerySnapshotAndInjectDeviceError(cudaEvent_t snapshot_complete) {
  const auto actual_result = cudaEventQuery(snapshot_complete);
  if (actual_result == cudaSuccess && ++completed_snapshot_count == 3) {
    std::fputs(
        "{\"protocol_version\":1,\"event\":\"fixture_device_fault\",\"completed_snapshot_count\":3,"
        "\"actual_cuda_event_result\":0,\"injected_cuda_error\":\"cudaErrorLaunchFailure\"}\n",
        stderr
    );
    return cudaErrorLaunchFailure;
  }
  return actual_result;
}

void CUDART_CB HoldSnapshotStream(void*) {
  std::this_thread::sleep_for(std::chrono::milliseconds(1250));
}

// The first device-to-host transfer belongs to the initial snapshot. Queue a
// real stream callback before the copy so its elapsed CUDA events include the
// deliberate stall while the solver's independent heartbeat remains active.
cudaError_t CopyWithInitialSnapshotDelay(
    void* destination,
    const void* source,
    std::size_t byte_count,
    cudaMemcpyKind direction,
    cudaStream_t copy_stream
) {
  if (direction == cudaMemcpyDeviceToHost && delay_next_device_to_host_copy) {
    delay_next_device_to_host_copy = false;
    const auto queued_delay = cudaLaunchHostFunc(copy_stream, HoldSnapshotStream, nullptr);
    if (queued_delay != cudaSuccess) {
      return queued_delay;
    }
  }
  return cudaMemcpyAsync(destination, source, byte_count, direction, copy_stream);
}
}  // namespace

void ConfigureSimulateFaultFixture(std::vector<std::string>& arguments) {
  if (arguments.empty()) {
    return;
  }
  if (arguments.front() == "--fixture-device-error") {
    cuda_internal::LiquidArenaRuntimeCalls().event_query = QuerySnapshotAndInjectDeviceError;
  } else if (arguments.front() == "--fixture-delay-copy") {
    delay_next_device_to_host_copy = true;
    cuda_internal::LiquidArenaRuntimeCalls().memcpy_async = CopyWithInitialSnapshotDelay;
  } else {
    return;
  }
  arguments.erase(arguments.begin());
}
}  // namespace fluidity::simulate
