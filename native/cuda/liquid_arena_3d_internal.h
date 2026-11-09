#pragma once
#include <cuda_runtime_api.h>

#include <array>

#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "liquid_arena_3d_bridge.h"
#include "mac_pressure_3d_bridge.h"
namespace fluidity::cuda_internal {
// Thread-local dispatch for setup/submission and final drain. Defaults call the
// real runtime; alternate implementations must preserve the same ownership ABI.
// Tests replace selected calls with real-call-then-error controls, outside attempts.
struct LiquidArenaRuntimeCalls3D {
  decltype(&cudaMemsetAsync) memset_async = cudaMemsetAsync;
  decltype(&cudaMemcpyAsync) memcpy_async = cudaMemcpyAsync;
  decltype(&cudaEventRecord) event_record = cudaEventRecord;
  decltype(&cudaEventSynchronize) event_synchronize = cudaEventSynchronize;
  decltype(&LaunchMacPressure3D) launch_pressure = LaunchMacPressure3D;
  decltype(&SortLiquidBins3D) sort_bins = SortLiquidBins3D;
  decltype(&LaunchLiquidArenaArchitecture3D) launch_architecture = LaunchLiquidArenaArchitecture3D;
  decltype(&cudaStreamEndCapture) end_capture = cudaStreamEndCapture;
  decltype(&cudaGraphLaunch) graph_launch = cudaGraphLaunch;
  decltype(&cudaStreamSynchronize) stream_synchronize = cudaStreamSynchronize;
};
LiquidArenaRuntimeCalls3D& LiquidArenaRuntimeCalls();
struct LiquidArenaLedger3D {
  CudaLiquidArenaMetrics3D metrics;
  std::uint64_t next_generation = 1;
  std::size_t references = 1;
  int device_ordinal = 0;
  bool runtime_failed = false;  // Job-wide poison; only teardown may drain/release afterward.
};
struct LiquidArenaRange3D {
  std::size_t offset = 0;
  std::size_t bytes = 0;
};
// Device control belongs to the immutable version, including valid extents.
struct LiquidArenaControl3D {
  UniformMacGeometry3D geometry;
  LiquidClock3D clock;
  LiquidStateCapacities3D active;
  std::uint64_t random_seed;
  std::uint64_t next_particle_id;
};
struct LiquidArenaStorage3D {
  LiquidArenaStorage3D() = default;
  LiquidArenaLedger3D* ledger = nullptr;
  std::size_t references = 1;
  std::array<std::size_t, 2> leases{};
  std::array<CudaLiquidEpochIdentity3D, 2> identities{};
  std::array<LiquidStateCapacities3D, 2> active{};
  LiquidStateCapacities3D capacities;
  CudaLiquidStateLayout3D state_layout;
  std::array<LiquidArenaRange3D, static_cast<std::size_t>(CudaLiquidRange3D::kCount)> ranges{};
  std::byte* device = nullptr;
  std::byte* pinned = nullptr;
  std::size_t reserved_device_bytes = 0;
  std::size_t reserved_pinned_bytes = 0;
  std::size_t sort_capacity = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t completion = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  LiquidArenaStorage3D* retired_next = nullptr;
  LiquidArenaStorage3D* quarantined_next = nullptr;
  CudaLiquidRangeView3D Range(CudaLiquidRange3D name) const;

 private:
  // Release must establish a successful drain before deleting this owner.
  friend void ReleaseLiquidArenaStorage3D(LiquidArenaStorage3D*) noexcept;
  ~LiquidArenaStorage3D();
};
// Takes one never-reused ticket. Exhaustion returns OutOfRange and leaves the
// sentinel unchanged; this helper also makes the integer boundary independently testable.
absl::StatusOr<std::uint64_t> TakeLiquidArenaGeneration3D(LiquidArenaLedger3D& ledger);
// Zeroes deterministic execution padding and packs each semantic field without
// copying host padding. Caller has validated source and all reserved extents.
void PackLiquidArenaState3D(
    const LiquidState3D& source,
    LiquidClock3D clock,
    const CudaLiquidStateLayout3D& layout,
    std::byte* destination
);
// Caller has fenced D2H and verified output capacities. No fallible operations.
void UnpackLiquidArenaState3D(
    const CudaLiquidStateLayout3D& layout,
    const std::byte* source,
    LiquidState3D& destination
);
// Releases one reference outside attempts. The final release checks drain and
// cleanup results. Uncertain remaining resources and their ledger are retained in
// a process-lifetime quarantine; never retried or recycled by a later arena.
void ReleaseLiquidArenaStorage3D(LiquidArenaStorage3D* storage) noexcept;
}  // namespace fluidity::cuda_internal
