#pragma once
#include <memory>

#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
namespace fluidity {
struct CudaLiquidSubstepUsage3D {
  std::size_t device_reserved_bytes = 0;  // Live explicit device payload, including leased retired arenas.
  std::size_t pinned_reserved_bytes = 0;  // Live pinned payload, including pressure preparation scratch.
  std::size_t device_memory_ceiling_bytes = 0;  // Fixed job budget against peak simultaneous payload.
  std::size_t device_allocation_count = 0;  // Actual device reservation calls, including failures.
  std::size_t device_free_count = 0;  // Actual release calls for acquired device allocations.
  std::size_t graph_preparation_count = 0;  // Actual graph creation attempts across growth.
  std::size_t graph_replay_count = 0;  // Authoritative signed-sort graph replays, including empty input.
  CudaLiquidArenaMetrics3D arena_metrics{};  // All active, prospective and leased owners.
  int compiled_architecture = 0;  // Device-written __CUDA_ARCH__.
  // Latest successful topology-through-publication CUDA event interval, ms.
  // Includes host preparation and dispatch gaps; not a pure kernel duration.
  float substep_device_timeline_ms = 0;
};
// Persistent real CUDA operations for the common transactional scheduler. Create
// reserves device/pinned storage and graph resources outside physical attempts.
// Unusable hardware fails without CPU substitution. The owner must outlive its
// single scheduler; moves preserve callback context. Single-threaded, non-reentrant.
// Every fallible publication operation completes before the host accepts; commit
// only flips metadata. A runtime fault permanently denies subsequent device work.
// Create, growth, collection, move and destruction run outside physical attempts.
class CudaLiquidSubstepBackend3D {
 public:
  CudaLiquidSubstepBackend3D(CudaLiquidSubstepBackend3D&&) noexcept;
  CudaLiquidSubstepBackend3D& operator=(CudaLiquidSubstepBackend3D&&) noexcept;
  ~CudaLiquidSubstepBackend3D();
  CudaLiquidSubstepBackend3D(const CudaLiquidSubstepBackend3D&) = delete;
  CudaLiquidSubstepBackend3D& operator=(const CudaLiquidSubstepBackend3D&) = delete;
  // Requires the qualified CC12.0 device and checked capacity extents.
  // Propagates the device, image, capacity and runtime statuses from
  // CudaLiquidArena3D::Create. Owner allocation failure returns ResourceExhausted.
  // Reserve wetted-wall scratch for the uniform translating-piston path.
  // Adaptive state uses the qualified explicit solver on device 0.
  static absl::StatusOr<CudaLiquidSubstepBackend3D> Create(
      LiquidStateCapacities3D capacities,
      int device_ordinal,
      bool measure_job = false,
      bool needs_wetted_piston = false,
      bool reserve_adaptive_state = false
  );
  // Immutable committed device source, including its nonreused generation and
  // physical clock. FailedPrecondition before scheduler initialization; ownership
  // counter exhaustion is OutOfRange. Acquire allocates nothing; keep the lease until the consumer's
  // D2H completion fence. Release outside attempts. See CudaLiquidEpochLease3D.
  absl::StatusOr<CudaLiquidEpochLease3D> AcquireCommitted();
  // Outside-attempt scheduler gate. False is bounded snapshot backpressure;
  // poll copies/release completed source leases before starting the next attempt.
  // It is neither a physical rejection nor capacity demand. Propagates arena
  // Internal poison / FailedPrecondition sequencing. See arena PublicationReady.
  absl::StatusOr<bool> PublicationReady() const;
  // Collect only unleased retired owners outside physical attempts. A failed
  // drain poisons the job and quarantines storage; no subsequent reuse is allowed.
  void CollectRetired() noexcept;
  // Complete borrowed typed dispatch. Moving the owner preserves its context.
  // Opt in to speculative scheduler gather runs. Field batches require both
  // flags and apply only to particles whose grouped gather has succeeded.
  LiquidSubstepBackend3D operations(bool enable_gather_runs = false, bool enable_gathered_field_batches = false);
  // Actual reservations and CUDA call counts, including failed/replayed work.
  CudaLiquidSubstepUsage3D usage() const;
  // Consumes the backend after scheduler and snapshot drain, collecting the
  // same ownership ledger after arena teardown. External leases remain counted.
  CudaLiquidArenaMetrics3D DrainMetrics() &&;

 private:
  struct Impl;
  explicit CudaLiquidSubstepBackend3D(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace fluidity
