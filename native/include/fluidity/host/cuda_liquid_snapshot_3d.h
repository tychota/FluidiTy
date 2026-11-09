#pragma once
#include <memory>

#include "fluidity/host/cuda_liquid_arena_3d.h"
namespace fluidity {
namespace cuda_internal {
struct LiquidSnapshotStorage3D;
}
// Job/queue-local reservation identity. A recycled slot receives a never-reused
// reservation number; all methods compare that number and the full source epoch.
// Tickets are values, not resource owners, and must be used with their own queue.
struct LiquidSnapshotTicket3D {
  std::size_t slot = 0;
  std::uint64_t reservation = 0;
  CudaLiquidEpochIdentity3D epoch;
  bool operator==(const LiquidSnapshotTicket3D&) const = default;
};
// Pending is ordinary nonblocking backpressure. Canceled is returned once when
// a canceled pending copy completes safely; its ticket is stale afterward.
enum class LiquidSnapshotProgress3D {
  kPending,
  kReady,
  kCanceled
};
// Move-only immutable host destination. Retain through cache serialization and
// publication; destroying/replacing it releases its reservation for reuse. The
// decoded state, pinned payload and handles outlive queue/arena destruction.
// Release outside physical attempts, on the same single job thread.
class LiquidSnapshotLease3D {
 public:
  LiquidSnapshotLease3D() = default;
  LiquidSnapshotLease3D(LiquidSnapshotLease3D&&) noexcept;
  LiquidSnapshotLease3D& operator=(LiquidSnapshotLease3D&&) noexcept;
  ~LiquidSnapshotLease3D();
  LiquidSnapshotLease3D(const LiquidSnapshotLease3D&) = delete;
  LiquidSnapshotLease3D& operator=(const LiquidSnapshotLease3D&) = delete;
  // Complete validated semantic state; null for a default/moved-from lease.
  // Borrow is immutable and ends with this lease. No device read is performed.
  const LiquidState3D* state() const;
  // Exact producing generation and completed-frame clock, zero if empty.
  CudaLiquidEpochIdentity3D identity() const;
  // CUDA event interval around D2H submission, milliseconds; includes any stream
  // dependencies or host submission gaps between events, not a kernel duration.
  float copy_milliseconds() const;

 private:
  friend class CudaLiquidSnapshotQueue3D;
  cuda_internal::LiquidSnapshotStorage3D* storage_ = nullptr;
  std::size_t slot_ = 0;
};
// Two fixed reservations, each with pinned ABI bytes, decoded host records, an
// explicit nonblocking stream and two timing events. A third independent host
// state retains the latest validated snapshot for device-failure diagnostics.
// Single-threaded/non-reentrant; all operations run outside physical attempts.
// The selected device must remain current. No successful steady-state operation
// globally synchronizes, allocates or creates CUDA resources. Final destruction
// drains private streams; unproven cleanup quarantines all remaining resources
// and source leases for process lifetime, charged to the shared arena ledger.
class CudaLiquidSnapshotQueue3D {
 public:
  CudaLiquidSnapshotQueue3D(CudaLiquidSnapshotQueue3D&&) noexcept;
  CudaLiquidSnapshotQueue3D& operator=(CudaLiquidSnapshotQueue3D&&) noexcept;
  ~CudaLiquidSnapshotQueue3D();
  CudaLiquidSnapshotQueue3D(const CudaLiquidSnapshotQueue3D&) = delete;
  CudaLiquidSnapshotQueue3D& operator=(const CudaLiquidSnapshotQueue3D&) = delete;
  // Reserves the exact trusted source layout/capacities and joins its job ledger.
  // Does not consume/retain shape's source lease and does not perform D2H. Before
  // arena growth, finish/cancel copies and retain writer leases as needed; create
  // a replacement queue for the grown shape outside attempts. Preserve the old
  // queue's last-valid state until the new queue has a validated snapshot.
  // InvalidArgument for empty shape, FailedPrecondition for wrong current device,
  // ResourceExhausted for setup OOM, OutOfRange for ownership counter exhaustion,
  // Internal for sticky runtime failure. No partially initialized queue escapes.
  static absl::StatusOr<CudaLiquidSnapshotQueue3D> Create(const CudaLiquidEpochLease3D& shape);
  // Captures a completed-frame source (substep zero, including the initial state)
  // with the exact supplied epoch. InvalidArgument rejects wrong job/shape/epoch;
  // AlreadyExists rejects a live duplicate; FailedPrecondition rejects an epoch
  // at/before the last validated generation; OutOfRange rejects ticket exhaustion.
  // Unavailable means both destinations are held: service the queue/writers and
  // retry outside attempts. Validation/backpressure never consumes source.
  // Once submission starts, ownership transfers even on Internal: all referenced
  // particle/carrier/topology/control bytes remain fenced through success or
  // proven cleanup. Runtime errors poison the shared job; no recovery/reread.
  absl::StatusOr<LiquidSnapshotTicket3D> BeginCopy(CudaLiquidEpochIdentity3D epoch, CudaLiquidEpochLease3D&& source);
  // Nonblocking event query. Pending preserves both source and destination;
  // success releases only the source. FailedPrecondition for stale/writer tickets,
  // Internal for sticky CUDA failure (including failures during source release).
  // Ready does not publish or permit destination reuse. Never rereads the device.
  absl::StatusOr<LiquidSnapshotProgress3D> Poll(LiquidSnapshotTicket3D ticket);
  // Requires Ready and a generation/frame/time no older than last valid. Checks
  // copied active extents, exact clock, flags and every decoded state invariant
  // before publishing a writer lease and replacing LastValidHostSnapshot.
  // FailedPrecondition for pending/stale/out-of-order/canceled tickets; DataLoss
  // for malformed copied bytes (also poisons job); Internal for runtime poison;
  // OutOfRange for ownership exhaustion. Every failure preserves last valid.
  absl::StatusOr<LiquidSnapshotLease3D> Finish(LiquidSnapshotTicket3D ticket);
  // Suppresses publication. Pending still owns both DMA sides: continue Poll
  // until Canceled or destroy the queue safely. Ready releases its reservation.
  // FailedPrecondition for stale/writer tickets; Internal for poisoned jobs.
  absl::Status Cancel(LiquidSnapshotTicket3D ticket);
  // Null until first successful Finish; otherwise a read-only borrow until the
  // next successful Finish or queue destruction/move. Independent of writer
  // leases; failed/canceled/sticky operations preserve all its bytes. No CUDA call.
  const LiquidState3D* LastValidHostSnapshot() const;

 private:
  CudaLiquidSnapshotQueue3D() = default;
  cuda_internal::LiquidSnapshotStorage3D* storage_ = nullptr;
};
}  // namespace fluidity
