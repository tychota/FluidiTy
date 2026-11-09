#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <new>
#include <utility>

#include "fluidity/host/cuda_liquid_snapshot_3d.h"
#include "liquid_arena_3d_internal.h"

namespace fluidity::cuda_internal {
// Pending owns both sides of DMA. Ready releases the source but retains the
// destination. A writer holds its destination until its separate lease ends.
enum class SnapshotSlotState3D {
  kFree,
  kPending,
  kReady,
  kWriter
};
struct LiquidSnapshotSlot3D {
  CudaLiquidEpochLease3D source;
  LiquidSnapshotTicket3D ticket;
  LiquidStateCapacities3D active;
  LiquidState3D decoded;
  std::byte* pinned = nullptr;
  cudaStream_t stream = nullptr;
  std::array<cudaEvent_t, 2> events{};
  SnapshotSlotState3D state = SnapshotSlotState3D::kFree;
  bool canceled = false;
  float copy_milliseconds = 0;
};
struct LiquidSnapshotStorage3D {
  LiquidArenaLedger3D* ledger = nullptr;
  CudaLiquidStateLayout3D layout;
  LiquidStateCapacities3D capacities;
  std::array<LiquidSnapshotSlot3D, 2> slots;
  LiquidState3D last_valid;
  CudaLiquidEpochIdentity3D last_identity;
  std::size_t references = 1;
  LiquidSnapshotStorage3D* quarantined_next = nullptr;
};
}  // namespace fluidity::cuda_internal
namespace fluidity {
namespace {
using cuda_internal::LiquidSnapshotSlot3D;
using cuda_internal::LiquidSnapshotStorage3D;
using cuda_internal::SnapshotSlotState3D;
absl::Status Error(absl::StatusCode code) {
  return {code, ""};
}
// Keep the same poison/accounting domain as the producing device arena. NotReady
// is handled before this function and never becomes job failure.
absl::Status RuntimeResult(LiquidSnapshotStorage3D& storage, cudaError_t result) {
  if (result != cudaSuccess) {
    storage.ledger->runtime_failed = true;
  }
  return storage.ledger->runtime_failed ? Error(absl::StatusCode::kInternal) : absl::OkStatus();
}
std::atomic<LiquidSnapshotStorage3D*> quarantined_snapshots{nullptr};
void Quarantine(LiquidSnapshotStorage3D& storage) {
  ++storage.ledger->metrics.quarantined_storage_count;
  storage.references = 1;
  auto* head = quarantined_snapshots.load(std::memory_order_relaxed);
  do {
    storage.quarantined_next = head;
  } while (
      !quarantined_snapshots.compare_exchange_weak(head, &storage, std::memory_order_release, std::memory_order_relaxed)
  );
}
bool CleanupSucceeded(LiquidSnapshotStorage3D& storage, cudaError_t result) {
  (void)RuntimeResult(storage, result);
  return result == cudaSuccess;
}
// Drains every reservation before releasing any potentially referenced source.
// A failed drain retains the entire owner, including both leases and pinned bytes.
void Release(LiquidSnapshotStorage3D* storage) {
  if (!storage || --storage->references != 0) {
    return;
  }
  int previous_device = -1;
  if (!CleanupSucceeded(*storage, cudaGetDevice(&previous_device))
      || !CleanupSucceeded(*storage, cudaSetDevice(storage->ledger->device_ordinal))) {
    Quarantine(*storage);
    return;
  }
  const auto cleanup = [&]() {
    for (auto& slot : storage->slots) {
      if (slot.stream
          && !CleanupSucceeded(*storage, cuda_internal::LiquidArenaRuntimeCalls().stream_synchronize(slot.stream))) {
        return false;
      }
    }
    for (auto& slot : storage->slots) {
      slot.source = {};
      for (auto& event : slot.events) {
        if (event) {
          ++storage->ledger->metrics.event_destructions;
          if (!CleanupSucceeded(*storage, cudaEventDestroy(event))) {
            return false;
          }
          event = nullptr;
        }
      }
      if (slot.stream) {
        ++storage->ledger->metrics.stream_destructions;
        if (!CleanupSucceeded(*storage, cudaStreamDestroy(slot.stream))) {
          return false;
        }
        slot.stream = nullptr;
      }
      if (slot.pinned) {
        ++storage->ledger->metrics.pinned_frees;
        if (!CleanupSucceeded(*storage, cudaFreeHost(slot.pinned))) {
          return false;
        }
        storage->ledger->metrics.live_pinned_bytes -= storage->layout.total_state_bytes;
        slot.pinned = nullptr;
      }
    }
    return true;
  };
  const bool released = cleanup();
  if (previous_device != storage->ledger->device_ordinal) {
    (void)RuntimeResult(*storage, cudaSetDevice(previous_device));
  }
  if (!released) {
    Quarantine(*storage);
    return;
  }
  if (--storage->ledger->references == 0) {
    delete storage->ledger;
  }
  delete storage;
}
LiquidSnapshotSlot3D* Find(LiquidSnapshotStorage3D* storage, LiquidSnapshotTicket3D ticket) {
  if (!storage || ticket.slot >= storage->slots.size()) {
    return nullptr;
  }
  auto& slot = storage->slots[ticket.slot];
  if (slot.state == SnapshotSlotState3D::kFree || slot.ticket != ticket) {
    return nullptr;
  }
  return &slot;
}
bool SameLayout(const CudaLiquidStateLayout3D& a, const CudaLiquidStateLayout3D& b) {
  return a.particle_bins_byte_offset == b.particle_bins_byte_offset
      && a.block_records_byte_offset == b.block_records_byte_offset
      && a.cell_records_byte_offset == b.cell_records_byte_offset
      && a.face_records_byte_offset == b.face_records_byte_offset
      && a.source_records_byte_offset == b.source_records_byte_offset
      && a.control_record_byte_offset == b.control_record_byte_offset && a.total_state_bytes == b.total_state_bytes
      && a.particle_bin_stride_bytes == b.particle_bin_stride_bytes;
}
// Reserved destination capacities are identical. This copies semantic records
// only; it cannot allocate, fail or alias the source reservation.
void CopyState(const LiquidState3D& source, LiquidState3D& destination) {
  const auto copy = [](const auto& input, auto& output) {
    (void)output.Resize(input.size());
    std::copy(input.values().begin(), input.values().end(), output.values().begin());
  };
  copy(source.particles, destination.particles);
  copy(source.blocks, destination.blocks);
  copy(source.cells, destination.cells);
  copy(source.faces, destination.faces);
  copy(source.sources, destination.sources);
  destination.geometry = source.geometry;
  destination.clock = source.clock;
  destination.random_seed = source.random_seed;
  destination.next_particle_id = source.next_particle_id;
}
// Bounds come from the source lease's trusted layout and captured active extents.
// Check copied counts/clock and bool/flag representation before typed decode.
absl::Status Decode(LiquidSnapshotStorage3D& storage, LiquidSnapshotSlot3D& slot) {
  const auto& control = *reinterpret_cast<const cuda_internal::LiquidArenaControl3D*>(
      slot.pinned + storage.layout.control_record_byte_offset
  );
  if (control.active != slot.active || control.clock != slot.ticket.epoch.clock) {
    return Error(absl::StatusCode::kDataLoss);
  }
  const auto* bins =
      reinterpret_cast<const CudaLiquidParticleBin3D*>(slot.pinned + storage.layout.particle_bins_byte_offset);
  for (std::size_t index = 0; index < slot.active.particles; ++index) {
    if (bins[index / 32].affine_pic_flags[index % 32] > 3U) {
      return Error(absl::StatusCode::kDataLoss);
    }
  }
  for (std::size_t index = 0; index < slot.active.faces; ++index) {
    const auto* flag = slot.pinned + storage.layout.face_records_byte_offset + index * sizeof(LiquidFaceState3D)
        + offsetof(LiquidFaceState3D, structurally_eligible);
    const bool false_value = false;
    const bool true_value = true;
    if (std::memcmp(flag, &false_value, sizeof(bool)) != 0 && std::memcmp(flag, &true_value, sizeof(bool)) != 0) {
      return Error(absl::StatusCode::kDataLoss);
    }
  }
  cuda_internal::UnpackLiquidArenaState3D(storage.layout, slot.pinned, slot.decoded);
  if (!ValidateLiquidState3D(slot.decoded).ok()) {
    return Error(absl::StatusCode::kDataLoss);
  }
  return absl::OkStatus();
}
}  // namespace
CudaLiquidSnapshotQueue3D::CudaLiquidSnapshotQueue3D(CudaLiquidSnapshotQueue3D&& other) noexcept
    : storage_(std::exchange(other.storage_, nullptr)) {
}
CudaLiquidSnapshotQueue3D& CudaLiquidSnapshotQueue3D::operator=(CudaLiquidSnapshotQueue3D&& other) noexcept {
  if (this != &other) {
    Release(storage_);
    storage_ = std::exchange(other.storage_, nullptr);
  }
  return *this;
}
CudaLiquidSnapshotQueue3D::~CudaLiquidSnapshotQueue3D() {
  Release(storage_);
}
LiquidSnapshotLease3D::LiquidSnapshotLease3D(LiquidSnapshotLease3D&& other) noexcept
    : storage_(std::exchange(other.storage_, nullptr)), slot_(other.slot_) {
}
LiquidSnapshotLease3D& LiquidSnapshotLease3D::operator=(LiquidSnapshotLease3D&& other) noexcept {
  if (this != &other) {
    if (storage_) {
      storage_->slots[slot_].state = SnapshotSlotState3D::kFree;
      Release(storage_);
    }
    storage_ = std::exchange(other.storage_, nullptr);
    slot_ = other.slot_;
  }
  return *this;
}
LiquidSnapshotLease3D::~LiquidSnapshotLease3D() {
  if (storage_) {
    storage_->slots[slot_].state = SnapshotSlotState3D::kFree;
    Release(storage_);
  }
}
const LiquidState3D* LiquidSnapshotLease3D::state() const {
  return storage_ ? &storage_->slots[slot_].decoded : nullptr;
}
CudaLiquidEpochIdentity3D LiquidSnapshotLease3D::identity() const {
  return storage_ ? storage_->slots[slot_].ticket.epoch : CudaLiquidEpochIdentity3D{};
}
float LiquidSnapshotLease3D::copy_milliseconds() const {
  return storage_ ? storage_->slots[slot_].copy_milliseconds : 0;
}
absl::StatusOr<CudaLiquidSnapshotQueue3D> CudaLiquidSnapshotQueue3D::Create(const CudaLiquidEpochLease3D& shape) {
  if (!shape.storage_) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  auto* ledger = shape.storage_->ledger;
  if (ledger->runtime_failed) {
    return Error(absl::StatusCode::kInternal);
  }
  if (ledger->references == SIZE_MAX) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  const auto bytes = shape.layout().total_state_bytes;
  if (bytes > (SIZE_MAX - ledger->metrics.live_pinned_bytes) / 2) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    ledger->runtime_failed = true;
    return Error(absl::StatusCode::kInternal);
  }
  if (device != ledger->device_ordinal) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  CudaLiquidSnapshotQueue3D queue;
  queue.storage_ = new (std::nothrow) LiquidSnapshotStorage3D;
  if (!queue.storage_) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  auto& storage = *queue.storage_;
  storage.ledger = ledger;
  ++ledger->references;
  storage.layout = shape.layout();
  storage.capacities = shape.storage_->capacities;
  // Cleanup may reveal a more serious runtime failure than setup OOM. The
  // caller's shape keeps the ledger alive while we check the final cleanup result.
  const auto fail = [&](absl::Status status) {
    Release(std::exchange(queue.storage_, nullptr));
    return ledger->runtime_failed ? Error(absl::StatusCode::kInternal) : status;
  };
  auto last = CloneLiquidState3D(LiquidState3D{}, storage.capacities);
  if (!last.ok()) {
    return fail(last.status());
  }
  storage.last_valid = std::move(*last);
  for (auto& slot : storage.slots) {
    auto decoded = CloneLiquidState3D(LiquidState3D{}, storage.capacities);
    if (!decoded.ok()) {
      return fail(decoded.status());
    }
    slot.decoded = std::move(*decoded);
    ++ledger->metrics.pinned_reservations;
    const auto allocation = cudaMallocHost(reinterpret_cast<void**>(&slot.pinned), bytes);
    if (allocation != cudaSuccess) {
      if (allocation == cudaErrorMemoryAllocation) {
        return fail(Error(absl::StatusCode::kResourceExhausted));
      }
      return fail(RuntimeResult(storage, allocation));
    }
    new (slot.pinned) std::byte[bytes];
    ledger->metrics.live_pinned_bytes += bytes;
    ledger->metrics.peak_pinned_bytes = std::max(ledger->metrics.peak_pinned_bytes, ledger->metrics.live_pinned_bytes);
    ++ledger->metrics.stream_creations;
    auto status = RuntimeResult(storage, cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
    if (!status.ok()) {
      return fail(status);
    }
    for (auto& event : slot.events) {
      ++ledger->metrics.event_creations;
      status = RuntimeResult(storage, cudaEventCreate(&event));
      if (!status.ok()) {
        return fail(status);
      }
    }
  }
  return queue;
}
absl::StatusOr<LiquidSnapshotTicket3D>
CudaLiquidSnapshotQueue3D::BeginCopy(CudaLiquidEpochIdentity3D epoch, CudaLiquidEpochLease3D&& source) {
  if (!storage_) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  auto& storage = *storage_;
  if (storage.ledger->runtime_failed) {
    return Error(absl::StatusCode::kInternal);
  }
  if (!source.storage_ || source.storage_->ledger != storage.ledger || source.identity() != epoch
      || epoch.generation == 0 || epoch.clock.substep_ordinal != 0) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (!SameLayout(source.layout(), storage.layout) || source.storage_->capacities != storage.capacities) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (epoch.generation <= storage.last_identity.generation) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  for (const auto& slot : storage.slots) {
    if (slot.state != SnapshotSlotState3D::kFree && slot.ticket.epoch == epoch) {
      return Error(absl::StatusCode::kAlreadyExists);
    }
  }
  if (storage.ledger->next_snapshot_reservation == UINT64_MAX) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  for (std::size_t index = 0; index < storage.slots.size(); ++index) {
    auto& slot = storage.slots[index];
    if (slot.state != SnapshotSlotState3D::kFree) {
      continue;
    }
    slot.ticket = {index, storage.ledger->next_snapshot_reservation++, epoch};
    slot.active = source.storage_->active[source.bank_];
    slot.source = std::move(source);
    slot.state = SnapshotSlotState3D::kPending;
    slot.canceled = false;
    slot.copy_milliseconds = 0;
    auto& calls = cuda_internal::LiquidArenaRuntimeCalls();
    auto status = RuntimeResult(storage, calls.event_record(slot.events[0], slot.stream));
    if (!status.ok()) {
      return status;
    }
    status = RuntimeResult(
        storage,
        calls.memcpy_async(
            slot.pinned,
            slot.source.device_bytes().data(),
            storage.layout.total_state_bytes,
            cudaMemcpyDeviceToHost,
            slot.stream
        )
    );
    if (!status.ok()) {
      return status;
    }
    status = RuntimeResult(storage, calls.event_record(slot.events[1], slot.stream));
    if (!status.ok()) {
      return status;
    }
    return slot.ticket;
  }
  return Error(absl::StatusCode::kUnavailable);
}
absl::StatusOr<LiquidSnapshotProgress3D> CudaLiquidSnapshotQueue3D::Poll(LiquidSnapshotTicket3D ticket) {
  auto* slot = Find(storage_, ticket);
  if (!slot) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  if (storage_->ledger->runtime_failed) {
    return Error(absl::StatusCode::kInternal);
  }
  if (slot->state == SnapshotSlotState3D::kWriter) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  if (slot->state == SnapshotSlotState3D::kReady) {
    return LiquidSnapshotProgress3D::kReady;
  }
  const auto result = cuda_internal::LiquidArenaRuntimeCalls().event_query(slot->events[1]);
  if (result == cudaErrorNotReady) {
    return LiquidSnapshotProgress3D::kPending;
  }
  auto status = RuntimeResult(*storage_, result);
  if (!status.ok()) {
    return status;
  }
  status = RuntimeResult(*storage_, cudaEventElapsedTime(&slot->copy_milliseconds, slot->events[0], slot->events[1]));
  if (!status.ok()) {
    return status;
  }
  if (!std::isfinite(slot->copy_milliseconds) || slot->copy_milliseconds < 0) {
    storage_->ledger->runtime_failed = true;
    return Error(absl::StatusCode::kInternal);
  }
  slot->source = {};
  if (storage_->ledger->runtime_failed) {
    return Error(absl::StatusCode::kInternal);
  }
  if (slot->canceled) {
    slot->state = SnapshotSlotState3D::kFree;
    return LiquidSnapshotProgress3D::kCanceled;
  }
  slot->state = SnapshotSlotState3D::kReady;
  return LiquidSnapshotProgress3D::kReady;
}
absl::StatusOr<LiquidSnapshotLease3D> CudaLiquidSnapshotQueue3D::Finish(LiquidSnapshotTicket3D ticket) {
  auto* slot = Find(storage_, ticket);
  if (!slot) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  if (storage_->ledger->runtime_failed) {
    return Error(absl::StatusCode::kInternal);
  }
  if (slot->state != SnapshotSlotState3D::kReady || slot->canceled
      || ticket.epoch.generation <= storage_->last_identity.generation
      || ticket.epoch.clock.frame_ordinal < storage_->last_identity.clock.frame_ordinal
      || ticket.epoch.clock.time < storage_->last_identity.clock.time) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  if (storage_->references == SIZE_MAX) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  auto status = Decode(*storage_, *slot);
  if (!status.ok()) {
    storage_->ledger->runtime_failed = true;
    return status;
  }
  CopyState(slot->decoded, storage_->last_valid);
  storage_->last_identity = ticket.epoch;
  slot->state = SnapshotSlotState3D::kWriter;
  ++storage_->references;
  LiquidSnapshotLease3D lease;
  lease.storage_ = storage_;
  lease.slot_ = ticket.slot;
  return lease;
}
absl::Status CudaLiquidSnapshotQueue3D::Cancel(LiquidSnapshotTicket3D ticket) {
  auto* slot = Find(storage_, ticket);
  if (!slot || slot->state == SnapshotSlotState3D::kWriter) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  if (storage_->ledger->runtime_failed) {
    return Error(absl::StatusCode::kInternal);
  }
  slot->canceled = true;
  if (slot->state == SnapshotSlotState3D::kReady) {
    slot->state = SnapshotSlotState3D::kFree;
  }
  return absl::OkStatus();
}
const LiquidState3D* CudaLiquidSnapshotQueue3D::LastValidHostSnapshot() const {
  return storage_ && storage_->last_identity.generation != 0 ? &storage_->last_valid : nullptr;
}
}  // namespace fluidity
