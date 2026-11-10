#pragma once
#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "liquid_arena_3d_internal.h"
namespace fluidity::cuda_internal {
template <typename Value>
struct LiquidTypedRange3D {
  Value* device;
  std::span<Value> pinned;
};
// Borrows count active typed records from the paired device and pinned ranges.
// Checks readiness, reservation size and both pointer alignments. Host-only
// ranges are rejected. The borrow ends on growth commit, owner move or
// destruction, or runtime poison. Host staging can defer device scratch
// accounting until dispatch.
template <typename Value>
absl::StatusOr<LiquidTypedRange3D<Value>> CheckedLiquidRange3D(
    CudaLiquidArena3D& arena,
    CudaLiquidRange3D name,
    std::size_t count,
    bool observe_device_scratch = true
) {
  static_assert(std::is_trivially_copyable_v<Value>);
  const auto ready = arena.RecordRuntimeResult(cudaSuccess);
  if (!ready.ok()) {
    return ready;
  }
  const auto range = arena.range(name);
  if (count > range.pinned.size() / sizeof(Value)) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  if (!range.device || !range.pinned.data() || reinterpret_cast<std::uintptr_t>(range.device) % alignof(Value) != 0
      || reinterpret_cast<std::uintptr_t>(range.pinned.data()) % alignof(Value) != 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (observe_device_scratch) {
    arena.ObserveDeviceScratch(name, count * sizeof(Value));
  }
  return LiquidTypedRange3D<Value>{
      reinterpret_cast<Value*>(range.device),
      {reinterpret_cast<Value*>(range.pinned.data()), count},
  };
}
// Enqueues the active bytes on the arena stream. Upload copies pinned to
// device. Empty ranges only check readiness. Keep both endpoints live and wait
// for a successful fence before reading downloaded bytes or reusing in-flight
// pinned storage. Submission failures poison the arena.
template <typename Value>
absl::Status CopyLiquidRange3D(CudaLiquidArena3D& arena, const LiquidTypedRange3D<Value>& range, bool upload) {
  const auto ready = arena.RecordRuntimeResult(cudaSuccess);
  if (!ready.ok() || range.pinned.empty()) {
    return ready;
  }
  const auto measured = arena.BeginMeasuredDispatch();
  if (!measured.ok()) {
    return measured;
  }
  return arena.RecordRuntimeResult(LiquidArenaRuntimeCalls().memcpy_async(
      upload ? range.device : range.pinned.data(),
      upload ? range.pinned.data() : range.device,
      range.pinned.size_bytes(),
      upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
      static_cast<cudaStream_t>(arena.stream())
  ));
}
// Fences prior work on the arena stream and collects enabled dispatch timing.
// Independent snapshot copies require their own completion fence.
inline absl::Status FinishLiquidDispatch3D(CudaLiquidArena3D& arena) {
  return arena.FinishDispatch();
}
// Tests whether two byte ranges overlap. An empty range never overlaps.
inline bool
LiquidByteRangesOverlap3D(const void* first, std::size_t first_bytes, const void* second, std::size_t second_bytes) {
  if (first_bytes == 0 || second_bytes == 0) {
    return false;
  }
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left < first_bytes : left - right < second_bytes;
}
// Tests a host span against every reserved pinned scratch range. Check
// overlap between external spans separately.
template <typename Value>
bool AliasesLiquidScratch3D(CudaLiquidArena3D& arena, std::span<Value> values) {
  for (std::size_t index = 0; index < static_cast<std::size_t>(CudaLiquidRange3D::kCount); ++index) {
    const auto range = arena.range(static_cast<CudaLiquidRange3D>(index));
    if (LiquidByteRangesOverlap3D(values.data(), values.size_bytes(), range.pinned.data(), range.pinned.size())) {
      return true;
    }
  }
  return false;
}
}  // namespace fluidity::cuda_internal
