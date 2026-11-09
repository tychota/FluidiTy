#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "liquid_arena_3d_internal.h"
#include "liquid_boundary_3d_bridge.h"
#include "liquid_boundary_transfer_3d_bridge.h"
#include "mac_pressure_3d_bridge.h"

namespace fluidity {
namespace {
using cuda_internal::LiquidArenaControl3D;
using cuda_internal::LiquidArenaLedger3D;
using cuda_internal::LiquidArenaStorage3D;
using cuda_internal::LiquidSortKey3D;
using cuda_internal::ReleaseLiquidArenaStorage3D;
constexpr std::size_t kAlignment = 128;
constexpr std::size_t kByteLimit = static_cast<std::size_t>(PTRDIFF_MAX);
absl::Status Error(absl::StatusCode code) {
  return absl::Status(code, {});
}
absl::Status CheckedCuda(cudaError_t result) {
  if (result == cudaSuccess) {
    return absl::OkStatus();
  }
  if (result == cudaErrorMemoryAllocation) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  return Error(absl::StatusCode::kInternal);
}
// Every product and alignment addition is checked before advancing the cursor.
class ByteLayout {
 public:
  bool Add(std::size_t count, std::size_t stride, std::size_t& offset) {
    if (bytes > kByteLimit - (kAlignment - 1)) {
      return false;
    }
    const auto aligned = (bytes + kAlignment - 1) / kAlignment * kAlignment;
    if (stride != 0 && count > (kByteLimit - aligned) / stride) {
      return false;
    }
    offset = aligned;
    bytes = aligned + count * stride;
    return true;
  }
  bool Finish() {
    std::size_t unused = 0;
    return Add(0, 0, unused);
  }
  std::size_t bytes = 0;
};
bool Fits(LiquidStateCapacities3D active, LiquidStateCapacities3D capacity) {
  return active.particles <= capacity.particles && active.blocks <= capacity.blocks && active.cells <= capacity.cells
      && active.faces <= capacity.faces && active.sources <= capacity.sources;
}
LiquidStateCapacities3D Active(const LiquidState3D& state) {
  return {state.particles.size(), state.blocks.size(), state.cells.size(), state.faces.size(), state.sources.size()};
}
bool SameGeometry(UniformMacGeometry3D left, UniformMacGeometry3D right) {
  return left.cell_width == right.cell_width && left.world_origin.x == right.world_origin.x
      && left.world_origin.y == right.world_origin.y && left.world_origin.z == right.world_origin.z;
}
template <typename Value>
Value* Device(LiquidArenaStorage3D& storage, CudaLiquidRange3D name) {
  return reinterpret_cast<Value*>(storage.Range(name).device);
}
template <typename Value>
Value* Pinned(LiquidArenaStorage3D& storage, CudaLiquidRange3D name) {
  return reinterpret_cast<Value*>(storage.Range(name).pinned.data());
}
// Query the same fixed CUB shape used in capture. Numeric capacities bound every
// legal pressure level by the fine count, every CSR edge by the fine face count,
// and each terminal factor by 32 entries per unknown. Over-reservation is explicit.
absl::Status PlanStorage(LiquidArenaStorage3D& storage) {
  const auto capacity = storage.capacities;
  if (capacity.particles > static_cast<std::size_t>(INT32_MAX)) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  const auto pressure = QueryMacPressureWorkspaceBytes3D(capacity.cells, capacity.faces);
  if (!pressure.ok()) {
    return pressure.status();
  }
  ByteLayout state;
  auto& layout = storage.state_layout;
  const std::size_t bin_count = capacity.particles / 32 + (capacity.particles % 32 != 0 ? 1 : 0);
  if (!state.Add(bin_count, sizeof(CudaLiquidParticleBin3D), layout.particle_bins_byte_offset)
      || !state.Add(capacity.blocks, sizeof(LiquidBlockState3D), layout.block_records_byte_offset)
      || !state.Add(capacity.cells, sizeof(LiquidCellState3D), layout.cell_records_byte_offset)
      || !state.Add(capacity.faces, sizeof(LiquidFaceState3D), layout.face_records_byte_offset)
      || !state.Add(capacity.sources, sizeof(LiquidSourceState3D), layout.source_records_byte_offset)
      || !state.Add(1, sizeof(LiquidArenaControl3D), layout.control_record_byte_offset) || !state.Finish()) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  layout.total_state_bytes = state.bytes;
  ByteLayout all;
  std::size_t unused = 0;
  if (!all.Add(2, state.bytes, unused)) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  const auto add = [&](CudaLiquidRange3D name, std::size_t count, std::size_t stride) {
    auto& range = storage.ranges[static_cast<std::size_t>(name)];
    if (!all.Add(count, stride, range.offset)) {
      return false;
    }
    range.bytes = count * stride;
    return true;
  };
  storage.sort_capacity = std::max<std::size_t>(capacity.particles, 1);
  std::size_t temporary_bytes = 0;
  auto status = CheckedCuda(static_cast<cudaError_t>(
      cuda_internal::
          SortLiquidBins3D(nullptr, temporary_bytes, nullptr, nullptr, nullptr, nullptr, storage.sort_capacity, nullptr)
  ));
  if (!status.ok()) {
    return status;
  }
  const auto level_count = std::max<std::size_t>(pressure->maximum_levels, 1);
  const auto count_sum =
      [](std::size_t first, std::size_t first_factor, std::size_t second, std::size_t second_factor, std::size_t extra
      ) -> absl::StatusOr<std::size_t> {
    if (extra > kByteLimit || first > (kByteLimit - extra) / first_factor) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    const auto partial = extra + first * first_factor;
    if (second > (kByteLimit - partial) / second_factor) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    return partial + second * second_factor;
  };
  const auto index_count = count_sum(capacity.cells, 1 + 3 * level_count, capacity.faces, 4 * level_count, level_count);
  const auto coefficient_count = count_sum(capacity.cells, 34 * level_count, capacity.faces, level_count, 0);
  const auto pressure_output_count = count_sum(capacity.cells, 4, capacity.faces, 1, 0);
  if (!index_count.ok() || !coefficient_count.ok() || !pressure_output_count.ok()) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  using Range = CudaLiquidRange3D;
  const bool fits = add(Range::kSortInputKeys, storage.sort_capacity, sizeof(LiquidSortKey3D))
      && add(Range::kSortOutputKeys, storage.sort_capacity, sizeof(LiquidSortKey3D))
      && add(Range::kSortInputOrdinals, storage.sort_capacity, sizeof(std::size_t))
      && add(Range::kSortOutputOrdinals, storage.sort_capacity, sizeof(std::size_t))
      && add(Range::kSortEntries, capacity.particles, sizeof(LiquidParticleSortEntry3D))
      && add(Range::kSortTemporary, std::max<std::size_t>(temporary_bytes, 1), 1)
      && add(Range::kArchitecture, 1, sizeof(int))
      && add(Range::kGeometryInput, 1, sizeof(boundary_internal::GeometryInput))
      && add(Range::kContactInput, 1, sizeof(boundary_internal::ContactInput))
      && add(Range::kBoundaryReport, 1, sizeof(cuda_internal::BoundaryReport3D))
      && add(Range::kTransferQuery, 1, sizeof(cuda_internal::BoundaryTransferQuery3D))
      && add(Range::kTransferFaces, capacity.faces, sizeof(cuda_internal::BoundaryTransferFace3D))
      && add(Range::kTransferParticles, capacity.particles, sizeof(cuda_internal::BoundaryTransferParticle3D))
      && add(Range::kTransferFaceResults, capacity.faces, sizeof(cuda_internal::BoundaryTransferFaceResult3D))
      && add(Range::kTransferParticleResults,
             capacity.particles,
             sizeof(cuda_internal::BoundaryTransferParticleResult3D))
      && add(Range::kTransferReport, 1, sizeof(cuda_internal::BoundaryTransferReport3D))
      && add(Range::kPressureCells, capacity.cells, sizeof(cuda_internal::PressureCell3D))
      && add(Range::kPressureLinks, capacity.faces, sizeof(cuda_internal::PressureLink3D))
      && add(Range::kPressureLevels, level_count, sizeof(cuda_internal::PressureLevel3D))
      && add(Range::kPressureBlocks, capacity.cells, level_count * sizeof(cuda_internal::PressureBlock3D))
      && add(Range::kPressureIndices, *index_count, sizeof(std::size_t))
      && add(Range::kPressureCoefficients, *coefficient_count, sizeof(double))
      && add(Range::kPressureFineScratch, capacity.cells, 6 * sizeof(float))
      && add(Range::kPressureCoarseScratch, capacity.cells, 3 * level_count * sizeof(double))
      && add(Range::kPressureRows, capacity.cells, 2 * sizeof(cuda_internal::PressureRow3D))
      && add(Range::kPressureOutput, *pressure_output_count, sizeof(double))
      && add(Range::kPressureState, 1, sizeof(cuda_internal::PressureState3D))
      && add(Range::kSamplePositions, std::max({capacity.particles, capacity.faces, capacity.cells}), 8 * sizeof(Vec3))
      && add(Range::kSampleDistances, std::max({capacity.particles, capacity.faces, capacity.cells}), 8 * sizeof(double)
      )
      && add(Range::kSampleParticles, capacity.particles, sizeof(LiquidParticleState3D));
  if (!fits) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  if (!all.Finish()) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  storage.reserved_device_bytes = all.bytes;
  auto& preparation = storage.ranges[static_cast<std::size_t>(Range::kPressurePreparationScratch)];
  if (!all.Add(pressure->scratch_bytes, 1, preparation.offset) || !all.Finish()) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  preparation.bytes = pressure->scratch_bytes;
  storage.reserved_pinned_bytes = all.bytes;
  return absl::OkStatus();
}
absl::Status Fence(LiquidArenaStorage3D& storage) {
  auto status = CheckedCuda(cudaEventRecord(storage.completion, storage.stream));
  if (!status.ok()) {
    return status;
  }
  return CheckedCuda(cudaEventSynchronize(storage.completion));
}
absl::Status TransferRange(LiquidArenaStorage3D& storage, CudaLiquidRange3D name, bool upload) {
  const auto range = storage.Range(name);
  return CheckedCuda(cudaMemcpyAsync(
      upload ? range.device : range.pinned.data(),
      upload ? range.pinned.data() : range.device,
      range.pinned.size(),
      upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
      storage.stream
  ));
}
// Construction and cleanup happen outside attempts. A single device/pinned
// allocation makes all named slices stable and accounts padding exactly.
absl::StatusOr<LiquidArenaStorage3D*> BuildStorage(LiquidArenaLedger3D& ledger, LiquidStateCapacities3D capacities) {
  std::unique_ptr<LiquidArenaStorage3D> storage(new (std::nothrow) LiquidArenaStorage3D{});
  if (!storage) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  storage->ledger = &ledger;
  if (ledger.references == SIZE_MAX) {
    storage->ledger = nullptr;
    return Error(absl::StatusCode::kOutOfRange);
  }
  ++ledger.references;
  storage->capacities = capacities;
  auto status = PlanStorage(*storage);
  if (!status.ok()) {
    return status;
  }
  auto& metrics = ledger.metrics;
  if (storage->reserved_device_bytes > metrics.device_byte_limit - metrics.live_device_bytes) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  if (metrics.live_pinned_bytes > kByteLimit
      || storage->reserved_pinned_bytes > kByteLimit - metrics.live_pinned_bytes) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  ++metrics.device_reservations;
  status = CheckedCuda(cudaMalloc(reinterpret_cast<void**>(&storage->device), storage->reserved_device_bytes));
  if (!status.ok()) {
    return status;
  }
  metrics.live_device_bytes += storage->reserved_device_bytes;
  metrics.peak_device_bytes = std::max(metrics.peak_device_bytes, metrics.live_device_bytes);
  ++metrics.pinned_reservations;
  status = CheckedCuda(cudaMallocHost(reinterpret_cast<void**>(&storage->pinned), storage->reserved_pinned_bytes));
  if (!status.ok()) {
    return status;
  }
  metrics.live_pinned_bytes += storage->reserved_pinned_bytes;
  metrics.peak_pinned_bytes = std::max(metrics.peak_pinned_bytes, metrics.live_pinned_bytes);
  // Beginning a byte-array lifetime also permits implicit-lifetime POD objects
  // in its storage (C++20); cudaMallocHost itself is not a C++ allocation expression.
  storage->pinned = new (storage->pinned) std::byte[storage->reserved_pinned_bytes]{};
  ++metrics.stream_creations;
  status = CheckedCuda(cudaStreamCreateWithFlags(&storage->stream, cudaStreamNonBlocking));
  if (!status.ok()) {
    return status;
  }
  ++metrics.event_creations;
  status = CheckedCuda(cudaEventCreateWithFlags(&storage->completion, cudaEventDisableTiming));
  if (!status.ok()) {
    return status;
  }
  status = CheckedCuda(cudaMemsetAsync(storage->device, 0, storage->reserved_device_bytes, storage->stream));
  if (!status.ok()) {
    return status;
  }
  status = Fence(*storage);
  if (!status.ok()) {
    return status;
  }
  status = CheckedCuda(cudaStreamBeginCapture(storage->stream, cudaStreamCaptureModeThreadLocal));
  if (!status.ok()) {
    return status;
  }
  auto temporary = storage->Range(CudaLiquidRange3D::kSortTemporary);
  auto temporary_bytes = temporary.pinned.size();
  const auto sort_error = static_cast<cudaError_t>(cuda_internal::SortLiquidBins3D(
      temporary.device,
      temporary_bytes,
      Device<LiquidSortKey3D>(*storage, CudaLiquidRange3D::kSortInputKeys),
      Device<LiquidSortKey3D>(*storage, CudaLiquidRange3D::kSortOutputKeys),
      Device<std::size_t>(*storage, CudaLiquidRange3D::kSortInputOrdinals),
      Device<std::size_t>(*storage, CudaLiquidRange3D::kSortOutputOrdinals),
      storage->sort_capacity,
      storage->stream
  ));
  const auto launch_error = static_cast<cudaError_t>(cuda_internal::LaunchLiquidArenaArchitecture3D(
      Device<int>(*storage, CudaLiquidRange3D::kArchitecture),
      storage->stream
  ));
  ++metrics.graph_creations;
  const auto capture_error = cudaStreamEndCapture(storage->stream, &storage->graph);
  if (sort_error != cudaSuccess) {
    return CheckedCuda(sort_error);
  }
  if (launch_error != cudaSuccess) {
    return CheckedCuda(launch_error);
  }
  if (capture_error != cudaSuccess) {
    return CheckedCuda(capture_error);
  }
  ++metrics.executable_creations;
  status = CheckedCuda(cudaGraphInstantiate(&storage->executable, storage->graph, 0));
  if (!status.ok()) {
    return status;
  }
  // Warm device/runtime graph paths outside attempts, then require device evidence.
  status = CheckedCuda(cudaGraphLaunch(storage->executable, storage->stream));
  if (!status.ok()) {
    return status;
  }
  status = TransferRange(*storage, CudaLiquidRange3D::kArchitecture, false);
  if (!status.ok()) {
    return status;
  }
  status = Fence(*storage);
  if (!status.ok()) {
    return status;
  }
  metrics.compiled_architecture = *Pinned<int>(*storage, CudaLiquidRange3D::kArchitecture);
  if (metrics.compiled_architecture != 1200) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  return storage.release();
}
// Reserve the ticket before an upload: even rejected preparations cannot reuse
// a generation, while UINT64_MAX stays the exhausted sentinel.
absl::Status
StageState(LiquidArenaStorage3D& storage, std::size_t bank, const LiquidState3D& state, LiquidClock3D clock) {
  if (storage.leases[bank] != 0) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  if (!Fits(Active(state), storage.capacities)) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  auto status = ValidateLiquidState3D(state);
  if (!status.ok()) {
    return status;
  }
  LiquidState3D clock_check;
  clock_check.clock = clock;
  status = ValidateLiquidState3D(clock_check);
  if (!status.ok()) {
    return status;
  }
  const auto generation = cuda_internal::TakeLiquidArenaGeneration3D(*storage.ledger);
  if (!generation.ok()) {
    return generation.status();
  }
  const auto offset = bank * storage.state_layout.total_state_bytes;
  cuda_internal::PackLiquidArenaState3D(state, clock, storage.state_layout, storage.pinned + offset);
  status = CheckedCuda(cudaMemcpyAsync(
      storage.device + offset,
      storage.pinned + offset,
      storage.state_layout.total_state_bytes,
      cudaMemcpyHostToDevice,
      storage.stream
  ));
  if (!status.ok()) {
    return status;
  }
  status = Fence(storage);
  if (!status.ok()) {
    return status;
  }
  storage.identities[bank] = {*generation, clock};
  storage.active[bank] = Active(state);
  return absl::OkStatus();
}
}  // namespace
namespace cuda_internal {
absl::StatusOr<std::uint64_t> TakeLiquidArenaGeneration3D(LiquidArenaLedger3D& ledger) {
  if (ledger.next_generation == UINT64_MAX) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  return ledger.next_generation++;
}
CudaLiquidRangeView3D LiquidArenaStorage3D::Range(CudaLiquidRange3D name) const {
  const auto ordinal = static_cast<std::size_t>(name);
  if (ordinal >= ranges.size()) {
    return {};
  }
  const auto range = ranges[ordinal];
  return {
      name == CudaLiquidRange3D::kPressurePreparationScratch ? nullptr : device + range.offset,
      {pinned + range.offset, range.bytes},
  };
}
LiquidArenaStorage3D::~LiquidArenaStorage3D() {
  if (!ledger) {
    return;
  }
  auto& metrics = ledger->metrics;
  int previous_device = -1;
  (void)cudaGetDevice(&previous_device);
  (void)cudaSetDevice(ledger->device_ordinal);
  if (stream) {
    (void)cudaStreamSynchronize(stream);
  }
  if (executable) {
    ++metrics.executable_destructions;
    (void)cudaGraphExecDestroy(executable);
  }
  if (graph) {
    ++metrics.graph_destructions;
    (void)cudaGraphDestroy(graph);
  }
  if (completion) {
    ++metrics.event_destructions;
    (void)cudaEventDestroy(completion);
  }
  if (stream) {
    ++metrics.stream_destructions;
    (void)cudaStreamDestroy(stream);
  }
  if (device) {
    ++metrics.device_frees;
    (void)cudaFree(device);
    metrics.live_device_bytes -= reserved_device_bytes;
  }
  if (pinned) {
    ++metrics.pinned_frees;
    (void)cudaFreeHost(pinned);
    metrics.live_pinned_bytes -= reserved_pinned_bytes;
  }
  if (previous_device >= 0 && previous_device != ledger->device_ordinal) {
    (void)cudaSetDevice(previous_device);
  }
  if (--ledger->references == 0) {
    delete ledger;
  }
}
void ReleaseLiquidArenaStorage3D(LiquidArenaStorage3D* storage) noexcept {
  if (storage && --storage->references == 0) {
    delete storage;
  }
}
}  // namespace cuda_internal
struct CudaLiquidArena3D::Impl {
  LiquidArenaLedger3D* ledger = nullptr;
  LiquidArenaStorage3D* current = nullptr;
  LiquidArenaStorage3D* growth = nullptr;
  LiquidArenaStorage3D* retired = nullptr;
  std::size_t committed_bank = 0;
  bool initialized = false;
  bool prepared = false;
  ~Impl() {
    ReleaseLiquidArenaStorage3D(growth);
    ReleaseLiquidArenaStorage3D(current);
    while (retired) {
      auto* next = retired->retired_next;
      retired->retired_next = nullptr;
      ReleaseLiquidArenaStorage3D(retired);
      retired = next;
    }
    if (ledger && --ledger->references == 0) {
      delete ledger;
    }
  }
};
CudaLiquidArena3D::CudaLiquidArena3D() = default;
CudaLiquidArena3D::CudaLiquidArena3D(CudaLiquidArena3D&&) noexcept = default;
CudaLiquidArena3D& CudaLiquidArena3D::operator=(CudaLiquidArena3D&&) noexcept = default;
CudaLiquidArena3D::~CudaLiquidArena3D() = default;
absl::StatusOr<CudaLiquidArena3D>
CudaLiquidArena3D::Create(LiquidStateCapacities3D capacities, int device_ordinal, std::size_t byte_limit) {
  if (device_ordinal < 0) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  int count = 0;
  auto status = CheckedCuda(cudaGetDeviceCount(&count));
  if (!status.ok()) {
    return status;
  }
  if (device_ordinal >= count) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  status = CheckedCuda(cudaSetDevice(device_ordinal));
  if (!status.ok()) {
    return status;
  }
  cudaDeviceProp properties{};
  status = CheckedCuda(cudaGetDeviceProperties(&properties, device_ordinal));
  if (!status.ok()) {
    return status;
  }
  if (properties.major != 12 || properties.minor != 0) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  CudaLiquidArena3D arena;
  arena.impl_.reset(new (std::nothrow) Impl{});
  if (!arena.impl_) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  arena.impl_->ledger = new (std::nothrow) LiquidArenaLedger3D{};
  if (!arena.impl_->ledger) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  auto& ledger = *arena.impl_->ledger;
  ledger.device_ordinal = device_ordinal;
  auto& metrics = ledger.metrics;
  status = CheckedCuda(cudaMemGetInfo(&metrics.free_device_bytes_at_start, &metrics.total_device_bytes));
  if (!status.ok()) {
    return status;
  }
  // Integer floors avoid conversion overflow/round-up at the resource ceiling.
  const auto total_limit = metrics.total_device_bytes / 10 * 7 + metrics.total_device_bytes % 10 * 7 / 10;
  const auto free_limit = metrics.free_device_bytes_at_start / 5 * 4 + metrics.free_device_bytes_at_start % 5 * 4 / 5;
  metrics.device_byte_limit = std::min({total_limit, free_limit, byte_limit});
  metrics.cccl_version = cuda_internal::LiquidArenaCcclVersion3D();
  metrics.cub_version = cuda_internal::LiquidArenaCubVersion3D();
  auto storage = BuildStorage(ledger, capacities);
  if (!storage.ok()) {
    return storage.status();
  }
  arena.impl_->current = *storage;
  return arena;
}
absl::Status CudaLiquidArena3D::Initialize(const LiquidState3D& initial) {
  if (impl_->initialized || impl_->growth) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  auto status = StageState(*impl_->current, 0, initial, initial.clock);
  if (status.ok()) {
    impl_->initialized = true;
  }
  return status;
}
absl::Status CudaLiquidArena3D::PreparePublication(const LiquidState3D& trial, LiquidClock3D accepted_clock) {
  if (!impl_->initialized || impl_->prepared || impl_->growth) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  const auto& prior = impl_->current->identities[impl_->committed_bank].clock;
  if (trial.clock != prior) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (prior.epoch == UINT64_MAX || prior.accepted_step_ordinal == UINT64_MAX || prior.substep_ordinal == UINT64_MAX) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  const bool interior = accepted_clock.frame_ordinal == prior.frame_ordinal
      && accepted_clock.substep_ordinal == prior.substep_ordinal + 1;
  const bool completed_frame = prior.frame_ordinal != UINT64_MAX
      && accepted_clock.frame_ordinal == prior.frame_ordinal + 1 && accepted_clock.substep_ordinal == 0;
  if (!(accepted_clock.time > prior.time) || accepted_clock.epoch != prior.epoch + 1
      || accepted_clock.accepted_step_ordinal != prior.accepted_step_ordinal + 1 || (!interior && !completed_frame)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const auto offset = impl_->committed_bank * impl_->current->state_layout.total_state_bytes
      + impl_->current->state_layout.control_record_byte_offset;
  const auto& control = *reinterpret_cast<const LiquidArenaControl3D*>(impl_->current->pinned + offset);
  if (!SameGeometry(control.geometry, trial.geometry) || control.random_seed != trial.random_seed
      || trial.next_particle_id < control.next_particle_id) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  auto status = StageState(*impl_->current, 1 - impl_->committed_bank, trial, accepted_clock);
  if (status.ok()) {
    impl_->prepared = true;
  }
  return status;
}
void CudaLiquidArena3D::CommitPublication() noexcept {
  if (impl_->prepared) {
    impl_->committed_bank = 1 - impl_->committed_bank;
    impl_->prepared = false;
  }
}
void CudaLiquidArena3D::DiscardPublication() noexcept {
  impl_->prepared = false;
}
absl::Status CudaLiquidArena3D::PrepareGrowth(LiquidStateCapacities3D capacities, const LiquidState3D& frame_start) {
  if (!impl_->initialized || impl_->prepared || impl_->growth) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  if (capacities == impl_->current->capacities || !Fits(impl_->current->capacities, capacities)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  auto status = ValidateLiquidState3D(frame_start);
  if (!status.ok()) {
    return status;
  }
  if (!Fits(Active(frame_start), capacities)) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  auto storage = BuildStorage(*impl_->ledger, capacities);
  if (!storage.ok()) {
    return storage.status();
  }
  std::unique_ptr<LiquidArenaStorage3D> staged(*storage);
  status = StageState(*staged, 0, frame_start, frame_start.clock);
  if (!status.ok()) {
    return status;
  }
  impl_->growth = staged.release();
  return absl::OkStatus();
}
void CudaLiquidArena3D::CommitGrowth() noexcept {
  if (!impl_->growth) {
    return;
  }
  impl_->current->retired_next = impl_->retired;
  impl_->retired = impl_->current;
  impl_->current = std::exchange(impl_->growth, nullptr);
  impl_->committed_bank = 0;
}
void CudaLiquidArena3D::DiscardGrowth() noexcept {
  ReleaseLiquidArenaStorage3D(std::exchange(impl_->growth, nullptr));
}
void CudaLiquidArena3D::CollectRetired() noexcept {
  auto** link = &impl_->retired;
  while (*link) {
    auto* storage = *link;
    if (storage->references == 1) {
      *link = storage->retired_next;
      storage->retired_next = nullptr;
      ReleaseLiquidArenaStorage3D(storage);
    } else {
      link = &storage->retired_next;
    }
  }
}
absl::StatusOr<CudaLiquidEpochLease3D> CudaLiquidArena3D::AcquireCommitted() {
  if (!impl_->initialized) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  auto* storage = impl_->current;
  if (storage->references == SIZE_MAX || storage->leases[impl_->committed_bank] == SIZE_MAX) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  CudaLiquidEpochLease3D lease;
  lease.storage_ = storage;
  lease.bank_ = impl_->committed_bank;
  lease.identity_ = storage->identities[lease.bank_];
  ++storage->references;
  ++storage->leases[lease.bank_];
  return lease;
}
absl::Status CudaLiquidArena3D::PrepareTopology(
    std::span<const LiquidParticleState3D> particles,
    std::span<const CellIndex3D> pressure_cells,
    LiquidStepWorkspace3D& workspace,
    const LiquidState3D* previous_state
) {
  if (!impl_->initialized || impl_->growth) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  auto& storage = *impl_->current;
  if (particles.size() > storage.capacities.particles) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  const auto offset =
      impl_->committed_bank * storage.state_layout.total_state_bytes + storage.state_layout.control_record_byte_offset;
  const auto& control = *reinterpret_cast<const LiquidArenaControl3D*>(storage.pinned + offset);
  auto* keys = Pinned<LiquidSortKey3D>(storage, CudaLiquidRange3D::kSortInputKeys);
  auto* ordinals = Pinned<std::size_t>(storage, CudaLiquidRange3D::kSortInputOrdinals);
  for (std::size_t ordinal = 0; ordinal < storage.sort_capacity; ++ordinal) {
    keys[ordinal] = {INT32_MAX, INT32_MAX, INT32_MAX};
    ordinals[ordinal] = ordinal;
  }
  for (std::size_t ordinal = 0; ordinal < particles.size(); ++ordinal) {
    const auto bin = ComputeParticleSortBin3D(control.geometry, particles[ordinal].world_position);
    if (!bin.ok()) {
      return bin.status();
    }
    keys[ordinal] = {bin->x, bin->y, bin->z};
  }
  for (const auto name : {CudaLiquidRange3D::kSortInputKeys, CudaLiquidRange3D::kSortInputOrdinals}) {
    auto status = TransferRange(storage, name, true);
    if (!status.ok()) {
      return status;
    }
  }
  auto status = CheckedCuda(cudaGraphLaunch(storage.executable, storage.stream));
  if (!status.ok()) {
    return status;
  }
  ++storage.ledger->metrics.sort_replays;
  for (const auto name :
       {CudaLiquidRange3D::kSortOutputKeys, CudaLiquidRange3D::kSortOutputOrdinals, CudaLiquidRange3D::kArchitecture}) {
    status = TransferRange(storage, name, false);
    if (!status.ok()) {
      return status;
    }
  }
  status = Fence(storage);
  if (!status.ok()) {
    return status;
  }
  if (*Pinned<int>(storage, CudaLiquidRange3D::kArchitecture) != 1200) {
    return Error(absl::StatusCode::kInternal);
  }
  auto* entries = Pinned<LiquidParticleSortEntry3D>(storage, CudaLiquidRange3D::kSortEntries);
  const auto* sorted_keys = Pinned<LiquidSortKey3D>(storage, CudaLiquidRange3D::kSortOutputKeys);
  const auto* sorted_ordinals = Pinned<std::size_t>(storage, CudaLiquidRange3D::kSortOutputOrdinals);
  for (std::size_t ordinal = 0; ordinal < particles.size(); ++ordinal) {
    entries[ordinal] = {
        {sorted_keys[ordinal].x, sorted_keys[ordinal].y, sorted_keys[ordinal].z},
        sorted_ordinals[ordinal],
    };
  }
  return workspace
      .PrepareTrialTopologyFromOrder(particles, pressure_cells, {entries, particles.size()}, previous_state);
}
CudaLiquidRangeView3D CudaLiquidArena3D::range(CudaLiquidRange3D name) const {
  return impl_->current->Range(name);
}
LiquidStateCapacities3D CudaLiquidArena3D::capacities() const {
  return impl_->current->capacities;
}
CudaLiquidArenaMetrics3D CudaLiquidArena3D::metrics() const {
  return impl_->ledger->metrics;
}
void* CudaLiquidArena3D::stream() const {
  return impl_->current->stream;
}
void* CudaLiquidArena3D::completion_event() const {
  return impl_->current->completion;
}
absl::Status CudaLiquidArena3D::Synchronize() const {
  return CheckedCuda(cudaStreamSynchronize(impl_->current->stream));
}
CudaLiquidEpochLease3D::CudaLiquidEpochLease3D(CudaLiquidEpochLease3D&& other) noexcept
    : storage_(std::exchange(other.storage_, nullptr)), bank_(other.bank_), identity_(other.identity_) {
}
CudaLiquidEpochLease3D& CudaLiquidEpochLease3D::operator=(CudaLiquidEpochLease3D&& other) noexcept {
  if (this != &other) {
    if (storage_) {
      --storage_->leases[bank_];
      ReleaseLiquidArenaStorage3D(storage_);
    }
    storage_ = std::exchange(other.storage_, nullptr);
    bank_ = other.bank_;
    identity_ = other.identity_;
  }
  return *this;
}
CudaLiquidEpochLease3D::~CudaLiquidEpochLease3D() {
  if (storage_) {
    --storage_->leases[bank_];
    ReleaseLiquidArenaStorage3D(storage_);
  }
}
CudaLiquidEpochIdentity3D CudaLiquidEpochLease3D::identity() const {
  return identity_;
}
std::span<const std::byte> CudaLiquidEpochLease3D::device_bytes() const {
  if (!storage_) {
    return {};
  }
  return {
      storage_->device + bank_ * storage_->state_layout.total_state_bytes,
      storage_->state_layout.total_state_bytes,
  };
}
CudaLiquidStateLayout3D CudaLiquidEpochLease3D::layout() const {
  return storage_ ? storage_->state_layout : CudaLiquidStateLayout3D{};
}
CudaLiquidArenaMetrics3D CudaLiquidEpochLease3D::metrics() const {
  return storage_ ? storage_->ledger->metrics : CudaLiquidArenaMetrics3D{};
}
absl::Status CudaLiquidEpochLease3D::Readback(LiquidState3D& output) const {
  if (!storage_) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  const LiquidStateCapacities3D capacity{
      output.particles.capacity(),
      output.blocks.capacity(),
      output.cells.capacity(),
      output.faces.capacity(),
      output.sources.capacity(),
  };
  if (!Fits(storage_->active[bank_], capacity)) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  const auto offset = bank_ * storage_->state_layout.total_state_bytes;
  auto status = CheckedCuda(cudaMemcpyAsync(
      storage_->pinned + offset,
      storage_->device + offset,
      storage_->state_layout.total_state_bytes,
      cudaMemcpyDeviceToHost,
      storage_->stream
  ));
  if (!status.ok()) {
    return status;
  }
  status = Fence(*storage_);
  if (!status.ok()) {
    return status;
  }
  const auto& control = *reinterpret_cast<const LiquidArenaControl3D*>(
      storage_->pinned + offset + storage_->state_layout.control_record_byte_offset
  );
  if (control.active != storage_->active[bank_] || control.clock != identity_.clock) {
    return Error(absl::StatusCode::kDataLoss);
  }
  cuda_internal::UnpackLiquidArenaState3D(storage_->state_layout, storage_->pinned + offset, output);
  return absl::OkStatus();
}
}  // namespace fluidity
