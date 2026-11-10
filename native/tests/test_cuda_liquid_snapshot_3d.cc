#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include "../cuda/liquid_arena_3d_internal.h"
#include "fluidity/host/cuda_liquid_snapshot_3d.h"
#include "gtest/gtest.h"
namespace fluidity {
namespace {
constexpr LiquidStateCapacities3D kSmall{33, 1, 1, 192, 2};
LiquidState3D Fixture() {
  auto state = CloneLiquidState3D(LiquidState3D{}, kSmall);
  EXPECT_TRUE(state.ok());
  if (!state.ok()) {
    return {};
  }
  EXPECT_TRUE(state->particles.Resize(33).ok());
  EXPECT_TRUE(state->blocks.Resize(1).ok());
  EXPECT_TRUE(state->cells.Resize(1).ok());
  EXPECT_TRUE(state->faces.Resize(1).ok());
  EXPECT_TRUE(state->sources.Resize(2).ok());
  state->next_particle_id = 100;
  state->random_seed = 0xfedcba9876543210ULL;
  for (std::size_t index = 0; index < 33; ++index) {
    auto& particle = state->particles[index];
    particle.id = 99 - index;
    particle.tag = static_cast<std::uint32_t>(0x12340000 + index);
    particle.world_position = {2, 2, 2};
    particle.world_velocity = {static_cast<double>(index), -0.125, 1.25};
    particle.velocity_slopes = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    particle.mass = 2.5;
    particle.reference_volume = 0.0025;
    particle.volume_ratio = 1.125;
    particle.side_tag = -17;
    particle.affine_enabled = index % 2 == 0;
    particle.pic_initialized = index % 3 == 0;
  }
  state->blocks[0].block_index = {-2, 1, -3};
  state->cells[0].cell_index = {-7, 6, -10};
  state->cells[0].occupancy = LiquidCellOccupancy3D::kSolid;
  state->faces[0].face_index = {FaceNormalAxis::kY, {-7, 6, -10}};
  state->faces[0].face_slot = 105;
  state->blocks[0].generation = 19;
  state->blocks[0].activation_age = 23;
  state->blocks[0].inactive_age = 2;
  state->cells[0].previous_pressure = -7.25;
  state->faces[0].carrier_velocity = 4.125;
  state->faces[0].validity = LiquidFaceValidity3D::kValid;
  state->faces[0].side_tag = -17;
  state->faces[0].structurally_eligible = false;
  state->sources[0] = {3, 8, 0.25};
  state->sources[1] = {7, 11, 0.75};
  EXPECT_TRUE(ValidateLiquidState3D(*state).ok());
  return std::move(*state);
}

thread_local cudaEvent_t copy_gate = nullptr;
cudaError_t
GatedCopy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  if (kind == cudaMemcpyDeviceToHost && copy_gate) {
    const auto result = cudaStreamWaitEvent(stream, copy_gate, 0);
    if (result != cudaSuccess) {
      return result;
    }
  }
  return cudaMemcpyAsync(destination, source, bytes, kind, stream);
}
class CopyGate {
 public:
  CopyGate() {
    EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
    EXPECT_EQ(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), cudaSuccess);
    // Record AFTER the queued latch. Waiting on an unrecorded event would not gate a copy.
    EXPECT_EQ(
        cudaLaunchHostFunc(
            stream_,
            [](void* context) {
              auto& released = *static_cast<std::atomic<bool>*>(context);
              while (!released.load(std::memory_order_acquire)) {
                std::this_thread::yield();
              }
            },
            &released_
        ),
        cudaSuccess
    );
    EXPECT_EQ(cudaEventRecord(event_, stream_), cudaSuccess);
    copy_gate = event_;
    cuda_internal::LiquidArenaRuntimeCalls().memcpy_async = GatedCopy;
  }
  void Open() {
    released_.store(true, std::memory_order_release);
  }
  ~CopyGate() {
    Open();
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    copy_gate = nullptr;
    cuda_internal::LiquidArenaRuntimeCalls().memcpy_async = cudaMemcpyAsync;
    EXPECT_EQ(cudaEventDestroy(event_), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }

 private:
  std::atomic<bool> released_{false};
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
};
absl::StatusOr<LiquidSnapshotProgress3D> Await(CudaLiquidSnapshotQueue3D& queue, LiquidSnapshotTicket3D ticket) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (;;) {
    auto result = queue.Poll(ticket);
    if (!result.ok() || *result != LiquidSnapshotProgress3D::kPending) {
      return result;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return absl::DeadlineExceededError("");
    }
    std::this_thread::yield();
  }
}
void Advance(CudaLiquidArena3D& arena, LiquidState3D& state) {
  auto clock = state.clock;
  clock.time += 0.01;
  ++clock.frame_ordinal;
  ++clock.accepted_step_ordinal;
  ++clock.epoch;
  ASSERT_TRUE(arena.PreparePublication(state, clock).ok());
  arena.CommitPublication();
  state.clock = clock;
}
void Receipt(const char* name, const CudaLiquidArena3D& arena, float milliseconds) {
  cudaDeviceProp device{};
  ASSERT_EQ(cudaGetDeviceProperties(&device, 0), cudaSuccess);
  EXPECT_STREQ(device.name, "NVIDIA GeForce RTX 5060 Ti");
  EXPECT_EQ(device.major, 12);
  EXPECT_EQ(device.minor, 0);
  EXPECT_EQ(arena.metrics().compiled_architecture, 1200);
  EXPECT_GT(milliseconds, 0);
  std::cout << "operator=" << name << " compiled_architecture=" << arena.metrics().compiled_architecture
            << " snapshot_copy_ms=" << std::fixed << std::setprecision(9) << milliseconds
            << " timing_scope=d2h_stream_events\n";
}
#if defined(_WIN32) && defined(_DEBUG)
std::size_t snapshot_allocations = 0;
std::size_t snapshot_frees = 0;
_CRT_ALLOC_HOOK previous_snapshot_hook = nullptr;
int SnapshotAllocationHook(
    int operation,
    void* data,
    std::size_t size,
    int type,
    long request,
    const unsigned char* filename,
    int line
) {
  if (operation == _HOOK_ALLOC || operation == _HOOK_REALLOC) {
    ++snapshot_allocations;
  }
  if (operation == _HOOK_FREE) {
    ++snapshot_frees;
  }
  return previous_snapshot_hook ? previous_snapshot_hook(operation, data, size, type, request, filename, line) : 1;
}
void CheckNoSubmissionAllocations() {
  auto state = Fixture();
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(arena->Initialize(state).ok());
  auto source = arena->AcquireCommitted();
  auto duplicate = arena->AcquireCommitted();
  ASSERT_TRUE(source.ok());
  ASSERT_TRUE(duplicate.ok());
  auto queue = CudaLiquidSnapshotQueue3D::Create(*source);
  ASSERT_TRUE(queue.ok());
  const auto before = arena->metrics();
  snapshot_allocations = 0;
  snapshot_frees = 0;
  bool completed = false;
  bool duplicate_rejected = false;
  LiquidSnapshotLease3D writer;
  previous_snapshot_hook = _CrtSetAllocHook(SnapshotAllocationHook);
  {
    auto ticket = queue->BeginCopy(source->identity(), std::move(*source));
    auto rejected = queue->BeginCopy(duplicate->identity(), std::move(*duplicate));
    duplicate_rejected = rejected.status().code() == absl::StatusCode::kAlreadyExists;
    if (ticket.ok()) {
      for (int query = 0; query < 100000; ++query) {
        auto progress = queue->Poll(*ticket);
        if (!progress.ok()) {
          break;
        }
        if (*progress == LiquidSnapshotProgress3D::kReady) {
          auto snapshot = queue->Finish(*ticket);
          completed = snapshot.ok();
          if (completed) {
            writer = std::move(*snapshot);
          }
          break;
        }
      }
    }
  }
  _CrtSetAllocHook(previous_snapshot_hook);
  EXPECT_TRUE(completed);
  EXPECT_TRUE(duplicate_rejected);
  EXPECT_EQ(snapshot_allocations, 0);
  EXPECT_EQ(snapshot_frees, 0);
  EXPECT_EQ(arena->metrics().device_reservations, before.device_reservations);
  EXPECT_EQ(arena->metrics().pinned_reservations, before.pinned_reservations);
  EXPECT_EQ(arena->metrics().event_creations, before.event_creations);
  EXPECT_EQ(arena->metrics().stream_creations, before.stream_creations);
}
#endif
void CheckStaleAndWriterLifetimes() {
  auto state = Fixture();
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(arena->Initialize(state).ok());
  auto first = arena->AcquireCommitted();
  ASSERT_TRUE(first.ok());
  auto queue = CudaLiquidSnapshotQueue3D::Create(*first);
  ASSERT_TRUE(queue.ok());
  auto wrong = first->identity();
  ++wrong.generation;
  EXPECT_EQ(queue->BeginCopy(wrong, std::move(*first)).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(first->device_bytes().empty());
  auto first_ticket = queue->BeginCopy(first->identity(), std::move(*first));
  ASSERT_TRUE(first_ticket.ok());
  state.particles[0].tag = 99;
  Advance(*arena, state);
  auto second = arena->AcquireCommitted();
  ASSERT_TRUE(second.ok());
  auto second_ticket = queue->BeginCopy(second->identity(), std::move(*second));
  ASSERT_TRUE(second_ticket.ok());
  ASSERT_EQ(*Await(*queue, *second_ticket), LiquidSnapshotProgress3D::kReady);
  auto newer = queue->Finish(*second_ticket);
  ASSERT_TRUE(newer.ok());
  ASSERT_EQ(*Await(*queue, *first_ticket), LiquidSnapshotProgress3D::kReady);
  EXPECT_FALSE(queue->Finish(*first_ticket).ok());
  EXPECT_EQ(*queue->LastValidHostSnapshot(), state);
  ASSERT_TRUE(queue->Cancel(*first_ticket).ok());
  Advance(*arena, state);
  auto third = arena->AcquireCommitted();
  ASSERT_TRUE(third.ok());
  auto third_ticket = queue->BeginCopy(third->identity(), std::move(*third));
  ASSERT_TRUE(third_ticket.ok());
  EXPECT_EQ(first_ticket->slot, third_ticket->slot);
  EXPECT_NE(first_ticket->reservation, third_ticket->reservation);
  EXPECT_FALSE(queue->Poll(*first_ticket).ok());
  ASSERT_EQ(*Await(*queue, *third_ticket), LiquidSnapshotProgress3D::kReady);
  auto newest = queue->Finish(*third_ticket);
  ASSERT_TRUE(newest.ok());
  Advance(*arena, state);
  auto fourth = arena->AcquireCommitted();
  ASSERT_TRUE(fourth.ok());
  EXPECT_EQ(queue->BeginCopy(fourth->identity(), std::move(*fourth)).status().code(), absl::StatusCode::kUnavailable);
  EXPECT_FALSE(fourth->device_bytes().empty());
  EXPECT_EQ(newer->state()->clock.frame_ordinal, 1);
  EXPECT_EQ(newest->state()->clock.frame_ordinal, 2);
  // Queue destruction cannot free or mutate destinations retained by the writer.
  const auto retained_pinned_bytes = 2 * fourth->layout().total_state_bytes;
  auto replacement = CudaLiquidSnapshotQueue3D::Create(*fourth);
  ASSERT_TRUE(replacement.ok());
  *queue = std::move(*replacement);
  const auto before_writer_release = arena->metrics();
  EXPECT_EQ(newer->state()->clock.frame_ordinal, 1);
  EXPECT_EQ(newest->state()->clock.frame_ordinal, 2);
  *newer = LiquidSnapshotLease3D{};
  *newest = LiquidSnapshotLease3D{};
  EXPECT_EQ(arena->metrics().live_pinned_bytes, before_writer_release.live_pinned_bytes - retained_pinned_bytes);
  EXPECT_EQ(arena->metrics().pinned_frees, before_writer_release.pinned_frees + 2);
  EXPECT_EQ(arena->metrics().stream_destructions, before_writer_release.stream_destructions + 2);
  EXPECT_EQ(arena->metrics().event_destructions, before_writer_release.event_destructions + 4);
  // An interior substep must never become a frame snapshot.
  fourth = CudaLiquidEpochLease3D{};
  auto interior = state.clock;
  interior.time += 0.001;
  ++interior.substep_ordinal;
  ++interior.accepted_step_ordinal;
  ++interior.epoch;
  ASSERT_TRUE(arena->PreparePublication(state, interior).ok());
  arena->CommitPublication();
  auto substep = arena->AcquireCommitted();
  ASSERT_TRUE(substep.ok());
  EXPECT_EQ(
      queue->BeginCopy(substep->identity(), std::move(*substep)).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  substep = CudaLiquidEpochLease3D{};
  auto grown = kSmall;
  grown.particles = 65;
  ASSERT_TRUE(arena->PrepareGrowth(grown, state).ok());
  arena->CommitGrowth();
  auto larger = arena->AcquireCommitted();
  ASSERT_TRUE(larger.ok());
  EXPECT_EQ(
      queue->BeginCopy(larger->identity(), std::move(*larger)).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_FALSE(larger->device_bytes().empty());
}
TEST(CudaLiquidSnapshot3D, DelayedCopyFencesSourceEpoch) {
  auto state = Fixture();
  auto arena = CudaLiquidArena3D::Create(kSmall, 0, std::numeric_limits<std::size_t>::max(), true);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(arena->Initialize(state).ok());
  auto first = arena->AcquireCommitted();
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(first->metrics().compiled_architecture, 1200);
  auto queue = CudaLiquidSnapshotQueue3D::Create(*first);
  ASSERT_TRUE(queue.ok()) << queue.status();
  auto original = CloneLiquidState3D(state, kSmall);
  ASSERT_TRUE(original.ok());
  state.particles[0].world_velocity.x = 999;
  Advance(*arena, state);
  auto second = arena->AcquireCommitted();
  ASSERT_TRUE(second.ok());
  // Both epochs are prepared before the latch: WDDM may serialize a later
  // synchronous publication upload behind a pending host callback.
  CopyGate gate;
  auto ticket = queue->BeginCopy(first->identity(), std::move(*first));
  ASSERT_TRUE(ticket.ok());
  EXPECT_EQ(*queue->Poll(*ticket), LiquidSnapshotProgress3D::kPending);
  const auto single_output_bytes = arena->metrics().live_snapshot_output_bytes;
  EXPECT_GT(single_output_bytes, 0);
  auto second_ticket = queue->BeginCopy(second->identity(), std::move(*second));
  ASSERT_TRUE(second_ticket.ok());
  EXPECT_EQ(*queue->Poll(*second_ticket), LiquidSnapshotProgress3D::kPending);
  EXPECT_EQ(arena->metrics().live_snapshot_output_bytes, 2 * single_output_bytes);
  EXPECT_EQ(arena->metrics().snapshot_d2h_events.interval_count, 0);
  auto next_clock = state.clock;
  next_clock.time += 0.01;
  ++next_clock.frame_ordinal;
  ++next_clock.accepted_step_ordinal;
  ++next_clock.epoch;
  ASSERT_TRUE(arena->PublicationReady().ok());
  EXPECT_FALSE(*arena->PublicationReady());
  EXPECT_FALSE(arena->PreparePublication(state, next_clock).ok());
  EXPECT_EQ(queue->LastValidHostSnapshot(), nullptr);
  EXPECT_TRUE(queue->Cancel(*second_ticket).ok());
  EXPECT_EQ(arena->metrics().live_snapshot_output_bytes, 2 * single_output_bytes);
  gate.Open();
  ASSERT_EQ(*Await(*queue, *ticket), LiquidSnapshotProgress3D::kReady);
  EXPECT_TRUE(*arena->PublicationReady());
  auto snapshot = queue->Finish(*ticket);
  ASSERT_TRUE(snapshot.ok());
  ASSERT_NE(snapshot->state(), nullptr);
  EXPECT_EQ(*snapshot->state(), *original);
  ASSERT_NE(queue->LastValidHostSnapshot(), nullptr);
  EXPECT_EQ(*queue->LastValidHostSnapshot(), *original);
  EXPECT_EQ(*Await(*queue, *second_ticket), LiquidSnapshotProgress3D::kCanceled);
  const auto completed = arena->metrics();
  EXPECT_EQ(completed.live_snapshot_output_bytes, single_output_bytes);
  EXPECT_EQ(completed.peak_snapshot_output_bytes, 2 * single_output_bytes);
  EXPECT_EQ(completed.snapshot_d2h_events.interval_count, 2);
  EXPECT_GT(completed.snapshot_d2h_events.seconds, snapshot->copy_milliseconds() / 1000.0);
  std::cout << "delayed_snapshot_output_peak_bytes=" << completed.peak_snapshot_output_bytes
            << " completed_d2h_intervals=" << completed.snapshot_d2h_events.interval_count
            << " summed_d2h_seconds=" << completed.snapshot_d2h_events.seconds << '\n';
  EXPECT_FALSE(queue->Finish(*second_ticket).ok());
  EXPECT_FALSE(queue->Poll(*second_ticket).ok());
  Advance(*arena, state);
  EXPECT_EQ(*snapshot->state(), *original);
  EXPECT_FALSE(queue->Finish(*ticket).ok());
  Receipt("liquid_snapshot_3d", *arena, snapshot->copy_milliseconds());
  *snapshot = LiquidSnapshotLease3D{};
  EXPECT_EQ(arena->metrics().live_snapshot_output_bytes, 0);
  ASSERT_NO_FATAL_FAILURE(CheckStaleAndWriterLifetimes());
#if defined(_WIN32) && defined(_DEBUG)
  ASSERT_NO_FATAL_FAILURE(CheckNoSubmissionAllocations());
#endif
}
cudaError_t FailCompletedEvent(cudaEvent_t event) {
  const auto result = cudaEventQuery(event);
  return result == cudaSuccess ? cudaErrorUnknown : result;
}
enum class SnapshotFault {
  kCopy,
  kFence,
  kCount,
  kClock,
  kFlags,
  kBoolean,
  kNan,
  kDrain
};
thread_local SnapshotFault snapshot_fault;
thread_local std::byte* copied_destination = nullptr;
thread_local CudaLiquidStateLayout3D copied_layout;
thread_local int copy_calls = 0;
thread_local int event_records = 0;
cudaError_t
ControlledCopy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  const auto result = cudaMemcpyAsync(destination, source, bytes, kind, stream);
  if (kind != cudaMemcpyDeviceToHost || result != cudaSuccess) {
    return result;
  }
  ++copy_calls;
  copied_destination = static_cast<std::byte*>(destination);
  return snapshot_fault == SnapshotFault::kCopy ? cudaErrorUnknown : result;
}
cudaError_t ControlledRecord(cudaEvent_t event, cudaStream_t stream) {
  const auto result = cudaEventRecord(event, stream);
  ++event_records;
  return result == cudaSuccess && snapshot_fault == SnapshotFault::kFence && event_records == 2 ? cudaErrorUnknown
                                                                                                : result;
}
cudaError_t CorruptCompletedCopy(cudaEvent_t event) {
  const auto result = cudaEventQuery(event);
  if (result != cudaSuccess || !copied_destination) {
    return result;
  }
  auto& control = *reinterpret_cast<cuda_internal::LiquidArenaControl3D*>(
      copied_destination + copied_layout.control_record_byte_offset
  );
  auto* bins = reinterpret_cast<CudaLiquidParticleBin3D*>(copied_destination + copied_layout.particle_bins_byte_offset);
  switch (snapshot_fault) {
    case SnapshotFault::kCount:
      control.active.particles = SIZE_MAX;
      break;
    case SnapshotFault::kClock:
      ++control.clock.epoch;
      break;
    case SnapshotFault::kFlags:
      bins[0].affine_pic_flags[0] = 7;
      break;
    case SnapshotFault::kBoolean:
      copied_destination[copied_layout.face_records_byte_offset + offsetof(LiquidFaceState3D, structurally_eligible)] =
          std::byte{7};
      break;
    case SnapshotFault::kNan:
      bins[0].particle_masses[0] = std::numeric_limits<double>::quiet_NaN();
      break;
    default:
      break;
  }
  return result;
}
cudaError_t UnprovenDrain(cudaStream_t stream) {
  const auto result = cudaStreamSynchronize(stream);
  return result == cudaSuccess ? cudaErrorUnknown : result;
}
class RuntimeControls {
 public:
  RuntimeControls() : saved_(cuda_internal::LiquidArenaRuntimeCalls()) {
  }
  ~RuntimeControls() {
    cuda_internal::LiquidArenaRuntimeCalls() = saved_;
  }

 private:
  cuda_internal::LiquidArenaRuntimeCalls3D saved_;
};
void CheckSnapshotFault(SnapshotFault fault) {
  SCOPED_TRACE(static_cast<int>(fault));
  auto state = Fixture();
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(arena->Initialize(state).ok());
  auto source = arena->AcquireCommitted();
  ASSERT_TRUE(source.ok());
  auto queue = CudaLiquidSnapshotQueue3D::Create(*source);
  ASSERT_TRUE(queue.ok());
  auto ticket = queue->BeginCopy(source->identity(), std::move(*source));
  ASSERT_TRUE(ticket.ok());
  ASSERT_EQ(*Await(*queue, *ticket), LiquidSnapshotProgress3D::kReady);
  auto valid = queue->Finish(*ticket);
  ASSERT_TRUE(valid.ok());
  auto original = CloneLiquidState3D(state, kSmall);
  ASSERT_TRUE(original.ok());
  Advance(*arena, state);
  source = arena->AcquireCommitted();
  ASSERT_TRUE(source.ok());
  const auto before = arena->metrics();
  RuntimeControls controls;
  snapshot_fault = fault;
  copied_layout = source->layout();
  copy_calls = 0;
  event_records = 0;
  auto& calls = cuda_internal::LiquidArenaRuntimeCalls();
  calls.memcpy_async = ControlledCopy;
  calls.event_record = ControlledRecord;
  calls.event_query = CorruptCompletedCopy;
  ticket = queue->BeginCopy(source->identity(), std::move(*source));
  if (fault == SnapshotFault::kCopy || fault == SnapshotFault::kFence) {
    EXPECT_EQ(ticket.status().code(), absl::StatusCode::kInternal);
    EXPECT_EQ(arena->PublicationReady().status().code(), absl::StatusCode::kInternal);
  } else if (fault == SnapshotFault::kDrain) {
    ASSERT_TRUE(ticket.ok());
    // The injected failure follows a REAL successful stream drain. The owner
    // nevertheless cannot prove safety from its returned status, so quarantine.
    *valid = LiquidSnapshotLease3D{};
    calls.stream_synchronize = UnprovenDrain;
    queue = absl::CancelledError("");
    calls.stream_synchronize = cudaStreamSynchronize;
    EXPECT_EQ(arena->metrics().quarantined_storage_count, before.quarantined_storage_count + 1);
    EXPECT_EQ(arena->metrics().live_pinned_bytes, before.live_pinned_bytes);
    EXPECT_EQ(arena->PublicationReady().status().code(), absl::StatusCode::kInternal);
    EXPECT_EQ(copy_calls, 1);
    return;
  } else {
    ASSERT_TRUE(ticket.ok());
    ASSERT_EQ(*Await(*queue, *ticket), LiquidSnapshotProgress3D::kReady);
    EXPECT_EQ(queue->Finish(*ticket).status().code(), absl::StatusCode::kDataLoss);
  }
  EXPECT_EQ(copy_calls, 1);
  ASSERT_NE(queue->LastValidHostSnapshot(), nullptr);
  EXPECT_EQ(*queue->LastValidHostSnapshot(), *original);
  EXPECT_EQ(*valid->state(), *original);
  EXPECT_EQ(arena->PublicationReady().status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(arena->metrics().device_reservations, before.device_reservations);
  EXPECT_EQ(arena->metrics().pinned_reservations, before.pinned_reservations);
  EXPECT_EQ(arena->metrics().event_creations, before.event_creations);
  EXPECT_EQ(arena->metrics().stream_creations, before.stream_creations);
}
TEST(CudaLiquidSnapshot3D, DeviceFailureUsesLastValidHostSnapshot) {
  auto state = Fixture();
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(arena->Initialize(state).ok());
  auto source = arena->AcquireCommitted();
  ASSERT_TRUE(source.ok());
  EXPECT_EQ(source->metrics().compiled_architecture, 1200);
  auto queue = CudaLiquidSnapshotQueue3D::Create(*source);
  ASSERT_TRUE(queue.ok()) << queue.status();
  auto ticket = queue->BeginCopy(source->identity(), std::move(*source));
  ASSERT_TRUE(ticket.ok());
  ASSERT_EQ(*Await(*queue, *ticket), LiquidSnapshotProgress3D::kReady);
  EXPECT_TRUE(*arena->PublicationReady());
  auto snapshot = queue->Finish(*ticket);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_EQ(*snapshot->state(), state);
  auto original = CloneLiquidState3D(state, kSmall);
  ASSERT_TRUE(original.ok());
  state.particles[0].volume_ratio = 2;
  Advance(*arena, state);
  source = arena->AcquireCommitted();
  ASSERT_TRUE(source.ok());
  ticket = queue->BeginCopy(source->identity(), std::move(*source));
  ASSERT_TRUE(ticket.ok());
  cuda_internal::LiquidArenaRuntimeCalls().event_query = FailCompletedEvent;
  const auto failure = Await(*queue, *ticket);
  cuda_internal::LiquidArenaRuntimeCalls().event_query = cudaEventQuery;
  EXPECT_EQ(failure.status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(queue->Poll(*ticket).status().code(), absl::StatusCode::kInternal);
  EXPECT_FALSE(queue->Finish(*ticket).ok());
  EXPECT_FALSE(arena->Synchronize().ok());
  ASSERT_NE(queue->LastValidHostSnapshot(), nullptr);
  EXPECT_EQ(*queue->LastValidHostSnapshot(), *original);
  EXPECT_EQ(*snapshot->state(), *original);
  Receipt("liquid_snapshot_3d_failure", *arena, snapshot->copy_milliseconds());
  for (const auto fault :
       {SnapshotFault::kCopy,
        SnapshotFault::kFence,
        SnapshotFault::kCount,
        SnapshotFault::kClock,
        SnapshotFault::kFlags,
        SnapshotFault::kBoolean,
        SnapshotFault::kNan,
        SnapshotFault::kDrain}) {
    ASSERT_NO_FATAL_FAILURE(CheckSnapshotFault(fault));
  }
}
}  // namespace
}  // namespace fluidity
