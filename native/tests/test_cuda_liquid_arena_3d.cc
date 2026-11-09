#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "../cuda/liquid_arena_3d_internal.h"
#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "fluidity/host/cuda_probe.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
constexpr LiquidStateCapacities3D kSmall{33, 1, 1, 192, 2};
void CheckHardware() {
  ProbeReport report;
  ASSERT_TRUE(RunCudaProbe(0, &report).ok());
  ASSERT_EQ(report.device_name, "NVIDIA GeForce RTX 5060 Ti");
  ASSERT_EQ(report.compute_major, 12);
  ASSERT_EQ(report.compute_minor, 0);
  ASSERT_EQ(report.compiled_architecture, 1200);
  std::cout << "arena_device=" << report.device_name << " arch=" << report.compiled_architecture << '\n';
}
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
std::vector<std::byte> PackedBytes(const CudaLiquidEpochLease3D& lease) {
  std::vector<std::byte> bytes(lease.device_bytes().size());
  EXPECT_EQ(cudaMemcpy(bytes.data(), lease.device_bytes().data(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  return bytes;
}
void ExpectNoReservationChange(const CudaLiquidArenaMetrics3D& before, const CudaLiquidArenaMetrics3D& after) {
  EXPECT_EQ(before.device_reservations, after.device_reservations);
  EXPECT_EQ(before.device_frees, after.device_frees);
  EXPECT_EQ(before.pinned_reservations, after.pinned_reservations);
  EXPECT_EQ(before.pinned_frees, after.pinned_frees);
  EXPECT_EQ(before.stream_creations, after.stream_creations);
  EXPECT_EQ(before.event_creations, after.event_creations);
  EXPECT_EQ(before.graph_creations, after.graph_creations);
  EXPECT_EQ(before.executable_creations, after.executable_creations);
}
TEST(CudaLiquidArena3D, StableRepeatedAndEmptyGraphOrdering) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  constexpr LiquidStateCapacities3D capacities{1024, 64, 64, 12288, 2};
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok()) << arena.status();
  auto workspace = LiquidStepWorkspace3D::Create(capacities, {{0, 0, 0}, 1});
  ASSERT_TRUE(workspace.ok());
  auto initial = Fixture();
  ASSERT_TRUE(arena->Initialize(initial).ok());
  std::array<LiquidParticleState3D, 5> particles{};
  constexpr std::array<Vec3, 5> positions{{{6, 2, 2}, {-2, 2, 2}, {-2, 2, 2}, {2, 2, -2}, {2, -2, 2}}};
  for (std::size_t index = 0; index < 5; ++index) {
    particles[index].world_position = positions[index];
    particles[index].id = index + 1;
  }
  const auto before = arena->metrics();
  cudaEvent_t start_event = nullptr;
  cudaEvent_t finish_event = nullptr;
  ASSERT_EQ(cudaEventCreate(&start_event), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&finish_event), cudaSuccess);
  const auto explicit_stream = static_cast<cudaStream_t>(arena->stream());
  ASSERT_EQ(cudaEventRecord(start_event, explicit_stream), cudaSuccess);
  const auto wall_start = std::chrono::steady_clock::now();
  const auto sort_address = arena->range(CudaLiquidRange3D::kSortInputKeys).device;
  for (int replay = 0; replay < 3; ++replay) {
    ASSERT_TRUE(arena->PrepareTopology(particles, {}, *workspace).ok());
    const auto view = workspace->prepared();
    ASSERT_TRUE(view.ok());
    constexpr std::array<std::size_t, 5> expected{3, 4, 1, 2, 0};
    for (std::size_t index = 0; index < expected.size(); ++index) {
      EXPECT_EQ(view->particle_order[index].original_ordinal, expected[index]);
      EXPECT_EQ(view->particles[index].id, expected[index] + 1);
    }
  }
  ASSERT_TRUE(arena->PrepareTopology(std::span(particles).first(3), {}, *workspace).ok());
  EXPECT_EQ(workspace->prepared()->particle_order[0].original_ordinal, 1);
  EXPECT_EQ(workspace->prepared()->particle_order[1].original_ordinal, 2);
  ASSERT_TRUE(arena->PrepareTopology({}, {}, *workspace).ok());
  EXPECT_TRUE(workspace->prepared()->particle_order.empty());
  EXPECT_EQ(arena->range(CudaLiquidRange3D::kSortInputKeys).device, sort_address);
  EXPECT_EQ(arena->metrics().sort_replays - before.sort_replays, 5);
  EXPECT_EQ(arena->metrics().compiled_architecture, 1200);
  EXPECT_EQ(arena->metrics().cccl_version, 3004002);
  EXPECT_EQ(arena->metrics().cub_version, 300402);
  ASSERT_EQ(cudaEventRecord(finish_event, explicit_stream), cudaSuccess);
  ASSERT_EQ(cudaEventSynchronize(finish_event), cudaSuccess);
  float device_milliseconds = 0;
  ASSERT_EQ(cudaEventElapsedTime(&device_milliseconds, start_event, finish_event), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(finish_event), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(start_event), cudaSuccess);
  const auto wall_microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wall_start).count();
  std::cout << "arena_sort fixed_capacity=1024 replays=5 device_interval_ms=" << device_milliseconds
            << " wall_us=" << wall_microseconds
            << " temporary_bytes=" << arena->range(CudaLiquidRange3D::kSortTemporary).pinned.size() << '\n';
  ExpectNoReservationChange(before, arena->metrics());
  particles[0].world_position.x = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(arena->PrepareTopology(particles, {}, *workspace).ok());
  EXPECT_TRUE(workspace->prepared()->particle_order.empty());
}
TEST(CudaLiquidArena3D, PublicationRejectAndLeasedStorage) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok()) << arena.status();
  auto initial = Fixture();
  auto output = CloneLiquidState3D(initial, kSmall);
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(arena->Initialize(initial).ok());
  auto original = arena->AcquireCommitted();
  ASSERT_TRUE(original.ok());
  const auto bytes = PackedBytes(*original);
  EXPECT_EQ(original->layout().particle_bin_stride_bytes, sizeof(CudaLiquidParticleBin3D));
  EXPECT_EQ(original->layout().particle_bin_stride_bytes % 128, 0);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(original->device_bytes().data()) % 128, 0);
  ASSERT_TRUE(original->Readback(*output).ok());
  EXPECT_EQ(*output, initial);
  auto trial = CloneLiquidState3D(initial, kSmall);
  ASSERT_TRUE(trial.ok());
  trial->particles[32].world_position.z += 0.125;
  trial->cells[0].previous_pressure = 19;
  trial->faces[0].carrier_velocity = -3;
  trial->sources[1].fractional_remainder = 0.125;
  LiquidClock3D clock{0.125, 1, 0, 1, 1};
  const auto before = arena->metrics();
  ASSERT_TRUE(arena->PreparePublication(*trial, clock).ok());
  arena->DiscardPublication();
  EXPECT_EQ(PackedBytes(*original), bytes);
  ASSERT_TRUE(arena->AcquireCommitted()->Readback(*output).ok());
  EXPECT_EQ(*output, initial);
  ASSERT_TRUE(arena->PreparePublication(*trial, clock).ok());
  arena->CommitPublication();
  auto accepted = arena->AcquireCommitted();
  ASSERT_TRUE(accepted.ok());
  EXPECT_GT(accepted->identity().generation, original->identity().generation + 1);
  EXPECT_EQ(accepted->identity().clock, clock);
  ASSERT_TRUE(accepted->Readback(*output).ok());
  trial->clock = clock;
  EXPECT_EQ(*output, *trial);
  EXPECT_EQ(PackedBytes(*original), bytes);
  clock = {0.25, 2, 0, 2, 2};
  EXPECT_EQ(arena->PreparePublication(*trial, clock).code(), absl::StatusCode::kFailedPrecondition);
  arena->DiscardPublication();
  ExpectNoReservationChange(before, arena->metrics());
  original = CudaLiquidEpochLease3D{};
  ASSERT_TRUE(arena->PreparePublication(*trial, clock).ok());
  arena->DiscardPublication();
  trial->particles[0].mass = -1;
  EXPECT_FALSE(arena->PreparePublication(*trial, clock).ok());
  EXPECT_EQ(accepted->identity().clock.epoch, 1);
}
TEST(CudaLiquidArena3D, GrowthRollbackAndPeakBudget) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok()) << arena.status();
  auto initial = Fixture();
  ASSERT_TRUE(arena->Initialize(initial).ok());
  auto frame_start = CloneLiquidState3D(initial, kSmall);
  ASSERT_TRUE(frame_start.ok());
  auto clock = LiquidClock3D{0.125, 0, 1, 1, 1};
  initial.particles[0].world_position.x += 0.125;
  ASSERT_TRUE(arena->PreparePublication(initial, clock).ok());
  arena->CommitPublication();
  auto leased = arena->AcquireCommitted();
  ASSERT_TRUE(leased.ok());
  const auto old_bytes = PackedBytes(*leased);
  const auto old_metrics = arena->metrics();
  constexpr LiquidStateCapacities3D grown{65, 2, 2, 384, 3};
  ASSERT_TRUE(arena->PrepareGrowth(grown, *frame_start).ok());
  EXPECT_GT(arena->metrics().live_device_bytes, old_metrics.live_device_bytes);
  arena->DiscardGrowth();
  EXPECT_EQ(arena->metrics().live_device_bytes, old_metrics.live_device_bytes);
  EXPECT_EQ(PackedBytes(*leased), old_bytes);
  const auto graph_before = arena->metrics().graph_creations;
  ASSERT_TRUE(arena->PrepareGrowth(grown, *frame_start).ok());
  arena->CommitGrowth();
  EXPECT_EQ(arena->capacities(), grown);
  EXPECT_EQ(arena->metrics().graph_creations, graph_before + 1);
  auto restored = arena->AcquireCommitted();
  ASSERT_TRUE(restored.ok());
  EXPECT_EQ(restored->identity().clock, frame_start->clock);
  EXPECT_GT(restored->identity().generation, leased->identity().generation);
  auto output = CloneLiquidState3D(*frame_start, grown);
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(restored->Readback(*output).ok());
  EXPECT_EQ(*output, *frame_start);
  arena->CollectRetired();
  EXPECT_EQ(PackedBytes(*leased), old_bytes);
  const auto retained_bytes = arena->metrics().live_device_bytes;
  leased = CudaLiquidEpochLease3D{};
  arena->CollectRetired();
  EXPECT_EQ(arena->metrics().live_device_bytes, retained_bytes - old_metrics.live_device_bytes);
  // Exact old reservation ceiling allows initial allocation, but no old+new overlap.
  auto limited = CudaLiquidArena3D::Create(kSmall, 0, old_metrics.live_device_bytes);
  ASSERT_TRUE(limited.ok());
  ASSERT_TRUE(limited->Initialize(*frame_start).ok());
  const auto before = limited->metrics();
  EXPECT_EQ(limited->PrepareGrowth(grown, *frame_start).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(limited->metrics().live_device_bytes, before.live_device_bytes);
  ASSERT_TRUE(limited->AcquireCommitted()->Readback(*output).ok());
  EXPECT_EQ(*output, *frame_start);
  EXPECT_EQ(CudaLiquidArena3D::Create(kSmall, 0, 1).status().code(), absl::StatusCode::kResourceExhausted);
  auto impossible = grown;
  impossible.particles = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(arena->PrepareGrowth(impossible, *frame_start).code(), absl::StatusCode::kResourceExhausted);
}

TEST(CudaLiquidArena3D, LeaseSurvivesOwnerAndPackedReplayIsExact) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto initial = Fixture();
  initial.geometry.world_origin = {100000000, -2, 3};
  initial.geometry.cell_width = 2;
  CudaLiquidEpochLease3D detached;
  std::vector<std::byte> first_bytes;
  {
    auto arena = CudaLiquidArena3D::Create(kSmall);
    ASSERT_TRUE(arena.ok());
    ASSERT_TRUE(arena->Initialize(initial).ok());
    auto lease = arena->AcquireCommitted();
    ASSERT_TRUE(lease.ok());
    detached = std::move(*lease);
    first_bytes = PackedBytes(detached);
  }
  EXPECT_EQ(PackedBytes(detached), first_bytes);
  auto replay = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(replay.ok());
  ASSERT_TRUE(replay->Initialize(initial).ok());
  auto repeated = replay->AcquireCommitted();
  ASSERT_TRUE(repeated.ok());
  EXPECT_EQ(PackedBytes(*repeated), first_bytes);
  auto output = CloneLiquidState3D(initial, kSmall);
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(detached.Readback(*output).ok());
  EXPECT_EQ(*output, initial);
  LiquidState3D insufficient;
  insufficient.random_seed = 31;
  EXPECT_EQ(detached.Readback(insufficient).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(insufficient.random_seed, 31);
  EXPECT_TRUE(insufficient.particles.values().empty());
  // Actual physical ABI: 18 FP64 fields, u64 ID/i64 side, u32 tag/flags.
  EXPECT_EQ(sizeof(CudaLiquidParticleBin3D), 5376);
  // Read the second lane in the final partial bin without assuming vector alignment.
  std::uint64_t unused_id = 1;
  std::memcpy(
      &unused_id,
      first_bytes.data() + sizeof(CudaLiquidParticleBin3D) + offsetof(CudaLiquidParticleBin3D, particle_ids)
          + sizeof(std::uint64_t),
      sizeof(unused_id)
  );
  EXPECT_EQ(unused_id, 0);
}
TEST(CudaLiquidArena3D, EmptyCapacityAndExhaustedPhysicalClock) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create({});
  ASSERT_TRUE(arena.ok());
  LiquidState3D initial;
  initial.clock.epoch = UINT64_MAX;
  ASSERT_TRUE(arena->Initialize(initial).ok());
  auto workspace = LiquidStepWorkspace3D::Create({}, initial.geometry);
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(arena->PrepareTopology({}, {}, *workspace).ok());
  auto clock = initial.clock;
  clock.time = 1;
  EXPECT_EQ(arena->PreparePublication(initial, clock).code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(arena->AcquireCommitted()->identity().clock.epoch, UINT64_MAX);
  EXPECT_EQ(arena->Initialize(initial).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(CudaLiquidArena3D::Create({}, -1).status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(CudaLiquidArena3D, WorstCaseNamedRangesRemainDisjointAndMeasured) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  constexpr LiquidStateCapacities3D capacity{1024, 128, 4096, 24576, 32};
  const auto start = std::chrono::steady_clock::now();
  auto arena = CudaLiquidArena3D::Create(capacity);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(arena->Initialize(LiquidState3D{}).ok());
  std::vector<CudaLiquidRangeView3D> ranges;
  for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(CudaLiquidRange3D::kCount); ++ordinal) {
    const auto range = arena->range(static_cast<CudaLiquidRange3D>(ordinal));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(range.device) % 128, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(range.pinned.data()) % 128, 0);
    if (range.device == nullptr) {
      EXPECT_EQ(static_cast<CudaLiquidRange3D>(ordinal), CudaLiquidRange3D::kPressurePreparationScratch);
      continue;
    }
    for (const auto earlier : ranges) {
      const auto left = reinterpret_cast<std::uintptr_t>(earlier.device);
      const auto right = reinterpret_cast<std::uintptr_t>(range.device);
      EXPECT_TRUE(left + earlier.pinned.size() <= right || right + range.pinned.size() <= left);
    }
    ranges.push_back(range);
  }
  const auto metrics = arena->metrics();
  EXPECT_LE(metrics.live_device_bytes, metrics.device_byte_limit);
  EXPECT_LE(metrics.device_byte_limit, metrics.total_device_bytes / 10 * 7 + metrics.total_device_bytes % 10 * 7 / 10);
  EXPECT_LE(
      metrics.device_byte_limit,
      metrics.free_device_bytes_at_start / 5 * 4 + metrics.free_device_bytes_at_start % 5 * 4 / 5
  );
  EXPECT_GT(metrics.live_pinned_bytes, metrics.live_device_bytes);
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "arena_reservation device_bytes=" << metrics.live_device_bytes
            << " pinned_bytes=" << metrics.live_pinned_bytes << " ceiling_bytes=" << metrics.device_byte_limit
            << " particle_bin_stride_bytes=" << sizeof(CudaLiquidParticleBin3D)
            << " bytes_per_full_particle_lane=" << sizeof(CudaLiquidParticleBin3D) / 32 << " setup_wall_us=" << elapsed
            << " cccl=" << metrics.cccl_version << " cub=" << metrics.cub_version << '\n';
}
#if defined(_WIN32) && defined(_DEBUG)
std::size_t arena_allocations = 0;
std::size_t arena_frees = 0;
std::size_t arena_fail_allocation = 0;
_CRT_ALLOC_HOOK arena_previous_hook = nullptr;
int ArenaAllocationHook(
    int operation,
    void* data,
    std::size_t size,
    int block_type,
    long request,
    const unsigned char* filename,
    int line
) {
  if (operation == _HOOK_ALLOC || operation == _HOOK_REALLOC) {
    ++arena_allocations;
    if (arena_allocations == arena_fail_allocation) {
      return 0;
    }
  }
  if (operation == _HOOK_FREE) {
    ++arena_frees;
  }
  return arena_previous_hook ? arena_previous_hook(operation, data, size, block_type, request, filename, line) : 1;
}
TEST(CudaLiquidArena3D, AttemptsAndRejectedStatusesAllocateAndFreeNothing) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create({33, 8, 1, 1536, 2});
  ASSERT_TRUE(arena.ok());
  auto initial = Fixture();
  ASSERT_TRUE(arena->Initialize(initial).ok());
  auto workspace = LiquidStepWorkspace3D::Create({33, 8, 1, 1536, 2}, initial.geometry);
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(arena->PrepareTopology(initial.particles.values(), {}, *workspace).ok());
  const auto before = arena->metrics();
  bool sorted = false;
  bool prepared = false;
  bool rejected = false;
  bool invalid_bin_rejected = false;
  arena_allocations = 0;
  arena_frees = 0;
  arena_previous_hook = _CrtSetAllocHook(ArenaAllocationHook);
  {
    sorted = arena->PrepareTopology(initial.particles.values(), {}, *workspace).ok();
    prepared = arena->PreparePublication(initial, {0.125, 1, 0, 1, 1}).ok();
    arena->CommitPublication();
    rejected = arena->PreparePublication(initial, {0.25, 2, 0, 2, 2}).code() == absl::StatusCode::kInvalidArgument;
    arena->DiscardPublication();
    initial.particles[0].world_position.x = std::numeric_limits<double>::infinity();
    invalid_bin_rejected = !arena->PrepareTopology(initial.particles.values(), {}, *workspace).ok();
  }
  _CrtSetAllocHook(arena_previous_hook);
  EXPECT_TRUE(sorted);
  EXPECT_TRUE(prepared);
  EXPECT_TRUE(rejected);
  EXPECT_TRUE(invalid_bin_rejected);
  EXPECT_EQ(arena_allocations, 0);
  EXPECT_EQ(arena_frees, 0);
  ExpectNoReservationChange(before, arena->metrics());
}
TEST(CudaLiquidArena3D, ActualGrowthOwnerAllocationFailurePreservesStorage) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  auto initial = Fixture();
  ASSERT_TRUE(arena->Initialize(initial).ok());
  auto lease = arena->AcquireCommitted();
  ASSERT_TRUE(lease.ok());
  const auto bytes = PackedBytes(*lease);
  const auto before = arena->metrics();
  arena_allocations = 0;
  arena_frees = 0;
  arena_fail_allocation = 1;
  bool exhausted = false;
  arena_previous_hook = _CrtSetAllocHook(ArenaAllocationHook);
  {
    exhausted = arena->PrepareGrowth({65, 2, 2, 384, 3}, initial).code() == absl::StatusCode::kResourceExhausted;
  }
  _CrtSetAllocHook(arena_previous_hook);
  arena_fail_allocation = 0;
  EXPECT_TRUE(exhausted);
  EXPECT_EQ(arena_allocations, 1);
  EXPECT_EQ(arena_frees, 0);
  ExpectNoReservationChange(before, arena->metrics());
  EXPECT_EQ(PackedBytes(*lease), bytes);
}
#endif

TEST(CudaLiquidArena3D, GenerationExhaustionCannotReuseEarlierTicket) {
  cuda_internal::LiquidArenaLedger3D ledger;
  ledger.next_generation = UINT64_MAX - 1;
  auto last = cuda_internal::TakeLiquidArenaGeneration3D(ledger);
  ASSERT_TRUE(last.ok());
  EXPECT_EQ(*last, UINT64_MAX - 1);
  EXPECT_EQ(cuda_internal::TakeLiquidArenaGeneration3D(ledger).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(ledger.next_generation, UINT64_MAX);
  EXPECT_EQ(cuda_internal::TakeLiquidArenaGeneration3D(ledger).status().code(), absl::StatusCode::kOutOfRange);
}
}  // namespace
}  // namespace fluidity
