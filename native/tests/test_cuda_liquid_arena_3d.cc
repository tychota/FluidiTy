#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "../cuda/liquid_arena_3d_internal.h"
#include "../cuda/liquid_arena_dispatch_3d.h"
#include "../cuda/liquid_boundary_3d_bridge.h"
#include "../cuda/liquid_boundary_transfer_3d_bridge.h"
#include "../cuda/liquid_geometry_stages_3d.h"
#include "../cuda/liquid_transfer_stages_3d.h"
#include "../src/liquid_substep_3d_internal.h"
#include "fluidity/host/cuda_liquid_arena_3d.h"
#include "fluidity/host/cuda_liquid_boundary_3d.h"
#include "fluidity/host/cuda_liquid_snapshot_3d.h"
#include "fluidity/host/cuda_probe.h"
#include "gtest/gtest.h"
#include "liquid_regular_cell_transport_3d_cases.h"

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
TEST(CudaLiquidArena3D, AdaptivePublicationAndGrowthKeepBothEpochsIntact) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  constexpr LiquidStateCapacities3D capacity{0, 0, 2, 14, 0};
  LiquidState3D initial;
  initial.geometry.cell_width = 0.5;
  auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(2);
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(14);
  ASSERT_TRUE(cells.ok());
  ASSERT_TRUE(faces.ok());
  initial.adaptive_cells = std::move(*cells);
  initial.adaptive_faces = std::move(*faces);
  ASSERT_TRUE(initial.adaptive_cells.Resize(2).ok());
  ASSERT_TRUE(initial.adaptive_faces.Resize(14).ok());
  initial.adaptive_cells[0] = {
      .key = {.lower_corner = {-2, 0, 0}, .width_exponent = 1},
      .material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian,
      .inventory =
          {.mass_kg = 4,
           .reference_volume_m3 = 0.5,
           .current_volume_m3 = 0.75,
           .momentum_kg_m_per_s = {1, -2, 3},
           .enthalpy_j = -4,
           .phase_id = 9,
           .constituent_count = 2,
           .constituents = {{{11, 0.25}, {21, 0.5}}}},
      .previous_pressure = -2,
  };
  initial.adaptive_cells[1].key = {.lower_corner = {0, 0, 0}, .width_exponent = 0};
  initial.adaptive_faces[0] = {
      .carrier_velocity = -3,
      .validity = LiquidFaceValidity3D::kValid,
      .side_tag = 23,
      .structurally_eligible = false
  };
  ASSERT_TRUE(ValidateLiquidState3D(initial).ok());
  auto uniform_arena = CudaLiquidArena3D::Create(capacity);
  ASSERT_TRUE(uniform_arena.ok());
  EXPECT_EQ(uniform_arena->Initialize(initial).code(), absl::StatusCode::kInvalidArgument);
  auto arena = CudaLiquidArena3D::Create(capacity, 0, SIZE_MAX, false, true);
  ASSERT_TRUE(arena.ok()) << arena.status();
  ASSERT_TRUE(arena->Initialize(initial).ok());
  auto old = arena->AcquireCommitted();
  ASSERT_TRUE(old.ok());
  const auto old_bytes = PackedBytes(*old);
  auto output = CloneLiquidState3D(initial, capacity);
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(old->Readback(*output).ok());
  EXPECT_EQ(*output, initial);

  auto trial = CloneLiquidState3D(initial, capacity);
  ASSERT_TRUE(trial.ok());
  trial->adaptive_cells[0].inventory.enthalpy_j += 2;
  trial->adaptive_cells[0].material_owner = AdaptiveLiquidMaterialOwner3D::kParticles;
  trial->adaptive_cells[1].material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian;
  trial->adaptive_faces[0].carrier_velocity = 3;
  constexpr LiquidClock3D accepted_clock{1, 1, 0, 1, 1};
  ASSERT_TRUE(trial->adaptive_faces.Resize(13).ok());
  EXPECT_EQ(arena->PreparePublication(*trial, accepted_clock).code(), absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(trial->adaptive_faces.Resize(14).ok());
  trial->adaptive_cells[1].material_owner = static_cast<AdaptiveLiquidMaterialOwner3D>(2);
  EXPECT_EQ(arena->PreparePublication(*trial, accepted_clock).code(), absl::StatusCode::kInvalidArgument);
  trial->adaptive_cells[1].material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian;
  EXPECT_EQ(arena->AcquireCommitted()->identity(), old->identity());
  ASSERT_TRUE(arena->PreparePublication(*trial, accepted_clock).ok());
  arena->DiscardPublication();
  EXPECT_EQ(arena->AcquireCommitted()->identity(), old->identity());
  ASSERT_TRUE(arena->PreparePublication(*trial, accepted_clock).ok());
  arena->CommitPublication();
  auto accepted = arena->AcquireCommitted();
  ASSERT_TRUE(accepted.ok());
  EXPECT_EQ(accepted->identity().clock, accepted_clock);
  ASSERT_TRUE(accepted->Readback(*output).ok());
  trial->clock = accepted_clock;
  EXPECT_EQ(*output, *trial);
  EXPECT_EQ(PackedBytes(*old), old_bytes);

  auto queue = CudaLiquidSnapshotQueue3D::Create(*accepted);
  ASSERT_TRUE(queue.ok()) << queue.status();
  auto snapshot = queue->BeginCopy(accepted->identity(), std::move(*accepted));
  ASSERT_TRUE(snapshot.ok()) << snapshot.status();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto progress = queue->Poll(*snapshot);
  ASSERT_TRUE(progress.ok()) << progress.status();
  ASSERT_EQ(*progress, LiquidSnapshotProgress3D::kReady);
  auto frame = queue->Finish(*snapshot);
  ASSERT_TRUE(frame.ok()) << frame.status();
  EXPECT_EQ(*frame->state(), *trial);

  auto grown = capacity;
  grown.cells += 1;
  grown.faces += 3;
  ASSERT_TRUE(arena->PrepareGrowth(grown, *trial).ok());
  arena->CommitGrowth();
  auto restored = arena->AcquireCommitted();
  ASSERT_TRUE(restored.ok());
  EXPECT_GT(restored->identity().generation, old->identity().generation);
  ASSERT_TRUE(restored->Readback(*output).ok());
  EXPECT_EQ(*output, *trial);
  EXPECT_EQ(PackedBytes(*old), old_bytes);

  const auto restored_identity = restored->identity();
  const auto restored_bytes = PackedBytes(*restored);
  auto& calls = cuda_internal::LiquidArenaRuntimeCalls();
  const auto previous_event_record = calls.event_record;
  calls.event_record = [](cudaEvent_t event, cudaStream_t stream) {
    const auto recorded = cudaEventRecord(event, stream);
    return recorded == cudaSuccess ? cudaErrorUnknown : recorded;
  };
  constexpr LiquidClock3D next_clock{2, 2, 0, 2, 2};
  trial->adaptive_cells[0].inventory.mass_kg += 1;
  const auto failed = arena->PreparePublication(*trial, next_clock);
  calls.event_record = previous_event_record;
  EXPECT_EQ(failed.code(), absl::StatusCode::kInternal);
  arena->DiscardPublication();
  EXPECT_EQ(arena->AcquireCommitted()->identity(), restored_identity);
  EXPECT_EQ(PackedBytes(*restored), restored_bytes);
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
  const std::size_t one_state_bytes = 2 * sizeof(CudaLiquidParticleBin3D) + sizeof(LiquidBlockState3D)
      + sizeof(LiquidCellState3D) + sizeof(LiquidFaceState3D) + 2 * sizeof(LiquidSourceState3D)
      + sizeof(cuda_internal::LiquidArenaControl3D);
  EXPECT_EQ(old_metrics.live_physical_state_bytes, 2 * one_state_bytes);
  auto old_queue = CudaLiquidSnapshotQueue3D::Create(*leased);
  ASSERT_TRUE(old_queue.ok());
  constexpr LiquidStateCapacities3D grown{65, 2, 2, 384, 3};
  ASSERT_TRUE(arena->PrepareGrowth(grown, *frame_start).ok());
  EXPECT_GT(arena->metrics().live_device_bytes, old_metrics.live_device_bytes);
  EXPECT_EQ(arena->metrics().live_physical_state_bytes, 3 * one_state_bytes);
  arena->DiscardGrowth();
  EXPECT_EQ(arena->metrics().live_device_bytes, old_metrics.live_device_bytes);
  EXPECT_EQ(PackedBytes(*leased), old_bytes);
  EXPECT_EQ(arena->metrics().live_physical_state_bytes, 2 * one_state_bytes);
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
  {
    auto replacement_queue = CudaLiquidSnapshotQueue3D::Create(*restored);
    ASSERT_TRUE(replacement_queue.ok());
    const auto overlap = arena->metrics();
    const auto queue_pinned_bytes = 2 * (leased->layout().total_state_bytes + restored->layout().total_state_bytes);
    EXPECT_EQ(overlap.live_snapshot_pinned_bytes, queue_pinned_bytes);
    EXPECT_EQ(overlap.peak_snapshot_pinned_bytes, queue_pinned_bytes);
    EXPECT_EQ(overlap.peak_device_bytes, retained_bytes);
    EXPECT_EQ(overlap.peak_pinned_bytes, overlap.live_pinned_bytes);
    EXPECT_EQ(overlap.live_physical_state_bytes, 3 * one_state_bytes);
    EXPECT_EQ(overlap.peak_physical_state_bytes, 3 * one_state_bytes);
    EXPECT_EQ(overlap.live_device_scratch_bytes, retained_bytes - queue_pinned_bytes);
    std::cout << "growth_overlap_device_bytes=" << overlap.live_device_bytes
              << " pinned_bytes=" << overlap.live_pinned_bytes << " snapshot_pinned_bytes=" << queue_pinned_bytes
              << " physical_state_bytes=" << overlap.live_physical_state_bytes
              << " device_scratch_bytes=" << overlap.live_device_scratch_bytes << '\n';
  }
  EXPECT_EQ(arena->metrics().live_snapshot_pinned_bytes, 2 * leased->layout().total_state_bytes);
  leased = CudaLiquidEpochLease3D{};
  arena->CollectRetired();
  EXPECT_EQ(arena->metrics().live_device_bytes, retained_bytes - old_metrics.live_device_bytes);
  EXPECT_EQ(arena->metrics().live_physical_state_bytes, one_state_bytes);
  // The small arena reserves its configured face extent, not the full 512-query batch.
  EXPECT_EQ(
      arena->range(CudaLiquidRange3D::kGeometryInput).pinned.size(),
      grown.faces * sizeof(boundary_internal::GeometryInput)
  );
  EXPECT_EQ(
      arena->range(CudaLiquidRange3D::kGeometryBatchResults).pinned.size(),
      grown.faces * sizeof(boundary_internal::GeometryBatchResult)
  );
  // Exact old reservation ceiling allows initial allocation, but no old+new overlap.
  ASSERT_GT(old_metrics.live_device_bytes, 1u);
  EXPECT_EQ(
      CudaLiquidArena3D::Create(kSmall, 0, old_metrics.live_device_bytes - 1).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  auto limited = CudaLiquidArena3D::Create(kSmall, 0, old_metrics.live_device_bytes);
  ASSERT_TRUE(limited.ok());
  EXPECT_EQ(
      limited->range(CudaLiquidRange3D::kGeometryInput).pinned.size(),
      kSmall.faces * sizeof(boundary_internal::GeometryInput)
  );
  EXPECT_EQ(
      limited->range(CudaLiquidRange3D::kGeometryBatchResults).pinned.size(),
      kSmall.faces * sizeof(boundary_internal::GeometryBatchResult)
  );
  ASSERT_TRUE(limited->Initialize(*frame_start).ok());
  const auto before = limited->metrics();
  EXPECT_EQ(limited->PrepareGrowth(grown, *frame_start).code(), absl::StatusCode::kResourceExhausted);
  const auto refused = limited->metrics();
  EXPECT_EQ(refused.live_device_bytes, before.live_device_bytes);
  EXPECT_EQ(refused.peak_device_bytes, before.peak_device_bytes);
  EXPECT_EQ(refused.budget_refusals, before.budget_refusals + 1);
  ExpectNoReservationChange(before, refused);
  std::cout << "growth_budget_bytes=" << refused.device_byte_limit
            << " refused_allocation_attempts=" << refused.device_reservations - before.device_reservations << '\n';
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
  using Range = CudaLiquidRange3D;
  constexpr std::array wetted_piston_ranges{
      Range::kWettedPistonCenters,
      Range::kWettedPistonDisks,
      Range::kWettedPistonIndices,
      Range::kWettedPistonMeasures,
      Range::kWettedPistonIntervals,
  };
  const auto disjoint = [](const std::byte* left, std::size_t left_size, const std::byte* right, std::size_t right_size
                        ) {
    const auto left_address = reinterpret_cast<std::uintptr_t>(left);
    const auto right_address = reinterpret_cast<std::uintptr_t>(right);
    return left_address + left_size <= right_address || right_address + right_size <= left_address;
  };
  const auto check_ranges = [&](const CudaLiquidArena3D& owner, bool with_wetted_piston) {
    std::vector<CudaLiquidRangeView3D> ranges;
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(Range::kCount); ++ordinal) {
      SCOPED_TRACE(ordinal);
      const auto name = static_cast<Range>(ordinal);
      const auto range = owner.range(name);
      const bool is_wetted_piston = std::ranges::find(wetted_piston_ranges, name) != wetted_piston_ranges.end();
      if (is_wetted_piston && !with_wetted_piston) {
        EXPECT_EQ(range.device, nullptr);
        EXPECT_TRUE(range.pinned.empty());
        EXPECT_EQ(range.pinned.data(), nullptr);
        continue;
      }
      EXPECT_FALSE(range.pinned.empty());
      EXPECT_EQ(range.device == nullptr, name == Range::kPressurePreparationScratch);
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(range.device) % 128, 0);
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(range.pinned.data()) % 128, 0);
      for (const auto earlier : ranges) {
        EXPECT_TRUE(disjoint(earlier.pinned.data(), earlier.pinned.size(), range.pinned.data(), range.pinned.size()));
        if (earlier.device && range.device) {
          EXPECT_TRUE(disjoint(earlier.device, earlier.pinned.size(), range.device, range.pinned.size()));
        }
      }
      ranges.push_back(range);
    }
  };
  check_ranges(*arena, false);
  EXPECT_EQ(
      arena->range(CudaLiquidRange3D::kGeometryInput).pinned.size(),
      512 * sizeof(boundary_internal::GeometryInput)
  );
  EXPECT_EQ(
      arena->range(CudaLiquidRange3D::kGeometryBatchResults).pinned.size(),
      512 * sizeof(boundary_internal::GeometryBatchResult)
  );
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

  auto wetted_piston_arena = CudaLiquidArena3D::CreateWithWettedPiston(capacity);
  ASSERT_TRUE(wetted_piston_arena.ok()) << wetted_piston_arena.status();
  ASSERT_TRUE(wetted_piston_arena->Initialize(LiquidState3D{}).ok());
  check_ranges(*wetted_piston_arena, true);
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

namespace fluidity {
namespace {
TEST(CudaLiquidGeometryStages3D, SignedSphereAndOrderedUnionMatchAnalyticAndCpu) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  const std::array<Vec3, 4> positions{{{0, 0, 0}, {3, 4, 0}, {-2, 0, 0}, {1, 0, 0}}};
  const LiquidSceneField3D sphere = LiquidAnalyticSphere3D{{0, 0, 0}, 2};
  std::array<double, 4> distances{};
  ASSERT_TRUE(cuda_internal::SampleLiquidField3D(*arena, sphere, {}, positions, distances).ok());
  const std::array expected{-2., 3., 0., -1.};
  EXPECT_EQ(distances, expected);
  std::array<LiquidParticleState3D, 2> particles{};
  particles[0].world_position = {-1, 0, 0};
  particles[1].world_position = {1, 0, 0};
  const LiquidSceneField3D field = LiquidParticleSphereUnion3D{0.5};
  ASSERT_TRUE(cuda_internal::SampleLiquidField3D(*arena, field, particles, positions, distances).ok());
  EXPECT_DOUBLE_EQ(distances[0], 0.5);
  EXPECT_DOUBLE_EQ(distances[3], -0.5);
  for (std::size_t i = 0; i < positions.size(); ++i) {
    EXPECT_NEAR(distances[i], liquid_step_internal::FieldDistance(field, particles, positions[i]), 2e-14);
  }
}
TEST(CudaLiquidGeometryStages3D, CrossingWallAnchorUsesTimeAndBothRequestedSlabSides) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(grid.ok());
  std::vector<LiquidBoundaryFaceTag3D> tags(grid->face_count()), cpu_tags(tags.size());
  std::vector<LiquidPrescribedWallSample3D> walls(tags.size()), cpu_walls(tags.size());
  const auto anchor = grid->FaceSlot({FaceNormalAxis::kX, {1, 1, 1}});
  ASSERT_TRUE(anchor.ok());
  for (double coordinate : {0.5, 1.1, 1.5}) {
    const LiquidObstacle3D obstacle = TranslatingSlab3D{{1, 0, 0}, coordinate, 0.2, {1, 2, 3}};
    for (std::int64_t side : {-1, 1}) {
      ASSERT_TRUE(cuda_internal::SampleLiquidFaces3D(*arena, grid->view(), obstacle, side, tags, walls).ok());
      ASSERT_TRUE(liquid_step_internal::FillFaceTags(grid->view(), obstacle, cpu_tags).ok());
      ASSERT_TRUE(liquid_step_internal::FillWalls(grid->view(), obstacle, side, cpu_walls).ok());
      EXPECT_EQ(tags[*anchor].side_tag, coordinate <= 1 ? 1 : -1);
      EXPECT_EQ(tags[*anchor].structurally_eligible, coordinate != 1.1);
      EXPECT_DOUBLE_EQ(walls[*anchor].unit_normal.x, static_cast<double>(side));
      for (std::size_t i = 0; i < tags.size(); ++i) {
        EXPECT_EQ(tags[i].side_tag, cpu_tags[i].side_tag);
        EXPECT_EQ(tags[i].structurally_eligible, cpu_tags[i].structurally_eligible);
        EXPECT_EQ(walls[i].available, cpu_walls[i].available);
        EXPECT_EQ(walls[i].side_tag, cpu_walls[i].side_tag);
        EXPECT_DOUBLE_EQ(walls[i].unit_normal.x, cpu_walls[i].unit_normal.x);
        EXPECT_DOUBLE_EQ(walls[i].unit_normal.y, cpu_walls[i].unit_normal.y);
        EXPECT_DOUBLE_EQ(walls[i].unit_normal.z, cpu_walls[i].unit_normal.z);
        EXPECT_DOUBLE_EQ(walls[i].wall_velocity.x, cpu_walls[i].wall_velocity.x);
        EXPECT_DOUBLE_EQ(walls[i].wall_velocity.y, cpu_walls[i].wall_velocity.y);
        EXPECT_DOUBLE_EQ(walls[i].wall_velocity.z, cpu_walls[i].wall_velocity.z);
      }
    }
  }
  auto workspace = LiquidBoundaryWorkspace3D::Create(kSmall);
  ASSERT_TRUE(workspace.ok());
  std::vector<double> velocities(tags.size(), 0);
  std::vector<LiquidFaceValidity3D> validity(tags.size(), LiquidFaceValidity3D::kValid);
  for (std::int64_t side : {-1, 1}) {
    const LiquidObstacle3D start = TranslatingSlab3D{{1, 0, 0}, 0.5, 0.2, {1, 2, 3}};
    const auto end = liquid_step_internal::ObstacleAtTime(start, 1);
    ASSERT_TRUE(cuda_internal::SampleLiquidFaces3D(*arena, grid->view(), start, side, tags, walls).ok());
    const auto old_history =
        workspace->SaveHistory({0, 0, velocities, validity, tags, walls}, LiquidVelocityTimeLevel3D::kOld);
    ASSERT_TRUE(old_history.ok());
    ASSERT_TRUE(cuda_internal::SampleLiquidFaces3D(*arena, grid->view(), end, side, tags, walls).ok());
    const auto new_history =
        workspace->SaveHistory({1, 1, velocities, validity, tags, walls}, LiquidVelocityTimeLevel3D::kNew);
    ASSERT_TRUE(new_history.ok());
    EXPECT_EQ(old_history->face_tags[*anchor].side_tag, 1);
    EXPECT_EQ(new_history->face_tags[*anchor].side_tag, -1);
    EXPECT_DOUBLE_EQ(old_history->wall_samples[*anchor].unit_normal.x, static_cast<double>(side));
    EXPECT_DOUBLE_EQ(new_history->wall_samples[*anchor].unit_normal.x, static_cast<double>(side));
  }
}
TEST(CudaLiquidGeometryStages3D, FrozenClippingAndTypedLinksMatchCpuAndUnitCube) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  auto workspace = LiquidBoundaryWorkspace3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(grid.ok());
  const std::array cells{CellIndex3D{1, 1, 1}};
  std::vector<LiquidCenterDistance3D> centers;
  std::vector<LiquidCornerDistance3D> corners;
  for (int z = 0; z < 3; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        centers.push_back({{x, y, z}, x == 1 && y == 1 && z == 1 ? -0.5 : 0.5});
        corners.push_back({{x, y, z}, -1});
      }
    }
  }
  const LiquidBoundaryGeometryViewInputs3D inputs{grid->view(), cells, centers, corners, {}, 0.01, 1000, 0};
  auto frozen = cuda_internal::FreezeLiquidGeometry3D(*arena, inputs, *workspace);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  EXPECT_DOUBLE_EQ(frozen->cells[0].fluid_volume, 1);
  const std::array fields{LiquidPressureCellFields3D{0, std::numeric_limits<double>::infinity(), 7}};
  const std::array<double, 6> velocities{};
  auto links = cuda_internal::BuildLiquidLinks3D(*arena, *frozen, {fields, velocities, 9}, *workspace);
  ASSERT_TRUE(links.ok()) << links.status();
  for (const auto& face : links->faces) {
    EXPECT_DOUBLE_EQ(face.pressure_gradient_distance, 0.5);
    EXPECT_DOUBLE_EQ(std::get<DirichletMacPressureLink3D>(face.pressure_link).exterior_pressure, 9);
  }

  // Independent derivative fixture: g=(0.5,0,0), H=(3,2,2), kappa=8.
  for (auto& sample : centers) {
    sample.signed_distance = 0.99;
    if (sample.cell_index.x == 1 && sample.cell_index.y == 1 && sample.cell_index.z == 1) {
      sample.signed_distance = -0.01;
    }
    if (sample.cell_index.x == 2 && sample.cell_index.y == 1 && sample.cell_index.z == 1) {
      sample.signed_distance = 1.99;
    }
  }
  auto capillary_inputs = inputs;
  capillary_inputs.surface_tension = 0.072;
  frozen = cuda_internal::FreezeLiquidGeometry3D(*arena, capillary_inputs, *workspace);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  EXPECT_NEAR(frozen->cells[0].curvature, 8, 2e-14);
  EXPECT_NEAR(frozen->capillary_step_bound, std::sqrt(1000 / (4 * std::acos(-1.) * 0.072)), 1e-13);
  links = cuda_internal::BuildLiquidLinks3D(*arena, *frozen, {fields, velocities, 3}, *workspace);
  ASSERT_TRUE(links.ok()) << links.status();
  EXPECT_EQ(links->lower_theta_regularization_count, 6);
  for (const auto& face : links->faces) {
    EXPECT_DOUBLE_EQ(face.pressure_gradient_distance, 0.05);
    EXPECT_NEAR(std::get<DirichletMacPressureLink3D>(face.pressure_link).exterior_pressure, 3.576, 2e-14);
  }
  auto cpu = LiquidBoundaryWorkspace3D::Create(kSmall);
  ASSERT_TRUE(cpu.ok());
  auto cpu_geometry = cpu->Freeze(capillary_inputs);
  ASSERT_TRUE(cpu_geometry.ok());
  auto cpu_links = cpu->BuildLinks(*cpu_geometry, {fields, velocities, 3});
  ASSERT_TRUE(cpu_links.ok());
  EXPECT_DOUBLE_EQ(frozen->cells[0].curvature, cpu_geometry->cells[0].curvature);
  for (std::size_t i = 0; i < links->faces.size(); ++i) {
    EXPECT_DOUBLE_EQ(links->faces[i].pressure_gradient_distance, cpu_links->faces[i].pressure_gradient_distance);
    EXPECT_DOUBLE_EQ(
        std::get<DirichletMacPressureLink3D>(links->faces[i].pressure_link).exterior_pressure,
        std::get<DirichletMacPressureLink3D>(cpu_links->faces[i].pressure_link).exterior_pressure
    );
  }
  auto invalid = capillary_inputs;
  invalid.step_duration = 2 * frozen->capillary_step_bound;
  EXPECT_EQ(
      cuda_internal::FreezeLiquidGeometry3D(*arena, invalid, *workspace).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_DOUBLE_EQ(frozen->cells[0].curvature, 8);
  auto invalid_fields = fields;
  invalid_fields[0].previous_pressure = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
      cuda_internal::BuildLiquidLinks3D(*arena, *frozen, {invalid_fields, velocities, 3}, *workspace).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_NEAR(std::get<DirichletMacPressureLink3D>(links->faces[0].pressure_link).exterior_pressure, 3.576, 2e-14);
  auto cut = inputs;
  cut.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 1.25, {0.2, 0, 0}};
  auto clipped = cuda_internal::FreezeLiquidGeometry3D(*arena, cut, *workspace);
  ASSERT_TRUE(clipped.ok()) << clipped.status();
  EXPECT_NEAR(clipped->cells[0].fluid_volume, 0.75, 2e-14);
  const auto scalar_clipped = CudaFreezeLiquidBoundaryGeometry3D(
      {*grid, cells, centers, corners, cut.obstacle, cut.step_duration, cut.liquid_density, cut.surface_tension}
  );
  ASSERT_TRUE(scalar_clipped.ok()) << scalar_clipped.status();
  ASSERT_EQ(clipped->cells.size(), scalar_clipped->geometry.cells().size());
  ASSERT_EQ(clipped->faces.size(), scalar_clipped->geometry.faces().size());
  EXPECT_EQ(
      std::bit_cast<std::uint64_t>(clipped->cells[0].fluid_volume),
      std::bit_cast<std::uint64_t>(scalar_clipped->geometry.cells()[0].fluid_volume)
  );
  for (std::size_t ordinal = 0; ordinal < clipped->faces.size(); ++ordinal) {
    const auto& face = clipped->faces[ordinal];
    const auto& scalar_face = scalar_clipped->geometry.faces()[ordinal];
    EXPECT_EQ(face.face_slot, scalar_face.face_slot);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(face.open_area), std::bit_cast<std::uint64_t>(scalar_face.open_area));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(face.blocked_area), std::bit_cast<std::uint64_t>(scalar_face.blocked_area));
    if (face.face_index.axis != FaceNormalAxis::kX) {
      EXPECT_NEAR(face.open_area, 0.75, 2e-14);
    }
  }
  for (const auto scratch_range : {CudaLiquidRange3D::kGeometryInput, CudaLiquidRange3D::kGeometryBatchResults}) {
    const auto bytes = arena->range(scratch_range).pinned;
    ASSERT_GE(bytes.size(), sizeof(BlockIndex3D));
    constexpr BlockIndex3D kActiveBlock{0, 0, 0};
    auto* active_block = std::construct_at(reinterpret_cast<BlockIndex3D*>(bytes.data()), kActiveBlock);
    const auto aliased_grid = SparseMacGridView3D::Create(grid->geometry(), {active_block, 1});
    ASSERT_TRUE(aliased_grid.ok()) << aliased_grid.status();

    auto aliased_frozen = *clipped;
    aliased_frozen.mac_grid = *aliased_grid;
    EXPECT_EQ(
        cuda_internal::BuildLiquidLinks3D(*arena, aliased_frozen, {fields, velocities, 9}, *workspace).status().code(),
        absl::StatusCode::kInvalidArgument
    );
    EXPECT_EQ(active_block->x, kActiveBlock.x);
    EXPECT_EQ(active_block->y, kActiveBlock.y);
    EXPECT_EQ(active_block->z, kActiveBlock.z);

    std::construct_at(active_block, kActiveBlock);
    auto aliased_inputs = inputs;
    aliased_inputs.mac_grid = *aliased_grid;
    EXPECT_EQ(
        cuda_internal::FreezeLiquidGeometry3D(*arena, aliased_inputs, *workspace).status().code(),
        absl::StatusCode::kInvalidArgument
    );
    EXPECT_EQ(active_block->x, kActiveBlock.x);
    EXPECT_EQ(active_block->y, kActiveBlock.y);
    EXPECT_EQ(active_block->z, kActiveBlock.z);

    using WettedFluxes = std::array<std::optional<double>, 6>;
    ASSERT_GE(bytes.size(), sizeof(WettedFluxes));
    auto* aliased_fluxes = std::construct_at(reinterpret_cast<WettedFluxes*>(bytes.data()));
    EXPECT_EQ(
        cuda_internal::BuildLiquidLinks3D(*arena, *clipped, {fields, velocities, 9, *aliased_fluxes}, *workspace)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument
    );
  }
#if defined(_WIN32) && defined(_DEBUG)
  std::vector<LiquidBoundaryFaceTag3D> tags(grid->face_count());
  std::vector<LiquidPrescribedWallSample3D> walls(tags.size());
  const LiquidSceneField3D sphere = LiquidAnalyticSphere3D{{}, 2};
  ASSERT_TRUE(cuda_internal::SampleLiquidField3D(*arena, sphere, {}, Vec3{3, 0, 0}).ok());
  ASSERT_TRUE(cuda_internal::SampleLiquidFaces3D(*arena, grid->view(), cut.obstacle, 0, tags, walls).ok());
  const auto before = arena->metrics();
  bool geometry_ok = false, links_ok = false, field_ok = false, faces_ok = false, rejected = false;
  arena_allocations = 0;
  arena_frees = 0;
  arena_previous_hook = _CrtSetAllocHook(ArenaAllocationHook);
  {
    const auto trial = cuda_internal::FreezeLiquidGeometry3D(*arena, capillary_inputs, *workspace);
    geometry_ok = trial.ok();
    if (trial.ok()) {
      links_ok = cuda_internal::BuildLiquidLinks3D(*arena, *trial, {fields, velocities, 3}, *workspace).ok();
    }
    field_ok = cuda_internal::SampleLiquidField3D(*arena, sphere, {}, Vec3{3, 0, 0}).ok();
    faces_ok = cuda_internal::SampleLiquidFaces3D(*arena, grid->view(), cut.obstacle, 0, tags, walls).ok();
    rejected = cuda_internal::FreezeLiquidGeometry3D(*arena, invalid, *workspace).status().code()
        == absl::StatusCode::kFailedPrecondition;
  }
  _CrtSetAllocHook(arena_previous_hook);
  EXPECT_TRUE(geometry_ok && links_ok && field_ok && faces_ok && rejected);
  EXPECT_EQ(arena_allocations, 0);
  EXPECT_EQ(arena_frees, 0);
  ExpectNoReservationChange(before, arena->metrics());
#endif
}
}  // namespace
}  // namespace fluidity

namespace fluidity {
namespace {
TEST(CudaLiquidGeometryStages3D, RuntimeSubmissionFailureIsFatalAndCannotPublishPreparedCandidate) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  auto initial = Fixture();
  ASSERT_TRUE(arena->Initialize(initial).ok());
  const auto original = arena->AcquireCommitted()->identity();
  auto accepted = initial.clock;
  accepted.time = 0.01;
  accepted.epoch = 1;
  accepted.accepted_step_ordinal = 1;
  accepted.substep_ordinal = 1;
  ASSERT_TRUE(arena->PreparePublication(initial, accepted).ok());
  const auto before = arena->metrics();
  // Inject only the shared dispatch result, without inducing a process-wide CUDA
  // fault. Even a runtime allocation error is fatal after submission begins.
  EXPECT_EQ(arena->RecordRuntimeResult(cudaErrorMemoryAllocation).code(), absl::StatusCode::kInternal);
  EXPECT_EQ(arena->RecordRuntimeResult(cudaSuccess).code(), absl::StatusCode::kInternal);
  arena->CommitPublication();
  EXPECT_EQ(arena->AcquireCommitted()->identity(), original);
  EXPECT_TRUE(arena->range(CudaLiquidRange3D::kSamplePositions).pinned.empty());
  double distance = 29;
  const Vec3 position{};
  EXPECT_EQ(
      cuda_internal::SampleLiquidField3D(*arena, LiquidAnalyticSphere3D{{}, 1}, {}, {&position, 1}, {&distance, 1})
          .code(),
      absl::StatusCode::kInternal
  );
  EXPECT_DOUBLE_EQ(distance, 29);
  auto growth = kSmall;
  growth.particles = 64;
  arena->DiscardPublication();
  EXPECT_EQ(arena->PrepareGrowth(growth, initial).code(), absl::StatusCode::kInternal);
  ExpectNoReservationChange(before, arena->metrics());
}
TEST(CudaLiquidGeometryStages3D, InvalidAndOverflowedSamplesPreserveOutputsAndAllowValidReuse) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  std::array<Vec3, 1> positions{Vec3{1, 0, 0}};
  std::array<double, 1> output{42};
  EXPECT_EQ(
      cuda_internal::SampleLiquidField3D(*arena, LiquidParticleSphereUnion3D{1}, {}, positions, output).code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(output[0], 42);
  positions[0].x = std::numeric_limits<double>::max();
  EXPECT_EQ(
      cuda_internal::SampleLiquidField3D(
          *arena,
          LiquidAnalyticSphere3D{{-std::numeric_limits<double>::max(), 0, 0}, 1},
          {},
          positions,
          output
      )
          .code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(output[0], 42);
  positions[0].x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
      cuda_internal::SampleLiquidField3D(*arena, LiquidAnalyticSphere3D{{}, 1}, {}, positions, output).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(output[0], 42);
  positions[0] = {3, 0, 0};
  ASSERT_TRUE(cuda_internal::SampleLiquidField3D(*arena, LiquidAnalyticSphere3D{{}, 1}, {}, positions, output).ok());
  EXPECT_EQ(output[0], 2);
  const auto oversized = cuda_internal::CheckedLiquidRange3D<double>(
      *arena,
      CudaLiquidRange3D::kSampleDistances,
      std::numeric_limits<std::size_t>::max()
  );
  EXPECT_EQ(oversized.status().code(), absl::StatusCode::kResourceExhausted);
  ASSERT_TRUE(cuda_internal::SampleLiquidField3D(
                  *arena,
                  LiquidAnalyticSphere3D{{}, 1},
                  {},
                  std::span<const Vec3>{},
                  std::span<double>{}
  )
                  .ok());
}
}  // namespace
}  // namespace fluidity

namespace fluidity {
namespace {
TEST(CudaLiquidGeometryStages3D, OverflowedFaceDistanceCannotPublishEligibility) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  auto grid = SparseMacGrid3D::Create({{1.1e308, 1.1e308, 1.1e308}, 1e292}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(grid.ok()) << grid.status();
  std::vector<LiquidBoundaryFaceTag3D> tags(grid->face_count(), {99, false});
  const double component = 1 / std::sqrt(3.);
  const LiquidObstacle3D obstacle = TranslatingHalfspace3D{{component, component, component}, 0, {}};
  const auto result = cuda_internal::SampleLiquidFaces3D(*arena, grid->view(), obstacle, 0, tags, {});
  EXPECT_EQ(result.code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(tags[0].side_tag, 99);
  EXPECT_FALSE(tags[0].structurally_eligible);
}
}  // namespace
}  // namespace fluidity

namespace fluidity {
namespace {
enum class ArenaRuntimeFault {
  kMemset,
  kSort,
  kArchitecture,
  kEndCapture,
  kWarmLaunch,
  kDrain
};
class ScopedArenaRuntimeFault {
 public:
  explicit ScopedArenaRuntimeFault(ArenaRuntimeFault fault) : previous_(cuda_internal::LiquidArenaRuntimeCalls()) {
    selected_ = fault;
    calls_ = 0;
    auto& runtime = cuda_internal::LiquidArenaRuntimeCalls();
    runtime.memset_async = Memset;
    runtime.sort_bins = Sort;
    runtime.launch_architecture = Architecture;
    runtime.end_capture = EndCapture;
    runtime.graph_launch = GraphLaunch;
    runtime.stream_synchronize = Drain;
  }
  ~ScopedArenaRuntimeFault() {
    cuda_internal::LiquidArenaRuntimeCalls() = previous_;
  }
  std::size_t calls() const {
    return calls_;
  }

 private:
  static cudaError_t Inject(ArenaRuntimeFault call, cudaError_t result) {
    if (selected_ == call && result == cudaSuccess) {
      ++calls_;
      return cudaErrorMemoryAllocation;
    }
    return result;
  }
  static cudaError_t Memset(void* pointer, int value, std::size_t bytes, cudaStream_t stream) {
    return Inject(ArenaRuntimeFault::kMemset, cudaMemsetAsync(pointer, value, bytes, stream));
  }
  static int Sort(
      void* temporary,
      std::size_t& temporary_bytes,
      cuda_internal::LiquidSortKey3D* input,
      cuda_internal::LiquidSortKey3D* output,
      std::size_t* incoming,
      std::size_t* ordered,
      std::size_t capacity,
      void* stream
  ) {
    return static_cast<int>(Inject(
        ArenaRuntimeFault::kSort,
        static_cast<cudaError_t>(
            cuda_internal::
                SortLiquidBins3D(temporary, temporary_bytes, input, output, incoming, ordered, capacity, stream)
        )
    ));
  }
  static int Architecture(int* architecture, void* stream) {
    return static_cast<int>(Inject(
        ArenaRuntimeFault::kArchitecture,
        static_cast<cudaError_t>(cuda_internal::LaunchLiquidArenaArchitecture3D(architecture, stream))
    ));
  }
  static cudaError_t EndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    return Inject(ArenaRuntimeFault::kEndCapture, cudaStreamEndCapture(stream, graph));
  }
  static cudaError_t GraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    return Inject(ArenaRuntimeFault::kWarmLaunch, cudaGraphLaunch(graph, stream));
  }
  static cudaError_t Drain(cudaStream_t stream) {
    // Real successful synchronization establishes safety in this process before
    // the synthetic failure exercises conservative production quarantine.
    return Inject(ArenaRuntimeFault::kDrain, cudaStreamSynchronize(stream));
  }
  cuda_internal::LiquidArenaRuntimeCalls3D previous_;
  static inline ArenaRuntimeFault selected_ = ArenaRuntimeFault::kMemset;
  static inline std::size_t calls_ = 0;
};
TEST(CudaLiquidArena3D, GrowthRuntimeSubmissionErrorsPoisonWithoutChangingCommittedLease) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  constexpr std::array faults{
      ArenaRuntimeFault::kMemset,
      ArenaRuntimeFault::kSort,
      ArenaRuntimeFault::kArchitecture,
      ArenaRuntimeFault::kEndCapture,
      ArenaRuntimeFault::kWarmLaunch,
  };
  for (const auto fault : faults) {
    SCOPED_TRACE(static_cast<int>(fault));
    auto arena = CudaLiquidArena3D::Create(kSmall);
    ASSERT_TRUE(arena.ok());
    auto initial = Fixture();
    ASSERT_TRUE(arena->Initialize(initial).ok());
    auto lease = arena->AcquireCommitted();
    ASSERT_TRUE(lease.ok());
    const auto identity = lease->identity();
    const auto committed_bytes = PackedBytes(*lease);
    const auto* device_bytes = lease->device_bytes().data();
    const auto before = arena->metrics();
    auto growth = kSmall;
    growth.particles = 64;
    {
      ScopedArenaRuntimeFault injection(fault);
      EXPECT_EQ(arena->PrepareGrowth(growth, initial).code(), absl::StatusCode::kInternal);
      EXPECT_EQ(injection.calls(), 1);
    }
    EXPECT_EQ(arena->RecordRuntimeResult(cudaSuccess).code(), absl::StatusCode::kInternal);
    EXPECT_EQ(arena->AcquireCommitted()->identity(), identity);
    EXPECT_EQ(lease->identity(), identity);
    EXPECT_EQ(lease->device_bytes().data(), device_bytes);
    // Injection returned an error after real successful calls; the actual CUDA
    // context remains healthy, so this test-only raw read can compare bytes.
    EXPECT_EQ(PackedBytes(*lease), committed_bytes);
    EXPECT_TRUE(arena->range(CudaLiquidRange3D::kSamplePositions).pinned.empty());
    const auto after = arena->metrics();
    EXPECT_EQ(after.device_reservations, before.device_reservations + 1);
    EXPECT_EQ(after.pinned_reservations, before.pinned_reservations + 1);
    EXPECT_EQ(after.device_frees, before.device_frees + 1);
    EXPECT_EQ(after.pinned_frees, before.pinned_frees + 1);
    EXPECT_EQ(after.live_device_bytes, before.live_device_bytes);
    EXPECT_EQ(after.live_pinned_bytes, before.live_pinned_bytes);
    EXPECT_EQ(after.quarantined_storage_count, 0);
    EXPECT_EQ(after.stream_destructions, before.stream_destructions + 1);
    EXPECT_EQ(after.event_destructions, before.event_destructions + 3);
    EXPECT_EQ(after.graph_destructions - before.graph_destructions, after.graph_creations - before.graph_creations);
    EXPECT_EQ(
        after.executable_destructions - before.executable_destructions,
        after.executable_creations - before.executable_creations
    );
    EXPECT_EQ(arena->PrepareGrowth(growth, initial).code(), absl::StatusCode::kInternal);
    ExpectNoReservationChange(after, arena->metrics());
    EXPECT_EQ(
        cuda_internal::SampleLiquidField3D(*arena, LiquidAnalyticSphere3D{{}, 1}, {}, Vec3{}).status().code(),
        absl::StatusCode::kInternal
    );
  }
}
TEST(CudaLiquidArena3D, FailedRetiredDrainQuarantinesPayloadAndPoisonsSurvivingLease) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto arena = CudaLiquidArena3D::Create(kSmall);
  ASSERT_TRUE(arena.ok());
  auto initial = Fixture();
  ASSERT_TRUE(arena->Initialize(initial).ok());
  const auto original_reservation = arena->metrics();
  auto growth = kSmall;
  growth.particles = 64;
  ASSERT_TRUE(arena->PrepareGrowth(growth, initial).ok());
  arena->CommitGrowth();
  auto lease = arena->AcquireCommitted();
  ASSERT_TRUE(lease.ok());
  const auto identity = lease->identity();
  const auto* device_bytes = lease->device_bytes().data();
  const auto before = arena->metrics();
  {
    ScopedArenaRuntimeFault injection(ArenaRuntimeFault::kDrain);
    arena->CollectRetired();
    EXPECT_EQ(injection.calls(), 1);
  }
  const auto after = arena->metrics();
  EXPECT_EQ(after.quarantined_storage_count, 1);
  ExpectNoReservationChange(before, after);
  EXPECT_EQ(after.stream_destructions, before.stream_destructions);
  EXPECT_EQ(after.event_destructions, before.event_destructions);
  EXPECT_EQ(after.graph_destructions, before.graph_destructions);
  EXPECT_EQ(after.executable_destructions, before.executable_destructions);
  EXPECT_EQ(after.live_device_bytes, before.live_device_bytes);
  EXPECT_EQ(after.live_pinned_bytes, before.live_pinned_bytes);
  EXPECT_EQ(arena->RecordRuntimeResult(cudaSuccess).code(), absl::StatusCode::kInternal);
  EXPECT_EQ(arena->AcquireCommitted()->identity(), identity);
  EXPECT_EQ(lease->identity(), identity);
  EXPECT_EQ(lease->device_bytes().data(), device_bytes);
  EXPECT_EQ(lease->Readback(initial).code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(arena->range(CudaLiquidRange3D::kSamplePositions).pinned.empty());
  arena->CollectRetired();
  ExpectNoReservationChange(after, arena->metrics());
  {
    auto retiring_arena = std::move(*arena);
  }
  // Quarantine and the surviving immutable lease keep both physical owners and
  // the poisoned ledger alive after destruction of the scheduler's arena owner.
  EXPECT_EQ(lease->metrics().quarantined_storage_count, 1);
  EXPECT_EQ(lease->metrics().live_device_bytes, after.live_device_bytes);
  EXPECT_EQ(lease->metrics().live_pinned_bytes, after.live_pinned_bytes);
  EXPECT_EQ(lease->identity(), identity);
  EXPECT_EQ(lease->Readback(initial).code(), absl::StatusCode::kInternal);
  std::cout << "fatal_quarantine owners=" << after.quarantined_storage_count
            << " retained_device_bytes=" << original_reservation.live_device_bytes
            << " retained_pinned_bytes=" << original_reservation.live_pinned_bytes << '\n';
}
}  // namespace
}  // namespace fluidity

namespace fluidity {
namespace {
LiquidBoundaryParticle3D TransferParticle(Vec3 world_position, Vec3 world_velocity, std::int64_t side_tag = 0) {
  return {
      .transfer =
          {
              .sorted_world_position = world_position,
              .trial_world_position = world_position,
              .particle_mass = 1,
              .particle_velocity = world_velocity,
              .particle_velocity_slopes = {},
          },
      .side_tag = side_tag,
      .affine_enabled = true,
  };
}
class CudaLiquidTransferStages3D : public ::testing::Test {
 protected:
  static constexpr LiquidStateCapacities3D capacities{4, 1, 1, 192, 0};
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(CheckHardware());
  }
};
TEST_F(CudaLiquidTransferStages3D, ExactDepositAndTransactionalBounds) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  auto cpu = LiquidBoundaryWorkspace3D::Create(capacities);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{-1, -1, -1}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(cpu.ok());
  ASSERT_TRUE(grid.ok());
  std::array<LiquidBoundaryFaceTag3D, 192> tags{};
  std::array particles{TransferParticle({-2, -2, -2}, {-1.5, 2.5, -3.5}, 0)};
  LiquidBoundaryDepositViewInputs3D inputs{grid->view(), particles, tags, {0x1p-30, 0x1p-30, 4}};
  auto expected = cpu->Deposit(inputs);
  ASSERT_TRUE(expected.ok());
  auto result = cuda_internal::DepositLiquid3D(*arena, inputs, *workspace);
  ASSERT_TRUE(result.ok()) << result.status();
  for (std::size_t i = 0; i < 192; ++i) {
    EXPECT_EQ(result->face_quanta[i].mass_quanta_sum, expected->face_quanta[i].mass_quanta_sum);
    EXPECT_EQ(result->face_quanta[i].pure_momentum_quanta_sum, expected->face_quanta[i].pure_momentum_quanta_sum);
  }
  const auto previous = result->face_quanta[42];
  const auto* address = result->face_quanta.data();
  particles[0].transfer.particle_velocity.x = 1e100;
  EXPECT_EQ(cuda_internal::DepositLiquid3D(*arena, inputs, *workspace).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(address[42].mass_quanta_sum, previous.mass_quanta_sum);
}
TEST_F(CudaLiquidTransferStages3D, ContendedDepositPreservesBoundaryQuantaAndFirstError) {
  constexpr std::size_t kParticleCount = 256;
  constexpr std::int64_t kMaximumFaceQuanta = std::int64_t{1} << 62;
  auto large = capacities;
  large.particles = kParticleCount;
  auto arena = CudaLiquidArena3D::Create(large);
  auto workspace = LiquidBoundaryWorkspace3D::Create(large);
  auto cpu = LiquidBoundaryWorkspace3D::Create(large);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{-1, -1, -1}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(cpu.ok());
  ASSERT_TRUE(grid.ok());
  std::array<LiquidBoundaryFaceTag3D, 192> tags;
  tags.fill({.structurally_eligible = false});
  constexpr std::array axes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
  std::array<std::size_t, axes.size()> selected_faces{};
  for (std::size_t index = 0; index < axes.size(); ++index) {
    const auto face = grid->FaceSlot({axes[index], {-2, -2, -2}});
    ASSERT_TRUE(face.ok()) << face.status();
    tags[*face].structurally_eligible = true;
    selected_faces[index] = *face;
  }
  std::vector particles(kParticleCount, TransferParticle({-2, -2, -2}, {1, -1, 1}));
  for (std::size_t index = 0; index < particles.size(); ++index) {
    particles[index].transfer.particle_mass = 0x1p54;
    particles[index].transfer.particle_velocity.z = index % 2 == 0 ? 1 : -1;
  }
  const LiquidBoundaryDepositViewInputs3D inputs{grid->view(), particles, tags, {1, 1, kParticleCount}};
  const auto expected = cpu->Deposit(inputs);
  ASSERT_TRUE(expected.ok()) << expected.status();
  const auto actual = cuda_internal::DepositLiquid3D(*arena, inputs, *workspace);
  ASSERT_TRUE(actual.ok()) << actual.status();
  const auto expect_exact_quanta = [&] {
    for (std::size_t slot = 0; slot < tags.size(); ++slot) {
      EXPECT_EQ(actual->face_quanta[slot].mass_quanta_sum, expected->face_quanta[slot].mass_quanta_sum);
      EXPECT_EQ(
          actual->face_quanta[slot].pure_momentum_quanta_sum,
          expected->face_quanta[slot].pure_momentum_quanta_sum
      );
      EXPECT_EQ(actual->face_quanta[slot].stress_force_quanta_sum, 0);
    }
  };
  expect_exact_quanta();
  for (const auto slot : selected_faces) {
    EXPECT_EQ(actual->face_quanta[slot].mass_quanta_sum, kMaximumFaceQuanta);
  }
  EXPECT_EQ(actual->face_quanta[selected_faces[0]].pure_momentum_quanta_sum, kMaximumFaceQuanta);
  EXPECT_EQ(actual->face_quanta[selected_faces[1]].pure_momentum_quanta_sum, -kMaximumFaceQuanta);
  EXPECT_EQ(actual->face_quanta[selected_faces[2]].pure_momentum_quanta_sum, 0);

  particles[0].transfer.particle_velocity.x = 2;
  particles[1].side_tag = 1;
  EXPECT_EQ(cuda_internal::DepositLiquid3D(*arena, inputs, *workspace).status().code(), absl::StatusCode::kOutOfRange);
  expect_exact_quanta();
  particles[0].side_tag = 1;
  particles[1].side_tag = 0;
  particles[1].transfer.particle_velocity.x = 2;
  EXPECT_EQ(
      cuda_internal::DepositLiquid3D(*arena, inputs, *workspace).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  expect_exact_quanta();
}

TEST_F(CudaLiquidTransferStages3D, NormalizationRetainsLowMass) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  std::array<MacFaceDepositQuanta3D, 3> quanta{{{0, 0, 0}, {1, -3, 0}, {4, -6, 0}}};
  auto result = cuda_internal::NormalizeLiquid3D(*arena, quanta, {0.5, 0.5, 4}, 0.5, *workspace);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->validity[0], LiquidFaceValidity3D::kMissing);
  EXPECT_EQ(result->validity[1], LiquidFaceValidity3D::kLowMass);
  EXPECT_DOUBLE_EQ(result->face_masses[1], 0.5);
  EXPECT_DOUBLE_EQ(result->face_velocities[2], -1.5);

  quanta = {{{2, 0, 0}, {-1, 0, 0}, {0, 0, 0}}};
  EXPECT_EQ(
      cuda_internal::NormalizeLiquid3D(*arena, quanta, {0x1p1023, 1, 4}, 0, *workspace).status().code(),
      absl::StatusCode::kOutOfRange
  );
  std::swap(quanta[0], quanta[1]);
  EXPECT_EQ(
      cuda_internal::NormalizeLiquid3D(*arena, quanta, {0x1p1023, 1, 4}, 0, *workspace).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(result->validity[1], LiquidFaceValidity3D::kLowMass);
  EXPECT_DOUBLE_EQ(result->face_velocities[2], -1.5);
}
TEST_F(CudaLiquidTransferStages3D, NormalizationCopiesOnlyActiveQuantaAndDecodedResults) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  static std::array<const void*, 2> face_buffers;
  static std::array<std::size_t, 2> copy_bytes;
  face_buffers = {
      arena->range(CudaLiquidRange3D::kTransferFaces).device,
      arena->range(CudaLiquidRange3D::kTransferFaceResults).device,
  };
  copy_bytes = {};
  auto& runtime = cuda_internal::LiquidArenaRuntimeCalls();
  const auto previous_copy = runtime.memcpy_async;
  runtime.memcpy_async =
      [](void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream
      ) -> cudaError_t {
    if (kind == cudaMemcpyHostToDevice && destination == face_buffers[0]) {
      copy_bytes[0] += bytes;
    }
    if (kind == cudaMemcpyDeviceToHost && source == face_buffers[1]) {
      copy_bytes[1] += bytes;
    }
    return cudaMemcpyAsync(destination, source, bytes, kind, stream);
  };
  const std::array<MacFaceDepositQuanta3D, 3> quanta{{{0, 0, 0}, {1, -3, 0}, {4, -6, 0}}};
  const auto result = cuda_internal::NormalizeLiquid3D(*arena, quanta, {0.5, 0.5, 4}, 0.5, *workspace);
  runtime.memcpy_async = previous_copy;
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(copy_bytes[0], quanta.size() * sizeof(MacFaceDepositQuanta3D));
  EXPECT_EQ(copy_bytes[1], quanta.size() * 3 * sizeof(double));
  EXPECT_DOUBLE_EQ(result->face_velocities[2], -1.5);
}

TEST_F(CudaLiquidTransferStages3D, ForcesApplyAccelerationOnce) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(grid.ok());
  std::array<double, 192> old{};
  old.fill(2);
  std::array<double, 192> result{};
  ASSERT_TRUE(cuda_internal::ApplyLiquidForces3D(*arena, grid->view(), old, {1, -2, 3}, 0.25, result).ok());
  EXPECT_DOUBLE_EQ(result[0], 2.25);
  EXPECT_DOUBLE_EQ(result[64], 1.5);
  EXPECT_DOUBLE_EQ(result[128], 2.75);
}
TEST_F(CudaLiquidTransferStages3D, SeparateGatherAndMissingPrediction) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(grid.ok());
  std::array<double, 192> old{}, next{};
  old.fill(1);
  next.fill(3);
  std::array<LiquidFaceValidity3D, 192> valid{};
  valid.fill(LiquidFaceValidity3D::kValid);
  std::array<LiquidBoundaryFaceTag3D, 192> tags{};
  std::array<LiquidPrescribedWallSample3D, 192> walls{};
  LiquidBoundaryVelocityHistory3D old_history{0, 0, old, valid, tags, walls};
  LiquidBoundaryVelocityHistory3D new_history{1, 1, next, valid, tags, walls};
  std::array particles{TransferParticle({2, 2, 2}, {2, 2, 2}, 0)};
  auto result =
      cuda_internal::GatherLiquid3D(*arena, {grid->view(), particles, 0, 1, old_history, new_history}, *workspace);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->particles[0].new_pic_velocity.x, 3);
  EXPECT_DOUBLE_EQ(result->particles[0].particle_minus_old_pic_velocity.x, 1);
  valid[42] = LiquidFaceValidity3D::kMissing;
  auto prediction = cuda_internal::PredictMissingLiquidFace3D(
      *arena,
      {grid->view(), old_history, 42, 0, {2, 0, 0}, 0.25},
      *workspace
  );
  ASSERT_TRUE(prediction.ok()) << prediction.status();
  EXPECT_NEAR(*prediction, 1.5, 1e-12);
}
TEST_F(CudaLiquidTransferStages3D, TrueGradientVolumeAndTwoVelocities) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  std::array<LiquidParticleState3D, 1> particles{};
  particles[0].world_position = {2, 2, 2};
  particles[0].id = 7;
  particles[0].mass = 3;
  particles[0].reference_volume = 0.2;
  particles[0].pic_initialized = true;
  std::array<LiquidBoundaryParticleGather3D, 1> gathered{};
  gathered[0].new_pic_velocity = {1, 0, 0};
  gathered[0].particle_minus_old_pic_velocity = {2, 0, 0};
  gathered[0].velocity_gradient.x_row.x = 0.5;
  gathered[0].fitted_velocity_slopes.x_row.x = 9;
  std::array<double, 1> distances{-0.5};
  LiquidParticleUpdateInputs3D inputs{gathered, distances, {}, 1, 0.1, 0.6, 0, 1e-10};
  ASSERT_TRUE(cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles).ok());
  EXPECT_NEAR(particles[0].volume_ratio, std::exp(0.05), 2e-14);
  EXPECT_NEAR(particles[0].world_velocity.x, 2.188, 2e-14);
  EXPECT_NEAR(particles[0].world_position.x, 2.17128, 2e-14);
  EXPECT_EQ(particles[0].id, 7);
  EXPECT_EQ(particles[0].mass, 3);
  EXPECT_EQ(particles[0].reference_volume, 0.2);
}
}  // namespace
}  // namespace fluidity

namespace fluidity {
namespace {
TEST_F(CudaLiquidTransferStages3D, SignedTiesGuardEndpointAndMissingOriginalOwner) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(grid.ok());
  std::array<LiquidBoundaryFaceTag3D, 192> tags{};
  for (auto& tag : tags) {
    tag.structurally_eligible = false;
  }
  std::array particles{TransferParticle({2, 2, 2}, {-1.5, 2.5, -3.5}, 0)};
  const std::array axes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
  std::array<std::size_t, 3> selected{};
  for (std::size_t component = 0; component < 3; ++component) {
    auto stencil = BuildOwnedMacFaceStencil3D(grid->view(), axes[component], {2, 2, 2}, {2, 2, 2});
    ASSERT_TRUE(stencil.ok());
    selected[component] = (*stencil)[13].face_slot;
    tags[selected[component]].structurally_eligible = true;
  }
  LiquidBoundaryDepositViewInputs3D inputs{grid->view(), particles, tags, {1, 1, 4}};
  auto result = cuda_internal::DepositLiquid3D(*arena, inputs, *workspace);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->face_quanta[selected[0]].pure_momentum_quanta_sum, -2);
  EXPECT_EQ(result->face_quanta[selected[1]].pure_momentum_quanta_sum, 2);
  EXPECT_EQ(result->face_quanta[selected[2]].pure_momentum_quanta_sum, -4);
  particles[0].transfer.particle_mass = 0x1p60;
  particles[0].transfer.particle_velocity = {-1, 1, -1};
  result = cuda_internal::DepositLiquid3D(*arena, inputs, *workspace);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->face_quanta[selected[0]].mass_quanta_sum, std::int64_t{1} << 60);
  EXPECT_EQ(result->face_quanta[selected[0]].pure_momentum_quanta_sum, -(std::int64_t{1} << 60));
  const auto* prior = result->face_quanta.data();
  const auto previous = prior[selected[0]].mass_quanta_sum;
  particles[0].transfer.particle_mass = std::nextafter(0x1p60, INFINITY);
  EXPECT_EQ(cuda_internal::DepositLiquid3D(*arena, inputs, *workspace).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(prior[selected[0]].mass_quanta_sum, previous);
  particles[0].transfer.particle_mass = 1;
  particles[0].transfer.sorted_world_position = {0, 0, 0};
  particles[0].transfer.trial_world_position = {0, 0, 0};
  for (auto& tag : tags) {
    tag.structurally_eligible = false;
  }
  EXPECT_EQ(
      cuda_internal::DepositLiquid3D(*arena, inputs, *workspace).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(prior[selected[0]].mass_quanta_sum, previous);
}
TEST_F(CudaLiquidTransferStages3D, MissingPredictionUsesNearestSlotAndRequestedWallSide) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(grid.ok());
  std::array<double, 192> velocities{};
  std::array<LiquidFaceValidity3D, 192> validity{};
  std::array<LiquidBoundaryFaceTag3D, 192> tags{};
  std::array<LiquidPrescribedWallSample3D, 192> walls{};
  // Equal-distance X donors at x=1 and x=3 around target x=2. Lower owned slot wins.
  validity[41] = LiquidFaceValidity3D::kValid;
  validity[43] = LiquidFaceValidity3D::kValid;
  velocities[41] = 7;
  velocities[43] = 9;
  LiquidBoundaryVelocityHistory3D history{0, 0, velocities, validity, tags, walls};
  LiquidMissingFacePredictionInputs3D inputs{grid->view(), history, 42, 0, {2, 0, 0}, 0.25};
  auto result = cuda_internal::PredictMissingLiquidFace3D(*arena, inputs, *workspace);
  ASSERT_TRUE(result.ok());
  EXPECT_DOUBLE_EQ(*result, 7.5);
  tags[41].side_tag = -1;
  result = cuda_internal::PredictMissingLiquidFace3D(*arena, inputs, *workspace);
  ASSERT_TRUE(result.ok());
  EXPECT_DOUBLE_EQ(*result, 9.5);
  validity[43] = LiquidFaceValidity3D::kLowMass;
  walls[42] = {true, 0, {1, 0, 0}, {3, 4, 5}};
  result = cuda_internal::PredictMissingLiquidFace3D(*arena, inputs, *workspace);
  ASSERT_TRUE(result.ok());
  EXPECT_DOUBLE_EQ(*result, 3.5);
  walls[42].side_tag = 1;
  EXPECT_EQ(
      cuda_internal::PredictMissingLiquidFace3D(*arena, inputs, *workspace).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}
TEST_F(CudaLiquidTransferStages3D, GatherUsesIndependentOldNewSidesWallsAndLowMass) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  auto cpu = LiquidBoundaryWorkspace3D::Create(capacities);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(cpu.ok());
  ASSERT_TRUE(grid.ok());
  std::array<double, 192> old{}, next{};
  old.fill(77);
  next.fill(99);
  std::array<LiquidFaceValidity3D, 192> old_valid{}, new_valid{};
  old_valid.fill(LiquidFaceValidity3D::kValid);
  new_valid.fill(LiquidFaceValidity3D::kLowMass);
  std::array<LiquidBoundaryFaceTag3D, 192> old_tags{}, new_tags{};
  std::array<LiquidPrescribedWallSample3D, 192> old_walls{}, new_walls{};
  std::array particles{TransferParticle({2, 2, 2}, {8, 8, 8}, 1)};
  for (const std::int64_t side : {-1, 1}) {
    particles[0].side_tag = side;
    for (std::size_t slot = 0; slot < 192; ++slot) {
      old_tags[slot].side_tag = -side;
      new_tags[slot].side_tag = side;
      old_walls[slot] = {true, side, {static_cast<double>(side), 0, 0}, {1 * static_cast<double>(side), 2, 3}};
      new_walls[slot] = {true, side, {static_cast<double>(side), 0, 0}, {4 * static_cast<double>(side), 5, 6}};
    }
    LiquidBoundaryGatherViewInputs3D inputs{
        grid->view(),
        particles,
        0,
        1,
        {0, 0, old, old_valid, old_tags, old_walls},
        {1, 1, next, new_valid, new_tags, new_walls},
    };
    auto expected = cpu->Gather(inputs);
    ASSERT_TRUE(expected.ok());
    auto result = cuda_internal::GatherLiquid3D(*arena, inputs, *workspace);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_NEAR(result->particles[0].new_pic_velocity.x, 4 * static_cast<double>(side), 2e-14);
    EXPECT_NEAR(result->particles[0].grid_velocity_increment.x, 3 * static_cast<double>(side), 2e-14);
    EXPECT_EQ(result->old_pic_fallback_particle_count, 1);
    EXPECT_EQ(result->new_pic_fallback_particle_count, 1);
    EXPECT_EQ(result->low_mass_fallback_particle_count, 1);
    EXPECT_FALSE(result->particles[0].affine_enabled);
    EXPECT_DOUBLE_EQ(result->particles[0].particle_minus_old_pic_velocity.x, 0);
    EXPECT_NEAR(result->particles[0].new_pic_velocity.y, expected->particles[0].new_pic_velocity.y, 2e-14);
    const auto* previous = result->particles.data();
    const double saved = previous[0].new_pic_velocity.x;
    inputs.new_history.wall_sample_time = 0;
    EXPECT_EQ(
        cuda_internal::GatherLiquid3D(*arena, inputs, *workspace).status().code(),
        absl::StatusCode::kInvalidArgument
    );
    EXPECT_DOUBLE_EQ(previous[0].new_pic_velocity.x, saved);
  }
}
TEST_F(CudaLiquidTransferStages3D, SingleAndGroupedGatherMatchSerialBitsAndFailureOrder) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(grid.ok());
  std::array<double, 192> old{}, next{};
  std::array<LiquidFaceValidity3D, 192> old_validity{}, new_validity{};
  old_validity.fill(LiquidFaceValidity3D::kValid);
  new_validity.fill(LiquidFaceValidity3D::kValid);
  old_validity[42] = LiquidFaceValidity3D::kMissing;
  new_validity[106] = LiquidFaceValidity3D::kMissing;
  std::array<LiquidBoundaryFaceTag3D, 192> tags{};
  std::array<LiquidPrescribedWallSample3D, 192> walls{};
  for (std::size_t slot = 0; slot < old.size(); ++slot) {
    const double face_slot = static_cast<double>(slot);
    old[slot] = 1 + face_slot * face_slot / 1024;
    next[slot] = -1 + face_slot * face_slot * face_slot / 65536;
  }
  const auto particle = TransferParticle({2, 2, 2}, {1, 2, 3});
  auto second_particle = particle;
  second_particle.transfer.trial_world_position.y += 0.25;
  const std::array particles{particle, second_particle};
  LiquidBoundaryGatherViewInputs3D inputs{
      grid->view(),
      std::span(particles).first(1),
      0,
      1,
      {0, 0, old, old_validity, tags, walls},
      {1, 1, next, new_validity, tags, walls},
  };
  auto single = cuda_internal::GatherLiquid3D(*arena, inputs, *workspace);
  ASSERT_TRUE(single.ok()) << single.status();
  ASSERT_EQ(single->particles.size(), 1);
  const auto gathered = single->particles[0];
  EXPECT_TRUE(gathered.affine_enabled);
  inputs.particles = particles;
  auto serial = cuda_internal::GatherLiquid3D(*arena, inputs, *workspace);
  ASSERT_TRUE(serial.ok()) << serial.status();
  ASSERT_EQ(serial->particles.size(), 2);
  const auto expect_vector_bits = [](Vec3 actual, Vec3 expected) {
    for (auto component : {&Vec3::x, &Vec3::y, &Vec3::z}) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.*component), std::bit_cast<std::uint64_t>(expected.*component));
    }
  };
  const auto expect_gather_bits = [&](const auto& gathered, const auto& expected) {
    expect_vector_bits(gathered.new_pic_velocity, expected.new_pic_velocity);
    expect_vector_bits(gathered.grid_velocity_increment, expected.grid_velocity_increment);
    expect_vector_bits(gathered.particle_minus_old_pic_velocity, expected.particle_minus_old_pic_velocity);
    for (auto row : {&Mat3::x_row, &Mat3::y_row, &Mat3::z_row}) {
      expect_vector_bits(gathered.fitted_velocity_slopes.*row, expected.fitted_velocity_slopes.*row);
      expect_vector_bits(gathered.velocity_gradient.*row, expected.velocity_gradient.*row);
    }
    EXPECT_EQ(gathered.affine_enabled, expected.affine_enabled);
    EXPECT_EQ(gathered.old_pic_fallback, expected.old_pic_fallback);
    EXPECT_EQ(gathered.new_pic_fallback, expected.new_pic_fallback);
    EXPECT_EQ(gathered.low_mass_fallback, expected.low_mass_fallback);
  };
  expect_gather_bits(gathered, serial->particles[0]);
  auto grouped = cuda_internal::GatherLiquidRun3D(*arena, inputs, *workspace);
  ASSERT_TRUE(grouped.ok()) << grouped.status();
  ASSERT_EQ(grouped->particles.size(), 2);
  EXPECT_EQ(grouped->old_pic_fallback_particle_count, serial->old_pic_fallback_particle_count);
  EXPECT_EQ(grouped->new_pic_fallback_particle_count, serial->new_pic_fallback_particle_count);
  for (std::size_t index = 0; index < particles.size(); ++index) {
    expect_gather_bits(grouped->particles[index], serial->particles[index]);
  }
  const auto saved_second_gather = grouped->particles[1];

  // Finite X samples overflow the X increment. Later Y samples have no donor or wall.
  old.fill(-0x1p1023);
  next.fill(0x1p1023);
  old_validity.fill(LiquidFaceValidity3D::kMissing);
  for (std::size_t slot = 0; slot < old.size(); ++slot) {
    const auto face = grid->view().FaceIndex(slot);
    ASSERT_TRUE(face.ok());
    if (face->axis == FaceNormalAxis::kX) {
      old_validity[slot] = LiquidFaceValidity3D::kValid;
    }
  }
  new_validity = old_validity;
  for (const std::size_t particle_count : {1, 2}) {
    inputs.particles = std::span(particles).first(particle_count);
    EXPECT_EQ(cuda_internal::GatherLiquid3D(*arena, inputs, *workspace).status().code(), absl::StatusCode::kOutOfRange);
  }
  EXPECT_EQ(
      cuda_internal::GatherLiquidRun3D(*arena, inputs, *workspace).status().code(),
      absl::StatusCode::kOutOfRange
  );
  expect_gather_bits(grouped->particles[1], saved_second_gather);
  old = next;
  for (const std::size_t particle_count : {1, 2}) {
    inputs.particles = std::span(particles).first(particle_count);
    EXPECT_EQ(
        cuda_internal::GatherLiquid3D(*arena, inputs, *workspace).status().code(),
        absl::StatusCode::kFailedPrecondition
    );
  }
  EXPECT_EQ(
      cuda_internal::GatherLiquidRun3D(*arena, inputs, *workspace).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  expect_gather_bits(grouped->particles[1], saved_second_gather);
}
TEST_F(CudaLiquidTransferStages3D, ContinuousSlabContactMatchesCpuOnBothSides) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  std::array<LiquidParticleState3D, 2> particles{};
  std::array<LiquidBoundaryParticleGather3D, 2> gathered{};
  std::array<double, 2> distances{-0.2, -0.2};
  for (std::size_t index = 0; index < 2; ++index) {
    const double side = index == 0 ? -1 : 1;
    particles[index].world_position = {0.3 * side, 0, 0};
    particles[index].pic_initialized = true;
    particles[index].id = index + 1;
    particles[index].side_tag = static_cast<std::int64_t>(side);
    gathered[index].new_pic_velocity = {-2 * side, 0.25, 0};
    gathered[index].particle_minus_old_pic_velocity = {-side, 1, 0};
    gathered[index].velocity_gradient.x_row.x = 0.1;
  }
  auto expected = particles;
  LiquidParticleUpdateInputs3D
      inputs{gathered, distances, TranslatingSlab3D{{1, 0, 0}, 0, 0.1, {0.2, 0, 0}}, 1, 0.15, 0.6, 0, 1e-10};
  auto cpu = CpuLiquidSubstepBackend3D();
  ASSERT_TRUE(cpu.update_particles(nullptr, inputs, expected).ok());
  ASSERT_TRUE(cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles).ok());
  for (std::size_t index = 0; index < 2; ++index) {
    EXPECT_NEAR(particles[index].world_position.x, expected[index].world_position.x, 2e-14);
    EXPECT_NEAR(particles[index].world_position.y, 0.0375, 2e-14);
    EXPECT_NEAR(particles[index].world_velocity.y, 0.844, 2e-14);
    EXPECT_NEAR(particles[index].world_velocity.x, 0.2, 2e-14);
    EXPECT_NEAR(particles[index].volume_ratio, expected[index].volume_ratio, 2e-14);
  }
  EXPECT_LT(particles[0].world_position.x, 0.03 - 0.1 + 1e-12);
  EXPECT_GT(particles[1].world_position.x, 0.03 + 0.1 - 1e-12);
}
TEST_F(CudaLiquidTransferStages3D, RegularCellTransportPairsAdvectionWithVolumeAndResetsStaging) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  const auto before = arena->metrics();
  test_internal::CheckRegularCellTransport([&](const auto& inputs, std::span<LiquidParticleState3D> particles) {
    return cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles);
  });
  ExpectNoReservationChange(before, arena->metrics());
}

TEST_F(CudaLiquidTransferStages3D, RegularCellTransportUsesSweptContact) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  test_internal::CheckRegularCellTransportContact([&](const auto& inputs, std::span<LiquidParticleState3D> particles) {
    return cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles);
  });
}

TEST_F(CudaLiquidTransferStages3D, RejectsMalformedRegularCellTransportBeforePublication) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  test_internal::CheckMalformedRegularCellTransport([&](const auto& inputs,
                                                        std::span<LiquidParticleState3D> particles) {
    return cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles);
  });
}

TEST_F(CudaLiquidTransferStages3D, RejectsRegularCellTransportAliasingOutputOrScratch) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  test_internal::CheckAliasedRegularCellTransport([&](const auto& inputs, std::span<LiquidParticleState3D> particles) {
    return cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles);
  });
  test_internal::RegularCellUpdateFixture3D fixture;
  const auto range = arena->range(CudaLiquidRange3D::kTransferParticles);
  auto* sample = reinterpret_cast<LiquidRegularCellTransport3D*>(range.pinned.data());
  sample[0] = fixture.transport[0];
  sample[1] = fixture.transport[1];
  fixture.inputs.regular_cell_transport = {sample, fixture.transport.size()};
  const auto initial = fixture.particles;
  EXPECT_EQ(
      cuda_internal::UpdateLiquidParticles3D(*arena, fixture.inputs, fixture.particles).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(fixture.particles, initial);
}
TEST_F(CudaLiquidTransferStages3D, RegularCellTransportPreservesParticleErrorOrder) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  test_internal::CheckRegularCellTransportErrorOrder([&](const auto& inputs,
                                                         std::span<LiquidParticleState3D> particles) {
    return cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles);
  });
}

TEST_F(CudaLiquidTransferStages3D, RegularCellTransportRejectsInvalidParticleInputsBeforeRetry) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  test_internal::CheckRegularCellTransportRejectsInvalidParticleInputs([&](const auto& inputs,
                                                                           std::span<LiquidParticleState3D> particles) {
    return cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles);
  });
}

TEST_F(CudaLiquidTransferStages3D, ParticleFallbackAndLateRejectionPreserveEveryRecord) {
  auto arena = CudaLiquidArena3D::Create(capacities, 0, std::numeric_limits<std::size_t>::max(), true);
  ASSERT_TRUE(arena.ok());
  std::array<LiquidParticleState3D, 2> particles{};
  std::array<LiquidBoundaryParticleGather3D, 2> gathered{};
  std::array<double, 2> distances{-0.5, -0.5};
  for (std::size_t i = 0; i < 2; ++i) {
    particles[i].world_position = {2, 2, 2};
    particles[i].pic_initialized = true;
    particles[i].id = i + 1;
    gathered[i].new_pic_velocity = {1, 0, 0};
    gathered[i].particle_minus_old_pic_velocity = {100, 0, 0};
    gathered[i].low_mass_fallback = true;
    gathered[i].affine_enabled = false;
  }
  arena->SetWorkPhase(LiquidWorkPhase3D::kGatherContactAcceptance);
  LiquidParticleUpdateInputs3D inputs{gathered, distances, {}, 1, 0.1, 1, 0, 1e-10};
  ASSERT_TRUE(cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles).ok());
  constexpr std::size_t particle_scratch_byte_budget =
      sizeof(cuda_internal::BoundaryTransferQuery3D) + sizeof(cuda_internal::BoundaryTransferReport3D) + 2 * 1024;
  EXPECT_LE(arena->metrics().peak_active_device_scratch_bytes, particle_scratch_byte_budget);
  EXPECT_DOUBLE_EQ(particles[0].world_velocity.x, 1);
  EXPECT_NEAR(particles[0].world_position.x, 2.1, 2e-14);
  EXPECT_FALSE(particles[0].affine_enabled);
  const auto previous = particles;
  gathered[1].velocity_gradient.x_row.x = 1e308;
  EXPECT_EQ(cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles).code(), absl::StatusCode::kAborted);
  EXPECT_EQ(particles, previous);
  gathered[1].velocity_gradient = {};
  gathered[1].new_pic_velocity = {10, 0, 0};
  EXPECT_EQ(cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles).code(), absl::StatusCode::kAborted);
  EXPECT_EQ(particles, previous);
}
TEST_F(CudaLiquidTransferStages3D, EmptyStagesAndRejectedOutputsHaveBoundedStorage) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  SparseMacGridView3D empty;
  EXPECT_TRUE(cuda_internal::DepositLiquid3D(*arena, {empty, {}, {}, {1, 1, 4}}, *workspace).ok());
  EXPECT_TRUE(cuda_internal::NormalizeLiquid3D(*arena, {}, {1, 1, 4}, 0, *workspace).ok());
  EXPECT_TRUE(cuda_internal::GatherLiquid3D(
                  *arena,
                  {empty, {}, 0, 1, {0, 0, {}, {}, {}, {}}, {1, 1, {}, {}, {}, {}}},
                  *workspace
  )
                  .ok());
  EXPECT_TRUE(cuda_internal::ApplyLiquidForces3D(*arena, empty, {}, {}, 0.1, {}).ok());
  EXPECT_TRUE(cuda_internal::UpdateLiquidParticles3D(*arena, {{}, {}, {}, 1, 0.1, 0, 0, 0}, {}).ok());
  std::array<MacFaceDepositQuanta3D, 193> excessive{};
  EXPECT_EQ(
      cuda_internal::NormalizeLiquid3D(*arena, excessive, {1, 1, 4}, 0, *workspace).status().code(),
      absl::StatusCode::kResourceExhausted
  );
  std::array<MacFaceDepositQuanta3D, 1> quanta{{{2, 4, 0}}};
  auto result = cuda_internal::NormalizeLiquid3D(*arena, quanta, {1, 1, 4}, 0, *workspace);
  ASSERT_TRUE(result.ok());
  const auto* previous = result->face_velocities.data();
  quanta[0].stress_force_quanta_sum = 1;
  EXPECT_EQ(
      cuda_internal::NormalizeLiquid3D(*arena, quanta, {1, 1, 4}, 0, *workspace).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_DOUBLE_EQ(previous[0], 2);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(grid.ok());
  std::array<double, 192> old{}, predicted{};
  predicted.fill(17);
  old.fill(1);
  old.back() = 1e308;
  EXPECT_EQ(
      cuda_internal::ApplyLiquidForces3D(*arena, grid->view(), old, {0, 0, 1e308}, 2, predicted).code(),
      absl::StatusCode::kOutOfRange
  );
  for (const auto value : predicted) {
    EXPECT_DOUBLE_EQ(value, 17);
  }
}
#if defined(_WIN32) && defined(_DEBUG)
TEST_F(CudaLiquidTransferStages3D, SuccessfulAndRejectedStagesAllocateAndFreeNothing) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(grid.ok());
  std::array<LiquidBoundaryFaceTag3D, 192> tags{};
  std::array transfer{TransferParticle({2, 2, 2}, {1, 2, 3}, 0)};
  std::array<double, 192> old{}, predicted{};
  old.fill(1);
  std::array<LiquidFaceValidity3D, 192> valid{};
  valid.fill(LiquidFaceValidity3D::kValid);
  std::array<LiquidPrescribedWallSample3D, 192> walls{};
  LiquidBoundaryVelocityHistory3D old_history{0, 0, old, valid, tags, walls};
  LiquidBoundaryVelocityHistory3D new_history{1, 1, predicted, valid, tags, walls};
  std::array<LiquidParticleState3D, 1> particles{};
  particles[0].world_position = {2, 2, 2};
  std::array<double, 1> distances{-0.5};
  const auto before = arena->metrics();
  std::array<bool, 9> checks{};
  arena_allocations = 0;
  arena_frees = 0;
  arena_previous_hook = _CrtSetAllocHook(ArenaAllocationHook);
  {
    auto deposit =
        cuda_internal::DepositLiquid3D(*arena, {grid->view(), transfer, tags, {0x1p-30, 0x1p-30, 4}}, *workspace);
    checks[0] = deposit.ok();
    if (deposit.ok()) {
      checks[1] =
          cuda_internal::NormalizeLiquid3D(*arena, deposit->face_quanta, {0x1p-30, 0x1p-30, 4}, 0, *workspace).ok();
    }
    checks[2] = cuda_internal::ApplyLiquidForces3D(*arena, grid->view(), old, {1, 0, 0}, 0.1, predicted).ok();
    checks[3] = cuda_internal::PredictMissingLiquidFace3D(
                    *arena,
                    {grid->view(), old_history, 42, 0, {1, 0, 0}, 0.1},
                    *workspace
    )
                    .ok();
    auto gather =
        cuda_internal::GatherLiquid3D(*arena, {grid->view(), transfer, 0, 1, old_history, new_history}, *workspace);
    checks[4] = gather.ok();
    if (gather.ok()) {
      checks[5] = cuda_internal::UpdateLiquidParticles3D(
                      *arena,
                      {gather->particles, distances, {}, 1, 0.1, 0.6, 0, 1e-10},
                      particles
      )
                      .ok();
    }
    transfer[0].transfer.particle_velocity.x = 1e100;
    checks[6] =
        cuda_internal::DepositLiquid3D(*arena, {grid->view(), transfer, tags, {1, 1, 4}}, *workspace).status().code()
        == absl::StatusCode::kOutOfRange;
    transfer[0].transfer.sorted_world_position = {0, 0, 0};
    transfer[0].transfer.trial_world_position = {0, 0, 0};
    checks[7] = !cuda_internal::DepositLiquid3D(*arena, {grid->view(), transfer, tags, {1, 1, 4}}, *workspace).ok();
    checks[8] = !cuda_internal::ApplyLiquidForces3D(*arena, grid->view(), old, {1, 0, 0}, 0.1, old).ok();
  }
  _CrtSetAllocHook(arena_previous_hook);
  for (const auto check : checks) {
    EXPECT_TRUE(check);
  }
  EXPECT_EQ(arena_allocations, 0);
  EXPECT_EQ(arena_frees, 0);
  ExpectNoReservationChange(before, arena->metrics());
}
#endif
}  // namespace
}  // namespace fluidity

namespace fluidity {
namespace {
TEST_F(CudaLiquidTransferStages3D, RejectsBorrowedScratchTopologyBeforeStaging) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  const auto range = arena->range(CudaLiquidRange3D::kTransferQuery);
  auto* block = reinterpret_cast<BlockIndex3D*>(range.pinned.data());
  *block = {0, 0, 0};
  auto grid = SparseMacGridView3D::Create({{0, 0, 0}, 1}, {block, 1});
  ASSERT_TRUE(grid.ok());
  std::array<double, 192> old{}, predicted{};
  predicted.fill(17);
  EXPECT_EQ(
      cuda_internal::ApplyLiquidForces3D(*arena, *grid, old, {1, 0, 0}, 0.1, predicted).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_DOUBLE_EQ(predicted[0], 17);
}
TEST_F(CudaLiquidTransferStages3D, RejectsInvalidObstacleOnEmptyUpdate) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  LiquidParticleUpdateInputs3D inputs{{}, {}, TranslatingSlab3D{{2, 0, 0}, 0, 0.1, {}}, 1, 0.1, 0, 0, 0};
  EXPECT_EQ(cuda_internal::UpdateLiquidParticles3D(*arena, inputs, {}).code(), absl::StatusCode::kInvalidArgument);
}
class TransferRuntimeFailure {
 public:
  explicit TransferRuntimeFailure(bool fail_fence) : saved_(cuda_internal::LiquidArenaRuntimeCalls()) {
    submissions_ = 0;
    fail_fence_ = fail_fence;
    cuda_internal::LiquidArenaRuntimeCalls().memcpy_async = Copy;
    cuda_internal::LiquidArenaRuntimeCalls().event_synchronize = Fence;
  }
  ~TransferRuntimeFailure() {
    cuda_internal::LiquidArenaRuntimeCalls() = saved_;
  }

 private:
  static cudaError_t CUDARTAPI
  Copy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    const auto result = cudaMemcpyAsync(destination, source, bytes, kind, stream);
    ++submissions_;
    return result == cudaSuccess && !fail_fence_ && submissions_ == 2 ? cudaErrorInvalidValue : result;
  }
  static cudaError_t CUDARTAPI Fence(cudaEvent_t event) {
    const auto result = cudaEventSynchronize(event);
    return result == cudaSuccess && fail_fence_ ? cudaErrorInvalidValue : result;
  }
  static inline int submissions_ = 0;
  static inline bool fail_fence_ = false;
  cuda_internal::LiquidArenaRuntimeCalls3D saved_;
};
TEST_F(CudaLiquidTransferStages3D, PartialUploadAndFenceFailurePoisonWithoutReusingPinnedInputs) {
  for (const bool fail_fence : {false, true}) {
    auto arena = CudaLiquidArena3D::Create(capacities);
    auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
    ASSERT_TRUE(arena.ok());
    ASSERT_TRUE(workspace.ok());
    std::array<MacFaceDepositQuanta3D, 1> quanta{{{2, 4, 0}}};
    auto published = cuda_internal::NormalizeLiquid3D(*arena, quanta, {1, 1, 4}, 0, *workspace);
    ASSERT_TRUE(published.ok());
    const auto range = arena->range(CudaLiquidRange3D::kTransferFaces);
    std::vector<std::byte> staged_bytes(range.pinned.size());
    const auto before = arena->metrics();
    {
      TransferRuntimeFailure fault(fail_fence);
      EXPECT_EQ(
          cuda_internal::NormalizeLiquid3D(*arena, quanta, {1, 1, 4}, 0, *workspace).status().code(),
          absl::StatusCode::kInternal
      );
      std::copy(range.pinned.begin(), range.pinned.end(), staged_bytes.begin());
      quanta[0].pure_momentum_quanta_sum = 99;
      EXPECT_EQ(
          cuda_internal::NormalizeLiquid3D(*arena, quanta, {1, 1, 4}, 0, *workspace).status().code(),
          absl::StatusCode::kInternal
      );
      EXPECT_EQ(std::memcmp(staged_bytes.data(), range.pinned.data(), staged_bytes.size()), 0);
    }
    EXPECT_DOUBLE_EQ(published->face_velocities[0], 2);
    ExpectNoReservationChange(before, arena->metrics());
  }
}
}  // namespace
}  // namespace fluidity

namespace fluidity {
namespace {
TEST_F(CudaLiquidTransferStages3D, SignedAffineGatherAndFullRankPredictionMatchAnalyticField) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
  auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, std::array{BlockIndex3D{-1, -1, -1}}, 1);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(grid.ok());
  std::array<double, 192> old{}, next{};
  std::array<LiquidFaceValidity3D, 192> validity{};
  validity.fill(LiquidFaceValidity3D::kValid);
  std::array<LiquidBoundaryFaceTag3D, 192> tags{};
  std::array<LiquidPrescribedWallSample3D, 192> walls{};
  for (std::size_t slot = 0; slot < 192; ++slot) {
    auto face = grid->view().FaceIndex(slot);
    ASSERT_TRUE(face.ok());
    auto anchor = grid->view().FaceWorldAnchor(*face);
    ASSERT_TRUE(anchor.ok());
    const double velocity = 1 + 2 * anchor->x + 3 * anchor->y - 4 * anchor->z;
    old[slot] = velocity;
    next[slot] = velocity + 0.5;
  }
  std::array particles{TransferParticle({-2, -2, -2}, {2, 2, 2}, 0)};
  LiquidBoundaryVelocityHistory3D history{0, 0, old, validity, tags, walls};
  auto gathered = cuda_internal::GatherLiquid3D(
      *arena,
      {grid->view(), particles, 0, 1, history, {1, 1, next, validity, tags, walls}},
      *workspace
  );
  ASSERT_TRUE(gathered.ok());
  EXPECT_NEAR(gathered->particles[0].new_pic_velocity.x, -0.5, 2e-14);
  EXPECT_NEAR(gathered->particles[0].grid_velocity_increment.x, 0.5, 2e-14);
  EXPECT_NEAR(gathered->particles[0].particle_minus_old_pic_velocity.x, 3, 2e-14);
  for (auto row : {&Mat3::x_row, &Mat3::y_row, &Mat3::z_row}) {
    const auto gradient = gathered->particles[0].velocity_gradient.*row;
    const auto slopes = gathered->particles[0].fitted_velocity_slopes.*row;
    EXPECT_NEAR(gradient.x, 2, 2e-14);
    EXPECT_NEAR(gradient.y, 3, 2e-14);
    EXPECT_NEAR(gradient.z, -4, 2e-14);
    EXPECT_NEAR(slopes.x, 2, 2e-14);
    EXPECT_NEAR(slopes.y, 3, 2e-14);
    EXPECT_NEAR(slopes.z, -4, 2e-14);
  }
  validity[42] = LiquidFaceValidity3D::kMissing;
  auto prediction =
      cuda_internal::PredictMissingLiquidFace3D(*arena, {grid->view(), history, 42, 0, {2, 0, 0}, 0.25}, *workspace);
  ASSERT_TRUE(prediction.ok());
  EXPECT_NEAR(*prediction, old[42] + 0.5, 2e-12);
}
TEST_F(CudaLiquidTransferStages3D, NewParticlesAndNonSurfaceExpansionUsePicAdvection) {
  auto arena = CudaLiquidArena3D::Create(capacities);
  ASSERT_TRUE(arena.ok());
  std::array<LiquidParticleState3D, 1> particles{};
  std::array<LiquidBoundaryParticleGather3D, 1> gathered{};
  std::array<double, 1> distance{-2};
  gathered[0].new_pic_velocity = {1, 0, 0};
  gathered[0].particle_minus_old_pic_velocity = {2, 0, 0};
  gathered[0].velocity_gradient.x_row.x = 0.5;
  LiquidParticleUpdateInputs3D inputs{gathered, distance, {}, 1, 0.1, 1, 0, 0};
  ASSERT_TRUE(cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles).ok());
  EXPECT_DOUBLE_EQ(particles[0].world_velocity.x, 1);
  EXPECT_NEAR(particles[0].world_position.x, 0.1, 2e-14);
  ASSERT_TRUE(cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles).ok());
  EXPECT_NEAR(particles[0].world_velocity.x, 2.98, 2e-14);
  EXPECT_NEAR(particles[0].world_position.x, 0.2, 2e-14);
  inputs.particle_energy_blend = 0;
  ASSERT_TRUE(cuda_internal::UpdateLiquidParticles3D(*arena, inputs, particles).ok());
  EXPECT_DOUBLE_EQ(particles[0].world_velocity.x, 1);
  EXPECT_NEAR(particles[0].world_position.x, 0.3, 2e-14);
}
}  // namespace
}  // namespace fluidity
