#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include <iomanip>
#include <iostream>
#include <limits>

#include "../cuda/liquid_arena_3d_internal.h"
#include "fluidity/host/cuda_liquid_substep_3d.h"
#include "gtest/gtest.h"
#include "liquid_tank_staging_3d_cases.h"
#include "liquid_wall_sample_3d_cases.h"
namespace fluidity {
namespace {
// Declared before implementation: retain FP64 particles and existing FP32 PCG.
constexpr double kPressureTolerance = 2e-5;
constexpr double kVelocityTolerance = 2e-8;
constexpr double kPositionTolerance = 2e-10;
LiquidState3D Initial() {
  LiquidState3D state;
  state.particles = *LiquidRecords3D<LiquidParticleState3D>::Create(2);
  EXPECT_TRUE(state.particles.Resize(2).ok());
  state.next_particle_id = 3;
  for (std::size_t index = 0; index < 2; ++index) {
    state.particles[index].id = 2 - index;
    state.particles[index].world_position = {2.5, 2.5, 2.5};
    state.particles[index].world_velocity = {0.1, 0, 0};
    state.particles[index].pic_initialized = true;
  }
  return state;
}
LiquidStepOptions3D Options() {
  LiquidStepOptions3D options;
  options.capacities = {16, 64, 128, 12288, 1};
  return options;
}
LiquidSceneInputs3D Scene() {
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 1.1};
  scene.surface_tension = 0.1;
  scene.ambient_pressure = 3;
  scene.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {0.1, 0, 0}};
  return scene;
}
double MaximumComponentError(Vec3 reference, Vec3 measured) {
  return std::max({
      std::abs(reference.x - measured.x),
      std::abs(reference.y - measured.y),
      std::abs(reference.z - measured.z),
  });
}

void ExpectState(const LiquidState3D& cpu, const LiquidState3D& gpu) {
  EXPECT_EQ(cpu.clock, gpu.clock);
  EXPECT_EQ(cpu.next_particle_id, gpu.next_particle_id);
  EXPECT_EQ(cpu.random_seed, gpu.random_seed);
  EXPECT_EQ(cpu.geometry.world_origin.x, gpu.geometry.world_origin.x);
  EXPECT_EQ(cpu.geometry.world_origin.y, gpu.geometry.world_origin.y);
  EXPECT_EQ(cpu.geometry.world_origin.z, gpu.geometry.world_origin.z);
  EXPECT_EQ(cpu.geometry.cell_width, gpu.geometry.cell_width);
  EXPECT_EQ(cpu.sources, gpu.sources);
  ASSERT_EQ(cpu.particles.size(), gpu.particles.size());
  ASSERT_EQ(cpu.blocks.size(), gpu.blocks.size());
  ASSERT_EQ(cpu.cells.size(), gpu.cells.size());
  ASSERT_EQ(cpu.faces.size(), gpu.faces.size());
  double max_world_position_error = 0;
  double max_world_velocity_error = 0;
  double max_pressure_error = 0;
  double max_carrier_velocity_error = 0;
  for (std::size_t index = 0; index < cpu.particles.size(); ++index) {
    const auto& cpu_particle = cpu.particles[index];
    const auto& cuda_particle = gpu.particles[index];
    EXPECT_EQ(cpu_particle.id, cuda_particle.id);
    EXPECT_EQ(cpu_particle.tag, cuda_particle.tag);
    const std::array cpu_velocity_slopes{
        cpu_particle.velocity_slopes.x_row,
        cpu_particle.velocity_slopes.y_row,
        cpu_particle.velocity_slopes.z_row,
    };
    const std::array cuda_velocity_slopes{
        cuda_particle.velocity_slopes.x_row,
        cuda_particle.velocity_slopes.y_row,
        cuda_particle.velocity_slopes.z_row,
    };
    for (std::size_t axis = 0; axis < cpu_velocity_slopes.size(); ++axis) {
      EXPECT_NEAR(cpu_velocity_slopes[axis].x, cuda_velocity_slopes[axis].x, kVelocityTolerance);
      EXPECT_NEAR(cpu_velocity_slopes[axis].y, cuda_velocity_slopes[axis].y, kVelocityTolerance);
      EXPECT_NEAR(cpu_velocity_slopes[axis].z, cuda_velocity_slopes[axis].z, kVelocityTolerance);
    }
    EXPECT_EQ(cpu_particle.mass, cuda_particle.mass);
    EXPECT_EQ(cpu_particle.reference_volume, cuda_particle.reference_volume);
    EXPECT_EQ(cpu_particle.side_tag, cuda_particle.side_tag);
    EXPECT_EQ(cpu_particle.affine_enabled, cuda_particle.affine_enabled);
    EXPECT_EQ(cpu_particle.pic_initialized, cuda_particle.pic_initialized);
    EXPECT_NEAR(cpu_particle.world_position.x, cuda_particle.world_position.x, kPositionTolerance);
    EXPECT_NEAR(cpu_particle.world_position.y, cuda_particle.world_position.y, kPositionTolerance);
    EXPECT_NEAR(cpu_particle.world_position.z, cuda_particle.world_position.z, kPositionTolerance);
    EXPECT_NEAR(cpu_particle.world_velocity.x, cuda_particle.world_velocity.x, kVelocityTolerance);
    EXPECT_NEAR(cpu_particle.world_velocity.y, cuda_particle.world_velocity.y, kVelocityTolerance);
    EXPECT_NEAR(cpu_particle.world_velocity.z, cuda_particle.world_velocity.z, kVelocityTolerance);
    EXPECT_NEAR(cpu_particle.volume_ratio, cuda_particle.volume_ratio, kPositionTolerance);
    max_world_position_error = std::max(
        max_world_position_error,
        MaximumComponentError(cpu_particle.world_position, cuda_particle.world_position)
    );
    max_world_velocity_error = std::max(
        max_world_velocity_error,
        MaximumComponentError(cpu_particle.world_velocity, cuda_particle.world_velocity)
    );
  }
  for (std::size_t index = 0; index < cpu.blocks.size(); ++index) {
    EXPECT_EQ(cpu.blocks[index], gpu.blocks[index]);
  }
  for (std::size_t index = 0; index < cpu.cells.size(); ++index) {
    EXPECT_EQ(cpu.cells[index].cell_index.x, gpu.cells[index].cell_index.x);
    EXPECT_EQ(cpu.cells[index].cell_index.y, gpu.cells[index].cell_index.y);
    EXPECT_EQ(cpu.cells[index].cell_index.z, gpu.cells[index].cell_index.z);
    EXPECT_EQ(cpu.cells[index].occupancy, gpu.cells[index].occupancy);
    EXPECT_NEAR(cpu.cells[index].previous_pressure, gpu.cells[index].previous_pressure, kPressureTolerance);
    max_pressure_error =
        std::max(max_pressure_error, std::abs(cpu.cells[index].previous_pressure - gpu.cells[index].previous_pressure));
  }
  for (std::size_t index = 0; index < cpu.faces.size(); ++index) {
    EXPECT_EQ(cpu.faces[index].face_index.axis, gpu.faces[index].face_index.axis);
    EXPECT_EQ(cpu.faces[index].face_index.lattice_index.x, gpu.faces[index].face_index.lattice_index.x);
    EXPECT_EQ(cpu.faces[index].face_index.lattice_index.y, gpu.faces[index].face_index.lattice_index.y);
    EXPECT_EQ(cpu.faces[index].face_index.lattice_index.z, gpu.faces[index].face_index.lattice_index.z);
    EXPECT_EQ(cpu.faces[index].face_slot, gpu.faces[index].face_slot);
    EXPECT_EQ(cpu.faces[index].side_tag, gpu.faces[index].side_tag);
    EXPECT_EQ(cpu.faces[index].structurally_eligible, gpu.faces[index].structurally_eligible);
    EXPECT_EQ(cpu.faces[index].validity, gpu.faces[index].validity);
    EXPECT_NEAR(cpu.faces[index].carrier_velocity, gpu.faces[index].carrier_velocity, kVelocityTolerance);
    max_carrier_velocity_error = std::max(
        max_carrier_velocity_error,
        std::abs(cpu.faces[index].carrier_velocity - gpu.faces[index].carrier_velocity)
    );
  }
  std::cout << std::scientific << std::setprecision(9) << "f9_max_world_position_error_m=" << max_world_position_error
            << " f9_max_world_velocity_error_m_s=" << max_world_velocity_error
            << " f9_max_pressure_error_pa=" << max_pressure_error
            << " f9_max_carrier_velocity_error_m_s=" << max_carrier_velocity_error << '\n';
}
void Receipt(const char* name, CudaLiquidSubstepUsage3D usage) {
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  EXPECT_EQ(properties.major, 12);
  EXPECT_EQ(properties.minor, 0);
#if defined(FLUIDITY_QUALIFIED_CUDA_DEVICE)
  EXPECT_STREQ(properties.name, FLUIDITY_QUALIFIED_CUDA_DEVICE);
#endif
  EXPECT_EQ(usage.compiled_architecture, 1200);
  std::cout << "scheduler_device_bytes=" << usage.device_reserved_bytes
            << " scheduler_pinned_bytes=" << usage.pinned_reserved_bytes
            << " scheduler_event_creations=" << usage.arena_metrics.event_creations << '\n';
  EXPECT_TRUE(std::isfinite(usage.substep_device_timeline_ms));
  EXPECT_GT(usage.substep_device_timeline_ms, 0);
  EXPECT_GT(usage.graph_replay_count, 0);
  EXPECT_GT(usage.device_reserved_bytes, 0);
  EXPECT_LE(usage.device_reserved_bytes, usage.device_memory_ceiling_bytes);
  std::cout << "operator=" << name << " compiled_architecture=" << usage.compiled_architecture
            << " substep_device_timeline_ms=" << std::fixed << std::setprecision(9) << usage.substep_device_timeline_ms
            << " timing_scope=topology_through_publication\n";
}

void ExpectNoOwnershipCalls(const CudaLiquidArenaMetrics3D& before, const CudaLiquidArenaMetrics3D& after) {
  EXPECT_EQ(before.device_reservations, after.device_reservations);
  EXPECT_EQ(before.device_frees, after.device_frees);
  EXPECT_EQ(before.pinned_reservations, after.pinned_reservations);
  EXPECT_EQ(before.pinned_frees, after.pinned_frees);
  EXPECT_EQ(before.stream_creations, after.stream_creations);
  EXPECT_EQ(before.stream_destructions, after.stream_destructions);
  EXPECT_EQ(before.event_creations, after.event_creations);
  EXPECT_EQ(before.event_destructions, after.event_destructions);
  EXPECT_EQ(before.graph_creations, after.graph_creations);
  EXPECT_EQ(before.graph_destructions, after.graph_destructions);
  EXPECT_EQ(before.executable_creations, after.executable_creations);
  EXPECT_EQ(before.executable_destructions, after.executable_destructions);
}
std::vector<std::byte> DeviceBytes(const CudaLiquidEpochLease3D& lease) {
  std::vector<std::byte> bytes(lease.device_bytes().size());
  EXPECT_EQ(cudaMemcpy(bytes.data(), lease.device_bytes().data(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  return bytes;
}
#if defined(_WIN32) && defined(_DEBUG)
std::size_t attempt_allocations = 0;
std::size_t attempt_frees = 0;
_CRT_ALLOC_HOOK previous_allocation_hook = nullptr;
int AttemptAllocationHook(
    int operation,
    void* data,
    std::size_t size,
    int type,
    long request,
    const unsigned char* filename,
    int line
) {
  if (operation == _HOOK_ALLOC || operation == _HOOK_REALLOC) {
    ++attempt_allocations;
  }
  if (operation == _HOOK_FREE) {
    ++attempt_frees;
  }
  return previous_allocation_hook ? previous_allocation_hook(operation, data, size, type, request, filename, line) : 1;
}
#endif
// Enclose the complete attempt AND result destruction; assertions run afterward.
void MeasuredAttempt(
    LiquidSubstepScheduler3D& scheduler,
    CudaLiquidSubstepBackend3D& backend,
    double duration,
    LiquidAttemptDisposition3D expected
) {
  const auto before = backend.usage().arena_metrics;
  bool succeeded = false;
  auto disposition = LiquidAttemptDisposition3D::kFatal;
#if defined(_WIN32) && defined(_DEBUG)
  attempt_allocations = 0;
  attempt_frees = 0;
  previous_allocation_hook = _CrtSetAllocHook(AttemptAllocationHook);
#endif
  {
    const auto result = scheduler.Attempt(duration);
    succeeded = result.ok();
    if (succeeded) {
      disposition = result->disposition;
    }
  }
#if defined(_WIN32) && defined(_DEBUG)
  _CrtSetAllocHook(previous_allocation_hook);
  EXPECT_EQ(attempt_allocations, 0);
  EXPECT_EQ(attempt_frees, 0);
#endif
  ASSERT_TRUE(succeeded);
  EXPECT_EQ(disposition, expected);
  ExpectNoOwnershipCalls(before, backend.usage().arena_metrics);
}
struct HistorySnapshot {
  explicit HistorySnapshot(std::size_t count) : velocities(count), validity(count), tags(count), walls(count) {
  }
  std::vector<double> velocities;
  std::vector<LiquidFaceValidity3D> validity;
  std::vector<LiquidBoundaryFaceTag3D> tags;
  std::vector<LiquidPrescribedWallSample3D> walls;
};
struct StageAudit {
  explicit StageAudit(std::size_t count) : quanta(count), old_history(count), new_history(count) {
  }
  LiquidSubstepBackend3D actual;
  std::vector<MacFaceDepositQuanta3D> quanta;
  HistorySnapshot old_history;
  HistorySnapshot new_history;
  std::size_t face_count = 0;
  std::size_t gather_count = 0;
  std::size_t mismatches = 0;
  bool compare = false;
  bool signed_momentum = false;
  bool histories_differ = false;
  bool eligibility_changed = false;
};
thread_local StageAudit* stage_audit = nullptr;
void AuditHistory(
    const LiquidBoundaryVelocityHistory3D& history,
    HistorySnapshot& saved,
    bool compare,
    std::size_t& mismatches
) {
  for (std::size_t index = 0; index < history.face_velocities.size(); ++index) {
    if (compare) {
      const auto& wall = history.wall_samples[index];
      const auto& previous_wall = saved.walls[index];
      mismatches += std::abs(saved.velocities[index] - history.face_velocities[index]) > kVelocityTolerance;
      mismatches += saved.validity[index] != history.validity[index];
      mismatches += saved.tags[index].side_tag != history.face_tags[index].side_tag;
      mismatches += saved.tags[index].structurally_eligible != history.face_tags[index].structurally_eligible;
      mismatches += previous_wall.available != wall.available || previous_wall.side_tag != wall.side_tag;
      if (wall.available) {
        mismatches += previous_wall.wall_velocity.x != wall.wall_velocity.x;
        mismatches += previous_wall.wall_velocity.y != wall.wall_velocity.y;
        mismatches += previous_wall.wall_velocity.z != wall.wall_velocity.z;
        mismatches += previous_wall.unit_normal.x != wall.unit_normal.x;
        mismatches += previous_wall.unit_normal.y != wall.unit_normal.y;
        mismatches += previous_wall.unit_normal.z != wall.unit_normal.z;
      }
    } else {
      saved.velocities[index] = history.face_velocities[index];
      saved.validity[index] = history.validity[index];
      saved.tags[index] = history.face_tags[index];
      saved.walls[index] = history.wall_samples[index];
    }
  }
}
LiquidSubstepBackend3D AuditedOperations(StageAudit& audit, LiquidSubstepBackend3D actual) {
  audit.actual = actual;
  actual.deposit =
      [](void* context, const LiquidBoundaryDepositViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
        auto& audit = *stage_audit;
        auto result = audit.actual.deposit(context, inputs, workspace);
        if (result.ok()) {
          if (audit.compare) {
            audit.mismatches += audit.face_count != result->face_quanta.size();
          } else {
            audit.face_count = result->face_quanta.size();
          }
          for (std::size_t index = 0; index < result->face_quanta.size(); ++index) {
            const auto& quanta = result->face_quanta[index];
            audit.signed_momentum |= quanta.pure_momentum_quanta_sum < 0;
            if (audit.compare) {
              audit.mismatches += audit.quanta[index].mass_quanta_sum != quanta.mass_quanta_sum;
              audit.mismatches += audit.quanta[index].pure_momentum_quanta_sum != quanta.pure_momentum_quanta_sum;
              audit.mismatches += audit.quanta[index].stress_force_quanta_sum != quanta.stress_force_quanta_sum;
            } else {
              audit.quanta[index] = quanta;
            }
          }
        }
        return result;
      };
  actual.gather = [](void* context, const LiquidBoundaryGatherViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace
                  ) {
    auto& audit = *stage_audit;
    ++audit.gather_count;
    audit.mismatches += inputs.old_time != 0 || inputs.new_time != 0.01;
    audit.mismatches += inputs.old_history.velocity_sample_time != inputs.old_time;
    audit.mismatches += inputs.old_history.wall_sample_time != inputs.old_time;
    audit.mismatches += inputs.new_history.velocity_sample_time != inputs.new_time;
    audit.mismatches += inputs.new_history.wall_sample_time != inputs.new_time;
    AuditHistory(inputs.old_history, audit.old_history, audit.compare, audit.mismatches);
    AuditHistory(inputs.new_history, audit.new_history, audit.compare, audit.mismatches);
    for (std::size_t index = 0; index < inputs.old_history.face_velocities.size(); ++index) {
      audit.eligibility_changed |= inputs.old_history.face_tags[index].structurally_eligible
          != inputs.new_history.face_tags[index].structurally_eligible;
      audit.histories_differ |= inputs.old_history.face_velocities[index] != inputs.new_history.face_velocities[index];
    }
    return audit.actual.gather(context, inputs, workspace);
  };
  return actual;
}
void CheckSignedQuantaAndHistories(bool moving_wall) {
  auto initial = Initial();
  initial.particles[0].world_velocity = {0.1, -0.05, 0.02};
  initial.particles[1].world_velocity = {-0.2, -0.05, 0.02};
  auto options = Options();
  auto scene = Scene();
  if (moving_wall) {
    scene.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 1.995, {1, 0, 0}};
  }
  StageAudit audit(options.capacities.faces);
  stage_audit = &audit;
  auto cpu =
      LiquidSubstepScheduler3D::Create(initial, scene, options, AuditedOperations(audit, CpuLiquidSubstepBackend3D()));
  ASSERT_TRUE(cpu.ok());
  ASSERT_TRUE(cpu->AdvanceFrame(0.01).ok());
  EXPECT_TRUE(audit.signed_momentum);
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0, true);
  ASSERT_TRUE(backend.ok());
  audit.compare = true;
  audit.gather_count = 0;
  auto gpu = LiquidSubstepScheduler3D::Create(initial, scene, options, AuditedOperations(audit, backend->operations()));
  ASSERT_TRUE(gpu.ok());
  ASSERT_TRUE(gpu->BeginFrame(0.01).ok());
  MeasuredAttempt(*gpu, *backend, 0.01, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_TRUE(gpu->FinishFrame().ok());
  const auto events = backend->usage().arena_metrics;
  EXPECT_GT(events.pre_topology_field_events.interval_count, 0);
  EXPECT_GT(events.pre_topology_field_events.seconds, 0);
  for (const auto& phase : events.work_phase_events) {
    EXPECT_GT(phase.interval_count, 0);
    EXPECT_GT(phase.seconds, 0);
  }
  EXPECT_EQ(audit.mismatches, 0);
  EXPECT_EQ(audit.gather_count, 2);
  EXPECT_TRUE(audit.histories_differ);
  EXPECT_TRUE(audit.eligibility_changed);
  ExpectState(cpu->committed(), gpu->committed());
  stage_audit = nullptr;
}
void CheckReplayMoveAndEmpty() {
  auto initial = Initial();
  auto options = Options();
  std::vector<std::byte> replay_bytes;
  for (int replay = 0; replay < 2; ++replay) {
    auto created = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
    ASSERT_TRUE(created.ok());
    auto gpu = LiquidSubstepScheduler3D::Create(initial, Scene(), options, created->operations());
    ASSERT_TRUE(gpu.ok());
    // The scheduler retains the context created before BOTH moves.
    auto moved = std::move(*created);
    auto backend = std::move(moved);
    ASSERT_TRUE(gpu->BeginFrame(0.01).ok());
    MeasuredAttempt(*gpu, backend, 0.01, LiquidAttemptDisposition3D::kAccepted);
    ASSERT_TRUE(gpu->FinishFrame().ok());
    auto lease = backend.AcquireCommitted();
    ASSERT_TRUE(lease.ok());
    EXPECT_EQ(lease->identity().clock, gpu->committed().clock);
    auto readback = CloneLiquidState3D(initial, options.capacities);
    ASSERT_TRUE(readback.ok());
    ASSERT_TRUE(lease->Readback(*readback).ok());
    EXPECT_EQ(*readback, gpu->committed());
    auto bytes = DeviceBytes(*lease);
    if (replay == 0) {
      replay_bytes = std::move(bytes);
    } else {
      EXPECT_EQ(bytes, replay_bytes);
    }
  }
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{1};
  LiquidState3D empty;
  auto cpu = LiquidSubstepScheduler3D::Create(empty, scene, options);
  ASSERT_TRUE(cpu.ok());
  ASSERT_TRUE(cpu->AdvanceFrame(0.01).ok());
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
  ASSERT_TRUE(backend.ok());
  auto gpu = LiquidSubstepScheduler3D::Create(empty, scene, options, backend->operations());
  ASSERT_TRUE(gpu.ok());
  const auto before = backend->usage();
  ASSERT_TRUE(gpu->BeginFrame(0.01).ok());
  MeasuredAttempt(*gpu, *backend, 0.01, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_TRUE(gpu->FinishFrame().ok());
  EXPECT_EQ(cpu->committed(), gpu->committed());
  EXPECT_EQ(backend->usage().graph_replay_count, before.graph_replay_count + 1);
  EXPECT_GT(backend->usage().substep_device_timeline_ms, 0);
}
void CheckRejectedAttempts() {
  for (const auto code :
       {absl::StatusCode::kAborted, absl::StatusCode::kResourceExhausted, absl::StatusCode::kInternal}) {
    auto initial = Initial();
    auto options = Options();
    auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
    ASSERT_TRUE(backend.ok());
    auto gpu = LiquidSubstepScheduler3D::Create(initial, Scene(), options, backend->operations());
    ASSERT_TRUE(gpu.ok());
    auto lease = backend->AcquireCommitted();
    ASSERT_TRUE(lease.ok());
    const auto original_identity = lease->identity();
    const auto bytes = DeviceBytes(*lease);
    auto failure = code;
    gpu->SetStageFaultHookForTesting({&failure, [](void* context, LiquidPhysicalStage3D stage, double) {
                                        return stage == LiquidPhysicalStage3D::kAccept
                                            ? absl::Status(*static_cast<absl::StatusCode*>(context), {})
                                            : absl::OkStatus();
                                      }});
    ASSERT_TRUE(gpu->BeginFrame(0.01).ok());
    const auto disposition = code == absl::StatusCode::kAborted ? LiquidAttemptDisposition3D::kRetryDt
        : code == absl::StatusCode::kResourceExhausted          ? LiquidAttemptDisposition3D::kCapacity
                                                                : LiquidAttemptDisposition3D::kFatal;
    MeasuredAttempt(*gpu, *backend, 0.01, disposition);
    EXPECT_EQ(gpu->committed(), initial);
    EXPECT_EQ(DeviceBytes(*lease), bytes);
    auto current = backend->AcquireCommitted();
    ASSERT_TRUE(current.ok());
    EXPECT_EQ(current->identity(), original_identity);
    EXPECT_EQ(gpu->diagnostics().failure_code, code);
    if (code == absl::StatusCode::kAborted) {
      gpu->SetStageFaultHookForTesting({});
      MeasuredAttempt(*gpu, *backend, 0.005, LiquidAttemptDisposition3D::kAccepted);
      auto accepted = backend->AcquireCommitted();
      ASSERT_TRUE(accepted.ok());
      EXPECT_GT(accepted->identity().generation, original_identity.generation);
      EXPECT_EQ(accepted->identity().clock, gpu->committed().clock);
    }
  }
}
struct RuntimeFault {
  cuda_internal::LiquidArenaRuntimeCalls3D previous = cuda_internal::LiquidArenaRuntimeCalls();
  static inline std::size_t copies = 0;
  static cudaError_t CUDARTAPI
  Copy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    ++copies;
    const auto result = cudaMemcpyAsync(destination, source, bytes, kind, stream);
    return result == cudaSuccess ? cudaErrorUnknown : result;
  }
  static cudaError_t CUDARTAPI Clear(void* destination, int value, std::size_t bytes, cudaStream_t stream) {
    const auto result = cudaMemsetAsync(destination, value, bytes, stream);
    return result == cudaSuccess ? cudaErrorUnknown : result;
  }
  static cudaError_t CUDARTAPI Event(cudaEvent_t event, cudaStream_t stream) {
    const auto result = cudaEventRecord(event, stream);
    return result == cudaSuccess ? cudaErrorUnknown : result;
  }
  ~RuntimeFault() {
    cuda_internal::LiquidArenaRuntimeCalls() = previous;
  }
};
void CheckRuntimeFailure() {
  for (const bool publication_failure : {false, true}) {
    auto initial = Initial();
    const auto options = Options();
    auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
    ASSERT_TRUE(backend.ok());
    auto gpu = LiquidSubstepScheduler3D::Create(initial, Scene(), options, backend->operations());
    ASSERT_TRUE(gpu.ok());
    auto lease = backend->AcquireCommitted();
    ASSERT_TRUE(lease.ok());
    const auto original_identity = lease->identity();
    const auto original_bytes = DeviceBytes(*lease);
    ASSERT_TRUE(gpu->BeginFrame(0.01).ok());
    RuntimeFault fault;
    RuntimeFault::copies = 0;
    if (publication_failure) {
      gpu->SetStageFaultHookForTesting({nullptr, [](void*, LiquidPhysicalStage3D stage, double) {
                                          if (stage == LiquidPhysicalStage3D::kAccept) {
                                            cuda_internal::LiquidArenaRuntimeCalls().event_record = RuntimeFault::Event;
                                          }
                                          return absl::OkStatus();
                                        }});
    } else {
      cuda_internal::LiquidArenaRuntimeCalls().memcpy_async = RuntimeFault::Copy;
    }
    MeasuredAttempt(*gpu, *backend, 0.01, LiquidAttemptDisposition3D::kFatal);
    EXPECT_EQ(gpu->diagnostics().failure_code, absl::StatusCode::kInternal);
    EXPECT_EQ(gpu->committed(), initial);
    EXPECT_EQ(lease->identity(), original_identity);
    EXPECT_EQ(DeviceBytes(*lease), original_bytes);
    if (!publication_failure) {
      EXPECT_EQ(RuntimeFault::copies, 1);
    }
    const auto calls_after_failure = RuntimeFault::copies;
    const auto operations = backend->operations();
    EXPECT_EQ(
        operations.sample_field(operations.context, Scene().field, initial.particles.values(), {2, 2, 2})
            .status()
            .code(),
        absl::StatusCode::kInternal
    );
    EXPECT_EQ(RuntimeFault::copies, calls_after_failure);
  }
}

void CheckGrowthLeasesAndRollback() {
  for (const bool fail_growth : {false, true}) {
    auto initial = Initial();
    auto options = Options();
    options.maximum_step_duration = 0.005;
    options.capacity_replay_reservations = options.capacities;
    options.capacity_replay_reservations.particles = 32;
    auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
    ASSERT_TRUE(backend.ok());
    auto gpu = LiquidSubstepScheduler3D::Create(initial, Scene(), options, backend->operations());
    ASSERT_TRUE(gpu.ok());
    auto initial_lease = backend->AcquireCommitted();
    ASSERT_TRUE(initial_lease.ok());
    const auto initial_bytes = DeviceBytes(*initial_lease);
    const auto before = backend->usage();
    std::vector<std::byte> accepted_bytes(initial_bytes.size());
    auto accepted_state = CloneLiquidState3D(initial, options.capacities);
    ASSERT_TRUE(accepted_state.ok());
    struct GrowthObservation {
      CudaLiquidSubstepBackend3D* backend;
      CudaLiquidEpochLease3D accepted_lease;
      std::span<std::byte> accepted_bytes;
      LiquidState3D* accepted_state;
      int emissions = 0;
      bool fail_growth;
      bool acquired = false;
    } observed{
        .backend = &*backend,
        .accepted_bytes = accepted_bytes,
        .accepted_state = &*accepted_state,
        .fail_growth = fail_growth,
    };
    RuntimeFault fault;
    gpu->SetStageFaultHookForTesting(
        {&observed,
         [](void* context, LiquidPhysicalStage3D stage, double) {
           auto& observed = *static_cast<GrowthObservation*>(context);
           if (stage == LiquidPhysicalStage3D::kEmit && ++observed.emissions == 2) {
             auto lease = observed.backend->AcquireCommitted();
             observed.acquired = lease.ok();
             if (!lease.ok()) {
               return lease.status();
             }
             // Empty destination: no owner release occurs inside the attempt.
             observed.accepted_lease = std::move(*lease);
             const auto snapshot = observed.accepted_lease.Readback(*observed.accepted_state);
             if (!snapshot.ok()) {
               return snapshot;
             }
             const auto source = observed.accepted_lease.device_bytes();
             if (source.size() != observed.accepted_bytes.size()
                 || cudaMemcpy(observed.accepted_bytes.data(), source.data(), source.size(), cudaMemcpyDeviceToHost)
                     != cudaSuccess) {
               return absl::Status(absl::StatusCode::kInternal, {});
             }
             if (observed.fail_growth) {
               cuda_internal::LiquidArenaRuntimeCalls().memset_async = RuntimeFault::Clear;
             }
             return absl::Status(absl::StatusCode::kResourceExhausted, {});
           }
           return absl::OkStatus();
         }}
    );
    const auto result = gpu->AdvanceFrame(0.01);
    EXPECT_TRUE(observed.acquired);
    EXPECT_GT(observed.accepted_lease.identity().generation, initial_lease->identity().generation);
    EXPECT_EQ(DeviceBytes(*initial_lease), initial_bytes);
    EXPECT_EQ(observed.accepted_lease.identity().clock.time, 0.005);
    EXPECT_EQ(DeviceBytes(observed.accepted_lease), accepted_bytes);
    if (fail_growth) {
      EXPECT_EQ(result.code(), absl::StatusCode::kInternal);
      EXPECT_EQ(gpu->committed().clock, observed.accepted_lease.identity().clock);
      EXPECT_EQ(gpu->committed(), *accepted_state);
      auto reference = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
      ASSERT_TRUE(reference.ok());
      ASSERT_TRUE(reference->BeginFrame(0.01).ok());
      ASSERT_TRUE(reference->Attempt(0.005).ok());
      ExpectState(reference->committed(), gpu->committed());
      EXPECT_EQ(backend->usage().device_reserved_bytes, before.device_reserved_bytes);
      EXPECT_EQ(DeviceBytes(observed.accepted_lease), accepted_bytes);
      continue;
    }
    ASSERT_TRUE(result.ok());
    auto final_lease = backend->AcquireCommitted();
    ASSERT_TRUE(final_lease.ok());
    EXPECT_GT(final_lease->identity().generation, observed.accepted_lease.identity().generation);
    EXPECT_EQ(final_lease->identity().clock, gpu->committed().clock);
    EXPECT_EQ(gpu->diagnostics().accepted_attempts, 3);
    EXPECT_EQ(gpu->committed().clock.accepted_step_ordinal, 2);
    const auto after_growth = backend->usage();
    EXPECT_GT(after_growth.device_reserved_bytes, before.device_reserved_bytes);
    EXPECT_GE(after_growth.arena_metrics.peak_device_bytes, after_growth.device_reserved_bytes);
    backend->CollectRetired();
    EXPECT_EQ(backend->usage().device_reserved_bytes, after_growth.device_reserved_bytes);
    EXPECT_EQ(DeviceBytes(observed.accepted_lease), accepted_bytes);
    auto grown_options = options;
    grown_options.capacities = options.capacity_replay_reservations;
    grown_options.capacity_replay_reservations = {};
    auto direct_backend = CudaLiquidSubstepBackend3D::Create(grown_options.capacities, 0);
    ASSERT_TRUE(direct_backend.ok());
    auto direct = LiquidSubstepScheduler3D::Create(initial, Scene(), grown_options, direct_backend->operations());
    ASSERT_TRUE(direct.ok());
    ASSERT_TRUE(direct->AdvanceFrame(0.01).ok());
    EXPECT_EQ(direct->committed(), gpu->committed());
    auto direct_lease = direct_backend->AcquireCommitted();
    ASSERT_TRUE(direct_lease.ok());
    EXPECT_EQ(DeviceBytes(*direct_lease), DeviceBytes(*final_lease));
    *initial_lease = {};
    observed.accepted_lease = {};
    backend->CollectRetired();
    EXPECT_EQ(
        backend->usage().device_reserved_bytes,
        after_growth.device_reserved_bytes - before.device_reserved_bytes
    );
    EXPECT_EQ(backend->usage().device_free_count, before.device_free_count + 1);
    std::cout << "growth_peak_device_bytes=" << after_growth.arena_metrics.peak_device_bytes
              << " growth_peak_pinned_bytes=" << after_growth.arena_metrics.peak_pinned_bytes << '\n';
  }
}

void CheckSignedTopologyAndSlab() {
  for (const bool slab : {false, true}) {
    auto initial = Initial();
    auto scene = Scene();
    scene.field = LiquidParticleSphereUnion3D{0.9};
    scene.world_acceleration = {};
    scene.surface_tension = 0;
    scene.ambient_pressure = 0;
    if (slab) {
      initial.particles[0].world_position.x = 1.4;
      initial.particles[0].side_tag = -1;
      initial.particles[1].world_position.x = 2.6;
      initial.particles[1].side_tag = 1;
      scene.obstacle = TranslatingSlab3D{{1, 0, 0}, 2, 0.05, {0.1, 0, 0}};
    } else {
      initial.particles[0].world_position = {6, 6, 6};
      initial.particles[1].world_position = {-2, -2, -2};
      scene.obstacle = {};
    }
    auto options = Options();
    auto cpu = LiquidSubstepScheduler3D::Create(initial, scene, options);
    ASSERT_TRUE(cpu.ok());
    ASSERT_TRUE(cpu->AdvanceFrame(0.01).ok());
    auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
    ASSERT_TRUE(backend.ok());
    auto gpu = LiquidSubstepScheduler3D::Create(initial, scene, options, backend->operations());
    ASSERT_TRUE(gpu.ok());
    ASSERT_TRUE(gpu->AdvanceFrame(0.01).ok());
    ExpectState(cpu->committed(), gpu->committed());
    if (slab) {
      EXPECT_LT(gpu->committed().particles[0].world_position.x, 1.96);
      EXPECT_GT(gpu->committed().particles[1].world_position.x, 2.06);
    } else {
      EXPECT_EQ(gpu->committed().particles[0].id, 1);
      EXPECT_EQ(gpu->committed().particles[1].id, 2);
      EXPECT_LT(gpu->committed().blocks[0].block_index.x, 0);
    }
  }
}
void CheckStalePublicationAndCompleteTable() {
  auto initial = Initial();
  const auto options = Options();
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
  ASSERT_TRUE(backend.ok());
  auto incomplete = backend->operations();
  incomplete.project = nullptr;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, Scene(), options, incomplete).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  auto gpu = LiquidSubstepScheduler3D::Create(initial, Scene(), options, backend->operations());
  ASSERT_TRUE(gpu.ok());
  auto lease = backend->AcquireCommitted();
  ASSERT_TRUE(lease.ok());
  const auto bytes = DeviceBytes(*lease);
  const auto identity = lease->identity();
  const auto before = backend->usage();
  const auto operations = backend->operations();
  auto stale = CloneLiquidState3D(initial, options.capacities);
  ASSERT_TRUE(stale.ok());
  stale->clock.epoch = 1;
  LiquidClock3D accepted{
      .time = 0.01,
      .frame_ordinal = 1,
      .substep_ordinal = 0,
      .accepted_step_ordinal = 1,
      .epoch = 1,
  };
  EXPECT_EQ(
      operations.prepare_publication(operations.context, *stale, accepted).code(),
      absl::StatusCode::kInvalidArgument
  );
  operations.discard_publication(operations.context);
  EXPECT_EQ(DeviceBytes(*lease), bytes);
  EXPECT_EQ(lease->identity(), identity);
  EXPECT_EQ(gpu->committed(), initial);
  ExpectNoOwnershipCalls(before.arena_metrics, backend->usage().arena_metrics);
  EXPECT_EQ(
      CudaLiquidSubstepBackend3D::Create(options.capacities, -1).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

void CheckNumericalRejection() {
  auto initial = Initial();
  initial.particles[0].mass = 1e100;
  const auto options = Options();
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
  ASSERT_TRUE(backend.ok());
  auto gpu = LiquidSubstepScheduler3D::Create(initial, Scene(), options, backend->operations());
  ASSERT_TRUE(gpu.ok());
  auto lease = backend->AcquireCommitted();
  ASSERT_TRUE(lease.ok());
  const auto bytes = DeviceBytes(*lease);
  ASSERT_TRUE(gpu->BeginFrame(0.01).ok());
  MeasuredAttempt(*gpu, *backend, 0.01, LiquidAttemptDisposition3D::kFatal);
  EXPECT_EQ(gpu->diagnostics().failure_code, absl::StatusCode::kOutOfRange);
  EXPECT_EQ(gpu->diagnostics().failed_stage, LiquidPhysicalStage3D::kDeposit);
  EXPECT_EQ(gpu->committed(), initial);
  EXPECT_EQ(DeviceBytes(*lease), bytes);
}

// Count actual CUDA calls and forward them to the current real dispatch.
class FieldRuntimeObservation {
 public:
  FieldRuntimeObservation() : saved_(cuda_internal::LiquidArenaRuntimeCalls()) {
    launches = 0;
    fences = 0;
    auto& runtime = cuda_internal::LiquidArenaRuntimeCalls();
    runtime.launch_field_samples = [](const cuda_internal::LiquidSampleQuery3D* query,
                                      const Vec3* centers,
                                      const Vec3* positions,
                                      double* distances,
                                      int* architecture,
                                      std::size_t sample_count,
                                      void* stream) {
      ++launches;
      return cuda_internal::LaunchLiquidFieldSamples3D(
          query,
          centers,
          positions,
          distances,
          architecture,
          sample_count,
          stream
      );
    };
    runtime.event_synchronize = [](cudaEvent_t event) {
      ++fences;
      return cudaEventSynchronize(event);
    };
  }
  ~FieldRuntimeObservation() {
    cuda_internal::LiquidArenaRuntimeCalls() = saved_;
  }
  static inline std::size_t launches = 0;
  static inline std::size_t fences = 0;

 private:
  cuda_internal::LiquidArenaRuntimeCalls3D saved_;
};

void CheckRejectedFieldBatch() {
  const auto initial = Initial();
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 2.1};
  const auto options = Options();
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
  ASSERT_TRUE(backend.ok()) << backend.status();
  auto operations = backend->operations();
  static const auto real_batch = operations.sample_field_batch;
  static std::size_t batch_count;
  batch_count = 0;
  operations.sample_field_batch = [](void* context,
                                     const LiquidSceneField3D& field,
                                     std::span<const LiquidParticleState3D> particles,
                                     std::span<const Vec3> positions,
                                     std::span<double> distances) {
    if (++batch_count != 2) {
      return real_batch(context, field, particles, positions, distances);
    }
    std::array<Vec3, kLiquidFieldBatchCapacity3D> overflowed_positions;
    std::ranges::copy(positions, overflowed_positions.begin());
    constexpr double maximum = std::numeric_limits<double>::max();
    overflowed_positions[positions.size() - 1] = {maximum, maximum, 0};
    return real_batch(context, field, particles, std::span(overflowed_positions).first(positions.size()), distances);
  };
  auto gpu = LiquidSubstepScheduler3D::Create(initial, scene, options, operations);
  ASSERT_TRUE(gpu.ok()) << gpu.status();
  auto lease = backend->AcquireCommitted();
  ASSERT_TRUE(lease.ok()) << lease.status();
  const auto bytes = DeviceBytes(*lease);
  const auto identity = lease->identity();
  ASSERT_TRUE(gpu->BeginFrame(0.01).ok());
  {
    const FieldRuntimeObservation calls;
    MeasuredAttempt(*gpu, *backend, 0.01, LiquidAttemptDisposition3D::kFatal);
    EXPECT_EQ(calls.launches, 2);
    EXPECT_EQ(calls.fences, 2);
  }
  EXPECT_EQ(batch_count, 2);
  EXPECT_EQ(gpu->diagnostics().failure_code, absl::StatusCode::kOutOfRange);
  EXPECT_EQ(gpu->diagnostics().failed_stage, LiquidPhysicalStage3D::kActivate);
  EXPECT_EQ(gpu->committed(), initial);
  const auto after = backend->AcquireCommitted();
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(after->identity(), identity);
  EXPECT_EQ(DeviceBytes(*after), bytes);
}

TEST(CudaLiquidSubstep3D, OpenTankSchedulerMatchesCpu) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  ASSERT_EQ(properties.major, 12);
  ASSERT_EQ(properties.minor, 0);
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{1.1};
  scene.surface_tension = 0;
  scene.obstacle = AxisAlignedLiquidTank3D{1.75, 3.5, 2, 1.5, 3.5, 0.125};
  const auto initial = Initial();
  const auto options = Options();
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
  ASSERT_TRUE(backend.ok()) << backend.status();
  auto gpu = LiquidSubstepScheduler3D::Create(initial, scene, options, backend->operations());
  ASSERT_TRUE(gpu.ok()) << gpu.status();
  const auto gpu_status = gpu->AdvanceFrame(0.01);
  ASSERT_TRUE(gpu_status.ok()) << gpu_status;
  Receipt("liquid_open_tank_substep_3d", backend->usage());
  {
    const FieldRuntimeObservation calls;
    ASSERT_NO_FATAL_FAILURE(tank_staging_test::ExpectMixedSignedFieldSupport(backend->operations(), 1e-12));
    EXPECT_EQ(calls.launches, 2);
    EXPECT_EQ(calls.fences, 2);
  }
  {
    const FieldRuntimeObservation calls;
    ASSERT_NO_FATAL_FAILURE(tank_staging_test::ExpectFieldBatchContract(backend->operations()));
    // Valid world, compensated plane, and the world arithmetic error reach
    // the hardware. Empty, invalid, and oversized batches submit nothing.
    EXPECT_EQ(calls.launches, 3);
    EXPECT_EQ(calls.fences, 3);
  }
  ASSERT_NO_FATAL_FAILURE(tank_staging_test::ExpectWallSamples(backend->operations()));
  ASSERT_NO_FATAL_FAILURE(tank_staging_test::ExpectTranslatedFaceUsesLatticeResidual(backend->operations()));
  ASSERT_NO_FATAL_FAILURE(tank_staging_test::ExpectParticleUpdateUsesBothVelocityFields(backend->operations()));
  auto cpu = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(cpu.ok()) << cpu.status();
  const auto cpu_status = cpu->AdvanceFrame(0.01);
  ASSERT_TRUE(cpu_status.ok()) << cpu_status;
  ExpectState(cpu->committed(), gpu->committed());
  auto plane_backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
  ASSERT_TRUE(plane_backend.ok()) << plane_backend.status();
  ASSERT_NO_FATAL_FAILURE(tank_staging_test::ExpectPlaneSupport(plane_backend->operations(), options));
  EXPECT_EQ(plane_backend->usage().compiled_architecture, 1200);
  auto translated_backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
  ASSERT_TRUE(translated_backend.ok()) << translated_backend.status();
  ASSERT_NO_FATAL_FAILURE(
      tank_staging_test::ExpectTranslatedPlaneKeepsWetCell(translated_backend->operations(), options)
  );
  EXPECT_EQ(translated_backend->usage().compiled_architecture, 1200);
  ASSERT_NO_FATAL_FAILURE(CheckRejectedFieldBatch());
}

void CheckSlabWallSamplesAndRefreshFailures() {
  namespace wall_test = liquid_wall_samples_test;
  const auto options = Options();
  const auto initial = wall_test::SlabParticles();
  const auto scene = wall_test::SlabScene();
  std::vector<std::byte> retry_bytes;
  for (const bool fail_new : {false, true}) {
    SCOPED_TRACE(fail_new);
    auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
    ASSERT_TRUE(backend.ok()) << backend.status();
    wall_test::Audit audit(options.capacities.faces);
    auto scheduler =
        LiquidSubstepScheduler3D::Create(initial, scene, options, wall_test::Observe(audit, backend->operations()));
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    auto retained = backend->AcquireCommitted();
    ASSERT_TRUE(retained.ok()) << retained.status();
    const auto initial_bytes = DeviceBytes(*retained);
    const auto initial_identity = retained->identity();
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    audit.fail_at_wall_coordinate = fail_new ? 2.501 : 2.5;
    ASSERT_NO_FATAL_FAILURE(MeasuredAttempt(*scheduler, *backend, 0.01, LiquidAttemptDisposition3D::kRetryDt));
    const auto failed_calls = wall_test::SlabCalls(2.501);
    wall_test::ExpectCalls(audit, std::span(failed_calls).first(fail_new ? 8 : 7));
    EXPECT_EQ(audit.gather_count, 2);
    EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kGather);
    EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kAborted);
    EXPECT_EQ(scheduler->diagnostics().old_pic_fallback_count, audit.old_fallbacks);
    EXPECT_EQ(scheduler->diagnostics().new_pic_fallback_count, audit.new_fallbacks);
    EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
    EXPECT_EQ(scheduler->committed(), initial);
    auto current = backend->AcquireCommitted();
    ASSERT_TRUE(current.ok()) << current.status();
    EXPECT_EQ(current->identity(), initial_identity);
    EXPECT_EQ(DeviceBytes(*current), initial_bytes);

    audit.Reset();
    audit.fail_at_wall_coordinate = 0;
    ASSERT_NO_FATAL_FAILURE(MeasuredAttempt(*scheduler, *backend, 0.005, LiquidAttemptDisposition3D::kAccepted));
    wall_test::ExpectCalls(audit, wall_test::SlabCalls(2.5005));
    EXPECT_EQ(scheduler->diagnostics().old_pic_fallback_count, audit.old_fallbacks);
    EXPECT_EQ(scheduler->diagnostics().new_pic_fallback_count, audit.new_fallbacks);
    EXPECT_EQ(DeviceBytes(*retained), initial_bytes);
    auto accepted = backend->AcquireCommitted();
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    EXPECT_EQ(accepted->identity().clock, scheduler->committed().clock);
    auto readback = CloneLiquidState3D(initial, options.capacities);
    ASSERT_TRUE(readback.ok()) << readback.status();
    ASSERT_TRUE(accepted->Readback(*readback).ok());
    EXPECT_EQ(*readback, scheduler->committed());
    if (retry_bytes.empty()) {
      retry_bytes = DeviceBytes(*accepted);
    } else {
      EXPECT_EQ(DeviceBytes(*accepted), retry_bytes);
    }
  }
}

TEST(CudaLiquidSubstep3D, PersistentSubstepMatchesCpu) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  auto initial = Initial();
  auto options = Options();
  auto cpu = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(cpu.ok()) << cpu.status();
  ASSERT_TRUE(cpu->AdvanceFrame(0.01).ok());
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
  ASSERT_TRUE(backend.ok()) << backend.status();
  auto gpu = LiquidSubstepScheduler3D::Create(initial, Scene(), options, backend->operations());
  ASSERT_TRUE(gpu.ok()) << gpu.status();
  const auto before = backend->usage();
  const auto wall_start = std::chrono::steady_clock::now();
  ASSERT_TRUE(gpu->AdvanceFrame(0.01).ok());
  std::cout
      << "scheduler_frame_wall_us="
      << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wall_start).count()
      << '\n';
  ExpectState(cpu->committed(), gpu->committed());
  EXPECT_EQ(before.device_allocation_count, backend->usage().device_allocation_count);
  EXPECT_EQ(before.device_free_count, backend->usage().device_free_count);
  const auto final_usage = backend->usage();
  std::cout << "f9_cpu_pressure_iterations=" << cpu->diagnostics().pressure_diagnostics.iteration_count
            << " f9_cuda_pressure_iterations=" << gpu->diagnostics().pressure_diagnostics.iteration_count
            << " f9_cuda_host_reserved_bytes=" << gpu->reserved_bytes()
            << " f9_cuda_live_device_bytes=" << final_usage.arena_metrics.live_device_bytes
            << " f9_cuda_live_pinned_bytes=" << final_usage.arena_metrics.live_pinned_bytes << '\n';
  Receipt("liquid_substep_3d", final_usage);
  ASSERT_NO_FATAL_FAILURE(CheckSignedQuantaAndHistories(false));
  ASSERT_NO_FATAL_FAILURE(CheckSignedQuantaAndHistories(true));
  ASSERT_NO_FATAL_FAILURE(CheckReplayMoveAndEmpty());
  ASSERT_NO_FATAL_FAILURE(CheckSignedTopologyAndSlab());
  ASSERT_NO_FATAL_FAILURE(CheckSlabWallSamplesAndRefreshFailures());
  ASSERT_NO_FATAL_FAILURE(CheckStalePublicationAndCompleteTable());
  ASSERT_NO_FATAL_FAILURE(CheckRejectedAttempts());
  ASSERT_NO_FATAL_FAILURE(CheckNumericalRejection());
  ASSERT_NO_FATAL_FAILURE(CheckRuntimeFailure());
}

TEST(CudaLiquidSubstep3D, ReplaysAfterCapacityGrowth) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wall_test = liquid_wall_samples_test;
  const auto initial = wall_test::SlabParticles(2);
  const auto scene = wall_test::SlabScene();
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  auto cpu = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(cpu.ok()) << cpu.status();
  ASSERT_TRUE(cpu->AdvanceFrame(0.01).ok());
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0);
  ASSERT_TRUE(backend.ok()) << backend.status();
  wall_test::Audit audit(options.capacities.faces);
  auto gpu =
      LiquidSubstepScheduler3D::Create(initial, scene, options, wall_test::Observe(audit, backend->operations()));
  ASSERT_TRUE(gpu.ok()) << gpu.status();
  int emits = 0;
  gpu->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                      if (stage == LiquidPhysicalStage3D::kEmit && ++*static_cast<int*>(context) == 2) {
                                        return absl::Status(absl::StatusCode::kResourceExhausted, {});
                                      }
                                      return absl::OkStatus();
                                    }});
  const auto before = backend->usage();
  const auto wall_start = std::chrono::steady_clock::now();
  ASSERT_TRUE(gpu->AdvanceFrame(0.01).ok());
  std::cout
      << "scheduler_frame_wall_us="
      << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wall_start).count()
      << '\n';
  ExpectState(cpu->committed(), gpu->committed());
  EXPECT_EQ(gpu->diagnostics().capacity_replays, 1);
  wall_test::ExpectGrowthRefreshes(audit);
  EXPECT_GT(backend->usage().graph_preparation_count, before.graph_preparation_count);
  Receipt("liquid_substep_3d_growth", backend->usage());
  ASSERT_NO_FATAL_FAILURE(CheckGrowthLeasesAndRollback());
}
}  // namespace
}  // namespace fluidity
