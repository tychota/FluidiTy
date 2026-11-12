#include <cuda_runtime_api.h>

#include <new>
#include <utility>

#include "../src/liquid_substep_3d_internal.h"
#include "fluidity/host/cuda_liquid_substep_3d.h"
#include "fluidity/host/cuda_wetted_piston_3d.h"
#include "liquid_geometry_stages_3d.h"
#include "liquid_pressure_stage_3d.h"
#include "liquid_transfer_stages_3d.h"

namespace fluidity {
struct CudaLiquidSubstepBackend3D::Impl {
  Impl(CudaLiquidArena3D storage, bool measure_job, bool needs_wetted_piston)
      : arena(std::move(storage)), measure_job(measure_job), needs_wetted_piston(needs_wetted_piston) {
  }
  CudaLiquidArena3D arena;
  bool measure_job;
  bool needs_wetted_piston;
  float substep_device_timeline_ms = 0;
};
CudaLiquidSubstepBackend3D::CudaLiquidSubstepBackend3D(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}
CudaLiquidSubstepBackend3D::CudaLiquidSubstepBackend3D(CudaLiquidSubstepBackend3D&&) noexcept = default;
CudaLiquidSubstepBackend3D& CudaLiquidSubstepBackend3D::operator=(CudaLiquidSubstepBackend3D&&) noexcept = default;
CudaLiquidSubstepBackend3D::~CudaLiquidSubstepBackend3D() = default;
absl::StatusOr<CudaLiquidSubstepBackend3D> CudaLiquidSubstepBackend3D::Create(
    LiquidStateCapacities3D capacities,
    int device_ordinal,
    bool measure_job,
    bool needs_wetted_piston
) {
  constexpr auto no_byte_limit = static_cast<std::size_t>(-1);
  auto arena = needs_wetted_piston
      ? CudaLiquidArena3D::CreateWithWettedPiston(capacities, device_ordinal, no_byte_limit, measure_job)
      : CudaLiquidArena3D::Create(capacities, device_ordinal, no_byte_limit, measure_job);
  if (!arena.ok()) {
    return arena.status();
  }
  auto impl = std::unique_ptr<Impl>(new (std::nothrow) Impl(std::move(*arena), measure_job, needs_wetted_piston));
  if (!impl) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  return CudaLiquidSubstepBackend3D(std::move(impl));
}
absl::StatusOr<CudaLiquidEpochLease3D> CudaLiquidSubstepBackend3D::AcquireCommitted() {
  return impl_->arena.AcquireCommitted();
}
void CudaLiquidSubstepBackend3D::CollectRetired() noexcept {
  impl_->arena.CollectRetired();
}
LiquidSubstepBackend3D
CudaLiquidSubstepBackend3D::operations(bool enable_gather_runs, bool enable_gathered_field_batches) {
  LiquidSubstepBackend3D operations{
      .context = impl_.get(),
      .sample_field =
          [](void* context,
             const LiquidSceneField3D& field,
             std::span<const LiquidParticleState3D> particles,
             Vec3 position) {
            return cuda_internal::SampleLiquidField3D(static_cast<Impl*>(context)->arena, field, particles, position);
          },
      .sample_faces =
          [](void* context,
             SparseMacGridView3D grid,
             const LiquidObstacle3D& obstacle,
             std::int64_t side,
             std::span<LiquidBoundaryFaceTag3D> tags,
             std::span<LiquidPrescribedWallSample3D> walls) {
            return cuda_internal::SampleLiquidFaces3D(
                static_cast<Impl*>(context)->arena,
                grid,
                obstacle,
                side,
                tags,
                walls
            );
          },
      .initialize =
          [](void* context, const LiquidState3D& initial) {
            return static_cast<Impl*>(context)->arena.Initialize(initial);
          },
      .prepare_publication =
          [](void* context, const LiquidState3D& trial, LiquidClock3D accepted_clock) {
            auto& owner = *static_cast<Impl*>(context);
            auto status = owner.arena.PreparePublication(trial, accepted_clock);
            if (!status.ok()) {
              return status;
            }
            const auto elapsed = owner.arena.EndDeviceTimeline();
            if (!elapsed.ok()) {
              return elapsed.status();
            }
            owner.substep_device_timeline_ms = *elapsed;
            return absl::OkStatus();
          },
      .commit_publication =
          [](void* context) noexcept {
            static_cast<Impl*>(context)->arena.CommitPublication();
          },
      .discard_publication =
          [](void* context) noexcept {
            static_cast<Impl*>(context)->arena.DiscardPublication();
          },
      .prepare_growth =
          [](void* context, LiquidStateCapacities3D capacities, const LiquidState3D& frame_start) {
            return static_cast<Impl*>(context)->arena.PrepareGrowth(capacities, frame_start);
          },
      .commit_growth =
          [](void* context) noexcept {
            static_cast<Impl*>(context)->arena.CommitGrowth();
          },
      .discard_growth =
          [](void* context) noexcept {
            static_cast<Impl*>(context)->arena.DiscardGrowth();
          },
      .topology =
          [](void* context,
             std::span<const LiquidParticleState3D> particles,
             std::span<const CellIndex3D> pressure_candidates,
             const LiquidState3D& committed,
             LiquidStepWorkspace3D& workspace) {
            auto& arena = static_cast<Impl*>(context)->arena;
            const auto status = arena.BeginDeviceTimeline();
            if (!status.ok()) {
              return status;
            }
            return arena.PrepareTopology(particles, pressure_candidates, workspace, &committed);
          },
      .deposit =
          [](void* context, const LiquidBoundaryDepositViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
            return cuda_internal::DepositLiquid3D(static_cast<Impl*>(context)->arena, inputs, workspace);
          },
      .normalize =
          [](void* context,
             std::span<const MacFaceDepositQuanta3D> face_quanta,
             MacDepositQuantization3D quantization,
             double minimum_face_mass,
             LiquidBoundaryWorkspace3D& workspace) {
            return cuda_internal::NormalizeLiquid3D(
                static_cast<Impl*>(context)->arena,
                face_quanta,
                quantization,
                minimum_face_mass,
                workspace
            );
          },
      .predict_missing_face =
          [](void* context, const LiquidMissingFacePredictionInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
            return cuda_internal::PredictMissingLiquidFace3D(static_cast<Impl*>(context)->arena, inputs, workspace);
          },
      .forces =
          [](void* context,
             SparseMacGridView3D grid,
             std::span<const double> old_face_velocities,
             Vec3 world_acceleration,
             double step_duration,
             std::span<double> predicted_face_velocities) {
            return cuda_internal::ApplyLiquidForces3D(
                static_cast<Impl*>(context)->arena,
                grid,
                old_face_velocities,
                world_acceleration,
                step_duration,
                predicted_face_velocities
            );
          },
      .freeze =
          [](void* context, const LiquidBoundaryGeometryViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
            return cuda_internal::FreezeLiquidGeometry3D(static_cast<Impl*>(context)->arena, inputs, workspace);
          },
      .wetted_piston =
          [](void* context, const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace) {
            return CudaComputeWettedPiston3D(static_cast<Impl*>(context)->arena, inputs, workspace);
          },
      .links =
          [](void* context,
             const FrozenLiquidBoundaryGeometryView3D& geometry,
             const LiquidPressureFields3D& fields,
             LiquidBoundaryWorkspace3D& workspace) {
            return cuda_internal::BuildLiquidLinks3D(static_cast<Impl*>(context)->arena, geometry, fields, workspace);
          },
      .project =
          [](void* context,
             SparseMacGridView3D grid,
             double step_duration,
             const LiquidPressureProblemView3D& problem,
             MacPressureSolverOptions3D options,
             MacPressureWorkspace3D& workspace) {
            return cuda_internal::ProjectLiquidPressure3D(
                static_cast<Impl*>(context)->arena,
                grid,
                step_duration,
                problem,
                options,
                workspace
            );
          },
      .project_viscous =
          [](void* context,
             const LiquidBoundaryGeometryViewInputs3D& geometry,
             const LiquidPressureProblemView3D& problem,
             double dynamic_viscosity,
             std::span<const double> cell_dynamic_viscosity,
             MacPressureSolverOptions3D options,
             MacPressureWorkspace3D& workspace) {
            auto& arena = static_cast<Impl*>(context)->arena;
            return liquid_step_internal::ProjectUniformViscosity(
                geometry,
                problem,
                dynamic_viscosity,
                cell_dynamic_viscosity,
                options,
                workspace,
                [&](const pressure_internal::UniformViscousPressureInputs3D& inputs,
                    std::span<const MacStrainRow3D> rows,
                    MacPressureSolverOptions3D controls,
                    MacPressureWorkspace3D& pressure_workspace) {
                  return cuda_internal::ProjectArenaViscousPressure3D(
                      arena,
                      inputs,
                      rows,
                      controls,
                      pressure_workspace
                  );
                }
            );
          },
      .gather =
          [](void* context, const LiquidBoundaryGatherViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
            return cuda_internal::GatherLiquid3D(static_cast<Impl*>(context)->arena, inputs, workspace);
          },
      .update_particles =
          [](void* context, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> trial_particles
          ) {
            return cuda_internal::UpdateLiquidParticles3D(static_cast<Impl*>(context)->arena, inputs, trial_particles);
          },
      .sample_field_batch =
          [](void* context,
             const LiquidSceneField3D& field,
             std::span<const LiquidParticleState3D> particles,
             std::span<const Vec3> positions,
             std::span<double> distances) {
            auto& arena = static_cast<Impl*>(context)->arena;
            const auto ready = arena.RecordRuntimeResult(cudaSuccess);
            if (!ready.ok()) {
              return ready;
            }
            if (positions.size() == distances.size() && positions.size() > kLiquidFieldBatchCapacity3D) {
              return absl::Status(absl::StatusCode::kResourceExhausted, {});
            }
            return cuda_internal::SampleLiquidField3D(arena, field, particles, positions, distances);
          },
      .sample_lattice_plane =
          [](void* context,
             const LiquidHorizontalPlaneSurface3D& plane,
             UniformMacGeometry3D geometry,
             std::span<const double> sample_lattice_ordinates_y,
             std::span<double> distances) {
            return cuda_internal::SampleLiquidLatticePlane3D(
                static_cast<Impl*>(context)->arena,
                plane,
                geometry,
                sample_lattice_ordinates_y,
                distances
            );
          },
  };
  if (!impl_->needs_wetted_piston) {
    operations.wetted_piston = nullptr;
  }
  if (enable_gather_runs) {
    operations.gather_run =
        [](void* context, const LiquidBoundaryGatherViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
          return cuda_internal::GatherLiquidRun3D(static_cast<Impl*>(context)->arena, inputs, workspace);
        };
    if (enable_gathered_field_batches) {
      operations.sample_gathered_fields = operations.sample_field_batch;
    }
  }
  if (impl_->measure_job) {
    operations.set_work_phase = [](void* context, LiquidWorkPhase3D phase) {
      static_cast<Impl*>(context)->arena.SetWorkPhase(phase);
    };
  }
  return operations;
}
CudaLiquidArenaMetrics3D CudaLiquidSubstepBackend3D::DrainMetrics() && {
  return std::move(impl_->arena).DrainMetrics();
}
absl::StatusOr<bool> CudaLiquidSubstepBackend3D::PublicationReady() const {
  return impl_->arena.PublicationReady();
}
CudaLiquidSubstepUsage3D CudaLiquidSubstepBackend3D::usage() const {
  const auto metrics = impl_->arena.metrics();
  return {
      .device_reserved_bytes = metrics.live_device_bytes,
      .pinned_reserved_bytes = metrics.live_pinned_bytes,
      .device_memory_ceiling_bytes = metrics.device_byte_limit,
      .device_allocation_count = static_cast<std::size_t>(metrics.device_reservations),
      .device_free_count = static_cast<std::size_t>(metrics.device_frees),
      .graph_preparation_count = static_cast<std::size_t>(metrics.graph_creations),
      .graph_replay_count = static_cast<std::size_t>(metrics.sort_replays),
      .arena_metrics = metrics,
      .compiled_architecture = metrics.compiled_architecture,
      .substep_device_timeline_ms = impl_->substep_device_timeline_ms,
  };
}
}  // namespace fluidity
