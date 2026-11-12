#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#include "../src/mac_viscous_pressure_3d.h"
#include "coarse_fine_transfer_3d_bridge.h"
#include "fluidity/host/coarse_fine_transfer_3d.h"
#include "fluidity/host/cuda_liquid_substep_3d.h"
#include "fluidity/host/cuda_mac_pressure_3d.h"
#include "fluidity/host/cuda_viscosity.h"
#include "graded_tank_fixture.h"
#include "gtest/gtest.h"
#include "mac_pressure_explicit_cases.h"
#include "operator_runtime.h"

namespace fluidity {
namespace {

using cuda_internal::CoarseFineCudaResult3D;
using internal::CoarseFineCoreQuery3D;
using internal::CoarseFineCoreStatus3D;

std::vector<CoarseFineCudaResult3D> RunKernel(std::span<const CoarseFineCoreQuery3D> queries) {
  const auto run = [&]() -> absl::StatusOr<std::vector<CoarseFineCudaResult3D>> {
    absl::Status cleanup;
    std::vector<CoarseFineCudaResult3D> results(queries.size());
    const auto launch = [&]() -> absl::Status {
      cuda_internal::CudaOperatorRuntime runtime(cleanup);
      if (const auto status = runtime.Initialize(); !status.ok()) {
        return status;
      }
      CoarseFineCoreQuery3D* device_queries = nullptr;
      CoarseFineCudaResult3D* device_results = nullptr;
      if (const auto status = runtime.Upload(queries, device_queries); !status.ok()) {
        return status;
      }
      if (const auto status = runtime.AllocateZeroed(results.size(), device_results); !status.ok()) {
        return status;
      }
      if (const auto status = runtime.RecordKernelStart(); !status.ok()) {
        return status;
      }
      const auto launch_error = cuda_internal::LaunchCoarseFineStencilCuda3D(
          device_queries,
          device_results,
          queries.size(),
          runtime.stream()
      );
      if (const auto status = cuda_internal::CheckCuda(launch_error, "launch coarse-fine stencil"); !status.ok()) {
        return status;
      }
      if (const auto status = runtime.SynchronizeAndMeasureKernel(); !status.ok()) {
        return status;
      }
      return runtime.Download(device_results, std::span{results});
    }();
    if (!launch.ok()) {
      return launch;
    }
    if (!cleanup.ok()) {
      return cleanup;
    }
    return results;
  }();
  EXPECT_TRUE(run.ok()) << run.status();
  return run.ok() ? *run : std::vector<CoarseFineCudaResult3D>{};
}

enum class GradedTankVariant {
  kInviscid,
  kEos,
  kMiscible,
  kCapillary,
  kViscous,
  kMiscibleViscous,
};

void ExpectRejectedGradedTrialPreservesDeviceEpoch(GradedTankVariant variant) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto authored = test::PopulateGradedTankFixture3D(state, scene);
  ASSERT_TRUE(authored.ok()) << authored;
  if (variant == GradedTankVariant::kEos) {
    test::SetGradedTankEos3D(state, scene);
  }
  if (variant == GradedTankVariant::kMiscible) {
    test::SetGradedTankMiscible3D(state, scene);
  }
  if (variant == GradedTankVariant::kViscous) {
    scene.dynamic_viscosity = 2.0;
  }
  if (variant == GradedTankVariant::kMiscibleViscous) {
    test::SetGradedTankMiscibleViscosity3D(state, scene);
  }
  const bool capillary = variant == GradedTankVariant::kCapillary;
  if (capillary) {
    scene.surface_tension = 100.0;
  }
  LiquidStepOptions3D options;
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0, false, false, true);
  ASSERT_TRUE(backend.ok()) << backend.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options, backend->operations());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  bool reject = !capillary;
  scheduler->SetStageFaultHookForTesting({
      .context = &reject,
      .before_stage =
          [](void* context, LiquidPhysicalStage3D stage, double) {
            auto& pending = *static_cast<bool*>(context);
            if (pending && stage == LiquidPhysicalStage3D::kAccept) {
              pending = false;
              return absl::Status(absl::StatusCode::kAborted, {});
            }
            return absl::OkStatus();
          },
  });
  const auto usage_before_attempts = backend->usage();
  const double frame_duration = capillary ? 0.02 : 0.01;
  ASSERT_TRUE(scheduler->BeginFrame(frame_duration).ok());
  const auto rejected = scheduler->Attempt(frame_duration);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_EQ(scheduler->committed(), state);
  auto readback = CloneLiquidState3D(state, options.capacities);
  ASSERT_TRUE(readback.ok()) << readback.status();
  for (auto& particle : readback->particles.values()) {
    particle.phase_id = 0;
    particle.constituent_count = 0;
  }
  const auto expect_device_state = [&](const LiquidState3D& host) {
    auto expected = CloneLiquidState3D(host, options.capacities);
    ASSERT_TRUE(expected.ok()) << expected.status();
    // The device particle ABI omits host composition metadata. Graded cell
    // inventory, including composition, is part of the published device bank.
    for (auto& particle : expected->particles.values()) {
      particle.phase_id = 0;
      particle.constituent_count = 0;
    }
    EXPECT_EQ(*readback, *expected);
  };
  {
    const auto old = backend->AcquireCommitted();
    ASSERT_TRUE(old.ok()) << old.status();
    EXPECT_EQ(old->identity().clock, state.clock);
    const auto copied = old->Readback(*readback);
    ASSERT_TRUE(copied.ok()) << copied;
    expect_device_state(state);
  }
  for (int step = 0; step < 2; ++step) {
    const auto accepted = scheduler->Attempt(frame_duration / 2.0);
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
  }
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  const auto epoch = backend->AcquireCommitted();
  ASSERT_TRUE(epoch.ok()) << epoch.status();
  EXPECT_EQ(epoch->identity().clock, scheduler->committed().clock);
  const auto copied = epoch->Readback(*readback);
  ASSERT_TRUE(copied.ok()) << copied;
  expect_device_state(scheduler->committed());
  const auto usage_after_attempts = backend->usage();
  EXPECT_EQ(usage_after_attempts.compiled_architecture, 1200);
  if (variant == GradedTankVariant::kViscous || variant == GradedTankVariant::kMiscibleViscous) {
    ASSERT_TRUE(scheduler->diagnostics().accepted_measurement);
    EXPECT_TRUE(scheduler->diagnostics().accepted_measurement->viscous_max_momentum_residual_kg_m_per_s);
    EXPECT_EQ(usage_after_attempts.device_allocation_count, usage_before_attempts.device_allocation_count);
    EXPECT_EQ(usage_after_attempts.device_free_count, usage_before_attempts.device_free_count);
    EXPECT_EQ(usage_after_attempts.device_reserved_bytes, usage_before_attempts.device_reserved_bytes);
    EXPECT_EQ(usage_after_attempts.pinned_reserved_bytes, usage_before_attempts.pinned_reserved_bytes);
  }
  if (variant == GradedTankVariant::kEos) {
    EXPECT_LT(readback->particles[0].volume_ratio, 1.0);
    EXPECT_GT(readback->adaptive_cells[0].previous_pressure, 0.0);
  }
}
TEST(CudaAdaptiveSimulate3D, RejectedGradedTrialPreservesDeviceEpoch) {
  ExpectRejectedGradedTrialPreservesDeviceEpoch(GradedTankVariant::kInviscid);
}

TEST(CudaAdaptiveSimulate3D, RejectedGradedEosTrialPreservesDeviceEpoch) {
  ExpectRejectedGradedTrialPreservesDeviceEpoch(GradedTankVariant::kEos);
}

TEST(CudaAdaptiveSimulate3D, RejectedGradedMiscibleTrialPreservesDeviceEpoch) {
  ExpectRejectedGradedTrialPreservesDeviceEpoch(GradedTankVariant::kMiscible);
}

TEST(CudaAdaptiveSimulate3D, RejectedGradedCapillaryTrialPreservesDeviceEpoch) {
  ExpectRejectedGradedTrialPreservesDeviceEpoch(GradedTankVariant::kCapillary);
}

TEST(CudaAdaptiveSimulate3D, RejectedGradedMiscibleViscousTrialPreservesDeviceEpoch) {
  ExpectRejectedGradedTrialPreservesDeviceEpoch(GradedTankVariant::kMiscibleViscous);
}

TEST(CudaAdaptiveSimulate3D, RejectedGradedViscousTrialPreservesDeviceEpoch) {
  ExpectRejectedGradedTrialPreservesDeviceEpoch(GradedTankVariant::kViscous);
}

TEST(CudaAdaptiveSimulate3D, CoupledGradedNormalTractionMatchesManufacturedReference) {
  auto fixture = pressure_test::MakeAdaptiveCutPressureFixture();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const auto rows = pressure_test::ConfigureAdaptiveNormalTraction(*fixture);
  std::size_t device_solves = 0;
  const auto projection = pressure_internal::ProjectAdaptiveMacViscousPressureWithVelocitySolver3D(
      fixture->topology,
      1.0,
      fixture->cells,
      fixture->subfaces,
      rows,
      [&](const MacViscousVelocityBlockInputs3D& block,
          std::span<const double> rhs,
          MacViscousVelocityBlockOptions3D options) -> absl::StatusOr<MacViscousVelocityBlockSolution3D> {
        auto result = CudaSolveMacViscousVelocityBlock3D(block, rhs, options);
        if (!result.ok()) {
          return result.status();
        }
        EXPECT_EQ(result->measurement.compiled_architecture, 1200);
        ++device_solves;
        return std::move(result->solution);
      }
  );
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_GT(device_solves, 0U);
  const double interface_velocity = projection->face_velocities[fixture->internal_slot];
  const double outlet_velocity = projection->face_velocities[fixture->outlet_slot];
  constexpr double kTolerance = 2e-5;
  EXPECT_NEAR(interface_velocity, 0.25, kTolerance);
  EXPECT_NEAR(outlet_velocity, 0.375, kTolerance);
  EXPECT_NEAR(projection->cell_pressures[0], 2.0, kTolerance);
  EXPECT_NEAR(projection->cell_pressures[1], 2.5, kTolerance);
  EXPECT_LE(projection->max_cell_constraint_defect, kTolerance);
  EXPECT_LE(projection->max_momentum_residual, kTolerance);
  const double coarse_traction = 2.0 * interface_velocity - projection->cell_pressures[0];
  const double fine_traction = 4.0 * (outlet_velocity - interface_velocity) / 0.5 - projection->cell_pressures[1];
  EXPECT_NEAR(coarse_traction, -1.5, kTolerance);
  EXPECT_NEAR(fine_traction, -1.5, kTolerance);
}

TEST(CudaAdaptiveSimulate3D, AdaptiveInterfacePressureJumpPreservesOwnedFlux) {
  auto fixture = pressure_test::MakeAdaptiveCutPressureFixture();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  for (const double jump_pa : {0.5, -0.5}) {
    SCOPED_TRACE(jump_pa);
    fixture->subfaces[fixture->internal_slot].pressure_jump_pa = jump_pa;
    const auto graph = pressure_internal::PrepareAdaptiveMacPressureSystem3D(
        fixture->topology,
        1.0,
        fixture->cells,
        fixture->subfaces
    );
    ASSERT_TRUE(graph.ok()) << graph.status();
    const auto result = pressure_internal::CudaProjectPreparedMacPressure3D(*graph, {});
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->measurement.compiled_architecture, 1200);
    EXPECT_NEAR(result->projection.cell_pressures[0], 1.28125 - jump_pa, 2e-5);
    EXPECT_NEAR(result->projection.cell_pressures[1], 1.09375, 2e-5);
    EXPECT_NEAR(result->projection.corrected_face_velocities[fixture->internal_slot], 0.25, 2e-5);
    EXPECT_NEAR(result->projection.corrected_face_velocities[fixture->outlet_slot], 0.375, 2e-5);
    EXPECT_LE(result->projection.diagnostics.max_cell_constraint_defect, 2e-5);
  }
}

TEST(CudaAdaptiveSimulate3D, GradedExplicitQuadraticRejectionPreservesDeviceEpoch) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto particle_id = test::PopulateGradedQuadraticTankFixture3D(state, scene);
  ASSERT_TRUE(particle_id.ok()) << particle_id.status();
  LiquidStepOptions3D options;
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0, false, false, true);
  ASSERT_TRUE(backend.ok()) << backend.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options, backend->operations());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  auto readback = CloneLiquidState3D(state, options.capacities);
  ASSERT_TRUE(readback.ok()) << readback.status();
  scheduler->SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kParticleUpdate ? absl::AbortedError("reject after quadratic gather")
                                                                   : absl::OkStatus();
          },
  });
  ASSERT_TRUE(scheduler->BeginFrame(0.001).ok());
  const auto rejected = scheduler->AttemptWithAdaptiveQuadraticTransfer(0.001, state.clock.epoch, *particle_id);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kParticleUpdate);
  {
    const auto epoch = backend->AcquireCommitted();
    ASSERT_TRUE(epoch.ok()) << epoch.status();
    ASSERT_TRUE(epoch->Readback(*readback).ok());
    EXPECT_EQ(*readback, state);
    EXPECT_EQ(scheduler->committed(), state);
  }
  scheduler->SetStageFaultHookForTesting({});
  const auto accepted = scheduler->AttemptWithAdaptiveQuadraticTransfer(0.0005, state.clock.epoch, *particle_id);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
  const auto epoch = backend->AcquireCommitted();
  ASSERT_TRUE(epoch.ok()) << epoch.status();
  ASSERT_TRUE(epoch->Readback(*readback).ok());
  EXPECT_EQ(*readback, scheduler->committed());
  EXPECT_EQ(readback->clock.epoch, state.clock.epoch + 1);
  EXPECT_EQ(backend->usage().compiled_architecture, 1200);
}

TEST(CudaAdaptiveSimulate3D, PublishesLiveFamilyAndPreservesRejectedDeviceEpoch) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedFamilyFixture3D(state, scene).ok());
  LiquidStepOptions3D options;
  options.capacities = {12, 0, 12, 64, 0};
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0, false, false, true);
  ASSERT_TRUE(backend.ok()) << backend.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options, backend->operations());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  auto readback = CloneLiquidState3D(state, options.capacities);
  ASSERT_TRUE(readback.ok()) << readback.status();
  scheduler->SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject graded family")
                                                           : absl::OkStatus();
          },
  });
  const auto parent = state.adaptive_cells[0].key;
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto rejected = scheduler->AttemptWithAdaptiveFamily(0.01, parent, true);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  {
    const auto epoch = backend->AcquireCommitted();
    ASSERT_TRUE(epoch.ok()) << epoch.status();
    ASSERT_TRUE(epoch->Readback(*readback).ok());
    EXPECT_EQ(*readback, state);
    EXPECT_EQ(scheduler->committed(), state);
  }
  scheduler->SetStageFaultHookForTesting({});
  const auto refined = scheduler->AttemptWithAdaptiveFamily(0.005, parent, true);
  ASSERT_TRUE(refined.ok()) << refined.status();
  ASSERT_EQ(refined->disposition, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_EQ(scheduler->committed().adaptive_cells.size(), 12);
  {
    const auto epoch = backend->AcquireCommitted();
    ASSERT_TRUE(epoch.ok()) << epoch.status();
    ASSERT_TRUE(epoch->Readback(*readback).ok());
    EXPECT_EQ(*readback, scheduler->committed());
  }
  const auto coarsened = scheduler->AttemptWithAdaptiveFamily(0.005, parent, false);
  ASSERT_TRUE(coarsened.ok()) << coarsened.status();
  ASSERT_EQ(coarsened->disposition, LiquidAttemptDisposition3D::kAccepted);
  ASSERT_EQ(scheduler->committed().adaptive_cells.size(), 5);
  const auto epoch = backend->AcquireCommitted();
  ASSERT_TRUE(epoch.ok()) << epoch.status();
  ASSERT_TRUE(epoch->Readback(*readback).ok());
  EXPECT_EQ(*readback, scheduler->committed());
  EXPECT_EQ(backend->usage().compiled_architecture, 1200);
}

TEST(CudaAdaptiveSimulate3D, PublishesEulerianOwnerAndReseedsDeviceEpoch) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  ASSERT_TRUE(test::PopulateGradedBandFixture3D(state, scene).ok());
  LiquidStepOptions3D options;
  options.capacities = {8, 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0, false, false, true);
  ASSERT_TRUE(backend.ok()) << backend.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options, backend->operations());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  auto readback = CloneLiquidState3D(state, options.capacities);
  ASSERT_TRUE(readback.ok()) << readback.status();
  const auto key = state.adaptive_cells[1].key;
  const LiquidHandoffCarrier3D carrier{.current_density = 2, .volume_ratio = 1};
  constexpr auto kFull = LiquidHandoffOccupancy3D::kFullLiquid;
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  scheduler->SetStageFaultHookForTesting({
      .context = nullptr,
      .before_stage =
          [](void*, LiquidPhysicalStage3D stage, double) {
            return stage == LiquidPhysicalStage3D::kAccept ? absl::AbortedError("reject band handoff")
                                                           : absl::OkStatus();
          },
  });
  const auto rejected = scheduler->AttemptWithAdaptiveBandRetirement(0.01, key, kFull, carrier);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  const auto expect_device_state = [&](const LiquidState3D& expected) {
    const auto epoch = backend->AcquireCommitted();
    ASSERT_TRUE(epoch.ok()) << epoch.status();
    ASSERT_TRUE(epoch->Readback(*readback).ok());
    EXPECT_EQ(*readback, expected);
  };
  expect_device_state(state);
  scheduler->SetStageFaultHookForTesting({});
  const auto retired = scheduler->AttemptWithAdaptiveBandRetirement(0.005, key, kFull, carrier);
  ASSERT_TRUE(retired.ok()) << retired.status();
  ASSERT_EQ(retired->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
  ASSERT_EQ(scheduler->committed().adaptive_cells[1].material_owner, AdaptiveLiquidMaterialOwner3D::kEulerian);
  ASSERT_EQ(scheduler->committed().particles.size(), 4);
  expect_device_state(scheduler->committed());
  const auto reseeded = scheduler->AttemptWithAdaptiveCoreReseed(0.005, key, kFull, carrier, 1);
  ASSERT_TRUE(reseeded.ok()) << reseeded.status();
  ASSERT_EQ(reseeded->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
  ASSERT_EQ(scheduler->committed().adaptive_cells[1].material_owner, AdaptiveLiquidMaterialOwner3D::kParticles);
  ASSERT_EQ(scheduler->committed().particles.size(), 5);
  expect_device_state(scheduler->committed());
  EXPECT_EQ(backend->usage().compiled_architecture, 1200);
}

TEST(CudaCoarseFineTransfer3D, OracleQueriesAndAllOrientationsMatchCpuOnSm120) {
  constexpr Vec3 origin{-4, 2, -3};
  constexpr std::array<std::array<double, 3>, 10> positions{{
      {-2.25, -2.25, -2.25},
      {0, 0, 0},
      {-0.125, 0.1875, -0.375},
      {0.375, -0.125, 0.625},
      {0.625, 0.375, -0.125},
      {0.875, 0.75, 0.625},
      {1, 1, 1},
      {1.25, 0.875, 1.125},
      {1.375, 1.625, 1.125},
      {2.25, 2.25, 2.25},
  }};
  std::vector<CoarseFineCoreQuery3D> queries;
  for (int codimension = 1; codimension <= 3; ++codimension) {
    for (int rotation = 0; rotation < 2; ++rotation) {
      const std::array signs = rotation == 0 ? std::array{1, 1, 1} : std::array{-1, 1, -1};
      int fine_side[3]{};
      for (int axis = 0; axis < codimension; ++axis) {
        const int world_axis = (axis + rotation) % 3;
        fine_side[world_axis] = signs[world_axis];
      }
      for (const auto normal : {FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ}) {
        for (const double width : {0.5, 2.0}) {
          for (const auto& local : positions) {
            queries.push_back({
                .world_origin = origin,
                .coarse_cell_width = width,
                .fine_side = {fine_side[0], fine_side[1], fine_side[2]},
                .face_normal = normal,
                .particle_world_position =
                    {
                        origin.x + width * signs[0] * local[(3 - rotation) % 3],
                        origin.y + width * signs[1] * local[(4 - rotation) % 3],
                        origin.z + width * signs[2] * local[(5 - rotation) % 3],
                    },
            });
          }
        }
      }
    }
  }
  // Exercise every signed half-space, edge and octant with an asymmetric off-knot position.
  for (int x_side = -1; x_side <= 1; ++x_side) {
    for (int y_side = -1; y_side <= 1; ++y_side) {
      for (int z_side = -1; z_side <= 1; ++z_side) {
        if (x_side == 0 && y_side == 0 && z_side == 0) {
          continue;
        }
        for (const auto normal : {FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ}) {
          queries.push_back({
              .world_origin = origin,
              .coarse_cell_width = 0.5,
              .fine_side = {x_side, y_side, z_side},
              .face_normal = normal,
              .particle_world_position =
                  {
                      origin.x + (x_side < 0 ? -0.625 : 0.625),
                      origin.y + (y_side < 0 ? -0.4375 : 0.4375),
                      origin.z + (z_side < 0 ? -0.5625 : 0.5625),
                  },
          });
        }
      }
    }
  }
  ASSERT_EQ(queries.size(), 360 + 26 * 3);
  const auto results = RunKernel(queries);
  ASSERT_EQ(results.size(), queries.size());
  float peak_weight_error = 0;
  float peak_gradient_error = 0;
  for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
    SCOPED_TRACE(query_index);
    const auto& query = queries[query_index];
    const auto expected = EvaluateCoarseFineStencil3D(
        {query.world_origin, query.coarse_cell_width, {query.fine_side[0], query.fine_side[1], query.fine_side[2]}},
        query.face_normal,
        query.particle_world_position
    );
    ASSERT_TRUE(expected.ok()) << expected.status();
    const auto cpu_rows = expected->rows();
    const auto& actual = results[query_index];
    ASSERT_EQ(actual.compiled_architecture, 1200);
    ASSERT_EQ(actual.stencil.status, CoarseFineCoreStatus3D::kOk);
    ASSERT_EQ(actual.stencil.row_count, cpu_rows.size());
    for (std::size_t index = 0; index < cpu_rows.size(); ++index) {
      const auto& cpu = cpu_rows[index];
      const auto& gpu = actual.stencil.rows[index];
      EXPECT_EQ(gpu.level, cpu.level);
      EXPECT_EQ(gpu.owner.axis, cpu.owner.axis);
      EXPECT_EQ(gpu.owner.lattice_index.x, cpu.owner.lattice_index.x);
      EXPECT_EQ(gpu.owner.lattice_index.y, cpu.owner.lattice_index.y);
      EXPECT_EQ(gpu.owner.lattice_index.z, cpu.owner.lattice_index.z);
      EXPECT_NEAR(gpu.weight, cpu.weight, 2e-7f);
      peak_weight_error = std::max(peak_weight_error, std::abs(gpu.weight - cpu.weight));
      for (int axis = 0; axis < 3; ++axis) {
        EXPECT_EQ(gpu.offset_coarse_units[axis], cpu.offset_coarse_units[axis]);
        EXPECT_NEAR(gpu.gradient_coarse_units[axis], cpu.gradient_coarse_units[axis], 2e-7f);
        peak_gradient_error =
            std::max(peak_gradient_error, std::abs(gpu.gradient_coarse_units[axis] - cpu.gradient_coarse_units[axis]));
      }
    }
  }
  std::cout << "coarse_fine_cuda compiled_architecture=1200 queries=" << queries.size()
            << " peak_weight_error=" << peak_weight_error << " peak_gradient_error=" << peak_gradient_error << '\n';
}

TEST(CudaCoarseFineTransfer3D, MatchesMeasuredNumpyEdgeAndCornerRows) {
  struct OracleCase {
    CoarseFineCoreQuery3D query;
    std::uint8_t level;
    FaceLatticeIndex3D owner;
    float weight;
    std::array<float, 3> gradient;
  };
  constexpr Vec3 origin{-4, 2, -3};
  constexpr std::array cases{
      OracleCase{
          .query = {origin, 0.5, {1, 1, 0}, FaceNormalAxis::kX, {-3.375, 2.4375, -2.4375}},
          .level = 1,
          .owner = {2, 1, 2},
          .weight = 0.22648111f,
          .gradient = {-0.59082031f, -0.32942706f, 0.32942706f},
      },
      OracleCase{
          .query = {origin, 0.5, {1, 1, 1}, FaceNormalAxis::kY, {-3.6875, 2.1875, -3.0625}},
          .level = 0,
          .owner = {0, 0, -1},
          .weight = 0.24720836f,
          .gradient = {-0.08415604f, -0.24123573f, -0.30425644f},
      },
      OracleCase{
          .query = {origin, 0.5, {-1, 1, -1}, FaceNormalAxis::kX, {-3.9375, 2.3125, -3.1875}},
          .level = 0,
          .owner = {0, 0, -1},
          .weight = 0.35163778f,
          .gradient = {-0.12218666f, -0.11970644f, -0.11970645f},
      },
  };
  const std::array queries{cases[0].query, cases[1].query, cases[2].query};
  const auto results = RunKernel(queries);
  ASSERT_EQ(results.size(), cases.size());
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto& expected = cases[index];
    const auto& actual = results[index];
    ASSERT_EQ(actual.compiled_architecture, 1200);
    ASSERT_EQ(actual.stencil.status, CoarseFineCoreStatus3D::kOk);
    ASSERT_LE(actual.stencil.row_count, internal::kMaxCoarseFineCoreRows3D);
    const auto rows = std::span{actual.stencil.rows}.first(actual.stencil.row_count);
    const auto oracle_row = std::ranges::find_if(rows, [&](const auto& row) {
      const auto [x, y, z] = row.owner.lattice_index;
      return row.level == expected.level && row.owner.axis == expected.query.face_normal && x == expected.owner.x
          && y == expected.owner.y && z == expected.owner.z;
    });
    ASSERT_NE(oracle_row, rows.end());
    EXPECT_NEAR(oracle_row->weight, expected.weight, 2e-7f);
    for (std::size_t axis = 0; axis < expected.gradient.size(); ++axis) {
      EXPECT_NEAR(oracle_row->gradient_coarse_units[axis], expected.gradient[axis], 2e-7f);
    }
  }
}

TEST(CudaCoarseFineTransfer3D, InvalidQueriesPublishNoRows) {
  EXPECT_EQ(cuda_internal::LaunchCoarseFineStencilCuda3D(nullptr, nullptr, 0, nullptr), cudaSuccess);
  EXPECT_EQ(cuda_internal::LaunchCoarseFineStencilCuda3D(nullptr, nullptr, 1, nullptr), cudaErrorInvalidValue);
  const std::array queries{
      CoarseFineCoreQuery3D{{}, 1, {1, 0, 0}, FaceNormalAxis::kX, {0, -65, 0}},
      CoarseFineCoreQuery3D{{}, 0, {1, 0, 0}, FaceNormalAxis::kY, {}},
      CoarseFineCoreQuery3D{{}, 1, {0, -2, 0}, FaceNormalAxis::kZ, {}},
      CoarseFineCoreQuery3D{{}, 1, {1, 0, 0}, FaceNormalAxis::kX, {std::numeric_limits<double>::quiet_NaN(), 0, 0}},
  };
  const auto results = RunKernel(queries);
  ASSERT_EQ(results.size(), queries.size());
  EXPECT_EQ(results[0].stencil.status, CoarseFineCoreStatus3D::kOutOfRange);
  for (std::size_t index = 1; index < results.size(); ++index) {
    EXPECT_EQ(results[index].stencil.status, CoarseFineCoreStatus3D::kInvalidArgument);
  }
  for (const auto& result : results) {
    EXPECT_EQ(result.compiled_architecture, 1200);
    EXPECT_EQ(result.stencil.row_count, 0);
  }
}

}  // namespace
}  // namespace fluidity
