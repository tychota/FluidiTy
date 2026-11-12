#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#include "coarse_fine_transfer_3d_bridge.h"
#include "fluidity/host/coarse_fine_transfer_3d.h"
#include "fluidity/host/cuda_liquid_substep_3d.h"
#include "graded_tank_fixture.h"
#include "gtest/gtest.h"
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

TEST(CudaAdaptiveSimulate3D, RejectedGradedTrialPreservesDeviceEpoch) {
  LiquidState3D state;
  LiquidSceneInputs3D scene;
  const auto authored = test::PopulateGradedTankFixture3D(state, scene);
  ASSERT_TRUE(authored.ok()) << authored;
  LiquidStepOptions3D options;
  options.capacities = {state.particles.size(), 0, state.adaptive_cells.size(), state.adaptive_faces.size(), 0};
  auto backend = CudaLiquidSubstepBackend3D::Create(options.capacities, 0, false, false, true);
  ASSERT_TRUE(backend.ok()) << backend.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(state, scene, options, backend->operations());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  bool reject = true;
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
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto rejected = scheduler->Attempt(0.01);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  ASSERT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_EQ(scheduler->committed(), state);
  auto readback = CloneLiquidState3D(state, options.capacities);
  ASSERT_TRUE(readback.ok()) << readback.status();
  {
    const auto old = backend->AcquireCommitted();
    ASSERT_TRUE(old.ok()) << old.status();
    EXPECT_EQ(old->identity().clock, state.clock);
    const auto copied = old->Readback(*readback);
    ASSERT_TRUE(copied.ok()) << copied;
    EXPECT_EQ(*readback, state);
  }
  for (int step = 0; step < 2; ++step) {
    const auto accepted = scheduler->Attempt(0.005);
    ASSERT_TRUE(accepted.ok()) << accepted.status();
    ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted) << scheduler->diagnostics().failure_code;
  }
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  const auto epoch = backend->AcquireCommitted();
  ASSERT_TRUE(epoch.ok()) << epoch.status();
  EXPECT_EQ(epoch->identity().clock, scheduler->committed().clock);
  const auto copied = epoch->Readback(*readback);
  ASSERT_TRUE(copied.ok()) << copied;
  EXPECT_EQ(*readback, scheduler->committed());
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
