#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

#include "../cuda/liquid_arena_3d_internal.h"
#include "../cuda/liquid_pressure_stage_3d.h"
#include "../cuda/mac_pressure_3d_bridge.h"
#include "../src/mac_pressure_3d_workspace_internal.h"
#include "fluidity/host/cuda_mac_pressure_3d.h"
#include "fluidity/host/cuda_probe.h"
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include "gtest/gtest.h"
#include "mac_pressure_3d_fixture.h"

namespace fluidity {
namespace {
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr UniformMacGeometry3D kUnitGeometry{{0.0, 0.0, 0.0}, 1.0};
std::int32_t FixtureOwnerAxis(std::int32_t index) {
  return index / 4 - (index % 4 < 0 ? 1 : 0);
}
// Build an independent affine solution: each predicted face velocity is the
// analytic pressure slope, so correction leaves zero flux on every open face.
struct AffineProblem {
  std::vector<MacPressureCell3D> cells;
  std::vector<MacPressureFace3D> faces;
  std::vector<BlockIndex3D> blocks;
};

double AffinePressure(CellIndex3D index) {
  return index.x + 2.0 * index.y + 4.0 * index.z;
}

AffineProblem MakeAffineProblem(std::int32_t minimum, std::int32_t extent, bool anchored) {
  AffineProblem problem;
  std::set<std::tuple<int, int, int, int>> face_keys;
  for (int z = minimum; z < minimum + extent; ++z) {
    for (int y = minimum; y < minimum + extent; ++y) {
      for (int x = minimum; x < minimum + extent; ++x) {
        problem.cells.push_back({
            .cell_index = {x, y, z},
            .fluid_volume = 1.0,
            .bulk_modulus = kInfinity,
        });
        for (int axis = 0; axis < 3; ++axis) {
          for (int side = 0; side < 2; ++side) {
            face_keys.emplace(axis, x + (axis == 0 ? side : 0), y + (axis == 1 ? side : 0), z + (axis == 2 ? side : 0));
          }
        }
      }
    }
  }
  std::set<std::tuple<int, int, int>> block_keys;
  for (const auto& [axis, x, y, z] : face_keys) {
    const std::array coordinates{x, y, z};
    const bool low = coordinates[axis] == minimum;
    const bool high = coordinates[axis] == minimum + extent;
    const bool boundary = low || high;
    MacPressureFace3D face{
        .face_index = {static_cast<FaceNormalAxis>(axis), {x, y, z}},
        .open_area = boundary && !anchored ? 0.0 : 1.0,
        .inverse_density = 1.0,
        .pressure_gradient_distance = 1.0,
        .predicted_normal_velocity = boundary && !anchored ? 0.0 : static_cast<double>(1 << axis),
        .pressure_link = InternalMacPressureLink3D{0.0},
    };
    if (boundary && anchored) {
      const CellIndex3D exterior{
          x - (axis == 0 && low ? 1 : 0),
          y - (axis == 1 && low ? 1 : 0),
          z - (axis == 2 && low ? 1 : 0),
      };
      face.pressure_link = DirichletMacPressureLink3D{AffinePressure(exterior)};
    } else if (boundary) {
      face.pressure_link = SealedMacPressureLink3D{};
    }
    problem.faces.push_back(face);
    block_keys.emplace(FixtureOwnerAxis(x), FixtureOwnerAxis(y), FixtureOwnerAxis(z));
  }
  for (const auto& [x, y, z] : block_keys) {
    problem.blocks.push_back({x, y, z});
  }
  return problem;
}

void SortFaces(AffineProblem& problem, const SparseMacGrid3D& grid) {
  std::ranges::sort(problem.faces, [&](const auto& left, const auto& right) {
    return *grid.FaceSlot(left.face_index) < *grid.FaceSlot(right.face_index);
  });
}

void CheckHardware() {
  ProbeReport report;
  ASSERT_TRUE(RunCudaProbe(0, &report).ok());
  ASSERT_EQ(report.device_name, "NVIDIA GeForce RTX 5060 Ti");
  ASSERT_EQ(report.compute_major, 12);
  ASSERT_EQ(report.compute_minor, 0);
  ASSERT_EQ(report.compiled_architecture, 1200);
}
MacPressureProjection3D CopyProjection(const MacPressureProjectionView3D& view) {
  return {
      {view.cell_pressures.begin(), view.cell_pressures.end()},
      {view.corrected_face_velocities.begin(), view.corrected_face_velocities.end()},
      {view.integrated_rhs.begin(), view.integrated_rhs.end()},
      {view.true_residuals.begin(), view.true_residuals.end()},
      {view.cell_constraint_defects.begin(), view.cell_constraint_defects.end()},
      {view.gauge_cell_ordinals.begin(), view.gauge_cell_ordinals.end()},
      view.diagnostics,
  };
}
TEST(CudaLiquidPressureStage3D, MatchesIndependentMixedLinkFixtures) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  std::vector<pressure_test::MacPressureFixtureCase3D> fixtures;
  ASSERT_NO_FATAL_FAILURE(pressure_test::LoadMacPressureFixtureCases3D(fixtures));
  auto arena = CudaLiquidArena3D::Create({1, 128, 4096, 24576, 0});
  auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name);
    const auto result = cuda_internal::ProjectLiquidPressure3D(
        *arena,
        fixture.grid.view(),
        fixture.step_duration,
        {fixture.cells, fixture.faces},
        {},
        *workspace
    );
    ASSERT_TRUE(result.ok()) << result.status();
    pressure_test::ExpectMacPressureFixtureProjection3D(
        CopyProjection(*result),
        fixture.expected_projection,
        fixture.absolute_tolerance,
        fixture.relative_tolerance
    );
    const auto range = arena->range(CudaLiquidRange3D::kPressureState);
    const auto* state = reinterpret_cast<const cuda_internal::PressureState3D*>(range.pinned.data());
    EXPECT_EQ(state->compiled_architecture, 1200);
    EXPECT_TRUE(state->converged);
  }
}
TEST(CudaLiquidPressureStage3D, SignedMultilevelGaugeWarmStartAndReplay) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto problem = MakeAffineProblem(-3, 4, false);
  auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  auto arena = CudaLiquidArena3D::Create({1, 128, 4096, 24576, 0});
  auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  const auto result =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto first = CopyProjection(*result);
  const auto cpu = ProjectMacPressure3D({*grid, 1, problem.cells, problem.faces});
  const auto owning_cuda = CudaProjectMacPressure3D({*grid, 1, problem.cells, problem.faces});
  ASSERT_TRUE(cpu.ok());
  ASSERT_TRUE(owning_cuda.ok());
  pressure_test::ExpectMacPressureFixtureProjection3D(first, *cpu, 1e-9, 1e-9);
  pressure_test::ExpectMacPressureFixtureProjection3D(first, owning_cuda->projection, 0, 0);
  double maximum_velocity = 0;
  for (double velocity : owning_cuda->projection.corrected_face_velocities) {
    maximum_velocity = std::max(maximum_velocity, std::abs(velocity));
  }
  std::cout << "unchanged_owning_cuda_max_analytic_velocity_error=" << std::setprecision(17) << maximum_velocity
            << " bounded_exact_match=true" << '\n';

  ASSERT_GE(result->diagnostics.multigrid_level_count, 2);
  ASSERT_EQ(result->gauge_cell_ordinals.size(), 1);
  EXPECT_EQ(result->gauge_cell_ordinals[0], 0);
  for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
    EXPECT_NEAR(result->cell_pressures[cell], AffinePressure(problem.cells[cell].cell_index) + 21, 5e-5);
    EXPECT_NEAR(result->true_residuals[cell], -result->cell_constraint_defects[cell], 1e-12);
  }
  for (double velocity : result->corrected_face_velocities) {
    // Inherited analytic threshold for this fixture; CPU/CUDA parity above is tighter.
    EXPECT_NEAR(velocity, 0, 5e-5);
  }
  const auto repeat =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(repeat.ok());
  EXPECT_EQ(
      std::memcmp(first.cell_pressures.data(), repeat->cell_pressures.data(), repeat->cell_pressures.size_bytes()),
      0
  );
  EXPECT_EQ(
      std::memcmp(
          first.corrected_face_velocities.data(),
          repeat->corrected_face_velocities.data(),
          repeat->corrected_face_velocities.size_bytes()
      ),
      0
  );
  EXPECT_EQ(
      std::memcmp(first.integrated_rhs.data(), repeat->integrated_rhs.data(), repeat->integrated_rhs.size_bytes()),
      0
  );
  EXPECT_EQ(
      std::memcmp(first.true_residuals.data(), repeat->true_residuals.data(), repeat->true_residuals.size_bytes()),
      0
  );
  EXPECT_EQ(
      std::memcmp(
          first.cell_constraint_defects.data(),
          repeat->cell_constraint_defects.data(),
          repeat->cell_constraint_defects.size_bytes()
      ),
      0
  );
  EXPECT_EQ(
      std::memcmp(
          first.gauge_cell_ordinals.data(),
          repeat->gauge_cell_ordinals.data(),
          repeat->gauge_cell_ordinals.size_bytes()
      ),
      0
  );
  for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
    problem.cells[cell].previous_pressure = first.cell_pressures[cell] + 19;
  }
  const auto warm =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(warm.ok());
  EXPECT_EQ(warm->diagnostics.iteration_count, 0);
  EXPECT_EQ(warm->cell_pressures.front(), 0);
}
TEST(CudaLiquidPressureStage3D, FailurePreservesPriorProjection) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto problem = MakeAffineProblem(-3, 4, false);
  auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  auto arena = CudaLiquidArena3D::Create({1, 128, 4096, 24576, 0});
  auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  const auto result =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto first = CopyProjection(*result);
  const auto failed =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {1}, *workspace);
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kAborted);
  problem.cells[0].target_divergence = 1;
  const auto incompatible =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  EXPECT_EQ(incompatible.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(
      std::memcmp(first.cell_pressures.data(), result->cell_pressures.data(), result->cell_pressures.size_bytes()),
      0
  );
}

// Every override executes the real operation first. Failures model observed
// submission errors without inducing an unrecoverable fault in the test GPU.
class PressureRuntimeScope {
 public:
  explicit PressureRuntimeScope(int failure = 0) : saved_(cuda_internal::LiquidArenaRuntimeCalls()) {
    copies = 0;
    launches = 0;
    clears = 0;
    fences = 0;
    failure_ = failure;
    auto& calls = cuda_internal::LiquidArenaRuntimeCalls();
    calls.memcpy_async = Copy;
    calls.launch_pressure_batch = Launch;
    calls.memset_async = Clear;
    calls.event_synchronize = Fence;
  }
  ~PressureRuntimeScope() {
    cuda_internal::LiquidArenaRuntimeCalls() = saved_;
  }
  static inline std::size_t copies = 0;
  static inline std::size_t launches = 0;
  static inline std::size_t clears = 0;
  static inline std::size_t fences = 0;

 private:
  static cudaError_t CUDARTAPI
  Copy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    const auto result = cudaMemcpyAsync(destination, source, bytes, kind, stream);
    ++copies;
    return result == cudaSuccess && failure_ == 1 && copies == 2 ? cudaErrorInvalidValue : result;
  }
  static int
  Launch(cuda_internal::PressureDevice3D device, bool initialize, std::size_t iteration_count, void* stream) {
    const int result = cuda_internal::LaunchMacPressureBatch3D(device, initialize, iteration_count, stream);
    ++launches;
    return result == cudaSuccess && failure_ == 2 ? cudaErrorInvalidValue : result;
  }
  static cudaError_t CUDARTAPI Clear(void* destination, int value, std::size_t bytes, cudaStream_t stream) {
    const auto result = cudaMemsetAsync(destination, value, bytes, stream);
    ++clears;
    return result == cudaSuccess && failure_ == 3 ? cudaErrorInvalidValue : result;
  }
  static cudaError_t CUDARTAPI Fence(cudaEvent_t event) {
    const auto result = cudaEventSynchronize(event);
    ++fences;
    return result == cudaSuccess && failure_ == 4 ? cudaErrorInvalidValue : result;
  }
  static inline int failure_ = 0;
  cuda_internal::LiquidArenaRuntimeCalls3D saved_;
};
TEST(CudaLiquidPressureStage3D, CountsRealDeviceRecurrenceAndMeasuresExecution) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto problem = MakeAffineProblem(-3, 4, false);
  auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  auto arena = CudaLiquidArena3D::Create({1, 128, 4096, 24576, 0});
  auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  cudaEvent_t start = nullptr;
  cudaEvent_t finish = nullptr;
  ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&finish), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(start, static_cast<cudaStream_t>(arena->stream())), cudaSuccess);
  const auto before = arena->metrics();
  const auto wall_start = std::chrono::steady_clock::now();
  PressureRuntimeScope operations;
  const auto result =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(result.ok());
  const auto wall_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wall_start).count();
  ASSERT_EQ(cudaEventRecord(finish, static_cast<cudaStream_t>(arena->stream())), cudaSuccess);
  ASSERT_EQ(cudaEventSynchronize(finish), cudaSuccess);
  float elapsed_ms = 0;
  ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, finish), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(finish), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
  const auto iterations = result->diagnostics.iteration_count;
  constexpr auto batch_size = cuda_internal::kMacPressureIterationsPerBatch3D;
  ASSERT_GT(iterations, 1);
  EXPECT_EQ(PressureRuntimeScope::launches, (iterations + batch_size - 1) / batch_size);
  EXPECT_LT(PressureRuntimeScope::launches, iterations + 1);
  EXPECT_EQ(PressureRuntimeScope::copies, 7 + PressureRuntimeScope::launches);
  EXPECT_EQ(PressureRuntimeScope::fences, 1 + PressureRuntimeScope::launches);
  EXPECT_EQ(PressureRuntimeScope::clears, 5);
  EXPECT_EQ(before.device_reservations, arena->metrics().device_reservations);
  EXPECT_EQ(before.pinned_reservations, arena->metrics().pinned_reservations);
  EXPECT_EQ(before.device_frees, arena->metrics().device_frees);
  EXPECT_EQ(before.pinned_frees, arena->metrics().pinned_frees);
  std::cout << "bounded_pressure_arch=1200 launches=" << PressureRuntimeScope::launches
            << " copies=" << PressureRuntimeScope::copies << " clears=" << PressureRuntimeScope::clears
            << " fences=" << PressureRuntimeScope::fences << " elapsed_device_timeline_ms=" << elapsed_ms
            << " wall_us=" << wall_us << " device_bytes=" << before.live_device_bytes
            << " pinned_bytes=" << before.live_pinned_bytes << '\n';
}
TEST(CudaLiquidPressureStage3D, RuntimeFailuresPoisonBeforePinnedReuse) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto problem = MakeAffineProblem(-3, 4, false);
  auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  for (int failure : {1, 2, 3, 4}) {
    SCOPED_TRACE(failure);
    auto arena = CudaLiquidArena3D::Create({1, 128, 4096, 24576, 0});
    auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
    ASSERT_TRUE(arena.ok());
    ASSERT_TRUE(workspace.ok());
    const auto result =
        cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
    ASSERT_TRUE(result.ok());
    const auto preserved = CopyProjection(*result);
    const auto range = arena->range(CudaLiquidRange3D::kPressureCells);
    std::vector<std::byte> staged(range.pinned.size());
    {
      PressureRuntimeScope fault(failure);
      EXPECT_EQ(
          cuda_internal::ProjectLiquidPressure3D(
              *arena,
              grid->view(),
              1,
              {problem.cells, problem.faces},
              {},
              *workspace
          )
              .status()
              .code(),
          absl::StatusCode::kInternal
      );
      std::ranges::copy(range.pinned, staged.begin());
      const auto submitted = PressureRuntimeScope::copies;
      EXPECT_EQ(
          cuda_internal::ProjectLiquidPressure3D(
              *arena,
              grid->view(),
              0.5,
              {problem.cells, problem.faces},
              {},
              *workspace
          )
              .status()
              .code(),
          absl::StatusCode::kInternal
      );
      EXPECT_EQ(PressureRuntimeScope::copies, submitted);
      EXPECT_EQ(std::memcmp(staged.data(), range.pinned.data(), staged.size()), 0);
    }
    pressure_test::ExpectMacPressureFixtureProjection3D(CopyProjection(*result), preserved, 0, 0);
  }
}

TEST(CudaLiquidPressureStage3D, RejectsCapacityEmptyMalformedAndScratchAliasesWithoutPublication) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto problem = MakeAffineProblem(-3, 4, false);
  auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  auto arena = CudaLiquidArena3D::Create({1, 128, 4096, 24576, 0});
  auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  const auto published =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(published.ok());
  const auto preserved = CopyProjection(*published);
  const auto cells_range = arena->range(CudaLiquidRange3D::kPressureCells);
  auto* aliased_cells = reinterpret_cast<MacPressureCell3D*>(cells_range.pinned.data());
  std::copy(problem.cells.begin(), problem.cells.end(), aliased_cells);
  const auto faces_range = arena->range(CudaLiquidRange3D::kPressureLinks);
  auto* aliased_faces = reinterpret_cast<MacPressureFace3D*>(faces_range.pinned.data());
  std::copy(problem.faces.begin(), problem.faces.end(), aliased_faces);
  const auto topology_range = arena->range(CudaLiquidRange3D::kPressurePreparationScratch);
  auto* aliased_blocks = reinterpret_cast<BlockIndex3D*>(topology_range.pinned.data());
  std::ranges::copy(grid->active_block_indices(), aliased_blocks);
  const auto alias_grid =
      SparseMacGridView3D::Create(kUnitGeometry, {aliased_blocks, grid->active_block_indices().size()});
  ASSERT_TRUE(alias_grid.ok());
  auto tiny_workspace = MacPressureWorkspace3D::Create(1, 6);
  auto tiny_arena = CudaLiquidArena3D::Create({1, 128, 1, 24576, 0});
  ASSERT_TRUE(tiny_workspace.ok());
  ASSERT_TRUE(tiny_arena.ok());
  PressureRuntimeScope operations;
  for (const auto input : {
           LiquidPressureProblemView3D{},
           LiquidPressureProblemView3D{{aliased_cells, problem.cells.size()}, problem.faces},
           LiquidPressureProblemView3D{problem.cells, {aliased_faces, problem.faces.size()}},
           LiquidPressureProblemView3D{problem.cells, std::span(problem.faces).first(problem.faces.size() - 1)},
       }) {
    EXPECT_EQ(
        cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, input, {}, *workspace).status().code(),
        absl::StatusCode::kInvalidArgument
    );
  }
  EXPECT_EQ(
      cuda_internal::ProjectLiquidPressure3D(*arena, *alias_grid, 1, {problem.cells, problem.faces}, {}, *workspace)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      cuda_internal::ProjectLiquidPressure3D(
          *tiny_arena,
          grid->view(),
          1,
          {problem.cells, problem.faces},
          {},
          *workspace
      )
          .status()
          .code(),
      absl::StatusCode::kResourceExhausted
  );
  EXPECT_EQ(
      cuda_internal::ProjectLiquidPressure3D(
          *arena,
          grid->view(),
          1,
          {problem.cells, problem.faces},
          {},
          *tiny_workspace
      )
          .status()
          .code(),
      absl::StatusCode::kResourceExhausted
  );
  for (const std::size_t iterations : {std::size_t{0}, std::size_t{257}}) {
    EXPECT_EQ(
        cuda_internal::ProjectLiquidPressure3D(
            *arena,
            grid->view(),
            1,
            {problem.cells, problem.faces},
            {iterations},
            *workspace
        )
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument
    );
  }
  EXPECT_EQ(PressureRuntimeScope::copies, 0);
  EXPECT_EQ(PressureRuntimeScope::launches, 0);
  pressure_test::ExpectMacPressureFixtureProjection3D(CopyProjection(*published), preserved, 0, 0);
}
TEST(CudaLiquidPressureStage3D, MaximumHierarchyPackingAndPhysicalCorrectionMatchPreparedSpans) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto problem = MakeAffineProblem(-8, 16, true);
  auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  auto arena = CudaLiquidArena3D::Create({1, 128, 4096, 24576, 0});
  auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  const auto bytes = QueryMacPressureWorkspaceBytes3D(4096, 24576);
  ASSERT_TRUE(bytes.ok());
  std::vector<std::max_align_t> storage(
      (bytes->scratch_bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)
  );
  pressure_internal::MacPressureScratch3D scratch(storage);
  const auto prepared =
      pressure_internal::PrepareBoundedMacPressureUpload3D(grid->view(), 1, problem.cells, problem.faces, scratch);
  ASSERT_TRUE(prepared.ok());
  const auto cpu = ProjectMacPressure3D({*grid, 1, problem.cells, problem.faces});
  ASSERT_TRUE(cpu.ok());
  const auto result =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(result.ok()) << result.status();
  pressure_test::ExpectMacPressureFixtureProjection3D(CopyProjection(*result), *cpu, 1e-9, 1e-9);
  ASSERT_EQ(result->cell_pressures.size(), 4096);
  ASSERT_EQ(result->corrected_face_velocities.size(), 13056);
  ASSERT_GE(result->diagnostics.multigrid_level_count, 4);
  const auto* cells = reinterpret_cast<const cuda_internal::PressureCell3D*>(
      arena->range(CudaLiquidRange3D::kPressureCells).pinned.data()
  );
  for (std::size_t ordinal = 0; ordinal < problem.cells.size(); ++ordinal) {
    const auto& expected = prepared->system.cells()[ordinal];
    EXPECT_EQ(cells[ordinal].fluid_volume, expected.fluid_volume);
    EXPECT_EQ(cells[ordinal].compliance, expected.compliance);
    EXPECT_EQ(cells[ordinal].integrated_target_flux, expected.integrated_target_flux);
    EXPECT_EQ(cells[ordinal].previous_pressure, expected.previous_pressure);
    EXPECT_EQ(cells[ordinal].integrated_rhs, prepared->system.integrated_rhs()[ordinal]);
    EXPECT_EQ(cells[ordinal].rhs_absolute_term_sum, prepared->system.rhs_absolute_term_sums()[ordinal]);
    EXPECT_EQ(cells[ordinal].initial_pressure, prepared->initial_pressure[ordinal]);
  }

  const auto* links = reinterpret_cast<const cuda_internal::PressureLink3D*>(
      arena->range(CudaLiquidRange3D::kPressureLinks).pinned.data()
  );
  for (std::size_t ordinal = 0; ordinal < problem.faces.size(); ++ordinal) {
    const auto& expected = prepared->system.links()[ordinal];
    const auto& actual = links[ordinal];
    EXPECT_EQ(actual.tail_cell_ordinal, expected.tail_cell_ordinal.value_or(problem.cells.size()));
    EXPECT_EQ(actual.head_cell_ordinal, expected.head_cell_ordinal.value_or(problem.cells.size()));
    EXPECT_EQ(actual.open_area, expected.open_area);
    EXPECT_EQ(actual.blocked_area, expected.blocked_area);
    EXPECT_EQ(actual.pressure_to_velocity_scale, expected.pressure_to_velocity_scale);
    EXPECT_EQ(actual.conductance, expected.conductance);
    EXPECT_EQ(actual.predicted_normal_velocity, expected.predicted_normal_velocity);
    EXPECT_EQ(actual.prescribed_wall_flux, expected.prescribed_wall_flux);
    const auto* internal = std::get_if<InternalMacPressureLink3D>(&expected.pressure_link);
    const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&expected.pressure_link);
    EXPECT_EQ(actual.jump, internal ? internal->pressure_jump : 0);
    EXPECT_EQ(actual.exterior_pressure, boundary ? boundary->exterior_pressure : 0);
  }
  const auto* levels = reinterpret_cast<const cuda_internal::PressureLevel3D*>(
      arena->range(CudaLiquidRange3D::kPressureLevels).pinned.data()
  );
  const auto* blocks = reinterpret_cast<const cuda_internal::PressureBlock3D*>(
      arena->range(CudaLiquidRange3D::kPressureBlocks).pinned.data()
  );
  const auto* indices =
      reinterpret_cast<const std::size_t*>(arena->range(CudaLiquidRange3D::kPressureIndices).pinned.data());
  const auto* coefficients =
      reinterpret_cast<const double*>(arena->range(CudaLiquidRange3D::kPressureCoefficients).pinned.data());
  EXPECT_TRUE(
      std::equal(prepared->hierarchy.unknown_to_cell.begin(), prepared->hierarchy.unknown_to_cell.end(), indices)
  );
  std::size_t total_unknowns = 0;
  for (std::size_t ordinal = 0; ordinal < prepared->hierarchy.levels.size(); ++ordinal) {
    const auto& expected = prepared->hierarchy.levels[ordinal];
    const auto& actual = levels[ordinal];
    EXPECT_EQ(actual.count, expected.coordinates.size());
    EXPECT_EQ(actual.scratch_offset, total_unknowns);
    total_unknowns += expected.coordinates.size();
    const auto equal = [](const auto span, const auto* values) {
      return std::equal(span.begin(), span.end(), values);
    };
    EXPECT_TRUE(equal(expected.anchor_diagonal, coefficients + actual.anchor_offset));
    EXPECT_TRUE(equal(expected.diagonal, coefficients + actual.diagonal_offset));
    EXPECT_TRUE(equal(expected.row_offsets, indices + actual.row_offset));
    EXPECT_TRUE(equal(expected.aggregate_ordinals, indices + actual.aggregate_offset));
    for (std::size_t edge = 0; edge < expected.edges.size(); ++edge) {
      EXPECT_EQ(coefficients[actual.edge_offset + edge], expected.edges[edge].conductance);
    }
    for (std::size_t neighbor = 0; neighbor < expected.neighbors.size(); ++neighbor) {
      EXPECT_EQ(indices[actual.neighbor_offset + neighbor], expected.neighbors[neighbor].unknown_ordinal);
      EXPECT_EQ(indices[actual.neighbor_edge_offset + neighbor], expected.neighbors[neighbor].edge_ordinal);
    }
    const auto terminal_count = expected.terminal_offsets.empty() ? 0 : expected.terminal_offsets.size() - 1;
    EXPECT_EQ(actual.block_count, terminal_count);
    std::size_t factor_offset = 0;
    for (std::size_t block = 0; block < terminal_count; ++block) {
      const auto& packed = blocks[actual.block_offset + block];
      const auto start = expected.terminal_offsets[block];
      const auto count = expected.terminal_offsets[block + 1] - start;
      EXPECT_EQ(packed.count, count);
      EXPECT_TRUE(equal(expected.terminal_unknowns.subspan(start, count), indices + packed.unknown_offset));
      EXPECT_TRUE(
          equal(expected.terminal_factors.subspan(factor_offset, count * count), coefficients + packed.factor_offset)
      );
      factor_offset += count * count;
    }
  }
  std::cout << "maximum_pressure_cells=" << problem.cells.size() << " faces=" << problem.faces.size()
            << " levels=" << prepared->hierarchy.levels.size() << " level_unknowns=" << total_unknowns
            << " host_preparation_used=" << scratch.used() << " host_preparation_high_water=" << scratch.high_water()
            << " iterations=" << result->diagnostics.iteration_count
            << " physical_defect=" << result->diagnostics.max_cell_constraint_defect << '\n';
}
#if defined(_WIN32) && defined(_DEBUG)
std::size_t pressure_allocations = 0;
std::size_t pressure_frees = 0;
_CRT_ALLOC_HOOK previous_pressure_hook = nullptr;
int PressureAllocationHook(
    int operation,
    void* data,
    std::size_t size,
    int block_type,
    long request,
    const unsigned char* filename,
    int line
) {
  if (operation == _HOOK_ALLOC || operation == _HOOK_REALLOC) {
    ++pressure_allocations;
  }
  if (operation == _HOOK_FREE) {
    ++pressure_frees;
  }
  return previous_pressure_hook ? previous_pressure_hook(operation, data, size, block_type, request, filename, line)
                                : 1;
}
TEST(CudaLiquidPressureStage3D, SuccessfulAndRejectedProjectionAllocateAndFreeNothing) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto problem = MakeAffineProblem(-3, 4, false);
  auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  auto arena = CudaLiquidArena3D::Create({1, 128, 4096, 24576, 0});
  auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  const auto before = arena->metrics();
  bool success = false;
  bool exhausted = false;
  bool incompatible = false;
  bool invalid = false;
  pressure_allocations = 0;
  pressure_frees = 0;
  previous_pressure_hook = _CrtSetAllocHook(PressureAllocationHook);
  {
    success =
        cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace)
            .ok();
    exhausted =
        cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {1}, *workspace)
            .status()
            .code()
        == absl::StatusCode::kAborted;
    problem.cells[0].target_divergence = 1;
    incompatible =
        cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace)
            .status()
            .code()
        == absl::StatusCode::kFailedPrecondition;
    invalid = cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {}, {}, *workspace).status().code()
        == absl::StatusCode::kInvalidArgument;
  }
  _CrtSetAllocHook(previous_pressure_hook);
  problem.cells[0].target_divergence = 0;
  bool runtime_failed = false;
  bool remains_poisoned = false;
  {
    PressureRuntimeScope fault(1);
    previous_pressure_hook = _CrtSetAllocHook(PressureAllocationHook);
    {
      runtime_failed = cuda_internal::ProjectLiquidPressure3D(
                           *arena,
                           grid->view(),
                           1,
                           {problem.cells, problem.faces},
                           {},
                           *workspace
                       )
                           .status()
                           .code()
          == absl::StatusCode::kInternal;
      remains_poisoned = cuda_internal::ProjectLiquidPressure3D(
                             *arena,
                             grid->view(),
                             1,
                             {problem.cells, problem.faces},
                             {},
                             *workspace
                         )
                             .status()
                             .code()
          == absl::StatusCode::kInternal;
    }
    _CrtSetAllocHook(previous_pressure_hook);
  }
  EXPECT_TRUE(runtime_failed);
  EXPECT_TRUE(remains_poisoned);
  EXPECT_TRUE(success);
  EXPECT_TRUE(exhausted);
  EXPECT_TRUE(incompatible);
  EXPECT_TRUE(invalid);
  EXPECT_EQ(pressure_allocations, 0);
  EXPECT_EQ(pressure_frees, 0);
  const auto after = arena->metrics();
  EXPECT_EQ(before.device_reservations, after.device_reservations);
  EXPECT_EQ(before.pinned_reservations, after.pinned_reservations);
  EXPECT_EQ(before.device_frees, after.device_frees);
  EXPECT_EQ(before.pinned_frees, after.pinned_frees);
  EXPECT_EQ(before.stream_creations, after.stream_creations);
  EXPECT_EQ(before.event_creations, after.event_creations);
  EXPECT_EQ(before.graph_creations, after.graph_creations);
  EXPECT_EQ(before.executable_creations, after.executable_creations);
  std::cout << "pressure_attempt_crt_allocations=" << pressure_allocations << " frees=" << pressure_frees << '\n';
}
#endif

TEST(CudaLiquidPressureStage3D, ReacquiresRangesAfterGrowthAndPreservesEpochAndOutputBorrows) {
  ASSERT_NO_FATAL_FAILURE(CheckHardware());
  auto problem = MakeAffineProblem(-3, 4, false);
  auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok());
  SortFaces(problem, *grid);
  auto arena = CudaLiquidArena3D::Create({1, 128, 64, 24576, 0});
  auto workspace = MacPressureWorkspace3D::Create(4096, 24576);
  ASSERT_TRUE(arena.ok());
  ASSERT_TRUE(workspace.ok());
  const LiquidState3D frame_start;
  ASSERT_TRUE(arena->Initialize(frame_start).ok());
  auto lease = arena->AcquireCommitted();
  ASSERT_TRUE(lease.ok());
  const auto lease_identity = lease->identity();
  std::vector<std::byte> committed_bytes(lease->device_bytes().size());
  ASSERT_EQ(
      cudaMemcpy(committed_bytes.data(), lease->device_bytes().data(), committed_bytes.size(), cudaMemcpyDeviceToHost),
      cudaSuccess
  );
  const auto first =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(first.ok());
  const auto expected = CopyProjection(*first);
  const auto old_range = arena->range(CudaLiquidRange3D::kPressureCells).device;
  ASSERT_TRUE(arena->PrepareGrowth({2, 128, 4096, 24576, 0}, frame_start).ok());
  arena->CommitGrowth();
  EXPECT_NE(old_range, arena->range(CudaLiquidRange3D::kPressureCells).device);
  pressure_test::ExpectMacPressureFixtureProjection3D(CopyProjection(*first), expected, 0, 0);
  const auto next =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(next.ok());
  pressure_test::ExpectMacPressureFixtureProjection3D(CopyProjection(*next), expected, 0, 0);
  auto cpu = ProjectMacPressure3D(grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(cpu.ok());
  const auto cpu_saved = CopyProjection(*cpu);
  EXPECT_EQ(
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {1}, *workspace)
          .status()
          .code(),
      absl::StatusCode::kAborted
  );
  pressure_test::ExpectMacPressureFixtureProjection3D(CopyProjection(*cpu), cpu_saved, 0, 0);
  const auto final =
      cuda_internal::ProjectLiquidPressure3D(*arena, grid->view(), 1, {problem.cells, problem.faces}, {}, *workspace);
  ASSERT_TRUE(final.ok());
  pressure_test::ExpectMacPressureFixtureProjection3D(CopyProjection(*final), expected, 0, 0);
  std::vector<std::byte> retained(committed_bytes.size());
  ASSERT_EQ(
      cudaMemcpy(retained.data(), lease->device_bytes().data(), retained.size(), cudaMemcpyDeviceToHost),
      cudaSuccess
  );
  EXPECT_EQ(retained, committed_bytes);
  EXPECT_EQ(lease->identity(), lease_identity);
}
}  // namespace
}  // namespace fluidity
