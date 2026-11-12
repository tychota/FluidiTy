#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

#include "fluidity/host/cuda_mac_pressure_3d.h"
#include "gtest/gtest.h"
#include "mac_pressure_3d_fixture.h"
#include "mac_pressure_explicit_cases.h"
namespace fluidity {
namespace {
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr UniformMacGeometry3D kUnitGeometry{{0.0, 0.0, 0.0}, 1.0};
std::int32_t FixtureOwnerAxis(std::int32_t index) {
  return index / 4 - (index % 4 < 0 ? 1 : 0);
}
void ExpectDeviceAndMeasurement(CudaOperatorMeasurement measurement) {
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  EXPECT_EQ(properties.major, 12);
  EXPECT_EQ(properties.minor, 0);
#if defined(FLUIDITY_QUALIFIED_CUDA_DEVICE)
  EXPECT_STREQ(properties.name, FLUIDITY_QUALIFIED_CUDA_DEVICE);
#endif
  EXPECT_EQ(measurement.compiled_architecture, 1200);
  EXPECT_TRUE(std::isfinite(measurement.kernel_ms));
  EXPECT_GE(measurement.kernel_ms, 0.0F);
}
void Receipt(const char* name, CudaOperatorMeasurement measurement) {
  std::cout << "operator=" << name << " compiled_architecture=" << measurement.compiled_architecture
            << " kernel_ms=" << std::setprecision(9) << measurement.kernel_ms << '\n';
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

TEST(CudaMacPressure3D, ExplicitGraphMatchesEdgeAndCornerReferencePressure) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  CudaOperatorMeasurement maximum_measurement{};
  for (const auto& pressure_case : pressure_test::kExplicitPressureCases) {
    SCOPED_TRACE(pressure_case.plane_count);
    SCOPED_TRACE(pressure_case.reversed);
    const auto fixture = pressure_test::MakeExplicitPressureFixture(pressure_case);
    const auto system = pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, fixture.cells, fixture.faces);
    ASSERT_TRUE(system.ok()) << system.status();
    const auto result = pressure_internal::CudaProjectPreparedMacPressure3D(*system);
    ASSERT_TRUE(result.ok()) << result.status();
    const auto& projection = result->projection;
    ExpectDeviceAndMeasurement(result->measurement);
    if (result->measurement.kernel_ms >= maximum_measurement.kernel_ms) {
      maximum_measurement = result->measurement;
    }
    EXPECT_GT(projection.diagnostics.iteration_count, 0);
    EXPECT_LE(projection.diagnostics.max_cell_constraint_defect, 2e-5);
    EXPECT_LE(projection.diagnostics.max_volume_scaled_residual, 2e-5);
    ASSERT_EQ(projection.cell_pressures.size(), fixture.cells.size());
    for (std::size_t cell = 0; cell < fixture.cells.size(); ++cell) {
      EXPECT_NEAR(projection.cell_pressures[cell], pressure_case.reference_pressure[cell], 2e-5);
    }
    const auto oracle_pressure = std::span(pressure_case.reference_pressure).first(fixture.cells.size());
    const auto oracle = pressure_internal::EvaluateMacPressure3D(*system, oracle_pressure);
    ASSERT_TRUE(oracle.ok()) << oracle.status();
    ASSERT_EQ(projection.corrected_face_velocities.size(), fixture.faces.size());
    double max_face_flux_error = 0.0;
    for (std::size_t face = 0; face < fixture.faces.size(); ++face) {
      const double velocity_error =
          projection.corrected_face_velocities[face] - oracle->corrected_face_velocities[face];
      max_face_flux_error = std::max(max_face_flux_error, std::abs(fixture.faces[face].open_area * velocity_error));
    }
    EXPECT_LE(max_face_flux_error, 5e-6);
    std::cout << "explicit_graph_cuda planes=" << pressure_case.plane_count << " reversed=" << pressure_case.reversed
              << " iterations=" << projection.diagnostics.iteration_count
              << " max_divergence=" << projection.diagnostics.max_cell_constraint_defect
              << " max_face_flux_error=" << max_face_flux_error << '\n';
  }
  Receipt("explicit_coarse_fine_pressure", maximum_measurement);
}

TEST(CudaMacPressure3D, MixedLinksMatchOracle) {
  std::vector<pressure_test::MacPressureFixtureCase3D> cases;
  ASSERT_NO_FATAL_FAILURE(pressure_test::LoadMacPressureFixtureCases3D(cases));
  ASSERT_EQ(cases.size(), 8);
  for (const auto& fixture : cases) {
    SCOPED_TRACE(fixture.name);
    const auto result = CudaProjectMacPressure3D({fixture.grid, fixture.step_duration, fixture.cells, fixture.faces});
    ASSERT_TRUE(result.ok()) << result.status();
    ExpectDeviceAndMeasurement(result->measurement);
    pressure_test::ExpectMacPressureFixtureProjection3D(
        result->projection,
        fixture.expected_projection,
        fixture.absolute_tolerance,
        fixture.relative_tolerance
    );
    double maximum_pressure_error = 0.0;
    for (std::size_t i = 0; i < fixture.cells.size(); ++i) {
      maximum_pressure_error = std::max(
          maximum_pressure_error,
          std::abs(result->projection.cell_pressures[i] - fixture.expected_projection.cell_pressures[i])
      );
    }
    if (fixture.name == "mixed") {
      std::cout << "mixed_max_pressure_error=" << std::setprecision(17) << maximum_pressure_error << '\n';
      Receipt("pressure_3d_mixed", result->measurement);
    }
  }
}
TEST(CudaMacPressure3D, MultilevelProjectionMatchesCpu) {
  auto problem = MakeAffineProblem(-3, 4, false);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const MacPressureInputs3D inputs{*grid, 1.0, problem.cells, problem.faces};
  const auto cpu = ProjectMacPressure3D(inputs);
  ASSERT_TRUE(cpu.ok()) << cpu.status();
  const auto result = CudaProjectMacPressure3D(inputs);
  ASSERT_TRUE(result.ok()) << result.status();
  ExpectDeviceAndMeasurement(result->measurement);
  const auto& projection = result->projection;
  EXPECT_GT(projection.diagnostics.iteration_count, 1);
  EXPECT_GE(projection.diagnostics.multigrid_level_count, 2);
  EXPECT_LE(projection.diagnostics.max_volume_scaled_residual, 2e-5);
  EXPECT_LE(projection.diagnostics.max_cell_constraint_defect, 2e-5);
  pressure_test::ExpectMacPressureFixtureProjection3D(projection, *cpu, 1e-9, 1e-9);
  double maximum_pressure_error = 0.0;
  double maximum_velocity_error = 0.0;
  for (std::size_t i = 0; i < problem.cells.size(); ++i) {
    const double expected = AffinePressure(problem.cells[i].cell_index) + 21.0;
    maximum_pressure_error = std::max(maximum_pressure_error, std::abs(projection.cell_pressures[i] - expected));
    EXPECT_NEAR(projection.cell_pressures[i], expected, 5e-5);
    EXPECT_NEAR(projection.true_residuals[i], -projection.cell_constraint_defects[i], 1e-12);
  }
  for (double velocity : projection.corrected_face_velocities) {
    maximum_velocity_error = std::max(maximum_velocity_error, std::abs(velocity));
    EXPECT_NEAR(velocity, 0.0, 5e-5);
  }
  std::cout << "iterations=" << projection.diagnostics.iteration_count
            << " levels=" << projection.diagnostics.multigrid_level_count << " p_error=" << std::setprecision(17)
            << maximum_pressure_error << " u_error=" << maximum_velocity_error
            << " residual_l2=" << projection.diagnostics.true_residual_l2
            << " defect=" << projection.diagnostics.max_cell_constraint_defect << '\n';
  Receipt("pressure_3d_multilevel", result->measurement);
  // Exercise all six Dirichlet orientations and a third level without adding
  // receipts or conflating these timings with the named 64-cell fixture.
  for (const int extent : {4, 8}) {
    auto anchored = MakeAffineProblem(-4, extent, true);
    const auto anchored_grid = SparseMacGrid3D::Create(kUnitGeometry, anchored.blocks, anchored.blocks.size());
    ASSERT_TRUE(anchored_grid.ok()) << anchored_grid.status();
    SortFaces(anchored, *anchored_grid);
    const MacPressureInputs3D anchored_inputs{*anchored_grid, 1.0, anchored.cells, anchored.faces};
    const auto anchored_cpu = ProjectMacPressure3D(anchored_inputs);
    ASSERT_TRUE(anchored_cpu.ok()) << anchored_cpu.status();
    const auto anchored_gpu = CudaProjectMacPressure3D(anchored_inputs);
    ASSERT_TRUE(anchored_gpu.ok()) << anchored_gpu.status();
    EXPECT_GT(anchored_gpu->projection.diagnostics.iteration_count, 0);
    EXPECT_GE(anchored_gpu->projection.diagnostics.multigrid_level_count, extent == 8 ? 3 : 2);
    pressure_test::ExpectMacPressureFixtureProjection3D(anchored_gpu->projection, *anchored_cpu, 1e-9, 1e-9);
    for (std::size_t i = 0; i < anchored.cells.size(); ++i) {
      EXPECT_NEAR(anchored_gpu->projection.cell_pressures[i], AffinePressure(anchored.cells[i].cell_index), 5e-5);
    }
    for (double velocity : anchored_gpu->projection.corrected_face_velocities) {
      EXPECT_NEAR(velocity, 0.0, 5e-5);
    }
  }
}
TEST(CudaMacPressure3D, InteriorMovingWallFluxMatchesHostPhysicalDefect) {
  auto problem = MakeAffineProblem(0, 1, false);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  auto& cell = problem.cells.front();
  cell.fluid_volume = 1.0;
  cell.bulk_modulus = 2.0;
  cell.previous_pressure = 4.0;
  cell.integrated_inward_wall_flux = 0.5;
  const MacPressureInputs3D inputs{*grid, 1.0, problem.cells, problem.faces};
  const auto result = CudaProjectMacPressure3D(inputs);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->projection.integrated_rhs.front(), 2.5);
  EXPECT_DOUBLE_EQ(result->projection.cell_pressures.front(), 5.0);
  EXPECT_DOUBLE_EQ(result->projection.cell_constraint_defects.front(), 0.0);
  cell.bulk_modulus = kInfinity;
  EXPECT_EQ(CudaProjectMacPressure3D(inputs).status().code(), absl::StatusCode::kFailedPrecondition);
}
TEST(CudaMacPressure3D, RejectsIncompatibleAndUnconverged) {
  auto problem = MakeAffineProblem(-3, 4, false);
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, problem.blocks, problem.blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  SortFaces(problem, *grid);
  const MacPressureInputs3D inputs{*grid, 1.0, problem.cells, problem.faces};
  const auto exhausted = CudaProjectMacPressure3D(inputs, {.max_iterations = 1});
  EXPECT_EQ(exhausted.status().code(), absl::StatusCode::kAborted);
  EXPECT_NE(exhausted.status().message().find("iteration budget"), std::string_view::npos);
  const auto converged = CudaProjectMacPressure3D(inputs);
  ASSERT_TRUE(converged.ok()) << converged.status();
  EXPECT_GT(converged->projection.diagnostics.iteration_count, 1);
  problem.cells[0].target_divergence = 1.0;
  const auto incompatible = CudaProjectMacPressure3D(inputs);
  EXPECT_EQ(incompatible.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(problem.cells[0].previous_pressure, 0.0);
  for (const std::size_t budget : {std::size_t{0}, std::size_t{257}}) {
    EXPECT_EQ(
        CudaProjectMacPressure3D(inputs, {.max_iterations = budget}).status().code(),
        absl::StatusCode::kInvalidArgument
    );
  }
  auto singleton = MakeAffineProblem(0, 1, false);
  const auto singleton_grid = SparseMacGrid3D::Create(kUnitGeometry, singleton.blocks, singleton.blocks.size());
  ASSERT_TRUE(singleton_grid.ok()) << singleton_grid.status();
  SortFaces(singleton, *singleton_grid);
  auto& cell = singleton.cells.front();
  cell.fluid_volume = 1e-6;
  cell.target_divergence = 1e6;
  cell.bulk_modulus = 1e-6 / 1.1;
  const MacPressureInputs3D single_inputs{*singleton_grid, 1.0, singleton.cells, singleton.faces};
  // FP32 recursive zero after one step is not true convergence. The same
  // budget must count the reliable replacement and restarted correction.
  EXPECT_EQ(CudaProjectMacPressure3D(single_inputs, {.max_iterations = 1}).status().code(), absl::StatusCode::kAborted);
  const auto replaced = CudaProjectMacPressure3D(single_inputs);
  ASSERT_TRUE(replaced.ok()) << replaced.status();
  EXPECT_GE(replaced->projection.diagnostics.iteration_count, 2);
  EXPECT_NEAR(replaced->projection.cell_pressures[0], 1.0 / 1.1, 2e-11);
  EXPECT_LE(replaced->projection.diagnostics.max_volume_scaled_residual, 2e-5);
  EXPECT_LE(replaced->projection.diagnostics.max_cell_constraint_defect, 2e-5);
  // A small integrated RHS does not excuse a large volume-scaled defect.
  cell.bulk_modulus = kInfinity;
  cell.target_divergence = 0.1;
  auto& boundary = singleton.faces[1];
  boundary.open_area = 1.0;
  boundary.inverse_density = 1.0;
  boundary.pressure_gradient_distance = 1.0;
  boundary.pressure_link = DirichletMacPressureLink3D{0.0};
  const auto tiny = CudaProjectMacPressure3D(single_inputs);
  ASSERT_TRUE(tiny.ok()) << tiny.status();
  EXPECT_GT(tiny->projection.diagnostics.iteration_count, 0);
  EXPECT_NEAR(tiny->projection.cell_pressures[0], 1e-7, 2e-11);
  EXPECT_LE(tiny->projection.diagnostics.max_cell_constraint_defect, 2e-5);
  constexpr double quantum = std::numeric_limits<double>::denorm_min();
  // Preparation accepts this warm start. The device original operator must
  // reject a lossy subnormal product, while the exact-subnormal control passes.
  cell.fluid_volume = 1.0;
  cell.target_divergence = 0.0;
  cell.previous_pressure = quantum;
  boundary.open_area = 0.25;
  EXPECT_EQ(ProjectMacPressure3D(single_inputs).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(CudaProjectMacPressure3D(single_inputs).status().code(), absl::StatusCode::kOutOfRange);
  cell.previous_pressure = 4.0 * quantum;
  const auto exact = CudaProjectMacPressure3D(single_inputs);
  ASSERT_TRUE(exact.ok()) << exact.status();
  EXPECT_DOUBLE_EQ(exact->projection.cell_pressures[0], 4.0 * quantum);
  EXPECT_DOUBLE_EQ(exact->projection.true_residuals[0], -quantum);
  EXPECT_DOUBLE_EQ(exact->projection.cell_constraint_defects[0], quantum);
  EXPECT_EQ(exact->projection.diagnostics.iteration_count, 0);
  ExpectDeviceAndMeasurement(exact->measurement);
  // An operator product can be exact while velocity correction underflows.
  const UniformMacGeometry3D wide_geometry{{0.0, 0.0, 0.0}, 4.0};
  const auto wide_grid = SparseMacGrid3D::Create(wide_geometry, singleton.blocks, singleton.blocks.size());
  ASSERT_TRUE(wide_grid.ok()) << wide_grid.status();
  boundary.open_area = 16.0;
  boundary.inverse_density = 0.25;
  cell.previous_pressure = quantum;
  const MacPressureInputs3D wide_inputs{*wide_grid, 1.0, singleton.cells, singleton.faces};
  EXPECT_EQ(ProjectMacPressure3D(wide_inputs).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(CudaProjectMacPressure3D(wide_inputs).status().code(), absl::StatusCode::kOutOfRange);
  // Host rejection must retain the accepted preparation semantics too.
  boundary.open_area = 0.0;
  boundary.pressure_link = SealedMacPressureLink3D{};
  cell.previous_pressure = 0.0;
  cell.fluid_volume = quantum;
  cell.target_divergence = 0.25;
  EXPECT_EQ(CudaProjectMacPressure3D(single_inputs).status().code(), absl::StatusCode::kOutOfRange);
  cell.fluid_volume = 8.0 * quantum;
  boundary.wall_normal_velocity = 2.0 * quantum;
  const auto balanced = CudaProjectMacPressure3D(single_inputs);
  ASSERT_TRUE(balanced.ok()) << balanced.status();
  EXPECT_DOUBLE_EQ(balanced->projection.cell_constraint_defects[0], 0.0);
  EXPECT_EQ(balanced->projection.diagnostics.iteration_count, 0);
  ExpectDeviceAndMeasurement(balanced->measurement);
}
}  // namespace
}  // namespace fluidity
