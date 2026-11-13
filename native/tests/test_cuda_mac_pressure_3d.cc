#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "../cuda/liquid_pressure_stage_3d.h"
#include "../src/graded_liquid_boundary_geometry_3d.h"
#include "adaptive_pressure_checkpoint_fixture.h"
#include "curved_matched_pressure_fixture.h"
#include "fluidity/host/cuda_mac_pressure_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "gtest/gtest.h"
#include "mac_pressure_3d_fixture.h"
#include "mac_pressure_acoustic_cases.h"
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

TEST(CudaMacPressure3D, BackwardEulerStandingWaveMatchesAnalyticMode) {
  double kernel_milliseconds = 0.0;
  pressure_test::ExpectBackwardEulerAcousticColumn(
      pressure_test::AcousticInitialCondition::kStandingWave,
      [&](const auto& system) -> absl::StatusOr<MacPressureProjection3D> {
        auto result = pressure_internal::CudaProjectPreparedMacPressure3D(system);
        if (!result.ok()) {
          return result.status();
        }
        EXPECT_EQ(result->measurement.compiled_architecture, 1200);
        kernel_milliseconds += result->measurement.kernel_ms;
        return std::move(result->projection);
      }
  );
  RecordProperty("acoustic_kernel_milliseconds", std::to_string(kernel_milliseconds));
}

TEST(CudaMacPressure3D, BackwardEulerPressurePulseMatchesModalOracle) {
  double kernel_milliseconds = 0.0;
  pressure_test::ExpectBackwardEulerAcousticColumn(
      pressure_test::AcousticInitialCondition::kLocalizedPulse,
      [&](const auto& system) -> absl::StatusOr<MacPressureProjection3D> {
        auto result = pressure_internal::CudaProjectPreparedMacPressure3D(system);
        if (!result.ok()) {
          return result.status();
        }
        EXPECT_EQ(result->measurement.compiled_architecture, 1200);
        kernel_milliseconds += result->measurement.kernel_ms;
        return std::move(result->projection);
      }
  );
  RecordProperty("acoustic_kernel_milliseconds", std::to_string(kernel_milliseconds));
}

TEST(CudaMacPressure3D, WaterHammerValveClosureMatchesAcousticOracle) {
  double kernel_milliseconds = 0.0;
  pressure_test::ExpectBackwardEulerAcousticColumn(
      pressure_test::AcousticInitialCondition::kValveClosure,
      [&](const auto& system) -> absl::StatusOr<MacPressureProjection3D> {
        auto result = pressure_internal::CudaProjectPreparedMacPressure3D(system);
        if (!result.ok()) {
          return result.status();
        }
        EXPECT_EQ(result->measurement.compiled_architecture, 1200);
        kernel_milliseconds += result->measurement.kernel_ms;
        return std::move(result->projection);
      }
  );
  RecordProperty("acoustic_kernel_milliseconds", std::to_string(kernel_milliseconds));
}

TEST(CudaMacPressure3D, CompliantPistonConvergesToEosCompression) {
  double kernel_milliseconds = 0.0;
  pressure_test::ExpectCompliantPistonCompression(
      [&](const MacPressureInputs3D& inputs) -> absl::StatusOr<MacPressureProjection3D> {
        auto result = CudaProjectMacPressure3D(inputs);
        if (!result.ok()) {
          return result.status();
        }
        EXPECT_EQ(result->measurement.compiled_architecture, 1200);
        kernel_milliseconds += result->measurement.kernel_ms;
        return std::move(result->projection);
      }
  );
  RecordProperty("piston_kernel_milliseconds", std::to_string(kernel_milliseconds));
}

TEST(CudaMacPressure3D, SparseWetCoarseFineSubsetPreservesGlobalSlots) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  constexpr std::array keys{AdaptiveMacCellKey3D{{-2, -2, 0}, 1}, AdaptiveMacCellKey3D{{-2, 0, 0}, 0}};
  const auto topology = AdaptiveMacTopology3D::Create(keys, 1);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  ASSERT_TRUE(ownership.ok()) << ownership.status();
  ASSERT_EQ(ownership->interior_face_slots.size(), 1);
  const std::size_t shared_slot = ownership->interior_face_slots[0];
  bool observed_sparse_outlet = false;
  for (const std::size_t wet_leaf : {std::size_t{0}, std::size_t{1}}) {
    SCOPED_TRACE(wet_leaf);
    const auto geometry = FreezeGradedPlanarLiquidGeometry3D(
        *topology,
        {},
        {.unit_normal = {0, 1, 0}, .start_plane_coordinate = -10},
        {.unit_normal = {0, wet_leaf == 0 ? 1.0 : -1.0, 0}, .plane_coordinate_m = 0},
        0.25
    );
    ASSERT_TRUE(geometry.ok()) << geometry.status();
    std::array<LiquidAdaptiveCellState3D, 2> accepted{{
        {.key = keys[0], .occupancy = LiquidCellOccupancy3D::kAir},
        {.key = keys[1], .occupancy = LiquidCellOccupancy3D::kAir},
    }};
    accepted[wet_leaf].occupancy = LiquidCellOccupancy3D::kLiquid;
    accepted[wet_leaf].inventory = {.mass_kg = 1, .reference_volume_m3 = 1, .current_volume_m3 = 1};
    std::array<GradedLiquidPressureCellFields3D, 2> cells{};
    cells[wet_leaf].target_divergence_s_inv = 0.125;
    std::vector<GradedLiquidPressureSubfaceFields3D> faces(geometry->subfaces().size());
    const auto outlet = std::ranges::find_if(ownership->exterior_faces, [&](const auto& face) {
      return face.cell_ordinal == wet_leaf && face.face_index.axis == FaceNormalAxis::kX && !face.cell_is_tail;
    });
    ASSERT_NE(outlet, ownership->exterior_faces.end());
    faces[outlet->face_slot] = {
        .inverse_density_m3_per_kg = 1,
        .predicted_normal_velocity_m_per_s = 0.5,
        .exterior_pressure = GradedLiquidPressureExterior3D{.pressure_pa = 2, .distance_m = 0.5},
    };
    const auto subset = PrepareGradedWetPressureSubset3D(*geometry, accepted, cells, faces);
    ASSERT_TRUE(subset.ok()) << subset.status();
    ASSERT_EQ(subset->leaf_ordinals, std::vector<std::size_t>{wet_leaf});
    const auto& system = subset->system;
    const auto links = system.links();
    ASSERT_LT(links.size(), faces.size());
    auto cpu_workspace = MacPressureWorkspace3D::Create(system.cells().size(), links.size());
    ASSERT_TRUE(cpu_workspace.ok()) << cpu_workspace.status();
    const auto cpu = cpu_workspace->ProjectPrepared(system);
    ASSERT_TRUE(cpu.ok()) << cpu.status();
    const auto cuda = pressure_internal::CudaProjectPreparedMacPressure3D(system);
    ASSERT_TRUE(cuda.ok()) << cuda.status();
    ExpectDeviceAndMeasurement(cuda->measurement);
    EXPECT_GT(cuda->projection.diagnostics.iteration_count, 0);
    ASSERT_EQ(cuda->projection.cell_pressures.size(), 1);
    const double pressure_error = std::abs(cuda->projection.cell_pressures[0] - cpu->cell_pressures[0]);
    EXPECT_LE(pressure_error, 2e-5);
    EXPECT_LE(cuda->projection.diagnostics.max_cell_constraint_defect, 2e-5);
    ASSERT_EQ(cuda->projection.corrected_face_velocities.size(), links.size());

    std::vector<double> global_velocities(faces.size(), std::numeric_limits<double>::quiet_NaN());
    double maximum_velocity_error = 0;
    for (std::size_t link_ordinal = 0; link_ordinal < links.size(); ++link_ordinal) {
      const auto& link = links[link_ordinal];
      ASSERT_LT(link.face_slot, global_velocities.size());
      EXPECT_TRUE(std::isnan(global_velocities[link.face_slot]));
      const double cuda_velocity = cuda->projection.corrected_face_velocities[link_ordinal];
      global_velocities[link.face_slot] = cuda_velocity;
      maximum_velocity_error =
          std::max(maximum_velocity_error, std::abs(cuda_velocity - cpu->corrected_face_velocities[link_ordinal]));
      if (link.face_slot == shared_slot) {
        EXPECT_EQ(link.tail_cell_ordinal.has_value(), wet_leaf == 0);
        EXPECT_EQ(link.head_cell_ordinal.has_value(), wet_leaf == 1);
        EXPECT_DOUBLE_EQ(link.open_area, 0);
      }
      if (link.face_slot == outlet->face_slot) {
        observed_sparse_outlet |= link_ordinal != link.face_slot;
      }
    }
    EXPECT_LE(maximum_velocity_error, 2e-5);
    const double expected_outlet_flux = wet_leaf == 0 ? -1.0 : -0.125;
    const double expected_pressure = wet_leaf == 0 ? 3.5 : 3.25;
    EXPECT_NEAR(cuda->projection.cell_pressures[0], expected_pressure, 2e-5);
    EXPECT_NEAR(global_velocities[outlet->face_slot] * outlet->face_area, expected_outlet_flux, 2e-5);
    EXPECT_DOUBLE_EQ(global_velocities[shared_slot], 0);
    EXPECT_EQ(
        std::ranges::count_if(
            global_velocities,
            [](double velocity) {
              return std::isfinite(velocity);
            }
        ),
        links.size()
    );
    for (const auto& exterior : ownership->exterior_faces) {
      if (exterior.cell_ordinal != wet_leaf) {
        EXPECT_TRUE(std::isnan(global_velocities[exterior.face_slot]));
      }
    }
    std::cout << "wet_leaf=" << wet_leaf << " global_slots=" << faces.size() << " pressure_links=" << links.size()
              << " shared_slot=" << shared_slot << " outlet_slot=" << outlet->face_slot
              << " iterations=" << cuda->projection.diagnostics.iteration_count
              << " pressure_pa=" << cuda->projection.cell_pressures[0]
              << " outlet_flux_m3_per_s=" << global_velocities[outlet->face_slot] * outlet->face_area
              << " max_pressure_error_pa=" << pressure_error << " max_velocity_error_m_per_s=" << maximum_velocity_error
              << " max_defect_per_s=" << cuda->projection.diagnostics.max_cell_constraint_defect << '\n';
    Receipt("sparse_wet_coarse_fine_pressure", cuda->measurement);
  }
  EXPECT_TRUE(observed_sparse_outlet);
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
  auto fixture = pressure_test::MakeAdaptiveCutPressureFixture();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const auto& topology = fixture->topology;
  const auto& cells = fixture->cells;
  auto& subfaces = fixture->subfaces;
  const auto internal_slot = fixture->internal_slot;
  const auto physical_graph = pressure_internal::PrepareAdaptiveMacPressureSystem3D(topology, 1.0, cells, subfaces);
  ASSERT_TRUE(physical_graph.ok()) << physical_graph.status();
  const auto physical_result = pressure_internal::CudaProjectPreparedMacPressure3D(*physical_graph);
  ASSERT_TRUE(physical_result.ok()) << physical_result.status();
  ExpectDeviceAndMeasurement(physical_result->measurement);
  const auto& physical_projection = physical_result->projection;
  ASSERT_EQ(physical_projection.cell_pressures.size(), cells.size());
  EXPECT_NEAR(physical_projection.cell_pressures[0], 1.28125, 2e-5);
  EXPECT_NEAR(physical_projection.cell_pressures[1], 1.09375, 2e-5);
  EXPECT_NEAR(physical_projection.corrected_face_velocities[internal_slot], 0.25, 2e-5);
  EXPECT_NEAR(physical_projection.corrected_face_velocities[fixture->outlet_slot], 0.375, 2e-5);
  EXPECT_LE(physical_projection.diagnostics.max_cell_constraint_defect, 2e-5);
  std::cout << "adaptive_cut_subface_cuda max_divergence=" << physical_projection.diagnostics.max_cell_constraint_defect
            << '\n';
  if (physical_result->measurement.kernel_ms >= maximum_measurement.kernel_ms) {
    maximum_measurement = physical_result->measurement;
  }
  pressure_test::QualifyAdaptivePressureCheckpoint([](const auto& system) -> absl::StatusOr<MacPressureProjection3D> {
    auto projected = pressure_internal::CudaProjectPreparedMacPressure3D(system);
    if (!projected.ok()) {
      return projected.status();
    }
    EXPECT_EQ(projected->measurement.compiled_architecture, 1200);
    return std::move(projected->projection);
  });
  Receipt("explicit_coarse_fine_pressure", maximum_measurement);
}

TEST(CudaMacPressure3D, MatchesUniformCurvedPressureAndGeometryErrorInArena) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  constexpr LiquidStateCapacities3D capacity{1, 0, 448, 1520, 0};
  auto arena = CudaLiquidArena3D::Create(capacity, 0, std::numeric_limits<std::size_t>::max(), true, true);
  ASSERT_TRUE(arena.ok()) << arena.status();
  auto workspace = MacPressureWorkspace3D::Create(capacity.cells, capacity.faces);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  arena->SetWorkPhase(LiquidWorkPhase3D::kForcesGeometryPressure);
  const auto before = arena->metrics();
  pressure_test::CheckCurvedMatchedPressure([&](const auto& graph) {
    return cuda_internal::ProjectArenaAdaptivePressure3D(*arena, graph, {}, *workspace);
  });
  const auto after = arena->metrics();
  EXPECT_EQ(after.compiled_architecture, 1200);
  EXPECT_EQ(after.device_reservations, before.device_reservations);
  EXPECT_EQ(after.device_frees, before.device_frees);
  EXPECT_EQ(after.live_device_bytes, before.live_device_bytes);
  EXPECT_GT(after.peak_active_device_scratch_bytes, 0U);
}

TEST(CudaMacPressure3D, ExplicitAdaptiveGraphUsesPersistentArenaAndKeepsPublishedResultOnFailure) {
  const auto fixture = pressure_test::MakeAdaptiveCutPressureFixture();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const auto graph =
      pressure_internal::PrepareAdaptiveMacPressureSystem3D(fixture->topology, 1, fixture->cells, fixture->subfaces);
  ASSERT_TRUE(graph.ok()) << graph.status();
  const auto independent = pressure_internal::CudaProjectPreparedMacPressure3D(*graph);
  ASSERT_TRUE(independent.ok()) << independent.status();

  const LiquidStateCapacities3D capacity{1, 0, graph->cells().size(), graph->links().size(), 0};
  auto arena = CudaLiquidArena3D::Create(capacity, 0, std::numeric_limits<std::size_t>::max(), true, true);
  ASSERT_TRUE(arena.ok()) << arena.status();
  auto workspace = MacPressureWorkspace3D::Create(capacity.cells, capacity.faces);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  arena->SetWorkPhase(LiquidWorkPhase3D::kForcesGeometryPressure);
  const auto before = arena->metrics();
  const auto projection = cuda_internal::ProjectArenaAdaptivePressure3D(*arena, *graph, {}, *workspace);
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_TRUE(std::ranges::equal(projection->cell_pressures, independent->projection.cell_pressures));
  EXPECT_TRUE(
      std::ranges::equal(projection->corrected_face_velocities, independent->projection.corrected_face_velocities)
  );
  EXPECT_EQ(projection->diagnostics.iteration_count, independent->projection.diagnostics.iteration_count);
  EXPECT_LE(projection->diagnostics.max_cell_constraint_defect, 2e-5);
  const auto refused = cuda_internal::ProjectArenaAdaptivePressure3D(*arena, *graph, {.max_iterations = 0}, *workspace);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(std::ranges::equal(projection->cell_pressures, independent->projection.cell_pressures));
  const auto after = arena->metrics();
  EXPECT_EQ(after.device_reservations, before.device_reservations);
  EXPECT_EQ(after.device_frees, before.device_frees);
  EXPECT_EQ(after.live_device_bytes, before.live_device_bytes);
  EXPECT_GT(after.peak_active_device_scratch_bytes, 0U);
  EXPECT_LE(after.peak_device_bytes, after.device_budget_bytes);
  std::cout << "adaptive_pressure_device_bytes=" << after.live_device_bytes
            << " active_scratch_bytes=" << after.peak_active_device_scratch_bytes << '\n';
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
      // Keep the operator fixed and shrink every RHS contribution below the
      // absolute global tolerance. Linear pressure should shrink by the same factor.
      constexpr double scale = 1e-8;
      auto weak_cells = fixture.cells;
      auto weak_faces = fixture.faces;
      for (auto& cell : weak_cells) {
        cell.target_divergence *= scale;
        cell.previous_pressure *= scale;
      }
      for (auto& face : weak_faces) {
        face.predicted_normal_velocity *= scale;
        face.wall_normal_velocity *= scale;
        if (face.prescribed_wall_flux) {
          *face.prescribed_wall_flux *= scale;
        }
        if (auto* jump = std::get_if<InternalMacPressureLink3D>(&face.pressure_link)) {
          jump->pressure_jump *= scale;
        }
        if (auto* exterior = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link)) {
          exterior->exterior_pressure *= scale;
        }
      }
      const MacPressureInputs3D weak_input{fixture.grid, fixture.step_duration, weak_cells, weak_faces};
      const auto weak = CudaProjectMacPressure3D(weak_input);
      ASSERT_TRUE(weak.ok()) << weak.status();
      EXPECT_GT(weak->projection.diagnostics.iteration_count, 0);
      for (std::size_t index = 0; index < weak_cells.size(); ++index) {
        const double expected = scale * fixture.expected_projection.cell_pressures[index];
        EXPECT_NEAR(weak->projection.cell_pressures[index], expected, 5e-3 * std::abs(expected) + 1e-12);
      }
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
