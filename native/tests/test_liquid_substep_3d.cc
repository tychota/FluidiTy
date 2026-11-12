#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#include "../src/liquid_substep_3d_internal.h"
#include "../src/viscosity_velocity_block_3d.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_substep_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "gtest/gtest.h"
#include "liquid_cavity_scheduler_cases.h"
#include "liquid_eos_hydrostatic_cases.h"
#include "liquid_regular_cell_transport_3d_cases.h"
#include "liquid_tank_staging_3d_cases.h"
#include "liquid_viscous_tank_cases.h"
#include "liquid_wall_sample_3d_cases.h"
#include "wetted_piston_pressure_3d_cases.h"
namespace fluidity {
namespace {
LiquidState3D Initial(std::size_t count = 1) {
  LiquidState3D state;
  state.particles = *LiquidRecords3D<LiquidParticleState3D>::Create(count);
  EXPECT_TRUE(state.particles.Resize(count).ok());
  state.next_particle_id = count + 1;
  for (std::size_t i = 0; i < count; ++i) {
    state.particles[i].id = i + 1;
    state.particles[i].world_position = {2.5, 2.5, 2.5};
    state.particles[i].pic_initialized = true;
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
  return scene;
}
// Serialize through the production checkpoint codec so storage omitted by a
// hand-picked field comparison cannot pass the F9 bridge.
void ExpectIdenticalSerializedState(const LiquidState3D& before, const LiquidState3D& after) {
  const auto before_bytes = EncodeLiquidState3D(before);
  const auto after_bytes = EncodeLiquidState3D(after);
  ASSERT_TRUE(before_bytes.ok()) << before_bytes.status();
  ASSERT_TRUE(after_bytes.ok()) << after_bytes.status();
  const auto before_digest = LiquidContentDigest3D(before_bytes->values());
  const auto after_digest = LiquidContentDigest3D(after_bytes->values());
  EXPECT_TRUE(std::ranges::equal(before_bytes->values(), after_bytes->values()));
  EXPECT_EQ(before_digest, after_digest);
  std::cout << "f9_state_before=" << LiquidDigestHex3D(before_digest).data()
            << " f9_state_after=" << LiquidDigestHex3D(after_digest).data() << '\n';
}

TEST(LiquidSubstep3D, EosHydrostaticCompressionMatchesColumn) {
  pressure_test::ExpectEosHydrostaticColumn(
      [](const LiquidState3D& initial, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options
      ) -> absl::StatusOr<pressure_test::HydrostaticRun> {
        auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
        if (!scheduler.ok()) {
          return scheduler.status();
        }
        return pressure_test::AdvanceHydrostaticColumn(*scheduler, options.capacities);
      }
  );
}

TEST(LiquidSubstep3D, SchedulerConsumesResolvedViscosityOnFullCells) {
  auto initial = Initial(2);
  initial.particles[0].world_position = {2.5, 2.5, 2.5};
  initial.particles[1].world_position = {3.5, 2.5, 2.5};
  initial.particles[0].world_velocity = {0.0, 0.0, 0.125};
  initial.particles[1].world_velocity = {0.0, 0.0, -0.125};
  LiquidResolvedSceneView3D request;
  request.scene.field = LiquidHorizontalPlaneSurface3D{3.0};
  request.scene.obstacle = AxisAlignedLiquidTank3D{2.0, 4.0, 2.0, 2.0, 4.0, 0.0};
  request.scene.world_acceleration = {};
  request.scene.dynamic_viscosity = 100.0;
  const auto encoded = EncodeLiquidResolvedScene3D(request);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto resolved = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  auto scene = resolved->View().scene;
  auto viscous = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(viscous.ok()) << viscous.status();
  const auto advanced = viscous->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  scene.dynamic_viscosity = 0.0;
  auto inviscid = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(inviscid.ok()) << inviscid.status();
  const auto inviscid_advanced = inviscid->AdvanceFrame(0.01);
  ASSERT_TRUE(inviscid_advanced.ok()) << inviscid_advanced;
  EXPECT_GT(
      std::abs(inviscid->committed().particles[0].world_velocity.z)
          - std::abs(viscous->committed().particles[0].world_velocity.z),
      1e-7
  );
}

TEST(LiquidSubstep3D, SamplesResolvedConstituentViscosityInCoupledProjection) {
  auto initial = Initial(2);
  for (std::size_t index = 0; index < initial.particles.size(); ++index) {
    auto& particle = initial.particles[index];
    particle.world_position = {2.5 + static_cast<double>(index), 2.5, 2.5};
    particle.world_velocity.z = index == 0 ? 0.125 : -0.125;
    particle.phase_id = 4;
    particle.constituent_count = 2;
    particle.constituents[0] = {8, 0.25};
    particle.constituents[1] = {14, 0.75};
  }
  LiquidResolvedSceneView3D request;
  request.scene.field = LiquidHorizontalPlaneSurface3D{3.0};
  request.scene.obstacle = AxisAlignedLiquidTank3D{2.0, 4.0, 2.0, 2.0, 4.0, 0.0};
  request.scene.world_acceleration = {};
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{
      {8, 4, 1000.0, 40.0},
      {14, 4, 1000.0, 120.0},
  }};
  request.scene.constituent_materials = kMaterials;
  const auto encoded = EncodeLiquidResolvedScene3D(request);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto resolved = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  auto scene = resolved->View().scene;
  ASSERT_EQ(scene.constituent_materials[0].dynamic_viscosity_pa_s, 40.0);
  ASSERT_EQ(scene.constituent_materials[1].dynamic_viscosity_pa_s, 120.0);
  std::vector<double> sampled;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &sampled;
  backend.project_viscous = [](void* context,
                               const LiquidBoundaryGeometryViewInputs3D& geometry,
                               const LiquidPressureProblemView3D& problem,
                               double viscosity,
                               std::span<const double> cell_viscosity,
                               MacPressureSolverOptions3D controls,
                               MacPressureWorkspace3D& workspace) {
    auto& observed = *static_cast<std::vector<double>*>(context);
    observed.assign(cell_viscosity.begin(), cell_viscosity.end());
    return CpuLiquidSubstepBackend3D()
        .project_viscous(nullptr, geometry, problem, viscosity, cell_viscosity, controls, workspace);
  };
  auto mixture = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(mixture.ok()) << mixture.status();
  const auto advanced = mixture->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  ASSERT_EQ(sampled.size(), 4);
  for (const double viscosity : sampled) {
    EXPECT_DOUBLE_EQ(viscosity, 100.0);
  }
  constexpr std::array<LiquidConstituentMaterial3D, 2> kUniformMaterials{{
      {8, 4, 1000.0},
      {14, 4, 1000.0},
  }};
  scene.constituent_materials = kUniformMaterials;
  scene.dynamic_viscosity = 100.0;
  auto uniform = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(uniform.ok()) << uniform.status();
  ASSERT_TRUE(uniform->AdvanceFrame(0.01).ok());
  const auto actual = EncodeLiquidState3D(mixture->committed());
  const auto expected = EncodeLiquidState3D(uniform->committed());
  ASSERT_TRUE(actual.ok() && expected.ok());
  EXPECT_TRUE(std::ranges::equal(actual->values(), expected->values()));
}
TEST(LiquidSubstep3D, CouplesImmiscibleViscosityAcrossPlanarTankCells) {
  auto initial = Initial(4);
  for (std::size_t index = 0; index < initial.particles.size(); ++index) {
    auto& particle = initial.particles[index];
    const bool oil = index % 2 != 0;
    particle.world_position = {oil ? 3.5 : 2.5, 2.5, index < 2 ? 2.5 : 3.5};
    particle.world_velocity.z = oil ? -0.125 : 0.125;
    particle.mass = oil ? 820.0 : 1000.0;
    particle.phase_id = oil ? 5 : 4;
    particle.constituent_count = 1;
    particle.constituents[0] = {oil ? 25u : 8u, 1.0};
  }
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{
      {8, 4, 1000.0, 40.0},
      {25, 5, 820.0, 120.0},
  }};
  constexpr std::array<LiquidPhasePair3D, 1> kPairs{{{4, 5, 0.031}}};
  LiquidSceneInputs3D scene;
  scene.field = LiquidHorizontalPlaneSurface3D{3.0};
  scene.obstacle = AxisAlignedLiquidTank3D{2.0, 4.0, 2.0, 2.0, 4.0, 0.0};
  scene.world_acceleration = {};
  scene.constituent_materials = kMaterials;
  scene.phase_pairs = kPairs;
  struct Audit {
    std::size_t crossings = 0;
    std::size_t viscous_calls = 0;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    static_cast<Audit*>(context)->crossings += fields.internal_interfaces.size();
    EXPECT_EQ(fields.internal_interfaces.size(), 2);
    for (const auto& crossing : fields.internal_interfaces) {
      EXPECT_DOUBLE_EQ(crossing.tail_fraction, 0.5);
      EXPECT_NEAR(crossing.curvature_tail_to_head, 0, 1e-13);
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  backend.project_viscous = [](void* context,
                               const LiquidBoundaryGeometryViewInputs3D& geometry,
                               const LiquidPressureProblemView3D& problem,
                               double viscosity,
                               std::span<const double> cell_viscosity,
                               MacPressureSolverOptions3D controls,
                               MacPressureWorkspace3D& workspace) {
    ++static_cast<Audit*>(context)->viscous_calls;
    EXPECT_EQ(cell_viscosity.size(), 4);
    for (std::size_t index = 0; index < cell_viscosity.size(); ++index) {
      EXPECT_DOUBLE_EQ(cell_viscosity[index], problem.cells[index].cell_index.x == 2 ? 40.0 : 120.0);
    }
    return CpuLiquidSubstepBackend3D()
        .project_viscous(nullptr, geometry, problem, viscosity, cell_viscosity, controls, workspace);
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_EQ(audit.crossings, 2);
  EXPECT_EQ(audit.viscous_calls, 1);
  const auto measurement = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement.has_value());
  EXPECT_LT(measurement->viscous_max_momentum_residual_kg_m_per_s.value_or(1), 1e-5);
  for (const auto& particle : scheduler->committed().particles.values()) {
    EXPECT_LT(std::abs(particle.world_velocity.z), 0.125);
  }
}
TEST(LiquidSubstep3D, PressureHistoryBacktracesAcrossSignedBlocks) {
  LiquidState3D committed;
  auto blocks = LiquidRecords3D<LiquidBlockState3D>::Create(2);
  auto cells = LiquidRecords3D<LiquidCellState3D>::Create(2);
  auto faces = LiquidRecords3D<LiquidFaceState3D>::Create(2);
  ASSERT_TRUE(blocks.ok() && cells.ok() && faces.ok());
  committed.blocks = std::move(*blocks);
  committed.cells = std::move(*cells);
  committed.faces = std::move(*faces);
  ASSERT_TRUE(committed.blocks.Resize(2).ok());
  ASSERT_TRUE(committed.cells.Resize(2).ok());
  ASSERT_TRUE(committed.faces.Resize(2).ok());
  committed.blocks[0].block_index = {-1, 0, 0};
  committed.blocks[1].block_index = {0, 0, 0};
  committed.cells[0] = {{-1, 0, 0}, LiquidCellOccupancy3D::kLiquid, 4.0};
  committed.cells[1] = {{0, 0, 0}, LiquidCellOccupancy3D::kLiquid, 8.0};
  committed.faces[0] = {{FaceNormalAxis::kX, {0, 0, 0}}, 192, 1.0, LiquidFaceValidity3D::kPredicted};
  committed.faces[1] = {{FaceNormalAxis::kX, {1, 0, 0}}, 193, 3.0, LiquidFaceValidity3D::kValid};
  ASSERT_TRUE(ValidateLiquidState3D(committed).ok());
  const auto transported = liquid_step_internal::AdvectLiquidPressureHistory3D(committed, {0, 0, 0}, 0.125, 1.0);
  ASSERT_TRUE(transported.ok()) << transported.status();
  EXPECT_EQ(*transported, 7.0);
  committed.cells[0].occupancy = LiquidCellOccupancy3D::kSolid;
  const auto clipped = liquid_step_internal::AdvectLiquidPressureHistory3D(committed, {0, 0, 0}, 0.125, 1.0);
  ASSERT_TRUE(clipped.ok()) << clipped.status();
  EXPECT_EQ(*clipped, 8.0);
  const auto new_cell = liquid_step_internal::AdvectLiquidPressureHistory3D(committed, {2, 0, 0}, 0.125, 1.0);
  ASSERT_TRUE(new_cell.ok()) << new_cell.status();
  EXPECT_EQ(*new_cell, 1.0);
}

TEST(LiquidSubstep3D, MeasuresLaplacePressureOnCurvedParticlePhase) {
  constexpr double kRadius = 0.8;
  constexpr double kTension = 0.031;
  constexpr double kExpectedCurvature = 2.0 / kRadius;
  constexpr std::size_t kShellDonors = 1000;
  auto initial = Initial(kShellDonors + 1);
  initial.geometry.cell_width = 0.5;
  initial.particles[0].world_position = {};
  initial.particles[0].mass = 1000;
  initial.particles[0].phase_id = 4;
  initial.particles[0].constituent_count = 1;
  initial.particles[0].constituents[0] = {8, 1};
  constexpr double kGoldenAngle = std::numbers::pi * (3.0 - 2.2360679774997896964);
  for (std::size_t index = 0; index < kShellDonors; ++index) {
    const double latitude = 1.0 - (2.0 * static_cast<double>(index) + 1.0) / kShellDonors;
    const double azimuth = static_cast<double>(index) * kGoldenAngle;
    const double radial = std::sqrt(1.0 - latitude * latitude);
    auto& particle = initial.particles[index + 1];
    particle.world_position = {
        2 * kRadius * radial * std::cos(azimuth),
        2 * kRadius * latitude,
        2 * kRadius * radial * std::sin(azimuth),
    };
    particle.reference_volume = 0.01;
    particle.mass = 820 * particle.reference_volume;
    particle.phase_id = 5;
    particle.constituent_count = 1;
    particle.constituents[0] = {25, 1};
  }
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1000}, {25, 5, 820}}};
  constexpr std::array<LiquidPhasePair3D, 1> kPairs{{{4, 5, kTension}}};
  LiquidSceneInputs3D scene;
  scene.field = LiquidHorizontalPlaneSurface3D{2};
  scene.obstacle = AxisAlignedLiquidTank3D{-2, 2, -2, -2, 2, 0};
  scene.world_acceleration = {};
  scene.constituent_materials = kMaterials;
  scene.phase_pairs = kPairs;
  auto options = Options();
  options.capacities = {kShellDonors + 1, 64, 1024, 12288, 1};
  struct Audit {
    std::size_t crossings = 0;
    double square_error = 0.0;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    auto& audit = *static_cast<Audit*>(context);
    for (const auto& crossing : fields.internal_interfaces) {
      const auto& face = geometry.faces[crossing.face_ordinal];
      const auto anchor = geometry.mac_grid.FaceWorldAnchor(face.face_index);
      if (!anchor.ok()) {
        return absl::StatusOr<LiquidPressureProblemView3D>(anchor.status());
      }
      const std::array coordinate{anchor->x, anchor->y, anchor->z};
      const auto axis = static_cast<std::size_t>(face.face_index.axis);
      const double outward_curvature = crossing.curvature_tail_to_head * std::copysign(1.0, coordinate[axis]);
      const double error = outward_curvature / kExpectedCurvature - 1.0;
      audit.square_error += error * error;
      ++audit.crossings;
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  for (const double viscosity : {0.0, 0.001}) {
    SCOPED_TRACE(viscosity);
    scene.dynamic_viscosity = viscosity;
    audit = {};
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    const auto advanced = scheduler->AdvanceFrame(0.001);
    ASSERT_TRUE(advanced.ok()) << advanced << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
    ASSERT_GT(audit.crossings, 0);
    const double relative_curvature_rms = std::sqrt(audit.square_error / static_cast<double>(audit.crossings));
    double core_pressure = 0.0;
    std::size_t core_cells = 0;
    for (const auto& cell : scheduler->committed().cells.values()) {
      const auto index = cell.cell_index;
      if (index.x >= -1 && index.x <= 0 && index.y >= -1 && index.y <= 0 && index.z >= -1 && index.z <= 0) {
        core_pressure += cell.previous_pressure;
        ++core_cells;
      }
    }
    ASSERT_EQ(core_cells, 8);
    core_pressure /= static_cast<double>(core_cells);
    std::cout << "curved_phase_crossings=" << audit.crossings << " relative_curvature_rms=" << relative_curvature_rms
              << " core_pressure_pa=" << core_pressure << '\n';
    EXPECT_LT(relative_curvature_rms, 0.25);
    EXPECT_NEAR(core_pressure, kTension * kExpectedCurvature, 0.25 * kTension * kExpectedCurvature);
  }
}
TEST(LiquidSubstep3D, DiscoversParticlePhaseCrossingAndRetriesAtItsCapillaryLimit) {
  constexpr std::array<LiquidConstituentMaterial3D, 3> kMaterials{{
      {8, 4, 1000.0},
      {14, 4, 1020.0},
      {25, 5, 820.0},
  }};
  constexpr std::array<LiquidPhasePair3D, 1> kPairs{{{4, 5, 3.0e6}}};
  auto initial = Initial(2);
  initial.particles[0].phase_id = 4;
  initial.particles[0].constituent_count = 2;
  initial.particles[0].constituents[0] = {8, 0.25};
  initial.particles[0].constituents[1] = {14, 0.75};
  initial.particles[0].mass = 1015.0;
  initial.particles[1].world_position.x += 1.0;
  initial.particles[1].phase_id = 5;
  initial.particles[1].constituent_count = 1;
  initial.particles[1].constituents[0] = {25, 1.0};
  initial.particles[1].mass = 820.0;

  std::size_t crossings_seen = 0;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &crossings_seen;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    EXPECT_EQ(fields.cells.size(), 2);
    if (fields.cells.size() == 2) {
      EXPECT_DOUBLE_EQ(fields.cells[0].density.value_or(0.0), 1015.0);
      EXPECT_DOUBLE_EQ(fields.cells[1].density.value_or(0.0), 820.0);
    }
    EXPECT_EQ(fields.internal_interfaces.size(), 1);
    for (const auto& crossing : fields.internal_interfaces) {
      ++*static_cast<std::size_t*>(context);
      EXPECT_DOUBLE_EQ(crossing.tail_fraction, 0.5);
      EXPECT_DOUBLE_EQ(crossing.surface_tension, 3.0e6);
      EXPECT_NEAR(crossing.curvature_tail_to_head, 0.0, 1.0e-13);
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{.radius = 0.9};
  scene.world_acceleration = {};
  scene.constituent_materials = kMaterials;
  scene.phase_pairs = kPairs;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(status.ok()) << status << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 1);
  EXPECT_EQ(crossings_seen, 3);

  initial.particles[1].world_position = initial.particles[0].world_position;
  auto unresolved = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(unresolved.ok()) << unresolved.status();
  EXPECT_EQ(unresolved->AdvanceFrame(0.01).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(unresolved->committed() == initial);
}

TEST(LiquidSubstep3D, UsesCurrentVolumeMixtureDensityInPressureLinks) {
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1000.0}, {14, 4, 1020.0}}};
  struct Observation {
    std::size_t mixture_links = 0;
    bool finite_eos = false;
  } observation;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observation;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    constexpr double kExpectedDensity = 1007.5;
    auto& observed = *static_cast<Observation*>(context);
    EXPECT_EQ(fields.cells.size(), 1);
    if (!fields.cells.empty()) {
      EXPECT_DOUBLE_EQ(fields.cells.front().density.value_or(0.0), kExpectedDensity);
      if (observed.finite_eos) {
        EXPECT_DOUBLE_EQ(fields.cells.front().bulk_modulus, 4.0 * kExpectedDensity);
      }
    }
    const auto links = CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
    if (links.ok()) {
      for (const auto& face : links->faces) {
        if (face.open_area > 0.0) {
          EXPECT_DOUBLE_EQ(face.inverse_density, 1.0 / kExpectedDensity);
          ++observed.mixture_links;
        }
      }
    }
    return links;
  };
  auto initial = Initial(2);
  initial.particles[0].reference_volume = 0.5;
  initial.particles[0].volume_ratio = 2.0;
  initial.particles[0].mass = 1015.0;
  initial.particles[0].phase_id = 4;
  initial.particles[0].constituent_count = 2;
  initial.particles[0].constituents[0] = {8, 0.25};
  initial.particles[0].constituents[1] = {14, 0.75};
  initial.particles[1].reference_volume = 2.0;
  initial.particles[1].volume_ratio = 0.5;
  initial.particles[1].mass = 1000.0;
  initial.particles[1].phase_id = 4;
  initial.particles[1].constituent_count = 1;
  initial.particles[1].constituents[0] = {8, 1.0};
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 0.9};
  scene.world_acceleration = {};
  scene.constituent_materials = kMaterials;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(observation.mixture_links, 6);

  observation = {.finite_eos = true};
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0, .law = LiquidEosLaw::kTait};
  auto compliant = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(compliant.ok()) << compliant.status();
  const auto compliant_status = compliant->AdvanceFrame(0.01);
  ASSERT_TRUE(compliant_status.ok()) << compliant_status;
  EXPECT_EQ(observation.mixture_links, 6);
}

TEST(LiquidSubstep3D, TaitFreezesComplianceAtAdvectedPressureAndRestarts) {
  struct Observation {
    std::size_t cells = 0;
    bool saw_nonzero_pressure = false;
  } observation;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observation;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    auto& observed = *static_cast<Observation*>(context);
    for (const auto& cell : fields.cells) {
      ++observed.cells;
      observed.saw_nonzero_pressure |= std::abs(cell.previous_pressure) > 1e-8;
      EXPECT_NEAR(cell.bulk_modulus, 4000.0 + 7.0 * cell.previous_pressure, 1e-10);
      EXPECT_TRUE(cell.density.has_value());
      const double expected_density = 1000.0 * std::pow(1.0 + 7.0 * cell.previous_pressure / 4000.0, 1.0 / 7.0);
      EXPECT_NEAR(cell.density.value_or(0.0), expected_density, 1e-10);
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.ambient_pressure = 10.0;
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0, .law = LiquidEosLaw::kTait};
  auto scheduler = LiquidSubstepScheduler3D::Create(Initial(), scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto first = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(first.ok()) << first;
  const auto checkpoint = EncodeLiquidState3D(scheduler->committed());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const auto restored = DecodeLiquidState3D(checkpoint->values());
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto resumed = LiquidSubstepScheduler3D::Create(*restored, scene, Options());
  ASSERT_TRUE(resumed.ok()) << resumed.status();
  observation = {};
  const auto second = scheduler->AdvanceFrame(0.02);
  ASSERT_TRUE(second.ok()) << second;
  const auto resumed_step = resumed->AdvanceFrame(0.02);
  ASSERT_TRUE(resumed_step.ok()) << resumed_step;
  ExpectIdenticalSerializedState(scheduler->committed(), resumed->committed());
  EXPECT_GT(observation.cells, 0U);
  EXPECT_TRUE(observation.saw_nonzero_pressure);
}

TEST(LiquidSubstep3D, ExpandedEosDensityTightensCapillaryStep) {
  const auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 1.0};
  scene.ambient_pressure = -1000.0 * std::log(4.0);
  scene.surface_tension = 1000.0 / (4.0 * std::numbers::pi * 0.1 * 0.1);
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.08).ok());
  const auto attempted = scheduler->Attempt(0.08);
  ASSERT_TRUE(attempted.ok()) << attempted.status();
  EXPECT_EQ(attempted->disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kInterface);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, IncompressibleDriftBalancesSealedBandAndChecksPhysicalSources) {
  struct Observation {
    std::vector<MacPressureFace3D> sealed_faces;
    std::vector<MacPressureCell3D> corrected_cells;
  } observation;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observation;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    auto result = CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
    if (result.ok()) {
      auto& faces = static_cast<Observation*>(context)->sealed_faces;
      faces.assign(result->faces.begin(), result->faces.end());
      for (auto& face : faces) {
        if (std::holds_alternative<DirichletMacPressureLink3D>(face.pressure_link)) {
          face.open_area = 0.0;
          face.pressure_link = SealedMacPressureLink3D{};
        }
      }
      result->faces = faces;
    }
    return result;
  };
  backend.project = [](void* context,
                       SparseMacGridView3D grid,
                       double duration,
                       const LiquidPressureProblemView3D& problem,
                       MacPressureSolverOptions3D options,
                       MacPressureWorkspace3D& workspace) {
    auto& corrected = static_cast<Observation*>(context)->corrected_cells;
    corrected.assign(problem.cells.begin(), problem.cells.end());
    return CpuLiquidSubstepBackend3D().project(nullptr, grid, duration, problem, options, workspace);
  };
  auto initial = Initial(2);
  initial.particles[1].world_position.x = 3.5;
  initial.particles[0].volume_ratio = 2.0;
  LiquidSceneInputs3D scene;
  scene.field = LiquidHorizontalPlaneSurface3D{3.0};
  scene.obstacle = AxisAlignedLiquidTank3D{2.0, 4.0, 2.0, 2.0, 4.0, 0.0};
  scene.world_acceleration = {};
  scene.drift = LiquidSceneDrift3D{.correction_time = 1.0, .maximum_rate = 0.3};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  ASSERT_EQ(observation.corrected_cells.size(), 4U);
  EXPECT_NEAR(observation.corrected_cells[0].target_divergence, -0.3, 1e-14);
  EXPECT_NEAR(observation.corrected_cells[1].target_divergence, 0.3, 1e-14);
  EXPECT_EQ(observation.corrected_cells[2].target_divergence, 0.0);
  EXPECT_EQ(observation.corrected_cells[3].target_divergence, 0.0);

  observation.corrected_cells.clear();
  scene.target_divergence = 0.01;
  auto incompatible = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(incompatible.ok()) << incompatible.status();
  EXPECT_EQ(incompatible->AdvanceFrame(0.01).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(observation.corrected_cells.empty());
  ExpectIdenticalSerializedState(initial, incompatible->committed());
}

TEST(LiquidSubstep3D, FiniteEosDriftUsesOnlySensedCells) {
  struct Observation {
    std::size_t sensed = 0;
    std::size_t unsensed = 0;
  } observation;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observation;
  backend.links = [](void* context,
                     const FrozenLiquidBoundaryGeometryView3D& geometry,
                     const LiquidPressureFields3D& fields,
                     LiquidBoundaryWorkspace3D& workspace) {
    auto& observed = *static_cast<Observation*>(context);
    for (std::size_t index = 0; index < geometry.cells.size(); ++index) {
      const auto cell = geometry.cells[index].cell_index;
      if (cell.x == 2 && cell.y == 2 && cell.z == 2) {
        ++observed.sensed;
        EXPECT_NEAR(fields.cells[index].target_divergence, 0.1 - 0.3, 1e-15);
      } else {
        ++observed.unsensed;
        EXPECT_EQ(fields.cells[index].target_divergence, 0.1);
      }
    }
    return CpuLiquidSubstepBackend3D().links(nullptr, geometry, fields, workspace);
  };
  auto initial = Initial();
  initial.particles[0].volume_ratio = 2.0;
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.target_divergence = 0.1;
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0};
  scene.drift = LiquidSceneDrift3D{.correction_time = 0.2, .maximum_rate = 0.3};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  EXPECT_EQ(observation.sensed, 1U);
  EXPECT_GT(observation.unsensed, 0U);
  initial.particles[0].side_tag = 1;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, scene, Options()).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidSubstep3D, TaitRejectsCandidateOutsideDomainBeforePublication) {
  std::vector<double> candidate_pressures;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &candidate_pressures;
  backend.project = [](void* context,
                       SparseMacGridView3D grid,
                       double step_duration,
                       const LiquidPressureProblemView3D& problem,
                       MacPressureSolverOptions3D options,
                       MacPressureWorkspace3D& workspace) {
    auto result = CpuLiquidSubstepBackend3D().project(nullptr, grid, step_duration, problem, options, workspace);
    if (result.ok()) {
      auto& pressures = *static_cast<std::vector<double>*>(context);
      pressures.assign(result->cell_pressures.size(), -1000.0);
      result->cell_pressures = pressures;
    }
    return result;
  };
  const auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0, .law = LiquidEosLaw::kTait};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0.01);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(candidate_pressures.empty());
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, UnsupportedViscousGeometryPreservesCommittedState) {
  const auto initial = Initial();
  auto scene = Scene();
  scene.dynamic_viscosity = 1.0;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.project_viscous = nullptr;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend).status().code(),
      absl::StatusCode::kUnimplemented
  );

  const auto check_rejection = [&](absl::StatusCode expected) {
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    const auto attempt = scheduler->Attempt(0.01);
    ASSERT_TRUE(attempt.ok()) << attempt.status();
    EXPECT_EQ(attempt->disposition, LiquidAttemptDisposition3D::kFatal);
    EXPECT_EQ(scheduler->diagnostics().failure_code, expected);
    ExpectIdenticalSerializedState(initial, scheduler->committed());
  };
  check_rejection(absl::StatusCode::kFailedPrecondition);
  scene.obstacle = TranslatingHalfspace3D{{0.0, 1.0, 0.0}, 2.25, {}};
  check_rejection(absl::StatusCode::kUnimplemented);
}

TEST(LiquidSubstep3D, OceanInletRetiresDryParticleAndDiscardsRejectedBirths) {
  auto seed = Initial(2);
  seed.particles[0].world_position = {2.25, 1.25, 2.25};
  seed.particles[1].world_position = {2.25, 3.25, 2.25};
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{1.1};
  scene.world_acceleration = {};
  auto options = Options();
  options.ocean_boundary = OceanBoundarySettings3D{
      .swell =
          {
              .still_water_height_m = 3,
              .water_depth_m = 3,
              .wave_height_m = 0.5,
              .wave_period_s = 4,
              .phase_at_world_origin_rad = std::numbers::pi,
          },
      .inlet_wall_x_m = 2,
      .inlet_interior_x_m = 2.5,
      .outlet_interior_x_m = 5,
      .outlet_wall_x_m = 8,
      .inlet_particle_spacing_m = 0.5,
      .inlet_low_z_m = 2,
      .inlet_high_z_m = 2.5,
  };
  auto rejected = LiquidSubstepScheduler3D::Create(seed, scene, options);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  rejected->SetStageFaultHookForTesting({
      nullptr,
      [](void*, LiquidPhysicalStage3D stage, double) {
        return stage == LiquidPhysicalStage3D::kActivate ? absl::InternalError("reject after inlet edit")
                                                         : absl::OkStatus();
      },
  });
  ASSERT_TRUE(rejected->BeginFrame(0.01).ok());
  const auto attempt = rejected->Attempt(0.01);
  ASSERT_TRUE(attempt.ok()) << attempt.status();
  EXPECT_EQ(attempt->disposition, LiquidAttemptDisposition3D::kFatal);
  ExpectIdenticalSerializedState(seed, rejected->committed());

  auto accepted = LiquidSubstepScheduler3D::Create(seed, scene, options);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  const auto advanced = accepted->AdvanceFrame(0.01);
  ASSERT_TRUE(advanced.ok()) << advanced;
  const auto& state = accepted->committed();
  EXPECT_FALSE(std::ranges::any_of(state.particles.values(), [](const auto& particle) {
    return particle.id == 2;
  }));
  EXPECT_TRUE(std::ranges::any_of(state.particles.values(), [](const auto& particle) {
    return particle.id == 1;
  }));
  EXPECT_GT(state.next_particle_id, seed.next_particle_id);
}

TEST(LiquidSubstep3D, WettedPistonUsesClippedAndAttachedAreaInProjection) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wet = wetted_piston_pressure_test;
  for (const double height : {0.75, 1.125}) {
    for (const double speed : {wet::kPistonSpeed, -wet::kPistonSpeed, 0.0}) {
      SCOPED_TRACE(::testing::Message() << height << " " << speed);
      wet::ProjectionAudit audit;
      auto scheduler = LiquidSubstepScheduler3D::Create(
          {},
          wet::PlaneScene(height, speed),
          Options(),
          wet::Observe(audit, CpuLiquidSubstepBackend3D())
      );
      ASSERT_TRUE(scheduler.ok()) << scheduler.status();
      const auto status = scheduler->AdvanceFrame(wet::kDuration);
      ASSERT_TRUE(status.ok()) << status;
      wet::CheckPlaneProjection(*scheduler, audit, height, speed);
    }
  }
}

TEST(LiquidSubstep3D, UnresolvedAnalyticSphereWithNoAirLinksCanAdvance) {
  const LiquidState3D initial;
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{0.25, 0.25, 0.25}, std::numeric_limits<double>::denorm_min()};
  scene.surface_tension = 1;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(0x1p-7);
  ASSERT_TRUE(advanced.ok()) << advanced;
  EXPECT_TRUE(scheduler->committed().cells.values().empty());
  ASSERT_TRUE(scheduler->diagnostics().accepted_measurement.has_value());
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement->step_start_capillary_bound_seconds.has_value());
}

TEST(LiquidSubstep3D, MovingPistonGatherGroupsByPressureSupportAndPreservesCarrier) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wet = wetted_piston_pressure_test;
  const auto state = wet::PressureSupportGatherState();
  ASSERT_TRUE(state.ok()) << state.status();
  const auto& initial = *state;
  struct Audit {
    std::vector<bool> incident_faces;
    std::vector<std::size_t> run_sizes;
    std::vector<std::vector<LiquidFaceValidity3D>> new_validities;
    std::vector<std::vector<double>> new_velocities;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.freeze =
      [](void* context, const LiquidBoundaryGeometryViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
        const auto frozen = CpuLiquidSubstepBackend3D().freeze(nullptr, inputs, workspace);
        if (frozen.ok()) {
          auto& incident = static_cast<Audit*>(context)->incident_faces;
          incident.assign(frozen->mac_grid.face_count(), false);
          for (const auto& face : frozen->faces) {
            incident[face.face_slot] = true;
          }
        }
        return frozen;
      };
  backend.gather_run =
      [](void* context, const LiquidBoundaryGatherViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace) {
        auto& observed = *static_cast<Audit*>(context);
        observed.run_sizes.push_back(input.particles.size());
        observed.new_validities.emplace_back(input.new_history.validity.begin(), input.new_history.validity.end());
        observed.new_velocities.emplace_back(
            input.new_history.face_velocities.begin(),
            input.new_history.face_velocities.end()
        );
        return GatherLiquidBoundaryParticles3D(input, workspace);
      };
  auto options = Options();
  options.capacities.particles = initial.particles.size();
  const auto scene = wet::PressureSupportGatherScene();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0x1p-7);
  ASSERT_TRUE(status.ok()) << status << static_cast<int>(scheduler->diagnostics().failed_stage);
  ASSERT_EQ(audit.run_sizes, (std::vector<std::size_t>{2, 2}));
  const auto& supported_validities = audit.new_validities[0];
  const auto& unsupported_validities = audit.new_validities[1];
  ASSERT_EQ(supported_validities.size(), scheduler->committed().faces.size());
  ASSERT_EQ(unsupported_validities.size(), scheduler->committed().faces.size());
  ASSERT_EQ(audit.new_velocities[0].size(), scheduler->committed().faces.size());
  ASSERT_EQ(audit.new_velocities[1].size(), scheduler->committed().faces.size());
  std::size_t masked_donors = 0;
  for (const auto& face : scheduler->committed().faces.values()) {
    const auto slot = face.face_slot;
    const bool usable =
        face.validity == LiquidFaceValidity3D::kValid || face.validity == LiquidFaceValidity3D::kPredicted;
    const bool masked = !audit.incident_faces[slot] && face.structurally_eligible && usable;
    EXPECT_EQ(supported_validities[slot], masked ? LiquidFaceValidity3D::kMissing : face.validity);
    EXPECT_EQ(unsupported_validities[slot], face.validity);
    EXPECT_DOUBLE_EQ(audit.new_velocities[0][slot], face.carrier_velocity);
    EXPECT_DOUBLE_EQ(audit.new_velocities[1][slot], face.carrier_velocity);
    masked_donors += masked;
  }
  EXPECT_GT(masked_donors, 0);
  auto scalar = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scalar.ok()) << scalar.status();
  const auto scalar_status = scalar->AdvanceFrame(0x1p-7);
  ASSERT_TRUE(scalar_status.ok()) << scalar_status;
  ExpectIdenticalSerializedState(scalar->committed(), scheduler->committed());
}

TEST(LiquidSubstep3D, WettedPistonRejectsWetEmptyPressureWithoutRetry) {
  namespace wet = wetted_piston_pressure_test;
  const LiquidState3D initial;
  auto scene = wet::PlaneScene(0.25, wet::kPistonSpeed);
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  EXPECT_EQ(scheduler->AdvanceFrame(wet::kDuration).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 0);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
  wet::ExpectCheckpointEqual(initial, scheduler->committed());
  scene.field = LiquidParticleSphereUnion3D{0.25};
  auto dry = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(dry.ok()) << dry.status();
  ASSERT_TRUE(dry->AdvanceFrame(wet::kDuration).ok());
  ASSERT_TRUE(dry->diagnostics().accepted_measurement.has_value());
  EXPECT_DOUBLE_EQ(dry->diagnostics().accepted_measurement->step_start_inward_piston_flux_m3_per_s, 0);
  EXPECT_EQ(dry->committed().clock.epoch, 1);
}

TEST(LiquidSubstep3D, WettedPistonRejectsArithmeticOverflowAfterFreeze) {
  namespace wet = wetted_piston_pressure_test;
  const LiquidState3D initial;
  // The legacy 1 m² face flux is finite. The 1.125 m² wet wall
  // overflows signed flux even though the piston advances less than half a cell.
  wet::ProjectionAudit audit;
  auto scheduler = LiquidSubstepScheduler3D::Create(
      initial,
      wet::PlaneScene(1.125, wet::kOverflowPistonSpeed),
      Options(),
      wet::Observe(audit, CpuLiquidSubstepBackend3D())
  );
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(wet::kOverflowStepDuration).ok());
  const auto result = scheduler->Attempt(wet::kOverflowStepDuration);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kInterface);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kOutOfRange);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 0);
  EXPECT_EQ(audit.wetted_stage_calls, 1);
  EXPECT_EQ(audit.projection_count, 0);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
  wet::ExpectCheckpointEqual(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, WettedPistonReplaysAfterGrowth) {
  namespace wet = wetted_piston_pressure_test;
  const auto scene = wet::PlaneScene(1.125, wet::kPistonSpeed);
  auto small = Options();
  small.capacities.faces = 64;
  small.capacity_replay_reservations = Options().capacities;
  auto replay = LiquidSubstepScheduler3D::Create({}, scene, small);
  ASSERT_TRUE(replay.ok()) << replay.status();
  const auto advanced = replay->AdvanceFrame(wet::kDuration);
  ASSERT_TRUE(advanced.ok()) << advanced;
  EXPECT_EQ(replay->diagnostics().capacity_replays, 1);
  auto direct = LiquidSubstepScheduler3D::Create({}, scene, Options());
  ASSERT_TRUE(direct.ok()) << direct.status();
  ASSERT_TRUE(direct->AdvanceFrame(wet::kDuration).ok());
  wet::ExpectCheckpointEqual(direct->committed(), replay->committed());
}

TEST(LiquidSubstep3D, WettedPistonRequiresSelectedOperationOnlyForMovingTank) {
  namespace wet = wetted_piston_pressure_test;
  auto operations = CpuLiquidSubstepBackend3D();
  operations.wetted_piston = nullptr;
  auto scene = wet::PlaneScene(0.75, wet::kPistonSpeed);
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create({}, scene, Options(), operations).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  scene.obstacle = wet::PlaneScene(0.75, 0).obstacle;
  auto stationary = LiquidSubstepScheduler3D::Create({}, scene, Options(), operations);
  ASSERT_TRUE(stationary.ok()) << stationary.status();
  EXPECT_TRUE(stationary->AdvanceFrame(wet::kDuration).ok());
  auto absent = LiquidSubstepScheduler3D::Create(Initial(), Scene(), Options(), operations);
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_TRUE(absent->AdvanceFrame(wet::kDuration).ok());
}

absl::Status
UpdateCpuParticles(const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
  return CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
}

TEST(LiquidSubstep3D, RegularCellTransportPairsAdvectionWithVolume) {
  test_internal::CheckRegularCellTransport(UpdateCpuParticles);
}

TEST(LiquidSubstep3D, RegularCellTransportUsesSweptContact) {
  test_internal::CheckRegularCellTransportContact(UpdateCpuParticles);
}

TEST(LiquidSubstep3D, RejectsMalformedRegularCellTransportBeforeMutation) {
  test_internal::CheckMalformedRegularCellTransport(UpdateCpuParticles);
}

TEST(LiquidSubstep3D, RejectsRegularCellTransportAliasingTrialParticles) {
  test_internal::CheckAliasedRegularCellTransport(UpdateCpuParticles);
}
TEST(LiquidSubstep3D, RegularCellTransportPreservesParticleErrorOrder) {
  test_internal::CheckRegularCellTransportErrorOrder(UpdateCpuParticles);
}

TEST(LiquidSubstep3D, RegularCellTransportRejectsInvalidParticleInputsBeforeRetry) {
  test_internal::CheckRegularCellTransportRejectsInvalidParticleInputs(UpdateCpuParticles);
}

struct RegularTransportSchedulerAudit {
  std::vector<FrozenLiquidCell3D> cells;
  std::vector<FrozenLiquidFace3D> faces;
  std::vector<LiquidRegularCellTransport3D> transport;
  std::size_t update_count = 0;
};

LiquidSubstepBackend3D ObserveRegularTransport(RegularTransportSchedulerAudit& audit) {
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.freeze =
      [](void* context, const LiquidBoundaryGeometryViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
        auto& observed = *static_cast<RegularTransportSchedulerAudit*>(context);
        const auto frozen = CpuLiquidSubstepBackend3D().freeze(nullptr, inputs, workspace);
        if (frozen.ok()) {
          observed.cells.assign(frozen->cells.begin(), frozen->cells.end());
          observed.faces.assign(frozen->faces.begin(), frozen->faces.end());
        }
        return frozen;
      };
  backend.update_particles =
      [](void* context, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
        auto& observed = *static_cast<RegularTransportSchedulerAudit*>(context);
        ++observed.update_count;
        observed.transport.assign(inputs.regular_cell_transport.begin(), inputs.regular_cell_transport.end());
        return CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
      };
  return backend;
}

constexpr double kRegularTransportDuration = 0x1p-7;
constexpr Vec3 kRegularTransportStart{-1.375, 0.375, 2.625};

LiquidState3D RegularTransportInitial() {
  auto initial = Initial();
  initial.particles[0].world_position = kRegularTransportStart;
  initial.particles[0].world_velocity = {0.5, -0.25, 0.125};
  return initial;
}

LiquidSceneInputs3D RegularTransportScene() {
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{0.7};
  scene.obstacle = AxisAlignedLiquidTank3D{
      .start_low_x = -4,
      .high_x = 12,
      .floor_y = 0,
      .low_z = 0,
      .high_z = 5,
      .piston_velocity_x = 0,
  };
  return scene;
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportUsesProjectedFacesAndStationaryFloor) {
  const auto initial = RegularTransportInitial();
  const auto scene = RegularTransportScene();
  RegularTransportSchedulerAudit audit;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), ObserveRegularTransport(audit));
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto advanced = scheduler->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(advanced.ok()) << advanced;
  ASSERT_EQ(audit.update_count, 1);
  ASSERT_EQ(audit.cells.size(), 1);
  const auto cell = audit.cells.front().cell_index;
  EXPECT_EQ(cell.x, -2);
  EXPECT_EQ(cell.y, 0);
  EXPECT_EQ(cell.z, 2);
  ASSERT_EQ(audit.faces.size(), 6);
  std::array<double, 3> low{};
  std::array<double, 3> high{};
  std::size_t sealed_faces = 0;
  for (const auto& face : audit.faces) {
    const auto axis = static_cast<std::size_t>(face.face_index.axis);
    const auto lattice = face.face_index.lattice_index;
    const std::array face_coordinates{lattice.x, lattice.y, lattice.z};
    const std::array cell_coordinates{cell.x, cell.y, cell.z};
    if (face.open_area == 0) {
      ++sealed_faces;
      EXPECT_EQ(axis, 1);
      EXPECT_EQ(lattice.y, 0);
      EXPECT_DOUBLE_EQ(face.blocked_area, 1);
      EXPECT_DOUBLE_EQ(face.wall_normal_velocity, 0);
      EXPECT_NE(scheduler->committed().faces[face.face_slot].carrier_velocity, 0);
      continue;
    }
    EXPECT_DOUBLE_EQ(face.open_area, 1);
    const auto& projected = scheduler->committed().faces[face.face_slot];
    EXPECT_TRUE(projected.structurally_eligible);
    EXPECT_EQ(projected.side_tag, initial.particles[0].side_tag);
    EXPECT_EQ(projected.validity, LiquidFaceValidity3D::kValid);
    if (face_coordinates[axis] == cell_coordinates[axis]) {
      low[axis] = projected.carrier_velocity;
    } else {
      EXPECT_EQ(face_coordinates[axis], cell_coordinates[axis] + 1);
      high[axis] = projected.carrier_velocity;
    }
  }
  ASSERT_EQ(sealed_faces, 1);
  const std::array fraction{0.625, 0.375, 0.625};
  EXPECT_DOUBLE_EQ(low[1], 0);
  EXPECT_NE(high[1], low[1]);
  const auto& particle = scheduler->committed().particles[0];
  const std::array start{kRegularTransportStart.x, kRegularTransportStart.y, kRegularTransportStart.z};
  const std::array end{particle.world_position.x, particle.world_position.y, particle.world_position.z};
  double divergence = 0;
  for (std::size_t axis = 0; axis < fraction.size(); ++axis) {
    const double derivative = high[axis] - low[axis];
    divergence += derivative;
    const double velocity = low[axis] + fraction[axis] * derivative;
    EXPECT_DOUBLE_EQ(end[axis], start[axis] + kRegularTransportDuration * velocity);
  }
  EXPECT_DOUBLE_EQ(particle.volume_ratio, std::exp(kRegularTransportDuration * divergence));
  ASSERT_EQ(audit.transport.size(), 1);
  const auto& sample = audit.transport.front();
  EXPECT_TRUE(sample.active);
  EXPECT_DOUBLE_EQ(sample.position_in_cell_widths.x, fraction[0]);
  EXPECT_DOUBLE_EQ(sample.position_in_cell_widths.y, fraction[1]);
  EXPECT_DOUBLE_EQ(sample.position_in_cell_widths.z, fraction[2]);
  EXPECT_DOUBLE_EQ(sample.low_face_normal_velocity.x, low[0]);
  EXPECT_DOUBLE_EQ(sample.low_face_normal_velocity.y, low[1]);
  EXPECT_DOUBLE_EQ(sample.low_face_normal_velocity.z, low[2]);
  EXPECT_DOUBLE_EQ(sample.high_face_normal_velocity.x, high[0]);
  EXPECT_DOUBLE_EQ(sample.high_face_normal_velocity.y, high[1]);
  EXPECT_DOUBLE_EQ(sample.high_face_normal_velocity.z, high[2]);
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportDeclinesFractionalFloorAndUnsupportedCell) {
  auto initial = RegularTransportInitial();
  auto scene = RegularTransportScene();
  auto& tank = std::get<AxisAlignedLiquidTank3D>(scene.obstacle);
  tank.floor_y = 0.125;
  RegularTransportSchedulerAudit partial_audit;
  auto partial = LiquidSubstepScheduler3D::Create(initial, scene, Options(), ObserveRegularTransport(partial_audit));
  ASSERT_TRUE(partial.ok()) << partial.status();
  const auto partial_status = partial->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(partial_status.ok()) << partial_status;
  EXPECT_EQ(partial_audit.update_count, 1);
  ASSERT_EQ(partial_audit.cells.size(), 1);
  EXPECT_TRUE(std::ranges::any_of(partial_audit.faces, [](const auto& face) {
    return face.open_area > 0 && face.open_area < 1;
  }));
  ASSERT_EQ(partial_audit.transport.size(), 1);
  EXPECT_FALSE(partial_audit.transport.front().active);

  initial = Initial(2);
  initial.particles[0] = RegularTransportInitial().particles[0];
  initial.particles[1].world_position = {8.0625, 2.0625, 2.0625};
  scene = RegularTransportScene();
  RegularTransportSchedulerAudit mixed_audit;
  auto mixed = LiquidSubstepScheduler3D::Create(initial, scene, Options(), ObserveRegularTransport(mixed_audit));
  ASSERT_TRUE(mixed.ok()) << mixed.status();
  const auto mixed_status = mixed->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(mixed_status.ok()) << mixed_status;
  EXPECT_EQ(mixed_audit.update_count, 1);
  ASSERT_EQ(mixed_audit.cells.size(), 1);
  EXPECT_EQ(mixed_audit.cells.front().cell_index.x, -2);
  ASSERT_EQ(mixed_audit.transport.size(), 2);
  for (std::size_t index = 0; index < mixed_audit.transport.size(); ++index) {
    EXPECT_EQ(mixed_audit.transport[index].active, mixed->committed().particles[index].id == 1);
  }
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportKeepsOtherSceneClassesUnstaged) {
  const auto initial = RegularTransportInitial();
  const auto candidate_scene = RegularTransportScene();
  const auto candidate = LiquidSubstepScheduler3D::Create(initial, candidate_scene, Options());
  ASSERT_TRUE(candidate.ok()) << candidate.status();
  const auto check = [&](const LiquidSceneInputs3D& scene) {
    RegularTransportSchedulerAudit audit;
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), ObserveRegularTransport(audit));
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    const auto advanced = scheduler->AdvanceFrame(kRegularTransportDuration);
    ASSERT_TRUE(advanced.ok()) << advanced;
    EXPECT_EQ(audit.update_count, 1);
    EXPECT_TRUE(audit.transport.empty());
    EXPECT_FALSE(audit.cells.empty());
    return;
  };
  auto analytic = candidate_scene;
  analytic.field = LiquidAnalyticSphere3D{kRegularTransportStart, 0.7};
  const auto noncandidate = LiquidSubstepScheduler3D::Create(initial, analytic, Options());
  ASSERT_TRUE(noncandidate.ok()) << noncandidate.status();
  EXPECT_EQ(
      candidate->reserved_bytes(),
      noncandidate->reserved_bytes() + Options().capacities.particles * sizeof(LiquidRegularCellTransport3D)
  );
  check(analytic);
  auto moving = candidate_scene;
  std::get<AxisAlignedLiquidTank3D>(moving.obstacle).piston_velocity_x = 0.125;
  check(moving);
  auto compliant = candidate_scene;
  compliant.bulk_modulus = 1e6;
  check(compliant);
  auto divergence_source = candidate_scene;
  divergence_source.target_divergence = 0.125;
  check(divergence_source);
  auto capillary = candidate_scene;
  capillary.surface_tension = 1;
  check(capillary);
  auto open_domain = candidate_scene;
  open_domain.obstacle = std::monostate{};
  check(open_domain);
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportRollsBackAndReplaysAfterGrowth) {
  const auto initial = RegularTransportInitial();
  const auto scene = RegularTransportScene();
  auto options = Options();
  options.capacities.faces = 64;
  options.capacity_replay_reservations = Options().capacities;
  RegularTransportSchedulerAudit audit;
  auto replay = LiquidSubstepScheduler3D::Create(initial, scene, options, ObserveRegularTransport(audit));
  ASSERT_TRUE(replay.ok()) << replay.status();
  bool reject_once = true;
  replay->SetStageFaultHookForTesting({
      .context = &reject_once,
      .before_stage =
          [](void* context, LiquidPhysicalStage3D stage, double) {
            if (stage == LiquidPhysicalStage3D::kAccept && *static_cast<bool*>(context)) {
              *static_cast<bool*>(context) = false;
              return absl::AbortedError("");
            }
            return absl::OkStatus();
          },
  });
  // Grow before the first complete update, then reject its selected trial.
  const auto advanced = replay->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(advanced.ok()) << advanced;
  EXPECT_EQ(replay->diagnostics().capacity_replays, 1);
  EXPECT_EQ(replay->diagnostics().dt_retries, 1);
  EXPECT_EQ(audit.update_count, 3);
  ASSERT_EQ(audit.transport.size(), 1);
  EXPECT_TRUE(audit.transport.front().active);
  auto direct_options = Options();
  direct_options.maximum_step_duration = kRegularTransportDuration / 2;
  auto direct = LiquidSubstepScheduler3D::Create(initial, scene, direct_options);
  ASSERT_TRUE(direct.ok()) << direct.status();
  const auto direct_status = direct->AdvanceFrame(kRegularTransportDuration);
  ASSERT_TRUE(direct_status.ok()) << direct_status;
  EXPECT_EQ(replay->reserved_bytes(), direct->reserved_bytes());
  ExpectIdenticalSerializedState(direct->committed(), replay->committed());
}

TEST(LiquidSubstep3D, SchedulerRegularCellTransportPreservesDamMaterialThroughFrameSix) {
  constexpr double cell_width = 0.0625;
  constexpr double reference_volume = cell_width * cell_width * cell_width;
  constexpr std::size_t particle_count = 256;
  auto initial = Initial(particle_count);
  initial.geometry.cell_width = cell_width;
  std::size_t ordinal = 0;
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        auto& particle = initial.particles[ordinal++];
        particle.world_position = {(x + 0.5) * cell_width, (y + 0.5) * cell_width, (z + 0.5) * cell_width};
        particle.reference_volume = reference_volume;
        particle.mass = 1000 * reference_volume;
        particle.pic_initialized = false;
      }
    }
  }
  // Match the public dam fixture and its frozen default blend through the first drifting frame.
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{0.9 * cell_width};
  scene.obstacle = AxisAlignedLiquidTank3D{
      .start_low_x = 0,
      .high_x = 2,
      .floor_y = 0,
      .low_z = 0,
      .high_z = 0.25,
      .piston_velocity_x = 0,
  };
  LiquidStepOptions3D options;
  options.particle_energy_blend = 0.6;
  options.maximum_step_duration = 0.01;
  options.capacities = {1024, 128, 4096, 24576, 256};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  for (int frame = 1; frame <= 6; ++frame) {
    SCOPED_TRACE(frame);
    const auto advanced = scheduler->AdvanceFrame(frame / 24.0);
    ASSERT_TRUE(advanced.ok()) << advanced;
    ASSERT_EQ(scheduler->committed().particles.size(), particle_count);
    std::array<bool, particle_count> seen{};
    double volume = 0;
    for (const auto& particle : scheduler->committed().particles.values()) {
      ASSERT_GE(particle.id, 1);
      ASSERT_LE(particle.id, particle_count);
      EXPECT_FALSE(seen[particle.id - 1]);
      seen[particle.id - 1] = true;
      EXPECT_DOUBLE_EQ(particle.reference_volume, reference_volume);
      EXPECT_DOUBLE_EQ(particle.mass, 1000 * reference_volume);
      volume += particle.reference_volume * particle.volume_ratio;
    }
    EXPECT_LE(std::abs(volume / (particle_count * reference_volume) - 1), 0.01);
  }
}

TEST(LiquidSubstep3D, DeduplicatesSignedFieldKeysBeforeSampling) {
  auto operations = CpuLiquidSubstepBackend3D();
  std::size_t evaluations = 0;
  operations.context = &evaluations;
  operations.sample_field_batch = [](void* context,
                                     const LiquidSceneField3D& field,
                                     std::span<const LiquidParticleState3D> particles,
                                     std::span<const Vec3> positions,
                                     std::span<double> distances) {
    *static_cast<std::size_t*>(context) += positions.size();
    return CpuLiquidSubstepBackend3D().sample_field_batch(nullptr, field, particles, positions, distances);
  };
  ASSERT_NO_FATAL_FAILURE(tank_staging_test::ExpectMixedSignedFieldSupport(operations, 0));
  EXPECT_EQ(evaluations, 262);
}

TEST(LiquidSubstep3D, CheckedFieldBatchesPreserveOutputs) {
  tank_staging_test::ExpectFieldBatchContract(CpuLiquidSubstepBackend3D());
}

TEST(LiquidSubstep3D, LateMalformedFieldBatchPreservesCommittedBytes) {
  const auto initial = Initial();
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 2.1};
  auto operations = CpuLiquidSubstepBackend3D();
  std::size_t batches = 0;
  operations.context = &batches;
  operations.sample_field_batch = [](void* context,
                                     const LiquidSceneField3D& field,
                                     std::span<const LiquidParticleState3D> particles,
                                     std::span<const Vec3> positions,
                                     std::span<double> distances) {
    if (++*static_cast<std::size_t*>(context) == 2) {
      distances = distances.first(distances.size() - 1);
    }
    return CpuLiquidSubstepBackend3D().sample_field_batch(nullptr, field, particles, positions, distances);
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), operations);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(batches, 2);
  ExpectIdenticalSerializedState(initial, scheduler->committed());

  operations.sample_field_batch = nullptr;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, scene, Options(), operations).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}

TEST(LiquidSubstep3D, HorizontalPlaneActivatesSignedPressureAndFaceOwnersWithoutParticles) {
  tank_staging_test::ExpectPlaneSupport(CpuLiquidSubstepBackend3D(), Options());
}

TEST(LiquidSubstep3D, TranslatedHorizontalPlaneRetainsWetPressureCell) {
  tank_staging_test::ExpectTranslatedPlaneKeepsWetCell(CpuLiquidSubstepBackend3D(), Options());
}

TEST(LiquidSubstep3D, OpenTankSamplesPistonAndFixedWallsSeparately) {
  tank_staging_test::ExpectWallSamples(CpuLiquidSubstepBackend3D());
}

TEST(LiquidSubstep3D, OpenTankGhostTagPreservesTranslatedFaceResidual) {
  tank_staging_test::ExpectTranslatedFaceUsesLatticeResidual(CpuLiquidSubstepBackend3D());
}

TEST(LiquidSubstep3D, OpenTankConstrainedUpdateKeepsIndependentVelocityFields) {
  tank_staging_test::ExpectParticleUpdateUsesBothVelocityFields(CpuLiquidSubstepBackend3D());
}

TEST(LiquidSubstep3D, TankPoseClosureLeavesCommittedStateIntact) {
  auto scene = Scene();
  scene.field = LiquidHorizontalPlaneSurface3D{3};
  scene.obstacle = AxisAlignedLiquidTank3D{1.75, 3.5, 2, 1.5, 3.5, 2};
  const auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(1).ok());
  const auto result = scheduler->Attempt(1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 0);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, WettedPistonRefusesUncertifiedUnownedBroadSphere) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wet = wetted_piston_pressure_test;
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{1.1};
  scene.obstacle = wet::kBroadUncertifiedSphereTank;
  const auto initial = Initial();
  wet::ProjectionAudit audit;
  auto scheduler =
      LiquidSubstepScheduler3D::Create(initial, scene, Options(), wet::Observe(audit, CpuLiquidSubstepBackend3D()));
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kInterface);
  EXPECT_EQ(scheduler->diagnostics().dt_retries, 0);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
  wet::ExpectCheckpointEqual(initial, scheduler->committed());
  wet::ExpectBroadSphereRefusal(audit);
}

TEST(LiquidSubstep3D, AdvancesOpenTankWithTransportedField) {
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{wetted_piston_pressure_test::kSphereRadius};
  scene.obstacle = wetted_piston_pressure_test::kNarrowSphereTank;
  auto initial = Initial();
  initial.particles[0].world_position.z = 2.75;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(0.01);
  ASSERT_TRUE(status.ok()) << status << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
  EXPECT_FALSE(scheduler->committed().cells.values().empty());
  EXPECT_EQ(scheduler->committed().particles[0].side_tag, 0);
  wetted_piston_pressure_test::ExpectSphereWallFlux(*scheduler);
}

TEST(LiquidSubstep3D, PhiloxPublishedZeroKnownVector) {
  EXPECT_EQ(
      LiquidPhilox4x32_10({0, 0, 0, 0}, {0, 0}),
      (std::array<std::uint32_t, 4>{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8})
  );
}
TEST(LiquidSubstep3D, OrderedStagesPublishExactFrameEpoch) {
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
  EXPECT_EQ(scheduler->committed().clock.frame_ordinal, 1);
  EXPECT_EQ(scheduler->committed().clock.substep_ordinal, 0);
  EXPECT_DOUBLE_EQ(scheduler->committed().clock.time, 0.01);
  ASSERT_EQ(scheduler->diagnostics().stage_count, kLiquidPhysicalStageCount3D);
  for (std::size_t i = 0; i < kLiquidPhysicalStageCount3D; ++i) {
    EXPECT_EQ(static_cast<std::size_t>(scheduler->diagnostics().stages[i]), i);
  }
}
class LiquidGravity3D : public testing::TestWithParam<int> {};
TEST_P(LiquidGravity3D, AccelerationIndependentOfRetention) {
  auto options = Options();
  options.particle_energy_blend = std::array{0.0, 0.6, 1.0}[GetParam()];
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.y, -0.0981, 0.0000981);
}
INSTANTIATE_TEST_SUITE_P(LiquidSubstep3D, LiquidGravity3D, testing::Values(0, 1, 2));
TEST(LiquidSubstep3D, StoredResidualDoesNotAdvectInterior) {
  auto initial = Initial(2);
  initial.particles[0].world_velocity.x = 1;
  initial.particles[1].world_velocity.x = -1;
  auto scene = Scene();
  scene.world_acceleration = {};
  auto options = Options();
  options.particle_energy_blend = 1;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.x, 0.99, 1e-8);
  EXPECT_NEAR(scheduler->committed().particles[0].world_position.x, 2.5, 1e-8);
}
TEST(LiquidSubstep3D, FiniteExponentialVolumePreservesMassAndReferenceVolume) {
  auto initial = Initial();
  initial.particles[0].velocity_slopes = {{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}};
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.target_divergence = 0.3;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_GT(scheduler->committed().particles[0].volume_ratio, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].mass, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().particles[0].reference_volume, 1);
}
TEST(LiquidSubstep3D, AnalyticSpherePreservesExactRestPressureAtBothResolutions) {
  constexpr double kRadius = 0.125;
  constexpr double kDensity = 1000;
  struct AirPressureAudit {
    std::size_t air_links = 0;
    std::size_t incorrect_pressures = 0;
    std::size_t pressure_iterations = 0;
    double maximum_corrected_speed = 0;
  };
  for (const int cells_per_radius : {3, 5}) {
    SCOPED_TRACE(cells_per_radius);
    const double cell_width = kRadius / cells_per_radius;
    auto initial = Initial(cells_per_radius == 3 ? 136 : 552);
    initial.geometry.cell_width = cell_width;
    const double reference_volume = (4 * std::numbers::pi * kRadius * kRadius * kRadius / 3) / initial.particles.size();
    std::size_t ordinal = 0;
    for (int z = -cells_per_radius; z < cells_per_radius; ++z) {
      for (int y = -cells_per_radius; y < cells_per_radius; ++y) {
        for (int x = -cells_per_radius; x < cells_per_radius; ++x) {
          const Vec3 position{(x + 0.5) * cell_width, (y + 0.5) * cell_width, (z + 0.5) * cell_width};
          if (std::hypot(position.x, position.y, position.z) < kRadius) {
            ASSERT_LT(ordinal, initial.particles.size());
            auto& particle = initial.particles[ordinal++];
            particle.world_position = position;
            particle.reference_volume = reference_volume;
            particle.mass = kDensity * reference_volume;
            particle.pic_initialized = false;
          }
        }
      }
    }
    ASSERT_EQ(ordinal, initial.particles.size());
    LiquidSceneInputs3D scene;
    scene.field = LiquidAnalyticSphere3D{{}, kRadius};
    scene.world_acceleration = {};
    scene.liquid_density = kDensity;
    scene.surface_tension = 1;
    scene.ambient_pressure = 3;
    LiquidStepOptions3D options;
    options.capacities = {1024, 128, 4096, 24576, 256};
    AirPressureAudit audit;
    auto backend = CpuLiquidSubstepBackend3D();
    backend.context = &audit;
    backend.project = [](void* context,
                         SparseMacGridView3D grid,
                         double duration,
                         const LiquidPressureProblemView3D& problem,
                         MacPressureSolverOptions3D solver_options,
                         MacPressureWorkspace3D& workspace) {
      auto& observed = *static_cast<AirPressureAudit*>(context);
      for (const auto& face : problem.faces) {
        if (const auto* air = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link)) {
          ++observed.air_links;
          observed.incorrect_pressures += air->exterior_pressure != 3 + 2 / kRadius;
        }
      }
      const auto projection =
          CpuLiquidSubstepBackend3D().project(nullptr, grid, duration, problem, solver_options, workspace);
      if (projection.ok()) {
        observed.pressure_iterations += projection->diagnostics.iteration_count;
        for (const double velocity : projection->corrected_face_velocities) {
          observed.maximum_corrected_speed = std::max(observed.maximum_corrected_speed, std::abs(velocity));
        }
      }
      return projection;
    };
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    const auto status = scheduler->AdvanceFrame(1.0 / 120);
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ(audit.air_links, cells_per_radius == 3 ? 192 : 480);
    EXPECT_EQ(audit.incorrect_pressures, 0);
    EXPECT_EQ(audit.pressure_iterations, 0);
    EXPECT_DOUBLE_EQ(audit.maximum_corrected_speed, 0);
    scene.dynamic_viscosity = 1.0;
    auto viscous = LiquidSubstepScheduler3D::Create(initial, scene, options);
    ASSERT_TRUE(viscous.ok()) << viscous.status();
    const auto viscous_status = viscous->AdvanceFrame(1.0 / 120);
    ASSERT_TRUE(viscous_status.ok()) << viscous_status;
    for (const auto& cell : viscous->committed().cells.values()) {
      EXPECT_NEAR(cell.previous_pressure, 19.0, 1e-3);
    }
    for (const auto& particle : viscous->committed().particles.values()) {
      const auto velocity = particle.world_velocity;
      EXPECT_LT(std::hypot(velocity.x, velocity.y, velocity.z), 1e-5);
    }
    const auto& accepted = viscous->diagnostics().accepted_measurement;
    ASSERT_TRUE(accepted.has_value());
    ASSERT_TRUE(accepted->viscous_max_momentum_residual_kg_m_per_s.has_value());
    EXPECT_LT(*accepted->viscous_max_momentum_residual_kg_m_per_s, 1e-5);
  }
}

TEST(LiquidSubstep3D, ViscousFractionalTankFillPreservesHydrostaticRest) {
  auto fixture = viscous_tank_test::FractionalFill();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(viscous_tank_test::kStepDuration);
  ASSERT_TRUE(status.ok()) << status;
  viscous_tank_test::ExpectHydrostaticRest(*scheduler);

  auto& tank = std::get<AxisAlignedLiquidTank3D>(fixture->scene.obstacle);
  tank.start_low_x = 0.75;
  for (auto& particle : fixture->initial.particles.values()) {
    if (particle.world_position.x < 1.0) {
      particle.world_position.x = 0.875;
    }
  }
  auto solid_cut = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options);
  ASSERT_TRUE(solid_cut.ok()) << solid_cut.status();
  const auto rejected = solid_cut->AdvanceFrame(viscous_tank_test::kStepDuration);
  EXPECT_EQ(rejected.code(), absl::StatusCode::kUnimplemented);
  ExpectIdenticalSerializedState(fixture->initial, solid_cut->committed());
}

TEST(LiquidSubstep3D, ViscousCartesianCutTankPreservesHydrostaticRest) {
  const auto fixture = viscous_tank_test::CartesianCut();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(1.0 / 50.0);
  ASSERT_TRUE(status.ok()) << status;
  viscous_tank_test::ExpectHydrostaticRest(*scheduler);
}

TEST(LiquidSubstep3D, ViscousCartesianCutPistonUsesWallNormalStrainAndFlux) {
  auto fixture = viscous_tank_test::CartesianCut();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  auto& tank = std::get<AxisAlignedLiquidTank3D>(fixture->scene.obstacle);
  constexpr double kWallSpeed = 0.05;
  tank.piston_velocity_x = kWallSpeed;
  struct PistonAudit {
    std::size_t wall_rows = 0;
    double top_outward_flux = 0.0;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.project_viscous = [](void* context,
                               const LiquidBoundaryGeometryViewInputs3D& geometry,
                               const LiquidPressureProblemView3D& problem,
                               double viscosity,
                               std::span<const double> cell_viscosity,
                               MacPressureSolverOptions3D options,
                               MacPressureWorkspace3D& workspace) {
    return liquid_step_internal::ProjectUniformViscosity(
        geometry,
        problem,
        viscosity,
        cell_viscosity,
        options,
        workspace,
        [&](const pressure_internal::UniformViscousPressureInputs3D& inputs,
            std::span<const MacStrainRow3D> rows,
            MacPressureSolverOptions3D controls,
            pressure_internal::MacPressureScratch3D& pressure_scratch) {
          if (!rows.empty() && rows[0].low_cell.x == 0 && rows[0].low_cell.y == 0 && rows[0].low_cell.z == 0
              && rows[0].first_axis == FaceNormalAxis::kX && rows[0].second_axis == FaceNormalAxis::kX) {
            EXPECT_DOUBLE_EQ(rows[0].quadrature_weight, 15.0);
            EXPECT_DOUBLE_EQ(rows[0].terms[0].coefficient, -1.0 / 0.75);
            EXPECT_DOUBLE_EQ(rows[0].terms[1].coefficient, 1.0 / 0.75);
            ++static_cast<PistonAudit*>(context)->wall_rows;
          }
          const auto result =
              pressure_internal::ProjectScratchMacViscousPressure3D(inputs, rows, controls, pressure_scratch);
          if (result.ok()) {
            auto& observed = *static_cast<PistonAudit*>(context);
            observed.top_outward_flux = 0.0;
            for (std::size_t face = 0; face < inputs.pressure_faces.size(); ++face) {
              const auto& link = inputs.pressure_faces[face];
              if (link.face_index.axis == FaceNormalAxis::kY
                  && std::holds_alternative<DirichletMacPressureLink3D>(link.pressure_link)) {
                observed.top_outward_flux += link.open_area * result->face_velocities[face];
              }
            }
          }
          return result;
        }
    );
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options, backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  const auto status = scheduler->AdvanceFrame(viscous_tank_test::kStepDuration);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_GT(audit.wall_rows, 0u);
  const auto& accepted = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(accepted.has_value());
  const double expected_flux = kWallSpeed * 3 * viscous_tank_test::kFillHeight;
  EXPECT_NEAR(accepted->step_start_inward_piston_flux_m3_per_s, expected_flux, 1e-12);
  EXPECT_NEAR(audit.top_outward_flux, expected_flux, 0.01 * expected_flux);
  EXPECT_LT(accepted->viscous_max_momentum_residual_kg_m_per_s.value_or(1.0), 1e-5);
}

TEST(LiquidSubstep3D, CapillaryAmbientAndMovingWallAreResolved) {
  auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.surface_tension = 0.1;
  scene.ambient_pressure = 3;
  scene.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 0, {0.1, 0, 0}};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  ASSERT_GT(scheduler->committed().cells.size(), 0);
  EXPECT_GT(scheduler->committed().cells[0].previous_pressure, 3);
  EXPECT_LE(scheduler->diagnostics().pressure_diagnostics.max_cell_constraint_defect, 2e-5);
  const auto& measurement = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement.has_value());
  EXPECT_EQ(measurement->accepted_clock, scheduler->committed().clock);
  EXPECT_DOUBLE_EQ(measurement->accepted_dt_seconds, 0.01);
  ASSERT_TRUE(measurement->step_start_capillary_bound_seconds.has_value());
  EXPECT_NEAR(*measurement->step_start_capillary_bound_seconds, 28.209479177387813, 1e-12);
  EXPECT_GT(measurement->pressure_rhs_l2_m3_per_s, 0);
  EXPECT_FALSE(measurement->piston_pose.has_value());
  EXPECT_DOUBLE_EQ(measurement->step_start_inward_piston_flux_m3_per_s, 0);
}
class LiquidStageFailure3D : public testing::TestWithParam<int> {};
TEST_P(LiquidStageFailure3D, CommittedInventoryUnchanged) {
  auto initial = Initial();
  initial.random_seed = 0xfedcba9876543210ULL;
  initial.sources = *LiquidRecords3D<LiquidSourceState3D>::Create(1);
  ASSERT_TRUE(initial.sources.Resize(1).ok());
  initial.sources[0] = {0, 8, 0.25};
  LiquidEmissionSource3D source;
  source.particles_per_second = 130;
  source.world_position = {2.5, 2.5, 2.5};
  source.world_position_jitter = {0.1, 0.1, 0.1};
  auto scene = Scene();
  scene.sources = std::span(&source, 1);
  scene.ambient_pressure = 3;
  scene.surface_tension = 0.1;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  auto snapshot = CloneLiquidState3D(scheduler->committed(), Options().capacities);
  ASSERT_TRUE(snapshot.ok());
  ASSERT_EQ(snapshot->random_seed, initial.random_seed);
  ASSERT_GT(snapshot->next_particle_id, initial.next_particle_id);
  ASSERT_GT(snapshot->sources[0].next_child_ordinal, initial.sources[0].next_child_ordinal);
  ASSERT_GT(snapshot->sources[0].fractional_remainder, 0);
  ASSERT_FALSE(snapshot->blocks.values().empty());
  ASSERT_TRUE(std::ranges::any_of(snapshot->cells.values(), [](const LiquidCellState3D& cell) {
    return cell.previous_pressure != 0;
  }));
  ASSERT_TRUE(std::ranges::any_of(snapshot->faces.values(), [](const LiquidFaceState3D& face) {
    return face.carrier_velocity != 0 && face.validity != LiquidFaceValidity3D::kMissing;
  }));
  struct RejectionProbe {
    LiquidPhysicalStage3D selected_stage;
    std::size_t injection_count = 0;
  } rejection_probe{.selected_stage = static_cast<LiquidPhysicalStage3D>(GetParam())};
  scheduler->SetStageFaultHookForTesting({
      &rejection_probe,
      [](void* context, LiquidPhysicalStage3D current_stage, double) {
        auto& probe = *static_cast<RejectionProbe*>(context);
        if (current_stage == probe.selected_stage) {
          ++probe.injection_count;
          return absl::InternalError("");
        }
        return absl::OkStatus();
      },
  });
  ASSERT_TRUE(scheduler->BeginFrame(0.02).ok());
  auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_TRUE(scheduler->committed() == *snapshot);
  ExpectIdenticalSerializedState(*snapshot, scheduler->committed());
  EXPECT_EQ(rejection_probe.injection_count, 1);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, rejection_probe.selected_stage);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kInternal);
}
INSTANTIATE_TEST_SUITE_P(
    LiquidSubstep3D,
    LiquidStageFailure3D,
    testing::Range(0, static_cast<int>(LiquidPhysicalStage3D::kAdapt))
);
TEST(LiquidSubstep3D, RetryHalvesTwiceButIncompatibilityIsFatal) {
  for (const auto code : {absl::StatusCode::kAborted, absl::StatusCode::kFailedPrecondition}) {
    auto initial = Initial();
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    auto error = code;
    scheduler->SetStageFaultHookForTesting({&error, [](void* context, LiquidPhysicalStage3D stage, double) {
                                              return stage == LiquidPhysicalStage3D::kPressure
                                                  ? absl::Status(*static_cast<absl::StatusCode*>(context), {})
                                                  : absl::OkStatus();
                                            }});
    EXPECT_FALSE(scheduler->AdvanceFrame(0.01).ok());
    EXPECT_EQ(scheduler->diagnostics().attempts, code == absl::StatusCode::kAborted ? 3 : 1);
    EXPECT_TRUE(scheduler->committed() == initial);
  }
}
TEST(LiquidSubstep3D, SourceRetryPreservesCounterAndSeededBirths) {
  auto initial = Initial(0);
  initial.sources = *LiquidRecords3D<LiquidSourceState3D>::Create(1);
  ASSERT_TRUE(initial.sources.Resize(1).ok());
  LiquidEmissionSource3D source;
  source.particles_per_second = 100;
  source.world_position = {2.5, 2.5, 2.5};
  source.world_position_jitter = {0.1, 0.1, 0.1};
  auto scene = Scene();
  scene.sources = std::span(&source, 1);
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(scheduler->committed().particles.size(), 1);
  EXPECT_EQ(scheduler->committed().sources[0].next_child_ordinal, 1);
  EXPECT_EQ(scheduler->committed().next_particle_id, 2);
}
TEST(LiquidSubstep3D, CapacityGrowthReplaysWholeFrameOnce) {
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  namespace wall_test = liquid_wall_samples_test;
  const auto initial = wall_test::SlabParticles(2);
  const auto scene = wall_test::SlabScene();
  auto reference = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(reference.ok());
  ASSERT_TRUE(reference->AdvanceFrame(0.01).ok());
  wall_test::Audit audit(options.capacities.faces);
  auto scheduler =
      LiquidSubstepScheduler3D::Create(initial, scene, options, wall_test::Observe(audit, CpuLiquidSubstepBackend3D()));
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            if (stage == LiquidPhysicalStage3D::kEmit
                                                && ++*static_cast<int*>(context) == 2) {
                                              return absl::Status(absl::StatusCode::kResourceExhausted, {});
                                            }
                                            return absl::OkStatus();
                                          }});
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(scheduler->diagnostics().capacity_replays, 1);
  EXPECT_EQ(scheduler->committed().clock.epoch, 2);
  wall_test::ExpectGrowthRefreshes(audit);
  ExpectIdenticalSerializedState(reference->committed(), scheduler->committed());
}
TEST(LiquidSubstep3D, ProgressObserverInvalidatesDiscardedPassAndPreservesSchedule) {
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  auto reference = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(reference.ok());
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            if (stage == LiquidPhysicalStage3D::kEmit
                                                && ++*static_cast<int*>(context) == 2) {
                                              return absl::ResourceExhaustedError("");
                                            }
                                            return absl::OkStatus();
                                          }});
  std::vector<std::pair<std::uint64_t, bool>> observations;
  ASSERT_TRUE(scheduler
                  ->AdvanceFrame(
                      0.01,
                      {&observations,
                       [](void* context, const LiquidState3D& state, const LiquidStepDiagnostics3D&, bool rollback) {
                         static_cast<decltype(observations)*>(context)->emplace_back(state.clock.epoch, rollback);
                         return absl::OkStatus();
                       }}
                  )
                  .ok());
  EXPECT_EQ(observations, (decltype(observations){{1, false}, {0, true}, {1, false}, {2, false}}));
  ASSERT_TRUE(reference->AdvanceFrame(0.01).ok());
  EXPECT_TRUE(scheduler->committed() == reference->committed());
}
TEST(LiquidSubstep3D, ProgressStopRetainsOneAcceptedSubstepWithoutFrameCompletion) {
  auto options = Options();
  options.maximum_step_duration = 0.005;
  auto scheduler = LiquidSubstepScheduler3D::Create(Initial(), Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(
      scheduler
          ->AdvanceFrame(
              0.01,
              {nullptr,
               [](void*, const LiquidState3D& state, const LiquidStepDiagnostics3D& diagnostics, bool rollback) {
                 EXPECT_FALSE(rollback);
                 EXPECT_EQ(state.clock.epoch, 1);
                 EXPECT_EQ(diagnostics.accepted_attempts, 1);
                 return absl::CancelledError("");
               }}
          )
          .code(),
      absl::StatusCode::kCancelled
  );
  EXPECT_EQ(scheduler->committed().clock.frame_ordinal, 0);
  EXPECT_EQ(scheduler->committed().clock.substep_ordinal, 1);
  EXPECT_DOUBLE_EQ(scheduler->committed().clock.time, 0.005);
}
TEST(LiquidSubstep3D, RepeatedGravityAndUniformTranslationRemainAnalytic) {
  for (double energy : {0.0, 0.6, 1.0}) {
    auto initial = Initial();
    initial.particles[0].world_velocity = {0.5, 0.25, -0.1};
    auto options = Options();
    options.particle_energy_blend = energy;
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
    ASSERT_TRUE(scheduler.ok());
    for (int frame = 1; frame <= 5; ++frame) {
      ASSERT_TRUE(scheduler->AdvanceFrame(0.01 * frame).ok())
          << static_cast<int>(scheduler->diagnostics().failed_stage);
    }
    EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.x, 0.5, 0.0005);
    EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.y, 0.25 - 9.81 * 0.05, 0.0004905);
    EXPECT_NEAR(scheduler->committed().particles[0].world_velocity.z, -0.1, 0.0001);
  }
}
TEST(LiquidSubstep3D, MovingWallKeepsDistinctOldAndNewEligibility) {
  auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.obstacle = TranslatingHalfspace3D{{1, 0, 0}, 1.995, {1, 0, 0}};
  struct Observed {
    bool changed = false;
    bool times_match = true;
  } observed;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observed;
  backend.gather = [](void* context,
                      const LiquidBoundaryGatherViewInputs3D& inputs,
                      LiquidBoundaryWorkspace3D& workspace) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    auto& result = *static_cast<Observed*>(context);
    result.times_match = result.times_match && inputs.old_history.wall_sample_time == inputs.old_time
        && inputs.new_history.wall_sample_time == inputs.new_time;
    for (std::size_t i = 0; i < inputs.old_history.face_tags.size(); ++i) {
      result.changed = result.changed
          || (inputs.old_history.face_tags[i].structurally_eligible
              && !inputs.new_history.face_tags[i].structurally_eligible);
    }
    return GatherLiquidBoundaryParticles3D(inputs, workspace);
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok()) << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_TRUE(observed.changed);
  EXPECT_TRUE(observed.times_match);
}
TEST(LiquidSubstep3D, ThinSlabSharedStencilNodesUseBothIncidentWallNormals) {
  auto initial = Initial(2);
  initial.particles[0].world_position.x = 1.4;
  initial.particles[0].side_tag = -1;
  initial.particles[1].world_position.x = 2.6;
  initial.particles[1].side_tag = 1;
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{0.9};
  scene.world_acceleration = {};
  scene.obstacle = TranslatingSlab3D{{1, 0, 0}, 2, 0.05, {0.1, 0, 0}};
  std::array<bool, 2> seen{};
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &seen;
  backend.gather = [](void* context, const LiquidBoundaryGatherViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace
                   ) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    auto& observed = *static_cast<std::array<bool, 2>*>(context);
    const auto side = input.particles[0].side_tag;
    for (const auto& wall : input.old_history.wall_samples) {
      if (wall.available) {
        EXPECT_EQ(wall.side_tag, side);
        EXPECT_DOUBLE_EQ(wall.unit_normal.x, static_cast<double>(side));
        observed[side > 0 ? 1 : 0] = true;
      }
    }
    return GatherLiquidBoundaryParticles3D(input, workspace);
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok()) << static_cast<int>(scheduler->diagnostics().failed_stage);
  EXPECT_TRUE(seen[0]);
  EXPECT_TRUE(seen[1]);
  EXPECT_LT(scheduler->committed().particles[0].world_position.x, 1.96);
  EXPECT_GT(scheduler->committed().particles[1].world_position.x, 2.06);
}

TEST(LiquidSubstep3D, ReusesWallSamplesInCanonicalSlabSideRuns) {
  namespace wall_test = liquid_wall_samples_test;
  const auto options = Options();
  for (const std::size_t particle_count : {0, 2, 5}) {
    SCOPED_TRACE(particle_count);
    const auto initial = wall_test::SlabParticles(particle_count);
    wall_test::Audit audit(options.capacities.faces);
    auto scheduler = LiquidSubstepScheduler3D::Create(
        initial,
        wall_test::SlabScene(),
        options,
        wall_test::Observe(audit, CpuLiquidSubstepBackend3D())
    );
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
    const auto calls = wall_test::SlabCalls(2.501);
    const std::size_t expected_call_count = particle_count == 0 ? 0 : particle_count == 2 ? 6 : calls.size();
    wall_test::ExpectCalls(audit, std::span(calls).first(expected_call_count));
    EXPECT_EQ(audit.gather_count, particle_count);
    EXPECT_EQ(scheduler->diagnostics().old_pic_fallback_count, audit.old_fallbacks);
    EXPECT_EQ(scheduler->diagnostics().new_pic_fallback_count, audit.new_fallbacks);
    for (std::size_t index = 0; index < particle_count; ++index) {
      const auto& particle = scheduler->committed().particles[index];
      EXPECT_EQ(particle.id, index + 1);
      EXPECT_EQ(particle.side_tag, wall_test::kSides[index]);
      if (particle.side_tag < 0) {
        EXPECT_LT(particle.world_position.x, 2.45);
      } else {
        EXPECT_GT(particle.world_position.x, 2.551);
      }
    }
  }
}

TEST(LiquidSubstep3D, RejectedGatherRunReplaysOnlyThroughEarlierFieldFailure) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wall_test = liquid_wall_samples_test;
  struct Audit {
    int batch_calls = 0;
    int scalar_gather_calls = 0;
    int field_calls_after_batch = 0;
  } audit;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &audit;
  backend.gather_run = [](void* context,
                          const LiquidBoundaryGatherViewInputs3D& input,
                          LiquidBoundaryWorkspace3D&) -> absl::StatusOr<LiquidBoundaryGatherView3D> {
    auto& observed = *static_cast<Audit*>(context);
    ++observed.batch_calls;
    EXPECT_EQ(input.particles.size(), 2);
    EXPECT_EQ(input.particles.front().side_tag, input.particles.back().side_tag);
    return absl::OutOfRangeError("");
  };
  backend.gather = [](void* context, const LiquidBoundaryGatherViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace
                   ) {
    ++static_cast<Audit*>(context)->scalar_gather_calls;
    EXPECT_EQ(input.particles.size(), 1);
    return GatherLiquidBoundaryParticles3D(input, workspace);
  };
  backend.sample_field =
      [](void* context, const LiquidSceneField3D& field, std::span<const LiquidParticleState3D> particles, Vec3 position
      ) -> absl::StatusOr<double> {
    auto& observed = *static_cast<Audit*>(context);
    if (observed.batch_calls > 0) {
      ++observed.field_calls_after_batch;
      return absl::InvalidArgumentError("");
    }
    return CpuLiquidSubstepBackend3D().sample_field(nullptr, field, particles, position);
  };
  const auto initial = wall_test::SlabParticles();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, wall_test::SlabScene(), Options(), backend);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(audit.batch_calls, 1);
  EXPECT_EQ(audit.scalar_gather_calls, 1);
  EXPECT_EQ(audit.field_calls_after_batch, 1);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}

TEST(LiquidSubstep3D, GatheredFieldBatchMatchesScalarAndReplaysRejectionInOrder) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wall_test = liquid_wall_samples_test;
  const auto initial = wall_test::SlabParticles();
  const auto scene = wall_test::SlabScene();
  const auto options = Options();
  auto baseline = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  ASSERT_TRUE(baseline->AdvanceFrame(0.01).ok());
  for (const bool reject_batch : {false, true}) {
    SCOPED_TRACE(reject_batch);
    struct Audit {
      bool reject_batch;
      std::size_t batch_calls = 0;
      std::size_t gathered_field_samples = 0;
      std::size_t scalar_calls_after_batch = 0;
    } audit{reject_batch};
    auto backend = CpuLiquidSubstepBackend3D();
    backend.context = &audit;
    backend.gather_run = [](void*, const LiquidBoundaryGatherViewInputs3D& input, LiquidBoundaryWorkspace3D& workspace
                         ) {
      return GatherLiquidBoundaryParticles3D(input, workspace);
    };
    backend.sample_gathered_fields = [](void* context,
                                        const LiquidSceneField3D& field,
                                        std::span<const LiquidParticleState3D> particles,
                                        std::span<const Vec3> positions,
                                        std::span<double> distances) {
      auto& observed = *static_cast<Audit*>(context);
      ++observed.batch_calls;
      EXPECT_EQ(positions.size(), 2);
      EXPECT_EQ(positions.size(), distances.size());
      for (const auto position : positions) {
        const auto expected = wall_test::ParticlePosition(observed.gathered_field_samples++);
        EXPECT_DOUBLE_EQ(position.x, expected.x);
        EXPECT_DOUBLE_EQ(position.y, expected.y);
        EXPECT_DOUBLE_EQ(position.z, expected.z);
      }
      if (observed.reject_batch) {
        return absl::InvalidArgumentError("");
      }
      return CpuLiquidSubstepBackend3D().sample_field_batch(nullptr, field, particles, positions, distances);
    };
    backend.sample_field = [](void* context,
                              const LiquidSceneField3D& field,
                              std::span<const LiquidParticleState3D> particles,
                              Vec3 position) -> absl::StatusOr<double> {
      auto& observed = *static_cast<Audit*>(context);
      if (observed.batch_calls > 0) {
        ++observed.scalar_calls_after_batch;
        if (observed.reject_batch) {
          EXPECT_DOUBLE_EQ(position.y, wall_test::ParticlePosition(0).y);
          return absl::OutOfRangeError("");
        }
      }
      return CpuLiquidSubstepBackend3D().sample_field(nullptr, field, particles, position);
    };
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    const auto status = scheduler->AdvanceFrame(0.01);
    if (reject_batch) {
      EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
      EXPECT_EQ(audit.batch_calls, 1);
      EXPECT_EQ(audit.gathered_field_samples, 2);
      ExpectIdenticalSerializedState(initial, scheduler->committed());
    } else {
      ASSERT_TRUE(status.ok()) << status;
      EXPECT_EQ(audit.batch_calls, 2);
      EXPECT_EQ(audit.gathered_field_samples, 4);
      ExpectIdenticalSerializedState(baseline->committed(), scheduler->committed());
      EXPECT_EQ(scheduler->diagnostics().old_pic_fallback_count, baseline->diagnostics().old_pic_fallback_count);
      EXPECT_EQ(scheduler->diagnostics().new_pic_fallback_count, baseline->diagnostics().new_pic_fallback_count);
    }
    EXPECT_EQ(audit.scalar_calls_after_batch, 1);
  }
}

TEST(LiquidSubstep3D, FailedOldOrNewWallRefreshRetainsOnlyEarlierDiagnostics) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  namespace wall_test = liquid_wall_samples_test;
  const auto options = Options();
  const auto initial = wall_test::SlabParticles();
  const auto scene = wall_test::SlabScene();
  for (const bool fail_new : {false, true}) {
    SCOPED_TRACE(fail_new);
    wall_test::Audit audit(options.capacities.faces);
    auto scheduler = LiquidSubstepScheduler3D::Create(
        initial,
        scene,
        options,
        wall_test::Observe(audit, CpuLiquidSubstepBackend3D())
    );
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
    audit.fail_at_wall_coordinate = fail_new ? 2.501 : 2.5;
    const auto outcome = scheduler->Attempt(0.01);
    ASSERT_TRUE(outcome.ok()) << outcome.status();
    EXPECT_EQ(outcome->disposition, LiquidAttemptDisposition3D::kRetryDt);
    const auto failed_calls = wall_test::SlabCalls(2.501);
    wall_test::ExpectCalls(audit, std::span(failed_calls).first(fail_new ? 8 : 7));
    EXPECT_EQ(audit.gather_count, 2);
    EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kGather);
    EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kAborted);
    EXPECT_EQ(scheduler->diagnostics().old_pic_fallback_count, audit.old_fallbacks);
    EXPECT_EQ(scheduler->diagnostics().new_pic_fallback_count, audit.new_fallbacks);
    EXPECT_EQ(scheduler->diagnostics().accepted_attempts, 0);
    ExpectIdenticalSerializedState(initial, scheduler->committed());

    audit.Reset();
    audit.fail_at_wall_coordinate = 0;
    const auto retried = scheduler->Attempt(0.005);
    ASSERT_TRUE(retried.ok()) << retried.status();
    ASSERT_EQ(retried->disposition, LiquidAttemptDisposition3D::kAccepted);
    wall_test::ExpectCalls(audit, wall_test::SlabCalls(2.5005));
    auto reference = LiquidSubstepScheduler3D::Create(initial, scene, options);
    ASSERT_TRUE(reference.ok());
    ASSERT_TRUE(reference->BeginFrame(0.01).ok());
    const auto expected = reference->Attempt(0.005);
    ASSERT_TRUE(expected.ok());
    ASSERT_EQ(expected->disposition, LiquidAttemptDisposition3D::kAccepted);
    ExpectIdenticalSerializedState(reference->committed(), scheduler->committed());
  }
}

TEST(LiquidSubstep3D, EmptyUnionAndMalformedSignedExtentAreDistinguished) {
  auto empty = Initial(0);
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{0.5};
  auto scheduler = LiquidSubstepScheduler3D::Create(empty, scene, Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(scheduler->committed().cells.size(), 0);
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
  scene.field = LiquidAnalyticSphere3D{{static_cast<double>(INT32_MAX), 0, 0}, 1};
  auto invalid = LiquidSubstepScheduler3D::Create(empty, scene, Options());
  ASSERT_TRUE(invalid.ok());
  EXPECT_EQ(invalid->AdvanceFrame(0.01).code(), absl::StatusCode::kOutOfRange);
  EXPECT_TRUE(invalid->committed() == empty);
}
TEST(LiquidSubstep3D, SourceRetryAndWholeFrameReplayProduceIdenticalBirths) {
  std::cout << "CTEST_FULL_OUTPUT\n";
  auto initial = Initial();
  initial.random_seed = 123456789;
  initial.sources = *LiquidRecords3D<LiquidSourceState3D>::Create(1);
  ASSERT_TRUE(initial.sources.Resize(1).ok());
  initial.sources[0] = {0, 3, 0.25};
  LiquidEmissionSource3D source;
  source.particles_per_second = 400;
  source.world_position = {2.5, 2.5, 2.5};
  source.world_position_jitter = {0.2, 0.2, 0.2};
  source.emitted_particle_template.phase_id = 4;
  source.emitted_particle_template.constituent_count = 2;
  source.emitted_particle_template.constituents[0] = {8, 0.25};
  source.emitted_particle_template.constituents[1] = {14, 0.75};
  auto scene = Scene();
  scene.sources = std::span(&source, 1);
  constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1000.0}, {14, 4, 1020.0}}};
  scene.constituent_materials = kMaterials;
  scene.ambient_pressure = 3;
  scene.surface_tension = 0.1;
  auto options = Options();
  // Dyadic times keep the retried and uninterrupted schedules bit-identical.
  options.maximum_step_duration = 0.0078125;
  auto bootstrap = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(bootstrap.ok());
  ASSERT_TRUE(bootstrap->AdvanceFrame(0.0078125).ok());
  const auto& warm_start = bootstrap->committed();
  ASSERT_TRUE(std::ranges::any_of(warm_start.cells.values(), [](const LiquidCellState3D& cell) {
    return cell.previous_pressure != 0;
  }));
  auto reference = LiquidSubstepScheduler3D::Create(warm_start, scene, options);
  ASSERT_TRUE(reference.ok());
  ASSERT_TRUE(reference->AdvanceFrame(0.0234375).ok());
  options.maximum_step_duration = 0.015625;
  auto retry = LiquidSubstepScheduler3D::Create(warm_start, scene, options);
  ASSERT_TRUE(retry.ok());
  int acceptance_stage_count = 0;
  retry->SetStageFaultHookForTesting({&acceptance_stage_count, [](void* context, LiquidPhysicalStage3D stage, double) {
                                        if (stage == LiquidPhysicalStage3D::kAccept
                                            && ++*static_cast<int*>(context) == 1) {
                                          return absl::Status(absl::StatusCode::kAborted, {});
                                        }
                                        return absl::OkStatus();
                                      }});
  ASSERT_TRUE(retry->AdvanceFrame(0.0234375).ok());
  ASSERT_EQ(retry->diagnostics().dt_retries, 1);
  EXPECT_TRUE(retry->committed() == reference->committed());
  ExpectIdenticalSerializedState(reference->committed(), retry->committed());
  // Dyadic times keep the retried and uninterrupted schedules bit-identical.
  options.maximum_step_duration = 0.0078125;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  auto replay = LiquidSubstepScheduler3D::Create(warm_start, scene, options);
  ASSERT_TRUE(replay.ok());
  acceptance_stage_count = 0;
  replay->SetStageFaultHookForTesting({&acceptance_stage_count, [](void* context, LiquidPhysicalStage3D stage, double) {
                                         if (stage == LiquidPhysicalStage3D::kAccept
                                             && ++*static_cast<int*>(context) == 2) {
                                           return absl::Status(absl::StatusCode::kResourceExhausted, {});
                                         }
                                         return absl::OkStatus();
                                       }});
  ASSERT_TRUE(replay->AdvanceFrame(0.0234375).ok());
  ASSERT_EQ(replay->diagnostics().capacity_replays, 1);
  EXPECT_TRUE(replay->committed() == reference->committed());
  for (const auto& particle : replay->committed().particles.values()) {
    if (particle.id != 1) {
      EXPECT_EQ(particle.phase_id, 4);
      ASSERT_EQ(particle.constituent_count, 2);
      EXPECT_EQ(particle.constituents[0], (LiquidConstituentFraction3D{8, 0.25}));
      EXPECT_EQ(particle.constituents[1], (LiquidConstituentFraction3D{14, 0.75}));
    }
  }
  ExpectIdenticalSerializedState(reference->committed(), replay->committed());
  std::cout << "f9_retry_attempts=" << retry->diagnostics().attempts
            << " f9_replay_attempts=" << replay->diagnostics().attempts
            << " f9_pressure_iterations=" << replay->diagnostics().pressure_diagnostics.iteration_count
            << " f9_reserved_bytes=" << replay->reserved_bytes() << '\n';
}
TEST(LiquidSubstep3D, RepeatedCapacityFailureRetainsSerializedFrameStart) {
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.005).ok());
  auto frame_start = CloneLiquidState3D(scheduler->committed(), options.capacities);
  ASSERT_TRUE(frame_start.ok());

  int acceptance_stage_count = 0;
  scheduler->SetStageFaultHookForTesting({
      &acceptance_stage_count,
      [](void* context, LiquidPhysicalStage3D stage, double) {
        if (stage == LiquidPhysicalStage3D::kAccept && ++*static_cast<int*>(context) >= 2) {
          return absl::ResourceExhaustedError("");
        }
        return absl::OkStatus();
      },
  });
  EXPECT_EQ(scheduler->AdvanceFrame(0.015).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(acceptance_stage_count, 3);
  EXPECT_EQ(scheduler->diagnostics().accepted_attempts, 1);
  EXPECT_EQ(scheduler->diagnostics().capacity_replays, 1);
  ExpectIdenticalSerializedState(*frame_start, scheduler->committed());
}

TEST(LiquidSubstep3D, RetainedAgesAdvanceOnlyOnAcceptanceAndFinishDoesNotMutateEpoch) {
  auto initial = Initial();
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  for (const auto& block : scheduler->committed().blocks.values()) {
    EXPECT_EQ(block.activation_age, 0);
  }
  ASSERT_TRUE(scheduler->BeginFrame(0.02).ok());
  auto accepted = scheduler->Attempt(0.01);
  ASSERT_TRUE(accepted.ok());
  ASSERT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
  for (const auto& block : scheduler->committed().blocks.values()) {
    EXPECT_EQ(block.activation_age, 1);
  }
  auto snapshot = CloneLiquidState3D(scheduler->committed(), Options().capacities);
  ASSERT_TRUE(snapshot.ok());
  ASSERT_TRUE(scheduler->FinishFrame().ok());
  // M1 adaptation is an identity stage recorded after publication. Closing
  // the schedule cannot install another committed version.
  ASSERT_GT(scheduler->diagnostics().stage_count, 0);
  EXPECT_EQ(scheduler->diagnostics().stages[scheduler->diagnostics().stage_count - 1], LiquidPhysicalStage3D::kAdapt);
  EXPECT_TRUE(scheduler->committed() == *snapshot);
  ExpectIdenticalSerializedState(*snapshot, scheduler->committed());
}
TEST(LiquidSubstep3D, ExponentialUsesTrueGradientAndSeparationUsesSurfaceExpansion) {
  auto initial = Initial(2);
  for (auto& particle : initial.particles.values()) {
    particle.world_position.x = 3.3;
    particle.velocity_slopes = {{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}};
  }
  initial.particles[0].world_velocity.x = 1;
  initial.particles[1].world_velocity.x = -1;
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.target_divergence = 0.3;
  struct Observation {
    double divergence = 0;
    double pic = 0;
    double residual = 0;
  } observed;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observed;
  backend.update_particles =
      [](void* c, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
        const auto& value = inputs.gathered[0];
        *static_cast<Observation*>(c) = {
            value.velocity_gradient.x_row.x + value.velocity_gradient.y_row.y + value.velocity_gradient.z_row.z,
            value.new_pic_velocity.x,
            value.particle_minus_old_pic_velocity.x,
        };
        return CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
      };
  auto options = Options();
  options.particle_energy_blend = 0.6;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  ASSERT_GT(observed.divergence, 0);
  ASSERT_GT(observed.residual, 0.5);
  const auto& particle = scheduler->committed().particles[0];
  EXPECT_NEAR(particle.volume_ratio, std::exp(0.01 * observed.divergence), 1e-13);
  EXPECT_NEAR(particle.world_velocity.x, observed.pic + 0.594 * observed.residual, 1e-12);
  EXPECT_NEAR(particle.world_position.x, 3.3 + 0.01 * (observed.pic + 0.6 * 0.594 * observed.residual), 1e-12);
}
TEST(LiquidSubstep3D, CompleteBackendIsRequiredAndRealPressureNonconvergenceRetries) {
  auto initial = Initial(8);
  for (std::size_t i = 0; i < 8; ++i) {
    auto& p = initial.particles[i];
    p.world_position = {1.5 + 2 * (i & 1), 1.5 + 2 * ((i >> 1) & 1), 1.5 + 2 * ((i >> 2) & 1)};
    p.world_velocity = {
        0.1 * static_cast<double>(i),
        0.03 * static_cast<double>(i * i),
        -0.07 * static_cast<double>(i),
    };
  }
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 2.1};
  auto options = Options();
  options.pressure_solver_options.max_iterations = 1;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.project = nullptr;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, scene, options, backend).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok());
  std::vector<LiquidAttemptMeasurement3D> attempts;
  const LiquidProgressObserver3D observer{
      .context = &attempts,
      .observe_attempt =
          [](void* context, const LiquidAttemptMeasurement3D& attempt) {
            static_cast<decltype(attempts)*>(context)->push_back(attempt);
          },
  };
  EXPECT_EQ(scheduler->AdvanceFrame(0.01, observer).code(), absl::StatusCode::kAborted);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  EXPECT_EQ(scheduler->diagnostics().attempts, 3);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement.has_value());
  EXPECT_TRUE(scheduler->committed() == initial);
  ASSERT_EQ(attempts.size(), 3);
  constexpr std::array attempted_durations{0.01, 0.005, 0.0025};
  for (std::size_t index = 0; index < attempts.size(); ++index) {
    const auto& attempt = attempts[index];
    EXPECT_EQ(attempt.start_clock, initial.clock);
    EXPECT_EQ(attempt.attempt_in_frame, index + 1);
    EXPECT_EQ(attempt.capacity_pass, 0);
    EXPECT_DOUBLE_EQ(attempt.attempted_dt_seconds, attempted_durations[index]);
    EXPECT_TRUE(std::isfinite(attempt.work_wall_seconds));
    EXPECT_GE(attempt.work_wall_seconds, 0);
    EXPECT_EQ(attempt.failure_code, absl::StatusCode::kAborted);
    if (index + 1 == attempts.size()) {
      EXPECT_EQ(attempt.disposition, LiquidAttemptDisposition3D::kFatal);
      EXPECT_FALSE(attempt.retry_dt_seconds.has_value());
    } else {
      EXPECT_EQ(attempt.disposition, LiquidAttemptDisposition3D::kRetryDt);
      ASSERT_TRUE(attempt.retry_dt_seconds.has_value());
      EXPECT_DOUBLE_EQ(*attempt.retry_dt_seconds, attempted_durations[index + 1]);
    }
  }
}
#if defined(_WIN32) && defined(_DEBUG)
std::size_t scheduler_allocations = 0;
std::size_t scheduler_requested_bytes = 0;
std::size_t scheduler_last_request = 0;
std::size_t scheduler_frees = 0;
std::size_t scheduler_fail_allocation = 0;
_CRT_ALLOC_HOOK scheduler_previous_hook = nullptr;
int SchedulerAllocationHook(
    int operation,
    void* data,
    std::size_t size,
    int block_type,
    long request,
    const unsigned char* filename,
    int line
) {
  if (operation == _HOOK_ALLOC || operation == _HOOK_REALLOC) {
    ++scheduler_allocations;
    scheduler_requested_bytes += size;
    scheduler_last_request = size;
    if (scheduler_allocations == scheduler_fail_allocation) {
      return 0;
    }
  }
  if (operation == _HOOK_FREE) {
    ++scheduler_frees;
  }
  return scheduler_previous_hook ? scheduler_previous_hook(operation, data, size, block_type, request, filename, line)
                                 : 1;
}
TEST(Viscosity, BoundedVelocityBlockSolveAllocatesNothingAndMatchesOwningPath) {
  const std::array<double, 3> inertia{2.0, 3.0, 0.0};
  std::array<double, 3> rhs{2.0, -1.0, 0.0};
  const std::array rows{MacStrainRow3D{
      .low_cell = {},
      .first_axis = FaceNormalAxis::kX,
      .second_axis = FaceNormalAxis::kY,
      .terms =
          {{{.face_slot = 0, .coefficient = -1.0},
            {.face_slot = 1, .coefficient = 1.0},
            {.face_slot = 2, .coefficient = 0.5}}},
      .term_count = 3,
      .quadrature_weight = 4.0,
  }};
  const MacViscousVelocityBlockInputs3D inputs{rows, inertia, 0.2};
  const auto owning = SolveMacViscousVelocityBlock3D(inputs, rhs);
  ASSERT_TRUE(owning.ok()) << owning.status();
  std::array<MacStrainRow3D, 1> weighted_rows;
  std::array<double, 3> diagonal{}, velocity{}, residual{}, preconditioned{}, direction{}, applied{};
  const viscosity_internal::VelocityBlockScratch3D scratch{
      .weighted_rows = weighted_rows,
      .diagonal = diagonal,
      .velocity = velocity,
      .residual = residual,
      .preconditioned = preconditioned,
      .direction = direction,
      .applied = applied,
  };
  scheduler_allocations = 0;
  scheduler_frees = 0;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  const auto bounded = viscosity_internal::SolveVelocityBlockWithScratch3D(inputs, rhs, {}, scratch);
  _CrtSetAllocHook(scheduler_previous_hook);
  ASSERT_TRUE(bounded.ok()) << bounded.status();
  EXPECT_TRUE(std::ranges::equal(bounded->velocity, owning->velocity));
  EXPECT_EQ(bounded->iterations, owning->iterations);
  EXPECT_DOUBLE_EQ(bounded->max_residual, owning->max_residual);
  EXPECT_EQ(scheduler_allocations, 0u);
  EXPECT_EQ(scheduler_frees, 0u);

  auto alias = scratch;
  alias.velocity = residual;
  scheduler_allocations = 0;
  scheduler_frees = 0;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  const auto rejected = viscosity_internal::SolveVelocityBlockWithScratch3D(inputs, rhs, {}, alias);
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(scheduler_allocations, 0u);
  EXPECT_EQ(scheduler_frees, 0u);
  const auto original_rhs = rhs;
  scheduler_allocations = 0;
  scheduler_frees = 0;
  auto failure_code = absl::StatusCode::kOk;
  bool empty_message = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    const auto exhausted =
        viscosity_internal::SolveVelocityBlockWithScratch3D(inputs, rhs, {.max_iterations = 1}, scratch);
    failure_code = exhausted.status().code();
    empty_message = exhausted.status().message().empty();
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_EQ(failure_code, absl::StatusCode::kAborted);
  EXPECT_TRUE(empty_message);
  EXPECT_EQ(rhs, original_rhs);
  EXPECT_EQ(scheduler_allocations, 0u);
  EXPECT_EQ(scheduler_frees, 0u);
}

TEST(LiquidSubstep3D, PositiveViscosityAttemptsAllocateAndFreeNothing) {
  const auto fixture = viscous_tank_test::FractionalFill();
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  constexpr double duration = viscous_tank_test::kStepDuration;
  for (auto failure : {absl::StatusCode::kOk, absl::StatusCode::kAborted, absl::StatusCode::kFailedPrecondition}) {
    auto scheduler = LiquidSubstepScheduler3D::Create(fixture->initial, fixture->scene, fixture->options);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(duration).ok());
    scheduler->SetStageFaultHookForTesting({&failure, [](void* context, LiquidPhysicalStage3D stage, double) {
                                              if (stage != LiquidPhysicalStage3D::kAccept) {
                                                return absl::OkStatus();
                                              }
                                              return absl::Status(*static_cast<absl::StatusCode*>(context), {});
                                            }});
    scheduler_allocations = 0;
    scheduler_frees = 0;
    bool succeeded = false;
    auto disposition = LiquidAttemptDisposition3D::kFatal;
    scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
    {
      const auto result = scheduler->Attempt(duration);
      succeeded = result.ok();
      if (succeeded) {
        disposition = result->disposition;
      }
    }
    _CrtSetAllocHook(scheduler_previous_hook);
    ASSERT_TRUE(succeeded);
    EXPECT_EQ(scheduler_allocations, 0u);
    EXPECT_EQ(scheduler_frees, 0u);
    if (failure == absl::StatusCode::kOk) {
      EXPECT_EQ(disposition, LiquidAttemptDisposition3D::kAccepted);
      EXPECT_DOUBLE_EQ(scheduler->committed().clock.time, duration);
    } else {
      const auto expected = failure == absl::StatusCode::kAborted ? LiquidAttemptDisposition3D::kRetryDt
                                                                  : LiquidAttemptDisposition3D::kFatal;
      EXPECT_EQ(disposition, expected);
      ExpectIdenticalSerializedState(fixture->initial, scheduler->committed());
    }
  }
}

TEST(LiquidSubstep3D, WettedPistonAttemptsAllocateAndFreeNothing) {
  namespace wet = wetted_piston_pressure_test;
  for (const bool overflow : {false, true}) {
    const double speed = overflow ? wet::kOverflowPistonSpeed : wet::kPistonSpeed;
    const double duration = overflow ? wet::kOverflowStepDuration : wet::kDuration;
    auto scheduler = LiquidSubstepScheduler3D::Create({}, wet::PlaneScene(1.125, speed), Options());
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    ASSERT_TRUE(scheduler->BeginFrame(duration).ok());
    scheduler_allocations = 0;
    scheduler_frees = 0;
    bool completed = false;
    auto disposition = LiquidAttemptDisposition3D::kFatal;
    scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
    {
      const auto result = scheduler->Attempt(duration);
      completed = result.ok();
      if (completed) {
        disposition = result->disposition;
      }
    }
    _CrtSetAllocHook(scheduler_previous_hook);
    ASSERT_TRUE(completed);
    EXPECT_EQ(disposition, overflow ? LiquidAttemptDisposition3D::kFatal : LiquidAttemptDisposition3D::kAccepted);
    EXPECT_EQ(scheduler_allocations, 0);
    EXPECT_EQ(scheduler_frees, 0);
  }
}

TEST(LiquidSubstep3D, ActualSuccessfulAndRejectedAttemptsAllocateAndFreeNothing) {
  for (const auto failure :
       {LiquidPhysicalStage3D::kCount, LiquidPhysicalStage3D::kPressure, LiquidPhysicalStage3D::kAccept}) {
    auto initial = Initial();
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
    ASSERT_TRUE(scheduler.ok());
    ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
    ASSERT_TRUE(scheduler->BeginFrame(0.02).ok());
    auto stage = failure;
    scheduler->SetStageFaultHookForTesting({&stage, [](void* c, LiquidPhysicalStage3D current, double) {
                                              return current == *static_cast<LiquidPhysicalStage3D*>(c)
                                                  ? absl::Status(absl::StatusCode::kAborted, {})
                                                  : absl::OkStatus();
                                            }});
    scheduler_allocations = 0;
    scheduler_frees = 0;
    bool succeeded = false;
    LiquidAttemptDisposition3D disposition = LiquidAttemptDisposition3D::kFatal;
    scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
    {
      auto result = scheduler->Attempt(0.01);
      succeeded = result.ok();
      if (succeeded) {
        disposition = result->disposition;
      }
    }
    _CrtSetAllocHook(scheduler_previous_hook);
    EXPECT_TRUE(succeeded);
    EXPECT_EQ(
        disposition,
        failure == LiquidPhysicalStage3D::kCount ? LiquidAttemptDisposition3D::kAccepted
                                                 : LiquidAttemptDisposition3D::kRetryDt
    );
    EXPECT_EQ(scheduler_allocations, 0);
    EXPECT_EQ(scheduler_frees, 0);
  }
}
TEST(LiquidSubstep3D, PartialSchedulerReservationFailureIsCheckedAndCleansUp) {
  auto initial = Initial();
  auto scene = Scene();
  auto options = Options();
  scheduler_allocations = 0;
  scheduler_frees = 0;
  scheduler_fail_allocation = 4;
  absl::StatusCode code = absl::StatusCode::kOk;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
    code = scheduler.status().code();
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  scheduler_fail_allocation = 0;
  EXPECT_EQ(code, absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(scheduler_allocations, 4);
  EXPECT_EQ(scheduler_frees, 3);
}
#endif
TEST(LiquidSubstep3D, CapacityWithoutReplayReservationReportsResourceExhausted) {
  auto initial = Initial();
  auto options = Options();
  options.capacities.blocks = 0;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE(scheduler->committed() == initial);
  EXPECT_EQ(scheduler->diagnostics().capacity_replays, 0);
}
TEST(LiquidSubstep3D, OwnsResolvedCavityAcrossAcceptedSteps) {
  const auto shell = cavity_test::ParticleShell();
  ASSERT_TRUE(shell.ok()) << shell.status();
  const auto& initial = *shell;
  int initializations = 0;
  LiquidSceneInputs3D scene;
  scene.field = LiquidParticleSphereUnion3D{.radius = 0.9};
  scene.world_acceleration = {};
  scene.cavity = LiquidSceneCavity3D{
      .atmospheric_absolute_pressure = 100000.0,
      .context = &initializations,
      .initialize = [](void* context,
                       const LiquidAirComponent3D& component) -> absl::StatusOr<LiquidCavityInventory3D> {
        ++*static_cast<int*>(context);
        return InitializeSubmergedLiquidCavity3D(component.gas_volume, 100000.0, 0.0, 0.0);
      },
  };
  const auto options = cavity_test::ShellOptions(initial.particles.size());
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  int pressure_attempts = 0;
  scheduler->SetStageFaultHookForTesting({
      .context = &pressure_attempts,
      .before_stage =
          [](void* context, LiquidPhysicalStage3D stage, double) {
            if (stage == LiquidPhysicalStage3D::kPressure && ++*static_cast<int*>(context) == 2) {
              return absl::Status(absl::StatusCode::kResourceExhausted, {});
            }
            return absl::OkStatus();
          },
  });
  for (const double target : {0.002, 0.003}) {
    const auto status = scheduler->AdvanceFrame(target);
    ASSERT_TRUE(status.ok()) << status << " stage=" << static_cast<int>(scheduler->diagnostics().failed_stage);
    const auto& measurement = scheduler->diagnostics().accepted_measurement;
    EXPECT_EQ(scheduler->diagnostics().capacity_replays, target == 0.002 ? 1 : 0);
    ASSERT_TRUE(measurement && measurement->cavity);
    EXPECT_EQ(measurement->cavity->boundaries.active_cavity_count, 1);
    EXPECT_GT(measurement->cavity->boundaries.air_pressure_count, 0);
    EXPECT_EQ(measurement->cavity->lineage.next_id, 2);
    EXPECT_DOUBLE_EQ(measurement->cavity->lineage.unmatched_inventory, 0.0);
  }
  EXPECT_EQ(initializations, 1);

  LiquidResolvedSceneView3D resolved;
  resolved.scene = scene;
  resolved.scene.cavity = LiquidSceneCavity3D{
      .atmospheric_absolute_pressure = 100000.0,
      .birth_hydrostatic_gauge_pressure = 100.0,
  };
  resolved.options = options;
  const auto encoded = EncodeLiquidResolvedScene3D(resolved);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  auto loaded = LiquidSubstepScheduler3D::Create(initial, decoded->scene, options);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const auto loaded_step = loaded->AdvanceFrame(0.001);
  ASSERT_TRUE(loaded_step.ok()) << loaded_step;
  const auto& loaded_measurement = loaded->diagnostics().accepted_measurement;
  ASSERT_TRUE(loaded_measurement && loaded_measurement->cavity);
  EXPECT_EQ(loaded_measurement->cavity->boundaries.active_cavity_count, 1);
  EXPECT_GT(loaded_measurement->pressure_rhs_l2_m3_per_s, 0.0);

  ASSERT_EQ(loaded->committed().next_cavity_id, 2);
  ASSERT_GT(loaded->committed().cavity_gas.size(), 0);
  LiquidResumeMetadata3D resume;
  resume.resolved_fingerprint = *ResolvedSceneFingerprint(decoded->View());
  resume.execution_id = LiquidExecutionId3D(resume.resolved_fingerprint);
  resume.capacities = options.capacities;
  resume.capacity_replay_reservations = options.capacity_replay_reservations;
  const auto checkpoint = EncodeLiquidCheckpoint3D(loaded->committed(), resume);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const auto restored = DecodeLiquidCheckpoint3D(checkpoint->values());
  ASSERT_TRUE(restored.ok()) << restored.status();
  auto restarted = LiquidSubstepScheduler3D::Create(restored->state, decoded->scene, options);
  ASSERT_TRUE(restarted.ok()) << restarted.status();
  ASSERT_TRUE(loaded->AdvanceFrame(0.002).ok());
  ASSERT_TRUE(restarted->AdvanceFrame(0.002).ok());
  EXPECT_EQ(restarted->committed().next_cavity_id, 2);
  ExpectIdenticalSerializedState(loaded->committed(), restarted->committed());
}

TEST(LiquidSubstep3D, PreparedCavityPressuresApplyOnlyAfterRadialRetry) {
  struct CavityFixture {
    std::array<LiquidAirCellPressure3D, 256> pressures;
    int calls = 0;
  } fixture;
  auto initial = Initial();
  auto scene = Scene();
  scene.world_acceleration = {};
  scene.cavity_provider = {
      .context = &fixture,
      .prepare = [](void* context, double step_start_time, const LiquidBoundaryGeometryViewInputs3D& samples
                 ) -> absl::StatusOr<LiquidStepCavityBoundaries3D> {
        auto& cavity = *static_cast<CavityFixture*>(context);
        ++cavity.calls;
        EXPECT_DOUBLE_EQ(step_start_time, 0.0);
        std::size_t count = 0;
        for (const auto& center : samples.center_distances) {
          if (center.signed_distance >= 0.0) {
            if (count == cavity.pressures.size()) {
              return absl::ResourceExhaustedError("cavity fixture capacity");
            }
            cavity.pressures[count++] = {center.cell_index, 17.0};
          }
        }
        return LiquidStepCavityBoundaries3D{
            .air_pressures = std::span(cavity.pressures).first(count),
            .maximum_step_duration = 0.006,
        };
      },
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto rejected = scheduler->Attempt(0.01);
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_EQ(rejected->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ASSERT_TRUE(rejected->retry_duration);
  EXPECT_TRUE(scheduler->committed() == initial);
  EXPECT_FALSE(scheduler->diagnostics().accepted_measurement);

  const auto accepted = scheduler->Attempt(*rejected->retry_duration);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  EXPECT_EQ(accepted->disposition, LiquidAttemptDisposition3D::kAccepted);
  EXPECT_EQ(fixture.calls, 2);
  const auto& measurement = scheduler->diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement && measurement->step_start_cavity_bound_seconds);
  EXPECT_DOUBLE_EQ(*measurement->step_start_cavity_bound_seconds, 0.006);
  ASSERT_GT(scheduler->committed().cells.size(), 0);
  EXPECT_NEAR(scheduler->committed().cells[0].previous_pressure, 17.0, 1e-7);
}

TEST(LiquidSubstep3D, ExplicitCapillaryOversizeRequestsSmallerDuration) {
  auto initial = Initial();
  auto scene = Scene();
  scene.surface_tension = 1e8;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kRetryDt);
  ASSERT_TRUE(result->retry_duration.has_value());
  EXPECT_DOUBLE_EQ(*result->retry_duration, 0.005);
  EXPECT_TRUE(scheduler->committed() == initial);
}
TEST(LiquidSubstep3D, ActualSealedIncompatiblePressureIsFatalWithoutRetry) {
  auto initial = Initial();
  auto scene = Scene();
  scene.target_divergence = 1;
  // This backend qualification fixture closes every frozen face. The real bounded
  // pressure assembly must reject its physical source, rather than retry the solver.
  std::array<MacPressureFace3D, 128> sealed_faces{};
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &sealed_faces;
  backend.project = [](void* context,
                       SparseMacGridView3D grid,
                       double dt,
                       const LiquidPressureProblemView3D& problem,
                       MacPressureSolverOptions3D options,
                       MacPressureWorkspace3D& workspace) -> absl::StatusOr<MacPressureProjectionView3D> {
    auto& faces = *static_cast<std::array<MacPressureFace3D, 128>*>(context);
    if (problem.faces.size() > faces.size()) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    std::ranges::copy(problem.faces, faces.begin());
    for (auto& face : std::span(faces).first(problem.faces.size())) {
      face.open_area = 0;
      face.pressure_link = SealedMacPressureLink3D{};
    }
    return ProjectMacPressure3D(
        grid,
        dt,
        {problem.cells, std::span(faces).first(problem.faces.size())},
        options,
        workspace
    );
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(scheduler->diagnostics().attempts, 1);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  EXPECT_TRUE(scheduler->committed() == initial);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}
TEST(LiquidSubstep3D, CandidateValidationRejectsPoisonAfterActualNumericalUpdate) {
  auto initial = Initial();
  std::size_t poison_injection_count = 0;
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &poison_injection_count;
  backend.update_particles =
      [](void* context, const LiquidParticleUpdateInputs3D& inputs, std::span<LiquidParticleState3D> particles) {
        const auto status = CpuLiquidSubstepBackend3D().update_particles(nullptr, inputs, particles);
        if (status.ok()) {
          particles[0].volume_ratio = std::numeric_limits<double>::quiet_NaN();
          ++*static_cast<std::size_t*>(context);
        }
        return status;
      };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(poison_injection_count, 1);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kValidate);
  EXPECT_TRUE(scheduler->committed() == initial);
  ExpectIdenticalSerializedState(initial, scheduler->committed());
}
TEST(LiquidSubstep3D, StableSignedBinsAndZeroWeightOwnersSurviveIntegratedStep) {
  auto initial = Initial(3);
  initial.particles[0].world_position = {6, 6, 6};
  initial.particles[1].world_position = {-2, -2, -2};
  initial.particles[2].world_position = {-2, -2, -2};
  auto scene = Scene();
  scene.field = LiquidParticleSphereUnion3D{0.9};
  scene.world_acceleration = {};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  const auto& state = scheduler->committed();
  EXPECT_EQ(state.particles[0].id, 2);
  EXPECT_EQ(state.particles[1].id, 3);
  EXPECT_EQ(state.particles[2].id, 1);
  auto workspace = LiquidStepWorkspace3D::Create(Options().capacities, state.geometry);
  ASSERT_TRUE(workspace.ok());
  ASSERT_TRUE(workspace->PrepareTrialTopology(state.particles.values(), {}, &state).ok());
  auto topology = workspace->prepared();
  ASSERT_TRUE(topology.ok());
  for (const auto& particle : state.particles.values()) {
    for (auto axis : {FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ}) {
      auto stencil = BuildOwnedMacFaceStencil3D(topology->grid, axis, particle.world_position, particle.world_position);
      ASSERT_TRUE(stencil.ok());
      EXPECT_EQ(stencil->size(), 27);
    }
  }
}
#if defined(_WIN32) && defined(_DEBUG)
TEST(LiquidSubstep3D, ActualDepositRejectionAllocatesAndFreesNothing) {
  auto initial = Initial();
  initial.particles[0].mass = 1e100;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  scheduler_allocations = 0;
  scheduler_frees = 0;
  bool fatal = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    auto result = scheduler->Attempt(0.01);
    fatal = result.ok() && result->disposition == LiquidAttemptDisposition3D::kFatal;
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_TRUE(fatal);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kDeposit);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kOutOfRange);
  EXPECT_EQ(scheduler_allocations, 0);
  EXPECT_EQ(scheduler_frees, 0);
  EXPECT_TRUE(scheduler->committed() == initial);
}
TEST(LiquidSubstep3D, ActualPressureNonconvergenceAllocatesAndFreesNothing) {
  auto initial = Initial(8);
  for (std::size_t i = 0; i < 8; ++i) {
    auto& p = initial.particles[i];
    p.world_position = {1.5 + 2 * (i & 1), 1.5 + 2 * ((i >> 1) & 1), 1.5 + 2 * ((i >> 2) & 1)};
    p.world_velocity = {
        0.1 * static_cast<double>(i),
        0.03 * static_cast<double>(i * i),
        -0.07 * static_cast<double>(i),
    };
  }
  auto scene = Scene();
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 2.1};
  auto options = Options();
  options.pressure_solver_options.max_iterations = 1;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  scheduler_allocations = 0;
  scheduler_frees = 0;
  bool retry = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    auto result = scheduler->Attempt(0.01);
    retry = result.ok() && result->retry_duration.has_value();
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_TRUE(retry);
  EXPECT_EQ(scheduler->diagnostics().failed_stage, LiquidPhysicalStage3D::kPressure);
  EXPECT_EQ(scheduler->diagnostics().failure_code, absl::StatusCode::kAborted);
  EXPECT_EQ(scheduler_allocations, 0);
  EXPECT_EQ(scheduler_frees, 0);
  EXPECT_TRUE(scheduler->committed() == initial);
}
#endif
TEST(LiquidSubstep3D, BoundedParticleCapacityCompleteStepMeasurement) {
  auto initial = Initial(1024);
  auto options = Options();
  options.capacities.particles = 1024;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options);
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto begin = std::chrono::steady_clock::now();
  const auto result = scheduler->Attempt(0.01);
  const auto end = std::chrono::steady_clock::now();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kAccepted);
  EXPECT_EQ(scheduler->committed().particles.size(), 1024);
  RecordProperty(
      "complete_attempt_microseconds",
      static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())
  );
  RecordProperty("reserved_payload_bytes", std::to_string(scheduler->reserved_bytes()));
  RecordProperty("particle_count", 1024);
}
TEST(LiquidSubstep3D, ExplicitWallMotionAboveHalfCellRequestsRetry) {
  auto initial = Initial();
  auto scene = Scene();
  scene.obstacle = TranslatingHalfspace3D{{1, 0, 0}, -100, {100, 0, 0}};
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, Options());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kRetryDt);
  EXPECT_TRUE(scheduler->committed() == initial);
}
#if defined(_WIN32) && defined(_DEBUG)
TEST(LiquidSubstep3D, ReportedReservationMatchesActualRequestedPayload) {
  auto initial = Initial();
  auto scene = Scene();
  auto options = Options();
  scheduler_allocations = 0;
  scheduler_frees = 0;
  scheduler_requested_bytes = 0;
  scheduler_last_request = 0;
  std::size_t reported = 0;
  std::size_t owner_bytes = 0;
  bool created = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options);
    created = scheduler.ok();
    if (created) {
      reported = scheduler->reserved_bytes();
      owner_bytes = scheduler_last_request;
    }
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_TRUE(created);
  EXPECT_EQ(scheduler_requested_bytes, reported + owner_bytes);
  EXPECT_EQ(scheduler_allocations, scheduler_frees);
  RecordProperty("reserved_payload_bytes", std::to_string(reported));
  RecordProperty("owner_object_bytes", std::to_string(owner_bytes));
}
#endif

struct BackendLifecycleObservation {
  int initialized = 0;
  int prepared = 0;
  int committed = 0;
  int discarded = 0;
  int growth_prepared = 0;
  int growth_committed = 0;
  int growth_discarded = 0;
  int samples = 0;
  int face_samples = 0;
  bool fail_publication = false;
  bool fail_growth = false;
  LiquidClock3D staged_clock{};
  LiquidClock3D growth_clock{};
};
LiquidSubstepBackend3D ObservedBackend(BackendLifecycleObservation& observed) {
  auto backend = CpuLiquidSubstepBackend3D();
  backend.context = &observed;
  backend.initialize = [](void* context, const LiquidState3D&) {
    ++static_cast<BackendLifecycleObservation*>(context)->initialized;
    return absl::OkStatus();
  };
  backend.prepare_publication = [](void* context, const LiquidState3D&, LiquidClock3D clock) {
    auto& observation = *static_cast<BackendLifecycleObservation*>(context);
    ++observation.prepared;
    observation.staged_clock = clock;
    return absl::Status(observation.fail_publication ? absl::StatusCode::kInternal : absl::StatusCode::kOk, {});
  };
  backend.commit_publication = [](void* context) noexcept {
    ++static_cast<BackendLifecycleObservation*>(context)->committed;
  };
  backend.discard_publication = [](void* context) noexcept {
    ++static_cast<BackendLifecycleObservation*>(context)->discarded;
  };
  backend.prepare_growth = [](void* context, LiquidStateCapacities3D, const LiquidState3D& start) {
    auto& observation = *static_cast<BackendLifecycleObservation*>(context);
    ++observation.growth_prepared;
    observation.growth_clock = start.clock;
    return absl::Status(observation.fail_growth ? absl::StatusCode::kResourceExhausted : absl::StatusCode::kOk, {});
  };
  backend.commit_growth = [](void* context) noexcept {
    ++static_cast<BackendLifecycleObservation*>(context)->growth_committed;
  };
  backend.discard_growth = [](void* context) noexcept {
    ++static_cast<BackendLifecycleObservation*>(context)->growth_discarded;
  };
  backend.sample_field =
      [](void* context, const LiquidSceneField3D& field, std::span<const LiquidParticleState3D> particles, Vec3 position
      ) {
        ++static_cast<BackendLifecycleObservation*>(context)->samples;
        return CpuLiquidSubstepBackend3D().sample_field(nullptr, field, particles, position);
      };
  backend.sample_faces = [](void* context,
                            SparseMacGridView3D grid,
                            const LiquidObstacle3D& obstacle,
                            std::int64_t side,
                            std::span<LiquidBoundaryFaceTag3D> tags,
                            std::span<LiquidPrescribedWallSample3D> walls) {
    ++static_cast<BackendLifecycleObservation*>(context)->face_samples;
    return CpuLiquidSubstepBackend3D().sample_faces(nullptr, grid, obstacle, side, tags, walls);
  };
  return backend;
}
TEST(LiquidSubstep3D, BackendStagesExactAcceptedClockAndSamplesAllFields) {
  auto initial = Initial();
  auto options = Options();
  options.maximum_step_duration = 0.005;
  BackendLifecycleObservation observed;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options, ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(observed.initialized, 1);
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  for (int step = 0; step < 2; ++step) {
    const auto result = scheduler->Attempt(0.005);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kAccepted);
    EXPECT_EQ(observed.staged_clock, scheduler->committed().clock);
    EXPECT_EQ(observed.prepared, step + 1);
    EXPECT_EQ(observed.committed, step + 1);
  }
  EXPECT_EQ(observed.discarded, 0);
  EXPECT_GT(observed.samples, 1);
  EXPECT_GT(observed.face_samples, 1);
}
TEST(LiquidSubstep3D, BackendPublicationFailurePreservesCompleteCommittedState) {
  auto initial = Initial();
  BackendLifecycleObservation observed;
  observed.fail_publication = true;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  const auto result = scheduler->Attempt(0.01);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->disposition, LiquidAttemptDisposition3D::kFatal);
  EXPECT_EQ(observed.prepared, 1);
  EXPECT_EQ(observed.committed, 0);
  EXPECT_EQ(observed.discarded, 1);
  EXPECT_TRUE(scheduler->committed() == initial);
}
TEST(LiquidSubstep3D, BackendGrowthPreparationFailureKeepsLastAcceptedVersion) {
  auto initial = Initial();
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  BackendLifecycleObservation observed;
  observed.fail_growth = true;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options, ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kEmit
                                                    && ++*static_cast<int*>(context) == 2
                                                ? absl::Status(absl::StatusCode::kResourceExhausted, {})
                                                : absl::OkStatus();
                                          }});
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
  EXPECT_EQ(scheduler->committed().clock.time, 0.005);
  EXPECT_EQ(observed.growth_prepared, 1);
  EXPECT_EQ(observed.growth_committed, 0);
  EXPECT_EQ(observed.growth_discarded, 1);
  EXPECT_EQ(observed.growth_clock, initial.clock);
}
TEST(LiquidSubstep3D, BackendGrowthRestoresFrameStartBeforeReplay) {
  auto initial = Initial();
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  BackendLifecycleObservation observed;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options, ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kEmit
                                                    && ++*static_cast<int*>(context) == 2
                                                ? absl::Status(absl::StatusCode::kResourceExhausted, {})
                                                : absl::OkStatus();
                                          }});
  ASSERT_TRUE(scheduler->AdvanceFrame(0.01).ok());
  EXPECT_EQ(observed.growth_prepared, 1);
  EXPECT_EQ(observed.growth_committed, 1);
  EXPECT_EQ(observed.growth_discarded, 0);
  EXPECT_EQ(observed.growth_clock, initial.clock);
  EXPECT_EQ(observed.committed, 3);
  EXPECT_EQ(observed.staged_clock, scheduler->committed().clock);
}
TEST(LiquidSubstep3D, BackendSamplerFailureAndMissingLifecycleNeverFallBack) {
  auto initial = Initial();
  auto backend = CpuLiquidSubstepBackend3D();
  backend.sample_field = [](void*, const LiquidSceneField3D&, std::span<const LiquidParticleState3D>, Vec3
                         ) -> absl::StatusOr<double> {
    return absl::Status(absl::StatusCode::kInternal, {});
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), backend);
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler->AdvanceFrame(0.01).code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(scheduler->committed() == initial);
  backend = CpuLiquidSubstepBackend3D();
  backend.commit_growth = nullptr;
  EXPECT_EQ(
      LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), backend).status().code(),
      absl::StatusCode::kInvalidArgument
  );
}
#if defined(_WIN32) && defined(_DEBUG)
TEST(LiquidSubstep3D, BackendGrowthDiscardRunsAfterActualHostReservationFailure) {
  auto initial = Initial();
  auto options = Options();
  options.maximum_step_duration = 0.005;
  options.capacity_replay_reservations = options.capacities;
  options.capacity_replay_reservations.particles = 32;
  BackendLifecycleObservation observed;
  auto backend = ObservedBackend(observed);
  backend.prepare_growth = [](void* context, LiquidStateCapacities3D, const LiquidState3D& start) {
    auto& observation = *static_cast<BackendLifecycleObservation*>(context);
    ++observation.growth_prepared;
    observation.growth_clock = start.clock;
    scheduler_allocations = 0;
    scheduler_fail_allocation = 1;
    scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
    return absl::OkStatus();
  };
  backend.discard_growth = [](void* context) noexcept {
    _CrtSetAllocHook(scheduler_previous_hook);
    scheduler_fail_allocation = 0;
    ++static_cast<BackendLifecycleObservation*>(context)->growth_discarded;
  };
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), options, backend);
  ASSERT_TRUE(scheduler.ok());
  int emits = 0;
  scheduler->SetStageFaultHookForTesting({&emits, [](void* context, LiquidPhysicalStage3D stage, double) {
                                            return stage == LiquidPhysicalStage3D::kEmit
                                                    && ++*static_cast<int*>(context) == 2
                                                ? absl::Status(absl::StatusCode::kResourceExhausted, {})
                                                : absl::OkStatus();
                                          }});
  const auto result = scheduler->AdvanceFrame(0.01);
  _CrtSetAllocHook(scheduler_previous_hook);
  scheduler_fail_allocation = 0;
  EXPECT_EQ(result.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(scheduler_allocations, 1);
  EXPECT_EQ(observed.growth_prepared, 1);
  EXPECT_EQ(observed.growth_committed, 0);
  EXPECT_EQ(observed.growth_discarded, 1);
  EXPECT_EQ(observed.growth_clock, initial.clock);
  EXPECT_EQ(scheduler->committed().clock.time, 0.005);
  EXPECT_EQ(scheduler->committed().clock.epoch, 1);
}
TEST(LiquidSubstep3D, BackendPublicationFailureAllocatesAndFreesNothing) {
  auto initial = Initial();
  BackendLifecycleObservation observed;
  observed.fail_publication = true;
  auto scheduler = LiquidSubstepScheduler3D::Create(initial, Scene(), Options(), ObservedBackend(observed));
  ASSERT_TRUE(scheduler.ok());
  ASSERT_TRUE(scheduler->BeginFrame(0.01).ok());
  scheduler_allocations = 0;
  scheduler_frees = 0;
  bool rejected = false;
  scheduler_previous_hook = _CrtSetAllocHook(SchedulerAllocationHook);
  {
    const auto result = scheduler->Attempt(0.01);
    rejected = result.ok() && result->disposition == LiquidAttemptDisposition3D::kFatal;
  }
  _CrtSetAllocHook(scheduler_previous_hook);
  EXPECT_TRUE(rejected);
  EXPECT_EQ(scheduler_allocations, 0);
  EXPECT_EQ(scheduler_frees, 0);
  EXPECT_EQ(observed.discarded, 1);
  EXPECT_EQ(observed.committed, 0);
  EXPECT_TRUE(scheduler->committed() == initial);
}
#endif
}  // namespace
}  // namespace fluidity
