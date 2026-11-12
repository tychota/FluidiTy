#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "../cuda/liquid_boundary_transfer_3d_internal.h"
#include "../cuda/liquid_geometry_stages_3d.h"
#include "fixture_data.h"
#include "fluidity/host/cuda_liquid_boundary_3d.h"
#include "fluidity/host/cuda_mac_pressure_3d.h"
#include "gtest/gtest.h"
#include "liquid_boundary_geometry_3d_cases.h"
#include "liquid_boundary_transfer_3d_cases.h"
#include "liquid_open_tank_lattice_3d_cases.h"
#include "liquid_wall_contact_3d_cases.h"
namespace fluidity {
namespace {
void Hardware() {
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  EXPECT_EQ(properties.major, 12);
  EXPECT_EQ(properties.minor, 0);
#if defined(FLUIDITY_QUALIFIED_CUDA_DEVICE)
  EXPECT_STREQ(properties.name, FLUIDITY_QUALIFIED_CUDA_DEVICE);
#endif
  std::cout << "CTEST_FULL_OUTPUT\nboundary_device=" << properties.name << '\n';
}
void Receipt(const char* name, CudaOperatorMeasurement measurement) {
  EXPECT_EQ(measurement.compiled_architecture, 1200);
  EXPECT_TRUE(std::isfinite(measurement.kernel_ms));
  EXPECT_GE(measurement.kernel_ms, 0);
  std::cout << "operator=" << name << " compiled_architecture=" << measurement.compiled_architecture
            << " kernel_ms=" << std::fixed << std::setprecision(9) << measurement.kernel_ms << '\n';
}
TEST(CudaLiquidBoundary3D, OpenTankGeometryAndPressureMatchOracle) {
  Hardware();
  CudaOperatorMeasurement measurement{};
  const auto record = [&](CudaOperatorMeasurement sample) {
    EXPECT_EQ(sample.compiled_architecture, 1200);
    EXPECT_TRUE(std::isfinite(sample.kernel_ms));
    EXPECT_GE(sample.kernel_ms, 0);
    measurement.compiled_architecture = sample.compiled_architecture;
    measurement.kernel_ms += sample.kernel_ms;
  };
  boundary_test::GeometryCases cases(
      [&](const LiquidBoundaryGeometryInputs3D& inputs) -> absl::StatusOr<FrozenLiquidBoundaryGeometry3D> {
        const auto cpu = FreezeLiquidBoundaryGeometry3D(inputs);
        auto gpu = CudaFreezeLiquidBoundaryGeometry3D(inputs);
        EXPECT_EQ(cpu.status().code(), gpu.status().code());
        if (!gpu.ok()) {
          return gpu.status();
        }
        record(gpu->measurement);
        if (cpu.ok() && cpu->cells().size() == gpu->geometry.cells().size()) {
          for (std::size_t cell = 0; cell < cpu->cells().size(); ++cell) {
            EXPECT_NEAR(
                cpu->cells()[cell].fluid_volume,
                gpu->geometry.cells()[cell].fluid_volume,
                2e-13 * cpu->cells()[cell].fluid_volume
            );
          }
        }
        if (cpu.ok() && cpu->faces().size() == gpu->geometry.faces().size()) {
          for (std::size_t face = 0; face < cpu->faces().size(); ++face) {
            EXPECT_EQ(cpu->faces()[face].face_slot, gpu->geometry.faces()[face].face_slot);
            EXPECT_EQ(cpu->faces()[face].prescribed_wall_flux, gpu->geometry.faces()[face].prescribed_wall_flux);
            EXPECT_DOUBLE_EQ(cpu->faces()[face].open_area, gpu->geometry.faces()[face].open_area);
          }
        }
        return std::move(gpu->geometry);
      },
      [&](const FrozenLiquidBoundaryGeometry3D& geometry,
          const LiquidPressureFields3D& fields) -> absl::StatusOr<LiquidPressureProblem3D> {
        auto gpu = CudaBuildLiquidPressureLinks3D(geometry, fields);
        if (!gpu.ok()) {
          return gpu.status();
        }
        record(gpu->measurement);
        return std::move(gpu->problem);
      },
      [&](const MacPressureInputs3D& inputs) -> absl::StatusOr<MacPressureProjection3D> {
        auto gpu = CudaProjectMacPressure3D(inputs);
        if (!gpu.ok()) {
          return gpu.status();
        }
        record(gpu->measurement);
        return std::move(gpu->projection);
      }
  );
  cases.OpenTankClipsVolumeAndUsesPistonOnlyFlux();
  cases.OpenTankPreservesNarrowAndLiquidClippedMeasures();
  auto arena = CudaLiquidArena3D::Create(boundary_test::kTranslatedTankCapacities);
  ASSERT_TRUE(arena.ok()) << arena.status();
  const auto freeze = [&](const LiquidBoundaryGeometryViewInputs3D& inputs, LiquidBoundaryWorkspace3D& workspace) {
    return cuda_internal::FreezeLiquidGeometry3D(*arena, inputs, workspace);
  };
  boundary_test::ExpectOpenTankRejectsUnresolvedTranslatedLattice(freeze);
  boundary_test::ExpectOpenTankPreservesLatticeResiduals(freeze);
  Receipt("liquid_open_tank_geometry_3d", measurement);
}

TEST(CudaLiquidBoundary3D, OpenTankSweptContactMatchesOracle) {
  Hardware();
  CudaOperatorMeasurement measurement{};
  boundary_test::ContactCases cases(
      [&](const LiquidWallContactInputs3D& inputs) -> absl::StatusOr<LiquidWallContactResult3D> {
        const auto cpu = ResolveLiquidWallContact3D(inputs);
        const auto gpu = CudaResolveLiquidWallContact3D(inputs);
        EXPECT_EQ(cpu.status().code(), gpu.status().code());
        if (!gpu.ok()) {
          return gpu.status();
        }
        EXPECT_EQ(gpu->measurement.compiled_architecture, 1200);
        EXPECT_TRUE(std::isfinite(gpu->measurement.kernel_ms));
        EXPECT_GE(gpu->measurement.kernel_ms, 0);
        measurement.compiled_architecture = gpu->measurement.compiled_architecture;
        measurement.kernel_ms += gpu->measurement.kernel_ms;
        if (cpu.ok()) {
          boundary_transfer_test::ExpectVec(gpu->contact.world_position, cpu->world_position, 0);
          boundary_transfer_test::ExpectVec(gpu->contact.stored_world_velocity, cpu->stored_world_velocity, 0);
          boundary_transfer_test::ExpectVec(gpu->contact.advecting_world_velocity, cpu->advecting_world_velocity, 0);
          EXPECT_EQ(gpu->contact.contact_fraction, cpu->contact_fraction);
        }
        return gpu->contact;
      }
  );
  cases.OpenTankSweepsAndConstrainsBothVelocities();
  cases.OpenTankRejectsInvalidFrictionExtentsAndEndpointWidth();
  Receipt("liquid_open_tank_contact_3d", measurement);
}

TEST(CudaLiquidBoundary3D, GeometryAndCapillaryMatchOracle) {
  Hardware();
  const std::array<BlockIndex3D, 1> blocks{{{0, 0, 0}}};
  const auto grid = SparseMacGrid3D::Create({{0, 0, 0}, 1}, blocks, 1);
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array<CellIndex3D, 1> candidates{{{1, 1, 1}}};
  std::vector<LiquidCenterDistance3D> centers;
  std::vector<LiquidCornerDistance3D> corners;
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        centers.push_back({{x, y, z}, -1});
        corners.push_back({{x, y, z}, -1});
      }
    }
  }
  const auto all = fixtures::AllFixtures();
  const auto fixture = std::ranges::find(all, "uniform_mac_liquid_boundary_3d", &fixtures::FixtureRecord::name);
  ASSERT_NE(fixture, all.end());
  const auto offsets = fixtures::FindArray(*fixture, "unit_cell_plane_offsets");
  const auto volumes = fixtures::FindArray(*fixture, "expected_open_cell_volumes");
  ASSERT_TRUE(offsets.ok());
  ASSERT_TRUE(volumes.ok());
  CudaOperatorMeasurement measurement{};
  for (int axis = 0; axis < 3; ++axis) {
    const Vec3 normal{axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0, axis == 2 ? 1.0 : 0.0};
    for (std::size_t row = 0; row < (*offsets)->f64_data.size(); ++row) {
      const LiquidBoundaryGeometryInputs3D inputs{
          *grid,
          candidates,
          centers,
          corners,
          TranslatingHalfspace3D{normal, 1 + (*offsets)->f64_data[row], {2, 3, 4}},
          .01,
          1000,
          0,
      };
      const auto cpu = FreezeLiquidBoundaryGeometry3D(inputs);
      const auto gpu = CudaFreezeLiquidBoundaryGeometry3D(inputs);
      ASSERT_TRUE(cpu.ok()) << cpu.status();
      ASSERT_TRUE(gpu.ok()) << gpu.status();
      ASSERT_EQ(gpu->geometry.cells().size(), 1);
      EXPECT_NEAR(gpu->geometry.cells()[0].fluid_volume, (*volumes)->f64_data[row], 2e-14);
      EXPECT_NEAR(gpu->geometry.cells()[0].fluid_volume, cpu->cells()[0].fluid_volume, 2e-14);
      ASSERT_EQ(gpu->geometry.faces().size(), cpu->faces().size());
      for (std::size_t face = 0; face < cpu->faces().size(); ++face) {
        EXPECT_NEAR(gpu->geometry.faces()[face].open_area, cpu->faces()[face].open_area, 2e-14);
        EXPECT_DOUBLE_EQ(gpu->geometry.faces()[face].wall_normal_velocity, cpu->faces()[face].wall_normal_velocity);
      }
      measurement = gpu->measurement;
    }
  }
  double geometry_device_ms = 0;
  double links_device_ms = 0;
  double pressure_device_ms = 0;
  double geometry_wall_ms = 0;
  double links_wall_ms = 0;
  double pressure_wall_ms = 0;
  const auto elapsed_ms = [](auto start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  };
  boundary_test::GeometryCases cases(
      [&](const LiquidBoundaryGeometryInputs3D& input) -> absl::StatusOr<FrozenLiquidBoundaryGeometry3D> {
        const auto cpu = FreezeLiquidBoundaryGeometry3D(input);
        const auto started = std::chrono::steady_clock::now();
        const auto gpu = CudaFreezeLiquidBoundaryGeometry3D(input);
        geometry_wall_ms += elapsed_ms(started);
        EXPECT_EQ(cpu.status().code(), gpu.status().code());
        if (!gpu.ok()) {
          return gpu.status();
        }
        measurement = gpu->measurement;
        geometry_device_ms += measurement.kernel_ms;
        if (input.sampled_solid) {
          const LiquidStateCapacities3D capacities{2, 8, 8, input.mac_grid.face_count(), 0};
          auto arena = CudaLiquidArena3D::Create(capacities);
          auto workspace = LiquidBoundaryWorkspace3D::Create(capacities);
          EXPECT_TRUE(arena.ok()) << arena.status();
          EXPECT_TRUE(workspace.ok()) << workspace.status();
          if (!arena.ok() || !workspace.ok()) {
            return !arena.ok() ? arena.status() : workspace.status();
          }
          const LiquidBoundaryGeometryViewInputs3D view{
              .mac_grid = input.mac_grid.view(),
              .cell_indices = input.cell_indices,
              .center_distances = input.center_distances,
              .corner_distances = input.corner_distances,
              .obstacle = input.obstacle,
              .step_duration = input.step_duration,
              .liquid_density = input.liquid_density,
              .surface_tension = input.surface_tension,
              .sampled_solid = input.sampled_solid,
          };
          const auto batched = cuda_internal::FreezeLiquidGeometry3D(*arena, view, *workspace);
          EXPECT_TRUE(batched.ok()) << batched.status();
          if (!batched.ok()) {
            return batched.status();
          }
          EXPECT_EQ(batched->cells.size(), gpu->geometry.cells().size());
          EXPECT_EQ(batched->faces.size(), gpu->geometry.faces().size());
          for (std::size_t i = 0; i < batched->cells.size() && i < gpu->geometry.cells().size(); ++i) {
            EXPECT_DOUBLE_EQ(batched->cells[i].fluid_volume, gpu->geometry.cells()[i].fluid_volume);
          }
          for (std::size_t i = 0; i < batched->faces.size() && i < gpu->geometry.faces().size(); ++i) {
            EXPECT_EQ(batched->faces[i].face_slot, gpu->geometry.faces()[i].face_slot);
            EXPECT_DOUBLE_EQ(batched->faces[i].open_area, gpu->geometry.faces()[i].open_area);
            EXPECT_DOUBLE_EQ(batched->faces[i].wall_normal_velocity, gpu->geometry.faces()[i].wall_normal_velocity);
          }
        }
        EXPECT_EQ(measurement.compiled_architecture, 1200);
        if (cpu.ok()) {
          EXPECT_EQ(cpu->cells().size(), gpu->geometry.cells().size());
          EXPECT_EQ(cpu->faces().size(), gpu->geometry.faces().size());
          for (std::size_t i = 0; i < cpu->cells().size() && i < gpu->geometry.cells().size(); ++i) {
            EXPECT_NEAR(
                cpu->cells()[i].fluid_volume,
                gpu->geometry.cells()[i].fluid_volume,
                std::abs(cpu->cells()[i].fluid_volume) * 2e-13
            );
            EXPECT_NEAR(
                cpu->cells()[i].curvature,
                gpu->geometry.cells()[i].curvature,
                std::max(1.0, std::abs(cpu->cells()[i].curvature)) * 2e-13
            );
          }
          for (std::size_t i = 0; i < cpu->faces().size() && i < gpu->geometry.faces().size(); ++i) {
            EXPECT_EQ(cpu->faces()[i].face_slot, gpu->geometry.faces()[i].face_slot);
            EXPECT_NEAR(
                cpu->faces()[i].open_area,
                gpu->geometry.faces()[i].open_area,
                std::abs(cpu->faces()[i].open_area) * 2e-13
            );
          }
        }
        return gpu->geometry;
      },
      [&](const FrozenLiquidBoundaryGeometry3D& geometry,
          const LiquidPressureFields3D& fields) -> absl::StatusOr<LiquidPressureProblem3D> {
        const auto cpu = BuildLiquidPressureLinks3D(geometry, fields);
        const auto started = std::chrono::steady_clock::now();
        const auto gpu = CudaBuildLiquidPressureLinks3D(geometry, fields);
        links_wall_ms += elapsed_ms(started);
        EXPECT_EQ(cpu.status().code(), gpu.status().code());
        if (!gpu.ok()) {
          return gpu.status();
        }
        links_device_ms += gpu->measurement.kernel_ms;
        if (cpu.ok()) {
          EXPECT_EQ(cpu->lower_theta_regularization_count, gpu->problem.lower_theta_regularization_count);
          EXPECT_EQ(cpu->faces.size(), gpu->problem.faces.size());
          for (std::size_t i = 0; i < cpu->faces.size() && i < gpu->problem.faces.size(); ++i) {
            EXPECT_DOUBLE_EQ(
                cpu->faces[i].pressure_gradient_distance,
                gpu->problem.faces[i].pressure_gradient_distance
            );
            if (const auto* expected = std::get_if<DirichletMacPressureLink3D>(&cpu->faces[i].pressure_link)) {
              const auto* actual = std::get_if<DirichletMacPressureLink3D>(&gpu->problem.faces[i].pressure_link);
              EXPECT_NE(actual, nullptr);
              if (actual != nullptr) {
                EXPECT_DOUBLE_EQ(expected->exterior_pressure, actual->exterior_pressure);
              }
            }
          }
        }
        return gpu->problem;
      },
      [&](const MacPressureInputs3D& inputs) -> absl::StatusOr<MacPressureProjection3D> {
        const auto expected = ProjectMacPressure3D(inputs);
        const auto started = std::chrono::steady_clock::now();
        const auto actual = CudaProjectMacPressure3D(inputs);
        pressure_wall_ms += elapsed_ms(started);
        EXPECT_EQ(expected.status().code(), actual.status().code());
        if (!actual.ok()) {
          return actual.status();
        }
        EXPECT_EQ(actual->measurement.compiled_architecture, 1200);
        pressure_device_ms += actual->measurement.kernel_ms;
        if (expected.ok()) {
          for (std::size_t i = 0; i < expected->cell_pressures.size(); ++i) {
            EXPECT_NEAR(expected->cell_pressures[i], actual->projection.cell_pressures[i], 1e-9);
          }
          for (std::size_t i = 0; i < expected->corrected_face_velocities.size(); ++i) {
            EXPECT_NEAR(expected->corrected_face_velocities[i], actual->projection.corrected_face_velocities[i], 1e-9);
          }
        }
        std::cout << "boundary_pressure_kernel_ms=" << actual->measurement.kernel_ms
                  << " pressure=" << actual->projection.cell_pressures[0]
                  << " max_constraint_defect=" << actual->projection.diagnostics.max_cell_constraint_defect << '\n';
        return actual->projection;
      }
  );
  cases.AnalyticPlaneFixtureAllAxesSignedIndicesAndTinyOpening();
  cases.ObliqueAndSimultaneousLiquidSolidClips();
  cases.ThinSlabInvisibleToCornersAndSplitCellRefusal();
  cases.SharedFaceHasOneOwnerAndMatchingTriangulation();
  cases.SampledSolidFreezeSharesOnePressureFace();
  cases.ThetaFixtureAndSignedPressureMapping();
  cases.CavityGasPressureAddsSignedCapillaryJumpOnce();
  {
    SCOPED_TRACE("internal liquid interface");
    cases.SharedLiquidInterfaceUsesSeriesResistanceAndOneJump();
  }
  cases.PlaneAndSphereCenteredCurvature();
  cases.CapillaryLiftOnceAndCorrectionMismatchControl();
  cases.RejectsMissingSamplesOwnersOrderingAndInvalidObstacle();
  cases.CapillaryBoundZeroSigmaAndDegenerateDerivativeChecks();
  cases.FrozenSnapshotPreservesWallFluxAndSeparatesPressureHistory();
  cases.InternalLinkUsesFullFrozenAreaAndZeroJump();
  cases.RejectsPressureFieldsMissingAirAndNegativeOmittedNeighbor();
  cases.RejectsUnrepresentableMeasuresAndInvalidPhysicalInputs();
  cases.InteriorPositiveSigmaDoesNotRequireUnusedCurvature();
  cases.SealedLowFaceAtMinimumSignedIndexNeedsNoExteriorSample();
  cases.ResolvesRepresentableTinyLiquidTetrahedron();
  cases.RejectsCapillaryLiftThatUnderflowsToZero();
  cases.ZeroCornerPlateauHasNoStrictlyNegativeLiquidVolume();
  cases.MixedZeroPlateauCountsOnlyStrictlyNegativeTetrahedra();
  cases.PhysicalVolumeSurvivesUnderflowOfNormalizedCubicMeasure();
  cases.PhysicalFaceAreaSurvivesUnderflowOfNormalizedSquareMeasure();
  std::cout << std::setprecision(17) << "geometry_success_device_ms=" << geometry_device_ms
            << " geometry_all_calls_wall_ms=" << geometry_wall_ms << " links_success_device_ms=" << links_device_ms
            << " links_all_calls_wall_ms=" << links_wall_ms << " pressure_success_device_ms=" << pressure_device_ms
            << " pressure_all_calls_wall_ms=" << pressure_wall_ms << '\n';
  Receipt("liquid_boundary_geometry_3d", measurement);
}
TEST(CudaLiquidBoundary3D, SweptContactMatchesOracle) {
  Hardware();
  const LiquidWallContactInputs3D inputs{
      .start_world_position = {-1, 0, 0},
      .stored_world_velocity = {3, 4, 0},
      .advecting_world_velocity = {2, 2, 0},
      .obstacle = TranslatingSlab3D{{1, 0, 0}, 0, 1.0 / 32, {}},
      .step_duration = 1,
      .friction_coefficient = .5,
  };
  const auto cpu = ResolveLiquidWallContact3D(inputs);
  const auto gpu = CudaResolveLiquidWallContact3D(inputs);
  ASSERT_TRUE(cpu.ok()) << cpu.status();
  ASSERT_TRUE(gpu.ok()) << gpu.status();
  ASSERT_TRUE(gpu->contact.contact_fraction.has_value());
  EXPECT_DOUBLE_EQ(*gpu->contact.contact_fraction, 31.0 / 64);
  EXPECT_DOUBLE_EQ(gpu->contact.world_position.x, -1.0 / 32);
  EXPECT_DOUBLE_EQ(gpu->contact.stored_world_velocity.y, 2.5);
  EXPECT_DOUBLE_EQ(gpu->contact.advecting_world_velocity.y, 1);
  EXPECT_DOUBLE_EQ(gpu->contact.world_position.y, cpu->world_position.y);
  double contact_device_ms = 0;
  double contact_wall_ms = 0;
  boundary_test::ContactCases cases(
      [&](const LiquidWallContactInputs3D& input) -> absl::StatusOr<LiquidWallContactResult3D> {
        const auto expected = ResolveLiquidWallContact3D(input);
        const auto started = std::chrono::steady_clock::now();
        const auto actual = CudaResolveLiquidWallContact3D(input);
        contact_wall_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        EXPECT_EQ(expected.status().code(), actual.status().code());
        if (!actual.ok()) {
          return actual.status();
        }
        EXPECT_EQ(actual->measurement.compiled_architecture, 1200);
        contact_device_ms += actual->measurement.kernel_ms;
        if (expected.ok()) {
          const auto near = [](Vec3 left, Vec3 right) {
            EXPECT_NEAR(left.x, right.x, std::max(1.0, std::abs(left.x)) * 2e-14);
            EXPECT_NEAR(left.y, right.y, std::max(1.0, std::abs(left.y)) * 2e-14);
            EXPECT_NEAR(left.z, right.z, std::max(1.0, std::abs(left.z)) * 2e-14);
          };
          near(expected->world_position, actual->contact.world_position);
          near(expected->stored_world_velocity, actual->contact.stored_world_velocity);
          near(expected->advecting_world_velocity, actual->contact.advecting_world_velocity);
          EXPECT_EQ(expected->diagnostic, actual->contact.diagnostic);
          EXPECT_EQ(expected->contact_fraction, actual->contact.contact_fraction);
          EXPECT_DOUBLE_EQ(
              expected->incoming_stored_relative_normal_speed,
              actual->contact.incoming_stored_relative_normal_speed
          );
          EXPECT_DOUBLE_EQ(
              expected->incoming_advecting_relative_normal_speed,
              actual->contact.incoming_advecting_relative_normal_speed
          );
        }
        return actual->contact;
      }
  );
  cases.ThinSlabSweepMatchesFixtureOnAllAxesAndReversedNormals();
  cases.ObliqueReversedSlabPreservesIncidentSide();
  cases.OneSidedPlaneOnlyBlocksMotionIntoSolid();
  cases.TangentialMotionDoesNotTriggerFriction();
  cases.TouchingWallConstrainsStoredInwardVelocityDuringTangentAdvection();
  cases.FrictionUsesEachIncomingRelativeNormalSpeed();
  cases.RecoversTolerablePenetrationOnNearestSide();
  cases.RejectsDeepAndAmbiguousPenetration();
  cases.DetectsEndpointContactAndFreeAdvection();
  cases.RejectsInvalidContactInputs();
  cases.RejectsUnrepresentableTrajectoryAndWallPose();
  cases.CoMovingRoundedEndpointMayLandExactlyOnWall();
  cases.CoMovingRoundedEndpointInsideExactWallRejects();
  cases.ExactObliqueTangencyDoesNotBecomeAnImpact();
  cases.FrictionlessContactPreservesSmallWorldTangentAtFastWall();
  cases.ObliqueMovingPlaneCarriesTheConstrainedRemainder();
  std::cout << std::setprecision(17) << "contact_success_device_ms=" << contact_device_ms
            << " contact_all_calls_wall_ms=" << contact_wall_ms << '\n';
  Receipt("liquid_wall_contact_3d", gpu->measurement);
}
TEST(CudaLiquidBoundary3D, CompatibleTransferAndGhostHistoryMatchCpu) {
  Hardware();
  CudaOperatorMeasurement measurement{};
  double deposit_device_ms = 0, normalize_device_ms = 0, gather_device_ms = 0;
  double transfer_wall_ms = 0, max_gather_error = 0;
  std::size_t old_fallback = 0, new_fallback = 0, low_mass_fallback = 0;
  const auto compare_vector = [&](Vec3 actual, Vec3 expected) {
    max_gather_error = std::max(
        {max_gather_error,
         std::abs(actual.x - expected.x),
         std::abs(actual.y - expected.y),
         std::abs(actual.z - expected.z)}
    );
    boundary_transfer_test::ExpectVec(actual, expected, 1e-11);
  };

  boundary_transfer_test::TransferCases cases{
      [&](const LiquidBoundaryDepositInputs3D& input) -> absl::StatusOr<LiquidBoundaryFaceDeposits3D> {
        const auto expected = DepositCompatibleMacQuanta3D(input);
        const auto started = std::chrono::steady_clock::now();
        auto result = CudaDepositCompatibleMacQuanta3D(input);
        transfer_wall_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        EXPECT_EQ(result.status().code(), expected.status().code());
        if (!result.ok()) {
          return result.status();
        }
        measurement = result->measurement;
        deposit_device_ms += measurement.kernel_ms;
        EXPECT_EQ(measurement.compiled_architecture, 1200);
        if (expected.ok()) {
          for (std::size_t slot = 0; slot < expected->face_quanta.size(); ++slot) {
            EXPECT_EQ(result->deposits.face_quanta[slot].mass_quanta_sum, expected->face_quanta[slot].mass_quanta_sum);
            EXPECT_EQ(
                result->deposits.face_quanta[slot].pure_momentum_quanta_sum,
                expected->face_quanta[slot].pure_momentum_quanta_sum
            );
            EXPECT_EQ(result->deposits.face_quanta[slot].stress_force_quanta_sum, 0);
          }
          for (std::size_t particle = 0; particle < expected->compatible_stencils.size(); ++particle) {
            for (std::size_t component = 0; component < 3; ++component) {
              for (std::size_t sample = 0; sample < 27; ++sample) {
                const auto& actual = result->deposits.compatible_stencils[particle][component][sample];
                const auto& oracle = expected->compatible_stencils[particle][component][sample];
                EXPECT_EQ(actual.face_slot, oracle.face_slot);
                EXPECT_EQ(actual.compatible, oracle.compatible);
                EXPECT_DOUBLE_EQ(actual.normalized_weight, oracle.normalized_weight);
                compare_vector(actual.normalized_weight_gradient, oracle.normalized_weight_gradient);
                compare_vector(actual.centered_face_offset, oracle.centered_face_offset);
              }
            }
          }
        }
        return result->deposits;
      },
      [&](std::span<const MacFaceDepositQuanta3D> quanta, MacDepositQuantization3D quantization, double minimum
      ) -> absl::StatusOr<NormalizedLiquidMacFaces3D> {
        const auto expected = NormalizeLiquidMacFaces3D(quanta, quantization, minimum);
        const auto started = std::chrono::steady_clock::now();
        auto result = CudaNormalizeLiquidMacFaces3D(quanta, quantization, minimum);
        transfer_wall_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        EXPECT_EQ(result.status().code(), expected.status().code());
        if (!result.ok()) {
          return result.status();
        }
        normalize_device_ms += result->measurement.kernel_ms;
        EXPECT_EQ(result->measurement.compiled_architecture, 1200);
        if (expected.ok()) {
          EXPECT_EQ(result->faces.validity, expected->validity);
          EXPECT_EQ(result->faces.face_masses, expected->face_masses);
          EXPECT_EQ(result->faces.face_velocities, expected->face_velocities);
        }
        return result->faces;
      },
      [&](const LiquidBoundaryGatherInputs3D& input) -> absl::StatusOr<LiquidBoundaryGatherResults3D> {
        const auto expected = GatherLiquidBoundaryParticles3D(input);
        const auto started = std::chrono::steady_clock::now();
        auto result = CudaGatherLiquidBoundaryParticles3D(input);
        transfer_wall_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        EXPECT_EQ(result.status().code(), expected.status().code());
        if (!result.ok()) {
          return result.status();
        }
        gather_device_ms += result->measurement.kernel_ms;
        EXPECT_EQ(result->measurement.compiled_architecture, 1200);
        old_fallback += result->gathered.old_pic_fallback_particle_count;
        new_fallback += result->gathered.new_pic_fallback_particle_count;
        low_mass_fallback += result->gathered.low_mass_fallback_particle_count;
        if (expected.ok()) {
          EXPECT_EQ(result->gathered.old_pic_fallback_particle_count, expected->old_pic_fallback_particle_count);
          EXPECT_EQ(result->gathered.new_pic_fallback_particle_count, expected->new_pic_fallback_particle_count);
          EXPECT_EQ(result->gathered.low_mass_fallback_particle_count, expected->low_mass_fallback_particle_count);
          for (std::size_t index = 0; index < expected->particles.size(); ++index) {
            const auto& actual = result->gathered.particles[index];
            const auto& oracle = expected->particles[index];
            compare_vector(actual.new_pic_velocity, oracle.new_pic_velocity);
            compare_vector(actual.grid_velocity_increment, oracle.grid_velocity_increment);
            compare_vector(actual.particle_minus_old_pic_velocity, oracle.particle_minus_old_pic_velocity);
            for (auto row : {&Mat3::x_row, &Mat3::y_row, &Mat3::z_row}) {
              compare_vector(actual.fitted_velocity_slopes.*row, oracle.fitted_velocity_slopes.*row);
              compare_vector(actual.velocity_gradient.*row, oracle.velocity_gradient.*row);
            }
            EXPECT_EQ(actual.affine_enabled, oracle.affine_enabled);
            EXPECT_EQ(actual.old_pic_fallback, oracle.old_pic_fallback);
            EXPECT_EQ(actual.new_pic_fallback, oracle.new_pic_fallback);
            EXPECT_EQ(actual.low_mass_fallback, oracle.low_mass_fallback);
          }
        }
        return result->gathered;
      },
      cuda_internal::CudaFitLiquidGhostAffine3D,
  };
  cases.MaskedDepositRecentersMomentumAndQuotientGradient();
  cases.RejectsEmptyZeroSupportAndMissingExcludedOwner();
  cases.RoundsActualMaskedContributionsAndHonorsAffineOff();
  cases.NormalizationSeparatesLowMassFromMissingAndRejectsMalformedQuanta();
  cases.SameSideAffineGhostsExcludePoisonedDonorsAndSeparateHistories();
  cases.IdenticalHistoriesKeepParticleRetentionDistinctFromGridIncrement();
  cases.WallHistoriesUseSeparateTimesAndConstrainObliqueNormal();
  cases.SingularDonorsUseNearestContinuationAndSuppressRetention();
  cases.LowMassDisablesAffineAndRetentionEvenWithFullRankGhost();
  cases.NonaffineCompleteFieldReportsFitSeparatelyFromTrueGradient();
  cases.RankRevealingQrRejectsNearDependentNormalizedDirection();
  cases.NearestFallbackBreaksDistanceTiesByOwnedSlotOnlyInNewStage();
  cases.QuantizesEachParticleBeforeSummingAndIncludesGuardEndpoint();
  cases.CompleteBoundaryDepositPreservesUnmaskedApiAtSignedTranslatedScale();
  cases.RejectsMalformedHistoryCapacityAndArithmeticWithoutMutatingInputs();
  std::cout << std::setprecision(17) << "transfer_deposit_device_ms=" << deposit_device_ms
            << " transfer_normalize_device_ms=" << normalize_device_ms
            << " transfer_gather_device_ms=" << gather_device_ms << " transfer_all_calls_wall_ms=" << transfer_wall_ms
            << " transfer_max_cpu_error=" << max_gather_error << " transfer_old_fallback_count=" << old_fallback
            << " transfer_new_fallback_count=" << new_fallback << " transfer_low_mass_count=" << low_mass_fallback
            << '\n';
  Receipt("liquid_boundary_transfer_3d", measurement);
}
TEST(CudaLiquidBoundary3D, CoupledBoundaryOperatorsPreserveFailure) {
  Hardware();
  boundary_transfer_test::BoundaryFixture fixture;
  fixture.particle.transfer.particle_velocity = {-2, 4, 1};
  const double step_duration = 0.5;
  const LiquidObstacle3D obstacle = TranslatingHalfspace3D{{1, 0, 0}, 1.25, {0.25, 0, 0}};
  const std::array candidates{CellIndex3D{1, 1, 1}};
  std::vector<LiquidCenterDistance3D> centers;
  std::vector<LiquidCornerDistance3D> corners;
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        // Signed distance to the box of half-width .75 centered on the selected cell.
        const auto distance = [](double x, double y, double z) {
          const double dx = std::abs(x - 1.5) - .75;
          const double dy = std::abs(y - 1.5) - .75;
          const double dz = std::abs(z - 1.5) - .75;
          return std::hypot(std::max(dx, 0.0), std::max(dy, 0.0), std::max(dz, 0.0))
              + std::min(std::max({dx, dy, dz}), 0.0);
        };
        centers.push_back({{x, y, z}, distance(x + .5, y + .5, z + .5)});
        corners.push_back({{x, y, z}, distance(x, y, z)});
        for (auto axis : boundary_transfer_test::kAxes) {
          const auto slot = fixture.Slot(axis, x, y, z);
          const auto anchor = fixture.grid.FaceWorldAnchor({axis, {x, y, z}});
          ASSERT_TRUE(anchor.ok());
          fixture.tags[slot].structurally_eligible = anchor->x >= 1.25;
          fixture.old_walls[slot] = {true, 7, {1, 0, 0}, {.25, 0, 0}};
          fixture.new_walls[slot] = fixture.old_walls[slot];
        }
      }
    }
  }
  const LiquidBoundaryGeometryInputs3D
      geometry_inputs{fixture.grid, candidates, centers, corners, obstacle, step_duration, 1000, 0};
  const auto frozen = CudaFreezeLiquidBoundaryGeometry3D(geometry_inputs);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->geometry.cells().size(), 1);
  EXPECT_DOUBLE_EQ(frozen->geometry.cells()[0].fluid_volume, .75);
  const MacDepositQuantization3D coupling_quantization{0x1p-40, 0x1p-40, 1};
  auto deposit_inputs = fixture.DepositInputs();
  deposit_inputs.quantization = coupling_quantization;
  const auto deposits = CudaDepositCompatibleMacQuanta3D(deposit_inputs);
  ASSERT_TRUE(deposits.ok()) << deposits.status();
  const auto normalized = CudaNormalizeLiquidMacFaces3D(deposits->deposits.face_quanta, coupling_quantization, 0);
  ASSERT_TRUE(normalized.ok()) << normalized.status();
  fixture.old_values = normalized->faces.face_velocities;
  fixture.validity = normalized->faces.validity;
  // Explicit predicted extension supplies zero-mass same-side donors. Wall-side
  // nodes still go through separate old/new wall-constrained reconstruction.
  for (std::size_t slot = 0; slot < fixture.grid.face_count(); ++slot) {
    if (fixture.validity[slot] == LiquidFaceValidity3D::kMissing) {
      fixture.validity[slot] = LiquidFaceValidity3D::kPredicted;
      const auto axis = slot / 64;
      fixture.old_values[slot] = axis == 0 ? -2.0 : (axis == 1 ? 4.0 : 1.0);
    }
  }
  const std::array cell_fields{LiquidPressureCellFields3D{0, std::numeric_limits<double>::infinity(), 0}};
  std::vector<double> predicted;
  for (const auto& face : frozen->geometry.faces()) {
    predicted.push_back(fixture.old_values[face.face_slot]);
  }
  const auto links = CudaBuildLiquidPressureLinks3D(frozen->geometry, {cell_fields, predicted, 0});
  ASSERT_TRUE(links.ok()) << links.status();
  const auto projected =
      CudaProjectMacPressure3D({fixture.grid, step_duration, links->problem.cells, links->problem.faces});
  ASSERT_TRUE(projected.ok()) << projected.status();
  // Independent single-row balance: flux=-2.25, sum A=4, theta=.75,
  // dt/rho=.0005, hence p=843.75 and high-x correction=.5625 m/s.
  EXPECT_NEAR(projected->projection.cell_pressures[0], 843.75, 1e-9);
  fixture.new_values = fixture.old_values;
  for (std::size_t index = 0; index < links->problem.faces.size(); ++index) {
    const auto slot = fixture.grid.FaceSlot(links->problem.faces[index].face_index);
    ASSERT_TRUE(slot.ok());
    fixture.new_values[*slot] = projected->projection.corrected_face_velocities[index];
  }
  auto gather_inputs = fixture.GatherInputs();
  gather_inputs.new_time = step_duration;
  gather_inputs.new_history.velocity_sample_time = step_duration;
  gather_inputs.new_history.wall_sample_time = step_duration;
  const auto gathered = CudaGatherLiquidBoundaryParticles3D(gather_inputs);
  ASSERT_TRUE(gathered.ok()) << gathered.status();
  const auto& particle = gathered->gathered.particles[0];
  boundary_transfer_test::ExpectVec(particle.new_pic_velocity, {-.716796875, 4, 1}, 1e-10);
  boundary_transfer_test::ExpectVec(particle.grid_velocity_increment, {.158203125, 0, 0}, 1e-10);
  boundary_transfer_test::ExpectVec(particle.particle_minus_old_pic_velocity, {-1.125, 0, 0}, 1e-10);
  Vec3 stored = particle.new_pic_velocity;
  stored.x += particle.particle_minus_old_pic_velocity.x;
  stored.y += particle.particle_minus_old_pic_velocity.y;
  stored.z += particle.particle_minus_old_pic_velocity.z;
  const LiquidWallContactInputs3D contact_inputs{
      fixture.particle.transfer.trial_world_position,
      stored,
      particle.new_pic_velocity,
      obstacle,
      step_duration,
      .5,
      0
  };
  const auto contact = CudaResolveLiquidWallContact3D(contact_inputs);
  ASSERT_TRUE(contact.ok()) << contact.status();
  ASSERT_TRUE(contact->contact.contact_fraction.has_value());
  EXPECT_NEAR(*contact->contact.contact_fraction, .25 / (.966796875 * .5), 1e-10);
  EXPECT_NEAR(contact->contact.world_position.x, 1.375, 1e-12);
  EXPECT_NEAR(contact->contact.stored_world_velocity.x, .25, 1e-12);
  EXPECT_NEAR(contact->contact.advecting_world_velocity.x, .25, 1e-12);
  EXPECT_NEAR(contact->contact.incoming_stored_relative_normal_speed, -2.091796875, 1e-10);
  EXPECT_NEAR(contact->contact.incoming_advecting_relative_normal_speed, -.966796875, 1e-10);
  EXPECT_NEAR(contact->contact.stored_world_velocity.y, 4 * (1 - .5 * 2.091796875 / std::sqrt(17.0)), 1e-10);
  EXPECT_NEAR(contact->contact.advecting_world_velocity.y, 4 * (1 - .5 * .966796875 / std::sqrt(17.0)), 1e-10);
  const auto accepted = contact->contact;
  const auto accepted_gather = particle;
  const auto accepted_quanta = deposits->deposits.face_quanta;
  auto invalid_gather = gather_inputs;
  invalid_gather.old_history.wall_sample_time = step_duration;
  EXPECT_EQ(CudaGatherLiquidBoundaryParticles3D(invalid_gather).status().code(), absl::StatusCode::kInvalidArgument);
  auto invalid_pressure_faces = links->problem.faces;
  invalid_pressure_faces.pop_back();
  EXPECT_FALSE(
      CudaProjectMacPressure3D({fixture.grid, step_duration, links->problem.cells, invalid_pressure_faces}).ok()
  );
  auto invalid_particle = fixture.particle;
  invalid_particle.transfer.particle_velocity.x = std::numeric_limits<double>::max();
  auto invalid_deposit = fixture.DepositInputs();
  invalid_deposit.particles = {&invalid_particle, 1};
  EXPECT_EQ(CudaDepositCompatibleMacQuanta3D(invalid_deposit).status().code(), absl::StatusCode::kOutOfRange);
  auto invalid_contact = contact_inputs;
  invalid_contact.start_world_position.x = 0;
  EXPECT_EQ(CudaResolveLiquidWallContact3D(invalid_contact).status().code(), absl::StatusCode::kFailedPrecondition);
  boundary_transfer_test::ExpectVec(contact->contact.world_position, accepted.world_position, 0);
  boundary_transfer_test::ExpectVec(contact->contact.stored_world_velocity, accepted.stored_world_velocity, 0);
  boundary_transfer_test::ExpectVec(contact->contact.advecting_world_velocity, accepted.advecting_world_velocity, 0);
  boundary_transfer_test::ExpectVec(particle.new_pic_velocity, accepted_gather.new_pic_velocity, 0);
  for (std::size_t slot = 0; slot < accepted_quanta.size(); ++slot) {
    EXPECT_EQ(deposits->deposits.face_quanta[slot].mass_quanta_sum, accepted_quanta[slot].mass_quanta_sum);
    EXPECT_EQ(
        deposits->deposits.face_quanta[slot].pure_momentum_quanta_sum,
        accepted_quanta[slot].pure_momentum_quanta_sum
    );
  }
  std::cout << std::setprecision(17) << "coupled_pressure=" << projected->projection.cell_pressures[0]
            << " coupled_constraint_defect=" << projected->projection.diagnostics.max_cell_constraint_defect
            << " coupled_old_fallback=" << gathered->gathered.old_pic_fallback_particle_count
            << " coupled_new_fallback=" << gathered->gathered.new_pic_fallback_particle_count
            << " coupled_contact_fraction=" << *contact->contact.contact_fraction << '\n';
  Receipt("liquid_boundary_coupled_3d", gathered->measurement);
}

}  // namespace
}  // namespace fluidity
