#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "fixture_data.h"
#include "fluidity/host/cuda_liquid_boundary_3d.h"
#include "fluidity/host/cuda_mac_pressure_3d.h"
#include "gtest/gtest.h"
#include "liquid_boundary_geometry_3d_cases.h"
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
  cases.ThetaFixtureAndSignedPressureMapping();
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
}  // namespace
}  // namespace fluidity
