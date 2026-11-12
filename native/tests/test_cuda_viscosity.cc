#include <array>
#include <cmath>
#include <vector>

#include "fluidity/host/cuda_viscosity.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

std::array<MacPressureFace3D, 6> SealedFacesAtX(std::int32_t x) {
  return {{
      {.face_index = {FaceNormalAxis::kX, {x, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kX, {x + 1, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kY, {x, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kY, {x, 1, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kZ, {x, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      {.face_index = {FaceNormalAxis::kZ, {x, 0, 1}}, .pressure_link = SealedMacPressureLink3D{}},
  }};
}

TEST(CudaMacViscosity3D, CoupledProjectionBalancesMomentumAndCompliantFlux) {
  constexpr UniformMacGeometry3D kUnitGeometry{{0.0, 0.0, 0.0}, 1.0};
  constexpr std::array kUnitBlock{BlockIndex3D{0, 0, 0}};
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, kUnitBlock, kUnitBlock.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto faces = SealedFacesAtX(0);
  for (std::size_t face = 0; face < 2; ++face) {
    faces[face].open_area = 1.0;
    faces[face].inverse_density = 1.0;
    faces[face].pressure_gradient_distance = 1.0;
  }
  faces[0].predicted_normal_velocity = -1.0;
  faces[0].pressure_link = DirichletMacPressureLink3D{2.0};
  faces[1].predicted_normal_velocity = 8.0;
  faces[1].pressure_link = DirichletMacPressureLink3D{5.0};
  const std::array cells{MacPressureCell3D{
      .cell_index = {0, 0, 0},
      .fluid_volume = 1.0,
      .target_divergence = 3.0,
      .bulk_modulus = 2.0,
      .previous_pressure = 2.0,
  }};
  const std::array viscosity{MacViscosityCell3D{.cell_index = {0, 0, 0}, .dynamic_viscosity = 1.0}};
  const auto strain = AssembleFullCellMacStrain3D(grid->view(), viscosity);
  ASSERT_TRUE(strain.ok()) << strain.status();
  const MacPressureInputs3D inputs{*grid, 1.0, cells, faces};
  const auto cpu = ProjectMacViscousPressure3D(inputs, *strain);
  ASSERT_TRUE(cpu.ok()) << cpu.status();
  const auto result = CudaProjectMacViscousPressure3D(inputs, *strain);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->pressure_iterations, cpu->pressure_iterations);
  EXPECT_NEAR(result->cell_pressures[0], cpu->cell_pressures[0], 1e-12);
  for (std::size_t face = 0; face < result->face_velocities.size(); ++face) {
    EXPECT_NEAR(result->face_velocities[face], cpu->face_velocities[face], 1e-12);
  }
  EXPECT_NEAR(result->cell_pressures[0], 4.0, 1e-9);
  EXPECT_NEAR(result->face_velocities[0], 1.0, 1e-9);
  EXPECT_NEAR(result->face_velocities[1], 3.0, 1e-9);
  EXPECT_NEAR(result->cell_constraint_defects[0], 0.0, 1e-9);
  EXPECT_LT(result->max_momentum_residual, 1e-9);
}

TEST(CudaMacViscosity3D, VelocityBlockMatchesCpuAndTrueResidual) {
  const std::array rows{
      MacStrainRow3D{
          .terms = {{{.face_slot = 0, .coefficient = -2.0}, {.face_slot = 1, .coefficient = 1.0}}},
          .term_count = 2,
          .quadrature_weight = 4.0,
      },
      MacStrainRow3D{
          .terms = {{{.face_slot = 1, .coefficient = 2.0}, {.face_slot = 2, .coefficient = -1.0}}},
          .term_count = 2,
          .quadrature_weight = 0.125,
      },
      MacStrainRow3D{
          .terms = {{{.face_slot = 2, .coefficient = 1.0}, {.face_slot = 3, .coefficient = -2.0}}},
          .term_count = 2,
          .quadrature_weight = 8.0,
      },
  };
  const std::array inertia{1.0, 2.0, 3.0, 0.0};
  const std::array rhs{3.0, -2.0, 5.0, 0.0};
  const MacViscousVelocityBlockInputs3D inputs{rows, inertia, 0.5};
  const auto cpu = SolveMacViscousVelocityBlock3D(inputs, rhs);
  ASSERT_TRUE(cpu.ok()) << cpu.status();
  const auto gpu = CudaSolveMacViscousVelocityBlock3D(inputs, rhs);
  ASSERT_TRUE(gpu.ok()) << gpu.status();
  EXPECT_GT(gpu->measurement.compiled_architecture, 0);
  EXPECT_GE(gpu->measurement.kernel_ms, 0.0F);
  EXPECT_EQ(gpu->solution.iterations, cpu->iterations);
  for (std::size_t face = 0; face < rhs.size(); ++face) {
    EXPECT_NEAR(gpu->solution.velocity[face], cpu->velocity[face], 1e-13);
  }
  const auto& velocity = gpu->solution.velocity;
  const std::array defects{
      rhs[0] - (9.0 * velocity[0] - 4.0 * velocity[1]),
      rhs[1] - (-4.0 * velocity[0] + 4.25 * velocity[1] - 0.125 * velocity[2]),
      rhs[2] - (-0.125 * velocity[1] + 7.0625 * velocity[2]),
  };
  for (const double defect : defects) {
    EXPECT_LE(std::abs(defect), 5e-10);
  }
  EXPECT_LE(gpu->solution.max_residual, 5e-10);
  EXPECT_EQ(velocity[3], 0.0);
  RecordProperty("compiled_architecture", gpu->measurement.compiled_architecture);
}

TEST(CudaMacViscosity3D, RejectsExhaustionAndEliminatedMomentum) {
  const std::array inertia{1.0, 1.0, 0.0};
  const std::array rows{MacStrainRow3D{
      .terms = {{{.face_slot = 0, .coefficient = 1.0}, {.face_slot = 1, .coefficient = -1.0}}},
      .term_count = 2,
      .quadrature_weight = 4.0,
  }};
  const MacViscousVelocityBlockInputs3D inputs{rows, inertia, 1.0};
  const std::array rhs{1.0, 0.0, 0.0};
  EXPECT_EQ(
      CudaSolveMacViscousVelocityBlock3D(inputs, rhs, {.max_iterations = 1}).status().code(),
      absl::StatusCode::kAborted
  );
  const std::array invalid_rhs{1.0, 0.0, 1.0};
  EXPECT_EQ(
      CudaSolveMacViscousVelocityBlock3D(inputs, invalid_rhs).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  const std::array zero_rhs{0.0, 0.0, 0.0};
  const auto zero = CudaSolveMacViscousVelocityBlock3D(inputs, zero_rhs);
  ASSERT_TRUE(zero.ok()) << zero.status();
  EXPECT_EQ(zero->solution.iterations, 0U);
  EXPECT_EQ(zero->solution.max_residual, 0.0);
  EXPECT_EQ(zero->solution.velocity, (std::vector<double>{0.0, 0.0, 0.0}));
}

}  // namespace
}  // namespace fluidity
