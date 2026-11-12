#include <array>
#include <cmath>
#include <vector>

#include "fluidity/host/cuda_viscosity.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

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
