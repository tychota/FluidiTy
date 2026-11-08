#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

#include "fluidity/host/cuda_transfers.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr Grid2D kGrid{4, 4, 1.0};
constexpr std::size_t kFaceCount = 40;
constexpr std::size_t kXFace = 6;  // X-normal column 1, row 1.
constexpr CudaDepositScales kUnitScales{1.0, 1.0, 1.0};

template <std::size_t Count>
struct ParticleBatch {
  std::array<Vec2, Count> positions{};
  std::array<double, Count> masses{};
  std::array<Vec2, Count> velocities{};
  std::array<Mat2, Count> affine{};
  std::array<double, Count> volumes{};
  std::array<Mat2, Count> stresses{};

  ParticleView2D view() const {
    return {positions, masses, velocities, affine};
  }
};

struct DepositResult {
  static constexpr std::int64_t kSentinel = -99;
  static constexpr CudaOperatorMeasurement kSentinelMeasurement{-4, -7.0F};
  std::array<std::int64_t, kFaceCount> mass;
  std::array<std::int64_t, kFaceCount> momentum;
  std::array<std::int64_t, kFaceCount> force;
  CudaOperatorMeasurement measurement = kSentinelMeasurement;

  DepositResult() {
    mass.fill(kSentinel);
    momentum.fill(kSentinel);
    force.fill(kSentinel);
  }

  CudaDepositViews views() {
    return {mass, momentum, force};
  }

  void ExpectSentinels() const {
    std::array<std::int64_t, kFaceCount> sentinel;
    sentinel.fill(kSentinel);
    EXPECT_EQ(mass, sentinel);
    EXPECT_EQ(momentum, sentinel);
    EXPECT_EQ(force, sentinel);
    EXPECT_EQ(measurement.compiled_architecture, kSentinelMeasurement.compiled_architecture);
    EXPECT_EQ(measurement.kernel_ms, kSentinelMeasurement.kernel_ms);
  }
};

template <std::size_t Count>
absl::Status Deposit(const ParticleBatch<Count>& batch, CudaDepositScales scales, DepositResult& result) {
  return CudaP2G(kGrid, batch.view(), batch.volumes, batch.stresses, scales, result.views(), result.measurement);
}

void ExpectMeasurement(CudaOperatorMeasurement measurement) {
  EXPECT_GT(measurement.compiled_architecture, 0);
  EXPECT_TRUE(std::isfinite(measurement.kernel_ms));
  EXPECT_GE(measurement.kernel_ms, 0.0F);
}

TEST(CudaTransfer, DepositAnalyticAndCpuParity) {
  constexpr double scale = 0x1p-36;
  constexpr CudaDepositScales scales{scale, scale, scale};
  const ParticleBatch<1> analytic{
      .positions = {{{1.5, 1.5}}},
      .masses = {1.0},
      .velocities = {{{2.0, -3.0}}},
      .volumes = {2.0},
      .stresses = {{{4.0, 2.0, -3.0, -2.0}}},
  };
  DepositResult result;
  const auto analytic_status = Deposit(analytic, scales, result);
  ASSERT_TRUE(analytic_status.ok()) << analytic_status;
  EXPECT_EQ(static_cast<double>(result.mass[kXFace]) * scale, 0.375);
  EXPECT_EQ(static_cast<double>(result.momentum[kXFace]) * scale, 0.75);
  EXPECT_EQ(static_cast<double>(result.force[kXFace]) * scale, 6.0);
  EXPECT_EQ(static_cast<double>(result.force[7]) * scale, -6.0);
  EXPECT_EQ(static_cast<double>(result.force[1]) * scale, 2.0);
  EXPECT_EQ(static_cast<double>(result.force[25]) * scale, -3.0);
  EXPECT_EQ(static_cast<double>(result.force[29]) * scale, 3.0);
  EXPECT_EQ(static_cast<double>(result.force[24]) * scale, -2.0);
  ExpectMeasurement(result.measurement);

  constexpr Grid2D shifted_grid{4, 4, 1.0, {12.0, -7.0}};
  const ParticleBatch<2> affine{
      .positions = {{{13.25, -5.25}, {14.1, -4.7}}},
      .masses = {1.0, 2.5},
      .velocities = {{{2.0, -3.0}, {-1.0, 0.75}}},
      .affine = {{{0.5, -1.5, 2.0, 0.25}, {-2.0, 1.0, -0.5, 1.75}}},
  };
  std::array<double, kFaceCount> cpu_mass;
  std::array<double, kFaceCount> cpu_momentum;
  std::array<double, 2 * kFaceCount> scratch;
  const auto cpu_status = P2G(shifted_grid, affine.view(), cpu_mass, cpu_momentum, scratch);
  ASSERT_TRUE(cpu_status.ok()) << cpu_status;
  const auto gpu_status =
      CudaP2G(shifted_grid, affine.view(), affine.volumes, affine.stresses, scales, result.views(), result.measurement);
  ASSERT_TRUE(gpu_status.ok()) << gpu_status;
  ExpectMeasurement(result.measurement);

  // At most two contributions per face, each with at most half a scale unit of rounding.
  constexpr double quantization_bound = 2.0 * scale / 2.0;
  constexpr double fp_allowance = 2e-14;
  for (std::size_t face = 0; face < kFaceCount; ++face) {
    SCOPED_TRACE(face);
    EXPECT_NEAR(static_cast<double>(result.mass[face]) * scale, cpu_mass[face], quantization_bound + fp_allowance);
    EXPECT_NEAR(
        static_cast<double>(result.momentum[face]) * scale,
        cpu_momentum[face],
        quantization_bound + fp_allowance
    );
  }
  std::cout << "operator=p2g compiled_architecture=" << result.measurement.compiled_architecture
            << " kernel_ms=" << std::fixed << std::setprecision(9) << result.measurement.kernel_ms << '\n';
}

TEST(CudaTransfer, DepositRoundsContributionsBeforeOrderIndependentSum) {
  ParticleBatch<3> batch;
  batch.positions.fill({1.5, 1.5});
  batch.masses.fill(1.0);
  batch.velocities = {{{4.0, 0.0}, {4.0, 0.0}, {-12.0, 0.0}}};
  batch.volumes.fill(2.0);
  batch.stresses = {{{1.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {-3.0, 0.0, 0.0, 0.0}}};
  DepositResult forward;
  const auto forward_status = Deposit(batch, kUnitScales, forward);
  ASSERT_TRUE(forward_status.ok()) << forward_status;
  // At this face all mass contributions round down from 0.375. Momentum and force
  // round (1.5, 1.5, -4.5) to (2, 2, -4) rather than rounding the sum to -2.
  EXPECT_EQ(forward.mass[kXFace], 0);
  EXPECT_EQ(forward.momentum[kXFace], 0);
  EXPECT_EQ(forward.force[kXFace], 0);
  EXPECT_EQ(forward.force[7], 0);  // Opposite signs on the adjacent x-normal face.

  std::ranges::reverse(batch.velocities);
  std::ranges::reverse(batch.stresses);
  DepositResult reverse;
  const auto reverse_status = Deposit(batch, kUnitScales, reverse);
  ASSERT_TRUE(reverse_status.ok()) << reverse_status;
  EXPECT_EQ(reverse.mass, forward.mass);
  EXPECT_EQ(reverse.momentum, forward.momentum);
  EXPECT_EQ(reverse.force, forward.force);

  ParticleBatch<1024> boundary;
  boundary.positions.fill({1.5, 1.5});
  boundary.masses.fill(8.0 / 3.0);  // 0.375 * mass is exactly 1 after FP64 rounding.
  boundary.velocities.fill({0x1p52, 0.0});
  const auto boundary_status = Deposit(boundary, kUnitScales, forward);
  ASSERT_TRUE(boundary_status.ok()) << boundary_status;
  EXPECT_EQ(forward.momentum[kXFace], std::int64_t{1} << 62);
}

TEST(CudaTransfer, DepositRejectsInvalidSupportRangeAndCapacity) {
  DepositResult result;
  const auto expect_error = [&](const absl::Status& status, absl::StatusCode code) {
    EXPECT_EQ(status.code(), code) << status;
    result.ExpectSentinels();
  };

  ParticleBatch<2> batch;
  batch.positions.fill({1.5, 1.5});
  batch.masses.fill(8.0 / 3.0);
  batch.velocities = {{{1.0, 0.0}, {0x1.0000000000001p52, 0.0}}};
  expect_error(Deposit(batch, kUnitScales, result), absl::StatusCode::kOutOfRange);
  batch.velocities.back().x = 0.0;
  batch.affine.back().xx = std::numeric_limits<double>::max();
  expect_error(Deposit(batch, kUnitScales, result), absl::StatusCode::kOutOfRange);
  batch.affine.back().xx = 0.0;
  batch.volumes.back() = 2.0;
  batch.stresses.back().xx = std::numeric_limits<double>::max();
  expect_error(Deposit(batch, kUnitScales, result), absl::StatusCode::kOutOfRange);
  batch.volumes.back() = 0.0;
  batch.stresses.back().xx = 0.0;
  batch.positions.back().x = 3.5;  // A zero-weight x-normal node lies outside the lattice.
  expect_error(Deposit(batch, kUnitScales, result), absl::StatusCode::kInvalidArgument);
  batch.positions.back().x = 1.5;

  expect_error(Deposit(batch, {3.0, 1.0, 1.0}, result), absl::StatusCode::kInvalidArgument);
  expect_error(
      Deposit(batch, {std::numeric_limits<double>::denorm_min(), 1.0, 1.0}, result),
      absl::StatusCode::kInvalidArgument
  );
  batch.volumes.back() = -1.0;
  expect_error(Deposit(batch, kUnitScales, result), absl::StatusCode::kInvalidArgument);
  batch.volumes.back() = 1.0;
  batch.stresses.back().xx = std::numeric_limits<double>::infinity();
  expect_error(Deposit(batch, kUnitScales, result), absl::StatusCode::kInvalidArgument);
  batch.stresses.back().xx = 1.0;

  CudaDepositViews overlap = result.views();
  overlap.force = overlap.mass;
  expect_error(
      CudaP2G(kGrid, batch.view(), batch.volumes, batch.stresses, kUnitScales, overlap, result.measurement),
      absl::StatusCode::kInvalidArgument
  );
  expect_error(
      CudaP2G(kGrid, batch.view(), {}, batch.stresses, kUnitScales, result.views(), result.measurement),
      absl::StatusCode::kInvalidArgument
  );
  expect_error(Deposit(ParticleBatch<1025>{}, kUnitScales, result), absl::StatusCode::kResourceExhausted);
  expect_error(
      CudaP2G(Grid2D{64, 64, 1.0}, {}, {}, {}, kUnitScales, result.views(), result.measurement),
      absl::StatusCode::kResourceExhausted
  );
}

TEST(CudaTransfer, DepositEmptyBatchClearsOutputs) {
  DepositResult result;
  const auto status = Deposit(ParticleBatch<0>{}, {0x1p-1022, 0x1p1023, 1.0}, result);
  ASSERT_TRUE(status.ok()) << status;
  const std::array<std::int64_t, kFaceCount> zero{};
  EXPECT_EQ(result.mass, zero);
  EXPECT_EQ(result.momentum, zero);
  EXPECT_EQ(result.force, zero);
  ExpectMeasurement(result.measurement);
}

}  // namespace
}  // namespace fluidity
