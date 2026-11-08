#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

#include "fixture_data.h"
#include "fluidity/host/cuda_transfers.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr Grid2D kGrid{4, 4, 1.0};
constexpr std::size_t kXNormalFaceCount = (kGrid.nx + 1) * kGrid.ny;
constexpr std::size_t kFaceCount = 40;
constexpr std::size_t kInteriorXNormalFaceIndex = 6;  // X-normal column 1, row 1.
constexpr CudaDepositScales kUnitScales{1.0, 1.0, 1.0};

template <std::size_t ParticleCount>
struct ParticleBatch {
  std::array<Vec2, ParticleCount> positions{};
  std::array<double, ParticleCount> masses{};
  std::array<Vec2, ParticleCount> velocities{};
  std::array<Mat2, ParticleCount> velocity_fit_slopes{};
  std::array<double, ParticleCount> volumes{};
  std::array<Mat2, ParticleCount> stresses{};

  ParticleView2D view() const {
    return {positions, masses, velocities, velocity_fit_slopes};
  }
};

struct DepositResult {
  static constexpr std::int64_t kSentinel = -99;
  static constexpr CudaOperatorMeasurement kSentinelMeasurement{-4, -7.0F};
  std::array<std::int64_t, kFaceCount> mass_quanta;
  std::array<std::int64_t, kFaceCount> pure_momentum_quanta;
  std::array<std::int64_t, kFaceCount> stress_force_quanta;
  CudaOperatorMeasurement measurement = kSentinelMeasurement;

  DepositResult() {
    mass_quanta.fill(kSentinel);
    pure_momentum_quanta.fill(kSentinel);
    stress_force_quanta.fill(kSentinel);
  }

  CudaDepositViews views() {
    return {mass_quanta, pure_momentum_quanta, stress_force_quanta};
  }

  void ExpectSentinels() const {
    std::array<std::int64_t, kFaceCount> sentinel;
    sentinel.fill(kSentinel);
    EXPECT_EQ(mass_quanta, sentinel);
    EXPECT_EQ(pure_momentum_quanta, sentinel);
    EXPECT_EQ(stress_force_quanta, sentinel);
    EXPECT_EQ(measurement.compiled_architecture, kSentinelMeasurement.compiled_architecture);
    EXPECT_EQ(measurement.kernel_ms, kSentinelMeasurement.kernel_ms);
  }
};

template <std::size_t ParticleCount>
absl::Status Deposit(const ParticleBatch<ParticleCount>& batch, CudaDepositScales scales, DepositResult& result) {
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
  EXPECT_EQ(static_cast<double>(result.mass_quanta[kInteriorXNormalFaceIndex]) * scale, 0.375);
  EXPECT_EQ(static_cast<double>(result.pure_momentum_quanta[kInteriorXNormalFaceIndex]) * scale, 0.75);
  EXPECT_EQ(static_cast<double>(result.stress_force_quanta[kInteriorXNormalFaceIndex]) * scale, 6.0);
  EXPECT_EQ(static_cast<double>(result.stress_force_quanta[7]) * scale, -6.0);
  EXPECT_EQ(static_cast<double>(result.stress_force_quanta[1]) * scale, 2.0);
  EXPECT_EQ(static_cast<double>(result.stress_force_quanta[25]) * scale, -3.0);
  EXPECT_EQ(static_cast<double>(result.stress_force_quanta[29]) * scale, 3.0);
  EXPECT_EQ(static_cast<double>(result.stress_force_quanta[24]) * scale, -2.0);
  ExpectMeasurement(result.measurement);

  constexpr Grid2D shifted_grid{4, 4, 1.0, {12.0, -7.0}};
  const ParticleBatch<2> affine_particle_batch{
      .positions = {{{13.25, -5.25}, {14.1, -4.7}}},
      .masses = {1.0, 2.5},
      .velocities = {{{2.0, -3.0}, {-1.0, 0.75}}},
      .velocity_fit_slopes = {{{0.5, -1.5, 2.0, 0.25}, {-2.0, 1.0, -0.5, 1.75}}},
  };
  std::array<double, kFaceCount> cpu_mass;
  std::array<double, kFaceCount> cpu_momentum;
  std::array<double, 2 * kFaceCount> scratch;
  const auto cpu_status = P2G(shifted_grid, affine_particle_batch.view(), cpu_mass, cpu_momentum, scratch);
  ASSERT_TRUE(cpu_status.ok()) << cpu_status;
  const auto gpu_status = CudaP2G(
      shifted_grid,
      affine_particle_batch.view(),
      affine_particle_batch.volumes,
      affine_particle_batch.stresses,
      scales,
      result.views(),
      result.measurement
  );
  ASSERT_TRUE(gpu_status.ok()) << gpu_status;
  ExpectMeasurement(result.measurement);

  // At most two contributions per face, each with at most half a scale unit of rounding.
  constexpr double quantization_bound = 2.0 * scale / 2.0;
  constexpr double floating_point_tolerance = 2e-14;
  for (std::size_t face = 0; face < kFaceCount; ++face) {
    SCOPED_TRACE(face);
    EXPECT_NEAR(
        static_cast<double>(result.mass_quanta[face]) * scale,
        cpu_mass[face],
        quantization_bound + floating_point_tolerance
    );
    EXPECT_NEAR(
        static_cast<double>(result.pure_momentum_quanta[face]) * scale,
        cpu_momentum[face],
        quantization_bound + floating_point_tolerance
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
  EXPECT_EQ(forward.mass_quanta[kInteriorXNormalFaceIndex], 0);
  EXPECT_EQ(forward.pure_momentum_quanta[kInteriorXNormalFaceIndex], 0);
  EXPECT_EQ(forward.stress_force_quanta[kInteriorXNormalFaceIndex], 0);
  EXPECT_EQ(forward.stress_force_quanta[7], 0);  // Opposite signs on the adjacent x-normal face.

  std::ranges::reverse(batch.velocities);
  std::ranges::reverse(batch.stresses);
  DepositResult reverse;
  const auto reverse_status = Deposit(batch, kUnitScales, reverse);
  ASSERT_TRUE(reverse_status.ok()) << reverse_status;
  EXPECT_EQ(reverse.mass_quanta, forward.mass_quanta);
  EXPECT_EQ(reverse.pure_momentum_quanta, forward.pure_momentum_quanta);
  EXPECT_EQ(reverse.stress_force_quanta, forward.stress_force_quanta);

  ParticleBatch<1024> boundary;
  boundary.positions.fill({1.5, 1.5});
  boundary.masses.fill(8.0 / 3.0);  // 0.375 * mass is exactly 1 after FP64 rounding.
  boundary.velocities.fill({0x1p52, 0.0});
  const auto boundary_status = Deposit(boundary, kUnitScales, forward);
  ASSERT_TRUE(boundary_status.ok()) << boundary_status;
  EXPECT_EQ(forward.pure_momentum_quanta[kInteriorXNormalFaceIndex], std::int64_t{1} << 62);
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
  batch.velocity_fit_slopes.back().xx = std::numeric_limits<double>::max();
  expect_error(Deposit(batch, kUnitScales, result), absl::StatusCode::kOutOfRange);
  batch.velocity_fit_slopes.back().xx = 0.0;
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
  overlap.stress_force_quanta = overlap.mass_quanta;
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
  EXPECT_EQ(result.mass_quanta, zero);
  EXPECT_EQ(result.pure_momentum_quanta, zero);
  EXPECT_EQ(result.stress_force_quanta, zero);
  ExpectMeasurement(result.measurement);
}

template <std::size_t ParticleCount>
struct GatherResult {
  std::array<Vec2, ParticleCount> pic_velocities;
  std::array<Vec2, ParticleCount> grid_velocity_increments;
  std::array<Mat2, ParticleCount> velocity_fit_slopes;
  std::array<Mat2, ParticleCount> velocity_gradients;
  CudaOperatorMeasurement kernel_measurement{-4, -7.0F};

  CudaGatherViews views() {
    return {pic_velocities, grid_velocity_increments, velocity_fit_slopes, velocity_gradients};
  }
};

void ExpectGatherNear(double actual, double expected) {
  constexpr double kAbsoluteTolerance = 2e-12;
  constexpr double kRelativeTolerance = 2e-12;
  EXPECT_NEAR(actual, expected, kAbsoluteTolerance + kRelativeTolerance * std::abs(expected));
}

void ExpectGatherVector(Vec2 actual, Vec2 expected) {
  ExpectGatherNear(actual.x, expected.x);
  ExpectGatherNear(actual.y, expected.y);
}

void ExpectGatherMatrix(Mat2 actual, Mat2 expected) {
  ExpectGatherNear(actual.xx, expected.xx);
  ExpectGatherNear(actual.xy, expected.xy);
  ExpectGatherNear(actual.yx, expected.yx);
  ExpectGatherNear(actual.yy, expected.yy);
}

template <std::size_t ParticleCount>
void ExpectGatherCpuParity(
    const Grid2D& mac_grid,
    const std::array<Vec2, ParticleCount>& particle_world_positions,
    const std::array<double, kFaceCount>& old_face_velocities,
    const std::array<double, kFaceCount>& new_face_velocities,
    const GatherResult<ParticleCount>& cuda_gather
) {
  GatherResult<ParticleCount> cpu_gather;
  constexpr std::size_t kCpuGatherScratchDoublesPerParticle = 12;
  std::array<double, kCpuGatherScratchDoublesPerParticle * ParticleCount> cpu_gather_scratch;
  const auto cpu_gather_status =
      G2P(mac_grid,
          particle_world_positions,
          old_face_velocities,
          new_face_velocities,
          cpu_gather.pic_velocities,
          cpu_gather.grid_velocity_increments,
          cpu_gather.velocity_fit_slopes,
          cpu_gather.velocity_gradients,
          cpu_gather_scratch);
  ASSERT_TRUE(cpu_gather_status.ok()) << cpu_gather_status;
  for (std::size_t particle_index = 0; particle_index < ParticleCount; ++particle_index) {
    SCOPED_TRACE(particle_index);
    ExpectGatherVector(cuda_gather.pic_velocities[particle_index], cpu_gather.pic_velocities[particle_index]);
    ExpectGatherVector(
        cuda_gather.grid_velocity_increments[particle_index],
        cpu_gather.grid_velocity_increments[particle_index]
    );
    ExpectGatherMatrix(cuda_gather.velocity_fit_slopes[particle_index], cpu_gather.velocity_fit_slopes[particle_index]);
    ExpectGatherMatrix(cuda_gather.velocity_gradients[particle_index], cpu_gather.velocity_gradients[particle_index]);
  }
}

TEST(CudaTransfer, GatherAffineMatchesFixtureAndCpu) {
  const auto fixture_records = fixtures::AllFixtures();
  const auto affine_fixture = std::ranges::find(fixture_records, "affine_apic", &fixtures::FixtureRecord::name);
  ASSERT_NE(affine_fixture, fixture_records.end());
  const auto fixture_position = fixtures::FindArray(*affine_fixture, "position");
  const auto fixture_sample_positions = fixtures::FindArray(*affine_fixture, "sample_positions");
  const auto fixture_velocity_samples = fixtures::FindArray(*affine_fixture, "values");
  const auto fixture_expected_pic = fixtures::FindArray(*affine_fixture, "expected_value");
  const auto fixture_expected_affine = fixtures::FindArray(*affine_fixture, "expected_affine");
  for (const auto* fixture_array_result :
       {&fixture_position,
        &fixture_sample_positions,
        &fixture_velocity_samples,
        &fixture_expected_pic,
        &fixture_expected_affine}) {
    ASSERT_TRUE(fixture_array_result->ok()) << fixture_array_result->status();
    ASSERT_EQ((**fixture_array_result)->dtype, fixtures::DType::kFloat64);
  }
  ASSERT_EQ((*fixture_position)->f64_data.size(), 2);
  ASSERT_EQ((*fixture_sample_positions)->f64_data.size(), 8);
  ASSERT_EQ((*fixture_velocity_samples)->f64_data.size(), 4);
  ASSERT_EQ((*fixture_expected_pic)->f64_data.size(), 1);
  ASSERT_EQ((*fixture_expected_affine)->f64_data.size(), 2);

  const double expected_center_pic = (*fixture_expected_pic)->f64_data[0];
  const Vec2 expected_x_affine_row{(*fixture_expected_affine)->f64_data[0], (*fixture_expected_affine)->f64_data[1]};
  const Vec2 fixture_particle_position{(*fixture_position)->f64_data[0], (*fixture_position)->f64_data[1]};
  constexpr Grid2D kShiftedMacGrid{4, 4, 1.0, {12.0, -7.0}};
  constexpr Vec2 kXFaceCenterInGrid{2.0, 1.5};
  constexpr Vec2 kXFaceCenterInWorld = kShiftedMacGrid.origin + kXFaceCenterInGrid;
  const Vec2 fixture_translation = kXFaceCenterInWorld - fixture_particle_position;
  const std::array<Vec2, 2> particle_world_positions{{
      kXFaceCenterInWorld,
      kShiftedMacGrid.origin + Vec2{1.25, 2.125},
  }};
  std::array<double, kFaceCount> old_face_velocities;
  std::array<double, kFaceCount> new_face_velocities;
  constexpr Vec2 kExpectedGridVelocityIncrement{1.5, -0.75};
  std::size_t embedded_sample_count = 0;
  for (std::size_t face_index = 0; face_index < new_face_velocities.size(); ++face_index) {
    const auto face_geometry = FaceGeometry(kShiftedMacGrid, face_index);
    ASSERT_TRUE(face_geometry.ok()) << face_geometry.status();
    const Vec2 face_offset = face_geometry->world_position - kXFaceCenterInWorld;
    const bool is_x_normal = face_index < kXNormalFaceCount;
    if (is_x_normal) {
      new_face_velocities[face_index] =
          expected_center_pic + expected_x_affine_row.x * face_offset.x + expected_x_affine_row.y * face_offset.y;
      // Embed the fixture's four axial samples on their actual MAC faces.
      for (std::size_t sample_index = 0; sample_index < (*fixture_velocity_samples)->f64_data.size(); ++sample_index) {
        const Vec2 local_sample_position{
            (*fixture_sample_positions)->f64_data[2 * sample_index],
            (*fixture_sample_positions)->f64_data[2 * sample_index + 1],
        };
        const Vec2 sample_face_position = local_sample_position + fixture_translation;
        if (face_geometry->world_position.x == sample_face_position.x
            && face_geometry->world_position.y == sample_face_position.y) {
          new_face_velocities[face_index] = (*fixture_velocity_samples)->f64_data[sample_index];
          ++embedded_sample_count;
        }
      }
      old_face_velocities[face_index] = new_face_velocities[face_index] - kExpectedGridVelocityIncrement.x;
    } else {
      new_face_velocities[face_index] = -2.0 + 1.25 * face_offset.x - 0.5 * face_offset.y;
      old_face_velocities[face_index] = new_face_velocities[face_index] - kExpectedGridVelocityIncrement.y;
    }
  }
  ASSERT_EQ(embedded_sample_count, 4);

  GatherResult<particle_world_positions.size()> cuda_gather;
  const auto cuda_gather_status = CudaG2P(
      kShiftedMacGrid,
      particle_world_positions,
      old_face_velocities,
      new_face_velocities,
      cuda_gather.views(),
      cuda_gather.kernel_measurement
  );
  ASSERT_TRUE(cuda_gather_status.ok()) << cuda_gather_status;
  EXPECT_NEAR(
      cuda_gather.pic_velocities.front().x,
      expected_center_pic,
      affine_fixture->atol + affine_fixture->rtol * std::abs(expected_center_pic)
  );
  EXPECT_NEAR(
      cuda_gather.velocity_fit_slopes.front().xx,
      expected_x_affine_row.x,
      affine_fixture->atol + affine_fixture->rtol * std::abs(expected_x_affine_row.x)
  );
  EXPECT_NEAR(
      cuda_gather.velocity_fit_slopes.front().xy,
      expected_x_affine_row.y,
      affine_fixture->atol + affine_fixture->rtol * std::abs(expected_x_affine_row.y)
  );
  ExpectGatherVector(cuda_gather.grid_velocity_increments.front(), kExpectedGridVelocityIncrement);
  ExpectGatherCpuParity(
      kShiftedMacGrid,
      particle_world_positions,
      old_face_velocities,
      new_face_velocities,
      cuda_gather
  );
  ExpectMeasurement(cuda_gather.kernel_measurement);
  std::cout << "operator=g2p compiled_architecture=" << cuda_gather.kernel_measurement.compiled_architecture
            << " kernel_ms=" << std::fixed << std::setprecision(9) << cuda_gather.kernel_measurement.kernel_ms << '\n';
}

TEST(CudaTransfer, GatherNonlinearKeepsTrueGradientSeparate) {
  constexpr std::array<Vec2, 1> particle_world_positions{{{1.25, 1.5}}};
  std::array<double, kFaceCount> shared_face_velocities;
  for (std::size_t face_index = 0; face_index < shared_face_velocities.size(); ++face_index) {
    const auto face_geometry = FaceGeometry(kGrid, face_index);
    ASSERT_TRUE(face_geometry.ok()) << face_geometry.status();
    const Vec2 face_position = face_geometry->world_position;
    shared_face_velocities[face_index] =
        face_index < kXNormalFaceCount ? face_position.x * face_position.x : 2.0 * face_position.y * face_position.y;
  }

  GatherResult<particle_world_positions.size()> cuda_gather;
  const auto cuda_gather_status = CudaG2P(
      kGrid,
      particle_world_positions,
      shared_face_velocities,
      shared_face_velocities,
      cuda_gather.views(),
      cuda_gather.kernel_measurement
  );
  ASSERT_TRUE(cuda_gather_status.ok()) << cuda_gather_status;
  // At x=1.25 the three x weights are (1,22,9)/32 and their derivatives are
  // (-1,-2,3)/4. The centered moment fit and derivative sum differ on x squared.
  ExpectGatherNear(cuda_gather.pic_velocities.front().x, 1.8125);
  ExpectGatherNear(cuda_gather.velocity_fit_slopes.front().xx, 2.6875);
  ExpectGatherNear(cuda_gather.velocity_gradients.front().xx, 2.5);
  EXPECT_EQ(cuda_gather.grid_velocity_increments.front().x, 0.0);
  EXPECT_EQ(cuda_gather.grid_velocity_increments.front().y, 0.0);
  ExpectGatherCpuParity(kGrid, particle_world_positions, shared_face_velocities, shared_face_velocities, cuda_gather);
}

TEST(CudaTransfer, GatherFailurePreservesOutputs) {
  constexpr Vec2 kPicSentinel{11.0, 12.0};
  constexpr Vec2 kIncrementSentinel{13.0, 14.0};
  constexpr Mat2 kAffineSentinel{15.0, 16.0, 17.0, 18.0};
  constexpr Mat2 kGradientSentinel{19.0, 20.0, 21.0, 22.0};
  GatherResult<2> cuda_gather;
  cuda_gather.pic_velocities.fill(kPicSentinel);
  cuda_gather.grid_velocity_increments.fill(kIncrementSentinel);
  cuda_gather.velocity_fit_slopes.fill(kAffineSentinel);
  cuda_gather.velocity_gradients.fill(kGradientSentinel);
  const CudaOperatorMeasurement measurement_sentinel = cuda_gather.kernel_measurement;

  std::array<Vec2, 2> particle_world_positions{{{2.5, 2.5}, {1.5, 1.5}}};
  std::array<double, kFaceCount> old_face_velocities{};
  std::array<double, kFaceCount> new_face_velocities{};
  const auto gather = [&] {
    return CudaG2P(
        kGrid,
        particle_world_positions,
        old_face_velocities,
        new_face_velocities,
        cuda_gather.views(),
        cuda_gather.kernel_measurement
    );
  };
  const auto expect_preserved = [&] {
    const auto vector_entries = [](Vec2 vector) {
      return std::array{vector.x, vector.y};
    };
    const auto matrix_entries = [](Mat2 matrix) {
      return std::array{matrix.xx, matrix.xy, matrix.yx, matrix.yy};
    };
    for (std::size_t particle_index = 0; particle_index < particle_world_positions.size(); ++particle_index) {
      EXPECT_EQ(vector_entries(cuda_gather.pic_velocities[particle_index]), vector_entries(kPicSentinel));
      EXPECT_EQ(
          vector_entries(cuda_gather.grid_velocity_increments[particle_index]),
          vector_entries(kIncrementSentinel)
      );
      EXPECT_EQ(matrix_entries(cuda_gather.velocity_fit_slopes[particle_index]), matrix_entries(kAffineSentinel));
      EXPECT_EQ(matrix_entries(cuda_gather.velocity_gradients[particle_index]), matrix_entries(kGradientSentinel));
    }
    EXPECT_EQ(cuda_gather.kernel_measurement.compiled_architecture, measurement_sentinel.compiled_architecture);
    EXPECT_EQ(cuda_gather.kernel_measurement.kernel_ms, measurement_sentinel.kernel_ms);
  };
  const auto expect_error = [&](const absl::Status& gather_status, absl::StatusCode expected_code) {
    EXPECT_EQ(gather_status.code(), expected_code) << gather_status;
    expect_preserved();
  };

  old_face_velocities.back() = std::numeric_limits<double>::infinity();
  expect_error(gather(), absl::StatusCode::kInvalidArgument);
  old_face_velocities.back() = 0.0;
  new_face_velocities.back() = std::numeric_limits<double>::quiet_NaN();
  expect_error(gather(), absl::StatusCode::kInvalidArgument);
  new_face_velocities.back() = 0.0;

  particle_world_positions.back().x = 3.5;  // One required x-normal node has zero weight and is outside the lattice.
  expect_error(gather(), absl::StatusCode::kInvalidArgument);
  particle_world_positions.back().x = 1.5;
  CudaGatherViews overlapping_outputs = cuda_gather.views();
  overlapping_outputs.velocity_gradients = overlapping_outputs.velocity_fit_slopes;
  expect_error(
      CudaG2P(
          kGrid,
          particle_world_positions,
          old_face_velocities,
          new_face_velocities,
          overlapping_outputs,
          cuda_gather.kernel_measurement
      ),
      absl::StatusCode::kInvalidArgument
  );
  expect_error(
      CudaG2P(
          kGrid,
          cuda_gather.pic_velocities,
          old_face_velocities,
          new_face_velocities,
          cuda_gather.views(),
          cuda_gather.kernel_measurement
      ),
      absl::StatusCode::kInvalidArgument
  );
  expect_error(
      CudaG2P(
          kGrid,
          particle_world_positions,
          {},
          new_face_velocities,
          cuda_gather.views(),
          cuda_gather.kernel_measurement
      ),
      absl::StatusCode::kInvalidArgument
  );
  expect_error(
      CudaG2P(
          kGrid,
          particle_world_positions,
          old_face_velocities,
          new_face_velocities,
          {},
          cuda_gather.kernel_measurement
      ),
      absl::StatusCode::kInvalidArgument
  );
  const std::array<Vec2, 1025> oversized_particle_positions{};
  expect_error(
      CudaG2P(
          kGrid,
          oversized_particle_positions,
          old_face_velocities,
          new_face_velocities,
          {},
          cuda_gather.kernel_measurement
      ),
      absl::StatusCode::kResourceExhausted
  );
  expect_error(
      CudaG2P(Grid2D{64, 64, 1.0}, {}, {}, {}, {}, cuda_gather.kernel_measurement),
      absl::StatusCode::kResourceExhausted
  );

  const auto empty_status =
      CudaG2P(kGrid, {}, old_face_velocities, new_face_velocities, {}, cuda_gather.kernel_measurement);
  EXPECT_TRUE(empty_status.ok()) << empty_status;
  expect_preserved();

  // h*h is the smallest subnormal, so every weighted centered moment term rounds to zero.
  constexpr double kTinySpacing = 0x1p-537;
  constexpr Grid2D kSingularMomentGrid{4, 4, kTinySpacing};
  constexpr std::array<Vec2, 2> singular_moment_particle_positions{{
      {2.5 * kTinySpacing, 2.5 * kTinySpacing},
      {1.5 * kTinySpacing, 1.5 * kTinySpacing},
  }};
  expect_error(
      CudaG2P(
          kSingularMomentGrid,
          singular_moment_particle_positions,
          old_face_velocities,
          new_face_velocities,
          cuda_gather.views(),
          cuda_gather.kernel_measurement
      ),
      absl::StatusCode::kFailedPrecondition
  );

  // At this origin the first three x-normal columns have world offsets (-4, 0, 0)
  // from the particle. The first column's x weight is 2^-105, so its positive
  // centered x moment is about 3.94e-31 versus a y moment of about 1.78.
  constexpr double kNearTieSpacing = 0x1.5555555555556p+1;  // Next double above rounded 8/3.
  constexpr Grid2D kIllConditionedMomentGrid{4, 4, kNearTieSpacing, {0x1p54, 0.0}};
  constexpr Vec2 kIllConditionedParticlePosition{kIllConditionedMomentGrid.origin.x + 4.0, 2.0 * kNearTieSpacing};
  constexpr std::array ill_conditioned_particle_positions{
      kIllConditionedParticlePosition,
      kIllConditionedParticlePosition,
  };
  expect_error(
      CudaG2P(
          kIllConditionedMomentGrid,
          ill_conditioned_particle_positions,
          old_face_velocities,
          new_face_velocities,
          cuda_gather.views(),
          cuda_gather.kernel_measurement
      ),
      absl::StatusCode::kFailedPrecondition
  );

  // The second particle adds two gradient terms of 0.75 * max, which overflow
  // the FP64 derivative sum despite finite samples and a representable PIC value.
  constexpr std::size_t kRightXNormalFaceIndex = 7;
  new_face_velocities[kInteriorXNormalFaceIndex] = -std::numeric_limits<double>::max();
  new_face_velocities[kRightXNormalFaceIndex] = std::numeric_limits<double>::max();
  expect_error(gather(), absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace fluidity
