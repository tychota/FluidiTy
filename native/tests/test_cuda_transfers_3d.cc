#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "fixture_data.h"
#include "fluidity/host/cuda_transfers_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr UniformMacGeometry3D kUnitGeometry{{0.0, 0.0, 0.0}, 1.0};
constexpr UniformMacGeometry3D kTranslatedGeometry{{1.0, -3.0, 5.0}, 2.0};
constexpr std::array kCornerBlocks{
    BlockIndex3D{0, -1, 0},
    BlockIndex3D{1, -1, 0},
    BlockIndex3D{0, 0, 0},
    BlockIndex3D{1, 0, 0},
    BlockIndex3D{0, -1, 1},
    BlockIndex3D{1, -1, 1},
    BlockIndex3D{0, 0, 1},
    BlockIndex3D{1, 0, 1},
};
constexpr std::array kFaceAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
constexpr Vec3 kCornerPosition{3.75, -0.25, 3.75};
constexpr Vec3 kParticleVelocity{1.0, -2.0, 4.0};
constexpr Vec3 kOldVelocity{2.0, 3.0, -4.0};
constexpr Mat3 kAffineRows{
    .x_row = {2.0, 4.0, -1.0},
    .y_row = {-5.0, 1.0, 5.0},
    .z_row = {7.0, -2.0, 3.0},
};
constexpr CudaOperatorMeasurement kMeasurementSentinel{-4, -7.0F};
constexpr MacFaceDepositQuanta3D kFaceSentinel{17, -18, 19};
constexpr MacParticleGatherResult3D kGatherSentinel{
    .new_pic_velocity = {31.0, 32.0, 33.0},
    .grid_velocity_increment = {34.0, 35.0, 36.0},
    .fitted_velocity_slopes = {{37.0, 38.0, 39.0}, {40.0, 41.0, 42.0}, {43.0, 44.0, 45.0}},
    .velocity_gradient = {{46.0, 47.0, 48.0}, {49.0, 50.0, 51.0}, {52.0, 53.0, 54.0}},
};

MacTransferParticle3D CornerParticle(UniformMacGeometry3D geometry) {
  const Vec3 particle_world_position{
      .x = geometry.world_origin.x + geometry.cell_width * kCornerPosition.x,
      .y = geometry.world_origin.y + geometry.cell_width * kCornerPosition.y,
      .z = geometry.world_origin.z + geometry.cell_width * kCornerPosition.z,
  };
  return {particle_world_position, particle_world_position, 2.0, kParticleVelocity, kAffineRows};
}

std::array<double, 3> Components(Vec3 vector) {
  return {vector.x, vector.y, vector.z};
}

void ExpectQualifiedDevice() {
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  EXPECT_EQ(properties.major, 12);
  EXPECT_EQ(properties.minor, 0);
#if defined(FLUIDITY_QUALIFIED_CUDA_DEVICE)
  EXPECT_STREQ(properties.name, FLUIDITY_QUALIFIED_CUDA_DEVICE);
#endif
}

void ExpectMeasurement(CudaOperatorMeasurement measurement) {
  EXPECT_EQ(measurement.compiled_architecture, 1200);
  EXPECT_TRUE(std::isfinite(measurement.kernel_ms));
  EXPECT_GE(measurement.kernel_ms, 0.0F);
}

void ExpectMeasurementSentinel(CudaOperatorMeasurement measurement) {
  EXPECT_EQ(measurement.compiled_architecture, kMeasurementSentinel.compiled_architecture);
  EXPECT_EQ(measurement.kernel_ms, kMeasurementSentinel.kernel_ms);
}

void PrintMeasurement(std::string_view operator_name, CudaOperatorMeasurement measurement) {
  std::cout << "operator=" << operator_name << " compiled_architecture=" << measurement.compiled_architecture
            << " kernel_ms=" << std::fixed << std::setprecision(9) << measurement.kernel_ms << '\n';
}

void ExpectQuanta(MacFaceDepositQuanta3D actual, MacFaceDepositQuanta3D expected) {
  EXPECT_EQ(actual.mass_quanta_sum, expected.mass_quanta_sum);
  EXPECT_EQ(actual.pure_momentum_quanta_sum, expected.pure_momentum_quanta_sum);
  EXPECT_EQ(actual.stress_force_quanta_sum, expected.stress_force_quanta_sum);
}

void ExpectQuanta(std::span<const MacFaceDepositQuanta3D> actual, std::span<const MacFaceDepositQuanta3D> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t face_slot = 0; face_slot < actual.size(); ++face_slot) {
    SCOPED_TRACE(face_slot);
    ExpectQuanta(actual[face_slot], expected[face_slot]);
  }
}

void ExpectNear(Vec3 actual, Vec3 expected, double tolerance) {
  EXPECT_NEAR(actual.x, expected.x, tolerance);
  EXPECT_NEAR(actual.y, expected.y, tolerance);
  EXPECT_NEAR(actual.z, expected.z, tolerance);
}

void ExpectNear(Mat3 actual, Mat3 expected, double tolerance) {
  ExpectNear(actual.x_row, expected.x_row, tolerance);
  ExpectNear(actual.y_row, expected.y_row, tolerance);
  ExpectNear(actual.z_row, expected.z_row, tolerance);
}

void ExpectGather(MacParticleGatherResult3D actual, MacParticleGatherResult3D expected, double tolerance = 2e-12) {
  ExpectNear(actual.new_pic_velocity, expected.new_pic_velocity, tolerance);
  ExpectNear(actual.grid_velocity_increment, expected.grid_velocity_increment, tolerance);
  ExpectNear(actual.fitted_velocity_slopes, expected.fitted_velocity_slopes, tolerance);
  ExpectNear(actual.velocity_gradient, expected.velocity_gradient, tolerance);
}

// Sample the manufactured velocity fields at every owned physical anchor.
template <typename Field>
void SampleField(const SparseMacGrid3D& mac_grid, Field field, std::span<double> face_velocities) {
  for (const auto block : mac_grid.active_block_indices()) {
    for (std::size_t axis_index = 0; axis_index < kFaceAxes.size(); ++axis_index) {
      for (std::int32_t local_z = 0; local_z < 4; ++local_z) {
        for (std::int32_t local_y = 0; local_y < 4; ++local_y) {
          for (std::int32_t local_x = 0; local_x < 4; ++local_x) {
            const FaceIndex3D face{
                .axis = kFaceAxes[axis_index],
                .lattice_index = {4 * block.x + local_x, 4 * block.y + local_y, 4 * block.z + local_z},
            };
            const auto face_slot = mac_grid.FaceSlot(face);
            ASSERT_TRUE(face_slot.ok()) << face_slot.status();
            const auto face_anchor = mac_grid.FaceWorldAnchor(face);
            ASSERT_TRUE(face_anchor.ok()) << face_anchor.status();
            face_velocities[*face_slot] = field(axis_index, *face_anchor);
          }
        }
      }
    }
  }
}

TEST(CudaMacTransfer3D, DepositCornerMatchesOracleAndCpu) {
  ASSERT_NO_FATAL_FAILURE(ExpectQualifiedDevice());
  const auto mac_grid = SparseMacGrid3D::Create(kUnitGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  const auto particle = CornerParticle(kUnitGeometry);
  // Every contribution in this dyadic fixture is an exact multiple of the scale.
  constexpr double scale = 0x1p-20;
  constexpr MacDepositQuantization3D quantization{scale, scale, 3};
  std::vector<MacFaceDepositQuanta3D> gpu_face_quanta(mac_grid->face_count(), kFaceSentinel);
  std::vector<MacFaceDepositQuanta3D> cpu_face_quanta(mac_grid->face_count());
  CudaOperatorMeasurement measurement = kMeasurementSentinel;
  const auto status = CudaDepositMacQuanta3D(*mac_grid, {&particle, 1}, quantization, gpu_face_quanta, measurement);
  ASSERT_TRUE(status.ok()) << status;
  ExpectMeasurement(measurement);
  const auto corner_measurement = measurement;
  ASSERT_TRUE(DepositMacQuanta3D(*mac_grid, {&particle, 1}, quantization, cpu_face_quanta).ok());
  ExpectQuanta(gpu_face_quanta, cpu_face_quanta);

  const auto cases = fixtures::AllFixtures();
  const auto fixture = std::ranges::find(cases, "uniform_mac_transfer_3d", &fixtures::FixtureRecord::name);
  ASSERT_NE(fixture, cases.end());
  const auto face_indices = fixtures::FindArray(*fixture, "face_indices");
  const auto expected_mass = fixtures::FindArray(*fixture, "expected_mass");
  const auto expected_pure_momentum = fixtures::FindArray(*fixture, "expected_pure_momentum");
  ASSERT_TRUE(face_indices.ok()) << face_indices.status();
  ASSERT_TRUE(expected_mass.ok()) << expected_mass.status();
  ASSERT_TRUE(expected_pure_momentum.ok()) << expected_pure_momentum.status();
  constexpr std::size_t kSampleCount = 81;
  ASSERT_EQ((*face_indices)->i64_data.size(), 4 * kSampleCount);
  ASSERT_EQ((*expected_mass)->f64_data.size(), kSampleCount);
  ASSERT_EQ((*expected_pure_momentum)->f64_data.size(), kSampleCount);
  std::vector<MacFaceDepositQuanta3D> oracle_face_quanta(mac_grid->face_count());
  for (std::size_t sample_index = 0; sample_index < kSampleCount; ++sample_index) {
    const auto index = (*face_indices)->i64_data.subspan(4 * sample_index, 4);
    ASSERT_GE(index[0], 0);
    ASSERT_LT(index[0], 3);
    const FaceIndex3D face{
        .axis = kFaceAxes[static_cast<std::size_t>(index[0])],
        .lattice_index =
            {
                static_cast<std::int32_t>(index[1]),
                static_cast<std::int32_t>(index[2]),
                static_cast<std::int32_t>(index[3]),
            },
    };
    const auto face_slot = mac_grid->FaceSlot(face);
    ASSERT_TRUE(face_slot.ok()) << face_slot.status();
    const double mass_quanta = (*expected_mass)->f64_data[sample_index] / scale;
    const double pure_momentum_quanta = (*expected_pure_momentum)->f64_data[sample_index] / scale;
    ASSERT_EQ(mass_quanta, std::trunc(mass_quanta));
    ASSERT_EQ(pure_momentum_quanta, std::trunc(pure_momentum_quanta));
    oracle_face_quanta[*face_slot] = {
        .mass_quanta_sum = static_cast<std::int64_t>(mass_quanta),
        .pure_momentum_quanta_sum = static_cast<std::int64_t>(pure_momentum_quanta),
        .stress_force_quanta_sum = 0,
    };
  }
  ExpectQuanta(gpu_face_quanta, oracle_face_quanta);

  // At the selected face the three weights are 1/2, so each mass is one
  // quantum. Signed ties (1.5, 1.5, -4.5) round to (2, 2, -4).
  constexpr Vec3 tie_position{3.5, 0.0, 4.0};
  const auto tie_slot = mac_grid->FaceSlot({FaceNormalAxis::kX, {3, -1, 3}});
  ASSERT_TRUE(tie_slot.ok()) << tie_slot.status();
  std::array<MacTransferParticle3D, 3> ties{{
      {tie_position, tie_position, 8.0, {1.5, 0.0, 0.0}, {}},
      {tie_position, tie_position, 8.0, {1.5, 0.0, 0.0}, {}},
      {tie_position, tie_position, 8.0, {-4.5, 0.0, 0.0}, {}},
  }};
  constexpr MacDepositQuantization3D tie_quantization{1.0, 1.0, ties.size()};
  ASSERT_TRUE(DepositMacQuanta3D(*mac_grid, ties, tie_quantization, cpu_face_quanta).ok());
  ASSERT_TRUE(CudaDepositMacQuanta3D(*mac_grid, ties, tie_quantization, gpu_face_quanta, measurement).ok());
  ExpectQuanta(gpu_face_quanta[*tie_slot], {3, 0, 0});
  ExpectQuanta(gpu_face_quanta, cpu_face_quanta);
  std::reverse(ties.begin(), ties.end());
  ASSERT_TRUE(CudaDepositMacQuanta3D(*mac_grid, ties, tie_quantization, gpu_face_quanta, measurement).ok());
  ExpectQuanta(gpu_face_quanta, cpu_face_quanta);

  PrintMeasurement("p2g_3d", corner_measurement);
}

TEST(CudaMacTransfer3D, GatherCornerMatchesOracleAndCpu) {
  ASSERT_NO_FATAL_FAILURE(ExpectQualifiedDevice());
  const auto mac_grid = SparseMacGrid3D::Create(kTranslatedGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  auto particle = CornerParticle(mac_grid->geometry());
  const Vec3 trial_position = particle.trial_world_position;
  particle.particle_mass = std::numeric_limits<double>::quiet_NaN();  // Deposit-only data is irrelevant.
  std::vector<double> old_face_velocities(mac_grid->face_count());
  std::vector<double> new_face_velocities(mac_grid->face_count());
  SampleField(
      *mac_grid,
      [](std::size_t axis_index, Vec3) {
        return Components(kOldVelocity)[axis_index];
      },
      old_face_velocities
  );
  SampleField(
      *mac_grid,
      [trial_position](std::size_t axis_index, Vec3 anchor) {
        const std::array velocity_slope_rows{kAffineRows.x_row, kAffineRows.y_row, kAffineRows.z_row};
        const Vec3 row = velocity_slope_rows[axis_index];
        return Components(kParticleVelocity)[axis_index] + row.x * (anchor.x - trial_position.x)
            + row.y * (anchor.y - trial_position.y) + row.z * (anchor.z - trial_position.z);
      },
      new_face_velocities
  );
  MacParticleGatherResult3D gpu_particle_gather = kGatherSentinel;
  MacParticleGatherResult3D cpu_particle_gather{};
  CudaOperatorMeasurement measurement = kMeasurementSentinel;
  const auto status = CudaGatherMac3D(
      *mac_grid,
      {&particle, 1},
      old_face_velocities,
      new_face_velocities,
      {&gpu_particle_gather, 1},
      measurement
  );
  ASSERT_TRUE(status.ok()) << status;
  ExpectMeasurement(measurement);
  const auto corner_measurement = measurement;
  ASSERT_TRUE(
      GatherMac3D(*mac_grid, {&particle, 1}, old_face_velocities, new_face_velocities, {&cpu_particle_gather, 1}).ok()
  );
  ExpectGather(gpu_particle_gather, cpu_particle_gather);
  ExpectGather(gpu_particle_gather, {kParticleVelocity, {-1.0, -5.0, 8.0}, kAffineRows, kAffineRows});

  const double half_cell = 0.5 * mac_grid->geometry().cell_width;
  particle.sorted_world_position = {
      .x = trial_position.x - half_cell,
      .y = trial_position.y + half_cell,
      .z = trial_position.z - half_cell,
  };
  std::ranges::fill(old_face_velocities, 0.0);
  SampleField(
      *mac_grid,
      [trial_position](std::size_t axis_index, Vec3 anchor) {
        const double displacement = Components(anchor)[axis_index] - Components(trial_position)[axis_index];
        return displacement * displacement;
      },
      new_face_velocities
  );
  ASSERT_TRUE(CudaGatherMac3D(
                  *mac_grid,
                  {&particle, 1},
                  old_face_velocities,
                  new_face_velocities,
                  {&gpu_particle_gather, 1},
                  measurement
  )
                  .ok());
  ASSERT_TRUE(
      GatherMac3D(*mac_grid, {&particle, 1}, old_face_velocities, new_face_velocities, {&cpu_particle_gather, 1}).ok()
  );
  ExpectGather(gpu_particle_gather, cpu_particle_gather);
  constexpr MacParticleGatherResult3D squared_field_oracle{
      .new_pic_velocity = {1.0, 1.0, 1.0},
      .grid_velocity_increment = {1.0, 1.0, 1.0},
      .fitted_velocity_slopes = {{-0.375, 0.0, 0.0}, {0.0, -0.375, 0.0}, {0.0, 0.0, -0.375}},
      .velocity_gradient = {},
  };
  ExpectGather(gpu_particle_gather, squared_field_oracle);
  PrintMeasurement("g2p_3d", corner_measurement);
}

TEST(CudaMacTransfer3D, RejectionPreservesOutputs) {
  ASSERT_NO_FATAL_FAILURE(ExpectQualifiedDevice());
  const auto mac_grid = SparseMacGrid3D::Create(kUnitGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  const auto particle = CornerParticle(kUnitGeometry);
  auto excessive_particle = particle;
  excessive_particle.particle_velocity.x = 0x1p64;
  const std::array particles{particle, excessive_particle};
  std::vector<MacFaceDepositQuanta3D> quanta(mac_grid->face_count(), kFaceSentinel);
  CudaOperatorMeasurement measurement = kMeasurementSentinel;
  const auto range_status = CudaDepositMacQuanta3D(*mac_grid, particles, {1.0, 1.0, 2}, quanta, measurement);
  EXPECT_EQ(range_status.code(), absl::StatusCode::kOutOfRange) << range_status;
  // The finite input passes host validation. Only the downloaded device error
  // can produce this diagnostic, after other lanes may have updated scratch.
  EXPECT_TRUE(range_status.message().starts_with("CUDA 3D deposit kernel")) << range_status;
  for (const auto face : quanta) {
    ExpectQuanta(face, kFaceSentinel);
  }
  ExpectMeasurementSentinel(measurement);

  const auto incomplete_grid =
      SparseMacGrid3D::Create(kUnitGeometry, std::span(kCornerBlocks).first(7), kCornerBlocks.size());
  ASSERT_TRUE(incomplete_grid.ok()) << incomplete_grid.status();
  auto partial_quanta = std::span(quanta).first(incomplete_grid->face_count());
  const auto missing_deposit =
      CudaDepositMacQuanta3D(*incomplete_grid, {&particle, 1}, {1.0, 1.0, 1}, partial_quanta, measurement);
  EXPECT_EQ(missing_deposit.code(), absl::StatusCode::kFailedPrecondition) << missing_deposit;
  for (const auto face : quanta) {
    ExpectQuanta(face, kFaceSentinel);
  }
  ExpectMeasurementSentinel(measurement);

  MacParticleGatherResult3D gathered = kGatherSentinel;
  const std::vector<double> missing_field(incomplete_grid->face_count(), 1.0);
  const auto missing_gather =
      CudaGatherMac3D(*incomplete_grid, {&particle, 1}, missing_field, missing_field, {&gathered, 1}, measurement);
  EXPECT_EQ(missing_gather.code(), absl::StatusCode::kFailedPrecondition) << missing_gather;
  ExpectGather(gathered, kGatherSentinel, 0.0);
  ExpectMeasurementSentinel(measurement);

  const std::vector<double> finite_field(mac_grid->face_count(), 0.0);
  auto nonfinite_field = finite_field;
  nonfinite_field.back() = std::numeric_limits<double>::infinity();
  EXPECT_EQ(
      CudaGatherMac3D(*mac_grid, {}, finite_field, nonfinite_field, {}, measurement).code(),
      absl::StatusCode::kInvalidArgument
  );
  ASSERT_TRUE(CudaGatherMac3D(*mac_grid, {}, finite_field, finite_field, {}, measurement).ok());
  ExpectMeasurementSentinel(measurement);

  const auto empty_deposit = CudaDepositMacQuanta3D(*mac_grid, {}, {1.0, 1.0, 0}, quanta, measurement);
  ASSERT_TRUE(empty_deposit.ok()) << empty_deposit;
  ExpectMeasurement(measurement);
  std::cout << "empty_deposit_3d compiled_architecture=" << measurement.compiled_architecture
            << " kernel_ms=" << std::fixed << std::setprecision(9) << measurement.kernel_ms << '\n';
  for (const auto face : quanta) {
    ExpectQuanta(face, {});
  }
}

}  // namespace
}  // namespace fluidity
