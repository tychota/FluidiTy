#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "absl/status/status.h"
#include "fixture_data.h"
#include "fluidity/host/transfers_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr UniformMacGeometry3D kUnitGeometry{
    .world_origin = {0.0, 0.0, 0.0},
    .cell_width = 1.0,
};
constexpr UniformMacGeometry3D kTranslatedGeometry{
    .world_origin = {1.0, -3.0, 5.0},
    .cell_width = 2.0,
};
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
constexpr std::size_t kFaceCount = 192 * kCornerBlocks.size();
constexpr Vec3 kCornerPosition{3.75, -0.25, 3.75};
constexpr Vec3 kParticleVelocity{1.0, -2.0, 4.0};
constexpr Vec3 kOldVelocity{2.0, 3.0, -4.0};
constexpr Mat3 kAffineRows{
    .x_row = {2.0, 4.0, -1.0},
    .y_row = {-5.0, 1.0, 5.0},
    .z_row = {7.0, -2.0, 3.0},
};
constexpr MacDepositQuantization3D kUnitQuanta{1.0, 1.0, 1};
constexpr Vec3 kTiePosition{3.5, 0.0, 4.0};
constexpr FaceIndex3D kTieFace{FaceNormalAxis::kX, {3, -1, 3}};

std::array<double, 3> Components(Vec3 value) {
  return {value.x, value.y, value.z};
}

void ExpectNear(double actual, double expected) {
  EXPECT_NEAR(actual, expected, 2e-12 + 2e-12 * std::abs(expected));
}

void ExpectNear(Vec3 actual, Vec3 expected) {
  ExpectNear(actual.x, expected.x);
  ExpectNear(actual.y, expected.y);
  ExpectNear(actual.z, expected.z);
}

void ExpectNear(Mat3 actual, Mat3 expected) {
  ExpectNear(actual.x_row, expected.x_row);
  ExpectNear(actual.y_row, expected.y_row);
  ExpectNear(actual.z_row, expected.z_row);
}

MacTransferParticle3D CornerParticle(UniformMacGeometry3D geometry) {
  const Vec3 particle_world_position{
      .x = geometry.world_origin.x + geometry.cell_width * kCornerPosition.x,
      .y = geometry.world_origin.y + geometry.cell_width * kCornerPosition.y,
      .z = geometry.world_origin.z + geometry.cell_width * kCornerPosition.z,
  };
  return {particle_world_position, particle_world_position, 2.0, kParticleVelocity, kAffineRows};
}

MacTransferParticle3D TieParticle(double mass, double normal_velocity) {
  return {kTiePosition, kTiePosition, mass, {normal_velocity, 0.0, 0.0}, {}};
}

// Evaluate the manufactured field at every owned physical anchor.
template <typename Field>
void SampleField(const SparseMacGrid3D& mac_grid, Field field, std::span<double> values) {
  for (const auto block : mac_grid.active_block_indices()) {
    for (std::size_t axis_index = 0; axis_index < kFaceAxes.size(); ++axis_index) {
      for (std::int32_t local_z = 0; local_z < 4; ++local_z) {
        for (std::int32_t local_y = 0; local_y < 4; ++local_y) {
          for (std::int32_t local_x = 0; local_x < 4; ++local_x) {
            const FaceIndex3D face{
                .axis = kFaceAxes[axis_index],
                .lattice_index = {4 * block.x + local_x, 4 * block.y + local_y, 4 * block.z + local_z},
            };
            const auto resolved_face_slot = mac_grid.FaceSlot(face);
            ASSERT_TRUE(resolved_face_slot.ok()) << resolved_face_slot.status();
            const auto anchor = mac_grid.FaceWorldAnchor(face);
            ASSERT_TRUE(anchor.ok()) << anchor.status();
            values[*resolved_face_slot] = field(axis_index, *anchor);
          }
        }
      }
    }
  }
}

void ExpectAffineGather(const SparseMacGrid3D& mac_grid) {
  auto particle = CornerParticle(mac_grid.geometry());
  const Vec3 particle_world_position = particle.trial_world_position;
  std::vector<double> old_velocities(mac_grid.face_count());
  std::vector<double> new_velocities(mac_grid.face_count());
  SampleField(
      mac_grid,
      [](std::size_t axis_index, Vec3) {
        return Components(kOldVelocity)[axis_index];
      },
      old_velocities
  );
  SampleField(
      mac_grid,
      [particle_world_position](std::size_t axis_index, Vec3 anchor) {
        const std::array rows{kAffineRows.x_row, kAffineRows.y_row, kAffineRows.z_row};
        const Vec3 row = rows[axis_index];
        return Components(kParticleVelocity)[axis_index] + row.x * (anchor.x - particle_world_position.x)
            + row.y * (anchor.y - particle_world_position.y) + row.z * (anchor.z - particle_world_position.z);
      },
      new_velocities
  );
  // Gather does not inspect deposit-only data.
  particle.particle_mass = std::numeric_limits<double>::quiet_NaN();
  particle.particle_velocity.x = std::numeric_limits<double>::infinity();
  particle.particle_velocity_slopes.z_row.y = std::numeric_limits<double>::infinity();
  MacParticleGatherResult3D gathered{};
  const auto status = GatherMac3D(mac_grid, {&particle, 1}, old_velocities, new_velocities, {&gathered, 1});
  ASSERT_TRUE(status.ok()) << status;
  ExpectNear(gathered.new_pic_velocity, kParticleVelocity);
  ExpectNear(gathered.grid_velocity_increment, Vec3{-1.0, -5.0, 8.0});
  ExpectNear(gathered.fitted_velocity_slopes, kAffineRows);
  ExpectNear(gathered.velocity_gradient, kAffineRows);
}

TEST(MacTransfer3D, AffineCornerMatchesOracle) {
  const auto mac_grid = SparseMacGrid3D::Create(kUnitGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  const auto particle = CornerParticle(kUnitGeometry);
  std::vector<MacFaceDeposit3D> physical_face_deposits(mac_grid->face_count(), {42.0, -42.0});
  const auto status = DepositMac3D(*mac_grid, {&particle, 1}, physical_face_deposits);
  ASSERT_TRUE(status.ok()) << status;

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
  ASSERT_EQ((*face_indices)->i64_data.size(), kSampleCount * 4);
  ASSERT_EQ((*expected_mass)->f64_data.size(), kSampleCount);
  ASSERT_EQ((*expected_pure_momentum)->f64_data.size(), kSampleCount);

  std::vector<bool> fixture_slots(mac_grid->face_count());
  for (std::size_t sample_index = 0; sample_index < kSampleCount; ++sample_index) {
    SCOPED_TRACE(sample_index);
    const auto index = (*face_indices)->i64_data.subspan(sample_index * 4, 4);
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
    const auto resolved_face_slot = mac_grid->FaceSlot(face);
    ASSERT_TRUE(resolved_face_slot.ok()) << resolved_face_slot.status();
    EXPECT_FALSE(fixture_slots[*resolved_face_slot]);
    fixture_slots[*resolved_face_slot] = true;
    ExpectNear(physical_face_deposits[*resolved_face_slot].deposited_mass, (*expected_mass)->f64_data[sample_index]);
    ExpectNear(
        physical_face_deposits[*resolved_face_slot].deposited_pure_momentum,
        (*expected_pure_momentum)->f64_data[sample_index]
    );
  }
  for (std::size_t face_slot = 0; face_slot < physical_face_deposits.size(); ++face_slot) {
    if (!fixture_slots[face_slot]) {
      EXPECT_EQ(physical_face_deposits[face_slot].deposited_mass, 0.0);
      EXPECT_EQ(physical_face_deposits[face_slot].deposited_pure_momentum, 0.0);
    }
  }
  ExpectAffineGather(*mac_grid);

  const auto translated_grid = SparseMacGrid3D::Create(kTranslatedGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(translated_grid.ok()) << translated_grid.status();
  ExpectAffineGather(*translated_grid);
}

TEST(MacTransfer3D, QuadraticRowDistinguishesFitAndGradient) {
  for (const auto geometry : {kUnitGeometry, kTranslatedGeometry}) {
    SCOPED_TRACE(geometry.cell_width);
    const auto mac_grid = SparseMacGrid3D::Create(geometry, kCornerBlocks, kCornerBlocks.size());
    ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
    auto particle = CornerParticle(geometry);
    const Vec3 particle_world_position = particle.trial_world_position;
    const double half_cell = 0.5 * geometry.cell_width;
    particle.sorted_world_position = {
        .x = particle_world_position.x - half_cell,
        .y = particle_world_position.y + half_cell,
        .z = particle_world_position.z - half_cell,
    };
    const std::vector<double> old_velocities(mac_grid->face_count());
    std::vector<double> new_velocities(mac_grid->face_count());
    SampleField(
        *mac_grid,
        [particle_world_position](std::size_t axis_index, Vec3 anchor) {
          const double displacement = Components(anchor)[axis_index] - Components(particle_world_position)[axis_index];
          return displacement * displacement;
        },
        new_velocities
    );
    MacParticleGatherResult3D gathered{};
    const auto status = GatherMac3D(*mac_grid, {&particle, 1}, old_velocities, new_velocities, {&gathered, 1});
    ASSERT_TRUE(status.ok()) << status;
    const double expected_pic_velocity = geometry.cell_width == 1.0 ? 0.25 : 1.0;
    const double expected_slope = geometry.cell_width == 1.0 ? -0.1875 : -0.375;
    const Vec3 expected_velocity{expected_pic_velocity, expected_pic_velocity, expected_pic_velocity};
    const Mat3 expected_fitted_velocity_slopes{
        .x_row = {expected_slope, 0.0, 0.0},
        .y_row = {0.0, expected_slope, 0.0},
        .z_row = {0.0, 0.0, expected_slope},
    };
    ExpectNear(gathered.new_pic_velocity, expected_velocity);
    ExpectNear(gathered.grid_velocity_increment, expected_velocity);
    ExpectNear(gathered.fitted_velocity_slopes, expected_fitted_velocity_slopes);
    ExpectNear(gathered.velocity_gradient, Mat3{});
  }
}

TEST(MacTransfer3D, UncutElasticWorkBalancesAtTranslatedCorner) {
  const auto mac_grid = SparseMacGrid3D::Create(kTranslatedGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  const auto particle = CornerParticle(kTranslatedGeometry);
  const Vec3 position = particle.trial_world_position;
  std::vector<double> face_velocities(mac_grid->face_count());
  SampleField(
      *mac_grid,
      [position](std::size_t axis_index, Vec3 anchor) {
        const double displacement = Components(anchor)[axis_index] - Components(position)[axis_index];
        return displacement * displacement + 2.0 * displacement;
      },
      face_velocities
  );
  const Mat3 deformation_gradient{
      .x_row = {2.0, 0.0, 0.0},
      .y_row = {0.0, 1.0, 0.0},
      .z_row = {0.0, 0.0, 1.0},
  };
  const auto stress = NeoHookeanKirchhoffStress3D(deformation_gradient, 2.0, 0.0);
  ASSERT_TRUE(stress.ok()) << stress.status();
  ExpectNear(stress->x_row.x, 6.0);

  constexpr double kReferenceVolume = 0.25;
  std::vector<double> stress_forces(mac_grid->face_count());
  const auto deposition =
      DepositElasticStress3D(*mac_grid, position, position, kReferenceVolume, *stress, stress_forces);
  ASSERT_TRUE(deposition.ok()) << deposition;
  MacParticleGatherResult3D gathered{};
  const auto gather = GatherMac3D(*mac_grid, {&particle, 1}, face_velocities, face_velocities, {&gathered, 1});
  ASSERT_TRUE(gather.ok()) << gather;
  ExpectNear(gathered.velocity_gradient.x_row.x, 2.0);

  double grid_power = 0.0;
  for (std::size_t face_slot = 0; face_slot < stress_forces.size(); ++face_slot) {
    grid_power += face_velocities[face_slot] * stress_forces[face_slot];
  }
  const double deformation_power = kReferenceVolume * stress->x_row.x * gathered.velocity_gradient.x_row.x;
  EXPECT_NEAR(grid_power, -3.0, 3e-6);
  EXPECT_NEAR(
      grid_power + deformation_power,
      0.0,
      1e-6 * std::max({1.0, std::abs(grid_power), std::abs(deformation_power)})
  );
}

TEST(MacTransfer3D, MaskedElasticWorkReportsDerivativeMismatch) {
  const auto mac_grid = SparseMacGrid3D::Create(kTranslatedGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  const Vec3 position = CornerParticle(kTranslatedGeometry).trial_world_position;
  std::vector<double> completed_face_velocities(mac_grid->face_count());
  SampleField(
      *mac_grid,
      [position](std::size_t axis_index, Vec3 anchor) {
        const double displacement = Components(anchor)[axis_index] - Components(position)[axis_index];
        return displacement * displacement + 2.0 * displacement;
      },
      completed_face_velocities
  );
  std::vector<std::uint8_t> compatible_faces(mac_grid->face_count(), 1);
  for (std::size_t face_slot = 0; face_slot < compatible_faces.size(); ++face_slot) {
    const auto face_index = mac_grid->view().FaceIndex(face_slot);
    ASSERT_TRUE(face_index.ok()) << face_index.status();
    if (face_index->axis == FaceNormalAxis::kX && face_index->lattice_index.x < 4) {
      compatible_faces[face_slot] = 0;
    }
  }
  constexpr Mat3 kStress{
      .x_row = {6.0, 0.0, 0.0},
      .y_row = {},
      .z_row = {},
  };
  constexpr double kDt = 0.125;
  constexpr double kVolume = 0.25;
  constexpr double kCompatibleDerivative = 800.0 / 529.0;
  std::vector<double> face_forces(mac_grid->face_count(), 999.0);
  const auto budget = DepositCutElasticStress3D(
      *mac_grid,
      position,
      position,
      kVolume,
      kStress,
      kDt,
      compatible_faces,
      completed_face_velocities,
      face_forces
  );
  ASSERT_TRUE(budget.ok()) << budget.status();
  ExpectNear(budget->complete_velocity_gradient.x_row.x, 2.0);
  ExpectNear(budget->compatible_velocity_gradient.x_row.x, kCompatibleDerivative);
  ExpectNear(budget->cut_work, kDt * kVolume * 6.0 * (2.0 - kCompatibleDerivative));
  ExpectNear(budget->grid_stress_work, -kDt * kVolume * 6.0 * kCompatibleDerivative);
  ExpectNear(budget->stress_impulse, Vec3{});
  ExpectNear(budget->stress_angular_impulse_about_particle, Vec3{});
  for (std::size_t face_slot = 0; face_slot < compatible_faces.size(); ++face_slot) {
    if (compatible_faces[face_slot] == 0) {
      EXPECT_EQ(face_forces[face_slot], 0.0);
    }
  }

  const auto accepted_forces = face_forces;
  std::ranges::fill(compatible_faces, 0);
  const auto rejected = DepositCutElasticStress3D(
      *mac_grid,
      position,
      position,
      kVolume,
      kStress,
      kDt,
      compatible_faces,
      completed_face_velocities,
      face_forces
  );
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(face_forces, accepted_forces);
}

TEST(MacTransfer3D, RoundsContributionsBeforeSummation) {
  const auto mac_grid = SparseMacGrid3D::Create(kUnitGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  const auto resolved_face_slot = mac_grid->FaceSlot(kTieFace);
  ASSERT_TRUE(resolved_face_slot.ok()) << resolved_face_slot.status();
  std::vector<MacFaceDepositQuanta3D> face_deposit_quanta(mac_grid->face_count());

  // At this X face each axis contributes weight 1/2.
  const std::array half_quanta{TieParticle(4.0, 1.0), TieParticle(4.0, 1.0)};
  auto status = DepositMacQuanta3D(*mac_grid, half_quanta, {1.0, 1.0, half_quanta.size()}, face_deposit_quanta);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(face_deposit_quanta[*resolved_face_slot].mass_quanta_sum, 0);
  EXPECT_EQ(face_deposit_quanta[*resolved_face_slot].pure_momentum_quanta_sum, 0);
  EXPECT_EQ(face_deposit_quanta[*resolved_face_slot].stress_force_quanta_sum, 0);

  constexpr std::array tie_velocities{-2.5, -1.5, -0.5, 0.5, 1.5, 2.5};
  constexpr std::array<std::int64_t, 6> expected_quanta{-2, -2, 0, 0, 2, 2};
  std::array<MacTransferParticle3D, 7> particles{};
  for (std::size_t tie_index = 0; tie_index < tie_velocities.size(); ++tie_index) {
    SCOPED_TRACE(tie_velocities[tie_index]);
    particles[tie_index] = TieParticle(8.0, tie_velocities[tie_index]);
    status = DepositMacQuanta3D(*mac_grid, {&particles[tie_index], 1}, kUnitQuanta, face_deposit_quanta);
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ(face_deposit_quanta[*resolved_face_slot].mass_quanta_sum, 1);
    EXPECT_EQ(face_deposit_quanta[*resolved_face_slot].pure_momentum_quanta_sum, expected_quanta[tie_index]);
    EXPECT_EQ(face_deposit_quanta[*resolved_face_slot].stress_force_quanta_sum, 0);
  }
  particles.back() = TieParticle(6.0, -3.0);
  constexpr MacDepositQuantization3D quantization{0.5, 2.0, particles.size()};
  status = DepositMacQuanta3D(*mac_grid, particles, quantization, face_deposit_quanta);
  ASSERT_TRUE(status.ok()) << status;
  std::vector<MacFaceDeposit3D> physical_face_deposits(mac_grid->face_count());
  status = DepositMac3D(*mac_grid, particles, physical_face_deposits);
  ASSERT_TRUE(status.ok()) << status;

  std::reverse(particles.begin(), particles.end());
  std::vector<MacFaceDepositQuanta3D> reversed_face_deposit_quanta(mac_grid->face_count());
  status = DepositMacQuanta3D(*mac_grid, particles, quantization, reversed_face_deposit_quanta);
  ASSERT_TRUE(status.ok()) << status;
  for (std::size_t face_slot = 0; face_slot < physical_face_deposits.size(); ++face_slot) {
    SCOPED_TRACE(face_slot);
    EXPECT_EQ(face_deposit_quanta[face_slot].mass_quanta_sum, reversed_face_deposit_quanta[face_slot].mass_quanta_sum);
    EXPECT_EQ(
        face_deposit_quanta[face_slot].pure_momentum_quanta_sum,
        reversed_face_deposit_quanta[face_slot].pure_momentum_quanta_sum
    );
    EXPECT_EQ(face_deposit_quanta[face_slot].stress_force_quanta_sum, 0);
    const double contributor_count =
        physical_face_deposits[face_slot].deposited_mass > 0.0 ? static_cast<double>(particles.size()) : 0.0;
    const double decoded_mass =
        static_cast<double>(face_deposit_quanta[face_slot].mass_quanta_sum) * quantization.mass_per_quantum;
    const double decoded_momentum = static_cast<double>(face_deposit_quanta[face_slot].pure_momentum_quanta_sum)
        * quantization.pure_momentum_per_quantum;
    EXPECT_LE(
        std::abs(decoded_mass - physical_face_deposits[face_slot].deposited_mass),
        contributor_count * quantization.mass_per_quantum / 2.0
    );
    EXPECT_LE(
        std::abs(decoded_momentum - physical_face_deposits[face_slot].deposited_pure_momentum),
        contributor_count * quantization.pure_momentum_per_quantum / 2.0
    );
  }
}

TEST(MacTransfer3D, RejectsCapacityRangeAndAliasingWithoutPublication) {
  const auto mac_grid = SparseMacGrid3D::Create(kUnitGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  ASSERT_EQ(mac_grid->face_count(), kFaceCount);
  const auto resolved_face_slot = mac_grid->FaceSlot(kTieFace);
  ASSERT_TRUE(resolved_face_slot.ok()) << resolved_face_slot.status();
  const auto invalid_argument = absl::StatusCode::kInvalidArgument;
  const auto capacity_exceeded = absl::StatusCode::kResourceExhausted;
  const auto out_of_range = absl::StatusCode::kOutOfRange;
  constexpr MacFaceDepositQuanta3D kQuantaSentinel{17, -18, 19};
  constexpr MacFaceDeposit3D kDepositSentinel{21.0, -22.0};
  constexpr MacParticleGatherResult3D kGatherSentinel{
      .new_pic_velocity = {31.0, 32.0, 33.0},
      .grid_velocity_increment = {34.0, 35.0, 36.0},
      .fitted_velocity_slopes = {{37.0, 38.0, 39.0}, {40.0, 41.0, 42.0}, {43.0, 44.0, 45.0}},
      .velocity_gradient = {{46.0, 47.0, 48.0}, {49.0, 50.0, 51.0}, {52.0, 53.0, 54.0}},
  };
  std::vector<MacFaceDepositQuanta3D> face_deposit_quanta(kFaceCount, kQuantaSentinel);
  std::vector<MacFaceDeposit3D> physical_face_deposits(kFaceCount, kDepositSentinel);
  std::array gathered{kGatherSentinel, kGatherSentinel};
  const std::vector<double> velocities(kFaceCount, 4.0);

  auto guard_particle = TieParticle(8.0, 0.0);
  // Capacity three reserves four contributors. At the selected face, w*m=1
  // and the normal displacement is -1/2, giving the positive inclusive guard.
  guard_particle.particle_velocity_slopes.x_row.x = -0x1p61;
  std::array particles{guard_particle, guard_particle, guard_particle, guard_particle};
  const std::span<const MacTransferParticle3D> pair(particles.data(), 2);
  const std::span<const MacTransferParticle3D> triple(particles.data(), 3);
  constexpr MacDepositQuantization3D quantization{1.0, 1.0, 3};
  EXPECT_EQ(DepositMacQuanta3D(*mac_grid, pair, {1.0, 1.0, 0}, face_deposit_quanta).code(), capacity_exceeded);
  EXPECT_EQ(DepositMacQuanta3D(*mac_grid, particles, quantization, face_deposit_quanta).code(), capacity_exceeded);
  EXPECT_EQ(
      DepositMacQuanta3D(*mac_grid, {}, {1.0, 1.0, (1u << 20) + 1}, face_deposit_quanta).code(),
      capacity_exceeded
  );
  for (const double invalid_scale :
       {0.0, -1.0, 3.0, std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::infinity()}) {
    EXPECT_EQ(
        DepositMacQuanta3D(*mac_grid, pair, {invalid_scale, 1.0, 3}, face_deposit_quanta).code(),
        invalid_argument
    );
    EXPECT_EQ(
        DepositMacQuanta3D(*mac_grid, pair, {1.0, invalid_scale, 3}, face_deposit_quanta).code(),
        invalid_argument
    );
  }
  particles[2].particle_velocity_slopes.x_row.x =
      std::nextafter(guard_particle.particle_velocity_slopes.x_row.x, -std::numeric_limits<double>::infinity());
  EXPECT_EQ(DepositMacQuanta3D(*mac_grid, triple, quantization, face_deposit_quanta).code(), out_of_range);
  particles[2] = guard_particle;

  for (const double invalid_mass :
       {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    particles[1].particle_mass = invalid_mass;
    EXPECT_EQ(DepositMac3D(*mac_grid, pair, physical_face_deposits).code(), invalid_argument);
    EXPECT_EQ(DepositMacQuanta3D(*mac_grid, pair, quantization, face_deposit_quanta).code(), invalid_argument);
  }
  particles[1] = guard_particle;
  particles[1].trial_world_position.y = std::numeric_limits<double>::infinity();
  EXPECT_EQ(GatherMac3D(*mac_grid, pair, velocities, velocities, gathered).code(), invalid_argument);
  particles[1] = guard_particle;

  const auto wrong_face_extent = std::span(physical_face_deposits).first(kFaceCount - 1);
  const auto wrong_quanta_extent = std::span(face_deposit_quanta).first(kFaceCount - 1);
  const auto wrong_velocity_extent = std::span(velocities).first(kFaceCount - 1);
  EXPECT_EQ(DepositMac3D(*mac_grid, pair, wrong_face_extent).code(), invalid_argument);
  EXPECT_EQ(DepositMacQuanta3D(*mac_grid, pair, quantization, wrong_quanta_extent).code(), invalid_argument);
  EXPECT_EQ(GatherMac3D(*mac_grid, pair, wrong_velocity_extent, velocities, gathered).code(), invalid_argument);
  EXPECT_EQ(GatherMac3D(*mac_grid, pair, velocities, wrong_velocity_extent, gathered).code(), invalid_argument);
  EXPECT_EQ(
      GatherMac3D(*mac_grid, pair, velocities, velocities, std::span(gathered).first(1)).code(),
      invalid_argument
  );

  // Rejected views share an address with an active output member. The input
  // member only supplies an address for the upfront byte-overlap check.
  union OverlappingViews {
    std::array<MacFaceDeposit3D, kFaceCount> physical_face_deposits;
    std::array<MacFaceDepositQuanta3D, kFaceCount> face_deposit_quanta;
    MacParticleGatherResult3D gathered;
    MacTransferParticle3D particle;
  };
  OverlappingViews deposit_alias{.physical_face_deposits = {}};
  OverlappingViews quanta_alias{.face_deposit_quanta = {}};
  OverlappingViews gather_alias{.gathered = kGatherSentinel};
  EXPECT_EQ(
      DepositMac3D(*mac_grid, {&deposit_alias.particle, 1}, deposit_alias.physical_face_deposits).code(),
      invalid_argument
  );
  EXPECT_EQ(
      DepositMacQuanta3D(*mac_grid, {&quanta_alias.particle, 1}, quantization, quanta_alias.face_deposit_quanta).code(),
      invalid_argument
  );
  EXPECT_EQ(
      GatherMac3D(*mac_grid, {&gather_alias.particle, 1}, velocities, velocities, {&gather_alias.gathered, 1}).code(),
      invalid_argument
  );
  for (const auto& face : face_deposit_quanta) {
    EXPECT_EQ(face.mass_quanta_sum, kQuantaSentinel.mass_quanta_sum);
    EXPECT_EQ(face.pure_momentum_quanta_sum, kQuantaSentinel.pure_momentum_quanta_sum);
    EXPECT_EQ(face.stress_force_quanta_sum, kQuantaSentinel.stress_force_quanta_sum);
  }
  for (const auto& face : physical_face_deposits) {
    EXPECT_EQ(face.deposited_mass, kDepositSentinel.deposited_mass);
    EXPECT_EQ(face.deposited_pure_momentum, kDepositSentinel.deposited_pure_momentum);
  }
  for (const auto& result : gathered) {
    ExpectNear(result.new_pic_velocity, kGatherSentinel.new_pic_velocity);
    ExpectNear(result.grid_velocity_increment, kGatherSentinel.grid_velocity_increment);
    ExpectNear(result.fitted_velocity_slopes, kGatherSentinel.fitted_velocity_slopes);
    ExpectNear(result.velocity_gradient, kGatherSentinel.velocity_gradient);
  }

  auto status = DepositMacQuanta3D(*mac_grid, triple, quantization, face_deposit_quanta);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(face_deposit_quanta[*resolved_face_slot].mass_quanta_sum, 3);
  EXPECT_EQ(face_deposit_quanta[*resolved_face_slot].pure_momentum_quanta_sum, std::int64_t{3} << 60);
  status = DepositMacQuanta3D(*mac_grid, {}, {1.0, 1.0, 0}, face_deposit_quanta);
  ASSERT_TRUE(status.ok()) << status;
  status = DepositMac3D(*mac_grid, {}, physical_face_deposits);
  ASSERT_TRUE(status.ok()) << status;
  for (std::size_t face_slot = 0; face_slot < physical_face_deposits.size(); ++face_slot) {
    EXPECT_EQ(physical_face_deposits[face_slot].deposited_mass, 0.0);
    EXPECT_EQ(physical_face_deposits[face_slot].deposited_pure_momentum, 0.0);
    EXPECT_EQ(face_deposit_quanta[face_slot].mass_quanta_sum, 0);
    EXPECT_EQ(face_deposit_quanta[face_slot].pure_momentum_quanta_sum, 0);
    EXPECT_EQ(face_deposit_quanta[face_slot].stress_force_quanta_sum, 0);
  }
  status = GatherMac3D(*mac_grid, pair, velocities, velocities, gathered);
  ASSERT_TRUE(status.ok()) << status;
  for (const auto& result : gathered) {
    ExpectNear(result.new_pic_velocity, Vec3{4.0, 4.0, 4.0});
    ExpectNear(result.grid_velocity_increment, Vec3{});
  }
}

}  // namespace
}  // namespace fluidity
