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
#include "fluidity/host/elastic_blade_deformation_3d.h"
#include "fluidity/host/liquid_wall_contact_3d.h"
#include "fluidity/host/solid_interaction_checkpoint_3d.h"
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

TEST(MacTransfer3D, NeoHookeanRejectsRoundedSingularDeformation) {
  constexpr Vec3 kRepeatedRow{0.23645552527159452, 0.3692706737201661, 0.5042420323006809};
  constexpr Mat3 kSingular{
      .x_row = kRepeatedRow,
      .y_row = {0.7048832636792213, 0.05054362863302231, 0.3695183543022722},
      .z_row = kRepeatedRow,
  };
  EXPECT_EQ(NeoHookeanKirchhoffStress3D(kSingular, 2.0, 1.0).status().code(), absl::StatusCode::kInvalidArgument);
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
  const TranslatingSlab3D blade{
      .unit_normal = {1.0, 0.0, 0.0},
      .start_center_coordinate = 6.5,
      .half_width = 1.0,
      .wall_velocity = {2.0, 0.0, 0.0},
  };
  const auto invalid_time = BuildElasticBladeFaceTags3D(mac_grid->view(), blade, -0.25);
  EXPECT_EQ(invalid_time.status().code(), absl::StatusCode::kInvalidArgument);
  const auto tags = BuildElasticBladeFaceTags3D(mac_grid->view(), blade, 0.25);
  ASSERT_TRUE(tags.ok()) << tags.status();
  std::vector<std::uint8_t> compatible_faces(mac_grid->face_count());
  for (std::size_t face_slot = 0; face_slot < compatible_faces.size(); ++face_slot) {
    const auto face = mac_grid->view().FaceIndex(face_slot);
    ASSERT_TRUE(face.ok()) << face.status();
    const auto anchor = mac_grid->FaceWorldAnchor(*face);
    ASSERT_TRUE(anchor.ok()) << anchor.status();
    const auto tag = (*tags)[face_slot];
    const bool expected_positive = anchor->x >= 8.0;
    const bool expected_negative = anchor->x <= 6.0;
    EXPECT_EQ(tag.structurally_eligible, expected_positive || expected_negative);
    EXPECT_EQ(tag.side_tag, expected_positive ? 1 : expected_negative ? -1 : 0);
    compatible_faces[face_slot] = tag.structurally_eligible && tag.side_tag == 1;
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

TEST(MacTransfer3D, ElasticDeformationUsesSameSideGhostGradientAndRejectsCollapse) {
  const auto grid = SparseMacGrid3D::Create(kTranslatedGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const auto particle = CornerParticle(kTranslatedGeometry);
  const TranslatingSlab3D blade{
      .unit_normal = {1.0, 0.0, 0.0},
      .start_center_coordinate = 6.5,
      .half_width = 1.0,
      .wall_velocity = {2.0, 0.0, 0.0},
  };
  const auto old_tags = BuildElasticBladeFaceTags3D(grid->view(), blade, 0.0);
  const auto new_tags = BuildElasticBladeFaceTags3D(grid->view(), blade, 0.25);
  ASSERT_TRUE(old_tags.ok()) << old_tags.status();
  ASSERT_TRUE(new_tags.ok()) << new_tags.status();
  std::vector<double> old_velocity(grid->face_count());
  std::vector<double> new_velocity(grid->face_count());
  SampleField(
      *grid,
      [](std::size_t axis_index, Vec3 anchor) {
        return axis_index == 0 ? 2.0 * anchor.x + anchor.y : 0.0;
      },
      new_velocity
  );
  for (std::size_t face_slot = 0; face_slot < grid->face_count(); ++face_slot) {
    if ((*new_tags)[face_slot].side_tag != 1) {
      new_velocity[face_slot] = 1000.0;
    }
    if ((*old_tags)[face_slot].side_tag != 1) {
      old_velocity[face_slot] = -1000.0;
    }
  }
  const std::vector validity(grid->face_count(), LiquidFaceValidity3D::kValid);
  const std::vector<LiquidPrescribedWallSample3D> walls(grid->face_count());
  const LiquidBoundaryVelocityHistory3D old_history{0.0, 0.0, old_velocity, validity, *old_tags, walls};
  const LiquidBoundaryVelocityHistory3D new_history{0.25, 0.25, new_velocity, validity, *new_tags, walls};
  constexpr Mat3 kDeformation{
      .x_row = {1.0, 0.0, 0.0},
      .y_row = {0.0, 2.0, 0.0},
      .z_row = {0.0, 0.0, 1.0},
  };
  const ElasticBladeSide3D side{.blade_id = 17, .sign = 1};
  const auto result = GatherElasticBladeDeformation3D(*grid, particle, side, old_history, new_history, kDeformation);
  ASSERT_TRUE(result.ok()) << result.status();
  ExpectNear(result->gather.velocity_gradient.x_row, {2.0, 1.0, 0.0});
  ExpectNear(result->deformation_gradient.x_row, {1.5, 0.5, 0.0});

  SolidInteractionCheckpoint3D checkpoint{
      .accepted_step = 1,
      .sides = {{.particle_id = 7, .side = side}},
      .elastic_particles = {{
          .particle_id = 7,
          .body_id = 23,
          .transfer = particle,
          .reference_volume = 0.25,
          .deformation_gradient = result->deformation_gradient,
      }},
  };
  const auto encoded = EncodeSolidInteractionCheckpoint3D(checkpoint);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto restored = DecodeSolidInteractionCheckpoint3D(*encoded);
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_EQ(restored->elastic_particles.size(), 1);
  ASSERT_EQ(restored->sides.size(), 1);
  const auto& restored_particle = restored->elastic_particles.front();
  EXPECT_EQ(restored_particle.body_id, 23);
  EXPECT_DOUBLE_EQ(restored_particle.reference_volume, 0.25);
  ExpectNear(restored_particle.transfer.trial_world_position, particle.trial_world_position);
  ExpectNear(restored_particle.transfer.particle_velocity_slopes, particle.particle_velocity_slopes);
  const auto uninterrupted =
      GatherElasticBladeDeformation3D(*grid, particle, side, old_history, new_history, result->deformation_gradient);
  ASSERT_TRUE(uninterrupted.ok()) << uninterrupted.status();
  const auto replayed = GatherElasticBladeDeformation3D(
      *grid,
      restored_particle.transfer,
      restored->sides.front().side,
      old_history,
      new_history,
      restored_particle.deformation_gradient
  );
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  checkpoint.elastic_particles.front().deformation_gradient = uninterrupted->deformation_gradient;
  auto replay_checkpoint = *restored;
  replay_checkpoint.elastic_particles.front().deformation_gradient = replayed->deformation_gradient;
  const auto uninterrupted_bytes = EncodeSolidInteractionCheckpoint3D(checkpoint);
  const auto replay_bytes = EncodeSolidInteractionCheckpoint3D(replay_checkpoint);
  ASSERT_TRUE(uninterrupted_bytes.ok()) << uninterrupted_bytes.status();
  ASSERT_TRUE(replay_bytes.ok()) << replay_bytes.status();
  EXPECT_EQ(*uninterrupted_bytes, *replay_bytes);
  checkpoint.elastic_particles.front().deformation_gradient = {};
  EXPECT_EQ(EncodeSolidInteractionCheckpoint3D(checkpoint).status().code(), absl::StatusCode::kInvalidArgument);

  for (double& velocity : new_velocity) {
    velocity *= -2.0;
  }
  const auto collapsed = GatherElasticBladeDeformation3D(*grid, particle, side, old_history, new_history, kDeformation);
  EXPECT_EQ(collapsed.status().code(), absl::StatusCode::kFailedPrecondition);
  ExpectNear(kDeformation.x_row, {1.0, 0.0, 0.0});

  auto accepted = *restored;
  const auto rejected = AdvanceElasticBladeParticles3D(*grid, blade, 17, 0.0, 0.0, old_history, new_history, accepted);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);
  const auto after_rejection = EncodeSolidInteractionCheckpoint3D(accepted);
  ASSERT_TRUE(after_rejection.ok()) << after_rejection.status();
  EXPECT_EQ(*after_rejection, *encoded);

  for (double& velocity : new_velocity) {
    velocity *= -0.0625;
  }
  const auto accepted_status =
      AdvanceElasticBladeParticles3D(*grid, blade, 17, 0.0, 0.0, old_history, new_history, accepted);
  ASSERT_TRUE(accepted_status.ok()) << accepted_status.status();
  EXPECT_EQ(accepted.accepted_step, 2);
  const auto& accepted_particle = accepted.elastic_particles.front();
  ExpectNear(accepted_particle.deformation_gradient.x_row, {1.59375, 0.59375, 0.0});
  EXPECT_GT(accepted_particle.transfer.trial_world_position.x, particle.trial_world_position.x);
  EXPECT_EQ(accepted.sides.front().side.sign, 1);
}

TEST(MacTransfer3D, ElasticBladeStepBuildsHistoriesAndReplaysUniformTransport) {
  const auto grid = SparseMacGrid3D::Create(kTranslatedGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto particle = CornerParticle(kTranslatedGeometry);
  particle.particle_velocity = {0.125, 0.0, 0.0};
  particle.particle_velocity_slopes = {};
  const TranslatingSlab3D blade{
      .unit_normal = {1.0, 0.0, 0.0},
      .start_center_coordinate = 6.5,
      .half_width = 1.0,
      .wall_velocity = {},
  };
  constexpr Mat3 kIdentity{
      .x_row = {1.0, 0.0, 0.0},
      .y_row = {0.0, 1.0, 0.0},
      .z_row = {0.0, 0.0, 1.0},
  };
  SolidInteractionCheckpoint3D state{
      .accepted_step = 0,
      .sides = {{.particle_id = 7, .side = {.blade_id = 17, .sign = 1}}},
      .elastic_particles = {{
          .particle_id = 7,
          .body_id = 23,
          .transfer = particle,
          .reference_volume = 0.25,
          .deformation_gradient = kIdentity,
      }},
  };
  const ElasticBladeGridStep3D settings{
      .blade = blade,
      .blade_id = 17,
      .duration = 0.125,
      .shear_modulus = 2.0,
      .first_lame_parameter = 1.0,
      .quantization = {0x1p-40, 0x1p-40, 1},
  };
  const auto first = StepElasticBladeGrid3D(*grid, settings, state);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(state.accepted_step, 1);
  EXPECT_NEAR(
      state.elastic_particles.front().transfer.trial_world_position.x,
      particle.trial_world_position.x + 0.015625,
      1e-9
  );
  ExpectNear(state.elastic_particles.front().deformation_gradient, kIdentity);
  const auto bytes = EncodeSolidInteractionCheckpoint3D(state);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  auto restored = DecodeSolidInteractionCheckpoint3D(*bytes);
  ASSERT_TRUE(restored.ok()) << restored.status();
  const auto wave_speed = NeoHookeanAcousticSpeedBound3D(kIdentity, 2.0, 1.0, 8.0);
  ASSERT_TRUE(wave_speed.ok()) << wave_speed.status();
  ExpectNear(*wave_speed, std::sqrt(5.0 / 8.0));
  auto stretched = kIdentity;
  stretched.x_row.x = 2.0;
  const auto stretched_speed = NeoHookeanAcousticSpeedBound3D(stretched, 2.0, 1.0, 8.0);
  ASSERT_TRUE(stretched_speed.ok()) << stretched_speed.status();
  ExpectNear(*stretched_speed, std::sqrt((11.0 - std::log(2.0)) / 8.0));
  stretched.x_row.x = 32.0;
  EXPECT_EQ(
      NeoHookeanAcousticSpeedBound3D(stretched, 2.0, 1.0, 8.0).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  auto oversized = settings;
  oversized.duration = 2.0;
  EXPECT_EQ(StepElasticBladeGrid3D(*grid, oversized, *restored).status().code(), absl::StatusCode::kFailedPrecondition);
  const auto after_oversized = EncodeSolidInteractionCheckpoint3D(*restored);
  ASSERT_TRUE(after_oversized.ok()) << after_oversized.status();
  EXPECT_EQ(*after_oversized, *bytes);
  auto rejected_settings = settings;
  rejected_settings.quantization.declared_particle_capacity = 0;
  EXPECT_FALSE(StepElasticBladeGrid3D(*grid, rejected_settings, *restored).ok());
  const auto after_rejection = EncodeSolidInteractionCheckpoint3D(*restored);
  ASSERT_TRUE(after_rejection.ok()) << after_rejection.status();
  EXPECT_EQ(*after_rejection, *bytes);
  ASSERT_TRUE(StepElasticBladeGrid3D(*grid, settings, *restored).ok());
  ASSERT_TRUE(StepElasticBladeGrid3D(*grid, settings, state).ok());
  const auto replayed = EncodeSolidInteractionCheckpoint3D(*restored);
  const auto uninterrupted = EncodeSolidInteractionCheckpoint3D(state);
  ASSERT_TRUE(replayed.ok() && uninterrupted.ok());
  EXPECT_EQ(*replayed, *uninterrupted);

  auto stressed = state;
  stressed.elastic_particles.front().deformation_gradient.x_row.x = 1.01;
  const auto stressed_step = StepElasticBladeGrid3D(*grid, settings, stressed);
  ASSERT_TRUE(stressed_step.ok()) << stressed_step.status();
  ASSERT_EQ(stressed_step->contacts.size(), 1);
  EXPECT_GT(stressed_step->grid_stress_work, 0.0);
  EXPECT_GT(std::abs(stressed_step->cut_work), 1e-12);
  EXPECT_NEAR(stressed_step->grid_stress_work + stressed_step->deformation_work, stressed_step->cut_work, 1e-12);
  EXPECT_LT(stressed.elastic_particles.front().deformation_gradient.x_row.x, 1.01);
}

TEST(MacTransfer3D, MovingBladeSceneRetainsSideAndReplaysAtTwoTimesteps) {
  const auto grid = SparseMacGrid3D::Create(kTranslatedGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto particle = CornerParticle(kTranslatedGeometry);
  particle.particle_velocity = {};
  particle.particle_velocity_slopes = {};
  constexpr Mat3 kIdentity{
      .x_row = {1.0, 0.0, 0.0},
      .y_row = {0.0, 1.0, 0.0},
      .z_row = {0.0, 0.0, 1.0},
  };
  particle.trial_world_position.x = 9.0;
  particle.sorted_world_position = particle.trial_world_position;
  auto supporting_particle = particle;
  supporting_particle.trial_world_position.x = 10.5;
  supporting_particle.sorted_world_position = supporting_particle.trial_world_position;
  SolidInteractionCheckpoint3D initial{
      .accepted_step = 0,
      .sides = {{.particle_id = 7, .side = {.blade_id = 17, .sign = 1}}},
      .elastic_particles = {{
          .particle_id = 7,
          .body_id = 23,
          .transfer = particle,
          .reference_volume = 0.25,
          .deformation_gradient = kIdentity,
      }},
  };
  initial.sides.push_back({.particle_id = 8, .side = {.blade_id = 17, .sign = 1}});
  initial.elastic_particles.push_back({
      .particle_id = 8,
      .body_id = 23,
      .transfer = supporting_particle,
      .reference_volume = 0.25,
      .deformation_gradient = kIdentity,
  });
  for (const double duration : {0.125, 0.0625}) {
    SCOPED_TRACE(duration);
    ElasticBladeGridStep3D settings{
        .blade =
            {
                .unit_normal = {1.0, 0.0, 0.0},
                .start_center_coordinate = 8.8125,
                .half_width = 0.125,
                .wall_velocity = {0.5, 0.0, 0.0},
            },
        .blade_id = 17,
        .duration = duration,
        .shear_modulus = 2.0,
        .first_lame_parameter = 1.0,
        .quantization = {0x1p-40, 0x1p-40, 2},
    };
    auto state = initial;
    double blade_work = 0.0;
    double normal_impulse = 0.0;
    const int steps = duration == 0.125 ? 1 : 2;
    std::vector<std::uint8_t> halfway_checkpoint;
    for (int step = 0; step < steps; ++step) {
      const auto result = StepElasticBladeGrid3D(*grid, settings, state);
      ASSERT_TRUE(result.ok()) << result.status();
      ASSERT_EQ(result->contacts.size(), 2);
      const auto& contact = result->contacts.front();
      blade_work += contact.blade_work;
      normal_impulse += contact.particle_impulse.x;
      EXPECT_EQ(state.sides.front().side.sign, 1);
      EXPECT_NEAR(result->cut_work, 0.0, 1e-10);
      if (steps == 2 && step == 0) {
        const auto encoded = EncodeSolidInteractionCheckpoint3D(state);
        ASSERT_TRUE(encoded.ok()) << encoded.status();
        halfway_checkpoint = *encoded;
      }
      if (steps == 2 && step == 1) {
        auto restarted = DecodeSolidInteractionCheckpoint3D(halfway_checkpoint);
        ASSERT_TRUE(restarted.ok()) << restarted.status();
        const auto replay = StepElasticBladeGrid3D(*grid, settings, *restarted);
        ASSERT_TRUE(replay.ok()) << replay.status();
        const auto replay_bytes = EncodeSolidInteractionCheckpoint3D(*restarted);
        const auto accepted_bytes = EncodeSolidInteractionCheckpoint3D(state);
        ASSERT_TRUE(replay_bytes.ok() && accepted_bytes.ok());
        EXPECT_EQ(*replay_bytes, *accepted_bytes);
        EXPECT_DOUBLE_EQ(replay->contacts.front().blade_work, contact.blade_work);
      }
      settings.blade.start_center_coordinate += 0.5 * duration;
    }
    const auto final = state.elastic_particles.front().transfer.trial_world_position;
    EXPECT_NEAR(final.x, 9.0, 1e-9);
    EXPECT_NEAR(normal_impulse, 1.0, 1e-9);
    EXPECT_NEAR(blade_work, 0.5, 1e-9);
  }
}

TEST(MacTransfer3D, ObliqueBladeStepReconstructsDistantSameSideGhosts) {
  const auto grid = SparseMacGrid3D::Create(kTranslatedGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto particle = CornerParticle(kTranslatedGeometry);
  particle.particle_velocity = {};
  particle.particle_velocity_slopes = {};
  const double normal_component = std::sqrt(0.5);
  const Vec3 position = particle.trial_world_position;
  SolidInteractionCheckpoint3D state{
      .accepted_step = 0,
      .sides = {{.particle_id = 7, .side = {.blade_id = 17, .sign = 1}}},
      .elastic_particles = {{
          .particle_id = 7,
          .body_id = 23,
          .transfer = particle,
          .reference_volume = 0.25,
          .deformation_gradient = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
      }},
  };
  const ElasticBladeGridStep3D settings{
      .blade =
          {
              .unit_normal = {normal_component, normal_component, 0.0},
              .start_center_coordinate = (position.x + position.y) * normal_component - 0.1875,
              .half_width = 0.125,
              .wall_velocity = {0.625 * normal_component, 0.625 * normal_component, 0.0},
          },
      .blade_id = 17,
      .duration = 0.125,
      .shear_modulus = 2.0,
      .first_lame_parameter = 1.0,
      .quantization = {0x1p-40, 0x1p-40, 1},
  };
  const auto step = StepElasticBladeGrid3D(*grid, settings, state);
  ASSERT_TRUE(step.ok()) << step.status();
  ASSERT_EQ(step->contacts.size(), 1);
  const auto& contact = step->contacts.front();
  ASSERT_TRUE(contact.contact.contact_fraction.has_value());
  EXPECT_NEAR(*contact.contact.contact_fraction, 0.8, 1e-10);
  EXPECT_NEAR(contact.blade_work, 0.78125, 1e-10);
  EXPECT_EQ(state.accepted_step, 1);
  EXPECT_EQ(state.sides.front().side.sign, 1);
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
