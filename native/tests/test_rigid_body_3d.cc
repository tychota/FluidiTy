#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include "fluidity/host/rigid_body_3d.h"
#include "fluidity/host/solid_interaction_checkpoint_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(RigidBody3D, PressureTractionUsesWettedAreaAndOffCenterMoment) {
  const RigidBody3D body{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {2.0, 3.0, 4.0},
      .center = {2.0, 0.0, 0.0},
      .force = {0.0, -4.0, 0.0},
  };
  const std::array pressures{18.0, 14.0};
  std::array patches{
      RigidPressurePatch3D{
          .patch_id = 1,
          .body_id = 23,
          .pressure_cell_index = 0,
          .wetted_area = 0.5,
          .world_centroid = {3.0, 0.0, 0.0},
          .outward_body_normal = {0.0, -1.0, 0.0},
      },
      RigidPressurePatch3D{
          .patch_id = 2,
          .body_id = 23,
          .pressure_cell_index = 1,
          .wetted_area = 0.25,
          .world_centroid = {1.0, 0.0, 0.0},
          .outward_body_normal = {0.0, -1.0, 0.0},
      },
  };
  const auto loaded = AccumulateRigidBodyPressure3D(body, pressures, 10.0, patches);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_DOUBLE_EQ(loaded->force.y, 1.0);
  EXPECT_DOUBLE_EQ(loaded->torque.z, 3.0);
  const auto step = AdvanceRigidBody3D(*loaded, 0.125);
  ASSERT_TRUE(step.ok()) << step.status();
  EXPECT_DOUBLE_EQ(step->end.linear_velocity.y, 0.0625);
  EXPECT_DOUBLE_EQ(step->end.angular_velocity.z, 0.09375);
  patches[1].patch_id = 1;
  EXPECT_EQ(
      AccumulateRigidBodyPressure3D(body, pressures, 10.0, patches).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_DOUBLE_EQ(body.force.y, -4.0);
}

TEST(RigidBody3D, FloatingBoxClipsWetFacesAndUsesExteriorPressureCells) {
  const RigidBody3D body{
      .body_id = 23,
      .mass = 0.5,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0.5, 0.5, 0.5},
      .force = {0.0, -5.0, 0.0},
  };
  constexpr std::array cells{
      CellIndex3D{0, 0, -1},
      CellIndex3D{0, -1, 0},
      CellIndex3D{-1, 0, 0},
      CellIndex3D{0, 0, 0},  // Interior at the bottom and side planes.
      CellIndex3D{1, 0, 0},
      CellIndex3D{0, 0, 1},
  };
  constexpr std::array pressure{2.5, 5.0, 2.5, 1000.0, 2.5, 2.5};
  const auto patches =
      BuildFloatingBoxPressurePatches3D(body.body_id, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, 0.5, {{}, 1.0}, cells);
  ASSERT_TRUE(patches.ok()) << patches.status();
  ASSERT_EQ(patches->size(), 5);
  EXPECT_DOUBLE_EQ((*patches)[0].wetted_area, 0.5);
  EXPECT_DOUBLE_EQ((*patches)[0].world_centroid.y, 0.25);
  EXPECT_DOUBLE_EQ((*patches)[2].wetted_area, 1.0);
  EXPECT_EQ((*patches)[2].pressure_cell_index, 1);
  const auto loaded = AccumulateRigidBodyPressure3D(body, pressure, 0.0, *patches);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_DOUBLE_EQ(loaded->force.x, 0.0);
  EXPECT_DOUBLE_EQ(loaded->force.y, 0.0);
  EXPECT_DOUBLE_EQ(loaded->force.z, 0.0);
  EXPECT_DOUBLE_EQ(loaded->torque.x, 0.0);
  EXPECT_DOUBLE_EQ(loaded->torque.y, 0.0);
  EXPECT_DOUBLE_EQ(loaded->torque.z, 0.0);

  const auto oriented = BuildOrientedBoxPressurePatches3D(body, {0.5, 0.5, 0.5}, 0.5, {{}, 1.0}, cells);
  ASSERT_TRUE(oriented.ok()) << oriented.status();
  ASSERT_EQ(oriented->size(), patches->size());
  for (std::size_t index = 0; index < patches->size(); ++index) {
    EXPECT_EQ((*oriented)[index].pressure_cell_index, (*patches)[index].pressure_cell_index);
    EXPECT_DOUBLE_EQ((*oriented)[index].wetted_area, (*patches)[index].wetted_area);
  }
  const auto missing = BuildFloatingBoxPressurePatches3D(
      body.body_id,
      {0.0, 0.0, 0.0},
      {1.0, 1.0, 1.0},
      0.5,
      {{}, 1.0},
      std::span(cells).subspan(1)
  );
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kFailedPrecondition);
  auto duplicate = cells;
  duplicate[3] = duplicate[2];
  const auto malformed =
      BuildFloatingBoxPressurePatches3D(body.body_id, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, 0.5, {{}, 1.0}, duplicate);
  EXPECT_EQ(malformed.status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(RigidBody3D, OrientedBoxWetPatchesRecoverLiftAndOffCenterTorque) {
  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1.0, 2.0, 3.0},
      .orientation = {.w = std::sqrt(0.9), .vector = {0.0, 0.0, std::sqrt(0.1)}},
      .linear_velocity = {0.0, 0.25, 0.0},
      .angular_velocity = {0.0, 0.0, 0.5},
  };
  constexpr UniformMacGeometry3D kGrid{{0.0, -2.0, -2.0}, 4.0};
  constexpr std::array cells{CellIndex3D{-1, 0, 0}, CellIndex3D{0, 0, 0}};
  constexpr Vec3 kHalfExtent{0.5, 0.5, 0.5};
  const auto patches = BuildOrientedBoxPressurePatches3D(body, kHalfExtent, 0.4, kGrid, cells);
  ASSERT_TRUE(patches.ok()) << patches.status();
  ASSERT_GT(patches->size(), 6);
  std::vector<RigidPressurePatch3D> streamed;
  const auto cell_at = [&](std::size_t ordinal) {
    return cells[ordinal];
  };
  const auto visit = VisitOrientedBoxPressurePatches3D(
      body,
      kHalfExtent,
      0.4,
      kGrid,
      cells.size(),
      cell_at,
      [&](const RigidPressurePatch3D& patch) {
        streamed.push_back(patch);
        return absl::OkStatus();
      }
  );
  ASSERT_TRUE(visit.ok()) << visit;
  ASSERT_EQ(streamed.size(), patches->size());
  for (std::size_t index = 0; index < streamed.size(); ++index) {
    EXPECT_EQ(streamed[index].patch_id, (*patches)[index].patch_id);
    EXPECT_EQ(streamed[index].pressure_cell_index, (*patches)[index].pressure_cell_index);
    EXPECT_DOUBLE_EQ(streamed[index].wetted_area, (*patches)[index].wetted_area);
    EXPECT_DOUBLE_EQ(streamed[index].world_centroid.x, (*patches)[index].world_centroid.x);
    EXPECT_DOUBLE_EQ(streamed[index].world_centroid.y, (*patches)[index].world_centroid.y);
    EXPECT_DOUBLE_EQ(streamed[index].world_centroid.z, (*patches)[index].world_centroid.z);
  }
  const auto visitor_error = VisitOrientedBoxPressurePatches3D(
      body,
      kHalfExtent,
      0.4,
      kGrid,
      cells.size(),
      cell_at,
      [](const RigidPressurePatch3D&) {
        return absl::CancelledError();
      }
  );
  EXPECT_EQ(visitor_error.code(), absl::StatusCode::kCancelled);
  constexpr std::array pressure{8.0, 8.0};
  const auto loaded = AccumulateRigidBodyPressure3D(body, pressure, 0.0, *patches);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  // The horizontal section is [-0.3, 0.325] times one meter in z.
  // Closing the wetted surface with this cap gives the uniform-pressure oracle.
  EXPECT_NEAR(loaded->force.x, 0.0, 1e-12);
  EXPECT_NEAR(loaded->force.y, 5.0, 1e-12);
  EXPECT_NEAR(loaded->force.z, 0.0, 1e-12);
  EXPECT_NEAR(loaded->torque.z, 0.0625, 1e-12);
  const auto flux = ComputeRigidBodyInwardWallFlux3D(body, cells.size(), *patches);
  ASSERT_TRUE(flux.ok()) << flux.status();
  const double power = 0.25 * loaded->force.y + 0.5 * loaded->torque.z;
  EXPECT_NEAR(power + 8.0 * ((*flux)[0] + (*flux)[1]), 0.0, 1e-12);

  const auto missing = BuildOrientedBoxPressurePatches3D(body, kHalfExtent, 0.4, kGrid, std::span(cells).first(1));
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kFailedPrecondition);
}
TEST(RigidBody3D, WallFluxAndTractionExchangeEqualAndOppositePower) {
  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .linear_velocity = {0.25, 0.5, 0.0},
      .angular_velocity = {0.0, 0.0, 1.0},
  };
  const std::array patches{
      RigidPressurePatch3D{1, body.body_id, 1, 0.5, {0.5, 0.0, 0.0}, {0.0, -1.0, 0.0}},
      RigidPressurePatch3D{2, body.body_id, 1, 0.25, {0.0, 0.5, 0.0}, {-1.0, 0.0, 0.0}},
  };
  const auto fluxes = ComputeRigidBodyInwardWallFlux3D(body, 2, patches);
  ASSERT_TRUE(fluxes.ok()) << fluxes.status();
  EXPECT_DOUBLE_EQ((*fluxes)[0], 0.0);
  EXPECT_DOUBLE_EQ((*fluxes)[1], -0.4375);

  const auto loaded = AccumulateRigidBodyPressure3D(body, std::array{0.0, 8.0}, 0.0, patches);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const double body_power = loaded->force.x * body.linear_velocity.x + loaded->force.y * body.linear_velocity.y
      + loaded->torque.z * body.angular_velocity.z;
  EXPECT_DOUBLE_EQ(body_power, 3.5);
  EXPECT_DOUBLE_EQ(body_power + 8.0 * (*fluxes)[1], 0.0);
  EXPECT_EQ(ComputeRigidBodyInwardWallFlux3D(body, 1, patches).status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(RigidBody3D, FloatingBoxRejectsInteriorHoleWithCompleteOuterEdges) {
  std::vector<CellIndex3D> cells;
  for (int z = -1; z <= 3; ++z) {
    for (int y = -1; y <= 0; ++y) {
      for (int x = -1; x <= 3; ++x) {
        if (x == 1 && y == -1 && z == 1) {
          continue;
        }
        cells.push_back({x, y, z});
      }
    }
  }
  const auto missing = BuildFloatingBoxPressurePatches3D(1, {}, {3, 1, 3}, 0.5, {{}, 1.0}, cells);
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(missing.status().message(), "floating-box wet pressure coverage has missing tiles");
  const auto dry = BuildFloatingBoxPressurePatches3D(1, {}, {3, 1, 3}, 0.0, {{}, 1.0}, {});
  ASSERT_TRUE(dry.ok()) << dry.status();
  EXPECT_TRUE(dry->empty());
}
TEST(RigidBody3D, AdvancesRotatedAsymmetricInertiaAndOffCenterForce) {
  const double half_sqrt_two = std::numbers::sqrt2 / 2.0;
  const RigidBody3D initial{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {2.0, 3.0, 4.0},
      .center = {},
      .orientation = {.w = half_sqrt_two, .vector = {0.0, 0.0, half_sqrt_two}},
      .linear_velocity = {},
      .angular_velocity = {1.0, 2.0, 3.0},
  };
  const auto loaded = AccumulateRigidBodyPointForce3D(initial, {0.0, 2.0, 0.0}, {4.0, 0.0, 0.0});
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_DOUBLE_EQ(loaded->torque.z, -8.0);
  const auto step = AdvanceRigidBody3D(*loaded, 0.125);
  ASSERT_TRUE(step.ok()) << step.status();
  EXPECT_DOUBLE_EQ(step->end.center.x, 0.03125);
  EXPECT_DOUBLE_EQ(step->end.linear_velocity.x, 0.25);
  EXPECT_NEAR(step->end.angular_velocity.x, 0.5, 1e-12);
  EXPECT_NEAR(step->end.angular_velocity.y, 2.1875, 1e-12);
  EXPECT_NEAR(step->end.angular_velocity.z, 2.8125, 1e-12);
  EXPECT_DOUBLE_EQ(step->end.force.x, 0.0);
  EXPECT_DOUBLE_EQ(step->end.torque.z, 0.0);

  const auto halfway = SampleRigidBodyPose3D(*step, 0.5);
  ASSERT_TRUE(halfway.ok()) << halfway.status();
  EXPECT_DOUBLE_EQ(halfway->center.x, 0.015625);
  const auto end_pose = SampleRigidBodyPose3D(*step, 1.0);
  ASSERT_TRUE(end_pose.ok()) << end_pose.status();
  EXPECT_DOUBLE_EQ(end_pose->orientation.w, step->end.orientation.w);
  const auto& [w, vector] = end_pose->orientation;
  EXPECT_NEAR(w * w + vector.x * vector.x + vector.y * vector.y + vector.z * vector.z, 1.0, 1e-12);
  EXPECT_GT(end_pose->orientation.vector.x, 0.0);
  const auto bound = RigidBodySweepDistanceBound3D(*step, 2.0);
  ASSERT_TRUE(bound.ok()) << bound.status();
  EXPECT_GE(*bound, 0.03125 + 2.0 * 0.125 * std::hypot(0.5, 2.1875, 2.8125));
}

TEST(RigidBody3D, RejectsInvalidInertiaOrDurationWithoutConsumingLoad) {
  RigidBody3D loaded{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {2.0, 3.0, 0.0},
      .force = {4.0, 0.0, 0.0},
  };
  EXPECT_EQ(AdvanceRigidBody3D(loaded, 0.125).status().code(), absl::StatusCode::kInvalidArgument);
  loaded.principal_inertia.z = 4.0;
  EXPECT_EQ(AdvanceRigidBody3D(loaded, -1.0).status().code(), absl::StatusCode::kInvalidArgument);
  const auto retry = AdvanceRigidBody3D(loaded, 0.125);
  ASSERT_TRUE(retry.ok()) << retry.status();
  EXPECT_DOUBLE_EQ(retry->end.linear_velocity.x, 0.25);
  EXPECT_DOUBLE_EQ(loaded.force.x, 4.0);
}

TEST(RigidBody3D, SweepBoundCoversRoundedTranslationAtLargeOrigin) {
  const RigidBody3D body{
      .body_id = 23,
      .mass = 1.0,
      .principal_inertia = {1.0, 1.0, 1.0},
      .center = {0x1p53, 0.0, 0.0},
      .linear_velocity = {1.25, 0.0, 0.0},
  };
  const auto step = AdvanceRigidBody3D(body, 1.0);
  ASSERT_TRUE(step.ok()) << step.status();
  const auto bound = RigidBodySweepDistanceBound3D(*step, 0.0);
  ASSERT_TRUE(bound.ok()) << bound.status();
  EXPECT_DOUBLE_EQ(step->end.center.x - body.center.x, 2.0);
  EXPECT_GE(*bound, step->end.center.x - body.center.x);
}

TEST(RigidBody3D, InteractionCheckpointReplaysBodyAndBladeSideAfterRejectedAttempts) {
  const RigidBody3D body{
      .body_id = 23,
      .mass = 2.0,
      .principal_inertia = {2.0, 3.0, 4.0},
      .angular_velocity = {0.0, 0.0, 1.0},
      .force = {4.0, 0.0, 0.0},
  };
  const auto first = AdvanceRigidBody3D(body, 0.125);
  ASSERT_TRUE(first.ok()) << first.status();
  const LiquidWallContactInputs3D blade_contact{
      .start_world_position = {},
      .stored_world_velocity = {},
      .advecting_world_velocity = {},
      .obstacle =
          TranslatingSlab3D{
              .unit_normal = {1.0, 0.0, 0.0},
              .start_center_coordinate = 1.0,
              .half_width = 1.0 / 32.0,
              .wall_velocity = {-2.0, 0.0, 0.0},
          },
      .step_duration = 1.0,
  };
  const auto initial_cut = ResolveElasticBladeContact3D(blade_contact, 2.0, 17, {});
  ASSERT_TRUE(initial_cut.ok()) << initial_cut.status();
  ASSERT_EQ(initial_cut->side.sign, -1);
  const SolidInteractionCheckpoint3D checkpoint{
      .accepted_step = 1,
      .bodies = {{first->end, first->start}},
      .sides = {{.particle_id = 7, .side = initial_cut->side}},
  };
  const auto bytes = EncodeSolidInteractionCheckpoint3D(checkpoint);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  std::vector<std::uint8_t> unaligned(bytes->size() + 1);
  std::ranges::copy(*bytes, unaligned.begin() + 1);
  EXPECT_EQ(
      DecodeSolidInteractionCheckpoint3D(std::span(unaligned).subspan(1)).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  auto corrupt = *bytes;
  corrupt.pop_back();
  EXPECT_EQ(DecodeSolidInteractionCheckpoint3D(corrupt).status().code(), absl::StatusCode::kInvalidArgument);
  const auto restored = DecodeSolidInteractionCheckpoint3D(*bytes);
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_TRUE(restored->elastic_particles.empty());
  ASSERT_EQ(restored->accepted_step, 1);
  ASSERT_EQ(restored->bodies.size(), 1);
  ASSERT_EQ(restored->sides.size(), 1);
  EXPECT_EQ(restored->sides[0].particle_id, 7);
  EXPECT_EQ(restored->sides[0].side.blade_id, 17);
  EXPECT_EQ(restored->sides[0].side.sign, -1);
  EXPECT_DOUBLE_EQ(restored->bodies[0].previous_pose.center.x, 0.0);

  const auto before_retry = EncodeSolidInteractionCheckpoint3D(*restored);
  ASSERT_TRUE(before_retry.ok()) << before_retry.status();
  EXPECT_EQ(AdvanceRigidBody3D(restored->bodies[0].body, 0.0).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ResolveElasticBladeContact3D(blade_contact, 2.0, 99, restored->sides[0].side).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  const auto after_rejection = EncodeSolidInteractionCheckpoint3D(*restored);
  ASSERT_TRUE(after_rejection.ok()) << after_rejection.status();
  EXPECT_EQ(*before_retry, *after_rejection);
  const auto replayed = AdvanceRigidBody3D(restored->bodies[0].body, 0.125);
  const auto uninterrupted = AdvanceRigidBody3D(first->end, 0.125);
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  ASSERT_TRUE(uninterrupted.ok()) << uninterrupted.status();

  auto resumed = *restored;
  auto continuous = checkpoint;
  resumed.bodies[0] = {replayed->end, replayed->start};
  continuous.bodies[0] = {uninterrupted->end, uninterrupted->start};
  auto next_contact = blade_contact;
  next_contact.start_world_position = initial_cut->contact.world_position;
  next_contact.stored_world_velocity = initial_cut->contact.stored_world_velocity;
  next_contact.advecting_world_velocity = initial_cut->contact.advecting_world_velocity;
  auto& moving_blade = std::get<TranslatingSlab3D>(next_contact.obstacle);
  moving_blade.start_center_coordinate += moving_blade.wall_velocity.x * next_contact.step_duration;
  const auto replayed_cut = ResolveElasticBladeContact3D(next_contact, 2.0, 17, restored->sides[0].side);
  const auto uninterrupted_cut = ResolveElasticBladeContact3D(next_contact, 2.0, 17, checkpoint.sides[0].side);
  ASSERT_TRUE(replayed_cut.ok()) << replayed_cut.status();
  ASSERT_TRUE(uninterrupted_cut.ok()) << uninterrupted_cut.status();
  EXPECT_DOUBLE_EQ(replayed_cut->contact.world_position.x, uninterrupted_cut->contact.world_position.x);
  EXPECT_DOUBLE_EQ(replayed_cut->blade_work, uninterrupted_cut->blade_work);
  resumed.sides[0].side = replayed_cut->side;
  continuous.sides[0].side = uninterrupted_cut->side;
  ++resumed.accepted_step;
  ++continuous.accepted_step;
  const auto replay_bytes = EncodeSolidInteractionCheckpoint3D(resumed);
  const auto continuous_bytes = EncodeSolidInteractionCheckpoint3D(continuous);
  ASSERT_TRUE(replay_bytes.ok()) << replay_bytes.status();
  ASSERT_TRUE(continuous_bytes.ok()) << continuous_bytes.status();
  EXPECT_EQ(*replay_bytes, *continuous_bytes);
}

}  // namespace
}  // namespace fluidity
