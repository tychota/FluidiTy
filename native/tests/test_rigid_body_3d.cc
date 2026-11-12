#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include "fluidity/host/rigid_body_3d.h"
#include "fluidity/host/solid_interaction_checkpoint_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(RigidBody3D, AcceptedInteractionCheckpointRoundTripsLiquidAndPlastic) {
  LiquidState3D liquid;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(1);
  ASSERT_TRUE(particles.ok()) << particles.status();
  ASSERT_TRUE(particles->Resize(1).ok());
  liquid.particles = std::move(*particles);
  liquid.particles[0] = {.id = 5, .world_position = {1, 2, 3}};
  liquid.next_particle_id = 6;
  const DrySolidMaterial3D snow = SnowParameters3D{2, 3, 0.025, 0.01, 4};
  const DrySolidParticle3D compressed_snow{
      .world_position = {2.25, 2.25, 2.25},
      .mass = 1,
      .reference_volume = 0.001,
      .elastic_deformation = {{1, 0, 0}, {0, 0.975, 0}, {0, 0, 1}},
      .log_plastic_jacobian = std::log(0.8),
  };
  const auto captured = CapturePlasticSolidCheckpointParticle3D(7, 9, 11, snow, compressed_snow);
  ASSERT_TRUE(captured.ok()) << captured.status();
  const std::array bindings{PlasticSolidCheckpointBinding3D{9, 11, snow}};
  SolidInteractionCheckpoint3D solid{.accepted_step = 0, .bodies = {}, .sides = {}};
  const auto legacy = EncodeSolidInteractionCheckpoint3D(solid);
  ASSERT_TRUE(legacy.ok()) << legacy.status();
  ASSERT_TRUE(DecodeSolidInteractionCheckpoint3D(*legacy).ok());
  solid.plastic_particles = {*captured};
  const auto bytes = EncodeCoupledInteractionCheckpoint3D(liquid, solid, bindings);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  EXPECT_EQ(DecodeCoupledInteractionCheckpoint3D(*bytes).status().code(), absl::StatusCode::kInvalidArgument);
  const auto restored = DecodeCoupledInteractionCheckpoint3D(*bytes, bindings);
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ(restored->liquid, liquid);
  ASSERT_EQ(restored->solid.plastic_particles.size(), 1);
  EXPECT_DOUBLE_EQ(restored->solid.plastic_particles.front().log_plastic_jacobian, std::log(0.8));
  const auto replayed = EncodeCoupledInteractionCheckpoint3D(restored->liquid, restored->solid, bindings);
  ASSERT_TRUE(replayed.ok()) << replayed.status();
  EXPECT_EQ(*bytes, *replayed);
  const auto solid_only = EncodeSolidInteractionCheckpoint3D(solid, bindings);
  ASSERT_TRUE(solid_only.ok()) << solid_only.status();
  EXPECT_TRUE(DecodeSolidInteractionCheckpoint3D(*solid_only, bindings).ok());
  EXPECT_EQ(
      DecodeCoupledInteractionCheckpoint3D(*solid_only, bindings).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  solid.plastic_particles.front().log_plastic_jacobian = 1000;
  EXPECT_EQ(
      EncodeCoupledInteractionCheckpoint3D(liquid, solid, bindings).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
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
