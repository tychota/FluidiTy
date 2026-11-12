#include <cmath>
#include <numbers>

#include "fluidity/host/rigid_body_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

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

}  // namespace
}  // namespace fluidity
