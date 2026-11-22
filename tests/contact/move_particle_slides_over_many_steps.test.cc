#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>

#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/solid_contact_move_particle_cases.h"

namespace fluidity::contact {
namespace {

TEST(MoveParticle, SlowSlideIntoSliverWedge_FiftyStepsNeverInNarrowGap) {
  // A particle sliding along the floor at 0.1 m/s into a wedge of 1e-8 rad, 50 steps of 0.01 s, each from the end of
  // the one before with its for_next_step: it reaches the side and stops, at least tol / 2 from both solids, and never
  // starts in a narrow gap, at rest at {205368709 tol, 0, tol} (the model). Before the floor was an active surface of
  // the side's hit, step 6 was kStartsInNarrowGap.
  const MoveParticleCase scene = FindMoveParticleCase("SlowSlideIntoSliverWedge");
  SolidContactInputs inputs = ToSolidContactInputs(scene);
  SolidContactResult result;
  ContactFailure failure;
  int failed_steps = 0;
  for (int step = 1; step <= 50; ++step) {
    const ContactError error = MoveParticle(inputs, &result, &failure);
    failed_steps += error == ContactError::kNone ? 0 : 1;
    inputs.start_position = result.position;
    inputs.velocity = {.for_next_step = result.velocity.for_next_step, .for_this_step = result.velocity.for_next_step};
  }
  EXPECT_EQ(failed_steps, 0);
  CheckBothVelocitiesNear(result, {}, 1e-15);
  ExpectVec3Near(result.position, {.x = 205368709 * kTolerance, .y = 0, .z = kTolerance}, kTolerance);
}

TEST(MoveParticle, SlidesTenThousandSteps_NeverStartsInside) {
  // A particle pressed onto a moving oblique flat solid for 10,000 steps of 1/240 s, each step starting where the last
  // ended, with both velocities set again each step to the solid's velocity + {0.2, 0.1, -0.3} - 0.5 normal m/s, and
  // friction 0.3. The production contact_tolerance of a scene 100 m across. Every step succeeds with one hit, the
  // response at the start, and ends between tol / 2 and 2 tol from the solid: rounding never builds up over the steps.
  const Vec3 normal = math::Normalized({.x = 1, .y = 2, .z = 3});
  const SolidMotion motion = {.linear_velocity = {.x = 0.5, .y = -0.25, .z = 0.125}, .angular_velocity = {}};
  const Vec3 point_on_surface = {.x = 0.3, .y = -0.2, .z = 0.1};  // m
  std::array solids = {move_particle_cases::BuildFlatSolid(point_on_surface, normal, motion, 1)};
  const double contact_tolerance = ComputeContactTolerance({
      .largest_coordinate = 100,
      .smallest_cell_width = 0.01,
      .thinnest_solid = 0.01,
  });  // m
  constexpr double kStepDuration = 1.0 / 240;  // s
  const Vec3 velocity = motion.linear_velocity + Vec3{.x = 0.2, .y = 0.1, .z = -0.3} - 0.5 * normal;  // m/s
  SolidContactInputs inputs;
  inputs.start_position = point_on_surface + contact_tolerance * normal;
  inputs.solids = solids.data();
  inputs.solid_count = 1;
  inputs.step_duration = kStepDuration;
  inputs.friction = 0.3;
  inputs.contact_tolerance = contact_tolerance;
  int failed_steps = 0;
  int steps_not_one_hit = 0;
  int steps_out_of_range = 0;
  double least_share = 2;  // Of tol.
  double largest_share = 0;  // Of tol.
  for (int step = 0; step < 10000; ++step) {
    inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
    SolidContactResult result;
    ContactFailure failure;
    if (MoveParticle(inputs, &result, &failure) != ContactError::kNone) {
      ++failed_steps;
      break;
    }
    Solid& solid = solids.at(0);
    solid.pose_at_step_start = ComputeSolidPose(solid, kStepDuration);
    const double end_share =
        ComputeDistanceToSolid(solid, solid.pose_at_step_start, result.position).distance / contact_tolerance;
    least_share = std::min(least_share, end_share);
    largest_share = std::max(largest_share, end_share);
    steps_out_of_range += end_share >= 0.5 && end_share <= 2 ? 0 : 1;
    steps_not_one_hit += result.hit_count == 1 ? 0 : 1;
    inputs.start_position = result.position;
  }
  std::cout << "ten thousand slide steps: least end distance " << least_share << " tol, largest " << largest_share
            << " tol\n";
  EXPECT_EQ(failed_steps, 0);
  EXPECT_EQ(steps_out_of_range, 0);
  EXPECT_EQ(steps_not_one_hit, 0);
}

}  // namespace
}  // namespace fluidity::contact
