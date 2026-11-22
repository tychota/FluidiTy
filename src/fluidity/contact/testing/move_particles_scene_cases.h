// Test-only: the helpers that the unit tests and the functional tests of move_particles_scene share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    kTolerance, kNoMotion, kGravity, ToReactingSolid, FallingParticle, SlideOffSphere, ComputeSlideOffSphere,
//          kAnalyticLeavingAngle
// Uses:    the headers it includes
// Used by: src/fluidity/contact/move_particles.scene.test.cc and tests/contact/move_particles_scenes.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_error_message.h"
#include "fluidity/contact/move_particles.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/contact_main_comparison_cases.h"
#include "fluidity/contact/testing/move_particles_runs.h"

namespace fluidity::contact {

constexpr double kTolerance = kTestContactTolerance;  // m, 2^-30.

constexpr SolidMotion kNoMotion = move_particle_cases::kNoMotion;

constexpr double kGravity = 9.81;  // m/s^2, along -z. The tests add it to the velocities themselves.

// A reacting solid: its hits fill the particles' reactions.
inline Solid ToReactingSolid(Solid solid) {
  solid.reports_reaction = true;
  return solid;
}

// --- A particle on a round solid over many steps ---

// One step of a particle under gravity on the fixed solids of a scene: gravity first, as the step's forces would add
// it, then MoveParticle; the end and both velocities are the next step's start.
struct FallingParticle {
  Vec3 position{};  // m, world.
  Vec3 velocity{};  // m/s, world, both velocities.
};

// What a particle sliding off the top of a fixed frictionless sphere of radius 1 m did, until it ended a step more than
// 0.01 m off the surface.
struct SlideOffSphere {
  int step_count = 0;
  bool has_failed = false;  // A step's MoveParticle returned an error; the slide stopped there.
  int largest_hit_count = 0;  // Over every step.
  double last_hit_angle = -1;  // rad from the top: acos(z / |x|) of the end of the last step with kind kHit.
  double start_energy = 0;  // m^2/s^2 per unit mass: g z + |v|^2 / 2 at the start.
  double end_energy = 0;  // m^2/s^2 per unit mass, at the end of the last step.
  double largest_energy_gain = 0;  // m^2/s^2 per unit mass: the most any step's end had above the start.
  // m. Up to the last hit: the highest end above the rest gap, |x| - 1 - tol, and the highest rise of a straight
  // tangent of one step's travel above the sphere, (|v| T)^2 / (2 R), with v the end's velocity.
  double largest_hop_height = 0;
  double largest_tangent_rise = 0;
};

// Slides a particle from the top of the fixed sphere of radius 1 m at the origin, {0, 0, 1 + tol}, at {0.01, 0, 0}
// m/s, friction 0, in steps of step_duration that each first subtract 9.81 step_duration from the z of both
// velocities, then call MoveParticle and feed its end and velocity.for_next_step back; it stops when the end is more
// than 0.01 m off the surface, or after 100,000 steps.
inline SlideOffSphere ComputeSlideOffSphere(double step_duration) {
  const std::vector<Solid> solids = {ToSolid(Sphere{.center = {}, .radius = 1}, kNoMotion, 1)};
  FallingParticle particle = {
      .position = {.x = 0, .y = 0, .z = 1 + kTolerance},
      .velocity = {.x = 0.01, .y = 0, .z = 0},
  };
  const auto compute_energy = [](const FallingParticle& falling) {
    return (kGravity * falling.position.z) + (0.5 * math::Dot(falling.velocity, falling.velocity));
  };
  SlideOffSphere slide;
  slide.start_energy = compute_energy(particle);
  double hop_height_so_far = 0;  // m
  double tangent_rise_so_far = 0;  // m
  while (math::Length(particle.position) - 1 <= 0.01 && slide.step_count < 100000) {
    particle.velocity.z -= kGravity * step_duration;
    SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
        .start_position = particle.position,
        .velocity = particle.velocity,
        .step_duration = step_duration,
        .friction = 0,
    });
    inputs.solids = solids.data();
    inputs.solid_count = 1;
    SolidContactResult result;
    if (MoveParticle(inputs, &result, nullptr) != ContactError::kNone) {
      slide.has_failed = true;
      return slide;
    }
    slide.largest_hit_count = std::max<int>(slide.largest_hit_count, result.hit_count);
    const double travel = math::Length(result.velocity.for_next_step) * step_duration;  // m
    hop_height_so_far = std::max(hop_height_so_far, math::Length(result.position) - 1 - kTolerance);
    tangent_rise_so_far = std::max(tangent_rise_so_far, travel * travel / 2);
    if (result.kind == ContactKind::kHit) {
      slide.last_hit_angle = std::acos(result.position.z / math::Length(result.position));
      slide.largest_hop_height = hop_height_so_far;
      slide.largest_tangent_rise = tangent_rise_so_far;
    }
    particle = {.position = result.position, .velocity = result.velocity.for_next_step};
    slide.end_energy = compute_energy(particle);
    slide.largest_energy_gain = std::max(slide.largest_energy_gain, slide.end_energy - slide.start_energy);
    ++slide.step_count;
  }
  return slide;
}

// rad. Where a bead leaves a frictionless sphere: gravity's part along the normal no longer holds it on the curve,
// v^2 = v0^2 + 2 g R (1 - cos), leaving when v^2 / R = g cos: cos(angle) = 2/3 + v0^2 / (3 g R), here
// acos(2/3 + 0.01^2 / (3 * 9.81)) from the top.
constexpr double kAnalyticLeavingAngle = 0.8410641118063884;

}  // namespace fluidity::contact
