// Scene tests of contact: what a solid's reaction adds up to over many particles, a particle on a round solid over many
// steps, and the cases of the old contact code on branch main.
#include "fluidity/contact/move_particles.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/contact_main_comparison_cases.h"
#include "fluidity/contact/testing/move_particles_runs.h"
#include "fluidity/contact/testing/move_particles_scene_cases.h"

namespace fluidity::contact {
namespace {

// --- What a reacting solid is charged ---

TEST(MoveParticles, SymmetricHitsOnReactingBox_SummedChangeHasNoAngularMomentum) {
  // A fixed reacting box of half size 0.5 m at the origin and two particles of equal mass at y = +0.25 and -0.25 m,
  // both at {2, 0, 0} m/s for 1 s, friction 0: each stops on kLowX at x = -0.5 - tol, its change {-2, 0, 0} m/s.
  // The two changes sum to {-4, 0, 0}, and their angular momenta about the origin, (hit - origin) x change =
  // {0, 0, 2 y}, cancel.
  ParticleRange range;
  const Box box = {.center = {}, .axes = kIdentity, .half_size = {.x = 0.5, .y = 0.5, .z = 0.5}};
  range.solids = {ToReactingSolid(ToSolid(box, kNoMotion, 1))};
  for (const double side : {0.25, -0.25}) {
    const SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
        .start_position = {.x = -1, .y = side, .z = 0},
        .velocity = {.x = 2, .y = 0, .z = 0},
        .step_duration = 1,
        .friction = 0,
    });
    range.AddParticle(inputs, range.size() + 1);
  }
  const MovedRange moved = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  ASSERT_TRUE(moved.result.has_value());
  EXPECT_EQ(moved.result->hit_count, 2U);
  Vec3 change_sum{};
  Vec3 angular_change_sum{};
  for (const SolidReaction& reaction : moved.reaction) {
    EXPECT_EQ(reaction.solid_index, 0U);
    change_sum = change_sum + reaction.velocity_for_next_step_change;
    angular_change_sum =
        angular_change_sum + math::Cross(reaction.hit_position, reaction.velocity_for_next_step_change);
  }
  CheckVec3IsNear(change_sum, {.x = -4, .y = 0, .z = 0}, 4 * 0x1p-44);
  CheckVec3IsNear(angular_change_sum, {}, 1e-15);
}

// Classifies a straight path from_centre + velocity t, 0 <= t <= step_duration, against a sphere of radius radius about
// the origin: its closest approach, reached within the step, below radius - 1e-6 m is a hit; above radius + 1e-6 m, or
// reached at or before t = 0, a miss.
ExpectedHit ComputeSpherePathClass(Vec3 from_centre, Vec3 velocity, double radius, double step_duration) {
  const double closest_time = -math::Dot(from_centre, velocity) / math::Dot(velocity, velocity);  // s
  const double closest_distance = math::Length(from_centre + (closest_time * velocity));  // m
  if (closest_time <= 0 || closest_distance > radius + 1e-6) {
    return ExpectedHit::kMiss;
  }
  const bool is_hit = closest_distance < radius - 1e-6 && closest_time < step_duration;
  return is_hit ? ExpectedHit::kHit : ExpectedHit::kTooCloseToTell;
}

// m. The angular momentum of a change about a point, per unit of the change: |(hit_position - point) x change| /
// |change|, the change's lever arm.
double ComputeLeverArm(const SolidReaction& reaction, Vec3 point) {
  const Vec3 change = reaction.velocity_for_next_step_change;  // m/s
  return math::Length(math::Cross(reaction.hit_position - point, change)) / math::Length(change);
}

// The reacting sphere's scene: a fixed sphere of radius 0.5 m at {1, 2, 3}, 100 particles, friction 0, step 3 s, and
// the classification of each path.
constexpr Vec3 kSphereSceneCentre = {.x = 1, .y = 2, .z = 3};  // m

constexpr double kSphereSceneRadius = 0.5;  // m

struct SphereScene {
  ParticleRange range;
  std::vector<ExpectedHit> expected_hits;  // ComputeSpherePathClass, one per particle.
};

// Builds the reacting sphere's scene. CaseGenerator, per particle in this order: the direction (three NextSigned, drawn
// again while shorter than 0.25, then normalized), the jitter, 0.3 times three NextSigned. Start centre + 2 direction,
// both velocities -direction + jitter.
SphereScene BuildSphereScene() {
  constexpr double kStepDuration = 3;  // s
  SphereScene scene;
  scene.range.solids = {
      ToReactingSolid(ToSolid(Sphere{.center = kSphereSceneCentre, .radius = kSphereSceneRadius}, kNoMotion, 1)),
  };
  CaseGenerator generator;
  for (int index = 0; index < 100; ++index) {
    const Vec3 direction = far_from_origin_cases::BuildDirection(generator);
    const Vec3 jitter = 0.3 * far_from_origin_cases::BuildSignedVector(generator);
    const Vec3 velocity = jitter - direction;  // m/s
    const SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
        .start_position = kSphereSceneCentre + (2 * direction),
        .velocity = velocity,
        .step_duration = kStepDuration,
        .friction = 0,
    });
    scene.range.AddParticle(inputs, static_cast<std::uint64_t>(index) + 1);
    scene.expected_hits.push_back(ComputeSpherePathClass(2 * direction, velocity, kSphereSceneRadius, kStepDuration));
  }
  return scene;
}

// What a run of the reacting sphere's scene showed.
struct SphereSceneCheck {
  std::vector<std::size_t> failing_particles;  // Indexes classified wrong or with more than one hit.
  double largest_lever_arm = 0;  // m, ComputeLeverArm over the particles with a reaction.
};

SphereSceneCheck ComputeSphereSceneCheck(const SphereScene& scene, const MovedRange& moved) {
  SphereSceneCheck check;
  for (std::size_t index = 0; index < scene.range.size(); ++index) {
    const SolidReaction& reaction = moved.reaction[index];
    const bool is_hit = reaction.solid_index != kNoSolid;
    const bool is_one_hit_at_most = moved.record[index].hit_count <= 1;
    const ExpectedHit seen_hit = is_hit ? ExpectedHit::kHit : ExpectedHit::kMiss;
    if (!ComputeIsClassifiedRight(scene.expected_hits[index], seen_hit) || !is_one_hit_at_most) {
      check.failing_particles.push_back(index);
    }
    if (is_hit) {
      check.largest_lever_arm = std::max(check.largest_lever_arm, ComputeLeverArm(reaction, kSphereSceneCentre));
    }
  }
  return check;
}

TEST(MoveParticles, FrictionlessHitsOnReactingSphere_ChangeHasNoAngularMomentumAboutCentre) {
  // The reacting sphere's scene (BuildSphereScene). Every normal of a sphere passes through its centre, so a
  // frictionless change has no angular momentum about it: |(hit_position - centre) x change| <= 1e-12 |change|. Each
  // path is classified without MoveParticle (ComputeSpherePathClass), and the hits seen must be those paths, one hit
  // each: one convex solid.
  const SphereScene scene = BuildSphereScene();
  EXPECT_GE(std::ranges::count(scene.expected_hits, ExpectedHit::kHit), 20);
  EXPECT_GE(std::ranges::count(scene.expected_hits, ExpectedHit::kMiss), 20);
  const MovedRange moved = ComputeMovedRange(RunOn::kCpu, scene.range, ReactionArray::kGiven);
  ASSERT_TRUE(moved.result.has_value());
  const SphereSceneCheck check = ComputeSphereSceneCheck(scene, moved);
  std::cout << "sphere paths: largest |(hit - centre) x change| / |change| " << check.largest_lever_arm << " m\n";
  EXPECT_EQ(check.failing_particles, std::vector<std::size_t>{});
  EXPECT_LE(check.largest_lever_arm, 1e-12);
}

TEST(MoveParticle, SlidingOffFrictionlessSphere_LastHitWithinFifteenThousandthsOfAnalyticAngle) {
  // Steps of 1 ms take 2105 steps, each with at most one hit (one convex solid). Measured (a finding): the last hit is
  // at 0.8267744 rad, 0.0143 rad before the analytic angle, more than the 0.01 rad first expected. A response leaves
  // the velocity along the tangent at the hit, and the rest of the step follows that straight tangent, which rises off
  // a convex surface; so from about 0.75 rad the particle hops, up to 3,520 tol (3.3 micrometres) above the rest gap
  // (..._HopsAsHighAsOneStepsTangentRise), with a hit every few steps, and the last of those hops comes before the
  // angle. Over 4, 2, 1, 0.5, 0.25, 0.125 and 0.0625 ms the last hit is at 0.8369, 0.8420, 0.8268, 0.8356, 0.8287,
  // 0.8369 and 0.8372 rad: a scatter of about 0.01 rad at the coarser steps, of which 1 ms is the worst, settling near
  // 0.837, about 0.004 rad before the analytic angle, at the finer ones (..._SmallStepsLastHitNearAnalyticAngle). A
  // Python model of the scheme gave 0.8369.
  // The exact pin: the path uses only + - * /, Sqrt and comparisons (a fixed sphere does not turn, so no Sin), under
  // -ffp-contract=off or /fp:precise, so the 2105 steps give the same bits on every supported platform; acos is only in
  // the test's reading. One step moves the angle by about 1.6e-3 rad, so 1e-6 pins the exact step of the last hit, and
  // any deliberate change of the contact response must measure it again.
  const SlideOffSphere slide = ComputeSlideOffSphere(1e-3);
  std::cout << std::setprecision(10) << "sliding off: " << slide.step_count << " steps, last hit at "
            << slide.last_hit_angle << " rad (analytic " << kAnalyticLeavingAngle << ", difference "
            << slide.last_hit_angle - kAnalyticLeavingAngle << " rad)\n";
  ASSERT_FALSE(slide.has_failed);
  EXPECT_EQ(slide.step_count, 2105);
  EXPECT_LE(slide.largest_hit_count, 1);
  EXPECT_NEAR(slide.last_hit_angle, kAnalyticLeavingAngle, 0.015);
  EXPECT_NEAR(slide.last_hit_angle, 0.8267744296, 1e-6);
}

TEST(MoveParticle, SlidingOffFrictionlessSphere_HopsAsHighAsOneStepsTangentRise) {
  // The hops of the 1 ms slide: a response leaves the velocity along the tangent, and one step of travel v T along a
  // straight tangent rises (v T)^2 / (2 R) above the sphere. At the last hits v = Sqrt(2 g (1 - cos 0.83)) = 2.56 m/s,
  // so the rise is 3.3 micrometres. Measured: the highest end before the last hit is 3,520 tol = 3.278 micrometres
  // above the rest gap, and the largest rise of one step at the speeds of those steps is 3.157 micrometres: the hop is
  // 1.04 of it, a little higher because a hop can last a few steps, which the outward speed it leaves with lifts on.
  // Asserted: the hop within 10 % of the rise.
  const SlideOffSphere slide = ComputeSlideOffSphere(1e-3);
  std::cout << std::setprecision(6) << "sliding off: highest hop " << slide.largest_hop_height << " m, one step's "
            << "tangent rise " << slide.largest_tangent_rise << " m\n";
  ASSERT_FALSE(slide.has_failed);
  EXPECT_NEAR(slide.largest_hop_height, slide.largest_tangent_rise, 0.1 * slide.largest_tangent_rise);
}

// A groove of two fixed flat solids meeting along a line through the origin that descends at kGrooveSlope from the
// horizontal toward +x, each face tilted kGrooveFaceTilt from the groove's own horizontal plane: a groove whose faces
// meet at a right angle. In the groove's axes, down the line e = {cos b, 0, -sin b}, its up z' = {sin b, 0, cos b} and
// across y, the faces' normals are cos f z' + sin f y and cos f z' - sin f y.
constexpr double kGrooveSlope = std::numbers::pi / 6;  // rad, b: 30 degrees.

constexpr double kGrooveFaceTilt = std::numbers::pi / 4;  // rad, f: 45 degrees.

constexpr Vec3 kGrooveDown = {.x = 0.8660254037844387, .y = 0, .z = -0.5};  // {cos b, 0, -sin b}

constexpr Vec3 kGrooveUp = {.x = 0.5, .y = 0, .z = 0.8660254037844387};  // {sin b, 0, cos b}

// What a particle in the groove did over many steps.
struct GrooveRun {
  int move_count = 0;  // Successful moves.
  int exact_rest_count = 0;  // Moves that end with both velocities exactly 0.
  Vec3 start_position{};  // m.
  Vec3 end_position{};  // m.
  Vec3 end_velocity{};  // m/s, velocity.for_next_step at the end.
};

// Moves a particle at the groove's rest point, tol / cos f above its line, at rest, for 1000 steps of 1 ms with the
// given friction; each step subtracts 9.81 T from the z of both velocities.
GrooveRun ComputeGrooveRun(double friction) {
  constexpr int kStepCount = 1000;
  constexpr double kStepDuration = 1e-3;  // s
  const Vec3 across = {.x = 0, .y = std::sin(kGrooveFaceTilt), .z = 0};
  const Vec3 up = std::cos(kGrooveFaceTilt) * kGrooveUp;
  const std::vector<Solid> solids = {
      move_particle_cases::BuildFlatSolid({}, up + across, kNoMotion, 1),
      move_particle_cases::BuildFlatSolid({}, up - across, kNoMotion, 2),
  };
  GrooveRun run;
  run.start_position = (kTolerance / std::cos(kGrooveFaceTilt)) * kGrooveUp;
  Vec3 position = run.start_position;
  Vec3 velocity{};
  for (int step = 0; step < kStepCount; ++step) {
    velocity.z -= kGravity * kStepDuration;
    SolidContactInputs inputs = move_particle_cases::BuildCaseInputs({
        .start_position = position,
        .velocity = velocity,
        .step_duration = kStepDuration,
        .friction = friction,
    });
    inputs.solids = solids.data();
    inputs.solid_count = 2;
    SolidContactResult result;
    if (MoveParticle(inputs, &result, nullptr) != ContactError::kNone) {
      break;
    }
    ++run.move_count;
    position = result.position;
    velocity = result.velocity.for_next_step;
    const Vec3 this_step = result.velocity.for_this_step;
    const bool is_at_rest = velocity.x == 0 && velocity.y == 0 && velocity.z == 0 && this_step.x == 0
        && this_step.y == 0 && this_step.z == 0;
    run.exact_rest_count += is_at_rest ? 1 : 0;
  }
  run.end_position = position;
  run.end_velocity = velocity;
  return run;
}

TEST(MoveParticle, RestingInRightAngledGrooveWithFriction_StaysAtRest) {
  // The groove at 30 degrees with friction 0.5, 1000 steps of 1 ms. Each step gravity's velocity g T splits into
  // g T sin b down the line and g T cos b into the groove. The joint response removes the part into the groove with
  // two equal pushes of g T cos b / (2 cos f) each; their friction limits add, 2 mu g T cos b / (2 cos f), against the
  // g T sin b down the line. So the particle sticks when mu >= tan b cos f = 0.4082: with mu = 0.5 it takes the
  // groove's velocity, exactly 0, every step, and stays at its rest point (within tol / 64).
  const GrooveRun run = ComputeGrooveRun(0.5);
  EXPECT_EQ(run.move_count, 1000);
  EXPECT_EQ(run.exact_rest_count, 1000);
  CheckVec3IsNear(run.end_position, run.start_position, kTolerance / 64);
}

TEST(MoveParticle, InRightAngledGrooveWithLowFriction_SlidesDownTheLine) {
  // The same groove with friction 0.3, below tan b cos f = 0.4082: each step the particle keeps g T sin b down the line
  // less the friction limit mu g T cos b / cos f, so after N steps its speed is N g T (sin b - mu cos b / cos f) =
  // 1000 * 9.81e-3 * (0.5 - 0.3 * 0.8660 / 0.7071) = 1.3006 m/s down the line, within 1e-9 of it, with no speed across
  // or into the groove beyond 1e-12 m/s, and it travels about N T / 2 times that, 0.65 m, down the line.
  const GrooveRun run = ComputeGrooveRun(0.3);
  const double expected_speed = 1000 * kGravity * 1e-3
      * (std::sin(kGrooveSlope) - (0.3 * std::cos(kGrooveSlope) / std::cos(kGrooveFaceTilt)));  // m/s
  std::cout << std::setprecision(10) << "groove slide: speed " << math::Dot(run.end_velocity, kGrooveDown)
            << " m/s, expected " << expected_speed << ", travel " << math::Dot(run.end_position, kGrooveDown) << " m\n";
  EXPECT_EQ(run.move_count, 1000);
  EXPECT_EQ(run.exact_rest_count, 0);
  CheckVec3IsNear(run.end_velocity, expected_speed * kGrooveDown, 1e-9);
  EXPECT_NEAR(math::Dot(run.end_position, kGrooveDown), expected_speed * 0.5, 0.01);
}

// --- The cases of the old contact code ---

TEST(MainComparisonCases, ToZUp_IsAQuarterTurnAboutX) {
  // The old up {0, 1, 0} is this code's up {0, 0, 1}, and the turn keeps the right hand: x cross y is z in both.
  CheckEqualAxes(ToZUp({.x = 0, .y = 1, .z = 0}), {.x = 0, .y = 0, .z = 1});
  CheckEqualAxes(ToZUp({.x = 1, .y = 2, .z = 3}), {.x = 1, .y = -3, .z = 2});
  CheckEqualAxes(
      math::Cross(ToZUp({.x = 1, .y = 0, .z = 0}), ToZUp({.x = 0, .y = 1, .z = 0})),
      ToZUp({.x = 0, .y = 0, .z = 1})
  );
}

// Finds the old case of one source line, and fails the calling test when there is none.
MainComparisonCase FindMainCase(int source_line) {
  for (const MainComparisonCase& old_case : BuildMainComparisonCases()) {
    if (old_case.source_line == source_line) {
      return old_case;
    }
  }
  ADD_FAILURE() << "no old case at line " << source_line;
  return {};
}

TEST(MoveParticle, MainStartOnMiddleOfPlate_MovedOutWhereMainFailed) {
  // Line 337: the plate |x| <= 1 and a start at its middle with an overlap tolerance of 2 m. The old code could not
  // tell the side and failed; this code counts the middle plane as the kHighZ side (ComputeDistanceToSolid) and moves
  // the start out through kHighZ to the rest gap at x = 1 + tol, kind kStartedInsideSolid, at rest.
  const MainComparisonCase old_case = FindMainCase(337);
  ASSERT_EQ(old_case.agreement, MainAgreement::kDiffers);
  EXPECT_EQ(old_case.old_status, MainStatusCode::kFailedPrecondition);
  const ZUpMove move = ComputeZUpMove(old_case);
  ASSERT_EQ(move.error, ContactError::kNone);
  EXPECT_EQ(move.result.kind, ContactKind::kStartedInsideSolid);
  EXPECT_EQ(move.result.face, SolidFace::kHighZ);
  CheckVec3IsNear(move.result.position, {.x = 1 + kTolerance, .y = 0, .z = 0}, kTolerance / 64);
  CheckEqualAxes(move.result.velocity.for_next_step, {});
}

TEST(MoveParticle, MainSolidSlidingAtTenToTheTwenty_IsTooFastForStep) {
  // Line 456: a flat solid sliding along its surface at 1e20 m/s. The old code kept the particle's small sideways
  // speed; this code bounds the speeds involved times the time left by 2^40 tol (here 1024 m) and fails.
  const MainComparisonCase old_case = FindMainCase(456);
  ASSERT_EQ(old_case.agreement, MainAgreement::kDiffers);
  const ZUpMove move = ComputeZUpMove(old_case);
  EXPECT_EQ(move.error, ContactError::kTooCloseToCall);
  EXPECT_EQ(move.failure.close_call, ContactCloseCall::kMovesTooFastForStep);
  EXPECT_EQ(move.failure.solid_id, 1U);
}

}  // namespace
}  // namespace fluidity::contact
