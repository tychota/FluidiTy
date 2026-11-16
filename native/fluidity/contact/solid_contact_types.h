// The types of one particle's move through the solids, and what a hit does to the particle's velocity.
//
// Purpose: declare what one particle's move through the solids reads and returns, and change a particle's velocity
//   where it meets a solid's surface, so it does not move into the solid and loses some sideways speed to friction.
//   The contact functions that move a particle and their private steps (contact/solid_contact.h,
//   contact/solid_contact_steps.h) both build on this header, which needs neither.
//
// Idea: at a surface, a velocity splits into a part along the normal and a part along the surface, both measured
//   from the moving solid. A solid blocks motion into it and lets motion away from it pass, so only an approaching
//   velocity changes. Its speed into the solid is removed. Then friction takes friction times that removed speed off
//   the sideways speed. When that would take all the sideways speed or more, the particle sticks: it takes the
//   solid's velocity. Otherwise it slips with less sideways speed. Papers call this Coulomb friction.
//   Example: a fixed flat solid with normal {0, 0, 1} and a velocity {3, 0, -1} m/s. The approach speed is -1 m/s and
//   the sideways speed is 3 m/s. Friction 0.5 leaves {2.5, 0, 0} m/s; friction 3 or more leaves {0, 0, 0} m/s.
//
// Words:
//   hit             a moment in a step, along the particle's path or at its start, at which both its velocities get
//                   the joint response over every surface it is on, whatever their number; at most kMaxHitsPerStep
//                   per step. FirstHit is the earliest hit along one straight path
//   approach speed  m/s. The particle's speed along the normal out of the solid, measured from the moving solid:
//                   Dot(normal_out_of_solid, velocity - solid_velocity). Negative means approaching
//   sideways speed  m/s. The particle's speed along the surface, measured from the moving solid
//   friction        no unit, zero or more. How much sideways speed a solid takes off per unit of speed into it.
//                   Papers: the Coulomb coefficient
//   stick           the particle takes the solid's velocity, because friction takes all its sideways speed
//   slip            the particle keeps moving along the surface with less sideways speed
//   reaction        the change a solid makes to a particle's velocity for the next step, which the solid needs to
//                   compute the force on it
//   approach left after the response  the approach speed that rounding leaves in a velocity the response returned:
//                   zero up to a few rounding units, of either sign
//   tangent contact  two convex solids touching at a point or along a line, a ball resting on a floor or two touching
//                   balls; the free space around the contact is a cusp, whose angle shrinks to 0
//   wedge angle     of two surfaces, 180 degrees minus the angle between their normals
// Owns:    ContactKind, kMaxHitsPerStep, kMaxAdvanceCount, kTurningSolidHitGapShare, kTouchedSolidRestingGapShare,
//          kBoundingSurfaceRestingDepthShare, kBoundingSurfaceSinkShare, kRimHitGapShare, kMaxRimAdvanceCount,
//          kMaxRestGapMoveInTolerances, kMaxRestGapIterations, kMaxConstrainingSurfaces, kMaxSlideDirections,
//          kMaxUsedDirections, kMaxFrictionOrders, kSameNormalAxisDifference, kMaxCorrectionRefinements,
//          kVelocityPairLimitShare, kVelocityTripleLimit, kMaxSqueezeOutSpeedFactor, SolidContactInputs, SolidReaction,
//          SolidContactResult, FirstHit, CorrectedQuantity, CorrectionTargets, SmallestCorrection,
//          kApproachLeftAfterResponseShare, ComputeLargestApproachLeftAfterResponse, ComputeContactResponse
// Uses:    Vec3, math::Dot, math::Length, math::IsFinite (base/vec3.h), math::Abs (base/scalar_math.h),
//          PerParticleVelocity (base/per_particle_velocity.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), Solid,
//          PlateSide, SolidFace, SolidSide, DistanceToSolid, kNoSolid (contact/solid_shapes.h)
// Used by: contact/solid_contact.h and contact/solid_contact_steps.h, and through them the MoveParticles stage
// Read:    docs/method-paper/fluidity-method.typ <sec-contact-response>, <eq:contact-response>,
//          <fig-contact-response>
//          Stomakhin 2013, doi:10.1145/2461912.2461948

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// What happened to a particle during one step's move.
enum class ContactKind : std::uint8_t {
  kNone,  // The particle met no solid.
  kHit,  // The path met a solid during the step.
  kStartedInsideSolid,  // The start was slightly inside a solid and was moved out to the contact tolerance.
};

// The most hits one step allows: a hit is one moment at which both velocities get the joint response, whatever the
// number of surfaces the particle is on, so a wedge or a corner of three faces is one hit. Four bound the responses and
// the sweeps of a step; a particle that keeps meeting solids at separate moments is in a gap too narrow for the step.
inline constexpr int kMaxHitsPerStep = 4;
// The most advances of the stepping search toward a turning solid in one sweep, a cap on its cost. Each advance is
// the larger of the gap over the fastest the gap can shrink and the second-order advance, which follows the true
// rate at which the gap closes: 3 in the method paper's example; first order needed 30. A path that hovers near the
// solid or grazes it, with almost no motion along the normal, needs more and ends in a close call, which is rare
// (FindHitOnTurningSolid states measured counts).
inline constexpr int kMaxAdvanceCount = 128;
// No unit. The stepping search toward a turning solid stops with a hit once the gap is at most this share of the
// contact tolerance, so a hit after at least one advance sits between tol and 1.25 tol from the surface (a hit at 0
// reports the start's distance, at most 1.25 tol). A quarter gives the search a range of tol / 4 to land in, while
// the hit stays far from the touching rule's window [-tol, 0] and from the end check at tol / 2. A power of two
// keeps the test exact.
inline constexpr double kTurningSolidHitGapShare = 0.25;
// No unit. The share of the contact tolerance above zero up to which the start gap to a solid still counts as resting
// on it once the particle had a response in this step, and the window of the active surfaces, tol + tol / 16.
// Rounding leaves a hit position a few rounding units off the rest gap (at most tol / 128,
// contact/contact_tolerance.h), so a sixteenth is far above it, and far below the tol that the touching rule allows
// below zero. A power of two keeps the test exact.
inline constexpr double kTouchedSolidRestingGapShare = 0x1p-4;
// No unit. How far below its rest gap, as a share of the contact tolerance, a start may be and still rest on a
// bounding surface that does not give its distance (plan decision P21, item 1). A particle at rest is within tol / 128
// of the rest gap after the settle and comes at most tol / 16 nearer after a response; an eighth holds both. A power
// of two keeps the test exact.
inline constexpr double kBoundingSurfaceRestingDepthShare = 0x1p-3;
// No unit. The most a path may sink toward a bounding surface it rests on during one sweep, as a share of the contact
// tolerance: what a response may leave, allowance * time left <= tol / 64, so a velocity a response left never sinks
// more, and neither do coplanar faces whose normals differ by rounding.
inline constexpr double kBoundingSurfaceSinkShare = 0x1p-6;
// No unit. The rim search stops with a hit once the gap is at most this share of the contact tolerance, so a rim hit
// is at most tol + tol / 1024 from the rim. A path that passes the rim within tol / 1024 is hit, and loses at most
// Sqrt(2 * 2^-10) = 0.044 of its speed; a finer share is below the rounding of a distance where the tolerance is at its
// floor of 2^-40 of the scene.
inline constexpr double kRimHitGapShare = 0x1p-10;
// The most advances of the rim search in one sweep; at the limit it reports the hit where it stands, the safe side. A
// path tangent to the round rim is within the hit share after about 5 advances (the gap falls about fourfold per
// advance), so 16 is never reached on the measured paths.
inline constexpr int kMaxRimAdvanceCount = 16;

// No unit, in contact tolerances. The longest move off several near surfaces at once, at the start or in the settle
// (MoveToRestGapOfNearSurfaces). The particle is displaced without a matching velocity, so the move must stay small
// against a cell, which the grid and the step's limit on how far a particle moves resolve, not against tol. With tol =
// 2^-30 h + 2^-40 L (ComputeContactTolerance, h the smallest cell width, L the largest coordinate), 2^22 tol = h / 256
// + L / 2^18: a 256th of a cell plus 4 micrometres in a one-metre scene. It covers the rest point of every tangent
// contact: a ball of radius R <= L resting on a floor rests 2 Sqrt(R tol) <= 2 Sqrt(2^40) tol = 2^21 tol from the
// contact point. A flat wedge needs more only when it is sharper than about 2^-21 rad, where 2 shortfall / wedge angle
// passes 2^22 tol.
inline constexpr double kMaxRestGapMoveInTolerances = 0x1p22;
// The most iterates of the move off several near surfaces. Two flat faces need one, and one more to see that it
// moves no more. The cusp under a ball resting on a floor needs 5 at R / tol = 2^29, and after 8 it is within 1e-4 of
// its rest point at R / tol = 2^40, the largest radius a scene allows (both measured, move_particle_test.cc).
inline constexpr int kMaxRestGapIterations = 8;
// The most directions one move off near surfaces reads at once: the three faces of a corner and one more. Surfaces with
// the same normal share one (kSameNormalAxisDifference); a start near surfaces facing a fifth way is
// kStartsInNarrowGap.
inline constexpr int kMaxConstrainingSurfaces = 4;
// The most directions one joint response reads at once. In three dimensions the smallest change meets
// at most three of them as equalities (three independent normals fix a vector; a cone spanned by more is spanned by
// three of them), so SolveSmallestCorrection searches the subsets of at most three among all of them: 8 + 28 + 56 = 92
// at this limit. Eight is twice the directions one move reads: a domain corner's three sides, two chamfers across it
// and more stay within it.
inline constexpr int kMaxSlideDirections = 2 * kMaxConstrainingSurfaces;
// The most directions one smallest change of a velocity uses: three independent normals fix a vector.
inline constexpr int kMaxUsedDirections = 3;
// The orders in which kMaxUsedDirections surfaces can apply their friction one after the other: 3 * 2 * 1.
inline constexpr int kMaxFrictionOrders = 6;
static_assert(
    kMaxFrictionOrders == kMaxUsedDirections * (kMaxUsedDirections - 1) * (kMaxUsedDirections - 2),
    "the joint response walks every order of kMaxUsedDirections surfaces"
);
// No unit. Two near surfaces whose unit normals differ by at most this on every axis share one target of the move off
// near surfaces, the larger: for one normal n, Dot(n, w) >= max(t_1, t_2) meets both. Normals that differ by 2^-40 on
// each axis let the shared target miss the other by at most Sqrt(3) 2^-40 |w| <= 2^-17 tol within the move cap of
// 2^22 tol, far below the move's slack of tol / 1024. Coincident and stacked surfaces then count once toward
// kMaxConstrainingSurfaces: a ground lying on the domain's bottom has exactly the bottom's normal.
inline constexpr double kSameNormalAxisDifference = 0x1p-40;
// The most extra solves SolveSmallestCorrection makes for a velocity on the same surfaces, each removing the residual
// the one before left. Each multiplies the residual by about 2^-52 / (1 + c), c the dot of a pair's normals (measured
// at most 1.25 rounding units over 1 + c, kVelocityPairLimitShare); at the pair limit below, 3 leave at most 16
// rounding units of the speeds (measured), so 4 always suffice.
inline constexpr int kMaxCorrectionRefinements = 4;
// No unit. A pair of surfaces is used for a velocity only when 1 + Dot(n_A, n_B) >= 2^-40: a wedge angle of at least
// 2 asin(2^-20.5) = 1.35e-6 rad. One solve leaves at most 0.75 to 1.25 rounding units of 2^-52 over (1 + c) of the
// target (measured over 3000 pairs at each 1 + c = 2^-40, 2^-28, 2^-16, 2^-4; the test bounds it by 4), so at the
// limit the extra solves still converge. Every cusp a particle can rest in is above it: at the
// rest point the angle is about 2 Sqrt(tol / R) >= 2^-19 rad, so 1 + c >= 2^-39. Positions need no such limit: an
// imprecise iterate is corrected by the next one, and the move's last test reads the true distances.
inline constexpr double kVelocityPairLimitShare = 0x1p-40;
// No unit. A triple of surfaces is used for a velocity only when |Dot(n_1, Cross(n_2, n_3))| >= 2^-20, the triple
// product of their unit normals: three faces whose normals are nearly in one plane leave the velocity's change
// ill-conditioned, as a pair below kVelocityPairLimitShare does.
inline constexpr double kVelocityTripleLimit = 0x1p-20;
// No unit. The fastest velocity a joint response may give, as a multiple of the largest speed involved: the particle's
// two velocities before the response and the velocities of the active surfaces' solids at the particle. A wedge closing
// on a particle pushes it out at the closing speed over the wedge's sine, unbounded as the wedge flattens (measured
// without a cap: a 1e-5 rad wedge closing at 1 m/s, 1e5 m/s, a kilometre in a step of 0.01 s), and a particle thrown
// that far is worse than a squeeze reported to the step. 64 lets one face closing at speed V on a fixed one carry the
// particle out at V / sin(a) down to a = asin(1 / 64) = 0.8953 degrees, and two faces closing symmetrically, each at V
// along its own normal, at V / sin(a / 2) down to 2 asin(1 / 64) = 1.7906 degrees. Both limits are closed forms, and
// the code meets them: the cap reads the speeds before any friction. A power of two keeps the test exact. The geometry
// and the speeds decide, not the step length, so a shorter step does not cure a squeeze it finds.
inline constexpr double kMaxSqueezeOutSpeedFactor = 64;

// What one particle's move through the solids reads for one step.
struct SolidContactInputs {
  Vec3 start_position{};  // m, world, at the start of the step.
  PerParticleVelocity<Vec3> velocity{};  // m/s, world, before the response. for_this_step moves the particle.
  const Solid* solids = nullptr;  // solid_count solids, in ascending solid_id order.
  std::uint32_t solid_count = 0;
  double step_duration = 0;  // s, positive.
  double friction = 0;  // No unit, zero or more. Papers: Coulomb coefficient.
  double start_overlap_tolerance = 0;  // m, zero or more. The deepest start inside a solid that is moved out.
  double contact_tolerance = 0;  // m, positive. From ComputeContactTolerance.
  PlateSide plate_side{};  // The side this particle is committed to, or {0, kNoPlate}.
};

// What a solid needs to compute the force on it from one particle's move.
struct SolidReaction {
  std::uint32_t solid_index = kNoSolid;  // Index into solids, not the solid_id. kNoSolid: no reacting solid was hit.
  Vec3 hit_position{};  // m, world, where the first change of velocity.for_next_step against this solid happened.
  // m/s, world. velocity.for_next_step after minus before, summed over every change this solid caused.
  Vec3 velocity_for_next_step_change{};
};
static_assert(sizeof(SolidReaction) == 56, "the step's memory estimate charges 56 bytes per particle for a reaction");

// What one particle's move through the solids returns for one step.
struct SolidContactResult {
  Vec3 position{};  // m, world, at the end of the step.
  PerParticleVelocity<Vec3> velocity{};  // m/s, world, after the response.
  ContactKind kind = ContactKind::kNone;
  double hit_fraction = 0;  // No unit, 0 to 1. When the first hit happens, as a share of the step. 0 without a hit.
  Vec3 normal_out_of_solid{};  // No unit, unit vector, world, at the first hit. Zero without a hit.
  PerParticleVelocity<double> approach_speed{};  // m/s, at the first hit. Negative means approaching.
  std::uint32_t solid_index = kNoSolid;  // Index into solids of the first hit's solid, not its solid_id.
  SolidFace face = SolidFace::kHighZ;  // The face of the first hit.
  SolidSide side_at_hit = SolidSide::kNoPlate;  // The side of the plate that was hit, when it is a plate.
  // Hits during the step, 0 to kMaxHitsPerStep: moments with a joint response, not surfaces. A corner is one hit.
  std::uint8_t hit_count = 0;
  SolidReaction reaction{};
};

// The first hit along one straight path, or none.
struct FirstHit {
  double elapsed = 0;  // s, since the start of the sweep.
  std::uint32_t solid_index = kNoSolid;  // Index into solids, not the solid_id. kNoSolid: the path is free.
  DistanceToSolid at_hit{};  // The distance, normal and face at the hit.
};

// What SolveSmallestCorrection changes: it sets how the surfaces are solved and how much rounding a result may keep.
enum class CorrectedQuantity : std::uint8_t {
  // A point (m). No refinement and no conditioning limit: an imprecise iterate is corrected by the next one. A
  // candidate may also miss each target by the rounding of composing the change, 2^-49 of the summed multipliers.
  kPosition,
  // A velocity (m/s). Up to kMaxCorrectionRefinements more solves remove the residual, and a pair or a triple is used
  // only above kVelocityPairLimitShare or kVelocityTripleLimit.
  kVelocity,
};

// What SolveSmallestCorrection reads: count surfaces, each a unit normal and a target. It looks for the smallest
// change w with Dot(normals[i], w) >= targets[i] for every surface.
struct CorrectionTargets {
  // A C array: GPU code cannot call std::array's member functions (nvcc without --expt-relaxed-constexpr).
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): see the comment above.
  Vec3 normals[kMaxSlideDirections]{};  // No unit, unit vectors, world.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as above.
  double targets[kMaxSlideDirections]{};  // m for a position, m/s for a velocity: the least change along each.
  // 1 to kMaxConstrainingSurfaces for a position (the move off near surfaces reads no more), 1 to kMaxSlideDirections
  // for a velocity.
  int count = 0;
  // m/s, for a velocity only: |v|_1 + sum over the surfaces' solids of |u_i|_1, the speeds before the change. With the
  // change's own |w|_1 it gives the change's allowance, kApproachLeftAfterResponseShare times their sum, the least
  // slack a velocity's candidate gets (ComputeCorrectionSlack). 0 for a position.
  double speeds_before_change = 0;
  // No unit, for a velocity only: the friction (friction * friction_scale) of each direction's pushing surface. Among
  // the changes no longer than the shortest plus its slack, the solver keeps the one with the largest sum of
  // frictions[k] * multipliers[k] (plan decision P20, item 3). All 0 for a position.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as above.
  double frictions[kMaxSlideDirections]{};
};

// The smallest change SolveSmallestCorrection found, or none: change = sum over k < used_count of multipliers[k] *
// normals[used_surfaces[k]], the surfaces it met as equalities.
struct SmallestCorrection {
  bool is_found = false;  // false: no subset of one, two or three surfaces meets every target.
  Vec3 change{};  // m or m/s, world. Zero when none was found, or when a velocity needs none.
  // 1 to 3 when found; 0 for a velocity that already meets every target within its slack, with a zero change.
  int used_count = 0;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  int used_surfaces[3]{};  // Indexes into the targets, ascending.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a C array, as in CorrectionTargets.
  double multipliers[3]{};  // m or m/s: how far the change goes along each used normal, zero or more before refinement.
  int refinement_count = 0;  // 0 to kMaxCorrectionRefinements; always 0 for a position.
};

// No unit. The share of the speeds involved that bounds the approach speed left after the response: 64 rounding units
// of 2^-52, about 9 times the at most 7 that the derivation of ComputeLargestApproachLeftAfterResponse gives.
inline constexpr double kApproachLeftAfterResponseShare = 0x1p-46;

// Computes the largest size that Dot(normal_out_of_solid, result - solid_velocity), computed in doubles, can have when
// result is ComputeContactResponse(velocity, solid_velocity, normal_out_of_solid, friction), for any unit normal and
// friction.
//
// Method: kApproachLeftAfterResponseShare * A, with A = |velocity.x| + |velocity.y| + |velocity.z| +
//   |solid_velocity.x| + |solid_velocity.y| + |solid_velocity.z|: each vector's three in x, y, z order, then the two
//   sums. A sum of absolute axes, not a
//   Length: it squares nothing, so it neither underflows for tiny speeds nor overflows for large ones, and it is at
//   least S = Length(velocity) + Length(solid_velocity), so a bound in S holds in A.
//   Derivation, with e = 2^-52: one operation rounds by at most e / 2 of its result, and a dot product with a unit
//   normal by at most 2 e of the length of its other vector (its subtraction and its three products and two sums).
//   1 The first removal leaves w with w - solid_velocity the sideways part of velocity - solid_velocity, so
//     |w - solid_velocity| <= S and |w| <= 2 S, up to terms of order e S.
//   2 The second removal recomputes that approach with an error of at most 2 e S. Its subtraction rounds each axis of
//     w by at most e / 2 of that axis, so the approach by at most e / 2 * |w| <= e S. At most 3 e S of approach
//     stays. With friction 0 that is the result.
//   3 A slip returns w + fraction * (solid_velocity - w), fraction in [0, 1]: the exact blend keeps (1 - fraction)
//     times the approach left, at most 3 e S, and its subtraction, product and sum round by at most e / 2 * S,
//     e / 2 * S and e / 2 * 2 S along the normal: at most 2 e S more, 5 e S in all. A stick returns solid_velocity,
//     whose approach computes to exactly 0.
//   4 The caller's own Dot of the result adds at most 2 e S. Total: at most 7 e S <= 7 e A, below the 64 e A of the
//     bound.
//   The normal's length, within 2^-41 of 1, changes these figures by a negligible share. Below 2^-1022 an operation
//   rounds by up to 2^-1075 absolute instead; the about 40 operations of the response and the Dot add at most
//   2^-1069, far below 57 e A for any A >= 2^-1000. A Python model of the code found at most 0.72 e A over 150,000
//   approaching cases with A from 2^-1000 to 2^500 m/s, slips and frictions just below the stick limit included.
// Units: m/s, for the two velocities and the result.
// Example: velocity {-2, 0, -2}, a fixed solid: A = 4 m/s and the bound is 2^-44 = 5.7e-14 m/s. With normal
//   {0.6, 0, 0.8} and friction 0, the response leaves an approach of -2^-55 = -2.8e-17 m/s. The same velocity times
//   2^-830 (about 1.4e-250 m/s) leaves -2^-885 against a bound of 2^-874, where a bound built from Length would be 0,
//   because the squares underflow.
// Requires: velocity and solid_velocity are finite, and A is 0 or between 2^-1000 (about 1e-301) and 2^500 (about
//   3e150) m/s: above the speeds where rounding becomes absolute, and below those where the response's Length
//   overflows its squares.
// Read: docs/method-paper/fluidity-method.typ <sec-contact-response>
FLUIDITY_CPU_AND_GPU inline double ComputeLargestApproachLeftAfterResponse(Vec3 velocity, Vec3 solid_velocity) {
  assert(math::IsFinite(velocity));
  assert(math::IsFinite(solid_velocity));
  // Method paper <sec-contact-response>:  A = sum of |v_i| + sum of |u_i|
  const double velocity_axes_sum = math::Abs(velocity.x) + math::Abs(velocity.y) + math::Abs(velocity.z);
  const double solid_velocity_axes_sum =
      math::Abs(solid_velocity.x) + math::Abs(solid_velocity.y) + math::Abs(solid_velocity.z);
  const double speeds_involved = velocity_axes_sum + solid_velocity_axes_sum;  // m/s
  assert(speeds_involved == 0 || speeds_involved >= 0x1p-1000);
  assert(speeds_involved <= 0x1p500);
  return kApproachLeftAfterResponseShare * speeds_involved;
}

// Computes one velocity after it meets a solid's surface: no speed into the solid up to rounding, and the sideways
// speed reduced by friction.
//
// Method: with approach_speed = Dot(normal_out_of_solid, velocity - solid_velocity):
//   1 approach_speed >= 0: the velocity leaves or slides along the surface and comes back unchanged, bit for bit.
//   2 velocity - approach_speed * normal_out_of_solid removes the approach. The approach left by rounding is
//     computed from that velocity and removed once more.
//   3 friction 0: that velocity is the result. Otherwise sideways_speed = Length(that velocity - solid_velocity) and
//     into_solid_speed = -approach_speed. The particle sticks, and the result is solid_velocity exactly, when
//     friction >= sideways_speed / into_solid_speed; with no sideways speed any positive friction sticks. Otherwise it
//     slips: that velocity + (friction * into_solid_speed / sideways_speed) * (solid_velocity - that velocity).
//   Afterwards the approach speed Dot(normal_out_of_solid, result - solid_velocity) is zero only up to rounding: at
//   most ComputeLargestApproachLeftAfterResponse(velocity, solid_velocity) in size, and possibly slightly negative,
//   still approaching. A caller that tests whether a returned velocity still approaches compares with that bound,
//   not with 0. A stick leaves exactly 0.
//   Same bits on CPU and GPU: only + - * / and Length, which base/ promises the same bits for.
// Units: m/s, world, for the three velocities. friction has no unit; normal_out_of_solid has none.
// Example: normal {0, 0, 1}, a fixed solid, velocity {3, 0, -1}: approach speed -1, sideways speed 3. Friction 0.5
//   gives {2.5, 0, 0}; friction 3 or 4 gives {0, 0, 0}. Velocity {0, 0, -2} at a solid moving at {0, 0, 0.5}
//   gives {0, 0, 0.5}. Velocity {1, 0, 2} comes back unchanged. Normal {0.6, 0, 0.8}, a fixed solid, velocity
//   {-2, 0, -2}, friction 0: the result's approach is -2^-55 m/s, within the bound of 5.7e-14 m/s.
// Requires: velocity and solid_velocity are finite. normal_out_of_solid is a unit vector: its squared length is
//   within 2^-40 of 1. friction is zero or more; +infinity is allowed and always sticks.
// Read: docs/method-paper/fluidity-method.typ <eq:contact-response>, <fig-contact-response>
//       Stomakhin 2013, doi:10.1145/2461912.2461948
FLUIDITY_CPU_AND_GPU inline Vec3
ComputeContactResponse(Vec3 velocity, Vec3 solid_velocity, Vec3 normal_out_of_solid, double friction) {
  assert(math::IsFinite(velocity));
  assert(math::IsFinite(solid_velocity));
  assert(math::Abs(math::Dot(normal_out_of_solid, normal_out_of_solid) - 1) <= 0x1p-40);
  assert(friction >= 0);
  // Steps: 1 keep a velocity that does not approach  2 remove the approach  3 friction
  // Step 1: the approach is measured from the moving solid.
  // Method paper <eq:contact-response>:  a = n . (v - u)
  const double approach_speed = math::Dot(normal_out_of_solid, velocity - solid_velocity);
  if (approach_speed >= 0) {
    return velocity;
  }

  // Step 2: the approach is removed from the world velocity, not from the velocity measured from the solid, which
  // would then be added back. Near a fast solid that round trip would erase a small sideways velocity:
  // (1e-9 - 1e8) + 1e8 is 0, since doubles near 1e8 are 1.5e-8 apart. The second removal takes off most of what the
  // rounding of the first one left; what still stays is bounded by ComputeLargestApproachLeftAfterResponse.
  // Method paper <eq:contact-response>:  v' = v - a n
  Vec3 velocity_along_surface = velocity - approach_speed * normal_out_of_solid;
  const double approach_left_by_rounding = math::Dot(normal_out_of_solid, velocity_along_surface - solid_velocity);
  velocity_along_surface = velocity_along_surface - approach_left_by_rounding * normal_out_of_solid;
  if (friction == 0) {
    return velocity_along_surface;
  }

  // Step 3: friction takes friction * into_solid_speed off the sideways speed. The stick test compares friction with
  // the quotient sideways_speed / into_solid_speed, as the method paper's equation does, so a friction equal to that
  // double sticks; the product form friction * into_solid_speed >= sideways_speed rounds differently at the limit.
  // Overflow is no reason for the division: an infinite product would still compare correctly. With no sideways
  // speed the quotient is 0, so any positive friction sticks.
  // Method paper <eq:contact-response>:  stick when mu >= s_side / (-a)
  const double into_solid_speed = -approach_speed;  // m/s, positive.
  const double sideways_speed = math::Length(velocity_along_surface - solid_velocity);
  const bool friction_takes_all_sideways_speed = friction >= sideways_speed / into_solid_speed;
  if (friction_takes_all_sideways_speed) {
    return solid_velocity;
  }
  // Method paper <eq:contact-response>:  v'' = v' + (mu (-a) / s_side) (u - v')
  const double friction_fraction = friction * into_solid_speed / sideways_speed;
  return velocity_along_surface + friction_fraction * (solid_velocity - velocity_along_surface);
}

}  // namespace fluidity::contact
