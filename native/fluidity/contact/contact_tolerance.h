// The contact tolerance: the distance at which a particle rests on a solid, from the scale of the scene, and the
// check that the scene's thinnest feature is wide enough for it.
//
// Purpose: give contact one length that sits far above the rounding of every distance to a solid, so a particle
//   kept that far outside a surface is outside by any later evaluation, and fail the check of a scene that length
//   cannot protect.
//
// Idea: a distance to a solid computed with doubles is wrong by an amount that grows with the size of the
//   coordinates, not with the size of the solid. So the tolerance grows with the largest coordinate a solid reaches
//   during the step, plus a small share of the smallest cell width that keeps it positive for a scene at the origin:
//     contact_tolerance = 2^-30 * smallest_cell_width + 2^-40 * largest_coordinate
//   Why 2^-40: the machine epsilon is 2^-52, about 2.2e-16, the gap between 1 and the next double. Computing a
//   distance subtracts the origin, turns the point with three-term dot products and applies the shape's formula.
//   Together they misplace it by at most about 32 epsilons of the largest coordinate. 2^-40 is 4096 epsilons,
//   128 times that error. A hit is placed about as finely: to first order the closed forms put it within 14 epsilons
//   of the largest own coordinate along the path plus 3 epsilons of the largest coordinate, at most tol / 64 when
//   every coordinate is within 2^40 tol (FindSphereHit in contact/solid_contact_steps.h). That premise is
//   MoveParticle's to require: the check below tests thickness only, and only this tolerance's 2^-40 share, with the
//   particles inside the largest coordinate, gives it.
//   Example: largest coordinate 10 m, smallest cell 0.01 m: 9.31e-12 + 9.09e-12 = 1.84e-11 m. The rounding of a
//   distance is at most 32 * 2.2e-16 * 10 = 7.1e-14 m, 259 times smaller.
//   Every sweep runs against the solid grown by the tolerance, so a particle in contact rests one tolerance
//   outside the surface: the rest gap. A solid or a cell narrower than 1024 tolerances fails the check rather than
//   being protected badly: with the defaults, a feature about 1e9 times smaller than its distance from the origin,
//   or about 1e6 times smaller than the smallest cell. The tolerance is never made larger to let contact pass: a larger
//   tolerance would swallow the thin feature it is meant to protect.
//
// Words:
//   contact tolerance  the distance from a solid's surface at which a particle rests; far above rounding. A kept
//                      distance, not an allowed error
//   rest gap           the one contact tolerance a particle in contact keeps between itself and the surface
//   reach              how far a bounded solid's surface can be from its origin: Length(half_size)
// Owns:    kContactToleranceRelative, kContactToleranceFloorInCellWidths, kMinFeatureInTolerances, ContactScale,
//          ComputeContactScale, ComputeContactTolerance, CheckContactTolerance
// Uses:    Vec3, math::Length (base/vec3.h), math::Abs, math::IsFinite, math::Max, math::Min (base/scalar_math.h),
//          FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), ContactError (contact/contact_error.h), Solid, SolidShape,
//          ComputeSolidPose (contact/solid_shapes.h)
// Used by: the step, which computes and checks the tolerance once per step, and every contact function that takes
//          a contact_tolerance
// Read:    docs/method-paper/fluidity-method.typ <sec-contact-tolerance>, <eq:contact-tolerance>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// No unit. The share of the largest coordinate in the contact tolerance: 2^-40, 4096 machine epsilons, 128 times
// the rounding of a computed distance (about 32 epsilons of the largest coordinate).
inline constexpr double kContactToleranceRelative = 0x1p-40;
// No unit. The share of the smallest cell width in the contact tolerance. It keeps the tolerance positive for a
// scene at the origin, and equals kMinFeatureInTolerances * kContactToleranceRelative.
inline constexpr double kContactToleranceFloorInCellWidths = 0x1p-30;
// No unit. The thinnest solid and the smallest cell must each be at least this many contact tolerances wide.
// 1024 * 2^-40 = 2^-30 and 1024 * 2^-30 = 2^-20, so with the defaults a feature fails only when it is about 1e9
// times smaller than its distance from the origin, or about 1e6 times smaller than the smallest cell.
inline constexpr double kMinFeatureInTolerances = 1024;
static_assert(kMinFeatureInTolerances * kContactToleranceRelative == kContactToleranceFloorInCellWidths);

// The lengths of a scene that set its contact tolerance and its check.
struct ContactScale {
  // m. The largest |x|, |y| or |z| a solid origin takes during the step, plus that solid's reach, its
  // Length(half_size), for a box, a domain box, a sphere or a cylinder; the largest over all solids. The domain box
  // brings the particles' extent, since every particle is inside it.
  double largest_coordinate = 0;
  double smallest_cell_width = 0;  // m, positive.
  // m, positive. The smallest plate thickness (2 * half_thickness), box or domain box width (2 * smallest half
  // size), sphere diameter (2 * radius) or cylinder diameter or height (min(2 * radius, 2 * half_height)); the
  // smallest cell width when no solid has a thickness.
  double thinnest_solid = 0;
};

// Computes the lengths of a scene that set its contact tolerance, for one step of step_duration seconds.
//
// Method: for each solid, the largest |coordinate| of its origin at the start and at the end of the step (from
//   ComputeSolidPose; the origin moves in a straight line, so its two ends bound it), plus its reach: Length(half_size)
//   for a box, a domain box, a sphere or a cylinder, nothing for a flat solid or a plate. largest_coordinate is the
//   largest of these over the solids, 0 without a solid. thinnest_solid is the smallest thickness: 2 * half_thickness
//   for a plate, 2 * min(half_size) for a box or a domain box, 2 * radius for a sphere, min(2 * radius,
//   2 * half_height) for a cylinder; a flat solid has none. With no thickness at all it is smallest_cell_width.
//   Same bits on CPU and GPU: only + and *, Abs, Max, Min and Length, which base/ promises the same bits for.
// Units: seconds for step_duration; m for smallest_cell_width and the result.
// Example: a plate of half thickness 0.002 m at the origin and a box of half sizes {1, 2, 3} at {10, 0, 0}:
//   thinnest_solid 0.004, largest_coordinate 10 + Length({1, 2, 3}) = 13.74. A flat solid through the origin moving at
//   {0, 0, -5} m/s with step_duration 2 s: largest_coordinate 10, thinnest_solid smallest_cell_width.
// Requires: solids points to solid_count solids (it may be null when solid_count is 0), each of which passed
//   CheckSolid for step_duration, so no value is NaN: math::Max and math::Min would drop a NaN and give too small a
//   scale. step_duration and smallest_cell_width are finite and positive.
// Read: docs/method-paper/fluidity-method.typ <sec-contact-tolerance>

// clang-tidy: the count follows its array, as in every pointer-and-count pair, and the two lengths follow it, as
// contact design 3.3 orders the parameters; the swappable-parameters check flags the count and the two doubles.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU inline ContactScale
ComputeContactScale(const Solid* solids, std::uint32_t solid_count, double step_duration, double smallest_cell_width) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  assert(solids != nullptr || solid_count == 0);
  assert(math::IsFinite(step_duration) && step_duration > 0);
  assert(math::IsFinite(smallest_cell_width) && smallest_cell_width > 0);
  // Steps: 1 reach and thickness of each solid  2 the cell width when no solid has a thickness
  const auto compute_largest_coordinate = [](Vec3 point) -> double {
    return math::Max(math::Max(math::Abs(point.x), math::Abs(point.y)), math::Abs(point.z));
  };
  double largest_coordinate = 0;
  double thinnest_solid = 0;
  bool has_thickness = false;

  // Step 1: a flat solid and a plate are unbounded sideways, so they have no reach; their origin is the point of
  // their surface or middle plane that contact measures from.
  for (std::uint32_t solid_index = 0; solid_index < solid_count; ++solid_index) {
    const Solid& solid = solids[solid_index];
    const Vec3 half_size = solid.half_size;
    const Vec3 start_origin = solid.pose_at_step_start.origin;
    const Vec3 end_origin = ComputeSolidPose(solid, step_duration).origin;
    const double origin_coordinate =
        math::Max(compute_largest_coordinate(start_origin), compute_largest_coordinate(end_origin));
    double reach = 0;  // m
    double thickness = 0;  // m
    switch (solid.shape) {
      case SolidShape::kFlatSolid:
        break;
      case SolidShape::kPlate:
        thickness = 2 * half_size.z;
        break;
      case SolidShape::kBox:
      case SolidShape::kDomainBox:
        reach = math::Length(half_size);
        thickness = 2 * math::Min(math::Min(half_size.x, half_size.y), half_size.z);
        break;
      case SolidShape::kSphere:
        reach = math::Length(half_size);
        thickness = 2 * half_size.x;
        break;
      case SolidShape::kCylinder:
        // Doubling is exact, so 2 * min(radius, half_height) is min(2 * radius, 2 * half_height).
        reach = math::Length(half_size);
        thickness = 2 * math::Min(half_size.x, half_size.z);
        break;
    }
    largest_coordinate = math::Max(largest_coordinate, origin_coordinate + reach);
    const bool shape_has_thickness = solid.shape != SolidShape::kFlatSolid;
    const bool is_thinner = !has_thickness || thickness < thinnest_solid;
    if (shape_has_thickness && is_thinner) {
      thinnest_solid = thickness;
      has_thickness = true;
    }
  }

  // Step 2: with only flat solids, the cells are the thinnest feature contact must protect.
  if (!has_thickness) {
    thinnest_solid = smallest_cell_width;
  }
  return {
      .largest_coordinate = largest_coordinate,
      .smallest_cell_width = smallest_cell_width,
      .thinnest_solid = thinnest_solid,
  };
}

// Computes the contact tolerance of a scene: kContactToleranceFloorInCellWidths * smallest_cell_width +
// kContactToleranceRelative * largest_coordinate.
//
// Method: both shares are powers of two times a length, so for any cell width above 1e-290 m each product is exact
//   and only the sum rounds, correctly. Same bits on CPU and GPU. thinnest_solid is not used: CheckContactTolerance
//   reads it.
// Units: m.
// Example: largest_coordinate 0, smallest_cell_width 0.01: 2^-30 * 0.01 = 9.31e-12 m exactly. largest_coordinate 8,
//   smallest_cell_width 2^-7: 2^-37 + 2^-37 = 2^-36 m. largest_coordinate 10, smallest_cell_width 0.01: 1.84e-11 m.
// Read: docs/method-paper/fluidity-method.typ <eq:contact-tolerance>
FLUIDITY_CPU_AND_GPU constexpr double ComputeContactTolerance(ContactScale scale) {
  // Method paper <eq:contact-tolerance>:  tol = 2^-30 w_min + 2^-40 L
  const double cell_width_share = kContactToleranceFloorInCellWidths * scale.smallest_cell_width;
  const double largest_coordinate_share = kContactToleranceRelative * scale.largest_coordinate;
  return cell_width_share + largest_coordinate_share;
}

// Checks that the thinnest solid and the smallest cell are each at least kMinFeatureInTolerances contact
// tolerances wide, so the tolerance can protect them.
//
// Method: kNone when kMinFeatureInTolerances * contact_tolerance <= thinnest_solid and <= smallest_cell_width, both
//   limits included. The test is written as "wide enough", so a NaN anywhere gives kSolidTooThin. The tolerance is
//   never made larger to pass: the check fails instead.
// Units: m for contact_tolerance.
// Example: contact_tolerance 2^-30 m, smallest_cell_width 1 m. thinnest_solid 1024 * 2^-30 m: kNone; 1023 * 2^-30 m:
//   kSolidTooThin. thinnest_solid 1 m and smallest_cell_width 1023 * 2^-30 m: kSolidTooThin.
// Fails: kSolidTooThin  the thinnest solid or the smallest cell is narrower than 1024 contact tolerances: the solid
//                       is too thin, or the cells too small, for its distance from the origin. A thicker solid,
//                       larger cells or a scene closer to the origin fixes it.
// Read: docs/method-paper/fluidity-method.typ <sec-contact-tolerance>
FLUIDITY_CPU_AND_GPU constexpr ContactError CheckContactTolerance(ContactScale scale, double contact_tolerance) {
  const double narrowest_allowed = kMinFeatureInTolerances * contact_tolerance;  // m
  const bool solid_is_thick_enough = narrowest_allowed <= scale.thinnest_solid;
  const bool cell_is_wide_enough = narrowest_allowed <= scale.smallest_cell_width;
  if (solid_is_thick_enough && cell_is_wide_enough) {
    return ContactError::kNone;
  }
  return ContactError::kSolidTooThin;
}

}  // namespace fluidity::contact
