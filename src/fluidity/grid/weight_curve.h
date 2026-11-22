// The weight curve: how much a face gets from a particle at a given distance.
//
// Purpose: turn the distance between a particle and a grid face, measured in cell widths, into the weight the face
//   gets, and into how fast that weight changes with the distance. Code on the CPU and on the GPU both call it.
//
// Idea: a particle passes its values to the faces near it, and reads values back from them, with weights. Along one
//   axis the weight of a face is a curve of the distance t in cell widths (papers call it the quadratic B-spline):
//     Q(t)  = 3/4 - t^2             if |t| < 1/2
//           = (3/2 - |t|)^2 / 2     if 1/2 <= |t| < 3/2
//           = 0                     otherwise
//     Q'(t) = -2 t                  if |t| < 1/2
//           = -(3/2 - |t|) sign(t)  if 1/2 <= |t| < 3/2
//           = 0                     otherwise
//   The curve is never negative, and it and its slope are zero at and beyond 1.5 cell widths. For faces one width
//   apart, the weights of the three faces nearest the particle add up to 1, their slopes add up to 0, and the sum of
//   weight times squared offset from the particle is a quarter of a cell width squared. That last number is the
//   diagonal entry of the weighted spread when all 27 same-size faces are kept (ComputeCompleteWeightedSpread).
//   Example: at t = 0.25 the weight is 3/4 - 1/16 = 0.6875 and the slope is -0.5. At t = -1.25 the weight is
//   (1/4)^2 / 2 = 0.03125 and the slope is +0.25. At t = 0.5 both branches give weight 0.5 and slope -1.
//
// Words:
//   weight curve        the function Q above, from a distance in cell widths to a weight
//   weighted spread     the symmetric 3 x 3 matrix S of <eq:weighted-spread>: the sum over the kept faces of the
//                       scaled weight times the outer product of the face's offset from the weighted mean offset
//                       with itself
//   nearby face list    the faces of one axis that can have weight for a position (grid/nearby_face_weights.h)
// Owns:    WeightCurveValue, ComputeWeightCurve, kWeightCurveReach, kMaxNearbyFaces, kMinWeightSum,
//          kMinPivotShare, ComputeCompleteWeightedSpread
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), math::Abs (base/scalar_math.h)
// Used by: the nearby face weights (grid/nearby_face_weights.h) and every module that moves values between particles
//          and faces
// Read:    docs/method-paper/fluidity-method.typ <eq:weight-curve>, <sec-nearby-faces>, <eq:weighted-spread>

#pragma once

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"

namespace fluidity::grid {

// The weight curve and its slope at one distance.
struct WeightCurveValue {
  double weight;  // No unit. Never negative; 0 at and beyond kWeightCurveReach.
  double slope;  // No unit (per cell width). Derivative of `weight` with respect to the distance.
};

// The distance in cell widths at and beyond which the weight curve is zero.
inline constexpr double kWeightCurveReach = 1.5;

// Most faces one axis can list for one position: no two candidates share one of the 6 x 6 x 6 half-width places of
// the window around the position, so a valid grid stays at or below 212 (<sec-nearby-faces>). A half-width place is
// not a grid place (CellPlaceIndex).
inline constexpr int kMaxNearbyFaces = 216;

// Below this sum of kept weights (no unit) a position has too little weight to fit how the velocity varies.
inline constexpr double kMinWeightSum = 0.1;

// A pivot of the fit at or below this share (no unit) of the trace is too small to divide by. The fit factors the
// weighted spread divided by the cell width squared, and the trace is that matrix's own.
inline constexpr double kMinPivotShare = 1e-12;

// Computes the weight curve Q and its slope Q' at one distance.
//
// Method: the inner branch for |t| < 1/2, the outer branch for 1/2 <= |t| < kWeightCurveReach, zero beyond.
//   Both branches give weight 0.5 and slope -1 at t = 0.5, so the curve and its slope have no jump.
// Units: `distance_in_cell_widths` is the face position minus the particle position, divided by the cell width.
// Ties: at |t| = kWeightCurveReach the weight and the slope are exactly 0.
// Order: one fixed order of operations, written once for both processors. Every step is an IEEE 754 operation
//   (absolute value, subtraction, multiplication) that gives the same bits on both, and no step is fused with
//   another, so the CPU and the GPU give identical bits. The order is that of the formula as written above; the
//   product 0.5 r r gives the same bits grouped either way, except for a subnormal result.
// Example: ComputeWeightCurve(0.25) is weight 0.6875, slope -0.5.
// Requires: `distance_in_cell_widths` is finite.
// Read: docs/method-paper/fluidity-method.typ <eq:weight-curve>
FLUIDITY_CPU_AND_GPU constexpr WeightCurveValue ComputeWeightCurve(double distance_in_cell_widths) noexcept {
  const double absolute_distance = math::Abs(distance_in_cell_widths);
  if (absolute_distance < 0.5) {
    const double distance_squared = distance_in_cell_widths * distance_in_cell_widths;
    return {.weight = 0.75 - distance_squared, .slope = -2.0 * distance_in_cell_widths};
  }
  if (absolute_distance < kWeightCurveReach) {
    const double distance_to_outer_edge = kWeightCurveReach - absolute_distance;
    const double half_distance_to_outer_edge = 0.5 * distance_to_outer_edge;
    const double weight = half_distance_to_outer_edge * distance_to_outer_edge;
    // The slope is -(3/2 - |t|) sign(t): it points toward the particle's side of the face.
    const double slope = distance_in_cell_widths < 0.0 ? distance_to_outer_edge : -distance_to_outer_edge;
    return {.weight = weight, .slope = slope};
  }
  return {.weight = 0.0, .slope = 0.0};
}

// Computes the diagonal entry of the weighted spread S of <eq:weighted-spread> when all 27 same-size faces of one axis
// are kept: cell_width^2 / 4. In that case S is that number times the identity.
//
// Units: `cell_width` in metres; the result in square metres.
// Order: the square, then a multiplication by 0.25, which only changes the exponent, so the result is the double
//   nearest cell_width^2 / 4 on both processors.
// Example: ComputeCompleteWeightedSpread(2.0) is 1.0.
// Read: docs/method-paper/fluidity-method.typ <eq:weighted-spread>
FLUIDITY_CPU_AND_GPU constexpr double ComputeCompleteWeightedSpread(double cell_width) noexcept {
  const double cell_width_square = cell_width * cell_width;
  return 0.25 * cell_width_square;
}

}  // namespace fluidity::grid
