// Test-only: the helpers that the unit tests and the functional tests of the nearby face weights share.
//
// Purpose: let the unit tests and the functional tests of nearby_face_weights.h use the same checks, written once.
//
// Idea: the weights of every position obey the same few laws, so each law is one check here: the weights of the kept
//   faces are never negative and add up to 1, their changes add up to 0, and in the complete case the sums, the mean
//   offset and the weighted spread are the closed form of the weight curve.
//
// Owns:    FindWeightsOfAllFaces, ComputeKeptWeightTotal, ComputeKeptWeightChangeTotal, CheckWeightsAddUp,
//          CheckCompleteClosedForm, ComputeKeptRawChangeMagnitude, CheckIsCompleteWithUnitWeightSum,
//          CheckKeptWeightsAreNotNegative, CheckVectorIsNear, CheckMatrixIsNear, ComputeZeroWeightCount,
//          ComputePositiveWeightCount, CheckHasNoWeight, kWeightSumTolerance, kWeightChangeSumTolerance, LinearField,
//          ValueCentring, ComputeLinearValue, ComputeFittedMatrixRow, CheckHasSameBits, CheckFaceWeightHasSameBits
// Uses:    the headers it includes
// Used by: src/fluidity/grid/nearby_face_weights.test.cc, src/fluidity/grid/nearby_face_weights.face_seen.test.cc,
// tests/grid/nearby_face_weights_on_many_positions.test.cc and
//          tests/grid/nearby_face_weights_on_random_grids.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "fluidity/base/axis.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/weight_curve.h"

namespace fluidity::grid {

// How far the kept weights of one axis may be from adding up to 1, and the scale of the sum of their changes, which
// is this number divided by the cell width in m.
inline constexpr double kWeightSumTolerance = 1e-14;
inline constexpr double kWeightChangeSumTolerance = 1e-13;

// Finds the nearby faces of one axis and their weights with every face kept. A failure fails the calling test.
inline NearbyFaceWeights FindWeightsOfAllFaces(const GridTables& tables, Vec3 position, Axis axis) {
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, position, axis, CellBlockIndex{kNoIndex}, &error);
  EXPECT_EQ(error, GridError::kNone);
  return ComputeNearbyFaceWeights(tables, list, position, ToMaskOfAll(list));
}

// Adds the weights of the kept faces, as ComputeFaceWeight gives them, in list order.
inline double ComputeKeptWeightTotal(const GridTables& tables, const NearbyFaceWeights& weights) {
  double total = 0.0;
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeIsKept(weights.kept, position_in_list)) {
      total += ComputeFaceWeight(tables, weights, position_in_list).weight;
    }
  }
  return total;
}

// Adds the weight changes of the kept faces, as ComputeFaceWeight gives them, in list order.
inline Vec3 ComputeKeptWeightChangeTotal(const GridTables& tables, const NearbyFaceWeights& weights) {
  Vec3 total = {.x = 0.0, .y = 0.0, .z = 0.0};
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeIsKept(weights.kept, position_in_list)) {
      total = total + ComputeFaceWeight(tables, weights, position_in_list).weight_change;
    }
  }
  return total;
}

// The linear field of the method paper, v_f = base_value + slope . (x_f - x_p): `base_value` is the value at the
// position, in the unit of the field, and `slope` its slope, in that unit per m.
struct LinearField {
  double base_value;
  Vec3 slope;
};

// Whether the values of a fit have their weighted mean taken off before the moment is formed.
enum class ValueCentring : std::uint8_t {
  kUncentred,
  kCentred,
};

// The value of a linear field on a face whose offset from the position is `offset`, in m.
inline double ComputeLinearValue(LinearField field, Vec3 offset) {
  return field.base_value + math::Dot(field.slope, offset);
}

// Fits one matrix row to the values of a linear field on the kept faces, as the method paper's transfer does: the
// inverse weighted spread times the sum of weight x value x offset from the mean. With kCentred the weighted mean of
// the values (the weights added in list order) is taken off each value first, which changes nothing in exact
// arithmetic.
// Requires: weights.fit is kFull.
inline Vec3 ComputeFittedMatrixRow(
    const GridTables& tables,
    const NearbyFaceWeights& weights,
    LinearField field,
    ValueCentring centring
) {
  double weighted_value_sum = 0.0;
  if (centring == ValueCentring::kCentred) {
    for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
      if (ComputeIsKept(weights.kept, position_in_list)) {
        const FaceWeight face_weight = ComputeFaceWeight(tables, weights, position_in_list);
        weighted_value_sum += face_weight.weight * ComputeLinearValue(field, face_weight.offset);
      }
    }
  }
  Vec3 moment = {.x = 0.0, .y = 0.0, .z = 0.0};
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeIsKept(weights.kept, position_in_list)) {
      const FaceWeight face_weight = ComputeFaceWeight(tables, weights, position_in_list);
      const double value = ComputeLinearValue(field, face_weight.offset) - weighted_value_sum;
      moment = moment + (face_weight.offset_from_mean * (face_weight.weight * value));
    }
  }
  return weights.inverse_weighted_spread * moment;
}

// Checks that two doubles have the same bits.
inline void CheckHasSameBits(double actual, double expected) {
  EXPECT_EQ(std::bit_cast<std::uint64_t>(actual), std::bit_cast<std::uint64_t>(expected));
}

// Checks that two weights of a face have the same bits in `weight`, `offset`, `offset_from_mean` and `weight_change`,
// and name the same face.
inline void CheckFaceWeightHasSameBits(const FaceWeight& actual, const FaceWeight& expected) {
  EXPECT_EQ(actual.face.value, expected.face.value);
  CheckHasSameBits(actual.weight, expected.weight);
  CheckHasSameBits(actual.offset.x, expected.offset.x);
  CheckHasSameBits(actual.offset.y, expected.offset.y);
  CheckHasSameBits(actual.offset.z, expected.offset.z);
  CheckHasSameBits(actual.offset_from_mean.x, expected.offset_from_mean.x);
  CheckHasSameBits(actual.offset_from_mean.y, expected.offset_from_mean.y);
  CheckHasSameBits(actual.offset_from_mean.z, expected.offset_from_mean.z);
  CheckHasSameBits(actual.weight_change.x, expected.weight_change.x);
  CheckHasSameBits(actual.weight_change.y, expected.weight_change.y);
  CheckHasSameBits(actual.weight_change.z, expected.weight_change.z);
}

// Checks that three numbers are within `tolerance` of the three of `expected`; a tolerance of 0 asks for equality.
inline void CheckVectorIsNear(Vec3 actual, Vec3 expected, double tolerance) {
  EXPECT_NEAR(actual.x, expected.x, tolerance);
  EXPECT_NEAR(actual.y, expected.y, tolerance);
  EXPECT_NEAR(actual.z, expected.z, tolerance);
}

// Checks that nine numbers are within `tolerance` of the nine of `expected`, matrix row by matrix row; a tolerance of 0
// asks for equality.
inline void CheckMatrixIsNear(Mat3 actual, Mat3 expected, double tolerance) {
  CheckVectorIsNear(
      {.x = actual.xx, .y = actual.xy, .z = actual.xz},
      {.x = expected.xx, .y = expected.xy, .z = expected.xz},
      tolerance
  );
  CheckVectorIsNear(
      {.x = actual.yx, .y = actual.yy, .z = actual.yz},
      {.x = expected.yx, .y = expected.yy, .z = expected.yz},
      tolerance
  );
  CheckVectorIsNear(
      {.x = actual.zx, .y = actual.zy, .z = actual.zz},
      {.x = expected.zx, .y = expected.zy, .z = expected.zz},
      tolerance
  );
}

// Checks that no kept weight is negative.
inline void CheckKeptWeightsAreNotNegative(const GridTables& tables, const NearbyFaceWeights& weights) {
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeIsKept(weights.kept, position_in_list)) {
      EXPECT_GE(ComputeFaceWeight(tables, weights, position_in_list).weight, 0.0) << position_in_list;
    }
  }
}

// Adds the absolute values of the kept raw weight changes on each axis and divides by W: the size of the terms whose
// rounding the scaled changes carry.
inline Vec3 ComputeKeptRawChangeMagnitude(const GridTables& tables, const NearbyFaceWeights& weights) {
  Vec3 total = {.x = 0.0, .y = 0.0, .z = 0.0};
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeIsKept(weights.kept, position_in_list)) {
      const RawFaceWeight raw =
          ComputeRawFaceWeight(tables, weights.list.faces[position_in_list], weights.position, weights.list.cell_width);
      total.x += std::abs(raw.weight_change.x);
      total.y += std::abs(raw.weight_change.y);
      total.z += std::abs(raw.weight_change.z);
    }
  }
  return total * (1.0 / weights.weight_sum);
}

// Checks that no kept weight is negative, that the kept weights add up to 1 within kWeightSumTolerance and that their
// changes add up to 0 within kWeightChangeSumTolerance / cell width on each axis, plus 8 rounding steps of the sum of
// the absolute raw changes divided by W. The second term matters when W is small: each change is (g - weight G) / W,
// so its rounding error grows as 1 / W even where the change itself is small. The factor 8 is measured, not derived:
// the main term is the rounding of the sum G itself, which a derivation would bound by the count of kept faces times
// a rounding step, and the random grids stay well below that.
inline void CheckWeightsAddUp(const GridTables& tables, const NearbyFaceWeights& weights) {
  CheckKeptWeightsAreNotNegative(tables, weights);
  EXPECT_NEAR(ComputeKeptWeightTotal(tables, weights), 1.0, kWeightSumTolerance);
  const double base_tolerance = kWeightChangeSumTolerance / weights.list.cell_width;
  const double rounding_step = 8.0 * std::numeric_limits<double>::epsilon();
  const Vec3 magnitude = ComputeKeptRawChangeMagnitude(tables, weights);
  const Vec3 change_total = ComputeKeptWeightChangeTotal(tables, weights);
  EXPECT_NEAR(change_total.x, 0.0, base_tolerance + (rounding_step * magnitude.x));
  EXPECT_NEAR(change_total.y, 0.0, base_tolerance + (rounding_step * magnitude.y));
  EXPECT_NEAR(change_total.z, 0.0, base_tolerance + (rounding_step * magnitude.z));
}

// Checks that the faces of a list were all kept, were 27 and all of the cell's size, and that the weights add up to 1:
// is_complete, a fit of kFull and a weight sum within `weight_sum_tolerance` of 1.
inline void CheckIsCompleteWithUnitWeightSum(const NearbyFaceWeights& weights, double weight_sum_tolerance) {
  EXPECT_EQ(weights.list.count, 27);
  EXPECT_TRUE(weights.is_complete);
  EXPECT_EQ(weights.fit, NearbyFaceFit::kFull);
  EXPECT_NEAR(weights.weight_sum, 1.0, weight_sum_tolerance);
}

// Checks the closed form of the complete case within 1e-14: W = 1, the mean offset 0 within 1e-14 h, and the weighted
// spread h^2 / 4 times the identity within 1e-14 h^2, and its inverse 4 / h^2 times the identity within a relative
// 1e-12 (the factor 1 / h^2 of the fit); also is_complete, a fit of kFull and 27 faces.
inline void CheckCompleteClosedForm(const NearbyFaceWeights& weights) {
  const double cell_width = weights.list.cell_width;
  CheckIsCompleteWithUnitWeightSum(weights, kWeightSumTolerance);
  CheckVectorIsNear(weights.mean_offset, {.x = 0.0, .y = 0.0, .z = 0.0}, 1e-14 * cell_width);
  const double spread_diagonal = ComputeCompleteWeightedSpread(cell_width);
  const Mat3 expected_spread = {
      .xx = spread_diagonal,
      .xy = 0.0,
      .xz = 0.0,
      .yx = 0.0,
      .yy = spread_diagonal,
      .yz = 0.0,
      .zx = 0.0,
      .zy = 0.0,
      .zz = spread_diagonal,
  };
  CheckMatrixIsNear(weights.weighted_spread, expected_spread, 1e-14 * cell_width * cell_width);
  const double inverse_diagonal = 1.0 / spread_diagonal;
  const Mat3 expected_inverse = {
      .xx = inverse_diagonal,
      .xy = 0.0,
      .xz = 0.0,
      .yx = 0.0,
      .yy = inverse_diagonal,
      .yz = 0.0,
      .zx = 0.0,
      .zy = 0.0,
      .zz = inverse_diagonal,
  };
  CheckMatrixIsNear(weights.inverse_weighted_spread, expected_inverse, 1e-12 * inverse_diagonal);
}

// Counts the kept faces whose weight is exactly 0.
inline int ComputeZeroWeightCount(const GridTables& tables, const NearbyFaceWeights& weights) {
  int count = 0;
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeIsKept(weights.kept, position_in_list)
        && ComputeFaceWeight(tables, weights, position_in_list).weight == 0.0) {
      ++count;
    }
  }
  return count;
}

// Counts the kept faces whose weight is above 0.
inline int ComputePositiveWeightCount(const GridTables& tables, const NearbyFaceWeights& weights) {
  int count = 0;
  for (int position_in_list = 0; position_in_list < weights.list.count; ++position_in_list) {
    if (ComputeIsKept(weights.kept, position_in_list)
        && ComputeFaceWeight(tables, weights, position_in_list).weight > 0.0) {
      ++count;
    }
  }
  return count;
}

// Checks that weights without any weight are as the fit kNone leaves them: every sum, the mean offset and both matrices
// exactly 0, is_complete false, and no number that is not a number.
inline void CheckHasNoWeight(const NearbyFaceWeights& weights) {
  const Vec3 zero_vector = {.x = 0.0, .y = 0.0, .z = 0.0};
  EXPECT_EQ(weights.fit, NearbyFaceFit::kNone);
  EXPECT_FALSE(weights.is_complete);
  EXPECT_EQ(weights.weight_sum, 0.0);
  CheckVectorIsNear(weights.mean_offset, zero_vector, 0.0);
  CheckVectorIsNear(weights.raw_weight_change_sum, zero_vector, 0.0);
  CheckMatrixIsNear(weights.weighted_spread, Mat3{}, 0.0);
  CheckMatrixIsNear(weights.inverse_weighted_spread, Mat3{}, 0.0);
}

}  // namespace fluidity::grid
