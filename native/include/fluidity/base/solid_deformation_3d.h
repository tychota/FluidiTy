#pragma once

#include "fluidity/base/m5_constitutive_3d.h"

namespace fluidity {

struct SolidDeformationState3D {
  Mat3 elastic_deformation;  // Returned elastic map from reference to spatial coordinates.
  double log_plastic_jacobian;
  Vec3 elastic_stretches;  // Returned positive stretches in the decomposition order.
  Mat3 kirchhoff_stress;  // World-frame symmetric stress in Pa, for reference-volume deposition.
};

namespace solid_detail {

FLUIDITY_HD inline double Dot(Vec3 lhs, Vec3 rhs) noexcept {
  return (lhs.x * rhs.x + lhs.y * rhs.y) + lhs.z * rhs.z;
}

FLUIDITY_HD inline Vec3 Combine(Vec3 first, double first_scale, Vec3 second, double second_scale) noexcept {
  return {
      first_scale * first.x + second_scale * second.x,
      first_scale * first.y + second_scale * second.y,
      first_scale * first.z + second_scale * second.z
  };
}

FLUIDITY_HD inline void RotateColumns(Vec3& first, Vec3& second, double cosine, double sine) noexcept {
  const Vec3 rotated_first = Combine(first, cosine, second, -sine);
  second = Combine(first, sine, second, cosine);
  first = rotated_first;
}

struct PrincipalFrame3D {
  Vec3 left[3];
  Vec3 right[3];
  Vec3 stretches;
};

// One-sided Jacobi works on F directly and avoids squaring its condition number in a Gram matrix.
FLUIDITY_HD inline bool FindPrincipalFrame(Mat3 deformation, PrincipalFrame3D& frame) noexcept {
  if (!Finite(deformation.x_row) || !Finite(deformation.y_row) || !Finite(deformation.z_row)) {
    return false;
  }

  Vec3 columns[]{
      {deformation.x_row.x, deformation.y_row.x, deformation.z_row.x},
      {deformation.x_row.y, deformation.y_row.y, deformation.z_row.y},
      {deformation.x_row.z, deformation.y_row.z, deformation.z_row.z},
  };
  double scale = 0;
  for (const Vec3 column : columns) {
    scale = ::fmax(scale, ::fmax(::fabs(column.x), ::fmax(::fabs(column.y), ::fabs(column.z))));
  }
  if (scale == 0) {
    return false;
  }
  for (Vec3& column : columns) {
    column = {column.x / scale, column.y / scale, column.z / scale};
  }

  // Certify orientation before rotating. Jacobi roundoff can otherwise create a
  // small positive stretch from an exactly rank-deficient matrix.
  const Vec3 first_column = columns[0];
  const Vec3 second_column = columns[1];
  const Vec3 third_column = columns[2];
  const double positive_terms[]{
      first_column.x * second_column.y * third_column.z,
      first_column.y * second_column.z * third_column.x,
      first_column.z * second_column.x * third_column.y,
  };
  const double negative_terms[]{
      first_column.x * second_column.z * third_column.y,
      first_column.y * second_column.x * third_column.z,
      first_column.z * second_column.y * third_column.x,
  };
  double scaled_determinant = 0;
  double determinant_term_sum = 0;
  for (int index = 0; index < 3; ++index) {
    scaled_determinant += positive_terms[index] - negative_terms[index];
    determinant_term_sum += ::fabs(positive_terms[index]) + ::fabs(negative_terms[index]);
  }
  constexpr double kDeterminantRelativeError = 1.5e-14;
  if (scaled_determinant <= kDeterminantRelativeError * determinant_term_sum) {
    return false;
  }

  frame.right[0] = {1, 0, 0};
  frame.right[1] = {0, 1, 0};
  frame.right[2] = {0, 0, 1};
  constexpr double kRotationTolerance = 2e-15;
  for (int sweep = 0; sweep < 16; ++sweep) {
    bool rotated = false;
    for (int first = 0; first < 2; ++first) {
      for (int second = first + 1; second < 3; ++second) {
        const double first_squared = Dot(columns[first], columns[first]);
        const double second_squared = Dot(columns[second], columns[second]);
        const double correlation = Dot(columns[first], columns[second]);
        const double norm_product = ::sqrt(first_squared) * ::sqrt(second_squared);
        if (::fabs(correlation) <= kRotationTolerance * norm_product) {
          continue;
        }
        const double norm_difference = second_squared - first_squared;
        const double twice_correlation = 2 * correlation;
        const double denominator =
            norm_difference + ::copysign(::hypot(norm_difference, twice_correlation), norm_difference);
        const double tangent = twice_correlation / denominator;
        const double cosine = 1 / ::sqrt(1 + tangent * tangent);
        const double sine = tangent * cosine;
        RotateColumns(columns[first], columns[second], cosine, sine);
        RotateColumns(frame.right[first], frame.right[second], cosine, sine);
        rotated = true;
      }
    }
    if (!rotated) {
      break;
    }
  }

  double stretches[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    const Vec3 column = columns[axis];
    const double length = numeric::hypot(column.x, column.y, column.z);
    if (!PositiveFinite(length)) {
      return false;
    }
    frame.left[axis] = {column.x / length, column.y / length, column.z / length};
    stretches[axis] = length * scale;
  }
  frame.stretches = {stretches[0], stretches[1], stretches[2]};
  if (!PositiveStretches(frame.stretches)) {
    return false;
  }
  constexpr double kOrthogonalityTolerance = 1e-12;
  if (::fabs(Dot(frame.left[0], frame.left[1])) > kOrthogonalityTolerance
      || ::fabs(Dot(frame.left[0], frame.left[2])) > kOrthogonalityTolerance
      || ::fabs(Dot(frame.left[1], frame.left[2])) > kOrthogonalityTolerance) {
    return false;
  }
  // Right rotations preserve orientation. The normalized left frame therefore detects reflection.
  const Vec3 first = frame.left[0];
  const Vec3 second = frame.left[1];
  const Vec3 cross{
      first.y * second.z - first.z * second.y,
      first.z * second.x - first.x * second.z,
      first.x * second.y - first.y * second.x
  };
  return Dot(cross, frame.left[2]) > 0;
}

FLUIDITY_HD inline Mat3 Recompose(const Vec3 (&left)[3], Vec3 principal_values, const Vec3 (&right)[3]) noexcept {
  constexpr double Vec3::* kComponents[]{&Vec3::x, &Vec3::y, &Vec3::z};
  Vec3 rows[3]{};
  for (int row = 0; row < 3; ++row) {
    const auto component = kComponents[row];
    rows[row] =
        Combine(right[0], left[0].*component * principal_values.x, right[1], left[1].*component * principal_values.y);
    rows[row] = Combine(rows[row], 1, right[2], left[2].*component * principal_values.z);
  }
  return {rows[0], rows[1], rows[2]};
}

template <typename Parameters, typename ReturnPrincipal>
FLUIDITY_HD inline bool ReturnDeformation(
    Mat3 trial,
    double previous_log_plastic_jacobian,
    Parameters parameters,
    ReturnPrincipal return_principal,
    SolidDeformationState3D& output
) noexcept {
  PrincipalFrame3D frame{};
  PrincipalSolidState3D principal{};
  if (!FindPrincipalFrame(trial, frame)
      || !return_principal(frame.stretches, previous_log_plastic_jacobian, parameters, principal)) {
    return false;
  }
  const SolidDeformationState3D candidate{
      .elastic_deformation = Recompose(frame.left, principal.elastic_stretches, frame.right),
      .log_plastic_jacobian = principal.log_plastic_jacobian,
      .elastic_stretches = principal.elastic_stretches,
      .kirchhoff_stress = Recompose(frame.left, principal.kirchhoff_stress, frame.left),
  };
  const Mat3 deformation = candidate.elastic_deformation;
  const Mat3 stress = candidate.kirchhoff_stress;
  if (!Finite(deformation.x_row) || !Finite(deformation.y_row) || !Finite(deformation.z_row) || !Finite(stress.x_row)
      || !Finite(stress.y_row) || !Finite(stress.z_row)) {
    return false;
  }
  output = candidate;
  return true;
}

}  // namespace solid_detail

// Returns the granular elastic map and world Kirchhoff stress from a positive-orientation trial map.
// Rejects invalid, singular, unresolved or unrepresentable state and leaves output unchanged on failure.
FLUIDITY_HD inline bool ReturnGranularDeformation3D(
    Mat3 trial_elastic_deformation,
    double previous_log_plastic_jacobian,
    GranularParameters3D parameters,
    SolidDeformationState3D& output
) noexcept {
  return solid_detail::ReturnDeformation(
      trial_elastic_deformation,
      previous_log_plastic_jacobian,
      parameters,
      ReturnGranularPrincipal3D,
      output
  );
}

// Returns the clamped snow elastic map and hardened world Kirchhoff stress from a positive-orientation trial map.
// Rejects invalid, singular, unresolved or unrepresentable state and leaves output unchanged on failure.
FLUIDITY_HD inline bool ReturnSnowDeformation3D(
    Mat3 trial_elastic_deformation,
    double previous_log_plastic_jacobian,
    SnowParameters3D parameters,
    SolidDeformationState3D& output
) noexcept {
  return solid_detail::ReturnDeformation(
      trial_elastic_deformation,
      previous_log_plastic_jacobian,
      parameters,
      ReturnSnowPrincipal3D,
      output
  );
}

}  // namespace fluidity
