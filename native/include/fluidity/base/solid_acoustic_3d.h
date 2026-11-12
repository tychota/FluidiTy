#pragma once

#include "fluidity/base/m5_constitutive_3d.h"

namespace fluidity {
namespace solid_detail {

FLUIDITY_HD inline bool
PublishAcousticBound(double maximum_row_sum, double reference_density, double& speed_squared) noexcept {
  if (!PositiveFinite(reference_density)) {
    return false;
  }
  const double candidate = maximum_row_sum / reference_density;
  if (!PositiveFinite(candidate)) {
    return false;
  }
  speed_squared = candidate;
  return true;
}

// Minimize the acoustic 2x2 minor over directions in the two principal axes.
FLUIDITY_HD inline bool PositivePrincipalPlane(
    double first_normal,
    double second_normal,
    double first_shear,
    double second_shear,
    double coupling
) noexcept {
  const double normal_geometric_mean = ::sqrt(first_normal) * ::sqrt(second_normal);
  const double shear_geometric_mean = ::sqrt(first_shear) * ::sqrt(second_shear);
  return ::fabs(coupling) - shear_geometric_mean < normal_geometric_mean;
}

}  // namespace solid_detail

// Bounds squared elastic wave speed in m²/s² for the Hencky law at positive elastic stretches.
// Uses the scaled Hessian in §8.2 and reference density in kg/m³. This is a continuous bound.
// Rejects invalid or unrepresentable tangents and loss of ellipticity in a principal plane. Failure preserves output.
FLUIDITY_HD inline bool GranularAcousticBoundSquared3D(
    Vec3 elastic_stretches,
    GranularParameters3D parameters,
    double reference_density,
    double& speed_squared
) noexcept {
  namespace detail = solid_detail;
  if (!detail::PositiveStretches(elastic_stretches) || !detail::PositiveFinite(parameters.shear_modulus)
      || !detail::NonnegativeFinite(parameters.lame_lambda)) {
    return false;
  }
  const double twice_shear_modulus = 2 * parameters.shear_modulus;
  const Vec3 log_stretches = detail::LogStretches(elastic_stretches);
  const double volumetric_stress = parameters.lame_lambda * detail::Trace(log_stretches);
  const double logs[]{log_stretches.x, log_stretches.y, log_stretches.z};
  double stresses[3]{};
  double normal_diagonals[3]{};
  double maximum_row_sum = 0;
  for (int axis = 0; axis < 3; ++axis) {
    stresses[axis] = twice_shear_modulus * logs[axis] + volumetric_stress;
    const double normal_diagonal = twice_shear_modulus + parameters.lame_lambda - stresses[axis];
    normal_diagonals[axis] = normal_diagonal;
    const double row_sum = normal_diagonal + 2 * parameters.lame_lambda;
    if (!detail::PositiveFinite(normal_diagonal) || !detail::PositiveFinite(row_sum)) {
      return false;
    }
    maximum_row_sum = ::fmax(maximum_row_sum, row_sum);
  }
  for (int first = 0; first < 2; ++first) {
    for (int second = first + 1; second < 3; ++second) {
      const int low = logs[first] <= logs[second] ? first : second;
      const int high = first + second - low;
      const double log_ratio = logs[high] - logs[low];
      // Weight the divided difference before dividing to avoid squaring the stretches.
      const double high_diagonal =
          log_ratio == 0 ? parameters.shear_modulus : twice_shear_modulus * (log_ratio / -::expm1(-2 * log_ratio));
      const double low_diagonal = high_diagonal * ::exp(-2 * log_ratio);
      const double off_diagonal = low_diagonal - stresses[low];
      const double row_sum = high_diagonal + ::fabs(off_diagonal);
      if (!detail::PositiveFinite(low_diagonal) || !detail::PositiveFinite(row_sum)
          || !detail::PositivePrincipalPlane(
              normal_diagonals[first],
              normal_diagonals[second],
              low_diagonal,
              high_diagonal,
              parameters.lame_lambda + off_diagonal
          )) {
        return false;
      }
      maximum_row_sum = ::fmax(maximum_row_sum, row_sum);
    }
  }
  return detail::PublishAcousticBound(maximum_row_sum, reference_density, speed_squared);
}

// Bounds squared elastic wave speed in m²/s² for snow, freezing the moduli at the current plastic state.
// Uses positive elastic stretches and reference density in kg/m³. This is a continuous bound.
// Rejects invalid or unrepresentable tangents and loss of ellipticity in a principal plane. Failure preserves output.
FLUIDITY_HD inline bool SnowAcousticBoundSquared3D(
    Vec3 elastic_stretches,
    double log_plastic_jacobian,
    SnowParameters3D parameters,
    double reference_density,
    double& speed_squared
) noexcept {
  namespace detail = solid_detail;
  if (!detail::PositiveStretches(elastic_stretches) || !detail::PositiveFinite(::exp(log_plastic_jacobian))
      || !detail::PositiveFinite(parameters.shear_modulus) || !detail::NonnegativeFinite(parameters.lame_lambda)
      || !detail::NonnegativeFinite(parameters.hardening)) {
    return false;
  }
  const double multiplier = ::exp(-parameters.hardening * ::expm1(log_plastic_jacobian));
  const double twice_shear_modulus = multiplier * (2 * parameters.shear_modulus);
  const double lame_lambda = multiplier * parameters.lame_lambda;
  const double jacobian = ::exp(detail::Trace(detail::LogStretches(elastic_stretches)));
  if (!detail::PositiveFinite(jacobian) || !detail::PositiveFinite(twice_shear_modulus)
      || !detail::NonnegativeFinite(lame_lambda)) {
    return false;
  }
  const double normal_off_diagonal = lame_lambda * (2 * jacobian - 1) * jacobian;
  const double bulk_diagonal = lame_lambda * jacobian * jacobian;
  const double stretches[]{elastic_stretches.x, elastic_stretches.y, elastic_stretches.z};
  double maximum_row_sum = 0;
  double normal_diagonals[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    const double stretch = stretches[axis];
    const double normal_diagonal = (twice_shear_modulus * stretch) * stretch + bulk_diagonal;
    normal_diagonals[axis] = normal_diagonal;
    const double row_sum = normal_diagonal + 2 * ::fabs(normal_off_diagonal);
    if (!detail::PositiveFinite(normal_diagonal) || !detail::PositiveFinite(row_sum)) {
      return false;
    }
    maximum_row_sum = ::fmax(maximum_row_sum, row_sum);
  }
  for (int first = 0; first < 2; ++first) {
    for (int second = first + 1; second < 3; ++second) {
      const double larger = ::fmax(stretches[first], stretches[second]);
      const double smaller = ::fmin(stretches[first], stretches[second]);
      const double inverse_sum = 1 / (larger + smaller);
      const double shear_diagonal_factor = twice_shear_modulus * (1 - inverse_sum);
      const double off_diagonal =
          twice_shear_modulus * (larger * inverse_sum) * smaller - lame_lambda * (jacobian - 1) * jacobian;
      const double high_diagonal = (shear_diagonal_factor * larger) * larger;
      const double low_diagonal = (shear_diagonal_factor * smaller) * smaller;
      const double row_sum = high_diagonal + ::fabs(off_diagonal);
      if (!detail::PositiveFinite(low_diagonal) || !detail::PositiveFinite(row_sum)
          || !detail::PositivePrincipalPlane(
              normal_diagonals[first],
              normal_diagonals[second],
              low_diagonal,
              high_diagonal,
              normal_off_diagonal + off_diagonal
          )) {
        return false;
      }
      maximum_row_sum = ::fmax(maximum_row_sum, row_sum);
    }
  }
  return detail::PublishAcousticBound(maximum_row_sum, reference_density, speed_squared);
}

}  // namespace fluidity
