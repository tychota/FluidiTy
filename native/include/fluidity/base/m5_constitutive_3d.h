#pragma once

#if defined(__CUDACC__)
#include <cuda/std/cmath>
#else
#include <cmath>
#endif

#include "fluidity/base/host_device.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

struct GranularParameters3D {
  double shear_modulus;  // Pa, strictly positive.
  double lame_lambda;  // Pa, nonnegative.
  double friction_angle;  // Radians in [0, pi/2).
  double cohesion_log_strain;  // Nonnegative translation of the strain cone along its trace.
};

struct SnowParameters3D {
  double shear_modulus;  // Unhardened modulus in Pa, strictly positive.
  double lame_lambda;  // Unhardened modulus in Pa, nonnegative.
  double compression_limit;  // Dimensionless elastic shortening in [0, 1).
  double stretch_limit;  // Nonnegative dimensionless elastic elongation.
  double hardening;  // Nonnegative exponent coefficient.
};

struct PrincipalSolidState3D {
  Vec3 elastic_stretches;  // Returned stretches in input principal-axis order.
  double log_plastic_jacobian;
  Vec3 kirchhoff_stress;  // Pa in input principal-axis order, for reference-volume deposition.
};

namespace solid_detail {

#if defined(__CUDACC__)
namespace numeric = cuda::std;
#else
namespace numeric = std;
#endif

FLUIDITY_HD inline bool Finite(Vec3 value) noexcept {
  return numeric::isfinite(value.x) && numeric::isfinite(value.y) && numeric::isfinite(value.z);
}

FLUIDITY_HD inline bool PositiveFinite(double value) noexcept {
  return value > 0 && numeric::isfinite(value);
}

FLUIDITY_HD inline bool NonnegativeFinite(double value) noexcept {
  return value >= 0 && numeric::isfinite(value);
}

FLUIDITY_HD inline bool PositiveStretches(Vec3 stretches) noexcept {
  return PositiveFinite(stretches.x) && PositiveFinite(stretches.y) && PositiveFinite(stretches.z);
}

FLUIDITY_HD inline Vec3 LogStretches(Vec3 stretches) noexcept {
  return {::log(stretches.x), ::log(stretches.y), ::log(stretches.z)};
}

FLUIDITY_HD inline double Trace(Vec3 principal_values) noexcept {
  return principal_values.x + principal_values.y + principal_values.z;
}

FLUIDITY_HD inline bool PublishPrincipalState(
    PrincipalSolidState3D candidate,
    double log_total_jacobian,
    PrincipalSolidState3D& output
) noexcept {
  const double log_elastic_jacobian = Trace(LogStretches(candidate.elastic_stretches));
  if (PositiveStretches(candidate.elastic_stretches) && PositiveFinite(::exp(log_elastic_jacobian))
      && PositiveFinite(::exp(candidate.log_plastic_jacobian)) && PositiveFinite(::exp(log_total_jacobian))
      && Finite(candidate.kirchhoff_stress)) {
    output = candidate;
    return true;
  }
  return false;
}

}  // namespace solid_detail

// Applies the Hencky/Drucker–Prager return in the trial principal frame.
// Stretches and plastic, elastic, and total Jacobians must be positive and finite.
// Returns false on invalid input or unrepresentable state, leaving output unchanged.
FLUIDITY_HD inline bool ReturnGranularPrincipal3D(
    Vec3 trial_stretches,
    double previous_log_plastic_jacobian,
    GranularParameters3D parameters,
    PrincipalSolidState3D& output
) noexcept {
  if (!solid_detail::PositiveStretches(trial_stretches)
      || !solid_detail::PositiveFinite(::exp(previous_log_plastic_jacobian))
      || !solid_detail::PositiveFinite(parameters.shear_modulus)
      || !solid_detail::NonnegativeFinite(parameters.lame_lambda)) {
    return false;
  }
  constexpr double kHalfPi = 1.5707963267948966;
  const bool admissible_friction = parameters.friction_angle >= 0 && parameters.friction_angle < kHalfPi;
  if (!admissible_friction || !solid_detail::NonnegativeFinite(parameters.cohesion_log_strain)) {
    return false;
  }

  Vec3 returned_log_strain = solid_detail::LogStretches(trial_stretches);
  const double trial_log_jacobian = solid_detail::Trace(returned_log_strain);
  const double cone_trace = trial_log_jacobian - parameters.cohesion_log_strain;
  double log_plastic_increment = 0;
  Vec3 returned_stretches = trial_stretches;
  if (cone_trace >= 0) {
    const double apex_strain = parameters.cohesion_log_strain / 3;
    returned_log_strain = {apex_strain, apex_strain, apex_strain};
    const double apex_stretch = ::exp(apex_strain);
    returned_stretches = {apex_stretch, apex_stretch, apex_stretch};
    log_plastic_increment = cone_trace;
  } else {
    const double mean_strain = trial_log_jacobian / 3;
    const Vec3 deviator{
        returned_log_strain.x - mean_strain,
        returned_log_strain.y - mean_strain,
        returned_log_strain.z - mean_strain,
    };
    const double deviator_norm = ::sqrt(deviator.x * deviator.x + deviator.y * deviator.y + deviator.z * deviator.z);
    const double sine_friction = ::sin(parameters.friction_angle);
    const double friction_slope = ::sqrt(2.0 / 3) * (2 * sine_friction / (3 - sine_friction));
    const double cone_radius = friction_slope == 0
        ? 0
        : (-cone_trace * friction_slope) * (1 + 1.5 * (parameters.lame_lambda / parameters.shear_modulus));
    if (solid_detail::numeric::isnan(cone_radius)) {
      return false;
    }
    // Shear flow changes the deviator at fixed trace, so it leaves plastic volume unchanged.
    if (deviator_norm > cone_radius) {
      const double deviator_scale = cone_radius / deviator_norm;
      returned_log_strain = {
          mean_strain + deviator_scale * deviator.x,
          mean_strain + deviator_scale * deviator.y,
          mean_strain + deviator_scale * deviator.z,
      };
      returned_stretches = {::exp(returned_log_strain.x), ::exp(returned_log_strain.y), ::exp(returned_log_strain.z)};
    }
  }

  const double returned_log_jacobian = trial_log_jacobian - log_plastic_increment;
  const double volumetric_stress = parameters.lame_lambda * returned_log_jacobian;
  const PrincipalSolidState3D candidate{
      .elastic_stretches = returned_stretches,
      .log_plastic_jacobian = previous_log_plastic_jacobian + log_plastic_increment,
      .kirchhoff_stress =
          {
              parameters.shear_modulus * (2 * returned_log_strain.x) + volumetric_stress,
              parameters.shear_modulus * (2 * returned_log_strain.y) + volumetric_stress,
              parameters.shear_modulus * (2 * returned_log_strain.z) + volumetric_stress,
          },
  };
  return solid_detail::PublishPrincipalState(candidate, previous_log_plastic_jacobian + trial_log_jacobian, output);
}

// Clamps trial snow stretches, transfers volume to the plastic Jacobian and evaluates hardened corotational stress.
// Stretches and plastic, elastic, and total Jacobians must be positive and finite.
// Returns false on invalid input or unrepresentable state, leaving output unchanged.
FLUIDITY_HD inline bool ReturnSnowPrincipal3D(
    Vec3 trial_stretches,
    double previous_log_plastic_jacobian,
    SnowParameters3D parameters,
    PrincipalSolidState3D& output
) noexcept {
  if (!solid_detail::PositiveStretches(trial_stretches)
      || !solid_detail::PositiveFinite(::exp(previous_log_plastic_jacobian))
      || !solid_detail::PositiveFinite(parameters.shear_modulus)
      || !solid_detail::NonnegativeFinite(parameters.lame_lambda)) {
    return false;
  }
  const bool admissible_compression = parameters.compression_limit >= 0 && parameters.compression_limit < 1;
  if (!admissible_compression || !solid_detail::NonnegativeFinite(parameters.stretch_limit)
      || !solid_detail::NonnegativeFinite(parameters.hardening)) {
    return false;
  }

  const double minimum_stretch = 1 - parameters.compression_limit;
  const double maximum_stretch = 1 + parameters.stretch_limit;
  const Vec3 returned_stretches{
      ::fmin(maximum_stretch, ::fmax(minimum_stretch, trial_stretches.x)),
      ::fmin(maximum_stretch, ::fmax(minimum_stretch, trial_stretches.y)),
      ::fmin(maximum_stretch, ::fmax(minimum_stretch, trial_stretches.z)),
  };
  const double trial_log_jacobian = solid_detail::Trace(solid_detail::LogStretches(trial_stretches));
  const double returned_log_jacobian = solid_detail::Trace(solid_detail::LogStretches(returned_stretches));
  const double log_plastic_increment = trial_log_jacobian - returned_log_jacobian;
  const double log_plastic_jacobian = previous_log_plastic_jacobian + log_plastic_increment;
  const double hardening_multiplier = ::exp(-parameters.hardening * ::expm1(log_plastic_jacobian));
  const double hardened_shear_modulus = hardening_multiplier * parameters.shear_modulus;
  const double hardened_lame_lambda = hardening_multiplier * parameters.lame_lambda;
  if (!solid_detail::PositiveFinite(hardened_shear_modulus) || !solid_detail::NonnegativeFinite(hardened_lame_lambda)) {
    return false;
  }

  const double elastic_jacobian = ::exp(returned_log_jacobian);
  const double volumetric_stress = hardened_lame_lambda * ::expm1(returned_log_jacobian) * elastic_jacobian;
  const PrincipalSolidState3D candidate{
      .elastic_stretches = returned_stretches,
      .log_plastic_jacobian = log_plastic_jacobian,
      .kirchhoff_stress =
          {
              hardened_shear_modulus * (2 * (returned_stretches.x - 1)) * returned_stretches.x + volumetric_stress,
              hardened_shear_modulus * (2 * (returned_stretches.y - 1)) * returned_stretches.y + volumetric_stress,
              hardened_shear_modulus * (2 * (returned_stretches.z - 1)) * returned_stretches.z + volumetric_stress,
          },
  };
  return solid_detail::PublishPrincipalState(candidate, previous_log_plastic_jacobian + trial_log_jacobian, output);
}

}  // namespace fluidity
