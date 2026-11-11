#include "fluidity/host/ocean_boundary_3d.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

namespace fluidity {
namespace {
bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

// The bracket covers both the shallow sqrt(q) and deep q asymptotes.
double SolveDepthPhase(double dispersion_rhs) {
  double lower = 0;
  double upper = dispersion_rhs + 2 * std::sqrt(dispersion_rhs);
  for (int iteration = 0; iteration < 64; ++iteration) {
    const double middle = std::midpoint(lower, upper);
    if (middle == lower || middle == upper) {
      break;
    }
    if (middle * std::tanh(middle) < dispersion_rhs) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return std::midpoint(lower, upper);
}
}  // namespace

absl::StatusOr<ResolvedOceanSwell3D> ResolvedOceanSwell3D::Resolve(const OceanSwellSettings3D& settings) {
  if (!std::isfinite(settings.still_water_height_m) || !std::isfinite(settings.water_depth_m)
      || !std::isfinite(settings.wave_height_m) || !std::isfinite(settings.wave_period_s)
      || !std::isfinite(settings.gravity_m_per_s2) || !std::isfinite(settings.phase_at_world_origin_rad)
      || !Finite(settings.propagation_direction)) {
    return absl::InvalidArgumentError("ocean swell controls must be finite");
  }
  if (settings.water_depth_m <= 0 || settings.wave_height_m < 0 || settings.wave_period_s <= 0
      || settings.gravity_m_per_s2 <= 0 || settings.propagation_direction.y != 0) {
    return absl::InvalidArgumentError("expected positive depth, period and gravity, and horizontal propagation");
  }
  const double direction_length = std::hypot(settings.propagation_direction.x, settings.propagation_direction.z);
  if (direction_length <= 0 || !std::isfinite(direction_length)) {
    return absl::InvalidArgumentError("ocean propagation direction must have finite positive length");
  }
  constexpr double kFullTurn = 2 * std::numbers::pi;
  const double frequency = kFullTurn / settings.wave_period_s;
  const double dispersion_rhs = frequency * (frequency / settings.gravity_m_per_s2) * settings.water_depth_m;
  if (!std::isnormal(dispersion_rhs)) {
    return absl::OutOfRangeError("ocean dispersion right-hand side is outside the supported numeric range");
  }
  const double depth_phase = SolveDepthPhase(dispersion_rhs);
  const double wave_number = depth_phase / settings.water_depth_m;
  if (!std::isnormal(wave_number)) {
    return absl::OutOfRangeError("ocean wave number is outside the supported numeric range");
  }
  const double amplitude = settings.wave_height_m / 2;
  const double sigma = std::tanh(depth_phase);
  // Bound the second elevation harmonic relative to the first and reject
  // waves outside this weakly nonlinear model's declared operating range.
  const double harmonic_ratio =
      amplitude == 0 ? 0 : (wave_number * amplitude / sigma) / sigma / sigma * (3 - sigma * sigma) / 4;
  if (!std::isfinite(harmonic_ratio) || harmonic_ratio > 0.25) {
    return absl::OutOfRangeError("second-order ocean swell exceeds the supported harmonic ratio");
  }
  ResolvedOceanSwell3D result;
  result.direction_ = {
      settings.propagation_direction.x / direction_length,
      0,
      settings.propagation_direction.z / direction_length
  };
  result.still_water_height_m_ = settings.still_water_height_m;
  result.water_depth_m_ = settings.water_depth_m;
  result.depth_phase_rad_ = depth_phase;
  result.wave_number_rad_per_m_ = wave_number;
  result.angular_frequency_rad_per_s_ = frequency;
  result.phase_at_world_origin_rad_ = std::remainder(settings.phase_at_world_origin_rad, kFullTurn);
  result.amplitude_m_ = amplitude;
  result.elevation_second_harmonic_m_ = amplitude * harmonic_ratio;
  if (amplitude > 0) {
    // Factor out the large exponential in sinh(kh) and cosh(n*k*y).
    const double scaled_sinh = -std::expm1(-2 * depth_phase);
    const double amplitude_speed = amplitude * frequency;
    result.velocity_first_harmonic_m_per_s_ = amplitude_speed / scaled_sinh;
    result.velocity_second_harmonic_m_per_s_ = 6 * amplitude_speed * (amplitude * wave_number)
        * std::exp(-2 * depth_phase) / scaled_sinh / scaled_sinh / scaled_sinh / scaled_sinh;
    result.return_current_m_per_s_ = -0.5 * amplitude_speed * (amplitude / settings.water_depth_m) / sigma;
  }
  if (!std::isfinite(result.velocity_first_harmonic_m_per_s_)
      || !std::isfinite(result.velocity_second_harmonic_m_per_s_) || !std::isfinite(result.return_current_m_per_s_)) {
    return absl::OutOfRangeError("ocean swell velocity coefficients overflow");
  }
  return result;
}

absl::StatusOr<OceanSwellSample3D> ResolvedOceanSwell3D::Sample(Vec3 world_position_m, double time_s) const {
  if (!Finite(world_position_m) || !std::isfinite(time_s)) {
    return absl::InvalidArgumentError("ocean swell sample coordinates and time must be finite");
  }
  if (amplitude_m_ == 0) {
    return OceanSwellSample3D{.surface_height_m = still_water_height_m_, .world_velocity_m_per_s = {}};
  }
  constexpr double kFullTurn = 2 * std::numbers::pi;
  const double propagation_distance_m = direction_.x * world_position_m.x + direction_.z * world_position_m.z;
  const double spatial_phase = propagation_distance_m * wave_number_rad_per_m_;
  const double temporal_phase = time_s * angular_frequency_rad_per_s_;
  if (!std::isfinite(spatial_phase) || !std::isfinite(temporal_phase)) {
    return absl::OutOfRangeError("ocean swell phase overflow");
  }
  const double phase =
      std::remainder(spatial_phase, kFullTurn) - std::remainder(temporal_phase, kFullTurn) + phase_at_world_origin_rad_;
  const double cosine = std::cos(phase);
  const double sine = std::sin(phase);
  const double cosine_twice = cosine * cosine - sine * sine;
  const double sine_twice = 2 * sine * cosine;
  const double below_surface_m = std::clamp(still_water_height_m_ - world_position_m.y, 0.0, water_depth_m_);
  const double depth_fraction = below_surface_m / water_depth_m_;
  const double depth_below_surface = depth_phase_rad_ * depth_fraction;
  const double depth_above_bed = depth_phase_rad_ * (1 - depth_fraction);
  const double first_scale = std::exp(-depth_below_surface);
  const double first_reflected = std::exp(-2 * depth_above_bed);
  const double second_reflected = first_reflected * first_reflected;
  const double first_velocity = velocity_first_harmonic_m_per_s_ * first_scale;
  const double second_velocity = velocity_second_harmonic_m_per_s_ * first_scale * first_scale;
  const double horizontal_velocity = return_current_m_per_s_ + first_velocity * (1 + first_reflected) * cosine
      + second_velocity * (1 + second_reflected) * cosine_twice;
  const double vertical_velocity = first_velocity * -std::expm1(-2 * depth_above_bed) * sine
      + second_velocity * -std::expm1(-4 * depth_above_bed) * sine_twice;
  const OceanSwellSample3D result{
      .surface_height_m = still_water_height_m_ + amplitude_m_ * cosine + elevation_second_harmonic_m_ * cosine_twice,
      .world_velocity_m_per_s =
          {direction_.x * horizontal_velocity, vertical_velocity, direction_.z * horizontal_velocity},
  };
  if (!std::isfinite(result.surface_height_m) || !Finite(result.world_velocity_m_per_s)) {
    return absl::OutOfRangeError("ocean swell sample overflow");
  }
  return result;
}

absl::StatusOr<double> RelaxOceanVelocity(
    double velocity_m_per_s,
    double target_m_per_s,
    double zone_fraction,
    double rate_per_s,
    double duration_s
) {
  if (!std::isfinite(velocity_m_per_s) || !std::isfinite(target_m_per_s) || !std::isfinite(zone_fraction)
      || !std::isfinite(rate_per_s) || !std::isfinite(duration_s) || zone_fraction < 0 || zone_fraction > 1
      || rate_per_s < 0 || duration_s < 0) {
    return absl::InvalidArgumentError("invalid ocean absorption velocities, zone fraction, rate or duration");
  }
  const double ramp = zone_fraction * zone_fraction * (3 - 2 * zone_fraction);
  const double weight = -std::expm1(-(ramp * rate_per_s) * duration_s);
  return std::lerp(velocity_m_per_s, target_m_per_s, weight);
}
}  // namespace fluidity
