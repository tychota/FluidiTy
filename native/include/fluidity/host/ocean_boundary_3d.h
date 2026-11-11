#pragma once

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {
struct OceanSwellSettings3D {
  double still_water_height_m = 0;
  double water_depth_m = 1;
  double wave_height_m = 0;  // Twice the first harmonic amplitude.
  double wave_period_s = 1;
  double gravity_m_per_s2 = 9.81;
  Vec3 propagation_direction{1, 0, 0};  // Horizontal, normalized at resolve.
  double phase_at_world_origin_rad = 0;
};

struct OceanBoundarySettings3D {
  OceanSwellSettings3D swell{};
  double inlet_wall_x_m = 0;
  double inlet_interior_x_m = 1;
  double outlet_interior_x_m = 2;
  double outlet_wall_x_m = 3;
  double inlet_rate_per_s = 0;
  double outlet_rate_per_s = 0;
  // Zero disables particle level control. Positive spacing seeds vacant wet
  // sites in the inlet strip, with one spacing-cubed reference volume per site.
  double inlet_particle_spacing_m = 0;
  double inlet_low_z_m = 0;
  double inlet_high_z_m = 0;
};

struct OceanSwellSample3D {
  double surface_height_m;
  Vec3 world_velocity_m_per_s;
};

// Finite-depth second-order Stokes target with a uniform return current to
// cancel depth-integrated Stokes drift to second order. Resolving precomputes
// the dispersion root and harmonic coefficients once per authored swell.
class ResolvedOceanSwell3D {
 public:
  // Returns InvalidArgument for nonphysical or nonfinite controls and
  // OutOfRange if the dispersion solve fails or the second elevation harmonic
  // exceeds one quarter of the first.
  static absl::StatusOr<ResolvedOceanSwell3D> Resolve(const OceanSwellSettings3D& settings);

  // Samples the phase at world position (m) and time (s). Vertical velocity
  // profiles are clamped to the bed and still-water elevations. Returns
  // InvalidArgument for nonfinite arguments and OutOfRange on overflow.
  absl::StatusOr<OceanSwellSample3D> Sample(Vec3 world_position_m, double time_s) const;

  // Positive dispersion root in radians per world meter.
  double wave_number_rad_per_m() const {
    return wave_number_rad_per_m_;
  }

 private:
  ResolvedOceanSwell3D() = default;
  Vec3 direction_{};
  double still_water_height_m_ = 0;
  double water_depth_m_ = 0;
  double depth_phase_rad_ = 0;
  double wave_number_rad_per_m_ = 0;
  double angular_frequency_rad_per_s_ = 0;
  double phase_at_world_origin_rad_ = 0;
  double amplitude_m_ = 0;
  double elevation_second_harmonic_m_ = 0;
  double velocity_first_harmonic_m_per_s_ = 0;
  double velocity_second_harmonic_m_per_s_ = 0;
  double return_current_m_per_s_ = 0;
};

// Exact frozen-target exponential relaxation with cubic ramp over zone
// fraction [0,1], where zero is the interior edge. Rates are in 1/s, duration
// in seconds, velocities in m/s. Returns InvalidArgument for invalid controls;
// does not mutate its input.
absl::StatusOr<double> RelaxOceanVelocity(
    double velocity_m_per_s,
    double target_m_per_s,
    double zone_fraction,
    double rate_per_s,
    double duration_s
);
class ResolvedOceanBoundary3D {
 public:
  // Resolve the inlet swell and two non-overlapping x intervals. The outlet
  // relaxes to rest. Each rate is nonnegative, with zero disabling that zone.
  static absl::StatusOr<ResolvedOceanBoundary3D> Resolve(const OceanBoundarySettings3D& settings);

  // Relax one checked MAC face velocity toward the frozen target at time_s.
  // The input velocity is returned unchanged in the interior.
  absl::StatusOr<double>
  RelaxFace(FaceNormalAxis axis, Vec3 world_anchor_m, double velocity_m_per_s, double time_s, double duration_s) const;

  // Samples the prescribed inlet elevation and world velocity.
  absl::StatusOr<OceanSwellSample3D> SampleInlet(Vec3 world_position_m, double time_s) const {
    return swell_.Sample(world_position_m, time_s);
  }

 private:
  ResolvedOceanBoundary3D(OceanBoundarySettings3D settings, ResolvedOceanSwell3D swell)
      : settings_(settings), swell_(swell) {
  }
  OceanBoundarySettings3D settings_;
  ResolvedOceanSwell3D swell_;
};
}  // namespace fluidity
