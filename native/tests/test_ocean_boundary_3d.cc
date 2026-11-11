#include <cmath>
#include <numbers>
#include <vector>

#include "fluidity/host/ocean_boundary_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
TEST(OceanBoundary3D, SatisfiesDispersionAndSecondOrderFiniteDepthProfile) {
  const auto swell = ResolvedOceanSwell3D::Resolve({
      .still_water_height_m = 1,
      .water_depth_m = 4,
      .wave_height_m = 1,
      .wave_period_s = 4,
      .propagation_direction = {0, 0, 2},
  });
  ASSERT_TRUE(swell.ok()) << swell.status();
  const double k = swell->wave_number_rad_per_m();
  const double omega = std::numbers::pi / 2;
  EXPECT_NEAR(9.81 * k * std::tanh(4 * k), omega * omega, 1e-13);
  const double sigma = std::tanh(4 * k);
  const double second_elevation = k * 0.25 * (3 - sigma * sigma) / (4 * sigma * sigma * sigma);
  const auto crest = swell->Sample({0, -3, 0}, 0);
  const auto trough = swell->Sample({0, -3, 0}, 2);
  ASSERT_TRUE(crest.ok()) << crest.status();
  ASSERT_TRUE(trough.ok()) << trough.status();
  EXPECT_NEAR(crest->surface_height_m, 1.5 + second_elevation, 1e-14);
  EXPECT_NEAR(trough->surface_height_m, 0.5 + second_elevation, 1e-14);
  EXPECT_DOUBLE_EQ(crest->world_velocity_m_per_s.x, 0);
  EXPECT_DOUBLE_EQ(crest->world_velocity_m_per_s.y, 0);
  const auto bed_quadrature = swell->Sample({0, -3, 0}, 1);
  const auto water_quadrature = swell->Sample({0, -1, 0}, 1);
  ASSERT_TRUE(bed_quadrature.ok()) << bed_quadrature.status();
  ASSERT_TRUE(water_quadrature.ok()) << water_quadrature.status();
  EXPECT_DOUBLE_EQ(bed_quadrature->world_velocity_m_per_s.y, 0);
  EXPECT_NEAR(water_quadrature->world_velocity_m_per_s.y, -0.5 * omega * std::sinh(2 * k) / std::sinh(4 * k), 1e-14);
  const double bed_second_velocity = 0.75 * 0.25 * omega * k / std::pow(std::sinh(4 * k), 4);
  const double return_current = -0.25 * omega / (8 * sigma);
  EXPECT_NEAR(
      crest->world_velocity_m_per_s.z,
      0.5 * omega / std::sinh(4 * k) + bed_second_velocity + return_current,
      1e-14
  );
  EXPECT_NEAR(
      (crest->world_velocity_m_per_s.z + trough->world_velocity_m_per_s.z) / 2
          + bed_quadrature->world_velocity_m_per_s.z,
      2 * return_current,
      1e-14
  );
}

TEST(OceanBoundary3D, ReferenceInletReturnsTheDepthIntegratedStokesDrift) {
  constexpr double kDepthM = 3;
  constexpr double kPeriodS = 4;
  constexpr double kAmplitudeM = 0.5;
  const auto swell = ResolvedOceanSwell3D::Resolve({
      .water_depth_m = kDepthM,
      .wave_height_m = 2 * kAmplitudeM,
      .wave_period_s = kPeriodS,
  });
  ASSERT_TRUE(swell.ok()) << swell.status();
  const double wave_number = swell->wave_number_rad_per_m();
  const double angular_frequency = 2 * std::numbers::pi / kPeriodS;
  constexpr int kDepthSamples = 256;
  const double depth_spacing_m = kDepthM / kDepthSamples;
  const double sinh_depth_phase = std::sinh(wave_number * kDepthM);
  double return_flux_m2_per_s = 0;
  double drift_flux_m2_per_s = 0;
  for (int index = 0; index < kDepthSamples; ++index) {
    const double height_above_bed_m = (index + 0.5) * depth_spacing_m;
    const Vec3 position{0, height_above_bed_m - kDepthM, 0};
    double mean_velocity_m_per_s = 0;
    for (int quarter = 0; quarter < 4; ++quarter) {
      const auto sample = swell->Sample(position, quarter * kPeriodS / 4);
      ASSERT_TRUE(sample.ok()) << sample.status();
      mean_velocity_m_per_s += sample->world_velocity_m_per_s.x / 4;
    }
    return_flux_m2_per_s += mean_velocity_m_per_s * depth_spacing_m;
    const double drift_velocity_m_per_s = kAmplitudeM * kAmplitudeM * angular_frequency * wave_number
        * std::cosh(2 * wave_number * height_above_bed_m) / (2 * sinh_depth_phase * sinh_depth_phase);
    drift_flux_m2_per_s += drift_velocity_m_per_s * depth_spacing_m;
  }
  EXPECT_GT(drift_flux_m2_per_s, 0.2);
  EXPECT_LT(return_flux_m2_per_s, -0.2);
  EXPECT_NEAR(return_flux_m2_per_s + drift_flux_m2_per_s, 0, 1e-6);
}

TEST(OceanBoundary3D, OutletReducesReturnEnergyInLinearWaveFlume) {
  constexpr std::size_t kCells = 400;
  constexpr double kSpacingM = 0.05;
  constexpr double kTimeStepS = 0.00625;
  constexpr double kDepthM = 1;
  constexpr double kGravityMPerS2 = 9.81;
  constexpr double kAmplitudeM = 0.01;
  constexpr double kPulseCenterM = 4;
  constexpr double kPulseWidthM = 1;
  constexpr std::size_t kGauge = 200;
  constexpr int kSteps = 1840;
  const double celerity_m_per_s = std::sqrt(kGravityMPerS2 * kDepthM);
  const auto elevation_at = [&](double position_m) {
    const double normalized_offset = (position_m - kPulseCenterM) / kPulseWidthM;
    return kAmplitudeM * std::exp(-0.5 * normalized_offset * normalized_offset);
  };
  struct GaugeEnergy {
    double incident = 0;
    double reflected = 0;
  };
  const auto measure = [&](double outlet_rate_per_s) -> absl::StatusOr<GaugeEnergy> {
    const auto boundary = ResolvedOceanBoundary3D::Resolve({
        .outlet_interior_x_m = 14,
        .outlet_wall_x_m = 20,
        .outlet_rate_per_s = outlet_rate_per_s,
    });
    if (!boundary.ok()) {
      return boundary.status();
    }
    // Staggered linear shallow-water flume, with a closed wall at each end.
    // A right-going pulse starts upstream of the gauge. Sum characteristic
    // energy separately over the first outgoing and returning arrivals.
    std::vector<double> elevation_m(kCells);
    std::vector<double> face_velocity_m_per_s(kCells + 1);
    for (std::size_t index = 0; index < kCells; ++index) {
      elevation_m[index] = elevation_at((index + 0.5) * kSpacingM);
    }
    for (std::size_t index = 1; index < kCells; ++index) {
      const double prior_x_m = index * kSpacingM + 0.5 * kTimeStepS * celerity_m_per_s;
      face_velocity_m_per_s[index] = elevation_at(prior_x_m) * celerity_m_per_s / kDepthM;
    }
    GaugeEnergy energy;
    for (int step = 0; step < kSteps; ++step) {
      const double time_s = (step + 1) * kTimeStepS;
      for (std::size_t index = 1; index < kCells; ++index) {
        const double slope = (elevation_m[index] - elevation_m[index - 1]) / kSpacingM;
        const double predicted_velocity_m_per_s = face_velocity_m_per_s[index] - kGravityMPerS2 * kTimeStepS * slope;
        const auto relaxed = boundary->RelaxFace(
            FaceNormalAxis::kX,
            {index * kSpacingM, 0, 0},
            predicted_velocity_m_per_s,
            time_s,
            kTimeStepS
        );
        if (!relaxed.ok()) {
          return relaxed.status();
        }
        face_velocity_m_per_s[index] = *relaxed;
      }
      for (std::size_t index = 0; index < kCells; ++index) {
        const double velocity_gradient_per_s =
            (face_velocity_m_per_s[index + 1] - face_velocity_m_per_s[index]) / kSpacingM;
        elevation_m[index] -= kDepthM * kTimeStepS * velocity_gradient_per_s;
      }
      const double gauge_elevation_m = 0.5 * (elevation_m[kGauge - 1] + elevation_m[kGauge]);
      const double characteristic_velocity_m = face_velocity_m_per_s[kGauge] * kDepthM / celerity_m_per_s;
      if (time_s < 4) {
        const double outgoing = 0.5 * (gauge_elevation_m + characteristic_velocity_m);
        energy.incident += outgoing * outgoing * kTimeStepS;
      } else {
        const double returning = 0.5 * (gauge_elevation_m - characteristic_velocity_m);
        energy.reflected += returning * returning * kTimeStepS;
      }
    }
    return energy;
  };
  const auto closed = measure(0);
  const auto absorbing = measure(2);
  ASSERT_TRUE(closed.ok()) << closed.status();
  ASSERT_TRUE(absorbing.ok()) << absorbing.status();
  ASSERT_GT(closed->incident, 0);
  const double closed_reflection = closed->reflected / closed->incident;
  const double absorbed_reflection = absorbing->reflected / absorbing->incident;
  RecordProperty("closed_reflected_energy_fraction", closed_reflection);
  RecordProperty("absorbing_reflected_energy_fraction", absorbed_reflection);
  EXPECT_NEAR(absorbing->incident, closed->incident, closed->incident * 0.001);
  EXPECT_NEAR(closed_reflection, 1, 0.02);
  EXPECT_LT(absorbed_reflection, 0.25);
}

TEST(OceanBoundary3D, DeepWaterAndFrozenTargetSubdivisionRemainFinite) {
  const auto swell = ResolvedOceanSwell3D::Resolve({.water_depth_m = 20000, .wave_height_m = 0.1});
  ASSERT_TRUE(swell.ok()) << swell.status();
  const auto sample = swell->Sample({0, 0, 0}, 0);
  ASSERT_TRUE(sample.ok()) << sample.status();
  EXPECT_NEAR(sample->surface_height_m, 0.05 + 0.5 * 0.05 * 0.05 * swell->wave_number_rad_per_m(), 1e-14);
  const auto whole = RelaxOceanVelocity(3, -1, 0.6, 4, 0.2);
  auto split = RelaxOceanVelocity(3, -1, 0.6, 4, 0.1);
  ASSERT_TRUE(split.ok()) << split.status();
  split = RelaxOceanVelocity(*split, -1, 0.6, 4, 0.1);
  ASSERT_TRUE(split.ok()) << split.status();
  ASSERT_TRUE(whole.ok()) << whole.status();
  EXPECT_NEAR(*split, *whole, 1e-14);
  EXPECT_EQ(RelaxOceanVelocity(3, -1, 0, 4, 0.2).value(), 3);
  EXPECT_FALSE(RelaxOceanVelocity(3, -1, 1.1, 4, 0.2).ok());
  EXPECT_FALSE(ResolvedOceanSwell3D::Resolve({.water_depth_m = 0}).ok());
  EXPECT_FALSE(ResolvedOceanSwell3D::Resolve({.propagation_direction = {1, 1, 0}}).ok());
  EXPECT_FALSE(ResolvedOceanSwell3D::Resolve({.water_depth_m = 0.1, .wave_height_m = 1, .wave_period_s = 4}).ok());
}
}  // namespace
}  // namespace fluidity
