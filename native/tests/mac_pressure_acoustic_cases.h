#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

#include "../src/mac_pressure_3d_system.h"
#include "gtest/gtest.h"

namespace fluidity::pressure_test {

enum class AcousticInitialPressure {
  kStandingWave,
  kLocalizedPulse
};

// A unit-length, unit-area column with sealed ends. The cosine pressure and
// sine velocity modes are an exact eigenpair of the staggered incidence.
template <typename Project>
void ExpectBackwardEulerAcousticColumn(AcousticInitialPressure initial, Project project) {
  constexpr double kDensity = 1000.0;
  constexpr double kSoundSpeed = 2.0;
  constexpr double kBulkModulus = kDensity * kSoundSpeed * kSoundSpeed;
  constexpr double kImpedance = kDensity * kSoundSpeed;
  constexpr double kInitialPressureAmplitude = 2.0;
  constexpr int kStepCount = 16;
  const auto start = std::chrono::steady_clock::now();
  std::size_t iterations = 0;
  double maximum_pressure_error = 0.0;
  double maximum_scaled_velocity_error = 0.0;
  for (const int cell_count : {8, 16}) {
    const double spacing = 1.0 / cell_count;
    const double lattice_angle = std::numbers::pi * spacing;
    const bool pulse = initial == AcousticInitialPressure::kLocalizedPulse;
    const int pulse_cell = cell_count / 4;
    // The sealed-column cosine basis is orthogonal. The constant mode
    // carries the pulse mean, and every nonconstant mode has squared norm N/2.
    const double mean_pressure = pulse ? kInitialPressureAmplitude / cell_count : 0.0;
    for (const double step_duration : {0.01, 0.025}) {
      SCOPED_TRACE(::testing::Message() << "cells=" << cell_count << " dt=" << step_duration);
      std::vector<MacPressureCell3D> cells;
      std::vector<pressure_internal::ExplicitMacPressureFace3D> faces;
      for (int cell = 0; cell < cell_count; ++cell) {
        cells.push_back({
            .cell_index = {static_cast<std::int32_t>(cell), 0, 0},
            .fluid_volume = spacing,
            .bulk_modulus = kBulkModulus,
            .previous_pressure = pulse ? (cell == pulse_cell ? kInitialPressureAmplitude : 0.0)
                                       : kInitialPressureAmplitude * std::cos(lattice_angle * (cell + 0.5)),
        });
      }
      for (int face = 1; face < cell_count; ++face) {
        faces.push_back({
            .face_slot = static_cast<std::size_t>(face - 1),
            .tail_cell_ordinal = static_cast<std::size_t>(face - 1),
            .head_cell_ordinal = static_cast<std::size_t>(face),
            .open_area = 1.0,
            .inverse_density = 1.0 / kDensity,
            .pressure_gradient_distance = spacing,
            .predicted_normal_velocity = 0.0,
        });
      }
      double accumulated_error_bound = 0.0;
      for (int step = 1; step <= kStepCount; ++step) {
        SCOPED_TRACE(step);
        const auto system = pressure_internal::PrepareExplicitMacPressureSystem3D(step_duration, cells, faces);
        ASSERT_TRUE(system.ok()) << system.status();
        ASSERT_TRUE(system->gauge_cell_ordinals().empty());
        const auto projected = project(*system);
        ASSERT_TRUE(projected.ok()) << projected.status();
        ASSERT_EQ(projected->cell_pressures.size(), cells.size());
        ASSERT_EQ(projected->corrected_face_velocities.size(), faces.size());
        ASSERT_EQ(projected->cell_constraint_defects.size(), cells.size());
        double squared_defect = 0.0;
        for (const double defect : projected->cell_constraint_defects) {
          squared_defect += defect * defect;
        }
        // In [p, rho*c*u] coordinates, the equal-volume backward-Euler
        // update is a contraction. A pressure-row defect adds K*dt*defect
        // to the pressure equation, so its L2 norm bounds accumulated error.
        accumulated_error_bound += kBulkModulus * step_duration * std::sqrt(squared_defect);
        double squared_error = 0.0;
        std::vector<double> expected_pressure(cells.size(), mean_pressure);
        std::vector<double> expected_scaled_velocity(faces.size());
        const int mode_limit = pulse ? cell_count : 2;
        for (int mode = 1; mode < mode_limit; ++mode) {
          const double mode_angle = mode * lattice_angle;
          const double frequency = kSoundSpeed * 2.0 * std::sin(mode_angle / 2.0) / spacing;
          const double frequency_step = frequency * step_duration;
          const double damping = 1.0 / std::hypot(1.0, frequency_step);
          const double phase = step * std::atan(frequency_step);
          const double initial_amplitude =
              pulse ? 2.0 * mean_pressure * std::cos(mode_angle * (pulse_cell + 0.5)) : kInitialPressureAmplitude;
          const double amplitude = initial_amplitude * std::pow(damping, step);
          const double pressure_amplitude = amplitude * std::cos(phase);
          const double scaled_velocity_amplitude = amplitude * std::sin(phase);
          for (std::size_t cell = 0; cell < cells.size(); ++cell) {
            expected_pressure[cell] += pressure_amplitude * std::cos(mode_angle * (static_cast<double>(cell) + 0.5));
          }
          for (std::size_t face = 0; face < faces.size(); ++face) {
            expected_scaled_velocity[face] +=
                scaled_velocity_amplitude * std::sin(mode_angle * static_cast<double>(face + 1));
          }
        }
        for (std::size_t cell = 0; cell < cells.size(); ++cell) {
          const double error = projected->cell_pressures[cell] - expected_pressure[cell];
          squared_error += error * error;
          maximum_pressure_error = std::max(maximum_pressure_error, std::abs(error));
          cells[cell].previous_pressure = projected->cell_pressures[cell];
        }
        for (std::size_t face = 0; face < faces.size(); ++face) {
          const double error = kImpedance * projected->corrected_face_velocities[face] - expected_scaled_velocity[face];
          squared_error += error * error;
          maximum_scaled_velocity_error = std::max(maximum_scaled_velocity_error, std::abs(error));
          faces[face].predicted_normal_velocity = projected->corrected_face_velocities[face];
        }
        EXPECT_LE(std::sqrt(squared_error), accumulated_error_bound + 2e-12);
        iterations += projected->diagnostics.iteration_count;
      }
    }
  }
  // Independently cap pointwise waveform error as a fraction of the initial
  // amplitude. The single-cell pulse excites the full spectrum.
  const double waveform_tolerance = initial == AcousticInitialPressure::kLocalizedPulse ? 0.001 : 0.0005;
  EXPECT_LE(maximum_pressure_error, waveform_tolerance * kInitialPressureAmplitude);
  EXPECT_LE(maximum_scaled_velocity_error, waveform_tolerance * kInitialPressureAmplitude);
  const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
  ::testing::Test::RecordProperty("acoustic_pressure_iterations", static_cast<int>(iterations));
  ::testing::Test::RecordProperty("acoustic_wall_milliseconds", std::to_string(elapsed.count()));
  ::testing::Test::RecordProperty("acoustic_max_pressure_error_pa", std::to_string(maximum_pressure_error));
  ::testing::Test::RecordProperty(
      "acoustic_max_scaled_velocity_error_pa",
      std::to_string(maximum_scaled_velocity_error)
  );
}

}  // namespace fluidity::pressure_test
