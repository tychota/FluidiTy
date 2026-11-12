#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "../src/mac_pressure_3d_system.h"
#include "fluidity/host/liquid_eos.h"
#include "gtest/gtest.h"

namespace fluidity::pressure_test {

enum class AcousticInitialCondition {
  kStandingWave,
  kLocalizedPulse,
  kValveClosure
};

// A unit-length, unit-area column with sealed ends. The cosine pressure and
// sine velocity modes are an exact eigenpair of the staggered incidence.
template <typename Project>
void ExpectBackwardEulerAcousticColumn(AcousticInitialCondition initial, Project project) {
  constexpr double kDensity = 1000.0;
  constexpr double kSoundSpeed = 2.0;
  constexpr double kBulkModulus = kDensity * kSoundSpeed * kSoundSpeed;
  constexpr double kImpedance = kDensity * kSoundSpeed;
  constexpr double kInitialPressureAmplitude = 2.0;
  const bool valve = initial == AcousticInitialCondition::kValveClosure;
  const bool pulse = initial == AcousticInitialCondition::kLocalizedPulse;
  const double initial_velocity = valve ? kInitialPressureAmplitude / kImpedance : 0.0;
  struct ColumnCase {
    int cell_count;
    double step_duration;
    int step_count;
  };
  const std::vector<ColumnCase> cases = valve
      ? std::vector<ColumnCase>{{16, 1.0 / 128.0, 32}, {32, 1.0 / 256.0, 64}, {64, 1.0 / 512.0, 128}}
      : std::vector<ColumnCase>{{8, 0.01, 16}, {8, 0.025, 16}, {16, 0.01, 16}, {16, 0.025, 16}};
  const auto start = std::chrono::steady_clock::now();
  std::size_t iterations = 0;
  double maximum_pressure_error = 0.0;
  double maximum_scaled_velocity_error = 0.0;
  double valve_plateau_error = std::numeric_limits<double>::infinity();
  for (const auto& [cell_count, step_duration, step_count] : cases) {
    const double spacing = 1.0 / cell_count;
    const double lattice_angle = std::numbers::pi * spacing;
    const int pulse_cell = cell_count / 4;
    // The sealed-column cosine basis is orthogonal. The constant mode
    // carries the pulse mean, and every nonconstant mode has squared norm N/2.
    const double mean_pressure = pulse ? kInitialPressureAmplitude / cell_count : 0.0;
    SCOPED_TRACE(::testing::Message() << "cells=" << cell_count << " dt=" << step_duration);
    std::vector<MacPressureCell3D> cells;
    std::vector<pressure_internal::ExplicitMacPressureFace3D> faces;
    for (int cell = 0; cell < cell_count; ++cell) {
      cells.push_back({
          .cell_index = {static_cast<std::int32_t>(cell), 0, 0},
          .fluid_volume = spacing,
          .bulk_modulus = kBulkModulus,
          .previous_pressure = valve ? 0.0
              : pulse                ? (cell == pulse_cell ? kInitialPressureAmplitude : 0.0)
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
          .predicted_normal_velocity = initial_velocity,
      });
    }
    double accumulated_error_bound = 0.0;
    for (int step = 1; step <= step_count; ++step) {
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
      const int mode_limit = pulse || valve ? cell_count : 2;
      for (int mode = 1; mode < mode_limit; ++mode) {
        const double mode_angle = mode * lattice_angle;
        const double frequency = kSoundSpeed * 2.0 * std::sin(mode_angle / 2.0) / spacing;
        const double frequency_step = frequency * step_duration;
        const double damping = 1.0 / std::hypot(1.0, frequency_step);
        const double phase = step * std::atan(frequency_step);
        const double initial_amplitude =
            pulse ? 2.0 * mean_pressure * std::cos(mode_angle * (pulse_cell + 0.5)) : kInitialPressureAmplitude;
        // For valve closure, project the pre-closure uniform interior
        // velocity onto the sine modes of the newly sealed column.
        double initial_scaled_velocity = 0.0;
        if (valve) {
          for (int face = 1; face < cell_count; ++face) {
            initial_scaled_velocity += std::sin(mode_angle * face);
          }
          initial_scaled_velocity *= 2.0 * kImpedance * initial_velocity / cell_count;
        }
        const double damping_at_step = std::pow(damping, step);
        const double initial_pressure = valve ? 0.0 : initial_amplitude;
        const double pressure_amplitude =
            damping_at_step * (initial_pressure * std::cos(phase) - initial_scaled_velocity * std::sin(phase));
        const double scaled_velocity_amplitude =
            damping_at_step * (initial_pressure * std::sin(phase) + initial_scaled_velocity * std::cos(phase));
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
    if (valve) {
      // At 0.25 s the closure wave has reached half the one-metre
      // column. Average the last eighth, away from the wavefront.
      const auto plateau = std::span(cells).last(static_cast<std::size_t>(cell_count / 8));
      double pressure_sum = 0.0;
      for (const auto& cell : plateau) {
        pressure_sum += cell.previous_pressure;
      }
      const double plateau_pressure = pressure_sum / static_cast<double>(plateau.size());
      const double relative_error = std::abs(plateau_pressure / (kImpedance * initial_velocity) - 1.0);
      EXPECT_LT(relative_error, valve_plateau_error);
      valve_plateau_error = relative_error;
    }
  }
  if (valve) {
    EXPECT_LT(valve_plateau_error, 0.05);
    ::testing::Test::RecordProperty("water_hammer_finest_plateau_relative_error", std::to_string(valve_plateau_error));
  }
  // Independently cap pointwise waveform error as a fraction of the initial
  // amplitude. The single-cell pulse excites the full spectrum.
  const double waveform_tolerance = valve ? 0.005 : pulse ? 0.001 : 0.0005;
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

// A sealed pressure cell with a prescribed piston flux over a quarter face.
// The swept volume and scalar EOS pressure recurrence are independent of the
// geometry and pressure assembly code.
template <typename Project>
void ExpectCompliantPistonCompression(Project project) {
  constexpr double kDensity = 1000.0;
  constexpr double kSoundSpeed = 50.0;
  constexpr double kBulkModulus = kDensity * kSoundSpeed * kSoundSpeed;
  constexpr double kPistonSpeed = 0.08;
  constexpr double kWettedArea = 0.25;
  constexpr double kInwardFlux = kPistonSpeed * kWettedArea;
  constexpr double kFinalVolumeRatio = 1.0 - kInwardFlux;
  constexpr UniformMacGeometry3D kGeometry{{0.0, 0.0, 0.0}, 1.0};
  constexpr std::array kBlocks{BlockIndex3D{0, 0, 0}};
  const auto grid = SparseMacGrid3D::Create(kGeometry, kBlocks, kBlocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array faces{
      MacPressureFace3D{
          .face_index = {FaceNormalAxis::kX, {0, 0, 0}},
          .wall_normal_velocity = kPistonSpeed,
          .pressure_link = SealedMacPressureLink3D{},
          .prescribed_wall_flux = kInwardFlux,
      },
      MacPressureFace3D{.face_index = {FaceNormalAxis::kX, {1, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      MacPressureFace3D{.face_index = {FaceNormalAxis::kY, {0, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      MacPressureFace3D{.face_index = {FaceNormalAxis::kY, {0, 1, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      MacPressureFace3D{.face_index = {FaceNormalAxis::kZ, {0, 0, 0}}, .pressure_link = SealedMacPressureLink3D{}},
      MacPressureFace3D{.face_index = {FaceNormalAxis::kZ, {0, 0, 1}}, .pressure_link = SealedMacPressureLink3D{}},
  };
  const auto start = std::chrono::steady_clock::now();
  std::size_t iterations = 0;
  double maximum_finest_relative_error = 0.0;
  double maximum_constraint_defect = 0.0;
  for (const auto law : {LiquidEosLaw::kHencky, LiquidEosLaw::kTait}) {
    SCOPED_TRACE(static_cast<int>(law));
    const double exact_pressure = law == LiquidEosLaw::kHencky
        ? -kBulkModulus * std::log(kFinalVolumeRatio)
        : (kBulkModulus / 7.0) * std::expm1(-7.0 * std::log(kFinalVolumeRatio));
    double previous_relative_error = std::numeric_limits<double>::infinity();
    for (const int step_count : {16, 32, 64}) {
      SCOPED_TRACE(step_count);
      const double dt = 1.0 / step_count;
      double pressure = 0.0;
      for (int step = 0; step < step_count; ++step) {
        SCOPED_TRACE(step);
        const double volume = 1.0 - step * dt * kInwardFlux;
        const auto eos = EvaluateLiquidEosAtPressure3D({kDensity, kSoundSpeed, law}, pressure);
        ASSERT_TRUE(eos.ok()) << eos.status();
        const double expected_modulus = law == LiquidEosLaw::kHencky ? kBulkModulus : kBulkModulus + 7.0 * pressure;
        EXPECT_NEAR(eos->bulk_modulus, expected_modulus, 1e-9);
        const std::array cells{MacPressureCell3D{
            .cell_index = {0, 0, 0},
            .fluid_volume = volume,
            .bulk_modulus = eos->bulk_modulus,
            .previous_pressure = pressure,
        }};
        const auto projected = project(MacPressureInputs3D{*grid, dt, cells, faces});
        ASSERT_TRUE(projected.ok()) << projected.status();
        ASSERT_TRUE(projected->gauge_cell_ordinals.empty());
        ASSERT_EQ(projected->cell_pressures.size(), 1U);
        ASSERT_EQ(projected->cell_constraint_defects.size(), 1U);
        const double defect = std::abs(projected->cell_constraint_defects.front());
        EXPECT_LE(defect, 2e-5);
        maximum_constraint_defect = std::max(maximum_constraint_defect, defect);
        const double expected_pressure = pressure + dt * expected_modulus * kInwardFlux / volume;
        EXPECT_NEAR(projected->cell_pressures.front(), expected_pressure, dt * expected_modulus * defect + 1e-8);
        EXPECT_GT(projected->cell_pressures.front(), pressure);
        pressure = projected->cell_pressures.front();
        iterations += projected->diagnostics.iteration_count;
      }
      const double relative_error = std::abs(pressure - exact_pressure) / exact_pressure;
      EXPECT_LT(relative_error, 0.55 * previous_relative_error);
      previous_relative_error = relative_error;
    }
    EXPECT_LT(previous_relative_error, 0.002);
    maximum_finest_relative_error = std::max(maximum_finest_relative_error, previous_relative_error);
  }
  const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
  ::testing::Test::RecordProperty("piston_pressure_iterations", static_cast<int>(iterations));
  ::testing::Test::RecordProperty("piston_wall_milliseconds", std::to_string(elapsed.count()));
  ::testing::Test::RecordProperty("piston_max_finest_relative_error", std::to_string(maximum_finest_relative_error));
  std::ostringstream defect_receipt;
  defect_receipt << std::scientific << maximum_constraint_defect;
  ::testing::Test::RecordProperty("piston_max_constraint_defect_per_s", defect_receipt.str());
}

}  // namespace fluidity::pressure_test
