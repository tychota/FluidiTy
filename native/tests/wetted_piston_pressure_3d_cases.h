#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/liquid_substep_backend_3d.h"
#include "gtest/gtest.h"

namespace fluidity::wetted_piston_pressure_test {

constexpr double kDuration = 0x1p-7;
constexpr double kPistonSpeed = 0x1p-4;

inline LiquidSceneInputs3D PlaneScene(double height, double speed) {
  LiquidSceneInputs3D scene;
  scene.field = LiquidHorizontalPlaneSurface3D{height};
  scene.obstacle = AxisAlignedLiquidTank3D{0, 1, 0, 0, 1, speed};
  scene.world_acceleration = {};
  scene.liquid_density = 1;
  return scene;
}

// Two particles in the selected sphere cell and two above its pressure support.
inline absl::StatusOr<LiquidState3D> PressureSupportGatherState() {
  constexpr std::array positions{
      Vec3{2.5, 2.375, 2.5},
      Vec3{2.5, 2.625, 2.5},
      Vec3{2.5, 4.375, 2.5},
      Vec3{2.5, 4.625, 2.5},
  };
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(positions.size());
  if (!particles.ok()) {
    return particles.status();
  }
  const auto resized = particles->Resize(positions.size());
  if (!resized.ok()) {
    return resized;
  }
  LiquidState3D state;
  state.particles = std::move(*particles);
  state.next_particle_id = positions.size() + 1;
  for (std::size_t ordinal = 0; ordinal < positions.size(); ++ordinal) {
    auto& particle = state.particles[ordinal];
    particle.id = ordinal + 1;
    particle.world_position = positions[ordinal];
    particle.pic_initialized = true;
  }
  return state;
}

// The translating piston is dry. Gravity still distinguishes projected and
// force-only faces around the isolated sphere.
inline LiquidSceneInputs3D PressureSupportGatherScene() {
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{2.5, 2.5, 2.5}, 0.9375};
  scene.obstacle = AxisAlignedLiquidTank3D{-8, 8, 0, -8, 8, kPistonSpeed};
  return scene;
}

inline constexpr AxisAlignedLiquidTank3D kNarrowSphereTank{1.4375, 3.5, 2, 2.46875, 3.03125, 0.125};
inline constexpr double kSphereRadius = 2.125;
inline constexpr AxisAlignedLiquidTank3D kBroadUncertifiedSphereTank{1.75, 3.5, 2, 1.5, 3.5, 0.125};

inline void ExpectSphereWallFlux(const LiquidSubstepScheduler3D& scheduler) {
  const auto& measurement = scheduler.diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement.has_value());
  // At x=1.4375, the sphere centered at (2.5, 2.5, 2.75) cuts a disk.
  // The floor removes its lower arc. Integrate the upper chord over
  // the z interval; the section also extends above the highest selected wall row.
  constexpr double offset_x = 1.0625;
  constexpr double half_width_z = 0.28125;
  constexpr double center_above_floor = 0.5;
  const double disk_radius_squared = kSphereRadius * kSphereRadius - offset_x * offset_x;
  const double chord_integral = half_width_z * std::sqrt(disk_radius_squared - half_width_z * half_width_z)
      + disk_radius_squared * std::asin(half_width_z / std::sqrt(disk_radius_squared));
  const double expected_area = 2 * half_width_z * center_above_floor + chord_integral;
  const double expected_flux = kNarrowSphereTank.piston_velocity_x * expected_area;
  EXPECT_NEAR(measurement->step_start_inward_piston_flux_m3_per_s, expected_flux, expected_flux * 0.0005);
}

inline constexpr double kOverflowPistonSpeed = 0x1.ep1023;
inline constexpr double kOverflowStepDuration = 0x1p-1025;

struct ProjectionAudit {
  LiquidSubstepBackend3D actual;
  std::size_t cell_count = 0;
  std::size_t face_count = 0;
  std::size_t projection_count = 0;
  std::size_t wetted_stage_calls = 0;
  std::size_t piston_face_count = 0;
  std::int32_t piston_face_z = 0;
  std::size_t attached_patch_count = 0;
  int wetted_compiled_architecture = 0;
  double volume = 0;
  double rhs = 0;
  double residual = 0;
  double defect = 0;
  double pressure = 0;
  double top_velocity = 0;
  double top_open_area = 0;
  double top_distance = 0;
  double wall_open_area = 0;
  double wall_speed = 0;
  std::optional<double> wall_flux;
};

inline thread_local ProjectionAudit* active_audit = nullptr;

inline LiquidSubstepBackend3D Observe(ProjectionAudit& audit, LiquidSubstepBackend3D backend) {
  audit.actual = backend;
  active_audit = &audit;
  backend.wetted_piston = [](void* context, const WettedPistonInputs3D& inputs, WettedPistonWorkspace3D& workspace) {
    auto& observed = *active_audit;
    ++observed.wetted_stage_calls;
    observed.piston_face_count = 0;
    for (const auto& face : inputs.frozen.faces) {
      if (face.is_low_x_piston_face) {
        ++observed.piston_face_count;
        observed.piston_face_z = face.face_index.lattice_index.z;
      }
    }
    const auto result = observed.actual.wetted_piston(context, inputs, workspace);
    if (result.ok()) {
      observed.attached_patch_count = result->attached_patch_count;
      observed.wetted_compiled_architecture = result->compiled_architecture;
    }
    return result;
  };
  backend.project = [](void* context,
                       SparseMacGridView3D grid,
                       double duration,
                       const LiquidPressureProblemView3D& problem,
                       MacPressureSolverOptions3D options,
                       MacPressureWorkspace3D& workspace) {
    auto& observed = *active_audit;
    const auto result = observed.actual.project(context, grid, duration, problem, options, workspace);
    if (result.ok() && !problem.cells.empty()) {
      ++observed.projection_count;
      observed.cell_count = problem.cells.size();
      observed.face_count = problem.faces.size();
      observed.volume = problem.cells.front().fluid_volume;
      observed.rhs = result->integrated_rhs.front();
      observed.residual = result->true_residuals.front();
      observed.defect = result->cell_constraint_defects.front();
      observed.pressure = result->cell_pressures.front();
      for (std::size_t i = 0; i < problem.faces.size(); ++i) {
        const auto& face = problem.faces[i];
        if (face.face_index.axis == FaceNormalAxis::kY && face.face_index.lattice_index.y == 1) {
          observed.top_velocity = result->corrected_face_velocities[i];
          observed.top_open_area = face.open_area;
          observed.top_distance = face.pressure_gradient_distance;
        }
        if (face.face_index.axis == FaceNormalAxis::kX && face.face_index.lattice_index.x == 0) {
          observed.wall_open_area = face.open_area;
          observed.wall_speed = face.wall_normal_velocity;
          observed.wall_flux = face.prescribed_wall_flux;
        }
      }
    }
    return result;
  };
  return backend;
}

inline void CheckPlaneProjection(
    const LiquidSubstepScheduler3D& scheduler,
    const ProjectionAudit& audit,
    double height,
    double speed
) {
  // The 1 m wide wall has exactly one selected pressure cell, whose top is
  // the only open face. A height above 1 wets the directly connected upper
  // row without selecting it. The analytic wall area is therefore height.
  const double flux = speed * height;
  const double distance = height - 0.5;
  EXPECT_EQ(audit.wetted_stage_calls, speed == 0 ? 0 : 1);
  EXPECT_EQ(audit.attached_patch_count, speed != 0 && height > 1 ? 1 : 0);
  ASSERT_EQ(audit.projection_count, 1);
  EXPECT_EQ(audit.cell_count, 1);
  EXPECT_EQ(audit.face_count, 6);
  EXPECT_NEAR(audit.volume, std::min(height, 1.0), 1e-14);
  EXPECT_DOUBLE_EQ(audit.wall_open_area, 0);
  EXPECT_DOUBLE_EQ(audit.wall_speed, speed);
  ASSERT_TRUE(audit.wall_flux.has_value());
  EXPECT_NEAR(*audit.wall_flux, flux, 1e-14);
  EXPECT_DOUBLE_EQ(audit.top_open_area, 1);
  EXPECT_DOUBLE_EQ(audit.top_distance, distance);
  EXPECT_NEAR(audit.rhs, flux, 1e-14);
  EXPECT_NEAR(audit.pressure, flux * distance / kDuration, 1e-5);
  EXPECT_NEAR(audit.top_velocity, flux, 2e-6);
  EXPECT_NEAR(audit.defect, (audit.top_velocity - flux) / audit.volume, 1e-14);
  EXPECT_NEAR(audit.defect, -audit.residual / audit.volume, 1e-14);
  const auto& measurement = scheduler.diagnostics().accepted_measurement;
  ASSERT_TRUE(measurement.has_value());
  EXPECT_NEAR(measurement->step_start_inward_piston_flux_m3_per_s, flux, 1e-14);
  EXPECT_NEAR(measurement->pressure_rhs_l2_m3_per_s, std::abs(flux), 1e-14);
  std::cout << "wall_area_m2=" << height << " speed_m_s=" << speed << " expected_flux_m3_s=" << flux
            << " native_flux_m3_s=" << measurement->step_start_inward_piston_flux_m3_per_s
            << " original_rhs_m3_s=" << audit.rhs
            << " independent_defect_s=" << (audit.top_velocity - flux) / audit.volume
            << " native_defect_s=" << audit.defect << '\n';
}

inline void ExpectBroadSphereRefusal(const ProjectionAudit& audit) {
  // The frozen cell clips against the piston and retains the central
  // receiver. The disk extends over half a cell in both z directions,
  // reaching two columns without any receiver.
  EXPECT_EQ(audit.wetted_stage_calls, 1);
  EXPECT_EQ(audit.piston_face_count, 1);
  EXPECT_EQ(audit.piston_face_z, 2);
  EXPECT_EQ(audit.projection_count, 0);

  // Independently bound this single disk against the floor and the z walls.
  // Its radial vertical interval only widens toward the disk's z center, so
  // band endpoints give the extremes without calling the wall-stage code.
  constexpr double radius_squared = 1.1 * 1.1 - 0.75 * 0.75;
  static_assert(radius_squared > 0.25);
  const auto chord_above_floor = [](double z_distance) {
    const double half_chord = std::sqrt(std::max(0.0, radius_squared - z_distance * z_distance));
    return 0.5 + half_chord - std::max(0.0, 0.5 - half_chord);
  };
  constexpr double dz = 1.0 / 64;
  double lower = 0;
  double upper = 0;
  for (int band = 0; band < 128; ++band) {
    const double left = -1 + band * dz;
    const double right = left + dz;
    lower += dz * chord_above_floor(std::max(std::abs(left), std::abs(right)));
    upper += dz * chord_above_floor(std::min(std::abs(left), std::abs(right)));
  }
  ASSERT_GT(lower, 0);
  EXPECT_GT((upper - lower) / lower, 0.01);
  std::cout << "broad_sphere_wall_width_over_lower=" << (upper - lower) / lower
            << " frozen_piston_faces=" << audit.piston_face_count << '\n';
}

inline void ExpectCheckpointEqual(const LiquidState3D& before, const LiquidState3D& after) {
  const auto before_bytes = EncodeLiquidState3D(before);
  const auto after_bytes = EncodeLiquidState3D(after);
  ASSERT_TRUE(before_bytes.ok()) << before_bytes.status();
  ASSERT_TRUE(after_bytes.ok()) << after_bytes.status();
  EXPECT_TRUE(std::ranges::equal(before_bytes->values(), after_bytes->values()));
}

}  // namespace fluidity::wetted_piston_pressure_test
