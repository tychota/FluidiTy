#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include "fluidity/host/liquid_substep_backend_3d.h"
#include "gtest/gtest.h"

namespace fluidity::exact_curvature_test {

inline constexpr LiquidStateCapacities3D kStaticSphereCapacities{1024, 128, 4096, 24576, 256};

struct ProjectionAudit {
  LiquidSubstepBackend3D actual;
  std::size_t projections = 0;
  std::size_t air_links = 0;
  std::size_t incorrect_exterior_pressures = 0;
  double maximum_rhs_error = 0;
  double maximum_correction_error = 0;
};

inline thread_local ProjectionAudit* active_audit = nullptr;

inline void ExpectAnalyticSphereExteriorPressure(LiquidSubstepBackend3D backend) {
  struct Resolution {
    int cells_per_radius;
    std::size_t air_links;
  };
  constexpr double kRadius = 0.125;
  constexpr double kExteriorPressure = 16;
  constexpr double kDensity = 1000;
  constexpr double kDuration = 1.0 / 120;
  LiquidSceneInputs3D scene;
  scene.field = LiquidAnalyticSphere3D{{}, kRadius};
  scene.surface_tension = 1;
  scene.world_acceleration = {};
  LiquidStepOptions3D options;
  options.capacities = kStaticSphereCapacities;

  ProjectionAudit audit{.actual = backend};
  backend.project = [](void* context,
                       SparseMacGridView3D grid,
                       double duration,
                       const LiquidPressureProblemView3D& problem,
                       MacPressureSolverOptions3D solver_options,
                       MacPressureWorkspace3D& workspace) {
    auto& observed = *active_audit;
    const auto projection = observed.actual.project(context, grid, duration, problem, solver_options, workspace);
    if (!projection.ok()) {
      return projection;
    }
    ++observed.projections;
    std::vector<double> expected_rhs(problem.cells.size());
    for (std::size_t face_ordinal = 0; face_ordinal < problem.faces.size(); ++face_ordinal) {
      const auto& face = problem.faces[face_ordinal];
      const auto* air = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link);
      if (air == nullptr) {
        continue;
      }
      ++observed.air_links;
      if (air->exterior_pressure != kExteriorPressure) {
        ++observed.incorrect_exterior_pressures;
      }
      const auto lattice = face.face_index.lattice_index;
      const CellIndex3D head{lattice.x, lattice.y, lattice.z};
      const CellIndex3D tail{
          head.x - (face.face_index.axis == FaceNormalAxis::kX),
          head.y - (face.face_index.axis == FaceNormalAxis::kY),
          head.z - (face.face_index.axis == FaceNormalAxis::kZ),
      };
      for (std::size_t cell_ordinal = 0; cell_ordinal < problem.cells.size(); ++cell_ordinal) {
        const auto cell_index = problem.cells[cell_ordinal].cell_index;
        const auto same_cell = [&](CellIndex3D endpoint) {
          return cell_index.x == endpoint.x && cell_index.y == endpoint.y && cell_index.z == endpoint.z;
        };
        const bool liquid_at_head = same_cell(head);
        if (!liquid_at_head && !same_cell(tail)) {
          continue;
        }
        const double correction_scale = duration / (kDensity * face.pressure_gradient_distance);
        expected_rhs[cell_ordinal] += face.open_area * correction_scale * kExteriorPressure;
        const double liquid_minus_air = projection->cell_pressures[cell_ordinal] - kExteriorPressure;
        const double expected_velocity = (liquid_at_head ? -1 : 1) * correction_scale * liquid_minus_air;
        observed.maximum_correction_error = std::max(
            observed.maximum_correction_error,
            std::abs(projection->corrected_face_velocities[face_ordinal] - expected_velocity)
        );
        break;
      }
    }
    for (std::size_t i = 0; i < expected_rhs.size(); ++i) {
      observed.maximum_rhs_error =
          std::max(observed.maximum_rhs_error, std::abs(projection->integrated_rhs[i] - expected_rhs[i]));
    }
    return projection;
  };

  for (const auto resolution : {Resolution{3, 192}, Resolution{5, 480}}) {
    const int extent = resolution.cells_per_radius;
    SCOPED_TRACE(extent);
    LiquidState3D initial;
    initial.geometry.cell_width = kRadius / extent;
    std::vector<LiquidParticleState3D> particles;
    for (int z = -extent; z < extent; ++z) {
      for (int y = -extent; y < extent; ++y) {
        for (int x = -extent; x < extent; ++x) {
          const double h = initial.geometry.cell_width;
          const Vec3 position{(x + 0.5) * h, (y + 0.5) * h, (z + 0.5) * h};
          if (std::hypot(position.x, position.y, position.z) < kRadius) {
            particles.push_back({.id = particles.size() + 1, .world_position = position});
          }
        }
      }
    }
    const double particle_volume = (4 * std::numbers::pi * kRadius * kRadius * kRadius / 3) / particles.size();
    for (auto& particle : particles) {
      particle.mass = kDensity * particle_volume;
      particle.reference_volume = particle_volume;
    }
    auto storage = LiquidRecords3D<LiquidParticleState3D>::Create(particles.size());
    ASSERT_TRUE(storage.ok()) << storage.status();
    ASSERT_TRUE(storage->Resize(particles.size()).ok());
    std::ranges::copy(particles, storage->values().begin());
    initial.particles = std::move(*storage);
    initial.next_particle_id = particles.size() + 1;

    audit = {.actual = audit.actual};
    auto scheduler = LiquidSubstepScheduler3D::Create(initial, scene, options, backend);
    ASSERT_TRUE(scheduler.ok()) << scheduler.status();
    active_audit = &audit;
    const auto status = scheduler->AdvanceFrame(kDuration);
    active_audit = nullptr;
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ(audit.projections, 1);
    EXPECT_EQ(audit.air_links, resolution.air_links);
    EXPECT_EQ(audit.incorrect_exterior_pressures, 0);
    EXPECT_LE(audit.maximum_rhs_error, 1e-16);
    EXPECT_LE(audit.maximum_correction_error, 1e-15);
  }
}

template <class Freeze, class Links>
void ExpectCheckedFreeze(Freeze freeze, Links build_links) {
  using Limits = std::numeric_limits<double>;
  const auto grid = SparseMacGrid3D::Create({{}, 1}, std::array{BlockIndex3D{0, 0, 0}}, 1);
  ASSERT_TRUE(grid.ok()) << grid.status();
  const std::array cells{CellIndex3D{1, 1, 1}};
  const std::array<LiquidCenterDistance3D, 7> centers{{
      {{1, 1, 0}, 0.5},
      {{1, 0, 1}, 0.5},
      {{0, 1, 1}, 0.5},
      {{1, 1, 1}, -0.5},
      {{2, 1, 1}, 0.5},
      {{1, 2, 1}, 0.5},
      {{1, 1, 2}, 0.5},
  }};
  std::vector<LiquidCornerDistance3D> corners;
  for (int z = 1; z <= 2; ++z) {
    for (int y = 1; y <= 2; ++y) {
      for (int x = 1; x <= 2; ++x) {
        corners.push_back({{x, y, z}, -1});
      }
    }
  }
  const LiquidBoundaryGeometryViewInputs3D inputs{
      .mac_grid = grid->view(),
      .cell_indices = cells,
      .center_distances = centers,
      .corner_distances = corners,
      .step_duration = 0.01,
      .liquid_density = 1000,
      .surface_tension = 1,
      .exact_outward_interface_curvature = -16,
  };
  const auto frozen = freeze(inputs);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells.size(), 1);
  ASSERT_EQ(frozen->faces.size(), 6);
  EXPECT_DOUBLE_EQ(frozen->cells.front().curvature, -16);
  EXPECT_DOUBLE_EQ(frozen->cells.front().fluid_volume, 1);
  EXPECT_NEAR(frozen->capillary_step_bound, std::sqrt(1000 / (4 * std::numbers::pi)), 1e-13);

  const std::array fields{LiquidPressureCellFields3D{0, Limits::infinity(), 0}};
  const std::array<double, 6> velocities{};
  const auto links = build_links(*frozen, LiquidPressureFields3D{fields, velocities, 3});
  ASSERT_TRUE(links.ok()) << links.status();
  for (const auto& face : links->faces) {
    EXPECT_DOUBLE_EQ(face.open_area, 1);
    EXPECT_DOUBLE_EQ(face.pressure_gradient_distance, 0.5);
    const auto* air = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link);
    ASSERT_NE(air, nullptr);
    EXPECT_DOUBLE_EQ(air->exterior_pressure, -13);
  }

  auto trial = inputs;
  trial.exact_outward_interface_curvature.reset();
  EXPECT_EQ(freeze(trial).status().code(), absl::StatusCode::kFailedPrecondition);
  trial = inputs;
  trial.center_distances = std::span(centers).first(6);
  EXPECT_EQ(freeze(trial).status().code(), absl::StatusCode::kFailedPrecondition);
  trial = inputs;
  trial.corner_distances = std::span(corners).first(7);
  EXPECT_EQ(freeze(trial).status().code(), absl::StatusCode::kFailedPrecondition);
  for (const auto nonfinite : {Limits::quiet_NaN(), Limits::infinity(), -Limits::infinity()}) {
    trial = inputs;
    trial.exact_outward_interface_curvature = nonfinite;
    EXPECT_EQ(freeze(trial).status().code(), absl::StatusCode::kInvalidArgument);
  }

  struct UnrepresentableLift {
    double surface_tension;
    double curvature;
  };
  for (const auto invalid :
       {UnrepresentableLift{2, Limits::max()},
        UnrepresentableLift{0.5, Limits::min()},
        UnrepresentableLift{0.25, Limits::denorm_min()}}) {
    trial = inputs;
    trial.surface_tension = invalid.surface_tension;
    trial.exact_outward_interface_curvature = invalid.curvature;
    EXPECT_EQ(freeze(trial).status().code(), absl::StatusCode::kOutOfRange);
    EXPECT_DOUBLE_EQ(frozen->cells.front().curvature, -16);
    EXPECT_DOUBLE_EQ(frozen->cells.front().fluid_volume, 1);
    EXPECT_DOUBLE_EQ(frozen->faces.front().open_area, 1);
  }

  trial = inputs;
  trial.exact_outward_interface_curvature = 0;
  const auto zero_curvature = freeze(trial);
  ASSERT_TRUE(zero_curvature.ok()) << zero_curvature.status();
  EXPECT_DOUBLE_EQ(zero_curvature->cells.front().curvature, 0);
  EXPECT_TRUE(std::isfinite(zero_curvature->capillary_step_bound));

  trial.surface_tension = 0;
  trial.exact_outward_interface_curvature = Limits::max();
  const auto zero_sigma = freeze(trial);
  ASSERT_TRUE(zero_sigma.ok()) << zero_sigma.status();
  EXPECT_DOUBLE_EQ(zero_sigma->cells.front().curvature, 0);
  EXPECT_EQ(zero_sigma->capillary_step_bound, Limits::infinity());
  const auto zero_sigma_links = build_links(*zero_sigma, LiquidPressureFields3D{fields, velocities, 3});
  ASSERT_TRUE(zero_sigma_links.ok()) << zero_sigma_links.status();
  for (const auto& face : zero_sigma_links->faces) {
    EXPECT_DOUBLE_EQ(std::get<DirichletMacPressureLink3D>(face.pressure_link).exterior_pressure, 3);
  }

  auto interior_centers = centers;
  for (auto& sample : interior_centers) {
    sample.signed_distance = -1;
  }
  trial.center_distances = interior_centers;
  trial.surface_tension = 2;
  const auto interior = freeze(trial);
  ASSERT_TRUE(interior.ok()) << interior.status();
  EXPECT_DOUBLE_EQ(interior->cells.front().curvature, 0);
  EXPECT_EQ(interior->capillary_step_bound, Limits::infinity());
}

}  // namespace fluidity::exact_curvature_test
