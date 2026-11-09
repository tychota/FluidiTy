#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "../src/mac_pressure_3d_workspace_internal.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "gtest/gtest.h"

namespace fluidity::rest_pressure_test {
inline constexpr double kTinyPressure = 1e-10;
inline constexpr double kInfinity = std::numeric_limits<double>::infinity();
inline constexpr BlockIndex3D kBlock{};
inline constexpr double kStepDuration = 0.25;

struct RestProblem {
  std::vector<MacPressureCell3D> cells;
  std::vector<MacPressureFace3D> faces;
};

// Two rows with either one internal face or two independent boundary shells.
// Keep one sealed face in both arrangements, with all other exterior faces open.
inline RestProblem MakeRestProblem(const SparseMacGrid3D& grid, double exterior_pressure, bool disconnected) {
  RestProblem problem;
  const double width = grid.view().geometry().cell_width;
  const double area = width * width;
  const int last_x = disconnected ? 2 : 1;
  for (const int x : {0, last_x}) {
    problem.cells.push_back({
        .cell_index = {x, 0, 0},
        .fluid_volume = area * width,
        .bulk_modulus = kInfinity,
    });
    for (int normal = 0; normal < 3; ++normal) {
      for (int side = 0; side < 2; ++side) {
        if (!disconnected && x == 1 && normal == 0 && side == 0) {
          continue;
        }
        std::array coordinates{x, 0, 0};
        coordinates[normal] += side;
        MacPressureFace3D face{
            .face_index = {static_cast<FaceNormalAxis>(normal), {coordinates[0], coordinates[1], coordinates[2]}},
            .open_area = area,
            .inverse_density = 1,
            .pressure_gradient_distance = width,
            .pressure_link = DirichletMacPressureLink3D{exterior_pressure},
        };
        if (!disconnected && x == 0 && normal == 0 && side == 1) {
          face.pressure_link = InternalMacPressureLink3D{};
        }
        if (x == 0 && normal == 1 && side == 0) {
          face.open_area = 0;
          face.pressure_link = SealedMacPressureLink3D{};
        }
        problem.faces.push_back(face);
      }
    }
  }
  std::ranges::sort(problem.faces, [&](const auto& left, const auto& right) {
    return *grid.FaceSlot(left.face_index) < *grid.FaceSlot(right.face_index);
  });
  return problem;
}

template <typename Link>
MacPressureFace3D& FaceWithLink(RestProblem& problem) {
  return *std::ranges::find_if(problem.faces, [](const auto& face) {
    return std::holds_alternative<Link>(face.pressure_link);
  });
}

inline void ExpectUnchangedPreparation(
    const SparseMacGrid3D& grid,
    const RestProblem& problem,
    std::span<const double> expected_default,
    std::span<const double> expected_seed
) {
  const auto bytes = QueryMacPressureWorkspaceBytes3D(problem.cells.size(), problem.faces.size());
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  const std::size_t alignment_units = bytes->scratch_bytes / sizeof(std::max_align_t);
  std::vector<std::max_align_t> default_storage(alignment_units);
  std::vector<std::max_align_t> opted_in_storage(alignment_units);
  pressure_internal::MacPressureScratch3D default_scratch(default_storage);
  pressure_internal::MacPressureScratch3D opted_in_scratch(opted_in_storage);
  const auto ordinary = pressure_internal::PrepareBoundedMacPressureUpload3D(
      grid.view(),
      kStepDuration,
      problem.cells,
      problem.faces,
      default_scratch
  );
  const auto opted_in = pressure_internal::PrepareBoundedMacPressureUpload3D(
      grid.view(),
      kStepDuration,
      problem.cells,
      problem.faces,
      opted_in_scratch,
      true
  );
  ASSERT_TRUE(ordinary.ok()) << ordinary.status();
  ASSERT_TRUE(opted_in.ok()) << opted_in.status();
  EXPECT_TRUE(std::ranges::equal(ordinary->initial_pressure, expected_default));
  EXPECT_TRUE(std::ranges::equal(opted_in->initial_pressure, expected_seed));
  EXPECT_TRUE(std::ranges::equal(ordinary->system.integrated_rhs(), opted_in->system.integrated_rhs()));
  EXPECT_TRUE(std::ranges::equal(ordinary->system.rhs_absolute_term_sums(), opted_in->system.rhs_absolute_term_sums()));
  EXPECT_TRUE(std::ranges::equal(ordinary->system.gauge_cell_ordinals(), opted_in->system.gauge_cell_ordinals()));
  EXPECT_EQ(ordinary->stopping.integrated_residual_limit, opted_in->stopping.integrated_residual_limit);
  EXPECT_EQ(default_scratch.used(), opted_in_scratch.used());
  EXPECT_EQ(default_scratch.high_water(), opted_in_scratch.high_water());
  for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
    EXPECT_EQ(ordinary->system.cells()[cell].previous_pressure, problem.cells[cell].previous_pressure);
    EXPECT_EQ(opted_in->system.cells()[cell].previous_pressure, problem.cells[cell].previous_pressure);
  }
}

template <typename Project>
void ExpectExactAnchoredRest(Project project) {
  struct Case {
    double width;
    double exterior_pressure;
    bool disconnected;
  };
  for (const auto& test :
       {Case{1, 16, false}, Case{0.5, 16, false}, Case{0.5, -16, true}, Case{1, kTinyPressure, false}}) {
    SCOPED_TRACE(test.width);
    SCOPED_TRACE(test.exterior_pressure);
    SCOPED_TRACE(test.disconnected);
    const auto grid = SparseMacGrid3D::Create({{}, test.width}, {&kBlock, 1}, 1);
    ASSERT_TRUE(grid.ok()) << grid.status();
    const auto problem = MakeRestProblem(*grid, test.exterior_pressure, test.disconnected);
    LiquidPressureProblemView3D inputs{problem.cells, problem.faces};
    if (test.exterior_pressure == kTinyPressure) {
      const auto ordinary = project(grid->view(), inputs);
      ASSERT_TRUE(ordinary.ok()) << ordinary.status();
      EXPECT_EQ(ordinary->diagnostics.iteration_count, 0);
      for (const double pressure : ordinary->cell_pressures) {
        EXPECT_EQ(pressure, 0);
      }
    }
    inputs.seed_uniform_dirichlet_rest_pressure = true;
    const auto projection = project(grid->view(), inputs);
    ASSERT_TRUE(projection.ok()) << projection.status();
    ASSERT_EQ(projection->cell_pressures.size(), problem.cells.size());
    ASSERT_EQ(projection->corrected_face_velocities.size(), problem.faces.size());
    EXPECT_TRUE(projection->gauge_cell_ordinals.empty());
    EXPECT_EQ(projection->diagnostics.iteration_count, 0);
    EXPECT_EQ(projection->diagnostics.true_residual_l2, 0);
    EXPECT_EQ(projection->diagnostics.max_volume_scaled_residual, 0);
    EXPECT_EQ(projection->diagnostics.max_cell_constraint_defect, 0);
    for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
      EXPECT_EQ(projection->cell_pressures[cell], test.exterior_pressure);
      EXPECT_EQ(projection->true_residuals[cell], 0);
      EXPECT_EQ(projection->cell_constraint_defects[cell], 0);
    }
    for (const double velocity : projection->corrected_face_velocities) {
      EXPECT_EQ(velocity, 0);
    }
    ASSERT_NO_FATAL_FAILURE(ExpectUnchangedPreparation(
        *grid,
        problem,
        std::array{0.0, 0.0},
        std::array{test.exterior_pressure, test.exterior_pressure}
    ));
  }
}

template <typename Project>
void ExpectOrdinaryNonRest(Project project) {
  using Boundary = DirichletMacPressureLink3D;
  using Internal = InternalMacPressureLink3D;
  using Sealed = SealedMacPressureLink3D;
  struct Control {
    const char* name;
    void (*change)(RestProblem&);
    bool disconnected = false;
    std::array<double, 2> warm_start{};
    std::size_t gauges = 0;
    double dormant_velocity = 0;
  };
  const std::array controls{
      Control{
          "unequal exterior pressures",
          [](RestProblem& p) {
            FaceWithLink<Boundary>(p).pressure_link = Boundary{2 * kTinyPressure};
          }
      },
      Control{
          "target flux",
          [](RestProblem& p) {
            p.cells.front().target_divergence = kTinyPressure;
          }
      },
      Control{
          "predicted velocity",
          [](RestProblem& p) {
            FaceWithLink<Boundary>(p).predicted_normal_velocity = kTinyPressure;
          }
      },
      Control{
          "implicit wall flux",
          [](RestProblem& p) {
            FaceWithLink<Sealed>(p).wall_normal_velocity = kTinyPressure;
          }
      },
      Control{
          "two-ended explicit wall flux",
          [](RestProblem& p) {
            FaceWithLink<Internal>(p).open_area = 0.5;
            FaceWithLink<Internal>(p).prescribed_wall_flux = kTinyPressure;
          }
      },
      Control{
          "internal jump",
          [](RestProblem& p) {
            FaceWithLink<Internal>(p).pressure_link = Internal{kTinyPressure};
          }
      },
      Control{
          "compliance",
          [](RestProblem& p) {
            p.cells.front().bulk_modulus = 4;
          }
      },
      Control{
          "previous pressure",
          [](RestProblem& p) {
            p.cells.front().previous_pressure = 2 * kTinyPressure;
          },
          false,
          {2 * kTinyPressure, 0}
      },
      Control{
          "compliant history",
          [](RestProblem& p) {
            p.cells.front().bulk_modulus = 4;
            p.cells.front().previous_pressure = 2 * kTinyPressure;
          },
          false,
          {2 * kTinyPressure, 0}
      },
      Control{
          "disconnected gauge",
          [](RestProblem& p) {
            for (auto& face : p.faces) {
              if (face.face_index.lattice_index.x >= 2) {
                face.open_area = 0;
                face.pressure_link = Sealed{};
              }
            }
          },
          true,
          {},
          1
      },
      Control{
          "no Dirichlet links",
          [](RestProblem& p) {
            for (auto& face : p.faces) {
              if (std::holds_alternative<Boundary>(face.pressure_link)) {
                face.open_area = 0;
                face.pressure_link = Sealed{};
              }
            }
          },
          false,
          {},
          1
      },
      Control{
          "dormant closed velocity",
          [](RestProblem& p) {
            FaceWithLink<Sealed>(p).predicted_normal_velocity = 3;
          },
          false,
          {},
          0,
          3
      },
  };
  const auto grid = SparseMacGrid3D::Create({{}, 1}, {&kBlock, 1}, 1);
  ASSERT_TRUE(grid.ok()) << grid.status();
  for (const auto& control : controls) {
    SCOPED_TRACE(control.name);
    auto problem = MakeRestProblem(*grid, kTinyPressure, control.disconnected);
    control.change(problem);
    for (bool opted_in : {false, true}) {
      SCOPED_TRACE(opted_in);
      const auto projection = project(grid->view(), {problem.cells, problem.faces, 0, opted_in});
      ASSERT_TRUE(projection.ok()) << projection.status();
      EXPECT_EQ(projection->diagnostics.iteration_count, 0);
      EXPECT_EQ(projection->gauge_cell_ordinals.size(), control.gauges);
      EXPECT_TRUE(std::ranges::equal(projection->cell_pressures, control.warm_start));
      if (control.dormant_velocity != 0) {
        const std::size_t face = static_cast<std::size_t>(&FaceWithLink<Sealed>(problem) - problem.faces.data());
        EXPECT_EQ(projection->corrected_face_velocities[face], control.dormant_velocity);
      }
    }
    ASSERT_NO_FATAL_FAILURE(ExpectUnchangedPreparation(*grid, problem, control.warm_start, control.warm_start));
  }
}
inline void ExpectUnrepresentableIdentityBoundDeclinesSeed() {
  const auto grid = SparseMacGrid3D::Create({{}, 1}, {&kBlock, 1}, 1);
  ASSERT_TRUE(grid.ok()) << grid.status();
  // Each boundary product and each prepared row magnitude fits, as does the
  // RHS norm. The original identity check would sum two copies of a row
  // magnitude greater than half of the FP64 limit.
  const auto problem = MakeRestProblem(*grid, std::numeric_limits<double>::max() / 2, false);
  ASSERT_NO_FATAL_FAILURE(ExpectUnchangedPreparation(*grid, problem, std::array{0.0, 0.0}, std::array{0.0, 0.0}));
}
}  // namespace fluidity::rest_pressure_test
