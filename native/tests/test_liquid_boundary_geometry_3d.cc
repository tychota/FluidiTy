#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "fixture_data.h"
#include "fluidity/host/liquid_boundary_geometry_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {
constexpr double kInfinity = std::numeric_limits<double>::infinity();
using Field = std::function<double(double, double, double)>;
struct Samples {
  std::vector<CellIndex3D> candidates{{-2, -2, -2}};
  std::vector<LiquidCenterDistance3D> centers;
  std::vector<LiquidCornerDistance3D> corners;
};
Samples Sample(const Field& field) {
  Samples samples;
  for (int z = -4; z <= 1; ++z) {
    for (int y = -4; y <= 1; ++y) {
      for (int x = -4; x <= 1; ++x) {
        samples.centers.push_back({{x, y, z}, field(x + 2.5, y + 2.5, z + 2.5)});
        samples.corners.push_back({{x, y, z}, field(x + 2.0, y + 2.0, z + 2.0)});
      }
    }
  }
  return samples;
}
SparseMacGrid3D Grid(double h = 1.0) {
  std::vector<BlockIndex3D> blocks;
  for (int z = -1; z <= 0; ++z) {
    for (int y = -1; y <= 0; ++y) {
      for (int x = -1; x <= 0; ++x) {
        blocks.push_back({x, y, z});
      }
    }
  }
  return *SparseMacGrid3D::Create({{2 * h, 2 * h, 2 * h}, h}, blocks, blocks.size());
}
auto Freeze(
    const SparseMacGrid3D& grid,
    const Samples& samples,
    LiquidObstacle3D obstacle = {},
    double sigma = 0.0,
    double dt = 0.01,
    double rho = 1000.0
) {
  return FreezeLiquidBoundaryGeometry3D(
      {grid, samples.candidates, samples.centers, samples.corners, obstacle, dt, rho, sigma}
  );
}
LiquidPressureProblem3D Links(const FrozenLiquidBoundaryGeometry3D& frozen, double ambient = 0.0) {
  std::vector<LiquidPressureCellFields3D> cells(frozen.cells().size(), {0.0, kInfinity, 0.0});
  std::vector<double> velocities(frozen.faces().size(), 0.0);
  auto result = BuildLiquidPressureLinks3D(frozen, {cells, velocities, ambient});
  EXPECT_TRUE(result.ok()) << result.status();
  return result.ok() ? *result : LiquidPressureProblem3D{};
}

TEST(LiquidBoundaryGeometry3D, AnalyticPlaneFixtureAllAxesSignedIndicesAndTinyOpening) {
  const auto all = fixtures::AllFixtures();
  const auto fixture = std::ranges::find(all, "uniform_mac_liquid_boundary_3d", &fixtures::FixtureRecord::name);
  ASSERT_NE(fixture, all.end());
  const auto offsets = fixtures::FindArray(*fixture, "unit_cell_plane_offsets");
  const auto areas = fixtures::FindArray(*fixture, "expected_open_face_areas");
  const auto volumes = fixtures::FindArray(*fixture, "expected_open_cell_volumes");
  ASSERT_TRUE(offsets.ok());
  ASSERT_TRUE(areas.ok());
  ASSERT_TRUE(volumes.ok());
  const auto grid = Grid();
  const auto samples = Sample([](double, double, double) {
    return -1.0;
  });
  double max_error = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    Vec3 normal{axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0, axis == 2 ? 1.0 : 0.0};
    for (std::size_t row = 0; row < (*offsets)->f64_data.size(); ++row) {
      const auto frozen = Freeze(grid, samples, TranslatingHalfspace3D{normal, (*offsets)->f64_data[row], {2, 3, 4}});
      ASSERT_TRUE(frozen.ok()) << frozen.status();
      ASSERT_EQ(frozen->cells().size(), 1);
      max_error = std::max(max_error, std::abs(frozen->cells()[0].fluid_volume - (*volumes)->f64_data[row]));
      EXPECT_NEAR(frozen->cells()[0].fluid_volume, (*volumes)->f64_data[row], 2e-14);
      ASSERT_EQ(frozen->faces().size(), 6);
      for (const auto& face : frozen->faces()) {
        EXPECT_DOUBLE_EQ(face.blocked_area, 1.0 - face.open_area);
        EXPECT_DOUBLE_EQ(face.wall_normal_velocity, 2.0 + static_cast<int>(face.face_index.axis));
        if (static_cast<int>(face.face_index.axis) != axis) {
          max_error = std::max(max_error, std::abs(face.open_area - (*areas)->f64_data[row]));
          EXPECT_NEAR(face.open_area, (*areas)->f64_data[row], 2e-14);
        }
      }
    }
  }
  std::cout << std::setprecision(17) << "analytic_max_area_volume_error=" << max_error << '\n';
}

TEST(LiquidBoundaryGeometry3D, ObliqueAndSimultaneousLiquidSolidClips) {
  const auto grid = Grid();
  auto samples = Sample([](double x, double y, double z) {
    return x + y + z - 1.0;
  });
  // Explicit center samples are a physical reconstruction independent of corners.
  for (auto& sample : samples.centers) {
    sample.signed_distance = -1.0;
  }
  const auto frozen = Freeze(grid, samples, TranslatingHalfspace3D{{1, 0, 0}, 0.25, {0, 0, 0}});
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  EXPECT_NEAR(frozen->cells()[0].fluid_volume, 9.0 / 128.0, 2e-14);
  samples = Sample([](double, double, double) {
    return -1.0;
  });
  const double q = 1.0 / std::sqrt(3.0);
  const auto oblique = Freeze(grid, samples, TranslatingHalfspace3D{{q, q, q}, q, {0, 0, 0}});
  ASSERT_TRUE(oblique.ok()) << oblique.status();
  EXPECT_NEAR(oblique->cells()[0].fluid_volume, 5.0 / 6.0, 2e-14);
  for (const auto& face : oblique->faces()) {
    const auto i = face.face_index.lattice_index;
    const int coordinate = face.face_index.axis == FaceNormalAxis::kX ? i.x
        : face.face_index.axis == FaceNormalAxis::kY                  ? i.y
                                                                      : i.z;
    EXPECT_NEAR(face.open_area, coordinate == -2 ? 0.5 : 1.0, 2e-14);
  }
}

TEST(LiquidBoundaryGeometry3D, ThinSlabInvisibleToCornersAndSplitCellRefusal) {
  const auto grid = Grid();
  const TranslatingSlab3D slab{{1, 0, 0}, 0.5, 1.0 / 32.0, {0, 0, 0}};
  auto samples = Sample([](double x, double, double) {
    return x - 0.52;
  });
  const auto frozen = Freeze(grid, samples, slab);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  EXPECT_NEAR(frozen->cells()[0].fluid_volume, 15.0 / 32.0, 2e-14);
  for (const auto& face : frozen->faces()) {
    if (face.face_index.axis != FaceNormalAxis::kX) {
      EXPECT_NEAR(face.blocked_area, 1.0 / 16.0, 2e-14);
    }
  }
  samples = Sample([](double, double, double) {
    return -1.0;
  });
  const auto split = Freeze(grid, samples, slab);
  ASSERT_FALSE(split.ok());
  EXPECT_EQ(split.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(LiquidBoundaryGeometry3D, SharedFaceHasOneOwnerAndMatchingTriangulation) {
  const auto grid = Grid();
  auto samples = Sample([](double, double, double) {
    return -1.0;
  });
  samples.candidates.push_back({-1, -2, -2});
  const double q = 1.0 / std::sqrt(2.0);
  const auto frozen = Freeze(grid, samples, TranslatingHalfspace3D{{0, q, q}, q, {0, 0, 0}});
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 2);
  ASSERT_EQ(frozen->faces().size(), 11);
  EXPECT_NEAR(frozen->cells()[0].fluid_volume, 0.5, 2e-14);
  EXPECT_NEAR(frozen->cells()[1].fluid_volume, 0.5, 2e-14);
  std::size_t previous = 0;
  for (std::size_t i = 0; i < frozen->faces().size(); ++i) {
    const auto& face = frozen->faces()[i];
    if (i > 0) {
      EXPECT_GT(face.face_slot, previous);
    }
    previous = face.face_slot;
    if (face.face_index.axis == FaceNormalAxis::kX) {
      EXPECT_NEAR(face.open_area, 0.5, 2e-14);
    }
  }
}

Samples IsolatedLiquid(double center = -0.01) {
  auto samples = Sample([](double, double, double) {
    return 0.99;
  });
  for (auto& sample : samples.corners) {
    sample.signed_distance = -1.0;
  }
  for (auto& sample : samples.centers) {
    const auto i = sample.cell_index;
    if (i.x == -2 && i.y == -2 && i.z == -2) {
      sample.signed_distance = center;
    }
  }
  return samples;
}
TEST(LiquidBoundaryGeometry3D, ThetaFixtureAndSignedPressureMapping) {
  const auto grid = Grid();
  for (const double center : {-0.01, -0.25}) {
    auto samples = IsolatedLiquid(center);
    for (auto& sample : samples.centers) {
      if (sample.signed_distance > 0) {
        sample.signed_distance = center + 1.0;
      }
    }
    const auto frozen = Freeze(grid, samples);
    ASSERT_TRUE(frozen.ok()) << frozen.status();
    const auto links = Links(*frozen, 7.0);
    ASSERT_EQ(links.faces.size(), 6);
    EXPECT_EQ(links.lower_theta_regularization_count, center == -0.01 ? 6 : 0);
    for (const auto& face : links.faces) {
      EXPECT_DOUBLE_EQ(face.pressure_gradient_distance, center == -0.01 ? 0.05 : 0.25);
      EXPECT_DOUBLE_EQ(std::get<DirichletMacPressureLink3D>(face.pressure_link).exterior_pressure, 7.0);
    }
    const auto projection = ProjectMacPressure3D({grid, 0.01, links.cells, links.faces});
    ASSERT_TRUE(projection.ok()) << projection.status();
    EXPECT_NEAR(projection->cell_pressures[0], 7.0, 1e-6);
    for (double velocity : projection->corrected_face_velocities) {
      EXPECT_NEAR(velocity, 0.0, 1e-9);
    }
  }
}
TEST(LiquidBoundaryGeometry3D, PlaneAndSphereCenteredCurvature) {
  const auto grid = Grid();
  const auto plane = Freeze(
      grid,
      Sample([](double x, double, double) {
        return x - 0.75;
      }),
      {},
      0.072
  );
  ASSERT_TRUE(plane.ok()) << plane.status();
  EXPECT_NEAR(plane->cells()[0].curvature, 0.0, 1e-14);
  double max_error = 0.0;
  for (double radius : {8.0, 16.0, 32.0}) {
    for (const Vec3 direction :
         {Vec3{1, 0, 0}, Vec3{0.6, 0.8, 0}, Vec3{1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)}}) {
      auto samples = Sample([&](double x, double y, double z) {
        return std::hypot(
                   x - 0.5 + radius * direction.x,
                   y - 0.5 + radius * direction.y,
                   z - 0.5 + radius * direction.z
               )
            - radius - 0.01;
      });
      const auto frozen = Freeze(grid, samples, {}, 0.072);
      ASSERT_TRUE(frozen.ok()) << frozen.status();
      const double error = std::abs(frozen->cells()[0].curvature / (2.0 / radius) - 1.0);
      max_error = std::max(max_error, error);
      EXPECT_LT(error, 0.05);
    }
  }
  std::cout << "sphere_max_relative_curvature_error=" << max_error << '\n';
}
TEST(LiquidBoundaryGeometry3D, CapillaryLiftOnceAndCorrectionMismatchControl) {
  const auto grid = Grid();
  auto samples = IsolatedLiquid();
  for (auto& sample : samples.centers) {
    if (sample.cell_index.x == -1 && sample.cell_index.y == -2 && sample.cell_index.z == -2) {
      sample.signed_distance = 1.99;
    }
  }
  const auto frozen = Freeze(grid, samples, {}, 0.072);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  const auto links = Links(*frozen, 3.0);
  ASSERT_EQ(links.faces.size(), 6);
  // g=(0.5,0,0), H=diag(3,2,2), so kappa=(Hyy+Hzz)/|g|=8.
  for (const auto& face : links.faces) {
    EXPECT_NEAR(std::get<DirichletMacPressureLink3D>(face.pressure_link).exterior_pressure, 3.576, 1e-14);
  }
  const auto projection = ProjectMacPressure3D({grid, 0.01, links.cells, links.faces});
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_NEAR(projection->cell_pressures[0], 3.576, 1e-6);
  for (double velocity : projection->corrected_face_velocities) {
    EXPECT_NEAR(velocity, 0.0, 1e-9);
  }
  auto driven = links;
  driven.faces[0].predicted_normal_velocity = 0.2;
  const auto projected = ProjectMacPressure3D({grid, 0.01, driven.cells, driven.faces});
  ASSERT_TRUE(projected.ok()) << projected.status();
  double mismatched_flux = 0.0;
  for (std::size_t i = 0; i < driven.faces.size(); ++i) {
    const auto& face = driven.faces[i];
    const auto index = face.face_index.lattice_index;
    const int normal_index = face.face_index.axis == FaceNormalAxis::kX ? index.x
        : face.face_index.axis == FaceNormalAxis::kY                    ? index.y
                                                                        : index.z;
    const double bad_velocity = face.predicted_normal_velocity
        + 1.1 * (projected->corrected_face_velocities[i] - face.predicted_normal_velocity);
    mismatched_flux += (normal_index == -2 ? -1.0 : 1.0) * face.open_area * bad_velocity;
  }
  EXPECT_GT(std::abs(mismatched_flux), 0.019);
  EXPECT_LT(projected->diagnostics.max_cell_constraint_defect, 2e-5);
}
TEST(LiquidBoundaryGeometry3D, RejectsMissingSamplesOwnersOrderingAndInvalidObstacle) {
  const auto grid = Grid();
  auto samples = IsolatedLiquid();
  samples.corners.clear();
  EXPECT_EQ(Freeze(grid, samples).status().code(), absl::StatusCode::kFailedPrecondition);
  samples = IsolatedLiquid();
  samples.centers.push_back(samples.centers[0]);
  EXPECT_EQ(Freeze(grid, samples).status().code(), absl::StatusCode::kInvalidArgument);
  samples = IsolatedLiquid();
  EXPECT_EQ(
      Freeze(grid, samples, TranslatingHalfspace3D{{2, 0, 0}, 0, {0, 0, 0}}).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  const auto empty = SparseMacGrid3D::Create(grid.geometry(), {}, 0);
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(Freeze(*empty, samples).status().code(), absl::StatusCode::kFailedPrecondition);
}
TEST(LiquidBoundaryGeometry3D, CapillaryBoundZeroSigmaAndDegenerateDerivativeChecks) {
  const auto grid = Grid();
  auto samples = IsolatedLiquid();
  const auto zero = Freeze(grid, samples);
  ASSERT_TRUE(zero.ok()) << zero.status();
  EXPECT_EQ(zero->capillary_step_bound(), kInfinity);
  EXPECT_EQ(Freeze(grid, samples, {}, 0.072).status().code(), absl::StatusCode::kFailedPrecondition);
  samples = Sample([](double x, double, double) {
    return x - 0.75;
  });
  const auto positive = Freeze(grid, samples, {}, 0.072);
  ASSERT_TRUE(positive.ok()) << positive.status();
  const double bound = std::sqrt(1000.0 / (4.0 * std::acos(-1.0) * 0.072));
  EXPECT_NEAR(positive->capillary_step_bound(), bound, 1e-13);
  EXPECT_EQ(Freeze(grid, samples, {}, 0.072, bound * 1.001).status().code(), absl::StatusCode::kFailedPrecondition);
  samples.centers.erase(samples.centers.begin() + 87);
  EXPECT_FALSE(Freeze(grid, samples, {}, 0.072).ok());
}
TEST(LiquidBoundaryGeometry3D, FrozenSnapshotPreservesWallFluxAndSeparatesPressureHistory) {
  const auto grid = Grid();
  auto samples = IsolatedLiquid(-0.25);
  const auto frozen = Freeze(grid, samples, TranslatingHalfspace3D{{1, 0, 0}, 0.25, {2, 3, 4}});
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  samples.centers.clear();
  samples.corners.clear();
  samples.candidates.clear();
  std::vector<LiquidPressureCellFields3D> fields{{0.0, kInfinity, 11.0}};
  std::vector<double> velocities;
  for (const auto& face : frozen->faces()) {
    velocities.push_back(2.0 + static_cast<int>(face.face_index.axis));
  }
  const auto links = BuildLiquidPressureLinks3D(*frozen, {fields, velocities, 11.0});
  ASSERT_TRUE(links.ok()) << links.status();
  EXPECT_DOUBLE_EQ(links->cells[0].previous_pressure, 11.0);
  EXPECT_NEAR(links->cells[0].fluid_volume, 0.75, 2e-14);
  EXPECT_TRUE(std::holds_alternative<SealedMacPressureLink3D>(links->faces[0].pressure_link));
  const auto projection =
      ProjectMacPressure3D({frozen->mac_grid(), frozen->step_duration(), links->cells, links->faces});
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_NEAR(projection->cell_pressures[0], 11.0, 1e-6);
  for (std::size_t i = 0; i < velocities.size(); ++i) {
    EXPECT_NEAR(projection->corrected_face_velocities[i], velocities[i], 1e-9);
  }
}
TEST(LiquidBoundaryGeometry3D, InternalLinkUsesFullFrozenAreaAndZeroJump) {
  const auto grid = Grid();
  auto samples = IsolatedLiquid(-0.25);
  samples.candidates.push_back({-1, -2, -2});
  for (auto& sample : samples.centers) {
    if (sample.cell_index.x == -1 && sample.cell_index.y == -2 && sample.cell_index.z == -2) {
      sample.signed_distance = -0.25;
    }
  }
  const auto frozen = Freeze(grid, samples);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  const auto links = Links(*frozen, 4.0);
  std::size_t internal_count = 0;
  for (const auto& face : links.faces) {
    if (const auto* internal = std::get_if<InternalMacPressureLink3D>(&face.pressure_link)) {
      ++internal_count;
      EXPECT_DOUBLE_EQ(internal->pressure_jump, 0.0);
      EXPECT_DOUBLE_EQ(face.pressure_gradient_distance, 1.0);
      EXPECT_DOUBLE_EQ(face.open_area, 1.0);
    }
  }
  EXPECT_EQ(internal_count, 1);
  const auto projection = ProjectMacPressure3D({grid, 0.01, links.cells, links.faces});
  ASSERT_TRUE(projection.ok()) << projection.status();
  for (double pressure : projection->cell_pressures) {
    EXPECT_NEAR(pressure, 4.0, 1e-6);
  }
}
TEST(LiquidBoundaryGeometry3D, RejectsPressureFieldsMissingAirAndNegativeOmittedNeighbor) {
  const auto grid = Grid();
  auto samples = IsolatedLiquid();
  const auto frozen = Freeze(grid, samples);
  ASSERT_TRUE(frozen.ok());
  const std::vector<LiquidPressureCellFields3D> fields{{0, kInfinity, 0}};
  const std::vector<double> velocities(6, 0);
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(*frozen, {{}, velocities, 0}).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  auto invalid_velocities = velocities;
  invalid_velocities[0] = kInfinity;
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(*frozen, {fields, invalid_velocities, 0}).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  samples.centers.erase(samples.centers.begin() + 87);
  const auto missing = Freeze(grid, samples);
  ASSERT_TRUE(missing.ok());
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(*missing, {fields, velocities, 0}).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  samples = Sample([](double, double, double) {
    return -1.0;
  });
  const auto omitted = Freeze(grid, samples);
  ASSERT_TRUE(omitted.ok());
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(*omitted, {fields, velocities, 0}).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
}
TEST(LiquidBoundaryGeometry3D, RejectsUnrepresentableMeasuresAndInvalidPhysicalInputs) {
  const auto grid = Grid();
  const auto samples = IsolatedLiquid();
  EXPECT_EQ(Freeze(grid, samples, {}, -1).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Freeze(grid, samples, {}, 0, 0).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Freeze(grid, samples, {}, 0, 0.01, 0).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Freeze(Grid(1e-120), samples).status().code(), absl::StatusCode::kOutOfRange);
  // A positive thin blocked strip must not round away in h²-open_area.
  EXPECT_EQ(
      Freeze(grid, samples, TranslatingHalfspace3D{{1, 0, 0}, 1e-20, {0, 0, 0}}).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(
      Freeze(grid, samples, TranslatingSlab3D{{1, 0, 0}, 1e30, 1e-20, {0, 0, 0}}).status().code(),
      absl::StatusCode::kOutOfRange
  );
  auto malformed = samples;
  malformed.centers[0].signed_distance = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(Freeze(grid, malformed).status().code(), absl::StatusCode::kInvalidArgument);
  malformed = samples;
  malformed.candidates.resize(4097, {-2, -2, -2});
  EXPECT_EQ(Freeze(grid, malformed).status().code(), absl::StatusCode::kResourceExhausted);
}
TEST(LiquidBoundaryGeometry3D, InteriorPositiveSigmaDoesNotRequireUnusedCurvature) {
  const auto grid = Grid();
  auto samples = Sample([](double, double, double) {
    return -1.0;
  });
  std::erase_if(samples.centers, [](const auto& sample) {
    const auto index = sample.cell_index;
    return std::abs(index.x + 2) + std::abs(index.y + 2) + std::abs(index.z + 2) > 1;
  });
  const auto frozen = Freeze(grid, samples, {}, 0.072, 1e6);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  EXPECT_EQ(frozen->cells().size(), 1);
  EXPECT_EQ(frozen->capillary_step_bound(), kInfinity);
}
TEST(LiquidBoundaryGeometry3D, SealedLowFaceAtMinimumSignedIndexNeedsNoExteriorSample) {
  constexpr auto minimum_index = std::numeric_limits<std::int32_t>::min();
  const std::array blocks{BlockIndex3D{minimum_index / 4, -1, -1}};
  const auto grid = SparseMacGrid3D::Create({{2147483648.0, 2.0, 2.0}, 1.0}, blocks, 1);
  ASSERT_TRUE(grid.ok()) << grid.status();
  auto samples = IsolatedLiquid();
  std::erase_if(samples.centers, [](const auto& sample) {
    return sample.cell_index.x < -2;
  });
  std::erase_if(samples.corners, [](const auto& sample) {
    return sample.corner_index.x < -2;
  });
  for (auto& sample : samples.centers) {
    sample.cell_index.x = minimum_index + (sample.cell_index.x + 2);
  }
  for (auto& sample : samples.corners) {
    sample.corner_index.x = minimum_index + (sample.corner_index.x + 2);
  }
  samples.candidates[0].x = minimum_index;
  const auto frozen = Freeze(*grid, samples, TranslatingHalfspace3D{{1, 0, 0}, 0.25, {0, 0, 0}});
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  const auto links = Links(*frozen, 2.0);
  const auto projection = ProjectMacPressure3D({*grid, 0.01, links.cells, links.faces});
  ASSERT_TRUE(projection.ok()) << projection.status();
  EXPECT_NEAR(projection->cell_pressures[0], 2.0, 1e-6);
}
TEST(LiquidBoundaryGeometry3D, ResolvesRepresentableTinyLiquidTetrahedron) {
  const auto grid = Grid();
  auto samples = Sample([](double x, double y, double z) {
    return x + y + z - 1e-100;
  });
  for (auto& sample : samples.centers) {
    sample.signed_distance = -1;
  }
  const auto frozen = Freeze(grid, samples);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  EXPECT_NEAR(frozen->cells()[0].fluid_volume / 1e-300, 1.0 / 6.0, 1e-14);
}
TEST(LiquidBoundaryGeometry3D, RejectsCapillaryLiftThatUnderflowsToZero) {
  const auto grid = Grid(1e100);
  auto samples = IsolatedLiquid();
  for (auto& sample : samples.centers) {
    if (sample.cell_index.x == -1 && sample.cell_index.y == -2 && sample.cell_index.z == -2) {
      sample.signed_distance = 1.99;
    }
    sample.signed_distance *= 1e100;
  }
  for (auto& sample : samples.corners) {
    sample.signed_distance *= 1e100;
  }
  const auto frozen = Freeze(grid, samples, {}, 1e-300, 0.01, 1e-300);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  const std::vector<LiquidPressureCellFields3D> fields{{0, kInfinity, 0}};
  const std::vector<double> velocities(6, 0);
  EXPECT_EQ(
      BuildLiquidPressureLinks3D(*frozen, {fields, velocities, 0}).status().code(),
      absl::StatusCode::kOutOfRange
  );
}
TEST(LiquidBoundaryGeometry3D, ZeroCornerPlateauHasNoStrictlyNegativeLiquidVolume) {
  const auto grid = Grid();
  auto samples = IsolatedLiquid();
  for (auto& sample : samples.corners) {
    sample.signed_distance = 0;
  }
  const auto frozen = Freeze(grid, samples);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  EXPECT_TRUE(frozen->cells().empty());
  EXPECT_TRUE(frozen->faces().empty());
}
TEST(LiquidBoundaryGeometry3D, MixedZeroPlateauCountsOnlyStrictlyNegativeTetrahedra) {
  const auto grid = Grid();
  auto samples = IsolatedLiquid();
  for (auto& sample : samples.corners) {
    const auto index = sample.corner_index;
    sample.signed_distance = index.x == -1 && index.y == -2 && index.z == -2 ? -1.0 : 0.0;
  }
  const auto frozen = Freeze(grid, samples);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  // Exactly the two tetrahedra whose increasing coordinate walk starts in x
  // contain the negative corner (1,0,0); their remaining boundary vertices are zero.
  EXPECT_NEAR(frozen->cells()[0].fluid_volume, 1.0 / 3.0, 2e-14);
}
TEST(LiquidBoundaryGeometry3D, PhysicalVolumeSurvivesUnderflowOfNormalizedCubicMeasure) {
  const auto grid = Grid(1e100);
  auto samples = Sample([](double x, double y, double z) {
    return x + y + z - 1e-110;
  });
  for (auto& sample : samples.centers) {
    sample.signed_distance = -1;
  }
  const auto frozen = Freeze(grid, samples);
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  // Three world-space intercepts are 1e-10 m, so the volume is 1e-30/6 m³.
  EXPECT_NEAR(frozen->cells()[0].fluid_volume / 1e-30, 1.0 / 6.0, 2e-14);
  std::cout << "scaled_tetrahedron_volume_error=" << std::abs(frozen->cells()[0].fluid_volume - 1e-30 / 6.0) << '\n';
}
TEST(LiquidBoundaryGeometry3D, PhysicalFaceAreaSurvivesUnderflowOfNormalizedSquareMeasure) {
  const auto grid = Grid(1e100);
  const auto samples = Sample([](double, double, double) {
    return -1.0;
  });
  const double diagonal_component = 1.0 / std::sqrt(2.0);
  const auto frozen = Freeze(
      grid,
      samples,
      TranslatingHalfspace3D{
          {-diagonal_component, -diagonal_component, 0},
          -diagonal_component * 1e-70,
          {0, 0, 0},
      }
  );
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_EQ(frozen->cells().size(), 1);
  // The open prism has two world-space intercepts of 1e-70 m and length 1e100 m.
  EXPECT_NEAR(frozen->cells()[0].fluid_volume / 1e-40, 0.5, 2e-14);
  std::size_t transverse_faces = 0;
  double maximum_area_error = 0;
  for (const auto& face : frozen->faces()) {
    if (face.face_index.axis == FaceNormalAxis::kZ) {
      ++transverse_faces;
      EXPECT_NEAR(face.open_area / 1e-140, 0.5, 2e-14);
      maximum_area_error = std::max(maximum_area_error, std::abs(face.open_area - 5e-141));
      EXPECT_DOUBLE_EQ(face.blocked_area, 1e200 - face.open_area);
    }
  }
  EXPECT_EQ(transverse_faces, 2);
  std::cout << "scaled_prism_volume_error=" << std::abs(frozen->cells()[0].fluid_volume - 5e-41)
            << " scaled_face_area_error=" << maximum_area_error << '\n';
}
}  // namespace
}  // namespace fluidity
