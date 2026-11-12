#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "fixture_data.h"
#include "fluidity/base/liquid_boundary_geometry_math_3d.h"
#include "fluidity/host/liquid_boundary_geometry_3d.h"
#include "fluidity/host/liquid_cavity.h"
#include "gtest/gtest.h"

namespace fluidity::boundary_test {
class GeometryCases {
 public:
  using FreezeFunction =
      std::function<absl::StatusOr<FrozenLiquidBoundaryGeometry3D>(const LiquidBoundaryGeometryInputs3D&)>;
  using LinksFunction = std::function<
      absl::StatusOr<LiquidPressureProblem3D>(const FrozenLiquidBoundaryGeometry3D&, const LiquidPressureFields3D&)>;
  using ProjectFunction = std::function<absl::StatusOr<MacPressureProjection3D>(const MacPressureInputs3D&)>;
  GeometryCases(
      FreezeFunction freeze,
      LinksFunction links,
      ProjectFunction project =
          [](const MacPressureInputs3D& inputs) {
            return ProjectMacPressure3D(inputs);
          }
  )
      : freeze_(freeze), links_(links), project_(project) {
  }
  ProjectFunction project_;
  FreezeFunction freeze_;
  LinksFunction links_;

  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
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
    return freeze_({grid, samples.candidates, samples.centers, samples.corners, obstacle, dt, rho, sigma});
  }
  LiquidPressureProblem3D Links(const FrozenLiquidBoundaryGeometry3D& frozen, double ambient = 0.0) {
    std::vector<LiquidPressureCellFields3D> cells(frozen.cells().size(), {0.0, kInfinity, 0.0});
    std::vector<double> velocities(frozen.faces().size(), 0.0);
    auto result = links_(frozen, {cells, velocities, ambient});
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? *result : LiquidPressureProblem3D{};
  }

  void OpenTankClipsVolumeAndUsesPistonOnlyFlux() {
    std::cout << "CTEST_FULL_OUTPUT\n";
    const auto grid = Grid();
    auto samples = Sample([](double, double, double) {
      return -1.0;
    });
    for (auto& center : samples.centers) {
      if (center.cell_index.x != -2 || center.cell_index.y != -2 || center.cell_index.z != -2) {
        center.signed_distance = 1;
      }
    }
    AxisAlignedLiquidTank3D tank{.25, 2, .125, .125, .75, .125};
    const auto frozen = Freeze(grid, samples, tank, 0, .5, 1);
    ASSERT_TRUE(frozen.ok()) << frozen.status();
    ASSERT_EQ(frozen->cells().size(), 1);
    EXPECT_NEAR(frozen->cells()[0].fluid_volume, .41015625, 2e-14);
    ASSERT_EQ(frozen->faces().size(), 6);

    const auto head = [](const FrozenLiquidFace3D& face) {
      const auto index = face.face_index.lattice_index;
      return index.x == -2 && index.y == -2 && index.z == -2;
    };
    const auto aperture = [&](const FrozenLiquidFace3D& face) {
      if (!head(face)) {
        switch (face.face_index.axis) {
          case FaceNormalAxis::kX:
            return .546875;
          case FaceNormalAxis::kY:
            return .46875;
          case FaceNormalAxis::kZ:
            break;
        }
      }
      return 0.0;
    };
    constexpr double kPistonFlux = .068359375;
    std::size_t low_x = 0;
    std::size_t high_x = 0;
    for (std::size_t ordinal = 0; ordinal < frozen->faces().size(); ++ordinal) {
      const auto& face = frozen->faces()[ordinal];
      EXPECT_DOUBLE_EQ(face.open_area, aperture(face));
      EXPECT_DOUBLE_EQ(face.blocked_area, 1 - aperture(face));
      ASSERT_TRUE(face.prescribed_wall_flux.has_value());
      const bool piston = head(face) && face.face_index.axis == FaceNormalAxis::kX;
      EXPECT_DOUBLE_EQ(*face.prescribed_wall_flux, piston ? kPistonFlux : 0);
      EXPECT_DOUBLE_EQ(face.wall_normal_velocity, piston ? .125 : 0);
      if (face.face_index.axis == FaceNormalAxis::kX) {
        (head(face) ? low_x : high_x) = ordinal;
      }
    }

    auto problem = Links(*frozen);
    ASSERT_EQ(problem.faces.size(), 6);
    ASSERT_EQ(problem.cells.size(), 1);
    ASSERT_EQ(problem.faces[low_x].prescribed_wall_flux, kPistonFlux);
    const auto project_and_check = [&](double piston_outward_flux) {
      const auto projected = project_({grid, .5, problem.cells, problem.faces});
      ASSERT_TRUE(projected.ok()) << projected.status();
      ASSERT_EQ(projected->integrated_rhs.size(), 1);
      EXPECT_DOUBLE_EQ(projected->integrated_rhs[0], -piston_outward_flux);
      // The two open high faces have theta=1/2 and dt/rho=1/2.
      EXPECT_NEAR(projected->cell_pressures[0], -piston_outward_flux / 1.015625, 2e-8);
      double corrected_outward_flux = piston_outward_flux;
      for (std::size_t ordinal = 0; ordinal < frozen->faces().size(); ++ordinal) {
        const auto& face = frozen->faces()[ordinal];
        const double outward_sign = head(face) ? -1.0 : 1.0;
        corrected_outward_flux += outward_sign * aperture(face) * projected->corrected_face_velocities[ordinal];
      }
      EXPECT_NEAR(projected->cell_constraint_defects[0], corrected_outward_flux / .41015625, 2e-14);
      EXPECT_NEAR(corrected_outward_flux, 0.0, 2e-8);
      std::cout << std::setprecision(17) << "tank_rhs=" << projected->integrated_rhs[0]
                << " tank_pressure=" << projected->cell_pressures[0]
                << " tank_defect=" << projected->cell_constraint_defects[0] << '\n';
    };
    project_and_check(-kPistonFlux);
    const std::array cell_fields{LiquidPressureCellFields3D{0, kInfinity, 0}};
    const std::array<double, 6> velocities{};
    std::array<std::optional<double>, 6> fluxes{};
    const auto build_with_fluxes = [&](std::span<const std::optional<double>> fluxes_by_face) {
      return links_(*frozen, {cell_fields, velocities, 0, fluxes_by_face});
    };
    EXPECT_EQ(build_with_fluxes(std::span(fluxes).first(5)).status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(build_with_fluxes(fluxes).status().code(), absl::StatusCode::kInvalidArgument);
    for (const double invalid :
         {-0.25, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
      fluxes[low_x] = invalid;
      EXPECT_EQ(build_with_fluxes(fluxes).status().code(), absl::StatusCode::kInvalidArgument);
    }
    fluxes[low_x] = 0;
    fluxes[high_x] = 0;
    EXPECT_EQ(build_with_fluxes(fluxes).status().code(), absl::StatusCode::kInvalidArgument);
    fluxes[high_x].reset();
    for (const double explicit_flux : {0.0, 0.25}) {
      fluxes[low_x] = explicit_flux;
      const auto overridden = build_with_fluxes(fluxes);
      ASSERT_TRUE(overridden.ok()) << overridden.status();
      ASSERT_EQ(overridden->cells.size(), 1);
      EXPECT_DOUBLE_EQ(overridden->cells[0].fluid_volume, problem.cells[0].fluid_volume);
      ASSERT_EQ(overridden->faces.size(), problem.faces.size());
      for (std::size_t ordinal = 0; ordinal < overridden->faces.size(); ++ordinal) {
        const auto& before = problem.faces[ordinal];
        const auto& after = overridden->faces[ordinal];
        EXPECT_EQ(after.face_index.axis, before.face_index.axis);
        EXPECT_EQ(after.face_index.lattice_index.x, before.face_index.lattice_index.x);
        EXPECT_EQ(after.face_index.lattice_index.y, before.face_index.lattice_index.y);
        EXPECT_EQ(after.face_index.lattice_index.z, before.face_index.lattice_index.z);
        EXPECT_DOUBLE_EQ(after.open_area, before.open_area);
        EXPECT_DOUBLE_EQ(after.inverse_density, before.inverse_density);
        EXPECT_DOUBLE_EQ(after.pressure_gradient_distance, before.pressure_gradient_distance);
        EXPECT_DOUBLE_EQ(after.wall_normal_velocity, before.wall_normal_velocity);
        EXPECT_EQ(after.pressure_link.index(), before.pressure_link.index());
        EXPECT_EQ(after.prescribed_wall_flux, ordinal == low_x ? fluxes[low_x] : before.prescribed_wall_flux);
      }
      const auto legacy = problem;
      problem = *overridden;
      project_and_check(-explicit_flux);
      problem = legacy;

      // Closing the two air links removes the anchor without changing the
      // applied wall flux. An equal sink balances the inward flux.
      auto sealed = *overridden;
      for (auto& face : sealed.faces) {
        face.open_area = 0;
        face.pressure_link = SealedMacPressureLink3D{};
      }
      if (explicit_flux > 0) {
        EXPECT_EQ(
            project_({grid, .5, sealed.cells, sealed.faces}).status().code(),
            absl::StatusCode::kFailedPrecondition
        );
      }
      sealed.cells[0].target_divergence = -explicit_flux / sealed.cells[0].fluid_volume;
      const auto compatible = project_({grid, .5, sealed.cells, sealed.faces});
      ASSERT_TRUE(compatible.ok()) << compatible.status();
      EXPECT_EQ(compatible->gauge_cell_ordinals, (std::vector<std::size_t>{0}));
      EXPECT_NEAR(compatible->integrated_rhs[0], 0, 1e-15);
      EXPECT_NEAR(compatible->cell_constraint_defects[0], 0, 1e-15);
    }
    // Move the same signed physical datum to a face where this cell is the tail.
    problem.faces[low_x].prescribed_wall_flux = 0;
    problem.faces[high_x].prescribed_wall_flux = kPistonFlux;
    project_and_check(kPistonFlux);
    problem.faces[high_x].prescribed_wall_flux = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(project_({grid, .5, problem.cells, problem.faces}).status().code(), absl::StatusCode::kInvalidArgument);
    problem.faces[high_x].prescribed_wall_flux = kPistonFlux;
    problem.faces[high_x].open_area = 1;
    EXPECT_EQ(project_({grid, .5, problem.cells, problem.faces}).status().code(), absl::StatusCode::kInvalidArgument);

    tank.piston_velocity_x = -.125;
    const auto retreating_piston = Freeze(grid, samples, tank, 0, .5, 1);
    ASSERT_TRUE(retreating_piston.ok()) << retreating_piston.status();
    EXPECT_EQ(retreating_piston->faces()[low_x].prescribed_wall_flux, -kPistonFlux);

    tank.high_x = 1;
    const auto with_opposite_wall = Freeze(grid, samples, tank, 0, .5, 1);
    ASSERT_TRUE(with_opposite_wall.ok()) << with_opposite_wall.status();
    EXPECT_DOUBLE_EQ(with_opposite_wall->faces()[high_x].open_area, 0);
    EXPECT_DOUBLE_EQ(with_opposite_wall->faces()[high_x].wall_normal_velocity, 0);
    EXPECT_EQ(with_opposite_wall->faces()[high_x].prescribed_wall_flux, 0);
  }

  void OpenTankPreservesNarrowAndLiquidClippedMeasures() {
    const auto grid = Grid();
    const auto samples = Sample([](double, double y, double) {
      return y - .75;
    });
    AxisAlignedLiquidTank3D tank{.25, 2, .125, .125, .75, .125};
    const auto clipped = Freeze(grid, samples, tank);
    ASSERT_TRUE(clipped.ok()) << clipped.status();
    ASSERT_EQ(clipped->cells().size(), 1);
    EXPECT_NEAR(clipped->cells()[0].fluid_volume, .29296875, 2e-14);

    const auto thin_liquid = Sample([](double, double y, double) {
      return y == .5 ? -1.0 : y - (.125 + 0x1p-30);
    });
    const auto near_empty = Freeze(grid, thin_liquid, tank);
    ASSERT_TRUE(near_empty.ok()) << near_empty.status();
    ASSERT_EQ(near_empty->cells().size(), 1);
    EXPECT_NEAR(near_empty->cells()[0].fluid_volume, .46875 * 0x1p-30, 2e-17);
    // Pressure apertures and piston flux use fixed-wall geometry on the
    // selected cell face, even when the liquid tetrahedra are nearly dry.
    const auto piston_face = std::ranges::find_if(near_empty->faces(), [](const auto& face) {
      return face.face_index.axis == FaceNormalAxis::kX && face.face_index.lattice_index.x == -2;
    });
    ASSERT_NE(piston_face, near_empty->faces().end());
    EXPECT_EQ(piston_face->prescribed_wall_flux, .068359375);

    tank.piston_velocity_x = 0;
    tank.start_low_x = 0;
    tank.high_x = 0x1p-45;
    const auto narrow = Freeze(grid, samples, tank);
    ASSERT_TRUE(narrow.ok()) << narrow.status();
    ASSERT_EQ(narrow->cells().size(), 1);
    EXPECT_NEAR(narrow->cells()[0].fluid_volume, .390625 * 0x1p-45, 1e-27);

    // The physical cut at the right end of [-1,0] is positive but the
    // tank's low-x plane rounds to 1 in the unit cell frame.
    auto collapsed_local_cut = samples;
    collapsed_local_cut.candidates.front().x = -3;
    const AxisAlignedLiquidTank3D below_local_resolution{-0x1p-60, 0, .125, .125, .75, 0};
    EXPECT_EQ(Freeze(grid, collapsed_local_cut, below_local_resolution).status().code(), absl::StatusCode::kOutOfRange);

    tank.high_x = 0;
    EXPECT_EQ(Freeze(grid, samples, tank).status().code(), absl::StatusCode::kInvalidArgument);
    tank.high_x = 1;
    tank.high_z = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(Freeze(grid, samples, tank).status().code(), absl::StatusCode::kInvalidArgument);
    tank.high_z = .75;
    tank.piston_velocity_x = 1;
    EXPECT_EQ(Freeze(grid, samples, tank, 0, 1).status().code(), absl::StatusCode::kFailedPrecondition);
    tank.piston_velocity_x = kInfinity;
    EXPECT_EQ(Freeze(grid, samples, tank).status().code(), absl::StatusCode::kInvalidArgument);
    tank.piston_velocity_x = -std::numeric_limits<double>::max();
    EXPECT_EQ(Freeze(grid, samples, tank, 0, 2).status().code(), absl::StatusCode::kOutOfRange);
  }

  void AnalyticPlaneFixtureAllAxesSignedIndicesAndTinyOpening() {
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

  void ObliqueAndSimultaneousLiquidSolidClips() {
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

  void ThinSlabInvisibleToCornersAndSplitCellRefusal() {
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

  void SharedFaceHasOneOwnerAndMatchingTriangulation() {
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
  void SharedLiquidInterfaceUsesSeriesResistanceAndOneJump() {
    const auto grid = Grid();
    auto samples = Sample([](double, double, double) {
      return -1.0;
    });
    samples.candidates.push_back({-1, -2, -2});
    for (auto& center : samples.centers) {
      const auto index = center.cell_index;
      if (index.y != -2 || index.z != -2 || (index.x != -2 && index.x != -1)) {
        center.signed_distance = 1.0;
      }
    }
    const auto frozen = Freeze(grid, samples);
    ASSERT_TRUE(frozen.ok()) << frozen.status();
    ASSERT_EQ(frozen->cells().size(), 2);
    const auto faces = frozen->faces();
    const auto shared = std::ranges::find_if(faces, [](const FrozenLiquidFace3D& face) {
      const auto index = face.face_index.lattice_index;
      return face.face_index.axis == FaceNormalAxis::kX && index.x == -1 && index.y == -2 && index.z == -2;
    });
    ASSERT_NE(shared, faces.end());
    const auto ordinal = static_cast<std::size_t>(shared - faces.begin());
    std::vector<double> velocities(faces.size(), 0.0);
    std::array cells{
        LiquidPressureCellFields3D{0.0, kInfinity, 0.0, 1000.0},
        LiquidPressureCellFields3D{0.0, kInfinity, 0.0, 800.0},
    };
    std::array crossings{LiquidInternalInterface3D{ordinal, 0.25, 0.5, 6.0, 1.0}};
    const LiquidPressureFields3D fields{
        .cells = cells,
        .predicted_face_velocities = velocities,
        .ambient_pressure = 0.0,
        .internal_interfaces = crossings,
    };
    for (const double expected_jump : {-2.0, 2.0}) {
      const auto links = links_(*frozen, fields);
      ASSERT_TRUE(links.ok()) << links.status();
      const auto& face = links->faces[ordinal];
      EXPECT_DOUBLE_EQ(face.inverse_density, 1.0 / 850.0);
      const auto* internal = std::get_if<InternalMacPressureLink3D>(&face.pressure_link);
      ASSERT_NE(internal, nullptr);
      EXPECT_DOUBLE_EQ(internal->pressure_jump, expected_jump);

      std::vector<MacPressureFace3D> sealed(links->faces.begin(), links->faces.end());
      for (std::size_t index = 0; index < sealed.size(); ++index) {
        if (index != ordinal) {
          sealed[index].open_area = 0.0;
          sealed[index].pressure_link = SealedMacPressureLink3D{};
        }
      }
      const auto projection = project_({grid, 0.01, links->cells, sealed});
      ASSERT_TRUE(projection.ok()) << projection.status();
      ASSERT_EQ(projection->cell_pressures.size(), 2);
      EXPECT_NEAR(projection->cell_pressures[1] - projection->cell_pressures[0], expected_jump, 1.0e-6);
      EXPECT_NEAR(projection->corrected_face_velocities[ordinal], 0.0, 1.0e-9);

      std::swap(cells[0].density, cells[1].density);
      crossings[0].tail_fraction = 1.0 - crossings[0].tail_fraction;
      crossings[0].curvature_tail_to_head = -crossings[0].curvature_tail_to_head;
      crossings[0].normal_viscous_jump_tail_to_head = -crossings[0].normal_viscous_jump_tail_to_head;
    }
    cells[0].density.reset();
    EXPECT_EQ(links_(*frozen, fields).status().code(), absl::StatusCode::kInvalidArgument);
    cells[0].density = 1000.0;
    crossings[0].face_ordinal = (ordinal + 1) % faces.size();
    EXPECT_EQ(links_(*frozen, fields).status().code(), absl::StatusCode::kInvalidArgument);
  }

  void CavityGasPressureAddsSignedCapillaryJumpOnce() {
    const auto grid = Grid();
    const auto samples = IsolatedLiquid(-0.25);
    constexpr double kSigma = 0.5;
    constexpr double kCurvature = -2.0;
    const auto frozen = freeze_({
        .mac_grid = grid,
        .cell_indices = samples.candidates,
        .center_distances = samples.centers,
        .corner_distances = samples.corners,
        .obstacle = {},
        .step_duration = 0.01,
        .liquid_density = 1000.0,
        .surface_tension = kSigma,
        .exact_outward_interface_curvature = kCurvature,
    });
    ASSERT_TRUE(frozen.ok()) << frozen.status();
    ASSERT_EQ(frozen->cells().size(), 1U);
    const auto cavity = EvaluateLiquidCavity3D({1.0, 100019.0}, 1.0, 100000.0, 1000.0, kSigma, kCurvature, 0.25);
    ASSERT_TRUE(cavity.ok()) << cavity.status();
    std::array air_pressures{
        LiquidAirCellPressure3D{.cell_index = {-3, -2, -2}, .gas_gauge_pressure = cavity->gas_gauge_pressure},
        LiquidAirCellPressure3D{.cell_index = {-1, -2, -2}, .gas_gauge_pressure = -3.0},
    };
    const std::array cells{LiquidPressureCellFields3D{0.0, kInfinity, 0.0}};
    const std::vector<double> velocities(frozen->faces().size(), 0.0);
    const LiquidPressureFields3D fields{
        .cells = cells,
        .predicted_face_velocities = velocities,
        .ambient_pressure = 7.0,
        .air_cell_pressures = air_pressures,
    };
    const auto links = links_(*frozen, fields);
    ASSERT_TRUE(links.ok()) << links.status();
    ASSERT_EQ(links->faces.size(), 6U);
    std::size_t cavity_faces = 0;
    for (const auto& face : links->faces) {
      const auto* air = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link);
      ASSERT_NE(air, nullptr);
      if (face.face_index.axis != FaceNormalAxis::kX) {
        EXPECT_EQ(air->exterior_pressure, 6.0);
        continue;
      }
      ++cavity_faces;
      const bool low_face = face.face_index.lattice_index.x == -2;
      EXPECT_EQ(air->exterior_pressure, low_face ? 18.0 : -4.0);
      if (low_face) {
        EXPECT_EQ(air->exterior_pressure, cavity->boundary_gauge_pressure);
      }
    }
    EXPECT_EQ(cavity_faces, 2U);
    const auto projection = project_({grid, 0.01, links->cells, links->faces});
    ASSERT_TRUE(projection.ok()) << projection.status();
    ASSERT_EQ(projection->cell_pressures.size(), 1U);
    EXPECT_NEAR(projection->cell_pressures[0], (18.0 - 4.0 + 4.0 * 6.0) / 6.0, 1e-6);
    EXPECT_LE(projection->diagnostics.max_cell_constraint_defect, 2e-5);
    air_pressures[1].cell_index = {-2, -2, -2};
    EXPECT_EQ(links_(*frozen, fields).status().code(), absl::StatusCode::kInvalidArgument);
    air_pressures[1].cell_index = air_pressures[0].cell_index;
    EXPECT_EQ(links_(*frozen, fields).status().code(), absl::StatusCode::kInvalidArgument);
  }

  void ThetaFixtureAndSignedPressureMapping() {
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
      const auto projection = project_({grid, 0.01, links.cells, links.faces});
      ASSERT_TRUE(projection.ok()) << projection.status();
      EXPECT_NEAR(projection->cell_pressures[0], 7.0, 1e-6);
      for (double velocity : projection->corrected_face_velocities) {
        EXPECT_NEAR(velocity, 0.0, 1e-9);
      }
    }
  }
  void PlaneAndSphereCenteredCurvature() {
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
  void CapillaryLiftOnceAndCorrectionMismatchControl() {
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
    const auto projection = project_({grid, 0.01, links.cells, links.faces});
    ASSERT_TRUE(projection.ok()) << projection.status();
    EXPECT_NEAR(projection->cell_pressures[0], 3.576, 1e-6);
    for (double velocity : projection->corrected_face_velocities) {
      EXPECT_NEAR(velocity, 0.0, 1e-9);
    }
    auto driven = links;
    driven.faces[0].predicted_normal_velocity = 0.2;
    const auto projected = project_({grid, 0.01, driven.cells, driven.faces});
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
  void RejectsMissingSamplesOwnersOrderingAndInvalidObstacle() {
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
  void CapillaryBoundZeroSigmaAndDegenerateDerivativeChecks() {
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
  void FrozenSnapshotPreservesWallFluxAndSeparatesPressureHistory() {
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
    const auto links = links_(*frozen, {fields, velocities, 11.0});
    ASSERT_TRUE(links.ok()) << links.status();
    EXPECT_DOUBLE_EQ(links->cells[0].previous_pressure, 11.0);
    EXPECT_NEAR(links->cells[0].fluid_volume, 0.75, 2e-14);
    EXPECT_TRUE(std::holds_alternative<SealedMacPressureLink3D>(links->faces[0].pressure_link));
    const auto projection = project_({frozen->mac_grid(), frozen->step_duration(), links->cells, links->faces});
    ASSERT_TRUE(projection.ok()) << projection.status();
    EXPECT_NEAR(projection->cell_pressures[0], 11.0, 1e-6);
    for (std::size_t i = 0; i < velocities.size(); ++i) {
      EXPECT_NEAR(projection->corrected_face_velocities[i], velocities[i], 1e-9);
    }
  }
  void InternalLinkUsesFullFrozenAreaAndZeroJump() {
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
    const auto projection = project_({grid, 0.01, links.cells, links.faces});
    ASSERT_TRUE(projection.ok()) << projection.status();
    for (double pressure : projection->cell_pressures) {
      EXPECT_NEAR(pressure, 4.0, 1e-6);
    }
  }
  void RejectsPressureFieldsMissingAirAndNegativeOmittedNeighbor() {
    const auto grid = Grid();
    auto samples = IsolatedLiquid();
    const auto frozen = Freeze(grid, samples);
    ASSERT_TRUE(frozen.ok());
    const std::vector<LiquidPressureCellFields3D> fields{{0, kInfinity, 0}};
    const std::vector<double> velocities(6, 0);
    EXPECT_EQ(links_(*frozen, {{}, velocities, 0}).status().code(), absl::StatusCode::kInvalidArgument);
    auto invalid_velocities = velocities;
    invalid_velocities[0] = kInfinity;
    EXPECT_EQ(links_(*frozen, {fields, invalid_velocities, 0}).status().code(), absl::StatusCode::kInvalidArgument);
    samples.centers.erase(samples.centers.begin() + 87);
    const auto missing = Freeze(grid, samples);
    ASSERT_TRUE(missing.ok());
    EXPECT_EQ(links_(*missing, {fields, velocities, 0}).status().code(), absl::StatusCode::kFailedPrecondition);
    samples = Sample([](double, double, double) {
      return -1.0;
    });
    const auto omitted = Freeze(grid, samples);
    ASSERT_TRUE(omitted.ok());
    EXPECT_EQ(links_(*omitted, {fields, velocities, 0}).status().code(), absl::StatusCode::kFailedPrecondition);
  }
  void RejectsUnrepresentableMeasuresAndInvalidPhysicalInputs() {
    // Exercise internal bounded-storage failure directly: the certified public
    // two-cut tetrahedron does not exhaust this capacity. Clipping propagates
    // overflow into an empty container, which must never become zero volume.
    boundary_internal::ClippingPolyhedron overflowed_polyhedron;
    while (!overflowed_polyhedron.overflow) {
      overflowed_polyhedron.push_back({});
    }
    EXPECT_TRUE(std::isnan(boundary_internal::ComputeClippedVolume(overflowed_polyhedron, 1.0)));
    const auto clipped_overflow = boundary_internal::ClipPolyhedron(overflowed_polyhedron, {{1.0, 0.0, 0.0}, 0.0});
    ASSERT_TRUE(clipped_overflow.overflow);
    ASSERT_TRUE(clipped_overflow.empty());
    EXPECT_TRUE(std::isnan(boundary_internal::ComputeClippedVolume(clipped_overflow, 1.0)));

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
  void InteriorPositiveSigmaDoesNotRequireUnusedCurvature() {
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
  void SealedLowFaceAtMinimumSignedIndexNeedsNoExteriorSample() {
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
    const auto projection = project_({*grid, 0.01, links.cells, links.faces});
    ASSERT_TRUE(projection.ok()) << projection.status();
    EXPECT_NEAR(projection->cell_pressures[0], 2.0, 1e-6);
  }
  void ResolvesRepresentableTinyLiquidTetrahedron() {
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
  void RejectsCapillaryLiftThatUnderflowsToZero() {
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
    EXPECT_EQ(links_(*frozen, {fields, velocities, 0}).status().code(), absl::StatusCode::kOutOfRange);
  }
  void ZeroCornerPlateauHasNoStrictlyNegativeLiquidVolume() {
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
  void MixedZeroPlateauCountsOnlyStrictlyNegativeTetrahedra() {
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
  void PhysicalVolumeSurvivesUnderflowOfNormalizedCubicMeasure() {
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
  void PhysicalFaceAreaSurvivesUnderflowOfNormalizedSquareMeasure() {
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
};
}  // namespace fluidity::boundary_test
