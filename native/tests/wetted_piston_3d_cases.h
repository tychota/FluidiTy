#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "fluidity/host/wetted_piston_3d.h"
#include "gtest/gtest.h"

namespace fluidity::wetted_piston_test {

inline constexpr LiquidStateCapacities3D kCapacity{16, 8, 8, 96, 0};

struct Fixture {
  AxisAlignedLiquidTank3D tank{0, 4, 0, 0, 1, .25};
  UniformMacGeometry3D geometry{{0, 0, 0}, 1};
  std::vector<CellIndex3D> selected{{0, 0, 0}};

  template <typename Compute>
  absl::StatusOr<WettedPistonView3D> Run(
      Compute compute,
      const LiquidSceneField3D& field,
      std::span<const LiquidParticleState3D> particles,
      WettedPistonWorkspace3D& workspace
  ) const {
    // Use the checked freeze to create real slots and low-face incidence. The
    // wet stage evaluates the true field independently of these coarse samples.
    constexpr std::array blocks{
        BlockIndex3D{-1, -1, -1},
        BlockIndex3D{0, -1, -1},
        BlockIndex3D{-1, 0, -1},
        BlockIndex3D{0, 0, -1},
        BlockIndex3D{-1, -1, 0},
        BlockIndex3D{0, -1, 0},
        BlockIndex3D{-1, 0, 0},
        BlockIndex3D{0, 0, 0},
    };
    auto grid = SparseMacGrid3D::Create(geometry, blocks, blocks.size());
    EXPECT_TRUE(grid.ok()) << grid.status();
    if (!grid.ok()) {
      return grid.status();
    }
    std::vector<LiquidCenterDistance3D> centers;
    for (const auto cell : selected) {
      centers.push_back({cell, -1});
    }
    std::vector<LiquidCornerDistance3D> corners;
    for (int z = -2; z <= 2; ++z) {
      for (int y = -2; y <= 3; ++y) {
        for (int x = -2; x <= 3; ++x) {
          corners.push_back({{x, y, z}, -1});
        }
      }
    }
    auto freeze_workspace = LiquidBoundaryWorkspace3D::Create(kCapacity);
    EXPECT_TRUE(freeze_workspace.ok()) << freeze_workspace.status();
    if (!freeze_workspace.ok()) {
      return freeze_workspace.status();
    }
    auto frozen = FreezeLiquidBoundaryGeometry3D(
        LiquidBoundaryGeometryViewInputs3D{grid->view(), selected, centers, corners, tank, .1, 1000, 0},
        *freeze_workspace
    );
    EXPECT_TRUE(frozen.ok()) << frozen.status();
    if (!frozen.ok()) {
      return frozen.status();
    }
    return compute(WettedPistonInputs3D{field, particles, *frozen, tank}, workspace);
  }
};

inline void Contains(WettedPistonBounds3D bounds, double value) {
  EXPECT_LE(bounds.lower, value);
  EXPECT_GE(bounds.upper, value);
}

inline std::vector<double> ReceiverFluxes(const WettedPistonView3D& result) {
  std::vector<double> fluxes;
  for (const auto& flux : result.face_fluxes) {
    if (flux) {
      fluxes.push_back(*flux);
    }
  }
  return fluxes;
}

template <typename Compute>
void PlaneClippingAndSignedFlux(Compute compute, int architecture) {
  auto workspace = WettedPistonWorkspace3D::Create(kCapacity);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  Fixture fixture;
  const auto full = fixture.Run(compute, LiquidHorizontalPlaneSurface3D{1}, {}, *workspace);
  ASSERT_TRUE(full.ok()) << full.status();
  EXPECT_DOUBLE_EQ(full->whole_area.lower, 1);
  EXPECT_DOUBLE_EQ(full->whole_area.upper, 1);
  fixture.tank.low_z = .25;
  fixture.tank.high_z = .875;
  fixture.tank.floor_y = .125;
  for (double velocity : {0.25, -0.25, 0.0}) {
    SCOPED_TRACE(velocity);
    fixture.tank.piston_velocity_x = velocity;
    for (auto [surface, area] : {std::pair{1.0, .546875}, std::pair{.625, .3125}, std::pair{.125, 0.0}}) {
      SCOPED_TRACE(surface);
      const auto result = fixture.Run(compute, LiquidHorizontalPlaneSurface3D{surface}, {}, *workspace);
      ASSERT_TRUE(result.ok()) << result.status();
      if (velocity != 0) {
        EXPECT_EQ(result->compiled_architecture, architecture);
      }
      Contains(result->whole_area, area);
      Contains(result->whole_flux, velocity * area);
      const auto fluxes = ReceiverFluxes(*result);
      ASSERT_EQ(fluxes.size(), 1);
      EXPECT_DOUBLE_EQ(fluxes.front(), velocity * area);
    }
  }
}

template <typename Compute>
void OverlappingThinCapEnclosesAnalyticArea(Compute compute, int architecture) {
  auto workspace = WettedPistonWorkspace3D::Create(kCapacity);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  Fixture fixture;
  // The radius-one base section reaches y=1. The other two
  // equal spheres have coincident radius-~.11 disks in the unrepresented row.
  // Their union contributes one circular segment, too thin for any corner.
  constexpr double x = .994;
  constexpr double center_y = .892;
  const double disk_radius = std::sqrt((1 - x) * (1 + x));
  const double cap_height = 1 - center_y;
  const double cap_area = disk_radius * disk_radius * std::acos(cap_height / disk_radius)
      - cap_height * std::sqrt(disk_radius * disk_radius - cap_height * cap_height);
  std::array<LiquidParticleState3D, 3> particles{};
  particles[0].world_position = {0, 0, .5};
  particles[1].world_position = {x, center_y, .5};
  particles[2].world_position = particles[1].world_position;

  // Put the top of the base section at y=1 and keep the base nearly full by
  // clipping the tank to a short span around the circle's center in z.
  fixture.tank.low_z = .44;
  fixture.tank.high_z = .56;
  const double half_z_width = .06;
  const auto integrated_chord = [](double radius, double half_width) {
    return half_width * std::sqrt(radius * radius - half_width * half_width)
        + radius * radius * std::asin(half_width / radius);
  };
  // The smaller disk also rises above the base between their two circle
  // intersections, including a sliver below y=1. Integrate that top envelope.
  const double intersection_y = (1 - disk_radius * disk_radius + center_y * center_y) / (2 * center_y);
  const double crossing_half_z = std::sqrt((1 - intersection_y) * (1 + intersection_y));
  const double whole_area = integrated_chord(1, half_z_width) + 2 * center_y * crossing_half_z
      + integrated_chord(disk_radius, crossing_half_z) - integrated_chord(1, crossing_half_z);
  const auto result = fixture.Run(compute, LiquidParticleSphereUnion3D{1}, particles, *workspace);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->compiled_architecture, architecture);
  Contains(result->whole_area, whole_area);
  EXPECT_LE(result->whole_area.upper - result->whole_area.lower, .001 * result->whole_area.lower);
  EXPECT_EQ(result->attached_patch_count, 1);
  ASSERT_EQ(result->patches.size(), 2);
  Contains(result->patches.back().area, cap_area);
  EXPECT_GT(result->patches.back().area.upper - result->patches.back().area.lower, .001 * cap_area);
  ASSERT_TRUE(result->patches.back().receiver_face_ordinal.has_value());
  const auto fluxes = ReceiverFluxes(*result);
  ASSERT_EQ(fluxes.size(), 1);
  Contains(result->whole_flux, fluxes.front());
  std::cout << "wet_cap_oracle architecture=" << result->compiled_architecture << std::setprecision(17)
            << " whole_lower=" << result->whole_area.lower << " whole_upper=" << result->whole_area.upper
            << " cap_lower=" << result->patches.back().area.lower << " cap_upper=" << result->patches.back().area.upper
            << " flux=" << fluxes.front() << '\n';

  // A slightly taller cap has a 64-band relative width over the whole-wall
  // limit, even though it has the same certified ownership.
  particles[1].world_position.y = .895;
  particles[2].world_position.y = .895;
  const auto too_wide = fixture.Run(compute, LiquidParticleSphereUnion3D{1}, particles, *workspace);
  EXPECT_EQ(too_wide.status().code(), absl::StatusCode::kFailedPrecondition);
}

template <typename Compute>
void TangencyAndUncertifiableThinWall(Compute compute, int architecture) {
  auto workspace = WettedPistonWorkspace3D::Create(kCapacity);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  Fixture fixture;
  fixture.selected.clear();
  const auto tangent = fixture.Run(compute, LiquidAnalyticSphere3D{{1, .5, .5}, 1}, {}, *workspace);
  ASSERT_TRUE(tangent.ok()) << tangent.status();
  EXPECT_EQ(tangent->compiled_architecture, architecture);
  EXPECT_EQ(tangent->whole_area.lower, 0);
  EXPECT_EQ(tangent->whole_area.upper, 0);
  EXPECT_TRUE(ReceiverFluxes(*tangent).empty());
  fixture.selected.push_back({0, 0, 0});
  const auto tiny = fixture.Run(compute, LiquidAnalyticSphere3D{{std::nextafter(1., 0.), .5, .5}, 1}, {}, *workspace);
  EXPECT_EQ(tiny.status().code(), absl::StatusCode::kFailedPrecondition);
}

template <typename Compute>
void RefusesUnsupportedOwnershipAndKeepsPublication(Compute compute) {
  auto workspace = WettedPistonWorkspace3D::Create(kCapacity);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  Fixture fixture;
  const auto first = fixture.Run(compute, LiquidHorizontalPlaneSurface3D{1.125}, {}, *workspace);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->attached_patch_count, 1);
  const auto initial_fluxes = ReceiverFluxes(*first);
  ASSERT_EQ(initial_fluxes.size(), 1);
  EXPECT_DOUBLE_EQ(initial_fluxes.front(), .28125);
  ASSERT_EQ(first->patches.size(), 2);
  EXPECT_GT(first->patches.front().receiver_area.lower, 1.0);

  const auto second_layer = fixture.Run(compute, LiquidHorizontalPlaneSurface3D{2.125}, {}, *workspace);
  EXPECT_EQ(second_layer.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(ReceiverFluxes(*first), initial_fluxes);
  EXPECT_EQ(first->patches.size(), 2);
  Contains(first->patches.front().receiver_area, 1.125);

  std::array<LiquidParticleState3D, 3> particles{};
  particles[0].world_position = {0, 0, .5};
  fixture.tank.low_z = .44;
  fixture.tank.high_z = .56;
  particles[1].world_position = {.994, .891, .5};
  const auto connected =
      fixture.Run(compute, LiquidParticleSphereUnion3D{1}, std::span(particles).first(2), *workspace);
  ASSERT_TRUE(connected.ok()) << connected.status();
  EXPECT_EQ(connected->attached_patch_count, 1);
  const auto connected_fluxes = ReceiverFluxes(*connected);
  ASSERT_EQ(connected_fluxes.size(), 1);
  particles[2].world_position = {.99999995, 1.08, .5};
  const auto disconnected = fixture.Run(compute, LiquidParticleSphereUnion3D{1}, particles, *workspace);
  EXPECT_EQ(disconnected.status().code(), absl::StatusCode::kFailedPrecondition);

  fixture.selected.clear();
  const auto no_owner = fixture.Run(compute, LiquidHorizontalPlaneSurface3D{1}, {}, *workspace);
  EXPECT_EQ(no_owner.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(ReceiverFluxes(*connected), connected_fluxes);
}

template <typename Compute>
void PreservesSignedLatticeAndChecksCapacity(Compute compute, int architecture) {
  auto workspace = WettedPistonWorkspace3D::Create(kCapacity);
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  constexpr double origin = 0x1p51;
  Fixture fixture;
  fixture.geometry = {{origin, origin, origin}, 1.25};
  fixture.tank = {origin - 1.25, origin + 4, origin - 1, origin - 1, origin, .25};
  // origin-1.25 lies exactly on the signed -1 plane. The next double to its
  // left lies in -2, although both reconstructed faces can be misleading.
  fixture.selected = {{-1, -1, -1}};
  for (int x : {-1, -2}) {
    if (x == -2) {
      fixture.tank.start_low_x = std::nextafter(fixture.tank.start_low_x, -std::numeric_limits<double>::infinity());
    }
    fixture.selected.front().x = x;
    const auto result = fixture.Run(compute, LiquidHorizontalPlaneSurface3D{origin}, {}, *workspace);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->compiled_architecture, architecture);
    ASSERT_EQ(result->patches.size(), 1);
    EXPECT_EQ(result->patches.front().lattice_y, -1);
    EXPECT_EQ(result->patches.front().lattice_z, -1);
    Contains(result->whole_area, 1);
    ASSERT_EQ(ReceiverFluxes(*result).size(), 1);
    EXPECT_DOUBLE_EQ(ReceiverFluxes(*result).front(), .25);
  }

  Fixture huge;
  const auto capacity = huge.Run(compute, LiquidHorizontalPlaneSurface3D{1000}, {}, *workspace);
  EXPECT_EQ(capacity.status().code(), absl::StatusCode::kResourceExhausted);
  const auto overflow =
      huge.Run(compute, LiquidHorizontalPlaneSurface3D{std::numeric_limits<double>::max()}, {}, *workspace);
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kOutOfRange);
  const auto invalid = huge.Run(compute, LiquidAnalyticSphere3D{{0, 0, 0}, 0}, {}, *workspace);
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace fluidity::wetted_piston_test
