#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

#include "../src/graded_sampled_liquid_boundary_geometry_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "gtest/gtest.h"

namespace fluidity::pressure_test {

template <typename Project>
void CheckCurvedMatchedPressure(Project project) {
  constexpr double kWidth = 0.25;
  constexpr double kDuration = 0.5;
  constexpr double kExactVolume = 19.0 / 3.0;
  constexpr double kVolumeErrorLimit = 0.011;
  constexpr double kPressureErrorLimit = 1e-4;
  constexpr double kDefectLimit = 2e-5;
  constexpr double kFluxErrorLimit = 2e-6;
  const auto exact_pressure = [](double z) {
    return 1 + 0.5 * z;
  };
  std::array<double, 2> volume_errors{};
  std::array<std::size_t, 2> row_counts{};
  for (const bool graded : {false, true}) {
    SCOPED_TRACE(graded);
    std::vector<AdaptiveMacCellKey3D> keys;
    for (int z = -4; z < 4; ++z) {
      for (int y = -4; y < 4; ++y) {
        for (int x = -4; x < 4; ++x) {
          const bool coarse = graded && z < 0;
          if (!coarse || (x % 2 == 0 && y % 2 == 0 && z % 2 == 0)) {
            keys.push_back({{x, y, z}, static_cast<std::uint8_t>(coarse)});
          }
        }
      }
    }
    const auto topology = AdaptiveMacTopology3D::Create(keys, kWidth);
    ASSERT_TRUE(topology.ok()) << topology.status();
    const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
    ASSERT_TRUE(ownership.ok()) << ownership.status();
    const auto add_corners = [](auto& roster, auto lower, int width, int normal_axis) {
      for (int bits = 0; bits < 8; ++bits) {
        if (normal_axis >= 0 && (bits & (1 << normal_axis)) != 0) {
          continue;
        }
        const int x = lower.x + ((bits & 1) != 0 ? width : 0);
        const int y = lower.y + ((bits & 2) != 0 ? width : 0);
        const int z = lower.z + ((bits & 4) != 0 ? width : 0);
        const double world_x = kWidth * x;
        const double world_y = kWidth * y;
        const double height = 0.5 + (world_x * world_x + world_y * world_y) / 8;
        roster.try_emplace(std::tuple{z, y, x}, GradedLiquidCornerSample3D{{x, y, z}, kWidth * z - height, 1});
      }
    };
    std::map<std::tuple<int, int, int>, GradedLiquidCornerSample3D> roster;
    std::vector<std::uint8_t> air;
    std::vector<LiquidAdaptiveCellState3D> accepted;
    for (const auto& leaf : topology->cells()) {
      const auto key = leaf.key;
      add_corners(roster, key.lower_corner, 1 << key.width_exponent, -1);
      const bool dry = key.lower_corner.z == 3;
      air.push_back(dry);
      accepted.push_back({.key = key, .occupancy = dry ? LiquidCellOccupancy3D::kAir : LiquidCellOccupancy3D::kLiquid});
    }
    for (const auto& face : topology->interior_faces()) {
      add_corners(
          roster,
          face.face_index.lattice_index,
          1 << face.width_exponent,
          static_cast<int>(face.face_index.axis)
      );
    }
    for (const auto& face : ownership->exterior_faces) {
      add_corners(
          roster,
          face.face_index.lattice_index,
          1 << face.width_exponent,
          static_cast<int>(face.face_index.axis)
      );
    }
    std::vector<GradedLiquidCornerSample3D> samples;
    for (const auto& [key, sample] : roster) {
      samples.push_back(sample);
    }
    const auto geometry = FreezeGradedSampledLiquidGeometry3D(*topology, {}, kDuration, samples, air);
    ASSERT_TRUE(geometry.ok()) << geometry.status();
    std::vector<GradedLiquidPressureCellFields3D> cell_fields(air.size());
    std::vector<GradedLiquidPressureSubfaceFields3D> face_fields(geometry->subfaces().size());
    std::size_t mixed_links = 0;
    const auto center_z = [&](std::size_t ordinal) {
      const auto& leaf = topology->cells()[ordinal];
      return kWidth * leaf.key.lower_corner.z + 0.5 * leaf.cell_width;
    };
    for (std::size_t ordinal = 0; ordinal < topology->interior_faces().size(); ++ordinal) {
      const auto& face = topology->interior_faces()[ordinal];
      const auto slot = ownership->interior_face_slots[ordinal];
      if (geometry->subfaces()[slot].wet_open_area_m2 == 0) {
        continue;
      }
      const auto tail = face.tail_cell_ordinal;
      const auto head = face.head_cell_ordinal;
      mixed_links += topology->cells()[tail].key.width_exponent != topology->cells()[head].key.width_exponent;
      face_fields[slot].inverse_density_m3_per_kg = 1;
      face_fields[slot].predicted_normal_velocity_m_per_s =
          kDuration * (exact_pressure(center_z(head)) - exact_pressure(center_z(tail))) / face.cell_center_distance;
    }
    for (const auto& face : ownership->exterior_faces) {
      if (face.face_index.axis != FaceNormalAxis::kZ || face.face_index.lattice_index.z != -4) {
        continue;
      }
      const double distance_m = 0.5 * topology->cells()[face.cell_ordinal].cell_width;
      face_fields[face.face_slot] = {
          .inverse_density_m3_per_kg = 1,
          .predicted_normal_velocity_m_per_s = kDuration * 0.5,
          .exterior_pressure =
              GradedLiquidPressureExterior3D{
                  .pressure_pa = exact_pressure(-1),
                  .distance_m = distance_m,
              },
      };
    }
    EXPECT_EQ(mixed_links, graded ? 64 : 0);
    const auto subset = PrepareGradedWetPressureSubset3D(*geometry, accepted, cell_fields, face_fields);
    ASSERT_TRUE(subset.ok()) << subset.status();
    const auto& system = subset->system;
    const auto solved = project(system);
    ASSERT_TRUE(solved.ok()) << solved.status();
    double volume = 0;
    double pressure_error = 0;
    double flux_error = 0;
    for (std::size_t row = 0; row < system.cells().size(); ++row) {
      volume += system.cells()[row].fluid_volume;
      pressure_error = std::max(
          pressure_error,
          std::abs(solved->cell_pressures[row] - exact_pressure(center_z(subset->leaf_ordinals[row])))
      );
    }
    for (std::size_t link = 0; link < system.links().size(); ++link) {
      flux_error =
          std::max(flux_error, std::abs(system.links()[link].open_area * solved->corrected_face_velocities[link]));
    }
    volume_errors[graded] = std::abs(volume - kExactVolume);
    row_counts[graded] = system.cells().size();
    EXPECT_LE(volume_errors[graded], kVolumeErrorLimit);
    EXPECT_LE(pressure_error, kPressureErrorLimit);
    EXPECT_LE(solved->diagnostics.max_cell_constraint_defect, kDefectLimit);
    EXPECT_LE(flux_error, kFluxErrorLimit);
    std::cout << "curved graded=" << graded << " leaves=" << keys.size() << " wet_rows=" << system.cells().size()
              << " links=" << system.links().size() << " mixed_links=" << mixed_links
              << " volume_error_m3=" << volume_errors[graded] << " pressure_error_pa=" << pressure_error
              << " flux_error_m3_per_s=" << flux_error
              << " defect_per_s=" << solved->diagnostics.max_cell_constraint_defect << '\n';
  }
  EXPECT_NEAR(volume_errors[0], volume_errors[1], 1e-12);
  EXPECT_LT(row_counts[1], row_counts[0]);
}

}  // namespace fluidity::pressure_test
