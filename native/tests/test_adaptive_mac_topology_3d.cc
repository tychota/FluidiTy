#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

#include "../src/mac_pressure_3d_system.h"
#include "absl/status/status.h"
#include "fluidity/host/adaptive_mac_topology_3d.h"
#include "fluidity/host/mac_pressure_workspace_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

std::vector<AdaptiveMacCellKey3D> MakeJunction(std::size_t plane_count, bool high_side) {
  constexpr std::int32_t kLower = -4;
  std::vector<AdaptiveMacCellKey3D> keys;
  for (std::size_t axis = 0; axis < plane_count; ++axis) {
    std::array lower{kLower, kLower, kLower};
    lower[axis] += high_side ? 2 : -2;
    keys.push_back({.lower_corner = {lower[0], lower[1], lower[2]}, .width_exponent = 1});
  }
  for (std::int32_t z = kLower; z < kLower + 2; ++z) {
    for (std::int32_t y = kLower; y < kLower + 2; ++y) {
      for (std::int32_t x = kLower; x < kLower + 2; ++x) {
        keys.push_back({.lower_corner = {x, y, z}, .width_exponent = 0});
      }
    }
  }
  return keys;
}

auto FaceKey(const AdaptiveMacInteriorFace3D& face) {
  const auto& index = face.face_index.lattice_index;
  return std::tuple{face.face_index.axis, index.z, index.y, index.x, face.width_exponent};
}

double AffinePressure(const AdaptiveMacLeaf3D& cell) {
  const auto& index = cell.key.lower_corner;
  return (index.x + 2.0 * index.y - 0.5 * index.z) / 16.0 + cell.cell_width * 2.5 / 16.0;
}

TEST(AdaptiveMacTopology3D, OwnsSignedEdgeAndCornerSubfaces) {
  for (const std::size_t plane_count : {2, 3}) {
    for (const bool high_side : {false, true}) {
      SCOPED_TRACE(plane_count);
      SCOPED_TRACE(high_side);
      auto keys = MakeJunction(plane_count, high_side);
      const auto topology = AdaptiveMacTopology3D::Create(keys, 0.5);
      ASSERT_TRUE(topology.ok()) << topology.status();
      const auto cells = topology->cells();
      const auto faces = topology->interior_faces();
      ASSERT_EQ(cells.size(), 8 + plane_count);
      ASSERT_EQ(faces.size(), 12 + 4 * plane_count);

      std::array<std::size_t, 3> mixed_face_counts{};
      for (const auto& face : faces) {
        const auto& tail = cells[face.tail_cell_ordinal];
        const auto& head = cells[face.head_cell_ordinal];
        EXPECT_EQ(face.width_exponent, 0);
        EXPECT_DOUBLE_EQ(face.face_area, 0.25);
        if (tail.key.width_exponent != head.key.width_exponent) {
          EXPECT_DOUBLE_EQ(face.cell_center_distance, 0.75);
          EXPECT_EQ(tail.key.width_exponent, high_side ? 0 : 1);
          EXPECT_EQ(head.key.width_exponent, high_side ? 1 : 0);
          ++mixed_face_counts[static_cast<std::size_t>(face.face_index.axis)];
        } else {
          EXPECT_DOUBLE_EQ(face.cell_center_distance, 0.5);
        }
      }
      for (std::size_t axis = 0; axis < mixed_face_counts.size(); ++axis) {
        EXPECT_EQ(mixed_face_counts[axis], axis < plane_count ? 4 : 0);
      }

      std::ranges::reverse(keys);
      const auto reversed = AdaptiveMacTopology3D::Create(keys, 0.5);
      ASSERT_TRUE(reversed.ok()) << reversed.status();
      for (std::size_t slot = 0; slot < faces.size(); ++slot) {
        if (slot != 0) {
          EXPECT_LT(FaceKey(faces[slot - 1]), FaceKey(faces[slot]));
        }
        EXPECT_EQ(FaceKey(faces[slot]), FaceKey(reversed->interior_faces()[slot]));
        EXPECT_EQ(faces[slot].tail_cell_ordinal, reversed->interior_faces()[slot].tail_cell_ordinal);
        EXPECT_EQ(faces[slot].head_cell_ordinal, reversed->interior_faces()[slot].head_cell_ordinal);
      }
    }
  }
}

TEST(AdaptiveMacTopology3D, SolvesPressureOnOwnedCoarseFineJunction) {
  const auto keys = MakeJunction(3, false);
  const auto topology = AdaptiveMacTopology3D::Create(keys, 0.5);
  ASSERT_TRUE(topology.ok()) << topology.status();
  std::vector<MacPressureCell3D> cells;
  std::vector<double> reference_pressure;
  for (const auto& cell : topology->cells()) {
    cells.push_back({
        .cell_index = cell.key.lower_corner,
        .fluid_volume = cell.cell_volume,
        .target_divergence = 0.0,
        .bulk_modulus = std::numeric_limits<double>::infinity(),
        .previous_pressure = 0.0,
    });
    reference_pressure.push_back(AffinePressure(cell));
  }
  std::vector<pressure_internal::ExplicitMacPressureFace3D> faces;
  for (const auto& face : topology->interior_faces()) {
    const double pressure_difference =
        reference_pressure[face.head_cell_ordinal] - reference_pressure[face.tail_cell_ordinal];
    faces.push_back({
        .face_slot = faces.size(),
        .tail_cell_ordinal = face.tail_cell_ordinal,
        .head_cell_ordinal = face.head_cell_ordinal,
        .open_area = face.face_area,
        .inverse_density = 1.0,
        .pressure_gradient_distance = face.cell_center_distance,
        .predicted_normal_velocity = pressure_difference / face.cell_center_distance,
    });
  }

  const auto system = pressure_internal::PrepareExplicitMacPressureSystem3D(1.0, cells, faces);
  ASSERT_TRUE(system.ok()) << system.status();
  auto workspace = MacPressureWorkspace3D::Create(cells.size(), faces.size());
  ASSERT_TRUE(workspace.ok()) << workspace.status();
  const auto projection = workspace->ProjectPrepared(*system);
  ASSERT_TRUE(projection.ok()) << projection.status();
  ASSERT_EQ(projection->gauge_cell_ordinals.size(), 1);
  const double gauge_pressure = reference_pressure[projection->gauge_cell_ordinals.front()];
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    EXPECT_NEAR(projection->cell_pressures[cell], reference_pressure[cell] - gauge_pressure, 2e-5);
  }
  EXPECT_LE(projection->diagnostics.max_cell_constraint_defect, 2e-5);
}

TEST(AdaptiveMacTopology3D, MatchesIndependentPairwiseGeometryWithHoles) {
  std::vector<AdaptiveMacCellKey3D> keys;
  for (std::int32_t coarse_z = -2; coarse_z < 2; ++coarse_z) {
    for (std::int32_t coarse_y = -2; coarse_y < 2; ++coarse_y) {
      for (std::int32_t coarse_x = -2; coarse_x < 2; ++coarse_x) {
        const std::int32_t selector = coarse_x + 2 * coarse_y + 3 * coarse_z;
        if (selector % 5 == 0) {
          continue;
        }
        if (selector % 2 == 0) {
          keys.push_back({.lower_corner = {2 * coarse_x, 2 * coarse_y, 2 * coarse_z}, .width_exponent = 1});
          continue;
        }
        for (std::int32_t child_z = 0; child_z < 2; ++child_z) {
          for (std::int32_t child_y = 0; child_y < 2; ++child_y) {
            for (std::int32_t child_x = 0; child_x < 2; ++child_x) {
              if (child_x + child_y + child_z != 2) {
                keys.push_back({
                    .lower_corner = {2 * coarse_x + child_x, 2 * coarse_y + child_y, 2 * coarse_z + child_z},
                    .width_exponent = 0,
                });
              }
            }
          }
        }
      }
    }
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, 0.5);
  ASSERT_TRUE(topology.ok()) << topology.status();
  const auto cells = topology->cells();
  using FaceKeyAndEndpoints = std::tuple<FaceNormalAxis, int, int, int, std::uint8_t, std::size_t, std::size_t>;
  std::set<FaceKeyAndEndpoints> expected;
  for (std::size_t tail_ordinal = 0; tail_ordinal < cells.size(); ++tail_ordinal) {
    const auto& tail = cells[tail_ordinal].key;
    const std::array tail_lower{tail.lower_corner.x, tail.lower_corner.y, tail.lower_corner.z};
    const int tail_width = 1 << tail.width_exponent;
    for (std::size_t head_ordinal = 0; head_ordinal < cells.size(); ++head_ordinal) {
      const auto& head = cells[head_ordinal].key;
      const std::array head_lower{head.lower_corner.x, head.lower_corner.y, head.lower_corner.z};
      const int head_width = 1 << head.width_exponent;
      for (std::size_t normal = 0; normal < 3; ++normal) {
        if (tail_lower[normal] + tail_width != head_lower[normal]) {
          continue;
        }
        std::array<int, 3> corner = head_lower;
        bool shares_area = true;
        for (std::size_t transverse = 0; transverse < 3; ++transverse) {
          if (transverse == normal) {
            continue;
          }
          corner[transverse] = std::max(tail_lower[transverse], head_lower[transverse]);
          const int high = std::min(tail_lower[transverse] + tail_width, head_lower[transverse] + head_width);
          shares_area = shares_area && corner[transverse] < high;
        }
        if (shares_area) {
          expected.emplace(
              static_cast<FaceNormalAxis>(normal),
              corner[2],
              corner[1],
              corner[0],
              std::min(tail.width_exponent, head.width_exponent),
              tail_ordinal,
              head_ordinal
          );
        }
      }
    }
  }
  std::set<FaceKeyAndEndpoints> actual;
  for (const auto& face : topology->interior_faces()) {
    const auto& index = face.face_index.lattice_index;
    EXPECT_TRUE(actual
                    .emplace(
                        face.face_index.axis,
                        index.z,
                        index.y,
                        index.x,
                        face.width_exponent,
                        face.tail_cell_ordinal,
                        face.head_cell_ordinal
                    )
                    .second);
  }
  EXPECT_EQ(actual, expected);
}

TEST(AdaptiveMacTopology3D, RejectsOverlapMisalignmentOverflowAndUngradedContacts) {
  using absl::StatusCode;
  const std::array nested{AdaptiveMacCellKey3D{{0, 0, 0}, 2}, AdaptiveMacCellKey3D{{1, 1, 1}, 0}};
  const std::array duplicate{AdaptiveMacCellKey3D{{0, 0, 0}, 0}, AdaptiveMacCellKey3D{{0, 0, 0}, 0}};
  const std::array misaligned{AdaptiveMacCellKey3D{{-3, 0, 0}, 1}};
  const std::array overflow{AdaptiveMacCellKey3D{{std::numeric_limits<std::int32_t>::max(), 0, 0}, 0}};
  EXPECT_EQ(AdaptiveMacTopology3D::Create(nested, 0.5).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(AdaptiveMacTopology3D::Create(duplicate, 0.5).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(AdaptiveMacTopology3D::Create(misaligned, 0.5).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(AdaptiveMacTopology3D::Create(overflow, 0.5).status().code(), StatusCode::kOutOfRange);

  for (const CellIndex3D fine_lower : {CellIndex3D{4, 0, 0}, CellIndex3D{4, 4, 0}, CellIndex3D{4, 4, 4}}) {
    const std::array ungraded{AdaptiveMacCellKey3D{{0, 0, 0}, 2}, AdaptiveMacCellKey3D{fine_lower, 0}};
    EXPECT_EQ(AdaptiveMacTopology3D::Create(ungraded, 0.5).status().code(), StatusCode::kFailedPrecondition);
  }
}

}  // namespace
}  // namespace fluidity
