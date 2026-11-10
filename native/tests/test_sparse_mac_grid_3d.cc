#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

#include "../cuda/liquid_boundary_donor_slots_3d.h"
#include "absl/status/status.h"
#include "fluidity/base/liquid_boundary_transfer_math_3d.h"
#include "fluidity/host/sparse_mac_grid_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr UniformMacGeometry3D kUnitGeometry{
    .world_origin = {0.0, 0.0, 0.0},
    .cell_width = 1.0,
};
constexpr std::array kCornerBlocks{
    BlockIndex3D{0, -1, 0},
    BlockIndex3D{1, -1, 0},
    BlockIndex3D{0, 0, 0},
    BlockIndex3D{1, 0, 0},
    BlockIndex3D{0, -1, 1},
    BlockIndex3D{1, -1, 1},
    BlockIndex3D{0, 0, 1},
    BlockIndex3D{1, 0, 1},
};
constexpr std::array kFaceAxes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
constexpr auto kMinIndex = std::numeric_limits<std::int32_t>::min();
constexpr auto kMaxIndex = std::numeric_limits<std::int32_t>::max();

void ExpectPosition(Vec3 actual, Vec3 expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
}

TEST(SparseMacGrid3D, CudaDonorNeighborhoodMatchesCanonicalSlotOrder) {
  const std::array blocks{
      BlockIndex3D{0, 0, 0},
      BlockIndex3D{-1, -1, -1},
      BlockIndex3D{0, -1, -1},
      BlockIndex3D{-1, 0, -1},
      BlockIndex3D{0, 0, -1},
      BlockIndex3D{-1, -1, 0},
      BlockIndex3D{-1, 0, 0},
      BlockIndex3D{kMinIndex / 4, kMinIndex / 4, kMinIndex / 4},
      BlockIndex3D{kMaxIndex / 4, kMaxIndex / 4, kMaxIndex / 4},
  };
  const auto grid = SparseMacGrid3D::Create(kUnitGeometry, blocks, blocks.size());
  ASSERT_TRUE(grid.ok()) << grid.status();
  std::vector<cuda_internal::BoundaryTransferFace3D> anchors;
  for (std::size_t face_slot = 0; face_slot < grid->face_count(); ++face_slot) {
    const auto face = grid->view().FaceIndex(face_slot);
    ASSERT_TRUE(face.ok()) << face.status();
    const auto anchor = grid->FaceWorldAnchor(*face);
    ASSERT_TRUE(anchor.ok()) << anchor.status();
    const auto resolved_face_slot = grid->FaceSlot(*face);
    ASSERT_TRUE(resolved_face_slot.ok()) << resolved_face_slot.status();
    ASSERT_EQ(*resolved_face_slot, face_slot);
    anchors.push_back({
        .axis = face->axis,
        .anchor_in_cell_widths = *anchor,
    });
  }
  for (std::size_t target_face_slot = 0; target_face_slot < anchors.size(); ++target_face_slot) {
    SCOPED_TRACE(target_face_slot);
    const Vec3 target_anchor = anchors[target_face_slot].anchor_in_cell_widths;
    for (std::size_t component_index = 0; component_index < kFaceAxes.size(); ++component_index) {
      SCOPED_TRACE(component_index);
      std::vector<std::size_t> expected_slots;
      for (std::size_t face_slot = 0; face_slot < anchors.size(); ++face_slot) {
        const Vec3 donor_anchor = anchors[face_slot].anchor_in_cell_widths;
        if (anchors[face_slot].axis == kFaceAxes[component_index] && std::abs(donor_anchor.x - target_anchor.x) <= 2
            && std::abs(donor_anchor.y - target_anchor.y) <= 2 && std::abs(donor_anchor.z - target_anchor.z) <= 2) {
          expected_slots.push_back(face_slot);
        }
      }
      cuda_internal::DonorFaceRows3D neighborhood(anchors, target_anchor, component_index);
      std::vector<std::size_t> actual_slots;
      cuda_internal::DonorFaceRow3D row;
      while (neighborhood.Next(row)) {
        ASSERT_LT(row.first_face_slot, row.past_last_face_slot);
        ASSERT_LE(row.past_last_face_slot, anchors.size());
        for (std::size_t donor_face_slot = row.first_face_slot; donor_face_slot < row.past_last_face_slot;
             ++donor_face_slot) {
          ASSERT_LT(actual_slots.size(), 125u);
          actual_slots.push_back(donor_face_slot);
        }
      }
      EXPECT_EQ(actual_slots, expected_slots);
    }
  }
}

TEST(SparseMacGrid3D, CompactGhostFitMatchesFullScratchOnRadiusTwoDonors) {
  for (const bool staggered : {false, true}) {
    SCOPED_TRACE(staggered);
    const int first_transverse_half_cell = staggered ? -3 : -4;
    std::vector<internal::LiquidGhostFitSample3D> donors;
    for (int half_z = -4; half_z <= 4; half_z += 2) {
      for (int half_y = first_transverse_half_cell; half_y <= 4; half_y += 2) {
        for (int half_x = first_transverse_half_cell; half_x <= 4; half_x += 2) {
          const int squared_half_cell_distance = half_x * half_x + half_y * half_y + half_z * half_z;
          if (squared_half_cell_distance > 16) {
            continue;
          }
          donors.push_back({
              .offset_in_cell_widths = {0.5 * half_x, 0.5 * half_y, 0.5 * half_z},
              .velocity = 0.125 * (half_x * half_x + half_y * half_z - half_x + 3 * half_z + 1),
          });
        }
      }
    }
    EXPECT_EQ(donors.size(), staggered ? 36u : 33u);
    EXPECT_LE(donors.size(), internal::kMaxRadiusTwoGhostDonors3D);
    const auto full = internal::FitLiquidGhostAffineCore3D(donors);
    const auto compact = internal::FitBoundedLiquidGhostAffineCore3D<internal::kMaxRadiusTwoGhostDonors3D>(donors);
    ASSERT_TRUE(full.ok());
    ASSERT_TRUE(compact.ok());
    ASSERT_TRUE(full->full_rank);
    ASSERT_TRUE(compact->full_rank);
    const std::array full_coefficients{
        full->velocity_at_target,
        full->velocity_slopes_in_cell_widths.x,
        full->velocity_slopes_in_cell_widths.y,
        full->velocity_slopes_in_cell_widths.z,
    };
    const std::array compact_coefficients{
        compact->velocity_at_target,
        compact->velocity_slopes_in_cell_widths.x,
        compact->velocity_slopes_in_cell_widths.y,
        compact->velocity_slopes_in_cell_widths.z,
    };
    for (std::size_t coefficient_index = 0; coefficient_index < full_coefficients.size(); ++coefficient_index) {
      EXPECT_EQ(
          std::bit_cast<std::uint64_t>(full_coefficients[coefficient_index]),
          std::bit_cast<std::uint64_t>(compact_coefficients[coefficient_index])
      );
    }
  }
}

TEST(SparseMacGrid3D, SignedFaceOwnership) {
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {1.0, 2.0, 3.0},
      .cell_width = 2.0,
  };
  constexpr std::array blocks{BlockIndex3D{0, -1, 0}, BlockIndex3D{-1, -1, 0}};
  const auto mac_grid = SparseMacGrid3D::Create(geometry, blocks, blocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  EXPECT_EQ(mac_grid->face_count(), 384u);
  const auto sorted_blocks = mac_grid->active_block_indices();
  ASSERT_EQ(sorted_blocks.size(), 2u);
  EXPECT_EQ(sorted_blocks[0].x, -1);
  EXPECT_EQ(sorted_blocks[1].x, 0);
  ExpectPosition(mac_grid->geometry().world_origin, geometry.world_origin);
  EXPECT_EQ(mac_grid->geometry().cell_width, 2.0);

  constexpr std::size_t seam_x_face_slot = 192 * 1 + 4 * 3 + 0;
  constexpr std::size_t left_x_face_slot = 192 * 0 + 4 * 3 + 3;
  static_assert(seam_x_face_slot == 204);
  static_assert(left_x_face_slot == 15);
  constexpr std::array anchor_positions{Vec3{1.0, 1.0, 4.0}, Vec3{2.0, 0.0, 4.0}, Vec3{2.0, 1.0, 3.0}};
  for (std::size_t axis_index = 0; axis_index < kFaceAxes.size(); ++axis_index) {
    const FaceNormalAxis axis = kFaceAxes[axis_index];
    const FaceIndex3D seam_face{axis, {0, -1, 0}};
    const auto seam_slot = mac_grid->FaceSlot(seam_face);
    const auto left_slot = mac_grid->FaceSlot({axis, {-1, -1, 0}});
    const auto anchor = mac_grid->FaceWorldAnchor(seam_face);
    ASSERT_TRUE(seam_slot.ok()) << seam_slot.status();
    ASSERT_TRUE(left_slot.ok()) << left_slot.status();
    ASSERT_TRUE(anchor.ok()) << anchor.status();
    EXPECT_EQ(*seam_slot, seam_x_face_slot + 64 * axis_index);
    EXPECT_EQ(*left_slot, left_x_face_slot + 64 * axis_index);
    ExpectPosition(*anchor, anchor_positions[axis_index]);
  }
  const auto cell_center = mac_grid->CellWorldCenter({-1, -1, 0});
  ASSERT_TRUE(cell_center.ok()) << cell_center.status();
  ExpectPosition(*cell_center, {0.0, 1.0, 4.0});
}

TEST(SparseMacGrid3D, RejectsInvalidLayout) {
  constexpr std::array one_block{BlockIndex3D{0, 0, 0}};
  constexpr std::array duplicates{BlockIndex3D{0, 0, 0}, BlockIndex3D{1, 0, 0}, BlockIndex3D{0, 0, 0}};
  EXPECT_EQ(SparseMacGrid3D::Create(kUnitGeometry, duplicates, 3).status().code(), absl::StatusCode::kInvalidArgument);

  UniformMacGeometry3D geometry = kUnitGeometry;
  geometry.world_origin.z = std::numeric_limits<double>::infinity();
  EXPECT_EQ(SparseMacGrid3D::Create(geometry, one_block, 1).status().code(), absl::StatusCode::kInvalidArgument);
  geometry = kUnitGeometry;
  for (double invalid_width :
       {0.0,
        -1.0,
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    geometry.cell_width = invalid_width;
    EXPECT_EQ(SparseMacGrid3D::Create(geometry, one_block, 1).status().code(), absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(SparseMacGrid3D::Create(kUnitGeometry, one_block, 0).status().code(), absl::StatusCode::kResourceExhausted);

  for (BlockIndex3D invalid_block : {BlockIndex3D{-536870913, 0, 0}, BlockIndex3D{0, 536870912, 0}}) {
    const std::array blocks{invalid_block};
    EXPECT_EQ(SparseMacGrid3D::Create(kUnitGeometry, blocks, 1).status().code(), absl::StatusCode::kOutOfRange);
  }
  const auto empty = SparseMacGrid3D::Create(kUnitGeometry, {}, 0);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(empty->face_count(), 0u);
  EXPECT_TRUE(empty->active_block_indices().empty());
  const auto empty_owner = empty->FaceSlot({FaceNormalAxis::kX, {0, 0, 0}});
  EXPECT_EQ(empty_owner.status().code(), absl::StatusCode::kFailedPrecondition);
  constexpr FaceIndex3D invalid_axis{static_cast<FaceNormalAxis>(255), {0, 0, 0}};
  EXPECT_EQ(empty->FaceSlot(invalid_axis).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(empty->FaceWorldAnchor(invalid_axis).status().code(), absl::StatusCode::kInvalidArgument);

  constexpr std::array endpoint_blocks{
      BlockIndex3D{-536870912, -536870912, -536870912},
      BlockIndex3D{536870911, 536870911, 536870911},
  };
  const auto endpoints = SparseMacGrid3D::Create(kUnitGeometry, endpoint_blocks, 2);
  ASSERT_TRUE(endpoints.ok()) << endpoints.status();
  ASSERT_EQ(endpoints->face_count(), 384u);
  const auto first = endpoints->FaceSlot({FaceNormalAxis::kX, {kMinIndex, kMinIndex, kMinIndex}});
  const auto last_in_first = endpoints->FaceSlot({FaceNormalAxis::kZ, {kMinIndex + 3, kMinIndex + 3, kMinIndex + 3}});
  const auto first_in_last = endpoints->FaceSlot({FaceNormalAxis::kX, {kMaxIndex - 3, kMaxIndex - 3, kMaxIndex - 3}});
  const auto last = endpoints->FaceSlot({FaceNormalAxis::kZ, {kMaxIndex, kMaxIndex, kMaxIndex}});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(last_in_first.ok()) << last_in_first.status();
  ASSERT_TRUE(first_in_last.ok()) << first_in_last.status();
  ASSERT_TRUE(last.ok()) << last.status();
  EXPECT_EQ(*first, 0u);
  EXPECT_EQ(*last_in_first, 191u);
  EXPECT_EQ(*first_in_last, 192u);
  EXPECT_EQ(*last, 383u);

  geometry = kUnitGeometry;
  geometry.cell_width = std::numeric_limits<double>::max();
  const auto large_grid = SparseMacGrid3D::Create(geometry, one_block, 1);
  ASSERT_TRUE(large_grid.ok()) << large_grid.status();
  EXPECT_EQ(
      large_grid->FaceWorldAnchor({FaceNormalAxis::kX, {2, 0, 0}}).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(large_grid->CellWorldCenter({2, 0, 0}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(SparseMacGrid3D, CompleteQuadraticStencilCrossesBlockCorner) {
  constexpr std::array unsorted_blocks{
      kCornerBlocks[7],
      kCornerBlocks[4],
      kCornerBlocks[3],
      kCornerBlocks[5],
      kCornerBlocks[1],
      kCornerBlocks[2],
      kCornerBlocks[0],
      kCornerBlocks[6],
  };
  const auto mac_grid = SparseMacGrid3D::Create(kUnitGeometry, unsorted_blocks, unsorted_blocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  constexpr Vec3 particle{3.75, -0.25, 3.75};
  constexpr UniformMacGeometry3D translated_geometry{
      .world_origin = {1.0, -3.0, 5.0},
      .cell_width = 2.0,
  };
  constexpr Vec3 translated_particle{8.5, -3.5, 12.5};
  const auto translated_grid = SparseMacGrid3D::Create(translated_geometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(translated_grid.ok()) << translated_grid.status();
  constexpr std::array first_indices{
      FaceLatticeIndex3D{3, -2, 2},
      FaceLatticeIndex3D{2, -1, 2},
      FaceLatticeIndex3D{2, -2, 3},
  };
  constexpr std::array shifts{Vec3{0.0, 0.5, 0.5}, Vec3{0.5, 0.0, 0.5}, Vec3{0.5, 0.5, 0.0}};
  constexpr double tolerance = 2e-12;
  for (std::size_t axis_index = 0; axis_index < kFaceAxes.size(); ++axis_index) {
    SCOPED_TRACE(axis_index);
    const FaceNormalAxis axis = kFaceAxes[axis_index];
    const auto stencil = BuildOwnedMacFaceStencil3D(*mac_grid, axis, particle, particle);
    ASSERT_TRUE(stencil.ok()) << stencil.status();
    const auto translated_stencil =
        BuildOwnedMacFaceStencil3D(*translated_grid, axis, translated_particle, translated_particle);
    ASSERT_TRUE(translated_stencil.ok()) << translated_stencil.status();
    const FaceLatticeIndex3D first_index = first_indices[axis_index];
    const Vec3 shift = shifts[axis_index];
    std::set<std::size_t> visited_slots;
    double weight_sum = 0.0;
    std::array<double, 3> first_moment{};
    std::array<double, 3> gradient_sum{};
    std::array<std::array<double, 3>, 3> second_moment{};
    std::array<std::array<double, 3>, 3> derivative_moment{};
    std::size_t sample_index = 0;
    for (std::int32_t z = first_index.z; z < first_index.z + 3; ++z) {
      for (std::int32_t y = first_index.y; y < first_index.y + 3; ++y) {
        for (std::int32_t x = first_index.x; x < first_index.x + 3; ++x) {
          const MacFaceStencilSample3D& sample = (*stencil)[sample_index];
          const MacFaceStencilSample3D& translated_sample = (*translated_stencil)[sample_index];
          ++sample_index;
          EXPECT_EQ(translated_sample.face_slot, sample.face_slot);
          EXPECT_EQ(translated_sample.weight, sample.weight);
          ExpectPosition(
              translated_sample.face_offset_from_particle,
              {2.0 * sample.face_offset_from_particle.x,
               2.0 * sample.face_offset_from_particle.y,
               2.0 * sample.face_offset_from_particle.z}
          );
          ExpectPosition(
              translated_sample.weight_gradient,
              {0.5 * sample.weight_gradient.x, 0.5 * sample.weight_gradient.y, 0.5 * sample.weight_gradient.z}
          );
          const auto face_slot = mac_grid->FaceSlot({axis, {x, y, z}});
          ASSERT_TRUE(face_slot.ok()) << face_slot.status();
          EXPECT_EQ(sample.face_slot, *face_slot);
          if (axis == FaceNormalAxis::kX && x == 4 && y == -1 && z == 2) {
            // Owner (1,-1,0) has ordinal 1. Local (0,3,2) adds 44.
            EXPECT_EQ(sample.face_slot, 236u);
          }
          if (axis == FaceNormalAxis::kX && x == 4 && y == -1 && z == 4) {
            EXPECT_EQ(sample.face_slot, 972u);
          }
          visited_slots.insert(sample.face_slot);

          const Vec3 displacement = sample.face_offset_from_particle;
          ExpectPosition(displacement, {x + shift.x - particle.x, y + shift.y - particle.y, z + shift.z - particle.z});
          const std::array offset{displacement.x, displacement.y, displacement.z};
          const Vec3 gradient = sample.weight_gradient;
          const std::array derivative{gradient.x, gradient.y, gradient.z};
          weight_sum += sample.weight;
          for (std::size_t row = 0; row < offset.size(); ++row) {
            first_moment[row] += sample.weight * offset[row];
            gradient_sum[row] += derivative[row];
            for (std::size_t column = 0; column < offset.size(); ++column) {
              second_moment[row][column] += sample.weight * offset[row] * offset[column];
              derivative_moment[row][column] += offset[row] * derivative[column];
            }
          }
        }
      }
    }
    EXPECT_EQ(visited_slots.size(), 27u);
    EXPECT_NEAR(weight_sum, 1.0, tolerance);
    for (std::size_t row = 0; row < first_moment.size(); ++row) {
      EXPECT_NEAR(first_moment[row], 0.0, tolerance);
      EXPECT_NEAR(gradient_sum[row], 0.0, tolerance);
      for (std::size_t column = 0; column < first_moment.size(); ++column) {
        EXPECT_NEAR(second_moment[row][column], row == column ? 0.25 : 0.0, tolerance);
        EXPECT_NEAR(derivative_moment[row][column], row == column ? 1.0 : 0.0, tolerance);
      }
    }
  }
}

// Evaluate the centered piecewise spline independently of the stencil's local
// floor polynomials. Finite differences below never use a derivative formula.
double CenteredQuadraticWeight(double particle_minus_anchor) {
  const double distance = std::abs(particle_minus_anchor);
  if (distance <= 0.5) {
    return 0.75 - distance * distance;
  }
  if (distance < 1.5) {
    const double tail = 1.5 - distance;
    return 0.5 * tail * tail;
  }
  return 0.0;
}

TEST(SparseMacGrid3D, ExactBasisBoundaryAndFiniteDifferenceBridge) {
  constexpr std::array normalized_positions{
      Vec3{3.75, -0.25, 3.75},
      Vec3{3.5, -0.5, 3.5},
      Vec3{4.0, 0.0, 4.0},
      Vec3{3.0, -1.0, 3.0},
  };
  constexpr double normalized_step = 1e-5;
  for (const double cell_width : {1.0, 2.0}) {
    const UniformMacGeometry3D geometry{{1.0, -3.0, 5.0}, cell_width};
    const auto mac_grid = SparseMacGrid3D::Create(geometry, kCornerBlocks, kCornerBlocks.size());
    ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
    for (const Vec3 normalized : normalized_positions) {
      const Vec3 particle_world_position{
          geometry.world_origin.x + cell_width * normalized.x,
          geometry.world_origin.y + cell_width * normalized.y,
          geometry.world_origin.z + cell_width * normalized.z,
      };
      for (const FaceNormalAxis axis : kFaceAxes) {
        const auto stencil =
            BuildOwnedMacFaceStencil3D(*mac_grid, axis, particle_world_position, particle_world_position);
        ASSERT_TRUE(stencil.ok()) << stencil.status();
        double partition = 0.0;
        std::array<double, 3> first_moment{};
        std::array<double, 3> derivative_sum{};
        std::array<std::array<double, 3>, 3> second_moment{};
        std::array<std::array<double, 3>, 3> derivative_moment{};
        for (const auto& sample : *stencil) {
          const std::array displacement{
              sample.face_offset_from_particle.x,
              sample.face_offset_from_particle.y,
              sample.face_offset_from_particle.z,
          };
          const std::array gradient{sample.weight_gradient.x, sample.weight_gradient.y, sample.weight_gradient.z};
          std::array<double, 3> particle_minus_anchor{};
          double independent_weight = 1.0;
          for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
            particle_minus_anchor[coordinate] = -displacement[coordinate] / cell_width;
            independent_weight *= CenteredQuadraticWeight(particle_minus_anchor[coordinate]);
          }
          EXPECT_NEAR(sample.weight, independent_weight, 2e-12);
          partition += sample.weight;
          for (std::size_t row = 0; row < 3; ++row) {
            double forward_weight = 1.0;
            double backward_weight = 1.0;
            for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
              const double step = coordinate == row ? normalized_step : 0.0;
              forward_weight *= CenteredQuadraticWeight(particle_minus_anchor[coordinate] + step);
              backward_weight *= CenteredQuadraticWeight(particle_minus_anchor[coordinate] - step);
            }
            const double difference_gradient =
                (forward_weight - backward_weight) / (2.0 * normalized_step * cell_width);
            // At a piece boundary the central difference has O(step) error.
            EXPECT_NEAR(gradient[row], difference_gradient, normalized_step / cell_width);
            first_moment[row] += sample.weight * displacement[row];
            derivative_sum[row] += gradient[row];
            for (std::size_t column = 0; column < 3; ++column) {
              second_moment[row][column] += sample.weight * displacement[row] * displacement[column];
              derivative_moment[row][column] += displacement[row] * gradient[column];
            }
          }
        }
        EXPECT_NEAR(partition, 1.0, 2e-12);
        for (std::size_t row = 0; row < 3; ++row) {
          EXPECT_NEAR(first_moment[row], 0.0, 2e-12);
          EXPECT_NEAR(derivative_sum[row], 0.0, 2e-12);
          for (std::size_t column = 0; column < 3; ++column) {
            EXPECT_NEAR(second_moment[row][column], row == column ? cell_width * cell_width / 4.0 : 0.0, 2e-12);
            EXPECT_NEAR(derivative_moment[row][column], row == column ? 1.0 : 0.0, 2e-12);
          }
        }
      }
    }
  }
}

TEST(SparseMacGrid3D, PreservesTranslatedSupportEdge) {
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {-3.0, 0.0, 0.0},
      .cell_width = 1.0,
  };
  constexpr std::array blocks{BlockIndex3D{0, 0, 0}};
  const auto mac_grid = SparseMacGrid3D::Create(geometry, blocks, blocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  constexpr Vec3 particle{-0x1p-53, 2.25, 2.25};
  const auto stencil = BuildOwnedMacFaceStencil3D(*mac_grid, FaceNormalAxis::kY, particle, particle);
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_EQ(stencil->front().face_slot, 85u);
  EXPECT_GT((*stencil)[12].weight, 0.0);
  EXPECT_EQ((*stencil)[12].weight_gradient.x, -0x1.e4p-55);
}

TEST(SparseMacGrid3D, PreservesNonDyadicSupportEdge) {
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {-3.3, 0.0, 0.0},
      .cell_width = 1.1,
  };
  constexpr std::array blocks{BlockIndex3D{0, 0, 0}};
  const auto mac_grid = SparseMacGrid3D::Create(geometry, blocks, blocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  // The exact world edge -3.3 + 3 * 1.1 for these stored doubles is 2^-51.
  const Vec3 particle{std::nextafter(0x1p-51, 0.0), 2.5, 2.5};
  const auto stencil = BuildOwnedMacFaceStencil3D(*mac_grid, FaceNormalAxis::kY, particle, particle);
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_EQ(stencil->front().face_slot, 85u);
  EXPECT_LT((*stencil)[12].weight_gradient.x, 0.0);
}

TEST(SparseMacGrid3D, PreservesDisplacementAtTranslatedAnchor) {
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {-2.5, 0.0, 0.0},
      .cell_width = 1.0,
  };
  constexpr std::array blocks{BlockIndex3D{0, 0, 0}};
  const auto mac_grid = SparseMacGrid3D::Create(geometry, blocks, blocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  constexpr Vec3 particle{-0x1p-53, 2.25, 2.25};
  const auto stencil = BuildOwnedMacFaceStencil3D(*mac_grid, FaceNormalAxis::kY, particle, particle);
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_EQ((*stencil)[13].face_offset_from_particle.x, 0x1p-53);
}

TEST(SparseMacGrid3D, PreservesTranslatedParticleBinEdge) {
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {-5.5, 0.0, 0.0},
      .cell_width = 1.0,
  };
  const auto bin = ComputeParticleSortBin3D(geometry, {-0x1p-53, 2.25, 2.25});
  ASSERT_TRUE(bin.ok()) << bin.status();
  EXPECT_EQ(bin->x, 0);
}

TEST(SparseMacGrid3D, FaceAnchorRetainsTranslatedWorldResidual) {
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {-0.5, 0.0, 0.0},
      .cell_width = 0.1,
  };
  constexpr std::array blocks{BlockIndex3D{1, 0, 0}};
  const auto mac_grid = SparseMacGrid3D::Create(geometry, blocks, blocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  const auto anchor = mac_grid->FaceWorldAnchor({FaceNormalAxis::kX, {5, 2, 2}});
  ASSERT_TRUE(anchor.ok()) << anchor.status();
  // For the stored cell width, the exact affine map is -0.5 + 5 * 0.1 = 2^-55.
  EXPECT_EQ(anchor->x, 0x1p-55);
  const auto stencil = BuildOwnedMacFaceStencil3D(*mac_grid, FaceNormalAxis::kY, *anchor, *anchor);
  EXPECT_TRUE(stencil.ok()) << stencil.status();
}

TEST(SparseMacGrid3D, CellCenterRetainsTranslatedWorldResidual) {
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {-0.45, 0.0, 0.0},
      .cell_width = 0.1,
  };
  const auto mac_grid = SparseMacGrid3D::Create(geometry, {}, 0);
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  const auto center = mac_grid->CellWorldCenter({4, 2, 2});
  ASSERT_TRUE(center.ok()) << center.status();
  EXPECT_EQ(center->x, 0x1p-56);
}

TEST(SparseMacGrid3D, ComputesFiniteAnchorAfterProductOverflow) {
  constexpr double largest = std::numeric_limits<double>::max();
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {-largest, 0.0, 0.0},
      .cell_width = largest,
  };
  const auto mac_grid = SparseMacGrid3D::Create(geometry, {}, 0);
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  const auto anchor = mac_grid->FaceWorldAnchor({FaceNormalAxis::kX, {2, 0, 0}});
  ASSERT_TRUE(anchor.ok()) << anchor.status();
  ExpectPosition(*anchor, {largest, largest / 2.0, largest / 2.0});
}

TEST(SparseMacGrid3D, ComputesFiniteParticleBinAfterDifferenceOverflow) {
  constexpr double largest = std::numeric_limits<double>::max();
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {-largest, 0.0, 0.0},
      .cell_width = largest,
  };
  const auto bin = ComputeParticleSortBin3D(geometry, {largest, 0.0, 0.0});
  ASSERT_TRUE(bin.ok()) << bin.status();
  EXPECT_EQ(bin->x, 0);
}

TEST(SparseMacGrid3D, PreservesRepresentableGradientWithUnderflowingAxisWeight) {
  constexpr double cell_width = 0x1p-300;
  constexpr UniformMacGeometry3D geometry{
      .world_origin = {0.0, 0.0, 0.0},
      .cell_width = cell_width,
  };
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
  const auto mac_grid = SparseMacGrid3D::Create(geometry, blocks, blocks.size());
  ASSERT_TRUE(mac_grid.ok()) << mac_grid.status();
  constexpr Vec3 particle{0x1p-900, cell_width / 4.0, cell_width / 4.0};
  const auto stencil = BuildOwnedMacFaceStencil3D(*mac_grid, FaceNormalAxis::kZ, particle, particle);
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  // At (1,-1,0), wx = 2^-1201, dwy = -3/4, wz = 11/16, and 1/h = 2^300.
  EXPECT_EQ((*stencil)[11].weight_gradient.y, -0x1.08p-902);
}

TEST(SparseMacGrid3D, RejectsIncompleteOrUnrepresentableSupport) {
  constexpr std::array incomplete_blocks{
      kCornerBlocks[0],
      kCornerBlocks[2],
      kCornerBlocks[3],
      kCornerBlocks[4],
      kCornerBlocks[5],
      kCornerBlocks[6],
      kCornerBlocks[7],
  };
  const auto incomplete = SparseMacGrid3D::Create(kUnitGeometry, incomplete_blocks, incomplete_blocks.size());
  const auto complete = SparseMacGrid3D::Create(kUnitGeometry, kCornerBlocks, kCornerBlocks.size());
  ASSERT_TRUE(incomplete.ok()) << incomplete.status();
  ASSERT_TRUE(complete.ok()) << complete.status();
  constexpr Vec3 zero_weight_edge{3.0, -0.25, 3.75};
  const auto edge_stencil =
      BuildOwnedMacFaceStencil3D(*complete, FaceNormalAxis::kY, zero_weight_edge, zero_weight_edge);
  ASSERT_TRUE(edge_stencil.ok()) << edge_stencil.status();
  const auto zero_weight_slot = complete->FaceSlot({FaceNormalAxis::kY, {4, -1, 2}});
  ASSERT_TRUE(zero_weight_slot.ok()) << zero_weight_slot.status();
  EXPECT_EQ((*edge_stencil)[2].face_slot, *zero_weight_slot);
  EXPECT_EQ((*edge_stencil)[2].weight, 0.0);
  EXPECT_EQ(
      BuildOwnedMacFaceStencil3D(*incomplete, FaceNormalAxis::kY, zero_weight_edge, zero_weight_edge).status().code(),
      absl::StatusCode::kFailedPrecondition
  );

  const double infinity = std::numeric_limits<double>::infinity();
  struct BinCase {
    double sorted_x;
    std::int32_t expected_bin_x;
  };
  const std::array bin_cases{
      BinCase{-4.0, -2},
      BinCase{std::nextafter(-2.5, -infinity), -2},
      BinCase{-2.5, -1},
      BinCase{std::nextafter(-2.5, infinity), -1},
      BinCase{0.0, -1},
      BinCase{std::nextafter(1.5, -infinity), -1},
      BinCase{1.5, 0},
      BinCase{std::nextafter(1.5, infinity), 0},
      BinCase{std::nextafter(5.5, -infinity), 0},
      BinCase{5.5, 1},
      BinCase{std::nextafter(5.5, infinity), 1},
  };
  for (const BinCase& bin_case : bin_cases) {
    const auto bin = ComputeParticleSortBin3D(kUnitGeometry, {bin_case.sorted_x, -4.0, 0.0});
    ASSERT_TRUE(bin.ok()) << bin.status();
    EXPECT_EQ(bin->x, bin_case.expected_bin_x);
    EXPECT_EQ(bin->y, -2);
    EXPECT_EQ(bin->z, -1);
  }

  constexpr std::array seam_blocks{
      BlockIndex3D{-2, -1, 0},
      BlockIndex3D{-1, -1, 0},
      BlockIndex3D{0, -1, 0},
  };
  const auto seam_grid = SparseMacGrid3D::Create(kUnitGeometry, seam_blocks, seam_blocks.size());
  ASSERT_TRUE(seam_grid.ok()) << seam_grid.status();
  for (double x : {-4.0, std::nextafter(-4.0, -infinity), std::nextafter(-4.0, infinity), 0.0}) {
    const Vec3 particle_world_position{x, -1.75, 1.75};
    const auto stencil =
        BuildOwnedMacFaceStencil3D(*seam_grid, FaceNormalAxis::kX, particle_world_position, particle_world_position);
    EXPECT_TRUE(stencil.ok()) << stencil.status();
  }

  const Vec3 below_normal_tie{std::nextafter(-0.5, -infinity), -1.75, 1.75};
  const auto below_normal_stencil =
      BuildOwnedMacFaceStencil3D(*seam_grid, FaceNormalAxis::kX, below_normal_tie, below_normal_tie);
  ASSERT_TRUE(below_normal_stencil.ok()) << below_normal_stencil.status();
  const auto first_normal_tie_slot = seam_grid->FaceSlot({FaceNormalAxis::kX, {-2, -3, 0}});
  ASSERT_TRUE(first_normal_tie_slot.ok()) << first_normal_tie_slot.status();
  EXPECT_EQ(below_normal_stencil->front().face_slot, *first_normal_tie_slot);
  EXPECT_GT(below_normal_stencil->front().weight, 0.0);

  const Vec3 below_transverse_tie{std::nextafter(0.0, -infinity), -1.75, 1.75};
  const auto below_transverse_stencil =
      BuildOwnedMacFaceStencil3D(*seam_grid, FaceNormalAxis::kY, below_transverse_tie, below_transverse_tie);
  ASSERT_TRUE(below_transverse_stencil.ok()) << below_transverse_stencil.status();
  const auto first_transverse_tie_slot = seam_grid->FaceSlot({FaceNormalAxis::kY, {-2, -3, 0}});
  ASSERT_TRUE(first_transverse_tie_slot.ok()) << first_transverse_tie_slot.status();
  EXPECT_EQ(below_transverse_stencil->front().face_slot, *first_transverse_tie_slot);

  for (double trial_x : {2.1, 5.9}) {
    const Vec3 particle_world_position{trial_x, -0.25, 3.75};
    const auto stencil =
        BuildOwnedMacFaceStencil3D(*complete, FaceNormalAxis::kX, particle_world_position, particle_world_position);
    EXPECT_TRUE(stencil.ok()) << stencil.status();
  }
  const auto crosses_bin_edge = BuildOwnedMacFaceStencil3D(
      *complete,
      FaceNormalAxis::kX,
      {std::nextafter(5.5, -infinity), -0.25, 3.75},
      {5.9, -0.25, 3.75}
  );
  EXPECT_TRUE(crosses_bin_edge.ok()) << crosses_bin_edge.status();

  constexpr Vec3 sorted{4.0, -0.25, 3.75};
  for (double trial_x : {3.5, 4.5}) {
    const auto stencil = BuildOwnedMacFaceStencil3D(*complete, FaceNormalAxis::kX, sorted, {trial_x, -0.25, 3.75});
    EXPECT_TRUE(stencil.ok()) << stencil.status();
  }
  for (double trial_x : {std::nextafter(3.5, -infinity), std::nextafter(4.5, infinity)}) {
    const auto stencil = BuildOwnedMacFaceStencil3D(*complete, FaceNormalAxis::kX, sorted, {trial_x, -0.25, 3.75});
    EXPECT_EQ(stencil.status().code(), absl::StatusCode::kFailedPrecondition);
  }
  const auto rounded_motion = BuildOwnedMacFaceStencil3D(
      *seam_grid,
      FaceNormalAxis::kX,
      {0.5, -1.75, 1.75},
      {std::nextafter(0.0, -infinity), -1.75, 1.75}
  );
  EXPECT_EQ(rounded_motion.status().code(), absl::StatusCode::kFailedPrecondition);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto invalid_axis = BuildOwnedMacFaceStencil3D(*complete, static_cast<FaceNormalAxis>(255), sorted, sorted);
  EXPECT_EQ(invalid_axis.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ComputeParticleSortBin3D(kUnitGeometry, {nan, 0.0, 0.0}).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      BuildOwnedMacFaceStencil3D(*complete, FaceNormalAxis::kX, sorted, {nan, 0.0, 0.0}).status().code(),
      absl::StatusCode::kInvalidArgument
  );
  const Vec3 huge_position{std::numeric_limits<double>::max(), -0.25, 3.75};
  EXPECT_EQ(ComputeParticleSortBin3D(kUnitGeometry, huge_position).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(
      BuildOwnedMacFaceStencil3D(*complete, FaceNormalAxis::kX, huge_position, huge_position).status().code(),
      absl::StatusCode::kOutOfRange
  );

  constexpr double smallest_bin_edge = 4.0 * static_cast<double>(kMinIndex) + 1.5;
  constexpr double largest_bin_end = 4.0 * (static_cast<double>(kMaxIndex) + 1.0) + 1.5;
  const auto extreme_bins =
      ComputeParticleSortBin3D(kUnitGeometry, {smallest_bin_edge, std::nextafter(largest_bin_end, -infinity), 0.0});
  ASSERT_TRUE(extreme_bins.ok()) << extreme_bins.status();
  EXPECT_EQ(extreme_bins->x, kMinIndex);
  EXPECT_EQ(extreme_bins->y, kMaxIndex);
  EXPECT_EQ(
      ComputeParticleSortBin3D(kUnitGeometry, {std::nextafter(smallest_bin_edge, -infinity), 0.0, 0.0}).status().code(),
      absl::StatusCode::kOutOfRange
  );
  EXPECT_EQ(
      ComputeParticleSortBin3D(kUnitGeometry, {largest_bin_end, 0.0, 0.0}).status().code(),
      absl::StatusCode::kOutOfRange
  );

  constexpr double sorted_endpoint = static_cast<double>(kMinIndex) + 0.5;
  constexpr double trial_endpoint = static_cast<double>(kMinIndex) + 1.0;
  constexpr Vec3 sorted_min{sorted_endpoint, sorted_endpoint, sorted_endpoint};
  constexpr Vec3 trial_min{trial_endpoint, trial_endpoint, trial_endpoint};
  const auto bin = ComputeParticleSortBin3D(kUnitGeometry, sorted_min);
  ASSERT_TRUE(bin.ok()) << bin.status();
  EXPECT_EQ(bin->x, -536870913);
  EXPECT_EQ(bin->y, -536870913);
  EXPECT_EQ(bin->z, -536870913);
  constexpr std::array first_owner{BlockIndex3D{-536870912, -536870912, -536870912}};
  const auto first_grid = SparseMacGrid3D::Create(kUnitGeometry, first_owner, first_owner.size());
  ASSERT_TRUE(first_grid.ok()) << first_grid.status();
  for (FaceNormalAxis axis : kFaceAxes) {
    const auto stencil = BuildOwnedMacFaceStencil3D(*first_grid, axis, sorted_min, trial_min);
    EXPECT_TRUE(stencil.ok()) << stencil.status();
  }
  const auto below_first_face = BuildOwnedMacFaceStencil3D(*first_grid, FaceNormalAxis::kY, sorted_min, sorted_min);
  EXPECT_EQ(below_first_face.status().code(), absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace fluidity
