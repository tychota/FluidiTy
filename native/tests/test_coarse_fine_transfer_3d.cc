#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "fluidity/host/coarse_fine_transfer_3d.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

constexpr Vec3 kOrigin{-4, 2, -3};

auto OwnerKey(const CoarseFineStencilRow3D& row) {
  const auto [x, y, z] = row.owner.lattice_index;
  return std::tuple{static_cast<int>(row.level), x, y, z};
}

TEST(CoarseFineTransfer3D, ReproducesAffineFieldsAcrossOracleQueries) {
  constexpr std::array<std::array<double, 3>, 10> local_positions{{
      {-2.25, -2.25, -2.25},
      {0, 0, 0},
      {-0.125, 0.1875, -0.375},
      {0.375, -0.125, 0.625},
      {0.625, 0.375, -0.125},
      {0.875, 0.75, 0.625},
      {1, 1, 1},
      {1.25, 0.875, 1.125},
      {1.375, 1.625, 1.125},
      {2.25, 2.25, 2.25},
  }};
  constexpr std::array<float, 3> slopes{0.375f, -0.25f, 0.5f};
  std::array<float, 6> peaks{};
  for (int codimension = 0; codimension <= 3; ++codimension) {
    int mixed_queries = 0;
    for (int rotation = 0; rotation < 2; ++rotation) {
      const std::array signs = rotation == 0 ? std::array{1, 1, 1} : std::array{-1, 1, -1};
      std::array<int, 3> fine_side{};
      for (int axis = 0; axis < codimension; ++axis) {
        const int world_axis = (axis + rotation) % 3;
        fine_side[world_axis] = signs[world_axis];
      }
      for (const auto normal : {FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ}) {
        for (const double width : {0.5, 2.0}) {
          for (const auto& local : local_positions) {
            const Vec3 position{
                kOrigin.x + width * signs[0] * local[(3 - rotation) % 3],
                kOrigin.y + width * signs[1] * local[(4 - rotation) % 3],
                kOrigin.z + width * signs[2] * local[(5 - rotation) % 3],
            };
            const auto stencil = EvaluateCoarseFineStencil3D({kOrigin, width, fine_side}, normal, position);
            ASSERT_TRUE(stencil.ok()) << stencil.status();
            const auto rows = stencil->rows();
            ASSERT_FALSE(rows.empty());
            mixed_queries += rows.front().level != rows.back().level;
            float partition = 0;
            float affine_value = 0;
            std::array<float, 3> first_moment{};
            std::array<float, 3> gradient_sum{};
            std::array<float, 3> affine_gradient{};
            std::array<std::array<float, 3>, 3> gradient_moment{};
            for (std::size_t i = 0; i < rows.size(); ++i) {
              const auto& row = rows[i];
              ASSERT_GT(row.weight, 0);
              EXPECT_EQ(row.owner.axis, normal);
              if (i > 0) {
                EXPECT_LT(OwnerKey(rows[i - 1]), OwnerKey(row));
              }
              float sample = 0.25f;
              for (int axis = 0; axis < 3; ++axis) {
                sample += slopes[axis] * row.offset_coarse_units[axis];
              }
              partition += row.weight;
              affine_value += row.weight * sample;
              for (int axis = 0; axis < 3; ++axis) {
                const float gradient = row.gradient_coarse_units[axis];
                first_moment[axis] += row.weight * row.offset_coarse_units[axis];
                gradient_sum[axis] += gradient;
                affine_gradient[axis] += sample * gradient;
                for (int component = 0; component < 3; ++component) {
                  gradient_moment[component][axis] += row.offset_coarse_units[component] * gradient;
                }
              }
            }
            const auto check = [&](int metric, float defect, float limit) {
              ASSERT_TRUE(std::isfinite(defect));
              EXPECT_LE(std::abs(defect), limit);
              peaks[metric] = std::max(peaks[metric], std::abs(defect));
            };
            check(0, partition - 1, 1e-6f);
            check(4, affine_value - 0.25f, 1e-6f);
            for (int axis = 0; axis < 3; ++axis) {
              check(1, first_moment[axis], 1e-6f);
              check(2, gradient_sum[axis], 2e-6f);
              check(5, affine_gradient[axis] - slopes[axis], 2e-6f);
              for (int component = 0; component < 3; ++component) {
                check(3, gradient_moment[component][axis] - (axis == component ? 1.0f : 0.0f), 2e-6f);
              }
            }
          }
        }
      }
    }
    constexpr std::array expected_mixed_queries{0, 48, 48, 44};
    EXPECT_EQ(mixed_queries, expected_mixed_queries[codimension]);
  }
  constexpr std::array names{
      "partition",
      "first_moment",
      "gradient_sum",
      "affine_gradient_moment",
      "affine_value",
      "affine_gradient",
  };
  for (std::size_t metric = 0; metric < peaks.size(); ++metric) {
    RecordProperty(names[metric], std::to_string(peaks[metric] * 1e9f) + "e-9");
  }
}

TEST(CoarseFineTransfer3D, MatchesMeasuredEdgeAndSignedCornerRows) {
  struct OracleRow {
    std::array<int, 3> fine_side;
    FaceNormalAxis normal;
    Vec3 position;
    int coarse_count;
    int fine_count;
    int level;
    FaceLatticeIndex3D owner;
    float weight;
    std::array<float, 3> gradient;
  };
  constexpr std::array<OracleRow, 3> cases{{
      {
          .fine_side = {1, 1, 0},
          .normal = FaceNormalAxis::kX,
          .position = {-3.375, 2.4375, -2.4375},
          .coarse_count = 30,
          .fine_count = 18,
          .level = 1,
          .owner = {2, 1, 2},
          .weight = 0.22648111f,
          .gradient = {-0.59082031f, -0.32942706f, 0.32942706f},
      },
      {
          .fine_side = {1, 1, 1},
          .normal = FaceNormalAxis::kY,
          .position = {-3.6875, 2.1875, -3.0625},
          .coarse_count = 36,
          .fine_count = 0,
          .level = 0,
          .owner = {0, 0, -1},
          .weight = 0.24720836f,
          .gradient = {-0.08415604f, -0.24123573f, -0.30425644f},
      },
      {
          .fine_side = {-1, 1, -1},
          .normal = FaceNormalAxis::kX,
          .position = {-3.9375, 2.3125, -3.1875},
          .coarse_count = 36,
          .fine_count = 0,
          .level = 0,
          .owner = {0, 0, -1},
          .weight = 0.35163778f,
          .gradient = {-0.12218666f, -0.11970644f, -0.11970645f},
      },
  }};
  for (const auto& expected : cases) {
    const auto stencil =
        EvaluateCoarseFineStencil3D({kOrigin, 0.5, expected.fine_side}, expected.normal, expected.position);
    ASSERT_TRUE(stencil.ok()) << stencil.status();
    const auto rows = stencil->rows();
    EXPECT_EQ(std::ranges::count(rows, 0, &CoarseFineStencilRow3D::level), expected.coarse_count);
    EXPECT_EQ(std::ranges::count(rows, 1, &CoarseFineStencilRow3D::level), expected.fine_count);
    const auto [x, y, z] = expected.owner;
    const auto key = std::tuple{expected.level, x, y, z};
    const auto row = std::ranges::find(rows, key, OwnerKey);
    ASSERT_NE(row, rows.end());
    EXPECT_NEAR(row->weight, expected.weight, 2e-7f);
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_NEAR(row->gradient_coarse_units[axis], expected.gradient[axis], 2e-7f);
    }
  }
}

TEST(CoarseFineTransfer3D, ClosedFineSupportAndCoarseAnchorHaveDifferentOwners) {
  const auto stencil = EvaluateCoarseFineStencil3D({{}, 1, {1, 1, 1}}, FaceNormalAxis::kX, {1, 1, 1});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  const auto rows = stencil->rows();
  EXPECT_EQ(std::ranges::count(rows, 0, &CoarseFineStencilRow3D::level), 8);
  EXPECT_EQ(std::ranges::count(rows, 1, &CoarseFineStencilRow3D::level), 8);
  const auto coarse = std::ranges::find(rows, std::tuple{0, 1, 0, 0}, OwnerKey);
  const auto fine = std::ranges::find(rows, std::tuple{1, 2, 1, 1}, OwnerKey);
  ASSERT_NE(coarse, rows.end());
  ASSERT_NE(fine, rows.end());
  EXPECT_NEAR(coarse->weight, 1.0f / 48, 1e-7f);
  EXPECT_NEAR(fine->weight, 1.0f / 6, 1e-7f);
}

TEST(CoarseFineTransfer3D, DistinguishesIncompleteCoplanarAndEmptyMasks) {
  constexpr std::array expected_partition_defects{0.23014325f, 0.23382568f, 0.23739302f};
  constexpr std::array expected_coplanar_counts{18, 15, 13};
  for (std::size_t topology = 0; topology < expected_partition_defects.size(); ++topology) {
    std::array<int, 3> fine_side{};
    std::ranges::fill(std::span{fine_side}.first(topology + 1), 1);
    const auto stencil = EvaluateCoarseFineStencil3D({{}, 1, fine_side}, FaceNormalAxis::kX, {0.875, 0.875, 0.875});
    ASSERT_TRUE(stencil.ok()) << stencil.status();
    const auto rows = stencil->rows();
    std::array<bool, CoarseFineStencil3D::kMaxRows> keep{};
    const auto mask = std::span{keep}.first(rows.size());
    const auto expect_disposition = [&](CoarseFineMaskDisposition3D expected) {
      const auto disposition = ClassifyCoarseFineMask3D(*stencil, mask);
      ASSERT_TRUE(disposition.ok()) << disposition.status();
      EXPECT_EQ(*disposition, expected);
    };
    std::ranges::fill(mask, true);
    expect_disposition(CoarseFineMaskDisposition3D::kComplete);
    const auto largest = std::ranges::max_element(rows, {}, &CoarseFineStencilRow3D::weight);
    keep[largest - rows.begin()] = false;
    expect_disposition(CoarseFineMaskDisposition3D::kPicRequiredIncomplete);
    EXPECT_NEAR(largest->weight, 0.28925580f, 2e-7f);
    float partition = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      keep[i] = rows[i].offset_coarse_units[0] == largest->offset_coarse_units[0];
      if (keep[i]) {
        partition += rows[i].weight;
      }
    }
    expect_disposition(CoarseFineMaskDisposition3D::kPicRequiredRankDeficient);
    EXPECT_EQ(std::ranges::count(mask, true), expected_coplanar_counts[topology]);
    EXPECT_NEAR(1 - partition, expected_partition_defects[topology], 2e-7f);
    const auto off_plane = std::ranges::find(mask, false);
    ASSERT_NE(off_plane, mask.end());
    *off_plane = true;
    expect_disposition(CoarseFineMaskDisposition3D::kPicRequiredIncomplete);
    std::ranges::fill(mask, false);
    EXPECT_EQ(ClassifyCoarseFineMask3D(*stencil, mask).status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(ClassifyCoarseFineMask3D(*stencil, mask.first(1)).status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST(CoarseFineTransfer3D, RejectsInvalidGeometryAndUnrepresentableQueries) {
  const auto check = [](CoarseFinePatch3D patch, Vec3 position, FaceNormalAxis normal = FaceNormalAxis::kX) {
    return EvaluateCoarseFineStencil3D(patch, normal, position).status().code();
  };
  EXPECT_EQ(check({{}, 0, {1, 0, 0}}, {}), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(check({{}, 1, {2, 0, 0}}, {}), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(check({{}, 1, {1, 0, 0}}, {}, static_cast<FaceNormalAxis>(3)), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      check({{}, 1, {1, 0, 0}}, {std::numeric_limits<double>::quiet_NaN(), 0, 0}),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(check({{}, 1, {1, 0, 0}}, {0, -65, 0}), absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace fluidity
