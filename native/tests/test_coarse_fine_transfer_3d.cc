#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

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

TEST(CoarseFineTransfer3D, MaskedPicReproducesConstantAndClearsAffectedHistory) {
  const auto stencil = EvaluateCoarseFineStencil3D({{}, 1, {-1, 1, -1}}, FaceNormalAxis::kY, {-0.875, 0.875, -0.875});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  const auto rows = stencil->rows();
  std::array<bool, CoarseFineStencil3D::kMaxRows> keep{};
  const auto mask = std::span{keep}.first(rows.size());
  std::vector<CoarseFinePicCarrierSample3D> carriers;
  constexpr double velocity = 0.1;
  for (std::size_t i = 1; i < rows.size(); ++i) {
    keep[i] = true;
    carriers.push_back({rows[i].owner, rows[i].level, velocity});
  }
  LiquidBoundaryParticleGather3D gather{
      .new_pic_velocity = {1, 2, 3},
      .grid_velocity_increment = {4, 5, 6},
      .particle_minus_old_pic_velocity = {7, 8, 9},
      .fitted_velocity_slopes = {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}},
      .velocity_gradient = {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}},
  };
  const auto status = ApplyCoarseFinePicFallback3D(*stencil, mask, carriers, gather);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(gather.new_pic_velocity.y, velocity);
  EXPECT_EQ(gather.grid_velocity_increment.y, 0);
  EXPECT_EQ(gather.particle_minus_old_pic_velocity.y, 0);
  const auto components = [](Vec3 value) {
    return std::array{value.x, value.y, value.z};
  };
  EXPECT_EQ(components(gather.fitted_velocity_slopes.y_row), (std::array{0.0, 0.0, 0.0}));
  EXPECT_EQ(components(gather.velocity_gradient.y_row), (std::array{0.0, 0.0, 0.0}));
  EXPECT_EQ(components(gather.new_pic_velocity), (std::array{1.0, velocity, 3.0}));
  EXPECT_EQ(components(gather.grid_velocity_increment), (std::array{4.0, 0.0, 6.0}));
  EXPECT_EQ(components(gather.particle_minus_old_pic_velocity), (std::array{7.0, 0.0, 9.0}));
  EXPECT_EQ(components(gather.fitted_velocity_slopes.x_row), (std::array{10.0, 11.0, 12.0}));
  EXPECT_EQ(components(gather.fitted_velocity_slopes.z_row), (std::array{16.0, 17.0, 18.0}));
  EXPECT_EQ(components(gather.velocity_gradient.x_row), (std::array{19.0, 20.0, 21.0}));
  EXPECT_EQ(components(gather.velocity_gradient.z_row), (std::array{25.0, 26.0, 27.0}));
  EXPECT_FALSE(gather.affine_enabled);
  EXPECT_FALSE(gather.old_pic_fallback);
  EXPECT_TRUE(gather.new_pic_fallback);
  EXPECT_FALSE(gather.low_mass_fallback);
}

TEST(CoarseFineTransfer3D, CoplanarPicUsesWeightedNonAffineDonorsInCanonicalOrder) {
  const auto stencil = EvaluateCoarseFineStencil3D({{}, 1, {1, 1, 1}}, FaceNormalAxis::kX, {0.875, 0.875, 0.875});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  const auto rows = stencil->rows();
  std::array<bool, CoarseFineStencil3D::kMaxRows> keep{};
  const auto mask = std::span{keep}.first(rows.size());
  std::vector<CoarseFinePicCarrierSample3D> carriers;
  double total_weight = 0;
  double impulse_weight = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (rows[i].offset_coarse_units[0] != 0.125f) {
      continue;
    }
    keep[i] = true;
    const bool impulse = carriers.empty();
    if (impulse) {
      impulse_weight = rows[i].weight;
    }
    carriers.push_back({rows[i].owner, rows[i].level, impulse ? 13.0 : -5.0});
    total_weight += rows[i].weight;
  }
  ASSERT_GT(carriers.size(), 4);
  const auto disposition = ClassifyCoarseFineMask3D(*stencil, mask);
  ASSERT_TRUE(disposition.ok()) << disposition.status();
  ASSERT_EQ(*disposition, CoarseFineMaskDisposition3D::kPicRequiredRankDeficient);
  LiquidBoundaryParticleGather3D forward{};
  ASSERT_TRUE(ApplyCoarseFinePicFallback3D(*stencil, mask, carriers, forward).ok());
  EXPECT_NEAR(forward.new_pic_velocity.x, -5.0 + 18.0 * impulse_weight / total_weight, 1e-14);
  std::ranges::reverse(carriers);
  LiquidBoundaryParticleGather3D reverse{};
  ASSERT_TRUE(ApplyCoarseFinePicFallback3D(*stencil, mask, carriers, reverse).ok());
  EXPECT_EQ(reverse.new_pic_velocity.x, forward.new_pic_velocity.x);
  for (auto& carrier : carriers) {
    carrier.new_face_velocity = 0.0;
  }
  carriers.back().new_face_velocity = -0.0;
  ASSERT_TRUE(ApplyCoarseFinePicFallback3D(*stencil, mask, carriers, reverse).ok());
  std::ranges::reverse(carriers);
  ASSERT_TRUE(ApplyCoarseFinePicFallback3D(*stencil, mask, carriers, forward).ok());
  EXPECT_EQ(std::signbit(reverse.new_pic_velocity.x), std::signbit(forward.new_pic_velocity.x));
}

TEST(CoarseFineTransfer3D, CertifiedLowMassEulerianCarrierCanBeTheOnlyPicDonor) {
  const auto stencil = EvaluateCoarseFineStencil3D({{}, 1, {1, 1, 0}}, FaceNormalAxis::kZ, {0.875, 0.875, 0.875});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  const auto rows = stencil->rows();
  const auto& row = rows.front();
  std::array<bool, CoarseFineStencil3D::kMaxRows> keep{true};
  const auto mask = std::span{keep}.first(rows.size());
  const std::array carriers{CoarseFinePicCarrierSample3D{
      .owner = row.owner,
      .level = row.level,
      .new_face_velocity = 7.0,
      .low_mass = true,
  }};
  LiquidBoundaryParticleGather3D gather{};
  const auto status = ApplyCoarseFinePicFallback3D(*stencil, mask, carriers, gather);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(gather.new_pic_velocity.z, 7.0);
  EXPECT_TRUE(gather.low_mass_fallback);
  EXPECT_TRUE(gather.new_pic_fallback);
}

TEST(CoarseFineTransfer3D, PicRefusesInvalidCarrierSetsWithoutPublishing) {
  const auto stencil = EvaluateCoarseFineStencil3D({{}, 1, {1, 1, 0}}, FaceNormalAxis::kX, {0.875, 0.875, 0.875});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  const auto rows = stencil->rows();
  std::array<bool, CoarseFineStencil3D::kMaxRows> keep{true, true};
  const auto mask = std::span{keep}.first(rows.size());
  const CoarseFinePicCarrierSample3D first{rows[0].owner, rows[0].level, 1};
  const CoarseFinePicCarrierSample3D second{rows[1].owner, rows[1].level, 2};
  const auto check = [&](std::span<const CoarseFinePicCarrierSample3D> carriers, absl::StatusCode expected) {
    LiquidBoundaryParticleGather3D gather{
        .new_pic_velocity = {3, 4, 5},
        .grid_velocity_increment = {6, 7, 8},
        .particle_minus_old_pic_velocity = {9, 10, 11},
        .fitted_velocity_slopes = {{12, 13, 14}, {}, {}},
        .velocity_gradient = {{15, 16, 17}, {}, {}},
    };
    const auto status = ApplyCoarseFinePicFallback3D(*stencil, mask, carriers, gather);
    EXPECT_EQ(status.code(), expected) << status;
    EXPECT_EQ(gather.new_pic_velocity.x, 3);
    EXPECT_EQ(gather.grid_velocity_increment.x, 6);
    EXPECT_EQ(gather.particle_minus_old_pic_velocity.x, 9);
    EXPECT_EQ(gather.fitted_velocity_slopes.x_row.x, 12);
    EXPECT_EQ(gather.velocity_gradient.x_row.x, 15);
    EXPECT_TRUE(gather.affine_enabled);
    EXPECT_FALSE(gather.new_pic_fallback);
    EXPECT_FALSE(gather.low_mass_fallback);
  };
  check(std::array{first}, absl::StatusCode::kFailedPrecondition);
  check(std::array{first, first}, absl::StatusCode::kInvalidArgument);
  auto wrong_owner = second;
  wrong_owner.owner.axis = FaceNormalAxis::kZ;
  check(std::array{first, wrong_owner}, absl::StatusCode::kInvalidArgument);
  wrong_owner.owner = rows[2].owner;
  wrong_owner.level = rows[2].level;
  check(std::array{first, wrong_owner}, absl::StatusCode::kInvalidArgument);
  auto nonfinite = second;
  nonfinite.new_face_velocity = std::numeric_limits<double>::infinity();
  check(std::array{first, nonfinite}, absl::StatusCode::kInvalidArgument);
  std::ranges::fill(mask, false);
  check({}, absl::StatusCode::kFailedPrecondition);
  std::ranges::fill(mask, true);
  check({}, absl::StatusCode::kFailedPrecondition);
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
