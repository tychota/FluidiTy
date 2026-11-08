#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#include "absl/status/status.h"
#include "fluidity/base/types.h"
#include "fluidity/host/basis.h"
#include "fluidity/host/grid.h"
#include "fluidity/host/views.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(Base, Vec2Arithmetic) {
  static_assert(std::is_trivial_v<Vec2> && std::is_standard_layout_v<Vec2>);
  static_assert(std::is_trivial_v<Mat2> && std::is_standard_layout_v<Mat2>);
  static_assert(sizeof(Vec2) == 2 * sizeof(double));
  static_assert(sizeof(Mat2) == 4 * sizeof(double));

  const Vec2 result = Vec2{3.0, -2.0} + 2.0 * Vec2{1.0, 4.0} - Vec2{2.0, 1.0} * 0.5;
  EXPECT_EQ(result.x, 4.0);
  EXPECT_EQ(result.y, 5.5);
}

TEST(Grid, TwoCellOrientation) {
  const Grid2D grid{2, 1, 1.0, {0.0, 0.0}};
  ASSERT_TRUE(ValidateGrid(grid).ok());
  const auto cells = CellCount(grid);
  const auto faces = FaceCount(grid);
  ASSERT_TRUE(cells.ok()) << cells.status();
  ASSERT_TRUE(faces.ok()) << faces.status();
  EXPECT_EQ(*cells, 2u);
  ASSERT_EQ(*faces, 7u);

  // Each tail receives +1 incidence and each head receives -1.
  const std::array<std::int64_t, 7> expected_tails{-1, 0, 1, -1, -1, 0, 1};
  const std::array<std::int64_t, 7> expected_heads{0, 1, -1, 0, 1, -1, -1};
  for (std::size_t face_index = 0; face_index < expected_tails.size(); ++face_index) {
    SCOPED_TRACE(face_index);
    const auto info = FaceGeometry(grid, face_index);
    ASSERT_TRUE(info.ok()) << info.status();
    EXPECT_EQ(info->tail_cell_index, expected_tails[face_index]);
    EXPECT_EQ(info->head_cell_index, expected_heads[face_index]);
  }
}

TEST(Grid, NonunitOriginAndDistance) {
  const Grid2D grid{2, 2, 0.25, {-2.0, 3.0}};
  const auto volume = CellVolume(grid);
  ASSERT_TRUE(volume.ok()) << volume.status();
  EXPECT_EQ(*volume, 0.0625);

  constexpr std::array positions{
      Vec2{-2.0, 3.125},
      Vec2{-1.75, 3.125},
      Vec2{-1.5, 3.125},
      Vec2{-2.0, 3.375},
      Vec2{-1.75, 3.375},
      Vec2{-1.5, 3.375},
      Vec2{-1.875, 3.0},
      Vec2{-1.625, 3.0},
      Vec2{-1.875, 3.25},
      Vec2{-1.625, 3.25},
      Vec2{-1.875, 3.5},
      Vec2{-1.625, 3.5},
  };
  constexpr std::array distances{
      0.125,
      0.25,
      0.125,
      0.125,
      0.25,
      0.125,
      0.125,
      0.125,
      0.25,
      0.25,
      0.125,
      0.125,
  };
  for (std::size_t face_index = 0; face_index < positions.size(); ++face_index) {
    SCOPED_TRACE(face_index);
    const auto info = FaceGeometry(grid, face_index);
    ASSERT_TRUE(info.ok()) << info.status();
    EXPECT_EQ(info->world_position.x, positions[face_index].x);
    EXPECT_EQ(info->world_position.y, positions[face_index].y);
    EXPECT_EQ(info->area, 0.25);
    EXPECT_EQ(info->pressure_gradient_distance, distances[face_index]);
  }
  const auto internal_y = FaceGeometry(grid, 9);
  ASSERT_TRUE(internal_y.ok()) << internal_y.status();
  EXPECT_EQ(internal_y->tail_cell_index, 1);
  EXPECT_EQ(internal_y->head_cell_index, 3);
}

TEST(Grid, RejectsInvalidInput) {
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(CellCount({0, 1, 1.0, {0.0, 0.0}}).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(FaceCount({1, 0, 1.0, {0.0, 0.0}}).status().code(), absl::StatusCode::kInvalidArgument);
  for (double spacing : {0.0, -1.0, infinity, nan}) {
    EXPECT_EQ(ValidateGrid({1, 1, spacing, {0.0, 0.0}}).code(), absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(ValidateGrid({1, 1, 1.0, {0.0, infinity}}).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(FaceGeometry({2, 1, 1.0, {0.0, 0.0}}, 7).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(Grid, RejectsCountOverflow) {
  constexpr auto max_dimension = std::numeric_limits<std::uint32_t>::max();
  const Grid2D grid{max_dimension, max_dimension, 1.0, {0.0, 0.0}};
  EXPECT_EQ(ValidateGrid(grid).code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(CellCount(grid).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(FaceCount(grid).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(FaceGeometry(grid, 0).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(Grid, WidensDimensionsBeforeAddingBoundaryFaces) {
  constexpr auto max_dimension = std::numeric_limits<std::uint32_t>::max();
  if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
    const Grid2D grid{max_dimension, 1, 1.0, {0.0, 0.0}};
    const auto faces = FaceCount(grid);
    ASSERT_TRUE(faces.ok()) << faces.status();
    EXPECT_EQ(*faces, 12884901886ull);
    const auto high_x = FaceGeometry(grid, max_dimension);
    ASSERT_TRUE(high_x.ok()) << high_x.status();
    EXPECT_EQ(high_x->tail_cell_index, 4294967294ll);
    EXPECT_EQ(high_x->head_cell_index, -1);
  }
}

TEST(Grid, RejectsUnrepresentableMeasures) {
  for (double spacing : {std::numeric_limits<double>::denorm_min(), 1e-200, 1e200}) {
    const Grid2D grid{1, 1, spacing, {0.0, 0.0}};
    EXPECT_EQ(ValidateGrid(grid).code(), absl::StatusCode::kOutOfRange);
    EXPECT_EQ(CellVolume(grid).status().code(), absl::StatusCode::kOutOfRange);
    EXPECT_EQ(FaceGeometry(grid, 0).status().code(), absl::StatusCode::kOutOfRange);
  }
}

TEST(Basis, Moments) {
  constexpr double spacing = 0.25;
  constexpr double tolerance = 1e-12;
  const Vec2 position{-0.37, 0.82};
  const Vec2 offset{0.5, 0.0};
  const auto stencil = QuadraticStencil2D(position, spacing, offset);
  ASSERT_TRUE(stencil.ok()) << stencil.status();

  double weight_sum = 0.0;
  std::array<double, 2> first_moment{};
  std::array<double, 2> derivative_sum{};
  std::array<std::array<double, 2>, 2> second_moment{};
  std::array<std::array<double, 2>, 2> derivative_moment{};
  for (std::size_t node_index = 0; node_index < stencil->nodes.size(); ++node_index) {
    const auto [node_x, node_y] = stencil->nodes[node_index];
    const std::array displacement{
        static_cast<double>(node_x) + offset.x - position.x / spacing,
        static_cast<double>(node_y) + offset.y - position.y / spacing,
    };
    const double weight = stencil->weights[node_index];
    const std::array derivative{stencil->gradients[node_index].x * spacing, stencil->gradients[node_index].y * spacing};
    EXPECT_GE(weight, 0.0);
    weight_sum += weight;
    for (std::size_t axis = 0; axis < 2; ++axis) {
      first_moment[axis] += weight * displacement[axis];
      derivative_sum[axis] += derivative[axis];
      for (std::size_t other_axis = 0; other_axis < 2; ++other_axis) {
        second_moment[axis][other_axis] += weight * displacement[axis] * displacement[other_axis];
        derivative_moment[axis][other_axis] += derivative[axis] * displacement[other_axis];
      }
    }
  }
  EXPECT_NEAR(weight_sum, 1.0, tolerance);
  for (std::size_t axis = 0; axis < 2; ++axis) {
    EXPECT_NEAR(first_moment[axis], 0.0, tolerance);
    EXPECT_NEAR(derivative_sum[axis], 0.0, tolerance);
    for (std::size_t other_axis = 0; other_axis < 2; ++other_axis) {
      EXPECT_NEAR(second_moment[axis][other_axis], axis == other_axis ? 0.25 : 0.0, tolerance);
      EXPECT_NEAR(derivative_moment[axis][other_axis], axis == other_axis ? 1.0 : 0.0, tolerance);
    }
  }
}

TEST(Basis, NegativeCoordinatesPreserveNodeOrder) {
  const auto stencil = QuadraticStencil2D({-0.125, -1.125}, 0.5, {0.5, 0.0});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  const std::array<std::int64_t, 9> expected_x{-2, -2, -2, -1, -1, -1, 0, 0, 0};
  const std::array<std::int64_t, 9> expected_y{-3, -2, -1, -3, -2, -1, -3, -2, -1};
  for (std::size_t node_index = 0; node_index < expected_x.size(); ++node_index) {
    SCOPED_TRACE(node_index);
    EXPECT_EQ(stencil->nodes[node_index].x, expected_x[node_index]);
    EXPECT_EQ(stencil->nodes[node_index].y, expected_y[node_index]);
  }
}

TEST(Basis, HalfCellTiesChooseTheHighStencil) {
  const auto stencil = QuadraticStencil2D({0.5, -0.5}, 1.0, {0.0, 0.0});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_EQ(stencil->nodes[0].x, 0);
  EXPECT_EQ(stencil->nodes[0].y, -1);
  EXPECT_EQ(stencil->nodes[8].x, 2);
  EXPECT_EQ(stencil->nodes[8].y, 1);
  EXPECT_EQ(stencil->weights[0], 0.25);
  EXPECT_EQ(stencil->gradients[0].x, -0.5);
  EXPECT_EQ(stencil->gradients[0].y, -0.5);
  EXPECT_EQ(stencil->weights[8], 0.0);
  EXPECT_EQ(stencil->gradients[8].x, 0.0);
  EXPECT_EQ(stencil->gradients[8].y, 0.0);
}

TEST(Basis, PreservesStencilAtLargeIntegerCoordinates) {
  const auto stencil = QuadraticStencil2D({0x1p53, -0x1p53}, 1.0, {0.0, 0.0});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_EQ(stencil->nodes[0].x, 9007199254740991ll);
  EXPECT_EQ(stencil->nodes[0].y, -9007199254740993ll);
  EXPECT_EQ(stencil->nodes[8].x, 9007199254740993ll);
  EXPECT_EQ(stencil->nodes[8].y, -9007199254740991ll);
  EXPECT_EQ(stencil->weights[4], 0.5625);
}

TEST(Basis, PreservesHalfOffsetAtLargeCoordinates) {
  const auto stencil = QuadraticStencil2D({9007199254740991.0, 0.0}, 1.0, {0.5, 0.0});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_EQ(stencil->nodes[0].x, 9007199254740990ll);
  EXPECT_EQ(stencil->nodes[8].x, 9007199254740992ll);
  EXPECT_EQ(stencil->weights[4], 0.375);
  EXPECT_EQ(stencil->gradients[1].x, -0.75);
}

TEST(Basis, PreservesOffsetRemainderBelowHalfCellTie) {
  const auto stencil = QuadraticStencil2D({-0x1p-55, 0.0}, 1.0, {0.5, 0.0});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_EQ(stencil->nodes[0].x, -2);
  EXPECT_EQ(stencil->nodes[8].x, 0);
  EXPECT_EQ(stencil->weights[1], 0x1.8p-112);
  EXPECT_EQ(stencil->gradients[1].x, -0x1.8p-56);
}

TEST(Basis, ChecksShiftedStencilAtSignedIndexLimits) {
  constexpr auto low = std::numeric_limits<std::int64_t>::min();
  constexpr auto high = std::numeric_limits<std::int64_t>::max();
  const auto stencil = QuadraticStencil2D({-0x1p63, 0x1p63}, 1.0, {-1.0, 2.0});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_EQ(stencil->nodes[0].x, low);
  EXPECT_EQ(stencil->nodes[8].x, low + 2);
  EXPECT_EQ(stencil->nodes[0].y, high - 2);
  EXPECT_EQ(stencil->nodes[8].y, high);
  EXPECT_EQ(stencil->weights[4], 0.5625);
  EXPECT_EQ(QuadraticStencil2D({0.0, 0x1p63}, 1.0, {0.0, 1.0}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(Basis, PreservesSmallSignedCenterDerivative) {
  const auto stencil = QuadraticStencil2D({-0x1p-55, 0.0}, 1.0, {0.0, 0.0});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_EQ(stencil->gradients[4].x, 0x1.8p-55);
}

TEST(Basis, RejectsInvalidInput) {
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (double spacing : {0.0, -1.0, infinity, nan}) {
    EXPECT_EQ(QuadraticStencil2D({0.0, 0.0}, spacing, {0.0, 0.0}).status().code(), absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(QuadraticStencil2D({infinity, 0.0}, 1.0, {0.0, 0.0}).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(QuadraticStencil2D({0.0, 0.0}, 1.0, {nan, 0.0}).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(Basis, RejectsUnrepresentableNodesAndArithmetic) {
  for (const double x_position : {-0x1p63, 0x1p63}) {
    EXPECT_EQ(QuadraticStencil2D({x_position, 0.0}, 1.0, {0.0, 0.0}).status().code(), absl::StatusCode::kOutOfRange);
  }
  EXPECT_EQ(QuadraticStencil2D({1e200, 0.0}, 1e-200, {0.0, 0.0}).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(QuadraticStencil2D({0.0, 0.0}, 1e-320, {0.0, 0.0}).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(Basis, ScalesTensorDerivativeAfterWeighting) {
  constexpr double spacing = 2.5e-309;
  const auto stencil = QuadraticStencil2D({0.0, 0.0}, spacing, {0.0, 0.0});
  ASSERT_TRUE(stencil.ok()) << stencil.status();
  EXPECT_NEAR(stencil->gradients[1].x * spacing, -0.375, 1e-12);
  EXPECT_NEAR(stencil->gradients[3].y * spacing, -0.375, 1e-12);
}

TEST(Views, ChecksExactExtentsAndProducts) {
  EXPECT_TRUE(ValidateExtent(0, 0).ok());
  EXPECT_TRUE(ValidateExtent(5, 5).ok());
  EXPECT_EQ(ValidateExtent(4, 5).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ValidateExtent(6, 5).code(), absl::StatusCode::kInvalidArgument);

  const auto product = CheckedExtentProduct(3, 4);
  ASSERT_TRUE(product.ok()) << product.status();
  EXPECT_EQ(*product, 12u);
  const auto zero = CheckedExtentProduct(0, std::numeric_limits<std::size_t>::max());
  ASSERT_TRUE(zero.ok()) << zero.status();
  EXPECT_EQ(*zero, 0u);
  const auto overflow = CheckedExtentProduct(std::numeric_limits<std::size_t>::max(), 2);
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kOutOfRange);
}

TEST(Views, DetectsByteOverlapWithoutChangingStorage) {
  struct Workspace {
    double scratch[4];
    int metadata;
  };
  Workspace workspace{{1.0, 2.0, 3.0, 4.0}, 5};
  const std::array<double, 2> independent_values{6.0, 7.0};
  const auto whole = std::as_bytes(std::span{&workspace, 1});
  const auto scratch = std::as_bytes(std::span{workspace.scratch});
  const auto metadata = std::as_bytes(std::span{&workspace.metadata, 1});
  EXPECT_EQ(ValidateDisjoint(whole, metadata).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ValidateDisjoint(metadata, whole).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ValidateDisjoint(scratch.first(2), scratch.subspan(1, 2)).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(ValidateDisjoint(scratch.first(2), scratch.subspan(2, 2)).ok());
  EXPECT_TRUE(ValidateDisjoint(scratch, std::as_bytes(std::span{independent_values})).ok());
  EXPECT_TRUE(ValidateDisjoint(scratch, scratch.first(0)).ok());
  EXPECT_EQ(workspace.scratch[0], 1.0);
  EXPECT_EQ(workspace.metadata, 5);
  EXPECT_EQ(independent_values[0], 6.0);
}

}  // namespace
}  // namespace fluidity
