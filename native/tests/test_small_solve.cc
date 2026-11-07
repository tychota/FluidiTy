#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

#include "absl/status/status.h"
#include "fluidity/host/small_solve.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(SmallSolve, HandTwoByTwo) {
  const std::array matrix{2.0, -1.0, -1.0, 2.0};
  const std::array rhs{1.0, 0.0};
  std::array<double, 2> solution{};
  SmallSolveWorkspace workspace;

  ASSERT_TRUE(SolveSmallSpd(matrix, rhs, solution, workspace).ok());
  EXPECT_NEAR(solution[0], 2.0 / 3.0, 1e-15);
  EXPECT_NEAR(solution[1], 1.0 / 3.0, 1e-15);
}

TEST(SmallSolve, TinyDiagonalScale) {
  const std::array matrix{1e-13, 0.0, 0.0, 2e-13};
  const std::array rhs{2e-13, -6e-13};
  std::array<double, 2> solution{};
  SmallSolveWorkspace workspace;

  ASSERT_TRUE(SolveSmallSpd(matrix, rhs, solution, workspace).ok());
  EXPECT_NEAR(solution[0], 2.0, 1e-15);
  EXPECT_NEAR(solution[1], -3.0, 1e-15);
}

TEST(SmallSolve, Residual) {
  constexpr std::array regular_matrix{2.0, -1.0, -1.0, 2.0};
  constexpr std::array regular_rhs{1.0, 0.0};
  constexpr std::array tiny_matrix{1e-13, 0.0, 0.0, 2e-13};
  constexpr std::array tiny_rhs{2e-13, -6e-13};
  constexpr std::array coupled_matrix{4.0, 1.0, -1.0, 1.0, 3.0, 0.5, -1.0, 0.5, 2.0};
  constexpr std::array coupled_rhs{9.0, -2.0, 0.5};
  struct System {
    std::span<const double> matrix;
    std::span<const double> rhs;
  };
  const std::array systems{
      System{regular_matrix, regular_rhs},
      System{tiny_matrix, tiny_rhs},
      System{coupled_matrix, coupled_rhs},
  };
  SmallSolveWorkspace workspace;

  for (const auto& system : systems) {
    const std::size_t unknowns = system.rhs.size();
    std::array<double, 3> storage{};
    const auto solution = std::span(storage).first(unknowns);
    ASSERT_TRUE(SolveSmallSpd(system.matrix, system.rhs, solution, workspace).ok());

    double max_residual = 0.0;
    double max_rhs = 0.0;
    for (std::size_t row = 0; row < unknowns; ++row) {
      double matrix_product = 0.0;
      for (std::size_t column = 0; column < unknowns; ++column) {
        matrix_product += system.matrix[row * unknowns + column] * solution[column];
      }
      ASSERT_TRUE(std::isfinite(matrix_product));
      max_residual = std::max(max_residual, std::abs(system.rhs[row] - matrix_product));
      max_rhs = std::max(max_rhs, std::abs(system.rhs[row]));
    }
    EXPECT_LE(max_residual, 1e-12 * std::max(1.0, max_rhs));
  }
}

TEST(SmallSolve, EmptySystem) {
  SmallSolveWorkspace workspace;

  EXPECT_TRUE(SolveSmallSpd({}, {}, {}, workspace).ok());
}

TEST(SmallSolve, RejectsIncorrectExtents) {
  const std::array matrix{2.0, -1.0, -1.0, 2.0};
  const std::array rhs{1.0, 0.0};
  std::array<double, 2> solution{7.0, 8.0};
  SmallSolveWorkspace workspace;

  EXPECT_EQ(
      SolveSmallSpd(std::span(matrix).first(3), rhs, solution, workspace).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(
      SolveSmallSpd(matrix, rhs, std::span(solution).first(1), workspace).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(SolveSmallSpd(matrix, {}, {}, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SolveSmallSpd({}, {}, solution, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(solution, (std::array{7.0, 8.0}));
}

TEST(SmallSolve, MaximumSupportedSize) {
  constexpr std::size_t unknowns = 64;
  std::array<double, unknowns * unknowns> matrix{};
  std::array<double, unknowns> rhs{};
  std::array<double, unknowns> solution{};
  SmallSolveWorkspace workspace;
  for (std::size_t row = 0; row < unknowns; ++row) {
    matrix[row * unknowns + row] = 4.0;
    rhs[row] = 4.0 * static_cast<double>(row);
  }

  ASSERT_TRUE(SolveSmallSpd(matrix, rhs, solution, workspace).ok());
  for (std::size_t row = 0; row < unknowns; ++row) {
    EXPECT_DOUBLE_EQ(solution[row], static_cast<double>(row));
  }
}

TEST(SmallSolve, RejectsOversizedSystem) {
  constexpr std::size_t unknowns = 65;
  const std::array<double, unknowns * unknowns> matrix{};
  const std::array<double, unknowns> rhs{};
  std::array<double, unknowns> solution;
  solution.fill(9.0);
  const auto initial_solution = solution;
  SmallSolveWorkspace workspace;

  EXPECT_EQ(SolveSmallSpd(matrix, rhs, solution, workspace).code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(solution, initial_solution);
}

TEST(SmallSolve, RejectsTinyNonsymmetricMatrix) {
  const std::array matrix{1e-13, 9e-13, 0.0, 1e-13};
  const std::array rhs{1.0, 0.0};
  std::array<double, 2> solution{7.0, 8.0};
  SmallSolveWorkspace workspace;

  EXPECT_EQ(SolveSmallSpd(matrix, rhs, solution, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(solution, (std::array{7.0, 8.0}));
}

TEST(SmallSolve, AveragesRoundingLevelAsymmetry) {
  const std::array matrix{1.0, 0x1p-50, 0.0, 1.0};
  const std::array rhs{0.0, 1.0};
  std::array<double, 2> solution{};
  SmallSolveWorkspace workspace;

  ASSERT_TRUE(SolveSmallSpd(matrix, rhs, solution, workspace).ok());
  EXPECT_DOUBLE_EQ(solution[0], -0x1p-51);
  EXPECT_DOUBLE_EQ(solution[1], 1.0);
}

TEST(SmallSolve, RejectsIndefiniteMatrix) {
  const std::array matrix{1.0, 2.0, 2.0, 1.0};
  const std::array rhs{1.0, 0.0};
  std::array<double, 2> solution{7.0, 8.0};
  SmallSolveWorkspace workspace;

  EXPECT_EQ(SolveSmallSpd(matrix, rhs, solution, workspace).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(solution, (std::array{7.0, 8.0}));
}

TEST(SmallSolve, RejectsSingularMatrix) {
  const std::array matrix{2.0, 2.0, 2.0, 2.0};
  const std::array rhs{0.0, 0.0};
  std::array<double, 2> solution{7.0, 8.0};
  SmallSolveWorkspace workspace;

  EXPECT_EQ(SolveSmallSpd(matrix, rhs, solution, workspace).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(solution, (std::array{7.0, 8.0}));
}

TEST(SmallSolve, RejectsUnsafeRelativePivot) {
  const std::array matrix{1e-13, 0.0, 0.0, 1e-29};
  const std::array rhs{1.0, 0.0};
  std::array<double, 2> solution{7.0, 8.0};
  SmallSolveWorkspace workspace;

  EXPECT_EQ(SolveSmallSpd(matrix, rhs, solution, workspace).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(solution, (std::array{7.0, 8.0}));
}

TEST(SmallSolve, RejectsNonfiniteInput) {
  constexpr double infinity = std::numeric_limits<double>::infinity();
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  const std::array matrix{2.0, 0.0, 0.0, 2.0};
  const std::array nonfinite_matrix{2.0, infinity, 0.0, 2.0};
  const std::array rhs{1.0, 0.0};
  const std::array nonfinite_rhs{nan, 0.0};
  std::array<double, 2> solution{7.0, 8.0};
  SmallSolveWorkspace workspace;

  EXPECT_EQ(SolveSmallSpd(nonfinite_matrix, rhs, solution, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SolveSmallSpd(matrix, nonfinite_rhs, solution, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(solution, (std::array{7.0, 8.0}));
}

TEST(SmallSolve, LeavesOutputUnchangedWhenSubstitutionOverflows) {
  const std::array matrix{1e-13, 0.0, 0.0, 1e-13};
  const std::array rhs{1e300, 2e-13};
  std::array<double, 2> solution{7.0, 8.0};
  SmallSolveWorkspace workspace;

  EXPECT_EQ(SolveSmallSpd(matrix, rhs, solution, workspace).code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(solution, (std::array{7.0, 8.0}));
}

TEST(SmallSolve, RejectsOutputOverlapWithInputs) {
  std::array matrix{2.0, -1.0, -1.0, 2.0};
  std::array rhs{1.0, 0.0};
  SmallSolveWorkspace workspace;

  EXPECT_EQ(
      SolveSmallSpd(matrix, rhs, std::span(matrix).subspan(1, 2), workspace).code(),
      absl::StatusCode::kInvalidArgument
  );
  EXPECT_EQ(SolveSmallSpd(matrix, rhs, rhs, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(matrix, (std::array{2.0, -1.0, -1.0, 2.0}));
  EXPECT_EQ(rhs, (std::array{1.0, 0.0}));
}

TEST(SmallSolve, RejectsOverlapWithInactiveWorkspaceStorage) {
  const std::array matrix{2.0, -1.0, -1.0, 2.0};
  const std::array rhs{1.0, 0.0};
  std::array<double, 2> solution{7.0, 8.0};
  SmallSolveWorkspace workspace;
  auto workspace_matrix = std::span(workspace.factor).last(4);
  auto workspace_vector = std::span(workspace.working_rhs).last(2);
  std::ranges::copy(matrix, workspace_matrix.begin());
  std::ranges::copy(rhs, workspace_vector.begin());

  EXPECT_EQ(SolveSmallSpd(workspace_matrix, rhs, solution, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SolveSmallSpd(matrix, workspace_vector, solution, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SolveSmallSpd(matrix, rhs, workspace_vector, workspace).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(solution, (std::array{7.0, 8.0}));
  EXPECT_EQ(workspace_vector[0], 1.0);
  EXPECT_EQ(workspace_vector[1], 0.0);
}

}  // namespace
}  // namespace fluidity
