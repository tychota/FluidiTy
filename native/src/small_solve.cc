#include "fluidity/host/small_solve.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/host/views.h"

namespace fluidity {
namespace {

constexpr double kRelativeSymmetryTolerance = 8.0 * std::numeric_limits<double>::epsilon();
constexpr double kRelativePivotFloor = kMaxDenseUnknowns * std::numeric_limits<double>::epsilon();

absl::Status ValidateSolveViews(
    std::span<const double> matrix,
    std::span<const double> rhs,
    std::span<double> solution,
    const SmallSolveWorkspace& workspace
) {
  const auto matrix_extent = CheckedExtentProduct(rhs.size(), rhs.size());
  if (!matrix_extent.ok()) {
    return matrix_extent.status();
  }
  if (const auto status = ValidateExtent(matrix.size(), *matrix_extent); !status.ok()) {
    return status;
  }
  if (const auto status = ValidateExtent(solution.size(), rhs.size()); !status.ok()) {
    return status;
  }

  const auto solution_bytes = std::as_bytes(solution);
  const auto workspace_bytes = std::as_bytes(std::span(&workspace, 1));
  const std::array input_views{std::as_bytes(matrix), std::as_bytes(rhs)};
  for (const auto input_bytes : input_views) {
    if (const auto status = ValidateDisjoint(solution_bytes, input_bytes); !status.ok()) {
      return status;
    }
    if (const auto status = ValidateDisjoint(workspace_bytes, input_bytes); !status.ok()) {
      return status;
    }
  }
  return ValidateDisjoint(workspace_bytes, solution_bytes);
}

// Stage the symmetric lower triangle once so factorization and both solves use the same matrix.
absl::StatusOr<double>
StageSymmetricLower(std::span<const double> matrix, std::size_t unknowns, std::span<double> factor) {
  double matrix_scale = 0.0;
  for (const double entry : matrix) {
    if (!std::isfinite(entry)) {
      return absl::InvalidArgumentError("SPD matrix entries must be finite");
    }
    matrix_scale = std::max(matrix_scale, std::abs(entry));
  }
  if (matrix_scale == 0.0) {
    return absl::FailedPreconditionError("SPD matrix is zero");
  }

  for (std::size_t row = 0; row < unknowns; ++row) {
    factor[row * unknowns + row] = matrix[row * unknowns + row];
    for (std::size_t column = 0; column < row; ++column) {
      const double lower = matrix[row * unknowns + column];
      const double upper = matrix[column * unknowns + row];
      // Normalize before subtracting to avoid overflow for large opposite-sign entries.
      const double relative_difference = std::abs(lower / matrix_scale - upper / matrix_scale);
      if (relative_difference > kRelativeSymmetryTolerance) {
        return absl::InvalidArgumentError("SPD matrix must be symmetric within rounding tolerance");
      }
      factor[row * unknowns + column] = std::midpoint(lower, upper);
    }
  }
  return matrix_scale;
}

// Overwrite the staged lower triangle with L in A = L L^T. The pivot floor screens
// cancellation at the scale of A and rejects tiny positive pivots as well as nonpositive ones.
absl::Status FactorCholesky(std::span<double> factor, std::size_t unknowns, double matrix_scale) {
  for (std::size_t row = 0; row < unknowns; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      const std::size_t entry_index = row * unknowns + column;
      double residual = factor[entry_index];
      for (std::size_t previous = 0; previous < column; ++previous) {
        residual = std::fma(-factor[row * unknowns + previous], factor[column * unknowns + previous], residual);
      }
      if (!std::isfinite(residual)) {
        return absl::OutOfRangeError("Cholesky update is outside the representable range");
      }

      if (row == column) {
        if (residual <= 0.0 || residual / matrix_scale <= kRelativePivotFloor) {
          return absl::FailedPreconditionError("SPD matrix has a nonpositive or numerically unsafe Cholesky pivot");
        }
        factor[entry_index] = std::sqrt(residual);
      } else {
        const double entry = residual / factor[column * unknowns + column];
        if (!std::isfinite(entry)) {
          return absl::OutOfRangeError("Cholesky factor entry is outside the representable range");
        }
        factor[entry_index] = entry;
      }
    }
  }
  return absl::OkStatus();
}

// Solve L y = b forward, then L^T x = y backward, reusing the same vector.
absl::Status SolveFactoredSystem(std::span<const double> factor, std::span<const double> rhs, std::span<double> work) {
  const std::size_t unknowns = rhs.size();
  for (std::size_t row = 0; row < unknowns; ++row) {
    double residual = rhs[row];
    for (std::size_t column = 0; column < row; ++column) {
      residual = std::fma(-factor[row * unknowns + column], work[column], residual);
    }
    const double entry = residual / factor[row * unknowns + row];
    if (!std::isfinite(entry)) {
      return absl::OutOfRangeError("forward substitution is outside the representable range");
    }
    work[row] = entry;
  }

  for (std::size_t remaining = unknowns; remaining > 0; --remaining) {
    const std::size_t row = remaining - 1;
    double residual = work[row];
    for (std::size_t column = row + 1; column < unknowns; ++column) {
      residual = std::fma(-factor[column * unknowns + row], work[column], residual);
    }
    const double entry = residual / factor[row * unknowns + row];
    if (!std::isfinite(entry)) {
      return absl::OutOfRangeError("back substitution is outside the representable range");
    }
    work[row] = entry;
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status SolveSmallSpd(
    std::span<const double> row_major_matrix,
    std::span<const double> rhs,
    std::span<double> solution,
    SmallSolveWorkspace& workspace
) {
  const std::size_t unknowns = rhs.size();
  if (unknowns > kMaxDenseUnknowns) {
    return absl::UnimplementedError("dense SPD oracle supports at most 64 unknowns");
  }
  if (const auto status = ValidateSolveViews(row_major_matrix, rhs, solution, workspace); !status.ok()) {
    return status;
  }
  if (unknowns == 0) {
    return absl::OkStatus();
  }
  for (const double entry : rhs) {
    if (!std::isfinite(entry)) {
      return absl::InvalidArgumentError("SPD right-hand side entries must be finite");
    }
  }

  const auto factor = std::span(workspace.factor).first(row_major_matrix.size());
  const auto matrix_scale = StageSymmetricLower(row_major_matrix, unknowns, factor);
  if (!matrix_scale.ok()) {
    return matrix_scale.status();
  }
  if (const auto status = FactorCholesky(factor, unknowns, *matrix_scale); !status.ok()) {
    return status;
  }

  const auto work = std::span(workspace.working_rhs).first(unknowns);
  if (const auto status = SolveFactoredSystem(factor, rhs, work); !status.ok()) {
    return status;
  }
  std::ranges::copy(work, solution.begin());
  return absl::OkStatus();
}

}  // namespace fluidity
