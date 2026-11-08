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

constexpr double kMinNormalDouble = std::numeric_limits<double>::min();
constexpr double kRelativeSymmetryTolerance = 8.0 * std::numeric_limits<double>::epsilon();
constexpr double kRelativePivotFloor = kMaxDenseUnknowns * std::numeric_limits<double>::epsilon();

absl::Status ValidateSolveViews(
    std::span<const double> matrix,
    std::span<const double> rhs,
    std::span<double> solution,
    const SmallSolveWorkspace& workspace
) {
  const auto matrix_coefficient_count = CheckedExtentProduct(rhs.size(), rhs.size());
  if (!matrix_coefficient_count.ok()) {
    return matrix_coefficient_count.status();
  }
  if (const auto status = ValidateExtent(matrix.size(), *matrix_coefficient_count); !status.ok()) {
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
StageSymmetricLower(std::span<const double> input_matrix, std::size_t unknown_count, std::span<double> staged_lower) {
  double matrix_scale = 0.0;
  for (const double matrix_coefficient : input_matrix) {
    if (!std::isfinite(matrix_coefficient)) {
      return absl::InvalidArgumentError("SPD matrix entries must be finite");
    }
    matrix_scale = std::max(matrix_scale, std::abs(matrix_coefficient));
  }
  if (matrix_scale == 0.0) {
    return absl::FailedPreconditionError("SPD matrix is zero");
  }

  for (std::size_t row = 0; row < unknown_count; ++row) {
    staged_lower[row * unknown_count + row] = input_matrix[row * unknown_count + row];
    for (std::size_t column = 0; column < row; ++column) {
      const double lower_coefficient = input_matrix[row * unknown_count + column];
      const double upper_coefficient = input_matrix[column * unknown_count + row];
      // Normalize before subtracting to avoid overflow for large opposite-sign entries.
      const double relative_difference = std::abs(lower_coefficient / matrix_scale - upper_coefficient / matrix_scale);
      if (relative_difference > kRelativeSymmetryTolerance) {
        return absl::InvalidArgumentError("SPD matrix must be symmetric within rounding tolerance");
      }
      staged_lower[row * unknown_count + column] = std::midpoint(lower_coefficient, upper_coefficient);
    }
  }
  // Cholesky updates can lose relative precision at subnormal matrix scales.
  if (matrix_scale < kMinNormalDouble) {
    return absl::FailedPreconditionError("SPD matrix scale is subnormal");
  }
  return matrix_scale;
}

// Overwrite the staged lower triangle with L in A = L L^T. The pivot floor screens
// cancellation at the scale of A and rejects tiny positive pivots as well as nonpositive ones.
absl::Status FactorCholesky(std::span<double> lower_factor, std::size_t unknown_count, double matrix_scale) {
  for (std::size_t row = 0; row < unknown_count; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      const std::size_t coefficient_index = row * unknown_count + column;
      // Earlier columns already hold L; this slot still holds its symmetric A coefficient.
      double residual = lower_factor[coefficient_index];
      for (std::size_t previous_column = 0; previous_column < column; ++previous_column) {
        residual = std::fma(
            -lower_factor[row * unknown_count + previous_column],
            lower_factor[column * unknown_count + previous_column],
            residual
        );
      }
      if (!std::isfinite(residual)) {
        return absl::OutOfRangeError("Cholesky update is outside the representable range");
      }

      if (row == column) {
        const bool normal_pivot = residual >= kMinNormalDouble;
        if (!normal_pivot || residual / matrix_scale <= kRelativePivotFloor) {
          return absl::FailedPreconditionError("SPD matrix has a nonpositive or numerically unsafe Cholesky pivot");
        }
        lower_factor[coefficient_index] = std::sqrt(residual);
      } else {
        const double factor_coefficient = residual / lower_factor[column * unknown_count + column];
        if (!std::isfinite(factor_coefficient)) {
          return absl::OutOfRangeError("Cholesky factor entry is outside the representable range");
        }
        lower_factor[coefficient_index] = factor_coefficient;
      }
    }
  }
  return absl::OkStatus();
}

// The inverse Frobenius norm squared bounds the inverse spectral norm of L L^T.
// Probe the columns of sqrt(matrix_scale) L^-1 to keep the estimate scale-relative.
// The O(n^3) pass reuses the RHS scratch for one column at a time.
absl::Status
CheckFactorConditioning(std::span<const double> lower_factor, double matrix_scale, std::span<double> inverse_column) {
  const std::size_t unknown_count = inverse_column.size();
  const double matrix_order = static_cast<double>(unknown_count);
  // Allow for O(n) rounded operations per matrix entry and n entries per row.
  const double relative_roundoff_budget =
      4.0 * matrix_order * (matrix_order + 1.0) * std::numeric_limits<double>::epsilon();
  const double inverse_norm_squared_limit = 1.0 / relative_roundoff_budget;
  const double scaled_basis_entry = std::sqrt(matrix_scale);
  double scaled_inverse_norm_squared = 0.0;

  for (std::size_t basis_column = 0; basis_column < unknown_count; ++basis_column) {
    // Entries above this column's diagonal are implicitly zero.
    for (std::size_t row = basis_column; row < unknown_count; ++row) {
      double residual = row == basis_column ? scaled_basis_entry : 0.0;
      for (std::size_t column = basis_column; column < row; ++column) {
        residual = std::fma(-lower_factor[row * unknown_count + column], inverse_column[column], residual);
      }
      const double inverse_coefficient = residual / lower_factor[row * unknown_count + row];
      scaled_inverse_norm_squared = std::fma(inverse_coefficient, inverse_coefficient, scaled_inverse_norm_squared);
      if (!std::isfinite(scaled_inverse_norm_squared)) {
        return absl::OutOfRangeError("Cholesky conditioning probe is outside the representable range");
      }
      if (scaled_inverse_norm_squared >= inverse_norm_squared_limit) {
        return absl::FailedPreconditionError("SPD matrix is too ill-conditioned for a reliable Cholesky solve");
      }
      inverse_column[row] = inverse_coefficient;
    }
  }
  return absl::OkStatus();
}

// Solve L y = b forward, then L^T x = y backward, reusing the same vector.
absl::Status SolveFactoredSystem(
    std::span<const double> lower_factor,
    std::span<const double> rhs,
    std::span<double> staged_solution
) {
  const std::size_t unknown_count = rhs.size();
  for (std::size_t row = 0; row < unknown_count; ++row) {
    double residual = rhs[row];
    for (std::size_t column = 0; column < row; ++column) {
      residual = std::fma(-lower_factor[row * unknown_count + column], staged_solution[column], residual);
    }
    const double forward_coefficient = residual / lower_factor[row * unknown_count + row];
    if (!std::isfinite(forward_coefficient)) {
      return absl::OutOfRangeError("forward substitution is outside the representable range");
    }
    staged_solution[row] = forward_coefficient;
  }

  for (std::size_t remaining = unknown_count; remaining > 0; --remaining) {
    const std::size_t row = remaining - 1;
    double residual = staged_solution[row];
    for (std::size_t column = row + 1; column < unknown_count; ++column) {
      residual = std::fma(-lower_factor[column * unknown_count + row], staged_solution[column], residual);
    }
    const double solution_coefficient = residual / lower_factor[row * unknown_count + row];
    if (!std::isfinite(solution_coefficient)) {
      return absl::OutOfRangeError("back substitution is outside the representable range");
    }
    staged_solution[row] = solution_coefficient;
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
  const std::size_t unknown_count = rhs.size();
  if (unknown_count > kMaxDenseUnknowns) {
    return absl::UnimplementedError("dense SPD oracle supports at most 64 unknowns");
  }
  if (const auto status = ValidateSolveViews(row_major_matrix, rhs, solution, workspace); !status.ok()) {
    return status;
  }
  if (unknown_count == 0) {
    return absl::OkStatus();
  }
  for (const double entry : rhs) {
    if (!std::isfinite(entry)) {
      return absl::InvalidArgumentError("SPD right-hand side entries must be finite");
    }
  }

  const auto factor = std::span(workspace.factor).first(row_major_matrix.size());
  const auto matrix_scale = StageSymmetricLower(row_major_matrix, unknown_count, factor);
  if (!matrix_scale.ok()) {
    return matrix_scale.status();
  }
  if (const auto status = FactorCholesky(factor, unknown_count, *matrix_scale); !status.ok()) {
    return status;
  }

  const auto work = std::span(workspace.working_rhs).first(unknown_count);
  if (const auto status = CheckFactorConditioning(factor, *matrix_scale, work); !status.ok()) {
    return status;
  }
  if (const auto status = SolveFactoredSystem(factor, rhs, work); !status.ok()) {
    return status;
  }
  std::ranges::copy(work, solution.begin());
  return absl::OkStatus();
}

}  // namespace fluidity
