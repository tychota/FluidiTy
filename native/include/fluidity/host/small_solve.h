#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "absl/status/status.h"

namespace fluidity {

inline constexpr std::size_t kMaxDenseUnknowns = 64;

struct SmallSolveWorkspace {
  // The averaged lower triangle of A, overwritten in place by L with row stride n.
  // The upper triangle is unused.
  std::array<double, kMaxDenseUnknowns * kMaxDenseUnknowns> factor;
  // Holds one scaled inverse column during conditioning, then the forward solve and final x.
  std::array<double, kMaxDenseUnknowns> working_rhs;
};

// Solves A x = b by dense Cholesky factorization using caller-owned scratch.
// With n = rhs.size(), the row-major matrix and solution must have exactly n*n and n
// entries. The empty system succeeds. The matrix and RHS can share storage.
// Solution and the entire workspace must be disjoint
// from each other and both inputs. Solution is written only on success. Scratch may change
// on failure and has no persistent state.
//
// kUnimplemented means n exceeds kMaxDenseUnknowns. kInvalidArgument means incorrect extents,
// overlap, nonfinite input, or normalized off-diagonal entries differing by more than eight double
// epsilons. Entries are normalized by the largest absolute matrix entry for the symmetry check.
// Off-diagonal pairs within that tolerance are averaged once. kFailedPrecondition means zero or
// subnormal matrix scale, a nonpositive, subnormal or numerically unsafe pivot, or failed conditioning.
// kOutOfRange means nonfinite arithmetic during factorization, conditioning or substitution.
//
// With s = max |A_ij| and L the computed lower factor, the O(n^3) conditioning check requires
// the estimate of s * ||L^-1||_F^2 to be below 1 / (4 * n * (n + 1) * double epsilon).
// This conservative screen is independent of the RHS. Accuracy is intended for well-conditioned
// systems whose nonzero arithmetic stays in normal double range.
absl::Status SolveSmallSpd(
    std::span<const double> row_major_matrix,
    std::span<const double> rhs,
    std::span<double> solution,
    SmallSolveWorkspace& workspace
);

}  // namespace fluidity
