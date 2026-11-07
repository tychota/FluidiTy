#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "absl/status/status.h"

namespace fluidity {

inline constexpr std::size_t kMaxDenseUnknowns = 64;

struct SmallSolveWorkspace {
  // Scratch for the lower-triangular factor, stored with row stride n.
  std::array<double, kMaxDenseUnknowns * kMaxDenseUnknowns> factor;
  // Scratch for forward substitution and the staged solution.
  std::array<double, kMaxDenseUnknowns> working_rhs;
};

// Solves A x = b by dense Cholesky factorization using caller-owned scratch.
// With n = rhs.size(), the row-major matrix and solution must have exactly n*n and n
// entries. The empty system succeeds. Solution and the entire workspace must be disjoint
// from each other and both inputs. Solution is written only on success. Scratch may change
// on failure and has no persistent state.
//
// kUnimplemented means n exceeds kMaxDenseUnknowns. kInvalidArgument means incorrect extents,
// overlap, nonfinite input, or asymmetry exceeding 8 double epsilons relative to max |A_ij|.
// Off-diagonal pairs within that tolerance are averaged once. kFailedPrecondition means
// a nonpositive pivot or one at most 64 double epsilons times max |A_ij|. kOutOfRange means
// a nonfinite computed factor or substitution result.
//
// Intended for well-conditioned systems whose nonzero arithmetic stays in normal double
// range. The pivot check is a safety screen and gives no accuracy guarantee outside this domain.
absl::Status SolveSmallSpd(
    std::span<const double> row_major_matrix,
    std::span<const double> rhs,
    std::span<double> solution,
    SmallSolveWorkspace& workspace
);

}  // namespace fluidity
