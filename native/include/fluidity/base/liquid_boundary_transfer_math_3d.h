#pragma once
#include "fluidity/base/liquid_boundary_math.h"
#include "fluidity/base/liquid_boundary_transfer_types_3d.h"
#if defined(__CUDACC__)
#include <cuda/std/span>
#else
#include <span>
#endif
namespace fluidity::internal {
struct LiquidGhostFitSample3D {
  Vec3 offset_in_cell_widths;  // Dimensionless displacement from the target.
  double velocity;  // Matching staggered-component m/s.
  double row_scale = 1.0;  // Square root of the relative least-squares weight.
};
struct LiquidGhostAffineFit3D {
  bool full_rank;  // False means coefficients must not be used.
  double velocity_at_target;
  Vec3 velocity_slopes_in_cell_widths;  // m/s per normalized spatial coordinate.
};

// A radius-two sphere on the canonical MAC lattice contains 33 offsets
// within one component. Across components, two half-cell shifts give 36.
// The spherical test includes its boundary, and sparse ownership can only remove donors.
inline constexpr std::size_t kMaxRadiusTwoGhostDonors3D = 36;

FLUIDITY_HD inline bool TransferFinite(Vec3 value) {
  return boundary_internal::numeric::isfinite(value.x) && boundary_internal::numeric::isfinite(value.y)
      && boundary_internal::numeric::isfinite(value.z);
}
// Bounded column-pivoted QR in h units. Recompute norms, retain column ties,
// and reject pivots below 1e-10 of the largest without squaring conditioning.
// Accepts at most kSampleCapacity finite rows in caller order. Excess or
// invalid rows return kInvalid, and nonfinite arithmetic returns kRange.
// Rank loss returns a successful zero result with full_rank=false.
template <std::size_t kSampleCapacity>
FLUIDITY_HD inline boundary_internal::Value<LiquidGhostAffineFit3D> FitBoundedLiquidGhostAffineCore3D(
    boundary_internal::numeric::span<const LiquidGhostFitSample3D> samples
) {
  static_assert(kSampleCapacity >= 4 && kSampleCapacity <= 125);
  if (samples.size() > kSampleCapacity) {
    return boundary_internal::Error::kInvalid;
  }
  boundary_internal::numeric::array<boundary_internal::numeric::array<double, 4>, kSampleCapacity> matrix{};
  boundary_internal::numeric::array<double, kSampleCapacity> rhs{};
  double velocity_scale = 0.0;
  for (std::size_t row = 0; row < samples.size(); ++row) {
    const auto& sample = samples[row];
    if (!TransferFinite(sample.offset_in_cell_widths) || !boundary_internal::numeric::isfinite(sample.velocity)) {
      return boundary_internal::Error::kInvalid;
    }
    const double scale = sample.row_scale;
    if (!boundary_internal::numeric::isfinite(scale) || scale <= 0) {
      return boundary_internal::Error::kInvalid;
    }
    matrix[row] = {
        scale,
        scale * sample.offset_in_cell_widths.x,
        scale * sample.offset_in_cell_widths.y,
        scale * sample.offset_in_cell_widths.z,
    };
    velocity_scale = boundary_internal::numeric::max(velocity_scale, boundary_internal::numeric::abs(sample.velocity));
  }
  if (samples.size() < 4) {
    return LiquidGhostAffineFit3D{};
  }
  if (velocity_scale == 0.0) {
    velocity_scale = 1.0;
  }
  for (std::size_t row = 0; row < samples.size(); ++row) {
    rhs[row] = samples[row].row_scale * (samples[row].velocity / velocity_scale);
  }
  boundary_internal::numeric::array<std::size_t, 4> column_order{0, 1, 2, 3};
  double largest_pivot = 0.0;
  // Householder QR with recomputed residual norms avoids squaring a condition
  // number as normal equations would. Strict pivot comparison gives stable
  // column-order ties. All spatial columns are dimensionless offsets / h.
  for (std::size_t pivot = 0; pivot < 4; ++pivot) {
    std::size_t selected_column = pivot;
    double selected_norm = 0.0;
    for (std::size_t column = pivot; column < 4; ++column) {
      double norm = 0.0;
      for (std::size_t row = pivot; row < samples.size(); ++row) {
        norm = boundary_internal::numeric::hypot(norm, matrix[row][column]);
      }
      if (norm > selected_norm) {
        selected_norm = norm;
        selected_column = column;
      }
    }
    if (!boundary_internal::numeric::isfinite(selected_norm)) {
      return boundary_internal::Error::kRange;
    }
    largest_pivot = boundary_internal::numeric::max(largest_pivot, selected_norm);
    if (selected_norm == 0.0 || selected_norm < 1e-10 * largest_pivot) {
      return LiquidGhostAffineFit3D{};
    }
    boundary_internal::numeric::swap(column_order[pivot], column_order[selected_column]);
    for (std::size_t row = 0; row < samples.size(); ++row) {
      boundary_internal::numeric::swap(matrix[row][pivot], matrix[row][selected_column]);
    }
    const double diagonal = -boundary_internal::numeric::copysign(selected_norm, matrix[pivot][pivot]);
    boundary_internal::numeric::array<double, kSampleCapacity> reflector{};
    double reflector_norm = 0.0;
    for (std::size_t row = pivot; row < samples.size(); ++row) {
      // Divide before subtracting: each reflector entry stays bounded by two.
      reflector[row] = matrix[row][pivot] / selected_norm;
      if (row == pivot) {
        reflector[row] -= diagonal / selected_norm;
      }
      reflector_norm = boundary_internal::numeric::hypot(reflector_norm, reflector[row]);
    }
    for (std::size_t row = pivot; row < samples.size(); ++row) {
      reflector[row] /= reflector_norm;
    }
    for (std::size_t column = pivot + 1; column < 4; ++column) {
      double projection = 0.0;
      for (std::size_t row = pivot; row < samples.size(); ++row) {
        projection += reflector[row] * matrix[row][column];
      }
      for (std::size_t row = pivot; row < samples.size(); ++row) {
        matrix[row][column] -= 2.0 * reflector[row] * projection;
        if (!boundary_internal::numeric::isfinite(matrix[row][column])) {
          return boundary_internal::Error::kRange;
        }
      }
    }
    double rhs_projection = 0.0;
    for (std::size_t row = pivot; row < samples.size(); ++row) {
      rhs_projection += reflector[row] * rhs[row];
    }
    for (std::size_t row = pivot; row < samples.size(); ++row) {
      rhs[row] -= 2.0 * reflector[row] * rhs_projection;
    }
    matrix[pivot][pivot] = diagonal;
  }
  boundary_internal::numeric::array<double, 4> pivoted_coefficients{};
  boundary_internal::numeric::array<double, 4> coefficients{};
  for (int pivot = 3; pivot >= 0; --pivot) {
    double residual = rhs[pivot];
    for (int column = pivot + 1; column < 4; ++column) {
      residual -= matrix[pivot][column] * pivoted_coefficients[column];
    }
    pivoted_coefficients[pivot] = residual / matrix[pivot][pivot];
    coefficients[column_order[pivot]] = pivoted_coefficients[pivot] * velocity_scale;
    if (!boundary_internal::numeric::isfinite(coefficients[column_order[pivot]])) {
      return boundary_internal::Error::kRange;
    }
  }
  return LiquidGhostAffineFit3D{
      .full_rank = true,
      .velocity_at_target = coefficients[0],
      .velocity_slopes_in_cell_widths = {coefficients[1], coefficients[2], coefficients[3]},
  };
}
// Fits up to 125 caller-ordered samples with the same rank and error contract.
FLUIDITY_HD inline boundary_internal::Value<LiquidGhostAffineFit3D> FitLiquidGhostAffineCore3D(
    boundary_internal::numeric::span<const LiquidGhostFitSample3D> samples
) {
  return FitBoundedLiquidGhostAffineCore3D<125>(samples);
}
}  // namespace fluidity::internal
