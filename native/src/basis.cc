#include "fluidity/host/basis.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/status/status.h"

namespace fluidity {
namespace {

constexpr std::size_t kAxisNodeCount = 3;

struct AxisStencil {
  std::int64_t first_node;
  std::array<double, kAxisNodeCount> weights;
  std::array<double, kAxisNodeCount> derivatives;
};

// Computes the three nodes and dimensionless derivatives for one axis. Keep the normalized
// position and offset separate so their difference retains precision near half-cell ties.
absl::StatusOr<AxisStencil> QuadraticAxis(double normalized_position, double lattice_offset) {
  const double rounded_lattice_coordinate = normalized_position - lattice_offset;
  constexpr double index_limit = 0x1p63;
  const bool coordinate_fits = rounded_lattice_coordinate >= -index_limit && rounded_lattice_coordinate <= index_limit;
  if (!coordinate_fits) {
    return absl::OutOfRangeError("quadratic stencil node is outside the int64_t range");
  }

  // TwoSum recovers the subtraction residual from the position and negated lattice offset.
  const double recovered_offset = normalized_position - rounded_lattice_coordinate;
  const double subtraction_residual =
      (normalized_position - (rounded_lattice_coordinate + recovered_offset)) + (recovered_offset - lattice_offset);
  double rounded_integral_part;
  const double rounded_fraction = std::modf(rounded_lattice_coordinate, &rounded_integral_part);
  double residual_integral_part;
  const double residual_fraction = std::modf(subtraction_residual, &residual_integral_part);

  // Round the corrected lattice coordinate to the nearest node, with exact ties going up.
  // Compare to the half-cell thresholds before adding the fractional parts.
  int half_cell_shift = 0;
  if (residual_fraction >= 0.5 - rounded_fraction) {
    half_cell_shift = 1;
  } else if (residual_fraction < -0.5 - rounded_fraction) {
    half_cell_shift = -1;
  }

  constexpr auto min_index = std::numeric_limits<std::int64_t>::min();
  constexpr auto max_index = std::numeric_limits<std::int64_t>::max();
  // The rounded coordinate may equal 2^63 even though the corrected center fits.
  const bool at_upper_limit = rounded_integral_part == index_limit;
  const auto provisional_center_index = at_upper_limit ? max_index : static_cast<std::int64_t>(rounded_integral_part);
  // The coordinate bound limits the subtraction residual to 1024.
  const auto center_index_correction =
      static_cast<std::int64_t>(residual_integral_part) + half_cell_shift + (at_upper_limit ? 1 : 0);
  // Check both support endpoints before applying the correction in signed arithmetic.
  const bool first_node_out_of_range =
      center_index_correction <= 0 && provisional_center_index < min_index + 1 - center_index_correction;
  if (first_node_out_of_range) {
    return absl::OutOfRangeError("quadratic stencil node is outside the int64_t range");
  }
  const bool last_node_out_of_range =
      center_index_correction >= 0 && provisional_center_index > max_index - 1 - center_index_correction;
  if (last_node_out_of_range) {
    return absl::OutOfRangeError("quadratic stencil node is outside the int64_t range");
  }

  const std::int64_t first_node = provisional_center_index + center_index_correction - 1;
  const double normalized_center_displacement = (rounded_fraction - half_cell_shift) + residual_fraction;
  // Keep the support-edge distances even when the center displacement rounds to a half cell.
  const double left_support_distance = ((0.5 + half_cell_shift) - rounded_fraction) - residual_fraction;
  const double right_support_distance = ((0.5 - half_cell_shift) + rounded_fraction) + residual_fraction;
  const std::array weights{
      0.5 * left_support_distance * left_support_distance,
      0.75 - normalized_center_displacement * normalized_center_displacement,
      0.5 * right_support_distance * right_support_distance,
  };
  return AxisStencil{
      .first_node = first_node,
      .weights = weights,
      .derivatives = {-left_support_distance, -2.0 * normalized_center_displacement, right_support_distance},
  };
}

}  // namespace

absl::StatusOr<Stencil2D> QuadraticStencil2D(Vec2 relative_position, double spacing, Vec2 offset) {
  const bool valid_spacing = spacing > 0.0 && std::isfinite(spacing);
  if (!valid_spacing) {
    return absl::InvalidArgumentError("stencil spacing must be finite and positive");
  }
  if (!std::isfinite(relative_position.x) || !std::isfinite(relative_position.y)) {
    return absl::InvalidArgumentError("stencil position must be finite");
  }
  if (!std::isfinite(offset.x) || !std::isfinite(offset.y)) {
    return absl::InvalidArgumentError("stencil offset must be finite");
  }

  const auto x_stencil = QuadraticAxis(relative_position.x / spacing, offset.x);
  if (!x_stencil.ok()) {
    return x_stencil.status();
  }
  const auto y_stencil = QuadraticAxis(relative_position.y / spacing, offset.y);
  if (!y_stencil.ok()) {
    return y_stencil.status();
  }

  Stencil2D stencil;
  for (std::size_t x_slot = 0; x_slot < kAxisNodeCount; ++x_slot) {
    for (std::size_t y_slot = 0; y_slot < kAxisNodeCount; ++y_slot) {
      const std::size_t node_index = kAxisNodeCount * x_slot + y_slot;
      stencil.nodes[node_index] = {
          .x = x_stencil->first_node + static_cast<std::int64_t>(x_slot),
          .y = y_stencil->first_node + static_cast<std::int64_t>(y_slot),
      };
      const double x_weight = x_stencil->weights[x_slot];
      const double y_weight = y_stencil->weights[y_slot];
      stencil.weights[node_index] = x_weight * y_weight;

      // Weight before dividing by tiny spacing to avoid spurious overflow.
      const Vec2 gradient{
          .x = x_stencil->derivatives[x_slot] * y_weight / spacing,
          .y = x_weight * y_stencil->derivatives[y_slot] / spacing,
      };
      if (!std::isfinite(gradient.x) || !std::isfinite(gradient.y)) {
        return absl::OutOfRangeError("stencil gradient is outside the representable range");
      }
      stencil.gradients[node_index] = gradient;
    }
  }
  return stencil;
}

}  // namespace fluidity
