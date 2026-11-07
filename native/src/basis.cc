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

// Computes the three nodes and dimensionless derivatives for one lattice coordinate.
absl::StatusOr<AxisStencil> QuadraticAxis(double coordinate) {
  const double integer_part = std::floor(coordinate);
  constexpr double index_limit = 0x1p63;
  const bool index_fits = integer_part >= -index_limit && integer_part < index_limit;
  if (!index_fits) {
    return absl::OutOfRangeError("quadratic stencil node is outside the int64_t range");
  }
  const auto integral_index = static_cast<std::int64_t>(integer_part);

  // Compute floor(coordinate - 0.5) without rounding away the half-cell shift at large indices.
  const bool upper_half = coordinate - integer_part >= 0.5;
  if (!upper_half && integral_index == std::numeric_limits<std::int64_t>::min()) {
    return absl::OutOfRangeError("quadratic stencil node is outside the int64_t range");
  }
  const std::int64_t first_node = upper_half ? integral_index : integral_index - 1;
  if (first_node > std::numeric_limits<std::int64_t>::max() - 2) {
    return absl::OutOfRangeError("quadratic stencil node is outside the int64_t range");
  }

  const double center_delta = coordinate - static_cast<double>(first_node + 1);
  const double left_support_distance = 0.5 - center_delta;
  const double right_support_distance = 0.5 + center_delta;
  const std::array weights{
      0.5 * left_support_distance * left_support_distance,
      0.75 - center_delta * center_delta,
      0.5 * right_support_distance * right_support_distance,
  };
  return AxisStencil{
      .first_node = first_node,
      .weights = weights,
      .derivatives = {-left_support_distance, -2.0 * center_delta, right_support_distance},
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

  const auto x_stencil = QuadraticAxis(relative_position.x / spacing - offset.x);
  if (!x_stencil.ok()) {
    return x_stencil.status();
  }
  const auto y_stencil = QuadraticAxis(relative_position.y / spacing - offset.y);
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
