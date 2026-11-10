#pragma once

#include "fluidity/base/liquid_boundary_math.h"

namespace fluidity {
struct LiquidRegularCellTransport3D {
  bool active{};  // False selects the original quadratic transport and ignores all samples below.
  Vec3 position_in_cell_widths{};  // Local X/Y/Z fractions from the cell's low corner, in [0,1].
  Vec3 low_face_normal_velocity{};  // Projected Cartesian components at the low X/Y/Z faces, m/s.
  Vec3 high_face_normal_velocity{};  // Projected Cartesian components at the high X/Y/Z faces, m/s.
};

namespace internal {
struct RegularCellTransportField3D {
  Vec3 advecting_world_velocity;  // m/s.
  double divergence;  // Inverse seconds, from the same field as advection.
};

// Evaluate an active record on a positive normal cell width in meters. Each
// component varies only along its own axis. Invalid inputs return kInvalid;
// a nonfinite velocity or derivative returns kRange.
FLUIDITY_HD inline boundary_internal::Value<RegularCellTransportField3D>
EvaluateRegularCellTransport3D(const LiquidRegularCellTransport3D& sample, double cell_width) {
  using boundary_internal::Error;
  namespace numeric = boundary_internal::numeric;
  if (!numeric::isnormal(cell_width) || cell_width <= 0) {
    return Error::kInvalid;
  }
  constexpr double Vec3::* kComponents[]{&Vec3::x, &Vec3::y, &Vec3::z};
  Vec3 advecting_world_velocity{};
  Vec3 normal_derivatives{};
  for (const auto component : kComponents) {
    const double position_fraction = sample.position_in_cell_widths.*component;
    const double low_normal_velocity = sample.low_face_normal_velocity.*component;
    const double high_normal_velocity = sample.high_face_normal_velocity.*component;
    if (!numeric::isfinite(position_fraction) || position_fraction < 0 || position_fraction > 1) {
      return Error::kInvalid;
    }
    if (!numeric::isfinite(low_normal_velocity) || !numeric::isfinite(high_normal_velocity)) {
      return Error::kInvalid;
    }
    const double normal_velocity_difference = high_normal_velocity - low_normal_velocity;
    const double normal_velocity = low_normal_velocity + position_fraction * normal_velocity_difference;
    const double normal_derivative = normal_velocity_difference / cell_width;
    if (!numeric::isfinite(normal_velocity) || !numeric::isfinite(normal_derivative)) {
      return Error::kRange;
    }
    advecting_world_velocity.*component = normal_velocity;
    normal_derivatives.*component = normal_derivative;
  }
  const double divergence = (normal_derivatives.x + normal_derivatives.y) + normal_derivatives.z;
  if (!numeric::isfinite(divergence)) {
    return Error::kRange;
  }
  return RegularCellTransportField3D{advecting_world_velocity, divergence};
}
}  // namespace internal
}  // namespace fluidity
