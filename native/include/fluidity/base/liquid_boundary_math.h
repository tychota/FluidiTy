#pragma once
#include <cstddef>

#include "fluidity/base/host_device.h"
#include "fluidity/base/mac_types_3d.h"
#if defined(__CUDACC__)
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#else
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#endif
namespace fluidity {
// The open top has no y maximum. The low-x piston translates along world +x;
// the opposite wall, floor, and two z walls are fixed.
struct AxisAlignedLiquidTank3D {
  double start_low_x;  // World x in meters at substep start.
  double high_x;  // World x in meters.
  double floor_y;  // World y in meters.
  double low_z;  // World z in meters.
  double high_z;  // World z in meters.
  double piston_velocity_x;  // Signed world x velocity in m/s.
};
}  // namespace fluidity

namespace fluidity::boundary_internal {
#if defined(__CUDACC__)
namespace numeric = cuda::std;
#else
namespace numeric = std;
#endif
// Arithmetic outcomes cross the device ABI without host Status or containers.
enum class Error {
  kNone,
  kInvalid,
  kPrecondition,
  kRange,
};
template <typename T>
struct Maybe {
  T value{};
  bool present = false;
  Maybe() = default;
  FLUIDITY_HD Maybe(T input) : value(input), present(true) {
  }
  FLUIDITY_HD bool has_value() const {
    return present;
  }
  FLUIDITY_HD const T& operator*() const {
    return value;
  }
};
template <typename T>
struct Value {
  T value{};
  Error error = Error::kNone;
  FLUIDITY_HD Value(T input) : value(input) {
  }
  FLUIDITY_HD Value(Error input) : error(input) {
  }
  FLUIDITY_HD bool ok() const {
    return error == Error::kNone;
  }
  FLUIDITY_HD Error status() const {
    return error;
  }
  FLUIDITY_HD T& operator*() {
    return value;
  }
  FLUIDITY_HD const T& operator*() const {
    return value;
  }
  FLUIDITY_HD T* operator->() {
    return &value;
  }
  FLUIDITY_HD const T* operator->() const {
    return &value;
  }
};
struct RoundedSum {
  double value;
  double residual;
};

// FastTwoSum with the larger operand first. The residual retains offsets
// that disappear from the rounded coordinate or velocity difference.
FLUIDITY_HD inline RoundedSum AddWithResidual(double left, double right) {
  if (numeric::abs(left) < numeric::abs(right)) {
    numeric::swap(left, right);
  }
  const double sum = left + right;
  if (!numeric::isfinite(sum)) {
    return {sum, 0};
  }
  return {sum, right - (sum - left)};
}

// Geometry admission and contact supply a positive finite step duration.
// Linear translation bounds the width throughout the step by its endpoint widths.
FLUIDITY_HD inline Error CheckTankStep(const AxisAlignedLiquidTank3D& tank, double step_duration) {
  if (!numeric::isfinite(tank.start_low_x) || !numeric::isfinite(tank.high_x) || !numeric::isfinite(tank.floor_y)
      || !numeric::isfinite(tank.low_z) || !numeric::isfinite(tank.high_z)
      || !numeric::isfinite(tank.piston_velocity_x)) {
    return Error::kInvalid;
  }
  if (tank.start_low_x >= tank.high_x || tank.low_z >= tank.high_z) {
    return Error::kInvalid;
  }
  const double piston_displacement = tank.piston_velocity_x * step_duration;
  const double piston_end_x = numeric::fma(tank.piston_velocity_x, step_duration, tank.start_low_x);
  const double start_width_x = tank.high_x - tank.start_low_x;
  const double end_width_x = tank.high_x - piston_end_x;
  const double width_z = tank.high_z - tank.low_z;
  if (!numeric::isfinite(piston_displacement) || !numeric::isfinite(piston_end_x) || !numeric::isfinite(start_width_x)
      || !numeric::isfinite(end_width_x) || !numeric::isfinite(width_z)) {
    return Error::kRange;
  }
  // Requiring the rounded end pose to lie strictly inside also excludes an
  // exact closure and an end clearance too small to resolve in world space.
  if (piston_end_x >= tank.high_x) {
    return Error::kPrecondition;
  }
  return Error::kNone;
}

enum class ObstacleKind {
  kAbsent,
  kHalfspace,
  kSlab,
  kOpenTank
};
// Planar coordinates are world meters at start. Only the active shape is read.
struct Obstacle {
  ObstacleKind kind = ObstacleKind::kAbsent;
  Vec3 normal{};
  double coordinate = 0;
  double half_width = 0;
  Vec3 velocity{};
  AxisAlignedLiquidTank3D tank{};
};
struct ContactInput {
  Vec3 start_world_position{};
  Vec3 stored_world_velocity{};
  Vec3 advecting_world_velocity{};
  Obstacle obstacle{};
  double step_duration = 0;
  double friction_coefficient = 0;
  double penetration_tolerance = 0;
};
struct ContactOutput {
  Vec3 world_position{};
  Vec3 stored_world_velocity{};
  Vec3 advecting_world_velocity{};
  Maybe<double> contact_fraction{};
  Vec3 outward_normal{};
  double incoming_stored_relative_normal_speed = 0;
  double incoming_advecting_relative_normal_speed = 0;
  int diagnostic = 0;  // 0 none, 1 swept, 2 initial penetration.
};
}  // namespace fluidity::boundary_internal
