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

constexpr double kInfinity = numeric::numeric_limits<double>::infinity();

FLUIDITY_HD inline double AddDown(double left, double right) {
  const auto sum = AddWithResidual(left, right);
  return sum.residual < 0 ? numeric::nextafter(sum.value, -kInfinity) : sum.value;
}

FLUIDITY_HD inline double AddUp(double left, double right) {
  const auto sum = AddWithResidual(left, right);
  return sum.residual > 0 ? numeric::nextafter(sum.value, kInfinity) : sum.value;
}

struct ProjectionBounds {
  double estimate;
  double lower;
  double upper;

  FLUIDITY_HD bool finite() const {
    return numeric::isfinite(estimate) && numeric::isfinite(lower) && numeric::isfinite(upper);
  }

  // A certified sign suffices to classify an event. Computing the hit time
  // or impulse also needs a resolved magnitude after cancellation.
  FLUIDITY_HD bool magnitude_is_resolved() const {
    constexpr double kRelativeResolution = 16 * numeric::numeric_limits<double>::epsilon();
    return finite() && upper - lower <= kRelativeResolution * numeric::abs(estimate);
  }

  FLUIDITY_HD Maybe<int> sign() const {
    if (!finite()) {
      return {};
    }
    if (lower > 0) {
      return 1;
    }
    if (upper < 0) {
      return -1;
    }
    if (lower == 0 && upper == 0) {
      return 0;
    }
    return {};
  }
};

// Binary64 arithmetic in round-to-nearest. Retain sum and product residuals
// separately so an endpoint on an oblique or translated wall can be certified.
class ProjectedSum {
 public:
  FLUIDITY_HD void Add(double term) {
    if (!numeric::isfinite(sum_) || !numeric::isfinite(term)) {
      sum_ = kInfinity;
      return;
    }
    const auto sum = AddWithResidual(sum_, term);
    sum_ = sum.value;
    AddCorrection(sum.residual);
  }

  FLUIDITY_HD void AddDot(Vec3 normal, Vec3 vector) {
    AddScaledProduct(normal.x, vector.x, 1);
    AddScaledProduct(normal.y, vector.y, 1);
    AddScaledProduct(normal.z, vector.z, 1);
  }

  FLUIDITY_HD void AddRelativeMotion(Vec3 normal, Vec3 world_velocity, Vec3 wall_velocity, double duration) {
    AddRelativeComponent(normal.x, world_velocity.x, wall_velocity.x, duration);
    AddRelativeComponent(normal.y, world_velocity.y, wall_velocity.y, duration);
    AddRelativeComponent(normal.z, world_velocity.z, wall_velocity.z, duration);
  }

  FLUIDITY_HD void AddProduct(double left, double right) {
    const double product = left * right;
    Add(product);
    if (numeric::isfinite(product) && left != 0 && right != 0 && numeric::abs(left) != 1 && numeric::abs(right) != 1) {
      AddCorrection(numeric::fma(left, right, -product));
      // A product of two binary64 significands can require 106 bits.
      // Below this binade its residual can itself round in the subnormal range.
      if (numeric::abs(product) <= 0x1p-969) {
        const double residual_roundoff = numeric::numeric_limits<double>::denorm_min();
        lower_correction_ = AddDown(lower_correction_, -residual_roundoff);
        upper_correction_ = AddUp(upper_correction_, residual_roundoff);
      }
    }
  }

  FLUIDITY_HD ProjectionBounds Bounds() const {
    if (!numeric::isfinite(sum_) || !numeric::isfinite(lower_correction_) || !numeric::isfinite(upper_correction_)) {
      return {kInfinity, kInfinity, kInfinity};
    }
    return {
        sum_ + numeric::midpoint(lower_correction_, upper_correction_),
        AddDown(sum_, lower_correction_),
        AddUp(sum_, upper_correction_),
    };
  }

 private:
  FLUIDITY_HD void AddCorrection(double correction) {
    if (!numeric::isfinite(lower_correction_) || !numeric::isfinite(upper_correction_)) {
      sum_ = kInfinity;
      return;
    }
    lower_correction_ = AddDown(lower_correction_, correction);
    upper_correction_ = AddUp(upper_correction_, correction);
  }

  FLUIDITY_HD void AddScaledProduct(double left, double right, double scale) {
    if (left == 0 || right == 0 || scale == 0) {
      return;
    }
    if (scale == 1) {
      AddProduct(left, right);
      return;
    }
    if (numeric::abs(left) == 1 || numeric::abs(right) == 1) {
      AddProduct(left * right, scale);
      return;
    }
    const double product = left * right;
    if (!numeric::isfinite(product) || numeric::abs(product) <= 0x1p-969) {
      sum_ = kInfinity;
      return;
    }
    AddProduct(product, scale);
    AddProduct(numeric::fma(left, right, -product), scale);
  }

  FLUIDITY_HD void AddRelativeComponent(double normal, double world_velocity, double wall_velocity, double duration) {
    if (normal == 0 || duration == 0) {
      return;
    }
    const auto relative = AddWithResidual(world_velocity, -wall_velocity);
    if (!numeric::isfinite(relative.value)) {
      sum_ = kInfinity;
      return;
    }
    AddScaledProduct(normal, relative.value, duration);
    AddScaledProduct(normal, relative.residual, duration);
  }

  double sum_ = 0;
  double lower_correction_ = 0;
  double upper_correction_ = 0;
};

// Geometry and contact supply a positive finite duration. Zero checks a single pose.
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

struct TankLatticeAxis {
  double world_origin;
  double lattice_index;
  double cell_width;
};

// Subtract in the lattice frame. A rounded world anchor alone can coincide
// with a tank plane even when the true lattice face is strictly inside.
FLUIDITY_HD inline ProjectedSum TankPlaneOffset(double world_coordinate, const TankLatticeAxis& lattice) {
  ProjectedSum offset;
  offset.Add(world_coordinate);
  offset.Add(-lattice.world_origin);
  offset.AddProduct(-lattice.cell_width, lattice.lattice_index);
  return offset;
}

// Signed liquid distance at an original grid sample, retaining the product
// and translation residuals that a reconstructed world-space y can lose.
FLUIDITY_HD inline Value<double>
HorizontalPlaneLatticeDistance(double world_surface_height_y, const TankLatticeAxis& lattice_y) {
  if (!numeric::isfinite(world_surface_height_y) || !numeric::isfinite(lattice_y.world_origin)
      || !numeric::isfinite(lattice_y.lattice_index) || !numeric::isfinite(lattice_y.cell_width)
      || lattice_y.cell_width <= 0) {
    return Error::kInvalid;
  }
  const auto offset = TankPlaneOffset(world_surface_height_y, lattice_y).Bounds();
  if (!offset.magnitude_is_resolved()) {
    return Error::kRange;
  }
  return offset.estimate == 0 ? 0.0 : -offset.estimate;
}

struct TankWallSample {
  double inward_distance;
  Vec3 inward_normal;
  Vec3 wall_velocity;
};

// The active constraint has the least inward clearance. Ties use x, y, z
// order, then the low side first. An open top has no constraint.
FLUIDITY_HD inline Value<TankWallSample>
SelectTankWall(const AxisAlignedLiquidTank3D& tank, const numeric::array<double, 5>& inward_distances) {
  const TankWallSample walls[]{
      {inward_distances[0], {1, 0, 0}, {tank.piston_velocity_x, 0, 0}},
      {inward_distances[1], {-1, 0, 0}, {}},
      {inward_distances[2], {0, 1, 0}, {}},
      {inward_distances[3], {0, 0, 1}, {}},
      {inward_distances[4], {0, 0, -1}, {}},
  };
  TankWallSample active = walls[0];
  for (const auto& wall : walls) {
    if (!numeric::isfinite(wall.inward_distance)) {
      return Error::kRange;
    }
    if (wall.inward_distance < active.inward_distance) {
      active = wall;
    }
  }
  return active;
}

FLUIDITY_HD inline Value<TankWallSample> SampleTankWall(const AxisAlignedLiquidTank3D& tank, Vec3 world_position) {
  return SelectTankWall(
      tank,
      {
          world_position.x - tank.start_low_x,
          tank.high_x - world_position.x,
          world_position.y - tank.floor_y,
          world_position.z - tank.low_z,
          tank.high_z - world_position.z,
      }
  );
}

FLUIDITY_HD inline Vec3 TankFaceAnchorInCellWidths(FaceIndex3D face) {
  return {
      face.lattice_index.x + (face.axis == FaceNormalAxis::kX ? 0.0 : 0.5),
      face.lattice_index.y + (face.axis == FaceNormalAxis::kY ? 0.0 : 0.5),
      face.lattice_index.z + (face.axis == FaceNormalAxis::kZ ? 0.0 : 0.5),
  };
}

// Resolve all five inward clearances from the original staggered lattice.
// An uncertified magnitude cannot choose the wall or classify its ghost band.
FLUIDITY_HD inline Value<TankWallSample>
SampleTankFace(const AxisAlignedLiquidTank3D& tank, UniformMacGeometry3D geometry, Vec3 anchor_in_cell_widths) {
  const TankLatticeAxis x{geometry.world_origin.x, anchor_in_cell_widths.x, geometry.cell_width};
  const TankLatticeAxis y{geometry.world_origin.y, anchor_in_cell_widths.y, geometry.cell_width};
  const TankLatticeAxis z{geometry.world_origin.z, anchor_in_cell_widths.z, geometry.cell_width};
  const ProjectionBounds offsets[]{
      TankPlaneOffset(tank.start_low_x, x).Bounds(),
      TankPlaneOffset(tank.high_x, x).Bounds(),
      TankPlaneOffset(tank.floor_y, y).Bounds(),
      TankPlaneOffset(tank.low_z, z).Bounds(),
      TankPlaneOffset(tank.high_z, z).Bounds(),
  };
  for (const auto& offset : offsets) {
    if (!offset.magnitude_is_resolved()) {
      return Error::kRange;
    }
  }
  return SelectTankWall(
      tank,
      {-offsets[0].estimate, offsets[1].estimate, -offsets[2].estimate, -offsets[3].estimate, offsets[4].estimate}
  );
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
