#pragma once

#include "fluidity/base/liquid_boundary_math.h"

namespace fluidity {

// Inclusive binary64 enclosure of a physical or lattice-relative quantity.
struct WettedPistonBounds3D {
  double lower = 0;
  double upper = 0;
};

namespace wetted_piston_internal {
using boundary_internal::Error;
using boundary_internal::Value;
namespace numeric = boundary_internal::numeric;
using boundary_internal::kInfinity;
using Bounds = WettedPistonBounds3D;

inline constexpr int kBandsPerPatch = 64;
inline constexpr std::size_t kDevicePatchBatch = 32;

struct Field {
  UniformMacGeometry3D geometry;
  AxisAlignedLiquidTank3D tank;
  double sphere_radius = 0;
  double plane_height_y = 0;
  bool is_plane = false;
};

// Section in cell-width units. The center bounds are in signed lattice
// coordinates; the radius is squared and has no lattice origin.
struct Disk {
  Bounds lattice_y{};
  Bounds lattice_z{};
  Bounds radius_squared{};
  Error error = Error::kNone;
};

struct PatchIndex {
  std::int32_t y = 0;
  std::int32_t z = 0;
  bool operator==(const PatchIndex&) const = default;
};

struct PatchMeasure {
  Bounds area{};  // m²
  double midpoint_flux = 0;  // Signed m³/s.
  bool all_wet_sources_cross_lower_edge = true;
  Error error = Error::kNone;
};

struct Interval {
  double low;
  double high;
};

FLUIDITY_HD inline bool Finite(Bounds bounds) {
  return numeric::isfinite(bounds.lower) && numeric::isfinite(bounds.upper) && bounds.lower <= bounds.upper;
}

FLUIDITY_HD inline Bounds Point(double value) {
  return {value, value};
}

FLUIDITY_HD inline Bounds Add(Bounds a, Bounds b) {
  return {
      boundary_internal::AddDown(a.lower, b.lower),
      boundary_internal::AddUp(a.upper, b.upper),
  };
}

FLUIDITY_HD inline Bounds Negate(Bounds a) {
  return {-a.upper, -a.lower};
}

FLUIDITY_HD inline Bounds Subtract(Bounds a, Bounds b) {
  return Add(a, Negate(b));
}

// Product, quotient and square-root residuals use explicit FMA. When the
// residual itself could underflow, step in both directions. Each rounded
// operation is at most one adjacent double from either directed result.
FLUIDITY_HD inline double DirectedResult(double rounded, double residual, double scale, bool up) {
  if (!numeric::isfinite(rounded)) {
    return rounded;
  }
  constexpr double kResidualUnderflowThreshold = 0x1p-969;
  const bool uncertain = residual == 0 && numeric::abs(scale) <= kResidualUnderflowThreshold;
  if (uncertain || (up ? residual > 0 : residual < 0)) {
    return numeric::nextafter(rounded, up ? kInfinity : -kInfinity);
  }
  return rounded;
}

FLUIDITY_HD inline double DirectedProduct(double a, double b, bool up) {
  const double product = a * b;
  if (a == 0 || b == 0 || numeric::abs(a) == 1 || numeric::abs(b) == 1 || !numeric::isfinite(product)) {
    return product;
  }
  return DirectedResult(product, numeric::fma(a, b, -product), product, up);
}

FLUIDITY_HD inline Bounds MultiplyNonnegative(Bounds a, Bounds b) {
  return {
      numeric::max(0.0, DirectedProduct(a.lower, b.lower, false)),
      DirectedProduct(a.upper, b.upper, true),
  };
}

FLUIDITY_HD inline Bounds WettedAreaToSignedWallFlux(Bounds area, double velocity) {
  const auto magnitude = MultiplyNonnegative(area, Point(numeric::abs(velocity)));
  return velocity < 0 ? Negate(magnitude) : magnitude;
}

FLUIDITY_HD inline double DirectedQuotient(double value, double divisor, bool up) {
  const double quotient = value / divisor;
  if (value == 0 || divisor == 1 || !numeric::isfinite(quotient)) {
    return quotient;
  }
  return DirectedResult(quotient, numeric::fma(-quotient, divisor, value), value, up);
}

FLUIDITY_HD inline Bounds Divide(Bounds a, double positive_divisor) {
  return {
      DirectedQuotient(a.lower, positive_divisor, false),
      DirectedQuotient(a.upper, positive_divisor, true),
  };
}

FLUIDITY_HD inline double DirectedRoot(double value, bool up) {
  if (value <= 0) {
    return 0;
  }
  const double root = numeric::sqrt(value);
  const double outward = DirectedResult(root, numeric::fma(-root, root, value), value, up);
  return numeric::max(0.0, outward);
}

FLUIDITY_HD inline Bounds Root(Bounds a) {
  return {DirectedRoot(a.lower, false), DirectedRoot(a.upper, true)};
}

FLUIDITY_HD inline Bounds Maximum(Bounds a, Bounds b) {
  return {numeric::max(a.lower, b.lower), numeric::max(a.upper, b.upper)};
}

FLUIDITY_HD inline Bounds Minimum(Bounds a, Bounds b) {
  return {numeric::min(a.lower, b.lower), numeric::min(a.upper, b.upper)};
}

FLUIDITY_HD inline Bounds Magnitude(Bounds a) {
  return {
      numeric::max(0.0, numeric::max(a.lower, -a.upper)),
      numeric::max(a.upper, -a.lower),
  };
}

FLUIDITY_HD inline Bounds Square(Bounds a) {
  const auto magnitude = Magnitude(a);
  return MultiplyNonnegative(magnitude, magnitude);
}

FLUIDITY_HD inline Bounds
LatticeOffset(double world_coordinate, double world_origin, double lattice_index, double cell_width) {
  const auto world_offset =
      boundary_internal::TankPlaneOffset(world_coordinate, {world_origin, lattice_index, cell_width}).Bounds();
  if (!world_offset.finite()) {
    return {kInfinity, kInfinity};
  }
  return Divide({world_offset.lower, world_offset.upper}, cell_width);
}

// The factored section retains the positive difference near tangency. Scaling
// the factors before multiplying avoids squaring a large world-space radius.
FLUIDITY_HD inline Disk Section(const Field& field, Vec3 sphere_world_center) {
  const double cell_width = field.geometry.cell_width;
  const auto piston_distance = Magnitude(Subtract(Point(sphere_world_center.x), Point(field.tank.start_low_x)));
  if (!Finite(piston_distance)) {
    return {.error = Error::kRange};
  }
  if (piston_distance.lower >= field.sphere_radius) {
    return {};
  }
  const auto sphere_radius = Point(field.sphere_radius);
  const auto world_radius_minus_distance = Maximum(Point(0), Subtract(sphere_radius, piston_distance));
  const auto lattice_radius_plus_distance = Add(Divide(sphere_radius, cell_width), Divide(piston_distance, cell_width));
  const auto lattice_disk_radius_squared =
      MultiplyNonnegative(Divide(world_radius_minus_distance, cell_width), lattice_radius_plus_distance);
  const auto center_lattice_y = LatticeOffset(sphere_world_center.y, field.geometry.world_origin.y, 0, cell_width);
  const auto center_lattice_z = LatticeOffset(sphere_world_center.z, field.geometry.world_origin.z, 0, cell_width);
  if (!Finite(lattice_disk_radius_squared) || !Finite(center_lattice_y) || !Finite(center_lattice_z)
      || lattice_disk_radius_squared.upper <= 0) {
    return {.error = Error::kRange};
  }
  return {
      .lattice_y = center_lattice_y,
      .lattice_z = center_lattice_z,
      .radius_squared = lattice_disk_radius_squared,
  };
}

struct Rectangle {
  Bounds low_y;
  Bounds high_y;
  Bounds low_z;
  Bounds high_z;
};

FLUIDITY_HD inline Bounds Width(Bounds low, Bounds high) {
  return Maximum(Point(0), Subtract(high, low));
}

// Distance to any point in the closed interval, evaluated at its nearest
// and farthest positions. Clipping boundaries may themselves be enclosures.
FLUIDITY_HD inline Bounds Nearest(Bounds center, Bounds low, Bounds high) {
  return Maximum(Point(0), Maximum(Subtract(low, center), Subtract(center, high)));
}

FLUIDITY_HD inline Bounds Farthest(Bounds center, Bounds low, Bounds high) {
  return Maximum(Magnitude(Subtract(center, low)), Magnitude(Subtract(center, high)));
}

FLUIDITY_HD inline Bounds Gap(Bounds radius_squared, Bounds y_distance, Bounds z_distance) {
  return Subtract(Subtract(radius_squared, Square(y_distance)), Square(z_distance));
}

FLUIDITY_HD inline bool IntervalLess(Interval a, Interval b) {
  return a.low < b.low || (a.low == b.low && a.high < b.high);
}

FLUIDITY_HD inline void SiftIntervals(Interval* intervals, std::size_t count, std::size_t root) {
  while (root < count / 2) {
    std::size_t child = root * 2 + 1;
    if (child + 1 < count && IntervalLess(intervals[child], intervals[child + 1])) {
      ++child;
    }
    if (!IntervalLess(intervals[root], intervals[child])) {
      break;
    }
    numeric::swap(intervals[root], intervals[child]);
    root = child;
  }
}

// In-place heap sort bounds cost even for a fully overlapping particle union.
// The two banks respectively contain only inner or only outer y intervals.
FLUIDITY_HD inline double UnionLength(Interval* intervals, std::size_t count, bool upper) {
  if (count == 0) {
    return 0;
  }
  for (std::size_t start = count / 2; start > 0; --start) {
    SiftIntervals(intervals, count, start - 1);
  }
  for (std::size_t end = count - 1; end > 0; --end) {
    numeric::swap(intervals[0], intervals[end]);
    SiftIntervals(intervals, end, 0);
  }
  Bounds length;
  Interval current = intervals[0];
  for (std::size_t index = 1; index < count; ++index) {
    const auto next = intervals[index];
    if (next.low <= current.high) {
      current.high = numeric::max(current.high, next.high);
      continue;
    }
    length = Add(length, Subtract(Point(current.high), Point(current.low)));
    current = next;
  }
  length = Add(length, Subtract(Point(current.high), Point(current.low)));
  return upper ? length.upper : length.lower;
}

FLUIDITY_HD inline Bounds WorldArea(Bounds area_lattice, double h) {
  return MultiplyNonnegative(MultiplyNonnegative(area_lattice, Point(h)), Point(h));
}

// Each band is anchored to the original patch lattice. Band clipping and
// nearest/farthest circle radii give pointwise interval-union bounds.
FLUIDITY_HD inline PatchMeasure IntegratePatch(
    const Field& field,
    PatchIndex patch,
    const Disk* disks,
    std::size_t disk_count,
    Interval* inner,
    Interval* outer
) {
  const auto origin = field.geometry.world_origin;
  const double h = field.geometry.cell_width;
  const auto floor = LatticeOffset(field.tank.floor_y, origin.y, patch.y, h);
  const auto low_z = LatticeOffset(field.tank.low_z, origin.z, patch.z, h);
  const auto high_z = LatticeOffset(field.tank.high_z, origin.z, patch.z, h);
  const auto surface = field.is_plane ? LatticeOffset(field.plane_height_y, origin.y, patch.y, h) : Point(1);
  if (!Finite(floor) || !Finite(low_z) || !Finite(high_z) || !Finite(surface)) {
    return {.error = Error::kRange};
  }
  const Rectangle rectangle{
      .low_y = Maximum(Point(0), floor),
      .high_y = Minimum(Point(1), surface),
      .low_z = Maximum(Point(0), low_z),
      .high_z = Minimum(Point(1), high_z),
  };
  const auto y_width = Width(rectangle.low_y, rectangle.high_y);
  const auto z_width = Width(rectangle.low_z, rectangle.high_z);
  if (y_width.upper == 0 || z_width.upper == 0) {
    return {};
  }
  const auto rectangle_area = MultiplyNonnegative(y_width, z_width);
  if (field.is_plane) {
    const auto area = WorldArea(rectangle_area, h);
    return {
        .area = area,
        .all_wet_sources_cross_lower_edge = floor.upper <= 0 && surface.lower > 0 && z_width.lower > 0,
        .error = Finite(area) ? Error::kNone : Error::kRange,
    };
  }

  bool all_wet_sources_cross_lower_edge = floor.upper <= 0 && z_width.lower > 0;
  bool possibly_wet = false;
  for (std::size_t index = 0; index < disk_count; ++index) {
    const auto& disk = disks[index];
    if (disk.radius_squared.upper == 0) {
      continue;
    }
    const auto y = Subtract(disk.lattice_y, Point(patch.y));
    const auto z = Subtract(disk.lattice_z, Point(patch.z));
    const auto nearest_y = Nearest(y, rectangle.low_y, rectangle.high_y);
    const auto nearest_z = Nearest(z, rectangle.low_z, rectangle.high_z);
    const auto patch_gap = Gap(disk.radius_squared, nearest_y, nearest_z);
    if (!Finite(patch_gap)) {
      return {.error = Error::kRange};
    }
    if (patch_gap.upper <= 0) {
      continue;
    }
    possibly_wet = true;
    const auto edge_gap = Gap(disk.radius_squared, Magnitude(y), nearest_z);
    if (!Finite(edge_gap)) {
      return {.error = Error::kRange};
    }
    all_wet_sources_cross_lower_edge = all_wet_sources_cross_lower_edge && edge_gap.lower > 0;
  }
  if (!possibly_wet) {
    return {};
  }

  Bounds area_lattice;
  for (int band = 0; band < kBandsPerPatch; ++band) {
    const auto band_low = Maximum(rectangle.low_z, Point(static_cast<double>(band) / kBandsPerPatch));
    const auto band_high = Minimum(rectangle.high_z, Point(static_cast<double>(band + 1) / kBandsPerPatch));
    const auto band_width = Width(band_low, band_high);
    if (band_width.upper == 0) {
      continue;
    }
    std::size_t inner_count = 0;
    std::size_t outer_count = 0;
    for (std::size_t index = 0; index < disk_count; ++index) {
      const auto& disk = disks[index];
      if (disk.radius_squared.upper == 0) {
        continue;
      }
      const auto z = Subtract(disk.lattice_z, Point(patch.z));
      const auto near_squared = Square(Nearest(z, band_low, band_high));
      const auto far_squared = Square(Farthest(z, band_low, band_high));
      const auto outer_squared = Subtract(disk.radius_squared, near_squared);
      const auto inner_squared = Subtract(disk.radius_squared, far_squared);
      if (!Finite(outer_squared) || !Finite(inner_squared)) {
        return {.error = Error::kRange};
      }
      if (outer_squared.upper <= 0) {
        continue;
      }
      const auto y = Subtract(disk.lattice_y, Point(patch.y));
      const double outer_radius = DirectedRoot(outer_squared.upper, true);
      const double outer_low = Maximum(rectangle.low_y, Subtract(y, Point(outer_radius))).lower;
      const double outer_high = Minimum(rectangle.high_y, Add(y, Point(outer_radius))).upper;
      if (outer_low < outer_high) {
        outer[outer_count++] = {outer_low, outer_high};
      }
      if (inner_squared.lower > 0) {
        const double inner_radius = DirectedRoot(inner_squared.lower, false);
        const double inner_low = Maximum(rectangle.low_y, Subtract(y, Point(inner_radius))).upper;
        const double inner_high = Minimum(rectangle.high_y, Add(y, Point(inner_radius))).lower;
        if (inner_low < inner_high) {
          inner[inner_count++] = {inner_low, inner_high};
        }
      }
    }
    const Bounds length{
        UnionLength(inner, inner_count, false),
        UnionLength(outer, outer_count, true),
    };
    area_lattice = Add(area_lattice, MultiplyNonnegative(length, band_width));
  }
  const auto area = WorldArea(Minimum(area_lattice, rectangle_area), h);
  return {
      .area = area,
      .all_wet_sources_cross_lower_edge = all_wet_sources_cross_lower_edge,
      .error = Finite(area) ? Error::kNone : Error::kRange,
  };
}

// Integrate the clipped patch, certify its lower-edge contact and multiply the
// area midpoint by wall velocity. The integrator sorts both writable interval
// banks in place. Each bank must hold disk_count records and be disjoint from the
// other bank and the input disks.
FLUIDITY_HD inline PatchMeasure MeasurePatch(
    const Field& field,
    PatchIndex patch,
    const Disk* disks,
    std::size_t disk_count,
    Interval* inner,
    Interval* outer
) {
  auto measure = IntegratePatch(field, patch, disks, disk_count, inner, outer);
  if (measure.error == Error::kNone) {
    const double midpoint_area = numeric::midpoint(measure.area.lower, measure.area.upper);
    measure.midpoint_flux = midpoint_area * field.tank.piston_velocity_x;
    if (!numeric::isfinite(measure.midpoint_flux)) {
      measure.error = Error::kRange;
    }
  }
  return measure;
}
}  // namespace wetted_piston_internal
}  // namespace fluidity
