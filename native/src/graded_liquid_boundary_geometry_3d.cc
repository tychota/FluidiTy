#include "graded_liquid_boundary_geometry_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "fluidity/base/sampled_solid_liquid_geometry_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "graded_sampled_liquid_boundary_geometry_3d.h"
#include "liquid_boundary_backend.h"
#include "mac_pressure_3d_arithmetic.h"

namespace fluidity {
namespace {

using boundary_internal::CartesianComponents;
using boundary_internal::CartesianVector;
using boundary_internal::ClippingPolygon;
using boundary_internal::ClippingVertex;
using boundary_internal::HasFiniteComponents;
using boundary_internal::IsNormalOrZero;

constexpr double kMeasureRoundoff = 64 * std::numeric_limits<double>::epsilon();

struct WorldBounds {
  CartesianVector lower;
  CartesianVector upper;
  CartesianVector extent;
};

Vec3 ToVec3(CartesianVector values) {
  return {values[0], values[1], values[2]};
}

bool UnitNormal(CartesianVector normal) {
  return HasFiniteComponents(normal) && std::abs(std::hypot(normal[0], normal[1], normal[2]) - 1) <= 1e-12;
}

double SignedPlaneDistance(CartesianVector normal, CartesianVector point, double plane_coordinate_m) {
  return std::fma(
      normal[0],
      point[0],
      std::fma(normal[1], point[1], std::fma(normal[2], point[2], -plane_coordinate_m))
  );
}

// Evaluate shared endpoints from their canonical finest-lattice indices,
// including the high endpoint rather than adding a rounded world width.
absl::StatusOr<WorldBounds> MakeWorldBounds(
    const std::array<std::int32_t, 3>& lower_corner,
    std::uint8_t width_exponent,
    int face_normal_axis,
    double finest_width_m,
    CartesianVector world_origin_m
) {
  WorldBounds bounds{};
  const double lattice_width = std::ldexp(1.0, width_exponent);
  const double nominal_width_m = std::ldexp(finest_width_m, width_exponent);
  for (int axis = 0; axis < 3; ++axis) {
    const double low_index = lower_corner[axis];
    const double high_index = axis == face_normal_axis ? low_index : low_index + lattice_width;
    const double low_m = std::fma(low_index, finest_width_m, world_origin_m[axis]);
    const double high_m = std::fma(high_index, finest_width_m, world_origin_m[axis]);
    const double extent_m = high_m - low_m;
    if (!std::isfinite(low_m) || !std::isfinite(high_m) || !std::isfinite(extent_m)) {
      return absl::OutOfRangeError("Graded liquid world bounds are unrepresentable");
    }
    if (axis != face_normal_axis
        && (extent_m <= 0 || std::nextafter(low_m, high_m) - low_m > nominal_width_m
            || high_m - std::nextafter(high_m, low_m) > nominal_width_m)) {
      return absl::OutOfRangeError("Graded liquid world bounds do not resolve the lattice width");
    }
    bounds.lower[axis] = low_m;
    bounds.upper[axis] = high_m;
    bounds.extent[axis] = extent_m;
  }
  return bounds;
}

CartesianVector Corner(const WorldBounds& bounds, int corner_bits) {
  return {
      (corner_bits & 1) == 0 ? bounds.lower[0] : bounds.upper[0],
      (corner_bits & 2) == 0 ? bounds.lower[1] : bounds.upper[1],
      (corner_bits & 4) == 0 ? bounds.lower[2] : bounds.upper[2],
  };
}

// Both sampled affine fields use the same physical endpoints. Clipping is in
// unit cube coordinates, so the physical rectangular measure has one scale.
absl::StatusOr<double> RescaleMeasure(double measure, double nominal_full, double physical_full) {
  if (!std::isnormal(nominal_full) || !std::isnormal(physical_full) || !IsNormalOrZero(measure) || measure < 0
      || measure / nominal_full > 1 + kMeasureRoundoff) {
    return absl::OutOfRangeError("Graded liquid clipped measure is unrepresentable");
  }
  const double physical_measure = measure * (physical_full / nominal_full);
  if (!IsNormalOrZero(physical_measure) || (measure > 0 && physical_measure == 0)) {
    return absl::OutOfRangeError("Graded liquid physical measure is unrepresentable");
  }
  return std::min(physical_measure, physical_full);
}

struct PlanePair {
  CartesianVector solid_normal;
  double solid_coordinate_m;
  CartesianVector liquid_normal;
  double liquid_coordinate_m;
};

struct CornerDistances {
  double liquid;
  double solid;
};

template <typename LatticeIndex>
CellCornerIndex3D LatticeCorner(LatticeIndex lower, std::uint8_t exponent, int bits) {
  const auto width = std::int32_t{1} << exponent;
  return {
      lower.x + ((bits & 1) ? width : 0),
      lower.y + ((bits & 2) ? width : 0),
      lower.z + ((bits & 4) ? width : 0),
  };
}

CornerDistances SamplePlanes(const PlanePair& planes, CartesianVector point) {
  return {
      SignedPlaneDistance(planes.liquid_normal, point, planes.liquid_coordinate_m),
      SignedPlaneDistance(planes.solid_normal, point, planes.solid_coordinate_m),
  };
}

template <typename Sample>
absl::StatusOr<FrozenGradedLiquidLeaf3D>
MeasureLeaf(const WorldBounds& bounds, const AdaptiveMacLeaf3D& leaf, const Sample& sample) {
  bool known_air = true;
  boundary_internal::SampledSolidLiquidCell query{.cell_width = leaf.cell_width};
  for (int corner_bits = 0; corner_bits < 8; ++corner_bits) {
    const auto distance =
        sample(LatticeCorner(leaf.key.lower_corner, leaf.key.width_exponent, corner_bits), Corner(bounds, corner_bits));
    if (!distance.ok()) {
      return distance.status();
    }
    query.solid_distance[corner_bits] = distance->solid;
    query.liquid_distance[corner_bits] = distance->liquid;
    if (!std::isfinite(query.solid_distance[corner_bits]) || !std::isfinite(query.liquid_distance[corner_bits])) {
      return absl::OutOfRangeError("Graded liquid corner distance is unrepresentable");
    }
    known_air &= query.solid_distance[corner_bits] > 0 && query.liquid_distance[corner_bits] >= 0;
  }
  const auto measured = boundary_internal::MeasureSampledSolidLiquidCell(query);
  if (!measured.ok()) {
    return boundary_internal::ArithmeticStatus(measured.status());
  }
  const double physical_full_volume_m3 = bounds.extent[0] * bounds.extent[1] * bounds.extent[2];
  const auto volume = RescaleMeasure(measured->wet_volume, leaf.cell_volume, physical_full_volume_m3);
  if (!volume.ok()) {
    return volume.status();
  }
  return FrozenGradedLiquidLeaf3D{
      .world_lower_m = ToVec3(bounds.lower),
      .world_upper_m = ToVec3(bounds.upper),
      .fluid_volume_m3 = *volume,
      .known_air = known_air,
  };
}

template <typename Sample>
absl::StatusOr<FrozenGradedLiquidSubface3D> MeasureSubface(
    FaceIndex3D face_index,
    std::uint8_t width_exponent,
    bool exterior,
    double finest_width_m,
    CartesianVector world_origin_m,
    const Sample& sample,
    CartesianVector wall_velocity
) {
  const int normal_axis = static_cast<int>(face_index.axis);
  const auto lattice_index = face_index.lattice_index;
  const auto bounds = MakeWorldBounds(
      {lattice_index.x, lattice_index.y, lattice_index.z},
      width_exponent,
      normal_axis,
      finest_width_m,
      world_origin_m
  );
  if (!bounds.ok()) {
    return bounds.status();
  }
  const int first_axis = normal_axis == 0 ? 1 : 0;
  const int second_axis = normal_axis == 2 ? 1 : 2;
  const int first_bit = 1 << first_axis;
  const int second_bit = 1 << second_axis;
  const int upper_corner = first_bit | second_bit;
  const double nominal_width_m = std::ldexp(finest_width_m, width_exponent);
  const double nominal_full_area_m2 = nominal_width_m * nominal_width_m;
  const double physical_full_area_m2 = bounds->extent[first_axis] * bounds->extent[second_axis];

  std::array<ClippingVertex, 8> vertices{};
  std::array<double, 8> solid_distances{};
  const std::array corner_bits{0, first_bit, second_bit, upper_corner};
  double largest_solid_distance = 0;
  for (const int corner : corner_bits) {
    const auto distance = sample(LatticeCorner(lattice_index, width_exponent, corner), Corner(*bounds, corner));
    if (!distance.ok()) {
      return distance.status();
    }
    auto& vertex = vertices[corner];
    vertex.cell_width_position[first_axis] = (corner & first_bit) != 0 ? 1 : 0;
    vertex.cell_width_position[second_axis] = (corner & second_bit) != 0 ? 1 : 0;
    vertex.liquid_distance = distance->liquid;
    const double solid_distance = distance->solid;
    if (!std::isfinite(vertex.liquid_distance) || !std::isfinite(solid_distance)) {
      return absl::OutOfRangeError("Graded liquid subface corner distance is unrepresentable");
    }
    solid_distances[corner] = solid_distance;
    largest_solid_distance = std::max(largest_solid_distance, std::abs(solid_distance));
  }
  int solid_distance_exponent = 0;
  (void)std::frexp(largest_solid_distance, &solid_distance_exponent);
  for (const int corner : corner_bits) {
    const double original_distance = solid_distances[corner];
    solid_distances[corner] = std::ldexp(original_distance, -solid_distance_exponent);
    if (original_distance != 0 && !std::isnormal(solid_distances[corner])) {
      return absl::OutOfRangeError("Graded liquid subface solid distance loses precision under normalization");
    }
  }

  double wet_open_area_m2 = 0;
  double wet_solid_area_m2 = 0;
  struct CentroidMoment {
    double area_m2 = 0;
    CartesianVector fraction{};
  };
  CentroidMoment open_centroid;
  CentroidMoment solid_centroid;
  // Accumulate in normalized face coordinates to avoid world-offset cancellation.
  const auto accumulate_centroid = [&](const ClippingPolygon& polygon, CentroidMoment& centroid) {
    const auto anchor = polygon[0].cell_width_position;
    for (std::size_t corner = 1; corner + 1 < polygon.size(); ++corner) {
      const auto second = polygon[corner].cell_width_position;
      const auto third = polygon[corner + 1].cell_width_position;
      const ClippingPolygon triangle{polygon[0], polygon[corner], polygon[corner + 1]};
      const double area_m2 = boundary_internal::ComputeClippedArea(triangle, nominal_width_m);
      if (area_m2 == 0) {
        continue;
      }
      centroid.area_m2 += area_m2;
      const double relative_weight = area_m2 / centroid.area_m2;
      for (const int axis : {first_axis, second_axis}) {
        const double triangle_centroid = (anchor[axis] + second[axis] + third[axis]) / 3.0;
        centroid.fraction[axis] = std::lerp(centroid.fraction[axis], triangle_centroid, relative_weight);
      }
    }
  };
  for (const int middle_corner : {first_bit, second_bit}) {
    const int triangle_corners[]{0, middle_corner, upper_corner};
    const ClippingPolygon triangle{vertices[0], vertices[middle_corner], vertices[upper_corner]};
    if (std::ranges::none_of(triangle, [](const ClippingVertex& vertex) {
          return vertex.liquid_distance < 0;
        })) {
      continue;
    }
    const auto wet = boundary_internal::ClipPolygon(triangle, {{}, 0, true});
    const auto fluid_side = boundary_internal::SampledSolidExteriorPlane(solid_distances, triangle_corners);
    const boundary_internal::ClippingPlane solid_side{
        .normal = boundary_internal::Scale(fluid_side.normal, -1),
        .plane_coordinate = -fluid_side.plane_coordinate,
    };
    const double open_area_m2 = boundary_internal::ComputeOpenFaceArea(wet, fluid_side, nominal_width_m);
    wet_open_area_m2 += open_area_m2;
    if (open_area_m2 > 0) {
      accumulate_centroid(boundary_internal::ClipPolygon(wet, fluid_side), open_centroid);
    }
    const auto blocked = boundary_internal::ClipPolygon(wet, solid_side);
    const double solid_area_m2 = boundary_internal::ComputeClippedArea(blocked, nominal_width_m);
    wet_solid_area_m2 += solid_area_m2;
    if (solid_area_m2 > 0) {
      accumulate_centroid(blocked, solid_centroid);
    }
  }
  if ((wet_open_area_m2 + wet_solid_area_m2) / nominal_full_area_m2 > 1 + kMeasureRoundoff) {
    return absl::OutOfRangeError("Graded liquid wet areas exceed the subface area");
  }
  const auto physical_open = RescaleMeasure(wet_open_area_m2, nominal_full_area_m2, physical_full_area_m2);
  if (!physical_open.ok()) {
    return physical_open.status();
  }
  const auto physical_solid = RescaleMeasure(wet_solid_area_m2, nominal_full_area_m2, physical_full_area_m2);
  if (!physical_solid.ok()) {
    return physical_solid.status();
  }
  const auto world_centroid = [&](double area_m2,
                                  const CentroidMoment& centroid) -> absl::StatusOr<std::optional<Vec3>> {
    if (area_m2 == 0) {
      return std::nullopt;
    }
    if (!std::isnormal(centroid.area_m2)) {
      return absl::OutOfRangeError("Graded liquid wet centroid weight is unrepresentable");
    }
    auto position = bounds->lower;
    for (const int axis : {first_axis, second_axis}) {
      const double fraction = centroid.fraction[axis];
      if (!std::isfinite(fraction) || fraction < -kMeasureRoundoff || fraction > 1 + kMeasureRoundoff) {
        return absl::OutOfRangeError("Graded liquid wet centroid lies outside its subface");
      }
      position[axis] = std::lerp(bounds->lower[axis], bounds->upper[axis], std::clamp(fraction, 0.0, 1.0));
    }
    return std::optional<Vec3>{ToVec3(position)};
  };
  const auto wet_open_centroid_m = world_centroid(*physical_open, open_centroid);
  if (!wet_open_centroid_m.ok()) {
    return wet_open_centroid_m.status();
  }
  const auto wet_solid_centroid_m = world_centroid(*physical_solid, solid_centroid);
  if (!wet_solid_centroid_m.ok()) {
    return wet_solid_centroid_m.status();
  }
  const double wall_normal_flux_m3_per_s = *physical_solid * wall_velocity[normal_axis];
  if (!pressure_internal::arithmetic::IsRepresentablePressureProduct(
          *physical_solid,
          wall_velocity[normal_axis],
          wall_normal_flux_m3_per_s
      )) {
    return absl::OutOfRangeError("Graded liquid wall flux is unrepresentable");
  }
  return FrozenGradedLiquidSubface3D{
      .face_index = face_index,
      .width_exponent = width_exponent,
      .exterior = exterior,
      .world_lower_m = ToVec3(bounds->lower),
      .world_upper_m = ToVec3(bounds->upper),
      .full_area_m2 = physical_full_area_m2,
      .wet_open_area_m2 = *physical_open,
      .wet_open_centroid_m = *wet_open_centroid_m,
      .wet_solid_area_m2 = *physical_solid,
      .wet_solid_centroid_m = *wet_solid_centroid_m,
      .wall_normal_flux_m3_per_s = wall_normal_flux_m3_per_s,
  };
}

bool MatchesPressureWidth(Vec3 world_lower_m, Vec3 world_upper_m, double width_m, int face_normal_axis) {
  const auto lower = CartesianComponents(world_lower_m);
  const auto upper = CartesianComponents(world_upper_m);
  for (int axis = 0; axis < 3; ++axis) {
    if (axis != face_normal_axis && std::abs((upper[axis] - lower[axis]) / width_m - 1) > kMeasureRoundoff) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<FrozenGradedPlanarLiquidGeometry3D> FreezeGradedPlanarLiquidGeometry3D(
    const AdaptiveMacTopology3D& topology,
    Vec3 world_origin_m,
    const TranslatingHalfspace3D& solid,
    const GradedLiquidPlanarSurface3D& liquid_surface,
    double step_duration_s
) {
  const auto origin = CartesianComponents(world_origin_m);
  const auto wall_velocity = CartesianComponents(solid.wall_velocity);
  const PlanePair planes{
      .solid_normal = CartesianComponents(solid.unit_normal),
      .solid_coordinate_m = solid.start_plane_coordinate,
      .liquid_normal = CartesianComponents(liquid_surface.unit_normal),
      .liquid_coordinate_m = liquid_surface.plane_coordinate_m,
  };
  if (!HasFiniteComponents(origin) || !HasFiniteComponents(wall_velocity) || !UnitNormal(planes.solid_normal)
      || !UnitNormal(planes.liquid_normal) || !std::isfinite(planes.solid_coordinate_m)
      || !std::isfinite(planes.liquid_coordinate_m) || !std::isfinite(step_duration_s) || step_duration_s <= 0) {
    return absl::InvalidArgumentError("Invalid graded liquid world origin, planar geometry or duration");
  }
  const double wall_plane_speed_m_per_s = SignedPlaneDistance(planes.solid_normal, wall_velocity, 0);
  if (!std::isfinite(std::fma(step_duration_s, wall_plane_speed_m_per_s, planes.solid_coordinate_m))) {
    return absl::OutOfRangeError("Graded liquid wall end plane is unrepresentable");
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto interior = topology.interior_faces();
  const std::size_t face_count = interior.size() + ownership->exterior_faces.size();
  if (face_count > pressure_internal::arithmetic::kMaxPressureFaces) {
    return absl::ResourceExhaustedError("Graded liquid subface count exceeds the state limit");
  }
  const auto sample = [&](CellCornerIndex3D, CartesianVector world) -> absl::StatusOr<CornerDistances> {
    return SamplePlanes(planes, world);
  };
  const double finest_width_m = topology.finest_cell_width();
  FrozenGradedPlanarLiquidGeometry3D result(topology, world_origin_m, step_duration_s);
  result.leaves_.reserve(topology.cells().size());
  result.subfaces_.resize(face_count);
  for (const auto& leaf : topology.cells()) {
    const auto corner = leaf.key.lower_corner;
    const auto bounds =
        MakeWorldBounds({corner.x, corner.y, corner.z}, leaf.key.width_exponent, -1, finest_width_m, origin);
    if (!bounds.ok()) {
      return bounds.status();
    }
    const auto measured = MeasureLeaf(*bounds, leaf, sample);
    if (!measured.ok()) {
      return measured.status();
    }
    result.leaves_.push_back(*measured);
  }
  for (std::size_t ordinal = 0; ordinal < interior.size(); ++ordinal) {
    const auto& face = interior[ordinal];
    const auto measured =
        MeasureSubface(face.face_index, face.width_exponent, false, finest_width_m, origin, sample, wall_velocity);
    if (!measured.ok()) {
      return measured.status();
    }
    result.subfaces_[ownership->interior_face_slots[ordinal]] = *measured;
  }
  for (const auto& face : ownership->exterior_faces) {
    const auto measured =
        MeasureSubface(face.face_index, face.width_exponent, true, finest_width_m, origin, sample, wall_velocity);
    if (!measured.ok()) {
      return measured.status();
    }
    result.subfaces_[face.face_slot] = *measured;
  }
  return result;
}

namespace {

auto SampleOrder(CellCornerIndex3D corner) {
  return std::tuple{corner.z, corner.y, corner.x};
}

// Require one strict classification across both traces of every subdivided
// side. This is sufficient for the piecewise-affine aperture to agree even
// when the two tessellations interpolate different distance magnitudes.
template <typename Sample>
absl::Status CheckSubdividedSide(
    const AdaptiveMacLeaf3D& leaf,
    FaceIndex3D face,
    std::uint8_t exponent,
    bool tail,
    const Sample& sample
) {
  if (leaf.key.width_exponent == exponent) {
    return absl::OkStatus();
  }
  const int normal_bit = 1 << static_cast<int>(face.axis);
  const auto reference =
      sample(LatticeCorner(leaf.key.lower_corner, leaf.key.width_exponent, tail ? normal_bit : 0), {});
  if (!reference.ok()) {
    return reference.status();
  }
  for (int bits = 0; bits < 8; ++bits) {
    if ((bits & normal_bit) != 0) {
      continue;
    }
    const auto coarse =
        sample(LatticeCorner(leaf.key.lower_corner, leaf.key.width_exponent, bits | (tail ? normal_bit : 0)), {});
    const auto fine = sample(LatticeCorner(face.lattice_index, exponent, bits), {});
    if (!coarse.ok()) {
      return coarse.status();
    }
    if (!fine.ok()) {
      return fine.status();
    }
    const auto same_sign = [](double left, double right) {
      return (left > 0 && right > 0) || (left < 0 && right < 0);
    };
    for (const auto value : {*coarse, *fine}) {
      if (!same_sign(reference->liquid, value.liquid) || !same_sign(reference->solid, value.solid)) {
        return absl::FailedPreconditionError("Sampled graded geometry has a cut or uncertain subdivided coarse side");
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<FrozenGradedPlanarLiquidGeometry3D> FreezeGradedSampledLiquidGeometry3D(
    const AdaptiveMacTopology3D& topology,
    Vec3 world_origin_m,
    double step_duration_s,
    std::span<const GradedLiquidCornerSample3D> samples,
    std::span<const std::uint8_t> certified_air_leaves
) {
  const auto origin = CartesianComponents(world_origin_m);
  const auto leaves = topology.cells();
  if (!HasFiniteComponents(origin) || !std::isfinite(step_duration_s) || step_duration_s <= 0
      || certified_air_leaves.size() != leaves.size() || samples.empty()) {
    return absl::InvalidArgumentError("Invalid sampled graded geometry extents, origin or duration");
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto interior = topology.interior_faces();
  const std::size_t face_count = interior.size() + ownership->exterior_faces.size();
  if (face_count > pressure_internal::arithmetic::kMaxPressureFaces) {
    return absl::ResourceExhaustedError("Sampled graded subface count exceeds the state limit");
  }
  if (samples.size() > 8 * leaves.size() + 4 * face_count) {
    return absl::InvalidArgumentError("Sampled graded geometry has excessive corner samples");
  }
  for (std::size_t ordinal = 0; ordinal < samples.size(); ++ordinal) {
    const auto& entry = samples[ordinal];
    if (!std::isfinite(entry.liquid_distance_m) || !std::isfinite(entry.solid_distance_m)
        || (ordinal > 0 && SampleOrder(samples[ordinal - 1].corner) >= SampleOrder(entry.corner))) {
      return absl::InvalidArgumentError("Sampled graded corners must be finite, unique and canonically ordered");
    }
  }

  std::vector<std::uint8_t> used(samples.size());
  const auto sample = [&](CellCornerIndex3D corner, CartesianVector) -> absl::StatusOr<CornerDistances> {
    const auto found = std::ranges::lower_bound(samples, SampleOrder(corner), {}, [](const auto& entry) {
      return SampleOrder(entry.corner);
    });
    if (found == samples.end() || SampleOrder(found->corner) != SampleOrder(corner)) {
      return absl::FailedPreconditionError("Sampled graded geometry is missing a canonical corner");
    }
    used[static_cast<std::size_t>(found - samples.begin())] = 1;
    return CornerDistances{found->liquid_distance_m, found->solid_distance_m};
  };

  const double finest_width_m = topology.finest_cell_width();
  FrozenGradedPlanarLiquidGeometry3D result(topology, world_origin_m, step_duration_s);
  result.leaves_.reserve(leaves.size());
  result.subfaces_.resize(face_count);
  for (std::size_t ordinal = 0; ordinal < leaves.size(); ++ordinal) {
    const auto& leaf = leaves[ordinal];
    const auto corner = leaf.key.lower_corner;
    const auto bounds =
        MakeWorldBounds({corner.x, corner.y, corner.z}, leaf.key.width_exponent, -1, finest_width_m, origin);
    if (!bounds.ok()) {
      return bounds.status();
    }
    auto measured = MeasureLeaf(*bounds, leaf, sample);
    if (!measured.ok()) {
      return measured.status();
    }
    const auto certificate = certified_air_leaves[ordinal];
    if (certificate > 1) {
      return absl::InvalidArgumentError("Sampled graded air certificate must be zero or one");
    }
    const bool air = certificate == 1;
    if ((air && !measured->known_air) || (air != (measured->fluid_volume_m3 == 0))) {
      return absl::FailedPreconditionError("Sampled graded geometry has uncertified or conflicting dry occupancy");
    }
    measured->known_air = air;
    result.leaves_.push_back(*measured);
  }

  for (std::size_t ordinal = 0; ordinal < interior.size(); ++ordinal) {
    const auto& face = interior[ordinal];
    const auto tail =
        CheckSubdividedSide(leaves[face.tail_cell_ordinal], face.face_index, face.width_exponent, true, sample);
    if (!tail.ok()) {
      return tail;
    }
    const auto head =
        CheckSubdividedSide(leaves[face.head_cell_ordinal], face.face_index, face.width_exponent, false, sample);
    if (!head.ok()) {
      return head;
    }
    const auto measured =
        MeasureSubface(face.face_index, face.width_exponent, false, finest_width_m, origin, sample, {});
    if (!measured.ok()) {
      return measured.status();
    }
    result.subfaces_[ownership->interior_face_slots[ordinal]] = *measured;
  }
  for (const auto& face : ownership->exterior_faces) {
    const auto trace =
        CheckSubdividedSide(leaves[face.cell_ordinal], face.face_index, face.width_exponent, face.cell_is_tail, sample);
    if (!trace.ok()) {
      return trace;
    }
    const auto measured =
        MeasureSubface(face.face_index, face.width_exponent, true, finest_width_m, origin, sample, {});
    if (!measured.ok()) {
      return measured.status();
    }
    result.subfaces_[face.face_slot] = *measured;
  }
  if (std::ranges::find(used, 0) != used.end()) {
    return absl::InvalidArgumentError("Sampled graded geometry has unrelated corner samples");
  }
  return result;
}

absl::StatusOr<GradedLiquidPressureInputs3D> ComposeGradedPlanarLiquidPressureInputs3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const GradedLiquidPressureCellFields3D> cell_fields,
    std::span<const GradedLiquidPressureSubfaceFields3D> subface_fields,
    GradedLiquidPressureDomain3D domain
) {
  const auto leaves = geometry.leaves();
  const auto subfaces = geometry.subfaces();
  if (leaves.empty() || cell_fields.size() != leaves.size() || subface_fields.size() != subfaces.size()) {
    return absl::InvalidArgumentError("Graded liquid pressure fields must cover the frozen geometry");
  }
  const auto& topology = geometry.topology();
  std::vector<MacPressureCell3D> cells;
  cells.reserve(leaves.size());
  for (std::size_t ordinal = 0; ordinal < leaves.size(); ++ordinal) {
    const auto& measured = leaves[ordinal];
    const auto& leaf = topology.cells()[ordinal];
    const auto& fields = cell_fields[ordinal];
    if (measured.fluid_volume_m3 == 0 && (domain != GradedLiquidPressureDomain3D::kKnownAir || !measured.known_air)) {
      return absl::FailedPreconditionError("Graded liquid pressure requires positive volume or certified air");
    }
    if (!MatchesPressureWidth(measured.world_lower_m, measured.world_upper_m, leaf.cell_width, -1)) {
      return absl::OutOfRangeError("Graded liquid world leaf extent differs from the pressure metric");
    }
    cells.push_back({
        .cell_index = leaf.key.lower_corner,
        .fluid_volume = std::min(measured.fluid_volume_m3, leaf.cell_volume),
        .target_divergence = fields.target_divergence_s_inv,
        .bulk_modulus = fields.bulk_modulus_pa,
        .previous_pressure = fields.previous_pressure_pa,
    });
  }
  std::vector<pressure_internal::AdaptiveMacPressureSubface3D> faces;
  faces.reserve(subfaces.size());
  for (std::size_t slot = 0; slot < subfaces.size(); ++slot) {
    const auto& measured = subfaces[slot];
    const auto& fields = subface_fields[slot];
    const double nominal_width_m = std::ldexp(topology.finest_cell_width(), measured.width_exponent);
    if (!MatchesPressureWidth(
            measured.world_lower_m,
            measured.world_upper_m,
            nominal_width_m,
            static_cast<int>(measured.face_index.axis)
        )) {
      return absl::OutOfRangeError("Graded liquid world subface extent differs from the pressure metric");
    }
    const bool sealed_exterior = measured.exterior && !fields.exterior_pressure;
    const double nominal_full_area_m2 = nominal_width_m * nominal_width_m;
    faces.push_back({
        .open_area_m2 = sealed_exterior ? 0 : std::min(measured.wet_open_area_m2, nominal_full_area_m2),
        .inverse_density_m3_per_kg = fields.inverse_density_m3_per_kg,
        .predicted_normal_velocity_m_per_s = fields.predicted_normal_velocity_m_per_s,
        .prescribed_wall_flux_m3_per_s = measured.wall_normal_flux_m3_per_s,
        .exterior_pressure_distance_m = fields.exterior_pressure ? fields.exterior_pressure->distance_m : 0,
        .exterior_pressure_pa =
            fields.exterior_pressure ? std::optional{fields.exterior_pressure->pressure_pa} : std::nullopt,
        .pressure_jump_pa = fields.pressure_jump_pa,
    });
  }
  return GradedLiquidPressureInputs3D{.cells = std::move(cells), .subfaces = std::move(faces)};
}

absl::StatusOr<pressure_internal::PreparedMacPressureSystem3D> PrepareGradedPlanarLiquidPressure3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const GradedLiquidPressureCellFields3D> cell_fields,
    std::span<const GradedLiquidPressureSubfaceFields3D> subface_fields
) {
  const auto inputs = ComposeGradedPlanarLiquidPressureInputs3D(geometry, cell_fields, subface_fields);
  if (!inputs.ok()) {
    return inputs.status();
  }
  return pressure_internal::PrepareAdaptiveMacPressureSystem3D(
      geometry.topology(),
      geometry.step_duration_s(),
      inputs->cells,
      inputs->subfaces
  );
}

namespace {

bool EmptyAirInventory(const LiquidRemapInventory3D& inventory) {
  return inventory.mass_kg == 0 && inventory.reference_volume_m3 == 0 && inventory.current_volume_m3 == 0
      && inventory.momentum_kg_m_per_s.x == 0 && inventory.momentum_kg_m_per_s.y == 0
      && inventory.momentum_kg_m_per_s.z == 0 && inventory.enthalpy_j == 0 && inventory.phase_id == 0
      && inventory.constituent_count == 0;
}

bool EmptyDryFaceFields(const pressure_internal::AdaptiveMacPressureSubface3D& face) {
  return face.open_area_m2 == 0 && face.inverse_density_m3_per_kg == 0 && face.predicted_normal_velocity_m_per_s == 0
      && face.prescribed_wall_flux_m3_per_s == 0 && face.pressure_jump_pa == 0 && !face.exterior_pressure_pa
      && face.exterior_pressure_distance_m == 0;
}

}  // namespace

absl::StatusOr<GradedWetPressureSubset3D> PrepareGradedWetPressureSubset3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const LiquidAdaptiveCellState3D> accepted_leaves,
    std::span<const GradedLiquidPressureCellFields3D> cell_fields,
    std::span<const GradedLiquidPressureSubfaceFields3D> subface_fields
) {
  const auto inputs = ComposeGradedPlanarLiquidPressureInputs3D(
      geometry,
      cell_fields,
      subface_fields,
      GradedLiquidPressureDomain3D::kKnownAir
  );
  if (!inputs.ok()) {
    return inputs.status();
  }
  return PrepareGradedWetPressureSubsetFromInputs3D(geometry, accepted_leaves, inputs->cells, inputs->subfaces);
}

absl::StatusOr<GradedWetPressureSubset3D> PrepareGradedWetPressureSubsetFromInputs3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const LiquidAdaptiveCellState3D> accepted_leaves,
    std::span<const MacPressureCell3D> cell_inputs,
    std::span<const pressure_internal::AdaptiveMacPressureSubface3D> subface_inputs
) {
  const auto leaves = geometry.leaves();
  if (accepted_leaves.size() != leaves.size() || cell_inputs.size() != leaves.size()
      || subface_inputs.size() != geometry.subfaces().size()) {
    return absl::InvalidArgumentError("Sparse pressure inputs must cover the frozen graded geometry");
  }
  const auto& topology = geometry.topology();
  std::vector<std::size_t> leaf_ordinals;
  std::vector<std::optional<std::size_t>> pressure_ordinals(leaves.size());
  std::vector<MacPressureCell3D> pressure_cells;
  for (std::size_t ordinal = 0; ordinal < leaves.size(); ++ordinal) {
    const auto& accepted = accepted_leaves[ordinal];
    const auto& cell = cell_inputs[ordinal];
    const auto& expected_key = topology.cells()[ordinal].key;
    const auto& accepted_corner = accepted.key.lower_corner;
    const auto& expected_corner = expected_key.lower_corner;
    if (accepted.key.width_exponent != expected_key.width_exponent || accepted_corner.x != expected_corner.x
        || accepted_corner.y != expected_corner.y || accepted_corner.z != expected_corner.z) {
      return absl::InvalidArgumentError("Accepted leaf key differs from the frozen graded geometry");
    }
    if (cell.cell_index.x != expected_corner.x || cell.cell_index.y != expected_corner.y
        || cell.cell_index.z != expected_corner.z
        || cell.fluid_volume != std::min(leaves[ordinal].fluid_volume_m3, topology.cells()[ordinal].cell_volume)) {
      return absl::InvalidArgumentError("Sparse pressure cell differs from the composed frozen geometry");
    }
    if (cell.fluid_volume > 0) {
      if (accepted.occupancy != LiquidCellOccupancy3D::kLiquid) {
        return absl::FailedPreconditionError("Graded pressure subset requires accepted liquid on every wet leaf");
      }
      pressure_ordinals[ordinal] = pressure_cells.size();
      pressure_cells.push_back(cell);
      leaf_ordinals.push_back(ordinal);
      continue;
    }
    if (!leaves[ordinal].known_air || accepted.occupancy != LiquidCellOccupancy3D::kAir
        || !EmptyAirInventory(accepted.inventory)) {
      return absl::FailedPreconditionError("Graded pressure subset requires accepted empty air on every dry leaf");
    }
    if (cell.target_divergence != 0 || cell.previous_pressure != 0
        || cell.bulk_modulus != std::numeric_limits<double>::infinity()) {
      return absl::FailedPreconditionError("Graded pressure subset cannot discard pressure data on an air leaf");
    }
  }
  if (pressure_cells.empty()) {
    return absl::FailedPreconditionError("Graded pressure subset requires a wet leaf");
  }

  const auto ownership = BuildAdaptiveMacFaceOwnership3D(topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  std::vector<pressure_internal::ExplicitMacPressureFace3D> interior;
  std::vector<pressure_internal::ExplicitMacPressureBoundaryFace3D> boundary;
  const auto interior_faces = topology.interior_faces();
  for (std::size_t ordinal = 0; ordinal < interior_faces.size(); ++ordinal) {
    const auto& face = interior_faces[ordinal];
    const std::size_t slot = ownership->interior_face_slots[ordinal];
    const auto& fields = subface_inputs[slot];
    const auto tail = pressure_ordinals[face.tail_cell_ordinal];
    const auto head = pressure_ordinals[face.head_cell_ordinal];
    if (fields.exterior_pressure_pa || fields.exterior_pressure_distance_m != 0) {
      return absl::InvalidArgumentError("Interior graded subface cannot carry exterior pressure data");
    }
    if (tail && head) {
      interior.push_back({
          .face_slot = slot,
          .tail_cell_ordinal = *tail,
          .head_cell_ordinal = *head,
          .open_area = fields.open_area_m2,
          .inverse_density = fields.inverse_density_m3_per_kg,
          .pressure_gradient_distance = face.cell_center_distance,
          .predicted_normal_velocity = fields.predicted_normal_velocity_m_per_s,
          .full_face_area = face.face_area,
          .prescribed_wall_flux = fields.prescribed_wall_flux_m3_per_s,
          .pressure_jump_pa = fields.pressure_jump_pa,
      });
      continue;
    }
    if (!EmptyDryFaceFields(fields)) {
      return absl::FailedPreconditionError("Graded pressure subset cannot discard flux or fields at an air subface");
    }
    if (tail || head) {
      boundary.push_back({
          .face_slot = slot,
          .cell_ordinal = tail ? *tail : *head,
          .cell_is_tail = tail.has_value(),
          .full_face_area = face.face_area,
          .open_area = 0,
          .inverse_density = 0,
          .pressure_gradient_distance = 0,
          .predicted_normal_velocity = 0,
          .prescribed_wall_flux = 0,
          .exterior_pressure = std::nullopt,
      });
    }
  }
  for (const auto& face : ownership->exterior_faces) {
    const auto& fields = subface_inputs[face.face_slot];
    const auto row = pressure_ordinals[face.cell_ordinal];
    if (!row) {
      if (geometry.subfaces()[face.face_slot].wet_open_area_m2 != 0 || !EmptyDryFaceFields(fields)) {
        return absl::FailedPreconditionError(
            "Graded pressure subset cannot discard aperture or fields on an air exterior"
        );
      }
      continue;
    }
    if (fields.pressure_jump_pa != 0 || (!fields.exterior_pressure_pa && fields.exterior_pressure_distance_m != 0)) {
      return absl::InvalidArgumentError("Exterior graded subface pressure data conflicts with its boundary type");
    }
    boundary.push_back({
        .face_slot = face.face_slot,
        .cell_ordinal = *row,
        .cell_is_tail = face.cell_is_tail,
        .full_face_area = face.face_area,
        .open_area = fields.open_area_m2,
        .inverse_density = fields.inverse_density_m3_per_kg,
        .pressure_gradient_distance = fields.exterior_pressure_distance_m,
        .predicted_normal_velocity = fields.predicted_normal_velocity_m_per_s,
        .prescribed_wall_flux = fields.prescribed_wall_flux_m3_per_s,
        .exterior_pressure = fields.exterior_pressure_pa,
    });
  }
  std::ranges::sort(boundary, {}, &pressure_internal::ExplicitMacPressureBoundaryFace3D::face_slot);
  auto system = pressure_internal::PrepareExplicitMacPressureSystem3D(
      geometry.step_duration_s(),
      pressure_cells,
      interior,
      boundary
  );
  if (!system.ok()) {
    return system.status();
  }
  return GradedWetPressureSubset3D{.system = std::move(*system), .leaf_ordinals = std::move(leaf_ordinals)};
}

}  // namespace fluidity
