#include "graded_liquid_boundary_geometry_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "fluidity/base/sampled_solid_liquid_geometry_3d.h"
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

absl::StatusOr<double>
MeasureLeafVolume(const WorldBounds& bounds, const AdaptiveMacLeaf3D& leaf, const PlanePair& planes) {
  boundary_internal::SampledSolidLiquidCell query{.cell_width = leaf.cell_width};
  for (int corner_bits = 0; corner_bits < 8; ++corner_bits) {
    const auto world_corner = Corner(bounds, corner_bits);
    query.solid_distance[corner_bits] =
        SignedPlaneDistance(planes.solid_normal, world_corner, planes.solid_coordinate_m);
    query.liquid_distance[corner_bits] =
        SignedPlaneDistance(planes.liquid_normal, world_corner, planes.liquid_coordinate_m);
    if (!std::isfinite(query.solid_distance[corner_bits]) || !std::isfinite(query.liquid_distance[corner_bits])) {
      return absl::OutOfRangeError("Graded liquid corner distance is unrepresentable");
    }
  }
  const auto measured = boundary_internal::MeasureSampledSolidLiquidCell(query);
  if (!measured.ok()) {
    return boundary_internal::ArithmeticStatus(measured.status());
  }
  const double physical_full_volume_m3 = bounds.extent[0] * bounds.extent[1] * bounds.extent[2];
  return RescaleMeasure(measured->wet_volume, leaf.cell_volume, physical_full_volume_m3);
}

absl::StatusOr<FrozenGradedLiquidSubface3D> MeasureSubface(
    FaceIndex3D face_index,
    std::uint8_t width_exponent,
    bool exterior,
    double finest_width_m,
    CartesianVector world_origin_m,
    const PlanePair& planes,
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
    const auto world_corner = Corner(*bounds, corner);
    auto& vertex = vertices[corner];
    vertex.cell_width_position[first_axis] = (corner & first_bit) != 0 ? 1 : 0;
    vertex.cell_width_position[second_axis] = (corner & second_bit) != 0 ? 1 : 0;
    vertex.liquid_distance = SignedPlaneDistance(planes.liquid_normal, world_corner, planes.liquid_coordinate_m);
    const double solid_distance = SignedPlaneDistance(planes.solid_normal, world_corner, planes.solid_coordinate_m);
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
  double centroid_weight = 0;
  CartesianVector centroid_fraction{};
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
      // Take moments in normalized face coordinates to avoid world-offset
      // cancellation and avoid changing the pressure area arithmetic.
      const auto open = boundary_internal::ClipPolygon(wet, fluid_side);
      const auto anchor = open[0].cell_width_position;
      for (std::size_t corner = 1; corner + 1 < open.size(); ++corner) {
        const auto second = open[corner].cell_width_position;
        const auto third = open[corner + 1].cell_width_position;
        const ClippingPolygon fan_triangle{open[0], open[corner], open[corner + 1]};
        const double weight = boundary_internal::ComputeClippedArea(fan_triangle, nominal_width_m);
        if (weight == 0) {
          continue;
        }
        centroid_weight += weight;
        const double relative_weight = weight / centroid_weight;
        for (const int axis : {first_axis, second_axis}) {
          const double triangle_centroid = (anchor[axis] + second[axis] + third[axis]) / 3.0;
          centroid_fraction[axis] = std::lerp(centroid_fraction[axis], triangle_centroid, relative_weight);
        }
      }
    }
    wet_solid_area_m2 +=
        boundary_internal::ComputeClippedArea(boundary_internal::ClipPolygon(wet, solid_side), nominal_width_m);
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
  std::optional<Vec3> wet_open_centroid_m;
  if (*physical_open > 0) {
    if (!std::isnormal(centroid_weight)) {
      return absl::OutOfRangeError("Graded liquid wet centroid weight is unrepresentable");
    }
    auto world_centroid = bounds->lower;
    for (const int axis : {first_axis, second_axis}) {
      const double fraction = centroid_fraction[axis];
      if (!std::isfinite(fraction) || fraction < -kMeasureRoundoff || fraction > 1 + kMeasureRoundoff) {
        return absl::OutOfRangeError("Graded liquid wet centroid lies outside its subface");
      }
      world_centroid[axis] = std::lerp(bounds->lower[axis], bounds->upper[axis], std::clamp(fraction, 0.0, 1.0));
    }
    wet_open_centroid_m = ToVec3(world_centroid);
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
      .wet_open_centroid_m = wet_open_centroid_m,
      .wet_solid_area_m2 = *physical_solid,
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
  const double finest_width_m = topology.finest_cell_width();
  FrozenGradedPlanarLiquidGeometry3D result(topology, step_duration_s);
  result.leaves_.reserve(topology.cells().size());
  result.subfaces_.resize(face_count);
  for (const auto& leaf : topology.cells()) {
    const auto corner = leaf.key.lower_corner;
    const auto bounds =
        MakeWorldBounds({corner.x, corner.y, corner.z}, leaf.key.width_exponent, -1, finest_width_m, origin);
    if (!bounds.ok()) {
      return bounds.status();
    }
    const auto volume = MeasureLeafVolume(*bounds, leaf, planes);
    if (!volume.ok()) {
      return volume.status();
    }
    result.leaves_.push_back({
        .world_lower_m = ToVec3(bounds->lower),
        .world_upper_m = ToVec3(bounds->upper),
        .fluid_volume_m3 = *volume,
    });
  }
  for (std::size_t ordinal = 0; ordinal < interior.size(); ++ordinal) {
    const auto& face = interior[ordinal];
    const auto measured =
        MeasureSubface(face.face_index, face.width_exponent, false, finest_width_m, origin, planes, wall_velocity);
    if (!measured.ok()) {
      return measured.status();
    }
    result.subfaces_[ownership->interior_face_slots[ordinal]] = *measured;
  }
  for (const auto& face : ownership->exterior_faces) {
    const auto measured =
        MeasureSubface(face.face_index, face.width_exponent, true, finest_width_m, origin, planes, wall_velocity);
    if (!measured.ok()) {
      return measured.status();
    }
    result.subfaces_[face.face_slot] = *measured;
  }
  return result;
}

absl::StatusOr<pressure_internal::PreparedMacPressureSystem3D> PrepareGradedPlanarLiquidPressure3D(
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const GradedLiquidPressureCellFields3D> cell_fields,
    std::span<const GradedLiquidPressureSubfaceFields3D> subface_fields
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
    if (measured.fluid_volume_m3 == 0) {
      return absl::FailedPreconditionError("Graded liquid pressure requires positive volume on every leaf");
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
  return pressure_internal::PrepareAdaptiveMacPressureSystem3D(topology, geometry.step_duration_s(), cells, faces);
}

}  // namespace fluidity
