#pragma once

#include "fluidity/base/liquid_boundary_geometry_math_3d.h"

namespace fluidity::boundary_internal {

struct SampledSolidLiquidCell {
  double cell_width;  // World metres.
  numeric::array<double, 8> liquid_distance{};  // Negative inside liquid, x/y/z corner bits.
  numeric::array<double, 8> solid_distance{};  // Positive outside solid, same corners in metres.
};

struct SampledSolidLiquidMeasures {
  double wet_volume = 0;  // Liquid outside solid, m³.
  numeric::array<double, 6> open_area{};  // Area outside solid, m², low/high x, y, and z.
};

// The simplex vertices form a chain of increasing corner bits. Use one
// changing axis per edge to extend the affine samples off the simplex.
template <std::size_t CornerCount>
FLUIDITY_HD inline ClippingPlane
SampledSolidExteriorPlane(const numeric::array<double, 8>& solid_distance, const int (&corners)[CornerCount]) {
  ClippingPlane plane{.normal = {}, .plane_coordinate = solid_distance[corners[0]]};
  for (std::size_t edge = 1; edge < CornerCount; ++edge) {
    const int previous_corner = corners[edge - 1];
    const int next_corner = corners[edge];
    const int direction = next_corner ^ previous_corner;
    const int axis = (direction & 1) != 0 ? 0 : ((direction & 2) != 0 ? 1 : 2);
    plane.normal[axis] = solid_distance[previous_corner] - solid_distance[next_corner];
  }
  return plane;
}

// Restricts a face polygon to the open side of an optional analytic
// boundary. A polygon exactly on that boundary is closed.
FLUIDITY_HD inline ClippingPolygon
ClipSampledAvailableFace(const ClippingPolygon& polygon, const ClippingPlane* analytic_fluid_side) {
  if (analytic_fluid_side == nullptr || polygon.overflow) {
    return polygon;
  }
  if (numeric::none_of(polygon.begin(), polygon.end(), [&](const ClippingVertex& vertex) {
        return EvaluateClipDistance(vertex, *analytic_fluid_side) < 0;
      })) {
    return {};
  }
  return ClipPolygon(polygon, *analytic_fluid_side);
}

// Wet tetrahedra are convex. Their only triangular interior faces
// form a ring around the cube's body diagonal. A positive-area wet
// opening on a shared triangle connects its incident pieces.
FLUIDITY_HD inline Error CheckSampledWetConnectivity(
    const ClippingVertex (&vertices)[8],
    const numeric::array<double, 8>& solid_distance,
    const bool (&wet)[6],
    double width,
    const ClippingPlane* analytic_fluid_side
) {
  const auto wet_corner = [&](int corner) {
    return vertices[corner].liquid_distance < 0 && solid_distance[corner] > 0
        && (analytic_fluid_side == nullptr || EvaluateClipDistance(vertices[corner], *analytic_fluid_side) < 0);
  };
  if (wet_corner(0) || wet_corner(7)) {
    return Error::kNone;
  }
  int components = 0;
  for (const bool piece : wet) {
    components += piece;
  }
  if (components <= 1) {
    return Error::kNone;
  }
  constexpr int kTetrahedronRing[6]{0, 1, 4, 5, 3, 2};
  constexpr int kSharedCorner[6]{1, 5, 4, 6, 2, 3};
  int group[6]{0, 1, 2, 3, 4, 5};
  for (int edge = 0; edge < 6; ++edge) {
    const int first = kTetrahedronRing[edge];
    const int second = kTetrahedronRing[(edge + 1) % 6];
    if (!wet[first] || !wet[second] || group[first] == group[second]) {
      continue;
    }
    const int middle = kSharedCorner[edge];
    const ClippingPolygon triangle{vertices[0], vertices[middle], vertices[7]};
    if (numeric::none_of(triangle.begin(), triangle.end(), [](const ClippingVertex& vertex) {
          return vertex.liquid_distance < 0;
        })) {
      continue;
    }
    const int corners[3]{0, middle, 7};
    if (solid_distance[0] <= 0 && solid_distance[middle] <= 0 && solid_distance[7] <= 0) {
      continue;
    }
    const auto liquid = ClipPolygon(triangle, {{}, 0, true});
    const auto available_liquid = ClipSampledAvailableFace(liquid, analytic_fluid_side);
    const double wet_area =
        ComputeOpenFaceArea(available_liquid, SampledSolidExteriorPlane(solid_distance, corners), width);
    if (!IsNormalOrZero(wet_area)) {
      return Error::kRange;
    }
    if (wet_area > 0) {
      const int old_group = group[second];
      const int new_group = group[first];
      for (int& member : group) {
        if (member == old_group) {
          member = new_group;
        }
      }
      --components;
    }
  }
  return components == 1 ? Error::kNone : Error::kPrecondition;
}

// Measures piecewise-linear liquid outside piecewise-linear solid. Adjacent
// cells must borrow identical samples on their shared face. Uses the pressure
// geometry's tetrahedra and face diagonals. Zero plateaus are closed solid
// and contribute no wet volume.
// Invalid widths or samples return kInvalid. Unrepresentable measures or a
// normalized solid sample losing precision return kRange. Disconnected wet
// regions return kPrecondition. No partial measures
// are returned, and the input remains unchanged.
// If supplied, analytic_fluid_side is the checked open halfspace in
// local cell-width coordinates. Its boundary is excluded from openings.
FLUIDITY_HD inline Value<SampledSolidLiquidMeasures>
MeasureSampledSolidLiquidCell(const SampledSolidLiquidCell& cell, const ClippingPlane* analytic_fluid_side = nullptr) {
  const double width = cell.cell_width;
  if (!numeric::isfinite(width) || width <= 0) {
    return Error::kInvalid;
  }
  const double full_area = width * width;
  const double full_volume = full_area * width;
  if (!numeric::isnormal(full_area) || !numeric::isnormal(full_volume)) {
    return Error::kRange;
  }
  ClippingVertex vertices[8]{};
  double maximum_solid_distance = 0;
  for (int corner = 0; corner < 8; ++corner) {
    const double liquid_distance = cell.liquid_distance[corner];
    const double solid_distance = cell.solid_distance[corner];
    if (!numeric::isfinite(liquid_distance) || !numeric::isfinite(solid_distance)) {
      return Error::kInvalid;
    }
    vertices[corner] = {
        .cell_width_position =
            {
                static_cast<double>(corner & 1),
                static_cast<double>((corner >> 1) & 1),
                static_cast<double>((corner >> 2) & 1),
            },
        .liquid_distance = liquid_distance,
    };
    maximum_solid_distance = numeric::max(maximum_solid_distance, numeric::abs(solid_distance));
  }
  // A common power of two bounds edge differences without changing the
  // clipping plane. Face samples then keep their ratios across cell scales.
  int solid_exponent = 0;
  (void)numeric::frexp(maximum_solid_distance, &solid_exponent);
  numeric::array<double, 8> scaled_solid_distance{};
  for (int corner = 0; corner < 8; ++corner) {
    scaled_solid_distance[corner] = numeric::ldexp(cell.solid_distance[corner], -solid_exponent);
    if (cell.solid_distance[corner] != 0 && !numeric::isnormal(scaled_solid_distance[corner])) {
      return Error::kRange;
    }
  }

  SampledSolidLiquidMeasures result;
  bool wet_tetrahedra[6]{};
  for (std::size_t tetrahedron = 0; tetrahedron < 6; ++tetrahedron) {
    const int corners[4]{
        0,
        CellTetrahedronCorner(tetrahedron, 1),
        CellTetrahedronCorner(tetrahedron, 2),
        7,
    };
    bool has_liquid = false;
    bool has_exterior = false;
    for (const int corner : corners) {
      has_liquid = has_liquid || vertices[corner].liquid_distance < 0;
      has_exterior = has_exterior || scaled_solid_distance[corner] > 0;
    }
    if (!has_liquid || !has_exterior) {
      continue;
    }
    const auto a = vertices[corners[0]];
    const auto b = vertices[corners[1]];
    const auto c = vertices[corners[2]];
    const auto d = vertices[corners[3]];
    const auto liquid = ClipPolyhedron({{a, b, c}, {a, b, d}, {a, c, d}, {b, c, d}}, {{}, 0, true});
    const auto exterior = ClipPolyhedron(liquid, SampledSolidExteriorPlane(scaled_solid_distance, corners));
    const double wet_volume = analytic_fluid_side == nullptr
        ? ComputeClippedVolume(exterior, width)
        : ComputeClippedVolume(ClipPolyhedron(exterior, *analytic_fluid_side), width);
    wet_tetrahedra[tetrahedron] = wet_volume > 0;
    result.wet_volume += wet_volume;
  }

  constexpr double kMeasureSlack = 64 * numeric::numeric_limits<double>::epsilon();
  if (!IsNormalOrZero(result.wet_volume) || result.wet_volume / full_volume > 1 + kMeasureSlack) {
    return Error::kRange;
  }
  result.wet_volume = numeric::min(result.wet_volume, full_volume);
  const auto connectivity =
      CheckSampledWetConnectivity(vertices, scaled_solid_distance, wet_tetrahedra, width, analytic_fluid_side);
  if (connectivity != Error::kNone) {
    return connectivity;
  }

  for (int axis = 0; axis < 3; ++axis) {
    const int axis_bit = 1 << axis;
    for (int side = 0; side < 2; ++side) {
      const int first_corner = side * axis_bit;
      const int last_corner = first_corner | (7 ^ axis_bit);
      double open_area = 0;
      for (int transverse_axis = 0; transverse_axis < 3; ++transverse_axis) {
        if (transverse_axis == axis) {
          continue;
        }
        const int middle_corner = first_corner | (1 << transverse_axis);
        const int corners[3]{first_corner, middle_corner, last_corner};
        const ClippingPolygon triangle{vertices[first_corner], vertices[middle_corner], vertices[last_corner]};
        const auto available = ClipSampledAvailableFace(triangle, analytic_fluid_side);
        open_area += ComputeOpenFaceArea(available, SampledSolidExteriorPlane(scaled_solid_distance, corners), width);
      }
      if (!IsNormalOrZero(open_area) || open_area / full_area > 1 + kMeasureSlack) {
        return Error::kRange;
      }
      result.open_area[2 * axis + side] = numeric::min(open_area, full_area);
    }
  }
  return result;
}

// Integrates the four piecewise-linear corner basis functions over the
// blocked part of a square MAC face. Samples are ordered by the two
// transverse corner bits, in metres and positive outside solid. The shared
// diagonal is 0 to 3. The resulting area weights in m² multiply sampled
// normal velocities for wall flux and distribute its conjugate load by
// the transpose. An all-zero solid plateau blocks the entire face.
// An optional analytic clip uses the same local face coordinates and
// excludes the portion occupied by the analytic obstacle.
// Invalid samples or width return kInvalid. Unrepresentable weights return
// kRange. Failure returns no partial weights.
FLUIDITY_HD inline Value<numeric::array<double, 4>> MeasureSampledSolidFaceWeights(
    double width,
    const numeric::array<double, 4>& solid_distance,
    const ClippingPlane* analytic_fluid_side = nullptr
) {
  if (!numeric::isfinite(width) || width <= 0) {
    return Error::kInvalid;
  }
  const double full_area = width * width;
  if (!numeric::isnormal(full_area)) {
    return Error::kRange;
  }

  double maximum_distance = 0;
  for (const double distance : solid_distance) {
    if (!numeric::isfinite(distance)) {
      return Error::kInvalid;
    }
    maximum_distance = numeric::max(maximum_distance, numeric::abs(distance));
  }
  int distance_exponent = 0;
  (void)numeric::frexp(maximum_distance, &distance_exponent);
  numeric::array<double, 8> scaled_distance{};
  ClippingVertex vertices[4]{};
  for (int corner = 0; corner < 4; ++corner) {
    scaled_distance[corner] = numeric::ldexp(solid_distance[corner], -distance_exponent);
    if (solid_distance[corner] != 0 && !numeric::isnormal(scaled_distance[corner])) {
      return Error::kRange;
    }
    vertices[corner].cell_width_position = {
        static_cast<double>(corner & 1),
        static_cast<double>((corner >> 1) & 1),
        0,
    };
  }

  numeric::array<double, 4> weights{};
  for (int axis = 0; axis < 2; ++axis) {
    const int middle = 1 << axis;
    const int corners[3]{0, middle, 3};
    const ClippingPolygon triangle{vertices[0], vertices[middle], vertices[3]};
    const auto exterior = SampledSolidExteriorPlane(scaled_distance, corners);
    const ClippingPlane interior{
        .normal = Scale(exterior.normal, -1),
        .plane_coordinate = -exterior.plane_coordinate,
    };
    const auto available = ClipSampledAvailableFace(triangle, analytic_fluid_side);
    const auto blocked = ClipPolygon(available, interior);
    if (blocked.overflow) {
      return Error::kRange;
    }
    for (std::size_t vertex = 1; vertex + 1 < blocked.size(); ++vertex) {
      const ClippingPolygon piece{blocked[0], blocked[vertex], blocked[vertex + 1]};
      const double area = ComputeClippedArea(piece, width);
      if (!IsNormalOrZero(area)) {
        return Error::kRange;
      }
      const auto centroid = Scale(
          Add(Add(piece[0].cell_width_position, piece[1].cell_width_position), piece[2].cell_width_position),
          1.0 / 3
      );
      const double along = numeric::clamp(centroid[axis], 0.0, 1.0);
      const double across = numeric::clamp(centroid[1 - axis], 0.0, along);
      weights[0] += area * (1 - along);
      weights[middle] += area * (along - across);
      weights[3] += area * across;
    }
  }

  const double measured_area = weights[0] + weights[1] + weights[2] + weights[3];
  constexpr double kMeasureSlack = 64 * numeric::numeric_limits<double>::epsilon();
  if (!IsNormalOrZero(measured_area) || measured_area / full_area > 1 + kMeasureSlack) {
    return Error::kRange;
  }
  return weights;
}

// Applies the transpose of the blocked-face wall-flux weights. Returns
// the solid corner forces in N along the positive tail-to-head normal.
// Pressures are physical endpoint pressures in Pa, with a common gauge.
// A capillary or numerical pressure jump is not part of this load.
// The resulting corner power equals (tail_pressure - head_pressure)
// times the signed wall flux. Nonfinite pressures or invalid area
// weights return kInvalid. Unrepresentable forces return kRange.
FLUIDITY_HD inline Value<numeric::array<double, 4>> ComputeSampledSolidFacePressureForces(
    const numeric::array<double, 4>& blocked_area_weights,
    double tail_pressure,
    double head_pressure
) {
  if (!numeric::isfinite(tail_pressure) || !numeric::isfinite(head_pressure)) {
    return Error::kInvalid;
  }
  const double solid_pressure_difference = tail_pressure - head_pressure;
  if (!numeric::isfinite(solid_pressure_difference)) {
    return Error::kRange;
  }
  numeric::array<double, 4> corner_force{};
  for (std::size_t corner = 0; corner < corner_force.size(); ++corner) {
    const double area_weight = blocked_area_weights[corner];
    if (!numeric::isfinite(area_weight) || area_weight < 0) {
      return Error::kInvalid;
    }
    corner_force[corner] = area_weight * solid_pressure_difference;
    if (!IsNormalOrZero(corner_force[corner])
        || (area_weight > 0 && solid_pressure_difference != 0 && corner_force[corner] == 0)) {
      return Error::kRange;
    }
  }
  return corner_force;
}

FLUIDITY_HD inline bool IsSampledSolidMode(GeometryMode mode) {
  return mode == GeometryMode::kSampledSolidCell || mode == GeometryMode::kSampledSolidFace;
}

// Maps the staged samples to the existing freeze query result. A face
// query supplies the four corners on the low side of the selected axis.
FLUIDITY_HD inline Value<GeometryOutput> ComputeSampledSolidGeometry(const GeometryInput& input) {
  const bool is_face = input.mode == GeometryMode::kSampledSolidFace;
  if (is_face && (input.axis < 0 || input.axis >= 3)) {
    return Error::kInvalid;
  }
  SampledSolidLiquidCell cell{.cell_width = input.geometry.cell_width};
  for (int corner = 0; corner < 8; ++corner) {
    cell.liquid_distance[corner] = is_face ? 0.0 : input.corner_distances[corner];
    cell.solid_distance[corner] = input.derivative_distances[corner];
  }
  const auto measures = MeasureSampledSolidLiquidCell(cell);
  if (!measures.ok()) {
    return measures.status();
  }
  GeometryOutput result;
  if (is_face) {
    result.open_face_area = measures->open_area[2 * input.axis];
    result.blocked_face_area = cell.cell_width * cell.cell_width - result.open_face_area;
    numeric::array<double, 4> solid_distance{};
    numeric::array<double, 4> normal_velocity{};
    int face_corner = 0;
    for (int corner = 0; corner < 8; ++corner) {
      if ((corner & (1 << input.axis)) == 0) {
        solid_distance[face_corner] = input.derivative_distances[corner];
        normal_velocity[face_corner] = input.corner_distances[corner];
        if (!numeric::isfinite(normal_velocity[face_corner])) {
          return Error::kInvalid;
        }
        ++face_corner;
      }
    }
    const auto weights = MeasureSampledSolidFaceWeights(cell.cell_width, solid_distance);
    if (!weights.ok()) {
      return weights.status();
    }
    double weighted_area = 0.0;
    for (int corner = 0; corner < 4; ++corner) {
      weighted_area += (*weights)[corner];
      result.prescribed_wall_flux += (*weights)[corner] * normal_velocity[corner];
    }
    if (!IsNormalOrZero(result.blocked_face_area) || !IsNormalOrZero(result.prescribed_wall_flux)
        || (weighted_area > 0 && result.blocked_face_area == 0)) {
      return Error::kRange;
    }
    if (weighted_area > 0) {
      result.piston_normal_velocity = result.prescribed_wall_flux / weighted_area;
      if (!numeric::isfinite(result.piston_normal_velocity)) {
        return Error::kRange;
      }
    }
  } else {
    result.high_side_liquid_volume = measures->wet_volume;
  }
  return result;
}

}  // namespace fluidity::boundary_internal
