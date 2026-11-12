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

// The cube's tetrahedra and their boundary triangles have vertices on a
// chain of Cartesian edges. The solid interpolant is affine on each simplex.
template <std::size_t CornerCount>
FLUIDITY_HD inline ClippingPlane
SampledSolidExteriorPlane(const numeric::array<double, 8>& solid_distance, const int (&corners)[CornerCount]) {
  ClippingPlane plane{.normal = {}, .plane_coordinate = solid_distance[corners[0]]};
  for (std::size_t edge = 1; edge < CornerCount; ++edge) {
    const int previous_corner = corners[edge - 1];
    const int next_corner = corners[edge];
    const int direction = next_corner ^ previous_corner;
    const int axis = direction == 4 ? 2 : (direction == 2 ? 1 : 0);
    plane.normal[axis] = solid_distance[previous_corner] - solid_distance[next_corner];
  }
  return plane;
}

// Measures piecewise-linear liquid outside piecewise-linear solid. Adjacent
// cells must borrow identical samples on their shared face. Uses the pressure
// geometry's tetrahedra and face diagonals. Zero plateaus are closed solid
// and contribute no wet volume.
// Invalid widths or samples return kInvalid. Unrepresentable measures or a
// normalized solid sample losing precision return kRange. No partial measures
// are returned, and the input remains unchanged.
FLUIDITY_HD inline Value<SampledSolidLiquidMeasures> MeasureSampledSolidLiquidCell(const SampledSolidLiquidCell& cell) {
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
    result.wet_volume += ComputeClippedVolume(exterior, width);
  }

  constexpr double kMeasureSlack = 64 * numeric::numeric_limits<double>::epsilon();
  if (!IsNormalOrZero(result.wet_volume) || result.wet_volume / full_volume > 1 + kMeasureSlack) {
    return Error::kRange;
  }
  result.wet_volume = numeric::min(result.wet_volume, full_volume);

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
        open_area += ComputeOpenFaceArea(triangle, SampledSolidExteriorPlane(scaled_solid_distance, corners), width);
      }
      if (!IsNormalOrZero(open_area) || open_area / full_area > 1 + kMeasureSlack) {
        return Error::kRange;
      }
      result.open_area[2 * axis + side] = numeric::min(open_area, full_area);
    }
  }
  return result;
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
    cell.liquid_distance[corner] = input.corner_distances[corner];
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
    if (!IsNormalOrZero(result.blocked_face_area)) {
      return Error::kRange;
    }
  } else {
    result.high_side_liquid_volume = measures->wet_volume;
  }
  return result;
}

}  // namespace fluidity::boundary_internal
