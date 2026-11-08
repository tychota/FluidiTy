#pragma once
#include "fluidity/base/liquid_boundary_math.h"
namespace fluidity::boundary_internal {
// A tetrahedron clipped by liquid and one solid plane has at most six faces.
// Caps retain duplicate edge hits temporarily; 32 vertices bounds that scratch.
template <typename T, std::size_t Capacity>
struct BoundedList {
  T values[Capacity]{};
  std::size_t count = 0;
  bool overflow = false;
  BoundedList() = default;
  FLUIDITY_HD BoundedList(numeric::initializer_list<T> inputs) {
    for (const auto& input : inputs) {
      push_back(input);
    }
  }
  FLUIDITY_HD void push_back(const T& value) {
    if (count == Capacity) {
      overflow = true;
      return;
    }
    values[count++] = value;
  }
  FLUIDITY_HD bool empty() const {
    return count == 0;
  }
  FLUIDITY_HD std::size_t size() const {
    return count;
  }
  FLUIDITY_HD T* begin() {
    return values;
  }
  FLUIDITY_HD T* end() {
    return values + count;
  }
  FLUIDITY_HD const T* begin() const {
    return values;
  }
  FLUIDITY_HD const T* end() const {
    return values + count;
  }
  FLUIDITY_HD T& operator[](std::size_t index) {
    return values[index];
  }
  FLUIDITY_HD const T& operator[](std::size_t index) const {
    return values[index];
  }
  FLUIDITY_HD const T& front() const {
    return values[0];
  }
  FLUIDITY_HD const T& back() const {
    return values[count - 1];
  }
};
template <typename T, typename Less>
FLUIDITY_HD inline void InsertionSort(T* first, T* last, Less less) {
  if (first == last) {
    return;
  }
  for (T* current = first + 1; current != last; ++current) {
    T value = *current;
    T* position = current;
    while (position != first && less(value, position[-1])) {
      *position = position[-1];
      --position;
    }
    *position = value;
  }
}
// Each query is one checked topology element. No host-computed aperture,
// volume, gradient, curvature, capillary bound, theta or pressure lift is uploaded.
enum class GeometryMode {
  kMarker,
  kCell,
  kFace,
  kCurvature,
  kCapillary,
  kLink,
};
struct GeometryInput {
  GeometryMode mode{};
  UniformMacGeometry3D geometry{};
  CellIndex3D index{};
  Obstacle obstacle{};
  double corner_distances[8]{};  // Physical liquid distances, m; x/y/z corner bits.
  double derivative_distances[27]{};  // Physical distances, m; 3x3x3 halo, x fastest; eight outer corners unused.
  int axis = 0;
  double liquid_density = 0;  // kg/m^3.
  double surface_tension = 0;  // N/m.
  double ambient_pressure = 0;  // Gauge Pa.
  double interface_curvature = 0;  // Outward 1/m.
  double liquid_center_distance = 0;  // m, negative.
  double air_center_distance = 0;  // m, nonnegative.
};
struct GeometryOutput {
  double low_side_liquid_volume = 0;  // m^3.
  double high_side_liquid_volume = 0;  // m^3; all volume when no obstacle.
  double open_face_area = 0;  // m^2.
  double blocked_face_area = 0;  // Exactly h^2 - open_face_area, m^2.
  double interface_curvature = 0;  // Outward 1/m.
  double capillary_step_bound = 0;  // s.
  double pressure_gradient_distance = 0;  // m, regularized theta*h.
  double exterior_pressure = 0;  // Pa, ambient plus exactly one lift.
  bool lower_theta_regularized = false;
};
using CartesianVector = numeric::array<double, 3>;
FLUIDITY_HD inline CartesianVector Add(CartesianVector left, CartesianVector right) {
  return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}
FLUIDITY_HD inline CartesianVector Sub(CartesianVector left, CartesianVector right) {
  return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}
FLUIDITY_HD inline CartesianVector Scale(CartesianVector vector, double factor) {
  return {vector[0] * factor, vector[1] * factor, vector[2] * factor};
}
FLUIDITY_HD inline double Dot(CartesianVector left, CartesianVector right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}
FLUIDITY_HD inline CartesianVector Cross(CartesianVector left, CartesianVector right) {
  return {
      left[1] * right[2] - left[2] * right[1],
      left[2] * right[0] - left[0] * right[2],
      left[0] * right[1] - left[1] * right[0],
  };
}
FLUIDITY_HD inline CartesianVector CartesianComponents(Vec3 vector) {
  return {vector.x, vector.y, vector.z};
}
FLUIDITY_HD inline bool HasFiniteComponents(CartesianVector vector) {
  return numeric::all_of(vector.begin(), vector.end(), [](double component) {
    return numeric::isfinite(component);
  });
}
FLUIDITY_HD inline bool IsPositiveFinite(double value) {
  return numeric::isfinite(value) && value > 0;
}
FLUIDITY_HD inline bool IsNormalOrZero(double value) {
  return value == 0.0 || numeric::isnormal(value);
}
// Clip in cell-width coordinates, retaining the interpolated physical liquid
// value. The same diagonal triangulation induces every tetrahedron boundary.
struct ClippingVertex {
  CartesianVector cell_width_position;
  double liquid_distance;
};
using ClippingPolygon = BoundedList<ClippingVertex, 32>;
using ClippingPolyhedron = BoundedList<ClippingPolygon, 8>;
struct ClippingPlane {
  CartesianVector normal;
  double plane_coordinate;
  bool uses_liquid_distance = false;
};
FLUIDITY_HD inline double EvaluateClipDistance(const ClippingVertex& vertex, const ClippingPlane& plane) {
  return plane.uses_liquid_distance ? vertex.liquid_distance
                                    : Dot(plane.normal, vertex.cell_width_position) - plane.plane_coordinate;
}
FLUIDITY_HD inline ClippingPolygon
ClipPolygon(const ClippingPolygon& polygon, const ClippingPlane& plane, ClippingPolygon* intersections = nullptr) {
  ClippingPolygon clipped;
  if (polygon.overflow) {
    clipped.overflow = true;
    return clipped;
  }
  if (polygon.empty()) {
    return clipped;
  }
  ClippingVertex previous = polygon.back();
  double previous_distance = EvaluateClipDistance(previous, plane);
  for (const auto& current : polygon) {
    const double current_distance = EvaluateClipDistance(current, plane);
    if ((previous_distance <= 0) != (current_distance <= 0)) {
      // Normalize before subtraction so finite distances of opposite sign cannot overflow.
      const double magnitude = numeric::max(numeric::abs(previous_distance), numeric::abs(current_distance));
      const double previous_scaled = previous_distance / magnitude;
      const double current_scaled = current_distance / magnitude;
      const double scaled_difference = previous_scaled - current_scaled;
      const double previous_weight = -current_scaled / scaled_difference;
      const double current_weight = previous_scaled / scaled_difference;
      // Compute both weights directly: 1-t loses a tiny positive cut when t rounds to one.
      ClippingVertex intersection{
          Add(Scale(previous.cell_width_position, previous_weight), Scale(current.cell_width_position, current_weight)),
          previous_weight * previous.liquid_distance + current_weight * current.liquid_distance,
      };
      clipped.push_back(intersection);
      if (intersections != nullptr) {
        intersections->push_back(intersection);
      }
    }
    if (current_distance <= 0) {
      clipped.push_back(current);
    }
    previous = current;
    previous_distance = current_distance;
  }
  return clipped;
}
// Normalize before cross products: local cap directions can be tiny even
// when the final physical face area or cell volume is comfortably normal.
FLUIDITY_HD inline CartesianVector UnitDirection(CartesianVector direction) {
  const double largest_component =
      numeric::max({numeric::abs(direction[0]), numeric::abs(direction[1]), numeric::abs(direction[2])});
  if (largest_component == 0) {
    return {};
  }
  int exponent = 0;
  (void)numeric::frexp(largest_component, &exponent);
  for (double& component : direction) {
    component = numeric::ldexp(component, -exponent);
  }
  return Scale(direction, 1 / numeric::hypot(direction[0], direction[1], direction[2]));
}
FLUIDITY_HD inline ClippingPolyhedron ClipPolyhedron(const ClippingPolyhedron& polyhedron, const ClippingPlane& plane) {
  ClippingPolyhedron result;
  if (polyhedron.overflow) {
    result.overflow = true;
    return result;
  }
  ClippingPolygon intersections;
  for (const auto& polygon : polyhedron) {
    auto clipped = ClipPolygon(polygon, plane, &intersections);
    if (clipped.overflow || intersections.overflow) {
      result.overflow = true;
      return result;
    }
    if (clipped.size() >= 3) {
      result.push_back(numeric::move(clipped));
    }
  }
  ClippingPolygon cap;
  for (const auto& vertex : intersections) {
    if (numeric::none_of(cap.begin(), cap.end(), [&](const auto& other) {
          return other.cell_width_position == vertex.cell_width_position;
        })) {
      cap.push_back(vertex);
    }
  }
  if (cap.size() >= 3) {
    CartesianVector center{};
    for (const auto& vertex : cap) {
      center = Add(center, Scale(vertex.cell_width_position, 1.0 / cap.size()));
    }
    const CartesianVector tangent = UnitDirection(Sub(cap.front().cell_width_position, center));
    CartesianVector normal{};
    for (numeric::size_t i = 1; i < cap.size(); ++i) {
      normal = UnitDirection(Cross(tangent, UnitDirection(Sub(cap[i].cell_width_position, center))));
      if (numeric::hypot(normal[0], normal[1], normal[2]) > 0) {
        break;
      }
    }
    const CartesianVector bitangent = Cross(normal, tangent);
    const double tangent_length = numeric::hypot(tangent[0], tangent[1], tangent[2]);
    const double bitangent_length = numeric::hypot(bitangent[0], bitangent[1], bitangent[2]);
    if (tangent_length > 0 && bitangent_length > 0) {
      const auto unit_tangent = Scale(tangent, 1 / tangent_length);
      const auto unit_bitangent = Scale(bitangent, 1 / bitangent_length);
      InsertionSort(cap.begin(), cap.end(), [&](const auto& a, const auto& b) {
        const auto delta_a = Sub(a.cell_width_position, center);
        const auto delta_b = Sub(b.cell_width_position, center);
        return numeric::atan2(Dot(delta_a, unit_bitangent), Dot(delta_a, unit_tangent))
            < numeric::atan2(Dot(delta_b, unit_bitangent), Dot(delta_b, unit_tangent));
      });
      result.push_back(numeric::move(cap));
    }
  }
  return result;
}
// Coordinate rows are scaled independently before a determinant. This also
// handles long thin prisms, whose edge vectors can be nearly parallel: merely
// scaling each entire vector would still multiply two tiny transverse entries.
template <numeric::size_t VectorCount>
FLUIDITY_HD inline int NormalizeCoordinateRow(numeric::array<CartesianVector, VectorCount>& edges, int axis) {
  double largest_component = 0;
  for (const auto& edge : edges) {
    largest_component = numeric::max(largest_component, numeric::abs(edge[axis]));
  }
  if (largest_component == 0) {
    return 0;
  }
  int exponent = 0;
  (void)numeric::frexp(largest_component, &exponent);
  for (auto& edge : edges) {
    edge[axis] = numeric::ldexp(edge[axis], -exponent);
  }
  return exponent;
}

// Combine the local determinant's binary exponent with h^dimension before
// rounding the physical measure. Nonzero contributions that cannot be represented
// propagate NaN to the checked caller rather than masquerading as empty geometry.
FLUIDITY_HD inline double PhysicalMeasure(double determinant, int local_exponent, double cell_width, int dimension) {
  if (determinant == 0) {
    return 0;
  }
  int width_exponent = 0;
  const double width_mantissa = numeric::frexp(cell_width, &width_exponent);
  double physical_mantissa = determinant;
  for (int axis = 0; axis < dimension; ++axis) {
    physical_mantissa *= width_mantissa;
  }
  const double physical_measure = numeric::ldexp(physical_mantissa, local_exponent + dimension * width_exponent);
  if (!numeric::isfinite(physical_measure) || physical_measure == 0) {
    return numeric::numeric_limits<double>::quiet_NaN();
  }
  return physical_measure;
}

FLUIDITY_HD inline double ComputeClippedVolume(const ClippingPolyhedron& polyhedron, double cell_width) {
  if (polyhedron.overflow) {
    return numeric::numeric_limits<double>::quiet_NaN();
  }
  CartesianVector center{};
  numeric::size_t count = 0;
  for (const auto& polygon : polyhedron) {
    if (polygon.overflow) {
      return numeric::numeric_limits<double>::quiet_NaN();
    }
    for (const auto& vertex : polygon) {
      center = Add(center, vertex.cell_width_position);
      ++count;
    }
  }
  if (count == 0) {
    return 0;
  }
  center = Scale(center, 1.0 / count);
  double physical_volume = 0;
  for (const auto& polygon : polyhedron) {
    for (numeric::size_t vertex = 1; vertex + 1 < polygon.size(); ++vertex) {
      numeric::array edges{
          Sub(polygon[0].cell_width_position, center),
          Sub(polygon[vertex].cell_width_position, center),
          Sub(polygon[vertex + 1].cell_width_position, center),
      };
      int determinant_exponent = 0;
      for (int axis = 0; axis < 3; ++axis) {
        determinant_exponent += NormalizeCoordinateRow(edges, axis);
      }
      const double determinant = numeric::abs(Dot(edges[0], Cross(edges[1], edges[2]))) / 6;
      physical_volume += PhysicalMeasure(determinant, determinant_exponent, cell_width, 3);
    }
  }
  return physical_volume;
}
FLUIDITY_HD inline double ComputeClippedArea(const ClippingPolygon& polygon, double cell_width) {
  if (polygon.overflow) {
    return numeric::numeric_limits<double>::quiet_NaN();
  }
  double physical_area = 0;
  for (numeric::size_t vertex = 1; vertex + 1 < polygon.size(); ++vertex) {
    const numeric::array edges{
        Sub(polygon[vertex].cell_width_position, polygon[0].cell_width_position),
        Sub(polygon[vertex + 1].cell_width_position, polygon[0].cell_width_position),
    };
    CartesianVector physical_half_cross{};
    for (int axis = 0; axis < 3; ++axis) {
      auto normalized_edges = edges;
      const int first_axis = (axis + 1) % 3;
      const int second_axis = (axis + 2) % 3;
      const int first_axis_exponent = NormalizeCoordinateRow(normalized_edges, first_axis);
      const int second_axis_exponent = NormalizeCoordinateRow(normalized_edges, second_axis);
      const int area_exponent = first_axis_exponent + second_axis_exponent;
      const double determinant = (normalized_edges[0][first_axis] * normalized_edges[1][second_axis]
                                  - normalized_edges[0][second_axis] * normalized_edges[1][first_axis])
          / 2;
      physical_half_cross[axis] = PhysicalMeasure(determinant, area_exponent, cell_width, 2);
    }
    physical_area += numeric::hypot(physical_half_cross[0], physical_half_cross[1], physical_half_cross[2]);
  }
  return physical_area;
}
// A polygon contained in the solid boundary is closed. Strictness changes area
// only for this coplanar case, and avoids counting the same face on both sides.
FLUIDITY_HD inline double ComputeOpenFaceArea(const ClippingPolygon& polygon, ClippingPlane plane, double cell_width) {
  if (polygon.overflow) {
    return numeric::numeric_limits<double>::quiet_NaN();
  }
  if (numeric::none_of(polygon.begin(), polygon.end(), [&](const auto& vertex) {
        return EvaluateClipDistance(vertex, plane) < 0;
      })) {
    return 0;
  }
  return ComputeClippedArea(ClipPolygon(polygon, plane), cell_width);
}
struct ObstaclePlanes {
  bool present = false;
  bool slab = false;
  CartesianVector normal{};
  double low = 0;
  double high = 0;
  CartesianVector velocity{};
};

FLUIDITY_HD inline Value<ObstaclePlanes> LocalObstacle(const GeometryInput& input) {
  const auto& obstacle = input.obstacle;
  ObstaclePlanes result{
      obstacle.kind != 0,
      obstacle.kind == 2,
      CartesianComponents(obstacle.normal),
      obstacle.coordinate - (obstacle.kind == 2 ? obstacle.half_width : 0),
      obstacle.coordinate + (obstacle.kind == 2 ? obstacle.half_width : 0),
      CartesianComponents(obstacle.velocity),
  };
  if (!result.present) {
    return result;
  }
  const auto geometry = input.geometry;
  const auto index = input.index;
  const CartesianVector anchor{
      geometry.world_origin.x + geometry.cell_width * index.x,
      geometry.world_origin.y + geometry.cell_width * index.y,
      geometry.world_origin.z + geometry.cell_width * index.z,
  };
  const double origin_projection = Dot(result.normal, anchor);
  result.low = (result.low - origin_projection) / geometry.cell_width;
  result.high = (result.high - origin_projection) / geometry.cell_width;
  if (!HasFiniteComponents(anchor) || !numeric::isfinite(result.low) || !numeric::isfinite(result.high)
      || (result.slab && result.low >= result.high)) {
    return Error::kRange;
  }
  return result;
}
FLUIDITY_HD inline Value<GeometryOutput> ComputeGeometry(const GeometryInput& input) {
  const double h = input.geometry.cell_width;
  const double full_area = h * h;
  const double full_volume = full_area * h;
  GeometryOutput output;
  if (input.mode == GeometryMode::kMarker) {
    return output;
  }
  if (input.mode == GeometryMode::kCapillary) {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    const double square = input.liquid_density * full_volume / (4 * kPi * input.surface_tension);
    if (!numeric::isnormal(square)) {
      return Error::kRange;
    }
    output.capillary_step_bound = numeric::sqrt(square);
    return output;
  }
  if (input.mode == GeometryMode::kLink) {
    const double magnitude = numeric::max(-input.liquid_center_distance, input.air_center_distance);
    const double liquid_scaled = -input.liquid_center_distance / magnitude;
    const double theta = liquid_scaled / (liquid_scaled + input.air_center_distance / magnitude);
    output.lower_theta_regularized = theta < .05;
    output.pressure_gradient_distance = numeric::clamp(theta, .05, 1.0) * h;
    const double lift = input.surface_tension * input.interface_curvature;
    output.exterior_pressure = input.ambient_pressure + lift;
    if ((input.surface_tension > 0 && input.interface_curvature != 0 && lift == 0) || !IsNormalOrZero(lift)
        || !numeric::isfinite(output.exterior_pressure) || !numeric::isnormal(output.pressure_gradient_distance)) {
      return Error::kRange;
    }
    return output;
  }
  if (input.mode == GeometryMode::kCurvature) {
    CartesianVector gradient{};
    numeric::array<CartesianVector, 3> hessian{};
    constexpr int strides[3]{1, 3, 9};
    for (int axis = 0; axis < 3; ++axis) {
      const double low = input.derivative_distances[13 - strides[axis]];
      const double high = input.derivative_distances[13 + strides[axis]];
      gradient[axis] = (high - low) / (2 * h);
      hessian[axis][axis] = ((high - input.derivative_distances[13]) + (low - input.derivative_distances[13])) / h / h;
      for (int other = 0; other < axis; ++other) {
        double mixed = 0;
        for (int first = -1; first <= 1; first += 2) {
          for (int second = -1; second <= 1; second += 2) {
            mixed +=
                first * second * (input.derivative_distances[13 + first * strides[axis] + second * strides[other]] / 4);
          }
        }
        hessian[axis][other] = mixed / h / h;
        hessian[other][axis] = hessian[axis][other];
      }
    }
    const double norm = numeric::hypot(gradient[0], gradient[1], gradient[2]);
    if (!HasFiniteComponents(gradient) || !numeric::isfinite(norm)) {
      return Error::kRange;
    }
    if (norm < 1e-8) {
      return Error::kPrecondition;
    }
    const CartesianVector normal = Scale(gradient, 1 / norm);
    for (int axis = 0; axis < 3; ++axis) {
      if (!HasFiniteComponents(hessian[axis])) {
        return Error::kRange;
      }
      output.interface_curvature += (hessian[axis][axis] - normal[axis] * Dot(hessian[axis], normal)) / norm;
    }
    if (!numeric::isfinite(output.interface_curvature)) {
      return Error::kRange;
    }
    return output;
  }
  const auto local = LocalObstacle(input);
  if (!local.ok()) {
    return local.status();
  }
  if (input.mode == GeometryMode::kCell) {
    ClippingVertex vertices[8]{};
    for (int corner = 0; corner < 8; ++corner) {
      for (int axis = 0; axis < 3; ++axis) {
        vertices[corner].cell_width_position[axis] = (corner >> axis) & 1;
      }
      vertices[corner].liquid_distance = input.corner_distances[corner];
    }
    // Six body-diagonal tetrahedra induce identical diagonals on shared faces.
    constexpr int permutations[6][3]{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (const auto& permutation : permutations) {
      const int corners[4]{0, 1 << permutation[0], (1 << permutation[0]) | (1 << permutation[1]), 7};
      bool negative = false;
      for (int corner : corners) {
        negative = negative || vertices[corner].liquid_distance < 0;
      }
      // An all-zero simplex is an interface plateau, not liquid volume.
      if (!negative) {
        continue;
      }
      const auto a = vertices[corners[0]];
      const auto b = vertices[corners[1]];
      const auto c = vertices[corners[2]];
      const auto d = vertices[corners[3]];
      const auto liquid = ClipPolyhedron({{a, b, c}, {a, b, d}, {a, c, d}, {b, c, d}}, {{}, 0, true});
      if (!local->present) {
        output.high_side_liquid_volume += ComputeClippedVolume(liquid, h);
      } else {
        output.high_side_liquid_volume +=
            ComputeClippedVolume(ClipPolyhedron(liquid, {Scale(local->normal, -1), -local->high}), h);
        if (local->slab) {
          output.low_side_liquid_volume += ComputeClippedVolume(ClipPolyhedron(liquid, {local->normal, local->low}), h);
        }
      }
    }
    return output;
  }
  int transverse[2]{};
  int count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis != input.axis) {
      transverse[count++] = axis;
    }
  }
  ClippingVertex vertices[4]{};
  vertices[1].cell_width_position[transverse[0]] = 1;
  vertices[2].cell_width_position[transverse[0]] = 1;
  vertices[2].cell_width_position[transverse[1]] = 1;
  vertices[3].cell_width_position[transverse[1]] = 1;
  double geometric_blocked = 0;
  double geometric_open = 0;
  for (int triangle = 0; triangle < 2; ++triangle) {
    const ClippingPolygon polygon{vertices[0], vertices[triangle + 1], vertices[triangle + 2]};
    if (!local->present) {
      geometric_open += ComputeClippedArea(polygon, h);
      continue;
    }
    auto blocked = ClipPolygon(polygon, {local->normal, local->high});
    if (local->slab) {
      blocked = ClipPolygon(blocked, {Scale(local->normal, -1), -local->low});
    }
    geometric_blocked += ComputeClippedArea(blocked, h);
    geometric_open += ComputeOpenFaceArea(polygon, {Scale(local->normal, -1), -local->high}, h);
    if (local->slab) {
      geometric_open += ComputeOpenFaceArea(polygon, {local->normal, local->low}, h);
    }
  }
  output.open_face_area = numeric::min(geometric_open, full_area);
  output.blocked_face_area = full_area - output.open_face_area;
  if (!numeric::isfinite(geometric_open) || !numeric::isfinite(geometric_blocked)
      || geometric_open / full_area > 1 + 64 * numeric::numeric_limits<double>::epsilon()
      || !IsNormalOrZero(output.open_face_area) || !IsNormalOrZero(output.blocked_face_area)
      || (geometric_blocked > 0 && output.blocked_face_area == 0)) {
    return Error::kRange;
  }
  return output;
}
}  // namespace fluidity::boundary_internal
