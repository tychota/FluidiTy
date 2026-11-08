#include "fluidity/host/liquid_boundary_geometry_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <tuple>
#include <utility>

#include "absl/status/status.h"

namespace fluidity {
namespace {
using CellLatticeKey = std::tuple<std::int32_t, std::int32_t, std::int32_t>;
template <typename Index>
CellLatticeKey CellKey(Index index) {
  return {index.z, index.y, index.x};
}
using CartesianVector = std::array<double, 3>;
CartesianVector Add(CartesianVector left, CartesianVector right) {
  return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}
CartesianVector Sub(CartesianVector left, CartesianVector right) {
  return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}
CartesianVector Scale(CartesianVector vector, double factor) {
  return {vector[0] * factor, vector[1] * factor, vector[2] * factor};
}
double Dot(CartesianVector left, CartesianVector right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}
CartesianVector Cross(CartesianVector left, CartesianVector right) {
  return {
      left[1] * right[2] - left[2] * right[1],
      left[2] * right[0] - left[0] * right[2],
      left[0] * right[1] - left[1] * right[0],
  };
}
CartesianVector CartesianComponents(Vec3 vector) {
  return {vector.x, vector.y, vector.z};
}
bool HasFiniteComponents(CartesianVector vector) {
  return std::ranges::all_of(vector, [](double component) {
    return std::isfinite(component);
  });
}
bool IsPositiveFinite(double value) {
  return std::isfinite(value) && value > 0;
}
bool IsNormalOrZero(double value) {
  return value == 0.0 || std::isnormal(value);
}
absl::Status UnrepresentableGeometry() {
  return absl::OutOfRangeError("Unrepresentable boundary geometry arithmetic");
}
absl::Status MissingBoundarySample() {
  return absl::FailedPreconditionError("Missing boundary sample or liquid endpoint");
}

absl::StatusOr<CellIndex3D> OffsetCellIndex(CellIndex3D index, int axis, int amount) {
  std::array<std::int64_t, 3> values{index.x, index.y, index.z};
  values[axis] += amount;
  if (values[axis] < std::numeric_limits<std::int32_t>::min()
      || values[axis] > std::numeric_limits<std::int32_t>::max()) {
    return UnrepresentableGeometry();
  }
  return CellIndex3D{
      static_cast<std::int32_t>(values[0]),
      static_cast<std::int32_t>(values[1]),
      static_cast<std::int32_t>(values[2]),
  };
}
using SignedDistanceMap = std::map<CellLatticeKey, double>;
absl::StatusOr<double> LookupSignedDistance(const SignedDistanceMap& samples, CellIndex3D index) {
  const auto found = samples.find(CellKey(index));
  if (found == samples.end()) {
    return MissingBoundarySample();
  }
  return found->second;
}

// Clip in cell-width coordinates, retaining the interpolated physical liquid
// value. The same diagonal triangulation induces every tetrahedron boundary.
struct ClippingVertex {
  CartesianVector cell_width_position;
  double liquid_distance;
};
using ClippingPolygon = std::vector<ClippingVertex>;
using ClippingPolyhedron = std::vector<ClippingPolygon>;
struct ClippingPlane {
  CartesianVector normal;
  double plane_coordinate;
  bool uses_liquid_distance = false;
};
double EvaluateClipDistance(const ClippingVertex& vertex, const ClippingPlane& plane) {
  return plane.uses_liquid_distance ? vertex.liquid_distance
                                    : Dot(plane.normal, vertex.cell_width_position) - plane.plane_coordinate;
}
ClippingPolygon
ClipPolygon(const ClippingPolygon& polygon, const ClippingPlane& plane, ClippingPolygon* intersections = nullptr) {
  ClippingPolygon clipped;
  if (polygon.empty()) {
    return clipped;
  }
  ClippingVertex previous = polygon.back();
  double previous_distance = EvaluateClipDistance(previous, plane);
  for (const auto& current : polygon) {
    const double current_distance = EvaluateClipDistance(current, plane);
    if ((previous_distance <= 0) != (current_distance <= 0)) {
      // Normalize before subtraction so finite distances of opposite sign cannot overflow.
      const double magnitude = std::max(std::abs(previous_distance), std::abs(current_distance));
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
ClippingPolyhedron ClipPolyhedron(const ClippingPolyhedron& polyhedron, const ClippingPlane& plane) {
  ClippingPolyhedron result;
  ClippingPolygon intersections;
  for (const auto& polygon : polyhedron) {
    auto clipped = ClipPolygon(polygon, plane, &intersections);
    if (clipped.size() >= 3) {
      result.push_back(std::move(clipped));
    }
  }
  ClippingPolygon cap;
  for (const auto& vertex : intersections) {
    if (std::ranges::none_of(cap, [&](const auto& other) {
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
    const CartesianVector tangent = Sub(cap.front().cell_width_position, center);
    CartesianVector normal{};
    for (std::size_t i = 1; i < cap.size(); ++i) {
      normal = Cross(tangent, Sub(cap[i].cell_width_position, center));
      if (std::hypot(normal[0], normal[1], normal[2]) > 0) {
        break;
      }
    }
    const CartesianVector bitangent = Cross(normal, tangent);
    const double tangent_length = std::hypot(tangent[0], tangent[1], tangent[2]);
    const double bitangent_length = std::hypot(bitangent[0], bitangent[1], bitangent[2]);
    if (tangent_length > 0 && bitangent_length > 0) {
      const auto unit_tangent = Scale(tangent, 1 / tangent_length);
      const auto unit_bitangent = Scale(bitangent, 1 / bitangent_length);
      std::ranges::sort(cap, [&](const auto& a, const auto& b) {
        const auto delta_a = Sub(a.cell_width_position, center);
        const auto delta_b = Sub(b.cell_width_position, center);
        return std::atan2(Dot(delta_a, unit_bitangent), Dot(delta_a, unit_tangent))
            < std::atan2(Dot(delta_b, unit_bitangent), Dot(delta_b, unit_tangent));
      });
      result.push_back(std::move(cap));
    }
  }
  return result;
}
double ComputeClippedVolume(const ClippingPolyhedron& polyhedron) {
  CartesianVector center{};
  std::size_t count = 0;
  for (const auto& polygon : polyhedron) {
    for (const auto& vertex : polygon) {
      center = Add(center, vertex.cell_width_position);
      ++count;
    }
  }
  if (count == 0) {
    return 0;
  }
  center = Scale(center, 1.0 / count);
  double volume = 0;
  for (const auto& polygon : polyhedron) {
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
      volume += std::abs(Dot(
                    Sub(polygon[0].cell_width_position, center),
                    Cross(Sub(polygon[i].cell_width_position, center), Sub(polygon[i + 1].cell_width_position, center))
                ))
          / 6;
    }
  }
  return volume;
}
double ComputeClippedArea(const ClippingPolygon& polygon) {
  double area = 0;
  for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
    const auto cross = Cross(
        Sub(polygon[i].cell_width_position, polygon[0].cell_width_position),
        Sub(polygon[i + 1].cell_width_position, polygon[0].cell_width_position)
    );
    area += std::hypot(cross[0], cross[1], cross[2]) / 2;
  }
  return area;
}
// A polygon contained in the solid boundary is closed. Strictness changes area
// only for this coplanar case, and avoids counting the same face on both sides.
double ComputeOpenFaceArea(const ClippingPolygon& polygon, ClippingPlane plane) {
  if (std::ranges::none_of(polygon, [&](const auto& vertex) {
        return EvaluateClipDistance(vertex, plane) < 0;
      })) {
    return 0;
  }
  return ComputeClippedArea(ClipPolygon(polygon, plane));
}
struct ObstaclePlanes {
  bool present = false;
  bool slab = false;
  CartesianVector normal{};
  double low = 0;
  double high = 0;
  CartesianVector velocity{};
};
absl::StatusOr<ObstaclePlanes> ValidateObstacle(const LiquidObstacle3D& obstacle) {
  ObstaclePlanes result;
  if (const auto* halfspace = std::get_if<TranslatingHalfspace3D>(&obstacle)) {
    result = {
        true,
        false,
        CartesianComponents(halfspace->unit_normal),
        halfspace->start_plane_coordinate,
        halfspace->start_plane_coordinate,
        CartesianComponents(halfspace->wall_velocity),
    };
  } else if (const auto* slab = std::get_if<TranslatingSlab3D>(&obstacle)) {
    if (!IsPositiveFinite(slab->half_width)) {
      return absl::InvalidArgumentError("Slab half width must be positive finite");
    }
    result = {
        true,
        true,
        CartesianComponents(slab->unit_normal),
        slab->start_center_coordinate - slab->half_width,
        slab->start_center_coordinate + slab->half_width,
        CartesianComponents(slab->wall_velocity),
    };
    if (!std::isfinite(slab->start_center_coordinate)) {
      return absl::InvalidArgumentError("Invalid slab center");
    }
    if (!std::isfinite(result.low) || !std::isfinite(result.high) || result.low >= result.high) {
      return UnrepresentableGeometry();
    }
  }
  if (result.present
      && (!HasFiniteComponents(result.normal) || !HasFiniteComponents(result.velocity) || !std::isfinite(result.low)
          || std::abs(Dot(result.normal, result.normal) - 1.0) > 1e-12)) {
    return absl::InvalidArgumentError("Invalid unit-normal obstacle");
  }
  return result;
}
absl::StatusOr<ObstaclePlanes>
ObstacleInCellCoordinates(ObstaclePlanes obstacle, UniformMacGeometry3D geometry, CellIndex3D index) {
  if (!obstacle.present) {
    return obstacle;
  }
  const CartesianVector anchor{
      geometry.world_origin.x + geometry.cell_width * index.x,
      geometry.world_origin.y + geometry.cell_width * index.y,
      geometry.world_origin.z + geometry.cell_width * index.z,
  };
  const double origin_projection = Dot(obstacle.normal, anchor);
  obstacle.low = (obstacle.low - origin_projection) / geometry.cell_width;
  obstacle.high = (obstacle.high - origin_projection) / geometry.cell_width;
  if (!HasFiniteComponents(anchor) || !std::isfinite(obstacle.low) || !std::isfinite(obstacle.high)
      || (obstacle.slab && obstacle.low >= obstacle.high)) {
    return UnrepresentableGeometry();
  }
  return obstacle;
}

absl::StatusOr<double> ComputeCenteredCurvature(const SignedDistanceMap& samples, CellIndex3D center, double h) {
  const auto center_value = LookupSignedDistance(samples, center);
  if (!center_value.ok()) {
    return center_value.status();
  }
  CartesianVector gradient{};
  std::array<CartesianVector, 3> hessian{};
  for (int axis = 0; axis < 3; ++axis) {
    const auto low = OffsetCellIndex(center, axis, -1);
    const auto high = OffsetCellIndex(center, axis, 1);
    if (!low.ok() || !high.ok()) {
      return UnrepresentableGeometry();
    }
    const auto low_value = LookupSignedDistance(samples, *low);
    const auto high_value = LookupSignedDistance(samples, *high);
    if (!low_value.ok() || !high_value.ok()) {
      return MissingBoundarySample();
    }
    gradient[axis] = (*high_value - *low_value) / (2 * h);
    hessian[axis][axis] = ((*high_value - *center_value) + (*low_value - *center_value)) / h / h;
    for (int other = 0; other < axis; ++other) {
      double mixed = 0;
      for (int first : {-1, 1}) {
        for (int second : {-1, 1}) {
          const auto first_index = OffsetCellIndex(center, axis, first);
          if (!first_index.ok()) {
            return first_index.status();
          }
          const auto diagonal = OffsetCellIndex(*first_index, other, second);
          if (!diagonal.ok()) {
            return diagonal.status();
          }
          const auto value = LookupSignedDistance(samples, *diagonal);
          if (!value.ok()) {
            return value.status();
          }
          mixed += first * second * (*value / 4);
        }
      }
      hessian[axis][other] = mixed / h / h;
      hessian[other][axis] = hessian[axis][other];
    }
  }
  const double norm = std::hypot(gradient[0], gradient[1], gradient[2]);
  if (!HasFiniteComponents(gradient) || !std::isfinite(norm)) {
    return UnrepresentableGeometry();
  }
  if (norm < 1e-8) {
    return absl::FailedPreconditionError("Degenerate liquid curvature gradient");
  }
  // Contract the Hessian with a normalized gradient to avoid a cubic norm.
  const CartesianVector normal = Scale(gradient, 1 / norm);
  double curvature = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (!HasFiniteComponents(hessian[axis])) {
      return UnrepresentableGeometry();
    }
    curvature += (hessian[axis][axis] - normal[axis] * Dot(hessian[axis], normal)) / norm;
  }
  if (!std::isfinite(curvature)) {
    return UnrepresentableGeometry();
  }
  return curvature;
}
}  // namespace

absl::StatusOr<FrozenLiquidBoundaryGeometry3D> FreezeLiquidBoundaryGeometry3D(
    const LiquidBoundaryGeometryInputs3D& inputs
) {
  if (!IsPositiveFinite(inputs.step_duration) || !IsPositiveFinite(inputs.liquid_density)
      || !std::isfinite(inputs.surface_tension) || inputs.surface_tension < 0) {
    return absl::InvalidArgumentError("Invalid liquid material or step");
  }
  if (inputs.cell_indices.size() > 4096) {
    return absl::ResourceExhaustedError("Boundary cell capacity exceeded");
  }
  const auto obstacle = ValidateObstacle(inputs.obstacle);
  if (!obstacle.ok()) {
    return obstacle.status();
  }
  const auto grid_geometry = inputs.mac_grid.geometry();
  const double h = grid_geometry.cell_width;
  const double full_area = h * h;
  const double full_volume = full_area * h;
  if (!std::isnormal(full_area) || !std::isnormal(full_volume) || !std::isnormal(1 / inputs.liquid_density)) {
    return UnrepresentableGeometry();
  }
  FrozenLiquidBoundaryGeometry3D result(inputs.mac_grid);
  result.step_duration_ = inputs.step_duration;
  result.liquid_density_ = inputs.liquid_density;
  result.surface_tension_ = inputs.surface_tension;
  result.capillary_step_bound_ = std::numeric_limits<double>::infinity();
  SignedDistanceMap center_distance_samples;
  SignedDistanceMap corner_distance_samples;
  for (const auto& sample : inputs.center_distances) {
    const CellLatticeKey key = CellKey(sample.cell_index);
    if (!std::isfinite(sample.signed_distance)
        || (!center_distance_samples.empty() && key <= center_distance_samples.rbegin()->first)) {
      return absl::InvalidArgumentError("Center samples must be finite and strictly sorted");
    }
    center_distance_samples.emplace(key, sample.signed_distance);
  }
  for (const auto& sample : inputs.corner_distances) {
    const CellLatticeKey key = CellKey(sample.corner_index);
    if (!std::isfinite(sample.signed_distance)
        || (!corner_distance_samples.empty() && key <= corner_distance_samples.rbegin()->first)) {
      return absl::InvalidArgumentError("Corner samples must be finite and strictly sorted");
    }
    corner_distance_samples.emplace(key, sample.signed_distance);
  }
  result.center_distances_.assign(inputs.center_distances.begin(), inputs.center_distances.end());
  std::map<std::size_t, FaceIndex3D> incident_faces;
  for (std::size_t ordinal = 0; ordinal < inputs.cell_indices.size(); ++ordinal) {
    const auto index = inputs.cell_indices[ordinal];
    if (ordinal > 0 && CellKey(index) <= CellKey(inputs.cell_indices[ordinal - 1])) {
      return absl::InvalidArgumentError("Candidates must be strictly sorted");
    }
    const auto center_distance = LookupSignedDistance(center_distance_samples, index);
    if (!center_distance.ok()) {
      return center_distance.status();
    }
    std::array<ClippingVertex, 8> vertices;
    for (int corner = 0; corner < 8; ++corner) {
      CellIndex3D corner_index = index;
      CartesianVector cell_width_position{};
      for (int axis = 0; axis < 3; ++axis) {
        const int offset = (corner >> axis) & 1;
        const auto shifted = OffsetCellIndex(corner_index, axis, offset);
        if (!shifted.ok()) {
          return shifted.status();
        }
        corner_index = *shifted;
        cell_width_position[axis] = offset;
      }
      const auto distance = LookupSignedDistance(corner_distance_samples, corner_index);
      if (!distance.ok()) {
        return distance.status();
      }
      vertices[corner] = {cell_width_position, *distance};
    }
    const auto local_obstacle = ObstacleInCellCoordinates(*obstacle, grid_geometry, index);
    if (!local_obstacle.ok()) {
      return local_obstacle.status();
    }
    double low_side_liquid_volume_fraction = 0;
    double high_side_liquid_volume_fraction = 0;
    // All six permutations walk from (0,0,0) to (1,1,1). The resulting
    // coordinate-increasing face diagonals agree on both sides of every face.
    std::array<int, 3> permutation{0, 1, 2};
    do {
      const std::array<int, 4> corners_of_tet{0, 1 << permutation[0], (1 << permutation[0]) | (1 << permutation[1]), 7};
      const ClippingVertex a = vertices[corners_of_tet[0]];
      const ClippingVertex b = vertices[corners_of_tet[1]];
      const ClippingVertex c = vertices[corners_of_tet[2]];
      const ClippingVertex d = vertices[corners_of_tet[3]];
      auto liquid_polyhedron = ClipPolyhedron({{a, b, c}, {a, b, d}, {a, c, d}, {b, c, d}}, {{}, 0, true});
      if (!local_obstacle->present) {
        high_side_liquid_volume_fraction += ComputeClippedVolume(liquid_polyhedron);
      } else {
        high_side_liquid_volume_fraction += ComputeClippedVolume(
            ClipPolyhedron(liquid_polyhedron, {Scale(local_obstacle->normal, -1), -local_obstacle->high})
        );
        if (local_obstacle->slab) {
          low_side_liquid_volume_fraction +=
              ComputeClippedVolume(ClipPolyhedron(liquid_polyhedron, {local_obstacle->normal, local_obstacle->low}));
        }
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    if (low_side_liquid_volume_fraction > 0 && high_side_liquid_volume_fraction > 0 && *center_distance < 0) {
      return absl::FailedPreconditionError("Slab splits one pressure cell into disconnected liquid regions");
    }
    const double fraction = low_side_liquid_volume_fraction + high_side_liquid_volume_fraction;
    if (!std::isfinite(fraction) || fraction < 0 || fraction > 1 + 64 * std::numeric_limits<double>::epsilon()) {
      return UnrepresentableGeometry();
    }
    const double volume = std::min(1.0, fraction) * full_volume;
    if (!IsNormalOrZero(volume) || (fraction > 0 && volume == 0)) {
      return UnrepresentableGeometry();
    }
    if (*center_distance >= 0 || volume == 0) {
      continue;
    }
    result.cells_.push_back({index, volume, *center_distance, 0.0});
    for (int axis = 0; axis < 3; ++axis) {
      for (int side = 0; side < 2; ++side) {
        const auto shifted = OffsetCellIndex(index, axis, side);
        if (!shifted.ok()) {
          return shifted.status();
        }
        const FaceIndex3D face{static_cast<FaceNormalAxis>(axis), {shifted->x, shifted->y, shifted->z}};
        const auto slot = inputs.mac_grid.FaceSlot(face);
        if (!slot.ok()) {
          return slot.status();
        }
        incident_faces.emplace(*slot, face);
      }
    }
  }
  if (incident_faces.size() > 24576) {
    return absl::ResourceExhaustedError("Boundary face capacity exceeded");
  }
  for (const auto& [slot, face] : incident_faces) {
    const auto face_lattice_index = face.lattice_index;
    const auto local_obstacle = ObstacleInCellCoordinates(
        *obstacle,
        grid_geometry,
        {face_lattice_index.x, face_lattice_index.y, face_lattice_index.z}
    );
    if (!local_obstacle.ok()) {
      return local_obstacle.status();
    }
    std::array<int, 2> transverse{};
    int transverse_axis_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      if (axis != static_cast<int>(face.axis)) {
        transverse[transverse_axis_count++] = axis;
      }
    }
    std::array<ClippingVertex, 4> vertices{};
    vertices[1].cell_width_position[transverse[0]] = 1;
    vertices[2].cell_width_position[transverse[0]] = 1;
    vertices[2].cell_width_position[transverse[1]] = 1;
    vertices[3].cell_width_position[transverse[1]] = 1;
    double blocked_fraction = 0;
    double open_fraction = 0;
    for (int triangle = 0; triangle < 2; ++triangle) {
      const ClippingPolygon polygon{vertices[0], vertices[triangle + 1], vertices[triangle + 2]};
      if (!local_obstacle->present) {
        open_fraction += ComputeClippedArea(polygon);
        continue;
      }
      auto blocked = ClipPolygon(polygon, {local_obstacle->normal, local_obstacle->high});
      if (local_obstacle->slab) {
        blocked = ClipPolygon(blocked, {Scale(local_obstacle->normal, -1), -local_obstacle->low});
      }
      blocked_fraction += ComputeClippedArea(blocked);
      open_fraction += ComputeOpenFaceArea(polygon, {Scale(local_obstacle->normal, -1), -local_obstacle->high});
      if (local_obstacle->slab) {
        open_fraction += ComputeOpenFaceArea(polygon, {local_obstacle->normal, local_obstacle->low});
      }
    }
    // Compute both sides geometrically, then publish the exact subtraction used
    // by pressure assembly. A positive measure must not disappear in rounding.
    const double open_area = std::clamp(open_fraction, 0.0, 1.0) * full_area;
    const double blocked_area = full_area - open_area;
    if (!std::isfinite(open_fraction) || !std::isfinite(blocked_fraction)
        || open_fraction > 1 + 64 * std::numeric_limits<double>::epsilon() || !IsNormalOrZero(open_area)
        || !IsNormalOrZero(blocked_area) || (open_fraction > 0 && open_area == 0)
        || (blocked_fraction > 0 && blocked_area == 0)) {
      return UnrepresentableGeometry();
    }
    result.faces_.push_back({face, slot, open_area, blocked_area, obstacle->velocity[static_cast<int>(face.axis)]});
  }
  // Curvature and the capillary bound are interface-local. Interior or closed
  // cells neither consume a derivative halo nor impose a capillary restriction.
  std::map<CellLatticeKey, std::size_t> selected_cells;
  for (std::size_t ordinal = 0; ordinal < result.cells_.size(); ++ordinal) {
    selected_cells.emplace(CellKey(result.cells_[ordinal].cell_index), ordinal);
  }
  std::vector<bool> needs_curvature(result.cells_.size(), false);
  for (const auto& face : result.faces_) {
    if (face.open_area == 0 || inputs.surface_tension == 0) {
      continue;
    }
    const auto lattice = face.face_index.lattice_index;
    const CellIndex3D head{lattice.x, lattice.y, lattice.z};
    const auto tail = OffsetCellIndex(head, static_cast<int>(face.face_index.axis), -1);
    if (!tail.ok()) {
      return tail.status();
    }
    const auto head_cell = selected_cells.find(CellKey(head));
    const auto tail_cell = selected_cells.find(CellKey(*tail));
    const bool has_head = head_cell != selected_cells.end();
    const bool has_tail = tail_cell != selected_cells.end();
    if (has_head && has_tail) {
      continue;
    }
    const auto air_distance = LookupSignedDistance(center_distance_samples, has_head ? *tail : head);
    if (!air_distance.ok()) {
      return air_distance.status();
    }
    if (*air_distance >= 0) {
      needs_curvature[has_head ? head_cell->second : tail_cell->second] = true;
    }
  }
  for (std::size_t ordinal = 0; ordinal < result.cells_.size(); ++ordinal) {
    if (!needs_curvature[ordinal]) {
      continue;
    }
    auto& cell = result.cells_[ordinal];
    const auto curvature = ComputeCenteredCurvature(center_distance_samples, cell.cell_index, h);
    if (!curvature.ok()) {
      return curvature.status();
    }
    cell.curvature = *curvature;
  }
  if (std::ranges::any_of(needs_curvature, [](bool needed) {
        return needed;
      })) {
    const double capillary_square =
        inputs.liquid_density * full_volume / (4 * std::numbers::pi * inputs.surface_tension);
    if (!std::isnormal(capillary_square)) {
      return UnrepresentableGeometry();
    }
    result.capillary_step_bound_ = std::sqrt(capillary_square);
    if (inputs.step_duration > result.capillary_step_bound_) {
      return absl::FailedPreconditionError("Step exceeds capillary bound");
    }
  }
  return result;
}

absl::StatusOr<LiquidPressureProblem3D>
BuildLiquidPressureLinks3D(const FrozenLiquidBoundaryGeometry3D& geometry, const LiquidPressureFields3D& fields) {
  if (fields.cells.size() != geometry.cells_.size() || fields.predicted_face_velocities.size() != geometry.faces_.size()
      || !std::isfinite(fields.ambient_pressure)) {
    return absl::InvalidArgumentError("Invalid pressure field extents or ambient pressure");
  }
  LiquidPressureProblem3D result;
  std::map<CellLatticeKey, std::size_t> cell_ordinals;
  SignedDistanceMap distances;
  for (const auto& sample : geometry.center_distances_) {
    distances.emplace(CellKey(sample.cell_index), sample.signed_distance);
  }
  for (std::size_t i = 0; i < geometry.cells_.size(); ++i) {
    const auto& cell = geometry.cells_[i];
    const auto& field = fields.cells[i];
    if (!std::isfinite(field.target_divergence) || !std::isfinite(field.previous_pressure)
        || std::isnan(field.bulk_modulus) || field.bulk_modulus <= 0) {
      return absl::InvalidArgumentError("Invalid pressure cell field");
    }
    result.cells.push_back(
        {cell.cell_index, cell.fluid_volume, field.target_divergence, field.bulk_modulus, field.previous_pressure}
    );
    cell_ordinals.emplace(CellKey(cell.cell_index), i);
  }
  const double h = geometry.mac_grid_.geometry().cell_width;
  for (std::size_t i = 0; i < geometry.faces_.size(); ++i) {
    const auto& face = geometry.faces_[i];
    if (!std::isfinite(fields.predicted_face_velocities[i])) {
      return absl::InvalidArgumentError("Invalid predicted velocity");
    }
    MacPressureFace3D link{
        face.face_index,
        face.open_area,
        1 / geometry.liquid_density_,
        h,
        fields.predicted_face_velocities[i],
        face.wall_normal_velocity,
        SealedMacPressureLink3D{}
    };
    const auto index = face.face_index.lattice_index;
    const CellIndex3D head{index.x, index.y, index.z};
    const auto tail = OffsetCellIndex(head, static_cast<int>(face.face_index.axis), -1);
    if (!tail.ok()) {
      if (face.open_area > 0) {
        return tail.status();
      }
      result.faces.push_back(link);
      continue;
    }
    const auto head_found = cell_ordinals.find(CellKey(head));
    const auto tail_found = cell_ordinals.find(CellKey(*tail));
    const bool has_head = head_found != cell_ordinals.end();
    const bool has_tail = tail_found != cell_ordinals.end();
    if (has_head && has_tail) {
      link.pressure_link = InternalMacPressureLink3D{0};
    } else if (face.open_area > 0) {
      const auto air_distance = LookupSignedDistance(distances, has_head ? *tail : head);
      if (!air_distance.ok()) {
        return air_distance.status();
      }
      if (*air_distance < 0) {
        return MissingBoundarySample();
      }
      const auto& adjacent_liquid_cell = geometry.cells_[has_head ? head_found->second : tail_found->second];
      const double magnitude = std::max(-adjacent_liquid_cell.center_distance, *air_distance);
      const double liquid_scaled = -adjacent_liquid_cell.center_distance / magnitude;
      const double theta = liquid_scaled / (liquid_scaled + *air_distance / magnitude);
      if (theta < 0.05) {
        ++result.lower_theta_regularization_count;
      }
      link.pressure_gradient_distance = std::clamp(theta, 0.05, 1.0) * h;
      // A Dirichlet exterior pressure carries the entire capillary lift.
      // No internal jump or separate surface-force impulse accompanies it.
      const double pressure_lift = geometry.surface_tension_ * adjacent_liquid_cell.curvature;
      const double exterior_pressure = fields.ambient_pressure + pressure_lift;
      const bool lost_capillary_lift =
          geometry.surface_tension_ > 0 && adjacent_liquid_cell.curvature != 0 && pressure_lift == 0;
      if (lost_capillary_lift || !IsNormalOrZero(pressure_lift) || !std::isfinite(exterior_pressure)
          || !std::isnormal(link.pressure_gradient_distance)) {
        return UnrepresentableGeometry();
      }
      link.pressure_link = DirichletMacPressureLink3D{exterior_pressure};
    }
    result.faces.push_back(link);
  }
  return result;
}
}  // namespace fluidity
