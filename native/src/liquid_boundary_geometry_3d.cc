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
#include "liquid_boundary_backend.h"

namespace fluidity {
namespace {
using CellLatticeKey = std::tuple<std::int32_t, std::int32_t, std::int32_t>;
template <typename Index>
CellLatticeKey CellKey(Index index) {
  return {index.z, index.y, index.x};
}
using boundary_internal::CartesianComponents;
using boundary_internal::CartesianVector;
using boundary_internal::ClippingVertex;
using boundary_internal::Dot;
using boundary_internal::GeometryBackend;
using boundary_internal::GeometryInput;
using boundary_internal::GeometryMode;
using boundary_internal::HasFiniteComponents;
using boundary_internal::IsNormalOrZero;
using boundary_internal::IsPositiveFinite;
using boundary_internal::ObstaclePlanes;
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
// Stage only the centered derivative stencil, preserving absent unused halos.
absl::StatusOr<double>
ComputeCenteredCurvature(const SignedDistanceMap& samples, CellIndex3D center, double h, GeometryBackend backend) {
  GeometryInput query{};
  query.mode = GeometryMode::kCurvature;
  query.geometry.cell_width = h;
  for (int z = -1; z <= 1; ++z) {
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        if (x != 0 && y != 0 && z != 0) {
          continue;
        }
        CellIndex3D index = center;
        const int offsets[3]{x, y, z};
        for (int axis = 0; axis < 3; ++axis) {
          const auto shifted = OffsetCellIndex(index, axis, offsets[axis]);
          if (!shifted.ok()) {
            return shifted.status();
          }
          index = *shifted;
        }
        const auto value = LookupSignedDistance(samples, index);
        if (!value.ok()) {
          return value.status();
        }
        query.derivative_distances[(z + 1) * 9 + (y + 1) * 3 + x + 1] = *value;
      }
    }
  }
  const auto result = backend.compute(backend.context, query);
  if (!result.ok()) {
    return result.status();
  }
  return result->interface_curvature;
}
absl::StatusOr<boundary_internal::GeometryOutput> CpuGeometry(void*, const GeometryInput& query) {
  const auto result = boundary_internal::ComputeGeometry(query);
  if (!result.ok()) {
    return boundary_internal::ArithmeticStatus(result.status());
  }
  return *result;
}
}  // namespace

absl::StatusOr<FrozenLiquidBoundaryGeometry3D>
boundary_internal::FreezeGeometry(const LiquidBoundaryGeometryInputs3D& inputs, GeometryBackend backend) {
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
    GeometryInput query{};
    query.mode = GeometryMode::kCell;
    query.geometry = grid_geometry;
    query.index = index;
    query.obstacle = PackObstacle(inputs.obstacle);
    for (int corner = 0; corner < 8; ++corner) {
      query.corner_distances[corner] = vertices[corner].liquid_distance;
    }
    const auto volumes = backend.compute(backend.context, query);
    if (!volumes.ok()) {
      return volumes.status();
    }
    const double low_side_liquid_volume = volumes->low_side_liquid_volume;
    const double high_side_liquid_volume = volumes->high_side_liquid_volume;
    if (low_side_liquid_volume > 0 && high_side_liquid_volume > 0 && *center_distance < 0) {
      return absl::FailedPreconditionError("Slab splits one pressure cell into disconnected liquid regions");
    }
    const double unclamped_volume = low_side_liquid_volume + high_side_liquid_volume;
    if (!std::isfinite(unclamped_volume) || unclamped_volume < 0
        || unclamped_volume / full_volume > 1 + 64 * std::numeric_limits<double>::epsilon()) {
      return UnrepresentableGeometry();
    }
    const double volume = std::min(full_volume, unclamped_volume);
    if (!IsNormalOrZero(volume)) {
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
    GeometryInput query{};
    query.mode = GeometryMode::kFace;
    query.geometry = grid_geometry;
    query.index = {face_lattice_index.x, face_lattice_index.y, face_lattice_index.z};
    query.obstacle = PackObstacle(inputs.obstacle);
    query.axis = static_cast<int>(face.axis);
    const auto areas = backend.compute(backend.context, query);
    if (!areas.ok()) {
      return areas.status();
    }
    const double open_area = areas->open_face_area;
    const double blocked_area = areas->blocked_face_area;
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
    const auto curvature = ComputeCenteredCurvature(center_distance_samples, cell.cell_index, h, backend);
    if (!curvature.ok()) {
      return curvature.status();
    }
    cell.curvature = *curvature;
  }
  if (std::ranges::any_of(needs_curvature, [](bool needed) {
        return needed;
      })) {
    GeometryInput query{};
    query.mode = GeometryMode::kCapillary;
    query.geometry = grid_geometry;
    query.liquid_density = inputs.liquid_density;
    query.surface_tension = inputs.surface_tension;
    const auto capillary = backend.compute(backend.context, query);
    if (!capillary.ok()) {
      return capillary.status();
    }
    result.capillary_step_bound_ = capillary->capillary_step_bound;
    if (inputs.step_duration > result.capillary_step_bound_) {
      return absl::FailedPreconditionError("Step exceeds capillary bound");
    }
  }
  return result;
}

absl::StatusOr<LiquidPressureProblem3D> boundary_internal::BuildLinks(
    const FrozenLiquidBoundaryGeometry3D& geometry,
    const LiquidPressureFields3D& fields,
    GeometryBackend backend
) {
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
        SealedMacPressureLink3D{},
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
      GeometryInput query{};
      query.mode = GeometryMode::kLink;
      query.geometry = geometry.mac_grid_.geometry();
      query.surface_tension = geometry.surface_tension_;
      query.ambient_pressure = fields.ambient_pressure;
      query.interface_curvature = adjacent_liquid_cell.curvature;
      query.liquid_center_distance = adjacent_liquid_cell.center_distance;
      query.air_center_distance = *air_distance;
      const auto values = backend.compute(backend.context, query);
      if (!values.ok()) {
        return values.status();
      }
      result.lower_theta_regularization_count += values->lower_theta_regularized ? 1 : 0;
      link.pressure_gradient_distance = values->pressure_gradient_distance;
      const double exterior_pressure = values->exterior_pressure;
      link.pressure_link = DirichletMacPressureLink3D{exterior_pressure};
    }
    result.faces.push_back(link);
  }
  return result;
}
absl::StatusOr<FrozenLiquidBoundaryGeometry3D> FreezeLiquidBoundaryGeometry3D(
    const LiquidBoundaryGeometryInputs3D& inputs
) {
  return boundary_internal::FreezeGeometry(inputs, {CpuGeometry, nullptr});
}
absl::StatusOr<LiquidPressureProblem3D>
BuildLiquidPressureLinks3D(const FrozenLiquidBoundaryGeometry3D& geometry, const LiquidPressureFields3D& fields) {
  return boundary_internal::BuildLinks(geometry, fields, {CpuGeometry, nullptr});
}
}  // namespace fluidity
