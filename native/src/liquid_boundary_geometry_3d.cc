#include "fluidity/host/liquid_boundary_geometry_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "fluidity/host/liquid_air_components.h"
#include "fluidity/host/liquid_boundary_workspace_3d.h"
#include "fluidity/host/views.h"
#include "liquid_boundary_backend.h"

namespace fluidity {
namespace {
using CellLatticeKey = std::tuple<std::int32_t, std::int32_t, std::int32_t>;
template <typename Index>
CellLatticeKey CellKey(Index index) {
  return {index.z, index.y, index.x};
}
using boundary_internal::CartesianComponents;
using boundary_internal::Dot;
using boundary_internal::GeometryBackend;
using boundary_internal::GeometryInput;
using boundary_internal::GeometryMode;
using boundary_internal::HasFiniteComponents;
using boundary_internal::IsNormalOrZero;
using boundary_internal::IsPositiveFinite;
using boundary_internal::kGeometryQueryBatchCapacity3D;
using boundary_internal::ObstaclePlanes;
using boundary_internal::PackObstacle;
absl::Status UnrepresentableGeometry(bool bounded) {
  return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "Unrepresentable boundary geometry arithmetic");
}
absl::Status MissingBoundarySample(bool bounded) {
  return absl::Status(
      absl::StatusCode::kFailedPrecondition,
      bounded ? "" : "Missing boundary sample or liquid endpoint"
  );
}

absl::StatusOr<CellIndex3D> OffsetCellIndex(CellIndex3D index, int axis, int amount, bool bounded) {
  std::array<std::int64_t, 3> values{index.x, index.y, index.z};
  values[axis] += amount;
  if (values[axis] < std::numeric_limits<std::int32_t>::min()
      || values[axis] > std::numeric_limits<std::int32_t>::max()) {
    return UnrepresentableGeometry(bounded);
  }
  return CellIndex3D{
      static_cast<std::int32_t>(values[0]),
      static_cast<std::int32_t>(values[1]),
      static_cast<std::int32_t>(values[2]),
  };
}
// Samples arrive in strict signed lattice order. Binary lookup needs no tree nodes.
template <typename Sample>
auto SampleKey(const Sample& sample) {
  if constexpr (requires { sample.cell_index; }) {
    return CellKey(sample.cell_index);
  } else {
    return CellKey(sample.corner_index);
  }
}
template <typename Sample>
absl::Status ValidateSamples(std::span<const Sample> samples, bool bounded) {
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (!std::isfinite(samples[i].signed_distance) || (i > 0 && SampleKey(samples[i]) <= SampleKey(samples[i - 1]))) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          bounded ? "" : "Samples must be finite and strictly sorted"
      );
    }
  }
  return absl::OkStatus();
}
template <typename Sample>
absl::StatusOr<double> LookupSignedDistance(std::span<const Sample> samples, CellIndex3D index, bool bounded) {
  const auto key = CellKey(index);
  const auto found = std::ranges::lower_bound(samples, key, {}, [](const Sample& sample) {
    return SampleKey(sample);
  });
  if (found == samples.end() || SampleKey(*found) != key) {
    return MissingBoundarySample(bounded);
  }
  return found->signed_distance;
}
// All append extents are checked before the common core writes into its private bank.
template <typename T>
struct OutputRecords {
  std::span<T> storage;
  std::size_t count = 0;
  std::size_t size() const {
    return count;
  }
  auto begin() const {
    return storage.begin();
  }
  auto end() const {
    return storage.begin() + count;
  }
  T& operator[](std::size_t i) const {
    return storage[i];
  }
  void push_back(T value) {
    storage[count++] = value;
  }
  std::span<T> values() const {
    return storage.first(count);
  }
};
struct GeometryStorage {
  OutputRecords<FrozenLiquidCell3D> cells_;
  OutputRecords<FrozenLiquidFace3D> faces_;
  double step_duration_ = 0;
  double liquid_density_ = 0;
  double surface_tension_ = 0;
  double capillary_step_bound_ = 0;
};
std::size_t FindCell(std::span<const FrozenLiquidCell3D> cells, CellIndex3D index) {
  const auto key = CellKey(index);
  const auto found = std::ranges::lower_bound(cells, key, {}, [](const auto& cell) {
    return CellKey(cell.cell_index);
  });
  return found != cells.end() && CellKey(found->cell_index) == key ? static_cast<std::size_t>(found - cells.begin())
                                                                   : cells.size();
}
absl::StatusOr<ObstaclePlanes> ValidateObstacle(const LiquidObstacle3D& obstacle, double step_duration, bool bounded) {
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
      return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Slab half width must be positive finite");
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
      return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Invalid slab center");
    }
    if (!std::isfinite(result.low) || !std::isfinite(result.high) || result.low >= result.high) {
      return UnrepresentableGeometry(bounded);
    }
  }
  if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
    const auto status =
        boundary_internal::ArithmeticStatus(boundary_internal::CheckTankStep(*tank, step_duration), bounded);
    if (!status.ok()) {
      return status;
    }
  }
  if (result.present
      && (!HasFiniteComponents(result.normal) || !HasFiniteComponents(result.velocity) || !std::isfinite(result.low)
          || std::abs(Dot(result.normal, result.normal) - 1.0) > 1e-12)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Invalid unit-normal obstacle");
  }
  return result;
}
// Stage only the centered derivative stencil, preserving absent unused halos.
absl::StatusOr<bool> StageCenteredCurvature(
    std::span<const LiquidCenterDistance3D> samples,
    CellIndex3D center,
    double h,
    bool bounded,
    GeometryInput& query
) {
  query = {};
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
          const auto shifted = OffsetCellIndex(index, axis, offsets[axis], bounded);
          if (!shifted.ok()) {
            return shifted.status();
          }
          index = *shifted;
        }
        const auto value = LookupSignedDistance(samples, index, bounded);
        if (!value.ok()) {
          return value.status();
        }
        query.derivative_distances[(z + 1) * 9 + (y + 1) * 3 + x + 1] = *value;
      }
    }
  }
  return true;
}

// Staging is speculative within a batch, but errors are observed in canonical
// order. If staging fails, replay the completed prefix before returning it.
// A successful preparation returns true only when the ordinal has device arithmetic.
template <typename Prepare, typename Consume, typename Selected>
absl::Status RunGeometryQueries(
    GeometryBackend backend,
    std::size_t count,
    bool bounded,
    Prepare prepare,
    Consume consume,
    Selected selected
) {
  if (!backend.batch) {
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
      if (!selected(ordinal)) {
        continue;
      }
      GeometryInput query{};
      const auto staged = prepare(ordinal, 0, query);
      if (!staged.ok()) {
        return staged.status();
      }
      if (!*staged) {
        continue;
      }
      const auto output = backend.compute(backend.context, query);
      if (!output.ok()) {
        return output.status();
      }
      const auto consumed = consume(ordinal, 0, *output);
      if (!consumed.ok()) {
        return consumed;
      }
    }
    return absl::OkStatus();
  }
  auto& batch = *backend.batch;
  const std::size_t capacity = std::min({batch.queries.size(), batch.results.size(), kGeometryQueryBatchCapacity3D});
  if (capacity == 0) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  std::array<std::size_t, kGeometryQueryBatchCapacity3D> ordinals;
  std::size_t ordinal = 0;
  while (ordinal < count) {
    std::size_t submitted = 0;
    absl::Status staged;
    for (; ordinal < count && submitted < capacity; ++ordinal) {
      if (!selected(ordinal)) {
        continue;
      }
      const auto prepared = prepare(ordinal, submitted, batch.queries[submitted]);
      if (!prepared.ok()) {
        staged = prepared.status();
        break;
      }
      if (*prepared) {
        ordinals[submitted++] = ordinal;
      }
    }
    if (submitted != 0) {
      const auto dispatched = batch.compute(batch.context, submitted);
      if (!dispatched.ok()) {
        return dispatched;
      }
      for (std::size_t slot = 0; slot < submitted; ++slot) {
        const auto& output = batch.results[slot];
        if (output.error != boundary_internal::Error::kNone) {
          return boundary_internal::ArithmeticStatus(output.error, bounded);
        }
        const auto consumed = consume(ordinals[slot], slot, output.output);
        if (!consumed.ok()) {
          return consumed;
        }
      }
    }
    if (!staged.ok()) {
      return staged;
    }
  }
  return absl::OkStatus();
}
// A nonnull CPU-only context requests inline status codes. CUDA callbacks retain
// their own context and still execute every calculation through the same core.
absl::StatusOr<boundary_internal::GeometryOutput> CpuGeometry(void* context, const GeometryInput& query) {
  const auto result = boundary_internal::ComputeGeometry(query);
  if (!result.ok()) {
    return boundary_internal::ArithmeticStatus(result.status(), context != nullptr);
  }
  return *result;
}
absl::Status FreezeCore(
    const LiquidBoundaryGeometryViewInputs3D& inputs,
    GeometryBackend backend,
    GeometryStorage& result,
    std::span<bool> needs_curvature,
    bool bounded
) {
  if (!IsPositiveFinite(inputs.step_duration) || !IsPositiveFinite(inputs.liquid_density)
      || !std::isfinite(inputs.surface_tension) || inputs.surface_tension < 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Invalid liquid material or step");
  }
  const auto exact_curvature = inputs.exact_outward_interface_curvature;
  if (exact_curvature && !std::isfinite(*exact_curvature)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Invalid exact interface curvature");
  }
  if (inputs.cell_indices.size() > 4096) {
    return absl::Status(absl::StatusCode::kResourceExhausted, bounded ? "" : "Boundary cell capacity exceeded");
  }
  const auto obstacle = ValidateObstacle(inputs.obstacle, inputs.step_duration, bounded);
  if (!obstacle.ok()) {
    return obstacle.status();
  }
  const auto packed_obstacle = PackObstacle(inputs.obstacle);
  const auto grid_geometry = inputs.mac_grid.geometry();
  const double h = grid_geometry.cell_width;
  const double full_area = h * h;
  const double full_volume = full_area * h;
  if (!std::isnormal(full_area) || !std::isnormal(full_volume) || !std::isnormal(1 / inputs.liquid_density)) {
    return UnrepresentableGeometry(bounded);
  }

  result.step_duration_ = inputs.step_duration;
  result.liquid_density_ = inputs.liquid_density;
  result.surface_tension_ = inputs.surface_tension;
  result.capillary_step_bound_ = std::numeric_limits<double>::infinity();
  const auto center_distance_samples = inputs.center_distances;
  const auto corner_distance_samples = inputs.corner_distances;
  if (const auto status = ValidateSamples(center_distance_samples, bounded); !status.ok()) {
    return status;
  }
  if (const auto status = ValidateSamples(corner_distance_samples, bounded); !status.ok()) {
    return status;
  }
  std::array<double, kGeometryQueryBatchCapacity3D> center_distances;
  const auto all_ordinals = [](std::size_t) {
    return true;
  };
  const auto stage_cell = [&](std::size_t ordinal, std::size_t batch_slot, GeometryInput& query
                          ) -> absl::StatusOr<bool> {
    const auto index = inputs.cell_indices[ordinal];
    if (ordinal > 0 && CellKey(index) <= CellKey(inputs.cell_indices[ordinal - 1])) {
      return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Candidates must be strictly sorted");
    }
    const auto center_distance = LookupSignedDistance(center_distance_samples, index, bounded);
    if (!center_distance.ok()) {
      return center_distance.status();
    }
    center_distances[batch_slot] = *center_distance;
    query = {};
    query.mode = GeometryMode::kCell;
    query.geometry = grid_geometry;
    query.index = index;
    query.obstacle = packed_obstacle;
    for (int corner = 0; corner < 8; ++corner) {
      CellIndex3D corner_index = index;
      for (int axis = 0; axis < 3; ++axis) {
        const int offset = (corner >> axis) & 1;
        const auto shifted = OffsetCellIndex(corner_index, axis, offset, bounded);
        if (!shifted.ok()) {
          return shifted.status();
        }
        corner_index = *shifted;
      }
      const auto distance = LookupSignedDistance(corner_distance_samples, corner_index, bounded);
      if (!distance.ok()) {
        return distance.status();
      }
      query.corner_distances[corner] = *distance;
    }
    return true;
  };
  const auto consume_cell = [&](std::size_t ordinal,
                                std::size_t batch_slot,
                                const boundary_internal::GeometryOutput& volumes) -> absl::Status {
    const auto index = inputs.cell_indices[ordinal];
    const double center_distance = center_distances[batch_slot];
    const double low_side_liquid_volume = volumes.low_side_liquid_volume;
    const double high_side_liquid_volume = volumes.high_side_liquid_volume;
    if (low_side_liquid_volume > 0 && high_side_liquid_volume > 0 && center_distance < 0) {
      return absl::Status(
          absl::StatusCode::kFailedPrecondition,
          bounded ? "" : "Slab splits one pressure cell into disconnected liquid regions"
      );
    }
    const double unclamped_volume = low_side_liquid_volume + high_side_liquid_volume;
    if (!std::isfinite(unclamped_volume) || unclamped_volume < 0
        || unclamped_volume / full_volume > 1 + 64 * std::numeric_limits<double>::epsilon()) {
      return UnrepresentableGeometry(bounded);
    }
    const double volume = std::min(full_volume, unclamped_volume);
    if (!IsNormalOrZero(volume)) {
      return UnrepresentableGeometry(bounded);
    }
    if (center_distance >= 0 || volume == 0) {
      return absl::OkStatus();
    }
    result.cells_.push_back({index, volume, center_distance, 0.0});
    for (int axis = 0; axis < 3; ++axis) {
      for (int side = 0; side < 2; ++side) {
        const auto shifted = OffsetCellIndex(index, axis, side, bounded);
        if (!shifted.ok()) {
          return shifted.status();
        }
        const FaceIndex3D face{static_cast<FaceNormalAxis>(axis), {shifted->x, shifted->y, shifted->z}};
        const auto slot = inputs.mac_grid.FaceSlot(face);
        if (!slot.ok()) {
          return slot.status();
        }
        result.faces_.push_back({face, *slot, 0, 0, 0});
      }
    }
    return absl::OkStatus();
  };
  if (const auto status =
          RunGeometryQueries(backend, inputs.cell_indices.size(), bounded, stage_cell, consume_cell, all_ordinals);
      !status.ok()) {
    return status;
  }
  std::ranges::sort(result.faces_.values(), {}, &FrozenLiquidFace3D::face_slot);
  const auto duplicates = std::ranges::unique(result.faces_.values(), {}, &FrozenLiquidFace3D::face_slot);
  result.faces_.count = static_cast<std::size_t>(duplicates.begin() - result.faces_.values().begin());
  if (result.faces_.size() > 24576) {
    return absl::Status(absl::StatusCode::kResourceExhausted, bounded ? "" : "Boundary face capacity exceeded");
  }
  const auto stage_face = [&](std::size_t ordinal, std::size_t, GeometryInput& query) -> absl::StatusOr<bool> {
    const auto face = result.faces_[ordinal].face_index;
    const auto face_lattice_index = face.lattice_index;
    query = {};
    query.mode = GeometryMode::kFace;
    query.geometry = grid_geometry;
    query.index = {face_lattice_index.x, face_lattice_index.y, face_lattice_index.z};
    query.obstacle = packed_obstacle;
    query.axis = static_cast<int>(face.axis);
    return true;
  };
  const auto consume_face = [&](std::size_t ordinal, std::size_t, const boundary_internal::GeometryOutput& areas) {
    auto& frozen_face = result.faces_[ordinal];
    frozen_face.open_area = areas.open_face_area;
    frozen_face.blocked_area = areas.blocked_face_area;
    if (packed_obstacle.kind == boundary_internal::ObstacleKind::kOpenTank) {
      frozen_face.wall_normal_velocity = areas.piston_normal_velocity;
      frozen_face.prescribed_wall_flux = areas.piston_flux;
      frozen_face.is_low_x_piston_face = areas.is_low_x_piston_face;
    } else {
      frozen_face.wall_normal_velocity = obstacle->velocity[static_cast<int>(frozen_face.face_index.axis)];
    }
    return absl::OkStatus();
  };
  if (const auto status =
          RunGeometryQueries(backend, result.faces_.size(), bounded, stage_face, consume_face, all_ordinals);
      !status.ok()) {
    return status;
  }
  // Curvature and the capillary bound are interface-local. Interior or closed
  // cells neither consume a derivative halo nor impose a capillary restriction.
  needs_curvature = needs_curvature.first(result.cells_.size());
  std::ranges::fill(needs_curvature, false);
  for (const auto& face : result.faces_) {
    if (face.open_area == 0 || inputs.surface_tension == 0) {
      continue;
    }
    const auto lattice = face.face_index.lattice_index;
    const CellIndex3D head{lattice.x, lattice.y, lattice.z};
    const auto tail = OffsetCellIndex(head, static_cast<int>(face.face_index.axis), -1, bounded);
    if (!tail.ok()) {
      return tail.status();
    }
    const auto head_cell = FindCell(result.cells_.values(), head);
    const auto tail_cell = FindCell(result.cells_.values(), *tail);
    const bool has_head = head_cell != result.cells_.size();
    const bool has_tail = tail_cell != result.cells_.size();
    if (has_head && has_tail) {
      continue;
    }
    const auto air_distance = LookupSignedDistance(center_distance_samples, has_head ? *tail : head, bounded);
    if (!air_distance.ok()) {
      return air_distance.status();
    }
    if (*air_distance >= 0) {
      needs_curvature[has_head ? head_cell : tail_cell] = true;
    }
  }
  const bool has_capillary_interface = std::ranges::any_of(needs_curvature, [](bool needed) {
    return needed;
  });
  if (exact_curvature) {
    if (has_capillary_interface && *exact_curvature != 0) {
      const double capillary_pressure_lift = inputs.surface_tension * *exact_curvature;
      if (!std::isnormal(capillary_pressure_lift)) {
        return UnrepresentableGeometry(bounded);
      }
    }
    for (std::size_t ordinal = 0; ordinal < needs_curvature.size(); ++ordinal) {
      if (needs_curvature[ordinal]) {
        result.cells_[ordinal].curvature = *exact_curvature;
      }
    }
  } else {
    const auto stage_curvature = [&](std::size_t ordinal, std::size_t, GeometryInput& query) {
      return StageCenteredCurvature(center_distance_samples, result.cells_[ordinal].cell_index, h, bounded, query);
    };
    const auto consume_curvature = [&](std::size_t ordinal, std::size_t, const boundary_internal::GeometryOutput& output
                                   ) {
      result.cells_[ordinal].curvature = output.interface_curvature;
      return absl::OkStatus();
    };
    const auto needs_ordinal = [&](std::size_t ordinal) {
      return needs_curvature[ordinal];
    };
    if (const auto status = RunGeometryQueries(
            backend,
            result.cells_.size(),
            bounded,
            stage_curvature,
            consume_curvature,
            needs_ordinal
        );
        !status.ok()) {
      return status;
    }
  }
  if (has_capillary_interface) {
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
      return absl::Status(absl::StatusCode::kFailedPrecondition, bounded ? "" : "Step exceeds capillary bound");
    }
  }
  return absl::OkStatus();
}

struct LinkStorage {
  OutputRecords<MacPressureCell3D> cells;
  OutputRecords<MacPressureFace3D> faces;
  std::size_t lower_theta_regularization_count = 0;
};
absl::Status LinksCore(
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const LiquidPressureFields3D& fields,
    GeometryBackend backend,
    LinkStorage& result,
    bool bounded
) {
  const auto wetted_piston_face_fluxes = fields.wetted_piston_face_fluxes;
  const bool has_wetted_piston_face_fluxes = !wetted_piston_face_fluxes.empty();
  if (fields.cells.size() != geometry.cells.size() || fields.predicted_face_velocities.size() != geometry.faces.size()
      || (has_wetted_piston_face_fluxes && wetted_piston_face_fluxes.size() != geometry.faces.size())
      || fields.air_cell_pressures.size() > geometry.center_distances.size()
      || !std::isfinite(fields.ambient_pressure)) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        bounded ? "" : "Invalid pressure field extents or ambient pressure"
    );
  }
  const auto distances = geometry.center_distances;
  for (std::size_t ordinal = 0; ordinal < fields.air_cell_pressures.size(); ++ordinal) {
    const auto& pressure = fields.air_cell_pressures[ordinal];
    if (!std::isfinite(pressure.gas_gauge_pressure)
        || (ordinal > 0 && CellKey(fields.air_cell_pressures[ordinal - 1].cell_index) >= CellKey(pressure.cell_index)
        )) {
      return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Invalid air pressure records");
    }
    const auto distance = LookupSignedDistance(distances, pressure.cell_index, bounded);
    if (!distance.ok()) {
      return distance.status();
    }
    if (*distance < 0.0 || FindCell(geometry.cells, pressure.cell_index) != geometry.cells.size()) {
      return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Air pressure requires an air center");
    }
  }
  for (std::size_t i = 0; i < geometry.cells.size(); ++i) {
    const auto& cell = geometry.cells[i];
    const auto& field = fields.cells[i];
    if (!std::isfinite(field.target_divergence) || !std::isfinite(field.previous_pressure)
        || std::isnan(field.bulk_modulus) || field.bulk_modulus <= 0
        || (field.density && !IsPositiveFinite(*field.density))) {
      return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Invalid pressure cell field");
    }
    result.cells.push_back(
        {cell.cell_index, cell.fluid_volume, field.target_divergence, field.bulk_modulus, field.previous_pressure}
    );
  }
  const double h = geometry.mac_grid.geometry().cell_width;
  const auto stage_link = [&](std::size_t i, std::size_t, GeometryInput& query) -> absl::StatusOr<bool> {
    const auto& face = geometry.faces[i];
    if (!std::isfinite(fields.predicted_face_velocities[i])) {
      return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Invalid predicted velocity");
    }
    MacPressureFace3D link{
        face.face_index,
        face.open_area,
        1 / geometry.liquid_density,
        h,
        fields.predicted_face_velocities[i],
        face.wall_normal_velocity,
        SealedMacPressureLink3D{},
        face.prescribed_wall_flux,
    };
    const auto index = face.face_index.lattice_index;
    const CellIndex3D head{index.x, index.y, index.z};
    const auto tail = OffsetCellIndex(head, static_cast<int>(face.face_index.axis), -1, bounded);
    const auto head_found = FindCell(geometry.cells, head);
    const auto tail_found = tail.ok() ? FindCell(geometry.cells, *tail) : geometry.cells.size();
    const bool has_head = head_found != geometry.cells.size();
    const bool has_tail = tail_found != geometry.cells.size();
    if (has_wetted_piston_face_fluxes) {
      const bool piston_receiver = face.is_low_x_piston_face && face.face_index.axis == FaceNormalAxis::kX
          && face.open_area == 0 && has_head && !has_tail && face.prescribed_wall_flux.has_value();
      const auto signed_piston_flux = wetted_piston_face_fluxes[i];
      if (piston_receiver != signed_piston_flux.has_value()) {
        return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Invalid wetted piston incidence");
      }
      if (signed_piston_flux.has_value()) {
        const double wall_normal_velocity = face.wall_normal_velocity;
        const bool wrong_direction = (*signed_piston_flux > 0 && wall_normal_velocity <= 0)
            || (*signed_piston_flux < 0 && wall_normal_velocity >= 0);
        if (!std::isfinite(*signed_piston_flux) || wrong_direction) {
          return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "Invalid signed wetted piston flux");
        }
        link.prescribed_wall_flux = signed_piston_flux;
      }
    }
    if (!tail.ok()) {
      if (face.open_area > 0) {
        return tail.status();
      }
      result.faces.storage[i] = link;
      return false;
    }
    if (!has_head && !has_tail) {
      return MissingBoundarySample(bounded);
    }
    if (face.open_area > 0) {
      const auto head_density = has_head ? fields.cells[head_found].density : std::nullopt;
      const auto tail_density = has_tail ? fields.cells[tail_found].density : std::nullopt;
      if (head_density || tail_density) {
        const double adjacent_density = has_head ? head_density.value_or(geometry.liquid_density) : *tail_density;
        // Internal centers contribute two equal half-cell distances. At a
        // free surface only the liquid-side density enters the mobility.
        const double face_density = has_head && has_tail
            ? std::midpoint(adjacent_density, tail_density.value_or(geometry.liquid_density))
            : adjacent_density;
        link.inverse_density = 1.0 / face_density;
        if (!std::isnormal(link.inverse_density)) {
          return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "Pressure mobility is unrepresentable");
        }
      }
    }
    if (has_head && has_tail) {
      link.pressure_link = InternalMacPressureLink3D{0};
    } else if (face.open_area > 0) {
      const auto air_cell = has_head ? *tail : head;
      const auto air_distance = LookupSignedDistance(distances, air_cell, bounded);
      if (!air_distance.ok()) {
        return air_distance.status();
      }
      if (*air_distance < 0) {
        return MissingBoundarySample(bounded);
      }
      const auto& adjacent_liquid_cell = geometry.cells[has_head ? head_found : tail_found];
      query = {};
      query.mode = GeometryMode::kLink;
      query.geometry = geometry.mac_grid.geometry();
      query.surface_tension = geometry.surface_tension;
      const auto air_key = CellKey(air_cell);
      const auto pressure = std::ranges::lower_bound(fields.air_cell_pressures, air_key, {}, [](const auto& record) {
        return CellKey(record.cell_index);
      });
      query.ambient_pressure = pressure != fields.air_cell_pressures.end() && CellKey(pressure->cell_index) == air_key
          ? pressure->gas_gauge_pressure
          : fields.ambient_pressure;
      query.interface_curvature = adjacent_liquid_cell.curvature;
      query.liquid_center_distance = adjacent_liquid_cell.center_distance;
      query.air_center_distance = *air_distance;
      result.faces.storage[i] = link;
      return true;
    }
    result.faces.storage[i] = link;
    return false;
  };
  const auto consume_link = [&](std::size_t ordinal, std::size_t, const boundary_internal::GeometryOutput& values) {
    result.lower_theta_regularization_count += values.lower_theta_regularized ? 1 : 0;
    auto& link = result.faces.storage[ordinal];
    link.pressure_gradient_distance = values.pressure_gradient_distance;
    link.pressure_link = DirichletMacPressureLink3D{values.exterior_pressure};
    return absl::OkStatus();
  };
  const auto all_ordinals = [](std::size_t) {
    return true;
  };
  const auto status =
      RunGeometryQueries(backend, geometry.faces.size(), bounded, stage_link, consume_link, all_ordinals);
  if (!status.ok()) {
    return status;
  }
  result.faces.count = geometry.faces.size();
  return absl::OkStatus();
}
// Frozen borrows are public values: check ordering and enum/slot metadata before
// binary search or signed-axis indexing, even if a caller assembled the view.
absl::Status ValidateFrozenView(const FrozenLiquidBoundaryGeometryView3D& geometry) {
  const auto invalid = [] {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  };
  if (!IsPositiveFinite(geometry.step_duration) || !IsPositiveFinite(geometry.liquid_density)
      || !std::isfinite(geometry.surface_tension) || geometry.surface_tension < 0) {
    return invalid();
  }
  if (geometry.cells.size() > 4096 || geometry.faces.size() > 24576) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  const double cell_width = geometry.mac_grid.geometry().cell_width;
  const double full_face_area = cell_width * cell_width;
  const double full_cell_volume = full_face_area * cell_width;
  const double inverse_density = 1 / geometry.liquid_density;
  // Public borrows must preserve the physical representability checks from freeze.
  if (!std::isnormal(full_face_area) || !std::isnormal(full_cell_volume) || !std::isnormal(inverse_density)) {
    return UnrepresentableGeometry(true);
  }
  const auto samples_status = ValidateSamples(geometry.center_distances, true);
  if (!samples_status.ok()) {
    return samples_status;
  }
  for (std::size_t i = 0; i < geometry.cells.size(); ++i) {
    const auto& cell = geometry.cells[i];
    if (!IsPositiveFinite(cell.fluid_volume) || cell.fluid_volume > full_cell_volume
        || !std::isfinite(cell.center_distance) || cell.center_distance >= 0 || !std::isfinite(cell.curvature)) {
      return invalid();
    }
    if (i > 0 && CellKey(cell.cell_index) <= CellKey(geometry.cells[i - 1].cell_index)) {
      return invalid();
    }
  }
  for (std::size_t i = 0; i < geometry.faces.size(); ++i) {
    const auto& face = geometry.faces[i];
    const auto slot = geometry.mac_grid.FaceSlot(face.face_index);
    if (!slot.ok()) {
      return slot.status();
    }
    if (*slot != face.face_slot || (i > 0 && face.face_slot <= geometry.faces[i - 1].face_slot)) {
      return invalid();
    }
    if (!std::isfinite(face.open_area) || face.open_area < 0 || face.open_area > full_face_area
        || face.blocked_area != full_face_area - face.open_area || !std::isfinite(face.wall_normal_velocity)) {
      return invalid();
    }
    if (face.prescribed_wall_flux.has_value()
        && (!std::isfinite(*face.prescribed_wall_flux) || (face.blocked_area == 0 && *face.prescribed_wall_flux != 0)
        )) {
      return invalid();
    }
    const auto index = face.face_index.lattice_index;
    const CellIndex3D head{index.x, index.y, index.z};
    const auto tail = OffsetCellIndex(head, static_cast<int>(face.face_index.axis), -1, true);
    const bool has_head = FindCell(geometry.cells, head) != geometry.cells.size();
    const bool has_tail = tail.ok() && FindCell(geometry.cells, *tail) != geometry.cells.size();
    // A sealed face at INT32_MIN has no representable tail, but its selected head
    // still makes it incident. Without either endpoint it is an extraneous face.
    if (!has_head && !has_tail) {
      return MissingBoundarySample(true);
    }
    if (!tail.ok() && face.open_area > 0) {
      return tail.status();
    }
  }
  // The sorted face union must contain every incident owner, including closed
  // faces. Check membership without an allocating set or mutable scratch.
  for (const auto& cell : geometry.cells) {
    for (int axis = 0; axis < 3; ++axis) {
      for (int side = 0; side < 2; ++side) {
        const auto shifted = OffsetCellIndex(cell.cell_index, axis, side, true);
        if (!shifted.ok()) {
          return shifted.status();
        }
        const FaceIndex3D face{
            static_cast<FaceNormalAxis>(axis),
            {shifted->x, shifted->y, shifted->z},
        };
        const auto slot = geometry.mac_grid.FaceSlot(face);
        if (!slot.ok()) {
          return slot.status();
        }
        const auto found = std::ranges::lower_bound(geometry.faces, *slot, {}, &FrozenLiquidFace3D::face_slot);
        if (found == geometry.faces.end() || found->face_slot != *slot) {
          return MissingBoundarySample(true);
        }
      }
    }
  }
  return absl::OkStatus();
}
template <typename T>
void Resize(LiquidRecords3D<T>& records, std::size_t size) {
  const auto status = records.Resize(size);
  (void)status;
}
absl::StatusOr<double> MeasureGasFace(GeometryInput face, const std::array<double, 8>& gas_distances, int side) {
  using boundary_internal::CartesianVector;
  using boundary_internal::ClippingPolygon;
  using boundary_internal::ClippingVertex;
  using boundary_internal::ClipPolygon;
  using boundary_internal::ComputeClippedArea;
  using boundary_internal::ObstacleKind;
  using boundary_internal::TankClipBounds;
  const double width = face.geometry.cell_width;
  const bool tank = face.obstacle.kind == ObstacleKind::kOpenTank;
  TankClipBounds tank_bounds;
  if (tank) {
    const auto bounds = boundary_internal::LocalTankBounds(face);
    if (!bounds.ok()) {
      return boundary_internal::ArithmeticStatus(bounds.status(), true);
    }
    const auto opening = boundary_internal::ComputeTankFace(face, *bounds);
    if (!opening.ok()) {
      return boundary_internal::ArithmeticStatus(opening.status(), true);
    }
    if (opening->open_face_area == 0.0) {
      return 0.0;
    }
    tank_bounds = *bounds;
  }
  const auto obstacle = boundary_internal::LocalObstacle(face);
  if (!obstacle.ok()) {
    return boundary_internal::ArithmeticStatus(obstacle.status(), true);
  }
  std::array<int, 2> transverse{};
  std::size_t count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis != face.axis) {
      transverse[count++] = axis;
    }
  }
  constexpr int kFirst[]{0, 1, 1, 0};
  constexpr int kSecond[]{0, 0, 1, 1};
  std::array<ClippingVertex, 4> vertices{};
  for (std::size_t corner = 0; corner < vertices.size(); ++corner) {
    const int bits = (side << face.axis) | (kFirst[corner] << transverse[0]) | (kSecond[corner] << transverse[1]);
    vertices[corner].cell_width_position[transverse[0]] = kFirst[corner];
    vertices[corner].cell_width_position[transverse[1]] = kSecond[corner];
    vertices[corner].liquid_distance = gas_distances[bits];
  }
  double gas_area = 0.0;
  for (int triangle = 0; triangle < 2; ++triangle) {
    const ClippingPolygon polygon{vertices[0], vertices[triangle + 1], vertices[triangle + 2]};
    if (std::ranges::none_of(polygon, [](const ClippingVertex& vertex) {
          return vertex.liquid_distance < 0.0;
        })) {
      continue;
    }
    auto gas = ClipPolygon(polygon, {{}, 0.0, true});
    if (tank) {
      for (const int axis : transverse) {
        if (tank_bounds.low[axis] > 0.0) {
          CartesianVector normal{};
          normal[axis] = -1.0;
          gas = ClipPolygon(gas, {normal, -tank_bounds.low[axis], false, axis});
        }
        if (tank_bounds.high[axis] < 1.0) {
          CartesianVector normal{};
          normal[axis] = 1.0;
          gas = ClipPolygon(gas, {normal, tank_bounds.high[axis], false, axis});
        }
      }
    }
    if (obstacle->present) {
      gas_area += boundary_internal::ComputeOpenFaceArea(
          gas,
          {boundary_internal::Scale(obstacle->normal, -1.0), -obstacle->high},
          width
      );
      if (obstacle->slab) {
        gas_area += boundary_internal::ComputeOpenFaceArea(gas, {obstacle->normal, obstacle->low}, width);
      }
    } else {
      gas_area += ComputeClippedArea(gas, width);
    }
  }
  const double full_area = width * width;
  if (!IsNormalOrZero(gas_area) || gas_area / full_area > 1.0 + 64.0 * std::numeric_limits<double>::epsilon()) {
    return UnrepresentableGeometry(true);
  }
  return std::min(gas_area, full_area);
}

// Each clipped tetrahedron is convex on the selected side of the obstacle.
// A positive-area gas patch on a shared triangle joins two such pieces.
absl::Status CertifyGasCellConnectivity(GeometryInput query, bool low_slab_side) {
  using boundary_internal::CartesianVector;
  using boundary_internal::CellTetrahedronCorner;
  using boundary_internal::ClippingPolygon;
  using boundary_internal::ClippingVertex;
  using boundary_internal::ObstacleKind;
  using boundary_internal::TankClipBounds;
  constexpr std::size_t kTetrahedra = 6;
  const double width = query.geometry.cell_width;
  const bool tank = query.obstacle.kind == ObstacleKind::kOpenTank;
  TankClipBounds tank_bounds;
  if (tank) {
    const auto bounds = boundary_internal::LocalTankBounds(query);
    if (!bounds.ok()) {
      return boundary_internal::ArithmeticStatus(bounds.status(), true);
    }
    tank_bounds = *bounds;
  }
  const auto obstacle = boundary_internal::LocalObstacle(query);
  if (!obstacle.ok()) {
    return boundary_internal::ArithmeticStatus(obstacle.status(), true);
  }
  const auto clip_obstacle = [&](auto piece, auto clip) {
    if (tank) {
      for (int axis = 0; axis < 3; ++axis) {
        if (tank_bounds.low[axis] > 0.0) {
          CartesianVector normal{};
          normal[axis] = -1.0;
          piece = clip(piece, {normal, -tank_bounds.low[axis], false, axis});
        }
        if (tank_bounds.high[axis] < 1.0) {
          CartesianVector normal{};
          normal[axis] = 1.0;
          piece = clip(piece, {normal, tank_bounds.high[axis], false, axis});
        }
      }
    } else if (obstacle->present) {
      if (low_slab_side) {
        piece = clip(piece, {obstacle->normal, obstacle->low});
      } else {
        piece = clip(piece, {boundary_internal::Scale(obstacle->normal, -1.0), -obstacle->high});
      }
    }
    return piece;
  };
  const auto clip_polyhedron = [](auto piece, boundary_internal::ClippingPlane plane) {
    return boundary_internal::ClipPolyhedron(piece, plane);
  };
  const auto clip_polygon = [](auto piece, boundary_internal::ClippingPlane plane) {
    return boundary_internal::ClipPolygon(piece, plane);
  };
  std::array<ClippingVertex, 8> vertices{};
  for (std::size_t corner = 0; corner < vertices.size(); ++corner) {
    for (int axis = 0; axis < 3; ++axis) {
      vertices[corner].cell_width_position[axis] = static_cast<double>((corner >> axis) & 1);
    }
    vertices[corner].liquid_distance = query.corner_distances[corner];
  }
  const auto has_gas = [](const ClippingVertex& vertex) {
    return vertex.liquid_distance < 0.0;
  };
  std::array<bool, kTetrahedra> active{};
  std::size_t active_count = 0;
  for (std::size_t tetrahedron = 0; tetrahedron < kTetrahedra; ++tetrahedron) {
    const auto a = vertices[0];
    const auto b = vertices[CellTetrahedronCorner(tetrahedron, 1)];
    const auto c = vertices[CellTetrahedronCorner(tetrahedron, 2)];
    const auto d = vertices[7];
    if (!has_gas(a) && !has_gas(b) && !has_gas(c) && !has_gas(d)) {
      continue;
    }
    const auto gas = clip_polyhedron(
        boundary_internal::ClippingPolyhedron{{a, b, c}, {a, b, d}, {a, c, d}, {b, c, d}},
        {{}, 0.0, true}
    );
    const double volume = boundary_internal::ComputeClippedVolume(clip_obstacle(gas, clip_polyhedron), width);
    if (!IsNormalOrZero(volume)) {
      return UnrepresentableGeometry(true);
    }
    if (volume > 0.0) {
      active[tetrahedron] = true;
      ++active_count;
    }
  }
  if (active_count == 0) {
    return UnrepresentableGeometry(true);
  }
  std::array<std::size_t, kTetrahedra> queue{};
  std::array<bool, kTetrahedra> reached{};
  const auto seed = static_cast<std::size_t>(std::ranges::find(active, true) - active.begin());
  queue[0] = seed;
  reached[seed] = true;
  std::size_t queued = 1;
  for (std::size_t cursor = 0; cursor < queued; ++cursor) {
    const std::size_t current = queue[cursor];
    for (std::size_t neighbor = 0; neighbor < kTetrahedra; ++neighbor) {
      if (!active[neighbor] || reached[neighbor]) {
        continue;
      }
      for (std::size_t inner_corner = 1; inner_corner < 3; ++inner_corner) {
        const int shared_corner = CellTetrahedronCorner(current, inner_corner);
        if (shared_corner != CellTetrahedronCorner(neighbor, 1)
            && shared_corner != CellTetrahedronCorner(neighbor, 2)) {
          continue;
        }
        const ClippingPolygon triangle{vertices[0], vertices[shared_corner], vertices[7]};
        if (std::ranges::none_of(triangle, has_gas)) {
          break;
        }
        const auto gas = clip_polygon(triangle, {{}, 0.0, true});
        const double area = boundary_internal::ComputeClippedArea(clip_obstacle(gas, clip_polygon), width);
        if (!IsNormalOrZero(area)) {
          return UnrepresentableGeometry(true);
        }
        if (area > 0.0) {
          reached[neighbor] = true;
          queue[queued++] = neighbor;
        }
        break;
      }
    }
  }
  if (queued != active_count) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  return absl::OkStatus();
}

absl::StatusOr<LiquidAirCellMeasures3D> MeasureGasCell(const LiquidAirCellMeasureInputs3D& inputs) {
  const double width = inputs.geometry.cell_width;
  const double full_volume = width * width * width;
  if (!IsPositiveFinite(width) || !HasFiniteComponents(CartesianComponents(inputs.geometry.world_origin))
      || std::ranges::any_of(inputs.liquid_corner_distances, [](double distance) {
           return !std::isfinite(distance);
         })) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (!std::isnormal(full_volume) || !std::isnormal(width * width)) {
    return UnrepresentableGeometry(true);
  }
  const auto obstacle = ValidateObstacle(inputs.obstacle, 0.0, true);
  if (!obstacle.ok()) {
    return obstacle.status();
  }
  GeometryInput query{
      .mode = GeometryMode::kCell,
      .geometry = inputs.geometry,
      .index = inputs.cell_index,
      .obstacle = PackObstacle(inputs.obstacle),
  };
  std::array<double, 8> gas_distances{};
  for (std::size_t corner = 0; corner < gas_distances.size(); ++corner) {
    gas_distances[corner] = -inputs.liquid_corner_distances[corner];
    query.corner_distances[corner] = gas_distances[corner];
  }
  const auto volume = boundary_internal::ComputeGeometry(query);
  if (!volume.ok()) {
    return boundary_internal::ArithmeticStatus(volume.status(), true);
  }
  if (volume->low_side_liquid_volume > 0.0 && volume->high_side_liquid_volume > 0.0) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  const double gas_volume = volume->low_side_liquid_volume + volume->high_side_liquid_volume;
  if (!IsNormalOrZero(gas_volume) || gas_volume / full_volume > 1.0 + 64.0 * std::numeric_limits<double>::epsilon()) {
    return UnrepresentableGeometry(true);
  }
  if (gas_volume > 0.0) {
    const auto connected = CertifyGasCellConnectivity(query, volume->low_side_liquid_volume > 0.0);
    if (!connected.ok()) {
      return connected;
    }
  }
  LiquidAirCellMeasures3D result{.gas_volume = std::min(gas_volume, full_volume)};
  for (int axis = 0; axis < 3; ++axis) {
    for (int side = 0; side < 2; ++side) {
      const auto index = OffsetCellIndex(inputs.cell_index, axis, side, true);
      if (!index.ok()) {
        return index.status();
      }
      query.mode = GeometryMode::kFace;
      query.index = *index;
      query.axis = axis;
      const auto area = MeasureGasFace(query, gas_distances, side);
      if (!area.ok()) {
        return area.status();
      }
      result.gas_face_areas[2 * axis + side] = *area;
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<LiquidAirCellMeasures3D> MeasureLiquidAirCell3D(const LiquidAirCellMeasureInputs3D& inputs) {
  return MeasureGasCell(inputs);
}
absl::StatusOr<std::size_t> BuildLiquidAirDomain3D(
    UniformMacGeometry3D geometry,
    const LiquidObstacle3D& obstacle,
    std::span<const CellIndex3D> sampled_cells,
    std::span<const LiquidCornerDistance3D> corners,
    std::span<const FaceIndex3D> atmosphere_outlets,
    std::span<LiquidAirVolumeCell3D> gas_cells,
    LiquidAirExterior3D exterior
) {
  const auto invalid = [] {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  };
  const bool certify_exterior = exterior == LiquidAirExterior3D::kUnobstructedBoundingBox;
  if (exterior > LiquidAirExterior3D::kUnobstructedBoundingBox
      || (certify_exterior && (!std::holds_alternative<std::monostate>(obstacle) || !atmosphere_outlets.empty()))) {
    return invalid();
  }
  const auto output_bytes = std::as_bytes(gas_cells);
  for (const auto input_bytes : {
           std::as_bytes(sampled_cells),
           std::as_bytes(corners),
           std::as_bytes(atmosphere_outlets),
       }) {
    if (ByteViewsOverlap(output_bytes, input_bytes)) {
      return invalid();
    }
  }
  if (ByteViewsOverlap(output_bytes, std::as_bytes(std::span(&obstacle, 1)))) {
    return invalid();
  }
  if (!IsPositiveFinite(geometry.cell_width) || !HasFiniteComponents(CartesianComponents(geometry.world_origin))) {
    return invalid();
  }
  const auto obstacle_status = ValidateObstacle(obstacle, 0.0, true);
  if (!obstacle_status.ok()) {
    return obstacle_status.status();
  }
  const auto samples_status = ValidateSamples(corners, true);
  if (!samples_status.ok()) {
    return samples_status;
  }
  const auto face_key = [](FaceIndex3D face) {
    return std::tuple{face.axis, face.lattice_index.z, face.lattice_index.y, face.lattice_index.x};
  };
  for (std::size_t outlet = 0; outlet < atmosphere_outlets.size(); ++outlet) {
    const auto& face = atmosphere_outlets[outlet];
    if (face.axis > FaceNormalAxis::kZ || (outlet > 0 && face_key(atmosphere_outlets[outlet - 1]) >= face_key(face))) {
      return invalid();
    }
  }
  for (std::size_t cell = 1; cell < sampled_cells.size(); ++cell) {
    if (CellKey(sampled_cells[cell - 1]) >= CellKey(sampled_cells[cell])) {
      return invalid();
    }
  }

  std::array<std::int32_t, 3> low{};
  std::array<std::int32_t, 3> high{};
  if (certify_exterior && !sampled_cells.empty()) {
    const auto& first = sampled_cells.front();
    low = {first.x, first.y, first.z};
    high = low;
    for (const auto cell : sampled_cells) {
      const std::array coordinates{cell.x, cell.y, cell.z};
      for (std::size_t axis = 0; axis < coordinates.size(); ++axis) {
        low[axis] = std::min(low[axis], coordinates[axis]);
        high[axis] = std::max(high[axis], coordinates[axis]);
      }
    }
  }

  std::size_t gas_count = 0;
  std::size_t used_outlets = 0;
  for (const auto cell : sampled_cells) {
    LiquidAirCellMeasureInputs3D inputs{
        .geometry = geometry,
        .cell_index = cell,
        .obstacle = obstacle,
        .liquid_corner_distances = {},
    };
    for (std::size_t corner = 0; corner < inputs.liquid_corner_distances.size(); ++corner) {
      CellIndex3D corner_index = cell;
      for (int axis = 0; axis < 3; ++axis) {
        const auto shifted = OffsetCellIndex(corner_index, axis, static_cast<int>((corner >> axis) & 1), true);
        if (!shifted.ok()) {
          return shifted.status();
        }
        corner_index = *shifted;
      }
      const auto distance = LookupSignedDistance(corners, corner_index, true);
      if (!distance.ok()) {
        return distance.status();
      }
      inputs.liquid_corner_distances[corner] = *distance;
    }
    const auto measures = MeasureLiquidAirCell3D(inputs);
    if (!measures.ok()) {
      return measures.status();
    }
    if (measures->gas_volume == 0.0) {
      continue;
    }
    if (gas_count == gas_cells.size()) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    LiquidAirVolumeCell3D record{.cell_index = cell, .gas_volume = measures->gas_volume};
    const std::array coordinates{cell.x, cell.y, cell.z};
    for (int axis = 0; axis < 3; ++axis) {
      for (int side = 0; side < 2; ++side) {
        const auto passage = static_cast<std::size_t>(2 * axis + side);
        if (measures->gas_face_areas[passage] == 0.0) {
          continue;
        }
        record.passages[passage] = LiquidAirPassage3D::kOpen;
        if (certify_exterior) {
          const auto normal_axis = static_cast<std::size_t>(axis);
          const auto plane = side == 0 ? low[normal_axis] : high[normal_axis];
          if (coordinates[normal_axis] == plane) {
            record.passages[passage] = LiquidAirPassage3D::kAtmosphere;
          }
          continue;
        }
        if (atmosphere_outlets.empty()) {
          continue;
        }
        const auto face = OffsetCellIndex(cell, axis, side, true);
        if (!face.ok()) {
          return face.status();
        }
        const auto key = face_key({static_cast<FaceNormalAxis>(axis), {face->x, face->y, face->z}});
        if (!std::ranges::binary_search(atmosphere_outlets, key, {}, face_key)) {
          continue;
        }
        const auto neighbor = OffsetCellIndex(cell, axis, side == 0 ? -1 : 1, true);
        if (neighbor.ok() && std::ranges::binary_search(sampled_cells, CellKey(*neighbor), {}, CellKey<CellIndex3D>)) {
          return invalid();
        }
        record.passages[passage] = LiquidAirPassage3D::kAtmosphere;
        ++used_outlets;
      }
    }
    gas_cells[gas_count++] = record;
  }
  if (used_outlets != atmosphere_outlets.size()) {
    return invalid();
  }
  return gas_count;
}

absl::StatusOr<FrozenLiquidBoundaryGeometry3D>
boundary_internal::FreezeGeometry(const LiquidBoundaryGeometryInputs3D& inputs, GeometryBackend backend) {
  if (inputs.cell_indices.size() > 4096) {
    return absl::ResourceExhaustedError("Boundary cell capacity exceeded");
  }
  FrozenLiquidBoundaryGeometry3D result(inputs.mac_grid);
  result.cells_.resize(inputs.cell_indices.size());
  result.faces_.resize(6 * inputs.cell_indices.size());
  auto needs_curvature = LiquidRecords3D<bool>::Create(inputs.cell_indices.size());
  if (!needs_curvature.ok()) {
    return needs_curvature.status();
  }
  Resize(*needs_curvature, inputs.cell_indices.size());
  GeometryStorage storage{{result.cells_}, {result.faces_}};
  const LiquidBoundaryGeometryViewInputs3D borrowed_inputs{
      .mac_grid = inputs.mac_grid.view(),
      .cell_indices = inputs.cell_indices,
      .center_distances = inputs.center_distances,
      .corner_distances = inputs.corner_distances,
      .obstacle = inputs.obstacle,
      .step_duration = inputs.step_duration,
      .liquid_density = inputs.liquid_density,
      .surface_tension = inputs.surface_tension,
      .exact_outward_interface_curvature = inputs.exact_outward_interface_curvature,
  };
  const auto status = FreezeCore(borrowed_inputs, backend, storage, needs_curvature->values(), false);
  if (!status.ok()) {
    return status;
  }
  result.cells_.resize(storage.cells_.size());
  result.faces_.resize(storage.faces_.size());
  result.center_distances_.assign(inputs.center_distances.begin(), inputs.center_distances.end());
  result.step_duration_ = storage.step_duration_;
  result.liquid_density_ = storage.liquid_density_;
  result.surface_tension_ = storage.surface_tension_;
  result.capillary_step_bound_ = storage.capillary_step_bound_;
  return result;
}
absl::StatusOr<LiquidPressureProblem3D> boundary_internal::BuildLinks(
    const FrozenLiquidBoundaryGeometry3D& geometry,
    const LiquidPressureFields3D& fields,
    GeometryBackend backend
) {
  LiquidPressureProblem3D result;
  result.cells.resize(geometry.cells_.size());
  result.faces.resize(geometry.faces_.size());
  FrozenLiquidBoundaryGeometryView3D storage{
      geometry.mac_grid_.view(),
      geometry.center_distances_,
      geometry.cells_,
      geometry.faces_,
      geometry.step_duration_,
      geometry.liquid_density_,
      geometry.surface_tension_,
      geometry.capillary_step_bound_,
  };
  LinkStorage output{{result.cells}, {result.faces}};
  const auto status = LinksCore(storage, fields, backend, output, false);
  if (!status.ok()) {
    return status;
  }
  result.lower_theta_regularization_count = output.lower_theta_regularization_count;
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
absl::StatusOr<FrozenLiquidBoundaryGeometryView3D> LiquidBoundaryWorkspace3D::Freeze(
    const LiquidBoundaryGeometryViewInputs3D& inputs
) {
  return FreezeWithBackend(inputs, {CpuGeometry, this});
}
absl::StatusOr<FrozenLiquidBoundaryGeometryView3D> LiquidBoundaryWorkspace3D::FreezeWithBackend(
    const LiquidBoundaryGeometryViewInputs3D& inputs,
    boundary_internal::GeometryBackend backend
) {
  if (inputs.cell_indices.size() > capacities_.cells || inputs.center_distances.size() > 27 * capacities_.cells) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  auto& bank = geometry_[1 - geometry_bank_];
  Resize(bank.centers, inputs.center_distances.size());
  Resize(bank.cells, inputs.cell_indices.size());
  Resize(bank.faces, 6 * inputs.cell_indices.size());
  Resize(bank.needs_curvature, inputs.cell_indices.size());
  RecordHighWater();
  GeometryStorage storage{{bank.cells.values()}, {bank.faces.values()}};
  const auto status = FreezeCore(inputs, backend, storage, bank.needs_curvature.values(), true);
  if (!status.ok()) {
    return status;
  }
  if (storage.faces_.size() > capacities_.faces) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  std::ranges::copy(inputs.center_distances, bank.centers.values().begin());
  Resize(bank.cells, storage.cells_.size());
  Resize(bank.faces, storage.faces_.size());
  geometry_bank_ = 1 - geometry_bank_;
  return FrozenLiquidBoundaryGeometryView3D{
      inputs.mac_grid,
      bank.centers.values(),
      bank.cells.values(),
      bank.faces.values(),
      storage.step_duration_,
      storage.liquid_density_,
      storage.surface_tension_,
      storage.capillary_step_bound_,
  };
}
absl::StatusOr<LiquidPressureProblemView3D> LiquidBoundaryWorkspace3D::BuildLinks(
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const LiquidPressureFields3D& fields
) {
  return BuildLinksWithBackend(geometry, fields, {CpuGeometry, this});
}
absl::StatusOr<LiquidPressureProblemView3D> LiquidBoundaryWorkspace3D::BuildLinksWithBackend(
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const LiquidPressureFields3D& fields,
    boundary_internal::GeometryBackend backend
) {
  if (geometry.cells.size() > capacities_.cells || geometry.faces.size() > capacities_.faces) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  const auto view_status = ValidateFrozenView(geometry);
  if (!view_status.ok()) {
    return view_status;
  }
  auto& bank = links_[1 - link_bank_];
  for (const auto override_bytes :
       {std::as_bytes(fields.wetted_piston_face_fluxes), std::as_bytes(fields.air_cell_pressures)}) {
    const auto reserved_cell_bytes = std::as_bytes(std::span(bank.cells.values().data(), bank.cells.capacity()));
    const auto reserved_face_bytes = std::as_bytes(std::span(bank.faces.values().data(), bank.faces.capacity()));
    if (ByteViewsOverlap(override_bytes, reserved_cell_bytes)
        || ByteViewsOverlap(override_bytes, reserved_face_bytes)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  Resize(bank.cells, geometry.cells.size());
  Resize(bank.faces, geometry.faces.size());
  RecordHighWater();
  LinkStorage output{{bank.cells.values()}, {bank.faces.values()}};
  const auto status = LinksCore(geometry, fields, backend, output, true);
  if (!status.ok()) {
    return status;
  }
  link_bank_ = 1 - link_bank_;
  return LiquidPressureProblemView3D{bank.cells.values(), bank.faces.values(), output.lower_theta_regularization_count};
}
}  // namespace fluidity
