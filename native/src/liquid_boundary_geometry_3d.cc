#include "fluidity/host/liquid_boundary_geometry_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
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
struct FloatingBoxContext {
  const AxisAlignedFloatingBoxGeometry3D& box;
  bool bounded;
};

absl::Status ValidateFloatingBoxInputs(
    const AxisAlignedFloatingBoxGeometry3D& box,
    UniformMacGeometry3D geometry,
    const LiquidObstacle3D& obstacle,
    std::span<const LiquidCenterDistance3D> center_distances,
    bool bounded
) {
  const auto lower = CartesianComponents(box.world_low);
  const auto upper = CartesianComponents(box.world_high);
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle);
  if ((tank == nullptr && !std::holds_alternative<std::monostate>(obstacle)) || !HasFiniteComponents(lower)
      || !HasFiniteComponents(upper) || !std::isfinite(box.liquid_height) || lower[0] >= upper[0]
      || lower[1] >= upper[1] || lower[2] >= upper[2]) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        bounded ? "" : "invalid finite box or conflicting obstacle"
    );
  }
  if (tank != nullptr) {
    const auto tank_status = boundary_internal::CheckTankStep(*tank, 0.0);
    if (tank_status != boundary_internal::Error::kNone) {
      return boundary_internal::ArithmeticStatus(tank_status, bounded);
    }
    if (tank->piston_velocity_x != 0.0 || lower[0] <= tank->start_low_x || upper[0] >= tank->high_x
        || lower[1] <= tank->floor_y || lower[2] <= tank->low_z || upper[2] >= tank->high_z) {
      return absl::Status(
          absl::StatusCode::kFailedPrecondition,
          bounded ? "" : "finite box requires strict clearance inside a stationary tank"
      );
    }
  }
  for (const auto& sample : center_distances) {
    const double y_index = static_cast<double>(sample.cell_index.y) + 0.5;
    const auto expected_distance = boundary_internal::HorizontalPlaneLatticeDistance(
        box.liquid_height,
        {geometry.world_origin.y, y_index, geometry.cell_width}
    );
    if (!expected_distance.ok()) {
      return boundary_internal::ArithmeticStatus(expected_distance.status(), bounded);
    }
    if (sample.signed_distance != *expected_distance) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          bounded ? "" : "finite box requires horizontal liquid distance samples"
      );
    }
  }
  return absl::OkStatus();
}
// Decompose the complement into disjoint low/high strips in axis order.
// This retains a small surviving region without subtracting nearly equal volumes.
absl::StatusOr<double> BoxComplementMeasure(
    boundary_internal::TankClipBounds outer,
    const boundary_internal::TankClipBounds& solid,
    double cell_width,
    int excluded_axis,
    bool reject_split,
    bool bounded
) {
  const auto domain = outer;
  double measure = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis == excluded_axis) {
      continue;
    }
    const double cut_low = std::max(outer.low[axis], solid.low[axis]);
    const double cut_high = std::min(outer.high[axis], solid.high[axis]);
    if (cut_low >= cut_high) {
      return measure + boundary_internal::TankBoxMeasure(outer, cell_width, excluded_axis);
    }
    if (reject_split && cut_low > outer.low[axis] && cut_high < outer.high[axis]) {
      bool spans_transverse_axes = true;
      for (int transverse = 0; transverse < 3; ++transverse) {
        if (transverse != axis) {
          spans_transverse_axes &=
              solid.low[transverse] <= domain.low[transverse] && solid.high[transverse] >= domain.high[transverse];
        }
      }
      if (spans_transverse_axes) {
        return absl::Status(
            absl::StatusCode::kFailedPrecondition,
            bounded ? "" : "finite box splits a wet pressure cell"
        );
      }
    }
    auto low_strip = outer;
    low_strip.high[axis] = cut_low;
    auto high_strip = outer;
    high_strip.low[axis] = cut_high;
    measure += boundary_internal::TankBoxMeasure(low_strip, cell_width, excluded_axis);
    measure += boundary_internal::TankBoxMeasure(high_strip, cell_width, excluded_axis);
    outer.low[axis] = cut_low;
    outer.high[axis] = cut_high;
  }
  return measure;
}

absl::StatusOr<boundary_internal::GeometryOutput>
ComputeFloatingBoxGeometry(void* context, const GeometryInput& query) {
  auto& request = *static_cast<FloatingBoxContext*>(context);
  if (query.mode != GeometryMode::kCell && query.mode != GeometryMode::kFace) {
    return CpuGeometry(request.bounded ? context : nullptr, query);
  }
  const auto& box = request.box;
  const auto lower = CartesianComponents(box.world_low);
  const auto upper = CartesianComponents(box.world_high);
  const double cell_width = query.geometry.cell_width;
  const bool face_query = query.mode == GeometryMode::kFace;
  const int excluded_axis = face_query ? query.axis : -1;
  boundary_internal::GeometryOutput output{};
  boundary_internal::TankClipBounds outer;
  const bool has_tank = query.obstacle.kind == boundary_internal::ObstacleKind::kOpenTank;
  if (has_tank) {
    const auto bounds = boundary_internal::LocalTankBounds(query);
    if (!bounds.ok()) {
      return boundary_internal::ArithmeticStatus(bounds.status(), request.bounded);
    }
    outer = *bounds;
    if (face_query) {
      const auto tank_face = boundary_internal::ComputeTankFace(query, outer);
      if (!tank_face.ok()) {
        return boundary_internal::ArithmeticStatus(tank_face.status(), request.bounded);
      }
      output = *tank_face;
      if (output.open_face_area == 0.0) {
        return output;
      }
    }
  }
  if (face_query) {
    const auto lattice = boundary_internal::TankAxis(query, excluded_axis);
    const auto low_sign = boundary_internal::TankPlaneOffsetSign(lower[excluded_axis], lattice);
    const auto high_sign = boundary_internal::TankPlaneOffsetSign(upper[excluded_axis], lattice);
    if (!low_sign.ok() || !high_sign.ok()) {
      return UnrepresentableGeometry(request.bounded);
    }
    if (*low_sign > 0 || *high_sign < 0) {
      if (!has_tank) {
        output.open_face_area = cell_width * cell_width;
      }
      return output;
    }
  } else if (!boundary_internal::TankCellHasResolvedWorldExtent(query)) {
    return UnrepresentableGeometry(request.bounded);
  }

  boundary_internal::TankClipBounds solid;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis == excluded_axis) {
      continue;
    }
    const auto lattice = boundary_internal::TankAxis(query, axis);
    const auto low = boundary_internal::TankClipCoordinate(lower[axis], lattice);
    const auto high = boundary_internal::TankClipCoordinate(upper[axis], lattice);
    if (!low.ok() || !high.ok() || (*low == *high && *low > 0.0 && *high < 1.0)) {
      return UnrepresentableGeometry(request.bounded);
    }
    solid.low[axis] = *low;
    solid.high[axis] = *high;
  }
  if (!face_query) {
    const auto vertical_lattice = boundary_internal::TankAxis(query, 1);
    const auto liquid_top = boundary_internal::TankClipCoordinate(box.liquid_height, vertical_lattice);
    if (!liquid_top.ok()) {
      return UnrepresentableGeometry(request.bounded);
    }
    for (int corner = 0; corner < 8; ++corner) {
      const double y_index = vertical_lattice.lattice_index + ((corner >> 1) & 1);
      const auto expected_distance = boundary_internal::HorizontalPlaneLatticeDistance(
          box.liquid_height,
          {vertical_lattice.world_origin, y_index, cell_width}
      );
      if (!expected_distance.ok()) {
        return boundary_internal::ArithmeticStatus(expected_distance.status(), request.bounded);
      }
      if (query.corner_distances[corner] != *expected_distance) {
        return absl::Status(
            absl::StatusCode::kInvalidArgument,
            request.bounded ? "" : "finite box requires horizontal liquid distance samples"
        );
      }
    }
    outer.high[1] = std::min(outer.high[1], *liquid_top);
    if (outer.high[1] <= outer.low[1]) {
      return output;
    }
  }
  const auto open_measure = BoxComplementMeasure(outer, solid, cell_width, excluded_axis, !face_query, request.bounded);
  if (!open_measure.ok()) {
    return open_measure.status();
  }
  if (!face_query) {
    output.high_side_liquid_volume = *open_measure;
    return output;
  }
  const double full_area = cell_width * cell_width;
  output.open_face_area = std::min(full_area, *open_measure);
  output.blocked_face_area = full_area - output.open_face_area;
  const double geometric_blocked_area = boundary_internal::TankBoxMeasure(solid, cell_width, excluded_axis);
  if (!std::isfinite(*open_measure) || !IsNormalOrZero(output.open_face_area)
      || !IsNormalOrZero(output.blocked_face_area) || (geometric_blocked_area > 0 && output.blocked_face_area == 0)) {
    return UnrepresentableGeometry(request.bounded);
  }
  return output;
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
      || !std::isfinite(fields.ambient_pressure)) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        bounded ? "" : "Invalid pressure field extents or ambient pressure"
    );
  }
  const auto distances = geometry.center_distances;
  for (std::size_t i = 0; i < geometry.cells.size(); ++i) {
    const auto& cell = geometry.cells[i];
    const auto& field = fields.cells[i];
    if (!std::isfinite(field.target_divergence) || !std::isfinite(field.previous_pressure)
        || std::isnan(field.bulk_modulus) || field.bulk_modulus <= 0) {
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
    if (has_head && has_tail) {
      link.pressure_link = InternalMacPressureLink3D{0};
    } else if (face.open_area > 0) {
      const auto air_distance = LookupSignedDistance(distances, has_head ? *tail : head, bounded);
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
      query.ambient_pressure = fields.ambient_pressure;
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
}  // namespace
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
absl::StatusOr<FrozenLiquidBoundaryGeometry3D> FreezeFloatingBoxLiquidBoundaryGeometry3D(
    const LiquidBoundaryGeometryInputs3D& inputs,
    const AxisAlignedFloatingBoxGeometry3D& box
) {
  const auto status =
      ValidateFloatingBoxInputs(box, inputs.mac_grid.geometry(), inputs.obstacle, inputs.center_distances, false);
  if (!status.ok()) {
    return status;
  }
  FloatingBoxContext context{box, false};
  return boundary_internal::FreezeGeometry(inputs, {ComputeFloatingBoxGeometry, &context});
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
absl::StatusOr<FrozenLiquidBoundaryGeometryView3D> LiquidBoundaryWorkspace3D::FreezeFloatingBox(
    const LiquidBoundaryGeometryViewInputs3D& inputs,
    const AxisAlignedFloatingBoxGeometry3D& box
) {
  if (inputs.cell_indices.size() > capacities_.cells || inputs.center_distances.size() > 27 * capacities_.cells) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  const auto status =
      ValidateFloatingBoxInputs(box, inputs.mac_grid.geometry(), inputs.obstacle, inputs.center_distances, true);
  if (!status.ok()) {
    return status;
  }
  FloatingBoxContext context{box, true};
  return FreezeWithBackend(inputs, {ComputeFloatingBoxGeometry, &context});
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
  const auto wetted_flux_bytes = std::as_bytes(fields.wetted_piston_face_fluxes);
  if (!wetted_flux_bytes.empty()) {
    const auto reserved_cell_bytes = std::as_bytes(std::span(bank.cells.values().data(), bank.cells.capacity()));
    const auto reserved_face_bytes = std::as_bytes(std::span(bank.faces.values().data(), bank.faces.capacity()));
    if (ByteViewsOverlap(wetted_flux_bytes, reserved_cell_bytes)
        || ByteViewsOverlap(wetted_flux_bytes, reserved_face_bytes)) {
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
