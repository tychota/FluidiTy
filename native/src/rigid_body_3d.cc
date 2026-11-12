#include "fluidity/host/rigid_body_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <tuple>

#include "fluidity/base/liquid_boundary_geometry_math_3d.h"

namespace fluidity {
namespace {

bool IsFinite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

Vec3 Add(Vec3 first, Vec3 second) {
  return {first.x + second.x, first.y + second.y, first.z + second.z};
}

Vec3 Subtract(Vec3 first, Vec3 second) {
  return {first.x - second.x, first.y - second.y, first.z - second.z};
}

Vec3 Scale(double scalar, Vec3 vector) {
  return {scalar * vector.x, scalar * vector.y, scalar * vector.z};
}

Vec3 Cross(Vec3 first, Vec3 second) {
  return {
      first.y * second.z - first.z * second.y,
      first.z * second.x - first.x * second.z,
      first.x * second.y - first.y * second.x,
  };
}

double Dot(Vec3 first, Vec3 second) {
  return first.x * second.x + first.y * second.y + first.z * second.z;
}

double Norm(Vec3 vector) {
  return std::hypot(vector.x, vector.y, vector.z);
}

bool IsUnit(RigidOrientation3D orientation) {
  return std::abs(std::hypot(orientation.w, Norm(orientation.vector)) - 1.0) <= 1e-12;
}

RigidOrientation3D Multiply(RigidOrientation3D first, RigidOrientation3D second) {
  return {
      .w = first.w * second.w - Dot(first.vector, second.vector),
      .vector =
          Add(Add(Scale(first.w, second.vector), Scale(second.w, first.vector)), Cross(first.vector, second.vector)),
  };
}

Vec3 Rotate(RigidOrientation3D orientation, Vec3 vector) {
  const Vec3 double_cross = Scale(2.0, Cross(orientation.vector, vector));
  return Add(vector, Add(Scale(orientation.w, double_cross), Cross(orientation.vector, double_cross)));
}

bool ValidBody(const RigidBody3D& body) {
  const Vec3 inertia = body.principal_inertia;
  const bool valid_mass = std::isfinite(body.mass) && body.mass > 0.0;
  const bool valid_inertia = IsFinite(inertia) && inertia.x > 0.0 && inertia.y > 0.0 && inertia.z > 0.0;
  return body.body_id != 0 && valid_mass && valid_inertia && IsFinite(body.center) && IsUnit(body.orientation)
      && IsFinite(body.linear_velocity) && IsFinite(body.angular_velocity) && IsFinite(body.force)
      && IsFinite(body.torque);
}

bool ValidPressurePatch(
    const RigidPressurePatch3D& patch,
    std::uint64_t body_id,
    std::uint64_t previous_patch_id,
    std::size_t cell_count
) {
  const double normal_length = Norm(patch.outward_body_normal);
  return patch.patch_id > previous_patch_id && patch.body_id == body_id && patch.pressure_cell_index < cell_count
      && std::isfinite(patch.wetted_area) && patch.wetted_area > 0.0 && IsFinite(patch.world_centroid)
      && std::isfinite(normal_length) && std::abs(normal_length - 1.0) <= 1e-12;
}
bool ValidDrift(const RigidBodyStep3D& step) {
  return std::isfinite(step.duration) && step.duration > 0.0 && IsFinite(step.start.center)
      && IsUnit(step.start.orientation) && IsFinite(step.end.linear_velocity) && IsFinite(step.end.angular_velocity);
}

}  // namespace

absl::Status ValidateRigidBodyPose3D(RigidBodyPose3D pose) {
  if (!IsFinite(pose.center) || !IsUnit(pose.orientation)) {
    return absl::InvalidArgumentError("invalid rigid body pose");
  }
  return absl::OkStatus();
}

absl::StatusOr<Vec3> RotateRigidBodyVector3D(RigidOrientation3D orientation, Vec3 body_vector) {
  if (!IsUnit(orientation) || !IsFinite(body_vector)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto rotated = Rotate(orientation, body_vector);
  if (!IsFinite(rotated)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return rotated;
}
absl::Status ValidateRigidBody3D(const RigidBody3D& body) {
  if (!ValidBody(body)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  return absl::OkStatus();
}

absl::StatusOr<RigidBody3D>
AccumulateRigidBodyPointForce3D(RigidBody3D body, Vec3 force_world_position, Vec3 world_force) {
  if (!ValidBody(body) || !IsFinite(force_world_position) || !IsFinite(world_force)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  body.force = Add(body.force, world_force);
  const Vec3 moment_arm = Subtract(force_world_position, body.center);
  body.torque = Add(body.torque, Cross(moment_arm, world_force));
  if (!IsFinite(body.force) || !IsFinite(body.torque)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return body;
}

enum class RectangularPressureSurface {
  kBoxExterior,
  kOpenTankInterior
};

static absl::Status VisitRectangularPressurePatches(
    std::uint64_t body_id,
    Vec3 box_low,
    Vec3 box_high,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::size_t pressure_cell_count,
    absl::FunctionRef<CellIndex3D(std::size_t)> pressure_cell_at,
    absl::FunctionRef<absl::Status(const RigidPressurePatch3D&)> visit,
    bool bounded,
    RectangularPressureSurface surface = RectangularPressureSurface::kBoxExterior
) {
  const bool tank_interior = surface == RectangularPressureSurface::kOpenTankInterior;
  if (body_id == 0 || !IsFinite(box_low) || !IsFinite(box_high) || !std::isfinite(liquid_height)
      || !IsFinite(geometry.world_origin) || !std::isfinite(geometry.cell_width) || geometry.cell_width <= 0.0
      || box_low.x >= box_high.x || (tank_interior ? box_low.y > box_high.y : box_low.y >= box_high.y)
      || box_low.z >= box_high.z) {
    return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "invalid floating box or pressure grid");
  }

  struct Receiver {
    std::array<std::int32_t, 3> index;
    std::array<double, 3> low;
    std::array<double, 3> high;
  };
  const std::array origin{geometry.world_origin.x, geometry.world_origin.y, geometry.world_origin.z};
  const auto receiver_at = [&](std::size_t ordinal) {
    const auto cell = pressure_cell_at(ordinal);
    Receiver receiver{.index = {cell.x, cell.y, cell.z}, .low = {}, .high = {}};
    for (std::size_t axis = 0; axis < origin.size(); ++axis) {
      const double index = receiver.index[axis];
      receiver.low[axis] = std::fma(index, geometry.cell_width, origin[axis]);
      receiver.high[axis] = std::fma(index + 1.0, geometry.cell_width, origin[axis]);
    }
    return receiver;
  };
  CellIndex3D previous{};
  for (std::size_t ordinal = 0; ordinal < pressure_cell_count; ++ordinal) {
    const auto cell = pressure_cell_at(ordinal);
    if (ordinal > 0 && std::tie(cell.z, cell.y, cell.x) <= std::tie(previous.z, previous.y, previous.x)) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          bounded ? "" : "floating-box pressure cells must be strictly ordered"
      );
    }
    previous = cell;
    const auto receiver = receiver_at(ordinal);
    for (std::size_t axis = 0; axis < origin.size(); ++axis) {
      if (!std::isfinite(receiver.low[axis]) || !std::isfinite(receiver.high[axis])
          || receiver.low[axis] >= receiver.high[axis]) {
        return absl::Status(
            absl::StatusCode::kOutOfRange,
            bounded ? "" : "floating-box pressure cell bounds cannot be resolved"
        );
      }
    }
  }
  const std::array lower{box_low.x, box_low.y, box_low.z};
  const std::array upper{box_high.x, box_high.y, box_high.z};
  auto wet_upper = upper;
  wet_upper[1] = std::min(box_high.y, liquid_height);
  std::uint64_t patch_id = 0;
  for (std::size_t axis = 0; axis < lower.size(); ++axis) {
    const std::size_t first_tangent = (axis + 1) % lower.size();
    const std::size_t second_tangent = (axis + 2) % lower.size();
    const std::array tangents{first_tangent, second_tangent};
    for (const double sign : {-1.0, 1.0}) {
      const double plane = sign < 0.0 ? lower[axis] : upper[axis];
      const bool open_top = tank_interior && axis == 1 && sign > 0.0;
      const double normal_sign = tank_interior ? -sign : sign;
      const bool dry = axis == 1 ? plane >= liquid_height : wet_upper[1] <= lower[1];
      if (open_top || dry) {
        continue;
      }

      const std::size_t first_patch = patch_id;
      std::array min_index{std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::max()};
      std::array max_index{std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::min()};
      std::array min_coordinate{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
      std::array max_coordinate{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
      for (std::size_t cell_index = 0; cell_index < pressure_cell_count; ++cell_index) {
        const auto receiver = receiver_at(cell_index);
        const bool owns_plane = normal_sign < 0.0 ? receiver.low[axis] < plane && plane <= receiver.high[axis]
                                                  : receiver.low[axis] <= plane && plane < receiver.high[axis];
        if (!owns_plane) {
          continue;
        }
        const double a0 = std::max(lower[first_tangent], receiver.low[first_tangent]);
        const double a1 = std::min(wet_upper[first_tangent], receiver.high[first_tangent]);
        const double b0 = std::max(lower[second_tangent], receiver.low[second_tangent]);
        const double b1 = std::min(wet_upper[second_tangent], receiver.high[second_tangent]);
        if (a0 >= a1 || b0 >= b1) {
          continue;
        }
        const double area = (a1 - a0) * (b1 - b0);
        if (!std::isfinite(area) || area <= 0.0) {
          return absl::Status(
              absl::StatusCode::kOutOfRange,
              bounded ? "" : "floating-box wetted patch area cannot be resolved"
          );
        }

        std::array<double, 3> centroid{};
        centroid[axis] = plane;
        centroid[first_tangent] = std::midpoint(a0, a1);
        centroid[second_tangent] = std::midpoint(b0, b1);
        std::array<double, 3> normal{};
        normal[axis] = normal_sign;
        const double wet_height = std::min(wet_upper[1], receiver.high[1]) - std::max(lower[1], receiver.low[1]);
        const double vertical_covariance = tank_interior && axis != 1 ? area * wet_height * wet_height / 12.0 : 0.0;
        if (!std::isfinite(vertical_covariance)) {
          return absl::Status(absl::StatusCode::kOutOfRange, {});
        }
        const auto status = visit({
            .patch_id = ++patch_id,
            .body_id = body_id,
            .pressure_cell_index = cell_index,
            .wetted_area = area,
            .world_centroid = {centroid[0], centroid[1], centroid[2]},
            .outward_body_normal = {normal[0], normal[1], normal[2]},
            .area_vertical_covariance = {0.0, vertical_covariance, 0.0},
        });

        if (!status.ok()) {
          return status;
        }

        const std::array overlap_low{a0, b0};
        const std::array overlap_high{a1, b1};
        for (std::size_t t = 0; t < tangents.size(); ++t) {
          min_index[t] = std::min(min_index[t], receiver.index[tangents[t]]);
          max_index[t] = std::max(max_index[t], receiver.index[tangents[t]]);
          min_coordinate[t] = std::min(min_coordinate[t], overlap_low[t]);
          max_coordinate[t] = std::max(max_coordinate[t], overlap_high[t]);
        }
      }

      // Unique receivers lie in a single normal layer. Their tangent index
      // rectangle must have no holes and its clipped edges must reach the face.
      if (min_coordinate != std::array{lower[first_tangent], lower[second_tangent]}
          || max_coordinate != std::array{wet_upper[first_tangent], wet_upper[second_tangent]}) {
        return absl::Status(
            absl::StatusCode::kFailedPrecondition,
            bounded ? "" : "floating-box wet surface lacks exterior pressure coverage"
        );
      }
      const auto width = static_cast<std::uint64_t>(static_cast<std::int64_t>(max_index[0]) - min_index[0] + 1);
      const auto height = static_cast<std::uint64_t>(static_cast<std::int64_t>(max_index[1]) - min_index[1] + 1);
      const auto patch_count = patch_id - first_patch;
      if (patch_count % width != 0 || patch_count / width != height) {
        return absl::Status(
            absl::StatusCode::kFailedPrecondition,
            bounded ? "" : "floating-box wet pressure coverage has missing tiles"
        );
      }
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<double>
ComputeRigidBoxSubmergedVolume3D(RigidBodyPose3D pose, Vec3 body_half_extent, double liquid_height) {
  const auto valid_pose = ValidateRigidBodyPose3D(pose);
  if (!valid_pose.ok()) {
    return valid_pose;
  }
  if (!IsFinite(body_half_extent) || body_half_extent.x <= 0 || body_half_extent.y <= 0 || body_half_extent.z <= 0
      || !std::isfinite(liquid_height)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  namespace clip = boundary_internal;
  const auto offset = clip::HorizontalPlaneLatticeDistance(liquid_height, {pose.center.y, 0.0, 1.0});
  const double full_volume = (2 * body_half_extent.x) * (2 * body_half_extent.y) * (2 * body_half_extent.z);
  if (!offset.ok() || !std::isnormal(full_volume)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  const auto orientation = pose.orientation;
  const auto normal = Rotate({orientation.w, Scale(-1.0, orientation.vector)}, {0.0, 1.0, 0.0});
  const clip::ClippingPlane waterplane{{normal.x, normal.y, normal.z}, -*offset};
  const auto extent = clip::CartesianComponents(body_half_extent);
  std::array<clip::ClippingVertex, 8> corners{};
  for (std::size_t corner = 0; corner < corners.size(); ++corner) {
    for (std::size_t axis = 0; axis < extent.size(); ++axis) {
      corners[corner].cell_width_position[axis] = ((corner >> axis) & 1) ? extent[axis] : -extent[axis];
    }
  }
  constexpr int kPermutations[6][3]{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  double submerged_volume = 0.0;
  for (const auto& permutation : kPermutations) {
    const auto a = corners[0];
    const auto b = corners[std::size_t{1} << permutation[0]];
    const auto c = corners[(std::size_t{1} << permutation[0]) | (std::size_t{1} << permutation[1])];
    const auto d = corners[7];
    const clip::ClippingPolyhedron tetrahedron{{a, b, c}, {a, b, d}, {a, c, d}, {b, c, d}};
    submerged_volume += clip::ComputeClippedVolume(clip::ClipPolyhedron(tetrahedron, waterplane), 1.0);
  }
  if (!std::isfinite(submerged_volume)
      || submerged_volume / full_volume > 1.0 + 256.0 * std::numeric_limits<double>::epsilon()) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  return std::min(submerged_volume, full_volume);
}

absl::Status VisitStationaryTankPressurePatches3D(
    std::uint64_t tank_id,
    const AxisAlignedLiquidTank3D& tank,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::size_t pressure_cell_count,
    absl::FunctionRef<CellIndex3D(std::size_t)> pressure_cell_at,
    absl::FunctionRef<absl::Status(const RigidPressurePatch3D&)> visit
) {
  if (tank.piston_velocity_x != 0.0 || !std::isfinite(liquid_height) || !std::isfinite(tank.floor_y)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  // A dry tank has no wet faces, but still requires valid horizontal bounds.
  const double wet_top = std::max(liquid_height, tank.floor_y);
  return VisitRectangularPressurePatches(
      tank_id,
      {tank.start_low_x, tank.floor_y, tank.low_z},
      {tank.high_x, wet_top, tank.high_z},
      liquid_height,
      geometry,
      pressure_cell_count,
      pressure_cell_at,
      visit,
      true,
      RectangularPressureSurface::kOpenTankInterior
  );
}

absl::Status VisitFloatingBoxPressurePatches3D(
    std::uint64_t body_id,
    Vec3 box_low,
    Vec3 box_high,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::size_t pressure_cell_count,
    absl::FunctionRef<CellIndex3D(std::size_t)> pressure_cell_at,
    absl::FunctionRef<absl::Status(const RigidPressurePatch3D&)> visit
) {
  return VisitRectangularPressurePatches(
      body_id,
      box_low,
      box_high,
      liquid_height,
      geometry,
      pressure_cell_count,
      pressure_cell_at,
      visit,
      true
  );
}

absl::StatusOr<std::vector<RigidPressurePatch3D>> BuildFloatingBoxPressurePatches3D(
    std::uint64_t body_id,
    Vec3 box_low,
    Vec3 box_high,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::span<const CellIndex3D> pressure_cells
) {
  std::vector<RigidPressurePatch3D> patches;
  const auto status = VisitRectangularPressurePatches(
      body_id,
      box_low,
      box_high,
      liquid_height,
      geometry,
      pressure_cells.size(),
      [&](std::size_t ordinal) {
        return pressure_cells[ordinal];
      },
      [&](const RigidPressurePatch3D& patch) {
        patches.push_back(patch);
        return absl::OkStatus();
      },
      false
  );
  if (!status.ok()) {
    return status;
  }
  return patches;
}
namespace {

using boundary_internal::CartesianVector;
using boundary_internal::ClippingPolygon;

struct WetPolygonMoment {
  double area = 0.0;
  Vec3 weighted_centroid{};
  Vec3 area_vertical_covariance{};
};

absl::StatusOr<WetPolygonMoment> MeasureWetPolygon(const ClippingPolygon& polygon, bool bounded) {
  if (polygon.overflow) {
    return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "oriented box clipping capacity exceeded");
  }
  WetPolygonMoment result;
  if (polygon.size() < 3) {
    return result;
  }
  const auto position = [&](std::size_t vertex) {
    const auto& coordinates = polygon[vertex].cell_width_position;
    return Vec3{coordinates[0], coordinates[1], coordinates[2]};
  };
  const Vec3 first = position(0);
  for (std::size_t vertex = 1; vertex + 1 < polygon.size(); ++vertex) {
    const Vec3 second = position(vertex);
    const Vec3 third = position(vertex + 1);
    const double area = 0.5 * Norm(Cross(Subtract(second, first), Subtract(third, first)));
    const Vec3 centroid = Scale(1.0 / 3.0, Add(Add(first, second), third));
    result.area += area;
    result.weighted_centroid = Add(result.weighted_centroid, Scale(area, centroid));
  }
  if (result.area > 0.0) {
    const auto center = Scale(1.0 / result.area, result.weighted_centroid);
    for (std::size_t vertex = 1; vertex + 1 < polygon.size(); ++vertex) {
      const Vec3 second = position(vertex);
      const Vec3 third = position(vertex + 1);
      const double area = 0.5 * Norm(Cross(Subtract(second, first), Subtract(third, first)));
      const std::array centered{Subtract(first, center), Subtract(second, center), Subtract(third, center)};
      Vec3 sum{};
      Vec3 products{};
      for (const auto point : centered) {
        sum = Add(sum, point);
        products = Add(products, Scale(point.y, point));
      }
      // Exact degree-two triangle integration in centroid-relative coordinates.
      result.area_vertical_covariance =
          Add(result.area_vertical_covariance, Scale(area / 12.0, Add(Scale(sum.y, sum), products)));
    }
  }
  if (!std::isfinite(result.area) || !IsFinite(result.weighted_centroid)
      || !IsFinite(result.area_vertical_covariance)) {
    return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "oriented box patch moment is unrepresentable");
  }
  return result;
}

struct PressureReceiverBounds {
  CartesianVector low;
  CartesianVector high;
};

// Clip in center-relative world coordinates. Coplanar faces need exterior
// ownership since inclusive clipping alone would assign them to both cells.
ClippingPolygon
ClipWetFaceToReceiver(ClippingPolygon face, const PressureReceiverBounds& receiver, CartesianVector outward_normal) {
  for (int axis = 0; axis < 3; ++axis) {
    if (face.empty()) {
      break;
    }
    const double coordinate = face.front().cell_width_position[axis];
    const bool coplanar = std::ranges::all_of(face, [&](const auto& vertex) {
      return vertex.cell_width_position[axis] == coordinate;
    });
    if (coplanar) {
      const bool exterior = outward_normal[axis] < 0.0
          ? receiver.low[axis] < coordinate && coordinate <= receiver.high[axis]
          : receiver.low[axis] <= coordinate && coordinate < receiver.high[axis];
      if (!exterior) {
        return {};
      }
      continue;
    }
    CartesianVector normal{};
    normal[axis] = -1.0;
    face = boundary_internal::ClipPolygon(face, {normal, -receiver.low[axis], false, axis});
    normal[axis] = 1.0;
    face = boundary_internal::ClipPolygon(face, {normal, receiver.high[axis], false, axis});
  }
  return face;
}

absl::Status VisitOrientedBoxPressurePatches(
    const RigidBody3D& body,
    Vec3 body_half_extent,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::size_t pressure_cell_count,
    absl::FunctionRef<CellIndex3D(std::size_t)> pressure_cell_at,
    absl::FunctionRef<absl::Status(const RigidPressurePatch3D&)> visit,
    bool bounded
) {
  if (!ValidBody(body) || !IsFinite(body_half_extent) || body_half_extent.x <= 0.0 || body_half_extent.y <= 0.0
      || body_half_extent.z <= 0.0 || !std::isfinite(liquid_height) || !IsFinite(geometry.world_origin)
      || !std::isfinite(geometry.cell_width) || geometry.cell_width <= 0.0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, bounded ? "" : "invalid oriented box or pressure grid");
  }
  const double liquid_level = liquid_height - body.center.y;
  if (!std::isfinite(liquid_level)) {
    return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "oriented box liquid level is unrepresentable");
  }
  const auto center = boundary_internal::CartesianComponents(body.center);
  const auto origin = boundary_internal::CartesianComponents(geometry.world_origin);
  const auto receiver_bounds = [&](CellIndex3D cell) -> absl::StatusOr<PressureReceiverBounds> {
    const std::array coordinates{cell.x, cell.y, cell.z};
    PressureReceiverBounds bounds{};
    for (std::size_t axis = 0; axis < coordinates.size(); ++axis) {
      const double lattice_index = coordinates[axis];
      const auto relative_coordinate = [&](double ordinate) {
        return boundary_internal::HorizontalPlaneLatticeDistance(
            center[axis],
            {origin[axis], ordinate, geometry.cell_width}
        );
      };
      const auto low = relative_coordinate(lattice_index);
      const auto high = relative_coordinate(lattice_index + 1.0);
      if (!low.ok() || !high.ok() || *low >= *high) {
        return absl::Status(
            absl::StatusCode::kOutOfRange,
            bounded ? "" : "oriented box pressure cell bounds are unrepresentable"
        );
      }
      bounds.low[axis] = *low;
      bounds.high[axis] = *high;
    }
    return bounds;
  };
  for (std::size_t ordinal = 0; ordinal < pressure_cell_count; ++ordinal) {
    const auto cell = pressure_cell_at(ordinal);
    if (ordinal > 0) {
      const auto previous = pressure_cell_at(ordinal - 1);
      if (std::tie(cell.z, cell.y, cell.x) <= std::tie(previous.z, previous.y, previous.x)) {
        return absl::Status(
            absl::StatusCode::kInvalidArgument,
            bounded ? "" : "oriented box pressure cells must be strictly ordered"
        );
      }
    }
    const auto bounds = receiver_bounds(cell);
    if (!bounds.ok()) {
      return bounds.status();
    }
  }

  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  constexpr std::array<std::array<double, 2>, 4> kCornerSigns{{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}};
  std::uint64_t next_patch_id = 1;
  for (std::size_t axis = 0; axis < kComponents.size(); ++axis) {
    const auto normal_component = kComponents[axis];
    const auto first_tangent = kComponents[(axis + 1) % kComponents.size()];
    const auto second_tangent = kComponents[(axis + 2) % kComponents.size()];
    for (const double sign : {-1.0, 1.0}) {
      Vec3 body_normal{};
      body_normal.*normal_component = sign;
      const Vec3 outward_normal = Rotate(body.orientation, body_normal);
      ClippingPolygon face;
      bool has_wet_vertex = false;
      for (const auto& corner : kCornerSigns) {
        Vec3 local{};
        local.*normal_component = sign * (body_half_extent.*normal_component);
        local.*first_tangent = corner[0] * (body_half_extent.*first_tangent);
        local.*second_tangent = corner[1] * (body_half_extent.*second_tangent);
        const Vec3 rotated = Rotate(body.orientation, local);
        const double distance = rotated.y - liquid_level;
        if (!IsFinite(rotated) || !std::isfinite(distance)) {
          return absl::Status(absl::StatusCode::kOutOfRange, bounded ? "" : "oriented box vertex is unrepresentable");
        }
        has_wet_vertex |= distance < 0.0;
        face.push_back({boundary_internal::CartesianComponents(rotated), distance});
      }
      if (!has_wet_vertex) {
        continue;
      }
      const auto wet_face = boundary_internal::ClipPolygon(face, {{0.0, 1.0, 0.0}, liquid_level, false, 1});
      const auto expected = MeasureWetPolygon(wet_face, bounded);
      if (!expected.ok()) {
        return expected.status();
      }
      if (expected->area <= 0.0) {
        return absl::Status(
            absl::StatusCode::kOutOfRange,
            bounded ? "" : "oriented box wetted face area is unrepresentable"
        );
      }
      double covered_area = 0.0;
      for (std::size_t ordinal = 0; ordinal < pressure_cell_count; ++ordinal) {
        // Recompute the bounds to preserve face-major ordering without receiver storage.
        const auto bounds = receiver_bounds(pressure_cell_at(ordinal));
        if (!bounds.ok()) {
          return bounds.status();
        }
        const auto clipped =
            ClipWetFaceToReceiver(wet_face, *bounds, boundary_internal::CartesianComponents(outward_normal));
        const auto moment = MeasureWetPolygon(clipped, bounded);
        if (!moment.ok()) {
          return moment.status();
        }
        if (moment->area == 0.0) {
          continue;
        }
        const Vec3 centroid = Add(body.center, Scale(1.0 / moment->area, moment->weighted_centroid));
        if (!IsFinite(centroid)) {
          return absl::Status(
              absl::StatusCode::kOutOfRange,
              bounded ? "" : "oriented box pressure centroid is unrepresentable"
          );
        }
        covered_area += moment->area;
        const auto status = visit({
            .patch_id = next_patch_id++,
            .body_id = body.body_id,
            .pressure_cell_index = ordinal,
            .wetted_area = moment->area,
            .world_centroid = centroid,
            .outward_body_normal = outward_normal,
            .area_vertical_covariance = moment->area_vertical_covariance,
        });
        if (!status.ok()) {
          return status;
        }
      }
      constexpr double kAreaRoundoff = 256 * std::numeric_limits<double>::epsilon();
      if (!std::isfinite(covered_area) || std::abs(covered_area - expected->area) > kAreaRoundoff * expected->area) {
        return absl::Status(
            absl::StatusCode::kFailedPrecondition,
            bounded ? "" : "oriented box wet surface lacks exterior pressure coverage"
        );
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status VisitOrientedBoxPressurePatches3D(
    const RigidBody3D& body,
    Vec3 body_half_extent,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::size_t pressure_cell_count,
    absl::FunctionRef<CellIndex3D(std::size_t)> pressure_cell_at,
    absl::FunctionRef<absl::Status(const RigidPressurePatch3D&)> visit
) {
  return VisitOrientedBoxPressurePatches(
      body,
      body_half_extent,
      liquid_height,
      geometry,
      pressure_cell_count,
      pressure_cell_at,
      visit,
      true
  );
}

absl::StatusOr<std::vector<RigidPressurePatch3D>> BuildOrientedBoxPressurePatches3D(
    const RigidBody3D& body,
    Vec3 body_half_extent,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::span<const CellIndex3D> pressure_cells
) {
  std::vector<RigidPressurePatch3D> patches;
  const auto status = VisitOrientedBoxPressurePatches(
      body,
      body_half_extent,
      liquid_height,
      geometry,
      pressure_cells.size(),
      [&](std::size_t ordinal) {
        return pressure_cells[ordinal];
      },
      [&](const RigidPressurePatch3D& patch) {
        patches.push_back(patch);
        return absl::OkStatus();
      },
      false
  );
  if (!status.ok()) {
    return status;
  }
  return patches;
}
absl::StatusOr<std::vector<double>> ComputeRigidBodyInwardWallFlux3D(
    const RigidBody3D& body,
    std::size_t pressure_cell_count,
    std::span<const RigidPressurePatch3D> patches
) {
  if (!ValidBody(body)) {
    return absl::InvalidArgumentError("invalid rigid body for wall flux");
  }
  std::vector<double> inward_wall_fluxes(pressure_cell_count);
  std::uint64_t previous_patch_id = 0;
  for (const auto& patch : patches) {
    if (!ValidPressurePatch(patch, body.body_id, previous_patch_id, pressure_cell_count)) {
      return absl::InvalidArgumentError("invalid rigid wall flux patch");
    }
    previous_patch_id = patch.patch_id;
    const Vec3 moment_arm = Subtract(patch.world_centroid, body.center);
    const Vec3 surface_velocity = Add(body.linear_velocity, Cross(body.angular_velocity, moment_arm));
    const double normal_speed = Dot(surface_velocity, patch.outward_body_normal);
    const double patch_flux = patch.wetted_area * normal_speed;
    auto& inward_wall_flux = inward_wall_fluxes[patch.pressure_cell_index];
    inward_wall_flux += patch_flux;
    if (!std::isfinite(inward_wall_flux)) {
      return absl::OutOfRangeError("nonfinite integrated rigid wall flux");
    }
  }
  return inward_wall_fluxes;
}
absl::StatusOr<RigidBody3D> AccumulateRigidBodyPressure3D(
    RigidBody3D body,
    std::span<const double> solved_pressures,
    double ambient_pressure,
    std::span<const RigidPressurePatch3D> patches,
    double pressure_gradient_y
) {
  if (!ValidBody(body) || !std::isfinite(ambient_pressure) || !std::isfinite(pressure_gradient_y)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  std::uint64_t previous_patch_id = 0;
  for (const auto& patch : patches) {
    if (!ValidPressurePatch(patch, body.body_id, previous_patch_id, solved_pressures.size())) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    previous_patch_id = patch.patch_id;
    const double pressure = solved_pressures[patch.pressure_cell_index];
    if (!std::isfinite(pressure)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const double pressure_difference = pressure - ambient_pressure;
    const Vec3 pressure_force = Scale(-patch.wetted_area * pressure_difference, patch.outward_body_normal);
    if (!IsFinite(pressure_force)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    const auto loaded = AccumulateRigidBodyPointForce3D(body, patch.world_centroid, pressure_force);
    if (!loaded.ok()) {
      return loaded.status();
    }
    body = *loaded;
    if (pressure_gradient_y != 0.0) {
      body.torque =
          Add(body.torque,
              Scale(-pressure_gradient_y, Cross(patch.area_vertical_covariance, patch.outward_body_normal)));
      if (!IsFinite(body.torque)) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
    }
  }
  return body;
}

absl::StatusOr<RigidBodyStep3D> AdvanceRigidBody3D(RigidBody3D committed_body, double duration) {
  if (!ValidBody(committed_body) || !std::isfinite(duration) || duration <= 0.0) {
    return absl::InvalidArgumentError("invalid rigid body or duration");
  }
  RigidBodyStep3D step{
      .start = {committed_body.center, committed_body.orientation},
      .end = committed_body,
      .duration = duration,
  };
  const Vec3 linear_acceleration{
      committed_body.force.x / committed_body.mass,
      committed_body.force.y / committed_body.mass,
      committed_body.force.z / committed_body.mass,
  };
  step.end.linear_velocity = Add(committed_body.linear_velocity, Scale(duration, linear_acceleration));

  // Evaluate I^-1(T - omega x I omega) in the principal-axis frame.
  const RigidOrientation3D world_to_body{
      .w = committed_body.orientation.w,
      .vector = Scale(-1.0, committed_body.orientation.vector),
  };
  const Vec3 body_angular_velocity = Rotate(world_to_body, committed_body.angular_velocity);
  const Vec3 body_torque = Rotate(world_to_body, committed_body.torque);
  const Vec3 inertia = committed_body.principal_inertia;
  const Vec3 body_angular_momentum{
      inertia.x * body_angular_velocity.x,
      inertia.y * body_angular_velocity.y,
      inertia.z * body_angular_velocity.z,
  };
  const Vec3 effective_body_torque = Subtract(body_torque, Cross(body_angular_velocity, body_angular_momentum));
  const Vec3 body_angular_acceleration{
      effective_body_torque.x / inertia.x,
      effective_body_torque.y / inertia.y,
      effective_body_torque.z / inertia.z,
  };
  const Vec3 world_angular_acceleration = Rotate(committed_body.orientation, body_angular_acceleration);
  step.end.angular_velocity = Add(committed_body.angular_velocity, Scale(duration, world_angular_acceleration));
  if (!IsFinite(step.end.linear_velocity) || !IsFinite(step.end.angular_velocity)) {
    return absl::OutOfRangeError("nonfinite rigid body velocity");
  }

  const auto pose = SampleRigidBodyPose3D(step, 1.0);
  if (!pose.ok()) {
    return pose.status();
  }
  step.end.center = pose->center;
  step.end.orientation = pose->orientation;
  step.end.force = {};
  step.end.torque = {};
  return step;
}

absl::StatusOr<RigidBodyPose3D> SampleRigidBodyPose3D(const RigidBodyStep3D& step, double fraction) {
  if (!ValidDrift(step) || !std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0) {
    return absl::InvalidArgumentError("invalid rigid body drift or sample fraction");
  }
  if (fraction == 0.0) {
    return step.start;
  }
  const double elapsed = fraction * step.duration;
  const Vec3 center = Add(step.start.center, Scale(elapsed, step.end.linear_velocity));
  const Vec3 rotation_vector = Scale(elapsed, step.end.angular_velocity);
  const double angle = Norm(rotation_vector);
  if (!IsFinite(center) || !std::isfinite(angle)) {
    return absl::OutOfRangeError("nonfinite rigid body pose");
  }
  if (angle == 0.0) {
    return RigidBodyPose3D{center, step.start.orientation};
  }
  const double half_angle = 0.5 * angle;
  const double vector_scale = half_angle == 0.0 ? 0.5 : std::sin(half_angle) / angle;
  const RigidOrientation3D world_rotation{
      .w = std::cos(half_angle),
      .vector = Scale(vector_scale, rotation_vector),
  };
  const auto orientation = Multiply(world_rotation, step.start.orientation);
  const double inverse_norm = 1.0 / std::hypot(orientation.w, Norm(orientation.vector));
  return RigidBodyPose3D{center, {inverse_norm * orientation.w, Scale(inverse_norm, orientation.vector)}};
}

absl::StatusOr<double> RigidBodySweepDistanceBound3D(const RigidBodyStep3D& step, double radius) {
  if (!ValidDrift(step) || !std::isfinite(radius) || radius < 0.0) {
    return absl::InvalidArgumentError("invalid rigid body drift or sweep radius");
  }
  const Vec3 translation = Scale(step.duration, step.end.linear_velocity);
  // Each sampled center component moves monotonically. Include its rounded
  // endpoint displacement so a large world origin cannot hide that motion.
  const Vec3 rounded_translation = Subtract(Add(step.start.center, translation), step.start.center);
  const double translation_distance = std::max(Norm(translation), Norm(rounded_translation));
  const double rotation_angle = Norm(Scale(step.duration, step.end.angular_velocity));
  const double travel = std::fma(radius, rotation_angle, translation_distance);
  // Allow for rounding in the drift products, norms and arc-length sum.
  const double rounded_bound = std::nextafter(
      travel * (1.0 + 8.0 * std::numeric_limits<double>::epsilon()),
      std::numeric_limits<double>::infinity()
  );
  if (!std::isfinite(rounded_bound)) {
    return absl::OutOfRangeError("nonfinite rigid sweep bound");
  }
  return rounded_bound;
}

absl::StatusOr<std::optional<RigidBoxSweepHit3D>> SweepRigidBoxPoint3D(
    const RigidBodyStep3D& drift,
    Vec3 body_half_extent,
    Vec3 start_world_point,
    Vec3 world_point_velocity
) {
  const auto error = [](absl::StatusCode code) {
    return absl::Status(code, {});
  };
  if (!ValidDrift(drift) || !IsFinite(body_half_extent) || body_half_extent.x <= 0.0 || body_half_extent.y <= 0.0
      || body_half_extent.z <= 0.0 || !IsFinite(start_world_point) || !IsFinite(world_point_velocity)) {
    return error(absl::StatusCode::kInvalidArgument);
  }
  const Vec3 relative_start = Subtract(start_world_point, drift.start.center);
  const Vec3 relative_travel = Scale(drift.duration, Subtract(world_point_velocity, drift.end.linear_velocity));
  const Vec3 world_point_end = Add(start_world_point, Scale(drift.duration, world_point_velocity));
  const double rotation_angle = Norm(Scale(drift.duration, drift.end.angular_velocity));
  if (!IsFinite(relative_start) || !IsFinite(relative_travel) || !IsFinite(world_point_end)
      || !std::isfinite(rotation_angle)) {
    return error(absl::StatusCode::kOutOfRange);
  }
  const auto end_pose = SampleRigidBodyPose3D(drift, 1.0);
  if (!end_pose.ok()) {
    return error(end_pose.status().code());
  }
  if (rotation_angle > std::numbers::pi / 2.0) {
    return error(absl::StatusCode::kAborted);
  }

  // The point relative to the box center is affine in the normalized time.
  // Its norm is bounded by the larger endpoint norm. In rotating body
  // coordinates, each gap changes at most by translation plus angular travel.
  const double distance_scale = Norm(relative_start) + Norm(relative_travel) + Norm(body_half_extent);
  const double roundoff = boundary_internal::AddUp(
      128.0 * std::numeric_limits<double>::epsilon() * distance_scale,
      std::numeric_limits<double>::min()
  );
  const double relative_radius = std::max(Norm(relative_start), Norm(Add(relative_start, relative_travel)));
  const double gap_travel_bound =
      boundary_internal::AddUp(std::fma(rotation_angle, relative_radius, Norm(relative_travel)), roundoff);
  if (!std::isfinite(gap_travel_bound) || !std::isfinite(roundoff)) {
    return error(absl::StatusCode::kOutOfRange);
  }

  // Normalize the same initial quaternion for every sample, including t=0.
  // Only orientation is sampled here to keep the point path center-relative.
  RigidBodyStep3D rotation = drift;
  rotation.start.center = {};
  rotation.end.linear_velocity = {};
  const auto start_orientation = drift.start.orientation;
  const double inverse_norm = 1.0 / std::hypot(start_orientation.w, Norm(start_orientation.vector));
  rotation.start.orientation = {inverse_norm * start_orientation.w, Scale(inverse_norm, start_orientation.vector)};
  struct GapSample {
    double gap;
    double next_gap;
    double coordinate_magnitude;
    std::size_t axis;
    int sign;
  };
  const auto gap_at = [&](double fraction) -> absl::StatusOr<GapSample> {
    const auto pose = SampleRigidBodyPose3D(rotation, fraction);
    if (!pose.ok()) {
      return error(pose.status().code());
    }
    const Vec3 relative_position{
        std::fma(fraction, relative_travel.x, relative_start.x),
        std::fma(fraction, relative_travel.y, relative_start.y),
        std::fma(fraction, relative_travel.z, relative_start.z),
    };
    const auto orientation = pose->orientation;
    const Vec3 local_point = Rotate({orientation.w, Scale(-1.0, orientation.vector)}, relative_position);
    if (!IsFinite(local_point)) {
      return error(absl::StatusCode::kOutOfRange);
    }
    constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
    GapSample result{
        .gap = -std::numeric_limits<double>::infinity(),
        .next_gap = -std::numeric_limits<double>::infinity(),
        .coordinate_magnitude = 0.0,
        .axis = 0,
        .sign = 1,
    };
    for (std::size_t axis = 0; axis < kComponents.size(); ++axis) {
      const auto component = kComponents[axis];
      const double coordinate = local_point.*component;
      const double gap = std::abs(coordinate) - body_half_extent.*component;
      if (gap > result.gap) {
        result.next_gap = result.gap;
        result.gap = gap;
        result.coordinate_magnitude = std::abs(coordinate);
        result.axis = axis;
        result.sign = coordinate < 0.0 ? -1 : 1;
      } else {
        result.next_gap = std::max(result.next_gap, gap);
      }
    }
    return result;
  };

  double exterior_prefix_end = 0.0;
  const auto start_gap = gap_at(0.0);
  if (!start_gap.ok()) {
    return start_gap.status();
  }
  if (start_gap->gap < -roundoff) {
    return error(absl::StatusCode::kFailedPrecondition);
  }
  if (start_gap->gap <= roundoff) {
    // A point carried to the face by the previous step can start within
    // roundoff of it. Use the initial wall-relative motion to resolve entry.
    constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
    Vec3 body_normal{};
    body_normal.*kComponents[start_gap->axis] = start_gap->sign;
    const auto world_normal = Rotate(rotation.start.orientation, body_normal);
    const auto angular_travel = Scale(drift.duration, drift.end.angular_velocity);
    const auto relative_face_travel = Subtract(relative_travel, Cross(angular_travel, relative_start));
    const double normal_travel = Dot(world_normal, relative_face_travel);
    const bool unique_face =
        start_gap->gap - start_gap->next_gap > 2.0 * roundoff && start_gap->coordinate_magnitude > 2.0 * roundoff;
    if (!unique_face || !std::isfinite(normal_travel)) {
      return error(absl::StatusCode::kAborted);
    }
    const bool tangential = std::abs(normal_travel) <= roundoff;
    const auto normal_rate = Cross(angular_travel, world_normal);
    const double normal_curvature =
        Dot(Cross(angular_travel, normal_rate), relative_start) + 2.0 * Dot(normal_rate, relative_travel);
    // With a fixed orientation, an exactly tangential path has zero affine
    // face-gap slope. Carry the point so stored normal motion is still resolved.
    bool fixed_tangent = false;
    if (rotation_angle == 0.0 && Norm(drift.end.angular_velocity) == 0.0) {
      boundary_internal::ProjectedSum normal_speed;
      normal_speed.AddRelativeMotion(world_normal, world_point_velocity, drift.end.linear_velocity, 1.0);
      const auto direction = normal_speed.Bounds().sign();
      fixed_tangent = direction.has_value() && *direction == 0;
    }
    if (tangential && !fixed_tangent
        && (!std::isfinite(normal_curvature) || std::abs(normal_curvature) <= 4.0 * roundoff)) {
      return error(absl::StatusCode::kAborted);
    }
    if (fixed_tangent || (tangential ? normal_curvature < 0.0 : normal_travel < 0.0)) {
      return std::optional<RigidBoxSweepHit3D>{{
          .fraction_lower = 0.0,
          .fraction_upper = 0.0,
          .body_axis = start_gap->axis,
          .outward_sign = start_gap->sign,
      }};
    }

    // The signed gap to this face has second derivative bounded by the
    // rotating normal's second derivative and twice its rotation against the
    // relative point travel. Keep at least half the resolved outward slope.
    const double curvature_bound = boundary_internal::AddUp(
        rotation_angle * (rotation_angle * relative_radius + 2.0 * Norm(relative_travel)),
        roundoff
    );
    if (!std::isfinite(curvature_bound)) {
      return error(absl::StatusCode::kOutOfRange);
    }
    const double outward_travel = boundary_internal::AddDown(normal_travel, -roundoff);
    exterior_prefix_end = std::min(1.0, std::nextafter(0.5 * outward_travel / curvature_bound, 0.0));
    if (tangential) {
      // Classify the near-zero initial slope as tangential within the contact
      // roundoff band. Bound the change in curvature on the outward prefix.
      const double curvature_change_bound = boundary_internal::AddUp(
          rotation_angle * rotation_angle * (rotation_angle * relative_radius + 3.0 * Norm(relative_travel)),
          roundoff
      );
      if (!std::isfinite(curvature_change_bound)) {
        return error(absl::StatusCode::kOutOfRange);
      }
      const double outward_curvature = boundary_internal::AddDown(normal_curvature, -roundoff);
      exterior_prefix_end = std::min(1.0, std::nextafter(0.5 * outward_curvature / curvature_change_bound, 0.0));
    }
    const auto departing_gap = gap_at(exterior_prefix_end);
    if (!departing_gap.ok()) {
      return departing_gap.status();
    }
    if (departing_gap->gap <= roundoff) {
      return error(absl::StatusCode::kAborted);
    }
    if (exterior_prefix_end == 1.0) {
      return std::optional<RigidBoxSweepHit3D>{};
    }
  }

  struct Interval {
    double begin;
    double end;
  };
  constexpr double kLeafWidth = 0x1p-40;
  constexpr double kMaximumHitWidth = 0x1p-32;
  constexpr int kMaximumProbes = 4096;
  // A rounded midpoint after a non-dyadic prefix can need one extra split.
  std::array<Interval, 42> pending;
  std::size_t count = 1;
  pending.front() = {exterior_prefix_end, 1.0};
  std::optional<double> possible_entry;
  for (int probe = 0; count > 0 && probe < kMaximumProbes; ++probe) {
    const auto interval = pending[--count];
    if (possible_entry && interval.begin - *possible_entry > kMaximumHitWidth) {
      return error(absl::StatusCode::kAborted);
    }
    const double width = interval.end - interval.begin;
    const double midpoint = std::midpoint(interval.begin, interval.end);
    const auto middle = gap_at(midpoint);
    if (!middle.ok()) {
      return middle.status();
    }
    const double excursion = std::nextafter(gap_travel_bound * (0.5 * width), std::numeric_limits<double>::infinity());
    if (middle->gap > boundary_internal::AddUp(roundoff, excursion)) {
      continue;
    }
    if (width > kLeafWidth) {
      // Visit the earlier half first so a preceding ambiguous contact cannot
      // be skipped in favor of a later interior sample.
      pending[count++] = {midpoint, interval.end};
      pending[count++] = {interval.begin, midpoint};
      continue;
    }
    if (!possible_entry) {
      possible_entry = interval.begin;
    }
    const auto witness = gap_at(interval.end);
    if (!witness.ok()) {
      return witness.status();
    }
    if (witness->gap < -roundoff) {
      const double bracket_width = interval.end - *possible_entry;
      // Require the same face and sign throughout the bracket. Two distinct
      // gaps can each change by the travel bound before the witness.
      const double face_uncertainty = boundary_internal::AddUp(2.0 * roundoff, 2.0 * gap_travel_bound * bracket_width);
      if (bracket_width > kMaximumHitWidth || witness->gap - witness->next_gap <= face_uncertainty
          || witness->coordinate_magnitude <= face_uncertainty) {
        return error(absl::StatusCode::kAborted);
      }
      return std::optional<RigidBoxSweepHit3D>{{
          .fraction_lower = *possible_entry,
          .fraction_upper = interval.end,
          .body_axis = witness->axis,
          .outward_sign = witness->sign,
      }};
    }
  }
  if (possible_entry || count > 0) {
    return error(absl::StatusCode::kAborted);
  }
  return std::optional<RigidBoxSweepHit3D>{};
}
}  // namespace fluidity
