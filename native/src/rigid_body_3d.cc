#include "fluidity/host/rigid_body_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

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

absl::Status ValidateRigidBody3D(const RigidBody3D& body) {
  if (!ValidBody(body)) {
    return absl::InvalidArgumentError("invalid rigid body");
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

absl::StatusOr<std::vector<RigidPressurePatch3D>> BuildFloatingBoxPressurePatches3D(
    std::uint64_t body_id,
    Vec3 box_low,
    Vec3 box_high,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::span<const CellIndex3D> pressure_cells
) {
  if (body_id == 0 || !IsFinite(box_low) || !IsFinite(box_high) || !std::isfinite(liquid_height)
      || !IsFinite(geometry.world_origin) || !std::isfinite(geometry.cell_width) || geometry.cell_width <= 0.0
      || box_low.x >= box_high.x || box_low.y >= box_high.y || box_low.z >= box_high.z) {
    return absl::InvalidArgumentError("invalid floating box or pressure grid");
  }

  struct Receiver {
    std::array<std::int32_t, 3> index;
    std::array<double, 3> low;
    std::array<double, 3> high;
  };
  const std::array origin{geometry.world_origin.x, geometry.world_origin.y, geometry.world_origin.z};
  std::vector<Receiver> receivers;
  receivers.reserve(pressure_cells.size());
  for (const auto cell : pressure_cells) {
    Receiver receiver{.index = {cell.x, cell.y, cell.z}, .low = {}, .high = {}};
    if (!receivers.empty()) {
      const auto& previous = receivers.back().index;
      if (std::tie(cell.z, cell.y, cell.x) <= std::tie(previous[2], previous[1], previous[0])) {
        return absl::InvalidArgumentError("floating-box pressure cells must be strictly ordered");
      }
    }
    for (std::size_t axis = 0; axis < origin.size(); ++axis) {
      const double index = receiver.index[axis];
      receiver.low[axis] = std::fma(index, geometry.cell_width, origin[axis]);
      receiver.high[axis] = std::fma(index + 1.0, geometry.cell_width, origin[axis]);
      if (!std::isfinite(receiver.low[axis]) || !std::isfinite(receiver.high[axis])
          || receiver.low[axis] >= receiver.high[axis]) {
        return absl::OutOfRangeError("floating-box pressure cell bounds cannot be resolved");
      }
    }
    receivers.push_back(receiver);
  }

  const std::array lower{box_low.x, box_low.y, box_low.z};
  const std::array upper{box_high.x, box_high.y, box_high.z};
  auto wet_upper = upper;
  wet_upper[1] = std::min(box_high.y, liquid_height);
  std::vector<RigidPressurePatch3D> patches;
  for (std::size_t axis = 0; axis < lower.size(); ++axis) {
    const std::size_t first_tangent = (axis + 1) % lower.size();
    const std::size_t second_tangent = (axis + 2) % lower.size();
    const std::array tangents{first_tangent, second_tangent};
    for (const double sign : {-1.0, 1.0}) {
      const double plane = sign < 0.0 ? lower[axis] : upper[axis];
      const bool dry = axis == 1 ? plane >= liquid_height : wet_upper[1] <= lower[1];
      if (dry) {
        continue;
      }

      const std::size_t first_patch = patches.size();
      std::array min_index{std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::max()};
      std::array max_index{std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::min()};
      std::array min_coordinate{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
      std::array max_coordinate{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
      for (std::size_t cell_index = 0; cell_index < receivers.size(); ++cell_index) {
        const auto& receiver = receivers[cell_index];
        const bool exterior_receiver = sign < 0.0 ? receiver.low[axis] < plane && plane <= receiver.high[axis]
                                                  : receiver.low[axis] <= plane && plane < receiver.high[axis];
        if (!exterior_receiver) {
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
          return absl::OutOfRangeError("floating-box wetted patch area cannot be resolved");
        }

        std::array<double, 3> centroid{};
        centroid[axis] = plane;
        centroid[first_tangent] = std::midpoint(a0, a1);
        centroid[second_tangent] = std::midpoint(b0, b1);
        std::array<double, 3> normal{};
        normal[axis] = sign;
        patches.push_back({
            .patch_id = patches.size() + 1,
            .body_id = body_id,
            .pressure_cell_index = cell_index,
            .wetted_area = area,
            .world_centroid = {centroid[0], centroid[1], centroid[2]},
            .outward_body_normal = {normal[0], normal[1], normal[2]},
        });

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
        return absl::FailedPreconditionError("floating-box wet surface lacks exterior pressure coverage");
      }
      const auto width = static_cast<std::uint64_t>(static_cast<std::int64_t>(max_index[0]) - min_index[0] + 1);
      const auto height = static_cast<std::uint64_t>(static_cast<std::int64_t>(max_index[1]) - min_index[1] + 1);
      const auto patch_count = patches.size() - first_patch;
      if (patch_count % width != 0 || patch_count / width != height) {
        return absl::FailedPreconditionError("floating-box wet pressure coverage has missing tiles");
      }
    }
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
    std::span<const RigidPressurePatch3D> patches
) {
  if (!ValidBody(body) || !std::isfinite(ambient_pressure)) {
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

}  // namespace fluidity
