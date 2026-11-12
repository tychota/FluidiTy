#include "fluidity/host/rigid_body_3d.h"

#include <cmath>
#include <limits>

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

bool ValidDrift(const RigidBodyStep3D& step) {
  return std::isfinite(step.duration) && step.duration > 0.0 && IsFinite(step.start.center)
      && IsUnit(step.start.orientation) && IsFinite(step.end.linear_velocity) && IsFinite(step.end.angular_velocity);
}

}  // namespace

absl::StatusOr<RigidBody3D>
AccumulateRigidBodyPointForce3D(RigidBody3D body, Vec3 force_world_position, Vec3 world_force) {
  if (!ValidBody(body) || !IsFinite(force_world_position) || !IsFinite(world_force)) {
    return absl::InvalidArgumentError("invalid rigid body or point load");
  }
  body.force = Add(body.force, world_force);
  const Vec3 moment_arm = Subtract(force_world_position, body.center);
  body.torque = Add(body.torque, Cross(moment_arm, world_force));
  if (!IsFinite(body.force) || !IsFinite(body.torque)) {
    return absl::OutOfRangeError("nonfinite rigid body load");
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
  const double translation_distance = Norm(Scale(step.duration, step.end.linear_velocity));
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
