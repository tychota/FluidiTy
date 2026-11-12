#pragma once

#include <cstdint>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {

// Unit quaternion mapping principal-axis body coordinates to world coordinates.
struct RigidOrientation3D {
  double w = 1.0;
  Vec3 vector{};
};

struct RigidBody3D {
  std::uint64_t body_id = 0;  // Stable nonzero identity.
  double mass = 0.0;  // Kilograms.
  Vec3 principal_inertia{};  // Diagonal moments in the body frame, kg m^2.
  Vec3 center{};  // World center of mass in meters.
  RigidOrientation3D orientation{};
  Vec3 linear_velocity{};  // World m/s.
  Vec3 angular_velocity{};  // World radians/s.
  Vec3 force{};  // World force accumulator, newtons.
  Vec3 torque{};  // World torque accumulator about center, newton-meters.
};

struct RigidBodyPose3D {
  Vec3 center;
  RigidOrientation3D orientation;
};

struct RigidBodyStep3D {
  RigidBodyPose3D start;
  RigidBody3D end;
  double duration;  // Seconds.
};

// Checks finite world position and a unit body-to-world quaternion within 1e-12.
absl::Status ValidateRigidBodyPose3D(RigidBodyPose3D pose);

// Checks body identity, positive finite mass and principal inertia, pose and
// finite velocities and loads. A malformed body returns InvalidArgument.
absl::Status ValidateRigidBody3D(const RigidBody3D& body);

// Adds a world force at the given world position and its torque about the
// current center. Returns InvalidArgument for invalid state or nonfinite input
// and OutOfRange on overflow. Does not modify the input.
absl::StatusOr<RigidBody3D>
AccumulateRigidBodyPointForce3D(RigidBody3D body, Vec3 force_world_position, Vec3 world_force);

// Evaluates force and the gyroscopic term at the start, updates velocities,
// and drifts with the resulting constant world linear and angular velocities.
// Returns a candidate with normalized orientation and cleared accumulators.
// Invalid duration, body ID, mass, inertia, pose or loads return InvalidArgument.
// Nonfinite results return OutOfRange. Does not modify committed state.
absl::StatusOr<RigidBodyStep3D> AdvanceRigidBody3D(RigidBody3D committed_body, double duration);

// Samples the same drift used by AdvanceRigidBody3D at a finite fraction in
// [0, 1]. Invalid step or fraction returns InvalidArgument, overflow OutOfRange.
absl::StatusOr<RigidBodyPose3D> SampleRigidBodyPose3D(const RigidBodyStep3D& step, double fraction);

// Bounds travel of a point within radius meters of the body center during the
// drift by linear travel plus the angular arc length. Requires a valid step
// and a finite nonnegative radius. Invalid input returns InvalidArgument,
// overflow OutOfRange.
absl::StatusOr<double> RigidBodySweepDistanceBound3D(const RigidBodyStep3D& step, double radius);

}  // namespace fluidity
