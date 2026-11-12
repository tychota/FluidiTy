#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/functional/function_ref.h"
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

struct RigidPressurePatch3D {
  std::uint64_t patch_id;  // Stable nonzero identity within the frozen geometry.
  std::uint64_t body_id;
  std::size_t pressure_cell_index;  // Index in the solved pressure vector.
  double wetted_area;  // Certified wet surface area in m².
  Vec3 world_centroid;  // Meters.
  Vec3 outward_body_normal;  // Unit normal pointing out of the rigid body.
};

// Clips a finite world-axis-aligned box to a horizontal liquid plane (wet
// below liquid_height). Each positive-area rectangle is assigned to the
// exterior pressure cell, with cells strictly ordered by (z, y, x). Returns
// patches in x-/x+/y-/y+/z-/z+ and then pressure-cell order. Coordinates are
// meters. Invalid bounds or ordering return InvalidArgument, missing wet
// coverage returns FailedPrecondition, and unrepresentable geometry returns
// OutOfRange. Failure returns no patches.
absl::StatusOr<std::vector<RigidPressurePatch3D>> BuildFloatingBoxPressurePatches3D(
    std::uint64_t body_id,
    Vec3 box_low,
    Vec3 box_high,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::span<const CellIndex3D> pressure_cells
);
// Streams the same patches without allocation. Callbacks borrow a stable
// pressure-cell ordering and must stage any writes until success because a
// later validation or visitor error can reject the visit. Errors are code-only.
absl::Status VisitFloatingBoxPressurePatches3D(
    std::uint64_t body_id,
    Vec3 box_low,
    Vec3 box_high,
    double liquid_height,
    UniformMacGeometry3D geometry,
    std::size_t pressure_cell_count,
    absl::FunctionRef<CellIndex3D(std::size_t)> pressure_cell_at,
    absl::FunctionRef<absl::Status(const RigidPressurePatch3D&)> visit
);
// Integrates the rigid surface velocity over wetted patches. Returns one
// flux in m³/s per pressure cell, positive into the liquid. Affine rigid
// velocity makes centroid evaluation exact for each planar patch. Patches
// use the same ordering and ownership as pressure traction. Only pass
// patches not already represented by a prescribed MAC face flux. Malformed
// inputs return InvalidArgument and range failures return OutOfRange.
absl::StatusOr<std::vector<double>> ComputeRigidBodyInwardWallFlux3D(
    const RigidBody3D& body,
    std::size_t pressure_cell_count,
    std::span<const RigidPressurePatch3D> patches
);
// Adds pressure traction and its moment about the body center, in patch ID
// order. Pressure and ambient pressure share a reference and are in pascals.
// Patches must be disjoint, strictly ordered by ID, and belong to the body.
// Malformed patches or pressure indices return InvalidArgument. Nonfinite
// load arithmetic returns OutOfRange. Input body and spans stay unchanged.
absl::StatusOr<RigidBody3D> AccumulateRigidBodyPressure3D(
    RigidBody3D body,
    std::span<const double> solved_pressures,
    double ambient_pressure,
    std::span<const RigidPressurePatch3D> patches
);

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
