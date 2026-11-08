#pragma once

#include <optional>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_boundary_geometry_3d.h"

namespace fluidity {

struct LiquidWallContactInputs3D {
  Vec3 start_world_position{};  // World meters at substep start.
  Vec3 stored_world_velocity{};  // Particle state in world m/s before response.
  Vec3 advecting_world_velocity{};  // World m/s of the proposed sweep.
  LiquidObstacle3D obstacle{};  // Constant translation of the typed start pose.
  double step_duration = 0;  // Positive finite seconds.
  double friction_coefficient = 0;  // Finite nonnegative dimensionless Coulomb coefficient.
  double penetration_tolerance = 0;  // Finite nonnegative initial overlap allowed, in meters.
};

enum class LiquidWallContactDiagnostic3D {
  kNone,
  kSweptImpact,  // Includes endpoint impact and inward velocity at a touching start.
  kInitialPenetration,  // The start was projected onto its nearest solid face.
};

struct LiquidWallContactResult3D {
  Vec3 world_position{};  // Accepted end position in world meters.
  Vec3 stored_world_velocity{};  // Corrected stored velocity in world m/s.
  Vec3 advecting_world_velocity{};  // Corrected remainder velocity in world m/s.
  std::optional<double> contact_fraction;  // Earliest contact in [0, 1], absent for free advection.
  Vec3 outward_normal{};  // Incident face normal pointing into fluid, zero without contact.
  // Signed wall-relative normal components in m/s before either correction.
  // Negative is approaching. Both are zero without contact.
  double incoming_stored_relative_normal_speed = 0;
  double incoming_advecting_relative_normal_speed = 0;
  LiquidWallContactDiagnostic3D diagnostic = LiquidWallContactDiagnostic3D::kNone;
};

// Sweeps the advecting path against one translating halfspace or slab (or advects
// freely with no obstacle). At a hit, removes inward wall-relative normal
// motion and applies Coulomb friction independently to each velocity, then
// advances the remaining time on the incident fluid side. Initial overlap
// within tolerance is projected to the nearest face at fraction zero.
// Returns InvalidArgument for malformed inputs, FailedPrecondition for
// excessive overlap or an ambiguous slab side, and OutOfRange for arithmetic
// that cannot certify contact or its accepted endpoint. Failure publishes no result.
absl::StatusOr<LiquidWallContactResult3D> ResolveLiquidWallContact3D(const LiquidWallContactInputs3D& inputs);

}  // namespace fluidity
