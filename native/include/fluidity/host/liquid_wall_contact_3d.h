#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/base/liquid_boundary_transfer_types_3d.h"
#include "fluidity/host/liquid_boundary_geometry_3d.h"

namespace fluidity {

struct LiquidWallContactInputs3D {
  Vec3 start_world_position{};  // World meters at substep start.
  Vec3 stored_world_velocity{};  // Particle state in world m/s before response.
  Vec3 advecting_world_velocity{};  // World m/s of the proposed sweep.
  LiquidObstacle3D obstacle{};  // Typed world pose and constant wall velocities during the step.
  double step_duration = 0;  // Positive finite seconds.
  double friction_coefficient = 0;  // Finite nonnegative dimensionless Coulomb coefficient.
  double penetration_tolerance = 0;  // Finite nonnegative initial overlap allowed, in meters.
};

enum class LiquidWallContactDiagnostic3D {
  kNone,
  kSweptImpact,  // Includes endpoint impact and inward velocity at a touching start.
  kInitialPenetration,  // The start was projected onto the incident solid boundary.
};

struct LiquidWallContactResult3D {
  Vec3 world_position{};  // Accepted end position in world meters.
  Vec3 stored_world_velocity{};  // Corrected stored velocity in world m/s.
  Vec3 advecting_world_velocity{};  // Corrected remainder velocity in world m/s.
  std::optional<double> contact_fraction;  // Earliest contact in [0, 1], absent for free advection.
  Vec3 outward_normal{};  // First incident face normal pointing into fluid, zero without contact.
  // Signed wall-relative normal components in m/s before either correction.
  // Negative is approaching. Both are zero without contact.
  double incoming_stored_relative_normal_speed = 0;
  double incoming_advecting_relative_normal_speed = 0;
  LiquidWallContactDiagnostic3D diagnostic = LiquidWallContactDiagnostic3D::kNone;
};

// Sweeps the advecting path against the typed obstacle (or advects freely with
// no obstacle). At a hit, removes inward wall-relative normal
// motion and applies Coulomb friction independently to each velocity, then
// advances the remaining time on the incident fluid side. Initial overlap
// within tolerance is projected to the incident boundary at fraction zero.
// An open tank supports only zero friction. Orthogonal contacts constrain both
// velocity fields. Ties report the first normal and speeds in x, y, z order.
// Returns InvalidArgument for malformed inputs or tank friction, FailedPrecondition
// for excessive overlap, ambiguous slab side or tank closure, and OutOfRange for arithmetic
// that cannot certify contact or its accepted endpoint. Failure publishes no result.
absl::StatusOr<LiquidWallContactResult3D> ResolveLiquidWallContact3D(const LiquidWallContactInputs3D& inputs);

// Stable blade ID and incident side in the blade's authored normal convention.
// Zero ID and zero sign denote an unassigned particle. Assigned signs are +/-1.
struct ElasticBladeSide3D {
  std::uint64_t blade_id = 0;
  int sign = 0;
};

struct ElasticBladeContact3D {
  LiquidWallContactResult3D contact;
  ElasticBladeSide3D side;
  Vec3 particle_impulse;  // Newton-seconds on stored particle momentum.
  Vec3 particle_angular_impulse_about_world_origin;  // Newton-meter-seconds.
  double blade_work;  // Joules transferred from the prescribed moving blade.
};

// Sweeps one elastic point against a constant-velocity slab and computes a
// candidate side and reaction for the accepted step. The slab must maintain
// its authored normal direction for this blade ID. Mass is positive finite
// kilograms. The reaction on the blade is the opposite of the particle
// impulse. The advection correction contributes no second impulse.
// Invalid parameters return InvalidArgument. A different blade, conflicting start or impact
// side, or initially penetrating contact returns FailedPrecondition. An
// unrepresentable sweep or budget returns OutOfRange. Inputs remain intact.
absl::StatusOr<ElasticBladeContact3D> ResolveElasticBladeContact3D(
    const LiquidWallContactInputs3D& inputs,
    double particle_mass,
    std::uint64_t blade_id,
    ElasticBladeSide3D committed_side
);

// Classifies owned MAC anchors at elapsed time in seconds from the slab's
// starting pose. The two exterior regions use the authored normal signs -1
// and +1, including each incident plane. Interior anchors are ineligible.
// Tags describe one blade and use owned-slot order. Invalid pose or time
// returns InvalidArgument. Unrepresentable planes or an ambiguous projection
// returns OutOfRange. Failure returns no tags.
absl::StatusOr<std::vector<LiquidBoundaryFaceTag3D>>
BuildElasticBladeFaceTags3D(SparseMacGridView3D mac_grid, const TranslatingSlab3D& blade, double elapsed_time);

}  // namespace fluidity
