#pragma once

#include "fluidity/host/motion_interval_3d.h"
#include "fluidity/host/whitewater_3d.h"

namespace fluidity {
struct WhitewaterTransportPoint3D {
  WhitewaterKinematics3D kinematics{};
  WhitewaterKind3D kind = WhitewaterKind3D::kFoam;
  double birth_time_seconds = 0;
};
struct WhitewaterTransportOptions3D {
  double cell_width_m = 1;
  double classification_radius_m = 0.1;
  double spray_drag_per_second = 0;
  double bubble_drag_per_second = 0;
  double bubble_buoyancy_scale = 1;
  double foam_lifetime_seconds = 2;  // From birth, enforced only while classified as foam.
  Vec3 gravity{};  // World m/s², applied to spray and opposite/scaled for bubbles.
  Vec3 ambient_velocity{};  // World m/s, spray drag target.
  std::size_t maximum_substeps = 4096;
};
struct WhitewaterTransportResult3D {
  WhitewaterTransportPoint3D point{};
  std::size_t accepted_substeps = 0;
  std::size_t bubble_fallback_substeps = 0;
  double maximum_displacement_m = 0;
  bool expired = false;
};
// Positive is air. The geometry query is independent of carrier validity.
using WhitewaterDistanceSampler3D = absl::FunctionRef<absl::StatusOr<double>(Vec3 world_position, double time_seconds)>;

// Advances a point in a collision-free domain from its birth time or interval
// start (whichever is later) to the final accepted stamp. Position/velocity
// are the state at that start. Splits at carrier knots; foam uses bounded RK2,
// spray and bubbles use exact frozen drag with each displacement ≤0.5h.
// Classifies at completed transport boundaries. Carrier FailedPrecondition
// means missing support: foam fails, bubbles use zero velocity and keep their
// class, spray uses ambient forcing without a liquid carrier query. Other
// errors propagate. A birth at the target does no transport or queries.
// Foam stops at expiry before later carrier queries. No obstacle response.
// Invalid controls return
// InvalidArgument, overflow OutOfRange, and work/refinement bounds
// ResourceExhausted. Inputs are borrowed, no partial output on failure.
absl::StatusOr<WhitewaterTransportResult3D> AdvanceWhitewaterPoint3D(
    const WhitewaterTransportPoint3D& point,
    const MotionInterval3D& interval,
    const WhitewaterTransportOptions3D& options,
    MotionSpatialSampler3D carrier,
    WhitewaterDistanceSampler3D liquid_distance
);
}  // namespace fluidity
