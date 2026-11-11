#pragma once

#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_types_3d.h"

namespace fluidity {
enum class WhitewaterKind3D : std::uint8_t {
  kSpray,
  kFoam,
  kBubble
};
struct WhitewaterSourceCell3D {
  CellIndex3D index;  // Signed coordinate on the scene-fixed physical source grid.
  Vec3 velocity;  // Valid source carrier, world m/s.
  Vec3 normal;  // Unit outward world normal, explicitly validated even at rest.
  double curvature_per_m;  // Signed mean-curvature convention from Simulate.
  bool carrier_valid = true;
};
struct WhitewaterRamp3D {
  double low;
  double high;  // Strictly above low, in the statistic's physical units.
};
struct WhitewaterSourceOptions3D {
  double spacing_m;  // Fixed across camera and simulation quality changes.
  double radius_m;
  WhitewaterRamp3D kinetic;  // m²/s².
  WhitewaterRamp3D trapped_air;  // m/s.
  WhitewaterRamp3D crest;  // 1/m.
  double trapped_air_weight;  // Nonnegative dimensionless coefficient.
  double crest_weight;
};
struct WhitewaterSourceStatistics3D {
  double specific_energy_m2_per_s2;
  double trapped_air_m_per_s;
  double crest_per_m;
  double potential;  // Dimensionless Phi, before occupied volume and accepted dt.
};
// Computes the §10 fixed-grid variant from up to 26 unique adjacent neighbors.
// A supplied self record is excluded, so 27 records including self are allowed.
// Evaluates relative velocity weights, specific kinetic energy and outward crest
// criterion without using particle mass or camera state. Invalid fields,
// normals, thresholds or neighbor coordinates return InvalidArgument; absent
// carrier FailedPrecondition; numerical overflow OutOfRange. No partial output.
absl::StatusOr<WhitewaterSourceStatistics3D> ComputeWhitewaterSource3D(
    const WhitewaterSourceCell3D& center,
    std::span<const WhitewaterSourceCell3D> neighbors,
    const WhitewaterSourceOptions3D& options
);
struct WhitewaterKinematics3D {
  Vec3 position;  // World metres.
  Vec3 velocity;  // World m/s.
};
// Integrates dv/dt=k*(carrier-v)+acceleration exactly for frozen forcing.
// Uses expm1 and the analytic zero-drag limit, with a small-k*dt series for
// displacement. Nonfinite input or negative rate/duration returns
// InvalidArgument; nonfinite output OutOfRange. No partial result on failure.
absl::StatusOr<WhitewaterKinematics3D> IntegrateWhitewaterDrag3D(
    WhitewaterKinematics3D initial,
    Vec3 carrier_velocity,
    Vec3 acceleration,
    double drag_rate_per_second,
    double duration_seconds
);
// Classifies signed liquid distance (positive air) using foam thresholds ±r
// and a 0.5r hysteresis margin. Supports direct spray/bubble transitions for
// large displacements. Invalid kind/distance/radius returns InvalidArgument.
absl::StatusOr<WhitewaterKind3D>
ClassifyWhitewater3D(WhitewaterKind3D previous, double signed_distance_m, double radius_m);
struct WhitewaterBirthEvent3D {
  double time_seconds;  // Accepted quadrature timestamp; births are not frame-end events.
  Vec3 position;
  Vec3 velocity;
  Vec3 normal;  // Finite unit surface normal.
  double volume_time_m3_seconds;  // Occupied source volume * potential * accepted dt.
};
struct WhitewaterEmissionState3D {
  std::uint64_t next_id = 1;  // Single emitter's never-reused identity counter.
  double remainder = 0;  // Fractional represented count in [0,1).
};
struct WhitewaterBirthPoint3D {
  std::uint64_t id;
  double birth_time_seconds;
  Vec3 position;
  Vec3 velocity;
};
struct WhitewaterEmission3D {
  WhitewaterEmissionState3D state;
  std::vector<WhitewaterBirthPoint3D> points{};
};
// Emits one event without averaging its velocity with any other event. Computes
// floor(q*volume_time/represented_count + remainder) points at the quadrature
// timestamp and returns the new remainder/ID counter with those points. The
// state is for one source emitter; callers with multiple cells must keep
// separate remainders and disjoint/global IDs. Positive rate has units m^-3 s^-1.
// represented_count is positive physical-particle count per output point.
// Invalid arguments return InvalidArgument, overflow OutOfRange, count/ID limits
// ResourceExhausted. No input mutation, including failure.
absl::StatusOr<WhitewaterEmission3D> EmitWhitewaterEvent3D(
    const WhitewaterBirthEvent3D& event,
    double rate_per_m3_second,
    double represented_count,
    WhitewaterEmissionState3D state,
    std::size_t maximum_points
);
}  // namespace fluidity
