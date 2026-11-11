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
struct WhitewaterCellBirth3D {
  CellIndex3D source_cell{};  // Signed coordinate on the fixed physical source grid.
  WhitewaterBirthEvent3D event{};
};
struct AcceptedWhitewaterBirth3D {
  std::uint64_t accepted_step_ordinal = 0;
  WhitewaterCellBirth3D birth{};
};
struct WhitewaterOccupiedCell3D {
  WhitewaterSourceCell3D sample{};  // Authoritative carrier/normal/curvature at quadrature time.
  double occupied_volume_m3 = 0;  // Positive liquid volume, bounded by one source cell.
};
struct WhitewaterSourceStep3D {
  double start_time_seconds = 0;
  double end_time_seconds = 0;
  double quadrature_time_seconds = 0;
};

// Builds positive birth events for one accepted step from the complete sparse
// set of eligible occupied source cells at quadrature time. Input cells are
// unique and ordered by signed z/y/x. An omitted neighbor is ineligible; a
// supplied cell with missing carrier fails. Uses grid origin (lower cell
// corner) and physical spacing for cell centres, and Vc*Phi*(end-start) for
// event weight. Returns the same cell order and no zero-weight events.
// Invalid records/time/controls return InvalidArgument, missing carrier
// FailedPrecondition, unrepresentable position/weight OutOfRange, more than
// 4096 source cells or too many births ResourceExhausted. No input mutation.
absl::StatusOr<std::vector<WhitewaterCellBirth3D>> ComputeWhitewaterBirths3D(
    std::span<const WhitewaterOccupiedCell3D> cells,
    Vec3 world_grid_origin,
    WhitewaterSourceStep3D step,
    const WhitewaterSourceOptions3D& options,
    std::size_t maximum_births = 4096
);

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

struct WhitewaterCellRemainder3D {
  CellIndex3D source_cell{};
  double fractional_count = 0;  // Nonzero remainder in (0,1), dimensionless.
};
struct WhitewaterEmissionCheckpoint3D {
  std::uint64_t next_point_id = 1;
  std::uint64_t last_accepted_step_ordinal = 0;
  std::vector<WhitewaterCellRemainder3D> remainders{};  // Unique signed z/y/x order.
};
struct WhitewaterBirthBatch3D {
  WhitewaterEmissionCheckpoint3D checkpoint{};
  std::vector<WhitewaterBirthPoint3D> points{};  // Birth order, increasing global point ID.
};

// Consumes accepted births after the checkpoint through a strictly later
// accepted step (inclusive). Events are ordered by accepted step, and each
// step has one timestamp and unique signed z/y/x cells. Keeps an independent
// fractional count per source cell and assigns one global non-reusing ID
// sequence in event order. Empty event intervals still advance the checkpoint.
// Invalid or repeated records/controls return InvalidArgument, count overflow
// OutOfRange, and configured or hard source/point limits ResourceExhausted.
// At most 2^20 input births/points and 4096 source cells are supported. Inputs
// are borrowed and unchanged, including on failure. Does not advect points.
absl::StatusOr<WhitewaterBirthBatch3D> EmitWhitewaterBirthsThroughStep3D(
    std::span<const AcceptedWhitewaterBirth3D> births,
    const WhitewaterEmissionCheckpoint3D& previous,
    std::uint64_t completed_accepted_step_ordinal,
    double rate_per_m3_second,
    double represented_count,
    std::size_t maximum_points,
    std::size_t maximum_source_cells = 4096
);
}  // namespace fluidity
