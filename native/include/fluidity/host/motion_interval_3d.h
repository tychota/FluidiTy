#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "fluidity/host/liquid_state_3d.h"

namespace fluidity {

struct MotionStamp3D {
  double time_seconds;  // Accepted simulation time in seconds.
  std::uint64_t accepted_epoch;  // Simulate's coherent accepted-state identity.
};

class MotionInterval3D {
 public:
  // Owns an ordered copy of at least two stamps supplied by the accepted-state producer.
  // They must span adjacent completed frames at the caller's output cadence.
  // Returns InvalidArgument if times are nonfinite, negative or nonincreasing, epochs are
  // nonincreasing, or the first and last stamps differ from the supplied frame time or epoch.
  static absl::StatusOr<MotionInterval3D>
  Create(LiquidClock3D first_frame, LiquidClock3D last_frame, std::span<const MotionStamp3D> stamps);

  // Borrows the stamps in chronological order while this interval stays alive and unmoved.
  std::span<const MotionStamp3D> stamps() const;

 private:
  explicit MotionInterval3D(std::span<const MotionStamp3D> stamps);

  std::vector<MotionStamp3D> stamps_;
};

// A read-only query of the spatial carrier at one ordered stamp. Position is in world metres.
// Return the velocity in m/s on success, or a failing status if the query has no valid support.
using MotionSpatialSampler3D = absl::FunctionRef<absl::StatusOr<Vec3>(std::size_t stamp_index, Vec3 world_position)>;

// Samples at world_position and time_seconds using one spatial sample at an exact stamp or linear
// interpolation of the two bracketing spatial velocities. Never extrapolates. The sampler is
// borrowed through the call. Propagates spatial errors, rejects nonfinite positions with
// InvalidArgument, and returns OutOfRange for out-of-interval times or nonfinite velocities.
// A moved-from interval returns FailedPrecondition if it no longer owns stamps.
absl::StatusOr<Vec3> SampleMotionCarrier3D(
    const MotionInterval3D& interval,
    Vec3 world_position,
    double time_seconds,
    MotionSpatialSampler3D sample_spatially
);

struct MotionAdvection3D {
  Vec3 world_position;  // End position in metres.
  std::size_t accepted_substeps;
  double maximum_displacement_m;  // Largest accepted substep displacement.
};

// Transports a world point with RK2 from from_time_seconds to to_time_seconds inside the interval.
// Splits at sample timestamps and limits each accepted displacement to half the positive finite
// cell_width_m. Zero duration returns the input point without a query or a substep. Propagates
// spatial errors and returns ResourceExhausted if the substep budget or refinement cannot advance.
// Invalid controls return InvalidArgument. Nonfinite intermediate geometry returns OutOfRange.
// The input interval must still own its stamps. Query errors propagate unchanged.
// No result is returned on failure. The sampler is borrowed through the call.
absl::StatusOr<MotionAdvection3D> AdvectMotionPoint3D(
    const MotionInterval3D& interval,
    Vec3 world_position,
    double from_time_seconds,
    double to_time_seconds,
    double cell_width_m,
    std::size_t maximum_substeps,
    MotionSpatialSampler3D sample_spatially
);

}  // namespace fluidity
