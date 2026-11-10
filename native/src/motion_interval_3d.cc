#include "fluidity/host/motion_interval_3d.h"

#include <algorithm>
#include <cmath>

#include "absl/status/status.h"

namespace fluidity {
namespace {

bool IsFinite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

Vec3 OffsetPosition(Vec3 position, double duration_seconds, Vec3 velocity) {
  return {
      position.x + duration_seconds * velocity.x,
      position.y + duration_seconds * velocity.y,
      position.z + duration_seconds * velocity.z,
  };
}

absl::Status CheckCoveredTime(std::span<const MotionStamp3D> stamps, double time_seconds) {
  if (stamps.empty()) {
    return absl::FailedPreconditionError("motion interval has no stamps");
  }
  if (!std::isfinite(time_seconds) || time_seconds < stamps.front().time_seconds
      || time_seconds > stamps.back().time_seconds) {
    return absl::OutOfRangeError("motion query time is outside the interval");
  }
  return absl::OkStatus();
}

absl::StatusOr<Vec3>
SampleFiniteVelocity(MotionSpatialSampler3D sample_spatially, std::size_t stamp_index, Vec3 world_position) {
  auto velocity = sample_spatially(stamp_index, world_position);
  if (!velocity.ok()) {
    return velocity.status();
  }
  if (!IsFinite(*velocity)) {
    return absl::OutOfRangeError("spatial carrier velocity must be finite");
  }
  return *velocity;
}

}  // namespace

MotionInterval3D::MotionInterval3D(std::span<const MotionStamp3D> stamps) : stamps_(stamps.begin(), stamps.end()) {
}

absl::StatusOr<MotionInterval3D>
MotionInterval3D::Create(LiquidClock3D first_frame, LiquidClock3D last_frame, std::span<const MotionStamp3D> stamps) {
  const bool adjacent_frames =
      last_frame.frame_ordinal > first_frame.frame_ordinal && last_frame.frame_ordinal - first_frame.frame_ordinal == 1;
  if (!adjacent_frames || first_frame.substep_ordinal != 0 || last_frame.substep_ordinal != 0) {
    return absl::InvalidArgumentError("motion interval requires adjacent completed frames");
  }
  if (stamps.size() < 2) {
    return absl::InvalidArgumentError("motion interval requires at least two stamps");
  }
  const bool matching_endpoints = stamps.front().time_seconds == first_frame.time
      && stamps.back().time_seconds == last_frame.time && stamps.front().accepted_epoch == first_frame.epoch
      && stamps.back().accepted_epoch == last_frame.epoch;
  if (!matching_endpoints) {
    return absl::InvalidArgumentError("motion stamps must match frame times and epochs");
  }
  if (!std::isfinite(stamps.front().time_seconds) || stamps.front().time_seconds < 0) {
    return absl::InvalidArgumentError("motion time must be finite and nonnegative");
  }
  for (std::size_t index = 1; index < stamps.size(); ++index) {
    const MotionStamp3D& previous = stamps[index - 1];
    const MotionStamp3D& current = stamps[index];
    if (!std::isfinite(current.time_seconds) || current.time_seconds <= previous.time_seconds
        || current.accepted_epoch <= previous.accepted_epoch) {
      return absl::InvalidArgumentError("motion stamp times and epochs must be strictly increasing");
    }
  }
  return MotionInterval3D(stamps);
}

std::span<const MotionStamp3D> MotionInterval3D::stamps() const {
  return stamps_;
}

absl::StatusOr<Vec3> SampleMotionCarrier3D(
    const MotionInterval3D& interval,
    Vec3 world_position,
    double time_seconds,
    MotionSpatialSampler3D sample_spatially
) {
  if (!IsFinite(world_position)) {
    return absl::InvalidArgumentError("motion query position must be finite");
  }
  const auto stamps = interval.stamps();
  const absl::Status covered = CheckCoveredTime(stamps, time_seconds);
  if (!covered.ok()) {
    return covered;
  }
  const auto later =
      std::lower_bound(stamps.begin(), stamps.end(), time_seconds, [](const MotionStamp3D& stamp, double time) {
        return stamp.time_seconds < time;
      });
  const auto later_index = static_cast<std::size_t>(later - stamps.begin());
  if (later->time_seconds == time_seconds) {
    return SampleFiniteVelocity(sample_spatially, later_index, world_position);
  }
  const auto earlier_velocity = SampleFiniteVelocity(sample_spatially, later_index - 1, world_position);
  if (!earlier_velocity.ok()) {
    return earlier_velocity.status();
  }
  const auto later_velocity = SampleFiniteVelocity(sample_spatially, later_index, world_position);
  if (!later_velocity.ok()) {
    return later_velocity.status();
  }
  const double earlier_time_seconds = stamps[later_index - 1].time_seconds;
  const double weight = (time_seconds - earlier_time_seconds) / (later->time_seconds - earlier_time_seconds);
  return Vec3{
      std::lerp(earlier_velocity->x, later_velocity->x, weight),
      std::lerp(earlier_velocity->y, later_velocity->y, weight),
      std::lerp(earlier_velocity->z, later_velocity->z, weight),
  };
}

absl::StatusOr<MotionAdvection3D> AdvectMotionPoint3D(
    const MotionInterval3D& interval,
    Vec3 world_position,
    double from_time_seconds,
    double to_time_seconds,
    double cell_width_m,
    std::size_t maximum_substeps,
    MotionSpatialSampler3D sample_spatially
) {
  const double displacement_limit_m = 0.5 * cell_width_m;
  if (!IsFinite(world_position) || !std::isfinite(cell_width_m) || displacement_limit_m <= 0
      || to_time_seconds < from_time_seconds) {
    return absl::InvalidArgumentError("expected finite position, positive cell width and forward transport");
  }
  const auto stamps = interval.stamps();
  for (const double endpoint_seconds : {from_time_seconds, to_time_seconds}) {
    const absl::Status covered = CheckCoveredTime(stamps, endpoint_seconds);
    if (!covered.ok()) {
      return covered;
    }
  }
  MotionAdvection3D result{
      .world_position = world_position,
      .accepted_substeps = 0,
      .maximum_displacement_m = 0,
  };
  double time_seconds = from_time_seconds;
  while (time_seconds < to_time_seconds) {
    if (result.accepted_substeps == maximum_substeps) {
      return absl::ResourceExhaustedError("motion substep budget exhausted");
    }
    const auto next_stamp =
        std::upper_bound(stamps.begin(), stamps.end(), time_seconds, [](double time, const MotionStamp3D& stamp) {
          return time < stamp.time_seconds;
        });
    const double step_limit_seconds = std::min(next_stamp->time_seconds, to_time_seconds);
    const auto velocity = SampleMotionCarrier3D(interval, result.world_position, time_seconds, sample_spatially);
    if (!velocity.ok()) {
      return velocity.status();
    }
    const double speed_m_per_second = std::hypot(velocity->x, velocity->y, velocity->z);
    double duration_seconds = step_limit_seconds - time_seconds;
    if (speed_m_per_second > 0) {
      duration_seconds = std::min(duration_seconds, displacement_limit_m / speed_m_per_second);
    }
    // The midpoint speed can exceed the predictor speed, so test the actual world displacement.
    bool accepted = false;
    constexpr int kMaximumHalvings = 32;
    for (int halving = 0; halving <= kMaximumHalvings; ++halving) {
      const double next_time_seconds = std::min(time_seconds + duration_seconds, step_limit_seconds);
      if (next_time_seconds <= time_seconds) {
        break;
      }
      const double step_seconds = next_time_seconds - time_seconds;
      const double half_step_seconds = 0.5 * step_seconds;
      const Vec3 midpoint = OffsetPosition(result.world_position, half_step_seconds, *velocity);
      if (!IsFinite(midpoint)) {
        return absl::OutOfRangeError("motion predictor position is not finite");
      }
      const auto midpoint_velocity =
          SampleMotionCarrier3D(interval, midpoint, time_seconds + half_step_seconds, sample_spatially);
      if (!midpoint_velocity.ok()) {
        return midpoint_velocity.status();
      }
      const Vec3 next_position = OffsetPosition(result.world_position, step_seconds, *midpoint_velocity);
      if (!IsFinite(next_position)) {
        return absl::OutOfRangeError("motion result position is not finite");
      }
      const double displacement_m = std::hypot(
          next_position.x - result.world_position.x,
          next_position.y - result.world_position.y,
          next_position.z - result.world_position.z
      );
      if (displacement_m <= displacement_limit_m) {
        result.world_position = next_position;
        result.maximum_displacement_m = std::max(result.maximum_displacement_m, displacement_m);
        ++result.accepted_substeps;
        time_seconds = next_time_seconds;
        accepted = true;
        break;
      }
      duration_seconds *= 0.5;
    }
    if (!accepted) {
      return absl::ResourceExhaustedError("motion step refinement could not advance");
    }
  }
  return result;
}

}  // namespace fluidity
