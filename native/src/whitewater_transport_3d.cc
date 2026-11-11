#include "fluidity/host/whitewater_transport_3d.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

namespace fluidity {
namespace {
bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
bool Nonnegative(double value) {
  return std::isfinite(value) && value >= 0;
}

struct DragSubstep {
  WhitewaterKinematics3D kinematics;
  double end_time_seconds;
  double displacement_m;
};

// Retry only time, with the same frozen start forcing. A rejected displacement
// never alters the point or its phase.
absl::StatusOr<DragSubstep> BoundedDragSubstep(
    WhitewaterKinematics3D initial,
    Vec3 carrier_velocity,
    Vec3 acceleration,
    double drag_per_second,
    double start_time_seconds,
    double end_time_seconds,
    double displacement_limit_m
) {
  for (std::size_t refinement = 0; refinement <= 32 && end_time_seconds > start_time_seconds; ++refinement) {
    auto candidate = IntegrateWhitewaterDrag3D(
        initial,
        carrier_velocity,
        acceleration,
        drag_per_second,
        end_time_seconds - start_time_seconds
    );
    if (!candidate.ok()) {
      return candidate.status();
    }
    const double displacement_m = std::hypot(
        candidate->position.x - initial.position.x,
        candidate->position.y - initial.position.y,
        candidate->position.z - initial.position.z
    );
    if (displacement_m <= displacement_limit_m) {
      return DragSubstep{
          .kinematics = *candidate,
          .end_time_seconds = end_time_seconds,
          .displacement_m = displacement_m,
      };
    }
    end_time_seconds = std::midpoint(start_time_seconds, end_time_seconds);
  }
  return absl::ResourceExhaustedError("whitewater drag displacement cannot be resolved");
}

absl::Status FoamCarrierStatus(const absl::Status& status) {
  return absl::IsFailedPrecondition(status)
      ? absl::FailedPreconditionError("whitewater foam is missing required carrier")
      : status;
}
}  // namespace

absl::StatusOr<WhitewaterTransportResult3D> AdvanceWhitewaterPoint3D(
    const WhitewaterTransportPoint3D& point,
    const MotionInterval3D& interval,
    const WhitewaterTransportOptions3D& options,
    MotionSpatialSampler3D carrier,
    WhitewaterDistanceSampler3D liquid_distance
) {
  const auto stamps = interval.stamps();
  if (stamps.size() < 2) {
    return absl::FailedPreconditionError("whitewater transport has no accepted motion interval");
  }
  const double half_cell_width_m = 0.5 * options.cell_width_m;
  const std::size_t maximum_substeps = std::min(options.maximum_substeps, std::size_t{1} << 20);
  if (!Finite(point.kinematics.position) || !Finite(point.kinematics.velocity) || point.kind > WhitewaterKind3D::kBubble
      || !Nonnegative(point.birth_time_seconds) || point.birth_time_seconds > stamps.back().time_seconds
      || !std::isfinite(half_cell_width_m) || half_cell_width_m <= 0 || !std::isfinite(options.classification_radius_m)
      || options.classification_radius_m <= 0 || !Nonnegative(options.spray_drag_per_second)
      || !Nonnegative(options.bubble_drag_per_second) || !Nonnegative(options.bubble_buoyancy_scale)
      || !std::isfinite(options.foam_lifetime_seconds) || options.foam_lifetime_seconds <= 0 || !Finite(options.gravity)
      || !Finite(options.ambient_velocity) || maximum_substeps == 0) {
    return absl::InvalidArgumentError("invalid whitewater point or transport controls");
  }
  const double foam_expiry_time_seconds = point.birth_time_seconds + options.foam_lifetime_seconds;
  if (!std::isfinite(foam_expiry_time_seconds) || foam_expiry_time_seconds <= point.birth_time_seconds) {
    return absl::OutOfRangeError("whitewater foam expiry time is unrepresentable");
  }
  const Vec3 bubble_acceleration{
      -options.bubble_buoyancy_scale * options.gravity.x,
      -options.bubble_buoyancy_scale * options.gravity.y,
      -options.bubble_buoyancy_scale * options.gravity.z,
  };
  if (!Finite(bubble_acceleration)) {
    return absl::OutOfRangeError("whitewater bubble acceleration overflowed");
  }
  WhitewaterTransportResult3D result{.point = point};
  auto& current = result.point;
  double time_seconds = std::max(stamps.front().time_seconds, point.birth_time_seconds);
  for (const auto stamp : stamps) {
    while (time_seconds < stamp.time_seconds) {
      const std::size_t remaining_substeps = maximum_substeps - result.accepted_substeps;
      if (remaining_substeps == 0) {
        return absl::ResourceExhaustedError("whitewater transport substep budget exceeded");
      }
      std::optional<Vec3> endpoint_carrier;
      bool bubble_missing_carrier = false;
      if (current.kind == WhitewaterKind3D::kFoam) {
        if (time_seconds >= foam_expiry_time_seconds) {
          result.expired = true;
          return result;
        }
        const double foam_step_end = std::min(stamp.time_seconds, foam_expiry_time_seconds);
        const auto advected = AdvectMotionPoint3D(
            interval,
            current.kinematics.position,
            time_seconds,
            foam_step_end,
            options.cell_width_m,
            remaining_substeps,
            carrier
        );
        if (!advected.ok()) {
          return FoamCarrierStatus(advected.status());
        }
        current.kinematics.position = advected->world_position;
        time_seconds = foam_step_end;
        result.accepted_substeps += advected->accepted_substeps;
        result.maximum_displacement_m = std::max(result.maximum_displacement_m, advected->maximum_displacement_m);
        if (time_seconds >= foam_expiry_time_seconds) {
          result.expired = true;
          return result;
        }
        const auto sampled = SampleMotionCarrier3D(interval, current.kinematics.position, time_seconds, carrier);
        if (!sampled.ok()) {
          return FoamCarrierStatus(sampled.status());
        }
        endpoint_carrier = *sampled;
        current.kinematics.velocity = *sampled;
      } else {
        const bool bubble = current.kind == WhitewaterKind3D::kBubble;
        Vec3 frozen_carrier = options.ambient_velocity;
        if (bubble) {
          const auto sampled = SampleMotionCarrier3D(interval, current.kinematics.position, time_seconds, carrier);
          if (!sampled.ok() && !absl::IsFailedPrecondition(sampled.status())) {
            return sampled.status();
          }
          bubble_missing_carrier = !sampled.ok();
          frozen_carrier = sampled.ok() ? *sampled : Vec3{};
        }
        const auto step = BoundedDragSubstep(
            current.kinematics,
            frozen_carrier,
            bubble ? bubble_acceleration : options.gravity,
            bubble ? options.bubble_drag_per_second : options.spray_drag_per_second,
            time_seconds,
            stamp.time_seconds,
            half_cell_width_m
        );
        if (!step.ok()) {
          return step.status();
        }
        current.kinematics = step->kinematics;
        time_seconds = step->end_time_seconds;
        ++result.accepted_substeps;
        result.bubble_fallback_substeps += bubble_missing_carrier;
        result.maximum_displacement_m = std::max(result.maximum_displacement_m, step->displacement_m);
        if (bubble && !bubble_missing_carrier) {
          const auto sampled = SampleMotionCarrier3D(interval, current.kinematics.position, time_seconds, carrier);
          if (!sampled.ok() && !absl::IsFailedPrecondition(sampled.status())) {
            return sampled.status();
          }
          bubble_missing_carrier = !sampled.ok();
          if (sampled.ok()) {
            endpoint_carrier = *sampled;
          }
        }
      }
      if (bubble_missing_carrier) {
        continue;
      }
      const auto distance_m = liquid_distance(current.kinematics.position, time_seconds);
      if (!distance_m.ok()) {
        return distance_m.status();
      }
      const auto kind = ClassifyWhitewater3D(current.kind, *distance_m, options.classification_radius_m);
      if (!kind.ok()) {
        return kind.status();
      }
      current.kind = *kind;
      if (current.kind == WhitewaterKind3D::kFoam) {
        if (time_seconds >= foam_expiry_time_seconds) {
          result.expired = true;
          return result;
        }
        if (!endpoint_carrier) {
          const auto sampled = SampleMotionCarrier3D(interval, current.kinematics.position, time_seconds, carrier);
          if (!sampled.ok()) {
            return FoamCarrierStatus(sampled.status());
          }
          endpoint_carrier = *sampled;
        }
        current.kinematics.velocity = *endpoint_carrier;
      }
    }
  }
  return result;
}
}  // namespace fluidity
