#include "fluidity/host/whitewater_frame_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "fluidity/host/motion_carrier_3d.h"

namespace fluidity {
namespace {
bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
bool SameGrid(UniformMacGeometry3D a, UniformMacGeometry3D b) {
  return a.cell_width == b.cell_width && a.world_origin.x == b.world_origin.x && a.world_origin.y == b.world_origin.y
      && a.world_origin.z == b.world_origin.z;
}
}  // namespace

absl::StatusOr<WhitewaterFrameResult3D> ReconstructWhitewaterSphereFrame3D(
    const MotionFrame3D& source,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& step_options,
    const WhitewaterFrame3D* previous,
    const WhitewaterFrameOptions3D& options
) {
  const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field);
  if (sphere == nullptr || !std::holds_alternative<std::monostate>(scene.obstacle)) {
    return absl::UnimplementedError("whitewater frame requires the analytic sphere without obstacles");
  }
  if (source.samples.size() > 65) {
    return absl::ResourceExhaustedError("whitewater motion sample budget exceeded");
  }
  if (source.samples.empty() || source.samples.front().clock != source.first_clock
      || source.samples.back().clock != source.last_clock || !Finite(sphere->world_center)
      || !std::isfinite(sphere->radius) || sphere->radius <= 0 || !std::isfinite(options.emission_rate_per_m3_second)
      || options.emission_rate_per_m3_second <= 0 || !std::isfinite(options.represented_count_per_point)
      || options.represented_count_per_point <= 0 || !std::isfinite(options.transport.classification_radius_m)
      || options.transport.classification_radius_m <= 0) {
    return absl::InvalidArgumentError("invalid whitewater frame source or physical controls");
  }
  WhitewaterFrameResult3D result;
  result.frame.clock = source.last_clock;
  if (source.last_clock == LiquidClock3D{}) {
    if (previous != nullptr || source.samples.size() != 1 || !source.whitewater_births.empty()) {
      return absl::InvalidArgumentError("invalid whitewater seed frame");
    }
    return result;
  }
  if (previous == nullptr || previous->clock != source.first_clock
      || previous->emission.last_accepted_step_ordinal != source.first_clock.accepted_step_ordinal) {
    return absl::FailedPreconditionError("whitewater interval lacks its exact predecessor");
  }
  const auto maximum_points = std::min(options.maximum_points, std::size_t{1} << 20);
  if (previous->points.size() > maximum_points) {
    return absl::ResourceExhaustedError("whitewater previous point budget exceeded");
  }
  std::uint64_t previous_id = 0;
  for (const auto& record : previous->points) {
    if (record.id <= previous_id || record.id >= previous->emission.next_point_id
        || record.point.birth_time_seconds > previous->clock.time) {
      return absl::InvalidArgumentError("invalid whitewater checkpoint point identity or birth time");
    }
    previous_id = record.id;
  }
  std::vector<MotionStamp3D> stamps;
  stamps.reserve(source.samples.size());
  const auto& first_sample = source.samples.front();
  for (const auto& sample : source.samples) {
    if (!SameGrid(sample.geometry, first_sample.geometry) || sample.random_seed != first_sample.random_seed) {
      return absl::InvalidArgumentError("whitewater accepted sample grid or seed differs");
    }
    if (sample.particles.size() > 1024 || sample.blocks.size() > 128 || sample.faces.size() > 24576) {
      return absl::ResourceExhaustedError("whitewater accepted carrier record budget exceeded");
    }
    for (const auto& particle : sample.particles.values()) {
      if (particle.side_tag != 0) {
        return absl::UnimplementedError("whitewater sphere transport requires uniform material sides");
      }
    }
    stamps.push_back({sample.clock.time, sample.clock.epoch});
  }
  const auto interval = MotionInterval3D::Create(source.first_clock, source.last_clock, stamps);
  if (!interval.ok()) {
    return interval.status();
  }
  for (const auto& birth : source.whitewater_births) {
    if (birth.birth.event.time_seconds < source.first_clock.time
        || birth.birth.event.time_seconds > source.last_clock.time) {
      return absl::InvalidArgumentError("whitewater birth time lies outside its accepted frame");
    }
  }
  auto emitted = EmitWhitewaterBirthsThroughStep3D(
      source.whitewater_births,
      previous->emission,
      source.last_clock.accepted_step_ordinal,
      options.emission_rate_per_m3_second,
      options.represented_count_per_point,
      maximum_points - previous->points.size(),
      options.maximum_source_cells
  );
  if (!emitted.ok()) {
    return emitted.status();
  }
  result.frame.emission = std::move(emitted->checkpoint);
  result.birth_count = emitted->points.size();
  if (previous->points.empty() && emitted->points.empty()) {
    return result;
  }
  std::vector<FrozenMotionCarrier3D> carriers;
  carriers.reserve(source.samples.size());
  for (const auto& sample : source.samples) {
    auto carrier = FrozenMotionCarrier3D::Create(sample, scene, step_options);
    if (!carrier.ok()) {
      return carrier.status();
    }
    carriers.push_back(std::move(*carrier));
  }
  const auto sample_carrier = [&](std::size_t stamp, Vec3 position) -> absl::StatusOr<Vec3> {
    const auto sample = carriers[stamp].Sample(position);
    if (!sample.ok()) {
      return sample.status();
    }
    return sample->velocity;
  };
  const auto signed_distance = [&](Vec3 position, double) -> absl::StatusOr<double> {
    const double distance_m = std::hypot(
                                  position.x - sphere->world_center.x,
                                  position.y - sphere->world_center.y,
                                  position.z - sphere->world_center.z
                              )
        - sphere->radius;
    if (!std::isfinite(distance_m)) {
      return absl::OutOfRangeError("whitewater sphere distance overflowed");
    }
    return distance_m;
  };
  auto transport = options.transport;
  transport.cell_width_m = first_sample.geometry.cell_width;
  transport.gravity = scene.world_acceleration;
  result.frame.points.reserve(previous->points.size() + emitted->points.size());
  const auto advance = [&](const WhitewaterPointRecord3D& record) -> absl::Status {
    const auto moved = AdvanceWhitewaterPoint3D(record.point, *interval, transport, sample_carrier, signed_distance);
    if (!moved.ok()) {
      return moved.status();
    }
    if (moved->accepted_substeps > std::numeric_limits<std::size_t>::max() - result.transport_substeps) {
      return absl::OutOfRangeError("whitewater transport measurement overflowed");
    }
    result.transport_substeps += moved->accepted_substeps;
    result.bubble_fallback_substeps += moved->bubble_fallback_substeps;
    if (moved->expired) {
      ++result.expired_count;
    } else {
      result.frame.points.push_back({.id = record.id, .point = moved->point});
    }
    return absl::OkStatus();
  };
  for (const auto& record : previous->points) {
    const auto status = advance(record);
    if (!status.ok()) {
      return status;
    }
  }
  const double radius_m = transport.classification_radius_m;
  for (const auto& birth : emitted->points) {
    const auto distance_m = signed_distance(birth.position, birth.birth_time_seconds);
    if (!distance_m.ok()) {
      return distance_m.status();
    }
    auto kind = WhitewaterKind3D::kFoam;
    if (*distance_m > radius_m) {
      kind = WhitewaterKind3D::kSpray;
    } else if (*distance_m < -radius_m) {
      kind = WhitewaterKind3D::kBubble;
    }
    const auto status = advance({
        .id = birth.id,
        .point =
            {
                .kinematics = {.position = birth.position, .velocity = birth.velocity},
                .kind = kind,
                .birth_time_seconds = birth.birth_time_seconds,
            },
    });
    if (!status.ok()) {
      return status;
    }
  }
  return result;
}
}  // namespace fluidity
