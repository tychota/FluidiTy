#include "fluidity/host/whitewater_frame_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "fluidity/host/motion_carrier_3d.h"
#include "fluidity/host/whitewater_sphere_source_3d.h"
#include "liquid_substep_3d_internal.h"

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

absl::StatusOr<WhitewaterFrameResult3D> ReconstructWhitewaterFrame3D(
    const MotionFrame3D& source,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& step_options,
    const WhitewaterFrame3D* previous,
    const WhitewaterFrameOptions3D& options
) {
  const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field);
  const auto* union_field = std::get_if<LiquidParticleSphereUnion3D>(&scene.field);
  const auto* beach = std::get_if<TranslatingHalfspace3D>(&scene.obstacle);
  if ((!sphere && !union_field)
      || (!std::holds_alternative<std::monostate>(scene.obstacle) && !IsStationaryWhitewaterBeach3D(scene.obstacle))) {
    return absl::UnimplementedError("whitewater frame requires sphere fields and stationary beach or no wall");
  }
  const double field_radius_m = sphere ? sphere->radius : union_field->radius;
  if (source.samples.size() > 65) {
    return absl::ResourceExhaustedError("whitewater motion sample budget exceeded");
  }
  if (source.samples.empty() || source.samples.front().clock != source.first_clock
      || source.samples.back().clock != source.last_clock || (sphere && !Finite(sphere->world_center))
      || !std::isfinite(field_radius_m) || field_radius_m <= 0 || !std::isfinite(options.emission_rate_per_m3_second)
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
    if (beach) {
      const double clearance_m = liquid_step_internal::SolidDistance(scene.obstacle, position);
      const double margin_m = 1e-9 * first_sample.geometry.cell_width;
      if (clearance_m < margin_m) {
        position = liquid_step_internal::Add(
            position,
            liquid_step_internal::Scale(beach->unit_normal, margin_m - clearance_m)
        );
      }
    }
    const auto sample = carriers[stamp].Sample(position);
    if (!sample.ok()) {
      return sample.status();
    }
    return sample->velocity;
  };
  const auto signed_distance = [&](Vec3 position, double time_s) -> absl::StatusOr<double> {
    double distance_m = 0;
    if (sphere) {
      distance_m = liquid_step_internal::FieldDistance(scene.field, {}, position);
    } else {
      const auto upper = std::ranges::lower_bound(stamps, time_s, {}, &MotionStamp3D::time_seconds);
      if (upper == stamps.end()) {
        return absl::OutOfRangeError("whitewater distance is beyond its accepted samples");
      }
      const auto index = static_cast<std::size_t>(upper - stamps.begin());
      const auto sample_distance = [&](std::size_t stamp) {
        const auto particles = source.samples[stamp].particles.values();
        // An empty accepted liquid state is known air within this frame.
        return particles.empty() ? 2 * options.transport.classification_radius_m
                                 : liquid_step_internal::FieldDistance(scene.field, particles, position);
      };
      distance_m = sample_distance(index);
      if (upper != stamps.begin() && upper->time_seconds != time_s) {
        const double lower_time_s = stamps[index - 1].time_seconds;
        const double fraction = (time_s - lower_time_s) / (upper->time_seconds - lower_time_s);
        distance_m = std::lerp(sample_distance(index - 1), distance_m, fraction);
      }
    }
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
    const auto moved =
        AdvanceWhitewaterPoint3D(record.point, *interval, transport, sample_carrier, signed_distance, beach);
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
