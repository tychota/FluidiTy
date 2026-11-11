#include "fluidity/host/whitewater_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

namespace fluidity {
namespace {
bool Finite(Vec3 p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
}  // namespace

absl::StatusOr<WhitewaterSourceStatistics3D> ComputeWhitewaterSource3D(
    const WhitewaterSourceCell3D& center,
    std::span<const WhitewaterSourceCell3D> neighbors,
    const WhitewaterSourceOptions3D& options
) {
  const auto valid_ramp = [](WhitewaterRamp3D ramp) {
    return std::isfinite(ramp.low) && std::isfinite(ramp.high) && ramp.low >= 0 && ramp.high > ramp.low;
  };
  if (!std::isfinite(options.spacing_m) || options.spacing_m <= 0 || !std::isfinite(options.radius_m)
      || options.radius_m <= 0 || !valid_ramp(options.kinetic) || !valid_ramp(options.trapped_air)
      || !valid_ramp(options.crest) || !std::isfinite(options.trapped_air_weight) || options.trapped_air_weight < 0
      || !std::isfinite(options.crest_weight) || options.crest_weight < 0 || neighbors.size() > 27) {
    return absl::InvalidArgumentError("invalid whitewater physical source settings");
  }
  const auto valid_cell = [](const WhitewaterSourceCell3D& cell) {
    return Finite(cell.velocity) && Finite(cell.normal) && std::isfinite(cell.curvature_per_m)
        && std::abs(std::hypot(cell.normal.x, cell.normal.y, cell.normal.z) - 1) <= 1e-8;
  };
  if (!valid_cell(center)) {
    return absl::InvalidArgumentError("invalid whitewater source geometry");
  }
  if (!center.carrier_valid) {
    return absl::FailedPreconditionError("whitewater source has no valid carrier");
  }
  const double speed = std::hypot(center.velocity.x, center.velocity.y, center.velocity.z);
  const double outward_speed =
      center.velocity.x * center.normal.x + center.velocity.y * center.normal.y + center.velocity.z * center.normal.z;
  WhitewaterSourceStatistics3D result{
      .specific_energy_m2_per_s2 = 0.5 * speed * speed,
      .trapped_air_m_per_s = 0,
      .crest_per_m = speed > 0 && outward_speed >= 0.6 * speed ? std::max(0.0, center.curvature_per_m) : 0,
      .potential = 0,
  };
  std::set<std::tuple<std::int32_t, std::int32_t, std::int32_t>> seen;
  for (const auto& neighbor : neighbors) {
    const auto index = neighbor.index;
    if (!seen.emplace(index.z, index.y, index.x).second) {
      return absl::InvalidArgumentError("duplicate whitewater source neighbor");
    }
    const std::array<std::int64_t, 3> offset{
        static_cast<std::int64_t>(center.index.x) - index.x,
        static_cast<std::int64_t>(center.index.y) - index.y,
        static_cast<std::int64_t>(center.index.z) - index.z,
    };
    if (offset == std::array<std::int64_t, 3>{}) {
      continue;
    }
    if (std::any_of(
            offset.begin(),
            offset.end(),
            [](auto coordinate) {
              return coordinate < -1 || coordinate > 1;
            }
        )
        || !valid_cell(neighbor)) {
      return absl::InvalidArgumentError("invalid whitewater source neighbor");
    }
    if (!neighbor.carrier_valid) {
      return absl::FailedPreconditionError("whitewater neighbor has no valid carrier");
    }
    const double index_distance = std::hypot(offset[0], offset[1], offset[2]);
    const double kernel_weight = std::max(0.0, 1 - 0.5 * index_distance * (options.spacing_m / options.radius_m));
    const Vec3 relative{
        center.velocity.x - neighbor.velocity.x,
        center.velocity.y - neighbor.velocity.y,
        center.velocity.z - neighbor.velocity.z
    };
    const double directed = (relative.x * offset[0] + relative.y * offset[1] + relative.z * offset[2]) / index_distance;
    const double contribution = std::max(0.0, std::hypot(relative.x, relative.y, relative.z) - directed);
    result.trapped_air_m_per_s += contribution * kernel_weight;
  }
  const auto ramp = [](double value, WhitewaterRamp3D thresholds) {
    return std::clamp((value - thresholds.low) / (thresholds.high - thresholds.low), 0.0, 1.0);
  };
  result.potential = ramp(result.specific_energy_m2_per_s2, options.kinetic)
      * (options.trapped_air_weight * ramp(result.trapped_air_m_per_s, options.trapped_air)
         + options.crest_weight * ramp(result.crest_per_m, options.crest));
  if (!std::isfinite(result.specific_energy_m2_per_s2) || !std::isfinite(result.trapped_air_m_per_s)
      || !std::isfinite(result.potential)) {
    return absl::OutOfRangeError("whitewater source statistics overflowed");
  }
  return result;
}

absl::StatusOr<WhitewaterKinematics3D> IntegrateWhitewaterDrag3D(
    WhitewaterKinematics3D initial,
    Vec3 carrier,
    Vec3 acceleration,
    double rate,
    double duration
) {
  if (!Finite(initial.position) || !Finite(initial.velocity) || !Finite(carrier) || !Finite(acceleration)
      || !std::isfinite(rate) || rate < 0 || !std::isfinite(duration) || duration < 0) {
    return absl::InvalidArgumentError("invalid frozen whitewater forcing");
  }
  if (duration == 0) {
    return initial;
  }
  const double scaled_duration = rate * duration;
  const double decay = std::exp(-scaled_duration);
  double velocity_integral = duration;
  double acceleration_integral = 0.5 * duration * duration;
  if (rate > 0 && scaled_duration > 0) {
    velocity_integral = -std::expm1(-scaled_duration) / rate;
    if (scaled_duration < 1e-3) {
      const double z = scaled_duration;
      const double series = 0.5 + z * (-1.0 / 6 + z * (1.0 / 24 + z * (-1.0 / 120 + z * (1.0 / 720 - z / 5040))));
      acceleration_integral = duration * duration * series;
    } else {
      acceleration_integral = (duration - velocity_integral) / rate;
    }
  }
  const auto advance = [&](double x, double v, double u, double a) {
    // v*dt + (u-v)*(dt-A) avoids cancellation between carrier terms near k=0.
    return x + v * velocity_integral + u * (duration - velocity_integral) + a * acceleration_integral;
  };
  const auto velocity = [&](double v, double u, double a) {
    return v * decay + u * (-std::expm1(-scaled_duration)) + a * velocity_integral;
  };
  WhitewaterKinematics3D result{
      .position =
          {advance(initial.position.x, initial.velocity.x, carrier.x, acceleration.x),
           advance(initial.position.y, initial.velocity.y, carrier.y, acceleration.y),
           advance(initial.position.z, initial.velocity.z, carrier.z, acceleration.z)},
      .velocity =
          {velocity(initial.velocity.x, carrier.x, acceleration.x),
           velocity(initial.velocity.y, carrier.y, acceleration.y),
           velocity(initial.velocity.z, carrier.z, acceleration.z)},
  };
  if (!Finite(result.position) || !Finite(result.velocity)) {
    return absl::OutOfRangeError("whitewater drag overflowed");
  }
  return result;
}

absl::StatusOr<WhitewaterKind3D> ClassifyWhitewater3D(WhitewaterKind3D previous, double distance, double radius) {
  if (previous > WhitewaterKind3D::kBubble || !std::isfinite(distance) || !std::isfinite(radius) || radius <= 0) {
    return absl::InvalidArgumentError("invalid whitewater classification");
  }
  const double depth = distance / radius;
  if (previous == WhitewaterKind3D::kSpray && depth > 0.5) {
    return previous;
  }
  if (previous == WhitewaterKind3D::kBubble && depth < -0.5) {
    return previous;
  }
  if (depth > 1.5) {
    return WhitewaterKind3D::kSpray;
  }
  if (depth < -1.5) {
    return WhitewaterKind3D::kBubble;
  }
  return WhitewaterKind3D::kFoam;
}

absl::StatusOr<WhitewaterEmission3D> EmitWhitewaterEvent3D(
    const WhitewaterBirthEvent3D& event,
    double rate,
    double represented_count,
    WhitewaterEmissionState3D state,
    std::size_t maximum_points
) {
  if (!std::isfinite(event.time_seconds) || event.time_seconds < 0 || !Finite(event.position) || !Finite(event.velocity)
      || !Finite(event.normal) || std::abs(std::hypot(event.normal.x, event.normal.y, event.normal.z) - 1) > 1e-8
      || !std::isfinite(event.volume_time_m3_seconds) || event.volume_time_m3_seconds < 0 || !std::isfinite(rate)
      || rate <= 0 || !std::isfinite(represented_count) || represented_count <= 0 || state.next_id == 0
      || !std::isfinite(state.remainder) || state.remainder < 0 || state.remainder >= 1) {
    return absl::InvalidArgumentError("invalid whitewater event or emission state");
  }
  const double expected = rate * event.volume_time_m3_seconds / represented_count + state.remainder;
  if (!std::isfinite(expected)) {
    return absl::OutOfRangeError("whitewater emission count overflowed");
  }
  const double count_value = std::floor(expected);
  if (count_value > static_cast<double>(std::min(maximum_points, std::size_t{1} << 20))) {
    return absl::ResourceExhaustedError("whitewater birth budget exceeded");
  }
  const auto count = static_cast<std::size_t>(count_value);
  if (count > std::numeric_limits<std::uint64_t>::max() - state.next_id) {
    return absl::ResourceExhaustedError("whitewater IDs exhausted");
  }
  WhitewaterEmission3D result{.state = {.next_id = state.next_id + count, .remainder = expected - count_value}};
  result.points.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    result.points.push_back(
        {.id = state.next_id + i,
         .birth_time_seconds = event.time_seconds,
         .position = event.position,
         .velocity = event.velocity}
    );
  }
  return result;
}
}  // namespace fluidity
