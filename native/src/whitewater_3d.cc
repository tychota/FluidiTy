#include "fluidity/host/whitewater_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace fluidity {
namespace {
bool Finite(Vec3 p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
bool ValidSourceOptions(const WhitewaterSourceOptions3D& options) {
  const auto valid_ramp = [](WhitewaterRamp3D ramp) {
    return std::isfinite(ramp.low) && std::isfinite(ramp.high) && ramp.low >= 0 && ramp.high > ramp.low;
  };
  return std::isfinite(options.spacing_m) && options.spacing_m > 0 && std::isfinite(options.radius_m)
      && options.radius_m > 0 && valid_ramp(options.kinetic) && valid_ramp(options.trapped_air)
      && valid_ramp(options.crest) && std::isfinite(options.trapped_air_weight) && options.trapped_air_weight >= 0
      && std::isfinite(options.crest_weight) && options.crest_weight >= 0;
}
bool ValidSourceCell(const WhitewaterSourceCell3D& cell) {
  return Finite(cell.velocity) && Finite(cell.normal) && std::isfinite(cell.curvature_per_m)
      && std::abs(std::hypot(cell.normal.x, cell.normal.y, cell.normal.z) - 1) <= 1e-8;
}
auto CellOrder(CellIndex3D cell) {
  return std::tuple(cell.z, cell.y, cell.x);
}
}  // namespace

absl::StatusOr<WhitewaterSourceStatistics3D> ComputeWhitewaterSource3D(
    const WhitewaterSourceCell3D& center,
    std::span<const WhitewaterSourceCell3D> neighbors,
    const WhitewaterSourceOptions3D& options
) {
  if (!ValidSourceOptions(options) || neighbors.size() > 27) {
    return absl::InvalidArgumentError("invalid whitewater physical source settings");
  }
  if (!ValidSourceCell(center)) {
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
  std::uint32_t seen_offsets = 0;
  for (const auto& neighbor : neighbors) {
    const auto index = neighbor.index;
    const std::array<std::int64_t, 3> offset{
        static_cast<std::int64_t>(center.index.x) - index.x,
        static_cast<std::int64_t>(center.index.y) - index.y,
        static_cast<std::int64_t>(center.index.z) - index.z,
    };
    if (std::any_of(offset.begin(), offset.end(), [](auto coordinate) {
          return coordinate < -1 || coordinate > 1;
        })) {
      return absl::InvalidArgumentError("invalid whitewater source neighbor");
    }
    const auto offset_index = (offset[2] + 1) * 9 + (offset[1] + 1) * 3 + offset[0] + 1;
    const auto offset_bit = std::uint32_t{1} << offset_index;
    if ((seen_offsets & offset_bit) != 0) {
      return absl::InvalidArgumentError("duplicate whitewater source neighbor");
    }
    seen_offsets |= offset_bit;
    if (offset == std::array<std::int64_t, 3>{}) {
      continue;
    }
    if (!ValidSourceCell(neighbor)) {
      return absl::InvalidArgumentError("invalid whitewater source neighbor geometry");
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
    const double relative_speed = std::hypot(relative.x, relative.y, relative.z);
    if (!std::isfinite(directed) || !std::isfinite(relative_speed)) {
      return absl::OutOfRangeError("whitewater relative velocity overflowed");
    }
    const double contribution = std::max(0.0, relative_speed - directed);
    if (!std::isfinite(contribution)) {
      return absl::OutOfRangeError("whitewater trapped-air contribution overflowed");
    }
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

absl::StatusOr<std::vector<WhitewaterCellBirth3D>> ComputeWhitewaterBirths3D(
    std::span<const WhitewaterOccupiedCell3D> cells,
    Vec3 world_grid_origin,
    WhitewaterSourceStep3D step,
    const WhitewaterSourceOptions3D& options,
    std::size_t maximum_births
) {
  if (cells.size() > 4096) {
    return absl::ResourceExhaustedError("whitewater occupied source cell budget exceeded");
  }
  if (!ValidSourceOptions(options) || !Finite(world_grid_origin) || !std::isfinite(step.start_time_seconds)
      || step.start_time_seconds < 0 || !std::isfinite(step.end_time_seconds)
      || step.end_time_seconds <= step.start_time_seconds || !std::isfinite(step.quadrature_time_seconds)
      || step.quadrature_time_seconds < step.start_time_seconds
      || step.quadrature_time_seconds > step.end_time_seconds) {
    return absl::InvalidArgumentError("invalid whitewater source grid or accepted step time");
  }
  const double duration_seconds = step.end_time_seconds - step.start_time_seconds;
  const double spacing_m = options.spacing_m;
  const double cell_volume_m3 = spacing_m * spacing_m * spacing_m;
  if (!std::isfinite(cell_volume_m3) || cell_volume_m3 <= 0) {
    return absl::OutOfRangeError("whitewater source cell volume is unrepresentable");
  }
  const auto cell_order = [](const WhitewaterOccupiedCell3D& cell) {
    return CellOrder(cell.sample.index);
  };
  for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
    const auto& cell = cells[cell_index];
    if (!ValidSourceCell(cell.sample) || !std::isfinite(cell.occupied_volume_m3) || cell.occupied_volume_m3 <= 0
        || cell.occupied_volume_m3 > cell_volume_m3
        || (cell_index > 0 && cell_order(cells[cell_index - 1]) >= cell_order(cell))) {
      return absl::InvalidArgumentError("invalid or unordered occupied whitewater source cell");
    }
    if (!cell.sample.carrier_valid) {
      return absl::FailedPreconditionError("occupied whitewater source cell is missing carrier");
    }
  }

  std::vector<WhitewaterCellBirth3D> births;
  for (const auto& cell : cells) {
    const auto index = cell.sample.index;
    std::array<WhitewaterSourceCell3D, 26> neighbors{};
    std::size_t neighbor_count = 0;
    for (std::int64_t dz = -1; dz <= 1; ++dz) {
      for (std::int64_t dy = -1; dy <= 1; ++dy) {
        for (std::int64_t dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0 && dz == 0) {
            continue;
          }
          const auto x = index.x + dx;
          const auto y = index.y + dy;
          const auto z = index.z + dz;
          if (!std::in_range<std::int32_t>(x) || !std::in_range<std::int32_t>(y) || !std::in_range<std::int32_t>(z)) {
            continue;
          }
          const auto key =
              std::tuple(static_cast<std::int32_t>(z), static_cast<std::int32_t>(y), static_cast<std::int32_t>(x));
          const auto neighbor = std::ranges::lower_bound(cells, key, {}, cell_order);
          if (neighbor != cells.end() && cell_order(*neighbor) == key) {
            neighbors[neighbor_count++] = neighbor->sample;
          }
        }
      }
    }
    const auto statistics = ComputeWhitewaterSource3D(cell.sample, std::span(neighbors).first(neighbor_count), options);
    if (!statistics.ok()) {
      return statistics.status();
    }
    if (statistics->potential == 0) {
      continue;
    }
    if (births.size() == maximum_births) {
      return absl::ResourceExhaustedError("accepted whitewater birth output budget exceeded");
    }
    const Vec3 world_cell_center{
        world_grid_origin.x + (index.x + 0.5) * spacing_m,
        world_grid_origin.y + (index.y + 0.5) * spacing_m,
        world_grid_origin.z + (index.z + 0.5) * spacing_m,
    };
    const double volume_time_m3_seconds = cell.occupied_volume_m3 * statistics->potential * duration_seconds;
    if (!Finite(world_cell_center) || !std::isfinite(volume_time_m3_seconds) || volume_time_m3_seconds <= 0) {
      return absl::OutOfRangeError("accepted whitewater birth position or weight is unrepresentable");
    }
    births.push_back({
        .source_cell = index,
        .event =
            {
                .time_seconds = step.quadrature_time_seconds,
                .position = world_cell_center,
                .velocity = cell.sample.velocity,
                .normal = cell.sample.normal,
                .volume_time_m3_seconds = volume_time_m3_seconds,
            },
    });
  }
  return births;
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

namespace {
struct EmissionCount {
  WhitewaterEmissionState3D state;
  std::size_t count;
};

absl::StatusOr<EmissionCount> CountWhitewaterBirthPoints(
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
  return EmissionCount{
      .state = {.next_id = state.next_id + count, .remainder = expected - count_value},
      .count = count,
  };
}

void AppendBirthPoints(
    const WhitewaterBirthEvent3D& event,
    std::uint64_t first_id,
    std::size_t count,
    std::vector<WhitewaterBirthPoint3D>& points
) {
  for (std::size_t i = 0; i < count; ++i) {
    points.push_back({
        .id = first_id + i,
        .birth_time_seconds = event.time_seconds,
        .position = event.position,
        .velocity = event.velocity,
    });
  }
}
}  // namespace

absl::StatusOr<WhitewaterEmission3D> EmitWhitewaterEvent3D(
    const WhitewaterBirthEvent3D& event,
    double rate,
    double represented_count,
    WhitewaterEmissionState3D state,
    std::size_t maximum_points
) {
  const auto emission = CountWhitewaterBirthPoints(event, rate, represented_count, state, maximum_points);
  if (!emission.ok()) {
    return emission.status();
  }
  WhitewaterEmission3D result{.state = emission->state};
  result.points.reserve(emission->count);
  AppendBirthPoints(event, state.next_id, emission->count, result.points);
  return result;
}

absl::StatusOr<WhitewaterBirthBatch3D> EmitWhitewaterBirthsThroughStep3D(
    std::span<const AcceptedWhitewaterBirth3D> births,
    const WhitewaterEmissionCheckpoint3D& previous,
    std::uint64_t completed_accepted_step_ordinal,
    double rate_per_m3_second,
    double represented_count,
    std::size_t maximum_points,
    std::size_t maximum_source_cells
) {
  maximum_points = std::min(maximum_points, std::size_t{1} << 20);
  maximum_source_cells = std::min(maximum_source_cells, std::size_t{4096});
  if (births.size() > (std::size_t{1} << 20) || previous.remainders.size() > maximum_source_cells) {
    return absl::ResourceExhaustedError("whitewater checkpoint source or event budget exceeded");
  }
  if (previous.next_point_id == 0 || completed_accepted_step_ordinal <= previous.last_accepted_step_ordinal
      || !std::isfinite(rate_per_m3_second) || rate_per_m3_second <= 0 || !std::isfinite(represented_count)
      || represented_count <= 0) {
    return absl::InvalidArgumentError("invalid whitewater checkpoint controls");
  }
  std::vector<CellIndex3D> source_cells;
  source_cells.reserve(previous.remainders.size() + births.size());
  const auto remainder_key = [](const WhitewaterCellRemainder3D& record) {
    return CellOrder(record.source_cell);
  };
  for (std::size_t index = 0; index < previous.remainders.size(); ++index) {
    const auto& record = previous.remainders[index];
    if (!std::isfinite(record.fractional_count) || record.fractional_count <= 0 || record.fractional_count >= 1
        || (index > 0 && remainder_key(previous.remainders[index - 1]) >= remainder_key(record))) {
      return absl::InvalidArgumentError("invalid or unordered whitewater source remainder");
    }
    source_cells.push_back(record.source_cell);
  }
  for (std::size_t index = 0; index < births.size(); ++index) {
    const auto& accepted = births[index];
    if (accepted.accepted_step_ordinal <= previous.last_accepted_step_ordinal
        || accepted.accepted_step_ordinal > completed_accepted_step_ordinal
        || !std::isfinite(accepted.birth.event.time_seconds) || accepted.birth.event.time_seconds < 0
        || !std::isfinite(accepted.birth.event.volume_time_m3_seconds)
        || accepted.birth.event.volume_time_m3_seconds <= 0) {
      return absl::InvalidArgumentError("whitewater birth is invalid or outside the new accepted interval");
    }
    if (index > 0) {
      const auto& earlier = births[index - 1];
      const bool same_step = earlier.accepted_step_ordinal == accepted.accepted_step_ordinal;
      const double earlier_time = earlier.birth.event.time_seconds;
      const double current_time = accepted.birth.event.time_seconds;
      if (earlier.accepted_step_ordinal > accepted.accepted_step_ordinal || earlier_time > current_time
          || (same_step
              && (earlier_time != current_time
                  || CellOrder(earlier.birth.source_cell) >= CellOrder(accepted.birth.source_cell)))) {
        return absl::InvalidArgumentError("whitewater births are not in accepted-step and source-cell order");
      }
    }
    source_cells.push_back(accepted.birth.source_cell);
  }
  std::ranges::sort(source_cells, {}, CellOrder);
  const auto duplicates = std::ranges::unique(source_cells, {}, CellOrder);
  source_cells.erase(duplicates.begin(), duplicates.end());
  if (source_cells.size() > maximum_source_cells) {
    return absl::ResourceExhaustedError("whitewater source remainder budget exceeded");
  }

  WhitewaterBirthBatch3D result;
  result.checkpoint.next_point_id = previous.next_point_id;
  result.checkpoint.remainders.reserve(source_cells.size());
  std::size_t previous_index = 0;
  for (const auto cell : source_cells) {
    double fractional_count = 0;
    if (previous_index < previous.remainders.size()
        && remainder_key(previous.remainders[previous_index]) == CellOrder(cell)) {
      fractional_count = previous.remainders[previous_index++].fractional_count;
    }
    result.checkpoint.remainders.push_back({.source_cell = cell, .fractional_count = fractional_count});
  }
  for (const auto& accepted : births) {
    const auto cell_key = CellOrder(accepted.birth.source_cell);
    auto remainder = std::ranges::lower_bound(result.checkpoint.remainders, cell_key, {}, remainder_key);
    if (remainder == result.checkpoint.remainders.end() || remainder_key(*remainder) != cell_key) {
      return absl::InternalError("whitewater source remainder was not allocated");
    }
    const WhitewaterEmissionState3D state{
        .next_id = result.checkpoint.next_point_id,
        .remainder = remainder->fractional_count,
    };
    const auto emission = CountWhitewaterBirthPoints(
        accepted.birth.event,
        rate_per_m3_second,
        represented_count,
        state,
        maximum_points - result.points.size()
    );
    if (!emission.ok()) {
      return emission.status();
    }
    AppendBirthPoints(accepted.birth.event, state.next_id, emission->count, result.points);
    remainder->fractional_count = emission->state.remainder;
    result.checkpoint.next_point_id = emission->state.next_id;
  }
  const auto empty = std::ranges::remove_if(result.checkpoint.remainders, [](const WhitewaterCellRemainder3D& record) {
    return record.fractional_count == 0;
  });
  result.checkpoint.remainders.erase(empty.begin(), empty.end());
  result.checkpoint.last_accepted_step_ordinal = completed_accepted_step_ordinal;
  return result;
}
}  // namespace fluidity
