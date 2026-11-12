#include "liquid_eos_sensor_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>

namespace fluidity::liquid_step_internal {
namespace {

auto CellKey(CellIndex3D cell) {
  return std::tuple{cell.z, cell.y, cell.x};
}

}  // namespace

absl::Status SampleLiquidCellVolumeRatios3D(
    UniformMacGeometry3D geometry,
    std::span<const FrozenLiquidCell3D> cells,
    std::span<const LiquidParticleState3D> particles,
    std::int64_t liquid_side_tag,
    std::span<CellVolumeRatioSensor3D> sensors
) {
  if (sensors.size() != cells.size()) {
    return absl::InvalidArgumentError("volume-ratio sensor extent differs from selected cells");
  }
  std::ranges::fill(sensors, CellVolumeRatioSensor3D{});
  if (cells.empty()) {
    return absl::OkStatus();
  }
  const auto origin = geometry.world_origin;
  const double width = geometry.cell_width;
  constexpr auto kMinimumCellIndex = std::numeric_limits<std::int32_t>::min();
  constexpr auto kMaximumCellIndex = std::numeric_limits<std::int32_t>::max();
  for (const auto& particle : particles) {
    if (particle.side_tag != liquid_side_tag) {
      continue;
    }
    const auto position = particle.world_position;
    const std::array cell_coordinates{
        (position.x - origin.x) / width - 0.5,
        (position.y - origin.y) / width - 0.5,
        (position.z - origin.z) / width - 0.5,
    };
    if (std::ranges::any_of(cell_coordinates, [](double coordinate) {
          return !std::isfinite(coordinate);
        })) {
      return absl::OutOfRangeError("volume-ratio sensor lattice coordinate is unrepresentable");
    }
    if (std::ranges::any_of(cell_coordinates, [](double coordinate) {
          return coordinate <= static_cast<double>(kMinimumCellIndex) - 1.0
              || coordinate >= static_cast<double>(kMaximumCellIndex) + 1.0;
        })) {
      continue;
    }
    std::array<std::int64_t, 3> low{};
    std::array<double, 3> fraction{};
    for (std::size_t axis = 0; axis < low.size(); ++axis) {
      const double low_coordinate = std::floor(cell_coordinates[axis]);
      low[axis] = static_cast<std::int64_t>(low_coordinate);
      fraction[axis] = cell_coordinates[axis] - low_coordinate;
    }
    for (std::size_t corner = 0; corner < 8; ++corner) {
      const std::array upper{corner & 1, (corner >> 1) & 1, (corner >> 2) & 1};
      const std::array indices{
          low[0] + static_cast<std::int64_t>(upper[0]),
          low[1] + static_cast<std::int64_t>(upper[1]),
          low[2] + static_cast<std::int64_t>(upper[2]),
      };
      if (std::ranges::any_of(indices, [](std::int64_t coordinate) {
            return coordinate < kMinimumCellIndex || coordinate > kMaximumCellIndex;
          })) {
        continue;
      }
      const double x_weight = upper[0] ? fraction[0] : 1.0 - fraction[0];
      const double y_weight = upper[1] ? fraction[1] : 1.0 - fraction[1];
      const double z_weight = upper[2] ? fraction[2] : 1.0 - fraction[2];
      if (x_weight == 0.0 || y_weight == 0.0 || z_weight == 0.0) {
        continue;
      }
      const auto key = CellKey({
          static_cast<std::int32_t>(indices[0]),
          static_cast<std::int32_t>(indices[1]),
          static_cast<std::int32_t>(indices[2]),
      });
      const auto cell = std::ranges::lower_bound(cells, key, {}, [](const FrozenLiquidCell3D& record) {
        return CellKey(record.cell_index);
      });
      if (cell == cells.end() || CellKey(cell->cell_index) != key) {
        continue;
      }
      const double weight = x_weight * y_weight * z_weight;
      const double weighted_reference_volume = weight * particle.reference_volume;
      auto& sensor = sensors[static_cast<std::size_t>(cell - cells.begin())];
      const double previous_reference_volume = sensor.sampled_reference_volume;
      const double total_reference_volume = previous_reference_volume + weighted_reference_volume;
      if (weighted_reference_volume == 0.0 || !std::isfinite(total_reference_volume)) {
        return absl::OutOfRangeError("weighted sensor reference volume is unrepresentable");
      }
      // Interpolate from the heavier sample using the smaller weight ratio.
      // Taking its complement can erase a light donor with a large J.
      const double light_fraction =
          std::min(weighted_reference_volume, previous_reference_volume) / total_reference_volume;
      const double mean = weighted_reference_volume > previous_reference_volume
          ? std::lerp(particle.volume_ratio, sensor.mean_volume_ratio, light_fraction)
          : std::lerp(sensor.mean_volume_ratio, particle.volume_ratio, light_fraction);
      if ((previous_reference_volume > 0.0 && light_fraction == 0.0) || mean <= 0.0 || !std::isfinite(mean)) {
        return absl::OutOfRangeError("sensor volume ratio is unrepresentable");
      }
      sensor.sampled_reference_volume = total_reference_volume;
      sensor.mean_volume_ratio = mean;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<double> LiquidVolumeDriftRate3D(
    CellVolumeRatioSensor3D sensor,
    double target_log_volume_ratio,
    double correction_time,
    double maximum_rate
) {
  if (!std::isfinite(target_log_volume_ratio) || !std::isfinite(correction_time) || correction_time <= 0.0
      || !std::isfinite(maximum_rate) || maximum_rate < 0.0 || !std::isfinite(sensor.sampled_reference_volume)
      || sensor.sampled_reference_volume < 0.0) {
    return absl::InvalidArgumentError("invalid EOS drift controls or sensor support");
  }
  if (sensor.sampled_reference_volume == 0.0) {
    return 0.0;
  }
  if (!std::isfinite(sensor.mean_volume_ratio) || sensor.mean_volume_ratio <= 0.0) {
    return absl::InvalidArgumentError("invalid sensed volume ratio");
  }
  const double drift_rate = (target_log_volume_ratio - std::log(sensor.mean_volume_ratio)) / correction_time;
  return std::clamp(drift_rate, -maximum_rate, maximum_rate);
}

}  // namespace fluidity::liquid_step_internal
