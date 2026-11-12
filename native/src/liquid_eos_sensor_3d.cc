#include "liquid_eos_sensor_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

namespace fluidity::liquid_step_internal {
namespace {

auto CellKey(CellIndex3D cell) {
  return std::tuple{cell.z, cell.y, cell.x};
}

template <typename Deposit>
absl::Status ScatterCellTent(
    UniformMacGeometry3D geometry,
    std::span<const FrozenLiquidCell3D> cells,
    Vec3 position,
    Deposit deposit
) {
  const auto origin = geometry.world_origin;
  const double width = geometry.cell_width;
  constexpr auto kMinimumCellIndex = std::numeric_limits<std::int32_t>::min();
  constexpr auto kMaximumCellIndex = std::numeric_limits<std::int32_t>::max();
  const std::array cell_coordinates{
      (position.x - origin.x) / width - 0.5,
      (position.y - origin.y) / width - 0.5,
      (position.z - origin.z) / width - 0.5,
  };
  if (std::ranges::any_of(cell_coordinates, [](double coordinate) {
        return !std::isfinite(coordinate);
      })) {
    return absl::OutOfRangeError("liquid cell sensor lattice coordinate is unrepresentable");
  }
  if (std::ranges::any_of(cell_coordinates, [](double coordinate) {
        return coordinate <= static_cast<double>(kMinimumCellIndex) - 1.0
            || coordinate >= static_cast<double>(kMaximumCellIndex) + 1.0;
      })) {
    return absl::OkStatus();
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
    const auto status = deposit(static_cast<std::size_t>(cell - cells.begin()), weight);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
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
  for (const auto& particle : particles) {
    if (particle.side_tag != liquid_side_tag) {
      continue;
    }
    const auto scattered =
        ScatterCellTent(geometry, cells, particle.world_position, [&](std::size_t cell, double weight) {
          const double weighted_reference_volume = weight * particle.reference_volume;
          auto& sensor = sensors[cell];
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
          return absl::OkStatus();
        });
    if (!scattered.ok()) {
      return scattered;
    }
  }
  return absl::OkStatus();
}

absl::Status SampleLiquidCellCompositionDensity3D(
    UniformMacGeometry3D geometry,
    std::span<const FrozenLiquidCell3D> cells,
    std::span<const LiquidParticleState3D> particles,
    std::span<const LiquidConstituentMaterial3D> materials,
    double legacy_density,
    std::span<CellCompositionDensity3D> sensors
) {
  if (sensors.size() != cells.size()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  std::ranges::fill(sensors, CellCompositionDensity3D{});
  std::optional<std::uint64_t> phase;
  for (const auto& particle : particles) {
    double particle_density = legacy_density;
    if (particle.constituent_count != 0) {
      if (phase && *phase != particle.phase_id) {
        return absl::Status(absl::StatusCode::kFailedPrecondition, {});
      }
      phase = particle.phase_id;
      particle_density = 0.0;
      for (const auto& constituent : std::span(particle.constituents).first(particle.constituent_count)) {
        const auto material = std::ranges::lower_bound(
            materials,
            constituent.substance_id,
            {},
            &LiquidConstituentMaterial3D::substance_id
        );
        if (material == materials.end() || material->substance_id != constituent.substance_id
            || material->phase_id != particle.phase_id) {
          return absl::Status(absl::StatusCode::kInvalidArgument, {});
        }
        particle_density = std::fma(constituent.volume_fraction, material->density_kg_per_m3, particle_density);
      }
    }
    const double current_volume = particle.reference_volume * particle.volume_ratio;
    const auto scattered =
        ScatterCellTent(geometry, cells, particle.world_position, [&](std::size_t cell, double weight) {
          const double weighted_volume = weight * current_volume;
          auto& sensor = sensors[cell];
          sensor.sampled_volume_m3 += weighted_volume;
          sensor.sampled_mass_kg = std::fma(weighted_volume, particle_density, sensor.sampled_mass_kg);
          if (weighted_volume <= 0.0 || !std::isfinite(sensor.sampled_volume_m3)
              || !std::isfinite(sensor.sampled_mass_kg) || sensor.sampled_mass_kg <= 0.0) {
            return absl::Status(absl::StatusCode::kOutOfRange, {});
          }
          return absl::OkStatus();
        });
    if (!scattered.ok()) {
      return scattered;
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

absl::Status BalanceSealedLiquidDrift3D(std::span<SealedLiquidDriftCell3D> sensed_cells, double maximum_rate) {
  if (!std::isfinite(maximum_rate) || maximum_rate < 0.0) {
    return absl::InvalidArgumentError("invalid sealed drift rate limit");
  }
  double maximum_volume = 0.0;
  for (const auto& cell : sensed_cells) {
    if (!std::isfinite(cell.raw_rate) || !std::isfinite(cell.fluid_volume) || cell.fluid_volume <= 0.0) {
      return absl::InvalidArgumentError("invalid sealed drift cell data");
    }
    maximum_volume = std::max(maximum_volume, cell.fluid_volume);
  }
  if (sensed_cells.empty() || maximum_rate == 0.0) {
    for (auto& cell : sensed_cells) {
      cell.corrected_rate = 0.0;
    }
    return absl::OkStatus();
  }

  std::ranges::sort(sensed_cells, [](const SealedLiquidDriftCell3D& left, const SealedLiquidDriftCell3D& right) {
    return std::tie(left.raw_rate, left.cell_ordinal) < std::tie(right.raw_rate, right.cell_ordinal);
  });
  double total_weight = 0.0;
  for (const auto& cell : sensed_cells) {
    total_weight += cell.fluid_volume / maximum_volume;
  }
  double median_rate = sensed_cells.back().raw_rate;
  double prefix_weight = 0.0;
  for (const auto& cell : sensed_cells) {
    prefix_weight += cell.fluid_volume / maximum_volume;
    if (prefix_weight >= 0.5 * total_weight) {
      median_rate = cell.raw_rate;
      break;
    }
  }

  // Rebase at the weighted median before shifting. A root lies within one
  // clamp radius of the median, even when raw rates have a large common offset.
  const auto rate = [&](const SealedLiquidDriftCell3D& cell, double offset) {
    const double difference = cell.raw_rate - median_rate;
    if (std::isfinite(difference)) {
      return std::clamp(difference - offset, -maximum_rate, maximum_rate);
    }
    if (maximum_rate <= std::numeric_limits<double>::max() / 2.0) {
      return std::copysign(maximum_rate, difference);
    }
    const double half_difference = cell.raw_rate * 0.5 - median_rate * 0.5;
    const double half_maximum_rate = maximum_rate * 0.5;
    return 2.0 * std::clamp(half_difference - offset * 0.5, -half_maximum_rate, half_maximum_rate);
  };
  const auto balance = [&](double offset) {
    double rate_scale = 0.0;
    for (const auto& cell : sensed_cells) {
      rate_scale = std::max(rate_scale, std::abs(rate(cell, offset)));
    }
    if (rate_scale == 0.0) {
      return std::tuple{0.0, 0.0, true};
    }
    double sum = 0.0;
    double roundoff = 0.0;
    double absolute_sum = 0.0;
    bool terms_representable = true;
    for (const auto& cell : sensed_cells) {
      const double corrected_rate = rate(cell, offset);
      const double term = (cell.fluid_volume / maximum_volume) * (corrected_rate / rate_scale);
      terms_representable &= corrected_rate == 0.0 || std::isnormal(term);
      const double corrected_term = term - roundoff;
      const double next = sum + corrected_term;
      roundoff = (next - sum) - corrected_term;
      sum = next;
      absolute_sum += std::abs(term);
    }
    return std::tuple{sum, absolute_sum, terms_representable};
  };
  const double offset_radius = std::min(
      maximum_rate,
      std::max(median_rate - sensed_cells.front().raw_rate, sensed_cells.back().raw_rate - median_rate)
  );
  double lower = -offset_radius;
  double upper = offset_radius;
  double offset = 0.0;
  constexpr double kBalanceTolerance = 8.0 * std::numeric_limits<double>::epsilon();
  // Check the balance of rounded output rates, including when an endpoint
  // supplies the closest representable offset.
  while (true) {
    const auto [sum, absolute_sum, representable] = balance(offset);
    if (representable && std::abs(sum) <= kBalanceTolerance * absolute_sum) {
      for (auto& cell : sensed_cells) {
        cell.corrected_rate = rate(cell, offset);
      }
      return absl::OkStatus();
    }
    if (sum > 0.0) {
      lower = offset;
    } else {
      upper = offset;
    }
    const double next = std::midpoint(lower, upper);
    if (next == lower || next == upper) {
      for (const double endpoint : {lower, upper}) {
        const auto [endpoint_sum, endpoint_absolute_sum, endpoint_representable] = balance(endpoint);
        if (endpoint_representable && std::abs(endpoint_sum) <= kBalanceTolerance * endpoint_absolute_sum) {
          for (auto& cell : sensed_cells) {
            cell.corrected_rate = rate(cell, endpoint);
          }
          return absl::OkStatus();
        }
      }
      return absl::OutOfRangeError("sealed liquid drift balance is unrepresentable");
    }
    offset = next;
  }
}

}  // namespace fluidity::liquid_step_internal
