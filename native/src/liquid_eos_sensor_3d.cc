#include "liquid_eos_sensor_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

#include "fluidity/base/liquid_boundary_geometry_math_3d.h"
#include "liquid_boundary_backend.h"

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

double
ExplicitParticleDensity(const LiquidParticleState3D& particle, std::span<const LiquidConstituentMaterial3D> materials) {
  double density = 0.0;
  for (const auto& constituent : std::span(particle.constituents).first(particle.constituent_count)) {
    const auto material =
        std::ranges::lower_bound(materials, constituent.substance_id, {}, &LiquidConstituentMaterial3D::substance_id);
    if (material == materials.end() || material->substance_id != constituent.substance_id
        || material->phase_id != particle.phase_id) {
      return 0.0;
    }
    density = std::fma(constituent.volume_fraction, material->density_kg_per_m3, density);
  }
  return density;
}

struct PhaseFieldSample {
  double distance;
  std::size_t nearest_particle;
};

PhaseFieldSample
SamplePhaseField(std::span<const LiquidParticleState3D> particles, const LiquidPhasePair3D& pair, Vec3 position) {
  std::array distances{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  std::array<std::size_t, 2> nearest{};
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto& particle = particles[index];
    const std::size_t phase = particle.phase_id == pair.phase_a ? 0 : 1;
    const double distance = std::hypot(
        position.x - particle.world_position.x,
        position.y - particle.world_position.y,
        position.z - particle.world_position.z
    );
    if (distance < distances[phase]) {
      distances[phase] = distance;
      nearest[phase] = index;
    }
  }
  return {distances[0] - distances[1], nearest[distances[0] <= distances[1] ? 0 : 1]};
}

absl::Status DepositCompositionDensity(
    CellCompositionDensity3D& sensor,
    double weight,
    double current_volume,
    double particle_density
) {
  const double weighted_volume = weight * current_volume;
  sensor.sampled_volume_m3 += weighted_volume;
  sensor.sampled_mass_kg = std::fma(weighted_volume, particle_density, sensor.sampled_mass_kg);
  if (weighted_volume <= 0.0 || !std::isfinite(sensor.sampled_volume_m3) || !std::isfinite(sensor.sampled_mass_kg)
      || sensor.sampled_mass_kg <= 0.0) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
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
      particle_density = ExplicitParticleDensity(particle, materials);
      if (!std::isfinite(particle_density) || particle_density <= 0.0) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
    }
    const double current_volume = particle.reference_volume * particle.volume_ratio;
    const auto scattered =
        ScatterCellTent(geometry, cells, particle.world_position, [&](std::size_t cell, double weight) {
          return DepositCompositionDensity(sensors[cell], weight, current_volume, particle_density);
        });
    if (!scattered.ok()) {
      return scattered;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::size_t> PrepareLiquidPhaseCrossings3D(
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    std::span<const LiquidParticleState3D> particles,
    std::span<const LiquidConstituentMaterial3D> materials,
    const LiquidPhasePair3D& pair,
    std::span<CellCompositionDensity3D> sensors,
    std::span<LiquidInternalInterface3D> crossings
) {
  if (sensors.size() != geometry.cells.size()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  for (const auto& particle : particles) {
    if (particle.constituent_count == 0 || (particle.phase_id != pair.phase_a && particle.phase_id != pair.phase_b)) {
      return absl::Status(absl::StatusCode::kFailedPrecondition, {});
    }
  }
  std::size_t phase_a_cell_count = 0;
  for (std::size_t index = 0; index < sensors.size(); ++index) {
    const auto position = geometry.mac_grid.CellWorldCenter(geometry.cells[index].cell_index);
    if (!position.ok()) {
      return position.status();
    }
    const auto field = SamplePhaseField(particles, pair, *position);
    if (!std::isfinite(field.distance)) {
      return absl::Status(absl::StatusCode::kFailedPrecondition, {});
    }
    sensors[index] = {.phase_distance = field.distance, .nearest_particle = field.nearest_particle};
    phase_a_cell_count += field.distance <= 0.0;
  }
  if (phase_a_cell_count == 0 || phase_a_cell_count == sensors.size()) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  for (const auto& particle : particles) {
    const double density = ExplicitParticleDensity(particle, materials);
    if (!std::isfinite(density) || density <= 0.0) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    const double current_volume = particle.reference_volume * particle.volume_ratio;
    const auto scattered = ScatterCellTent(
        geometry.mac_grid.geometry(),
        geometry.cells,
        particle.world_position,
        [&](std::size_t cell, double weight) {
          const bool cell_is_a = sensors[cell].phase_distance <= 0.0;
          if (cell_is_a != (particle.phase_id == pair.phase_a)) {
            return absl::OkStatus();
          }
          return DepositCompositionDensity(sensors[cell], weight, current_volume, density);
        }
    );
    if (!scattered.ok()) {
      return scattered;
    }
  }
  for (auto& sensor : sensors) {
    if (sensor.sampled_volume_m3 == 0.0) {
      sensor.extrapolated_density = ExplicitParticleDensity(particles[sensor.nearest_particle], materials);
      if (!std::isfinite(sensor.extrapolated_density) || sensor.extrapolated_density <= 0.0) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
    }
  }

  const auto find_cell = [&](CellIndex3D index) {
    const auto key = CellKey(index);
    const auto found = std::ranges::lower_bound(geometry.cells, key, {}, [](const FrozenLiquidCell3D& cell) {
      return CellKey(cell.cell_index);
    });
    return found != geometry.cells.end() && CellKey(found->cell_index) == key
        ? static_cast<std::size_t>(found - geometry.cells.begin())
        : geometry.cells.size();
  };
  std::size_t count = 0;
  for (std::size_t ordinal = 0; ordinal < geometry.faces.size(); ++ordinal) {
    const auto& face = geometry.faces[ordinal];
    if (face.open_area <= 0.0) {
      continue;
    }
    const auto lattice = face.face_index.lattice_index;
    const CellIndex3D head{lattice.x, lattice.y, lattice.z};
    std::array<std::int32_t, 3> tail{head.x, head.y, head.z};
    const auto axis = static_cast<std::size_t>(face.face_index.axis);
    if (tail[axis] == std::numeric_limits<std::int32_t>::min()) {
      continue;
    }
    --tail[axis];
    const auto head_cell = find_cell(head);
    const auto tail_cell = find_cell({tail[0], tail[1], tail[2]});
    if (head_cell == sensors.size() || tail_cell == sensors.size()) {
      continue;
    }
    const double head_distance = sensors[head_cell].phase_distance;
    const double tail_distance = sensors[tail_cell].phase_distance;
    if ((head_distance <= 0.0) == (tail_distance <= 0.0)) {
      continue;
    }
    if (count == crossings.size()) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    const double scale = std::max(std::abs(head_distance), std::abs(tail_distance));
    const double scaled_tail = std::abs(tail_distance) / scale;
    const double tail_fraction = scaled_tail / (scaled_tail + std::abs(head_distance) / scale);
    double curvature = 0.0;
    if (pair.surface_tension > 0.0) {
      const auto anchor = geometry.mac_grid.FaceWorldAnchor(face.face_index);
      if (!anchor.ok()) {
        return anchor.status();
      }
      const double h = geometry.mac_grid.geometry().cell_width;
      std::array position{anchor->x, anchor->y, anchor->z};
      position[axis] += (tail_fraction - 0.5) * h;
      boundary_internal::GeometryInput query{
          .mode = boundary_internal::GeometryMode::kCurvature,
          .geometry = geometry.mac_grid.geometry(),
      };
      for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
          for (int x = -1; x <= 1; ++x) {
            const auto field =
                SamplePhaseField(particles, pair, {position[0] + x * h, position[1] + y * h, position[2] + z * h});
            query.derivative_distances[(z + 1) * 9 + (y + 1) * 3 + x + 1] = field.distance;
          }
        }
      }
      const auto result = boundary_internal::ComputeGeometry(query);
      if (!result.ok()) {
        return boundary_internal::ArithmeticStatus(result.status(), true);
      }
      curvature = tail_distance <= 0.0 ? result->interface_curvature : -result->interface_curvature;
    }
    crossings[count++] = {
        .face_ordinal = ordinal,
        .tail_fraction = tail_fraction,
        .surface_tension = pair.surface_tension,
        .curvature_tail_to_head = curvature,
        .normal_viscous_jump_tail_to_head = 0.0,
    };
  }
  return count;
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
