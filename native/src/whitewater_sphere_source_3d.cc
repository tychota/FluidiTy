#include "fluidity/host/whitewater_sphere_source_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "fluidity/host/motion_carrier_3d.h"
#include "liquid_substep_3d_internal.h"

namespace fluidity {
namespace {
bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

struct SourceCellBox {
  std::array<std::int32_t, 3> minimum;
  std::array<std::int32_t, 3> maximum;
  std::size_t cell_count;
};

// Bounds the whole candidate box before any grid workspace allocation.
absl::StatusOr<SourceCellBox> SourceBounds(Vec3 lower_m, Vec3 upper_m, Vec3 origin, double spacing_m) {
  const std::array lower_bound{lower_m.x, lower_m.y, lower_m.z};
  const std::array upper_bound{upper_m.x, upper_m.y, upper_m.z};
  const std::array grid_origin{origin.x, origin.y, origin.z};
  SourceCellBox box{.minimum = {}, .maximum = {}, .cell_count = 1};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double lower = std::floor((lower_bound[axis] - grid_origin[axis]) / spacing_m);
    const double upper = std::floor((upper_bound[axis] - grid_origin[axis]) / spacing_m);
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower < std::numeric_limits<std::int32_t>::min()
        || upper > std::numeric_limits<std::int32_t>::max() || upper < lower) {
      return absl::OutOfRangeError("whitewater source sphere leaves signed grid index bounds");
    }
    box.minimum[axis] = static_cast<std::int32_t>(lower);
    box.maximum[axis] = static_cast<std::int32_t>(upper);
    const auto extent =
        static_cast<std::uint64_t>(static_cast<std::int64_t>(box.maximum[axis]) - box.minimum[axis] + 1);
    if (extent > 4096 / box.cell_count) {
      return absl::ResourceExhaustedError("whitewater source bounds exceed fixed grid cell budget");
    }
    box.cell_count *= static_cast<std::size_t>(extent);
  }
  return box;
}
}  // namespace

bool IsStationaryWhitewaterBeach3D(const LiquidObstacle3D& obstacle) {
  const auto* beach = std::get_if<TranslatingHalfspace3D>(&obstacle);
  return beach && Finite(beach->unit_normal) && std::isfinite(beach->start_plane_coordinate)
      && std::abs(std::hypot(beach->unit_normal.x, beach->unit_normal.y, beach->unit_normal.z) - 1) <= 1e-12
      && beach->wall_velocity.x == 0 && beach->wall_velocity.y == 0 && beach->wall_velocity.z == 0;
}

absl::StatusOr<std::vector<WhitewaterCellBirth3D>> SampleAcceptedWhitewaterStep3D(
    const LiquidState3D& accepted,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& step_options,
    Vec3 world_source_grid_origin,
    double previous_accepted_time_seconds,
    const WhitewaterSourceOptions3D& source_options,
    std::size_t maximum_births
) {
  const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field);
  const auto* particle_field = std::get_if<LiquidParticleSphereUnion3D>(&scene.field);
  const bool beach = IsStationaryWhitewaterBeach3D(scene.obstacle);
  if ((!sphere && !particle_field) || (!std::holds_alternative<std::monostate>(scene.obstacle) && !beach)) {
    return absl::UnimplementedError("whitewater source requires a sphere field and stationary beach or no wall");
  }
  const double radius_m = sphere ? sphere->radius : particle_field->radius;
  const double spacing_m = source_options.spacing_m;
  const double source_layer_m = 2 * spacing_m;
  if ((sphere && !Finite(sphere->world_center)) || !Finite(world_source_grid_origin) || !std::isfinite(spacing_m)
      || spacing_m <= 0 || !std::isfinite(radius_m) || radius_m <= 0 || (sphere && radius_m <= source_layer_m)
      || !std::isfinite(previous_accepted_time_seconds) || previous_accepted_time_seconds < 0
      || !std::isfinite(accepted.clock.time) || accepted.clock.time <= previous_accepted_time_seconds
      || accepted.clock.accepted_step_ordinal == 0) {
    return absl::InvalidArgumentError("invalid whitewater sphere, source spacing or accepted duration");
  }
  const WhitewaterSourceStep3D source_step{
      .start_time_seconds = previous_accepted_time_seconds,
      .end_time_seconds = accepted.clock.time,
      .quadrature_time_seconds = accepted.clock.time,
  };
  // Validate physical controls even for an empty source, before any sampling.
  const auto empty =
      ComputeWhitewaterBirths3D({}, world_source_grid_origin, source_step, source_options, maximum_births);
  if (!empty.ok()) {
    return empty.status();
  }
  if (accepted.particles.size() == 0) {
    const auto valid = ValidateLiquidState3D(accepted);
    if (!valid.ok()) {
      return valid;
    }
    return std::vector<WhitewaterCellBirth3D>{};
  }
  for (const auto& particle : accepted.particles.values()) {
    if (particle.side_tag != 0) {
      return absl::UnimplementedError("whitewater sphere source requires uniform material sides");
    }
  }
  const auto particles = accepted.particles.values();
  Vec3 lower_m = sphere ? sphere->world_center : particles.front().world_position;
  Vec3 upper_m = lower_m;
  if (particle_field) {
    for (const auto& particle : particles) {
      const auto position = particle.world_position;
      lower_m = {std::min(lower_m.x, position.x), std::min(lower_m.y, position.y), std::min(lower_m.z, position.z)};
      upper_m = {std::max(upper_m.x, position.x), std::max(upper_m.y, position.y), std::max(upper_m.z, position.z)};
    }
  }
  const auto box = SourceBounds(
      {lower_m.x - radius_m, lower_m.y - radius_m, lower_m.z - radius_m},
      {upper_m.x + radius_m, upper_m.y + radius_m, upper_m.z + radius_m},
      world_source_grid_origin,
      spacing_m
  );
  if (!box.ok()) {
    return box.status();
  }
  const auto carrier = FrozenMotionCarrier3D::Create(accepted, scene, step_options);
  if (!carrier.ok()) {
    return carrier.status();
  }
  const double curvature_per_m = 2 / radius_m;
  // On a smooth union patch, the closest sphere supplies its exact normal and
  // curvature. A tie resolves in accepted particle order.
  const auto closest_radial = [&](Vec3 position) {
    Vec3 radial{};
    double distance_m = std::numeric_limits<double>::infinity();
    const auto consider = [&](Vec3 origin) {
      const Vec3 candidate{position.x - origin.x, position.y - origin.y, position.z - origin.z};
      const double candidate_distance_m = std::hypot(candidate.x, candidate.y, candidate.z);
      if (candidate_distance_m < distance_m) {
        radial = candidate;
        distance_m = candidate_distance_m;
      }
    };
    if (sphere) {
      consider(sphere->world_center);
    } else {
      for (const auto& particle : particles) {
        consider(particle.world_position);
      }
    }
    return std::pair{radial, distance_m};
  };
  const double cell_volume_m3 = spacing_m * spacing_m * spacing_m;
  const double probe_offset_m = 0.25 * spacing_m;
  const std::array offsets{-probe_offset_m, probe_offset_m};
  std::vector<WhitewaterOccupiedCell3D> occupied_cells;
  occupied_cells.reserve(box->cell_count);
  for (std::int64_t z = box->minimum[2]; z <= box->maximum[2]; ++z) {
    for (std::int64_t y = box->minimum[1]; y <= box->maximum[1]; ++y) {
      for (std::int64_t x = box->minimum[0]; x <= box->maximum[0]; ++x) {
        const Vec3 center{
            world_source_grid_origin.x + (static_cast<double>(x) + 0.5) * spacing_m,
            world_source_grid_origin.y + (static_cast<double>(y) + 0.5) * spacing_m,
            world_source_grid_origin.z + (static_cast<double>(z) + 0.5) * spacing_m,
        };
        if (beach && !(liquid_step_internal::SolidDistance(scene.obstacle, center) > 0)) {
          continue;
        }
        const auto [radial, center_distance_m] = closest_radial(center);
        if (!std::isfinite(center_distance_m)) {
          return absl::OutOfRangeError("whitewater source sphere coordinate overflowed");
        }
        if (center_distance_m == 0 || center_distance_m < radius_m - source_layer_m) {
          continue;
        }
        unsigned occupied_probes = 0;
        for (const double dz : offsets) {
          for (const double dy : offsets) {
            for (const double dx : offsets) {
              const Vec3 probe{center.x + dx, center.y + dy, center.z + dz};
              if (beach && !(liquid_step_internal::SolidDistance(scene.obstacle, probe) > 0)) {
                continue;
              }
              occupied_probes += sphere ? std::hypot(radial.x + dx, radial.y + dy, radial.z + dz) < radius_m
                                        : closest_radial(probe).second < radius_m;
            }
          }
        }
        if (occupied_probes == 0) {
          continue;
        }
        const auto sample = carrier->Sample(center);
        if (!sample.ok()) {
          return sample.status();
        }
        occupied_cells.push_back({
            .sample =
                {
                    .index = {static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), static_cast<std::int32_t>(z)},
                    .velocity = sample->velocity,
                    .normal =
                        {radial.x / center_distance_m, radial.y / center_distance_m, radial.z / center_distance_m},
                    .curvature_per_m = curvature_per_m,
                },
            .occupied_volume_m3 = cell_volume_m3 * (static_cast<double>(occupied_probes) / 8),
        });
      }
    }
  }
  return ComputeWhitewaterBirths3D(
      occupied_cells,
      world_source_grid_origin,
      source_step,
      source_options,
      maximum_births
  );
}
}  // namespace fluidity
