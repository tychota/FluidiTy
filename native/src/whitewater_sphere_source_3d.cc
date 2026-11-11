#include "fluidity/host/whitewater_sphere_source_3d.h"

#include <array>
#include <cmath>
#include <limits>

#include "fluidity/host/motion_carrier_3d.h"

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
absl::StatusOr<SourceCellBox> SphereSourceCellBox(LiquidAnalyticSphere3D sphere, Vec3 origin, double spacing_m) {
  const std::array center{sphere.world_center.x, sphere.world_center.y, sphere.world_center.z};
  const std::array grid_origin{origin.x, origin.y, origin.z};
  SourceCellBox box{.minimum = {}, .maximum = {}, .cell_count = 1};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double lower = std::floor((center[axis] - sphere.radius - grid_origin[axis]) / spacing_m);
    const double upper = std::floor((center[axis] + sphere.radius - grid_origin[axis]) / spacing_m);
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower < std::numeric_limits<std::int32_t>::min()
        || upper > std::numeric_limits<std::int32_t>::max() || upper < lower) {
      return absl::OutOfRangeError("whitewater source sphere leaves signed grid index bounds");
    }
    box.minimum[axis] = static_cast<std::int32_t>(lower);
    box.maximum[axis] = static_cast<std::int32_t>(upper);
    const auto extent =
        static_cast<std::uint64_t>(static_cast<std::int64_t>(box.maximum[axis]) - box.minimum[axis] + 1);
    if (extent > 4096 / box.cell_count) {
      return absl::ResourceExhaustedError("whitewater source sphere exceeds fixed grid cell budget");
    }
    box.cell_count *= static_cast<std::size_t>(extent);
  }
  return box;
}
}  // namespace

absl::StatusOr<std::vector<WhitewaterCellBirth3D>> SampleAcceptedWhitewaterSphereStep3D(
    const LiquidState3D& accepted,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& step_options,
    Vec3 world_source_grid_origin,
    double previous_accepted_time_seconds,
    const WhitewaterSourceOptions3D& source_options,
    std::size_t maximum_births
) {
  const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field);
  if (sphere == nullptr || !std::holds_alternative<std::monostate>(scene.obstacle)) {
    return absl::UnimplementedError("whitewater sphere source requires analytic liquid without obstacles");
  }
  const double spacing_m = source_options.spacing_m;
  const double source_layer_m = 2 * spacing_m;
  if (!Finite(sphere->world_center) || !Finite(world_source_grid_origin) || !std::isfinite(spacing_m) || spacing_m <= 0
      || !std::isfinite(sphere->radius) || sphere->radius <= source_layer_m
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
  const auto box = SphereSourceCellBox(*sphere, world_source_grid_origin, spacing_m);
  if (!box.ok()) {
    return box.status();
  }
  const auto carrier = FrozenMotionCarrier3D::Create(accepted, scene, step_options);
  if (!carrier.ok()) {
    return carrier.status();
  }
  const double curvature_per_m = 2 / sphere->radius;
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
        const Vec3 radial{
            center.x - sphere->world_center.x,
            center.y - sphere->world_center.y,
            center.z - sphere->world_center.z,
        };
        const double center_distance_m = std::hypot(radial.x, radial.y, radial.z);
        if (!std::isfinite(center_distance_m)) {
          return absl::OutOfRangeError("whitewater source sphere coordinate overflowed");
        }
        if (center_distance_m < sphere->radius - source_layer_m) {
          continue;
        }
        unsigned occupied_probes = 0;
        for (const double dz : offsets) {
          for (const double dy : offsets) {
            for (const double dx : offsets) {
              occupied_probes += std::hypot(radial.x + dx, radial.y + dy, radial.z + dz) < sphere->radius;
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
