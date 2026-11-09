#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <tuple>

#include "liquid_substep_3d_internal.h"
namespace fluidity {
std::array<std::uint32_t, 4>
LiquidPhilox4x32_10(std::array<std::uint32_t, 4> counter, std::array<std::uint32_t, 2> key) {
  for (int round = 0; round < 10; ++round) {
    const std::uint64_t first = std::uint64_t{0xd2511f53} * counter[0];
    const std::uint64_t second = std::uint64_t{0xcd9e8d57} * counter[2];
    counter = {
        static_cast<std::uint32_t>(second >> 32) ^ counter[1] ^ key[0],
        static_cast<std::uint32_t>(second),
        static_cast<std::uint32_t>(first >> 32) ^ counter[3] ^ key[1],
        static_cast<std::uint32_t>(first),
    };
    key[0] += 0x9e3779b9;
    key[1] += 0xbb67ae85;
  }
  return counter;
}
namespace liquid_step_internal {
#define STEP_TRY(expression) \
  do { \
    const auto status = (expression); \
    if (!status.ok()) { \
      return status; \
    } \
  } while (false)
bool Finite(Vec3 v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
double Component(Vec3 v, FaceNormalAxis axis) {
  return axis == FaceNormalAxis::kX ? v.x : (axis == FaceNormalAxis::kY ? v.y : v.z);
}
double MaximumComponentMagnitude(Vec3 v) {
  return std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)});
}
double Dot(Vec3 left, Vec3 right) {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}
Vec3 Add(Vec3 left, Vec3 right) {
  return {left.x + right.x, left.y + right.y, left.z + right.z};
}
Vec3 Scale(Vec3 a, double factor) {
  return {factor * a.x, factor * a.y, factor * a.z};
}
Vec3 WallVelocity(const LiquidObstacle3D& obstacle) {
  if (const auto* wall = std::get_if<TranslatingHalfspace3D>(&obstacle)) {
    return wall->wall_velocity;
  }
  if (const auto* wall = std::get_if<TranslatingSlab3D>(&obstacle)) {
    return wall->wall_velocity;
  }
  return {};
}
LiquidObstacle3D ObstacleAtTime(const LiquidObstacle3D& obstacle, double time) {
  auto result = obstacle;
  if (auto* wall = std::get_if<TranslatingHalfspace3D>(&result)) {
    wall->start_plane_coordinate += time * Dot(wall->unit_normal, wall->wall_velocity);
  }
  if (auto* wall = std::get_if<TranslatingSlab3D>(&result)) {
    wall->start_center_coordinate += time * Dot(wall->unit_normal, wall->wall_velocity);
  }
  return result;
}
double SolidDistance(const LiquidObstacle3D& obstacle, Vec3 position) {
  if (const auto* wall = std::get_if<TranslatingHalfspace3D>(&obstacle)) {
    return Dot(wall->unit_normal, position) - wall->start_plane_coordinate;
  }
  if (const auto* wall = std::get_if<TranslatingSlab3D>(&obstacle)) {
    return std::abs(Dot(wall->unit_normal, position) - wall->start_center_coordinate) - wall->half_width;
  }
  return std::numeric_limits<double>::infinity();
}
double FieldDistance(const LiquidSceneField3D& field, std::span<const LiquidParticleState3D> particles, Vec3 position) {
  const auto distance = [&](Vec3 center, double radius) {
    return std::hypot(position.x - center.x, position.y - center.y, position.z - center.z) - radius;
  };
  if (const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&field)) {
    return distance(sphere->world_center, sphere->radius);
  }
  double result = std::numeric_limits<double>::infinity();
  for (const auto& particle : particles) {
    result = std::min(result, distance(particle.world_position, std::get<LiquidParticleSphereUnion3D>(field).radius));
  }
  return result;
}
namespace {
template <typename T>
absl::Status Reserve(LiquidRecords3D<T>& records, std::size_t count, std::size_t& bytes) {
  if (count > (static_cast<std::size_t>(PTRDIFF_MAX) - bytes) / sizeof(T)) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  auto owner = LiquidRecords3D<T>::Create(count);
  if (!owner.ok()) {
    return owner.status();
  }
  bytes += count * sizeof(T);
  records = std::move(*owner);
  return absl::OkStatus();
}
auto Key(const auto& value) {
  return std::tie(value.z, value.y, value.x);
}
template <typename T, typename KeyFunction>
absl::Status InsertUnique(LiquidRecords3D<T>& records, T value, KeyFunction key) {
  auto entries = records.values();
  const auto found = std::lower_bound(entries.begin(), entries.end(), value, [&](const T& left, const T& right) {
    return key(left) < key(right);
  });
  if (found != entries.end() && key(*found) == key(value)) {
    return absl::OkStatus();
  }
  const std::size_t ordinal = entries.empty() ? 0 : static_cast<std::size_t>(found - entries.begin());
  const auto old_size = records.size();
  STEP_TRY(records.Resize(old_size + 1));
  for (std::size_t index = old_size; index > ordinal; --index) {
    records[index] = records[index - 1];
  }
  records[ordinal] = value;
  return absl::OkStatus();
}
}  // namespace
absl::StatusOr<Staging> Staging::Create(LiquidStateCapacities3D capacities) {
  if (capacities.cells > 4096 || capacities.faces > 24576 || capacities.particles > 1024) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  Staging staging;
  STEP_TRY(Reserve(staging.candidates, capacities.cells, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.centers, 27 * capacities.cells, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.corners, 8 * capacities.cells, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.particles, capacities.particles, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.tags, capacities.faces, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.new_tags, capacities.faces, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.old_velocities, capacities.faces, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.new_velocities, capacities.faces, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.validity, capacities.faces, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.old_walls, capacities.faces, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.new_walls, capacities.faces, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.pressure_fields, capacities.cells, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.incident_velocities, capacities.faces, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.gathered, capacities.particles, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.particle_distances, capacities.particles, staging.reserved_bytes));
  return staging;
}
absl::Status
ValidateScene(const LiquidState3D& initial, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options) {
  const bool valid_material = std::isnormal(scene.liquid_density) && scene.liquid_density > 0
      && std::isfinite(scene.surface_tension) && scene.surface_tension >= 0 && std::isfinite(scene.ambient_pressure)
      && std::isfinite(scene.target_divergence) && scene.bulk_modulus > 0;
  if (!valid_material) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const bool valid_step = std::isfinite(options.particle_energy_blend) && options.particle_energy_blend >= 0
      && options.particle_energy_blend <= 1 && std::isfinite(options.maximum_step_duration)
      && options.maximum_step_duration > 0 && std::isfinite(options.minimum_face_mass) && options.minimum_face_mass >= 0
      && options.pressure_solver_options.max_iterations >= 1 && options.pressure_solver_options.max_iterations <= 256;
  if (!valid_step) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const bool valid_forces = Finite(scene.world_acceleration) && std::isfinite(scene.friction_coefficient)
      && scene.friction_coefficient >= 0 && std::isfinite(scene.penetration_tolerance)
      && scene.penetration_tolerance >= 0;
  if (!valid_forces) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const auto growth = options.capacity_replay_reservations;
  const auto initial_capacity = options.capacities;
  if (growth != LiquidStateCapacities3D{}) {
    const bool nondecreasing = growth.particles >= initial_capacity.particles
        && growth.blocks >= initial_capacity.blocks && growth.cells >= initial_capacity.cells
        && growth.faces >= initial_capacity.faces && growth.sources >= initial_capacity.sources;
    if (!nondecreasing || growth == initial_capacity) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  for (const double scale : {options.mass_per_quantum, options.momentum_per_quantum}) {
    int exponent = 0;
    if (!std::isnormal(scale) || scale <= 0 || std::frexp(scale, &exponent) != 0.5) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field);
  const double radius = sphere ? sphere->radius : std::get<LiquidParticleSphereUnion3D>(scene.field).radius;
  if (!std::isfinite(radius) || radius <= 0 || (sphere && !Finite(sphere->world_center))) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (scene.sources.size() != initial.sources.size()) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  for (std::size_t index = 0; index < scene.sources.size(); ++index) {
    const auto& source = scene.sources[index];
    const auto& particle = source.emitted_particle_template;
    if (source.source_ordinal != initial.sources[index].source_ordinal || !std::isfinite(source.particles_per_second)
        || source.particles_per_second < 0 || !Finite(source.world_position) || !Finite(source.world_position_jitter)
        || source.world_position_jitter.x < 0 || source.world_position_jitter.y < 0
        || source.world_position_jitter.z < 0 || !Finite(particle.world_velocity)
        || !Finite(particle.velocity_slopes.x_row) || !Finite(particle.velocity_slopes.y_row)
        || !Finite(particle.velocity_slopes.z_row) || !std::isfinite(particle.mass) || particle.mass <= 0
        || !std::isfinite(particle.reference_volume) || particle.reference_volume <= 0
        || !std::isfinite(particle.volume_ratio) || particle.volume_ratio <= 0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  // The checked contact operator validates the same translated obstacle contract.
  const auto contact = ResolveLiquidWallContactBounded3D({
      .start_world_position = {},
      .obstacle = scene.obstacle,
      .step_duration = 1,
      .friction_coefficient = scene.friction_coefficient,
      .penetration_tolerance = std::numeric_limits<double>::max(),
  });
  if (!contact.ok() && contact.status().code() == absl::StatusCode::kInvalidArgument) {
    return contact.status();
  }
  return absl::OkStatus();
}
absl::Status Emit(const LiquidSceneInputs3D& scene, double duration, LiquidState3D& trial) {
  for (std::size_t ordinal = 0; ordinal < scene.sources.size(); ++ordinal) {
    const auto& source = scene.sources[ordinal];
    auto& state = trial.sources[ordinal];
    const double requested = state.fractional_remainder + duration * source.particles_per_second;
    if (!std::isfinite(requested) || requested >= 0x1p64) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    const auto count = static_cast<std::uint64_t>(std::floor(requested));
    if (count > trial.particles.capacity() - trial.particles.size()) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    if (count > UINT64_MAX - trial.next_particle_id || count > UINT64_MAX - state.next_child_ordinal) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    const auto stream = LiquidPhilox4x32_10(
        {
            static_cast<std::uint32_t>(trial.clock.accepted_step_ordinal),
            static_cast<std::uint32_t>(trial.clock.accepted_step_ordinal >> 32),
            static_cast<std::uint32_t>(source.source_ordinal),
            static_cast<std::uint32_t>(source.source_ordinal >> 32),
        },
        {static_cast<std::uint32_t>(trial.random_seed), static_cast<std::uint32_t>(trial.random_seed >> 32)}
    );
    for (std::uint64_t child = 0; child < count; ++child) {
      const auto identity = state.next_child_ordinal + child;
      const auto random = LiquidPhilox4x32_10(
          {static_cast<std::uint32_t>(identity), static_cast<std::uint32_t>(identity >> 32), stream[2], stream[3]},
          {stream[0], stream[1]}
      );
      auto particle = source.emitted_particle_template;
      particle.id = trial.next_particle_id++;
      particle.pic_initialized = false;
      const Vec3 jitter{
          2 * (static_cast<double>(random[0]) + 0.5) * 0x1p-32 - 1,
          2 * (static_cast<double>(random[1]) + 0.5) * 0x1p-32 - 1,
          2 * (static_cast<double>(random[2]) + 0.5) * 0x1p-32 - 1,
      };
      particle.world_position =
          Add(source.world_position,
              {
                  jitter.x * source.world_position_jitter.x,
                  jitter.y * source.world_position_jitter.y,
                  jitter.z * source.world_position_jitter.z,
              });
      const auto index = trial.particles.size();
      STEP_TRY(trial.particles.Resize(index + 1));
      trial.particles[index] = particle;
    }
    state.next_child_ordinal += count;
    state.fractional_remainder = requested - static_cast<double>(count);
  }
  return absl::OkStatus();
}
absl::Status PrepareField(
    const LiquidSceneField3D& field,
    const LiquidState3D& trial,
    Staging& staging,
    const LiquidSubstepBackend3D& backend
) {
  STEP_TRY(staging.candidates.Resize(0));
  STEP_TRY(staging.centers.Resize(0));
  STEP_TRY(staging.corners.Resize(0));
  const auto geometry = trial.geometry;
  const auto add_sphere = [&](Vec3 center, double radius) -> absl::Status {
    std::array<std::int32_t, 3> lower{}, upper{};
    const std::array centers{center.x, center.y, center.z};
    const std::array origins{geometry.world_origin.x, geometry.world_origin.y, geometry.world_origin.z};
    std::size_t box_count = 1;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double low = std::floor((centers[axis] - radius - origins[axis]) / geometry.cell_width);
      const double high = std::floor((centers[axis] + radius - origins[axis]) / geometry.cell_width);
      if (!std::isfinite(low) || !std::isfinite(high) || low < static_cast<double>(INT32_MIN) + 1
          || high > static_cast<double>(INT32_MAX) - 1) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      lower[axis] = static_cast<std::int32_t>(low);
      upper[axis] = static_cast<std::int32_t>(high);
      const auto extent = static_cast<std::size_t>(static_cast<std::int64_t>(upper[axis]) - lower[axis] + 1);
      if (extent > staging.candidates.capacity() / box_count) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      box_count *= extent;
    }
    for (std::int64_t z = lower[2]; z <= upper[2]; ++z) {
      for (std::int64_t y = lower[1]; y <= upper[1]; ++y) {
        for (std::int64_t x = lower[0]; x <= upper[0]; ++x) {
          STEP_TRY(InsertUnique(
              staging.candidates,
              CellIndex3D{static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), static_cast<std::int32_t>(z)},
              [](const auto& v) {
                return Key(v);
              }
          ));
        }
      }
    }
    return absl::OkStatus();
  };
  if (const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&field)) {
    STEP_TRY(add_sphere(sphere->world_center, sphere->radius));
  } else {
    for (const auto& particle : trial.particles.values()) {
      STEP_TRY(add_sphere(particle.world_position, std::get<LiquidParticleSphereUnion3D>(field).radius));
    }
  }
  for (const auto& cell : staging.candidates.values()) {
    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const CellIndex3D index{cell.x + x, cell.y + y, cell.z + z};
          const Vec3 position{
              geometry.world_origin.x + (static_cast<double>(index.x) + 0.5) * geometry.cell_width,
              geometry.world_origin.y + (static_cast<double>(index.y) + 0.5) * geometry.cell_width,
              geometry.world_origin.z + (static_cast<double>(index.z) + 0.5) * geometry.cell_width,
          };
          const auto sampled = backend.sample_field(backend.context, field, trial.particles.values(), position);
          if (!sampled.ok()) {
            return sampled.status();
          }
          const double distance = *sampled;
          if (!std::isfinite(distance)) {
            return Error(absl::StatusCode::kOutOfRange);
          }
          STEP_TRY(InsertUnique(staging.centers, LiquidCenterDistance3D{index, distance}, [](const auto& v) {
            return Key(v.cell_index);
          }));
        }
      }
    }
    for (int z = 0; z <= 1; ++z) {
      for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
          const CellCornerIndex3D index{cell.x + x, cell.y + y, cell.z + z};
          const Vec3 position{
              geometry.world_origin.x + index.x * geometry.cell_width,
              geometry.world_origin.y + index.y * geometry.cell_width,
              geometry.world_origin.z + index.z * geometry.cell_width,
          };
          const auto sampled = backend.sample_field(backend.context, field, trial.particles.values(), position);
          if (!sampled.ok()) {
            return sampled.status();
          }
          const double distance = *sampled;
          if (!std::isfinite(distance)) {
            return Error(absl::StatusCode::kOutOfRange);
          }
          STEP_TRY(InsertUnique(staging.corners, LiquidCornerDistance3D{index, distance}, [](const auto& v) {
            return Key(v.corner_index);
          }));
        }
      }
    }
  }
  return absl::OkStatus();
}
absl::Status PrepareTransfers(const LiquidState3D& trial, SparseMacGridView3D grid, Staging& staging) {
  STEP_TRY(staging.particles.Resize(trial.particles.size()));
  STEP_TRY(staging.gathered.Resize(trial.particles.size()));
  STEP_TRY(staging.particle_distances.Resize(trial.particles.size()));
  for (std::size_t index = 0; index < trial.particles.size(); ++index) {
    const auto& particle = trial.particles[index];
    staging.particles[index] = {
        .transfer =
            {
                .sorted_world_position = particle.world_position,
                .trial_world_position = particle.world_position,
                .particle_mass = particle.mass,
                .particle_velocity = particle.world_velocity,
                .particle_velocity_slopes = particle.velocity_slopes,
            },
        .side_tag = particle.side_tag,
        .affine_enabled = particle.affine_enabled,
    };
  }
  const auto count = grid.face_count();
  STEP_TRY(staging.tags.Resize(count));
  STEP_TRY(staging.new_tags.Resize(count));
  STEP_TRY(staging.validity.Resize(count));
  STEP_TRY(staging.old_velocities.Resize(count));
  STEP_TRY(staging.new_velocities.Resize(count));
  STEP_TRY(staging.old_walls.Resize(count));
  STEP_TRY(staging.new_walls.Resize(count));
  return absl::OkStatus();
}
absl::Status
FillFaceTags(SparseMacGridView3D grid, const LiquidObstacle3D& obstacle, std::span<LiquidBoundaryFaceTag3D> tags) {
  for (std::size_t index = 0; index < tags.size(); ++index) {
    const auto face = grid.FaceIndex(index);
    if (!face.ok()) {
      return face.status();
    }
    const auto anchor = grid.FaceWorldAnchor(*face);
    if (!anchor.ok()) {
      return anchor.status();
    }
    std::int64_t side = 0;
    if (const auto* slab = std::get_if<TranslatingSlab3D>(&obstacle)) {
      side = Dot(slab->unit_normal, *anchor) >= slab->start_center_coordinate ? 1 : -1;
    }
    tags[index] = {side, SolidDistance(obstacle, *anchor) >= 0};
  }
  return absl::OkStatus();
}
absl::Status FillWalls(
    SparseMacGridView3D grid,
    const LiquidObstacle3D& obstacle,
    std::int64_t side,
    std::span<LiquidPrescribedWallSample3D> walls
) {
  for (std::size_t index = 0; index < walls.size(); ++index) {
    walls[index] = {};
    if (std::holds_alternative<std::monostate>(obstacle)) {
      continue;
    }
    const auto face = grid.FaceIndex(index);
    if (!face.ok()) {
      return face.status();
    }
    const auto anchor = grid.FaceWorldAnchor(*face);
    if (!anchor.ok()) {
      return anchor.status();
    }
    if (std::abs(SolidDistance(obstacle, *anchor)) > 2 * grid.geometry().cell_width) {
      continue;
    }
    Vec3 normal{};
    if (const auto* wall = std::get_if<TranslatingHalfspace3D>(&obstacle)) {
      normal = wall->unit_normal;
    }
    if (const auto* wall = std::get_if<TranslatingSlab3D>(&obstacle)) {
      if (side != 1 && side != -1) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
      normal = Scale(wall->unit_normal, static_cast<double>(side));
    }
    walls[index] = {true, side, normal, WallVelocity(obstacle)};
  }
  return absl::OkStatus();
}
#undef STEP_TRY
}  // namespace liquid_step_internal
}  // namespace fluidity
