#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <tuple>

#include "liquid_boundary_backend.h"
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
namespace {

auto PressureCellKey(CellIndex3D cell) {
  return std::tuple{cell.z, cell.y, cell.x};
}

// Matches the owned-slot order even when the set of active blocks changes.
auto CarrierFaceKey(FaceIndex3D face) {
  const auto owner = MacOwnerBlock3D(face.lattice_index);
  const auto index = face.lattice_index;
  return std::tuple{owner.z, owner.y, owner.x, face.axis, index.z, index.y, index.x};
}

std::optional<double> CommittedCarrierVelocity(std::span<const LiquidFaceState3D> faces, FaceIndex3D face) {
  const auto key = CarrierFaceKey(face);
  const auto found = std::ranges::lower_bound(faces, key, {}, [](const LiquidFaceState3D& record) {
    return CarrierFaceKey(record.face_index);
  });
  if (found == faces.end() || CarrierFaceKey(found->face_index) != key || !found->structurally_eligible) {
    return std::nullopt;
  }
  if (found->validity != LiquidFaceValidity3D::kValid && found->validity != LiquidFaceValidity3D::kPredicted) {
    return std::nullopt;
  }
  return found->carrier_velocity;
}

std::optional<double> CommittedLiquidPressure(std::span<const LiquidCellState3D> cells, CellIndex3D cell) {
  const auto key = PressureCellKey(cell);
  const auto found = std::ranges::lower_bound(cells, key, {}, [](const LiquidCellState3D& record) {
    return PressureCellKey(record.cell_index);
  });
  if (found == cells.end() || PressureCellKey(found->cell_index) != key
      || found->occupancy != LiquidCellOccupancy3D::kLiquid) {
    return std::nullopt;
  }
  return found->previous_pressure;
}

}  // namespace

absl::StatusOr<double> AdvectLiquidPressureHistory3D(
    const LiquidState3D& committed,
    CellIndex3D cell,
    double step_duration,
    double ambient_pressure
) {
  const auto cells = committed.cells.values();
  const auto faces = committed.faces.values();
  const auto fallback = CommittedLiquidPressure(cells, cell).value_or(ambient_pressure);
  if (cells.empty()) {
    return fallback;
  }
  const std::array coordinates{cell.x, cell.y, cell.z};
  constexpr std::array axes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
  std::array<std::int64_t, 3> low_donor{};
  std::array<double, 3> fraction{};
  const double lattice_step = step_duration / committed.geometry.cell_width;
  constexpr auto kMinIndex = std::numeric_limits<std::int32_t>::min();
  constexpr auto kMaxIndex = std::numeric_limits<std::int32_t>::max();
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    const auto low_velocity = CommittedCarrierVelocity(faces, {axes[axis], {cell.x, cell.y, cell.z}});
    std::optional<double> high_velocity;
    if (coordinates[axis] < kMaxIndex) {
      auto high_face = coordinates;
      ++high_face[axis];
      high_velocity = CommittedCarrierVelocity(faces, {axes[axis], {high_face[0], high_face[1], high_face[2]}});
    }
    double centered_velocity = low_velocity.value_or(high_velocity.value_or(0.0));
    if (low_velocity && high_velocity) {
      centered_velocity = std::midpoint(*low_velocity, *high_velocity);
    }
    const double displacement = centered_velocity == 0.0 ? 0.0 : lattice_step * centered_velocity;
    if (!std::isfinite(displacement)) {
      return absl::OutOfRangeError("pressure backtrace displacement is unrepresentable");
    }
    // Keep the fractional departure separate from the large signed cell index.
    const double low_offset = std::floor(-displacement);
    if (low_offset < static_cast<double>(kMinIndex) - coordinates[axis] - 1.0
        || low_offset > static_cast<double>(kMaxIndex) - coordinates[axis]) {
      return fallback;
    }
    low_donor[axis] = static_cast<std::int64_t>(coordinates[axis]) + static_cast<std::int64_t>(low_offset);
    fraction[axis] = -displacement - low_offset;
  }
  double predicted_pressure = 0.0;
  double total_weight = 0.0;
  for (std::size_t corner = 0; corner < 8; ++corner) {
    const std::array upper{corner & 1, (corner >> 1) & 1, (corner >> 2) & 1};
    const std::array index{
        low_donor[0] + static_cast<std::int64_t>(upper[0]),
        low_donor[1] + static_cast<std::int64_t>(upper[1]),
        low_donor[2] + static_cast<std::int64_t>(upper[2]),
    };
    if (std::ranges::any_of(index, [](std::int64_t coordinate) {
          return coordinate < kMinIndex || coordinate > kMaxIndex;
        })) {
      continue;
    }
    const double x_weight = upper[0] ? fraction[0] : 1.0 - fraction[0];
    const double y_weight = upper[1] ? fraction[1] : 1.0 - fraction[1];
    const double z_weight = upper[2] ? fraction[2] : 1.0 - fraction[2];
    const double weight = x_weight * y_weight * z_weight;
    if (weight == 0.0) {
      continue;
    }
    const auto pressure = CommittedLiquidPressure(
        cells,
        {static_cast<std::int32_t>(index[0]), static_cast<std::int32_t>(index[1]), static_cast<std::int32_t>(index[2])}
    );
    if (pressure) {
      total_weight += weight;
      predicted_pressure = std::lerp(predicted_pressure, *pressure, weight / total_weight);
    }
  }
  return total_weight == 0.0 ? fallback : predicted_pressure;
}

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
  if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
    return {tank->piston_velocity_x, 0, 0};
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
  if (auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&result)) {
    tank->start_low_x = std::fma(tank->piston_velocity_x, time, tank->start_low_x);
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
  if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
    const auto wall = boundary_internal::SampleTankWall(*tank, position);
    return wall.ok() ? wall->inward_distance : std::numeric_limits<double>::quiet_NaN();
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
  if (const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&field)) {
    return position.y - plane->world_surface_height_y;
  }
  const auto* spheres = std::get_if<LiquidParticleSphereUnion3D>(&field);
  if (!spheres) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double result = std::numeric_limits<double>::infinity();
  for (const auto& particle : particles) {
    result = std::min(result, distance(particle.world_position, spheres->radius));
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
absl::StatusOr<Staging> Staging::Create(
    LiquidStateCapacities3D capacities,
    bool needs_regular_transport,
    bool needs_wetted_piston,
    bool needs_isolated_drop,
    bool needs_drift_sensors,
    bool needs_incompressible_drift
) {
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
  if (needs_drift_sensors) {
    STEP_TRY(Reserve(staging.drift_sensors.emplace(), capacities.cells, staging.reserved_bytes));
  }
  if (needs_incompressible_drift) {
    STEP_TRY(Reserve(staging.drift_pressure_cells.emplace(), capacities.cells, staging.reserved_bytes));
    STEP_TRY(Reserve(staging.sealed_drift_cells.emplace(), capacities.cells, staging.reserved_bytes));
  }
  STEP_TRY(Reserve(staging.incident_velocities, capacities.faces, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.gathered, capacities.particles, staging.reserved_bytes));
  STEP_TRY(Reserve(staging.particle_distances, capacities.particles, staging.reserved_bytes));
  if (needs_regular_transport) {
    STEP_TRY(Reserve(staging.regular_transport.emplace(), capacities.particles, staging.reserved_bytes));
  }
  if (needs_wetted_piston || needs_isolated_drop) {
    STEP_TRY(Reserve(staging.projected_gather_validity.emplace(), capacities.faces, staging.reserved_bytes));
  }
  if (needs_wetted_piston) {
    auto workspace = WettedPistonWorkspace3D::Create(capacities);
    if (!workspace.ok()) {
      return workspace.status();
    }
    if (workspace->reserved_bytes() > static_cast<std::size_t>(PTRDIFF_MAX) - staging.reserved_bytes) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    staging.reserved_bytes += workspace->reserved_bytes();
    staging.wetted_piston.emplace(std::move(*workspace));
  }
  if (needs_isolated_drop) {
    STEP_TRY(Reserve(staging.isolated_drop_restricted_gathers.emplace(), capacities.particles, staging.reserved_bytes));
    STEP_TRY(Reserve(staging.isolated_drop_pressure_faces.emplace(), capacities.faces, staging.reserved_bytes));
  }
  return staging;
}
absl::Status
ValidateScene(const LiquidState3D& initial, const LiquidSceneInputs3D& scene, const LiquidStepOptions3D& options) {
  const bool valid_material = std::isnormal(scene.liquid_density) && scene.liquid_density > 0
      && std::isfinite(scene.dynamic_viscosity) && scene.dynamic_viscosity >= 0.0
      && std::isfinite(scene.surface_tension) && scene.surface_tension >= 0 && std::isfinite(scene.ambient_pressure)
      && std::isfinite(scene.target_divergence) && scene.bulk_modulus > 0;
  if (!valid_material || !ValidLiquidConstituentMaterials3D(scene.constituent_materials)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (std::ranges::any_of(initial.particles.values(), [&](const LiquidParticleState3D& particle) {
        return !LiquidParticleUsesRegisteredConstituents3D(particle, scene.constituent_materials);
      })) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  if (scene.eos) {
    if (!std::isinf(scene.bulk_modulus) || !std::isfinite(scene.eos->reference_sound_speed)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    const auto eos = EvaluateLiquidEosAtPressure3D(
        {scene.liquid_density, scene.eos->reference_sound_speed, scene.eos->law},
        scene.ambient_pressure
    );
    if (!eos.ok()) {
      return eos.status();
    }
  }
  if (scene.drift
      && ((!scene.eos && !std::isinf(scene.bulk_modulus)) || !std::isfinite(scene.drift->correction_time)
          || scene.drift->correction_time <= 0.0 || !std::isfinite(scene.drift->maximum_rate)
          || scene.drift->maximum_rate < 0.0)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  const bool valid_step = std::isfinite(options.particle_energy_blend) && options.particle_energy_blend >= 0
      && options.particle_energy_blend <= 1 && std::isfinite(options.maximum_step_duration)
      && options.maximum_step_duration > 0 && std::isfinite(options.minimum_face_mass) && options.minimum_face_mass >= 0
      && options.pressure_solver_options.max_iterations >= 1 && options.pressure_solver_options.max_iterations <= 256
      && options.motion_sample_intervals <= 64 && (!options.whitewater_capture || options.motion_sample_intervals != 0);
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
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.obstacle);
  if (const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field)) {
    if (!std::isfinite(sphere->radius) || sphere->radius <= 0 || !Finite(sphere->world_center)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  } else if (const auto* spheres = std::get_if<LiquidParticleSphereUnion3D>(&scene.field)) {
    if (!std::isfinite(spheres->radius) || spheres->radius <= 0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  } else if (const auto* drop = std::get_if<LiquidIsolatedCovarianceDrop3D>(&scene.field)) {
    if (!std::isfinite(drop->physical_radius) || drop->physical_radius <= 0
        || !std::isfinite(drop->seed_covariance_scale) || drop->seed_covariance_scale <= 0
        || !std::holds_alternative<std::monostate>(scene.obstacle) || !scene.sources.empty()
        || scene.surface_tension <= 0 || !std::isinf(scene.bulk_modulus) || scene.eos || scene.drift
        || scene.target_divergence != 0 || initial.particles.size() != kMaxIsolatedDropParticles3D) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    for (const auto& particle : initial.particles.values()) {
      if (particle.side_tag != 0) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
    }
  } else {
    const auto& plane = std::get<LiquidHorizontalPlaneSurface3D>(scene.field);
    if (!tank || !std::isfinite(plane.world_surface_height_y) || plane.world_surface_height_y <= tank->floor_y) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  if (tank) {
    if (scene.friction_coefficient != 0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    STEP_TRY(boundary_internal::ArithmeticStatus(boundary_internal::CheckTankStep(*tank, initial.clock.time), true));
    for (const auto& particle : initial.particles.values()) {
      if (particle.side_tag != 0) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
    }
  }
  if (scene.drift && scene.drift->maximum_rate > 0.0
      && std::ranges::any_of(initial.particles.values(), [](const LiquidParticleState3D& particle) {
           return particle.side_tag != 0;
         })) {
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
        || !std::isfinite(particle.volume_ratio) || particle.volume_ratio <= 0
        || !LiquidParticleUsesRegisteredConstituents3D(particle, scene.constituent_materials)
        || ((tank || (scene.drift && scene.drift->maximum_rate > 0.0)) && particle.side_tag != 0)) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
  }
  if (tank) {
    return absl::OkStatus();
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
absl::StatusOr<OceanInletParticles3D>
OceanInletParticles3D::Create(const OceanBoundarySettings3D& settings, double density) {
  OceanInletParticles3D inlet;
  const auto& swell = settings.swell;
  inlet.spacing_m_ = settings.inlet_particle_spacing_m;
  inlet.lower_m_ = {settings.inlet_wall_x_m, swell.still_water_height_m - swell.water_depth_m, settings.inlet_low_z_m};
  const Vec3 requested_upper_m{
      settings.inlet_interior_x_m,
      swell.still_water_height_m + swell.wave_height_m,
      settings.inlet_high_z_m
  };
  const std::array widths_m{
      requested_upper_m.x - inlet.lower_m_.x,
      requested_upper_m.y - inlet.lower_m_.y,
      requested_upper_m.z - inlet.lower_m_.z
  };
  constexpr std::size_t kMaximumSites = 4096;
  std::size_t site_count = 1;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double sites = std::floor(widths_m[axis] / inlet.spacing_m_);
    if (!std::isfinite(sites) || sites < 1 || sites > kMaximumSites / site_count) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    inlet.extent_[axis] = static_cast<std::size_t>(sites);
    site_count *= inlet.extent_[axis];
  }
  inlet.upper_m_ = {
      inlet.lower_m_.x + inlet.extent_[0] * inlet.spacing_m_,
      inlet.lower_m_.y + inlet.extent_[1] * inlet.spacing_m_,
      inlet.lower_m_.z + inlet.extent_[2] * inlet.spacing_m_
  };
  if (!Finite(inlet.upper_m_) || inlet.upper_m_.x <= inlet.lower_m_.x || inlet.upper_m_.y <= inlet.lower_m_.y
      || inlet.upper_m_.z <= inlet.lower_m_.z) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  inlet.particle_volume_m3_ = inlet.spacing_m_ * inlet.spacing_m_ * inlet.spacing_m_;
  inlet.particle_mass_kg_ = inlet.particle_volume_m3_ * density;
  if (!std::isnormal(inlet.particle_volume_m3_) || !std::isnormal(inlet.particle_mass_kg_)) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  auto occupied = LiquidRecords3D<std::uint8_t>::Create(site_count);
  if (!occupied.ok()) {
    return occupied.status();
  }
  STEP_TRY(occupied->Resize(site_count));
  inlet.occupied_ = std::move(*occupied);
  auto births = LiquidRecords3D<LiquidParticleState3D>::Create(site_count);
  if (!births.ok()) {
    return births.status();
  }
  inlet.births_ = std::move(*births);
  return inlet;
}

std::size_t OceanInletParticles3D::reserved_bytes() const {
  return occupied_.capacity() * sizeof(std::uint8_t) + births_.capacity() * sizeof(LiquidParticleState3D);
}

absl::Status OceanInletParticles3D::Apply(
    const ResolvedOceanBoundary3D& boundary,
    const LiquidObstacle3D& obstacle,
    double time_s,
    LiquidState3D& trial
) {
  std::ranges::fill(occupied_.values(), 0);
  STEP_TRY(births_.Resize(0));
  const auto slot = [&](std::size_t x, std::size_t y, std::size_t z) {
    return (z * extent_[1] + y) * extent_[0] + x;
  };
  // Preserve the existing particle order. Only particles in the controlled
  // strip and above its prescribed surface are retired.
  std::size_t retained = 0;
  for (const auto& particle : trial.particles.values()) {
    const auto position = particle.world_position;
    if (position.x >= lower_m_.x && position.x < upper_m_.x && position.y >= lower_m_.y && position.y < upper_m_.y
        && position.z >= lower_m_.z && position.z < upper_m_.z) {
      const auto sample = boundary.SampleInlet(position, time_s);
      if (!sample.ok()) {
        return sample.status();
      }
      if (position.y > sample->surface_height_m) {
        continue;
      }
      const auto index = [&](double coordinate, double lower, std::size_t extent) {
        return std::min(static_cast<std::size_t>((coordinate - lower) / spacing_m_), extent - 1);
      };
      occupied_[slot(
          index(position.x, lower_m_.x, extent_[0]),
          index(position.y, lower_m_.y, extent_[1]),
          index(position.z, lower_m_.z, extent_[2])
      )] = 1;
    }
    trial.particles[retained++] = particle;
  }
  // Candidate sites are visited in signed-world z/y/x order. Existing wet
  // particles claim their lattice bin before any new particle is considered.
  for (std::size_t z = 0; z < extent_[2]; ++z) {
    for (std::size_t y = 0; y < extent_[1]; ++y) {
      for (std::size_t x = 0; x < extent_[0]; ++x) {
        if (occupied_[slot(x, y, z)] != 0) {
          continue;
        }
        const Vec3 position{
            lower_m_.x + (x + 0.5) * spacing_m_,
            lower_m_.y + (y + 0.5) * spacing_m_,
            lower_m_.z + (z + 0.5) * spacing_m_,
        };
        if (!(SolidDistance(obstacle, position) > 0)) {
          continue;
        }
        const auto sample = boundary.SampleInlet(position, time_s);
        if (!sample.ok()) {
          return sample.status();
        }
        if (position.y >= sample->surface_height_m) {
          continue;
        }
        const auto birth = births_.size();
        STEP_TRY(births_.Resize(birth + 1));
        births_[birth] = {
            .world_position = position,
            .world_velocity = sample->world_velocity_m_per_s,
            .mass = particle_mass_kg_,
            .reference_volume = particle_volume_m3_,
        };
      }
    }
  }
  if (births_.size() > trial.particles.capacity() - retained) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  if (births_.size() > UINT64_MAX - trial.next_particle_id) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  STEP_TRY(trial.particles.Resize(retained));
  STEP_TRY(trial.particles.Resize(retained + births_.size()));
  for (std::size_t birth = 0; birth < births_.size(); ++birth) {
    births_[birth].id = trial.next_particle_id++;
    trial.particles[retained + birth] = births_[birth];
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
namespace {
// The division estimates the signed lattice index. Certify both bounding
// integer planes with the same residual arithmetic used by tank clipping.
absl::StatusOr<std::int32_t>
TankCandidateBound(double world_coordinate, double world_origin, double cell_width, bool upper) {
  const double estimate = std::floor((world_coordinate - world_origin) / cell_width);
  if (!std::isfinite(estimate) || estimate < INT32_MIN || estimate > INT32_MAX) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  for (const double index : {estimate, estimate - 1, estimate + 1}) {
    const auto low = boundary_internal::TankPlaneOffsetSign(world_coordinate, {world_origin, index, cell_width});
    const auto high = boundary_internal::TankPlaneOffsetSign(world_coordinate, {world_origin, index + 1, cell_width});
    if (!low.ok() || !high.ok()) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    if (*low >= 0 && *high < 0) {
      const double bound = upper && *low == 0 ? index - 1 : index;
      // Center differences need the signed one-cell halo at both ends.
      if (bound < static_cast<double>(INT32_MIN) + 1 || bound > static_cast<double>(INT32_MAX) - 1) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      return static_cast<std::int32_t>(bound);
    }
  }
  return Error(absl::StatusCode::kOutOfRange);
}

absl::Status PrepareTankCandidates(
    const LiquidHorizontalPlaneSurface3D& plane,
    const AxisAlignedLiquidTank3D& tank,
    UniformMacGeometry3D geometry,
    LiquidRecords3D<CellIndex3D>& candidates
) {
  const std::array low{tank.start_low_x, tank.floor_y, tank.low_z};
  const std::array high{tank.high_x, plane.world_surface_height_y, tank.high_z};
  const std::array origin{geometry.world_origin.x, geometry.world_origin.y, geometry.world_origin.z};
  std::array<std::int32_t, 3> lower{}, upper{};
  std::size_t box_count = 1;
  for (std::size_t axis = 0; axis < low.size(); ++axis) {
    const auto first = TankCandidateBound(low[axis], origin[axis], geometry.cell_width, false);
    if (!first.ok()) {
      return first.status();
    }
    const auto last = TankCandidateBound(high[axis], origin[axis], geometry.cell_width, true);
    if (!last.ok()) {
      return last.status();
    }
    if (*last < *first) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    lower[axis] = *first;
    upper[axis] = *last;
    const auto extent = static_cast<std::size_t>(static_cast<std::int64_t>(*last) - *first + 1);
    if (extent > candidates.capacity() / box_count) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    box_count *= extent;
  }
  STEP_TRY(candidates.Resize(box_count));
  // Topology activates each candidate's six face owners. The common sampling
  // loop below supplies the 27 center distances and eight corners for each.
  std::size_t ordinal = 0;
  for (std::int32_t z = lower[2]; z <= upper[2]; ++z) {
    for (std::int32_t y = lower[1]; y <= upper[1]; ++y) {
      for (std::int32_t x = lower[0]; x <= upper[0]; ++x) {
        candidates[ordinal++] = {x, y, z};
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status PrepareField(
    const LiquidSceneField3D& field,
    const LiquidObstacle3D& obstacle,
    const LiquidState3D& trial,
    Staging& staging,
    const LiquidSubstepBackend3D& backend,
    bool sample_cavity_interior
) {
  STEP_TRY(staging.candidates.Resize(0));
  STEP_TRY(staging.centers.Resize(0));
  STEP_TRY(staging.corners.Resize(0));
  const auto geometry = trial.geometry;
  staging.isolated_drop.reset();
  std::optional<PreparedIsolatedDrop3D> isolated_drop;
  if (const auto* drop = std::get_if<LiquidIsolatedCovarianceDrop3D>(&field)) {
    const auto fit = PrepareIsolatedDrop3D(
        trial.particles.values(),
        geometry.cell_width,
        drop->physical_radius,
        drop->seed_covariance_scale
    );
    if (!fit.ok()) {
      return fit.status();
    }
    isolated_drop = *fit;
  }
  const auto add_sphere = [&](Vec3 center, double radius, bool require_coverage = false) -> absl::Status {
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
          const CellIndex3D index{
              static_cast<std::int32_t>(x),
              static_cast<std::int32_t>(y),
              static_cast<std::int32_t>(z)
          };
          const auto key = [](const CellIndex3D& cell) {
            return Key(cell);
          };
          if (require_coverage) {
            if (!std::ranges::binary_search(staging.candidates.values(), Key(index), {}, key)) {
              return Error(absl::StatusCode::kFailedPrecondition);
            }
          } else {
            STEP_TRY(InsertUnique(staging.candidates, index, key));
          }
        }
      }
    }
    return absl::OkStatus();
  };
  const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&field);
  if (const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&field)) {
    STEP_TRY(add_sphere(sphere->world_center, sphere->radius));
  } else if (plane) {
    STEP_TRY(PrepareTankCandidates(*plane, std::get<AxisAlignedLiquidTank3D>(obstacle), geometry, staging.candidates));
  } else {
    const double support_radius =
        isolated_drop ? 3 * geometry.cell_width : std::get<LiquidParticleSphereUnion3D>(field).radius;
    for (const auto& particle : trial.particles.values()) {
      STEP_TRY(add_sphere(particle.world_position, support_radius));
    }
    if (isolated_drop) {
      // Retain the seed's particle activation and certify its fitted interface coverage.
      STEP_TRY(add_sphere(isolated_drop->world_center(), isolated_drop->maximum_semiaxis(), true));
    }
  }
  // Gas-only interior cells can lie outside all liquid support boxes.
  if (sample_cavity_interior && staging.candidates.size() != 0 && !plane) {
    auto lower = staging.candidates[0];
    auto upper = lower;
    for (const auto& cell : staging.candidates.values()) {
      lower.x = std::min(lower.x, cell.x);
      lower.y = std::min(lower.y, cell.y);
      lower.z = std::min(lower.z, cell.z);
      upper.x = std::max(upper.x, cell.x);
      upper.y = std::max(upper.y, cell.y);
      upper.z = std::max(upper.z, cell.z);
    }
    std::size_t box_count = 1;
    const std::array extents{
        static_cast<std::int64_t>(upper.x) - lower.x + 1,
        static_cast<std::int64_t>(upper.y) - lower.y + 1,
        static_cast<std::int64_t>(upper.z) - lower.z + 1
    };
    for (const auto extent : extents) {
      if (static_cast<std::uint64_t>(extent) > staging.candidates.capacity() / box_count) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      box_count *= static_cast<std::size_t>(extent);
    }
    STEP_TRY(staging.candidates.Resize(box_count));
    std::size_t index = 0;
    for (std::int64_t z = lower.z; z <= upper.z; ++z) {
      for (std::int64_t y = lower.y; y <= upper.y; ++y) {
        for (std::int64_t x = lower.x; x <= upper.x; ++x) {
          staging.candidates[index++] = {
              static_cast<std::int32_t>(x),
              static_cast<std::int32_t>(y),
              static_cast<std::int32_t>(z),
          };
        }
      }
    }
  }
  // These reservations already cover the full undeduplicated stencil. Keep
  // signed keys intact until both families are sorted and deduplicated.
  const std::size_t candidate_count = staging.candidates.size();
  STEP_TRY(staging.centers.Resize(27 * candidate_count));
  STEP_TRY(staging.corners.Resize(8 * candidate_count));
  std::size_t center_ordinal = 0;
  std::size_t corner_ordinal = 0;
  for (const auto& cell : staging.candidates.values()) {
    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          staging.centers[center_ordinal++].cell_index = {cell.x + x, cell.y + y, cell.z + z};
        }
      }
    }
    for (int z = 0; z <= 1; ++z) {
      for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
          staging.corners[corner_ordinal++].corner_index = {cell.x + x, cell.y + y, cell.z + z};
        }
      }
    }
  }
  const auto unique_samples = [](auto& records, auto key) {
    auto samples = records.values();
    std::ranges::sort(samples, {}, key);
    const auto duplicates = std::ranges::unique(samples, {}, key);
    return records.Resize(samples.size() - static_cast<std::size_t>(std::ranges::distance(duplicates)));
  };
  STEP_TRY(unique_samples(staging.centers, [](const auto& sample) {
    return Key(sample.cell_index);
  }));
  STEP_TRY(unique_samples(staging.corners, [](const auto& sample) {
    return Key(sample.corner_index);
  }));

  const std::size_t center_count = staging.centers.size();
  const std::size_t sample_count = center_count + staging.corners.size();
  if (sample_count == 0) {
    return absl::OkStatus();
  }
  // The CUDA arena reserves at least eight sample slots per candidate capacity.
  const std::size_t batch_capacity =
      std::min(kLiquidFieldBatchCapacity3D, kLiquidFieldSampleSlotsPerCapacity3D * staging.candidates.capacity());
  std::array<Vec3, kLiquidFieldBatchCapacity3D> world_positions;
  std::array<double, kLiquidFieldBatchCapacity3D> lattice_ordinates_y;
  std::array<double, kLiquidFieldBatchCapacity3D> distances;
  const auto lattice_position = [&](std::size_t ordinal) -> Vec3 {
    if (ordinal < center_count) {
      const auto index = staging.centers[ordinal].cell_index;
      return {index.x + 0.5, index.y + 0.5, index.z + 0.5};
    }
    const auto index = staging.corners[ordinal - center_count].corner_index;
    return {static_cast<double>(index.x), static_cast<double>(index.y), static_cast<double>(index.z)};
  };
  // Centers precede corners, with the same (z, y, x) order within each family
  // as the old staged fields. Do not round compensated plane y to world space.
  for (std::size_t first = 0; first < sample_count; first += batch_capacity) {
    const std::size_t count = std::min(batch_capacity, sample_count - first);
    const auto output = std::span(distances).first(count);
    if (plane) {
      for (std::size_t index = 0; index < count; ++index) {
        lattice_ordinates_y[index] = lattice_position(first + index).y;
      }
      STEP_TRY(backend.sample_lattice_plane(
          backend.context,
          *plane,
          geometry,
          std::span(lattice_ordinates_y).first(count),
          output
      ));
    } else {
      for (std::size_t index = 0; index < count; ++index) {
        const auto position = lattice_position(first + index);
        world_positions[index] = {
            geometry.world_origin.x + position.x * geometry.cell_width,
            geometry.world_origin.y + position.y * geometry.cell_width,
            geometry.world_origin.z + position.z * geometry.cell_width,
        };
      }
      if (isolated_drop) {
        for (std::size_t index = 0; index < count; ++index) {
          const auto sample = isolated_drop->QuadraticGridLevelSet(world_positions[index]);
          if (!sample.ok()) {
            return sample.status();
          }
          output[index] = *sample;
        }
      } else {
        STEP_TRY(backend.sample_field_batch(
            backend.context,
            field,
            trial.particles.values(),
            std::span(world_positions).first(count),
            output
        ));
      }
    }
    for (std::size_t index = 0; index < count; ++index) {
      const std::size_t ordinal = first + index;
      const double distance = output[index];
      if (!std::isfinite(distance)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      if (ordinal < center_count) {
        staging.centers[ordinal].signed_distance = distance;
      } else {
        staging.corners[ordinal - center_count].signed_distance = distance;
      }
    }
  }
  staging.isolated_drop = isolated_drop;
  return absl::OkStatus();
}

absl::Status SetIsolatedDropPressureLifts(
    UniformMacGeometry3D geometry,
    double surface_tension,
    double ambient_pressure,
    Staging& staging,
    LiquidPressureProblemView3D& pressure
) {
  if (!staging.isolated_drop || !staging.isolated_drop_pressure_faces) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  auto& faces = *staging.isolated_drop_pressure_faces;
  STEP_TRY(faces.Resize(pressure.faces.size()));
  std::ranges::copy(pressure.faces, faces.values().begin());
  for (auto& face : faces.values()) {
    auto* link = std::get_if<DirichletMacPressureLink3D>(&face.pressure_link);
    if (!link) {
      continue;
    }
    const auto axis = static_cast<std::size_t>(face.face_index.axis);
    if (axis >= 3) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    const auto index = face.face_index.lattice_index;
    const std::array<std::int64_t, 3> head_index{index.x, index.y, index.z};
    auto tail_index = head_index;
    --tail_index[axis];
    if (tail_index[axis] < std::numeric_limits<std::int32_t>::min()) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    // Match the center samples used to classify the pressure cells. Subtracting
    // h from the rounded head position can change a near-surface tail sample.
    const auto center = [&](const std::array<std::int64_t, 3>& cell) {
      return Vec3{
          geometry.world_origin.x + (static_cast<double>(cell[0]) + 0.5) * geometry.cell_width,
          geometry.world_origin.y + (static_cast<double>(cell[1]) + 0.5) * geometry.cell_width,
          geometry.world_origin.z + (static_cast<double>(cell[2]) + 0.5) * geometry.cell_width,
      };
    };
    const Vec3 head = center(head_index);
    const Vec3 tail = center(tail_index);
    const auto tail_level = staging.isolated_drop->QuadraticGridLevelSet(tail);
    if (!tail_level.ok()) {
      return tail_level.status();
    }
    const auto curvature = *tail_level < 0 ? staging.isolated_drop->AirLinkCurvature(tail, head)
                                           : staging.isolated_drop->AirLinkCurvature(head, tail);
    if (!curvature.ok()) {
      return curvature.status();
    }
    const double capillary_pressure = surface_tension * *curvature;
    const double exterior_pressure = ambient_pressure + capillary_pressure;
    if ((surface_tension > 0 && *curvature != 0 && capillary_pressure == 0)
        || !boundary_internal::IsNormalOrZero(capillary_pressure) || !std::isfinite(exterior_pressure)) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    link->exterior_pressure = exterior_pressure;
  }
  pressure.faces = faces.values();
  return absl::OkStatus();
}

std::optional<CellIndex3D> TankContainingCell(UniformMacGeometry3D geometry, Vec3 world_position) {
  const auto x = TankCandidateBound(world_position.x, geometry.world_origin.x, geometry.cell_width, false);
  const auto y = TankCandidateBound(world_position.y, geometry.world_origin.y, geometry.cell_width, false);
  const auto z = TankCandidateBound(world_position.z, geometry.world_origin.z, geometry.cell_width, false);
  if (!x.ok() || !y.ok() || !z.ok()) {
    return std::nullopt;
  }
  return CellIndex3D{*x, *y, *z};
}

absl::Status PrepareTransfers(const LiquidState3D& trial, SparseMacGridView3D grid, Staging& staging) {
  STEP_TRY(staging.particles.Resize(trial.particles.size()));
  STEP_TRY(staging.gathered.Resize(trial.particles.size()));
  STEP_TRY(staging.particle_distances.Resize(trial.particles.size()));
  if (staging.isolated_drop_restricted_gathers) {
    STEP_TRY(staging.isolated_drop_restricted_gathers->Resize(trial.particles.size()));
  }
  if (staging.regular_transport) {
    STEP_TRY(staging.regular_transport->Resize(trial.particles.size()));
  }
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
  if (staging.projected_gather_validity) {
    STEP_TRY(staging.projected_gather_validity->Resize(count));
  }
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
    if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
      const auto wall = boundary_internal::SampleTankFace(
          *tank,
          grid.geometry(),
          boundary_internal::TankFaceAnchorInCellWidths(*face)
      );
      if (!wall.ok()) {
        return boundary_internal::ArithmeticStatus(wall.status(), true);
      }
      tags[index] = {0, wall->inward_distance > 0};
    } else {
      tags[index] = {side, SolidDistance(obstacle, *anchor) >= 0};
    }
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
    if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
      const auto wall = boundary_internal::SampleTankFace(
          *tank,
          grid.geometry(),
          boundary_internal::TankFaceAnchorInCellWidths(*face)
      );
      if (!wall.ok()) {
        return boundary_internal::ArithmeticStatus(wall.status(), true);
      }
      if (std::abs(wall->inward_distance) <= 2 * grid.geometry().cell_width) {
        walls[index] = {true, 0, wall->inward_normal, wall->wall_velocity};
      }
      continue;
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
