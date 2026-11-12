#include "fluidity/host/plastic_solid_boundary_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "fluidity/base/sampled_solid_liquid_geometry_3d.h"
#include "liquid_boundary_backend.h"
#include "mac_pressure_3d_system.h"

namespace fluidity {
namespace {

bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double VelocityComponent(Vec3 slope, Vec3 offset, double velocity) {
  return std::fma(slope.x, offset.x, std::fma(slope.y, offset.y, std::fma(slope.z, offset.z, velocity)));
}

auto CornerKey(CellCornerIndex3D index) {
  return std::tuple{index.z, index.y, index.x};
}

}  // namespace

SampledSolidBoundary3D PlasticSolidBoundarySamples3D::View() const {
  return {.corner_distances = corners.values(), .face_normal_velocities = {}};
}

absl::StatusOr<PlasticSolidBoundarySamples3D> SamplePlasticSolidBoundary3D(
    SparseMacGridView3D grid,
    std::span<const LiquidCornerDistance3D> liquid_corner_keys,
    std::span<const PlasticSolidCheckpointParticle3D> accepted_particles,
    std::uint64_t body_id,
    double support_radius
) {
  if (body_id == 0 || !std::isfinite(support_radius) || support_radius <= 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  std::uint64_t previous_particle_id = 0;
  bool has_body = false;
  for (const auto& particle : accepted_particles) {
    if (particle.particle_id <= previous_particle_id) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    previous_particle_id = particle.particle_id;
    if (particle.body_id != body_id) {
      continue;
    }
    const auto& transfer = particle.transfer;
    const auto& slopes = transfer.particle_velocity_slopes;
    if (!Finite(transfer.sorted_world_position) || !Finite(transfer.particle_velocity) || !Finite(slopes.x_row)
        || !Finite(slopes.y_row) || !Finite(slopes.z_row)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    has_body = true;
  }
  if (!has_body) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }

  auto corners = LiquidRecords3D<SolidCornerDistance3D>::Create(liquid_corner_keys.size());
  if (!corners.ok()) {
    return corners.status();
  }
  auto owners = LiquidRecords3D<std::uint64_t>::Create(liquid_corner_keys.size());
  if (!owners.ok()) {
    return owners.status();
  }
  auto offsets = LiquidRecords3D<Vec3>::Create(liquid_corner_keys.size());
  if (!offsets.ok()) {
    return offsets.status();
  }
  const auto offset_extent = offsets->Resize(liquid_corner_keys.size());
  if (!offset_extent.ok()) {
    return offset_extent;
  }
  const auto corner_extent = corners->Resize(liquid_corner_keys.size());
  if (!corner_extent.ok()) {
    return corner_extent;
  }
  const auto owner_extent = owners->Resize(liquid_corner_keys.size());
  if (!owner_extent.ok()) {
    return owner_extent;
  }

  const auto geometry = grid.geometry();
  const auto origin = geometry.world_origin;
  for (std::size_t ordinal = 0; ordinal < liquid_corner_keys.size(); ++ordinal) {
    const auto index = liquid_corner_keys[ordinal].corner_index;
    if (ordinal > 0 && CornerKey(index) <= CornerKey(liquid_corner_keys[ordinal - 1].corner_index)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const Vec3 world_position{
        .x = std::fma(geometry.cell_width, static_cast<double>(index.x), origin.x),
        .y = std::fma(geometry.cell_width, static_cast<double>(index.y), origin.y),
        .z = std::fma(geometry.cell_width, static_cast<double>(index.z), origin.z),
    };
    if (!Finite(world_position)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    double nearest_distance = std::numeric_limits<double>::infinity();
    const PlasticSolidCheckpointParticle3D* nearest_particle = nullptr;
    Vec3 nearest_offset{};
    for (const auto& particle : accepted_particles) {
      if (particle.body_id != body_id) {
        continue;
      }
      const auto center = particle.transfer.sorted_world_position;
      const Vec3 offset{
          .x = world_position.x - center.x,
          .y = world_position.y - center.y,
          .z = world_position.z - center.z,
      };
      const double distance = std::hypot(offset.x, offset.y, offset.z);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest_particle = &particle;
        nearest_offset = offset;
      }
    }
    if (nearest_particle == nullptr) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    const auto& transfer = nearest_particle->transfer;
    const auto& slopes = transfer.particle_velocity_slopes;
    const Vec3 wall_velocity{
        .x = VelocityComponent(slopes.x_row, nearest_offset, transfer.particle_velocity.x),
        .y = VelocityComponent(slopes.y_row, nearest_offset, transfer.particle_velocity.y),
        .z = VelocityComponent(slopes.z_row, nearest_offset, transfer.particle_velocity.z),
    };
    if (!Finite(wall_velocity)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    corners->values()[ordinal] = {index, nearest_distance - support_radius, wall_velocity};
    owners->values()[ordinal] = nearest_particle->particle_id;
    offsets->values()[ordinal] = nearest_offset;
  }
  return PlasticSolidBoundarySamples3D{
      .corners = std::move(*corners),
      .particle_ids = std::move(*owners),
      .particle_offsets = std::move(*offsets)
  };
}

absl::StatusOr<std::array<PlasticSolidPressureImpulse3D, 4>> StagePlasticSolidPressureFace3D(
    SparseMacGridView3D grid,
    const PlasticSolidBoundarySamples3D& samples,
    const FrozenLiquidFace3D& face,
    double tail_pressure,
    double head_pressure,
    double step_duration,
    const LiquidObstacle3D& obstacle
) {
  if (!std::isfinite(step_duration) || step_duration <= 0 || !std::isfinite(tail_pressure)
      || !std::isfinite(head_pressure) || samples.corners.size() != samples.particle_ids.size()
      || samples.corners.size() != samples.particle_offsets.size()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto slot = grid.FaceSlot(face.face_index);
  if (!slot.ok() || *slot != face.face_slot || !face.prescribed_wall_flux) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }

  const auto corners = samples.corners.values();
  const auto lattice = face.face_index.lattice_index;
  const int normal_axis = static_cast<int>(face.face_index.axis);
  std::array<int, 2> transverse_axes{};
  int transverse_count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis != normal_axis) {
      transverse_axes[transverse_count++] = axis;
    }
  }
  std::array<std::size_t, 4> ordinals{};
  std::array<double, 4> solid_distance{};
  std::array<double, 4> normal_velocity{};
  for (int corner = 0; corner < 4; ++corner) {
    std::array<std::int32_t, 3> key{lattice.x, lattice.y, lattice.z};
    for (int bit = 0; bit < 2; ++bit) {
      if ((corner & (1 << bit)) != 0) {
        auto& ordinate = key[transverse_axes[bit]];
        if (ordinate == std::numeric_limits<std::int32_t>::max()) {
          return absl::Status(absl::StatusCode::kOutOfRange, {});
        }
        ++ordinate;
      }
    }
    const auto corner_key = CornerKey({key[0], key[1], key[2]});
    const auto found = std::ranges::lower_bound(corners, corner_key, {}, [](const SolidCornerDistance3D& sample) {
      return CornerKey(sample.corner_index);
    });
    if (found == corners.end() || CornerKey(found->corner_index) != corner_key) {
      return absl::Status(absl::StatusCode::kFailedPrecondition, {});
    }
    ordinals[corner] = static_cast<std::size_t>(found - corners.begin());
    solid_distance[corner] = found->signed_distance;
    normal_velocity[corner] = boundary_internal::CartesianComponents(found->wall_velocity)[normal_axis];
  }

  const double width = grid.geometry().cell_width;
  std::optional<boundary_internal::SampledSolidAnalyticFace> analytic;
  if (!std::holds_alternative<std::monostate>(obstacle)) {
    if (!std::holds_alternative<TranslatingHalfspace3D>(obstacle)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const boundary_internal::GeometryInput query{
        .mode = boundary_internal::GeometryMode::kSampledSolidFace,
        .geometry = grid.geometry(),
        .index = {lattice.x, lattice.y, lattice.z},
        .obstacle = boundary_internal::PackObstacle(obstacle),
        .axis = normal_axis,
    };
    const auto local = boundary_internal::LocalObstacle(query);
    if (!local.ok()) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    const auto measured = boundary_internal::MeasureSampledSolidAnalyticFace(query, *local);
    if (!measured.ok()) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    analytic = *measured;
  }
  const auto weights = boundary_internal::MeasureSampledSolidFaceWeights(
      width,
      solid_distance,
      analytic ? &analytic->fluid_side : nullptr
  );
  if (!weights.ok()) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  double blocked_area = analytic ? analytic->blocked_area : 0;
  double wall_flux = analytic ? analytic->wall_flux : 0;
  double flux_scale = std::abs(wall_flux);
  for (int corner = 0; corner < 4; ++corner) {
    const double contribution = (*weights)[corner] * normal_velocity[corner];
    blocked_area += (*weights)[corner];
    wall_flux += contribution;
    flux_scale += std::abs(contribution);
  }
  constexpr double kRoundoffSlack = 64 * std::numeric_limits<double>::epsilon();
  if (!std::isfinite(flux_scale) || !std::isfinite(face.blocked_area) || !std::isfinite(*face.prescribed_wall_flux)
      || std::abs(face.blocked_area - blocked_area) > kRoundoffSlack * width * width
      || std::abs(*face.prescribed_wall_flux - wall_flux) > kRoundoffSlack * flux_scale) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  const auto forces = boundary_internal::ComputeSampledSolidFacePressureForces(*weights, tail_pressure, head_pressure);
  if (!forces.ok()) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }

  std::array<PlasticSolidPressureImpulse3D, 4> loads{};
  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  for (int corner = 0; corner < 4; ++corner) {
    const auto ordinal = ordinals[corner];
    const double impulse = (*forces)[corner] * step_duration;
    if (!std::isfinite(impulse)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    loads[corner].particle_id = samples.particle_ids.values()[ordinal];
    loads[corner].world_offset = samples.particle_offsets.values()[ordinal];
    loads[corner].impulse.*kComponents[normal_axis] = impulse;
  }
  return loads;
}

absl::StatusOr<std::vector<DrySolidParticle3D>> StagePressureLoadedSnowParticles3D(
    std::span<const PlasticSolidCheckpointParticle3D> accepted,
    std::uint64_t body_id,
    std::uint64_t material_id,
    const SnowParameters3D& snow,
    double cell_width,
    std::span<const PlasticSolidPressureImpulse3D> loads
) {
  if (!std::isfinite(cell_width) || cell_width <= 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  auto trial = RestoreDrySolidCheckpointParticles3D(accepted, body_id, material_id, snow);
  if (!trial.ok()) {
    return trial.status();
  }

  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  constexpr std::array kSlopeRows{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};
  for (const auto& load : loads) {
    const auto owner =
        std::ranges::lower_bound(accepted, load.particle_id, {}, &PlasticSolidCheckpointParticle3D::particle_id);
    if (owner == accepted.end() || owner->particle_id != load.particle_id || !Finite(load.impulse)
        || !Finite(load.world_offset)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    auto& particle = (*trial)[static_cast<std::size_t>(owner - accepted.begin())];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto component = kComponents[axis];
      const double velocity_increment = (load.impulse.*component) / particle.mass;
      particle.world_velocity.*component += velocity_increment;
      auto& slope = particle.velocity_slopes.*kSlopeRows[axis];
      for (const auto coordinate : kComponents) {
        // The complete quadratic moment is I/4 in cell coordinates.
        const double normalized_offset = (load.world_offset.*coordinate) / cell_width;
        const double slope_increment = (4.0 * (velocity_increment * normalized_offset)) / cell_width;
        slope.*coordinate += slope_increment;
      }
    }
    const auto& slopes = particle.velocity_slopes;
    if (!Finite(particle.world_velocity) || !Finite(slopes.x_row) || !Finite(slopes.y_row) || !Finite(slopes.z_row)) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
  }
  return std::move(*trial);
}

absl::StatusOr<LiquidRecords3D<PlasticSolidPressureImpulse3D>> StagePlasticSolidPressureLoads3D(
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const PlasticSolidBoundarySamples3D& samples,
    std::span<const double> solved_cell_pressures,
    double exterior_pressure,
    const LiquidObstacle3D& obstacle
) {
  if (solved_cell_pressures.size() != geometry.cells.size() || !std::isfinite(exterior_pressure)
      || !std::isfinite(geometry.step_duration) || geometry.step_duration <= 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto cell_key = [](const FrozenLiquidCell3D& cell) {
    return CornerKey({cell.cell_index.x, cell.cell_index.y, cell.cell_index.z});
  };
  for (std::size_t i = 0; i < geometry.cells.size(); ++i) {
    if (!std::isfinite(solved_cell_pressures[i])
        || (i > 0 && cell_key(geometry.cells[i]) <= cell_key(geometry.cells[i - 1]))) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  std::size_t blocked_faces = 0;
  for (std::size_t i = 0; i < geometry.faces.size(); ++i) {
    const auto& face = geometry.faces[i];
    if (!std::isfinite(face.blocked_area) || face.blocked_area < 0
        || (i > 0 && face.face_slot <= geometry.faces[i - 1].face_slot)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    blocked_faces += face.blocked_area > 0;
  }
  if (blocked_faces > std::numeric_limits<std::size_t>::max() / 4) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  auto loads = LiquidRecords3D<PlasticSolidPressureImpulse3D>::Create(4 * blocked_faces);
  if (!loads.ok()) {
    return loads.status();
  }
  const auto extent = loads->Resize(4 * blocked_faces);
  if (!extent.ok()) {
    return extent;
  }
  const auto pressure_at = [&](const std::array<std::int32_t, 3>& cell) -> std::optional<double> {
    const auto key = CornerKey({cell[0], cell[1], cell[2]});
    const auto found = std::ranges::lower_bound(geometry.cells, key, {}, cell_key);
    if (found == geometry.cells.end() || cell_key(*found) != key) {
      return std::nullopt;
    }
    return solved_cell_pressures[static_cast<std::size_t>(found - geometry.cells.begin())];
  };
  std::size_t destination = 0;
  for (const auto& face : geometry.faces) {
    if (face.blocked_area == 0) {
      continue;
    }
    const auto axis = static_cast<std::size_t>(face.face_index.axis);
    if (axis >= 3) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const auto lattice = face.face_index.lattice_index;
    std::array<std::int32_t, 3> endpoint{lattice.x, lattice.y, lattice.z};
    const auto head = pressure_at(endpoint);
    std::optional<double> tail;
    if (endpoint[axis] != std::numeric_limits<std::int32_t>::min()) {
      --endpoint[axis];
      tail = pressure_at(endpoint);
    }
    if (!head && !tail) {
      return absl::Status(absl::StatusCode::kFailedPrecondition, {});
    }
    const auto face_loads = StagePlasticSolidPressureFace3D(
        geometry.mac_grid,
        samples,
        face,
        tail.value_or(exterior_pressure),
        head.value_or(exterior_pressure),
        geometry.step_duration,
        obstacle
    );
    if (!face_loads.ok()) {
      return face_loads.status();
    }
    for (const auto& load : *face_loads) {
      loads->values()[destination++] = load;
    }
  }
  return std::move(*loads);
}

absl::StatusOr<PlasticSolidPressureMobility3D> PlasticSolidPressureMobility3D::Create(
    const FrozenLiquidBoundaryGeometryView3D& geometry,
    const PlasticSolidBoundarySamples3D& samples,
    std::span<const PlasticSolidCheckpointParticle3D> accepted,
    const LiquidObstacle3D& obstacle
) {
  const double duration = geometry.step_duration;
  const double width = geometry.mac_grid.geometry().cell_width;
  if (accepted.empty() || !std::isfinite(duration) || duration <= 0) {
    return absl::InvalidArgumentError("invalid snow pressure mobility inputs");
  }
  std::uint64_t previous_id = 0;
  std::vector<double> inverse_sqrt_masses;
  inverse_sqrt_masses.reserve(accepted.size());
  for (const auto& particle : accepted) {
    const double mass = particle.transfer.particle_mass;
    if (particle.particle_id <= previous_id || !std::isfinite(mass) || mass <= 0) {
      return absl::InvalidArgumentError("invalid snow pressure mobility particle");
    }
    previous_id = particle.particle_id;
    inverse_sqrt_masses.push_back(std::sqrt(duration) / std::sqrt(mass));
  }
  const auto cell_key = [](const FrozenLiquidCell3D& cell) {
    return CornerKey({cell.cell_index.x, cell.cell_index.y, cell.cell_index.z});
  };
  for (std::size_t index = 1; index < geometry.cells.size(); ++index) {
    if (cell_key(geometry.cells[index - 1]) >= cell_key(geometry.cells[index])) {
      return absl::InvalidArgumentError("unordered snow pressure mobility cells");
    }
  }
  const auto cell_at = [&](std::array<std::int32_t, 3> index) {
    const auto key = CornerKey({index[0], index[1], index[2]});
    const auto cell = std::ranges::lower_bound(geometry.cells, key, {}, cell_key);
    return cell != geometry.cells.end() && cell_key(*cell) == key
        ? static_cast<std::size_t>(cell - geometry.cells.begin())
        : geometry.cells.size();
  };

  PlasticSolidPressureMobility3D mobility;
  mobility.particle_count_ = accepted.size();
  mobility.cell_count_ = geometry.cells.size();
  mobility.faces_.reserve(geometry.faces.size());
  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  for (const auto& face : geometry.faces) {
    const auto axis = static_cast<std::size_t>(face.face_index.axis);
    if (axis >= kComponents.size() || !std::isfinite(face.blocked_area) || face.blocked_area < 0) {
      return absl::InvalidArgumentError("invalid snow pressure mobility face");
    }
    const auto lattice = face.face_index.lattice_index;
    std::array<std::int32_t, 3> endpoint{lattice.x, lattice.y, lattice.z};
    Face row{
        .axis = axis,
        .tail_cell = mobility.cell_count_,
        .head_cell = cell_at(endpoint),
    };
    if (endpoint[axis] != std::numeric_limits<std::int32_t>::min()) {
      --endpoint[axis];
      row.tail_cell = cell_at(endpoint);
    }
    if (row.head_cell == mobility.cell_count_ && row.tail_cell == mobility.cell_count_) {
      return absl::InvalidArgumentError("snow pressure mobility face has no cell");
    }
    if (face.blocked_area > 0) {
      // At 1 Pa and 1 s the numerical impulses are the signed area coefficients.
      const auto unit = StagePlasticSolidPressureFace3D(geometry.mac_grid, samples, face, 1, 0, 1, obstacle);
      if (!unit.ok()) {
        return unit.status();
      }
      for (std::size_t index = 0; index < unit->size(); ++index) {
        const auto& corner = (*unit)[index];
        const double area = corner.impulse.*kComponents[axis];
        if (area == 0) {
          continue;
        }
        const auto owner =
            std::ranges::lower_bound(accepted, corner.particle_id, {}, &PlasticSolidCheckpointParticle3D::particle_id);
        if (owner == accepted.end() || owner->particle_id != corner.particle_id) {
          return absl::InvalidArgumentError("unknown snow pressure mobility owner");
        }
        const auto owner_index = static_cast<std::size_t>(owner - accepted.begin());
        const double scale = area * inverse_sqrt_masses[owner_index];
        const auto offset = corner.world_offset;
        auto& term = row.corners[index];
        term.owner = owner_index;
        term.coefficients = {
            scale,
            scale * (2 * (offset.x / width)),
            scale * (2 * (offset.y / width)),
            scale * (2 * (offset.z / width)),
        };
        if (!std::ranges::all_of(term.coefficients, [](double value) {
              return std::isfinite(value);
            })) {
          return absl::OutOfRangeError("nonfinite snow pressure mobility coefficient");
        }
      }
    }
    mobility.faces_.push_back(row);
  }
  return mobility;
}

absl::StatusOr<std::vector<double>> PlasticSolidPressureMobility3D::Apply(std::span<const double> pressure_increment
) const {
  if (pressure_increment.size() != cell_count_ || !std::ranges::all_of(pressure_increment, [](double value) {
        return std::isfinite(value);
      })) {
    return absl::InvalidArgumentError("invalid snow pressure mobility input");
  }
  const auto pressure_at = [&](std::size_t index) {
    return index == cell_count_ ? 0.0 : pressure_increment[index];
  };
  using NormalModes = std::array<double, 4>;
  std::vector<std::array<NormalModes, 3>> particle_response(particle_count_);
  for (const auto& face : faces_) {
    const double pressure_difference = pressure_at(face.tail_cell) - pressure_at(face.head_cell);
    for (const auto& corner : face.corners) {
      auto& response = particle_response[corner.owner][face.axis];
      for (std::size_t mode = 0; mode < response.size(); ++mode) {
        response[mode] = std::fma(corner.coefficients[mode], pressure_difference, response[mode]);
      }
    }
  }

  std::vector<double> flux;
  flux.reserve(faces_.size());
  for (const auto& face : faces_) {
    double face_flux = 0;
    for (const auto& corner : face.corners) {
      const auto& response = particle_response[corner.owner][face.axis];
      for (std::size_t mode = 0; mode < response.size(); ++mode) {
        face_flux = std::fma(corner.coefficients[mode], response[mode], face_flux);
      }
    }
    if (!std::isfinite(face_flux)) {
      return absl::OutOfRangeError("nonfinite snow pressure mobility flux");
    }
    flux.push_back(face_flux);
  }
  return flux;
}

absl::StatusOr<std::vector<double>> PlasticSolidPressureMobility3D::SolveWallFlux(
    const MacPressureInputs3D& liquid,
    double ambient_pressure,
    std::size_t maximum_iterations
) const {
  if (liquid.pressure_cells.size() != cell_count_ || liquid.pressure_faces.size() != faces_.size()
      || !std::isfinite(ambient_pressure) || maximum_iterations == 0) {
    return absl::InvalidArgumentError("invalid coupled snow pressure system");
  }
  const auto system = pressure_internal::PrepareMacPressureSystem3D(liquid);
  if (!system.ok()) {
    return system.status();
  }
  const auto links = system->links();
  const auto cells = system->cells();
  std::vector<bool> pinned(cell_count_);
  for (const auto cell : system->gauge_cell_ordinals()) {
    pinned[cell] = true;
  }
  std::vector<double> diagonal(cell_count_);
  for (std::size_t cell = 0; cell < cell_count_; ++cell) {
    diagonal[cell] = cells[cell].compliance;
  }
  for (std::size_t index = 0; index < links.size(); ++index) {
    const auto& link = links[index];
    const auto& face = faces_[index];
    if (link.tail_cell_ordinal.value_or(cell_count_) != face.tail_cell
        || link.head_cell_ordinal.value_or(cell_count_) != face.head_cell) {
      return absl::InvalidArgumentError("snow and liquid pressure face incidence differs");
    }
    if (face.tail_cell < cell_count_) {
      diagonal[face.tail_cell] += link.conductance;
    }
    if (face.head_cell < cell_count_) {
      diagonal[face.head_cell] += link.conductance;
    }
  }
  for (std::size_t cell = 0; cell < cell_count_; ++cell) {
    if (!pinned[cell] && (!std::isfinite(diagonal[cell]) || diagonal[cell] <= 0)) {
      return absl::FailedPreconditionError("invalid coupled snow pressure diagonal");
    }
  }

  // Shift the snow pressure by the exterior gauge. The same outward
  // incidence maps both the liquid conductance and the snow flux to cells.
  const auto apply = [&](std::span<const double> pressure, std::span<double> output) -> absl::Status {
    const auto snow_flux = Apply(pressure);
    if (!snow_flux.ok()) {
      return snow_flux.status();
    }
    for (std::size_t cell = 0; cell < cell_count_; ++cell) {
      output[cell] = cells[cell].compliance * pressure[cell];
    }
    for (std::size_t index = 0; index < links.size(); ++index) {
      const auto& face = faces_[index];
      const double tail = face.tail_cell < cell_count_ ? pressure[face.tail_cell] : 0;
      const double head = face.head_cell < cell_count_ ? pressure[face.head_cell] : 0;
      const double flux = std::fma(links[index].conductance, tail - head, (*snow_flux)[index]);
      if (face.tail_cell < cell_count_) {
        output[face.tail_cell] += flux;
      }
      if (face.head_cell < cell_count_) {
        output[face.head_cell] -= flux;
      }
    }
    return absl::OkStatus();
  };

  // p starts at zero in the liquid gauge. Moving the ambient contribution
  // to the right leaves pinned increments at zero throughout the solve.
  std::vector<double> ambient(cell_count_, ambient_pressure);
  const auto ambient_flux = Apply(ambient);
  if (!ambient_flux.ok()) {
    return ambient_flux.status();
  }
  const auto integrated_rhs = system->integrated_rhs();
  std::vector<double> rhs(integrated_rhs.begin(), integrated_rhs.end());
  for (std::size_t index = 0; index < faces_.size(); ++index) {
    const auto& face = faces_[index];
    if (face.tail_cell < cell_count_) {
      rhs[face.tail_cell] += (*ambient_flux)[index];
    }
    if (face.head_cell < cell_count_) {
      rhs[face.head_cell] -= (*ambient_flux)[index];
    }
  }
  std::vector<double> pressure(cell_count_);
  std::vector<double> residual(cell_count_);
  std::vector<double> scaled_residual(cell_count_);
  std::vector<double> direction(cell_count_);
  std::vector<double> action(cell_count_);
  for (std::size_t cell = 0; cell < cell_count_; ++cell) {
    if (pinned[cell]) {
      continue;
    }
    residual[cell] = rhs[cell];
    scaled_residual[cell] = residual[cell] / diagonal[cell];
    direction[cell] = scaled_residual[cell];
  }
  double product = std::inner_product(residual.begin(), residual.end(), scaled_residual.begin(), 0.0);
  for (std::size_t iteration = 0; iteration <= maximum_iterations; ++iteration) {
    double maximum_defect = 0;
    for (std::size_t cell = 0; cell < cell_count_; ++cell) {
      maximum_defect = std::max(maximum_defect, std::abs(residual[cell]) / cells[cell].fluid_volume);
    }
    if (maximum_defect <= 1e-8) {
      const auto verified = apply(pressure, action);
      if (!verified.ok()) {
        return verified;
      }
      for (std::size_t cell = 0; cell < cell_count_; ++cell) {
        const double defect = std::abs(rhs[cell] - action[cell]) / cells[cell].fluid_volume;
        if (!std::isfinite(defect) || defect > 1e-7) {
          return absl::AbortedError("coupled snow pressure residual failed verification");
        }
        pressure[cell] -= ambient_pressure;
      }
      return Apply(pressure);
    }
    if (iteration == maximum_iterations) {
      break;
    }
    const auto applied = apply(direction, action);
    if (!applied.ok()) {
      return applied;
    }
    const double curvature = std::inner_product(direction.begin(), direction.end(), action.begin(), 0.0);
    if (!std::isfinite(product) || product <= 0 || !std::isfinite(curvature) || curvature <= 0) {
      return absl::AbortedError("coupled snow pressure lost positive curvature");
    }
    const double alpha = product / curvature;
    for (std::size_t cell = 0; cell < cell_count_; ++cell) {
      if (!pinned[cell]) {
        pressure[cell] = std::fma(alpha, direction[cell], pressure[cell]);
        residual[cell] = std::fma(-alpha, action[cell], residual[cell]);
        scaled_residual[cell] = residual[cell] / diagonal[cell];
      }
    }
    const double next_product = std::inner_product(residual.begin(), residual.end(), scaled_residual.begin(), 0.0);
    const double beta = next_product / product;
    for (std::size_t cell = 0; cell < cell_count_; ++cell) {
      direction[cell] = std::fma(beta, direction[cell], scaled_residual[cell]);
    }
    product = next_product;
  }
  return absl::AbortedError("coupled snow pressure did not converge");
}

absl::Status PlasticSolidPressureMobility3D::VerifyProjectedFlux(
    std::span<const double> projected_pressure,
    double ambient_pressure,
    std::span<const double> imposed_flux,
    std::span<const FrozenLiquidCell3D> cells
) const {
  if (cells.size() != cell_count_ || imposed_flux.size() != faces_.size()) {
    return absl::InvalidArgumentError("invalid coupled snow pressure verification extents");
  }
  std::vector<double> relative_pressure;
  relative_pressure.reserve(projected_pressure.size());
  for (const double pressure : projected_pressure) {
    relative_pressure.push_back(pressure - ambient_pressure);
  }
  const auto actual_flux = Apply(relative_pressure);
  if (!actual_flux.ok()) {
    return actual_flux.status();
  }
  std::vector<double> defect(cell_count_);
  for (std::size_t index = 0; index < faces_.size(); ++index) {
    const auto& face = faces_[index];
    const double flux_error = (*actual_flux)[index] - imposed_flux[index];
    if (face.tail_cell < cell_count_) {
      defect[face.tail_cell] += flux_error;
    }
    if (face.head_cell < cell_count_) {
      defect[face.head_cell] -= flux_error;
    }
  }
  for (std::size_t cell = 0; cell < cell_count_; ++cell) {
    constexpr double kMaximumSnowConstraintDefect = 2e-5;
    const double scaled_defect = std::abs(defect[cell]) / cells[cell].fluid_volume;
    if (!std::isfinite(scaled_defect) || scaled_defect > kMaximumSnowConstraintDefect) {
      return absl::AbortedError("projected pressure and snow flux disagree");
    }
  }
  return absl::OkStatus();
}

}  // namespace fluidity
