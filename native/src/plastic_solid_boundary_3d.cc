#include "fluidity/host/plastic_solid_boundary_3d.h"

#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

#include "absl/status/status.h"

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
  }
  return PlasticSolidBoundarySamples3D{.corners = std::move(*corners), .particle_ids = std::move(*owners)};
}

}  // namespace fluidity
