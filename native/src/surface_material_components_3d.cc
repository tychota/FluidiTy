#include "fluidity/host/surface_material_components_3d.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>

namespace fluidity {
namespace {
bool PositiveFinite(double value) {
  return std::isfinite(value) && value > 0;
}

// Find with path halving. Union chooses the lower root so partitioning is
// deterministic even when several overlaps occur at the same position.
std::size_t ComponentRoot(std::span<std::size_t> parent, std::size_t index) {
  while (parent[index] != index) {
    parent[index] = parent[parent[index]];
    index = parent[index];
  }
  return index;
}
}  // namespace

absl::StatusOr<std::vector<SurfaceMaterialPartition3D>> PartitionSurfaceSphereUnion3D(
    std::span<const LiquidParticleState3D> particles,
    double sphere_radius_m,
    std::size_t maximum_particles
) {
  if (!PositiveFinite(sphere_radius_m)) {
    return absl::InvalidArgumentError("invalid material sphere radius");
  }
  if (particles.size() > std::min(maximum_particles, std::size_t{1024})) {
    return absl::ResourceExhaustedError("sphere union partition particle budget exceeded");
  }
  const double sphere_diameter_m = 2 * sphere_radius_m;
  if (!std::isfinite(sphere_diameter_m)) {
    return absl::OutOfRangeError("material sphere diameter overflowed");
  }
  std::vector<double> material_volume_m3;
  material_volume_m3.reserve(particles.size());
  for (const auto& particle : particles) {
    const auto& position = particle.world_position;
    if (particle.id == 0 || !std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)
        || !PositiveFinite(particle.reference_volume) || !PositiveFinite(particle.volume_ratio)) {
      return absl::InvalidArgumentError("invalid material sphere particle");
    }
    if (particle.side_tag != 0) {
      return absl::UnimplementedError("sphere union partition requires uniform material sides");
    }
    const double particle_volume_m3 = particle.reference_volume * particle.volume_ratio;
    if (!PositiveFinite(particle_volume_m3)) {
      return absl::OutOfRangeError("material particle volume is unrepresentable");
    }
    material_volume_m3.push_back(particle_volume_m3);
  }
  std::vector<std::size_t> particle_indices_by_id(particles.size());
  std::iota(particle_indices_by_id.begin(), particle_indices_by_id.end(), std::size_t{0});
  const auto particle_id = [&](std::size_t index) {
    return particles[index].id;
  };
  std::ranges::sort(particle_indices_by_id, {}, particle_id);
  if (std::ranges::adjacent_find(particle_indices_by_id, {}, particle_id) != particle_indices_by_id.end()) {
    return absl::InvalidArgumentError("duplicate material sphere particle ID");
  }

  std::vector<std::size_t> particle_indices_by_x = particle_indices_by_id;
  std::ranges::sort(particle_indices_by_x, {}, [&](std::size_t index) {
    return std::tuple(particles[index].world_position.x, particles[index].id);
  });
  std::vector<std::size_t> parent(particles.size());
  std::iota(parent.begin(), parent.end(), std::size_t{0});
  for (std::size_t sweep_index = 0; sweep_index < particle_indices_by_x.size(); ++sweep_index) {
    const auto particle_index = particle_indices_by_x[sweep_index];
    const auto& position = particles[particle_index].world_position;
    for (std::size_t neighbor_index = sweep_index + 1; neighbor_index < particle_indices_by_x.size();
         ++neighbor_index) {
      const auto neighbor_particle_index = particle_indices_by_x[neighbor_index];
      const auto& neighbor_position = particles[neighbor_particle_index].world_position;
      const double separation_x = neighbor_position.x - position.x;
      if (separation_x >= sphere_diameter_m) {
        break;
      }
      const double separation_y = std::abs(neighbor_position.y - position.y);
      const double separation_z = std::abs(neighbor_position.z - position.z);
      if (separation_y >= sphere_diameter_m || separation_z >= sphere_diameter_m) {
        continue;
      }
      const auto root = ComponentRoot(parent, particle_index);
      const auto neighbor_root = ComponentRoot(parent, neighbor_particle_index);
      if (root == neighbor_root) {
        continue;
      }
      if (std::hypot(separation_x, separation_y, separation_z) < sphere_diameter_m) {
        parent[std::max(root, neighbor_root)] = std::min(root, neighbor_root);
      }
    }
  }
  std::vector<SurfaceMaterialPartition3D> components;
  const std::size_t unassigned = particles.size();
  std::vector<std::size_t> component_indices_by_root(particles.size(), unassigned);
  for (const auto particle_index : particle_indices_by_id) {
    const auto root = ComponentRoot(parent, particle_index);
    auto& component_index = component_indices_by_root[root];
    if (component_index == unassigned) {
      component_index = components.size();
      components.emplace_back();
    }
    auto& component = components[component_index];
    const auto& particle = particles[particle_index];
    component.particle_indices.push_back(particle_index);
    component.members.push_back({
        .particle_id = particle.id,
        .reference_volume_m3 = particle.reference_volume,
    });
    component.target_volume_m3 += material_volume_m3[particle_index];
    if (!std::isfinite(component.target_volume_m3)) {
      return absl::OutOfRangeError("surface material component volume overflowed");
    }
  }
  return components;
}

absl::StatusOr<SurfaceComponentRepresentation3D>
ClassifySurfaceComponent3D(SurfaceComponentRepresentation3D previous, double diameter_m, double cell_width_m) {
  using Representation = SurfaceComponentRepresentation3D;
  if ((previous != Representation::kMesh && previous != Representation::kDroplet) || !PositiveFinite(diameter_m)
      || !PositiveFinite(cell_width_m)) {
    return absl::InvalidArgumentError("invalid liquid component diameter, cell width or representation");
  }
  const double diameter_cells = diameter_m / cell_width_m;
  if (!PositiveFinite(diameter_cells)) {
    return absl::OutOfRangeError("liquid component diameter ratio is unrepresentable");
  }
  if (previous == Representation::kMesh && diameter_cells < 2) {
    return Representation::kDroplet;
  }
  if (previous == Representation::kDroplet && diameter_cells > 3) {
    return Representation::kMesh;
  }
  return previous;
}
}  // namespace fluidity
