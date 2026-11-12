#include "graded_liquid_wet_transition_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "liquid_exact_inventory_sum_internal.h"

namespace fluidity {
namespace {

using Coordinates = std::array<double, 3>;
enum InventoryLane : std::size_t {
  kMass,
  kReferenceVolume,
  kCurrentVolume,
  kMomentumX,
  kMomentumY,
  kMomentumZ,
  kInventoryLaneCount,
};
using InventoryLanes = std::array<double, kInventoryLaneCount>;
using InventoryExpansion = std::array<liquid_exchange_internal::ExactInventorySum<64>, kInventoryLaneCount>;

Coordinates Components(Vec3 value) {
  return {value.x, value.y, value.z};
}

bool ContainsPosition(const FrozenGradedLiquidLeaf3D& leaf, Coordinates position) {
  const auto lower = Components(leaf.world_lower_m);
  const auto upper = Components(leaf.world_upper_m);
  for (std::size_t axis = 0; axis < position.size(); ++axis) {
    if (!(lower[axis] <= position[axis] && position[axis] < upper[axis])) {
      return false;
    }
  }
  return true;
}

// Bound the exact plane sign with one outward rounding step after each FMA.
// Since the planes translate and the particle path is straight, strict signs
// at both endpoints certify the entire segment by affine interpolation.
bool StrictPlaneSide(Vec3 normal, Coordinates position, double coordinate_m, bool positive_side) {
  const auto coefficients = Components(normal);
  const double outward =
      positive_side ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
  double bound = -coordinate_m;
  for (std::size_t axis = position.size(); axis > 0; --axis) {
    const double distance = std::fma(coefficients[axis - 1], position[axis - 1], bound);
    if (!std::isfinite(distance)) {
      return false;
    }
    bound = std::nextafter(distance, outward);
  }
  return positive_side ? bound > 0 : bound < 0;
}

bool CrossesOnlySubface(const FrozenGradedLiquidSubface3D& subface, Coordinates start, Coordinates end) {
  const auto lower = Components(subface.world_lower_m);
  const auto upper = Components(subface.world_upper_m);
  const auto normal_axis = static_cast<std::size_t>(subface.face_index.axis);
  if (!(std::min(start[normal_axis], end[normal_axis]) < lower[normal_axis]
        && lower[normal_axis] < std::max(start[normal_axis], end[normal_axis]))) {
    return false;
  }
  for (std::size_t axis = 0; axis < start.size(); ++axis) {
    if (axis == normal_axis) {
      continue;
    }
    if (start[axis] != end[axis] || !(lower[axis] < start[axis] && start[axis] < upper[axis])) {
      return false;
    }
  }
  return true;
}

absl::Status CheckPath(
    const LiquidParticleState3D& particle,
    const GradedLiquidParticlePath3D& path,
    const GradedLiquidWetTransitionRequest3D& request,
    double end_solid_coordinate_m,
    const FrozenGradedPlanarLiquidGeometry3D& geometry,
    std::span<const std::size_t> interior_ordinals_by_slot
) {
  const auto start = Components(path.start_m);
  const auto end = Components(path.end_m);
  if (path.particle_id != particle.id || start != Components(particle.world_position)) {
    return absl::FailedPreconditionError("Graded wet transition path differs from the accepted particle");
  }
  const auto leaves = geometry.leaves();
  if (path.start_leaf_ordinal >= leaves.size() || path.end_leaf_ordinal >= leaves.size()) {
    return absl::InvalidArgumentError("Graded wet transition path names a leaf outside the canonical roster");
  }
  if (!ContainsPosition(leaves[path.start_leaf_ordinal], start)
      || !ContainsPosition(leaves[path.end_leaf_ordinal], end)) {
    return absl::FailedPreconditionError("Graded wet transition endpoint is outside its named leaf");
  }
  if (!StrictPlaneSide(request.start_solid.unit_normal, start, request.start_solid.start_plane_coordinate, true)
      || !StrictPlaneSide(request.start_solid.unit_normal, end, end_solid_coordinate_m, true)
      || !StrictPlaneSide(request.start_liquid.unit_normal, start, request.start_liquid.plane_coordinate_m, false)
      || !StrictPlaneSide(request.start_liquid.unit_normal, end, request.end_liquid_plane_coordinate_m, false)) {
    return absl::FailedPreconditionError("Graded wet transition path leaves certified nonsolid liquid");
  }
  if (path.start_leaf_ordinal == path.end_leaf_ordinal) {
    if (path.crossing_face_slot) {
      return absl::FailedPreconditionError("Graded wet transition same-leaf path names an unexpected subface");
    }
    return absl::OkStatus();
  }
  if (!path.crossing_face_slot || *path.crossing_face_slot >= interior_ordinals_by_slot.size()) {
    return absl::FailedPreconditionError("Graded wet transition cross-leaf path requires a canonical interior slot");
  }
  const auto interior = geometry.topology().interior_faces();
  const std::size_t interior_ordinal = interior_ordinals_by_slot[*path.crossing_face_slot];
  if (interior_ordinal == interior.size()) {
    return absl::FailedPreconditionError("Graded wet transition path names an exterior subface");
  }
  const auto& face = interior[interior_ordinal];
  const bool forward =
      face.tail_cell_ordinal == path.start_leaf_ordinal && face.head_cell_ordinal == path.end_leaf_ordinal;
  const bool reverse =
      face.head_cell_ordinal == path.start_leaf_ordinal && face.tail_cell_ordinal == path.end_leaf_ordinal;
  if ((!forward && !reverse) || !CrossesOnlySubface(geometry.subfaces()[*path.crossing_face_slot], start, end)) {
    return absl::FailedPreconditionError("Graded wet transition path does not cross strictly inside its named subface");
  }
  return absl::OkStatus();
}

InventoryLanes Lanes(const LiquidRemapInventory3D& inventory) {
  return {
      inventory.mass_kg,
      inventory.reference_volume_m3,
      inventory.current_volume_m3,
      inventory.momentum_kg_m_per_s.x,
      inventory.momentum_kg_m_per_s.y,
      inventory.momentum_kg_m_per_s.z,
  };
}

bool AddRepresentedLanes(InventoryExpansion& expansion, const InventoryLanes& lanes, double sign) {
  for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
    if (!expansion[lane].Add(sign * lanes[lane])) {
      return false;
    }
  }
  return true;
}

absl::Status CheckExactBinnedTotal(InventoryExpansion residual, std::span<const InventoryLanes> bins) {
  for (const auto& bin : bins) {
    if (!AddRepresentedLanes(residual, bin, -1)) {
      return absl::OutOfRangeError("Graded wet transition exact inventory expansion is unrepresentable");
    }
  }
  for (const auto& lane : residual) {
    if (lane.sign() != 0) {
      return absl::FailedPreconditionError("Graded wet transition binned inventory loses represented material");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<LiquidCellOccupancy3D>
EndpointOccupancy(const FrozenGradedLiquidLeaf3D& geometry, double cell_volume_m3, const InventoryLanes& inventory) {
  if (geometry.fluid_volume_m3 > 0) {
    if (inventory[kMass] > 0 && inventory[kCurrentVolume] > 0
        && inventory[kCurrentVolume] <= std::min(geometry.fluid_volume_m3, cell_volume_m3)) {
      return LiquidCellOccupancy3D::kLiquid;
    }
    return absl::FailedPreconditionError("Graded wet transition wet leaf requires resident material within capacity");
  }
  if (geometry.known_air && inventory == InventoryLanes{}) {
    return LiquidCellOccupancy3D::kAir;
  }
  return absl::FailedPreconditionError("Graded wet transition dry leaf requires certified empty nonsolid air");
}

}  // namespace

absl::StatusOr<GradedLiquidWetTransition3D> PrepareGradedLiquidWetTransition3D(
    const LiquidState3D& accepted,
    const GradedLiquidWetTransitionRequest3D& request,
    std::span<const GradedLiquidParticlePath3D> paths
) {
  if (accepted.clock != request.expected_accepted_clock) {
    return absl::FailedPreconditionError("Graded wet transition request is stale");
  }
  if (accepted.adaptive_cells.size() == 0 || paths.size() != accepted.particles.size()
      || !std::isfinite(request.step_duration_s) || request.step_duration_s <= 0) {
    return absl::InvalidArgumentError("Graded wet transition requires an adaptive state and one path per particle");
  }
  if (const auto status = ValidateLiquidState3D(accepted); !status.ok()) {
    return status;
  }
  if (accepted.next_cavity_id != 0) {
    return absl::FailedPreconditionError("Graded wet transition requires a state without tracked cavities");
  }
  const double end_time_s = accepted.clock.time + request.step_duration_s;
  if (!std::isfinite(end_time_s) || end_time_s <= accepted.clock.time) {
    return absl::OutOfRangeError("Graded wet transition end time is unrepresentable");
  }
  std::vector<AdaptiveMacCellKey3D> keys;
  keys.reserve(accepted.adaptive_cells.size());
  for (const auto& leaf : accepted.adaptive_cells.values()) {
    if (leaf.material_owner != AdaptiveLiquidMaterialOwner3D::kParticles || leaf.inventory.phase_id != 0
        || leaf.inventory.constituent_count != 0 || leaf.inventory.enthalpy_j != 0) {
      return absl::FailedPreconditionError("Graded wet transition requires implicit particle-owned inventories");
    }
    keys.push_back(leaf.key);
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, accepted.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  auto before = FreezeGradedPlanarLiquidGeometry3D(
      *topology,
      accepted.geometry.world_origin,
      request.start_solid,
      request.start_liquid,
      request.step_duration_s
  );
  if (!before.ok()) {
    return before.status();
  }
  const auto& normal = request.start_solid.unit_normal;
  const auto& velocity = request.start_solid.wall_velocity;
  // Match the frozen geometry's signed-plane reduction order.
  const double wall_plane_speed_m_per_s =
      std::fma(normal.x, velocity.x, std::fma(normal.y, velocity.y, std::fma(normal.z, velocity.z, 0)));
  auto end_solid = request.start_solid;
  end_solid.start_plane_coordinate =
      std::fma(request.step_duration_s, wall_plane_speed_m_per_s, request.start_solid.start_plane_coordinate);
  auto end_liquid = request.start_liquid;
  end_liquid.plane_coordinate_m = request.end_liquid_plane_coordinate_m;
  auto after = FreezeGradedPlanarLiquidGeometry3D(
      *topology,
      accepted.geometry.world_origin,
      end_solid,
      end_liquid,
      request.step_duration_s
  );
  if (!after.ok()) {
    return after.status();
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto interior_slots = std::span(ownership->interior_face_slots);
  std::vector<std::size_t> interior_ordinals_by_slot(before->subfaces().size(), interior_slots.size());
  for (std::size_t ordinal = 0; ordinal < interior_slots.size(); ++ordinal) {
    interior_ordinals_by_slot[interior_slots[ordinal]] = ordinal;
  }

  std::vector<InventoryLanes> start_bins(keys.size());
  std::vector<InventoryLanes> end_bins(keys.size());
  InventoryExpansion represented;
  for (std::size_t ordinal = 0; ordinal < paths.size(); ++ordinal) {
    const auto& path = paths[ordinal];
    const auto& particle = accepted.particles[ordinal];
    if (particle.constituent_count != 0 || particle.side_tag != 0) {
      return absl::FailedPreconditionError("Graded wet transition particle has unsupported composition or side");
    }
    if (const auto status =
            CheckPath(particle, path, request, end_solid.start_plane_coordinate, *before, interior_ordinals_by_slot);
        !status.ok()) {
      return status;
    }
    const InventoryLanes particle_lanes{
        particle.mass,
        particle.reference_volume,
        particle.reference_volume * particle.volume_ratio,
        particle.mass * particle.world_velocity.x,
        particle.mass * particle.world_velocity.y,
        particle.mass * particle.world_velocity.z,
    };
    if (particle_lanes[kCurrentVolume] == 0 || (particle_lanes[kMomentumX] == 0 && particle.world_velocity.x != 0)
        || (particle_lanes[kMomentumY] == 0 && particle.world_velocity.y != 0)
        || (particle_lanes[kMomentumZ] == 0 && particle.world_velocity.z != 0)) {
      return absl::OutOfRangeError("Graded wet transition particle inventory underflow");
    }
    for (std::size_t lane = 0; lane < particle_lanes.size(); ++lane) {
      start_bins[path.start_leaf_ordinal][lane] += particle_lanes[lane];
      end_bins[path.end_leaf_ordinal][lane] += particle_lanes[lane];
      if (!std::isfinite(start_bins[path.start_leaf_ordinal][lane])
          || !std::isfinite(end_bins[path.end_leaf_ordinal][lane])) {
        return absl::OutOfRangeError("Graded wet transition particle inventory overflows its bin");
      }
    }
    if (!AddRepresentedLanes(represented, particle_lanes, 1)) {
      return absl::OutOfRangeError("Graded wet transition exact particle inventory is unrepresentable");
    }
  }
  if (const auto status = CheckExactBinnedTotal(represented, start_bins); !status.ok()) {
    return status;
  }
  if (const auto status = CheckExactBinnedTotal(represented, end_bins); !status.ok()) {
    return status;
  }

  std::vector<LiquidAdaptiveCellState3D> leaves;
  leaves.reserve(keys.size());
  for (std::size_t ordinal = 0; ordinal < keys.size(); ++ordinal) {
    auto leaf = accepted.adaptive_cells[ordinal];
    const double cell_volume_m3 = topology->cells()[ordinal].cell_volume;
    const auto start_occupancy = EndpointOccupancy(before->leaves()[ordinal], cell_volume_m3, start_bins[ordinal]);
    if (!start_occupancy.ok()) {
      return start_occupancy.status();
    }
    if (leaf.occupancy != *start_occupancy || Lanes(leaf.inventory) != start_bins[ordinal]) {
      return absl::FailedPreconditionError("Graded wet transition start occupancy or inventory differs from acceptance"
      );
    }
    const auto end_occupancy = EndpointOccupancy(after->leaves()[ordinal], cell_volume_m3, end_bins[ordinal]);
    if (!end_occupancy.ok()) {
      return end_occupancy.status();
    }
    if (leaf.occupancy != *end_occupancy) {
      leaf.previous_pressure = 0;
      leaf.coarsening_eligible_duration_s = 0;
    }
    leaf.occupancy = *end_occupancy;
    const auto& bin = end_bins[ordinal];
    leaf.inventory = {
        .mass_kg = bin[kMass],
        .reference_volume_m3 = bin[kReferenceVolume],
        .current_volume_m3 = bin[kCurrentVolume],
        .momentum_kg_m_per_s = {bin[kMomentumX], bin[kMomentumY], bin[kMomentumZ]},
    };
    leaves.push_back(leaf);
  }
  return GradedLiquidWetTransition3D{
      .request = request,
      .end_time_s = end_time_s,
      .before = std::move(*before),
      .after = std::move(*after),
      .leaves = std::move(leaves),
      .paths = {paths.begin(), paths.end()},
  };
}

}  // namespace fluidity
