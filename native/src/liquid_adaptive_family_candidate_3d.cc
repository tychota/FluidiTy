#include "liquid_adaptive_family_candidate_3d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "fluidity/host/adaptive_liquid_remap_3d.h"
#include "fluidity/host/graded_mac_transfer_3d.h"

namespace fluidity::liquid_step_internal {
namespace {

using FaceKey = std::tuple<FaceNormalAxis, std::int32_t, std::int32_t, std::int32_t, std::uint8_t>;

FaceKey CanonicalFaceKey(FaceIndex3D index, std::uint8_t width_exponent) {
  return {index.axis, index.lattice_index.z, index.lattice_index.y, index.lattice_index.x, width_exponent};
}

std::vector<FaceKey>
CanonicalFaceKeys(const AdaptiveMacTopology3D& topology, const AdaptiveMacFaceOwnership3D& ownership) {
  const auto interior_faces = topology.interior_faces();
  std::vector<FaceKey> keys(interior_faces.size() + ownership.exterior_faces.size());
  for (std::size_t face = 0; face < interior_faces.size(); ++face) {
    keys[ownership.interior_face_slots[face]] =
        CanonicalFaceKey(interior_faces[face].face_index, interior_faces[face].width_exponent);
  }
  for (const auto& face : ownership.exterior_faces) {
    keys[face.face_slot] = CanonicalFaceKey(face.face_index, face.width_exponent);
  }
  return keys;
}

// Use the same fused lattice-to-world bounds as AdaptiveLiquidStep3D.
absl::Status CheckTankBounds(
    const AdaptiveMacTopology3D& topology,
    Vec3 origin,
    const AxisAlignedLiquidTank3D& tank,
    double surface_height
) {
  const double finest_width = topology.finest_cell_width();
  for (const auto& leaf : topology.cells()) {
    const auto lower = leaf.key.lower_corner;
    const double lattice_width = std::ldexp(1.0, leaf.key.width_exponent);
    const Vec3 world_lower{
        std::fma(lower.x, finest_width, origin.x),
        std::fma(lower.y, finest_width, origin.y),
        std::fma(lower.z, finest_width, origin.z),
    };
    const Vec3 world_upper{
        std::fma(lower.x + lattice_width, finest_width, origin.x),
        std::fma(lower.y + lattice_width, finest_width, origin.y),
        std::fma(lower.z + lattice_width, finest_width, origin.z),
    };
    if (world_lower.x < tank.start_low_x || world_upper.x > tank.high_x || world_lower.y < tank.floor_y
        || world_upper.y > surface_height || world_lower.z < tank.low_z || world_upper.z > tank.high_z
        || world_lower.x >= world_upper.x || world_lower.y >= world_upper.y || world_lower.z >= world_upper.z) {
      return absl::InvalidArgumentError("graded family crosses a tank plane or has unrepresentable bounds");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<double>> ExteriorApertures(
    const AdaptiveMacFaceOwnership3D& ownership,
    const GradedMacTransfer3D& transfer,
    const AxisAlignedLiquidTank3D& tank,
    double surface_height
) {
  std::vector<double> areas_m2(ownership.exterior_faces.size());
  for (std::size_t ordinal = 0; ordinal < ownership.exterior_faces.size(); ++ordinal) {
    const auto& face = ownership.exterior_faces[ordinal];
    const auto anchor = transfer.faces()[face.face_slot].world_anchor;
    bool sealed = false;
    switch (face.face_index.axis) {
      case FaceNormalAxis::kX: {
        sealed = anchor.x == (face.cell_is_tail ? tank.high_x : tank.start_low_x);
        break;
      }
      case FaceNormalAxis::kY: {
        if (face.cell_is_tail && anchor.y == surface_height) {
          areas_m2[ordinal] = face.face_area;
          continue;
        }
        sealed = !face.cell_is_tail && anchor.y == tank.floor_y;
        break;
      }
      case FaceNormalAxis::kZ: {
        sealed = anchor.z == (face.cell_is_tail ? tank.high_z : tank.low_z);
        break;
      }
    }
    if (!sealed) {
      return absl::InvalidArgumentError("graded family has an exterior subface away from a tank plane");
    }
  }
  return areas_m2;
}

// A missing same-width sample holds the parent value at a wall or grade
// junction, giving zero limited slope on that axis.
LiquidRemapStencil3D ParentStencil(
    const AdaptiveMacTopology3D& topology,
    std::span<const LiquidRemapInventory3D> inventories,
    std::size_t parent_ordinal
) {
  LiquidRemapStencil3D stencil;
  for (auto& neighbors : stencil) {
    neighbors.fill(inventories[parent_ordinal]);
  }
  const auto parent_width = topology.cells()[parent_ordinal].key.width_exponent;
  for (const auto& face : topology.interior_faces()) {
    const bool parent_is_tail = face.tail_cell_ordinal == parent_ordinal;
    if (!parent_is_tail && face.head_cell_ordinal != parent_ordinal) {
      continue;
    }
    const auto neighbor = parent_is_tail ? face.head_cell_ordinal : face.tail_cell_ordinal;
    if (topology.cells()[neighbor].key.width_exponent == parent_width) {
      stencil[static_cast<std::size_t>(face.face_index.axis)][parent_is_tail] = inventories[neighbor];
    }
  }
  return stencil;
}

absl::Status CheckParticleOwnedInventories(const LiquidState3D& state, const GradedMacTransfer3D& transfer) {
  std::vector<LiquidRemapInventory3D> binned(state.adaptive_cells.size());
  for (const auto& particle : state.particles.values()) {
    const auto leaf = transfer.ParticleLeafOrdinal(particle.world_position);
    if (!leaf.ok()) {
      return leaf.status();
    }
    auto& inventory = binned[*leaf];
    inventory.mass_kg += particle.mass;
    inventory.reference_volume_m3 += particle.reference_volume;
    inventory.current_volume_m3 += particle.reference_volume * particle.volume_ratio;
    inventory.momentum_kg_m_per_s.x += particle.mass * particle.world_velocity.x;
    inventory.momentum_kg_m_per_s.y += particle.mass * particle.world_velocity.y;
    inventory.momentum_kg_m_per_s.z += particle.mass * particle.world_velocity.z;
  }
  for (std::size_t cell = 0; cell < binned.size(); ++cell) {
    auto particle_owned = state.adaptive_cells[cell];
    particle_owned.inventory = binned[cell];
    if (particle_owned != state.adaptive_cells[cell]) {
      return absl::FailedPreconditionError("graded family inventory requires certified particle resampling");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<double> IntegratedFlux(double carrier_velocity, double open_area_m2, double normal_sign) {
  const double flux_m3_per_s = normal_sign * carrier_velocity * open_area_m2;
  if (!std::isfinite(flux_m3_per_s) || (open_area_m2 > 0 && carrier_velocity != 0 && flux_m3_per_s == 0)) {
    return absl::OutOfRangeError("graded family face flux is unrepresentable");
  }
  return flux_m3_per_s;
}

absl::Status StageFace(LiquidAdaptiveFaceState3D& face, double flux_m3_per_s, double open_area_m2, double normal_sign) {
  if (open_area_m2 == 0) {
    face.carrier_velocity = 0;
  } else {
    face.carrier_velocity = normal_sign * flux_m3_per_s / open_area_m2;
  }
  if (!std::isfinite(face.carrier_velocity) || normal_sign * face.carrier_velocity * open_area_m2 != flux_m3_per_s) {
    return absl::OutOfRangeError("graded family face velocity is unrepresentable");
  }
  face.validity = LiquidFaceValidity3D::kValid;
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidFamilyCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D parent,
    bool refine
) {
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.obstacle);
  const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&scene.field);
  if (tank == nullptr || plane == nullptr || tank->piston_velocity_x != 0) {
    return absl::UnimplementedError("graded family change requires a stationary conforming planar tank");
  }
  if (committed.adaptive_cells.size() == 0 || committed.particles.size() == 0 || committed.next_cavity_id != 0) {
    return absl::FailedPreconditionError("graded family change requires particle-owned adaptive liquid");
  }
  const auto committed_status = ValidateLiquidState3D(committed);
  if (!committed_status.ok()) {
    return committed_status;
  }

  std::vector<AdaptiveMacCellKey3D> keys;
  std::vector<LiquidRemapInventory3D> inventories;
  std::vector<double> pressures_pa;
  keys.reserve(committed.adaptive_cells.size());
  inventories.reserve(committed.adaptive_cells.size());
  pressures_pa.reserve(committed.adaptive_cells.size());
  for (const auto& cell : committed.adaptive_cells.values()) {
    if (cell.occupancy != LiquidCellOccupancy3D::kLiquid || cell.inventory.enthalpy_j != 0
        || cell.inventory.phase_id != 0 || cell.inventory.constituent_count != 0) {
      return absl::UnimplementedError("graded family change requires implicit liquid leaves");
    }
    keys.push_back(cell.key);
    inventories.push_back(cell.inventory);
    pressures_pa.push_back(cell.previous_pressure);
  }
  for (const auto& particle : committed.particles.values()) {
    if (particle.phase_id != 0 || particle.constituent_count != 0) {
      return absl::UnimplementedError("graded family change requires implicit liquid particles");
    }
  }

  const auto topology = AdaptiveMacTopology3D::Create(keys, committed.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto family = refine ? PrepareAdaptiveRefinement3D(*topology, pressures_pa, parent)
                             : PrepareAdaptiveCoarsening3D(*topology, pressures_pa, parent);
  if (!family.ok()) {
    return family.status();
  }
  const auto old_ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  if (!old_ownership.ok()) {
    return old_ownership.status();
  }
  const auto new_ownership = BuildAdaptiveMacFaceOwnership3D(family->topology);
  if (!new_ownership.ok()) {
    return new_ownership.status();
  }
  const auto new_face_count = family->topology.interior_faces().size() + new_ownership->exterior_faces.size();
  if (family->topology.cells().size() > capacities.cells || new_face_count > capacities.faces) {
    return absl::ResourceExhaustedError("graded family change exceeds reserved cell or face capacity");
  }

  const auto origin = committed.geometry.world_origin;
  const Vec3 solid_upper{tank->high_x, 0, tank->high_z};
  const auto old_transfer = GradedMacTransfer3D::Create(*topology, origin, solid_upper);
  if (!old_transfer.ok()) {
    return old_transfer.status();
  }
  const auto new_transfer = GradedMacTransfer3D::Create(family->topology, origin, solid_upper);
  if (!new_transfer.ok()) {
    return new_transfer.status();
  }
  const double surface_height = plane->world_surface_height_y;
  for (const auto* tank_topology : {&*topology, &family->topology}) {
    const auto bounds = CheckTankBounds(*tank_topology, origin, *tank, surface_height);
    if (!bounds.ok()) {
      return bounds;
    }
  }
  const auto old_areas_m2 = ExteriorApertures(*old_ownership, *old_transfer, *tank, surface_height);
  if (!old_areas_m2.ok()) {
    return old_areas_m2.status();
  }
  const auto new_areas_m2 = ExteriorApertures(*new_ownership, *new_transfer, *tank, surface_height);
  if (!new_areas_m2.ok()) {
    return new_areas_m2.status();
  }
  const auto old_particle_inventory = CheckParticleOwnedInventories(committed, *old_transfer);
  if (!old_particle_inventory.ok()) {
    return old_particle_inventory;
  }

  const auto old_face_keys = CanonicalFaceKeys(*topology, *old_ownership);
  const auto new_face_keys = CanonicalFaceKeys(family->topology, *new_ownership);
  for (std::size_t slot = 0; slot < old_face_keys.size(); ++slot) {
    const auto& face = committed.adaptive_faces[slot];
    if (std::ranges::binary_search(new_face_keys, old_face_keys[slot])) {
      continue;
    }
    const bool missing_nonzero_carrier = face.validity == LiquidFaceValidity3D::kMissing && face.carrier_velocity != 0;
    if (face.side_tag != 0 || !face.structurally_eligible || missing_nonzero_carrier) {
      return absl::FailedPreconditionError("graded family change cannot remap incompatible face history");
    }
  }

  const auto interior_faces = topology->interior_faces();
  std::vector<double> interior_fluxes_m3_per_s(interior_faces.size());
  for (std::size_t ordinal = 0; ordinal < interior_faces.size(); ++ordinal) {
    const auto slot = old_ownership->interior_face_slots[ordinal];
    const auto& face = committed.adaptive_faces[slot];
    const auto flux = IntegratedFlux(face.carrier_velocity, interior_faces[ordinal].face_area, 1);
    if (!flux.ok()) {
      return flux.status();
    }
    interior_fluxes_m3_per_s[ordinal] = *flux;
  }
  std::vector<double> exterior_fluxes_m3_per_s(old_ownership->exterior_faces.size());
  for (std::size_t ordinal = 0; ordinal < exterior_fluxes_m3_per_s.size(); ++ordinal) {
    const auto& face = old_ownership->exterior_faces[ordinal];
    const auto flux = IntegratedFlux(
        committed.adaptive_faces[face.face_slot].carrier_velocity,
        (*old_areas_m2)[ordinal],
        face.cell_is_tail ? 1 : -1
    );
    if (!flux.ok()) {
      return flux.status();
    }
    exterior_fluxes_m3_per_s[ordinal] = *flux;
  }

  const AdaptiveLiquidRemapInput3D input{
      .inventories = inventories,
      .previous_pressures_pa = pressures_pa,
      .interior_fluxes_m3_per_s = interior_fluxes_m3_per_s,
      .exterior =
          AdaptiveLiquidExteriorRemapInput3D{
              .committed_open_areas_m2 = *old_areas_m2,
              .committed_outward_fluxes_m3_per_s = exterior_fluxes_m3_per_s,
              .candidate_open_areas_m2 = *new_areas_m2,
          },
  };
  const auto proposal = [&]() {
    if (refine) {
      const auto stencil = ParentStencil(*topology, inventories, family->parent_ordinal);
      return PrepareAdaptiveLiquidRefinement3D(*topology, input, parent, stencil);
    }
    return PrepareAdaptiveLiquidCoarsening3D(*topology, input, parent);
  }();
  if (!proposal.ok()) {
    return proposal.status();
  }

  auto candidate = CloneLiquidState3D(committed, capacities);
  if (!candidate.ok()) {
    return candidate.status();
  }
  const auto resized_cells = candidate->adaptive_cells.Resize(proposal->inventories.size());
  if (!resized_cells.ok()) {
    return resized_cells;
  }
  const auto resized_faces = candidate->adaptive_faces.Resize(new_face_count);
  if (!resized_faces.ok()) {
    return resized_faces;
  }
  for (std::size_t ordinal = 0; ordinal < proposal->inventories.size(); ++ordinal) {
    candidate->adaptive_cells[ordinal] = {
        .key = proposal->family.topology.cells()[ordinal].key,
        .occupancy = LiquidCellOccupancy3D::kLiquid,
        .inventory = proposal->inventories[ordinal],
        .previous_pressure = proposal->family.previous_pressures[ordinal],
    };
  }
  const auto particle_inventory = CheckParticleOwnedInventories(*candidate, *new_transfer);
  if (!particle_inventory.ok()) {
    return particle_inventory;
  }

  const auto publish_face = [&](std::size_t slot, double flux_m3_per_s, double open_area_m2, double normal_sign) {
    auto& face = candidate->adaptive_faces[slot];
    const auto match = std::ranges::lower_bound(old_face_keys, new_face_keys[slot]);
    if (match != old_face_keys.end() && *match == new_face_keys[slot]) {
      face = committed.adaptive_faces[static_cast<std::size_t>(match - old_face_keys.begin())];
      if (open_area_m2 == 0) {
        face.carrier_velocity = 0;
      }
      return absl::OkStatus();
    }
    face = LiquidAdaptiveFaceState3D{};
    return StageFace(face, flux_m3_per_s, open_area_m2, normal_sign);
  };
  const auto new_interior_faces = proposal->family.topology.interior_faces();
  for (std::size_t ordinal = 0; ordinal < new_interior_faces.size(); ++ordinal) {
    const auto status = publish_face(
        new_ownership->interior_face_slots[ordinal],
        proposal->interior_fluxes_m3_per_s[ordinal],
        new_interior_faces[ordinal].face_area,
        1
    );
    if (!status.ok()) {
      return status;
    }
  }
  for (std::size_t ordinal = 0; ordinal < new_ownership->exterior_faces.size(); ++ordinal) {
    const auto& face = new_ownership->exterior_faces[ordinal];
    const auto status = publish_face(
        face.face_slot,
        proposal->exterior_outward_fluxes_m3_per_s[ordinal],
        (*new_areas_m2)[ordinal],
        face.cell_is_tail ? 1 : -1
    );
    if (!status.ok()) {
      return status;
    }
  }

  const auto validation = ValidateLiquidState3D(*candidate);
  if (!validation.ok()) {
    return validation;
  }
  return candidate;
}

}  // namespace fluidity::liquid_step_internal
