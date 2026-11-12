#include "liquid_adaptive_family_candidate_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "fluidity/host/adaptive_liquid_remap_3d.h"
#include "fluidity/host/graded_mac_transfer_3d.h"
#include "fluidity/host/graded_quadratic_mac_transfer_3d.h"
#include "liquid_remap_3d_internal.h"

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
  std::vector<liquid_remap_internal::ConstituentSums> constituent_sums;
  if (state.particles[0].phase_id != 0) {
    constituent_sums.resize(binned.size());
  }
  for (const auto& particle : state.particles.values()) {
    const auto leaf = transfer.ParticleLeafOrdinal(particle.world_position);
    if (!leaf.ok()) {
      return leaf.status();
    }
    auto& inventory = binned[*leaf];
    inventory.mass_kg += particle.mass;
    inventory.reference_volume_m3 += particle.reference_volume;
    const double particle_volume = particle.reference_volume * particle.volume_ratio;
    inventory.current_volume_m3 += particle_volume;
    if (particle.constituent_count != 0) {
      inventory.phase_id = particle.phase_id;
      const auto constituents = std::span(particle.constituents).first(particle.constituent_count);
      for (const auto& constituent : constituents) {
        const auto added =
            constituent_sums[*leaf].Add(constituent.substance_id, constituent.volume_fraction * particle_volume);
        if (!added.ok()) {
          return added;
        }
      }
    }
    inventory.momentum_kg_m_per_s.x += particle.mass * particle.world_velocity.x;
    inventory.momentum_kg_m_per_s.y += particle.mass * particle.world_velocity.y;
    inventory.momentum_kg_m_per_s.z += particle.mass * particle.world_velocity.z;
  }
  for (std::size_t cell = 0; cell < binned.size(); ++cell) {
    if (!constituent_sums.empty()) {
      const auto stored = constituent_sums[cell].Store(binned[cell]);
      if (!stored.ok()) {
        return stored;
      }
    }
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

struct QuadraticResamplingRequest {
  const LiquidState3D& source;
  const AdaptiveMacTopology3D& source_topology;
  std::span<const std::uint64_t> particle_ids;
};

auto LeafKey(AdaptiveMacCellKey3D key) {
  const auto lower = key.lower_corner;
  return std::tuple{lower.z, lower.y, lower.x, key.width_exponent};
}

// The stationary tank admission proves that all interior subfaces are wet
// and unobstructed. The quadratic binder never uses an exterior subface.
std::vector<double>
QuadraticInteriorAreas(const AdaptiveMacTopology3D& topology, const AdaptiveMacFaceOwnership3D& ownership) {
  const auto interior = topology.interior_faces();
  std::vector<double> areas(interior.size() + ownership.exterior_faces.size());
  for (std::size_t ordinal = 0; ordinal < interior.size(); ++ordinal) {
    areas[ownership.interior_face_slots[ordinal]] = interior[ordinal].face_area;
  }
  return areas;
}

absl::Status ApplyQuadraticResampling(
    const QuadraticResamplingRequest& request,
    AdaptiveMacCellKey3D parent,
    bool refine,
    const AdaptiveMacTopology3D& candidate_topology,
    const AdaptiveMacFaceOwnership3D& candidate_ownership,
    LiquidState3D& candidate
) {
  const auto& source = request.source;
  const auto& source_topology = request.source_topology;
  const auto roster = candidate.particles.values();
  const auto source_location = GradedMacTransfer3D::Create(source_topology, source.geometry.world_origin);
  if (!source_location.ok()) {
    return source_location.status();
  }
  std::array<LiquidParticleState3D, 8> selected{};
  std::size_t selected_count = 0;
  std::size_t first_selected = roster.size();
  const auto parent_lower = parent.lower_corner;
  const auto parent_lattice_width = std::int64_t{1} << parent.width_exponent;
  for (std::size_t ordinal = 0; ordinal < roster.size(); ++ordinal) {
    const auto& particle = roster[ordinal];
    if (std::ranges::find(request.particle_ids, particle.id) == request.particle_ids.end()) {
      continue;
    }
    const auto source_leaf = source_location->ParticleLeafOrdinal(particle.world_position);
    if (!source_leaf.ok()) {
      return source_leaf.status();
    }
    const auto key = source_topology.cells()[*source_leaf].key;
    if (refine) {
      if (LeafKey(key) != LeafKey(parent)) {
        return absl::FailedPreconditionError("quadratic split particle is outside the requested parent leaf");
      }
    } else {
      const auto child_lower = key.lower_corner;
      const auto inside = [parent_lattice_width](std::int32_t child, std::int32_t low) {
        return child >= low && static_cast<std::int64_t>(child) < low + parent_lattice_width;
      };
      if (key.width_exponent + 1 != parent.width_exponent || !inside(child_lower.x, parent_lower.x)
          || !inside(child_lower.y, parent_lower.y) || !inside(child_lower.z, parent_lower.z)) {
        return absl::FailedPreconditionError("quadratic merge particle is outside the requested fine family");
      }
    }
    first_selected = std::min(first_selected, ordinal);
    selected[selected_count++] = particle;
  }
  if (selected_count != request.particle_ids.size()) {
    return absl::FailedPreconditionError("quadratic resampling requires distinct committed particle IDs");
  }

  const auto source_ownership = BuildAdaptiveMacFaceOwnership3D(source_topology);
  if (!source_ownership.ok()) {
    return source_ownership.status();
  }

  const auto source_transfer = GradedQuadraticMacTransfer3D::Create(source_topology, source.geometry.world_origin);
  if (!source_transfer.ok()) {
    return source_transfer.status();
  }
  const auto candidate_transfer =
      GradedQuadraticMacTransfer3D::Create(candidate_topology, candidate.geometry.world_origin);
  if (!candidate_transfer.ok()) {
    return candidate_transfer.status();
  }
  const auto source_areas = QuadraticInteriorAreas(source_topology, *source_ownership);
  const auto candidate_areas = QuadraticInteriorAreas(candidate_topology, candidate_ownership);
  const GradedQuadraticMacSupportSnapshot3D source_support{
      .transfer = *source_transfer,
      .open_areas_m2 = source_areas,
      .carrier_faces = source.adaptive_faces.values(),
  };
  const GradedQuadraticMacSupportSnapshot3D candidate_support{
      .transfer = *candidate_transfer,
      .open_areas_m2 = candidate_areas,
      .carrier_faces = candidate.adaptive_faces.values(),
  };
  // Both topologies already satisfy stationary tank bounds. The complete
  // wet support and exact per-leaf inventory check certify this implicit phase.
  const auto certify = [](const LiquidParticleState3D& checked, double, LiquidUniformResamplingBasis3D, void*) {
    if (checked.side_tag != 0 || checked.phase_id != 0 || checked.constituent_count != 0) {
      return absl::UnimplementedError("quadratic family resampling requires the implicit single liquid");
    }
    return absl::OkStatus();
  };

  const double parent_width_m = std::ldexp(source.geometry.cell_width, parent.width_exponent);
  if (refine) {
    constexpr std::size_t kAdditionalParticles = 7;
    if (candidate.particles.capacity() - roster.size() < kAdditionalParticles) {
      return absl::ResourceExhaustedError("quadratic split exceeds the particle capacity");
    }
    const auto split = SplitGradedQuadraticLiquidParticle3D(
        selected[0],
        candidate.next_particle_id,
        parent_width_m,
        source_support,
        candidate_support,
        certify,
        nullptr
    );
    if (!split.ok()) {
      return split.status();
    }
    const auto resized = candidate.particles.Resize(roster.size() + kAdditionalParticles);
    if (!resized.ok()) {
      return resized;
    }
    auto expanded = candidate.particles.values();
    const auto former_tail = expanded.subspan(first_selected + 1, roster.size() - first_selected - 1);
    std::ranges::move_backward(former_tail, expanded.end());
    std::ranges::copy(split->children, expanded.begin() + static_cast<std::ptrdiff_t>(first_selected));
    candidate.next_particle_id = split->next_particle_id;
    return absl::OkStatus();
  }

  const auto merged = MergeGradedQuadraticLiquidParticles3D(
      std::span(selected).first(selected_count),
      candidate.next_particle_id,
      parent_width_m / 2,
      source_support,
      candidate_support,
      certify,
      nullptr
  );
  if (!merged.ok()) {
    return merged.status();
  }
  // Compact forward in accepted roster order and put the replacement at the
  // first removed ordinal. All source records for F4 have already been copied.
  std::size_t target_ordinal = first_selected;
  roster[target_ordinal++] = merged->particle;
  for (std::size_t ordinal = first_selected + 1; ordinal < roster.size(); ++ordinal) {
    if (std::ranges::find(request.particle_ids, roster[ordinal].id) == request.particle_ids.end()) {
      roster[target_ordinal++] = roster[ordinal];
    }
  }
  const auto resized = candidate.particles.Resize(target_ordinal);
  if (!resized.ok()) {
    return resized;
  }
  candidate.next_particle_id = merged->next_particle_id;
  return absl::OkStatus();
}

absl::StatusOr<LiquidState3D> PrepareFamily(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D parent,
    bool refine,
    const QuadraticResamplingRequest* resampling
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
  const auto phase_id = committed.particles[0].phase_id;
  for (const auto& cell : committed.adaptive_cells.values()) {
    if (cell.material_owner != AdaptiveLiquidMaterialOwner3D::kParticles) {
      return absl::UnimplementedError("graded family change has no Eulerian liquid carrier");
    }
    if (cell.occupancy != LiquidCellOccupancy3D::kLiquid || cell.inventory.enthalpy_j != 0
        || (cell.inventory.phase_id != 0 && cell.inventory.phase_id != phase_id)) {
      return absl::UnimplementedError("graded family change requires leaves from one miscible liquid phase");
    }
    keys.push_back(cell.key);
    inventories.push_back(cell.inventory);
    pressures_pa.push_back(cell.previous_pressure);
  }
  for (const auto& particle : committed.particles.values()) {
    if (particle.phase_id != phase_id) {
      return absl::UnimplementedError("graded family change requires particles from one miscible liquid phase");
    }
    if (!LiquidParticleUsesRegisteredConstituents3D(particle, scene.constituent_materials)) {
      return absl::InvalidArgumentError("graded family particle composition is absent from the material registry");
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
  if (resampling == nullptr) {
    const auto particle_inventory = CheckParticleOwnedInventories(*candidate, *new_transfer);
    if (!particle_inventory.ok()) {
      return particle_inventory;
    }
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

  if (resampling != nullptr) {
    const auto resampled =
        ApplyQuadraticResampling(*resampling, parent, refine, family->topology, *new_ownership, *candidate);
    if (!resampled.ok()) {
      return resampled;
    }
    const auto particle_inventory = CheckParticleOwnedInventories(*candidate, *new_transfer);
    if (!particle_inventory.ok()) {
      return particle_inventory;
    }
  }

  const auto validation = ValidateLiquidState3D(*candidate);
  if (!validation.ok()) {
    return validation;
  }
  return candidate;
}

absl::StatusOr<AdaptiveMacTopology3D> TopologyFromAdaptiveState(const LiquidState3D& state) {
  std::vector<AdaptiveMacCellKey3D> keys;
  keys.reserve(state.adaptive_cells.size());
  for (const auto& leaf : state.adaptive_cells.values()) {
    keys.push_back(leaf.key);
  }
  return AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
}

// All edits remain detached. The selected family is last so its two
// certificates see the original accepted layout and the full candidate halo.
absl::StatusOr<LiquidState3D> StageFamilyEdits(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    std::span<const AdaptiveMacCellKey3D> ordered,
    bool refine,
    const QuadraticResamplingRequest* resampling
) {
  std::optional<LiquidState3D> staged;
  for (std::size_t ordinal = 0; ordinal < ordered.size(); ++ordinal) {
    const auto& source = staged ? *staged : committed;
    const auto* request = ordinal + 1 == ordered.size() ? resampling : nullptr;
    auto candidate = PrepareFamily(source, scene, capacities, ordered[ordinal], refine, request);
    if (!candidate.ok()) {
      return candidate.status();
    }
    staged.emplace(std::move(*candidate));
  }
  return std::move(*staged);
}

absl::StatusOr<LiquidState3D> PrepareRefinementClosure(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    const AdaptiveMacTopology3D& topology,
    std::span<const AdaptiveMacCellKey3D> requested_parents,
    const AdaptiveMacCellKey3D* split_parent,
    std::uint64_t split_particle_id,
    bool require_exact_closure
) {
  if (requested_parents.empty()) {
    return absl::InvalidArgumentError("adaptive refinement closure requires at least one parent");
  }
  if (requested_parents.size() > kMaximumAdaptiveLiquidCells3D) {
    return absl::ResourceExhaustedError("adaptive refinement request exceeds the leaf limit");
  }
  const auto leaves = topology.cells();
  const auto key_of_leaf = [](const AdaptiveMacLeaf3D& leaf) {
    return LeafKey(leaf.key);
  };
  std::vector<std::size_t> requested;
  requested.reserve(requested_parents.size());
  for (const auto key : requested_parents) {
    const auto lookup = LeafKey(key);
    const auto found = std::ranges::lower_bound(leaves, lookup, {}, key_of_leaf);
    if (found == leaves.end() || LeafKey(found->key) != lookup) {
      return absl::FailedPreconditionError("adaptive refinement closure requires committed parent leaves");
    }
    requested.push_back(static_cast<std::size_t>(found - leaves.begin()));
  }
  if (require_exact_closure) {
    std::ranges::sort(requested);
    if (std::ranges::adjacent_find(requested) != requested.end()) {
      return absl::InvalidArgumentError("certified refinement closure contains duplicate parents");
    }
  }
  auto closure = topology.RefinementClosure(requested, capacities.cells);
  if (!closure.ok()) {
    return closure.status();
  }
  if (require_exact_closure && closure->size() != requested.size()) {
    return absl::FailedPreconditionError("certified refinement closure omits a required parent");
  }
  const auto selected_key = split_parent == nullptr ? std::nullopt : std::optional(LeafKey(*split_parent));
  std::ranges::sort(*closure, {}, [&](std::size_t ordinal) {
    const auto key = leaves[ordinal].key;
    const bool selected = selected_key && LeafKey(key) == *selected_key;
    return std::tuple{-static_cast<int>(key.width_exponent), selected, ordinal};
  });
  if (selected_key && LeafKey(leaves[closure->back()].key) != *selected_key) {
    return absl::FailedPreconditionError("quadratic split parent must be the final graded closure edit");
  }

  std::vector<AdaptiveMacCellKey3D> ordered;
  ordered.reserve(closure->size());
  for (const auto ordinal : *closure) {
    ordered.push_back(leaves[ordinal].key);
  }
  const QuadraticResamplingRequest resampling{
      .source = committed,
      .source_topology = topology,
      .particle_ids = std::span(&split_particle_id, 1),
  };
  return StageFamilyEdits(committed, scene, capacities, ordered, true, selected_key ? &resampling : nullptr);
}

}  // namespace

absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidFamilyCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    AdaptiveMacCellKey3D parent,
    bool refine
) {
  return PrepareFamily(committed, scene, capacities, parent, refine, nullptr);
}

absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidRefinementClosureCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    const AdaptiveMacTopology3D& topology,
    std::span<const AdaptiveMacCellKey3D> requested_parents
) {
  return PrepareRefinementClosure(committed, scene, capacities, topology, requested_parents, nullptr, 0, false);
}

absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidQuadraticSplitCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    std::span<const AdaptiveMacCellKey3D> requested_parents,
    AdaptiveMacCellKey3D split_parent,
    std::uint64_t split_particle_id,
    bool require_exact_closure
) {
  if (split_particle_id == 0) {
    return absl::InvalidArgumentError("quadratic split requires a parent particle ID");
  }
  const auto topology = TopologyFromAdaptiveState(committed);
  if (!topology.ok()) {
    return topology.status();
  }
  return PrepareRefinementClosure(
      committed,
      scene,
      capacities,
      *topology,
      requested_parents,
      &split_parent,
      split_particle_id,
      require_exact_closure
  );
}

absl::StatusOr<LiquidState3D> PrepareAdaptiveLiquidQuadraticMergeCandidate3D(
    const LiquidState3D& committed,
    const LiquidSceneInputs3D& scene,
    LiquidStateCapacities3D capacities,
    std::span<const AdaptiveMacCellKey3D> requested_parents,
    AdaptiveMacCellKey3D merge_parent,
    std::span<const std::uint64_t> children,
    bool require_exact_closure
) {
  constexpr std::size_t kMaximumChildren = 8;
  if (children.size() < 2 || children.size() > kMaximumChildren || requested_parents.empty()
      || merge_parent.width_exponent == 0 || merge_parent.width_exponent > 30) {
    return absl::InvalidArgumentError("quadratic merge requires a valid parent, two to eight children, and a halo");
  }
  if (requested_parents.size() > kMaximumAdaptiveLiquidCells3D) {
    return absl::ResourceExhaustedError("quadratic merge halo exceeds the leaf limit");
  }
  std::array<std::uint64_t, kMaximumChildren> sorted_ids{};
  auto requested_ids = std::span(sorted_ids).first(children.size());
  std::ranges::copy(children, requested_ids.begin());
  std::ranges::sort(requested_ids);
  if (requested_ids.front() == 0 || std::ranges::adjacent_find(requested_ids) != requested_ids.end()) {
    return absl::InvalidArgumentError("quadratic merge requires distinct nonzero child IDs");
  }

  std::vector<AdaptiveMacCellKey3D> ordered(requested_parents.begin(), requested_parents.end());
  const auto selected_key = LeafKey(merge_parent);
  for (const auto key : ordered) {
    if (key.width_exponent != merge_parent.width_exponent) {
      return absl::FailedPreconditionError("quadratic merge halo requires one coarsening level");
    }
  }
  std::ranges::sort(ordered, {}, [&](const AdaptiveMacCellKey3D& key) {
    const auto canonical = LeafKey(key);
    return std::tuple{canonical == selected_key, canonical};
  });
  const auto duplicate_tail = std::ranges::unique(ordered, {}, LeafKey);
  if (require_exact_closure && !duplicate_tail.empty()) {
    return absl::InvalidArgumentError("certified coarsening halo contains duplicate parents");
  }
  ordered.erase(duplicate_tail.begin(), duplicate_tail.end());
  if (LeafKey(ordered.back()) != selected_key) {
    return absl::FailedPreconditionError("quadratic merge parent must be in the coarsening halo");
  }

  const auto topology = TopologyFromAdaptiveState(committed);
  if (!topology.ok()) {
    return topology.status();
  }
  const QuadraticResamplingRequest resampling{
      .source = committed,
      .source_topology = *topology,
      .particle_ids = children,
  };
  return StageFamilyEdits(committed, scene, capacities, ordered, false, &resampling);
}

}  // namespace fluidity::liquid_step_internal
