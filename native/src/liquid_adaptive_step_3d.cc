#include "liquid_adaptive_step_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "fluidity/base/liquid_boundary_geometry_math_3d.h"
#include "fluidity/host/adaptive_liquid_core_transport_3d.h"
#include "fluidity/host/liquid_eos.h"
#include "graded_liquid_boundary_geometry_3d.h"
#include "liquid_substep_3d_internal.h"
#include "mac_deposit_3d_internal.h"
#include "mac_pressure_3d_arithmetic.h"
#include "mac_viscous_pressure_3d.h"

namespace fluidity::liquid_step_internal {
namespace {

double NormalComponent(Vec3 value, FaceNormalAxis axis) {
  switch (axis) {
    case FaceNormalAxis::kX: {
      return value.x;
    }
    case FaceNormalAxis::kY: {
      return value.y;
    }
    case FaceNormalAxis::kZ: {
      return value.z;
    }
  }
  return 0;
}

bool IsEulerian(const LiquidAdaptiveCellState3D& cell) {
  return cell.material_owner == AdaptiveLiquidMaterialOwner3D::kEulerian;
}

bool IsWall(const AxisAlignedLiquidTank3D& tank, const AdaptiveMacExteriorFace3D& face, double plane) {
  switch (face.face_index.axis) {
    case FaceNormalAxis::kX: {
      return plane == (face.cell_is_tail ? tank.high_x : tank.start_low_x);
    }
    case FaceNormalAxis::kY: {
      return !face.cell_is_tail && plane == tank.floor_y;
    }
    case FaceNormalAxis::kZ: {
      return plane == (face.cell_is_tail ? tank.high_z : tank.low_z);
    }
  }
  return false;
}

void AddNormalComponent(Vec3& derivatives, FaceNormalAxis axis, double derivative) {
  switch (axis) {
    case FaceNormalAxis::kX: {
      derivatives.x += derivative;
      break;
    }
    case FaceNormalAxis::kY: {
      derivatives.y += derivative;
      break;
    }
    case FaceNormalAxis::kZ: {
      derivatives.z += derivative;
      break;
    }
  }
}

Vec3 Difference(Vec3 left, Vec3 right) {
  return {left.x - right.x, left.y - right.y, left.z - right.z};
}

double Dot(Vec3 first, Vec3 second) {
  return first.x * second.x + first.y * second.y + first.z * second.z;
}

Vec3 Cross(Vec3 first, Vec3 second) {
  return {
      first.y * second.z - first.z * second.y,
      first.z * second.x - first.x * second.z,
      first.x * second.y - first.y * second.x,
  };
}

using FaceGradient = std::array<std::array<MacStrainTerm3D, 4>, 3>;

// Select a deterministic local affine stencil. The determinant is measured
// in units of leaf width cubed. Sealed anchors have prescribed zero velocity.
absl::StatusOr<FaceGradient>
FitFaceGradient(std::span<const GradedMacFace3D> faces, Vec3 center, double width, FaceNormalAxis velocity_axis) {
  struct Anchor {
    std::size_t slot;
    Vec3 offset;
    double squared_distance;
  };
  std::vector<Anchor> candidates;
  constexpr double kRadius = 1.5;
  for (std::size_t slot = 0; slot < faces.size(); ++slot) {
    const auto& face = faces[slot];
    if (face.axis != velocity_axis) {
      continue;
    }
    const Vec3 offset{
        (face.world_anchor.x - center.x) / width,
        (face.world_anchor.y - center.y) / width,
        (face.world_anchor.z - center.z) / width,
    };
    if (std::max({std::abs(offset.x), std::abs(offset.y), std::abs(offset.z)}) <= kRadius) {
      candidates.push_back({slot, offset, Dot(offset, offset)});
    }
  }
  std::ranges::sort(candidates, [](const auto& first, const auto& second) {
    return std::tie(first.squared_distance, first.slot) < std::tie(second.squared_distance, second.slot);
  });
  if (candidates.size() < 4) {
    return absl::FailedPreconditionError("graded viscosity has fewer than four local face anchors");
  }
  constexpr double kMinimumRankMeasure = 1e-4;
  const auto base = candidates.front();
  const auto first = std::ranges::find_if(candidates, [&](const auto& candidate) {
    const auto offset = Difference(candidate.offset, base.offset);
    return Dot(offset, offset) >= kMinimumRankMeasure;
  });
  if (first == candidates.end()) {
    return absl::FailedPreconditionError("graded viscosity face anchors are coincident");
  }
  const auto first_offset = Difference(first->offset, base.offset);
  const auto second = std::ranges::find_if(candidates, [&](const auto& candidate) {
    const auto cross = Cross(first_offset, Difference(candidate.offset, base.offset));
    return Dot(cross, cross) >= kMinimumRankMeasure;
  });
  if (second == candidates.end()) {
    return absl::FailedPreconditionError("graded viscosity face anchors are collinear");
  }
  const auto second_offset = Difference(second->offset, base.offset);
  const auto normal = Cross(first_offset, second_offset);
  const auto third = std::ranges::find_if(candidates, [&](const auto& candidate) {
    return std::abs(Dot(normal, Difference(candidate.offset, base.offset))) >= kMinimumRankMeasure;
  });
  if (third == candidates.end()) {
    return absl::FailedPreconditionError("graded viscosity face anchors are coplanar");
  }
  const auto third_offset = Difference(third->offset, base.offset);
  const double inverse_determinant_per_meter = (1.0 / Dot(normal, third_offset)) / width;
  const std::array columns{Cross(second_offset, third_offset), Cross(third_offset, first_offset), normal};
  const std::array slots{first->slot, second->slot, third->slot};
  constexpr std::array axes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
  FaceGradient result{};
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    auto& terms = result[axis];
    terms[0] = {base.slot, 0.0};
    for (std::size_t column = 0; column < columns.size(); ++column) {
      const double coefficient = NormalComponent(columns[column], axes[axis]) * inverse_determinant_per_meter;
      terms[column + 1] = {slots[column], coefficient};
      terms[0].coefficient -= coefficient;
    }
  }
  return result;
}

// Combine finite-volume normal rows with symmetric shear rows from affine
// gradients at the owned staggered anchors. Fully blocked anchors contribute
// their prescribed velocities to the strain rather than to its unknowns.
absl::Status AppendViscousRows(
    const AdaptiveMacLeaf3D& leaf,
    Vec3 center,
    double viscosity,
    std::span<const GradedMacFace3D> faces,
    std::span<const pressure_internal::AdaptiveMacPressureSubface3D> subfaces,
    std::span<const double> sealed_wall_velocities,
    const std::array<MacStrainRow3D, 3>& normal_rows,
    std::vector<MacStrainRow3D>& rows
) {
  constexpr std::array axes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
  std::array<FaceGradient, 3> gradients{};
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    const auto gradient = FitFaceGradient(faces, center, leaf.cell_width, axes[axis]);
    if (!gradient.ok()) {
      return gradient.status();
    }
    gradients[axis] = *gradient;
  }
  const auto add_terms = [&](MacStrainRow3D& row, const auto& terms, double scale) {
    for (const auto& term : terms) {
      const double coefficient = scale * term.coefficient;
      if (coefficient == 0.0) {
        continue;
      }
      if (subfaces[term.face_slot].open_area_m2 > 0.0) {
        row.terms[row.term_count] = {term.face_slot, coefficient};
        ++row.term_count;
      } else {
        row.prescribed_strain += coefficient * sealed_wall_velocities[term.face_slot];
      }
    }
  };
  for (std::size_t first = 0; first < axes.size(); ++first) {
    for (std::size_t second = first; second < axes.size(); ++second) {
      if (first == second) {
        if (normal_rows[first].term_count != 0) {
          rows.push_back(normal_rows[first]);
        }
        continue;
      }
      const double quadrature_weight = 4.0 * viscosity * leaf.cell_volume;
      if (!std::isnormal(quadrature_weight)) {
        return absl::OutOfRangeError("graded viscous strain quadrature is unrepresentable");
      }
      MacStrainRow3D row{
          .low_cell = leaf.key.lower_corner,
          .first_axis = axes[first],
          .second_axis = axes[second],
          .terms = {},
          .term_count = 0,
          .quadrature_weight = quadrature_weight,
      };
      add_terms(row, gradients[first][second], 0.5);
      add_terms(row, gradients[second][first], 0.5);
      if (!std::isfinite(row.prescribed_strain)) {
        return absl::OutOfRangeError("graded prescribed shear strain is unrepresentable");
      }
      if (row.term_count != 0) {
        rows.push_back(row);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<MacStrainRow3D>> AssembleGradedViscousStrainRows3D(
    const AdaptiveMacTopology3D& topology,
    Vec3 world_origin,
    double dynamic_viscosity,
    std::span<const GradedMacFace3D> faces,
    std::span<const pressure_internal::AdaptiveMacPressureSubface3D> subfaces,
    std::span<std::size_t> leaf_row_offsets
) {
  if (faces.size() != subfaces.size() || !std::isfinite(dynamic_viscosity) || dynamic_viscosity <= 0.0) {
    return absl::InvalidArgumentError("graded viscosity needs one physical record per face and positive viscosity");
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto interior = topology.interior_faces();
  if (faces.size() != interior.size() + ownership->exterior_faces.size()) {
    return absl::InvalidArgumentError("graded strain face extent does not match canonical ownership");
  }
  const auto leaves = topology.cells();
  if (!leaf_row_offsets.empty()) {
    if (leaf_row_offsets.size() != leaves.size() + 1) {
      return absl::InvalidArgumentError("graded strain row offsets need one entry per leaf and an end offset");
    }
    leaf_row_offsets.front() = 0;
  }
  std::vector<std::array<MacStrainRow3D, 3>> normal_rows(leaves.size());
  constexpr std::array axes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
  for (std::size_t cell = 0; cell < leaves.size(); ++cell) {
    const auto& leaf = leaves[cell];
    const double weight = 2.0 * dynamic_viscosity * leaf.cell_volume;
    if (!std::isnormal(weight)) {
      return absl::OutOfRangeError("graded normal strain quadrature is unrepresentable");
    }
    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
      normal_rows[cell][axis] = {
          .low_cell = leaf.key.lower_corner,
          .first_axis = axes[axis],
          .second_axis = axes[axis],
          .terms = {},
          .term_count = 0,
          .quadrature_weight = weight,
      };
    }
  }
  std::vector<double> sealed_wall_velocities(faces.size());
  // The three normal rows sum to the pressure operator's finite-volume
  // divergence. A 2:1 leaf side contributes at most four owned subfaces.
  const auto add_normal = [&](std::size_t cell, std::size_t slot, FaceNormalAxis axis, double area, double sign) {
    auto& row = normal_rows[cell][static_cast<std::size_t>(axis)];
    const double open_area = subfaces[slot].open_area_m2;
    if (!std::isfinite(open_area) || open_area < 0.0 || open_area > area || faces[slot].axis != axis) {
      return absl::InvalidArgumentError("graded strain face geometry does not match its open subface");
    }
    const double wall_flux = subfaces[slot].prescribed_wall_flux_m3_per_s;
    if (!std::isfinite(wall_flux) || (wall_flux != 0.0 && open_area == area)) {
      return absl::InvalidArgumentError("graded prescribed wall flux needs finite blocked support");
    }
    if (wall_flux != 0.0) {
      if (open_area > 0.0) {
        return absl::UnimplementedError("graded moving cut strain requires separate wall quadrature");
      }
      const double wall_velocity = wall_flux / area;
      const double normal_strain = sign * wall_flux / leaves[cell].cell_volume;
      if (!std::isnormal(wall_velocity) || !std::isnormal(normal_strain)) {
        return absl::OutOfRangeError("graded prescribed wall velocity or strain is unrepresentable");
      }
      sealed_wall_velocities[slot] = wall_velocity;
      row.prescribed_strain += normal_strain;
      if (!std::isfinite(row.prescribed_strain)) {
        return absl::OutOfRangeError("graded prescribed normal strain is unrepresentable");
      }
    }
    if (open_area == 0.0) {
      return absl::OkStatus();
    }
    const double coefficient = sign * open_area / leaves[cell].cell_volume;
    if (row.term_count == row.terms.size() || !std::isnormal(coefficient)) {
      return absl::OutOfRangeError("graded normal strain coefficient or extent is unrepresentable");
    }
    row.terms[row.term_count++] = {slot, coefficient};
    return absl::OkStatus();
  };
  for (std::size_t face = 0; face < interior.size(); ++face) {
    const auto& link = interior[face];
    const auto slot = ownership->interior_face_slots[face];
    const auto tail = add_normal(link.tail_cell_ordinal, slot, link.face_index.axis, link.face_area, 1.0);
    if (!tail.ok()) {
      return tail;
    }
    const auto head = add_normal(link.head_cell_ordinal, slot, link.face_index.axis, link.face_area, -1.0);
    if (!head.ok()) {
      return head;
    }
  }
  for (const auto& link : ownership->exterior_faces) {
    const auto status = add_normal(
        link.cell_ordinal,
        link.face_slot,
        link.face_index.axis,
        link.face_area,
        link.cell_is_tail ? 1.0 : -1.0
    );
    if (!status.ok()) {
      return status;
    }
  }
  std::vector<MacStrainRow3D> rows;
  rows.reserve(6 * leaves.size());
  const double finest_width = topology.finest_cell_width();
  for (std::size_t cell = 0; cell < leaves.size(); ++cell) {
    const auto& leaf = leaves[cell];
    const auto lower = leaf.key.lower_corner;
    const double half_width = leaf.cell_width / 2.0;
    const Vec3 center{
        std::fma(lower.x, finest_width, world_origin.x) + half_width,
        std::fma(lower.y, finest_width, world_origin.y) + half_width,
        std::fma(lower.z, finest_width, world_origin.z) + half_width,
    };
    const auto status = AppendViscousRows(
        leaf,
        center,
        dynamic_viscosity,
        faces,
        subfaces,
        sealed_wall_velocities,
        normal_rows[cell],
        rows
    );
    if (!status.ok()) {
      return status;
    }
    if (!leaf_row_offsets.empty()) {
      leaf_row_offsets[cell + 1] = rows.size();
    }
  }
  if (rows.empty()) {
    return absl::FailedPreconditionError("graded viscosity has no open strain support");
  }
  return rows;
}

AdaptiveLiquidStep3D::AdaptiveLiquidStep3D(AdaptiveMacTopology3D topology, GradedMacTransfer3D transfer)
    : topology_(std::move(topology)), transfer_(std::move(transfer)) {
}

absl::StatusOr<AdaptiveLiquidStep3D> AdaptiveLiquidStep3D::Create(
    const LiquidState3D& initial,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& options
) {
  const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&scene.field);
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.obstacle);
  const bool physical = std::holds_alternative<TranslatingHalfspace3D>(scene.obstacle);
  const bool stationary_tank = tank != nullptr && tank->piston_velocity_x == 0;
  if (plane == nullptr || (!physical && !stationary_tank) || scene.target_divergence != 0 || initial.next_cavity_id != 0
      || options.whitewater_capture || !scene.sources.empty() || options.ocean_boundary.has_value()
      || options.capacity_replay_reservations != LiquidStateCapacities3D{}) {
    return absl::UnimplementedError("graded Simulate requires a conforming tank or explicit physical halfspace");
  }
  if (physical
      && (options.motion_sample_intervals != 0 || scene.surface_tension != 0 || scene.dynamic_viscosity != 0
          || scene.eos || !std::isinf(scene.bulk_modulus) || !scene.constituent_materials.empty()
          || std::ranges::any_of(initial.adaptive_cells.values(), IsEulerian))) {
    return absl::UnimplementedError("graded physical halfspace requires implicit incompressible particle-owned liquid");
  }
  if (options.motion_sample_intervals != 0
      && (scene.surface_tension != 0 || scene.dynamic_viscosity != 0 || scene.drift || scene.eos || scene.cavity
          || scene.cavity_provider.prepare || !scene.phase_pairs.empty() || !scene.constituent_materials.empty()
          || !std::isinf(scene.bulk_modulus))) {
    return absl::UnimplementedError("graded motion capture requires the implicit incompressible tank");
  }
  if (initial.particles.size() == 0) {
    return absl::InvalidArgumentError("graded Simulate requires particle-owned material");
  }
  const auto phase_id = initial.particles[0].phase_id;
  for (const auto& particle : initial.particles.values()) {
    if (physical && (particle.phase_id != 0 || particle.side_tag != 0)) {
      return absl::UnimplementedError("graded physical halfspace requires side-zero implicit particles");
    }
    if (particle.phase_id != phase_id) {
      return absl::UnimplementedError("graded tank requires one miscible phase");
    }
    if (!LiquidParticleUsesRegisteredConstituents3D(particle, scene.constituent_materials)) {
      return absl::InvalidArgumentError("graded particle composition is absent from the material registry");
    }
  }
  std::vector<AdaptiveMacCellKey3D> keys;
  keys.reserve(initial.adaptive_cells.size());
  for (const auto& cell : initial.adaptive_cells.values()) {
    if (cell.occupancy != LiquidCellOccupancy3D::kLiquid || (!IsEulerian(cell) && cell.inventory.enthalpy_j != 0)) {
      return absl::UnimplementedError("graded tank requires supported liquid inventory in every leaf");
    }
    if (IsEulerian(cell)
        && (phase_id != 0 || cell.inventory.phase_id != 0 || cell.inventory.constituent_count != 0
            || std::isfinite(scene.bulk_modulus) || scene.eos || scene.dynamic_viscosity > 0
            || scene.surface_tension > 0)) {
      return absl::UnimplementedError("graded extended liquids require particle-owned material");
    }
    keys.push_back(cell.key);
  }
  auto topology = AdaptiveMacTopology3D::Create(keys, initial.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto solid_upper = tank ? std::optional{Vec3{tank->high_x, 0, tank->high_z}} : std::nullopt;
  auto transfer = GradedMacTransfer3D::Create(*topology, initial.geometry.world_origin, solid_upper);
  if (!transfer.ok()) {
    return transfer.status();
  }
  AdaptiveLiquidStep3D result(std::move(*topology), std::move(*transfer));
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(result.topology_);
  if (!ownership.ok()) {
    return ownership.status();
  }
  result.interior_slots_ = ownership->interior_face_slots;
  result.exterior_faces_ = ownership->exterior_faces;
  result.surface_height_ = plane->world_surface_height_y;
  result.subfaces_.resize(initial.adaptive_faces.size());
  const auto leaves = result.topology_.cells();
  if (std::ranges::any_of(initial.adaptive_cells.values(), IsEulerian)) {
    auto core_transport = AdaptiveLiquidCoreTransportWorkspace3D::Create(leaves.size());
    if (!core_transport.ok()) {
      return core_transport.status();
    }
    result.core_transport_.emplace(std::move(*core_transport));
    result.core_faces_.resize(initial.adaptive_faces.size());
    result.core_fluxes_.resize(result.topology_.interior_faces().size());
  }
  result.pressure_cells_.resize(leaves.size());
  result.update_volume_ratio_ = std::isfinite(scene.bulk_modulus) || scene.eos.has_value();
  if (result.update_volume_ratio_) {
    result.normal_derivatives_.resize(leaves.size());
  }
  if (scene.eos) {
    result.pressure_history_.resize(leaves.size());
  }
  if (scene.eos || phase_id != 0) {
    result.predictor_density_.resize(leaves.size());
    result.reference_density_.resize(leaves.size());
  }
  if (phase_id != 0) {
    result.constituent_sums_.resize(leaves.size());
  }
  const auto origin = initial.geometry.world_origin;
  const double finest_width = initial.geometry.cell_width;
  for (std::size_t cell = 0; cell < leaves.size(); ++cell) {
    const auto& leaf = leaves[cell];
    const auto lower = leaf.key.lower_corner;
    const Vec3 world_lower{
        std::fma(lower.x, finest_width, origin.x),
        std::fma(lower.y, finest_width, origin.y),
        std::fma(lower.z, finest_width, origin.z),
    };
    const double lattice_width = std::ldexp(1.0, leaf.key.width_exponent);
    const Vec3 world_upper{
        std::fma(lower.x + lattice_width, finest_width, origin.x),
        std::fma(lower.y + lattice_width, finest_width, origin.y),
        std::fma(lower.z + lattice_width, finest_width, origin.z),
    };
    const bool crosses_tank = tank
        && (world_lower.x < tank->start_low_x || world_upper.x > tank->high_x || world_lower.y < tank->floor_y
            || world_upper.y > result.surface_height_ || world_lower.z < tank->low_z || world_upper.z > tank->high_z);
    if (crosses_tank || world_lower.x >= world_upper.x || world_lower.y >= world_upper.y
        || world_lower.z >= world_upper.z) {
      return absl::InvalidArgumentError("graded leaf crosses a tank plane or has unrepresentable bounds");
    }
    if (IsEulerian(initial.adaptive_cells[cell]) && initial.adaptive_cells[cell].inventory.current_volume_m3 <= 0) {
      return absl::FailedPreconditionError("Eulerian core leaf must contain liquid");
    }
    result.pressure_cells_[cell] = {
        .cell_index = lower,
        .fluid_volume = leaf.cell_volume,
        .target_divergence = scene.target_divergence,
        .bulk_modulus = scene.bulk_modulus,
        .previous_pressure = 0,
    };
  }
  const double inverse_density = 1 / scene.liquid_density;
  for (std::size_t face = 0; face < result.topology_.interior_faces().size(); ++face) {
    const auto& interior = result.topology_.interior_faces()[face];
    const auto slot = ownership->interior_face_slots[face];
    if (IsEulerian(initial.adaptive_cells[interior.tail_cell_ordinal])
        || IsEulerian(initial.adaptive_cells[interior.head_cell_ordinal])) {
      const auto& carrier = initial.adaptive_faces[slot];
      if (carrier.validity != LiquidFaceValidity3D::kValid || !carrier.structurally_eligible || carrier.side_tag != 0) {
        return absl::FailedPreconditionError("Eulerian core requires a valid implicit-liquid carrier");
      }
      result.core_faces_[slot] = 1;
    }
    result.subfaces_[slot] = {
        .open_area_m2 = interior.face_area,
        .inverse_density_m3_per_kg = inverse_density,
    };
  }
  for (const auto& face : ownership->exterior_faces) {
    if (physical) {
      continue;
    }
    const auto axis = face.face_index.axis;
    const double coordinate = NormalComponent(result.transfer_.faces()[face.face_slot].world_anchor, axis);
    if (IsWall(*tank, face, coordinate)) {
      continue;
    }
    if (axis != FaceNormalAxis::kY || !face.cell_is_tail || coordinate != result.surface_height_) {
      return absl::InvalidArgumentError("graded tank has an exterior subface away from a wall or the free surface");
    }
    if (IsEulerian(initial.adaptive_cells[face.cell_ordinal])) {
      return absl::FailedPreconditionError("Eulerian core may not reach the open tank surface");
    }
    result.subfaces_[face.face_slot] = {
        .open_area_m2 = face.face_area,
        .inverse_density_m3_per_kg = inverse_density,
        .exterior_pressure_distance_m = leaves[face.cell_ordinal].cell_width / 2,
        .exterior_pressure_pa = scene.ambient_pressure,
    };
  }
  result.old_faces_.resize(initial.adaptive_faces.size());
  result.gathered_.resize(initial.particles.size());
  result.distances_.resize(initial.particles.size());
  result.inventories_.resize(initial.adaptive_cells.size());
  const auto material = result.AccumulateInventories(initial);
  if (!material.ok()) {
    return material;
  }
  for (std::size_t cell = 0; cell < result.inventories_.size(); ++cell) {
    if (IsEulerian(initial.adaptive_cells[cell])) {
      continue;
    }
    auto particle_owned = initial.adaptive_cells[cell];
    particle_owned.inventory = result.inventories_[cell];
    if (particle_owned != initial.adaptive_cells[cell]) {
      return absl::InvalidArgumentError("graded tank inventory must match its particle-owned material");
    }
  }
  const bool composition_viscosity =
      phase_id != 0 && std::ranges::any_of(scene.constituent_materials, [&](const auto& material) {
        return material.phase_id == phase_id && material.dynamic_viscosity_pa_s.has_value();
      });
  bool can_apply_viscosity = scene.dynamic_viscosity > 0.0;
  if (composition_viscosity) {
    // Only substances already present in this closed tank can enter a leaf during transport.
    can_apply_viscosity = std::ranges::any_of(result.inventories_, [&](const auto& inventory) {
      const auto constituents = std::span(inventory.constituents).first(inventory.constituent_count);
      return std::ranges::any_of(constituents, [&](const auto& constituent) {
        const auto material = std::ranges::lower_bound(
            scene.constituent_materials,
            constituent.substance_id,
            {},
            &LiquidConstituentMaterial3D::substance_id
        );
        return material->dynamic_viscosity_pa_s.value_or(scene.dynamic_viscosity) > 0.0;
      });
    });
  }
  if (can_apply_viscosity) {
    if (composition_viscosity) {
      result.leaf_strain_row_offsets_.resize(leaves.size() + 1);
    }
    auto rows = AssembleGradedViscousStrainRows3D(
        result.topology_,
        origin,
        composition_viscosity ? 1.0 : scene.dynamic_viscosity,
        result.transfer_.faces(),
        result.subfaces_,
        result.leaf_strain_row_offsets_
    );
    if (!rows.ok()) {
      return rows.status();
    }
    if (composition_viscosity) {
      result.strain_rows_.reserve(rows->size());
      result.unit_strain_rows_ = std::move(*rows);
    } else {
      result.strain_rows_ = std::move(*rows);
    }
  }
  return result;
}

std::size_t AdaptiveLiquidStep3D::reserved_bytes() const {
  const std::size_t core_transport_bytes = core_transport_ ? core_transport_->reserved_bytes() : 0;
  const std::size_t physical_trial_bytes = physical_trial_
      ? physical_trial_->wet_open_faces.capacity() * sizeof(std::uint8_t)
          + physical_trial_->end_fluid_volumes_m3.capacity() * sizeof(double)
      : 0;
  return topology_.reserved_bytes() + transfer_.reserved_bytes() + interior_slots_.capacity() * sizeof(std::size_t)
      + physical_trial_bytes + subfaces_.capacity() * sizeof(pressure_internal::AdaptiveMacPressureSubface3D)
      + pressure_cells_.capacity() * sizeof(MacPressureCell3D)
      + (strain_rows_.capacity() + unit_strain_rows_.capacity()) * sizeof(MacStrainRow3D)
      + leaf_strain_row_offsets_.capacity() * sizeof(std::size_t)
      + old_faces_.capacity() * sizeof(LiquidAdaptiveFaceState3D)
      + gathered_.capacity() * sizeof(LiquidBoundaryParticleGather3D) + distances_.capacity() * sizeof(double)
      + inventories_.capacity() * sizeof(LiquidRemapInventory3D) + core_faces_.capacity() * sizeof(std::uint8_t)
      + core_fluxes_.capacity() * sizeof(double) + core_transport_bytes;
}

absl::Status AdaptiveLiquidStep3D::PreparePhysicalTrial(
    const LiquidState3D& accepted,
    double duration,
    const LiquidSceneInputs3D& scene,
    const LiquidAdaptivePhysicalGeometryRequest3D& request
) {
  if (request.target_divergences_s_inv.size() != topology_.cells().size()
      || (!request.exterior_pressures.empty() && request.exterior_pressures.size() != subfaces_.size())) {
    return absl::InvalidArgumentError("Graded physical request fields must cover the canonical layout");
  }
  const double end_time = accepted.clock.time + duration;
  if (!std::isfinite(end_time) || end_time <= accepted.clock.time) {
    return absl::OutOfRangeError("Graded physical trial end time is unrepresentable");
  }
  const auto start_wall = ObstacleAtTime(scene.obstacle, accepted.clock.time);
  const auto end_wall = ObstacleAtTime(scene.obstacle, end_time);
  const auto* start_halfspace = std::get_if<TranslatingHalfspace3D>(&start_wall);
  const auto* end_halfspace = std::get_if<TranslatingHalfspace3D>(&end_wall);
  if (start_halfspace == nullptr || end_halfspace == nullptr) {
    return absl::FailedPreconditionError("Graded physical requests require a translating halfspace scene");
  }
  const GradedLiquidPlanarSurface3D surface{
      .unit_normal = {0, 1, 0},
      .plane_coordinate_m = surface_height_,
  };
  const auto start_geometry = FreezeGradedPlanarLiquidGeometry3D(
      topology_,
      accepted.geometry.world_origin,
      *start_halfspace,
      surface,
      duration
  );
  if (!start_geometry.ok()) {
    return start_geometry.status();
  }
  auto frozen_end_wall = *end_halfspace;
  frozen_end_wall.wall_velocity = {};
  const auto end_geometry =
      FreezeGradedPlanarLiquidGeometry3D(topology_, accepted.geometry.world_origin, frozen_end_wall, surface, duration);
  if (!end_geometry.ok()) {
    return end_geometry.status();
  }

  PhysicalTrial3D physical{
      .start_wall = start_wall,
      .end_wall = end_wall,
      .wet_open_faces = {},
      .end_fluid_volumes_m3 = {},
  };
  physical.end_fluid_volumes_m3.reserve(topology_.cells().size());
  std::vector<GradedLiquidPressureCellFields3D> cells;
  cells.reserve(topology_.cells().size());
  for (std::size_t cell = 0; cell < topology_.cells().size(); ++cell) {
    const double start_volume_m3 = start_geometry->leaves()[cell].fluid_volume_m3;
    const double end_volume_m3 = end_geometry->leaves()[cell].fluid_volume_m3;
    if (start_volume_m3 <= 0 || end_volume_m3 <= 0) {
      return absl::FailedPreconditionError("Graded physical trial requires positive wet volume throughout the sweep");
    }
    if (accepted.adaptive_cells[cell].inventory.current_volume_m3 > start_volume_m3) {
      return absl::FailedPreconditionError("Graded physical trial inventory exceeds its initial wet volume");
    }
    physical.end_fluid_volumes_m3.push_back(end_volume_m3);
    cells.push_back({
        .target_divergence_s_inv = request.target_divergences_s_inv[cell],
        .previous_pressure_pa = accepted.adaptive_cells[cell].previous_pressure,
    });
  }
  for (const auto& particle : accepted.particles.values()) {
    const double wall_distance = SolidDistance(start_wall, particle.world_position);
    if (!std::isfinite(wall_distance) || wall_distance < 0 || particle.world_position.y >= surface_height_) {
      return absl::FailedPreconditionError("Graded physical trial starts with a particle outside the liquid");
    }
  }

  physical.wet_open_faces.reserve(subfaces_.size());
  std::vector<GradedLiquidPressureSubfaceFields3D> faces;
  faces.reserve(subfaces_.size());
  const double inverse_density = 1 / scene.liquid_density;
  for (std::size_t slot = 0; slot < subfaces_.size(); ++slot) {
    std::optional<GradedLiquidPressureExterior3D> exterior;
    if (!request.exterior_pressures.empty() && request.exterior_pressures[slot]) {
      const auto& input = *request.exterior_pressures[slot];
      exterior = GradedLiquidPressureExterior3D{.pressure_pa = input.pressure_pa, .distance_m = input.distance_m};
    }
    physical.wet_open_faces.push_back(start_geometry->subfaces()[slot].wet_open_area_m2 > 0);
    faces.push_back({
        .inverse_density_m3_per_kg = inverse_density,
        .exterior_pressure = exterior,
    });
  }
  auto pressure = ComposeGradedPlanarLiquidPressureInputs3D(*start_geometry, cells, faces);
  if (!pressure.ok()) {
    return pressure.status();
  }
  pressure_cells_ = std::move(pressure->cells);
  subfaces_ = std::move(pressure->subfaces);
  physical_trial_ = std::move(physical);
  return absl::OkStatus();
}

absl::StatusOr<std::optional<PreparedAdaptiveQuadraticTransfer3D>> AdaptiveLiquidStep3D::PrepareQuadraticTransfer(
    const LiquidState3D& committed,
    const LiquidStepOptions3D& options,
    std::uint64_t particle_id
) {
  return StageQuadraticTransfer(committed, options, std::span(&particle_id, 1), true);
}

absl::StatusOr<PreparedAdaptiveQuadraticTransfer3D> AdaptiveLiquidStep3D::PrepareCertifiedQuadraticTransfer(
    const LiquidState3D& candidate,
    const LiquidStepOptions3D& options,
    std::span<const std::uint64_t> replacement_ids
) {
  auto prepared = StageQuadraticTransfer(candidate, options, replacement_ids, false);
  if (!prepared.ok()) {
    return prepared.status();
  }
  if (!prepared->has_value()) {
    return absl::InternalError("certified quadratic transfer selected an unexpected fallback");
  }
  return std::move(**prepared);
}

absl::StatusOr<std::optional<PreparedAdaptiveQuadraticTransfer3D>> AdaptiveLiquidStep3D::StageQuadraticTransfer(
    const LiquidState3D& state,
    const LiquidStepOptions3D& options,
    std::span<const std::uint64_t> particle_ids,
    bool allow_radial_fallback
) {
  if (particle_ids.empty() || particle_ids.size() > PreparedAdaptiveQuadraticTransfer3D::kMaximumParticles) {
    return absl::InvalidArgumentError("quadratic transfer requires one to eight particle IDs");
  }
  if (physical_trial_ || !core_faces_.empty() || update_volume_ratio_) {
    return absl::FailedPreconditionError("explicit quadratic transfer requires incompressible particle-owned liquid");
  }
  std::array<std::uint64_t, PreparedAdaptiveQuadraticTransfer3D::kMaximumParticles> sorted_ids{};
  const auto requested = std::span(sorted_ids).first(particle_ids.size());
  std::ranges::copy(particle_ids, requested.begin());
  std::ranges::sort(requested);
  if (requested.front() == 0 || std::ranges::adjacent_find(requested) != requested.end()) {
    return absl::InvalidArgumentError("quadratic transfer requires distinct nonzero particle IDs");
  }

  const auto particles = state.particles.values();
  PreparedAdaptiveQuadraticTransfer3D prepared{.epoch = state.clock.epoch, .particles = {}, .particle_count = 0};
  for (std::size_t ordinal = 0; ordinal < particles.size(); ++ordinal) {
    const auto& particle = particles[ordinal];
    if (std::ranges::binary_search(requested, particle.id)) {
      if (prepared.particle_count == requested.size()) {
        return absl::InvalidArgumentError("staged state contains duplicate quadratic particle IDs");
      }
      prepared.particles[prepared.particle_count++] = {
          .particle_ordinal = ordinal,
          .particle_id = particle.id,
          .world_position_m = particle.world_position,
          .stencil = {},
      };
    }
  }
  if (prepared.particle_count != requested.size()) {
    return absl::InvalidArgumentError("quadratic transfer particle ID is absent from the staged state");
  }

  const auto transfer_particle = [](const LiquidParticleState3D& particle) {
    return MacTransferParticle3D{
        .sorted_world_position = particle.world_position,
        .trial_world_position = particle.world_position,
        .particle_mass = particle.mass,
        .particle_velocity = particle.world_velocity,
        .particle_velocity_slopes = particle.affine_enabled ? particle.velocity_slopes : Mat3{},
    };
  };
  const auto carrier_faces = state.adaptive_faces.values();
  for (auto& selection : std::span(prepared.particles).first(prepared.particle_count)) {
    const auto& particle = particles[selection.particle_ordinal];
    if (particle.phase_id != 0) {
      return absl::FailedPreconditionError("explicit quadratic transfer requires implicit liquid");
    }
    const GradedQuadraticMacParticle3D input{.transfer = transfer_particle(particle), .side_tag = particle.side_tag};
    const auto valid = internal::ValidateMacDepositParticle(input.transfer);
    if (!valid.ok()) {
      return valid;
    }
    const auto bound = GradedQuadraticMacTransfer3D::BindWithLayout(
        input,
        topology_,
        transfer_,
        state.geometry.world_origin,
        interior_slots_,
        [&](std::size_t slot) {
          return subfaces_[slot].open_area_m2;
        },
        carrier_faces
    );
    if (allow_radial_fallback && absl::IsFailedPrecondition(bound.status())) {
      return std::nullopt;
    }
    if (!bound.ok()) {
      return bound.status();
    }
    selection.stencil = *bound;
  }

  // The global particle order fixes rounding where any two contributions
  // share a face. The normalized old snapshot is reused by Deposit.
  std::vector<MacFaceDeposit3D> deposits(old_faces_.size());
  auto selections = prepared.selections();
  for (std::size_t ordinal = 0; ordinal < particles.size(); ++ordinal) {
    const auto& particle = particles[ordinal];
    const bool selected = !selections.empty() && ordinal == selections.front().particle_ordinal;
    const auto status = selected ? GradedQuadraticMacTransfer3D::AccumulateBoundDeposit(
                                       transfer_particle(particle),
                                       selections.front().stencil,
                                       deposits
                                   )
                                 : transfer_.AccumulateParticleDeposit(particle, carrier_faces, deposits);
    if (!status.ok()) {
      return status;
    }
    if (selected) {
      selections = selections.subspan(1);
    }
  }
  const auto normalized = NormalizeDeposits(state, options, deposits);
  if (!normalized.ok()) {
    return normalized;
  }
  for (const auto& selection : prepared.selections()) {
    const auto& particle = particles[selection.particle_ordinal];
    const auto carriers =
        GradedQuadraticMacTransfer3D::CheckBoundCarriers(selection.stencil, particle.side_tag, old_faces_);
    if (allow_radial_fallback && absl::IsFailedPrecondition(carriers)) {
      return std::nullopt;
    }
    if (!carriers.ok()) {
      return carriers;
    }
  }
  return prepared;
}

absl::Status AdaptiveLiquidStep3D::Deposit(
    const LiquidState3D& trial,
    const LiquidStepOptions3D& options,
    const PreparedAdaptiveQuadraticTransfer3D* quadratic
) {
  if (quadratic != nullptr) {
    if (trial.clock.epoch != quadratic->epoch) {
      return absl::FailedPreconditionError("prepared quadratic transfer does not match the trial epoch");
    }
    for (const auto& selection : quadratic->selections()) {
      if (selection.particle_ordinal >= trial.particles.size()) {
        return absl::FailedPreconditionError("prepared quadratic transfer does not match the trial extent");
      }
      const auto& particle = trial.particles[selection.particle_ordinal];
      const auto position = particle.world_position;
      const auto expected_position = selection.world_position_m;
      if (particle.id != selection.particle_id || position.x != expected_position.x || position.y != expected_position.y
          || position.z != expected_position.z) {
        return absl::FailedPreconditionError("prepared quadratic transfer does not match the trial particle");
      }
    }
    return absl::OkStatus();
  }
  auto deposition_faces = trial.adaptive_faces.values();
  if (physical_trial_) {
    std::ranges::copy(deposition_faces, old_faces_.begin());
    for (std::size_t face = 0; face < old_faces_.size(); ++face) {
      old_faces_[face].structurally_eligible =
          old_faces_[face].structurally_eligible && physical_trial_->wet_open_faces[face] != 0;
    }
    deposition_faces = old_faces_;
  }
  const auto deposits = transfer_.Deposit(trial.particles.values(), deposition_faces);
  if (!deposits.ok()) {
    return deposits.status();
  }
  return NormalizeDeposits(trial, options, *deposits);
}

absl::Status AdaptiveLiquidStep3D::NormalizeDeposits(
    const LiquidState3D& trial,
    const LiquidStepOptions3D& options,
    std::span<const MacFaceDeposit3D> deposits
) {
  if (deposits.size() != old_faces_.size() || trial.adaptive_faces.size() != old_faces_.size()) {
    return absl::InvalidArgumentError("graded deposit extent does not cover the canonical faces");
  }
  for (std::size_t face = 0; face < old_faces_.size(); ++face) {
    old_faces_[face] = trial.adaptive_faces[face];
    if (subfaces_[face].open_area_m2 == 0) {
      old_faces_[face].carrier_velocity = 0;
      const bool absent_physical_carrier = physical_trial_ && physical_trial_->wet_open_faces[face] == 0;
      old_faces_[face].validity =
          absent_physical_carrier ? LiquidFaceValidity3D::kMissing : LiquidFaceValidity3D::kValid;
      continue;
    }
    // The accepted Eulerian carrier is authoritative inside the core.
    if (!core_faces_.empty() && core_faces_[face] != 0) {
      continue;
    }
    const auto& deposit = deposits[face];
    if (deposit.deposited_mass > 0) {
      old_faces_[face].carrier_velocity = deposit.deposited_pure_momentum / deposit.deposited_mass;
      old_faces_[face].validity = deposit.deposited_mass > options.minimum_face_mass ? LiquidFaceValidity3D::kValid
                                                                                     : LiquidFaceValidity3D::kLowMass;
    }
    if (old_faces_[face].validity == LiquidFaceValidity3D::kMissing
        || !std::isfinite(old_faces_[face].carrier_velocity)) {
      return absl::FailedPreconditionError("open graded subface has no finite carrier");
    }
  }
  return absl::OkStatus();
}

// Backtrace the accepted leaf pressure using a bounded one-ring upwind
// interpolant. Each coarse side averages its donors by their owned subface
// areas. Sealed sides and missing donors keep the receiving leaf's pressure.
absl::Status AdaptiveLiquidStep3D::AdvectPressureHistory(const LiquidState3D& state, double duration) {
  std::ranges::fill(pressure_history_, PressureHistory3D{});
  const auto leaves = topology_.cells();
  const auto accumulate_velocity = [&](std::size_t cell, std::size_t slot, FaceNormalAxis axis) {
    const auto& carrier = state.adaptive_faces[slot];
    if (carrier.validity != LiquidFaceValidity3D::kValid && carrier.validity != LiquidFaceValidity3D::kPredicted) {
      return;
    }
    const double width = leaves[cell].cell_width;
    const double area_weight = subfaces_[slot].open_area_m2 / (2.0 * width * width);
    AddNormalComponent(pressure_history_[cell].centered_velocity, axis, area_weight * carrier.carrier_velocity);
  };
  const auto interior_faces = topology_.interior_faces();
  for (std::size_t face = 0; face < interior_faces.size(); ++face) {
    const auto& link = interior_faces[face];
    const auto slot = interior_slots_[face];
    accumulate_velocity(link.tail_cell_ordinal, slot, link.face_index.axis);
    accumulate_velocity(link.head_cell_ordinal, slot, link.face_index.axis);
  }
  for (const auto& link : exterior_faces_) {
    accumulate_velocity(link.cell_ordinal, link.face_slot, link.face_index.axis);
  }
  for (const auto& link : interior_faces) {
    const auto accumulate_donor = [&](std::size_t cell, std::size_t donor, double outward_sign) -> absl::Status {
      auto& history = pressure_history_[cell];
      const double outward_velocity = outward_sign * NormalComponent(history.centered_velocity, link.face_index.axis);
      if (outward_velocity >= 0.0) {
        return absl::OkStatus();
      }
      const double width = leaves[cell].cell_width;
      const double side_fraction = link.face_area / (width * width);
      const double donor_weight = side_fraction * (-duration * outward_velocity / link.cell_center_distance);
      history.donor_weight += donor_weight;
      if (!std::isfinite(history.donor_weight) || history.donor_weight > 1.0) {
        return absl::AbortedError("graded pressure backtrace exceeds the adjacent leaf support");
      }
      if (donor_weight > 0.0) {
        history.donor_pressure = std::lerp(
            history.donor_pressure,
            state.adaptive_cells[donor].previous_pressure,
            donor_weight / history.donor_weight
        );
      }
      return absl::OkStatus();
    };
    const auto tail = accumulate_donor(link.tail_cell_ordinal, link.head_cell_ordinal, 1.0);
    if (!tail.ok()) {
      return tail;
    }
    const auto head = accumulate_donor(link.head_cell_ordinal, link.tail_cell_ordinal, -1.0);
    if (!head.ok()) {
      return head;
    }
  }
  for (std::size_t cell = 0; cell < leaves.size(); ++cell) {
    const auto& history = pressure_history_[cell];
    pressure_cells_[cell].previous_pressure =
        std::lerp(state.adaptive_cells[cell].previous_pressure, history.donor_pressure, history.donor_weight);
  }
  return absl::OkStatus();
}

absl::Status AdaptiveLiquidStep3D::Project(
    LiquidState3D& trial,
    double duration,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& options,
    const LiquidSubstepBackend3D& backend,
    MacPressureWorkspace3D& workspace,
    LiquidAcceptedStepMeasurement3D& measurement
) {
  std::optional<LiquidEos3D> eos;
  if (scene.eos) {
    eos = LiquidEos3D{scene.liquid_density, scene.eos->reference_sound_speed, scene.eos->law};
    const auto advected = AdvectPressureHistory(trial, duration);
    if (!advected.ok()) {
      return advected;
    }
  }
  if (!unit_strain_rows_.empty()) {
    strain_rows_.clear();
  }
  for (std::size_t cell = 0; cell < pressure_cells_.size(); ++cell) {
    auto& pressure_cell = pressure_cells_[cell];
    if (!eos) {
      pressure_cell.previous_pressure = trial.adaptive_cells[cell].previous_pressure;
    }
    if (IsEulerian(trial.adaptive_cells[cell])) {
      pressure_cell.fluid_volume = trial.adaptive_cells[cell].inventory.current_volume_m3;
    }
    if (predictor_density_.empty()) {
      continue;
    }
    const auto& inventory = trial.adaptive_cells[cell].inventory;
    double density = scene.liquid_density;
    double viscosity = scene.dynamic_viscosity;
    if (inventory.constituent_count != 0) {
      std::array<LiquidConstituentFraction3D, kMaximumLiquidParticleConstituents3D> fractions{};
      const auto mixture = MixLiquidConstituentVolumes3D(
          inventory.phase_id,
          scene.constituent_materials,
          std::span(inventory.constituents).first(inventory.constituent_count),
          fractions
      );
      if (!mixture.ok()) {
        return mixture.status();
      }
      density = mixture->density_kg_per_m3;
      if (!unit_strain_rows_.empty()) {
        viscosity = 0.0;
        for (const auto& fraction : std::span(fractions).first(mixture->constituent_count)) {
          const auto material = std::ranges::lower_bound(
              scene.constituent_materials,
              fraction.substance_id,
              {},
              &LiquidConstituentMaterial3D::substance_id
          );
          viscosity = std::fma(
              fraction.volume_fraction,
              material->dynamic_viscosity_pa_s.value_or(scene.dynamic_viscosity),
              viscosity
          );
        }
      }
    }
    if (!unit_strain_rows_.empty() && viscosity > 0.0) {
      // The accepted leaf inventory supplies the current-volume mixture. Freeze its weights for this solve.
      const auto begin = leaf_strain_row_offsets_[cell];
      const auto end = leaf_strain_row_offsets_[cell + 1];
      for (auto ordinal = begin; ordinal < end; ++ordinal) {
        auto row = unit_strain_rows_[ordinal];
        row.quadrature_weight *= viscosity;
        if (!std::isnormal(row.quadrature_weight)) {
          return absl::OutOfRangeError("graded constituent viscosity quadrature is unrepresentable");
        }
        strain_rows_.push_back(row);
      }
    }
    reference_density_[cell] = density;
    predictor_density_[cell] = density;
    if (eos) {
      eos->reference_density = density;
      const auto predictor = EvaluateLiquidEosAtPressure3D(*eos, pressure_cell.previous_pressure);
      if (!predictor.ok()) {
        return predictor.status();
      }
      pressure_cell.bulk_modulus = predictor->bulk_modulus;
      predictor_density_[cell] = predictor->density;
    }
  }
  for (std::size_t face = 0; face < subfaces_.size(); ++face) {
    if (subfaces_[face].open_area_m2 > 0) {
      subfaces_[face].predicted_normal_velocity_m_per_s = old_faces_[face].carrier_velocity
          + duration * NormalComponent(scene.world_acceleration, transfer_.faces()[face].axis);
    }
  }
  // The pressure unknowns are at leaf centers. Include the constant body
  // potential across the tangential center offset of each coarse/fine edge.
  const auto interior_faces = topology_.interior_faces();
  const auto leaves = topology_.cells();
  for (std::size_t face = 0; face < interior_faces.size(); ++face) {
    const auto& interior = interior_faces[face];
    const auto& tail = leaves[interior.tail_cell_ordinal];
    const auto& head = leaves[interior.head_cell_ordinal];
    const auto tail_corner = tail.key.lower_corner;
    const auto head_corner = head.key.lower_corner;
    const double half_width_delta = (head.cell_width - tail.cell_width) / 2;
    const double lattice_width = topology_.finest_cell_width();
    const Vec3 displacement{
        (double(head_corner.x) - tail_corner.x) * lattice_width + half_width_delta,
        (double(head_corner.y) - tail_corner.y) * lattice_width + half_width_delta,
        (double(head_corner.z) - tail_corner.z) * lattice_width + half_width_delta,
    };
    const auto axis = interior.face_index.axis;
    const auto acceleration = scene.world_acceleration;
    const double potential_difference =
        acceleration.x * displacement.x + acceleration.y * displacement.y + acceleration.z * displacement.z;
    const auto slot = interior_slots_[face];
    if (!predictor_density_.empty()) {
      const double tail_fraction = tail.cell_width / (tail.cell_width + head.cell_width);
      const double face_density = std::lerp(
          predictor_density_[interior.head_cell_ordinal],
          predictor_density_[interior.tail_cell_ordinal],
          tail_fraction
      );
      subfaces_[slot].inverse_density_m3_per_kg = 1.0 / face_density;
    }
    subfaces_[slot].predicted_normal_velocity_m_per_s =
        old_faces_[slot].carrier_velocity + duration * potential_difference / NormalComponent(displacement, axis);
  }
  if (!predictor_density_.empty()) {
    for (const auto& exterior : exterior_faces_) {
      auto& subface = subfaces_[exterior.face_slot];
      if (subface.open_area_m2 > 0) {
        subface.inverse_density_m3_per_kg = 1.0 / predictor_density_[exterior.cell_ordinal];
      }
    }
  }
  if (scene.surface_tension > 0.0) {
    double capillary_bound = std::numeric_limits<double>::infinity();
    for (const auto& exterior : exterior_faces_) {
      const auto& subface = subfaces_[exterior.face_slot];
      if (subface.open_area_m2 == 0.0) {
        continue;
      }
      const boundary_internal::GeometryInput input{
          .mode = boundary_internal::GeometryMode::kCapillary,
          .geometry = {trial.geometry.world_origin, leaves[exterior.cell_ordinal].cell_width},
          .liquid_density =
              predictor_density_.empty() ? scene.liquid_density : predictor_density_[exterior.cell_ordinal],
          .surface_tension = scene.surface_tension,
      };
      const auto limit = boundary_internal::ComputeGeometry(input);
      if (!limit.ok()) {
        return absl::OutOfRangeError("graded surface capillary bound is unrepresentable");
      }
      capillary_bound = std::min(capillary_bound, (*limit).capillary_step_bound);
    }
    if (std::isfinite(capillary_bound)) {
      if (duration > capillary_bound) {
        return absl::AbortedError("graded step exceeds the exposed-leaf capillary bound");
      }
      // The authored horizontal surface is planar, so its pressure lift is zero.
      measurement.step_start_capillary_bound_seconds = capillary_bound;
    }
  }
  const auto graph =
      pressure_internal::PrepareAdaptiveMacPressureSystem3D(topology_, duration, pressure_cells_, subfaces_);
  if (!graph.ok()) {
    return graph.status();
  }
  const auto publish = [&](const auto& projection) -> absl::Status {
    if (projection.cell_pressures.size() != pressure_cells_.size()
        || projection.corrected_face_velocities.size() != subfaces_.size()) {
      return absl::Status(absl::StatusCode::kInternal, {});
    }
    if (eos) {
      for (std::size_t cell = 0; cell < projection.cell_pressures.size(); ++cell) {
        eos->reference_density = reference_density_[cell];
        const auto accepted = EvaluateLiquidEosAtPressure3D(*eos, projection.cell_pressures[cell]);
        if (!accepted.ok()) {
          return accepted.status();
        }
      }
    }
    if (update_volume_ratio_) {
      std::ranges::fill(normal_derivatives_, Vec3{});
      for (const auto& link : graph->links()) {
        const double integrated_flux =
            link.open_area * projection.corrected_face_velocities[link.face_slot] + link.prescribed_wall_flux;
        const auto axis = transfer_.faces()[link.face_slot].axis;
        pressure_internal::arithmetic::ForEachIncidentCell(link, [&](std::size_t cell, double sign) {
          const double derivative = sign * integrated_flux / pressure_cells_[cell].fluid_volume;
          AddNormalComponent(normal_derivatives_[cell], axis, derivative);
        });
      }
    }
    for (std::size_t cell = 0; cell < pressure_cells_.size(); ++cell) {
      trial.adaptive_cells[cell].previous_pressure = projection.cell_pressures[cell];
    }
    for (std::size_t face = 0; face < subfaces_.size(); ++face) {
      trial.adaptive_faces[face] = old_faces_[face];
      trial.adaptive_faces[face].carrier_velocity = projection.corrected_face_velocities[face];
    }
    measurement.pressure = projection.diagnostics;
    return absl::OkStatus();
  };
  if (!strain_rows_.empty()) {
    const auto projection = backend.project_adaptive_viscous != nullptr
        ? backend.project_adaptive_viscous(
              backend.context,
              *graph,
              duration,
              strain_rows_,
              options.pressure_solver_options
          )
        : pressure_internal::ProjectPreparedAdaptiveMacViscousPressure3D(
              *graph,
              duration,
              strain_rows_,
              options.pressure_solver_options
          );
    if (!projection.ok()) {
      return projection.status();
    }
    struct ProjectionView {
      std::span<const double> cell_pressures;
      std::span<const double> corrected_face_velocities;
      MacPressureDiagnostics3D diagnostics;
    };
    const auto status = publish(ProjectionView{
        .cell_pressures = projection->cell_pressures,
        .corrected_face_velocities = projection->face_velocities,
        .diagnostics =
            {
                .iteration_count = projection->pressure_iterations,
                .true_residual_l2 = projection->constraint_residual_l2,
                .max_volume_scaled_residual = projection->max_cell_constraint_defect,
                .max_cell_constraint_defect = projection->max_cell_constraint_defect,
            },
    });
    if (!status.ok()) {
      return status;
    }
    measurement.pressure_rhs_l2_m3_per_s = projection->initial_constraint_residual_l2;
    measurement.viscous_max_momentum_residual_kg_m_per_s = projection->max_momentum_residual;
  } else if (backend.project_adaptive != nullptr) {
    const auto result = backend.project_adaptive(backend.context, *graph, options.pressure_solver_options);
    if (!result.ok()) {
      return result.status();
    }
    const auto status = publish(*result);
    if (!status.ok()) {
      return status;
    }
  } else {
    const auto result = backend.project_adaptive_in_arena != nullptr
        ? backend.project_adaptive_in_arena(backend.context, *graph, options.pressure_solver_options, workspace)
        : workspace.ProjectPrepared(*graph, options.pressure_solver_options);
    if (!result.ok()) {
      return result.status();
    }
    const auto status = publish(*result);
    if (!status.ok()) {
      return status;
    }
  }
  if (strain_rows_.empty()) {
    for (const double rhs : graph->integrated_rhs()) {
      measurement.pressure_rhs_l2_m3_per_s = std::hypot(measurement.pressure_rhs_l2_m3_per_s, rhs);
    }
  }
  for (const auto& cell : pressure_cells_) {
    measurement.step_start_pressure_cell_volume_m3 += cell.fluid_volume;
  }
  return absl::OkStatus();
}

absl::Status
AdaptiveLiquidStep3D::Gather(const LiquidState3D& trial, const PreparedAdaptiveQuadraticTransfer3D* quadratic) {
  auto selections =
      quadratic != nullptr ? quadratic->selections() : std::span<const PreparedAdaptiveQuadraticParticle3D>{};
  for (std::size_t ordinal = 0; ordinal < trial.particles.size(); ++ordinal) {
    const auto& particle = trial.particles[ordinal];
    if (!selections.empty() && ordinal == selections.front().particle_ordinal) {
      Vec3 old_pic_velocity{};
      const auto gathered = GradedQuadraticMacTransfer3D::GatherBound(
          selections.front().stencil,
          particle.side_tag,
          old_faces_,
          trial.adaptive_faces.values(),
          old_pic_velocity
      );
      if (!gathered.ok()) {
        return gathered.status();
      }
      const Vec3 residual = Difference(particle.world_velocity, old_pic_velocity);
      if (!std::isfinite(residual.x) || !std::isfinite(residual.y) || !std::isfinite(residual.z)) {
        return absl::OutOfRangeError("quadratic old PIC residual is nonfinite");
      }
      gathered_[ordinal] = {
          .new_pic_velocity = gathered->new_pic_velocity,
          .grid_velocity_increment = gathered->grid_velocity_increment,
          .particle_minus_old_pic_velocity = residual,
          .fitted_velocity_slopes = gathered->fitted_velocity_slopes,
          // The implicit incompressible path holds particle volume fixed.
          // The spline derivative need not satisfy the leaf flux constraint.
          .velocity_gradient = {},
          .affine_enabled = true,
      };
      distances_[ordinal] = particle.world_position.y - surface_height_;
      selections = selections.subspan(1);
      continue;
    }
    const auto before = transfer_.Gather(particle, old_faces_);
    if (!before.ok()) {
      return before.status();
    }
    const auto after = transfer_.Gather(particle, trial.adaptive_faces.values());
    if (!after.ok()) {
      return after.status();
    }
    gathered_[ordinal] = {
        .new_pic_velocity = after->velocity,
        .grid_velocity_increment = Difference(after->velocity, before->velocity),
        .particle_minus_old_pic_velocity = Difference(particle.world_velocity, before->velocity),
        .fitted_velocity_slopes = after->velocity_slopes,
        .velocity_gradient = {},
        .affine_enabled = before->affine_enabled && after->affine_enabled,
        .old_pic_fallback = !before->affine_enabled,
        .new_pic_fallback = !after->affine_enabled,
    };
    if (update_volume_ratio_) {
      const auto leaf = transfer_.ParticleLeafOrdinal(particle.world_position);
      if (!leaf.ok()) {
        return leaf.status();
      }
      const auto derivatives = normal_derivatives_[*leaf];
      gathered_[ordinal].velocity_gradient = {
          .x_row = {derivatives.x, 0.0, 0.0},
          .y_row = {0.0, derivatives.y, 0.0},
          .z_row = {0.0, 0.0, derivatives.z},
      };
    }
    distances_[ordinal] = particle.world_position.y - surface_height_;
  }
  return absl::OkStatus();
}

absl::Status AdaptiveLiquidStep3D::Update(
    LiquidState3D& trial,
    double duration,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& options,
    const LiquidSubstepBackend3D& backend
) {
  const auto updated = backend.update_particles(
      backend.context,
      {
          .gathered = gathered_,
          .liquid_distances = distances_,
          .obstacle = physical_trial_ ? physical_trial_->start_wall : scene.obstacle,
          .cell_width = trial.geometry.cell_width,
          .step_duration = duration,
          .particle_energy_blend = options.particle_energy_blend,
          .friction_coefficient = scene.friction_coefficient,
          .penetration_tolerance = scene.penetration_tolerance,
      },
      trial.particles.values()
  );
  if (!updated.ok()) {
    return updated;
  }
  if (physical_trial_) {
    for (const auto& particle : trial.particles.values()) {
      const double wall_distance = SolidDistance(physical_trial_->end_wall, particle.world_position);
      if (!std::isfinite(wall_distance) || wall_distance < 0 || particle.world_position.y >= surface_height_) {
        return absl::AbortedError("Graded physical trial moved a particle outside the liquid");
      }
    }
  }
  const auto material = AccumulateInventories(trial);
  if (!material.ok()) {
    return material;
  }
  if (physical_trial_) {
    for (std::size_t cell = 0; cell < inventories_.size(); ++cell) {
      if (inventories_[cell].current_volume_m3 > physical_trial_->end_fluid_volumes_m3[cell]) {
        return absl::AbortedError("Graded physical trial particle inventory exceeds its final wet volume");
      }
    }
  }
  if (core_transport_) {
    for (std::size_t face = 0; face < interior_slots_.size(); ++face) {
      const auto slot = interior_slots_[face];
      core_fluxes_[face] = subfaces_[slot].open_area_m2 * trial.adaptive_faces[slot].carrier_velocity;
    }
    const auto core = core_transport_->Prepare(topology_, trial.adaptive_cells.values(), core_fluxes_, duration);
    if (!core.ok()) {
      return core.status();
    }
    for (std::size_t cell = 0; cell < inventories_.size(); ++cell) {
      if (IsEulerian(trial.adaptive_cells[cell])) {
        const auto& updated_inventory = (*core)[cell].inventory;
        if (updated_inventory.current_volume_m3 <= 0) {
          return absl::AbortedError("Eulerian core trial emptied a pressure leaf");
        }
        inventories_[cell] = updated_inventory;
      }
    }
  }
  for (std::size_t cell = 0; cell < inventories_.size(); ++cell) {
    trial.adaptive_cells[cell].inventory = inventories_[cell];
  }
  return absl::OkStatus();
}

absl::Status AdaptiveLiquidStep3D::AccumulateInventories(const LiquidState3D& state) {
  std::ranges::fill(inventories_, LiquidRemapInventory3D{});
  std::ranges::fill(constituent_sums_, liquid_remap_internal::ConstituentSums{});
  for (const auto& particle : state.particles.values()) {
    const auto leaf = transfer_.ParticleLeafOrdinal(particle.world_position);
    if (!leaf.ok()) {
      return absl::Status(absl::StatusCode::kAborted, {});
    }
    if (IsEulerian(state.adaptive_cells[*leaf])) {
      return absl::AbortedError("particle would duplicate Eulerian core ownership");
    }
    auto& inventory = inventories_[*leaf];
    inventory.mass_kg += particle.mass;
    inventory.reference_volume_m3 += particle.reference_volume;
    const double particle_volume = particle.reference_volume * particle.volume_ratio;
    inventory.current_volume_m3 += particle_volume;
    if (particle.constituent_count != 0) {
      inventory.phase_id = particle.phase_id;
      for (const auto& constituent : std::span(particle.constituents).first(particle.constituent_count)) {
        const auto status =
            constituent_sums_[*leaf].Add(constituent.substance_id, constituent.volume_fraction * particle_volume);
        if (!status.ok()) {
          return status;
        }
      }
    }
    inventory.momentum_kg_m_per_s.x += particle.mass * particle.world_velocity.x;
    inventory.momentum_kg_m_per_s.y += particle.mass * particle.world_velocity.y;
    inventory.momentum_kg_m_per_s.z += particle.mass * particle.world_velocity.z;
  }
  for (std::size_t cell = 0; cell < inventories_.size(); ++cell) {
    if (!constituent_sums_.empty()) {
      const auto status = constituent_sums_[cell].Store(inventories_[cell]);
      if (!status.ok()) {
        return status;
      }
    }
    if (inventories_[cell].current_volume_m3 > topology_.cells()[cell].cell_volume) {
      return absl::Status(absl::StatusCode::kAborted, {});
    }
  }
  return absl::OkStatus();
}

}  // namespace fluidity::liquid_step_internal
