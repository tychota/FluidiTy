#include "liquid_adaptive_step_3d.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "absl/status/status.h"

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

Vec3 Difference(Vec3 left, Vec3 right) {
  return {left.x - right.x, left.y - right.y, left.z - right.z};
}

}  // namespace

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
  if (plane == nullptr || tank == nullptr || tank->piston_velocity_x != 0 || scene.surface_tension != 0
      || !scene.sources.empty() || !std::isinf(scene.bulk_modulus) || scene.target_divergence != 0
      || options.ocean_boundary.has_value() || options.motion_sample_intervals != 0
      || options.capacity_replay_reservations != LiquidStateCapacities3D{}) {
    return absl::UnimplementedError("graded Simulate currently requires a stationary conforming planar tank");
  }
  if (initial.particles.size() == 0) {
    return absl::InvalidArgumentError("graded Simulate requires particle-owned material");
  }
  std::vector<AdaptiveMacCellKey3D> keys;
  keys.reserve(initial.adaptive_cells.size());
  for (const auto& cell : initial.adaptive_cells.values()) {
    if (cell.occupancy != LiquidCellOccupancy3D::kLiquid || cell.inventory.enthalpy_j != 0) {
      return absl::UnimplementedError("graded tank requires single-liquid particle-owned leaves");
    }
    keys.push_back(cell.key);
  }
  auto topology = AdaptiveMacTopology3D::Create(keys, initial.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  auto transfer =
      GradedMacTransfer3D::Create(*topology, initial.geometry.world_origin, Vec3{tank->high_x, 0, tank->high_z});
  if (!transfer.ok()) {
    return transfer.status();
  }
  AdaptiveLiquidStep3D result(std::move(*topology), std::move(*transfer));
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(result.topology_);
  if (!ownership.ok()) {
    return ownership.status();
  }
  result.interior_slots_ = ownership->interior_face_slots;
  result.surface_height_ = plane->world_surface_height_y;
  result.subfaces_.resize(initial.adaptive_faces.size());
  const auto leaves = result.topology_.cells();
  result.pressure_cells_.resize(leaves.size());
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
    if (world_lower.x < tank->start_low_x || world_upper.x > tank->high_x || world_lower.y < tank->floor_y
        || world_upper.y > result.surface_height_ || world_lower.z < tank->low_z || world_upper.z > tank->high_z
        || world_lower.x >= world_upper.x || world_lower.y >= world_upper.y || world_lower.z >= world_upper.z) {
      return absl::InvalidArgumentError("graded leaf crosses a tank plane or has unrepresentable bounds");
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
    result.subfaces_[ownership->interior_face_slots[face]] = {
        .open_area_m2 = result.topology_.interior_faces()[face].face_area,
        .inverse_density_m3_per_kg = inverse_density,
    };
  }
  for (const auto& face : ownership->exterior_faces) {
    const auto axis = face.face_index.axis;
    const double coordinate = NormalComponent(result.transfer_.faces()[face.face_slot].world_anchor, axis);
    if (IsWall(*tank, face, coordinate)) {
      continue;
    }
    if (axis != FaceNormalAxis::kY || !face.cell_is_tail || coordinate != result.surface_height_) {
      return absl::InvalidArgumentError("graded tank has an exterior subface away from a wall or the free surface");
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
    auto particle_owned = initial.adaptive_cells[cell];
    particle_owned.inventory = result.inventories_[cell];
    if (particle_owned != initial.adaptive_cells[cell]) {
      return absl::InvalidArgumentError("graded tank inventory must match its particle-owned material");
    }
  }
  return result;
}

absl::Status AdaptiveLiquidStep3D::Deposit(const LiquidState3D& trial, const LiquidStepOptions3D& options) {
  const auto deposits = transfer_.Deposit(trial.particles.values(), trial.adaptive_faces.values());
  if (!deposits.ok()) {
    return deposits.status();
  }
  for (std::size_t face = 0; face < old_faces_.size(); ++face) {
    old_faces_[face] = trial.adaptive_faces[face];
    if (subfaces_[face].open_area_m2 == 0) {
      old_faces_[face].carrier_velocity = 0;
      old_faces_[face].validity = LiquidFaceValidity3D::kValid;
      continue;
    }
    const auto& deposit = (*deposits)[face];
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

absl::Status AdaptiveLiquidStep3D::Project(
    LiquidState3D& trial,
    double duration,
    const LiquidSceneInputs3D& scene,
    const LiquidStepOptions3D& options,
    const LiquidSubstepBackend3D& backend,
    MacPressureWorkspace3D& workspace,
    LiquidAcceptedStepMeasurement3D& measurement
) {
  for (std::size_t cell = 0; cell < pressure_cells_.size(); ++cell) {
    pressure_cells_[cell].previous_pressure = trial.adaptive_cells[cell].previous_pressure;
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
    subfaces_[slot].predicted_normal_velocity_m_per_s =
        old_faces_[slot].carrier_velocity + duration * potential_difference / NormalComponent(displacement, axis);
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
  if (backend.project_adaptive != nullptr) {
    const auto result = backend.project_adaptive(backend.context, *graph, options.pressure_solver_options);
    if (!result.ok()) {
      return result.status();
    }
    const auto status = publish(*result);
    if (!status.ok()) {
      return status;
    }
  } else {
    const auto result = workspace.ProjectPrepared(*graph, options.pressure_solver_options);
    if (!result.ok()) {
      return result.status();
    }
    const auto status = publish(*result);
    if (!status.ok()) {
      return status;
    }
  }
  for (const double rhs : graph->integrated_rhs()) {
    measurement.pressure_rhs_l2_m3_per_s = std::hypot(measurement.pressure_rhs_l2_m3_per_s, rhs);
  }
  for (const auto& cell : pressure_cells_) {
    measurement.step_start_pressure_cell_volume_m3 += cell.fluid_volume;
  }
  return absl::OkStatus();
}

absl::Status AdaptiveLiquidStep3D::Gather(const LiquidState3D& trial) {
  for (std::size_t ordinal = 0; ordinal < trial.particles.size(); ++ordinal) {
    const auto& particle = trial.particles[ordinal];
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
          .obstacle = scene.obstacle,
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
  const auto material = AccumulateInventories(trial);
  if (!material.ok()) {
    return material;
  }
  for (std::size_t cell = 0; cell < inventories_.size(); ++cell) {
    trial.adaptive_cells[cell].inventory = inventories_[cell];
  }
  return absl::OkStatus();
}

absl::Status AdaptiveLiquidStep3D::AccumulateInventories(const LiquidState3D& state) {
  std::ranges::fill(inventories_, LiquidRemapInventory3D{});
  for (const auto& particle : state.particles.values()) {
    const auto leaf = transfer_.ParticleLeafOrdinal(particle.world_position);
    if (!leaf.ok()) {
      return absl::Status(absl::StatusCode::kAborted, {});
    }
    auto& inventory = inventories_[*leaf];
    inventory.mass_kg += particle.mass;
    inventory.reference_volume_m3 += particle.reference_volume;
    inventory.current_volume_m3 += particle.reference_volume * particle.volume_ratio;
    inventory.momentum_kg_m_per_s.x += particle.mass * particle.world_velocity.x;
    inventory.momentum_kg_m_per_s.y += particle.mass * particle.world_velocity.y;
    inventory.momentum_kg_m_per_s.z += particle.mass * particle.world_velocity.z;
  }
  for (std::size_t cell = 0; cell < inventories_.size(); ++cell) {
    if (inventories_[cell].current_volume_m3 > topology_.cells()[cell].cell_volume) {
      return absl::Status(absl::StatusCode::kAborted, {});
    }
  }
  return absl::OkStatus();
}

}  // namespace fluidity::liquid_step_internal
