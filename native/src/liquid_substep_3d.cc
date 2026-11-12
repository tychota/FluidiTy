#include "fluidity/host/liquid_substep_3d.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numbers>
#include <numeric>
#include <optional>
#include <tuple>

#include "absl/cleanup/cleanup.h"
#include "dry_solid_step_3d_internal.h"
#include "fluidity/host/plastic_solid_boundary_3d.h"
#include "fluidity/host/small_solve.h"
#include "liquid_adaptive_family_candidate_3d.h"
#include "liquid_adaptive_step_3d.h"
#include "liquid_boundary_backend.h"
#include "liquid_substep_3d_internal.h"
namespace fluidity {
namespace {
#define STEP_TRY(expression) \
  do { \
    const auto status = (expression); \
    if (!status.ok()) { \
      return status; \
    } \
  } while (false)
absl::Status Error(absl::StatusCode code) {
  return absl::Status(code, {});
}
auto CellKey(CellIndex3D index) {
  return std::tuple(index.z, index.y, index.x);
}

struct RoutedPressureImpulse3D {
  Vec3 angular_impulse_change{};
  std::size_t face_count = 0;
};

// Multi-source breadth-first search over the open pressure graph. Each
// component and axis can only receive impulses on its own supported faces.
absl::StatusOr<RoutedPressureImpulse3D> RouteUnsupportedPressureImpulse(
    SparseMacGridView3D grid,
    const LiquidPressureProblemView3D& problem,
    std::span<const FrozenLiquidFace3D> frozen_faces,
    std::span<const double> face_masses,
    std::span<double> impulses
) {
  constexpr std::size_t kCellCapacity = 4096;
  constexpr std::int32_t kMissing = -1;
  if (problem.cells.size() > kCellCapacity || problem.faces.size() != frozen_faces.size()) {
    return Error(absl::StatusCode::kInternal);
  }
  constexpr std::array kCoordinates{&CellIndex3D::x, &CellIndex3D::y, &CellIndex3D::z};
  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  const auto find_cell = [&](CellIndex3D index) -> std::int32_t {
    const auto key = CellKey(index);
    const auto found = std::ranges::lower_bound(problem.cells, key, {}, [](const auto& cell) {
      return CellKey(cell.cell_index);
    });
    return found == problem.cells.end() || CellKey(found->cell_index) != key
        ? kMissing
        : static_cast<std::int32_t>(found - problem.cells.begin());
  };
  const auto incident_cells = [&](FaceIndex3D face) {
    const auto index = face.lattice_index;
    const CellIndex3D head{index.x, index.y, index.z};
    auto tail = head;
    auto& coordinate = tail.*kCoordinates[static_cast<std::size_t>(face.axis)];
    if (coordinate == std::numeric_limits<std::int32_t>::min()) {
      return std::array{kMissing, find_cell(head)};
    }
    --coordinate;
    return std::array{find_cell(tail), find_cell(head)};
  };

  std::array<std::array<std::int32_t, 6>, kCellCapacity> neighbors;
  for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
    neighbors[cell].fill(kMissing);
  }
  for (const auto& face : problem.faces) {
    if (face.open_area == 0.0 || !std::holds_alternative<InternalMacPressureLink3D>(face.pressure_link)) {
      continue;
    }
    const auto [tail, head] = incident_cells(face.face_index);
    if (tail == kMissing || head == kMissing) {
      return Error(absl::StatusCode::kInternal);
    }
    const auto axis = static_cast<std::size_t>(face.face_index.axis);
    neighbors[tail][2 * axis + 1] = head;
    neighbors[head][2 * axis] = tail;
  }

  struct Route {
    std::size_t face_ordinal;
    std::size_t distance;
  };
  constexpr auto kNoFace = std::numeric_limits<std::size_t>::max();
  std::array<Route, kCellCapacity> routes;
  std::array<std::int32_t, kCellCapacity> queue;
  RoutedPressureImpulse3D routed;
  for (std::size_t axis = 0; axis < kComponents.size(); ++axis) {
    std::ranges::fill(std::span(routes).first(problem.cells.size()), Route{kNoFace, kNoFace});
    std::size_t queue_end = 0;
    // Pressure faces are in owned-slot order. Together with the ordered
    // neighbor walk this fixes equal-distance donor selection.
    for (std::size_t ordinal = 0; ordinal < problem.faces.size(); ++ordinal) {
      const auto& face = problem.faces[ordinal];
      if (static_cast<std::size_t>(face.face_index.axis) != axis || face.open_area == 0.0
          || face_masses[frozen_faces[ordinal].face_slot] == 0.0) {
        continue;
      }
      for (const auto cell : incident_cells(face.face_index)) {
        if (cell != kMissing && routes[cell].face_ordinal == kNoFace) {
          routes[cell] = {ordinal, 0};
          queue[queue_end++] = cell;
        }
      }
    }
    for (std::size_t cursor = 0; cursor < queue_end; ++cursor) {
      const auto cell = queue[cursor];
      for (const auto neighbor : neighbors[cell]) {
        if (neighbor != kMissing && routes[neighbor].face_ordinal == kNoFace) {
          routes[neighbor] = {routes[cell].face_ordinal, routes[cell].distance + 1};
          queue[queue_end++] = neighbor;
        }
      }
    }
    for (std::size_t ordinal = 0; ordinal < problem.faces.size(); ++ordinal) {
      const auto& face = problem.faces[ordinal];
      const auto slot = frozen_faces[ordinal].face_slot;
      if (static_cast<std::size_t>(face.face_index.axis) != axis || face_masses[slot] != 0.0 || impulses[slot] == 0.0) {
        continue;
      }
      Route target{kNoFace, kNoFace};
      for (const auto cell : incident_cells(face.face_index)) {
        if (cell != kMissing
            && std::tie(routes[cell].distance, routes[cell].face_ordinal)
                < std::tie(target.distance, target.face_ordinal)) {
          target = routes[cell];
        }
      }
      if (target.face_ordinal == kNoFace) {
        return Error(absl::StatusCode::kFailedPrecondition);
      }
      const auto target_slot = frozen_faces[target.face_ordinal].face_slot;
      const auto source = face.face_index.lattice_index;
      const auto target_index = problem.faces[target.face_ordinal].face_index.lattice_index;
      const double width = grid.geometry().cell_width;
      const Vec3 offset{
          width * (static_cast<double>(target_index.x) - source.x),
          width * (static_cast<double>(target_index.y) - source.y),
          width * (static_cast<double>(target_index.z) - source.z),
      };
      Vec3 impulse{};
      impulse.*kComponents[axis] = impulses[slot];
      routed.angular_impulse_change.x += offset.y * impulse.z - offset.z * impulse.y;
      routed.angular_impulse_change.y += offset.z * impulse.x - offset.x * impulse.z;
      routed.angular_impulse_change.z += offset.x * impulse.y - offset.y * impulse.x;
      impulses[target_slot] += impulses[slot];
      impulses[slot] = 0.0;
      ++routed.face_count;
      if (!std::isfinite(impulses[target_slot]) || !liquid_step_internal::Finite(routed.angular_impulse_change)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
    }
  }
  return routed;
}

// Mask anchors inside the box and use the nearest box face to complete those
// velocity samples. Open anchors continue to use the tank history.
absl::Status ApplyBoxFaceSamples(
    SparseMacGridView3D grid,
    const TranslatingContactBox3D& box,
    std::span<LiquidBoundaryFaceTag3D> tags,
    std::span<LiquidPrescribedWallSample3D> walls
) {
  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  const auto geometry = grid.geometry();
  for (std::size_t slot = 0; slot < grid.face_count(); ++slot) {
    const auto face = grid.FaceIndex(slot);
    if (!face.ok()) {
      return face.status();
    }
    const auto anchor = boundary_internal::TankFaceAnchorInCellWidths(*face);
    bool inside = true;
    double nearest = std::numeric_limits<double>::infinity();
    Vec3 outward{};
    for (std::size_t axis = 0; axis < kComponents.size(); ++axis) {
      const auto component = kComponents[axis];
      const boundary_internal::TankLatticeAxis lattice{
          geometry.world_origin.*component,
          anchor.*component,
          geometry.cell_width
      };
      const auto low = boundary_internal::TankPlaneOffset(box.world_low.*component, lattice).Bounds();
      const auto high = boundary_internal::TankPlaneOffset(box.world_high.*component, lattice).Bounds();
      const auto low_sign = low.sign();
      const auto high_sign = high.sign();
      if (!low_sign.has_value() || !high_sign.has_value()) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      if (*low_sign > 0 || *high_sign < 0) {
        inside = false;
        break;
      }
      if (-low.estimate < nearest) {
        nearest = -low.estimate;
        outward = {};
        outward.*component = -1.0;
      }
      if (high.estimate < nearest) {
        nearest = high.estimate;
        outward = {};
        outward.*component = 1.0;
      }
    }
    if (inside) {
      if (!tags.empty()) {
        tags[slot].structurally_eligible = false;
      }
      if (!walls.empty()) {
        walls[slot] = {true, 0, outward, box.wall_velocity};
      }
    }
  }
  return absl::OkStatus();
}

Vec3 RigidSurfaceVelocity(const RigidBody3D& body, Vec3 center_relative_position, Vec3 linear_velocity) {
  const auto angular = body.angular_velocity;
  const auto arm = center_relative_position;
  return {
      std::fma(angular.y, arm.z, -angular.z * arm.y) + linear_velocity.x,
      std::fma(angular.z, arm.x, -angular.x * arm.z) + linear_velocity.y,
      std::fma(angular.x, arm.y, -angular.y * arm.x) + linear_velocity.z,
  };
}

// Select the same body-frame solid for pressure aperture and MAC wall history.
absl::Status ApplyOrientedBoxFaceSamples(
    SparseMacGridView3D grid,
    const RigidBody3D& body,
    Vec3 body_half_extent,
    std::span<LiquidBoundaryFaceTag3D> tags,
    std::span<LiquidPrescribedWallSample3D> walls
) {
  constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
  constexpr std::array kUnitAxes{Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}, Vec3{0.0, 0.0, 1.0}};
  std::array<Vec3, 3> world_axes;
  for (std::size_t axis = 0; axis < world_axes.size(); ++axis) {
    const auto rotated = RotateRigidBodyVector3D(body.orientation, kUnitAxes[axis]);
    if (!rotated.ok()) {
      return rotated.status();
    }
    world_axes[axis] = *rotated;
  }
  const auto geometry = grid.geometry();
  for (std::size_t slot = 0; slot < grid.face_count(); ++slot) {
    const auto face = grid.FaceIndex(slot);
    if (!face.ok()) {
      return face.status();
    }
    const auto anchor = boundary_internal::TankFaceAnchorInCellWidths(*face);
    Vec3 arm{};
    for (const auto component : kComponents) {
      const auto distance = boundary_internal::HorizontalPlaneLatticeDistance(
          body.center.*component,
          {geometry.world_origin.*component, anchor.*component, geometry.cell_width}
      );
      if (!distance.ok()) {
        return boundary_internal::ArithmeticStatus(distance.status(), true);
      }
      arm.*component = *distance;
    }
    double nearest = std::numeric_limits<double>::infinity();
    Vec3 outward{};
    for (std::size_t axis = 0; axis < world_axes.size(); ++axis) {
      const double local_coordinate = liquid_step_internal::Dot(world_axes[axis], arm);
      const double clearance = body_half_extent.*kComponents[axis] - std::abs(local_coordinate);
      if (!std::isfinite(clearance)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      if (clearance < nearest) {
        nearest = clearance;
        outward = liquid_step_internal::Scale(world_axes[axis], local_coordinate < 0.0 ? -1.0 : 1.0);
      }
      if (clearance < 0.0) {
        break;
      }
    }
    if (nearest >= 0.0) {
      if (!tags.empty()) {
        tags[slot].structurally_eligible = false;
      }
      if (!walls.empty()) {
        const auto wall_velocity = RigidSurfaceVelocity(body, arm, body.linear_velocity);
        if (!liquid_step_internal::Finite(wall_velocity)) {
          return Error(absl::StatusCode::kOutOfRange);
        }
        walls[slot] = {true, 0, outward, wall_velocity};
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CheckOrientedBoxSweepInTank(
    const RigidBodyStep3D& step,
    Vec3 body_half_extent,
    const AxisAlignedLiquidTank3D& tank,
    double cell_width
) {
  const double radius = std::nextafter(
      std::hypot(body_half_extent.x, body_half_extent.y, body_half_extent.z),
      std::numeric_limits<double>::infinity()
  );
  const auto travel = RigidBodySweepDistanceBound3D(step, radius);
  if (!travel.ok()) {
    return travel.status();
  }
  const double swept_radius = boundary_internal::AddUp(radius, *travel);
  const auto center = step.start.center;
  if (!std::isfinite(swept_radius) || *travel > 0.5 * cell_width
      || boundary_internal::AddDown(center.x, -swept_radius) <= tank.start_low_x
      || boundary_internal::AddUp(center.x, swept_radius) >= tank.high_x
      || boundary_internal::AddDown(center.y, -swept_radius) <= tank.floor_y
      || boundary_internal::AddDown(center.z, -swept_radius) <= tank.low_z
      || boundary_internal::AddUp(center.z, swept_radius) >= tank.high_z) {
    return Error(absl::StatusCode::kAborted);
  }
  return absl::OkStatus();
}
// Check the complete owned payload before publishing any reservation or growth.
absl::StatusOr<std::size_t> ReservationBytes(
    LiquidStateCapacities3D capacities,
    std::size_t source_count,
    std::size_t constituent_material_count,
    std::size_t phase_pair_count,
    std::size_t workspace_bytes,
    std::size_t staging_bytes,
    bool cavity_enabled,
    bool adaptive = false
) {
  std::size_t total = 0;
  const auto add = [&](std::size_t count, std::size_t stride) -> absl::Status {
    if (count > (static_cast<std::size_t>(PTRDIFF_MAX) - total) / stride) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    total += count * stride;
    return absl::OkStatus();
  };
  STEP_TRY(add(capacities.particles, 3 * sizeof(LiquidParticleState3D)));
  STEP_TRY(add(capacities.blocks, 3 * sizeof(LiquidBlockState3D)));
  STEP_TRY(add(capacities.cells, 3 * (adaptive ? sizeof(LiquidAdaptiveCellState3D) : sizeof(LiquidCellState3D))));
  STEP_TRY(add(capacities.faces, 3 * (adaptive ? sizeof(LiquidAdaptiveFaceState3D) : sizeof(LiquidFaceState3D))));
  STEP_TRY(add(capacities.sources, 3 * sizeof(LiquidSourceState3D)));
  STEP_TRY(add(source_count, sizeof(LiquidEmissionSource3D)));
  STEP_TRY(add(constituent_material_count, sizeof(LiquidConstituentMaterial3D)));
  STEP_TRY(add(phase_pair_count, sizeof(LiquidPhasePair3D)));
  STEP_TRY(add(workspace_bytes, 1));
  STEP_TRY(add(staging_bytes, 1));
  if (cavity_enabled) {
    STEP_TRY(add(capacities.cells, 3 * (sizeof(LiquidCavityGasState3D) + sizeof(LiquidCavityComponentState3D))));
  }
  return total;
}
bool Complete(const LiquidSubstepBackend3D& backend) {
  return backend.sample_field && backend.sample_faces && backend.initialize && backend.prepare_publication
      && backend.commit_publication && backend.discard_publication && backend.prepare_growth && backend.commit_growth
      && backend.discard_growth && backend.topology && backend.deposit && backend.normalize
      && backend.predict_missing_face && backend.forces && backend.freeze && backend.links && backend.project
      && backend.gather && backend.update_particles;
}

// Select the projected new history only for particles in a frozen pressure cell.
// An unresolved containing cell has to be reported in scalar particle order.
std::optional<bool> UsesProjectedGatherHistory(const FrozenLiquidBoundaryGeometryView3D& frozen, Vec3 position) {
  const auto cell = liquid_step_internal::TankContainingCell(frozen.mac_grid.geometry(), position);
  if (!cell) {
    return std::nullopt;
  }
  const auto cell_key = CellKey(*cell);
  const auto found = std::ranges::lower_bound(frozen.cells, cell_key, {}, [](const auto& pressure_cell) {
    return CellKey(pressure_cell.cell_index);
  });
  return found != frozen.cells.end() && CellKey(found->cell_index) == cell_key;
}
auto FaceKey(FaceIndex3D index) {
  return std::tuple(index.axis, index.lattice_index.z, index.lattice_index.y, index.lattice_index.x);
}

using liquid_step_internal::BalanceSealedLiquidDrift3D;
using liquid_step_internal::LiquidVolumeDriftRate3D;
using liquid_step_internal::Staging;

// Check physical compatibility before adding any sensor correction.
absl::Status ApplyIncompressibleDrift(
    SparseMacGridView3D grid,
    double step_duration,
    LiquidSceneDrift3D controls,
    Staging& staging,
    MacPressureWorkspace3D& workspace,
    LiquidPressureProblemView3D& problem
) {
  const auto components = workspace.ClassifyPhysicalComponents(grid, step_duration, problem);
  if (!components.ok()) {
    return components.status();
  }
  auto& corrected = *staging.drift_pressure_cells;
  auto& sealed = *staging.sealed_drift_cells;
  STEP_TRY(corrected.Resize(problem.cells.size()));
  std::ranges::copy(problem.cells, corrected.values().begin());
  const auto add_correction = [&](std::size_t cell, double correction) {
    if (correction != 0.0) {
      corrected[cell].target_divergence += correction;
    }
    return std::isfinite(corrected[cell].target_divergence) ? absl::OkStatus() : Error(absl::StatusCode::kOutOfRange);
  };
  for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
    if (std::ranges::binary_search(components->sealed_roots, components->roots[cell])) {
      continue;
    }
    const auto correction =
        LiquidVolumeDriftRate3D((*staging.drift_sensors)[cell], 0.0, controls.correction_time, controls.maximum_rate);
    if (!correction.ok()) {
      return correction.status();
    }
    STEP_TRY(add_correction(cell, *correction));
  }
  STEP_TRY(sealed.Resize(problem.cells.size()));
  for (const std::size_t root : components->sealed_roots) {
    std::size_t sensed_count = 0;
    for (std::size_t cell = 0; cell < problem.cells.size(); ++cell) {
      const auto& sensor = (*staging.drift_sensors)[cell];
      if (components->roots[cell] != root || sensor.sampled_reference_volume == 0.0) {
        continue;
      }
      const double raw_rate = -std::log(sensor.mean_volume_ratio) / controls.correction_time;
      if (!std::isfinite(raw_rate)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      sealed[sensed_count++] = {
          .cell_ordinal = cell,
          .fluid_volume = problem.cells[cell].fluid_volume,
          .raw_rate = raw_rate,
      };
    }
    const auto sensed_cells = sealed.values().first(sensed_count);
    STEP_TRY(BalanceSealedLiquidDrift3D(sensed_cells, controls.maximum_rate));
    for (const auto& cell : sensed_cells) {
      STEP_TRY(add_correction(cell.cell_ordinal, cell.corrected_rate));
    }
  }
  problem.cells = corrected.values();
  return absl::OkStatus();
}

bool HasConstituentViscosity(const LiquidSceneInputs3D& scene) {
  return std::ranges::any_of(scene.constituent_materials, [](const auto& material) {
    return material.dynamic_viscosity_pa_s.has_value();
  });
}

bool CanApplyViscosity(const LiquidSceneInputs3D& scene) {
  return scene.dynamic_viscosity > 0.0 || std::ranges::any_of(scene.constituent_materials, [](const auto& material) {
           return material.dynamic_viscosity_pa_s.value_or(0.0) > 0.0;
         });
}
MacPressureWorkspaceMode3D PressureWorkspaceMode(const LiquidSceneInputs3D& scene) {
  return CanApplyViscosity(scene) ? MacPressureWorkspaceMode3D::kCoupledViscosity
                                  : MacPressureWorkspaceMode3D::kInviscid;
}

bool CanStageRegularCellTransport(const LiquidSceneInputs3D& scene, const AxisAlignedLiquidTank3D* tank) {
  return tank && tank->piston_velocity_x == 0 && std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field)
      && scene.sources.empty() && scene.target_divergence == 0 && std::isinf(scene.bulk_modulus) && !scene.eos
      && (!scene.drift || scene.drift->maximum_rate == 0.0) && scene.surface_tension == 0 && !CanApplyViscosity(scene);
}

// Continue projected changes only when the new donors are a strict subset of
// the old donors and both histories describe the same open, untagged surface.
bool CanContinueIsolatedDropProjection(
    const LiquidBoundaryVelocityHistory3D& old_history,
    const LiquidBoundaryVelocityHistory3D& projected_history
) {
  const auto usable = [](LiquidFaceValidity3D validity) {
    return validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted;
  };
  bool removed_donor = false;
  for (std::size_t slot = 0; slot < projected_history.validity.size(); ++slot) {
    const auto& old_tag = old_history.face_tags[slot];
    const auto& projected_tag = projected_history.face_tags[slot];
    if (old_tag.side_tag != projected_tag.side_tag
        || old_tag.structurally_eligible != projected_tag.structurally_eligible
        || old_history.wall_samples[slot].available || projected_history.wall_samples[slot].available) {
      return false;
    }
    const bool old_donor = usable(old_history.validity[slot]);
    const bool projected_donor = usable(projected_history.validity[slot]);
    if (projected_donor && !old_donor) {
      return false;
    }
    removed_donor = removed_donor || (old_donor && !projected_donor);
  }
  return removed_donor;
}

Vec3 ContinueProjectedChange(Vec3 full_old, Vec3 projected, Vec3 restricted_old) {
  return {
      full_old.x + (projected.x - restricted_old.x),
      full_old.y + (projected.y - restricted_old.y),
      full_old.z + (projected.z - restricted_old.z),
  };
}

Mat3 ContinueProjectedChange(Mat3 full_old, Mat3 projected, Mat3 restricted_old) {
  return {
      ContinueProjectedChange(full_old.x_row, projected.x_row, restricted_old.x_row),
      ContinueProjectedChange(full_old.y_row, projected.y_row, restricted_old.y_row),
      ContinueProjectedChange(full_old.z_row, projected.z_row, restricted_old.z_row),
  };
}

// The restricted and full old gathers are copied before reusing workspace.
// The projected result remains borrowed until the next gather.
absl::StatusOr<LiquidBoundaryGatherView3D> GatherIsolatedDropHistories(
    const LiquidSubstepBackend3D& backend,
    LiquidSubstepBackend3D::Gather& gather,
    const LiquidBoundaryGatherViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace,
    std::span<LiquidBoundaryParticleGather3D> restricted_old,
    std::span<LiquidBoundaryParticleGather3D> full_old
) {
  auto old_inputs = inputs;
  old_inputs.new_history.face_velocities = inputs.old_history.face_velocities;
  const auto restricted = gather(backend.context, old_inputs, workspace);
  if (!restricted.ok()) {
    return restricted.status();
  }
  if (restricted->particles.size() != inputs.particles.size()) {
    return Error(absl::StatusCode::kInternal);
  }
  std::ranges::copy(restricted->particles, restricted_old.begin());
  old_inputs.new_history.validity = inputs.old_history.validity;
  const auto full = gather(backend.context, old_inputs, workspace);
  if (!full.ok()) {
    return full.status();
  }
  if (full->particles.size() != inputs.particles.size()) {
    return Error(absl::StatusCode::kInternal);
  }
  std::ranges::copy(full->particles, full_old.begin());
  auto projected = gather(backend.context, inputs, workspace);
  if (projected.ok() && projected->particles.size() != inputs.particles.size()) {
    return Error(absl::StatusCode::kInternal);
  }
  return projected;
}

// Publish the corrected record only after checking all continued components.
absl::Status CorrectIsolatedDropProjectedChange(
    const LiquidBoundaryParticleGather3D& full_old,
    const LiquidBoundaryParticleGather3D& restricted_old,
    LiquidBoundaryParticleGather3D result,
    LiquidBoundaryParticleGather3D& corrected
) {
  result.grid_velocity_increment =
      ContinueProjectedChange(Vec3{}, result.new_pic_velocity, restricted_old.new_pic_velocity);
  result.new_pic_velocity =
      ContinueProjectedChange(full_old.new_pic_velocity, result.new_pic_velocity, restricted_old.new_pic_velocity);
  result.fitted_velocity_slopes = ContinueProjectedChange(
      full_old.fitted_velocity_slopes,
      result.fitted_velocity_slopes,
      restricted_old.fitted_velocity_slopes
  );
  result.velocity_gradient =
      ContinueProjectedChange(full_old.velocity_gradient, result.velocity_gradient, restricted_old.velocity_gradient);
  using liquid_step_internal::Finite;
  if (!Finite(result.grid_velocity_increment) || !Finite(result.new_pic_velocity)) {
    return Error(absl::StatusCode::kOutOfRange);
  }
  for (const auto row : {&Mat3::x_row, &Mat3::y_row, &Mat3::z_row}) {
    if (!Finite(result.fitted_velocity_slopes.*row) || !Finite(result.velocity_gradient.*row)) {
      return Error(absl::StatusCode::kOutOfRange);
    }
  }
  corrected = result;
  return absl::OkStatus();
}

absl::StatusOr<LiquidBoundaryGatherView3D> GatherIsolatedDropProjectedChange(
    const LiquidSubstepBackend3D& backend,
    const LiquidBoundaryGatherViewInputs3D& inputs,
    LiquidBoundaryWorkspace3D& workspace,
    LiquidBoundaryParticleGather3D& corrected
) {
  LiquidBoundaryParticleGather3D restricted_old;
  LiquidBoundaryParticleGather3D full_old;
  auto projected = GatherIsolatedDropHistories(
      backend,
      *backend.gather,
      inputs,
      workspace,
      std::span(&restricted_old, 1),
      std::span(&full_old, 1)
  );
  if (!projected.ok()) {
    return projected.status();
  }
  STEP_TRY(CorrectIsolatedDropProjectedChange(full_old, restricted_old, projected->particles[0], corrected));
  projected->particles = std::span(&corrected, 1);
  return projected;
}

// The frozen faces and pressure links share the owned-slot order. A sealed
// face supplies a boundary value independently of its dormant carrier sample.
std::optional<double> RegularTransportFaceVelocity(
    const FrozenLiquidBoundaryGeometryView3D& frozen,
    std::span<const MacPressureFace3D> links,
    const liquid_step_internal::Staging& staging,
    FaceIndex3D face_index,
    std::int64_t particle_side_tag,
    const boundary_internal::TankLatticeAxis& normal_lattice,
    double wall_coordinate
) {
  const auto slot = frozen.mac_grid.FaceSlot(face_index);
  if (!slot.ok()) {
    return std::nullopt;
  }
  const auto face = std::ranges::lower_bound(frozen.faces, *slot, {}, &FrozenLiquidFace3D::face_slot);
  if (face == frozen.faces.end() || face->face_slot != *slot) {
    return std::nullopt;
  }
  const auto ordinal = static_cast<std::size_t>(face - frozen.faces.begin());
  if (ordinal >= links.size()) {
    return std::nullopt;
  }
  const auto& link = links[ordinal];
  const bool matching_link = FaceKey(link.face_index) == FaceKey(face_index) && link.open_area == face->open_area;
  if (face->prescribed_wall_flux.value_or(0) != 0) {
    return std::nullopt;
  }
  if (!matching_link || link.prescribed_wall_flux.value_or(0) != 0) {
    return std::nullopt;
  }
  const double full_face_area = normal_lattice.cell_width * normal_lattice.cell_width;
  if (face->open_area == full_face_area && face->blocked_area == 0) {
    const auto& tag = staging.new_tags[*slot];
    const auto validity = staging.validity[*slot];
    const bool usable = validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted;
    const double velocity = staging.new_velocities[*slot];
    if (tag.structurally_eligible && tag.side_tag == particle_side_tag && usable && std::isfinite(velocity)) {
      return velocity;
    }
    return std::nullopt;
  }
  const bool fully_sealed = face->open_area == 0 && face->blocked_area == full_face_area;
  const bool stationary = face->wall_normal_velocity == 0 && link.wall_normal_velocity == 0;
  if (!fully_sealed || !stationary || !std::isfinite(wall_coordinate)) {
    return std::nullopt;
  }
  const auto wall_offset_sign = boundary_internal::TankPlaneOffsetSign(wall_coordinate, normal_lattice);
  if (wall_offset_sign.ok() && *wall_offset_sign == 0) {
    return 0;
  }
  return std::nullopt;
}

// Select only a frozen pressure cell with six usable normal samples.
// Use the same checked lattice arithmetic for the cell and local fractions.
LiquidRegularCellTransport3D RegularTransportSample(
    const FrozenLiquidBoundaryGeometryView3D& frozen,
    const LiquidPressureProblemView3D& links,
    const liquid_step_internal::Staging& staging,
    const AxisAlignedLiquidTank3D& tank,
    const LiquidParticleState3D& particle
) {
  const auto geometry = frozen.mac_grid.geometry();
  const auto cell = liquid_step_internal::TankContainingCell(geometry, particle.world_position);
  if (!cell) {
    return {};
  }
  const auto cell_key = CellKey(*cell);
  const auto pressure_cell = std::ranges::lower_bound(frozen.cells, cell_key, {}, [](const auto& frozen_cell) {
    return CellKey(frozen_cell.cell_index);
  });
  if (pressure_cell == frozen.cells.end() || CellKey(pressure_cell->cell_index) != cell_key) {
    return {};
  }
  const auto cell_ordinal = static_cast<std::size_t>(pressure_cell - frozen.cells.begin());
  if (cell_ordinal >= links.cells.size() || CellKey(links.cells[cell_ordinal].cell_index) != cell_key) {
    return {};
  }
  constexpr std::array components{&Vec3::x, &Vec3::y, &Vec3::z};
  constexpr std::array axes{FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ};
  constexpr std::array face_components{&FaceLatticeIndex3D::x, &FaceLatticeIndex3D::y, &FaceLatticeIndex3D::z};
  const Vec3 low_wall{tank.start_low_x, tank.floor_y, tank.low_z};
  const Vec3 high_wall{tank.high_x, std::numeric_limits<double>::infinity(), tank.high_z};
  LiquidRegularCellTransport3D sample;
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    const auto component = components[axis];
    const auto face_component = face_components[axis];
    const FaceIndex3D low_face{axes[axis], {cell->x, cell->y, cell->z}};
    const auto low_plane_index = low_face.lattice_index.*face_component;
    FaceIndex3D high_face = low_face;
    ++(high_face.lattice_index.*face_component);
    const boundary_internal::TankLatticeAxis low_lattice{
        geometry.world_origin.*component,
        static_cast<double>(low_plane_index),
        geometry.cell_width,
    };
    auto high_lattice = low_lattice;
    high_lattice.lattice_index += 1;
    const auto fraction = boundary_internal::TankClipCoordinate(particle.world_position.*component, low_lattice);
    if (!fraction.ok()) {
      return {};
    }
    const auto low_velocity = RegularTransportFaceVelocity(
        frozen,
        links.faces,
        staging,
        low_face,
        particle.side_tag,
        low_lattice,
        low_wall.*component
    );
    const auto high_velocity = RegularTransportFaceVelocity(
        frozen,
        links.faces,
        staging,
        high_face,
        particle.side_tag,
        high_lattice,
        high_wall.*component
    );
    if (!low_velocity || !high_velocity) {
      return {};
    }
    sample.position_in_cell_widths.*component = *fraction;
    sample.low_face_normal_velocity.*component = *low_velocity;
    sample.high_face_normal_velocity.*component = *high_velocity;
  }
  sample.active = true;
  return sample;
}
struct CavityStepScratch {
  LiquidRecords3D<LiquidAirVolumeCell3D> previous_gas;
  LiquidRecords3D<std::size_t> previous_labels;
  LiquidRecords3D<LiquidCavityLineageState3D> previous_lineage;
  LiquidRecords3D<LiquidAirVolumeCell3D> candidate_gas;
  LiquidRecords3D<std::size_t> candidate_labels;
  LiquidRecords3D<std::size_t> queue;
  LiquidRecords3D<LiquidAirComponent3D> components;
  LiquidRecords3D<LiquidAirComponentOverlap3D> overlaps;
  LiquidRecords3D<std::optional<LiquidCavityInventory3D>> inventories;
  LiquidRecords3D<LiquidCavityLineageCandidate3D> candidates;
  LiquidRecords3D<LiquidCavityBoundaryComponent3D> boundary_components;
  LiquidRecords3D<LiquidAirCellPressure3D> pressures;
  std::size_t reserved_bytes = 0;

  static absl::StatusOr<CavityStepScratch> Create(std::size_t capacity) {
    CavityStepScratch owner;
    const auto reserve = [&]<typename Record>(LiquidRecords3D<Record>& records) -> absl::Status {
      constexpr std::size_t stride = sizeof(Record);
      if (capacity > (SIZE_MAX - owner.reserved_bytes) / stride) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      auto storage = LiquidRecords3D<Record>::Create(capacity);
      if (!storage.ok()) {
        return storage.status();
      }
      STEP_TRY(storage->Resize(capacity));
      records = std::move(*storage);
      owner.reserved_bytes += capacity * stride;
      return absl::OkStatus();
    };
    STEP_TRY(reserve(owner.previous_gas));
    STEP_TRY(reserve(owner.previous_labels));
    STEP_TRY(reserve(owner.previous_lineage));
    STEP_TRY(reserve(owner.candidate_gas));
    STEP_TRY(reserve(owner.candidate_labels));
    STEP_TRY(reserve(owner.queue));
    STEP_TRY(reserve(owner.components));
    STEP_TRY(reserve(owner.overlaps));
    STEP_TRY(reserve(owner.inventories));
    STEP_TRY(reserve(owner.candidates));
    STEP_TRY(reserve(owner.boundary_components));
    STEP_TRY(reserve(owner.pressures));
    return owner;
  }

  absl::StatusOr<LiquidStepCavityBoundaries3D> Prepare(
      const LiquidSceneCavity3D& settings,
      const LiquidBoundaryGeometryViewInputs3D& samples,
      LiquidState3D& state,
      std::optional<LiquidCavityStepPreparation3D>& measurement
  ) {
    std::span<const FaceIndex3D> outlets;
    if (settings.atmosphere_outlets) {
      const auto supplied = settings.atmosphere_outlets(settings.context, samples);
      if (!supplied.ok()) {
        return supplied.status();
      }
      outlets = *supplied;
    }
    struct BirthEquilibrium {
      const LiquidSceneCavity3D& settings;
      double surface_tension;
    } equilibrium{settings, samples.surface_tension};
    const auto default_initialize = [](void* context, const LiquidAirComponent3D& component) {
      const auto& birth = *static_cast<const BirthEquilibrium*>(context);
      return InitializeSubmergedLiquidCavity3D(
          component.gas_volume,
          birth.settings.atmospheric_absolute_pressure,
          birth.settings.birth_hydrostatic_gauge_pressure,
          birth.surface_tension
      );
    };
    // The transaction supplies the last accepted snapshot, including on resume
    // and capacity replay. Gas passages are only needed for the current labeler.
    for (std::size_t index = 0; index < state.cavity_gas.size(); ++index) {
      const auto& saved = state.cavity_gas[index];
      previous_gas[index] = {.cell_index = saved.cell_index, .gas_volume = saved.gas_volume};
      previous_labels[index] = saved.component;
    }
    for (std::size_t index = 0; index < state.cavity_components.size(); ++index) {
      const auto& saved = state.cavity_components[index];
      auto& lineage = previous_lineage[index];
      lineage = {.id = saved.id, .inventory = std::nullopt};
      if (saved.id != 0) {
        lineage.inventory = LiquidCavityInventory3D{saved.initial_volume, saved.initial_absolute_pressure};
      }
    }
    const LiquidCavityStepInputs3D inputs{
        .geometry = samples.mac_grid.geometry(),
        .obstacle = samples.obstacle,
        .sampled_cells = samples.cell_indices,
        .corners = samples.corner_distances,
        .centers = samples.center_distances,
        .atmosphere_outlets = outlets,
        .previous_snapshot =
            {
                previous_gas.values().first(state.cavity_gas.size()),
                previous_labels.values().first(state.cavity_gas.size()),
                state.cavity_components.size(),
            },
        .previous_lineage = previous_lineage.values().first(state.cavity_components.size()),
        .next_id = state.next_cavity_id,
        .atmospheric_absolute_pressure = settings.atmospheric_absolute_pressure,
        .liquid_density = samples.liquid_density,
        .exterior = !settings.atmosphere_outlets && std::holds_alternative<std::monostate>(samples.obstacle)
            ? LiquidAirExterior3D::kUnobstructedBoundingBox
            : LiquidAirExterior3D::kExplicitOutlets,
        .initialization_context = settings.initialize ? settings.context : &equilibrium,
        .initialize = settings.initialize ? settings.initialize : +default_initialize,
    };
    const LiquidCavityStepScratch3D scratch{
        .gas_cells = candidate_gas.values(),
        .labels = candidate_labels.values(),
        .queue = queue.values(),
        .components = components.values(),
        .overlaps = overlaps.values(),
        .inventories = inventories.values(),
        .lineage = candidates.values(),
        .boundary_components = boundary_components.values(),
        .air_pressures = pressures.values(),
    };
    const auto prepared = PrepareLiquidCavityStep3D(inputs, scratch);
    if (!prepared.ok()) {
      return prepared.status();
    }
    STEP_TRY(state.cavity_gas.Resize(prepared->gas_cell_count));
    STEP_TRY(state.cavity_components.Resize(prepared->component_count));
    for (std::size_t index = 0; index < prepared->gas_cell_count; ++index) {
      const auto& cell = candidate_gas[index];
      state.cavity_gas[index] = {cell.cell_index, cell.gas_volume, candidate_labels[index]};
    }
    for (std::size_t index = 0; index < prepared->component_count; ++index) {
      const auto& lineage = candidates[index].state;
      auto& saved = state.cavity_components[index];
      saved = {.id = lineage.id, .initial_volume = 0.0, .initial_absolute_pressure = 0.0};
      if (lineage.inventory) {
        saved.initial_volume = lineage.inventory->initial_volume;
        saved.initial_absolute_pressure = lineage.inventory->initial_absolute_pressure;
      }
    }
    state.next_cavity_id = prepared->lineage.next_id;
    measurement = *prepared;
    return LiquidStepCavityBoundaries3D{
        .air_pressures = pressures.values().first(prepared->boundaries.air_pressure_count),
        .maximum_step_duration = prepared->boundaries.maximum_step_duration,
    };
  }
};

}  // namespace
struct LiquidSubstepScheduler3D::SnowTrial {
  const LiquidSnowStepInputs3D& inputs;
  std::vector<PlasticSolidCheckpointParticle3D> candidate;

  absl::Status Advance(
      double duration,
      const LiquidObstacle3D& obstacle,
      std::span<const PlasticSolidPressureImpulse3D> loads,
      const LiquidState3D& committed_liquid,
      const LiquidState3D& trial_liquid
  ) {
    auto particles = StagePressureLoadedSnowParticles3D(
        inputs.accepted_particles,
        inputs.body_id,
        inputs.material_id,
        inputs.snow,
        inputs.solid_grid.geometry().cell_width,
        loads
    );
    if (!particles.ok()) {
      return particles.status();
    }
    auto settings = inputs.solid_step;
    settings.duration = duration;
    if (const auto* platen = std::get_if<TranslatingHalfspace3D>(&obstacle)) {
      settings.platen = DrySolidPlaten3D{
          .start_y = -platen->start_plane_coordinate,
          .velocity_y = platen->wall_velocity.y,
      };
    }
    const auto dry_status = AdvanceDrySolid3D(inputs.solid_grid, inputs.snow, settings, *particles);
    if (internal::DrySolidDurationExceeded3D(dry_status)) {
      return absl::AbortedError(dry_status.message());
    }
    STEP_TRY(dry_status);
    for (const auto& liquid : trial_liquid.particles.values()) {
      const auto old_particles = committed_liquid.particles.values();
      const auto old_liquid = std::ranges::find(old_particles, liquid.id, &LiquidParticleState3D::id);
      const auto start_liquid = old_liquid == old_particles.end() ? liquid.world_position : old_liquid->world_position;
      for (std::size_t i = 0; i < particles->size(); ++i) {
        const auto old = inputs.accepted_particles[i].transfer.sorted_world_position;
        const auto next = (*particles)[i].world_position;
        const Vec3 start{start_liquid.x - old.x, start_liquid.y - old.y, start_liquid.z - old.z};
        const Vec3 end{
            liquid.world_position.x - next.x,
            liquid.world_position.y - next.y,
            liquid.world_position.z - next.z
        };
        const Vec3 sweep{end.x - start.x, end.y - start.y, end.z - start.z};
        const double distance = std::hypot(start.x, start.y, start.z);
        const double length = std::hypot(sweep.x, sweep.y, sweep.z);
        if (!std::isfinite(distance) || !std::isfinite(length)) {
          return Error(absl::StatusCode::kOutOfRange);
        }
        if (distance < inputs.support_radius) {
          return Error(absl::StatusCode::kFailedPrecondition);
        }
        if (length == 0) {
          continue;
        }
        const Vec3 direction{sweep.x / length, sweep.y / length, sweep.z / length};
        const double along =
            std::clamp(-(start.x * direction.x + start.y * direction.y + start.z * direction.z), 0.0, length);
        const double closest = std::hypot(
            std::fma(along, direction.x, start.x),
            std::fma(along, direction.y, start.y),
            std::fma(along, direction.z, start.z)
        );
        if (!std::isfinite(closest)) {
          return Error(absl::StatusCode::kOutOfRange);
        }
        if (closest < inputs.support_radius) {
          return Error(absl::StatusCode::kAborted);
        }
      }
    }
    candidate.reserve(particles->size());
    for (std::size_t i = 0; i < particles->size(); ++i) {
      const auto record = CapturePlasticSolidCheckpointParticle3D(
          inputs.accepted_particles[i].particle_id,
          inputs.body_id,
          inputs.material_id,
          inputs.snow,
          (*particles)[i]
      );
      if (!record.ok()) {
        return record.status();
      }
      candidate.push_back(*record);
    }
    return absl::OkStatus();
  }
};

struct LiquidSubstepScheduler3D::Impl {
  LiquidFrameTransaction3D transaction;
  LiquidStepWorkspace3D workspace;
  liquid_step_internal::Staging staging;
  LiquidRecords3D<LiquidEmissionSource3D> sources;
  LiquidRecords3D<LiquidConstituentMaterial3D> constituent_materials;
  LiquidRecords3D<LiquidPhasePair3D> phase_pairs;
  LiquidSceneInputs3D scene;
  std::optional<CavityStepScratch> cavity;
  LiquidStepOptions3D options;
  LiquidSubstepBackend3D backend;
  std::optional<liquid_step_internal::AdaptiveLiquidStep3D> adaptive;
  std::optional<ResolvedOceanBoundary3D> ocean_boundary;
  std::optional<liquid_step_internal::OceanInletParticles3D> ocean_inlet_particles;
  LiquidStepDiagnostics3D diagnostics;
  LiquidStageFaultHook3D hook;
  double target_time = 0;
  bool frame_active = false;
  Impl(
      LiquidFrameTransaction3D transaction_owner,
      LiquidStepWorkspace3D workspace_owner,
      liquid_step_internal::Staging staging_owner
  )
      : transaction(std::move(transaction_owner)),
        workspace(std::move(workspace_owner)),
        staging(std::move(staging_owner)) {
  }
  std::size_t reservation_bytes = 0;
  absl::Status Enter(LiquidPhysicalStage3D stage, double step_duration) {
    diagnostics.stages[diagnostics.stage_count++] = stage;
    if (hook.before_stage) {
      return hook.before_stage(hook.context, stage, step_duration);
    }
    return absl::OkStatus();
  }
  void SetWorkPhase(LiquidWorkPhaseObserver3D observer, LiquidWorkPhase3D phase) {
    if (observer.enter) {
      observer.enter(observer.context, phase);
    }
    if (backend.set_work_phase) {
      backend.set_work_phase(backend.context, phase);
    }
  }
  absl::Status ExecuteAdaptive(
      liquid_step_internal::AdaptiveLiquidStep3D& owner,
      double duration,
      LiquidState3D& trial,
      LiquidAcceptedStepMeasurement3D& measurement,
      LiquidWorkPhaseObserver3D work_phase
  ) {
    STEP_TRY(Enter(LiquidPhysicalStage3D::kActivate, duration));
    SetWorkPhase(work_phase, LiquidWorkPhase3D::kDepositNormalization);
    STEP_TRY(Enter(LiquidPhysicalStage3D::kDeposit, duration));
    STEP_TRY(owner.Deposit(trial, options));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kOldSnapshot, duration));
    SetWorkPhase(work_phase, LiquidWorkPhase3D::kForcesGeometryPressure);
    STEP_TRY(Enter(LiquidPhysicalStage3D::kPressure, duration));
    STEP_TRY(owner.Project(trial, duration, scene, options, backend, workspace.pressure(), measurement));
    diagnostics.pressure_diagnostics = measurement.pressure;
    STEP_TRY(Enter(LiquidPhysicalStage3D::kNewSnapshot, duration));
    SetWorkPhase(work_phase, LiquidWorkPhase3D::kGatherContactAcceptance);
    STEP_TRY(Enter(LiquidPhysicalStage3D::kGather, duration));
    STEP_TRY(owner.Gather(trial));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kParticleUpdate, duration));
    STEP_TRY(owner.Update(trial, duration, scene, options, backend));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kValidate, duration));
    STEP_TRY(ValidateLiquidState3D(trial));
    return Enter(LiquidPhysicalStage3D::kAccept, duration);
  }
  absl::Status Execute(
      double step_duration,
      LiquidState3D& trial,
      LiquidAcceptedStepMeasurement3D& accepted_measurement,
      LiquidWorkPhaseObserver3D work_phase,
      RigidBody3D* piston_load,
      bool dynamic_piston,
      bool coupled_piston,
      const LiquidPistonReactionUpdate3D* update_with_reaction,
      FloatingBoxTrial* floating_box,
      SnowTrial* snow
  ) {
    auto& boundary = workspace.boundary();
    const double old_time = transaction.committed().clock.time;
    const double new_time = old_time + step_duration;
    auto obstacle = liquid_step_internal::ObstacleAtTime(scene.obstacle, old_time);
    if (dynamic_piston) {
      auto& tank = std::get<AxisAlignedLiquidTank3D>(obstacle);
      tank.start_low_x = piston_load->center.x;
      tank.piston_velocity_x = piston_load->linear_velocity.x;
    }
    auto new_obstacle = liquid_step_internal::ObstacleAtTime(scene.obstacle, new_time);
    const bool oriented_box = floating_box != nullptr && floating_box->oriented_geometry.has_value();
    std::optional<RigidBodyStep3D> oriented_box_candidate;
    bool use_compatible_pressure_transfer = false;
    const auto record_oriented_pressure_load = [&](const RigidBody3D& loaded) {
      auto& loads = *floating_box->oriented_loads;
      loads.pressure_force =
          liquid_step_internal::Add(loaded.force, liquid_step_internal::Scale(loads.external_force, -1.0));
      loads.pressure_torque =
          liquid_step_internal::Add(loaded.torque, liquid_step_internal::Scale(loads.external_torque, -1.0));
    };
    TranslatingContactBox3D contact_box{};
    TranslatingContactBox3D end_box{};
    if (floating_box != nullptr && !oriented_box) {
      contact_box = {
          floating_box->geometry.world_low,
          floating_box->geometry.world_high,
          floating_box->body.linear_velocity,
      };
      const auto translate = [&](Vec3 point) {
        return Vec3{
            std::fma(step_duration, contact_box.wall_velocity.x, point.x),
            std::fma(step_duration, contact_box.wall_velocity.y, point.y),
            std::fma(step_duration, contact_box.wall_velocity.z, point.z),
        };
      };
      end_box = {translate(contact_box.world_low), translate(contact_box.world_high), contact_box.wall_velocity};
    }
    const MacDepositQuantization3D quantization{
        options.mass_per_quantum,
        options.momentum_per_quantum,
        options.capacities.particles,
    };
    STEP_TRY(Enter(LiquidPhysicalStage3D::kEmit, step_duration));
    if (const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle)) {
      STEP_TRY(boundary_internal::ArithmeticStatus(boundary_internal::CheckTankStep(*tank, step_duration), true));
      // Contact and frozen geometry use the start pose and this exact duration.
      new_obstacle = liquid_step_internal::ObstacleAtTime(obstacle, step_duration);
    }
    STEP_TRY(liquid_step_internal::Emit(scene, step_duration, trial));
    if (ocean_inlet_particles) {
      STEP_TRY(ocean_inlet_particles->Apply(*ocean_boundary, obstacle, old_time, trial));
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kActivate, step_duration));
    STEP_TRY(liquid_step_internal::PrepareField(scene.field, obstacle, trial, staging, backend, cavity.has_value()));
    STEP_TRY(backend.topology(
        backend.context,
        trial.particles.values(),
        staging.candidates.values(),
        transaction.committed(),
        workspace
    ));
    STEP_TRY(workspace.ApplyTrialTopology(trial));
    const auto topology = workspace.prepared();
    if (!topology.ok()) {
      return topology.status();
    }
    const auto grid = topology->grid;
    STEP_TRY(liquid_step_internal::PrepareTransfers(trial, grid, staging));
    STEP_TRY(backend.sample_faces(backend.context, grid, obstacle, 0, staging.tags.values(), {}));
    if (floating_box != nullptr) {
      if (oriented_box) {
        STEP_TRY(
            ApplyOrientedBoxFaceSamples(grid, floating_box->body, floating_box->half_extent, staging.tags.values(), {})
        );
      } else {
        STEP_TRY(ApplyBoxFaceSamples(grid, contact_box, staging.tags.values(), {}));
      }
    }
    SetWorkPhase(work_phase, LiquidWorkPhase3D::kDepositNormalization);
    STEP_TRY(Enter(LiquidPhysicalStage3D::kDeposit, step_duration));
    const auto deposits = backend.deposit(
        backend.context,
        {grid, staging.particles.values(), staging.tags.values(), quantization},
        boundary
    );
    if (!deposits.ok()) {
      return deposits.status();
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kOldSnapshot, step_duration));
    const auto normalized =
        backend.normalize(backend.context, deposits->face_quanta, quantization, options.minimum_face_mass, boundary);
    if (!normalized.ok()) {
      return normalized.status();
    }
    std::ranges::copy(normalized->face_velocities, staging.old_velocities.values().begin());
    std::ranges::copy(normalized->validity, staging.validity.values().begin());
    const LiquidBoundaryVelocityHistory3D old_history{
        .velocity_sample_time = old_time,
        .wall_sample_time = old_time,
        .face_velocities = staging.old_velocities.values(),
        .validity = normalized->validity,
        .face_tags = staging.tags.values(),
        .wall_samples = staging.old_walls.values(),
    };
    SetWorkPhase(work_phase, LiquidWorkPhase3D::kForcesGeometryPressure);
    STEP_TRY(Enter(LiquidPhysicalStage3D::kForces, step_duration));
    STEP_TRY(backend.forces(
        backend.context,
        grid,
        staging.old_velocities.values(),
        scene.world_acceleration,
        step_duration,
        staging.new_velocities.values()
    ));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kWaves, step_duration));
    // Both backends publish predicted velocities into host staging. Pressure
    // consumes the relaxed staging values, and old history remains untouched.
    const auto relax_face = [&](std::size_t slot) -> absl::Status {
      const auto face = grid.FaceIndex(slot);
      if (!face.ok()) {
        return face.status();
      }
      const auto anchor = grid.FaceWorldAnchor(*face);
      if (!anchor.ok()) {
        return anchor.status();
      }
      const auto velocity =
          ocean_boundary->RelaxFace(face->axis, *anchor, staging.new_velocities[slot], new_time, step_duration);
      if (!velocity.ok()) {
        return velocity.status();
      }
      staging.new_velocities[slot] = *velocity;
      return absl::OkStatus();
    };
    if (ocean_boundary) {
      for (std::size_t slot = 0; slot < grid.face_count(); ++slot) {
        const auto validity = normalized->validity[slot];
        if (validity == LiquidFaceValidity3D::kValid || validity == LiquidFaceValidity3D::kPredicted) {
          STEP_TRY(relax_face(slot));
        }
      }
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kBodyContact, step_duration));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kInterface, step_duration));
    std::optional<double> exact_curvature;
    const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&scene.field);
    const bool analytic_sphere_uses_capillarity = sphere && scene.surface_tension > 0;
    if (analytic_sphere_uses_capillarity) {
      const double outward_curvature = 2 / sphere->radius;
      if (std::isfinite(outward_curvature)) {
        exact_curvature = outward_curvature;
      }
    }
    std::optional<PlasticSolidBoundarySamples3D> snow_samples;
    LiquidRecords3D<PlasticSolidPressureImpulse3D> snow_loads;
    if (snow != nullptr) {
      auto samples = SamplePlasticSolidBoundary3D(
          grid,
          staging.corners.values(),
          snow->inputs.accepted_particles,
          snow->inputs.body_id,
          snow->inputs.support_radius
      );
      if (!samples.ok()) {
        return samples.status();
      }
      snow_samples = std::move(*samples);
    }
    const LiquidBoundaryGeometryViewInputs3D geometry_inputs{
        .mac_grid = grid,
        .cell_indices = staging.candidates.values(),
        .center_distances = staging.centers.values(),
        .corner_distances = staging.corners.values(),
        .obstacle = obstacle,
        .step_duration = step_duration,
        .liquid_density = scene.liquid_density,
        .surface_tension = scene.surface_tension,
        .exact_outward_interface_curvature = exact_curvature,
        .sampled_solid = snow_samples ? std::optional{snow_samples->View()} : std::nullopt,
    };
    LiquidStepCavityBoundaries3D cavity_boundaries;
    if (cavity || scene.cavity_provider.prepare) {
      const auto prepared = cavity
          ? cavity->Prepare(*scene.cavity, geometry_inputs, trial, accepted_measurement.cavity)
          : scene.cavity_provider.prepare(scene.cavity_provider.context, old_time, geometry_inputs);
      if (!prepared.ok()) {
        return prepared.status();
      }
      cavity_boundaries = *prepared;
      const double bound = cavity_boundaries.maximum_step_duration;
      if (!(bound > 0.0)) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
      if (std::isfinite(bound)) {
        accepted_measurement.step_start_cavity_bound_seconds = bound;
        if (step_duration > bound) {
          return Error(absl::StatusCode::kAborted);
        }
      }
    }
    const auto freeze = [&](const LiquidBoundaryGeometryViewInputs3D& inputs) {
      if (oriented_box) {
        return boundary.FreezeOrientedFloatingBox(inputs, *floating_box->oriented_geometry);
      }
      return floating_box == nullptr ? backend.freeze(backend.context, inputs, boundary)
                                     : boundary.FreezeFloatingBox(inputs, floating_box->geometry);
    };
    const auto frozen = freeze(geometry_inputs);
    if (!frozen.ok()) {
      // Geometry uses FailedPrecondition for both step_duration certification and incompatible
      // geometry. Recheck only the former at the conservative capillary duration;
      // an incompatible geometry remains fatal, regardless of attempted step_duration.
      if (frozen.status().code() == absl::StatusCode::kFailedPrecondition && scene.surface_tension > 0) {
        const double h = trial.geometry.cell_width;
        const double capillary_duration =
            std::sqrt(scene.liquid_density * h * h * h / (4 * std::numbers::pi * scene.surface_tension));
        if (std::isfinite(capillary_duration) && capillary_duration > 0 && step_duration > capillary_duration) {
          auto certified_inputs = geometry_inputs;
          certified_inputs.step_duration = capillary_duration;
          const auto certified = freeze(certified_inputs);
          if (certified.ok() && step_duration > certified->capillary_step_bound) {
            return Error(absl::StatusCode::kAborted);
          }
        }
      }
      return frozen.status();
    }
    if (frozen->capillary_step_bound != std::numeric_limits<double>::infinity()) {
      // An unresolved sphere without an open air link has no capillary
      // contribution. Require a representable curvature when a link uses it.
      if (analytic_sphere_uses_capillarity && !exact_curvature.has_value()) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      accepted_measurement.step_start_capillary_bound_seconds = frozen->capillary_step_bound;
    }
    const auto* tank_at_step_start = std::get_if<AxisAlignedLiquidTank3D>(&obstacle);
    if (tank_at_step_start && floating_box == nullptr) {
      accepted_measurement.piston_pose = LiquidPistonStepPose3D{
          .step_start_low_x_m = tank_at_step_start->start_low_x,
          .step_end_low_x_m = std::get<AxisAlignedLiquidTank3D>(new_obstacle).start_low_x,
      };
    }
    std::optional<LiquidEos3D> eos;
    if (scene.eos) {
      eos = LiquidEos3D{scene.liquid_density, scene.eos->reference_sound_speed, scene.eos->law};
    }
    double minimum_predictor_density = scene.liquid_density;
    STEP_TRY(staging.pressure_fields.Resize(frozen->cells.size()));
    STEP_TRY(staging.incident_velocities.Resize(frozen->faces.size()));
    const auto particles = trial.particles.values();
    const auto first_explicit = std::ranges::find_if(particles, [](const LiquidParticleState3D& particle) {
      return particle.constituent_count != 0;
    });
    const bool uses_composition_density = staging.composition_density && first_explicit != particles.end();
    const bool uses_composition_viscosity = uses_composition_density && staging.composition_viscosity.has_value();
    const double sensor_viscosity = uses_composition_viscosity ? scene.dynamic_viscosity : 0.0;
    std::span<const LiquidInternalInterface3D> internal_interfaces;
    if (uses_composition_density) {
      STEP_TRY(staging.composition_density->Resize(frozen->cells.size()));
      const auto second_phase = std::ranges::find_if(particles, [&](const LiquidParticleState3D& particle) {
        return particle.constituent_count != 0 && particle.phase_id != first_explicit->phase_id;
      });
      if (second_phase == particles.end()) {
        STEP_TRY(liquid_step_internal::SampleLiquidCellCompositionDensity3D(
            grid.geometry(),
            frozen->cells,
            particles,
            scene.constituent_materials,
            scene.liquid_density,
            staging.composition_density->values(),
            sensor_viscosity
        ));
      } else {
        const auto* pair = FindLiquidPhasePair3D(scene.phase_pairs, first_explicit->phase_id, second_phase->phase_id);
        const auto* planar_tank = std::get_if<AxisAlignedLiquidTank3D>(&obstacle);
        const bool supported_exterior = std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field)
            || (std::holds_alternative<LiquidHorizontalPlaneSurface3D>(scene.field) && planar_tank
                && planar_tank->piston_velocity_x == 0.0);
        if (!pair || !staging.phase_crossings || !supported_exterior || staging.drift_sensors) {
          return Error(absl::StatusCode::kFailedPrecondition);
        }
        STEP_TRY(staging.phase_crossings->Resize(frozen->faces.size()));
        const auto count = liquid_step_internal::PrepareLiquidPhaseCrossings3D(
            *frozen,
            particles,
            scene.constituent_materials,
            *pair,
            staging.composition_density->values(),
            staging.phase_crossings->values(),
            sensor_viscosity
        );
        if (!count.ok()) {
          return count.status();
        }
        internal_interfaces = staging.phase_crossings->values().first(*count);
      }
    }
    if (staging.drift_sensors) {
      STEP_TRY(staging.drift_sensors->Resize(frozen->cells.size()));
      STEP_TRY(liquid_step_internal::SampleLiquidCellVolumeRatios3D(
          grid.geometry(),
          frozen->cells,
          trial.particles.values(),
          0,
          staging.drift_sensors->values()
      ));
    }
    for (std::size_t index = 0; index < frozen->cells.size(); ++index) {
      const auto cells = trial.cells.values();
      const auto key = frozen->cells[index].cell_index;
      const auto found = std::lower_bound(cells.begin(), cells.end(), key, [](const auto& cell, auto value) {
        return CellKey(cell.cell_index) < CellKey(value);
      });
      if (found == cells.end() || CellKey(found->cell_index) != CellKey(key)) {
        return Error(absl::StatusCode::kInternal);
      }
      double target_divergence = scene.target_divergence;
      double predicted_pressure = found->previous_pressure;
      double tangent_bulk_modulus = scene.bulk_modulus;
      std::optional<double> predicted_density;
      if (uses_composition_density) {
        const auto& sensed = (*staging.composition_density)[index];
        if (sensed.sampled_volume_m3 > 0.0) {
          predicted_density = sensed.sampled_mass_kg / sensed.sampled_volume_m3;
          if (!std::isfinite(*predicted_density) || *predicted_density <= 0.0) {
            return Error(absl::StatusCode::kOutOfRange);
          }
        } else if (sensed.extrapolated_density > 0.0) {
          predicted_density = sensed.extrapolated_density;
        }
      }
      if (eos) {
        const auto pressure = liquid_step_internal::AdvectLiquidPressureHistory3D(
            transaction.committed(),
            key,
            step_duration,
            scene.ambient_pressure
        );
        if (!pressure.ok()) {
          return pressure.status();
        }
        predicted_pressure = *pressure;
        auto local_eos = *eos;
        local_eos.reference_density = predicted_density.value_or(scene.liquid_density);
        const auto predictor = EvaluateLiquidEosAtPressure3D(local_eos, predicted_pressure);
        if (!predictor.ok()) {
          return predictor.status();
        }
        tangent_bulk_modulus = predictor->bulk_modulus;
        predicted_density = predictor->density;
        if (staging.drift_sensors) {
          const auto correction = liquid_step_internal::LiquidVolumeDriftRate3D(
              (*staging.drift_sensors)[index],
              predictor->log_volume_ratio,
              scene.drift->correction_time,
              scene.drift->maximum_rate
          );
          if (!correction.ok()) {
            return correction.status();
          }
          if (*correction != 0.0) {
            target_divergence += *correction;
          }
          if (!std::isfinite(target_divergence)) {
            return Error(absl::StatusCode::kOutOfRange);
          }
        }
      }
      if (predicted_density) {
        minimum_predictor_density = std::min(minimum_predictor_density, *predicted_density);
      }
      staging.pressure_fields[index] = {
          .target_divergence = target_divergence,
          .bulk_modulus = tangent_bulk_modulus,
          .previous_pressure = predicted_pressure,
          .density = predicted_density,
      };
    }
    if ((eos || uses_composition_density) && accepted_measurement.step_start_capillary_bound_seconds) {
      // Freeze used reference density. Tighten its active bound when the
      // pressure operator sees lighter liquid at the advected predictor.
      const double density_scale = std::sqrt(minimum_predictor_density) / std::sqrt(scene.liquid_density);
      auto& capillary_bound = *accepted_measurement.step_start_capillary_bound_seconds;
      capillary_bound *= density_scale;
      if (!(capillary_bound > 0.0)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      if (step_duration > capillary_bound) {
        return Error(absl::StatusCode::kAborted);
      }
    }
    for (std::size_t index = 0; index < frozen->faces.size(); ++index) {
      const auto& face = frozen->faces[index];
      const auto slot = face.face_slot;
      if (face.open_area > 0
          && (normalized->validity[slot] == LiquidFaceValidity3D::kMissing
              || normalized->validity[slot] == LiquidFaceValidity3D::kLowMass)) {
        STEP_TRY(backend.sample_faces(
            backend.context,
            grid,
            obstacle,
            staging.tags[slot].side_tag,
            {},
            staging.old_walls.values()
        ));
        if (oriented_box) {
          STEP_TRY(ApplyOrientedBoxFaceSamples(
              grid,
              floating_box->body,
              floating_box->half_extent,
              {},
              staging.old_walls.values()
          ));
        } else if (floating_box != nullptr) {
          STEP_TRY(ApplyBoxFaceSamples(grid, contact_box, {}, staging.old_walls.values()));
        }
        const LiquidMissingFacePredictionInputs3D prediction_inputs{
            .grid = grid,
            .old_history = old_history,
            .face_slot = slot,
            .side_tag = staging.tags[slot].side_tag,
            .world_acceleration = scene.world_acceleration,
            .step_duration = step_duration,
        };
        const auto predicted_face_velocity = backend.predict_missing_face(backend.context, prediction_inputs, boundary);
        if (!predicted_face_velocity.ok()) {
          return predicted_face_velocity.status();
        }
        // Preserve old validity: pressure prediction must not erase a low-mass
        // retention fallback. The backend applies acceleration once to continuation.
        staging.new_velocities[slot] = *predicted_face_velocity;
        if (ocean_boundary) {
          STEP_TRY(relax_face(slot));
        }
      }
      staging.incident_velocities[index] = staging.new_velocities[slot];
    }
    // Evaluate the signed wall flux at the frozen pose. Pressure link assembly
    // consumes the face-order borrow before the wetted-piston workspace is reused.
    std::span<const std::optional<double>> wetted_piston_face_fluxes;
    std::optional<WettedPistonInputs3D> piston_inputs;
    std::optional<WettedPistonView3D> piston_wetted;
    if (staging.wetted_piston.has_value()) {
      const WettedPistonInputs3D wetted_piston_inputs{
          .field = scene.field,
          .particles = trial.particles.values(),
          .frozen = *frozen,
          .tank = *tank_at_step_start,
      };
      const auto wetted = backend.wetted_piston(backend.context, wetted_piston_inputs, *staging.wetted_piston);
      if (!wetted.ok()) {
        return wetted.status();
      }
      if (wetted->face_fluxes.size() != frozen->faces.size()) {
        return Error(absl::StatusCode::kInvalidArgument);
      }
      wetted_piston_face_fluxes = wetted->face_fluxes;
      piston_inputs.emplace(wetted_piston_inputs);
      piston_wetted = *wetted;
    }
    auto links = backend.links(
        backend.context,
        *frozen,
        {
            .cells = staging.pressure_fields.values(),
            .predicted_face_velocities = staging.incident_velocities.values(),
            .ambient_pressure = scene.ambient_pressure,
            .wetted_piston_face_fluxes = wetted_piston_face_fluxes,
            .air_cell_pressures = cavity_boundaries.air_pressures,
            .internal_interfaces = internal_interfaces,
        },
        boundary
    );
    if (!links.ok()) {
      return links.status();
    }
    if (staging.isolated_drop) {
      STEP_TRY(liquid_step_internal::SetIsolatedDropPressureLifts(
          grid.geometry(),
          scene.surface_tension,
          scene.ambient_pressure,
          staging,
          *links
      ));
    }
    diagnostics.lower_theta_regularization_count = links->lower_theta_regularization_count;
    STEP_TRY(Enter(LiquidPhysicalStage3D::kViscosity, step_duration));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kPressure, step_duration));
    STEP_TRY(trial.cells.Resize(frozen->cells.size()));
    // An empty, successfully frozen geometry has no constraint rows. All other
    // stages still execute, including transfer/forces and backend particle update.
    std::optional<double> coupled_velocity;
    std::optional<double> coupled_box_speed;
    if ((coupled_piston || floating_box != nullptr) && links->cells.empty()) {
      return Error(absl::StatusCode::kFailedPrecondition);
    }
    if (!links->cells.empty()) {
      auto pressure_problem = *links;
      if (staging.drift_pressure_cells) {
        STEP_TRY(
            ApplyIncompressibleDrift(grid, step_duration, *scene.drift, staging, workspace.pressure(), pressure_problem)
        );
      }
      pressure_problem.seed_uniform_dirichlet_rest_pressure = exact_curvature.has_value();
      std::span<const double> projected_cell_pressures;
      std::span<const double> projected_face_velocities;
      MacPressureDiagnostics3D pressure_diagnostics;
      MacPressureFrozenMeasurement3D frozen_measurement;
      std::span<double> cell_dynamic_viscosity;
      if (uses_composition_viscosity) {
        if (pressure_problem.cells.size() != frozen->cells.size()) {
          return absl::InternalError("viscosity staging lost frozen cells");
        }
        STEP_TRY(staging.composition_viscosity->Resize(pressure_problem.cells.size()));
        cell_dynamic_viscosity = staging.composition_viscosity->values();
        for (std::size_t index = 0; index < pressure_problem.cells.size(); ++index) {
          if (CellKey(pressure_problem.cells[index].cell_index) != CellKey(frozen->cells[index].cell_index)) {
            return absl::InternalError("viscosity staging lost frozen cell order");
          }
          const auto& sensor = (*staging.composition_density)[index];
          const double viscosity = sensor.sampled_volume_m3 > 0.0
              ? sensor.sampled_viscosity_volume / sensor.sampled_volume_m3
              : sensor.extrapolated_viscosity;
          if (!std::isfinite(viscosity) || viscosity < 0.0) {
            return absl::OutOfRangeError("sampled liquid viscosity is unrepresentable");
          }
          cell_dynamic_viscosity[index] = viscosity;
        }
      }
      const bool active_viscosity = cell_dynamic_viscosity.empty()
          ? scene.dynamic_viscosity > 0.0
          : std::ranges::any_of(cell_dynamic_viscosity, [](double viscosity) {
              return viscosity > 0.0;
            });
      const auto box_pressure_base_rows = pressure_problem.cells;
      const auto visit_box_patches = [&](absl::FunctionRef<absl::Status(const RigidPressurePatch3D&)> visit) {
        const auto pressure_cell_at = [&](std::size_t ordinal) {
          return links->cells[ordinal].cell_index;
        };
        if (oriented_box) {
          return VisitOrientedBoxPressurePatches3D(
              floating_box->body,
              floating_box->half_extent,
              floating_box->oriented_geometry->liquid_height,
              grid.geometry(),
              links->cells.size(),
              pressure_cell_at,
              visit
          );
        }
        const auto& box = floating_box->geometry;
        return VisitFloatingBoxPressurePatches3D(
            floating_box->body.body_id,
            box.world_low,
            box.world_high,
            box.liquid_height,
            grid.geometry(),
            links->cells.size(),
            pressure_cell_at,
            visit
        );
      };
      const auto set_box_velocity = [&](Vec3 velocity, Vec3 angular_velocity = {}) -> absl::Status {
        STEP_TRY(staging.pressure_cell_scratch.Resize(links->cells.size()));
        const auto rows = staging.pressure_cell_scratch.values();
        std::ranges::copy(box_pressure_base_rows, rows.begin());
        STEP_TRY(visit_box_patches([&](const RigidPressurePatch3D& patch) {
          Vec3 wall_velocity = velocity;
          if (oriented_box) {
            auto body = floating_box->body;
            body.angular_velocity = angular_velocity;
            const auto arm =
                liquid_step_internal::Add(patch.world_centroid, liquid_step_internal::Scale(body.center, -1.0));
            wall_velocity = RigidSurfaceVelocity(body, arm, velocity);
          }
          const double normal_speed = liquid_step_internal::Dot(patch.outward_body_normal, wall_velocity);
          auto& flux = rows[patch.pressure_cell_index].integrated_inward_wall_flux;
          flux += patch.wetted_area * normal_speed;
          return std::isfinite(flux) ? absl::OkStatus() : Error(absl::StatusCode::kOutOfRange);
        }));
        pressure_problem.cells = rows;
        return absl::OkStatus();
      };
      const auto accumulate_pressure_patch =
          [&](RigidBody3D& body, std::span<const double> pressures, const RigidPressurePatch3D& patch) -> absl::Status {
        auto pressure_patch = patch;
        auto patch_pressures = pressures;
        double reconstructed_pressure = 0.0;
        double pressure_gradient_y = 0.0;
        const bool supports_hydrostatic_gradient = !uses_composition_density || scene.constituent_materials.size() == 1;
        if (oriented_box && scene.world_acceleration.y != 0.0 && !eos && supports_hydrostatic_gradient) {
          const auto geometry = grid.geometry();
          const auto cell = links->cells[patch.pressure_cell_index].cell_index;
          const auto centroid_to_cell = boundary_internal::HorizontalPlaneLatticeDistance(
              patch.world_centroid.y,
              {geometry.world_origin.y, static_cast<double>(cell.y) + 0.5, geometry.cell_width}
          );
          if (!centroid_to_cell.ok()) {
            return boundary_internal::ArithmeticStatus(centroid_to_cell.status(), true);
          }
          const auto& receiver = staging.pressure_fields[patch.pressure_cell_index];
          pressure_gradient_y = receiver.density.value_or(scene.liquid_density) * scene.world_acceleration.y;
          reconstructed_pressure =
              std::fma(-pressure_gradient_y, *centroid_to_cell, pressures[patch.pressure_cell_index]);
          patch_pressures = {&reconstructed_pressure, 1};
          pressure_patch.pressure_cell_index = 0;
        }
        const auto loaded = AccumulateRigidBodyPressure3D(
            body,
            patch_pressures,
            scene.ambient_pressure,
            {&pressure_patch, 1},
            pressure_gradient_y
        );
        if (!loaded.ok()) {
          return loaded.status();
        }
        body = *loaded;
        return absl::OkStatus();
      };
      const auto load_box = [&](std::span<const double> pressures, RigidBody3D body) -> absl::StatusOr<RigidBody3D> {
        STEP_TRY(visit_box_patches([&](const RigidPressurePatch3D& patch) {
          return accumulate_pressure_patch(body, pressures, patch);
        }));
        return body;
      };
      if (floating_box != nullptr) {
        STEP_TRY(set_box_velocity(floating_box->body.linear_velocity, floating_box->body.angular_velocity));
      }
      if ((coupled_piston || floating_box != nullptr) && active_viscosity) {
        return Error(absl::StatusCode::kFailedPrecondition);
      }
      // MSVC Debug vectors allocate iterator metadata even when empty.
      // Only the snow path owns these buffers.
      std::optional<std::vector<MacPressureFace3D>> snow_pressure_faces;
      std::optional<PlasticSolidPressureMobility3D> snow_mobility;
      std::optional<std::vector<double>> coupled_snow_flux;
      if (snow_samples && active_viscosity) {
        return absl::UnimplementedError("sampled snow pressure mobility requires inviscid liquid");
      }
      if (snow_samples) {
        auto mobility =
            PlasticSolidPressureMobility3D::Create(*frozen, *snow_samples, snow->inputs.accepted_particles, obstacle);
        if (!mobility.ok()) {
          return mobility.status();
        }
        const auto pressure_grid =
            SparseMacGrid3D::Create(grid.geometry(), grid.active_block_indices(), grid.active_block_indices().size());
        if (!pressure_grid.ok()) {
          return pressure_grid.status();
        }
        auto flux = mobility->SolveWallFlux(
            {*pressure_grid, step_duration, pressure_problem.cells, pressure_problem.faces},
            scene.ambient_pressure,
            options.pressure_solver_options.max_iterations
        );
        if (!flux.ok()) {
          return flux.status();
        }
        snow_mobility = std::move(*mobility);
        coupled_snow_flux.emplace(std::move(*flux));
        snow_pressure_faces.emplace(pressure_problem.faces.begin(), pressure_problem.faces.end());
        for (std::size_t index = 0; index < snow_pressure_faces->size(); ++index) {
          auto& face = (*snow_pressure_faces)[index];
          const auto wall_flux = (*coupled_snow_flux)[index];
          if (wall_flux != 0) {
            face.prescribed_wall_flux = face.prescribed_wall_flux.value_or(0) + wall_flux;
          }
        }
        pressure_problem.faces = *snow_pressure_faces;
      }
      if (!active_viscosity) {
        auto pressure_options = options.pressure_solver_options;
        if (oriented_box && floating_box->coupled) {
          const auto extent = floating_box->half_extent;
          const double full_displaced_mass = 8.0 * extent.x * extent.y * extent.z * scene.liquid_density;
          // A small rigid mass amplifies pressure residuals in the twist
          // solve. Refine the inner response in that inertia regime.
          const double residual_scale = floating_box->body.mass < 0.005 * full_displaced_mass ? 1e-6 : 1e-3;
          pressure_options.cpu_residual_tolerance_scale =
              std::min(pressure_options.cpu_residual_tolerance_scale, residual_scale);
        }
        auto projection = backend.project(
            backend.context,
            grid,
            step_duration,
            pressure_problem,
            pressure_options,
            workspace.pressure()
        );
        if (!projection.ok()) {
          return projection.status();
        }
        if (oriented_box && floating_box->coupled) {
          const auto& body = floating_box->body;
          constexpr std::size_t kTwistSize = 6;
          using Twist = std::array<double, kTwistSize>;
          const double length = grid.geometry().cell_width;
          const RigidOrientation3D inverse_orientation{
              body.orientation.w,
              liquid_step_internal::Scale(body.orientation.vector, -1.0)
          };
          // Translation uses world axes. Rotation uses the frozen principal
          // axes, with angular speed scaled by cell width for conditioning.
          const auto twist = [&](const RigidBody3D& value) -> absl::StatusOr<Twist> {
            const auto local = RotateRigidBodyVector3D(inverse_orientation, value.angular_velocity);
            if (!local.ok()) {
              return local.status();
            }
            const auto linear = value.linear_velocity;
            return Twist{linear.x, linear.y, linear.z, length * local->x, length * local->y, length * local->z};
          };
          const auto generalized_force = [&](const RigidBody3D& value) -> absl::StatusOr<Twist> {
            const auto torque = RotateRigidBodyVector3D(inverse_orientation, value.torque);
            if (!torque.ok()) {
              return torque.status();
            }
            return Twist{
                value.force.x,
                value.force.y,
                value.force.z,
                torque->x / length,
                torque->y / length,
                torque->z / length
            };
          };
          const auto world_angular_velocity = [&](const Twist& value) {
            return RotateRigidBodyVector3D(body.orientation, {value[3] / length, value[4] / length, value[5] / length});
          };
          const auto set_twist = [&](const Twist& value) -> absl::Status {
            const auto angular = world_angular_velocity(value);
            if (!angular.ok()) {
              return angular.status();
            }
            return set_box_velocity({value[0], value[1], value[2]}, *angular);
          };
          const auto project_twist = [&](const Twist& value) -> absl::Status {
            STEP_TRY(set_twist(value));
            projection = backend.project(
                backend.context,
                grid,
                step_duration,
                pressure_problem,
                pressure_options,
                workspace.pressure()
            );
            return projection.status();
          };
          const auto start = twist(body);
          const auto baseline = load_box(projection->cell_pressures, body);
          if (!start.ok()) {
            return start.status();
          }
          if (!baseline.ok()) {
            return baseline.status();
          }
          const auto baseline_force = generalized_force(*baseline);
          if (!baseline_force.ok()) {
            return baseline_force.status();
          }
          const auto inertia = body.principal_inertia;
          const Twist mass{
              body.mass,
              body.mass,
              body.mass,
              inertia.x / length / length,
              inertia.y / length / length,
              inertia.z / length / length
          };
          std::array<double, kTwistSize * kTwistSize> system{};
          const double probe_scale =
              std::max(1.0, liquid_step_internal::MaximumComponentMagnitude(body.linear_velocity));
          for (std::size_t column = 0; column < kTwistSize; ++column) {
            auto probe = *start;
            probe[column] += std::copysign(std::max(probe_scale, std::abs(probe[column])), probe[column]);
            const double increment = probe[column] - (*start)[column];
            if (!std::isfinite(increment)) {
              return Error(absl::StatusCode::kOutOfRange);
            }
            STEP_TRY(project_twist(probe));
            const auto load = load_box(projection->cell_pressures, body);
            if (!load.ok()) {
              return load.status();
            }
            const auto force = generalized_force(*load);
            if (!force.ok()) {
              return force.status();
            }
            for (std::size_t row = 0; row < kTwistSize; ++row) {
              system[row * kTwistSize + column] = -step_duration * ((*force)[row] - (*baseline_force)[row]) / increment;
            }
          }
          // Reciprocal pressure traction and wall flux give a symmetric
          // response. Average iterative projection noise, then check the
          // actual unsymmetrized fluid/body residual after reprojecting.
          for (std::size_t row = 0; row < kTwistSize; ++row) {
            system[row * kTwistSize + row] += mass[row];
            for (std::size_t column = 0; column < row; ++column) {
              const double symmetric =
                  std::midpoint(system[row * kTwistSize + column], system[column * kTwistSize + row]);
              system[row * kTwistSize + column] = symmetric;
              system[column * kTwistSize + row] = symmetric;
            }
          }
          auto load = *baseline;
          auto speed = *start;
          SmallSolveWorkspace solve_workspace;
          constexpr int kMaximumCorrections = 4;
          for (int iteration = 0;; ++iteration) {
            const auto drift = AdvanceRigidBody3D(load, step_duration);
            if (!drift.ok()) {
              return drift.status();
            }
            const auto expected = twist(drift->end);
            if (!expected.ok()) {
              return expected.status();
            }
            Twist residual{};
            for (std::size_t axis = 0; axis < kTwistSize; ++axis) {
              residual[axis] = mass[axis] * ((*expected)[axis] - speed[axis]);
            }
            Twist correction{};
            STEP_TRY(SolveSmallSpd(system, residual, correction, solve_workspace));
            bool converged = true;
            for (std::size_t axis = 0; axis < kTwistSize; ++axis) {
              converged &= std::abs(correction[axis]) <= 1e-8 * std::max(1.0, std::abs(speed[axis]));
            }
            if (converged && iteration > 0) {
              // The verified pressure projection already used this twist.
              // Carry it through the pose and particle sweep unchanged.
              auto candidate = *drift;
              const auto angular = world_angular_velocity(speed);
              if (!angular.ok()) {
                return angular.status();
              }
              candidate.end.linear_velocity = {speed[0], speed[1], speed[2]};
              candidate.end.angular_velocity = *angular;
              const auto pose = SampleRigidBodyPose3D(candidate, 1.0);
              if (!pose.ok()) {
                return pose.status();
              }
              candidate.end.center = pose->center;
              candidate.end.orientation = pose->orientation;
              record_oriented_pressure_load(load);
              oriented_box_candidate = candidate;
              break;
            }
            if (iteration == kMaximumCorrections) {
              return Error(absl::StatusCode::kAborted);
            }
            for (std::size_t axis = 0; axis < kTwistSize; ++axis) {
              speed[axis] += correction[axis];
            }
            STEP_TRY(project_twist(speed));
            const auto next_load = load_box(projection->cell_pressures, body);
            if (!next_load.ok()) {
              return next_load.status();
            }
            load = *next_load;
          }
        }
        if (floating_box != nullptr && floating_box->coupled && !oriented_box) {
          const auto& body = floating_box->body;
          const double start_speed = body.linear_velocity.y;
          auto pressure_only = body;
          pressure_only.force = {};
          pressure_only.torque = {};
          const auto initial_load = load_box(projection->cell_pressures, pressure_only);
          if (!initial_load.ok()) {
            return initial_load.status();
          }
          const double probe_speed = start_speed + 1.0;
          if (!std::isfinite(probe_speed) || probe_speed == start_speed) {
            return Error(absl::StatusCode::kOutOfRange);
          }
          STEP_TRY(set_box_velocity({0.0, probe_speed, 0.0}));
          projection = backend.project(
              backend.context,
              grid,
              step_duration,
              pressure_problem,
              options.pressure_solver_options,
              workspace.pressure()
          );
          if (!projection.ok()) {
            return projection.status();
          }
          const auto probe_load = load_box(projection->cell_pressures, pressure_only);
          if (!probe_load.ok()) {
            return probe_load.status();
          }
          const double added_mass =
              -step_duration * (probe_load->force.y - initial_load->force.y) / (probe_speed - start_speed);
          const double effective_mass = body.mass + added_mass;
          if (!std::isfinite(added_mass) || !std::isfinite(effective_mass) || added_mass < 0.0
              || effective_mass <= 0.0) {
            return Error(absl::StatusCode::kFailedPrecondition);
          }
          double speed = std::fma(step_duration, (body.force.y + initial_load->force.y) / effective_mass, start_speed);
          // Each pressure solve satisfies the liquid residual tolerance. Check
          // the resulting body impulse too, scaled by the coupled inertia.
          // Correct the remaining wall-speed error on the same frozen domain.
          constexpr int kMaximumCouplingProjections = 4;
          for (int iteration = 0; iteration < kMaximumCouplingProjections; ++iteration) {
            if (!std::isfinite(speed)) {
              return Error(absl::StatusCode::kOutOfRange);
            }
            STEP_TRY(set_box_velocity({0.0, speed, 0.0}));
            projection = backend.project(
                backend.context,
                grid,
                step_duration,
                pressure_problem,
                options.pressure_solver_options,
                workspace.pressure()
            );
            if (!projection.ok()) {
              return projection.status();
            }
            const auto final_load = load_box(projection->cell_pressures, pressure_only);
            if (!final_load.ok()) {
              return final_load.status();
            }
            const double force = body.force.y + final_load->force.y;
            const double impulse_error = std::fma(step_duration, force, body.mass * (start_speed - speed));
            const double speed_correction = impulse_error / effective_mass;
            if (!std::isfinite(speed_correction)) {
              return Error(absl::StatusCode::kOutOfRange);
            }
            if (std::abs(speed_correction) <= 1e-8 * std::max(1.0, std::abs(speed))) {
              break;
            }
            if (iteration + 1 == kMaximumCouplingProjections) {
              return Error(absl::StatusCode::kAborted);
            }
            speed += speed_correction;
          }
          floating_box->added_mass_kg = added_mass;
          coupled_box_speed = speed;
          end_box.world_low.y = std::fma(step_duration, speed, contact_box.world_low.y);
          end_box.world_high.y = std::fma(step_duration, speed, contact_box.world_high.y);
          end_box.wall_velocity.y = speed;
        }
        if (coupled_piston) {
          auto& scratch = *staging.pressure_face_scratch;
          STEP_TRY(scratch.Resize(links->faces.size()));
          const auto coupled_faces = scratch.values();
          std::ranges::copy(links->faces, coupled_faces.begin());
          pressure_problem.faces = coupled_faces;

          // Receiver area is aggregated once in the wetted geometry. Attached
          // physical patches contribute their area through the same receiver.
          const auto set_wall_speed = [&](double speed) -> absl::Status {
            for (std::size_t face_index = 0; face_index < coupled_faces.size(); ++face_index) {
              if (piston_wetted->face_fluxes[face_index].has_value()) {
                auto& face = coupled_faces[face_index];
                face.wall_normal_velocity = speed;
                face.prescribed_wall_flux = 0.0;
              }
            }
            for (const auto& patch : piston_wetted->patches) {
              if (patch.receiver_area.upper > 0.0) {
                if (!patch.receiver_face_ordinal || *patch.receiver_face_ordinal >= coupled_faces.size()
                    || !piston_wetted->face_fluxes[*patch.receiver_face_ordinal].has_value()) {
                  return Error(absl::StatusCode::kFailedPrecondition);
                }
                const double area = std::midpoint(patch.receiver_area.lower, patch.receiver_area.upper);
                coupled_faces[*patch.receiver_face_ordinal].prescribed_wall_flux = area * speed;
              }
            }
            return absl::OkStatus();
          };
          auto pressure_only_body = *piston_load;
          pressure_only_body.force = {};
          pressure_only_body.torque = {};
          const auto load_at = [&](std::span<const double> pressures) {
            return AccumulatePlanarWettedPistonPressure3D(
                *piston_inputs,
                *piston_wetted,
                pressures,
                scene.ambient_pressure,
                pressure_only_body
            );
          };
          const auto initial_load = load_at(projection->cell_pressures);
          if (!initial_load.ok()) {
            return initial_load.status();
          }
          const double initial_speed = piston_load->linear_velocity.x;
          const double speed_increment =
              std::copysign(0.125 * grid.geometry().cell_width / step_duration, initial_speed);
          const double probe_speed = initial_speed + speed_increment;
          const double probe_delta = probe_speed - initial_speed;
          if (!std::isfinite(probe_speed) || !std::isfinite(probe_delta) || probe_delta == 0.0) {
            return Error(absl::StatusCode::kOutOfRange);
          }
          STEP_TRY(set_wall_speed(probe_speed));
          projection = backend.project(
              backend.context,
              grid,
              step_duration,
              pressure_problem,
              options.pressure_solver_options,
              workspace.pressure()
          );
          if (!projection.ok()) {
            return projection.status();
          }
          const auto probe_load = load_at(projection->cell_pressures);
          if (!probe_load.ok()) {
            return probe_load.status();
          }

          // Frozen linear pressure response gives the scalar Schur complement
          // for the axial body degree of freedom, regardless of pressure-cell count.
          const double added_mass = -step_duration * (probe_load->force.x - initial_load->force.x) / probe_delta;
          const double effective_mass = piston_load->mass + added_mass;
          const double total_force = piston_load->force.x + initial_load->force.x;
          const double speed = std::fma(step_duration, total_force / effective_mass, initial_speed);
          if (added_mass < 0.0 || !std::isfinite(speed) || !std::isfinite(effective_mass)) {
            return Error(absl::StatusCode::kOutOfRange);
          }
          auto& tank = std::get<AxisAlignedLiquidTank3D>(obstacle);
          tank.piston_velocity_x = speed;
          STEP_TRY(boundary_internal::ArithmeticStatus(boundary_internal::CheckTankStep(tank, step_duration), true));
          new_obstacle = liquid_step_internal::ObstacleAtTime(obstacle, step_duration);
          accepted_measurement.piston_pose->step_end_low_x_m =
              std::get<AxisAlignedLiquidTank3D>(new_obstacle).start_low_x;
          STEP_TRY(set_wall_speed(speed));
          projection = backend.project(
              backend.context,
              grid,
              step_duration,
              pressure_problem,
              options.pressure_solver_options,
              workspace.pressure()
          );
          if (!projection.ok()) {
            return projection.status();
          }
          coupled_velocity = speed;
        }
        projected_cell_pressures = projection->cell_pressures;
        projected_face_velocities = projection->corrected_face_velocities;
        pressure_diagnostics = projection->diagnostics;
        frozen_measurement = projection->frozen_measurement;
      } else {
        auto result = backend.project_viscous(
            backend.context,
            geometry_inputs,
            pressure_problem,
            scene.dynamic_viscosity,
            cell_dynamic_viscosity,
            options.pressure_solver_options,
            workspace.pressure()
        );
        if (!result.ok()) {
          return result.status();
        }
        const auto& coupled = *result;
        accepted_measurement.viscous_max_momentum_residual_kg_m_per_s = coupled.max_momentum_residual;
        projected_cell_pressures = coupled.cell_pressures;
        projected_face_velocities = coupled.face_velocities;
        pressure_diagnostics = {
            .iteration_count = coupled.pressure_iterations,
            .true_residual_l2 = coupled.constraint_residual_l2,
            .max_volume_scaled_residual = coupled.max_cell_constraint_defect,
            .max_cell_constraint_defect = coupled.max_cell_constraint_defect,
        };
        frozen_measurement.integrated_rhs_l2_m3_per_s = coupled.initial_constraint_residual_l2;
        frozen_measurement.inward_prescribed_wall_flux_m3_per_s = coupled.inward_prescribed_wall_flux_m3_per_s;
        for (const auto& cell : pressure_problem.cells) {
          frozen_measurement.pressure_cell_volume_m3 += cell.fluid_volume;
        }
      }
      if (projected_cell_pressures.size() != frozen->cells.size()
          || projected_face_velocities.size() != frozen->faces.size()
          || projected_face_velocities.size() != pressure_problem.faces.size()) {
        return absl::InternalError("pressure projection extents do not match frozen geometry");
      }
      if (snow_samples) {
        STEP_TRY(snow_mobility->VerifyProjectedFlux(
            projected_cell_pressures,
            scene.ambient_pressure,
            *coupled_snow_flux,
            frozen->cells
        ));
        auto staged = StagePlasticSolidPressureLoads3D(
            *frozen,
            *snow_samples,
            projected_cell_pressures,
            scene.ambient_pressure,
            obstacle
        );
        if (!staged.ok()) {
          return staged.status();
        }
        snow_loads = std::move(*staged);
      }
      if (eos) {
        for (const double pressure : projected_cell_pressures) {
          const auto accepted_eos = EvaluateLiquidEosAtPressure3D(*eos, pressure);
          if (!accepted_eos.ok()) {
            return accepted_eos.status();
          }
        }
      }
      if (piston_load != nullptr) {
        const auto loaded = AccumulatePlanarWettedPistonPressure3D(
            *piston_inputs,
            *piston_wetted,
            projected_cell_pressures,
            scene.ambient_pressure,
            *piston_load
        );
        if (!loaded.ok()) {
          return loaded.status();
        }
        *piston_load = *loaded;
      }
      if (floating_box != nullptr && !oriented_box_candidate) {
        const auto loaded = load_box(projected_cell_pressures, floating_box->body);
        if (!loaded.ok()) {
          return loaded.status();
        }
        if (oriented_box) {
          const auto candidate = AdvanceRigidBody3D(*loaded, step_duration);
          if (!candidate.ok()) {
            return candidate.status();
          }
          record_oriented_pressure_load(*loaded);
          oriented_box_candidate = *candidate;
        } else {
          floating_box->body = *loaded;
        }
      }
      if (oriented_box_candidate) {
        STEP_TRY(CheckOrientedBoxSweepInTank(
            *oriented_box_candidate,
            floating_box->half_extent,
            *tank_at_step_start,
            grid.geometry().cell_width
        ));
      }
      if (oriented_box) {
        constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
        Vec3 grid_impulse{};
        STEP_TRY(staging.face_pressure_impulses.Resize(grid.face_count()));
        std::ranges::fill(staging.face_pressure_impulses.values(), 0.0);
        const bool reconstruct_hydrostatic_pressure =
            !eos && (!uses_composition_density || scene.constituent_materials.size() == 1);
        const double hydrostatic_velocity_correction =
            reconstruct_hydrostatic_pressure ? -step_duration * scene.world_acceleration.y : 0.0;
        for (std::size_t index = 0; index < pressure_problem.faces.size(); ++index) {
          const auto& face = pressure_problem.faces[index];
          if (face.open_area == 0.0) {
            continue;
          }
          const double dual_mass = face.open_area * face.pressure_gradient_distance / face.inverse_density;
          const double velocity_change = projected_face_velocities[index] - face.predicted_normal_velocity;
          const auto component = kComponents[static_cast<std::size_t>(face.face_index.axis)];
          const double impulse = dual_mass * velocity_change;
          grid_impulse.*component += impulse;
          const double hydrostatic_correction =
              face.face_index.axis == FaceNormalAxis::kY ? hydrostatic_velocity_correction : 0.0;
          staging.face_pressure_impulses[frozen->faces[index].face_slot] =
              dual_mass * (velocity_change - hydrostatic_correction);
        }
        if (!liquid_step_internal::Finite(grid_impulse)) {
          return Error(absl::StatusCode::kOutOfRange);
        }
        floating_box->oriented_loads->pressure_grid_impulse = grid_impulse;
        if (!trial.particles.values().empty()) {
          STEP_TRY(staging.pressure_transfer_mass_scratch.Resize(grid.face_count()));
          STEP_TRY(staging.pressure_particle_impulses.Resize(trial.particles.size()));
          const auto distribute = [&]() {
            return DistributeCompatiblePressureImpulse3D(
                grid,
                staging.particles.values(),
                deposits->compatible_stencils,
                staging.face_pressure_impulses.values(),
                staging.pressure_transfer_mass_scratch.values(),
                staging.pressure_particle_impulses.values()
            );
          };
          auto distribution = distribute();
          if (distribution.code() == absl::StatusCode::kFailedPrecondition) {
            const auto routed = RouteUnsupportedPressureImpulse(
                grid,
                pressure_problem,
                frozen->faces,
                staging.pressure_transfer_mass_scratch.values(),
                staging.face_pressure_impulses.values()
            );
            if (!routed.ok()) {
              return routed.status();
            }
            distribution = distribute();
            if (distribution.ok()) {
              floating_box->oriented_loads->routed_pressure_faces = routed->face_count;
              floating_box->oriented_loads->routed_pressure_angular_impulse_change = routed->angular_impulse_change;
            }
          }
          if (distribution.ok()) {
            use_compatible_pressure_transfer = true;
            STEP_TRY(staging.prepressure_velocities.Resize(grid.face_count()));
            STEP_TRY(staging.pressure_stored_velocities.Resize(trial.particles.size()));
            std::ranges::copy(staging.new_velocities.values(), staging.prepressure_velocities.values().begin());
            for (std::size_t index = 0; index < pressure_problem.faces.size(); ++index) {
              const auto& face = pressure_problem.faces[index];
              const double hydrostatic_correction = face.open_area > 0.0 && face.face_index.axis == FaceNormalAxis::kY
                  ? hydrostatic_velocity_correction
                  : 0.0;
              staging.prepressure_velocities[frozen->faces[index].face_slot] =
                  face.predicted_normal_velocity + hydrostatic_correction;
            }
            Vec3 distributed_impulse{};
            for (const auto impulse : staging.pressure_particle_impulses.values()) {
              distributed_impulse = liquid_step_internal::Add(distributed_impulse, impulse);
            }
            floating_box->oriented_loads->particle_pressure_impulse = distributed_impulse;
          } else {
            return distribution;
          }
        }

        // Use the same checked traction accumulator for a fixed surface. The
        // zero center makes its accumulated moment relative to the world origin.
        constexpr std::uint64_t kTankPatchId = 1;
        RigidBody3D tank_load{.body_id = kTankPatchId, .mass = 1.0, .principal_inertia = {1.0, 1.0, 1.0}};
        const auto status = VisitStationaryTankPressurePatches3D(
            kTankPatchId,
            *tank_at_step_start,
            floating_box->oriented_geometry->liquid_height,
            grid.geometry(),
            links->cells.size(),
            [&](std::size_t ordinal) {
              return links->cells[ordinal].cell_index;
            },
            [&](const RigidPressurePatch3D& patch) {
              return accumulate_pressure_patch(tank_load, projected_cell_pressures, patch);
            }
        );
        if (status.ok()) {
          floating_box->oriented_loads->tank_pressure = TankPressureLoad3D{
              .force = liquid_step_internal::Scale(tank_load.force, -1.0),
              .torque = liquid_step_internal::Scale(tank_load.torque, -1.0),
          };
        } else if (status.code() != absl::StatusCode::kFailedPrecondition) {
          return status;
        }
      }
      diagnostics.pressure_diagnostics = pressure_diagnostics;
      accepted_measurement.pressure = pressure_diagnostics;
      accepted_measurement.pressure_rhs_l2_m3_per_s = frozen_measurement.integrated_rhs_l2_m3_per_s;
      accepted_measurement.step_start_pressure_cell_volume_m3 = frozen_measurement.pressure_cell_volume_m3;
      if (accepted_measurement.piston_pose.has_value()) {
        accepted_measurement.step_start_inward_piston_flux_m3_per_s =
            frozen_measurement.inward_prescribed_wall_flux_m3_per_s;
      }
      for (std::size_t index = 0; index < frozen->cells.size(); ++index) {
        trial.cells[index] = {
            frozen->cells[index].cell_index,
            LiquidCellOccupancy3D::kLiquid,
            projected_cell_pressures[index],
        };
      }
      for (std::size_t index = 0; index < frozen->faces.size(); ++index) {
        const auto slot = frozen->faces[index].face_slot;
        staging.new_velocities[slot] = projected_face_velocities[index];
        if (staging.validity[slot] == LiquidFaceValidity3D::kMissing && frozen->faces[index].open_area > 0) {
          staging.validity[slot] = LiquidFaceValidity3D::kPredicted;
        }
      }
    }
    SetWorkPhase(work_phase, LiquidWorkPhase3D::kGatherContactAcceptance);
    STEP_TRY(Enter(LiquidPhysicalStage3D::kNewSnapshot, step_duration));
    STEP_TRY(backend.sample_faces(backend.context, grid, new_obstacle, 0, staging.new_tags.values(), {}));
    if (oriented_box) {
      STEP_TRY(ApplyOrientedBoxFaceSamples(
          grid,
          oriented_box_candidate->end,
          floating_box->half_extent,
          staging.new_tags.values(),
          {}
      ));
    } else if (floating_box != nullptr) {
      STEP_TRY(ApplyBoxFaceSamples(grid, end_box, staging.new_tags.values(), {}));
    }
    for (std::size_t index = 0; index < trial.faces.size(); ++index) {
      trial.faces[index].carrier_velocity = staging.new_velocities[index];
      trial.faces[index].validity = staging.validity[index];
      trial.faces[index].side_tag = staging.new_tags[index].side_tag;
      trial.faces[index].structurally_eligible = staging.new_tags[index].structurally_eligible;
    }
    std::span<const LiquidFaceValidity3D> projected_gather_validity;
    if (staging.projected_gather_validity && !links->cells.empty()) {
      auto validity = staging.projected_gather_validity->values();
      // Preserve the committed carrier and old history. In the new history,
      // unprojected usable faces must use continuation from projected donors.
      for (std::size_t slot = 0; slot < validity.size(); ++slot) {
        const auto carrier_validity = staging.validity[slot];
        const bool usable =
            carrier_validity == LiquidFaceValidity3D::kValid || carrier_validity == LiquidFaceValidity3D::kPredicted;
        const bool unprojected_donor = staging.new_tags[slot].structurally_eligible && usable;
        validity[slot] = unprojected_donor ? LiquidFaceValidity3D::kMissing : carrier_validity;
      }
      for (const auto& face : frozen->faces) {
        validity[face.face_slot] = staging.validity[face.face_slot];
      }
      projected_gather_validity = validity;
    }
    const LiquidBoundaryVelocityHistory3D new_history{
        .velocity_sample_time = new_time,
        .wall_sample_time = new_time,
        .face_velocities = staging.new_velocities.values(),
        .validity = staging.validity.values(),
        .face_tags = staging.new_tags.values(),
        .wall_samples = staging.new_walls.values(),
    };
    STEP_TRY(Enter(LiquidPhysicalStage3D::kGather, step_duration));
    // Thin-slab stencil nodes can be shared by opposite incident walls. Reuse
    // the two frozen wall histories only within a consecutive run on one side.
    std::optional<std::int64_t> sampled_wall_side;
    std::span<const LiquidBoundaryParticleGather3D> pending_gathers;
    std::size_t attempted_run_end = 0;
    bool attempted_run_uses_projected_history = false;
    std::size_t field_chunk_end = 0;
    bool field_chunk_ready = false;
    bool continue_isolated_projection = false;
    for (std::size_t index = 0; index < trial.particles.size(); ++index) {
      const auto& particle = trial.particles[index];
      const auto side = particle.side_tag;
      if (sampled_wall_side != side) {
        STEP_TRY(backend.sample_faces(backend.context, grid, obstacle, side, {}, staging.old_walls.values()));
        STEP_TRY(backend.sample_faces(backend.context, grid, new_obstacle, side, {}, staging.new_walls.values()));
        if (oriented_box) {
          STEP_TRY(ApplyOrientedBoxFaceSamples(
              grid,
              floating_box->body,
              floating_box->half_extent,
              {},
              staging.old_walls.values()
          ));
          STEP_TRY(ApplyOrientedBoxFaceSamples(
              grid,
              oriented_box_candidate->end,
              floating_box->half_extent,
              {},
              staging.new_walls.values()
          ));
        } else if (floating_box != nullptr) {
          STEP_TRY(ApplyBoxFaceSamples(grid, contact_box, {}, staging.old_walls.values()));
          STEP_TRY(ApplyBoxFaceSamples(grid, end_box, {}, staging.new_walls.values()));
        }
        sampled_wall_side = side;
        if (staging.isolated_drop && !projected_gather_validity.empty()) {
          auto projected_history = new_history;
          projected_history.validity = projected_gather_validity;
          continue_isolated_projection = CanContinueIsolatedDropProjection(old_history, projected_history);
        }
      }
      bool use_projected_history = false;
      if (!projected_gather_validity.empty()) {
        if (index < attempted_run_end) {
          use_projected_history = attempted_run_uses_projected_history;
        } else {
          const auto selected = UsesProjectedGatherHistory(*frozen, particle.world_position);
          if (!selected.has_value()) {
            return Error(absl::StatusCode::kOutOfRange);
          }
          use_projected_history = *selected;
        }
      }
      LiquidBoundaryGatherViewInputs3D gather_inputs{
          .mac_grid = grid,
          .particles = staging.particles.values().subspan(index, 1),
          .old_time = old_time,
          .new_time = new_time,
          .old_history = old_history,
          .new_history = new_history,
      };
      if (use_projected_history) {
        gather_inputs.new_history.validity = projected_gather_validity;
      }
      if (use_compatible_pressure_transfer) {
        auto baseline_inputs = gather_inputs;
        baseline_inputs.new_history.face_velocities = staging.prepressure_velocities.values();
        const auto baseline = backend.gather(backend.context, baseline_inputs, boundary);
        if (!baseline.ok()) {
          return baseline.status();
        }
        if (baseline->particles.size() != 1) {
          return Error(absl::StatusCode::kInternal);
        }
        const auto& sample = baseline->particles.front();
        const bool fallback = sample.old_pic_fallback || sample.new_pic_fallback || sample.low_mass_fallback;
        const double alpha = particle.pic_initialized && !fallback ? 0.99 * options.particle_energy_blend : 0.0;
        const auto prepressure_velocity = liquid_step_internal::Add(
            sample.new_pic_velocity,
            liquid_step_internal::Scale(sample.particle_minus_old_pic_velocity, alpha)
        );
        staging.pressure_stored_velocities[index] = liquid_step_internal::Add(
            prepressure_velocity,
            liquid_step_internal::Scale(staging.pressure_particle_impulses[index], 1.0 / particle.mass)
        );
      }
      const bool continues_projected_change = use_projected_history && continue_isolated_projection;
      if (backend.gather_run && index >= attempted_run_end) {
        attempted_run_end = index + 1;
        attempted_run_uses_projected_history = use_projected_history;
        while (attempted_run_end < trial.particles.size() && trial.particles[attempted_run_end].side_tag == side) {
          if (!projected_gather_validity.empty()) {
            const auto& next = trial.particles[attempted_run_end];
            const auto selected = UsesProjectedGatherHistory(*frozen, next.world_position);
            // A speculative lookup must not report an error before earlier gathers or field samples.
            if (!selected.has_value() || *selected != use_projected_history) {
              break;
            }
          }
          ++attempted_run_end;
        }
        const auto run_count = attempted_run_end - index;
        if (run_count > 1) {
          auto run_inputs = gather_inputs;
          run_inputs.particles = staging.particles.values().subspan(index, run_count);
          const auto run = continues_projected_change
              ? GatherIsolatedDropHistories(
                    backend,
                    *backend.gather_run,
                    run_inputs,
                    boundary,
                    staging.isolated_drop_restricted_gathers->values().subspan(index, run_count),
                    staging.gathered.values().subspan(index, run_count)
                )
              : backend.gather_run(backend.context, run_inputs, boundary);
          if (run.ok()) {
            if (run->particles.size() != run_count) {
              return Error(absl::StatusCode::kInternal);
            }
            pending_gathers = run->particles;
          } else if (run.status().code() == absl::StatusCode::kInternal) {
            return run.status();
          }
          // Replay a reusable rejection so an earlier scalar field error wins.
        }
      }
      const bool used_grouped_gather = !pending_gathers.empty();
      if (used_grouped_gather) {
        const auto& gathered = pending_gathers.front();
        if (continues_projected_change) {
          STEP_TRY(CorrectIsolatedDropProjectedChange(
              staging.gathered[index],
              (*staging.isolated_drop_restricted_gathers)[index],
              gathered,
              staging.gathered[index]
          ));
        } else {
          staging.gathered[index] = gathered;
        }
        diagnostics.old_pic_fallback_count += gathered.old_pic_fallback;
        diagnostics.new_pic_fallback_count += gathered.new_pic_fallback;
        pending_gathers = pending_gathers.subspan(1);
      } else {
        const auto gather = continues_projected_change
            ? GatherIsolatedDropProjectedChange(backend, gather_inputs, boundary, staging.gathered[index])
            : backend.gather(backend.context, gather_inputs, boundary);
        if (!gather.ok()) {
          return gather.status();
        }
        staging.gathered[index] = gather->particles[0];
        diagnostics.old_pic_fallback_count += gather->old_pic_fallback_particle_count;
        diagnostics.new_pic_fallback_count += gather->new_pic_fallback_particle_count;
      }
      if (staging.isolated_drop) {
        const auto distance = staging.isolated_drop->ParticleSignedDistance(particle.world_position);
        if (!distance.ok()) {
          return distance.status();
        }
        staging.particle_distances[index] = *distance;
        continue;
      }
      if (used_grouped_gather && backend.sample_gathered_fields && index >= field_chunk_end) {
        const auto field_count = std::min(kLiquidFieldBatchCapacity3D, attempted_run_end - index);
        field_chunk_end = index + field_count;
        field_chunk_ready = false;
        if (field_count > 1) {
          std::array<Vec3, kLiquidFieldBatchCapacity3D> field_positions;
          for (std::size_t offset = 0; offset < field_count; ++offset) {
            field_positions[offset] = trial.particles[index + offset].world_position;
          }
          const auto distances = staging.particle_distances.values().subspan(index, field_count);
          const auto status = backend.sample_gathered_fields(
              backend.context,
              scene.field,
              trial.particles.values(),
              std::span(field_positions).first(field_count),
              distances
          );
          if (status.code() == absl::StatusCode::kInternal) {
            return status;
          }
          // Batch validation can see a later query failure before evaluating
          // an earlier query. Replay this chunk in scalar order on rejection.
          field_chunk_ready = status.ok() && std::ranges::all_of(distances, [](double distance) {
                                return std::isfinite(distance);
                              });
        }
      }
      if (field_chunk_ready && index < field_chunk_end) {
        continue;
      }
      const auto distance =
          backend.sample_field(backend.context, scene.field, trial.particles.values(), particle.world_position);
      if (!distance.ok()) {
        return distance.status();
      }
      if (!std::isfinite(*distance)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      staging.particle_distances[index] = *distance;
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kParticleUpdate, step_duration));
    if (staging.regular_transport && floating_box == nullptr) {
      const auto& tank = std::get<AxisAlignedLiquidTank3D>(obstacle);
      for (std::size_t index = 0; index < trial.particles.size(); ++index) {
        (*staging.regular_transport)[index] =
            RegularTransportSample(*frozen, *links, staging, tank, trial.particles[index]);
      }
    }
    const LiquidParticleUpdateInputs3D particle_update_inputs{
        .gathered = staging.gathered.values(),
        .liquid_distances = staging.particle_distances.values(),
        .obstacle = obstacle,
        .cell_width = trial.geometry.cell_width,
        .step_duration = step_duration,
        .particle_energy_blend = options.particle_energy_blend,
        .friction_coefficient = scene.friction_coefficient,
        .penetration_tolerance = scene.penetration_tolerance,
        .regular_cell_transport = staging.regular_transport && floating_box == nullptr
            ? staging.regular_transport->values()
            : std::span<const LiquidRegularCellTransport3D>{},
    };
    double piston_contact_impulse = 0.0;
    Vec3 box_contact_impulse{};
    if (dynamic_piston && trial.particles.size() != 0) {
      const auto impulse = update_with_reaction == nullptr
          ? UpdateCpuLiquidParticlesWithPistonReaction3D(particle_update_inputs, trial.particles.values())
          : (*update_with_reaction)(particle_update_inputs, trial.particles.values());
      if (!impulse.ok()) {
        return impulse.status();
      }
      piston_contact_impulse = *impulse;
    } else if (oriented_box && trial.particles.size() != 0) {
      const auto impulse = UpdateCpuLiquidParticlesWithRotatingBoxImpulse3D(
          particle_update_inputs,
          *oriented_box_candidate,
          floating_box->half_extent,
          trial.particles.values(),
          use_compatible_pressure_transfer ? staging.pressure_stored_velocities.values() : std::span<const Vec3>{}
      );
      if (!impulse.ok()) {
        return impulse.status();
      }
      floating_box->oriented_loads->particle_impulse = impulse->linear;
      floating_box->oriented_loads->particle_angular_impulse = impulse->angular;
      floating_box->oriented_loads->liquid_gather_impulse = impulse->liquid_gather_impulse;
      floating_box->oriented_loads->tank_particle_impulse = impulse->tank_particle_impulse;
      auto& body = oriented_box_candidate->end;
      body.linear_velocity = liquid_step_internal::Add(
          body.linear_velocity,
          liquid_step_internal::Scale(impulse->linear, 1.0 / body.mass)
      );
      const auto orientation = body.orientation;
      const auto local_impulse = RotateRigidBodyVector3D(
          {orientation.w, liquid_step_internal::Scale(orientation.vector, -1.0)},
          impulse->angular
      );
      if (!local_impulse.ok()) {
        return local_impulse.status();
      }
      const auto inertia = body.principal_inertia;
      const auto angular_change = RotateRigidBodyVector3D(
          orientation,
          {local_impulse->x / inertia.x, local_impulse->y / inertia.y, local_impulse->z / inertia.z}
      );
      if (!angular_change.ok()) {
        return angular_change.status();
      }
      body.angular_velocity = liquid_step_internal::Add(body.angular_velocity, *angular_change);
      STEP_TRY(ValidateRigidBody3D(body));
    } else if (floating_box != nullptr && trial.particles.size() != 0) {
      auto swept_box = contact_box;
      swept_box.wall_velocity.y = coupled_box_speed.value_or(contact_box.wall_velocity.y);
      const auto impulse =
          UpdateCpuLiquidParticlesWithBoxImpulse3D(particle_update_inputs, swept_box, trial.particles.values());
      if (!impulse.ok()) {
        return impulse.status();
      }
      box_contact_impulse = *impulse;
    } else {
      STEP_TRY(backend.update_particles(backend.context, particle_update_inputs, trial.particles.values()));
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kWhitewater, step_duration));
    STEP_TRY(Enter(LiquidPhysicalStage3D::kValidate, step_duration));
    const double wall_displacement_bound =
        step_duration * liquid_step_internal::MaximumComponentMagnitude(liquid_step_internal::WallVelocity(obstacle));
    if (wall_displacement_bound > 0.5 * trial.geometry.cell_width) {
      return Error(absl::StatusCode::kAborted);
    }
    const auto retained = transaction.committed().blocks.values();
    for (auto& block : trial.blocks.values()) {
      const auto key = [](BlockIndex3D v) {
        return std::tuple(v.z, v.y, v.x);
      };
      const auto found =
          std::lower_bound(retained.begin(), retained.end(), block.block_index, [&](const auto& old, auto index) {
            return key(old.block_index) < key(index);
          });
      if (found != retained.end() && key(found->block_index) == key(block.block_index)) {
        if (block.activation_age == UINT32_MAX) {
          return Error(absl::StatusCode::kOutOfRange);
        }
        ++block.activation_age;
      }
    }
    STEP_TRY(ValidateLiquidState3D(trial));
    if (dynamic_piston) {
      const double axial_acceleration = piston_load->force.x / piston_load->mass;
      const double next_velocity = std::fma(step_duration, axial_acceleration, piston_load->linear_velocity.x)
          + piston_contact_impulse / piston_load->mass;
      if (!std::isfinite(next_velocity)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      piston_load->center.x = std::get<AxisAlignedLiquidTank3D>(new_obstacle).start_low_x;
      if (coupled_velocity
          && std::abs(next_velocity - *coupled_velocity) > 1e-8 * std::max(1.0, std::abs(next_velocity))) {
        return Error(absl::StatusCode::kFailedPrecondition);
      }
      piston_load->linear_velocity.x = coupled_velocity.value_or(next_velocity);
      piston_load->force = {};
      piston_load->torque = {};
    }
    if (oriented_box) {
      floating_box->body = oriented_box_candidate->end;
    } else if (floating_box != nullptr) {
      auto& body = floating_box->body;
      auto drift_velocity = body.linear_velocity;
      drift_velocity.y = coupled_box_speed.value_or(drift_velocity.y);
      const auto displacement = liquid_step_internal::Scale(drift_velocity, step_duration);
      auto next_center = body.center;
      auto next_velocity = body.linear_velocity;
      constexpr std::array kComponents{&Vec3::x, &Vec3::y, &Vec3::z};
      for (const auto component : kComponents) {
        if (floating_box->vertical_guide && component != &Vec3::y) {
          continue;
        }
        next_center.*component = std::fma(step_duration, drift_velocity.*component, body.center.*component);
        const double pressure_velocity = coupled_box_speed.has_value()
            ? *coupled_box_speed
            : std::fma(step_duration, (body.force.*component) / body.mass, body.linear_velocity.*component);
        next_velocity.*component = pressure_velocity + (box_contact_impulse.*component) / body.mass;
      }
      const auto half_extent = floating_box->half_extent;
      const auto next_low = liquid_step_internal::Add(next_center, liquid_step_internal::Scale(half_extent, -1.0));
      const auto next_high = liquid_step_internal::Add(next_center, half_extent);
      if (!liquid_step_internal::Finite(displacement) || !liquid_step_internal::Finite(next_velocity)
          || !liquid_step_internal::Finite(next_low) || !liquid_step_internal::Finite(next_high)) {
        return Error(absl::StatusCode::kOutOfRange);
      }
      const auto& tank = *tank_at_step_start;
      if (liquid_step_internal::MaximumComponentMagnitude(displacement) > 0.5 * grid.geometry().cell_width
          || std::min(next_low.x, end_box.world_low.x) <= tank.start_low_x
          || std::max(next_high.x, end_box.world_high.x) >= tank.high_x
          || std::min(next_low.y, end_box.world_low.y) <= tank.floor_y
          || std::min(next_low.z, end_box.world_low.z) <= tank.low_z
          || std::max(next_high.z, end_box.world_high.z) >= tank.high_z) {
        return Error(absl::StatusCode::kAborted);
      }
      body.center = next_center;
      body.linear_velocity = next_velocity;
      body.force = {};
      body.torque = {};
    }
    if (snow != nullptr) {
      STEP_TRY(snow->Advance(step_duration, obstacle, snow_loads.values(), transaction.committed(), trial));
    }
    STEP_TRY(Enter(LiquidPhysicalStage3D::kAccept, step_duration));
    return absl::OkStatus();
  }
  absl::StatusOr<double> ScheduledDuration(double solid_bound) const {
    const auto& state = transaction.committed();
    const double remaining = target_time - state.clock.time;
    double speed = liquid_step_internal::MaximumComponentMagnitude(liquid_step_internal::WallVelocity(scene.obstacle));
    for (const auto& particle : state.particles.values()) {
      speed = std::max(speed, liquid_step_internal::MaximumComponentMagnitude(particle.world_velocity));
    }
    for (const auto& source : scene.sources) {
      speed = std::max(
          speed,
          liquid_step_internal::MaximumComponentMagnitude(source.emitted_particle_template.world_velocity)
      );
    }
    double maximum = std::min({options.maximum_step_duration, remaining, solid_bound});
    const double acceleration = liquid_step_internal::MaximumComponentMagnitude(scene.world_acceleration);
    // Bound v*step_duration + a*step_duration^2, including this step's full acceleration before advection.
    const double half_width = 0.5 * state.geometry.cell_width;
    if (speed > 0) {
      maximum = std::min(maximum, half_width / speed);
    }
    if (acceleration > 0) {
      const double root = std::hypot(speed, 2 * std::sqrt(acceleration) * std::sqrt(half_width));
      maximum = std::min(maximum, 2 * half_width / (speed + root));
    }
    if (scene.surface_tension > 0) {
      const double h = state.geometry.cell_width;
      maximum = std::min(
          maximum,
          std::sqrt(scene.liquid_density * h * h * h / (4 * std::numbers::pi * scene.surface_tension))
      );
    }
    if (!std::isfinite(maximum) || maximum <= 0) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    const double count = std::ceil(remaining / maximum);
    const double duration = remaining / count;
    if (!std::isfinite(count) || duration <= 0 || state.clock.time + duration == state.clock.time) {
      return Error(absl::StatusCode::kOutOfRange);
    }
    return duration;
  }
};
LiquidSubstepScheduler3D::LiquidSubstepScheduler3D(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}
LiquidSubstepScheduler3D::LiquidSubstepScheduler3D(LiquidSubstepScheduler3D&&) noexcept = default;
LiquidSubstepScheduler3D& LiquidSubstepScheduler3D::operator=(LiquidSubstepScheduler3D&&) noexcept = default;
LiquidSubstepScheduler3D::~LiquidSubstepScheduler3D() = default;
absl::StatusOr<LiquidSubstepScheduler3D> LiquidSubstepScheduler3D::Create(
    const LiquidState3D& initial,
    const LiquidSceneInputs3D& scene,
    LiquidStepOptions3D options
) {
  return Create(initial, scene, options, CpuLiquidSubstepBackend3D());
}
absl::StatusOr<LiquidSubstepScheduler3D> LiquidSubstepScheduler3D::Create(
    const LiquidState3D& initial,
    const LiquidSceneInputs3D& scene,
    LiquidStepOptions3D options,
    const LiquidSubstepBackend3D& backend
) {
  const bool adaptive_mode = initial.adaptive_cells.size() != 0;
  if (adaptive_mode && (scene.drift || scene.cavity || scene.cavity_provider.prepare || !scene.phase_pairs.empty())) {
    return absl::UnimplementedError("graded tank drift, cavity or immiscible settings require a graded operator");
  }
  if (adaptive_mode && backend.project_adaptive == nullptr && !backend.host_adaptive_pressure) {
    return absl::UnimplementedError("selected backend has no graded pressure projection");
  }
  if (adaptive_mode && CanApplyViscosity(scene) && !backend.host_adaptive_pressure
      && backend.project_adaptive_viscous == nullptr) {
    return absl::UnimplementedError("selected backend has no graded viscous velocity inverse");
  }
  const bool needs_lattice_plane = std::holds_alternative<LiquidHorizontalPlaneSurface3D>(scene.field);
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&scene.obstacle);
  const bool needs_wetted_piston = tank && tank->piston_velocity_x != 0;
  const bool needs_isolated_drop = std::holds_alternative<LiquidIsolatedCovarianceDrop3D>(scene.field);
  if (!Complete(backend) || (needs_lattice_plane ? !backend.sample_lattice_plane : !backend.sample_field_batch)
      || (needs_wetted_piston && !backend.wetted_piston)) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  STEP_TRY(liquid_step_internal::ValidateScene(initial, scene, options));
  if (CanApplyViscosity(scene) && !backend.project_viscous) {
    return absl::UnimplementedError("backend does not support coupled viscosity");
  }
  if (CanApplyViscosity(scene) && needs_wetted_piston && !needs_lattice_plane) {
    return absl::UnimplementedError("viscous projection requires certified moving-wall quadratures");
  }
  const auto bytes = QueryLiquidWorkspaceBytes3D(options.capacities, PressureWorkspaceMode(scene));
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto staging = liquid_step_internal::Staging::Create(
      options.capacities,
      !adaptive_mode && CanStageRegularCellTransport(scene, tank),
      needs_wetted_piston,
      needs_isolated_drop,
      scene.drift && scene.drift->maximum_rate > 0.0,
      scene.drift && scene.drift->maximum_rate > 0.0 && !scene.eos,
      !scene.constituent_materials.empty(),
      !scene.phase_pairs.empty(),
      HasConstituentViscosity(scene)
  );
  if (!staging.ok()) {
    return staging.status();
  }
  const auto capacities = options.capacities;
  const auto reservation_bytes = ReservationBytes(
      capacities,
      scene.sources.size(),
      scene.constituent_materials.size(),
      scene.phase_pairs.size(),
      bytes->total_bytes,
      staging->reserved_bytes,
      scene.cavity.has_value(),
      adaptive_mode
  );
  if (!reservation_bytes.ok()) {
    return reservation_bytes.status();
  }
  if ((!scene.cavity && initial.next_cavity_id != 0)
      || (scene.cavity && initial.clock.accepted_step_ordinal != 0 && initial.next_cavity_id == 0)) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  std::optional<CavityStepScratch> cavity;
  std::optional<LiquidState3D> cavity_seed;
  if (scene.cavity) {
    if (scene.cavity_provider.prepare || !std::isfinite(scene.cavity->atmospheric_absolute_pressure)
        || scene.cavity->atmospheric_absolute_pressure <= 0.0
        || !std::isfinite(scene.cavity->birth_hydrostatic_gauge_pressure)
        || scene.cavity->birth_hydrostatic_gauge_pressure < 0.0) {
      return Error(absl::StatusCode::kInvalidArgument);
    }
    const auto cavity_capacity = std::max(capacities.cells, options.capacity_replay_reservations.cells);
    auto scratch = CavityStepScratch::Create(cavity_capacity);
    if (!scratch.ok()) {
      return scratch.status();
    }
    if (scratch->reserved_bytes > SIZE_MAX - *reservation_bytes) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    cavity = std::move(*scratch);
    if (initial.next_cavity_id == 0) {
      const LiquidStateCapacities3D seed_capacities{
          initial.particles.size(),
          initial.blocks.size(),
          initial.cells.size(),
          initial.faces.size(),
          initial.sources.size(),
      };
      auto seed = CloneLiquidState3D(initial, seed_capacities);
      if (!seed.ok()) {
        return seed.status();
      }
      auto gas = LiquidRecords3D<LiquidCavityGasState3D>::Create(capacities.cells);
      auto components = LiquidRecords3D<LiquidCavityComponentState3D>::Create(capacities.cells);
      if (!gas.ok()) {
        return gas.status();
      }
      if (!components.ok()) {
        return components.status();
      }
      seed->cavity_gas = std::move(*gas);
      seed->cavity_components = std::move(*components);
      seed->next_cavity_id = 1;
      const auto obstacle = liquid_step_internal::ObstacleAtTime(scene.obstacle, initial.clock.time);
      // Frame zero has no activated grid. The host field sampler supplies the
      // initial geometry independently of the physical backend's publication.
      STEP_TRY(
          liquid_step_internal::PrepareField(scene.field, obstacle, *seed, *staging, CpuLiquidSubstepBackend3D(), true)
      );
      const auto grid = SparseMacGridView3D::Create(initial.geometry, {});
      if (!grid.ok()) {
        return grid.status();
      }
      const LiquidBoundaryGeometryViewInputs3D samples{
          .mac_grid = *grid,
          .cell_indices = staging->candidates.values(),
          .center_distances = staging->centers.values(),
          .corner_distances = staging->corners.values(),
          .obstacle = obstacle,
          .step_duration = 0.0,
          .liquid_density = scene.liquid_density,
          .surface_tension = scene.surface_tension,
      };
      std::optional<LiquidCavityStepPreparation3D> measurement;
      const auto prepared = cavity->Prepare(*scene.cavity, samples, *seed, measurement);
      if (!prepared.ok()) {
        return prepared.status();
      }
      cavity_seed = std::move(*seed);
    }
  }
  const auto& initial_version = cavity_seed ? *cavity_seed : initial;
  auto transaction = LiquidFrameTransaction3D::Create(initial_version, capacities, scene.cavity.has_value());
  if (!transaction.ok()) {
    return transaction.status();
  }
  auto workspace = LiquidStepWorkspace3D::Create(capacities, initial.geometry, PressureWorkspaceMode(scene));
  if (!workspace.ok()) {
    return workspace.status();
  }
  auto sources = LiquidRecords3D<LiquidEmissionSource3D>::Create(scene.sources.size());
  if (!sources.ok()) {
    return sources.status();
  }
  STEP_TRY(sources->Resize(scene.sources.size()));
  std::ranges::copy(scene.sources, sources->values().begin());
  auto materials = LiquidRecords3D<LiquidConstituentMaterial3D>::Create(scene.constituent_materials.size());
  if (!materials.ok()) {
    return materials.status();
  }
  STEP_TRY(materials->Resize(scene.constituent_materials.size()));
  std::ranges::copy(scene.constituent_materials, materials->values().begin());
  auto phase_pairs = LiquidRecords3D<LiquidPhasePair3D>::Create(scene.phase_pairs.size());
  if (!phase_pairs.ok()) {
    return phase_pairs.status();
  }
  STEP_TRY(phase_pairs->Resize(scene.phase_pairs.size()));
  std::ranges::copy(scene.phase_pairs, phase_pairs->values().begin());
  std::unique_ptr<Impl> impl(new (std::nothrow)
                                 Impl(std::move(*transaction), std::move(*workspace), std::move(*staging)));
  if (!impl) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  impl->reservation_bytes = *reservation_bytes + (cavity ? cavity->reserved_bytes : 0);
  impl->cavity = std::move(cavity);
  impl->sources = std::move(*sources);
  impl->scene = scene;
  impl->scene.sources = impl->sources.values();
  impl->constituent_materials = std::move(*materials);
  impl->scene.constituent_materials = impl->constituent_materials.values();
  impl->phase_pairs = std::move(*phase_pairs);
  impl->scene.phase_pairs = impl->phase_pairs.values();
  impl->options = options;
  if (options.ocean_boundary) {
    auto ocean_boundary = ResolvedOceanBoundary3D::Resolve(*options.ocean_boundary);
    if (!ocean_boundary.ok()) {
      return ocean_boundary.status();
    }
    impl->ocean_boundary = std::move(*ocean_boundary);
    if (options.ocean_boundary->inlet_particle_spacing_m > 0) {
      if (!std::holds_alternative<LiquidParticleSphereUnion3D>(scene.field)) {
        return absl::InvalidArgumentError("ocean particle level control needs a particle-derived liquid field");
      }
      auto inlet = liquid_step_internal::OceanInletParticles3D::Create(*options.ocean_boundary, scene.liquid_density);
      if (!inlet.ok()) {
        return inlet.status();
      }
      if (inlet->reserved_bytes() > SIZE_MAX - impl->reservation_bytes) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      impl->reservation_bytes += inlet->reserved_bytes();
      impl->ocean_inlet_particles = std::move(*inlet);
    }
  }
  if (adaptive_mode) {
    auto adaptive = liquid_step_internal::AdaptiveLiquidStep3D::Create(initial, impl->scene, options);
    if (!adaptive.ok()) {
      return adaptive.status();
    }
    if (adaptive->reserved_bytes() > static_cast<std::size_t>(PTRDIFF_MAX) - impl->reservation_bytes) {
      return Error(absl::StatusCode::kResourceExhausted);
    }
    impl->reservation_bytes += adaptive->reserved_bytes();
    impl->adaptive = std::move(*adaptive);
  }
  STEP_TRY(backend.initialize(backend.context, initial_version));
  impl->backend = backend;
  return LiquidSubstepScheduler3D(std::move(impl));
}
const LiquidState3D& LiquidSubstepScheduler3D::committed() const {
  return impl_->transaction.committed();
}
const LiquidStepDiagnostics3D& LiquidSubstepScheduler3D::diagnostics() const {
  return impl_->diagnostics;
}
std::size_t LiquidSubstepScheduler3D::reserved_bytes() const {
  return impl_->reservation_bytes;
}
void LiquidSubstepScheduler3D::SetStageFaultHookForTesting(LiquidStageFaultHook3D hook) {
  impl_->hook = hook;
}
absl::Status LiquidSubstepScheduler3D::BeginFrame(double target) {
  STEP_TRY(impl_->transaction.BeginFrame(target));
  impl_->target_time = target;
  impl_->frame_active = true;
  impl_->diagnostics = {};
  return absl::OkStatus();
}
struct LiquidSubstepScheduler3D::AdaptiveStateTrial {
  LiquidState3D state;
  liquid_step_internal::AdaptiveLiquidStep3D owner;
  std::size_t accepted_reservation_bytes;
};

absl::StatusOr<LiquidAttemptOutcome3D>
LiquidSubstepScheduler3D::AttemptWithAdaptiveFamily(double step_duration, AdaptiveMacCellKey3D parent, bool refine) {
  return AttemptPreparedAdaptiveState(step_duration, [&] {
    return liquid_step_internal::PrepareAdaptiveLiquidFamilyCandidate3D(
        committed(),
        impl_->scene,
        impl_->options.capacities,
        parent,
        refine
    );
  });
}

absl::StatusOr<LiquidAttemptOutcome3D> LiquidSubstepScheduler3D::AttemptWithAdaptiveBandRetirement(
    double step_duration,
    AdaptiveMacCellKey3D leaf,
    LiquidHandoffOccupancy3D occupancy,
    const LiquidHandoffCarrier3D& carrier
) {
  return AttemptPreparedAdaptiveState(step_duration, [&] {
    return PrepareAdaptiveLiquidBandRetirement3D(committed(), impl_->options.capacities, leaf, occupancy, carrier);
  });
}

absl::StatusOr<LiquidAttemptOutcome3D> LiquidSubstepScheduler3D::AttemptWithAdaptiveCoreReseed(
    double step_duration,
    AdaptiveMacCellKey3D leaf,
    LiquidHandoffOccupancy3D occupancy,
    const LiquidHandoffCarrier3D& carrier,
    std::size_t particle_count
) {
  return AttemptPreparedAdaptiveState(step_duration, [&] {
    return PrepareAdaptiveLiquidCoreReseed3D(
        committed(),
        impl_->options.capacities,
        leaf,
        occupancy,
        carrier,
        particle_count
    );
  });
}

absl::StatusOr<LiquidAttemptOutcome3D> LiquidSubstepScheduler3D::AttemptPreparedAdaptiveState(
    double step_duration,
    absl::FunctionRef<absl::StatusOr<LiquidState3D>()> prepare
) {
  auto& impl = *impl_;
  if (!impl.adaptive) {
    return absl::FailedPreconditionError("adaptive state replacement requires a graded scheduler");
  }
  if (!impl.frame_active || !std::isfinite(step_duration) || step_duration <= 0
      || step_duration > impl.target_time - committed().clock.time) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  auto prepared = prepare();
  if (!prepared.ok()) {
    return prepared.status();
  }
  auto owner = liquid_step_internal::AdaptiveLiquidStep3D::Create(*prepared, impl.scene, impl.options);
  if (!owner.ok()) {
    return owner.status();
  }
  if (owner->reserved_bytes() > static_cast<std::size_t>(PTRDIFF_MAX) - impl.reservation_bytes) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  const auto accepted_bytes = impl.reservation_bytes - impl.adaptive->reserved_bytes() + owner->reserved_bytes();
  AdaptiveStateTrial candidate{std::move(*prepared), std::move(*owner), accepted_bytes};
  return AttemptImpl(step_duration, {}, nullptr, false, false, nullptr, nullptr, nullptr, &candidate);
}

absl::StatusOr<LiquidAttemptOutcome3D>
LiquidSubstepScheduler3D::Attempt(double step_duration, LiquidWorkPhaseObserver3D work_phase) {
  return AttemptImpl(step_duration, work_phase, nullptr);
}

absl::StatusOr<LiquidPlanarPistonAttempt3D>
LiquidSubstepScheduler3D::AttemptWithPlanarPistonPressureLoad(double step_duration, RigidBody3D piston) {
  if (!impl_->staging.wetted_piston.has_value()
      || !std::holds_alternative<LiquidHorizontalPlaneSurface3D>(impl_->scene.field)) {
    return absl::FailedPreconditionError("piston load requires a moving planar piston scene");
  }
  const auto validation = ValidateRigidBody3D(piston);
  if (!validation.ok()) {
    return validation;
  }
  const auto outcome = AttemptImpl(step_duration, {}, &piston);
  if (!outcome.ok()) {
    return outcome.status();
  }
  return LiquidPlanarPistonAttempt3D{
      .outcome = *outcome,
      .loaded_piston = outcome->disposition == LiquidAttemptDisposition3D::kAccepted
          ? std::optional<RigidBody3D>{piston}
          : std::nullopt,
  };
}

absl::StatusOr<LiquidPlanarPistonAttempt3D>
LiquidSubstepScheduler3D::AttemptWithDynamicPlanarPiston(double step_duration, RigidBody3D piston) {
  return AttemptDynamicPistonImpl(step_duration, piston, false);
}

absl::StatusOr<LiquidPlanarPistonAttempt3D> LiquidSubstepScheduler3D::AttemptWithDynamicPlanarPiston(
    double step_duration,
    RigidBody3D piston,
    LiquidPistonReactionUpdate3D update_with_reaction
) {
  return AttemptDynamicPistonImpl(step_duration, piston, false, &update_with_reaction);
}

absl::StatusOr<LiquidPlanarPistonAttempt3D>
LiquidSubstepScheduler3D::AttemptWithCoupledPlanarPiston(double step_duration, RigidBody3D piston) {
  if (!committed().particles.values().empty() || std::isfinite(impl_->scene.bulk_modulus)
      || impl_->scene.surface_tension != 0.0) {
    return absl::FailedPreconditionError("coupled piston requires a particle-free incompressible planar scene");
  }
  return AttemptDynamicPistonImpl(step_duration, piston, true);
}

absl::StatusOr<LiquidPlanarPistonAttempt3D> LiquidSubstepScheduler3D::AttemptDynamicPistonImpl(
    double step_duration,
    RigidBody3D piston,
    bool coupled,
    const LiquidPistonReactionUpdate3D* update_with_reaction
) {
  if (!impl_->staging.wetted_piston.has_value()
      || !std::holds_alternative<LiquidHorizontalPlaneSurface3D>(impl_->scene.field) || !impl_->scene.sources.empty()
      || impl_->ocean_boundary) {
    return absl::FailedPreconditionError("dynamic piston requires a moving planar piston scene without sources");
  }
  if (update_with_reaction == nullptr && !committed().particles.values().empty()
      && impl_->backend.update_particles != CpuLiquidSubstepBackend3D().update_particles) {
    return absl::FailedPreconditionError("dynamic piston particles require the CPU contact update");
  }
  const auto validation = ValidateRigidBody3D(piston);
  if (!validation.ok()) {
    return validation;
  }
  if (piston.linear_velocity.y != 0.0 || piston.linear_velocity.z != 0.0 || piston.angular_velocity.x != 0.0
      || piston.angular_velocity.y != 0.0 || piston.angular_velocity.z != 0.0) {
    return absl::InvalidArgumentError("dynamic piston requires axial motion");
  }
  const auto outcome = AttemptImpl(step_duration, {}, &piston, true, coupled, update_with_reaction);
  if (!outcome.ok()) {
    return outcome.status();
  }
  return LiquidPlanarPistonAttempt3D{
      .outcome = *outcome,
      .loaded_piston = outcome->disposition == LiquidAttemptDisposition3D::kAccepted
          ? std::optional<RigidBody3D>{piston}
          : std::nullopt,
  };
}

absl::StatusOr<LiquidFloatingBoxAttempt3D>
LiquidSubstepScheduler3D::AttemptWithGuidedFloatingBox(double step_duration, RigidBody3D body, Vec3 half_extent) {
  return AttemptFloatingBox(step_duration, body, half_extent, FloatingBoxMotion::kVerticalGuide);
}

absl::StatusOr<LiquidFloatingBoxAttempt3D>
LiquidSubstepScheduler3D::AttemptWithTranslatingFloatingBox(double step_duration, RigidBody3D body, Vec3 half_extent) {
  return AttemptFloatingBox(step_duration, body, half_extent, FloatingBoxMotion::kTranslation);
}

absl::StatusOr<LiquidFloatingBoxAttempt3D> LiquidSubstepScheduler3D::AttemptWithCoupledGuidedFloatingBox(
    double step_duration,
    RigidBody3D body,
    Vec3 half_extent
) {
  if (std::isfinite(impl_->scene.bulk_modulus) || impl_->scene.surface_tension != 0.0) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  return AttemptFloatingBox(step_duration, body, half_extent, FloatingBoxMotion::kCoupledVerticalGuide);
}

absl::StatusOr<LiquidFloatingBoxAttempt3D> LiquidSubstepScheduler3D::AttemptWithOrientedFloatingBoxPressure(
    double step_duration,
    RigidBody3D body,
    Vec3 body_half_extent
) {
  if (committed().particles.size() != 0) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  return AttemptFloatingBox(step_duration, body, body_half_extent, FloatingBoxMotion::kOrientedPressure);
}

absl::StatusOr<LiquidFloatingBoxAttempt3D> LiquidSubstepScheduler3D::AttemptWithOrientedFloatingBox(
    double step_duration,
    RigidBody3D body,
    Vec3 body_half_extent
) {
  return AttemptFloatingBox(step_duration, body, body_half_extent, FloatingBoxMotion::kOrientedPressure);
}
absl::StatusOr<LiquidFloatingBoxAttempt3D> LiquidSubstepScheduler3D::AttemptWithCoupledOrientedFloatingBox(
    double step_duration,
    RigidBody3D body,
    Vec3 body_half_extent
) {
  if (std::isfinite(impl_->scene.bulk_modulus) || impl_->scene.surface_tension != 0.0) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  return AttemptFloatingBox(step_duration, body, body_half_extent, FloatingBoxMotion::kCoupledOrientedPressure);
}
absl::StatusOr<LiquidFloatingBoxAttempt3D> LiquidSubstepScheduler3D::AttemptFloatingBox(
    double step_duration,
    RigidBody3D body,
    Vec3 half_extent,
    FloatingBoxMotion motion
) {
  const bool oriented_coupling = motion == FloatingBoxMotion::kCoupledOrientedPressure;
  const bool oriented = motion == FloatingBoxMotion::kOrientedPressure || oriented_coupling;
  const bool coupled = motion == FloatingBoxMotion::kCoupledVerticalGuide;
  const bool vertical_guide = motion == FloatingBoxMotion::kVerticalGuide || coupled;
  const auto* surface = std::get_if<LiquidHorizontalPlaneSurface3D>(&impl_->scene.field);
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&impl_->scene.obstacle);
  if (!surface || !tank || tank->piston_velocity_x != 0.0 || !impl_->scene.sources.empty()
      || impl_->scene.friction_coefficient != 0.0 || impl_->backend.freeze != CpuLiquidSubstepBackend3D().freeze) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  const bool valid_extent =
      liquid_step_internal::Finite(half_extent) && half_extent.x > 0.0 && half_extent.y > 0.0 && half_extent.z > 0.0;
  const bool aligned = std::abs(body.orientation.w) == 1.0
      && liquid_step_internal::MaximumComponentMagnitude(body.orientation.vector) == 0.0;
  const bool violates_guide = vertical_guide && (body.linear_velocity.x != 0.0 || body.linear_velocity.z != 0.0);
  const bool violates_orientation_constraint =
      !oriented && (!aligned || liquid_step_internal::MaximumComponentMagnitude(body.angular_velocity) != 0.0);
  if (!ValidateRigidBody3D(body).ok() || !valid_extent || violates_guide || violates_orientation_constraint) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  FloatingBoxTrial box{
      .body = body,
      .geometry =
          {
              .world_low = liquid_step_internal::Add(body.center, liquid_step_internal::Scale(half_extent, -1.0)),
              .world_high = liquid_step_internal::Add(body.center, half_extent),
              .liquid_height = surface->world_surface_height_y,
          },
      .oriented_geometry = oriented
          ? std::optional<
                OrientedFloatingBoxGeometry3D>{{{body.center, body.orientation}, half_extent, surface->world_surface_height_y}
            }
          : std::nullopt,
      .half_extent = half_extent,
      .coupled = coupled || oriented_coupling,
      .vertical_guide = vertical_guide,
      .added_mass_kg = std::nullopt,
      .oriented_loads = oriented
          ? std::optional<OrientedFloatingBoxLoads3D>{{.external_force = body.force, .external_torque = body.torque}}
          : std::nullopt,
  };
  const auto outcome = AttemptImpl(step_duration, {}, nullptr, false, false, nullptr, &box);
  if (!outcome.ok()) {
    return outcome.status();
  }
  return LiquidFloatingBoxAttempt3D{
      .outcome = *outcome,
      .body = outcome->disposition == LiquidAttemptDisposition3D::kAccepted ? std::optional<RigidBody3D>{box.body}
                                                                            : std::nullopt,
      .added_mass_kg = outcome->disposition == LiquidAttemptDisposition3D::kAccepted ? box.added_mass_kg : std::nullopt,
      .oriented_loads =
          outcome->disposition == LiquidAttemptDisposition3D::kAccepted ? box.oriented_loads : std::nullopt,
  };
}

absl::Status LiquidSubstepScheduler3D::ValidateSnowStep(const LiquidSnowStepInputs3D& inputs) const {
  const auto& impl = *impl_;
  const auto* platen = std::get_if<TranslatingHalfspace3D>(&impl.scene.obstacle);
  const bool vertical_platen = platen != nullptr && platen->unit_normal.x == 0.0 && platen->unit_normal.y == -1.0
      && platen->unit_normal.z == 0.0 && platen->wall_velocity.x == 0.0 && platen->wall_velocity.z == 0.0;
  const bool supported_obstacle = std::holds_alternative<std::monostate>(impl.scene.obstacle) || vertical_platen;
  if (impl.adaptive || !supported_obstacle || impl.cavity || impl.scene.cavity_provider.prepare) {
    return absl::UnimplementedError(
        "sampled snow requires uniform liquid without cavities and supports a vertical platen"
    );
  }
  const auto liquid_geometry = committed().geometry;
  const auto solid_geometry = inputs.solid_grid.geometry();
  if (inputs.accepted_particles.empty() || liquid_geometry.cell_width != solid_geometry.cell_width
      || liquid_geometry.world_origin.x != solid_geometry.world_origin.x
      || liquid_geometry.world_origin.y != solid_geometry.world_origin.y
      || liquid_geometry.world_origin.z != solid_geometry.world_origin.z) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  return absl::OkStatus();
}

absl::StatusOr<LiquidSnowAttempt3D>
LiquidSubstepScheduler3D::AttemptWithSnow(double step_duration, const LiquidSnowStepInputs3D& inputs) {
  STEP_TRY(ValidateSnowStep(inputs));
  SnowTrial snow{.inputs = inputs, .candidate = {}};
  const auto outcome = AttemptImpl(step_duration, {}, nullptr, false, false, nullptr, nullptr, &snow);
  if (!outcome.ok()) {
    return outcome.status();
  }
  LiquidSnowAttempt3D result{.outcome = *outcome, .accepted_snow = std::nullopt};
  if (outcome->disposition == LiquidAttemptDisposition3D::kAccepted) {
    result.accepted_snow = std::move(snow.candidate);
  }
  return result;
}

absl::StatusOr<LiquidAttemptOutcome3D> LiquidSubstepScheduler3D::AttemptImpl(
    double step_duration,
    LiquidWorkPhaseObserver3D work_phase,
    RigidBody3D* piston_load,
    bool dynamic_piston,
    bool coupled_piston,
    const LiquidPistonReactionUpdate3D* update_with_reaction,
    FloatingBoxTrial* floating_box,
    SnowTrial* snow,
    AdaptiveStateTrial* adaptive_replacement
) {
  auto& impl = *impl_;
  if (impl.adaptive && (piston_load != nullptr || floating_box != nullptr)) {
    return absl::UnimplementedError("graded Simulate does not support coupled rigid trials");
  }
  auto& diagnostics = impl.diagnostics;
  if (!impl.frame_active || !std::isfinite(step_duration) || step_duration <= 0
      || step_duration > impl.target_time - committed().clock.time) {
    return Error(absl::StatusCode::kInvalidArgument);
  }
  impl.SetWorkPhase(work_phase, LiquidWorkPhase3D::kEmitActivationField);
  const absl::Cleanup end_phase = [&] {
    impl.SetWorkPhase(work_phase, LiquidWorkPhase3D::kCount);
  };
  auto trial = impl.transaction.ResetTrialFromCommitted();
  if (!trial.ok()) {
    return trial.status();
  }
  if (adaptive_replacement != nullptr) {
    **trial = std::move(adaptive_replacement->state);
  }
  ++diagnostics.attempts;
  diagnostics.accepted_measurement.reset();
  diagnostics.stage_count = 0;
  diagnostics.attempted_duration = step_duration;
  diagnostics.pressure_diagnostics = {};
  diagnostics.old_pic_fallback_count = 0;
  diagnostics.new_pic_fallback_count = 0;
  diagnostics.lower_theta_regularization_count = 0;
  diagnostics.failure_code = absl::StatusCode::kOk;
  diagnostics.failed_stage = LiquidPhysicalStage3D::kCount;
  LiquidAcceptedStepMeasurement3D accepted_measurement;
  auto* adaptive_owner =
      adaptive_replacement != nullptr ? &adaptive_replacement->owner : (impl.adaptive ? &*impl.adaptive : nullptr);
  auto status = adaptive_owner != nullptr
      ? impl.ExecuteAdaptive(*adaptive_owner, step_duration, **trial, accepted_measurement, work_phase)
      : impl.Execute(
            step_duration,
            **trial,
            accepted_measurement,
            work_phase,
            piston_load,
            dynamic_piston,
            coupled_piston,
            update_with_reaction,
            floating_box,
            snow
        );
  if (status.ok()) {
    const auto clock = impl.transaction.PreviewAcceptedClock(step_duration);
    status = clock.ok() ? impl.backend.prepare_publication(impl.backend.context, **trial, *clock) : clock.status();
  }
  auto disposition = LiquidAttemptDisposition3D::kAccepted;
  if (!status.ok()) {
    diagnostics.failure_code = status.code();
    diagnostics.failed_stage = diagnostics.stages[diagnostics.stage_count - 1];
    disposition = LiquidAttemptDisposition3D::kFatal;
    if (status.code() == absl::StatusCode::kResourceExhausted) {
      disposition = LiquidAttemptDisposition3D::kCapacity;
    }
    // Only explicitly step-dependent failures use Aborted. Geometry constraint,
    // source compatibility, invalid range and backend errors are never step_duration retries.
    if (status.code() == absl::StatusCode::kAborted) {
      disposition = LiquidAttemptDisposition3D::kRetryDt;
    }
  }
  auto outcome = impl.transaction.FinishAttempt(disposition, step_duration);
  if (!outcome.ok()) {
    impl.backend.discard_publication(impl.backend.context);
    diagnostics.failure_code = outcome.status().code();
    return outcome.status();
  }
  if (outcome->disposition == LiquidAttemptDisposition3D::kAccepted) {
    impl.backend.commit_publication(impl.backend.context);
    if (adaptive_replacement != nullptr) {
      impl.adaptive = std::move(adaptive_replacement->owner);
      impl.reservation_bytes = adaptive_replacement->accepted_reservation_bytes;
    }
    ++diagnostics.accepted_attempts;
    accepted_measurement.accepted_clock = committed().clock;
    accepted_measurement.accepted_dt_seconds = step_duration;
    diagnostics.accepted_measurement = accepted_measurement;
    diagnostics.stages[diagnostics.stage_count++] = LiquidPhysicalStage3D::kAdapt;
  }
  if (outcome->disposition != LiquidAttemptDisposition3D::kAccepted) {
    impl.backend.discard_publication(impl.backend.context);
  }
  if (outcome->retry_duration) {
    ++diagnostics.dt_retries;
  }
  return outcome;
}
absl::Status LiquidSubstepScheduler3D::FinishFrame() {
  STEP_TRY(impl_->transaction.FinishFrame());
  impl_->frame_active = false;
  return absl::OkStatus();
}
absl::Status LiquidSubstepScheduler3D::AdvanceFrame(double target, LiquidProgressObserver3D observer) {
  return AdvanceFrameImpl(target, observer, std::nullopt, nullptr);
}

absl::Status LiquidSubstepScheduler3D::AdvanceFrameWithSnow(
    double target,
    const LiquidSnowStepInputs3D& inputs,
    std::vector<PlasticSolidCheckpointParticle3D>& accepted_snow,
    LiquidProgressObserver3D observer
) {
  STEP_TRY(ValidateSnowStep(inputs));
  const double acoustic_cfl = inputs.solid_step.acoustic_cfl;
  if (!std::isfinite(acoustic_cfl) || acoustic_cfl <= 0 || acoustic_cfl > 1) {
    return absl::InvalidArgumentError("invalid snow acoustic CFL");
  }
  return AdvanceFrameImpl(target, observer, inputs, &accepted_snow);
}

absl::Status LiquidSubstepScheduler3D::AdvanceFrameImpl(
    double target,
    LiquidProgressObserver3D observer,
    std::optional<LiquidSnowStepInputs3D> snow_inputs,
    std::vector<PlasticSolidCheckpointParticle3D>* accepted_snow
) {
  STEP_TRY(BeginFrame(target));
  std::vector<PlasticSolidCheckpointParticle3D> frame_start_snow;
  if (snow_inputs) {
    const auto initial = snow_inputs->accepted_particles;
    *accepted_snow = std::vector(initial.begin(), initial.end());
    if (impl_->options.capacity_replay_reservations != LiquidStateCapacities3D{}) {
      frame_start_snow = *accepted_snow;
    }
  }
  std::optional<double> retry;
  while (committed().clock.time < target) {
    double snow_bound = std::numeric_limits<double>::infinity();
    if (snow_inputs) {
      snow_inputs->accepted_particles = *accepted_snow;
      const double cell_width = snow_inputs->solid_grid.geometry().cell_width;
      const auto particles = StagePressureLoadedSnowParticles3D(
          *accepted_snow,
          snow_inputs->body_id,
          snow_inputs->material_id,
          snow_inputs->snow,
          cell_width,
          {}
      );
      if (!particles.ok()) {
        return particles.status();
      }
      for (const auto& particle : *particles) {
        const auto state = internal::EvaluateAcceptedDrySolid3D(particle, snow_inputs->snow);
        if (!state.ok()) {
          return state.status();
        }
        const double wave_speed = std::sqrt(state->wave_speed_squared);
        if (wave_speed > 0) {
          const double acoustic_bound = snow_inputs->solid_step.acoustic_cfl * cell_width / wave_speed;
          // Round inward so the dry solver's multiply-back bound stays admissible.
          snow_bound = std::min(snow_bound, std::nextafter(acoustic_bound, 0.0));
        }
      }
    }
    auto duration = impl_->ScheduledDuration(snow_bound);
    if (!duration.ok()) {
      return duration.status();
    }
    if (retry) {
      duration = *retry;
    }
    const auto start_clock = committed().clock;
    const auto previous_attempt_count = diagnostics().attempts;
    const auto work_start =
        observer.observe_attempt ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto outcome = [&]() -> absl::StatusOr<LiquidAttemptOutcome3D> {
      if (!snow_inputs) {
        return Attempt(*duration, observer.work_phase);
      }
      SnowTrial snow{.inputs = *snow_inputs, .candidate = {}};
      auto attempt = AttemptImpl(*duration, observer.work_phase, nullptr, false, false, nullptr, nullptr, &snow);
      if (attempt.ok() && attempt->disposition == LiquidAttemptDisposition3D::kAccepted) {
        *accepted_snow = std::move(snow.candidate);
      }
      return attempt;
    }();
    if (observer.observe_attempt && diagnostics().attempts != previous_attempt_count) {
      const std::chrono::duration<double> work_elapsed = std::chrono::steady_clock::now() - work_start;
      observer.observe_attempt(
          observer.context,
          {
              .start_clock = start_clock,
              .attempt_in_frame = diagnostics().attempts,
              .capacity_pass = diagnostics().capacity_replays,
              .attempted_dt_seconds = *duration,
              .work_wall_seconds = work_elapsed.count(),
              .disposition = outcome.ok() ? outcome->disposition : LiquidAttemptDisposition3D::kFatal,
              .retry_dt_seconds = outcome.ok() ? outcome->retry_duration : std::nullopt,
              .failure_code = outcome.ok() ? diagnostics().failure_code : outcome.status().code(),
          }
      );
    }
    if (!outcome.ok()) {
      return outcome.status();
    }
    if (outcome->disposition == LiquidAttemptDisposition3D::kAccepted && observer.observe != nullptr) {
      STEP_TRY(observer.observe(observer.context, committed(), diagnostics(), false));
    }
    retry = outcome->retry_duration;
    if (outcome->disposition == LiquidAttemptDisposition3D::kFatal) {
      return Error(impl_->diagnostics.failure_code);
    }
    if (outcome->disposition == LiquidAttemptDisposition3D::kRetryDt && !retry) {
      return Error(absl::StatusCode::kAborted);
    }
    if (outcome->disposition == LiquidAttemptDisposition3D::kCapacity) {
      if (impl_->diagnostics.capacity_replays != 0) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      const auto growth = impl_->options.capacity_replay_reservations;
      const auto current = impl_->options.capacities;
      if (growth == LiquidStateCapacities3D{} || growth == current) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      auto staging = liquid_step_internal::Staging::Create(
          growth,
          impl_->staging.regular_transport.has_value(),
          impl_->staging.wetted_piston.has_value(),
          impl_->staging.isolated_drop_restricted_gathers.has_value(),
          impl_->staging.drift_sensors.has_value(),
          impl_->staging.drift_pressure_cells.has_value(),
          impl_->staging.composition_density.has_value(),
          impl_->staging.phase_crossings.has_value(),
          impl_->staging.composition_viscosity.has_value()
      );
      if (!staging.ok()) {
        return staging.status();
      }
      auto workspace = LiquidStepWorkspace3D::Create(growth, committed().geometry, PressureWorkspaceMode(impl_->scene));
      if (!workspace.ok()) {
        return workspace.status();
      }
      const auto reservation_bytes = ReservationBytes(
          growth,
          impl_->sources.size(),
          impl_->constituent_materials.size(),
          impl_->phase_pairs.size(),
          workspace->usage().reserved_bytes,
          staging->reserved_bytes,
          impl_->cavity.has_value()
      );
      if (!reservation_bytes.ok()) {
        return reservation_bytes.status();
      }
      const auto inlet_bytes = impl_->ocean_inlet_particles ? impl_->ocean_inlet_particles->reserved_bytes() : 0;
      const auto cavity_bytes = impl_->cavity ? impl_->cavity->reserved_bytes : 0;
      if (inlet_bytes > SIZE_MAX - *reservation_bytes || cavity_bytes > SIZE_MAX - *reservation_bytes - inlet_bytes) {
        return Error(absl::StatusCode::kResourceExhausted);
      }
      const auto prepared_growth =
          impl_->backend.prepare_growth(impl_->backend.context, growth, impl_->transaction.frame_start());
      if (!prepared_growth.ok()) {
        impl_->backend.discard_growth(impl_->backend.context);
        return prepared_growth;
      }
      const auto restarted = impl_->transaction.RestartFrameWithCapacity(growth);
      if (!restarted.ok()) {
        impl_->backend.discard_growth(impl_->backend.context);
        return restarted;
      }
      impl_->backend.commit_growth(impl_->backend.context);
      if (snow_inputs) {
        *accepted_snow = std::move(frame_start_snow);
      }
      impl_->reservation_bytes = *reservation_bytes + inlet_bytes + cavity_bytes;
      impl_->workspace = std::move(*workspace);
      impl_->staging = std::move(*staging);
      impl_->options.capacities = growth;
      ++impl_->diagnostics.capacity_replays;
      if (observer.observe != nullptr) {
        STEP_TRY(observer.observe(observer.context, committed(), diagnostics(), true));
      }
    }
  }
  return FinishFrame();
}
#undef STEP_TRY
}  // namespace fluidity
