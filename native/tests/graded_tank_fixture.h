#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity::test {

inline absl::Status PopulateGradedTankFixture3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  state.geometry.cell_width = 0.5;
  scene.field = LiquidHorizontalPlaneSurface3D{1};
  scene.obstacle = AxisAlignedLiquidTank3D{0, 1.5, 0, 0, 1, 0};
  scene.world_acceleration = {};
  scene.liquid_density = 2;

  const std::array keys{
      AdaptiveMacCellKey3D{.lower_corner = {0, 0, 0}, .width_exponent = 1},
      AdaptiveMacCellKey3D{.lower_corner = {2, 0, 0}, .width_exponent = 0},
      AdaptiveMacCellKey3D{.lower_corner = {2, 1, 0}, .width_exponent = 0},
      AdaptiveMacCellKey3D{.lower_corner = {2, 0, 1}, .width_exponent = 0},
      AdaptiveMacCellKey3D{.lower_corner = {2, 1, 1}, .width_exponent = 0},
  };
  const auto topology = AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto face_count = ownership->interior_face_slots.size() + ownership->exterior_faces.size();
  auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(keys.size());
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(keys.size());
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(face_count);
  if (!cells.ok() || !particles.ok() || !faces.ok()) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  if (!cells->Resize(keys.size()).ok() || !particles->Resize(keys.size()).ok() || !faces->Resize(face_count).ok()) {
    return absl::Status(absl::StatusCode::kInternal, {});
  }
  constexpr std::array velocities{
      Vec3{0.015, -0.012, 0.0},
      Vec3{-0.02, 0.01, 0.0},
      Vec3{0.025, 0.02, -0.01},
      Vec3{-0.005, -0.015, 0.01},
      Vec3{0.008, 0.004, 0.012},
  };
  for (std::size_t cell = 0; cell < keys.size(); ++cell) {
    const auto& leaf = topology->cells()[cell];
    const auto lower = leaf.key.lower_corner;
    auto& particle = (*particles)[cell];
    particle.id = cell + 1;
    particle.mass = scene.liquid_density * leaf.cell_volume;
    particle.reference_volume = leaf.cell_volume;
    particle.world_position = {
        0.5 * lower.x + leaf.cell_width / 2,
        0.5 * lower.y + leaf.cell_width / 2,
        0.5 * lower.z + leaf.cell_width / 2,
    };
    particle.world_velocity = velocities[cell];
    particle.pic_initialized = true;
    (*cells)[cell] = {
        .key = leaf.key,
        .inventory =
            {
                .mass_kg = particle.mass,
                .reference_volume_m3 = particle.reference_volume,
                .current_volume_m3 = particle.reference_volume,
                .momentum_kg_m_per_s =
                    {particle.mass * particle.world_velocity.x,
                     particle.mass * particle.world_velocity.y,
                     particle.mass * particle.world_velocity.z},
            },
    };
  }
  for (auto& face : faces->values()) {
    face.validity = LiquidFaceValidity3D::kPredicted;
  }
  state.next_particle_id = keys.size() + 1;
  state.adaptive_cells = std::move(*cells);
  state.adaptive_faces = std::move(*faces);
  state.particles = std::move(*particles);
  return ValidateLiquidState3D(state);
}

// Start with a particle inventory that can be rebinned without resampling.
inline absl::Status PopulateGradedFamilyFixture3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  const auto authored = PopulateGradedTankFixture3D(state, scene);
  if (!authored.ok()) {
    return authored;
  }
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(12);
  if (!particles.ok()) {
    return particles.status();
  }
  const auto resized = particles->Resize(12);
  if (!resized.ok()) {
    return resized;
  }
  for (std::size_t ordinal = 0; ordinal < particles->size(); ++ordinal) {
    auto& particle = (*particles)[ordinal];
    particle = state.particles[ordinal < 8 ? 0 : ordinal - 7];
    if (ordinal < 8) {
      particle.mass /= 8;
      particle.reference_volume /= 8;
      particle.world_position = {
          (ordinal & 1) != 0 ? 0.75 : 0.25,
          (ordinal & 2) != 0 ? 0.75 : 0.25,
          (ordinal & 4) != 0 ? 0.75 : 0.25,
      };
    }
    particle.id = ordinal + 1;
    particle.world_velocity = {};
  }
  for (auto& cell : state.adaptive_cells.values()) {
    cell.inventory.momentum_kg_m_per_s = {};
  }
  state.particles = std::move(*particles);
  state.next_particle_id = 13;
  return ValidateLiquidState3D(state);
}

inline absl::Status PopulateGradedBandFixture3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  const auto authored = PopulateGradedTankFixture3D(state, scene);
  if (!authored.ok()) {
    return authored;
  }
  for (auto& particle : state.particles.values()) {
    particle.world_velocity = {};
  }
  for (auto& cell : state.adaptive_cells.values()) {
    cell.inventory.momentum_kg_m_per_s = {};
  }
  for (auto& face : state.adaptive_faces.values()) {
    face.validity = LiquidFaceValidity3D::kValid;
  }
  return ValidateLiquidState3D(state);
}

}  // namespace fluidity::test
