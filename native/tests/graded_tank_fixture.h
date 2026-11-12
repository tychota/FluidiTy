#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
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

inline absl::Status
PopulateDeepGradedBandFixture3D(LiquidState3D& state, LiquidSceneInputs3D& scene, std::int32_t horizontal_cells = 1) {
  state.geometry.cell_width = 0.5;
  scene.field = LiquidHorizontalPlaneSurface3D{4};
  const double tank_width_m = 0.5 * horizontal_cells;
  scene.obstacle = AxisAlignedLiquidTank3D{0, tank_width_m, 0, 0, tank_width_m, 0};
  scene.world_acceleration = {};
  scene.liquid_density = 2;
  std::vector<AdaptiveMacCellKey3D> keys;
  for (std::int32_t z = 0; z < horizontal_cells; ++z) {
    for (std::int32_t y = 0; y < 8; ++y) {
      for (std::int32_t x = 0; x < horizontal_cells; ++x) {
        keys.push_back({.lower_corner = {x, y, z}, .width_exponent = 0});
      }
    }
  }
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
    return absl::ResourceExhaustedError("deep graded band fixture storage failed");
  }
  if (!cells->Resize(keys.size()).ok() || !particles->Resize(keys.size()).ok() || !faces->Resize(face_count).ok()) {
    return absl::InternalError("deep graded band fixture extent failed");
  }
  for (std::size_t ordinal = 0; ordinal < keys.size(); ++ordinal) {
    const auto& leaf = topology->cells()[ordinal];
    (*cells)[ordinal] = {
        .key = leaf.key,
        .inventory =
            {
                .mass_kg = 2 * leaf.cell_volume,
                .reference_volume_m3 = leaf.cell_volume,
                .current_volume_m3 = leaf.cell_volume,
            },
    };
    auto& particle = (*particles)[ordinal];
    particle.id = ordinal + 1;
    const auto lower = leaf.key.lower_corner;
    particle.world_position = {0.5 * lower.x + 0.25, 0.5 * lower.y + 0.25, 0.5 * lower.z + 0.25};
    particle.mass = 2 * leaf.cell_volume;
    particle.reference_volume = leaf.cell_volume;
    particle.pic_initialized = true;
  }
  for (auto& face : faces->values()) {
    face.validity = LiquidFaceValidity3D::kValid;
  }
  state.next_particle_id = keys.size() + 1;
  state.adaptive_cells = std::move(*cells);
  state.adaptive_faces = std::move(*faces);
  state.particles = std::move(*particles);
  return ValidateLiquidState3D(state);
}

inline void SetGradedTankEos3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0};
  scene.ambient_pressure = 1.0;
  for (auto& particle : state.particles.values()) {
    particle.world_velocity = {};
  }
  for (auto& cell : state.adaptive_cells.values()) {
    cell.inventory.momentum_kg_m_per_s = {};
  }
}

inline void SetGradedTankMiscible3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  static constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1.5}, {14, 4, 2.5}}};
  scene.constituent_materials = kMaterials;
  for (std::size_t ordinal = 0; ordinal < state.particles.size(); ++ordinal) {
    auto& particle = state.particles[ordinal];
    auto& inventory = state.adaptive_cells[ordinal].inventory;
    const double clear = ordinal % 2 == 0 ? 0.75 : 0.25;
    const double dyed = 1.0 - clear;
    particle.phase_id = 4;
    particle.constituent_count = 2;
    particle.constituents = {{{8, clear}, {14, dyed}}};
    particle.mass =
        particle.reference_volume * (clear * kMaterials[0].density_kg_per_m3 + dyed * kMaterials[1].density_kg_per_m3);
    inventory.mass_kg = particle.mass;
    inventory.momentum_kg_m_per_s = {
        particle.mass * particle.world_velocity.x,
        particle.mass * particle.world_velocity.y,
        particle.mass * particle.world_velocity.z,
    };
    inventory.phase_id = particle.phase_id;
    inventory.constituent_count = particle.constituent_count;
    inventory.constituents = {{{8, clear * particle.reference_volume}, {14, dyed * particle.reference_volume}}};
  }
}

inline void SetGradedTankMiscibleViscosity3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  static constexpr std::array<LiquidConstituentMaterial3D, 2> kMaterials{{{8, 4, 1.5, 1.0}, {14, 4, 2.5, 10.0}}};
  SetGradedTankMiscible3D(state, scene);
  scene.constituent_materials = kMaterials;
}

inline absl::StatusOr<double> GradedTankCarrierEnergy3D(const LiquidState3D& state, const LiquidSceneInputs3D& scene) {
  std::vector<AdaptiveMacCellKey3D> keys;
  for (const auto& cell : state.adaptive_cells.values()) {
    keys.push_back(cell.key);
  }
  const auto topology = AdaptiveMacTopology3D::Create(keys, state.geometry.cell_width);
  if (!topology.ok()) {
    return topology.status();
  }
  const auto ownership = BuildAdaptiveMacFaceOwnership3D(*topology);
  if (!ownership.ok()) {
    return ownership.status();
  }
  const auto energy = [&](std::size_t slot, double area, double distance) {
    const double speed = state.adaptive_faces[slot].carrier_velocity;
    return 0.5 * scene.liquid_density * area * distance * speed * speed;
  };
  double result = 0.0;
  for (std::size_t face = 0; face < topology->interior_faces().size(); ++face) {
    const auto& geometry = topology->interior_faces()[face];
    result += energy(ownership->interior_face_slots[face], geometry.face_area, geometry.cell_center_distance);
  }
  for (const auto& face : ownership->exterior_faces) {
    if (face.face_index.axis == FaceNormalAxis::kY && face.cell_is_tail) {
      result += energy(face.face_slot, face.face_area, topology->cells()[face.cell_ordinal].cell_width / 2.0);
    }
  }
  return result;
}

inline absl::Status PopulateGradedRefinementClosureFixture3D(
    LiquidState3D& state,
    LiquidSceneInputs3D& scene,
    bool cover_requested_children
) {
  state.geometry.cell_width = 0.5;
  scene.field = LiquidHorizontalPlaneSurface3D{2};
  scene.obstacle = AxisAlignedLiquidTank3D{0, 3, 0, 0, 2, 0};
  scene.world_acceleration = {};
  scene.liquid_density = 2;
  constexpr std::array keys{
      AdaptiveMacCellKey3D{.lower_corner = {0, 0, 0}, .width_exponent = 2},
      AdaptiveMacCellKey3D{.lower_corner = {4, 0, 0}, .width_exponent = 1},
      AdaptiveMacCellKey3D{.lower_corner = {4, 2, 0}, .width_exponent = 1},
      AdaptiveMacCellKey3D{.lower_corner = {4, 0, 2}, .width_exponent = 1},
      AdaptiveMacCellKey3D{.lower_corner = {4, 2, 2}, .width_exponent = 1},
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
  const std::size_t particle_count = cover_requested_children ? 19 : 12;
  auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(keys.size());
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(particle_count);
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(face_count);
  if (!cells.ok() || !particles.ok() || !faces.ok()) {
    return absl::ResourceExhaustedError("graded closure fixture storage failed");
  }
  if (!cells->Resize(keys.size()).ok() || !particles->Resize(particle_count).ok() || !faces->Resize(face_count).ok()) {
    return absl::InternalError("graded closure fixture extent failed");
  }
  std::size_t particle_ordinal = 0;
  for (std::size_t ordinal = 0; ordinal < topology->cells().size(); ++ordinal) {
    const auto& leaf = topology->cells()[ordinal];
    const auto lower = leaf.key.lower_corner;
    const std::size_t samples = ordinal == 0 || (ordinal == 1 && cover_requested_children) ? 8 : 1;
    const double particle_volume_m3 = leaf.cell_volume / static_cast<double>(samples);
    for (std::size_t sample = 0; sample < samples; ++sample) {
      const auto offset = [&](unsigned axis) {
        if (samples == 1) {
          return 0.5 * leaf.cell_width;
        }
        const bool high = ((sample >> axis) & 1) != 0;
        return (high ? 0.75 : 0.25) * leaf.cell_width;
      };
      auto& particle = (*particles)[particle_ordinal];
      particle.id = ++particle_ordinal;
      particle.mass = scene.liquid_density * particle_volume_m3;
      particle.reference_volume = particle_volume_m3;
      particle.world_position = {0.5 * lower.x + offset(0), 0.5 * lower.y + offset(1), 0.5 * lower.z + offset(2)};
      particle.pic_initialized = true;
    }
    (*cells)[ordinal] = {
        .key = leaf.key,
        .inventory =
            {
                .mass_kg = scene.liquid_density * leaf.cell_volume,
                .reference_volume_m3 = leaf.cell_volume,
                .current_volume_m3 = leaf.cell_volume,
            },
    };
  }
  for (auto& face : faces->values()) {
    face.validity = LiquidFaceValidity3D::kPredicted;
  }
  state.next_particle_id = particle_ordinal + 1;
  state.adaptive_cells = std::move(*cells);
  state.adaptive_faces = std::move(*faces);
  state.particles = std::move(*particles);
  return ValidateLiquidState3D(state);
}

}  // namespace fluidity::test
