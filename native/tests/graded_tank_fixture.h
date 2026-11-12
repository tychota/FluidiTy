#pragma once

#include <array>
#include <cmath>
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

// A real divergence-free circulation in the lower X/Z plane. The particle-owned
// cap separates the core from the open surface. Equal support weights on the
// two eligible carriers in each horizontal direction give zero PIC motion.
inline absl::Status PopulateGradedOwnerExchangeFixture3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  state.geometry.cell_width = 1;
  scene.field = LiquidHorizontalPlaneSurface3D{2};
  scene.obstacle = AxisAlignedLiquidTank3D{0, 2, 0, 0, 2, 0};
  scene.world_acceleration = {};
  scene.liquid_density = 1;
  scene.ambient_pressure = 0;

  std::array<AdaptiveMacCellKey3D, 8> keys{};
  std::size_t ordinal = 0;
  for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        keys[ordinal++].lower_corner = {x, y, z};
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
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(face_count);
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(3);
  if (!cells.ok() || !faces.ok() || !particles.ok()) {
    return absl::ResourceExhaustedError("graded exchange fixture storage failed");
  }
  if (!cells->Resize(keys.size()).ok() || !faces->Resize(face_count).ok() || !particles->Resize(3).ok()) {
    return absl::InternalError("graded exchange fixture extents failed");
  }
  for (std::size_t cell = 0; cell < keys.size(); ++cell) {
    (*cells)[cell].key = keys[cell];
  }
  (*cells)[0].material_owner = AdaptiveLiquidMaterialOwner3D::kEulerian;
  (*cells)[0].inventory = {.mass_kg = 0.5, .reference_volume_m3 = 0.5, .current_volume_m3 = 0.5};
  constexpr std::array particle_leaves{1U, 4U, 5U};
  constexpr std::array particle_masses{0.125, 0.25, 0.125};
  const double low_side = std::nextafter(1.0, 0.0);
  const std::array positions{Vec3{1, 0.5, low_side}, Vec3{low_side, 0.5, 1}, Vec3{1, 0.5, 1}};
  for (std::size_t particle = 0; particle < particle_masses.size(); ++particle) {
    const double mass = particle_masses[particle];
    (*particles)[particle] = {
        .id = particle + 1,
        .world_position = positions[particle],
        .world_velocity = {-1, 0, 1},
        .affine_enabled = false,
        .mass = mass,
        .reference_volume = mass,
    };
    (*cells)[particle_leaves[particle]].inventory = {
        .mass_kg = mass,
        .reference_volume_m3 = mass,
        .current_volume_m3 = mass,
        .momentum_kg_m_per_s = {-mass, 0, mass},
    };
  }
  for (auto& face : faces->values()) {
    face = {.validity = LiquidFaceValidity3D::kValid, .structurally_eligible = false};
  }
  for (std::size_t face_ordinal = 0; face_ordinal < topology->interior_faces().size(); ++face_ordinal) {
    const auto& face = topology->interior_faces()[face_ordinal];
    const auto lower = face.face_index.lattice_index;
    auto& carrier = (*faces)[ownership->interior_face_slots[face_ordinal]];
    if (face.face_index.axis == FaceNormalAxis::kY) {
      carrier.structurally_eligible = true;
    } else if (lower.y == 0) {
      carrier.structurally_eligible = true;
      if (face.face_index.axis == FaceNormalAxis::kX) {
        carrier.carrier_velocity = lower.z == 0 ? 1 : -1;
      } else {
        carrier.carrier_velocity = lower.x == 0 ? -1 : 1;
      }
    }
  }
  for (const auto& face : ownership->exterior_faces) {
    (*faces)[face.face_slot].structurally_eligible = face.face_index.axis == FaceNormalAxis::kY;
  }
  state.next_particle_id = 4;
  state.particles = std::move(*particles);
  state.adaptive_cells = std::move(*cells);
  state.adaptive_faces = std::move(*faces);
  return ValidateLiquidState3D(state);
}

// A fully wet same-level island adjoining coarser leaves.
inline absl::StatusOr<std::uint64_t>
PopulateGradedQuadraticTankFixture3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  constexpr int kExtent = 4;
  constexpr int kCoarseWidth = 2;
  constexpr std::size_t kCount = kExtent * kExtent * kExtent + 4;
  state.geometry.cell_width = 1;
  scene.field = LiquidHorizontalPlaneSurface3D{kExtent};
  scene.obstacle = AxisAlignedLiquidTank3D{0, kExtent + kCoarseWidth, 0, 0, kExtent, 0};
  scene.world_acceleration = {};
  scene.liquid_density = 1;
  std::vector<AdaptiveMacCellKey3D> keys;
  for (int z = 0; z < kExtent; ++z) {
    for (int y = 0; y < kExtent; ++y) {
      for (int x = 0; x < kExtent; ++x) {
        keys.push_back({.lower_corner = {x, y, z}, .width_exponent = 0});
      }
    }
  }
  for (int z = 0; z < kExtent; z += kCoarseWidth) {
    for (int y = 0; y < kExtent; y += kCoarseWidth) {
      keys.push_back({.lower_corner = {kExtent, y, z}, .width_exponent = 1});
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
  auto cells = LiquidRecords3D<LiquidAdaptiveCellState3D>::Create(kCount);
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(kCount);
  auto faces = LiquidRecords3D<LiquidAdaptiveFaceState3D>::Create(face_count);
  if (!cells.ok() || !particles.ok() || !faces.ok()) {
    return absl::ResourceExhaustedError("quadratic tank fixture storage failed");
  }
  if (!cells->Resize(kCount).ok() || !particles->Resize(kCount).ok() || !faces->Resize(face_count).ok()) {
    return absl::InternalError("quadratic tank fixture extent failed");
  }
  std::uint64_t selected_id = 0;
  for (std::size_t ordinal = 0; ordinal < kCount; ++ordinal) {
    const auto& leaf = topology->cells()[ordinal];
    const auto lower = leaf.key.lower_corner;
    auto& particle = (*particles)[ordinal];
    particle.id = ordinal + 1;
    particle.mass = leaf.cell_volume;
    particle.reference_volume = leaf.cell_volume;
    const double half_width = 0.5 * leaf.cell_width;
    particle.world_position = {lower.x + half_width, lower.y + half_width, lower.z + half_width};
    particle.pic_initialized = true;
    if (lower.x == 1 && lower.y == 1 && lower.z == 1) {
      selected_id = particle.id;
      particle.world_position = {1.75, 1.75, 1.75};
      particle.world_velocity = {0.015625, -0.03125, 0.0625};
    }
    (*cells)[ordinal] = {
        .key = leaf.key,
        .inventory =
            {
                .mass_kg = leaf.cell_volume,
                .reference_volume_m3 = leaf.cell_volume,
                .current_volume_m3 = leaf.cell_volume,
                .momentum_kg_m_per_s = particle.world_velocity,
            },
    };
  }
  for (auto& face : faces->values()) {
    face.validity = LiquidFaceValidity3D::kValid;
  }
  state.next_particle_id = kCount + 1;
  state.adaptive_cells = std::move(*cells);
  state.adaptive_faces = std::move(*faces);
  state.particles = std::move(*particles);
  const auto valid = ValidateLiquidState3D(state);
  if (!valid.ok()) {
    return valid;
  }
  return selected_id;
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

inline absl::Status PopulateDeepGradedBandFixture3D(LiquidState3D& state, LiquidSceneInputs3D& scene) {
  state.geometry.cell_width = 0.5;
  scene.field = LiquidHorizontalPlaneSurface3D{4};
  scene.obstacle = AxisAlignedLiquidTank3D{0, 0.5, 0, 0, 0.5, 0};
  scene.world_acceleration = {};
  scene.liquid_density = 2;
  std::array<AdaptiveMacCellKey3D, 8> keys{};
  for (std::size_t ordinal = 0; ordinal < keys.size(); ++ordinal) {
    keys[ordinal].lower_corner.y = static_cast<std::int32_t>(ordinal);
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
    particle.world_position = {0.25, 0.5 * leaf.key.lower_corner.y + 0.25, 0.25};
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

inline absl::Status PopulateGradedPhysicalHalfspaceFixture3D(
    LiquidState3D& state,
    LiquidSceneInputs3D& scene,
    TranslatingHalfspace3D wall
) {
  const auto authored = PopulateGradedBandFixture3D(state, scene);
  if (!authored.ok()) {
    return authored;
  }
  scene.obstacle = wall;
  for (std::size_t cell = 0; cell < state.adaptive_cells.size(); ++cell) {
    auto& particle = state.particles[cell];
    auto& inventory = state.adaptive_cells[cell].inventory;
    particle.mass *= 0.5;
    particle.reference_volume *= 0.5;
    inventory.mass_kg = particle.mass;
    inventory.reference_volume_m3 = particle.reference_volume;
    inventory.current_volume_m3 = particle.reference_volume;
  }
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
