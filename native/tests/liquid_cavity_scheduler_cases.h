#pragma once

#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity::cavity_test {

// A filled 9³ lattice with a connected interior void. Radius 0.9 m covers
// the lattice corners in the liquid shell at h=1 m.
inline absl::StatusOr<LiquidState3D> ParticleShell() {
  constexpr std::size_t kMaximumParticles = 729;
  auto particles = LiquidRecords3D<LiquidParticleState3D>::Create(kMaximumParticles);
  if (!particles.ok()) {
    return particles.status();
  }
  const auto resized = particles->Resize(kMaximumParticles);
  if (!resized.ok()) {
    return resized;
  }
  std::size_t count = 0;
  for (int z = -4; z <= 4; ++z) {
    for (int y = -4; y <= 4; ++y) {
      for (int x = -4; x <= 4; ++x) {
        if (x * x + y * y + z * z >= 15) {
          (*particles)[count] = {
              .id = count + 1,
              .world_position = {8.5 + x, 8.5 + y, 8.5 + z},
              .pic_initialized = true,
          };
          ++count;
        }
      }
    }
  }
  const auto trimmed = particles->Resize(count);
  if (!trimmed.ok()) {
    return trimmed;
  }
  LiquidState3D state;
  state.next_particle_id = count + 1;
  state.particles = std::move(*particles);
  return state;
}

inline LiquidStepOptions3D ShellOptions(std::size_t particle_count) {
  LiquidStepOptions3D options;
  options.capacities = {particle_count, 128, 2048, 24576, 1};
  options.capacity_replay_reservations = {particle_count, 128, 4096, 24576, 1};
  options.maximum_step_duration = 0.001;
  return options;
}

}  // namespace fluidity::cavity_test
