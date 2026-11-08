#pragma once
#include <cstddef>
#include <cstdint>

#include "fluidity/base/mac_types_3d.h"
namespace fluidity {
struct MacTransferParticle3D {
  Vec3 sorted_world_position;  // World position at the last sort, in meters.
  Vec3 trial_world_position;  // World position used for this transfer, in meters.
  double particle_mass;  // Kilograms.
  Vec3 particle_velocity;  // Cartesian particle velocity in meters per second.
  Mat3 particle_velocity_slopes;  // APIC slope rows in inverse seconds, ordered by velocity component.
};

struct MacDepositQuantization3D {
  double mass_per_quantum;  // Kilograms per quantum.
  double pure_momentum_per_quantum;  // Kilogram meters per second per quantum.
  std::size_t declared_particle_capacity;  // Declared upper bound on the particle count.
};

struct MacFaceDeposit3D {
  double deposited_mass;  // Kilograms.
  double deposited_pure_momentum;  // Normal momentum in kilogram meters per second.
};

struct MacFaceDepositQuanta3D {
  std::int64_t mass_quanta_sum;  // Mass quanta.
  std::int64_t pure_momentum_quanta_sum;  // Normal momentum quanta.
  std::int64_t stress_force_quanta_sum;  // Reserved stress-force channel, zero for liquid.
};

struct MacParticleGatherResult3D {
  Vec3 new_pic_velocity;  // Interpolated new grid velocity in meters per second.
  Vec3 grid_velocity_increment;  // Interpolated new minus old grid velocity in meters per second.
  Mat3 fitted_velocity_slopes;  // Weighted least-squares velocity slopes in inverse seconds.
  Mat3 velocity_gradient;  // Derivative of interpolated velocity in inverse seconds.
};

}  // namespace fluidity
