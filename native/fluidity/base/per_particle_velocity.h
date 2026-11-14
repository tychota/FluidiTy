// The two velocities that one particle carries.
//
// Purpose: give every module one type for "the velocity kept on the particle" and "the velocity that moves it", so
//   no function can swap them by passing two bare vectors in the wrong order.
//
// Idea: a particle is both a carrier of motion and a record of it. Each step, the grid computes a new velocity
//   field, and the particle takes its velocity from the grid. The velocity that moves the particle during this
//   step is not always the one it keeps for the next step. Example: a particle that hits a solid that does not
//   react moves along the solid for the rest of this step, but keeps the speed it had before the hit, so the next
//   step does not start from a velocity that the solid changed. (A reacting solid may change the kept one too.)
//   Papers call the kept one the particle state and the other one the travel velocity.
//
// Words:
//   for_next_step  the velocity the particle keeps until the next step
//   for_this_step  the velocity that moves the particle during this step
// Owns:    PerParticleVelocity
// Uses:    nothing
// Used by: contact, particles, transfer, step
// Read:    docs/method-paper/fluidity-method.typ <sec-particle-velocities>

#pragma once

namespace fluidity {

// One value for each of the particle's two velocities. `T` is Vec3 for a velocity and double for a speed.
// No default values: the type stays trivial, so GPU memory can hold arrays of it. Write PerParticleVelocity<T>{}
// for zeros.
template <typename T>
struct PerParticleVelocity {
  T for_next_step;  // Kept on the particle for the next step.
  T for_this_step;  // Moves the particle in this step.
};

}  // namespace fluidity
