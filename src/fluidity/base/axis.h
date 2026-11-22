// The axes of space, and the one place where the code base says which way is up.
//
// Purpose: state the axis convention once, so no module has to guess it, and name the axes and the up direction.
//
// Idea: positions use right-handed axes with Z up, as in Blender. Point the fingers of the right hand along X and
//   curl them toward Y: the thumb points along Z, so X cross Y is +Z. Gravity points along minus Z, {0, 0, -1}.
//   Many scene descriptions have Y up. A quarter turn about X brings a value from a Y-up description into this
//   convention. In matrix form it is the Mat3 with matrix rows {1, 0, 0}, {0, 0, -1}, {0, 1, 0}. It takes the old
//   gravity {0, -9.81, 0} to {0, 0, -9.81}. Its determinant is +1.
//   Swapping two axes instead looks as simple, but its determinant is -1: a swap is a reflection. It reverses the
//   order of the corners of every triangle, so a normal computed from the corners would point the wrong way.
//
// Words:
//   up  the direction opposite to gravity, +Z.
// Owns:    Axis, kUp, math::UnitVector
// Uses:    Vec3 (base/vec3.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h)
// Used by: every module
// Read:    docs/method-paper/fluidity-method.typ <sec-conventions>, <fig-axes>

#pragma once

#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/vec3.h"

namespace fluidity {

// One of the three axes of space.
enum class Axis : std::uint8_t {
  kX,  // Left to right: -X is the left side, +X the right side.
  kY,  // Front to back: -Y is the front side, +Y the back side.
  kZ,  // Bottom to top: -Z is the bottom side, +Z the top side. Up.
};

// The direction opposite to gravity. Gravity points along -kUp.
inline constexpr Vec3 kUp = {.x = 0, .y = 0, .z = 1};

namespace math {

// Computes the vector of length one that points along the positive direction of an axis.
//
// Example: UnitVector(Axis::kX) = {1, 0, 0}. UnitVector(Axis::kZ) = {0, 0, 1}, which equals kUp.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU constexpr Vec3 UnitVector(Axis axis) noexcept {
  switch (axis) {
    case Axis::kX:
      return {.x = 1, .y = 0, .z = 0};
    case Axis::kY:
      return {.x = 0, .y = 1, .z = 0};
    case Axis::kZ:
      break;
  }
  return kUp;
}

}  // namespace math
}  // namespace fluidity
