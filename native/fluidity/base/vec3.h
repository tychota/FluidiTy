// A vector with three axes, for positions, velocities, forces and any other quantity with a direction.
//
// Purpose: give every module one small vector type and the arithmetic it needs, on the CPU and the GPU.
//
// Idea: a Vec3 is three doubles, one for each of the axes x, y and z. The axes are right-handed and Z is up, as
//   in Blender: X cross Y is +Z. The vector carries no unit. It holds meters for a position, m/s for a velocity,
//   or any other unit, and the name of each variable says which.
//
// Words:
//   dot    a number: the sum of the products of matching axes. It is zero when two vectors are perpendicular.
//   cross  a vector perpendicular to both inputs, whose direction follows the right-hand rule.
//   length the distance from the origin to the tip of the vector: the square root of the dot of the vector with
//          itself.
// Owns:    Vec3, its arithmetic operators, math::Dot, math::Cross, math::Length, math::Normalized,
//          math::IsFinite(Vec3)
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), math::Sqrt and math::IsFinite (base/scalar_math.h)
// Used by: every module
// Read:    docs/method-paper/fluidity-method.typ <sec-conventions>

#pragma once

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"

namespace fluidity {

// Three doubles in x, y, z order. Every axis is in the unit of the quantity the vector holds.
// No default values: Vec3 stays a trivial type, so GPU memory can hold arrays of it without running constructors.
// Write Vec3{} for a zero vector.
struct Vec3 {
  double x;  // Along +X, in the unit of the quantity.
  double y;  // Along +Y, in the unit of the quantity.
  double z;  // Along +Z, which points up, in the unit of the quantity.
};

// Adds two vectors axis by axis.
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator+(Vec3 left, Vec3 right) noexcept {
  return {.x = left.x + right.x, .y = left.y + right.y, .z = left.z + right.z};
}

// Subtracts the right vector from the left one, axis by axis.
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator-(Vec3 left, Vec3 right) noexcept {
  return {.x = left.x - right.x, .y = left.y - right.y, .z = left.z - right.z};
}

// Flips the sign of every axis.
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator-(Vec3 vector) noexcept {
  return {.x = -vector.x, .y = -vector.y, .z = -vector.z};
}

// Multiplies every axis by the same number.
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator*(Vec3 vector, double factor) noexcept {
  return {.x = vector.x * factor, .y = vector.y * factor, .z = vector.z * factor};
}

// Multiplies every axis by the same number, written with the number first.
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator*(double factor, Vec3 vector) noexcept {
  return vector * factor;
}

namespace math {

// Computes the dot of two vectors: first.x * second.x + first.y * second.y + first.z * second.z.
//
// Units: the product of the two inputs' units.
// Order: the three products are added in x, y, z order.
// Example: Dot({1, 2, 3}, {4, 5, 6}) = 4 + 10 + 18 = 32. Dot({1, 0, 0}, {0, 1, 0}) = 0.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU constexpr double Dot(Vec3 first, Vec3 second) noexcept {
  return first.x * second.x + first.y * second.y + first.z * second.z;
}

// Computes the cross of two vectors, first cross second. The order matters: swapping the inputs flips the sign.
//
// Units: the product of the two inputs' units.
// Example: Cross({1, 0, 0}, {0, 1, 0}) = {0, 0, 1}, because the axes are right-handed.
//   Cross({1, 2, 3}, {1, 2, 3}) = {0, 0, 0}: a vector crossed with itself is zero.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU constexpr Vec3 Cross(Vec3 first, Vec3 second) noexcept {
  return {
      .x = first.y * second.z - first.z * second.y,
      .y = first.z * second.x - first.x * second.z,
      .z = first.x * second.y - first.y * second.x,
  };
}

// Computes the length of a vector: the square root of Dot(vector, vector).
//
// Method: same bits on CPU and GPU, unless the result is NaN (an axis is NaN), whose bits IEEE 754 leaves open. It
//   uses only multiply, add and Sqrt.
//   The result is accurate when every non-zero axis is between about 1.5e-154 and 7e153.
//   The square of an axis smaller than about 1.5e-154 loses precision, and below about 1e-162 it becomes zero.
//   An axis larger than about 1.3e154 overflows its square to infinity. Three equal axes larger than about 7.7e153
//   overflow the sum of the squares, which tops out near 1.8e308.
// Units: the unit of the vector.
// Order: the three squares are added in x, y, z order, as Dot adds them.
// Example: Length({3, 4, 0}) = 5. Length({2, -3, 6}) = Sqrt(4 + 9 + 36) = 7.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline double Length(Vec3 vector) noexcept {
  return Sqrt(Dot(vector, vector));
}

// Computes the vector of length one that points the same way: each axis divided by Length(vector).
// The caller makes sure every non-zero axis is in the accurate range of Length, about 1.5e-154 to 7e153, and that
// at least one axis is non-zero. This is not checked.
//
// Method: same bits on CPU and GPU, except an axis that is NaN, whose bits IEEE 754 leaves open. It uses Length and
//   one correctly rounded division per axis.
// Units: none. The input's unit cancels.
// Example: Normalized({2, -3, 6}) = {2 / 7, -3 / 7, 6 / 7}.
//   Outside the range the answer is wrong: Normalized({1e200, 0, 0}) gives zeros because the length overflowed to
//   infinity. Normalized({1e-170, 0, 0}) gives {infinity, NaN, NaN}, because the length became zero: 1e-170 / 0 is
//   infinite and 0 / 0 is NaN. A zero vector gives NaN on every axis.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline Vec3 Normalized(Vec3 vector) noexcept {
  const double length = Length(vector);
  return {.x = vector.x / length, .y = vector.y / length, .z = vector.z / length};
}

// Says whether every axis is an ordinary number: false when any axis is plus or minus infinity or NaN.
//
// Method: same bits on CPU and GPU, as IsFinite(double).
// Example: IsFinite({1, -2, 3}) is true. IsFinite({0, NaN, 0}) is false.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline bool IsFinite(Vec3 vector) noexcept {
  return IsFinite(vector.x) && IsFinite(vector.y) && IsFinite(vector.z);
}

}  // namespace math
}  // namespace fluidity
