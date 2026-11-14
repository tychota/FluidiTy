// A 3x3 matrix, for turning a vector, mirroring it or scaling it.
//
// Purpose: give every module one small matrix type and the arithmetic it needs, on the CPU and the GPU.
//
// Idea: a Mat3 is nine doubles in three matrix rows of three. Multiplying a matrix by a vector gives a new vector:
//   each axis of the answer is the dot of one matrix row with the vector. A matrix can turn a vector without
//   changing its length (a rotation), or it can mirror it. For a matrix that keeps lengths, the determinant tells
//   the two apart: +1 for a rotation, -1 for a mirror image (papers call this a reflection).
//
// Words:
//   matrix row  three entries side by side. The first letter of an entry's name says which matrix row it is in.
//   transpose   the matrix mirrored along its diagonal, so entry xy and entry yx change places.
//   determinant a number that says how the matrix changes volume. Its sign says whether it keeps the order
//               in which the corners of a triangle go round.
// Owns:    Mat3, its arithmetic operators, kIdentity, math::Transpose, math::Determinant
// Uses:    Vec3 (base/vec3.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h)
// Used by: every module that turns or scales vectors
// Read:    docs/method-paper/fluidity-method.typ <sec-conventions>

#pragma once

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/vec3.h"

namespace fluidity {

// Nine doubles in matrix-row order. The first letter of an entry's name is its matrix row, the second its column.
// No default values: Mat3 stays a trivial type, so GPU memory can hold arrays of it without running constructors.
// Write Mat3{} for a matrix of zeros. Functions take a Mat3 by value, as they take a Vec3. GPU code may then pass
// kIdentity: the CUDA compiler copies the value of this CPU constant into GPU code, but GPU code cannot bind a
// reference to it, because the constant's address exists only on the CPU.
struct Mat3 {
  double xx;  // Matrix row x, column x. No unit of its own: the unit is the output's unit per input unit.
  double xy;  // Matrix row x, column y.
  double xz;  // Matrix row x, column z.
  double yx;  // Matrix row y, column x.
  double yy;  // Matrix row y, column y.
  double yz;  // Matrix row y, column z.
  double zx;  // Matrix row z, column x.
  double zy;  // Matrix row z, column y.
  double zz;  // Matrix row z, column z.
};

// The matrix that leaves every vector as it is: ones on the diagonal, zeros elsewhere.
inline constexpr Mat3 kIdentity = {
    .xx = 1,
    .xy = 0,
    .xz = 0,
    .yx = 0,
    .yy = 1,
    .yz = 0,
    .zx = 0,
    .zy = 0,
    .zz = 1,
};

// Multiplies a matrix by a vector. Each axis of the result is the dot of one matrix row with the vector.
//
// Example: with matrix rows {1, 2, 3}, {4, 5, 6}, {7, 8, 9} and the vector {1, 0, -1}, the result is
//   {1 - 3, 4 - 6, 7 - 9} = {-2, -2, -2}.
FLUIDITY_CPU_AND_GPU constexpr Vec3 operator*(Mat3 matrix, Vec3 vector) noexcept {
  return {
      .x = matrix.xx * vector.x + matrix.xy * vector.y + matrix.xz * vector.z,
      .y = matrix.yx * vector.x + matrix.yy * vector.y + matrix.yz * vector.z,
      .z = matrix.zx * vector.x + matrix.zy * vector.y + matrix.zz * vector.z,
  };
}

// Multiplies two matrices. The result does what the right matrix does first and the left matrix does second.
// The order matters: a turn about X followed by a turn about Z is not a turn about Z followed by a turn about X.
//
// Example: with matrix rows {1, 2, 3}, {4, 5, 6}, {7, 8, 9} on the left and {9, 8, 7}, {6, 5, 4}, {3, 2, 1} on the
//   right, the result has matrix rows {30, 24, 18}, {84, 69, 54}, {138, 114, 90}. Its entry xx is
//   1 * 9 + 2 * 6 + 3 * 3 = 30.
FLUIDITY_CPU_AND_GPU constexpr Mat3 operator*(Mat3 left, Mat3 right) noexcept {
  return {
      .xx = left.xx * right.xx + left.xy * right.yx + left.xz * right.zx,
      .xy = left.xx * right.xy + left.xy * right.yy + left.xz * right.zy,
      .xz = left.xx * right.xz + left.xy * right.yz + left.xz * right.zz,
      .yx = left.yx * right.xx + left.yy * right.yx + left.yz * right.zx,
      .yy = left.yx * right.xy + left.yy * right.yy + left.yz * right.zy,
      .yz = left.yx * right.xz + left.yy * right.yz + left.yz * right.zz,
      .zx = left.zx * right.xx + left.zy * right.yx + left.zz * right.zx,
      .zy = left.zx * right.xy + left.zy * right.yy + left.zz * right.zy,
      .zz = left.zx * right.xz + left.zy * right.yz + left.zz * right.zz,
  };
}

// Adds two matrices entry by entry.
FLUIDITY_CPU_AND_GPU constexpr Mat3 operator+(Mat3 left, Mat3 right) noexcept {
  return {
      .xx = left.xx + right.xx,
      .xy = left.xy + right.xy,
      .xz = left.xz + right.xz,
      .yx = left.yx + right.yx,
      .yy = left.yy + right.yy,
      .yz = left.yz + right.yz,
      .zx = left.zx + right.zx,
      .zy = left.zy + right.zy,
      .zz = left.zz + right.zz,
  };
}

// Subtracts the right matrix from the left one, entry by entry.
FLUIDITY_CPU_AND_GPU constexpr Mat3 operator-(Mat3 left, Mat3 right) noexcept {
  return {
      .xx = left.xx - right.xx,
      .xy = left.xy - right.xy,
      .xz = left.xz - right.xz,
      .yx = left.yx - right.yx,
      .yy = left.yy - right.yy,
      .yz = left.yz - right.yz,
      .zx = left.zx - right.zx,
      .zy = left.zy - right.zy,
      .zz = left.zz - right.zz,
  };
}

// Multiplies every entry by the same number.
FLUIDITY_CPU_AND_GPU constexpr Mat3 operator*(Mat3 matrix, double factor) noexcept {
  return {
      .xx = matrix.xx * factor,
      .xy = matrix.xy * factor,
      .xz = matrix.xz * factor,
      .yx = matrix.yx * factor,
      .yy = matrix.yy * factor,
      .yz = matrix.yz * factor,
      .zx = matrix.zx * factor,
      .zy = matrix.zy * factor,
      .zz = matrix.zz * factor,
  };
}

namespace math {

// Computes the transpose of a matrix: the entry at matrix row r, column c moves to matrix row c, column r.
//
// Example: the matrix rows {1, 2, 3}, {4, 5, 6}, {7, 8, 9} become {1, 4, 7}, {2, 5, 8}, {3, 6, 9}.
//   For a rotation, the transpose undoes the turn.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU constexpr Mat3 Transpose(Mat3 matrix) noexcept {
  return {
      .xx = matrix.xx,
      .xy = matrix.yx,
      .xz = matrix.zx,
      .yx = matrix.xy,
      .yy = matrix.yy,
      .yz = matrix.zy,
      .zx = matrix.xz,
      .zy = matrix.yz,
      .zz = matrix.zz,
  };
}

// Computes the determinant of a matrix by expanding along matrix row x.
//
// Units: the product of the entries' units, three times.
// Order: the three terms are added in x, y, z order.
// Example: the determinant of kIdentity is 1. Swapping the y and z axes gives -1, a mirror image.
//   The quarter turn with matrix rows {1, 0, 0}, {0, 0, -1}, {0, 1, 0} gives +1, a rotation.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU constexpr double Determinant(Mat3 matrix) noexcept {
  const double minor_x = matrix.yy * matrix.zz - matrix.yz * matrix.zy;
  const double minor_y = matrix.yx * matrix.zz - matrix.yz * matrix.zx;
  const double minor_z = matrix.yx * matrix.zy - matrix.yy * matrix.zx;
  return matrix.xx * minor_x - matrix.xy * minor_y + matrix.xz * minor_z;
}

}  // namespace math
}  // namespace fluidity
