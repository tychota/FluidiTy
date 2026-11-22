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
//   matrix row       three entries side by side. The first letter of an entry's name says which matrix row it
//                    is in.
//   transpose        the matrix mirrored along its diagonal, so entry xy and entry yx change places.
//   determinant      a number that says how the matrix changes volume. Its sign says whether it keeps the order
//                    in which the corners of a triangle go round.
//   rotation vector  a vector that describes a turn: its direction is the axis of the turn and its length is
//                    the angle in radians. The turn is right-handed: with the thumb of the right hand along the
//                    vector, the fingers curl the way the turn goes.
//   cross matrix     the matrix that computes a cross: the cross matrix of r times v equals Cross(r, v).
// Owns:    Mat3, its arithmetic operators, kIdentity, math::Transpose, math::Determinant, math::IsFinite(Mat3),
//          math::Rotation
// Uses:    Vec3 and math::Dot (base/vec3.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h),
//          math::Sqrt, math::Sin and math::IsFinite (base/scalar_math.h)
// Used by: every module that turns or scales vectors
// Read:    docs/method-paper/fluidity-method.typ <sec-conventions>
//          Rodrigues' rotation formula

#pragma once

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"
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

// Says whether every entry is an ordinary number: false when any entry is plus or minus infinity or NaN.
//
// Method: same bits on CPU and GPU, as IsFinite(double).
// Example: IsFinite(kIdentity) is true. kIdentity with entry zz set to NaN is not finite.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
FLUIDITY_CPU_AND_GPU inline bool IsFinite(Mat3 matrix) noexcept {
  const bool matrix_row_x_is_finite = IsFinite(matrix.xx) && IsFinite(matrix.xy) && IsFinite(matrix.xz);
  const bool matrix_row_y_is_finite = IsFinite(matrix.yx) && IsFinite(matrix.yy) && IsFinite(matrix.yz);
  const bool matrix_row_z_is_finite = IsFinite(matrix.zx) && IsFinite(matrix.zy) && IsFinite(matrix.zz);
  return matrix_row_x_is_finite && matrix_row_y_is_finite && matrix_row_z_is_finite;
}

// Computes the rotation that a rotation vector describes: a right-handed turn by Length(rotation_vector) radians
// about the direction of rotation_vector. A zero vector gives kIdentity exactly.
//
// Method: may differ in the last bits between CPU and GPU, because it uses Sin. With r the rotation vector, a its
//   length and K its cross matrix, Rodrigues' rotation formula gives
//     rotation = I + (sin(a) / a) K + ((1 - cos(a)) / a^2) K^2.
//   Below a = 1e-4 rad the two factors come from their short series instead, which never divides by a. So a tiny
//   vector, even one whose length underflows to zero, still gives the identity plus its first-order turn K.
// Units: the rotation vector is in radians. The matrix has no unit.
// Example: Rotation({0, 0, pi / 2}) is the quarter turn about Z with matrix rows {0, -1, 0}, {1, 0, 0}, {0, 0, 1},
//   within a few rounding units. It sends +X to +Y. Rotation({1e-12, 0, 0}) has entry zy = 1e-12 and
//   entry yz = -1e-12, and its diagonal is 1.
// Read: docs/method-paper/fluidity-method.typ <sec-conventions>
//       Rodrigues' rotation formula
FLUIDITY_CPU_AND_GPU inline Mat3 Rotation(Vec3 rotation_vector) noexcept {
  // Steps: 1 return the identity for a zero vector  2 compute the two factors  3 add up the three terms
  // Step 1: the identity, exactly, with no rounding from the formula.
  const bool is_zero_vector = rotation_vector.x == 0 && rotation_vector.y == 0 && rotation_vector.z == 0;
  if (is_zero_vector) {
    return kIdentity;
  }

  // Step 2: the factor of K is sin(a) / a, and the factor of K^2 is (1 - cos(a)) / a^2.
  // Below this angle the series 1 - a^2 / 6 and 1/2 - a^2 / 24 leave out terms under a^4 / 120 < 1e-18. That is
  // far below the rounding unit of 1, so the short series is as good as the full formula.
  constexpr double kShortSeriesAngleSquaredLimit = 1e-8;  // rad^2: an angle of 1e-4 rad.
  const double angle_squared = Dot(rotation_vector, rotation_vector);
  // Start from the short series; a larger angle replaces both factors with the full formula below.
  double cross_matrix_factor = 1 - angle_squared / 6;
  double cross_matrix_squared_factor = 0.5 - angle_squared / 24;
  if (angle_squared >= kShortSeriesAngleSquaredLimit) {
    const double angle = Sqrt(angle_squared);
    const double half_angle = 0.5 * angle;
    cross_matrix_factor = Sin(angle) / angle;
    // 1 - cos(a) = 2 sin(a / 2)^2. Computing 1 - Cos(a) directly would lose digits for a small angle.
    const double half_angle_sine_ratio = Sin(half_angle) / half_angle;
    cross_matrix_squared_factor = 0.5 * half_angle_sine_ratio * half_angle_sine_ratio;
  }

  // Step 3: with r = {x, y, z}, K has matrix rows {0, -z, y}, {z, 0, -x}, {-y, x, 0}, and K^2 = r r^T - a^2 I.
  const double x = rotation_vector.x;
  const double y = rotation_vector.y;
  const double z = rotation_vector.z;
  return {
      .xx = 1 - cross_matrix_squared_factor * (y * y + z * z),
      .xy = -cross_matrix_factor * z + cross_matrix_squared_factor * (x * y),
      .xz = cross_matrix_factor * y + cross_matrix_squared_factor * (x * z),
      .yx = cross_matrix_factor * z + cross_matrix_squared_factor * (x * y),
      .yy = 1 - cross_matrix_squared_factor * (x * x + z * z),
      .yz = -cross_matrix_factor * x + cross_matrix_squared_factor * (y * z),
      .zx = -cross_matrix_factor * y + cross_matrix_squared_factor * (x * z),
      .zy = cross_matrix_factor * x + cross_matrix_squared_factor * (y * z),
      .zz = 1 - cross_matrix_squared_factor * (x * x + y * y),
  };
}

}  // namespace math
}  // namespace fluidity
