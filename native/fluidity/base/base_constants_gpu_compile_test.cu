// Proves that the shared headers of base/ compile as GPU code: label no_gpu.
//
// The GPU function below is never launched. Building it makes the CUDA compiler generate GPU code for every
// function of scalar_math.h, vec3.h, mat3.h, axis.h and tree_sum.h, for PerParticleVelocity<Vec3>, for reads of the
// class-type constants kUp and kIdentity at run time, and for a read of the double constant math::kPositiveInfinity.
// A shared function that calls a CPU-only function, or a constant that GPU code cannot read, fails this build. The
// test only takes the function's address, which needs no GPU.

#include <gtest/gtest.h>

#include <cstddef>

#include "fluidity/base/axis.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/tree_sum.h"
#include "fluidity/base/vec3.h"

namespace fluidity {
namespace {

// Reads its inputs from `values` (14 doubles) and writes one double per shared function to `results`, so the
// compiler cannot drop any call.
__global__ void ComputeWithEverySharedFunction(const double* values, double* results) {
  const double value = values[0];
  const double gravity = values[1];
  const Vec3 vector = {.x = values[2], .y = values[3], .z = values[4]};
  const Mat3 matrix = {
      .xx = values[5],
      .xy = values[6],
      .xz = values[7],
      .yx = values[8],
      .yy = values[9],
      .yz = values[10],
      .zx = values[11],
      .zy = values[12],
      .zz = values[13],
  };

  // The class-type constants, read at run time.
  const Vec3 gravity_vector = gravity * kUp;
  const Mat3 shifted_matrix = matrix + kIdentity;

  int result_index = 0;
  // scalar_math.h
  results[result_index++] = math::Floor(value);
  results[result_index++] = math::Abs(value);
  results[result_index++] = math::Sqrt(value);
  results[result_index++] = math::Cbrt(value);
  results[result_index++] = math::Exp(value);
  results[result_index++] = math::Expm1(value);
  results[result_index++] = math::Log(value);
  results[result_index++] = math::Pow(value, gravity);
  results[result_index++] = math::Sin(value);
  results[result_index++] = math::Cos(value);
  results[result_index++] = math::Min(value, gravity);
  results[result_index++] = math::Max(value, gravity);
  results[result_index++] = math::IsFinite(value) ? 1.0 : 0.0;
  results[result_index++] = math::Min(value, math::kPositiveInfinity);
  results[result_index++] = math::IsNan(value) ? 1.0 : 0.0;
  // vec3.h
  results[result_index++] = (vector + gravity_vector).x;
  results[result_index++] = (vector - gravity_vector).y;
  results[result_index++] = (-vector).z;
  results[result_index++] = (vector * value).x;
  results[result_index++] = (value * vector).y;
  results[result_index++] = math::Dot(vector, gravity_vector);
  results[result_index++] = math::Cross(vector, gravity_vector).x;
  results[result_index++] = math::Length(vector);
  results[result_index++] = math::Normalized(vector).z;
  results[result_index++] = math::IsFinite(vector) ? 1.0 : 0.0;
  // mat3.h
  results[result_index++] = (shifted_matrix * vector).x;
  results[result_index++] = (shifted_matrix * matrix).yy;
  results[result_index++] = (shifted_matrix - matrix).zz;
  results[result_index++] = (shifted_matrix * value).xy;
  results[result_index++] = math::Transpose(shifted_matrix).yx;
  results[result_index++] = math::Determinant(shifted_matrix);
  results[result_index++] = math::IsFinite(shifted_matrix) ? 1.0 : 0.0;
  results[result_index++] = math::Rotation(vector).xz;
  // axis.h
  results[result_index++] = math::UnitVector(Axis::kY).y;
  // tree_sum.h
  results[result_index++] = math::TreeSum(values, 14);
  results[result_index++] = math::TreeSumOf(3, [values](std::size_t index) {
    return values[index] * values[index];
  });
  results[result_index++] = math::LargerMagnitude(value, gravity);
  // per_particle_velocity.h
  const PerParticleVelocity<Vec3> particle_velocity = {.for_next_step = vector, .for_this_step = gravity_vector};
  results[result_index++] = particle_velocity.for_next_step.x + particle_velocity.for_this_step.z;
}

TEST(ComputeWithEverySharedFunction, BuiltForTheGpu_HasAnAddress) {
  const auto function_address = &ComputeWithEverySharedFunction;
  EXPECT_NE(function_address, nullptr);
}

}  // namespace
}  // namespace fluidity
