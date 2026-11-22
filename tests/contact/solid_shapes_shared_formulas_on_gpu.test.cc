// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Launches the shared contact formulas in GPU code on 32 fixed inputs and compares each result with the CPU's. For a
// solid that neither turns nor spins the results must have the same bits (or be NaN on both), as the headers promise;
// for a turning or spinning solid they are compared within the GPU bounds of method paper <sec-contact>. The test
// prints the largest difference it saw, so the hardware run can calibrate the bounds.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include "fluidity/base/mat3.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/shared_formulas_runner_gpu.h"

namespace fluidity::contact {
namespace {

constexpr int kInputCount = 32;

// Offsets from a solid's origin, in its own axes before any turn: the origin itself (a sphere's centre and a point
// on a cylinder's axis, where the degenerate normals apply), points outside past a wall, past a cap, past a rim, inside
// near a face, and two oblique points.
constexpr std::array<Vec3, 8> kOffsets = {
    Vec3{.x = 0, .y = 0, .z = 0},
    Vec3{.x = 0.7, .y = 0, .z = 0.1},
    Vec3{.x = 0.1, .y = 0.05, .z = 0.9},
    Vec3{.x = 0.6, .y = 0.2, .z = 0.8},
    Vec3{.x = 0.05, .y = -0.1, .z = 0.15},
    Vec3{.x = 0, .y = 0, .z = 0.2},
    Vec3{.x = -0.45, .y = 0.3, .z = -0.35},
    Vec3{.x = 1.5, .y = -1.25, .z = 0.75},
};

// The solid of input `input_index`: the six shapes in turn, moving; inputs 24 to 31 also turn or spin.
Solid BuildInputSolid(int input_index) {
  const double share = input_index / 32.0;
  const Vec3 center = {.x = 1 + share, .y = -2 + (3 * share), .z = 0.5 - share};  // m
  const Vec3 direction = math::Normalized({.x = 0.3 - share, .y = 0.2 + share, .z = 1});
  const bool turns = input_index >= 24;
  const SolidMotion motion = {
      .linear_velocity = {.x = 0.5 - share, .y = share, .z = -0.25},
      .angular_velocity = turns ? Vec3{.x = 0.2, .y = -0.1 * share, .z = 1.5} : Vec3{},
  };
  const auto solid_id = static_cast<std::uint32_t>(1 + input_index);
  switch (input_index % 6) {
    case 0:
      return ToSolid(FlatSolid{.point_on_surface = center, .normal_out_of_solid = direction}, motion, solid_id);
    case 1:
      return ToSolid(Plate{.center = center, .normal = direction, .half_thickness = 0.05}, motion, solid_id);
    case 2:
      return ToSolid(
          Box{
              .center = center,
              .axes = math::Rotation({.x = 0.3, .y = 0.1, .z = -0.2}),
              .half_size = {.x = 0.4, .y = 0.3, .z = 0.5},
          },
          motion,
          solid_id
      );
    case 3:
      return ToSolid(
          DomainBox{
              .low_corner = center - Vec3{.x = 2, .y = 2, .z = 2},
              .high_corner = center + Vec3{.x = 2, .y = 3, .z = 4},
          },
          solid_id
      );
    case 4:
      return ToSolid(Sphere{.center = center, .radius = 0.5}, motion, solid_id);
    default:
      break;
  }
  return ToSolid(Cylinder{.center = center, .axis = direction, .radius = 0.5, .half_height = 0.6}, motion, solid_id);
}

// Input `input_index`: its solid, a point at one of kOffsets from the solid's origin along its own axes, a time into
// the step, a velocity and a friction. The offset index input_index + input_index / 6 moves on by one more each round
// of the six shapes, so each shape meets five different offsets: the cylinders (inputs 5, 11, 17, 23, 29) get the axis
// point, the inside point, the rim, the cap and the wall; the spheres (4, 10, 16, 22, 28) end on the centre.
SharedFormulaInputs BuildInput(int input_index) {
  const Solid solid = BuildInputSolid(input_index);
  const Vec3 offset = kOffsets[static_cast<std::size_t>(input_index + (input_index / 6)) % kOffsets.size()];
  const double share = input_index / 32.0;
  return {
      .solid = solid,
      .point = solid.pose_at_step_start.origin + solid.pose_at_step_start.axes * offset,
      .elapsed = input_index % 3 == 0 ? 0 : 0.01 * share,
      .velocity = {.x = 1 - share, .y = -0.5, .z = -2 + share},
      .friction = 0.1 + share,
  };
}

// The GPU bound of method paper <sec-contact>, with the CPU value as reference, for turning or spinning solids: every
// value here is of order 1, so a floor of 2^-40 stands for each kind's floor; the relative term is 2^-40.
constexpr Tolerance kTurningTolerance = {.absolute_floor = 0x1p-40, .relative_term = 0x1p-40};

// Checks the GPU's values of one input against the CPU's: the same bits (or NaN on both) for a solid that neither
// turns nor spins, kTurningTolerance otherwise. Returns the largest difference.
double CheckInput(const SharedFormulaInputs& input, const SharedFormulaValues& gpu_values) {
  const SharedFormulaValues cpu_values = ComputeSharedFormulaValues(input);
  const Vec3 angular_velocity = input.solid.motion.angular_velocity;
  const bool turns = angular_velocity.x != 0 || angular_velocity.y != 0 || angular_velocity.z != 0;
  double largest_difference = 0;
  for (int value_index = 0; value_index < kSharedFormulaResultCount; ++value_index) {
    const double expected = cpu_values.values[value_index];
    const double actual = gpu_values.values[value_index];
    const ::testing::AssertionResult agrees =
        turns ? CheckWithinTolerance(expected, actual, kTurningTolerance) : CheckSameBitsOrBothNaN(expected, actual);
    EXPECT_TRUE(agrees) << "value " << value_index;
    const double difference = std::abs(expected - actual);
    largest_difference = std::isnan(difference) ? largest_difference : std::max(largest_difference, difference);
  }
  return largest_difference;
}

TEST(SharedFormulas, OnGpu_MatchCpu) {
  std::vector<SharedFormulaInputs> inputs;
  inputs.reserve(kInputCount);
  for (int input_index = 0; input_index < kInputCount; ++input_index) {
    inputs.push_back(BuildInput(input_index));
  }
  std::vector<SharedFormulaValues> gpu_values(kInputCount);
  ASSERT_EQ(ComputeSharedFormulasOnGpu(inputs.data(), kInputCount, gpu_values.data()), RunOnCallStatus::kDone);
  double largest_turning_difference = 0;
  for (int input_index = 0; input_index < kInputCount; ++input_index) {
    SCOPED_TRACE(input_index);
    const double difference = CheckInput(inputs[input_index], gpu_values[input_index]);
    largest_turning_difference = std::max(largest_turning_difference, difference);
  }
  std::cout << "largest difference, turning or spinning solids included: " << largest_turning_difference << "\n";
  const std::span<const SharedFormulaValues> digested(gpu_values);
  WriteResultDigest("Shared contact formulas on the GPU, 32 inputs", digested);
}

}  // namespace
}  // namespace fluidity::contact
