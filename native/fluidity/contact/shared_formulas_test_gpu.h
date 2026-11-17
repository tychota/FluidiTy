// The shared contact formulas evaluated for one input, and their GPU runner, for shared_formulas_gpu_test.cc, private
// to contact/.
//
// Purpose: run the contact formulas whose headers promise the same bits on CPU and GPU on fixed inputs, on both
//   processors, so the test can compare each result with the CPU's.
//
// Idea: ComputeSharedFormulaValues computes, for one solid and one point, ComputeDistanceToSolid, FindSolidSide,
//   ComputeSolidVelocitySample, ComputeContactResponse and ComputeContactTolerance, every field as a double in a fixed
//   order. It is marked FLUIDITY_CPU_AND_GPU, so the GPU function and the CPU loop of the test run the same code.
//   ComputeSharedFormulasOnGpu copies the inputs to the GPU, runs one thread per input and copies the results back.
//   The promise: the same bits, unless NaN, when the solid's angular velocity is zero (only + - * /, Abs and Sqrt); a
//   turning or spinning solid is posed through math::Rotation, whose Sin may differ in the last bits.
//
// Owns:    kSharedFormulaResultCount, SharedFormulaInputs, SharedFormulaValues, ComputeSharedFormulaValues,
//          ComputeSharedFormulasOnGpu
// Uses:    Vec3 (base/vec3.h), RunOnCallStatus (base/run_on_call.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h),
//          Solid, ComputeSolidPose, ComputeDistanceToSolid, FindSolidSide, ComputeSolidVelocity,
//          ComputeSolidVelocitySample (contact/solid_shapes.h), ComputeContactResponse
//          (contact/solid_contact_types.h), ContactScale, ComputeContactTolerance (contact/contact_tolerance.h)
// Used by: contact/shared_formulas_gpu_test.cc, and contact/shared_formulas_test_gpu.cu, which defines
//          ComputeSharedFormulasOnGpu
// Read:    docs/method-paper/fluidity-method.typ <sec-solid-shapes>, <sec-contact-response>, <sec-contact-tolerance>

#pragma once

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// The results of one input, in the order ComputeSharedFormulaValues lists them: the distance (5), the plate side (1),
// the velocity sample (10), two responses (6), the tolerance (1), the sample's distance (1) and the pose's origin x
// (1).
inline constexpr int kSharedFormulaResultCount = 25;

// One fixed input: a solid, a point, a time into the step, a particle velocity and a friction.
struct SharedFormulaInputs {
  Solid solid;
  Vec3 point;  // m, world.
  double elapsed;  // s, zero or more.
  Vec3 velocity;  // m/s, world: the particle velocity the response reads.
  double friction;  // No unit, zero or more.
};

// The results of one input, every field as a double, in the order of ComputeSharedFormulaValues.
struct SharedFormulaValues {
  // A C array, so the struct is copied between the processors as plain bytes.
  double values[kSharedFormulaResultCount];  // NOLINT(modernize-avoid-c-arrays): see the comment above.
};

// Every shared formula of one input. Discrete fields (faces, sides, indexes, flags) become doubles exactly.
//
// Method: pose = ComputeSolidPose(solid, elapsed); distance = ComputeDistanceToSolid(solid, pose, point) (values 0 to
//   4: distance, normal, face); FindSolidSide(solid, pose, point) (5); ComputeSolidVelocitySample(&solid, 1, elapsed,
//   point, 10 m) (6 to 15: has_solid, solid_index, side's plate id and side, normal, solid velocity); then
//   ComputeContactResponse(velocity, ComputeSolidVelocity(solid, pose, point), distance's normal, friction) (16 to 18
//   and, for the second velocity read, 19 to 21 with half the friction); ComputeContactTolerance of a scale whose
//   largest coordinate is 1 + |point.x| + |point.y| + |point.z| (22); the sample's distance (23) and the pose's origin
//   x (24).
// Requires: the solid passed CheckSolid; elapsed is finite and zero or more; the point and velocity are finite.
FLUIDITY_CPU_AND_GPU inline SharedFormulaValues ComputeSharedFormulaValues(SharedFormulaInputs input) {
  SharedFormulaValues results{};
  const Solid& solid = input.solid;
  const SolidPose pose = ComputeSolidPose(solid, input.elapsed);
  const DistanceToSolid distance = ComputeDistanceToSolid(solid, pose, input.point);
  results.values[0] = distance.distance;
  results.values[1] = distance.normal_out_of_solid.x;
  results.values[2] = distance.normal_out_of_solid.y;
  results.values[3] = distance.normal_out_of_solid.z;
  results.values[4] = static_cast<double>(distance.face);
  results.values[5] = static_cast<double>(FindSolidSide(solid, pose, input.point));
  constexpr double kSearchDistance = 10;  // m: every input's solid is nearer.
  const SolidVelocitySample sample = ComputeSolidVelocitySample(&solid, 1, input.elapsed, input.point, kSearchDistance);
  results.values[6] = sample.has_solid ? 1 : 0;
  results.values[7] = sample.solid_index;
  results.values[8] = sample.side.plate_id;
  results.values[9] = static_cast<double>(sample.side.side);
  results.values[10] = sample.normal_out_of_solid.x;
  results.values[11] = sample.normal_out_of_solid.y;
  results.values[12] = sample.normal_out_of_solid.z;
  results.values[13] = sample.solid_velocity.x;
  results.values[14] = sample.solid_velocity.y;
  results.values[15] = sample.solid_velocity.z;
  const Vec3 solid_velocity = ComputeSolidVelocity(solid, pose, input.point);
  const Vec3 normal = distance.normal_out_of_solid;
  const Vec3 response = ComputeContactResponse(input.velocity, solid_velocity, normal, input.friction);
  results.values[16] = response.x;
  results.values[17] = response.y;
  results.values[18] = response.z;
  const Vec3 half_friction_response =
      ComputeContactResponse(-input.velocity, solid_velocity, normal, 0.5 * input.friction);
  results.values[19] = half_friction_response.x;
  results.values[20] = half_friction_response.y;
  results.values[21] = half_friction_response.z;
  const double largest_coordinate =
      1 + math::Abs(input.point.x) + math::Abs(input.point.y) + math::Abs(input.point.z);  // m
  results.values[22] = ComputeContactTolerance({
      .largest_coordinate = largest_coordinate,
      .smallest_cell_width = 0.01,
      .thinnest_solid = 0.02,
  });
  results.values[23] = sample.distance;
  results.values[24] = pose.origin.x;
  return results;
}

// Computes ComputeSharedFormulaValues(inputs[i]) on the GPU for i below `count`, into the CPU array `results`.
// Fails: kOutOfMemory  the GPU had no memory for the inputs or the results
//        kCallFailed   another GPU call failed
RunOnCallStatus ComputeSharedFormulasOnGpu(const SharedFormulaInputs* inputs, int count, SharedFormulaValues* results);

}  // namespace fluidity::contact
