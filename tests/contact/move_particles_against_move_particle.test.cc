#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <string>
#include <vector>

#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/tree_sum.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/move_particles.h"
#include "fluidity/contact/move_particles_steps.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/move_particles_cases.h"

namespace fluidity::contact {
namespace {

// Every position, velocity_for_next_step and record of `arrays`, as the bits of their fields, in index order.
std::vector<std::uint64_t> ToOutputBits(const ParticleArrays& arrays) {
  std::vector<std::uint64_t> bits;
  for (std::size_t index = 0; index < arrays.position.size(); ++index) {
    for (const Vec3 vector : {arrays.position[index], arrays.velocity_for_next_step[index]}) {
      const std::array<std::uint64_t, 3> vector_bits = ToBits(vector);
      bits.insert(bits.end(), vector_bits.begin(), vector_bits.end());
    }
    const std::vector<std::uint64_t> record_bits = ToFieldBits(arrays.record[index]);
    bits.insert(bits.end(), record_bits.begin(), record_bits.end());
  }
  return bits;
}

// Checks the failure of a call against MoveParticle's: the same error, close call, whether a shorter step can cure it,
// solid and magnitude.
void CheckSameFailure(const ContactFailure& actual, const ContactFailure& expected) {
  EXPECT_EQ(actual.error, expected.error);
  EXPECT_EQ(actual.close_call, expected.close_call);
  EXPECT_EQ(actual.can_shorter_step_cure, expected.can_shorter_step_cure);
  EXPECT_EQ(actual.solid_id, expected.solid_id);
  EXPECT_EQ(fluidity::ToBits(actual.magnitude), fluidity::ToBits(expected.magnitude));
}

// The first solid of a scene whose origin over the step plus its reach, as ComputeContactScale counts them, lies
// beyond kLargestCoordinateInTolerances contact tolerances, or nullptr: a scene MoveParticles names as bad input
// before it reads a particle, and MoveParticle runs outside its premise on coordinates.
const Solid* FindSolidBeyondLargestCoordinate(const MoveParticleCase& test_case) {
  const double largest_coordinate = kLargestCoordinateInTolerances * test_case.inputs.contact_tolerance;  // m
  for (const Solid& solid : test_case.solids) {
    const double step_duration = test_case.inputs.step_duration;
    const double tolerance = test_case.inputs.contact_tolerance;
    if (ComputeContactScale(&solid, 1, step_duration, tolerance).largest_coordinate > largest_coordinate) {
      return &solid;
    }
  }
  return nullptr;
}

// Checks that a one-particle call names `solid` as bad input, beyond the largest coordinate, before any particle.
void CheckCallNamesSolidBeyond(ParticleArrays& arrays, const Solid& solid) {
  CpuContactBuffers buffers(1, arrays.solids.size());
  const ContactFailure failure =
      MoveParticles(arrays.ToInputs(), arrays.ToMoved(), buffers.buffers()).error_or(ContactFailure{});
  EXPECT_EQ(failure.error, ContactError::kBadInput);
  EXPECT_FALSE(failure.has_particle_id);
  EXPECT_EQ(failure.solid_id, solid.solid_id);
}

// Runs one catalogue scene as a one-particle call and checks it against MoveParticle on the same inputs, or, for a
// scene with a solid beyond the largest coordinate, that the call names that solid as bad input; says whether it
// failed.
bool CheckCaseSameAsMoveParticle(const MoveParticleCase& test_case) {
  ParticleArrays arrays;
  arrays.solids = test_case.solids;
  arrays.AddParticle(test_case.inputs, 77);
  const bool has_valid_scalars = test_case.inputs.contact_tolerance > 0 && test_case.inputs.step_duration > 0;
  const Solid* const solid_beyond = has_valid_scalars ? FindSolidBeyondLargestCoordinate(test_case) : nullptr;
  if (solid_beyond != nullptr) {
    CheckCallNamesSolidBeyond(arrays, *solid_beyond);
    return true;
  }
  SolidContactResult expected;
  ContactFailure expected_failure;
  const ContactError expected_error = MoveParticle(ToSolidContactInputs(test_case), &expected, &expected_failure);
  CpuContactBuffers buffers(1, arrays.solids.size());
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  if (!result.has_value()) {
    EXPECT_NE(expected_error, ContactError::kNone);
    CheckSameFailure(result.error(), expected_failure);
    return true;
  }
  EXPECT_EQ(expected_error, ContactError::kNone);
  const MoveParticleCounts expected_counts = ComputeParticleCounts(expected.kind);
  EXPECT_EQ(result->hit_count, expected_counts.hit_count);
  EXPECT_EQ(result->started_inside_count, expected_counts.started_inside_count);
  CheckSameOutputs(arrays, 0, expected);
  return false;
}

// The particles of the oblique cases whose move against the solid of case 0 succeeds, up to `count`, against that
// solid: many particles of one call with varied paths, frictions and both velocities.
ParticleArrays BuildObliqueParticles(std::size_t count) {
  const std::vector<MoveParticleCase> cases = BuildObliqueCases(4000);
  ParticleArrays arrays;
  arrays.solids = cases[0].solids;
  for (std::size_t case_index = 0; case_index < cases.size() && arrays.particle_id.size() < count; ++case_index) {
    SolidContactInputs inputs = cases[case_index].inputs;
    inputs.solids = arrays.solids.data();
    inputs.solid_count = 1;
    SolidContactResult result;
    if (MoveParticle(inputs, &result, nullptr) == ContactError::kNone) {
      arrays.AddParticle(inputs, 1 + case_index);
    }
  }
  return arrays;
}

// Moves the particles [first, end) of `arrays` in one call, with every pointer moved to `first`.
std::expected<MoveParticlesResult, ContactFailure>
ApplyMoveToRange(ParticleArrays& arrays, std::size_t first, std::size_t end, ContactBuffers& buffers) {
  MoveParticlesInputs inputs = arrays.ToInputs();
  inputs.particle_count = end - first;
  inputs.particle_id += first;
  inputs.start_position += first;
  inputs.velocity.for_next_step += first;
  inputs.velocity.for_this_step += first;
  inputs.friction += first;
  inputs.plate_side += first;
  MovedParticles moved = arrays.ToMoved();
  moved.position += first;
  moved.velocity_for_next_step += first;
  moved.record += first;
  moved.reaction += first;
  return MoveParticles(inputs, moved, buffers);
}

TEST(MoveParticles, EveryCase_SameAsMoveParticle) {
  // Each scene of the catalogue as a one-particle call, against MoveParticle on the same inputs: the outputs bit for
  // bit on success, the same error, close call, can_shorter_step_cure, solid and magnitude on failure, for every scene
  // but the five with a ball of radius 2^40 tol, whose top lies at 2^41 tol: the call names the ball as bad input,
  // beyond kLargestCoordinateInTolerances, and MoveParticle, which does not check its premise on coordinates, runs
  // them.
  int failing_case_count = 0;
  for (const MoveParticleCase& test_case : BuildMoveParticleCases()) {
    SCOPED_TRACE(std::string(test_case.name));
    failing_case_count += CheckCaseSameAsMoveParticle(test_case) ? 1 : 0;
  }
  EXPECT_GT(failing_case_count, 0);
}

TEST(MoveParticles, ManyParticles_SameAsMoveParticleInPlainLoop) {
  ParticleArrays arrays = BuildObliqueParticles(1000);
  ASSERT_EQ(arrays.particle_id.size(), 1000U);
  CpuContactBuffers buffers(1000, 1);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_TRUE(result.has_value());
  std::uint64_t hit_count = 0;
  for (std::size_t index = 0; index < arrays.particle_id.size(); ++index) {
    SolidContactResult expected;
    ASSERT_EQ(MoveParticle(arrays.ToSolidContactInputs(index), &expected, nullptr), ContactError::kNone);
    hit_count += expected.kind == ContactKind::kHit ? 1 : 0;
    CheckSameOutputs(arrays, index, expected);
  }
  EXPECT_EQ(result->hit_count, hit_count);
  EXPECT_GT(hit_count, 0U);
}

TEST(MoveParticles, TwoRanges_SameAsOneCall) {
  // The first 100 particles in one call, then as the ranges 0..37 and 37..100: the same bytes.
  ParticleArrays whole = BuildObliqueParticles(100);
  ASSERT_EQ(whole.particle_id.size(), 100U);
  ParticleArrays split = whole;
  CpuContactBuffers buffers(100, 1);
  const std::expected<MoveParticlesResult, ContactFailure> whole_result =
      ApplyMoveToRange(whole, 0, 100, buffers.buffers());
  const std::expected<MoveParticlesResult, ContactFailure> first_result =
      ApplyMoveToRange(split, 0, 37, buffers.buffers());
  const std::expected<MoveParticlesResult, ContactFailure> second_result =
      ApplyMoveToRange(split, 37, 100, buffers.buffers());
  const std::uint64_t whole_hit_count = whole_result.value_or(MoveParticlesResult{}).hit_count;
  const std::uint64_t split_hit_count =
      first_result.value_or(MoveParticlesResult{}).hit_count + second_result.value_or(MoveParticlesResult{}).hit_count;
  EXPECT_TRUE(whole_result.has_value() && first_result.has_value() && second_result.has_value());
  EXPECT_EQ(split_hit_count, whole_hit_count);
  EXPECT_EQ(ToOutputBits(split), ToOutputBits(whole));
}

TEST(MoveParticles, SameInputTwice_SameBytesAndSameReactionSums) {
  // The oblique particles against a reacting solid, twice: the same output bytes, and the reaction of the solid summed
  // over the particles in the fixed order of math::TreeSum gives the same bits.
  ParticleArrays arrays = BuildObliqueParticles(1000);
  arrays.solids[0].reports_reaction = true;
  CpuContactBuffers buffers(1000, 1);
  std::vector<std::vector<std::byte>> outputs;
  std::vector<std::array<std::uint64_t, 3>> reaction_sums;
  for (int run = 0; run < 2; ++run) {
    ApplyOutputFill(arrays, static_cast<unsigned char>(run));
    const MovedParticles moved = arrays.ToMoved();
    const std::expected<MoveParticlesResult, ContactFailure> result =
        MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
    ASSERT_TRUE(result.has_value());
    // The reaction has 4 bytes of padding after solid_index, which a run may leave as it found them.
    for (SolidReaction& reaction : arrays.reaction) {
      const SolidReaction copy = {
          .solid_index = reaction.solid_index,
          .hit_position = reaction.hit_position,
          .velocity_for_next_step_change = reaction.velocity_for_next_step_change,
      };
      std::memset(static_cast<void*>(&reaction), 0, sizeof(SolidReaction));
      reaction.solid_index = copy.solid_index;
      reaction.hit_position = copy.hit_position;
      reaction.velocity_for_next_step_change = copy.velocity_for_next_step_change;
    }
    outputs.push_back(CopyOutputBytes(arrays));
    const auto sum_axis = [&arrays](double Vec3::* axis) {
      return math::TreeSumOf(arrays.reaction.size(), [&arrays, axis](std::size_t index) {
        return arrays.reaction[index].velocity_for_next_step_change.*axis;
      });
    };
    reaction_sums.push_back({
        fluidity::ToBits(sum_axis(&Vec3::x)),
        fluidity::ToBits(sum_axis(&Vec3::y)),
        fluidity::ToBits(sum_axis(&Vec3::z)),
    });
  }
  EXPECT_EQ(outputs[0], outputs[1]);
  EXPECT_EQ(reaction_sums[0], reaction_sums[1]);
  EXPECT_NE(reaction_sums[0][0], fluidity::ToBits(0.0));
}

}  // namespace
}  // namespace fluidity::contact
