#include "fluidity/contact/move_particles.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/per_particle_velocity.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/move_particles_cases.h"

namespace fluidity::contact {
namespace {
constexpr SolidMotion kNoMotion = move_particle_cases::kNoMotion;

constexpr Vec3 kUnitX = move_particle_cases::kUnitX;

constexpr Vec3 kUnitZ = move_particle_cases::kUnitZ;

constexpr MemoryCounts kNoCounts = MemoryCounts();  // The contact estimate counts by given counts only.

TEST(MoveParticles, OneParticle_DocumentedExample) {
  // The example of MoveParticles: the flat solid x >= 1 and a particle at x = 0.9 moving at 1 m/s for 0.2 s.
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildUprightAtXOne(1)};
  SolidContactInputs inputs = BuildFloorParticle({.x = 0.9, .y = 0, .z = 0}, kUnitX);
  inputs.step_duration = 0.2;
  inputs.friction = 0;
  arrays.AddParticle(inputs, 5);
  CpuContactBuffers buffers(1, 1);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->hit_count, 1U);
  EXPECT_EQ(result->started_inside_count, 0U);
  EXPECT_EQ(arrays.position[0].x, 1 - kTolerance);
  EXPECT_EQ(ToBits(arrays.velocity_for_next_step[0]), ToBits(Vec3{}));
  // Every field of the record, written out by hand from MoveParticle's documented example: kind kHit, one hit, face
  // kHighZ (a flat solid's one face), no plate side, solid index 0, the hit fraction, the normal and both approach
  // speeds of -1 m/s.
  const std::vector<std::uint64_t> expected_record = {
      static_cast<std::uint64_t>(ContactKind::kHit),
      1,
      static_cast<std::uint64_t>(SolidFace::kHighZ),
      static_cast<std::uint64_t>(SolidSide::kNoPlate),
      0,
      fluidity::ToBits(((1 - 0.9) - kTolerance) / 0.2),  // 0x1.ffffffafffffep-2, as MoveParticle computes it.
      fluidity::ToBits(-1.0),
      fluidity::ToBits(-0.0),  // The flat solid's normal is -{1, 0, 0}: its zero axes are -0.
      fluidity::ToBits(-0.0),
      fluidity::ToBits(-1.0),
      fluidity::ToBits(-1.0),
  };
  EXPECT_EQ(ToFieldBits(arrays.record[0]), expected_record);
  EXPECT_DOUBLE_EQ(arrays.record[0].hit_fraction, 0.4999999953433871);
  EXPECT_EQ(arrays.reaction[0].solid_index, kNoSolid);
}

TEST(MoveParticles, NoParticles_CountsNothingAndWritesNothing) {
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildFloor(1)};
  MoveParticlesInputs inputs = arrays.ToInputs();
  inputs.step_duration = 0.1;
  inputs.contact_tolerance = kTolerance;
  const MovedParticles moved{};  // Every pointer null: with no particle nothing is read or written.
  CpuContactBuffers buffers(4, 1);
  const std::expected<MoveParticlesResult, ContactFailure> result = MoveParticles(inputs, moved, buffers.buffers());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->hit_count, 0U);
  EXPECT_EQ(result->started_inside_count, 0U);
}

TEST(MoveParticles, CountsHitsAndStartsInside) {
  // Three particles over the fixed floor: one moving along above it, one falling onto it, one starting 0.005 m
  // inside it with start_overlap_tolerance 0.01 m.
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildFloor(1)};
  SolidContactInputs free_particle = BuildFloorParticle({.x = 0, .y = 0, .z = 0.5}, kUnitX);
  free_particle.start_overlap_tolerance = 0.01;
  arrays.AddParticle(free_particle, 1);
  arrays.AddParticle(BuildFloorParticle({.x = 0, .y = 0, .z = 0.05}, -kUnitZ), 2);
  arrays.AddParticle(BuildFloorParticle({.x = 0, .y = 0, .z = -0.005}, {}), 3);
  CpuContactBuffers buffers(3, 1);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->hit_count, 1U);
  EXPECT_EQ(result->started_inside_count, 1U);
  EXPECT_EQ(arrays.record[0].kind, ContactKind::kNone);
  EXPECT_EQ(arrays.record[1].kind, ContactKind::kHit);
  EXPECT_EQ(arrays.record[2].kind, ContactKind::kStartedInsideSolid);
}

TEST(MoveParticles, TwoFailuresOnCpu_ReportsLowestParticleId) {
  // Ten particles over the floor; ids 50 at index 2 and 9 at index 7 start 0.01 m deep, start_overlap_tolerance 0.
  const std::array<std::size_t, 2> failing_indexes = {2, 7};
  ParticleArrays arrays = BuildFloorParticles(10, failing_indexes);
  arrays.particle_id[2] = 50;
  arrays.particle_id[7] = 9;
  CpuContactBuffers buffers(10, 1);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, ContactError::kStartsInsideSolid);
  EXPECT_TRUE(result.error().has_particle_id);
  EXPECT_EQ(result.error().particle_id, 9U);
  EXPECT_EQ(result.error().solid_id, 1U);
  EXPECT_EQ(result.error().magnitude, 0.01);
}

TEST(MoveParticles, EqualIdsFailing_ReportsLowerIndex) {
  // Against the Requires, ids 9 at indexes 3 and 6, 0.01 m and 0.02 m deep: the order (id, index) is still total, so
  // the call reports index 3's depth, as the GPU's two minima would.
  const std::array<std::size_t, 2> failing_indexes = {3, 6};
  ParticleArrays arrays = BuildFloorParticles(10, failing_indexes);
  arrays.particle_id[3] = 9;
  arrays.particle_id[6] = 9;
  arrays.start_position[6].z = -0.02;
  CpuContactBuffers buffers(10, 1);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().particle_id, 9U);
  EXPECT_EQ(result.error().magnitude, 0.01);
}

// The bytes of every input array, to show a call reads them and writes nothing there.
std::vector<std::byte> CopyInputBytes(const ParticleArrays& arrays) {
  std::vector<std::byte> bytes;
  const auto append = [&bytes](std::span<const std::byte> more) {
    bytes.insert(bytes.end(), more.begin(), more.end());
  };
  append(std::as_bytes(std::span(arrays.particle_id)));
  append(std::as_bytes(std::span(arrays.start_position)));
  append(std::as_bytes(std::span(arrays.velocity_for_next_step_before)));
  append(std::as_bytes(std::span(arrays.velocity_for_this_step)));
  append(std::as_bytes(std::span(arrays.friction)));
  append(std::as_bytes(std::span(arrays.plate_side)));
  return bytes;
}

TEST(MoveParticles, FailingParticleFirstMiddleOrLast_InputsUnchangedAndNamed) {
  // A failing particle at index 0, 120 or 199 of 200: the call names it, with its id and has_particle_id, and leaves
  // every input as it was. The outputs belong to the try the step discards; their contents are not part of the result.
  for (const std::size_t failing_index : {std::size_t{0}, std::size_t{120}, std::size_t{199}}) {
    SCOPED_TRACE(failing_index);
    const std::array<std::size_t, 1> failing_indexes = {failing_index};
    ParticleArrays arrays = BuildFloorParticles(200, failing_indexes);
    arrays.solids[0].reports_reaction = true;
    CpuContactBuffers buffers(200, 1);
    const std::vector<std::byte> inputs_before = CopyInputBytes(arrays);
    const MovedParticles moved = arrays.ToMoved();
    const ContactFailure failure =
        MoveParticles(arrays.ToInputs(), moved, buffers.buffers()).error_or(ContactFailure{});
    EXPECT_EQ(failure.error, ContactError::kStartsInsideSolid);
    EXPECT_TRUE(failure.has_particle_id);
    EXPECT_EQ(failure.particle_id, 1000 + failing_index);
    EXPECT_EQ(CopyInputBytes(arrays), inputs_before);
  }
}

TEST(MoveParticles, FailedCallCheck_WritesNothingAndKeepsSolidsCopy) {
  // A call with one floor, then a call with two solids and a NaN step: the second fails its call checks before any
  // particle is read, names no particle, writes no output byte, and solids_copy() keeps the first call's floor.
  ParticleArrays arrays = BuildFloorParticles(5, {});
  CpuContactBuffers buffers(5, 2);
  const MovedParticles moved = arrays.ToMoved();
  ASSERT_TRUE(MoveParticles(arrays.ToInputs(), moved, buffers.buffers()).has_value());
  const std::vector<Solid> first_solids = arrays.solids;
  arrays.solids.push_back(move_particle_cases::BuildUprightAtXOne(2));
  ApplyOutputFill(arrays, 0x5A);
  const std::vector<std::byte> outputs_before = CopyOutputBytes(arrays);
  MoveParticlesInputs inputs = arrays.ToInputs();
  inputs.step_duration = std::numeric_limits<double>::quiet_NaN();
  const ContactFailure failure = MoveParticles(inputs, moved, buffers.buffers()).error_or(ContactFailure{});
  EXPECT_EQ(failure.error, ContactError::kBadInput);
  EXPECT_FALSE(failure.has_particle_id);
  EXPECT_EQ(CopyOutputBytes(arrays), outputs_before);
  const std::span<const Solid> copy = buffers.buffers().solids_copy();
  ASSERT_EQ(copy.size(), 1U);
  EXPECT_EQ(ToFieldBits(copy[0]), ToFieldBits(first_solids[0]));
}

TEST(MoveParticles, MoreParticlesThanBuffers_IsBuffersTooSmall) {
  ParticleArrays arrays = BuildFloorParticles(3, {});
  CpuContactBuffers buffers(2, 1);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, ContactError::kBuffersTooSmall);
  EXPECT_FALSE(result.error().can_shorter_step_cure);
}

TEST(MoveParticles, MoreSolidsThanBuffers_IsBuffersTooSmall) {
  ParticleArrays arrays = BuildFloorParticles(3, {});
  arrays.solids.push_back(move_particle_cases::BuildUprightAtXOne(2));
  CpuContactBuffers buffers(3, 1);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, ContactError::kBuffersTooSmall);
}

TEST(MoveParticles, ReactingSolidWithoutReactionArray_IsBadInput) {
  ParticleArrays arrays = BuildFloorParticles(3, {});
  arrays.solids[0].reports_reaction = true;
  CpuContactBuffers buffers(3, 1);
  MovedParticles moved = arrays.ToMoved();
  moved.reaction = nullptr;
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, ContactError::kBadInput);
  // Without a reacting solid the reaction array may be null.
  arrays.solids[0].reports_reaction = false;
  const std::expected<MoveParticlesResult, ContactFailure> without_reaction =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  EXPECT_TRUE(without_reaction.has_value());
}

TEST(MoveParticles, NullRequiredArray_IsBadInput) {
  ParticleArrays arrays = BuildFloorParticles(3, {});
  CpuContactBuffers buffers(3, 1);
  MoveParticlesInputs inputs = arrays.ToInputs();
  inputs.friction = nullptr;
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result = MoveParticles(inputs, moved, buffers.buffers());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, ContactError::kBadInput);
  EXPECT_FALSE(result.error().has_particle_id);
  // A null plate side array is allowed: no particle is committed to a plate.
  inputs = arrays.ToInputs();
  inputs.plate_side = nullptr;
  EXPECT_TRUE(MoveParticles(inputs, moved, buffers.buffers()).has_value());
}

TEST(MoveParticles, OutputOverlappingAnArray_IsBadInput) {
  // The header's aliasing rules: no output may share a byte with an input or with another output; inputs may share.
  ParticleArrays arrays = BuildFloorParticles(4, {});
  CpuContactBuffers buffers(4, 1);
  std::vector<Vec3> shared_positions = arrays.start_position;
  // An output written over its own input, the update in place, fails as any other overlap.
  MoveParticlesInputs in_place_inputs = arrays.ToInputs();
  in_place_inputs.start_position = shared_positions.data();
  MovedParticles in_place = arrays.ToMoved();
  in_place.position = shared_positions.data();
  const std::expected<MoveParticlesResult, ContactFailure> in_place_result =
      MoveParticles(in_place_inputs, in_place, buffers.buffers());
  ASSERT_FALSE(in_place_result.has_value());
  EXPECT_EQ(in_place_result.error().error, ContactError::kBadInput);
  // An output that starts in the middle of an input.
  MovedParticles into_velocity = arrays.ToMoved();
  into_velocity.velocity_for_next_step = arrays.velocity_for_this_step.data() + 2;
  EXPECT_FALSE(MoveParticles(arrays.ToInputs(), into_velocity, buffers.buffers()).has_value());
  // Two outputs sharing their last and first values.
  std::vector<Vec3> both(7);
  MovedParticles shared_outputs = arrays.ToMoved();
  shared_outputs.position = both.data();
  shared_outputs.velocity_for_next_step = both.data() + 3;
  EXPECT_FALSE(MoveParticles(arrays.ToInputs(), shared_outputs, buffers.buffers()).has_value());
  // An output over the solids.
  std::vector<Solid> solid_bytes(4);
  solid_bytes[0] = arrays.solids[0];
  MoveParticlesInputs solids_inputs = arrays.ToInputs();
  solids_inputs.solids = std::span<const Solid>(solid_bytes.data(), 1);
  MovedParticles over_solids = arrays.ToMoved();
  over_solids.record = reinterpret_cast<ParticleContact*>(solid_bytes.data());  // NOLINT: an address, for the check.
  EXPECT_FALSE(MoveParticles(solids_inputs, over_solids, buffers.buffers()).has_value());
  // Both velocities read from one array: inputs may share.
  MoveParticlesInputs one_velocity = arrays.ToInputs();
  one_velocity.velocity.for_next_step = one_velocity.velocity.for_this_step;
  const MovedParticles moved = arrays.ToMoved();
  EXPECT_TRUE(MoveParticles(one_velocity, moved, buffers.buffers()).has_value());
}

TEST(MoveParticles, OutputOverIdsFrictionsOrPlateSides_IsBadInput) {
  // The reaction array placed over the ids, over the frictions, and over a plate side array that is given.
  ParticleArrays arrays = BuildFloorParticles(4, {});
  CpuContactBuffers buffers(4, 1);
  const std::array<void*, 3> input_bytes = {
      static_cast<void*>(arrays.particle_id.data()),
      static_cast<void*>(arrays.friction.data()),
      static_cast<void*>(arrays.plate_side.data()),
  };
  for (void* const bytes : input_bytes) {
    MovedParticles over_input = arrays.ToMoved();
    over_input.reaction = static_cast<SolidReaction*>(bytes);
    EXPECT_EQ(
        MoveParticles(arrays.ToInputs(), over_input, buffers.buffers()).error_or(ContactFailure{}).error,
        ContactError::kBadInput
    );
  }
}

TEST(MoveParticles, FallingTooFastForStep_IsCloseCallNamingParticleAndFloor) {
  // MoveParticle's check at a hit on one surface, through the call: tol 2^-30 m, so 2^40 tol = 1024 m, a step of
  // 0.1 s, a particle 2 tol above the fixed floor moving at 0.6 x 1024 / 0.1 m/s along x and down z: each axis stays
  // within 1024 m over the step, but the sum of absolute axes times the step is 1.2 x 1024 m. The call names the
  // particle, the floor and the close call; a step a hundred times shorter moves it.
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildFloor(1)};
  const double axis_speed = 0.6 * 0x1p40 * kTolerance / 0.1;  // m/s
  arrays.AddParticle(
      BuildFloorParticle({.x = 0, .y = 0, .z = 2 * kTolerance}, {.x = axis_speed, .y = 0, .z = -axis_speed}),
      4
  );
  CpuContactBuffers buffers(1, 1);
  const MovedParticles moved = arrays.ToMoved();
  const ContactFailure failure = MoveParticles(arrays.ToInputs(), moved, buffers.buffers()).error_or(ContactFailure{});
  EXPECT_EQ(failure.error, ContactError::kTooCloseToCall);
  EXPECT_EQ(failure.close_call, ContactCloseCall::kMovesTooFastForStep);
  EXPECT_TRUE(failure.has_particle_id);
  EXPECT_EQ(failure.particle_id, 4U);
  EXPECT_EQ(failure.solid_id, 1U);
  EXPECT_TRUE(failure.can_shorter_step_cure);
  MoveParticlesInputs shorter = arrays.ToInputs();
  shorter.step_duration = 0.001;
  EXPECT_TRUE(MoveParticles(shorter, moved, buffers.buffers()).has_value());
}

TEST(MoveParticles, FastParticleThatHitsNothing_Moves) {
  // The speeds are checked where a response needs them, at a hit: a particle at 4608 m/s along x and along y over the
  // floor, 921.6 m on each axis in a step of 0.2 s, within 2^40 tol = 1024 m, but whose sum of absolute axes times the
  // step, 1843.2 m, is above it, hits nothing and moves; so does one at rest 1000 m from the origin of a floor spinning
  // at 1 rad/s.
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildFloor(1)};
  SolidContactInputs fast = BuildFloorParticle({.x = 0, .y = 0, .z = 1}, {.x = 4608, .y = 4608, .z = 0});
  fast.step_duration = 0.2;
  arrays.AddParticle(fast, 3);
  CpuContactBuffers buffers(1, 1);
  const MovedParticles moved = arrays.ToMoved();
  EXPECT_TRUE(MoveParticles(arrays.ToInputs(), moved, buffers.buffers()).has_value());
  ParticleArrays far_arrays;
  far_arrays.solids = {move_particle_cases::BuildFlatSolid({}, kUnitZ, {.angular_velocity = kUnitZ}, 1)};
  SolidContactInputs far = BuildFloorParticle({.x = 1000, .y = 0, .z = 1}, {});
  far.step_duration = 0.2;
  far_arrays.AddParticle(far, 6);
  const MovedParticles far_moved = far_arrays.ToMoved();
  EXPECT_TRUE(MoveParticles(far_arrays.ToInputs(), far_moved, buffers.buffers()).has_value());
}

TEST(MoveParticles, SpeedAboveLargestInputSum_IsBadInputNamingIt) {
  // A velocity of 2^481 m/s, or a solid moving or turning that fast, is beyond what a response's rounding bound can
  // hold (kLargestInputSpeedSum): bad input, which no shorter step cures.
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildFloor(1)};
  arrays.AddParticle(BuildFloorParticle({.x = 0, .y = 0, .z = 1}, {.x = 0x1p481, .y = 0, .z = 0}), 2);
  CpuContactBuffers buffers(1, 1);
  const MovedParticles moved = arrays.ToMoved();
  const ContactFailure particle_failure =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers()).error_or(ContactFailure{});
  EXPECT_EQ(particle_failure.error, ContactError::kBadInput);
  EXPECT_EQ(particle_failure.particle_id, 2U);
  EXPECT_EQ(particle_failure.magnitude, 0x1p481);
  EXPECT_FALSE(particle_failure.can_shorter_step_cure);
  arrays.velocity_for_next_step_before[0] = {};
  arrays.velocity_for_this_step[0] = {};
  arrays.solids[0].motion.linear_velocity = {.x = 0x1p481, .y = 0, .z = 0};
  const ContactFailure solid_failure =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers()).error_or(ContactFailure{});
  EXPECT_EQ(solid_failure.error, ContactError::kBadInput);
  EXPECT_FALSE(solid_failure.has_particle_id);
  EXPECT_EQ(solid_failure.solid_id, 1U);
}

// Runs one particle with id 5 above the floor for a step of 0.125 s with tol 2^-30 m, among `solids` (the floor
// first), and returns the call's failure, or ContactFailure{} on success.
ContactFailure ComputeCoordinatesCall(std::vector<Solid> solids, Vec3 start, Vec3 velocity) {
  ParticleArrays arrays;
  arrays.solids = std::move(solids);
  SolidContactInputs inputs = BuildFloorParticle(start, velocity);
  inputs.step_duration = 0.125;
  arrays.AddParticle(inputs, 5);
  CpuContactBuffers buffers(1, arrays.solids.size());
  return MoveParticles(arrays.ToInputs(), arrays.ToMoved(), buffers.buffers()).error_or(ContactFailure{});
}

TEST(MoveParticles, ParticleBeyondLargestCoordinate_IsBadInputNamingIt) {
  // tol 2^-30 m, so MoveParticle's premise ends at 2^40 tol = 1024 m on every axis (kLargestCoordinateInTolerances),
  // and a step of 0.125 s. The start plus the travel along each axis: 1000 + 192 * 0.125 = 1024 m and a start at
  // -1024 m pass; 1000 + 193 * 0.125 = 1024.125 m and a start one double beyond -1024 m are bad input, naming the
  // particle and the coordinate.
  const std::vector<Solid> floor = {move_particle_cases::BuildFloor(1)};
  const Vec3 start = {.x = 1000, .y = 0, .z = 1};
  EXPECT_EQ(ComputeCoordinatesCall(floor, start, {.x = 192, .y = 0, .z = 0}).error, ContactError::kNone);
  EXPECT_EQ(ComputeCoordinatesCall(floor, {.x = 0, .y = -1024, .z = 1}, {}).error, ContactError::kNone);
  const ContactFailure fast = ComputeCoordinatesCall(floor, start, {.x = 193, .y = 0, .z = 0});
  EXPECT_EQ(fast.error, ContactError::kBadInput);
  EXPECT_TRUE(fast.has_particle_id);
  EXPECT_EQ(fast.particle_id, 5U);
  EXPECT_EQ(fast.solid_id, 0U);
  EXPECT_EQ(fast.magnitude, 1024.125);
  EXPECT_FALSE(fast.can_shorter_step_cure);
  const double beyond = std::nextafter(-1024.0, -2048.0);  // m
  const ContactFailure far = ComputeCoordinatesCall(floor, {.x = 0, .y = beyond, .z = 1}, {});
  EXPECT_EQ(far.error, ContactError::kBadInput);
  EXPECT_TRUE(far.has_particle_id);
  EXPECT_EQ(far.magnitude, -beyond);
}

TEST(MoveParticles, SolidBeyondLargestCoordinate_IsBadInputNamingIt) {
  // A box of half sizes {3, 4, 12}, reach Length = 13 m, with tol 2^-30 m (bound 1024 m) and a step of 0.125 s: its
  // origin at the end of the step plus its reach, 1000 + 88 * 0.125 + 13 = 1024 m, passes; moving at 96 m/s, 1025 m,
  // is bad input naming the box and the coordinate, before any particle is read.
  const auto build_box = [](double speed) -> Solid {
    return ToSolid(
        Box{.center = {.x = 1000, .y = 0, .z = 20}, .axes = kIdentity, .half_size = {.x = 3, .y = 4, .z = 12}},
        {.linear_velocity = {.x = speed, .y = 0, .z = 0}, .angular_velocity = {}},
        2
    );
  };
  const Vec3 start = {.x = 0, .y = 0, .z = 1};
  const ContactFailure within = ComputeCoordinatesCall({move_particle_cases::BuildFloor(1), build_box(88)}, start, {});
  EXPECT_EQ(within.error, ContactError::kNone);
  const ContactFailure beyond = ComputeCoordinatesCall({move_particle_cases::BuildFloor(1), build_box(96)}, start, {});
  EXPECT_EQ(beyond.error, ContactError::kBadInput);
  EXPECT_FALSE(beyond.has_particle_id);
  EXPECT_EQ(beyond.solid_id, 2U);
  EXPECT_EQ(beyond.magnitude, 1025.0);
}

TEST(MoveParticles, TinySpeed_MovesAsMoveParticle) {
  // A velocity whose sum of absolute axes is below 2^-1000 m/s, which repeated damping can leave: moved as any other,
  // the same bits as MoveParticle, which bounds its rounding at 2^-1000 m/s (ComputeLargestApproachLeftAfterResponse).
  // Resting on the floor and sinking at 2^-1010 m/s, it gets a response at the start.
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildFloor(1)};
  arrays.AddParticle(BuildFloorParticle({.x = 0, .y = 0, .z = kTolerance}, {.x = 0, .y = 0, .z = -0x1p-1010}), 2);
  CpuContactBuffers buffers(1, 1);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->hit_count, 1U);
  SolidContactResult expected;
  ASSERT_EQ(MoveParticle(arrays.ToSolidContactInputs(0), &expected, nullptr), ContactError::kNone);
  CheckSameOutputs(arrays, 0, expected);
}

TEST(MoveParticles, SolidTurningTooFar_IsTooCloseToCallNamingIt) {
  // MoveParticle requires every solid to pass CheckSolid; MoveParticles checks it once per call.
  ParticleArrays arrays = BuildFloorParticles(2, {});
  arrays.solids.push_back(ToSolid(
      Box{.center = {.x = 5, .y = 5, .z = 5}, .axes = kIdentity, .half_size = {.x = 1, .y = 1, .z = 1}},
      {.angular_velocity = {.x = 0, .y = 0, .z = 20}},
      4
  ));
  CpuContactBuffers buffers(2, 2);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, ContactError::kTooCloseToCall);
  EXPECT_EQ(result.error().close_call, ContactCloseCall::kSolidTurnsTooFar);
  EXPECT_EQ(result.error().solid_id, 4U);
  EXPECT_FALSE(result.error().has_particle_id);
  EXPECT_TRUE(result.error().can_shorter_step_cure);
}

TEST(MoveParticles, SolidTurningTooFar_CuredByShorterStep) {
  // The box turns at 20 rad/s: a step turns it by 20 T rad, above kMaxTurnPerStep (a quarter turn) for T above
  // 0.0785 s. Halving the step until the call succeeds takes the halvings that bring 20 T to a quarter turn or less.
  ParticleArrays arrays = BuildFloorParticles(2, {});
  arrays.solids.push_back(ToSolid(
      Box{.center = {.x = 5, .y = 5, .z = 5}, .axes = kIdentity, .half_size = {.x = 1, .y = 1, .z = 1}},
      {.angular_velocity = {.x = 0, .y = 0, .z = 20}},
      4
  ));
  CpuContactBuffers buffers(2, 2);
  const MovedParticles moved = arrays.ToMoved();
  MoveParticlesInputs inputs = arrays.ToInputs();
  const double full_step = inputs.step_duration;  // s
  int halving_count = 0;
  while (halving_count <= 8 && !MoveParticles(inputs, moved, buffers.buffers()).has_value()) {
    ++halving_count;
    inputs.step_duration = std::ldexp(full_step, -halving_count);
  }
  EXPECT_LE(halving_count, 8);
  EXPECT_LE(20 * inputs.step_duration, std::numbers::pi / 2);
  EXPECT_GT(20 * 2 * inputs.step_duration, std::numbers::pi / 2);
}

// Runs one scene of BuildShorterStepCase as a one-particle call with id 7 and returns its failure, or ContactFailure{}
// on success.
ContactFailure ComputeShorterStepCall(std::string_view name) {
  const MoveParticleCase scene = BuildShorterStepCase(name);
  ParticleArrays arrays;
  arrays.solids = scene.solids;
  arrays.AddParticle(scene.inputs, 7);
  CpuContactBuffers buffers(1, arrays.solids.size());
  return MoveParticles(arrays.ToInputs(), arrays.ToMoved(), buffers.buffers()).error_or(ContactFailure{});
}

TEST(MoveParticles, DescendingBoxClosingOnBottomSide_FailureSaysShorterStepCures) {
  // The squeeze at a hit 0.4 ms into the step, through the call: MoveParticle's flag, with the particle's id.
  const ContactFailure failure = ComputeShorterStepCall("DescendingBoxClosingOnBottomSide");
  EXPECT_EQ(failure.error, ContactError::kTooCloseToCall);
  EXPECT_EQ(failure.close_call, ContactCloseCall::kSqueezedBetweenSolids);
  EXPECT_TRUE(failure.can_shorter_step_cure);
  EXPECT_TRUE(failure.has_particle_id);
  EXPECT_EQ(failure.particle_id, 7U);
}

TEST(MoveParticles, DescendingBoxAtBottomSide_FailureSaysNoShorterStepCures) {
  // The squeeze at the start, through the call.
  const ContactFailure failure = ComputeShorterStepCall("DescendingBoxAtBottomSide");
  EXPECT_EQ(failure.error, ContactError::kTooCloseToCall);
  EXPECT_EQ(failure.close_call, ContactCloseCall::kSqueezedBetweenSolids);
  EXPECT_FALSE(failure.can_shorter_step_cure);
  EXPECT_TRUE(failure.has_particle_id);
  EXPECT_EQ(failure.particle_id, 7U);
}

TEST(MoveParticles, PaddleThroughBottomSide_FailureSaysWhetherShorterStepCures) {
  // The turning solid near another solid: resting on the bottom side, within the window, no shorter step cures it;
  // 1 mm above it, the closing bound shrinks with the step.
  EXPECT_FALSE(ComputeShorterStepCall("PaddleThroughBottomSideRestingOnIt").can_shorter_step_cure);
  const ContactFailure above = ComputeShorterStepCall("PaddleThroughBottomSideOneMillimetreAbove");
  EXPECT_EQ(above.close_call, ContactCloseCall::kTurningSolidNearOtherSolid);
  EXPECT_TRUE(above.can_shorter_step_cure);
}

TEST(MoveParticles, DrivenBoxesInCrease_FailureSaysWhetherShorterStepCures) {
  // Two reacting solids pushed by one response: at the start, no shorter step cures it; 1 mm away, a step that ends
  // before the boxes arrive does.
  const ContactFailure in_crease = ComputeShorterStepCall("DrivenBoxesInCrease");
  EXPECT_EQ(in_crease.close_call, ContactCloseCall::kSecondReactingSolid);
  EXPECT_FALSE(in_crease.can_shorter_step_cure);
  EXPECT_TRUE(ComputeShorterStepCall("DrivenBoxesNearCrease").can_shorter_step_cure);
}

TEST(MoveParticles, SeventeenFloors_FailureSaysWhetherShorterStepCures) {
  const ContactFailure at_start = ComputeShorterStepCall("SeventeenFloorsAtStart");
  EXPECT_EQ(at_start.close_call, ContactCloseCall::kTooManyHits);
  EXPECT_FALSE(at_start.can_shorter_step_cure);
  EXPECT_TRUE(ComputeShorterStepCall("SeventeenFloorsBelowFalling").can_shorter_step_cure);
}

TEST(MoveParticles, ScalarOutOfRange_IsBadInput) {
  ParticleArrays arrays = BuildFloorParticles(2, {});
  CpuContactBuffers buffers(2, 1);
  const MovedParticles moved = arrays.ToMoved();
  MoveParticlesInputs zero_step = arrays.ToInputs();
  zero_step.step_duration = 0;
  EXPECT_EQ(MoveParticles(zero_step, moved, buffers.buffers()).error().error, ContactError::kBadInput);
  MoveParticlesInputs zero_tolerance = arrays.ToInputs();
  zero_tolerance.contact_tolerance = 0;
  EXPECT_EQ(MoveParticles(zero_tolerance, moved, buffers.buffers()).error().error, ContactError::kBadInput);
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  MoveParticlesInputs infinite_step = arrays.ToInputs();
  infinite_step.step_duration = kInfinity;
  EXPECT_EQ(MoveParticles(infinite_step, moved, buffers.buffers()).error().error, ContactError::kBadInput);
  MoveParticlesInputs nan_tolerance = arrays.ToInputs();
  nan_tolerance.contact_tolerance = kNaN;
  EXPECT_EQ(MoveParticles(nan_tolerance, moved, buffers.buffers()).error().error, ContactError::kBadInput);
  MoveParticlesInputs negative_overlap = arrays.ToInputs();
  negative_overlap.start_overlap_tolerance = -1e-3;
  EXPECT_EQ(MoveParticles(negative_overlap, moved, buffers.buffers()).error().error, ContactError::kBadInput);
  MoveParticlesInputs nan_overlap = arrays.ToInputs();
  nan_overlap.start_overlap_tolerance = kNaN;
  EXPECT_EQ(MoveParticles(nan_overlap, moved, buffers.buffers()).error().error, ContactError::kBadInput);
}

#if !defined(FLUIDITY_HAS_GPU)

TEST(BuildContactBuffers, GpuInBuildWithoutGpuCode_IsRunOnCallFailed) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 4096);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<ContactBuffers, ContactFailure> buffers = BuildContactBuffers(RunOn::kGpu, 10, 2, *run_memory);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error().error, ContactError::kRunOnCallFailed);
  EXPECT_FALSE(buffers.error().can_shorter_step_cure);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

#endif

TEST(BuildContactMemoryEstimate, ThousandParticlesFourSolids_MatchesBufferSizes) {
  const std::vector<MemoryEstimateEntry> estimate = BuildContactMemoryEstimate(1000, 4);
  std::vector<std::string_view> names;
  names.reserve(estimate.size());
  for (const MemoryEstimateEntry& entry : estimate) {
    names.push_back(entry.name);
  }
  const std::vector<std::string_view> expected_names = {
      "contact: solids copy",
      "contact: failure per particle",
      "contact: call result",
      "contact: first failure",
  };
  EXPECT_EQ(names, expected_names);
  const bool is_whole_run_given_count = std::ranges::all_of(estimate, [](const MemoryEstimateEntry& entry) {
    return entry.counted_per == CountedPer::kGivenCount && entry.kept_for == KeptFor::kWholeRun
        && entry.where == MemoryWhere::kRunOn && entry.copies == 1;
  });
  EXPECT_TRUE(is_whole_run_given_count);
  const std::uint64_t expected_bytes =
      (1000 * sizeof(ContactFailure)) + (4 * sizeof(Solid)) + sizeof(MoveParticlesResult) + sizeof(ContactFailure);
  EXPECT_EQ(ComputeMemoryBytes(estimate, kNoCounts), expected_bytes);
  EXPECT_EQ(expected_bytes, 32848U);
}

TEST(BuildContactBuffers, IntoRunMemory_HandsOutExactlyTheEstimate) {
  const std::uint64_t bytes =
      ComputeRunMemoryBytes(BuildContactMemoryEstimate(1000, 4), kNoCounts, MemoryWhere::kRunOn);
  EXPECT_EQ(bytes, 33536U);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, bytes);
  ASSERT_TRUE(run_memory.has_value());
  {
    const std::expected<ContactBuffers, ContactFailure> buffers =
        BuildContactBuffers(RunOn::kCpu, 1000, 4, *run_memory);
    ASSERT_TRUE(buffers.has_value());
    EXPECT_EQ(buffers->run_on(), RunOn::kCpu);
    EXPECT_EQ(buffers->max_particle_count(), 1000U);
    EXPECT_EQ(buffers->max_solid_count(), 4U);
    EXPECT_TRUE(buffers->solids_copy().empty());
    EXPECT_EQ(run_memory->handed_out_bytes(), run_memory->reserved_bytes());
    EXPECT_EQ(run_memory->live_piece_count(), 4U);
  }
}

TEST(BuildContactBuffers, RunMemoryOnePieceShort_IsOutOfMemory) {
  // The block of the estimate less 256 bytes: the last piece, the first failure of 32 bytes, does not fit.
  const std::uint64_t bytes =
      ComputeRunMemoryBytes(BuildContactMemoryEstimate(1000, 4), kNoCounts, MemoryWhere::kRunOn);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, bytes - 256);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<ContactBuffers, ContactFailure> buffers = BuildContactBuffers(RunOn::kCpu, 1000, 4, *run_memory);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error().error, ContactError::kOutOfMemory);
  EXPECT_EQ(buffers.error().magnitude, static_cast<double>(sizeof(ContactFailure)));
  EXPECT_FALSE(buffers.error().can_shorter_step_cure);
  // The pieces already taken stay in the block.
  EXPECT_EQ(run_memory->handed_out_bytes(), bytes - 256);
}

TEST(ContactBuffers, SolidsCopy_IsTheLastCallsSolids) {
  // After a call with four solids, the copy holds 4 entries equal byte for byte to the input.
  ParticleArrays arrays = BuildFloorParticles(3, {});
  arrays.solids = {
      move_particle_cases::BuildFloor(1),
      move_particle_cases::BuildUprightAtXOne(2),
      ToSolid(Sphere{.center = {.x = 5, .y = 5, .z = 5}, .radius = 0.5}, kNoMotion, 3),
      ToSolid(
          Box{.center = {.x = -5, .y = 0, .z = 5}, .axes = kIdentity, .half_size = {.x = 1, .y = 2, .z = 3}},
          {.linear_velocity = kUnitX},
          4
      ),
  };
  CpuContactBuffers buffers(1000, 4);
  EXPECT_TRUE(buffers.buffers().solids_copy().empty());
  const MovedParticles moved = arrays.ToMoved();
  ASSERT_TRUE(MoveParticles(arrays.ToInputs(), moved, buffers.buffers()).has_value());
  const std::span<const Solid> copy = buffers.buffers().solids_copy();
  ASSERT_EQ(copy.size(), 4U);
  for (std::size_t solid_index = 0; solid_index < copy.size(); ++solid_index) {
    EXPECT_EQ(ToFieldBits(copy[solid_index]), ToFieldBits(arrays.solids[solid_index]));
  }
}

}  // namespace
}  // namespace fluidity::contact
