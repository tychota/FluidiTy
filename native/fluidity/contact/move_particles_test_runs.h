// The runs and comparisons that the tests of MoveParticles on both processors share.
//
// Purpose: give the CPU tests and the GPU tests one way to run a range of particles through MoveParticles on either
//   processor, one comparison of two runs, and one list of the bits a GPU run wrote for its repeat check.
//
// Idea: a ParticleRange holds the inputs of one call in CPU memory. ComputeMovedRange copies them to the processor
// chosen,
//   builds the contact buffers there from a RunMemory of exactly the contact estimate, runs MoveParticles and reads
//   every output back, so the same range runs on the CPU and on the GPU. FindMoveDifferences compares two runs, the
//   first the reference: the result or the failure, then, on success, every output of every particle, and names each
//   difference by its field with both values. The discrete fields (kinds, indexes, faces, sides, counts, errors) are
//   compared exactly. The floating-point fields must have the same bits when every solid's angular velocity is zero
//   (kSameBits: MoveParticle's promise), and otherwise lie within contact design 6's bounds (kWithinBounds):
//   floor + 2^-40 times the reference, the floor tol / 4 for positions, 2^-40 times the largest input speed for
//   velocities and speeds, 2^-40 for hit fractions and normals. CheckSameMove fails on any difference; SetComparison
//   applies the GPU tests' policy over a set of scenes, where a discrete difference on a turning scene is counted, not
//   failed, up to one scene in ten.
//   Example: a range of one particle falling onto a fixed floor gives the same bits on both processors, so
//   CheckSameMove(cpu_run, gpu_run, kSameBits, bounds) passes.
//
// Owns:    ParticleRange, BuildOneParticleRange, BuildFloorRange, BuildPackedObliqueRange, BuildCrowdedRange,
//          ReactionArray, MovedRange, kUnwrittenPosition, ComputeMovedRange,
//          Agreement, ComputeAgreement, ComparisonBounds, kComparisonRelativeTerm, ComputeComparisonBounds,
//          LargestDifference, ComputeLargestDifference, DoublePair, RunPair, MoveDifference, ToText,
//          FindMoveDifferences, kMostDifferencesListed, CheckSameMove, kAllowedDiscreteMismatchShareDivisor,
//          kMostFailingCasesListed, CheckWroteNothing, SetComparison, ToFieldBits
// Uses:    Buffer, BuildBuffer, ReadBuffer (base/buffer.h), RunMemory, BuildRunMemory (base/run_memory.h),
//          ComputeRunMemoryBytes (base/memory_count.h), RunOn (base/run_on.h), Vec3 (base/vec3.h), CPU-only guard
//          (base/cpu_only.h), BuildBufferHolding, ReadAll, ToBits, CheckSameBitsOrBothNaN, CheckWithinTolerance
//          (base/test_support.h), ContactFailure (contact/contact_error.h), MoveParticles and its types
//          (contact/move_particles.h), MoveParticleCase, CaseGenerator, BuildObliqueCases, BuildFloor, MoveParticle,
//          kTestContactTolerance (contact/contact_test_cases.h), ComputeContactTolerance (contact/contact_tolerance.h),
//          kIdentity (base/mat3.h), Solid, PlateSide, ToSolid
//          (contact/solid_shapes.h), ParticleContact, SolidReaction, SolidContactInputs
//          (contact/solid_contact_types.h), GoogleTest
// Used by: contact/move_particles_steps_test.cc, contact/move_particles_gpu_test.cc,
//          contact/move_particles_scene_test.cc
// Read:    docs/method-paper/fluidity-method.typ <sec-contact>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iomanip>
#include <ios>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/mat3.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/test_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_test_cases.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/move_particles.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {

// The inputs of one MoveParticles call, in CPU memory, one entry per particle at the same index, except the solids.
struct ParticleRange {
  std::vector<std::uint64_t> particle_id;
  std::vector<Vec3> start_position;  // m, world.
  std::vector<Vec3> velocity_for_next_step;  // m/s, world, before the response.
  std::vector<Vec3> velocity_for_this_step;  // m/s, world.
  std::vector<double> friction;  // No unit.
  std::vector<PlateSide> plate_side;
  std::vector<Solid> solids;
  double step_duration = 0;  // s.
  double start_overlap_tolerance = 0;  // m.
  double contact_tolerance = 0;  // m.

  // Adds one particle with the particle values of `inputs`; the call's scalars are those of the first particle.
  void AddParticle(const SolidContactInputs& inputs, std::uint64_t id) {
    if (particle_id.empty()) {
      step_duration = inputs.step_duration;
      start_overlap_tolerance = inputs.start_overlap_tolerance;
      contact_tolerance = inputs.contact_tolerance;
    }
    particle_id.push_back(id);
    start_position.push_back(inputs.start_position);
    velocity_for_next_step.push_back(inputs.velocity.for_next_step);
    velocity_for_this_step.push_back(inputs.velocity.for_this_step);
    friction.push_back(inputs.friction);
    plate_side.push_back(inputs.plate_side);
  }

  [[nodiscard]] std::size_t size() const {
    return particle_id.size();
  }
};

// Builds the range of one scene of contact_test_cases.h: its solids and its one particle, with id `id`.
inline ParticleRange BuildOneParticleRange(const MoveParticleCase& test_case, std::uint64_t id) {
  ParticleRange range;
  range.solids = test_case.solids;
  range.AddParticle(test_case.inputs, id);
  return range;
}

// Builds `count` particles above the fixed floor z <= 0 (solid id 1), each falling onto it or moving on, ids 1000 +
// index, step 0.1 s, friction 0.5, contact tolerance 2^-30 m; those at `failing_indexes` start 0.01 m deep in the
// floor, with start_overlap_tolerance 0, and fail with kStartsInsideSolid, magnitude 0.01. Example: BuildFloorRange(10,
// {{2, 7}}) fails at indexes 2 and 7; with ids 50 and 9 there the call reports 9.
inline ParticleRange BuildFloorRange(std::size_t count, std::span<const std::size_t> failing_indexes) {
  ParticleRange range;
  range.solids = {move_particle_cases::BuildFloor(1)};
  for (std::size_t index = 0; index < count; ++index) {
    const double share = static_cast<double>(index % 97) / 97.0;
    Vec3 start = {.x = share, .y = 0.5 - share, .z = 0.02 + (0.1 * share)};  // m
    if (std::ranges::find(failing_indexes, index) != failing_indexes.end()) {
      start.z = -0.01;
    }
    const Vec3 velocity = {.x = 1, .y = 0, .z = -1 + share};  // m/s
    SolidContactInputs inputs;
    inputs.start_position = start;
    inputs.velocity = {
        .for_next_step = velocity,
        .for_this_step = velocity,
    };
    inputs.step_duration = 0.1;
    inputs.friction = 0.5;
    inputs.contact_tolerance = kTestContactTolerance;
    range.AddParticle(inputs, 1000 + index);
  }
  return range;
}

// Builds one range of the particles of BuildObliqueCases(count) whose move against the solid of case 0 succeeds, all
// against that solid: one call with many paths, frictions and both velocities, most of which miss that solid and some
// of which hit it; ids 1 + case index. The other particles would start inside it, whose half of space is large.
// Example: BuildPackedObliqueRange(4000) holds 1000 particles or more, some of which hit (as move_particles_test.cc).
inline ParticleRange BuildPackedObliqueRange(int count) {
  const std::vector<MoveParticleCase> cases = BuildObliqueCases(count);
  ParticleRange range;
  range.solids = cases.front().solids;
  for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
    SolidContactInputs inputs = cases[case_index].inputs;
    inputs.solids = range.solids.data();
    inputs.solid_count = 1;
    SolidContactResult result;
    if (MoveParticle(inputs, &result, nullptr) == ContactError::kNone) {
      range.AddParticle(inputs, 1 + case_index);
    }
  }
  return range;
}

// Builds a crowded range of `particle_count` particles in the unit box among `solid_count` fixed solids: the floor
// z <= 0 (solid id 1) and solid_count - 1 spheres and boxes of half size 0.02 m centred at z = 0.1 m on a grid in x and
// y. Every 1000th particle starts 1 mm to 10 mm above the floor and falls onto it; the others start at z 0.15 m to 1 m
// and move at up to 1 m/s on each axis, so they meet nothing. Step 1 ms, friction 0.3, the contact tolerance of a 1 m
// scene with 0.01 m cells and 0.04 m solids, ids 1 + index. No solid turns, so both processors owe the same bits.
// Example: BuildCrowdedRange(1000000, 8): a floor, four spheres and three boxes; about 1000 particles hit the floor.
// clang-tidy: a count of particles, then a count of solids, as the name says.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline ParticleRange BuildCrowdedRange(std::size_t particle_count, int solid_count) {
  ParticleRange range;
  range.solids.push_back(move_particle_cases::BuildFloor(1));
  constexpr int kGridSide = 6;  // Solids per row of the grid in x and y: up to 36 besides the floor.
  for (int solid_index = 1; solid_index < solid_count; ++solid_index) {
    const int grid_place = solid_index - 1;
    const Vec3 center = {
        .x = 0.1 + (0.15 * (grid_place % kGridSide)),
        .y = 0.1 + (0.15 * ((grid_place / kGridSide) % kGridSide)),
        .z = 0.1,
    };  // m
    const auto solid_id = static_cast<std::uint32_t>(1 + solid_index);
    const bool is_sphere = solid_index % 2 == 1;
    range.solids.push_back(
        is_sphere ? ToSolid(Sphere{.center = center, .radius = 0.02}, SolidMotion(), solid_id)
                  : ToSolid(
                        Box{.center = center, .axes = kIdentity, .half_size = {.x = 0.02, .y = 0.02, .z = 0.02}},
                        SolidMotion(),
                        solid_id
                    )
    );
  }
  CaseGenerator generator;
  const double contact_tolerance = ComputeContactTolerance({
      .largest_coordinate = 1,
      .smallest_cell_width = 0.01,
      .thinnest_solid = 0.04,
  });  // m
  for (std::size_t index = 0; index < particle_count; ++index) {
    const bool is_near_floor = index % 1000 == 0;
    const double x = generator.Next();  // m
    const double y = generator.Next();  // m
    const double height_draw = generator.Next();
    const double z = is_near_floor ? 0.001 + (0.009 * height_draw) : 0.15 + (0.85 * height_draw);  // m
    Vec3 velocity = {.x = generator.NextSigned(), .y = generator.NextSigned(), .z = generator.NextSigned()};  // m/s
    if (is_near_floor) {
      velocity.z = -20 + velocity.z;
    }
    SolidContactInputs inputs;
    inputs.start_position = {.x = x, .y = y, .z = z};
    inputs.velocity = {
        .for_next_step = velocity,
        .for_this_step = velocity,
    };
    inputs.step_duration = 1e-3;
    inputs.friction = 0.3;
    inputs.contact_tolerance = contact_tolerance;
    range.AddParticle(inputs, 1 + index);
  }
  return range;
}

// Whether a run passes a reaction array to MoveParticles.
enum class ReactionArray : std::uint8_t {
  kGiven,  // A reaction array of one entry per particle.
  kLeftOut,  // A null reaction pointer, which a reacting solid makes a failed call check.
};

// m. The position every output array starts with before a run, so a test sees what a run did not write.
inline constexpr Vec3 kUnwrittenPosition = {.x = -7.5, .y = -7.5, .z = -7.5};

// What one run of MoveParticles returned and wrote, read back to CPU memory.
struct MovedRange {
  std::expected<MoveParticlesResult, ContactFailure> result;
  std::vector<Vec3> position;  // m, world. kUnwrittenPosition where nothing was written.
  std::vector<Vec3> velocity_for_next_step;  // m/s, world. kUnwrittenPosition where nothing was written.
  std::vector<ParticleContact> record;  // ParticleContact{} where nothing was written.
  std::vector<SolidReaction> reaction;  // SolidReaction{} where nothing was written; empty when left out.
};

// Runs MoveParticles over `range` on `run_on`: copies the inputs there, builds the contact buffers from a RunMemory of
// exactly the contact estimate (for at least one particle and one solid), fills the outputs with kUnwrittenPosition and
// default records and reactions, makes the call and reads every output back. Fails the calling test when a buffer
// cannot be built or read; the result is then a kRunOnCallFailed failure.
// Example: ComputeMovedRange(RunOn::kGpu, BuildOneParticleRange(FindMoveParticleCase("DocstringExample"), 1),
//   ReactionArray::kGiven) runs the documented example on the GPU.
inline MovedRange ComputeMovedRange(RunOn run_on, const ParticleRange& range, ReactionArray reaction_array) {
  const std::size_t count = range.size();
  MovedRange moved;
  moved.result = std::unexpected(ContactFailure{.error = ContactError::kRunOnCallFailed});
  const std::size_t max_particle_count = std::max<std::size_t>(count, 1);
  const std::size_t max_solid_count = std::max<std::size_t>(range.solids.size(), 1);
  const std::vector<MemoryEstimateEntry> estimate = BuildContactMemoryEstimate(max_particle_count, max_solid_count);
  const std::uint64_t bytes = ComputeRunMemoryBytes(estimate, MemoryCounts(), MemoryWhere::kRunOn);
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(run_on, bytes);
  EXPECT_TRUE(run_memory.has_value());
  if (!run_memory.has_value()) {
    return moved;
  }
  std::expected<ContactBuffers, ContactFailure> buffers =
      BuildContactBuffers(run_on, max_particle_count, max_solid_count, *run_memory);
  EXPECT_TRUE(buffers.has_value());
  if (!buffers.has_value()) {
    return moved;
  }

  const Buffer<std::uint64_t> particle_id = BuildBufferHolding(run_on, range.particle_id);
  const Buffer<Vec3> start_position = BuildBufferHolding(run_on, range.start_position);
  const Buffer<Vec3> velocity_for_next_step_before = BuildBufferHolding(run_on, range.velocity_for_next_step);
  const Buffer<Vec3> velocity_for_this_step = BuildBufferHolding(run_on, range.velocity_for_this_step);
  const Buffer<double> friction = BuildBufferHolding(run_on, range.friction);
  const Buffer<PlateSide> plate_side = BuildBufferHolding(run_on, range.plate_side);
  Buffer<Vec3> position = BuildBufferHolding(run_on, std::vector<Vec3>(count, kUnwrittenPosition));
  Buffer<Vec3> velocity_for_next_step = BuildBufferHolding(run_on, std::vector<Vec3>(count, kUnwrittenPosition));
  Buffer<ParticleContact> record = BuildBufferHolding(run_on, std::vector<ParticleContact>(count));
  Buffer<SolidReaction> reaction = BuildBufferHolding(run_on, std::vector<SolidReaction>(count));

  const MoveParticlesInputs inputs = {
      .particle_count = count,
      .particle_id = particle_id.data(),
      .start_position = start_position.data(),
      .velocity =
          {
              .for_next_step = velocity_for_next_step_before.data(),
              .for_this_step = velocity_for_this_step.data(),
          },
      .friction = friction.data(),
      .plate_side = plate_side.data(),
      .solids = range.solids,
      .step_duration = range.step_duration,
      .start_overlap_tolerance = range.start_overlap_tolerance,
      .contact_tolerance = range.contact_tolerance,
  };
  const MovedParticles outputs = {
      .position = position.data(),
      .velocity_for_next_step = velocity_for_next_step.data(),
      .record = record.data(),
      .reaction = reaction_array == ReactionArray::kGiven ? reaction.data() : nullptr,
  };
  moved.result = MoveParticles(inputs, outputs, *buffers);
  moved.position = ReadAll(position);
  moved.velocity_for_next_step = ReadAll(velocity_for_next_step);
  moved.record = ReadAll(record);
  if (reaction_array == ReactionArray::kGiven) {
    moved.reaction = ReadAll(reaction);
  }
  return moved;
}

// How two runs of one range must agree in their floating-point outputs.
enum class Agreement : std::uint8_t {
  kSameBits,  // Every double has the same bits, or both are NaN.
  kWithinBounds,  // Every double lies within ComparisonBounds of the reference.
};

// Computes the agreement two processors owe each other for these solids: kSameBits when every angular velocity is
// zero, the promise of MoveParticle; kWithinBounds otherwise, since a turning or spinning solid reaches Sin and Cos,
// whose last bits differ between processors.
// Example: a fixed floor gives kSameBits; a floor spinning at {0, 0, 1} rad/s gives kWithinBounds.
inline Agreement ComputeAgreement(std::span<const Solid> solids) {
  const bool has_turn = std::ranges::any_of(solids, [](const Solid& solid) {
    const Vec3 angular_velocity = solid.motion.angular_velocity;
    return angular_velocity.x != 0 || angular_velocity.y != 0 || angular_velocity.z != 0;
  });
  return has_turn ? Agreement::kWithinBounds : Agreement::kSameBits;
}

// The bounds of kWithinBounds (contact design 6): a compared value passes when it is within floor + 2^-40 |reference|
// of the reference.
struct ComparisonBounds {
  double position_floor = 0;  // m: contact_tolerance / 4.
  double speed_floor = 0;  // m/s: 2^-40 times the largest input speed.
  double share_floor = 0x1p-40;  // No unit: for hit fractions and unit normals.
};

// The relative term of every bound of kWithinBounds.
inline constexpr double kComparisonRelativeTerm = 0x1p-40;

// Computes the bounds for one range: contact_tolerance / 4 for positions, and 2^-40 times the largest speed among
// the particles' two velocities and the solids' linear velocities, for velocities and speeds.
// Example: tol 2^-30 m and speeds up to 10 m/s: {2^-32 m, 10 * 2^-40 m/s, 2^-40}.
inline ComparisonBounds ComputeComparisonBounds(const ParticleRange& range) {
  double largest_speed = 0;  // m/s
  for (std::size_t index = 0; index < range.size(); ++index) {
    largest_speed = std::max(largest_speed, math::Length(range.velocity_for_next_step[index]));
    largest_speed = std::max(largest_speed, math::Length(range.velocity_for_this_step[index]));
  }
  for (const Solid& solid : range.solids) {
    largest_speed = std::max(largest_speed, math::Length(solid.motion.linear_velocity));
  }
  return {
      .position_floor = range.contact_tolerance / 4,
      .speed_floor = 0x1p-40 * largest_speed,
      .share_floor = 0x1p-40,
  };
}

// The largest differences between two runs, over every particle, as the GPU tests print them.
struct LargestDifference {
  double position = 0;  // m: end positions and reaction hit positions.
  double speed = 0;  // m/s: velocities, approach speeds and reaction changes.
  double share = 0;  // No unit: hit fractions and normals.
};

// The two values of one double in two runs, the first the reference.
struct DoublePair {
  double reference = 0;
  double compared = 0;
};

// Two runs of one range, the first the reference.
struct RunPair {
  const MovedRange& reference;
  const MovedRange& compared;
};

namespace move_range_comparison {

// What a double of a run measures, which chooses its floor in ComparisonBounds.
enum class ValueKind : std::uint8_t {
  kPosition,  // m.
  kSpeed,  // m/s.
  kShare,  // No unit: a hit fraction or an axis of a unit normal.
};

// One double of a run with what it measures and the name of its field.
struct DoubleOfKind {
  double value = 0;
  ValueKind kind = ValueKind::kPosition;
  std::string_view field;
};

// One discrete field of a run and its name.
struct NamedInteger {
  std::uint64_t value = 0;
  std::string_view field;
};

// The floor of `bounds` for a value of `kind`.
inline double ComputeFloor(ComparisonBounds bounds, ValueKind kind) {
  switch (kind) {
    case ValueKind::kPosition:
      return bounds.position_floor;
    case ValueKind::kSpeed:
      return bounds.speed_floor;
    case ValueKind::kShare:
      break;
  }
  return bounds.share_floor;
}

// The field of `largest` that keeps the values of `kind`.
inline double& ToLargestOfKind(LargestDifference& largest, ValueKind kind) {
  switch (kind) {
    case ValueKind::kPosition:
      return largest.position;
    case ValueKind::kSpeed:
      return largest.speed;
    case ValueKind::kShare:
      break;
  }
  return largest.share;
}

// The most doubles of one particle: 3 + 3 for the outputs, 1 + 3 + 2 for the record, 3 + 3 for the reaction.
inline constexpr std::size_t kMostDoublesPerParticle = 18;

// The names of the doubles of ToDoublesOfKind, in its order.
inline constexpr std::array<std::string_view, kMostDoublesPerParticle> kDoubleFields = {
    "position.x",
    "position.y",
    "position.z",
    "velocity_for_next_step.x",
    "velocity_for_next_step.y",
    "velocity_for_next_step.z",
    "record.hit_fraction",
    "record.normal_out_of_solid.x",
    "record.normal_out_of_solid.y",
    "record.normal_out_of_solid.z",
    "record.approach_speed.for_next_step",
    "record.approach_speed.for_this_step",
    "reaction.hit_position.x",
    "reaction.hit_position.y",
    "reaction.hit_position.z",
    "reaction.velocity_for_next_step_change.x",
    "reaction.velocity_for_next_step_change.y",
    "reaction.velocity_for_next_step_change.z",
};

// The doubles of particle `index` in a fixed order, kDoubleFields': position, velocity_for_next_step, the record's
// hit fraction, normal and approach speeds, then the reaction's hit position and change when the run has reactions.
// Filled in a fixed array and copied once: GCC 13 warns of a false overflow in a vector that grows by push_back here.
inline std::vector<DoubleOfKind> ToDoublesOfKind(const MovedRange& moved, std::size_t index) {
  std::array<DoubleOfKind, kMostDoublesPerParticle> doubles{};
  std::size_t count = 0;
  const auto add_value = [&doubles, &count](double value, ValueKind kind) {
    doubles.at(count) = {.value = value, .kind = kind, .field = kDoubleFields.at(count)};
    ++count;
  };
  const auto add = [&add_value](Vec3 vector, ValueKind kind) {
    add_value(vector.x, kind);
    add_value(vector.y, kind);
    add_value(vector.z, kind);
  };
  add(moved.position[index], ValueKind::kPosition);
  add(moved.velocity_for_next_step[index], ValueKind::kSpeed);
  const ParticleContact& record = moved.record[index];
  add_value(record.hit_fraction, ValueKind::kShare);
  add(record.normal_out_of_solid, ValueKind::kShare);
  add_value(record.approach_speed.for_next_step, ValueKind::kSpeed);
  add_value(record.approach_speed.for_this_step, ValueKind::kSpeed);
  if (!moved.reaction.empty()) {
    add(moved.reaction[index].hit_position, ValueKind::kPosition);
    add(moved.reaction[index].velocity_for_next_step_change, ValueKind::kSpeed);
  }
  return {doubles.begin(), doubles.begin() + static_cast<std::ptrdiff_t>(count)};
}

// The discrete fields of particle `index`: the record's, then the reaction's solid index (kNoSolid without
// reactions).
inline std::vector<NamedInteger> ToIntegers(const MovedRange& moved, std::size_t index) {
  const ParticleContact& record = moved.record[index];
  return {
      {.value = static_cast<std::uint64_t>(record.kind), .field = "record.kind"},
      {.value = record.hit_count, .field = "record.hit_count"},
      {.value = static_cast<std::uint64_t>(record.face), .field = "record.face"},
      {.value = static_cast<std::uint64_t>(record.side_at_hit), .field = "record.side_at_hit"},
      {.value = record.solid_index, .field = "record.solid_index"},
      {
          .value = moved.reaction.empty() ? kNoSolid : moved.reaction[index].solid_index,
          .field = "reaction.solid_index",
      },
  };
}

// A double as text that reads back to the same bits: 17 significant digits, then its hexadecimal form.
inline std::string ToRoundTripText(double value) {
  std::ostringstream text;
  text << std::setprecision(17) << value << " (" << std::hexfloat << value << ")";
  return text.str();
}

}  // namespace move_range_comparison

// One difference between a reference run (the CPU's) and a compared run (the GPU's).
struct MoveDifference {
  bool is_discrete = false;  // A kind, index, face, side, count, error or the success of the call.
  std::string field;  // Such as "particle 0, record.face", "failure.error" or "counts.hit_count".
  std::string reference_value;
  std::string compared_value;
  std::string detail;  // For a double: the difference and the bound, or "same bits".
};

// Converts a difference to one line: "<field>: cpu <value>, gpu <value>[, <detail>]".
inline std::string ToText(const MoveDifference& difference) {
  std::string text = difference.field + ": cpu " + difference.reference_value + ", gpu " + difference.compared_value;
  if (!difference.detail.empty()) {
    text += ", " + difference.detail;
  }
  return text;
}

namespace move_range_comparison {

// Adds a discrete difference when the two integers differ.
inline void ApplyIntegerDifference(
    std::vector<MoveDifference>& differences,
    const std::string& field,
    std::uint64_t reference_value,
    std::uint64_t compared_value
) {
  if (reference_value != compared_value) {
    differences.push_back({
        .is_discrete = true,
        .field = field,
        .reference_value = std::to_string(reference_value),
        .compared_value = std::to_string(compared_value),
        .detail = {},
    });
  }
}

// Adds a continuous difference when the two doubles do not agree as `agreement` says, with the floor `floor`.
inline void ApplyDoubleDifference(
    std::vector<MoveDifference>& differences,
    const std::string& field,
    DoublePair values,
    Agreement agreement,
    double floor
) {
  const double difference = std::abs(values.reference - values.compared);
  std::string detail;
  if (agreement == Agreement::kSameBits) {
    if (CheckSameBitsOrBothNaN(values.reference, values.compared)) {
      return;
    }
    detail = "difference " + ToRoundTripText(difference) + ", bound: same bits";
  } else {
    const Tolerance tolerance = {.absolute_floor = floor, .relative_term = kComparisonRelativeTerm};
    if (CheckWithinTolerance(values.reference, values.compared, tolerance)) {
      return;
    }
    const double bound = floor + (kComparisonRelativeTerm * std::abs(values.reference));
    detail = "difference " + ToRoundTripText(difference) + ", bound " + ToRoundTripText(bound);
  }
  differences.push_back({
      .is_discrete = false,
      .field = field,
      .reference_value = ToRoundTripText(values.reference),
      .compared_value = ToRoundTripText(values.compared),
      .detail = detail,
  });
}

// Adds the differences of two failures: the discrete fields, then the magnitude as a position.
inline void ApplyFailureDifferences(
    std::vector<MoveDifference>& differences,
    const ContactFailure& reference,
    const ContactFailure& compared,
    Agreement agreement,
    ComparisonBounds bounds
) {
  ApplyIntegerDifference(
      differences,
      "failure.error",
      static_cast<std::uint64_t>(reference.error),
      static_cast<std::uint64_t>(compared.error)
  );
  ApplyIntegerDifference(
      differences,
      "failure.close_call",
      static_cast<std::uint64_t>(reference.close_call),
      static_cast<std::uint64_t>(compared.close_call)
  );
  ApplyIntegerDifference(
      differences,
      "failure.has_particle_id",
      reference.has_particle_id ? 1 : 0,
      compared.has_particle_id ? 1 : 0
  );
  ApplyIntegerDifference(differences, "failure.particle_id", reference.particle_id, compared.particle_id);
  ApplyIntegerDifference(differences, "failure.solid_id", reference.solid_id, compared.solid_id);
  ApplyDoubleDifference(
      differences,
      "failure.magnitude",
      {.reference = reference.magnitude, .compared = compared.magnitude},
      agreement,
      bounds.position_floor
  );
}

// Adds the differences of particle `index` of two successful runs of the same size.
inline void ApplyParticleDifferences(
    std::vector<MoveDifference>& differences,
    RunPair runs,
    std::size_t index,
    Agreement agreement,
    ComparisonBounds bounds
) {
  const std::string particle = "particle " + std::to_string(index) + ", ";
  const std::vector<NamedInteger> reference_integers = ToIntegers(runs.reference, index);
  const std::vector<NamedInteger> compared_integers = ToIntegers(runs.compared, index);
  for (std::size_t value_index = 0; value_index < reference_integers.size(); ++value_index) {
    ApplyIntegerDifference(
        differences,
        particle + std::string(reference_integers[value_index].field),
        reference_integers[value_index].value,
        compared_integers[value_index].value
    );
  }
  const std::vector<DoubleOfKind> reference_doubles = ToDoublesOfKind(runs.reference, index);
  const std::vector<DoubleOfKind> compared_doubles = ToDoublesOfKind(runs.compared, index);
  for (std::size_t value_index = 0; value_index < reference_doubles.size(); ++value_index) {
    const DoubleOfKind& reference_double = reference_doubles[value_index];
    ApplyDoubleDifference(
        differences,
        particle + std::string(reference_double.field),
        {.reference = reference_double.value, .compared = compared_doubles[value_index].value},
        agreement,
        ComputeFloor(bounds, reference_double.kind)
    );
  }
}

}  // namespace move_range_comparison

// Computes the largest absolute difference of each kind between two runs over the particles both moved. Runs that do
// not both succeed give zeros.
inline LargestDifference ComputeLargestDifference(const MovedRange& reference, const MovedRange& compared) {
  LargestDifference largest;
  if (!reference.result.has_value() || !compared.result.has_value()) {
    return largest;
  }
  for (std::size_t index = 0; index < reference.position.size(); ++index) {
    const auto reference_doubles = move_range_comparison::ToDoublesOfKind(reference, index);
    const auto compared_doubles = move_range_comparison::ToDoublesOfKind(compared, index);
    for (std::size_t value_index = 0; value_index < reference_doubles.size(); ++value_index) {
      const double difference = std::abs(reference_doubles[value_index].value - compared_doubles[value_index].value);
      double& kind_largest = move_range_comparison::ToLargestOfKind(largest, reference_doubles[value_index].kind);
      // A NaN difference is not a size; FindMoveDifferences reports it.
      if (!std::isnan(difference)) {
        kind_largest = std::max(kind_largest, difference);
      }
    }
  }
  return largest;
}

// Finds every difference between `compared` and `reference`, the first the reference: whether each succeeded; on
// failure the failure's fields; on success the counts, then every particle's discrete fields (exact) and doubles
// (same bits for kSameBits, within `bounds` for kWithinBounds). Discrete differences are marked, so a caller may
// count them apart.
// Example: a run whose particle 3 ends on face kLowZ instead of kHighZ gives one discrete difference
//   "particle 3, record.face".
inline std::vector<MoveDifference> FindMoveDifferences(
    const MovedRange& reference,
    const MovedRange& compared,
    Agreement agreement,
    ComparisonBounds bounds
) {
  // Steps: 1 success or failure  2 counts and sizes  3 every particle
  std::vector<MoveDifference> differences;
  // Step 1.
  const auto describe_result = [](const MovedRange& moved) -> std::string {
    if (moved.result.has_value()) {
      return "success";
    }
    return "failure " + std::to_string(static_cast<int>(moved.result.error().error)) + " of particle "
        + std::to_string(moved.result.error().particle_id);
  };
  if (reference.result.has_value() != compared.result.has_value()) {
    differences.push_back({
        .is_discrete = true,
        .field = "result",
        .reference_value = describe_result(reference),
        .compared_value = describe_result(compared),
        .detail = {},
    });
    return differences;
  }
  if (!reference.result.has_value()) {
    move_range_comparison::ApplyFailureDifferences(
        differences,
        reference.result.error(),
        compared.result.error(),
        agreement,
        bounds
    );
    return differences;
  }
  // Step 2.
  move_range_comparison::ApplyIntegerDifference(
      differences,
      "counts.hit_count",
      reference.result->hit_count,
      compared.result->hit_count
  );
  move_range_comparison::ApplyIntegerDifference(
      differences,
      "counts.started_inside_count",
      reference.result->started_inside_count,
      compared.result->started_inside_count
  );
  const bool has_same_sizes = reference.position.size() == compared.position.size()
      && reference.record.size() == compared.record.size() && reference.reaction.size() == compared.reaction.size();
  if (!has_same_sizes) {
    move_range_comparison::ApplyIntegerDifference(
        differences,
        "particle count",
        reference.position.size(),
        compared.position.size()
    );
    return differences;
  }
  // Step 3.
  for (std::size_t index = 0; index < reference.position.size(); ++index) {
    move_range_comparison::ApplyParticleDifferences(
        differences,
        {.reference = reference, .compared = compared},
        index,
        agreement,
        bounds
    );
  }
  return differences;
}

// The most differences one CheckSameMove message lists; the rest are counted.
inline constexpr std::size_t kMostDifferencesListed = 8;

// Checks that `compared` agrees with `reference` (FindMoveDifferences finds nothing). The failure message lists the
// first kMostDifferencesListed differences, each with its field and both values, and how many were cut.
// Example: two CPU runs of one range pass with kSameBits; a run whose one position moved by 1 m fails both ways, with
//   "particle 17, position.y: cpu ..., gpu ..., difference ..., bound ...".
inline ::testing::AssertionResult
CheckSameMove(const MovedRange& reference, const MovedRange& compared, Agreement agreement, ComparisonBounds bounds) {
  const std::vector<MoveDifference> differences = FindMoveDifferences(reference, compared, agreement, bounds);
  if (differences.empty()) {
    return ::testing::AssertionSuccess();
  }
  ::testing::AssertionResult failure = ::testing::AssertionFailure();
  const std::size_t listed_count = std::min(differences.size(), kMostDifferencesListed);
  for (std::size_t index = 0; index < listed_count; ++index) {
    failure << "\n  " << ToText(differences[index]);
  }
  if (differences.size() > listed_count) {
    failure << "\n  and " << (differences.size() - listed_count) << " more differences, not listed";
  }
  return failure;
}

// Checks that a run wrote no output: every position and velocity still kUnwrittenPosition and every record and
// reaction still the default, compared field by field, bit for bit (FindMoveDifferences). The result is not looked at.
// Example: a run that failed its call check passes; a run that moved one particle fails, naming that particle.
inline ::testing::AssertionResult CheckWroteNothing(const MovedRange& moved) {
  const std::size_t count = moved.position.size();
  MovedRange start;
  start.position.assign(count, kUnwrittenPosition);
  start.velocity_for_next_step.assign(count, kUnwrittenPosition);
  start.record.assign(count, ParticleContact{});
  start.reaction.assign(moved.reaction.size(), SolidReaction{});
  MovedRange outputs = moved;
  outputs.result = MoveParticlesResult{};
  return CheckSameMove(start, outputs, Agreement::kSameBits, ComparisonBounds{});
}

// The share of a set's scenes with a turning or spinning solid that may show a discrete mismatch without failing the
// set: one in ten, rounded down. Sin and Cos differ in their last bits between processors, and a scene built on a
// threshold (a hit at the edge of a window, a settle that depends on the order of near surfaces) can then end with
// another face or kind. The review of the catalogue expects 0 to 2 of its 25 such scenes to be that fragile; a
// mistake in the GPU's turning code would change most of them, far above one in ten.
inline constexpr int kAllowedDiscreteMismatchShareDivisor = 10;

// The most failing cases one SetComparison lists in its failure message; the rest are counted.
inline constexpr int kMostFailingCasesListed = 20;

// The comparisons of one set of ranges, the reference run (the CPU's) against the compared run (the GPU's), and the
// policy of the GPU tests:
//   a range whose solids do not turn (kSameBits) fails on any difference;
//   a range with a turning or spinning solid (kWithinBounds) fails on a continuous difference beyond the bounds, but a
//   discrete difference (kind, index, face, side, count, error, success) makes it a discrete mismatch instead, which
//   is reported and counted, not failed, as long as the set has at most one in ten such ranges with a mismatch
//   (kAllowedDiscreteMismatchShareDivisor). A mismatch's other differences follow from it and are listed with it.
// Example: 20 turning scenes, one ending on face kLowZ on the GPU and kHighZ on the CPU: one mismatch, two allowed,
//   the set passes and the mismatch is one note line.
class SetComparison {
 public:
  explicit SetComparison(std::string set_name) : set_name_(std::move(set_name)) {
  }

  // Compares one range, case `case_index` of the set (`scene`: its name, or empty).
  void
  AddCase(std::size_t case_index, std::string_view scene, RunPair runs, Agreement agreement, ComparisonBounds bounds) {
    const LargestDifference difference = ComputeLargestDifference(runs.reference, runs.compared);
    LargestDifference& largest = agreement == Agreement::kSameBits ? largest_without_turn_ : largest_with_turn_;
    largest.position = std::max(largest.position, difference.position);
    largest.speed = std::max(largest.speed, difference.speed);
    largest.share = std::max(largest.share, difference.share);
    if (agreement == Agreement::kWithinBounds) {
      ++turning_case_count_;
    }
    const std::vector<MoveDifference> differences =
        FindMoveDifferences(runs.reference, runs.compared, agreement, bounds);
    if (differences.empty()) {
      return;
    }
    const std::string label =
        set_name_ + ", case " + std::to_string(case_index) + (scene.empty() ? std::string() : " " + std::string(scene));
    const bool has_discrete = std::ranges::any_of(differences, [](const MoveDifference& found) {
      return found.is_discrete;
    });
    if (agreement == Agreement::kWithinBounds && has_discrete) {
      ++discrete_mismatch_count_;
      for (const MoveDifference& found : differences) {
        mismatch_lines_.push_back("discrete mismatch, " + label + ": " + ToText(found));
      }
      return;
    }
    ++failing_case_count_;
    if (failing_case_count_ > kMostFailingCasesListed) {
      return;
    }
    failure_lines_.push_back(label + ":");
    const std::size_t listed_count = std::min(differences.size(), kMostDifferencesListed);
    for (std::size_t index = 0; index < listed_count; ++index) {
      failure_lines_.push_back("  " + ToText(differences[index]));
    }
    if (differences.size() > listed_count) {
      failure_lines_.push_back("  and " + std::to_string(differences.size() - listed_count) + " more differences");
    }
  }

  // The discrete mismatches a set may have and still pass.
  [[nodiscard]] int allowed_discrete_mismatch_count() const {
    return turning_case_count_ / kAllowedDiscreteMismatchShareDivisor;
  }

  // The line that sums up the set's discrete mismatches, for the notes file.
  [[nodiscard]] std::string ToSummaryLine() const {
    return set_name_ + ": " + std::to_string(discrete_mismatch_count_) + " discrete mismatches in "
        + std::to_string(turning_case_count_) + " scenes with a turning or spinning solid (allowed "
        + std::to_string(allowed_discrete_mismatch_count()) + "), " + std::to_string(failing_case_count_)
        + " failing scenes";
  }

  // Checks that the set passes: no failing case, and at most the allowed discrete mismatches. The message lists the
  // first kMostFailingCasesListed failing cases, says how many were cut, and lists the mismatches.
  [[nodiscard]] ::testing::AssertionResult CheckPasses() const {
    const bool passes = failing_case_count_ == 0 && discrete_mismatch_count_ <= allowed_discrete_mismatch_count();
    if (passes) {
      return ::testing::AssertionSuccess();
    }
    ::testing::AssertionResult failure = ::testing::AssertionFailure();
    failure << ToSummaryLine();
    for (const std::string& line : failure_lines_) {
      failure << "\n" << line;
    }
    if (failing_case_count_ > kMostFailingCasesListed) {
      failure << "\nand " << (failing_case_count_ - kMostFailingCasesListed) << " more failing scenes, not listed";
    }
    for (const std::string& line : mismatch_lines_) {
      failure << "\n" << line;
    }
    return failure;
  }

  [[nodiscard]] const std::vector<std::string>& mismatch_lines() const {
    return mismatch_lines_;
  }
  [[nodiscard]] int failing_case_count() const {
    return failing_case_count_;
  }
  [[nodiscard]] int discrete_mismatch_count() const {
    return discrete_mismatch_count_;
  }
  [[nodiscard]] int turning_case_count() const {
    return turning_case_count_;
  }
  [[nodiscard]] LargestDifference largest_without_turn() const {
    return largest_without_turn_;
  }
  [[nodiscard]] LargestDifference largest_with_turn() const {
    return largest_with_turn_;
  }

 private:
  std::string set_name_;
  int failing_case_count_ = 0;
  int discrete_mismatch_count_ = 0;
  int turning_case_count_ = 0;
  std::vector<std::string> failure_lines_;
  std::vector<std::string> mismatch_lines_;
  LargestDifference largest_without_turn_;
  LargestDifference largest_with_turn_;
};

// Converts a run to the 64-bit words a digest reads: the result or the failure field by field, then every particle's
// integers and the bits of its doubles. Field by field, so padding bytes, which no run promises to write alike, never
// reach the digest.
// Example: a failed run gives {1, error, close call, has_particle_id, particle_id, solid_id, magnitude bits}.
inline std::vector<std::uint64_t> ToFieldBits(const MovedRange& moved) {
  std::vector<std::uint64_t> words;
  if (moved.result.has_value()) {
    words = {0, moved.result->hit_count, moved.result->started_inside_count};
    for (std::size_t index = 0; index < moved.position.size(); ++index) {
      for (const move_range_comparison::NamedInteger integer : move_range_comparison::ToIntegers(moved, index)) {
        words.push_back(integer.value);
      }
      for (const move_range_comparison::DoubleOfKind value : move_range_comparison::ToDoublesOfKind(moved, index)) {
        words.push_back(ToBits(value.value));
      }
    }
    return words;
  }
  const ContactFailure& failure = moved.result.error();
  return {
      1,
      static_cast<std::uint64_t>(failure.error),
      static_cast<std::uint64_t>(failure.close_call),
      failure.has_particle_id ? 1U : 0U,
      failure.particle_id,
      failure.solid_id,
      ToBits(failure.magnitude),
  };
}

}  // namespace fluidity::contact
