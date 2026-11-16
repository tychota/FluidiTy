#include "fluidity/contact/move_particles.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <limits>
#include <optional>
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
#include "fluidity/base/test_support.h"
#include "fluidity/base/tree_sum.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_test_cases.h"
#include "fluidity/contact/move_particles_steps.h"
#include "fluidity/contact/solid_contact.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

namespace fluidity::contact {
namespace {

constexpr double kTolerance = kTestContactTolerance;  // m, 2^-30.
constexpr SolidMotion kNoMotion = move_particle_cases::kNoMotion;
constexpr Vec3 kUnitX = move_particle_cases::kUnitX;
constexpr Vec3 kUnitZ = move_particle_cases::kUnitZ;
constexpr MemoryCounts kNoCounts = MemoryCounts();  // The contact estimate counts by given counts only.

// A RunMemory of exactly the contact estimate and the CPU contact buffers built from it. The block is declared first,
// so it is destroyed after its pieces.
class CpuContactBuffers {
 public:
  CpuContactBuffers(std::size_t max_particle_count, std::size_t max_solid_count) {
    const std::vector<MemoryEstimateEntry> estimate = BuildContactMemoryEstimate(max_particle_count, max_solid_count);
    const MemoryCounts no_counts;
    const std::uint64_t bytes = ComputeRunMemoryBytes(estimate, no_counts, MemoryWhere::kRunOn);
    std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, bytes);
    EXPECT_TRUE(run_memory.has_value());
    if (!run_memory.has_value()) {
      return;
    }
    RunMemory& block = run_memory_.emplace(std::move(*run_memory));
    std::expected<ContactBuffers, ContactFailure> buffers =
        BuildContactBuffers(RunOn::kCpu, max_particle_count, max_solid_count, block);
    EXPECT_TRUE(buffers.has_value());
    if (buffers.has_value()) {
      buffers_.emplace(std::move(*buffers));
    }
  }
  CpuContactBuffers(const CpuContactBuffers&) = delete;
  CpuContactBuffers& operator=(const CpuContactBuffers&) = delete;
  CpuContactBuffers(CpuContactBuffers&&) = delete;
  CpuContactBuffers& operator=(CpuContactBuffers&&) = delete;
  ~CpuContactBuffers() = default;

  // The buffers; a test whose buffers could not be built has already failed, and stops here.
  ContactBuffers& buffers() {
    if (!buffers_.has_value()) {
      std::abort();
    }
    return *buffers_;
  }

 private:
  std::optional<RunMemory> run_memory_;
  std::optional<ContactBuffers> buffers_;
};

// The particle arrays of one MoveParticles call on the CPU, with its outputs, filled one particle at a time.
struct ParticleArrays {
  std::vector<std::uint64_t> particle_id;
  std::vector<Vec3> start_position;
  std::vector<Vec3> velocity_for_next_step_before;
  std::vector<Vec3> velocity_for_this_step;
  std::vector<double> friction;
  std::vector<PlateSide> plate_side;
  std::vector<Solid> solids;
  double step_duration = 0;
  double start_overlap_tolerance = 0;
  double contact_tolerance = 0;
  std::vector<Vec3> position;
  std::vector<Vec3> velocity_for_next_step;
  std::vector<ParticleContact> record;
  std::vector<SolidReaction> reaction;

  // Adds one particle with the particle values of `inputs`; the call's scalars are those of the first particle.
  void AddParticle(const SolidContactInputs& inputs, std::uint64_t id) {
    if (particle_id.empty()) {
      step_duration = inputs.step_duration;
      start_overlap_tolerance = inputs.start_overlap_tolerance;
      contact_tolerance = inputs.contact_tolerance;
    }
    particle_id.push_back(id);
    start_position.push_back(inputs.start_position);
    velocity_for_next_step_before.push_back(inputs.velocity.for_next_step);
    velocity_for_this_step.push_back(inputs.velocity.for_this_step);
    friction.push_back(inputs.friction);
    plate_side.push_back(inputs.plate_side);
    position.emplace_back();
    velocity_for_next_step.emplace_back();
    record.emplace_back();
    reaction.emplace_back();
  }

  [[nodiscard]] MoveParticlesInputs ToInputs() const {
    return {
        .particle_count = particle_id.size(),
        .particle_id = particle_id.data(),
        .start_position = start_position.data(),
        .velocity =
            {.for_next_step = velocity_for_next_step_before.data(), .for_this_step = velocity_for_this_step.data()},
        .friction = friction.data(),
        .plate_side = plate_side.data(),
        .solids = solids,
        .step_duration = step_duration,
        .start_overlap_tolerance = start_overlap_tolerance,
        .contact_tolerance = contact_tolerance,
    };
  }

  MovedParticles ToMoved() {
    return {
        .position = position.data(),
        .velocity_for_next_step = velocity_for_next_step.data(),
        .record = record.data(),
        .reaction = reaction.data(),
    };
  }

  // The call the CPU half builds from these arrays, for running the per-particle functions directly.
  MoveParticlesCall ToCall() {
    return {
        .particle_count = particle_id.size(),
        .particle_id = particle_id.data(),
        .start_position = start_position.data(),
        .velocity =
            {.for_next_step = velocity_for_next_step_before.data(), .for_this_step = velocity_for_this_step.data()},
        .friction = friction.data(),
        .plate_side = plate_side.data(),
        .solids = solids.data(),
        .position = position.data(),
        .velocity_for_next_step = velocity_for_next_step.data(),
        .record = record.data(),
        .reaction = reaction.data(),
        .step_duration = step_duration,
        .start_overlap_tolerance = start_overlap_tolerance,
        .contact_tolerance = contact_tolerance,
        .solid_count = static_cast<std::uint32_t>(solids.size()),
    };
  }

  // The inputs of MoveParticle for particle `index`, pointing at `solids`.
  [[nodiscard]] SolidContactInputs ToSolidContactInputs(std::size_t index) const {
    SolidContactInputs inputs;
    inputs.start_position = start_position[index];
    inputs.velocity = {
        .for_next_step = velocity_for_next_step_before[index],
        .for_this_step = velocity_for_this_step[index],
    };
    inputs.solids = solids.empty() ? nullptr : solids.data();
    inputs.solid_count = static_cast<std::uint32_t>(solids.size());
    inputs.step_duration = step_duration;
    inputs.friction = friction[index];
    inputs.start_overlap_tolerance = start_overlap_tolerance;
    inputs.contact_tolerance = contact_tolerance;
    inputs.plate_side = plate_side[index];
    return inputs;
  }
};

// The bytes of every output array, to compare a call's outputs before and after.
std::vector<std::byte> CopyOutputBytes(const ParticleArrays& arrays) {
  std::vector<std::byte> bytes;
  const auto append = [&bytes](std::span<const std::byte> more) {
    bytes.insert(bytes.end(), more.begin(), more.end());
  };
  append(std::as_bytes(std::span(arrays.position)));
  append(std::as_bytes(std::span(arrays.velocity_for_next_step)));
  append(std::as_bytes(std::span(arrays.record)));
  append(std::as_bytes(std::span(arrays.reaction)));
  return bytes;
}

// Fills every output array with one byte value, so a write of any value shows.
void ApplyOutputFill(ParticleArrays& arrays, unsigned char value) {
  std::memset(static_cast<void*>(arrays.position.data()), value, arrays.position.size() * sizeof(Vec3));
  std::memset(static_cast<void*>(arrays.velocity_for_next_step.data()), value, arrays.position.size() * sizeof(Vec3));
  std::memset(static_cast<void*>(arrays.record.data()), value, arrays.record.size() * sizeof(ParticleContact));
  std::memset(static_cast<void*>(arrays.reaction.data()), value, arrays.reaction.size() * sizeof(SolidReaction));
}

// The 64 bits of each axis.
std::array<std::uint64_t, 3> ToBits(Vec3 vector) {
  return {fluidity::ToBits(vector.x), fluidity::ToBits(vector.y), fluidity::ToBits(vector.z)};
}

// The fields of a contact record as integers, the doubles as their bits, so two records compare with one ==.
std::vector<std::uint64_t> ToFieldBits(const ParticleContact& record) {
  return {
      static_cast<std::uint64_t>(record.kind),
      record.hit_count,
      static_cast<std::uint64_t>(record.face),
      static_cast<std::uint64_t>(record.side_at_hit),
      record.solid_index,
      fluidity::ToBits(record.hit_fraction),
      fluidity::ToBits(record.normal_out_of_solid.x),
      fluidity::ToBits(record.normal_out_of_solid.y),
      fluidity::ToBits(record.normal_out_of_solid.z),
      fluidity::ToBits(record.approach_speed.for_next_step),
      fluidity::ToBits(record.approach_speed.for_this_step),
  };
}

// The record fields of a move's result as ToFieldBits(const ParticleContact&) lists them, read from the result itself
// and not through ToParticleContact, so a wrong field in that conversion shows.
std::vector<std::uint64_t> ToRecordFieldBits(const SolidContactResult& result) {
  return {
      static_cast<std::uint64_t>(result.kind),
      result.hit_count,
      static_cast<std::uint64_t>(result.face),
      static_cast<std::uint64_t>(result.side_at_hit),
      result.solid_index,
      fluidity::ToBits(result.hit_fraction),
      fluidity::ToBits(result.normal_out_of_solid.x),
      fluidity::ToBits(result.normal_out_of_solid.y),
      fluidity::ToBits(result.normal_out_of_solid.z),
      fluidity::ToBits(result.approach_speed.for_next_step),
      fluidity::ToBits(result.approach_speed.for_this_step),
  };
}

// The fields of a solid as integers, the doubles as their bits, the axes included.
std::vector<std::uint64_t> ToFieldBits(const Solid& solid) {
  const Mat3& axes = solid.pose_at_step_start.axes;
  const std::array<Vec3, 4> vectors = {
      solid.pose_at_step_start.origin,
      solid.motion.linear_velocity,
      solid.motion.angular_velocity,
      solid.half_size,
  };
  std::vector<std::uint64_t> bits = {
      static_cast<std::uint64_t>(solid.shape),
      solid.solid_id,
      fluidity::ToBits(solid.friction_scale),
      solid.reports_reaction ? 1U : 0U,
  };
  for (const Vec3 vector : vectors) {
    bits.insert(bits.end(), {fluidity::ToBits(vector.x), fluidity::ToBits(vector.y), fluidity::ToBits(vector.z)});
  }
  for (const double axis_value : {axes.xx, axes.xy, axes.xz, axes.yx, axes.yy, axes.yz, axes.zx, axes.zy, axes.zz}) {
    bits.push_back(fluidity::ToBits(axis_value));
  }
  return bits;
}

// Checks that the outputs at `index` are, bit for bit, what MoveParticle gave in `expected`.
void ExpectSameOutputs(const ParticleArrays& arrays, std::size_t index, const SolidContactResult& expected) {
  EXPECT_EQ(ToBits(arrays.position[index]), ToBits(expected.position)) << "index " << index;
  EXPECT_EQ(ToBits(arrays.velocity_for_next_step[index]), ToBits(expected.velocity.for_next_step)) << "index " << index;
  EXPECT_EQ(ToFieldBits(arrays.record[index]), ToRecordFieldBits(expected)) << "index " << index;
  const SolidReaction& reaction = arrays.reaction[index];
  EXPECT_EQ(reaction.solid_index, expected.reaction.solid_index) << "index " << index;
  EXPECT_EQ(ToBits(reaction.hit_position), ToBits(expected.reaction.hit_position)) << "index " << index;
  EXPECT_EQ(ToBits(reaction.velocity_for_next_step_change), ToBits(expected.reaction.velocity_for_next_step_change))
      << "index " << index;
}

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

// One particle above the fixed floor z <= 0 (solid id 1), contact tolerance 2^-30 m, a step of 0.1 s.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): a start and a velocity, as SolidContactInputs orders them.
SolidContactInputs BuildFloorParticle(Vec3 start_position, Vec3 velocity) {
  SolidContactInputs inputs;
  inputs.start_position = start_position;
  inputs.velocity = {.for_next_step = velocity, .for_this_step = velocity};
  inputs.step_duration = 0.1;
  inputs.friction = 0.5;
  inputs.contact_tolerance = kTolerance;
  return inputs;
}

// `count` particles above the fixed floor (id 1), each falling or moving on, ids 1000 + index; those at
// `failing_indexes` start 0.01 m deep in the floor, with start_overlap_tolerance 0, and fail.
ParticleArrays BuildFloorParticles(std::size_t count, std::span<const std::size_t> failing_indexes) {
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildFloor(1)};
  for (std::size_t index = 0; index < count; ++index) {
    const double share = static_cast<double>(index % 97) / 97.0;
    Vec3 start = {.x = share, .y = 0.5 - share, .z = 0.02 + (0.1 * share)};
    for (const std::size_t failing_index : failing_indexes) {
      if (index == failing_index) {
        start.z = -0.01;
      }
    }
    arrays.AddParticle(BuildFloorParticle(start, {.x = 1, .y = 0, .z = -1 + share}), 1000 + index);
  }
  return arrays;
}

// Checks the failure of a call against MoveParticle's: the same error, close call, solid and magnitude.
void ExpectSameFailure(const ContactFailure& actual, const ContactFailure& expected) {
  EXPECT_EQ(actual.error, expected.error);
  EXPECT_EQ(actual.close_call, expected.close_call);
  EXPECT_EQ(actual.solid_id, expected.solid_id);
  EXPECT_EQ(fluidity::ToBits(actual.magnitude), fluidity::ToBits(expected.magnitude));
}

// Runs one catalogue scene as a one-particle call and checks it against MoveParticle on the same inputs; says whether
// it failed.
bool CheckCaseSameAsMoveParticle(const MoveParticleCase& test_case) {
  ParticleArrays arrays;
  arrays.solids = test_case.solids;
  arrays.AddParticle(test_case.inputs, 77);
  SolidContactResult expected;
  ContactFailure expected_failure;
  const ContactError expected_error = MoveParticle(ToSolidContactInputs(test_case), &expected, &expected_failure);
  CpuContactBuffers buffers(1, arrays.solids.size());
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  if (!result.has_value()) {
    EXPECT_NE(expected_error, ContactError::kNone);
    ExpectSameFailure(result.error(), expected_failure);
    return true;
  }
  EXPECT_EQ(expected_error, ContactError::kNone);
  const MoveParticleCounts expected_counts = ComputeParticleCounts(expected.kind);
  EXPECT_EQ(result->hit_count, expected_counts.hit_count);
  EXPECT_EQ(result->started_inside_count, expected_counts.started_inside_count);
  ExpectSameOutputs(arrays, 0, expected);
  return false;
}

TEST(MoveParticles, EveryCase_SameAsMoveParticle) {
  // Each scene of the catalogue as a one-particle call, against MoveParticle on the same inputs: the outputs bit for
  // bit on success, the same error, close call, solid and magnitude on failure, for every scene.
  int failing_case_count = 0;
  for (const MoveParticleCase& test_case : BuildMoveParticleCases()) {
    SCOPED_TRACE(std::string(test_case.name));
    failing_case_count += CheckCaseSameAsMoveParticle(test_case) ? 1 : 0;
  }
  EXPECT_GT(failing_case_count, 0);
}

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
    ExpectSameOutputs(arrays, index, expected);
  }
  EXPECT_EQ(result->hit_count, hit_count);
  EXPECT_GT(hit_count, 0U);
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

// One thread's view of the failures: whether any particle it visited failed, the first of them in the order of
// ComputeIsEarlierFailure, and its index.
struct ThreadFailure {
  bool has_failure = false;
  ContactFailure failure;
  std::size_t index = 0;
};

// Keeps `candidate` in `kept` when it comes first in the order of ComputeIsEarlierFailure.
void ApplyEarlierFailure(const ThreadFailure& candidate, ThreadFailure* kept) {
  const bool is_earlier = candidate.has_failure
      && (!kept->has_failure
          || ComputeIsEarlierFailure(
              candidate.failure.particle_id,
              candidate.index,
              kept->failure.particle_id,
              kept->index
          ));
  if (is_earlier) {
    *kept = candidate;
  }
}

// The failure a loop over `call` reports when `thread_count` threads share the particles grid-stride, as GPU threads
// do: each thread visits index thread, thread + thread_count, ... and keeps its first failure; the threads are then
// merged last to first, the order least like the CPU loop's.
ContactFailure FindFailureOfThreads(const MoveParticlesCall& call, std::size_t thread_count) {
  std::vector<ThreadFailure> thread_failures(thread_count);
  for (std::size_t thread = 0; thread < thread_count; ++thread) {
    for (std::size_t index = thread; index < call.particle_count; index += thread_count) {
      SolidContactResult particle_result;
      ThreadFailure candidate;
      candidate.index = index;
      candidate.has_failure =
          ComputeParticleMoveAt(call, index, &particle_result, &candidate.failure) != ContactError::kNone;
      ApplyEarlierFailure(candidate, &thread_failures[thread]);
    }
  }
  ThreadFailure reported;
  for (std::size_t thread = thread_count; thread-- > 0;) {
    ApplyEarlierFailure(thread_failures[thread], &reported);
  }
  return reported.failure;
}

TEST(MoveParticles, FailingParticleAnywhere_SameReportInEveryThreadCountAndRun) {
  // 1000 particles over the floor, one failing at each of several indexes in turn, with a second failure of a higher
  // id. MoveParticles twice, then the per-particle functions as any number of threads would share them: the same
  // failure every time.
  constexpr std::size_t kCount = 1000;
  for (const std::size_t failing_index :
       {std::size_t{0}, std::size_t{1}, std::size_t{499}, std::size_t{998}, std::size_t{999}}) {
    SCOPED_TRACE(failing_index);
    const std::array<std::size_t, 2> failing_indexes = {failing_index, 500};
    ParticleArrays arrays = BuildFloorParticles(kCount, failing_indexes);
    arrays.particle_id[failing_index] = 7;
    arrays.particle_id[500] = 8;
    CpuContactBuffers buffers(kCount, 1);
    std::vector<ContactFailure> reported;
    for (int run = 0; run < 2; ++run) {
      const MovedParticles moved = arrays.ToMoved();
      reported.push_back(MoveParticles(arrays.ToInputs(), moved, buffers.buffers()).error_or(ContactFailure{}));
    }
    for (const std::size_t thread_count :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{8}, std::size_t{256}, std::size_t{1024}}) {
      reported.push_back(FindFailureOfThreads(arrays.ToCall(), thread_count));
    }
    const bool is_particle_seven_every_time = std::ranges::all_of(reported, [](const ContactFailure& failure) {
      return failure.error == ContactError::kStartsInsideSolid && failure.particle_id == 7;
    });
    EXPECT_TRUE(is_particle_seven_every_time);
  }
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

TEST(MoveParticles, MoreParticlesThanBuffers_IsBuffersTooSmall) {
  ParticleArrays arrays = BuildFloorParticles(3, {});
  CpuContactBuffers buffers(2, 1);
  const MovedParticles moved = arrays.ToMoved();
  const std::expected<MoveParticlesResult, ContactFailure> result =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, ContactError::kBuffersTooSmall);
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
  // 0.1 s, a particle 2 tol above the fixed floor falling at 1.01 x 1024 / 0.1 m/s. The call names the particle, the
  // floor and the close call; a step a hundred times shorter moves it.
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildFloor(1)};
  const double bound_speed = 0x1p40 * kTolerance / 0.1;  // m/s
  arrays.AddParticle(
      BuildFloorParticle({.x = 0, .y = 0, .z = 2 * kTolerance}, {.x = 0, .y = 0, .z = -1.01 * bound_speed}),
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
  MoveParticlesInputs shorter = arrays.ToInputs();
  shorter.step_duration = 0.001;
  EXPECT_TRUE(MoveParticles(shorter, moved, buffers.buffers()).has_value());
}

TEST(MoveParticles, FastParticleThatHitsNothing_Moves) {
  // The speeds are checked where a response needs them, at a hit: a particle at 1e5 m/s along the floor, 2e4 m in a
  // step of 0.2 s, far above 2^40 tol = 1024 m, hits nothing and moves; so does one at rest 1000 m from the origin of
  // a floor spinning at 1 rad/s.
  ParticleArrays arrays;
  arrays.solids = {move_particle_cases::BuildFloor(1)};
  SolidContactInputs fast = BuildFloorParticle({.x = 0, .y = 0, .z = 1}, {.x = 1e5, .y = 0, .z = 0});
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
  arrays.velocity_for_next_step_before[0] = {};
  arrays.velocity_for_this_step[0] = {};
  arrays.solids[0].motion.linear_velocity = {.x = 0x1p481, .y = 0, .z = 0};
  const ContactFailure solid_failure =
      MoveParticles(arrays.ToInputs(), moved, buffers.buffers()).error_or(ContactFailure{});
  EXPECT_EQ(solid_failure.error, ContactError::kBadInput);
  EXPECT_FALSE(solid_failure.has_particle_id);
  EXPECT_EQ(solid_failure.solid_id, 1U);
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
  ExpectSameOutputs(arrays, 0, expected);
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
