// Needs a CUDA GPU: label gpu. Built in the CUDA container, run only on the owner's machine with a GPU.
//
// Runs MoveParticles on the GPU and compares every result with the CPU's on the same range
// (contact/testing/move_particles_runs.h): bit for bit when no solid turns or spins; within the GPU bounds of method
// paper <sec-contact> otherwise, where a discrete difference is a counted mismatch, written to the notes file, not a
// failure, up to one scene in ten (SetComparison). Each failure names the set, the case, the field and both values.
// Each set prints its largest differences, so the hardware run can calibrate the bounds. Every GPU result goes to
// WriteResultDigest under a unique name, so the repeat run shows the GPU wrote the same bits twice; the time per
// particle of the large ranges goes to WriteResultTiming.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/move_particles.h"
#include "fluidity/contact/move_particles_gpu.h"
#include "fluidity/contact/move_particles_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/move_particles_runs.h"

namespace fluidity::contact {
namespace {

// Prints the largest differences a set saw, for the hardware run's calibration of the bounds.
void PrintLargest(const std::string& name, LargestDifference largest) {
  std::cout << name << ": largest difference " << largest.position << " m, " << largest.speed << " m/s, "
            << largest.share << " (hit fractions and normals)\n";
}

// Runs `range` on both processors, compares the GPU with the CPU in `comparison` as case `case_index` (`scene`: its
// name, or empty), and appends the GPU run's field bits to `digest_words`.
void CompareOnBoth(
    const ParticleRange& range,
    std::size_t case_index,
    std::string_view scene,
    SetComparison& comparison,
    std::vector<std::uint64_t>& digest_words
) {
  const MovedRange cpu_run = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  const MovedRange gpu_run = ComputeMovedRange(RunOn::kGpu, range, ReactionArray::kGiven);
  comparison.AddCase(
      case_index,
      scene,
      {.reference = cpu_run, .compared = gpu_run},
      ComputeAgreement(range.solids),
      ComputeComparisonBounds(range)
  );
  const std::vector<std::uint64_t> words = ToFieldBits(gpu_run);
  digest_words.insert(digest_words.end(), words.begin(), words.end());
}

// Ends a set: prints and writes to the notes file its summary line and every discrete mismatch, prints its largest
// differences, and checks that it passes (SetComparison::CheckPasses).
void FinishSet(const std::string& name, const SetComparison& comparison) {
  std::cout << comparison.ToSummaryLine() << "\n";
  WriteResultNote("MoveParticles, " + comparison.ToSummaryLine());
  for (const std::string& line : comparison.mismatch_lines()) {
    std::cout << line << "\n";
    WriteResultNote(line);
  }
  PrintLargest(name + ", no turn", comparison.largest_without_turn());
  PrintLargest(name + ", turning or spinning", comparison.largest_with_turn());
  EXPECT_TRUE(comparison.CheckPasses());
}

TEST(MoveParticles, SweepAndMoveCasesOnGpu_MatchCpu) {
  // Every catalogue scene as a one-particle call. Scenes without angular velocity must give the same bits; the scenes
  // with a turning or spinning solid agree within the bounds, and their discrete mismatches are counted.
  SetComparison comparison("catalogue");
  std::vector<std::uint64_t> digest_words;
  std::size_t case_index = 0;
  for (const MoveParticleCase& test_case : BuildMoveParticleCases()) {
    CompareOnBoth(
        BuildOneParticleRange(test_case, case_index + 1),
        case_index,
        test_case.name,
        comparison,
        digest_words
    );
    ++case_index;
  }
  FinishSet("catalogue", comparison);
  WriteResultDigest("MoveParticles on the GPU, catalogue scenes", std::span<const std::uint64_t>(digest_words));
}

TEST(MoveParticles, ObliqueCasesOnGpu_MatchCpu) {
  // BuildObliqueCases(10000), one call per scene, then all of them packed into one call against the first solid
  // (case 10000).
  SetComparison comparison("oblique cases");
  std::vector<std::uint64_t> digest_words;
  std::size_t case_index = 0;
  for (const MoveParticleCase& test_case : BuildObliqueCases(10000)) {
    CompareOnBoth(BuildOneParticleRange(test_case, case_index + 1), case_index, {}, comparison, digest_words);
    ++case_index;
  }
  CompareOnBoth(BuildPackedObliqueRange(10000), case_index, "packed", comparison, digest_words);
  FinishSet("oblique cases", comparison);
  WriteResultDigest("MoveParticles on the GPU, oblique cases", std::span<const std::uint64_t>(digest_words));
}

TEST(MoveParticles, RoundCasesOnGpu_MatchCpu) {
  // BuildRoundCases(10000), one call per scene: each scene has its own sphere or cylinder.
  SetComparison comparison("round cases");
  std::vector<std::uint64_t> digest_words;
  std::size_t case_index = 0;
  for (const MoveParticleCase& test_case : BuildRoundCases(10000)) {
    CompareOnBoth(BuildOneParticleRange(test_case, case_index + 1), case_index, {}, comparison, digest_words);
    ++case_index;
  }
  FinishSet("round cases", comparison);
  WriteResultDigest("MoveParticles on the GPU, round cases", std::span<const std::uint64_t>(digest_words));
}

TEST(MoveParticles, SameInputTwiceOnGpu_SameBytes) {
  // Positions, velocities, records and reactions, field by field, and the counts.
  ParticleRange range = BuildPackedObliqueRange(10000);
  range.solids[0].reports_reaction = true;
  const MovedRange first = ComputeMovedRange(RunOn::kGpu, range, ReactionArray::kGiven);
  const MovedRange second = ComputeMovedRange(RunOn::kGpu, range, ReactionArray::kGiven);
  ASSERT_TRUE(first.result.has_value());
  EXPECT_GT(first.result->hit_count, 0U);
  EXPECT_EQ(ToFieldBits(first), ToFieldBits(second));
  const std::vector<std::uint64_t> words = ToFieldBits(first);
  WriteResultDigest("MoveParticles on the GPU, oblique cases with reactions", std::span<const std::uint64_t>(words));
}

TEST(MoveParticles, TwoFailuresOnGpu_ReportsLowestParticleId) {
  // 1000 particles; ids 50 at index 5 and 9 at index 700 (another thread block) start 0.01 m deep in the floor.
  const std::array<std::size_t, 2> failing_indexes = {5, 700};
  ParticleRange range = BuildFloorRange(1000, failing_indexes);
  range.particle_id[5] = 50;
  range.particle_id[700] = 9;
  const MovedRange gpu_run = ComputeMovedRange(RunOn::kGpu, range, ReactionArray::kGiven);
  ASSERT_FALSE(gpu_run.result.has_value());
  EXPECT_EQ(gpu_run.result.error().error, ContactError::kStartsInsideSolid);
  EXPECT_TRUE(gpu_run.result.error().has_particle_id);
  EXPECT_EQ(gpu_run.result.error().particle_id, 9U);
  EXPECT_EQ(gpu_run.result.error().solid_id, 1U);
  EXPECT_EQ(gpu_run.result.error().magnitude, 0.01);
  const MovedRange cpu_run = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  EXPECT_TRUE(CheckSameMove(cpu_run, gpu_run, Agreement::kSameBits, ComputeComparisonBounds(range)));
  const std::vector<std::uint64_t> words = ToFieldBits(gpu_run);
  WriteResultDigest("MoveParticles on the GPU, two failures", std::span<const std::uint64_t>(words));
}

// The answer of the GPU half for `range` in `launch_size`, called directly with GPU copies of the range.
struct GpuHalfAnswer {
  RunOnCallStatus status = RunOnCallStatus::kCallFailed;
  MoveParticleCounts counts;
  ContactFailure first_failure;
};

GpuHalfAnswer RunGpuHalf(const ParticleRange& range, LaunchSize launch_size) {
  const std::size_t count = range.size();
  const Buffer<std::uint64_t> particle_id = BuildBufferHolding(RunOn::kGpu, range.particle_id);
  const Buffer<Vec3> start_position = BuildBufferHolding(RunOn::kGpu, range.start_position);
  const Buffer<Vec3> velocity_for_next_step_before = BuildBufferHolding(RunOn::kGpu, range.velocity_for_next_step);
  const Buffer<Vec3> velocity_for_this_step = BuildBufferHolding(RunOn::kGpu, range.velocity_for_this_step);
  const Buffer<double> friction = BuildBufferHolding(RunOn::kGpu, range.friction);
  const Buffer<PlateSide> plate_side = BuildBufferHolding(RunOn::kGpu, range.plate_side);
  const Buffer<Solid> solids = BuildBufferHolding(RunOn::kGpu, range.solids);
  Buffer<Vec3> position = BuildBufferHolding(RunOn::kGpu, std::vector<Vec3>(count));
  Buffer<Vec3> velocity_for_next_step = BuildBufferHolding(RunOn::kGpu, std::vector<Vec3>(count));
  Buffer<ParticleContact> record = BuildBufferHolding(RunOn::kGpu, std::vector<ParticleContact>(count));
  Buffer<SolidReaction> reaction = BuildBufferHolding(RunOn::kGpu, std::vector<SolidReaction>(count));
  Buffer<ContactFailure> failure_per_particle = BuildBufferHolding(RunOn::kGpu, std::vector<ContactFailure>(count));
  Buffer<MoveParticleCounts> counts = BuildBufferHolding(RunOn::kGpu, std::vector<MoveParticleCounts>(1));
  Buffer<ContactFailure> first_failure = BuildBufferHolding(RunOn::kGpu, std::vector<ContactFailure>(1));
  const MoveParticlesCall call = {
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
      .solids = solids.data(),
      .position = position.data(),
      .velocity_for_next_step = velocity_for_next_step.data(),
      .record = record.data(),
      .reaction = reaction.data(),
      .failure_per_particle = failure_per_particle.data(),
      .step_duration = range.step_duration,
      .start_overlap_tolerance = range.start_overlap_tolerance,
      .contact_tolerance = range.contact_tolerance,
      .solid_count = static_cast<std::uint32_t>(range.solids.size()),
  };
  GpuHalfAnswer answer;
  answer.status = MoveParticlesOnGpuInLaunchSize(call, launch_size, counts.data(), first_failure.data());
  answer.counts = ReadAll(counts).front();
  answer.first_failure = ReadAll(first_failure).front();
  return answer;
}

// Launch sizes of at most kThreadBlockSize threads: the GPU half's own, one thread, and some where each thread takes
// several particles.
std::vector<LaunchSize> BuildLaunchSizes(std::size_t particle_count) {
  return {
      ComputeLaunchSize(particle_count),
      {.thread_block_count = 1, .thread_block_size = 1},
      {.thread_block_count = 1, .thread_block_size = 32},
      {.thread_block_count = 3, .thread_block_size = 64},
      {.thread_block_count = 7, .thread_block_size = 100},
  };
}

// Checks that a GPU half answer ran and reports `expected`, field by field, the magnitude by its bits.
::testing::AssertionResult CheckReports(const GpuHalfAnswer& answer, const ContactFailure& expected) {
  if (answer.status != RunOnCallStatus::kDone) {
    return ::testing::AssertionFailure() << "the GPU half did not finish";
  }
  const ContactFailure& actual = answer.first_failure;
  const bool is_same = actual.error == expected.error && actual.close_call == expected.close_call
      && actual.has_particle_id == expected.has_particle_id
      && actual.can_shorter_step_cure == expected.can_shorter_step_cure && actual.particle_id == expected.particle_id
      && actual.solid_id == expected.solid_id && ToBits(actual.magnitude) == ToBits(expected.magnitude);
  if (!is_same) {
    return ::testing::AssertionFailure() << "reports particle " << actual.particle_id << ", magnitude "
                                         << actual.magnitude << ", error " << static_cast<int>(actual.error);
  }
  return ::testing::AssertionSuccess();
}

// Checks every launch size of BuildLaunchSizes against `expected`, and appends what each reported to `digest_words`.
void CheckEveryLaunchSize(
    const ParticleRange& range,
    const ContactFailure& expected,
    std::vector<std::uint64_t>& digest_words
) {
  for (const LaunchSize launch_size : BuildLaunchSizes(range.size())) {
    const GpuHalfAnswer answer = RunGpuHalf(range, launch_size);
    EXPECT_TRUE(CheckReports(answer, expected))
        << launch_size.thread_block_count << " thread blocks of " << launch_size.thread_block_size << " threads";
    digest_words.push_back(answer.first_failure.particle_id);
    digest_words.push_back(ToBits(answer.first_failure.magnitude));
  }
}

TEST(MoveParticlesOnGpuInLaunchSize, FailingParticlesAnywhere_ReportLowestIdInEveryLaunchSize) {
  // Failing particles at the first and last index and at thread block edges; the lowest id at each in turn. The CPU
  // loop's failure is the expected one.
  const std::array<std::size_t, 5> failing_indexes = {0, 255, 256, 511, 999};
  std::vector<std::uint64_t> digest_words;
  for (const std::size_t lowest_index : failing_indexes) {
    SCOPED_TRACE(lowest_index);
    ParticleRange range = BuildFloorRange(1000, failing_indexes);
    for (const std::size_t failing_index : failing_indexes) {
      range.particle_id[failing_index] = 100 + failing_index;
    }
    range.particle_id[lowest_index] = 3;
    const MovedRange cpu_run = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
    ASSERT_FALSE(cpu_run.result.has_value());
    EXPECT_EQ(cpu_run.result.error().particle_id, 3U);
    CheckEveryLaunchSize(range, cpu_run.result.error(), digest_words);
  }
  WriteResultDigest(
      "MoveParticlesOnGpuInLaunchSize, failures at thread block edges",
      std::span<const std::uint64_t>(digest_words)
  );
}

TEST(MoveParticlesOnGpuInLaunchSize, EqualIdsFailing_ReportsLowerIndexInEveryLaunchSize) {
  // Against the Requires, ids 9 at indexes 3 and 600, 0.01 m and 0.02 m deep: index 3's depth.
  const std::array<std::size_t, 2> failing_indexes = {3, 600};
  ParticleRange range = BuildFloorRange(1000, failing_indexes);
  range.particle_id[3] = 9;
  range.particle_id[600] = 9;
  range.start_position[600].z = -0.02;
  const ContactFailure expected = {
      .error = ContactError::kStartsInsideSolid,
      .has_particle_id = true,
      .particle_id = 9,
      .solid_id = 1,
      .magnitude = 0.01,
  };
  std::vector<std::uint64_t> digest_words;
  CheckEveryLaunchSize(range, expected, digest_words);
  WriteResultDigest("MoveParticlesOnGpuInLaunchSize, equal failing ids", std::span<const std::uint64_t>(digest_words));
}

TEST(MoveParticlesOnGpuInLaunchSize, NoFailure_CountsInEveryLaunchSize) {
  const ParticleRange range = BuildPackedObliqueRange(2000);
  const MovedRange cpu_run = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  ASSERT_TRUE(cpu_run.result.has_value());
  const MoveParticlesResult expected = *cpu_run.result;
  std::vector<std::uint64_t> digest_words;
  for (const LaunchSize launch_size : BuildLaunchSizes(range.size())) {
    const GpuHalfAnswer answer = RunGpuHalf(range, launch_size);
    EXPECT_TRUE(CheckReports(answer, ContactFailure{}));
    const bool has_cpu_counts = answer.counts.hit_count == expected.hit_count
        && answer.counts.started_inside_count == expected.started_inside_count;
    EXPECT_TRUE(has_cpu_counts) << launch_size.thread_block_count << " thread blocks";
    digest_words.push_back(answer.counts.hit_count);
    digest_words.push_back(answer.counts.started_inside_count);
  }
  WriteResultDigest(
      "MoveParticlesOnGpuInLaunchSize, counts without failure",
      std::span<const std::uint64_t>(digest_words)
  );
}

TEST(MoveParticles, FailedCallCheckOnGpu_WritesNothing) {
  // A reacting floor without a reaction array: kBadInput before any particle is read, every output as it was.
  ParticleRange range = BuildFloorRange(1000, {});
  range.solids[0].reports_reaction = true;
  const MovedRange gpu_run = ComputeMovedRange(RunOn::kGpu, range, ReactionArray::kLeftOut);
  ASSERT_FALSE(gpu_run.result.has_value());
  EXPECT_EQ(gpu_run.result.error().error, ContactError::kBadInput);
  EXPECT_FALSE(gpu_run.result.error().has_particle_id);
  EXPECT_TRUE(CheckWroteNothing(gpu_run));
  const std::vector<std::uint64_t> words = ToFieldBits(gpu_run);
  WriteResultDigest("MoveParticles on the GPU, failed call check", std::span<const std::uint64_t>(words));
}

TEST(BuildContactStackEstimate, OnGpu_IsTheVisitStackForEveryResidentThread) {
  // The visit function's stack per thread (11 to 13 KB from CUDA 13.4, below the 512 KB a thread may have) for every
  // thread the GPU can hold at once; BuildContactBuffers then takes it.
  const std::expected<MemoryEstimateEntry, ContactFailure> entry = BuildContactStackEstimate(RunOn::kGpu);
  ASSERT_TRUE(entry.has_value());
  EXPECT_EQ(entry->where, MemoryWhere::kDriverStack);
  EXPECT_GT(entry->bytes_each, 0U);
  EXPECT_LE(entry->bytes_each, 512U * 1024U);
  EXPECT_GT(entry->given_count, 0U);
  std::cout << "visit stack: " << entry->bytes_each << " bytes for each of " << entry->given_count
            << " resident threads, " << ComputeMemoryBytes(std::span(&*entry, 1), MemoryCounts()) << " bytes\n";
  WriteResultNote(
      "MoveParticles visit stack: " + std::to_string(entry->bytes_each) + " bytes x "
      + std::to_string(entry->given_count) + " resident threads"
  );
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, 65536);
  ASSERT_TRUE(run_memory.has_value());
  EXPECT_TRUE(BuildContactBuffers(RunOn::kGpu, 10, 2, *run_memory).has_value());
}

TEST(MoveParticles, EmptyRangeOnGpu_CountsNothing) {
  ParticleRange range = BuildFloorRange(0, {});
  range.step_duration = 0.1;
  range.contact_tolerance = kTestContactTolerance;
  const MovedRange gpu_run = ComputeMovedRange(RunOn::kGpu, range, ReactionArray::kGiven);
  ASSERT_TRUE(gpu_run.result.has_value());
  EXPECT_EQ(gpu_run.result->hit_count, 0U);
  EXPECT_EQ(gpu_run.result->started_inside_count, 0U);
}

// The time of one MoveParticles call on `run_on` over `range`, best of three, in nanoseconds per particle, and the run
// of the last call. The buffers are built once, outside the timed calls; the call waits for the GPU before it returns.
struct TimedRun {
  double nanoseconds_per_particle = 0;
  MovedRange moved;
};

TimedRun TimeMoveParticles(RunOn run_on, const ParticleRange& range) {
  const std::size_t count = range.size();
  const std::vector<MemoryEstimateEntry> estimate = BuildContactMemoryEstimate(count, range.solids.size());
  std::expected<RunMemory, BufferError> run_memory =
      BuildRunMemory(run_on, ComputeRunMemoryBytes(estimate, MemoryCounts(), MemoryWhere::kRunOn));
  EXPECT_TRUE(run_memory.has_value());
  if (!run_memory.has_value()) {
    return {};
  }
  std::expected<ContactBuffers, ContactFailure> buffers =
      BuildContactBuffers(run_on, count, range.solids.size(), *run_memory);
  EXPECT_TRUE(buffers.has_value());
  if (!buffers.has_value()) {
    return {};
  }
  const Buffer<std::uint64_t> particle_id = BuildBufferHolding(run_on, range.particle_id);
  const Buffer<Vec3> start_position = BuildBufferHolding(run_on, range.start_position);
  const Buffer<Vec3> velocity_for_next_step_before = BuildBufferHolding(run_on, range.velocity_for_next_step);
  const Buffer<Vec3> velocity_for_this_step = BuildBufferHolding(run_on, range.velocity_for_this_step);
  const Buffer<double> friction = BuildBufferHolding(run_on, range.friction);
  const Buffer<PlateSide> plate_side = BuildBufferHolding(run_on, range.plate_side);
  Buffer<Vec3> position = BuildBufferHolding(run_on, std::vector<Vec3>(count));
  Buffer<Vec3> velocity_for_next_step = BuildBufferHolding(run_on, std::vector<Vec3>(count));
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
      .reaction = reaction.data(),
  };
  TimedRun timed;
  double best_seconds = 0;
  for (int run = 0; run < 3; ++run) {
    const auto start = std::chrono::steady_clock::now();
    timed.moved.result = MoveParticles(inputs, outputs, *buffers);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    best_seconds = run == 0 ? elapsed.count() : std::min(best_seconds, elapsed.count());
  }
  timed.nanoseconds_per_particle = 1e9 * best_seconds / static_cast<double>(count);
  timed.moved.position = ReadAll(position);
  timed.moved.velocity_for_next_step = ReadAll(velocity_for_next_step);
  timed.moved.record = ReadAll(record);
  timed.moved.reaction = ReadAll(reaction);
  return timed;
}

TEST(MoveParticles, MillionParticlesOnGpu_MatchCpuAndTimed) {
  // A million particles among 1, 8 and 32 fixed solids: the same bits as the CPU, and the time per particle of both.
  for (const int solid_count : {1, 8, 32}) {
    SCOPED_TRACE(solid_count);
    const ParticleRange range = BuildCrowdedRange(1000000, solid_count);
    const TimedRun gpu_run = TimeMoveParticles(RunOn::kGpu, range);
    const TimedRun cpu_run = TimeMoveParticles(RunOn::kCpu, range);
    ASSERT_TRUE(cpu_run.moved.result.has_value());
    EXPECT_GT(cpu_run.moved.result->hit_count, 900U);
    EXPECT_TRUE(CheckSameMove(cpu_run.moved, gpu_run.moved, Agreement::kSameBits, ComputeComparisonBounds(range)));
    const std::string name = "MoveParticles, 1,000,000 particles, " + std::to_string(solid_count) + " solids";
    std::cout << name << ": GPU " << gpu_run.nanoseconds_per_particle << " ns per particle, CPU "
              << cpu_run.nanoseconds_per_particle << " ns per particle\n";
    WriteResultTiming(name + ", GPU", gpu_run.nanoseconds_per_particle);
    WriteResultTiming(name + ", CPU one thread", cpu_run.nanoseconds_per_particle);
    const std::vector<std::uint64_t> words = ToFieldBits(gpu_run.moved);
    WriteResultDigest(name + " on the GPU", std::span<const std::uint64_t>(words));
  }
}

}  // namespace
}  // namespace fluidity::contact
