// Runs the GPU half of MoveParticles on the CPU, with the per-particle and per-thread functions its GPU functions call
// (contact/move_particles_steps.h), and compares the answer with the CPU loop's. Each GPU function is run as its launch
// would cover it: every thread block and every thread, each over its items by grid stride, once in launch order and
// once in reverse, in several launch sizes. An atomic minimum, or or addition of the GPU becomes a plain one here:
// their final values do not depend on the order, which the reverse runs show. The host steps follow
// contact/move_particles_gpu.cu (MoveParticlesOnGpuInLaunchSize). The tests of the comparison helpers that the GPU
// tests use (contact/testing/move_particles_runs.h) are here too, so they run in the gate now.

#include "fluidity/contact/move_particles_steps.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/move_particles.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/move_particles_runs.h"
#include "fluidity/contact/testing/move_particles_steps_cases.h"

namespace fluidity::contact {
namespace {

TEST(FirstFailureMinima, TwoFailuresInEveryLaunchAndOrder_ReportLowestParticleId) {
  // 1000 particles over the floor; ids 50 at index 5 and 9 at index 700, in another thread block, start 0.01 m deep.
  const std::array<std::size_t, 2> failing_indexes = {5, 700};
  ParticleRange range = BuildFloorRange(1000, failing_indexes);
  range.particle_id[5] = 50;
  range.particle_id[700] = 9;
  EXPECT_TRUE(CheckEveryFailure(RunGpuHalfInEveryLaunch(range), BuildFloorFailure(9, 0.01)));
  EXPECT_EQ(CheckGpuHalfSameAsCpuLoop(range), 8);
}

TEST(FirstFailureMinima, EqualIdsFailing_ReportsLowerIndex) {
  // Against the Requires, ids 9 at indexes 3 and 600, 0.01 m and 0.02 m deep: the second minimum picks index 3.
  const std::array<std::size_t, 2> failing_indexes = {3, 600};
  ParticleRange range = BuildFloorRange(1000, failing_indexes);
  range.particle_id[3] = 9;
  range.particle_id[600] = 9;
  range.start_position[600].z = -0.02;
  EXPECT_TRUE(CheckEveryFailure(RunGpuHalfInEveryLaunch(range), BuildFloorFailure(9, 0.01)));
  EXPECT_EQ(CheckGpuHalfSameAsCpuLoop(range), 8);
}

TEST(FirstFailureMinima, LargestIdFailingAlone_IsReported) {
  // The flag is apart from the minima, so a failing id equal to the minimum's start, 2^64 - 1 (against the
  // Requires), is still reported.
  const std::array<std::size_t, 1> failing_indexes = {42};
  ParticleRange range = BuildFloorRange(100, failing_indexes);
  range.particle_id[42] = 0xFFFFFFFFFFFFFFFF;
  CallArrays arrays(range);
  const ContactFailure failure =
      RunGpuHalfOnCpu(arrays.ToCall(), ComputeLaunchSize(range.size()), ThreadOrder::kReverse).first_failure;
  EXPECT_EQ(failure.error, ContactError::kStartsInsideSolid);
  EXPECT_EQ(failure.particle_id, 0xFFFFFFFFFFFFFFFF);
}

TEST(FirstFailureMinima, NoFailure_GivesNoFailure) {
  CallArrays arrays(BuildFloorRange(300, {}));
  const ContactFailure failure =
      RunGpuHalfOnCpu(arrays.ToCall(), ComputeLaunchSize(300), ThreadOrder::kLaunchOrder).first_failure;
  EXPECT_EQ(failure.error, ContactError::kNone);
  EXPECT_FALSE(failure.has_particle_id);
}

TEST(ComputeFirstFailure, FlagWithoutIndex_IsRunOnCallFailed) {
  CallArrays arrays(BuildFloorRange(10, {}));
  const FirstFailureMinima minima = {
      .smallest_failing_id = 9,
      .smallest_failing_index = 0xFFFFFFFFFFFFFFFF,
      .has_failure = 1,
  };
  EXPECT_EQ(ComputeFirstFailure(arrays.ToCall(), minima).error, ContactError::kRunOnCallFailed);
}

TEST(BuildContactStackEstimate, Cpu_CountsNothingOutsideTheBlock) {
  const std::expected<MemoryEstimateEntry, ContactFailure> entry = BuildContactStackEstimate(RunOn::kCpu);
  ASSERT_TRUE(entry.has_value());
  EXPECT_EQ(entry->where, MemoryWhere::kDriverStack);
  EXPECT_EQ(ComputeMemoryBytes(std::span(&*entry, 1), MemoryCounts()), 0U);
}

#if !defined(FLUIDITY_HAS_GPU)

TEST(BuildContactStackEstimate, GpuInBuildWithoutGpuCode_IsRunOnCallFailed) {
  const std::expected<MemoryEstimateEntry, ContactFailure> entry = BuildContactStackEstimate(RunOn::kGpu);
  ASSERT_FALSE(entry.has_value());
  EXPECT_EQ(entry.error().error, ContactError::kRunOnCallFailed);
}

#endif

TEST(ComputeFirstFailure, StartMinima_IsNoFailure) {
  CallArrays arrays(BuildFloorRange(1, {}));
  EXPECT_EQ(ComputeFirstFailure(arrays.ToCall(), FirstFailureMinima{}).error, ContactError::kNone);
}

TEST(ComputeIsFailureWithParticleId, FailingParticleWithThatId_IsTrue) {
  const std::array<std::size_t, 2> failing_indexes = {5, 700};
  ParticleRange range = BuildFloorRange(1000, failing_indexes);
  range.particle_id[5] = 50;
  range.particle_id[700] = 9;
  CallArrays arrays(range);
  const MoveParticlesCall call = arrays.ToCall();
  for (std::size_t index = 0; index < call.particle_count; ++index) {
    ApplyParticleVisitAt(call, index);
  }
  EXPECT_TRUE(ComputeIsFailureWithParticleId(call, 700, 9));
  EXPECT_FALSE(ComputeIsFailureWithParticleId(call, 5, 9));
  EXPECT_FALSE(ComputeIsFailureWithParticleId(call, 700, 50));
  EXPECT_FALSE(ComputeIsFailureWithParticleId(call, 6, 1006));
}

TEST(ComputeParticleCounts, SummedOverThreads_EqualsCpuCounts) {
  // Three particles over the fixed floor: one moving along above it, one falling onto it, one starting 0.005 m inside
  // it with start_overlap_tolerance 0.01 m: {1, 1} in every launch and order.
  ParticleRange range = BuildFloorRange(3, {});
  range.start_overlap_tolerance = 0.01;
  range.start_position = {{.x = 0, .y = 0, .z = 0.5}, {.x = 0, .y = 0, .z = 0.05}, {.x = 0, .y = 0, .z = -0.005}};
  range.velocity_for_this_step = {{.x = 1, .y = 0, .z = 0}, {.x = 0, .y = 0, .z = -1}, {}};
  range.velocity_for_next_step = range.velocity_for_this_step;
  const std::vector<GpuHalfAnswer> answers = RunGpuHalfInEveryLaunch(range);
  const bool is_one_and_one_every_time = std::ranges::all_of(answers, [](const GpuHalfAnswer& answer) {
    return answer.counts.hit_count == 1 && answer.counts.started_inside_count == 1;
  });
  EXPECT_TRUE(is_one_and_one_every_time);
  EXPECT_EQ(CheckGpuHalfSameAsCpuLoop(range), 8);
}

// One CPU run of the oblique particles packed against the solid of their first case, which hit and miss.
MovedRange RunPackedObliqueOnCpu() {
  return ComputeMovedRange(RunOn::kCpu, BuildPackedObliqueRange(300), ReactionArray::kGiven);
}

TEST(CheckSameMove, TwoCpuRuns_PassSameBits) {
  const ParticleRange range = BuildPackedObliqueRange(300);
  const MovedRange first = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  const MovedRange second = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  ASSERT_TRUE(first.result.has_value());
  EXPECT_GT(first.result->hit_count, 0U);
  EXPECT_TRUE(CheckSameMove(first, second, Agreement::kSameBits, ComputeComparisonBounds(range)));
  EXPECT_EQ(ToFieldBits(first), ToFieldBits(second));
}

TEST(CheckSameMove, PositionOneUnitInTheLastPlaceAway_FailsSameBitsPassesBounds) {
  const ParticleRange range = BuildPackedObliqueRange(300);
  const MovedRange reference = RunPackedObliqueOnCpu();
  MovedRange compared = reference;
  compared.position[17].y = std::nextafter(compared.position[17].y, 1e300);
  const ComparisonBounds bounds = ComputeComparisonBounds(range);
  EXPECT_FALSE(CheckSameMove(reference, compared, Agreement::kSameBits, bounds));
  EXPECT_TRUE(CheckSameMove(reference, compared, Agreement::kWithinBounds, bounds));
  EXPECT_GT(ComputeLargestDifference(reference, compared).position, 0);
  EXPECT_NE(ToFieldBits(reference), ToFieldBits(compared));
}

TEST(CheckSameMove, ValuesBeyondTheirFloors_FailBounds) {
  // A position tol / 2 away, a velocity 2^-30 of the largest speed away, a hit fraction 2^-30 away: each beyond its
  // floor of the GPU bounds (method paper <sec-contact>).
  const ParticleRange range = BuildPackedObliqueRange(300);
  const ComparisonBounds bounds = ComputeComparisonBounds(range);
  const MovedRange reference = RunPackedObliqueOnCpu();
  MovedRange moved_position = reference;
  moved_position.position[3].x += range.contact_tolerance / 2;
  EXPECT_FALSE(CheckSameMove(reference, moved_position, Agreement::kWithinBounds, bounds));
  MovedRange moved_velocity = reference;
  moved_velocity.velocity_for_next_step[3].z += 0x1p10 * bounds.speed_floor;
  EXPECT_FALSE(CheckSameMove(reference, moved_velocity, Agreement::kWithinBounds, bounds));
  MovedRange moved_share = reference;
  moved_share.record[3].hit_fraction += 0x1p-30;
  EXPECT_FALSE(CheckSameMove(reference, moved_share, Agreement::kWithinBounds, bounds));
}

TEST(CheckSameMove, DiscreteFieldDiffers_FailsBoth) {
  const ParticleRange range = BuildPackedObliqueRange(300);
  const MovedRange reference = RunPackedObliqueOnCpu();
  MovedRange compared = reference;
  compared.record[11].face = compared.record[11].face == SolidFace::kHighZ ? SolidFace::kLowZ : SolidFace::kHighZ;
  const ComparisonBounds bounds = ComputeComparisonBounds(range);
  EXPECT_FALSE(CheckSameMove(reference, compared, Agreement::kSameBits, bounds));
  EXPECT_FALSE(CheckSameMove(reference, compared, Agreement::kWithinBounds, bounds));
}

TEST(CheckSameMove, CountsDiffer_Fail) {
  const ParticleRange range = BuildPackedObliqueRange(300);
  const MovedRange reference = RunPackedObliqueOnCpu();
  MovedRange compared = reference;
  ++compared.result->hit_count;
  EXPECT_FALSE(CheckSameMove(reference, compared, Agreement::kWithinBounds, ComputeComparisonBounds(range)));
}

TEST(CheckSameMove, SuccessAgainstFailure_Fails) {
  const ParticleRange range = BuildPackedObliqueRange(300);
  const MovedRange reference = RunPackedObliqueOnCpu();
  MovedRange compared = reference;
  compared.result = std::unexpected(ContactFailure{.error = ContactError::kBadInput});
  EXPECT_FALSE(CheckSameMove(reference, compared, Agreement::kWithinBounds, ComputeComparisonBounds(range)));
}

TEST(CheckSameMove, FailuresNamingOtherParticles_Fail) {
  const std::array<std::size_t, 2> failing_indexes = {5, 700};
  const ParticleRange range = BuildFloorRange(1000, failing_indexes);
  const MovedRange reference = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  ASSERT_FALSE(reference.result.has_value());
  EXPECT_EQ(reference.result.error().particle_id, 1005U);
  MovedRange compared = reference;
  compared.result = std::unexpected(
      ContactFailure{
          .error = ContactError::kStartsInsideSolid,
          .has_particle_id = true,
          .particle_id = 1700,
          .solid_id = 1,
          .magnitude = 0.01,
      }
  );
  EXPECT_FALSE(CheckSameMove(reference, compared, Agreement::kSameBits, ComputeComparisonBounds(range)));
  EXPECT_TRUE(CheckSameMove(reference, reference, Agreement::kSameBits, ComputeComparisonBounds(range)));
}

TEST(ComputeAgreement, FixedFloor_IsSameBits) {
  const std::array<Solid, 1> solids = {move_particle_cases::BuildFloor(1)};
  EXPECT_EQ(ComputeAgreement(solids), Agreement::kSameBits);
}

TEST(ComputeAgreement, SpinningFloor_IsWithinBounds) {
  const SolidMotion spin = {.linear_velocity = {}, .angular_velocity = {.x = 0, .y = 0, .z = 1}};
  const std::array<Solid, 2> solids = {
      move_particle_cases::BuildFloor(1),
      move_particle_cases::BuildFlatSolid({}, move_particle_cases::kUnitZ, spin, 2),
  };
  EXPECT_EQ(ComputeAgreement(solids), Agreement::kWithinBounds);
}

TEST(ComputeComparisonBounds, FloorRange_TakesToleranceAndLargestSpeed) {
  // Velocities {1, 0, -1 + share}: the largest speed is Length({1, 0, -1}) = Sqrt(2) m/s.
  const ParticleRange range = BuildFloorRange(10, {});
  const ComparisonBounds bounds = ComputeComparisonBounds(range);
  EXPECT_EQ(bounds.position_floor, kTestContactTolerance / 4);
  EXPECT_EQ(bounds.speed_floor, 0x1p-40 * math::Sqrt(2));
  EXPECT_EQ(bounds.share_floor, 0x1p-40);
}

TEST(ToFieldBits, Failure_IsItsFieldsInOrder) {
  MovedRange moved;
  moved.result = std::unexpected(
      ContactFailure{
          .error = ContactError::kTooCloseToCall,
          .close_call = ContactCloseCall::kSqueezedBetweenSolids,
          .has_particle_id = true,
          .can_shorter_step_cure = true,
          .particle_id = 9,
          .solid_id = 1,
          .magnitude = 0.01,
      }
  );
  const std::vector<std::uint64_t> expected = {
      1,
      static_cast<std::uint64_t>(ContactError::kTooCloseToCall),
      static_cast<std::uint64_t>(ContactCloseCall::kSqueezedBetweenSolids),
      1,
      1,
      9,
      1,
      ToBits(0.01),
  };
  EXPECT_EQ(ToFieldBits(moved), expected);
}

TEST(ComputeMovedRange, FailedCallCheck_WritesNothing) {
  // A reacting floor without a reaction array fails the call check, before any particle is read.
  ParticleRange range = BuildFloorRange(10, {});
  range.solids[0].reports_reaction = true;
  const MovedRange moved = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kLeftOut);
  ASSERT_FALSE(moved.result.has_value());
  EXPECT_EQ(moved.result.error().error, ContactError::kBadInput);
  EXPECT_FALSE(moved.result.error().has_particle_id);
  EXPECT_TRUE(CheckWroteNothing(moved));
}

TEST(CheckWroteNothing, OneOutputWritten_FailsNamingIt) {
  const ParticleRange range = BuildFloorRange(10, {});
  MovedRange moved = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  EXPECT_FALSE(CheckWroteNothing(moved));
  moved.position.assign(10, kUnwrittenPosition);
  moved.velocity_for_next_step.assign(10, kUnwrittenPosition);
  moved.record.assign(10, ParticleContact{});
  moved.reaction.assign(10, SolidReaction{});
  EXPECT_TRUE(CheckWroteNothing(moved));
  moved.reaction[4].hit_position.y = 1;
  const ::testing::AssertionResult written = CheckWroteNothing(moved);
  EXPECT_FALSE(written);
  EXPECT_NE(std::string(written.message()).find("particle 4, reaction.hit_position.y"), std::string::npos);
}

// Ten turning scenes named "Scene", each the reference against itself except the cases in `flipped_cases`, whose
// compared run is `flipped`.
SetComparison CompareTenTurningScenes(
    const ParticleRange& range,
    const MovedRange& reference,
    const MovedRange& flipped,
    std::span<const std::size_t> flipped_cases
) {
  SetComparison comparison("test set");
  for (std::size_t case_index = 0; case_index < 10; ++case_index) {
    const bool is_flipped = std::ranges::find(flipped_cases, case_index) != flipped_cases.end();
    comparison.AddCase(
        case_index,
        "Scene",
        {.reference = reference, .compared = is_flipped ? flipped : reference},
        Agreement::kWithinBounds,
        ComputeComparisonBounds(range)
    );
  }
  return comparison;
}

TEST(SetComparison, TurningSceneWithDiscreteMismatch_IsCountedNotFailed) {
  // Ten turning scenes, one of which ends on another face on the "GPU": one mismatch, one allowed, the set passes and
  // the mismatch is a note line naming the case, the field and both values.
  const ParticleRange range = BuildFloorRange(1, {});
  const MovedRange reference = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  MovedRange flipped = reference;
  flipped.record[0].face = SolidFace::kLowZ;
  const std::array<std::size_t, 1> flipped_cases = {7};
  const SetComparison comparison = CompareTenTurningScenes(range, reference, flipped, flipped_cases);
  EXPECT_EQ(comparison.turning_case_count(), 10);
  EXPECT_EQ(comparison.discrete_mismatch_count(), 1);
  EXPECT_EQ(comparison.allowed_discrete_mismatch_count(), 1);
  EXPECT_TRUE(comparison.CheckPasses());
  ASSERT_EQ(comparison.mismatch_lines().size(), 1U);
  EXPECT_EQ(
      comparison.mismatch_lines()[0],
      "discrete mismatch, test set, case 7 Scene: particle 0, record.face: cpu "
          + std::to_string(static_cast<int>(reference.record[0].face)) + ", gpu "
          + std::to_string(static_cast<int>(SolidFace::kLowZ))
  );
}

TEST(SetComparison, TwoMismatchesInTenTurningScenes_Fails) {
  const ParticleRange range = BuildFloorRange(1, {});
  const MovedRange reference = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  MovedRange flipped = reference;
  flipped.record[0].solid_index = 5;
  const std::array<std::size_t, 2> flipped_cases = {0, 1};
  const SetComparison comparison = CompareTenTurningScenes(range, reference, flipped, flipped_cases);
  EXPECT_FALSE(comparison.CheckPasses());
}

TEST(SetComparison, DiscreteDifferenceWithoutTurn_Fails) {
  const ParticleRange range = BuildFloorRange(1, {});
  const MovedRange reference = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  MovedRange flipped = reference;
  flipped.record[0].hit_count = 3;
  SetComparison comparison("test set");
  comparison.AddCase(0, {}, {.reference = reference, .compared = flipped}, Agreement::kSameBits, {});
  EXPECT_EQ(comparison.failing_case_count(), 1);
  EXPECT_FALSE(comparison.CheckPasses());
}

TEST(SetComparison, ContinuousDifferenceBeyondBoundOnTurningScene_Fails) {
  const ParticleRange range = BuildFloorRange(1, {});
  const MovedRange reference = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  MovedRange moved = reference;
  moved.position[0].x += 1;
  SetComparison comparison("test set");
  comparison.AddCase(
      0,
      {},
      {.reference = reference, .compared = moved},
      Agreement::kWithinBounds,
      ComputeComparisonBounds(range)
  );
  EXPECT_EQ(comparison.failing_case_count(), 1);
  EXPECT_EQ(comparison.discrete_mismatch_count(), 0);
  EXPECT_FALSE(comparison.CheckPasses());
}

TEST(SetComparison, ManyFailingCases_ListsTwentyAndCountsTheRest) {
  const ParticleRange range = BuildFloorRange(1, {});
  const MovedRange reference = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  MovedRange moved = reference;
  moved.position[0].z += 1;
  SetComparison comparison("test set");
  for (std::size_t case_index = 0; case_index < 25; ++case_index) {
    comparison.AddCase(case_index, {}, {.reference = reference, .compared = moved}, Agreement::kSameBits, {});
  }
  const std::string message = comparison.CheckPasses().message();
  EXPECT_NE(message.find("test set, case 19:"), std::string::npos);
  EXPECT_EQ(message.find("test set, case 20:"), std::string::npos);
  EXPECT_NE(message.find("and 5 more failing scenes, not listed"), std::string::npos);
  EXPECT_NE(message.find("particle 0, position.z: cpu "), std::string::npos);
}

TEST(FindMoveDifferences, PositionApart_NamesFieldValuesDifferenceAndBound) {
  const ParticleRange range = BuildFloorRange(1, {});
  const MovedRange reference = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  MovedRange moved = reference;
  moved.position[0].x += 1;
  const std::vector<MoveDifference> differences =
      FindMoveDifferences(reference, moved, Agreement::kWithinBounds, ComputeComparisonBounds(range));
  ASSERT_EQ(differences.size(), 1U);
  EXPECT_FALSE(differences[0].is_discrete);
  EXPECT_EQ(differences[0].field, "particle 0, position.x");
  // 17 significant digits read back to the same double.
  EXPECT_EQ(
      std::stod(differences[0].reference_value.substr(0, differences[0].reference_value.find(' '))),
      reference.position[0].x
  );
  EXPECT_NE(differences[0].detail.find("bound "), std::string::npos);
}

}  // namespace
}  // namespace fluidity::contact
