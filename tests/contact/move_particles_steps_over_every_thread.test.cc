// Runs the GPU half of MoveParticles on the CPU, with the per-particle and per-thread functions its GPU functions call
// (contact/move_particles_steps.h), and compares the answer with the CPU loop's. Each GPU function is run as its launch
// would cover it: every thread block and every thread, each over its items by grid stride, once in launch order and
// once in reverse, in several launch sizes. An atomic minimum, or or addition of the GPU becomes a plain one here:
// their final values do not depend on the order, which the reverse runs show. The host steps follow
// contact/move_particles_gpu.cu (MoveParticlesOnGpuInLaunchSize). The tests of the comparison helpers that the GPU
// tests use (contact/testing/move_particles_runs.h) are here too, so they run in the gate now.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "fluidity/base/run_on_call.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/move_particles_runs.h"
#include "fluidity/contact/testing/move_particles_steps_cases.h"

namespace fluidity::contact {
namespace {

TEST(ApplyParticleVisitAt, EveryThreadBlockAndThread_SameAsCpuLoop) {
  // Every catalogue scene as a one-particle call, the 10,000 oblique particles as one call against the solid of their
  // first case, and the 10,000 round scenes as one call each.
  int compared_count = 0;
  std::uint64_t id = 1;
  for (const MoveParticleCase& test_case : BuildMoveParticleCases()) {
    SCOPED_TRACE(test_case.name);
    compared_count += CheckParallelHalfSameAsCpuLoop(BuildOneParticleRange(test_case, id++));
  }
  compared_count += CheckParallelHalfSameAsCpuLoop(BuildPackedObliqueRange(10000));
  for (const MoveParticleCase& test_case : BuildRoundCases(10000)) {
    compared_count += CheckParallelHalfSameAsCpuLoop(BuildOneParticleRange(test_case, id++));
  }
  EXPECT_GT(compared_count, 8 * 10000);
}

TEST(FirstFailureMinima, SeveralFailuresAnywhere_ReportLowestIdThenIndex) {
  // Failing particles at the first and last index and at thread block edges, with ids that put the lowest anywhere.
  const std::array<std::size_t, 5> failing_indexes = {0, 255, 256, 511, 999};
  for (const std::size_t lowest_index : failing_indexes) {
    SCOPED_TRACE(lowest_index);
    ParticleRange range = BuildFloorRange(1000, failing_indexes);
    for (const std::size_t failing_index : failing_indexes) {
      range.particle_id[failing_index] = 100 + failing_index;
    }
    range.particle_id[lowest_index] = 3;
    EXPECT_TRUE(CheckEveryFailure(ComputeParallelHalfInEveryLaunch(range), BuildFloorFailure(3, 0.01)));
  }
}

TEST(ComputeLaunchSize, GridStride_VisitsEveryParticleOnce) {
  // Around one thread block, and above the most thread blocks, where each thread takes two items.
  for (const std::size_t count : {
           std::size_t{1},
           std::size_t{255},
           std::size_t{256},
           std::size_t{257},
           std::size_t{1000000},
           (std::size_t{kMaxThreadBlockCount} * kThreadBlockSize) + 1,
       }) {
    std::vector<std::uint8_t> visit_counts(count, 0);
    ComputeLaunch(ComputeLaunchSize(count), count, ThreadOrder::kLaunchOrder, [&visit_counts](std::size_t item) {
      ++visit_counts[item];
    });
    EXPECT_TRUE(std::ranges::all_of(visit_counts, [](std::uint8_t visits) {
      return visits == 1;
    })) << count;
  }
}

}  // namespace
}  // namespace fluidity::contact
