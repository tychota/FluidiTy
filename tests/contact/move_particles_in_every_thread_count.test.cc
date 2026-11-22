#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/move_particles.h"
#include "fluidity/contact/move_particles_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/testing/move_particles_cases.h"

namespace fluidity::contact {
namespace {

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

}  // namespace
}  // namespace fluidity::contact
