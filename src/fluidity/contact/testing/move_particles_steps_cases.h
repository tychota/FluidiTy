// Test-only: the helpers that the unit tests and the functional tests of move_particles_steps share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ThreadOrder, ToVisited, RunLaunch, CallArrays, GpuHalfAnswer, RunGpuHalfOnCpu, BuildLaunchSizes,
//          RunGpuHalfInEveryLaunch, CheckGpuHalfSameAsCpuLoop, IsSameFailure, CheckEveryFailure, BuildFloorFailure
// Uses:    the headers it includes
// Used by: src/fluidity/contact/move_particles_steps.test.cc and
//          tests/contact/move_particles_steps_over_every_thread.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/move_particles.h"
#include "fluidity/contact/move_particles_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"
#include "fluidity/contact/testing/contact_cases.h"
#include "fluidity/contact/testing/move_particles_runs.h"

namespace fluidity::contact {

// The order in which the simulated thread blocks and threads of one launch run.
enum class ThreadOrder : std::uint8_t {
  kLaunchOrder,  // Thread block 0, thread 0 first.
  kReverse,  // The last thread of the last thread block first.
};

// Index `visit` of `count`, in `order`.
inline std::size_t ToVisited(std::size_t visit, std::size_t count, ThreadOrder order) {
  return order == ThreadOrder::kLaunchOrder ? visit : count - 1 - visit;
}

// Calls `run_item(item)` for every item a launch of `launch_size` covers by grid stride, thread by thread in `order`,
// as ComputeGridStride gives each thread its items.
template <typename RunItem>
void RunLaunch(LaunchSize launch_size, std::size_t item_count, ThreadOrder order, RunItem run_item) {
  const std::size_t thread_count = std::size_t{launch_size.thread_block_count} * launch_size.thread_block_size;
  for (std::size_t block_visit = 0; block_visit < launch_size.thread_block_count; ++block_visit) {
    const std::size_t thread_block = ToVisited(block_visit, launch_size.thread_block_count, order);
    for (std::size_t thread_visit = 0; thread_visit < launch_size.thread_block_size; ++thread_visit) {
      const std::size_t thread = ToVisited(thread_visit, launch_size.thread_block_size, order);
      const std::size_t first_item = (thread_block * launch_size.thread_block_size) + thread;
      for (std::size_t item = first_item; item < item_count; item += thread_count) {
        run_item(item);
      }
    }
  }
}

// The arrays of one call in CPU memory, as the GPU half reads and writes them, with every output at
// kUnwrittenPosition or its default.
struct CallArrays {
  ParticleRange range;
  std::vector<Vec3> position;
  std::vector<Vec3> velocity_for_next_step;
  std::vector<ParticleContact> record;
  std::vector<SolidReaction> reaction;
  std::vector<ContactFailure> failure_per_particle;

  explicit CallArrays(ParticleRange particle_range)
      : range(std::move(particle_range)),
        position(range.size(), kUnwrittenPosition),
        velocity_for_next_step(range.size(), kUnwrittenPosition),
        record(range.size()),
        reaction(range.size()),
        failure_per_particle(range.size()) {
  }

  // The call the CPU half would pass to the GPU half, with every pointer in CPU memory.
  MoveParticlesCall ToCall() {
    return {
        .particle_count = range.size(),
        .particle_id = range.particle_id.data(),
        .start_position = range.start_position.data(),
        .velocity =
            {
                .for_next_step = range.velocity_for_next_step.data(),
                .for_this_step = range.velocity_for_this_step.data(),
            },
        .friction = range.friction.data(),
        .plate_side = range.plate_side.data(),
        .solids = range.solids.data(),
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
  }

  // The outputs and the answer of the GPU half in the form of a MoveParticles run.
  [[nodiscard]] MovedRange ToMovedRange(MoveParticleCounts counts, const ContactFailure& first_failure) const {
    MovedRange moved{
        .result =
            MoveParticlesResult{.hit_count = counts.hit_count, .started_inside_count = counts.started_inside_count},
        .position = position,
        .velocity_for_next_step = velocity_for_next_step,
        .record = record,
        .reaction = reaction,
    };
    if (first_failure.error != ContactError::kNone) {
      moved.result = std::unexpected(first_failure);
    }
    return moved;
  }
};

// The answer of the GPU half: its counts and its first failure.
struct GpuHalfAnswer {
  MoveParticleCounts counts;
  ContactFailure first_failure;
};

// MoveParticlesOnGpuInLaunchSize run on the CPU: its four GPU functions, each over every thread block and thread in
// `order`, with plain minima, or and additions where the GPU uses atomics. Fails the calling test when a launch does
// not visit every particle exactly once.
inline GpuHalfAnswer RunGpuHalfOnCpu(const MoveParticlesCall& call, LaunchSize launch_size, ThreadOrder order) {
  // Step 1: StartReduction.
  GpuHalfAnswer answer;
  FirstFailureMinima minima;
  // Step 2: VisitEveryParticle.
  std::vector<int> visit_counts(call.particle_count, 0);
  RunLaunch(launch_size, call.particle_count, order, [&](std::size_t index) {
    ++visit_counts[index];
    const ParticleVisit visit = ApplyParticleVisitAt(call, index);
    if (visit.has_failure) {
      minima.smallest_failing_id = std::min(minima.smallest_failing_id, visit.particle_id);
      minima.has_failure |= 1U;
      return;
    }
    answer.counts.hit_count += visit.counts.hit_count;
    answer.counts.started_inside_count += visit.counts.started_inside_count;
  });
  EXPECT_TRUE(std::ranges::all_of(visit_counts, [](int count) {
    return count == 1;
  }));
  // Step 3: FindSmallestFailingIndex.
  if (minima.has_failure != 0) {
    const std::uint64_t smallest_failing_id = minima.smallest_failing_id;
    RunLaunch(launch_size, call.particle_count, order, [&](std::size_t index) {
      if (ComputeIsFailureWithParticleId(call, index, smallest_failing_id)) {
        minima.smallest_failing_index = std::min<std::uint64_t>(minima.smallest_failing_index, index);
      }
    });
  }
  // Step 4: WriteFirstFailure.
  answer.first_failure = ComputeFirstFailure(call, minima);
  return answer;
}

// The launch sizes every range runs in: the GPU half's own, one thread, and two that give each thread several items.
inline std::vector<LaunchSize> BuildLaunchSizes(std::size_t particle_count) {
  return {
      ComputeLaunchSize(particle_count),
      {.thread_block_count = 1, .thread_block_size = 1},
      {.thread_block_count = 3, .thread_block_size = 64},
      {.thread_block_count = 7, .thread_block_size = 100},
  };
}

// The answers of the GPU half run on the CPU over `range` in every launch size of BuildLaunchSizes and both orders,
// each on fresh outputs.
inline std::vector<GpuHalfAnswer> RunGpuHalfInEveryLaunch(const ParticleRange& range) {
  std::vector<GpuHalfAnswer> answers;
  for (const LaunchSize launch_size : BuildLaunchSizes(range.size())) {
    for (const ThreadOrder order : {ThreadOrder::kLaunchOrder, ThreadOrder::kReverse}) {
      CallArrays arrays(range);
      answers.push_back(RunGpuHalfOnCpu(arrays.ToCall(), launch_size, order));
    }
  }
  return answers;
}

// Runs `range` through the CPU half's loop (MoveParticles on the CPU) and through the GPU half run on the CPU in every
// launch size and both orders, and checks they agree bit for bit: the result or the failure, and on success every
// output. A range whose call check fails never reaches the GPU half and is not compared. Returns how many runs of the
// GPU half were compared.
inline int CheckGpuHalfSameAsCpuLoop(const ParticleRange& range) {
  const MovedRange cpu_run = ComputeMovedRange(RunOn::kCpu, range, ReactionArray::kGiven);
  const bool failed_call_check = !cpu_run.result.has_value() && !cpu_run.result.error().has_particle_id;
  if (failed_call_check) {
    return 0;
  }
  const ComparisonBounds bounds = ComputeComparisonBounds(range);
  int compared_count = 0;
  for (const LaunchSize launch_size : BuildLaunchSizes(range.size())) {
    for (const ThreadOrder order : {ThreadOrder::kLaunchOrder, ThreadOrder::kReverse}) {
      CallArrays arrays(range);
      const GpuHalfAnswer answer = RunGpuHalfOnCpu(arrays.ToCall(), launch_size, order);
      const MovedRange gpu_half_run = arrays.ToMovedRange(answer.counts, answer.first_failure);
      EXPECT_TRUE(CheckSameMove(cpu_run, gpu_half_run, Agreement::kSameBits, bounds))
          << launch_size.thread_block_count << " thread blocks of " << launch_size.thread_block_size << " threads";
      ++compared_count;
    }
  }
  return compared_count;
}

// Whether two failures are the same field by field, the magnitude by its bits.
inline bool IsSameFailure(const ContactFailure& first, const ContactFailure& second) {
  return first.error == second.error && first.close_call == second.close_call
      && first.has_particle_id == second.has_particle_id && first.can_shorter_step_cure == second.can_shorter_step_cure
      && first.particle_id == second.particle_id && first.solid_id == second.solid_id
      && ToBits(first.magnitude) == ToBits(second.magnitude);
}

// Checks that every answer reports `expected`.
inline ::testing::AssertionResult
CheckEveryFailure(const std::vector<GpuHalfAnswer>& answers, const ContactFailure& expected) {
  for (std::size_t run = 0; run < answers.size(); ++run) {
    if (!IsSameFailure(answers[run].first_failure, expected)) {
      return ::testing::AssertionFailure()
          << "run " << run << " reports particle " << answers[run].first_failure.particle_id << ", magnitude "
          << answers[run].first_failure.magnitude;
    }
  }
  return ::testing::AssertionSuccess();
}

// A floor failure of kStartsInsideSolid for particle `particle_id`, `depth` m deep, as the floor range gives.
inline ContactFailure BuildFloorFailure(std::uint64_t particle_id, double depth) {
  return {
      .error = ContactError::kStartsInsideSolid,
      .has_particle_id = true,
      .particle_id = particle_id,
      .solid_id = 1,
      .magnitude = depth,
  };
}

}  // namespace fluidity::contact
