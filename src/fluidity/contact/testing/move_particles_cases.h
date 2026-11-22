// Test-only: the helpers that the unit tests and the functional tests of move_particles share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    kTolerance, CpuContactBuffers, ParticleArrays, BuildFloorParticle, BuildFloorParticles
// Uses:    the headers it includes
// Used by: src/fluidity/contact/move_particles.test.cc and tests/contact/move_particles_in_every_thread_count.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

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

namespace fluidity::contact {

constexpr double kTolerance = kTestContactTolerance;  // m, 2^-30.

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

// One particle above the fixed floor z <= 0 (solid id 1), contact tolerance 2^-30 m, a step of 0.1 s.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): a start and a velocity, as SolidContactInputs orders them.
inline SolidContactInputs BuildFloorParticle(Vec3 start_position, Vec3 velocity) {
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
inline ParticleArrays BuildFloorParticles(std::size_t count, std::span<const std::size_t> failing_indexes) {
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

}  // namespace fluidity::contact
