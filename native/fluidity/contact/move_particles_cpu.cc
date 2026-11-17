#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/check_can_run_on.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/contact_tolerance.h"
#include "fluidity/contact/move_particles.h"
#include "fluidity/contact/move_particles_steps.h"
#include "fluidity/contact/solid_contact_types.h"
#include "fluidity/contact/solid_shapes.h"

#if defined(FLUIDITY_HAS_GPU)
#include "fluidity/base/run_on_call.h"
#include "fluidity/contact/move_particles_gpu.h"
#endif

namespace fluidity::contact {
namespace {

// The one mapping of a buffer failure into a contact failure, for BuildContactBuffers and MoveParticles:
// kSizeMismatch gives kBuffersTooSmall; kOutOfMemory and kRunMemoryTooSmall give kOutOfMemory with the bytes asked
// for; kRunOnCallFailed, and any other value, give kRunOnCallFailed.
ContactFailure ToContactFailure(BufferError error, double asked_bytes) {
  switch (error) {
    case BufferError::kSizeMismatch:
      return {.error = ContactError::kBuffersTooSmall};
    case BufferError::kOutOfMemory:
    case BufferError::kRunMemoryTooSmall:
      return {.error = ContactError::kOutOfMemory, .magnitude = asked_bytes};
    case BufferError::kNone:
    case BufferError::kRunOnCallFailed:
      break;
  }
  return {.error = ContactError::kRunOnCallFailed};
}

// The bytes one buffer of `count` values asks for, as a failure's magnitude. A double, so a count too large for the
// product in std::size_t still gives its size.
template <typename Value>
double ComputeAskedBytes(std::size_t count) {
  return static_cast<double>(count) * static_cast<double>(sizeof(Value));
}

// The bytes [first, end) one array of a call covers; first == end for an array that is null or empty.
struct ByteRange {
  std::uintptr_t first = 0;
  std::uintptr_t end = 0;
};

template <typename Value>
ByteRange ToByteRange(const Value* values, std::size_t count) {
  if (values == nullptr || count == 0) {
    return {};
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): an address compared as a number, never dereferenced.
  const auto first = reinterpret_cast<std::uintptr_t>(values);
  return {.first = first, .end = first + (count * sizeof(Value))};
}

// Whether two byte ranges share no byte. An empty range shares none.
bool AreApart(ByteRange first, ByteRange second) {
  const bool is_either_empty = first.first == first.end || second.first == second.end;
  return is_either_empty || first.end <= second.first || second.end <= first.first;
}

// Checks the pointers of a call: every required one is given, and no output shares a byte with an input or with
// another output. Inputs may share bytes with each other: both velocities may be one array. An output written over its
// own input would read the outputs of particles moved earlier in a call that fails after them.
ContactError CheckCallArrays(const MoveParticlesInputs& inputs, const MovedParticles& moved, bool has_reacting_solid) {
  // Steps: 1 required pointers  2 the reaction array  3 overlaps
  const std::size_t count = inputs.particle_count;
  // Step 1: with no particle nothing is read or written, so null pointers are fine.
  const bool has_inputs = inputs.particle_id != nullptr && inputs.start_position != nullptr
      && inputs.velocity.for_next_step != nullptr && inputs.velocity.for_this_step != nullptr
      && inputs.friction != nullptr;
  const bool has_outputs =
      moved.position != nullptr && moved.velocity_for_next_step != nullptr && moved.record != nullptr;
  if (count > 0 && !(has_inputs && has_outputs)) {
    return ContactError::kBadInput;
  }
  // Step 2: a reacting solid's changes have nowhere else to go.
  if (count > 0 && has_reacting_solid && moved.reaction == nullptr) {
    return ContactError::kBadInput;
  }
  // Step 3: an output over an input would mix the step's state with this call's outputs.
  const std::array<ByteRange, 7> input_ranges = {
      ToByteRange(inputs.particle_id, count),
      ToByteRange(inputs.start_position, count),
      ToByteRange(inputs.velocity.for_next_step, count),
      ToByteRange(inputs.velocity.for_this_step, count),
      ToByteRange(inputs.friction, count),
      ToByteRange(inputs.plate_side, count),
      ToByteRange(inputs.solids.data(), inputs.solids.size()),
  };
  const std::array<ByteRange, 4> output_ranges = {
      ToByteRange(moved.position, count),
      ToByteRange(moved.velocity_for_next_step, count),
      ToByteRange(moved.record, count),
      ToByteRange(moved.reaction, count),
  };
  for (std::size_t output_index = 0; output_index < output_ranges.size(); ++output_index) {
    const ByteRange output = output_ranges[output_index];
    const bool is_apart_from_inputs = std::ranges::all_of(input_ranges, [output](ByteRange input) {
      return AreApart(output, input);
    });
    if (!is_apart_from_inputs) {
      return ContactError::kBadInput;
    }
    for (std::size_t other_index = output_index + 1; other_index < output_ranges.size(); ++other_index) {
      if (!AreApart(output, output_ranges[other_index])) {
        return ContactError::kBadInput;
      }
    }
  }
  return ContactError::kNone;
}

// Checks the scalars of a call as CheckSolidContactInputs checks them for one particle, once for all particles.
ContactError CheckCallScalars(const MoveParticlesInputs& inputs) {
  const bool step_duration_is_positive = math::IsFinite(inputs.step_duration) && inputs.step_duration > 0;
  const bool overlap_tolerance_is_valid =
      math::IsFinite(inputs.start_overlap_tolerance) && inputs.start_overlap_tolerance >= 0;
  const bool contact_tolerance_is_valid = math::IsFinite(inputs.contact_tolerance) && inputs.contact_tolerance > 0;
  if (!step_duration_is_positive || !overlap_tolerance_is_valid || !contact_tolerance_is_valid) {
    return ContactError::kBadInput;
  }
  return ContactError::kNone;
}

// Checks the solids of a call once for all particles: fewer than kNoSolid, strictly ascending solid_ids (as
// CheckSolidContactInputs), then, for each solid, its linear and angular velocity summed over their axes at most
// kLargestInputSpeedSum, CheckSolid for the step, and its origin over the step with its reach, as ComputeContactScale
// counts them, at most kLargestCoordinateInTolerances * contact_tolerance: what MoveParticle requires and does not
// check. The failure names the solid those three fail for, with the coordinate as magnitude for the last; the count
// and the order name none.
ContactFailure CheckCallSolids(std::span<const Solid> solids, double step_duration, double contact_tolerance) {
  if (solids.size() >= kNoSolid) {
    return {.error = ContactError::kBadInput};
  }
  for (std::size_t solid_index = 1; solid_index < solids.size(); ++solid_index) {
    if (solids[solid_index].solid_id <= solids[solid_index - 1].solid_id) {
      return {.error = ContactError::kBadInput};
    }
  }
  const auto compute_axes_sum = [](Vec3 vector) -> double {
    return math::Abs(vector.x) + math::Abs(vector.y) + math::Abs(vector.z);
  };
  for (const Solid& solid : solids) {
    // The speeds' part of MoveParticle's Requires that it does not check (kLargestInputSpeedSum).
    const bool is_too_fast = compute_axes_sum(solid.motion.linear_velocity) > kLargestInputSpeedSum
        || compute_axes_sum(solid.motion.angular_velocity) > kLargestInputSpeedSum;
    if (is_too_fast) {
      return {.error = ContactError::kBadInput, .solid_id = solid.solid_id};
    }
    ContactCloseCall close_call = ContactCloseCall::kNone;
    const ContactError error = CheckSolid(solid, step_duration, &close_call);
    if (error != ContactError::kNone) {
      return {.error = error, .close_call = close_call, .solid_id = solid.solid_id};
    }
    // The coordinates' part of MoveParticle's Requires, after CheckSolid, which ComputeContactScale requires. The
    // cell width there only sets thinnest_solid, which this check does not read.
    const double solid_coordinate = ComputeContactScale(&solid, 1, step_duration, contact_tolerance).largest_coordinate;
    if (solid_coordinate > kLargestCoordinateInTolerances * contact_tolerance) {
      return {.error = ContactError::kBadInput, .solid_id = solid.solid_id, .magnitude = solid_coordinate};
    }
  }
  return {};
}

// The loop on the CPU: every particle in index order, each moved and its outputs written at its index
// (ComputeParticleMoveAt, ApplyParticleOutputsAt). Keeps the earliest failure in the order of ComputeIsEarlierFailure,
// so the report does not depend on the order of the visits, and goes on after a failure so that every failing particle
// is seen.
std::expected<MoveParticleCounts, ContactFailure> ApplyLoopOnCpu(const MoveParticlesCall& call) {
  MoveParticleCounts counts;
  ContactFailure first_failure;
  std::size_t first_failure_index = 0;
  bool has_failure = false;
  for (std::size_t index = 0; index < call.particle_count; ++index) {
    ContactFailure failure;
    SolidContactResult result;
    const ContactError error = ComputeParticleMoveAt(call, index, &result, &failure);
    if (error != ContactError::kNone) {
      const bool is_first_failure = !has_failure
          || ComputeIsEarlierFailure(failure.particle_id, index, first_failure.particle_id, first_failure_index);
      if (is_first_failure) {
        first_failure = failure;
        first_failure_index = index;
        has_failure = true;
      }
      continue;
    }
    ApplyParticleOutputsAt(call, index, result);
    const MoveParticleCounts particle_counts = ComputeParticleCounts(result.kind);
    counts.hit_count += particle_counts.hit_count;
    counts.started_inside_count += particle_counts.started_inside_count;
  }
  if (has_failure) {
    return std::unexpected(first_failure);
  }
  return counts;
}

}  // namespace

// clang-tidy: the two maxima follow the processor and come before the run's memory, as every Build<Name>Buffers orders
// its parameters (AGENTS.md, working buffers).
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::expected<ContactBuffers, ContactFailure>
BuildContactBuffers(RunOn run_on, std::size_t max_particle_count, std::size_t max_solid_count, RunMemory& run_memory) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  // Steps: 1 the processor  2 the four pieces, in the order of BuildContactMemoryEstimate  3 on the GPU, the stack
  // Step 1: a processor that cannot run is reported before the block is touched, so a GPU asked for in a build
  // without GPU code fails here whatever block was passed.
  if (!CheckCanRunOn(run_on).has_value()) {
    return std::unexpected(ContactFailure{.error = ContactError::kRunOnCallFailed});
  }
  // Requires: one processor (BuildBuffer from a RunMemory asserts it too).
  assert(run_on == run_memory.run_on());

  // Step 2: on failure the pieces already taken stay in the block; the buffers built so far are destroyed here.
  ContactBuffers buffers;
  buffers.run_on_ = run_on;
  std::expected<Buffer<Solid>, BufferError> solids_copy = BuildBuffer<Solid>(run_on, max_solid_count, run_memory);
  if (!solids_copy.has_value()) {
    return std::unexpected(ToContactFailure(solids_copy.error(), ComputeAskedBytes<Solid>(max_solid_count)));
  }
  buffers.solids_copy_ = std::move(*solids_copy);
  std::expected<Buffer<ContactFailure>, BufferError> failure_per_particle =
      BuildBuffer<ContactFailure>(run_on, max_particle_count, run_memory);
  if (!failure_per_particle.has_value()) {
    const double asked_bytes = ComputeAskedBytes<ContactFailure>(max_particle_count);
    return std::unexpected(ToContactFailure(failure_per_particle.error(), asked_bytes));
  }
  buffers.failure_per_particle_ = std::move(*failure_per_particle);
  std::expected<Buffer<MoveParticleCounts>, BufferError> call_result =
      BuildBuffer<MoveParticleCounts>(run_on, 1, run_memory);
  if (!call_result.has_value()) {
    return std::unexpected(ToContactFailure(call_result.error(), ComputeAskedBytes<MoveParticlesResult>(1)));
  }
  buffers.call_result_ = std::move(*call_result);
  std::expected<Buffer<ContactFailure>, BufferError> first_failure = BuildBuffer<ContactFailure>(run_on, 1, run_memory);
  if (!first_failure.has_value()) {
    return std::unexpected(ToContactFailure(first_failure.error(), ComputeAskedBytes<ContactFailure>(1)));
  }
  buffers.first_failure_ = std::move(*first_failure);

#if defined(FLUIDITY_HAS_GPU)
  // Step 3: the visits' stack, taken now so that a GPU too full for it fails here rather than at the first launch.
  if (run_on == RunOn::kGpu) {
    VisitStackOnGpu stack;
    if (ComputeVisitStackOnGpu(&stack) != RunOnCallStatus::kDone) {
      return std::unexpected(ContactFailure{.error = ContactError::kRunOnCallFailed});
    }
    const RunOnCallStatus reserved = ReserveVisitStackOnGpu(stack.bytes_per_thread);
    if (reserved == RunOnCallStatus::kOutOfMemory) {
      const double stack_bytes =
          static_cast<double>(stack.bytes_per_thread) * static_cast<double>(stack.resident_threads);
      return std::unexpected(ContactFailure{.error = ContactError::kOutOfMemory, .magnitude = stack_bytes});
    }
    if (reserved != RunOnCallStatus::kDone) {
      return std::unexpected(ContactFailure{.error = ContactError::kRunOnCallFailed});
    }
  }
#endif
  return buffers;
}

std::expected<MemoryEstimateEntry, ContactFailure> BuildContactStackEstimate(RunOn run_on) {
  MemoryEstimateEntry entry = {
      .name = "contact: GPU stack of the particle visits",
      .counted_per = CountedPer::kGivenCount,
      .given_count = 0,
      .bytes_each = 0,
      .copies = 1,
      .kept_for = KeptFor::kWholeRun,
      .where = MemoryWhere::kDriverStack,
  };
  if (run_on == RunOn::kCpu) {
    return entry;
  }
  if (!CheckCanRunOn(run_on).has_value()) {
    return std::unexpected(ContactFailure{.error = ContactError::kRunOnCallFailed});
  }
#if defined(FLUIDITY_HAS_GPU)
  VisitStackOnGpu stack;
  if (ComputeVisitStackOnGpu(&stack) != RunOnCallStatus::kDone) {
    return std::unexpected(ContactFailure{.error = ContactError::kRunOnCallFailed});
  }
  entry.given_count = stack.resident_threads;
  entry.bytes_each = static_cast<std::uint32_t>(stack.bytes_per_thread);
  return entry;
#else
  // Without GPU code CheckCanRunOn never accepts the GPU, so no caller reaches this.
  return std::unexpected(ContactFailure{.error = ContactError::kRunOnCallFailed});
#endif
}

std::expected<MoveParticlesResult, ContactFailure>
MoveParticles(const MoveParticlesInputs& inputs, const MovedParticles& moved, ContactBuffers& buffers) {
  // Steps (method paper <sec-contact>): 1 call checks  2 copy the solids  3 the loop
  // Step 1: every check of the call comes before any particle is read, so its failures name no particle.
  const bool fits_buffers =
      inputs.particle_count <= buffers.max_particle_count() && inputs.solids.size() <= buffers.max_solid_count();
  if (!fits_buffers) {
    return std::unexpected(ContactFailure{.error = ContactError::kBuffersTooSmall});
  }
  const bool has_reacting_solid = std::ranges::any_of(inputs.solids, [](const Solid& solid) {
    return solid.reports_reaction;
  });
  const ContactError arrays_error = CheckCallArrays(inputs, moved, has_reacting_solid);
  if (arrays_error != ContactError::kNone) {
    return std::unexpected(ContactFailure{.error = arrays_error});
  }
  const ContactError scalars_error = CheckCallScalars(inputs);
  if (scalars_error != ContactError::kNone) {
    return std::unexpected(ContactFailure{.error = scalars_error});
  }
  const ContactFailure solids_failure = CheckCallSolids(inputs.solids, inputs.step_duration, inputs.contact_tolerance);
  if (solids_failure.error != ContactError::kNone) {
    return std::unexpected(solids_failure);
  }

  // Step 2: the particles read the copy, on the processor that moves them.
  const std::expected<void, BufferError> copied = WriteBuffer(inputs.solids, buffers.solids_copy_, 0);
  if (!copied.has_value()) {
    return std::unexpected(ToContactFailure(copied.error(), 0));
  }
  buffers.solids_copy_count_ = inputs.solids.size();
  if (inputs.particle_count == 0) {
    return MoveParticlesResult{};
  }
  const MoveParticlesCall call = {
      .particle_count = inputs.particle_count,
      .particle_id = inputs.particle_id,
      .start_position = inputs.start_position,
      .velocity = inputs.velocity,
      .friction = inputs.friction,
      .plate_side = inputs.plate_side,
      .solids = buffers.solids_copy_.data(),
      .position = moved.position,
      .velocity_for_next_step = moved.velocity_for_next_step,
      .record = moved.record,
      .reaction = moved.reaction,
      .failure_per_particle = buffers.run_on() == RunOn::kGpu ? buffers.failure_per_particle_.data() : nullptr,
      .step_duration = inputs.step_duration,
      .start_overlap_tolerance = inputs.start_overlap_tolerance,
      .contact_tolerance = inputs.contact_tolerance,
      .solid_count = static_cast<std::uint32_t>(inputs.solids.size()),
  };

  // Step 3 on the GPU: the loop and the reduction run there; the CPU reads the two answers.
  if (buffers.run_on() == RunOn::kGpu) {
#if defined(FLUIDITY_HAS_GPU)
    const RunOnCallStatus status =
        MoveParticlesOneThreadPerParticleOnGpu(call, buffers.call_result_.data(), buffers.first_failure_.data());
    if (status != RunOnCallStatus::kDone) {
      return std::unexpected(ToContactFailure(ToBufferError(status), 0));
    }
    ContactFailure first_failure;
    MoveParticleCounts counts;
    const std::expected<void, BufferError> failure_read =
        ReadBuffer(buffers.first_failure_, 0, std::span<ContactFailure>(&first_failure, 1));
    const std::expected<void, BufferError> counts_read =
        ReadBuffer(buffers.call_result_, 0, std::span<MoveParticleCounts>(&counts, 1));
    if (!failure_read.has_value() || !counts_read.has_value()) {
      return std::unexpected(ContactFailure{.error = ContactError::kRunOnCallFailed});
    }
    if (first_failure.error != ContactError::kNone) {
      return std::unexpected(first_failure);
    }
    return MoveParticlesResult{.hit_count = counts.hit_count, .started_inside_count = counts.started_inside_count};
#else
    // Without GPU code BuildContactBuffers never builds GPU buffers, so no caller reaches this.
    return std::unexpected(ContactFailure{.error = ContactError::kRunOnCallFailed});
#endif
  }

  // Step 3: one visit per particle; after a failure the outputs are unspecified and the step discards the try.
  const std::expected<MoveParticleCounts, ContactFailure> counts = ApplyLoopOnCpu(call);
  if (!counts.has_value()) {
    return std::unexpected(counts.error());
  }
  return MoveParticlesResult{.hit_count = counts->hit_count, .started_inside_count = counts->started_inside_count};
}

std::vector<MemoryEstimateEntry>
BuildContactMemoryEstimate(std::size_t max_particle_count, std::size_t max_solid_count) {
  const auto build_entry = [](std::string_view name, std::size_t count, std::size_t bytes_each) -> MemoryEstimateEntry {
    return {
        .name = name,
        .counted_per = CountedPer::kGivenCount,
        .given_count = count,
        .bytes_each = static_cast<std::uint32_t>(bytes_each),
        .copies = 1,
        .kept_for = KeptFor::kWholeRun,
        .where = MemoryWhere::kRunOn,
    };
  };
  return {
      build_entry("contact: solids copy", max_solid_count, sizeof(Solid)),
      build_entry("contact: failure per particle", max_particle_count, sizeof(ContactFailure)),
      build_entry("contact: call result", 1, sizeof(MoveParticlesResult)),
      build_entry("contact: first failure", 1, sizeof(ContactFailure)),
  };
}

}  // namespace fluidity::contact
