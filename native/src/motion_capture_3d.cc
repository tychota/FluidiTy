#include "fluidity/host/motion_capture_3d.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

#include "liquid_cache_3d_internal.h"
#include "motion_frame_generated.h"

namespace fluidity {
namespace {
namespace detail = liquid_cache_internal;
namespace wire = effects::v1;
bool SameGeometry(UniformMacGeometry3D a, UniformMacGeometry3D b) {
  return a.cell_width == b.cell_width && a.world_origin.x == b.world_origin.x && a.world_origin.y == b.world_origin.y
      && a.world_origin.z == b.world_origin.z;
}

std::size_t MaximumBirths(LiquidCacheLimits3D limits) {
  return std::min(std::size_t{1} << 20, limits.maximum_bytes / sizeof(wire::AcceptedWhitewaterBirth));
}

auto CellOrder(CellIndex3D cell) {
  return std::tuple(cell.z, cell.y, cell.x);
}

bool ValidBirth(const WhitewaterBirthEvent3D& event, double start_time, double end_time) {
  return std::isfinite(event.time_seconds) && event.time_seconds >= start_time && event.time_seconds <= end_time
      && detail::Finite(event.position) && detail::Finite(event.velocity) && detail::Finite(event.normal)
      && std::abs(std::hypot(event.normal.x, event.normal.y, event.normal.z) - 1) <= 1e-8
      && detail::Positive(event.volume_time_m3_seconds);
}

bool FollowsBirth(const AcceptedWhitewaterBirth3D& previous, const AcceptedWhitewaterBirth3D& current) {
  if (previous.accepted_step_ordinal == current.accepted_step_ordinal) {
    return previous.birth.event.time_seconds == current.birth.event.time_seconds
        && CellOrder(previous.birth.source_cell) < CellOrder(current.birth.source_cell);
  }
  return previous.accepted_step_ordinal < current.accepted_step_ordinal
      && previous.birth.event.time_seconds <= current.birth.event.time_seconds;
}

absl::StatusOr<LiquidCacheBytes3D> Encode(
    LiquidClock3D first,
    LiquidClock3D last,
    std::span<const LiquidCacheBytes3D> snapshots,
    std::span<const AcceptedWhitewaterBirth3D> births,
    LiquidCacheLimits3D limits
) {
  if (births.size() > MaximumBirths(limits)) {
    return absl::ResourceExhaustedError("accepted whitewater birth budget exceeded");
  }
  std::size_t byte_count = births.size() * sizeof(wire::AcceptedWhitewaterBirth);
  for (const auto& state : snapshots) {
    if (state.size() > limits.maximum_bytes - byte_count) {
      return absl::ResourceExhaustedError("motion snapshots exceed byte limit");
    }
    byte_count += state.size();
  }
  const std::size_t tables = 2 + snapshots.size() + byte_count / 512 + (byte_count % 512 != 0);
  return detail::Build<wire::MotionFrame>(
      tables,
      limits,
      "FLMO",
      [&](auto& builder) -> absl::StatusOr<flatbuffers::Offset<wire::MotionFrame>> {
        auto states = detail::BuildVector<wire::MotionSnapshot>(builder, snapshots.size(), [&](std::size_t index) {
          const auto bytes = snapshots[index].values();
          builder.ForceVectorAlignment(bytes.size(), sizeof(std::uint8_t), 8);
          return wire::CreateMotionSnapshot(builder, builder.CreateVector(bytes.data(), bytes.size()));
        });
        if (!states.ok()) {
          return states.status();
        }
        const auto first_clock = detail::Pack(first);
        const auto last_clock = detail::Pack(last);
        flatbuffers::Offset<flatbuffers::Vector<const wire::AcceptedWhitewaterBirth*>> birth_data;
        if (!births.empty()) {
          std::vector<wire::AcceptedWhitewaterBirth> records;
          records.reserve(births.size());
          for (const auto& accepted : births) {
            const auto& cell = accepted.birth.source_cell;
            const auto& event = accepted.birth.event;
            records.emplace_back(
                accepted.accepted_step_ordinal,
                detail::wire::Index3(cell.x, cell.y, cell.z),
                event.time_seconds,
                detail::Pack(event.position),
                detail::Pack(event.velocity),
                detail::Pack(event.normal),
                event.volume_time_m3_seconds
            );
          }
          birth_data = builder.CreateVectorOfStructs(records);
        }
        return wire::CreateMotionFrame(builder, 1, 0, &first_clock, &last_clock, *states, birth_data);
      }
  );
}
}  // namespace

absl::StatusOr<LiquidCacheBytes3D> EncodeMotionSeed3D(const LiquidState3D& state, LiquidCacheLimits3D limits) {
  if (state.clock != LiquidClock3D{}) {
    return absl::InvalidArgumentError("motion seed must have a zero clock");
  }
  auto bytes = EncodeLiquidState3D(state, limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  return Encode(state.clock, state.clock, std::span(&*bytes, 1), {}, limits);
}

absl::StatusOr<MotionFrame3D> DecodeMotionFrame3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  auto verified = detail::Verify<wire::MotionFrame>(bytes, "FLMO", limits);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* frame = *verified;
  const auto first = detail::Unpack(frame->first_clock());
  const auto last = detail::Unpack(frame->last_clock());
  if (!detail::ValidClock(first) || !detail::ValidClock(last) || first.substep_ordinal != 0 || last.substep_ordinal != 0
      || frame->snapshots()->size() == 0 || frame->snapshots()->size() > 65) {
    return absl::DataLossError("invalid motion frame clocks or count");
  }
  MotionFrame3D result{.first_clock = first, .last_clock = last};
  if (const auto* births = frame->whitewater_births()) {
    if (births->size() > MaximumBirths(limits)) {
      return absl::ResourceExhaustedError("accepted whitewater birth budget exceeded");
    }
    result.whitewater_births.reserve(births->size());
    for (const auto* record : *births) {
      const auto& cell = record->source_cell();
      const AcceptedWhitewaterBirth3D birth{
          .accepted_step_ordinal = record->accepted_step(),
          .birth =
              {
                  .source_cell = {cell.x(), cell.y(), cell.z()},
                  .event =
                      {
                          .time_seconds = record->time_seconds(),
                          .position = detail::Unpack(&record->position()),
                          .velocity = detail::Unpack(&record->velocity()),
                          .normal = detail::Unpack(&record->normal()),
                          .volume_time_m3_seconds = record->volume_time_m3_seconds(),
                      },
              },
      };
      if (birth.accepted_step_ordinal <= first.accepted_step_ordinal
          || birth.accepted_step_ordinal > last.accepted_step_ordinal
          || !ValidBirth(birth.birth.event, first.time, last.time)
          || (!result.whitewater_births.empty() && !FollowsBirth(result.whitewater_births.back(), birth))) {
        return absl::DataLossError("invalid accepted whitewater birth or event order");
      }
      result.whitewater_births.push_back(birth);
    }
  }
  std::vector<MotionStamp3D> stamps;
  for (const auto* snapshot : *frame->snapshots()) {
    const auto* source = snapshot->state();
    auto state = DecodeLiquidState3D(std::span(source->data(), source->size()), limits);
    if (!state.ok()) {
      return state.status();
    }
    stamps.push_back({state->clock.time, state->clock.epoch});
    if (!result.samples.empty()) {
      const auto& initial = result.samples.front();
      const auto& previous = result.samples.back();
      const bool endpoint = result.samples.size() + 1 == frame->snapshots()->size();
      const bool valid_clock = endpoint
          ? state->clock == last
          : state->clock.frame_ordinal == first.frame_ordinal && state->clock.substep_ordinal > 0;
      if (!SameGeometry(state->geometry, initial.geometry) || state->random_seed != initial.random_seed
          || state->clock.accepted_step_ordinal <= previous.clock.accepted_step_ordinal || !valid_clock) {
        return absl::DataLossError("motion sample geometry, seed or clock changed inconsistently");
      }
    }
    result.samples.push_back(std::move(*state));
  }
  if (result.samples.front().clock != first || result.samples.back().clock != last) {
    return absl::DataLossError("motion endpoint clocks differ from samples");
  }
  if (last.frame_ordinal == 0) {
    if (first != LiquidClock3D{} || last != first || result.samples.size() != 1) {
      return absl::DataLossError("invalid motion seed");
    }
  } else {
    const auto interval = MotionInterval3D::Create(first, last, stamps);
    if (!interval.ok()) {
      return absl::DataLossError("invalid motion interval stamps");
    }
  }
  return result;
}

MotionCapture3D::MotionCapture3D(
    LiquidClock3D start,
    UniformMacGeometry3D geometry,
    std::uint64_t random_seed,
    double end_time,
    std::size_t intervals,
    LiquidCacheLimits3D limits
)
    : start_(start),
      geometry_(geometry),
      random_seed_(random_seed),
      last_observed_(start),
      end_time_(end_time),
      intervals_(intervals),
      limits_(limits) {
}

absl::StatusOr<MotionCapture3D> MotionCapture3D::Create(
    const LiquidState3D& frame_start,
    double frame_end_time,
    std::size_t intervals,
    LiquidCacheLimits3D limits
) {
  if (!std::isfinite(frame_end_time) || frame_end_time <= frame_start.clock.time
      || frame_start.clock.substep_ordinal != 0 || intervals == 0 || intervals > 64) {
    return absl::InvalidArgumentError("invalid motion capture interval");
  }
  auto initial = EncodeLiquidState3D(frame_start, limits);
  if (!initial.ok()) {
    return initial.status();
  }
  MotionCapture3D
      result(frame_start.clock, frame_start.geometry, frame_start.random_seed, frame_end_time, intervals, limits);
  result.payload_bytes_ = initial->size();
  result.samples_.push_back(std::move(*initial));
  return result;
}

absl::Status
MotionCapture3D::Observe(const LiquidState3D& accepted, bool rollback, std::span<const WhitewaterCellBirth3D> births) {
  if (samples_.empty()) {
    return absl::FailedPreconditionError("motion collector has no initial sample");
  }
  if (rollback) {
    if (!births.empty()) {
      return absl::InvalidArgumentError("rollback cannot carry accepted births");
    }
    auto bytes = EncodeLiquidState3D(accepted, limits_);
    if (!bytes.ok()) {
      return bytes.status();
    }
    if (!std::ranges::equal(bytes->values(), samples_.front().values())) {
      return absl::InvalidArgumentError("motion replay did not return to frame start");
    }
    samples_.resize(1);
    whitewater_births_.clear();
    payload_bytes_ = samples_.front().size();
    last_observed_ = start_;
    next_slot_ = 1;
    return absl::OkStatus();
  }
  const auto clock = accepted.clock;
  if (!SameGeometry(accepted.geometry, geometry_) || accepted.random_seed != random_seed_) {
    return absl::InvalidArgumentError("motion sample changes grid geometry or random seed");
  }
  const bool completed_endpoint = clock.time == end_time_ && clock.frame_ordinal > start_.frame_ordinal
      && clock.frame_ordinal - start_.frame_ordinal == 1 && clock.substep_ordinal == 0;
  const bool intermediate =
      clock.time < end_time_ && clock.frame_ordinal == start_.frame_ordinal && clock.substep_ordinal > 0;
  if (!detail::ValidClock(clock) || clock.time <= last_observed_.time || clock.time > end_time_
      || clock.accepted_step_ordinal <= last_observed_.accepted_step_ordinal || clock.epoch <= last_observed_.epoch
      || (!completed_endpoint && !intermediate)) {
    return absl::InvalidArgumentError("motion sample is not a later accepted state");
  }
  if (births.size() > MaximumBirths(limits_) - whitewater_births_.size()) {
    return absl::ResourceExhaustedError("accepted whitewater birth budget exceeded");
  }
  const std::size_t retained_birth_bytes =
      (whitewater_births_.size() + births.size()) * sizeof(wire::AcceptedWhitewaterBirth);
  if (retained_birth_bytes > limits_.maximum_bytes - payload_bytes_) {
    return absl::ResourceExhaustedError("motion capture byte limit exceeded by whitewater births");
  }
  for (std::size_t index = 0; index < births.size(); ++index) {
    const auto& birth = births[index];
    if (!ValidBirth(birth.event, last_observed_.time, clock.time)) {
      return absl::InvalidArgumentError("whitewater birth lies outside the accepted step or has invalid fields");
    }
    if (index > 0
        && (birth.event.time_seconds != births.front().event.time_seconds
            || CellOrder(births[index - 1].source_cell) >= CellOrder(birth.source_cell))) {
      return absl::InvalidArgumentError("whitewater births require one timestamp and unique ordered source cells");
    }
  }
  const double threshold = std::lerp(start_.time, end_time_, static_cast<double>(next_slot_) / intervals_);
  if (clock.time >= threshold || clock.time == end_time_) {
    auto bytes = EncodeLiquidState3D(accepted, limits_);
    if (!bytes.ok()) {
      return bytes.status();
    }
    if (bytes->size() > limits_.maximum_bytes - payload_bytes_ - retained_birth_bytes) {
      return absl::ResourceExhaustedError("motion capture byte limit exceeded");
    }
    payload_bytes_ += bytes->size();
    samples_.push_back(std::move(*bytes));
    const double phase = (clock.time - start_.time) / (end_time_ - start_.time);
    next_slot_ = std::min(
        intervals_ + 1,
        std::max(next_slot_ + 1, static_cast<std::size_t>(std::floor(phase * intervals_)) + 1)
    );
  }
  for (const auto& birth : births) {
    whitewater_births_.push_back({.accepted_step_ordinal = clock.accepted_step_ordinal, .birth = birth});
  }
  last_observed_ = clock;
  return absl::OkStatus();
}

std::size_t MotionCapture3D::sample_count() const {
  return samples_.size();
}

std::size_t MotionCapture3D::birth_count() const {
  return whitewater_births_.size();
}

absl::StatusOr<LiquidCacheBytes3D> MotionCapture3D::Finish(const LiquidState3D& completed) const {
  const auto clock = completed.clock;
  if (samples_.size() < 2 || clock.time != end_time_ || clock.time != last_observed_.time
      || clock.epoch != last_observed_.epoch || clock.frame_ordinal <= start_.frame_ordinal
      || clock.frame_ordinal - start_.frame_ordinal != 1 || clock.substep_ordinal != 0) {
    return absl::FailedPreconditionError("motion interval has no coherent completed endpoint");
  }
  {
    const auto final = EncodeLiquidState3D(completed, limits_);
    if (!final.ok()) {
      return final.status();
    }
    if (!std::ranges::equal(final->values(), samples_.back().values())) {
      return absl::FailedPreconditionError("completed state differs from the accepted endpoint");
    }
  }
  // The verified endpoint already lives in samples_. Reuse retained buffers
  // instead of copying the whole carrier before FlatBuffers serialization.
  return Encode(start_, clock, samples_, whitewater_births_, limits_);
}
}  // namespace fluidity
