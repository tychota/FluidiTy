#include "fluidity/host/motion_capture_3d.h"

#include <algorithm>
#include <cmath>
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

absl::StatusOr<LiquidCacheBytes3D> Encode(
    LiquidClock3D first,
    LiquidClock3D last,
    std::span<const LiquidCacheBytes3D> snapshots,
    LiquidCacheLimits3D limits
) {
  std::size_t byte_count = 0;
  for (const auto& state : snapshots) {
    if (state.size() > limits.maximum_bytes - byte_count) {
      return absl::ResourceExhaustedError("motion snapshots exceed byte limit");
    }
    byte_count += state.size();
  }
  const std::size_t tables = 2 + snapshots.size() + (byte_count + 511) / 512;
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
        return wire::CreateMotionFrame(builder, 1, 0, &first_clock, &last_clock, *states);
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
  return Encode(state.clock, state.clock, std::span(&*bytes, 1), limits);
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
    double end_time,
    std::size_t intervals,
    LiquidCacheLimits3D limits
)
    : start_(start),
      geometry_(geometry),
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
  MotionCapture3D result(frame_start.clock, frame_start.geometry, frame_end_time, intervals, limits);
  result.payload_bytes_ = initial->size();
  result.samples_.push_back(std::move(*initial));
  return result;
}

absl::Status MotionCapture3D::Observe(const LiquidState3D& accepted, bool rollback) {
  if (samples_.empty()) {
    return absl::FailedPreconditionError("motion collector has no initial sample");
  }
  if (rollback) {
    auto bytes = EncodeLiquidState3D(accepted, limits_);
    if (!bytes.ok()) {
      return bytes.status();
    }
    if (!std::ranges::equal(bytes->values(), samples_.front().values())) {
      return absl::InvalidArgumentError("motion replay did not return to frame start");
    }
    samples_.resize(1);
    payload_bytes_ = samples_.front().size();
    last_observed_ = start_;
    next_slot_ = 1;
    return absl::OkStatus();
  }
  const auto clock = accepted.clock;
  if (!SameGeometry(accepted.geometry, geometry_)) {
    return absl::InvalidArgumentError("motion sample changes grid geometry");
  }
  const bool completed_endpoint = clock.time == end_time_ && clock.frame_ordinal > start_.frame_ordinal
      && clock.frame_ordinal - start_.frame_ordinal == 1 && clock.substep_ordinal == 0;
  const bool intermediate = clock.time < end_time_ && clock.frame_ordinal == start_.frame_ordinal;
  if (!detail::ValidClock(clock) || clock.time <= last_observed_.time || clock.time > end_time_
      || clock.epoch <= last_observed_.epoch || (!completed_endpoint && !intermediate)) {
    return absl::InvalidArgumentError("motion sample is not a later accepted state");
  }
  const double threshold = std::lerp(start_.time, end_time_, static_cast<double>(next_slot_) / intervals_);
  if (clock.time >= threshold || clock.time == end_time_) {
    auto bytes = EncodeLiquidState3D(accepted, limits_);
    if (!bytes.ok()) {
      return bytes.status();
    }
    if (bytes->size() > limits_.maximum_bytes - payload_bytes_) {
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
  last_observed_ = clock;
  return absl::OkStatus();
}

std::size_t MotionCapture3D::sample_count() const {
  return samples_.size();
}

absl::StatusOr<LiquidCacheBytes3D> MotionCapture3D::Finish(const LiquidState3D& completed) const {
  const auto clock = completed.clock;
  if (samples_.size() < 2 || clock.time != end_time_ || clock.time != last_observed_.time
      || clock.epoch != last_observed_.epoch || clock.frame_ordinal <= start_.frame_ordinal
      || clock.frame_ordinal - start_.frame_ordinal != 1 || clock.substep_ordinal != 0) {
    return absl::FailedPreconditionError("motion interval has no coherent completed endpoint");
  }
  auto final = EncodeLiquidState3D(completed, limits_);
  if (!final.ok()) {
    return final.status();
  }
  if (!std::ranges::equal(final->values(), samples_.back().values())) {
    return absl::FailedPreconditionError("completed state differs from the accepted endpoint");
  }
  std::vector<LiquidCacheBytes3D> snapshots;
  for (std::size_t index = 0; index + 1 < samples_.size(); ++index) {
    const auto& source = samples_[index];
    auto bytes = LiquidCacheBytes3D::Create(source.size());
    if (!bytes.ok()) {
      return bytes.status();
    }
    const auto resized = bytes->Resize(source.size());
    if (!resized.ok()) {
      return resized;
    }
    std::copy(source.values().begin(), source.values().end(), bytes->values().begin());
    snapshots.push_back(std::move(*bytes));
  }
  snapshots.push_back(std::move(*final));
  return Encode(start_, clock, snapshots, limits_);
}
}  // namespace fluidity
