#include "fluidity/host/whitewater_cache_3d.h"

#include "liquid_cache_3d_internal.h"
#include "whitewater_frame_generated.h"
#include "whitewater_settings_generated.h"

namespace fluidity {
namespace {
namespace detail = liquid_cache_internal;
namespace wire = effects::v1;

absl::Status CheckCounts(std::size_t points, std::size_t cells, WhitewaterCacheLimits3D limits) {
  if (points > std::min(limits.maximum_points, std::size_t{1} << 20)
      || cells > std::min(limits.maximum_source_cells, std::size_t{4096})) {
    return absl::ResourceExhaustedError("whitewater checkpoint record budget exceeded");
  }
  const std::size_t record_bytes = points * sizeof(wire::WhitewaterPoint) + cells * sizeof(wire::WhitewaterRemainder);
  if (record_bytes > limits.cache.maximum_bytes) {
    return absl::ResourceExhaustedError("whitewater checkpoint record bytes exceed cache budget");
  }
  return absl::OkStatus();
}

auto CellOrder(CellIndex3D cell) {
  return std::tuple(cell.z, cell.y, cell.x);
}

absl::Status CheckFrame(const WhitewaterFrame3D& frame) {
  const auto& emission = frame.emission;
  if (!detail::ValidClock(frame.clock) || frame.clock.substep_ordinal != 0 || emission.next_point_id == 0
      || emission.last_accepted_step_ordinal != frame.clock.accepted_step_ordinal) {
    return absl::InvalidArgumentError("invalid whitewater checkpoint clock or emission cursor");
  }
  if (frame.clock.frame_ordinal == 0
      && (frame.clock != LiquidClock3D{} || emission.next_point_id != 1 || !emission.remainders.empty()
          || !frame.points.empty())) {
    return absl::InvalidArgumentError("whitewater seed checkpoint must be empty");
  }
  if (frame.clock.frame_ordinal != 0 && frame.clock.time <= 0) {
    return absl::InvalidArgumentError("nonzero whitewater frame requires positive time");
  }
  std::uint64_t previous_id = 0;
  for (const auto& record : frame.points) {
    const auto& point = record.point;
    if (record.id <= previous_id || record.id >= emission.next_point_id || point.kind > WhitewaterKind3D::kBubble
        || !detail::Nonnegative(point.birth_time_seconds) || point.birth_time_seconds > frame.clock.time
        || !detail::Finite(point.kinematics.position) || !detail::Finite(point.kinematics.velocity)) {
      return absl::InvalidArgumentError("invalid whitewater point checkpoint record");
    }
    previous_id = record.id;
  }
  for (std::size_t index = 0; index < emission.remainders.size(); ++index) {
    const auto& remainder = emission.remainders[index];
    if (!detail::Positive(remainder.fractional_count) || remainder.fractional_count >= 1
        || (index > 0 && CellOrder(emission.remainders[index - 1].source_cell) >= CellOrder(remainder.source_cell))) {
      return absl::InvalidArgumentError("invalid or unordered whitewater checkpoint remainder");
    }
  }
  return absl::OkStatus();
}

bool ValidSettings(const WhitewaterFrameOptions3D& options) {
  const auto& transport = options.transport;
  return detail::Positive(options.emission_rate_per_m3_second) && detail::Positive(options.represented_count_per_point)
      && detail::Positive(transport.classification_radius_m) && detail::Nonnegative(transport.spray_drag_per_second)
      && detail::Nonnegative(transport.bubble_drag_per_second) && detail::Nonnegative(transport.bubble_buoyancy_scale)
      && detail::Positive(transport.foam_lifetime_seconds) && detail::Finite(transport.ambient_velocity);
}
}  // namespace

absl::StatusOr<LiquidCacheBytes3D>
EncodeWhitewaterFrame3D(const WhitewaterFrame3D& frame, WhitewaterCacheLimits3D limits) {
  const auto counts = CheckCounts(frame.points.size(), frame.emission.remainders.size(), limits);
  if (!counts.ok()) {
    return counts;
  }
  const auto checked = CheckFrame(frame);
  if (!checked.ok()) {
    return checked;
  }
  std::vector<wire::WhitewaterPoint> points;
  points.reserve(frame.points.size());
  for (const auto& record : frame.points) {
    points.emplace_back(
        record.id,
        detail::Pack(record.point.kinematics.position),
        detail::Pack(record.point.kinematics.velocity),
        record.point.birth_time_seconds,
        static_cast<wire::WhitewaterKind>(record.point.kind)
    );
  }
  std::vector<wire::WhitewaterRemainder> remainders;
  remainders.reserve(frame.emission.remainders.size());
  for (const auto& remainder : frame.emission.remainders) {
    const auto& cell = remainder.source_cell;
    remainders.emplace_back(detail::wire::Index3(cell.x, cell.y, cell.z), remainder.fractional_count);
  }
  const auto record_bytes =
      points.size() * sizeof(wire::WhitewaterPoint) + remainders.size() * sizeof(wire::WhitewaterRemainder);
  return detail::Build<wire::WhitewaterFrame>(
      2 + (record_bytes + 511) / 512,
      limits.cache,
      "FLWW",
      [&](auto& builder) -> absl::StatusOr<flatbuffers::Offset<wire::WhitewaterFrame>> {
        const auto point_data = builder.CreateVectorOfStructs(points);
        const auto remainder_data = builder.CreateVectorOfStructs(remainders);
        const auto clock = detail::Pack(frame.clock);
        return wire::CreateWhitewaterFrame(
            builder,
            1,
            0,
            &clock,
            frame.emission.next_point_id,
            point_data,
            remainder_data
        );
      }
  );
}

absl::StatusOr<WhitewaterFrame3D>
DecodeWhitewaterFrame3D(std::span<const std::uint8_t> bytes, WhitewaterCacheLimits3D limits) {
  const auto verified = detail::Verify<wire::WhitewaterFrame>(bytes, "FLWW", limits.cache);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* source = *verified;
  const auto counts = CheckCounts(source->points()->size(), source->remainders()->size(), limits);
  if (!counts.ok()) {
    return counts;
  }
  WhitewaterFrame3D frame{.clock = detail::Unpack(source->clock())};
  frame.emission.next_point_id = source->next_point_id();
  frame.emission.last_accepted_step_ordinal = frame.clock.accepted_step_ordinal;
  frame.points.reserve(source->points()->size());
  for (const auto* point : *source->points()) {
    frame.points.push_back({
        .id = point->id(),
        .point =
            {
                .kinematics =
                    {.position = detail::Unpack(&point->position()), .velocity = detail::Unpack(&point->velocity())},
                .kind = static_cast<WhitewaterKind3D>(point->kind()),
                .birth_time_seconds = point->birth_time_seconds(),
            },
    });
  }
  frame.emission.remainders.reserve(source->remainders()->size());
  for (const auto* remainder : *source->remainders()) {
    const auto& cell = remainder->source_cell();
    frame.emission.remainders.push_back({
        .source_cell = {cell.x(), cell.y(), cell.z()},
        .fractional_count = remainder->fractional_count(),
    });
  }
  const auto checked = CheckFrame(frame);
  if (!checked.ok()) {
    return absl::DataLossError(checked.message());
  }
  return frame;
}

absl::StatusOr<LiquidCacheBytes3D>
EncodeWhitewaterSettings3D(const WhitewaterFrameOptions3D& options, LiquidCacheLimits3D limits) {
  if (!ValidSettings(options)) {
    return absl::InvalidArgumentError("invalid whitewater physical settings");
  }
  const auto& transport = options.transport;
  return detail::Build<wire::WhitewaterSettings>(
      2,
      limits,
      "FLWC",
      [&](auto& builder) -> absl::StatusOr<flatbuffers::Offset<wire::WhitewaterSettings>> {
        const auto ambient = detail::PackCanonical(transport.ambient_velocity);
        return wire::CreateWhitewaterSettings(
            builder,
            1,
            0,
            1,
            options.emission_rate_per_m3_second,
            options.represented_count_per_point,
            transport.classification_radius_m,
            detail::Canonical(transport.spray_drag_per_second),
            detail::Canonical(transport.bubble_drag_per_second),
            detail::Canonical(transport.bubble_buoyancy_scale),
            transport.foam_lifetime_seconds,
            &ambient
        );
      }
  );
}

absl::StatusOr<WhitewaterFrameOptions3D>
DecodeWhitewaterSettings3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  const auto verified = detail::Verify<wire::WhitewaterSettings>(bytes, "FLWC", limits);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* source = *verified;
  if (source->algorithm_version() != 1) {
    return absl::FailedPreconditionError("unsupported whitewater reconstruction algorithm");
  }
  WhitewaterFrameOptions3D options;
  options.emission_rate_per_m3_second = source->emission_rate_per_m3_second();
  options.represented_count_per_point = source->represented_count_per_point();
  options.transport.classification_radius_m = source->classification_radius_m();
  options.transport.spray_drag_per_second = source->spray_drag_per_second();
  options.transport.bubble_drag_per_second = source->bubble_drag_per_second();
  options.transport.bubble_buoyancy_scale = source->bubble_buoyancy_scale();
  options.transport.foam_lifetime_seconds = source->foam_lifetime_seconds();
  options.transport.ambient_velocity = detail::Unpack(source->ambient_velocity());
  if (!ValidSettings(options)) {
    return absl::DataLossError("invalid whitewater physical settings");
  }
  return options;
}
}  // namespace fluidity
