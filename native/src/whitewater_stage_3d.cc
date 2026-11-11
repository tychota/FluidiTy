#include "fluidity/host/whitewater_stage_3d.h"

#include <algorithm>
#include <limits>

namespace fluidity {
namespace {
absl::StatusOr<EffectsSource3D> OpenBirthSource(const std::filesystem::path& simulate_root) {
  auto source = EffectsSource3D::Open(simulate_root);
  if (!source.ok()) {
    return source.status();
  }
  if (!source->scene().options.whitewater_capture) {
    return absl::FailedPreconditionError("Simulate source has no resolved whitewater birth capture");
  }
  return source;
}

WhitewaterCacheLimits3D CacheLimits(const WhitewaterFrameOptions3D& options) {
  return {
      .maximum_points = options.maximum_points,
      .maximum_source_cells = options.maximum_source_cells,
  };
}
}  // namespace

WhitewaterStage3D::WhitewaterStage3D(
    EffectsSource3D source,
    EffectsCache3D cache,
    WhitewaterFrameOptions3D options,
    bool read_only
)
    : source_(std::move(source)), cache_(std::move(cache)), options_(options), read_only_(read_only) {
}

absl::StatusOr<WhitewaterStage3D> WhitewaterStage3D::OpenForPublication(
    const std::filesystem::path& simulate_root,
    const std::filesystem::path& output_root,
    LiquidDigest3D build,
    const WhitewaterFrameOptions3D& options
) {
  const auto root = ValidateEffectsOutputRoot3D(simulate_root, output_root, EffectsStage3D::kWhitewater);
  if (!root.ok()) {
    return root;
  }
  const auto settings = EncodeWhitewaterSettings3D(options);
  if (!settings.ok()) {
    return settings.status();
  }
  auto source = OpenBirthSource(simulate_root);
  if (!source.ok()) {
    return source.status();
  }
  const EffectsIdentity3D identity{
      .stage = EffectsStage3D::kWhitewater,
      .upstream_execution = source->execution_id(),
      .settings = LiquidContentDigest3D(settings->values()),
      .build = build,
  };
  auto cache = EffectsCache3D::Create(output_root, identity);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto published = cache->PublishStageSettings(settings->values());
  if (!published.ok()) {
    return published;
  }
  return WhitewaterStage3D(std::move(*source), std::move(*cache), options, false);
}

absl::StatusOr<WhitewaterStage3D> WhitewaterStage3D::OpenForReplay(
    const std::filesystem::path& simulate_root,
    const std::filesystem::path& output_root,
    LiquidDigest3D build
) {
  const auto root = ValidateEffectsOutputRoot3D(simulate_root, output_root, EffectsStage3D::kWhitewater);
  if (!root.ok()) {
    return root;
  }
  auto source = OpenBirthSource(simulate_root);
  if (!source.ok()) {
    return source.status();
  }
  auto cache = EffectsCache3D::OpenForInspection(output_root);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto& identity = cache->identity();
  if (identity.stage != EffectsStage3D::kWhitewater || identity.upstream_execution != source->execution_id()
      || identity.build != build) {
    return absl::FailedPreconditionError("Whitewater identity differs from source or executable");
  }
  const auto settings = cache->ReadStageSettings();
  if (!settings.ok()) {
    return settings.status();
  }
  const auto options = DecodeWhitewaterSettings3D(settings->values());
  if (!options.ok()) {
    return options.status();
  }
  return WhitewaterStage3D(std::move(*source), std::move(*cache), *options, true);
}

std::uint64_t WhitewaterStage3D::next_frame() const {
  return next_frame_;
}
const EffectsIdentity3D& WhitewaterStage3D::identity() const {
  return cache_.identity();
}
LiquidBackend3D WhitewaterStage3D::source_backend() const {
  return source_.scene().backend;
}

absl::StatusOr<std::uint64_t>
WhitewaterStage3D::RestoreCommittedPrefix(std::uint64_t maximum_frame, absl::FunctionRef<absl::Status()> check_stop) {
  if (read_only_ || next_frame_ != 0) {
    return absl::FailedPreconditionError("Whitewater restore requires a fresh publication owner");
  }
  if (maximum_frame == std::numeric_limits<std::uint64_t>::max()) {
    return absl::OutOfRangeError("Whitewater restore frame bound exhausts cursor");
  }
  std::optional<EffectsCommit3D> checkpoint_commit;
  std::uint64_t restored_count = 0;
  for (; restored_count <= maximum_frame; ++restored_count) {
    const auto stopped = check_stop();
    if (!stopped.ok()) {
      return stopped;
    }
    auto committed = cache_.ReadFrameIfCommitted(restored_count);
    if (!committed.ok()) {
      return committed.status();
    }
    if (!committed->has_value()) {
      break;
    }
    const auto accepted = source_.ReadFrame(restored_count);
    if (!accepted.ok()) {
      return absl::IsNotFound(accepted.status()) ? absl::DataLossError("committed Whitewater source is missing")
                                                 : accepted.status();
    }
    const auto& frame = **committed;
    const auto expected_previous = checkpoint_commit ? checkpoint_commit->payload_digest : LiquidDigest3D{};
    if (frame.clock != accepted->motion.last_clock || frame.upstream_content != accepted->source_payload
        || frame.previous_payload != expected_previous) {
      return absl::DataLossError("Whitewater checkpoint source or predecessor differs");
    }
    checkpoint_commit = std::move(**committed);
  }
  if (!checkpoint_commit) {
    return 0;
  }
  auto checkpoint = DecodeWhitewaterFrame3D(checkpoint_commit->payload.values(), CacheLimits(options_));
  if (!checkpoint.ok()) {
    return checkpoint.status();
  }
  if (checkpoint->clock != checkpoint_commit->clock) {
    return absl::DataLossError("Whitewater payload clock differs from its manifest");
  }
  previous_frame_ = std::move(*checkpoint);
  previous_payload_ = checkpoint_commit->payload_digest;
  next_frame_ = restored_count;
  return restored_count;
}

absl::StatusOr<WhitewaterStageFrame3D> WhitewaterStage3D::ProcessNextFrame(LiquidCachePublishControl3D control) {
  if (next_frame_ == std::numeric_limits<std::uint64_t>::max()) {
    return absl::OutOfRangeError("Whitewater frame cursor exhausted");
  }
  const auto accepted = source_.ReadFrame(next_frame_);
  if (!accepted.ok()) {
    return accepted.status();
  }
  const auto scene = source_.scene();
  auto reconstructed = ReconstructWhitewaterFrame3D(
      accepted->motion,
      scene.scene,
      scene.options,
      previous_frame_ ? &*previous_frame_ : nullptr,
      options_
  );
  if (!reconstructed.ok()) {
    return reconstructed.status();
  }
  auto& frame = reconstructed->frame;
  const auto payload = EncodeWhitewaterFrame3D(frame, CacheLimits(options_));
  if (!payload.ok()) {
    return payload.status();
  }
  const auto payload_digest = LiquidContentDigest3D(payload->values());
  if (read_only_) {
    const auto committed = cache_.ReadFrame(next_frame_);
    if (!committed.ok()) {
      return committed.status();
    }
    if (committed->clock != frame.clock || committed->upstream_content != accepted->source_payload
        || committed->previous_payload != previous_payload_ || committed->payload_digest != payload_digest
        || !std::ranges::equal(committed->payload.values(), payload->values())) {
      return absl::DataLossError("Whitewater replay differs from committed source, lineage or points");
    }
  } else {
    const auto published =
        cache_.Publish(frame.clock, accepted->source_payload, previous_payload_, payload->values(), control);
    if (!published.ok()) {
      return published;
    }
  }
  WhitewaterStageFrame3D result{
      .clock = frame.clock,
      .payload_digest = payload_digest,
      .birth_count = reconstructed->birth_count,
      .expired_count = reconstructed->expired_count,
      .bubble_fallback_substeps = reconstructed->bubble_fallback_substeps,
  };
  for (const auto& record : frame.points) {
    result.foam_count += record.point.kind == WhitewaterKind3D::kFoam;
    result.spray_count += record.point.kind == WhitewaterKind3D::kSpray;
    result.bubble_count += record.point.kind == WhitewaterKind3D::kBubble;
  }
  previous_frame_ = std::move(frame);
  previous_payload_ = payload_digest;
  ++next_frame_;
  return result;
}
}  // namespace fluidity
