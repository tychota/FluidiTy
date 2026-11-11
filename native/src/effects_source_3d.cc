#include "fluidity/host/effects_source_3d.h"

#include <algorithm>
#include <utility>

namespace fluidity {
namespace {
bool SameGeometry(UniformMacGeometry3D a, UniformMacGeometry3D b) {
  return a.cell_width == b.cell_width && a.world_origin.x == b.world_origin.x && a.world_origin.y == b.world_origin.y
      && a.world_origin.z == b.world_origin.z;
}
// Compare canonical producer bytes, not a selected attribute subset. This
// includes the equivalent-volume radius and completed clock in the join.
absl::Status
MatchesParent(const LiquidState3D& state, const LiquidCommittedFrame3D& parent, LiquidCacheLimits3D limits) {
  const auto points = BuildLiquidPointFrame3D(state, limits);
  if (!points.ok()) {
    return absl::IsResourceExhausted(points.status())
        ? points.status()
        : absl::DataLossError("source endpoint cannot produce its declared parent points");
  }
  const auto bytes = EncodeLiquidPointFrame3D(*points, limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  return VerifyLiquidContent3D(bytes->values(), parent.manifest.points, LiquidContentKind3D::kPoints, limits);
}
}  // namespace

EffectsSource3D::EffectsSource3D(
    LiquidCache3D parent,
    EffectsCache3D source,
    LiquidResolvedScene3D scene,
    LiquidDigest3D execution,
    LiquidCacheLimits3D limits
)
    : parent_(std::move(parent)),
      source_(std::move(source)),
      scene_(std::move(scene)),
      execution_(execution),
      limits_(limits) {
}

absl::StatusOr<EffectsSource3D> EffectsSource3D::Open(const std::filesystem::path& root, LiquidCacheLimits3D limits) {
  auto parent = LiquidCache3D::OpenForInspection(root, limits);
  if (!parent.ok()) {
    return parent.status();
  }
  auto scene = parent->ReadResolvedScene();
  if (!scene.ok()) {
    return scene.status();
  }
  if (scene->options.motion_sample_intervals == 0) {
    return absl::FailedPreconditionError("Simulate cache has no accepted effects capture");
  }
  auto fingerprint = ResolvedSceneFingerprint(scene->View(), limits);
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  auto source = EffectsCache3D::OpenForInspection(root / "effects" / "source", limits);
  if (!source.ok()) {
    return source.status();
  }
  const EffectsIdentity3D expected{
      .stage = EffectsStage3D::kSource,
      .upstream_execution = LiquidExecutionId3D(*fingerprint),
      .settings = *fingerprint,
      .build = scene->build_digest,
  };
  if (source->identity() != expected) {
    return absl::FailedPreconditionError("accepted source identity does not match Simulate");
  }
  auto execution = EffectsExecutionId3D(expected);
  if (!execution.ok()) {
    return execution.status();
  }
  return EffectsSource3D(std::move(*parent), std::move(*source), std::move(*scene), *execution, limits);
}

LiquidResolvedSceneView3D EffectsSource3D::scene() const {
  return scene_.View();
}
const EffectsIdentity3D& EffectsSource3D::identity() const {
  return source_.identity();
}
LiquidDigest3D EffectsSource3D::execution_id() const {
  return execution_;
}

absl::StatusOr<EffectsSourceFrame3D> EffectsSource3D::ReadFrame(std::uint64_t frame) const {
  const auto parent = parent_.ReadCommittedFrame(frame);
  if (!parent.ok()) {
    return parent.status();
  }
  auto source = source_.ReadFrame(frame);
  if (!source.ok()) {
    return source.status();
  }
  if (source->clock != parent->points.clock || source->upstream_content != parent->manifest.points.digest) {
    return absl::DataLossError("source manifest differs from its committed parent");
  }
  auto decoded = DecodeMotionFrame3D(source->payload.values(), limits_);
  if (!decoded.ok()) {
    return decoded.status();
  }
  if (decoded->last_clock != source->clock) {
    return absl::DataLossError("source payload clock differs from its manifest");
  }
  for (const auto& sample : decoded->samples) {
    if (!SameGeometry(sample.geometry, scene_.geometry) || sample.random_seed != scene_.random_seed) {
      return absl::DataLossError("source payload geometry or seed differs from resolved Simulate");
    }
  }
  const auto endpoint = MatchesParent(decoded->samples.back(), *parent, limits_);
  if (!endpoint.ok()) {
    return endpoint;
  }
  if (frame > 0) {
    const auto previous_parent = parent_.ReadCommittedFrame(frame - 1);
    if (!previous_parent.ok()) {
      return previous_parent.status();
    }
    const auto first = MatchesParent(decoded->samples.front(), *previous_parent, limits_);
    if (!first.ok()) {
      return first;
    }
    const auto previous = source_.ReadFrame(frame - 1);
    if (!previous.ok()) {
      return previous.status();
    }
    if (previous->payload_digest != source->previous_payload) {
      return absl::DataLossError("source predecessor payload changed");
    }
    const auto previous_decoded = DecodeMotionFrame3D(previous->payload.values(), limits_);
    if (!previous_decoded.ok()) {
      return previous_decoded.status();
    }
    const auto previous_bytes = EncodeLiquidState3D(previous_decoded->samples.back(), limits_);
    const auto current_bytes = EncodeLiquidState3D(decoded->samples.front(), limits_);
    if (!previous_bytes.ok()) {
      return previous_bytes.status();
    }
    if (!current_bytes.ok()) {
      return current_bytes.status();
    }
    if (!std::ranges::equal(previous_bytes->values(), current_bytes->values())) {
      return absl::DataLossError("accepted source interval is discontinuous beyond point attributes");
    }
  }
  return EffectsSourceFrame3D{
      .motion = std::move(*decoded),
      .source_payload = source->payload_digest,
      .parent_points = parent->manifest.points.digest,
  };
}
}  // namespace fluidity
