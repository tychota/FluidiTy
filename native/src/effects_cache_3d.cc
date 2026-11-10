#include "fluidity/host/effects_cache_3d.h"

#include <utility>

#include "liquid_cache_3d_internal.h"
#include "liquid_cache_files_3d_internal.h"
#include "stage_manifest_generated.h"

namespace fluidity {
namespace {
namespace detail = liquid_cache_internal;
namespace files = liquid_cache_files_internal;
namespace wire = effects::v1;
#define EFFECTS_TRY(expression) \
  do { \
    const auto status = (expression); \
    if (!status.ok()) { \
      return status; \
    } \
  } while (false)

bool ValidIdentity(const EffectsIdentity3D& identity) {
  return identity.stage <= EffectsStage3D::kVolumes && identity.upstream_execution != LiquidDigest3D{}
  && identity.settings != LiquidDigest3D{} && identity.build != LiquidDigest3D{};
}

flatbuffers::Offset<wire::StageIdentity>
PackIdentity(flatbuffers::FlatBufferBuilder& builder, const EffectsIdentity3D& identity) {
  const auto upstream = detail::Pack(identity.upstream_execution);
  const auto settings = detail::Pack(identity.settings);
  const auto build = detail::Pack(identity.build);
  return wire::CreateStageIdentity(
      builder,
      1,
      0,
      static_cast<wire::Stage>(identity.stage),
      &upstream,
      &settings,
      &build
  );
}

absl::StatusOr<EffectsIdentity3D> ReadIdentity(const wire::StageIdentity* identity) {
  EffectsIdentity3D result{
      .stage = static_cast<EffectsStage3D>(identity->stage()),
      .upstream_execution = detail::Unpack(identity->upstream_execution()),
      .settings = detail::Unpack(identity->settings()),
      .build = detail::Unpack(identity->build()),
  };
  if (identity->schema_version() != 1 || !ValidIdentity(result)) {
    return absl::DataLossError("invalid effects identity");
  }
  return result;
}

absl::StatusOr<EffectsIdentity3D> ReadRootIdentity(const std::filesystem::path& root, LiquidCacheLimits3D limits) {
  const auto bytes = files::ReadBytes(root / "identity.flei", limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto identity = detail::Verify<wire::StageIdentity>(bytes->values(), "FLEI", limits);
  if (!identity.ok()) {
    return identity.status();
  }
  return ReadIdentity(*identity);
}

std::filesystem::path PayloadPath(const std::filesystem::path& root, const LiquidDigest3D& digest) {
  return root / "blobs" / (std::string(LiquidDigestHex3D(digest).data()) + ".flef");
}

std::string Mirror(const wire::StageManifest& manifest) {
  const auto payload = detail::Unpack(manifest.payload_digest());
  const auto upstream = detail::Unpack(manifest.upstream_content());
  const auto previous = detail::Unpack(manifest.previous_payload());
  return "{\"schema_version\":1,\"stage\":" + std::to_string(static_cast<int>(manifest.identity()->stage()))
      + ",\"frame\":" + std::to_string(manifest.clock()->frame()) + ",\"upstream_content\":\""
      + LiquidDigestHex3D(upstream).data() + "\",\"previous_payload\":\"" + LiquidDigestHex3D(previous).data()
      + "\",\"payload_digest\":\"" + LiquidDigestHex3D(payload).data()
      + "\",\"payload_byte_length\":" + std::to_string(manifest.payload_byte_length()) + "}\n";
}

absl::StatusOr<LiquidCacheBytes3D> ReadManifest(
    const std::filesystem::path& root,
    const EffectsIdentity3D& expected,
    std::uint64_t frame,
    LiquidCacheLimits3D limits
) {
  auto bytes = files::ReadBytes(files::FramePath(root, frame, ".flem"), limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto manifest = detail::Verify<wire::StageManifest>(bytes->values(), "FLEM", limits);
  if (!manifest.ok()) {
    return manifest.status();
  }
  const auto identity = ReadIdentity((*manifest)->identity());
  const auto clock = detail::Unpack((*manifest)->clock());
  if (!identity.ok() || *identity != expected || !detail::ValidClock(clock) || clock.substep_ordinal != 0
      || clock.frame_ordinal != frame || (*manifest)->payload_byte_length() == 0
      || detail::Unpack((*manifest)->payload_digest()) == LiquidDigest3D{}
      || detail::Unpack((*manifest)->upstream_content()) == LiquidDigest3D{}) {
    return absl::DataLossError("invalid effects manifest");
  }
  if (frame == 0
      && (clock.time != 0 || clock.accepted_step_ordinal != 0 || clock.epoch != 0
          || detail::Unpack((*manifest)->previous_payload()) != LiquidDigest3D{})) {
    return absl::DataLossError("invalid effects seed clock");
  }
  EFFECTS_TRY(files::Matches(
      files::FramePath(root, frame, ".json"),
      files::Bytes(Mirror(**manifest)),
      limits,
      absl::StatusCode::kDataLoss
  ));
  return std::move(*bytes);
}

absl::Status CheckPredecessor(
    const std::filesystem::path& root,
    const EffectsIdentity3D& identity,
    LiquidClock3D clock,
    LiquidDigest3D previous,
    LiquidCacheLimits3D limits
) {
  if (clock.frame_ordinal == 0) {
    return previous == LiquidDigest3D{} ? absl::OkStatus() : absl::FailedPreconditionError("seed has a predecessor");
  }
  const auto bytes = ReadManifest(root, identity, clock.frame_ordinal - 1, limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto* manifest = flatbuffers::GetRoot<wire::StageManifest>(bytes->values().data());
  if (previous != detail::Unpack(manifest->payload_digest()) || clock.time <= manifest->clock()->time()
      || clock.epoch <= manifest->clock()->epoch()) {
    return absl::FailedPreconditionError("effects predecessor does not match");
  }
  const auto payload = files::ReadBytes(PayloadPath(root, previous), limits, manifest->payload_byte_length());
  if (!payload.ok()) {
    return payload.status();
  }
  if (LiquidContentDigest3D(payload->values()) != previous) {
    return absl::DataLossError("effects predecessor payload is corrupt");
  }
  return absl::OkStatus();
}
}  // namespace

EffectsCache3D::EffectsCache3D(
    std::filesystem::path root,
    EffectsIdentity3D identity,
    LiquidCacheLimits3D limits,
    bool read_only
)
    : root_(std::move(root)), identity_(identity), limits_(limits), read_only_(read_only) {
}

absl::StatusOr<EffectsCache3D>
EffectsCache3D::Create(const std::filesystem::path& root, EffectsIdentity3D identity, LiquidCacheLimits3D limits) {
  if (!ValidIdentity(identity)) {
    return absl::InvalidArgumentError("invalid effects identity");
  }
  std::error_code error;
  const auto absolute = std::filesystem::absolute(root, error);
  if (error) {
    return absl::UnavailableError("could not resolve effects root");
  }
  files::NativeFile lease;
  EFFECTS_TRY(files::Lease(absolute, lease, true));
  const auto bytes = detail::Build<wire::StageIdentity>(1, limits, "FLEI", [&](auto& builder) {
    return absl::StatusOr<flatbuffers::Offset<wire::StageIdentity>>(PackIdentity(builder, identity));
  });
  if (!bytes.ok()) {
    return bytes.status();
  }
  EFFECTS_TRY(files::PublishBytes(
      absolute / "identity.flei",
      bytes->values(),
      limits,
      {},
      absl::StatusCode::kFailedPrecondition
  ));
  return EffectsCache3D(absolute, identity, limits, false);
}

absl::StatusOr<EffectsCache3D>
EffectsCache3D::OpenForInspection(const std::filesystem::path& root, LiquidCacheLimits3D limits) {
  std::error_code error;
  const auto absolute = std::filesystem::absolute(root, error);
  if (error) {
    return absl::UnavailableError("could not resolve effects root");
  }
  files::NativeFile lease;
  EFFECTS_TRY(files::Lease(absolute, lease, false, true));
  auto identity = ReadRootIdentity(absolute, limits);
  if (!identity.ok()) {
    return identity.status();
  }
  return EffectsCache3D(absolute, *identity, limits, true);
}

const EffectsIdentity3D& EffectsCache3D::identity() const {
  return identity_;
}

absl::Status EffectsCache3D::Publish(
    LiquidClock3D clock,
    LiquidDigest3D upstream_content,
    LiquidDigest3D previous_payload,
    std::span<const std::uint8_t> payload,
    LiquidCachePublishControl3D control
) const {
  if (read_only_) {
    return absl::FailedPreconditionError("effects cache is read-only");
  }
  if (!detail::ValidClock(clock) || clock.substep_ordinal != 0 || upstream_content == LiquidDigest3D{}
      || payload.empty()
      || (clock.frame_ordinal == 0 && (clock.time != 0 || clock.accepted_step_ordinal != 0 || clock.epoch != 0))) {
    return absl::InvalidArgumentError("invalid effects frame");
  }
  if (payload.size() > limits_.maximum_bytes) {
    return absl::ResourceExhaustedError("effects payload exceeds byte limit");
  }
  files::NativeFile lease;
  EFFECTS_TRY(files::Lease(root_, lease, false, true));
  const auto current_identity = ReadRootIdentity(root_, limits_);
  if (!current_identity.ok()) {
    return current_identity.status();
  }
  if (*current_identity != identity_) {
    return absl::DataLossError("effects root identity changed");
  }
  EFFECTS_TRY(CheckPredecessor(root_, identity_, clock, previous_payload, limits_));
  const auto payload_digest = LiquidContentDigest3D(payload);
  auto bytes = detail::Build<wire::StageManifest>(2, limits_, "FLEM", [&](auto& builder) {
    const auto identity = PackIdentity(builder, identity_);
    const auto wire_clock = detail::Pack(clock);
    const auto upstream = detail::Pack(upstream_content);
    const auto previous = detail::Pack(previous_payload);
    const auto digest = detail::Pack(payload_digest);
    return absl::StatusOr<flatbuffers::Offset<wire::StageManifest>>(
        wire::CreateStageManifest(builder, 1, 0, identity, &wire_clock, &upstream, &previous, &digest, payload.size())
    );
  });
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto* manifest = flatbuffers::GetRoot<wire::StageManifest>(bytes->values().data());
  const auto destination = files::FramePath(root_, clock.frame_ordinal, ".flem");
  const auto existing = files::Matches(destination, bytes->values(), limits_, absl::StatusCode::kAlreadyExists);
  if (!existing.ok() && existing.code() != absl::StatusCode::kNotFound) {
    return existing;
  }
  EFFECTS_TRY(files::PublishBytes(PayloadPath(root_, payload_digest), payload, limits_, control));
  EFFECTS_TRY(files::Check(control, LiquidCachePublishStage3D::kContentPublished));
  const auto mirror = Mirror(*manifest);
  EFFECTS_TRY(files::PublishBytes(
      files::FramePath(root_, clock.frame_ordinal, ".json"),
      files::Bytes(mirror),
      limits_,
      control,
      absl::StatusCode::kAlreadyExists
  ));
  EFFECTS_TRY(files::Check(control, LiquidCachePublishStage3D::kBeforeManifest));
  EFFECTS_TRY(files::PublishBytes(destination, bytes->values(), limits_, control, absl::StatusCode::kAlreadyExists));
  return files::Check(control, LiquidCachePublishStage3D::kManifestPublished);
}

absl::StatusOr<EffectsCommit3D> EffectsCache3D::ReadFrame(std::uint64_t frame) const {
  files::NativeFile lease;
  EFFECTS_TRY(files::Lease(root_, lease, false, true));
  const auto current_identity = ReadRootIdentity(root_, limits_);
  if (!current_identity.ok()) {
    return current_identity.status();
  }
  if (*current_identity != identity_) {
    return absl::DataLossError("effects root identity changed");
  }
  const auto bytes = ReadManifest(root_, identity_, frame, limits_);
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto* manifest = flatbuffers::GetRoot<wire::StageManifest>(bytes->values().data());
  const auto clock = detail::Unpack(manifest->clock());
  const auto previous = detail::Unpack(manifest->previous_payload());
  const auto predecessor = CheckPredecessor(root_, identity_, clock, previous, limits_);
  if (!predecessor.ok()) {
    return absl::DataLossError("effects predecessor is invalid");
  }
  const auto digest = detail::Unpack(manifest->payload_digest());
  auto payload = files::ReadBytes(PayloadPath(root_, digest), limits_, manifest->payload_byte_length());
  if (!payload.ok()) {
    return payload.status();
  }
  if (LiquidContentDigest3D(payload->values()) != digest) {
    return absl::DataLossError("effects payload is corrupt");
  }
  return EffectsCommit3D{
      .identity = identity_,
      .clock = clock,
      .upstream_content = detail::Unpack(manifest->upstream_content()),
      .previous_payload = previous,
      .payload_digest = digest,
      .payload = std::move(*payload),
  };
}
#undef EFFECTS_TRY
}  // namespace fluidity
