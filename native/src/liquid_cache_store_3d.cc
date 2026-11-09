#include <algorithm>
#include <charconv>
#include <cstring>
#include <string>
#include <system_error>

#include "checkpoint_reference_generated.h"
#include "fluidity/host/liquid_cache_3d.h"
#include "liquid_cache_3d_internal.h"
#include "liquid_cache_files_3d_internal.h"

namespace fluidity {
namespace {
namespace fs = std::filesystem;
namespace detail = liquid_cache_internal;
namespace wire = cache::v1;
#define STORE_TRY(expression) \
  do { \
    const auto store_status = (expression); \
    if (!store_status.ok()) { \
      return store_status; \
    } \
  } while (false)

using liquid_cache_files_internal::Bytes;
using liquid_cache_files_internal::Check;
using liquid_cache_files_internal::ContentPath;
using liquid_cache_files_internal::Failure;
using liquid_cache_files_internal::FlushDirectory;
using liquid_cache_files_internal::FramePath;
using liquid_cache_files_internal::Lease;
using liquid_cache_files_internal::Matches;
using liquid_cache_files_internal::NativeFile;
using liquid_cache_files_internal::OrdinalName;
using liquid_cache_files_internal::PublishBytes;
using liquid_cache_files_internal::PublishContent;
using liquid_cache_files_internal::ReadBytes;
using liquid_cache_files_internal::ReadContent;
using liquid_cache_files_internal::Reference;
// Capture caller-relative directories once; parsed content names remain relative
// to this fixed root even if another part of the application later changes cwd.
absl::StatusOr<fs::path> AbsoluteDirectory(const fs::path& requested_directory) {
  if (requested_directory.empty()
      || requested_directory.native().find(fs::path::value_type{}) != fs::path::string_type::npos) {
    return Failure(absl::StatusCode::kInvalidArgument);
  }
  std::error_code error;
  auto directory = fs::absolute(requested_directory, error);
  if (error) {
    return Failure(absl::StatusCode::kUnavailable);
  }
  return directory;
}
std::string Number(double value) {
  std::array<char, 64> buffer{};
  const auto result = std::to_chars(
      buffer.data(),
      buffer.data() + buffer.size(),
      value,
      std::chars_format::general,
      std::numeric_limits<double>::max_digits10
  );
  return std::string(buffer.data(), result.ptr);
}
// Inspection is derived from the same validated values as the binary manifest,
// has no local directory/process data, and is checked byte-for-byte on reads.
std::string Inspection(const LiquidFrameManifest3D& manifest) {
  const auto name = LiquidContentName3D(manifest.points);
  return "{\"schema_version\":1,\"status\":\"complete\",\"clock\":{\"time_seconds\":" + Number(manifest.clock.time)
      + ",\"frame\":" + std::to_string(manifest.clock.frame_ordinal)
      + ",\"substep\":" + std::to_string(manifest.clock.substep_ordinal) + ",\"accepted_step\":"
      + std::to_string(manifest.clock.accepted_step_ordinal) + ",\"epoch\":" + std::to_string(manifest.clock.epoch)
      + "},\"resolved_fingerprint\":\"" + LiquidDigestHex3D(manifest.resolved_fingerprint).data()
      + "\",\"execution_id\":\"" + LiquidDigestHex3D(manifest.execution_id).data()
      + "\",\"points\":{\"kind\":\"points\",\"byte_length\":" + std::to_string(manifest.points.byte_length)
      + ",\"digest\":\"" + LiquidDigestHex3D(manifest.points.digest).data() + "\",\"relative_name\":\"" + name->data()
      + "\"}}\n";
}
absl::Status Job(const LiquidDigest3D& fingerprint, const LiquidDigest3D& execution, const LiquidDigest3D& expected) {
  if (fingerprint != expected || execution != LiquidExecutionId3D(expected)) {
    return Failure(absl::StatusCode::kFailedPrecondition);
  }
  return absl::OkStatus();
}
absl::StatusOr<LiquidResolvedScene3D> ReadResolved(
    const fs::path& root,
    const LiquidDigest3D& expected,
    LiquidCacheLimits3D limits,
    const LiquidContentReference3D* reference = nullptr
) {
  auto bytes = ReadBytes(root / "resolved.flrs", limits, reference == nullptr ? 0 : reference->byte_length);
  if (!bytes.ok()) {
    return bytes.status();
  }
  if (reference != nullptr) {
    STORE_TRY(VerifyLiquidContent3D(bytes->values(), *reference, LiquidContentKind3D::kResolvedScene, limits));
  }
  auto scene = DecodeLiquidResolvedScene3D(bytes->values(), limits);
  if (!scene.ok()) {
    return scene.status();
  }
  auto fingerprint = ResolvedSceneFingerprint(scene->View(), limits);
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  STORE_TRY(Job(*fingerprint, LiquidExecutionId3D(*fingerprint), expected));
  auto canonical = EncodeLiquidResolvedScene3D(scene->View(), limits);
  if (!canonical.ok()) {
    return canonical.status();
  }
  if (!std::ranges::equal(bytes->values(), canonical->values())) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  return std::move(*scene);
}
absl::StatusOr<LiquidCommittedFrame3D>
ReadFrame(const fs::path& root, std::uint64_t ordinal, const LiquidDigest3D& fingerprint, LiquidCacheLimits3D limits) {
  auto bytes = ReadBytes(FramePath(root, ordinal), limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto manifest = DecodeLiquidFrameManifest3D(bytes->values(), limits);
  if (!manifest.ok()) {
    return manifest.status();
  }
  STORE_TRY(Job(manifest->resolved_fingerprint, manifest->execution_id, fingerprint));
  if (manifest->clock.frame_ordinal != ordinal) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  STORE_TRY(
      Matches(FramePath(root, ordinal, ".json"), Bytes(Inspection(*manifest)), limits, absl::StatusCode::kDataLoss)
  );
  auto content = ReadContent(root, manifest->points, LiquidContentKind3D::kPoints, limits);
  if (!content.ok()) {
    return content.status();
  }
  auto points = DecodeLiquidPointFrame3D(content->values(), limits);
  if (!points.ok()) {
    return points.status();
  }
  if (points->clock != manifest->clock) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  return LiquidCommittedFrame3D{*manifest, std::move(*points)};
}
bool SameVector(Vec3 left, Vec3 right) {
  return left.x == right.x && left.y == right.y && left.z == right.z;
}
absl::Status CheckpointMatches(
    const LiquidState3D& state,
    const LiquidResumeMetadata3D& resume,
    const LiquidPointFrame3D& points,
    const LiquidResolvedScene3D& scene,
    const LiquidDigest3D& fingerprint
) {
  STORE_TRY(Job(resume.resolved_fingerprint, resume.execution_id, fingerprint));
  if (state.clock != points.clock || state.clock.frame_ordinal % 10 != 0
      || !SameVector(state.geometry.world_origin, scene.geometry.world_origin)
      || state.geometry.cell_width != scene.geometry.cell_width || state.random_seed != scene.random_seed) {
    return Failure(absl::StatusCode::kInvalidArgument);
  }
  if (state.clock.frame_ordinal == 0 && state.clock != LiquidClock3D{}) {
    return Failure(absl::StatusCode::kInvalidArgument);
  }
  // Ordinary frames can export a subset. Require it to preserve physical order
  // and every shared field, without inventing a radius from the physical state.
  std::size_t particle_index = 0;
  for (const auto& point : points.points.values()) {
    while (particle_index < state.particles.size() && state.particles[particle_index].id != point.id) {
      ++particle_index;
    }
    if (particle_index == state.particles.size()) {
      return Failure(absl::StatusCode::kInvalidArgument);
    }
    const auto& particle = state.particles[particle_index++];
    if (point.tag != particle.tag || !SameVector(point.world_position, particle.world_position)
        || !SameVector(point.world_velocity, particle.world_velocity) || point.volume_ratio != particle.volume_ratio) {
      return Failure(absl::StatusCode::kInvalidArgument);
    }
  }
  return absl::OkStatus();
}
struct CheckpointReference {
  LiquidClock3D clock;
  LiquidDigest3D fingerprint;
  LiquidDigest3D execution;
  LiquidContentReference3D state;
  LiquidContentReference3D manifest;
};
absl::StatusOr<LiquidCacheBytes3D> EncodeReference(const CheckpointReference& reference, LiquidCacheLimits3D limits) {
  return detail::Build<wire::CheckpointReference>(
      2,
      limits,
      wire::CheckpointReferenceIdentifier(),
      [&](flatbuffers::FlatBufferBuilder& builder) -> absl::StatusOr<flatbuffers::Offset<wire::CheckpointReference>> {
        const auto pack = [&](const LiquidContentReference3D& content) {
          const auto name = LiquidContentName3D(content);
          const auto relative_name = builder.CreateString(name->data(), 75);
          const auto digest = detail::Pack(content.digest);
          return wire::CreateContentReference(
              builder,
              static_cast<wire::ContentKind>(content.kind),
              content.byte_length,
              &digest,
              relative_name
          );
        };
        const auto state = pack(reference.state);
        const auto manifest = pack(reference.manifest);
        const auto clock = detail::Pack(reference.clock);
        const auto fingerprint = detail::Pack(reference.fingerprint);
        const auto execution = detail::Pack(reference.execution);
        return wire::CreateCheckpointReference(builder, 1, 0, &clock, &fingerprint, &execution, state, manifest);
      }
  );
}
absl::StatusOr<CheckpointReference> DecodeReference(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  auto root = detail::Verify<wire::CheckpointReference>(bytes, wire::CheckpointReferenceIdentifier(), limits);
  if (!root.ok()) {
    return root.status();
  }
  const auto unpack = [&](const wire::ContentReference* content,
                          LiquidContentKind3D kind) -> absl::StatusOr<LiquidContentReference3D> {
    LiquidContentReference3D result{
        static_cast<LiquidContentKind3D>(content->kind()),
        content->byte_length(),
        detail::Unpack(content->digest()),
    };
    const auto* name = content->relative_name();
    if (result.kind != kind || result.byte_length < 8
        || !ValidateLiquidContentName3D({name->c_str(), name->size()}, result).ok()) {
      return Failure(absl::StatusCode::kDataLoss);
    }
    if (result.byte_length > limits.maximum_bytes) {
      return Failure(absl::StatusCode::kResourceExhausted);
    }
    return result;
  };
  auto state = unpack((*root)->checkpoint(), LiquidContentKind3D::kState);
  auto manifest = unpack((*root)->frame_manifest(), LiquidContentKind3D::kManifest);
  if (!state.ok()) {
    return state.status();
  }
  if (!manifest.ok()) {
    return manifest.status();
  }
  CheckpointReference result{
      detail::Unpack((*root)->clock()),
      detail::Unpack((*root)->resolved_fingerprint()),
      detail::Unpack((*root)->execution_id()),
      *state,
      *manifest,
  };
  if (!detail::ValidClock(result.clock) || result.clock.substep_ordinal != 0 || result.clock.frame_ordinal % 10 != 0
      || result.execution != LiquidExecutionId3D(result.fingerprint)) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  return result;
}
// Only canonical 20-digit inventory names are considered. Parsed text is never
// used as a path: callers derive the filename again from the checked ordinal.
absl::StatusOr<std::uint64_t> CheckpointOrdinal(const fs::path& path) {
  const auto name = path.filename().string();
  if (name.size() != 25 || name.substr(20) != ".flcr") {
    return Failure(absl::StatusCode::kNotFound);
  }
  std::uint64_t ordinal = 0;
  const auto result = std::from_chars(name.data(), name.data() + 20, ordinal);
  if (result.ec != std::errc{} || result.ptr != name.data() + 20 || ordinal % 10 != 0
      || name != OrdinalName(ordinal, ".flcr")) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  return ordinal;
}
absl::StatusOr<CheckpointReference> ReadReference(
    const fs::path& root,
    std::uint64_t ordinal,
    const LiquidDigest3D& fingerprint,
    LiquidCacheLimits3D limits
) {
  auto bytes = ReadBytes(root / "checkpoints" / OrdinalName(ordinal, ".flcr"), limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto reference = DecodeReference(bytes->values(), limits);
  if (!reference.ok()) {
    return reference.status();
  }
  STORE_TRY(Job(reference->fingerprint, reference->execution, fingerprint));
  if (reference->clock.frame_ordinal != ordinal) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  auto manifest = ReadBytes(FramePath(root, ordinal), limits, reference->manifest.byte_length);
  if (!manifest.ok()) {
    return manifest.status();
  }
  STORE_TRY(VerifyLiquidContent3D(manifest->values(), reference->manifest, LiquidContentKind3D::kManifest, limits));
  return *reference;
}
absl::StatusOr<LiquidCheckpoint3D> ReadCheckpointRecord(
    const fs::path& root,
    const CheckpointReference& reference,
    const LiquidResolvedScene3D& scene,
    const LiquidDigest3D& fingerprint,
    LiquidCacheLimits3D limits
) {
  auto frame = ReadFrame(root, reference.clock.frame_ordinal, fingerprint, limits);
  if (!frame.ok()) {
    return frame.status();
  }
  auto bytes = ReadContent(root, reference.state, LiquidContentKind3D::kState, limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto checkpoint = DecodeLiquidCheckpoint3D(bytes->values(), limits);
  if (!checkpoint.ok()) {
    return checkpoint.status();
  }
  const auto valid = CheckpointMatches(checkpoint->state, checkpoint->resume, frame->points, scene, fingerprint);
  if (!valid.ok() || checkpoint->state.clock != reference.clock) {
    return Failure(absl::StatusCode::kDataLoss);
  }
  return std::move(*checkpoint);
}
template <typename Visit>
absl::Status VisitReferences(const fs::path& root, Visit visit) {
  std::error_code error;
  fs::directory_iterator iterator(root / "checkpoints", error);
  if (error) {
    return Failure(absl::StatusCode::kUnavailable);
  }
  const fs::directory_iterator end;
  for (; iterator != end; iterator.increment(error)) {
    if (error) {
      return Failure(absl::StatusCode::kUnavailable);
    }
    auto ordinal = CheckpointOrdinal(iterator->path());
    if (ordinal.status().code() == absl::StatusCode::kNotFound) {
      continue;
    }
    if (!ordinal.ok()) {
      return ordinal.status();
    }
    STORE_TRY(visit(*ordinal));
  }
  return error ? Failure(absl::StatusCode::kUnavailable) : absl::OkStatus();
}
struct RetainedCheckpoints {
  std::optional<std::uint64_t> newest;
  std::optional<std::uint64_t> second;
};
absl::StatusOr<RetainedCheckpoints>
Retained(const fs::path& root, const LiquidDigest3D& fingerprint, LiquidCacheLimits3D limits) {
  RetainedCheckpoints retained;
  STORE_TRY(VisitReferences(root, [&](std::uint64_t ordinal) {
    auto reference = ReadReference(root, ordinal, fingerprint, limits);
    if (!reference.ok()) {
      return reference.status();
    }
    if (!retained.newest || ordinal > *retained.newest) {
      retained.second = retained.newest;
      retained.newest = ordinal;
    } else if (!retained.second || ordinal > *retained.second) {
      retained.second = ordinal;
    }
    return absl::OkStatus();
  }));
  return retained;
}
absl::Status Prune(
    const fs::path& root,
    const LiquidResolvedScene3D& scene,
    const LiquidDigest3D& fingerprint,
    LiquidCacheLimits3D limits
) {
  auto retained = Retained(root, fingerprint, limits);
  if (!retained.ok()) {
    return retained.status();
  }
  // The immutable inventory determines eligibility, not the presence of a blob.
  // Verify both retained states before removing any older payload. Keeping the
  // references also preserves historical replay identity and avoids depending
  // on durable unlink ordering (which Windows does not expose for directories).
  for (const auto ordinal : {retained->newest, retained->second}) {
    if (!ordinal) {
      continue;
    }
    auto reference = ReadReference(root, *ordinal, fingerprint, limits);
    if (!reference.ok()) {
      return reference.status();
    }
    auto checkpoint = ReadCheckpointRecord(root, *reference, scene, fingerprint, limits);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
  }
  if (!retained->second) {
    return absl::OkStatus();
  }
  return VisitReferences(root, [&](std::uint64_t ordinal) {
    if (ordinal >= *retained->second) {
      return absl::OkStatus();
    }
    auto reference = ReadReference(root, ordinal, fingerprint, limits);
    if (!reference.ok()) {
      return reference.status();
    }
    // Missing old payload is expected after pruning. Existing payload must still
    // verify its exact bytes and complete physical identity before deletion.
    auto checkpoint = ReadCheckpointRecord(root, *reference, scene, fingerprint, limits);
    if (checkpoint.status().code() == absl::StatusCode::kNotFound) {
      auto path = ContentPath(root, reference->state);
      if (!path.ok()) {
        return path.status();
      }
      auto bytes = ReadBytes(*path, limits, reference->state.byte_length);
      if (bytes.status().code() == absl::StatusCode::kNotFound) {
        return absl::OkStatus();
      }
      return checkpoint.status();
    }
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
    auto path = ContentPath(root, reference->state);
    if (!path.ok()) {
      return path.status();
    }
    std::error_code error;
    fs::remove(*path, error);
    if (error) {
      return Failure(absl::StatusCode::kUnavailable);
    }
    return FlushDirectory(root / "blobs");
  });
}
// Validate immutable checkpoint identity before publishing any candidate bytes,
// including when retention has already removed the older state payload.
absl::StatusOr<LiquidCacheBytes3D>
PrepareCheckpointReference(const fs::path& root, const CheckpointReference& reference, LiquidCacheLimits3D limits) {
  auto bytes = EncodeReference(reference, limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  const auto existing = Matches(
      root / "checkpoints" / OrdinalName(reference.clock.frame_ordinal, ".flcr"),
      bytes->values(),
      limits,
      absl::StatusCode::kAlreadyExists
  );
  if (!existing.ok() && existing.code() != absl::StatusCode::kNotFound) {
    return existing;
  }
  return std::move(*bytes);
}
absl::Status PublishCheckpointRecord(
    const fs::path& root,
    const LiquidState3D& state,
    const LiquidResumeMetadata3D& resume,
    std::span<const std::uint8_t> encoded,
    const LiquidResolvedScene3D& scene,
    const LiquidDigest3D& fingerprint,
    LiquidCacheLimits3D limits,
    LiquidCachePublishControl3D control
) {
  auto frame = ReadFrame(root, state.clock.frame_ordinal, fingerprint, limits);
  if (!frame.ok()) {
    return frame.status();
  }
  STORE_TRY(CheckpointMatches(state, resume, frame->points, scene, fingerprint));
  auto manifest_bytes = ReadBytes(FramePath(root, state.clock.frame_ordinal), limits);
  if (!manifest_bytes.ok()) {
    return manifest_bytes.status();
  }
  CheckpointReference reference{
      state.clock,
      fingerprint,
      LiquidExecutionId3D(fingerprint),
      Reference(LiquidContentKind3D::kState, encoded),
      Reference(LiquidContentKind3D::kManifest, manifest_bytes->values()),
  };
  auto reference_bytes = PrepareCheckpointReference(root, reference, limits);
  if (!reference_bytes.ok()) {
    return reference_bytes.status();
  }
  STORE_TRY(PublishContent(root, reference.state, encoded, limits, control));
  STORE_TRY(PublishBytes(
      root / "checkpoints" / OrdinalName(state.clock.frame_ordinal, ".flcr"),
      reference_bytes->values(),
      limits,
      control,
      absl::StatusCode::kAlreadyExists
  ));
  STORE_TRY(Check(control, LiquidCachePublishStage3D::kCheckpointPublished));
  STORE_TRY(Check(control, LiquidCachePublishStage3D::kBeforePrune));
  return Prune(root, scene, fingerprint, limits);
}
}  // namespace

absl::StatusOr<LiquidCache3D> LiquidCache3D::Create(
    const fs::path& requested_directory,
    LiquidResolvedSceneView3D resolved_scene,
    LiquidCacheLimits3D limits
) {
  auto absolute_directory = AbsoluteDirectory(requested_directory);
  if (!absolute_directory.ok()) {
    return absolute_directory.status();
  }
  const auto& root_directory = *absolute_directory;
  auto bytes = EncodeLiquidResolvedScene3D(resolved_scene, limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto fingerprint = ResolvedSceneFingerprint(resolved_scene, limits);
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  NativeFile lease;
  STORE_TRY(Lease(root_directory, lease, true));
  STORE_TRY(
      PublishBytes(root_directory / "resolved.flrs", bytes->values(), limits, {}, absl::StatusCode::kFailedPrecondition)
  );
  STORE_TRY(FlushDirectory(root_directory));
  // Lease creation already persisted each ancestor before creating descendants.
  LiquidCache3D result;
  result.root_directory_ = root_directory;
  result.fingerprint_ = *fingerprint;
  result.resolved_reference_ = Reference(LiquidContentKind3D::kResolvedScene, bytes->values());
  result.limits_ = limits;
  return result;
}
absl::StatusOr<LiquidCache3D> LiquidCache3D::Open(
    const fs::path& requested_directory,
    const LiquidDigest3D& expected_fingerprint,
    LiquidCacheLimits3D limits
) {
  auto absolute_directory = AbsoluteDirectory(requested_directory);
  if (!absolute_directory.ok()) {
    return absolute_directory.status();
  }
  const auto& root_directory = *absolute_directory;
  NativeFile lease;
  STORE_TRY(Lease(root_directory, lease));
  auto scene = ReadResolved(root_directory, expected_fingerprint, limits);
  if (!scene.ok()) {
    return scene.status();
  }
  auto bytes = EncodeLiquidResolvedScene3D(scene->View(), limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  LiquidCache3D result;
  result.root_directory_ = root_directory;
  result.fingerprint_ = expected_fingerprint;
  result.resolved_reference_ = Reference(LiquidContentKind3D::kResolvedScene, bytes->values());
  result.limits_ = limits;
  return result;
}
absl::StatusOr<LiquidCache3D> LiquidCache3D::OpenForReplay(
    const fs::path& root_directory,
    const LiquidDigest3D& expected_fingerprint,
    LiquidCacheLimits3D limits
) {
  auto result = OpenForInspection(root_directory, limits);
  if (!result.ok()) {
    return result.status();
  }
  if (result->fingerprint_ != expected_fingerprint) {
    return Failure(absl::StatusCode::kFailedPrecondition);
  }
  result->inspection_only_ = false;
  return result;
}
absl::StatusOr<LiquidCache3D>
LiquidCache3D::OpenForInspection(const fs::path& requested_directory, LiquidCacheLimits3D limits) {
  auto absolute_directory = AbsoluteDirectory(requested_directory);
  if (!absolute_directory.ok()) {
    return absolute_directory.status();
  }
  const auto& root_directory = *absolute_directory;
  NativeFile lease;
  STORE_TRY(Lease(root_directory, lease, false, true));
  auto bytes = ReadBytes(root_directory / "resolved.flrs", limits);
  if (!bytes.ok()) {
    return bytes.status();
  }
  auto decoded = DecodeLiquidResolvedScene3D(bytes->values(), limits);
  if (!decoded.ok()) {
    return decoded.status();
  }
  auto fingerprint = ResolvedSceneFingerprint(decoded->View(), limits);
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  auto canonical = ReadResolved(root_directory, *fingerprint, limits);
  if (!canonical.ok()) {
    return canonical.status();
  }
  LiquidCache3D result;
  result.root_directory_ = root_directory;
  result.fingerprint_ = *fingerprint;
  result.resolved_reference_ = Reference(LiquidContentKind3D::kResolvedScene, bytes->values());
  result.limits_ = limits;
  result.inspection_only_ = true;
  result.read_only_ = true;
  return result;
}
absl::StatusOr<LiquidResolvedScene3D> LiquidCache3D::ReadResolvedScene() const {
  NativeFile lease;
  STORE_TRY(Lease(root_directory_, lease, false, read_only_));
  return ReadResolved(root_directory_, fingerprint_, limits_, &resolved_reference_);
}
absl::Status LiquidCache3D::PublishFrame(
    const LiquidPointFrame3D& points,
    const LiquidState3D* checkpoint,
    const LiquidResumeMetadata3D* resume,
    LiquidCachePublishControl3D control
) const {
  if (read_only_) {
    return Failure(absl::StatusCode::kFailedPrecondition);
  }
  const bool needs_checkpoint = points.clock.frame_ordinal % 10 == 0;
  if ((checkpoint != nullptr) != needs_checkpoint || (resume != nullptr) != needs_checkpoint) {
    return Failure(absl::StatusCode::kInvalidArgument);
  }
  auto point_bytes = EncodeLiquidPointFrame3D(points, limits_);
  if (!point_bytes.ok()) {
    return point_bytes.status();
  }
  NativeFile lease;
  STORE_TRY(Lease(root_directory_, lease, false, read_only_));
  auto scene = ReadResolved(root_directory_, fingerprint_, limits_, &resolved_reference_);
  if (!scene.ok()) {
    return scene.status();
  }
  if (points.clock.frame_ordinal != 0) {
    auto initial = ReadFrame(root_directory_, 0, fingerprint_, limits_);
    if (!initial.ok()) {
      return initial.status().code() == absl::StatusCode::kNotFound ? Failure(absl::StatusCode::kFailedPrecondition)
                                                                    : initial.status();
    }
    auto initial_reference = ReadReference(root_directory_, 0, fingerprint_, limits_);
    if (!initial_reference.ok()) {
      if (initial_reference.status().code() == absl::StatusCode::kNotFound) {
        return Failure(absl::StatusCode::kFailedPrecondition);
      }
      return initial_reference.status();
    }
  }
  LiquidCacheBytes3D checkpoint_bytes;
  if (needs_checkpoint) {
    STORE_TRY(CheckpointMatches(*checkpoint, *resume, points, *scene, fingerprint_));
    auto encoded = EncodeLiquidCheckpoint3D(*checkpoint, *resume, limits_);
    if (!encoded.ok()) {
      return encoded.status();
    }
    checkpoint_bytes = std::move(*encoded);
  }
  LiquidFrameManifest3D manifest{
      points.clock,
      fingerprint_,
      LiquidExecutionId3D(fingerprint_),
      Reference(LiquidContentKind3D::kPoints, point_bytes->values()),
  };
  auto manifest_bytes = EncodeLiquidFrameManifest3D(manifest, limits_);
  if (!manifest_bytes.ok()) {
    return manifest_bytes.status();
  }
  const auto manifest_path = FramePath(root_directory_, points.clock.frame_ordinal);
  auto existing = Matches(manifest_path, manifest_bytes->values(), limits_, absl::StatusCode::kAlreadyExists);
  if (existing.ok()) {
    auto prior = ReadFrame(root_directory_, points.clock.frame_ordinal, fingerprint_, limits_);
    if (!prior.ok()) {
      return prior.status();
    }
  } else if (existing.code() != absl::StatusCode::kNotFound) {
    return existing;
  }
  if (needs_checkpoint) {
    const CheckpointReference reference{
        points.clock,
        fingerprint_,
        LiquidExecutionId3D(fingerprint_),
        Reference(LiquidContentKind3D::kState, checkpoint_bytes.values()),
        Reference(LiquidContentKind3D::kManifest, manifest_bytes->values()),
    };
    auto prepared = PrepareCheckpointReference(root_directory_, reference, limits_);
    if (!prepared.ok()) {
      return prepared.status();
    }
  }
  STORE_TRY(PublishContent(root_directory_, manifest.points, point_bytes->values(), limits_, control));
  if (needs_checkpoint) {
    STORE_TRY(PublishContent(
        root_directory_,
        Reference(LiquidContentKind3D::kState, checkpoint_bytes.values()),
        checkpoint_bytes.values(),
        limits_,
        control
    ));
  }
  STORE_TRY(PublishContent(
      root_directory_,
      Reference(LiquidContentKind3D::kManifest, manifest_bytes->values()),
      manifest_bytes->values(),
      limits_,
      control
  ));
  const auto inspection = Inspection(manifest);
  STORE_TRY(PublishBytes(
      FramePath(root_directory_, points.clock.frame_ordinal, ".json"),
      Bytes(inspection),
      limits_,
      control,
      absl::StatusCode::kAlreadyExists
  ));
  STORE_TRY(Check(control, LiquidCachePublishStage3D::kInspectionPublished));
  STORE_TRY(Check(control, LiquidCachePublishStage3D::kBeforeManifest));
  // This is the sole point-frame commit. No incomplete trial is discoverable
  // before this no-replace promotion. Checkpoint metadata is independently repairable.
  STORE_TRY(PublishBytes(manifest_path, manifest_bytes->values(), limits_, control, absl::StatusCode::kAlreadyExists));
  STORE_TRY(Check(control, LiquidCachePublishStage3D::kManifestPublished));
  if (needs_checkpoint) {
    return PublishCheckpointRecord(
        root_directory_,
        *checkpoint,
        *resume,
        checkpoint_bytes.values(),
        *scene,
        fingerprint_,
        limits_,
        control
    );
  }
  return absl::OkStatus();
}
absl::StatusOr<LiquidCommittedFrame3D> LiquidCache3D::ReadCommittedFrame(std::uint64_t frame_ordinal) const {
  NativeFile lease;
  STORE_TRY(Lease(root_directory_, lease, false, read_only_));
  auto scene = ReadResolved(root_directory_, fingerprint_, limits_, &resolved_reference_);
  if (!scene.ok()) {
    return scene.status();
  }
  return ReadFrame(root_directory_, frame_ordinal, fingerprint_, limits_);
}
absl::Status LiquidCache3D::PublishCheckpoint(
    const LiquidState3D& checkpoint,
    const LiquidResumeMetadata3D& resume,
    LiquidCachePublishControl3D control
) const {
  if (read_only_) {
    return Failure(absl::StatusCode::kFailedPrecondition);
  }
  auto encoded = EncodeLiquidCheckpoint3D(checkpoint, resume, limits_);
  if (!encoded.ok()) {
    return encoded.status();
  }
  NativeFile lease;
  STORE_TRY(Lease(root_directory_, lease, false, read_only_));
  auto scene = ReadResolved(root_directory_, fingerprint_, limits_, &resolved_reference_);
  if (!scene.ok()) {
    return scene.status();
  }
  return PublishCheckpointRecord(
      root_directory_,
      checkpoint,
      resume,
      encoded->values(),
      *scene,
      fingerprint_,
      limits_,
      control
  );
}
absl::StatusOr<LiquidCheckpoint3D> LiquidCache3D::ReadCheckpoint(std::uint64_t maximum_frame) const {
  if (inspection_only_) {
    return Failure(absl::StatusCode::kFailedPrecondition);
  }
  NativeFile lease;
  STORE_TRY(Lease(root_directory_, lease, false, read_only_));
  auto scene = ReadResolved(root_directory_, fingerprint_, limits_, &resolved_reference_);
  if (!scene.ok()) {
    return scene.status();
  }
  auto retained = Retained(root_directory_, fingerprint_, limits_);
  if (!retained.ok()) {
    return retained.status();
  }
  std::optional<std::uint64_t> latest;
  if (retained->newest && *retained->newest <= maximum_frame) {
    latest = retained->newest;
  } else if (retained->second && *retained->second <= maximum_frame) {
    latest = retained->second;
  }
  if (!latest) {
    return Failure(absl::StatusCode::kNotFound);
  }
  auto reference = ReadReference(root_directory_, *latest, fingerprint_, limits_);
  if (!reference.ok()) {
    return reference.status();
  }
  return ReadCheckpointRecord(root_directory_, *reference, *scene, fingerprint_, limits_);
}
#undef STORE_TRY
}  // namespace fluidity
