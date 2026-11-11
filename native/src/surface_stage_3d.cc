#include "fluidity/host/surface_stage_3d.h"

#include <limits>

namespace fluidity {
namespace {
// Keep derived files out of Simulate's own publication namespaces. The declared
// effects/mesh subtree is reserved for this stage; sibling cache roots also work.
bool Within(const std::filesystem::path& path, const std::filesystem::path& root) {
  const auto relative = path.lexically_relative(root);
  return !relative.empty() && !relative.is_absolute() && *relative.begin() != "..";
}
absl::Status CheckOutputRoot(const std::filesystem::path& simulate_root, const std::filesystem::path& mesh_root) {
  std::error_code error;
  const auto source = std::filesystem::weakly_canonical(simulate_root, error);
  if (error) {
    return absl::UnavailableError("could not resolve Simulate source root");
  }
  const auto output = std::filesystem::weakly_canonical(mesh_root, error);
  if (error) {
    return absl::UnavailableError("could not resolve Mesh output root");
  }
  if (Within(source, output) || (Within(output, source) && !Within(output, source / "effects/mesh"))) {
    return absl::InvalidArgumentError("Mesh output overlaps Simulate-owned data");
  }
  return absl::OkStatus();
}
}  // namespace
SurfaceStage3D::SurfaceStage3D(
    EffectsSource3D source,
    EffectsCache3D cache,
    SurfaceReconstructionOptions3D options,
    bool read_only
)
    : source_(std::move(source)), cache_(std::move(cache)), options_(options), read_only_(read_only) {
}

absl::StatusOr<SurfaceStage3D> SurfaceStage3D::OpenForPublication(
    const std::filesystem::path& simulate_root,
    const std::filesystem::path& mesh_root,
    LiquidDigest3D build,
    const SurfaceReconstructionOptions3D& options
) {
  const auto output_root = CheckOutputRoot(simulate_root, mesh_root);
  if (!output_root.ok()) {
    return output_root;
  }
  auto settings = EncodeSurfaceSettings3D(options, options.limits.cache);
  if (!settings.ok()) {
    return settings.status();
  }
  auto source = EffectsSource3D::Open(simulate_root, options.limits.cache);
  if (!source.ok()) {
    return source.status();
  }
  const EffectsIdentity3D identity{
      .stage = EffectsStage3D::kMesh,
      .upstream_execution = source->execution_id(),
      .settings = LiquidContentDigest3D(settings->values()),
      .build = build,
  };
  auto cache = EffectsCache3D::Create(mesh_root, identity, options.limits.cache);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto published = cache->PublishStageSettings(settings->values());
  if (!published.ok()) {
    return published;
  }
  return SurfaceStage3D(std::move(*source), std::move(*cache), options, false);
}

absl::StatusOr<SurfaceStage3D> SurfaceStage3D::OpenForReplay(
    const std::filesystem::path& simulate_root,
    const std::filesystem::path& mesh_root,
    LiquidDigest3D build
) {
  const auto output_root = CheckOutputRoot(simulate_root, mesh_root);
  if (!output_root.ok()) {
    return output_root;
  }
  auto source = EffectsSource3D::Open(simulate_root);
  if (!source.ok()) {
    return source.status();
  }
  auto cache = EffectsCache3D::OpenForInspection(mesh_root);
  if (!cache.ok()) {
    return cache.status();
  }
  const auto& identity = cache->identity();
  if (identity.stage != EffectsStage3D::kMesh || identity.upstream_execution != source->execution_id()
      || identity.build != build) {
    return absl::FailedPreconditionError("Mesh identity differs from source or execution build");
  }
  const auto settings = cache->ReadStageSettings();
  if (!settings.ok()) {
    return settings.status();
  }
  auto options = DecodeSurfaceSettings3D(settings->values());
  if (!options.ok()) {
    return options.status();
  }
  return SurfaceStage3D(std::move(*source), std::move(*cache), *options, true);
}

std::uint64_t SurfaceStage3D::next_frame() const {
  return next_frame_;
}
const EffectsIdentity3D& SurfaceStage3D::identity() const {
  return cache_.identity();
}

absl::StatusOr<SurfaceStageFrame3D> SurfaceStage3D::ProcessNextFrame(LiquidCachePublishControl3D control) {
  if (next_frame_ == std::numeric_limits<std::uint64_t>::max()) {
    return absl::OutOfRangeError("Mesh frame cursor exhausted");
  }
  auto source = source_.ReadFrame(next_frame_);
  if (!source.ok()) {
    return source.status();
  }
  const auto scene = source_.scene();
  auto reconstructed = ReconstructSurfaceFrame3D(
      source->motion,
      scene.scene,
      scene.options,
      previous_frame_ ? &*previous_frame_ : nullptr,
      options_
  );
  if (!reconstructed.ok()) {
    return reconstructed.status();
  }
  auto payload = EncodeSurfaceFrame3D(reconstructed->frame, options_.limits);
  if (!payload.ok()) {
    return payload.status();
  }
  const auto payload_digest = LiquidContentDigest3D(payload->values());
  if (read_only_) {
    auto committed = cache_.ReadFrame(next_frame_);
    if (!committed.ok()) {
      return committed.status();
    }
    if (committed->clock != reconstructed->frame.clock || committed->upstream_content != source->source_payload
        || committed->previous_payload != previous_payload_ || committed->payload_digest != payload_digest
        || !std::ranges::equal(committed->payload.values(), payload->values())) {
      return absl::DataLossError("Mesh replay differs from committed source, lineage or geometry");
    }
  } else {
    const auto published =
        cache_
            .Publish(reconstructed->frame.clock, source->source_payload, previous_payload_, payload->values(), control);
    if (!published.ok()) {
      return published;
    }
  }
  previous_frame_ = reconstructed->frame;
  previous_payload_ = payload_digest;
  ++next_frame_;
  return SurfaceStageFrame3D{.reconstruction = std::move(*reconstructed), .payload_digest = payload_digest};
}
}  // namespace fluidity
