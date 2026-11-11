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

absl::StatusOr<std::uint64_t>
SurfaceStage3D::RestoreCommittedPrefix(std::uint64_t maximum_frame, absl::FunctionRef<absl::Status()> check_stop) {
  if (read_only_ || next_frame_ != 0) {
    return absl::FailedPreconditionError("Mesh checkpoint restore requires a fresh publication owner");
  }
  if (maximum_frame == std::numeric_limits<std::uint64_t>::max()) {
    return absl::OutOfRangeError("Mesh restore frame bound exhausts cursor");
  }
  std::optional<EffectsCommit3D> checkpoint_commit;
  double checkpoint_material_volume_m3 = 0;
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
    auto source = source_.ReadFrame(restored_count);
    if (!source.ok()) {
      return absl::IsNotFound(source.status()) ? absl::DataLossError("committed Mesh source is missing")
                                               : source.status();
    }
    const auto expected_previous = checkpoint_commit ? checkpoint_commit->payload_digest : LiquidDigest3D{};
    if ((**committed).clock != source->motion.last_clock || (**committed).upstream_content != source->source_payload
        || (**committed).previous_payload != expected_previous) {
      return absl::DataLossError("Mesh checkpoint source or predecessor differs");
    }
    checkpoint_material_volume_m3 = 0;
    for (const auto& particle : source->motion.samples.back().particles.values()) {
      checkpoint_material_volume_m3 += particle.reference_volume * particle.volume_ratio;
    }
    checkpoint_commit = std::move(**committed);
  }
  if (!checkpoint_commit) {
    return 0;
  }
  auto checkpoint = DecodeSurfaceFrame3D(checkpoint_commit->payload.values(), options_.limits);
  if (!checkpoint.ok()) {
    return checkpoint.status();
  }
  const double declared_material_volume_m3 = checkpoint->volumes.empty() ? 0 : checkpoint->volumes.front().material_m3;
  if (checkpoint->clock != checkpoint_commit->clock || checkpoint->cell_width_m != source_.scene().geometry.cell_width
      || checkpoint->volumes.size() > 1 || declared_material_volume_m3 != checkpoint_material_volume_m3) {
    return absl::DataLossError("Mesh checkpoint clock, grid or material volume differs from source");
  }
  previous_frame_ = std::move(*checkpoint);
  previous_payload_ = checkpoint_commit->payload_digest;
  next_frame_ = restored_count;
  return restored_count;
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
