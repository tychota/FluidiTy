#pragma once

#include "fluidity/host/effects_source_3d.h"
#include "fluidity/host/whitewater_cache_3d.h"

namespace fluidity {
struct WhitewaterStageFrame3D {
  LiquidClock3D clock{};
  LiquidDigest3D payload_digest{};
  std::size_t foam_count = 0;
  std::size_t spray_count = 0;
  std::size_t bubble_count = 0;
  std::size_t birth_count = 0;
  std::size_t expired_count = 0;
  std::size_t bubble_fallback_substeps = 0;
};

// Sequential host Whitewater publication over authenticated accepted births.
// The owner keeps only the latest point/remainder checkpoint. A caller serializes
// this owner and holds its own whole-job lease for multi-frame publication.
class WhitewaterStage3D {
 public:
  // Requires an accepted source with resolved birth capture enabled. Creates
  // the separate settings/stage identity, or verifies exact existing identity.
  // Invalid configuration/overlap returns InvalidArgument, different identity
  // FailedPrecondition. No Simulate data is written.
  static absl::StatusOr<WhitewaterStage3D> OpenForPublication(
      const std::filesystem::path& simulate_root,
      const std::filesystem::path& output_root,
      LiquidDigest3D build,
      const WhitewaterFrameOptions3D& options = {}
  );
  // Opens existing source and Whitewater stores without writes. Requires exact
  // source and build, and reads authenticated canonical physical settings.
  static absl::StatusOr<WhitewaterStage3D> OpenForReplay(
      const std::filesystem::path& simulate_root,
      const std::filesystem::path& output_root,
      LiquidDigest3D build
  );
  std::uint64_t next_frame() const;
  const EffectsIdentity3D& identity() const;
  LiquidBackend3D source_backend() const;
  // Authenticates the prefix through the bound or first absent manifest and
  // decodes its latest checkpoint. Returns restored frame count. Checks stop
  // between frames; a failed restore leaves the fresh owner's cursor unchanged.
  absl::StatusOr<std::uint64_t> RestoreCommittedPrefix(
      std::uint64_t maximum_frame,
      absl::FunctionRef<absl::Status()> check_stop =
          [] {
            return absl::OkStatus();
          }
  );
  // Evolves the next interval and publishes immutable point state, or compares
  // exact existing bytes during replay. Binds source payload and prior point
  // checkpoint. Advances only on success; retry after a late cancellation
  // verifies any durable publication rather than overwriting it.
  absl::StatusOr<WhitewaterStageFrame3D> ProcessNextFrame(LiquidCachePublishControl3D control = {});

 private:
  WhitewaterStage3D(EffectsSource3D source, EffectsCache3D cache, WhitewaterFrameOptions3D options, bool read_only);
  EffectsSource3D source_;
  EffectsCache3D cache_;
  WhitewaterFrameOptions3D options_;
  bool read_only_;
  std::uint64_t next_frame_ = 0;
  std::optional<WhitewaterFrame3D> previous_frame_;
  LiquidDigest3D previous_payload_{};
};
}  // namespace fluidity
