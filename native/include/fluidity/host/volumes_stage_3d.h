#pragma once

#include "fluidity/host/effects_cache_3d.h"
#include "fluidity/host/whitewater_volume_cache_3d.h"

namespace fluidity {
struct VolumesStageFrame3D {
  LiquidClock3D clock{};
  LiquidDigest3D payload_digest{};
  std::size_t known_voxels = 0;
  std::size_t nonzero_voxels = 0;
  double foam_inventory_count = 0;
  double bubble_inventory_count = 0;
  double gas_inventory_m3 = 0;
};

// Sequential native volume publication from authenticated Whitewater points.
// The caller serializes access and owns a whole-job lease for a publishing run.
class VolumesStage3D {
 public:
  // Validates the distinct cache roots, fixed physical volume settings, exact
  // Whitewater stage and represented count. Creates an immutable derived
  // identity and settings sidecar. Never writes to the Whitewater cache.
  static absl::StatusOr<VolumesStage3D> OpenForPublication(
      const std::filesystem::path& whitewater_root,
      const std::filesystem::path& output_root,
      LiquidDigest3D build,
      const WhitewaterVolumeSettings3D& settings
  );
  // Authenticates source, settings and build without filesystem writes.
  static absl::StatusOr<VolumesStage3D> OpenForReplay(
      const std::filesystem::path& whitewater_root,
      const std::filesystem::path& output_root,
      LiquidDigest3D build
  );
  std::uint64_t next_frame() const;
  const EffectsIdentity3D& identity() const;
  // Verifies committed lineage through the bound or first absent manifest.
  // Restores the latest digest and cursor, not a dense retained voxel array.
  // Failed validation or stop leaves the fresh owner's cursor unchanged.
  absl::StatusOr<std::uint64_t> RestoreCommittedPrefix(
      std::uint64_t maximum_frame,
      absl::FunctionRef<absl::Status()> check_stop =
          [] {
            return absl::OkStatus();
          }
  );
  // Deposits and measures conserved inventories before publication, or
  // regenerates and checks exact committed bytes in read-only replay. Advances
  // only on success. Missing frames, unsupported support and native I/O errors
  // propagate; inconsistent lineage or inventory is DataLoss.
  absl::StatusOr<VolumesStageFrame3D> ProcessNextFrame(LiquidCachePublishControl3D control = {});

 private:
  VolumesStage3D(
      EffectsCache3D source,
      EffectsCache3D cache,
      WhitewaterVolumeSettings3D settings,
      double represented_count_per_point,
      bool read_only
  );
  EffectsCache3D source_;
  EffectsCache3D cache_;
  WhitewaterVolumeSettings3D settings_;
  double represented_count_per_point_;
  bool read_only_;
  std::uint64_t next_frame_ = 0;
  LiquidDigest3D previous_payload_{};
};
}  // namespace fluidity
