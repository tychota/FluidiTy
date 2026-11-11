#pragma once

#include <optional>

#include "fluidity/host/effects_source_3d.h"
#include "fluidity/host/surface_settings_3d.h"

namespace fluidity {
struct SurfaceStageFrame3D {
  SurfaceReconstructionResult3D reconstruction;
  LiquidDigest3D payload_digest{};
};

// Sequential geometric Mesh job over authenticated accepted source intervals.
// Each owner has a private cursor; callers serialize access and retain the
// external whole-job lease across multiple publication calls.
class SurfaceStage3D {
 public:
  // Opens the read-only Simulate source first, then creates/verifies a Mesh store
  // bound to its execution, canonical physical settings and nonzero build digest.
  // Existing different identity returns FailedPrecondition. Invalid settings
  // or overlapping output roots return InvalidArgument. The effects/mesh subtree
  // of Simulate is allowed; other output roots must be disjoint from Simulate.
  // return InvalidArgument; native I/O/budget errors propagate. Re-running from
  // zero regenerates and verifies existing frames without replacing their bytes.
  static absl::StatusOr<SurfaceStage3D> OpenForPublication(
      const std::filesystem::path& simulate_root,
      const std::filesystem::path& mesh_root,
      LiquidDigest3D build,
      const SurfaceReconstructionOptions3D& options
  );
  // Opens existing stores/settings without creating files. Exact source and
  // execution build must match. Regenerates each requested prefix frame and
  // compares clock, lineage and payload bytes. Operational limits use defaults.
  static absl::StatusOr<SurfaceStage3D> OpenForReplay(
      const std::filesystem::path& simulate_root,
      const std::filesystem::path& mesh_root,
      LiquidDigest3D build
  );
  std::uint64_t next_frame() const;
  const EffectsIdentity3D& identity() const;
  // Consumes the next contiguous source interval, reconstructs and encodes its
  // complete persistent state, then publishes or compares an existing commit.
  // Derived upstream content is the exact source payload, not parent point data.
  // Cursor/checkpoint advance only after success. Missing source/derived frames
  // return NotFound; replay differences DataLoss; unsupported geometry and work
  // limits propagate. A publication control failure can follow a durable commit;
  // retry safely checks identical bytes. No in-memory partial result on failure.
  absl::StatusOr<SurfaceStageFrame3D> ProcessNextFrame(LiquidCachePublishControl3D control = {});

 private:
  SurfaceStage3D(EffectsSource3D source, EffectsCache3D cache, SurfaceReconstructionOptions3D options, bool read_only);
  EffectsSource3D source_;
  EffectsCache3D cache_;
  SurfaceReconstructionOptions3D options_;
  bool read_only_;
  std::uint64_t next_frame_ = 0;
  std::optional<SurfaceFrame3D> previous_frame_;
  LiquidDigest3D previous_payload_{};
};
}  // namespace fluidity
