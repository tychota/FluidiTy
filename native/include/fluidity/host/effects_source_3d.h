#pragma once

#include "fluidity/host/effects_cache_3d.h"
#include "fluidity/host/motion_capture_3d.h"

namespace fluidity {
// Verifies distinct trusted cache roots before a derived writer opens. Output
// may be disjoint or below the stage's reserved effects subtree. Rejects roots
// overlapping Simulate-owned data or another stage. No files are created.
absl::Status ValidateEffectsOutputRoot3D(
    const std::filesystem::path& simulate_root,
    const std::filesystem::path& output_root,
    EffectsStage3D stage
);

struct EffectsSourceFrame3D {
  MotionFrame3D motion;
  LiquidDigest3D source_payload;
  LiquidDigest3D parent_points;
};

// Read-only joined view of Simulate points and its accepted source products.
// A source publication preceding its parent manifest is not consumable yet.
class EffectsSource3D {
 public:
  // Opens existing caches without creating files. Requires enabled capture and
  // exact source stage/upstream/settings/build identity matching the resolved
  // Simulate scene. Missing files return NotFound, foreign configuration
  // FailedPrecondition, corruption DataLoss; native I/O/resource errors propagate.
  static absl::StatusOr<EffectsSource3D>
  Open(const std::filesystem::path& simulate_root, LiquidCacheLimits3D limits = {});
  // Returns a borrowed resolved view; reacquire after this owner moves and keep
  // this owner alive while using its source/upstream spans.
  LiquidResolvedSceneView3D scene() const;
  const EffectsIdentity3D& identity() const;
  LiquidDigest3D execution_id() const;
  // Verifies parent and source content, shared completed clock, declared geometry
  // and seed, and exact canonical point bytes regenerated from the source endpoint.
  // For later frames, also verifies the adjacent parent and full FLST continuity
  // from the previous source endpoint, beyond attributes exported as points.
  // Does not create files or certify the producer's numerical physics. A job
  // claiming prefix replay must read/verify every required frame in its prefix.
  // Returns owning samples without aliases. Corrupt joins return DataLoss;
  // missing parent/source records NotFound; I/O/resource errors propagate.
  absl::StatusOr<EffectsSourceFrame3D> ReadFrame(std::uint64_t frame) const;

 private:
  EffectsSource3D(
      LiquidCache3D parent,
      EffectsCache3D source,
      LiquidResolvedScene3D scene,
      LiquidDigest3D execution,
      LiquidCacheLimits3D limits
  );
  LiquidCache3D parent_;
  EffectsCache3D source_;
  LiquidResolvedScene3D scene_;
  LiquidDigest3D execution_;
  LiquidCacheLimits3D limits_;
};
}  // namespace fluidity
