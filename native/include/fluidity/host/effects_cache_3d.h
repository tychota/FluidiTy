#pragma once

#include "fluidity/host/liquid_cache_3d.h"

namespace fluidity {

enum class EffectsStage3D : std::uint8_t {
  kSource = 0,
  kMesh = 1,
  kWhitewater = 2,
  kVolumes = 3,
};
struct EffectsIdentity3D {
  EffectsStage3D stage = EffectsStage3D::kSource;
  LiquidDigest3D upstream_execution{};
  LiquidDigest3D settings{};  // Digest of canonical resolved stage settings.
  LiquidDigest3D build{};  // Exact execution artifact.
  bool operator==(const EffectsIdentity3D&) const = default;
};
struct EffectsCommit3D {
  EffectsIdentity3D identity;
  LiquidClock3D clock;
  LiquidDigest3D upstream_content{};
  LiquidDigest3D previous_payload{};
  LiquidDigest3D payload_digest{};
  LiquidCacheBytes3D payload;  // Authenticated bytes. The stage decoder validates their schema.
};

// Immutable downstream stage storage. The trusted root is captured as an absolute path.
// Reuses native no-follow reads, an exclusive nonblocking lease and no-replace publication.
// Payloads and JSON mirrors precede the binary manifest. Published frames are contiguous
// from zero, with each frame naming the preceding payload. Payload schemas remain stage-owned.
class EffectsCache3D {
 public:
  // Creates or verifies the exact stage identity. A different identity returns
  // FailedPrecondition. Invalid stage or zero identities return InvalidArgument.
  static absl::StatusOr<EffectsCache3D>
  Create(const std::filesystem::path& root, EffectsIdentity3D identity, LiquidCacheLimits3D limits = {});
  // Reads the existing identity without creating files. No publication is authorized.
  static absl::StatusOr<EffectsCache3D>
  OpenForInspection(const std::filesystem::path& root, LiquidCacheLimits3D limits = {});
  const EffectsIdentity3D& identity() const;
  // Publishes a nonempty bounded payload at a completed clock. Frame zero has zero
  // time, step and epoch and no previous payload. Later frames must name the verified
  // preceding payload and advance time and epoch. Identical replay preserves bytes
  // and mtimes. Conflicting replay returns AlreadyExists. Read-only stores return
  // FailedPrecondition before filesystem access. A control failure may occur after
  // commit; retry verifies bytes and restores the directory durability barrier.
  absl::Status Publish(
      LiquidClock3D clock,
      LiquidDigest3D upstream_content,
      LiquidDigest3D previous_payload,
      std::span<const std::uint8_t> payload,
      LiquidCachePublishControl3D control = {}
  ) const;
  // Verifies identity, clock, mirror, byte length, content and preceding lineage.
  // Missing records return NotFound, corruption DataLoss, excessive size
  // ResourceExhausted, busy stores Aborted and native I/O errors Unavailable.
  absl::StatusOr<EffectsCommit3D> ReadFrame(std::uint64_t frame) const;

 private:
  EffectsCache3D(std::filesystem::path root, EffectsIdentity3D identity, LiquidCacheLimits3D limits, bool read_only);
  std::filesystem::path root_;
  EffectsIdentity3D identity_;
  LiquidCacheLimits3D limits_;
  bool read_only_;
};
}  // namespace fluidity
