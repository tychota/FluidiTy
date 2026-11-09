#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "fluidity/host/liquid_substep_3d.h"

namespace fluidity {
// BLAKE3-256, raw digest bytes in algorithm order (not native integer words).
using LiquidDigest3D = std::array<std::uint8_t, 32>;
using LiquidCacheBytes3D = LiquidRecords3D<std::uint8_t>;
enum class LiquidContentKind3D : std::uint8_t {
  kState = 0,
  kPoints = 1,
  kResolvedScene = 2,
  kManifest = 3,
};
enum class LiquidBackend3D : std::uint8_t {
  kCpu = 0,
  kCuda = 1,
};
enum class LiquidUpstreamRole3D : std::uint8_t {
  kInitialParticles = 0,
  kSource = 1,
  kObstacle = 2,
};
// Resource limits apply before decoded record allocation. The byte budget bounds
// encoded input and encoder arena storage (4096 root bytes + 512 per child table);
// an encoder may need more arena bytes than final content. Every record family has
// an additional hard ceiling of 2^20. FlatBuffers uses signed 32-bit offsets.
struct LiquidCacheLimits3D {
  std::size_t maximum_bytes = 64 * 1024 * 1024;
  LiquidStateCapacities3D maximum_records{1024, 4096, 4096, 24576, 256};
  std::size_t maximum_upstream_records = 256;
};
struct LiquidContentReference3D {
  LiquidContentKind3D kind = LiquidContentKind3D::kState;
  std::uint64_t byte_length = 0;
  LiquidDigest3D digest{};
  bool operator==(const LiquidContentReference3D&) const = default;
};
// Separate from physical equality and ABI. No process identity, heartbeat, paths,
// device addresses, events or allocator capacity are serialized in LiquidState3D.
struct LiquidResumeMetadata3D {
  LiquidDigest3D resolved_fingerprint{};
  LiquidDigest3D execution_id{};
  LiquidStateCapacities3D capacities;
  LiquidStateCapacities3D capacity_replay_reservations;
  bool operator==(const LiquidResumeMetadata3D&) const = default;
};
struct LiquidCheckpoint3D {
  LiquidState3D state;
  LiquidResumeMetadata3D resume;
};
struct LiquidPoint3D {
  std::uint64_t id = 0;  // Stable nonzero particle ID, unique within this frame.
  std::uint32_t tag = 0;  // Packed substance/phase/flags, matching LiquidParticleState3D.
  Vec3 world_position{};  // m.
  Vec3 world_velocity{};  // Stored velocity, m/s.
  double radius = 1;  // Positive visualization radius, m.
  double volume_ratio = 1;  // Dimensionless J.
  std::uint8_t level = 0;  // Uniform M1 admits level zero only.
};
struct LiquidPointFrame3D {
  LiquidClock3D clock;  // Completed frame (substep zero).
  LiquidRecords3D<LiquidPoint3D> points;  // Stable physical particle order.
};
struct LiquidFrameManifest3D {
  LiquidClock3D clock;
  LiquidDigest3D resolved_fingerprint{};
  LiquidDigest3D execution_id{};
  LiquidContentReference3D points;  // Complete only; checkpoint retention has its own index.
};
struct LiquidUpstreamContent3D {
  LiquidUpstreamRole3D role = LiquidUpstreamRole3D::kInitialParticles;
  std::uint64_t ordinal = 0;
  std::uint64_t byte_length = 0;
  LiquidDigest3D digest{};  // Exact upstream artifact bytes, never a local filename.
};
// Callers resolve defaults before encoding. Omitted and explicit defaults produce
// identical bytes. Source/upstream spans borrow through the call only. Upstream
// order is unique ascending (role, ordinal); source order is unique ordinal order.
// Capacity fields in options are intentionally excluded: store them in resume.
struct LiquidResolvedSceneView3D {
  UniformMacGeometry3D geometry{{0, 0, 0}, 1};
  std::uint64_t random_seed = 0;
  LiquidSceneInputs3D scene;
  LiquidStepOptions3D options;
  LiquidBackend3D backend = LiquidBackend3D::kCpu;
  std::uint32_t model_version = 1;  // Uniform single-liquid algorithm contract.
  LiquidDigest3D build_digest{};  // Content identity of the numerical build artifact.
  std::span<const LiquidUpstreamContent3D> upstream;
};
// Owning decoded scene; View() binds spans to this owner's arrays. Reacquire after
// move; a view cannot outlive this object. No pointer into encoded bytes survives.
struct LiquidResolvedScene3D {
  UniformMacGeometry3D geometry{{0, 0, 0}, 1};
  std::uint64_t random_seed = 0;
  LiquidSceneInputs3D scene;  // sources remains empty; use View().scene.sources.
  LiquidStepOptions3D options;  // Capacity plans remain zero; read them from checkpoint.
  LiquidBackend3D backend = LiquidBackend3D::kCpu;
  std::uint32_t model_version = 1;
  LiquidDigest3D build_digest{};
  LiquidRecords3D<LiquidEmissionSource3D> sources;
  LiquidRecords3D<LiquidUpstreamContent3D> upstream;
  LiquidResolvedSceneView3D View() const;
};
// Exact unkeyed BLAKE3-256 over bytes; no domain prefix (standard known answers).
LiquidDigest3D LiquidContentDigest3D(std::span<const std::uint8_t> bytes);
// 64 lowercase hex characters plus NUL; no allocation.
std::array<char, 65> LiquidDigestHex3D(const LiquidDigest3D& digest);
// Exact relative content name: blobs/<64 lowercase hex>.<flst|flpt|flrs|flmf>.
// The 76-byte result is NUL terminated. Invalid kinds return InvalidArgument.
absl::StatusOr<std::array<char, 76>> LiquidContentName3D(const LiquidContentReference3D& reference);
// Reject any spelling other than that derived name, including traversal, absolute
// paths, separators, case variants, extra suffixes and embedded NUL. No filesystem I/O.
absl::Status ValidateLiquidContentName3D(std::string_view name, const LiquidContentReference3D& reference);
// Verify kind, exact byte count and exact-byte digest, without parsing a record.
absl::Status VerifyLiquidContent3D(
    std::span<const std::uint8_t> bytes,
    const LiquidContentReference3D& reference,
    LiquidContentKind3D expected_kind,
    LiquidCacheLimits3D limits = {}
);

// All encoders/readers below are host-only setup/I/O-boundary operations, not
// physical-attempt operations. They return independent bounded owners, retain no
// input aliases and never modify inputs. Read spans require 8-byte alignment;
// unaligned input is rejected as DataLoss before generated accessors.
// Encoders use explicit little-endian v1
// fields and ForceDefaults; no native padding/capacity is copied. Readers first
// bound size and verify identifier/structure, then version/declared length, counts,
// enums, finite/range/order/ownership relations before allocating decoded records.
// Invalid arguments return InvalidArgument; corrupt wire bytes return DataLoss;
// unsupported version/model returns FailedPrecondition; configured bounds or
// allocation failure return ResourceExhausted. No partial output escapes on error.
// A reader alone does not authenticate bytes: 7B must call VerifyLiquidContent3D
// against a trusted manifest/index reference before decoding referenced content.
absl::StatusOr<LiquidCacheBytes3D> EncodeLiquidState3D(const LiquidState3D& state, LiquidCacheLimits3D limits = {});
absl::StatusOr<LiquidState3D> DecodeLiquidState3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits = {});
// Checkpoints require a completed-frame clock and enough resume capacities for
// every active record. Replay capacities are all zero or a valid strict growth.
absl::StatusOr<LiquidCacheBytes3D> EncodeLiquidCheckpoint3D(
    const LiquidState3D& state,
    const LiquidResumeMetadata3D& resume,
    LiquidCacheLimits3D limits = {}
);
absl::StatusOr<LiquidCheckpoint3D>
DecodeLiquidCheckpoint3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits = {});
absl::StatusOr<LiquidCacheBytes3D>
EncodeLiquidPointFrame3D(const LiquidPointFrame3D& frame, LiquidCacheLimits3D limits = {});
absl::StatusOr<LiquidPointFrame3D>
DecodeLiquidPointFrame3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits = {});
absl::StatusOr<LiquidCacheBytes3D>
EncodeLiquidFrameManifest3D(const LiquidFrameManifest3D& manifest, LiquidCacheLimits3D limits = {});
absl::StatusOr<LiquidFrameManifest3D>
DecodeLiquidFrameManifest3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits = {});
absl::StatusOr<LiquidCacheBytes3D>
EncodeLiquidResolvedScene3D(LiquidResolvedSceneView3D scene, LiquidCacheLimits3D limits = {});
absl::StatusOr<LiquidResolvedScene3D>
DecodeLiquidResolvedScene3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits = {});
// Domain-separated BLAKE3-256 of canonical ForceDefaults resolved bytes. Prefix is
// ASCII "fluidity/simulate/resolved/v1" followed by one NUL. Canonical encoding
// normalizes signed zero and inactive variant fields; positive infinity is allowed
// only for bulk modulus. Build/backend/model/seed and upstream bytes affect this
// fingerprint; local paths, heartbeat and capacity reservations cannot affect it.
absl::StatusOr<LiquidDigest3D>
ResolvedSceneFingerprint(LiquidResolvedSceneView3D scene, LiquidCacheLimits3D limits = {});
// Deterministic same-artifact execution key: domain "fluidity/simulate/execution/v1"
// plus NUL followed by the 32-byte resolved fingerprint. Never a process/job ID.
LiquidDigest3D LiquidExecutionId3D(const LiquidDigest3D& resolved_fingerprint);
}  // namespace fluidity
