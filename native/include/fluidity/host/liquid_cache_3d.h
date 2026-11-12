#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include "fluidity/host/liquid_substep_3d.h"
#include "fluidity/host/solid_interaction_checkpoint_3d.h"

namespace fluidity {
// BLAKE3-256, raw digest bytes in algorithm order (not native integer words).
using LiquidDigest3D = std::array<std::uint8_t, 32>;
using LiquidCacheBytes3D = LiquidRecords3D<std::uint8_t>;
enum class LiquidContentKind3D : std::uint8_t {
  kState = 0,
  kPoints = 1,
  kResolvedScene = 2,
  kManifest = 3,
  kCoupledState = 4,
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
  LiquidDigest3D execution_id{};  // Must equal LiquidExecutionId3D(resolved_fingerprint).
  LiquidStateCapacities3D capacities;
  LiquidStateCapacities3D capacity_replay_reservations;
  bool operator==(const LiquidResumeMetadata3D&) const = default;
};
struct LiquidCheckpoint3D {
  LiquidState3D state;
  LiquidResumeMetadata3D resume;
  std::optional<SolidInteractionCheckpoint3D> solid;  // Present for a resolved mixed scene.
};
struct LiquidPoint3D {
  std::uint64_t id = 0;  // Stable nonzero particle ID, unique within this frame.
  std::uint32_t tag = 0;  // Packed substance/phase/flags, matching LiquidParticleState3D.
  Vec3 world_position{};  // m.
  Vec3 world_velocity{};  // Stored velocity, m/s.
  double radius = 1;  // Positive visualization radius, m.
  double volume_ratio = 1;  // Dimensionless J.
  std::uint8_t level = 0;  // Uniform M1 admits level zero only.
  std::uint64_t phase_id = 0;  // Stable phase when the active composition prefix is nonempty.
  std::uint8_t constituent_count = 0;
  std::array<LiquidConstituentFraction3D, kMaximumLiquidParticleConstituents3D> constituents{};
};
struct LiquidPointFrame3D {
  LiquidClock3D clock;  // Completed frame (substep zero).
  LiquidRecords3D<LiquidPoint3D> points;  // Stable physical particle order.
};
struct LiquidFrameManifest3D {
  LiquidClock3D clock;
  LiquidDigest3D resolved_fingerprint{};
  LiquidDigest3D execution_id{};  // Must equal LiquidExecutionId3D(resolved_fingerprint).
  LiquidContentReference3D points;  // Complete only; checkpoint retention has its own index.
};
struct LiquidUpstreamContent3D {
  LiquidUpstreamRole3D role = LiquidUpstreamRole3D::kInitialParticles;
  std::uint64_t ordinal = 0;
  std::uint64_t byte_length = 0;
  LiquidDigest3D digest{};  // Exact upstream artifact bytes, never a local filename.
};
// One authored width demand authenticated against a specific accepted state.
// The public FLRS route supplies no coarsening halo.
struct LiquidAdaptiveWidthAction3D {
  LiquidClock3D accepted_clock{};
  LiquidDigest3D accepted_state_digest{};  // Canonical FLST bytes.
  AdaptiveMacCellKey3D leaf{};
  AdaptiveWidthDemand3D demand{};
  double maximum_step_duration_s = 0;  // Evidence covers this duration and shorter retries.
};
inline constexpr std::uint32_t kLiquidOnlyResolvedModel3D = 1;
inline constexpr std::uint32_t kPlasticBodyResolvedModel3D = 2;

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
  std::uint32_t model_version = kLiquidOnlyResolvedModel3D;
  LiquidDigest3D build_digest{};  // Content identity of the numerical build artifact.
  std::span<const LiquidUpstreamContent3D> upstream;
  // Required in model 2, ordered by body ID. The codec owns the parameters, while execution is gated.
  std::span<const PlasticSolidCheckpointBinding3D> plastic_bodies;
  std::span<const LiquidAdaptiveWidthAction3D> adaptive_width_actions{};  // Strictly increasing accepted epoch.
  std::uint32_t adaptive_width_cell_capacity = 0;
  std::uint32_t adaptive_width_face_capacity = 0;
};
// Owning decoded scene; View() binds spans to this owner's arrays. Reacquire after
// move; a view cannot outlive this object. No pointer into encoded bytes survives.
struct LiquidResolvedScene3D {
  UniformMacGeometry3D geometry{{0, 0, 0}, 1};
  std::uint64_t random_seed = 0;
  LiquidSceneInputs3D scene;  // Borrowed arrays remain empty; use View().scene.
  LiquidStepOptions3D options;  // Capacity plans remain zero; read them from checkpoint.
  LiquidBackend3D backend = LiquidBackend3D::kCpu;
  std::uint32_t model_version = kLiquidOnlyResolvedModel3D;
  LiquidDigest3D build_digest{};
  LiquidRecords3D<LiquidEmissionSource3D> sources;
  LiquidRecords3D<LiquidConstituentMaterial3D> constituent_materials;
  LiquidRecords3D<LiquidPhasePair3D> phase_pairs;
  LiquidRecords3D<LiquidUpstreamContent3D> upstream;
  LiquidRecords3D<PlasticSolidCheckpointBinding3D> plastic_bodies;
  LiquidRecords3D<LiquidAdaptiveWidthAction3D> adaptive_width_actions;
  std::uint32_t adaptive_width_cell_capacity = 0;
  std::uint32_t adaptive_width_face_capacity = 0;
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
// Verifies kind, exact byte count and digest without parsing the record. An invalid or mismatched
// kind returns InvalidArgument, an excessive byte count returns ResourceExhausted, and a content
// length or digest mismatch returns DataLoss.
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
// unsupported version/model returns FailedPrecondition. Encoders propagate
// OutOfRange for unrepresentable numerical state. Configured bounds or allocation
// failure return ResourceExhausted. No partial output escapes on error.
// Authenticate referenced content with VerifyLiquidContent3D against the trusted manifest or
// checkpoint reference before decoding.
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
// Exports a checked completed state in stable particle order. Copies ID/tag,
// world position/velocity and J, with level zero and equivalent-volume radius
// cbrt(3*V0*J/(4*pi)) using the M1 numerical expression/order. Requires a completed
// clock. Invalid state/clock returns InvalidArgument, record/allocation bounds
// ResourceExhausted and unrepresentable radius OutOfRange. No input mutation or
// borrowed aliases; failures return no partial point frame.
absl::StatusOr<LiquidPointFrame3D> BuildLiquidPointFrame3D(const LiquidState3D& state, LiquidCacheLimits3D limits = {});
// Model-2 point frames append snow in checkpoint particle order. Snow IDs and
// tags set the high bit; the remaining bits hold the original ID and material
// ID. Liquid IDs and tags keep the high bit clear. The total volume ratio is
// det(Fe) * Jp. Inputs are an accepted liquid and snow pair at a frame boundary.
inline constexpr std::uint64_t kSnowPointIdBit = std::uint64_t{1} << 63;
inline constexpr std::uint32_t kSnowPointTagBit = std::uint32_t{1} << 31;
absl::StatusOr<LiquidPointFrame3D> BuildCoupledPointFrame3D(
    const LiquidState3D& state,
    const SolidInteractionCheckpoint3D& solid,
    LiquidCacheLimits3D limits = {}
);
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

// Cancellation/failure boundaries occur after real I/O, except BeforeManifest and
// BeforePrune. The callback borrows context for the call; it must not throw.
enum class LiquidCachePublishStage3D {
  kTemporaryWritten,
  kTemporaryFlushed,
  kContentPublished,
  kInspectionPublished,
  kBeforeManifest,
  kManifestPublished,
  kCheckpointPublished,
  kBeforePrune,
};
struct LiquidCachePublishControl3D {
  absl::Status (*check)(void*, LiquidCachePublishStage3D) = nullptr;
  void* context = nullptr;
};
struct LiquidCommittedFrame3D {
  LiquidFrameManifest3D manifest;
  LiquidPointFrame3D points;
};

// Host I/O boundary. root_directory is a trusted, exclusively cache-owned local
// directory, captured as an absolute directory at open, never a parsed record path.
// Its ancestors must not be renamed
// or modified concurrently. Child symlinks/reparse points are rejected. All record
// names are derived from checked integer ordinals or exact content references.
// Readers and writers take a nonblocking native exclusive lease (Aborted if busy);
// process exit releases the lease, even when its harmless lock file remains.
//
// Create/Open validate the resolved scene and expected job fingerprint. Invalid
// caller data is InvalidArgument, incompatible job identity FailedPrecondition,
// missing records NotFound, corrupt bytes/relationships DataLoss, conflicts
// AlreadyExists, bounds/OOM ResourceExhausted, and native I/O failures Unavailable.
// Owners and decoded records are returned only on success.
class LiquidCache3D {
 public:
  static absl::StatusOr<LiquidCache3D> Create(
      const std::filesystem::path& root_directory,
      LiquidResolvedSceneView3D resolved_scene,
      LiquidCacheLimits3D limits = {}
  );
  /// Opens a publication-capable cache after verifying canonical resolved bytes against the supplied fingerprint.
  /// Acquiring the cache I/O lease may create the persistent lock file.
  static absl::StatusOr<LiquidCache3D> Open(
      const std::filesystem::path& root_directory,
      const LiquidDigest3D& expected_resolved_fingerprint,
      LiquidCacheLimits3D limits = {}
  );
  /// Opens an existing cache for read-only replay. The caller establishes build and backend identity.
  /// Performs portable inspection using the existing `.writer.lock` file, then verifies the cache fingerprint
  /// against expected_resolved_fingerprint. Propagates inspection failures and returns `FailedPrecondition` on
  /// mismatch.
  /// @return On success, a cache that permits verified retained checkpoint reads and creates or writes no files.
  /// PublishFrame and PublishCheckpoint return `FailedPrecondition` before accessing the cache, even if the
  /// lock file has since been removed.
  static absl::StatusOr<LiquidCache3D> OpenForReplay(
      const std::filesystem::path& root_directory,
      const LiquidDigest3D& expected_resolved_fingerprint,
      LiquidCacheLimits3D limits = {}
  );
  // Verifies portable cache self-consistency without creating or writing files.
  // Requires an existing lease file. Inspection instances reject PublishFrame,
  // PublishCheckpoint and ReadCheckpoint with FailedPrecondition. Open and
  // OpenForReplay authenticate a fingerprint for checkpoint reads. Create and
  // Open authorize publication.
  static absl::StatusOr<LiquidCache3D>
  OpenForInspection(const std::filesystem::path& root_directory, LiquidCacheLimits3D limits = {});
  // Returns owning canonical resolved inputs after exact content/job verification.
  absl::StatusOr<LiquidResolvedScene3D> ReadResolvedScene() const;
  /// Publishes a committed point frame and, at frame zero and multiples of ten, its restart checkpoint.
  /// Frame zero requires time, accepted step and epoch zero. Its checkpoint metadata must commit before
  /// any later frame. Checkpoint clock, geometry, seed and exported particle fields must match the point
  /// frame and resolved scene. Resume metadata must identify this cache's job.
  /// @param points Point frame with a completed-frame clock (substep zero).
  /// @param checkpoint Required state for a checkpoint frame. Null for other frames.
  /// @param resume Required resume metadata for a checkpoint frame. Null for other frames.
  /// @param control Optional cancellation and failure callback. All inputs are borrowed through this call.
  /// @param solid Required with a checkpoint for a resolved mixed scene.
  /// @return OK when frame publication and any required checkpoint retention finish. Read-only caches return
  /// FailedPrecondition before cache access. A conflicting manifest or checkpoint reference returns AlreadyExists.
  /// Verified identical bytes are reused.
  /// @note The cache flushes and publishes the content and JSON inspection mirror before committing the point frame
  /// with the binary manifest. Publication uses no-replace promotion.
  /// Checkpoint reference publication and pruning follow the manifest commit. A failure can leave the
  /// point frame readable through ReadCommittedFrame, with checkpoint publication repairable through
  /// PublishCheckpoint. Earlier point frames remain intact.
  absl::Status PublishFrame(
      const LiquidPointFrame3D& points,
      const LiquidState3D* checkpoint = nullptr,
      const LiquidResumeMetadata3D* resume = nullptr,
      LiquidCachePublishControl3D control = {},
      const SolidInteractionCheckpoint3D* solid = nullptr
  ) const;
  /// Reads a committed point frame by ordinal, independently of restart checkpoint retention.
  /// Verifies the resolved scene, binary manifest, deterministic JSON inspection mirror and exact
  /// point content. The manifest and point frame must share a completed-frame clock.
  /// @return The owning manifest and point data on success. Missing required records return NotFound.
  /// Corrupt records or inconsistent clocks return DataLoss.
  absl::StatusOr<LiquidCommittedFrame3D> ReadCommittedFrame(std::uint64_t frame_ordinal) const;
  // Repairs/completes optional checkpoint metadata after a frame has committed.
  // Only frame zero and multiples of ten qualify. Must match the verified frame.
  // A resolved mixed scene requires the solid state from the same accepted step.
  // Keeps the newest two full-state blobs; immutable reference history remains.
  // The frame-zero payload remains until two later checkpoints are available.
  // Older checkpoint payloads are pruned, but all point frames remain readable.
  absl::Status PublishCheckpoint(
      const LiquidState3D& checkpoint,
      const LiquidResumeMetadata3D& resume,
      LiquidCachePublishControl3D control = {},
      const SolidInteractionCheckpoint3D* solid = nullptr
  ) const;
  /// Reads the newest eligible checkpoint at or before the inclusive `maximum_frame_ordinal` cutoff.
  /// Validates checkpoint references against their committed manifests and determines the globally newest two
  /// reference ordinals before applying the cutoff. Older references and unreferenced blobs are ineligible.
  /// @return On success, verified owning state and resume metadata matching the committed frame and resolved scene.
  /// Returns `FailedPrecondition` on an inspection-only cache and `NotFound` if no retained reference meets the
  /// cutoff. Missing required files can also return `NotFound`. A content digest mismatch or a checkpoint
  /// consistency mismatch returns `DataLoss`. A failure after reference selection ends the read without trying
  /// an older checkpoint.
  absl::StatusOr<LiquidCheckpoint3D> ReadCheckpoint(
      std::uint64_t maximum_frame_ordinal = std::numeric_limits<std::uint64_t>::max()
  ) const;

 private:
  std::filesystem::path root_directory_;
  LiquidDigest3D fingerprint_{};
  /// Expected kind, byte length and digest of the canonical resolved scene, captured at creation or open.
  LiquidContentReference3D resolved_reference_;
  LiquidCacheLimits3D limits_;
  /// When true, this cache verified portable self-consistency and disallows checkpoint reads.
  bool inspection_only_ = false;
  /// Portable inspection and replay require an existing cache I/O lease file and disallow publication.
  bool read_only_ = false;
};
}  // namespace fluidity
