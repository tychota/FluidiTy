#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/base/mac_transfer_types_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/liquid_wall_contact_3d.h"
#include "fluidity/host/plastic_solid_checkpoint_3d.h"
#include "fluidity/host/rigid_body_3d.h"

namespace fluidity {

struct LiquidCacheLimits3D;
struct LiquidResolvedSceneView3D;
struct LiquidResumeMetadata3D;

struct RigidBodyHistory3D {
  RigidBody3D body;
  RigidBodyPose3D previous_pose;
};

struct ElasticParticleBladeSide3D {
  std::uint64_t particle_id;
  ElasticBladeSide3D side;
};

struct ElasticCheckpointParticle3D {
  std::uint64_t particle_id;  // Stable nonzero particle identity.
  std::uint64_t body_id;  // Stable identity of the elastic body.
  MacTransferParticle3D transfer;
  double reference_volume;  // Cubic meters.
  Mat3 deformation_gradient;  // Dimensionless, at accepted time.
};

// Accepted interaction records. Scene inputs are held by the caller. Records
// are strictly ordered by nonzero body or particle ID. Legacy checkpoints
// may omit elastic and plastic particle records.
struct SolidInteractionCheckpoint3D {
  std::uint64_t accepted_step;
  std::vector<RigidBodyHistory3D> bodies;
  std::vector<ElasticParticleBladeSide3D> sides;
  std::vector<ElasticCheckpointParticle3D> elastic_particles{};
  std::vector<PlasticSolidCheckpointParticle3D> plastic_particles{};
};

// Encodes accepted interaction records as size-prefixed FLSI v1 or v2 bytes.
// A nonempty plastic payload uses v2 and requires resolved bindings.
// Invalid state returns InvalidArgument. More than 2^16 records in each
// family returns ResourceExhausted. Caller state is immutable.
absl::StatusOr<std::vector<std::uint8_t>> EncodeSolidInteractionCheckpoint3D(
    const SolidInteractionCheckpoint3D& state,
    std::span<const PlasticSolidCheckpointBinding3D> bindings = {}
);

// Verifies framing, schema and state before returning reconstructed records.
// The byte span must be aligned to alignof(std::uint64_t).
// Invalid bytes or state return InvalidArgument. Unsupported version returns
// FailedPrecondition. Record limits match EncodeSolidInteractionCheckpoint3D.
absl::StatusOr<SolidInteractionCheckpoint3D> DecodeSolidInteractionCheckpoint3D(
    std::span<const std::uint8_t> bytes,
    std::span<const PlasticSolidCheckpointBinding3D> bindings = {}
);

struct CoupledInteractionCheckpoint3D {
  LiquidState3D liquid;
  SolidInteractionCheckpoint3D solid;
};

// Encodes liquid and solid accepted state in one FLSI record. Their accepted
// step ordinals must match. Uses the liquid codec's validation and a 32 MiB
// envelope limit. Resolved scene settings remain the caller's responsibility.
absl::StatusOr<std::vector<std::uint8_t>> EncodeCoupledInteractionCheckpoint3D(
    const LiquidState3D& liquid,
    const SolidInteractionCheckpoint3D& solid,
    std::span<const PlasticSolidCheckpointBinding3D> bindings = {}
);

// Decodes an aligned FLSI envelope with both states present, rejecting a
// missing liquid state or mismatched accepted step ordinal.
absl::StatusOr<CoupledInteractionCheckpoint3D> DecodeCoupledInteractionCheckpoint3D(
    std::span<const std::uint8_t> bytes,
    std::span<const PlasticSolidCheckpointBinding3D> bindings = {}
);

// Writes FLSI v3 for a resolved mixed scene. The checkpoint binds the exact
// canonical scene fingerprint and takes plastic bindings from that scene.
// The liquid geometry, seed and accepted step must agree with the scene and
// solid payload. Failure returns no bytes.
absl::StatusOr<std::vector<std::uint8_t>> EncodeResolvedCoupledCheckpoint3D(
    const LiquidResolvedSceneView3D& scene,
    const LiquidState3D& liquid,
    const SolidInteractionCheckpoint3D& solid
);

// Reads FLSI v3 and requires the expected resolved mixed scene fingerprint.
// A different fingerprint returns FailedPrecondition. Legacy envelopes
// require the original unbound decoder.
absl::StatusOr<CoupledInteractionCheckpoint3D>
DecodeResolvedCoupledCheckpoint3D(const LiquidResolvedSceneView3D& scene, std::span<const std::uint8_t> bytes);

// Includes the liquid scheduler's resume metadata in the nested state.
// Its fingerprint and execution identity must match the resolved scene.
absl::StatusOr<std::vector<std::uint8_t>> EncodeResolvedCoupledCheckpoint3D(
    const LiquidResolvedSceneView3D& scene,
    const LiquidState3D& liquid,
    const SolidInteractionCheckpoint3D& solid,
    const LiquidResumeMetadata3D& resume,
    const LiquidCacheLimits3D& limits
);

// Requires a nested liquid checkpoint with resume metadata and publishes
// that metadata only after the complete scene-bound checkpoint validates.
absl::StatusOr<CoupledInteractionCheckpoint3D> DecodeResolvedCoupledCheckpoint3D(
    const LiquidResolvedSceneView3D& scene,
    std::span<const std::uint8_t> bytes,
    LiquidResumeMetadata3D& resume,
    const LiquidCacheLimits3D& limits
);

// Loads the canonical unbound FLSI initial artifact named by a resolved
// mixed scene. Exact upstream bytes, time-zero clock, geometry, seed,
// source counters, and plastic bindings must agree. Failure returns no state.
absl::StatusOr<CoupledInteractionCheckpoint3D> DecodeResolvedCoupledInitialState3D(
    const LiquidResolvedSceneView3D& scene,
    std::span<const std::uint8_t> bytes,
    const LiquidCacheLimits3D& limits
);

}  // namespace fluidity
