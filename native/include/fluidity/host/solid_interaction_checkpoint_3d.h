#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/liquid_wall_contact_3d.h"
#include "fluidity/host/rigid_body_3d.h"

namespace fluidity {

struct RigidBodyHistory3D {
  RigidBody3D body;
  RigidBodyPose3D previous_pose;
};

struct ElasticParticleBladeSide3D {
  std::uint64_t particle_id;
  ElasticBladeSide3D side;
};

// Accepted interaction records. Continuum particles and scene inputs are held
// by the caller. Records are strictly ordered by nonzero body or particle ID.
struct SolidInteractionCheckpoint3D {
  std::uint64_t accepted_step;
  std::vector<RigidBodyHistory3D> bodies;
  std::vector<ElasticParticleBladeSide3D> sides;
};

// Encodes accepted interaction records as size-prefixed FLSI v1 bytes.
// Invalid state returns InvalidArgument. More than 2^16 records in either
// family returns ResourceExhausted. Caller state is immutable.
absl::StatusOr<std::vector<std::uint8_t>> EncodeSolidInteractionCheckpoint3D(const SolidInteractionCheckpoint3D& state);

// Verifies framing, schema and state before returning reconstructed records.
// The byte span must be aligned to alignof(std::uint64_t).
// Invalid bytes or state return InvalidArgument. Unsupported version returns
// FailedPrecondition. Record limits match EncodeSolidInteractionCheckpoint3D.
absl::StatusOr<SolidInteractionCheckpoint3D> DecodeSolidInteractionCheckpoint3D(std::span<const std::uint8_t> bytes);

}  // namespace fluidity
