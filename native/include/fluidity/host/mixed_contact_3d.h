#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "fluidity/host/contact_face_bodies_3d.h"
#include "fluidity/host/liquid_state_3d.h"
#include "fluidity/host/plastic_solid_checkpoint_3d.h"

namespace fluidity {

// Applies detected contact to one liquid body's and any plastic bodies'
// candidate stored velocities in world m/s. Particle IDs must be globally
// unique across the supplied states. Liquid uses liquid_body_id. Each
// plastic record retains its own body ID, distinct from the liquid body.
// The two writable spans must be disjoint from each other and all face data.
// Returns face impulse receipts in face-slot order. Failure leaves both spans
// unchanged, with the same status categories as StageContactFaceBatch3D.
// An unrepresentable final velocity returns OutOfRange. The caller owns the
// surrounding accepted step and handles pressure and particle advection.
absl::StatusOr<std::vector<DistinctBodyContact3D>> ApplyMixedContactFaces3D(
    std::span<LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    std::span<const DetectedContactFace3D> faces
);

}  // namespace fluidity
