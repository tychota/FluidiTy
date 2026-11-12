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

struct PlanarMixedContactFace3D {
  DetectedContactFace3D pair;  // Candidate, admitted only when the sweep hits.
  Vec3 first_interface_world_position;  // Sample on the first body's interface, meters.
  Vec3 first_interface_world_velocity;  // Constant interface velocity in m/s over the attempt.
  double second_plane_start_coordinate;  // Dot(pair.normal_toward_first, second plane point), meters.
  Vec3 second_interface_world_velocity;  // Constant plane velocity in world m/s.
};

struct PlanarMixedContactReceipt3D {
  std::size_t face_slot;
  double contact_fraction;  // Earliest hit fraction in [0, 1].
  DistinctBodyContact3D contact;  // Two-body response using supplied face contributors.
};

// Sweeps supplied locally planar interfaces before applying their detected
// face pairs. Uses certified constant-velocity halfspace contact with zero
// initial penetration tolerance. Candidate slots must be strictly ordered.
// Only hits contribute impulses. The velocities used for the sweep are
// reconstructed interface motion; impulses use the supplied face samples.
// Invalid duration, geometry, ordering or overlapping writable views return
// InvalidArgument or the underlying sweep status. Unresolvable contact or
// scatter errors propagate. Any failure leaves both material spans unchanged.
// This updates candidate velocities only. Geometry reconstruction, position
// advancement after impact and accepted-step publication belong to the caller.
absl::StatusOr<std::vector<PlanarMixedContactReceipt3D>> ApplyPlanarMixedContactFaces3D(
    std::span<LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    std::span<const PlanarMixedContactFace3D> candidates,
    double duration
);

}  // namespace fluidity
