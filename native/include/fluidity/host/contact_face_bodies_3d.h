#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fluidity/base/contact_face_bodies_3d.h"

namespace fluidity {

// Converts a face deposit result to the checked host status. A fifth body
// maps to ResourceExhausted and E-CONTACT-BODY-CAPACITY.
inline absl::Status ContactFaceDepositStatus(ContactFaceDepositStatus3D result) {
  switch (result) {
    case ContactFaceDepositStatus3D::kAccepted:
      return absl::OkStatus();
    case ContactFaceDepositStatus3D::kInvalid:
      return absl::InvalidArgumentError("invalid contact face contribution");
    case ContactFaceDepositStatus3D::kRange:
      return absl::OutOfRangeError("unrepresentable contact face body state");
    case ContactFaceDepositStatus3D::kBodyCapacity:
      return absl::ResourceExhaustedError("E-CONTACT-BODY-CAPACITY");
  }
  return absl::InternalError("unknown contact face deposit result");
}

struct ContactFaceParticle3D {
  std::uint64_t particle_id;  // Nonzero stable identity.
  std::uint64_t body_id;  // Nonzero stable body identity.
  double mass;  // Full particle mass in kg.
  double weight;  // Dimensionless weight to this face in (0, 1].
  Vec3 world_face_velocity;  // Particle velocity evaluated at the face in m/s.
};

struct ContactFaceParticleResponse3D {
  DistinctBodyContact3D face_contact;
  std::vector<Vec3> velocity_deltas;  // World m/s increments in input particle order.
};

// Deposits one face's contributors, resolves the detected pair and scatters
// the equal-and-opposite impulse through its contributing weights. Inputs
// are ordered by strictly increasing particle ID. Third-body deltas are
// zero. normal_toward_first points from the second body toward the first.
// Returns InvalidArgument for malformed inputs, OutOfRange for
// unrepresentable deposit or scatter, and ResourceExhausted with
// E-CONTACT-BODY-CAPACITY for overflow. Invalid or unresolved contact
// returns FailedPrecondition. Failure returns no updates.
absl::StatusOr<ContactFaceParticleResponse3D> ResolveContactFaceParticles3D(
    std::span<const ContactFaceParticle3D> particles,
    std::uint64_t first_body_id,
    std::uint64_t second_body_id,
    Vec3 normal_toward_first,
    double friction_coefficient
);

struct ContactParticleIdentity3D {
  std::uint64_t particle_id;  // Nonzero stable identity.
  std::uint64_t body_id;  // Nonzero stable body identity.
  double mass;  // Full particle mass in kg.
};

struct DetectedContactFace3D {
  std::size_t face_slot;  // Owned face slot in this contact stencil.
  std::span<const ContactFaceParticle3D> contributors;
  std::uint64_t first_body_id;
  std::uint64_t second_body_id;
  Vec3 normal_toward_first;  // World unit normal pointing toward the first body.
  double friction_coefficient;  // Finite nonnegative Coulomb coefficient.
};

struct ContactFaceBatchResponse3D {
  std::vector<DistinctBodyContact3D> face_contacts;  // In input face order.
  std::vector<Vec3> velocity_deltas;  // In input particle order, world m/s.
};

// Stages one detected pair per face using a single vector-contact stencil.
// Faces are strictly ordered by slot, and particles by stable ID. Each face
// contributor must match the particle table's body and mass. Contributor
// weights across this batch may total at most one per particle, within 1e-12.
// Each face uses its supplied incoming velocities. Outputs sum in face order
// and are published only on success. Status categories match the single-face
// resolver. Invalid batch ordering, identity or weights return InvalidArgument.
absl::StatusOr<ContactFaceBatchResponse3D> StageContactFaceBatch3D(
    std::span<const ContactParticleIdentity3D> particles,
    std::span<const DetectedContactFace3D> faces
);

}  // namespace fluidity
