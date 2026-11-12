#include "fluidity/host/contact_face_bodies_3d.h"

#include <cmath>

namespace fluidity {

absl::StatusOr<ContactFaceParticleResponse3D> ResolveContactFaceParticles3D(
    std::span<const ContactFaceParticle3D> particles,
    std::uint64_t first_body_id,
    std::uint64_t second_body_id,
    Vec3 normal_toward_first,
    double friction_coefficient
) {
  ContactFaceBodies3D face;
  std::uint64_t previous_particle_id = 0;
  for (const auto& particle : particles) {
    if (particle.particle_id <= previous_particle_id || particle.body_id == 0 || !std::isfinite(particle.mass)
        || particle.mass <= 0 || !std::isfinite(particle.weight) || particle.weight <= 0 || particle.weight > 1
        || !distinct_contact_detail::Finite(particle.world_face_velocity)) {
      return absl::InvalidArgumentError("invalid contact face particle or order");
    }
    const double weighted_mass = particle.weight * particle.mass;
    const Vec3 weighted_momentum = distinct_contact_detail::AddScaled({}, particle.world_face_velocity, weighted_mass);
    if (weighted_mass <= 0 || !distinct_contact_detail::Finite(weighted_momentum)) {
      return absl::OutOfRangeError("unrepresentable contact face contribution");
    }
    const auto deposited = ContactFaceDepositStatus(face.Deposit(particle.body_id, weighted_mass, weighted_momentum));
    if (!deposited.ok()) {
      return deposited;
    }
    previous_particle_id = particle.particle_id;
  }

  ContactFaceParticleResponse3D response{};
  if (!face.ResolvePair(
          first_body_id,
          second_body_id,
          normal_toward_first,
          friction_coefficient,
          response.face_contact
      )) {
    return absl::FailedPreconditionError("unresolved contact face pair");
  }
  double first_mass = 0;
  double second_mass = 0;
  for (int slot = 0; slot < face.size(); ++slot) {
    ContactBodySample3D sample{};
    if (!face.Read(slot, sample)) {
      return absl::InternalError("missing accepted contact face body");
    }
    if (sample.body_id == first_body_id) {
      first_mass = sample.mass;
    }
    if (sample.body_id == second_body_id) {
      second_mass = sample.mass;
    }
  }
  const Vec3 first_impulse = response.face_contact.impulse_on_first;
  const Vec3 second_impulse{-first_impulse.x, -first_impulse.y, -first_impulse.z};
  response.velocity_deltas.resize(particles.size());
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto& particle = particles[index];
    if (particle.body_id != first_body_id && particle.body_id != second_body_id) {
      continue;
    }
    const bool first = particle.body_id == first_body_id;
    if (!ScatterContactFaceImpulse3D(
            first ? first_impulse : second_impulse,
            particle.weight,
            first ? first_mass : second_mass,
            response.velocity_deltas[index]
        )) {
      return absl::OutOfRangeError("unrepresentable scattered contact impulse");
    }
  }
  return response;
}

}  // namespace fluidity
