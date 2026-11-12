#include "fluidity/host/contact_face_bodies_3d.h"

#include <algorithm>
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

absl::StatusOr<ContactFaceBatchResponse3D> StageContactFaceBatch3D(
    std::span<const ContactParticleIdentity3D> particles,
    std::span<const DetectedContactFace3D> faces
) {
  std::uint64_t previous_particle_id = 0;
  for (const auto& particle : particles) {
    if (particle.particle_id <= previous_particle_id || particle.body_id == 0 || !std::isfinite(particle.mass)
        || particle.mass <= 0) {
      return absl::InvalidArgumentError("invalid contact batch particle or order");
    }
    previous_particle_id = particle.particle_id;
  }

  constexpr double kWeightTolerance = 1e-12;
  ContactFaceBatchResponse3D batch;
  batch.face_contacts.reserve(faces.size());
  batch.velocity_deltas.resize(particles.size());
  std::vector<double> particle_weight_sums(particles.size());
  for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
    const auto& face = faces[face_index];
    if (face_index > 0 && face.face_slot <= faces[face_index - 1].face_slot) {
      return absl::InvalidArgumentError("invalid contact batch face order");
    }
    const auto response = ResolveContactFaceParticles3D(
        face.contributors,
        face.first_body_id,
        face.second_body_id,
        face.normal_toward_first,
        face.friction_coefficient
    );
    if (!response.ok()) {
      return response.status();
    }

    for (std::size_t contributor_index = 0; contributor_index < face.contributors.size(); ++contributor_index) {
      const auto& contributor = face.contributors[contributor_index];
      const auto particle =
          std::ranges::lower_bound(particles, contributor.particle_id, {}, &ContactParticleIdentity3D::particle_id);
      if (particle == particles.end() || particle->particle_id != contributor.particle_id
          || particle->body_id != contributor.body_id || particle->mass != contributor.mass) {
        return absl::InvalidArgumentError("face contributor does not match contact batch particle");
      }
      const auto particle_index = static_cast<std::size_t>(particle - particles.begin());
      particle_weight_sums[particle_index] += contributor.weight;
      if (particle_weight_sums[particle_index] > 1 + kWeightTolerance) {
        return absl::InvalidArgumentError("contact batch weights exceed unity");
      }
      auto& velocity_delta = batch.velocity_deltas[particle_index];
      velocity_delta =
          distinct_contact_detail::AddScaled(velocity_delta, response->velocity_deltas[contributor_index], 1);
      if (!distinct_contact_detail::Finite(velocity_delta)) {
        return absl::OutOfRangeError("unrepresentable contact batch velocity delta");
      }
    }
    batch.face_contacts.push_back(response->face_contact);
  }
  return batch;
}

}  // namespace fluidity
