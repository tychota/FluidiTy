#include "fluidity/host/mixed_contact_3d.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "fluidity/host/liquid_wall_contact_3d.h"
#include "fluidity/host/views.h"

namespace fluidity {
namespace {

struct ContactVelocityBinding {
  ContactParticleIdentity3D identity;
  Vec3* stored_world_velocity;
};

}  // namespace

absl::StatusOr<std::vector<DistinctBodyContact3D>> ApplyMixedContactFaces3D(
    std::span<LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    std::span<const DetectedContactFace3D> faces
) {
  if (liquid_body_id == 0) {
    return absl::InvalidArgumentError("zero liquid contact body ID");
  }
  const ByteView liquid_bytes = std::as_bytes(liquid);
  const ByteView plastic_bytes = std::as_bytes(plastic);
  const auto disjoint = CheckWritableViews({liquid_bytes, plastic_bytes}, {std::as_bytes(faces)});
  if (!disjoint.ok()) {
    return disjoint;
  }
  for (const auto& face : faces) {
    const ByteView contributor_bytes = std::as_bytes(face.contributors);
    if (ByteViewsOverlap(liquid_bytes, contributor_bytes) || ByteViewsOverlap(plastic_bytes, contributor_bytes)) {
      return absl::InvalidArgumentError("mixed contact contributors alias output");
    }
  }

  std::vector<ContactVelocityBinding> bindings;
  bindings.reserve(liquid.size() + plastic.size());
  for (auto& particle : liquid) {
    bindings.push_back({{particle.id, liquid_body_id, particle.mass}, &particle.world_velocity});
  }
  for (auto& particle : plastic) {
    if (particle.body_id == liquid_body_id) {
      return absl::InvalidArgumentError("plastic particle belongs to the liquid body");
    }
    bindings.push_back({
        {particle.particle_id, particle.body_id, particle.transfer.particle_mass},
        &particle.transfer.particle_velocity,
    });
  }

  // Liquid records are in transfer sort order. Build a separate ID order
  // without changing the order stored by either checkpoint family.
  std::ranges::sort(bindings, {}, [](const auto& binding) {
    return binding.identity.particle_id;
  });
  std::vector<ContactParticleIdentity3D> identities;
  identities.reserve(bindings.size());
  for (const auto& binding : bindings) {
    if (!distinct_contact_detail::Finite(*binding.stored_world_velocity)) {
      return absl::InvalidArgumentError("nonfinite incoming stored contact velocity");
    }
    identities.push_back(binding.identity);
  }
  auto response = StageContactFaceBatch3D(identities, faces);
  if (!response.ok()) {
    return response.status();
  }

  // Convert deltas to final candidates before publishing any velocity.
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    const Vec3 incoming = *bindings[index].stored_world_velocity;
    auto& candidate = response->velocity_deltas[index];
    candidate = {
        candidate.x == 0 ? incoming.x : incoming.x + candidate.x,
        candidate.y == 0 ? incoming.y : incoming.y + candidate.y,
        candidate.z == 0 ? incoming.z : incoming.z + candidate.z,
    };
    if (!distinct_contact_detail::Finite(candidate)) {
      return absl::OutOfRangeError("unrepresentable stored mixed-contact velocity");
    }
  }
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    *bindings[index].stored_world_velocity = response->velocity_deltas[index];
  }
  return std::move(response->face_contacts);
}

absl::StatusOr<std::vector<PlanarMixedContactReceipt3D>> ApplyPlanarMixedContactFaces3D(
    std::span<LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    std::span<const PlanarMixedContactFace3D> candidates,
    double duration
) {
  if (!std::isfinite(duration) || duration <= 0) {
    return absl::InvalidArgumentError("invalid planar mixed contact duration");
  }
  const auto disjoint =
      CheckWritableViews({std::as_bytes(liquid), std::as_bytes(plastic)}, {std::as_bytes(candidates)});
  if (!disjoint.ok()) {
    return disjoint;
  }
  std::vector<DetectedContactFace3D> detected;
  std::vector<PlanarMixedContactReceipt3D> receipts;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const auto& candidate = candidates[index];
    const auto& pair = candidate.pair;
    if (index > 0 && pair.face_slot <= candidates[index - 1].pair.face_slot) {
      return absl::InvalidArgumentError("invalid planar mixed contact face order");
    }
    const LiquidWallContactInputs3D sweep{
        .start_world_position = candidate.first_interface_world_position,
        .stored_world_velocity = candidate.first_interface_world_velocity,
        .advecting_world_velocity = candidate.first_interface_world_velocity,
        .obstacle =
            TranslatingHalfspace3D{
                pair.normal_toward_first,
                candidate.second_plane_start_coordinate,
                candidate.second_interface_world_velocity,
            },
        .step_duration = duration,
    };
    const auto hit = ResolveLiquidWallContact3D(sweep);
    if (!hit.ok()) {
      return hit.status();
    }
    if (hit->contact_fraction.has_value()) {
      detected.push_back(pair);
      receipts.push_back({.face_slot = pair.face_slot, .contact_fraction = *hit->contact_fraction, .contact = {}});
    }
  }
  const auto applied = ApplyMixedContactFaces3D(liquid, liquid_body_id, plastic, detected);
  if (!applied.ok()) {
    return applied.status();
  }
  for (std::size_t index = 0; index < receipts.size(); ++index) {
    receipts[index].contact = (*applied)[index];
  }
  return receipts;
}

}  // namespace fluidity
