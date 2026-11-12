#include "fluidity/host/mixed_contact_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <utility>

#include "fluidity/host/liquid_wall_contact_3d.h"
#include "fluidity/host/views.h"

namespace fluidity {
namespace {

struct ContactVelocityBinding {
  ContactParticleIdentity3D identity;
  Vec3* stored_world_velocity;
};

absl::Status
CheckPlanarContactViews(ByteView liquid, ByteView plastic, std::span<const PlanarMixedContactFace3D> candidates) {
  const auto disjoint = CheckWritableViews({liquid, plastic}, {std::as_bytes(candidates)});
  if (!disjoint.ok()) {
    return disjoint;
  }
  for (const auto& candidate : candidates) {
    const auto contributors = std::as_bytes(candidate.pair.contributors);
    if (ByteViewsOverlap(liquid, contributors) || ByteViewsOverlap(plastic, contributors)) {
      return absl::InvalidArgumentError("planar mixed contact contributors alias output");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Vec3> AdvanceAtCommonImpact(
    Vec3 position,
    Vec3 before,
    Vec3 after,
    double before_duration,
    double after_duration,
    double maximum_path_length
) {
  const double path_length = before_duration * std::hypot(before.x, before.y, before.z)
      + after_duration * std::hypot(after.x, after.y, after.z);
  if (!std::isfinite(path_length)) {
    return absl::OutOfRangeError("nonfinite planar mixed contact path");
  }
  if (path_length > maximum_path_length) {
    return absl::FailedPreconditionError("planar mixed contact path exceeds motion bound");
  }
  const Vec3 impact_position = distinct_contact_detail::AddScaled(position, before, before_duration);
  const Vec3 final_position = distinct_contact_detail::AddScaled(impact_position, after, after_duration);
  if (!distinct_contact_detail::Finite(impact_position) || !distinct_contact_detail::Finite(final_position)) {
    return absl::OutOfRangeError("nonfinite planar mixed contact position");
  }
  return final_position;
}

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

absl::StatusOr<PlanarMixedContactFace3D> ReconstructAxisAlignedMixedContactFace3D(
    std::span<const LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<const PlasticSolidCheckpointParticle3D> plastic,
    const DetectedContactFace3D& pair,
    double first_interface_offset,
    double second_interface_offset
) {
  const bool first_is_liquid = pair.first_body_id == liquid_body_id;
  const bool second_is_liquid = pair.second_body_id == liquid_body_id;
  if (liquid_body_id == 0 || pair.first_body_id == 0 || pair.second_body_id == 0 || first_is_liquid == second_is_liquid
      || !std::isfinite(first_interface_offset) || first_interface_offset < 0 || !std::isfinite(second_interface_offset)
      || second_interface_offset < 0) {
    return absl::InvalidArgumentError("invalid planar mixed contact reconstruction");
  }
  constexpr double Vec3::* kComponents[]{&Vec3::x, &Vec3::y, &Vec3::z};
  double Vec3::* normal_component = nullptr;
  for (const auto component : kComponents) {
    const double value = pair.normal_toward_first.*component;
    if (value == 0) {
      continue;
    }
    if (normal_component != nullptr || std::abs(value) != 1) {
      return absl::InvalidArgumentError("planar mixed contact requires a signed Cartesian normal");
    }
    normal_component = component;
  }
  if (normal_component == nullptr) {
    return absl::InvalidArgumentError("zero planar mixed contact normal");
  }
  const double sign = pair.normal_toward_first.*normal_component;

  struct FrontSample {
    Vec3 position;
    Vec3 velocity;
    double coordinate;
    bool ambiguous;
  };
  std::optional<FrontSample> first_front;
  std::optional<FrontSample> second_front;
  std::uint64_t previous_id = 0;
  for (const auto& contributor : pair.contributors) {
    if (contributor.particle_id <= previous_id) {
      return absl::InvalidArgumentError("invalid planar mixed contact contributor order");
    }
    previous_id = contributor.particle_id;
    const bool first = contributor.body_id == pair.first_body_id;
    if (!first && contributor.body_id != pair.second_body_id) {
      continue;
    }

    Vec3 position{};
    if (contributor.body_id == liquid_body_id) {
      const auto particle = std::ranges::find(liquid, contributor.particle_id, &LiquidParticleState3D::id);
      if (particle == liquid.end() || particle->mass != contributor.mass) {
        return absl::InvalidArgumentError("liquid reconstruction contributor mismatch");
      }
      position = particle->world_position;
    } else {
      const auto particle =
          std::ranges::find(plastic, contributor.particle_id, &PlasticSolidCheckpointParticle3D::particle_id);
      if (particle == plastic.end() || particle->body_id != contributor.body_id
          || particle->transfer.particle_mass != contributor.mass) {
        return absl::InvalidArgumentError("plastic reconstruction contributor mismatch");
      }
      position = particle->transfer.trial_world_position;
    }
    if (!std::isfinite(contributor.mass) || contributor.mass <= 0 || !std::isfinite(contributor.weight)
        || contributor.weight <= 0 || contributor.weight > 1) {
      return absl::InvalidArgumentError("invalid planar mixed contact contributor");
    }
    const Vec3 velocity = contributor.world_face_velocity;
    if (!distinct_contact_detail::Finite(position) || !distinct_contact_detail::Finite(velocity)) {
      return absl::InvalidArgumentError("nonfinite mixed contact reconstruction sample");
    }
    const double coordinate = sign * (position.*normal_component);
    auto& front = first ? first_front : second_front;
    if (!front.has_value() || (first ? coordinate < front->coordinate : coordinate > front->coordinate)) {
      front = FrontSample{position, velocity, coordinate, false};
    } else if (coordinate == front->coordinate && velocity.*normal_component != front->velocity.*normal_component) {
      front->ambiguous = true;
    }
  }
  if (!first_front.has_value() || !second_front.has_value() || first_front->ambiguous || second_front->ambiguous) {
    return absl::FailedPreconditionError("missing or ambiguous planar mixed contact front");
  }
  const double first_coordinate = first_front->coordinate - first_interface_offset;
  const double second_coordinate = second_front->coordinate + second_interface_offset;
  if (!std::isfinite(first_coordinate) || !std::isfinite(second_coordinate)) {
    return absl::OutOfRangeError("unrepresentable planar mixed contact interface");
  }
  Vec3 first_interface = first_front->position;
  first_interface.*normal_component = sign * first_coordinate;
  return PlanarMixedContactFace3D{
      .pair = pair,
      .first_interface_world_position = first_interface,
      .first_interface_world_velocity = first_front->velocity,
      .second_plane_start_coordinate = second_coordinate,
      .second_interface_world_velocity = second_front->velocity,
  };
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
  const auto disjoint = CheckPlanarContactViews(std::as_bytes(liquid), std::as_bytes(plastic), candidates);
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

absl::StatusOr<std::vector<PlanarMixedContactReceipt3D>> AdvancePlanarMixedContact3D(
    std::span<LiquidParticleState3D> liquid,
    std::uint64_t liquid_body_id,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    std::span<const PlanarMixedContactFace3D> candidates,
    double duration,
    double maximum_path_length
) {
  if (!std::isfinite(maximum_path_length) || maximum_path_length <= 0) {
    return absl::InvalidArgumentError("invalid planar mixed contact motion bound");
  }
  const auto disjoint = CheckPlanarContactViews(std::as_bytes(liquid), std::as_bytes(plastic), candidates);
  if (!disjoint.ok()) {
    return disjoint;
  }
  std::vector<LiquidParticleState3D> liquid_trial(liquid.begin(), liquid.end());
  std::vector<PlasticSolidCheckpointParticle3D> plastic_trial(plastic.begin(), plastic.end());
  auto receipts = ApplyPlanarMixedContactFaces3D(liquid_trial, liquid_body_id, plastic_trial, candidates, duration);
  if (!receipts.ok()) {
    return receipts.status();
  }
  const double common_fraction = receipts->empty() ? 1 : receipts->front().contact_fraction;
  for (const auto& receipt : *receipts) {
    if (receipt.contact_fraction != common_fraction) {
      return absl::FailedPreconditionError("planar mixed contact requires one common impact time");
    }
  }
  const double before_duration = duration * common_fraction;
  const double after_duration = duration - before_duration;
  for (std::size_t index = 0; index < liquid.size(); ++index) {
    const auto position = AdvanceAtCommonImpact(
        liquid[index].world_position,
        liquid[index].world_velocity,
        liquid_trial[index].world_velocity,
        before_duration,
        after_duration,
        maximum_path_length
    );
    if (!position.ok()) {
      return position.status();
    }
    liquid_trial[index].world_position = *position;
  }
  for (std::size_t index = 0; index < plastic.size(); ++index) {
    const auto& initial = plastic[index].transfer;
    auto& trial = plastic_trial[index].transfer;
    const auto position = AdvanceAtCommonImpact(
        initial.trial_world_position,
        initial.particle_velocity,
        trial.particle_velocity,
        before_duration,
        after_duration,
        maximum_path_length
    );
    if (!position.ok()) {
      return position.status();
    }
    trial.trial_world_position = *position;
  }

  std::ranges::copy(liquid_trial, liquid.begin());
  std::ranges::copy(plastic_trial, plastic.begin());
  return std::move(*receipts);
}

absl::StatusOr<PlanarSnowPressureTrial3D> SolvePlanarSnowPressureTrial3D(
    const MacPressureInputs3D& inputs,
    std::size_t wall_face_ordinal,
    double snow_mass,
    double incoming_snow_normal_velocity,
    MixedPressureProjector3D project
) {
  if (inputs.pressure_cells.size() != 1 || inputs.pressure_faces.size() != 6
      || wall_face_ordinal >= inputs.pressure_faces.size() || project == nullptr) {
    return absl::InvalidArgumentError("planar snow pressure requires one cell and one selected wall");
  }
  const auto& wall = inputs.pressure_faces[wall_face_ordinal];
  const auto axis = static_cast<std::size_t>(wall.face_index.axis);
  if (axis >= 3 || wall.open_area != 0 || !std::holds_alternative<SealedMacPressureLink3D>(wall.pressure_link)) {
    return absl::InvalidArgumentError("planar snow pressure requires a sealed wall face");
  }
  const auto cell = inputs.pressure_cells.front().cell_index;
  const auto lattice = wall.face_index.lattice_index;
  const std::array<std::int64_t, 3> offsets{
      static_cast<std::int64_t>(lattice.x) - cell.x,
      static_cast<std::int64_t>(lattice.y) - cell.y,
      static_cast<std::int64_t>(lattice.z) - cell.z,
  };
  if (offsets[(axis + 1) % 3] != 0 || offsets[(axis + 2) % 3] != 0 || offsets[axis] < 0 || offsets[axis] > 1) {
    return absl::InvalidArgumentError("planar snow pressure wall is not incident to its cell");
  }
  const double inward_sign = offsets[axis] == 0 ? 1 : -1;
  const double width = inputs.mac_grid.geometry().cell_width;
  const double area = width * width;
  const double impulse_per_pressure = inputs.step_duration * area;
  if (!std::isfinite(impulse_per_pressure) || impulse_per_pressure <= 0 || !std::isfinite(snow_mass) || snow_mass <= 0
      || !std::isfinite(incoming_snow_normal_velocity)) {
    return absl::InvalidArgumentError("invalid planar snow pressure mass, duration, or velocity");
  }

  std::array<MacPressureFace3D, 6> pressure_faces;
  std::ranges::copy(inputs.pressure_faces, pressure_faces.begin());
  const MacPressureInputs3D trial_inputs{
      inputs.mac_grid,
      inputs.step_duration,
      inputs.pressure_cells,
      pressure_faces,
  };
  const auto project_at = [&](double inward_speed) -> absl::StatusOr<MacPressureProjection3D> {
    auto& trial_wall = pressure_faces[wall_face_ordinal];
    trial_wall.wall_normal_velocity = inward_sign * inward_speed;
    trial_wall.prescribed_wall_flux.reset();
    auto projection = project(trial_inputs);
    if (projection.ok()
        && (projection->cell_pressures.size() != 1 || !std::isfinite(projection->cell_pressures.front())
            || projection->corrected_face_velocities.size() != pressure_faces.size())) {
      return absl::FailedPreconditionError("invalid planar snow pressure projection");
    }
    return projection;
  };

  const auto stationary = project_at(0);
  if (!stationary.ok()) {
    return stationary.status();
  }
  const auto probe = project_at(1);
  if (!probe.ok()) {
    return probe.status();
  }
  const double stationary_pressure = stationary->cell_pressures.front();
  const double pressure_slope = probe->cell_pressures.front() - stationary_pressure;
  const double effective_mass = snow_mass + impulse_per_pressure * pressure_slope;
  const double inward_momentum = inward_sign * snow_mass * incoming_snow_normal_velocity;
  if (!std::isfinite(effective_mass) || pressure_slope < 0 || !std::isfinite(inward_momentum)) {
    return absl::FailedPreconditionError("invalid planar snow pressure response");
  }
  const double inward_speed = (inward_momentum - impulse_per_pressure * stationary_pressure) / effective_mass;
  if (!std::isfinite(inward_speed)) {
    return absl::FailedPreconditionError("unrepresentable planar snow pressure speed");
  }
  auto accepted = project_at(inward_speed);
  if (!accepted.ok()) {
    return accepted.status();
  }
  const double snow_impulse = -inward_sign * impulse_per_pressure * accepted->cell_pressures.front();
  const double wall_velocity = inward_sign * inward_speed;
  const double momentum_change = snow_mass * (wall_velocity - incoming_snow_normal_velocity);
  const double work = snow_impulse * wall_velocity;
  const double momentum_scale = std::max({1.0, std::abs(snow_impulse), std::abs(momentum_change)});
  if (!std::isfinite(snow_impulse) || !std::isfinite(momentum_change) || !std::isfinite(work)
      || std::abs(momentum_change - snow_impulse) > 2e-5 * momentum_scale) {
    return absl::FailedPreconditionError("planar snow pressure impulse did not balance");
  }
  return PlanarSnowPressureTrial3D{
      .liquid_projection = std::move(*accepted),
      .solved_wall_normal_velocity = wall_velocity,
      .pressure_impulse_on_snow = snow_impulse,
      .pressure_work_on_snow = work,
  };
}

}  // namespace fluidity
