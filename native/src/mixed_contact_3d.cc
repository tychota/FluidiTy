#include "fluidity/host/mixed_contact_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "fluidity/base/liquid_regular_cell_transport_3d.h"
#include "fluidity/host/liquid_wall_contact_3d.h"
#include "fluidity/host/views.h"
#include "liquid_substep_3d_internal.h"

namespace fluidity {
namespace {

double PlanarWallSpeedTolerance(double solved_wall_speed) {
  return 2e-5 * std::max(1.0, std::abs(solved_wall_speed));
}

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
  if (inputs.pressure_cells.empty() || wall_face_ordinal >= inputs.pressure_faces.size() || project == nullptr) {
    return absl::InvalidArgumentError("planar snow pressure requires pressure cells and a selected wall");
  }
  if (inputs.pressure_cells.size() > 4096 || inputs.pressure_faces.size() > 24576) {
    return absl::ResourceExhaustedError("planar snow pressure extent exceeds solver capacity");
  }
  const auto& wall = inputs.pressure_faces[wall_face_ordinal];
  const auto axis = static_cast<std::size_t>(wall.face_index.axis);
  if (axis >= 3 || wall.open_area != 0 || !std::holds_alternative<SealedMacPressureLink3D>(wall.pressure_link)) {
    return absl::InvalidArgumentError("planar snow pressure requires a sealed wall face");
  }
  const auto lattice = wall.face_index.lattice_index;
  std::optional<std::size_t> wall_cell_ordinal;
  double inward_sign = 0;
  for (std::size_t ordinal = 0; ordinal < inputs.pressure_cells.size(); ++ordinal) {
    const auto cell = inputs.pressure_cells[ordinal].cell_index;
    const std::array<std::int64_t, 3> offsets{
        static_cast<std::int64_t>(lattice.x) - cell.x,
        static_cast<std::int64_t>(lattice.y) - cell.y,
        static_cast<std::int64_t>(lattice.z) - cell.z,
    };
    const bool incident =
        offsets[(axis + 1) % 3] == 0 && offsets[(axis + 2) % 3] == 0 && offsets[axis] >= 0 && offsets[axis] <= 1;
    if (!incident) {
      continue;
    }
    if (wall_cell_ordinal.has_value()) {
      return absl::InvalidArgumentError("planar snow wall requires exactly one incident pressure cell");
    }
    wall_cell_ordinal = ordinal;
    inward_sign = offsets[axis] == 0 ? 1 : -1;
  }
  if (!wall_cell_ordinal.has_value()) {
    return absl::InvalidArgumentError("planar snow wall is missing its incident pressure cell");
  }
  const double width = inputs.mac_grid.geometry().cell_width;
  const double area = width * width;
  const double impulse_per_pressure = inputs.step_duration * area;
  if (!std::isfinite(impulse_per_pressure) || impulse_per_pressure <= 0 || !std::isfinite(snow_mass) || snow_mass <= 0
      || !std::isfinite(incoming_snow_normal_velocity)) {
    return absl::InvalidArgumentError("invalid planar snow pressure mass, duration, or velocity");
  }

  std::vector<MacPressureFace3D> pressure_faces(inputs.pressure_faces.begin(), inputs.pressure_faces.end());
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
        && (projection->cell_pressures.size() != inputs.pressure_cells.size()
            || !std::isfinite(projection->cell_pressures[*wall_cell_ordinal])
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
  const double stationary_pressure = stationary->cell_pressures[*wall_cell_ordinal];
  const double pressure_slope = probe->cell_pressures[*wall_cell_ordinal] - stationary_pressure;
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
  const double snow_impulse = -inward_sign * impulse_per_pressure * accepted->cell_pressures[*wall_cell_ordinal];
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

absl::StatusOr<PlanarSnowPressureTrial3D> ApplyPlanarSnowPressure3D(
    const MacPressureInputs3D& inputs,
    std::size_t wall_face_ordinal,
    std::span<const SnowPressureContributor3D> contributors,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    MixedPressureProjector3D project
) {
  const auto disjoint = CheckWritableViews(
      {std::as_writable_bytes(plastic)},
      {std::as_bytes(contributors), std::as_bytes(inputs.pressure_cells), std::as_bytes(inputs.pressure_faces)}
  );
  if (!disjoint.ok()) {
    return disjoint;
  }
  if (contributors.empty() || wall_face_ordinal >= inputs.pressure_faces.size()) {
    return absl::InvalidArgumentError("missing planar snow traction contributors or wall face");
  }
  constexpr std::array components{&Vec3::x, &Vec3::y, &Vec3::z};
  const auto axis = static_cast<std::size_t>(inputs.pressure_faces[wall_face_ordinal].face_index.axis);
  if (axis >= components.size()) {
    return absl::InvalidArgumentError("invalid planar snow pressure axis");
  }
  const auto component = components[axis];
  std::uint64_t previous_particle_id = 0;
  for (const auto& particle : plastic) {
    if (particle.particle_id <= previous_particle_id) {
      return absl::InvalidArgumentError("planar snow particles must be ordered by nonzero ID");
    }
    previous_particle_id = particle.particle_id;
  }

  struct StagedNormalVelocity {
    double* stored_velocity;
    double impulse_per_mass;
    double candidate_velocity;
  };
  std::vector<StagedNormalVelocity> staged;
  staged.reserve(contributors.size());
  double area_fraction_sum = 0;
  double inverse_wall_mass = 0;
  double incoming_wall_speed = 0;
  previous_particle_id = 0;
  std::uint64_t wall_body_id = 0;
  auto particle = plastic.begin();
  for (const auto& contributor : contributors) {
    const double weight = contributor.area_fraction;
    if (contributor.particle_id <= previous_particle_id || !std::isfinite(weight) || weight <= 0 || weight > 1) {
      return absl::InvalidArgumentError("invalid planar snow traction contributor");
    }
    particle = std::ranges::lower_bound(
        particle,
        plastic.end(),
        contributor.particle_id,
        {},
        &PlasticSolidCheckpointParticle3D::particle_id
    );
    if (particle == plastic.end() || particle->particle_id != contributor.particle_id) {
      return absl::InvalidArgumentError("missing planar snow traction particle");
    }
    if (particle->body_id == 0 || (wall_body_id != 0 && particle->body_id != wall_body_id)) {
      return absl::InvalidArgumentError("planar snow traction requires a single wall body");
    }
    wall_body_id = particle->body_id;
    const double mass = particle->transfer.particle_mass;
    double& normal_velocity = particle->transfer.particle_velocity.*component;
    if (!std::isfinite(mass) || mass <= 0 || !std::isfinite(normal_velocity)) {
      return absl::InvalidArgumentError("invalid planar snow traction mass or velocity");
    }
    const double weighted_inverse_mass = weight / mass;
    staged.push_back({&normal_velocity, weighted_inverse_mass, 0});
    area_fraction_sum += weight;
    inverse_wall_mass += weight * weighted_inverse_mass;
    incoming_wall_speed += weight * normal_velocity;
    previous_particle_id = contributor.particle_id;
  }
  if (std::abs(area_fraction_sum - 1) > 1e-12) {
    return absl::InvalidArgumentError("planar snow traction area fractions must sum to one");
  }
  const double wall_mass = 1 / inverse_wall_mass;
  if (!std::isfinite(wall_mass) || wall_mass <= 0 || !std::isfinite(incoming_wall_speed)) {
    return absl::OutOfRangeError("unrepresentable planar snow traction response");
  }
  auto trial = SolvePlanarSnowPressureTrial3D(inputs, wall_face_ordinal, wall_mass, incoming_wall_speed, project);
  if (!trial.ok()) {
    return trial.status();
  }
  double scattered_wall_speed = 0;
  for (std::size_t ordinal = 0; ordinal < staged.size(); ++ordinal) {
    auto& velocity = staged[ordinal];
    velocity.candidate_velocity =
        std::fma(velocity.impulse_per_mass, trial->pressure_impulse_on_snow, *velocity.stored_velocity);
    if (!std::isfinite(velocity.candidate_velocity)) {
      return absl::OutOfRangeError("unrepresentable planar snow traction velocity");
    }
    scattered_wall_speed += contributors[ordinal].area_fraction * velocity.candidate_velocity;
  }
  if (!std::isfinite(scattered_wall_speed)
      || std::abs(scattered_wall_speed - trial->solved_wall_normal_velocity)
          > PlanarWallSpeedTolerance(trial->solved_wall_normal_velocity)) {
    return absl::FailedPreconditionError("scattered planar snow traction did not recover wall speed");
  }
  for (const auto& velocity : staged) {
    *velocity.stored_velocity = velocity.candidate_velocity;
  }
  return trial;
}

absl::StatusOr<FrozenPlanarSnowPressureCell3D> FreezePlanarSnowPressureCell3D(
    const SparseMacGrid3D& grid,
    CellIndex3D cell,
    FaceNormalAxis axis,
    bool wall_on_low_face,
    std::span<const PlasticSolidCheckpointParticle3D> snow_front_particles,
    double surface_offset,
    double inverse_liquid_density,
    double outlet_pressure
) {
  constexpr std::array components{&Vec3::x, &Vec3::y, &Vec3::z};
  const auto normal_axis = static_cast<std::size_t>(axis);
  if (normal_axis >= components.size() || snow_front_particles.empty() || !std::isfinite(surface_offset)
      || surface_offset < 0 || !std::isfinite(inverse_liquid_density) || inverse_liquid_density <= 0
      || !std::isfinite(outlet_pressure)) {
    return absl::InvalidArgumentError("invalid planar snow pressure geometry inputs");
  }
  const auto component = components[normal_axis];
  const double sign = wall_on_low_face ? 1 : -1;
  double front = -std::numeric_limits<double>::infinity();
  const auto body_id = snow_front_particles.front().body_id;
  for (const auto& particle : snow_front_particles) {
    const auto position = particle.transfer.trial_world_position;
    if (body_id == 0 || particle.body_id != body_id || !distinct_contact_detail::Finite(position)) {
      return absl::InvalidArgumentError("invalid planar snow front particle");
    }
    front = std::max(front, sign * (position.*component));
  }
  const double wall_coordinate = front + surface_offset;
  const auto geometry = grid.geometry();
  const double width = geometry.cell_width;
  const std::array coordinates{cell.x, cell.y, cell.z};
  const double low = std::fma(static_cast<double>(coordinates[normal_axis]), width, geometry.world_origin.*component);
  const double high = low + width;
  const double wet_width = wall_on_low_face ? high - wall_coordinate : -wall_coordinate - low;
  if (!std::isfinite(wet_width) || wet_width <= 0 || wet_width > width) {
    return absl::FailedPreconditionError("planar snow front is outside the selected wall cell");
  }
  const double area = width * width;
  const double wet_volume = area * wet_width;
  if (!std::isfinite(wet_volume) || wet_volume <= 0) {
    return absl::OutOfRangeError("unrepresentable planar snow pressure volume");
  }
  FrozenPlanarSnowPressureCell3D frozen{
      .cell =
          {
              .cell_index = cell,
              .fluid_volume = wet_volume,
              .bulk_modulus = std::numeric_limits<double>::infinity(),
          },
      .faces = {},
      .wall_face_ordinal = 0,
      .wall_plane_coordinate = wall_coordinate,
  };
  struct OwnedFace {
    std::size_t slot;
    MacPressureFace3D face;
    bool wall;
  };
  std::array<OwnedFace, 6> owned_faces;
  for (std::size_t face_axis = 0; face_axis < components.size(); ++face_axis) {
    for (const int side : {0, 1}) {
      auto lattice = coordinates;
      if (side == 1 && lattice[face_axis] == std::numeric_limits<std::int32_t>::max()) {
        return absl::OutOfRangeError("planar snow pressure face index exceeds lattice range");
      }
      lattice[face_axis] += side;
      const bool normal_face = face_axis == normal_axis;
      const bool selected_wall = normal_face && (side == 0) == wall_on_low_face;
      const bool outlet = normal_face && !selected_wall;
      MacPressureFace3D face{
          .face_index = {static_cast<FaceNormalAxis>(face_axis), {lattice[0], lattice[1], lattice[2]}},
          .pressure_link = SealedMacPressureLink3D{},
      };
      if (outlet) {
        face.open_area = area;
        face.inverse_density = inverse_liquid_density;
        face.pressure_gradient_distance = 0.5 * wet_width;
        face.pressure_link = DirichletMacPressureLink3D{outlet_pressure};
      }
      const auto slot = grid.FaceSlot(face.face_index);
      if (!slot.ok()) {
        return slot.status();
      }
      owned_faces[2 * face_axis + side] = {*slot, face, selected_wall};
    }
  }
  std::ranges::sort(owned_faces, {}, &OwnedFace::slot);
  for (std::size_t ordinal = 0; ordinal < owned_faces.size(); ++ordinal) {
    frozen.faces[ordinal] = owned_faces[ordinal].face;
    if (owned_faces[ordinal].wall) {
      frozen.wall_face_ordinal = ordinal;
    }
  }
  return frozen;
}

absl::StatusOr<PlanarSnowPressureTrial3D> AdvancePlanarSnowPressure3D(
    const MacPressureInputs3D& inputs,
    std::size_t wall_face_ordinal,
    std::span<const SnowPressureContributor3D> contributors,
    std::span<LiquidParticleState3D> liquid,
    std::span<PlasticSolidCheckpointParticle3D> plastic,
    double maximum_path_length,
    MixedPressureProjector3D project,
    std::optional<double> wall_plane_coordinate,
    std::optional<SnowParameters3D> snow_layer_material
) {
  const auto disjoint = CheckWritableViews(
      {std::as_writable_bytes(liquid), std::as_writable_bytes(plastic)},
      {std::as_bytes(contributors), std::as_bytes(inputs.pressure_cells), std::as_bytes(inputs.pressure_faces)}
  );
  if (!disjoint.ok()) {
    return disjoint;
  }
  if (!std::isfinite(maximum_path_length) || maximum_path_length <= 0 || liquid.empty()) {
    return absl::InvalidArgumentError("invalid planar snow pressure transport extent");
  }
  std::vector<PlasticSolidCheckpointParticle3D> plastic_trial(plastic.begin(), plastic.end());
  auto pressure = ApplyPlanarSnowPressure3D(inputs, wall_face_ordinal, contributors, plastic_trial, project);
  if (!pressure.ok()) {
    return pressure.status();
  }

  if (wall_plane_coordinate.has_value()
      && (inputs.pressure_cells.size() != 1 || !std::isfinite(*wall_plane_coordinate))) {
    return absl::InvalidArgumentError("subcell snow transport requires one cell and a finite wall coordinate");
  }
  const auto geometry = inputs.mac_grid.geometry();
  const double width = geometry.cell_width;
  const double face_area = width * width;
  constexpr std::array components{&Vec3::x, &Vec3::y, &Vec3::z};
  const auto cell_key = [](CellIndex3D cell) {
    return std::array{cell.z, cell.y, cell.x};
  };
  const auto find_cell = [&](CellIndex3D index) -> std::optional<std::size_t> {
    const auto key = cell_key(index);
    const auto found = std::ranges::lower_bound(inputs.pressure_cells, key, {}, [&](const auto& pressure_cell) {
      return cell_key(pressure_cell.cell_index);
    });
    if (found == inputs.pressure_cells.end() || cell_key(found->cell_index) != key) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(found - inputs.pressure_cells.begin());
  };
  const auto cell_corner = [&](CellIndex3D cell) {
    return Vec3{
        std::fma(static_cast<double>(cell.x), width, geometry.world_origin.x),
        std::fma(static_cast<double>(cell.y), width, geometry.world_origin.y),
        std::fma(static_cast<double>(cell.z), width, geometry.world_origin.z),
    };
  };
  struct ProjectedCell {
    LiquidRegularCellTransport3D sample{.active = true};
    std::array<bool, 3> open_low{};
    std::array<bool, 3> open_high{};
  };
  std::vector<ProjectedCell> projected_cells(inputs.pressure_cells.size());
  TranslatingHalfspace3D wall{};
  const auto wall_axis = static_cast<std::size_t>(inputs.pressure_faces[wall_face_ordinal].face_index.axis);
  const auto wall_component = components[wall_axis];
  Vec3 transport_width{width, width, width};
  for (std::size_t ordinal = 0; ordinal < inputs.pressure_faces.size(); ++ordinal) {
    const auto& face = inputs.pressure_faces[ordinal];
    const auto axis = static_cast<std::size_t>(face.face_index.axis);
    if (axis >= components.size()) {
      return absl::InvalidArgumentError("invalid planar pressure transport face axis");
    }
    const auto lattice = face.face_index.lattice_index;
    const auto component = components[axis];
    const auto head = find_cell({lattice.x, lattice.y, lattice.z});
    std::array tail_lattice{lattice.x, lattice.y, lattice.z};
    std::optional<std::size_t> tail;
    if (tail_lattice[axis] > std::numeric_limits<std::int32_t>::min()) {
      --tail_lattice[axis];
      tail = find_cell({tail_lattice[0], tail_lattice[1], tail_lattice[2]});
    }
    const bool fully_open = face.open_area == face_area;
    double normal_velocity = pressure->liquid_projection.corrected_face_velocities[ordinal];
    if (ordinal == wall_face_ordinal) {
      normal_velocity = pressure->solved_wall_normal_velocity;
      const double inward_sign = head.has_value() ? 1 : -1;
      const auto plane = cell_corner({lattice.x, lattice.y, lattice.z});
      wall.unit_normal.*component = inward_sign;
      const double lattice_plane_coordinate = inward_sign * (plane.*component);
      wall.start_plane_coordinate = wall_plane_coordinate.value_or(lattice_plane_coordinate);
      const double inward_offset = wall.start_plane_coordinate - lattice_plane_coordinate;
      transport_width.*component = width - inward_offset;
      if (wall_plane_coordinate.has_value()) {
        const double wet_volume = face_area * (transport_width.*component);
        const double volume_error = std::abs(wet_volume - inputs.pressure_cells.front().fluid_volume);
        if (inward_offset < 0 || !(transport_width.*component > 0) || !std::isfinite(wet_volume)
            || volume_error > 1e-12 * face_area * width) {
          return absl::InvalidArgumentError("subcell snow wall disagrees with wet pressure volume");
        }
      }
      wall.wall_velocity.*component = normal_velocity;
    } else if (face.open_area == 0 && face.wall_normal_velocity == 0 && face.prescribed_wall_flux.value_or(0) == 0) {
      normal_velocity = 0;
    } else if (!fully_open || (wall_plane_coordinate.has_value() && axis != wall_axis)) {
      return absl::InvalidArgumentError("planar pressure transport requires full open or stationary sealed faces");
    }
    if (head.has_value()) {
      projected_cells[*head].sample.low_face_normal_velocity.*component = normal_velocity;
      projected_cells[*head].open_low[axis] = fully_open;
    }
    if (tail.has_value()) {
      projected_cells[*tail].sample.high_face_normal_velocity.*component = normal_velocity;
      projected_cells[*tail].open_high[axis] = fully_open;
    }
  }

  const TranslatingHalfspace3D liquid_side{
      {-wall.unit_normal.x, -wall.unit_normal.y, -wall.unit_normal.z},
      -wall.start_plane_coordinate,
      wall.wall_velocity,
  };
  const auto fraction_in_cell = [&](Vec3 world_position, Vec3 low_corner) {
    return Vec3{
        (world_position.x - low_corner.x) / transport_width.x,
        (world_position.y - low_corner.y) / transport_width.y,
        (world_position.z - low_corner.z) / transport_width.z,
    };
  };
  std::vector<LiquidParticleState3D> liquid_trial(liquid.begin(), liquid.end());
  for (auto& particle : liquid_trial) {
    const auto start_cell = liquid_step_internal::TankContainingCell(geometry, particle.world_position);
    const auto start_ordinal = start_cell.has_value() ? find_cell(*start_cell) : std::nullopt;
    if (!start_ordinal.has_value()) {
      return absl::FailedPreconditionError("planar pressure liquid transport starts outside pressure support");
    }
    Vec3 low_corner = cell_corner(*start_cell);
    if (wall_plane_coordinate.has_value() && wall.unit_normal.*wall_component > 0) {
      low_corner.*wall_component = *wall_plane_coordinate;
    }
    const auto& projected = projected_cells[*start_ordinal];
    auto sample = projected.sample;
    sample.position_in_cell_widths = fraction_in_cell(particle.world_position, low_corner);
    const auto field = internal::EvaluateRegularCellTransport3D(sample, width);
    if (!field.ok() || !std::isfinite(particle.volume_ratio) || particle.volume_ratio <= 0) {
      return absl::InvalidArgumentError("invalid planar pressure liquid transport state");
    }
    const auto velocity = field->advecting_world_velocity;
    const auto end = AdvanceAtCommonImpact(
        particle.world_position,
        velocity,
        velocity,
        inputs.step_duration,
        0,
        maximum_path_length
    );
    if (!end.ok()) {
      return end.status();
    }
    const auto swept = ResolveLiquidWallContact3D({
        .start_world_position = particle.world_position,
        .stored_world_velocity = velocity,
        .advecting_world_velocity = velocity,
        .obstacle = wall,
        .step_duration = inputs.step_duration,
    });
    if (!swept.ok()) {
      return swept.status();
    }
    if (swept->contact_fraction.has_value()) {
      return absl::FailedPreconditionError("planar pressure liquid transport needs additional wall contact");
    }
    const auto end_fraction = fraction_in_cell(*end, low_corner);
    std::size_t crossing_count = 0;
    for (std::size_t axis = 0; axis < components.size(); ++axis) {
      const double fraction = end_fraction.*components[axis];
      if (fraction >= 0 && fraction <= 1) {
        continue;
      }
      const bool open = fraction < 0 ? projected.open_low[axis] : projected.open_high[axis];
      if (!open) {
        return absl::FailedPreconditionError("planar pressure liquid transport crosses a closed face");
      }
      ++crossing_count;
    }
    if (crossing_count > 0) {
      const auto destination = liquid_step_internal::TankContainingCell(geometry, *end);
      if (crossing_count != 1 || !destination.has_value() || !find_cell(*destination).has_value()
          || std::abs(static_cast<std::int64_t>(destination->x) - start_cell->x)
                  + std::abs(static_cast<std::int64_t>(destination->y) - start_cell->y)
                  + std::abs(static_cast<std::int64_t>(destination->z) - start_cell->z)
              != 1) {
        return absl::FailedPreconditionError("planar pressure liquid transport leaves adjacent pressure support");
      }
    }
    const Vec3 gradient{
        (sample.high_face_normal_velocity.x - sample.low_face_normal_velocity.x) / transport_width.x,
        (sample.high_face_normal_velocity.y - sample.low_face_normal_velocity.y) / transport_width.y,
        (sample.high_face_normal_velocity.z - sample.low_face_normal_velocity.z) / transport_width.z,
    };
    const double divergence = (gradient.x + gradient.y) + gradient.z;
    const double volume_ratio = particle.volume_ratio * std::exp(inputs.step_duration * divergence);
    if (!std::isfinite(volume_ratio) || volume_ratio <= 0) {
      return absl::OutOfRangeError("unrepresentable planar pressure liquid volume");
    }
    particle.world_position = *end;
    particle.world_velocity = velocity;
    particle.velocity_slopes = {{gradient.x, 0, 0}, {0, gradient.y, 0}, {0, 0, gradient.z}};
    particle.affine_enabled = true;
    particle.pic_initialized = true;
    particle.volume_ratio = volume_ratio;
  }

  const auto drift_snow = [&](PlasticSolidCheckpointParticle3D& particle) -> absl::Status {
    const auto velocity = particle.transfer.particle_velocity;
    const auto end = AdvanceAtCommonImpact(
        particle.transfer.trial_world_position,
        velocity,
        velocity,
        inputs.step_duration,
        0,
        maximum_path_length
    );
    if (!end.ok()) {
      return end.status();
    }
    const auto swept = ResolveLiquidWallContact3D({
        .start_world_position = particle.transfer.trial_world_position,
        .stored_world_velocity = velocity,
        .advecting_world_velocity = velocity,
        .obstacle = liquid_side,
        .step_duration = inputs.step_duration,
    });
    if (!swept.ok()) {
      return swept.status();
    }
    if (swept->contact_fraction.has_value()) {
      return absl::FailedPreconditionError("planar pressure snow transport crosses the reconstructed wall");
    }
    particle.transfer.trial_world_position = *end;
    return absl::OkStatus();
  };
  if (snow_layer_material.has_value()) {
    const auto accepted_snow = RestoreDrySolidCheckpointParticles3D(
        plastic,
        plastic.front().body_id,
        plastic.front().material_id,
        DrySolidMaterial3D{*snow_layer_material}
    );
    if (!accepted_snow.ok()) {
      return accepted_snow.status();
    }
    const auto normal_coordinate = [&](const PlasticSolidCheckpointParticle3D& particle) {
      return (wall.unit_normal.*wall_component) * (particle.transfer.trial_world_position.*wall_component);
    };
    const auto [back, front] = std::ranges::minmax_element(plastic, {}, normal_coordinate);
    const auto back_ordinal = static_cast<std::size_t>(back - plastic.begin());
    const auto front_ordinal = static_cast<std::size_t>(front - plastic.begin());
    const double thickness = normal_coordinate(*front) - normal_coordinate(*back);
    if (!std::isfinite(thickness) || thickness <= 0) {
      return absl::InvalidArgumentError("planar snow strain requires a positive layer thickness");
    }
    const double geometry_tolerance = 1e-10 * width;
    for (const auto& contributor : contributors) {
      const auto particle = std::ranges::lower_bound(
          plastic,
          contributor.particle_id,
          {},
          &PlasticSolidCheckpointParticle3D::particle_id
      );
      if (std::abs(normal_coordinate(*particle) - normal_coordinate(*front)) > geometry_tolerance) {
        return absl::InvalidArgumentError("planar snow pressure contributor is off the front layer");
      }
    }
    for (auto& particle : plastic_trial) {
      if (particle.body_id != front->body_id) {
        return absl::InvalidArgumentError("planar snow strain requires one body");
      }
      const auto status = drift_snow(particle);
      if (!status.ok()) {
        return status;
      }
    }
    const double moved_back = normal_coordinate(plastic_trial[back_ordinal]);
    const double moved_thickness = normal_coordinate(plastic_trial[front_ordinal]) - moved_back;
    const double normal_stretch = moved_thickness / thickness;
    if (!std::isfinite(normal_stretch) || normal_stretch <= 0) {
      return absl::FailedPreconditionError("planar snow layer collapsed");
    }
    constexpr std::array deformation_rows{&Mat3::x_row, &Mat3::y_row, &Mat3::z_row};
    for (std::size_t ordinal = 0; ordinal < plastic_trial.size(); ++ordinal) {
      auto& particle = plastic_trial[ordinal];
      const double expected =
          moved_back + normal_stretch * (normal_coordinate(plastic[ordinal]) - normal_coordinate(*back));
      if (std::abs(normal_coordinate(particle) - expected) > geometry_tolerance) {
        return absl::FailedPreconditionError("planar snow layer motion is not affine");
      }
      auto trial_deformation = particle.elastic_deformation;
      auto& normal_row = trial_deformation.*deformation_rows[wall_axis];
      normal_row = distinct_contact_detail::AddScaled({}, normal_row, normal_stretch);
      SolidDeformationState3D returned{};
      if (!ReturnSnowDeformation3D(trial_deformation, particle.log_plastic_jacobian, *snow_layer_material, returned)) {
        return absl::FailedPreconditionError("inadmissible planar snow deformation");
      }
      particle.elastic_deformation = returned.elastic_deformation;
      particle.log_plastic_jacobian = returned.log_plastic_jacobian;
    }
  } else {
    auto particle = plastic_trial.begin();
    for (const auto& contributor : contributors) {
      particle = std::ranges::lower_bound(
          particle,
          plastic_trial.end(),
          contributor.particle_id,
          {},
          &PlasticSolidCheckpointParticle3D::particle_id
      );
      const auto status = drift_snow(*particle);
      if (!status.ok()) {
        return status;
      }
    }
  }
  std::ranges::copy(plastic_trial, plastic.begin());
  std::ranges::copy(liquid_trial, liquid.begin());
  return pressure;
}

absl::StatusOr<PlanarSnowPressureAttempt3D> AttemptPlanarSnowPressureStep3D(
    LiquidFrameTransaction3D& transaction,
    const MacPressureInputs3D& inputs,
    std::size_t wall_face_ordinal,
    std::span<const SnowPressureContributor3D> contributors,
    std::span<PlasticSolidCheckpointParticle3D> snow,
    double maximum_path_length,
    MixedPressureProjector3D project,
    std::optional<double> wall_plane_coordinate,
    std::optional<SnowParameters3D> snow_layer_material
) {
  const auto& committed = transaction.committed();
  const auto geometry = inputs.mac_grid.geometry();
  const auto origin = geometry.world_origin;
  const auto committed_origin = committed.geometry.world_origin;
  if (geometry.cell_width != committed.geometry.cell_width || origin.x != committed_origin.x
      || origin.y != committed_origin.y || origin.z != committed_origin.z
      || committed.cells.size() != inputs.pressure_cells.size()
      || committed.faces.size() != inputs.pressure_faces.size()) {
    return absl::InvalidArgumentError("planar pressure transaction topology mismatch");
  }
  const auto snow_disjoint = CheckWritableViews(
      {std::as_writable_bytes(snow)},
      {
          std::as_bytes(contributors),
          std::as_bytes(inputs.pressure_cells),
          std::as_bytes(inputs.pressure_faces),
          std::as_bytes(committed.blocks.values()),
          std::as_bytes(committed.sources.values()),
          std::as_bytes(committed.cavity_gas.values()),
          std::as_bytes(committed.cavity_components.values()),
          std::as_bytes(committed.particles.values()),
          std::as_bytes(committed.cells.values()),
          std::as_bytes(committed.faces.values()),
      }
  );
  if (!snow_disjoint.ok()) {
    return snow_disjoint;
  }
  std::vector<MacPressureCell3D> rows(inputs.pressure_cells.begin(), inputs.pressure_cells.end());
  for (std::size_t ordinal = 0; ordinal < rows.size(); ++ordinal) {
    const auto index = rows[ordinal].cell_index;
    const auto stored = committed.cells[ordinal].cell_index;
    if (index.x != stored.x || index.y != stored.y || index.z != stored.z) {
      return absl::InvalidArgumentError("planar pressure transaction cell mismatch");
    }
    rows[ordinal].previous_pressure = committed.cells[ordinal].previous_pressure;
  }
  for (std::size_t ordinal = 0; ordinal < inputs.pressure_faces.size(); ++ordinal) {
    const auto index = inputs.pressure_faces[ordinal].face_index;
    const auto stored = committed.faces[ordinal].face_index;
    const auto slot = inputs.mac_grid.FaceSlot(index);
    if (!slot.ok()) {
      return slot.status();
    }
    if (index.axis != stored.axis || index.lattice_index.x != stored.lattice_index.x
        || index.lattice_index.y != stored.lattice_index.y || index.lattice_index.z != stored.lattice_index.z
        || *slot != committed.faces[ordinal].face_slot) {
      return absl::InvalidArgumentError("planar pressure transaction face mismatch");
    }
  }

  const auto reset = transaction.ResetTrialFromCommitted();
  if (!reset.ok()) {
    return reset.status();
  }
  auto& liquid_trial = **reset;
  const auto prospective_clock = transaction.PreviewAcceptedClock(inputs.step_duration);
  if (!prospective_clock.ok()) {
    const auto rejected = transaction.FinishAttempt(LiquidAttemptDisposition3D::kFatal, inputs.step_duration);
    return rejected.ok() ? prospective_clock.status() : rejected.status();
  }
  std::vector<PlasticSolidCheckpointParticle3D> snow_trial(snow.begin(), snow.end());
  const MacPressureInputs3D trial_inputs{inputs.mac_grid, inputs.step_duration, rows, inputs.pressure_faces};
  auto pressure = AdvancePlanarSnowPressure3D(
      trial_inputs,
      wall_face_ordinal,
      contributors,
      liquid_trial.particles.values(),
      snow_trial,
      maximum_path_length,
      project,
      wall_plane_coordinate,
      snow_layer_material
  );
  if (!pressure.ok()) {
    const bool retry = pressure.status().code() == absl::StatusCode::kFailedPrecondition;
    const auto decision = retry ? LiquidAttemptDisposition3D::kRetryDt : LiquidAttemptDisposition3D::kFatal;
    const auto rejected = transaction.FinishAttempt(decision, inputs.step_duration);
    if (!rejected.ok()) {
      return rejected.status();
    }
    if (!retry) {
      return pressure.status();
    }
    return PlanarSnowPressureAttempt3D{.outcome = *rejected, .pressure = std::nullopt};
  }
  for (std::size_t ordinal = 0; ordinal < liquid_trial.cells.size(); ++ordinal) {
    liquid_trial.cells[ordinal].previous_pressure = pressure->liquid_projection.cell_pressures[ordinal];
  }
  for (std::size_t ordinal = 0; ordinal < liquid_trial.faces.size(); ++ordinal) {
    auto& face = liquid_trial.faces[ordinal];
    face.carrier_velocity = ordinal == wall_face_ordinal
        ? pressure->solved_wall_normal_velocity
        : pressure->liquid_projection.corrected_face_velocities[ordinal];
    face.validity = LiquidFaceValidity3D::kValid;
  }
  const auto accepted = transaction.FinishAttempt(LiquidAttemptDisposition3D::kAccepted, inputs.step_duration);
  if (!accepted.ok()) {
    return accepted.status();
  }
  std::ranges::copy(snow_trial, snow.begin());
  return PlanarSnowPressureAttempt3D{.outcome = *accepted, .pressure = std::move(*pressure)};
}

absl::StatusOr<PlanarSnowPressureFrame3D> AdvancePlanarSnowPressureFrame3D(
    const SparseMacGrid3D& grid,
    const LiquidState3D& accepted_liquid,
    std::span<const PlasticSolidCheckpointParticle3D> accepted_snow,
    std::span<const SnowPressureContributor3D> contributors,
    const PlanarSnowPressureFrameSettings3D& settings,
    MixedPressureProjector3D project
) {
  if (!std::isfinite(settings.nominal_step_duration) || settings.nominal_step_duration <= 0) {
    return absl::InvalidArgumentError("invalid planar snow frame timestep");
  }
  if (accepted_liquid.sources.size() != 0 || accepted_liquid.next_cavity_id != 0) {
    return absl::InvalidArgumentError("planar snow frame requires liquid without sources or cavities");
  }
  const LiquidStateCapacities3D capacities{
      .particles = accepted_liquid.particles.capacity(),
      .blocks = accepted_liquid.blocks.capacity(),
      .cells = accepted_liquid.cells.capacity(),
      .faces = accepted_liquid.faces.capacity(),
      .sources = accepted_liquid.sources.capacity(),
  };
  auto transaction = LiquidFrameTransaction3D::Create(accepted_liquid, capacities);
  if (!transaction.ok()) {
    return transaction.status();
  }
  const auto begin = transaction->BeginFrame(settings.target_time);
  if (!begin.ok()) {
    return begin;
  }

  std::vector<PlasticSolidCheckpointParticle3D> snow(accepted_snow.begin(), accepted_snow.end());
  std::size_t rejected_attempts = 0;
  const auto freeze_wall = [&] {
    return FreezePlanarSnowPressureCell3D(
        grid,
        settings.wall_cell,
        settings.wall_axis,
        settings.wall_on_low_face,
        snow,
        settings.surface_offset,
        settings.inverse_liquid_density,
        settings.outlet_pressure
    );
  };
  auto frozen = freeze_wall();
  if (!frozen.ok()) {
    return frozen.status();
  }
  const double wall_normal_sign = settings.wall_on_low_face ? 1 : -1;
  const double wall_tolerance = 1e-10 * grid.geometry().cell_width;
  while (transaction->committed().clock.time < settings.target_time) {
    const double remaining = settings.target_time - transaction->committed().clock.time;
    MacPressureInputs3D inputs{
        grid,
        std::min(remaining, settings.nominal_step_duration),
        {&frozen->cell, 1},
        frozen->faces
    };
    while (true) {
      const auto attempt = AttemptPlanarSnowPressureStep3D(
          *transaction,
          inputs,
          frozen->wall_face_ordinal,
          contributors,
          snow,
          settings.maximum_path_length,
          project,
          frozen->wall_plane_coordinate,
          settings.snow_layer_material
      );
      if (!attempt.ok()) {
        return attempt.status();
      }
      if (attempt->outcome.disposition == LiquidAttemptDisposition3D::kAccepted) {
        const double end_wall_coordinate = std::fma(
            inputs.step_duration,
            wall_normal_sign * attempt->pressure->solved_wall_normal_velocity,
            frozen->wall_plane_coordinate
        );
        auto end_geometry = freeze_wall();
        if (!end_geometry.ok()) {
          return end_geometry.status();
        }
        // Include the allowed discrepancy between the projected speed and scattered traction.
        const double pose_tolerance = wall_tolerance
            + inputs.step_duration * PlanarWallSpeedTolerance(attempt->pressure->solved_wall_normal_velocity);
        if (std::abs(end_geometry->wall_plane_coordinate - end_wall_coordinate) > pose_tolerance) {
          return absl::FailedPreconditionError("planar pressure wall disagrees with the advected snow front");
        }
        frozen = std::move(end_geometry);
        break;
      }
      if (!attempt->outcome.retry_duration.has_value()) {
        return absl::FailedPreconditionError("planar snow frame exhausted its pressure or motion retries");
      }
      inputs.step_duration = *attempt->outcome.retry_duration;
      ++rejected_attempts;
    }
  }
  const auto finish = transaction->FinishFrame();
  if (!finish.ok()) {
    return finish;
  }
  auto liquid = CloneLiquidState3D(transaction->committed(), capacities);
  if (!liquid.ok()) {
    return liquid.status();
  }
  return PlanarSnowPressureFrame3D{
      .liquid = std::move(*liquid),
      .snow = std::move(snow),
      .rejected_attempts = rejected_attempts,
  };
}

}  // namespace fluidity
