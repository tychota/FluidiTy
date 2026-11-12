#include "fluidity/host/solid_interaction_checkpoint_3d.h"

#include <cstddef>

#include "fluidity/host/transfers_3d.h"
#include "liquid_cache_3d_internal.h"
#include "solid_interaction_checkpoint_generated.h"

namespace fluidity {
namespace {

namespace wire = solid::v1;
using liquid_cache_internal::Pack;
using liquid_cache_internal::Unpack;
constexpr std::size_t kMaximumRecords = 65536;
constexpr std::size_t kMaximumBytes = 32 * 1024 * 1024;

wire::Pose PackPose(Vec3 center, RigidOrientation3D orientation) {
  return {Pack(center), wire::Orientation(orientation.w, Pack(orientation.vector))};
}

RigidBodyPose3D UnpackPose(const wire::Pose& pose) {
  return {Unpack(&pose.center()), {pose.orientation().w(), Unpack(&pose.orientation().vector())}};
}

cache::v1::Matrix3 PackMatrix(Mat3 matrix) {
  return {Pack(matrix.x_row), Pack(matrix.y_row), Pack(matrix.z_row)};
}

Mat3 UnpackMatrix(const cache::v1::Matrix3& matrix) {
  return {Unpack(&matrix.x_row()), Unpack(&matrix.y_row()), Unpack(&matrix.z_row())};
}

absl::Status ValidateCheckpoint(const SolidInteractionCheckpoint3D& state) {
  if (state.bodies.size() > kMaximumRecords || state.sides.size() > kMaximumRecords
      || state.elastic_particles.size() > kMaximumRecords) {
    return absl::ResourceExhaustedError("solid checkpoint record capacity exceeded");
  }
  std::uint64_t previous_id = 0;
  for (const auto& history : state.bodies) {
    if (history.body.body_id <= previous_id) {
      return absl::InvalidArgumentError("solid checkpoint body IDs must be strictly increasing");
    }
    const auto body_status = ValidateRigidBody3D(history.body);
    if (!body_status.ok()) {
      return body_status;
    }
    const auto pose_status = ValidateRigidBodyPose3D(history.previous_pose);
    if (!pose_status.ok()) {
      return pose_status;
    }
    previous_id = history.body.body_id;
  }
  previous_id = 0;
  for (const auto& entry : state.sides) {
    const bool assigned_sign = entry.side.sign == -1 || entry.side.sign == 1;
    if (entry.particle_id <= previous_id || entry.side.blade_id == 0 || !assigned_sign) {
      return absl::InvalidArgumentError("solid checkpoint requires ordered particle IDs and assigned blade sides");
    }
    previous_id = entry.particle_id;
  }
  previous_id = 0;
  for (const auto& particle : state.elastic_particles) {
    const auto& transfer = particle.transfer;
    const Mat3 slopes = transfer.particle_velocity_slopes;
    using liquid_cache_internal::Finite;
    using liquid_cache_internal::Positive;
    const bool valid_identity = particle.particle_id > previous_id && particle.body_id != 0;
    const bool valid_extent = Positive(particle.reference_volume) && Positive(transfer.particle_mass);
    const bool valid_kinematics = Finite(transfer.sorted_world_position) && Finite(transfer.trial_world_position)
        && Finite(transfer.particle_velocity) && Finite(slopes.x_row) && Finite(slopes.y_row) && Finite(slopes.z_row);
    const bool valid_deformation = NeoHookeanKirchhoffStress3D(particle.deformation_gradient, 1.0, 0.0).ok();
    if (!valid_identity || !valid_extent || !valid_kinematics || !valid_deformation) {
      return absl::InvalidArgumentError("invalid elastic checkpoint particle");
    }
    previous_id = particle.particle_id;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<std::uint8_t>> EncodeSolidInteractionCheckpoint3D(const SolidInteractionCheckpoint3D& state
) {
  const auto validation = ValidateCheckpoint(state);
  if (!validation.ok()) {
    return validation;
  }
  std::vector<wire::BodyHistory> bodies;
  bodies.reserve(state.bodies.size());
  for (const auto& history : state.bodies) {
    const auto& body = history.body;
    bodies.emplace_back(
        body.body_id,
        body.mass,
        Pack(body.principal_inertia),
        PackPose(body.center, body.orientation),
        PackPose(history.previous_pose.center, history.previous_pose.orientation),
        Pack(body.linear_velocity),
        Pack(body.angular_velocity),
        Pack(body.force),
        Pack(body.torque)
    );
  }
  std::vector<wire::BladeSide> sides;
  sides.reserve(state.sides.size());
  for (const auto& entry : state.sides) {
    sides.emplace_back(entry.particle_id, entry.side.blade_id, entry.side.sign);
  }
  std::vector<wire::ElasticParticle> elastic_particles;
  elastic_particles.reserve(state.elastic_particles.size());
  for (const auto& particle : state.elastic_particles) {
    const auto& transfer = particle.transfer;
    elastic_particles.emplace_back(
        particle.particle_id,
        particle.body_id,
        Pack(transfer.sorted_world_position),
        Pack(transfer.trial_world_position),
        transfer.particle_mass,
        Pack(transfer.particle_velocity),
        PackMatrix(transfer.particle_velocity_slopes),
        particle.reference_volume,
        PackMatrix(particle.deformation_gradient)
    );
  }
  flatbuffers::FlatBufferBuilder builder;
  builder.ForceDefaults(true);
  const auto root = wire::CreateSolidInteractionCheckpointDirect(
      builder,
      1,
      state.accepted_step,
      &bodies,
      &sides,
      elastic_particles.empty() ? nullptr : &elastic_particles
  );
  wire::FinishSizePrefixedSolidInteractionCheckpointBuffer(builder, root);
  return std::vector<std::uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

absl::StatusOr<SolidInteractionCheckpoint3D> DecodeSolidInteractionCheckpoint3D(std::span<const std::uint8_t> bytes) {
  if (bytes.size() > kMaximumBytes) {
    return absl::ResourceExhaustedError("solid checkpoint byte capacity exceeded");
  }
  if (bytes.size() < 8 || reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(std::uint64_t) != 0) {
    return absl::InvalidArgumentError("truncated or unaligned solid checkpoint");
  }
  flatbuffers::Verifier verifier(bytes.data(), bytes.size());
  if (!wire::VerifySizePrefixedSolidInteractionCheckpointBuffer(verifier)
      || flatbuffers::GetPrefixedSize(bytes.data()) != bytes.size() - sizeof(flatbuffers::uoffset_t)) {
    return absl::InvalidArgumentError("invalid solid checkpoint buffer");
  }
  const auto* root = wire::GetSizePrefixedSolidInteractionCheckpoint(bytes.data());
  if (root->schema_version() != 1) {
    return absl::FailedPreconditionError("unsupported solid checkpoint version");
  }
  const auto* elastic_particles = root->elastic_particles();
  if (root->bodies()->size() > kMaximumRecords || root->sides()->size() > kMaximumRecords
      || (elastic_particles != nullptr && elastic_particles->size() > kMaximumRecords)) {
    return absl::ResourceExhaustedError("solid checkpoint record capacity exceeded");
  }

  SolidInteractionCheckpoint3D result{.accepted_step = root->accepted_step(), .bodies = {}, .sides = {}};
  result.bodies.reserve(root->bodies()->size());
  result.sides.reserve(root->sides()->size());
  for (const auto* history : *root->bodies()) {
    const auto pose = UnpackPose(history->pose());
    result.bodies.push_back({
        .body =
            {
                .body_id = history->body_id(),
                .mass = history->mass(),
                .principal_inertia = Unpack(&history->principal_inertia()),
                .center = pose.center,
                .orientation = pose.orientation,
                .linear_velocity = Unpack(&history->linear_velocity()),
                .angular_velocity = Unpack(&history->angular_velocity()),
                .force = Unpack(&history->force()),
                .torque = Unpack(&history->torque()),
            },
        .previous_pose = UnpackPose(history->previous_pose()),
    });
  }
  for (const auto* side : *root->sides()) {
    result.sides.push_back({side->particle_id(), {side->blade_id(), side->sign()}});
  }
  if (elastic_particles != nullptr) {
    result.elastic_particles.reserve(elastic_particles->size());
    for (const auto* particle : *elastic_particles) {
      result.elastic_particles.push_back({
          .particle_id = particle->particle_id(),
          .body_id = particle->body_id(),
          .transfer =
              {
                  .sorted_world_position = Unpack(&particle->sorted_position()),
                  .trial_world_position = Unpack(&particle->position()),
                  .particle_mass = particle->mass(),
                  .particle_velocity = Unpack(&particle->velocity()),
                  .particle_velocity_slopes = UnpackMatrix(particle->velocity_slopes()),
              },
          .reference_volume = particle->reference_volume(),
          .deformation_gradient = UnpackMatrix(particle->deformation_gradient()),
      });
    }
  }
  const auto validation = ValidateCheckpoint(result);
  if (!validation.ok()) {
    return validation;
  }
  return result;
}

}  // namespace fluidity
