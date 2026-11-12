#include "fluidity/host/solid_interaction_checkpoint_3d.h"

#include <algorithm>
#include <cstddef>

#include "fluidity/host/transfers_3d.h"
#include "liquid_cache_3d_internal.h"
#include "plastic_solid_checkpoint_wire_3d.h"
#include "solid_interaction_checkpoint_generated.h"

namespace fluidity {
namespace {

namespace wire = solid::v1;
using liquid_cache_internal::Pack;
using liquid_cache_internal::Unpack;
constexpr std::size_t kMaximumRecords = 65536;
constexpr std::size_t kMaximumBytes = 32 * 1024 * 1024;
constexpr std::uint32_t kSceneBoundCheckpointVersion = 3;

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

namespace {

absl::StatusOr<std::vector<std::uint8_t>> EncodeInteractionCheckpoint(
    const SolidInteractionCheckpoint3D& state,
    std::span<const std::uint8_t> liquid_bytes,
    std::span<const PlasticSolidCheckpointBinding3D> bindings,
    const LiquidDigest3D* resolved_fingerprint = nullptr
) {
  if (liquid_bytes.size() > kMaximumBytes) {
    return absl::ResourceExhaustedError("liquid payload exceeds coupled checkpoint capacity");
  }
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
  const auto plastic =
      internal::EncodePlasticCheckpointWireVector3D(builder, state.plastic_particles, bindings, kMaximumRecords);
  if (!plastic.ok()) {
    return plastic.status();
  }
  const auto body_vector = builder.CreateVectorOfStructs(bodies);
  const auto side_vector = builder.CreateVectorOfStructs(sides);
  const auto elastic_vector = elastic_particles.empty()
      ? flatbuffers::Offset<flatbuffers::Vector<const wire::ElasticParticle*>>{}
      : builder.CreateVectorOfStructs(elastic_particles);
  flatbuffers::Offset<flatbuffers::Vector<std::uint8_t>> liquid_vector;
  if (!liquid_bytes.empty()) {
    builder.ForceVectorAlignment(liquid_bytes.size(), sizeof(std::uint8_t), 8);
    liquid_vector = builder.CreateVector(liquid_bytes.data(), liquid_bytes.size());
  }
  const auto packed_fingerprint = resolved_fingerprint == nullptr ? cache::v1::Digest{} : Pack(*resolved_fingerprint);
  const std::uint32_t schema_version = resolved_fingerprint == nullptr
      ? internal::InteractionCheckpointVersionForPlasticPayload3D(state.plastic_particles.size())
      : kSceneBoundCheckpointVersion;
  const auto root = wire::CreateSolidInteractionCheckpoint(
      builder,
      schema_version,
      state.accepted_step,
      body_vector,
      side_vector,
      elastic_vector,
      liquid_vector,
      *plastic,
      resolved_fingerprint == nullptr ? nullptr : &packed_fingerprint
  );
  wire::FinishSizePrefixedSolidInteractionCheckpointBuffer(builder, root);
  if (builder.GetSize() > kMaximumBytes) {
    return absl::ResourceExhaustedError("interaction checkpoint exceeds byte capacity");
  }
  return std::vector<std::uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

absl::StatusOr<SolidInteractionCheckpoint3D> DecodeInteractionCheckpoint(
    std::span<const std::uint8_t> bytes,
    LiquidState3D* liquid,
    std::span<const PlasticSolidCheckpointBinding3D> bindings,
    const LiquidDigest3D* resolved_fingerprint = nullptr,
    LiquidResumeMetadata3D* resume = nullptr,
    LiquidCacheLimits3D limits = {}
) {
  if (bytes.size() > std::min(kMaximumBytes, limits.maximum_bytes)) {
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
  const bool scene_bound = root->schema_version() == kSceneBoundCheckpointVersion;
  if (scene_bound != (resolved_fingerprint != nullptr)) {
    return absl::FailedPreconditionError("interaction checkpoint requires a matching scene-bound decoder");
  }
  const auto* stored_fingerprint = root->resolved_fingerprint();
  if (scene_bound != (stored_fingerprint != nullptr)) {
    return absl::InvalidArgumentError("interaction checkpoint version and scene fingerprint disagree");
  }
  if (scene_bound && Unpack(stored_fingerprint) != *resolved_fingerprint) {
    return absl::FailedPreconditionError("interaction checkpoint resolved scene fingerprint differs");
  }
  auto plastic = internal::DecodePlasticCheckpointWireVector3D(
      scene_bound ? 2 : root->schema_version(),
      root->plastic_particles(),
      bindings,
      kMaximumRecords
  );
  if (!plastic.ok()) {
    return plastic.status();
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
  result.plastic_particles = std::move(*plastic);
  const auto validation = ValidateCheckpoint(result);
  if (!validation.ok()) {
    return validation;
  }
  const auto* payload = root->liquid_state();
  if ((payload == nullptr) != (liquid == nullptr)) {
    return absl::InvalidArgumentError("interaction checkpoint kind does not match decoder");
  }
  if (payload != nullptr) {
    if (resume != nullptr) {
      auto checkpoint = DecodeLiquidCheckpoint3D({payload->data(), payload->size()}, limits);
      if (!checkpoint.ok()) {
        return checkpoint.status();
      }
      if (checkpoint->state.clock.accepted_step_ordinal != result.accepted_step
          || checkpoint->resume.resolved_fingerprint != *resolved_fingerprint) {
        return absl::InvalidArgumentError("coupled checkpoint liquid resume identity disagrees");
      }
      *liquid = std::move(checkpoint->state);
      *resume = checkpoint->resume;
      return result;
    }
    auto decoded = DecodeLiquidState3D({payload->data(), payload->size()}, limits);
    if (!decoded.ok()) {
      return decoded.status();
    }
    if (decoded->clock.accepted_step_ordinal != result.accepted_step) {
      return absl::InvalidArgumentError("coupled checkpoint accepted steps disagree");
    }
    *liquid = std::move(*decoded);
  }
  return result;
}

bool LiquidMatchesResolvedScene(const LiquidState3D& liquid, const LiquidResolvedSceneView3D& scene) {
  const auto actual = liquid.geometry.world_origin;
  const auto expected = scene.geometry.world_origin;
  return liquid.geometry.cell_width == scene.geometry.cell_width && actual.x == expected.x && actual.y == expected.y
      && actual.z == expected.z && liquid.random_seed == scene.random_seed;
}

}  // namespace

absl::StatusOr<std::vector<std::uint8_t>> EncodeSolidInteractionCheckpoint3D(
    const SolidInteractionCheckpoint3D& state,
    std::span<const PlasticSolidCheckpointBinding3D> bindings
) {
  return EncodeInteractionCheckpoint(state, {}, bindings);
}

absl::StatusOr<SolidInteractionCheckpoint3D> DecodeSolidInteractionCheckpoint3D(
    std::span<const std::uint8_t> bytes,
    std::span<const PlasticSolidCheckpointBinding3D> bindings
) {
  return DecodeInteractionCheckpoint(bytes, nullptr, bindings);
}

absl::StatusOr<std::vector<std::uint8_t>> EncodeCoupledInteractionCheckpoint3D(
    const LiquidState3D& liquid,
    const SolidInteractionCheckpoint3D& solid,
    std::span<const PlasticSolidCheckpointBinding3D> bindings
) {
  if (liquid.clock.accepted_step_ordinal != solid.accepted_step) {
    return absl::InvalidArgumentError("coupled checkpoint accepted steps disagree");
  }
  const auto payload = EncodeLiquidState3D(liquid);
  if (!payload.ok()) {
    return payload.status();
  }
  return EncodeInteractionCheckpoint(solid, payload->values(), bindings);
}

absl::StatusOr<CoupledInteractionCheckpoint3D> DecodeCoupledInteractionCheckpoint3D(
    std::span<const std::uint8_t> bytes,
    std::span<const PlasticSolidCheckpointBinding3D> bindings
) {
  LiquidState3D liquid;
  auto solid = DecodeInteractionCheckpoint(bytes, &liquid, bindings);
  if (!solid.ok()) {
    return solid.status();
  }
  return CoupledInteractionCheckpoint3D{.liquid = std::move(liquid), .solid = std::move(*solid)};
}

namespace {

absl::StatusOr<std::vector<std::uint8_t>> EncodeResolvedCheckpoint(
    const LiquidResolvedSceneView3D& scene,
    const LiquidState3D& liquid,
    const SolidInteractionCheckpoint3D& solid,
    const LiquidResumeMetadata3D* resume,
    LiquidCacheLimits3D limits
) {
  if (scene.model_version != kPlasticBodyResolvedModel3D || !LiquidMatchesResolvedScene(liquid, scene)
      || liquid.clock.accepted_step_ordinal != solid.accepted_step) {
    return absl::InvalidArgumentError("coupled checkpoint state does not match its resolved scene");
  }
  const auto fingerprint = ResolvedSceneFingerprint(scene, limits);
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  if (resume != nullptr && resume->resolved_fingerprint != *fingerprint) {
    return absl::InvalidArgumentError("liquid resume metadata does not match the resolved scene");
  }
  const auto payload =
      resume == nullptr ? EncodeLiquidState3D(liquid, limits) : EncodeLiquidCheckpoint3D(liquid, *resume, limits);
  if (!payload.ok()) {
    return payload.status();
  }
  auto encoded = EncodeInteractionCheckpoint(solid, payload->values(), scene.plastic_bodies, &*fingerprint);
  if (encoded.ok() && encoded->size() > limits.maximum_bytes) {
    return absl::ResourceExhaustedError("coupled checkpoint exceeds cache byte capacity");
  }
  return encoded;
}

absl::StatusOr<CoupledInteractionCheckpoint3D> DecodeResolvedCheckpoint(
    const LiquidResolvedSceneView3D& scene,
    std::span<const std::uint8_t> bytes,
    LiquidResumeMetadata3D* resume,
    LiquidCacheLimits3D limits
) {
  if (scene.model_version != kPlasticBodyResolvedModel3D) {
    return absl::InvalidArgumentError("coupled checkpoint requires a resolved plastic scene");
  }
  const auto fingerprint = ResolvedSceneFingerprint(scene, limits);
  if (!fingerprint.ok()) {
    return fingerprint.status();
  }
  LiquidState3D liquid;
  LiquidResumeMetadata3D staged_resume;
  auto solid = DecodeInteractionCheckpoint(
      bytes,
      &liquid,
      scene.plastic_bodies,
      &*fingerprint,
      resume == nullptr ? nullptr : &staged_resume,
      limits
  );
  if (!solid.ok()) {
    return solid.status();
  }
  if (!LiquidMatchesResolvedScene(liquid, scene)) {
    return absl::InvalidArgumentError("coupled checkpoint liquid does not match its resolved scene");
  }
  if (resume != nullptr) {
    *resume = staged_resume;
  }
  return CoupledInteractionCheckpoint3D{.liquid = std::move(liquid), .solid = std::move(*solid)};
}

}  // namespace

absl::StatusOr<std::vector<std::uint8_t>> EncodeResolvedCoupledCheckpoint3D(
    const LiquidResolvedSceneView3D& scene,
    const LiquidState3D& liquid,
    const SolidInteractionCheckpoint3D& solid
) {
  return EncodeResolvedCheckpoint(scene, liquid, solid, nullptr, {});
}

absl::StatusOr<std::vector<std::uint8_t>> EncodeResolvedCoupledCheckpoint3D(
    const LiquidResolvedSceneView3D& scene,
    const LiquidState3D& liquid,
    const SolidInteractionCheckpoint3D& solid,
    const LiquidResumeMetadata3D& resume,
    const LiquidCacheLimits3D& limits
) {
  return EncodeResolvedCheckpoint(scene, liquid, solid, &resume, limits);
}

absl::StatusOr<CoupledInteractionCheckpoint3D>
DecodeResolvedCoupledCheckpoint3D(const LiquidResolvedSceneView3D& scene, std::span<const std::uint8_t> bytes) {
  return DecodeResolvedCheckpoint(scene, bytes, nullptr, {});
}

absl::StatusOr<CoupledInteractionCheckpoint3D> DecodeResolvedCoupledCheckpoint3D(
    const LiquidResolvedSceneView3D& scene,
    std::span<const std::uint8_t> bytes,
    LiquidResumeMetadata3D& resume,
    const LiquidCacheLimits3D& limits
) {
  return DecodeResolvedCheckpoint(scene, bytes, &resume, limits);
}

}  // namespace fluidity
