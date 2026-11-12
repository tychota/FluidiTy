#include <array>
#include <cmath>
#include <cstdio>

#include "../src/plastic_solid_checkpoint_wire_3d.h"
#include "fluidity/host/plastic_solid_checkpoint_3d.h"
#if defined(FLUIDITY_M5_CUDA_SCENES)
#include "fluidity/host/cuda_dry_solid_step_3d.h"
#endif

namespace {

using fluidity::BlockIndex3D;
using fluidity::DrySolidMaterial3D;
using fluidity::DrySolidParticle3D;
using fluidity::DrySolidStep3D;
using fluidity::Mat3;
using fluidity::SparseMacGrid3D;

constexpr Mat3 kIdentity{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

bool TestPlasticCheckpointHandoff() {
  const BlockIndex3D block{0, 0, 0};
  const auto grid = SparseMacGrid3D::Create({{}, 1}, {&block, 1}, 1);
  if (!grid.ok()) {
    return false;
  }
  const DrySolidMaterial3D snow = fluidity::SnowParameters3D{2, 3, 0.025, 0.01, 4};
  const DrySolidParticle3D accepted{
      .world_position = {2.25, 2.25, 2.25},
      .world_velocity = {0.1, 0, 0},
      .velocity_slopes = {{0.1, -0.02, 0}, {0, 0.05, 0}, {0, 0, -0.1}},
      .mass = 1,
      .reference_volume = 0.001,
      .elastic_deformation = {{1, 0, 0}, {0, 0.975, 0}, {0, 0, 1}},
      .log_plastic_jacobian = std::log(0.8),
  };
  const auto same_vector = [](fluidity::Vec3 first, fluidity::Vec3 second) {
    return first.x == second.x && first.y == second.y && first.z == second.z;
  };
  const auto same_matrix = [same_vector](Mat3 first, Mat3 second) {
    return same_vector(first.x_row, second.x_row) && same_vector(first.y_row, second.y_row)
        && same_vector(first.z_row, second.z_row);
  };
  const auto same_particle = [same_vector,
                              same_matrix](const DrySolidParticle3D& first, const DrySolidParticle3D& second) {
    return same_vector(first.world_position, second.world_position)
        && same_vector(first.world_velocity, second.world_velocity)
        && same_matrix(first.velocity_slopes, second.velocity_slopes) && first.mass == second.mass
        && first.reference_volume == second.reference_volume
        && same_matrix(first.elastic_deformation, second.elastic_deformation)
        && first.log_plastic_jacobian == second.log_plastic_jacobian;
  };
  const auto captured = fluidity::CapturePlasticSolidCheckpointParticle3D(7, 9, 11, snow, accepted);
  if (!captured.ok()) {
    return false;
  }
  const auto packed = fluidity::internal::PackPlasticSolidCheckpointParticle3D(*captured);
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(builder.CreateVectorOfStructs(&packed, 1));
  using WireVector = fluidity::internal::PlasticCheckpointWireVector3D;
  const auto* wire_records = flatbuffers::GetRoot<WireVector>(builder.GetBufferPointer());
  if (wire_records->size() != 1) {
    return false;
  }
  const auto decoded = fluidity::internal::UnpackPlasticSolidCheckpointParticle3D(*wire_records->Get(0));
  if (decoded.particle_id != 7 || decoded.body_id != 9 || decoded.material_id != 11
      || !same_vector(decoded.transfer.sorted_world_position, accepted.world_position)) {
    return false;
  }
  auto restored = fluidity::RestoreDrySolidCheckpointParticles3D({&decoded, 1}, 9, 11, snow);
  if (!restored.ok() || restored->size() != 1 || !same_particle(accepted, restored->front())) {
    return false;
  }
  auto corrupt = *captured;
  corrupt.log_plastic_jacobian = 1000;
  if (fluidity::RestoreDrySolidCheckpointParticles3D({&corrupt, 1}, 9, 11, snow).ok()
      || fluidity::RestoreDrySolidCheckpointParticles3D({&*captured, 1}, 9, 12, snow).ok()) {
    return false;
  }
  const std::array duplicate{*captured, *captured};
  if (fluidity::RestoreDrySolidCheckpointParticles3D(duplicate, 9, 11, snow).ok()) {
    return false;
  }
  const DrySolidMaterial3D sand = fluidity::GranularParameters3D{2, 3, 0.5235987755982989, 0};
  auto sand_particle = accepted;
  sand_particle.elastic_deformation = kIdentity;
  sand_particle.log_plastic_jacobian = 0;
  const auto sand_record = fluidity::CapturePlasticSolidCheckpointParticle3D(5, 12, 13, sand, sand_particle);
  if (!sand_record.ok()) {
    return false;
  }
  const std::array bindings{
      fluidity::PlasticSolidCheckpointBinding3D{9, 11, snow},
      fluidity::PlasticSolidCheckpointBinding3D{12, 13, sand},
  };
  const std::array mixed_records{*sand_record, decoded};
  if (!fluidity::ValidatePlasticSolidCheckpointParticles3D(mixed_records, bindings).ok()) {
    return false;
  }
  const std::array reversed_bindings{bindings[1], bindings[0]};
  const std::array invalid_state{*sand_record, corrupt};
  auto duplicate_id = decoded;
  duplicate_id.particle_id = sand_record->particle_id;
  const std::array duplicate_ids{*sand_record, duplicate_id};
  if (fluidity::ValidatePlasticSolidCheckpointParticles3D(mixed_records, reversed_bindings).code()
          != absl::StatusCode::kInvalidArgument
      || fluidity::ValidatePlasticSolidCheckpointParticles3D(mixed_records, {&bindings[1], 1}).code()
          != absl::StatusCode::kInvalidArgument
      || fluidity::ValidatePlasticSolidCheckpointParticles3D(invalid_state, bindings).code()
          != absl::StatusCode::kFailedPrecondition
      || fluidity::ValidatePlasticSolidCheckpointParticles3D(duplicate_ids, bindings).code()
          != absl::StatusCode::kInvalidArgument) {
    return false;
  }
  const auto decode_batch = [&](const std::array<fluidity::PlasticSolidCheckpointParticle3D, 2>& records,
                                std::size_t maximum_particle_count,
                                std::uint32_t schema_version = 2) {
    const std::array wire_batch{
        fluidity::internal::PackPlasticSolidCheckpointParticle3D(records[0]),
        fluidity::internal::PackPlasticSolidCheckpointParticle3D(records[1]),
    };
    flatbuffers::FlatBufferBuilder batch_builder;
    batch_builder.Finish(batch_builder.CreateVectorOfStructs(wire_batch.data(), wire_batch.size()));
    const auto* batch = flatbuffers::GetRoot<WireVector>(batch_builder.GetBufferPointer());
    return fluidity::internal::DecodePlasticCheckpointWireVector3D(
        schema_version,
        batch,
        bindings,
        maximum_particle_count
    );
  };
  const auto decoded_batch = decode_batch(mixed_records, 2);
  const auto absent = fluidity::internal::DecodePlasticCheckpointWireVector3D(1, nullptr, bindings, 0);
  if (!decoded_batch.ok() || decoded_batch->size() != 2 || (*decoded_batch)[0].particle_id != 5
      || (*decoded_batch)[1].particle_id != 7 || !absent.ok() || !absent->empty()
      || decode_batch(mixed_records, 1).status().code() != absl::StatusCode::kResourceExhausted
      || decode_batch(invalid_state, 2).status().code() != absl::StatusCode::kFailedPrecondition
      || decode_batch(mixed_records, 2, 1).status().code() != absl::StatusCode::kInvalidArgument
      || decode_batch(mixed_records, 2, 3).status().code() != absl::StatusCode::kFailedPrecondition) {
    return false;
  }
  flatbuffers::FlatBufferBuilder owning_builder;
  const auto empty_vector = fluidity::internal::EncodePlasticCheckpointWireVector3D(owning_builder, {}, bindings, 0);
  if (!empty_vector.ok() || !empty_vector->IsNull() || owning_builder.GetSize() != 0) {
    return false;
  }
  const auto encoded_batch =
      fluidity::internal::EncodePlasticCheckpointWireVector3D(owning_builder, mixed_records, bindings, 2);
  if (!encoded_batch.ok() || encoded_batch->IsNull()) {
    return false;
  }
  const auto staged_size = owning_builder.GetSize();
  const auto over_capacity =
      fluidity::internal::EncodePlasticCheckpointWireVector3D(owning_builder, mixed_records, bindings, 1);
  const auto rejected_state =
      fluidity::internal::EncodePlasticCheckpointWireVector3D(owning_builder, invalid_state, bindings, 2);
  if (over_capacity.status().code() != absl::StatusCode::kResourceExhausted
      || rejected_state.status().code() != absl::StatusCode::kFailedPrecondition
      || owning_builder.GetSize() != staged_size) {
    return false;
  }
  owning_builder.Finish(*encoded_batch);
  const auto* encoded_wire_records = flatbuffers::GetRoot<WireVector>(owning_builder.GetBufferPointer());
  static_assert(fluidity::internal::InteractionCheckpointVersionForPlasticPayload3D(0) == 1);
  const auto replay = fluidity::internal::DecodePlasticCheckpointWireVector3D(
      fluidity::internal::InteractionCheckpointVersionForPlasticPayload3D(mixed_records.size()),
      encoded_wire_records,
      bindings,
      2
  );
  if (!replay.ok() || replay->size() != 2 || (*replay)[0].particle_id != 5 || (*replay)[1].particle_id != 7
      || (*replay)[1].log_plastic_jacobian != accepted.log_plastic_jacobian) {
    return false;
  }
  auto uninterrupted = accepted;
  const DrySolidStep3D step{.duration = 0.01, .acoustic_cfl = 0.25};
#if defined(FLUIDITY_M5_CUDA_SCENES)
  const auto first = fluidity::CudaAdvanceDrySolid3D(*grid, snow, step, {0x1p-42, 0x1p-38, 128}, {&uninterrupted, 1});
  const auto second = fluidity::CudaAdvanceDrySolid3D(*grid, snow, step, {0x1p-42, 0x1p-38, 128}, *restored);
#else
  const auto first = fluidity::AdvanceDrySolid3D(*grid, snow, step, {&uninterrupted, 1});
  const auto second = fluidity::AdvanceDrySolid3D(*grid, snow, step, *restored);
#endif
  return first.ok() && second.ok() && same_particle(uninterrupted, restored->front());
}

}  // namespace

int main() {
  if (!TestPlasticCheckpointHandoff()) {
    std::fprintf(stderr, "M5 plastic checkpoint replay fixture failed\n");
    return 1;
  }
  return 0;
}
