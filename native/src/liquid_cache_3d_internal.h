#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <tuple>

#include "common_generated.h"
#include "fluidity/host/liquid_cache_3d.h"

namespace fluidity::liquid_cache_internal {
namespace wire = cache::v1;
inline absl::Status Error(absl::StatusCode code) {
  return absl::Status(code, {});
}
inline absl::Status Corrupt() {
  return Error(absl::StatusCode::kDataLoss);
}
inline bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
inline bool Positive(double value) {
  return std::isfinite(value) && value > 0;
}
inline bool Nonnegative(double value) {
  return std::isfinite(value) && value >= 0;
}
inline double Canonical(double value) {
  return value == 0 ? 0.0 : value;
}
inline wire::Vector3 Pack(Vec3 value) {
  return {value.x, value.y, value.z};
}
inline wire::Vector3 PackCanonical(Vec3 value) {
  return {Canonical(value.x), Canonical(value.y), Canonical(value.z)};
}
inline Vec3 Unpack(const wire::Vector3* value) {
  return {value->x(), value->y(), value->z()};
}
inline wire::Digest Pack(const LiquidDigest3D& digest) {
  return wire::Digest(flatbuffers::span<const std::uint8_t, 32>(digest.data(), digest.size()));
}
inline LiquidDigest3D Unpack(const wire::Digest* digest) {
  LiquidDigest3D result;
  std::copy(digest->bytes()->begin(), digest->bytes()->end(), result.begin());
  return result;
}
inline wire::Clock Pack(LiquidClock3D clock) {
  return {clock.time, clock.frame_ordinal, clock.substep_ordinal, clock.accepted_step_ordinal, clock.epoch};
}
inline LiquidClock3D Unpack(const wire::Clock* clock) {
  return {clock->time(), clock->frame(), clock->substep(), clock->accepted_step(), clock->epoch()};
}
inline bool ValidClock(LiquidClock3D clock) {
  return Nonnegative(clock.time) && clock.frame_ordinal <= clock.accepted_step_ordinal
      && clock.epoch >= clock.accepted_step_ordinal
      && clock.substep_ordinal <= clock.accepted_step_ordinal - clock.frame_ordinal;
}
inline bool ValidParticle(const LiquidParticleState3D& particle) {
  return Finite(particle.world_position) && Finite(particle.world_velocity) && Finite(particle.velocity_slopes.x_row)
      && Finite(particle.velocity_slopes.y_row) && Finite(particle.velocity_slopes.z_row) && Positive(particle.mass)
      && Positive(particle.reference_volume) && Positive(particle.volume_ratio);
}
// Generated bool accessors normalize every nonzero byte; inspect the verified
// raw field first so corrupt values cannot silently become canonical true.
template <typename Table>
bool ValidBoolean(const Table* table, typename Table::FlatBuffersVTableOffset field) {
  const auto* base = reinterpret_cast<const flatbuffers::Table*>(table);
  const auto* value = base->GetAddressOf(static_cast<flatbuffers::voffset_t>(field));
  return value == nullptr || *value <= 1;
}
inline bool ValidParticleBooleans(const wire::Particle* particle) {
  return ValidBoolean(particle, wire::Particle::VT_AFFINE_ENABLED)
      && ValidBoolean(particle, wire::Particle::VT_PIC_INITIALIZED);
}
inline LiquidParticleState3D Unpack(const wire::Particle* particle) {
  const Mat3 slopes{
      .x_row = Unpack(&particle->slopes()->x_row()),
      .y_row = Unpack(&particle->slopes()->y_row()),
      .z_row = Unpack(&particle->slopes()->z_row()),
  };
  return {
      .id = particle->id(),
      .tag = particle->tag(),
      .world_position = Unpack(particle->position()),
      .world_velocity = Unpack(particle->velocity()),
      .velocity_slopes = slopes,
      .affine_enabled = particle->affine_enabled(),
      .mass = particle->mass(),
      .reference_volume = particle->reference_volume(),
      .volume_ratio = particle->volume_ratio(),
      .side_tag = particle->side(),
      .pic_initialized = particle->pic_initialized(),
  };
}
inline flatbuffers::Offset<wire::Particle>
PackParticle(flatbuffers::FlatBufferBuilder& builder, const LiquidParticleState3D& particle, bool canonical = false) {
  const auto vector = [&](Vec3 value) {
    return canonical ? PackCanonical(value) : Pack(value);
  };
  const auto position = vector(particle.world_position);
  const auto velocity = vector(particle.world_velocity);
  const wire::Matrix3 slopes(
      vector(particle.velocity_slopes.x_row),
      vector(particle.velocity_slopes.y_row),
      vector(particle.velocity_slopes.z_row)
  );
  return wire::CreateParticle(
      builder,
      particle.id,
      particle.tag,
      &position,
      &velocity,
      &slopes,
      particle.affine_enabled,
      particle.mass,
      particle.reference_volume,
      particle.volume_ratio,
      particle.side_tag,
      particle.pic_initialized
  );
}
inline bool Fits(LiquidStateCapacities3D used, LiquidStateCapacities3D limit) {
  return used.particles <= limit.particles && used.blocks <= limit.blocks && used.cells <= limit.cells
      && used.faces <= limit.faces && used.sources <= limit.sources;
}
// Fixed-buffer allocation is reserved with nothrow before FlatBuffers starts.
// Each codec budgets 512 bytes per table (payload, vtable, alignment and builder
// scratch) plus 4096 root bytes. No table exceeds 15 fields/200 payload bytes.
// Hitting allocate twice is an implementation invariant violation, not input OOM.
class ReservedAllocator final : public flatbuffers::Allocator {
 public:
  explicit ReservedAllocator(std::span<std::uint8_t> memory) : memory_(memory) {
  }
  std::uint8_t* allocate(std::size_t size) override {
    if (used_ || size > memory_.size()) {
      std::abort();
    }
    used_ = true;
    return memory_.data();
  }
  void deallocate(std::uint8_t*, std::size_t) override {
  }

 private:
  std::span<std::uint8_t> memory_;
  bool used_ = false;
};
inline absl::StatusOr<std::size_t> Budget(std::size_t tables, LiquidCacheLimits3D limits) {
  constexpr std::size_t base = 4096;
  const auto ceiling = std::min(limits.maximum_bytes, static_cast<std::size_t>(FLATBUFFERS_MAX_BUFFER_SIZE) - 1);
  if (ceiling < base || tables > (ceiling - base) / 512) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  return base + tables * 512;
}
template <typename Root, typename Create>
absl::StatusOr<LiquidCacheBytes3D>
Build(std::size_t table_count, LiquidCacheLimits3D limits, const char* file_identifier, Create create) {
  auto byte_budget = Budget(table_count, limits);
  if (!byte_budget.ok()) {
    return byte_budget.status();
  }
  auto memory = LiquidCacheBytes3D::Create(*byte_budget);
  if (!memory.ok()) {
    return memory.status();
  }
  auto resize = memory->Resize(*byte_budget);
  if (!resize.ok()) {
    return resize;
  }
  ReservedAllocator allocator(memory->values());
  flatbuffers::FlatBufferBuilder builder(*byte_budget, &allocator);
  builder.ForceDefaults(true);
  auto root = create(builder);
  if (!root.ok()) {
    return root.status();
  }
  builder.Finish(*root, file_identifier);
  auto* finished = flatbuffers::GetMutableRoot<Root>(builder.GetBufferPointer());
  if (!finished->mutate_byte_length(builder.GetSize())) {
    return Error(absl::StatusCode::kInternal);
  }
  // Reuse the reserved owner: exact active size is content, unused bytes are not.
  std::memmove(memory->values().data(), builder.GetBufferPointer(), builder.GetSize());
  resize = memory->Resize(builder.GetSize());
  if (!resize.ok()) {
    return resize;
  }
  return std::move(*memory);
}
template <typename Root>
absl::StatusOr<const Root*> Verify(
    std::span<const std::uint8_t> bytes,
    const char* expected_file_identifier,
    LiquidCacheLimits3D limits,
    std::uint32_t maximum_schema_version = 1
) {
  if (bytes.size() > limits.maximum_bytes || bytes.size() >= FLATBUFFERS_MAX_BUFFER_SIZE) {
    return Error(absl::StatusCode::kResourceExhausted);
  }
  if (bytes.size() < 8 || reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(std::uint64_t) != 0) {
    return Corrupt();
  }
  flatbuffers::Verifier::Options options;
  options.max_depth = 16;
  // At least four bytes per table reference; bounds verifier work by input size.
  options.max_tables = static_cast<flatbuffers::uoffset_t>(bytes.size() / 4);
  flatbuffers::Verifier verifier(bytes.data(), bytes.size(), options);
  if (!verifier.VerifyBuffer<Root>(expected_file_identifier)) {
    return Corrupt();
  }
  const auto* root = flatbuffers::GetRoot<Root>(bytes.data());
  if (root->schema_version() == 0 || root->schema_version() > maximum_schema_version) {
    return Error(absl::StatusCode::kFailedPrecondition);
  }
  if (root->byte_length() != bytes.size()) {
    return Corrupt();
  }
  return root;
}
// Offsets live in checked storage; no std::vector or throwing allocator is used.
template <typename Table, typename Create>
absl::StatusOr<flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Table>>>>
BuildVector(flatbuffers::FlatBufferBuilder& builder, std::size_t count, Create create) {
  auto offsets = LiquidRecords3D<std::uint32_t>::Create(count);
  if (!offsets.ok()) {
    return offsets.status();
  }
  auto status = offsets->Resize(count);
  if (!status.ok()) {
    return status;
  }
  for (std::size_t index = 0; index < count; ++index) {
    (*offsets)[index] = create(index).o;
  }
  builder.StartVector<flatbuffers::Offset<Table>>(count);
  for (std::size_t index = count; index > 0; --index) {
    builder.PushElement(flatbuffers::Offset<Table>((*offsets)[index - 1]));
  }
  return flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Table>>>(builder.EndVector(count));
}
template <typename Record, typename Source, typename Decode>
absl::Status Materialize(LiquidRecords3D<Record>& destination, const Source* source, Decode decode) {
  auto records = LiquidRecords3D<Record>::Create(source->size());
  if (!records.ok()) {
    return records.status();
  }
  auto status = records->Resize(source->size());
  if (!status.ok()) {
    return status;
  }
  for (std::size_t index = 0; index < source->size(); ++index) {
    (*records)[index] = decode(source->Get(static_cast<flatbuffers::uoffset_t>(index)));
  }
  destination = std::move(*records);
  return absl::OkStatus();
}
LiquidDigest3D DomainDigest(std::string_view domain, std::span<const std::uint8_t> bytes);
}  // namespace fluidity::liquid_cache_internal
