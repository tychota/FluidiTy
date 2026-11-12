#include "fluidity/host/liquid_cache_3d.h"

#include <blake3.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>

#include "frame_manifest_generated.h"
#include "liquid_cache_3d_internal.h"
#include "liquid_state_generated.h"
#include "point_frame_generated.h"

namespace fluidity {
namespace cache_detail = liquid_cache_internal;
namespace wire = cache::v1;
namespace {
// Native enums may grow independently; existing wire tags must never drift.
static_assert(static_cast<int>(FaceNormalAxis::kX) == static_cast<int>(wire::Axis::X));
static_assert(static_cast<int>(FaceNormalAxis::kY) == static_cast<int>(wire::Axis::Y));
static_assert(static_cast<int>(FaceNormalAxis::kZ) == static_cast<int>(wire::Axis::Z));
static_assert(static_cast<int>(LiquidCellOccupancy3D::kAir) == static_cast<int>(wire::Occupancy::Air));
static_assert(static_cast<int>(LiquidCellOccupancy3D::kLiquid) == static_cast<int>(wire::Occupancy::Liquid));
static_assert(static_cast<int>(LiquidCellOccupancy3D::kSolid) == static_cast<int>(wire::Occupancy::Solid));
static_assert(static_cast<int>(LiquidFaceValidity3D::kMissing) == static_cast<int>(wire::Validity::Missing));
static_assert(static_cast<int>(LiquidFaceValidity3D::kValid) == static_cast<int>(wire::Validity::Valid));
static_assert(static_cast<int>(LiquidFaceValidity3D::kPredicted) == static_cast<int>(wire::Validity::Predicted));
static_assert(static_cast<int>(LiquidFaceValidity3D::kLowMass) == static_cast<int>(wire::Validity::LowMass));
#define CACHE_TRY(expression) \
  do { \
    const auto status = (expression); \
    if (!status.ok()) { \
      return status; \
    } \
  } while (false)
absl::Status Invalid() {
  return cache_detail::Error(absl::StatusCode::kInvalidArgument);
}
LiquidStateCapacities3D Counts(const LiquidState3D& state) {
  return {
      state.particles.size(),
      state.blocks.size(),
      std::max(state.cells.size(), state.cavity_gas.size()),
      state.faces.size(),
      state.sources.size(),
  };
}
LiquidStateCapacities3D Counts(const wire::LiquidState* state) {
  return {
      state->particles()->size(),
      state->blocks()->size(),
      std::max(state->cells()->size(), state->cavity() ? state->cavity()->gas()->size() : 0U),
      state->faces()->size(),
      state->sources()->size(),
  };
}
absl::Status BoundCounts(LiquidStateCapacities3D record_counts, LiquidCacheLimits3D limits) {
  constexpr LiquidStateCapacities3D hard_record_limits{1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20};
  if (!cache_detail::Fits(record_counts, limits.maximum_records)
      || !cache_detail::Fits(record_counts, hard_record_limits)) {
    return cache_detail::Error(absl::StatusCode::kResourceExhausted);
  }
  return absl::OkStatus();
}
wire::Capacities PackCapacity(LiquidStateCapacities3D capacity) {
  return {capacity.particles, capacity.blocks, capacity.cells, capacity.faces, capacity.sources};
}
absl::StatusOr<LiquidStateCapacities3D> ReadCapacity(const wire::Capacities* capacity) {
  for (auto count :
       {capacity->particles(), capacity->blocks(), capacity->cells(), capacity->faces(), capacity->sources()}) {
    if (count > static_cast<std::uint64_t>(PTRDIFF_MAX) / sizeof(LiquidParticleState3D) / 5) {
      return cache_detail::Corrupt();
    }
  }
  return LiquidStateCapacities3D{
      static_cast<std::size_t>(capacity->particles()),
      static_cast<std::size_t>(capacity->blocks()),
      static_cast<std::size_t>(capacity->cells()),
      static_cast<std::size_t>(capacity->faces()),
      static_cast<std::size_t>(capacity->sources()),
  };
}
bool ValidResume(const LiquidResumeMetadata3D& resume, LiquidStateCapacities3D record_counts) {
  const auto reserved_capacities = resume.capacities;
  const auto replay_reservations = resume.capacity_replay_reservations;
  const bool replay_reservations_valid = replay_reservations == LiquidStateCapacities3D{}
      || (cache_detail::Fits(reserved_capacities, replay_reservations) && reserved_capacities != replay_reservations);
  constexpr std::size_t maximum = static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(LiquidParticleState3D) / 5;
  const LiquidStateCapacities3D reservation_count_limits{maximum, maximum, maximum, maximum, maximum};
  const bool execution_id_matches = resume.execution_id == LiquidExecutionId3D(resume.resolved_fingerprint);
  return cache_detail::Fits(record_counts, reserved_capacities) && replay_reservations_valid
      && cache_detail::Fits(reserved_capacities, reservation_count_limits)
      && cache_detail::Fits(replay_reservations, reservation_count_limits) && execution_id_matches;
}
absl::StatusOr<LiquidResumeMetadata3D>
ReadResume(const wire::ResumeMetadata* wire_resume, LiquidStateCapacities3D record_counts) {
  if (!wire_resume) {
    return cache_detail::Corrupt();
  }
  auto reserved_capacities = ReadCapacity(wire_resume->capacities());
  auto replay_reservations = ReadCapacity(wire_resume->replay_capacities());
  if (!reserved_capacities.ok() || !replay_reservations.ok()) {
    return cache_detail::Corrupt();
  }
  LiquidResumeMetadata3D resume_metadata{
      cache_detail::Unpack(wire_resume->resolved_fingerprint()),
      cache_detail::Unpack(wire_resume->execution_id()),
      *reserved_capacities,
      *replay_reservations,
  };
  if (!ValidResume(resume_metadata, record_counts)) {
    return cache_detail::Corrupt();
  }
  return resume_metadata;
}
auto Key(const wire::Index3* index) {
  return std::tuple(index->z(), index->y(), index->x());
}
// Signed ownership is floor division, including negative coordinates.
std::int32_t BlockCoordinate(std::int32_t coordinate) {
  return coordinate / 4 - (coordinate % 4 < 0 ? 1 : 0);
}
std::size_t Owner(const wire::LiquidState* state, const wire::Index3* index) {
  const auto key = std::tuple(BlockCoordinate(index->z()), BlockCoordinate(index->y()), BlockCoordinate(index->x()));
  std::size_t first = 0;
  std::size_t last = state->blocks()->size();
  while (first < last) {
    const auto middle = first + (last - first) / 2;
    if (Key(state->blocks()->Get(static_cast<flatbuffers::uoffset_t>(middle))->index()) < key) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  if (first == state->blocks()->size()
      || Key(state->blocks()->Get(static_cast<flatbuffers::uoffset_t>(first))->index()) != key) {
    return state->blocks()->size();
  }
  return first;
}
// Run semantic checks on verified, bounded wire views, before allocating any
// decoded record owner. Stable particle order is checked for unique IDs, not sorted.
absl::Status ValidateWireState(const wire::LiquidState* state, LiquidCacheLimits3D limits) {
  CACHE_TRY(BoundCounts(Counts(state), limits));
  if (!cache_detail::Finite(cache_detail::Unpack(state->world_origin())) || !cache_detail::Positive(state->cell_width())
      || !cache_detail::ValidClock(cache_detail::Unpack(state->clock())) || state->next_particle_id() == 0) {
    return cache_detail::Corrupt();
  }
  for (flatbuffers::uoffset_t index = 0; index < state->particles()->size(); ++index) {
    const auto* particle = state->particles()->Get(index);
    if (!cache_detail::ValidParticleEncoding(particle) || particle->id() == 0
        || particle->id() >= state->next_particle_id()
        || !cache_detail::ValidParticle(cache_detail::Unpack(particle))) {
      return cache_detail::Corrupt();
    }
    for (flatbuffers::uoffset_t previous = 0; previous < index; ++previous) {
      if (state->particles()->Get(previous)->id() == particle->id()) {
        return cache_detail::Corrupt();
      }
    }
  }
  for (flatbuffers::uoffset_t ordinal = 0; ordinal < state->blocks()->size(); ++ordinal) {
    const auto* index = state->blocks()->Get(ordinal)->index();
    for (const auto coordinate : {index->x(), index->y(), index->z()}) {
      if (coordinate < INT32_MIN / 4 || coordinate > (INT32_MAX - 3) / 4) {
        return cache_detail::Corrupt();
      }
    }
    if (ordinal > 0 && Key(state->blocks()->Get(ordinal - 1)->index()) >= Key(index)) {
      return cache_detail::Corrupt();
    }
  }
  for (flatbuffers::uoffset_t ordinal = 0; ordinal < state->cells()->size(); ++ordinal) {
    const auto* cell = state->cells()->Get(ordinal);
    if (cell->occupancy() > wire::Occupancy::Solid || !std::isfinite(cell->previous_pressure())
        || Owner(state, cell->index()) == state->blocks()->size()
        || (ordinal > 0 && Key(state->cells()->Get(ordinal - 1)->index()) >= Key(cell->index()))) {
      return cache_detail::Corrupt();
    }
  }
  for (flatbuffers::uoffset_t ordinal = 0; ordinal < state->faces()->size(); ++ordinal) {
    const auto* face = state->faces()->Get(ordinal);
    if (!cache_detail::ValidBoolean(face, wire::Face::VT_STRUCTURALLY_ELIGIBLE) || face->axis() > wire::Axis::Z
        || face->validity() > wire::Validity::LowMass || !std::isfinite(face->carrier_velocity())) {
      return cache_detail::Corrupt();
    }
    const auto owner = Owner(state, face->index());
    if (owner == state->blocks()->size()) {
      return cache_detail::Corrupt();
    }
    const auto* block = state->blocks()->Get(static_cast<flatbuffers::uoffset_t>(owner))->index();
    const auto* index = face->index();
    const auto local_x = static_cast<std::size_t>(index->x() - block->x() * 4);
    const auto local_y = static_cast<std::size_t>(index->y() - block->y() * 4);
    const auto local_z = static_cast<std::size_t>(index->z() - block->z() * 4);
    const auto slot = owner * 192 + static_cast<std::size_t>(face->axis()) * 64 + local_z * 16 + local_y * 4 + local_x;
    if (face->slot() != slot || (ordinal > 0 && state->faces()->Get(ordinal - 1)->slot() >= slot)) {
      return cache_detail::Corrupt();
    }
  }
  for (flatbuffers::uoffset_t ordinal = 0; ordinal < state->sources()->size(); ++ordinal) {
    const auto* source = state->sources()->Get(ordinal);
    if (!cache_detail::Nonnegative(source->fractional_remainder()) || source->fractional_remainder() >= 1
        || (ordinal > 0 && state->sources()->Get(ordinal - 1)->ordinal() >= source->ordinal())) {
      return cache_detail::Corrupt();
    }
  }
  if (const auto* cavity = state->cavity()) {
    if (cavity->next_id() == 0 || cavity->inventories()->size() > cavity->gas()->size()) {
      return cache_detail::Corrupt();
    }
    flatbuffers::uoffset_t next_component = 0;
    for (flatbuffers::uoffset_t index = 0; index < cavity->gas()->size(); ++index) {
      const auto* gas = cavity->gas()->Get(index);
      const auto label = gas->component();
      if (!cache_detail::Positive(gas->gas_volume()) || label >= cavity->inventories()->size() || label > next_component
          || (index > 0 && Key(cavity->gas()->Get(index - 1)->index()) >= Key(gas->index()))) {
        return cache_detail::Corrupt();
      }
      if (label == next_component) {
        ++next_component;
      }
    }
    if (next_component != cavity->inventories()->size()) {
      return cache_detail::Corrupt();
    }
    for (flatbuffers::uoffset_t index = 0; index < cavity->inventories()->size(); ++index) {
      const auto* inventory = cavity->inventories()->Get(index);
      if (inventory->id() == 0) {
        if (inventory->initial_volume() != 0.0 || inventory->initial_absolute_pressure() != 0.0) {
          return cache_detail::Corrupt();
        }
        continue;
      }
      if (inventory->id() >= cavity->next_id() || !cache_detail::Positive(inventory->initial_volume())
          || !cache_detail::Positive(inventory->initial_absolute_pressure())) {
        return cache_detail::Corrupt();
      }
      for (flatbuffers::uoffset_t previous = 0; previous < index; ++previous) {
        if (cavity->inventories()->Get(previous)->id() == inventory->id()) {
          return cache_detail::Corrupt();
        }
      }
    }
  }
  if (state->resume()) {
    if (state->clock()->substep() != 0) {
      return cache_detail::Corrupt();
    }
    auto resume = ReadResume(state->resume(), Counts(state));
    if (!resume.ok()) {
      return resume.status();
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<LiquidState3D> MaterializeState(const wire::LiquidState* state) {
  LiquidState3D result;
  result.geometry = {cache_detail::Unpack(state->world_origin()), state->cell_width()};
  result.clock = cache_detail::Unpack(state->clock());
  result.random_seed = state->random_seed();
  result.next_particle_id = state->next_particle_id();
  CACHE_TRY(cache_detail::Materialize(result.particles, state->particles(), [](const auto* particle) {
    return cache_detail::Unpack(particle);
  }));
  CACHE_TRY(cache_detail::Materialize(result.blocks, state->blocks(), [](const auto* block) {
    const auto* index = block->index();
    return LiquidBlockState3D{
        {index->x(), index->y(), index->z()},
        block->generation(),
        block->activation_age(),
        block->inactive_age(),
    };
  }));
  CACHE_TRY(cache_detail::Materialize(result.cells, state->cells(), [](const auto* cell) {
    const auto* index = cell->index();
    return LiquidCellState3D{
        {index->x(), index->y(), index->z()},
        static_cast<LiquidCellOccupancy3D>(cell->occupancy()),
        cell->previous_pressure(),
    };
  }));
  CACHE_TRY(cache_detail::Materialize(result.faces, state->faces(), [](const auto* face) {
    const auto* index = face->index();
    return LiquidFaceState3D{
        {static_cast<FaceNormalAxis>(face->axis()), {index->x(), index->y(), index->z()}},
        static_cast<std::size_t>(face->slot()),
        face->carrier_velocity(),
        static_cast<LiquidFaceValidity3D>(face->validity()),
        face->side(),
        face->structurally_eligible(),
    };
  }));
  CACHE_TRY(cache_detail::Materialize(result.sources, state->sources(), [](const auto* source) {
    return LiquidSourceState3D{source->ordinal(), source->next_child(), source->fractional_remainder()};
  }));
  if (const auto* cavity = state->cavity()) {
    result.next_cavity_id = cavity->next_id();
    CACHE_TRY(cache_detail::Materialize(result.cavity_gas, cavity->gas(), [](const auto* gas) {
      const auto* index = gas->index();
      return LiquidCavityGasState3D{{index->x(), index->y(), index->z()}, gas->gas_volume(), gas->component()};
    }));
    CACHE_TRY(cache_detail::Materialize(result.cavity_components, cavity->inventories(), [](const auto* inventory) {
      return LiquidCavityComponentState3D{
          inventory->id(),
          inventory->initial_volume(),
          inventory->initial_absolute_pressure()
      };
    }));
  }
  return result;
}
absl::StatusOr<LiquidCacheBytes3D>
EncodeState(const LiquidState3D& state, const LiquidResumeMetadata3D* resume, LiquidCacheLimits3D limits) {
  CACHE_TRY(BoundCounts(Counts(state), limits));
  CACHE_TRY(ValidateLiquidState3D(state));
  if (resume && (state.clock.substep_ordinal != 0 || !ValidResume(*resume, Counts(state)))) {
    return Invalid();
  }
  const auto table_count = state.particles.size() + state.blocks.size() + state.cells.size() + state.faces.size()
      + state.sources.size() + state.cavity_gas.size() + state.cavity_components.size() + (state.next_cavity_id != 0)
      + 1;
  return cache_detail::Build<wire::LiquidState>(
      table_count,
      limits,
      wire::LiquidStateIdentifier(),
      [&](flatbuffers::FlatBufferBuilder& builder) -> absl::StatusOr<flatbuffers::Offset<wire::LiquidState>> {
        auto particles =
            cache_detail::BuildVector<wire::Particle>(builder, state.particles.size(), [&](std::size_t index) {
              return cache_detail::PackParticle(builder, state.particles[index]);
            });
        if (!particles.ok()) {
          return particles.status();
        }
        auto blocks = cache_detail::BuildVector<wire::Block>(builder, state.blocks.size(), [&](std::size_t ordinal) {
          const auto& block = state.blocks[ordinal];
          const wire::Index3 index(block.block_index.x, block.block_index.y, block.block_index.z);
          return wire::CreateBlock(builder, &index, block.generation, block.activation_age, block.inactive_age);
        });
        if (!blocks.ok()) {
          return blocks.status();
        }
        auto cells = cache_detail::BuildVector<wire::Cell>(builder, state.cells.size(), [&](std::size_t ordinal) {
          const auto& cell = state.cells[ordinal];
          const wire::Index3 index(cell.cell_index.x, cell.cell_index.y, cell.cell_index.z);
          return wire::CreateCell(
              builder,
              &index,
              static_cast<wire::Occupancy>(cell.occupancy),
              cell.previous_pressure
          );
        });
        if (!cells.ok()) {
          return cells.status();
        }
        auto faces = cache_detail::BuildVector<wire::Face>(builder, state.faces.size(), [&](std::size_t ordinal) {
          const auto& face = state.faces[ordinal];
          const auto lattice = face.face_index.lattice_index;
          const wire::Index3 index(lattice.x, lattice.y, lattice.z);
          return wire::CreateFace(
              builder,
              &index,
              static_cast<wire::Axis>(face.face_index.axis),
              face.face_slot,
              face.carrier_velocity,
              static_cast<wire::Validity>(face.validity),
              face.side_tag,
              face.structurally_eligible
          );
        });
        if (!faces.ok()) {
          return faces.status();
        }
        auto sources =
            cache_detail::BuildVector<wire::SourceState>(builder, state.sources.size(), [&](std::size_t index) {
              const auto& source = state.sources[index];
              return wire::CreateSourceState(
                  builder,
                  source.source_ordinal,
                  source.next_child_ordinal,
                  source.fractional_remainder
              );
            });
        if (!sources.ok()) {
          return sources.status();
        }
        flatbuffers::Offset<wire::ResumeMetadata> resume_metadata_offset;
        if (resume) {
          const auto packed_resolved_fingerprint = cache_detail::Pack(resume->resolved_fingerprint);
          const auto packed_execution_id = cache_detail::Pack(resume->execution_id);
          const auto packed_reserved_capacities = PackCapacity(resume->capacities);
          const auto packed_replay_reservations = PackCapacity(resume->capacity_replay_reservations);
          resume_metadata_offset = wire::CreateResumeMetadata(
              builder,
              &packed_resolved_fingerprint,
              &packed_execution_id,
              &packed_reserved_capacities,
              &packed_replay_reservations
          );
        }
        flatbuffers::Offset<wire::CavitySnapshot> cavity;
        if (state.next_cavity_id != 0) {
          const auto gas = cache_detail::BuildVector<wire::CavityGasCell>(
              builder,
              state.cavity_gas.size(),
              [&](std::size_t ordinal) {
                const auto& record = state.cavity_gas[ordinal];
                const wire::Index3 index(record.cell_index.x, record.cell_index.y, record.cell_index.z);
                return wire::CreateCavityGasCell(
                    builder,
                    &index,
                    record.gas_volume,
                    static_cast<std::uint32_t>(record.component)
                );
              }
          );
          if (!gas.ok()) {
            return gas.status();
          }
          const auto inventories = cache_detail::BuildVector<wire::CavityInventory>(
              builder,
              state.cavity_components.size(),
              [&](std::size_t ordinal) {
                const auto& record = state.cavity_components[ordinal];
                return wire::CreateCavityInventory(
                    builder,
                    record.id,
                    cache_detail::Canonical(record.initial_volume),
                    cache_detail::Canonical(record.initial_absolute_pressure)
                );
              }
          );
          if (!inventories.ok()) {
            return inventories.status();
          }
          cavity = wire::CreateCavitySnapshot(builder, state.next_cavity_id, *gas, *inventories);
        }
        const auto origin = cache_detail::Pack(state.geometry.world_origin);
        const auto clock = cache_detail::Pack(state.clock);
        return wire::CreateLiquidState(
            builder,
            1,
            0,
            &origin,
            state.geometry.cell_width,
            &clock,
            state.random_seed,
            state.next_particle_id,
            *particles,
            *blocks,
            *cells,
            *faces,
            *sources,
            resume_metadata_offset,
            cavity
        );
      }
  );
}
bool ValidPoint(const LiquidPoint3D& point) {
  return point.id != 0 && cache_detail::Finite(point.world_position) && cache_detail::Finite(point.world_velocity)
      && cache_detail::Positive(point.radius) && cache_detail::Positive(point.volume_ratio) && point.level == 0;
}
LiquidPoint3D ReadPoint(const wire::Point* point) {
  return {
      point->id(),
      point->tag(),
      cache_detail::Unpack(point->position()),
      cache_detail::Unpack(point->velocity()),
      point->radius(),
      point->volume_ratio(),
      point->level(),
  };
}
absl::Status ValidManifest(const LiquidFrameManifest3D& manifest, LiquidCacheLimits3D limits) {
  const bool execution_matches = manifest.execution_id == LiquidExecutionId3D(manifest.resolved_fingerprint);
  if (!execution_matches || !cache_detail::ValidClock(manifest.clock) || manifest.clock.substep_ordinal != 0
      || manifest.points.kind != LiquidContentKind3D::kPoints || manifest.points.byte_length < 8) {
    return Invalid();
  }
  if (manifest.points.byte_length > limits.maximum_bytes
      || manifest.points.byte_length >= FLATBUFFERS_MAX_BUFFER_SIZE) {
    return cache_detail::Error(absl::StatusCode::kResourceExhausted);
  }
  return absl::OkStatus();
}
}  // namespace

LiquidDigest3D LiquidContentDigest3D(std::span<const std::uint8_t> bytes) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, bytes.data(), bytes.size());
  LiquidDigest3D digest;
  blake3_hasher_finalize(&hasher, digest.data(), digest.size());
  return digest;
}
LiquidDigest3D cache_detail::DomainDigest(std::string_view domain, std::span<const std::uint8_t> bytes) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, domain.data(), domain.size());
  constexpr char separator = 0;
  blake3_hasher_update(&hasher, &separator, 1);
  blake3_hasher_update(&hasher, bytes.data(), bytes.size());
  LiquidDigest3D digest;
  blake3_hasher_finalize(&hasher, digest.data(), digest.size());
  return digest;
}
LiquidDigest3D LiquidExecutionId3D(const LiquidDigest3D& resolved_fingerprint) {
  return cache_detail::DomainDigest("fluidity/simulate/execution/v1", resolved_fingerprint);
}
std::array<char, 65> LiquidDigestHex3D(const LiquidDigest3D& digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::array<char, 65> result{};
  for (std::size_t index = 0; index < digest.size(); ++index) {
    result[2 * index] = digits[digest[index] >> 4];
    result[2 * index + 1] = digits[digest[index] & 15];
  }
  return result;
}
absl::StatusOr<std::array<char, 76>> LiquidContentName3D(const LiquidContentReference3D& reference) {
  constexpr std::array<std::string_view, 4> extensions{".flst", ".flpt", ".flrs", ".flmf"};
  const auto kind = static_cast<std::size_t>(reference.kind);
  if (kind >= extensions.size()) {
    return Invalid();
  }
  std::array<char, 76> result{};
  std::memcpy(result.data(), "blobs/", 6);
  const auto hex = LiquidDigestHex3D(reference.digest);
  std::memcpy(result.data() + 6, hex.data(), 64);
  std::memcpy(result.data() + 70, extensions[kind].data(), 5);
  return result;
}
absl::Status ValidateLiquidContentName3D(std::string_view name, const LiquidContentReference3D& reference) {
  const auto expected = LiquidContentName3D(reference);
  if (!expected.ok()) {
    return expected.status();
  }
  if (name != std::string_view(expected->data(), 75)) {
    return Invalid();
  }
  return absl::OkStatus();
}
absl::Status VerifyLiquidContent3D(
    std::span<const std::uint8_t> bytes,
    const LiquidContentReference3D& reference,
    LiquidContentKind3D expected_kind,
    LiquidCacheLimits3D limits
) {
  if (static_cast<std::uint8_t>(expected_kind) > 3 || reference.kind != expected_kind) {
    return Invalid();
  }
  if (bytes.size() > limits.maximum_bytes || bytes.size() >= FLATBUFFERS_MAX_BUFFER_SIZE) {
    return cache_detail::Error(absl::StatusCode::kResourceExhausted);
  }
  if (reference.byte_length != bytes.size() || reference.digest != LiquidContentDigest3D(bytes)) {
    return cache_detail::Corrupt();
  }
  return absl::OkStatus();
}
absl::StatusOr<LiquidCacheBytes3D> EncodeLiquidState3D(const LiquidState3D& state, LiquidCacheLimits3D limits) {
  return EncodeState(state, nullptr, limits);
}
absl::StatusOr<LiquidState3D> DecodeLiquidState3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  auto state = cache_detail::Verify<wire::LiquidState>(bytes, wire::LiquidStateIdentifier(), limits);
  if (!state.ok()) {
    return state.status();
  }
  CACHE_TRY(ValidateWireState(*state, limits));
  return MaterializeState(*state);
}
absl::StatusOr<LiquidCacheBytes3D>
EncodeLiquidCheckpoint3D(const LiquidState3D& state, const LiquidResumeMetadata3D& resume, LiquidCacheLimits3D limits) {
  return EncodeState(state, &resume, limits);
}
absl::StatusOr<LiquidCheckpoint3D>
DecodeLiquidCheckpoint3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  auto state = cache_detail::Verify<wire::LiquidState>(bytes, wire::LiquidStateIdentifier(), limits);
  if (!state.ok()) {
    return state.status();
  }
  CACHE_TRY(ValidateWireState(*state, limits));
  auto metadata = ReadResume((*state)->resume(), Counts(*state));
  if (!metadata.ok()) {
    return metadata.status();
  }
  auto decoded = MaterializeState(*state);
  if (!decoded.ok()) {
    return decoded.status();
  }
  return LiquidCheckpoint3D{std::move(*decoded), *metadata};
}
absl::StatusOr<LiquidPointFrame3D> BuildLiquidPointFrame3D(const LiquidState3D& state, LiquidCacheLimits3D limits) {
  CACHE_TRY(BoundCounts(Counts(state), limits));
  CACHE_TRY(ValidateLiquidState3D(state));
  if (state.clock.substep_ordinal != 0) {
    return Invalid();
  }
  LiquidPointFrame3D result;
  result.clock = state.clock;
  auto points = LiquidRecords3D<LiquidPoint3D>::Create(state.particles.size());
  if (!points.ok()) {
    return points.status();
  }
  CACHE_TRY(points->Resize(state.particles.size()));
  for (std::size_t i = 0; i < state.particles.size(); ++i) {
    const auto& particle = state.particles[i];
    // Preserve the original producer's operation order for cache-byte identity.
    const double radius = std::cbrt(3 * particle.reference_volume * particle.volume_ratio / (4 * std::acos(-1.0)));
    if (!std::isfinite(radius) || radius <= 0) {
      return cache_detail::Error(absl::StatusCode::kOutOfRange);
    }
    (*points)[i] = {
        .id = particle.id,
        .tag = particle.tag,
        .world_position = particle.world_position,
        .world_velocity = particle.world_velocity,
        .radius = radius,
        .volume_ratio = particle.volume_ratio,
        .level = 0,
    };
  }
  result.points = std::move(*points);
  return result;
}

absl::StatusOr<LiquidCacheBytes3D>
EncodeLiquidPointFrame3D(const LiquidPointFrame3D& frame, LiquidCacheLimits3D limits) {
  CACHE_TRY(BoundCounts({frame.points.size(), 0, 0, 0, 0}, limits));
  if (!cache_detail::ValidClock(frame.clock) || frame.clock.substep_ordinal != 0) {
    return Invalid();
  }
  for (std::size_t index = 0; index < frame.points.size(); ++index) {
    if (!ValidPoint(frame.points[index])) {
      return Invalid();
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (frame.points[previous].id == frame.points[index].id) {
        return Invalid();
      }
    }
  }
  return cache_detail::Build<wire::PointFrame>(
      frame.points.size(),
      limits,
      wire::PointFrameIdentifier(),
      [&](flatbuffers::FlatBufferBuilder& builder) -> absl::StatusOr<flatbuffers::Offset<wire::PointFrame>> {
        auto points = cache_detail::BuildVector<wire::Point>(builder, frame.points.size(), [&](std::size_t index) {
          const auto& point = frame.points[index];
          const auto position = cache_detail::Pack(point.world_position);
          const auto velocity = cache_detail::Pack(point.world_velocity);
          return wire::CreatePoint(
              builder,
              point.id,
              point.tag,
              &position,
              &velocity,
              point.radius,
              point.volume_ratio,
              point.level
          );
        });
        if (!points.ok()) {
          return points.status();
        }
        const auto clock = cache_detail::Pack(frame.clock);
        return wire::CreatePointFrame(builder, 1, 0, &clock, *points);
      }
  );
}
absl::StatusOr<LiquidPointFrame3D>
DecodeLiquidPointFrame3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  auto verified = cache_detail::Verify<wire::PointFrame>(bytes, wire::PointFrameIdentifier(), limits);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* frame = *verified;
  CACHE_TRY(BoundCounts({frame->points()->size(), 0, 0, 0, 0}, limits));
  const auto clock = cache_detail::Unpack(frame->clock());
  if (!cache_detail::ValidClock(clock) || clock.substep_ordinal != 0) {
    return cache_detail::Corrupt();
  }
  for (flatbuffers::uoffset_t index = 0; index < frame->points()->size(); ++index) {
    const auto point = ReadPoint(frame->points()->Get(index));
    if (!ValidPoint(point)) {
      return cache_detail::Corrupt();
    }
    for (flatbuffers::uoffset_t previous = 0; previous < index; ++previous) {
      if (frame->points()->Get(previous)->id() == point.id) {
        return cache_detail::Corrupt();
      }
    }
  }
  LiquidPointFrame3D result;
  result.clock = clock;
  CACHE_TRY(cache_detail::Materialize(result.points, frame->points(), ReadPoint));
  return result;
}
absl::StatusOr<LiquidCacheBytes3D>
EncodeLiquidFrameManifest3D(const LiquidFrameManifest3D& manifest, LiquidCacheLimits3D limits) {
  CACHE_TRY(ValidManifest(manifest, limits));
  return cache_detail::Build<wire::FrameManifest>(
      1,
      limits,
      wire::FrameManifestIdentifier(),
      [&](flatbuffers::FlatBufferBuilder& builder) -> absl::StatusOr<flatbuffers::Offset<wire::FrameManifest>> {
        const auto name = LiquidContentName3D(manifest.points);
        if (!name.ok()) {
          return name.status();
        }
        const auto relative_name = builder.CreateString(name->data(), 75);
        const auto digest = cache_detail::Pack(manifest.points.digest);
        const auto points = wire::CreateContentReference(
            builder,
            wire::ContentKind::Points,
            manifest.points.byte_length,
            &digest,
            relative_name
        );
        const auto clock = cache_detail::Pack(manifest.clock);
        const auto fingerprint = cache_detail::Pack(manifest.resolved_fingerprint);
        const auto execution = cache_detail::Pack(manifest.execution_id);
        return wire::CreateFrameManifest(
            builder,
            1,
            0,
            &clock,
            wire::FrameStatus::Complete,
            &fingerprint,
            &execution,
            points
        );
      }
  );
}
absl::StatusOr<LiquidFrameManifest3D>
DecodeLiquidFrameManifest3D(std::span<const std::uint8_t> bytes, LiquidCacheLimits3D limits) {
  auto verified = cache_detail::Verify<wire::FrameManifest>(bytes, wire::FrameManifestIdentifier(), limits);
  if (!verified.ok()) {
    return verified.status();
  }
  const auto* frame = *verified;
  if (frame->status() != wire::FrameStatus::Complete || frame->points()->kind() != wire::ContentKind::Points) {
    return cache_detail::Corrupt();
  }
  LiquidFrameManifest3D result{
      cache_detail::Unpack(frame->clock()),
      cache_detail::Unpack(frame->resolved_fingerprint()),
      cache_detail::Unpack(frame->execution_id()),
      {LiquidContentKind3D::kPoints, frame->points()->byte_length(), cache_detail::Unpack(frame->points()->digest())},
  };
  const auto* name = frame->points()->relative_name();
  if (!ValidateLiquidContentName3D({name->c_str(), name->size()}, result.points).ok()) {
    return cache_detail::Corrupt();
  }
  const auto status = ValidManifest(result, limits);
  if (status.code() == absl::StatusCode::kResourceExhausted) {
    return status;
  }
  if (!status.ok()) {
    return cache_detail::Corrupt();
  }
  return result;
}
#undef CACHE_TRY
}  // namespace fluidity
