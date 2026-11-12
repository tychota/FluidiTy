#include "fluidity/host/liquid_state_3d.h"

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace fluidity {
namespace {
// Device staging and host acceptance use this clock arithmetic. Only the final interval may snap
// to the frame boundary, and only when the sum is one representable step away.
absl::StatusOr<LiquidClock3D>
ComputeAcceptedClock(LiquidClock3D committed_clock, double target_frame_time, double attempted_duration) {
  if (!std::isfinite(attempted_duration) || attempted_duration <= 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const double remaining_frame_duration = target_frame_time - committed_clock.time;
  double accepted_time = committed_clock.time + attempted_duration;
  if (attempted_duration > remaining_frame_duration || !std::isfinite(accepted_time)
      || accepted_time <= committed_clock.time) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  if (attempted_duration == remaining_frame_duration && accepted_time != target_frame_time) {
    if (std::nextafter(accepted_time, target_frame_time) != target_frame_time) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    accepted_time = target_frame_time;
  }
  if (accepted_time > target_frame_time || committed_clock.epoch == UINT64_MAX
      || committed_clock.accepted_step_ordinal == UINT64_MAX || committed_clock.substep_ordinal == UINT64_MAX
      || (accepted_time == target_frame_time && committed_clock.frame_ordinal == UINT64_MAX)) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  committed_clock.time = accepted_time;
  ++committed_clock.epoch;
  ++committed_clock.accepted_step_ordinal;
  if (accepted_time == target_frame_time) {
    ++committed_clock.frame_ordinal;
    committed_clock.substep_ordinal = 0;
  } else {
    ++committed_clock.substep_ordinal;
  }
  return committed_clock;
}
auto Components(Vec3 value) {
  return std::tuple(value.x, value.y, value.z);
}
auto Components(Mat3 value) {
  return std::tuple(Components(value.x_row), Components(value.y_row), Components(value.z_row));
}
template <typename Index>
auto LatticeSortKey(Index index) {
  return std::tuple(index.z, index.y, index.x);
}
bool SameGeometry(UniformMacGeometry3D left, UniformMacGeometry3D right) {
  return Components(left.world_origin) == Components(right.world_origin) && left.cell_width == right.cell_width;
}
bool Finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
bool Positive(double value) {
  return std::isfinite(value) && value > 0;
}
// C++ division truncates toward zero; signed block ownership requires floor.
std::int32_t OwnerBlockCoordinate(std::int32_t coordinate) {
  return coordinate / 4 - (coordinate % 4 < 0 ? 1 : 0);
}
// Returns the active block ordinal, or the block count when the owner is absent.
template <typename Index>
std::size_t FindOwnerBlockOrdinal(const LiquidState3D& state, Index index) {
  const BlockIndex3D owner_block_index{
      OwnerBlockCoordinate(index.x),
      OwnerBlockCoordinate(index.y),
      OwnerBlockCoordinate(index.z)
  };
  const auto blocks = state.blocks.values();
  const auto found = std::lower_bound(blocks.begin(), blocks.end(), owner_block_index, [](const auto& block, auto key) {
    return LatticeSortKey(block.block_index) < LatticeSortKey(key);
  });
  if (found == blocks.end() || LatticeSortKey(found->block_index) != LatticeSortKey(owner_block_index)) {
    return blocks.size();
  }
  return static_cast<std::size_t>(found - blocks.begin());
}
bool ActiveRecordsFit(const LiquidState3D& state, LiquidStateCapacities3D capacity) {
  return state.particles.size() <= capacity.particles && state.blocks.size() <= capacity.blocks
      && state.cells.size() <= capacity.cells && state.faces.size() <= capacity.faces
      && state.sources.size() <= capacity.sources && state.cavity_gas.size() <= capacity.cells
      && state.cavity_components.size() <= capacity.cells;
}
bool HasReservedCapacities(const LiquidState3D& state, LiquidStateCapacities3D capacity) {
  const std::size_t cavity_capacity = state.next_cavity_id == 0 ? 0 : capacity.cells;
  return state.cavity_gas.capacity() == cavity_capacity && state.cavity_components.capacity() == cavity_capacity
      && state.particles.capacity() == capacity.particles && state.blocks.capacity() == capacity.blocks
      && state.cells.capacity() == capacity.cells && state.faces.capacity() == capacity.faces
      && state.sources.capacity() == capacity.sources;
}
// All destination extents were checked when reserving the version. Trivial record
// assignment neither allocates nor changes storage ownership.
template <typename Record>
void CopyRecords(const LiquidRecords3D<Record>& source, LiquidRecords3D<Record>& destination) {
  const auto status = destination.Resize(source.size());
  (void)status;
  std::ranges::copy(source.values(), destination.values().begin());
}
void CopyState(const LiquidState3D& source, LiquidState3D& destination) {
  destination.geometry = source.geometry;
  destination.clock = source.clock;
  destination.random_seed = source.random_seed;
  destination.next_particle_id = source.next_particle_id;
  CopyRecords(source.particles, destination.particles);
  CopyRecords(source.blocks, destination.blocks);
  CopyRecords(source.cells, destination.cells);
  CopyRecords(source.faces, destination.faces);
  CopyRecords(source.sources, destination.sources);
  destination.next_cavity_id = source.next_cavity_id;
  CopyRecords(source.cavity_gas, destination.cavity_gas);
  CopyRecords(source.cavity_components, destination.cavity_components);
}
template <typename Record>
absl::Status AllocateRecords(LiquidRecords3D<Record>& records, std::size_t capacity) {
  auto allocation = LiquidRecords3D<Record>::Create(capacity);
  if (!allocation.ok()) {
    return allocation.status();
  }
  records = std::move(*allocation);
  return absl::OkStatus();
}
// Reject impossible aggregate byte counts before making any allocation. The three
// independent transaction versions all need to fit in the addressable byte range.
absl::Status CheckCapacities(LiquidStateCapacities3D capacity, bool cavity_enabled) {
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 3;
  std::size_t bytes = 0;
  const auto regions = std::to_array<std::pair<std::size_t, std::size_t>>({
      {capacity.particles, sizeof(LiquidParticleState3D)},
      {capacity.blocks, sizeof(LiquidBlockState3D)},
      {capacity.cells, sizeof(LiquidCellState3D)},
      {capacity.faces, sizeof(LiquidFaceState3D)},
      {capacity.sources, sizeof(LiquidSourceState3D)},
      {cavity_enabled ? capacity.cells : 0, sizeof(LiquidCavityGasState3D) + sizeof(LiquidCavityComponentState3D)},
  });
  for (const auto& [count, stride] : regions) {
    if (count > (limit - bytes) / stride) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    bytes += count * stride;
  }
  return absl::OkStatus();
}
}  // namespace

bool LiquidParticleState3D::operator==(const LiquidParticleState3D& other) const {
  return std::tie(id, tag, affine_enabled, mass, reference_volume, volume_ratio, side_tag, pic_initialized)
      == std::tie(
             other.id,
             other.tag,
             other.affine_enabled,
             other.mass,
             other.reference_volume,
             other.volume_ratio,
             other.side_tag,
             other.pic_initialized
      )
      && Components(world_position) == Components(other.world_position)
      && Components(world_velocity) == Components(other.world_velocity)
      && Components(velocity_slopes) == Components(other.velocity_slopes);
}
bool LiquidBlockState3D::operator==(const LiquidBlockState3D& other) const {
  return LatticeSortKey(block_index) == LatticeSortKey(other.block_index)
      && std::tie(generation, activation_age, inactive_age)
      == std::tie(other.generation, other.activation_age, other.inactive_age);
}
bool LiquidCellState3D::operator==(const LiquidCellState3D& other) const {
  return LatticeSortKey(cell_index) == LatticeSortKey(other.cell_index) && occupancy == other.occupancy
      && previous_pressure == other.previous_pressure;
}
bool LiquidFaceState3D::operator==(const LiquidFaceState3D& other) const {
  return face_index.axis == other.face_index.axis
      && LatticeSortKey(face_index.lattice_index) == LatticeSortKey(other.face_index.lattice_index)
      && std::tie(face_slot, carrier_velocity, validity, side_tag, structurally_eligible)
      == std::tie(other.face_slot, other.carrier_velocity, other.validity, other.side_tag, other.structurally_eligible);
}
bool LiquidCavityGasState3D::operator==(const LiquidCavityGasState3D& other) const {
  return LatticeSortKey(cell_index) == LatticeSortKey(other.cell_index) && gas_volume == other.gas_volume
      && component == other.component;
}

bool LiquidState3D::operator==(const LiquidState3D& other) const {
  return SameGeometry(geometry, other.geometry) && clock == other.clock && random_seed == other.random_seed
      && next_particle_id == other.next_particle_id && particles == other.particles && blocks == other.blocks
      && cells == other.cells && faces == other.faces && sources == other.sources
      && next_cavity_id == other.next_cavity_id && cavity_gas == other.cavity_gas
      && cavity_components == other.cavity_components;
}

absl::Status ValidateLiquidState3D(const LiquidState3D& state) {
  if (!Finite(state.geometry.world_origin) || !Positive(state.geometry.cell_width)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto& clock = state.clock;
  if (!std::isfinite(clock.time) || clock.time < 0 || clock.substep_ordinal > clock.accepted_step_ordinal
      || clock.frame_ordinal > clock.accepted_step_ordinal || clock.epoch < clock.accepted_step_ordinal
      || clock.substep_ordinal > clock.accepted_step_ordinal - clock.frame_ordinal) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (state.next_particle_id == 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (state.next_cavity_id == 0 && (state.cavity_gas.size() != 0 || state.cavity_components.size() != 0)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  std::size_t next_component = 0;
  for (std::size_t index = 0; index < state.cavity_gas.size(); ++index) {
    const auto& gas = state.cavity_gas[index];
    if (!Positive(gas.gas_volume) || gas.component >= state.cavity_components.size() || gas.component > next_component
        || (index > 0 && LatticeSortKey(state.cavity_gas[index - 1].cell_index) >= LatticeSortKey(gas.cell_index))) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (gas.component == next_component) {
      ++next_component;
    }
  }
  if (next_component != state.cavity_components.size()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  for (std::size_t index = 0; index < state.cavity_components.size(); ++index) {
    const auto& component = state.cavity_components[index];
    if (component.id == 0) {
      if (component.initial_volume != 0.0 || component.initial_absolute_pressure != 0.0) {
        return absl::Status(absl::StatusCode::kInvalidArgument, {});
      }
      continue;
    }
    if (component.id >= state.next_cavity_id || !Positive(component.initial_volume)
        || !Positive(component.initial_absolute_pressure)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (state.cavity_components[previous].id == component.id) {
        return absl::Status(absl::StatusCode::kInvalidArgument, {});
      }
    }
  }
  for (std::size_t index = 0; index < state.particles.size(); ++index) {
    const auto& particle = state.particles[index];
    if (particle.id == 0 || particle.id >= state.next_particle_id) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (!Finite(particle.world_position) || !Finite(particle.world_velocity) || !Finite(particle.velocity_slopes.x_row)
        || !Finite(particle.velocity_slopes.y_row) || !Finite(particle.velocity_slopes.z_row)
        || !Positive(particle.mass) || !Positive(particle.reference_volume) || !Positive(particle.volume_ratio)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    // Stable spatial order is not ID order. Check uniqueness without sorting or
    // mutating the physical array, or allocating during acceptance.
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (state.particles[previous].id == particle.id) {
        return absl::Status(absl::StatusCode::kInvalidArgument, {});
      }
    }
  }
  if (state.blocks.size() > std::numeric_limits<std::size_t>::max() / 192) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  for (std::size_t index = 0; index < state.blocks.size(); ++index) {
    const auto block = state.blocks[index].block_index;
    for (auto coordinate : {block.x, block.y, block.z}) {
      if (coordinate < std::numeric_limits<std::int32_t>::min() / 4
          || coordinate > (std::numeric_limits<std::int32_t>::max() - 3) / 4) {
        return absl::Status(absl::StatusCode::kOutOfRange, {});
      }
    }
    if (index > 0 && LatticeSortKey(state.blocks[index - 1].block_index) >= LatticeSortKey(block)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  for (std::size_t index = 0; index < state.cells.size(); ++index) {
    const auto& cell = state.cells[index];
    if (index > 0 && LatticeSortKey(state.cells[index - 1].cell_index) >= LatticeSortKey(cell.cell_index)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (FindOwnerBlockOrdinal(state, cell.cell_index) == state.blocks.size()) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (cell.occupancy > LiquidCellOccupancy3D::kSolid || !std::isfinite(cell.previous_pressure)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  for (std::size_t index = 0; index < state.faces.size(); ++index) {
    const auto& face = state.faces[index];
    if (face.face_index.axis > FaceNormalAxis::kZ || face.validity > LiquidFaceValidity3D::kLowMass
        || !std::isfinite(face.carrier_velocity)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const auto lattice_index = face.face_index.lattice_index;
    const auto owner_block_ordinal = FindOwnerBlockOrdinal(state, lattice_index);
    if (owner_block_ordinal == state.blocks.size()) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const auto owner_block_index = state.blocks[owner_block_ordinal].block_index;
    const auto local_x = static_cast<std::size_t>(lattice_index.x - owner_block_index.x * 4);
    const auto local_y = static_cast<std::size_t>(lattice_index.y - owner_block_index.y * 4);
    const auto local_z = static_cast<std::size_t>(lattice_index.z - owner_block_index.z * 4);
    const auto expected_face_slot = owner_block_ordinal * 192 + static_cast<std::size_t>(face.face_index.axis) * 64
        + local_z * 16 + local_y * 4 + local_x;
    if (face.face_slot != expected_face_slot || (index > 0 && state.faces[index - 1].face_slot >= expected_face_slot)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  for (std::size_t index = 0; index < state.sources.size(); ++index) {
    const auto& source = state.sources[index];
    if (!std::isfinite(source.fractional_remainder) || source.fractional_remainder < 0
        || source.fractional_remainder >= 1
        || (index > 0 && state.sources[index - 1].source_ordinal >= source.source_ordinal)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  return absl::OkStatus();
}

static absl::StatusOr<LiquidState3D>
CloneState(const LiquidState3D& source, LiquidStateCapacities3D capacities, bool enable_cavity) {
  const bool cavity_enabled = enable_cavity || source.next_cavity_id != 0;
  if (auto status = ValidateLiquidState3D(source); !status.ok()) {
    return status;
  }
  if (auto status = CheckCapacities(capacities, cavity_enabled); !status.ok()) {
    return status;
  }
  if (!ActiveRecordsFit(source, capacities)) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  LiquidState3D result;
  if (auto status = AllocateRecords(result.particles, capacities.particles); !status.ok()) {
    return status;
  }
  if (auto status = AllocateRecords(result.blocks, capacities.blocks); !status.ok()) {
    return status;
  }
  if (auto status = AllocateRecords(result.cells, capacities.cells); !status.ok()) {
    return status;
  }
  if (auto status = AllocateRecords(result.faces, capacities.faces); !status.ok()) {
    return status;
  }
  if (auto status = AllocateRecords(result.sources, capacities.sources); !status.ok()) {
    return status;
  }
  if (cavity_enabled) {
    if (auto status = AllocateRecords(result.cavity_gas, capacities.cells); !status.ok()) {
      return status;
    }
    if (auto status = AllocateRecords(result.cavity_components, capacities.cells); !status.ok()) {
      return status;
    }
  }
  CopyState(source, result);
  if (cavity_enabled && result.next_cavity_id == 0) {
    result.next_cavity_id = 1;
  }
  return result;
}

absl::StatusOr<LiquidState3D> CloneLiquidState3D(const LiquidState3D& source, LiquidStateCapacities3D capacities) {
  return CloneState(source, capacities, false);
}

absl::StatusOr<LiquidFrameTransaction3D> LiquidFrameTransaction3D::Create(
    const LiquidState3D& initial_state,
    LiquidStateCapacities3D capacities,
    bool enable_cavity
) {
  auto committed = CloneState(initial_state, capacities, enable_cavity);
  if (!committed.ok()) {
    return committed.status();
  }
  auto trial = CloneState(initial_state, capacities, enable_cavity);
  if (!trial.ok()) {
    return trial.status();
  }
  auto frame_start = CloneState(initial_state, capacities, enable_cavity);
  if (!frame_start.ok()) {
    return frame_start.status();
  }
  LiquidFrameTransaction3D result;
  result.committed_ = std::move(*committed);
  result.trial_ = std::move(*trial);
  result.frame_start_ = std::move(*frame_start);
  result.capacities_ = capacities;
  return result;
}
absl::Status LiquidFrameTransaction3D::BeginFrame(double target_frame_time) {
  if (frame_active_ || committed_.clock.substep_ordinal != 0) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  if (!std::isfinite(target_frame_time) || target_frame_time <= committed_.clock.time) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (committed_.clock.frame_ordinal == std::numeric_limits<std::uint64_t>::max()) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  CopyState(committed_, frame_start_);
  target_frame_time_ = target_frame_time;
  frame_active_ = true;
  capacity_restarted_ = false;
  fatal_ = false;
  retry_count_ = 0;
  retry_duration_.reset();
  return absl::OkStatus();
}
absl::StatusOr<LiquidState3D*> LiquidFrameTransaction3D::ResetTrialFromCommitted() {
  if (!frame_active_ || trial_active_ || capacity_pending_ || fatal_ || committed_.clock.time == target_frame_time_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  if (!HasReservedCapacities(trial_, capacities_)) {
    fatal_ = true;
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  CopyState(committed_, trial_);
  trial_active_ = true;
  return &trial_;
}
absl::StatusOr<LiquidClock3D> LiquidFrameTransaction3D::PreviewAcceptedClock(double attempted_duration) const {
  if (!trial_active_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  if (retry_duration_ && attempted_duration != *retry_duration_) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  return ComputeAcceptedClock(committed_.clock, target_frame_time_, attempted_duration);
}
absl::StatusOr<LiquidAttemptOutcome3D>
LiquidFrameTransaction3D::FinishAttempt(LiquidAttemptDisposition3D decision, double attempted_duration) {
  if (!trial_active_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  trial_active_ = false;
  // Any malformed request is terminal, but never changes a committed value.
  fatal_ = true;
  if (!Positive(attempted_duration)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (retry_duration_ && attempted_duration != *retry_duration_) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const double remaining = target_frame_time_ - committed_.clock.time;
  double accepted_time = committed_.clock.time + attempted_duration;
  if (attempted_duration > remaining || !std::isfinite(accepted_time) || accepted_time <= committed_.clock.time) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  // Only the scheduled final remaining interval may snap. The predecessor/successor
  // check makes the roundoff allowance explicit, rather than hiding a residual step.
  if (attempted_duration == remaining && accepted_time != target_frame_time_) {
    if (std::nextafter(accepted_time, target_frame_time_) != target_frame_time_) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    accepted_time = target_frame_time_;
  }
  if (accepted_time > target_frame_time_) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  switch (decision) {
    case LiquidAttemptDisposition3D::kFatal:
      return LiquidAttemptOutcome3D{decision, std::nullopt};
    case LiquidAttemptDisposition3D::kCapacity:
      capacity_pending_ = true;
      fatal_ = false;
      return LiquidAttemptOutcome3D{decision, std::nullopt};
    case LiquidAttemptDisposition3D::kRetryDt: {
      const double retry_duration = attempted_duration * 0.5;
      if (retry_count_ == 2 || !Positive(retry_duration)
          || committed_.clock.time + retry_duration <= committed_.clock.time) {
        return LiquidAttemptOutcome3D{LiquidAttemptDisposition3D::kFatal, std::nullopt};
      }
      ++retry_count_;
      retry_duration_ = retry_duration;
      fatal_ = false;
      return LiquidAttemptOutcome3D{decision, retry_duration};
    }
    case LiquidAttemptDisposition3D::kAccepted:
      break;
    default:
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (trial_.clock != committed_.clock || !SameGeometry(trial_.geometry, committed_.geometry)
      || trial_.random_seed != committed_.random_seed || trial_.next_particle_id < committed_.next_particle_id
      || trial_.next_cavity_id < committed_.next_cavity_id
      || (trial_.next_cavity_id == 0) != (committed_.next_cavity_id == 0)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (!HasReservedCapacities(trial_, capacities_) || !ActiveRecordsFit(trial_, capacities_)) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (committed_.clock.epoch == maximum || committed_.clock.accepted_step_ordinal == maximum
      || committed_.clock.substep_ordinal == maximum) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  if (auto status = ValidateLiquidState3D(trial_); !status.ok()) {
    return status;
  }
  if (trial_.sources.size() != committed_.sources.size()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  for (std::size_t index = 0; index < trial_.sources.size(); ++index) {
    if (trial_.sources[index].source_ordinal != committed_.sources[index].source_ordinal
        || trial_.sources[index].next_child_ordinal < committed_.sources[index].next_child_ordinal) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  const auto accepted_clock = ComputeAcceptedClock(committed_.clock, target_frame_time_, attempted_duration);
  if (!accepted_clock.ok()) {
    return accepted_clock.status();
  }
  trial_.clock = *accepted_clock;
  std::swap(committed_, trial_);
  retry_count_ = 0;
  retry_duration_.reset();
  fatal_ = false;
  return LiquidAttemptOutcome3D{decision, std::nullopt};
}
absl::Status LiquidFrameTransaction3D::RestartFrameWithCapacity(LiquidStateCapacities3D capacities) {
  if (!frame_active_ || trial_active_ || !capacity_pending_ || capacity_restarted_ || fatal_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  if (capacities == capacities_ || capacities.particles < capacities_.particles
      || capacities.blocks < capacities_.blocks || capacities.cells < capacities_.cells
      || capacities.faces < capacities_.faces || capacities.sources < capacities_.sources) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  // Construct every independent owner before replacing any live version. Real
  // nothrow allocation failures destroy only temporary owners and retain retry authority.
  auto replacement = Create(frame_start_, capacities);
  if (!replacement.ok()) {
    return replacement.status();
  }
  replacement->target_frame_time_ = target_frame_time_;
  replacement->frame_active_ = true;
  replacement->capacity_restarted_ = true;
  *this = std::move(*replacement);
  return absl::OkStatus();
}
absl::Status LiquidFrameTransaction3D::FinishFrame() {
  if (!frame_active_ || trial_active_ || capacity_pending_ || fatal_ || committed_.clock.time != target_frame_time_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  // The final accepted version already contains the completed-frame clock. Closing
  // scheduling must not give one epoch two different persistent byte states.
  frame_active_ = false;
  return absl::OkStatus();
}
}  // namespace fluidity
