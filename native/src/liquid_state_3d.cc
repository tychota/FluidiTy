#include "fluidity/host/liquid_state_3d.h"

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace fluidity {
namespace {
auto Components(Vec3 value) {
  return std::tuple(value.x, value.y, value.z);
}
auto Components(Mat3 value) {
  return std::tuple(Components(value.x_row), Components(value.y_row), Components(value.z_row));
}
template <typename Index>
auto Key(Index index) {
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
std::int32_t BlockCoordinate(std::int32_t coordinate) {
  return coordinate / 4 - (coordinate % 4 < 0 ? 1 : 0);
}
template <typename Index>
std::size_t FindOwner(const LiquidState3D& state, Index index) {
  const BlockIndex3D owner{BlockCoordinate(index.x), BlockCoordinate(index.y), BlockCoordinate(index.z)};
  const auto blocks = state.blocks.values();
  const auto found = std::lower_bound(blocks.begin(), blocks.end(), owner, [](const auto& block, auto key) {
    return Key(block.block_index) < Key(key);
  });
  if (found == blocks.end() || Key(found->block_index) != Key(owner)) {
    return blocks.size();
  }
  return static_cast<std::size_t>(found - blocks.begin());
}
bool Fits(const LiquidState3D& state, LiquidStateCapacities3D capacity) {
  return state.particles.size() <= capacity.particles && state.blocks.size() <= capacity.blocks
      && state.cells.size() <= capacity.cells && state.faces.size() <= capacity.faces
      && state.sources.size() <= capacity.sources;
}
bool Reserved(const LiquidState3D& state, LiquidStateCapacities3D capacity) {
  return state.particles.capacity() == capacity.particles && state.blocks.capacity() == capacity.blocks
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
absl::Status CheckCapacities(LiquidStateCapacities3D capacity) {
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 3;
  std::size_t bytes = 0;
  const auto regions = std::to_array<std::pair<std::size_t, std::size_t>>({
      {capacity.particles, sizeof(LiquidParticleState3D)},
      {capacity.blocks, sizeof(LiquidBlockState3D)},
      {capacity.cells, sizeof(LiquidCellState3D)},
      {capacity.faces, sizeof(LiquidFaceState3D)},
      {capacity.sources, sizeof(LiquidSourceState3D)},
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
  return Key(block_index) == Key(other.block_index)
      && std::tie(generation, activation_age, inactive_age)
      == std::tie(other.generation, other.activation_age, other.inactive_age);
}
bool LiquidCellState3D::operator==(const LiquidCellState3D& other) const {
  return Key(cell_index) == Key(other.cell_index) && occupancy == other.occupancy
      && previous_pressure == other.previous_pressure;
}
bool LiquidFaceState3D::operator==(const LiquidFaceState3D& other) const {
  return face_index.axis == other.face_index.axis
      && Key(face_index.lattice_index) == Key(other.face_index.lattice_index)
      && std::tie(face_slot, carrier_velocity, validity, side_tag, structurally_eligible)
      == std::tie(other.face_slot, other.carrier_velocity, other.validity, other.side_tag, other.structurally_eligible);
}
bool LiquidState3D::operator==(const LiquidState3D& other) const {
  return SameGeometry(geometry, other.geometry) && clock == other.clock && random_seed == other.random_seed
      && next_particle_id == other.next_particle_id && particles == other.particles && blocks == other.blocks
      && cells == other.cells && faces == other.faces && sources == other.sources;
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
    if (index > 0 && Key(state.blocks[index - 1].block_index) >= Key(block)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
  }
  for (std::size_t index = 0; index < state.cells.size(); ++index) {
    const auto& cell = state.cells[index];
    if (index > 0 && Key(state.cells[index - 1].cell_index) >= Key(cell.cell_index)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (FindOwner(state, cell.cell_index) == state.blocks.size()) {
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
    const auto lattice = face.face_index.lattice_index;
    const auto owner = FindOwner(state, lattice);
    if (owner == state.blocks.size()) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    const auto block = state.blocks[owner].block_index;
    const auto local_x = static_cast<std::size_t>(lattice.x - block.x * 4);
    const auto local_y = static_cast<std::size_t>(lattice.y - block.y * 4);
    const auto local_z = static_cast<std::size_t>(lattice.z - block.z * 4);
    const auto slot =
        owner * 192 + static_cast<std::size_t>(face.face_index.axis) * 64 + local_z * 16 + local_y * 4 + local_x;
    if (face.face_slot != slot || (index > 0 && state.faces[index - 1].face_slot >= slot)) {
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

absl::StatusOr<LiquidState3D> CloneLiquidState3D(const LiquidState3D& source, LiquidStateCapacities3D capacities) {
  if (auto status = ValidateLiquidState3D(source); !status.ok()) {
    return status;
  }
  if (auto status = CheckCapacities(capacities); !status.ok()) {
    return status;
  }
  if (!Fits(source, capacities)) {
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
  CopyState(source, result);
  return result;
}

absl::StatusOr<LiquidFrameTransaction3D>
LiquidFrameTransaction3D::Create(const LiquidState3D& initial_state, LiquidStateCapacities3D capacities) {
  auto committed = CloneLiquidState3D(initial_state, capacities);
  if (!committed.ok()) {
    return committed.status();
  }
  auto trial = CloneLiquidState3D(initial_state, capacities);
  if (!trial.ok()) {
    return trial.status();
  }
  auto frame_start = CloneLiquidState3D(initial_state, capacities);
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
absl::Status LiquidFrameTransaction3D::BeginFrame(double target_time) {
  if (frame_active_ || committed_.clock.substep_ordinal != 0) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  if (!std::isfinite(target_time) || target_time <= committed_.clock.time) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (committed_.clock.frame_ordinal == std::numeric_limits<std::uint64_t>::max()) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  CopyState(committed_, frame_start_);
  target_time_ = target_time;
  frame_active_ = true;
  capacity_restarted_ = false;
  fatal_ = false;
  retry_count_ = 0;
  retry_duration_.reset();
  return absl::OkStatus();
}
absl::StatusOr<LiquidState3D*> LiquidFrameTransaction3D::ResetTrialFromCommitted() {
  if (!frame_active_ || trial_active_ || capacity_pending_ || fatal_ || committed_.clock.time == target_time_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  if (!Reserved(trial_, capacities_)) {
    fatal_ = true;
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  CopyState(committed_, trial_);
  trial_active_ = true;
  return &trial_;
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
  const double remaining = target_time_ - committed_.clock.time;
  double accepted_time = committed_.clock.time + attempted_duration;
  if (attempted_duration > remaining || !std::isfinite(accepted_time) || accepted_time <= committed_.clock.time) {
    return absl::Status(absl::StatusCode::kOutOfRange, {});
  }
  // Only the scheduled final remaining interval may snap. The predecessor/successor
  // check makes the roundoff allowance explicit, rather than hiding a residual step.
  if (attempted_duration == remaining && accepted_time != target_time_) {
    if (std::nextafter(accepted_time, target_time_) != target_time_) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    accepted_time = target_time_;
  }
  if (accepted_time > target_time_) {
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
      || trial_.random_seed != committed_.random_seed || trial_.next_particle_id < committed_.next_particle_id) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  if (!Reserved(trial_, capacities_) || !Fits(trial_, capacities_)) {
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
  trial_.clock.time = accepted_time;
  ++trial_.clock.epoch;
  ++trial_.clock.accepted_step_ordinal;
  if (accepted_time == target_time_) {
    ++trial_.clock.frame_ordinal;
    trial_.clock.substep_ordinal = 0;
  } else {
    ++trial_.clock.substep_ordinal;
  }
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
  replacement->target_time_ = target_time_;
  replacement->frame_active_ = true;
  replacement->capacity_restarted_ = true;
  *this = std::move(*replacement);
  return absl::OkStatus();
}
absl::Status LiquidFrameTransaction3D::FinishFrame() {
  if (!frame_active_ || trial_active_ || capacity_pending_ || fatal_ || committed_.clock.time != target_time_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  // The final accepted version already contains the completed-frame clock. Closing
  // scheduling must not give one epoch two different persistent byte states.
  frame_active_ = false;
  return absl::OkStatus();
}
}  // namespace fluidity
