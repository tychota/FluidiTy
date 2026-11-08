#include "fluidity/host/liquid_step_workspace_3d.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

namespace fluidity {
namespace {
constexpr std::size_t kFacesPerBlock = 192;
constexpr std::size_t kParticleSupportSize = 81;
constexpr std::size_t kPressureActivationSize = 7;
constexpr auto kByteLimit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
template <typename Index>
auto Key(Index index) {
  return std::tuple(index.z, index.y, index.x);
}
bool SameGeometry(UniformMacGeometry3D left, UniformMacGeometry3D right) {
  return left.world_origin.x == right.world_origin.x && left.world_origin.y == right.world_origin.y
      && left.world_origin.z == right.world_origin.z && left.cell_width == right.cell_width;
}
// Owner-key, normal, local lattice ordering is independent of block ordinal.
// This is exactly owned-slot order even when old and new block arrays differ.
auto FaceKey(FaceIndex3D face) {
  return std::tuple(Key(MacOwnerBlock3D(face.lattice_index)), face.axis, Key(face.lattice_index));
}
// Create is outside attempts, but allocation failure must not allocate its error.
template <typename Record>
absl::Status Reserve(LiquidRecords3D<Record>& destination, std::size_t capacity) {
  auto records = LiquidRecords3D<Record>::Create(capacity);
  if (!records.ok()) {
    return records.status();
  }
  destination = std::move(*records);
  return absl::OkStatus();
}
// Every caller checked these extents before changing any published storage.
template <typename Record>
void ResizeWithinCapacity(LiquidRecords3D<Record>& records, std::size_t size) {
  const auto status = records.Resize(size);
  (void)status;
}
template <typename Record>
void CopyRecords(const LiquidRecords3D<Record>& source, LiquidRecords3D<Record>& destination) {
  ResizeWithinCapacity(destination, source.size());
  std::ranges::copy(source.values(), destination.values().begin());
}
}  // namespace

absl::StatusOr<LiquidWorkspaceBytes3D> QueryLiquidWorkspaceBytes3D(LiquidStateCapacities3D capacities) {
  LiquidWorkspaceBytes3D bytes;
  if (capacities.particles > kByteLimit / kParticleSupportSize) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  const std::size_t particle_keys = kParticleSupportSize * capacities.particles;
  if (capacities.cells > (kByteLimit - particle_keys) / kPressureActivationSize) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  bytes.activation_key_capacity = particle_keys + kPressureActivationSize * capacities.cells;
  const auto add_region = [&](std::size_t count, std::size_t stride, std::size_t& region_bytes) {
    if (count > (kByteLimit - bytes.total_bytes) / stride) {
      return false;
    }
    region_bytes = count * stride;
    bytes.total_bytes += region_bytes;
    return true;
  };
  const bool regions_fit =
      add_region(capacities.particles, 2 * sizeof(LiquidParticleSortEntry3D), bytes.particle_order_bytes)
      && add_region(capacities.particles, 2 * sizeof(LiquidParticleState3D), bytes.particle_state_bytes)
      && add_region(capacities.blocks, 2 * sizeof(BlockIndex3D), bytes.block_key_bytes)
      && add_region(capacities.blocks, 2 * sizeof(LiquidBlockState3D), bytes.block_state_bytes)
      && add_region(capacities.cells, 2 * sizeof(LiquidCellState3D), bytes.cell_state_bytes)
      && add_region(capacities.faces, 2 * sizeof(LiquidFaceState3D), bytes.face_state_bytes)
      && add_region(bytes.activation_key_capacity, sizeof(BlockIndex3D), bytes.activation_key_bytes);
  if (!regions_fit || capacities.blocks > kByteLimit / kFacesPerBlock) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  const auto boundary = QueryLiquidBoundaryWorkspaceBytes3D(capacities);
  if (!boundary.ok()) {
    return boundary.status();
  }
  if (boundary->total_bytes > kByteLimit - bytes.total_bytes) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  bytes.boundary_bytes = boundary->total_bytes;
  bytes.total_bytes += bytes.boundary_bytes;
  const auto pressure = QueryMacPressureWorkspaceBytes3D(capacities.cells, capacities.faces);
  if (!pressure.ok()) {
    return pressure.status();
  }
  if (pressure->total_bytes > kByteLimit - bytes.total_bytes) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  bytes.pressure_bytes = pressure->total_bytes;
  bytes.total_bytes += bytes.pressure_bytes;
  return bytes;
}

absl::StatusOr<LiquidStepWorkspace3D>
LiquidStepWorkspace3D::Create(LiquidStateCapacities3D capacities, UniformMacGeometry3D geometry) {
  const auto empty_grid = SparseMacGridView3D::Create(geometry, {});
  if (!empty_grid.ok()) {
    return empty_grid.status();
  }
  const auto bytes = QueryLiquidWorkspaceBytes3D(capacities);
  if (!bytes.ok()) {
    return bytes.status();
  }
  LiquidStepWorkspace3D workspace;
  workspace.geometry_ = geometry;
  workspace.capacities_ = capacities;
  workspace.bytes_ = *bytes;
  auto boundary = LiquidBoundaryWorkspace3D::Create(capacities);
  if (!boundary.ok()) {
    return boundary.status();
  }
  workspace.boundary_ = std::move(*boundary);
  auto pressure = MacPressureWorkspace3D::Create(capacities.cells, capacities.faces);
  if (!pressure.ok()) {
    return pressure.status();
  }
  workspace.pressure_ = std::move(*pressure);
  for (auto& bank : workspace.banks_) {
    if (const auto status = Reserve(bank.particle_order, capacities.particles); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.particles, capacities.particles); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.block_keys, capacities.blocks); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.blocks, capacities.blocks); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.cells, capacities.cells); !status.ok()) {
      return status;
    }
    if (const auto status = Reserve(bank.faces, capacities.faces); !status.ok()) {
      return status;
    }
  }
  if (const auto status = Reserve(workspace.activation_keys_, bytes->activation_key_capacity); !status.ok()) {
    return status;
  }
  return workspace;
}

absl::Status LiquidStepWorkspace3D::PrepareTrialTopology(
    std::span<const LiquidParticleState3D> particles,
    std::span<const CellIndex3D> pressure_cells,
    const LiquidState3D* previous_state
) {
  if (particles.size() > capacities_.particles || pressure_cells.size() > capacities_.cells) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  std::uint64_t next_generation = 1;
  if (previous_state != nullptr) {
    if (!SameGeometry(geometry_, previous_state->geometry)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    if (const auto status = ValidateLiquidState3D(*previous_state); !status.ok()) {
      return status;
    }
    if (previous_state->clock.epoch == std::numeric_limits<std::uint64_t>::max()) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    next_generation = previous_state->clock.epoch + 1;
  }
  const std::size_t candidate_index = 1 - published_bank_;
  auto& candidate = banks_[candidate_index];
  const std::size_t activation_count =
      kParticleSupportSize * particles.size() + kPressureActivationSize * pressure_cells.size();
  ResizeWithinCapacity(activation_keys_, activation_count);
  ResizeWithinCapacity(candidate.particle_order, particles.size());
  ResizeWithinCapacity(candidate.particles, particles.size());
  ResizeWithinCapacity(candidate.cells, pressure_cells.size());
  std::size_t key_count = 0;
  // Geometry enumeration precedes every owner lookup. Never filter on weight or tag:
  // a zero-weight edge still belongs to the original quadratic stencil.
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const Vec3 position = particles[index].world_position;
    const auto bin = ComputeParticleSortBin3D(geometry_, position);
    if (!bin.ok()) {
      return bin.status();
    }
    candidate.particle_order[index] = {.bin_index = *bin, .original_ordinal = index};
    for (const auto axis : {FaceNormalAxis::kX, FaceNormalAxis::kY, FaceNormalAxis::kZ}) {
      const auto stencil = BuildMacFaceGeometryStencil3D(geometry_, axis, position, position);
      if (!stencil.ok()) {
        return stencil.status();
      }
      for (const auto& sample : *stencil) {
        activation_keys_[key_count++] = MacOwnerBlock3D(sample.face_index.lattice_index);
      }
    }
  }
  for (std::size_t index = 0; index < pressure_cells.size(); ++index) {
    const auto cell = pressure_cells[index];
    // Widening is unnecessary after checking each positive incident neighbor.
    if (cell.x == std::numeric_limits<std::int32_t>::max() || cell.y == std::numeric_limits<std::int32_t>::max()
        || cell.z == std::numeric_limits<std::int32_t>::max()) {
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    }
    candidate.cells[index] = {.cell_index = cell};
    const FaceLatticeIndex3D low{cell.x, cell.y, cell.z};
    activation_keys_[key_count++] = MacOwnerBlock3D(low);
    const std::array high_faces{
        FaceLatticeIndex3D{cell.x + 1, cell.y, cell.z},
        FaceLatticeIndex3D{cell.x, cell.y + 1, cell.z},
        FaceLatticeIndex3D{cell.x, cell.y, cell.z + 1},
    };
    for (auto high : high_faces) {
      activation_keys_[key_count++] = MacOwnerBlock3D(low);
      activation_keys_[key_count++] = MacOwnerBlock3D(high);
    }
  }
  // The explicit ordinal makes an in-place bounded std::sort stable in physical
  // order without std::stable_sort's optional heap buffer.
  std::ranges::sort(candidate.particle_order.values(), [](const auto& left, const auto& right) {
    return std::tuple(Key(left.bin_index), left.original_ordinal)
        < std::tuple(Key(right.bin_index), right.original_ordinal);
  });
  for (std::size_t index = 0; index < particles.size(); ++index) {
    candidate.particles[index] = particles[candidate.particle_order[index].original_ordinal];
  }
  std::ranges::sort(activation_keys_.values(), {}, Key<BlockIndex3D>);
  const auto unique_keys = std::ranges::unique(activation_keys_.values(), {}, Key<BlockIndex3D>);
  const auto block_count = activation_keys_.size() == 0
      ? 0
      : static_cast<std::size_t>(unique_keys.begin() - activation_keys_.values().begin());
  if (block_count > capacities_.blocks || block_count > capacities_.faces / kFacesPerBlock) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  ResizeWithinCapacity(candidate.block_keys, block_count);
  ResizeWithinCapacity(candidate.blocks, block_count);
  std::copy_n(activation_keys_.values().begin(), block_count, candidate.block_keys.values().begin());
  const auto grid = SparseMacGridView3D::Create(geometry_, candidate.block_keys.values());
  if (!grid.ok()) {
    return grid.status();
  }
  for (std::size_t index = 0; index < block_count; ++index) {
    const auto key = candidate.block_keys[index];
    candidate.blocks[index] = {.block_index = key, .generation = next_generation};
    if (previous_state != nullptr) {
      const auto old_blocks = previous_state->blocks.values();
      const auto old = std::ranges::lower_bound(old_blocks, Key(key), {}, [](const auto& block) {
        return Key(block.block_index);
      });
      if (old != old_blocks.end() && Key(old->block_index) == Key(key)) {
        candidate.blocks[index] = *old;
        candidate.blocks[index].inactive_age = 0;
      }
    }
  }
  std::ranges::sort(candidate.cells.values(), {}, [](const auto& cell) {
    return Key(cell.cell_index);
  });
  const auto duplicate_cells = std::ranges::unique(candidate.cells.values(), {}, [](const auto& cell) {
    return Key(cell.cell_index);
  });
  const auto cell_count = candidate.cells.size() == 0
      ? 0
      : static_cast<std::size_t>(duplicate_cells.begin() - candidate.cells.values().begin());
  ResizeWithinCapacity(candidate.cells, cell_count);
  if (previous_state != nullptr) {
    const auto old_cells = previous_state->cells.values();
    for (auto& cell : candidate.cells.values()) {
      const auto old = std::ranges::lower_bound(old_cells, Key(cell.cell_index), {}, [](const auto& value) {
        return Key(value.cell_index);
      });
      if (old != old_cells.end() && Key(old->cell_index) == Key(cell.cell_index)) {
        cell = *old;
      }
    }
  }
  ResizeWithinCapacity(candidate.faces, grid->face_count());
  for (std::size_t slot = 0; slot < grid->face_count(); ++slot) {
    // Valid slot by loop bound and validated block lattice.
    const FaceIndex3D face_index = *grid->FaceIndex(slot);
    candidate.faces[slot] = {.face_index = face_index, .face_slot = slot};
    if (previous_state != nullptr) {
      const auto old_faces = previous_state->faces.values();
      const auto old = std::ranges::lower_bound(old_faces, FaceKey(face_index), {}, [](const auto& face) {
        return FaceKey(face.face_index);
      });
      if (old != old_faces.end() && FaceKey(old->face_index) == FaceKey(face_index)) {
        candidate.faces[slot] = *old;
        candidate.faces[slot].face_slot = slot;
      }
    }
  }
  // This is the only publication. All capacities, lattice arithmetic and history
  // validation have succeeded; the previous published bank was never written.
  published_bank_ = candidate_index;
  prepared_ = true;
  return absl::OkStatus();
}

absl::StatusOr<LiquidPreparedTopologyView3D> LiquidStepWorkspace3D::prepared() const {
  if (!prepared_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  const auto& bank = banks_[published_bank_];
  const auto grid = SparseMacGridView3D::Create(geometry_, bank.block_keys.values());
  if (!grid.ok()) {
    return grid.status();
  }
  return LiquidPreparedTopologyView3D{
      .grid = *grid,
      .particle_order = bank.particle_order.values(),
      .particles = bank.particles.values(),
      .blocks = bank.blocks.values(),
      .cells = bank.cells.values(),
      .faces = bank.faces.values(),
  };
}

absl::Status LiquidStepWorkspace3D::ApplyTrialTopology(LiquidState3D& trial) const {
  if (!prepared_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  if (!SameGeometry(geometry_, trial.geometry)) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto& bank = banks_[published_bank_];
  if (trial.particles.capacity() < bank.particles.size() || trial.blocks.capacity() < bank.blocks.size()
      || trial.cells.capacity() < bank.cells.size() || trial.faces.capacity() < bank.faces.size()) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  CopyRecords(bank.particles, trial.particles);
  CopyRecords(bank.blocks, trial.blocks);
  CopyRecords(bank.cells, trial.cells);
  CopyRecords(bank.faces, trial.faces);
  return absl::OkStatus();
}
LiquidWorkspaceUsage3D LiquidStepWorkspace3D::usage() const {
  LiquidWorkspaceUsage3D usage;
  usage.reserved_bytes = bytes_.total_bytes;
  usage.activation_keys = activation_keys_.size();
  usage.used_bytes = activation_keys_.size() * sizeof(BlockIndex3D) + boundary_.used_bytes() + pressure_.used_bytes();
  for (const auto& bank : banks_) {
    usage.used_bytes += bank.particle_order.size() * sizeof(LiquidParticleSortEntry3D)
        + bank.particles.size() * sizeof(LiquidParticleState3D) + bank.block_keys.size() * sizeof(BlockIndex3D)
        + bank.blocks.size() * sizeof(LiquidBlockState3D) + bank.cells.size() * sizeof(LiquidCellState3D)
        + bank.faces.size() * sizeof(LiquidFaceState3D);
  }
  if (prepared_) {
    const auto& bank = banks_[published_bank_];
    usage.active = {
        .particles = bank.particles.size(),
        .blocks = bank.blocks.size(),
        .cells = bank.cells.size(),
        .faces = bank.faces.size(),
        .sources = 0,
    };
  }
  return usage;
}
LiquidStateCapacities3D LiquidStepWorkspace3D::capacities() const {
  return capacities_;
}
}  // namespace fluidity
