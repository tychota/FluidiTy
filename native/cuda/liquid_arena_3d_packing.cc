#include <cstring>

#include "../src/liquid_adaptive_state_internal_3d.h"
#include "liquid_arena_3d_internal.h"
namespace fluidity::cuda_internal {
namespace {
template <typename Record>
Record* At(std::byte* bytes, std::size_t offset) {
  return reinterpret_cast<Record*>(bytes + offset);
}
template <typename Record>
const Record* At(const std::byte* bytes, std::size_t offset) {
  return reinterpret_cast<const Record*>(bytes + offset);
}
void PackVector(Vec3 vector, double (&values)[3][32], std::size_t lane) {
  values[0][lane] = vector.x;
  values[1][lane] = vector.y;
  values[2][lane] = vector.z;
}
Vec3 UnpackVector(const double (&values)[3][32], std::size_t lane) {
  return {values[0][lane], values[1][lane], values[2][lane]};
}
}  // namespace
void PackLiquidArenaState3D(
    const LiquidState3D& source,
    LiquidClock3D clock,
    const CudaLiquidStateLayout3D& layout,
    std::byte* destination
) {
  std::memset(destination, 0, layout.total_state_bytes);
  auto* bins = At<CudaLiquidParticleBin3D>(destination, layout.particle_bins_byte_offset);
  for (std::size_t ordinal = 0; ordinal < source.particles.size(); ++ordinal) {
    const auto& particle = source.particles[ordinal];
    auto& bin = bins[ordinal / 32];
    const auto lane = ordinal % 32;
    PackVector(particle.world_position, bin.world_positions, lane);
    PackVector(particle.world_velocity, bin.world_velocities, lane);
    const std::array rows{
        particle.velocity_slopes.x_row,
        particle.velocity_slopes.y_row,
        particle.velocity_slopes.z_row,
    };
    for (std::size_t row = 0; row < 3; ++row) {
      bin.particle_velocity_slopes[row * 3][lane] = rows[row].x;
      bin.particle_velocity_slopes[row * 3 + 1][lane] = rows[row].y;
      bin.particle_velocity_slopes[row * 3 + 2][lane] = rows[row].z;
    }
    bin.particle_masses[lane] = particle.mass;
    bin.reference_volumes[lane] = particle.reference_volume;
    bin.volume_ratios[lane] = particle.volume_ratio;
    bin.particle_ids[lane] = particle.id;
    bin.side_tags[lane] = particle.side_tag;
    bin.particle_tags[lane] = particle.tag;
    bin.affine_pic_flags[lane] = (particle.affine_enabled ? 1U : 0U) | (particle.pic_initialized ? 2U : 0U);
  }
  auto* blocks = At<LiquidBlockState3D>(destination, layout.block_records_byte_offset);
  for (std::size_t ordinal = 0; ordinal < source.blocks.size(); ++ordinal) {
    const auto& record = source.blocks[ordinal];
    blocks[ordinal].block_index.x = record.block_index.x;
    blocks[ordinal].block_index.y = record.block_index.y;
    blocks[ordinal].block_index.z = record.block_index.z;
    blocks[ordinal].generation = record.generation;
    blocks[ordinal].activation_age = record.activation_age;
    blocks[ordinal].inactive_age = record.inactive_age;
  }
  auto* cells = At<LiquidCellState3D>(destination, layout.cell_records_byte_offset);
  for (std::size_t ordinal = 0; ordinal < source.cells.size(); ++ordinal) {
    const auto& record = source.cells[ordinal];
    cells[ordinal].cell_index.x = record.cell_index.x;
    cells[ordinal].cell_index.y = record.cell_index.y;
    cells[ordinal].cell_index.z = record.cell_index.z;
    cells[ordinal].occupancy = record.occupancy;
    cells[ordinal].previous_pressure = record.previous_pressure;
  }
  auto* faces = At<LiquidFaceState3D>(destination, layout.face_records_byte_offset);
  for (std::size_t ordinal = 0; ordinal < source.faces.size(); ++ordinal) {
    const auto& record = source.faces[ordinal];
    faces[ordinal].face_index.axis = record.face_index.axis;
    faces[ordinal].face_index.lattice_index.x = record.face_index.lattice_index.x;
    faces[ordinal].face_index.lattice_index.y = record.face_index.lattice_index.y;
    faces[ordinal].face_index.lattice_index.z = record.face_index.lattice_index.z;
    faces[ordinal].face_slot = record.face_slot;
    faces[ordinal].carrier_velocity = record.carrier_velocity;
    faces[ordinal].validity = record.validity;
    faces[ordinal].side_tag = record.side_tag;
    faces[ordinal].structurally_eligible = record.structurally_eligible;
  }
  if (layout.adaptive_state) {
    auto* adaptive_cells = At<LiquidAdaptiveCellState3D>(destination, layout.cell_records_byte_offset);
    for (std::size_t ordinal = 0; ordinal < source.adaptive_cells.size(); ++ordinal) {
      const auto& record = source.adaptive_cells[ordinal];
      auto& cell = adaptive_cells[ordinal];
      cell.key.lower_corner.x = record.key.lower_corner.x;
      cell.key.lower_corner.y = record.key.lower_corner.y;
      cell.key.lower_corner.z = record.key.lower_corner.z;
      cell.key.width_exponent = record.key.width_exponent;
      cell.occupancy = record.occupancy;
      cell.material_owner = record.material_owner;
      cell.inventory.mass_kg = record.inventory.mass_kg;
      cell.inventory.reference_volume_m3 = record.inventory.reference_volume_m3;
      cell.inventory.current_volume_m3 = record.inventory.current_volume_m3;
      cell.inventory.momentum_kg_m_per_s.x = record.inventory.momentum_kg_m_per_s.x;
      cell.inventory.momentum_kg_m_per_s.y = record.inventory.momentum_kg_m_per_s.y;
      cell.inventory.momentum_kg_m_per_s.z = record.inventory.momentum_kg_m_per_s.z;
      cell.inventory.enthalpy_j = record.inventory.enthalpy_j;
      cell.inventory.phase_id = record.inventory.phase_id;
      cell.inventory.constituent_count = record.inventory.constituent_count;
      for (std::size_t constituent = 0; constituent < record.inventory.constituent_count; ++constituent) {
        cell.inventory.constituents[constituent].substance_id = record.inventory.constituents[constituent].substance_id;
        cell.inventory.constituents[constituent].volume_m3 = record.inventory.constituents[constituent].volume_m3;
      }
      cell.previous_pressure = record.previous_pressure;
      cell.coarsening_eligible_duration_s = record.coarsening_eligible_duration_s;
    }
    auto* adaptive_faces = At<LiquidAdaptiveFaceState3D>(destination, layout.face_records_byte_offset);
    for (std::size_t ordinal = 0; ordinal < source.adaptive_faces.size(); ++ordinal) {
      const auto& record = source.adaptive_faces[ordinal];
      auto& face = adaptive_faces[ordinal];
      face.carrier_velocity = record.carrier_velocity;
      face.validity = record.validity;
      face.side_tag = record.side_tag;
      face.structurally_eligible = record.structurally_eligible;
    }
  }
  auto* sources = At<LiquidSourceState3D>(destination, layout.source_records_byte_offset);
  for (std::size_t ordinal = 0; ordinal < source.sources.size(); ++ordinal) {
    sources[ordinal].source_ordinal = source.sources[ordinal].source_ordinal;
    sources[ordinal].next_child_ordinal = source.sources[ordinal].next_child_ordinal;
    sources[ordinal].fractional_remainder = source.sources[ordinal].fractional_remainder;
  }
  auto& control = *At<LiquidArenaControl3D>(destination, layout.control_record_byte_offset);
  control.geometry.world_origin.x = source.geometry.world_origin.x;
  control.geometry.world_origin.y = source.geometry.world_origin.y;
  control.geometry.world_origin.z = source.geometry.world_origin.z;
  control.geometry.cell_width = source.geometry.cell_width;
  control.clock.time = clock.time;
  control.clock.frame_ordinal = clock.frame_ordinal;
  control.clock.substep_ordinal = clock.substep_ordinal;
  control.clock.accepted_step_ordinal = clock.accepted_step_ordinal;
  control.clock.epoch = clock.epoch;
  control.active = {
      source.particles.size(),
      source.blocks.size(),
      layout.adaptive_state ? source.adaptive_cells.size() : source.cells.size(),
      layout.adaptive_state ? source.adaptive_faces.size() : source.faces.size(),
      source.sources.size(),
  };
  control.random_seed = source.random_seed;
  control.next_particle_id = source.next_particle_id;
}
absl::Status ValidateLiquidArenaEncoding3D(const CudaLiquidStateLayout3D& layout, const std::byte* source) {
  const auto& control = *At<LiquidArenaControl3D>(source, layout.control_record_byte_offset);
  const auto* bins = At<CudaLiquidParticleBin3D>(source, layout.particle_bins_byte_offset);
  const auto corrupt = absl::Status(absl::StatusCode::kDataLoss, {});
  for (std::size_t ordinal = 0; ordinal < control.active.particles; ++ordinal) {
    if (bins[ordinal / 32].affine_pic_flags[ordinal % 32] > 3U) {
      return corrupt;
    }
  }
  const auto stride = layout.adaptive_state ? sizeof(LiquidAdaptiveFaceState3D) : sizeof(LiquidFaceState3D);
  const auto flag_offset = layout.adaptive_state ? offsetof(LiquidAdaptiveFaceState3D, structurally_eligible)
                                                 : offsetof(LiquidFaceState3D, structurally_eligible);
  for (std::size_t ordinal = 0; ordinal < control.active.faces; ++ordinal) {
    const auto* flag = source + layout.face_records_byte_offset + ordinal * stride + flag_offset;
    const bool false_value = false;
    const bool true_value = true;
    if (std::memcmp(flag, &false_value, sizeof(bool)) != 0 && std::memcmp(flag, &true_value, sizeof(bool)) != 0) {
      return corrupt;
    }
  }
  if (!layout.adaptive_state) {
    return absl::OkStatus();
  }
  struct Records {
    const LiquidAdaptiveCellState3D* cells;
    const LiquidAdaptiveFaceState3D* faces;
  };
  const Records records{
      At<LiquidAdaptiveCellState3D>(source, layout.cell_records_byte_offset),
      At<LiquidAdaptiveFaceState3D>(source, layout.face_records_byte_offset),
  };
  const auto checked = liquid_state_internal::ValidateAdaptiveRecordView3D({
      .context = &records,
      .finest_cell_width = control.geometry.cell_width,
      .cell_count = control.active.cells,
      .face_count = control.active.faces,
      .read_cell =
          [](const void* context, std::size_t ordinal) {
            return static_cast<const Records*>(context)->cells[ordinal];
          },
      .read_face =
          [](const void* context, std::size_t ordinal) {
            return static_cast<const Records*>(context)->faces[ordinal];
          },
  });
  return checked.ok() ? absl::OkStatus() : corrupt;
}
void UnpackLiquidArenaState3D(
    const CudaLiquidStateLayout3D& layout,
    const std::byte* source,
    LiquidState3D& destination
) {
  const auto& control = *At<LiquidArenaControl3D>(source, layout.control_record_byte_offset);
  (void)destination.particles.Resize(control.active.particles);
  (void)destination.blocks.Resize(control.active.blocks);
  (void)destination.cells.Resize(layout.adaptive_state ? 0 : control.active.cells);
  (void)destination.faces.Resize(layout.adaptive_state ? 0 : control.active.faces);
  (void)destination.adaptive_cells.Resize(layout.adaptive_state ? control.active.cells : 0);
  (void)destination.adaptive_faces.Resize(layout.adaptive_state ? control.active.faces : 0);
  (void)destination.sources.Resize(control.active.sources);
  const auto* bins = At<CudaLiquidParticleBin3D>(source, layout.particle_bins_byte_offset);
  for (std::size_t ordinal = 0; ordinal < destination.particles.size(); ++ordinal) {
    auto& particle = destination.particles[ordinal];
    const auto& bin = bins[ordinal / 32];
    const auto lane = ordinal % 32;
    particle.world_position = UnpackVector(bin.world_positions, lane);
    particle.world_velocity = UnpackVector(bin.world_velocities, lane);
    particle.velocity_slopes = {
        {
            bin.particle_velocity_slopes[0][lane],
            bin.particle_velocity_slopes[1][lane],
            bin.particle_velocity_slopes[2][lane],
        },
        {
            bin.particle_velocity_slopes[3][lane],
            bin.particle_velocity_slopes[4][lane],
            bin.particle_velocity_slopes[5][lane],
        },
        {
            bin.particle_velocity_slopes[6][lane],
            bin.particle_velocity_slopes[7][lane],
            bin.particle_velocity_slopes[8][lane],
        },
    };
    particle.mass = bin.particle_masses[lane];
    particle.reference_volume = bin.reference_volumes[lane];
    particle.volume_ratio = bin.volume_ratios[lane];
    particle.id = bin.particle_ids[lane];
    particle.side_tag = bin.side_tags[lane];
    particle.tag = bin.particle_tags[lane];
    particle.affine_enabled = (bin.affine_pic_flags[lane] & 1U) != 0;
    particle.pic_initialized = (bin.affine_pic_flags[lane] & 2U) != 0;
  }
  std::copy_n(
      At<LiquidBlockState3D>(source, layout.block_records_byte_offset),
      control.active.blocks,
      destination.blocks.values().begin()
  );
  if (layout.adaptive_state) {
    std::copy_n(
        At<LiquidAdaptiveCellState3D>(source, layout.cell_records_byte_offset),
        control.active.cells,
        destination.adaptive_cells.values().begin()
    );
    std::copy_n(
        At<LiquidAdaptiveFaceState3D>(source, layout.face_records_byte_offset),
        control.active.faces,
        destination.adaptive_faces.values().begin()
    );
  } else {
    std::copy_n(
        At<LiquidCellState3D>(source, layout.cell_records_byte_offset),
        control.active.cells,
        destination.cells.values().begin()
    );
    std::copy_n(
        At<LiquidFaceState3D>(source, layout.face_records_byte_offset),
        control.active.faces,
        destination.faces.values().begin()
    );
  }
  std::copy_n(
      At<LiquidSourceState3D>(source, layout.source_records_byte_offset),
      control.active.sources,
      destination.sources.values().begin()
  );
  destination.geometry = control.geometry;
  destination.clock = control.clock;
  destination.random_seed = control.random_seed;
  destination.next_particle_id = control.next_particle_id;
}
}  // namespace fluidity::cuda_internal
