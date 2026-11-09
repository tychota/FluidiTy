#include <cstring>

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
  control.active =
      {source.particles.size(), source.blocks.size(), source.cells.size(), source.faces.size(), source.sources.size()};
  control.random_seed = source.random_seed;
  control.next_particle_id = source.next_particle_id;
}
void UnpackLiquidArenaState3D(
    const CudaLiquidStateLayout3D& layout,
    const std::byte* source,
    LiquidState3D& destination
) {
  const auto& control = *At<LiquidArenaControl3D>(source, layout.control_record_byte_offset);
  (void)destination.particles.Resize(control.active.particles);
  (void)destination.blocks.Resize(control.active.blocks);
  (void)destination.cells.Resize(control.active.cells);
  (void)destination.faces.Resize(control.active.faces);
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
