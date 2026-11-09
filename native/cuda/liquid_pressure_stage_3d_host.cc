#include <algorithm>
#include <array>
#include <cstdint>
#include <variant>

#include "../src/mac_pressure_3d_workspace_internal.h"
#include "liquid_arena_dispatch_3d.h"
#include "liquid_pressure_stage_3d.h"
#include "mac_pressure_3d_bridge.h"

namespace fluidity::pressure_internal {
// Only a fully fenced successful device projection enters the unused bank.
// Capacity and all input aliases are checked before either owner is modified.
struct BoundedCudaPressureAccess3D {
  static absl::Status
  Check(const MacPressureWorkspace3D& workspace, SparseMacGridView3D grid, const LiquidPressureProblemView3D& problem) {
    if (problem.cells.size() > workspace.cell_capacity_ || problem.faces.size() > workspace.face_capacity_) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    const auto overlaps_inputs = [&](const auto& records) {
      const auto* address = records.values().data();
      const std::size_t bytes = records.capacity() * sizeof(records[0]);
      return cuda_internal::LiquidByteRangesOverlap3D(address, bytes, problem.cells.data(), problem.cells.size_bytes())
          || cuda_internal::LiquidByteRangesOverlap3D(address, bytes, problem.faces.data(), problem.faces.size_bytes())
          || cuda_internal::LiquidByteRangesOverlap3D(
                 address,
                 bytes,
                 grid.active_block_indices().data(),
                 grid.active_block_indices().size_bytes()
          );
    };
    const auto& candidate = workspace.projections_[1 - workspace.published_bank_];
    if (overlaps_inputs(workspace.scratch_) || overlaps_inputs(candidate.pressures)
        || overlaps_inputs(candidate.velocities) || overlaps_inputs(candidate.rhs)
        || overlaps_inputs(candidate.residuals) || overlaps_inputs(candidate.defects)
        || overlaps_inputs(candidate.gauges)) {
      return absl::Status(absl::StatusCode::kInvalidArgument, {});
    }
    return absl::OkStatus();
  }
  static MacPressureProjectionView3D Publish(
      MacPressureWorkspace3D& workspace,
      const PreparedMacPressureUploadView3D& prepared,
      std::span<const double> output,
      const cuda_internal::PressureState3D& state
  ) {
    const std::size_t cell_count = prepared.system.cells().size();
    const std::size_t face_count = prepared.system.links().size();
    auto& candidate = workspace.projections_[1 - workspace.published_bank_];
    const auto copy = [](auto& records, const auto values) {
      (void)records.Resize(values.size());
      std::ranges::copy(values, records.values().begin());
    };
    copy(candidate.pressures, output.subspan(cell_count, cell_count));
    copy(candidate.residuals, output.subspan(2 * cell_count, cell_count));
    copy(candidate.defects, output.subspan(3 * cell_count, cell_count));
    copy(candidate.velocities, output.subspan(4 * cell_count, face_count));
    copy(candidate.rhs, prepared.system.integrated_rhs());
    copy(candidate.gauges, prepared.system.gauge_cell_ordinals());
    workspace.published_bank_ = 1 - workspace.published_bank_;
    workspace.high_water_bytes_ = std::max(workspace.high_water_bytes_, workspace.used_bytes());
    return {
        .cell_pressures = candidate.pressures.values(),
        .corrected_face_velocities = candidate.velocities.values(),
        .integrated_rhs = candidate.rhs.values(),
        .true_residuals = candidate.residuals.values(),
        .cell_constraint_defects = candidate.defects.values(),
        .gauge_cell_ordinals = candidate.gauges.values(),
        .diagnostics =
            {
                .iteration_count = state.iterations,
                .multigrid_level_count = prepared.hierarchy.levels.size(),
                .true_residual_l2 = state.residual_l2,
                .max_volume_scaled_residual = state.maximum_scaled_residual,
                .max_cell_constraint_defect = state.maximum_defect,
            },
    };
  }
};
}  // namespace fluidity::pressure_internal

namespace fluidity::cuda_internal {
namespace {
using pressure_internal::BoundedCudaPressureAccess3D;
using pressure_internal::PreparedMacPressureUploadView3D;

#define FLUIDITY_PRESSURE_TRY(expression) \
  do { \
    const auto status = (expression); \
    if (!status.ok()) { \
      return status; \
    } \
  } while (false)
#define FLUIDITY_PRESSURE_RANGE(name, type, range_name, count) \
  auto name##_result = CheckedLiquidRange3D<type>(arena, CudaLiquidRange3D::range_name, count); \
  if (!name##_result.ok()) { \
    return name##_result.status(); \
  } \
  auto name = *name##_result

// Append in the accepted hierarchy order. Every write is bounded by the named
// typed arena capacity; offsets count records, never bytes. No vector is built.
template <typename Value>
class PressurePacking3D {
 public:
  explicit PressurePacking3D(LiquidTypedRange3D<Value>& range) : range_(range) {
  }
  absl::Status Append(std::span<const Value> values, std::size_t& offset) {
    if (values.size() > range_.pinned.size() - used_) {
      return absl::Status(absl::StatusCode::kResourceExhausted, {});
    }
    offset = used_;
    std::ranges::copy(values, range_.pinned.begin() + used_);
    used_ += values.size();
    return absl::OkStatus();
  }
  absl::Status Append(Value value) {
    std::size_t unused_offset = 0;
    return Append(std::span<const Value>(&value, 1), unused_offset);
  }
  std::size_t used() const {
    return used_;
  }
  void Finish() {
    range_.pinned = range_.pinned.first(used_);
  }

 private:
  LiquidTypedRange3D<Value>& range_;
  std::size_t used_ = 0;
};

// Prepared spans come only from the checked assembly/hierarchy builder. This
// stage repacks immutable coefficients, without pressure recurrence or correction.
absl::Status PackHierarchy(
    const PreparedMacPressureUploadView3D& prepared,
    LiquidTypedRange3D<PressureLevel3D>& levels,
    LiquidTypedRange3D<PressureBlock3D>& blocks,
    LiquidTypedRange3D<std::size_t>& indices,
    LiquidTypedRange3D<double>& coefficients,
    PressureDevice3D& device
) {
  PressurePacking3D packed_indices(indices);
  PressurePacking3D packed_coefficients(coefficients);
  PressurePacking3D packed_blocks(blocks);
  FLUIDITY_PRESSURE_TRY(packed_indices.Append(prepared.hierarchy.unknown_to_cell, device.unknown_map_offset));
  std::size_t level_ordinal = 0;
  for (const auto& level : prepared.hierarchy.levels) {
    auto& packed = levels.pinned[level_ordinal++];
    packed = {};
    packed.count = level.coordinates.size();
    packed.scratch_offset = device.level_unknown_count;
    device.level_unknown_count += packed.count;
    FLUIDITY_PRESSURE_TRY(packed_coefficients.Append(level.anchor_diagonal, packed.anchor_offset));
    FLUIDITY_PRESSURE_TRY(packed_coefficients.Append(level.diagonal, packed.diagonal_offset));
    packed.edge_offset = packed_coefficients.used();
    for (const auto& edge : level.edges) {
      FLUIDITY_PRESSURE_TRY(packed_coefficients.Append(edge.conductance));
    }
    FLUIDITY_PRESSURE_TRY(packed_indices.Append(level.row_offsets, packed.row_offset));
    packed.neighbor_offset = packed_indices.used();
    for (const auto& neighbor : level.neighbors) {
      FLUIDITY_PRESSURE_TRY(packed_indices.Append(neighbor.unknown_ordinal));
    }
    packed.neighbor_edge_offset = packed_indices.used();
    for (const auto& neighbor : level.neighbors) {
      FLUIDITY_PRESSURE_TRY(packed_indices.Append(neighbor.edge_ordinal));
    }
    FLUIDITY_PRESSURE_TRY(packed_indices.Append(level.aggregate_ordinals, packed.aggregate_offset));
    packed.block_offset = packed_blocks.used();
    packed.block_count = level.terminal_offsets.empty() ? 0 : level.terminal_offsets.size() - 1;
    std::size_t factor_offset = 0;
    for (std::size_t block_ordinal = 0; block_ordinal < packed.block_count; ++block_ordinal) {
      const std::size_t unknown_offset = level.terminal_offsets[block_ordinal];
      const std::size_t count = level.terminal_offsets[block_ordinal + 1] - unknown_offset;
      PressureBlock3D block{.count = count};
      FLUIDITY_PRESSURE_TRY(packed_indices.Append(
          std::span<const std::size_t>(level.terminal_unknowns.subspan(unknown_offset, count)),
          block.unknown_offset
      ));
      FLUIDITY_PRESSURE_TRY(packed_coefficients.Append(
          std::span<const double>(level.terminal_factors.subspan(factor_offset, count * count)),
          block.factor_offset
      ));
      factor_offset += count * count;
      FLUIDITY_PRESSURE_TRY(packed_blocks.Append(block));
    }
  }
  packed_blocks.Finish();
  packed_indices.Finish();
  packed_coefficients.Finish();
  return absl::OkStatus();
}

// Device recurrence retains FP32 fine vectors and FP64 coarse reductions. The
// host reads only its status to decide whether another exact iteration is needed.
absl::Status Execute(
    CudaLiquidArena3D& arena,
    const PreparedMacPressureUploadView3D& prepared,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace,
    MacPressureProjectionView3D& projection
) {
  const std::size_t cell_count = prepared.system.cells().size();
  const std::size_t face_count = prepared.system.links().size();
  const std::size_t level_count = prepared.hierarchy.levels.size();
  const auto capacity = [&](CudaLiquidRange3D name, std::size_t stride) {
    return arena.range(name).pinned.size() / stride;
  };
  FLUIDITY_PRESSURE_RANGE(cells, PressureCell3D, kPressureCells, cell_count);
  FLUIDITY_PRESSURE_RANGE(links, PressureLink3D, kPressureLinks, face_count);
  FLUIDITY_PRESSURE_RANGE(levels, PressureLevel3D, kPressureLevels, level_count);
  FLUIDITY_PRESSURE_RANGE(
      blocks,
      PressureBlock3D,
      kPressureBlocks,
      capacity(CudaLiquidRange3D::kPressureBlocks, sizeof(PressureBlock3D))
  );
  FLUIDITY_PRESSURE_RANGE(
      indices,
      std::size_t,
      kPressureIndices,
      capacity(CudaLiquidRange3D::kPressureIndices, sizeof(std::size_t))
  );
  FLUIDITY_PRESSURE_RANGE(
      coefficients,
      double,
      kPressureCoefficients,
      capacity(CudaLiquidRange3D::kPressureCoefficients, sizeof(double))
  );
  PressureDevice3D device{};
  device.cells = cells.device;
  device.links = links.device;
  device.levels = levels.device;
  device.blocks = blocks.device;
  device.indices = indices.device;
  device.coefficients = coefficients.device;
  device.cell_count = cell_count;
  device.face_count = face_count;
  device.level_count = level_count;
  device.unknown_count = prepared.hierarchy.unknown_to_cell.size();
  device.max_iterations = options.max_iterations;
  device.integrated_limit = prepared.stopping.integrated_residual_limit;
  device.local_limit = prepared.stopping.volume_scaled_limit;
  FLUIDITY_PRESSURE_TRY(PackHierarchy(prepared, levels, blocks, indices, coefficients, device));
  // All products below are bounded by the checked 4096-cell/33-level builder.
  FLUIDITY_PRESSURE_RANGE(fine, float, kPressureFineScratch, 6 * device.unknown_count);
  FLUIDITY_PRESSURE_RANGE(coarse, double, kPressureCoarseScratch, 3 * device.level_unknown_count);
  FLUIDITY_PRESSURE_RANGE(rows, PressureRow3D, kPressureRows, 2 * cell_count);
  FLUIDITY_PRESSURE_RANGE(output, double, kPressureOutput, 4 * cell_count + face_count);
  FLUIDITY_PRESSURE_RANGE(state, PressureState3D, kPressureState, 1);
  device.fine_scratch = fine.device;
  device.coarse_scratch = coarse.device;
  device.rows = rows.device;
  device.output = output.device;
  device.state = state.device;
  for (std::size_t ordinal = 0; ordinal < cell_count; ++ordinal) {
    const auto& cell = prepared.system.cells()[ordinal];
    cells.pinned[ordinal] = {
        .fluid_volume = cell.fluid_volume,
        .integrated_target_flux = cell.integrated_target_flux,
        .compliance = cell.compliance,
        .previous_pressure = cell.previous_pressure,
        .integrated_rhs = prepared.system.integrated_rhs()[ordinal],
        .rhs_absolute_term_sum = prepared.system.rhs_absolute_term_sums()[ordinal],
        .initial_pressure = prepared.initial_pressure[ordinal],
    };
  }
  for (std::size_t ordinal = 0; ordinal < face_count; ++ordinal) {
    const auto& link = prepared.system.links()[ordinal];
    double jump = 0;
    double exterior_pressure = 0;
    if (const auto* internal = std::get_if<InternalMacPressureLink3D>(&link.pressure_link)) {
      jump = internal->pressure_jump;
    } else if (const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&link.pressure_link)) {
      exterior_pressure = boundary->exterior_pressure;
    }
    links.pinned[ordinal] = {
        .tail_cell_ordinal = link.tail_cell_ordinal.value_or(cell_count),
        .head_cell_ordinal = link.head_cell_ordinal.value_or(cell_count),
        .open_area = link.open_area,
        .blocked_area = link.blocked_area,
        .pressure_to_velocity_scale = link.pressure_to_velocity_scale,
        .conductance = link.conductance,
        .predicted_normal_velocity = link.predicted_normal_velocity,
        .wall_normal_velocity = link.wall_normal_velocity,
        .jump = jump,
        .exterior_pressure = exterior_pressure,
    };
  }
  FLUIDITY_PRESSURE_TRY(CopyLiquidRange3D(arena, cells, true));
  FLUIDITY_PRESSURE_TRY(CopyLiquidRange3D(arena, links, true));
  FLUIDITY_PRESSURE_TRY(CopyLiquidRange3D(arena, levels, true));
  FLUIDITY_PRESSURE_TRY(CopyLiquidRange3D(arena, blocks, true));
  FLUIDITY_PRESSURE_TRY(CopyLiquidRange3D(arena, indices, true));
  FLUIDITY_PRESSURE_TRY(CopyLiquidRange3D(arena, coefficients, true));
  const auto clear = [&](auto& range) {
    if (range.pinned.empty()) {
      return absl::OkStatus();
    }
    return arena.RecordRuntimeResult(
        LiquidArenaRuntimeCalls()
            .memset_async(range.device, 0, range.pinned.size_bytes(), static_cast<cudaStream_t>(arena.stream()))
    );
  };
  FLUIDITY_PRESSURE_TRY(clear(fine));
  FLUIDITY_PRESSURE_TRY(clear(coarse));
  FLUIDITY_PRESSURE_TRY(clear(rows));
  FLUIDITY_PRESSURE_TRY(clear(output));
  FLUIDITY_PRESSURE_TRY(clear(state));
  for (std::size_t launch = 0; launch <= options.max_iterations; ++launch) {
    FLUIDITY_PRESSURE_TRY(
        arena.RecordRuntimeResult(LiquidArenaRuntimeCalls().launch_pressure(device, launch == 0, arena.stream()))
    );
    FLUIDITY_PRESSURE_TRY(CopyLiquidRange3D(arena, state, false));
    FLUIDITY_PRESSURE_TRY(FinishLiquidDispatch3D(arena));
    if (state.pinned[0].error != PressureError3D::kNone || state.pinned[0].converged) {
      break;
    }
  }
  const auto& report = state.pinned[0];
  if (report.compiled_architecture != 1200) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, {});
  }
  switch (report.error) {
    case PressureError3D::kNone:
      break;
    case PressureError3D::kRange:
      return absl::Status(absl::StatusCode::kOutOfRange, {});
    case PressureError3D::kIdentity:
      return absl::Status(absl::StatusCode::kInternal, {});
    case PressureError3D::kCurvature:
    case PressureError3D::kExhausted:
      return absl::Status(absl::StatusCode::kAborted, {});
  }
  if (!report.converged) {
    return absl::Status(absl::StatusCode::kInternal, {});
  }
  FLUIDITY_PRESSURE_TRY(CopyLiquidRange3D(arena, output, false));
  FLUIDITY_PRESSURE_TRY(FinishLiquidDispatch3D(arena));
  projection = BoundedCudaPressureAccess3D::Publish(workspace, prepared, output.pinned, report);
  return absl::OkStatus();
}
#undef FLUIDITY_PRESSURE_RANGE
#undef FLUIDITY_PRESSURE_TRY
}  // namespace

absl::StatusOr<MacPressureProjectionView3D> ProjectLiquidPressure3D(
    CudaLiquidArena3D& arena,
    SparseMacGridView3D grid,
    double step_duration,
    const LiquidPressureProblemView3D& problem,
    MacPressureSolverOptions3D options,
    MacPressureWorkspace3D& workspace
) {
  const auto ready = arena.RecordRuntimeResult(cudaSuccess);
  if (!ready.ok()) {
    return ready;
  }
  if (options.max_iterations == 0 || options.max_iterations > 256) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto capacities = arena.capacities();
  if (problem.cells.size() > capacities.cells || problem.faces.size() > capacities.faces
      || grid.active_block_indices().size() > capacities.blocks) {
    return absl::Status(absl::StatusCode::kResourceExhausted, {});
  }
  if (AliasesLiquidScratch3D(arena, problem.cells) || AliasesLiquidScratch3D(arena, problem.faces)
      || AliasesLiquidScratch3D(arena, grid.active_block_indices())) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  const auto checked = BoundedCudaPressureAccess3D::Check(workspace, grid, problem);
  if (!checked.ok()) {
    return checked;
  }
  const auto preparation = arena.range(CudaLiquidRange3D::kPressurePreparationScratch).pinned;
  if (reinterpret_cast<std::uintptr_t>(preparation.data()) % alignof(std::max_align_t) != 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, {});
  }
  pressure_internal::MacPressureScratch3D scratch(
      {reinterpret_cast<std::max_align_t*>(preparation.data()), preparation.size() / sizeof(std::max_align_t)}
  );
  const auto prepared =
      pressure_internal::PrepareBoundedMacPressureUpload3D(grid, step_duration, problem.cells, problem.faces, scratch);
  if (!prepared.ok()) {
    return prepared.status();
  }
  MacPressureProjectionView3D result;
  const auto status = Execute(arena, *prepared, options, workspace, result);
  if (!status.ok()) {
    return status;
  }
  return result;
}
}  // namespace fluidity::cuda_internal
