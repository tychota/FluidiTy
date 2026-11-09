#include <cuda_runtime_api.h>

#include <span>
#include <utility>
#include <vector>

#include "../src/mac_pressure_3d_multigrid.h"
#include "fluidity/host/cuda_mac_pressure_3d.h"
#include "mac_pressure_3d_bridge.h"
#include "operator_runtime.h"

namespace fluidity {
namespace {

using cuda_internal::PressureDevice3D;
using cuda_internal::PressureState3D;
// Own all uploads until runtime cleanup has synchronized even an early failure.
struct PressureUpload3D {
  std::vector<cuda_internal::PressureCell3D> cells;
  std::vector<cuda_internal::PressureLink3D> links;
  std::vector<cuda_internal::PressureLevel3D> levels;
  std::vector<cuda_internal::PressureBlock3D> blocks;
  std::vector<std::size_t> indices;
  std::vector<double> coefficients;
  std::size_t level_unknown_count = 0;
  std::size_t unknown_map_offset = 0;
};
template <typename Value>
std::size_t Append(std::vector<Value>& destination, std::span<const Value> source) {
  const auto offset = destination.size();
  destination.insert(destination.end(), source.begin(), source.end());
  return offset;
}
// Only the checked system and shared hierarchy builder supply these ranges.
// Every packed extent is allocated through CudaOperatorRuntime's checked byte
// product; there are eleven tracked allocations independent of level count.
PressureUpload3D PrepareUpload(
    const pressure_internal::PreparedMacPressureSystem3D& system,
    const pressure_internal::MacPressureHierarchy3D& hierarchy,
    std::span<const double> initial
) {
  PressureUpload3D upload;
  for (std::size_t i = 0; i < system.cells().size(); ++i) {
    const auto& cell = system.cells()[i];
    upload.cells.push_back({
        .fluid_volume = cell.fluid_volume,
        .integrated_target_flux = cell.integrated_target_flux,
        .compliance = cell.compliance,
        .previous_pressure = cell.previous_pressure,
        .integrated_rhs = system.integrated_rhs()[i],
        .rhs_absolute_term_sum = system.rhs_absolute_term_sums()[i],
        .initial_pressure = initial[i],
    });
  }
  for (const auto& link : system.links()) {
    double jump = 0.0;
    double exterior = 0.0;
    if (const auto* internal = std::get_if<InternalMacPressureLink3D>(&link.pressure_link)) {
      jump = internal->pressure_jump;
    } else if (const auto* boundary = std::get_if<DirichletMacPressureLink3D>(&link.pressure_link)) {
      exterior = boundary->exterior_pressure;
    }
    upload.links.push_back({
        .tail_cell_ordinal = link.tail_cell_ordinal.value_or(system.cells().size()),
        .head_cell_ordinal = link.head_cell_ordinal.value_or(system.cells().size()),
        .open_area = link.open_area,
        .blocked_area = link.blocked_area,
        .pressure_to_velocity_scale = link.pressure_to_velocity_scale,
        .conductance = link.conductance,
        .predicted_normal_velocity = link.predicted_normal_velocity,
        .prescribed_wall_flux = link.prescribed_wall_flux,
        .jump = jump,
        .exterior_pressure = exterior,
    });
  }
  upload.unknown_map_offset = Append<std::size_t>(upload.indices, hierarchy.unknown_to_cell);
  for (const auto& level : hierarchy.levels) {
    cuda_internal::PressureLevel3D staged{};
    staged.count = level.coordinates.size();
    staged.scratch_offset = upload.level_unknown_count;
    upload.level_unknown_count += staged.count;
    staged.anchor_offset = Append<double>(upload.coefficients, level.anchor_diagonal);
    staged.diagonal_offset = Append<double>(upload.coefficients, level.diagonal);
    staged.edge_offset = upload.coefficients.size();
    for (const auto& edge : level.edges) {
      upload.coefficients.push_back(edge.conductance);
    }
    staged.row_offset = Append<std::size_t>(upload.indices, level.row_offsets);
    staged.neighbor_offset = upload.indices.size();
    for (const auto neighbor : level.neighbors) {
      upload.indices.push_back(neighbor.unknown_ordinal);
    }
    staged.neighbor_edge_offset = upload.indices.size();
    for (const auto neighbor : level.neighbors) {
      upload.indices.push_back(neighbor.edge_ordinal);
    }
    staged.aggregate_offset = Append<std::size_t>(upload.indices, level.aggregate_ordinals);
    staged.block_offset = upload.blocks.size();
    staged.block_count = level.terminal_blocks.size();
    for (const auto& block : level.terminal_blocks) {
      upload.blocks.push_back({
          .count = block.unknown_ordinals.size(),
          .unknown_offset = Append<std::size_t>(upload.indices, block.unknown_ordinals),
          .factor_offset = Append<double>(upload.coefficients, block.cholesky_lower),
      });
    }
    upload.levels.push_back(staged);
  }
  return upload;
}
// The only host iteration decision uses device status. No vector, reduction,
// correction or solver arithmetic is evaluated on the host.
absl::Status Execute(
    const PressureUpload3D& upload,
    PressureDevice3D device,
    std::vector<double>& output,
    PressureState3D& state,
    CudaOperatorMeasurement& measurement,
    cuda_internal::CudaOperatorRuntime& runtime
) {
  auto status = runtime.Initialize();
  if (!status.ok()) {
    return status;
  }
  cuda_internal::PressureCell3D* cells = nullptr;
  cuda_internal::PressureLink3D* links = nullptr;
  cuda_internal::PressureLevel3D* levels = nullptr;
  cuda_internal::PressureBlock3D* blocks = nullptr;
  std::size_t* indices = nullptr;
  double* coefficients = nullptr;
  status = runtime.Upload(std::span(upload.cells), cells);
  if (!status.ok()) {
    return status;
  }
  status = runtime.Upload(std::span(upload.links), links);
  if (!status.ok()) {
    return status;
  }
  status = runtime.Upload(std::span(upload.levels), levels);
  if (!status.ok()) {
    return status;
  }
  status = runtime.Upload(std::span(upload.blocks), blocks);
  if (!status.ok()) {
    return status;
  }
  status = runtime.Upload(std::span(upload.indices), indices);
  if (!status.ok()) {
    return status;
  }
  status = runtime.Upload(std::span(upload.coefficients), coefficients);
  if (!status.ok()) {
    return status;
  }
  device.cells = cells;
  device.links = links;
  device.levels = levels;
  device.blocks = blocks;
  device.indices = indices;
  device.coefficients = coefficients;
  status = runtime.AllocateZeroed(6 * device.unknown_count, device.fine_scratch);
  if (!status.ok()) {
    return status;
  }
  status = runtime.AllocateZeroed(3 * device.level_unknown_count, device.coarse_scratch);
  if (!status.ok()) {
    return status;
  }
  status = runtime.AllocateZeroed(2 * device.cell_count, device.rows);
  if (!status.ok()) {
    return status;
  }
  status = runtime.AllocateZeroed(output.size(), device.output);
  if (!status.ok()) {
    return status;
  }
  status = runtime.AllocateZeroed(1, device.state);
  if (!status.ok()) {
    return status;
  }
  status = runtime.RecordKernelStart();
  if (!status.ok()) {
    return status;
  }
  for (std::size_t launch = 0; launch <= device.max_iterations; ++launch) {
    status = cuda_internal::CheckCuda(
        static_cast<cudaError_t>(cuda_internal::LaunchMacPressure3D(device, launch == 0, runtime.stream())),
        "CUDA 3D pressure launch"
    );
    if (!status.ok()) {
      return status;
    }
    status = cuda_internal::CheckCuda(
        cudaMemcpyAsync(&state, device.state, sizeof(state), cudaMemcpyDeviceToHost, runtime.stream()),
        "CUDA 3D pressure status copy"
    );
    if (!status.ok()) {
      return status;
    }
    status = cuda_internal::CheckCuda(cudaStreamSynchronize(runtime.stream()), "CUDA 3D pressure iteration sync");
    if (!status.ok()) {
      return status;
    }
    if (state.error != cuda_internal::PressureError3D::kNone || state.converged) {
      break;
    }
  }
  status = runtime.SynchronizeAndMeasureKernel();
  if (!status.ok()) {
    return status;
  }
  const auto measured = runtime.ValidateKernelMeasurement(state.compiled_architecture);
  if (!measured.ok()) {
    return measured.status();
  }
  switch (state.error) {
    case cuda_internal::PressureError3D::kNone:
      break;
    case cuda_internal::PressureError3D::kRange:
      return absl::OutOfRangeError("CUDA 3D pressure arithmetic is unrepresentable");
    case cuda_internal::PressureError3D::kIdentity:
      return absl::InternalError("CUDA 3D pressure residual disagrees with physical defect");
    case cuda_internal::PressureError3D::kCurvature:
      return absl::AbortedError("CUDA 3D pressure lost positive curvature");
    case cuda_internal::PressureError3D::kExhausted:
      return absl::AbortedError("CUDA 3D pressure iteration budget exhausted before all true stopping gates passed");
  }
  if (!state.converged) {
    return absl::InternalError("CUDA 3D pressure missing termination status");
  }
  status = runtime.Download(device.output, std::span(output));
  if (!status.ok()) {
    return status;
  }
  measurement = *measured;
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<CudaMacPressureProjection3D>
CudaProjectMacPressure3D(const MacPressureInputs3D& inputs, MacPressureSolverOptions3D options) {
  if (options.max_iterations == 0 || options.max_iterations > 256) {
    return absl::InvalidArgumentError("pressure iteration budget must be in [1,256]");
  }
  const auto system = pressure_internal::PrepareMacPressureSystem3D(inputs);
  if (!system.ok()) {
    return system.status();
  }
  const auto hierarchy = pressure_internal::BuildMacPressureHierarchy3D(*system);
  if (!hierarchy.ok()) {
    return hierarchy.status();
  }
  const auto criteria = pressure_internal::MacPressureStoppingCriteria(*system);
  if (!criteria.ok()) {
    return criteria.status();
  }
  const auto initial = pressure_internal::InitialMacPressure3D(*system);
  if (!initial.ok()) {
    return initial.status();
  }
  const auto upload = PrepareUpload(*system, *hierarchy, *initial);
  PressureDevice3D device{};
  device.cell_count = upload.cells.size();
  device.face_count = upload.links.size();
  device.level_count = upload.levels.size();
  device.unknown_count = hierarchy->unknown_to_cell.size();
  device.level_unknown_count = upload.level_unknown_count;
  device.unknown_map_offset = upload.unknown_map_offset;
  device.max_iterations = options.max_iterations;
  device.integrated_limit = criteria->integrated_residual_limit;
  device.local_limit = criteria->volume_scaled_limit;
  std::vector<double> output(4 * device.cell_count + device.face_count);
  PressureState3D state{};
  CudaMacPressureProjection3D result;
  absl::Status cleanup_status;
  absl::Status execution_status;
  {
    cuda_internal::CudaOperatorRuntime runtime(cleanup_status);
    execution_status = Execute(upload, device, output, state, result.measurement, runtime);
  }
  execution_status.Update(cleanup_status);
  if (!execution_status.ok()) {
    return execution_status;
  }
  auto& projection = result.projection;
  const auto count = device.cell_count;
  projection.cell_pressures.assign(output.begin() + count, output.begin() + 2 * count);
  projection.true_residuals.assign(output.begin() + 2 * count, output.begin() + 3 * count);
  projection.cell_constraint_defects.assign(output.begin() + 3 * count, output.begin() + 4 * count);
  projection.corrected_face_velocities.assign(output.begin() + 4 * count, output.end());
  projection.integrated_rhs.assign(system->integrated_rhs().begin(), system->integrated_rhs().end());
  projection.gauge_cell_ordinals.assign(system->gauge_cell_ordinals().begin(), system->gauge_cell_ordinals().end());
  projection.diagnostics = {
      .iteration_count = state.iterations,
      .multigrid_level_count = hierarchy->levels.size(),
      .true_residual_l2 = state.residual_l2,
      .max_volume_scaled_residual = state.maximum_scaled_residual,
      .max_cell_constraint_defect = state.maximum_defect,
  };
  return result;
}
}  // namespace fluidity
