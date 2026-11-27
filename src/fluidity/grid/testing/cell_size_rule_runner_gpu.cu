// The GPU runner of cell_size_rule.gpu.test.cc: the cell size rule over every case, one thread per case.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-size-rule>

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/cell_size_rule.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/cell_size_rule_runner_gpu.h"

namespace fluidity::grid {
namespace {

// results[i] = ComputeCellSizeRuleResult(cases[i], widths, settings, volume) for i below count.
__global__ void ComputeEveryCellSizeRuleResult(
    const CellSizeRuleCase* cases,
    int count,
    CellWidthsBySizeLevel widths,
    CellSizeSettings settings,
    RefinementFactors volume,
    CellSizeRuleResult* results
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < static_cast<std::size_t>(count); index += stride.item_step) {
    results[index] = ComputeCellSizeRuleResult(cases[index], widths, settings, volume);
  }
}

// *table = ComputeCellSizeFactorTable(), written by the thread of item 0.
__global__ void WriteCellSizeFactorTable(CellSizeFactorTable* table) {
  const GridStride stride = ComputeGridStride();
  if (stride.first_item == 0) {
    *table = ComputeCellSizeFactorTable();
  }
}

}  // namespace

RunOnCallStatus ComputeCellSizeRuleOnGpu(
    const CellSizeRuleCase* cases,
    int count,
    const CellWidthsBySizeLevel& widths,
    CellSizeSettings settings,
    const RefinementFactors& volume,
    CellSizeRuleResult* results,
    CellSizeFactorTable* table
) {
  // Steps: 1 the layers  2 the cases  3 the GPU functions  4 the results back
  // Step 1: each layer that holds data goes to GPU memory of its own; the GPU copy of the volume points there.
  const int layer_count = volume.layer_count < kMaxSizeLevels ? volume.layer_count : kMaxSizeLevels;
  std::array<GpuArray<std::int8_t>, kMaxSizeLevels> gpu_layers;
  RefinementFactors gpu_volume = volume;
  for (int layer = 0; layer < kMaxSizeLevels; ++layer) {
    gpu_volume.layer_codes[layer] = nullptr;
  }
  for (int layer = 0; layer < layer_count; ++layer) {
    if (volume.layer_codes[layer] == nullptr) {
      continue;
    }
    const auto unsigned_layer = static_cast<unsigned>(layer);
    const std::uint64_t entry_count = ComputeRefinementLayerExtent(volume.count_x, unsigned_layer)
        * ComputeRefinementLayerExtent(volume.count_y, unsigned_layer)
        * ComputeRefinementLayerExtent(volume.count_z, unsigned_layer);
    GpuArray<std::int8_t>& gpu_layer = gpu_layers.at(static_cast<std::size_t>(layer));
    if (const RunOnCallStatus status = gpu_layer.Build(entry_count); status != RunOnCallStatus::kDone) {
      return status;
    }
    const std::size_t layer_bytes = sizeof(std::int8_t) * entry_count;
    if (const cudaError_t error =
            cudaMemcpy(gpu_layer.values(), volume.layer_codes[layer], layer_bytes, cudaMemcpyHostToDevice);
        error != cudaSuccess) {
      return ToRunOnCallStatus(error);
    }
    gpu_volume.layer_codes[layer] = gpu_layer.values();
  }

  // Step 2: the cases, the results and the table.
  const auto case_count = static_cast<std::size_t>(count);
  GpuArray<CellSizeRuleCase> gpu_cases;
  GpuArray<CellSizeRuleResult> gpu_results;
  GpuArray<CellSizeFactorTable> gpu_table;
  if (const RunOnCallStatus status = gpu_cases.Build(case_count); status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_results.Build(case_count); status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_table.Build(1); status != RunOnCallStatus::kDone) {
    return status;
  }
  const std::size_t case_bytes = sizeof(CellSizeRuleCase) * case_count;
  if (const cudaError_t error = cudaMemcpy(gpu_cases.values(), cases, case_bytes, cudaMemcpyHostToDevice);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }

  // Step 3: one thread per case, then one thread for the table.
  if (const RunOnCallStatus status = ApplyGpuFunction(
          ComputeEveryCellSizeRuleResult,
          ComputeLaunchSize(case_count),
          gpu_cases.values(),
          count,
          widths,
          settings,
          gpu_volume,
          gpu_results.values()
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status =
          ApplyGpuFunction(WriteCellSizeFactorTable, ComputeLaunchSize(1), gpu_table.values());
      status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 4: the copies wait for the GPU functions, so an error while they ran reports here.
  const std::size_t result_bytes = sizeof(CellSizeRuleResult) * case_count;
  if (const cudaError_t error = cudaMemcpy(results, gpu_results.values(), result_bytes, cudaMemcpyDeviceToHost);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (const cudaError_t error =
          cudaMemcpy(table, gpu_table.values(), sizeof(CellSizeFactorTable), cudaMemcpyDeviceToHost);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (const RunOnCallStatus status = gpu_cases.Free(); status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_results.Free(); status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_table.Free(); status != RunOnCallStatus::kDone) {
    return status;
  }
  for (GpuArray<std::int8_t>& gpu_layer : gpu_layers) {
    if (gpu_layer.values() == nullptr) {
      continue;
    }
    if (const RunOnCallStatus status = gpu_layer.Free(); status != RunOnCallStatus::kDone) {
      return status;
    }
  }
  return RunOnCallStatus::kDone;
}

}  // namespace fluidity::grid
