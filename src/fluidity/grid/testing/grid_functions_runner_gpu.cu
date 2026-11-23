// The GPU runners of the GPU tests of the shared grid functions: each function is asked over every query, one thread
// per query.
// Read: docs/method-paper/fluidity-method.typ <sec-run-on>, <sec-nearby-faces>

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_functions_answers.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"

namespace fluidity::grid {
namespace {

// The text of the last failure, as ReadLastRunnerFailure gives it. Each runner of a thread clears it first.
std::string& FindLastFailure() {
  static thread_local std::string last_failure;
  return last_failure;
}

// Records the failed CUDA call: its step, cudaGetErrorName and cudaGetErrorString of `error`. Returns `status`.
RunOnCallStatus RecordFailure(RunOnCallStatus status, std::string_view step, cudaError_t error) {
  std::string& last_failure = FindLastFailure();
  last_failure = std::string(step) + ": " + cudaGetErrorName(error) + " (" + cudaGetErrorString(error) + ")";
  return status;
}

// Returns the status of a call that returned `error`, and records the failure when it did not succeed.
RunOnCallStatus ToRecordedStatus(cudaError_t error, std::string_view step) {
  return error == cudaSuccess ? RunOnCallStatus::kDone : RecordFailure(ToRunOnCallStatus(error), step, error);
}

// Builds the GPU memory for `count` values. A failed cudaMalloc leaves its error as the last error, which is read here.
template <typename T>
RunOnCallStatus BuildGpuArray(GpuArray<T>& array, std::size_t count, std::string_view step) {
  const RunOnCallStatus status = array.Build(count);
  if (status == RunOnCallStatus::kDone) {
    return status;
  }
  return RecordFailure(status, step, cudaGetLastError());
}

// results[i] = ComputeResult(tables, queries[i]) for i below count.
template <typename Query, typename Result, Result (*ComputeResult)(const GridTables&, const Query&)>
__global__ void ComputeEveryResult(GridTables tables, const Query* queries, int count, Result* results) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < static_cast<std::size_t>(count); index += stride.item_step) {
    results[index] = ComputeResult(tables, queries[index]);
  }
}

// results[index] = ComputeBorderBlockResult(tables, set, queries[index]) for index below count. `set` holds GPU
// pointers: the runner copied the sorted list to the GPU.
__global__ void ComputeEveryBorderBlockResult(
    GridTables tables,
    BlockSet set,
    const BorderBlockQuery* queries,
    int count,
    BorderBlockResult* results
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < static_cast<std::size_t>(count); index += stride.item_step) {
    results[index] = ComputeBorderBlockResult(tables, set, queries[index]);
  }
}

// results[index] = ComputePieceStepResult(tables, set, queries[index]) for index below count. `set` holds GPU pointers:
// the runner copied the border blocks and the piece map to the GPU.
__global__ void ComputeEveryPieceStepResult(
    GridTables tables,
    PieceStepSet set,
    const PieceStepQuery* queries,
    int count,
    PieceStepResult* results
) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < static_cast<std::size_t>(count); index += stride.item_step) {
    results[index] = ComputePieceStepResult(tables, set, queries[index]);
  }
}

// Starts ComputeEveryResult as ApplyGpuFunction does, one thread per query, and records the error of a launch that did
// not start: ApplyGpuFunction reads cudaGetLastError and clears it, so its name would be lost.
template <typename Query, typename Result, Result (*ComputeResult)(const GridTables&, const Query&)>
RunOnCallStatus
StartOneThreadPerQuery(const GridTables& tables, const Query* gpu_queries, int count, Result* gpu_results) {
  const LaunchSize launch_size = ComputeLaunchSize(static_cast<std::size_t>(count));
  ComputeEveryResult<Query, Result, ComputeResult>
      <<<launch_size.thread_block_count, launch_size.thread_block_size>>>(tables, gpu_queries, count, gpu_results);
  return ToRecordedStatus(cudaGetLastError(), "start of the GPU function");
}

// Copies the queries to the GPU, runs ComputeEveryResult with one thread per query, and copies the results back.
template <typename Query, typename Result, Result (*ComputeResult)(const GridTables&, const Query&)>
RunOnCallStatus RunOneThreadPerQuery(const GridTables& tables, const Query* queries, int count, Result* results) {
  FindLastFailure().clear();
  // Steps: 1 the GPU memory  2 the queries  3 the GPU function  4 the results back
  // Step 1: memory for the queries and the results.
  const auto query_count = static_cast<std::size_t>(count);
  GpuArray<Query> gpu_queries;
  GpuArray<Result> gpu_results;
  if (const RunOnCallStatus status = BuildGpuArray(gpu_queries, query_count, "allocation of the queries");
      status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = BuildGpuArray(gpu_results, query_count, "allocation of the results");
      status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 2: the queries go to the GPU.
  const std::size_t query_bytes = sizeof(Query) * query_count;
  if (const RunOnCallStatus status = ToRecordedStatus(
          cudaMemcpy(gpu_queries.values(), queries, query_bytes, cudaMemcpyHostToDevice),
          "copy of the queries to the GPU"
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 3: one thread per query.
  if (const RunOnCallStatus status = StartOneThreadPerQuery<Query, Result, ComputeResult>(
          tables,
          gpu_queries.values(),
          count,
          gpu_results.values()
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 4: the copy waits for the GPU function, so an error while it ran reports here.
  const std::size_t result_bytes = sizeof(Result) * query_count;
  if (const RunOnCallStatus status = ToRecordedStatus(
          cudaMemcpy(results, gpu_results.values(), result_bytes, cudaMemcpyDeviceToHost),
          "copy of the results to the CPU, which also waits for the GPU function"
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_queries.Free(); status != RunOnCallStatus::kDone) {
    return RecordFailure(status, "freeing the queries", cudaGetLastError());
  }
  const RunOnCallStatus status = gpu_results.Free();
  return status == RunOnCallStatus::kDone ? status : RecordFailure(status, "freeing the results", cudaGetLastError());
}

// Reads the attributes of ComputeEveryResult for one runner and adds them to `functions`.
template <typename Query, typename Result, Result (*ComputeResult)(const GridTables&, const Query&)>
RunOnCallStatus AddFunctionInfo(std::string name, std::vector<RunnerFunctionInfo>& functions) {
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(&attributes, ComputeEveryResult<Query, Result, ComputeResult>);
  if (error != cudaSuccess) {
    return RecordFailure(ToRunOnCallStatus(error), "attributes of " + name, error);
  }
  functions.push_back({
      .name = std::move(name),
      .local_bytes = static_cast<std::uint64_t>(attributes.localSizeBytes),
      .register_count = attributes.numRegs,
      .most_threads_per_block = attributes.maxThreadsPerBlock,
  });
  return RunOnCallStatus::kDone;
}

// Reads the attributes of ComputeEveryPieceStepResult and adds them to `functions`.
RunOnCallStatus AddPieceStepFunctionInfo(std::string name, std::vector<RunnerFunctionInfo>& functions) {
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(&attributes, ComputeEveryPieceStepResult);
  if (error != cudaSuccess) {
    return RecordFailure(ToRunOnCallStatus(error), "attributes of " + name, error);
  }
  functions.push_back({
      .name = std::move(name),
      .local_bytes = static_cast<std::uint64_t>(attributes.localSizeBytes),
      .register_count = attributes.numRegs,
      .most_threads_per_block = attributes.maxThreadsPerBlock,
  });
  return RunOnCallStatus::kDone;
}

// Reads the attributes of ComputeEveryBorderBlockResult and adds them to `functions`.
RunOnCallStatus AddBorderBlockFunctionInfo(std::string name, std::vector<RunnerFunctionInfo>& functions) {
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(&attributes, ComputeEveryBorderBlockResult);
  if (error != cudaSuccess) {
    return RecordFailure(ToRunOnCallStatus(error), "attributes of " + name, error);
  }
  functions.push_back({
      .name = std::move(name),
      .local_bytes = static_cast<std::uint64_t>(attributes.localSizeBytes),
      .register_count = attributes.numRegs,
      .most_threads_per_block = attributes.maxThreadsPerBlock,
  });
  return RunOnCallStatus::kDone;
}

}  // namespace

RunOnCallStatus
FindCellsAtOnGpu(const GridTables& tables, const FindCellAtQuery* queries, int count, FoundCells* results) {
  return RunOneThreadPerQuery<FindCellAtQuery, FoundCells, FindCellsAtPosition>(tables, queries, count, results);
}

RunOnCallStatus ComputeNearbyFaceResultsOnGpu(
    const GridTables& tables,
    const NearbyFaceQuery* queries,
    int count,
    NearbyFaceResult* results
) {
  return RunOneThreadPerQuery<NearbyFaceQuery, NearbyFaceResult, ComputeNearbyFaceResult>(
      tables,
      queries,
      count,
      results
  );
}

RunOnCallStatus ComputeFaceWeightResultsOnGpu(
    const GridTables& tables,
    const FaceWeightQuery* queries,
    int count,
    FaceWeightResult* results
) {
  return RunOneThreadPerQuery<FaceWeightQuery, FaceWeightResult, ComputeFaceWeightResult>(
      tables,
      queries,
      count,
      results
  );
}

RunOnCallStatus
FindCellCornersOnGpu(const GridTables& tables, const CellPlaceIndex* queries, int count, CellCorners* results) {
  return RunOneThreadPerQuery<CellPlaceIndex, CellCorners, FindCellCornersOf>(tables, queries, count, results);
}

RunOnCallStatus
FindFaceCornersOnGpu(const GridTables& tables, const FacePlaceIndex* queries, int count, FaceCorners* results) {
  return RunOneThreadPerQuery<FacePlaceIndex, FaceCorners, FindFaceCornersOf>(tables, queries, count, results);
}

RunOnCallStatus
ComputeCornerPositionsOnGpu(const GridTables& tables, const CornerPlaceIndex* queries, int count, Vec3* results) {
  return RunOneThreadPerQuery<CornerPlaceIndex, Vec3, ComputeCornerPositionOf>(tables, queries, count, results);
}

RunOnCallStatus
FindCellsAroundEdgesOnGpu(const GridTables& tables, const EdgeQuery* queries, int count, CellsAroundEdge* results) {
  return RunOneThreadPerQuery<EdgeQuery, CellsAroundEdge, FindCellsAroundEdgeOf>(tables, queries, count, results);
}

RunOnCallStatus
FindTouchingCellsOnGpu(const GridTables& tables, const CellPlaceIndex* queries, int count, TouchingCells* results) {
  return RunOneThreadPerQuery<CellPlaceIndex, TouchingCells, FindTouchingCellsOf>(tables, queries, count, results);
}

RunOnCallStatus
FindFacePiecesOnGpu(const GridTables& tables, const CellPlaceIndex* queries, int count, CellFacePieces* results) {
  return RunOneThreadPerQuery<CellPlaceIndex, CellFacePieces, FindFacePiecesOf>(tables, queries, count, results);
}

RunOnCallStatus
FindCellsNearFacesOnGpu(const GridTables& tables, const FacePlaceIndex* queries, int count, CellsNearFace* results) {
  return RunOneThreadPerQuery<FacePlaceIndex, CellsNearFace, FindCellsNearFaceOf>(tables, queries, count, results);
}

RunOnCallStatus
WalkFaceSearchesOnGpu(const GridTables& tables, const FaceSearchQuery* queries, int count, FaceSearchWalk* results) {
  return RunOneThreadPerQuery<FaceSearchQuery, FaceSearchWalk, FindFaceSearchWalk>(tables, queries, count, results);
}

RunOnCallStatus
ComputeFormulasOnGpu(const GridTables& tables, const FormulaQuery* queries, int count, FormulaResult* results) {
  return RunOneThreadPerQuery<FormulaQuery, FormulaResult, ComputeFormulaResult>(tables, queries, count, results);
}

RunOnCallStatus ComputeCellPlaceFactsOnGpu(
    const GridTables& tables,
    const CellPlaceIndex* queries,
    int count,
    CellPlaceFacts* results
) {
  return RunOneThreadPerQuery<CellPlaceIndex, CellPlaceFacts, ComputeCellPlaceFacts>(tables, queries, count, results);
}

RunOnCallStatus ComputeFacePlaceFactsOnGpu(
    const GridTables& tables,
    const FacePlaceIndex* queries,
    int count,
    FacePlaceFacts* results
) {
  return RunOneThreadPerQuery<FacePlaceIndex, FacePlaceFacts, ComputeFacePlaceFacts>(tables, queries, count, results);
}

RunOnCallStatus ComputeReadMarksResultsOnGpu(
    const GridTables& tables,
    const ReadMarksQuery* queries,
    int count,
    ReadMarksResult* results
) {
  return RunOneThreadPerQuery<ReadMarksQuery, ReadMarksResult, ComputeReadMarksResult>(tables, queries, count, results);
}

RunOnCallStatus ComputePieceStepResultsOnGpu(
    const GridTables& tables,
    const PieceStepSet& set,
    const PieceStepQuery* queries,
    int count,
    PieceStepResult* results
) {
  FindLastFailure().clear();
  // Steps: 1 the GPU memory  2 the lists and the queries  3 the GPU function  4 the results back
  // Step 1: one array for the border blocks followed by the piece map (none when both are empty), the queries and the
  // results.
  const auto query_count = static_cast<std::size_t>(count);
  const std::size_t list_count = std::size_t{set.border_block_count} + set.loaded_block_count;
  GpuArray<CellBlockIndex> gpu_lists;
  GpuArray<PieceStepQuery> gpu_queries;
  GpuArray<PieceStepResult> gpu_results;
  if (list_count > 0) {
    if (const RunOnCallStatus status = BuildGpuArray(gpu_lists, list_count, "allocation of the lists");
        status != RunOnCallStatus::kDone) {
      return status;
    }
  }
  if (const RunOnCallStatus status = BuildGpuArray(gpu_queries, query_count, "allocation of the queries");
      status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = BuildGpuArray(gpu_results, query_count, "allocation of the results");
      status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 2: the lists and the queries go to the GPU; the set refers to the GPU copy of the lists.
  PieceStepSet gpu_set = set;
  if (list_count > 0) {
    gpu_set.border_blocks = gpu_lists.values();
    gpu_set.map = gpu_lists.values() + set.border_block_count;
    if (set.border_block_count > 0) {
      if (const RunOnCallStatus status = ToRecordedStatus(
              cudaMemcpy(
                  gpu_lists.values(),
                  set.border_blocks,
                  sizeof(CellBlockIndex) * set.border_block_count,
                  cudaMemcpyHostToDevice
              ),
              "copy of the border blocks to the GPU"
          );
          status != RunOnCallStatus::kDone) {
        return status;
      }
    }
    if (set.loaded_block_count > 0) {
      if (const RunOnCallStatus status = ToRecordedStatus(
              cudaMemcpy(
                  gpu_lists.values() + set.border_block_count,
                  set.map,
                  sizeof(CellBlockIndex) * set.loaded_block_count,
                  cudaMemcpyHostToDevice
              ),
              "copy of the piece map to the GPU"
          );
          status != RunOnCallStatus::kDone) {
        return status;
      }
    }
  }
  if (const RunOnCallStatus status = ToRecordedStatus(
          cudaMemcpy(gpu_queries.values(), queries, sizeof(PieceStepQuery) * query_count, cudaMemcpyHostToDevice),
          "copy of the queries to the GPU"
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 3: one thread per query.
  const LaunchSize launch_size = ComputeLaunchSize(query_count);
  ComputeEveryPieceStepResult<<<launch_size.thread_block_count, launch_size.thread_block_size>>>(
      tables,
      gpu_set,
      gpu_queries.values(),
      count,
      gpu_results.values()
  );
  if (const RunOnCallStatus status = ToRecordedStatus(cudaGetLastError(), "start of the GPU function");
      status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 4: the copy waits for the GPU function, so an error while it ran reports here.
  if (const RunOnCallStatus status = ToRecordedStatus(
          cudaMemcpy(results, gpu_results.values(), sizeof(PieceStepResult) * query_count, cudaMemcpyDeviceToHost),
          "copy of the results to the CPU, which also waits for the GPU function"
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_lists.values() == nullptr ? RunOnCallStatus::kDone : gpu_lists.Free();
      status != RunOnCallStatus::kDone) {
    return RecordFailure(status, "freeing the lists", cudaGetLastError());
  }
  if (const RunOnCallStatus status = gpu_queries.Free(); status != RunOnCallStatus::kDone) {
    return RecordFailure(status, "freeing the queries", cudaGetLastError());
  }
  const RunOnCallStatus status = gpu_results.Free();
  return status == RunOnCallStatus::kDone ? status : RecordFailure(status, "freeing the results", cudaGetLastError());
}

RunOnCallStatus ComputeBorderBlockResultsOnGpu(
    const GridTables& tables,
    CellBlockRange range,
    std::span<const CellBlockIndex> sorted_blocks,
    const BorderBlockQuery* queries,
    int count,
    BorderBlockResult* results
) {
  FindLastFailure().clear();
  // Steps: 1 the GPU memory  2 the set and the queries  3 the GPU function  4 the results back
  // Step 1: memory for the sorted list (none when it is empty), the queries and the results.
  const auto query_count = static_cast<std::size_t>(count);
  GpuArray<CellBlockIndex> gpu_sorted_blocks;
  GpuArray<BorderBlockQuery> gpu_queries;
  GpuArray<BorderBlockResult> gpu_results;
  if (!sorted_blocks.empty()) {
    if (const RunOnCallStatus status = BuildGpuArray(gpu_sorted_blocks, sorted_blocks.size(), "allocation of the set");
        status != RunOnCallStatus::kDone) {
      return status;
    }
  }
  if (const RunOnCallStatus status = BuildGpuArray(gpu_queries, query_count, "allocation of the queries");
      status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = BuildGpuArray(gpu_results, query_count, "allocation of the results");
      status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 2: the sorted list and the queries go to the GPU; the set refers to the GPU copy of the list.
  if (!sorted_blocks.empty()) {
    if (const RunOnCallStatus status = ToRecordedStatus(
            cudaMemcpy(
                gpu_sorted_blocks.values(),
                sorted_blocks.data(),
                sizeof(CellBlockIndex) * sorted_blocks.size(),
                cudaMemcpyHostToDevice
            ),
            "copy of the set to the GPU"
        );
        status != RunOnCallStatus::kDone) {
      return status;
    }
  }
  if (const RunOnCallStatus status = ToRecordedStatus(
          cudaMemcpy(gpu_queries.values(), queries, sizeof(BorderBlockQuery) * query_count, cudaMemcpyHostToDevice),
          "copy of the queries to the GPU"
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  const BlockSet gpu_set = {
      .range = range,
      .sorted_blocks = gpu_sorted_blocks.values(),
      .sorted_count = static_cast<std::uint32_t>(sorted_blocks.size()),
  };

  // Step 3: one thread per query.
  const LaunchSize launch_size = ComputeLaunchSize(query_count);
  ComputeEveryBorderBlockResult<<<launch_size.thread_block_count, launch_size.thread_block_size>>>(
      tables,
      gpu_set,
      gpu_queries.values(),
      count,
      gpu_results.values()
  );
  if (const RunOnCallStatus status = ToRecordedStatus(cudaGetLastError(), "start of the GPU function");
      status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 4: the copy waits for the GPU function, so an error while it ran reports here.
  if (const RunOnCallStatus status = ToRecordedStatus(
          cudaMemcpy(results, gpu_results.values(), sizeof(BorderBlockResult) * query_count, cudaMemcpyDeviceToHost),
          "copy of the results to the CPU, which also waits for the GPU function"
      );
      status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status =
          gpu_sorted_blocks.values() == nullptr ? RunOnCallStatus::kDone : gpu_sorted_blocks.Free();
      status != RunOnCallStatus::kDone) {
    return RecordFailure(status, "freeing the set", cudaGetLastError());
  }
  if (const RunOnCallStatus status = gpu_queries.Free(); status != RunOnCallStatus::kDone) {
    return RecordFailure(status, "freeing the queries", cudaGetLastError());
  }
  const RunOnCallStatus status = gpu_results.Free();
  return status == RunOnCallStatus::kDone ? status : RecordFailure(status, "freeing the results", cudaGetLastError());
}

RunOnCallStatus ReadRunnerFunctionInfo(std::vector<RunnerFunctionInfo>* functions) {
  FindLastFailure().clear();
  std::vector<RunnerFunctionInfo> read;
  const std::array<RunOnCallStatus, 17> statuses = {
      AddFunctionInfo<FindCellAtQuery, FoundCells, FindCellsAtPosition>("FindCellsAtOnGpu", read),
      AddFunctionInfo<NearbyFaceQuery, NearbyFaceResult, ComputeNearbyFaceResult>(
          "ComputeNearbyFaceResultsOnGpu",
          read
      ),
      AddFunctionInfo<FaceWeightQuery, FaceWeightResult, ComputeFaceWeightResult>(
          "ComputeFaceWeightResultsOnGpu",
          read
      ),
      AddFunctionInfo<CellPlaceIndex, CellCorners, FindCellCornersOf>("FindCellCornersOnGpu", read),
      AddFunctionInfo<FacePlaceIndex, FaceCorners, FindFaceCornersOf>("FindFaceCornersOnGpu", read),
      AddFunctionInfo<CornerPlaceIndex, Vec3, ComputeCornerPositionOf>("ComputeCornerPositionsOnGpu", read),
      AddFunctionInfo<EdgeQuery, CellsAroundEdge, FindCellsAroundEdgeOf>("FindCellsAroundEdgesOnGpu", read),
      AddFunctionInfo<CellPlaceIndex, TouchingCells, FindTouchingCellsOf>("FindTouchingCellsOnGpu", read),
      AddFunctionInfo<CellPlaceIndex, CellFacePieces, FindFacePiecesOf>("FindFacePiecesOnGpu", read),
      AddFunctionInfo<FacePlaceIndex, CellsNearFace, FindCellsNearFaceOf>("FindCellsNearFacesOnGpu", read),
      AddFunctionInfo<FaceSearchQuery, FaceSearchWalk, FindFaceSearchWalk>("WalkFaceSearchesOnGpu", read),
      AddFunctionInfo<FormulaQuery, FormulaResult, ComputeFormulaResult>("ComputeFormulasOnGpu", read),
      AddFunctionInfo<CellPlaceIndex, CellPlaceFacts, ComputeCellPlaceFacts>("ComputeCellPlaceFactsOnGpu", read),
      AddFunctionInfo<FacePlaceIndex, FacePlaceFacts, ComputeFacePlaceFacts>("ComputeFacePlaceFactsOnGpu", read),
      AddBorderBlockFunctionInfo("ComputeBorderBlockResultsOnGpu", read),
      AddFunctionInfo<ReadMarksQuery, ReadMarksResult, ComputeReadMarksResult>("ComputeReadMarksResultsOnGpu", read),
      AddPieceStepFunctionInfo("ComputePieceStepResultsOnGpu", read),
  };
  for (const RunOnCallStatus status : statuses) {
    if (status != RunOnCallStatus::kDone) {
      return status;
    }
  }
  *functions = std::move(read);
  return RunOnCallStatus::kDone;
}

std::string ReadLastRunnerFailure() {
  return FindLastFailure();
}

}  // namespace fluidity::grid
