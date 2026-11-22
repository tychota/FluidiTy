// The GPU runners of the GPU tests of the shared grid functions: each function is asked over every query, one thread
// per query.
// Read: docs/method-paper/fluidity-method.typ <sec-run-on>, <sec-nearby-faces>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "fluidity/base/gpu_launch.cuh"
#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_functions_runner_gpu.h"

namespace fluidity::grid {
namespace {

// results[i] = ComputeResult(tables, queries[i]) for i below count.
template <typename Query, typename Result, Result (*ComputeResult)(const GridTables&, const Query&)>
__global__ void ComputeEveryResult(GridTables tables, const Query* queries, int count, Result* results) {
  const GridStride stride = ComputeGridStride();
  for (std::size_t index = stride.first_item; index < static_cast<std::size_t>(count); index += stride.item_step) {
    results[index] = ComputeResult(tables, queries[index]);
  }
}

// Copies the queries to the GPU, runs ComputeEveryResult with one thread per query, and copies the results back.
template <typename Query, typename Result, Result (*ComputeResult)(const GridTables&, const Query&)>
RunOnCallStatus RunOneThreadPerQuery(const GridTables& tables, const Query* queries, int count, Result* results) {
  // Steps: 1 the GPU memory  2 the queries  3 the GPU function  4 the results back
  // Step 1: memory for the queries and the results.
  const auto query_count = static_cast<std::size_t>(count);
  GpuArray<Query> gpu_queries;
  GpuArray<Result> gpu_results;
  if (const RunOnCallStatus status = gpu_queries.Build(query_count); status != RunOnCallStatus::kDone) {
    return status;
  }
  if (const RunOnCallStatus status = gpu_results.Build(query_count); status != RunOnCallStatus::kDone) {
    return status;
  }

  // Step 2: the queries go to the GPU.
  const std::size_t query_bytes = sizeof(Query) * query_count;
  if (const cudaError_t error = cudaMemcpy(gpu_queries.values(), queries, query_bytes, cudaMemcpyHostToDevice);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }

  // Step 3: one thread per query.
  if (const RunOnCallStatus status = ApplyGpuFunction(
          ComputeEveryResult<Query, Result, ComputeResult>,
          ComputeLaunchSize(query_count),
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
  if (const cudaError_t error = cudaMemcpy(results, gpu_results.values(), result_bytes, cudaMemcpyDeviceToHost);
      error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  if (const RunOnCallStatus status = gpu_queries.Free(); status != RunOnCallStatus::kDone) {
    return status;
  }
  return gpu_results.Free();
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
  return RunOneThreadPerQuery<FaceSearchQuery, FaceSearchWalk, WalkFaceSearch>(tables, queries, count, results);
}

RunOnCallStatus ComputeWeightsRunnerStackOnGpu(std::uint64_t* bytes_per_thread) {
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(
      &attributes,
      ComputeEveryResult<NearbyFaceQuery, NearbyFaceResult, ComputeNearbyFaceResult>
  );
  if (error != cudaSuccess) {
    return ToRunOnCallStatus(error);
  }
  *bytes_per_thread = static_cast<std::uint64_t>(attributes.localSizeBytes);
  return RunOnCallStatus::kDone;
}

}  // namespace fluidity::grid
