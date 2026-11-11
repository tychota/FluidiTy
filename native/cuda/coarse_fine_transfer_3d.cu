#include "coarse_fine_transfer_3d_bridge.h"

namespace fluidity::cuda_internal {
namespace {

__global__ void EvaluateCoarseFineStencils(
    const internal::CoarseFineCoreQuery3D* queries,
    CoarseFineCudaResult3D* results,
    std::size_t query_count
) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= query_count) {
    return;
  }
  auto& result = results[index];
#if defined(__CUDA_ARCH__)
  result.compiled_architecture = __CUDA_ARCH__;
#else
  result.compiled_architecture = 0;
#endif
  internal::EvaluateCoarseFineCore3D(queries[index], result.stencil);
}

}  // namespace

cudaError_t LaunchCoarseFineStencilCuda3D(
    const internal::CoarseFineCoreQuery3D* queries,
    CoarseFineCudaResult3D* results,
    std::size_t query_count,
    cudaStream_t stream
) {
  if (query_count == 0) {
    return cudaSuccess;
  }
  if (query_count > kMaxCoarseFineCudaQueries3D || queries == nullptr || results == nullptr) {
    return cudaErrorInvalidValue;
  }
  constexpr unsigned int threads_per_block = 64;
  const auto block_count = static_cast<unsigned int>((query_count + threads_per_block - 1) / threads_per_block);
  EvaluateCoarseFineStencils<<<block_count, threads_per_block, 0, stream>>>(queries, results, query_count);
  return cudaGetLastError();
}

}  // namespace fluidity::cuda_internal
