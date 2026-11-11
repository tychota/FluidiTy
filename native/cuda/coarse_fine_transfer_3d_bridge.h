#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "fluidity/base/coarse_fine_transfer_math_3d.h"

namespace fluidity::cuda_internal {

inline constexpr std::size_t kMaxCoarseFineCudaQueries3D = 1024;

struct CoarseFineCudaResult3D {
  int compiled_architecture;  // Executed image's __CUDA_ARCH__.
  internal::CoarseFineCoreResult3D stencil;
};

// Enqueues one stencil evaluation per query. The device input and result arrays must each
// have query_count disjoint entries and stay live through stream completion. A zero count is a
// no-op. Returns cudaErrorInvalidValue for a nonzero count above capacity or null pointers.
// Each invalid geometry or position publishes an error and zero rows for its own query.
cudaError_t LaunchCoarseFineStencilCuda3D(
    const internal::CoarseFineCoreQuery3D* queries,
    CoarseFineCudaResult3D* results,
    std::size_t query_count,
    cudaStream_t stream
);

}  // namespace fluidity::cuda_internal
