#include <cuda_runtime.h>
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4530)  // Thrust's system_error formatter, not production exception handling.
#endif
#include <cub/device/device_radix_sort.cuh>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#include <cub/version.cuh>
#include <cuda/std/tuple>

#include "liquid_arena_3d_bridge.h"
namespace fluidity::cuda_internal {
namespace {
static_assert(CCCL_VERSION == 3004002);
static_assert(CUB_VERSION == 300402);
struct SignedBinDecomposer {
  __host__ __device__ auto operator()(LiquidSortKey3D& key) const {
    return cuda::std::tie(key.z, key.y, key.x);
  }
};
__global__ void WriteLiquidArenaArchitecture3D(int* architecture) {
#if defined(__CUDA_ARCH__)
  *architecture = __CUDA_ARCH__;
#endif
}
}  // namespace
int SortLiquidBins3D(
    void* temporary,
    std::size_t& temporary_bytes,
    LiquidSortKey3D* input,
    LiquidSortKey3D* output,
    std::size_t* incoming,
    std::size_t* ordered,
    std::size_t capacity,
    void* stream
) {
  return static_cast<int>(cub::DeviceRadixSort::SortPairs(
      temporary,
      temporary_bytes,
      input,
      output,
      incoming,
      ordered,
      capacity,
      SignedBinDecomposer{},
      0,
      96,
      static_cast<cudaStream_t>(stream)
  ));
}
int LaunchLiquidArenaArchitecture3D(int* architecture, void* stream) {
  WriteLiquidArenaArchitecture3D<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(architecture);
  return static_cast<int>(cudaPeekAtLastError());
}
int LiquidArenaCcclVersion3D() {
  return CCCL_VERSION;
}
int LiquidArenaCubVersion3D() {
  return CUB_VERSION;
}
}  // namespace fluidity::cuda_internal
