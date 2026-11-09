#pragma once
#include <cstddef>
#include <cstdint>
namespace fluidity::cuda_internal {
struct LiquidSortKey3D {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;
};
// Classic explicit-storage CCCL 3.4.2 stable signed (z,y,x) radix sort.
// Null temporary storage queries bytes; otherwise pointers/count stay fixed
// across capture/replay. Returns CUDA error code, never allocates explicitly.
int SortLiquidBins3D(
    void* temporary,
    std::size_t& temporary_bytes,
    LiquidSortKey3D* input,
    LiquidSortKey3D* output,
    std::size_t* incoming,
    std::size_t* ordered,
    std::size_t capacity,
    void* stream
);
// Writes the actual compiled architecture on the caller's explicit stream.
int LaunchLiquidArenaArchitecture3D(int* architecture, void* stream);
int LiquidArenaCcclVersion3D();
int LiquidArenaCubVersion3D();
}  // namespace fluidity::cuda_internal
