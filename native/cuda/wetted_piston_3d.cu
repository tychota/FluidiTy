#include <cuda_runtime.h>

#include "wetted_piston_3d_bridge.h"

namespace fluidity::cuda_internal {
namespace {
namespace wet = wetted_piston_internal;

// Each patch integrates its own bands and sorts its own interval lists. Spread
// these independent, divergent workloads across SMs with the same scratch size.
constexpr unsigned int kPatchThreadsPerBlock = 4;

__global__ void WettedPistonSections3DKernel(
    wet::Field field,
    const Vec3* centers,
    wet::Disk* disks,
    std::size_t count,
    int* architecture
) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
#if defined(__CUDA_ARCH__)
  if (index == 0) {
    *architecture = __CUDA_ARCH__;
  }
#endif
  disks[index] = wet::Section(field, centers[index]);
}

__global__ void WettedPistonPatches3DKernel(
    wet::Field field,
    const wet::Disk* disks,
    std::size_t source_count,
    const wet::PatchIndex* indices,
    wet::PatchMeasure* measures,
    std::size_t patch_count,
    wet::Interval* intervals,
    std::size_t interval_stride,
    int* architecture
) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= patch_count) {
    return;
  }
#if defined(__CUDA_ARCH__)
  if (index == 0) {
    *architecture = __CUDA_ARCH__;
  }
#endif
  auto* inner = intervals + 2 * index * interval_stride;
  auto* outer = inner + interval_stride;
  measures[index] = wet::MeasurePatch(field, indices[index], disks, source_count, inner, outer);
}

}  // namespace

int LaunchWettedPistonSections3D(
    wet::Field field,
    const Vec3* centers,
    wet::Disk* disks,
    std::size_t source_count,
    int* architecture,
    void* stream
) {
  constexpr unsigned int kThreads = 128;
  const auto blocks = static_cast<unsigned int>(1 + (source_count - 1) / kThreads);
  static_cast<void>(cudaGetLastError());
  WettedPistonSections3DKernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      field,
      centers,
      disks,
      source_count,
      architecture
  );
  return cudaGetLastError();
}

int LaunchWettedPistonPatches3D(
    wet::Field field,
    const wet::Disk* disks,
    std::size_t source_count,
    const wet::PatchIndex* indices,
    wet::PatchMeasure* measures,
    std::size_t patch_count,
    wet::Interval* intervals,
    std::size_t interval_stride,
    int* architecture,
    void* stream
) {
  const auto blocks = static_cast<unsigned int>(1 + (patch_count - 1) / kPatchThreadsPerBlock);
  static_cast<void>(cudaGetLastError());
  WettedPistonPatches3DKernel<<<blocks, kPatchThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(
      field,
      disks,
      source_count,
      indices,
      measures,
      patch_count,
      intervals,
      interval_stride,
      architecture
  );
  return cudaGetLastError();
}

}  // namespace fluidity::cuda_internal
