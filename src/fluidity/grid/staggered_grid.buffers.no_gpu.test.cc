// Needs a build with GPU code on a machine without a usable GPU: label no_gpu. Runs in the CUDA container only.

#include "fluidity/grid/staggered_grid.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_memory_estimate.h"

namespace fluidity::grid {
namespace {

// The counts of a grid of cell_blocks blocks, as a memory estimate reads them.
MemoryCounts BuildCounts(std::uint32_t cell_blocks) {
  GridCounts counts = {};
  counts.cell_blocks = cell_blocks;
  return ToMemoryCounts(counts, 0);
}

TEST(BuildGridBuilderBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  // A CPU block of the GPU estimate's run-memory bytes: the GPU cannot run, so the call fails before it takes a memory
  // piece.
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kGpu, 1, 1000);
  const std::size_t gpu_bytes = ComputeRunMemoryBytes(estimate, BuildCounts(1000), MemoryWhere::kRunOn);
  std::expected<RunMemory, BufferError> cpu_run_memory = BuildRunMemory(RunOn::kCpu, gpu_bytes);
  ASSERT_TRUE(cpu_run_memory.has_value());
  const std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kGpu, 1000, *cpu_run_memory);
  ASSERT_FALSE(builder_buffers.has_value());
  EXPECT_EQ(builder_buffers.error().error, GridError::kRunOnCallFailed);
  EXPECT_EQ(cpu_run_memory->handed_out_bytes(), 0U);
}

TEST(BuildGridBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  // CPU builder buffers lent to grid buffers asked for on the GPU: the processor check comes first, so the call fails
  // before it takes a memory piece.
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, 1000);
  const MemoryCounts counts = BuildCounts(1000);
  const std::size_t cpu_bytes = ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kRunOn)
      + ComputeRunMemoryBytes(estimate, counts, MemoryWhere::kCpu);
  std::expected<RunMemory, BufferError> cpu_run_memory = BuildRunMemory(RunOn::kCpu, cpu_bytes);
  ASSERT_TRUE(cpu_run_memory.has_value());
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, 1000, *cpu_run_memory);
  ASSERT_TRUE(builder_buffers.has_value());
  const std::size_t handed_out_bytes = cpu_run_memory->handed_out_bytes();
  const std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = 1000, .run_on = RunOn::kGpu}, *builder_buffers, *cpu_run_memory);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error().error, GridError::kRunOnCallFailed);
  EXPECT_EQ(cpu_run_memory->handed_out_bytes(), handed_out_bytes);
}

}  // namespace
}  // namespace fluidity::grid
