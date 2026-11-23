// Needs a build with GPU code on a machine without a usable GPU: label no_gpu. Runs in the CUDA container only.

#include "fluidity/grid/cell_block_range.h"

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"

namespace fluidity::grid {
namespace {

TEST(BuildBorderSearchBuffers, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  // A CPU block: the GPU cannot run, so the call fails before it takes a memory piece.
  std::expected<RunMemory, BufferError> cpu_run_memory = BuildRunMemory(RunOn::kCpu, 1048576U);
  ASSERT_TRUE(cpu_run_memory.has_value());
  const std::expected<BorderSearchBuffers, GridFailure> search_buffers =
      BuildBorderSearchBuffers(RunOn::kGpu, 1000, *cpu_run_memory);
  ASSERT_FALSE(search_buffers.has_value());
  EXPECT_EQ(search_buffers.error().error, GridError::kRunOnCallFailed);
  EXPECT_EQ(cpu_run_memory->handed_out_bytes(), 0U);
}

TEST(BuildBorderSearchStackEstimate, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  const std::expected<MemoryEstimateEntry, GridFailure> entry = BuildBorderSearchStackEstimate(RunOn::kGpu);
  ASSERT_FALSE(entry.has_value());
  EXPECT_EQ(entry.error().error, GridError::kRunOnCallFailed);
}

}  // namespace
}  // namespace fluidity::grid
