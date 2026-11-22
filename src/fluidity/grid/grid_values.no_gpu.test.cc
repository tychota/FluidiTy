#include "fluidity/grid/grid_values.h"

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_error.h"

namespace fluidity::grid {
namespace {

// A machine without a GPU cannot run the GPU: the call fails before it touches the block.
TEST(BuildFaceValues, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  std::expected<RunMemory, BufferError> cpu_run_memory = BuildRunMemory(RunOn::kCpu, 4096);
  ASSERT_TRUE(cpu_run_memory.has_value());
  const std::expected<FaceValues<double>, GridFailure> values =
      BuildFaceValues<double>(RunOn::kGpu, 1000, *cpu_run_memory);
  ASSERT_FALSE(values.has_value());
  EXPECT_EQ(values.error().error, GridError::kRunOnCallFailed);
  EXPECT_EQ(cpu_run_memory->handed_out_bytes(), 0U);
}

}  // namespace
}  // namespace fluidity::grid
