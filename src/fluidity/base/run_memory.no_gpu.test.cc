// Runs on a machine without a usable GPU, such as the CUDA container: label no_gpu.

#include "fluidity/base/run_memory.h"

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(BuildRunMemory, OnGpuWithoutAGpu_IsRunOnCallFailed) {
  const std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kGpu, 4096);
  ASSERT_FALSE(run_memory.has_value());
  EXPECT_EQ(run_memory.error(), BufferError::kRunOnCallFailed) << ToName(run_memory.error());
}

TEST(BuildRunMemory, LockedForCopiesWithoutAGpu_HandsOutCpuPieces) {
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 1024, CpuMemory::kLockedForCopies);
  ASSERT_TRUE(run_memory.has_value()) << ToName(run_memory.error());
  const std::expected<Buffer<double>, BufferError> piece = BuildBuffer<double>(RunOn::kCpu, 16, *run_memory);
  ASSERT_TRUE(piece.has_value()) << ToName(piece.error());
  EXPECT_EQ(piece->run_on(), RunOn::kCpu);
  EXPECT_EQ(run_memory->handed_out_bytes(), 256U);
}

}  // namespace
}  // namespace fluidity
