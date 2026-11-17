// Needs a build with GPU code on a machine without a usable GPU: label no_gpu. Runs in the CUDA container only.

#include <gtest/gtest.h>

#include <expected>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/contact/contact_error.h"
#include "fluidity/contact/move_particles.h"

namespace fluidity::contact {
namespace {

TEST(BuildContactBuffers, GpuWithoutAGpu_IsRunOnCallFailed) {
  // The GPU was asked for and cannot run: the call fails before it touches the block, whatever block it is given.
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, 65536);
  ASSERT_TRUE(run_memory.has_value());
  const std::expected<ContactBuffers, ContactFailure> buffers = BuildContactBuffers(RunOn::kGpu, 1000, 4, *run_memory);
  ASSERT_FALSE(buffers.has_value());
  EXPECT_EQ(buffers.error().error, ContactError::kRunOnCallFailed);
  EXPECT_EQ(run_memory->handed_out_bytes(), 0U);
}

TEST(BuildContactStackEstimate, GpuWithoutAGpu_IsRunOnCallFailed) {
  const std::expected<MemoryEstimateEntry, ContactFailure> entry = BuildContactStackEstimate(RunOn::kGpu);
  ASSERT_FALSE(entry.has_value());
  EXPECT_EQ(entry.error().error, ContactError::kRunOnCallFailed);
}

}  // namespace
}  // namespace fluidity::contact
