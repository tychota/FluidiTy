#include "fluidity/base/processor_description.h"

#include <gtest/gtest.h>

#include <expected>
#include <string>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(ReadMemoryAvailable, OnCpu_FreeIsPositive) {
  const std::expected<MemoryAvailable, BufferError> memory = ReadMemoryAvailable(RunOn::kCpu);
  ASSERT_TRUE(memory.has_value()) << ToName(memory.error());
  EXPECT_GT(memory->total_bytes, 0U);
  EXPECT_GT(memory->free_bytes, 0U);
}

TEST(ReadMemoryAvailable, OnCpu_FreeIsNotAboveTotal) {
  const std::expected<MemoryAvailable, BufferError> memory = ReadMemoryAvailable(RunOn::kCpu);
  ASSERT_TRUE(memory.has_value()) << ToName(memory.error());
  EXPECT_LE(memory->free_bytes, memory->total_bytes);
}

TEST(ReadProcessorDescription, OnCpu_ModelIsNotEmpty) {
  const ProcessorDescription description = ReadProcessorDescription(RunOn::kCpu);
  EXPECT_FALSE(description.model.empty());
}

TEST(ReadProcessorDescription, OnCpu_DriverIsNotEmpty) {
  EXPECT_FALSE(ReadProcessorDescription(RunOn::kCpu).driver.empty());
}

TEST(ReadProcessorDescription, OnCpu_RuntimeIsNotEmpty) {
  EXPECT_FALSE(ReadProcessorDescription(RunOn::kCpu).runtime.empty());
}

#if !defined(FLUIDITY_HAS_GPU)
TEST(ReadProcessorDescription, OnGpuInBuildWithoutGpuCode_AllFieldsEmpty) {
  const ProcessorDescription description = ReadProcessorDescription(RunOn::kGpu);
  EXPECT_TRUE(description.model.empty());
  EXPECT_TRUE(description.driver.empty());
  EXPECT_TRUE(description.runtime.empty());
}

TEST(ReadMemoryAvailable, OnGpuInBuildWithoutGpuCode_IsRunOnCallFailed) {
  const std::expected<MemoryAvailable, BufferError> memory = ReadMemoryAvailable(RunOn::kGpu);
  ASSERT_FALSE(memory.has_value());
  EXPECT_EQ(memory.error(), BufferError::kRunOnCallFailed);
}
#endif

}  // namespace
}  // namespace fluidity
