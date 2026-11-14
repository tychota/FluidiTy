// Runs on a machine with a usable GPU: label gpu.

#include <gtest/gtest.h>

#include "fluidity/base/processor_description.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

TEST(ReadProcessorDescription, OnGpu_ModelIsNotEmpty) {
  const ProcessorDescription description = ReadProcessorDescription(RunOn::kGpu);
  EXPECT_FALSE(description.model.empty());
  EXPECT_FALSE(description.driver.empty());
  EXPECT_FALSE(description.runtime.empty());
}

}  // namespace
}  // namespace fluidity
