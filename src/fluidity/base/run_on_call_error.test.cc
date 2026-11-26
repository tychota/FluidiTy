#include "fluidity/base/run_on_call_error.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <string_view>

namespace fluidity {
namespace {

TEST(FindLastRunOnCallError, NoFailure_HasNoError) {
  ApplyNoRunOnCallError();
  const RunOnCallError error = FindLastRunOnCallError();
  EXPECT_FALSE(error.has_error);
  EXPECT_EQ(std::string_view(error.name), "");
  EXPECT_EQ(std::string_view(error.text), "");
}

TEST(FindLastRunOnCallError, AfterApply_HoldsTheErrorTexts) {
  ApplyLastRunOnCallError("cudaErrorInvalidValue", "invalid argument");
  const RunOnCallError error = FindLastRunOnCallError();
  EXPECT_TRUE(error.has_error);
  EXPECT_EQ(std::string_view(error.name), "cudaErrorInvalidValue");
  EXPECT_EQ(std::string_view(error.text), "invalid argument");
}

TEST(FindLastRunOnCallError, TwoFailures_HoldsTheLast) {
  ApplyLastRunOnCallError("cudaErrorLaunchFailure", "unspecified launch failure");
  ApplyLastRunOnCallError("cudaErrorInvalidConfiguration", "invalid configuration argument");
  const RunOnCallError error = FindLastRunOnCallError();
  EXPECT_EQ(std::string_view(error.name), "cudaErrorInvalidConfiguration");
  EXPECT_EQ(std::string_view(error.text), "invalid configuration argument");
}

TEST(ApplyNoRunOnCallError, AfterFailure_ClearsTheRecord) {
  ApplyLastRunOnCallError("cudaErrorMemoryAllocation", "out of memory");
  ApplyNoRunOnCallError();
  const RunOnCallError error = FindLastRunOnCallError();
  EXPECT_FALSE(error.has_error);
  EXPECT_EQ(std::string_view(error.name), "");
}

TEST(ApplyLastRunOnCallError, TooLongText_IsCutToTheLimit) {
  const std::string long_text(kMaxRunOnCallErrorChars + 50, 'x');
  ApplyLastRunOnCallError(long_text.c_str(), long_text.c_str());
  const RunOnCallError error = FindLastRunOnCallError();
  EXPECT_EQ(std::string_view(error.name).size(), kMaxRunOnCallErrorChars);
  EXPECT_EQ(std::string_view(error.text).size(), kMaxRunOnCallErrorChars);
}

TEST(ApplyLastRunOnCallError, NullPointers_GiveEmptyTexts) {
  ApplyLastRunOnCallError(nullptr, nullptr);
  const RunOnCallError error = FindLastRunOnCallError();
  EXPECT_TRUE(error.has_error);
  EXPECT_EQ(std::string_view(error.name), "");
  EXPECT_EQ(std::string_view(error.text), "");
}

}  // namespace
}  // namespace fluidity
