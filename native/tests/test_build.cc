#include <concepts>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

namespace fluidity {
namespace {

TEST(Build, Cxx20StatusRoundTrip) {
  static_assert(std::same_as<absl::StatusOr<int>::value_type, int>);
  const absl::StatusOr<int> success(7);
  ASSERT_TRUE(success.ok()) << success.status();
  EXPECT_EQ(*success, 7);

  const absl::StatusOr<int> failure(absl::InvalidArgumentError("build smoke test"));
  ASSERT_FALSE(failure.ok());
  EXPECT_EQ(failure.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(failure.status().message(), "build smoke test");
}

}  // namespace
}  // namespace fluidity
