#include "fluidity/base/test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"

namespace fluidity {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr Tolerance kTolerance = {.absolute_floor = 1e-15, .relative_term = 1e-15};

TEST(ToBits, TwoZeros_Differ) {
  EXPECT_NE(ToBits(0.0), ToBits(-0.0));
}

TEST(ToBits, One_IsTheIeeeBits) {
  EXPECT_EQ(ToBits(1.0), 0x3FF0000000000000ULL);
}

TEST(CheckWithinTolerance, DifferenceBelowTheAllowed_Passes) {
  EXPECT_TRUE(CheckWithinTolerance(2.0, 2.0 + 2e-15, kTolerance));
}

TEST(CheckWithinTolerance, DifferenceAboveTheAllowed_Fails) {
  EXPECT_FALSE(CheckWithinTolerance(2.0, 2.0 + 1e-14, kTolerance));
}

TEST(CheckWithinTolerance, ValueNearZero_UsesTheAbsoluteFloor) {
  EXPECT_TRUE(CheckWithinTolerance(0.0, 5e-16, kTolerance));
}

TEST(CheckWithinTolerance, EqualInfinities_Pass) {
  EXPECT_TRUE(
      CheckWithinTolerance(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), kTolerance)
  );
}

TEST(CheckWithinTolerance, OppositeInfinities_Fail) {
  EXPECT_FALSE(CheckWithinTolerance(
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      kTolerance
  ));
}

TEST(CheckWithinTolerance, NaN_Fails) {
  EXPECT_FALSE(CheckWithinTolerance(kNaN, kNaN, kTolerance));
}

TEST(CheckSameBitsOrBothNaN, EqualValues_Pass) {
  EXPECT_TRUE(CheckSameBitsOrBothNaN(1.5, 1.5));
}

TEST(CheckSameBitsOrBothNaN, TwoZeros_Fail) {
  EXPECT_FALSE(CheckSameBitsOrBothNaN(0.0, -0.0));
}

TEST(CheckSameBitsOrBothNaN, TwoNaNsWithOtherBits_Pass) {
  EXPECT_TRUE(CheckSameBitsOrBothNaN(kNaN, -kNaN));
}

TEST(CheckSameBitsOrBothNaN, NaNAgainstANumber_Fails) {
  EXPECT_FALSE(CheckSameBitsOrBothNaN(kNaN, 1.0));
}

TEST(ToBits, ArrayOfDoubles_GivesTheBitsInOrder) {
  const std::array values = {1.0, -0.0};
  EXPECT_EQ(ToBits(values), (std::vector<std::uint64_t>{0x3FF0000000000000ULL, 0x8000000000000000ULL}));
}

TEST(BuildBufferHolding, OnCpu_ReadAllGivesTheValuesBack) {
  const Buffer<double> buffer = BuildBufferHolding(RunOn::kCpu, std::array{1.5, -2.0, 3.25});
  EXPECT_EQ(buffer.run_on(), RunOn::kCpu);
  EXPECT_EQ(ReadAll(buffer), (std::vector{1.5, -2.0, 3.25}));
}

TEST(ComputeResultDigest, NoBytes_IsTheStartingValue) {
  EXPECT_EQ(ComputeResultDigest({}), 0xcbf29ce484222325ULL);
}

TEST(ComputeResultDigest, OneByteA_IsTheKnownFnvValue) {
  // The FNV-1a test vector for the text "a".
  const std::array bytes = {std::byte{'a'}};
  EXPECT_EQ(ComputeResultDigest(bytes), 0xaf63dc4c8601ec8cULL);
}

TEST(WriteResultDigestToFile, TwoResults_AppendTwoLines) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "fluidity_test_support_digests.txt";
  std::filesystem::remove(path);
  const std::array first = {std::byte{'a'}};
  WriteResultDigestToFile(path, "first", first);
  WriteResultDigestToFile(path, "second", {});
  std::stringstream text;
  {
    // The file is closed before it is removed: Windows refuses to remove an open file.
    const std::ifstream file(path);
    text << file.rdbuf();
  }
  EXPECT_EQ(text.str(), "first af63dc4c8601ec8c\nsecond cbf29ce484222325\n");
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace fluidity
