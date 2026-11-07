#include <bit>
#include <cstdint>
#include <limits>
#include <string_view>

#include "fixture_escaping_data.h"
#include "gtest/gtest.h"

namespace fluidity::fixtures {
namespace {

TEST(FixtureBridge, EscapedStringsAndTypedLiteralsRoundTrip) {
  const auto fixtures = AllFixtures();
  ASSERT_EQ(fixtures.size(), 1);
  const auto& fixture = fixtures.front();
  EXPECT_EQ(fixture.provenance, "Derived from \"one\" at C:\\test.");
  EXPECT_EQ(fixture.default_policy, std::string_view("line\n\0\303\251", 8));

  const auto ints = FindArray(fixture, "int \"limits\" \\ array");
  ASSERT_TRUE(ints.ok()) << ints.status();
  EXPECT_EQ((*ints)->units, "unit \"i\" \\");
  ASSERT_EQ((*ints)->i64_data.size(), 2);
  EXPECT_EQ((*ints)->i64_data[0], std::numeric_limits<std::int64_t>::min());
  EXPECT_EQ((*ints)->i64_data[1], std::numeric_limits<std::int64_t>::max());

  const auto floats = FindArray(fixture, "float");
  ASSERT_TRUE(floats.ok()) << floats.status();
  ASSERT_EQ((*floats)->f64_data.size(), 3);
  EXPECT_EQ(std::bit_cast<std::uint64_t>((*floats)->f64_data[0]), 0x8000000000000000ULL);
  EXPECT_EQ(std::bit_cast<std::uint64_t>((*floats)->f64_data[1]), 0x0000000000000001ULL);
  EXPECT_EQ(std::bit_cast<std::uint64_t>((*floats)->f64_data[2]), 0x3fb999999999999aULL);
}

}  // namespace
}  // namespace fluidity::fixtures
