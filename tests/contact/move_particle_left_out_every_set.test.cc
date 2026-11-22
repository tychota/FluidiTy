#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "fluidity/contact/testing/left_out_reference.h"
#include "fluidity/contact/testing/left_out_solids_sets.h"

namespace fluidity::contact {
namespace {

TEST(LeftOutSolids, EverySet_SameBitsAsReadingEverySolid) {
  // The catalogue, the oblique and round sets, the crowded, far, mixed and boundary sets: every bit of every answer is
  // the same with the rule and with every solid read.
  for (int set_index = 0; set_index < kLeftOutSetCount; ++set_index) {
    SCOPED_TRACE(set_index);
    const std::vector<std::uint64_t> with_rule = ComputeResultBits(static_cast<LeftOutSet>(set_index));
    const std::vector<std::uint64_t> reading_every_solid = left_out_reference::ComputeReferenceResultBits(set_index);
    EXPECT_FALSE(with_rule.empty());
    EXPECT_EQ(with_rule, reading_every_solid);
  }
}

}  // namespace
}  // namespace fluidity::contact
