#include "fluidity/base/memory_count.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace fluidity {
namespace {

constexpr std::uint64_t kLargestBytes = std::numeric_limits<std::uint64_t>::max();

TEST(ComputeMemoryBytes, EmptyEstimate_IsZero) {
  const MemoryCounts counts = {.particles = 1000};
  EXPECT_EQ(ComputeMemoryBytes({}, counts), 0U);
  EXPECT_EQ(ComputeMemoryBytes({}, counts, MemoryWhere::kCpu), 0U);
}

TEST(ComputeMemoryBytes, EntryPerParticleWithTwoCopies_MultipliesAll) {
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "particles: positions",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 24,
          .copies = 2,
      },
  };
  const MemoryCounts counts = {.particles = 1000};
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts), 48000U);
}

TEST(ComputeMemoryBytes, GivenCountEntry_IgnoresSceneCounts) {
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "table",
          .counted_per = CountedPer::kGivenCount,
          .given_count = 7,
          .bytes_each = 8,
      },
  };
  const MemoryCounts counts = {
      .cell_blocks = 5,
      .cell_places = 6,
      .face_places = 7,
      .corner_places = 8,
      .particles = 9,
  };
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts), 56U);
}

TEST(ComputeMemoryBytes, EveryCountedPerValue_UsesItsOwnCount) {
  const MemoryCounts counts = {
      .cell_blocks = 1,
      .cell_places = 10,
      .face_places = 100,
      .corner_places = 1000,
      .particles = 10000,
  };
  const std::array<CountedPer, 5> kinds = {
      CountedPer::kCellBlock,
      CountedPer::kCellPlace,
      CountedPer::kFacePlace,
      CountedPer::kCornerPlace,
      CountedPer::kParticle,
  };
  const std::array<std::uint64_t, 5> expected_bytes = {1, 10, 100, 1000, 10000};
  for (std::size_t index = 0; index < kinds.size(); ++index) {
    const std::array estimate = {MemoryEstimateEntry{.name = "one", .counted_per = kinds[index], .bytes_each = 1}};
    EXPECT_EQ(ComputeMemoryBytes(estimate, counts), expected_bytes[index]) << index;
  }
}

TEST(ComputeMemoryBytes, WithWhere_CountsOnlyEntriesKeptThere) {
  const std::array estimate = {
      MemoryEstimateEntry{.name = "on the processor", .counted_per = CountedPer::kParticle, .bytes_each = 4},
      MemoryEstimateEntry{
          .name = "in CPU memory",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 8,
          .where = MemoryWhere::kCpu,
      },
  };
  const MemoryCounts counts = {.particles = 10};
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kRunOn), 40U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts, MemoryWhere::kCpu), 80U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts), 120U);
}

TEST(ComputeMemoryBytes, KeptForOtherThanWholeRun_StillCounts) {
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "working buffers",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 4,
          .kept_for = KeptFor::kOneCall,
      },
  };
  const MemoryCounts counts = {.particles = 10};
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts), 40U);
}

TEST(ComputeMemoryBytes, ZeroCopies_CountsNothing) {
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "absent",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 24,
          .copies = 0,
      },
  };
  const MemoryCounts counts = {.particles = 1000};
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts), 0U);
}

TEST(ComputeMemoryBytes, CountedPerOutsideTheEnum_CountsZeroItems) {
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "broken",
          .counted_per = static_cast<CountedPer>(200),
          .given_count = 50,
          .bytes_each = 8,
      },
  };
  const MemoryCounts counts = {.particles = 1000};
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts), 0U);
}

TEST(ComputeMemoryBytes, HugeCounts_SaturateInsteadOfWrapping) {
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "huge",
          .counted_per = CountedPer::kParticle,
          .bytes_each = 24,
          .copies = 2,
      },
  };
  const MemoryCounts counts = {.particles = kLargestBytes / 2};
  EXPECT_EQ(ComputeMemoryBytes(estimate, counts), kLargestBytes);
}

TEST(ComputeMemoryBytes, HugeCountsInOneEntry_Saturates) {
  // The first product (items x bytes_each) fits; the second (x copies) does not.
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "huge copies",
          .counted_per = CountedPer::kGivenCount,
          .given_count = kLargestBytes / 4,
          .bytes_each = 2,
          .copies = 3,
      },
  };
  EXPECT_EQ(ComputeMemoryBytes(estimate, MemoryCounts{}), kLargestBytes);
}

TEST(ComputeMemoryBytes, SumOfEntriesOverflowing_Saturates) {
  // Each entry fits on its own; their sum does not.
  const std::array estimate = {
      MemoryEstimateEntry{
          .name = "half",
          .counted_per = CountedPer::kGivenCount,
          .given_count = kLargestBytes / 2 + 1,
          .bytes_each = 1,
      },
      MemoryEstimateEntry{
          .name = "other half",
          .counted_per = CountedPer::kGivenCount,
          .given_count = kLargestBytes / 2 + 1,
          .bytes_each = 1,
      },
  };
  EXPECT_EQ(ComputeMemoryBytes(estimate, MemoryCounts{}), kLargestBytes);
}

TEST(ComputeRunMemoryBytes, DriverStackEntry_IsNotInTheBlock) {
  // The stack the driver holds for a GPU function counts in the total (as the largest such entry), never in the
  // run's block.
  const std::array estimate = {
      MemoryEstimateEntry{.name = "a: buffer", .given_count = 1000, .bytes_each = 4},
      MemoryEstimateEntry{
          .name = "b: stack",
          .given_count = 2048,
          .bytes_each = 12800,
          .where = MemoryWhere::kDriverStack,
      },
  };
  const MemoryCounts no_counts;
  EXPECT_EQ(ComputeRunMemoryBytes(estimate, no_counts, MemoryWhere::kRunOn), 4096U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, no_counts, MemoryWhere::kDriverStack), 2048U * 12800U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, no_counts), 4000U + (2048U * 12800U));
}

MemoryEstimateEntry BuildStackEntry(std::uint64_t byte_count, MemoryWhere where) {
  return MemoryEstimateEntry{
      .name = "stack",
      .given_count = byte_count,
      .bytes_each = 1,
      .where = where,
  };
}

TEST(ComputeLargestEntryBytes, NoEntry_IsZero) {
  EXPECT_EQ(ComputeLargestEntryBytes({}, MemoryCounts{}, MemoryWhere::kDriverStack), 0U);
}

TEST(ComputeLargestEntryBytes, OneEntry_IsItsBytes) {
  const std::array estimate = {BuildStackEntry(700, MemoryWhere::kDriverStack)};
  EXPECT_EQ(ComputeLargestEntryBytes(estimate, MemoryCounts{}, MemoryWhere::kDriverStack), 700U);
}

TEST(ComputeLargestEntryBytes, MiddleEntryLargest_IsTheMiddleNotTheSum) {
  // 100, 700, 300 bytes: the largest is 700, the sum would be 1,100.
  const std::array estimate = {
      BuildStackEntry(100, MemoryWhere::kDriverStack),
      BuildStackEntry(700, MemoryWhere::kDriverStack),
      BuildStackEntry(300, MemoryWhere::kDriverStack),
  };
  EXPECT_EQ(ComputeLargestEntryBytes(estimate, MemoryCounts{}, MemoryWhere::kDriverStack), 700U);
  EXPECT_EQ(ComputeMemoryBytes(estimate, MemoryCounts{}, MemoryWhere::kDriverStack), 1100U);
}

TEST(ComputeLargestEntryBytes, EntriesOfAnotherWhere_AreIgnored) {
  // The kRunOn entry of 5,000 bytes and the kCpu entry of 9,000 bytes do not count for kDriverStack.
  const std::array estimate = {
      BuildStackEntry(5000, MemoryWhere::kRunOn),
      BuildStackEntry(200, MemoryWhere::kDriverStack),
      BuildStackEntry(9000, MemoryWhere::kCpu),
  };
  EXPECT_EQ(ComputeLargestEntryBytes(estimate, MemoryCounts{}, MemoryWhere::kDriverStack), 200U);
  EXPECT_EQ(ComputeLargestEntryBytes(estimate, MemoryCounts{}, MemoryWhere::kCpu), 9000U);
}

}  // namespace
}  // namespace fluidity
