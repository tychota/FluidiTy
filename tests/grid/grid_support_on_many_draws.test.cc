#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/grid_support_cases.h"

namespace fluidity::grid {
namespace {

// The C++ standard fixes this value ([rand.predef]); a library that gives another one would change every random grid.
TEST(StandardGenerator, TenThousandthOutput_IsTheValueTheStandardFixes) {
  std::mt19937_64 generator;  // NOLINT(bugprone-random-generator-seed): the default seed, 5489, is the point.
  generator.discard(9999);
  EXPECT_EQ(generator(), 9981545732273789042ULL);
}

TEST(BuildRandomGridCells, OtherSizeLevelCountsAndBoxes_KeepTheirDigest) {
  // Pairs (size_level_count, biggest_cells_per_axis), each for seeds 1..10.
  constexpr std::array<std::array<int, 2>, 5> kSizes = {{{1, 3}, {2, 4}, {3, 3}, {5, 2}, {3, 1}}};
  std::vector<std::uint64_t> words;
  for (const std::array<int, 2>& size : kSizes) {
    for (std::uint64_t seed = 1; seed <= 10; ++seed) {
      ApplyCellsToWords(BuildRandomGridCells(seed, size.at(0), size.at(1)), words);
    }
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x9b6001a17bb5ed90ULL);
}

}  // namespace
}  // namespace fluidity::grid
