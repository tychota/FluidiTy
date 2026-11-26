#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fluidity/grid/testing/border_block_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

constexpr std::uint32_t kRangeBlocks = 4;

// The stacked layers of every other framed random grid (origins and widths that are no power of two), on the tested
// ranges of 4 blocks: the first, the middle, the last and the whole grid, at four layers. It has its own program
// because it takes about 50 s in release, and the program of the other stacked checks takes over 100 s.
TEST(FindBorderBlocks, StackedLayersOfFramedRandomGrids_AnswerAsTheWholeGrid) {
  const RandomGridsWithCheckers& built = FindFramedRandomGridsWithCheckers();
  std::vector<std::size_t> grids;
  for (std::size_t grid = 0; grid < built.grids.size(); grid += 2) {
    grids.push_back(grid);
  }
  CheckStackedLayersAnswerAsTheWholeGrid(built, grids, kRangeBlocks);
}

}  // namespace
}  // namespace fluidity::grid
