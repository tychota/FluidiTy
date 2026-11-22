#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/testing/cell_size_rule_cases.h"
#include "fluidity/grid/testing/cell_size_rule_factor_cases.h"

namespace fluidity::grid {
namespace {

// The code of the size-level-`size_level` cell at coordinate `coord` on `axis` and 0 on the others, in `volume`.
// clang-tidy: three integers of different meaning; the swappable-parameters check flags them.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int ComputeLongVolumeCode(const TestRefinementVolume& volume, int axis, int size_level, int coord) {
  const int x = axis == 0 ? coord : 0;
  const int y = axis == 1 ? coord : 0;
  const int z = axis == 2 ? coord : 0;
  return ComputeFixtureCode(volume.factors, size_level, x, y, z);
}

// The next two tests: 2^20 entries on one axis, each array as long as its counts say, codes 0 except -7 at entries
// kMaxCoord = 2^18 - 1 and 2^20 - 1, with smallest size level 4 and spacing 0. A cell coordinate is at most kMaxCoord.
constexpr int kLongCount = 1'048'576;

TEST(ComputeCellSizeFactorCode, CountOfTwoToTheTwenty_ReadsLayerZeroAtTheLargestCoordinate) {
  // A size-level-4 cell reads layer 0 at its coordinate: entry kMaxCoord, and the entry before it holds 0.
  for (int axis = 0; axis < 3; ++axis) {
    const TestRefinementVolume volume = BuildLongVolume(axis, kLongCount, 0, {kMaxCoord, kLongCount - 1});
    EXPECT_EQ(ComputeLongVolumeCode(volume, axis, 4, kMaxCoord), -7) << axis;
    EXPECT_EQ(ComputeLongVolumeCode(volume, axis, 4, kMaxCoord - 1), 0) << axis;
  }
}

TEST(ComputeCellSizeFactorCode, CountOfTwoToTheTwenty_ReadsItsLastEntry) {
  // A size-level-2 cell is 4 smallest cells wide and reads layer 2 at index kMaxCoord: the smallest code over layer-0
  // entries 4 * kMaxCoord = 2^20 - 4 up to 2^20 - 1, so the last entry is reached this way. The cell before it covers
  // entries 2^20 - 8 up to 2^20 - 5, which hold 0.
  for (int axis = 0; axis < 3; ++axis) {
    const TestRefinementVolume volume = BuildLongVolume(axis, kLongCount, 0, {kMaxCoord, kLongCount - 1});
    EXPECT_EQ(ComputeLongVolumeCode(volume, axis, 2, kMaxCoord), -7) << axis;
    EXPECT_EQ(ComputeLongVolumeCode(volume, axis, 2, kMaxCoord - 1), 0) << axis;
  }
}

}  // namespace
}  // namespace fluidity::grid
