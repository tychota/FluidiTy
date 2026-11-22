#include <gtest/gtest.h>

#include <cstddef>

#include "fluidity/grid/cell_size_rule.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/testing/cell_size_rule_factor_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

TEST(ComputeCellSizeFactorCode, NoVolume_IsZero) {
  const RefinementFactors no_volume{};
  for (const CellId cell : BuildRandomGridCells(7, 5, 4)) {
    EXPECT_EQ(ComputeCellSizeFactorCode(no_volume, kSmallestSizeLevel, cell), 0);
  }
}

TEST(ComputeCellSizeFactorCode, EveryStoredCode_IndexesTheTable) {
  for (int stored = -128; stored <= 127; ++stored) {
    const int table_index = ComputeCodeOfStoredCode(stored) + kMaxCellSizeFactorCode;
    EXPECT_GE(table_index, 0) << stored;
    EXPECT_LE(table_index, 48) << stored;
  }
}

}  // namespace
}  // namespace fluidity::grid
