// Test-only: the helpers that the unit tests and the functional tests of cell_size_rule_factor share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    kSmallestSizeLevel, ToTestCell, ComputeFixtureCode, BuildLongVolume
// Uses:    the headers it includes
// Used by: src/fluidity/grid/cell_size_rule.factor.test.cc and tests/grid/cell_size_factor_on_large_volume.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "fluidity/base/scalar_math.h"
#include "fluidity/grid/cell_size_rule.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/testing/cell_size_rule_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

// The fixture of testing/cell_size_rule_cases.h, which the GPU test shares.
constexpr SizeLevel kSmallestSizeLevel = kFactorFixtureSmallestSizeLevel;

inline CellId ToTestCell(int size_level, int x, int y, int z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// The factor code of the cell of `size_level` at (x, y, z), with the fixture's smallest size level.
// clang-tidy: a size level and three coordinates; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline int ComputeFixtureCode(const RefinementFactors& factors, int size_level, int x, int y, int z) {
  return ComputeCellSizeFactorCode(factors, kSmallestSizeLevel, ToTestCell(size_level, x, y, z));
}

// A volume of `count` entries on `axis` (0 for x, 1 for y, 2 for z) and 1 on the others, spacing 0, 3 layers, layer 0
// holding `code` at every entry except those of `coded_entries`, which hold -7.
// clang-tidy: three integers of different meaning; the swappable-parameters check flags them.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline TestRefinementVolume
BuildLongVolume(int axis, int count, std::int8_t code, const std::vector<int>& coded_entries) {
  std::vector<std::int8_t> codes(static_cast<std::size_t>(count), code);
  for (const int coded_entry : coded_entries) {
    codes[static_cast<std::size_t>(coded_entry)] = -7;
  }
  return BuildTestRefinementFactors(axis == 0 ? count : 1, axis == 1 ? count : 1, axis == 2 ? count : 1, 0, 3, codes);
}

}  // namespace fluidity::grid
