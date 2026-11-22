// Test-only: the helpers that the unit tests and the functional tests of staggered_grid_counts_on_buffers share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ToUnitFrame, ToInputs, BuildCellsInBlocksOfTheirOwn
// Uses:    the headers it includes
// Used by: src/fluidity/grid/staggered_grid.counts_on_buffers.test.cc and tests/grid/grid_counts_on_buffers.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/allocation_counter.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

inline GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

// The inputs of a CPU list of version 1.
inline GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

// Size-level-0 cells {4 i, 0, 0} for i = 0..count - 1: each alone in a block of its own.
inline std::vector<CellId> BuildCellsInBlocksOfTheirOwn(std::int32_t count) {
  std::vector<CellId> cells;
  cells.reserve(static_cast<std::size_t>(count));
  for (std::int32_t index = 0; index < count; ++index) {
    cells.push_back(ToCellId(SizeLevel{0}, {.x = 4 * index, .y = 0, .z = 0}));
  }
  return cells;
}

}  // namespace fluidity::grid
