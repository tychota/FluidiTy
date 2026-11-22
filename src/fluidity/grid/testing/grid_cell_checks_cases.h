// Test-only: the helpers that the unit tests and the functional tests of grid_cell_checks share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ToUnitFrame, ToCell, ToValues, ToSizeLevelNumber, ToSmallCells, CpuGridBuffers, BuildCpuGridBuffers,
//          ToInputs, ReadLentCellIds, ExpectFailure, BuildTwoSizeLevelsApart, CheckThroughSortedCellList
// Uses:    the headers it includes
// Used by: src/fluidity/grid/grid_cell_checks.test.cc and tests/grid/grid_cell_checks_on_random_grids.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

// A frame with its origin at 0 and biggest cells 1 m wide.
inline GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

inline CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

inline std::vector<std::uint64_t> ToValues(std::span<const CellId> cells) {
  std::vector<std::uint64_t> values;
  values.reserve(cells.size());
  for (const CellId cell : cells) {
    values.push_back(cell.value);
  }
  return values;
}

inline int ToSizeLevelNumber(CellId cell) {
  return static_cast<int>(ToSizeLevel(cell));
}

// The 8 small cells of a big cell, by the test's own arithmetic: 2 c and 2 c + 1 on each axis.
inline std::vector<CellId> ToSmallCells(CellId big_cell) {
  const CellCoord coord = ToCellCoord(big_cell);
  const int small_size_level = ToSizeLevelNumber(big_cell) + 1;
  std::vector<CellId> small_cells;
  for (std::int32_t dz = 0; dz < 2; ++dz) {
    for (std::int32_t dy = 0; dy < 2; ++dy) {
      for (std::int32_t dx = 0; dx < 2; ++dx) {
        small_cells.push_back(ToCell(small_size_level, (2 * coord.x) + dx, (2 * coord.y) + dy, (2 * coord.z) + dz));
      }
    }
  }
  return small_cells;
}

// A CPU RunMemory with grid buffers and the builder buffers lent to them. Kept behind a pointer, since lent builder
// buffers must not move; the members are destroyed in reverse order, the grid buffers first.
struct CpuGridBuffers {
  std::unique_ptr<RunMemory> run_memory;
  std::unique_ptr<GridBuilderBuffers> builder_buffers;
  std::unique_ptr<GridBuffers> buffers;
};

// Builds CPU grid buffers of max_cell_blocks blocks in a RunMemory of exactly their estimate. Null, with a test
// failure, when a builder fails.
inline std::unique_ptr<CpuGridBuffers> BuildCpuGridBuffers(std::uint32_t max_cell_blocks) {
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(RunOn::kCpu, 1, max_cell_blocks);
  GridCounts room_counts = {};
  room_counts.cell_blocks = max_cell_blocks;
  const MemoryCounts memory_counts = ToMemoryCounts(room_counts, 0);
  // A CPU run keeps its kRunOn and its kCpu entries in one CPU block.
  const std::size_t byte_count = ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kRunOn)
      + ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kCpu);
  auto built = std::make_unique<CpuGridBuffers>();
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(RunOn::kCpu, byte_count);
  if (!run_memory.has_value()) {
    ADD_FAILURE() << "BuildRunMemory failed";
    return nullptr;
  }
  built->run_memory = std::make_unique<RunMemory>(std::move(*run_memory));
  std::expected<GridBuilderBuffers, GridFailure> builder_buffers =
      BuildGridBuilderBuffers(RunOn::kCpu, max_cell_blocks, *built->run_memory);
  if (!builder_buffers.has_value()) {
    ADD_FAILURE() << "BuildGridBuilderBuffers failed";
    return nullptr;
  }
  built->builder_buffers = std::make_unique<GridBuilderBuffers>(std::move(*builder_buffers));
  const GridSettings settings = {.max_cell_blocks = max_cell_blocks, .run_on = RunOn::kCpu};
  std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers(settings, *built->builder_buffers, *built->run_memory);
  if (!buffers.has_value()) {
    ADD_FAILURE() << "BuildGridBuffers failed";
    return nullptr;
  }
  built->buffers = std::make_unique<GridBuffers>(std::move(*buffers));
  return built;
}

// The inputs of a CPU list of version 1.
inline GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

// The first `count` cell ids of builder buffers, as values: the part a check of `count` cells uses.
inline std::vector<std::uint64_t> ReadLentCellIds(const GridBuilderBuffers& builder_buffers, std::size_t count) {
  const Buffer<CellId>& cell_ids = ToGridBuilderBuffersArrays(builder_buffers).cell_ids;
  return ToValues(std::span<const CellId>(cell_ids.data(), std::min(count, cell_ids.size())));
}

inline void ExpectFailure(
    const std::expected<void, GridFailure>& result,
    GridError error,
    CellId cell = CellId{0},
    CellId other_cell = CellId{0}
) {
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().error, error);
  EXPECT_EQ(result.error().cell.value, cell.value);
  EXPECT_EQ(result.error().other_cell.value, other_cell.value);
  EXPECT_EQ(result.error().magnitude, 0U);
}

// Size level 0 {0,0,0}, which covers [0, 1) m on each axis, and a size-level-2 cell 0.25 m wide just beyond its high
// side: {4,4,4} covers [1, 1.25) on each axis and touches it at the corner (1, 1, 1); {4,4,0} along the edge x = y = 1;
// {4,0,0} on the side x = 1. Each pair is two size levels apart.
inline std::vector<CellId> BuildTwoSizeLevelsApart(CellId small_cell) {
  return {ToCell(0, 0, 0, 0), small_cell};
}

// ---------------------------------------------------------------------------------------------------------------------
// BuildSortedCellList
// ---------------------------------------------------------------------------------------------------------------------

// Runs BuildSortedCellList and drops the count, for ExpectFailure.
inline std::expected<void, GridFailure>
CheckThroughSortedCellList(GridFrame frame, std::span<const CellId> cells, GridBuffers& buffers) {
  const std::expected<std::uint32_t, GridFailure> built =
      BuildSortedCellList(ToInputs(frame, cells), buffers, SiblingCheck::kRequire);
  if (!built.has_value()) {
    return std::unexpected(built.error());
  }
  return {};
}

}  // namespace fluidity::grid
