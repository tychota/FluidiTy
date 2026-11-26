// Test-only: the grids and places that the GPU tests of the shared grid functions share.
//
// Purpose: let each GPU test build the same random grids on both processors and list the places a runner is asked
//   about, written once; and give the GPU tests one way to say on which card a number was measured and to compare the
//   card's free bytes.
//
// Idea: a GPU test builds each grid twice from one list, on the CPU as the reference and on the GPU, and asks the same
//   queries of both.
//
// Owns:    GridOnBothProcessors, BuildGridOnBothProcessors, BuildRandomGridsOnBothProcessors,
//          FindActiveCellPlaces, FindFacePiecePlaces, ReadGpuNameAndRuntime, CheckFreeBytesUnchanged
// Uses:    the headers it includes
// Used by: the GPU tests of tests/grid/ (*.gpu.test.cc)
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/processor_description.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/testing/base_support.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

// One random grid built twice from the same list of active cells.
struct GridOnBothProcessors {
  std::vector<CellId> cells;  // The list, in no particular order.
  TestGrid on_cpu;  // The reference.
  TestGrid on_gpu;  // Built on the GPU from a GPU copy of the list.
};

// Builds one grid twice from a list of active cells, version 1: on the CPU as the reference and on the GPU from a GPU
// copy of the list. A grid that fails to build on either processor fails the calling test and gives no value, so a
// caller checks it.
inline std::optional<GridOnBothProcessors>
BuildGridOnBothProcessors(GridFrame frame, const std::vector<CellId>& cells) {
  const GridInputs inputs = {.frame = frame, .active_cells = cells, .version = {.value = 1}};
  std::expected<TestGrid, GridFailure> on_cpu = BuildTestGrid(inputs);
  std::expected<TestGrid, GridFailure> on_gpu = BuildTestGrid(inputs, {.run_on = RunOn::kGpu});
  EXPECT_TRUE(on_cpu.has_value()) << "the grid on the CPU";
  EXPECT_TRUE(on_gpu.has_value()) << "the grid on the GPU";
  if (!on_cpu.has_value() || !on_gpu.has_value()) {
    return std::nullopt;
  }
  return GridOnBothProcessors{.cells = cells, .on_cpu = std::move(*on_cpu), .on_gpu = std::move(*on_gpu)};
}

// Builds the random grids of the entries 0..grid_count - 1 of BuildShuffledRandomGrids (the seeds 1..grid_count) on
// both processors, version 1, in the frame the random grids share. A grid that fails to build fails the calling test
// and is left out, so a caller checks the count.
inline std::vector<GridOnBothProcessors> BuildRandomGridsOnBothProcessors(int grid_count) {
  const GridFrame frame = {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(kRandomGridSizeLevelCount),
  };
  std::vector<GridOnBothProcessors> grids;
  grids.reserve(static_cast<std::size_t>(grid_count));
  for (int entry = 0; entry < grid_count; ++entry) {
    std::optional<GridOnBothProcessors> grid =
        BuildGridOnBothProcessors(frame, BuildShuffledRandomGrids().at(static_cast<std::size_t>(entry)));
    EXPECT_TRUE(grid.has_value()) << "grid " << entry;
    if (grid.has_value()) {
      grids.push_back(std::move(*grid));
    }
  }
  return grids;
}

// The places of the active cells of a list, in the list's order.
inline std::vector<CellPlaceIndex> FindActiveCellPlaces(const GridTables& tables, const std::vector<CellId>& cells) {
  std::vector<CellPlaceIndex> places;
  places.reserve(cells.size());
  for (const CellId cell : cells) {
    places.push_back(FindCellPlace(tables, cell));
  }
  return places;
}

// The places of every face piece of the tables, in ascending order: the places whose kind is not kNotAPiece.
inline std::vector<FacePlaceIndex> FindFacePiecePlaces(const GridTables& tables) {
  std::vector<FacePlaceIndex> places;
  const std::uint64_t place_count = std::uint64_t{tables.block_count} * static_cast<std::uint64_t>(kFacePlacesPerBlock);
  for (std::uint64_t place = 0; place < place_count; ++place) {
    if (tables.face_piece_kinds[place] != FacePieceKind::kNotAPiece) {
      places.push_back({.value = static_cast<std::uint32_t>(place)});
    }
  }
  return places;
}

// The GPU's name and the CUDA runtime version, for the failure message of a number that the compiler or the card
// decides, such as the stack bytes of a GPU function. The name stands for the architecture: the RTX 5060 Ti is sm_120
// (docs/gpu-runs.md).
// Example: "NVIDIA GeForce RTX 5060 Ti, CUDA runtime 12.8".
inline std::string ReadGpuNameAndRuntime() {
  const ProcessorDescription description = ReadProcessorDescription(RunOn::kGpu);
  return description.model + ", CUDA runtime " + description.runtime;
}

// Checks that the GPU's free bytes are the same before and after `work`, which must take no GPU memory. The check
// stays exact: a tolerance wide enough for another program would hide a small allocation of the grid's own. The free
// bytes are the card's, so a program that takes or gives back GPU memory meanwhile (on a card that drives the display:
// the desktop, a browser) changes them too. A difference fails the calling test with both numbers and the difference,
// and writes the same line to the notes (docs/gpu-runs.md, "A card that also drives the display").
// Example: before 15,000,000,000 and after 14,997,902,848 free bytes: "<work>: the GPU's free bytes went from
//   15000000000 to 14997902848, 2097152 fewer".
inline void CheckFreeBytesUnchanged(std::uint64_t before_bytes, std::uint64_t after_bytes, std::string_view work) {
  if (after_bytes == before_bytes) {
    return;
  }
  const bool has_fewer = after_bytes < before_bytes;
  const std::uint64_t difference = has_fewer ? before_bytes - after_bytes : after_bytes - before_bytes;
  const std::string line = std::string(work) + ": the GPU's free bytes went from " + std::to_string(before_bytes)
      + " to " + std::to_string(after_bytes) + ", " + std::to_string(difference) + (has_fewer ? " fewer" : " more")
      + "; another program using the GPU also changes them";
  WriteResultNote(line);
  ADD_FAILURE() << line;
}

}  // namespace fluidity::grid
