// The settings and counts of a grid, and how its builder buffers are shared.
//
// Purpose: declare the values a caller chooses before it builds a grid (the room and the processor), the counts a
//   built grid reports, and the choice of how many sets of builder buffers a run keeps.
//
// Idea: a grid lives in arrays with room for a fixed number of cell blocks, chosen once for the run. Building a
//   grid also needs temporary arrays, the builder buffers. They hold nothing between builds, so two grids that
//   are never built at the same time can lend one set of them to each other. The memory estimate of the grid
//   (grid/grid_memory_estimate.h) counts them once or once per grid, as BuilderBuffersSharing says.
//   Example: the step keeps a committed grid and a try grid. With kOneForAllGrids they share one set of builder
//   buffers, and the estimate counts that set once.
//
// Words:
//   cell block       4 x 4 x 4 cells kept together
//   builder buffers  the temporary arrays a grid build uses: cell ids, sort keys, two orders and the working
//                    buffers of the stable order
//   room             the number of cell blocks an array has memory for: max_cell_blocks
// Owns:    kMaxCellBlocksLimit, GridSettings, GridCounts, BuilderBuffersSharing
// Uses:    CPU-only guard (base/cpu_only.h), RunOn (base/run_on.h), kMaxSizeLevels (grid/grid_ids.h)
// Used by: the grid's builders and memory estimate, and the step, which chooses the room
// Read:    docs/method-paper/fluidity-method.typ <sec-grid-memory>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstdint>

#include "fluidity/base/run_on.h"
#include "fluidity/grid/grid_ids.h"

namespace fluidity::grid {

// The largest max_cell_blocks: floor((2^32 - 1) / 240) = 17,895,697. The face places bind: with 240 per block, every
// face place index fits 32 bits and stays below kNoIndex (240 x 17,895,697 = 4,294,967,280). The cell places, 64 per
// block, then fit the 32-bit orders and prefix sums of base/, and the corner places, 125 per block, fit 32 bits too.
inline constexpr std::uint32_t kMaxCellBlocksLimit = 17895697;
static_assert(std::uint64_t{kFacePlacesPerBlock} * kMaxCellBlocksLimit <= kNoIndex, "face places fit 32 bits");
static_assert(std::uint64_t{kCornerPlacesPerBlock} * kMaxCellBlocksLimit <= kNoIndex, "corner places fit 32 bits");
static_assert(std::uint64_t{kCellPlacesPerBlock} * kMaxCellBlocksLimit <= kNoIndex, "cell places fit 32 bits");
static_assert(std::uint64_t{kFacePlacesPerBlock} * (kMaxCellBlocksLimit + 1) > kNoIndex, "the largest such room");

// What a caller chooses once for a grid's buffers.
struct GridSettings {
  // Room, 1..kMaxCellBlocksLimit. The default 0 is no room: a build that takes these settings fails with kBadInput.
  // A run's room comes from its plan, never from a constant.
  std::uint32_t max_cell_blocks = 0;
  RunOn run_on = RunOn::kCpu;  // The caller's choice. Never changed silently.
};

// What a built grid knows about its size, kept on the CPU. Everything the memory estimate needs from a grid.
struct GridCounts {
  std::uint32_t cell_blocks;
  std::uint64_t active_cells_per_size_level[kMaxSizeLevels];  // NOLINT(modernize-avoid-c-arrays): as GridBuildStatus.
  std::uint64_t active_cells;
  std::uint64_t face_pieces;  // Places whose kind is not kNotAPiece.
};

// How many sets of builder buffers the "builder buffers" entries of BuildGridMemoryEstimate count.
enum class BuilderBuffersSharing : std::uint8_t {
  kOnePerGrid,  // One set of builder buffers for each of the grid_count sets of grid buffers.
  kOneForAllGrids,  // One set lent to all of them.
};

}  // namespace fluidity::grid
