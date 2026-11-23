// The grid's memory estimate: how many bytes a grid needs, array by array, before any buffer exists.
//
// Purpose: let the step add up the memory of its grids and compare it with the memory of the processor before it
//   builds anything.
//
// Idea: every array of a grid version has room for max_cell_blocks cell blocks, so its memory is counted per cell
//   block. Each array is one entry of the estimate, listed in the order the builders take their memory pieces. The
//   five tables are the grid itself, 456 bytes per block. The builder buffers are what a build needs while it runs.
//   The CPU builder needs one array, the cell ids, which it sorts in place. The GPU build also keeps sort keys, two
//   orders and the working buffers of the stable order. Three of those entries hold one more item than the cell
//   places, because a prefix sum writes its total after the last entry. A GPU grid keeps nothing in CPU memory: the
//   only value that comes back from a build on the GPU is its status record.
//   Example: one grid on the CPU takes 972 bytes per block, 456 in the run's memory and 516 in CPU memory. Two
//   grids on the GPU that share one set of builder buffers take 2 x 460 + 1,920 = 2,840 bytes per block.
//
// Per block, from method paper <tab-grid-memory>:
//   tables                            456 bytes, always in the run's memory
//   CPU grid, builder buffers + map   512 + 4 bytes in CPU memory: 972 bytes in all
//   GPU grid, tables and piece map    456 + 4 = 460 bytes for each grid
//   GPU builder buffers               512 + 512 + 256 + 256 + 64 x 6 = 1,920 bytes once, plus 4 + 4 + (4,096 + 6)
//                                     = 4,110 bytes once, plus one build status record for each grid
//   The grid's build and check have no entry held by the driver (MemoryWhere::kDriverStack): their GPU functions keep
//   under the driver's default stack of 1,024 bytes per thread. The border search has one, which
//   BuildBorderSearchStackEstimate (grid/cell_block_range.h) gives. A module that computes nearby-face weights on the
//   GPU declares its own. The driver keeps one stack, sized for the largest need of any GPU function launched, so
//   adding this entry to another module's driver-stack entry counts more than the driver takes.
//
// Words:
//   memory estimate  the list of entries of a module or a run: every array it will allocate
//   entry            one line of a memory estimate: one array (or a group of identical arrays) and its size
//   cell block       4 x 4 x 4 cells kept together
//   builder buffers  the temporary arrays a grid build uses
//   piece map        for each block of a grid piece, the index of the same block in the whole grid; 4 bytes per block
// Owns:    ToMemoryCounts, BuildGridMemoryEstimate, BuildBorderSearchMemoryEstimate
// Uses:    CPU-only guard (base/cpu_only.h), MemoryCounts, MemoryEstimateEntry (base/memory_count.h), RunOn
//          (base/run_on.h), GridCounts, BuilderBuffersSharing (grid/staggered_grid.h)
// Implementation uses: ComputeOrderBuffersSize (base/stable_order.h), the entry names and item bytes
//          (grid/grid_buffers_internal.h), GridBuildStatus (grid/grid_build_status.h)
// Used by: the step's memory budget, and the grid's buffer builders and BuildBorderSearchBuffers, which take one memory
//          piece for each entry
// Read:    docs/method-paper/fluidity-method.typ <sec-grid-memory>, <tab-grid-memory>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <vector>

#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {

// Converts the counts of a grid to the counts of items a memory estimate asks for.
//
// Method: cell_places = 64, face_places = 240 and corner_places = 125 times counts.cell_blocks; the block count and
//   `particles` are copied.
// Example: 9 blocks give 576 cell places, 2,160 face places and 1,125 corner places.
// Read: docs/method-paper/fluidity-method.typ <sec-grid-memory>
MemoryCounts ToMemoryCounts(const GridCounts& counts, std::uint64_t particles);

// Builds the memory estimate of `grid_count` sets of grid buffers of max_cell_blocks cell blocks on `run_on`, with
// their builder buffers: one entry per array, in the order the builders take their memory pieces.
//
// Method: every entry is KeptFor::kWholeRun, its name starts with "grid: ", and its copies is grid_count. With
//   kOneForAllGrids the entries named "builder buffers: ..." have copies = 1, because one set is lent to every grid.
//   Entries, in this order:
//   1 The five tables, MemoryWhere::kRunOn: block ids (8 bytes per block), same-size-level neighbour blocks (108),
//     bigger and smaller blocks (36: one bigger and 8 smaller blocks, laid out as ComputeSmallerBlocksStart of
//     grid/grid_buffers_internal.h states), cell states (1 byte per cell place), face piece kinds (1 byte per face
//     place). 456 bytes per block.
//   2 On the CPU: builder buffers: cell ids (8 bytes per cell place) and piece map (4 per block), both
//     MemoryWhere::kCpu.
//   3 On the GPU, all MemoryWhere::kRunOn: builder buffers: cell ids (8 per cell place), builder buffers: sort keys
//     (8 per cell place), builder buffers: order and builder buffers: second order (each 4 x (64 x max_cell_blocks
//     + 1) bytes), builder buffers: order buffers (bytes_per_item x (64 x max_cell_blocks + 1) + fixed_bytes of
//     ComputeOrderBuffersSize(RunOn::kGpu)), piece map (4 per block), build status (one GridBuildStatus).
//   An array that has a per-item part and a fixed part is one entry of its exact bytes (kGivenCount, 1 byte each),
//   so ComputeRunMemoryBytes equals the bytes the builders hand out for any room.
// Units: bytes, through ComputeMemoryBytes.
// Example: on the GPU at 1,000 blocks, ComputeMemoryBytes of the estimate with MemoryWhere::kRunOn gives
//   1,000 x (456 + 1,536 + 4 + 64 x 6) + 4,096 + 6 + 8 + 192 = 2,384,302 bytes, and 0 with MemoryWhere::kCpu. On
//   the CPU at 1,000 blocks it gives 456,000 with kRunOn and 516,000 with kCpu.
// Requires: MemoryCounts::cell_blocks equals max_cell_blocks when the entries are summed, for the entries counted
//   per block or place. grid_count is the number of sets of grid buffers. With grid_count 0 and kOnePerGrid every entry
//   counts nothing; with kOneForAllGrids the builder buffers entries still count their one set, whatever grid_count is.
// Fails: never.
// Read: docs/method-paper/fluidity-method.typ <sec-grid-memory>, <tab-grid-memory>
// clang-tidy: a count of grids and a count of blocks; the swappable-parameters check flags the two integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::vector<MemoryEstimateEntry> BuildGridMemoryEstimate(
    RunOn run_on,
    std::uint32_t grid_count,
    std::uint32_t max_cell_blocks,
    BuilderBuffersSharing builder_buffers_sharing = BuilderBuffersSharing::kOnePerGrid
);
// NOLINTEND(bugprone-easily-swappable-parameters)

// Builds the memory estimate of one BorderSearchBuffers for grids of at most max_cell_blocks blocks on `run_on`: one
// entry per array, in the order BuildBorderSearchBuffers takes its memory pieces.
//
// Method: every entry is KeptFor::kWholeRun, MemoryWhere::kRunOn, copies 1, counted as a kGivenCount entry of exact
//   bytes (bytes_each 1), and its name starts with "grid: ". Entries, in this order:
//   1 border marks: max_cell_blocks bytes, one std::uint8_t per block, on both processors.
//   On the GPU also:
//   2 border sums: 4 x (max_cell_blocks + 1) bytes, one 0/1 word per block, then their exclusive prefix sum in place,
//     the total one past the last.
//   3 border order buffers: bytes_per_item x (max_cell_blocks + 1) + fixed_bytes of
//     ComputeOrderBuffersSize(RunOn::kGpu), the working buffers of that prefix sum.
//   4 border status: one GridBuildStatus.
// Units: bytes, through ComputeMemoryBytes.
// Example: on the CPU at 1,000 blocks, one entry of 1,000 bytes. On the GPU at 1,000 blocks, the kRunOn sum is
//   1,000 + 4,004 + 6 x 1,001 + 4,096 + 192 = 15,298 bytes.
// Fails: never.
// Read: docs/method-paper/fluidity-method.typ <sec-grid-memory>, <tab-grid-memory>
std::vector<MemoryEstimateEntry> BuildBorderSearchMemoryEstimate(RunOn run_on, std::uint32_t max_cell_blocks);

}  // namespace fluidity::grid
