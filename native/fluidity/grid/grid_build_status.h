// The record a grid build on the GPU reports its status in.
//
// Purpose: declare the one record that GPU code writes during a grid build and the CPU reads back, so a build on the
//   GPU returns to the CPU only this record and no table.
//
// Idea: a build on the GPU checks, sorts and numbers the cells with many threads at once. Every thread that finds a
//   problem offers one integer, its failure key, and the GPU keeps the smallest. The key is (step, position, detail)
//   packed into 64 bits with the step in the high bits, so the smallest key is the failure of the earliest check, at
//   the earliest position: the same failure whatever order the threads run in, and the one the CPU build reports,
//   since both processors compute the same key. The input checks (the frame, the version, the shape and the length
//   of the list) come before every keyed check, so they never take part in the minimum.
//   The record also carries the counts of a successful build, which later GPU functions read as their own counts.
//   The CPU reads the whole record back with one copy. Its size is an entry of the grid's memory estimate
//   (grid/grid_memory_estimate.h), so the layout is checked here.
//   Example: a failure of check 2 (overlaps) at place 5 of the fixed cell order, detail 0, has the key
//   (2 << 56) | (5 << 8) | 0 = 144115188075857152, below the key of any failure of check 3 or later.
//
// Words:
//   failure key   the number (step, position, detail) whose smallest value names the first failure of a build
//   cell block    4 x 4 x 4 cells kept together
//   face piece    the part of a face between two cells or at the grid's edge, kept at the size level of its smaller
//                 cell. Not a grid piece, not a memory piece
// Owns:    kNoFailureKey, GridBuildStatus
// Uses:    GridError (grid/grid_error.h), CellId, kMaxSizeLevels (grid/grid_ids.h)
// Used by: the grid's builders and checks on the GPU, and the grid's memory estimate
// Read:    docs/method-paper/fluidity-method.typ <sec-parallel-build>, <sec-grid-memory>

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"

namespace fluidity::grid {

// The failure key of a build that has not failed: the largest 64-bit number, so every real key is smaller.
inline constexpr std::uint64_t kNoFailureKey = ~std::uint64_t{0};

// What one grid operation on the GPU returns to the CPU. Written by GPU code, read back byte for byte.
struct GridBuildStatus {
  // kNoFailureKey, or the smallest failure key: (step << 56) | (position << 8) | detail. The step takes the high 8
  // bits: the number of the check, 1 the ids, 2 overlaps, 3 siblings, 4 the size step rule, 5 the active and split
  // cells against the room, 6 the blocks against the room, 7 a block without its bigger block. The position takes 48
  // bits: the index in the input list for step 1, the place in the fixed cell order for steps 2 to 4, 0 for steps 5 and
  // 6, and the block's index for step 7. The detail takes the low 8 bits: which sibling is missing, or in which of the
  // 26 directions the size step rule breaks.
  std::uint64_t failure_key;
  GridError error;  // kNone on success; else the failure's error.
  CellId cell;  // GridFailure::cell. CellId{0} when unused.
  // GridFailure::other_cell: the second cell of an overlap or of a broken size step rule, or the big cell of a missing
  // sibling cell. CellId{0} when unused.
  CellId other_cell;
  std::uint32_t cell_blocks;  // Cell blocks built, of a grid or a grid piece. Later GPU functions read it as a count.
  std::uint32_t split_cells;  // Split cells the build added. Later GPU functions read it; the CPU build ignores it.
  std::uint32_t result_count;  // Entries a search for border blocks found. 0 for every other operation.
  std::uint64_t active_cells;  // Active cells.
  // Active cells of each size level.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, so GPU code and the CPU read one layout.
  std::uint64_t active_cells_per_size_level[kMaxSizeLevels];
  std::uint64_t face_pieces;  // Places whose kind is not kNotAPiece.
};

// The GPU writes the record and the CPU reads it back byte for byte, so both compilers must lay it out alike.
static_assert(std::is_trivially_copyable_v<GridBuildStatus>, "a copy between processors moves the record's bytes");
static_assert(sizeof(GridBuildStatus) == 192, "the memory estimate counts 192 bytes for the build status");
static_assert(offsetof(GridBuildStatus, error) == 8, "the CPU and the GPU read one layout of GridBuildStatus");
static_assert(offsetof(GridBuildStatus, cell) == 16, "the CPU and the GPU read one layout of GridBuildStatus");
static_assert(offsetof(GridBuildStatus, other_cell) == 24, "the CPU and the GPU read one layout of GridBuildStatus");
static_assert(offsetof(GridBuildStatus, cell_blocks) == 32, "the CPU and the GPU read one layout of GridBuildStatus");
static_assert(offsetof(GridBuildStatus, split_cells) == 36, "the CPU and the GPU read one layout of GridBuildStatus");
static_assert(offsetof(GridBuildStatus, result_count) == 40, "the CPU and the GPU read one layout of GridBuildStatus");
static_assert(offsetof(GridBuildStatus, active_cells) == 48, "the CPU and the GPU read one layout of GridBuildStatus");
static_assert(
    offsetof(GridBuildStatus, active_cells_per_size_level) == 56,
    "the CPU and the GPU read one layout of GridBuildStatus"
);
static_assert(offsetof(GridBuildStatus, face_pieces) == 184, "the CPU and the GPU read one layout of GridBuildStatus");

}  // namespace fluidity::grid
