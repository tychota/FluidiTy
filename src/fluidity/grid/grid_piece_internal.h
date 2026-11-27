// The host checks and the status reading of a grid piece's build, for the grid module's own code and tests.
//
// Purpose: name the two host steps that every build of a grid piece shares, so a test runs them on the CPU after a
//   replay of the GPU steps, and a build on the GPU runs the very same code.
//
// Idea: a build of a grid piece on the GPU is wrapped by host code. Before the first launch the host checks the range
//   against the whole grid and the room (CheckPieceRangeAndRoom). After the last launch it reads one status
//   and turns it into the piece's counts and the start of the range, or into an error (ToPieceBuildSummary). Neither
//   needs a GPU, so both are compiled in every build and a test replays the launches in between on the CPU.
//   Example: a status with no failure key, 3 blocks loaded, 2 active cells, 5 face pieces and 1 border
//   block below the range gives the summary counts {3 blocks, 2 active cells, 5 face pieces} and range start 1.
//
// Words:
//   face piece  the part of a face between two cells or at the grid's edge, kept at the size level of its smaller
//               cell. Not a grid piece, not a memory piece
// Owns:    PieceBuildSummary, CheckPieceRangeAndRoom, ToPieceBuildSummary
// Uses:    CPU-only guard (base/cpu_only.h), GridBuildStatus (grid/grid_build_status.h), GridFailure
//          (grid/grid_error.h), CellBlockRange (grid/grid_ids.h), GridCounts (grid/staggered_grid.h)
// Implementation uses: ToFailureStep (grid/grid_build_steps.h), kPieceBorderStep (grid/grid_piece_steps.h)
// Used by: grid/grid_piece_cpu.cc; the tests of grid pieces
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <expected>

#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {

// What a build of a grid piece found: the counts of its tables and where its range starts in the piece.
struct PieceBuildSummary {
  GridCounts counts;  // The grid counts of the piece: its blocks, active cells and face pieces.
  std::uint32_t range_start_in_piece;  // Border blocks below the range, so the index of the range's first block.
};

// Checks that the range lies inside the whole grid's blocks, then the room: the blocks of the range and of the border
// list against the buffers' room. Every build of a piece makes these two checks first, the GPU build
// before its first launch. An empty range is bad input unless the whole grid has no block. Changes nothing.
// Example: a whole grid of 10 blocks, range {8, 3}, any border: kBadInput. Range {4, 3}, 5 border blocks and a room of
//   7: kTooManyCellBlocks.
// Fails: kBadInput           the range is empty (block_count 0) while the whole grid has blocks, or ends past the last
//                            block of the whole grid.
//        kTooManyCellBlocks  the range's blocks plus the border blocks are more than max_cell_blocks. No cell is named,
//                            since the caller raises the room.
//        A call that fails in both ways reports kBadInput.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
std::expected<void, GridFailure> CheckPieceRangeAndRoom(
    std::uint32_t whole_block_count,
    CellBlockRange range,
    std::uint64_t border_block_count,
    std::uint32_t max_cell_blocks
);

// Converts the status that the launches of a build left to the piece's counts and the start of the range, or
// to the failure. Changes nothing.
// Example: a status whose key is of step 1 gives kBadInput; one whose key is of step 2 gives kInternalFault.
// Requires: the status is that of a build of loaded_count blocks with a border list of border_block_count entries.
// Fails: kBadInput       the key is of the border step: the caller's border list is not ascending, repeats a block,
//                        goes beyond the grid or holds a block of the range.
//        kInternalFault  a key of any other step, or a count above what the loaded blocks' places hold, or a range
//                        start above the border's entries: the status contradicts the steps that wrote it, and no input
//                        causes it.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
std::expected<PieceBuildSummary, GridFailure>
ToPieceBuildSummary(const GridBuildStatus& status, std::uint32_t loaded_count, std::uint32_t border_block_count);

}  // namespace fluidity::grid
