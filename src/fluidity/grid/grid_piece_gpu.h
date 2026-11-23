// The GPU half of the build of a grid piece, private to grid/: what grid/grid_piece_gpu.cu defines.
//
// Purpose: let the buffer form of BuildGridPiece (grid/grid_piece_cpu.cc) build the piece map and the tables of a grid
//   piece of a GPU whole grid on the GPU, without including a CUDA header.
//
// Idea: the border list is already ascending and every loaded item has a place of its own (ToPiecePlace), so the build
//   needs no sort and no prefix sum: six GPU functions, each a thin grid-stride loop that calls a per-item rule of
//   grid/grid_piece_steps.h, which a CPU test replays for every thread block and thread. Function 1 resets the status
//   record. Function 2 checks each border entry, one thread for each entry. Function 3 writes the piece map, one thread
//   for each loaded item. Function 4 copies the block id and translates the 36 neighbour entries, one thread for each
//   piece block. Function 5 copies the cell states, one thread for each cell place of the piece, and counts the active
//   cells of each size level. Function 6 writes the face piece kinds, one thread for each face place, and counts the
//   face pieces. The functions use the default stream, so each starts after the one before has finished and reads only
//   what an earlier one wrote: function 3 the border list, function 4 the piece map, functions 5 and 6 the piece map
//   and the whole grid's tables. After a failure the later functions do nothing. The CPU half calls one function here
//   for each step, then reads the status record once. An error while a GPU function runs is reported by the wait at
//   the end of the last function.
//   The threads share only the status record: its failure key, to which they offer keys with an integer atomic
//   minimum, and its counts, to which they add with integer atomic adds, so neither the smallest key nor a count
//   depends on the order the threads run in. No two threads of a launch write one place: function 3 writes the place
//   ToPiecePlace gives each item, and no two items have one place; functions 4 to 6 write the arrays of their own piece
//   block, cell place or face place. No thread reads what another thread of its launch writes, except the failure key,
//   which functions 3 and 4 read only for a key of function 2 (ComputeHasEarlierStepFailed ignores a key of the
//   thread's own step, so a key another thread of function 4 offers changes nothing). This header is used only when GPU
//   code is compiled in, so the CPU file names it only under
//   `#if defined(FLUIDITY_HAS_GPU)`. It uses the C++20 subset and includes no CPU-only header.
//
// Stack: every function here keeps under the driver's default of 1,024 B per thread, so a piece build has no entry held
//   by the driver and no reserve step. The compiler gave every GPU function of this file 0 bytes of stack and no spill,
//   and at most 40 registers per thread, on sm_86, sm_89 and sm_120 (the ptxas lines of the gate's cuda build,
//   release). ComputeLargestGridPieceStackOnGpu reads the bytes the compiler gave, and a test pins them at 1,024 or
//   less.
//
// Owns:    ApplyPieceStatusResetOnGpu, CheckPieceBorderOnGpu, WritePieceMapOnGpu, CopyPieceBlocksOnGpu,
//          CopyPieceCellStatesOnGpu, WritePieceFaceKindsOnGpu, ComputeLargestGridPieceStackOnGpu
// Uses:    RunOnCallStatus (base/run_on_call.h), GridBuildStatus (grid/grid_build_status.h), PieceLoad, PieceBlocks,
//          PieceTableArrays (grid/grid_piece_steps.h), CellState, FacePieceKind, GridTables (grid/grid_tables.h)
// Used by: grid/grid_piece_cpu.cc and the gpu tests of grid/; defined in grid/grid_piece_gpu.cu
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include <cstdint>

#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_piece_steps.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// Function 1, the reset: one thread writes ComputeResetBuildStatus() to *status (no failure, every count zero), so no
// key or count of an earlier call takes part. Requires: status is GPU memory. Fails: kCallFailed  the GPU function did
// not start
RunOnCallStatus ApplyPieceStatusResetOnGpu(GridBuildStatus* status);

// Function 2: one thread for each border entry offers ComputePieceBorderFailureKey of it to the failure key of *status
// with an integer atomic minimum, and adds 1 to status->result_count when the entry lies below the range
// (ComputeIsBorderEntryBelowRange), so after a good list result_count is the first block of the range in the piece.
// Both results are the same whatever the order of the threads. load.border_block_count 0 launches nothing.
// Requires: load.border_blocks is GPU memory of load.border_block_count entries; status is GPU memory that launch 1
//   reset.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus CheckPieceBorderOnGpu(PieceLoad load, std::uint32_t whole_block_count, GridBuildStatus* status);

// Function 3: one thread for each loaded item (load.range.block_count + load.border_block_count of them) writes the
// piece map entry of its item with ApplyPieceMapWriteAt, or returns at once when the failure key of *status is set, so
// a bad border list writes nothing. Nothing is launched for 0 items. Requires: map is GPU memory of that many entries;
// load.border_blocks is GPU memory; status is GPU memory that launch
// 2
//   has finished.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus WritePieceMapOnGpu(PieceLoad load, const GridBuildStatus* status, CellBlockIndex* map);

// Function 4: one thread for each piece block copies the block id and translates the 36 neighbour entries with
// ApplyPieceBlockCopyAt, and offers the failure key of a fault to *status with an integer atomic minimum. A thread
// returns at once when the key is set. Nothing is launched for 0 blocks.
// Requires: whole is the tables of the whole grid, GPU memory; blocks.map is the piece map function 3 wrote,
// blocks.count
//   entries; `piece` is GPU memory with room for blocks.count blocks; status is GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus
CopyPieceBlocksOnGpu(const GridTables& whole, PieceBlocks blocks, PieceTableArrays piece, GridBuildStatus* status);

// Function 5: one thread for each cell place of the piece (64 for each block) copies the state with
// ApplyPieceCellStateCopyAt and, for an active cell, adds 1 to the active cells of its size level in *status with an
// integer atomic add. A thread returns at once when the failure key is set. Nothing is launched for 0 blocks.
// Requires: as function 4; cell_states is GPU memory of 64 entries for each piece block; blocks.count is at most
//   kMaxCellBlocksLimit (grid/staggered_grid.h), so every cell place fits 32 bits.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus
CopyPieceCellStatesOnGpu(const GridTables& whole, PieceBlocks blocks, CellState* cell_states, GridBuildStatus* status);

// Function 6: one thread for each face place of the piece (240 for each block, in a grid-stride loop) writes the kind
// with ApplyPieceFaceKindWriteAt; a thread adds up its face pieces and adds the sum to status->face_pieces with one
// integer atomic add. A thread returns at once when the failure key is set. Then the function waits for the GPU, so an
// error of any function since the call began is reported here. Nothing is launched for 0 blocks. The launch is not cut
// into chunks: a thread does a bounded amount of work (two reads of the whole grid's tables to find the two cells of a
// face piece and two binary searches of the piece map), so a launch over the most blocks a room may hold, 17,895,697,
// is a launch like the grid build's own over the face places of its room. Requires: as function 4; face_piece_kinds is
// GPU memory of 240 entries for each piece block; blocks.count is at most
//   kMaxCellBlocksLimit, so every face place fits 32 bits and is below kNoIndex.
// Fails: kCallFailed  a GPU function did not start, or the GPU work failed
RunOnCallStatus WritePieceFaceKindsOnGpu(
    const GridTables& whole,
    PieceBlocks blocks,
    FacePieceKind* face_piece_kinds,
    GridBuildStatus* status
);

// Reads the largest local memory per thread, in bytes, that the compiler gave any GPU function of this file
// (cudaFuncGetAttributes, localSizeBytes).
// Fails: kCallFailed  a query failed; *bytes_per_thread is then unchanged
RunOnCallStatus ComputeLargestGridPieceStackOnGpu(std::uint64_t* bytes_per_thread);

}  // namespace fluidity::grid
