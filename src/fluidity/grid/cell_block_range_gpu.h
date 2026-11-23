// The GPU half of the border search of a block range, private to grid/: what grid/cell_block_range_gpu.cu defines.
//
// Purpose: let the buffer form of FindBorderBlocks (grid/cell_block_range_cpu.cc) find the border of a range and the
//   border so far on the GPU, without including a CUDA header.
//
// Idea: the border is found forward: one thread for each place of the set runs the readers from that place and writes
//   the value 1 into a mark byte for every block it reads (grid/border_block_steps.h); then one thread for each block
//   reads the marks, a prefix sum gives each listed block its place, and the ascending list is written. The CPU half
//   calls one function here for each step and the prefix sum between them. Each function starts GPU functions over
//   plain GPU pointers and counts: a thin grid-stride loop that calls a per-item rule of grid/border_block_steps.h,
//   which a CPU test replays for every thread block and thread. The functions use the default stream, so the GPU runs
//   them, and the prefix sum between them, in the order they are called. Each returns how its start went; an error
//   while a GPU function runs is reported by the next wait, the prefix sum's or the last function's.
//   The count of the border is the only value that comes back: the last function leaves it in the status record, which
//   the CPU half reads once. A bad border so far is the other: its failure key is there too.
//   This header is used only when GPU code is compiled in, so the CPU file names it only under
//   `#if defined(FLUIDITY_HAS_GPU)`. It uses the C++20 subset and includes no CPU-only header.
//
// Stack: the thread that runs the marking functions keeps their results in local arrays (TouchingCells, CellsNearFace,
//   CellCorners and, at kFourCellWidths, a FaceSearch), more than the driver's default of 1,024 B per thread. The
//   compiler gave the largest GPU function of this file, WriteBorderReadMarks, 3,600 bytes of stack and 255 registers
//   per thread on sm_120 (3,048 bytes on sm_86 and sm_89), and no local memory beyond the stack (cuobjdump
//   --dump-resource-usage, release build); the others keep none. kBorderSearchStackBytes is the number that a test
//   pins. BuildBorderSearchBuffers reserves the stack (ReserveBorderSearchStackOnGpu), so a GPU too full for it fails
//   there; BuildBorderSearchStackEstimate (grid/cell_block_range.h) gives the run's memory plan the entry for it.
//
// Owns:    ApplyBorderStatusResetOnGpu, ResetBorderMarksOnGpu, CheckBorderSoFarOnGpu, WriteBorderReadMarksOnGpu,
//          WriteListedMarksOnGpu, WriteBorderBlocksOnGpu, kBorderSearchStackBytes,
//          ComputeLargestBorderSearchStackOnGpu, ComputeResidentThreadsOnGpu, ReserveBorderSearchStackOnGpu
// Uses:    RunOnCallStatus (base/run_on_call.h), BorderWidth (grid/border_block_steps.h), GridBuildStatus
//          (grid/grid_build_status.h), CellBlockIndex, CellBlockRange (grid/grid_ids.h), GridTables
//          (grid/grid_tables.h)
// Used by: grid/cell_block_range_cpu.cc and the gpu tests of grid/; defined in grid/cell_block_range_gpu.cu
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>, <sec-face-search>

#pragma once

#include <cstdint>

#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// Step 1 (a), the reset: one thread writes ComputeResetBuildStatus() to *status (no failure, no count), so no key or
// count of an earlier call takes part.
// Requires: status is GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus ApplyBorderStatusResetOnGpu(GridBuildStatus* status);

// Step 1 (b): one thread for each block sets its mark to 0, so the marks hold only what this call's places write. A
// mark left from an earlier call would add a block to the list silently. block_count 0 launches nothing.
// Requires: marks is GPU memory of block_count bytes or more.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus ResetBorderMarksOnGpu(std::uint8_t* marks, std::uint32_t block_count);

// Step 2: one thread for each entry of the border so far offers ComputeBorderSoFarFailureKey of it to the failure key
// of *status with an integer atomic minimum, so the key is the smallest of the entries whatever the order the threads
// run in. border_so_far_count 0 launches nothing. Requires: border_so_far is GPU memory of border_so_far_count entries;
// status is GPU memory that step 1 reset. Fails: kCallFailed  the GPU function did not start
RunOnCallStatus CheckBorderSoFarOnGpu(
    CellBlockRange range,
    const CellBlockIndex* border_so_far,
    std::uint32_t border_so_far_count,
    std::uint32_t block_count,
    GridBuildStatus* status
);

// Step 3, once for each chunk of the set (ComputeMarkingChunk): one thread for each place of the chunk, which is
// `chunk.set_place_count` places from the set's place `chunk.first_set_place` (kPlacesPerSetBlock for each block of
// `range`, then of the border so far), returns at once when the failure key of *status is set, and otherwise runs
// ApplySetPlaceReadMarks. The threads write only the byte value 1, with plain stores, never a read-modify-write and
// never a read of a mark, so two threads that write one mark at once write the same value to a byte whose neighbours a
// byte store leaves unchanged (CUDA guarantees it); the marks are those of any order of the places, so the chunks may
// run in any split. No thread tests a mark to stop early. The function waits for its launch to end, so the chunk is
// complete when it returns, and step 4 starts after the last chunk. A chunk with no place launches nothing.
// If a chunk fails after earlier chunks marked, the function returns the failure, the caller starts no later chunk,
// and the output is not written: the marks hold a part of the result, but step 4 and step 6 never run, so no list or
// count is written; the next call clears the marks in its step 1. Requires: marks is GPU memory of tables.block_count
// bytes, all 0 at the first chunk; tables, border_so_far and status
//   are GPU memory; the chunks are those of a set of at most kMaxSetBlocks blocks, so every place fits the 32-bit place
//   index (the caller checks ComputeIsSetWithinPlaceLimit before step 1).
// Fails: kCallFailed  a GPU function did not start, or the GPU work failed
RunOnCallStatus WriteBorderReadMarksOnGpu(
    const GridTables& tables,
    CellBlockRange range,
    const CellBlockIndex* border_so_far,
    std::uint32_t border_so_far_count,
    SetPlaceChunk chunk,
    BorderWidth width,
    const GridBuildStatus* status,
    std::uint8_t* marks
);

// Step 4: one thread for each block writes sums[block] = ComputeListedMark(marks, range, block), or 0 when the failure
// key of *status is set, so the prefix sum that follows never adds a word of an earlier call. block_count 0 launches
// nothing. sums[block_count] is the prefix sum's to write.
// Requires: marks and sums are GPU memory, marks of block_count bytes and sums of block_count + 1 words; step 3 has
//   finished.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus WriteListedMarksOnGpu(
    const std::uint8_t* marks,
    CellBlockRange range,
    std::uint32_t block_count,
    const GridBuildStatus* status,
    std::uint32_t* sums
);

// Step 6, after the prefix sum of the listed marks: one thread for each block runs ApplyBorderBlockWriteAt, and thread
// 0 also writes the total, sums[block_count], to status->result_count. Then it waits for the GPU, so an error of any
// function since the prefix sum's wait is reported here. At least one thread runs, so the total is written for a grid
// of no block.
// Requires: sums is GPU memory of block_count + 1 words, the exclusive prefix sums of step 4's words; border_blocks is
//   GPU memory of border_block_room entries; status is GPU memory.
// Fails: kCallFailed  a GPU function did not start, or the GPU work failed
RunOnCallStatus WriteBorderBlocksOnGpu(
    const std::uint32_t* sums,
    std::uint32_t block_count,
    std::uint32_t border_block_room,
    CellBlockIndex* border_blocks,
    GridBuildStatus* status
);

// The local bytes per thread that the stack sentence of this file's header states, and that a test pins: the largest
// local memory any GPU function of this file may keep. Bytes.
inline constexpr std::uint64_t kBorderSearchStackBytes = 3600;

// Reads the largest local memory per thread, in bytes, that the compiler gave any GPU function of this file
// (cudaFuncGetAttributes, localSizeBytes).
// Fails: kCallFailed  a query failed; *bytes_per_thread is then unchanged
RunOnCallStatus ComputeLargestBorderSearchStackOnGpu(std::uint64_t* bytes_per_thread);

// Raises the per-thread stack of the current GPU to at least bytes_per_thread (cudaDeviceSetLimit with
// cudaLimitStackSize), so the first marking launch never has to grow it. The CUDA manuals: the stack "will be
// immediately resized", and a failure to provide the memory returns cudaErrorMemoryAllocation. The limit belongs to
// the GPU's context, so every later GPU function of the process gets at least this stack; the driver keeps it until
// the process ends or the limit is set again. A stack already large enough is left as it is.
// Fails: kOutOfMemory  the GPU could not provide the stack; the error is cleared, so the next GPU call does not see it
//        kCallFailed   another query or the setting failed, or the GPU gave a smaller stack than asked
RunOnCallStatus ReserveBorderSearchStackOnGpu(std::uint64_t bytes_per_thread);

// Reads the threads the current GPU can hold at once: maxThreadsPerMultiProcessor x multiProcessorCount, the number of
// threads the driver keeps a stack for.
// Example: a 36-multiprocessor sm_120 card with 1,536 threads in each gives 55,296.
// Fails: kCallFailed  a query failed; *resident_threads is then unchanged
RunOnCallStatus ComputeResidentThreadsOnGpu(std::uint64_t* resident_threads);

}  // namespace fluidity::grid
