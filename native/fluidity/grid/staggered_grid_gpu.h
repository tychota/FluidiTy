// The GPU half of the check of a cell list and of the first steps of a build, private to grid/: what
// grid/staggered_grid_gpu.cu defines.
//
// Purpose: let grid/grid_cell_checks.cc run method paper <proc-build-grid-in-parallel> steps P1 to P6, and the failure
//   half of P12, and let the grid's builder run P7 and P8, without including a CUDA header.
//
// Idea: the CPU half has run the input checks of P1 and calls one function here per step, with the stable orders of P3
//   between them. Each function starts one GPU function over plain GPU pointers and counts: a thin grid-stride loop
//   that calls a per-item rule of grid/grid_build_steps.h for each item, which a CPU test replays for every thread
//   block and thread. The threads share only the status record's failure key, to which they offer keys with an integer
//   atomic minimum, so the smallest key does not depend on the order the threads run in. The functions use the default
//   stream, so the GPU runs them, and the stable orders between them, in the order they are called. Each returns how
//   its start went. An error while a GPU function runs is reported by the next wait: the wait at the end of each
//   BuildStableOrder for the functions before the copy in sorted order, as BufferError kRunOnCallFailed, or the wait at
//   the end of WriteFailureCellsOnGpu for the rest. Either way the check fails with kRunOnCallFailed.
//   The functions of P7 and P8 read counts that only the GPU knows, the split cells and the blocks, from the status
//   record, so no count comes back during a build. After an earlier step's failure each does nothing, except where a
//   prefix sum reads what it writes: there it writes 0, so the sum never adds words of an earlier launch. Both keys of
//   P7 (b), the room's and kSortCountFaultKey, are of step 5, so every later function stops on either: P8 never sorts
//   a list cut short.
//   This file exists only in a build with GPU code, so grid/grid_cell_checks.cc names it only under
//   `#if defined(FLUIDITY_HAS_GPU)`. It uses the C++20 subset and includes no CPU-only header.
//
// Stack: every function here keeps under the driver's default of 1,024 B per thread, so the grid has no entry held by
//   the driver. The GPU functions keep no local array: a few indexes, a failure key, a CellCheck (24 bytes) and the
//   coordinates of the checks of grid/grid_build_steps.h. ComputeLargestStackOnGpu reads the bytes the compiler gave.
//
// Owns:    ApplyBuildStatusResetOnGpu, CheckCellIdsOnGpu, WriteCellSortKeysOnGpu, CopySortedCellsOnGpu,
//          CheckSortedCellsOnGpu, WriteFailureCellsOnGpu, WriteSplitCellCountsOnGpu, CheckCellPlacesOnGpu,
//          CheckBlockCountOnGpu, WriteSplitCellsOnGpu, WriteSortTailOnGpu, WriteBlockSortKeysOnGpu,
//          WriteBlockStartMarksOnGpu, WriteBlockIdsOnGpu, ComputeLargestStackOnGpu
// Uses:    RunOnCallStatus (base/run_on_call.h), GridBuildStatus (grid/grid_build_status.h), CellId, CellBlockId
//          (grid/grid_ids.h), GridFrame (grid/grid_tables.h)
// Used by: grid/grid_cell_checks.cc, grid/grid_cell_checks_gpu_test.cc and the grid's builder; defined in
//          grid/staggered_grid_gpu.cu
// Read:    docs/method-paper/fluidity-method.typ <proc-build-grid-in-parallel>, <sec-parallel-build>

#pragma once

#include <cstdint>

#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// Step P1, the reset: one thread writes ComputeResetBuildStatus() to *status (no failure, every count zero), so no key
// or count of an earlier call takes part. Nothing is written from the CPU.
// Requires: status is GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus ApplyBuildStatusResetOnGpu(GridBuildStatus* status);

// Step P2: one thread per input index i offers ComputeCellIdFailureKey(frame, input_cells, i) to status->failure_key
// with an integer atomic minimum. input_count 0 starts nothing.
// Requires: input_cells holds input_count ids and status one record, both GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus
CheckCellIdsOnGpu(GridFrame frame, const CellId* input_cells, std::uint32_t input_count, GridBuildStatus* status);

// Step P3, one pass of the keys: one thread per input index i writes ToCellSortKey(input_cells[i], pass) to
// sort_keys[i]. Every one of the input_count keys is written, so the stable order that follows reads no key of an
// earlier call. input_count 0 starts nothing.
// Requires: pass is 0, 1 or 2; input_cells and sort_keys hold input_count values, both GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus
WriteCellSortKeysOnGpu(const CellId* input_cells, std::uint32_t input_count, int pass, std::uint64_t* sort_keys);

// Step P3, the copy in sorted order: one thread per sorted position j writes input_cells[order[j]] to sorted_cells[j].
// count 0 starts nothing.
// Requires: order is a permutation of 0..count - 1; the three arrays hold count values, are GPU memory and do not
//   overlap one another.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus
CopySortedCellsOnGpu(const CellId* input_cells, const std::uint32_t* order, std::uint32_t count, CellId* sorted_cells);

// Steps P4 to P6, one of them: one thread per sorted position j first reads status->failure_key and does nothing when
// ComputeHasEarlierStepFailed(key, step); otherwise it offers ComputeSortedCellFailureKey(sorted_cells, count, j, step)
// with an integer atomic minimum. A key of this step offered by another thread does not stop a thread, so the smallest
// key of the step is kept whatever the order of the threads. count 0 starts nothing.
// Requires: step is kOverlapStep, kSiblingStep or kSizeStepRuleStep (grid/grid_build_steps.h); sorted_cells holds count
//   ids in the fixed order; both pointers are GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus
CheckSortedCellsOnGpu(const CellId* sorted_cells, std::uint32_t count, int step, GridBuildStatus* status);

// Step P12, the failure half: one thread runs ApplyFailureCells, which writes status->error, cell and other_cell from
// the smallest key (kNone and zero cells without a failure). Then the CPU waits for the GPU.
// Requires: the lists are the ones P2 to P6 checked, GPU memory; status is GPU memory.
// Fails: kCallFailed  the GPU function did not start, or a GPU function started since the last wait (the copy in
//                     sorted order, P4 to P6 or this one) failed while it ran; the record is then unspecified. An
//                     earlier one's failure is reported by the wait of a BuildStableOrder first.
RunOnCallStatus WriteFailureCellsOnGpu(
    GridFrame frame,
    const CellId* input_cells,
    std::uint32_t input_count,
    const CellId* sorted_cells,
    std::uint32_t count,
    GridBuildStatus* status
);

// Step P7 (a): one thread per sorted active cell j writes ComputeSplitCellCountToAdd(cell_ids[j]) to
// split_cell_counts[j], or 0 when ComputeHasEarlierStepFailed(status->failure_key, kCellPlacesStep). Every one of the
// active_count words is written, so the prefix sum that follows adds no word of an earlier launch. active_count 0
// starts nothing.
// Requires: cell_ids holds active_count sorted active cells; split_cell_counts holds active_count words and does not
//   overlap cell_ids; all GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus WriteSplitCellCountsOnGpu(
    const CellId* cell_ids,
    std::uint32_t active_count,
    const GridBuildStatus* status,
    std::uint32_t* split_cell_counts
);

// Step P7 (b): one thread reads the total sums[active_count] of the counts' prefix sum into status->split_cells, then
// offers ComputeCellPlacesFailureKey(active_count + split_cells, room_cell_places, sort_count): the room's key of step
// 5 when the cells pass room_cell_places = 64 max_cell_blocks, else kSortCountFaultKey, step 5 with detail 1, when
// they pass sort_count = ComputeSortCount(active_count, max_cell_blocks), which no valid list does. After an earlier
// failure the counts were 0, so split_cells is 0; P1 checked active_count <= 64 max_cell_blocks, and the sort count is
// at least active_count, so no key is offered.
// Requires: sums holds active_count + 1 words; both pointers are GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus CheckCellPlacesOnGpu(
    const std::uint32_t* sums,
    std::uint32_t active_count,
    std::uint64_t room_cell_places,
    std::uint32_t sort_count,
    GridBuildStatus* status
);

// Step P8, the block count: one thread reads the total sums[sort_count] of the marks' prefix sum into
// status->cell_blocks, then offers ComputeRoomFailureKey(cell_blocks, max_cell_blocks, kBlockCountStep). After an
// earlier failure the marks were 0, so cell_blocks is 0 and no key is offered.
// Requires: sums holds sort_count + 1 words; both pointers are GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus CheckBlockCountOnGpu(
    const std::uint32_t* sums,
    std::uint32_t sort_count,
    std::uint32_t max_cell_blocks,
    GridBuildStatus* status
);

// Step P7 (c): one thread per sorted active cell j runs ApplySplitCellWritesAt(cell_ids, sums, active_count, j), which
// writes the split cells j adds after the active cells. Nothing is written when ComputeHasEarlierStepFailed(key,
// kBlockCountStep): after a failure of step 5 the places could lie past the cell ids. active_count 0 starts nothing.
// Requires: sums holds the exclusive prefix sums of P7 (a); cell_ids has room for the active and split cells; both are
//   GPU memory and do not overlap.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus WriteSplitCellsOnGpu(
    CellId* cell_ids,
    const std::uint32_t* sums,
    std::uint32_t active_count,
    const GridBuildStatus* status
);

// Step P7 (d), the tail fill: one thread per place p below sort_count, ComputeSortCount(active_count, max_cell_blocks),
// runs ApplySortTailFillAt(cell_ids, active_count + status->split_cells, p), so every place after the active and split
// cells holds CellId{0} and no cell of an earlier build becomes a block. Nothing is written when
// ComputeHasEarlierStepFailed(key, kBlockCountStep). sort_count 0 starts nothing.
// Requires: cell_ids holds sort_count places; P7 (b) wrote status->split_cells; GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus WriteSortTailOnGpu(
    CellId* cell_ids,
    std::uint32_t active_count,
    std::uint32_t sort_count,
    const GridBuildStatus* status
);

// Step P8, one pass of the block keys: one thread per place p below sort_count writes ToBlockSortKey(cell_ids[p],
// pass) to sort_keys[p]. Nothing is written when ComputeHasEarlierStepFailed(key, kBlockCountStep): the stable order
// that follows may read keys left from before, which cannot make it fail, and its result is then never used.
// sort_count 0 starts nothing.
// Requires: pass is 0, 1 or 2; cell_ids and sort_keys hold sort_count values, both GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus WriteBlockSortKeysOnGpu(
    const CellId* cell_ids,
    std::uint32_t sort_count,
    int pass,
    const GridBuildStatus* status,
    std::uint64_t* sort_keys
);

// Step P8, the marks: one thread per sorted place j writes ComputeBlockStartMark(cell_ids, order, j) to marks[j], or 0
// when ComputeHasEarlierStepFailed(key, kBlockCountStep), a failure of step 5 included. Every one of the sort_count
// words is written, so the prefix sum that follows adds no word of an earlier launch. sort_count 0 starts nothing.
// Requires: order is the final order of the three passes, a permutation of 0..sort_count - 1; marks holds sort_count
//   words and overlaps neither cell_ids nor order; all GPU memory.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus WriteBlockStartMarksOnGpu(
    const CellId* cell_ids,
    const std::uint32_t* order,
    std::uint32_t sort_count,
    const GridBuildStatus* status,
    std::uint32_t* marks
);

// Step P8, the scatter: one thread per sorted place j runs ApplyBlockIdWriteAt(block_ids, cell_ids, order, sums, j), so
// the first entry of each block writes its id at its sum and the block ids come out in the fixed order. Nothing is
// written when ComputeHasEarlierStepFailed(key, kBlockFaultStep): after a failure of step 6 the sums could lie past
// the block ids. sort_count 0 starts nothing.
// Requires: sums holds the exclusive prefix sums of the marks; block_ids has room for max_cell_blocks ids; all GPU
//   memory, block_ids overlapping none of the others.
// Fails: kCallFailed  the GPU function did not start
RunOnCallStatus WriteBlockIdsOnGpu(
    CellBlockId* block_ids,
    const CellId* cell_ids,
    const std::uint32_t* order,
    const std::uint32_t* sums,
    std::uint32_t sort_count,
    const GridBuildStatus* status
);

// Reads the largest local memory per thread, in bytes, that the compiler gave any GPU function of this file
// (cudaFuncGetAttributes, localSizeBytes), so a test can compare it with the driver's default stack of 1,024 B.
// Fails: kCallFailed  a query failed; *bytes_per_thread is then unchanged
RunOnCallStatus ComputeLargestStackOnGpu(std::uint64_t* bytes_per_thread);

}  // namespace fluidity::grid
