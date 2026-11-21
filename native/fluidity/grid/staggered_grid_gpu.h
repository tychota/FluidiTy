// The GPU half of the check of a cell list, private to grid/: what grid/staggered_grid_gpu.cu defines.
//
// Purpose: let grid/grid_cell_checks.cc run method paper <proc-build-grid-in-parallel> steps P1 to P6, and the failure
//   half of P12, without including a CUDA header.
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
//   This file exists only in a build with GPU code, so grid/grid_cell_checks.cc names it only under
//   `#if defined(FLUIDITY_HAS_GPU)`. It uses the C++20 subset and includes no CPU-only header.
//
// Stack: every function here keeps under the driver's default of 1,024 B per thread, so the grid has no entry held by
//   the driver. The GPU functions keep no local array: a few indexes, a failure key, a CellCheck (24 bytes) and the
//   coordinates of the checks of grid/grid_build_steps.h. ComputeLargestStackOnGpu reads the bytes the compiler gave.
//
// Owns:    ApplyBuildStatusResetOnGpu, CheckCellIdsOnGpu, WriteCellSortKeysOnGpu, CopySortedCellsOnGpu,
//          CheckSortedCellsOnGpu, WriteFailureCellsOnGpu, ComputeLargestStackOnGpu
// Uses:    RunOnCallStatus (base/run_on_call.h), GridBuildStatus (grid/grid_build_status.h), CellId (grid/grid_ids.h),
//          GridFrame (grid/grid_tables.h)
// Used by: grid/grid_cell_checks.cc and grid/grid_cell_checks_gpu_test.cc; defined in grid/staggered_grid_gpu.cu
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

// Reads the largest local memory per thread, in bytes, that the compiler gave any GPU function of this file
// (cudaFuncGetAttributes, localSizeBytes), so a test can compare it with the driver's default stack of 1,024 B.
// Fails: kCallFailed  a query failed; *bytes_per_thread is then unchanged
RunOnCallStatus ComputeLargestStackOnGpu(std::uint64_t* bytes_per_thread);

}  // namespace fluidity::grid
