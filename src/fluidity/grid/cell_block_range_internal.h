// The status reading of the border search, for the grid module's own code and tests.
//
// Purpose: name the one host step that turns the status record of a border search into its count or its error, so
//   a test runs it on the CPU after a replay of the GPU steps, and a search on the GPU runs the very same code.
//
// Idea: a border search on the GPU ends with one status record read back to the CPU. The record holds the failure key
//   of the one step that can fail on the caller's input, the check of the border so far, and the number of blocks
//   the search listed. A key of that step is bad input. A key of any other step, or a count above the grid's block
//   count, contradicts the steps that wrote the record, and no input causes it: it is an internal fault. The reading
//   needs no GPU, so it is compiled in every build and a test replays the launches before it on the CPU.
//   Example: a record with no key and the count 3, for a grid of 10 blocks, gives 3; one with the key of the border
//   step gives kBadInput; one with the count 11 gives kInternalFault.
//
// Owns:    ToBorderBlockCount
// Uses:    CPU-only guard (base/cpu_only.h), GridBuildStatus (grid/grid_build_status.h), GridFailure
//          (grid/grid_error.h)
// Implementation uses: ToFailureStep, kIdStep (grid/grid_build_steps.h)
// Used by: grid/cell_block_range_cpu.cc; the tests of the border search
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include "fluidity/base/cpu_only.h"

#include <cstdint>
#include <expected>

#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_error.h"

namespace fluidity::grid {

// Converts the status that the launches of a border search left to the number of blocks it listed, or to the failure.
// Changes nothing.
// Example: no key and result_count 3, with block_count 10, give 3; the key of step 1 gives kBadInput.
// Requires: the status is that of a search on a grid of block_count blocks.
// Fails: kBadInput        the key is of the id step: the border so far is not ascending, holds a block of the range or
//                         an index at or above the grid's block count.
//        kInternalFault   a key of any other step, or a count above block_count: the status contradicts the steps that
//                         wrote it, and no input causes it.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
std::expected<std::uint32_t, GridFailure> ToBorderBlockCount(const GridBuildStatus& status, std::uint32_t block_count);

}  // namespace fluidity::grid
