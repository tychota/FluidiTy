// Why a grid call failed, and the record that carries it.
//
// Purpose: give the grid module its one error enum and a small record for a failure that GPU code can fill.
//
// Idea: the grid is the set of active cells, kept in cell blocks. Building it, checking it and looking up a position
//   can fail in a dozen ways: a bad value, cells that overlap, a missing sibling cell, a broken size step rule, no
//   room for the cell blocks, a position outside the grid, or a fault in FluidiTy itself. Each way is a GridError.
//   GridFailure holds the error, one number and up to two cell ids for the cases that need them. It is a plain struct
//   that never allocates. Code shared with the GPU, such as the search for nearby faces, reports through a GridError,
//   so this header may not include a host-only header. The text of each value is in grid/grid_error_message.h, which
//   only the CPU may include.
//
// Words:
//   cell block  a fixed chunk of cells
//   size step rule  neighbouring cells differ by at most one size level
// Owns:    GridError, GridFailure
// Uses:    CellId (grid/grid_ids.h)
// Used by: every grid function that can fail, and the step, which maps a failure to a rejected try
// Read:    docs/method-paper/fluidity-method.typ <sec-grid>

#pragma once

#include <cstddef>
#include <cstdint>

#include "fluidity/grid/grid_ids.h"

namespace fluidity::grid {

// Why a grid call failed. The first value is no error. Values keep their numbers: a new value is added last, except
// that kOutOfMemory stays last.
enum class GridError : std::uint8_t {
  kNone,  // No error. For shared functions that report through a GridError*.
  // A value is not finite, not positive, or out of range; a version of 0; a span of the wrong size; a block list that
  // is not ascending or is out of range.
  kBadInput,
  kCoordOutOfRange,  // A cell coordinate does not fit the largest coordinate at its size level.
  kCellsOverlap,  // The same cell twice, or a cell and a bigger cell that contains it.
  kSiblingCellMissing,  // A big cell is split, but one of its 8 small cells is absent.
  kSizeStepRuleBroken,  // Two touching cells differ by more than one size level.
  // More cell blocks than the buffers or the values have room for (max_cell_blocks).
  kTooManyCellBlocks,
  kRunOnCallFailed,  // The chosen processor cannot run here, or a call to it failed. Nothing falls back to the CPU.
  kWrongGridVersion,  // A buffer or index of another grid version was passed, or values that belong to no grid.
  kPositionOutsideGrid,  // No active cell contains the position.
  // A state the grid's own code should never reach, checked in every build, such as more nearby-face candidates than
  // the list holds. No input causes it: it is a fault in FluidiTy, not in the scene, and never reported as bad input.
  kInternalFault,
  // The run's memory or the processor could not hold a buffer. Only the grid builders return it; the failure's
  // magnitude is the number of bytes asked for. Last value.
  kOutOfMemory,
};

// A failure with its context. Fields that do not apply keep their default.
struct GridFailure {
  GridError error = GridError::kNone;
  // The failing cell. Of a bad input id: the first by input index, as listed. Of a check of the sorted list: the cell
  // at the first failing position in the fixed order, or, for a missing sibling cell, that sibling, which need not be
  // listed. CellId{0}, never a valid id, when none, and for a missing sibling cell whose coordinate lies outside the
  // range, which no id can name.
  CellId cell{};
  // The second cell of an overlap or of a broken size step rule, or the big cell of a missing sibling cell. CellId{0}
  // when unused.
  CellId other_cell{};
  std::uint64_t magnitude = 0;  // The bytes asked for kOutOfMemory. 0 otherwise.
};
// The GPU writes a failure and the CPU reads it back byte for byte, so both compilers must lay it out alike: error at
// 0, cell at 8, other_cell at 16, magnitude at 24. Bytes 1 to 7 are padding.
static_assert(sizeof(GridFailure) == 32, "the CPU and the GPU read one layout of GridFailure");
static_assert(offsetof(GridFailure, cell) == 8, "the CPU and the GPU read one layout of GridFailure");
static_assert(offsetof(GridFailure, other_cell) == 16, "the CPU and the GPU read one layout of GridFailure");
static_assert(offsetof(GridFailure, magnitude) == 24, "the CPU and the GPU read one layout of GridFailure");

}  // namespace fluidity::grid
