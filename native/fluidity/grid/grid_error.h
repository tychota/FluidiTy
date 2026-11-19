// Why a grid call failed, and the record that carries it.
//
// Purpose: give the grid module its one error enum and a small record for a failure that GPU code can fill.
//
// Idea: the grid is the set of active cells, kept in cell blocks. Building it, checking it and looking up a position
//   can fail in a dozen ways: a bad value, cells that overlap, a missing sibling cell, a broken size step rule, no
//   room for the cell blocks, a position outside the grid, or a fault in FluidiTy itself. Each way is a GridError.
//   GridFailure holds the error and one number for the cases that need it. It is a plain struct that never allocates.
//   Code shared with the GPU, such as the search for nearby faces, reports through a GridError, so this header may
//   not include a host-only header. The text of each value is in grid/grid_error_message.h, which only the CPU may
//   include.
//
// Words:
//   cell block  a fixed chunk of cells
//   size step rule  neighbouring cells differ by at most one size level
// Owns:    GridError, GridFailure
// Uses:    nothing
// Used by: every grid function that can fail, and the step, which maps a failure to a rejected try
// Read:    docs/method-paper/fluidity-method.typ <sec-grid>

#pragma once

#include <cstdint>

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
  std::uint64_t magnitude = 0;  // The bytes asked for kOutOfMemory. 0 otherwise.
};

}  // namespace fluidity::grid
