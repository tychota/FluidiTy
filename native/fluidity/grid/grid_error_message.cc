#include "fluidity/grid/grid_error_message.h"

#include <string_view>

#include "fluidity/grid/grid_error.h"

namespace fluidity::grid {

std::string_view ToMessage(GridError error) {
  switch (error) {
    case GridError::kNone:
      return "There is no grid error.";
    case GridError::kBadInput:
      return "A grid value is not a finite positive number or is out of range, or a list has the wrong size or order. "
             "Check the grid origin and cell width, the grid version, and that the lists of cells and cell blocks are "
             "complete and in order.";
    case GridError::kCoordOutOfRange:
      return "A cell lies too far from the grid origin for its size. Move the scene closer to the origin or use a "
             "larger cell width.";
    case GridError::kCellsOverlap:
      return "Two active cells overlap: the same cell is listed twice, or a cell lies inside a bigger one. List each "
             "place once.";
    case GridError::kSiblingCellMissing:
      return "A big cell is split into small cells, but one of its 8 small cells is missing. List all 8 small cells "
             "or use the big cell.";
    case GridError::kSizeStepRuleBroken:
      return "Two touching cells differ by more than one size level. Add cells of the in-between size, or make the "
             "neighbouring cells more alike.";
    case GridError::kTooManyCellBlocks:
      return "The grid needs more cell blocks than the limit. Raise max_cell_blocks or use fewer small cells.";
    case GridError::kRunOnCallFailed:
      return "The chosen processor cannot run here, or a call to it failed. Check the GPU and its driver, "
             "or choose the CPU.";
    case GridError::kWrongGridVersion:
      return "A buffer or an index belongs to another grid version, or to no grid. "
             "Build the buffer or the values again from the current grid.";
    case GridError::kPositionOutsideGrid:
      return "No active cell contains this position. Check that the position lies inside the cells of the grid, or add "
             "cells there.";
    case GridError::kInternalFault:
      return "The grid code reached a state it should never reach. This is a fault in FluidiTy, not in the scene. "
             "Please report it and send the scene.";
    case GridError::kOutOfMemory:
      return "There is not enough memory for the grid's buffers or for a copy of its list of cells. Use fewer cells "
             "or free memory; when the buffers do not fit, the other processor may have more.";
  }
  return "An unknown grid error happened.";
}

std::string_view ToName(GridError error) {
  switch (error) {
    case GridError::kNone:
      return "none";
    case GridError::kBadInput:
      return "bad_input";
    case GridError::kCoordOutOfRange:
      return "coord_out_of_range";
    case GridError::kCellsOverlap:
      return "cells_overlap";
    case GridError::kSiblingCellMissing:
      return "sibling_cell_missing";
    case GridError::kSizeStepRuleBroken:
      return "size_step_rule_broken";
    case GridError::kTooManyCellBlocks:
      return "too_many_cell_blocks";
    case GridError::kRunOnCallFailed:
      return "run_on_call_failed";
    case GridError::kWrongGridVersion:
      return "wrong_grid_version";
    case GridError::kPositionOutsideGrid:
      return "position_outside_grid";
    case GridError::kInternalFault:
      return "internal_fault";
    case GridError::kOutOfMemory:
      return "out_of_memory";
  }
  return "unknown";
}

}  // namespace fluidity::grid
