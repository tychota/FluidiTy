// What the committed cells around a new cell say about it.
//
// Purpose: give the step module the two facts about a box that is not in the grid, read from the committed cells that
//   touch it, so it can make new cells that obey the size step rule.
//
// Idea: empty space comes in whole biggest cells. A new cell starts as a biggest cell and is split into its 8 small
//   cells only when a committed cell two or more size levels smaller touches it. The grid module answers the one
//   exact question that loop needs: which committed cells touch the box, as the two facts of NewCellNeighbors.
//
// Words:
//   committed cell  an active cell of the grid that is built
//   new cell        a box that is not in the grid and may become an active cell
// Owns:    NewCellNeighbors
// Uses:    nothing
// Used by: the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-size-rule>

#pragma once

namespace fluidity::grid {

// What the committed cells around a cell that is not in the grid say about it.
struct NewCellNeighbors {
  // A committed cell two or more size levels smaller touches it: the new cell must be split.
  bool touches_much_smaller_cell;
  // A committed cell one or more size levels bigger touches it: its small cells would break the size step rule.
  bool touches_bigger_cell;
};

}  // namespace fluidity::grid
