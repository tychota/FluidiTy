// Test-only: a geometric checker of mixed-size grids, and random grids that every machine builds alike.
//
// Purpose: give grid tests an answer to compare the grid's tables against, computed another way, and lists of
//   active cells drawn from a seed that give the same list on every machine.
//
// Idea: GeometricGrid sees only the list of active cells. It turns each cell into a box of whole units, a unit being
//   the width of a cell of the frame's last size level, and answers every question by looking at every cell or face
//   piece in turn: no cell blocks, no tables, no binary search. It restates each rule of the method paper in its own
//   plain code and calls no grid function that decides what it checks, so a grid function and the checker agree only
//   when both follow the paper. It is slow on purpose: most loops are quadratic, fine up to a few thousand cells.
//   Random numbers come from std::mt19937_64, whose raw output the C++ standard fixes for a seed, mapped by
//   ToUniformIndex and ToUnitDouble only. The standard's distribution classes and std::shuffle are not used: their
//   output differs between standard libraries.
//   For the tests that use it: the checker lists its answers in its own order, stated with each answer, and a grid
//   function lists them in place order, so a test compares the two as sets (or orders both by one key it states) and
//   checks the grid's own order apart. FacesWithin makes one pass over every face piece per call, so a test over many
//   grids calls it for a few chosen faces of each grid, not for every face.
//   Example: the size-level-0 cell {0,0,0} beside the 8 size-level-1 cells of the size-level-0 cell {1,0,0}, in a
//   frame of 2 size levels and biggest cells 1 m wide, has 41 face pieces: the big cell's 5 whole sides, the 4 quarters
//   between it and the small cells, the 12 sides between small cells and their 20 outer sides.
//
// Words:
//   unit             the width of a cell of the frame's last size level; every box of the checker is whole units
//   size-step closure  the list after every active cell two or more size levels bigger than a cell it touches has
//                    been split, again and again, until none is left
//   window quarter   n_p of the method paper: the quarter of its cell that holds a position, 0..3 on each axis
// Owns:    ToUniformIndex, ToUnitDouble, BuildRandomGridCells, ApplyFixedShuffle, BuildSizeStepClosure, UnitBox,
//          ToUnitBox, GeometricFacePiece, GeometricGrid
// Uses:    Axis (base/axis.h), Vec3 (base/vec3.h), CellId, CellCoord, SizeLevel, FacePieceId (grid/grid_ids.h),
//          GridFrame (grid/grid_tables.h), CPU-only guard (base/cpu_only.h)
// Implementation uses: ToCellId, ToCellCoord, ToSizeLevel, ToFacePieceId, ComputeSmallCell,
//          ComputeIsCoordInRange (grid/grid_ids.h), GoogleTest (ADD_FAILURE)
// Used by: the tests of grid/ (the library fluidity_grid_test_support), never the library itself
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-face-pieces>,
//          <sec-cell-blocks>, <sec-nearby-faces>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// Maps a raw 64-bit number to 0..count - 1: raw % count. The only integer mapping random tests use.
//
// Example: ToUniformIndex(20, 7) = 6.
// Requires: count >= 1. A count of 0 fails the calling test and gives 0.
//
// clang-tidy: a raw number and a count; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::uint64_t ToUniformIndex(std::uint64_t raw, std::uint64_t count);

// Maps a raw 64-bit number to [0, 1): its 53 high bits times 2^-53. Exact, so the same on every machine.
//
// Example: ToUnitDouble(2^63) = 0.5; ToUnitDouble(2^64 - 1) = 1 - 2^-53.
double ToUnitDouble(std::uint64_t raw);

// Builds a random valid list of active cells: siblings together and the size-step rule kept, in the fixed order.
//
// Method: starts from the box of size-level-0 cells {0..n-1}^3, n = biggest_cells_per_axis, with one std::mt19937_64 of
//   `seed`. It visits the cells size level by size level, each size level in the fixed order (the order TouchingCells
//   states), and draws one raw number per cell it visits: a size-level-0 cell is split when
//   ToUniformIndex(raw, 2) == 0, a smaller cell when ToUniformIndex(raw, 4) == 0. Cells of the last size level,
//   size_level_count - 1, cannot be split and take no number. The list is then closed by BuildSizeStepClosure. Every
//   step is integer work on the generator's raw output, so every machine gives the same list.
// Example: (seed, 4, 3) gives at most 27 x 8^3 = 13,824 cells, of size levels 0 to 3, inside 0..3 m of a frame of
//   biggest cells 1 m wide.
// Requires: size_level_count is 1..kMaxSizeLevels; biggest_cells_per_axis is 1..kMaxCoord + 1.
//
// clang-tidy: a count of size levels and a count of cells; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<CellId> BuildRandomGridCells(std::uint64_t seed, int size_level_count, int biggest_cells_per_axis);

// Puts cells in an order drawn from `seed`, the same on every machine.
//
// Method: one std::mt19937_64 of `seed`; for i from the last index down to 1, swaps entry i with entry
//   ToUniformIndex(raw, i + 1), one raw number per swap. Papers call it the Fisher-Yates shuffle.
// Example: a list of 0 or 1 cells takes no number and is unchanged.
void ApplyFixedShuffle(std::vector<CellId>& cells, std::uint64_t seed);

// Builds the size-step closure of a list: splits every active cell two or more size levels bigger than an active cell
// it touches, by a side, an edge or a corner, into its 8 small cells, and repeats until no such cell is left. The
// result is in the fixed order, stated in units as TouchingCells orders. For random grids and hand-built scenes alike.
//
// Method: brute force over every pair of cells in each round, in whole units of size level size_level_count - 1.
// Example: the size-level-0 cell {0,0,0} beside size-level-2 cells in its +X neighbour, 3 size levels: it becomes its
//   8 size-level-1 cells, which differ by one size level from the size-level-2 cells.
// Requires: every cell's size level is below size_level_count; the list has no overlapping cells. A split whose small
//   cells would leave -kMaxCoord..kMaxCoord fails the calling test and is not made.
std::vector<CellId> BuildSizeStepClosure(std::vector<CellId> cells, int size_level_count);

// The box of one cell in whole units of the last size level: it covers low <= u < low + width on each axis.
struct UnitBox {
  std::array<std::int64_t, 3> low{};  // Units, indexed by axis (x, y, z).
  std::int64_t width = 0;  // Units: 2^(size_level_count - 1 - the cell's size level).
};

// Converts a cell to its box in whole units of size level size_level_count - 1.
//
// Example: with 3 size levels, size-level-1 cell {-1, 2, 0} is the box low {-2, 4, 0}, width 2.
// Requires: the cell's size level is below size_level_count.
UnitBox ToUnitBox(CellId cell, int size_level_count);

// One face piece as the geometric checker sees it.
struct GeometricFacePiece {
  FacePieceId id{};  // ToFacePieceId(axis, size_level, cell_high_coord).
  Axis axis = Axis::kX;  // The face's normal is +axis.
  SizeLevel size_level{};  // The size level of the smaller of its two cells, or of its one cell at the grid's edge.
  // In cells of size_level: the place on the high side. Along the axis it may be kMaxCoord + 1.
  CellCoord cell_high_coord{};
  std::optional<CellId> cell_low;  // The active cell on the -axis side, or none at the grid's edge.
  std::optional<CellId> cell_high;  // The active cell on the +axis side, or none at the grid's edge.
  double area = 0;  // m2. The face area of a cell of size_level.
};

// A mixed-size grid seen through its active cells alone, for tests: every answer is found by brute force over the
// cells or the face pieces, in whole units, so it shares no table and no search with the grid functions.
//
// Requires (for every member): the list keeps siblings together, has no overlapping cells, and every size level is
//   below frame.size_level_count. A side of a cell that other cells cover only in part fails the calling test.
class GeometricGrid {
 public:
  // Keeps the frame and the list, and finds every face piece.
  GeometricGrid(GridFrame frame, std::vector<CellId> active_cells);

  // Every face piece, each once, ordered by axis, then by centre in z, then y, then x.
  //
  // Method: for each axis and each active cell, looks at every cell whose low side lies on the cell's high plane and
  //   shares some area with its high side. Each such pair is one piece, of the smaller cell's size level and side. A
  //   side that no cell covers is one piece at the grid's edge (method paper, <sec-face-pieces>). Cells are grouped by
  //   plane first, only so that a cell is not compared with cells on other planes.
  // Example: the grid of the file header gives 13 X, 14 Y and 14 Z pieces.
  [[nodiscard]] const std::vector<GeometricFacePiece>& FacePieces() const;

  // The active cells that share at least one point with `cell` (a side, an edge or a corner), `cell` itself left out,
  // `cell` need not be active.
  //
  // Order: the checker's own key, by the low corner in units, z, then y, then x, and at one low corner the bigger cell
  //   first. It is the fixed order of method paper <sec-cell-blocks>, found without ComesBefore, so a test may compare
  //   the two.
  //
  // Example: a size-level-0 cell whose 26 neighbours are all split touches 6 x 4 + 12 x 2 + 8 = 56 cells.
  [[nodiscard]] std::vector<CellId> TouchingCells(CellId cell) const;

  // Every corner of every active cell, each point once, as whole units (x, y, z) from the frame's origin, ordered by
  // z, then y, then x. A corner of a small cell in the middle of a big cell's side or edge is one of them. Exact in
  // any frame, so a test of corner places compares these.
  //
  // Example: the grid of the file header has 31 corners; {3, 1, 1} is the corner at {1.5, 0.5, 0.5} m.
  [[nodiscard]] std::vector<std::array<std::int64_t, 3>> CornersInUnits() const;

  // The corners of CornersInUnits, in its order, as world positions.
  //
  // Units: m, world: origin + whole units x the width of a cell of the last size level. Exact when every term is, as
  //   with origin 0 and a power-of-two biggest cell width; compare CornersInUnits otherwise.
  // Example: the grid of the file header has 31 corners: the big cell's 8 and the small cells' 27 share 4.
  [[nodiscard]] std::vector<Vec3> Corners() const;

  // The active cell that holds `position`, or none.
  //
  // Method: per axis, the whole unit floor((x - origin) / w) with w the width of the last size level's cells, then
  //   the one cell whose box holds that unit. A position that is not finite, or whose unit lies outside the cells a
  //   size-level-0 id can name, -kMaxCoord 2^(L-1) to (kMaxCoord + 1) 2^(L-1) - 1 with L = size_level_count, is in no
  //   cell (method paper, <sec-cell-sizes>). A cell owns its low sides, not its high sides.
  // Units: m, world.
  // Example: in the grid of the file header, {1, 0, 0} m is in size-level-1 cell {2,0,0}, not in the big cell.
  [[nodiscard]] std::optional<CellId> ContainingCell(Vec3 position) const;

  // The candidate faces of one axis for a position in `cell` whose window quarters are `window_quarters`, in the
  // order of FacePieces().
  //
  // Method: the rule of method paper <sec-nearby-faces>, tried on every face piece of the axis. A piece is a candidate
  //   when one of its two cells is `cell` or touches it, its size level is that of `cell` minus 1, the same or plus 1,
  //   and on each axis its centre n_f, in quarter widths of `cell` from the low corner of `cell`, satisfies
  //   n_p - 5 <= n_f <= n_p + 6. The test is made on whole numbers: 4 (centre - low corner) is compared with
  //   (n_p - 5) w and (n_p + 6) w, w the width of `cell`, all in units.
  // Example: in a grid of one size level, every cell away from the edge has 27 candidates in each of its 64 windows.
  // Requires: `cell` is active; each window quarter is 0..3.
  [[nodiscard]] std::vector<GeometricFacePiece>
  NearbyFaces(CellId cell, std::array<int, 3> window_quarters, Axis axis) const;

  // The face pieces of the axis of `face` whose centre lies strictly within radius_in_cell_widths widths of the
  // centre of `face` on each axis, `face` itself included, in the order of FacePieces(). The width is that of a cell
  // of the size level of `face`. The radius is used as given, however large.
  //
  // Example: in a grid of one size level and width 1 m, radius 1.5 around an X face piece away from the edge finds
  //   the 3 x 3 x 3 X face pieces centred on it: 27.
  // Requires: `face` is the id of one of FacePieces(); otherwise the calling test fails and the list is empty.
  [[nodiscard]] std::vector<GeometricFacePiece> FacesWithin(FacePieceId face, double radius_in_cell_widths) const;

 private:
  GridFrame frame_;
  std::vector<CellId> active_cells_;  // As given.
  std::vector<UnitBox> active_boxes_;  // One per entry of active_cells_.
  std::vector<GeometricFacePiece> face_pieces_;  // Found once, by the constructor.
};

}  // namespace fluidity::grid
