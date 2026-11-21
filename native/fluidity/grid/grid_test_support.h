// Test-only: a geometric checker of mixed-size grids, random grids that every machine builds alike, and grids that
// own their memory.
//
// Purpose: give grid tests an answer to compare the grid's tables against, computed another way, lists of active cells
//   drawn from a seed that give the same list on every machine, and one call that builds a grid with its own memory.
//
// Idea: GeometricGrid sees only the list of active cells. It turns each cell into a box of whole units, a unit being
//   the width of a cell of the frame's last size level. It finds the cells across each side of a cell with its own
//   lookup of the list by place, an id per place made with ToCellId, and answers the other questions by looking at
//   every cell or face piece in turn: no cell blocks, no grid tables, no binary search. It restates each rule of the
//   method paper in its own plain code and calls no grid function that decides what it checks, so a grid function and
//   the checker agree only when both follow the paper. Its questions stay plain rather than fast: each looks at every
//   cell or face piece, fine up to a few thousand cells.
//   Random numbers come from std::mt19937_64, whose raw output the C++ standard fixes for a seed, mapped by
//   ToUniformIndex and ToUnitDouble only. The standard's distribution classes and std::shuffle are not used: their
//   output differs between standard libraries.
//   For the tests that use it: the checker lists its answers in its own order, stated with each answer, and a grid
//   function lists them in place order, so a test compares the two as sets (or orders both by one key it states) and
//   checks the grid's own order apart. FacesWithin makes one pass over every face piece per call, so a test over many
//   grids calls it for a few chosen faces of each grid, not for every face.
//   TestGrid keeps a built grid together with the RunMemory, the builder buffers and the list its buffers come from,
//   destroyed in the order the lending rules ask for.
//   Example: the size-level-0 cell {0,0,0} beside the 8 size-level-1 cells of the size-level-0 cell {1,0,0}, in a
//   frame of 2 size levels and biggest cells 1 m wide, has 41 face pieces: the big cell's 5 whole sides, the 4 quarters
//   between it and the small cells, the 12 sides between small cells and their 20 outer sides.
//
// Words:
//   unit             the width of a cell of the frame's last size level; every box of the checker is whole units
//   size-step closure  the list after every active cell two or more size levels bigger than a cell it touches has
//                    been split, again and again, until none is left
//   window quarter   n_p of the method paper: the quarter of its cell that holds a position, 0..3 on each axis
// Owns:    ToUniformIndex, ToUnitDouble, BuildRandomGridCells, ApplyFixedShuffle, kRandomGridCount,
//          BuildShuffledRandomGrids, BuildSizeStepClosure, UnitBox, ToUnitBox, GeometricFacePiece, GeometricGrid,
//          TestGrid, BuildTestGrid, TakeEmptyGridBuffers, kRandomGridSizeLevelCount, kMaxRandomGridCellBlocks,
//          ToGridCountFields, ComputeIsSameUsedTables, ComputeIsSameGrid, FindRandomGridsSpreadBySize,
//          BuildReferenceGrids, kMutatedGridCount, BuildMutatedGridCells,
//          BuildLargeBoxOfCells, BuildSizeStepBreakWithSiblings, BuildLoneCellsAlongX, BuildSmallCellsOfEightBigCells,
//          BuildTwoSizeFixtureCells
// Uses:    Axis (base/axis.h), Buffer (base/buffer.h), RunMemory (base/run_memory.h), Vec3 (base/vec3.h), GridFailure
//          (grid/grid_error.h), CellId, CellCoord, SizeLevel, FacePieceId (grid/grid_ids.h), GridFrame
//          (grid/grid_tables.h), GridSettings, GridInputs, GridBuilderBuffers, StaggeredGrid (grid/staggered_grid.h),
//          CPU-only guard (base/cpu_only.h)
// Implementation uses: ToCellId, ToCellCoord, ToSizeLevel, ToFacePieceId, ComputeSmallCell,
//          ComputeIsCoordInRange (grid/grid_ids.h), BuildRunMemory, BuildBuffer, WriteBuffer, ComputeRunMemoryBytes,
//          ComputeRunMemoryPieceBytes (base/), BuildGridMemoryEstimate, ToMemoryCounts (grid/grid_memory_estimate.h),
//          ToGridFailure (grid/grid_buffers_internal.h), BuildGridBuilderBuffers, BuildGridBuffers, BuildGrid,
//          ComputeGridCounts (grid/staggered_grid.h), GoogleTest (ADD_FAILURE)
// Used by: the tests of grid/ (the library fluidity_grid_test_support), never the library itself
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-face-pieces>,
//          <sec-cell-blocks>, <sec-nearby-faces>

#pragma once

// The guard comes first, so GPU code that includes this file stops at the guard before any library header.
#include "fluidity/base/cpu_only.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/buffer.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

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

// The number of lists BuildShuffledRandomGrids holds.
inline constexpr std::uint64_t kRandomGridCount = 200;

// Builds the random grids that grid tests share, once per test program: for each seed s of 1..kRandomGridCount,
// entry s - 1 is BuildRandomGridCells(s, 4, 3) shuffled with ApplyFixedShuffle(cells, s + 1000). Each list is valid
// and in no particular order; 4 size levels in a box of 3 x 3 x 3 size-level-0 cells give at most 252 cell blocks.
//
// Method: a function-local static, built on the first call, so the tests of one program build the lists once.
// Example: entry 0 is the list of seed 1, shuffled with seed 1001.
const std::vector<std::vector<CellId>>& BuildShuffledRandomGrids();

// The number of mutated lists the tests run: BuildMutatedGridCells of seeds 1..kMutatedGridCount.
inline constexpr std::uint64_t kMutatedGridCount = 500;

// Builds a list that breaks the rules of a grid: the random list of BuildRandomGridCells(seed, 4, 3), shuffled with
// ApplyFixedShuffle(cells, seed + 1000), then changed by 1 to 4 mutations drawn from one std::mt19937_64 of seed +
// 7000, the same on every machine.
//
// Method: the first raw number gives the count, 1 + ToUniformIndex(raw, 4); then each mutation takes one raw number for
//   its kind, ToUniformIndex(raw, 7), and its own numbers after it:
//   0 repeat a cell, inserted at a drawn index (an overlap);
//   1 drop a cell (a missing sibling, unless it was a size-level-0 cell);
//   2 add the big cell of a cell of size level 1 or more (an overlap);
//   3 add the 8 small cells of the cell one size level smaller than a cell of size level 0 or 1, just beyond the box of
//     size-level-0 cells beside it: cells two size levels smaller beside it, with their siblings (the size step rule;
//     with siblings required, the missing siblings of their big cell come first);
//   4 drop every active cell inside one cell of size level 1 or more (the big cell of a cell of size level 2 or more,
//     or a cell of size level 1), so a whole sibling of a split cell is missing;
//   5 clear bit 63 of one id (bad input);
//   6 set one coordinate of one id to kMaxCoord + 1 (a coordinate out of range).
//   A mutation that needs a valid id (bit 63 set, coordinates in range, size level below 4) looks for one from a drawn
//   index on and is left out when there is none. A mutation may undo or hide an earlier one; the tests compare two
//   checks of the same list, not a planned failure.
// Example: seed 1 starts from entry 0 of BuildShuffledRandomGrids.
std::vector<CellId> BuildMutatedGridCells(std::uint64_t seed);

// Builds a large list: the size-level-0 cells of the box 0..46 on each axis, 103,823 cells, with `extra_cells` before
// them, all shuffled with ApplyFixedShuffle(cells, 4711).
// Example: with the size-level-2 cell {188,188,188}, which touches the box's last cell at the corner (47, 47, 47) m,
//   the list has 103,824 cells and, sorted, breaks the size step rule at its last position, 103,823.
std::vector<CellId> BuildLargeBoxOfCells(const std::vector<CellId>& extra_cells);

// Builds a list whose siblings are all present and which breaks the size step rule, for a frame of 3 size levels: size
// level 0 {0,0,0}, the size-level-1 cells of eighths 1 to 7 of size level 0 {1,0,0}, and the 8 size-level-2 cells of
// its eighth 0, size level 1 {2,0,0}: 16 cells.
// Example: sorted, size level 2 {4,0,0} is at position 1; its place in direction 12, (dz, dy, dx) = (0, 0, -1), is size
//   level 2 x = 3, inside size level 0 {0,0,0}, two size levels bigger: kSizeStepRuleBroken, cell size level 2
//   {4,0,0}, other_cell size level 0 {0,0,0}, with siblings required or not.
std::vector<CellId> BuildSizeStepBreakWithSiblings();

// Builds `count` lone size-level-0 cells {4 i, 0, 0}, i = 0..count - 1, each in a block of its own.
// Example: 9 such cells need 9 blocks, so buffers for 8 blocks fail with kTooManyCellBlocks at the block count.
std::vector<CellId> BuildLoneCellsAlongX(std::size_t count);

// Builds the 8 size-level-0 cells {0..1, 0..1, 0..1}, each split into its 8 size-level-1 cells: 64 active cells that
// add 8 split cells, 72 in all, above the 64 cell places of a room of 1 block.
std::vector<CellId> BuildSmallCellsOfEightBigCells();

// Builds the two-size fixture: size level 0 {0,0,0} beside the 8 size-level-1 cells of size level 0 {1,0,0}, 9 cells,
// for a frame of 2 or more size levels.
// Example: built, it has 2 blocks, size level 0 (0,0,0) and size level 1 (0,0,0), 9 active cells and 41 face pieces.
std::vector<CellId> BuildTwoSizeFixtureCells();

// Builds the size-step closure of a list: splits every active cell two or more size levels bigger than an active cell
// it touches, by a side, an edge or a corner, into its 8 small cells, and repeats until no such cell is left. The
// result is in the fixed order, stated in units as TouchingCells orders. For random grids and hand-built scenes alike.
//
// Method: rounds, in whole units of size level size_level_count - 1. A round first judges every cell on the list as
//   it was when the round began, then splits the cells judged too big. To judge a cell it looks, with the checker's
//   own lookup of the list by place, for a touching active cell two or more size levels smaller: it starts from the
//   26 places of the cell's size level around it and goes down through the places that touch it and hold smaller
//   cells.
// Example: the size-level-0 cell {0,0,0} beside size-level-2 cells in its +X neighbour, 3 size levels: it becomes its
//   8 size-level-1 cells, which differ by one size level from the size-level-2 cells.
// Requires: every cell's size level is below size_level_count; the list has no repeated and no overlapping cells. In a
//   debug build a repeated or overlapping cell stops the program at an assert; a release build gives an invalid list.
//   A split whose small cells would leave -kMaxCoord..kMaxCoord fails the calling test and is not made.
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

// A mixed-size grid seen through its active cells alone, for tests: the face pieces come from its own lookup of the
// list by place, every other answer from a pass over the cells or the face pieces, in whole units, so it shares no
// table and no search with the grid functions.
//
// Requires (for every member): the list keeps siblings together, has no repeated and no overlapping cells, and every
//   size level is below frame.size_level_count. In a debug build the constructor stops the program at an assert on a
//   repeated or overlapping cell; a release build gives wrong answers. A side of a cell that other cells cover only in
//   part fails the calling test.
class GeometricGrid {
 public:
  // Keeps the frame and the list, and finds every face piece.
  GeometricGrid(GridFrame frame, std::vector<CellId> active_cells);

  // Every face piece, each once, ordered by axis, then by centre in z, then y, then x.
  //
  // Method: for each axis and each active cell, looks at every cell whose low side lies on the cell's high plane and
  //   shares some area with its high side. Each such pair is one piece, of the smaller cell's size level and side. A
  //   side that no cell covers is one piece at the grid's edge (method paper, <sec-face-pieces>). The cells across a
  //   side come from the checker's own lookup of the list by place: the place across the side at each size level up
  //   to the cell's own, then down through the places that hold smaller cells.
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

// A grid with its own memory, for tests. run_memory is declared first, so it is destroyed last, after the buffers that
// are memory pieces of it. The builder buffers are held by pointer so that moving a TestGrid never moves the lent
// GridBuilderBuffers. Move construction only: a RunMemory must not be assigned over while its pieces are alive.
struct TestGrid {
  TestGrid(
      RunMemory run_memory,
      Buffer<CellId> cell_list,
      std::unique_ptr<GridBuilderBuffers> builder_buffers,
      StaggeredGrid grid
  );
  TestGrid(const TestGrid&) = delete;
  TestGrid& operator=(const TestGrid&) = delete;
  TestGrid(TestGrid&& other) noexcept = default;
  TestGrid& operator=(TestGrid&& other) = delete;
  ~TestGrid() = default;

  RunMemory run_memory;
  Buffer<CellId> cell_list;  // The list on the grid's processor; empty for a CPU grid.
  std::unique_ptr<GridBuilderBuffers> builder_buffers;
  StaggeredGrid grid;
};

// Builds a grid with memory of its own: reserves exactly the estimate for one grid of the room (for a GPU grid, plus
// one memory piece for the list), then BuildGridBuilderBuffers, BuildGridBuffers and BuildGrid on settings.run_on.
//
// Method: the list is inputs.active_cells, a span in CPU memory. For a GPU grid the span's length is taken on the CPU
//   and the span is copied to cell_list, which the build then reads. With settings.max_cell_blocks 0 (the default) the
//   room is ComputeGridCounts(inputs).cell_blocks, at least 1.
// Example: the 9 size-level-0 cells {4 i, 0, 0} with default settings give a CPU grid whose max_cell_blocks() is 9.
// Requires: inputs.active_cell_buffer is null; otherwise the calling test fails and kBadInput is returned.
// Fails: as ComputeGridCounts (default room only), BuildRunMemory (through ToGridFailure), BuildGridBuilderBuffers,
//        BuildGridBuffers and BuildGrid; kBadInput for a room above kMaxCellBlocksLimit, before any memory is reserved.
std::expected<TestGrid, GridFailure> BuildTestGrid(const GridInputs& inputs, const GridSettings& settings = {});

// The size levels of the frame the random grids are built in (BuildShuffledRandomGrids).
inline constexpr int kRandomGridSizeLevelCount = 4;

// The most cell blocks a random grid has (BuildShuffledRandomGrids): the room of a reference grid that keeps its own
// memory.
inline constexpr std::uint32_t kMaxRandomGridCellBlocks = 252;

// Takes the empty buffers out of `owner`, a grid of no cells built with BuildTestGrid, so a test has grid buffers of
// the owner's room on the CPU. Their builder buffers and memory stay in `owner`, which must outlive the buffers.
// Requires: `owner` holds a grid.
GridBuffers TakeEmptyGridBuffers(std::expected<TestGrid, GridFailure>& owner);

// Every field of counts, as one comparable value: the blocks, the active cells per size level, the active cells and the
// face pieces.
std::tuple<std::uint32_t, std::array<std::uint64_t, kMaxSizeLevels>, std::uint64_t, std::uint64_t> ToGridCountFields(
    const GridCounts& counts
);

// Says whether two sets of tables have the same version, the same block count and the same bytes in the used part of
// every table: the block ids, 27 neighbours, the bigger block and 8 smaller blocks per block, 64 cell states and 240
// face piece kinds per block. Allocates nothing, so a test may call it while it counts allocations.
bool ComputeIsSameUsedTables(const GridTables& first, const GridTables& second);

// Says whether two grids have the same frame, the same counts and the same used tables on the CPU. Allocates nothing.
bool ComputeIsSameGrid(const StaggeredGrid& first, const StaggeredGrid& second);

// A spread of `count` random lists by size, so tests see small, middling and large grids and the sizes between them.
//
// Method: the kRandomGridCount entries are ordered by cell count, most cells first and the entry with the lower index
//   first among equal counts. Position i = 0..count - 1 of the result takes the entry of rank floor(i x
//   (kRandomGridCount - 1) / (count - 1)) in that order, in integer arithmetic, so every machine picks the same grids;
//   a count of 1 takes rank 0. The first position is the entry with the most cells and the last the one with the
//   fewest.
// Example: count 4 takes ranks 0, 66, 132 and 199; count 3 takes ranks 0, 99 and 199.
// Requires: 1 <= count <= kRandomGridCount; otherwise the result repeats entries or is empty.
std::vector<std::size_t> FindRandomGridsSpreadBySize(std::size_t count);

// The random grids of the given entries, each version 1 with its own memory, in a room of kMaxRandomGridCellBlocks. A
// grid that fails to build fails the calling test and is left out.
std::vector<TestGrid> BuildReferenceGrids(const std::vector<std::size_t>& grids);

}  // namespace fluidity::grid
