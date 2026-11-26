// The active cells near a face piece, and the walk over the face pieces within a radius of one.
//
// Purpose: give every stage the active cells near a face piece and a search that lists, one by one, the face pieces of
//   its axis within a radius, from the tables of one grid version, on the CPU or the GPU.
//
// Idea: two questions about the surroundings of one face piece. Which cells can list it? Its two cells and their
//   touching cells, at most 112, in a plain fixed array (FindCellsNearFace). Which face pieces lie near it? A face
//   search walks the face pieces of its axis, one size level bigger, the same
//   or one smaller, within a radius of at most 4 of its widths (BuildFaceSearch, FindNextFace). It reads at most 270
//   blocks, chosen by position around its block, and stores where the walk is, not a list.
//   Readers. All are total: FindCellsNearFace, BuildFaceSearch and FindNextFace give a defined answer for any face
//   index, kNoIndex included (count 0, a search of no block, kNoIndex), when every block index entry of the tables
//   is kNoIndex or below block_count, as in every built grid. The other functions are for this header's own
//   functions only, and their Requires lines say what they need.
//
// Words:
//   touching cell   an active cell that shares a side, an edge or a corner with a given cell
//   cells near a face  a face piece's cell_low, its cell_high and the touching cells of either
//   face search     a walk from one face piece, the start, over the face pieces of its axis and of size levels within
//                   one of its own whose centres lie strictly within a radius of its centre on each axis
//   quarter width   a quarter of the width of a cell of the start's size level, the unit of a face search's test
// Owns:    kMaxCellsNearFace, CellsNearFace, FindCellsNearFace, kMaxFaceSearchRadius, kMaxFaceSearchBlocks, FaceSearch,
//          BuildFaceSearch, FindNextFace; and, for this header's own functions only, ApplyCellNearFaceInsert,
//          ComputeClampedFaceSearchRadius, FindBlockAtCoord, ApplyFaceSearchBlockInsert, QuarterWidthPoint,
//          ComputeFaceCenterInQuarterWidths and ComputeIsWithinQuarterRadius
// Uses:    Axis (base/axis.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), TouchingCells, FindTouchingCells
//          (grid/touching_cells.h), GridTables, FindCellLow, FindCellHigh, FindCellBlock, ToSizeLevel, ToAxis,
//          ComputeHighPlaceCoord, ApplyAscendingInsert (grid/grid_tables.h), the id types and constants
//          (grid/grid_ids.h)
// Used by: the particle stages that look for nearby faces, nearby_face_weights.h, and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-nearby-faces>, <sec-face-search>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {

// The most active cells near one face piece (FindCellsNearFace): cell_low and cell_high, and the touching cells of
// each, 2 + 2 x 56, less 2 because in a built grid each of the two cells touches the other. Proof at FindCellsNearFace;
// the largest count known is 82.
inline constexpr int kMaxCellsNearFace = 112;

// The active cells near one face piece, as FindCellsNearFace gives them. A default-made value is an empty list: count 0
// and every entry kNoIndex.
struct CellsNearFace {
  FLUIDITY_CPU_AND_GPU constexpr CellsNearFace() noexcept {
    for (CellPlaceIndex& entry : cells) {
      entry = CellPlaceIndex{kNoIndex};
    }
  }

  // The places of the cells, cells[0..count - 1], in ascending index order; kNoIndex from count on.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  CellPlaceIndex cells[kMaxCellsNearFace] = {};
  int count = 0;  // 0..kMaxCellsNearFace.
};

// Inserts `cell` into the list in ascending index order, unless it is kNoIndex or the list already holds
// kMaxCellsNearFace cells, so no input writes past the array. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr void
ApplyCellNearFaceInsert(CellsNearFace& cells_near_face, CellPlaceIndex cell) noexcept {
  const bool is_list_full = cells_near_face.count >= kMaxCellsNearFace;
  if (cell.value == kNoIndex || is_list_full) {
    return;
  }
  ApplyAscendingInsert(cells_near_face.cells, cells_near_face.count, cell);
}

// Finds the active cells near a face piece: its cell_low, its cell_high and the touching cells of either. Every cell
// whose particles can list the face piece as a nearby face is among them: a particle's nearby faces are face pieces of
// its own cell and of the cells touching it (<sec-nearby-faces>).
//
// Method: FindCellLow and FindCellHigh give the two cells, kNoIndex on a side without one; FindTouchingCells gives the
//   touching cells of each. Each cell is inserted once, in ascending order.
//   Some listed cells can never list the face piece: a cell that touches a bigger cell_low or cell_high may be two
//   size levels bigger than the face piece, and a particle's nearby faces are of size levels within one of its own
//   cell's. They are listed all the same.
//   The bound. In a built grid cell_low and cell_high share the face piece, so each touches the other, and
//   FindTouchingCells finds every touching cell. So cell_low with its touching cells, at most 1 + 56, holds cell_high,
//   and cell_high with its touching cells holds cell_low: together at most 57 + 57 - 2 = 112 = kMaxCellsNearFace.
//   Tables that contradict themselves, such as neighbour links that disagree, can break "each touches the other" and
//   reach 114. The insert then stops at 112, so the array is never passed.
//   So 112 is an upper bound with room to spare. The largest count known is 82: two cells of one size beside each
//   other, whose 34 places of their size level around the pair are all split. A face piece between two sizes gives
//   at most 1 + 56 + 16 = 73.
// Order: ascending CellPlaceIndex. The cells are inserted in this order: cell_low, cell_high, the touching cells of
//   cell_low, then those of cell_high. When the list is full, the cells inserted later are left out.
// Example: in the two-size example of touching_cells.h, the face piece at X place 2 of block 1, between the big cell
//   (place 0) and the small cell {2,0,0} (place 64 + 2), gives the 9 cells of the grid: 0, 64 + 2, 64 + 3, 64 + 6,
//   64 + 7, 64 + 18, 64 + 19, 64 + 22 and 64 + 23.
// Requires: every block index entry of the tables is kNoIndex or below block_count, as in every built grid. Any `face`:
//   a place of kind kNotAPiece, an index of no place (kNoIndex, or a block at or beyond block_count) and the tables of
//   no grid give count 0 and read nothing beyond the tables. A face piece at the grid's edge, or a kOutsidePiece whose
//   other cell the grid piece did not load, gives its one cell and that cell's touching cells. Every entry from count
//   on is kNoIndex, so the result has the same bytes on the CPU and the GPU.
// Read: docs/method-paper/fluidity-method.typ <sec-face-search>, <sec-nearby-faces>, <sec-face-pieces>
FLUIDITY_CPU_AND_GPU constexpr CellsNearFace FindCellsNearFace(const GridTables& tables, FacePlaceIndex face) noexcept {
  CellsNearFace cells_near_face;
  if (ToCellBlockIndex(face).value >= tables.block_count) {
    return cells_near_face;
  }
  if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
    return cells_near_face;
  }
  const CellPlaceIndex cell_low = FindCellLow(tables, face);
  const CellPlaceIndex cell_high = FindCellHigh(tables, face);
  ApplyCellNearFaceInsert(cells_near_face, cell_low);
  ApplyCellNearFaceInsert(cells_near_face, cell_high);
  // FindTouchingCells gives count 0 for kNoIndex.
  const TouchingCells low_touching = FindTouchingCells(tables, cell_low);
  for (int entry = 0; entry < low_touching.count; ++entry) {
    ApplyCellNearFaceInsert(cells_near_face, low_touching.cells[entry]);
  }
  const TouchingCells high_touching = FindTouchingCells(tables, cell_high);
  for (int entry = 0; entry < high_touching.count; ++entry) {
    ApplyCellNearFaceInsert(cells_near_face, high_touching.cells[entry]);
  }
  return cells_near_face;
}

// The largest radius of a face search, in widths of a cell of the start face piece's size level: one block width. The
// blocks a search reads hold every face piece it finds up to this radius (proof at BuildFaceSearch).
inline constexpr double kMaxFaceSearchRadius = 4.0;

// The most blocks a face search reads: 27 block positions of the start's size level, 27 of the size level one bigger
// and 8 smaller positions under each of the first 27, 27 + 27 + 216.
inline constexpr int kMaxFaceSearchBlocks = 270;

// A walk over the face pieces near one face piece, made by BuildFaceSearch and moved on by FindNextFace. It stores no
// list of face pieces, only where the walk is. A default-made value is a search of no block, from face kNoIndex.
struct FaceSearch {
  FLUIDITY_CPU_AND_GPU constexpr FaceSearch() noexcept {
    for (CellBlockIndex& entry : blocks) {
      entry = CellBlockIndex{kNoIndex};
    }
  }

  FacePlaceIndex face = {.value = kNoIndex};  // The face piece the search starts from: the start.
  // No unit: widths of a cell of the start's size level, 0..kMaxFaceSearchRadius, as BuildFaceSearch clamped it.
  double radius = 0;
  // The blocks to read, blocks[0..block_count - 1], in ascending index order, each once; kNoIndex from block_count on.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  CellBlockIndex blocks[kMaxFaceSearchBlocks] = {};
  int block_count = 0;  // 0..kMaxFaceSearchBlocks.
  int next_block = 0;  // The entry of `blocks` the walk reads next, 0..block_count; block_count when finished.
  int next_place = 0;  // The place on the start's axis, 0..79, the walk reads next in that block; 80 past its last.
  // The struct holds 4 padding bytes after `face`, which no function writes: compare searches field by field.
};

// Clamps a face search radius: kMaxFaceSearchRadius above it, +infinity included; 0 for a radius that is not above 0,
// NaN included. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr double ComputeClampedFaceSearchRadius(double radius_in_cell_widths) noexcept {
  const bool is_above_zero = radius_in_cell_widths > 0;
  if (!is_above_zero) {
    return 0;
  }
  return radius_in_cell_widths < kMaxFaceSearchRadius ? radius_in_cell_widths : kMaxFaceSearchRadius;
}

// Finds the block of a size level at a block coordinate: FindCellBlock of its id; kNoIndex when the coordinate lies
// outside -kMaxCoord..kMaxCoord on an axis, where no id is packed, so none names another block, or when the tables
// hold no such block. For this header's own functions only.
// Requires: size_level is 0..kMaxSizeLevels - 1.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
FindBlockAtCoord(const GridTables& tables, int size_level, CellCoord block_coord) noexcept {
  assert(size_level >= 0 && size_level < kMaxSizeLevels);
  if (!ComputeIsCoordInRange(block_coord)) {
    return kNoIndex;
  }
  const auto block_size_level = static_cast<SizeLevel>(size_level);
  return FindCellBlock(tables, {.value = ComputePackedId(kIdCellMark, block_size_level, block_coord)}).value;
}

// Inserts a block into the search's blocks in ascending index order, each once; kNoIndex is left out. For this
// header's own functions only: BuildFaceSearch calls it at most kMaxFaceSearchBlocks times, so it never writes past
// the array.
FLUIDITY_CPU_AND_GPU constexpr void ApplyFaceSearchBlockInsert(FaceSearch& search, std::uint32_t block) noexcept {
  if (block == kNoIndex) {
    return;
  }
  ApplyAscendingInsert(search.blocks, search.block_count, CellBlockIndex{block});
}

// The centre of a face piece in quarter widths of the start's size level, from the frame's origin. For this header's
// own functions and for FindNearbyFacesWithRoom (grid/nearby_face_weights.h) only.
struct QuarterWidthPoint {
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;
};

// Computes the centre of the face piece at a face place in quarter widths of size level start_size_level, l. With m
// its block's size level and g the coordinate of its high cell place, the centre is g h_m along its axis and
// (g + 1/2) h_m across, and h_m = 2^(l - m) h_l: g 2^(l - m + 2) quarter widths along and (2 g + 1) 2^(l - m + 1)
// across, integers for m = l - 1, l or l + 1. For this header's own functions and for FindNearbyFacesWithRoom
// (grid/nearby_face_weights.h) only.
// Requires: `face` is a place of block 0..block_count - 1, of size level l - 1, l or l + 1.
FLUIDITY_CPU_AND_GPU constexpr QuarterWidthPoint
ComputeFaceCenterInQuarterWidths(const GridTables& tables, FacePlaceIndex face, int start_size_level) noexcept {
  const CellCoord high_coord = ComputeHighPlaceCoord(tables, face);
  const auto face_size_level = static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(face)));
  const int across_shift = start_size_level - face_size_level + 1;
  assert(across_shift >= 0 && across_shift <= 2);
  const auto across_scale = static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(across_shift));
  const Axis axis = ToAxis(face);
  // Along the axis a plane, g times twice the scale; across a centre, an odd number 2 g + 1 times the scale.
  const auto to_quarter_widths = [axis, across_scale](std::int32_t coord, Axis coord_axis) {
    if (coord_axis == axis) {
      return std::int64_t{coord} * 2 * across_scale;
    }
    return ((2 * std::int64_t{coord}) + 1) * across_scale;
  };
  return {
      .x = to_quarter_widths(high_coord.x, Axis::kX),
      .y = to_quarter_widths(high_coord.y, Axis::kY),
      .z = to_quarter_widths(high_coord.z, Axis::kZ),
  };
}

// Says whether two centres in quarter widths lie strictly within `quarter_radius` of each other on every axis. Each
// distance is an integer below 2^23, so its double is exact. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsWithinQuarterRadius(QuarterWidthPoint first, QuarterWidthPoint second, double quarter_radius) noexcept {
  const auto to_distance = [](std::int64_t first_coord, std::int64_t second_coord) {
    const std::int64_t difference = first_coord - second_coord;
    return static_cast<double>(difference < 0 ? -difference : difference);
  };
  const bool is_x_within = to_distance(first.x, second.x) < quarter_radius;
  const bool is_y_within = to_distance(first.y, second.y) < quarter_radius;
  const bool is_z_within = to_distance(first.z, second.z) < quarter_radius;
  return is_x_within && is_y_within && is_z_within;
}

// Makes a face search from `face`, the start: the blocks that hold every face piece FindNextFace may give, and the
// walk at its beginning.
//
// Method: with l the size level of the start's block and B its block coordinate, the search reads the blocks at three
//   sets of block positions. They are chosen by position, never by which neighbours exist:
//   (i) the 27 positions B + d of size level l, each axis of d -1, 0 or +1, from same_size_level_blocks of the
//     start's block;
//   (ii) for l >= 1, the 27 positions P + d of size level l - 1 around P = floor(B / 2), the position of the bigger
//     block, whether or not that block exists: from same_size_level_blocks of bigger_block when the tables have it,
//     else by FindCellBlock of each position's id;
//   (iii) for l + 1 a size level of the frame, the 8 positions 2 (B + d) + (0 or 1 on each axis) of size level l + 1
//     under each position of (i), 216 in all: from smaller_blocks of the block at B + d when it exists, else by
//     FindCellBlock of each position's id.
//   A position outside -kMaxCoord..kMaxCoord makes no id and holds no block. Each block found is listed once, in
//   ascending index order: at most kMaxFaceSearchBlocks.
//   Cost. One call makes at most 27 + 26 x 8 = 235 binary searches in the block ids: 27 when the tables lack the
//   bigger block, and 8 for each of the 26 positions of (i) beside the start's block that holds no block, when l + 1
//   is a size level of the frame. At the frame's smallest size level no block of size level l + 1 can exist, so those
//   searches are not made. In a built grid every block of size level 1 or more has its bigger block, so only the 8
//   searches per empty position of (i) remain, and they find nothing; they find blocks only in a grid piece.
//   Why these blocks hold every face piece G of size level m = l - 1, l or l + 1 within a radius r <= 4. Count in
//   widths h_l from the origin, a the start's axis and j an axis across it. The start's block covers 4 B <= x < 4 B + 4
//   on each axis. The start's centre p lies on a plane of its block, 4 B_a <= p_a <= 4 B_a + 4, plane 4 included, and
//   at a cell centre across, 4 B_j < p_j < 4 B_j + 4. G within r has its centre q at |q - p| < r <= 4 on each axis, so
//   4 B_a - 4 < q_a < 4 B_a + 8 and 4 B_j - 4 < q_j < 4 B_j + 8. G is kept in the block of its high cell place, or at
//   plane 4 of the block below along a (<sec-face-pieces>). Either way, with S that block's position and w its width
//   in h_l, w S_a <= q_a <= w S_a + w along a and w S_j < q_j < w S_j + w across. Then:
//   - m = l, w = 4: S_a <= B_a - 2 would give q_a <= 4 B_a - 4, and S_a >= B_a + 2 would give q_a >= 4 B_a + 8; the
//     same holds across. So S is a position of (i).
//   - m = l + 1, w = 2: S_a <= 2 B_a - 3 gives q_a <= 4 B_a - 4, and S_a >= 2 B_a + 4 gives q_a >= 4 B_a + 8; the
//     same holds across. So 2 B - 2 <= S <= 2 B + 3 on each axis: a position of (iii).
//   - m = l - 1, w = 8: B is 2 P or 2 P + 1, so 4 B - 4 >= 8 P - 4 and 4 B + 8 <= 8 P + 12. S_a <= P_a - 2 gives
//     q_a <= 8 P_a - 8, and S_a >= P_a + 2 gives q_a >= 8 P_a + 16; the same holds across. So S is a position of (ii).
//   The radius test is strict, and the proof needs it: a face piece exactly 4 widths along the axis from a start at
//   plane 4 lies at plane 0 of the block two positions up.
//   Face pieces two or more size levels away from the start's are never found, for two reasons. Smaller: a search
//   over every smaller size level would have no fixed bound on its blocks, since a cell one size level smaller is
//   h_l / 2 wide, the next h_l / 4, and so on, so face pieces of any smaller size level can lie within 1.5 h_l of a
//   face piece; it could not keep its state in a fixed array on the GPU, and a grid piece would have no finite border
//   for it. Bigger: each bigger size level would add at most 8 blocks within 4 h_l, a fixed cost, so leaving them out
//   is a choice. It keeps one rule for both directions, l - 1 to l + 1, the size levels of a particle's nearby-face
//   candidates (<sec-nearby-faces>), so the search loses no face piece a particle of a cell near the start could use.
//   In a grid piece the search reads the blocks at these positions that the grid piece holds, so it finds their face
//   pieces only.
// Units: radius_in_cell_widths in widths of a cell of the start's size level, no unit.
// Example: on a grid of one size level, cells 1 m wide, with blocks all around the start's block, the search reads
//   those 27 blocks, as no bigger or smaller block exists, and its walk at radius 1.5 gives the 3 x 3 x 3 face pieces
//   of the start's axis centred on it, itself included: 27.
// Requires: every block index entry of the tables (same_size_level_blocks, bigger_block, smaller_blocks) is kNoIndex
//   or the block at its position, below block_count, and kNoIndex only where the tables hold no block there, as in
//   every built grid and grid piece. Any `face` and any radius: a place of kind kNotAPiece, an index of no place and
//   the tables of no grid give a search of no block, whose walk gives nothing. A radius above kMaxFaceSearchRadius,
//   +infinity included, is kMaxFaceSearchRadius; a radius that is not above 0, NaN included, is 0 and finds nothing,
//   not even the start. Every entry of `blocks` from block_count on is kNoIndex.
// Read: docs/method-paper/fluidity-method.typ <sec-face-search>, <sec-face-pieces>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr FaceSearch
BuildFaceSearch(const GridTables& tables, FacePlaceIndex face, double radius_in_cell_widths) noexcept {
  // Steps: 1 check the start  2 the positions of (i) and (iii)  3 the positions of (ii)
  FaceSearch search;
  search.face = face;
  search.radius = ComputeClampedFaceSearchRadius(radius_in_cell_widths);
  // Step 1.
  const std::uint32_t block = ToCellBlockIndex(face).value;
  if (block >= tables.block_count) {
    return search;
  }
  if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
    return search;
  }
  const auto size_level = static_cast<int>(ToSizeLevel(tables, CellBlockIndex{block}));
  const CellCoord block_coord = ToCellCoord(CellId{.value = tables.block_ids[block].value});
  // Step 2: each position of size level l around the block, and the 8 positions of size level l + 1 under it.
  for (int entry = 0; entry < 27; ++entry) {
    const std::uint32_t around_block = tables.same_size_level_blocks[(static_cast<std::uint64_t>(block) * 27) + entry];
    ApplyFaceSearchBlockInsert(search, around_block);
    if (around_block != kNoIndex) {
      for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
        const std::uint64_t smaller_entry = (static_cast<std::uint64_t>(around_block) * 8) + cell_eighth;
        ApplyFaceSearchBlockInsert(search, tables.smaller_blocks[smaller_entry]);
      }
      continue;
    }
    // A grid piece may hold smaller blocks under a position whose block it did not load. Past the frame's smallest
    // size level no block exists, and past size level 15 no id can be made.
    const bool is_smaller_size_level_in_frame =
        size_level + 1 < static_cast<int>(tables.frame.size_level_count) && size_level + 1 < kMaxSizeLevels;
    if (!is_smaller_size_level_in_frame) {
      continue;
    }
    const CellCoord around_coord = {
        .x = block_coord.x + (entry % 3) - 1,
        .y = block_coord.y + ((entry / 3) % 3) - 1,
        .z = block_coord.z + (entry / 9) - 1,
    };
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      const CellCoord smaller_coord = {
          .x = (2 * around_coord.x) + (cell_eighth % 2),
          .y = (2 * around_coord.y) + ((cell_eighth / 2) % 2),
          .z = (2 * around_coord.z) + (cell_eighth / 4),
      };
      ApplyFaceSearchBlockInsert(search, FindBlockAtCoord(tables, size_level + 1, smaller_coord));
    }
  }
  // Step 3: the 27 positions of size level l - 1 around the position of the bigger block.
  if (size_level == 0) {
    return search;
  }
  const std::uint32_t bigger_block = tables.bigger_block[block];
  const CellCoord bigger_coord = {
      .x = ComputeBiggerCellCoord(block_coord.x, 1),
      .y = ComputeBiggerCellCoord(block_coord.y, 1),
      .z = ComputeBiggerCellCoord(block_coord.z, 1),
  };
  for (int entry = 0; entry < 27; ++entry) {
    if (bigger_block != kNoIndex) {
      const std::uint64_t around_entry = (static_cast<std::uint64_t>(bigger_block) * 27) + entry;
      ApplyFaceSearchBlockInsert(search, tables.same_size_level_blocks[around_entry]);
      continue;
    }
    const CellCoord around_bigger_coord = {
        .x = bigger_coord.x + (entry % 3) - 1,
        .y = bigger_coord.y + ((entry / 3) % 3) - 1,
        .z = bigger_coord.z + (entry / 9) - 1,
    };
    ApplyFaceSearchBlockInsert(search, FindBlockAtCoord(tables, size_level - 1, around_bigger_coord));
  }
  return search;
}

// Finds the next face piece of a face search: the next face piece, in ascending FacePlaceIndex, of the start's axis,
// of size level l - 1, l or l + 1 for the start's size level l, whose centre lies strictly within the radius of the
// start's centre on each axis. kNoIndex when the walk is finished, and on every later call.
//
// Method: reads the search's blocks in order, and in each the 80 face places of the start's axis in order, from where
//   the last call stopped; a place counts when it is a face piece (any kind but kNotAPiece) and lies within. "Within"
//   is tested on exact numbers. Both centres are counted in quarter widths of h_l from the origin, which are integers:
//   every face piece of size levels l - 1, l and l + 1 has its centre on that lattice. On each axis the distance, an
//   integer below 2^23 and so an exact double, is compared with 4 r, exact for the clamped radius r: |n_g - n_f| < 4 r.
//   That is |x_g - x_f| < r h_l without rounding. The frame's origin and width play no part, so the answer is the same
//   in every frame and on the CPU and the GPU. The distance is the same from either centre, so at one radius two face
//   pieces of one size level find each other or neither. Between size levels the radius counts the start's own width,
//   so a search from the smaller face piece stops at half the distance. This is not the half-open window of
//   <sec-nearby-faces>, which relates a particle to faces.
// Order: ascending FacePlaceIndex: the blocks ascend, and the places of one axis ascend within a block.
// Example: on a grid of one size level, cells 1 m wide, radius 1.5 around an X face piece away from the grid's edge
//   gives 27 face pieces, itself included: along the axis the planes 0 and 4 quarter widths away count and 8 does not
//   (the bound is 6); across, the centres 0 and 4 away count.
// Requires: tables as BuildFaceSearch. A null `search` gives kNoIndex. A FaceSearch not made by BuildFaceSearch for
//   these tables never makes a read outside the tables: a start that is no face piece of the tables gives kNoIndex,
//   entries of `blocks` at or beyond block_count and blocks of other size levels are skipped, block_count is read as at
//   most kMaxFaceSearchBlocks, and next_block below 0 or next_place outside 0..80 give kNoIndex. The face pieces it
//   gives then mean nothing. A search belongs to the tables it was made for: continued after the tables change, as for
//   a new grid version, its walk reads only places of the new tables, and the face pieces it gives mean nothing.
// Read: docs/method-paper/fluidity-method.typ <sec-face-search>, <sec-face-pieces>
FLUIDITY_CPU_AND_GPU constexpr FacePlaceIndex FindNextFace(const GridTables& tables, FaceSearch* search) noexcept {
  constexpr FacePlaceIndex kNoFace = {.value = kNoIndex};
  if (search == nullptr) {
    return kNoFace;
  }
  const FacePlaceIndex start = search->face;
  if (ToCellBlockIndex(start).value >= tables.block_count) {
    return kNoFace;
  }
  if (tables.face_piece_kinds[start.value] == FacePieceKind::kNotAPiece) {
    return kNoFace;
  }
  const bool is_next_block_in_range = search->next_block >= 0;
  const bool is_next_place_in_range = search->next_place >= 0 && search->next_place <= 80;
  if (!is_next_block_in_range || !is_next_place_in_range) {
    return kNoFace;
  }
  const int block_count = search->block_count < kMaxFaceSearchBlocks ? search->block_count : kMaxFaceSearchBlocks;
  const auto axis_offset = 80U * static_cast<std::uint32_t>(ToAxis(start));
  const auto start_size_level = static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(start)));
  const QuarterWidthPoint start_center = ComputeFaceCenterInQuarterWidths(tables, start, start_size_level);
  const double quarter_radius = 4.0 * ComputeClampedFaceSearchRadius(search->radius);
  for (; search->next_block < block_count; ++search->next_block, search->next_place = 0) {
    const std::uint32_t block = search->blocks[search->next_block].value;
    if (block >= tables.block_count) {
      continue;
    }
    const auto block_size_level = static_cast<int>(ToSizeLevel(tables, CellBlockIndex{block}));
    const bool is_size_level_near =
        block_size_level >= start_size_level - 1 && block_size_level <= start_size_level + 1;
    if (!is_size_level_near) {
      continue;
    }
    for (; search->next_place < 80; ++search->next_place) {
      const FacePlaceIndex face = {
          .value = (block * static_cast<std::uint32_t>(kFacePlacesPerBlock)) + axis_offset
              + static_cast<std::uint32_t>(search->next_place),
      };
      if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
        continue;
      }
      const QuarterWidthPoint face_center = ComputeFaceCenterInQuarterWidths(tables, face, start_size_level);
      if (ComputeIsWithinQuarterRadius(face_center, start_center, quarter_radius)) {
        ++search->next_place;
        return face;
      }
    }
  }
  return kNoFace;
}

}  // namespace fluidity::grid
