// The border of a set of blocks: the test of one block, and the marks one place of the set makes, private to grid/.
//
// Purpose: say which blocks of a whole grid belong to the border of a set of blocks, from the tables of one grid
//   version alone, in two forms that give the same set: asked from each block's side (ComputeIsBorderBlock), for the
//   CPU walk over the blocks, and run forward from each place of the set (ApplyCellPlaceReadMarks,
//   ApplyFacePlaceReadMarks), for a GPU pass of one thread per place.
//
// Idea: a stage writes values kept in the blocks of a set and may read the values kept in the border blocks. A run in
//   batches loads only the set and its border into a grid piece, where every neighbour entry naming a block not
//   loaded reads as none. So the border is the smallest set of blocks outside the set such that every reader of the
//   stage, run from any place kept in the set (an active cell, a face piece or a corner place), gives in that piece the
//   answer it gives on the whole grid: every block a reader reads, on the way or as a result. Three examples:
//   - a face piece between a small cell and a bigger cell is kept at the small cell's size level, at plane 0 of the
//     block above when the small cell sits at local 3 of the block below. A stage that writes that face piece and
//     reads its two cells (FindCellLow, FindCellHigh) reads the block below, which may hold no cell touching a cell
//     of the set;
//   - a corner is kept at the smallest size level whose lattice holds the point, possibly in a block that holds no
//     cell touching it. A stage that writes a cell's value and reads its corners (FindCornerPlaces) reads that block;
//   - FindCellAt walks down from the size-level-0 block through every bigger block of the cell it finds, so every
//     bigger block, up to size level 0, of a block of the set is read, though no place a reader returns is kept there.
//   Each reader reads places at most two steps away, and bigger blocks, so the relation "a reader starting in block
//   S reads block X" is bounded. ComputeIsBorderBlock asks it from X's side, through the readers themselves and the
//   grid's neighbour tables, with no other search structure; the marking functions run the readers forward.
//   Example: one size level, cells 1 m wide, active cells {3,3,0} (block A), {4,3,0} (block B) and {6,6,0} (block C),
//   the blocks in the order A, B, C. The cell {4,3,0} touches {3,3,0}, so B is in the border of {A}. Its +Y face
//   piece is plane 4 of B, which is plane 0 of C, so C is in the border of {A} too, though none of its cells touches
//   {3,3,0}. And B is in the border of {C}: that face piece is kept in C, and its cell_low is {4,3,0}; so is A, since
//   {3,3,0} touches {4,3,0} and its particles can list the face piece, which transfer reads.
//
// Words:
//   block set     the blocks a stage writes: a run of consecutive blocks, a sorted list of blocks, or both
//   border block  a block outside the block set that a reader reads, on the way or as a result, from a place kept in
//                 the set
//   border width  how far the readers of a stage read: kTouchingCells or kFourCellWidths
// Owns:    BorderWidth, BlockSet, ComputeIsInBlockSet, ComputeIsBorderBlock, ApplyCellPlaceReadMarks,
//          ApplyFacePlaceReadMarks, kPlacesPerSetBlock, ComputeBorderSoFarFailureKey, ApplySetPlaceReadMarks,
//          ComputeListedMark, ApplyBorderBlockWriteAt, kMaxSetBlocks, kSetBlocksPerMarkingLaunch, SetPlaceChunk,
//          ComputeIsSetWithinPlaceLimit, ComputeMarkingChunkCount, ComputeMarkingChunk; and, for this header's own
//          functions only,
//          ComputeSmallerBlockPosition, ComputeIsPositionWithin, ComputeIsBlockSetAroundOneBlock,
//          ComputeIsBlockSetUnderNear, ComputeIsBlockSetUnderWide, ComputeIsBlockSetAround,
//          ComputeHasFacePieceInBlockSet, ComputeIsBiggerCellBesideTouchingSet, ComputeIsCellOfBlockRead,
//          ComputeIsFacePieceOfBlockRead, ComputeReadCornerMask, ComputeIsReadCornerKeptIn, ComputeIsCornerOfBlockRead,
//          ComputeIsBlockInFaceSearchOfSetBlock, ComputeIsBlockInFaceSearchOfBlockSet, ComputeIsBlockReadDirectly,
//          kMaxWaitingBlocks, ApplyReadMark and ApplyPlaceReadMark
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), CellCorners, FindCornerPlaces (grid/cell_corners.h),
//          TouchingCells, CellFacePieces, FindTouchingCells, FindFacePieces (grid/touching_cells.h), FaceSearch,
//          BuildFaceSearch, CellsNearFace, FindCellsNearFace, kMaxFaceSearchRadius (grid/face_search.h), GridTables,
//          BlockLocalCoord, FindLocalCellPlace, FindCellLow, FindCellHigh, ToSizeLevel, ToAxis (grid/grid_tables.h),
//          ToFailureKey, kIdStep, kNoFailureKey (grid/grid_build_steps.h, grid/grid_build_status.h), the id types and
//          constants (grid/grid_ids.h)
// Used by: cell_block_range_cpu.cc, cell_block_range_gpu.cu, and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>, <sec-face-search>

#pragma once

#include <cstdint>
#include <limits>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {

// How far the readers of a stage read from the places kept in its blocks. At either width a stage calls, from a place
// kept in the set only, the readers listed here and no lookup by id, and the border also holds every bigger block,
// up to size level 0, of each block of the set and of each block read (ComputeIsBorderBlock).
//
// What a stage may call, from a place of its range, at each width:
//   - from an active cell: FindTouchingCells, FindFacePieces of it and of each touching cell, FindCornerPlaces,
//     FindCellsAroundEdge, FindCellAt for a position in its closed box, and FindNearbyFaces for a position that
//     FindCellAt gives to it;
//   - from a face piece: its two cells (FindCellLow, FindCellHigh), its corners and, for transfer, the cells near it at
//     most one size level bigger (FindCellsNearFace), where FindNearbyFaces can list it;
//   - from a corner place: ComputeCornerPosition;
//   - at kFourCellWidths also, from a face piece: every cell near it and the blocks of its face search.
// The transfer step. The border at kTouchingCells holds what a face piece of the range needs to weigh a particle, given
//   the particle's NearbyFaceSums: the sums are NOT in that promise. A particle's list holds face pieces of the
//   cells that touch its own cell, which can lie beyond the near border of another range. Example: the cells [0,1) and
//   [1,2) are active, [-1,0) and [2,3) are split. The range of the block of x in [2,4) holds the face piece at x = 2,
//   and a particle of [0,1) near x = 0.95 lists it, and also the face piece at x = -0.5, which is not in the border.
//   So a particle's sums are computed in the range that holds the particle's cell, where the cell is in the set, and
//   carried with the particle; a face piece of another range reads the carried sums. Each particle is handled by the
//   range that holds FindCellAt of the exact position its weights use: a particle grouped at an old position and moved
//   since must be grouped again first, or it may lie in a cell outside the range whose list goes one step further.
// Outside the rule. In a grid piece, a reader called for a place outside what the rule covers gives none, the answer
//   of the edge of the grid and of a place that is not there. FindCellAt for a position outside the range's cells may
//   give a border cell, when the piece was built with a border that reaches it, or kNoIndex when no loaded block holds
//   it; FindNearbyFaces then says kPositionOutsideGrid for the second. A stage must rely on neither answer for a place
//   outside its range, and keeps every particle inside the closed boxes of the cells of its range.
// Corners. No reader goes from a corner place to the cells that have it as a corner. A stage that computes a corner
//   value from the cells around the corner needs the blocks of those cells, which the border does not promise.
enum class BorderWidth : std::uint8_t {
  // Every reader above but the last one. The width of transfer and of every stage that reads only the places next to
  // the ones it writes.
  kTouchingCells,
  // Also, from a face piece: the blocks of a face search of four cell widths (BuildFaceSearch, FindNextFace) and every
  // cell near the face piece (FindCellsNearFace).
  kFourCellWidths,
};

// A set of blocks: the blocks of `range`, and the blocks of `sorted_blocks`. Either part may be empty.
struct BlockSet {
  CellBlockRange range;  // Block indexes first_block..first_block + block_count - 1; {0, 0} for none.
  const CellBlockIndex* sorted_blocks;  // sorted_count indexes in ascending order, each once; may be null for 0.
  std::uint32_t sorted_count;
};

// Says whether `block` is in `set`: in its range, or found by binary search in its sorted list.
// Example: the range {2, 3} and the list {7, 9} hold 2, 3, 4, 7 and 9, and no other block.
// Requires: set.sorted_blocks is ascending with no repeats.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsInBlockSet(const BlockSet& set, CellBlockIndex block) noexcept {
  const bool is_in_range =
      block.value >= set.range.first_block && block.value - set.range.first_block < set.range.block_count;
  if (is_in_range) {
    return true;
  }
  std::uint32_t low = 0;
  std::uint32_t high = set.sorted_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) / 2);
    if (set.sorted_blocks[middle].value < block.value) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < set.sorted_count && set.sorted_blocks[low].value == block.value;
}

// The position of a smaller block, relative to a block of the bigger size level: twice the bigger position, plus
// the halves of `cell_eighth` (x + 2 y + 4 z). For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr BlockLocalCoord
ComputeSmallerBlockPosition(BlockLocalCoord bigger_position, unsigned cell_eighth) noexcept {
  return {
      .x = (2 * bigger_position.x) + static_cast<int>(cell_eighth & 1U),
      .y = (2 * bigger_position.y) + static_cast<int>((cell_eighth >> 1U) & 1U),
      .z = (2 * bigger_position.z) + static_cast<int>(cell_eighth >> 2U),
  };
}

// Says whether each coordinate of `position` is lowest..highest. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsPositionWithin(BlockLocalCoord position, int lowest, int highest) noexcept {
  const bool is_x_within = position.x >= lowest && position.x <= highest;
  const bool is_y_within = position.y >= lowest && position.y <= highest;
  const bool is_z_within = position.z >= lowest && position.z <= highest;
  return is_x_within && is_y_within && is_z_within;
}

// Says whether the set holds one of the 27 blocks around `block`; false when `block` is kNoIndex. For this header's own
// functions only.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsBlockSetAroundOneBlock(const GridTables& tables, const BlockSet& set, std::uint32_t block) noexcept {
  if (block == kNoIndex) {
    return false;
  }
  for (int entry = 0; entry < 27; ++entry) {
    const std::uint32_t around_block = tables.same_size_level_blocks[(static_cast<std::uint64_t>(block) * 27) + entry];
    if (around_block != kNoIndex && ComputeIsInBlockSet(set, CellBlockIndex{around_block})) {
      return true;
    }
  }
  return false;
}

// Says whether the set holds one of the smaller blocks under `around_block`, at `around_offset` from the block asked
// about, that lie at -1..2 from that block on each axis. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsBlockSetUnderNear(
    const GridTables& tables,
    const BlockSet& set,
    std::uint32_t around_block,
    BlockLocalCoord around_offset
) noexcept {
  for (unsigned cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    if (!ComputeIsPositionWithin(ComputeSmallerBlockPosition(around_offset, cell_eighth), -1, 2)) {
      continue;
    }
    const std::uint32_t smaller_block =
        tables.smaller_blocks[(static_cast<std::uint64_t>(around_block) * 8) + cell_eighth];
    if (smaller_block != kNoIndex && ComputeIsInBlockSet(set, CellBlockIndex{smaller_block})) {
      return true;
    }
  }
  return false;
}

// Says whether the set holds one of the blocks under `around_block`, at `around_offset` from the block asked about,
// that a face search or FindCellsNearFace can start from and ComputeIsBlockSetUnderNear does not ask: the smaller
// blocks at -2 or 3 on some axis, and the smaller blocks, at -1..4, of the smaller blocks at -1..2. For this header's
// own functions only.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsBlockSetUnderWide(
    const GridTables& tables,
    const BlockSet& set,
    std::uint32_t around_block,
    BlockLocalCoord around_offset
) noexcept {
  for (unsigned cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    const BlockLocalCoord smaller_position = ComputeSmallerBlockPosition(around_offset, cell_eighth);
    const std::uint32_t smaller_block =
        tables.smaller_blocks[(static_cast<std::uint64_t>(around_block) * 8) + cell_eighth];
    if (smaller_block == kNoIndex) {
      continue;
    }
    if (!ComputeIsPositionWithin(smaller_position, -1, 2)) {
      if (ComputeIsInBlockSet(set, CellBlockIndex{smaller_block})) {
        return true;
      }
      continue;
    }
    for (unsigned second_eighth = 0; second_eighth < 8; ++second_eighth) {
      const bool is_second_smaller_block_near =
          ComputeIsPositionWithin(ComputeSmallerBlockPosition(smaller_position, second_eighth), -1, 4);
      const std::uint32_t second_smaller_block =
          tables.smaller_blocks[(static_cast<std::uint64_t>(smaller_block) * 8) + second_eighth];
      const bool is_in_set =
          second_smaller_block != kNoIndex && ComputeIsInBlockSet(set, CellBlockIndex{second_smaller_block});
      if (is_second_smaller_block_near && is_in_set) {
        return true;
      }
    }
  }
  return false;
}

// Says whether the set holds one of the blocks from which a reader of `width` can read `block`: the 27 blocks around
// it, around its bigger block and around that block's bigger block, and smaller blocks under the 27 around it. For
// this header's own functions only.
//
// Method: positions in blocks of each size level, relative to `block`. Under the block at offset d (each axis -1, 0
//   or +1) the smaller block of cell eighth e lies at 2 d + e, -2..3, and its own smaller block of eighth e' at
//   2 (2 d + e) + e'. At kTouchingCells the smaller blocks at -1..2 on each axis are asked, 64 at most; at
//   kFourCellWidths all 216, and under those at -1..2 their smaller blocks at -1..4, 216 at most. With the 3 x 27
//   blocks around, that is at most 145 searches of the set at kTouchingCells and 513 at kFourCellWidths. The proof
//   that no reader reads `block` from elsewhere is at ComputeIsBorderBlock.
// Requires: block is below tables.block_count; the tables are those of a whole grid.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsBlockSetAround(
    const GridTables& tables,
    const BlockSet& set,
    std::uint32_t block,
    BorderWidth width
) noexcept {
  for (int entry = 0; entry < 27; ++entry) {
    const std::uint32_t around_block = tables.same_size_level_blocks[(static_cast<std::uint64_t>(block) * 27) + entry];
    if (around_block == kNoIndex) {
      continue;
    }
    if (ComputeIsInBlockSet(set, CellBlockIndex{around_block})) {
      return true;
    }
    const BlockLocalCoord around_offset = {.x = (entry % 3) - 1, .y = ((entry / 3) % 3) - 1, .z = (entry / 9) - 1};
    if (ComputeIsBlockSetUnderNear(tables, set, around_block, around_offset)) {
      return true;
    }
    const bool is_four_cell_widths = width == BorderWidth::kFourCellWidths;
    if (is_four_cell_widths && ComputeIsBlockSetUnderWide(tables, set, around_block, around_offset)) {
      return true;
    }
  }
  const std::uint32_t bigger_block = tables.bigger_block[block];
  if (bigger_block == kNoIndex) {
    return false;
  }
  if (ComputeIsBlockSetAroundOneBlock(tables, set, bigger_block)) {
    return true;
  }
  return ComputeIsBlockSetAroundOneBlock(tables, set, tables.bigger_block[bigger_block]);
}

// Says whether one of the face pieces of the active cell at `cell` (FindFacePieces) of size level highest_size_level
// or less is kept in the set; kMaxSizeLevels counts every face piece. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeHasFacePieceInBlockSet(
    const GridTables& tables,
    const BlockSet& set,
    CellPlaceIndex cell,
    int highest_size_level
) noexcept {
  const CellFacePieces face_pieces = FindFacePieces(tables, cell);
  for (int entry = 0; entry < face_pieces.count; ++entry) {
    const CellBlockIndex face_block = ToCellBlockIndex(face_pieces.faces[entry]);
    const bool is_low_enough = static_cast<int>(ToSizeLevel(tables, face_block)) <= highest_size_level;
    if (is_low_enough && ComputeIsInBlockSet(set, face_block)) {
      return true;
    }
  }
  return false;
}

// Says whether an active cell one size level bigger than the active cell at `cell`, across one of its sides, touches
// a cell kept in the set. FindFacePieces of that bigger cell, run because it touches the set, then reads the block of
// `cell`. For this header's own functions only.
//
// Method: the side of the bigger cell is split at the size level of `cell`, so FindFacePieces of the bigger cell goes
//   through the split place to the block of its small cells, which holds `cell`, and reads its tables, even where the
//   face pieces it lists are kept in the next block (plane 4 of that block is plane 0 of its high neighbour). The face
//   piece between the two cells is a face piece of `cell` whose other cell is the bigger one, so the bigger cells
//   across the sides of `cell` are the other cells of its face pieces that are one size level bigger.
// Requires: `cell` is an active cell of the tables of a whole grid.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsBiggerCellBesideTouchingSet(const GridTables& tables, const BlockSet& set, CellPlaceIndex cell) noexcept {
  const int cell_size_level = static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(cell)));
  const CellFacePieces face_pieces = FindFacePieces(tables, cell);
  for (int entry = 0; entry < face_pieces.count; ++entry) {
    const FacePlaceIndex face = face_pieces.faces[entry];
    const CellPlaceIndex cell_low = FindCellLow(tables, face);
    const CellPlaceIndex other_cell = cell_low.value == cell.value ? FindCellHigh(tables, face) : cell_low;
    if (other_cell.value == kNoIndex) {
      continue;
    }
    if (static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(other_cell))) != cell_size_level - 1) {
      continue;
    }
    const TouchingCells touching = FindTouchingCells(tables, other_cell);
    for (int touching_entry = 0; touching_entry < touching.count; ++touching_entry) {
      if (ComputeIsInBlockSet(set, ToCellBlockIndex(touching.cells[touching_entry]))) {
        return true;
      }
    }
  }
  return false;
}

// Says whether a reader of `width` reads an active cell kept in `block`, or reads through one, from a place kept in the
// set. For this header's own functions only.
//
// Method: for each active cell q of the block, the reverse of each reader that gives cells or reads through them:
//   - FindCellLow and FindCellHigh of a face piece f kept in the set give q exactly when f is one of q's face pieces:
//     FindFacePieces lists the face pieces on q's sides, and a face piece lies on a side of its two cells only;
//   - FindTouchingCells of a cell of the set gives q exactly when that cell touches q, as touching is symmetric;
//   - FindCellsNearFace(f) lists f's two cells and their touching cells, so it gives q exactly when f is a face piece
//     of q or of a cell touching q. At kFourCellWidths every such q counts. At kTouchingCells only those at most one
//     size level bigger than f, the cells whose nearby faces can list f for transfer (proof at ComputeIsBorderBlock):
//     f of size level level(q) + 1 or less;
//   - FindFacePieces of a cell t that touches a cell of the set reads q's block when q lies across a side of t and is
//     one size level smaller (ComputeIsBiggerCellBesideTouchingSet).
//   A touching cell kept in the block itself is skipped: it is not in the set, and it is asked as q.
// Requires: `block` is below tables.block_count and not in the set; the tables are those of a whole grid.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsCellOfBlockRead(
    const GridTables& tables,
    const BlockSet& set,
    std::uint32_t block,
    BorderWidth width
) noexcept {
  const int block_size_level = static_cast<int>(ToSizeLevel(tables, CellBlockIndex{block}));
  // FindCellsNearFace counts a face piece of every size level at kFourCellWidths, and of size level q's + 1 at most
  // for transfer at kTouchingCells.
  const int highest_near_face_size_level =
      width == BorderWidth::kFourCellWidths ? kMaxSizeLevels : block_size_level + 1;
  for (int place_in_block = 0; place_in_block < kCellPlacesPerBlock; ++place_in_block) {
    const CellPlaceIndex cell = {.value = (block * static_cast<std::uint32_t>(kCellPlacesPerBlock)) + place_in_block};
    if (tables.cell_states[cell.value] != CellState::kActive) {
      continue;
    }
    if (ComputeHasFacePieceInBlockSet(tables, set, cell, kMaxSizeLevels)) {
      return true;
    }
    const TouchingCells touching = FindTouchingCells(tables, cell);
    for (int entry = 0; entry < touching.count; ++entry) {
      const CellBlockIndex touching_block = ToCellBlockIndex(touching.cells[entry]);
      if (touching_block.value == block) {
        continue;
      }
      if (ComputeIsInBlockSet(set, touching_block)) {
        return true;
      }
      if (ComputeHasFacePieceInBlockSet(tables, set, touching.cells[entry], highest_near_face_size_level)) {
        return true;
      }
    }
    if (ComputeIsBiggerCellBesideTouchingSet(tables, set, cell)) {
      return true;
    }
  }
  return false;
}

// Says whether a reader reads a face piece kept in `block` from a place kept in the set: FindFacePieces of a cell of
// the set or of a cell touching one, and so FindNearbyFaces. For this header's own functions only.
//
// Method: FindFacePieces of a cell lists a face piece f exactly when the cell is FindCellLow(f) or FindCellHigh(f), so
//   f is read exactly when one of its two cells is kept in the set or touches a cell kept in the set. A cell kept in
//   the block itself is not in the set, and ComputeIsCellOfBlockRead, asked first, found that it touches no cell of
//   the set, so only cells kept in other blocks are asked.
// Requires: `block` is below tables.block_count and not in the set; ComputeIsCellOfBlockRead gave false for it; the
//   tables are those of a whole grid.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsFacePieceOfBlockRead(const GridTables& tables, const BlockSet& set, std::uint32_t block) noexcept {
  for (int place_in_block = 0; place_in_block < kFacePlacesPerBlock; ++place_in_block) {
    const FacePlaceIndex face = {.value = (block * static_cast<std::uint32_t>(kFacePlacesPerBlock)) + place_in_block};
    if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
      continue;
    }
    // Side 0 is cell_low, side 1 cell_high.
    for (int face_side = 0; face_side < 2; ++face_side) {
      const CellPlaceIndex face_cell = face_side == 0 ? FindCellLow(tables, face) : FindCellHigh(tables, face);
      if (face_cell.value == kNoIndex || ToCellBlockIndex(face_cell).value == block) {
        continue;
      }
      if (ComputeIsInBlockSet(set, ToCellBlockIndex(face_cell))) {
        return true;
      }
      const TouchingCells touching = FindTouchingCells(tables, face_cell);
      for (int entry = 0; entry < touching.count; ++entry) {
        if (ComputeIsInBlockSet(set, ToCellBlockIndex(touching.cells[entry]))) {
          return true;
        }
      }
    }
  }
  return false;
}

// Finds which corners of the active cell at `cell` the set reads: all 8 when the cell is kept in the set, and the 4 on
// each side whose whole face piece, of the cell's size level, is kept in the set. Bit k is corner k of CellCorners:
// bit 0 of k the high x side, bit 1 high y, bit 2 high z. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr unsigned
ComputeReadCornerMask(const GridTables& tables, const BlockSet& set, CellPlaceIndex cell) noexcept {
  constexpr unsigned kEveryCorner = 0xFFU;
  if (ComputeIsInBlockSet(set, ToCellBlockIndex(cell))) {
    return kEveryCorner;
  }
  const SizeLevel cell_size_level = ToSizeLevel(tables, ToCellBlockIndex(cell));
  unsigned read_corner_mask = 0;
  const CellFacePieces face_pieces = FindFacePieces(tables, cell);
  for (int entry = 0; entry < face_pieces.count; ++entry) {
    const FacePlaceIndex face = face_pieces.faces[entry];
    const CellBlockIndex face_block = ToCellBlockIndex(face);
    const bool is_whole_side = ToSizeLevel(tables, face_block) == cell_size_level;
    if (!is_whole_side || !ComputeIsInBlockSet(set, face_block)) {
      continue;
    }
    // The cell is cell_high of the face piece on its low side.
    const bool is_high_side = FindCellHigh(tables, face).value != cell.value;
    const auto axis_number = static_cast<unsigned>(ToAxis(face));
    for (unsigned corner_number = 0; corner_number < 8; ++corner_number) {
      const bool is_corner_on_high_side = ((corner_number >> axis_number) & 1U) != 0;
      if (is_corner_on_high_side == is_high_side) {
        read_corner_mask |= 1U << corner_number;
      }
    }
  }
  return read_corner_mask;
}

// Says whether one of the corners of `cell` that `read_corner_mask` names is kept in `block` (FindCornerPlaces). For
// this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsReadCornerKeptIn(
    const GridTables& tables,
    CellPlaceIndex cell,
    unsigned read_corner_mask,
    CellBlockIndex block
) noexcept {
  const CellCorners cell_corners = FindCornerPlaces(tables, cell);
  for (unsigned corner_number = 0; corner_number < 8; ++corner_number) {
    const CornerPlaceIndex corner = cell_corners.corners[corner_number];
    const bool is_read = ((read_corner_mask >> corner_number) & 1U) != 0;
    if (is_read && corner.value != kNoIndex && ToCellBlockIndex(corner).value == block.value) {
      return true;
    }
  }
  return false;
}

// Says whether a corner place kept in `block` is a corner of an active cell kept in the set or of a face piece kept in
// the set (FindCornerPlaces of either form). For this header's own functions only.
//
// Method: a reverse search names every cell that can have a corner kept in the block. For each, the corners the set
//   reads are those of the cell when it is kept in the set, and those on each side whose whole face piece is kept in
//   the set (ComputeReadCornerMask); FindCornerPlaces of the cell, run forward, says whether one of them is kept in
//   the block (ComputeIsReadCornerKeptIn).
//   Which cells. Let m be the block's size level. A corner of an active cell of size level l is kept at size level
//   l, l + 1 or l + 2 (the bound proved at FindCornerPlaceOfPoint, cell_corners.h), so l is m - 2, m - 1 or m. The
//   corner is a point of the block's lattice, so the cell's closed box meets the block's closed box. At size level
//   m - s the block's box is the cells local_low..local_low + w - 1 of its s-th bigger block on each axis, with
//   w = 4 >> s and local_low = 4 (b mod 2^s) >> s, b the block's coordinate. The cells whose boxes meet it are at
//   local_low - 1..local_low + w: 6 x 6 x 6 = 216 places at m, 4 x 4 x 4 = 64 at m - 1 and 3 x 3 x 3 = 27 at m - 2,
//   307 in all, each local coordinate -1..4, found by FindLocalCellPlace through that bigger block's neighbours. A
//   place outside every block of its size level holds no cell there.
//   Which face pieces. A face piece f of size level l lies on a side of its smaller cell c, of size level l, and is
//   that whole side: across f lies a cell of size level l or l - 1, or nothing at the grid's edge, never a split place,
//   whose side would be four face pieces of size level l + 1. So FindFacePieces(c) lists f, the 4 corners of f are
//   the corners of c on that side, and FindCornerPlaces(f) finds the same corner places as FindCornerPlaces(c), the
//   same points at the same size level. When a corner of f is kept in the block, c is among the 307 places.
//   Exact both ways: a true answer is a corner place that FindCornerPlaces of a cell of the set, or of a face piece of
//   the set, gives; and every such cell and face piece with a corner kept in the block is tried.
// Requires: `block` is below tables.block_count and not in the set; the tables are those of a whole grid, which keeps
//   the size step rule and the siblings rule.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsCornerOfBlockRead(const GridTables& tables, const BlockSet& set, std::uint32_t block) noexcept {
  const CellCoord block_coord = ToCellCoord(CellId{.value = tables.block_ids[block].value});
  std::uint32_t bigger_block = block;
  for (unsigned size_level_step = 0; size_level_step <= 2; ++size_level_step) {
    if (size_level_step > 0) {
      bigger_block = tables.bigger_block[bigger_block];
    }
    if (bigger_block == kNoIndex) {
      return false;
    }
    // The low bits of a coordinate as unsigned are its residue, for a negative coordinate too.
    const std::uint32_t residue_mask = (1U << size_level_step) - 1U;
    const BlockLocalCoord local_low = {
        .x = static_cast<int>((4U * (static_cast<std::uint32_t>(block_coord.x) & residue_mask)) >> size_level_step),
        .y = static_cast<int>((4U * (static_cast<std::uint32_t>(block_coord.y) & residue_mask)) >> size_level_step),
        .z = static_cast<int>((4U * (static_cast<std::uint32_t>(block_coord.z) & residue_mask)) >> size_level_step),
    };
    // The places local_low - 1..local_low + w on each axis, w + 2 of them, numbered x first.
    const int places_per_axis = static_cast<int>(static_cast<unsigned>(kBlockWidthInCells) >> size_level_step) + 2;
    for (int place_number = 0; place_number < places_per_axis * places_per_axis * places_per_axis; ++place_number) {
      const BlockLocalCoord local = {
          .x = local_low.x - 1 + (place_number % places_per_axis),
          .y = local_low.y - 1 + ((place_number / places_per_axis) % places_per_axis),
          .z = local_low.z - 1 + (place_number / (places_per_axis * places_per_axis)),
      };
      const std::uint32_t place = FindLocalCellPlace(tables, bigger_block, local);
      if (place == kNoIndex || tables.cell_states[place] != CellState::kActive) {
        continue;
      }
      const unsigned read_corner_mask = ComputeReadCornerMask(tables, set, CellPlaceIndex{place});
      const bool is_read_corner_kept_in_block = read_corner_mask != 0
          && ComputeIsReadCornerKeptIn(tables, CellPlaceIndex{place}, read_corner_mask, CellBlockIndex{block});
      if (is_read_corner_kept_in_block) {
        return true;
      }
    }
  }
  return false;
}

// Says whether `candidate_block`, kNoIndex for none, is in the set and keeps a face piece whose face search lists
// `block`. One face piece stands for all: BuildFaceSearch chooses blocks from the start's block alone. For this
// header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsBlockInFaceSearchOfSetBlock(
    const GridTables& tables,
    const BlockSet& set,
    std::uint32_t candidate_block,
    CellBlockIndex block
) noexcept {
  if (candidate_block == kNoIndex || !ComputeIsInBlockSet(set, CellBlockIndex{candidate_block})) {
    return false;
  }
  for (int place_in_block = 0; place_in_block < kFacePlacesPerBlock; ++place_in_block) {
    const FacePlaceIndex start = {
        .value = (candidate_block * static_cast<std::uint32_t>(kFacePlacesPerBlock)) + place_in_block,
    };
    if (tables.face_piece_kinds[start.value] == FacePieceKind::kNotAPiece) {
      continue;
    }
    const FaceSearch search = BuildFaceSearch(tables, start, kMaxFaceSearchRadius);
    for (int entry = 0; entry < search.block_count; ++entry) {
      if (search.blocks[entry].value == block.value) {
        return true;
      }
    }
    return false;
  }
  return false;
}

// Says whether the block, outside the set, is in the block list that BuildFaceSearch makes for a face piece kept in
// the set. For this header's own functions only.
//
// Method: BuildFaceSearch lists blocks by position from the start's block alone, never from which face piece of it is
//   the start, so one face piece of a block stands for all of its face pieces. A block with no face piece makes no
//   search. The candidates are the 27 blocks around `block`, the 8 smaller blocks of each and the 27 blocks around its
//   bigger block, 270 at most, and each one in the set that has a face piece is searched with BuildFaceSearch itself
//   (ComputeIsBlockInFaceSearchOfSetBlock), so this function and the search read one rule.
//   Why the candidates hold every block whose search lists `block`. Let S be that block, of size level l, B its
//   coordinate, and let `block` be at position X of size level m. BuildFaceSearch lists three sets of positions:
//   (i) m = l, X = B + d with d in -1..1 on each axis: S = X - d is one of the 27 positions around X;
//   (ii) m = l - 1, X = floor(B / 2) + d: S lies under the position X - d of size level m, one of the 8 smaller
//     blocks of one of the 27 positions around X;
//   (iii) m = l + 1, X = 2 (B + d) + (0 or 1): the position floor(X / 2) = B + d has S = floor(X / 2) - d among its
//     27 positions around, as a block of size level m - 1 around the bigger block of `block`.
//   In a whole grid every position named is a block when it holds a block of the set: a block of size level 1 or
//   more has its bigger block, and a smaller block lies under an existing block. So the neighbour tables find S.
// Requires: `block` is below tables.block_count and not in the set; the tables are those of a whole grid.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsBlockInFaceSearchOfBlockSet(const GridTables& tables, const BlockSet& set, std::uint32_t block) noexcept {
  for (int entry = 0; entry < 27; ++entry) {
    const std::uint32_t around_block = tables.same_size_level_blocks[(static_cast<std::uint64_t>(block) * 27) + entry];
    if (around_block == kNoIndex) {
      continue;
    }
    if (ComputeIsBlockInFaceSearchOfSetBlock(tables, set, around_block, CellBlockIndex{block})) {
      return true;
    }
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      const std::uint32_t smaller_block =
          tables.smaller_blocks[(static_cast<std::uint64_t>(around_block) * 8) + cell_eighth];
      if (ComputeIsBlockInFaceSearchOfSetBlock(tables, set, smaller_block, CellBlockIndex{block})) {
        return true;
      }
    }
  }
  const std::uint32_t bigger_block = tables.bigger_block[block];
  if (bigger_block == kNoIndex) {
    return false;
  }
  for (int entry = 0; entry < 27; ++entry) {
    const std::uint32_t around_block =
        tables.same_size_level_blocks[(static_cast<std::uint64_t>(bigger_block) * 27) + entry];
    if (ComputeIsBlockInFaceSearchOfSetBlock(tables, set, around_block, CellBlockIndex{block})) {
      return true;
    }
  }
  return false;
}

// Says whether a reader of `width` reads `block` directly from a place kept in the set: it gives or lists a place kept
// in the block, or it goes through the block to small cells. The bigger blocks a reader walks through are added by
// ComputeIsBorderBlock. For this header's own functions only.
//
// Method: the block is asked about each place it keeps, in the reverse direction: its active cells
//   (ComputeIsCellOfBlockRead), its face pieces (ComputeIsFacePieceOfBlockRead), its corner places
//   (ComputeIsCornerOfBlockRead) and, at kFourCellWidths, its place in a face search
//   (ComputeIsBlockInFaceSearchOfBlockSet). Each is exact (its own Method).
//   Why the look around the block first (ComputeIsBlockSetAround) loses nothing. Count positions in cells of the
//   block's size level m, the block covering 0..4 on each axis. A place that a reader of the set starts from lies at
//   most two touching steps from a place of the block (c touches q; f lies on a side of a cell that is, or touches, a
//   cell of the block; a cell touching the set lies across a side of a cell of the block; a corner of c is a corner of
//   the block's lattice), and the size step rule keeps it between size levels m - 2 and m + 2. So it is kept in a
//   block of size level m - 2 or m - 1 around the block's first or second bigger block (it lies within -6..10), of
//   size level m among the 27 around it, or of size level m + 1 among the smaller blocks at -1..2 under those 27: a
//   cell of size level m + 1 touching a cell of the block or touching a cell of one of its face pieces, and a face
//   piece of size level m + 1 or less beside such a cell, lie at -3..10 in cells of size level m + 1. At
//   kFourCellWidths a face search can start from any of the 8 smaller blocks under each of the 27, and
//   FindCellsNearFace from a face piece of size level m + 2 on a side of a cell of size level m + 1 touching a cell of
//   the block, at -2..18 in its own cells: one of the smaller blocks at -1..4 under the smaller blocks at -1..2. When
//   the set holds none of these blocks, no reader of the set reads the block directly.
// Requires: block is below tables.block_count and not in the set; the tables are those of a whole grid.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsBlockReadDirectly(
    const GridTables& tables,
    const BlockSet& set,
    std::uint32_t block,
    BorderWidth width
) noexcept {
  if (!ComputeIsBlockSetAround(tables, set, block, width)) {
    return false;
  }
  if (ComputeIsCellOfBlockRead(tables, set, block, width)) {
    return true;
  }
  if (ComputeIsFacePieceOfBlockRead(tables, set, block)) {
    return true;
  }
  if (ComputeIsCornerOfBlockRead(tables, set, block)) {
    return true;
  }
  if (width == BorderWidth::kFourCellWidths) {
    return ComputeIsBlockInFaceSearchOfBlockSet(tables, set, block);
  }
  return false;
}

// The most blocks ComputeIsBorderBlock keeps waiting to be asked: the walk down from a size-level-0 block leaves at
// most 7 waiting at each of 15 size levels it has passed and lists 8 more, 113, which fits 8 for each of the
// 16 size levels. For this header's own functions only.
inline constexpr int kMaxWaitingBlocks = 8 * kMaxSizeLevels;

// Says whether `block` is a border block of `set` at `width`: outside the set, and read by a reader of that width
// from a place kept in the set, an active cell, a face piece or a corner place, either directly or on the way.
//
// Method: the readers, by the place they start from, and the blocks each reads (method paper <sec-cell-blocks>,
//   "Ranges and border"):
//   kTouchingCells. From an active cell c: FindTouchingCells(c); FindFacePieces of c and of each cell t touching c,
//   which also goes through the block of the small cells across a split side of t; FindCornerPlaces(c);
//   FindCellsAroundEdge(c), whose cells touch c; FindCellAt for a position in c's closed box, which gives c or a cell
//   touching c; FindNearbyFaces for a position that FindCellAt gives to c, which lists face pieces of c and of the
//   cells touching c. From a face piece f: FindCellLow(f), FindCellHigh(f) and FindCornerPlaces(f); and, for transfer,
//   the cells of FindCellsNearFace(f) at most one size level bigger than f, with what FindNearbyFaces reads from a
//   position in such a cell q to list f. A cell two size levels bigger than f never lists it: FindNearbyFaces keeps
//   only face pieces of size levels level(q) - 1 to level(q) + 1, and level(f) = level(q) + 2. FindNearbyFaces lists
//   f from q through FindCellAt (q's bigger blocks), FindTouchingCells(q) to the cell of f that is q or touches q,
//   and FindFacePieces of that cell to f, which read only q, the cells of f, f and their bigger blocks. From a
//   corner place: ComputeCornerPosition, which reads its own block only.
//   kFourCellWidths. Also, from a face piece f: the block list of BuildFaceSearch(f), which holds every face piece
//   FindNextFace gives, and every cell of FindCellsNearFace(f).
//   On the way. FindCellAt walks down from the size-level-0 block through every bigger block of the cell it gives.
//   FindCellLow, FindCellHigh, FindTouchingCells and FindCellsAroundEdge find a bigger cell through the bigger block
//   of the block they start from. FindTouchingCells and FindFacePieces find small cells through the block that holds
//   the split place, the bigger block of the small cells' block. So the border also holds every bigger block, up to
//   size level 0, of each block of the set and of each block read directly. Some of these bigger blocks are on no
//   walk, such as the bigger block of a block that only keeps a face piece of a touching cell: the rule takes them
//   all, since which ones a walk passes would need a case for each reader, and they are few: about one border block
//   in twelve for one block of the random test grids.
//   Why the border holds every block FindNextFace reads. FindNextFace of a search made by BuildFaceSearch for a face
//   piece `start` kept in the set reads the blocks in search.blocks, and in each the face places of the start's axis,
//   nothing else. A radius below 4 changes which face pieces of the listed blocks are given, never which blocks are
//   listed, so it reads fewer blocks and is covered.
//   The test. The block is a border block when it is read directly (ComputeIsBlockReadDirectly), or when a block under
//   it, at any depth, is in the set or is read directly: then it is a bigger block of that one. The blocks under it
//   are asked through smaller_blocks, depth first, at most kMaxWaitingBlocks waiting at once.
// Work: per block asked, a fixed number of table reads, whatever the grid's size, apart from the binary searches:
//   log2 of the set's sorted count for each search of the set, and log2 of the grid's block count for each
//   FindCellBlock inside FindCornerPlaces. At most, per block asked: the look around, 145 searches of the set at
//   kTouchingCells and 513 at kFourCellWidths; per active cell of the block (64), FindFacePieces, FindTouchingCells,
//   FindFacePieces of each of its up to 56 touching cells and up to 24 + 56 x 25 searches, then for each of up to 3
//   bigger cells across its sides a FindCellLow, a FindCellHigh, a FindTouchingCells and 56 searches; per face piece
//   (240), FindCellLow, FindCellHigh, two FindTouchingCells and 114 searches; for the corners, 307 places of 2 reads,
//   and per active one FindFacePieces, up to 25 searches and 24 FindCellHigh, and one FindCornerPlaces of up to 192
//   FindCellBlock searches; at kFourCellWidths up to 270 BuildFaceSearch. FindTouchingCells reads about 350 table
//   entries and moves up to about 1,500 entries in its sorted inserts; FindFacePieces reads about 80 and moves up to
//   about 300; BuildFaceSearch reads about 300 and moves up to about 36,000. So a block near the set costs up to
//   about 3 x 10^6 reads and moves at kTouchingCells, and up to about 10^7 at kFourCellWidths, where the 270 face
//   searches dominate; a block that the look around clears costs at most its 145 or 513 searches. The blocks asked
//   are the block and every block under it, so a size-level-0 block of a large grid asks every block under it: this
//   form suits the CPU walk and the tests; ApplyCellPlaceReadMarks and ApplyFacePlaceReadMarks are the forward form.
// Example: the example of the file header: block C (index 2) is a border block of {block 0} at kTouchingCells, and
//   block B (index 1) is one of {block 2}. With 2 size levels, the size-level-0 cells {0..3}^3 all active but
//   {0..2}^3, which are split, the size-level-0 block is a border block of the size-level-1 block (0,0,0): FindCellAt
//   walks down from it.
// Requires: block is below tables.block_count; the tables are those of a whole grid on the CPU, or the same tables on
//   the GPU.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsBorderBlock(const GridTables& tables, const BlockSet& set, CellBlockIndex block, BorderWidth width) noexcept {
  if (ComputeIsInBlockSet(set, block)) {
    return false;
  }
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  std::uint32_t waiting_blocks[kMaxWaitingBlocks] = {};
  int waiting_count = 1;
  waiting_blocks[0] = block.value;
  while (waiting_count > 0) {
    --waiting_count;
    const std::uint32_t asked_block = waiting_blocks[waiting_count];
    const bool is_under_block = asked_block != block.value;
    if (is_under_block && ComputeIsInBlockSet(set, CellBlockIndex{asked_block})) {
      return true;
    }
    if (ComputeIsBlockReadDirectly(tables, set, asked_block, width)) {
      return true;
    }
    // Only a smaller block one size level down is followed, so tables that contradict themselves cannot make the walk
    // loop or pass the array: the depth is at most 16 size levels.
    const int asked_size_level = static_cast<int>(ToSizeLevel(tables, CellBlockIndex{asked_block}));
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      const std::uint32_t smaller_block =
          tables.smaller_blocks[(static_cast<std::uint64_t>(asked_block) * 8) + cell_eighth];
      const bool is_next_size_level = smaller_block < tables.block_count
          && static_cast<int>(ToSizeLevel(tables, CellBlockIndex{smaller_block})) == asked_size_level + 1;
      if (is_next_size_level && waiting_count < kMaxWaitingBlocks) {
        waiting_blocks[waiting_count] = smaller_block;
        ++waiting_count;
      }
    }
  }
  return false;
}

// Marks `block` and each of its bigger blocks, up to size level 0: marks[b] = 1 for each, and nothing else is
// written. kNoIndex marks nothing. It reads no mark: it writes the whole chain every time, at most kMaxSizeLevels
// bytes, so threads that run it at once only write the value 1 to bytes they never read. For this header's own
// functions only.
FLUIDITY_CPU_AND_GPU constexpr void
ApplyReadMark(const GridTables& tables, std::uint32_t block, std::uint8_t* marks) noexcept {
  for (int size_level_step = 0; size_level_step < kMaxSizeLevels && block < tables.block_count; ++size_level_step) {
    marks[block] = 1;
    block = tables.bigger_block[block];
  }
}

// Marks the block of `place`, a cell, face or corner place, and each of its bigger blocks (ApplyReadMark). kNoIndex
// marks nothing. For this header's own functions only.
template <typename PlaceIndex>
FLUIDITY_CPU_AND_GPU constexpr void
ApplyPlaceReadMark(const GridTables& tables, PlaceIndex place, std::uint8_t* marks) noexcept {
  if (place.value != kNoIndex) {
    ApplyReadMark(tables, ToCellBlockIndex(place).value, marks);
  }
}

// Marks every block that a reader of `width` reads from the cell place `cell`, with each block's bigger blocks: the
// forward form of the border, run from one place. A cell place of any kind marks its block and that block's bigger
// blocks, so the bigger blocks of every block of a set are marked; an active cell also marks what its readers read.
//
// Method: the readers from an active cell c, as ComputeIsBorderBlock lists them, run forward: the blocks of c, of
//   each cell of FindTouchingCells(c), of each face piece FindFacePieces gives for c and for each touching cell t, of
//   the small cell across each face piece of a touching cell t one size level smaller than t (the block FindFacePieces
//   goes through at a split side of t; at a split side of c that cell touches c), and of each corner of
//   FindCornerPlaces(c). Each is marked with its bigger blocks (ApplyReadMark). FindCellsAroundEdge, FindCellAt and
//   FindNearbyFaces read nothing more.
//   The width does not change what an active cell reads; it is taken so both forms read alike.
//   What a GPU pass must do. Two threads that write the value 1 to one byte at once are formally a data race in the
//   CUDA memory model, and the result is right only because of how a byte store works, so the pass keeps to this: each
//   thread writes only the value 1 to a byte and never reads a mark (ApplyReadMark reads none), and the launch that
//   scans the marks, to list the border, starts after the marking launch has finished. No atomic is used, and the marks
//   hold no other value, so a pass must not set marks for a set or for a border found earlier in the same array.
// Ties: only the value 1 is written, so threads that run this for different places of one set at once, into the
//   same marks, cannot conflict, and the marks do not depend on the order of the places.
// Work: FindTouchingCells, 57 FindFacePieces, up to 56 x 24 x 2 FindCellLow or FindCellHigh, one FindCornerPlaces
//   (8 corners, each up to 8 blocks at each of up to 3 size levels: up to 192 FindCellBlock searches) and up to about
//   2,800 marks of at most 16 reads each: up to about 10^5 table reads, moves and writes. It allocates nothing. A set
//   of n places costs n times this, whatever the grid's size: the border of a set is found by calling this for each
//   place of the set and one pass over the marks.
// Example: in the grid of the file header, cell {3,3,0} of block A marks A, B (the touching cell {4,3,0}) and C (the
//   +Y face piece of {4,3,0}, plane 0 of C).
// Requires: `cell` is a cell place of block 0..tables.block_count - 1; `marks` holds tables.block_count entries, each 0
//   or written by these marking functions; the tables are those of a whole grid.
FLUIDITY_CPU_AND_GPU constexpr void ApplyCellPlaceReadMarks(
    const GridTables& tables,
    CellPlaceIndex cell,
    BorderWidth width,
    std::uint8_t* marks
) noexcept {
  static_cast<void>(width);
  ApplyReadMark(tables, ToCellBlockIndex(cell).value, marks);
  if (tables.cell_states[cell.value] != CellState::kActive) {
    return;
  }
  const TouchingCells touching = FindTouchingCells(tables, cell);
  for (int entry = -1; entry < touching.count; ++entry) {
    const CellPlaceIndex read_cell = entry < 0 ? cell : touching.cells[entry];
    ApplyPlaceReadMark(tables, read_cell, marks);
    const int read_cell_size_level = static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(read_cell)));
    const CellFacePieces face_pieces = FindFacePieces(tables, read_cell);
    for (int face_entry = 0; face_entry < face_pieces.count; ++face_entry) {
      const FacePlaceIndex face = face_pieces.faces[face_entry];
      const CellBlockIndex face_block = ToCellBlockIndex(face);
      ApplyReadMark(tables, face_block.value, marks);
      // The small cell across a split side of c itself touches c, so it is marked already.
      const bool is_small_side_of_touching_cell =
          entry >= 0 && static_cast<int>(ToSizeLevel(tables, face_block)) > read_cell_size_level;
      if (is_small_side_of_touching_cell) {
        const CellPlaceIndex cell_low = FindCellLow(tables, face);
        ApplyPlaceReadMark(tables, cell_low.value == read_cell.value ? FindCellHigh(tables, face) : cell_low, marks);
      }
    }
  }
  for (const CornerPlaceIndex corner : FindCornerPlaces(tables, cell).corners) {
    ApplyPlaceReadMark(tables, corner, marks);
  }
}

// Marks every block that a reader of `width` reads from the face place `face`, with each block's bigger blocks: the
// forward form of the border, run from one place. A face place that holds no face piece marks nothing.
//
// Method: the readers from a face piece f, as ComputeIsBorderBlock lists them, run forward: the blocks of f, of its
//   two cells, of its corners (FindCornerPlaces), and of the cells of FindCellsNearFace(f): at kTouchingCells those
//   at most one size level bigger than f, for transfer; at kFourCellWidths all of them, and every block of the list
//   of BuildFaceSearch(f) at the largest radius. Each is marked with its bigger blocks (ApplyReadMark).
// Ties: only the value 1 is written and no mark is read, as ApplyCellPlaceReadMarks, whose Method says what a GPU pass
//   must do.
// Work: FindCellLow, FindCellHigh, FindCornerPlaces (4 corners, each up to 8 blocks at each of up to 3 size levels: up
//   to 96 FindCellBlock searches), FindCellsNearFace (2 FindTouchingCells and up to 112 sorted inserts) and up to 119
//   marks; at kFourCellWidths also BuildFaceSearch and up to 270 marks: up to about 4 x 10^4 table reads, moves and
//   writes. It allocates nothing.
// Example: in the grid of the file header, the +Y face piece of {4,3,0}, kept in C, marks C, B (the block of its
//   cell_low) and A: the cell {3,3,0} touches that cell_low, so its particles can list the face piece.
// Requires: `face` is a face place of block 0..tables.block_count - 1; `marks` holds tables.block_count entries, each 0
//   or written by these marking functions; the tables are those of a whole grid.
FLUIDITY_CPU_AND_GPU constexpr void ApplyFacePlaceReadMarks(
    const GridTables& tables,
    FacePlaceIndex face,
    BorderWidth width,
    std::uint8_t* marks
) noexcept {
  if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
    return;
  }
  const CellBlockIndex face_block = ToCellBlockIndex(face);
  ApplyReadMark(tables, face_block.value, marks);
  ApplyPlaceReadMark(tables, FindCellLow(tables, face), marks);
  ApplyPlaceReadMark(tables, FindCellHigh(tables, face), marks);
  for (const CornerPlaceIndex corner : FindCornerPlaces(tables, face).corners) {
    ApplyPlaceReadMark(tables, corner, marks);
  }
  const bool is_four_cell_widths = width == BorderWidth::kFourCellWidths;
  // Transfer reads the cells whose nearby faces can list f: size level f's - 1 or more.
  const int lowest_near_cell_size_level = static_cast<int>(ToSizeLevel(tables, face_block)) - 1;
  const CellsNearFace cells_near_face = FindCellsNearFace(tables, face);
  for (int entry = 0; entry < cells_near_face.count; ++entry) {
    const CellBlockIndex near_block = ToCellBlockIndex(cells_near_face.cells[entry]);
    if (is_four_cell_widths || static_cast<int>(ToSizeLevel(tables, near_block)) >= lowest_near_cell_size_level) {
      ApplyReadMark(tables, near_block.value, marks);
    }
  }
  if (!is_four_cell_widths) {
    return;
  }
  const FaceSearch search = BuildFaceSearch(tables, face, kMaxFaceSearchRadius);
  for (int entry = 0; entry < search.block_count; ++entry) {
    ApplyReadMark(tables, search.blocks[entry].value, marks);
  }
}

// The places of one block of a set that the forward form runs a thread for: its 64 cell places, then its 240
// face places.
inline constexpr std::uint32_t kPlacesPerSetBlock =
    static_cast<std::uint32_t>(kCellPlacesPerBlock) + static_cast<std::uint32_t>(kFacePlacesPerBlock);

// The most blocks a set can hold so that every place of the set is a 32-bit number: ApplySetPlaceReadMarks takes its
// place as a std::uint32_t. This is 14,128,181 blocks. The grid's own limit on blocks (kMaxCellBlocksLimit) keeps only
// the 240 face places of each block in 32 bits, not these 304, so a set of a range plus a long border so far can pass
// this count, and a place beyond it would wrap onto a low place and leave its reads unmarked.
inline constexpr std::uint32_t kMaxSetBlocks = std::numeric_limits<std::uint32_t>::max() / kPlacesPerSetBlock;
static_assert(
    std::uint64_t{kMaxSetBlocks} * kPlacesPerSetBlock - 1 <= std::numeric_limits<std::uint32_t>::max(),
    "the last place of the largest set is a 32-bit number"
);
static_assert(
    (std::uint64_t{kMaxSetBlocks} + 1) * kPlacesPerSetBlock - 1 > std::numeric_limits<std::uint32_t>::max(),
    "one more block would have a place that is not a 32-bit number"
);

// The most blocks of the set that one marking launch of the GPU form runs places for. The value is an estimate, not a
// measurement: this GPU code has not run yet. The GPU function that marks keeps 255 registers, so one thread block of
// 256 threads is resident on each multiprocessor, about 9,000 threads at once on a 36-multiprocessor card; a dense
// block should cost that card about 70 to 270 microseconds (up to 10^5 dependent table reads for each of its active
// cell places); so 2,048 blocks should take about 0.15 to 0.55 s, far below the 2 s after which the display watchdog
// of Windows ends a GPU function, and 622,592 threads are still many passes of the resident ones. A longer set runs
// in several launches. The marks do not depend on the order of the places, so the chunks change no result.
inline constexpr std::uint32_t kSetBlocksPerMarkingLaunch = 2048;

// The places of one chunk of a set: the places first_set_place to first_set_place + set_place_count - 1, in the numbers
// ApplySetPlaceReadMarks takes.
struct SetPlaceChunk {
  std::uint32_t first_set_place = 0;
  std::uint32_t set_place_count = 0;
};

// clang-tidy: two counts of blocks, or a count and an index; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
// Computes whether a set of range_block_count + border_so_far_count blocks has every place in 32 bits: the sum, made in
// 64 bits so a huge count cannot wrap, is at most kMaxSetBlocks. Both processors ask it before they write anything.
// Example: 14,128,181 blocks fit; 14,128,182 do not.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsSetWithinPlaceLimit(std::uint64_t range_block_count, std::uint64_t border_so_far_count) noexcept {
  return range_block_count + border_so_far_count <= kMaxSetBlocks;
}

// Computes how many chunks of set_blocks_per_chunk blocks cover a set of set_block_count blocks: the last chunk may be
// shorter. A set of no block has no chunk.
// Example: 5 blocks in chunks of 2 give 3.
// Requires: set_blocks_per_chunk is at least 1; set_block_count is at most kMaxSetBlocks.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
ComputeMarkingChunkCount(std::uint32_t set_block_count, std::uint32_t set_blocks_per_chunk) noexcept {
  return (set_block_count / set_blocks_per_chunk) + (set_block_count % set_blocks_per_chunk == 0 ? 0U : 1U);
}

// Computes the places of chunk `chunk_index`: its blocks are set_blocks_per_chunk blocks of the set from block
// chunk_index x set_blocks_per_chunk, or fewer at the end of the set, and their places are kPlacesPerSetBlock each.
// Together the chunks below ComputeMarkingChunkCount hold every place of the set once, in order. A chunk index at or
// above the count gives no place.
// Example: 5 blocks in chunks of 2: chunk 2 is block 4 alone, places 1,216 to 1,519.
// Requires: set_blocks_per_chunk is at least 1; set_block_count is at most kMaxSetBlocks.
FLUIDITY_CPU_AND_GPU constexpr SetPlaceChunk ComputeMarkingChunk(
    std::uint32_t set_block_count,
    std::uint32_t chunk_index,
    std::uint32_t set_blocks_per_chunk
) noexcept {
  const std::uint64_t first_block = std::uint64_t{chunk_index} * set_blocks_per_chunk;
  if (first_block >= set_block_count) {
    return {.first_set_place = 0, .set_place_count = 0};
  }
  const std::uint64_t blocks_left = set_block_count - first_block;
  const std::uint64_t chunk_blocks = blocks_left < set_blocks_per_chunk ? blocks_left : set_blocks_per_chunk;
  return {
      .first_set_place = static_cast<std::uint32_t>(first_block * kPlacesPerSetBlock),
      .set_place_count = static_cast<std::uint32_t>(chunk_blocks * kPlacesPerSetBlock),
  };
}

// NOLINTEND(bugprone-easily-swappable-parameters)

// clang-tidy: a count of blocks and an entry index; the swappable-parameters check flags the two integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
// Computes the failure key of entry `index` of the border so far, or kNoFailureKey: step 2 of the GPU pass, one thread
// for each entry, and the check of the CPU form, which asks it of every entry.
//
// Method: an entry fails when it is not above the entry before it (the first entry has none), when it is at or above
//   `block_count`, or when it is a block of `range`. The key is ToFailureKey(kIdStep, index, 0), whose step 1 is the
//   one check of the pass; the smallest key over the entries names the first failing entry. The list is bad input
//   whatever the entry, so no other detail is kept.
// Example: range {0, 1}, 5 blocks, entries {2, 4, 3}: entry 2 fails (3 is not above 4), the key is
//   ToFailureKey(1, 2, 0). Entries {2, 0}: entry 1 fails twice over (below the entry before, and in the range), the
//   key is the same one.
// Requires: `index` is below the entry count; `border_so_far` holds entries up to it.
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t ComputeBorderSoFarFailureKey(
    CellBlockRange range,
    const CellBlockIndex* border_so_far,
    std::uint32_t block_count,
    std::uint32_t index
) noexcept {
  const std::uint32_t block = border_so_far[index].value;
  const bool is_not_above_entry_before = index > 0 && border_so_far[index - 1].value >= block;
  const bool is_beyond_grid = block >= block_count;
  const bool is_in_range = block >= range.first_block && block - range.first_block < range.block_count;
  if (is_not_above_entry_before || is_beyond_grid || is_in_range) {
    return ToFailureKey(kIdStep, index, 0);
  }
  return kNoFailureKey;
}

// Marks every block that a reader of `width` reads from one place of a set: step 3 of the GPU pass, one thread for each
// place, and the loop of the CPU form. The set is the blocks of `range`, then the blocks of `border_so_far`.
//
// Method: set_place counts the places of the set in order, kPlacesPerSetBlock for each block. The block is entry
//   set_place / kPlacesPerSetBlock of the set; the place in that block, set_place % kPlacesPerSetBlock, is a cell place
//   below 64 (ApplyCellPlaceReadMarks) and else the face place that many less 64 (ApplyFacePlaceReadMarks). A
//   set_place at or above the set's places does nothing.
// Ties: as the two marking functions: only the value 1 is written and no mark is read, so threads that run this for
//   different places at once, into the same marks, cannot conflict, and the marks do not depend on their order. The
//   marks must hold 0 before the first place and nothing may mark the blocks of the set beforehand. The places run in
//   chunks (ComputeMarkingChunk), which changes nothing for the same reason.
// Example: range {4, 2}, border_so_far {9}: set_place 0 is cell place 0 of block 4, set_place 304 is cell place 0 of
//   block 5, set_place 608 is cell place 0 of block 9, and set_place 64 is face place 0 of block 4.
// Requires: the set holds at most kMaxSetBlocks blocks (ComputeIsSetWithinPlaceLimit), so set_place does not wrap;
//   border_so_far holds border_so_far_count entries that passed ComputeBorderSoFarFailureKey; `marks` holds
//   tables.block_count entries, each 0 or written by these marking functions; the tables are those of a whole grid.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>
FLUIDITY_CPU_AND_GPU constexpr void ApplySetPlaceReadMarks(
    const GridTables& tables,
    CellBlockRange range,
    const CellBlockIndex* border_so_far,
    std::uint32_t border_so_far_count,
    std::uint32_t set_place,
    BorderWidth width,
    std::uint8_t* marks
) noexcept {
  const std::uint32_t set_block_entry = set_place / kPlacesPerSetBlock;
  const std::uint32_t place_in_block = set_place % kPlacesPerSetBlock;
  if (set_block_entry >= range.block_count + border_so_far_count) {
    return;
  }
  const std::uint32_t block = set_block_entry < range.block_count
      ? range.first_block + set_block_entry
      : border_so_far[set_block_entry - range.block_count].value;
  constexpr auto kCellPlaces = static_cast<std::uint32_t>(kCellPlacesPerBlock);
  constexpr auto kFacePlaces = static_cast<std::uint32_t>(kFacePlacesPerBlock);
  if (place_in_block < kCellPlaces) {
    ApplyCellPlaceReadMarks(tables, CellPlaceIndex{(block * kCellPlaces) + place_in_block}, width, marks);
    return;
  }
  ApplyFacePlaceReadMarks(tables, FacePlaceIndex{(block * kFacePlaces) + place_in_block - kCellPlaces}, width, marks);
}

// NOLINTEND(bugprone-easily-swappable-parameters)

// Computes the listed mark of `block`: 1 when the block is marked and outside `range`, else 0. Step 4 of the GPU pass,
// one thread for each block, run after every mark of step 3 was written; the prefix sum then gives each listed block
// its place.
// Example: range {1, 2}, marks {1, 1, 1, 1, 0}: blocks 0 and 3 give 1, blocks 1 and 2 (in the range) and 4 give 0.
// Requires: `block` is below the marks' count.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
ComputeListedMark(const std::uint8_t* marks, CellBlockRange range, std::uint32_t block) noexcept {
  const bool is_in_range = block >= range.first_block && block - range.first_block < range.block_count;
  return marks[block] != 0 && !is_in_range ? 1U : 0U;
}

// Writes block `block` of the border list when it is listed: step 6 of the GPU pass, one thread for each block, run
// after the prefix sum of the listed marks.
//
// Method: `sums` holds the exclusive prefix sums of the listed marks, with the total one past the last block, so the
//   block is listed exactly when sums[block + 1] differs from sums[block], and it is entry sums[block] of the list. An
//   entry at or above `border_block_room` is not written, but still counted in the total.
// Example: listed marks {1, 0, 0, 1}, so sums {0, 1, 1, 1, 2}: block 0 writes entry 0 and block 3 writes entry 1; with
//   a room of 1, block 3 writes nothing and the total is still 2.
// Requires: sums has block + 2 entries or more; border_blocks holds border_block_room entries.
FLUIDITY_CPU_AND_GPU constexpr void ApplyBorderBlockWriteAt(
    const std::uint32_t* sums,
    std::uint32_t block,
    std::uint32_t border_block_room,
    CellBlockIndex* border_blocks
) noexcept {
  const std::uint32_t entry = sums[block];
  if (sums[block + 1] != entry && entry < border_block_room) {
    border_blocks[entry] = CellBlockIndex{block};
  }
}

}  // namespace fluidity::grid
