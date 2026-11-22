// The active cells that touch one active cell, and the face pieces of its sides.
//
// Purpose: give every stage the list of active cells that share a side, an edge or a corner with a cell, and the
//   list of the face pieces of its six sides, from the tables of one grid version, on the CPU or the GPU.
//
// Idea: around a cell. The 26 places of its size level around an active cell give its touching cells: the cell there,
//   the bigger active cell that covers the place, or the small cells of a split place next to it, at most 56 in all
//   (FindTouchingCells). Each of its six sides is one face piece, or four kept one size level smaller beside a split
//   place, at most 24 (FindFacePieces). Both lists are plain fixed arrays in ascending index order.
//   The two-size example used below: 2 size levels, biggest cells 1 m wide, origin 0; the size-level-0 cell {0,0,0}
//   active beside the 8 size-level-1 cells of the size-level-0 cell {1,0,0}. Block 0 is the size-level-0 block (0,0,0)
//   and block 1 the size-level-1 block (0,0,0). The big cell's +X side is four face pieces of 0.25 m2, kept at size
//   level 1 (X places 2, 2 + 5, 2 + 20 and 2 + 25 of block 1). Each has FindCellLow = the big cell, at place 0, and a
//   centre distance of 0.5 + 0.25 = 0.75 m.
//
// Words:
//   active cell     a cell that exists and is not split; it carries values
//   face piece      the part of a face between two cells or at the grid's edge, kept at the size level of its smaller
//                   cell. Not a grid piece, not a memory piece
//   touching cell   an active cell that shares a side, an edge or a corner with a given cell
// Owns:    kMaxTouchingCells, kMaxCellFacePieces, TouchingCells, CellFacePieces, FindTouchingCells, FindFacePieces;
//          and, for this header's own functions only, CellSide, SmallCellHalves, PlaceOffset, ComputeCoordAlongAxis,
//          FindHighSideFacePlace, FindSmallCellBlock, ApplyTouchingSmallCells and ApplyFacePiecesBesideSplitPlace
// Uses:    Axis (base/axis.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), GridTables, BlockLocalCoord, FacePlaceParts,
//          ToBlockLocalCoord, ComputeMovedAlongAxis, ToFacePlaceIndex, ApplyAscendingInsert,
//          ComputeIsActiveCellPlace, FindLocalCellPlace, FindActiveCellHolding, FindCellBlock, ToSizeLevel
//          (grid/grid_tables.h), the id types and constants (grid/grid_ids.h)
// Used by: face_search.h, the grid's builders and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-face-pieces>

#pragma once

#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// The most active cells that touch one cell under the size step rule: 4 across each of its 6 sides, 2 across each of
// its 12 edges and 1 across each of its 8 corners (FindTouchingCells).
inline constexpr int kMaxTouchingCells = 56;

// The most face pieces of one cell under the size step rule: 4 on each of its 6 sides (FindFacePieces).
inline constexpr int kMaxCellFacePieces = 24;

// The active cells that touch one cell, as FindTouchingCells gives them. A default-made value is an empty list: count
// 0 and every entry kNoIndex.
struct TouchingCells {
  FLUIDITY_CPU_AND_GPU constexpr TouchingCells() noexcept {
    for (CellPlaceIndex& entry : cells) {
      entry = CellPlaceIndex{kNoIndex};
    }
  }

  // The places of the touching cells, cells[0..count - 1], in ascending index order; kNoIndex from count on.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  CellPlaceIndex cells[kMaxTouchingCells] = {};
  int count = 0;  // 0..kMaxTouchingCells.
};

// The face pieces of one cell's six sides, as FindFacePieces gives them. A default-made value is an empty list: count
// 0 and every entry kNoIndex.
struct CellFacePieces {
  FLUIDITY_CPU_AND_GPU constexpr CellFacePieces() noexcept {
    for (FacePlaceIndex& entry : faces) {
      entry = FacePlaceIndex{kNoIndex};
    }
  }

  // The face places of the face pieces, faces[0..count - 1], in ascending index order; kNoIndex from count on.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  FacePlaceIndex faces[kMaxCellFacePieces] = {};
  int count = 0;  // 0..kMaxCellFacePieces.
};

// One side of a cell: its axis, and -1 for the low side or +1 for the high side. For this header's own functions only.
struct CellSide {
  Axis axis;
  int side_step;
};

// The half, 0 or 1, of a small cell of a split place on each axis: 0 for the low half, 1 for the high half. For this
// header's own functions only.
struct SmallCellHalves {
  int x;
  int y;
  int z;
};

// An offset of -1, 0 or +1 cell places on each axis, toward one of the 26 places around a cell. For this header's own
// functions only.
struct PlaceOffset {
  int x;
  int y;
  int z;
};

// Gives the coordinate of `local` along `axis`. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr int ComputeCoordAlongAxis(BlockLocalCoord local, Axis axis) noexcept {
  if (axis == Axis::kX) {
    return local.x;
  }
  return axis == Axis::kY ? local.y : local.z;
}

// Finds the face place that keeps the face piece on the high side, along `axis`, of local place `local` (0..3 on each
// axis) of `block`: plane local + 1 of the block, except that plane 4 is plane 0 of the block of the same size level
// on the high side when that block exists, as the face piece is kept once (<sec-face-pieces>). For this header's own
// functions only.
FLUIDITY_CPU_AND_GPU constexpr FacePlaceIndex
FindHighSideFacePlace(const GridTables& tables, std::uint32_t block, BlockLocalCoord local, Axis axis) noexcept {
  const BlockLocalCoord plane_local = ComputeMovedAlongAxis(local, axis, 1);
  if (ComputeCoordAlongAxis(plane_local, axis) == kBlockWidthInCells) {
    // Entry 13 is the block itself; +1, +3 and +9 step one block along x, y and z.
    int entry = 13 + 9;
    if (axis == Axis::kX) {
      entry = 13 + 1;
    } else if (axis == Axis::kY) {
      entry = 13 + 3;
    }
    const std::uint32_t high_block = tables.same_size_level_blocks[(static_cast<std::uint64_t>(block) * 27) + entry];
    if (high_block != kNoIndex) {
      return ToFacePlaceIndex({.block = high_block, .local = ComputeMovedAlongAxis(local, axis, -3)}, axis);
    }
  }
  return ToFacePlaceIndex({.block = block, .local = plane_local}, axis);
}

// Finds the block that holds the small cells of a split cell place: the smaller block of its block for the cell
// eighth of its local coordinates. kNoIndex when the tables have none, which a built grid never gives. For this
// header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
FindSmallCellBlock(const GridTables& tables, CellPlaceIndex split_place) noexcept {
  const BlockLocalCoord local = ToBlockLocalCoord(split_place);
  const int cell_eighth = (local.x / 2) + (2 * (local.y / 2)) + (4 * (local.z / 2));
  return tables.smaller_blocks[(static_cast<std::uint64_t>(ToCellBlockIndex(split_place).value) * 8) + cell_eighth];
}

// Inserts into `touching` the active small cells of a split place that touch the cell at offset -`offset` from it:
// on each axis, the low half of the split place where the offset is +1, the high half where it is -1, both where it is
// 0. So 4 small cells across a side, 2 across an edge, 1 across a corner. A small place that is not kActive is left
// out. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr void ApplyTouchingSmallCells(
    const GridTables& tables,
    CellPlaceIndex split_place,
    PlaceOffset offset,
    TouchingCells& touching
) noexcept {
  const std::uint32_t small_block = FindSmallCellBlock(tables, split_place);
  if (small_block == kNoIndex) {
    return;
  }
  const BlockLocalCoord split_local = ToBlockLocalCoord(split_place);
  // The halves of each axis: from (offset < 0 ? 1 : 0) to (offset > 0 ? 0 : 1).
  const auto first_half = [](int axis_offset) {
    return axis_offset < 0 ? 1 : 0;
  };
  const auto last_half = [](int axis_offset) {
    return axis_offset > 0 ? 0 : 1;
  };
  for (int half_z = first_half(offset.z); half_z <= last_half(offset.z); ++half_z) {
    for (int half_y = first_half(offset.y); half_y <= last_half(offset.y); ++half_y) {
      for (int half_x = first_half(offset.x); half_x <= last_half(offset.x); ++half_x) {
        // A small cell's local coordinate is twice the split place's parity plus its half, 0..3.
        const BlockLocalCoord small_local = {
            .x = (2 * (split_local.x % 2)) + half_x,
            .y = (2 * (split_local.y % 2)) + half_y,
            .z = (2 * (split_local.z % 2)) + half_z,
        };
        const std::uint32_t small_place = FindLocalCellPlace(tables, small_block, small_local);
        if (tables.cell_states[small_place] == CellState::kActive) {
          ApplyAscendingInsert(touching.cells, touching.count, CellPlaceIndex{small_place});
        }
      }
    }
  }
}

// Finds every active cell that shares a side, an edge or a corner with an active cell: its touching cells.
//
// Method: looks at the 26 places of the cell's size level l around it, through the same-size-level neighbours of its
//   block for a place beyond the block. Each place gives:
//   - kActive: that cell, of size level l;
//   - kSplit: its small cells, of size level l + 1, that touch the cell: 4 across a side, 2 across an edge, 1 across a
//     corner. The size step rule makes them active and leaves no smaller cell there; a small place that is not kActive
//     is left out;
//   - kInsideBiggerCell, or no block of size level l there: the active cell of size level l - 1 that covers the place,
//     when there is one; one bigger cell may cover up to 4 of the 26 places and is listed once;
//   - kAbsent: nothing.
//   Every touching cell is found this way. A cell of size level l - 1 covers 2 x 2 x 2 aligned places of size level l,
//   so one that touches the cell covers at least one of the 26 places. A cell of size level l + 1 that touches it lies
//   inside one of them. The size step rule leaves no other size.
//   The list has at most 6 x 4 + 12 x 2 + 8 x 1 = 56 = kMaxTouchingCells entries, reached when the 26 places are all
//   split: a place adds at most as many cells as a split place there would, 4, 2 or 1. That holds whatever the tables'
//   cell states say, so even tables that contradict themselves never make the list pass its array.
// Order: ascending CellPlaceIndex.
// Example: in the two-size example above, the big cell, place 0, touches the four size-level-1 cells with x = 2, at
//   places 64 + 2, 64 + 4 + 2, 64 + 16 + 2 and 64 + 20 + 2: count 4.
// Requires: every block index entry of the tables (same_size_level_blocks, bigger_block, smaller_blocks) is kNoIndex
//   or below block_count, as in every built grid. Any `cell`: a place that is not kActive (split, inside a bigger cell
//   or absent), an index of no place (kNoIndex, or a block at or beyond block_count) and the tables of no grid
//   (block_count 0) give count 0 and read nothing beyond the tables. Every entry from count on is kNoIndex, so the
//   result has the same bytes on the CPU and the GPU.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-sizes>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr TouchingCells FindTouchingCells(const GridTables& tables, CellPlaceIndex cell) noexcept {
  TouchingCells touching;
  if (!ComputeIsActiveCellPlace(tables, cell)) {
    return touching;
  }
  const std::uint32_t block = ToCellBlockIndex(cell).value;
  const BlockLocalCoord local = ToBlockLocalCoord(cell);
  for (int offset_z = -1; offset_z <= 1; ++offset_z) {
    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
      for (int offset_x = -1; offset_x <= 1; ++offset_x) {
        const bool is_cell_itself = offset_x == 0 && offset_y == 0 && offset_z == 0;
        if (is_cell_itself) {
          continue;
        }
        // Each local coordinate is -1..4, which FindLocalCellPlace and FindActiveCellHolding take on every axis.
        const BlockLocalCoord around_local = {
            .x = local.x + offset_x,
            .y = local.y + offset_y,
            .z = local.z + offset_z,
        };
        const std::uint32_t around_place = FindLocalCellPlace(tables, block, around_local);
        if (around_place != kNoIndex && tables.cell_states[around_place] == CellState::kSplit) {
          const PlaceOffset offset = {.x = offset_x, .y = offset_y, .z = offset_z};
          ApplyTouchingSmallCells(tables, CellPlaceIndex{around_place}, offset, touching);
          continue;
        }
        const std::uint32_t holding_place = FindActiveCellHolding(tables, block, around_local);
        if (holding_place != kNoIndex) {
          ApplyAscendingInsert(touching.cells, touching.count, CellPlaceIndex{holding_place});
        }
      }
    }
  }
  return touching;
}

// Inserts into `face_pieces` the four face pieces between a split place and the cell across one of its sides, `side`
// of the cell: side_step +1 when the split place lies on the cell's high side, -1 on its low side. Each is the side of
// one small cell of the split place that faces the cell, its low side for +1 and its high side for -1, found where
// <sec-face-pieces> keeps it. A face place of kind kNotAPiece, which only a grid piece gives there, is left out. For
// this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr void ApplyFacePiecesBesideSplitPlace(
    const GridTables& tables,
    CellPlaceIndex split_place,
    CellSide side,
    CellFacePieces& face_pieces
) noexcept {
  const std::uint32_t small_block = FindSmallCellBlock(tables, split_place);
  if (small_block == kNoIndex) {
    return;
  }
  const BlockLocalCoord split_local = ToBlockLocalCoord(split_place);
  // Along the axis, the half next to the cell: the low half when the split place lies above it, the high half below.
  const int half_along_axis = side.side_step > 0 ? 0 : 1;
  for (int small_cell_number = 0; small_cell_number < 4; ++small_cell_number) {
    // One small cell next to the cell: across the axis, both halves of each of the two other axes, in x, y, z order.
    const int first_half = small_cell_number % 2;
    const int second_half = small_cell_number / 2;
    SmallCellHalves halves = {.x = first_half, .y = second_half, .z = half_along_axis};
    if (side.axis == Axis::kX) {
      halves = {.x = half_along_axis, .y = first_half, .z = second_half};
    } else if (side.axis == Axis::kY) {
      halves = {.x = first_half, .y = half_along_axis, .z = second_half};
    }
    // A small cell's local coordinate is twice the split place's parity plus its half, 0..3.
    const BlockLocalCoord small_local = {
        .x = (2 * (split_local.x % 2)) + halves.x,
        .y = (2 * (split_local.y % 2)) + halves.y,
        .z = (2 * (split_local.z % 2)) + halves.z,
    };
    const FacePlaceIndex small_face = side.side_step > 0
        ? ToFacePlaceIndex({.block = small_block, .local = small_local}, side.axis)
        : FindHighSideFacePlace(tables, small_block, small_local, side.axis);
    if (tables.face_piece_kinds[small_face.value] != FacePieceKind::kNotAPiece) {
      ApplyAscendingInsert(face_pieces.faces, face_pieces.count, small_face);
    }
  }
}

// Finds the face pieces of an active cell's six sides.
//
// Method: on each side, the face place of the cell's size level l that the side lies on: the plane of the cell's local
//   coordinate for its low side, the next plane for its high side, where plane 4 is plane 0 of the block of size level
//   l on the high side when that block exists. When that place holds a face piece (any kind but kNotAPiece), it is the
//   side's one face piece: beside an active cell of size level l or l - 1, or at the grid's edge. When it does not, the
//   place across the side is split, and the side is four face pieces at size level l + 1: the sides of its four small
//   cells that face the cell, each found where <sec-face-pieces> keeps it, plane 4 included.
//   In a grid piece a kOutsidePiece face piece is listed like any other, and a side whose face pieces lie in a block
//   the grid piece did not load gives none.
//   The list has at most 6 x 4 = 24 = kMaxCellFacePieces entries, reached when the 6 side places are all split: a side
//   gives one face piece or the four of its split place. That holds whatever the tables' cell states and face piece
//   kinds say, so even tables that contradict themselves never make the list pass its array.
// Order: ascending FacePlaceIndex.
// Example: in the two-size example above, the big cell, place 0, has 9: its -X, -Y, +Y, -Z and +Z sides whole, at X
//   place 0 and Y places 80 and 80 + 4 and Z places 160 and 160 + 16 of block 0, and its +X side as the 4 face pieces
//   at X places 2, 2 + 5, 2 + 20 and 2 + 25 of block 1, 240 + 2 and so on.
// Requires: as FindTouchingCells. Any `cell`: a place that is not kActive, an index of no place and the tables of no
//   grid give count 0 and read nothing beyond the tables. Every entry from count on is kNoIndex, so the result has the
//   same bytes on the CPU and the GPU.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellFacePieces FindFacePieces(const GridTables& tables, CellPlaceIndex cell) noexcept {
  CellFacePieces face_pieces;
  if (!ComputeIsActiveCellPlace(tables, cell)) {
    return face_pieces;
  }
  const std::uint32_t block = ToCellBlockIndex(cell).value;
  const BlockLocalCoord local = ToBlockLocalCoord(cell);
  for (int axis_number = 0; axis_number < 3; ++axis_number) {
    // Side step -1 is the low side, +1 the high side.
    for (int side_step = -1; side_step <= 1; side_step += 2) {
      const CellSide side = {.axis = static_cast<Axis>(axis_number), .side_step = side_step};
      const FacePlaceIndex whole_side_face = side_step > 0
          ? FindHighSideFacePlace(tables, block, local, side.axis)
          : ToFacePlaceIndex({.block = block, .local = local}, side.axis);
      if (tables.face_piece_kinds[whole_side_face.value] != FacePieceKind::kNotAPiece) {
        ApplyAscendingInsert(face_pieces.faces, face_pieces.count, whole_side_face);
        continue;
      }
      const std::uint32_t across_place =
          FindLocalCellPlace(tables, block, ComputeMovedAlongAxis(local, side.axis, side_step));
      if (across_place != kNoIndex && tables.cell_states[across_place] == CellState::kSplit) {
        ApplyFacePiecesBesideSplitPlace(tables, CellPlaceIndex{across_place}, side, face_pieces);
      }
    }
  }
  return face_pieces;
}

}  // namespace fluidity::grid
