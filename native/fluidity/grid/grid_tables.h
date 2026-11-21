// What code on either processor reads about one grid: its frame, the state of each place, the tables that hold
// them, and the addressing and geometry of its places.
//
// Purpose: declare the frame that places the grid in the world, the element types of the grid's per-place tables,
//   and GridTables, the one way code on the CPU or the GPU reads the tables of one grid version; and give every stage
//   one set of functions that turn places into ids, positions, widths and areas and back, and that find the active
//   cell that contains a position.
//
// Idea: a grid's frame says where the cell with coordinate (0, 0, 0) starts, how wide the biggest cells are, and how
//   many size levels the grid uses; a cell of size level l is h_0 / 2^l wide. Each cell place of a cell block holds
//   one CellState: an active cell, a split cell, a place covered by an active cell one size level bigger, or no cell.
//   Each face place holds one FacePieceKind, which says whether a face piece is kept there and what lies on its two
//   sides. Every face piece is kept once, at the size level of its smaller cell. Beside the per-place tables, each
//   block lists the blocks around it: its 27 neighbours of the same size level, the block of the next bigger cells
//   that covers it, and the 8 blocks of the next smaller cells that it covers. A search is a binary search in the
//   block ids followed by moves through these tables.
//   Example: an active cell beside four active cells one size level smaller has, on that side, four face places of
//   kind kBetweenCells at the smaller size level, and its own face place there is kNotAPiece.
//   Reading the tables. A place index gives its block, and the block id its size level and block coordinate, so a
//   place's coordinate, width and centre follow with no search, by integer arithmetic and a few double operations. An
//   id gives its place back by one binary search in the block ids (FindCellBlock). A face piece's two cells are the
//   cell places beside it, or, where such a place lies inside a bigger cell, that bigger active cell, found through
//   the bigger block. An id is made only from a place that holds a cell or a face piece: other places give the id 0.
//   A position gives its active cell by one floor per axis and a walk down from its size-level-0 block (FindCellAt).
//   Precision. Write a width as s 2^e with s an odd integer of k bits, the bits its significand needs; h_0 and every
//   h_l have the same s. A width h_l is exact: each halving of a double is (unless h_0 is below 2^-1007, where halving
//   leaves the normal range). A centre or a face position is origin + m h_l, with m = c or c + 1/2 exact: two
//   correctly rounded operations, a product then a sum (no fused multiply-add: contraction is off on both
//   processors). It is exact when m h_l and its sum with the origin are both doubles, as when the origin and h_0 are
//   multiples of a power of two with few significant bits (origin 0 and h_0 = 1 m for every grid in range).
//   Otherwise it lies within 2^-53 (|m h_l| + |result|) of origin + m h_l. An area h_l^2 is one correctly rounded
//   product; s^2 needs at most 2 k bits, so it is exact when k <= 26. A centre distance adds two exact half widths
//   once: h_l, exact, between cells of one size; 3/2 h_l beside a bigger cell, where 3 s needs k + 1 or k + 2 bits,
//   so it is correctly rounded, and exact when k <= 51.
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
//   grid piece      one range of cell blocks of a whole grid with its border, built as a grid of its own
//   bigger block    of a block of size level l, the block of size level l - 1 whose cells cover it
//   smaller blocks  of a block of size level l, the 8 blocks of size level l + 1 that its cells cover, one per
//                   eighth of the block
// Owns:    GridFrame, CellState, FacePieceKind, GridTables, FindSortedBlock, ComputeCellWidth,
//          ToSizeLevel (of a block), FindCellBlock, ToAxis, ComputeCellCenter, ComputeFacePosition, ComputeFaceArea,
//          ToCellId (of a place), ToFacePieceId (of a place), FindCellLow, FindCellHigh, ComputeCellCenterDistance,
//          FindCellPlace, FindFacePlace, FindCellAt (two forms); and, for this header's own functions only,
//          BlockLocalCoord, FacePlaceParts, ToFacePlaceParts, ComputeLocalPlaceCoord, ComputeHighPlaceCoord,
//          ComputeSideBeyondBlock, FindLocalCellPlace, FindActiveCellHolding, kSmallestCellCoordBias,
//          BiasedSmallestCellCoord, ToBiasedSmallestCellCoord, ToBiggestBlockId, FindActiveCellBelow,
//          FindBiggestBlockFrom and FindCellAtCoord
// Uses:    Axis (base/axis.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), math::Floor, math::IsFinite
//          (base/scalar_math.h), Vec3 (base/vec3.h), CellId, CellBlockId, FacePieceId, CellBlockIndex,
//          CellPlaceIndex, FacePlaceIndex, GridVersion, CellCoord, SizeLevel, kNoIndex, kMaxCoord, kMaxSizeLevels,
//          kBlockWidthInCells, kCellPlacesPerBlock, kFacePlacesPerBlock, ComesBefore, ToCellId,
//          ToSizeLevel, ToCellCoord, ToCellBlockId, ToFacePieceId, ToCellBlockIndex, ToCellPlaceIndex,
//          ComputeBiggerCellCoord and the id bit layout (grid/grid_ids.h)
// Used by: the grid's buffers and builders, and every module that reads the grid's tables
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-face-pieces>, <sec-cell-blocks>,
//          <sec-nearby-faces>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_ids.h"

namespace fluidity::grid {

// Where a grid sits in the world and how big its cells are.
struct GridFrame {
  Vec3 origin{};  // m, world. Low corner of the cell with coordinate (0, 0, 0).
  double biggest_cell_width = 0;  // m, positive. The width h_0 of a cell of size level 0.
  std::uint8_t size_level_count = 0;  // 1..16. Size level size_level_count - 1 holds the smallest cells.
};

// What one cell place of a cell block holds.
enum class CellState : std::uint8_t {
  kAbsent,  // No cell here: outside the part of space the grid keeps.
  kInsideBiggerCell,  // Covered by an active cell one size level bigger.
  kActive,  // A cell that exists and is not split. It carries values.
  kSplit,  // Replaced by its 8 small cells, which all exist.
};

// What one face place of a cell block holds.
enum class FacePieceKind : std::uint8_t {
  kNotAPiece,  // No face piece is kept here, so no flow is counted here.
  kBetweenCells,  // An active cell on each side; the two differ by at most one size level.
  kGridEdgeLow,  // An active cell on the high side only. What lies beyond is not the grid's to say.
  kGridEdgeHigh,  // An active cell on the low side only.
  // In a grid piece only: a kBetweenCells face piece of the whole grid whose other cell lies in a block that the grid
  // piece did not load.
  kOutsidePiece,
};

// The tables of one grid version on one processor, as raw pointers, so shared code on either processor reads them. A
// GPU function takes it by value. It owns nothing: the arrays live in the grid's buffers, whose room may hold more
// blocks than block_count, and every array is read only for blocks 0..block_count - 1.
// Layouts, by block index b (CellBlockIndex), offset (dx, dy, dz) with each of -1, 0, +1, and cell eighth e:
//   same_size_level_blocks[b * 27 + ((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1)], entry 13 the block itself;
//   bigger_block[b]; smaller_blocks[b * 8 + e]; cell_states[64 b + (4 z + y) * 4 + x];
//   face_piece_kinds[240 b + 80 axis + the place on that axis's 5 planes of 4 x 4]: (z * 4 + y) * 5 + x for X,
//   (z * 5 + y) * 4 + x for Y, (z * 4 + y) * 4 + x for Z, where the coordinate along the axis is the plane 0..4 and
//   plane p is the low side of the cell place p.
// Every block index entry is kNoIndex where no such block exists.
// A cell place that is kInsideBiggerCell or kAbsent may lie outside -kMaxCoord..kMaxCoord at its size level (only
// -2^18, local 0 of a block at block coordinate -2^16 on that axis), so an id is made from a cell place only when it
// is kActive or kSplit.
struct GridTables {
  GridFrame frame{};
  GridVersion version{};  // 1 or more for a built grid. 0 with null arrays when the processor holds no tables.
  std::uint32_t block_count = 0;  // The blocks of this version.
  const CellBlockId* block_ids = nullptr;  // block_count ids, in the fixed order.
  // 27 per block: the blocks of the same size level around it, by offset in z, then y, then x.
  const std::uint32_t* same_size_level_blocks = nullptr;
  const std::uint32_t* bigger_block = nullptr;  // 1 per block; kNoIndex at size level 0.
  // 8 per block: the smaller blocks, by the eighth of the block they lie in, x + 2 y + 4 z of its low or high halves.
  const std::uint32_t* smaller_blocks = nullptr;
  const CellState* cell_states = nullptr;  // 64 per block, one per cell place.
  const FacePieceKind* face_piece_kinds = nullptr;  // 240 per block, one per face place.
};

// Finds a cell block in the block ids of a grid, sorted in the fixed order.
//
// Method: binary search with ComesBefore for the first id that does not come before `block`, then a comparison of the
//   ids. The grid's build and FindCellBlock both search with it.
// Example: in the blocks {size level 0 (0,0,0), size level 1 (0,0,0)}, size level 1 (0,0,0) is at index 1.
// Requires: sorted_blocks holds count ids in the fixed order; it may be null when count is 0.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellBlockIndex
FindSortedBlock(const CellBlockId* sorted_blocks, std::uint32_t count, CellBlockId block) noexcept {
  assert(sorted_blocks != nullptr || count == 0);
  std::uint32_t low = 0;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) / 2);
    if (ComesBefore(sorted_blocks[middle], block)) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const bool is_found = low < count && sorted_blocks[low].value == block.value;
  return {.value = is_found ? low : kNoIndex};
}

// Computes the width h_l = h_0 / 2^l of a cell of size_level, in m.
//
// Method: halves biggest_cell_width size_level times. Each halving of a double is exact, so the width is exact, unless
//   biggest_cell_width is below 2^-1007, where halving leaves the normal range.
// Units: m.
// Example: biggest cells 1 m wide give 0x1p-15 m at size level 15; 3 m wide give 0.75 m at size level 2.
// Requires: size_level < kMaxSizeLevels.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU constexpr double ComputeCellWidth(GridFrame frame, SizeLevel size_level) noexcept {
  assert(static_cast<int>(size_level) < kMaxSizeLevels);
  double cell_width = frame.biggest_cell_width;
  for (int halving = 0; halving < static_cast<int>(size_level); ++halving) {
    cell_width *= 0.5;
  }
  return cell_width;
}

// Gives the size level of a block of the grid: the size level of its id.
//
// Example: in the grid of the two-size example below, block 0 is of size level 0 and block 1 of size level 1.
// Requires: block < tables.block_count.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr SizeLevel ToSizeLevel(const GridTables& tables, CellBlockIndex block) noexcept {
  assert(block.value < tables.block_count);
  return ToSizeLevel(CellId{.value = tables.block_ids[block.value].value});
}

// Finds a block of the grid by its id: FindSortedBlock over the block ids, kNoIndex when the grid has no such block.
//
// Example: in the grid of the two-size example below, the id of the size-level-1 block (0,0,0) is at index 1, and
//   that of the size-level-1 block (1,0,0) gives kNoIndex.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellBlockIndex FindCellBlock(const GridTables& tables, CellBlockId block) noexcept {
  return FindSortedBlock(tables.block_ids, tables.block_count, block);
}

// Gives the axis of a face place: the face piece there has the normal +axis, and a value kept on it is positive along
// +axis, from its cell_low into its cell_high.
//
// Method: the face places of a block are 80 X places, then 80 Y places, then 80 Z places.
// Example: face place 79 is an X place, 80 a Y place, 240 + 160 a Z place of block 1.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr Axis ToAxis(FacePlaceIndex face) noexcept {
  const std::uint32_t place_in_block = face.value % static_cast<std::uint32_t>(kFacePlacesPerBlock);
  return static_cast<Axis>(place_in_block / 80U);
}

// The local coordinates of a cell place in a block, 0..3 inside it; -1 or 4 on at most one axis, one place beyond the
// block's low or high side. For this header's own functions only.
struct BlockLocalCoord {
  int x;
  int y;
  int z;
};

// The parts of a face place: its block, and the local coordinates of the cell place on its high side. Along the face
// place's axis the local coordinate is the plane, 0..4; plane 4 is one place beyond the block. For this header's own
// functions only.
struct FacePlaceParts {
  std::uint32_t block;
  BlockLocalCoord local;
};

// Splits a face place into its block and the local coordinates of its high cell place, by the layout of GridTables.
// For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr FacePlaceParts ToFacePlaceParts(FacePlaceIndex face) noexcept {
  const std::uint32_t block = face.value / static_cast<std::uint32_t>(kFacePlacesPerBlock);
  const auto place_on_axis = static_cast<int>((face.value % static_cast<std::uint32_t>(kFacePlacesPerBlock)) % 80U);
  switch (ToAxis(face)) {
    case Axis::kX:
      return {
          .block = block,
          .local = {.x = place_on_axis % 5, .y = (place_on_axis / 5) % 4, .z = place_on_axis / 20},
      };
    case Axis::kY:
      return {
          .block = block,
          .local = {.x = place_on_axis % 4, .y = (place_on_axis / 4) % 5, .z = place_on_axis / 20},
      };
    case Axis::kZ:
      break;
  }
  return {
      .block = block,
      .local = {.x = place_on_axis % 4, .y = (place_on_axis / 4) % 4, .z = place_on_axis / 16},
  };
}

// Computes the cell coordinate, at the block's size level, of local place (x, y, z) of a block: 4 times the block
// coordinate plus the local coordinate on each axis. A local coordinate may be -1 or 4, one place beyond the block.
// For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr CellCoord
ComputeLocalPlaceCoord(const GridTables& tables, std::uint32_t block, BlockLocalCoord local) noexcept {
  // A block id has the bits of the cell id at its block coordinate, so ToCellCoord reads it back.
  const CellCoord block_coord = ToCellCoord(CellId{.value = tables.block_ids[block].value});
  return {
      .x = (kBlockWidthInCells * block_coord.x) + local.x,
      .y = (kBlockWidthInCells * block_coord.y) + local.y,
      .z = (kBlockWidthInCells * block_coord.z) + local.z,
  };
}

// Computes the cell coordinate, at its block's size level, of the cell place on the high side of a face place. Along
// the axis it is one past the block at plane 4. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr CellCoord ComputeHighPlaceCoord(const GridTables& tables, FacePlaceIndex face) noexcept {
  const FacePlaceParts parts = ToFacePlaceParts(face);
  return ComputeLocalPlaceCoord(tables, parts.block, parts.local);
}

// Computes the side, -1, 0 or +1, on which a local coordinate lies beyond a block: -1 below 0, +1 above 3. For this
// header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr int ComputeSideBeyondBlock(int local_coord) noexcept {
  if (local_coord < 0) {
    return -1;
  }
  return local_coord > 3 ? 1 : 0;
}

// Finds local place (x, y, z) of `block`, of which at most one local coordinate is -1 or 4, among the cell places of
// the grid: in the block itself, or at local 3 or 0 of the same-size-level neighbour on that side. kNoIndex when that
// neighbour does not exist. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
FindLocalCellPlace(const GridTables& tables, std::uint32_t block, BlockLocalCoord local) noexcept {
  const int side_x = ComputeSideBeyondBlock(local.x);
  const int side_y = ComputeSideBeyondBlock(local.y);
  const int side_z = ComputeSideBeyondBlock(local.z);
  std::uint32_t holding_block = block;
  if (side_x != 0 || side_y != 0 || side_z != 0) {
    const int entry = ((((side_z + 1) * 3) + (side_y + 1)) * 3) + (side_x + 1);
    holding_block = tables.same_size_level_blocks[(static_cast<std::uint64_t>(block) * 27) + entry];
    if (holding_block == kNoIndex) {
      return kNoIndex;
    }
  }
  // The low two bits of -1 are 3 and of 4 are 0: the local coordinate in the neighbour.
  const std::uint32_t wrapped_x = static_cast<std::uint32_t>(local.x) & 3U;
  const std::uint32_t wrapped_y = static_cast<std::uint32_t>(local.y) & 3U;
  const std::uint32_t wrapped_z = static_cast<std::uint32_t>(local.z) & 3U;
  const std::uint32_t place_in_block = (((wrapped_z * 4U) + wrapped_y) * 4U) + wrapped_x;
  return (holding_block * static_cast<std::uint32_t>(kCellPlacesPerBlock)) + place_in_block;
}

// Finds the active cell that holds local place (x, y, z) of `block`, at most one of them -1 or 4: the place itself
// when it is kActive, else the active cell one size level bigger that covers it. kNoIndex when neither exists.
//
// Method: the place of the block's size level comes from FindLocalCellPlace. When it is kActive it is the answer; when
//   it is kSplit or kAbsent no active cell holds it. When it is kInsideBiggerCell, or no block of that size level holds
//   it, the cover is found through the bigger block: on each axis the block is the low half (block coordinate even)
//   or the high half (odd) of the bigger block, so the covering local coordinate is 2 (b & 1) + floor(c / 2), from
//   -1 to 4, and FindLocalCellPlace finds it in the bigger block or its neighbour. Only a kActive cover counts. For
//   this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
FindActiveCellHolding(const GridTables& tables, std::uint32_t block, BlockLocalCoord local) noexcept {
  const std::uint32_t same_size_place = FindLocalCellPlace(tables, block, local);
  if (same_size_place != kNoIndex) {
    const CellState same_size_state = tables.cell_states[same_size_place];
    if (same_size_state == CellState::kActive) {
      return same_size_place;
    }
    if (same_size_state != CellState::kInsideBiggerCell) {
      return kNoIndex;
    }
  }
  const std::uint32_t bigger_block = tables.bigger_block[block];
  if (bigger_block == kNoIndex) {
    return kNoIndex;
  }
  const CellCoord block_coord = ToCellCoord(CellId{.value = tables.block_ids[block].value});
  // The low bit of a coordinate as unsigned is its parity, for a negative coordinate too.
  const auto half_x = static_cast<int>(static_cast<std::uint32_t>(block_coord.x) & 1U);
  const auto half_y = static_cast<int>(static_cast<std::uint32_t>(block_coord.y) & 1U);
  const auto half_z = static_cast<int>(static_cast<std::uint32_t>(block_coord.z) & 1U);
  const BlockLocalCoord covering_local = {
      .x = (2 * half_x) + ComputeBiggerCellCoord(local.x, 1),
      .y = (2 * half_y) + ComputeBiggerCellCoord(local.y, 1),
      .z = (2 * half_z) + ComputeBiggerCellCoord(local.z, 1),
  };
  const std::uint32_t covering_place = FindLocalCellPlace(tables, bigger_block, covering_local);
  if (covering_place == kNoIndex || tables.cell_states[covering_place] != CellState::kActive) {
    return kNoIndex;
  }
  return covering_place;
}

// Computes the centre of a cell place, in m, world: origin + (c + 1/2) h_l on each axis, for the place's coordinate c
// and its block's size level l.
//
// Units: m, world. Two correctly rounded operations per axis: exact when (c + 1/2) h_l and its sum with the origin are
//   doubles (file header, "Precision"), else within 2^-53 (|(c + 1/2) h_l| + |result|).
// Example: in a frame of origin (-2, 0.5, 8) and biggest cells 1 m wide, the size-level-1 place (3, 1, 0) has its
//   centre at (-0.25, 1.25, 8.25).
// Requires: `cell` is a place of block 0..tables.block_count - 1. Any state: the centre is the place's own.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr Vec3 ComputeCellCenter(const GridTables& tables, CellPlaceIndex cell) noexcept {
  const std::uint32_t block = ToCellBlockIndex(cell).value;
  assert(block < tables.block_count);
  const auto place_in_block = static_cast<int>(cell.value % static_cast<std::uint32_t>(kCellPlacesPerBlock));
  const CellCoord cell_coord = ComputeLocalPlaceCoord(
      tables,
      block,
      {.x = place_in_block % 4, .y = (place_in_block / 4) % 4, .z = place_in_block / 16}
  );
  const double cell_width = ComputeCellWidth(tables.frame, ToSizeLevel(tables, CellBlockIndex{block}));
  return {
      .x = tables.frame.origin.x + ((static_cast<double>(cell_coord.x) + 0.5) * cell_width),
      .y = tables.frame.origin.y + ((static_cast<double>(cell_coord.y) + 0.5) * cell_width),
      .z = tables.frame.origin.z + ((static_cast<double>(cell_coord.z) + 0.5) * cell_width),
  };
}

// Computes the centre of the face piece at a face place, in m, world: on the plane c h_l along its axis and at
// (c + 1/2) h_l across, for the coordinate c of the cell place on its high side and the block's size level l.
//
// Units: m, world. Two correctly rounded operations per axis: exact when m h_l, m = c along the axis and c + 1/2
//   across, and its sum with the origin are doubles (file header, "Precision"), else within 2^-53 (|m h_l| + |result|).
// Example: in the two-size example below, each of the 4 face pieces of the big cell's +X side lies at x = 1 m, with
//   (y, z) one of (0.25, 0.25), (0.25, 0.75), (0.75, 0.25) and (0.75, 0.75) m.
// Requires: `face` is a place of block 0..tables.block_count - 1. Any kind: the position is the place's own.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr Vec3 ComputeFacePosition(const GridTables& tables, FacePlaceIndex face) noexcept {
  const CellBlockIndex block = ToCellBlockIndex(face);
  assert(block.value < tables.block_count);
  const CellCoord high_coord = ComputeHighPlaceCoord(tables, face);
  const Axis axis = ToAxis(face);
  const double cell_width = ComputeCellWidth(tables.frame, ToSizeLevel(tables, block));
  const double across_x = axis == Axis::kX ? 0.0 : 0.5;
  const double across_y = axis == Axis::kY ? 0.0 : 0.5;
  const double across_z = axis == Axis::kZ ? 0.0 : 0.5;
  return {
      .x = tables.frame.origin.x + ((static_cast<double>(high_coord.x) + across_x) * cell_width),
      .y = tables.frame.origin.y + ((static_cast<double>(high_coord.y) + across_y) * cell_width),
      .z = tables.frame.origin.z + ((static_cast<double>(high_coord.z) + across_z) * cell_width),
  };
}

// Computes the area of the face piece at a face place, in m2: h_l^2 for its block's size level l, the face area of its
// smaller cell. The whole face piece; no solid is taken off.
//
// Units: m2. One correctly rounded product: exact when h_l's significand needs at most 26 bits, as for h_0 = 1 m.
// Example: in the two-size example below, each face piece of the big cell's +X side is 0.25 m2; the big cell's
//   other sides are 1 m2.
// Requires: `face` is a place of block 0..tables.block_count - 1.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>
FLUIDITY_CPU_AND_GPU constexpr double ComputeFaceArea(const GridTables& tables, FacePlaceIndex face) noexcept {
  const CellBlockIndex block = ToCellBlockIndex(face);
  assert(block.value < tables.block_count);
  const double cell_width = ComputeCellWidth(tables.frame, ToSizeLevel(tables, block));
  return cell_width * cell_width;
}

// Gives the id of the cell at a cell place: its block's size level and its coordinate, when the place is kActive or
// kSplit; CellId{0} otherwise.
//
// Method: a place inside a bigger cell or absent holds no cell, and its coordinate may lie outside the range (only
//   -2^18, local 0 of a block at block coordinate -2^16 on that axis), where a packed id would name another cell. So
//   only a kActive or kSplit place makes an id. A kSplit place carries no values: a stage that wants a cell carrying
//   values checks that the place is kActive.
// Example: in the two-size example below, place 0 gives the size-level-0 cell {0,0,0}; place 64 + 2, local (2, 0, 0)
//   of block 1, gives the size-level-1 cell {2,0,0}; place 64, inside the big cell, gives CellId{0}.
// Requires: `cell` is a place of block 0..tables.block_count - 1.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellId ToCellId(const GridTables& tables, CellPlaceIndex cell) noexcept {
  const std::uint32_t block = ToCellBlockIndex(cell).value;
  assert(block < tables.block_count);
  const CellState state = tables.cell_states[cell.value];
  if (state != CellState::kActive && state != CellState::kSplit) {
    return CellId{0};
  }
  const auto place_in_block = static_cast<int>(cell.value % static_cast<std::uint32_t>(kCellPlacesPerBlock));
  const CellCoord cell_coord = ComputeLocalPlaceCoord(
      tables,
      block,
      {.x = place_in_block % 4, .y = (place_in_block / 4) % 4, .z = place_in_block / 16}
  );
  return ToCellId(ToSizeLevel(tables, CellBlockIndex{block}), cell_coord);
}

// Gives the id of the face piece at a face place: its axis, its block's size level and the coordinate of the cell place
// on its high side, when the place holds a face piece; FacePieceId{0} for kNotAPiece.
//
// Method: plane 4 of a block and plane 0 of the block above are one place in space, and the coordinate of a high cell
//   place at plane 0 of a block at block coordinate -2^16 is -2^18, outside the range. A face piece is kept at one of
//   them only and has its high coordinate in range, so only a face piece makes an id, and no two places give the
//   same id.
// Example: in the two-size example below, X place 2 of block 1 gives ToFacePieceId(Axis::kX, SizeLevel{1}, {2, 0, 0});
//   X place 1 of block 0, between the big cell and the split cell, gives FacePieceId{0}.
// Requires: `face` is a place of block 0..tables.block_count - 1.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>
FLUIDITY_CPU_AND_GPU constexpr FacePieceId ToFacePieceId(const GridTables& tables, FacePlaceIndex face) noexcept {
  const CellBlockIndex block = ToCellBlockIndex(face);
  assert(block.value < tables.block_count);
  if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
    return FacePieceId{0};
  }
  return ToFacePieceId(ToAxis(face), ToSizeLevel(tables, block), ComputeHighPlaceCoord(tables, face));
}

// Finds the place of the active cell on the low side of the face piece at a face place, its cell_low: the cell place of
// the block's size level beside it, or the active cell one size level bigger that covers that place.
//
// Method: the low cell place lies one plane below the face place along its axis; at plane 0 it is local 3 of the
//   block below. FindActiveCellHolding then takes that place when it is kActive, else the active cell that covers it
//   through the bigger block. It gives kNoIndex for kGridEdgeLow, where no cell lies on the low side, for kNotAPiece,
//   and for kOutsidePiece when the low cell lies in a block the grid piece did not load.
// Example: in the two-size example below, each of the 4 face pieces of the big cell's +X side gives place 0, the big
//   cell, though the face pieces are kept at size level 1.
// Requires: `face` is a place of block 0..tables.block_count - 1; every table of the grid is written.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellPlaceIndex FindCellLow(const GridTables& tables, FacePlaceIndex face) noexcept {
  assert(ToCellBlockIndex(face).value < tables.block_count);
  if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
    return {.value = kNoIndex};
  }
  const FacePlaceParts parts = ToFacePlaceParts(face);
  const Axis axis = ToAxis(face);
  const BlockLocalCoord low_local = {
      .x = parts.local.x - (axis == Axis::kX ? 1 : 0),
      .y = parts.local.y - (axis == Axis::kY ? 1 : 0),
      .z = parts.local.z - (axis == Axis::kZ ? 1 : 0),
  };
  return {.value = FindActiveCellHolding(tables, parts.block, low_local)};
}

// Finds the place of the active cell on the high side of the face piece at a face place, its cell_high: the cell place
// of the block's size level beside it, or the active cell one size level bigger that covers that place.
//
// Method: the high cell place has the face place's local coordinates, the plane along its axis; at plane 4 it is local
//   0 of the block above. FindActiveCellHolding then takes it as FindCellLow does. It gives kNoIndex for
//   kGridEdgeHigh, where no cell lies on the high side, for kNotAPiece, and for kOutsidePiece when the high cell lies
//   in a block the grid piece did not load.
// Example: in the two-size example below, X place 2 + 5 of block 1, the face piece between the big cell and the small
//   cell {2,1,0}, gives the place of that small cell, 64 + 4 + 2.
// Requires: `face` is a place of block 0..tables.block_count - 1; every table of the grid is written.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellPlaceIndex FindCellHigh(const GridTables& tables, FacePlaceIndex face) noexcept {
  assert(ToCellBlockIndex(face).value < tables.block_count);
  if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
    return {.value = kNoIndex};
  }
  const FacePlaceParts parts = ToFacePlaceParts(face);
  return {.value = FindActiveCellHolding(tables, parts.block, parts.local)};
}

// Computes the distance in m, along the axis of the face piece at a face place, between the centres of its cell_low
// and its cell_high: half the width of each, (h_low + h_high) / 2. At the grid's edge, half the width of the one cell.
//
// Method: FindCellLow and FindCellHigh give the two cells, and a side without a cell adds nothing. By kind:
//   kBetweenCells gives h_l when both cells have the face piece's size level and 3/4 h_(l - 1) beside a bigger cell;
//   kGridEdgeLow and kGridEdgeHigh give half the width of their one cell; kNotAPiece gives 0, since FindCellLow and
//   FindCellHigh give kNoIndex there. kOutsidePiece, whose other cell lies in a block the grid piece did not load,
//   gives only half the width of the loaded cell: the tables of the grid piece cannot know whether that cell is of
//   the same size level or one bigger. A caller must not divide by it as a distance between two cells there.
// Units: m. Each half width is exact, and they are added once. Between cells of one size and at the edge the result is
//   exact. Beside a bigger cell it is h_l + h_l / 2 = 3 s 2^(e - 1) for h_l = s 2^e: 3 s needs up to two bits more than
//   s, so the result is correctly rounded, and exact when h_0's significand needs at most 51 bits (h_0 = 1 m, say,
//   but not 0.7 m or 1 - 2^-53 m) (file header, "Precision").
// Example: in the two-size example below, each face piece of the big cell's +X side is 0.5 + 0.25 = 0.75 m from the
//   big cell's centre to the small cell's; a side of a lone cell 1 m wide gives 0.5 m.
// Requires: `face` is a place of block 0..tables.block_count - 1; every table of the grid is written.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>
FLUIDITY_CPU_AND_GPU constexpr double
ComputeCellCenterDistance(const GridTables& tables, FacePlaceIndex face) noexcept {
  const CellPlaceIndex cell_low = FindCellLow(tables, face);
  const CellPlaceIndex cell_high = FindCellHigh(tables, face);
  double center_distance = 0;
  if (cell_low.value != kNoIndex) {
    center_distance += ComputeCellWidth(tables.frame, ToSizeLevel(tables, ToCellBlockIndex(cell_low))) / 2;
  }
  if (cell_high.value != kNoIndex) {
    center_distance += ComputeCellWidth(tables.frame, ToSizeLevel(tables, ToCellBlockIndex(cell_high))) / 2;
  }
  return center_distance;
}

// Finds the place of a cell by its id: the place of its block whose local coordinates are its own, when that place is
// kActive or kSplit; kNoIndex when the grid has no such cell.
//
// Method: FindCellBlock of ToCellBlockId(cell), then ToCellPlaceIndex. A place inside a bigger cell or absent holds no
//   cell, so its id is not found; nor is a value that is not a cell id (bit 63 clear or bits 61..62 set).
//   A kSplit place is found, as ToCellId names it, so a stage that wants a cell carrying values checks kActive.
// Example: in the two-size example below, size level 1 {2,1,0} is at place 64 + 4 + 2; size level 1 {0,0,0}, inside the
//   big cell, gives kNoIndex; so does CellId{0}.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellPlaceIndex FindCellPlace(const GridTables& tables, CellId cell) noexcept {
  constexpr std::uint64_t kAxisBits = std::uint64_t{3} << kIdAxisShift;
  const bool is_cell_id = (cell.value & kIdCellMark) != 0 && (cell.value & kAxisBits) == 0;
  if (!is_cell_id) {
    return {.value = kNoIndex};
  }
  const CellBlockIndex block = FindCellBlock(tables, ToCellBlockId(cell));
  if (block.value == kNoIndex) {
    return {.value = kNoIndex};
  }
  const CellPlaceIndex place = ToCellPlaceIndex(block, ToCellCoord(cell));
  const CellState state = tables.cell_states[place.value];
  if (state != CellState::kActive && state != CellState::kSplit) {
    return {.value = kNoIndex};
  }
  return place;
}

// Finds the face place of a face piece by its id; kNoIndex when the grid has no such face piece.
//
// Method: the id gives the axis, the size level and the coordinate c of the high cell place. When the block that holds
//   that place exists, the face piece can only be at its face place with the local coordinates of c, the plane along
//   the axis. Otherwise, when c is local 0 along the axis, it can only be at plane 4 of the block below. Either place
//   counts only when its kind is not kNotAPiece. A value that is not a face piece id, with bit 63 set or with axis
//   bits 0, gives kNoIndex.
// Example: in the two-size example below, ToFacePieceId(Axis::kX, SizeLevel{1}, {2, 0, 0}) is at X place 2 of block 1,
//   240 + 2; ToFacePieceId(Axis::kX, SizeLevel{0}, {1, 0, 0}), the big cell's +X side as a whole, gives kNoIndex: its
//   place is kNotAPiece, since the side is kept as 4 face pieces at size level 1. FacePieceId{0} gives kNoIndex.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr FacePlaceIndex FindFacePlace(const GridTables& tables, FacePieceId piece) noexcept {
  const std::uint64_t axis_bits = (piece.value >> kIdAxisShift) & 3U;
  const bool is_face_piece_id = (piece.value & kIdCellMark) == 0 && axis_bits != 0;
  if (!is_face_piece_id) {
    return {.value = kNoIndex};
  }
  const auto axis = static_cast<Axis>(axis_bits - 1);
  const SizeLevel size_level = ComputeUnpackedSizeLevel(piece.value);
  const CellCoord high_coord = ComputeUnpackedCoord(piece.value);
  // An id's coordinates are -kMaxCoord..kMaxCoord + 1, which ComputeBiggerCellCoord divides by 4 exactly.
  CellCoord block_coord = {
      .x = ComputeBiggerCellCoord(high_coord.x, 2),
      .y = ComputeBiggerCellCoord(high_coord.y, 2),
      .z = ComputeBiggerCellCoord(high_coord.z, 2),
  };
  // The low two bits of a coordinate are its local coordinate, for a negative coordinate too.
  int local_x = static_cast<int>(static_cast<std::uint32_t>(high_coord.x) & 3U);
  int local_y = static_cast<int>(static_cast<std::uint32_t>(high_coord.y) & 3U);
  int local_z = static_cast<int>(static_cast<std::uint32_t>(high_coord.z) & 3U);
  CellBlockIndex block = FindCellBlock(tables, {.value = ComputePackedId(kIdCellMark, size_level, block_coord)});
  if (block.value == kNoIndex) {
    // Only the block below can keep it, at its plane 4, when c is local 0 along the axis. That block's coordinate is
    // in range: a high coordinate at local 0 is at least -2^18 + 4.
    const bool is_at_local_zero =
        (axis == Axis::kX && local_x == 0) || (axis == Axis::kY && local_y == 0) || (axis == Axis::kZ && local_z == 0);
    if (!is_at_local_zero) {
      return {.value = kNoIndex};
    }
    if (axis == Axis::kX) {
      local_x = 4;
      block_coord.x -= 1;
    } else if (axis == Axis::kY) {
      local_y = 4;
      block_coord.y -= 1;
    } else {
      local_z = 4;
      block_coord.z -= 1;
    }
    block = FindCellBlock(tables, {.value = ComputePackedId(kIdCellMark, size_level, block_coord)});
    if (block.value == kNoIndex) {
      return {.value = kNoIndex};
    }
  }
  int place_on_axis = (((local_z * 4) + local_y) * 4) + local_x;
  if (axis == Axis::kX) {
    place_on_axis = (((local_z * 4) + local_y) * 5) + local_x;
  } else if (axis == Axis::kY) {
    place_on_axis = (((local_z * 5) + local_y) * 4) + local_x;
  }
  const std::uint32_t face_place = (block.value * static_cast<std::uint32_t>(kFacePlacesPerBlock))
      + (80U * static_cast<std::uint32_t>(axis)) + static_cast<std::uint32_t>(place_on_axis);
  if (tables.face_piece_kinds[face_place] == FacePieceKind::kNotAPiece) {
    return {.value = kNoIndex};
  }
  return {.value = face_place};
}

// Added to a coordinate in cells of the frame's smallest size level, which lies in -2^33..2^33 inside the range of
// size-level-0 ids, so the sum is positive and a shift right by s divides it by 2^s, rounded toward minus infinity. It
// is a multiple of 4 x 2^15, so after a shift by up to 15 the low two bits are still those of the coordinate at that
// size level: its local coordinate in its block. For this header's own functions only.
inline constexpr std::uint64_t kSmallestCellCoordBias = std::uint64_t{1} << 34U;

// The coordinate of a position in cells of the frame's smallest size level on each axis, plus kSmallestCellCoordBias.
// For this header's own functions only.
struct BiasedSmallestCellCoord {
  std::uint64_t x;
  std::uint64_t y;
  std::uint64_t z;
  bool is_in_id_range;  // Every axis lies inside the cells a size-level-0 id can name. Otherwise x, y, z mean nothing.
};

// Computes, on each axis, f = floor((position - origin) / w_min), w_min the width of the frame's smallest cells, and
// says whether every f lies inside the cells a size-level-0 id can name. For this header's own functions only.
//
// Method: a size-level-0 cell at c covers f from c 2^(L - 1) to (c + 1) 2^(L - 1) - 1, L = size_level_count, and c
//   lies in -kMaxCoord..kMaxCoord, so the cells of size-level-0 ids are exactly -kMaxCoord 2^(L - 1) <= f <
//   (kMaxCoord + 1) 2^(L - 1); every smaller cell lies inside one of them. Both bounds are integers of at most 2^33,
//   so exact doubles. A NaN or infinite quotient fails the test too. Only an f that passes is converted to an integer,
//   so the conversion never overflows; inside, the size-level-0 coordinate is in range and its packed id names no other
//   cell.
FLUIDITY_CPU_AND_GPU inline BiasedSmallestCellCoord ToBiasedSmallestCellCoord(GridFrame frame, Vec3 position) noexcept {
  assert(frame.size_level_count >= 1 && frame.size_level_count <= kMaxSizeLevels);
  const int smallest_size_level = frame.size_level_count - 1;
  const double smallest_cell_width = ComputeCellWidth(frame, static_cast<SizeLevel>(smallest_size_level));
  // A size-level-0 cell is 2^(L - 1) smallest cells wide.
  const auto biggest_width_in_smallest_cells =
      static_cast<double>(std::uint64_t{1} << static_cast<unsigned>(smallest_size_level));
  const double lowest_coord = -static_cast<double>(kMaxCoord) * biggest_width_in_smallest_cells;
  const double beyond_highest_coord = (static_cast<double>(kMaxCoord) + 1.0) * biggest_width_in_smallest_cells;
  // Gives the biased coordinate of one axis, or 0 outside the range: inside, the biased value is at least 2^33.
  const auto to_biased_coord = [smallest_cell_width, lowest_coord, beyond_highest_coord](double offset) {
    const double floored_quotient = math::Floor(offset / smallest_cell_width);
    const bool is_finite = math::IsFinite(floored_quotient);
    const bool is_inside_bounds = floored_quotient >= lowest_coord && floored_quotient < beyond_highest_coord;
    if (!is_finite || !is_inside_bounds) {
      return std::uint64_t{0};
    }
    // |floored_quotient| <= 2^33 fits a 64-bit integer, and adding 2^34 makes it positive.
    const auto smallest_cell_coord = static_cast<std::int64_t>(floored_quotient);
    return static_cast<std::uint64_t>(smallest_cell_coord + static_cast<std::int64_t>(kSmallestCellCoordBias));
  };
  const std::uint64_t biased_x = to_biased_coord(position.x - frame.origin.x);
  const std::uint64_t biased_y = to_biased_coord(position.y - frame.origin.y);
  const std::uint64_t biased_z = to_biased_coord(position.z - frame.origin.z);
  return {
      .x = biased_x,
      .y = biased_y,
      .z = biased_z,
      .is_in_id_range = biased_x != 0 && biased_y != 0 && biased_z != 0,
  };
}

// Computes the id of the size-level-0 block that holds a biased coordinate in smallest cells, frame of
// size_level_count L: the block of the size-level-0 cell floor(f / 2^(L - 1)) on each axis. For this header's own
// functions only.
// Requires: coord.is_in_id_range, so the size-level-0 cell is in range and its id names no other cell.
FLUIDITY_CPU_AND_GPU constexpr CellBlockId
ToBiggestBlockId(BiasedSmallestCellCoord coord, int size_level_count) noexcept {
  assert(coord.is_in_id_range);
  const auto shift = static_cast<unsigned>(size_level_count - 1);
  const auto bias = static_cast<std::int64_t>(kSmallestCellCoordBias >> shift);
  const CellCoord biggest_cell_coord = {
      .x = static_cast<std::int32_t>(static_cast<std::int64_t>(coord.x >> shift) - bias),
      .y = static_cast<std::int32_t>(static_cast<std::int64_t>(coord.y >> shift) - bias),
      .z = static_cast<std::int32_t>(static_cast<std::int64_t>(coord.z >> shift) - bias),
  };
  return ToCellBlockId(ToCellId(SizeLevel{0}, biggest_cell_coord));
}

// Finds the active cell that holds a biased coordinate in smallest cells, from the size-level-0 block that holds it:
// the place of each size level, kSplit followed down through smaller_blocks, until a kActive place. kNoIndex when a
// place is kAbsent, kInsideBiggerCell or a split cell without its smaller block. For this header's own functions only.
//
// Method: at size level l the cell coordinate is f / 2^(L - 1 - l) rounded down, and its low two bits are its local
//   coordinate in its block. The small cell of a kSplit place lies in the smaller block of the eighth (local >> 1) on
//   each axis: that block's coordinate is 2 b + (local >> 1), for the block coordinate b. The walk makes no id.
//   kInsideBiggerCell marks the places of a smaller block that an active bigger cell covers. The walk enters a smaller
//   block only below a kSplit place, whose 8 small cells all exist, so it meets kInsideBiggerCell only in a broken
//   table; at size level 0 no place holds it. kAbsent is met at size level 0 only. At most size_level_count places are
//   read.
// Requires: biggest_block is a size-level-0 block of the tables; coord.is_in_id_range and lies in that block.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
FindActiveCellBelow(const GridTables& tables, std::uint32_t biggest_block, BiasedSmallestCellCoord coord) noexcept {
  assert(biggest_block < tables.block_count);
  const int size_level_count = tables.frame.size_level_count;
  std::uint32_t block = biggest_block;
  for (int size_level = 0; size_level < size_level_count; ++size_level) {
    const auto shift = static_cast<unsigned>(size_level_count - 1 - size_level);
    const std::uint64_t local_x = (coord.x >> shift) & 3U;
    const std::uint64_t local_y = (coord.y >> shift) & 3U;
    const std::uint64_t local_z = (coord.z >> shift) & 3U;
    const std::uint64_t place_in_block = (((local_z * 4U) + local_y) * 4U) + local_x;
    const auto place = static_cast<std::uint32_t>((std::uint64_t{block} * kCellPlacesPerBlock) + place_in_block);
    const CellState state = tables.cell_states[place];
    if (state == CellState::kActive) {
      return place;
    }
    if (state != CellState::kSplit) {
      return kNoIndex;
    }
    const std::uint64_t cell_eighth = (local_x >> 1U) + (2U * (local_y >> 1U)) + (4U * (local_z >> 1U));
    block = tables.smaller_blocks[(std::uint64_t{block} * 8U) + cell_eighth];
    if (block == kNoIndex) {
      return kNoIndex;
    }
  }
  return kNoIndex;
}

// Finds the size-level-0 block `target` by moves from `near`: up through bigger_block to its size-level-0 block, then
// at most two moves through same_size_level_blocks, each by -1, 0 or +1 toward the target on each axis. kNoIndex when
// `near` is not a block of the tables, a step finds no block, or two moves do not arrive at the target; the caller then
// searches. For this header's own functions, and for its test, which checks which targets the moves arrive at.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
FindBiggestBlockFrom(const GridTables& tables, CellBlockIndex near, CellBlockId target) noexcept {
  if (near.value >= tables.block_count) {
    return kNoIndex;
  }
  // Step 1: up to size level 0, one bigger block per size level. A grid piece may lack a bigger block.
  std::uint32_t block = near.value;
  for (int climb = 0; climb < kMaxSizeLevels && ToSizeLevel(tables, CellBlockIndex{block}) != SizeLevel{0}; ++climb) {
    block = tables.bigger_block[block];
    if (block == kNoIndex) {
      return kNoIndex;
    }
  }
  if (ToSizeLevel(tables, CellBlockIndex{block}) != SizeLevel{0}) {
    return kNoIndex;
  }
  // Step 2: at most two moves toward the target, each to one of the 26 blocks around. Block coordinates lie in
  // -2^16..2^16, so their differences fit an int. The moves arrive when the ids are equal.
  const CellCoord target_coord = ToCellCoord(CellId{.value = target.value});
  const auto to_move = [](std::int32_t difference) {
    if (difference < 0) {
      return -1;
    }
    return difference > 0 ? 1 : 0;
  };
  constexpr int kMaxMoves = 2;
  for (int move = 0; move <= kMaxMoves; ++move) {
    if (tables.block_ids[block].value == target.value) {
      return block;
    }
    if (move == kMaxMoves) {
      break;
    }
    const CellCoord block_coord = ToCellCoord(CellId{.value = tables.block_ids[block].value});
    const int move_x = to_move(target_coord.x - block_coord.x);
    const int move_y = to_move(target_coord.y - block_coord.y);
    const int move_z = to_move(target_coord.z - block_coord.z);
    const int entry = ((((move_z + 1) * 3) + (move_y + 1)) * 3) + (move_x + 1);
    block = tables.same_size_level_blocks[(static_cast<std::uint64_t>(block) * 27) + entry];
    if (block == kNoIndex) {
      return kNoIndex;
    }
  }
  return kNoIndex;
}

// Finds the active cell that holds a biased coordinate in smallest cells: FindCellBlock of its size-level-0 block id,
// then FindActiveCellBelow. The last step of both forms of FindCellAt. For this header's own functions only.
// Requires: coord.is_in_id_range.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
FindCellAtCoord(const GridTables& tables, BiasedSmallestCellCoord coord) noexcept {
  const CellBlockIndex biggest_block = FindCellBlock(tables, ToBiggestBlockId(coord, tables.frame.size_level_count));
  if (biggest_block.value == kNoIndex) {
    return kNoIndex;
  }
  return FindActiveCellBelow(tables, biggest_block.value, coord);
}

// Finds the active cell that contains a position: the place of the kActive cell whose box, low sides included and high
// sides left out, holds it; kNoIndex when no active cell does.
//
// Method: per axis, f = floor((x - origin) / w_min) counts cells of the frame's smallest size level, L - 1. The
//   size-level-0 block of floor(f / 2^(L - 1)) comes from FindCellBlock; the walk then goes down through each kSplit
//   place to the kActive one. A position that is not finite, or whose f lies outside the cells of size-level-0 ids,
//   -kMaxCoord 2^(L - 1) <= f < (kMaxCoord + 1) 2^(L - 1), gives kNoIndex before f becomes an integer. So does a
//   kAbsent place. The walk never meets kInsideBiggerCell: it starts at size level 0 and goes down only below kSplit
//   places.
//   Rounding. f is the one definition of "inside". The window quarter of <sec-nearby-faces> divides the same
//   difference by h_l = 2^(L - 1 - l) w_min, which scales the quotient by a power of two. x - origin and the quotient
//   are each correctly rounded, within 2^-53 of their value (file header, "Precision"). So f never decreases as x
//   grows, and the quotient is within (2^-52 + 2^-106) |q| of the exact q. A position farther than that, about
//   2^-52 |x - origin| m, from every side therefore lies in its exact cell; a nearer one may land on either side. The
//   error scales with |x - origin|, not with |x|. When c w_min and the side origin + c w_min are both doubles, a
//   position on the side gives f = c exactly and lies in the cell above. With origin 0 and h_0 a power of two nothing
//   rounds, so every position lies in its exact cell. One exception holds in every frame: a negative quotient of at
//   most 2^-1075 in size, a position within 2^-1075 w_min below the origin, rounds to -0 and lies in cell 0, not -1.
//   And a difference x - origin beyond the largest double is infinite: kNoIndex. No second test tries to recover the
//   exact side. Where the side is no double, nothing exact exists to compare with. And a second test would be a second
//   definition of "inside", which the window quarters, made from the same quotient, would not share. Both forms of
//   FindCellAt compute the same f, so they agree on every position.
// Units: m, world.
// Example: in the two-size example above, {1.0, 0.25, 0.25} m is in the size-level-1 cell {2,0,0}, on its low side;
//   {nextafter(1.0, 0.0), 0.5, 0.5} m is in the big cell {0,0,0}, place 0; {1.0, 0.5, 0.5} m is in the size-level-1
//   cell {2,1,1}, place 64 + (4 + 1) 4 + 2. {-0.5, 0.5, 0.5} m and x = 524288.25 m, whose size-level-0 coordinate 2^19
//   would pack into the id of {0,0,0}, give kNoIndex. With biggest cells 4 m wide and 1 size level, x = -2^-1074 m
//   gives cell 0: its quotient -2^-1076 rounds to -0.
// Requires: the tables of a built grid, whose frame passed the frame check (GridFrame): size_level_count 1..16, a
//   normal smallest width, every table written. Any position.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-cell-blocks>, <sec-nearby-faces>
FLUIDITY_CPU_AND_GPU inline CellPlaceIndex FindCellAt(const GridTables& tables, Vec3 position) noexcept {
  const BiasedSmallestCellCoord coord = ToBiasedSmallestCellCoord(tables.frame, position);
  if (!coord.is_in_id_range) {
    return {.value = kNoIndex};
  }
  return {.value = FindCellAtCoord(tables, coord)};
}

// Finds the active cell that contains a position, starting from a block near it: the same place as FindCellAt(tables,
// position) for every value of `near`.
//
// Method: the same f and the same guard as FindCellAt. Then from `near`, up through bigger_block to its size-level-0
//   block and at most two moves through same_size_level_blocks toward the size-level-0 block of f. When the moves do
//   not arrive there (`near` not below block_count, kNoIndex included, a move that finds no block, or a target more
//   than two blocks away), FindCellAt's own last step, the binary search and the walk down, gives the answer from the
//   same f. Otherwise the walk down starts from the block the moves found. A size-level-0 block is named by one id, so
//   both ways start from the same block, and the answer is FindCellAt's. `near` only saves the binary search, when the
//   position lies within two size-level-0 blocks of it: the block of the particle's cell at the previous step, say.
// Units: m, world.
// Example: in the two-size example above, {1.0, 0.25, 0.25} m with `near` 0, 1, 2 or kNoIndex gives the place of the
//   size-level-1 cell {2,0,0}, 64 + 2.
// Requires: as FindCellAt. Any position and any `near`: a block of another part of the grid, an index of no block or
//   kNoIndex never changes the answer.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU inline CellPlaceIndex
FindCellAt(const GridTables& tables, Vec3 position, CellBlockIndex near) noexcept {
  const BiasedSmallestCellCoord coord = ToBiasedSmallestCellCoord(tables.frame, position);
  if (!coord.is_in_id_range) {
    return {.value = kNoIndex};
  }
  const CellBlockId biggest_block_id = ToBiggestBlockId(coord, tables.frame.size_level_count);
  const std::uint32_t biggest_block = FindBiggestBlockFrom(tables, near, biggest_block_id);
  if (biggest_block == kNoIndex) {
    return {.value = FindCellAtCoord(tables, coord)};
  }
  return {.value = FindActiveCellBelow(tables, biggest_block, coord)};
}

}  // namespace fluidity::grid
