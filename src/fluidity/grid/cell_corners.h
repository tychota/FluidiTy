// Where the corners of a cell or a face piece are stored, and the cells around an edge of a cell.
//
// Purpose: give every stage the corner places of an active cell or a face piece, the position of a corner place, and
//   the cells that share one edge of a cell, from the tables of one grid version, on the CPU or the GPU.
//
// Idea: the cells around one edge of a cell are the cell and the three places of its size level that share that edge
//   (FindCellsAroundEdge).
//   Where a corner is stored. A corner is a point of the 5 x 5 x 5 lattice of every block whose box holds it, sides
//   included. Among the blocks of the grid whose lattice holds the point, it is stored in a block of the smallest
//   cells (the largest size level value); and among several blocks of that size level, in the one that comes last in
//   the fixed order (the highest z, then y, then x: the block on the high side, as for face plane 4). The size level
//   counts whether or not a cell of that block touches the point. So every corner has one corner place, whichever
//   cell or face piece it is reached from (FindCornerPlaces), and the place gives the point back by
//   ComputeCornerPosition. Under the size step rule a corner of a cell of size level l is stored at size level l + 2
//   at most, so the search starts there. A corner place belongs to one set of tables: on a grid piece's border a
//   corner may be stored at another size level than in the whole grid. There is no corner id, so a corner of a grid
//   piece and a corner of the whole grid are matched by position (ComputeCornerPosition of each place), or through
//   the piece map where the corner's block is loaded, never by the corner place alone.
//   Example 1, a corner on the high side of a block: 1 size level, cells 1 m wide, origin 0, cells {3,0,0} (block 0)
//   and {4,0,0} (block 1). The corner of {3,0,0} on its high x side at {4,0,0} m is lattice point (4, 0, 0) of block 0
//   and (0, 0, 0) of block 1. It is stored in block 1, corner place 125. With {3,0,0} alone it is corner place 4.
//   Example 2, a corner in the middle of a big cell's side, in the two-size example below: block 1, the size-level-1
//   block, covers 0 to 2 m. The point {1, 0.5, 0.5} m is the low corner of the size-level-1 cell {2,1,1}: lattice
//   point (2, 1, 1) of block 1, corner place 125 + (5 + 1) 5 + 2 = 157; it is none of the big cell's 8 corners. The
//   big cell's corner at {1,0,0} m is lattice point (2, 0, 0) of block 1, corner place 127, the low corner of the
//   size-level-1 cell {2,0,0} too. Its corner at {0,0,0} m is corner place 125, though no small cell touches it, and
//   its corner at {1,1,1} m is 125 + (10 + 2) 5 + 2 = 187.
//   A corner position is origin + m h_l, with m the integer coordinate of the corner's lattice point at its block's
//   size level l, |m| <= 2^18: the same two correctly rounded operations, a product then a sum, after integer
//   arithmetic only. The product is exact when the width's significand has at most 34 bits (k of the "Precision"
//   paragraph of grid/grid_tables.h), as m needs at most 19 bits. The same point at size level
//   l + j has the coordinate m 2^j and the width h_l / 2^j, so the product is the same number, rounded the same way:
//   the position does not depend on which size level's lattice it is computed from. A point has one corner place, so
//   a corner reached from any cell or face piece gives one double on each axis, the same bits on the CPU and the GPU.
//   The two-size example used below: 2 size levels, biggest cells 1 m wide, origin 0; the size-level-0 cell {0,0,0}
//   active beside the 8 size-level-1 cells of the size-level-0 cell {1,0,0}. Block 0 is the size-level-0 block (0,0,0)
//   and block 1 the size-level-1 block (0,0,0). The big cell's +X side is four face pieces of 0.25 m2, kept at size
//   level 1 (X places 2, 2 + 5, 2 + 20 and 2 + 25 of block 1). Each has FindCellLow = the big cell, at place 0, and a
//   centre distance of 0.5 + 0.25 = 0.75 m.
//   Which corner places hold a value. A corner place holds a value when FindCornerPlaces returns it for an active
//   cell or a face piece: one place per corner, kept in the block this rule names. The other lattice points of a
//   block hold no value (a block has 125 of them, a cell 8 corners).
//   Readers. All are total: FindCornerPlaces (cell and face forms), ComputeCornerPosition and FindCellsAroundEdge give
//   a defined answer for any index, kNoIndex included (an empty result, or +infinity for a position), when every
//   block index entry of the tables is kNoIndex or below block_count, as in every built grid.
//
// Words:
//   corner          a corner of a cell or of a face piece, as a point in space: corners of cells of two sizes or of
//                   two blocks that meet at one point are one corner
//   lattice         of a block, its 5 x 5 x 5 corner points (x, y, z), each 0..4: the corners of its 64 cell places
//   corner place    the one place where a corner is stored, a point of one block's lattice (CornerPlaceIndex)
//   cells around an edge  an active cell and the three cell places of its size level that share one of its 12 edges
// Owns:    CellCorners, FaceCorners, FindCornerPlaces (two forms), ComputeCornerPosition, CellsAroundEdge,
//          FindCellsAroundEdge; and, for this header's own functions only, LatticePoint, FindCornerPlaceInSizeLevel,
//          kMaxCornerSizeLevelSteps and FindCornerPlaceOfPoint
// Uses:    Axis (base/axis.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), Vec3 (base/vec3.h), GridTables, GridFrame,
//          BlockLocalCoord, ToBlockLocalCoord, ComputeMovedAlongAxis, ComputeIsActiveCellPlace, FindLocalCellPlace,
//          FindActiveCellHolding, ComputeLocalPlaceCoord, ComputeHighPlaceCoord, FindCellBlock, ToSizeLevel, ToAxis,
//          ComputeCellWidth (grid/grid_tables.h), the id types and constants (grid/grid_ids.h), math::kPositiveInfinity
//          (base/scalar_math.h)
// Used by: the viscosity solve, the grid's builders and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-blocks>

#pragma once

#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// The 8 corners of a cell, as FindCornerPlaces gives them. A default-made value has every entry kNoIndex.
struct CellCorners {
  FLUIDITY_CPU_AND_GPU constexpr CellCorners() noexcept {
    for (CornerPlaceIndex& entry : corners) {
      entry = CornerPlaceIndex{kNoIndex};
    }
  }

  // corners[k] is the corner on the high x side where bit 0 of k is set, on the high y side for bit 1 and on the high
  // z side for bit 2; on the low side where the bit is clear. corners[0] is the low corner, corners[7] the high one.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  CornerPlaceIndex corners[8] = {};
};

// The 4 corners of a face piece, as FindCornerPlaces gives them. A default-made value has every entry kNoIndex.
struct FaceCorners {
  FLUIDITY_CPU_AND_GPU constexpr FaceCorners() noexcept {
    for (CornerPlaceIndex& entry : corners) {
      entry = CornerPlaceIndex{kNoIndex};
    }
  }

  // Of the two axes across the face piece, the lower first (y and z for an X face piece, x and z for Y, x and y for Z):
  // corners[k] is on the high side of the first where bit 0 of k is set and of the second where bit 1 is set.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  CornerPlaceIndex corners[4] = {};
};

// A corner as a point of the lattices of one size level: on each axis its coordinate in cells of that size level, the
// low corner of the cell place there. For this header's own functions only.
struct LatticePoint {
  SizeLevel size_level;
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;
};

// Finds the corner place of a lattice point among the blocks of its own size level: of the up to 8 blocks whose
// lattice holds it, the last in the fixed order. kNoIndex when the grid has none of them. For this header's own
// functions, and for cell_corners.test.cc, whose search from the frame's smallest size level checks the bounded one.
//
// Method: on each axis the blocks b with 4 b <= p <= 4 b + 4: floor(p / 4), and floor(p / 4) - 1 as well when p is a
//   multiple of 4. The search tries the higher one first on z, then y, then x, so the first block found is the last in
//   the fixed order. A block coordinate outside -kMaxCoord..kMaxCoord names no block and is not searched, so no
//   packed id names another block.
FLUIDITY_CPU_AND_GPU constexpr CornerPlaceIndex
FindCornerPlaceInSizeLevel(const GridTables& tables, LatticePoint point) noexcept {
  // The coordinate p - 4 floor(p / 4), 0..3, for a negative p too.
  const auto to_local_coord = [](std::int64_t coord) {
    return ((coord % 4) + 4) % 4;
  };
  const std::int64_t local_x = to_local_coord(point.x);
  const std::int64_t local_y = to_local_coord(point.y);
  const std::int64_t local_z = to_local_coord(point.z);
  // At local 0 the block below holds the point too, at its local 4.
  const int low_step_count_x = local_x == 0 ? 1 : 0;
  const int low_step_count_y = local_y == 0 ? 1 : 0;
  const int low_step_count_z = local_z == 0 ? 1 : 0;
  for (int low_step_z = 0; low_step_z <= low_step_count_z; ++low_step_z) {
    for (int low_step_y = 0; low_step_y <= low_step_count_y; ++low_step_y) {
      for (int low_step_x = 0; low_step_x <= low_step_count_x; ++low_step_x) {
        const std::int64_t block_x = ((point.x - local_x) / 4) - low_step_x;
        const std::int64_t block_y = ((point.y - local_y) / 4) - low_step_y;
        const std::int64_t block_z = ((point.z - local_z) / 4) - low_step_z;
        const bool is_x_in_range = block_x >= -kMaxCoord && block_x <= kMaxCoord;
        const bool is_y_in_range = block_y >= -kMaxCoord && block_y <= kMaxCoord;
        const bool is_z_in_range = block_z >= -kMaxCoord && block_z <= kMaxCoord;
        if (!is_x_in_range || !is_y_in_range || !is_z_in_range) {
          continue;
        }
        const CellCoord block_coord = {
            .x = static_cast<std::int32_t>(block_x),
            .y = static_cast<std::int32_t>(block_y),
            .z = static_cast<std::int32_t>(block_z),
        };
        const CellBlockIndex block =
            FindCellBlock(tables, {.value = ComputePackedId(kIdCellMark, point.size_level, block_coord)});
        if (block.value == kNoIndex) {
          continue;
        }
        // The lattice coordinates in the block found, 0..4.
        const auto lattice_x = static_cast<std::uint32_t>(local_x + (std::int64_t{4} * low_step_x));
        const auto lattice_y = static_cast<std::uint32_t>(local_y + (std::int64_t{4} * low_step_y));
        const auto lattice_z = static_cast<std::uint32_t>(local_z + (std::int64_t{4} * low_step_z));
        const std::uint32_t place_in_block = (((lattice_z * 5U) + lattice_y) * 5U) + lattice_x;
        return {.value = (block.value * static_cast<std::uint32_t>(kCornerPlacesPerBlock)) + place_in_block};
      }
    }
  }
  return {.value = kNoIndex};
}

// The most size levels by which a corner's storing block can be smaller than the active cell the corner belongs to,
// under the size step rule (FindCornerPlaceOfPoint). For this header's own functions only.
inline constexpr int kMaxCornerSizeLevelSteps = 2;

// Finds the corner place of a corner of an active cell or a face piece of size level l = point.size_level: the lattice
// point of size level l, looked for at each size level from min(l + 2, the frame's smallest) up to l. For this header's
// own functions only.
//
// Method: at size level m >= l the point's coordinate is p 2^(m - l), an integer, and FindCornerPlaceInSizeLevel looks
//   at the blocks of size level m around it; the first size level with one gives the answer. |p| <= 2^18 and the
//   factor is at most 2^2, so the product fits 21 bits.
//   Why no block smaller than size level l + 2 holds the point. A block of size level m is the box of one place Q of
//   size level m - 2. For m >= l + 3 the point, a multiple of h_l and so of h_(m - 2), lies on that box only as a
//   corner of Q. The block exists only when a cell of size level m exists in Q, so Q is split, and by the siblings rule
//   the small cell of Q at that corner exists, and so on down to an active cell of size level m - 1 or more that has
//   the point as a corner. It touches the cell of size level l, so the size step rule gives m - 1 <= l + 1, against
//   m >= l + 3. Every built grid keeps both rules, and a grid piece holds a subset of the whole grid's blocks, so the
//   bound holds there too. A face piece's corners are those of its smaller cell, of the face piece's size level.
//   Guards for tables that contradict their frame: when l lies beyond the frame's last size level the search starts
//   at l itself, and it never starts beyond size level 15, so it still ends at the block that holds the cell or face
//   piece the point comes from, and no shift reaches 64 bits.
FLUIDITY_CPU_AND_GPU constexpr CornerPlaceIndex
FindCornerPlaceOfPoint(const GridTables& tables, LatticePoint point) noexcept {
  const int point_size_level = static_cast<int>(point.size_level);
  const int frame_smallest_size_level = tables.frame.size_level_count - 1;
  const int bounded_size_level = point_size_level + kMaxCornerSizeLevelSteps;
  const int first_size_level_in_frame =
      bounded_size_level < frame_smallest_size_level ? bounded_size_level : frame_smallest_size_level;
  const int first_size_level_from_point =
      first_size_level_in_frame > point_size_level ? first_size_level_in_frame : point_size_level;
  const int first_size_level =
      first_size_level_from_point < kMaxSizeLevels - 1 ? first_size_level_from_point : kMaxSizeLevels - 1;
  for (int size_level = first_size_level; size_level >= point_size_level; --size_level) {
    const auto scale =
        static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(size_level - point_size_level));
    const LatticePoint scaled_point = {
        .size_level = static_cast<SizeLevel>(size_level),
        .x = point.x * scale,
        .y = point.y * scale,
        .z = point.z * scale,
    };
    const CornerPlaceIndex corner = FindCornerPlaceInSizeLevel(tables, scaled_point);
    if (corner.value != kNoIndex) {
      return corner;
    }
  }
  return {.value = kNoIndex};
}

// Finds where the 8 corners of an active cell are stored, by the rule of the file header ("Where a corner is stored").
//
// Method: corner k of the cell of size level l at c is the lattice point c + (k & 1, (k >> 1) & 1, k >> 2) of size
//   level l. From size level min(l + 2, the frame's smallest) up to l, the up to 8 blocks of each size level whose
//   lattice holds the point are looked for with FindCellBlock; the first size level that has one stores the point, in
//   the last of its blocks in the fixed order. Under the size step rule and the siblings rule no block of a size level
//   beyond l + 2 holds a corner of the cell on its lattice (proof at FindCornerPlaceOfPoint), so the search misses no
//   block the rule could choose. The cell's own block holds every corner at size level l, so the search ends there at
//   the latest. A bigger size level never stores the point: the rule takes the largest size level value, and the
//   cell's own block already holds it.
//   Two cells that share a corner find the same place: each starts at or beyond the size level that stores the point,
//   the point is the same integer at every size level both look at, and neither goes past its own size level, whose
//   block holds the point. And two different points never share a place: within one block the lattice coordinates
//   differ. A corner place belongs to these tables only (file header, "Where a corner is stored").
// Order: corners[k] as CellCorners says: bit 0 the high x side, bit 1 high y, bit 2 high z.
// Example: examples 1 and 2 of the file header. In example 2 the big cell, place 0, gives corners[0] = 125,
//   corners[1] = 127 and corners[7] = 187, all in block 1; the small cells {2, 0..1, 0..1} share 157.
// Requires: Any `cell`: a place that is not kActive (split, inside a bigger cell or absent), an index of no place
//   (kNoIndex, or a block at or beyond block_count) and the tables of no grid (block_count 0) give every entry kNoIndex
//   and read nothing beyond the tables. Only block_ids, cell_states and the frame are read.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellCorners FindCornerPlaces(const GridTables& tables, CellPlaceIndex cell) noexcept {
  CellCorners cell_corners;
  if (!ComputeIsActiveCellPlace(tables, cell)) {
    return cell_corners;
  }
  const CellBlockIndex block = ToCellBlockIndex(cell);
  const CellCoord cell_coord = ComputeLocalPlaceCoord(tables, block.value, ToBlockLocalCoord(cell));
  const SizeLevel size_level = ToSizeLevel(tables, block);
  for (int corner_number = 0; corner_number < 8; ++corner_number) {
    const auto corner_bits = static_cast<std::uint32_t>(corner_number);
    const LatticePoint corner_point = {
        .size_level = size_level,
        .x = std::int64_t{cell_coord.x} + static_cast<std::int64_t>(corner_bits & 1U),
        .y = std::int64_t{cell_coord.y} + static_cast<std::int64_t>((corner_bits >> 1U) & 1U),
        .z = std::int64_t{cell_coord.z} + static_cast<std::int64_t>((corner_bits >> 2U) & 1U),
    };
    cell_corners.corners[corner_number] = FindCornerPlaceOfPoint(tables, corner_point);
  }
  return cell_corners;
}

// Finds where the 4 corners of the face piece at a face place are stored, by the same rule: the corners of its smaller
// cell on that side.
//
// Method: the face piece of size level l whose high cell place is at c lies on the plane c along its axis; its corner
//   k is c + 0 or 1 on each axis across it, by the bits of k, a lattice point of size level l. The search is that of
//   the cell form. The face place's own block holds the point at size level l, plane 4 included.
// Order: corners[k] as FaceCorners says: bit 0 the high side of the lower axis across, bit 1 of the higher.
// Example: in the two-size example of the file header, the face piece at X place 2 of block 1, between the big cell
//   and the small cell {2,0,0}, has the corners 127, 132, 152 and 157: lattice points (2, 0, 0), (2, 1, 0), (2, 0, 1)
//   and (2, 1, 1) of block 1. The big cell's whole -X side, X place 0 of block 0, has 125, 135, 175 and 185.
// Requires: Any `face`: a place of kind kNotAPiece, an index of no place and the tables of no grid give every entry
//   kNoIndex and read nothing beyond the tables. Only block_ids, face_piece_kinds and the frame are read.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>, <sec-face-pieces>
FLUIDITY_CPU_AND_GPU constexpr FaceCorners FindCornerPlaces(const GridTables& tables, FacePlaceIndex face) noexcept {
  FaceCorners face_corners;
  const CellBlockIndex block = ToCellBlockIndex(face);
  if (block.value >= tables.block_count) {
    return face_corners;
  }
  if (tables.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
    return face_corners;
  }
  const CellCoord high_coord = ComputeHighPlaceCoord(tables, face);
  const Axis axis = ToAxis(face);
  const SizeLevel size_level = ToSizeLevel(tables, block);
  // The two axes across the face piece, the lower first: Y and Z across X, X and Z across Y, X and Y across Z.
  const Axis first_across_axis = axis == Axis::kX ? Axis::kY : Axis::kX;
  const Axis second_across_axis = axis == Axis::kZ ? Axis::kY : Axis::kZ;
  for (int corner_number = 0; corner_number < 4; ++corner_number) {
    const auto corner_bits = static_cast<std::uint32_t>(corner_number);
    // The corner is the high cell place's low corner moved by bit 0 across the first axis and bit 1 across the second.
    const BlockLocalCoord corner_steps = ComputeMovedAlongAxis(
        ComputeMovedAlongAxis({.x = 0, .y = 0, .z = 0}, first_across_axis, static_cast<int>(corner_bits & 1U)),
        second_across_axis,
        static_cast<int>((corner_bits >> 1U) & 1U)
    );
    const LatticePoint corner_point = {
        .size_level = size_level,
        .x = std::int64_t{high_coord.x} + corner_steps.x,
        .y = std::int64_t{high_coord.y} + corner_steps.y,
        .z = std::int64_t{high_coord.z} + corner_steps.z,
    };
    face_corners.corners[corner_number] = FindCornerPlaceOfPoint(tables, corner_point);
  }
  return face_corners;
}

// Computes the position of a corner place, in m, world: origin + m h_l on each axis, for the coordinate m of its
// lattice point at its block's size level l, 4 times the block coordinate plus the lattice coordinate 0..4.
//
// Method: integer arithmetic gives m, then one product and one sum per axis,
//   as the paragraph on a corner position in the file header says.
// Units: m, world. Exact when m h_l and its sum with the origin are doubles, as with origin 0 and h_0 = 1 m; else
//   within 2^-53 (|m h_l| + |result|). The same corner place, so the same point, gives the same bits on the CPU and the
//   GPU, whichever cell or face piece it was found from.
// Example: in example 2 of the file header, corner place 125 is at {0, 0, 0} m and 157 at {1, 0.5, 0.5} m; in example
//   1, place 125 is at {4, 0, 0} m.
// Requires: Any `corner`: an index of no place (kNoIndex, or a block at or beyond block_count) and the tables of no
//   grid give +infinity on each axis, with fixed bits on both processors, and read nothing. A caller tells an index of
//   no place by the index itself: in a frame whose corner positions overflow, as a width of 1e304 m allows, a real
//   corner gives +infinity too.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr Vec3 ComputeCornerPosition(const GridTables& tables, CornerPlaceIndex corner) noexcept {
  const CellBlockIndex block = ToCellBlockIndex(corner);
  if (block.value >= tables.block_count) {
    return {.x = math::kPositiveInfinity, .y = math::kPositiveInfinity, .z = math::kPositiveInfinity};
  }
  const auto place_in_block = static_cast<int>(corner.value % static_cast<std::uint32_t>(kCornerPlacesPerBlock));
  // A lattice coordinate of 0..4 is a local coordinate of the block, 4 one place beyond it.
  const CellCoord lattice_coord = ComputeLocalPlaceCoord(
      tables,
      block.value,
      {.x = place_in_block % 5, .y = (place_in_block / 5) % 5, .z = place_in_block / 25}
  );
  const double cell_width = ComputeCellWidth(tables.frame, ToSizeLevel(tables, block));
  return {
      .x = tables.frame.origin.x + (static_cast<double>(lattice_coord.x) * cell_width),
      .y = tables.frame.origin.y + (static_cast<double>(lattice_coord.y) * cell_width),
      .z = tables.frame.origin.z + (static_cast<double>(lattice_coord.z) * cell_width),
  };
}

// The cell places around one edge of a cell, as FindCellsAroundEdge gives them. A default-made value is the empty
// result: every entry kNoIndex and is_same_size false.
struct CellsAroundEdge {
  FLUIDITY_CPU_AND_GPU constexpr CellsAroundEdge() noexcept {
    for (CellPlaceIndex& entry : cells) {
      entry = CellPlaceIndex{kNoIndex};
    }
  }

  // cells[0] is the cell. cells[k], k = 1..3, is the place of the cell's size level moved across first_axis where bit 0
  // of k is set and across second_axis where bit 1 is set, to the side the quarter names: cells[3] lies across both.
  // kNoIndex where that place holds no active cell of the cell's size level.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  CellPlaceIndex cells[4] = {};
  // False when any of the three places holds cells of another size: split, or inside a bigger active cell. An absent
  // place, at the grid's edge, is not another size.
  bool is_same_size = false;
  // The struct holds 3 padding bytes after is_same_size, which no reader writes: results are compared field by field,
  // never by comparing or hashing the bytes of the whole struct.
};

// Finds the cells around one edge of an active cell: the cell and the three places of its size level that share the
// edge, and whether every cell there is of the cell's size.
//
// Method: the edge is parallel to the third axis, on the high side of first_axis where bit 0 of the quarter is set
//   (the low side where it is clear) and on the high side of second_axis where bit 1 is set. Each of the three places
//   comes from FindLocalCellPlace, through the same-size-level neighbours of the cell's block. A kActive place is
//   listed. A kSplit place holds smaller cells and a kInsideBiggerCell place a bigger one: kNoIndex, and is_same_size
//   false. A kAbsent place gives kNoIndex and leaves is_same_size as it is. Where no block of the cell's size level
//   lies, the place is inside a bigger active cell when FindActiveCellHolding finds one through the bigger block, so
//   is_same_size false; otherwise it is absent.
// Order: cells[k] as CellsAroundEdge says: bit 0 of k across first_axis, bit 1 across second_axis.
// Example: in a grid of cells 1 m wide around {0,0,0} with every place active, first_axis Axis::kX, second_axis
//   Axis::kY and quarter 1 give the edge at x = 1 m, y = 0 m, parallel to z: cells[1] {1,0,0}, cells[2] {0,-1,0},
//   cells[3] {1,-1,0}, is_same_size true. In the two-size example of the file header, the big cell, place 0, with the
//   same axes and quarter 1 gives cells[1] kNoIndex, since its +X neighbour is split, and is_same_size false.
// Requires: every block index entry of the tables is kNoIndex or below block_count, as in every built grid. Any other
//   input: a place that is not kActive, an index of no place, the tables of no grid, two equal axes, an axis that is
//   not Axis::kX, Axis::kY or Axis::kZ, and a quarter outside 0..3 give the empty result (every entry kNoIndex,
//   is_same_size false) and read nothing beyond the tables.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellsAroundEdge FindCellsAroundEdge(
    const GridTables& tables,
    CellPlaceIndex cell,
    Axis first_axis,
    Axis second_axis,
    int quarter
) noexcept {
  CellsAroundEdge around;
  const bool is_first_axis_valid = static_cast<int>(first_axis) <= static_cast<int>(Axis::kZ);
  const bool is_second_axis_valid = static_cast<int>(second_axis) <= static_cast<int>(Axis::kZ);
  const bool are_axes_different = first_axis != second_axis;
  const bool is_quarter_valid = quarter >= 0 && quarter <= 3;
  if (!is_first_axis_valid || !is_second_axis_valid || !are_axes_different || !is_quarter_valid) {
    return around;
  }
  if (!ComputeIsActiveCellPlace(tables, cell)) {
    return around;
  }
  const std::uint32_t block = ToCellBlockIndex(cell).value;
  const BlockLocalCoord local = ToBlockLocalCoord(cell);
  const auto quarter_bits = static_cast<std::uint32_t>(quarter);
  const int first_step = (quarter_bits & 1U) != 0 ? 1 : -1;
  const int second_step = (quarter_bits & 2U) != 0 ? 1 : -1;
  around.cells[0] = cell;
  around.is_same_size = true;
  for (int around_number = 1; around_number < 4; ++around_number) {
    const auto around_bits = static_cast<std::uint32_t>(around_number);
    BlockLocalCoord around_local = local;
    if ((around_bits & 1U) != 0) {
      around_local = ComputeMovedAlongAxis(around_local, first_axis, first_step);
    }
    if ((around_bits & 2U) != 0) {
      around_local = ComputeMovedAlongAxis(around_local, second_axis, second_step);
    }
    // Each local coordinate is -1..4, which FindLocalCellPlace and FindActiveCellHolding take on every axis.
    const std::uint32_t same_size_place = FindLocalCellPlace(tables, block, around_local);
    if (same_size_place == kNoIndex) {
      if (FindActiveCellHolding(tables, block, around_local) != kNoIndex) {
        around.is_same_size = false;
      }
      continue;
    }
    const CellState state = tables.cell_states[same_size_place];
    if (state == CellState::kActive) {
      around.cells[around_number] = CellPlaceIndex{same_size_place};
    } else if (state != CellState::kAbsent) {
      around.is_same_size = false;
    }
  }
  return around;
}

}  // namespace fluidity::grid
