// Stable ids of cells, cell blocks and face pieces, and the places of one grid version.
//
// Purpose: name every cell, cell block and face piece by an id that the same cell keeps in every grid version, and
//   number the places of one grid version, so every grid table and every value buffer can say what it holds.
//
// Idea: a cell has a size level, 0 for the biggest cells, and an integer coordinate (x, y, z) counted in cells of its
//   own size. An id packs both into 64 bits, so a cell, the cell block that stores it and each face piece have one
//   number that does not depend on which other cells exist. The id 0 is never valid, so a zero field means "no cell".
//   A place index is different: it is a position in the arrays of one grid version, and the same cell may sit at
//   another place after the cells change. A cell block holds 4 x 4 x 4 cell places, 240 face places (3 axes x
//   5 planes x 4 x 4) and 125 corner places (its 5 x 5 x 5 corners). Cell place (x, y, z) of block b is
//   64 b + (4 z + y) * 4 + x, and corner place (x, y, z) of block b is 125 b + (5 z + y) * 5 + x.
//   Blocks and cells sort in one fixed order: by the low corner, z first, then y, then x, and the bigger cell first
//   when two share a low corner.
//   Example: the cell of size level 2 at (-1, -2, 3) lies in the big cell of size level 1 at (-1, -1, 1): each
//   coordinate is halved and rounded toward minus infinity. Its cell block at size level 2 has the block coordinate
//   (-1, -1, 0), a quarter of each coordinate rounded the same way.
//
// Words:
//   size level    0 for the biggest cells; each size level halves the cell width. At most kMaxSizeLevels
//   cell eighth   which of the 8 small cells of a big cell a cell is: x + 2 y + 4 z of its low or high halves
//   place         a position in the arrays of one grid version (CellPlaceIndex, FacePlaceIndex, CornerPlaceIndex)
//   face piece    the part of a face between two cells or at the grid's edge, kept at the size level of its smaller
//                 cell. Not a grid piece, not a memory piece
//   fixed order   blocks and cells by low corner, z, then y, then x, then the bigger cell first (ComesBefore)
//   grid version  a number that grows by one each time the set of active cells changes
// Owns:    kMaxSizeLevels, kBlockWidthInCells, kCellPlacesPerBlock, kFacePlacesPerBlock, kCornerPlacesPerBlock,
//          kMaxCoord, kNoIndex, CellCoord, SizeLevel, CellId, CellBlockId, FacePieceId, CellBlockIndex,
//          CellPlaceIndex, FacePlaceIndex, CornerPlaceIndex, GridVersion, CellBlockRange, ComputeIsCoordInRange,
//          ToCellId, ToSizeLevel, ToCellCoord, ToCellBlockId, ToFacePieceId, ComputeBigCell, ComputeSmallCell,
//          ComputeCellEighth, ComesBefore, ToCellBlockIndex, ToCellPlaceIndex; and, for this header's own functions
//          only, the bit layout kIdCoordBitCount, kIdCoordMask, kIdSizeLevelShift, kIdSizeLevelMask, kIdAxisShift,
//          kIdCellMark and the helpers ComputePackedId, ComputeUnpackedCoord, ComputeUnpackedSizeLevel,
//          ComputeBiggerCellCoord, ComputeIsEarlierInFixedOrder, of which grid/grid_tables.h, grid/grid_build_steps.h,
//          grid/face_search.h and grid/cell_corners.h also use ComputePackedId, ComputeUnpackedCoord,
//          ComputeUnpackedSizeLevel, kIdCellMark, kIdAxisShift and ComputeBiggerCellCoord
// Uses:    Axis (base/axis.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h)
// Used by: every grid table (grid/grid_tables.h), GridFailure (grid/grid_error.h), and the modules that keep values
//          per cell, per face piece or per cell block
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-sizes>, <sec-cell-blocks>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"

namespace fluidity::grid {

inline constexpr int kMaxSizeLevels = 16;  // Size levels 0..15: the 4 bits an id keeps for the size level.
inline constexpr int kBlockWidthInCells = 4;  // A cell block is 4 x 4 x 4 cell places.
inline constexpr int kCellPlacesPerBlock = 64;  // 4 x 4 x 4.
inline constexpr int kFacePlacesPerBlock = 240;  // 3 axes x (5 planes x 4 x 4): plane 4 is the block's high side.
inline constexpr int kCornerPlacesPerBlock = 125;  // 5 x 5 x 5 corners of the block's cells.
// The largest cell coordinate on each axis, at every size level: -kMaxCoord..kMaxCoord is the range of a cell. It
// leaves room in the 19 bits an id keeps per axis for kMaxCoord + 1, the high side of the last cell.
inline constexpr auto kMaxCoord = static_cast<std::int32_t>((1U << 18U) - 1U);
inline constexpr std::uint32_t kNoIndex = 0xFFFFFFFF;  // No block, cell place, face place or corner place here.

// The integer location of a cell, in cells of the cell's own size level. The cell of size level l at (i, j, k)
// covers i w <= x < (i + 1) w from the grid's origin, where w is the cell width of that size level, and the same
// along y and z.
struct CellCoord {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;
};

// 0 is the biggest cell; each size level halves the width. 0..kMaxSizeLevels - 1.
enum class SizeLevel : std::uint8_t {
};

// The stable identity of a cell: the same size level and coordinate give the same id in every grid version. Ids are
// opaque: other modules build and read them only through the functions below. CellId{0} is never valid.
struct CellId {
  std::uint64_t value;
};

// The stable identity of a cell block: its size level and its block coordinate, a quarter of the coordinate of each
// of its cells rounded toward minus infinity. CellBlockId{0} is never valid.
struct CellBlockId {
  std::uint64_t value;
};

// The stable identity of a face piece: its axis, its size level and the coordinate of the cell place on its high
// side at that size level. FacePieceId{0} is never valid.
struct FacePieceId {
  std::uint64_t value;
};

// A position in the block arrays of one grid version. Never kept across a change of grid version.
struct CellBlockIndex {
  std::uint32_t value;
};

// A cell place of one grid version: block * 64 + (z * 4 + y) * 4 + x. Never kept across a change of grid version.
struct CellPlaceIndex {
  std::uint32_t value;
};

// A face place of one grid version: block * 240 + axis * 80 + the place in that axis's 5 planes of 4 x 4. Never kept
// across a change of grid version.
struct FacePlaceIndex {
  std::uint32_t value;
};

// A corner place of one grid version: block * 125 + (z * 5 + y) * 5 + x. Never kept across a change of grid version.
struct CornerPlaceIndex {
  std::uint32_t value;
};

// Grows by one each time the set of active cells changes. 0 means "bound to no grid"; a built grid has 1 or more.
struct GridVersion {
  std::uint64_t value;
};

// A run of consecutive cell blocks in the fixed order: block indexes first_block..first_block + block_count - 1 of
// one grid version. A plain struct, so shared code of any module takes it by value on either processor.
struct CellBlockRange {
  std::uint32_t first_block;
  std::uint32_t block_count;
};

// The bit layout of an id, from bit 63 down. Other modules never read or write these bits: they make and read ids with
// ToCellId, ToCellCoord, ToSizeLevel, ToCellBlockId and ToFacePieceId. The constants and helpers from here to
// ComesBefore are for this header's functions. Four other grid headers also use some of them: grid/grid_build_steps.h
// tests bits 61..63 of an input id with kIdCellMark and kIdAxisShift and halves raw coordinates that may lie one past
// the range with ComputeBiggerCellCoord; grid/grid_tables.h, grid/face_search.h and grid/cell_corners.h pack and read
// the id of a block they look for with ComputePackedId, ComputeUnpackedCoord and ComputeUnpackedSizeLevel.
//   CellId, CellBlockId:  bit 63 set,  61..62 zero,      57..60 size level,  38..56 z,  19..37 y,  0..18 x
//   FacePieceId:          bit 63 zero, 61..62 axis + 1,  57..60 size level,  38..56 z,  19..37 y,  0..18 x
// A coordinate c is kept as c + kMaxCoord in 19 bits: -kMaxCoord..kMaxCoord are 0..2^19 - 2, and the last value,
// 2^19 - 1, is kMaxCoord + 1. Every valid CellId has bit 63 set and every valid FacePieceId an axis of 1..3, so 0 is
// never valid.
inline constexpr unsigned kIdCoordBitCount = 19;
inline constexpr std::uint64_t kIdCoordMask = (std::uint64_t{1} << kIdCoordBitCount) - 1;
inline constexpr unsigned kIdSizeLevelShift = 57;
inline constexpr std::uint64_t kIdSizeLevelMask = 0xF;
inline constexpr unsigned kIdAxisShift = 61;
inline constexpr std::uint64_t kIdCellMark = std::uint64_t{1} << 63U;

// Says whether every axis of a cell coordinate lies in -kMaxCoord..kMaxCoord. Code that looks at a place beside a
// cell calls it before it makes an id of that place.
//
// Example: {kMaxCoord, -kMaxCoord, 0} is in range; {kMaxCoord + 1, 0, 0} and {0, 0, -kMaxCoord - 1} are not.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsCoordInRange(CellCoord cell_coord) noexcept {
  const bool is_x_in_range = cell_coord.x >= -kMaxCoord && cell_coord.x <= kMaxCoord;
  const bool is_y_in_range = cell_coord.y >= -kMaxCoord && cell_coord.y <= kMaxCoord;
  const bool is_z_in_range = cell_coord.z >= -kMaxCoord && cell_coord.z <= kMaxCoord;
  return is_x_in_range && is_y_in_range && is_z_in_range;
}

// Packs a mark, a size level and a coordinate into the bits of an id. Each coordinate keeps its low 19 bits after
// kMaxCoord is added; the unsigned addition wraps instead of overflowing for any coordinate.
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t
ComputePackedId(std::uint64_t mark, SizeLevel size_level, CellCoord cell_coord) noexcept {
  const auto offset = static_cast<std::uint32_t>(kMaxCoord);
  const std::uint64_t kept_x = (static_cast<std::uint32_t>(cell_coord.x) + offset) & kIdCoordMask;
  const std::uint64_t kept_y = (static_cast<std::uint32_t>(cell_coord.y) + offset) & kIdCoordMask;
  const std::uint64_t kept_z = (static_cast<std::uint32_t>(cell_coord.z) + offset) & kIdCoordMask;
  const auto kept_size_level = static_cast<std::uint64_t>(size_level) & kIdSizeLevelMask;
  return mark | (kept_size_level << kIdSizeLevelShift) | (kept_z << (2 * kIdCoordBitCount))
      | (kept_y << kIdCoordBitCount) | kept_x;
}

// Reads the coordinate back from the bits of an id: -kMaxCoord..kMaxCoord + 1 on each axis.
FLUIDITY_CPU_AND_GPU constexpr CellCoord ComputeUnpackedCoord(std::uint64_t id_value) noexcept {
  const auto kept_x = static_cast<std::int32_t>(id_value & kIdCoordMask);
  const auto kept_y = static_cast<std::int32_t>((id_value >> kIdCoordBitCount) & kIdCoordMask);
  const auto kept_z = static_cast<std::int32_t>((id_value >> (2 * kIdCoordBitCount)) & kIdCoordMask);
  return {
      .x = kept_x - kMaxCoord,
      .y = kept_y - kMaxCoord,
      .z = kept_z - kMaxCoord,
  };
}

// Reads the size level back from the bits of an id.
FLUIDITY_CPU_AND_GPU constexpr SizeLevel ComputeUnpackedSizeLevel(std::uint64_t id_value) noexcept {
  return static_cast<SizeLevel>((id_value >> kIdSizeLevelShift) & kIdSizeLevelMask);
}

// Computes the coordinate, size_level_steps size levels bigger, of the cell that holds a cell coordinate: the
// coordinate divided by 2^size_level_steps, rounded toward minus infinity.
//
// Method: adds 2^18 first, so the shifted value is never negative, then removes 2^18 shifted the same way. Exact for
//   any coordinate of -2^18..kMaxCoord + 1, which holds every coordinate an id gives back.
// Example: -1 one size level bigger is -1; -262143 one size level bigger is -131072; 262143 two bigger is 65535.
// Requires: size_level_steps is 1 or 2; cell_coord >= -2^18.
//
// clang-tidy: a coordinate and a count of size levels; the swappable-parameters check flags the two integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU constexpr std::int32_t
ComputeBiggerCellCoord(std::int32_t cell_coord, unsigned size_level_steps) noexcept {
  constexpr std::uint32_t kBias = std::uint32_t{1} << 18U;
  const std::uint32_t biased_coord = static_cast<std::uint32_t>(cell_coord) + kBias;
  const std::uint32_t biased_bigger_coord = biased_coord >> size_level_steps;
  return static_cast<std::int32_t>(biased_bigger_coord) - static_cast<std::int32_t>(kBias >> size_level_steps);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Computes the id of a cell from its size level and coordinate.
//
// Method: keeps only 19 bits of each axis after kMaxCoord is added. kMaxCoord + 1, the high side of the last cell, is
//   the one coordinate outside -kMaxCoord..kMaxCoord that keeps its meaning (a grid's checks report such a cell as
//   kCoordOutOfRange). Any other coordinate outside the range gives the id of another cell, so a caller checks its
//   coordinates with ComputeIsCoordInRange first.
// Example: ToCellId(SizeLevel{5}, {-kMaxCoord, 0, kMaxCoord + 1}) has bit 63 set, 5 in bits 57..60, 2^19 - 1 in
//   bits 38..56, kMaxCoord in bits 19..37 and 0 in bits 0..18.
// Requires: size_level < kMaxSizeLevels.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU constexpr CellId ToCellId(SizeLevel size_level, CellCoord cell_coord) noexcept {
  assert(static_cast<int>(size_level) < kMaxSizeLevels);
  return {.value = ComputePackedId(kIdCellMark, size_level, cell_coord)};
}

// Gives the size level of a cell.
//
// Example: ToSizeLevel(ToCellId(SizeLevel{15}, {0, 0, 0})) = SizeLevel{15}.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU constexpr SizeLevel ToSizeLevel(CellId cell) noexcept {
  return ComputeUnpackedSizeLevel(cell.value);
}

// Gives the coordinate of a cell, in cells of its own size level: -kMaxCoord..kMaxCoord + 1 on each axis.
//
// Example: ToCellCoord(ToCellId(SizeLevel{4}, {kMaxCoord + 1, -kMaxCoord, 7})) = {kMaxCoord + 1, -kMaxCoord, 7}.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU constexpr CellCoord ToCellCoord(CellId cell) noexcept {
  return ComputeUnpackedCoord(cell.value);
}

// Gives the id of the cell block that stores a cell: the same size level, and on each axis a quarter of the cell's
// coordinate rounded toward minus infinity.
//
// Example: the cells of size level 1 at -4..-1 on every axis share one block, block coordinate (-1, -1, -1); the cells
//   at 0..3 share block (0, 0, 0).
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellBlockId ToCellBlockId(CellId cell) noexcept {
  const CellCoord cell_coord = ToCellCoord(cell);
  const CellCoord block_coord = {
      .x = ComputeBiggerCellCoord(cell_coord.x, 2),
      .y = ComputeBiggerCellCoord(cell_coord.y, 2),
      .z = ComputeBiggerCellCoord(cell_coord.z, 2),
  };
  return {.value = ComputePackedId(kIdCellMark, ToSizeLevel(cell), block_coord)};
}

// Computes the id of a face piece from its axis, its size level and the coordinate of the cell place on its high side
// at that size level. The face piece is the low side of that cell place.
//
// Method: packs as ToCellId, with axis + 1 in bits 61..62 and bit 63 clear. Along the axis, cell_high_coord may be
//   kMaxCoord + 1: the high side of the last cell.
// Example: ToFacePieceId(Axis::kX, SizeLevel{3}, {kMaxCoord + 1, -kMaxCoord, 0}) is the +X side of the cell at
//   {kMaxCoord, -kMaxCoord, 0}. It is not 0 and differs from the id with Axis::kY.
// Requires: axis is Axis::kX, Axis::kY or Axis::kZ; size_level < kMaxSizeLevels. A release build does not check
//   them: an axis value of 3 would give axis + 1 = 4, which sets bit 63 and clears bits 61..62, a value with the mark
//   of a cell and no valid face piece id.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>
FLUIDITY_CPU_AND_GPU constexpr FacePieceId
ToFacePieceId(Axis axis, SizeLevel size_level, CellCoord cell_high_coord) noexcept {
  assert(axis == Axis::kX || axis == Axis::kY || axis == Axis::kZ);
  assert(static_cast<int>(size_level) < kMaxSizeLevels);
  const std::uint64_t axis_mark = (static_cast<std::uint64_t>(axis) + 1) << kIdAxisShift;
  return {.value = ComputePackedId(axis_mark, size_level, cell_high_coord)};
}

// Computes the big cell that holds a cell: one size level bigger, each coordinate halved and rounded toward
// minus infinity.
//
// Example: the cell of size level 2 at (-1, -2, 3) gives the cell of size level 1 at (-1, -1, 1).
// Requires: the size level of small_cell is 1 or more.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU constexpr CellId ComputeBigCell(CellId small_cell) noexcept {
  const auto small_size_level = static_cast<int>(ToSizeLevel(small_cell));
  assert(small_size_level >= 1);
  const CellCoord small_coord = ToCellCoord(small_cell);
  const CellCoord big_coord = {
      .x = ComputeBiggerCellCoord(small_coord.x, 1),
      .y = ComputeBiggerCellCoord(small_coord.y, 1),
      .z = ComputeBiggerCellCoord(small_coord.z, 1),
  };
  return ToCellId(static_cast<SizeLevel>(small_size_level - 1), big_coord);
}

// Computes one of the 8 small cells of a big cell: one size level smaller, at 2 c + 0 or 1 on each axis.
//
// Method: cell_eighth = x + 2 y + 4 z, where x, y and z are 0 for the low half of the big cell and 1 for the high
//   half. When a doubled coordinate leaves -kMaxCoord..kMaxCoord + 1, the result is the id of another cell, as for
//   ToCellId; a caller checks the coordinate first.
// Example: the cell of size level 6 at (-3, 0, 5) and cell_eighth 6 give the cell of size level 7 at (-6, 1, 11).
// Requires: the size level of big_cell is below kMaxSizeLevels - 1; cell_eighth is 0..7.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU constexpr CellId ComputeSmallCell(CellId big_cell, int cell_eighth) noexcept {
  const auto big_size_level = static_cast<int>(ToSizeLevel(big_cell));
  assert(big_size_level < kMaxSizeLevels - 1);
  assert(cell_eighth >= 0 && cell_eighth < 8);
  const auto eighth_bits = static_cast<std::uint32_t>(cell_eighth);
  const auto high_half_x = static_cast<std::int32_t>(eighth_bits & 1U);
  const auto high_half_y = static_cast<std::int32_t>((eighth_bits >> 1U) & 1U);
  const auto high_half_z = static_cast<std::int32_t>((eighth_bits >> 2U) & 1U);
  const CellCoord big_coord = ToCellCoord(big_cell);
  const CellCoord small_coord = {
      .x = 2 * big_coord.x + high_half_x,
      .y = 2 * big_coord.y + high_half_y,
      .z = 2 * big_coord.z + high_half_z,
  };
  return ToCellId(static_cast<SizeLevel>(big_size_level + 1), small_coord);
}

// Computes which of the 8 small cells of its big cell a cell is: x + 2 y + 4 z, 0..7, where each of x, y and z is 1
// when that coordinate is odd.
//
// Method: an odd coordinate lies in the high half of the big cell, also for a negative one: -1 is the high half of
//   big cell -1.
// Example: the cell at (-6, 1, 11) is cell eighth 6.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU constexpr int ComputeCellEighth(CellId small_cell) noexcept {
  const CellCoord small_coord = ToCellCoord(small_cell);
  // The low bit of a coordinate as unsigned is its parity, for a negative coordinate too.
  const std::uint32_t is_x_odd = static_cast<std::uint32_t>(small_coord.x) & 1U;
  const std::uint32_t is_y_odd = static_cast<std::uint32_t>(small_coord.y) & 1U;
  const std::uint32_t is_z_odd = static_cast<std::uint32_t>(small_coord.z) & 1U;
  return static_cast<int>(is_x_odd + 2 * is_y_odd + 4 * is_z_odd);
}

// Says whether the id of first comes before the id of second in the fixed order, for two cell ids or two block ids.
//
// Method: compares coord * 2^(15 - size level) axis by axis. For a cell it is the low corner in widths of a
//   size-level-15 cell. For a block the low corner is 4 times that, a factor common to both ids, which changes neither
//   the order nor equality. Its size is at most 2^18 * 2^15 = 2^33, so 64-bit integers hold it exactly.
//
// clang-tidy: first and second are the two ids compared, in the order of the predicate's name; the
// swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsEarlierInFixedOrder(std::uint64_t first_value, std::uint64_t second_value) noexcept {
  constexpr auto kSmallestSizeLevel = static_cast<unsigned>(kMaxSizeLevels - 1);
  const auto first_size_level = static_cast<unsigned>(ComputeUnpackedSizeLevel(first_value));
  const auto second_size_level = static_cast<unsigned>(ComputeUnpackedSizeLevel(second_value));
  const auto first_scale = static_cast<std::int64_t>(std::uint64_t{1} << (kSmallestSizeLevel - first_size_level));
  const auto second_scale = static_cast<std::int64_t>(std::uint64_t{1} << (kSmallestSizeLevel - second_size_level));
  const CellCoord first_coord = ComputeUnpackedCoord(first_value);
  const CellCoord second_coord = ComputeUnpackedCoord(second_value);
  const std::int64_t first_z = first_coord.z * first_scale;
  const std::int64_t second_z = second_coord.z * second_scale;
  if (first_z != second_z) {
    return first_z < second_z;
  }
  const std::int64_t first_y = first_coord.y * first_scale;
  const std::int64_t second_y = second_coord.y * second_scale;
  if (first_y != second_y) {
    return first_y < second_y;
  }
  const std::int64_t first_x = first_coord.x * first_scale;
  const std::int64_t second_x = second_coord.x * second_scale;
  if (first_x != second_x) {
    return first_x < second_x;
  }
  // The same low corner: the bigger one, of the smaller size level, comes first.
  return first_size_level < second_size_level;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Says whether one cell block comes before another in the fixed order, the order of the grid's block arrays.
//
// Method: a block's low corner is 4 times its block coordinate on every axis, the same factor for every block, so the
//   block coordinates compare as the low corners do.
// Order: by the low corner, in widths of a size-level-15 cell, by z, then y, then x; at one low corner the bigger
//   block, of the smaller size level, comes first. A block never comes before itself.
// Example: the block of size level 1 whose low corner is the cell (-4, 4, 0) comes before the block of size level 2
//   whose low corner is the cell (-8, 8, 0), the same point.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr bool ComesBefore(CellBlockId first, CellBlockId second) noexcept {
  return ComputeIsEarlierInFixedOrder(first.value, second.value);
}

// Says whether one cell comes before another in the fixed order, the order of blocks applied to cells.
//
// Order: by the low corner, in widths of a size-level-15 cell, by z, then y, then x; at one low corner the bigger
//   cell, of the smaller size level, comes first. A cell never comes before itself.
// Example: the cell of size level 0 at (-kMaxCoord, 0, 0) comes before the cell of size level 15 at the same
//   coordinate, which comes before the cell of size level 0 at (kMaxCoord, 0, 0).
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr bool ComesBefore(CellId first, CellId second) noexcept {
  return ComputeIsEarlierInFixedOrder(first.value, second.value);
}

// Gives the block of a cell place: cell.value / 64.
//
// Example: cell place 130 is in block 2.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellBlockIndex ToCellBlockIndex(CellPlaceIndex cell) noexcept {
  return {.value = cell.value / static_cast<std::uint32_t>(kCellPlacesPerBlock)};
}

// Gives the block of a face place: face.value / 240.
//
// Example: face place 479 is in block 1.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellBlockIndex ToCellBlockIndex(FacePlaceIndex face) noexcept {
  return {.value = face.value / static_cast<std::uint32_t>(kFacePlacesPerBlock)};
}

// Gives the block of a corner place: corner.value / 125.
//
// Example: corner place 249 is in block 1, 250 in block 2.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellBlockIndex ToCellBlockIndex(CornerPlaceIndex corner) noexcept {
  return {.value = corner.value / static_cast<std::uint32_t>(kCornerPlacesPerBlock)};
}

// Gives the cell place of a cell in its block, given the block's index: 64 block + (4 z + y) * 4 + x, where x, y and z
// are the cell's local coordinates in the block, its coordinate minus 4 floor(c / 4) on each axis. The low two bits of
// a coordinate are that local coordinate, also for a negative coordinate. The block index is used as given, so the
// result is the cell's place only when `block` is the index of the block that holds the cell.
//
// Example: the cell {5, -1, 2} of block 3 has local coordinates (1, 3, 2): place 64 x 3 + (4 x 2 + 3) x 4 + 1 = 237.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellPlaceIndex ToCellPlaceIndex(CellBlockIndex block, CellCoord cell_coord) noexcept {
  const std::uint32_t local_x = static_cast<std::uint32_t>(cell_coord.x) & 3U;
  const std::uint32_t local_y = static_cast<std::uint32_t>(cell_coord.y) & 3U;
  const std::uint32_t local_z = static_cast<std::uint32_t>(cell_coord.z) & 3U;
  const std::uint32_t place_in_block = (((local_z * 4U) + local_y) * 4U) + local_x;
  return {.value = (block.value * static_cast<std::uint32_t>(kCellPlacesPerBlock)) + place_in_block};
}

}  // namespace fluidity::grid
