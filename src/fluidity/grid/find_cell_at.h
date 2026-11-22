// The active cell of a grid that contains a position.
//
// Purpose: give every stage one function that finds the active cell containing a position of the world, from the
//   tables of one grid version, on the CPU or the GPU.
//
// Idea: a position gives its cell coordinate at the frame's smallest size level by one floor per axis, plus a bias that
//   makes the coordinate positive. The coordinate gives the id of the size-level-0 block that holds it, and a walk
//   down the block tables, from that block to the smaller blocks that cover the position, ends at the active cell
//   (FindCellAt). A caller that has a block near the answer passes it, and the search starts there.
//
// Words:
//   active cell     a cell that exists and is not split; it carries values
// Owns:    FindCellAt (two forms); and, for this header's own functions only, kSmallestCellCoordBias,
//          BiasedSmallestCellCoord, ToBiasedSmallestCellCoord, ToBiggestBlockId, FindActiveCellBelow,
//          FindBiggestBlockFrom and FindCellAtCoord
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), math::Floor, math::IsFinite (base/scalar_math.h), Vec3
//          (base/vec3.h), GridFrame, GridTables, FindCellBlock, ToSizeLevel, ComputeCellWidth, ToCellId
//          (grid/grid_tables.h), the id types and constants (grid/grid_ids.h)
// Used by: the grid's builders, the particle stages and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

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
//   so the conversion never overflows; inside, the size-level-0 coordinate is in range and its packed id names no
//   other cell.
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
//   table; at size level 0 no place holds it. kAbsent is met at size level 0 only. At most size_level_count places
//   are read.
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
//   kAbsent place. The walk never meets kInsideBiggerCell: it starts at size level 0 and goes down only below
//   kSplit places.
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
