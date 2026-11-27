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
// Owns:    NewCellNeighbors, FindNewCellNeighbors; and, for this header's own functions only, SmallestCellPoint,
//          kPlaceDirectionCount, kPartCountPerPlace, PlaceDirection, PartHalves, ToPlaceDirection, ToPartHalves,
//          ComputeIsPartTouchingBox, FindSizeLevelAtPoint, ComputeTouchesBiggerCell, ComputeTouchesMuchSmallerCell
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), GridTables, ToSizeLevel (grid/grid_tables.h),
//          BiasedSmallestCellCoord, kSmallestCellCoordBias, FindCellAtCoord (grid/find_cell_at.h), the id types and
//          constants (grid/grid_ids.h)
// Used by: the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-size-rule>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// What the committed cells around a cell that is not in the grid say about it.
struct NewCellNeighbors {
  // A committed cell two or more size levels smaller touches it: the new cell must be split.
  bool touches_much_smaller_cell;
  // A committed cell one or more size levels bigger touches it. Read with the first fact: a box that is split has small
  // cells one size level smaller, which would sit beside that cell two or more size levels bigger, and a box left whole
  // beside a cell only one size level bigger obeys the rule.
  bool touches_bigger_cell;
};

// A point in cells of the frame's smallest size level, signed, with no bias: the integer position of one corner of a
// cell or of part of a cell. For this header's own functions only.
struct SmallestCellPoint {
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;
};

// The 27 places of a 3 x 3 x 3 block of places around a box, the box's own place included, and the 8 half-width parts
// of one place. For this header's own functions only.
inline constexpr int kPlaceDirectionCount = 27;
inline constexpr int kPartCountPerPlace = 8;

// The step of a place from the box, -1, 0 or +1 on each axis. For this header's own functions only.
struct PlaceDirection {
  int x;
  int y;
  int z;
};

// Which half of a place a part is on each axis: 0 for the low half and 1 for the high half. For this header's own
// functions only.
struct PartHalves {
  int x;
  int y;
  int z;
};

// Gives the step that place `place_index` of 0..26 takes from the box: place_index = (x + 1) + 3 (y + 1) + 9 (z + 1)
// for a step (x, y, z). Place 13 is the box itself. For this header's own functions only.
FLUIDITY_CPU_AND_GPU constexpr PlaceDirection ToPlaceDirection(int place_index) noexcept {
  return {.x = (place_index % 3) - 1, .y = ((place_index / 3) % 3) - 1, .z = (place_index / 9) - 1};
}

// Gives the halves that part `part_index` of 0..7 has: part_index = x + 2 y + 4 z. For this header's own functions
// only.
FLUIDITY_CPU_AND_GPU constexpr PartHalves ToPartHalves(int part_index) noexcept {
  return {.x = part_index % 2, .y = (part_index / 2) % 2, .z = part_index / 4};
}

// Says whether the part of a place with the given halves touches the box. For this header's own functions only.
//
// Method: along one axis, with the place's step d and the half h (0 low, 1 high), the half touches the box when d (2 h
//   - 1) <= 0: the high half of the place below the box (d = -1, h = 1), the low half of the place above it (d = 1,
//   h = 0), and both halves of a place level with it (d = 0). The part touches when its half does on every axis, so a
//   face place has 4 parts that touch, an edge place 2 and a corner place 1.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeIsPartTouchingBox(PlaceDirection direction, PartHalves halves) noexcept {
  const bool is_x_touching = direction.x * ((2 * halves.x) - 1) <= 0;
  const bool is_y_touching = direction.y * ((2 * halves.y) - 1) <= 0;
  const bool is_z_touching = direction.z * ((2 * halves.z) - 1) <= 0;
  return is_x_touching && is_y_touching && is_z_touching;
}

// Finds the size level of the committed cell that covers the cell of the smallest size level at `point`: -1 when the
// point lies outside the cells size-level-0 ids can name, or in space with no cell. For this header's own functions
// only.
//
// Method: the biased coordinate is the point plus kSmallestCellCoordBias, the form FindCellAtCoord reads. A point
//   outside -kMaxCoord 2^(L - 1) <= f < (kMaxCoord + 1) 2^(L - 1), L the number of size levels, counts as empty, as
//   FindCellAt does.
FLUIDITY_CPU_AND_GPU constexpr int FindSizeLevelAtPoint(const GridTables& committed, SmallestCellPoint point) noexcept {
  const int size_level_count = committed.frame.size_level_count;
  const auto biggest_width = static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(size_level_count - 1));
  const std::int64_t lowest_coord = -static_cast<std::int64_t>(kMaxCoord) * biggest_width;
  const std::int64_t beyond_highest_coord = (static_cast<std::int64_t>(kMaxCoord) + 1) * biggest_width;
  const bool is_x_in_range = point.x >= lowest_coord && point.x < beyond_highest_coord;
  const bool is_y_in_range = point.y >= lowest_coord && point.y < beyond_highest_coord;
  const bool is_z_in_range = point.z >= lowest_coord && point.z < beyond_highest_coord;
  if (!is_x_in_range || !is_y_in_range || !is_z_in_range) {
    return -1;
  }
  const auto bias = static_cast<std::int64_t>(kSmallestCellCoordBias);
  const BiasedSmallestCellCoord coord = {
      .x = static_cast<std::uint64_t>(point.x + bias),
      .y = static_cast<std::uint64_t>(point.y + bias),
      .z = static_cast<std::uint64_t>(point.z + bias),
      .is_in_id_range = true,
  };
  const std::uint32_t cell_place = FindCellAtCoord(committed, coord);
  if (cell_place == kNoIndex) {
    return -1;
  }
  return static_cast<int>(ToSizeLevel(committed, ToCellBlockIndex(CellPlaceIndex{.value = cell_place})));
}

// Says whether a committed cell one or more size levels bigger touches the box, from the 26 places around it.
// For this header's own functions only. `box_low` is the box's low corner and `box_width` its width, both in smallest
// cells.
//
// Method: a touching cell that is bigger than the box covers the whole place it meets, so the low corner of the place
//   tells. At most 26 look-ups.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeTouchesBiggerCell(
    const GridTables& committed,
    int size_level,
    SmallestCellPoint box_low,
    std::int64_t box_width
) noexcept {
  if (size_level == 0) {
    return false;
  }
  for (int place_index = 0; place_index < kPlaceDirectionCount; ++place_index) {
    const PlaceDirection direction = ToPlaceDirection(place_index);
    if (direction.x == 0 && direction.y == 0 && direction.z == 0) {
      continue;
    }
    const SmallestCellPoint place_low = {
        .x = box_low.x + (direction.x * box_width),
        .y = box_low.y + (direction.y * box_width),
        .z = box_low.z + (direction.z * box_width),
    };
    const int found_size_level = FindSizeLevelAtPoint(committed, place_low);
    if (found_size_level >= 0 && found_size_level <= size_level - 1) {
      return true;
    }
  }
  return false;
}

// Says whether a committed cell two or more size levels smaller touches the box, from the 56 half-width parts of the
// 26 places around it that touch it. For this header's own functions only. Arguments as ComputeTouchesBiggerCell.
//
// Method: a part is one cell of size level `size_level` + 1 wide. A cell found in it with a size level of
//   `size_level` + 2 or more is smaller than the part, so the part is split and every cell of it has
//   size level `size_level` + 2 or more. One of them touches the box, because the part touches the box and its cells
//   cover it. At most 56 look-ups, none when `size_level` + 2 is past the last size level.
FLUIDITY_CPU_AND_GPU constexpr bool ComputeTouchesMuchSmallerCell(
    const GridTables& committed,
    int size_level,
    SmallestCellPoint box_low,
    std::int64_t box_width
) noexcept {
  if (size_level + 2 > committed.frame.size_level_count - 1) {
    return false;
  }
  const std::int64_t part_width = box_width / 2;
  for (int place_index = 0; place_index < kPlaceDirectionCount; ++place_index) {
    const PlaceDirection direction = ToPlaceDirection(place_index);
    if (direction.x == 0 && direction.y == 0 && direction.z == 0) {
      continue;
    }
    for (int part_index = 0; part_index < kPartCountPerPlace; ++part_index) {
      const PartHalves halves = ToPartHalves(part_index);
      if (!ComputeIsPartTouchingBox(direction, halves)) {
        continue;
      }
      const SmallestCellPoint part_low = {
          .x = box_low.x + (direction.x * box_width) + (halves.x * part_width),
          .y = box_low.y + (direction.y * box_width) + (halves.y * part_width),
          .z = box_low.z + (direction.z * box_width) + (halves.z * part_width),
      };
      if (FindSizeLevelAtPoint(committed, part_low) >= size_level + 2) {
        return true;
      }
    }
  }
  return false;
}

// Finds the two facts the committed cells around a box that is not in the grid give about it: whether a committed cell
// two or more size levels smaller touches it, and whether one that is bigger does.
//
// Property: when the committed grid obeys the size step rule, a biggest cell is empty or fully covered, and no box
//   inside an empty biggest cell has both facts at once. So a box that must split never touches a bigger committed
//   cell. Why, in short: a box B must split because a committed cell K, at most a quarter of B's width, touches it, and
//   K lies in another biggest cell N. A committed cell C at least twice as wide as B and touching B is aligned to its
//   width, so it also meets the face, edge or corner that B shares with N. The committed cell D of N that lies there
//   then touches C or is C, so by the rule D is at least B's width, hence covers the place of B's size beside B, and K
//   touches D at four times its width or more, which the rule forbids. Likewise two new cells of one cube obey the
//   rule between themselves: if X and Y touch and Y's size is two or more size levels smaller, the next bigger cell of
//   Y was split by a committed cell K, and K breaks the rule as above with X in the place of C, except that what
//   bounds the other cell's width is that the new cell X was not split, so no committed cell two or more size levels
//   smaller touches it.
// How a caller uses the two facts: split the box into its 8 small cells when touches_much_smaller_cell is true,
//   whatever touches_bigger_cell says; otherwise the box is a new cell as it is. Both facts true at once cannot happen
//   on a valid committed grid, so a caller treats it as an internal fault of the committed grid.
// Method: integers only, no allocation, no recursion, fixed work. The box is the cell of `new_cell`; it and every place
//   below are boxes of whole smallest cells of the frame.
//   touches_bigger_cell: for each of the 26 places around the box at its size level, FindCellAtCoord at the place's low
//   corner; true when a cell is found whose size level is at most the box's minus 1. A touching cell that big covers a
//   whole neighbouring place.
//   touches_much_smaller_cell: for each of the 26 places, for each of its 8 half-width parts that touch the box (4 for
//   a face place, 2 for an edge place, 1 for a corner place: 56 in all), FindCellAtCoord at the part's low corner; true
//   when a cell is found whose size level is at least the box's plus 2. If that part is split, its cells cover it, so
//   one of them touches the box. False without any look-up when the box's size level plus 2 is past the last size
//   level. A place or part outside the cells size-level-0 ids can name counts as empty.
// Cost: at most 82 look-ups, 26 and 56. At the last two size levels the 56 are not made.
// Example: frame of 4 size levels, biggest cells 1 m wide, origin 0. The committed cell A is the size-level-0 cell
//   {1,-1,0}. The size-level-0 cell P at {0,0,0} is split into its 8 size-level-1 cells; its size-level-1 cell {1,1,0}
//   into 8 size-level-2 cells, and the one at {3,3,0} into 8 size-level-3 cells; all other cells active. The
//   size-level-0 cell Q at {1,0,0} (x 1 to 2 m, y 0 to 1 m, z 0 to 1 m) is the new cell. Of its 26 places, the one at
//   {1,-1,0} holds A, size level 0, not at most -1: no bigger cell. Its place at {0,0,0} has 4 parts that touch it, the
//   ones with x 0.5 to 1 m; the part at y 0.5 to 1 m, z 0 to 0.5 m has its low corner in the size-level-2 cell {2,2,0}:
//   2 >= 0 + 2, so touches_much_smaller_cell is true. Q's size-level-1 cell {2,0,0} (y 0 to 0.5 m) touches A, size
//   level 0 <= 1 - 1: touches_bigger_cell is true, and no cell of size level 3 reaches below y 0.75 m:
//   touches_much_smaller_cell is false.
// Requires: `new_cell` is a valid id whose box lies in space with no cell of the grid: no committed cell overlaps it
//   (a debug assertion looks up its low corner). `committed` is the tables of a built grid, whose frame passed the
//   frame check, and `new_cell` has a size level below the frame's size_level_count.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-size-rule>
FLUIDITY_CPU_AND_GPU constexpr NewCellNeighbors
FindNewCellNeighbors(const GridTables& committed, CellId new_cell) noexcept {
  const int size_level = static_cast<int>(ToSizeLevel(new_cell));
  const CellCoord coord = ToCellCoord(new_cell);
  const int size_level_count = committed.frame.size_level_count;
  assert(size_level < size_level_count);
  const auto box_width =
      static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(size_level_count - 1 - size_level));
  const SmallestCellPoint box_low = {
      .x = coord.x * box_width,
      .y = coord.y * box_width,
      .z = coord.z * box_width,
  };
  assert(FindSizeLevelAtPoint(committed, box_low) < 0);
  return {
      .touches_much_smaller_cell = ComputeTouchesMuchSmallerCell(committed, size_level, box_low, box_width),
      .touches_bigger_cell = ComputeTouchesBiggerCell(committed, size_level, box_low, box_width),
  };
}

}  // namespace fluidity::grid
