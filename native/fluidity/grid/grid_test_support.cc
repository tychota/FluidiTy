#include "fluidity/grid/grid_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/buffer.h"
#include "fluidity/base/memory_count.h"
#include "fluidity/base/run_memory.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_memory_estimate.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {
namespace {

constexpr std::array<Axis, 3> kAxes = {Axis::kX, Axis::kY, Axis::kZ};

// The two cells beside a face piece; either is none at the grid's edge.
struct CellsBeside {
  std::optional<CellId> low;
  std::optional<CellId> high;
};

int ToSizeLevelNumber(CellId cell) {
  return static_cast<int>(ToSizeLevel(cell));
}

// The width of a cell of size_level in units of size level size_level_count - 1: 2^(last size level - size_level).
std::int64_t ComputeWidthInUnits(SizeLevel size_level, int size_level_count) {
  const int last_size_level = size_level_count - 1;
  const auto halving_count = static_cast<unsigned>(last_size_level - static_cast<int>(size_level));
  return static_cast<std::int64_t>(std::uint64_t{1} << halving_count);
}

// m. The biggest cell width halved size_level times; each halving of a double is exact.
double ComputeCellWidth(GridFrame frame, int size_level) {
  double cell_width = frame.biggest_cell_width;
  for (int halving = 0; halving < size_level; ++halving) {
    cell_width /= 2;
  }
  return cell_width;
}

std::vector<UnitBox> ToUnitBoxes(const std::vector<CellId>& cells, int size_level_count) {
  std::vector<UnitBox> boxes;
  boxes.reserve(cells.size());
  for (const CellId cell : cells) {
    boxes.push_back(ToUnitBox(cell, size_level_count));
  }
  return boxes;
}

// Two boxes touch when they share at least one point: their closed intervals meet on every axis. Overlapping boxes
// touch too; a valid list has none.
bool ComputeIsTouching(const UnitBox& first, const UnitBox& second) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const bool is_first_beyond = first.low.at(axis) > second.low.at(axis) + second.width;
    const bool is_second_beyond = second.low.at(axis) > first.low.at(axis) + first.width;
    if (is_first_beyond || is_second_beyond) {
      return false;
    }
  }
  return true;
}

// Units. The length that the open intervals of two boxes share along one axis; 0 or less when they share none.
std::int64_t ComputeSharedLength(const UnitBox& first, const UnitBox& second, std::size_t axis) {
  const std::int64_t shared_low = std::max(first.low.at(axis), second.low.at(axis));
  const std::int64_t shared_high = std::min(first.low.at(axis) + first.width, second.low.at(axis) + second.width);
  return shared_high - shared_low;
}

// Units squared. The area two boxes share across `axis`, the square where their sides would meet; 0 when none.
std::int64_t ComputeSharedArea(const UnitBox& first, const UnitBox& second, std::size_t axis) {
  std::int64_t shared_area = 1;
  for (std::size_t across = 0; across < 3; ++across) {
    if (across != axis) {
      shared_area *= std::max<std::int64_t>(ComputeSharedLength(first, second, across), 0);
    }
  }
  return shared_area;
}

CellCoord ToCellCoordOfPlace(std::array<std::int32_t, 3> coord) {
  return {.x = coord.at(0), .y = coord.at(1), .z = coord.at(2)};
}

// Builds one face piece from its square: axis, size level and low corner in units. Along the axis the low corner is
// the plane of the piece, the low side of the place on its high side, so that place's coordinate at the piece's size
// level is the low corner divided by the piece's width.
GeometricFacePiece BuildFacePiece(
    GridFrame frame,
    Axis axis,
    int size_level,
    std::array<std::int64_t, 3> piece_low_corner,
    CellsBeside cells_beside
) {
  const std::int64_t width = ComputeWidthInUnits(static_cast<SizeLevel>(size_level), frame.size_level_count);
  std::array<std::int32_t, 3> high_place{};
  for (std::size_t index = 0; index < 3; ++index) {
    // Exact: a piece's square lies on the lattice of its own size level.
    EXPECT_EQ(piece_low_corner.at(index) % width, 0);
    high_place.at(index) = static_cast<std::int32_t>(piece_low_corner.at(index) / width);
  }
  std::array<std::int32_t, 3> low_place = high_place;
  low_place.at(static_cast<std::size_t>(axis)) -= 1;
  const CellCoord cell_high_coord = ToCellCoordOfPlace(high_place);
  // One of the two places is the piece's smaller cell, a listed cell, so it is in range. The id holds the high place,
  // which may be kMaxCoord + 1 along the axis.
  EXPECT_TRUE(ComputeIsCoordInRange(cell_high_coord) || ComputeIsCoordInRange(ToCellCoordOfPlace(low_place)));
  const double cell_width = ComputeCellWidth(frame, size_level);
  const auto piece_size_level = static_cast<SizeLevel>(size_level);
  return {
      .id = ToFacePieceId(axis, piece_size_level, cell_high_coord),
      .axis = axis,
      .size_level = piece_size_level,
      .cell_high_coord = cell_high_coord,
      .cell_low = cells_beside.low,
      .cell_high = cells_beside.high,
      .area = cell_width * cell_width,
  };
}

// The centre of a face piece in half units: along its axis twice its plane, across it twice its low corner plus its
// width. Half units keep the centre of a smallest cell's side a whole number.
std::array<std::int64_t, 3> ToDoubledCentre(const GeometricFacePiece& piece, int size_level_count) {
  const std::int64_t width = ComputeWidthInUnits(piece.size_level, size_level_count);
  const std::array<std::int64_t, 3> low_corner = {
      piece.cell_high_coord.x * width,
      piece.cell_high_coord.y * width,
      piece.cell_high_coord.z * width,
  };
  std::array<std::int64_t, 3> doubled_centre{};
  for (std::size_t index = 0; index < 3; ++index) {
    const bool is_along_axis = index == static_cast<std::size_t>(piece.axis);
    doubled_centre.at(index) = 2 * low_corner.at(index) + (is_along_axis ? 0 : width);
  }
  return doubled_centre;
}

// floor(value / divisor) for a positive divisor, a negative value included.
std::int64_t ComputeFloorDivision(std::int64_t value, std::int64_t divisor) {
  const std::int64_t quotient = value / divisor;
  const bool is_rounded_toward_zero = value % divisor != 0 && value < 0;
  return is_rounded_toward_zero ? quotient - 1 : quotient;
}

// The cell of one size level at given coordinates, listed or not. The coordinates may lie outside the range ids
// can name.
struct Place {
  int size_level = 0;
  std::array<std::int64_t, 3> coord{};  // In cells of size_level, indexed by axis (x, y, z).
};

Place ToPlace(CellId cell) {
  const CellCoord coord = ToCellCoord(cell);
  return {.size_level = ToSizeLevelNumber(cell), .coord = {coord.x, coord.y, coord.z}};
}

// The id of a place, or none when a coordinate lies outside -kMaxCoord..kMaxCoord: no id names such a place, so no
// listed cell is there, and none inside it either.
std::optional<CellId> FindPlaceId(Place place) {
  for (const std::int64_t coord : place.coord) {
    if (coord < -kMaxCoord || coord > kMaxCoord) {
      return std::nullopt;
    }
  }
  const CellCoord coord = {
      .x = static_cast<std::int32_t>(place.coord.at(0)),
      .y = static_cast<std::int32_t>(place.coord.at(1)),
      .z = static_cast<std::int32_t>(place.coord.at(2)),
  };
  return ToCellId(static_cast<SizeLevel>(place.size_level), coord);
}

// The box of a place in whole units of size level size_level_count - 1, as ToUnitBox gives for a cell.
UnitBox ToPlaceBox(Place place, int size_level_count) {
  const std::int64_t width = ComputeWidthInUnits(static_cast<SizeLevel>(place.size_level), size_level_count);
  return {
      .low = {place.coord.at(0) * width, place.coord.at(1) * width, place.coord.at(2) * width},
      .width = width,
  };
}

// The checker's own lookup of a list by place: the ids of the active cells, and of every place that holds a smaller
// active cell, which is split. Made from the list alone, with ToCellId to name places, so it shares nothing with the
// grid's tables.
struct CellsByPlace {
  std::unordered_set<std::uint64_t> active_values;
  std::unordered_set<std::uint64_t> split_values;
};

// Adds one active cell and every bigger place that holds it. The bigger places are coordinates floor(c / 2), one size
// level up at a time, and stay in range. A place already known as split has its bigger places known too, so the walk
// stops there.
void ApplyActiveCell(CellId cell, CellsByPlace& cells_by_place) {
  cells_by_place.active_values.insert(cell.value);
  Place place = ToPlace(cell);
  while (place.size_level > 0) {
    place.size_level -= 1;
    for (std::int64_t& coord : place.coord) {
      coord = ComputeFloorDivision(coord, 2);
    }
    const bool is_new = cells_by_place.split_values.insert(FindPlaceId(place).value_or(CellId{0}).value).second;
    if (!is_new) {
      break;
    }
  }
}

// Requires: no repeated and no overlapping cells, as BuildSizeStepClosure and GeometricGrid state. A broken
// requirement stops a debug build at an assert; the searches would otherwise give answers that differ from the plain
// scans they replace. A release build skips the checks.
CellsByPlace BuildCellsByPlace(const std::vector<CellId>& cells) {
  CellsByPlace cells_by_place;
  for (const CellId cell : cells) {
    const bool is_cell_listed_once = !cells_by_place.active_values.contains(cell.value);
    assert(is_cell_listed_once);
    static_cast<void>(is_cell_listed_once);
    ApplyActiveCell(cell, cells_by_place);
  }
  // Two cells overlap only when one holds the other, so the bigger one is active and among the split places too.
  for (const CellId cell : cells) {
    const bool holds_no_listed_cell = !cells_by_place.split_values.contains(cell.value);
    assert(holds_no_listed_cell);
    static_cast<void>(holds_no_listed_cell);
  }
  return cells_by_place;
}

// How far FindActiveCellsInside searches.
enum class SearchEnd : std::uint8_t {
  kEveryWantedCell,  // Finds every wanted cell.
  kFirstWantedCell,  // Stops at the first wanted cell, for a yes-or-no question.
};

// Finds the active cells that is_wanted accepts inside the places to visit, going down through split places: of a
// split place, its 8 small places are visited. A place that is_kept turns down is skipped with everything inside it.
// The order is the search's own.
//
// Requires: is_kept accepts a small place only if it accepts the place that holds it, so a place turned down hides
//   nothing the caller wants.
template <typename IsKept, typename IsWanted>
std::vector<CellId> FindActiveCellsInside(
    const CellsByPlace& cells_by_place,
    std::vector<Place> places_to_visit,
    const IsKept& is_kept,
    const IsWanted& is_wanted,
    SearchEnd search_end
) {
  std::vector<CellId> found_cells;
  while (!places_to_visit.empty()) {
    const Place place = places_to_visit.back();
    places_to_visit.pop_back();
    const std::optional<CellId> place_id = FindPlaceId(place);
    if (!place_id.has_value() || !is_kept(place)) {
      continue;
    }
    if (cells_by_place.active_values.contains(place_id->value)) {
      if (is_wanted(*place_id)) {
        found_cells.push_back(*place_id);
        if (search_end == SearchEnd::kFirstWantedCell) {
          return found_cells;
        }
      }
      continue;
    }
    if (!cells_by_place.split_values.contains(place_id->value)) {
      continue;
    }
    for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      const auto eighth_bits = static_cast<unsigned>(cell_eighth);
      Place small_place = {.size_level = place.size_level + 1, .coord = {}};
      for (std::size_t index = 0; index < 3; ++index) {
        const std::int64_t bit = (eighth_bits >> index) & 1U;
        small_place.coord.at(index) = 2 * place.coord.at(index) + bit;
      }
      places_to_visit.push_back(small_place);
    }
  }
  return found_cells;
}

// Finds the active cells beyond the high side of a cell along `axis`: the cells whose low side lies on that side's
// plane and shares some area with it. Each is found once; the order is the search's own.
//
// Method: a cell as big as this one or bigger is aligned, so it covers the whole side and holds the unit just beyond
//   the side's low corner; the place holding that unit is looked up at each size level 0 to the cell's own. Smaller
//   cells lie inside the place of the cell's size level beyond the side; FindActiveCellsInside goes down from it
//   through the places that touch the cell and share area with its side.
// Requires: the list has no overlapping cells, so a cell found the first way leaves no smaller cell to find.
std::vector<CellId>
FindCellsBeyondHighSide(const CellsByPlace& cells_by_place, CellId cell, Axis axis, int size_level_count) {
  const auto axis_index = static_cast<std::size_t>(axis);
  const UnitBox box = ToUnitBox(cell, size_level_count);

  std::array<std::int64_t, 3> unit_beyond = box.low;
  unit_beyond.at(axis_index) += box.width;
  for (int size_level = 0; size_level <= ToSizeLevelNumber(cell); ++size_level) {
    const std::int64_t place_width = ComputeWidthInUnits(static_cast<SizeLevel>(size_level), size_level_count);
    Place place = {.size_level = size_level, .coord = {}};
    for (std::size_t index = 0; index < 3; ++index) {
      place.coord.at(index) = ComputeFloorDivision(unit_beyond.at(index), place_width);
    }
    const std::optional<CellId> place_id = FindPlaceId(place);
    if (place_id.has_value() && cells_by_place.active_values.contains(place_id->value)) {
      return {*place_id};
    }
  }

  Place place_beyond = ToPlace(cell);
  place_beyond.coord.at(axis_index) += 1;
  const auto is_on_side = [&box, axis_index, size_level_count](Place place) {
    const UnitBox place_box = ToPlaceBox(place, size_level_count);
    return ComputeIsTouching(place_box, box) && ComputeSharedArea(place_box, box, axis_index) > 0;
  };
  const auto is_any_cell = [](CellId /*cell*/) {
    return true;
  };
  return FindActiveCellsInside(cells_by_place, {place_beyond}, is_on_side, is_any_cell, SearchEnd::kEveryWantedCell);
}

// Says whether an active cell two or more size levels smaller than `cell` touches it, by a side, an edge or a corner.
//
// Method: such a cell lies inside one of the 26 places of the size level of `cell` around it, and every bigger place
//   that holds it touches `cell` too. So FindActiveCellsInside starts from those 26 places and goes down through the
//   places that touch `cell`.
// Requires: `cell` is active and overlaps no other active cell.
bool ComputeIsTouchedByMuchSmallerCell(const CellsByPlace& cells_by_place, CellId cell, int size_level_count) {
  const int size_level = ToSizeLevelNumber(cell);
  // A cell of the last two size levels has no cell two size levels smaller.
  if (size_level + 2 >= size_level_count) {
    return false;
  }
  const UnitBox box = ToUnitBox(cell, size_level_count);
  const Place cell_place = ToPlace(cell);
  std::vector<Place> places_around;
  for (int neighbour = 0; neighbour < 27; ++neighbour) {
    const std::array<std::int64_t, 3> offset = {neighbour % 3 - 1, (neighbour / 3) % 3 - 1, neighbour / 9 - 1};
    if (offset == std::array<std::int64_t, 3>{0, 0, 0}) {
      continue;
    }
    Place place = cell_place;
    for (std::size_t index = 0; index < 3; ++index) {
      place.coord.at(index) += offset.at(index);
    }
    places_around.push_back(place);
  }
  const auto is_touching = [&box, size_level_count](Place place) {
    return ComputeIsTouching(ToPlaceBox(place, size_level_count), box);
  };
  const auto is_much_smaller = [size_level](CellId touching_cell) {
    return ToSizeLevelNumber(touching_cell) >= size_level + 2;
  };
  const std::vector<CellId> much_smaller_cells =
      FindActiveCellsInside(cells_by_place, places_around, is_touching, is_much_smaller, SearchEnd::kFirstWantedCell);
  return !much_smaller_cells.empty();
}

// A list of active cells with what BuildFacePieces reads of it: the frame, each entry's box, the checker's lookup by
// place, and each cell's entry by id. The checker requires distinct cells, so each id names one entry.
struct ListOfCells {
  GridFrame frame;
  const std::vector<CellId>& cells;
  const std::vector<UnitBox>& boxes;
  CellsByPlace cells_by_place;
  std::unordered_map<std::uint64_t, std::size_t> index_by_value;
};

ListOfCells BuildListOfCells(GridFrame frame, const std::vector<CellId>& cells, const std::vector<UnitBox>& boxes) {
  ListOfCells list = {
      .frame = frame,
      .cells = cells,
      .boxes = boxes,
      .cells_by_place = BuildCellsByPlace(cells),
      .index_by_value = {},
  };
  for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
    list.index_by_value.insert({cells.at(cell_index).value, cell_index});
  }
  return list;
}

// Units squared, per entry of a list and along one axis: the area of the entry's high side and of its low side that
// other cells cover.
struct SideCover {
  std::vector<std::int64_t> high_side_areas;
  std::vector<std::int64_t> low_side_areas;
};

// Appends the face pieces between two cells along `axis` and gives the area they cover on each side of each cell.
// Every cell whose low side lies on a cell's high side, and shares some area with it, makes one piece: the smaller
// cell's side, at its size level, since aligned cells of two sizes that meet share the whole side of the smaller one.
// So each piece is found once, from its low cell, and its area counts on the low cell's high side and on the high
// cell's low side.
SideCover ApplyPiecesBetweenCells(const ListOfCells& list, Axis axis, std::vector<GeometricFacePiece>& pieces) {
  const auto axis_index = static_cast<std::size_t>(axis);
  const int size_level_count = list.frame.size_level_count;
  SideCover cover = {
      .high_side_areas = std::vector<std::int64_t>(list.cells.size(), 0),
      .low_side_areas = std::vector<std::int64_t>(list.cells.size(), 0),
  };
  for (std::size_t cell_index = 0; cell_index < list.cells.size(); ++cell_index) {
    const CellId cell = list.cells.at(cell_index);
    const UnitBox& box = list.boxes.at(cell_index);
    for (const CellId other_cell : FindCellsBeyondHighSide(list.cells_by_place, cell, axis, size_level_count)) {
      const std::size_t other_index = list.index_by_value.at(other_cell.value);
      const UnitBox& other_box = list.boxes.at(other_index);
      const std::int64_t shared_area = ComputeSharedArea(box, other_box, axis_index);
      cover.high_side_areas.at(cell_index) += shared_area;
      cover.low_side_areas.at(other_index) += shared_area;
      const bool is_this_cell_smaller = box.width <= other_box.width;
      const CellId smaller_cell = is_this_cell_smaller ? cell : other_cell;
      std::array<std::int64_t, 3> piece_low_corner = is_this_cell_smaller ? box.low : other_box.low;
      piece_low_corner.at(axis_index) = box.low.at(axis_index) + box.width;
      pieces.push_back(BuildFacePiece(
          list.frame,
          axis,
          ToSizeLevelNumber(smaller_cell),
          piece_low_corner,
          {.low = cell, .high = other_cell}
      ));
    }
  }
  return cover;
}

// Appends one piece at the grid's edge for each side along `axis` that no cell covers. A side covered only in part
// breaks the siblings rule, which the checker requires, and fails the calling test.
void ApplyEdgePieces(
    const ListOfCells& list,
    Axis axis,
    const SideCover& cover,
    std::vector<GeometricFacePiece>& pieces
) {
  const auto axis_index = static_cast<std::size_t>(axis);
  for (std::size_t cell_index = 0; cell_index < list.cells.size(); ++cell_index) {
    const CellId cell = list.cells.at(cell_index);
    const UnitBox& box = list.boxes.at(cell_index);
    const int size_level = ToSizeLevelNumber(cell);
    const std::int64_t side_area = box.width * box.width;
    const std::int64_t high_side_covered_area = cover.high_side_areas.at(cell_index);
    const std::int64_t low_side_covered_area = cover.low_side_areas.at(cell_index);
    if (high_side_covered_area == 0) {
      std::array<std::int64_t, 3> piece_low_corner = box.low;
      piece_low_corner.at(axis_index) += box.width;
      pieces.push_back(
          BuildFacePiece(list.frame, axis, size_level, piece_low_corner, {.low = cell, .high = std::nullopt})
      );
    }
    if (low_side_covered_area == 0) {
      pieces.push_back(BuildFacePiece(list.frame, axis, size_level, box.low, {.low = std::nullopt, .high = cell}));
    }
    const bool is_high_side_whole = high_side_covered_area == 0 || high_side_covered_area == side_area;
    const bool is_low_side_whole = low_side_covered_area == 0 || low_side_covered_area == side_area;
    EXPECT_TRUE(is_high_side_whole && is_low_side_whole)
        << "a side of cell " << cell.value << " is covered only in part: siblings are missing";
  }
}

// Finds every face piece of a list (method paper, <sec-face-pieces>), in the order of FacePieces(): the pieces between
// two cells (ApplyPiecesBetweenCells), then a piece at the grid's edge for each side no cell covers.
std::vector<GeometricFacePiece>
BuildFacePieces(GridFrame frame, const std::vector<CellId>& cells, const std::vector<UnitBox>& boxes) {
  const int size_level_count = frame.size_level_count;
  const ListOfCells list = BuildListOfCells(frame, cells, boxes);
  std::vector<GeometricFacePiece> pieces;
  for (const Axis axis : kAxes) {
    const SideCover cover = ApplyPiecesBetweenCells(list, axis, pieces);
    ApplyEdgePieces(list, axis, cover, pieces);
  }

  // By axis, then by centre in z, y, x. Pieces of one axis never overlap, so no two share a centre and the order is
  // the same whatever order the sort compares in. Each key is found once.
  struct KeyedPiece {
    std::array<std::int64_t, 4> key;  // The axis, then the doubled centre's z, y and x.
    std::size_t index;  // The piece's entry in `pieces`.
  };
  std::vector<KeyedPiece> keyed_pieces;
  keyed_pieces.reserve(pieces.size());
  for (std::size_t index = 0; index < pieces.size(); ++index) {
    const std::array<std::int64_t, 3> centre = ToDoubledCentre(pieces.at(index), size_level_count);
    const auto axis_number = static_cast<std::int64_t>(pieces.at(index).axis);
    keyed_pieces.push_back({.key = {axis_number, centre.at(2), centre.at(1), centre.at(0)}, .index = index});
  }
  std::ranges::sort(keyed_pieces, [](const KeyedPiece& first, const KeyedPiece& second) {
    return first.key < second.key;
  });
  std::vector<GeometricFacePiece> ordered_pieces;
  ordered_pieces.reserve(pieces.size());
  for (const KeyedPiece& keyed_piece : keyed_pieces) {
    ordered_pieces.push_back(pieces.at(keyed_piece.index));
  }
  return ordered_pieces;
}

// Puts cells in the checker's own order: by the low corner in units, z, then y, then x, and at one low corner the
// bigger cell first. It is the fixed order of the method paper (<sec-cell-blocks>) stated in units, so a test can
// compare ComesBefore with it. Each cell's key is found once; only one cell has a given key, so the order is the same
// whatever order the sort compares in.
void ApplyUnitOrder(std::vector<CellId>& cells, int size_level_count) {
  struct KeyedCell {
    std::array<std::int64_t, 4> key;  // Units: low z, low y, low x, minus the width.
    CellId cell;
  };
  std::vector<KeyedCell> keyed_cells;
  keyed_cells.reserve(cells.size());
  for (const CellId cell : cells) {
    const UnitBox box = ToUnitBox(cell, size_level_count);
    keyed_cells.push_back({.key = {box.low.at(2), box.low.at(1), box.low.at(0), -box.width}, .cell = cell});
  }
  std::ranges::sort(keyed_cells, [](const KeyedCell& first, const KeyedCell& second) {
    return first.key < second.key;
  });
  for (std::size_t index = 0; index < cells.size(); ++index) {
    cells.at(index) = keyed_cells.at(index).cell;
  }
}

}  // namespace

std::uint64_t ToUniformIndex(std::uint64_t raw, std::uint64_t count) {
  if (count == 0) {
    ADD_FAILURE() << "ToUniformIndex needs a count of 1 or more";
    return 0;
  }
  return raw % count;
}

double ToUnitDouble(std::uint64_t raw) {
  constexpr double kTwoToMinusFiftyThree = 1.0 / 9007199254740992.0;
  return static_cast<double>(raw >> 11U) * kTwoToMinusFiftyThree;
}

// clang-tidy: as in the header, a seed and two counts.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<CellId> BuildRandomGridCells(std::uint64_t seed, int size_level_count, int biggest_cells_per_axis) {
  EXPECT_TRUE(size_level_count >= 1 && size_level_count <= kMaxSizeLevels);
  EXPECT_TRUE(biggest_cells_per_axis >= 1 && biggest_cells_per_axis <= kMaxCoord + 1);
  std::mt19937_64 generator(seed);

  // Size-level-0 cells in the fixed order: z, then y, then x.
  std::vector<CellId> cells_to_visit;
  for (std::int32_t z = 0; z < biggest_cells_per_axis; ++z) {
    for (std::int32_t y = 0; y < biggest_cells_per_axis; ++y) {
      for (std::int32_t x = 0; x < biggest_cells_per_axis; ++x) {
        cells_to_visit.push_back(ToCellId(SizeLevel{0}, {.x = x, .y = y, .z = z}));
      }
    }
  }

  std::vector<CellId> active_cells;
  const int last_size_level = size_level_count - 1;
  for (int size_level = 0; size_level < last_size_level; ++size_level) {
    // One chance in two for a size-level-0 cell, one in four for a smaller one.
    const std::uint64_t split_choice_count = size_level == 0 ? 2 : 4;
    std::vector<CellId> small_cells;
    for (const CellId cell : cells_to_visit) {
      const std::uint64_t raw = generator();
      if (ToUniformIndex(raw, split_choice_count) != 0) {
        active_cells.push_back(cell);
        continue;
      }
      for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
        small_cells.push_back(ComputeSmallCell(cell, cell_eighth));
      }
    }
    ApplyUnitOrder(small_cells, size_level_count);
    cells_to_visit = std::move(small_cells);
  }
  // Cells of the last size level cannot be split and take no number.
  active_cells.insert(active_cells.end(), cells_to_visit.begin(), cells_to_visit.end());
  return BuildSizeStepClosure(std::move(active_cells), size_level_count);
}

void ApplyFixedShuffle(std::vector<CellId>& cells, std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  if (cells.size() < 2) {
    return;
  }
  for (std::size_t index = cells.size() - 1; index >= 1; --index) {
    const std::uint64_t other_index = ToUniformIndex(generator(), index + 1);
    std::swap(cells.at(index), cells.at(static_cast<std::size_t>(other_index)));
  }
}

const std::vector<std::vector<CellId>>& BuildShuffledRandomGrids() {
  static const std::vector<std::vector<CellId>> shuffled_grids = [] {
    std::vector<std::vector<CellId>> grids;
    grids.reserve(kRandomGridCount);
    for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
      std::vector<CellId> cells = BuildRandomGridCells(seed, 4, 3);
      ApplyFixedShuffle(cells, seed + 1000);
      grids.push_back(std::move(cells));
    }
    return grids;
  }();
  return shuffled_grids;
}

std::vector<CellId> BuildSizeStepClosure(std::vector<CellId> cells, int size_level_count) {
  // Each round splits every cell that touches a cell two or more size levels smaller. A split makes new such pairs
  // only with cells bigger still, so the rounds end.
  // The lookup is built once and kept up to date as cells are split.
  CellsByPlace cells_by_place = BuildCellsByPlace(cells);
  while (true) {
    // Every cell of a round is judged on the list as it was when the round began, then the round's splits are made.
    std::vector<bool> is_too_big(cells.size(), false);
    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
      is_too_big.at(cell_index) =
          ComputeIsTouchedByMuchSmallerCell(cells_by_place, cells.at(cell_index), size_level_count);
    }
    std::vector<CellId> closed_cells;
    int split_count = 0;
    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
      const CellId cell = cells.at(cell_index);
      if (!is_too_big.at(cell_index)) {
        closed_cells.push_back(cell);
        continue;
      }
      // The small cells are at 2 c or 2 c + 1 on each axis: both must be in range before ids of them are made.
      const CellCoord coord = ToCellCoord(cell);
      const CellCoord low_small_coord = {.x = 2 * coord.x, .y = 2 * coord.y, .z = 2 * coord.z};
      const CellCoord high_small_coord = {.x = 2 * coord.x + 1, .y = 2 * coord.y + 1, .z = 2 * coord.z + 1};
      if (!ComputeIsCoordInRange(low_small_coord) || !ComputeIsCoordInRange(high_small_coord)) {
        ADD_FAILURE() << "cell " << cell.value << " must be split, but its small cells lie beyond kMaxCoord";
        closed_cells.push_back(cell);
        continue;
      }
      cells_by_place.active_values.erase(cell.value);
      for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
        const CellId small_cell = ComputeSmallCell(cell, cell_eighth);
        closed_cells.push_back(small_cell);
        ApplyActiveCell(small_cell, cells_by_place);
      }
      ++split_count;
    }
    cells = std::move(closed_cells);
    if (split_count == 0) {
      break;
    }
  }
  ApplyUnitOrder(cells, size_level_count);
  return cells;
}

UnitBox ToUnitBox(CellId cell, int size_level_count) {
  const int size_level = ToSizeLevelNumber(cell);
  EXPECT_LT(size_level, size_level_count) << cell.value;
  const std::int64_t width = ComputeWidthInUnits(static_cast<SizeLevel>(size_level), size_level_count);
  const CellCoord coord = ToCellCoord(cell);
  return {
      .low = {coord.x * width, coord.y * width, coord.z * width},
      .width = width,
  };
}

GeometricGrid::GeometricGrid(GridFrame frame, std::vector<CellId> active_cells)
    : frame_(frame),
      active_cells_(std::move(active_cells)),
      active_boxes_(ToUnitBoxes(active_cells_, frame.size_level_count)),
      face_pieces_(BuildFacePieces(frame_, active_cells_, active_boxes_)) {
}

const std::vector<GeometricFacePiece>& GeometricGrid::FacePieces() const {
  return face_pieces_;
}

std::vector<CellId> GeometricGrid::TouchingCells(CellId cell) const {
  const UnitBox box = ToUnitBox(cell, frame_.size_level_count);
  std::vector<CellId> touching_cells;
  for (std::size_t index = 0; index < active_cells_.size(); ++index) {
    const CellId other = active_cells_.at(index);
    const bool is_same_cell = other.value == cell.value;
    if (!is_same_cell && ComputeIsTouching(box, active_boxes_.at(index))) {
      touching_cells.push_back(other);
    }
  }
  ApplyUnitOrder(touching_cells, frame_.size_level_count);
  return touching_cells;
}

std::vector<std::array<std::int64_t, 3>> GeometricGrid::CornersInUnits() const {
  // Each corner as (z, y, x), so the set's own order is z, then y, then x.
  std::set<std::array<std::int64_t, 3>> corners_by_z_y_x;
  for (const UnitBox& box : active_boxes_) {
    for (int corner = 0; corner < 8; ++corner) {
      const auto corner_bits = static_cast<unsigned>(corner);
      const std::int64_t x = box.low.at(0) + ((corner_bits & 1U) != 0 ? box.width : 0);
      const std::int64_t y = box.low.at(1) + ((corner_bits & 2U) != 0 ? box.width : 0);
      const std::int64_t z = box.low.at(2) + ((corner_bits & 4U) != 0 ? box.width : 0);
      corners_by_z_y_x.insert({z, y, x});
    }
  }
  std::vector<std::array<std::int64_t, 3>> corners;
  corners.reserve(corners_by_z_y_x.size());
  for (const std::array<std::int64_t, 3>& corner : corners_by_z_y_x) {
    corners.push_back({corner.at(2), corner.at(1), corner.at(0)});
  }
  return corners;
}

std::vector<Vec3> GeometricGrid::Corners() const {
  const double unit_width = ComputeCellWidth(frame_, frame_.size_level_count - 1);
  std::vector<Vec3> corners;
  for (const std::array<std::int64_t, 3>& corner : CornersInUnits()) {
    corners.push_back({
        .x = frame_.origin.x + static_cast<double>(corner.at(0)) * unit_width,
        .y = frame_.origin.y + static_cast<double>(corner.at(1)) * unit_width,
        .z = frame_.origin.z + static_cast<double>(corner.at(2)) * unit_width,
    });
  }
  return corners;
}

std::optional<CellId> GeometricGrid::ContainingCell(Vec3 position) const {
  const int size_level_count = frame_.size_level_count;
  const double unit_width = ComputeCellWidth(frame_, size_level_count - 1);
  // The units a size-level-0 id can name: -kMaxCoord w_0 up to (kMaxCoord + 1) w_0 - 1, w_0 the size-level-0 width in
  // units. Below 2^34, so exact as doubles.
  const std::int64_t biggest_width = ComputeWidthInUnits(SizeLevel{0}, size_level_count);
  const auto lowest_unit = static_cast<double>(-std::int64_t{kMaxCoord} * biggest_width);
  const auto beyond_highest_unit = static_cast<double>((std::int64_t{kMaxCoord} + 1) * biggest_width);
  const std::array<double, 3> offsets = {
      position.x - frame_.origin.x,
      position.y - frame_.origin.y,
      position.z - frame_.origin.z,
  };
  std::array<std::int64_t, 3> unit{};
  for (std::size_t index = 0; index < 3; ++index) {
    // Method paper <sec-cell-sizes>: floor((x - origin) / w_min), tested before any integer conversion.
    const double scaled_offset = offsets.at(index) / unit_width;
    if (!std::isfinite(scaled_offset)) {
      return std::nullopt;
    }
    const double floored_offset = std::floor(scaled_offset);
    if (floored_offset < lowest_unit || floored_offset >= beyond_highest_unit) {
      return std::nullopt;
    }
    unit.at(index) = static_cast<std::int64_t>(floored_offset);
  }
  for (std::size_t cell_index = 0; cell_index < active_cells_.size(); ++cell_index) {
    const UnitBox& box = active_boxes_.at(cell_index);
    bool holds_unit = true;
    for (std::size_t index = 0; index < 3; ++index) {
      const bool is_inside_axis = box.low.at(index) <= unit.at(index) && unit.at(index) < box.low.at(index) + box.width;
      holds_unit = holds_unit && is_inside_axis;
    }
    if (holds_unit) {
      return active_cells_.at(cell_index);
    }
  }
  return std::nullopt;
}

std::vector<GeometricFacePiece>
GeometricGrid::NearbyFaces(CellId cell, std::array<int, 3> window_quarters, Axis axis) const {
  const bool is_cell_active = std::ranges::any_of(active_cells_, [cell](CellId active_cell) {
    return active_cell.value == cell.value;
  });
  const bool are_quarters_in_range = std::ranges::all_of(window_quarters, [](int quarter) {
    return quarter >= 0 && quarter <= 3;
  });
  if (!is_cell_active || !are_quarters_in_range) {
    ADD_FAILURE() << "NearbyFaces needs an active cell and window quarters of 0..3";
    return {};
  }
  const int size_level_count = frame_.size_level_count;
  const int size_level = ToSizeLevelNumber(cell);
  const UnitBox box = ToUnitBox(cell, size_level_count);

  // The cell and every cell touching it.
  std::set<std::uint64_t> near_cell_values = {cell.value};
  for (const CellId touching_cell : TouchingCells(cell)) {
    near_cell_values.insert(touching_cell.value);
  }
  const auto is_near_cell = [&near_cell_values](const std::optional<CellId>& side_cell) {
    return side_cell.has_value() && near_cell_values.contains(side_cell->value);
  };

  std::vector<GeometricFacePiece> candidates;
  for (const GeometricFacePiece& piece : face_pieces_) {
    if (piece.axis != axis) {
      continue;
    }
    const bool belongs_to_near_cell = is_near_cell(piece.cell_low) || is_near_cell(piece.cell_high);
    const bool is_size_level_near = std::abs(static_cast<int>(piece.size_level) - size_level) <= 1;
    if (!belongs_to_near_cell || !is_size_level_near) {
      continue;
    }
    // Method paper <sec-nearby-faces>: n_p - 5 <= n_f <= n_p + 6, n_f = 4 (centre - low corner) / w. Multiplied by
    // w, every term is a whole number of units: 4 (centre - low corner) = 2 (doubled centre - 2 low corner).
    const std::array<std::int64_t, 3> doubled_centre = ToDoubledCentre(piece, size_level_count);
    bool is_in_window = true;
    for (std::size_t index = 0; index < 3; ++index) {
      const std::int64_t four_offsets = 2 * (doubled_centre.at(index) - 2 * box.low.at(index));
      const std::int64_t window_low = (window_quarters.at(index) - 5) * box.width;
      const std::int64_t window_high = (window_quarters.at(index) + 6) * box.width;
      is_in_window = is_in_window && window_low <= four_offsets && four_offsets <= window_high;
    }
    if (is_in_window) {
      candidates.push_back(piece);
    }
  }
  return candidates;
}

std::vector<GeometricFacePiece> GeometricGrid::FacesWithin(FacePieceId face, double radius_in_cell_widths) const {
  const auto found = std::ranges::find_if(face_pieces_, [face](const GeometricFacePiece& piece) {
    return piece.id.value == face.value;
  });
  if (found == face_pieces_.end()) {
    ADD_FAILURE() << "FacesWithin needs the id of a face piece of this grid, not " << face.value;
    return {};
  }
  const int size_level_count = frame_.size_level_count;
  const std::array<std::int64_t, 3> face_centre = ToDoubledCentre(*found, size_level_count);
  // In half units, as the centres: twice the radius times the face's width in units.
  const auto face_width = static_cast<double>(ComputeWidthInUnits(found->size_level, size_level_count));
  const double doubled_radius = 2.0 * radius_in_cell_widths * face_width;

  std::vector<GeometricFacePiece> faces_within;
  for (const GeometricFacePiece& piece : face_pieces_) {
    if (piece.axis != found->axis) {
      continue;
    }
    const std::array<std::int64_t, 3> piece_centre = ToDoubledCentre(piece, size_level_count);
    bool is_within = true;
    for (std::size_t index = 0; index < 3; ++index) {
      const std::int64_t doubled_distance = std::abs(piece_centre.at(index) - face_centre.at(index));
      is_within = is_within && static_cast<double>(doubled_distance) < doubled_radius;
    }
    if (is_within) {
      faces_within.push_back(piece);
    }
  }
  return faces_within;
}

TestGrid::TestGrid(
    RunMemory run_memory,
    Buffer<CellId> cell_list,
    std::unique_ptr<GridBuilderBuffers> builder_buffers,
    StaggeredGrid grid
)
    : run_memory(std::move(run_memory)),
      cell_list(std::move(cell_list)),
      builder_buffers(std::move(builder_buffers)),
      grid(std::move(grid)) {
}

std::expected<TestGrid, GridFailure> BuildTestGrid(const GridInputs& inputs, const GridSettings& settings) {
  // Steps: 1 the room  2 the RunMemory  3 the list on the grid's processor  4 the builder buffers and the buffers
  // 5 the grid
  if (inputs.active_cell_buffer != nullptr) {
    ADD_FAILURE() << "BuildTestGrid takes its list as a span in CPU memory";
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }
  // Step 1.
  std::uint32_t max_cell_blocks = settings.max_cell_blocks;
  if (max_cell_blocks == 0) {
    const std::expected<GridCounts, GridFailure> counts = ComputeGridCounts(inputs);
    if (!counts.has_value()) {
      return std::unexpected(counts.error());
    }
    max_cell_blocks = std::max<std::uint32_t>(counts->cell_blocks, 1);
  }
  // A room above the limit is bad input for the builders; reserving memory for it first could fail another way.
  if (max_cell_blocks > kMaxCellBlocksLimit) {
    return std::unexpected(GridFailure{.error = GridError::kBadInput});
  }

  // Step 2: exactly the estimate of one grid; on the GPU, one memory piece more for the list.
  const RunOn run_on = settings.run_on;
  const std::vector<MemoryEstimateEntry> estimate = BuildGridMemoryEstimate(run_on, 1, max_cell_blocks);
  GridCounts room_counts{};
  room_counts.cell_blocks = max_cell_blocks;
  const MemoryCounts memory_counts = ToMemoryCounts(room_counts, 0);
  std::size_t reserved_bytes = ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kRunOn);
  if (run_on == RunOn::kCpu) {
    reserved_bytes += ComputeRunMemoryBytes(estimate, memory_counts, MemoryWhere::kCpu);
  } else {
    reserved_bytes += ComputeRunMemoryPieceBytes(inputs.active_cells.size_bytes());
  }
  std::expected<RunMemory, BufferError> run_memory = BuildRunMemory(run_on, reserved_bytes);
  if (!run_memory.has_value()) {
    return std::unexpected(ToGridFailure(run_memory.error(), reserved_bytes));
  }

  // Step 3: a GPU build reads its list from the GPU.
  Buffer<CellId> cell_list;
  GridInputs grid_inputs = inputs;
  if (run_on == RunOn::kGpu) {
    std::expected<Buffer<CellId>, BufferError> list =
        BuildBuffer<CellId>(run_on, inputs.active_cells.size(), *run_memory);
    if (!list.has_value()) {
      return std::unexpected(ToGridFailure(list.error(), inputs.active_cells.size_bytes()));
    }
    if (const std::expected<void, BufferError> written = WriteBuffer(inputs.active_cells, *list);
        !written.has_value()) {
      return std::unexpected(ToGridFailure(written.error(), inputs.active_cells.size_bytes()));
    }
    cell_list = std::move(*list);
    grid_inputs.active_cells = {};
    grid_inputs.active_cell_buffer = &cell_list;
    grid_inputs.active_cell_buffer_count = static_cast<std::uint32_t>(inputs.active_cells.size());
  }

  // Step 4: the builder buffers on the heap, so a move of the TestGrid never moves what the buffers borrow.
  std::expected<GridBuilderBuffers, GridFailure> built_builder_buffers =
      BuildGridBuilderBuffers(run_on, max_cell_blocks, *run_memory);
  if (!built_builder_buffers.has_value()) {
    return std::unexpected(built_builder_buffers.error());
  }
  auto builder_buffers = std::make_unique<GridBuilderBuffers>(std::move(*built_builder_buffers));
  std::expected<GridBuffers, GridFailure> buffers =
      BuildGridBuffers({.max_cell_blocks = max_cell_blocks, .run_on = run_on}, *builder_buffers, *run_memory);
  if (!buffers.has_value()) {
    return std::unexpected(buffers.error());
  }

  // Step 5. On failure the buffers are destroyed before the builder buffers they borrow, and both before the RunMemory.
  std::expected<StaggeredGrid, GridFailure> grid = BuildGrid(grid_inputs, *buffers);
  if (!grid.has_value()) {
    return std::unexpected(grid.error());
  }
  return TestGrid(std::move(*run_memory), std::move(cell_list), std::move(builder_buffers), std::move(*grid));
}

namespace {

// Says whether `count` items at two addresses have the same bytes.
template <typename Value>
bool ComputeIsSameBytes(const Value* first, const Value* second, std::size_t count) {
  if (count == 0) {
    return true;
  }
  if (first == nullptr || second == nullptr) {
    return false;
  }
  return std::memcmp(first, second, count * sizeof(Value)) == 0;
}

}  // namespace

GridBuffers TakeEmptyGridBuffers(std::expected<TestGrid, GridFailure>& owner) {
  EXPECT_TRUE(owner.has_value());
  return TakeBuffers(std::move(owner->grid));
}

std::tuple<std::uint32_t, std::array<std::uint64_t, kMaxSizeLevels>, std::uint64_t, std::uint64_t> ToGridCountFields(
    const GridCounts& counts
) {
  std::array<std::uint64_t, kMaxSizeLevels> per_size_level{};
  for (std::size_t size_level = 0; size_level < per_size_level.size(); ++size_level) {
    per_size_level.at(size_level) = counts.active_cells_per_size_level[size_level];
  }
  return {counts.cell_blocks, per_size_level, counts.active_cells, counts.face_pieces};
}

bool ComputeIsSameUsedTables(const GridTables& first, const GridTables& second) {
  if (first.version.value != second.version.value || first.block_count != second.block_count) {
    return false;
  }
  const std::size_t block_count = first.block_count;
  return ComputeIsSameBytes(first.block_ids, second.block_ids, block_count)
      && ComputeIsSameBytes(first.same_size_level_blocks, second.same_size_level_blocks, 27 * block_count)
      && ComputeIsSameBytes(first.bigger_block, second.bigger_block, block_count)
      && ComputeIsSameBytes(first.smaller_blocks, second.smaller_blocks, 8 * block_count)
      && ComputeIsSameBytes(first.cell_states, second.cell_states, kCellPlacesPerBlock * block_count)
      && ComputeIsSameBytes(first.face_piece_kinds, second.face_piece_kinds, kFacePlacesPerBlock * block_count);
}

bool ComputeIsSameGrid(const StaggeredGrid& first, const StaggeredGrid& second) {
  return ComputeIsSameUsedTables(first.tables(RunOn::kCpu), second.tables(RunOn::kCpu))
      && ToGridCountFields(first.counts()) == ToGridCountFields(second.counts());
}

std::vector<std::size_t> FindRandomGridsSpreadBySize(std::size_t group_size) {
  std::vector<std::size_t> ordered_grids(kRandomGridCount);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    ordered_grids.at(grid) = grid;
  }
  std::ranges::stable_sort(ordered_grids, [](std::size_t first, std::size_t second) {
    return BuildShuffledRandomGrids().at(first).size() > BuildShuffledRandomGrids().at(second).size();
  });
  const std::array<std::size_t, 3> group_starts = {
      0,
      (kRandomGridCount / 2) - (group_size / 2),
      kRandomGridCount - group_size,
  };
  std::vector<std::size_t> grids;
  grids.reserve(3 * group_size);
  for (const std::size_t group_start : group_starts) {
    for (std::size_t position = group_start; position < group_start + group_size; ++position) {
      grids.push_back(ordered_grids.at(position));
    }
  }
  return grids;
}

std::vector<TestGrid> BuildReferenceGrids(const std::vector<std::size_t>& grids) {
  const GridFrame frame = {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(kRandomGridSizeLevelCount),
  };
  std::vector<TestGrid> reference_grids;
  reference_grids.reserve(grids.size());
  for (const std::size_t grid : grids) {
    const GridInputs inputs = {
        .frame = frame,
        .active_cells = BuildShuffledRandomGrids().at(grid),
        .version = {.value = 1},
    };
    std::expected<TestGrid, GridFailure> reference_grid =
        BuildTestGrid(inputs, {.max_cell_blocks = kMaxRandomGridCellBlocks});
    EXPECT_TRUE(reference_grid.has_value()) << grid;
    if (reference_grid.has_value()) {
      reference_grids.push_back(std::move(*reference_grid));
    }
  }
  return reference_grids;
}

}  // namespace fluidity::grid
