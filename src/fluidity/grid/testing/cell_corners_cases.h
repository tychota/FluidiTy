// Test-only: the helpers that the unit tests and the functional tests of cell_corners share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    SideOfCell, ToUnitFrame, ToCell, ToInputs, ApplySmallCellsToList, ToTablesOf, EdgeValues, ToEdgeValues,
//          ComputeIsSameVec3, UnitPoint, ToWidthInUnits, ToCellCornerUnits, ToUnitPosition, ToCellCornerOfFaceCorner,
//          ToMovedCoord, ToCoordsAroundEdge, kAxisPairs, BuildSplitCellAcrossEdge, BuildBigCellOnLowSide, PlaceChecker,
//          CornerAndEdgeMistakes, UnitPointHash, GridUnderCheck, ApplyCellCornerChecks, ApplyFaceCornerChecks,
//          CheckerPlacesAround, ToCheckerPlacesAround, ToCheckerEdgeValues, ApplyEdgeChecks, FindCornerAndEdgeMistakes,
//          CheckNoMistake, CheckNoCornerOrEdgeMistake, NamedGrid, BuildCellsDownToSmallestAtOrigin,
//          BuildCornerStoredTwoSizeLevelsDown, BuildGridsAroundOrigin
// Uses:    the headers it includes
// Used by: src/fluidity/grid/cell_corners.test.cc and tests/grid/cell_corners_on_random_grids.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/scalar_math.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

// A frame with its origin at 0 and biggest cells 1 m wide: every corner position in it is exact.
constexpr GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

inline CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// The inputs of a CPU list of version 1.
inline GridInputs ToInputs(GridFrame frame, std::span<const CellId> active_cells) {
  return {.frame = frame, .active_cells = active_cells, .version = {.value = 1}};
}

// Appends the 8 small cells of `big_cell`, in cell eighth order.
inline void ApplySmallCellsToList(CellId big_cell, std::vector<CellId>& cells) {
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
}

inline GridTables ToTablesOf(const std::expected<TestGrid, GridFailure>& test_grid) {
  return test_grid->grid.tables(RunOn::kCpu);
}

// A result of FindCellsAroundEdge as plain numbers: its 4 places, then 1 when is_same_size and 0 otherwise.
using EdgeValues = std::array<std::uint32_t, 5>;

inline EdgeValues ToEdgeValues(const CellsAroundEdge& around) {
  return {
      around.cells[0].value,
      around.cells[1].value,
      around.cells[2].value,
      around.cells[3].value,
      around.is_same_size ? 1U : 0U,
  };
}

inline bool ComputeIsSameVec3(Vec3 first, Vec3 second) {
  return first.x == second.x && first.y == second.y && first.z == second.z;
}

// A point in whole units of the frame's last size level, (x, y, z).
using UnitPoint = std::array<std::int64_t, 3>;

// The width of a cell of `size_level` in units of the frame's last size level.
inline std::int64_t ToWidthInUnits(SizeLevel size_level, int size_level_count) {
  return static_cast<std::int64_t>(
      std::uint64_t{1} << static_cast<unsigned>(size_level_count - 1 - static_cast<int>(size_level))
  );
}

// Corner `corner_number` of a cell in units: bit 0 the high x side, bit 1 high y, bit 2 high z, by the test's own
// arithmetic on the id.
//
// clang-tidy: a corner number and a count of size levels; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline UnitPoint ToCellCornerUnits(CellId cell, int corner_number, int size_level_count) {
  const std::int64_t width = ToWidthInUnits(ToSizeLevel(cell), size_level_count);
  const CellCoord cell_coord = ToCellCoord(cell);
  const auto corner_bits = static_cast<unsigned>(corner_number);
  return {
      (cell_coord.x + std::int64_t{(corner_bits & 1U) != 0 ? 1 : 0}) * width,
      (cell_coord.y + std::int64_t{(corner_bits & 2U) != 0 ? 1 : 0}) * width,
      (cell_coord.z + std::int64_t{(corner_bits & 4U) != 0 ? 1 : 0}) * width,
  };
}

// The world position of a point in units, in a frame of origin 0 and a power-of-two biggest width: exact.
inline Vec3 ToUnitPosition(GridFrame frame, UnitPoint point) {
  const double unit_width = ComputeCellWidth(frame, static_cast<SizeLevel>(frame.size_level_count - 1));
  return {
      .x = frame.origin.x + (static_cast<double>(point.at(0)) * unit_width),
      .y = frame.origin.y + (static_cast<double>(point.at(1)) * unit_width),
      .z = frame.origin.z + (static_cast<double>(point.at(2)) * unit_width),
  };
}

// Which side of a cell along an axis: the low side or the high side.
enum class SideOfCell : std::uint8_t {
  kLow = 0,  // The side of the lower coordinate.
  kHigh = 1,  // The side of the higher coordinate.
};

// The corner of a cell that is face corner `face_corner_number` of its side along `axis`, on the high side of the
// cell when `side` is kHigh, on the low side when it is kLow: the axis bit set or clear, and the two bits across it
// from the face corner's bits.
inline int ToCellCornerOfFaceCorner(Axis axis, int face_corner_number, SideOfCell side) {
  const auto axis_number = static_cast<unsigned>(axis);
  const unsigned first_across_axis = axis_number == 0 ? 1 : 0;
  const unsigned second_across_axis = axis_number == 2 ? 1 : 2;
  const auto face_corner_bits = static_cast<unsigned>(face_corner_number);
  const unsigned cell_corner_bits = (static_cast<unsigned>(side) << axis_number)
      | ((face_corner_bits & 1U) << first_across_axis) | (((face_corner_bits >> 1U) & 1U) << second_across_axis);
  return static_cast<int>(cell_corner_bits);
}

// The place of `cell_coord` moved by -1 or +1 along two axes by the quarter, as the test computes it.
inline CellCoord ToMovedCoord(CellCoord cell_coord, Axis axis, int step) {
  return {
      .x = cell_coord.x + (axis == Axis::kX ? step : 0),
      .y = cell_coord.y + (axis == Axis::kY ? step : 0),
      .z = cell_coord.z + (axis == Axis::kZ ? step : 0),
  };
}

// The three places around an edge of a cell of size level 0: cells[1] across first_axis, cells[2] across second_axis,
// cells[3] across both, each to the side the quarter's bit names.
inline std::array<CellCoord, 4>
ToCoordsAroundEdge(CellCoord cell_coord, Axis first_axis, Axis second_axis, int quarter) {
  const int first_step = (static_cast<unsigned>(quarter) & 1U) != 0 ? 1 : -1;
  const int second_step = (static_cast<unsigned>(quarter) & 2U) != 0 ? 1 : -1;
  const CellCoord across_first = ToMovedCoord(cell_coord, first_axis, first_step);
  return {
      cell_coord,
      across_first,
      ToMovedCoord(cell_coord, second_axis, second_step),
      ToMovedCoord(across_first, second_axis, second_step),
  };
}

// The 6 ordered pairs of two different axes; the first 3 hold each pair once, the lower axis first.
constexpr std::array<std::array<Axis, 2>, 6> kAxisPairs = {
    std::array{Axis::kX, Axis::kY},
    std::array{Axis::kX, Axis::kZ},
    std::array{Axis::kY, Axis::kZ},
    std::array{Axis::kY, Axis::kX},
    std::array{Axis::kZ, Axis::kX},
    std::array{Axis::kZ, Axis::kY},
};

// The size-level-0 cells {0,0,0}, {1,0,0} and {0,1,0} and the 8 small cells of {1,1,0}, 2 size levels: only the place
// across both axes of the edge x = y = 1 m of {0,0,0} is split.
inline std::vector<CellId> BuildSplitCellAcrossEdge() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0), ToCell(0, 1, 0, 0), ToCell(0, 0, 1, 0)};
  ApplySmallCellsToList(ToCell(0, 1, 1, 0), cells);
  return cells;
}

// The size-level-0 cell {-1,0,0} beside the 8 small cells of {0,0,0}, 2 size levels: the size-level-1 places at x = -1
// lie in no size-level-1 block and inside the big cell.
inline std::vector<CellId> BuildBigCellOnLowSide() {
  std::vector<CellId> cells = {ToCell(0, -1, 0, 0)};
  ApplySmallCellsToList(ToCell(0, 0, 0, 0), cells);
  return cells;
}

// The checker's view of the cell places of every size level, from the list alone: the active cells, and the split
// cells, every bigger cell that holds an active cell.
class PlaceChecker {
 public:
  explicit PlaceChecker(const std::vector<CellId>& cells) {
    for (const CellId cell : cells) {
      active_ids_.insert(cell.value);
      CellId bigger = cell;
      while (static_cast<int>(ToSizeLevel(bigger)) > 0) {
        bigger = ComputeBigCell(bigger);
        split_ids_.insert(bigger.value);
      }
    }
  }

  // The state of place `cell_coord` of `size_level`. A coordinate out of range is absent.
  [[nodiscard]] CellState ToState(SizeLevel size_level, CellCoord cell_coord) const {
    if (!ComputeIsCoordInRange(cell_coord)) {
      return CellState::kAbsent;
    }
    const CellId cell = ToCellId(size_level, cell_coord);
    if (active_ids_.contains(cell.value)) {
      return CellState::kActive;
    }
    if (split_ids_.contains(cell.value)) {
      return CellState::kSplit;
    }
    CellId bigger = cell;
    while (static_cast<int>(ToSizeLevel(bigger)) > 0) {
      bigger = ComputeBigCell(bigger);
      if (active_ids_.contains(bigger.value)) {
        return CellState::kInsideBiggerCell;
      }
    }
    return CellState::kAbsent;
  }

 private:
  std::unordered_set<std::uint64_t> active_ids_;
  std::unordered_set<std::uint64_t> split_ids_;
};

// What one grid's readers got wrong, property by property: a count of mistakes each.
struct CornerAndEdgeMistakes {
  std::size_t corners_without_place = 0;  // A corner of an active cell or a face piece given kNoIndex.
  std::size_t places_with_two_points = 0;  // One corner place reached from two different points.
  std::size_t points_with_two_places = 0;  // One point given two corner places.
  std::size_t positions_not_the_points = 0;  // ComputeCornerPosition is not the point's exact position.
  std::size_t points_not_the_checkers = 0;  // 1 when the set of points differs from CornersInUnits.
  std::size_t positions_not_the_checkers = 0;  // 1 when the set of positions differs from Corners.
  std::size_t face_corners_not_the_cell_corners = 0;  // A face corner whose place is not that of the same cell corner.
  std::size_t edges_not_the_checkers = 0;  // A FindCellsAroundEdge result unlike the checker's.
  std::size_t edges_not_same_size = 0;  // Results with is_same_size false, to show both answers were met.
};

// A hash of a point for the checker's lookups: the three coordinates mixed by odd multipliers.
struct UnitPointHash {
  std::size_t operator()(const UnitPoint& point) const {
    const std::uint64_t mixed = (static_cast<std::uint64_t>(point[0]) * 0x9E3779B97F4A7C15U)
        ^ (static_cast<std::uint64_t>(point[1]) * 0xC2B2AE3D27D4EB4FU)
        ^ (static_cast<std::uint64_t>(point[2]) * 0x165667B19E3779F9U);
    return static_cast<std::size_t>(mixed ^ (mixed >> 29U));
  }
};

// One grid under check: its tables, the checker, its active cells, and the place and corner places of each active cell,
// found once.
struct GridUnderCheck {
  GridUnderCheck(const GridTables& grid_tables, const GeometricGrid& geometric_grid, const std::vector<CellId>& cells)
      : tables(grid_tables), checker(geometric_grid), active_cells(cells) {
    for (const CellId cell_id : active_cells) {
      const CellPlaceIndex cell = FindCellPlace(tables, cell_id);
      place_of_cell.emplace(cell_id.value, cell.value);
      corners_of_cell.emplace(cell_id.value, FindCornerPlaces(tables, cell));
    }
  }

  const GridTables& tables;
  const GeometricGrid& checker;
  const std::vector<CellId>& active_cells;
  std::unordered_map<std::uint64_t, std::uint32_t> place_of_cell;
  std::unordered_map<std::uint64_t, CellCorners> corners_of_cell;
};

// Checks the corners of every active cell, as places and as points by the test's own arithmetic: one point per place,
// one place per point, each place's position its point's, and the points and positions the checker's, by z, y, x.
inline void ApplyCellCornerChecks(const GridUnderCheck& grid, CornerAndEdgeMistakes& mistakes) {
  const int size_level_count = grid.tables.frame.size_level_count;
  std::unordered_map<std::uint32_t, UnitPoint> point_of_place;
  std::unordered_map<UnitPoint, std::uint32_t, UnitPointHash> place_of_point;
  for (const CellId cell_id : grid.active_cells) {
    const CellCorners& cell_corners = grid.corners_of_cell.at(cell_id.value);
    for (int corner_number = 0; corner_number < 8; ++corner_number) {
      const std::uint32_t place = cell_corners.corners[corner_number].value;
      if (place == kNoIndex) {
        ++mistakes.corners_without_place;
        continue;
      }
      const UnitPoint point = ToCellCornerUnits(cell_id, corner_number, size_level_count);
      const auto [place_entry, is_new_place] = point_of_place.try_emplace(place, point);
      mistakes.places_with_two_points += place_entry->second == point ? 0 : 1;
      const auto [point_entry, is_new_point] = place_of_point.try_emplace(point, place);
      mistakes.points_with_two_places += point_entry->second == place ? 0 : 1;
    }
  }
  std::vector<UnitPoint> points_by_z_y_x;
  std::vector<std::array<double, 3>> positions_by_z_y_x;
  for (const auto& [place, point] : point_of_place) {
    const Vec3 position = ComputeCornerPosition(grid.tables, CornerPlaceIndex{place});
    mistakes.positions_not_the_points += ComputeIsSameVec3(position, ToUnitPosition(grid.tables.frame, point)) ? 0 : 1;
    points_by_z_y_x.push_back({point[2], point[1], point[0]});
    positions_by_z_y_x.push_back({position.z, position.y, position.x});
  }
  std::ranges::sort(points_by_z_y_x);
  std::ranges::sort(positions_by_z_y_x);
  std::vector<UnitPoint> checker_points_by_z_y_x;
  for (const UnitPoint& point : grid.checker.CornersInUnits()) {
    checker_points_by_z_y_x.push_back({point[2], point[1], point[0]});
  }
  mistakes.points_not_the_checkers = points_by_z_y_x == checker_points_by_z_y_x ? 0 : 1;
  std::vector<std::array<double, 3>> checker_positions_by_z_y_x;
  for (const Vec3 position : grid.checker.Corners()) {
    checker_positions_by_z_y_x.push_back({position.z, position.y, position.x});
  }
  mistakes.positions_not_the_checkers = positions_by_z_y_x == checker_positions_by_z_y_x ? 0 : 1;
}

// Checks that the corners of every face piece are the corners of its smaller cell on that side, the cell of the face
// piece's size level: its cell_low, on that cell's high side, or its cell_high, on its low side.
inline void ApplyFaceCornerChecks(const GridUnderCheck& grid, CornerAndEdgeMistakes& mistakes) {
  for (const GeometricFacePiece& face_piece : grid.checker.FacePieces()) {
    const FaceCorners face_corners = FindCornerPlaces(grid.tables, FindFacePlace(grid.tables, face_piece.id));
    const bool is_cell_low_smaller =
        face_piece.cell_low.has_value() && ToSizeLevel(*face_piece.cell_low) == face_piece.size_level;
    const CellId smaller_cell = is_cell_low_smaller ? *face_piece.cell_low : face_piece.cell_high.value_or(CellId{0});
    const auto smaller_cell_corners = grid.corners_of_cell.find(smaller_cell.value);
    if (smaller_cell_corners == grid.corners_of_cell.end()) {
      mistakes.face_corners_not_the_cell_corners += 4;
      continue;
    }
    for (int corner_number = 0; corner_number < 4; ++corner_number) {
      const int cell_corner_number = ToCellCornerOfFaceCorner(
          face_piece.axis,
          corner_number,
          is_cell_low_smaller ? SideOfCell::kHigh : SideOfCell::kLow
      );
      const bool is_cell_corner =
          smaller_cell_corners->second.corners[cell_corner_number].value == face_corners.corners[corner_number].value;
      mistakes.face_corners_not_the_cell_corners += is_cell_corner ? 0 : 1;
    }
  }
}

// The checker's state and place of the 27 places of a cell's size level around it and at it, by offset (z, y, x) of
// -1..1: entry ((dz + 1) 3 + (dy + 1)) 3 + (dx + 1). A place is kNoIndex unless active.
struct CheckerPlacesAround {
  std::array<CellState, 27> states{};
  std::array<std::uint32_t, 27> places{};
};

inline CheckerPlacesAround
ToCheckerPlacesAround(const GridUnderCheck& grid, const PlaceChecker& place_checker, CellId cell_id) {
  CheckerPlacesAround around;
  const SizeLevel size_level = ToSizeLevel(cell_id);
  const CellCoord cell_coord = ToCellCoord(cell_id);
  for (std::size_t entry = 0; entry < 27; ++entry) {
    const CellCoord around_coord = {
        .x = cell_coord.x + static_cast<std::int32_t>(entry % 3) - 1,
        .y = cell_coord.y + static_cast<std::int32_t>((entry / 3) % 3) - 1,
        .z = cell_coord.z + static_cast<std::int32_t>(entry / 9) - 1,
    };
    around.states.at(entry) = place_checker.ToState(size_level, around_coord);
    around.places.at(entry) = around.states.at(entry) == CellState::kActive
        ? grid.place_of_cell.at(ToCellId(size_level, around_coord).value)
        : kNoIndex;
  }
  return around;
}

// The checker's result of FindCellsAroundEdge as plain numbers, as ToEdgeValues gives a result, for the places at
// `coords` around the cell at cell_coord.
inline EdgeValues ToCheckerEdgeValues(
    const CheckerPlacesAround& places_around,
    CellCoord cell_coord,
    const std::array<CellCoord, 4>& coords
) {
  EdgeValues edge_values = {};
  bool is_same_size = true;
  for (std::size_t around_number = 0; around_number < coords.size(); ++around_number) {
    const CellCoord& around_coord = coords.at(around_number);
    const int entry_number = ((around_coord.z - cell_coord.z + 1) * 9) + ((around_coord.y - cell_coord.y + 1) * 3)
        + (around_coord.x - cell_coord.x + 1);
    const auto entry = static_cast<std::size_t>(entry_number);
    const CellState state = places_around.states.at(entry);
    edge_values.at(around_number) = places_around.places.at(entry);
    is_same_size = is_same_size && (state == CellState::kActive || state == CellState::kAbsent);
  }
  edge_values.at(4) = is_same_size ? 1 : 0;
  return edge_values;
}

// Checks the cells around every edge of every active cell, for each pair of axes given and every quarter, against the
// checker's states of the places around the cell.
inline void ApplyEdgeChecks(
    const GridUnderCheck& grid,
    std::span<const std::array<Axis, 2>> axis_pairs,
    CornerAndEdgeMistakes& mistakes
) {
  const PlaceChecker place_checker(grid.active_cells);
  for (const CellId cell_id : grid.active_cells) {
    const CellPlaceIndex cell = {.value = grid.place_of_cell.at(cell_id.value)};
    const CellCoord cell_coord = ToCellCoord(cell_id);
    const CheckerPlacesAround places_around = ToCheckerPlacesAround(grid, place_checker, cell_id);
    for (const std::array<Axis, 2>& axes : axis_pairs) {
      for (int quarter = 0; quarter < 4; ++quarter) {
        const CellsAroundEdge around = FindCellsAroundEdge(grid.tables, cell, axes.at(0), axes.at(1), quarter);
        const std::array<CellCoord, 4> coords = ToCoordsAroundEdge(cell_coord, axes.at(0), axes.at(1), quarter);
        const bool is_checkers = ToEdgeValues(around) == ToCheckerEdgeValues(places_around, cell_coord, coords);
        mistakes.edges_not_the_checkers += is_checkers ? 0 : 1;
        mistakes.edges_not_same_size += around.is_same_size ? 0 : 1;
      }
    }
  }
}

// Checks FindCornerPlaces of every active cell and every face piece, ComputeCornerPosition of every corner place and
// FindCellsAroundEdge of every edge of every active cell, for the given pairs of axes, against the checker.
inline CornerAndEdgeMistakes FindCornerAndEdgeMistakes(
    const GridTables& tables,
    const GeometricGrid& checker,
    const std::vector<CellId>& cells,
    std::span<const std::array<Axis, 2>> axis_pairs
) {
  CornerAndEdgeMistakes mistakes;
  const GridUnderCheck grid(tables, checker, cells);
  ApplyCellCornerChecks(grid, mistakes);
  ApplyFaceCornerChecks(grid, mistakes);
  ApplyEdgeChecks(grid, axis_pairs, mistakes);
  return mistakes;
}

// Checks, as a test expectation, that `mistake_count` is 0, under a trace that names the property.
inline void CheckNoMistake(const char* property, std::size_t mistake_count) {
  SCOPED_TRACE(property);
  EXPECT_EQ(mistake_count, 0U);
}

// Checks, as test expectations, that every count of `mistakes` but the one kept for its spread is 0.
inline void CheckNoCornerOrEdgeMistake(const CornerAndEdgeMistakes& mistakes) {
  CheckNoMistake("FindCornerPlaces: every cell corner has a place", mistakes.corners_without_place);
  CheckNoMistake("EveryPhysicalCorner_HasOnePlace: one point per place", mistakes.places_with_two_points);
  CheckNoMistake("EveryPhysicalCorner_HasOnePlace: one place per point", mistakes.points_with_two_places);
  CheckNoMistake("ComputeCornerPosition: the point's exact position", mistakes.positions_not_the_points);
  CheckNoMistake("FindCornerPlaces: the points are CornersInUnits", mistakes.points_not_the_checkers);
  CheckNoMistake("ComputeCornerPosition: the positions are Corners", mistakes.positions_not_the_checkers);
  CheckNoMistake("FaceCorners_AreTheCellCornersOnThatSide", mistakes.face_corners_not_the_cell_corners);
  CheckNoMistake("FindCellsAroundEdge: the checker's places and flag", mistakes.edges_not_the_checkers);
}

// A hand-made grid for the readers: its name, its frame's size level count and its active cells.
struct NamedGrid {
  const char* name;
  int size_level_count;
  std::vector<CellId> cells;
};

// The size-level-0 cell {0,0,0} split down to size level 15 at its low corner: at each size level 1..14 the 7 cells
// other than the low one, and the 8 size-level-15 cells of {0,0,0} at size level 14. 106 cells; the size-level-15 block
// (0,0,0) exists.
inline std::vector<CellId> BuildCellsDownToSmallestAtOrigin() {
  std::vector<CellId> cells;
  for (int size_level = 1; size_level < kMaxSizeLevels - 1; ++size_level) {
    for (int cell_eighth = 1; cell_eighth < 8; ++cell_eighth) {
      cells.push_back(ComputeSmallCell(ToCell(size_level - 1, 0, 0, 0), cell_eighth));
    }
  }
  ApplySmallCellsToList(ToCell(kMaxSizeLevels - 2, 0, 0, 0), cells);
  return cells;
}

// A size-level-0 cell {0,0,0} whose corner at {1,0,0} m is stored two size levels down, 3 size levels: the size-level-1
// cells of {1,0,0} but {3,1,1}, and the 8 size-level-2 cells of {3,1,1}. Those touch no size-level-0 cell, and their
// block, the size-level-2 block (1,0,0), is the box of the place {1,0,0}, whose low corner is the big cell's corner.
inline std::vector<CellId> BuildCornerStoredTwoSizeLevelsDown() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 0; cell_eighth < 7; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(ToCell(0, 1, 0, 0), cell_eighth));
  }
  ApplySmallCellsToList(ToCell(1, 3, 1, 1), cells);
  return cells;
}

// Grids the random grids never reach around the origin: negative coordinates, cells at block corners, and corners
// stored one and two size levels below a cell.
inline std::vector<NamedGrid> BuildGridsAroundOrigin() {
  std::vector<NamedGrid> grids;
  std::vector<CellId> negative_two_size_cells = {ToCell(0, -1, -1, -1)};
  ApplySmallCellsToList(ToCell(0, 0, -1, -1), negative_two_size_cells);
  grids.push_back({.name = "two-size fixture at -1", .size_level_count = 2, .cells = negative_two_size_cells});
  // The 8 size-level-0 cells around the point (0, 0, 0), one per block, two of them split: 8 blocks of each size
  // level hold the point.
  std::vector<CellId> block_corner_cells;
  for (std::int32_t z = -1; z <= 0; ++z) {
    for (std::int32_t y = -1; y <= 0; ++y) {
      for (std::int32_t x = -1; x <= 0; ++x) {
        const bool is_split = (x == 0 && y == 0 && z == 0) || (x == -1 && y == 0 && z == -1);
        if (is_split) {
          ApplySmallCellsToList(ToCell(0, x, y, z), block_corner_cells);
        } else {
          block_corner_cells.push_back(ToCell(0, x, y, z));
        }
      }
    }
  }
  grids.push_back({.name = "eight blocks around a point", .size_level_count = 2, .cells = block_corner_cells});
  grids.push_back({.name = "split cell across an edge", .size_level_count = 2, .cells = BuildSplitCellAcrossEdge()});
  grids.push_back({.name = "big cell on the low side", .size_level_count = 2, .cells = BuildBigCellOnLowSide()});
  grids.push_back({
      .name = "corner stored two size levels down",
      .size_level_count = 3,
      .cells = BuildCornerStoredTwoSizeLevelsDown(),
  });
  grids.push_back({
      .name = "smallest cells at the origin, 16 size levels",
      .size_level_count = kMaxSizeLevels,
      .cells = BuildCellsDownToSmallestAtOrigin(),
  });
  return grids;
}

}  // namespace fluidity::grid
