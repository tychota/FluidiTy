#include "fluidity/grid/grid_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <tuple>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/test_support.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"

namespace fluidity::grid {
namespace {

// A frame with its origin at 0 and biggest cells 1 m wide.
GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// Appends the 8 small cells of big_cell to cells.
void AppendSmallCells(CellId big_cell, std::vector<CellId>& cells) {
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(big_cell, cell_eighth));
  }
}

// The size-level-0 cell {0,0,0} beside the 8 size-level-1 cells of size-level-0 cell {1,0,0}, in a frame of 2 size
// levels.
std::vector<CellId> BuildSplitCellBesideBigCell() {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  AppendSmallCells(ToCell(0, 1, 0, 0), cells);
  return cells;
}

// The value of a cell, or 0 for none: CellId{0} is never a valid cell.
std::uint64_t ToValue(const std::optional<CellId>& cell) {
  return cell.value_or(CellId{0}).value;
}

std::vector<std::uint64_t> ToValues(const std::vector<CellId>& cells) {
  std::vector<std::uint64_t> values;
  values.reserve(cells.size());
  for (const CellId cell : cells) {
    values.push_back(cell.value);
  }
  return values;
}

int CountFacePieces(const std::vector<GeometricFacePiece>& pieces, Axis axis) {
  return static_cast<int>(std::ranges::count_if(pieces, [axis](const GeometricFacePiece& piece) {
    return piece.axis == axis;
  }));
}

constexpr std::array<Axis, 3> kAxes = {Axis::kX, Axis::kY, Axis::kZ};

// ---------------------------------------------------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------------------------------------------------

TEST(ToUniformIndex, AnyRaw_IsTheRemainder) {
  EXPECT_EQ(ToUniformIndex(0, 7), 0U);
  EXPECT_EQ(ToUniformIndex(20, 7), 6U);
  EXPECT_EQ(ToUniformIndex(std::numeric_limits<std::uint64_t>::max(), 10), 5U);
}

TEST(ToUnitDouble, TopFiftyThreeBits_AreScaledByTwoToMinusFiftyThree) {
  EXPECT_EQ(ToUnitDouble(0), 0.0);
  // The low 11 bits are dropped: 2^11 is the smallest raw value above 0 that gives a nonzero double.
  EXPECT_EQ(ToUnitDouble(2047), 0.0);
  EXPECT_EQ(ToUnitDouble(2048), std::ldexp(1.0, -53));
  EXPECT_EQ(ToUnitDouble(std::uint64_t{1} << 63U), 0.5);
  EXPECT_EQ(ToUnitDouble(std::numeric_limits<std::uint64_t>::max()), 1.0 - std::ldexp(1.0, -53));
}

// The C++ standard fixes this value ([rand.predef]); a library that gives another one would change every random grid.
TEST(StandardGenerator, TenThousandthOutput_IsTheValueTheStandardFixes) {
  std::mt19937_64 generator;  // NOLINT(bugprone-random-generator-seed): the default seed, 5489, is the point.
  generator.discard(9999);
  EXPECT_EQ(generator(), 9981545732273789042ULL);
}

// ---------------------------------------------------------------------------------------------------------------------
// Face pieces
// ---------------------------------------------------------------------------------------------------------------------

TEST(ToUnitBox, NegativeCoord_ScalesToUnitsOfTheLastSizeLevel) {
  const UnitBox box = ToUnitBox(ToCell(1, -1, 2, 0), 3);

  EXPECT_EQ(box.low, (std::array<std::int64_t, 3>{-2, 4, 0}));
  EXPECT_EQ(box.width, 2);
}

TEST(GeometricGrid, OneBigCell_HasSixFacePieces) {
  const GeometricGrid grid(ToUnitFrame(1), {ToCell(0, 0, 0, 0)});

  const std::vector<GeometricFacePiece>& pieces = grid.FacePieces();

  // Each piece is a whole side of 1 m2 at the grid's edge: exactly one side has the cell.
  const auto is_whole_edge_side = [](const GeometricFacePiece& piece) {
    const bool is_at_edge = piece.cell_low.has_value() != piece.cell_high.has_value();
    return piece.area == 1.0 && static_cast<int>(piece.size_level) == 0 && is_at_edge;
  };
  EXPECT_EQ(pieces.size(), 6U);
  EXPECT_EQ(std::ranges::count_if(pieces, is_whole_edge_side), 6);
  EXPECT_EQ(CountFacePieces(pieces, Axis::kX), 2);
  EXPECT_EQ(CountFacePieces(pieces, Axis::kY), 2);
  EXPECT_EQ(CountFacePieces(pieces, Axis::kZ), 2);
}

TEST(GeometricGrid, SplitCellBesideBigCell_HasFourQuarterFacePieces) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());
  const CellId big_cell = ToCell(0, 0, 0, 0);

  const std::vector<GeometricFacePiece>& pieces = grid.FacePieces();

  // Big cell: 5 whole sides and 4 quarters on +X. Small cells: 12 sides between them and 20 outer sides.
  EXPECT_EQ(pieces.size(), 41U);
  const auto is_beside_big_cell = [big_cell](const GeometricFacePiece& piece) {
    return ToValue(piece.cell_low) == big_cell.value && piece.cell_high.has_value();
  };
  // A quarter: an X piece of size level 1 and 0.25 m2 whose high place and high cell are at x = 2.
  const auto is_quarter = [](const GeometricFacePiece& piece) {
    const bool is_small_x_piece = piece.axis == Axis::kX && static_cast<int>(piece.size_level) == 1;
    const bool is_at_x_two = piece.cell_high_coord.x == 2 && ToCellCoord(piece.cell_high.value_or(CellId{0})).x == 2;
    return is_small_x_piece && piece.area == 0.25 && is_at_x_two;
  };
  EXPECT_EQ(std::ranges::count_if(pieces, is_beside_big_cell), 4);
  const auto is_quarter_beside_big_cell = [&](const GeometricFacePiece& piece) {
    return is_beside_big_cell(piece) && is_quarter(piece);
  };
  EXPECT_EQ(std::ranges::count_if(pieces, is_quarter_beside_big_cell), 4);
}

TEST(GeometricGrid, SplitCellBesideBigCell_FacePiecesPerAxisMatchHandCount) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());

  const std::vector<GeometricFacePiece>& pieces = grid.FacePieces();

  // X: big -X, 4 quarters, 4 between the small cells, 4 small +X sides. Y and Z: 2 big sides and 3 planes of 4.
  EXPECT_EQ(CountFacePieces(pieces, Axis::kX), 13);
  EXPECT_EQ(CountFacePieces(pieces, Axis::kY), 14);
  EXPECT_EQ(CountFacePieces(pieces, Axis::kZ), 14);
  double area_sum = 0;
  for (const GeometricFacePiece& piece : pieces) {
    area_sum += piece.area;
  }
  // 5 whole sides of 1 m2 and 36 small sides of 0.25 m2.
  EXPECT_EQ(area_sum, 14.0);
}

TEST(GeometricGrid, SplitCellBesideBigCell_FacePieceIdNamesTheHighPlace) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());

  for (const GeometricFacePiece& piece : grid.FacePieces()) {
    EXPECT_EQ(piece.id.value, ToFacePieceId(piece.axis, piece.size_level, piece.cell_high_coord).value);
  }
  // The +X side of size-level-1 cell {3,1,1} is at x = 2 m: the place {4,1,1} of size level 1.
  const FacePieceId outer_side = ToFacePieceId(Axis::kX, SizeLevel{1}, {.x = 4, .y = 1, .z = 1});
  const auto found = std::ranges::find_if(grid.FacePieces(), [outer_side](const GeometricFacePiece& piece) {
    return piece.id.value == outer_side.value;
  });
  ASSERT_NE(found, grid.FacePieces().end());
  ASSERT_TRUE(found->cell_low.has_value());
  EXPECT_EQ(ToValue(found->cell_low), ToCell(1, 3, 1, 1).value);
  EXPECT_FALSE(found->cell_high.has_value());
}

TEST(GeometricGrid, CellsAtBothCoordLimits_Has48FacePieces) {
  std::vector<CellId> cells = {ToCell(0, -kMaxCoord, 0, 0), ToCell(0, kMaxCoord, 0, 0)};
  AppendSmallCells(ToCell(0, 0, 0, 0), cells);
  const GeometricGrid grid(ToUnitFrame(2), cells);

  // 6 sides for each lone cell and 36 for the 8 small cells, which touch nothing else.
  EXPECT_EQ(grid.FacePieces().size(), 48U);
  const FacePieceId last_high_side = ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = kMaxCoord + 1, .y = 0, .z = 0});
  const bool has_last_high_side = std::ranges::any_of(grid.FacePieces(), [last_high_side](const auto& piece) {
    return piece.id.value == last_high_side.value;
  });
  EXPECT_TRUE(has_last_high_side);
}

// ---------------------------------------------------------------------------------------------------------------------
// Touching cells, corners, containing cell
// ---------------------------------------------------------------------------------------------------------------------

TEST(GeometricGrid, SplitCellBesideBigCell_TouchingCellsMatchHandCount) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());

  // The big cell touches the 4 small cells at x = 2; a small cell at x = 2 touches its 7 siblings and the big cell;
  // one at x = 3 touches its 7 siblings only.
  EXPECT_EQ(grid.TouchingCells(ToCell(0, 0, 0, 0)).size(), 4U);
  EXPECT_EQ(grid.TouchingCells(ToCell(1, 2, 0, 0)).size(), 8U);
  EXPECT_EQ(grid.TouchingCells(ToCell(1, 3, 1, 1)).size(), 7U);
}

TEST(GeometricGrid, TouchingCells_AreInFixedOrder) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());

  const std::vector<CellId> touching = grid.TouchingCells(ToCell(1, 2, 0, 0));

  for (std::size_t index = 1; index < touching.size(); ++index) {
    EXPECT_TRUE(ComesBefore(touching.at(index - 1), touching.at(index))) << index;
  }
}

TEST(GeometricGrid, AllNeighboursSplit_Touches56Cells) {
  std::vector<CellId> cells = {ToCell(0, 1, 1, 1)};
  for (std::int32_t z = 0; z < 3; ++z) {
    for (std::int32_t y = 0; y < 3; ++y) {
      for (std::int32_t x = 0; x < 3; ++x) {
        if (x != 1 || y != 1 || z != 1) {
          AppendSmallCells(ToCell(0, x, y, z), cells);
        }
      }
    }
  }
  const GeometricGrid grid(ToUnitFrame(2), cells);

  // 6 sides of 4 small cells, 12 edges of 2, 8 corners of 1.
  EXPECT_EQ(grid.TouchingCells(ToCell(0, 1, 1, 1)).size(), 56U);
}

TEST(GeometricGrid, SplitCellBesideBigCell_Has31Corners) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());

  const std::vector<Vec3> corners = grid.Corners();

  // 8 corners of the big cell and the 3 x 3 x 3 corners of the small cells share the 4 corners at x = 1 m.
  EXPECT_EQ(corners.size(), 31U);
  std::set<std::tuple<double, double, double>> distinct_corners;
  for (const Vec3 corner : corners) {
    distinct_corners.insert({corner.x, corner.y, corner.z});
  }
  EXPECT_EQ(distinct_corners.size(), 31U);
  EXPECT_TRUE(distinct_corners.contains({1.5, 0.5, 0.5}));
  EXPECT_TRUE(distinct_corners.contains({0.0, 1.0, 1.0}));
  EXPECT_FALSE(distinct_corners.contains({0.5, 0.5, 0.0}));
}

TEST(GeometricGrid, SplitCellBesideBigCell_Has31CornersInUnits) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());

  const std::vector<std::array<std::int64_t, 3>> corners = grid.CornersInUnits();

  // Units of 0.5 m: the big cell's corners at x 0 and 2, the small cells' at x 2, 3 and 4, sharing the 4 at x = 2.
  EXPECT_EQ(corners.size(), 31U);
  EXPECT_EQ(corners.front(), (std::array<std::int64_t, 3>{0, 0, 0}));
  EXPECT_EQ(corners.back(), (std::array<std::int64_t, 3>{4, 2, 2}));
  EXPECT_EQ(std::ranges::count(corners, std::array<std::int64_t, 3>{3, 1, 1}), 1);
  EXPECT_EQ(std::ranges::count(corners, std::array<std::int64_t, 3>{1, 1, 0}), 0);
}

TEST(GeometricGrid, ContainingCell_OwnsItsLowSidesOnly) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());

  const std::optional<CellId> centre = grid.ContainingCell({.x = 0.5, .y = 0.5, .z = 0.5});
  const std::optional<CellId> low_corner = grid.ContainingCell({.x = 0.0, .y = 0.0, .z = 0.0});
  const std::optional<CellId> big_high_side = grid.ContainingCell({.x = 1.0, .y = 0.0, .z = 0.0});
  const std::optional<CellId> small_cell = grid.ContainingCell({.x = 1.75, .y = 0.75, .z = 0.25});
  const std::optional<CellId> beyond_high_side = grid.ContainingCell({.x = 2.0, .y = 0.0, .z = 0.0});
  const std::optional<CellId> below_low_side = grid.ContainingCell({.x = -0.0625, .y = 0.0, .z = 0.0});

  ASSERT_TRUE(centre.has_value());
  EXPECT_EQ(ToValue(centre), ToCell(0, 0, 0, 0).value);
  ASSERT_TRUE(low_corner.has_value());
  EXPECT_EQ(ToValue(low_corner), ToCell(0, 0, 0, 0).value);
  ASSERT_TRUE(big_high_side.has_value());
  EXPECT_EQ(ToValue(big_high_side), ToCell(1, 2, 0, 0).value);
  ASSERT_TRUE(small_cell.has_value());
  EXPECT_EQ(ToValue(small_cell), ToCell(1, 3, 1, 0).value);
  EXPECT_FALSE(beyond_high_side.has_value());
  EXPECT_FALSE(below_low_side.has_value());
}

TEST(GeometricGrid, PositionNotFiniteOrBeyondIdRange_IsInNoCell) {
  const GeometricGrid grid(ToUnitFrame(1), {ToCell(0, kMaxCoord, 0, 0), ToCell(0, -kMaxCoord, 0, 0)});

  EXPECT_FALSE(grid.ContainingCell({.x = std::nan(""), .y = 0, .z = 0}).has_value());
  EXPECT_FALSE(grid.ContainingCell({.x = std::numeric_limits<double>::infinity(), .y = 0, .z = 0}).has_value());
  // Size-level-0 coordinate kMaxCoord + 1 and -kMaxCoord - 1: no size-level-0 id can name them.
  EXPECT_FALSE(grid.ContainingCell({.x = 262144.0, .y = 0, .z = 0}).has_value());
  EXPECT_FALSE(grid.ContainingCell({.x = -262143.5, .y = 0, .z = 0}).has_value());
  const std::optional<CellId> last_cell = grid.ContainingCell({.x = 262143.5, .y = 0, .z = 0});
  ASSERT_TRUE(last_cell.has_value());
  EXPECT_EQ(ToValue(last_cell), ToCell(0, kMaxCoord, 0, 0).value);
  const std::optional<CellId> first_cell = grid.ContainingCell({.x = -262143.0, .y = 0, .z = 0});
  ASSERT_TRUE(first_cell.has_value());
  EXPECT_EQ(ToValue(first_cell), ToCell(0, -kMaxCoord, 0, 0).value);
}

// ---------------------------------------------------------------------------------------------------------------------
// Nearby faces and faces within a radius
// ---------------------------------------------------------------------------------------------------------------------

TEST(GeometricGrid, BigCellBesideSplitCell_NearbyFacesMatchHandCount) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());
  const CellId big_cell = ToCell(0, 0, 0, 0);

  // X: its -X side, the 4 quarters at x = 1 m, the 4 pieces at x = 1.5 m between small cells it touches. The 4
  // outer sides at x = 2 m belong to small cells it does not touch, so window 3 does not add them.
  EXPECT_EQ(grid.NearbyFaces(big_cell, {0, 0, 0}, Axis::kX).size(), 9U);
  EXPECT_EQ(grid.NearbyFaces(big_cell, {3, 0, 0}, Axis::kX).size(), 9U);
  // Y: its 2 sides and 6 pieces of the 4 small cells it touches (2 low sides, 2 between, 2 high sides).
  EXPECT_EQ(grid.NearbyFaces(big_cell, {0, 0, 0}, Axis::kY).size(), 8U);
}

TEST(GeometricGrid, SmallCellBesideBigCell_NearbyFacesMatchHandCount) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());
  const CellId small_cell = ToCell(1, 2, 0, 0);

  // Window 0 on X reaches x = 0.375..1.75 m: the planes x = 1 and 1.5 m, 4 pieces each.
  EXPECT_EQ(grid.NearbyFaces(small_cell, {0, 0, 0}, Axis::kX).size(), 8U);
  // Window 3 reaches x = 0.75..2.125 m: the planes x = 1, 1.5 and 2 m; the big cell's -X side at 0 m stays out.
  EXPECT_EQ(grid.NearbyFaces(small_cell, {3, 3, 3}, Axis::kX).size(), 12U);
}

// Three size levels, width 1 m: the size-level-0 cell {0,0,0}; the size-level-0 cell {1,0,0} split into its 8
// size-level-1 cells, of which {3,0,0} is split again into 8 size-level-2 cells.
TEST(GeometricGrid, PieceTwoSizeLevelsSmaller_IsNotANearbyFace) {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    const CellId small_cell = ComputeSmallCell(ToCell(0, 1, 0, 0), cell_eighth);
    if (small_cell.value == ToCell(1, 3, 0, 0).value) {
      AppendSmallCells(small_cell, cells);
    } else {
      cells.push_back(small_cell);
    }
  }
  ASSERT_EQ(BuildSizeStepClosure(cells, 3).size(), cells.size());
  const GeometricGrid grid(ToUnitFrame(3), cells);
  const CellId big_cell = ToCell(0, 0, 0, 0);

  // Hand count, X axis, the window reaching x = -1.25..1.5 m (n_p 0) or -0.5..2.25 m (n_p 3): the big cell's -X side
  // at 0 m, the 4 quarters at 1 m, and at 1.5 m the 3 size-level-1 pieces between {2,y,z} and {3,y,z}, (y, z) not
  // (0, 0): 8. The 4 size-level-2 pieces at 1.5 m lie in the window and belong to {2,0,0}, which touches the big cell,
  // but they are two size levels smaller than it: without that rule the count would be 12.
  EXPECT_EQ(grid.NearbyFaces(big_cell, {0, 0, 0}, Axis::kX).size(), 8U);
  EXPECT_EQ(grid.NearbyFaces(big_cell, {3, 0, 0}, Axis::kX).size(), 8U);
}

// Two size levels, width 1 m: the size-level-0 cell {0,0,0}, and the size-level-0 cell {0,1,0} above it in y split
// into its 8 size-level-1 cells. The cell is the size-level-1 cell {0,2,0}: low corner {0, 1, 0} m, quarter 0.125 m.
TEST(GeometricGrid, MixedSizes_WindowIsOpenBelowAndClosedAbove) {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  AppendSmallCells(ToCell(0, 0, 1, 0), cells);
  const GeometricGrid grid(ToUnitFrame(2), cells);

  // Window quarters (2, 2, 0): n_f from -3 to 8 along x and y, from -5 to 6 along z. Hand count of the X pieces:
  // - the big cell's -X side, centre {0, 0.5, 0.5} m, is at n_f (0, -4, 4): -4 is n_p - 6 on y, so it is out;
  // - its +X side, centre {1, 0.5, 0.5} m, is at n_f (8, -4, 4): out for the same reason;
  // - the 12 pieces of the small cells on the planes x = 0, 0.5 and 1 m (n_f 0, 4 and 8; 8 is n_p + 6, so in), with
  //   y and z centres at n_f 2 or 6: all in, all of the cell or of a sibling touching it.
  // So 12. A window closed below would add the big cell's two sides (14). One open above would drop the plane x = 1 m
  // and the z centres at n_f 6, both n_p + 6, leaving 2 planes x 2 y x 1 z (4).
  EXPECT_EQ(grid.NearbyFaces(ToCell(1, 0, 2, 0), {2, 2, 0}, Axis::kX).size(), 12U);
}

TEST(GeometricGrid, UniformGrid_Has27NearbyFacesInEveryWindow) {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 5; ++z) {
    for (std::int32_t y = 0; y < 5; ++y) {
      for (std::int32_t x = 0; x < 5; ++x) {
        cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  const GeometricGrid grid(ToUnitFrame(1), cells);
  const CellId centre_cell = ToCell(0, 2, 2, 2);

  for (const Axis axis : kAxes) {
    for (int window = 0; window < 64; ++window) {
      const std::array<int, 3> window_quarters = {window % 4, (window / 4) % 4, window / 16};
      EXPECT_EQ(grid.NearbyFaces(centre_cell, window_quarters, axis).size(), 27U) << window;
    }
  }
}

TEST(GeometricGrid, AllNeighboursSplit_Has76NearbyFacesInEveryWindow) {
  std::vector<CellId> cells = {ToCell(0, 1, 1, 1)};
  for (std::int32_t z = 0; z < 3; ++z) {
    for (std::int32_t y = 0; y < 3; ++y) {
      for (std::int32_t x = 0; x < 3; ++x) {
        if (x != 1 || y != 1 || z != 1) {
          AppendSmallCells(ToCell(0, x, y, z), cells);
        }
      }
    }
  }
  const GeometricGrid grid(ToUnitFrame(2), cells);
  const CellId centre_cell = ToCell(0, 1, 1, 1);

  for (const Axis axis : kAxes) {
    for (int window = 0; window < 64; ++window) {
      const std::array<int, 3> window_quarters = {window % 4, (window / 4) % 4, window / 16};
      EXPECT_EQ(grid.NearbyFaces(centre_cell, window_quarters, axis).size(), 76U) << window;
    }
  }
}

// Says whether the size-level-1 place at `offset` from cell {2,2,2} stays whole in the two-size case: the cell itself,
// its 8 corner neighbours, its 2 edge neighbours at y - 1 and x +-1, and its 4 edge neighbours at x 0, y +-1, z +-1.
bool ComputeIsKeptWhole(CellCoord offset) {
  const bool is_cell = offset.x == 0 && offset.y == 0 && offset.z == 0;
  const bool is_corner = std::abs(offset.x) == 1 && std::abs(offset.y) == 1 && std::abs(offset.z) == 1;
  const bool is_low_y_edge = std::abs(offset.x) == 1 && offset.y == -1 && offset.z == 0;
  const bool is_y_z_edge = offset.x == 0 && std::abs(offset.y) == 1 && std::abs(offset.z) == 1;
  return is_cell || is_corner || is_low_y_edge || is_y_z_edge;
}

// The 6 x 6 x 6 box of size-level-1 places of the two-size case: the places ComputeIsKeptWhole keeps are active, every
// other place is split into its 8 size-level-2 cells.
std::vector<CellId> BuildTwoSizesWorstCase() {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 6; ++z) {
    for (std::int32_t y = 0; y < 6; ++y) {
      for (std::int32_t x = 0; x < 6; ++x) {
        if (ComputeIsKeptWhole({.x = x - 2, .y = y - 2, .z = z - 2})) {
          cells.push_back(ToCell(1, x, y, z));
        } else {
          AppendSmallCells(ToCell(1, x, y, z), cells);
        }
      }
    }
  }
  return cells;
}

// The largest count known where two sizes meet: size-level-1 cell {2,2,2} keeps its 8 corner neighbours and 6 edge
// neighbours at its size; every other size-level-1 place of the 6 x 6 x 6 box is split.
TEST(GeometricGrid, TwoSizesWorstCase_Has124NearbyFaces) {
  const std::vector<CellId> cells = BuildTwoSizesWorstCase();
  ASSERT_EQ(cells.size(), 15U + 1608U);
  const GridFrame frame = {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 4.0,
      .size_level_count = 3,
  };
  const GeometricGrid grid(frame, cells);

  // The position {4.75, 4.25, 4.75} m lies in cell {2,2,2} (4 m to 6 m) at quarters 1, 0 and 1.
  const std::optional<CellId> cell = grid.ContainingCell({.x = 4.75, .y = 4.25, .z = 4.75});
  ASSERT_TRUE(cell.has_value());
  EXPECT_EQ(ToValue(cell), ToCell(1, 2, 2, 2).value);
  EXPECT_EQ(grid.NearbyFaces(cell.value_or(CellId{0}), {1, 0, 1}, Axis::kY).size(), 124U);
}

TEST(GeometricGrid, NearbyFaces_AreFacePiecesOfTheAxis) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());

  for (const GeometricFacePiece& piece : grid.NearbyFaces(ToCell(1, 2, 0, 0), {2, 1, 3}, Axis::kZ)) {
    EXPECT_EQ(piece.axis, Axis::kZ);
    const auto same_id = [&piece](const GeometricFacePiece& other) {
      return other.id.value == piece.id.value;
    };
    EXPECT_EQ(std::ranges::count_if(grid.FacePieces(), same_id), 1);
  }
}

TEST(GeometricGrid, SplitCellBesideBigCell_FacesWithinMatchHandCount) {
  const GeometricGrid grid(ToUnitFrame(2), BuildSplitCellBesideBigCell());
  // The quarter at x = 1 m, centre {1, 0.25, 0.25} m, width 0.5 m.
  const FacePieceId quarter = ToFacePieceId(Axis::kX, SizeLevel{1}, {.x = 2, .y = 0, .z = 0});
  // The big cell's -X side, centre {0, 0.5, 0.5} m, width 1 m.
  const FacePieceId big_low_side = ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 0, .y = 0, .z = 0});

  // Below 0.75 m on each axis: the 4 quarters and the 4 pieces at x = 1.5 m.
  EXPECT_EQ(grid.FacesWithin(quarter, 1.5).size(), 8U);
  // Below 1.25 m: every X piece, the big -X side and the 4 at x = 2 m included.
  EXPECT_EQ(grid.FacesWithin(quarter, 2.5).size(), 13U);
  EXPECT_EQ(grid.FacesWithin(quarter, 4.0).size(), 13U);
  // Below 1.5 m: itself and the 4 quarters; the pieces at x = 1.5 m are exactly 1.5 m away.
  EXPECT_EQ(grid.FacesWithin(big_low_side, 1.5).size(), 5U);
}

TEST(GeometricGrid, UniformGridRadiusOneAndAHalf_Finds27FacesWithin) {
  std::vector<CellId> cells;
  for (std::int32_t z = 0; z < 5; ++z) {
    for (std::int32_t y = 0; y < 5; ++y) {
      for (std::int32_t x = 0; x < 5; ++x) {
        cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  const GeometricGrid grid(ToUnitFrame(1), cells);

  const FacePieceId face = ToFacePieceId(Axis::kX, SizeLevel{0}, {.x = 2, .y = 2, .z = 2});

  EXPECT_EQ(grid.FacesWithin(face, 1.5).size(), 27U);
}

// ---------------------------------------------------------------------------------------------------------------------
// Size-step closure
// ---------------------------------------------------------------------------------------------------------------------

TEST(BuildSizeStepClosure, CellsTwoSizeLevelsSmallerAlongside_SplitTheBigCell) {
  // Size-level-0 cell {0,0,0}; size-level-0 cell {1,0,0} split; its size-level-1 cell {2,0,0} split again, so
  // size-level-2 cells touch the big cell at x = 1 m.
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 1; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(ToCell(0, 1, 0, 0), cell_eighth));
  }
  AppendSmallCells(ToCell(1, 2, 0, 0), cells);

  const std::vector<CellId> closed = BuildSizeStepClosure(cells, 3);

  // The big cell becomes its 8 small cells; nothing else changes.
  EXPECT_EQ(closed.size(), 8U + 7U + 8U);
  std::set<std::uint64_t> values;
  for (const CellId cell : closed) {
    values.insert(cell.value);
  }
  EXPECT_FALSE(values.contains(ToCell(0, 0, 0, 0).value));
  EXPECT_TRUE(values.contains(ToCell(1, 1, 1, 1).value));
  EXPECT_TRUE(values.contains(ToCell(2, 4, 0, 0).value));
}

TEST(BuildSizeStepClosure, ValidList_IsKeptInFixedOrder) {
  const std::vector<CellId> cells = BuildSplitCellBesideBigCell();

  const std::vector<CellId> closed = BuildSizeStepClosure(cells, 2);

  ASSERT_EQ(closed.size(), cells.size());
  for (std::size_t index = 1; index < closed.size(); ++index) {
    EXPECT_TRUE(ComesBefore(closed.at(index - 1), closed.at(index))) << index;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Random grids
// ---------------------------------------------------------------------------------------------------------------------

// Says whether all 8 small cells of the big cell of `cell` are in `listed`. A cell of size level 0 has no big cell.
bool ComputeHasAllSiblings(CellId cell, const std::set<std::uint64_t>& listed) {
  if (static_cast<int>(ToSizeLevel(cell)) == 0) {
    return true;
  }
  const CellId big_cell = ComputeBigCell(cell);
  for (int cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    if (!listed.contains(ComputeSmallCell(big_cell, cell_eighth).value)) {
      return false;
    }
  }
  return true;
}

// Checks by brute force that every cell is listed once with its siblings: the set of active cells and every bigger
// cell that holds one has, for each cell of size level 1 or more, all 8 small cells of its big cell; and no listed
// cell is a bigger cell of another.
::testing::AssertionResult CheckSiblingsTogether(const std::vector<CellId>& cells) {
  std::set<std::uint64_t> active_values;
  for (const CellId cell : cells) {
    if (!active_values.insert(cell.value).second) {
      return ::testing::AssertionFailure() << "a cell is listed twice";
    }
  }
  // Every bigger cell that holds an active cell is split.
  std::set<std::uint64_t> split_values;
  for (const CellId cell : cells) {
    CellId bigger_cell = cell;
    while (static_cast<int>(ToSizeLevel(bigger_cell)) > 0) {
      bigger_cell = ComputeBigCell(bigger_cell);
      split_values.insert(bigger_cell.value);
    }
  }
  const bool is_split_cell_active = std::ranges::any_of(split_values, [&active_values](std::uint64_t split_value) {
    return active_values.contains(split_value);
  });
  if (is_split_cell_active) {
    return ::testing::AssertionFailure() << "an active cell holds another active cell";
  }
  std::set<std::uint64_t> listed_values = active_values;
  listed_values.insert(split_values.begin(), split_values.end());
  for (const std::uint64_t listed_value : listed_values) {
    if (!ComputeHasAllSiblings(CellId{.value = listed_value}, listed_values)) {
      return ::testing::AssertionFailure() << "a sibling of cell " << listed_value << " is missing";
    }
  }
  return ::testing::AssertionSuccess();
}

// Says whether two boxes share at least one point: their closed intervals meet on every axis.
bool ComputeDoBoxesTouch(const UnitBox& first, const UnitBox& second) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const bool is_apart = first.low.at(axis) > second.low.at(axis) + second.width
        || second.low.at(axis) > first.low.at(axis) + first.width;
    if (is_apart) {
      return false;
    }
  }
  return true;
}

// Checks by brute force, on every pair of boxes, that every two touching active cells differ by at most one size
// level. The bigger cell of a broken pair is two or more size levels above the other, so only cells below size level
// size_level_count - 2 are paired.
::testing::AssertionResult CheckSizeStepByBruteForce(const std::vector<CellId>& cells, int size_level_count) {
  for (const CellId cell : cells) {
    const int size_level = static_cast<int>(ToSizeLevel(cell));
    if (size_level + 2 >= size_level_count) {
      continue;
    }
    const UnitBox box = ToUnitBox(cell, size_level_count);
    for (const CellId other : cells) {
      const int other_size_level = static_cast<int>(ToSizeLevel(other));
      const bool is_two_smaller = other_size_level >= size_level + 2;
      if (is_two_smaller && ComputeDoBoxesTouch(box, ToUnitBox(other, size_level_count))) {
        return ::testing::AssertionFailure()
            << "cells of size levels " << size_level << " and " << other_size_level << " touch";
      }
    }
  }
  return ::testing::AssertionSuccess();
}

// The two brute-force checks below must be able to fail, or the random-grid tests would prove nothing.
TEST(CheckSiblingsTogether, MissingSiblingOrSplitCellListed_Fails) {
  std::vector<CellId> seven_of_eight = BuildSplitCellBesideBigCell();
  seven_of_eight.pop_back();
  std::vector<CellId> big_and_small = BuildSplitCellBesideBigCell();
  big_and_small.push_back(ToCell(0, 1, 0, 0));

  EXPECT_TRUE(CheckSiblingsTogether(BuildSplitCellBesideBigCell()));
  EXPECT_FALSE(CheckSiblingsTogether(seven_of_eight));
  EXPECT_FALSE(CheckSiblingsTogether(big_and_small));
}

TEST(CheckSizeStepByBruteForce, CellsTwoSizeLevelsApart_FailsUntilClosed) {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 1; cell_eighth < 8; ++cell_eighth) {
    cells.push_back(ComputeSmallCell(ToCell(0, 1, 0, 0), cell_eighth));
  }
  AppendSmallCells(ToCell(1, 2, 0, 0), cells);
  const std::vector<CellId> closed = BuildSizeStepClosure(cells, 3);

  EXPECT_FALSE(CheckSizeStepByBruteForce(cells, 3));
  EXPECT_TRUE(CheckSizeStepByBruteForce(closed, 3));
  EXPECT_TRUE(CheckSiblingsTogether(cells));
  EXPECT_TRUE(CheckSiblingsTogether(closed));
}

TEST(BuildRandomGridCells, SameSeed_GivesSameList) {
  const std::vector<CellId> first = BuildRandomGridCells(17, 4, 3);
  const std::vector<CellId> second = BuildRandomGridCells(17, 4, 3);
  const std::vector<CellId> other_seed = BuildRandomGridCells(18, 4, 3);

  EXPECT_FALSE(first.empty());
  EXPECT_EQ(ToValues(first), ToValues(second));
  EXPECT_NE(ToValues(first), ToValues(other_seed));
}

TEST(ApplyFixedShuffle, SeedOne_GivesAPermutationThatDiffersFromSeedTwo) {
  std::vector<CellId> input;
  input.reserve(100);
  for (std::int32_t x = 0; x < 100; ++x) {
    input.push_back(ToCell(0, x, 0, 0));
  }
  std::vector<CellId> seed_one = input;
  std::vector<CellId> seed_one_again = input;
  std::vector<CellId> seed_two = input;

  ApplyFixedShuffle(seed_one, 1);
  ApplyFixedShuffle(seed_one_again, 1);
  ApplyFixedShuffle(seed_two, 2);

  std::vector<std::uint64_t> sorted_seed_one = ToValues(seed_one);
  std::ranges::sort(sorted_seed_one);
  std::vector<std::uint64_t> sorted_input = ToValues(input);
  std::ranges::sort(sorted_input);
  EXPECT_EQ(sorted_seed_one, sorted_input);
  EXPECT_EQ(ToValues(seed_one), ToValues(seed_one_again));
  EXPECT_NE(ToValues(seed_one), ToValues(seed_two));
  EXPECT_NE(ToValues(seed_one), ToValues(input));
}

TEST(BuildShuffledRandomGrids, FirstAndLastEntries_AreTheirSeedsShuffled) {
  const std::vector<std::vector<CellId>>& grids = BuildShuffledRandomGrids();
  ASSERT_EQ(grids.size(), kRandomGridCount);
  for (const std::uint64_t seed : {std::uint64_t{1}, kRandomGridCount}) {
    std::vector<CellId> expected = BuildRandomGridCells(seed, 4, 3);
    ApplyFixedShuffle(expected, seed + 1000);
    EXPECT_EQ(ToValues(grids.at(seed - 1)), ToValues(expected)) << seed;
  }
}

TEST(BuildRandomGridCells, AnySeed_KeepsSiblingsAndSizeStep) {
  for (std::uint64_t seed = 1; seed <= 100; ++seed) {
    const std::vector<CellId> cells = BuildRandomGridCells(seed, 4, 3);
    EXPECT_TRUE(CheckSiblingsTogether(cells)) << seed;
    EXPECT_TRUE(CheckSizeStepByBruteForce(cells, 4)) << seed;
  }
}

// Checks that the cells cover every unit of the box 0..box_width - 1 on each axis once and nothing outside it. Units
// are of the frame's last size level.
::testing::AssertionResult
CheckFillsBoxOnce(const std::vector<CellId>& cells, GridFrame frame, std::int64_t box_width) {
  std::vector<int> cover_count(static_cast<std::size_t>(box_width * box_width * box_width), 0);
  const auto is_unit_inside = [box_width](std::int64_t unit) {
    return unit >= 0 && unit < box_width;
  };
  for (const CellId cell : cells) {
    const UnitBox box = ToUnitBox(cell, frame.size_level_count);
    const bool is_box_inside =
        std::ranges::all_of(box.low, is_unit_inside) && std::ranges::all_of(box.low, [&](std::int64_t low) {
          return is_unit_inside(low + box.width - 1);
        });
    if (!is_box_inside) {
      return ::testing::AssertionFailure() << "cell " << cell.value << " lies outside the box";
    }
    for (std::int64_t z = box.low.at(2); z < box.low.at(2) + box.width; ++z) {
      for (std::int64_t y = box.low.at(1); y < box.low.at(1) + box.width; ++y) {
        for (std::int64_t x = box.low.at(0); x < box.low.at(0) + box.width; ++x) {
          ++cover_count.at(static_cast<std::size_t>((z * box_width + y) * box_width + x));
        }
      }
    }
  }
  if (!std::ranges::all_of(cover_count, [](int count) {
        return count == 1;
      })) {
    return ::testing::AssertionFailure() << "a unit is covered twice or not at all";
  }
  return ::testing::AssertionSuccess();
}

TEST(BuildRandomGridCells, AnySeed_FillsTheBoxOnce) {
  // In units of size level 3, the box of 3 size-level-0 cells per axis is 24 units wide.
  for (std::uint64_t seed = 1; seed <= 100; ++seed) {
    EXPECT_TRUE(CheckFillsBoxOnce(BuildRandomGridCells(seed, 4, 3), ToUnitFrame(4), 24)) << seed;
  }
}

TEST(BuildRandomGridCells, TwoHundredGrids_UseEverySizeLevel) {
  std::array<bool, 4> is_size_level_used = {};
  std::size_t largest_cell_count = 0;
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<CellId> cells = BuildRandomGridCells(seed, 4, 3);
    largest_cell_count = std::max(largest_cell_count, cells.size());
    for (const CellId cell : cells) {
      is_size_level_used.at(static_cast<std::size_t>(ToSizeLevel(cell))) = true;
    }
  }
  for (std::size_t size_level = 0; size_level < is_size_level_used.size(); ++size_level) {
    EXPECT_TRUE(is_size_level_used.at(size_level)) << size_level;
  }
  // 27 size-level-0 cells, each at most 8 x 8 x 8 cells of size level 3.
  EXPECT_LE(largest_cell_count, 13824U);
  RecordProperty("largest_cell_count", static_cast<int>(largest_cell_count));
}

// Checks that, for every cell and axis, its face pieces on the low side and on the high side each add up to its whole
// side. Areas are powers of two in a frame of biggest cells 1 m wide, so the sums are exact.
::testing::AssertionResult CheckCoversEveryCellSideOnce(const GeometricGrid& grid, const std::vector<CellId>& cells) {
  // Per cell and axis, the area of its pieces on the low side (entry 0) and on the high side (entry 1).
  std::map<std::tuple<std::uint64_t, Axis>, std::array<double, 2>> side_areas;
  for (const GeometricFacePiece& piece : grid.FacePieces()) {
    side_areas[{ToValue(piece.cell_high), piece.axis}].at(0) += piece.area;
    side_areas[{ToValue(piece.cell_low), piece.axis}].at(1) += piece.area;
  }
  for (const CellId cell : cells) {
    const double width = std::ldexp(1.0, -static_cast<int>(ToSizeLevel(cell)));
    const std::array<double, 2> whole_sides = {width * width, width * width};
    const bool is_covered_once = side_areas[{cell.value, Axis::kX}] == whole_sides
        && side_areas[{cell.value, Axis::kY}] == whole_sides && side_areas[{cell.value, Axis::kZ}] == whole_sides;
    if (!is_covered_once) {
      return ::testing::AssertionFailure() << "a side of cell " << cell.value << " is not covered once";
    }
  }
  return ::testing::AssertionSuccess();
}

TEST(GeometricGrid, RandomGrids_CoverEveryCellSideOnce) {
  for (std::uint64_t seed = 1; seed <= 20; ++seed) {
    const std::vector<CellId> cells = BuildRandomGridCells(seed, 4, 3);
    EXPECT_TRUE(CheckCoversEveryCellSideOnce(GeometricGrid(ToUnitFrame(4), cells), cells)) << seed;
  }
}

TEST(GeometricGrid, RandomGrids_FacePieceIdsAreDistinct) {
  for (std::uint64_t seed = 1; seed <= 20; ++seed) {
    const GeometricGrid grid(ToUnitFrame(4), BuildRandomGridCells(seed, 4, 3));
    std::set<std::uint64_t> ids;
    for (const GeometricFacePiece& piece : grid.FacePieces()) {
      ids.insert(piece.id.value);
    }
    EXPECT_EQ(ids.size(), grid.FacePieces().size()) << seed;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Recorded digests
// ---------------------------------------------------------------------------------------------------------------------
//
// Each test below turns a group of answers into 64-bit words, in the order the answers give them, and compares the
// ComputeResultDigest of those words (64-bit FNV-1a over their bytes, base/test_support.h) with the value the test
// support gave when the test was written. A faster way to find the same answers keeps every digest; a changed list,
// answer or order changes it. The words are read in the machine's byte order, little-endian on every machine the
// project builds for.

std::uint64_t ComputeWordsDigest(const std::vector<std::uint64_t>& words) {
  return ComputeResultDigest(std::as_bytes(std::span<const std::uint64_t>(words)));
}

// A signed whole number as a word: its two's-complement bits.
std::uint64_t ToWord(std::int64_t number) {
  return static_cast<std::uint64_t>(number);
}

// Appends the length of `cells`, then the value of each cell in order.
void ApplyCellsToWords(const std::vector<CellId>& cells, std::vector<std::uint64_t>& words) {
  words.push_back(cells.size());
  for (const CellId cell : cells) {
    words.push_back(cell.value);
  }
}

// Appends the length of `pieces`, then every field of each piece in order, the area by its bits.
void ApplyFacePiecesToWords(const std::vector<GeometricFacePiece>& pieces, std::vector<std::uint64_t>& words) {
  words.push_back(pieces.size());
  for (const GeometricFacePiece& piece : pieces) {
    words.push_back(piece.id.value);
    words.push_back(static_cast<std::uint64_t>(piece.axis));
    words.push_back(static_cast<std::uint64_t>(piece.size_level));
    words.push_back(ToWord(piece.cell_high_coord.x));
    words.push_back(ToWord(piece.cell_high_coord.y));
    words.push_back(ToWord(piece.cell_high_coord.z));
    words.push_back(ToValue(piece.cell_low));
    words.push_back(ToValue(piece.cell_high));
    words.push_back(std::bit_cast<std::uint64_t>(piece.area));
  }
}

// Moves every cell by offset_in_biggest_cells size-level-0 cells on each axis: a cell of size level l moves by
// offset_in_biggest_cells x 2^l of its own coordinates.
std::vector<CellId> BuildMovedCells(const std::vector<CellId>& cells, std::int32_t offset_in_biggest_cells) {
  std::vector<CellId> moved_cells;
  moved_cells.reserve(cells.size());
  for (const CellId cell : cells) {
    const int size_level = static_cast<int>(ToSizeLevel(cell));
    const auto cells_per_biggest_cell = static_cast<std::int32_t>(1U << static_cast<unsigned>(size_level));
    const std::int32_t offset = offset_in_biggest_cells * cells_per_biggest_cell;
    const CellCoord coord = ToCellCoord(cell);
    moved_cells.push_back(ToCell(size_level, coord.x + offset, coord.y + offset, coord.z + offset));
  }
  return moved_cells;
}

// Size-level-0 cell {1,1,1} and its 26 neighbours split into their size-level-1 cells, in a frame of 2 size levels.
std::vector<CellId> BuildAllNeighboursSplit() {
  std::vector<CellId> cells = {ToCell(0, 1, 1, 1)};
  for (std::int32_t z = 0; z < 3; ++z) {
    for (std::int32_t y = 0; y < 3; ++y) {
      for (std::int32_t x = 0; x < 3; ++x) {
        if (x != 1 || y != 1 || z != 1) {
          AppendSmallCells(ToCell(0, x, y, z), cells);
        }
      }
    }
  }
  return cells;
}

// A list whose checker answers the digest tests read.
struct DigestGrid {
  GridFrame frame;
  std::vector<CellId> cells;
};

// The grids of the checker's digest tests: hand-built grids of 1 to 3 size levels, shuffled random grids, one of them
// in a frame whose origin is away from 0, and random grids moved across zero and to both coordinate limits.
// Keep the moved grids: they are the only tests that place cells at negative coordinates, where a lookup that
// truncates instead of rounding down finds the wrong place beyond a side. The chain scenes of
// HandBuiltScenes_KeepTheirDigest are, likewise, the only tests that see a closure stop a round early.
std::vector<DigestGrid> BuildDigestGrids() {
  std::vector<CellId> at_coord_limits = {ToCell(0, -kMaxCoord, 0, 0), ToCell(0, kMaxCoord, 0, 0)};
  AppendSmallCells(ToCell(0, 0, 0, 0), at_coord_limits);
  std::vector<CellId> uniform_cells;
  for (std::int32_t z = 0; z < 5; ++z) {
    for (std::int32_t y = 0; y < 5; ++y) {
      for (std::int32_t x = 0; x < 5; ++x) {
        uniform_cells.push_back(ToCell(0, x, y, z));
      }
    }
  }
  const GridFrame worst_case_frame = {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 4.0,
      .size_level_count = 3,
  };
  const GridFrame moved_origin_frame = {
      .origin = {.x = -1.5, .y = 0.25, .z = 3.0},
      .biggest_cell_width = 0.5,
      .size_level_count = 4,
  };
  // The box moved to size-level-0 cells 2^15 - 3..2^15 - 1, and to -(2^15 - 1)..-(2^15 - 3).
  constexpr auto kHighOffset = static_cast<std::int32_t>((1U << 15U) - 3U);
  constexpr std::int32_t kLowOffset = -static_cast<std::int32_t>((1U << 15U) - 1U);
  const std::vector<std::vector<CellId>>& random_grids = BuildShuffledRandomGrids();
  return {
      {.frame = ToUnitFrame(2), .cells = BuildSplitCellBesideBigCell()},
      {.frame = ToUnitFrame(2), .cells = at_coord_limits},
      {.frame = ToUnitFrame(2), .cells = BuildAllNeighboursSplit()},
      {.frame = ToUnitFrame(1), .cells = uniform_cells},
      {.frame = worst_case_frame, .cells = BuildTwoSizesWorstCase()},
      {.frame = ToUnitFrame(4), .cells = random_grids.at(0)},
      {.frame = moved_origin_frame, .cells = random_grids.at(1)},
      {.frame = ToUnitFrame(4), .cells = random_grids.at(99)},
      {.frame = ToUnitFrame(4), .cells = random_grids.at(199)},
      {.frame = ToUnitFrame(4), .cells = BuildMovedCells(random_grids.at(2), -1)},
      {.frame = ToUnitFrame(4), .cells = BuildMovedCells(random_grids.at(3), kHighOffset)},
      {.frame = ToUnitFrame(4), .cells = BuildMovedCells(random_grids.at(4), kLowOffset)},
  };
}

// The step that visits about `wanted_count` of `count` entries, at least 1.
std::size_t ComputeVisitStep(std::size_t count, std::size_t wanted_count) {
  return std::max<std::size_t>(count / wanted_count, 1);
}

TEST(BuildRandomGridCells, SeedsOneToTwoHundred_KeepTheirDigest) {
  std::vector<std::uint64_t> words;
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    ApplyCellsToWords(BuildRandomGridCells(seed, 4, 3), words);
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0xcb78fa151cdbdd79ULL);
}

TEST(BuildRandomGridCells, OtherSizeLevelCountsAndBoxes_KeepTheirDigest) {
  // Pairs (size_level_count, biggest_cells_per_axis), each for seeds 1..10.
  constexpr std::array<std::array<int, 2>, 5> kSizes = {{{1, 3}, {2, 4}, {3, 3}, {5, 2}, {3, 1}}};
  std::vector<std::uint64_t> words;
  for (const std::array<int, 2>& size : kSizes) {
    for (std::uint64_t seed = 1; seed <= 10; ++seed) {
      ApplyCellsToWords(BuildRandomGridCells(seed, size.at(0), size.at(1)), words);
    }
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x9b6001a17bb5ed90ULL);
}

TEST(ApplyFixedShuffle, ListsOfSeveralLengths_KeepTheirDigest) {
  std::vector<std::uint64_t> words;
  for (const std::int32_t length : {0, 1, 2, 3, 100}) {
    for (const std::uint64_t seed : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{12345}}) {
      std::vector<CellId> cells;
      cells.reserve(static_cast<std::size_t>(length));
      for (std::int32_t x = 0; x < length; ++x) {
        cells.push_back(ToCell(0, x, 0, 0));
      }
      ApplyFixedShuffle(cells, seed);
      ApplyCellsToWords(cells, words);
    }
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x3f40ceca5040a133ULL);
}

TEST(BuildShuffledRandomGrids, AllEntries_KeepTheirDigest) {
  std::vector<std::uint64_t> words;
  for (const std::vector<CellId>& cells : BuildShuffledRandomGrids()) {
    ApplyCellsToWords(cells, words);
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x8c4637ae9848f431ULL);
}

// Size-level-0 cell {0,0,0} beside size level 0 {1,0,0} split down to last_size_level at its low corner, siblings
// kept. The closure takes several rounds: each splits the cells the last round's splits left too big.
std::vector<CellId> BuildSplitChain(int last_size_level) {
  std::vector<CellId> cells = {ToCell(0, 0, 0, 0)};
  CellId chain_cell = ToCell(0, 1, 0, 0);
  for (int size_level = 0; size_level < last_size_level; ++size_level) {
    for (int cell_eighth = 1; cell_eighth < 8; ++cell_eighth) {
      cells.push_back(ComputeSmallCell(chain_cell, cell_eighth));
    }
    chain_cell = ComputeSmallCell(chain_cell, 0);
  }
  cells.push_back(chain_cell);
  return cells;
}

TEST(BuildSizeStepClosure, HandBuiltScenes_KeepTheirDigest) {
  // Size-level-2 cells beside the size-level-0 cell {0,0,0} at x = 1 m, 3 size levels.
  std::vector<CellId> side_touch = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 1; cell_eighth < 8; ++cell_eighth) {
    side_touch.push_back(ComputeSmallCell(ToCell(0, 1, 0, 0), cell_eighth));
  }
  AppendSmallCells(ToCell(1, 2, 0, 0), side_touch);
  // Size-level-2 cell {4,4,4} touches the size-level-0 cell {0,0,0} at the corner {1, 1, 1} m only, 3 size levels.
  std::vector<CellId> corner_touch = {ToCell(0, 0, 0, 0)};
  for (int cell_eighth = 1; cell_eighth < 8; ++cell_eighth) {
    corner_touch.push_back(ComputeSmallCell(ToCell(0, 1, 1, 1), cell_eighth));
  }
  AppendSmallCells(ToCell(1, 2, 2, 2), corner_touch);
  const std::vector<CellId> chain = BuildSplitChain(3);
  const std::vector<CellId> long_chain = BuildSplitChain(5);
  // Size level 0 {-8,0,0} split along the path of eighth-7 cells down to size level 15, 16 size levels.
  std::vector<CellId> deep_path = {ToCell(0, -8, 0, 0)};
  CellId path_cell = ToCell(0, -8, 0, 0);
  for (int size_level = 0; size_level < 15; ++size_level) {
    std::erase_if(deep_path, [path_cell](CellId cell) {
      return cell.value == path_cell.value;
    });
    AppendSmallCells(path_cell, deep_path);
    path_cell = ComputeSmallCell(path_cell, 7);
  }
  std::vector<CellId> shuffled_chain = chain;
  ApplyFixedShuffle(shuffled_chain, 7);

  std::vector<std::uint64_t> words;
  ApplyCellsToWords(BuildSizeStepClosure({}, 3), words);
  ApplyCellsToWords(BuildSizeStepClosure({ToCell(0, 2, -3, 4)}, 3), words);
  ApplyCellsToWords(BuildSizeStepClosure(BuildSplitCellBesideBigCell(), 2), words);
  ApplyCellsToWords(BuildSizeStepClosure(side_touch, 3), words);
  ApplyCellsToWords(BuildSizeStepClosure(corner_touch, 3), words);
  ApplyCellsToWords(BuildSizeStepClosure(chain, 4), words);
  ApplyCellsToWords(BuildSizeStepClosure(chain, 6), words);
  ApplyCellsToWords(BuildSizeStepClosure(long_chain, 6), words);
  ApplyCellsToWords(BuildSizeStepClosure(shuffled_chain, 4), words);
  ApplyCellsToWords(BuildSizeStepClosure(BuildMovedCells(chain, -5), 4), words);
  ApplyCellsToWords(BuildSizeStepClosure(deep_path, kMaxSizeLevels), words);
  EXPECT_EQ(ComputeWordsDigest(words), 0x64ad4802c9506729ULL);
}

TEST(GeometricGrid, FacePiecesOfDigestGrids_KeepTheirDigest) {
  std::vector<std::uint64_t> words;
  for (const DigestGrid& digest_grid : BuildDigestGrids()) {
    ApplyFacePiecesToWords(GeometricGrid(digest_grid.frame, digest_grid.cells).FacePieces(), words);
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x98da295875fd6dafULL);
}

TEST(GeometricGrid, TouchingCellsOfDigestGrids_KeepTheirDigest) {
  // About 50 active cells of each grid, and the big cell of each, which is not active.
  std::vector<std::uint64_t> words;
  for (const DigestGrid& digest_grid : BuildDigestGrids()) {
    const GeometricGrid grid(digest_grid.frame, digest_grid.cells);
    const std::size_t step = ComputeVisitStep(digest_grid.cells.size(), 50);
    for (std::size_t index = 0; index < digest_grid.cells.size(); index += step) {
      const CellId cell = digest_grid.cells.at(index);
      ApplyCellsToWords(grid.TouchingCells(cell), words);
      if (static_cast<int>(ToSizeLevel(cell)) > 0) {
        ApplyCellsToWords(grid.TouchingCells(ComputeBigCell(cell)), words);
      }
    }
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0xb4b1cfce250b11bfULL);
}

TEST(GeometricGrid, CornersOfDigestGrids_KeepTheirDigest) {
  std::vector<std::uint64_t> words;
  for (const DigestGrid& digest_grid : BuildDigestGrids()) {
    const GeometricGrid grid(digest_grid.frame, digest_grid.cells);
    const std::vector<std::array<std::int64_t, 3>> corners_in_units = grid.CornersInUnits();
    words.push_back(corners_in_units.size());
    for (const std::array<std::int64_t, 3>& corner : corners_in_units) {
      words.insert(words.end(), {ToWord(corner.at(0)), ToWord(corner.at(1)), ToWord(corner.at(2))});
    }
    const std::vector<Vec3> corners = grid.Corners();
    words.push_back(corners.size());
    for (const Vec3 corner : corners) {
      words.insert(
          words.end(),
          {
              std::bit_cast<std::uint64_t>(corner.x),
              std::bit_cast<std::uint64_t>(corner.y),
              std::bit_cast<std::uint64_t>(corner.z),
          }
      );
    }
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x4b6b0e5485e9d832ULL);
}

TEST(GeometricGrid, ContainingCellOfDigestGrids_KeepsItsDigest) {
  // About 100 corners of each grid, each as itself and moved half a unit down and up on every axis, then positions
  // that are in no cell.
  std::vector<std::uint64_t> words;
  for (const DigestGrid& digest_grid : BuildDigestGrids()) {
    const GeometricGrid grid(digest_grid.frame, digest_grid.cells);
    const std::vector<Vec3> corners = grid.Corners();
    const double unit_width =
        std::ldexp(digest_grid.frame.biggest_cell_width, 1 - static_cast<int>(digest_grid.frame.size_level_count));
    const std::size_t step = ComputeVisitStep(corners.size(), 100);
    for (std::size_t index = 0; index < corners.size(); index += step) {
      const Vec3 corner = corners.at(index);
      for (const double shift : {0.0, -0.5, 0.5}) {
        const double offset = shift * unit_width;
        words.push_back(
            ToValue(grid.ContainingCell({.x = corner.x + offset, .y = corner.y + offset, .z = corner.z + offset}))
        );
      }
    }
    for (const double far : {std::nan(""), std::numeric_limits<double>::infinity(), 1e12, -1e12}) {
      words.push_back(ToValue(grid.ContainingCell({.x = far, .y = 0, .z = 0})));
    }
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x3cb7358576b363a0ULL);
}

// The window quarters the nearby-face digest reads: the 8 windows of quarters 0 and 3, and window (1, 2, 1).
std::vector<std::array<int, 3>> BuildDigestWindows() {
  std::vector<std::array<int, 3>> windows = {{1, 2, 1}};
  for (int corner = 0; corner < 8; ++corner) {
    const auto corner_bits = static_cast<unsigned>(corner);
    windows.push_back({
        (corner_bits & 1U) != 0 ? 3 : 0,
        (corner_bits & 2U) != 0 ? 3 : 0,
        (corner_bits & 4U) != 0 ? 3 : 0,
    });
  }
  return windows;
}

TEST(GeometricGrid, NearbyFacesOfDigestGrids_KeepTheirDigest) {
  // About 6 active cells of each grid, in every window of BuildDigestWindows, on every axis.
  std::vector<std::uint64_t> words;
  for (const DigestGrid& digest_grid : BuildDigestGrids()) {
    const GeometricGrid grid(digest_grid.frame, digest_grid.cells);
    const std::size_t step = ComputeVisitStep(digest_grid.cells.size(), 6);
    for (std::size_t index = 0; index < digest_grid.cells.size(); index += step) {
      for (const std::array<int, 3>& window_quarters : BuildDigestWindows()) {
        for (const Axis axis : kAxes) {
          ApplyFacePiecesToWords(grid.NearbyFaces(digest_grid.cells.at(index), window_quarters, axis), words);
        }
      }
    }
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x679836b2a613cb75ULL);
}

TEST(GeometricGrid, FacesWithinOfDigestGrids_KeepTheirDigest) {
  // About 10 face pieces of each grid, at radii 0.5, 1.5, 2.5 and 4 cell widths.
  std::vector<std::uint64_t> words;
  for (const DigestGrid& digest_grid : BuildDigestGrids()) {
    const GeometricGrid grid(digest_grid.frame, digest_grid.cells);
    const std::vector<GeometricFacePiece>& pieces = grid.FacePieces();
    const std::size_t step = ComputeVisitStep(pieces.size(), 10);
    for (std::size_t index = 0; index < pieces.size(); index += step) {
      for (const double radius : {0.5, 1.5, 2.5, 4.0}) {
        ApplyFacePiecesToWords(grid.FacesWithin(pieces.at(index).id, radius), words);
      }
    }
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x87afcf5bb15da649ULL);
}

// Appends every answer of the checker of one small grid: its face pieces, the touching cells of every active cell and
// of its big cell, its corners in units and in metres, the containing cell of every corner and of every corner moved
// half a unit down and up, the nearby faces of every cell in every window of BuildDigestWindows on every axis, and the
// faces within 0.5, 1.5, 2.5 and 4 cell widths of every face piece.
void ApplyEveryCheckerAnswerToWords(const DigestGrid& digest_grid, std::vector<std::uint64_t>& words) {
  const GeometricGrid grid(digest_grid.frame, digest_grid.cells);
  ApplyFacePiecesToWords(grid.FacePieces(), words);
  for (const CellId cell : digest_grid.cells) {
    ApplyCellsToWords(grid.TouchingCells(cell), words);
    if (static_cast<int>(ToSizeLevel(cell)) > 0) {
      ApplyCellsToWords(grid.TouchingCells(ComputeBigCell(cell)), words);
    }
  }
  for (const std::array<std::int64_t, 3>& corner : grid.CornersInUnits()) {
    words.insert(words.end(), {ToWord(corner.at(0)), ToWord(corner.at(1)), ToWord(corner.at(2))});
  }
  const double unit_width =
      std::ldexp(digest_grid.frame.biggest_cell_width, 1 - static_cast<int>(digest_grid.frame.size_level_count));
  for (const Vec3 corner : grid.Corners()) {
    words.insert(
        words.end(),
        {
            std::bit_cast<std::uint64_t>(corner.x),
            std::bit_cast<std::uint64_t>(corner.y),
            std::bit_cast<std::uint64_t>(corner.z),
        }
    );
    for (const double shift : {0.0, -0.5, 0.5}) {
      const double offset = shift * unit_width;
      const Vec3 position = {.x = corner.x + offset, .y = corner.y + offset, .z = corner.z + offset};
      words.push_back(ToValue(grid.ContainingCell(position)));
    }
  }
  for (const CellId cell : digest_grid.cells) {
    for (const std::array<int, 3>& window_quarters : BuildDigestWindows()) {
      for (const Axis axis : kAxes) {
        ApplyFacePiecesToWords(grid.NearbyFaces(cell, window_quarters, axis), words);
      }
    }
  }
  for (const GeometricFacePiece& piece : grid.FacePieces()) {
    for (const double radius : {0.5, 1.5, 2.5, 4.0}) {
      ApplyFacePiecesToWords(grid.FacesWithin(piece.id, radius), words);
    }
  }
}

// The 8 size-level-1 cells of size-level-0 cell {131071,0,0}, whose high sides at x = kMaxCoord + 1 face size-level-0
// cell {131072,0,0}, and size-level-0 cell {131070,0,0} below them, 2 size levels. Across those high sides no
// size-level-1 id exists, so the cell beyond is found one size level up. Recorded on the test support with the lookups
// by place; the scans they replaced give the same digest.
TEST(GeometricGrid, BiggerCellBeyondHighCoordLimit_KeepsItsDigest) {
  constexpr auto kLastSplitCoord = static_cast<std::int32_t>((1U << 17U) - 1U);
  std::vector<CellId> cells = {ToCell(0, kLastSplitCoord + 1, 0, 0), ToCell(0, kLastSplitCoord - 1, 0, 0)};
  AppendSmallCells(ToCell(0, kLastSplitCoord, 0, 0), cells);
  ASSERT_TRUE(CheckSiblingsTogether(cells));
  ASSERT_TRUE(CheckSizeStepByBruteForce(cells, 2));
  const GeometricGrid grid(ToUnitFrame(2), cells);
  EXPECT_EQ(grid.FacePieces().size(), 46U);

  std::vector<std::uint64_t> words;
  ApplyEveryCheckerAnswerToWords({.frame = ToUnitFrame(2), .cells = cells}, words);
  EXPECT_EQ(ComputeWordsDigest(words), 0xd15762d10c5f1fd6ULL);
}

#if !defined(NDEBUG)
// What each death test runs. A braced list inside EXPECT_DEATH cannot hold a comma outside parentheses, so each one is
// its own function.

// Size-level-0 cell {0,0,0} listed twice.
std::vector<CellId> BuildRepeatedCell() {
  return {ToCell(0, 0, 0, 0), ToCell(0, 0, 0, 0)};
}

// Size-level-0 cell {0,0,0} and its size-level-1 cell {0,0,0}, inside it.
std::vector<CellId> BuildOverlappingCells() {
  return {ToCell(0, 0, 0, 0), ToCell(1, 0, 0, 0)};
}

void BuildClosureOfRepeatedCell() {
  static_cast<void>(BuildSizeStepClosure(BuildRepeatedCell(), 2));
}

void BuildClosureOfOverlappingCells() {
  static_cast<void>(BuildSizeStepClosure(BuildOverlappingCells(), 2));
}

void BuildCheckerOfRepeatedCell() {
  const GeometricGrid grid(ToUnitFrame(2), BuildRepeatedCell());
  static_cast<void>(grid.FacePieces().size());
}

void BuildCheckerOfOverlappingCells() {
  const GeometricGrid grid(ToUnitFrame(2), BuildOverlappingCells());
  static_cast<void>(grid.FacePieces().size());
}
#endif

TEST(BuildSizeStepClosureDeathTest, RepeatedCell_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(BuildClosureOfRepeatedCell(), "is_cell_listed_once");
#endif
}

TEST(BuildSizeStepClosureDeathTest, OverlappingCells_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(BuildClosureOfOverlappingCells(), "holds_no_listed_cell");
#endif
}

TEST(GeometricGridDeathTest, RepeatedCell_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(BuildCheckerOfRepeatedCell(), "is_cell_listed_once");
#endif
}

TEST(GeometricGridDeathTest, OverlappingCells_Asserts) {
#if defined(NDEBUG)
  GTEST_SKIP() << "release build: assert compiles to nothing";
#else
  EXPECT_DEATH(BuildCheckerOfOverlappingCells(), "holds_no_listed_cell");
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// Helpers that compare grids and pick random grids
// ---------------------------------------------------------------------------------------------------------------------

// Tables of 2 blocks, owned by the test so one byte of each can be changed.
struct OwnedTables {
  static constexpr std::size_t kBlockCount = 2;

  std::vector<CellBlockId> block_ids = std::vector<CellBlockId>(kBlockCount, CellBlockId{7});
  std::vector<std::uint32_t> same_size_level_blocks = std::vector<std::uint32_t>(std::size_t{27} * kBlockCount, 1);
  std::vector<std::uint32_t> bigger_block = std::vector<std::uint32_t>(kBlockCount, 2);
  std::vector<std::uint32_t> smaller_blocks = std::vector<std::uint32_t>(std::size_t{8} * kBlockCount, 3);
  std::vector<CellState> cell_states =
      std::vector<CellState>(std::size_t{kCellPlacesPerBlock} * kBlockCount, CellState::kActive);
  std::vector<FacePieceKind> face_piece_kinds =
      std::vector<FacePieceKind>(std::size_t{kFacePlacesPerBlock} * kBlockCount, FacePieceKind::kBetweenCells);

  [[nodiscard]] GridTables ToTables() const {
    return {
        .version = {.value = 1},
        .block_count = kBlockCount,
        .block_ids = block_ids.data(),
        .same_size_level_blocks = same_size_level_blocks.data(),
        .bigger_block = bigger_block.data(),
        .smaller_blocks = smaller_blocks.data(),
        .cell_states = cell_states.data(),
        .face_piece_kinds = face_piece_kinds.data(),
    };
  }
};

// Tables equal to OwnedTables{} except for the last entry of the table `table`, 0..5 in the order of GridTables.
OwnedTables BuildTablesWithLastEntryChanged(std::size_t table) {
  OwnedTables tables;
  switch (table) {
    case 0:
      tables.block_ids.back().value ^= 1U;
      break;
    case 1:
      tables.same_size_level_blocks.back() ^= 1U;
      break;
    case 2:
      tables.bigger_block.back() ^= 1U;
      break;
    case 3:
      tables.smaller_blocks.back() ^= 1U;
      break;
    case 4:
      tables.cell_states.back() = CellState::kSplit;
      break;
    default:
      tables.face_piece_kinds.back() = FacePieceKind::kNotAPiece;
      break;
  }
  return tables;
}

TEST(ComputeIsSameUsedTables, EqualTables_AreSame) {
  const OwnedTables first;
  const OwnedTables second;
  EXPECT_TRUE(ComputeIsSameUsedTables(first.ToTables(), second.ToTables()));
}

// The last entry of each table is its last used byte, so a check that stops one item early would miss it.
TEST(ComputeIsSameUsedTables, OneByteDifferentInAnyTable_IsNotSame) {
  const OwnedTables first;
  for (std::size_t table = 0; table < 6; ++table) {
    SCOPED_TRACE(table);
    const OwnedTables second = BuildTablesWithLastEntryChanged(table);
    EXPECT_FALSE(ComputeIsSameUsedTables(first.ToTables(), second.ToTables()));
  }
}

TEST(ComputeIsSameUsedTables, DifferentVersion_IsNotSame) {
  const OwnedTables first;
  const OwnedTables second;
  GridTables other_version = second.ToTables();
  other_version.version = {.value = 2};
  EXPECT_FALSE(ComputeIsSameUsedTables(first.ToTables(), other_version));
}

TEST(ComputeIsSameGrid, SameListTwice_IsSameAndOtherListIsNot) {
  const std::vector<CellId> cells = BuildSplitCellBesideBigCell();
  const std::vector<CellId> other_cells = {ToCell(0, 0, 0, 0)};
  const std::expected<TestGrid, GridFailure> first =
      BuildTestGrid({.frame = ToUnitFrame(2), .active_cells = cells, .version = {.value = 1}});
  const std::expected<TestGrid, GridFailure> second =
      BuildTestGrid({.frame = ToUnitFrame(2), .active_cells = cells, .version = {.value = 1}});
  const std::expected<TestGrid, GridFailure> other =
      BuildTestGrid({.frame = ToUnitFrame(2), .active_cells = other_cells, .version = {.value = 1}});
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  ASSERT_TRUE(other.has_value());
  EXPECT_TRUE(ComputeIsSameGrid(first->grid, second->grid));
  EXPECT_FALSE(ComputeIsSameGrid(first->grid, other->grid));
}

// The same list and version in a frame with another origin: the tables are equal bytes, the grids are not the same.
TEST(ComputeIsSameGrid, OtherFrame_IsNotSame) {
  const std::vector<CellId> cells = BuildSplitCellBesideBigCell();
  GridFrame moved_frame = ToUnitFrame(2);
  moved_frame.origin.x = 0.5;
  const std::expected<TestGrid, GridFailure> first =
      BuildTestGrid({.frame = ToUnitFrame(2), .active_cells = cells, .version = {.value = 1}});
  const std::expected<TestGrid, GridFailure> moved =
      BuildTestGrid({.frame = moved_frame, .active_cells = cells, .version = {.value = 1}});
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(moved.has_value());
  EXPECT_FALSE(ComputeIsSameGrid(first->grid, moved->grid));
}

TEST(ToGridCountFields, EveryField_IsKeptInOrder) {
  GridCounts counts{};
  counts.cell_blocks = 2;
  counts.active_cells_per_size_level[0] = 1;
  counts.active_cells_per_size_level[1] = 8;
  counts.active_cells = 9;
  counts.face_pieces = 41;
  const auto [cell_blocks, per_size_level, active_cells, face_pieces] = ToGridCountFields(counts);
  EXPECT_EQ(cell_blocks, 2U);
  EXPECT_EQ(per_size_level.at(0), 1U);
  EXPECT_EQ(per_size_level.at(1), 8U);
  EXPECT_EQ(per_size_level.at(2), 0U);
  EXPECT_EQ(active_cells, 9U);
  EXPECT_EQ(face_pieces, 41U);
}

// Count 4 takes ranks floor(i x 199 / 3) for i = 0..3: 0, 66, 132 and 199. The test finds the rank of each grid it gets
// by counting the grids before it in the order (more cells, or as many cells and a lower index), not by sorting.
TEST(FindRandomGridsSpreadBySize, FourGrids_TakeRanks0_66_132_199) {
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(4);
  ASSERT_EQ(grids.size(), 4U);
  const std::array<std::size_t, 4> expected_ranks = {0, 66, 132, 199};
  for (std::size_t position = 0; position < grids.size(); ++position) {
    SCOPED_TRACE(position);
    const std::size_t grid = grids.at(position);
    std::size_t rank = 0;
    for (std::size_t other = 0; other < kRandomGridCount; ++other) {
      const std::size_t other_size = BuildShuffledRandomGrids().at(other).size();
      const std::size_t grid_size = BuildShuffledRandomGrids().at(grid).size();
      rank += (other_size > grid_size || (other_size == grid_size && other < grid)) ? 1 : 0;
    }
    EXPECT_EQ(rank, expected_ranks.at(position));
  }
}

TEST(FindRandomGridsSpreadBySize, OneGrid_IsTheBiggest) {
  const std::vector<std::size_t> grids = FindRandomGridsSpreadBySize(1);
  ASSERT_EQ(grids.size(), 1U);
  for (const std::vector<CellId>& other : BuildShuffledRandomGrids()) {
    EXPECT_LE(other.size(), BuildShuffledRandomGrids().at(grids.front()).size());
  }
}

// The first failure of a list by step, then by position: step 1 on the ids in input order, then CheckSortedCells on a
// copy sorted with ComesBefore. kNone when the list passes.
GridError ComputeFirstFailureError(const std::vector<CellId>& cells, SiblingCheck sibling_check) {
  for (const CellId cell : cells) {
    const GridError id_error = CheckCellId(ToUnitFrame(4), cell);
    if (id_error != GridError::kNone) {
      return id_error;
    }
  }
  std::vector<CellId> sorted_cells = cells;
  std::ranges::sort(sorted_cells, [](CellId first, CellId second) {
    return ComesBefore(first, second);
  });
  const std::expected<void, GridFailure> checked = CheckSortedCells(sorted_cells, sibling_check);
  return checked.has_value() ? GridError::kNone : checked.error().error;
}

TEST(BuildMutatedGridCells, FiveHundredLists_BreakEveryRule) {
  std::set<GridError> errors_seen;
  std::size_t passing_count = 0;
  for (std::uint64_t seed = 1; seed <= kMutatedGridCount; ++seed) {
    const std::vector<CellId> cells = BuildMutatedGridCells(seed);
    for (const SiblingCheck sibling_check : {SiblingCheck::kRequire, SiblingCheck::kSkip}) {
      const GridError error = ComputeFirstFailureError(cells, sibling_check);
      errors_seen.insert(error);
      passing_count += error == GridError::kNone ? 1 : 0;
    }
  }
  for (const GridError error : {
           GridError::kBadInput,
           GridError::kCoordOutOfRange,
           GridError::kCellsOverlap,
           GridError::kSiblingCellMissing,
           GridError::kSizeStepRuleBroken,
       }) {
    EXPECT_TRUE(errors_seen.contains(error)) << static_cast<int>(error);
  }
  RecordProperty("passing_checks", static_cast<int>(passing_count));
}

TEST(BuildMutatedGridCells, SameSeed_GivesSameList) {
  EXPECT_EQ(ToValues(BuildMutatedGridCells(17)), ToValues(BuildMutatedGridCells(17)));
}

}  // namespace
}  // namespace fluidity::grid
