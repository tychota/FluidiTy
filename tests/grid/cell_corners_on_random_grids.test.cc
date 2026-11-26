#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/cell_corners_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// The 3 pairs of two different axes, each once.
constexpr std::span<const std::array<Axis, 2>> kUnorderedAxisPairs = std::span(kAxisPairs).first<3>();

// FindCornerPlaces, ComputeCornerPosition and FindCellsAroundEdge at every active cell and face piece of the grids of
// `built`, the edges for each pair of axes once (the hand-made grids take both orders): every physical corner has one
// place and the place gives its point back, the points and positions are the checker's, the face corners are the cell
// corners at the same points, and the cells around every edge are the checker's. It checks these properties in one
// function because each TEST runs in a process of its own and would build the grids again. Gives how many edges had
// cells of another size.
std::size_t CheckCornersAndEdgesMatchGeometricChecker(const RandomGridsWithCheckers& built) {
  EXPECT_EQ(built.grids.size(), built.cell_lists.size());
  std::size_t edges_not_same_size = 0;
  for (std::size_t grid = 0; grid < built.grids.size(); ++grid) {
    SCOPED_TRACE(testing::Message() << "random grid " << grid);
    const CornerAndEdgeMistakes mistakes = FindCornerAndEdgeMistakes(
        built.grids.at(grid).grid.tables(RunOn::kCpu),
        built.checkers.at(grid),
        built.cell_lists.at(grid),
        kUnorderedAxisPairs
    );
    CheckNoCornerOrEdgeMistake(mistakes);
    edges_not_same_size += mistakes.edges_not_same_size;
  }
  return edges_not_same_size;
}

TEST(GridTables, RandomGrids_EveryCornerOrEdgeReaderMatchesGeometricChecker) {
  const RandomGridsWithCheckers& built = FindRandomGridsWithCheckers();
  ASSERT_EQ(built.grids.size(), kRandomGridCount);
  // Edges beside cells of another size were met, so the flag was checked both ways.
  EXPECT_GT(CheckCornersAndEdgesMatchGeometricChecker(built), 0U);
}

// The same on the framed random grids: the corner positions are origin + m width in frames with a non-dyadic origin
// and a width that is no power of two, at negative and distant cell coordinates.
TEST(GridTables, FramedRandomGrids_EveryCornerOrEdgeReaderMatchesGeometricChecker) {
  const RandomGridsWithCheckers& built = FindFramedRandomGridsWithCheckers();
  ASSERT_EQ(built.grids.size(), kFramedRandomGridCount);
  EXPECT_GT(CheckCornersAndEdgesMatchGeometricChecker(built), 0U);
}

// A test-only copy of the search before its bound: every size level from the frame's smallest up to the point's own.
CornerPlaceIndex FindCornerPlaceByFullSearch(const GridTables& tables, LatticePoint point) {
  const int point_size_level = static_cast<int>(point.size_level);
  for (int size_level = tables.frame.size_level_count - 1; size_level >= point_size_level; --size_level) {
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

// Counts the corners of active cells whose place differs from that of the full search.
int ComputeCornersNotTheFullSearchCount(const GridTables& tables, const std::vector<CellId>& cells) {
  int mistake_count = 0;
  for (const CellId cell_id : cells) {
    const CellCorners cell_corners = FindCornerPlaces(tables, FindCellPlace(tables, cell_id));
    const CellCoord cell_coord = ToCellCoord(cell_id);
    for (int corner_number = 0; corner_number < 8; ++corner_number) {
      const auto corner_bits = static_cast<unsigned>(corner_number);
      const LatticePoint corner_point = {
          .size_level = ToSizeLevel(cell_id),
          .x = std::int64_t{cell_coord.x} + static_cast<std::int64_t>(corner_bits & 1U),
          .y = std::int64_t{cell_coord.y} + static_cast<std::int64_t>((corner_bits >> 1U) & 1U),
          .z = std::int64_t{cell_coord.z} + static_cast<std::int64_t>((corner_bits >> 2U) & 1U),
      };
      const bool is_same_place =
          cell_corners.corners[corner_number].value == FindCornerPlaceByFullSearch(tables, corner_point).value;
      mistake_count += is_same_place ? 0 : 1;
    }
  }
  return mistake_count;
}

// The search bounded to size level l + 2 gives the places of the full search from the frame's smallest size level, on
// random grids of 6 size levels, 687 to 1,233 cells each (seeds 3 and 4 give a single cell and are left out), and on
// the grids around the origin, 16 size levels among them.
TEST(FindCornerPlaces, ManySizeLevels_BoundedSearchMatchesFullSearch) {
  constexpr int kDeepSizeLevelCount = 6;
  std::vector<NamedGrid> grids = BuildGridsAroundOrigin();
  for (const std::uint64_t seed : {1, 2, 5, 6}) {
    grids.push_back({
        .name = "random grid of 6 size levels",
        .size_level_count = kDeepSizeLevelCount,
        .cells = BuildRandomGridCells(seed, kDeepSizeLevelCount, 1),
    });
  }
  for (const NamedGrid& named_grid : grids) {
    const std::expected<TestGrid, GridFailure> test_grid =
        BuildTestGrid(ToInputs(ToUnitFrame(named_grid.size_level_count), named_grid.cells));
    ASSERT_TRUE(test_grid.has_value()) << named_grid.name;
    EXPECT_EQ(ComputeCornersNotTheFullSearchCount(ToTablesOf(test_grid), named_grid.cells), 0) << named_grid.name;
  }
}

}  // namespace
}  // namespace fluidity::grid
