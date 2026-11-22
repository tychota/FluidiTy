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
#include <set>
#include <span>
#include <tuple>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/grid_support_cases.h"

namespace fluidity::grid {
namespace {

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

// Size-level-0 cell {1,1,1} and its 26 neighbours split into their size-level-1 cells, in a frame of 2 size levels.
std::vector<CellId> BuildAllNeighboursSplit() {
  std::vector<CellId> cells = {ToCell(0, 1, 1, 1)};
  for (std::int32_t z = 0; z < 3; ++z) {
    for (std::int32_t y = 0; y < 3; ++y) {
      for (std::int32_t x = 0; x < 3; ++x) {
        if (x != 1 || y != 1 || z != 1) {
          ApplySmallCellsToList(ToCell(0, x, y, z), cells);
        }
      }
    }
  }
  return cells;
}

// The grids of the checker's digest tests: hand-built grids of 1 to 3 size levels, shuffled random grids, one of them
// in a frame whose origin is away from 0, and random grids moved across zero and to both coordinate limits.
// Keep the moved grids: they are the only tests that place cells at negative coordinates, where a lookup that
// truncates instead of rounding down finds the wrong place beyond a side. The chain scenes of
// HandBuiltScenes_KeepTheirDigest are, likewise, the only tests that see a closure stop a round early.
std::vector<DigestGrid> BuildDigestGrids() {
  std::vector<CellId> at_coord_limits = {ToCell(0, -kMaxCoord, 0, 0), ToCell(0, kMaxCoord, 0, 0)};
  ApplySmallCellsToList(ToCell(0, 0, 0, 0), at_coord_limits);
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

TEST(BuildShuffledRandomGrids, AllEntries_KeepTheirDigest) {
  std::vector<std::uint64_t> words;
  for (const std::vector<CellId>& cells : BuildShuffledRandomGrids()) {
    ApplyCellsToWords(cells, words);
  }
  EXPECT_EQ(ComputeWordsDigest(words), 0x8c4637ae9848f431ULL);
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

}  // namespace
}  // namespace fluidity::grid
