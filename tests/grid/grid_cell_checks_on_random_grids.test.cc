#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_cell_checks_cases.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {
namespace {

// The room of the CPU buffers that every random grid fits: at most 252 cell blocks each.
constexpr std::uint32_t kRandomGridRoom = 4096;

// The list in the fixed order, sorted with ComesBefore: the test's own sort, apart from the sort keys.
std::vector<CellId> ToSortedCells(std::vector<CellId> cells) {
  std::ranges::sort(cells, [](CellId first, CellId second) {
    return ComesBefore(first, second);
  });
  return cells;
}

// floor(coord / 2^size_level_steps) by integer division and a correction for a negative remainder: the test's own
// arithmetic, apart from the shift the grid uses.
//
// clang-tidy: a coordinate and a count of size levels; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::int32_t ToFloorOfHalving(std::int32_t coord, int size_level_steps) {
  const auto divisor = static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(size_level_steps));
  std::int64_t quotient = coord / divisor;
  if (coord % divisor != 0 && coord < 0) {
    quotient -= 1;
  }
  return static_cast<std::int32_t>(quotient);
}

// The big cell, one size level up, that holds `cell`, by the test's own floor.
CellId ToBigCellByDivision(CellId cell) {
  const CellCoord coord = ToCellCoord(cell);
  return ToCell(
      ToSizeLevelNumber(cell) - 1,
      ToFloorOfHalving(coord.x, 1),
      ToFloorOfHalving(coord.y, 1),
      ToFloorOfHalving(coord.z, 1)
  );
}

// Every field of a result, as one comparable value: whether it passed, then the failure's error, cells and magnitude
// (all zero on success).
std::tuple<bool, GridError, std::uint64_t, std::uint64_t, std::uint64_t> ToResultFields(
    const std::expected<void, GridFailure>& result
) {
  const GridFailure failure = result.has_value() ? GridFailure{} : result.error();
  return {result.has_value(), failure.error, failure.cell.value, failure.other_cell.value, failure.magnitude};
}

void ExpectSameResult(
    const std::expected<void, GridFailure>& actual,
    const std::expected<void, GridFailure>& expected
) {
  EXPECT_EQ(ToResultFields(actual), ToResultFields(expected));
}

std::array<std::vector<CellId>, 3> BuildEveryTwoSizeLevelsApartList() {
  return {
      BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 4)),
      BuildTwoSizeLevelsApart(ToCell(2, 4, 4, 0)),
      BuildTwoSizeLevelsApart(ToCell(2, 4, 0, 0)),
  };
}

TEST(CheckSizeStepRule, IntoBuffers_EqualsSpanForm) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(kRandomGridRoom);
  ASSERT_NE(grid_buffers, nullptr);
  for (const std::vector<CellId>& cells : BuildEveryTwoSizeLevelsApartList()) {
    const std::expected<void, GridFailure> span_result = CheckSizeStepRule(ToUnitFrame(3), cells);
    ASSERT_FALSE(span_result.has_value());
    ExpectSameResult(CheckSizeStepRule(ToInputs(ToUnitFrame(3), cells), *grid_buffers->buffers), span_result);
  }
  for (const std::vector<CellId>& cells : BuildShuffledRandomGrids()) {
    const std::expected<void, GridFailure> span_result = CheckSizeStepRule(ToUnitFrame(4), cells);
    EXPECT_TRUE(span_result.has_value());
    ExpectSameResult(CheckSizeStepRule(ToInputs(ToUnitFrame(4), cells), *grid_buffers->buffers), span_result);
  }
}

TEST(BuildSortedCellList, ShuffledList_LeavesFixedCellOrder) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(kRandomGridRoom);
  ASSERT_NE(grid_buffers, nullptr);
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    const std::vector<CellId>& cells = BuildShuffledRandomGrids().at(grid);
    const std::expected<std::uint32_t, GridFailure> built =
        BuildSortedCellList(ToInputs(ToUnitFrame(4), cells), *grid_buffers->buffers, SiblingCheck::kRequire);
    ASSERT_TRUE(built.has_value()) << grid;
    ASSERT_EQ(*built, cells.size()) << grid;
    const std::vector<std::uint64_t> sorted_values = ReadLentCellIds(*grid_buffers->builder_buffers, cells.size());
    EXPECT_EQ(sorted_values, ToValues(ToSortedCells(cells))) << grid;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The random grids, each changed in one way. The expected cells come from the change itself and from whole-unit boxes
// of the test support. Each list is also checked in a second order, ApplyFixedShuffle of seed + 2000.
// ---------------------------------------------------------------------------------------------------------------------

constexpr int kRandomGridSizeLevels = 4;

// The first cell of a list from index `start` on, wrapping around, for which `is_wanted` holds; none when no cell does.
template <typename CellPredicate>
std::optional<std::size_t> FindCellFrom(const std::vector<CellId>& cells, std::size_t start, CellPredicate is_wanted) {
  for (std::size_t step = 0; step < cells.size(); ++step) {
    const std::size_t index = (start + step) % cells.size();
    if (is_wanted(cells.at(index))) {
      return index;
    }
  }
  return std::nullopt;
}

// A cell that has a big cell: size level 1 or more.
bool ComputeHasBigCell(CellId cell) {
  return ToSizeLevelNumber(cell) >= 1;
}

// Says whether a cell has an odd coordinate on some axis, so it is not the eighth-0 small cell of its big cell.
bool ComputeIsAwayFromBigCellCorner(CellId cell) {
  const CellCoord coord = ToCellCoord(cell);
  const auto is_odd = [](std::int32_t axis_coord) {
    return ToFloorOfHalving(axis_coord, 1) * 2 != axis_coord;
  };
  return is_odd(coord.x) || is_odd(coord.y) || is_odd(coord.z);
}

// Checks `cells` and the same cells in a second order with `check`, and expects the failure each time.
template <typename CheckFunction>
void ExpectFailureInAnyOrder(
    std::vector<CellId> cells,
    std::uint64_t seed,
    CheckFunction check,
    GridError error,
    CellId cell,
    CellId other_cell
) {
  SCOPED_TRACE(seed);
  ExpectFailure(check(cells), error, cell, other_cell);
  ApplyFixedShuffle(cells, seed + 2000);
  ExpectFailure(check(cells), error, cell, other_cell);
}

// A cell listed twice is an overlap of that cell with itself, and the only failure of the list.
TEST(CheckSizeStepRule, RandomGridWithRepeatedCell_IsCellsOverlapOfThatCell) {
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    std::vector<CellId> cells = BuildShuffledRandomGrids().at(seed - 1);
    const CellId repeated_cell = cells.at(seed % cells.size());
    cells.push_back(repeated_cell);
    ExpectFailureInAnyOrder(
        cells,
        seed,
        [](const std::vector<CellId>& list) {
          return CheckSizeStepRule(ToUnitFrame(kRandomGridSizeLevels), list);
        },
        GridError::kCellsOverlap,
        repeated_cell,
        repeated_cell
    );
  }
}

// The big cell B of a listed cell is added. Every listed cell inside B overlaps it; in the fixed order (z, then y, then
// x of the low corner) the first of them is the one active cell at B's low corner, found here by whole-unit boxes. B
// itself sorts just before it and holds no overlap, since every cell bigger than B is split, not listed.
TEST(CheckSizeStepRule, RandomGridWithBigCellAdded_IsCellsOverlapAtItsLowCorner) {
  int changed_grid_count = 0;
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    std::vector<CellId> cells = BuildShuffledRandomGrids().at(seed - 1);
    const std::optional<std::size_t> small_index = FindCellFrom(cells, seed % cells.size(), ComputeHasBigCell);
    if (!small_index.has_value()) {
      continue;
    }
    const CellId big_cell = ToBigCellByDivision(cells.at(*small_index));
    const UnitBox big_box = ToUnitBox(big_cell, kRandomGridSizeLevels);
    const auto at_big_corner = std::ranges::find_if(cells, [&big_box](CellId cell) {
      return ToUnitBox(cell, kRandomGridSizeLevels).low == big_box.low;
    });
    ASSERT_NE(at_big_corner, cells.end()) << seed;
    const CellId expected_cell = *at_big_corner;
    cells.push_back(big_cell);
    ExpectFailureInAnyOrder(
        cells,
        seed,
        [](const std::vector<CellId>& list) {
          return CheckSizeStepRule(ToUnitFrame(kRandomGridSizeLevels), list);
        },
        GridError::kCellsOverlap,
        expected_cell,
        big_cell
    );
    ++changed_grid_count;
  }
  EXPECT_GE(changed_grid_count, 190);
}

// Says whether two boxes of cells that do not overlap share at least one point: a side, an edge or a corner. On every
// axis the two ranges then meet or overlap.
bool ComputeIsBoxTouching(const UnitBox& first, const UnitBox& second) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const bool is_meeting = first.low.at(axis) <= second.low.at(axis) + second.width
        && second.low.at(axis) <= first.low.at(axis) + first.width;
    if (!is_meeting) {
      return false;
    }
  }
  return true;
}

// Says whether whole-unit box `outer` holds box `inner`.
bool ComputeIsBoxInside(const UnitBox& inner, const UnitBox& outer) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const bool is_inside = inner.low.at(axis) >= outer.low.at(axis)
        && inner.low.at(axis) + inner.width <= outer.low.at(axis) + outer.width;
    if (!is_inside) {
      return false;
    }
  }
  return true;
}

// The listed cell two or more size levels bigger than `cell` that covers a place of `cell`'s size level beside it, for
// the first direction that has one: (dz, dy, dx), each -1, 0 or +1, in the order dz, then dy, then dx. Found by
// whole-unit boxes over every listed cell; a place lies wholly inside a bigger cell or wholly outside it.
std::optional<CellId> FindSizeStepBreakByBoxes(const std::vector<CellId>& cells, CellId cell) {
  const UnitBox box = ToUnitBox(cell, kRandomGridSizeLevels);
  for (std::int64_t dz = -1; dz <= 1; ++dz) {
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        if (dz == 0 && dy == 0 && dx == 0) {
          continue;
        }
        const UnitBox place = {
            .low =
                {box.low.at(0) + (dx * box.width), box.low.at(1) + (dy * box.width), box.low.at(2) + (dz * box.width)},
            .width = box.width,
        };
        for (const CellId other_cell : cells) {
          const bool is_two_bigger = ToSizeLevelNumber(other_cell) <= ToSizeLevelNumber(cell) - 2;
          if (is_two_bigger && ComputeIsBoxInside(place, ToUnitBox(other_cell, kRandomGridSizeLevels))) {
            return other_cell;
          }
        }
      }
    }
  }
  return std::nullopt;
}

// A cell c of size level 1 or 2 that touches a cell one size level bigger is replaced by its 8 small cells, which
// keeps the siblings together and adds no overlap. Only those small cells can break the rule: every other cell keeps
// its neighbours, or gains smaller ones. The first of them in the fixed order that touches a cell two size levels
// bigger is named, with the bigger cell of its first such direction.
TEST(CheckSizeStepRule, RandomGridWithCellSplitBesideBiggerCell_IsSizeStepRuleBroken) {
  int changed_grid_count = 0;
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    const std::vector<CellId> sorted_cells = ToSortedCells(BuildShuffledRandomGrids().at(seed - 1));
    const auto split_cell = std::ranges::find_if(sorted_cells, [&sorted_cells](CellId cell) {
      const int size_level = ToSizeLevelNumber(cell);
      if (size_level < 1 || size_level > kRandomGridSizeLevels - 2) {
        return false;
      }
      const UnitBox box = ToUnitBox(cell, kRandomGridSizeLevels);
      return std::ranges::any_of(sorted_cells, [size_level, &box](CellId other_cell) {
        const bool is_one_bigger = ToSizeLevelNumber(other_cell) == size_level - 1;
        return is_one_bigger && ComputeIsBoxTouching(box, ToUnitBox(other_cell, kRandomGridSizeLevels));
      });
    });
    if (split_cell == sorted_cells.end()) {
      continue;
    }
    std::vector<CellId> cells;
    std::ranges::copy_if(sorted_cells, std::back_inserter(cells), [split_cell](CellId cell) {
      return cell.value != split_cell->value;
    });
    const std::vector<CellId> small_cells = ToSortedCells(ToSmallCells(*split_cell));
    cells.insert(cells.end(), small_cells.begin(), small_cells.end());
    std::optional<CellId> broken_cell;
    std::optional<CellId> bigger_cell;
    for (const CellId small_cell : small_cells) {
      bigger_cell = FindSizeStepBreakByBoxes(cells, small_cell);
      if (bigger_cell.has_value()) {
        broken_cell = small_cell;
        break;
      }
    }
    if (!broken_cell.has_value() || !bigger_cell.has_value()) {
      ADD_FAILURE() << "no break found for seed " << seed;
      continue;
    }
    ExpectFailureInAnyOrder(
        cells,
        seed,
        [](const std::vector<CellId>& list) {
          return CheckSizeStepRule(ToUnitFrame(kRandomGridSizeLevels), list);
        },
        GridError::kSizeStepRuleBroken,
        *broken_cell,
        *bigger_cell
    );
    ++changed_grid_count;
  }
  EXPECT_GE(changed_grid_count, 190);
}

// The cell and the big cell a missing sibling names when `left_out` is the one cell left out of a valid list, by the
// rule stated at CheckSortedCells: the first cell above it, of size level 1 or more, that is not eighth 0 of its big
// cell, when `left_out` is eighth 0 and such a cell exists; otherwise `left_out` itself. Worked out with the test's own
// halving and parity.
std::pair<CellId, CellId> ToExpectedMissingCells(CellId left_out) {
  if (ComputeIsAwayFromBigCellCorner(left_out)) {
    return {left_out, ToBigCellByDivision(left_out)};
  }
  for (CellId bigger_cell = ToBigCellByDivision(left_out); ComputeHasBigCell(bigger_cell);
       bigger_cell = ToBigCellByDivision(bigger_cell)) {
    if (ComputeIsAwayFromBigCellCorner(bigger_cell)) {
      return {bigger_cell, ToBigCellByDivision(bigger_cell)};
    }
  }
  return {left_out, ToBigCellByDivision(left_out)};
}

bool ComputeIsAtBigCellCorner(CellId cell) {
  return !ComputeIsAwayFromBigCellCorner(cell);
}

// Leaves out of a random grid one cell of size level 1 or more for which `is_wanted` holds, and expects the failure
// ToExpectedMissingCells gives, in two input orders. Returns the cell left out, or none when the grid has none.
std::optional<CellId> ExpectLeftOutCellNamed(std::uint64_t seed, bool (*is_wanted)(CellId), GridBuffers& buffers) {
  std::vector<CellId> cells = BuildShuffledRandomGrids().at(seed - 1);
  const std::optional<std::size_t> left_out_index = FindCellFrom(cells, seed % cells.size(), [is_wanted](CellId cell) {
    return ComputeHasBigCell(cell) && is_wanted(cell);
  });
  if (!left_out_index.has_value()) {
    return std::nullopt;
  }
  const CellId left_out = cells.at(*left_out_index);
  cells.erase(cells.begin() + static_cast<std::ptrdiff_t>(*left_out_index));
  const auto [expected_cell, expected_other_cell] = ToExpectedMissingCells(left_out);
  ExpectFailureInAnyOrder(
      cells,
      seed,
      [&buffers](const std::vector<CellId>& list) {
        return CheckThroughSortedCellList(ToUnitFrame(kRandomGridSizeLevels), list, buffers);
      },
      GridError::kSiblingCellMissing,
      expected_cell,
      expected_other_cell
  );
  return left_out;
}

// One cell of size level 1 or more is left out, twice per grid: once a cell that is eighth 0 of its big cell and once
// one that is not. The named cells follow ToExpectedMissingCells; both kinds, and both answers for eighth-0 cells (the
// cell itself and a bigger cell), must occur.
TEST(BuildSortedCellList, RandomGridMissingOneCell_NamesTheCellByTheEighthZeroRule) {
  const std::unique_ptr<CpuGridBuffers> grid_buffers = BuildCpuGridBuffers(kRandomGridRoom);
  ASSERT_NE(grid_buffers, nullptr);
  // Counts: cells away from the big cell's corner left out, eighth-0 cells left out, and of those how many name the
  // cell itself and how many a bigger cell.
  std::array<int, 4> counts = {};
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    const std::optional<CellId> away_left_out =
        ExpectLeftOutCellNamed(seed, ComputeIsAwayFromBigCellCorner, *grid_buffers->buffers);
    counts.at(0) += static_cast<int>(away_left_out.has_value());
    const std::optional<CellId> corner_left_out =
        ExpectLeftOutCellNamed(seed, ComputeIsAtBigCellCorner, *grid_buffers->buffers);
    if (corner_left_out.has_value()) {
      ++counts.at(1);
      const bool is_named_itself = ToExpectedMissingCells(*corner_left_out).first.value == corner_left_out->value;
      ++counts.at(is_named_itself ? 2 : 3);
    }
  }
  const bool has_every_kind = counts.at(0) >= 190 && counts.at(1) >= 190 && counts.at(2) >= 1 && counts.at(3) >= 1;
  EXPECT_TRUE(has_every_kind) << counts.at(0) << " " << counts.at(1) << " " << counts.at(2) << " " << counts.at(3);
}

}  // namespace
}  // namespace fluidity::grid
