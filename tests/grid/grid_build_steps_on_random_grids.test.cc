#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fluidity/base/buffer.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_build_steps_cases.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/thread_replay.h"

namespace fluidity::grid {
namespace {

// The 200 random grids of the test support, each list sorted into the fixed order. Built once for the whole file,
// since several tests read them; seed s is entry s - 1.
const std::vector<std::vector<CellId>>& BuildSortedRandomGrids() {
  static const std::vector<std::vector<CellId>> sorted_grids = [] {
    std::vector<std::vector<CellId>> grids;
    grids.reserve(kRandomGridCount);
    for (const std::vector<CellId>& shuffled_cells : BuildShuffledRandomGrids()) {
      grids.push_back(ToSortedCells(shuffled_cells));
    }
    return grids;
  }();
  return sorted_grids;
}

const std::vector<CellId>& BuildSortedRandomGrid(std::uint64_t seed) {
  return BuildSortedRandomGrids().at(seed - 1);
}

TEST(ToBlockSortKey, ThreePasses_GiveComesBeforeOrderOfBlocks) {
  std::vector<CellId> cells = BuildRandomCellIds(20261115, 15, 10000);
  const std::vector<CellId> limit_cells = BuildEverySizeLevelAtCoordLimits();
  cells.insert(cells.end(), limit_cells.begin(), limit_cells.end());
  const std::vector<std::size_t> order = ToStableOrderByPasses(cells, ToBlockSortKey);
  std::vector<std::uint64_t> blocks_by_keys;
  blocks_by_keys.reserve(order.size());
  std::vector<CellBlockId> blocks;
  blocks.reserve(cells.size());
  for (const std::size_t index : order) {
    blocks_by_keys.push_back(ToCellBlockId(cells[index]).value);
  }
  for (const CellId cell : cells) {
    blocks.push_back(ToCellBlockId(cell));
  }
  std::ranges::sort(blocks, [](CellBlockId first, CellBlockId second) {
    return ComesBefore(first, second);
  });
  std::vector<std::uint64_t> sorted_blocks;
  sorted_blocks.reserve(blocks.size());
  for (const CellBlockId block : blocks) {
    sorted_blocks.push_back(block.value);
  }
  EXPECT_EQ(blocks_by_keys, sorted_blocks);
}

TEST(CheckOverlapAt, RandomGrids_Pass) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<CellId>& sorted_cells = BuildSortedRandomGrid(seed);
    EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 2, CheckOverlapAt), kNoFailureKey) << seed;
  }
}

TEST(CheckSiblingsAt, RandomGrids_Pass) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<CellId>& sorted_cells = BuildSortedRandomGrid(seed);
    EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 3, CheckSiblingsAt), kNoFailureKey) << seed;
  }
}

// Says whether an active cell lies inside `cell` or is `cell`, by unit boxes. With kSameLowCorner, only an active cell
// whose low corner is that of `cell` counts: one at the end of the eighth-0 chain of `cell`.
enum class CornerRule : std::uint8_t {
  kAnyCorner,
  kSameLowCorner,
};

bool ComputeHoldsActiveCell(
    const std::vector<CellId>& active_cells,
    CellId cell,
    int size_level_count,
    CornerRule corner_rule
) {
  const UnitBox outer_box = ToUnitBox(cell, size_level_count);
  return std::ranges::any_of(active_cells, [&](CellId active_cell) {
    const UnitBox inner_box = ToUnitBox(active_cell, size_level_count);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const bool is_inside = inner_box.low.at(axis) >= outer_box.low.at(axis)
          && inner_box.low.at(axis) + inner_box.width <= outer_box.low.at(axis) + outer_box.width;
      const bool is_corner_kept =
          corner_rule == CornerRule::kAnyCorner || inner_box.low.at(axis) == outer_box.low.at(axis);
      if (!is_inside || !is_corner_kept) {
        return false;
      }
    }
    return true;
  });
}

// A named cell has no active cell at the end of its eighth-0 chain, while its big cell holds one. When a removed cell
// ended the chain of a split cell, that split cell is named too: its chain is broken.
void CheckNamedCellAbsent(const std::vector<CellId>& cells, const CellCheck& cell_check, std::uint64_t seed) {
  EXPECT_EQ(cell_check.error, GridError::kSiblingCellMissing);
  EXPECT_FALSE(ComputeHoldsActiveCell(cells, cell_check.cell, 4, CornerRule::kSameLowCorner)) << seed;
  EXPECT_TRUE(ComputeHoldsActiveCell(cells, cell_check.other_cell, 4, CornerRule::kAnyCorner)) << seed;
}

// Leaves out of one random grid its first cell of size level 1 or more (a size-level-0 cell has no siblings to miss)
// and checks the siblings at every position.
void CheckMissingCellNamedOnlyWithAbsentCells(std::uint64_t seed) {
  std::vector<CellId> cells = BuildSortedRandomGrid(seed);
  const auto removed = std::ranges::find_if(cells, [](CellId cell) {
    return ToSizeLevel(cell) != SizeLevel{0};
  });
  if (removed == cells.end()) {
    return;
  }
  const CellId removed_cell = *removed;
  cells.erase(removed);
  bool is_removed_cell_named = false;
  for (std::uint32_t position = 0; position < ToCount(cells); ++position) {
    const CellCheck cell_check = CheckSiblingsAt(cells.data(), ToCount(cells), position);
    if (cell_check.error == GridError::kNone) {
      continue;
    }
    CheckNamedCellAbsent(cells, cell_check, seed);
    is_removed_cell_named = is_removed_cell_named || cell_check.cell.value == removed_cell.value;
  }
  EXPECT_TRUE(is_removed_cell_named) << seed;
}

TEST(CheckSiblingsAt, RandomGridMissingOneCell_NamesTheCellAndOnlyAbsentCells) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    CheckMissingCellNamedOnlyWithAbsentCells(seed);
  }
}

TEST(CheckSizeStepAt, RandomGrids_Pass) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<CellId>& sorted_cells = BuildSortedRandomGrid(seed);
    EXPECT_EQ(ComputeSmallestFailureKey(sorted_cells, 4, CheckSizeStepAt), kNoFailureKey) << seed;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Split cells: method paper <proc-build-grid> step 3, <proc-build-grid-in-parallel> step P7
// ---------------------------------------------------------------------------------------------------------------------

// The split cells of a valid list, by brute force: every cell of a bigger size level whose unit box holds an active
// cell's low corner. The random grids lie in 0..3 m, so every unit is 0 or more and plain division is the floor.
std::set<std::uint64_t> BuildSplitCellsByUnitBoxes(const std::vector<CellId>& cells, int size_level_count) {
  std::set<std::uint64_t> split_cells;
  for (const CellId cell : cells) {
    const UnitBox box = ToUnitBox(cell, size_level_count);
    for (int bigger_size_level = 0; bigger_size_level < static_cast<int>(ToSizeLevel(cell)); ++bigger_size_level) {
      const auto width_exponent = static_cast<unsigned>(size_level_count - 1 - bigger_size_level);
      const auto width = static_cast<std::int64_t>(std::uint64_t{1} << width_exponent);
      const CellId split_cell = ToCell(
          bigger_size_level,
          static_cast<std::int32_t>(box.low.at(0) / width),
          static_cast<std::int32_t>(box.low.at(1) / width),
          static_cast<std::int32_t>(box.low.at(2) / width)
      );
      split_cells.insert(split_cell.value);
    }
  }
  return split_cells;
}

TEST(ComputeSplitCellCountToAdd, RandomGrids_AddEachSplitCellOnce) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<CellId>& cells = BuildSortedRandomGrid(seed);
    std::vector<std::uint64_t> added_cells;
    for (const CellId cell : cells) {
      const int size_level = static_cast<int>(ToSizeLevel(cell));
      const int added_count = ComputeSplitCellCountToAdd(cell);
      for (int size_level_steps = 1; size_level_steps <= added_count; ++size_level_steps) {
        added_cells.push_back(
            ComputeBigCellAtSizeLevel(cell, static_cast<SizeLevel>(size_level - size_level_steps)).value
        );
      }
    }
    const std::set<std::uint64_t> distinct_added_cells(added_cells.begin(), added_cells.end());
    EXPECT_EQ(distinct_added_cells.size(), added_cells.size()) << seed;
    EXPECT_EQ(distinct_added_cells, BuildSplitCellsByUnitBoxes(cells, 4)) << seed;
    // Every split cell has 8 small cells, active or split, so the split cells are at most n / 7 of n active cells:
    // the bound the sort count of method paper <sec-parallel-build> rests on.
    EXPECT_LE(added_cells.size(), cells.size() / 7) << seed;
  }
}

// The exclusive prefix sums of ComputeSplitCellCountToAdd over `cells`, the total at the end, by the test's own loop.
std::vector<std::uint32_t> ToSplitCellSums(const std::vector<CellId>& cells) {
  std::vector<std::uint32_t> sums(cells.size() + 1, 0);
  for (std::size_t position = 0; position < cells.size(); ++position) {
    sums[position + 1] = sums[position] + static_cast<std::uint32_t>(ComputeSplitCellCountToAdd(cells[position]));
  }
  return sums;
}

// The cell ids after ApplySplitCellWritesAt ran for every position of `positions`, one after the other: `cells`, then
// room for the split cells and two places more, all first holding the stale id.
//
// clang-tidy: the sums and the positions; the swappable-parameters check flags the two lists of integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::vector<CellId> ComputeCellIdsWithSplitCells(
    const std::vector<CellId>& cells,
    const std::vector<std::uint32_t>& sums,
    const std::vector<std::uint32_t>& positions
) {
  std::vector<CellId> cell_ids(cells.size() + sums.back() + 2, ToCell(3, 1000, 1000, 1000));
  std::ranges::copy(cells, cell_ids.begin());
  for (const std::uint32_t position : positions) {
    ApplySplitCellWritesAt(cell_ids.data(), sums.data(), ToCount(cells), position);
  }
  return cell_ids;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Expects `cell_ids` to hold `cells`, then each split cell of `cells` once, then the two stale ids.
void CheckSplitCellsAppended(
    const std::vector<CellId>& cell_ids,
    const std::vector<CellId>& cells,
    std::uint64_t seed
) {
  const auto active_end = cell_ids.begin() + static_cast<std::ptrdiff_t>(cells.size());
  const auto used_end = cell_ids.end() - 2;
  const std::vector<std::uint64_t> appended_values = ToValues(std::vector<CellId>(active_end, used_end));
  const std::set<std::uint64_t> distinct_values(appended_values.begin(), appended_values.end());
  EXPECT_EQ(distinct_values.size(), appended_values.size()) << seed;
  EXPECT_EQ(distinct_values, BuildSplitCellsByUnitBoxes(cells, 4)) << seed;
  EXPECT_EQ(ToValues(std::vector<CellId>(cell_ids.begin(), active_end)), ToValues(cells)) << seed;
  const std::vector<std::uint64_t> stale_values = {
      ToCell(3, 1000, 1000, 1000).value,
      ToCell(3, 1000, 1000, 1000).value,
  };
  EXPECT_EQ(ToValues(std::vector<CellId>(used_end, cell_ids.end())), stale_values) << seed;
}

// Puts positions in an order drawn from `seed`, the same on every machine: the shuffle of ApplyFixedShuffle.
void ApplyFixedPositionShuffle(std::vector<std::uint32_t>& positions, std::uint64_t seed) {
  std::mt19937_64 random_numbers(seed);
  for (std::size_t index = positions.size(); index > 1; --index) {
    std::swap(positions[index - 1], positions[ToUniformIndex(random_numbers(), index)]);
  }
}

// Each position writes its own places, so the threads of P7 (c) may run in any order: here first to last, last to
// first and one seeded shuffle per grid. The places after the split cells keep the stale id.
TEST(ApplySplitCellWritesAt, RandomGrids_AppendExactlyTheSplitCells) {
  for (std::uint64_t seed = 1; seed <= kRandomGridCount; ++seed) {
    const std::vector<CellId>& cells = BuildSortedRandomGrid(seed);
    const std::vector<std::uint32_t> sums = ToSplitCellSums(cells);
    std::vector<std::uint32_t> positions(cells.size());
    for (std::uint32_t position = 0; position < ToCount(cells); ++position) {
      positions[position] = position;
    }
    CheckSplitCellsAppended(ComputeCellIdsWithSplitCells(cells, sums, positions), cells, seed);
    std::ranges::reverse(positions);
    CheckSplitCellsAppended(ComputeCellIdsWithSplitCells(cells, sums, positions), cells, seed);
    ApplyFixedPositionShuffle(positions, seed + 2000);
    CheckSplitCellsAppended(ComputeCellIdsWithSplitCells(cells, sums, positions), cells, seed);
  }
}

// The tables of the 200 random grids, built once for the whole file; seed s is entry s - 1.
const std::vector<TableVectors>& BuildRandomGridTables() {
  static const std::vector<TableVectors> grid_tables = [] {
    std::vector<TableVectors> tables;
    tables.reserve(200);
    for (std::uint64_t seed = 1; seed <= 200; ++seed) {
      tables.push_back(BuildTablesOfCells(ToUnitFrame(4), BuildSortedRandomGrid(seed)));
    }
    return tables;
  }();
  return grid_tables;
}

// The index of a block in sorted block ids by a map, or kNoIndex: a lookup apart from the binary search.
class BlockLookup {
 public:
  explicit BlockLookup(const std::vector<CellBlockId>& block_ids) {
    for (std::size_t block = 0; block < block_ids.size(); ++block) {
      block_index_by_id_[block_ids.at(block).value] = static_cast<std::uint32_t>(block);
    }
  }

  [[nodiscard]] std::uint32_t FindBlock(int size_level, CellCoord block_coord) const {
    const auto found = block_index_by_id_.find(ToBlock(size_level, block_coord.x, block_coord.y, block_coord.z).value);
    return found == block_index_by_id_.end() ? kNoIndex : found->second;
  }

 private:
  std::map<std::uint64_t, std::uint32_t> block_index_by_id_;
};

// Expects the 27 same-size-level entries of one block to be the blocks the lookup finds at b + (dx, dy, dz).
void CheckSameSizeLevelEntriesFound(const TableVectors& tables, const BlockLookup& lookup, std::uint32_t block) {
  const CellCoord block_coord = ToBlockCoord(tables.block_ids.at(block));
  const int size_level = ToSizeLevelNumber(tables.block_ids.at(block));
  std::uint32_t entry = 0;
  for (std::int32_t dz = -1; dz <= 1; ++dz) {
    for (std::int32_t dy = -1; dy <= 1; ++dy) {
      for (std::int32_t dx = -1; dx <= 1; ++dx) {
        const CellCoord neighbour_coord = {.x = block_coord.x + dx, .y = block_coord.y + dy, .z = block_coord.z + dz};
        EXPECT_EQ(
            tables.same_size_level_blocks.at((block * kSameSizeLevelEntriesPerBlock) + entry),
            lookup.FindBlock(size_level, neighbour_coord)
        ) << block
          << " " << entry;
        ++entry;
      }
    }
  }
}

// Expects the bigger block of one block to be the lookup's at floor(b / 2), size level l - 1, and none at size
// level 0; and its smaller block of eighth e the lookup's at 2 b + (e % 2, (e / 2) % 2, e / 4), size level l + 1.
void CheckBiggerAndSmallerEntriesFound(const TableVectors& tables, const BlockLookup& lookup, std::uint32_t block) {
  const CellCoord block_coord = ToBlockCoord(tables.block_ids.at(block));
  const int size_level = ToSizeLevelNumber(tables.block_ids.at(block));
  std::uint32_t expected_bigger = kNoIndex;
  if (size_level >= 1) {
    const CellCoord bigger_coord = {
        .x = ToFloorOfHalving(block_coord.x, 1),
        .y = ToFloorOfHalving(block_coord.y, 1),
        .z = ToFloorOfHalving(block_coord.z, 1),
    };
    expected_bigger = lookup.FindBlock(size_level - 1, bigger_coord);
  }
  EXPECT_EQ(tables.bigger_block.at(block), expected_bigger) << block;
  for (std::int32_t cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    const CellCoord smaller_coord = {
        .x = (2 * block_coord.x) + (cell_eighth % 2),
        .y = (2 * block_coord.y) + ((cell_eighth / 2) % 2),
        .z = (2 * block_coord.z) + (cell_eighth / 4),
    };
    EXPECT_EQ(
        tables.smaller_blocks.at((block * kSmallerBlocksPerBlock) + static_cast<std::uint32_t>(cell_eighth)),
        lookup.FindBlock(size_level + 1, smaller_coord)
    ) << block
      << " " << cell_eighth;
  }
}

TEST(FindBlockNeighbours, RandomGrids_MatchADirectLookup) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    SCOPED_TRACE(seed);
    const TableVectors& tables = BuildRandomGridTables().at(seed - 1);
    const BlockLookup lookup(tables.block_ids);
    for (std::uint32_t block = 0; block < tables.block_ids.size(); ++block) {
      CheckSameSizeLevelEntriesFound(tables, lookup, block);
      CheckBiggerAndSmallerEntriesFound(tables, lookup, block);
    }
  }
}

TEST(ComputeCoveredCellState, RandomGrids_MatchActiveCellsOneSizeLevelBigger) {
  for (std::uint64_t seed = 1; seed <= 200; ++seed) {
    const std::vector<std::uint64_t> active_list = ToValues(BuildSortedRandomGrid(seed));
    const std::set<std::uint64_t> active_values(active_list.begin(), active_list.end());
    const std::set<std::uint64_t> split_cells = BuildSplitCells(BuildSortedRandomGrid(seed));
    const TableVectors& tables = BuildRandomGridTables().at(seed - 1);
    for (std::uint32_t place = 0; place < tables.cell_states.size(); ++place) {
      const CellBlockId block = tables.block_ids.at(place / 64);
      const int size_level = ToSizeLevelNumber(block);
      const CellCoord block_coord = ToBlockCoord(block);
      const auto place_in_block = static_cast<std::int32_t>(place % 64);
      const CellId cell = ToCell(
          size_level,
          (4 * block_coord.x) + (place_in_block % 4),
          (4 * block_coord.y) + ((place_in_block / 4) % 4),
          (4 * block_coord.z) + (place_in_block / 16)
      );
      CellState expected_state = CellState::kAbsent;
      if (active_values.contains(cell.value)) {
        expected_state = CellState::kActive;
      } else if (split_cells.contains(cell.value)) {
        expected_state = CellState::kSplit;
      } else if (size_level >= 1 && active_values.contains(ToBigCellByDivision(cell, size_level - 1).value)) {
        expected_state = CellState::kInsideBiggerCell;
      }
      EXPECT_EQ(tables.cell_states.at(place), expected_state) << seed << " " << place;
    }
  }
}

TEST(ComputeFacePieceKind, RandomGrids_GiveTheCheckersFacePieces) {
  // 40 of the 200 grids: the checker takes most of this file's time, and the grid's build repeats the comparison on
  // the built grid for all 200.
  for (std::uint64_t seed = 1; seed <= 40; ++seed) {
    const TableVectors& tables = BuildRandomGridTables().at(seed - 1);
    EXPECT_EQ(BuildFacePiecesOfTables(tables), BuildFacePiecesOfChecker(ToUnitFrame(4), BuildSortedRandomGrid(seed)))
        << seed;
  }
}

// A list moved by `shift` size-level-0 cells on every axis: a cell of size level l moves by shift 2^l of its own.
std::vector<CellId> ToShiftedCells(const std::vector<CellId>& cells, std::int32_t shift) {
  std::vector<CellId> shifted_cells;
  shifted_cells.reserve(cells.size());
  for (const CellId cell : cells) {
    const int size_level = static_cast<int>(ToSizeLevel(cell));
    const std::int32_t cell_shift = shift * ComputePowerOfTwo(size_level);
    const CellCoord coord = ToCellCoord(cell);
    shifted_cells.push_back(ToCell(size_level, coord.x + cell_shift, coord.y + cell_shift, coord.z + cell_shift));
  }
  return ToSortedCells(shifted_cells);
}

TEST(ComputeFacePieceKind, RandomGridsAcrossZero_GiveTheCheckersFacePieces) {
  // Moved by -2 size-level-0 cells, the box 0..3 becomes -2..1 on every axis: blocks of negative and positive
  // coordinates, with the block side at 0 inside the box at every size level.
  for (std::uint64_t seed = 1; seed <= 20; ++seed) {
    const std::vector<CellId> cells = ToShiftedCells(BuildSortedRandomGrid(seed), -2);
    const TableVectors tables = BuildTablesOfCells(ToUnitFrame(4), cells);
    EXPECT_EQ(BuildFacePiecesOfTables(tables), BuildFacePiecesOfChecker(ToUnitFrame(4), cells)) << seed;
  }
}

// Replays P1 to P6 and the failure half of P12, the GPU path of BuildSortedCellList, as ApplyListCheckStepsByReplay
// says. Returns ToGridFailure of the status record it wrote.
GridFailure ApplyGpuStepsByReplay(
    GridFrame frame,
    std::span<const CellId> input_cells,
    SiblingCheck sibling_check,
    ReplayLaunch launch,
    ReplayArrays& arrays
) {
  if (!ApplyListCheckStepsByReplay(frame, input_cells, sibling_check, launch, arrays)) {
    return {.error = GridError::kInternalFault};
  }
  // P12, the failure half, one thread; then the CPU half's reading of the record.
  const auto count = static_cast<std::uint32_t>(input_cells.size());
  ApplyFailureCells(frame, input_cells.data(), count, arrays.cell_ids.data(), count, &arrays.status);
  return ToGridFailure(arrays.status);
}

// What one replay gave: the failure and the first n cell ids.
struct ReplayResult {
  GridFailure failure;
  std::vector<std::uint64_t> sorted_values;
};

// Replays the GPU check of `cells` in both launch orders, each after ApplyStaleValues, and returns the results in the
// order of kLaunchOrders.
std::array<ReplayResult, 2> ComputeReplaysInBothOrders(
    GridFrame frame,
    std::span<const CellId> cells,
    SiblingCheck sibling_check,
    std::size_t thread_count,
    ReplayArrays& arrays
) {
  std::array<ReplayResult, 2> results;
  for (std::size_t order_index = 0; order_index < kLaunchOrders.size(); ++order_index) {
    ApplyStaleValues(arrays);
    const ReplayLaunch launch = {.order = kLaunchOrders.at(order_index), .thread_count = thread_count};
    results.at(order_index).failure = ApplyGpuStepsByReplay(frame, cells, sibling_check, launch, arrays);
    results.at(order_index).sorted_values =
        ToValues(std::vector<CellId>(arrays.cell_ids.data(), arrays.cell_ids.data() + cells.size()));
  }
  return results;
}

// Says whether a failure is one of step 1, the ids: the replay's sorted list is then never used and is not compared.
bool ComputeIsIdFailure(const GridFailure& failure) {
  return failure.error == GridError::kBadInput || failure.error == GridError::kCoordOutOfRange;
}

// Expects each replay to give cpu_failure and, unless the ids failed, expected_sorted. `context` names the list and
// the sibling check.
void CheckReplaysEqualCpu(
    const std::array<ReplayResult, 2>& replays,
    const GridFailure& cpu_failure,
    const std::vector<std::uint64_t>& expected_sorted,
    const std::string& context
) {
  for (std::size_t order_index = 0; order_index < replays.size(); ++order_index) {
    const std::string order_context = context + (order_index == 0 ? ", forward" : ", reverse");
    EXPECT_EQ(ToFailureFields(replays.at(order_index).failure), ToFailureFields(cpu_failure)) << order_context;
    if (!ComputeIsIdFailure(cpu_failure)) {
      EXPECT_EQ(replays.at(order_index).sorted_values, expected_sorted) << order_context;
    }
  }
}

// Replays the GPU check of `cells` for each of `sibling_checks`, in both launch orders and with `thread_count` threads
// (0 for the GPU's own), on the replay's own cell ids, then runs BuildSortedCellList on the CPU buffers. Expects the
// same failure, and, unless the ids failed, the sorted list of ComesBefore, which is also the CPU check's when it
// passes. `name` says which list failed.
void CheckReplayEqualsCpu(
    GridFrame frame,
    std::span<const CellId> cells,
    std::span<const SiblingCheck> sibling_checks,
    std::size_t thread_count,
    GridBuffers& buffers,
    ReplayArrays& arrays,
    const std::string& name
) {
  const Buffer<CellId>& cpu_cell_ids =
      ToGridBuilderBuffersArrays(*ToGridBuffersArrays(buffers).builder_buffers).cell_ids;
  const GridInputs inputs = {.frame = frame, .active_cells = cells, .version = {.value = 1}};
  const std::vector<std::uint64_t> expected_sorted = ToValues(ToSortedCells({cells.begin(), cells.end()}));
  for (const SiblingCheck sibling_check : sibling_checks) {
    const std::array<ReplayResult, 2> replays =
        ComputeReplaysInBothOrders(frame, cells, sibling_check, thread_count, arrays);
    const std::expected<std::uint32_t, GridFailure> cpu_result = BuildSortedCellList(inputs, buffers, sibling_check);
    const std::string context = name + (sibling_check == SiblingCheck::kRequire ? ", siblings" : ", no siblings");
    if (cpu_result.has_value()) {
      const std::vector<CellId> cpu_sorted(cpu_cell_ids.data(), cpu_cell_ids.data() + *cpu_result);
      EXPECT_EQ(ToValues(cpu_sorted), expected_sorted) << context;
    }
    CheckReplaysEqualCpu(
        replays,
        cpu_result.has_value() ? GridFailure{} : cpu_result.error(),
        expected_sorted,
        context
    );
  }
}

// Size level 0 {0,0,0} and a size-level-2 cell beyond its high side that touches it at a corner, along an edge and on
// a side: the three lists two size levels apart of the CPU check's tests.
std::array<std::vector<CellId>, 3> BuildEveryTwoSizeLevelsApartList() {
  return {
      std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(2, 4, 4, 4)},
      std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(2, 4, 4, 0)},
      std::vector<CellId>{ToCell(0, 0, 0, 0), ToCell(2, 4, 0, 0)},
  };
}

// The mutated lists of the replay: seeds 1..kReplayedMutatedGridCount. They hold every combination of earliest failing
// step, sibling check and one or several failing items in that step that seeds 1..500 hold, so the other 400 would
// add about 5 s to a debug run and no new case; the gpu tests check all 500 on hardware.
constexpr std::uint64_t kReplayedMutatedGridCount = 100;

// Adds to errors_seen the error the CPU check gives `cells` with each sibling check, kNone when it passes.
void ApplyCpuErrors(
    GridFrame frame,
    std::span<const CellId> cells,
    GridBuffers& buffers,
    std::set<GridError>& errors_seen
) {
  const GridInputs inputs = {.frame = frame, .active_cells = cells, .version = {.value = 1}};
  for (const SiblingCheck sibling_check : kSiblingChecks) {
    const std::expected<std::uint32_t, GridFailure> cpu_result = BuildSortedCellList(inputs, buffers, sibling_check);
    errors_seen.insert(cpu_result.has_value() ? GridError::kNone : cpu_result.error().error);
  }
}

// The random grids are valid: no item offers a key, so they show the sorted list and that no per-item rule fails, with
// siblings required, which runs every step that skipping them runs, and P5 too. The lists that fail, two size levels
// apart, BuildSizeStepBreakWithSiblings (siblings pass, then the size step rule fails) and mutated, show the first
// failure, with and without siblings, where the order of the threads could matter.
TEST(BuildSortedCellList, EveryThreadBlockAndThread_SameFirstFailureAsCpu) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  constexpr std::array<SiblingCheck, 1> kSiblingsRequired = {SiblingCheck::kRequire};
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    const std::vector<CellId>& cells = BuildShuffledRandomGrids().at(grid);
    CheckReplayEqualsCpu(
        ToUnitFrame(4),
        cells,
        kSiblingsRequired,
        0,
        buffers,
        arrays,
        "random grid " + std::to_string(grid)
    );
  }
  for (const std::vector<CellId>& cells : BuildEveryTwoSizeLevelsApartList()) {
    CheckReplayEqualsCpu(ToUnitFrame(3), cells, kSiblingChecks, 0, buffers, arrays, "two size levels apart");
  }
  CheckReplayEqualsCpu(
      ToUnitFrame(3),
      BuildSizeStepBreakWithSiblings(),
      kSiblingChecks,
      0,
      buffers,
      arrays,
      "size step break with siblings"
  );
  std::set<GridError> errors_seen;
  for (std::uint64_t seed = 1; seed <= kReplayedMutatedGridCount; ++seed) {
    const std::vector<CellId> cells = BuildMutatedGridCells(seed);
    CheckReplayEqualsCpu(
        ToUnitFrame(4),
        cells,
        kSiblingChecks,
        0,
        buffers,
        arrays,
        "mutated list " + std::to_string(seed)
    );
    ApplyCpuErrors(ToUnitFrame(4), cells, buffers, errors_seen);
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
}

// The size-level-2 cell {188,188,188} touches the box's last cell, size level 0 {46,46,46}, at the corner (47, 47, 47)
// m: sorted last, it fails the size step rule, or first its siblings, at position 103,823. Each list runs with the
// GPU's own launch, one item per thread, and with 1,000 threads, about 104 items per thread, the shape the GPU takes
// for a list of more than 65,535 x 256 cells.
TEST(BuildSortedCellList, LargeListAfterStaleValues_SameFirstFailureAsCpu) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  const std::vector<CellId> failing_cells = BuildLargeBoxOfCells({ToCell(2, 188, 188, 188)});
  const std::vector<CellId> valid_cells = BuildLargeBoxOfCells({});
  for (const std::vector<CellId>* cells : {&failing_cells, &valid_cells}) {
    for (const std::size_t thread_count : {std::size_t{0}, std::size_t{1000}}) {
      CheckReplayEqualsCpu(
          ToUnitFrame(3),
          *cells,
          kSiblingChecks,
          thread_count,
          buffers,
          arrays,
          "large box of " + std::to_string(cells->size()) + ", threads " + std::to_string(thread_count)
      );
    }
  }
  // The failure is the one the CPU names, not the stale record's.
  const std::expected<std::uint32_t, GridFailure> skipped = BuildSortedCellList(
      {.frame = ToUnitFrame(3), .active_cells = failing_cells, .version = {.value = 1}},
      buffers,
      SiblingCheck::kSkip
  );
  ASSERT_FALSE(skipped.has_value());
  EXPECT_EQ(skipped.error().error, GridError::kSizeStepRuleBroken);
  EXPECT_EQ(skipped.error().cell.value, ToCell(2, 188, 188, 188).value);
}

// With 1 and 7 threads every thread has several items, so a thread of P4 to P6 returns after an item of its own when
// an earlier step failed: the lists that fail, in both launch orders.
TEST(BuildSortedCellList, SeveralItemsPerThread_SameFirstFailureAsCpu) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  for (const std::size_t thread_count : {std::size_t{1}, std::size_t{7}}) {
    const std::string threads = ", threads " + std::to_string(thread_count);
    for (const std::vector<CellId>& cells : BuildEveryTwoSizeLevelsApartList()) {
      CheckReplayEqualsCpu(ToUnitFrame(3), cells, kSiblingChecks, thread_count, buffers, arrays, "apart" + threads);
    }
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
      CheckReplayEqualsCpu(
          ToUnitFrame(4),
          BuildMutatedGridCells(seed),
          kSiblingChecks,
          thread_count,
          buffers,
          arrays,
          "mutated list " + std::to_string(seed) + threads
      );
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The build on the GPU up to the block ids (<proc-build-grid-in-parallel> P1 to P8), replayed on the CPU
// ---------------------------------------------------------------------------------------------------------------------

// Which arrays a build replay starts from.
enum class ReplayStart : std::uint8_t {
  kAfterStaleValues,  // ApplyStaleValues first.
  kAfterPreviousList,  // The arrays as the replay of the list before left them, cells of another grid included.
};

// Replays the build of `cells` in both launch orders with `thread_count` threads (0 for the GPU's own), the forward
// order from `start` and the reverse one after ApplyStaleValues, then runs the CPU build (ComputeCpuBuildOfList), and
// expects each replay to equal it (CheckBuildReplayEqualsCpuBuild). On success the status block count is the CPU's.
//
// clang-tidy: a room and a thread count; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void CheckBuildReplayEqualsCpu(
    GridFrame frame,
    std::span<const CellId> cells,
    std::size_t thread_count,
    ReplayStart start,
    GridBuffers& buffers,
    ReplayArrays& arrays,
    const std::string& name
) {
  const GridInputs inputs = {.frame = frame, .active_cells = cells, .version = {.value = 1}};
  std::array<BuildReplayResult, 2> replays;
  for (std::size_t order_index = 0; order_index < kLaunchOrders.size(); ++order_index) {
    if (order_index == 1 || start == ReplayStart::kAfterStaleValues) {
      ApplyStaleValues(arrays);
    }
    const ReplayLaunch launch = {.order = kLaunchOrders.at(order_index), .thread_count = thread_count};
    replays.at(order_index) = ComputeBuildReplay(inputs, buffers, launch, ReplayChange::kNone, arrays);
  }
  const CpuBuildOfList cpu_build = ComputeCpuBuildOfList(inputs, buffers);
  CheckBuildReplayEqualsCpuBuild(replays.at(0), cpu_build, name + ", forward");
  CheckBuildReplayEqualsCpuBuild(replays.at(1), cpu_build, name + ", reverse");
}

// Replays the full build of random grid 1 into `arrays` after ApplyStaleValues, then the build of the 14 lone cells
// {4 i, 0, 0} with `change`, both on the room of `buffers`, and returns the second.
BuildReplayResult ComputeLoneCellsAfterGridOne(ReplayChange change, const GridBuffers& buffers, ReplayArrays& arrays) {
  const GridInputs grid_one_inputs = {
      .frame = ToUnitFrame(4),
      .active_cells = BuildShuffledRandomGrids().at(0),
      .version = {.value = 1},
  };
  const std::vector<CellId> lone_cells = BuildLoneCellsAlongX(14);
  const GridInputs lone_inputs = {.frame = ToUnitFrame(4), .active_cells = lone_cells, .version = {.value = 1}};
  const ReplayLaunch launch = {.order = LaunchOrder::kForward, .thread_count = 0};
  ApplyStaleValues(arrays);
  const BuildReplayResult grid_one_build =
      ComputeBuildReplay(grid_one_inputs, buffers, launch, ReplayChange::kNone, arrays);
  EXPECT_EQ(grid_one_build.failure.error, GridError::kNone);
  return ComputeBuildReplay(lone_inputs, buffers, launch, change, arrays);
}

// Expects places 14 and 15 of the lone cells' build to hold cell ids just before step (d) and the id 0 after it.
void CheckTailStaleThenFilled(const ReplayArrays& arrays) {
  ASSERT_EQ(arrays.cells_before_tail_fill.size(), 16U);
  const std::array<std::uint64_t, 2> cell_marks_before = {
      arrays.cells_before_tail_fill.at(14).value & kIdCellMark,
      arrays.cells_before_tail_fill.at(15).value & kIdCellMark,
  };
  EXPECT_EQ(cell_marks_before, (std::array<std::uint64_t, 2>{kIdCellMark, kIdCellMark}));
  const std::array<std::uint64_t, 2> values_after = {
      arrays.cell_ids.data()[14].value,
      arrays.cell_ids.data()[15].value,
  };
  EXPECT_EQ(values_after, (std::array<std::uint64_t, 2>{0, 0}));
}

// Just before step (d) the tail places 14 and 15 hold cells of random grid 1, the build before; the fill turns them
// into the id 0, so the 14 lone cells give 14 blocks, the CPU's. The copy without the fill shows those cells become
// blocks: they share one block, so 15.
TEST(ApplySortTailFillAt, UsedBuffers_HoldNoStaleCell) {
  constexpr std::uint32_t kRoom = 100;
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner(kRoom);
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays(kRoom);
  ASSERT_TRUE(arrays.order_buffers.has_value());

  const BuildReplayResult lone_build = ComputeLoneCellsAfterGridOne(ReplayChange::kNone, buffers, arrays);
  CheckTailStaleThenFilled(arrays);
  const std::vector<CellId> lone_cells = BuildLoneCellsAlongX(14);
  const CpuBuildOfList cpu_build =
      ComputeCpuBuildOfList({.frame = ToUnitFrame(4), .active_cells = lone_cells, .version = {.value = 1}}, buffers);
  EXPECT_EQ(lone_build.cell_blocks, 14U);
  CheckBuildReplayEqualsCpuBuild(lone_build, cpu_build, "lone cells after random grid 1");

  const BuildReplayResult without_fill = ComputeLoneCellsAfterGridOne(ReplayChange::kWithoutTailFill, buffers, arrays);
  EXPECT_EQ(without_fill.cell_blocks, 15U);
}

// After P3 the order P7 (a) writes into holds a permutation of 0..n - 1, whose sum passes 2^32 - 1 from 92,683 cells.
// The box of 103,825 cells fails its siblings at the last position; a copy of P7 (a) that returns there instead of
// writing 0 leaves that permutation, and the first prefix sum fails. The GPU's sequence names the CPU's failure.
TEST(BuildGrid, SplitCountsThatReturnEarly_FailThePrefixSum) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  const std::vector<CellId> failing_cells = BuildLargeBoxOfCells({ToCell(2, 188, 188, 188)});
  const GridInputs inputs = {.frame = ToUnitFrame(3), .active_cells = failing_cells, .version = {.value = 1}};

  const ReplayLaunch launch = {.order = LaunchOrder::kForward, .thread_count = 0};
  ApplyStaleAllOnesWords(arrays);
  const BuildReplayResult as_on_gpu = ComputeBuildReplay(inputs, buffers, launch, ReplayChange::kNone, arrays);
  ApplyStaleAllOnesWords(arrays);
  const BuildReplayResult returning_early =
      ComputeBuildReplay(inputs, buffers, launch, ReplayChange::kSplitCountsReturnEarly, arrays);
  const std::expected<GridCounts, GridFailure> cpu_counts = ComputeGridCounts(inputs, buffers);
  ASSERT_FALSE(cpu_counts.has_value());
  EXPECT_EQ(cpu_counts.error().error, GridError::kSiblingCellMissing);
  EXPECT_EQ(as_on_gpu.failed_prefix_sum_count, 0);
  EXPECT_EQ(ToFailureFields(as_on_gpu.failure), ToFailureFields(cpu_counts.error()));
  EXPECT_EQ(returning_early.failed_prefix_sum_count, 1);
  EXPECT_EQ(returning_early.failure.error, GridError::kInternalFault);
}

// A failing list and the valid grid whose build the arrays hold before it.
struct ListAfterOtherGrid {
  std::vector<CellId> other_cells;  // A valid list that fits the room.
  std::vector<CellId> failing_cells;
  std::size_t thread_count = 0;  // 0 for the GPU's own.
};

// Replays and checks the build of lists.other_cells, then that of lists.failing_cells from the arrays it left
// (CheckBuildReplayEqualsCpu), so the failing list starts from buffers that held another grid.
void CheckFailingListAfterOtherGrid(
    const ListAfterOtherGrid& lists,
    GridBuffers& buffers,
    ReplayArrays& arrays,
    const std::string& name
) {
  CheckBuildReplayEqualsCpu(
      ToUnitFrame(4),
      lists.other_cells,
      lists.thread_count,
      ReplayStart::kAfterStaleValues,
      buffers,
      arrays,
      name + ", the grid before"
  );
  CheckBuildReplayEqualsCpu(
      ToUnitFrame(4),
      lists.failing_cells,
      lists.thread_count,
      ReplayStart::kAfterPreviousList,
      buffers,
      arrays,
      name
  );
}

// Active and split cells that fit the room but, with P7 (b) checking against one place less than they need, not the
// sort count: the check offers kSortCountFaultKey, which reads as kInternalFault with no cell; every later step stops,
// so the marks are 0 and no block id is written; no write leaves the arrays.
TEST(BuildGrid, CellsBeyondSortCount_IsInternalFaultWithNoBlockWritten) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  const GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  const GridInputs inputs = {
      .frame = ToUnitFrame(4),
      .active_cells = BuildShuffledRandomGrids().at(0),
      .version = {.value = 1},
  };
  ApplyStaleValues(arrays);
  const BuildReplayResult replay = ComputeBuildReplay(
      inputs,
      buffers,
      {.order = LaunchOrder::kForward, .thread_count = 0},
      ReplayChange::kSortCountBelowCells,
      arrays
  );
  EXPECT_EQ(arrays.status.failure_key, kSortCountFaultKey);
  EXPECT_EQ(ToFailureFields(replay.failure), ToFailureFields({.error = GridError::kInternalFault}));
  EXPECT_GT(replay.split_cells, 0U);
  EXPECT_EQ(replay.cell_blocks, 0U);
  EXPECT_EQ(arrays.block_ids.data()[0].value, ToCellBlockId(ToStaleCellId()).value);
}

// The lists of the gpu tests BuildGrid.OnGpu_EqualsCpuTablesBitForBit and BuildGrid.OnGpuFailures_EqualCpuFailures,
// replayed through P12. The empty list, one cell, the two-size fixture and the random grids pass every step: their
// version, counts and the used part of the five tables show P7 to P12 give the CPU's grid byte for byte. The mutated
// lists, seeds 1..kReplayedMutatedGridCount as in the check's replay (all 500 would take the test program past 45 s in
// the debug preset; the gpu test runs them all), fail at steps 1 to 4, and show every later step stops. The room
// checks: 9 blocks into room for 8 (step 6), 513 cells into the same (the length check of P1), and 72 active and split
// cells into room for 1 (step 5, after the split cells are counted). Each replay starts from arrays that held another
// grid, and the CPU build runs in CPU buffers that held one too.
TEST(BuildGrid, EveryThreadBlockAndThread_EqualsCpuTables) {
  std::expected<TestGrid, GridFailure> owner = BuildReplayBuffersOwner();
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  ReplayArrays arrays = BuildReplayArrays();
  ASSERT_TRUE(arrays.order_buffers.has_value());
  ApplyStaleValues(arrays);
  const std::vector<CellId> one_cell = {ToCell(0, 1, -2, 3)};
  const std::vector<CellId> two_size_cells = BuildTwoSizeFixtureCells();
  for (const auto& [frame, cells, name] : {
           std::tuple{ToUnitFrame(3), std::span<const CellId>(), std::string("the empty list")},
           std::tuple{ToUnitFrame(3), std::span<const CellId>(one_cell), std::string("one cell")},
           std::tuple{ToUnitFrame(3), std::span<const CellId>(two_size_cells), std::string("the two-size fixture")},
       }) {
    CheckBuildReplayEqualsCpu(frame, cells, 0, ReplayStart::kAfterPreviousList, buffers, arrays, name);
  }
  for (std::size_t grid = 0; grid < kRandomGridCount; ++grid) {
    CheckBuildReplayEqualsCpu(
        ToUnitFrame(4),
        BuildShuffledRandomGrids().at(grid),
        0,
        ReplayStart::kAfterPreviousList,
        buffers,
        arrays,
        "random grid " + std::to_string(grid)
    );
  }
  for (std::uint64_t seed = 1; seed <= kReplayedMutatedGridCount; ++seed) {
    CheckBuildReplayEqualsCpu(
        ToUnitFrame(4),
        BuildMutatedGridCells(seed),
        0,
        ReplayStart::kAfterPreviousList,
        buffers,
        arrays,
        "mutated list " + std::to_string(seed)
    );
  }

  std::expected<TestGrid, GridFailure> owner_of_8 = BuildReplayBuffersOwner(8);
  GridBuffers buffers_of_8 = TakeEmptyGridBuffers(owner_of_8);
  std::expected<TestGrid, GridFailure> owner_of_1 = BuildReplayBuffersOwner(1);
  GridBuffers buffers_of_1 = TakeEmptyGridBuffers(owner_of_1);
  ReplayArrays arrays_of_8 = BuildReplayArrays(8);
  ReplayArrays arrays_of_1 = BuildReplayArrays(1);
  ASSERT_TRUE(arrays_of_8.order_buffers.has_value());
  ASSERT_TRUE(arrays_of_1.order_buffers.has_value());
  const std::vector<CellId> eight_blocks = BuildLoneCellsAlongX(8);
  const std::vector<CellId> one_block = {ToCell(0, 0, 0, 0), ToCell(0, 1, 0, 0), ToCell(0, 2, 0, 0)};
  for (const std::size_t thread_count : {std::size_t{0}, std::size_t{1}, std::size_t{7}}) {
    const std::string threads = ", threads " + std::to_string(thread_count);
    CheckFailingListAfterOtherGrid(
        {.other_cells = eight_blocks, .failing_cells = BuildLoneCellsAlongX(9), .thread_count = thread_count},
        buffers_of_8,
        arrays_of_8,
        "nine blocks into eight" + threads
    );
    CheckFailingListAfterOtherGrid(
        {
            .other_cells = eight_blocks,
            .failing_cells = BuildLoneCellsAlongX((64 * 8) + 1),
            .thread_count = thread_count,
        },
        buffers_of_8,
        arrays_of_8,
        "513 cells into eight blocks" + threads
    );
    CheckFailingListAfterOtherGrid(
        {.other_cells = one_block, .failing_cells = BuildSmallCellsOfEightBigCells(), .thread_count = thread_count},
        buffers_of_1,
        arrays_of_1,
        "72 cells into one block" + threads
    );
  }
}

}  // namespace
}  // namespace fluidity::grid
