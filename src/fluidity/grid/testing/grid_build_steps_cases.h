// Test-only: the helpers that the unit tests and the functional tests of grid_build_steps share.
//
// Purpose: let the unit tests and the functional tests of one header use the same helpers, written once.
//
// Idea: the unit tests of a header and its functional tests call the same helpers, so each helper lives here once,
//   as it was written when the tests were in one file.
//
// Owns:    ComputePowerOfTwo, ToUnitFrame, ToCell, ToFloorOfHalving, ToBigCellByDivision, ToSortedCells, ToValues,
//          ToCount, ToStableOrderByPasses, BuildRandomCellIds, ComputeSmallestFailureKey,
//          BuildEverySizeLevelAtCoordLimits, kSameSizeLevelEntriesPerBlock, kSmallerBlocksPerBlock, TableVectors,
//          BuildEmptyTables, ToBlock, ToCellPlace, ApplyFoundNeighbours, BuildSplitCells, ToBlockCoordByDivision,
//          ToBlockCoord, ToSizeLevelNumber, ToBlockOfCell, BuildTablesOfCells, FacePieceWithKind,
//          BuildFacePiecesOfTables, BuildFacePiecesOfChecker, LaunchOrder, kLaunchOrders, kSiblingChecks, kReplayRoom,
//          ReplayLaunch, ApplyEveryThreadOfSizedLaunch, ApplyEveryThreadOfLaunch, ReplayArrays, kGuardCount,
//          BuildReplayArrays, BuildReplayArrays, ToGuardCellId, kGuardSortKey, kGuardOrderEntry, kGuardCellState,
//          kGuardFacePieceKind, ToGuardEntries, ToGuardEntries, ApplyGuardValues, ComputeAreGuardEntries,
//          ComputeAreGuardsUnchanged, ToStaleCellId, ApplyStaleValues, ApplyStaleAllOnesWords,
//          ApplyListCheckStepsByReplay, ReplayChange, ApplyPrefixSumOfReplay, ComputeCellPlacesKeyOfReplay,
//          ApplyBlockStepsByReplay, ToReplayTables, ComputeIsBlockEntryInRange, ComputeAreTableEntriesInRange,
//          ApplyTableStepsByReplay, ApplyGpuBuildStepsByReplay, ToFailureFields, BuildReplayBuffersOwner,
//          BuildReplayBuffersOwner, ToVector, ToTableVectors, BuildReplayResult, ToCountsOfStatus, ComputeBuildReplay,
//          ExpectedBlocks, ComputeExpectedBlocks, CpuBuildOfList, ComputeCpuBuildOfList, ComputeIsSameTableVectors,
//          CheckBuildReplayEqualsCpuBuild, kTwoToThe34, kTwoToThe38
// Uses:    the headers it includes
// Used by: src/fluidity/grid/grid_build_steps.test.cc, tests/grid/cell_lists_on_every_size_level.test.cc and
//          tests/grid/grid_build_steps_on_random_grids.test.cc
// Read:    the test files that include it

#pragma once

#include "fluidity/base/cpu_only.h"

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

#include "fluidity/base/axis.h"
#include "fluidity/base/buffer.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/stable_order.h"
#include "fluidity/grid/grid_buffers_internal.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_cell_checks.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"

namespace fluidity::grid {

// 2^exponent as a coordinate, shifted as an unsigned value.
constexpr std::int32_t ComputePowerOfTwo(int exponent) {
  return static_cast<std::int32_t>(1U << static_cast<unsigned>(exponent));
}

// A frame with its origin at 0 and biggest cells 1 m wide.
inline GridFrame ToUnitFrame(int size_level_count) {
  return {
      .origin = {.x = 0, .y = 0, .z = 0},
      .biggest_cell_width = 1.0,
      .size_level_count = static_cast<std::uint8_t>(size_level_count),
  };
}

inline CellId ToCell(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return ToCellId(static_cast<SizeLevel>(size_level), {.x = x, .y = y, .z = z});
}

// floor(coord / 2^size_level_steps), by integer division and a correction for a negative remainder: the test's own
// arithmetic, independent of the shift the grid uses.
//
// clang-tidy: a coordinate and a count of size levels; the swappable-parameters check flags the two integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline std::int32_t ToFloorOfHalving(std::int32_t coord, int size_level_steps) {
  const auto divisor = static_cast<std::int64_t>(std::uint64_t{1} << static_cast<unsigned>(size_level_steps));
  std::int64_t quotient = coord / divisor;
  if (coord % divisor != 0 && coord < 0) {
    quotient -= 1;
  }
  return static_cast<std::int32_t>(quotient);
}

// The cell of size level bigger_size_level that holds `cell`, by the test's own floor.
inline CellId ToBigCellByDivision(CellId cell, int bigger_size_level) {
  const int size_level_steps = static_cast<int>(ToSizeLevel(cell)) - bigger_size_level;
  const CellCoord coord = ToCellCoord(cell);
  return ToCell(
      bigger_size_level,
      ToFloorOfHalving(coord.x, size_level_steps),
      ToFloorOfHalving(coord.y, size_level_steps),
      ToFloorOfHalving(coord.z, size_level_steps)
  );
}

// The list in the fixed order, sorted with ComesBefore (std::sort is test-only here).
inline std::vector<CellId> ToSortedCells(std::vector<CellId> cells) {
  std::ranges::sort(cells, [](CellId first, CellId second) {
    return ComesBefore(first, second);
  });
  return cells;
}

inline std::vector<std::uint64_t> ToValues(const std::vector<CellId>& cells) {
  std::vector<std::uint64_t> values;
  values.reserve(cells.size());
  for (const CellId cell : cells) {
    values.push_back(cell.value);
  }
  return values;
}

inline std::uint32_t ToCount(const std::vector<CellId>& cells) {
  return static_cast<std::uint32_t>(cells.size());
}

// Orders indexes into `cells` by the three passes of a sort key, least significant first, each pass stable, as
// BuildStableOrder does; std::stable_sort stands in for it in this test.
template <typename SortKeyFunction>
std::vector<std::size_t> ToStableOrderByPasses(const std::vector<CellId>& cells, SortKeyFunction sort_key) {
  std::vector<std::size_t> order(cells.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    std::ranges::stable_sort(order, [&cells, &sort_key, pass](std::size_t first, std::size_t second) {
      return sort_key(cells[first], pass) < sort_key(cells[second], pass);
    });
  }
  return order;
}

// Random cell ids at size levels 0..max_size_level, with coordinates in -kMaxCoord..kMaxCoord; the first 100 have one
// coordinate at a limit: axis index % 3, -kMaxCoord for an even index / 3 and kMaxCoord for an odd one.
//
// clang-tidy: a seed, a size level and a count; the swappable-parameters check flags the three integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline std::vector<CellId> BuildRandomCellIds(std::uint64_t seed, int max_size_level, std::size_t count) {
  std::mt19937_64 random_numbers(seed);
  const std::uint64_t coord_count = (2 * static_cast<std::uint64_t>(kMaxCoord)) + 1;
  std::vector<CellId> cells;
  cells.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint64_t size_level_choices = static_cast<std::uint64_t>(max_size_level) + 1;
    const auto size_level = static_cast<int>(ToUniformIndex(random_numbers(), size_level_choices));
    std::array<std::int32_t, 3> coord{};
    for (std::int32_t& axis_coord : coord) {
      axis_coord = static_cast<std::int32_t>(ToUniformIndex(random_numbers(), coord_count)) - kMaxCoord;
    }
    if (index < 100) {
      coord.at(index % 3) = (index / 3) % 2 == 0 ? -kMaxCoord : kMaxCoord;
    }
    cells.push_back(ToCell(size_level, coord.at(0), coord.at(1), coord.at(2)));
  }
  return cells;
}

// The smallest failure key of one check over every position of a sorted list, or kNoFailureKey.
template <typename CheckFunction>
std::uint64_t ComputeSmallestFailureKey(const std::vector<CellId>& sorted_cells, int step, CheckFunction check) {
  std::uint64_t smallest_key = ~std::uint64_t{0};
  for (std::uint32_t position = 0; position < ToCount(sorted_cells); ++position) {
    const CellCheck cell_check = check(sorted_cells.data(), ToCount(sorted_cells), position);
    if (cell_check.error != GridError::kNone) {
      smallest_key = std::min(smallest_key, ToFailureKey(step, position, cell_check.detail));
    }
  }
  return smallest_key;
}

// Coordinates per axis: both limits, the three around 0, and -2^l and 2^l, whose low corners, 2^l x 2^(15 - l) = 2^15
// widths of size level 15, are those of size-level-0 cells -1 and 1, so every size level ties at those corners.
inline std::vector<CellId> BuildEverySizeLevelAtCoordLimits() {
  std::vector<CellId> cells;
  for (int size_level = 0; size_level < kMaxSizeLevels; ++size_level) {
    // A set, so size level 0, where 2^0 = 1, lists each cell once.
    const std::set<std::int32_t> axis_values =
        {-kMaxCoord, -ComputePowerOfTwo(size_level), -1, 0, 1, ComputePowerOfTwo(size_level), kMaxCoord};
    for (const std::int32_t z : axis_values) {
      for (const std::int32_t y : axis_values) {
        for (const std::int32_t x : axis_values) {
          cells.push_back(ToCell(size_level, x, y, z));
        }
      }
    }
  }
  return cells;
}

// ---------------------------------------------------------------------------------------------------------------------
// Neighbour tables, cell states and face piece kinds: method paper <proc-build-grid> steps 6 to 8
// ---------------------------------------------------------------------------------------------------------------------

constexpr std::uint32_t kSameSizeLevelEntriesPerBlock = 27;

constexpr std::uint32_t kSmallerBlocksPerBlock = 8;

// The tables of one grid in test vectors; ToGridTables points a GridTables into them.
struct TableVectors {
  GridFrame frame{};
  std::vector<CellBlockId> block_ids;
  std::vector<std::uint32_t> same_size_level_blocks;
  std::vector<std::uint32_t> bigger_block;
  std::vector<std::uint32_t> smaller_blocks;
  std::vector<CellState> cell_states;
  std::vector<FacePieceKind> face_piece_kinds;

  [[nodiscard]] GridTables ToGridTables() const {
    return {
        .frame = frame,
        .version = GridVersion{1},
        .block_count = static_cast<std::uint32_t>(block_ids.size()),
        .block_ids = block_ids.data(),
        .same_size_level_blocks = same_size_level_blocks.data(),
        .bigger_block = bigger_block.data(),
        .smaller_blocks = smaller_blocks.data(),
        .cell_states = cell_states.data(),
        .face_piece_kinds = face_piece_kinds.data(),
    };
  }
};

// Tables for the given blocks with no neighbour, every cell place kAbsent and every face place kNotAPiece.
inline TableVectors BuildEmptyTables(GridFrame frame, std::vector<CellBlockId> block_ids) {
  const std::size_t block_count = block_ids.size();
  return {
      .frame = frame,
      .block_ids = std::move(block_ids),
      .same_size_level_blocks = std::vector<std::uint32_t>(block_count * kSameSizeLevelEntriesPerBlock, kNoIndex),
      .bigger_block = std::vector<std::uint32_t>(block_count, kNoIndex),
      .smaller_blocks = std::vector<std::uint32_t>(block_count * kSmallerBlocksPerBlock, kNoIndex),
      .cell_states = std::vector<CellState>(block_count * kCellPlacesPerBlock, CellState::kAbsent),
      .face_piece_kinds = std::vector<FacePieceKind>(block_count * kFacePlacesPerBlock, FacePieceKind::kNotAPiece),
  };
}

// The block of size_level at a block coordinate, by the bits of the cell id at that coordinate (grid_ids.h).
inline CellBlockId ToBlock(int size_level, std::int32_t x, std::int32_t y, std::int32_t z) {
  return {.value = ToCell(size_level, x, y, z).value};
}

// Cell place (x, y, z) of a block: 64 b + (4 z + y) * 4 + x.
//
// clang-tidy: a block index and three local coordinates; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline std::uint32_t ToCellPlace(std::uint32_t block, int x, int y, int z) {
  return (block * 64) + static_cast<std::uint32_t>((((z * 4) + y) * 4) + x);
}

// Writes the 27, 1 and 8 entries of every block with FindBlockNeighbours.
inline void ApplyFoundNeighbours(TableVectors& tables) {
  const auto block_count = static_cast<std::uint32_t>(tables.block_ids.size());
  for (std::uint32_t block = 0; block < block_count; ++block) {
    FindBlockNeighbours(
        tables.block_ids.data(),
        block_count,
        CellBlockIndex{block},
        tables.same_size_level_blocks.data(),
        tables.bigger_block.data(),
        tables.smaller_blocks.data()
    );
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The three steps together, against a direct lookup and against the geometric checker
// ---------------------------------------------------------------------------------------------------------------------

// The split cells of a valid list: every cell of a bigger size level that holds an active cell, by the test's floor.
inline std::set<std::uint64_t> BuildSplitCells(const std::vector<CellId>& active_cells) {
  std::set<std::uint64_t> split_cells;
  for (const CellId cell : active_cells) {
    for (int bigger_size_level = 0; bigger_size_level < static_cast<int>(ToSizeLevel(cell)); ++bigger_size_level) {
      split_cells.insert(ToBigCellByDivision(cell, bigger_size_level).value);
    }
  }
  return split_cells;
}

// The block coordinate of a cell coordinate, by the test's floor: floor(c / 4) on each axis.
inline CellCoord ToBlockCoordByDivision(CellCoord cell_coord) {
  return {
      .x = ToFloorOfHalving(cell_coord.x, 2),
      .y = ToFloorOfHalving(cell_coord.y, 2),
      .z = ToFloorOfHalving(cell_coord.z, 2),
  };
}

inline CellCoord ToBlockCoord(CellBlockId block) {
  return ToCellCoord(CellId{.value = block.value});
}

inline int ToSizeLevelNumber(CellBlockId block) {
  return static_cast<int>(ToSizeLevel(CellId{.value = block.value}));
}

// The block of a cell, by the test's floor.
inline CellBlockId ToBlockOfCell(CellId cell) {
  const CellCoord block_coord = ToBlockCoordByDivision(ToCellCoord(cell));
  return ToBlock(static_cast<int>(ToSizeLevel(cell)), block_coord.x, block_coord.y, block_coord.z);
}

// The tables of a valid list of active cells, built with the three steps under test: block ids by the test (every
// block that holds an active or split cell, in the fixed order by ComesBefore), neighbours by FindBlockNeighbours,
// active and split marks by the test, then ComputeCoveredCellState at every other place and ComputeFacePieceKind at
// every face place.
inline TableVectors BuildTablesOfCells(GridFrame frame, const std::vector<CellId>& active_cells) {
  const std::set<std::uint64_t> split_cells = BuildSplitCells(active_cells);
  std::vector<CellId> active_and_split_cells = active_cells;
  for (const std::uint64_t split_cell : split_cells) {
    active_and_split_cells.push_back(CellId{split_cell});
  }
  std::set<std::uint64_t> block_values;
  for (const CellId cell : active_and_split_cells) {
    block_values.insert(ToBlockOfCell(cell).value);
  }
  std::vector<CellBlockId> block_ids;
  block_ids.reserve(block_values.size());
  for (const std::uint64_t block_value : block_values) {
    block_ids.push_back(CellBlockId{block_value});
  }
  std::ranges::sort(block_ids, [](CellBlockId first, CellBlockId second) {
    return ComesBefore(first, second);
  });
  std::map<std::uint64_t, std::uint32_t> block_index_by_id;
  for (std::size_t block = 0; block < block_ids.size(); ++block) {
    block_index_by_id[block_ids.at(block).value] = static_cast<std::uint32_t>(block);
  }

  TableVectors tables = BuildEmptyTables(frame, block_ids);
  ApplyFoundNeighbours(tables);
  for (const CellId cell : active_and_split_cells) {
    const CellCoord cell_coord = ToCellCoord(cell);
    const CellCoord block_coord = ToBlockCoordByDivision(cell_coord);
    const std::uint32_t place = ToCellPlace(
        block_index_by_id.at(ToBlockOfCell(cell).value),
        cell_coord.x - (4 * block_coord.x),
        cell_coord.y - (4 * block_coord.y),
        cell_coord.z - (4 * block_coord.z)
    );
    tables.cell_states.at(place) = split_cells.contains(cell.value) ? CellState::kSplit : CellState::kActive;
  }
  for (std::uint32_t place = 0; place < tables.cell_states.size(); ++place) {
    const CellState state = tables.cell_states.at(place);
    if (state != CellState::kActive && state != CellState::kSplit) {
      tables.cell_states.at(place) = ComputeCoveredCellState(tables.ToGridTables(), CellPlaceIndex{place});
    }
  }
  for (std::uint32_t face = 0; face < tables.face_piece_kinds.size(); ++face) {
    tables.face_piece_kinds.at(face) = ComputeFacePieceKind(tables.ToGridTables(), FacePlaceIndex{face});
  }
  return tables;
}

// A face piece as (id, kind). The face pieces of a grid are compared as multisets, so a face piece kept twice fails.
using FacePieceWithKind = std::pair<std::uint64_t, FacePieceKind>;

// The face pieces of the tables: every face place that is not kNotAPiece, with the id of its axis, its block's size
// level and the cell place on its high side, 4 b + the local coordinate, the plane along the axis. Plane 4 of a block
// and plane 0 of the block above give one id, so the id cannot say which block keeps the face piece; it also expects
// that a face piece on plane 4 has no block of the same size level on its high side, where plane 0 would keep it.
inline std::multiset<FacePieceWithKind> BuildFacePiecesOfTables(const TableVectors& tables) {
  const std::vector<std::uint64_t> block_list = [&tables] {
    std::vector<std::uint64_t> values;
    values.reserve(tables.block_ids.size());
    for (const CellBlockId block : tables.block_ids) {
      values.push_back(block.value);
    }
    return values;
  }();
  const std::set<std::uint64_t> block_values(block_list.begin(), block_list.end());
  std::multiset<FacePieceWithKind> pieces;
  for (std::uint32_t face = 0; face < tables.face_piece_kinds.size(); ++face) {
    const FacePieceKind kind = tables.face_piece_kinds.at(face);
    if (kind == FacePieceKind::kNotAPiece) {
      continue;
    }
    const CellBlockId block = tables.block_ids.at(face / 240);
    const auto axis_number = static_cast<std::size_t>((face % 240) / 80);
    // x varies fastest, then y, then z; along the axis the plane takes 5 values, on the other two axes 4.
    std::array<std::int32_t, 3> local_coord{};
    auto rest = static_cast<std::int32_t>(face % 80);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const std::int32_t extent = axis == axis_number ? 5 : 4;
      local_coord.at(axis) = rest % extent;
      rest /= extent;
    }
    const CellCoord block_coord = ToBlockCoord(block);
    const CellCoord cell_high_coord = {
        .x = (4 * block_coord.x) + local_coord.at(0),
        .y = (4 * block_coord.y) + local_coord.at(1),
        .z = (4 * block_coord.z) + local_coord.at(2),
    };
    const auto axis = static_cast<Axis>(axis_number);
    const auto size_level = static_cast<SizeLevel>(ToSizeLevelNumber(block));
    if (local_coord.at(axis_number) == 4) {
      std::array<std::int32_t, 3> high_block_coord = {block_coord.x, block_coord.y, block_coord.z};
      high_block_coord.at(axis_number) += 1;
      const CellBlockId high_block =
          ToBlock(ToSizeLevelNumber(block), high_block_coord.at(0), high_block_coord.at(1), high_block_coord.at(2));
      EXPECT_FALSE(block_values.contains(high_block.value)) << "face place " << face << " is kept on plane 4";
    }
    pieces.insert({ToFacePieceId(axis, size_level, cell_high_coord).value, kind});
  }
  return pieces;
}

// The face pieces of the geometric checker, the kind from its cells: both sides kBetweenCells, only the high side
// kGridEdgeLow, only the low side kGridEdgeHigh.
inline std::multiset<FacePieceWithKind>
BuildFacePiecesOfChecker(GridFrame frame, const std::vector<CellId>& active_cells) {
  const GeometricGrid grid(frame, active_cells);
  std::multiset<FacePieceWithKind> pieces;
  for (const GeometricFacePiece& piece : grid.FacePieces()) {
    FacePieceKind kind = FacePieceKind::kBetweenCells;
    if (!piece.cell_low.has_value()) {
      kind = FacePieceKind::kGridEdgeLow;
    } else if (!piece.cell_high.has_value()) {
      kind = FacePieceKind::kGridEdgeHigh;
    }
    pieces.insert({piece.id.value, kind});
  }
  return pieces;
}

// ---------------------------------------------------------------------------------------------------------------------
// The check of a list on the GPU (<proc-build-grid-in-parallel> P1 to P6), replayed on the CPU
// ---------------------------------------------------------------------------------------------------------------------

// The order in which the replay runs the threads of one launch.
enum class LaunchOrder : std::uint8_t {
  kForward,  // Thread block 0, thread 0 first.
  kReverse,  // The last thread of the last thread block first.
};

constexpr std::array<LaunchOrder, 2> kLaunchOrders = {LaunchOrder::kForward, LaunchOrder::kReverse};

constexpr std::array<SiblingCheck, 2> kSiblingChecks = {SiblingCheck::kRequire, SiblingCheck::kSkip};

// The room of the CPU buffers the replay tests check lists in: 262,144 cell places, more than any list here.
constexpr std::uint32_t kReplayRoom = 4096;

// How the replay runs the threads of every launch.
struct ReplayLaunch {
  LaunchOrder order = LaunchOrder::kForward;
  // Threads of a launch. 0 is the GPU's own: the thread blocks of ComputeLaunchSize(item count) times their size, one
  // item per thread for any list here. A smaller number gives each thread several items, as the GPU does for a list of
  // more than 65,535 x 256 cells.
  std::size_t thread_count = 0;
};

// Runs apply_item(index) for every item below item_count of one launch sized for launch_item_count items, one thread
// after the other in launch.order, each thread over its items as the grid-stride loop of a GPU function walks them:
// index = thread, thread + thread count, and so on, where thread = thread block x thread block size + thread in the
// thread block. apply_item returns false where the GPU function returns, which ends that thread's loop. A launch sized
// for 0 items does not start. The GPU functions of P9 to P11 are sized for the room and act on the items below a count
// they read from the status record, so their two counts differ.
//
// clang-tidy: two counts of items; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <typename ApplyItem>
void ApplyEveryThreadOfSizedLaunch(
    std::size_t launch_item_count,
    std::size_t item_count,
    ReplayLaunch launch,
    ApplyItem apply_item
) {
  if (launch_item_count == 0) {
    return;
  }
  const LaunchSize launch_size = ComputeLaunchSize(launch_item_count);
  const std::size_t launch_thread_count =
      static_cast<std::size_t>(launch_size.thread_block_count) * launch_size.thread_block_size;
  const std::size_t thread_count = launch.thread_count == 0 ? launch_thread_count : launch.thread_count;
  for (std::size_t visited = 0; visited < thread_count; ++visited) {
    const std::size_t thread = launch.order == LaunchOrder::kForward ? visited : thread_count - 1 - visited;
    for (std::size_t index = thread; index < item_count; index += thread_count) {
      if (!apply_item(index)) {
        break;
      }
    }
  }
}

// NOLINTEND(bugprone-easily-swappable-parameters)

// Runs apply_item over a launch sized for its items, as ApplyEveryThreadOfSizedLaunch says.
template <typename ApplyItem>
void ApplyEveryThreadOfLaunch(std::size_t item_count, ReplayLaunch launch, ApplyItem apply_item) {
  ApplyEveryThreadOfSizedLaunch(item_count, item_count, launch, apply_item);
}

// The GPU's working arrays of a check or a build, its cell ids and its block ids, built on the CPU by the replay. CPU
// builder buffers hold only the cell ids of the CPU check, and the replay keeps its own, so the CPU check never writes
// what the replay compares.
struct ReplayArrays {
  std::uint32_t max_cell_blocks = 0;  // The room the arrays are sized for.
  Buffer<CellId> cell_ids;
  Buffer<std::uint64_t> sort_keys;
  Buffer<std::uint32_t> order;
  Buffer<std::uint32_t> second_order;
  std::optional<OrderBuffers> order_buffers;
  Buffer<CellBlockId> block_ids;
  Buffer<std::uint32_t> same_size_level_blocks;  // 27 per block.
  // 1 per block, the bigger blocks, then 8 per block, the smaller blocks, from ComputeSmallerBlocksStart of the room.
  Buffer<std::uint32_t> bigger_and_smaller_blocks;
  Buffer<CellState> cell_states;  // 64 per block.
  Buffer<FacePieceKind> face_piece_kinds;  // 240 per block.
  GridBuildStatus status{};
  // The first sort count cell ids just before P7 (d), the tail fill, of the last build replay.
  std::vector<CellId> cells_before_tail_fill;
  int failed_prefix_sum_count = 0;  // Prefix sums of the last build replay that returned an error.
  // Places of P10 (c) and P11 whose block, bigger block or a neighbour of either named a block at or past the block
  // count: the replay counts them and skips the place, where the GPU would read past its tables.
  int stale_table_read_count = 0;
};

// The entries after the room's part of each replay array, filled with a fixed pattern that no step writes: a write
// past an array of the room on the GPU lands there in the replay, and the test sees it without a sanitizer.
constexpr std::size_t kGuardCount = 16;

// Builds the working arrays for a room of max_cell_blocks blocks, as BuildGridBuilderBuffers and BuildGridBuffers size
// them on the GPU, each followed by kGuardCount guard entries: 64 cell ids, sort keys and order entries per block, plus
// 1 order entry for the total of a prefix sum, and the five tables of the room. The guards hold their pattern
// (ApplyGuardValues).
inline ReplayArrays BuildReplayArrays(std::uint32_t max_cell_blocks) {
  const std::size_t place_count = std::size_t{kCellPlacesPerBlock} * max_cell_blocks;
  ReplayArrays arrays;
  arrays.max_cell_blocks = max_cell_blocks;
  std::expected<Buffer<CellId>, BufferError> cell_ids = BuildBuffer<CellId>(RunOn::kCpu, place_count + kGuardCount);
  std::expected<Buffer<std::uint64_t>, BufferError> sort_keys =
      BuildBuffer<std::uint64_t>(RunOn::kCpu, place_count + kGuardCount);
  std::expected<Buffer<std::uint32_t>, BufferError> order =
      BuildBuffer<std::uint32_t>(RunOn::kCpu, place_count + 1 + kGuardCount);
  std::expected<Buffer<std::uint32_t>, BufferError> second_order =
      BuildBuffer<std::uint32_t>(RunOn::kCpu, place_count + 1 + kGuardCount);
  std::expected<OrderBuffers, BufferError> order_buffers = BuildOrderBuffers(RunOn::kCpu, place_count + 1);
  std::expected<Buffer<CellBlockId>, BufferError> block_ids =
      BuildBuffer<CellBlockId>(RunOn::kCpu, max_cell_blocks + kGuardCount);
  const std::size_t room = max_cell_blocks;
  std::expected<Buffer<std::uint32_t>, BufferError> same_size_level_blocks =
      BuildBuffer<std::uint32_t>(RunOn::kCpu, (kSameSizeLevelEntriesPerBlock * room) + kGuardCount);
  std::expected<Buffer<std::uint32_t>, BufferError> bigger_and_smaller_blocks =
      BuildBuffer<std::uint32_t>(RunOn::kCpu, ((1 + kSmallerBlocksPerBlock) * room) + kGuardCount);
  std::expected<Buffer<CellState>, BufferError> cell_states =
      BuildBuffer<CellState>(RunOn::kCpu, place_count + kGuardCount);
  std::expected<Buffer<FacePieceKind>, BufferError> face_piece_kinds =
      BuildBuffer<FacePieceKind>(RunOn::kCpu, (std::size_t{kFacePlacesPerBlock} * room) + kGuardCount);
  const bool is_built = cell_ids.has_value() && sort_keys.has_value() && order.has_value() && second_order.has_value()
      && order_buffers.has_value() && block_ids.has_value() && same_size_level_blocks.has_value()
      && bigger_and_smaller_blocks.has_value() && cell_states.has_value() && face_piece_kinds.has_value();
  EXPECT_TRUE(is_built);
  if (is_built) {
    arrays.same_size_level_blocks = std::move(*same_size_level_blocks);
    arrays.bigger_and_smaller_blocks = std::move(*bigger_and_smaller_blocks);
    arrays.cell_states = std::move(*cell_states);
    arrays.face_piece_kinds = std::move(*face_piece_kinds);
    arrays.cell_ids = std::move(*cell_ids);
    arrays.sort_keys = std::move(*sort_keys);
    arrays.order = std::move(*order);
    arrays.second_order = std::move(*second_order);
    arrays.order_buffers = std::move(*order_buffers);
    arrays.block_ids = std::move(*block_ids);
  }
  return arrays;
}

inline ReplayArrays BuildReplayArrays() {
  return BuildReplayArrays(kReplayRoom);
}

// The guard patterns: values no step writes and no list of these tests holds.
inline CellId ToGuardCellId() {
  return ToCell(4, 2000, 2000, 2000);
}

constexpr std::uint64_t kGuardSortKey = 0x5A5A5A5A5A5A5A5AU;

constexpr std::uint32_t kGuardOrderEntry = 0xA5A5A5A5U;  // Also the guard of the two neighbour tables.

constexpr auto kGuardCellState = static_cast<CellState>(0xA5U);  // No value of the enum.

constexpr auto kGuardFacePieceKind = static_cast<FacePieceKind>(0x5AU);  // No value of the enum.

// The guard entries of one array: the last kGuardCount.
template <typename Value>
std::span<Value> ToGuardEntries(Buffer<Value>& buffer) {
  return std::span<Value>(buffer.data(), buffer.size()).last(kGuardCount);
}

template <typename Value>
std::span<const Value> ToGuardEntries(const Buffer<Value>& buffer) {
  return std::span<const Value>(buffer.data(), buffer.size()).last(kGuardCount);
}

// Writes the guard patterns after the room's part of every array.
inline void ApplyGuardValues(ReplayArrays& arrays) {
  std::ranges::fill(ToGuardEntries(arrays.cell_ids), ToGuardCellId());
  std::ranges::fill(ToGuardEntries(arrays.sort_keys), kGuardSortKey);
  std::ranges::fill(ToGuardEntries(arrays.order), kGuardOrderEntry);
  std::ranges::fill(ToGuardEntries(arrays.second_order), kGuardOrderEntry);
  std::ranges::fill(ToGuardEntries(arrays.block_ids), ToCellBlockId(ToGuardCellId()));
  std::ranges::fill(ToGuardEntries(arrays.same_size_level_blocks), kGuardOrderEntry);
  std::ranges::fill(ToGuardEntries(arrays.bigger_and_smaller_blocks), kGuardOrderEntry);
  std::ranges::fill(ToGuardEntries(arrays.cell_states), kGuardCellState);
  std::ranges::fill(ToGuardEntries(arrays.face_piece_kinds), kGuardFacePieceKind);
}

// Says whether every guard entry of one array holds `guard`.
template <typename Value>
bool ComputeAreGuardEntries(const Buffer<Value>& buffer, Value guard) {
  return std::ranges::all_of(ToGuardEntries(buffer), [guard](Value entry) {
    return entry == guard;
  });
}

// Says whether every guard entry still holds its pattern: no write left the room's part of an array.
inline bool ComputeAreGuardsUnchanged(const ReplayArrays& arrays) {
  const CellId guard_cell = ToGuardCellId();
  const CellBlockId guard_block = ToCellBlockId(guard_cell);
  return std::ranges::all_of(
             ToGuardEntries(arrays.cell_ids),
             [guard_cell](CellId cell) {
               return cell.value == guard_cell.value;
             }
         )
      && std::ranges::all_of(
             ToGuardEntries(arrays.sort_keys),
             [](std::uint64_t key) {
               return key == kGuardSortKey;
             }
      )
      && std::ranges::all_of(
             ToGuardEntries(arrays.order),
             [](std::uint32_t entry) {
               return entry == kGuardOrderEntry;
             }
      )
      && std::ranges::all_of(
             ToGuardEntries(arrays.second_order),
             [](std::uint32_t entry) {
               return entry == kGuardOrderEntry;
             }
      )
      && std::ranges::all_of(
             ToGuardEntries(arrays.block_ids),
             [guard_block](CellBlockId block) {
               return block.value == guard_block.value;
             }
      )
      && ComputeAreGuardEntries(arrays.same_size_level_blocks, kGuardOrderEntry)
      && ComputeAreGuardEntries(arrays.bigger_and_smaller_blocks, kGuardOrderEntry)
      && ComputeAreGuardEntries(arrays.cell_states, kGuardCellState)
      && ComputeAreGuardEntries(arrays.face_piece_kinds, kGuardFacePieceKind);
}

// A cell id that no list of these tests holds, written where a result is expected, so a result never written shows.
inline CellId ToStaleCellId() {
  return ToCell(3, 1000, 1000, 1000);
}

// Writes large values over every working array, the cell ids and the tables, and a status record that already holds
// the smallest key there is, as an earlier check could leave them. A replay that read any of them before writing it
// would differ from the CPU: the stale key would win the minimum, a stale sort key or order would misplace a cell, a
// stale cell id would stay in the sorted list or become a block, a stale neighbour entry, kSplit state or
// kOutsidePiece kind would stay in a table, kSplit even stopping P10 (c) at that place. The guards keep their patterns.
inline void ApplyStaleValues(ReplayArrays& arrays) {
  std::fill_n(arrays.cell_ids.data(), arrays.cell_ids.size(), ToStaleCellId());
  std::fill_n(arrays.sort_keys.data(), arrays.sort_keys.size(), ~std::uint64_t{0} - 1);
  std::fill_n(arrays.order.data(), arrays.order.size(), kNoIndex - 1);
  std::fill_n(arrays.second_order.data(), arrays.second_order.size(), kNoIndex - 1);
  std::fill_n(arrays.block_ids.data(), arrays.block_ids.size(), ToCellBlockId(ToStaleCellId()));
  std::fill_n(arrays.same_size_level_blocks.data(), arrays.same_size_level_blocks.size(), kNoIndex - 1);
  std::fill_n(arrays.bigger_and_smaller_blocks.data(), arrays.bigger_and_smaller_blocks.size(), kNoIndex - 1);
  std::fill_n(arrays.cell_states.data(), arrays.cell_states.size(), CellState::kSplit);
  std::fill_n(arrays.face_piece_kinds.data(), arrays.face_piece_kinds.size(), FacePieceKind::kOutsidePiece);
  ApplyGuardValues(arrays);
  arrays.status = ComputeResetBuildStatus();
  arrays.status.failure_key = ToFailureKey(kIdStep, 0, 0);
  arrays.status.error = GridError::kBadInput;
  arrays.status.cell = ToCell(0, 1, 2, 3);
}

// Writes 0xFFFFFFFF words over the sort keys, both orders and the whole status record, its counts included, and the
// stale id over the cell ids: a prefix sum that read two such words would pass 2^32 - 1 and fail. The guards keep their
// patterns.
inline void ApplyStaleAllOnesWords(ReplayArrays& arrays) {
  ApplyStaleValues(arrays);
  std::fill_n(arrays.sort_keys.data(), arrays.sort_keys.size(), ~std::uint64_t{0});
  std::fill_n(arrays.order.data(), arrays.order.size(), ~std::uint32_t{0});
  std::fill_n(arrays.second_order.data(), arrays.second_order.size(), ~std::uint32_t{0});
  ApplyGuardValues(arrays);
  arrays.status.cell_blocks = ~std::uint32_t{0};
  arrays.status.split_cells = ~std::uint32_t{0};
  arrays.status.result_count = ~std::uint32_t{0};
  arrays.status.active_cells = ~std::uint64_t{0};
  std::ranges::fill(arrays.status.active_cells_per_size_level, ~std::uint64_t{0});
  arrays.status.face_pieces = ~std::uint64_t{0};
}

// Replays <proc-build-grid-in-parallel> P1 to P6 on the CPU, with the per-item rules the GPU functions of
// grid/staggered_grid_gpu.cu call, in the same sequence as BuildSortedCellList's GPU path: every thread of every launch
// as `launch` says, the atomic minimum replaced by a plain minimum, BuildStableOrder on the CPU between the passes. The
// sorted list goes to arrays.cell_ids. Returns false, after a test failure, when BuildStableOrder fails.
inline bool ApplyListCheckStepsByReplay(
    GridFrame frame,
    std::span<const CellId> input_cells,
    SiblingCheck sibling_check,
    ReplayLaunch launch,
    ReplayArrays& arrays
) {
  if (!arrays.order_buffers.has_value()) {
    ADD_FAILURE() << "the replay has no order buffers";
    return false;
  }
  OrderBuffers& order_buffers = *arrays.order_buffers;
  const auto count = static_cast<std::uint32_t>(input_cells.size());
  const CellId* const input = input_cells.data();
  CellId* const sorted_cells = arrays.cell_ids.data();
  GridBuildStatus& status = arrays.status;
  const auto offer_key = [&status](std::uint64_t failure_key) {
    status.failure_key = std::min(status.failure_key, failure_key);
  };
  // P1: the reset, one thread.
  status = ComputeResetBuildStatus();
  // P2: the ids, one thread per input index.
  ApplyEveryThreadOfLaunch(count, launch, [&](std::size_t index) {
    offer_key(ComputeCellIdFailureKey(frame, input, index));
    return true;
  });
  // P3: three passes, alternating the two orders as the GPU path does, then the copy in sorted order.
  const std::array<const Buffer<std::uint32_t>*, kSortKeyPassCount> previous_orders = {
      nullptr,
      &arrays.order,
      &arrays.second_order,
  };
  const std::array<Buffer<std::uint32_t>*, kSortKeyPassCount> pass_orders = {
      &arrays.order,
      &arrays.second_order,
      &arrays.order,
  };
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    ApplyEveryThreadOfLaunch(count, launch, [&](std::size_t index) {
      arrays.sort_keys.data()[index] = ToCellSortKey(input[index], pass);
      return true;
    });
    const auto pass_index = static_cast<std::size_t>(pass);
    const std::expected<void, BufferError> ordered = BuildStableOrder(
        arrays.sort_keys,
        count,
        previous_orders.at(pass_index),
        *pass_orders.at(pass_index),
        order_buffers
    );
    if (!ordered.has_value()) {
      ADD_FAILURE() << "BuildStableOrder failed";
      return false;
    }
  }
  ApplyEveryThreadOfLaunch(count, launch, [&](std::size_t position) {
    sorted_cells[position] = input[arrays.order.data()[position]];
    return true;
  });
  // P4, P5 when siblings are required, P6: a thread returns once an earlier step's key is there.
  for (const int step : {kOverlapStep, kSiblingStep, kSizeStepRuleStep}) {
    if (step == kSiblingStep && sibling_check == SiblingCheck::kSkip) {
      continue;
    }
    ApplyEveryThreadOfLaunch(count, launch, [&](std::size_t position) {
      if (ComputeHasEarlierStepFailed(status.failure_key, step)) {
        return false;
      }
      offer_key(ComputeSortedCellFailureKey(sorted_cells, count, static_cast<std::uint32_t>(position), step));
      return true;
    });
  }
  return true;
}

// How a build replay differs from the GPU's sequence: a copy with one part changed, so a test shows the part matters.
enum class ReplayChange : std::uint8_t {
  kNone,  // The GPU's sequence.
  kWithoutTailFill,  // P7 (d), the tail fill, left out.
  kSplitCountsReturnEarly,  // A thread of P7 (a) returns after an earlier step's failure instead of writing 0.
  // P7 (b) checks the cells against one place less than they need in place of the sort count, the state the n / 7
  // bound rules out.
  kSortCountBelowCells,
};

// Runs ComputeExclusivePrefixSum in place over the first `count` words of `sums`, as the GPU path does between the
// launches of P7 and P8. An error is counted in arrays.failed_prefix_sum_count and returns false: the CPU half maps a
// size error of base/ to kInternalFault.
inline bool ApplyPrefixSumOfReplay(std::uint32_t count, ReplayArrays& arrays) {
  if (!arrays.order_buffers.has_value()) {
    return false;
  }
  const std::expected<void, BufferError> summed =
      ComputeExclusivePrefixSum(arrays.second_order, count, arrays.second_order, *arrays.order_buffers);
  if (!summed.has_value()) {
    ++arrays.failed_prefix_sum_count;
    return false;
  }
  return true;
}

// The key of P7 (b) in the replay: ComputeCellPlacesFailureKey against the room's cell places and the sort count, or,
// for ReplayChange::kSortCountBelowCells, against one place less than the cells need in place of the sort count.
//
// clang-tidy: three counts; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline std::uint64_t ComputeCellPlacesKeyOfReplay(
    std::uint32_t active_count,
    std::uint32_t split_cells,
    std::uint32_t max_cell_blocks,
    ReplayChange change
) {
  const std::uint64_t room_cell_places = std::uint64_t{kCellPlacesPerBlock} * max_cell_blocks;
  const std::uint64_t cell_count = std::uint64_t{active_count} + split_cells;
  const bool is_bound_lowered = change == ReplayChange::kSortCountBelowCells && cell_count > 0;
  const std::uint32_t sort_count = ComputeSortCount(active_count, max_cell_blocks);
  const auto checked_sort_count = static_cast<std::uint32_t>(is_bound_lowered ? cell_count - 1 : sort_count);
  return ComputeCellPlacesFailureKey(cell_count, room_cell_places, checked_sort_count);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Replays P7 and P8 on the sorted list of `active_count` cells in arrays.cell_ids, for a room of max_cell_blocks, with
// the per-item rules the GPU functions call, every thread of every launch as `launch` says, each prefix sum and stable
// order on the CPU between the launches. Returns false when a prefix sum or a stable order fails.
//
// clang-tidy: a count and a room; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline bool ApplyBlockStepsByReplay(
    std::uint32_t active_count,
    std::uint32_t max_cell_blocks,
    ReplayLaunch launch,
    ReplayChange change,
    ReplayArrays& arrays
) {
  if (!arrays.order_buffers.has_value()) {
    ADD_FAILURE() << "the replay has no order buffers";
    return false;
  }
  OrderBuffers& order_buffers = *arrays.order_buffers;
  CellId* const cell_ids = arrays.cell_ids.data();
  std::uint32_t* const sums = arrays.second_order.data();
  const std::uint32_t* const order = arrays.order.data();
  GridBuildStatus& status = arrays.status;
  const auto offer_key = [&status](std::uint64_t failure_key) {
    status.failure_key = std::min(status.failure_key, failure_key);
  };
  const auto has_earlier_failure = [&status](int step) {
    return ComputeHasEarlierStepFailed(status.failure_key, step);
  };
  // Step P7 (a): the count of each active cell, or 0 after an earlier failure, then the prefix sum in place.
  ApplyEveryThreadOfLaunch(active_count, launch, [&](std::size_t position) {
    if (has_earlier_failure(kCellPlacesStep)) {
      if (change == ReplayChange::kSplitCountsReturnEarly) {
        return false;
      }
      sums[position] = 0;
      return true;
    }
    sums[position] = static_cast<std::uint32_t>(ComputeSplitCellCountToAdd(cell_ids[position]));
    return true;
  });
  if (!ApplyPrefixSumOfReplay(active_count, arrays)) {
    return false;
  }
  // Step P7 (b), one thread: the cells against the room's cell places, then against the sort count.
  const std::uint32_t sort_count = ComputeSortCount(active_count, max_cell_blocks);
  status.split_cells = sums[active_count];
  offer_key(ComputeCellPlacesKeyOfReplay(active_count, status.split_cells, max_cell_blocks, change));
  // Step P7 (c).
  ApplyEveryThreadOfLaunch(active_count, launch, [&](std::size_t position) {
    if (has_earlier_failure(kBlockCountStep)) {
      return false;
    }
    ApplySplitCellWritesAt(cell_ids, sums, active_count, static_cast<std::uint32_t>(position));
    return true;
  });
  // Step P7 (d): the tail fill, over the CPU's sort count, with the used count from the status record.
  arrays.cells_before_tail_fill.assign(cell_ids, cell_ids + sort_count);
  if (change != ReplayChange::kWithoutTailFill) {
    ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t place) {
      if (has_earlier_failure(kBlockCountStep)) {
        return false;
      }
      ApplySortTailFillAt(cell_ids, active_count + status.split_cells, static_cast<std::uint32_t>(place));
      return true;
    });
  }
  // Step P8: three passes of the block keys, alternating the orders as P3 does, the final order in arrays.order.
  const std::array<const Buffer<std::uint32_t>*, kSortKeyPassCount> previous_orders = {
      nullptr,
      &arrays.order,
      &arrays.second_order,
  };
  const std::array<Buffer<std::uint32_t>*, kSortKeyPassCount> pass_orders = {
      &arrays.order,
      &arrays.second_order,
      &arrays.order,
  };
  for (int pass = 0; pass < kSortKeyPassCount; ++pass) {
    ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t place) {
      if (has_earlier_failure(kBlockCountStep)) {
        return false;
      }
      arrays.sort_keys.data()[place] = ToBlockSortKey(cell_ids[place], pass);
      return true;
    });
    const auto pass_index = static_cast<std::size_t>(pass);
    const std::expected<void, BufferError> ordered = BuildStableOrder(
        arrays.sort_keys,
        sort_count,
        previous_orders.at(pass_index),
        *pass_orders.at(pass_index),
        order_buffers
    );
    if (!ordered.has_value()) {
      ADD_FAILURE() << "BuildStableOrder failed";
      return false;
    }
  }
  // Step P8: the marks, or 0 after an earlier failure, step 5 included, then the prefix sum in place.
  ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t position) {
    sums[position] = has_earlier_failure(kBlockCountStep)
        ? 0U
        : ComputeBlockStartMark(cell_ids, order, static_cast<std::uint32_t>(position));
    return true;
  });
  if (!ApplyPrefixSumOfReplay(sort_count, arrays)) {
    return false;
  }
  // Step P8: the block count against the room, one thread, then the scatter of the block ids.
  status.cell_blocks = sums[sort_count];
  offer_key(ComputeRoomFailureKey(status.cell_blocks, max_cell_blocks, kBlockCountStep));
  ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t position) {
    if (has_earlier_failure(kBlockFaultStep)) {
      return false;
    }
    ApplyBlockIdWriteAt(arrays.block_ids.data(), cell_ids, order, sums, static_cast<std::uint32_t>(position));
    return true;
  });
  return true;
}

// The tables of the replay arrays for the block count of their status record, at most the room, as a GPU function of
// P10 (c) or P11 sees them.
inline GridTables ToReplayTables(GridFrame frame, GridVersion version, const ReplayArrays& arrays) {
  const std::uint32_t* const bigger_block = arrays.bigger_and_smaller_blocks.data();
  return {
      .frame = frame,
      .version = version,
      .block_count = std::min(arrays.status.cell_blocks, arrays.max_cell_blocks),
      .block_ids = arrays.block_ids.data(),
      .same_size_level_blocks = arrays.same_size_level_blocks.data(),
      .bigger_block = bigger_block,
      .smaller_blocks = bigger_block + ComputeSmallerBlocksStart(arrays.max_cell_blocks),
      .cell_states = arrays.cell_states.data(),
      .face_piece_kinds = arrays.face_piece_kinds.data(),
  };
}

// Says whether an entry is a block of the tables or kNoIndex.
inline bool ComputeIsBlockEntryInRange(const GridTables& tables, std::uint32_t entry) {
  return entry == kNoIndex || entry < tables.block_count;
}

// Says whether every block entry that a place of `block` may read through P10 (c) or P11 names a block of the tables or
// kNoIndex: the block's bigger block and 27 neighbours, and, when the bigger block is a block, its 27 neighbours.
inline bool ComputeAreTableEntriesInRange(const GridTables& tables, std::uint32_t block) {
  const auto are_neighbours_in_range = [&tables](std::uint32_t of_block) {
    const std::span<const std::uint32_t> neighbours(
        tables.same_size_level_blocks + (std::size_t{kSameSizeLevelEntriesPerBlock} * of_block),
        kSameSizeLevelEntriesPerBlock
    );
    return std::ranges::all_of(neighbours, [&tables](std::uint32_t entry) {
      return ComputeIsBlockEntryInRange(tables, entry);
    });
  };
  const std::uint32_t bigger_block = tables.bigger_block[block];
  if (!ComputeIsBlockEntryInRange(tables, bigger_block) || !are_neighbours_in_range(block)) {
    return false;
  }
  return bigger_block == kNoIndex || are_neighbours_in_range(bigger_block);
}

// Replays P9 to P11 into the tables of `arrays`, for a room of arrays.max_cell_blocks blocks, after P8 left the block
// ids, the final order in arrays.order and the sums of the marks in arrays.second_order: every thread of every launch
// as `launch` says, each launch sized for the room (or, for P10 (b), the sort count) as the GPU half sizes it, the
// threads acting on the items below the counts of the status record. The atomic minimum and adds become plain ones.
//
// clang-tidy: two counts; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
inline void ApplyTableStepsByReplay(
    GridFrame frame,
    std::uint32_t active_count,
    std::uint32_t sort_count,
    ReplayLaunch launch,
    ReplayArrays& arrays
) {
  GridBuildStatus& status = arrays.status;
  const std::uint32_t room = arrays.max_cell_blocks;
  const GridTables tables = ToReplayTables(frame, GridVersion{1}, arrays);
  const std::uint32_t block_count = tables.block_count;
  const auto has_earlier_failure = [&status](int step) {
    return ComputeHasEarlierStepFailed(status.failure_key, step);
  };
  // P9: one thread per block of the room; a block of size level 1 or more without its bigger block offers step 7.
  std::uint32_t* const same_size_level_blocks = arrays.same_size_level_blocks.data();
  std::uint32_t* const bigger_block = arrays.bigger_and_smaller_blocks.data();
  std::uint32_t* const smaller_blocks = bigger_block + ComputeSmallerBlocksStart(room);
  ApplyEveryThreadOfSizedLaunch(room, block_count, launch, [&](std::size_t block) {
    if (has_earlier_failure(kBlockFaultStep)) {
      return false;
    }
    const std::uint64_t failure_key = ApplyBlockNeighboursWriteAt(
        arrays.block_ids.data(),
        block_count,
        CellBlockIndex{static_cast<std::uint32_t>(block)},
        same_size_level_blocks,
        bigger_block,
        smaller_blocks
    );
    status.failure_key = std::min(status.failure_key, failure_key);
    return true;
  });
  // P10 (a), (b) and (c), then P11: any key stops them.
  const std::size_t room_cell_places = std::size_t{kCellPlacesPerBlock} * room;
  CellState* const cell_states = arrays.cell_states.data();
  const std::size_t used_cell_places = std::size_t{kCellPlacesPerBlock} * block_count;
  ApplyEveryThreadOfSizedLaunch(room_cell_places, used_cell_places, launch, [&](std::size_t place) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    ApplyAbsentStateAt(cell_states, CellPlaceIndex{static_cast<std::uint32_t>(place)});
    return true;
  });
  ApplyEveryThreadOfLaunch(sort_count, launch, [&](std::size_t position) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    const int size_level = ApplySortedCellStateAt(
        cell_states,
        arrays.cell_ids.data(),
        arrays.order.data(),
        arrays.second_order.data(),
        active_count,
        static_cast<std::uint32_t>(position)
    );
    if (size_level >= 0) {
      ++status.active_cells_per_size_level[static_cast<std::size_t>(size_level)];
    }
    return true;
  });
  // Which blocks read only blocks of the tables or kNoIndex, found once after P9: nothing after it changes a block
  // entry.
  std::vector<bool> is_block_in_range(block_count);
  for (std::uint32_t block = 0; block < block_count; ++block) {
    is_block_in_range[block] = ComputeAreTableEntriesInRange(tables, block);
  }
  // P10 (c): the answers into the sort keys, free after P8, then the copy into the states.
  std::uint64_t* const cover_answers = arrays.sort_keys.data();
  ApplyEveryThreadOfSizedLaunch(room_cell_places, used_cell_places, launch, [&](std::size_t place) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    if (!is_block_in_range[place / kCellPlacesPerBlock]) {
      ++arrays.stale_table_read_count;
      return true;
    }
    ApplyCoverAnswerAt(tables, cover_answers, CellPlaceIndex{static_cast<std::uint32_t>(place)});
    return true;
  });
  ApplyEveryThreadOfSizedLaunch(room_cell_places, used_cell_places, launch, [&](std::size_t place) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    ApplyCoverAnswerCopyAt(cover_answers, cell_states, CellPlaceIndex{static_cast<std::uint32_t>(place)});
    return true;
  });
  const std::size_t room_face_places = std::size_t{kFacePlacesPerBlock} * room;
  const std::size_t used_face_places = std::size_t{kFacePlacesPerBlock} * block_count;
  ApplyEveryThreadOfSizedLaunch(room_face_places, used_face_places, launch, [&](std::size_t face) {
    if (has_earlier_failure(kTableEntryStep)) {
      return false;
    }
    if (!is_block_in_range[face / kFacePlacesPerBlock]) {
      ++arrays.stale_table_read_count;
      return true;
    }
    status.face_pieces += ApplyFacePieceKindWriteAt(
        tables,
        arrays.face_piece_kinds.data(),
        FacePlaceIndex{static_cast<std::uint32_t>(face)}
    );
    return true;
  });
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Replays a build on the GPU, <proc-build-grid-in-parallel> P1 to P12: the input checks of P1 with CheckCellListInputs
// on `buffers`, which gives the room, then the steps of ApplyListCheckStepsByReplay with the siblings required,
// ApplyBlockStepsByReplay, ApplyTableStepsByReplay and ApplyBuildStatusResult. A failed prefix sum or stable order
// gives kInternalFault, as the CPU half's mapping of a size error does. Returns the failure the status names.
inline GridFailure ApplyGpuBuildStepsByReplay(
    const GridInputs& inputs,
    const GridBuffers& buffers,
    ReplayLaunch launch,
    ReplayChange change,
    ReplayArrays& arrays
) {
  arrays.failed_prefix_sum_count = 0;
  arrays.stale_table_read_count = 0;
  arrays.cells_before_tail_fill.clear();
  if (const std::expected<void, GridFailure> inputs_checked = CheckCellListInputs(inputs, buffers);
      !inputs_checked.has_value()) {
    return inputs_checked.error();
  }
  const std::uint32_t max_cell_blocks = buffers.max_cell_blocks();
  // The guards lie just after the room's part of each array only when the arrays are sized for this room.
  if (arrays.max_cell_blocks != max_cell_blocks) {
    ADD_FAILURE() << "the replay arrays are sized for another room";
    return {.error = GridError::kInternalFault};
  }
  const std::span<const CellId> input_cells = inputs.active_cells;
  if (!ApplyListCheckStepsByReplay(inputs.frame, input_cells, SiblingCheck::kRequire, launch, arrays)) {
    return {.error = GridError::kInternalFault};
  }
  const auto count = static_cast<std::uint32_t>(input_cells.size());
  if (!ApplyBlockStepsByReplay(count, max_cell_blocks, launch, change, arrays)) {
    return {.error = GridError::kInternalFault};
  }
  ApplyTableStepsByReplay(inputs.frame, count, ComputeSortCount(count, max_cell_blocks), launch, arrays);
  ApplyBuildStatusResult(inputs.frame, input_cells.data(), count, arrays.cell_ids.data(), count, &arrays.status);
  return ToGridFailure(arrays.status);
}

// Every field of a failure, as one comparable value.
inline std::tuple<GridError, std::uint64_t, std::uint64_t, std::uint64_t> ToFailureFields(const GridFailure& failure) {
  return {failure.error, failure.cell.value, failure.other_cell.value, failure.magnitude};
}

// A CPU grid of no cells in a room of max_cell_blocks blocks, whose buffers TakeEmptyGridBuffers hands to a test.
inline std::expected<TestGrid, GridFailure> BuildReplayBuffersOwner(std::uint32_t max_cell_blocks) {
  const GridInputs no_cells = {.frame = ToUnitFrame(4), .active_cells = {}, .version = {.value = 1}};
  return BuildTestGrid(no_cells, {.max_cell_blocks = max_cell_blocks});
}

inline std::expected<TestGrid, GridFailure> BuildReplayBuffersOwner() {
  return BuildReplayBuffersOwner(kReplayRoom);
}

// The `count` values from `first`, copied into a vector.
template <typename Value>
std::vector<Value> ToVector(const Value* first, std::size_t count) {
  return std::vector<Value>(first, first + count);
}

// The used part of the five tables of `tables`, block_count blocks of each, copied into vectors.
inline TableVectors ToTableVectors(const GridTables& tables) {
  const std::size_t block_count = tables.block_count;
  return {
      .frame = tables.frame,
      .block_ids = ToVector(tables.block_ids, block_count),
      .same_size_level_blocks = ToVector(tables.same_size_level_blocks, kSameSizeLevelEntriesPerBlock * block_count),
      .bigger_block = ToVector(tables.bigger_block, block_count),
      .smaller_blocks = ToVector(tables.smaller_blocks, kSmallerBlocksPerBlock * block_count),
      .cell_states = ToVector(tables.cell_states, kCellPlacesPerBlock * block_count),
      .face_piece_kinds = ToVector(tables.face_piece_kinds, kFacePlacesPerBlock * block_count),
  };
}

// What one build replay gave: the failure, on success the counts and the used tables, and the two counts the status
// record holds after P8.
struct BuildReplayResult {
  GridFailure failure;
  std::optional<TableVectors> tables;  // On success only.
  GridCounts counts{};  // On success only: the counts of the status record.
  std::uint32_t cell_blocks = 0;
  std::uint32_t split_cells = 0;
  int failed_prefix_sum_count = 0;
};

// The counts of a build's status record, as the CPU half of a build on the GPU reads them on success.
inline GridCounts ToCountsOfStatus(const GridBuildStatus& status) {
  GridCounts counts{};
  counts.cell_blocks = status.cell_blocks;
  std::ranges::copy(status.active_cells_per_size_level, &counts.active_cells_per_size_level[0]);
  counts.active_cells = status.active_cells;
  counts.face_pieces = status.face_pieces;
  return counts;
}

// Replays the build of `inputs` on the room of `buffers` into `arrays` (ApplyGpuBuildStepsByReplay), collects what
// it gave and expects every guard of the arrays unchanged: no step wrote past an array of the room.
inline BuildReplayResult ComputeBuildReplay(
    const GridInputs& inputs,
    const GridBuffers& buffers,
    ReplayLaunch launch,
    ReplayChange change,
    ReplayArrays& arrays
) {
  BuildReplayResult result;
  result.failure = ApplyGpuBuildStepsByReplay(inputs, buffers, launch, change, arrays);
  result.cell_blocks = arrays.status.cell_blocks;
  result.split_cells = arrays.status.split_cells;
  result.failed_prefix_sum_count = arrays.failed_prefix_sum_count;
  EXPECT_TRUE(ComputeAreGuardsUnchanged(arrays)) << "a write left the room's part of a replay array";
  EXPECT_EQ(arrays.stale_table_read_count, 0) << "P10 (c) or P11 read a block past the block count";
  if (result.failure.error == GridError::kNone && result.cell_blocks <= arrays.max_cell_blocks) {
    result.tables = ToTableVectors(ToReplayTables(inputs.frame, inputs.version, arrays));
    result.counts = ToCountsOfStatus(arrays.status);
  }
  return result;
}

// The split cells and the blocks of a list, by the test's own floor, for a list that passed steps 1 to 4.
struct ExpectedBlocks {
  std::uint32_t split_cells = 0;
  std::uint32_t cell_blocks = 0;
};

inline ExpectedBlocks ComputeExpectedBlocks(std::span<const CellId> cells) {
  const std::vector<CellId> active_cells(cells.begin(), cells.end());
  const std::set<std::uint64_t> split_cells = BuildSplitCells(active_cells);
  std::set<std::uint64_t> blocks;
  for (const CellId cell : active_cells) {
    blocks.insert(ToBlockOfCell(cell).value);
  }
  for (const std::uint64_t split_cell : split_cells) {
    blocks.insert(ToBlockOfCell(CellId{.value = split_cell}).value);
  }
  return {
      .split_cells = static_cast<std::uint32_t>(split_cells.size()),
      .cell_blocks = static_cast<std::uint32_t>(blocks.size()),
  };
}

// What the CPU build of a list gave, and what the status counts of its replay must be.
struct CpuBuildOfList {
  GridFailure failure;
  std::optional<TableVectors> tables;  // On success only: the used part of the grid's tables.
  GridCounts counts{};  // On success only.
  bool is_list_too_long = false;  // The input checks of P1 failed it: the replay never reached the GPU steps.
  ExpectedBlocks status_counts;  // The split cells and blocks the status record must hold otherwise.
};

// Runs BuildGrid on the CPU buffers, keeps the version check, the counts and the used tables of the grid, and takes
// the buffers back. The status counts: after a failure of steps 1 to 4, zero; otherwise the split cells of the list,
// and the blocks of the list unless step 5 failed, when the marks are 0.
inline CpuBuildOfList ComputeCpuBuildOfList(const GridInputs& inputs, GridBuffers& buffers) {
  CpuBuildOfList cpu_build;
  std::expected<StaggeredGrid, GridFailure> cpu_grid = BuildGrid(inputs, buffers);
  if (cpu_grid.has_value()) {
    EXPECT_EQ(cpu_grid->version().value, inputs.version.value);
    cpu_build.tables = ToTableVectors(cpu_grid->tables(RunOn::kCpu));
    cpu_build.counts = cpu_grid->counts();
    buffers = TakeBuffers(std::move(*cpu_grid));
  } else {
    cpu_build.failure = cpu_grid.error();
  }
  const std::uint64_t room_cell_places = std::uint64_t{kCellPlacesPerBlock} * buffers.max_cell_blocks();
  cpu_build.is_list_too_long = inputs.active_cells.size() > room_cell_places;
  const bool has_passed_sorted_checks =
      cpu_build.failure.error == GridError::kNone || cpu_build.failure.error == GridError::kTooManyCellBlocks;
  if (!has_passed_sorted_checks || cpu_build.is_list_too_long) {
    return cpu_build;
  }
  cpu_build.status_counts = ComputeExpectedBlocks(inputs.active_cells);
  if (inputs.active_cells.size() + cpu_build.status_counts.split_cells > room_cell_places) {
    cpu_build.status_counts.cell_blocks = 0;
  }
  return cpu_build;
}

// Says whether two copies of used tables hold the same bytes: the block count and every entry of the five tables.
inline bool
ComputeIsSameTableVectors(const std::optional<TableVectors>& first, const std::optional<TableVectors>& second) {
  if (!first.has_value() || !second.has_value()) {
    return first.has_value() == second.has_value();
  }
  return ComputeIsSameUsedTables(first->ToGridTables(), second->ToGridTables());
}

// Expects one build replay to give the CPU's failure, no failed prefix sum, on success the CPU's counts and the same
// bytes in the used part of the five tables, and, when the GPU steps ran, the status counts after P8.
inline void CheckBuildReplayEqualsCpuBuild(
    const BuildReplayResult& replay,
    const CpuBuildOfList& cpu_build,
    const std::string& context
) {
  SCOPED_TRACE(context);
  EXPECT_EQ(ToFailureFields(replay.failure), ToFailureFields(cpu_build.failure));
  EXPECT_EQ(replay.failed_prefix_sum_count, 0);
  EXPECT_TRUE(ComputeIsSameTableVectors(replay.tables, cpu_build.tables));
  EXPECT_EQ(ToGridCountFields(replay.counts), ToGridCountFields(cpu_build.counts));
  // The input checks of P1 stop a list too long before any GPU step, so the record keeps no count of it.
  const ExpectedBlocks replay_status_counts = {.split_cells = replay.split_cells, .cell_blocks = replay.cell_blocks};
  const ExpectedBlocks expected_status_counts =
      cpu_build.is_list_too_long ? replay_status_counts : cpu_build.status_counts;
  EXPECT_EQ(replay_status_counts.split_cells, expected_status_counts.split_cells);
  EXPECT_EQ(replay_status_counts.cell_blocks, expected_status_counts.cell_blocks);
}

inline constexpr std::uint64_t kTwoToThe34 = std::uint64_t{1} << 34U;

inline constexpr std::uint64_t kTwoToThe38 = std::uint64_t{1} << 38U;

}  // namespace fluidity::grid
